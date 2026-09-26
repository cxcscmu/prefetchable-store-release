"""The expert store and the learned-routing controls.

`HashMoE` is the prefetchable store: `rows` holds expert ids computed OUTSIDE the module from token ids alone
(`balanced_assignment`), so every address is known before the forward pass and the expert blocks can be streamed
from storage while the early layers run. `FineLearnedMoE` is the learned-routing control on the same expert bank:
either the conventional router (reads the store layer's own input, not prefetchable) or the previous-step router
(reads [embedding of token t ; store input at t-1], available before layer 0 of step t, prefetchable).

Both are `memory` modules for model.Layer: forward(rows, h) -> (delta, aux). Expert output projections are
zero-initialised, so every store starts as an exact no-op on top of its backbone.
"""

from __future__ import annotations

import hashlib
import heapq

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gen(seed: int, name: str) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "big"))
    return g


def _splitmix64(x: np.ndarray) -> np.ndarray:
    z = x.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def balanced_assignment(counts: np.ndarray, n_experts: int, salt: int, jitter: float = 0.0) -> np.ndarray:
    """token id -> expert id, balancing TOKEN MASS (Roller et al.'s 'balanced hash').

    Greedy: walk ids by descending corpus count (ties broken by a salted hash), always assigning to the currently
    lightest expert. Deterministic given (counts, n_experts, salt) and independent of the initialisation seed.
    `jitter > 0` (train.py --indep-hash) visits ids in a salted, noisily count-sorted order so different salts give
    different groupings; loads still use the true counts, so token mass stays balanced.
    """
    V = len(counts)
    tie = _splitmix64(np.arange(V, dtype=np.uint64) ^ np.uint64(salt))
    order = np.lexsort((tie, -counts.astype(np.int64)))
    if jitter > 0:
        rng = np.random.default_rng(int(salt) + 0x5EED)
        order = np.argsort(-(counts.astype(np.float64) + 0.5) * np.exp(jitter * rng.standard_normal(V)), kind="stable")
    heap = [(0, e) for e in range(n_experts)]
    heapq.heapify(heap)
    out = np.empty(V, dtype=np.int64)
    for tid in order:
        load, e = heapq.heappop(heap)
        out[tid] = e
        heapq.heappush(heap, (load + int(counts[tid]), e))
    return out


class _ExpertBank(nn.Module):
    """N SwiGLU expert FFNs held as batched tensors (3 * d_model * d_ff_e parameters each), computed by grouped bmm.

    Token mass is Zipfian, so per-expert token counts are skewed. Experts are grouped into power-of-two capacity
    tiers, which bounds padding waste below 2x. `down` is zero-initialised, so the bank is a no-op at step 0.
    """

    def __init__(self, n_experts: int, d_model: int, d_ff_e: int):
        super().__init__()
        self.n, self.d, self.f = n_experts, d_model, d_ff_e
        self.gate = nn.Parameter(torch.empty(n_experts, d_model, d_ff_e))
        self.up = nn.Parameter(torch.empty(n_experts, d_model, d_ff_e))
        self.down = nn.Parameter(torch.empty(n_experts, d_ff_e, d_model))

    def reset_parameters(self, seed: int, name: str):
        for pname, p in (("gate", self.gate), ("up", self.up)):
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=_gen(seed, f"{name}:{pname}")) * self.d**-0.5)
        nn.init.zeros_(self.down)

    def forward_grouped(
        self, h_flat: torch.Tensor, eid: torch.Tensor, capacity: int | None = None
    ) -> torch.Tensor:
        """h_flat [M, d], eid [M] -> delta [M, d]. capacity: Switch train-time cap
        (overflow tokens contribute zero delta); None = process everything."""
        M, d = h_flat.shape
        order = torch.argsort(eid, stable=True)
        counts = torch.bincount(eid, minlength=self.n)
        starts = torch.cumsum(counts, 0) - counts
        ranks = torch.arange(M, device=eid.device) - starts[eid[order]]
        if capacity is not None:
            kept = ranks < int(capacity)
            order, ranks = order[kept], ranks[kept]
            counts = torch.clamp(counts, max=int(capacity))
        pos_e = eid[order]  # expert of each kept token
        delta = torch.zeros_like(h_flat)
        cmax = int(counts.max().item())
        if cmax == 0:
            return delta
        # power-of-two capacity tiers over experts
        tier_of_expert = torch.clamp(torch.ceil(torch.log2(counts.clamp_min(1).float())), min=0).long()
        tier_of_expert[counts == 0] = -1
        for t in torch.unique(tier_of_expert):
            if t < 0:
                continue
            cap_t = int(2 ** int(t))
            experts_t = (tier_of_expert == t).nonzero(as_tuple=True)[0]
            k = len(experts_t)
            # map global expert id -> row inside this tier
            row_of = torch.full((self.n,), -1, dtype=torch.long, device=eid.device)
            row_of[experts_t] = torch.arange(k, device=eid.device)
            sel = row_of[pos_e] >= 0
            p_pos, p_rank, p_row = order[sel], ranks[sel], row_of[pos_e[sel]]
            x = torch.zeros(k, cap_t, d, dtype=h_flat.dtype, device=h_flat.device)
            x[p_row, p_rank] = h_flat[p_pos]
            y = torch.bmm(
                F.silu(torch.bmm(x, self.gate[experts_t])) * torch.bmm(x, self.up[experts_t]),
                self.down[experts_t],
            )
            # under autocast the bmm chain emits bf16 while delta is fp32 (h enters from
            # the fp32 embedding); cast at the write or CUDA rejects the index_put.
            delta[p_pos] = y[p_row, p_rank].to(delta.dtype)
        return delta


class HashMoE(nn.Module):
    """Token-id-addressed experts. rows = [B, T] expert ids, or [B, T, K] for K experts per token.

    With K > 1 the K expert outputs are summed (sum_k=True) or averaged (sum_k=False). slot_gate=K replaces the
    uniform combination by softmax weights of the store-layer state (zero-initialised, so it starts equal to the
    plain sum / average); the addresses still come from token ids alone.
    """

    def __init__(self, n_experts: int, d_model: int, d_ff_e: int, sum_k: bool = False, n_shared: int = 0,
                 slot_gate: int = 0):
        super().__init__()
        self.bank = _ExpertBank(n_experts, d_model, d_ff_e)
        self.shared = _ExpertBank(1, d_model, d_ff_e * n_shared) if n_shared else None
        self.last_aux_loss = None  # hash routing needs no aux loss
        self.sum_k = sum_k
        self.slot_gate = nn.Linear(d_model, slot_gate, bias=False) if slot_gate else None

    def reset_parameters(self, seed: int = 0, name: str = "hashmoe"):
        self.bank.reset_parameters(seed, name)
        if self.shared is not None:
            self.shared.reset_parameters(seed, name + ":shared")
        with torch.no_grad():
            if self.slot_gate is not None:
                self.slot_gate.weight.zero_()

    def forward(self, rows: torch.Tensor, h: torch.Tensor):
        B, T, d = h.shape
        if rows.dim() == 3:  # K experts per token: rows [B, T, K]
            hf = h.reshape(B * T, d)
            K = rows.shape[-1]
            if self.slot_gate is not None:
                assert self.slot_gate.out_features == K, "slot gate width must equal the number of hash slots"
                with torch.autocast(hf.device.type, enabled=False):
                    alpha = F.softmax(self.slot_gate(hf.float()), dim=-1)  # [M, K], uniform 1/K at init
                if self.sum_k:
                    alpha = alpha * K  # so the gated sum equals the plain sum at init
            delta = torch.zeros_like(hf)
            for k in range(K):
                out_k = self.bank.forward_grouped(hf, rows[..., k].reshape(B * T))
                if self.slot_gate is not None:
                    out_k = out_k * alpha[:, k : k + 1].to(out_k.dtype)
                delta = delta + out_k
            if not self.sum_k and self.slot_gate is None:
                delta = delta / K
            if self.shared is not None:
                delta = delta + self.shared.forward_grouped(hf, torch.zeros(B * T, dtype=torch.long, device=hf.device))
            return delta.reshape(B, T, d), {"score": None, "selected": None}
        hf = h.reshape(B * T, d)
        delta = self.bank.forward_grouped(hf, rows.reshape(B * T))
        if self.shared is not None:
            delta = delta + self.shared.forward_grouped(hf, torch.zeros(B * T, dtype=torch.long, device=hf.device))
        return delta.reshape(B, T, d), {"score": None, "selected": None}


class FineLearnedMoE(nn.Module):
    """Learned top-K routing on the same expert bank: softmax scores renormalised over the chosen K, optional
    `n_shared` always-on experts, Switch load-balancing loss generalised to K slots (stored on last_aux_loss, scaled
    by train.py), train-time capacity factor CF per expert with overflow dropped (CF = 0: dropless).

    prev=False: the conventional router reads the store layer's own input, so the address is known only at that
    layer (not prefetchable).
    prev=True: the previous-step router reads [RMSNorm(embedding of token t) ; store-layer input at t-1]. At decode
    time both exist before step t's forward pass begins, so the address has the same lead time as token-id
    addressing. The embedding arrives through `ctx["emb"]`, captured by a forward hook on model.embed.
    """

    CF = 1.25

    def __init__(self, n_experts: int, d_model: int, d_ff_e: int, topk: int, n_shared: int = 0, ctx: dict | None = None, prev: bool = False):
        super().__init__()
        self.bank = _ExpertBank(n_experts, d_model, d_ff_e)
        self.shared = _ExpertBank(1, d_model, d_ff_e * n_shared) if n_shared else None
        self.prev = prev
        self.router = nn.Linear(2 * d_model if prev else d_model, n_experts, bias=False)
        self.emb_norm = nn.RMSNorm(d_model) if prev else None  # the raw embedding is ~0.02-scale; the state is ~1
        self.topk, self._ctx = topk, ctx
        self.decode_prev = None  # incremental decode only: this layer's store input at t-1, set by DecodeSession
        self.last_aux_loss = self.last_util = self.last_drop = None

    def reset_parameters(self, seed: int = 0, name: str = "finemoe"):
        self.bank.reset_parameters(seed, name)
        if self.shared is not None:
            self.shared.reset_parameters(seed, name + ":shared")
        with torch.no_grad():
            self.router.weight.copy_(
                torch.randn(self.router.weight.shape, generator=_gen(seed, "router")) * self.router.in_features**-0.5
            )

    def forward(self, rows: torch.Tensor, h: torch.Tensor):
        B, T, d = h.shape
        hf = h.reshape(B * T, d)
        src = hf
        if self.prev:
            emb = self._ctx["emb"]
            assert emb.shape == h.shape, "embedding capture missing or stale"
            if self.decode_prev is not None:
                # incremental decode (T == 1): the caller carries this layer's store input from the previous step
                # (DecodeSession does), which is exactly what the roll below reads during a full-sequence forward
                assert T == 1 and self.decode_prev.shape == h.shape, "decode_prev must be this layer's [B,1,d] input at t-1"
                hp = self.decode_prev
            else:
                hp = torch.roll(h, 1, dims=1)  # position t reads the store input at t-1 (causal; zero at position 0)
                hp = torch.cat([torch.zeros_like(hp[:, :1]), hp[:, 1:]], dim=1)
            src = torch.cat([self.emb_norm(emb), hp], dim=-1).reshape(B * T, 2 * d)
        with torch.autocast(hf.device.type, enabled=False):
            probs = F.softmax(self.router(src.float()), dim=-1)
        topv, topi = probs.topk(self.topk, dim=-1)  # [M, K]
        if self.topk > 1:  # K = 1 would renormalise to the constant 1 and cut the router's gradient
            topv = topv / topv.sum(dim=-1, keepdim=True)  # with thousands of experts the raw scores are ~1/n and starve the experts
        n, M, K = self.bank.n, hf.shape[0], self.topk
        eid = topi.reshape(-1)
        frac = torch.bincount(eid, minlength=n).float() / eid.numel()
        self.last_aux_loss = n * (frac * probs.mean(dim=0)).sum()
        self.last_util = float((frac > 0).float().mean())
        cap = int(np.ceil(self.CF * M * K / n)) if (self.training and self.CF > 0) else None  # CF = 0: dropless
        # diagnostics only (no effect on the forward pass): share of (token, slot) pairs dropped by the capacity limit
        self.last_drop = float(1.0 - torch.bincount(eid, minlength=n).clamp(max=cap).sum() / eid.numel()) if cap is not None else 0.0
        # one grouped call over all (token, slot) pairs
        out = self.bank.forward_grouped(hf.repeat_interleave(K, dim=0), eid, capacity=cap)
        delta = (out.reshape(M, K, d) * topv.unsqueeze(-1).to(out.dtype)).sum(dim=1)
        if self.shared is not None:
            delta = delta + self.shared.forward_grouped(hf, torch.zeros(M, dtype=torch.long, device=hf.device))
        return delta.reshape(B, T, d), {"score": None, "selected": None}

