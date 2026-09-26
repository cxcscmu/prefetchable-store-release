"""Transformer backbone (Llama-style: RMSNorm, RoPE, grouped-query attention, SwiGLU, tied head).

A store is attached by passing `memory_layers={layer_index: module}`. The store's output is ADDED to
the residual stream after the layer's FFN; the backbone FFN is kept. Every backbone parameter is
initialised from a generator seeded by (seed, parameter name), so two models that differ only in
their store start from byte-identical backbones and consume byte-identical token streams
(`PairedLoader`). This makes every store-versus-dense comparison paired.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    d_model: int = 768
    n_layers: int = 12
    n_q_heads: int = 12
    n_kv_heads: int = 4
    head_dim: int = 64
    d_ff: int = 2048
    vocab: int = 32_000
    seq_len: int = 2048
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    init_seed: int = 0


def chunked_cross_entropy(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Tied-head cross-entropy computed in token chunks, so the full [tokens, vocab] logits tensor is never
    materialised. Mathematically identical to F.cross_entropy(hidden @ weight.T, targets)."""
    flat_h = hidden.reshape(-1, hidden.shape[-1])
    flat_t = targets.reshape(-1)
    total = flat_h.new_zeros(())
    n = flat_t.numel()
    for i in range(0, n, chunk):
        logits = F.linear(flat_h[i : i + chunk], weight)
        total = total + F.cross_entropy(logits, flat_t[i : i + chunk], reduction="sum")
    return total / n


def _rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    """Grouped-query attention with rotary position embeddings, no biases."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.q = nn.Linear(c.d_model, c.n_q_heads * c.head_dim, bias=False)
        self.k = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim, bias=False)
        self.v = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim, bias=False)
        self.o = nn.Linear(c.n_q_heads * c.head_dim, c.d_model, bias=False)
        self.rep = c.n_q_heads // c.n_kv_heads

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        c = self.c
        q = self.q(x).view(B, T, c.n_q_heads, c.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        k = k.repeat_interleave(self.rep, dim=1)
        v = v.repeat_interleave(self.rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(o.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.gate = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.up = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.down = nn.Linear(c.d_ff, c.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def _norm(d, eps):
    return nn.RMSNorm(d, eps=eps) if hasattr(nn, "RMSNorm") else nn.LayerNorm(d, eps=eps)


class Layer(nn.Module):
    def __init__(self, c: Config, memory=None):
        super().__init__()
        self.n1, self.n2 = _norm(c.d_model, c.norm_eps), _norm(c.d_model, c.norm_eps)
        self.attn, self.ffn = Attention(c), SwiGLU(c)
        self.memory = memory  # None, or a store module (moe.HashMoE / moe.FineLearnedMoE)
        self.mem_norm = None  # RMSNorm on the store's input (train.py --store-norm); None reads the raw residual

    def forward(self, x, cos, sin, rows=None):
        x = x + self.attn(self.n1(x), cos, sin)
        x = x + self.ffn(self.n2(x))
        if self.memory is not None and rows is not None:
            delta, _ = self.memory(rows, x if self.mem_norm is None else self.mem_norm(x))
            x = x + delta
        return x


class Transformer(nn.Module):
    def __init__(self, c: Config, memory_layers: dict | None = None):
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(c.vocab, c.d_model)
        mem = memory_layers or {}
        self.layers = nn.ModuleList([Layer(c, mem.get(i)) for i in range(c.n_layers)])
        self.norm_f = _norm(c.d_model, c.norm_eps)
        self.mem_idx = set(mem)
        hd = c.head_dim
        inv = 1.0 / (c.rope_theta ** (torch.arange(0, hd, 2).float() / hd))
        t = torch.arange(c.seq_len).float()
        f = torch.outer(t, inv)
        self.register_buffer("cos", f.cos()[None, None], persistent=False)
        self.register_buffer("sin", f.sin()[None, None], persistent=False)
        self._init_by_name(seed=c.init_seed)
        # residual projections are scaled by 1/sqrt(2 * n_layers)
        for i, layer in enumerate(self.layers):
            for tag, proj in (("attn.o", layer.attn.o), ("ffn.down", layer.ffn.down)):
                hv = int.from_bytes(hashlib.sha256(f"resid:{c.init_seed}:{i}:{tag}".encode()).digest()[:8], "big")
                g = torch.Generator().manual_seed(hv % (2**63))
                with torch.no_grad():
                    proj.weight.copy_(torch.empty(proj.weight.shape).normal_(0.0, 0.02 / (2 * c.n_layers) ** 0.5, generator=g))

    def _init_by_name(self, seed: int = 0) -> None:
        """Initialise each parameter from a generator seeded by (seed, parameter name), so initialisation does not
        depend on module count, order or placement. Store modules re-initialise themselves afterwards
        (train.py calls their reset_parameters)."""
        for name, p in sorted(self.named_parameters()):
            h = int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "big")
            g = torch.Generator().manual_seed(h % (2**63))
            with torch.no_grad():
                if name.endswith("bias"):
                    p.zero_()
                elif p.dim() >= 2 or "embed" in name:
                    p.copy_(torch.empty(p.shape).normal_(0.0, 0.02, generator=g))
                else:
                    p.fill_(1.0)  # norm weights

    def forward(self, tokens, targets=None, rows_by_layer=None):
        B, T = tokens.shape
        x = self.embed(tokens)
        cos, sin = self.cos[..., :T, :], self.sin[..., :T, :]
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, (rows_by_layer or {}).get(i))
        x = self.norm_f(x)
        if targets is None:
            return F.linear(x, self.embed.weight)  # tied head
        return chunked_cross_entropy(x, self.embed.weight, targets, chunk=4096)


class PairedLoader:
    """Walks an explicit list of sequence ids over a uint16 token memmap. Every model trained with the same ids
    sees byte-identical batches in the same order."""

    def __init__(self, tokens_path, seq_len=2048, batch=8, start=0, seq_ids=None):
        self.tok = np.memmap(tokens_path, dtype=np.uint16, mode="r")
        self.perm = np.asarray(seq_ids)
        self.seq_len, self.batch, self.pos = seq_len, batch, start

    def __iter__(self):
        return self

    def assert_budget(self, n_batches: int) -> None:
        need = n_batches * self.batch
        if need > len(self.perm):
            raise ValueError(f"run needs {need:,} sequences but the train split holds {len(self.perm):,}")

    def __next__(self):
        idx = self.perm[self.pos : self.pos + self.batch]
        if len(idx) < self.batch:
            raise StopIteration
        self.pos += self.batch
        L = self.seq_len
        seqs = np.stack([np.asarray(self.tok[i * L : (i + 1) * L + 1], dtype=np.int64) for i in idx])
        return torch.from_numpy(seqs[:, :-1]), torch.from_numpy(seqs[:, 1:])


def set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
