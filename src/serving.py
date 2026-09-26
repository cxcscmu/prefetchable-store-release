"""Serving stack for a model whose experts live on disk.

  DecodeSession  KV-cache incremental decode; tested equal to the full-sequence training forward.
  ExpertStore    experts serialised to a page-aligned file (bf16 / int8 / int4), read with O_DIRECT on Linux
                 (buffered fallback elsewhere).
  Prefetcher     background thread that reads the NEXT token's experts and copies them to the GPU while the
                 current token computes; prefetch=False is the synchronous ablation.
"""

from __future__ import annotations

import json
import mmap
import os
import threading

import torch
import torch.nn.functional as F

from model import _rope

PAGE = 4096


# ------------------------------------------------------------------ decode
class DecodeSession:
    """Batch-1 (or small-B) incremental decode over a model.Transformer.

    `rows_fn(x_t) -> {layer: rows}` supplies memory rows for the CURRENT token only --
    the same function the training loop and evaluation use.
    `expert_provider`, when set, is called as provider(layer, expert_ids_tensor) and must
    return the (gate, up, down) tensors to use -- the hook the disk store plugs into.
    """

    def __init__(self, model, rows_fn=None, expert_provider=None):
        self.m = model
        self.c = model.c
        self.rows_fn = rows_fn
        self.provider = expert_provider
        self.pos = 0
        self.kv = [None] * self.c.n_layers  # (k,v) per layer, [B,n_kv,t,hd]
        self.prev_in = {}  # previous-step router: per store layer, that layer's store input at the last decoded position

    @torch.no_grad()
    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        """x_t [B,1] int64 -> logits [B, vocab] for the next token."""
        m, c, p = self.m, self.c, self.pos
        cos = m.cos[..., p : p + 1, :]
        sin = m.sin[..., p : p + 1, :]
        x = m.embed(x_t)
        rbl = self.rows_fn(x_t) if self.rows_fn else {}
        for i, layer in enumerate(m.layers):
            h = layer.n1(x)
            a = layer.attn
            B = x.shape[0]
            q = a.q(h).view(B, 1, c.n_q_heads, c.head_dim).transpose(1, 2)
            k = a.k(h).view(B, 1, c.n_kv_heads, c.head_dim).transpose(1, 2)
            v = a.v(h).view(B, 1, c.n_kv_heads, c.head_dim).transpose(1, 2)
            q, k = _rope(q, cos, sin), _rope(k, cos, sin)
            if self.kv[i] is None:
                self.kv[i] = (k, v)
            else:
                pk, pv = self.kv[i]
                self.kv[i] = (torch.cat([pk, k], 2), torch.cat([pv, v], 2))
            fk, fv = self.kv[i]
            fk = fk.repeat_interleave(a.rep, dim=1)
            fv = fv.repeat_interleave(a.rep, dim=1)
            o = F.scaled_dot_product_attention(q, fk, fv, is_causal=False)
            x = x + a.o(o.transpose(1, 2).reshape(B, 1, -1))
            if layer.ffn is not None:
                x = x + layer.ffn(layer.n2(x))
            if layer.memory is not None and i in rbl:
                mem = layer.memory
                xin = x if getattr(layer, "mem_norm", None) is None else layer.mem_norm(x)  # store pre-norm, as in training
                if getattr(mem, "prev", False):
                    # previous-step router: hand it this layer's store input from the previous decode step (zeros at
                    # position 0), which is what its roll reads in a full-sequence forward; then remember this step's
                    mem.decode_prev = self.prev_in.get(i) if i in self.prev_in else torch.zeros_like(xin)
                    self.prev_in[i] = xin
                if self.provider is not None and hasattr(mem, "bank"):
                    delta = self._provided_expert_delta(mem, i, rbl[i], xin)
                else:
                    delta, _ = mem(rbl[i], xin)
                if getattr(mem, "prev", False):
                    mem.decode_prev = None
                x = x + delta
        x = m.norm_f(x)
        self.pos += 1
        return F.linear(x[:, 0], m.embed.weight)

    def _provided_expert_delta(self, mem, layer_idx, rows, h):
        """HashMoE path with weights supplied by the store instead of mem.bank tensors."""
        B, T = rows.shape
        gate, up, down = self.provider(layer_idx, rows.reshape(-1))
        if gate.dtype != h.dtype:  # weight-only quantization: the store dequantizes to fp32; compute in the model's dtype
            gate, up, down = gate.to(h.dtype), up.to(h.dtype), down.to(h.dtype)
        hf = h.reshape(B * T, 1, -1)
        y = torch.bmm(F.silu(torch.bmm(hf, gate)) * torch.bmm(hf, up), down)
        return y.reshape(B, T, -1)


# ------------------------------------------------------------------ store
def _quant(w: torch.Tensor, dtype: str):
    """w [d, f] fp32 -> (payload bytes-tensor, scales) for the store formats."""
    if dtype == "bf16":
        return w.to(torch.bfloat16).view(torch.uint8).reshape(-1), None
    if dtype == "int8":
        s = w.abs().amax(dim=0).clamp_min(1e-8) / 127.0  # per-column
        q = torch.clamp((w / s).round(), -127, 127).to(torch.int8)
        return q.view(torch.uint8).reshape(-1), s.to(torch.float16)
    if dtype == "int4":
        d, f = w.shape
        g = 64
        wg = w.reshape(d, f // g if f % g == 0 else 1, -1) if False else w
        # group along the FIRST axis in blocks of 64 (d is 512/853 -> pad)
        pad = (-d) % g
        wp = torch.cat([w, torch.zeros(pad, f)], 0) if pad else w
        wg = wp.reshape(-1, g, f)
        s = wg.abs().amax(dim=1).clamp_min(1e-8) / 7.0  # [d/g, f]
        q = torch.clamp((wg / s.unsqueeze(1)).round(), -7, 7).to(torch.int8) + 8
        flat = q.reshape(-1, f)
        packed = (flat[0::2] | (flat[1::2] << 4)).to(torch.uint8)
        return packed.reshape(-1), s.to(torch.float16)
    raise ValueError(dtype)


def _dequant(payload: torch.Tensor, scales, dtype: str, d: int, f: int) -> torch.Tensor:
    if dtype == "bf16":
        return payload.view(torch.bfloat16).reshape(d, f).to(torch.float32)
    if dtype == "int8":
        return payload.view(torch.int8).reshape(d, f).to(torch.float32) * scales.float()
    if dtype == "int4":
        g = 64
        pad = (-d) % g
        dp = d + pad
        pk = payload.reshape(dp // 2, f)
        lo = (pk & 0x0F).to(torch.int8) - 8
        hi = ((pk >> 4) & 0x0F).to(torch.int8) - 8
        flat = torch.empty(dp, f, dtype=torch.int8)
        flat[0::2], flat[1::2] = lo, hi
        w = flat.reshape(-1, g, f).float() * scales.float().unsqueeze(1)
        return w.reshape(dp, f)[:d]
    raise ValueError(dtype)


class ExpertStore:
    """One file per (dtype): every (layer, expert) record page-aligned and contiguous.

    write(): from a HashMoE bank state {layer: {gate,up,down fp32 tensors [N,d,f]/[N,f,d]}}.
    read_expert(): returns fp32 (gate[d,f], up[d,f], down[f,d]) for one expert.
    O_DIRECT on Linux; buffered fallback elsewhere (tests); `direct` reported.
    """

    def __init__(self, path: str):
        self.path = path
        self.meta = None
        self.fd = None
        self.direct = False

    @staticmethod
    def write(path: str, banks: dict, dtype: str):
        meta = {"dtype": dtype, "layers": {}, "page": PAGE}
        off = 0
        with open(path, "wb") as fh:
            for L, b in sorted(banks.items()):
                N, d, f = b["gate"].shape
                recs = []
                for e in range(N):
                    parts, scs = [], []
                    for nm in ("gate", "up", "down"):
                        w = b[nm][e].float()
                        pl, sc = _quant(w, dtype)
                        parts.append(pl)
                        scs.append(sc)
                    blob = torch.cat(parts).numpy().tobytes()
                    sblob = (
                        torch.cat([s.reshape(-1) for s in scs]).numpy().tobytes()
                        if scs[0] is not None
                        else b""
                    )
                    rec = blob + sblob
                    pad = (-len(rec)) % PAGE
                    fh.write(rec + b"\x00" * pad)
                    recs.append(
                        {
                            "off": off,
                            "len": len(rec),
                            "payloads": [len(p) for p in parts],
                            "scales": [0 if s is None else s.numel() * 2 for s in scs],
                        }
                    )
                    off += len(rec) + pad
                meta["layers"][str(L)] = {"N": N, "d": d, "f": f, "recs": recs}
        json.dump(meta, open(path + ".meta.json", "w"))
        return meta

    def open(self):
        self.meta = json.load(open(self.path + ".meta.json"))
        mx = max(r["len"] for lm in self.meta["layers"].values() for r in lm["recs"])
        self._buf = mmap.mmap(-1, mx + ((-mx) % PAGE))  # ONE reused aligned buffer --
        # attempt 1 allocated a fresh mmap per read and paid ~8x the disk's latency for it
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECT"):
            try:
                self.fd = os.open(self.path, flags | os.O_DIRECT)
                self.direct = True
                return self
            except OSError:
                pass
        self.fd = os.open(self.path, flags)
        return self

    def read_raw(self, layer: int, e: int, out: torch.Tensor | None = None):
        """Fast path: one preadv into the reused buffer, one copy out (into `out` if given,
        e.g. a pinned staging tensor). Returns (uint8 tensor of rec['len'], rec)."""
        rec = self.meta["layers"][str(layer)]["recs"][e]
        n = rec["len"]
        got = os.preadv(self.fd, [self._buf], rec["off"])
        assert got >= n, f"short read {got} < {n}"
        src = torch.frombuffer(self._buf, dtype=torch.uint8, count=n)
        if out is not None:
            out[:n].copy_(src)
            return out[:n], rec
        return src.clone(), rec

    def read_expert(self, layer: int, e: int):
        lm = self.meta["layers"][str(layer)]
        rec = lm["recs"][e]
        d, f, dtype = lm["d"], lm["f"], self.meta["dtype"]
        nbytes = rec["len"]
        aligned = nbytes + ((-nbytes) % PAGE)
        buf = mmap.mmap(-1, aligned)
        got = os.preadv(self.fd, [buf], rec["off"])
        assert got >= nbytes, f"short read {got} < {nbytes}"
        raw = torch.frombuffer(buf, dtype=torch.uint8, count=nbytes).clone()
        buf.close()
        out, pos = [], 0
        spos = sum(rec["payloads"])
        shapes = [(d, f), (d, f), (f, d)]
        for i, nm in enumerate(("gate", "up", "down")):
            pl = raw[pos : pos + rec["payloads"][i]]
            pos += rec["payloads"][i]
            sc = None
            if rec["scales"][i]:
                sc_raw = raw[spos : spos + rec["scales"][i]]
                spos += rec["scales"][i]
                sc = sc_raw.view(torch.float16)
                dd, ff = shapes[i]
                g = 64
                sc = sc.reshape(-1, ff) if dtype == "int4" else sc.reshape(ff)
            dd, ff = shapes[i]
            out.append(_dequant(pl, sc, dtype, dd, ff))
        return tuple(out)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)


def dequant_record(raw: torch.Tensor, rec: dict, lm: dict, dtype: str):
    """raw uint8 [rec.len] on ANY device -> (gate[d,f], up[d,f], down[f,d]) fp32, same
    device. Mirrors _dequant exactly; tests assert equality against the CPU path."""
    d, f = lm["d"], lm["f"]
    shapes = [(d, f), (d, f), (f, d)]
    out, pos = [], 0
    spos = sum(rec["payloads"])
    for i in range(3):
        pl = raw[pos : pos + rec["payloads"][i]]
        pos += rec["payloads"][i]
        dd, ff = shapes[i]
        if dtype == "bf16":
            out.append(pl.view(torch.bfloat16).reshape(dd, ff).to(torch.float32))
            continue
        sc = raw[spos : spos + rec["scales"][i]].view(torch.float16)
        spos += rec["scales"][i]
        if dtype == "int8":
            out.append(
                pl.view(torch.int8).reshape(dd, ff).to(torch.float32) * sc.reshape(ff).to(torch.float32)
            )
        else:
            g = 64
            pad = (-dd) % g
            dp = dd + pad
            pk = pl.reshape(dp // 2, ff)
            lo = (pk & 0x0F).to(torch.int8) - 8
            hi = ((pk >> 4) & 0x0F).to(torch.int8) - 8
            flat = torch.empty(dp, ff, dtype=torch.int8, device=raw.device)
            flat[0::2], flat[1::2] = lo, hi
            w = flat.reshape(-1, g, ff).to(torch.float32) * sc.reshape(-1, ff).to(torch.float32).unsqueeze(1)
            out.append(w.reshape(dp, ff)[:dd])
    return tuple(out)


# ------------------------------------------------------------------ prefetcher
class Prefetcher:
    """One-token-ahead expert prefetch on a background thread.

    request(layer_ids, expert_ids): issue fetches for the NEXT step. take(): block until
    ready, return {(layer, expert): (gate, up, down)}. Sync mode (prefetch=False) fetches
    inline inside take() -- the synchronous ablation, same code path minus the overlap.
    """

    def __init__(self, store: ExpertStore, prefetch: bool = True, device="cpu"):
        self.store, self.prefetch, self.dev = store, prefetch, device
        self._pending, self._result = None, None
        self._ev_go, self._ev_done = threading.Event(), threading.Event()
        self._stop = False
        self.last_fetch_s = 0.0
        mx = max(r["len"] for lm in store.meta["layers"].values() for r in lm["recs"])
        pin = device != "cpu" and torch.cuda.is_available()
        self._stage = torch.empty(mx, dtype=torch.uint8, pin_memory=pin)
        # side stream: attempt 3's worker used torch.cuda.synchronize(), which waits on the
        # WHOLE device -- including main-stream compute -- partially serializing the overlap
        # it exists to create. Event-scoped sync on a dedicated stream fixes it.
        self._stream = torch.cuda.Stream() if pin else None
        if prefetch:
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()

    def _fetch(self, reqs):
        import time as _t

        t0 = _t.time()
        out = {}
        dt = self.store.meta["dtype"]
        if self._stream is not None:
            with torch.cuda.stream(self._stream):
                for L, e in reqs:
                    raw, rec = self.store.read_raw(L, int(e), out=self._stage)
                    lm = self.store.meta["layers"][str(L)]
                    dev_raw = raw.to(self.dev, non_blocking=True)
                    out[(L, int(e))] = dequant_record(dev_raw, rec, lm, dt)
                ev = torch.cuda.Event()
                ev.record(self._stream)
            ev.synchronize()  # waits ONLY on the side stream
        else:
            for L, e in reqs:
                raw, rec = self.store.read_raw(L, int(e), out=self._stage)
                lm = self.store.meta["layers"][str(L)]
                out[(L, int(e))] = dequant_record(raw.clone(), rec, lm, dt)
        self.last_fetch_s = _t.time() - t0
        return out

    def _loop(self):
        while True:
            self._ev_go.wait()
            self._ev_go.clear()
            if self._stop:
                return
            self._result = self._fetch(self._pending)
            self._ev_done.set()

    def request(self, reqs):
        if self.prefetch:
            self._pending = list(reqs)
            self._ev_done.clear()
            self._ev_go.set()
        else:
            self._pending = list(reqs)

    def take(self):
        if self.prefetch:
            self._ev_done.wait()
            return self._result
        return self._fetch(self._pending)

    def stop(self):
        if self.prefetch:
            self._stop = True
            self._ev_go.set()
            # join: the caller closes the store next, and the last decode step's lookahead
            # request may still be in flight (Tier-0 batch sweep hit preadv on a recycled fd)
            self._t.join(timeout=60)
