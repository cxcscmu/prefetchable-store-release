"""Rebuild a model from a train.py checkpoint (dense, hash store, multi-expert store, learned or previous-step
router), for task scoring and serving. The configuration is inferred from the saved tensors; pass the run name so
averaged multi-expert stores ("-avg") can be told apart from summed ones.
"""
from __future__ import annotations

import torch

from model import Config, Transformer, _norm
from moe import FineLearnedMoE, HashMoE


def load_ckpt_model(ck: dict, dev: str, name: str | None = None):
    """-> (model.eval() on dev, assign or None, layers, info). See rows_fn for the rows argument."""
    a = ck["arch"]
    sd = {k: (v.to(torch.float32) if v.is_floating_point() else v) for k, v in ck["state_dict_bf16"].items()}
    cfg = Config(
        d_model=a["d_model"], n_layers=a["n_layers"], n_q_heads=a["q_heads"], n_kv_heads=a["kv_heads"],
        head_dim=a["head_dim"], d_ff=a["d_ff"], vocab=a["vocab"], seq_len=a["seq_len"], init_seed=0,
    )
    layers = list(a.get("moe_layers", ()))
    nex = int(a.get("n_experts", 0))
    info = {"kind": "dense", "store_norm": False, "router": None}
    mem, ctx = {}, None
    if nex and layers:
        L0 = layers[0]
        n, d, f = sd[f"layers.{L0}.memory.bank.gate"].shape
        n_shared = sd[f"layers.{L0}.memory.shared.gate"].shape[2] // f if f"layers.{L0}.memory.shared.gate" in sd else 0
        if f"layers.{L0}.memory.router.weight" in sd:
            prev = sd[f"layers.{L0}.memory.router.weight"].shape[1] == 2 * d  # previous-step router reads [emb ; state(t-1)]
            ctx = {} if prev else None
            topk = n // nex - n_shared
            mem = {L: FineLearnedMoE(n, d, f, topk=topk, n_shared=n_shared, ctx=ctx, prev=prev) for L in layers}
            info.update(kind="learned", router="prev" if prev else "conventional", experts=n, width=f, topk=topk, n_shared=n_shared)
        else:
            # K experts per token when the saved assignment has a slot axis; summed unless the run averaged them
            asg = ck.get("assign")
            multi = asg is not None and torch.as_tensor(asg[L0]).dim() == 2
            sum_k = (multi or n != nex or f != a["d_ff_e"]) and not (name and "-avg" in name)
            slot_gate = sd[f"layers.{L0}.memory.slot_gate.weight"].shape[0] if f"layers.{L0}.memory.slot_gate.weight" in sd else 0
            mem = {L: HashMoE(n, d, f, sum_k=sum_k, n_shared=n_shared, slot_gate=slot_gate) for L in layers}
            info.update(kind="hash", experts=n, width=f, multi=multi, sum_k=sum_k, n_shared=n_shared, slot_gate=slot_gate)
    m = Transformer(cfg, mem)
    if any(k.endswith("mem_norm.weight") for k in sd):
        for L in layers:
            m.layers[L].mem_norm = _norm(cfg.d_model, cfg.norm_eps)
        info["store_norm"] = True
    for mod in mem.values():
        if hasattr(mod, "CF"):
            mod.CF = 0.0  # evaluation never applies a capacity limit
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, f"checkpoint/model mismatch: missing {missing[:5]} unexpected {unexpected[:5]}"
    m = m.to(dev).eval()
    if ctx is not None:
        m.embed.register_forward_hook(lambda mod, i, o: ctx.__setitem__("emb", o))
    assign = None
    if info["kind"] == "hash":
        assert ck.get("assign") is not None, "hash-store checkpoint without its assignment"
        assign = {L: torch.as_tensor(ck["assign"][L]).to(dev) for L in layers}
    return m, assign, layers, info


def rows_fn(assign, layers):
    """The rows argument for model(x, y, rows): expert ids for hash stores, placeholders for learned routers."""
    if not layers:
        return lambda x: None
    if assign is None:
        return lambda x: {L: x for L in layers}
    return lambda x: {L: assign[L][x] for L in layers}
