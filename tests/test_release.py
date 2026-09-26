"""Fast checks (CPU, a few seconds): run names, the balanced hash assignment, and forward/backward passes of every
store type on a tiny model. Run with: python -m pytest tests"""

import argparse
import os
import shlex
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import Config, Transformer, _norm  # noqa: E402
from moe import FineLearnedMoE, HashMoE, balanced_assignment  # noqa: E402
from train import ARMS, run_name  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _parse(flags):
    ap = argparse.ArgumentParser()
    for f in ("--arm", "--mult", "--init-seed", "--gran", "--shared"):
        ap.add_argument(f, type=int, default=0 if f in ("--init-seed", "--shared") else 1)
    ap.add_argument("--cap-factor", type=float, default=1.25)
    ap.add_argument("--router", default="hash")
    for f in ("--store-norm", "--full-width", "--indep-hash", "--slot-avg", "--slot-gate"):
        ap.add_argument(f, action="store_true")
    return ap.parse_args(shlex.split(flags))


def test_paper_runs_reproduce_their_names():
    n = 0
    for line in open(os.path.join(ROOT, "scripts", "paper_runs.txt")):
        if line.startswith("#") or not line.strip():
            continue
        name, flags = line.split(None, 1)
        a = _parse(flags)
        assert run_name(ARMS[a.arm - 1], a) == name
        assert os.path.exists(os.path.join(ROOT, "results", "training", f"result_{name}.json"))
        n += 1
    assert n == len(os.listdir(os.path.join(ROOT, "results", "training")))


def test_balanced_assignment_balances_token_mass():
    counts = (1e6 / np.arange(1, 32_001) ** 1.1).astype(np.int64) + 1
    a = balanced_assignment(counts, 64, salt=2)
    mass = np.bincount(a, weights=counts, minlength=64)
    assert a.min() == 0 and a.max() == 63
    assert mass.max() <= mass.mean() + counts.max()  # greedy bound: a single token id cannot be split
    assert np.array_equal(a, balanced_assignment(counts, 64, salt=2))  # deterministic


def _tiny(mem):
    c = Config(d_model=32, n_layers=4, n_q_heads=4, n_kv_heads=2, head_dim=8, d_ff=64, vocab=32_000, seq_len=16)
    m = Transformer(c, mem)
    for L in mem:
        m.layers[L].mem_norm = _norm(32, c.norm_eps)
        mem[L].reset_parameters(0, name=f"experts:L{L}")
    return m


def _step(m, rows_of):
    x = torch.randint(0, 32_000, (2, 16))
    loss = m(x, x, rows_of(x))
    loss.backward()
    assert torch.isfinite(loss)
    return loss


def test_store_is_a_no_op_at_init_and_trains():
    counts = np.random.default_rng(0).integers(1, 100, 32_000)
    layers = (1, 2)
    for kw, K in [({}, 1), ({"sum_k": True}, 4), ({"sum_k": False}, 4), ({"sum_k": False, "slot_gate": 4}, 4)]:
        mem = {L: HashMoE(8, 32, 16, **kw) for L in layers}
        assign = {L: torch.from_numpy(np.stack([balanced_assignment(counts, 8, salt=L * 1000 + k, jitter=1.0) for k in range(K)], -1)
                                      if K > 1 else balanced_assignment(counts, 8, salt=L)) for L in layers}
        m = _tiny(mem)
        x = torch.randint(0, 32_000, (2, 16))
        with torch.no_grad():  # zero-initialised expert outputs: the store starts as an exact no-op
            assert torch.equal(m(x, x, {L: assign[L][x] for L in layers}), m(x, x, None))
        _step(m, lambda x: {L: assign[L][x] for L in layers})
        assert all(m.layers[L].memory.bank.down.grad is not None for L in layers)


def test_learned_routers_train():
    layers = (1, 2)
    for prev in (False, True):
        ctx = {} if prev else None
        mem = {L: FineLearnedMoE(32, 32, 4, topk=3, n_shared=1, ctx=ctx, prev=prev) for L in layers}
        for m_ in mem.values():
            m_.CF = 0.0
        m = _tiny(mem)
        if prev:
            m.embed.register_forward_hook(lambda mod, i, o: ctx.__setitem__("emb", o))
        _step(m, lambda x: {L: x for L in layers})
        assert all(m.layers[L].memory.router.weight.grad is not None for L in layers)
