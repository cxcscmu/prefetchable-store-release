"""Train one configuration of the paper: a dense reference model or a backbone with a token-id-addressed store.

Every configuration shares one recipe; the only free variable per run is the token budget (--mult x 1e9 tokens),
which also sets the training sequence count, so the first 1e9 tokens of every run are byte-identical.

    python src/train.py --arm 6 --mult 8 --store-norm                        # 27.7M backbone + hash-512 store, 8e9 tokens
    torchrun --standalone --nproc_per_node=8 src/train.py --arm 6 --mult 8 --store-norm --micro-batch 8 --accum 1

Data-parallel runs split each 64-sequence optimizer step across ranks, so the set and order of sequences per step
are identical to the single-GPU run. Rank 0 writes OUT/result_<name>.json (losses on every held-out split and the
training curve) and OUT/ckpt_<name>.pt. scripts/paper_runs.txt lists the command for every result in the paper.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from data_partition import TRAIN_SEQS, partition
from evaluation import EvalSuite, default_splits
from model import Config, PairedLoader, Transformer, _norm, set_determinism
from moe import FineLearnedMoE, HashMoE, balanced_assignment
from schedule import lr_multiplier

# ---- the shared recipe
VOCAB = 32_000
SEQ_LEN = 2048
BATCH = 64  # sequences per optimizer step
STEPS = 7629  # optimizer steps per 1e9 tokens (64 x 2048 x 7629 = 1.0e9)
PEAK_LR = 6e-4
WARMUP_STEPS = 300
LR_FLOOR_FRAC = 0.1  # cosine decay to 10% of peak
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
AUX_LOSS_COEF = 0.01  # load-balancing loss weight (learned routers only)
D_FF_E = 853  # expert width: 3 x 512 x 853 = 1.31M parameters per expert

ARMS = [  # index = --arm
    {"name": "tk-dense27", "d": 512, "l": 10, "ff": 1376, "q": 8, "kv": 2, "nex": 0},
    {"name": "tk-dense100", "d": 896, "l": 12, "ff": 2432, "q": 14, "kv": 2, "nex": 0},
    {"name": "tk-dense220", "d": 1152, "l": 16, "ff": 3072, "q": 18, "kv": 3, "nex": 0},
    {"name": "tk-dense405", "d": 1408, "l": 20, "ff": 3776, "q": 22, "kv": 2, "nex": 0},
    {"name": "tk-b27-e64", "d": 512, "l": 10, "ff": 1376, "q": 8, "kv": 2, "nex": 64},
    {"name": "tk-b27-e512", "d": 512, "l": 10, "ff": 1376, "q": 8, "kv": 2, "nex": 512},
    {"name": "tk-b100-e512", "d": 896, "l": 12, "ff": 2432, "q": 14, "kv": 2, "nex": 512},
    {"name": "tk-b220-e512", "d": 1152, "l": 16, "ff": 3072, "q": 18, "kv": 3, "nex": 512},
    {"name": "tk-b27-e128", "d": 512, "l": 10, "ff": 1376, "q": 8, "kv": 2, "nex": 128},
    {"name": "tk-b27-e256", "d": 512, "l": 10, "ff": 1376, "q": 8, "kv": 2, "nex": 256},
    {"name": "tk-b27-e1024", "d": 512, "l": 10, "ff": 1376, "q": 8, "kv": 2, "nex": 1024},
    {"name": "tk-dense1B", "d": 1792, "l": 30, "ff": 4800, "q": 28, "kv": 2, "nex": 0},
]


def store_layers_for(n_layers: int) -> tuple[int, int]:
    """The two store layers sit at depth fractions 0.2 and 0.6 ((2, 6) for the 10-layer backbone)."""
    return (int(0.2 * n_layers), int(0.6 * n_layers))


def train_token_counts(root: str, train_ids: np.ndarray) -> np.ndarray:
    """Token counts over the training split only; the balanced hash assignment is built from these."""
    tok = np.memmap(f"{root}/tokens_u16.bin", dtype=np.uint16, mode="r")
    counts = np.zeros(VOCAB, dtype=np.int64)
    ids = np.sort(train_ids)
    for i in range(0, len(ids), 20_000):
        chunk = ids[i : i + 20_000]
        seqs = np.concatenate([np.asarray(tok[j * SEQ_LEN : (j + 1) * SEQ_LEN]) for j in chunk])
        counts += np.bincount(seqs, minlength=VOCAB)
    return counts


def run_name(arm: dict, a) -> str:
    name = f"{arm['name']}-{a.mult}e9" + (f"-s{a.init_seed}" if a.init_seed else "")
    if a.gran > 1:
        name += f"-k{a.gran}" if a.full_width else f"-g{a.gran}"
    if a.indep_hash:
        name += "-ih"
    if a.slot_avg:
        name += "-avg"
    if a.slot_gate:
        name += "-sg"
    if a.router != "hash":
        name += {"learned": "-lrn", "prev": "-prev"}[a.router] + (f"-sh{a.shared}" if a.shared else "") + (f"-cf{a.cap_factor:g}" if a.cap_factor != 1.25 else "")
    if a.router == "hash" and a.shared:
        name += f"-sh{a.shared}"
    if a.store_norm:
        name += "-sn"
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", type=int, required=True, help="configuration index into ARMS (1-based)")
    ap.add_argument("--mult", type=int, required=True, help="token budget as a multiple of 1e9")
    ap.add_argument("--root", default="data/v3", help="corpus directory built by scripts/build_corpus.py")
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=8, help="micro-batch x accum x world size must equal 64")
    ap.add_argument("--out", default="out")
    ap.add_argument("--store-norm", action="store_true", help="RMS-normalise the store's input (all paper store runs use this)")
    ap.add_argument("--init-seed", type=int, default=0, help="initialisation seed; 0 for the main runs, 1 for the second seeds")
    ap.add_argument(
        "--gran", type=int, default=1,
        help="K experts per token and store layer. Without --full-width each expert is split into K experts of width "
        "d_ff_e/K (K x as many experts, stored and active parameters unchanged)",
    )
    ap.add_argument("--full-width", action="store_true", help="with --gran K: keep expert count and width and read K full-width experts per token")
    ap.add_argument("--indep-hash", action="store_true", help="decorrelate the balanced hash assignments across layers and across the K slots")
    ap.add_argument("--slot-avg", action="store_true", help="with --gran K (hash): average the K expert outputs instead of summing them")
    ap.add_argument("--slot-gate", action="store_true", help="with --gran K (hash): combine the K outputs with learned softmax weights of the store-layer state")
    ap.add_argument(
        "--router", choices=("hash", "learned", "prev"), default="hash",
        help="hash = token-id store; learned = conventional top-K router at the store layer; prev = previous-step router "
        "([embedding of token t ; store input at t-1], prefetchable)",
    )
    ap.add_argument("--shared", type=int, default=0, help="how many of the --gran experts are always-on shared experts")
    ap.add_argument("--cap-factor", type=float, default=1.25, help="learned routers: train-time capacity factor; 0 = dropless")
    ap.add_argument("--resume-every", type=int, default=0, help="save a resumable state every N steps and resume from it when present")
    ap.add_argument("--resume-dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--max-steps", type=int, default=0, help="testing only: stop after N optimizer steps")
    a = ap.parse_args()

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    assert a.micro_batch * a.accum * world == BATCH, "micro-batch x accum x world size must be 64"
    arm = ARMS[a.arm - 1]
    name = run_name(arm, a)
    steps = STEPS * a.mult
    train_seqs = TRAIN_SEQS if a.mult == 1 else steps * BATCH
    parts = partition(f"{a.root}/seq_permutation.npy", train_seqs=train_seqs)
    train_all = parts["train"]  # token counts for the hash assignment use the whole train split on every rank
    if world > 1:
        import torch.distributed as dist

        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        # rank r takes micro-batches r, r + world, ... of each step's accum x world micro-batches
        full = parts["train"][: steps * BATCH]
        per_step = full.reshape(steps, world * a.accum, a.micro_batch)
        parts = dict(parts)
        parts["train"] = per_step[:, rank::world, :].reshape(-1)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.max_steps:
        steps = a.max_steps
    set_determinism(1234)
    cfg = Config(d_model=arm["d"], n_layers=arm["l"], n_q_heads=arm["q"], n_kv_heads=arm["kv"], head_dim=64, d_ff=arm["ff"],
                 seq_len=SEQ_LEN, init_seed=a.init_seed)

    assign = None
    if arm["nex"]:
        layers = store_layers_for(arm["l"])
        ff_e = D_FF_E
        counts = train_token_counts(a.root, train_all)
        jit = 1.0 if a.indep_hash else 0.0
        if a.router != "hash":
            assert 0 <= a.shared < a.gran and not (a.full_width or a.indep_hash)
            learned_ctx = {} if a.router == "prev" else None
            mem = {
                L: FineLearnedMoE(arm["nex"] * a.gran, cfg.d_model, ff_e // a.gran, topk=a.gran - a.shared, n_shared=a.shared,
                                  ctx=learned_ctx, prev=(a.router == "prev")).to(dev)
                for L in layers
            }
            for m_ in mem.values():
                m_.CF = a.cap_factor
        elif a.gran > 1:
            assert 0 <= a.shared < a.gran, "--shared must be smaller than --gran"
            nex_g, ff_g = (arm["nex"], ff_e) if a.full_width else (arm["nex"] * a.gran, ff_e // a.gran)
            assign = {
                L: torch.from_numpy(
                    np.stack([balanced_assignment(counts, nex_g, salt=L * 1000 + k, jitter=jit) for k in range(a.gran - a.shared)], axis=-1)
                ).to(dev)
                for L in layers
            }
            mem = {
                L: HashMoE(nex_g, cfg.d_model, ff_g, sum_k=not a.slot_avg, n_shared=a.shared,
                           slot_gate=(a.gran - a.shared) if a.slot_gate else 0).to(dev)
                for L in layers
            }
        else:
            assert not (a.slot_avg or a.slot_gate), "--slot-avg / --slot-gate need --gran K > 1"
            assign = {L: torch.from_numpy(balanced_assignment(counts, arm["nex"], salt=L, jitter=jit)).to(dev) for L in layers}
            mem = {L: HashMoE(arm["nex"], cfg.d_model, ff_e).to(dev) for L in layers}
        model = Transformer(cfg, mem).to(dev)
        if a.store_norm:
            for L in layers:
                model.layers[L].mem_norm = _norm(cfg.d_model, cfg.norm_eps).to(dev)
        for L, m in mem.items():
            m.reset_parameters(a.init_seed, name=f"experts:L{L}")
        if a.router == "prev":
            model.embed.register_forward_hook(lambda m, i, o: learned_ctx.__setitem__("emb", o))
        if a.router != "hash":
            rows_of = lambda x: {L: x for L in layers}  # placeholder rows: the router picks the experts
        else:
            rows_of = lambda x: {L: assign[L][x] for L in layers}
        added = sum(p.numel() for m in mem.values() for p in m.parameters())
    else:
        layers, mem, added = (), {}, 0
        model = Transformer(cfg, None).to(dev)
        rows_of = lambda x: None

    nb = sum(p.numel() for n, p in model.named_parameters() if "embed" not in n)
    if rank == 0:
        print(json.dumps({"arm": name, "steps": steps, "train_seqs": int(train_seqs), "world": world, "tokens": int(steps * BATCH * SEQ_LEN),
                          "store_layers": list(layers), "non_embed_M": round(nb / 1e6, 1), "added_M": round(added / 1e6, 1)}), flush=True)
    raw_model = model
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None, gradient_as_bucket_view=True
        )

    # two parameter groups (backbone, store) with identical hyperparameters
    p_exp = [p for n, p in model.named_parameters() if "memory" in n]
    p_bb = [p for n, p in model.named_parameters() if "memory" not in n]
    groups = [{"params": p_bb, "weight_decay": WEIGHT_DECAY, "lr_mult": 1.0}]
    if p_exp:
        groups.append({"params": p_exp, "weight_decay": WEIGHT_DECAY, "lr_mult": 1.0})
    opt = torch.optim.AdamW(groups, lr=PEAK_LR, betas=(0.9, 0.95))
    evals = EvalSuite(default_splits(a.root), batch=8, max_batches=32)
    if rank == 0:
        os.makedirs(a.out, exist_ok=True)

    hist, t0, wall_prev, start_step = [], time.time(), 0.0, 0
    rpath = os.path.join(a.out, f"resume_{name}.pt")
    done_path = os.path.join(a.out, f"result_{name}.json")
    if os.path.exists(done_path) and not a.max_steps:
        prev = json.load(open(done_path))
        if prev.get("steps") == steps and prev.get("final_eval_on"):
            if rank == 0:
                print(json.dumps({"ALREADY_DONE": name, "main_on": prev["final_eval_on"]["main_heldout"]["loss"]}), flush=True)
            if world > 1:
                torch.distributed.destroy_process_group()
            return
    if a.resume_every and os.path.exists(rpath):
        st = torch.load(rpath, map_location=dev, weights_only=False)
        assert st["arm"] == name, "resume file from a different run"
        raw_model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start_step, hist, wall_prev = st["step"], st["hist"], st["wall_s"]
        if rank == 0:
            print(json.dumps({"RESUMED": name, "step": start_step}), flush=True)
        del st
    loader = PairedLoader(f"{a.root}/tokens_u16.bin", seq_len=SEQ_LEN, batch=a.micro_batch, seq_ids=parts["train"],
                          start=start_step * a.accum * a.micro_batch)
    loader.assert_budget(steps * a.accum)

    def save_resume(next_step):
        cast = (lambda t: t.to(torch.bfloat16)) if a.resume_dtype == "bf16" else (lambda t: t)
        msd = {k: (cast(v) if v.is_floating_point() else v) for k, v in raw_model.state_dict().items()}
        osd = opt.state_dict()
        osd["state"] = {
            i: {k: (cast(v) if torch.is_tensor(v) and v.is_floating_point() and k != "step" else v) for k, v in st_.items()}
            for i, st_ in osd["state"].items()
        }
        torch.save({"arm": name, "step": next_step, "hist": hist, "wall_s": wall_prev + time.time() - t0, "model": msd, "opt": osd},
                   rpath + ".tmp")
        os.replace(rpath + ".tmp", rpath)

    sched_steps = STEPS * a.mult  # the cosine horizon is the full budget even under --max-steps
    for step in range(start_step, steps):
        lr = PEAK_LR * lr_multiplier(step, sched_steps, WARMUP_STEPS, LR_FLOOR_FRAC, "cosine")
        for grp in opt.param_groups:
            grp["lr"] = lr * grp["lr_mult"]
        opt.zero_grad(set_to_none=True)
        for i in range(a.accum):
            x, y = next(loader)
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
                loss = model(x, y, rows_of(x)) / a.accum
                if a.router != "hash":
                    lm_loss = loss.detach()
                    loss = loss + AUX_LOSS_COEF * sum(raw_model.layers[L].memory.last_aux_loss for L in layers) / a.accum
            if world > 1 and i < a.accum - 1:
                with model.no_sync():
                    loss.backward()
            else:
                loss.backward()  # DDP averages over ranks: the mean over the 64-sequence step
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        if step % 100 == 0 and rank == 0:
            rec = {"step": step, "loss": float(lm_loss if a.router != "hash" else loss.detach()) * a.accum, "lr": lr,
                   "elapsed_s": wall_prev + time.time() - t0}
            if a.router != "hash":
                rec["util"] = [round(raw_model.layers[L].memory.last_util, 4) for L in layers]
                rec["drop"] = [round(raw_model.layers[L].memory.last_drop, 4) for L in layers]
            hist.append(rec)
            print(json.dumps(rec), flush=True)
        if a.resume_every and rank == 0 and (step + 1) % a.resume_every == 0 and step + 1 < steps:
            save_resume(step + 1)

    model = raw_model
    if world > 1:
        import torch.distributed as dist

        dist.barrier()
        if rank != 0:
            dist.destroy_process_group()
            return
    model.eval()
    if arm["nex"]:
        final_on = evals.evaluate(model, forward_fn=lambda m, x, y: m(x, y, rows_of(x)), device=dev)
        final_off = evals.evaluate(model, device=dev)  # the same model with the store switched off
    else:
        final_on, final_off = evals.evaluate(model, device=dev), None
    print(json.dumps({"DONE": name, "main_on": final_on["main_heldout"]["loss"],
                      "main_off": final_off["main_heldout"]["loss"] if final_off else None}), flush=True)
    json.dump(
        {"arm": name, "base": arm["name"], "mult": a.mult, "init_seed": a.init_seed, "steps": steps, "world": world,
         "train_seqs": int(train_seqs), "non_embed": nb, "added_params": added, "final_eval_on": final_on, "final_eval_off": final_off,
         "history": hist, "wall_s": wall_prev + time.time() - t0, "resumed_from_step": start_step},
        open(os.path.join(a.out, f"result_{name}.json"), "w"), indent=2,
    )
    sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in model.state_dict().items()}
    arch = {"kind": "hashmoe" if arm["nex"] else "dense", "d_model": arm["d"], "n_layers": arm["l"], "q_heads": arm["q"],
            "kv_heads": arm["kv"], "head_dim": 64, "d_ff": arm["ff"], "d_ff_e": D_FF_E, "n_experts": arm["nex"],
            "moe_layers": list(layers), "vocab": VOCAB, "seq_len": SEQ_LEN}
    torch.save({"state_dict_bf16": sd, "arch": arch, "assign": {L: assign[L].cpu().numpy() for L in layers} if assign else None},
               os.path.join(a.out, f"ckpt_{name}.pt"))
    if a.resume_every and os.path.exists(rpath):
        os.remove(rpath)
    if world > 1:
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
