"""Generate every table, figure and in-text number of the paper from the result files in results/.

    python analysis/make_assets.py       # writes analysis/out/tables/*.tex and analysis/out/figs/*.pdf

results/training  one result_<run>.json per training run (train.py output)
results/tasks     zero-shot accuracies (scripts/score_tasks.py output)
results/serving   serving measurements (scripts/serve_iso.py, scripts/served_precision.py, src/serve_bench.py)
All numbers in the text are LaTeX macros written to tables/facts2.tex.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import statistics as st
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
TRAIN, TASKDIR, SERVE = (os.path.join(RES, d) for d in ("training", "tasks", "serving"))
OUT_T = os.path.join(HERE, "out", "tables")
OUT_F = os.path.join(HERE, "out", "figs")
os.makedirs(OUT_T, exist_ok=True)
os.makedirs(OUT_F, exist_ok=True)
import frontier_fit as gt  # noqa: E402

# ----------------------------------------------------------------------------- style: one visual system (figstyle.py)
from figstyle import *  # noqa: F401,F403  (palette, rcParams, ptitle, ygrid)
C = {"dense": DARK, "store": BLUE, "store2": BLUE_L, "prev": ORANGE, "conv": TEAL, "lever": PURPLE, "gate": "#B8860B", "grid": GRID}


def J(p):
    return json.load(open(p))


def M(n):
    return f"{n / 1e6:.1f}M" if n < 1e8 else (f"{n / 1e6:.0f}M" if n < 1e9 else f"{n / 1e9:.2f}B")


# ----------------------------------------------------------------------------- dense frontier, every budget, seeds
DENSE_N = {"dense27": 27.7e6, "dense100": 100.5e6, "dense220": 219.5e6, "dense405": 405.6e6, "dense1B": 980.7e6}
dense = {k: {} for k in (1, 2, 4, 8)}
dense_seed = {}  # (name, k) -> [seed0, seed1]
for f in sorted(glob.glob(os.path.join(TRAIN, "result_tk-dense*.json"))):
    d = J(f)
    m = re.match(r"tk-(dense\w+?)-(\d)e9(-s1)?$", d["arm"])
    if not m:
        continue
    nm, k, s1 = m.group(1), int(m.group(2)), bool(m.group(3))
    L = d["final_eval_on"]["main_heldout"]["loss"]
    dense_seed.setdefault((nm, k), [None, None])[1 if s1 else 0] = L
    if not s1:
        dense[k][DENSE_N[nm]] = L
# The 1e9-token losses of the four smaller dense models are from the first training of each (identical recipe, data
# and seed, trained before train.py's multi-budget runs); those are the values the paper reports. The same-seed
# train.py reruns in results/training (e.g. dense-27.7M 3.1951) agree to within 0.01 nats and are not used here.
for N, L in {27.7e6: 3.2020, 100.5e6: 3.0239, 219.5e6: 2.9229, 405.6e6: 2.8714}.items():
    dense[1][N] = L
for nm, N in DENSE_N.items():
    if N in dense[1] and nm != "dense1B":
        dense_seed.setdefault((nm, 1), [None, None])[0] = dense[1][N]
FIT = {k: gt.fit_dense(v) for k, v in dense.items()}
assert all(len(dense[k]) == 5 for k in (1, 2, 4, 8)), {k: len(v) for k, v in dense.items()}

# ----------------------------------------------------------------------------- every store / router / lever result
R = {}  # run name -> record
for f in sorted(glob.glob(os.path.join(TRAIN, "result_*.json"))):
    d = J(f)
    e = d["final_eval_on"]
    R[d["arm"]] = {
        "loss": e["main_heldout"]["loss"], "k": d["mult"], "non_embed": d["non_embed"], "added": d["added_params"],
        "seed": d.get("init_seed", 0), "base": d["base"],
    }
# run-name alias used by the multi-expert tables
for _arm in list(R):
    if _arm.endswith("-g4-ih-sn") and "-lrn-" not in _arm and "-prev-" not in _arm:
        R.setdefault(_arm.replace("-g4-ih-sn", "-k4-ih-sn"), R[_arm])
BB = {"b27": 27.7e6, "b100": 100.5e6, "b220": 219.5e6}


def name(base, k, seed=0, suffix=""):
    return f"tk-{base}-{k}e9" + (f"-s{seed}" if seed else "") + suffix + "-sn"


def rec(base, k, seed=0, suffix=""):
    return R.get(name(base, k, seed, suffix))


def stats(base, k, seed=0, suffix=""):
    """loss, N_eq, r, overhead H for a store arm read against the frontier of its budget."""
    x = rec(base, k, seed, suffix)
    if x is None:
        return None
    nb = BB[base.split("-")[0]]
    neq = gt.dense_equiv(x["loss"], FIT[k])
    r = gt.exponent(x["loss"], nb, float(x["non_embed"]), FIT[k])
    return {"loss": x["loss"], "neq": neq, "r": r, "H": x["non_embed"] / neq, "P": x["non_embed"], "B": nb,
            "added": x["added"], "seed": seed}


STORES = ["b27-e64", "b27-e128", "b27-e256", "b27-e512"]
BUDGETS = [1, 2, 4, 8]
BACKS = ["b27-e512", "b100-e512", "b220-e512"]
CONV, PREV = "-g4-lrn-sh1-cf0", "-g4-prev-sh1-cf0"
LEVER_AVG, LEVER_SUM, LEVER_AVGSG = "-k4-ih-avg", "-k4-ih", "-k4-ih-avg-sg"

# ----------------------------------------------------------------------------- task scores
TASKS = ["lambada_openai", "piqa", "hellaswag", "arc_easy"]
T = {}
for f in sorted(glob.glob(os.path.join(TASKDIR, "tasks_*.json"))):
    for arm, v in J(f).items():
        T[arm] = {t: v[t]["acc,none"] for t in TASKS if t in v}


def tasks_str(arm, sep=" / "):
    v = T.get(arm)
    return sep.join(f"{100 * v[t]:.1f}" for t in TASKS) if v else sep.join(["--"] * len(TASKS))


# ----------------------------------------------------------------------------- facts
facts = {}


def F(k, v):
    """LaTeX macro names cannot contain digits: 1/2/4/8 become One/Two/Four/Eight."""
    for d_, w_ in (("1", "One"), ("2", "Two"), ("4", "Four"), ("8", "Eight"), ("6", "Six"), ("5", "Five"), ("0", "Zero"), ("3", "Three")):
        k = k.replace(d_, w_)
    facts[k] = v


def FK(k):
    for d_, w_ in (("1", "One"), ("2", "Two"), ("4", "Four"), ("8", "Eight"), ("6", "Six"), ("5", "Five"), ("0", "Zero"), ("3", "Three")):
        k = k.replace(d_, w_)
    return facts[k]


def fmt_delta(x):
    return "$0.000$" if abs(x) < 0.0005 else f"${x:+.3f}$"


# --- store axis
per_doubling = {}
for k in BUDGETS:
    ys = [stats(s, k)["loss"] for s in STORES]
    xs = [6, 7, 8, 9]
    mx, my = sum(xs) / 4, sum(ys) / 4
    per_doubling[k] = -sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    F(f"perDoubling{k}", f"{per_doubling[k]:.3f}")
    F(f"spread{k}", f"{ys[0] - ys[3]:.3f}")
# --- token axis: hash-512 vs dense-220M, gains over backbone
for k in BUDGETS:
    s = stats("b27-e512", k)
    F(f"gapHashFiveTwelve{k}", fmt_delta(s["loss"] - dense[k][219.5e6]))
    F(f"gainBTwentySeven{k}", f"{dense[k][27.7e6] - s["loss"]:.3f}")
    F(f"lossHashFiveTwelve{k}", f"{s['loss']:.4f}")
    F(f"rHashFiveTwelve{k}", f"{s['r']:.2f}")
    F(f"neqHashFiveTwelve{k}", M(s["neq"]))
    F(f"HHashFiveTwelve{k}", f"{s['H']:.1f}")
# --- backbone axis
for b, dn in [("b100-e512", 405.6e6), ("b220-e512", 980.7e6)]:
    tag = "Hundred" if b.startswith("b100") else "TwoTwenty"
    for k in BUDGETS:
        s = stats(b, k)
        F(f"gap{tag}{k}", fmt_delta(s["loss"] - dense[k][dn]))
        F(f"loss{tag}{k}", f"{s['loss']:.4f}")
        F(f"r{tag}{k}", f"{s['r']:.2f}")
        F(f"H{tag}{k}", f"{s['H']:.1f}")
        F(f"neq{tag}{k}", M(s["neq"]))
        F(f"added{tag}{k}", M(s["neq"] - s["B"]))
        F(f"gain{tag}{k}", f"{dense[k][s['B']] - s['loss']:.3f}")
for k in BUDGETS:
    s = stats("b27-e512", k)
    F(f"addedTwentySeven{k}", M(s["neq"] - s["B"]))
    F(f"realised{k}", " / ".join(f"{stats(b, k)['neq'] / stats(b, k)['P']:.2f}" for b in BACKS))
# task-level parity of the 220M store against dense-1B, and the 100M store's lead over dense-405M in absolute terms
_gaps = []
for k in BUDGETS:
    a, b_ = T.get(name("b220-e512", k)), T.get(f"tk-dense1B-{k}e9")
    if a and b_:
        _gaps += [abs(100 * (a[t] - b_[t])) for t in TASKS]
F("taskGapTwoTwenty", f"{max(_gaps):.1f}")
_a4, _b4 = T[name("b220-e512", 4)], T["tk-dense1B-4e9"]
F("lambadaGapFour", f"{100 * (_b4['lambada_openai'] - _a4['lambada_openai']):.1f}")
_a8, _b8 = T[name("b220-e512", 8)], T["tk-dense1B-8e9"]
F("lambadaGapEight", f"{100 * (_b8['lambada_openai'] - _a8['lambada_openai']):.1f}")
F("otherGapEight", f"{max(abs(100 * (_a8[t] - _b8[t])) for t in TASKS[1:]):.1f}")
F("otherGapFourEight", f"{max(abs(100 * (x[t] - y[t])) for x, y in ((_a4, _b4), (_a8, _b8)) for t in TASKS[1:]):.1f}")
_h = [dense[k][405.6e6] - stats("b100-e512", k)["loss"] for k in BUDGETS]
F("gapHundredAbsLo", f"{min(_h):.3f}")
F("gapHundredAbsHi", f"{max(_h):.3f}")
# --- gain over backbone for the larger stores (token axis, appendix B)
for b, N, tag in [("b100-e512", 100.5e6, "Hundred"), ("b220-e512", 219.5e6, "TwoTwenty")]:
    for k in BUDGETS:
        F(f"gainB{tag}{k}", f"{dense[k][N] - stats(b, k)['loss']:.3f}")
# --- Switch top-1 control (learned router, cap-factor 0) against the token-id store
for k in BUDGETS:
    _sw = rec("b27-e512", k, 0, "-lrn-cf0")  # the Switch top-1 control (COARSE, defined below)
    if _sw:
        F(f"switchGap{k}", f"{_sw['loss'] - stats('b27-e512', k)['loss']:.3f}")
# --- exchange-rate range across the grid at every budget
allr = [stats(s, k)["r"] for s in STORES for k in BUDGETS] + [stats(b, k)["r"] for b in BACKS[1:] for k in BUDGETS]
F("rMin", f"{min(allr):.2f}")
F("rMax", f"{max(allr):.2f}")
# --- overhead at 8e9
F("HGridEight", " / ".join(f"{stats(b, 8)['H']:.1f}" for b in BACKS))
# what the published joint law implies for a Switch MoE at our smallest backbone (values from docs/lit, consistent convention)
LAW_R_8E9 = (0.50, 0.53)
ratio = stats("b27-e512", 8)["P"] / stats("b27-e512", 8)["B"]
F("lawHLo", f"{ratio ** (1 - LAW_R_8E9[1]):.1f}")
F("lawHHi", f"{ratio ** (1 - LAW_R_8E9[0]):.1f}")
F("lawRLo", f"{LAW_R_8E9[0]:.2f}")
F("lawRHi", f"{LAW_R_8E9[1]:.2f}")
# frontier same-data overheads (total / matched dense), from each report
F("HDSMoESixteen", f"{16.4 / 6.9:.1f}")
F("HDSMoELarge", f"{144.6 / 67:.1f}")
F("HLing", f"{17.5 / 6.1:.1f}")
F("HQwen", f"{30.5 / 14.8:.1f}")
# dense frontier steepening
g1, g8 = dense[1][27.7e6] - dense[1][219.5e6], dense[8][27.7e6] - dense[8][219.5e6]
F("frontierSteepen", f"{100 * (g8 / g1 - 1):.0f}")

# --- routers
for k in BUDGETS:
    h, c, p = stats("b27-e512", k), stats("b27-e512", k, 0, CONV), stats("b27-e512", k, 0, PREV)
    F(f"lossConv{k}", f"{c['loss']:.4f}" if c else "--")
    F(f"lossPrev{k}", f"{p['loss']:.4f}")
    F(f"convMinusHash{k}", fmt_delta(c["loss"] - h["loss"]) if c else "--")
    F(f"prevMinusHash{k}", fmt_delta(p["loss"] - h["loss"]))
    F(f"prevMinusConv{k}", fmt_delta(p["loss"] - c["loss"]) if c else "--")
    F(f"convMinusHash{k}Abs", f"{abs(c['loss'] - h['loss']):.3f}" if c else "--")
    F(f"prevMinusHash{k}Abs", f"{abs(p['loss'] - h['loss']):.3f}")
    F(f"prevMinusConv{k}Abs", f"{abs(p['loss'] - c['loss']):.3f}" if c else "--")
    F(f"rConv{k}", f"{c['r']:.3f}" if c else "--")
    F(f"rPrev{k}", f"{p['r']:.3f}")
    F(f"gainConv{k}", f"{dense[k][27.7e6] - c['loss']:.3f}" if c else "--")
    F(f"gainPrev{k}", f"{dense[k][27.7e6] - p['loss']:.3f}")
c8, p8, h8 = stats("b27-e512", 8, 0, CONV), stats("b27-e512", 8, 0, PREV), stats("b27-e512", 8)
F("prevRetained", f"{100 * (dense[8][27.7e6] - p8['loss']) / (dense[8][27.7e6] - c8['loss']):.0f}")
ps1 = stats("b27-e512", 8, 1, PREV)
F("prevSeedSpreadEight", f"{abs(ps1['loss'] - p8['loss']):.3f}" if ps1 else "--")
ps1_1 = stats("b27-e512", 1, 1, PREV)
F("prevSeedSpreadOne", f"{abs(ps1_1['loss'] - stats('b27-e512', 1, 0, PREV)['loss']):.3f}" if ps1_1 else "--")

# --- store lever (1e9 and, when present, 8e9)
for k in (1, 8):
    b = stats("b27-e512", k)
    for tag, suf in [("Avg", LEVER_AVG), ("Sum", LEVER_SUM), ("AvgSg", LEVER_AVGSG)]:
        s = stats("b27-e512", k, 0, suf)
        F(f"loss{tag}{k}", f"{s['loss']:.4f}" if s else "--")
        F(f"delta{tag}{k}", fmt_delta(s["loss"] - b["loss"]) if s else "--")
        F(f"delta{tag}{k}Abs", f"{abs(s['loss'] - b['loss']):.3f}" if s else "--")
        F(f"r{tag}{k}", f"{s['r']:.3f}" if s else "--")
        F(f"neq{tag}{k}", M(s["neq"]) if s else "--")
        F(f"H{tag}{k}", f"{s['H']:.1f}" if s else "--")
        F(f"H{tag}{k}", f"{s['H']:.1f}" if s else "--")
        ss = stats("b27-e512", k, 1, suf)
        F(f"loss{tag}{k}SeedOne", f"{ss['loss']:.4f}" if ss else "--")
F("rBaseOne", f"{stats('b27-e512', 1)['r']:.3f}")
F("rBaseEight", f"{stats('b27-e512', 8)['r']:.3f}")
avg1 = stats("b27-e512", 1, 0, LEVER_AVG)
F("leverActive", f"{100 * ((avg1['B'] + 4 * avg1['added'] / 512) / (avg1['B'] + avg1['added'] / 512) - 1):.0f}")
F("leverPending", "" if stats("b27-e512", 8, 0, LEVER_AVG) else "pending")

# --- seeds
seed_rows = []
for label, base, suf in [("dense-27.7M", None, "dense27"), ("dense-100M", None, "dense100"), ("dense-220M", None, "dense220"),
                         ("dense-405M", None, "dense405"), ("dense-1B", None, "dense1B"),
                         ("hash-64", "b27-e64", ""), ("hash-128", "b27-e128", ""), ("hash-256", "b27-e256", ""), ("hash-512", "b27-e512", ""),
                         ("100M + hash-512", "b100-e512", ""), ("220M + hash-512", "b220-e512", ""),
                         ("four averaged", "b27-e512", LEVER_AVG),
                         ("conventional router", "b27-e512", CONV), ("previous-step router", "b27-e512", PREV)]:
    for k in BUDGETS:
        if base is None:
            pair = dense_seed.get((suf, k))
            if pair and pair[1] is not None:
                seed_rows.append((label, k, pair[0], pair[1]))
        else:
            a, b_ = rec(base, k, 0, suf), rec(base, k, 1, suf)
            if a and b_:
                seed_rows.append((label, k, a["loss"], b_["loss"]))
store_spreads = [abs(a - b) for lab, k, a, b in seed_rows if "dense" not in lab and "router" not in lab]
dense_spreads = [abs(a - b) for lab, k, a, b in seed_rows if "dense" in lab]
F("seedSpreadStoreLo", f"{min(store_spreads):.3f}")
F("seedSpreadStoreHi", f"{max(store_spreads):.3f}")
F("seedSpreadDenseLo", f"{min(dense_spreads):.3f}")
F("seedSpreadDenseHi", f"{max(dense_spreads):.3f}")
F("nSeedPairs", str(len(seed_rows)))
hs = [abs(a - b) for lab, k, a, b in seed_rows if lab == "hash-512" and k == 1]
F("seedSpreadHashFiveTwelveOne", f"{hs[0]:.3f}" if hs else "--")

# ----------------------------------------------------------------------------- serving
# serving head-to-head: the bf16-compute repeats (both arms bf16, store experts int8 on disk) when they exist,
# else the earlier repeats in which the store arm computed in fp32; the earlier ones are kept for the appendix either way
reps_fp32 = [J(p) for p in sorted(glob.glob(os.path.join(SERVE, "iso_fp32_rep*.json")))]
reps_bf16 = [J(p) for p in sorted(glob.glob(os.path.join(SERVE, "iso_bf16_rep*.json")))]
reps_bf16 = [r for r in reps_bf16 if "b220_hash512_disk_int8" in r and "tok_s" in r["b220_hash512_disk_int8"]]
SERVING_BF16 = len(reps_bf16) > 0
reps = reps_bf16 if SERVING_BF16 else reps_fp32
A_, B_, S_ = "dense1B_fp32", "dense1B_bf16", "b220_hash512_disk_int8"
F("isoStoreCompute", "bf16" if SERVING_BF16 else "fp32")
F("isoRunsFpThirtyTwo", str(len(reps_fp32)))
# held-out loss evaluated through the served weights (scripts/served_precision.py)
_sp = os.path.join(SERVE, "served_precision.json")
if os.path.exists(_sp):
    _spj = J(_sp)
    F("isoLossDenseServed", f"{_spj['dense1B_bf16']['main_heldout']:.4f}")
    F("isoLossStoreServed", f"{_spj['store_int8experts_bf16']['main_heldout']:.4f}")
    F("isoLossGapServed", f"{_spj['dense1B_bf16']['main_heldout'] - _spj['store_int8experts_bf16']['main_heldout']:.3f}")
    F("isoLossStoreIntEightFpThirtyTwo", f"{_spj['store_int8experts_fp32']['main_heldout']:.4f}")
    F("isoLossDenseFpThirtyTwoCheck", f"{_spj['dense1B_fp32']['main_heldout']:.4f}")
    F("isoLossStoreFpThirtyTwoCheck", f"{_spj['store_fp32']['main_heldout']:.4f}")
    F("isoIntEightQualityCost", f"{_spj['store_int8experts_fp32']['main_heldout'] - _spj['store_fp32']['main_heldout']:+.4f}")
    F("isoBfSixteenQualityCostDense", f"{_spj['dense1B_bf16']['main_heldout'] - _spj['dense1B_fp32']['main_heldout']:+.4f}")
    SERVED_LOSS = True
else:
    for k_ in ("isoLossDenseServed", "isoLossStoreServed", "isoLossGapServed", "isoLossStoreIntEightFpThirtyTwo", "isoLossDenseFpThirtyTwoCheck", "isoLossStoreFpThirtyTwoCheck", "isoIntEightQualityCost", "isoBfSixteenQualityCostDense"):
        F(k_, "--")
    SERVED_LOSS = False
# every repeat is used; the first ran on a contended node (both models ~45% slow) and is identified in the appendix
clean = list(reps)
excluded = []
slow = min(reps, key=lambda r: r[A_]["tok_s"])
def im(arm, key, rs=clean): return st.mean(r[arm][key] for r in rs)
def isd(arm, key, rs=clean): return st.pstdev(r[arm][key] for r in rs) if len(rs) > 1 else 0.0
ratios = [r[S_]["tok_s"] / r[B_]["tok_s"] for r in clean]
F("isoSpeedNew", f"{st.mean(ratios):.2f}")
F("isoSpeedSdNew", f"{st.pstdev(ratios):.2f}")
F("isoRunsNew", str(len(clean)))
F("isoExcluded", str(len(excluded)))
F("isoExcludedRatio", "--")
F("isoSpeedLoAll", f"{min(ratios):.2f}")
F("isoSpeedHiAll", f"{max(ratios):.2f}")
F("isoSpeedBfNew", f"{st.mean(r[S_]['tok_s'] / r[B_]['tok_s'] for r in reps):.2f}")
F("isoSlowNodeRatio", f"{slow[S_]['tok_s'] / slow[B_]['tok_s']:.2f}")
F("isoStepStore", f"{im(S_, 'step_ms'):.1f}")
F("isoFetchMed", f"{im(S_, 'fetch_ms'):.1f}")
F("isoStepStoreLo", f"{min(r[S_]['step_ms'] for r in reps):.1f}")
F("isoStepStoreHi", f"{max(r[S_]['step_ms'] for r in reps):.1f}")
F("isoVramFpNew", f"{im(A_, 'peak_vram_bytes') / im(S_, 'peak_vram_bytes'):.1f}")
F("isoVramBfNew", f"{im(B_, 'peak_vram_bytes') / im(S_, 'peak_vram_bytes'):.1f}")
F("isoVramStoreGB", f"{im(S_, 'peak_vram_bytes') / 1e9:.2f}")
F("isoVramFpGB", f"{im(A_, 'peak_vram_bytes') / 1e9:.2f}")
F("isoVramBfGB", f"{im(B_, 'peak_vram_bytes') / 1e9:.2f}")
F("isoLossStore", f"{clean[0][S_]['loss']:.4f}")
F("isoLossDense", f"{clean[0][A_]['loss']:.4f}")
F("isoLossGapNew", f"{clean[0][A_]['loss'] - clean[0][S_]['loss']:.3f}")
F("isoFetchLo", f"{min(r[S_]['fetch_ms'] for r in clean):.1f}")
F("isoFetchHi", f"{max(r[S_]['fetch_ms'] for r in clean):.1f}")
F("isoResidentNew", f"{clean[0][A_]['resident_params'] / clean[0][S_]['resident_params']:.1f}")
F("isoStoreGB", f"{clean[0][S_]['store_bytes'] / 1e9:.2f}")
F("isoPowerDense", f"{im(A_, 'watts'):.0f}")
F("isoPowerStore", f"{im(S_, 'watts'):.0f}")
with open(os.path.join(OUT_T, "serving.tex"), "w") as f:
    f.write("\\begin{tabular}{lrr}\n\\toprule\n")
    f.write(" & dense-1B, bf16, fully resident & 220M backbone + store on NVMe (int8) \\\\\n\\midrule\n")
    if SERVED_LOSS:
        f.write(f"held-out loss through the served weights & {_spj['dense1B_bf16']['main_heldout']:.4f} & \\textbf{{{_spj['store_int8experts_bf16']['main_heldout']:.4f}}} \\\\\n")
    else:
        f.write(f"held-out loss (fp32 checkpoint) & {clean[0][B_]['loss']:.4f} & \\textbf{{{clean[0][S_]['loss']:.4f}}} \\\\\n")
    f.write(f"decode speed (tok/s) & {im(B_, 'tok_s'):.0f} $\\pm$ {isd(B_, 'tok_s'):.0f} & \\textbf{{{im(S_, 'tok_s'):.0f} $\\pm$ {isd(S_, 'tok_s'):.0f}}} ({st.mean(r[S_]['tok_s'] / r[B_]['tok_s'] for r in reps):.2f}$\\times$) \\\\\n")
    f.write(f"step time (ms) & {im(B_, 'step_ms'):.1f} & {im(S_, 'step_ms'):.1f} ({im(S_, 'fetch_ms'):.1f} fetch, overlapped) \\\\\n")
    f.write(f"peak GPU memory (GB) & {im(B_, 'peak_vram_bytes') / 1e9:.2f} & \\textbf{{{im(S_, 'peak_vram_bytes') / 1e9:.2f}}} ({im(B_, 'peak_vram_bytes') / im(S_, 'peak_vram_bytes'):.1f}$\\times$) \\\\\n")
    f.write(f"board power (W) & {im(B_, 'watts'):.0f} & {im(S_, 'watts'):.0f} \\\\\n")
    f.write(f"resident parameters (incl.\\ embeddings) & {clean[0][B_]['resident_params'] / 1e6:.0f}M & \\textbf{{{clean[0][S_]['resident_params'] / 1e6:.0f}M}} \\\\\n")
    f.write(f"on-disk store (int8) & -- & {clean[0][S_]['store_bytes'] / 1e9:.2f} GB \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

with open(os.path.join(OUT_T, "servingreps.tex"), "w") as f:
    f.write("\\begin{tabular}{llrrrrrr}\n\\toprule\n")
    f.write("repeat & store compute & dense-1B bf16 tok/s & store tok/s & store / dense & store step (ms) & store fetch (ms) & store peak GB \\\\\n\\midrule\n")
    for lab, rs in ([("bf16", reps_bf16)] if SERVING_BF16 else []) + [("fp32", reps_fp32)]:
        for i, r in enumerate(rs, 1):
            f.write(f"{i} & {lab} & {r[B_]['tok_s']:.1f} & {r[S_]['tok_s']:.1f} & {r[S_]['tok_s'] / r[B_]['tok_s']:.2f} & {r[S_]['step_ms']:.1f} & {r[S_]['fetch_ms']:.1f} & {r[S_]['peak_vram_bytes'] / 1e9:.2f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ----------------------------------------------------------------------------- benchmark table: every main configuration, one protocol
COARSE = "-lrn-cf0"
def act_of(x, per_layer_experts, k=1):
    return x["non_embed"] - x["added"] + k * x["added"] / per_layer_experts
with open(os.path.join(OUT_T, "benchmark.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n")
    f.write(" & & & & \\multicolumn{2}{c}{held-out loss} & \\multicolumn{4}{c}{zero-shot accuracy at $10^9$ (\\%)} \\\\\n")
    f.write("model & backbone & active & total & $10^9$ & $8\\times10^9$ & LAMBADA & PIQA & HellaSwag & ARC-e \\\\\n\\midrule\n")
    for nm, N, lab in [("dense27", 27.7e6, "dense-27.7M"), ("dense100", 100.5e6, "dense-100M"), ("dense220", 219.5e6, "dense-220M"), ("dense405", 405.6e6, "dense-405M"), ("dense1B", 980.7e6, "dense-1B")]:
        f.write(f"{lab} & {M(N)} & {M(N)} & {M(N)} & {dense[1][N]:.4f} & {dense[8][N]:.4f} & {tasks_str(f'tk-{nm}-1e9', ' & ')} \\\\\n")
    f.write("\\midrule\n")
    rows = [("27.7M + hash-64", "b27-e64", "", 64, 1, False), ("27.7M + hash-128", "b27-e128", "", 128, 1, False), ("27.7M + hash-256", "b27-e256", "", 256, 1, False),
            ("27.7M + hash-512", "b27-e512", "", 512, 1, True),
            ("27.7M + Switch MoE-512 (control)", "b27-e512", COARSE, 512, 1, False),
            ("100M + hash-512", "b100-e512", "", 512, 1, True), ("220M + hash-512", "b220-e512", "", 512, 1, True)]
    for lab, b, suf, E_, k_, bold in rows:
        x1, x8 = rec(b, 1, 0, suf), rec(b, 8, 0, suf)
        Bres = x1["non_embed"] - x1["added"]
        act = act_of(x1, E_, k_)
        L1 = f"{x1['loss']:.4f}"; L8 = f"{x8['loss']:.4f}" if x8 else "--"
        cell = (lambda v: f"\\textbf{{{v}}}") if bold else (lambda v: v)
        f.write(f"{cell(lab)} & {M(Bres)} & {M(act)} & {M(x1['non_embed'])} & {cell(L1)} & {cell(L8)} & {tasks_str(name(b, 1, 0, suf), ' & ')} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# overhead against the literature: total parameters over the dense model matched on the same data
LIT = [("DeepSeekMoE-16B", 16.4, 6.9, "2T", "dai2024deepseekmoe"), ("DeepSeekMoE-145B", 144.6, 67.0, "245B", "dai2024deepseekmoe"),
       ("Ling-mini-beta", 17.5, 6.1, "1T", "ling2025leverage"), ("Qwen3-30B-A3B", 30.5, 14.8, "36T", "qwen3")]
with open(os.path.join(OUT_T, "overhead.tex"), "w") as f:
    f.write("\\begin{tabular}{llrrrl}\n\\toprule\n")
    f.write("model & inactive parameters live in & total & matched dense & total / dense & training tokens \\\\\n\\midrule\n")
    for nm_, tot, dn, tok, cite in LIT:
        f.write(f"{nm_} \\citep{{{cite}}} & accelerator memory & {tot:g}B & {dn:g}B & {tot / dn:.1f}$\\times$ & {tok} \\\\\n")
    f.write("\\midrule\n")
    for b, lab in [("b27-e512", "27.7M + hash-512 (ours)"), ("b100-e512", "100M + hash-512 (ours)"), ("b220-e512", "220M + hash-512 (ours)")]:
        s8 = stats(b, 8)
        f.write(f"{lab} & NVMe & {M(s8['P'])} & {M(s8['neq'])} & {s8['H']:.1f}$\\times$ & 8B \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ----------------------------------------------------------------------------- Table 1: the backbone axis vs its dense comparator
with open(os.path.join(OUT_T, "axes.tex"), "w") as f:
    f.write("\\begin{tabular}{llrrrr}\n\\toprule\n")
    f.write("training tokens & model & resident & loss & $\\Delta$ vs dense & LAMBADA / PIQA / HellaSwag / ARC-e \\\\\n\\midrule\n")
    for k in BUDGETS:
        s = stats("b220-e512", k)
        f.write(f"${k}\\times10^9$ & dense-1B (fully resident) & 981M & {dense[k][980.7e6]:.4f} & -- & {tasks_str(f'tk-dense1B-{k}e9')} \\\\\n")
        f.write(f" & \\textbf{{220M backbone + 3.0B store}} & \\textbf{{220M}} & \\textbf{{{s['loss']:.4f}}} & {fmt_delta(s['loss'] - dense[k][980.7e6])} & {tasks_str(name('b220-e512', k))} \\\\\n")
    f.write("\\midrule\n")
    for k in BUDGETS:
        s = stats("b100-e512", k)
        f.write(f"${k}\\times10^9$ & dense-405M (fully resident) & 406M & {dense[k][405.6e6]:.4f} & -- & {tasks_str(f'tk-dense405-{k}e9')} \\\\\n")
        f.write(f" & \\textbf{{100M backbone + 2.3B store}} & \\textbf{{100M}} & \\textbf{{{s['loss']:.4f}}} & {fmt_delta(s['loss'] - dense[k][405.6e6])} & {tasks_str(name('b100-e512', k))} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ----------------------------------------------------------------------------- Table 2: the ledger at 8e9 (main text) and the grid (appendix)
def ledger_rows(k):
    rows = []
    for b, E in [("b27-e64", 64), ("b27-e128", 128), ("b27-e256", 256), ("b27-e512", 512), ("b100-e512", 512), ("b220-e512", 512)]:
        s = stats(b, k)
        act = s["B"] + s["added"] / E  # one fetched expert per store layer, two layers
        nearest = min(dense[k].items(), key=lambda kv: abs(kv[1] - s["loss"]))
        rows.append((b, E, s, act, nearest))
    return rows


with open(os.path.join(OUT_T, "ledger.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrrrrrrr}\n\\toprule\n")
    f.write("backbone + store & resident $B$ & total $P$ & active & loss & nearest dense & $N_{\\mathrm{eq}}$ & overhead $P/N_{\\mathrm{eq}}$ & $r$ \\\\\n\\midrule\n")
    for b, E, s, act, nearest in ledger_rows(8):
        lab = {"b27": "27.7M", "b100": "100M", "b220": "220M"}[b.split("-")[0]] + f" + hash-{E}"
        f.write(f"{lab} & {M(s['B'])} & {M(s['P'])} & {M(act)} & {s['loss']:.4f} & {M(nearest[0])} ({nearest[1]:.3f}) & {M(s['neq'])} & {s['H']:.1f}$\\times$ & {s['r']:.2f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

with open(os.path.join(OUT_T, "grid.tex"), "w") as f:
    f.write("\\begin{tabular}{l" + "rr" * 4 + "}\n\\toprule\n")
    f.write(" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{${k}\\times10^9$ tokens}}" for k in BUDGETS) + " \\\\\n")
    f.write("backbone + store & " + " & ".join("loss & $N_{\\mathrm{eq}}$" for _ in BUDGETS) + " \\\\\n\\midrule\n")
    for b in STORES + BACKS[1:]:
        lab = {"b27": "27.7M", "b100": "100M", "b220": "220M"}[b.split("-")[0]] + " + hash-" + b.split("-e")[1]
        f.write(lab + " & " + " & ".join(f"{stats(b, k)['loss']:.4f} & {M(stats(b, k)['neq'])}" for k in BUDGETS) + " \\\\\n")
    f.write("\\midrule\n")
    for nm, N in DENSE_N.items():
        lab = {"dense27": "dense-27.7M", "dense100": "dense-100M", "dense220": "dense-220M", "dense405": "dense-405M", "dense1B": "dense-1B"}[nm]
        f.write(lab + " & " + " & ".join(f"{dense[k][N]:.4f} & --" for k in BUDGETS) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ----------------------------------------------------------------------------- appendix tables: routers, lever, seeds, tasks
with open(os.path.join(OUT_T, "routers.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
    f.write("router (2,048 experts per store layer, 1 shared + top-3, dropless) & $10^9$ & $2\\times10^9$ & $4\\times10^9$ & $8\\times10^9$ \\\\\n\\midrule\n")
    f.write("token-id hash (reference) & " + " & ".join(f"{stats('b27-e512', k)['loss']:.4f}" for k in BUDGETS) + " \\\\\n")
    f.write("conventional (store-layer state, not prefetchable) & " + " & ".join(FK(f"lossConv{k}") for k in BUDGETS) + " \\\\\n")
    f.write("\\textbf{previous-step (prefetchable)} & " + " & ".join(f"\\textbf{{{FK(f'lossPrev{k}')}}}" for k in BUDGETS) + " \\\\\n")
    f.write("\\midrule\n")
    f.write("previous-step $-$ hash & " + " & ".join(FK(f"prevMinusHash{k}") for k in BUDGETS) + " \\\\\n")
    f.write("previous-step $-$ conventional & " + " & ".join(FK(f"prevMinusConv{k}") for k in BUDGETS) + " \\\\\n")
    f.write("gain over dense-27.7M: hash / conventional / previous-step & " + " & ".join(f"{FK(f'gainBTwentySeven{k}')} / {FK(f'gainConv{k}')} / {FK(f'gainPrev{k}')}" for k in BUDGETS) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

with open(os.path.join(OUT_T, "lever.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrrrr}\n\\toprule\n")
    f.write("hash-512 store, 27.7M backbone & fetches / token & loss ($10^9$) & overhead & loss ($8\\times10^9$) & overhead \\\\\n\\midrule\n")
    def _H(tag, k): return FK(f"H{tag}{k}") + ("$\\times$" if FK(f"H{tag}{k}") != "--" else "")
    f.write(f"one expert per store layer (baseline) & 2 & {stats('b27-e512', 1)['loss']:.4f} & {stats('b27-e512', 1)['H']:.1f}$\\times$ & {stats('b27-e512', 8)['loss']:.4f} & {stats('b27-e512', 8)['H']:.1f}$\\times$ \\\\\n")
    f.write(f"four experts, outputs summed (control) & 8 & {FK('lossSum1')} & {_H('Sum', 1)} & {FK('lossSum8')} & {_H('Sum', 8)} \\\\\n")
    f.write(f"\\textbf{{four experts, outputs averaged}} & 8 & \\textbf{{{FK('lossAvg1')}}} & \\textbf{{{_H('Avg', 1)}}} & \\textbf{{{FK('lossAvg8')}}} & \\textbf{{{_H('Avg', 8)}}} \\\\\n")
    f.write(f"four experts, averaged, learned weights & 8 & {FK('lossAvgSg1')} & {_H('AvgSg', 1)} & {FK('lossAvgSg8')} & {_H('AvgSg', 8)} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

with open(os.path.join(OUT_T, "seeds.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrrr}\n\\toprule\narm & tokens & seed 0 & seed 1 & $|\\Delta|$ \\\\\n\\midrule\n")
    for lab, k, a, b in seed_rows:
        f.write(f"{lab} & ${k}\\times10^9$ & {a:.4f} & {b:.4f} & {abs(a - b):.3f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

with open(os.path.join(OUT_T, "tasksall.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrrrr}\n\\toprule\nmodel & tokens & LAMBADA & PIQA & HellaSwag & ARC-e \\\\\n\\midrule\n")
    rows = []
    for k in BUDGETS:
        for lab, arm in [("dense-27.7M", f"tk-dense27-{k}e9"), ("dense-100M", f"tk-dense100-{k}e9"), ("dense-220M", f"tk-dense220-{k}e9"), ("dense-405M", f"tk-dense405-{k}e9"), ("dense-1B", f"tk-dense1B-{k}e9")]:
            if arm in T:
                rows.append((lab, k, T[arm]))
        for b in STORES + BACKS[1:]:
            arm = name(b, k)
            if arm in T:
                lab = {"b27": "27.7M", "b100": "100M", "b220": "220M"}[b.split("-")[0]] + " + hash-" + b.split("-e")[1]
                rows.append((lab, k, T[arm]))
        for lab, suf in [("27.7M + hash-512, previous-step router", PREV), ("27.7M + hash-512, four averaged", LEVER_AVG)]:
            arm = name("b27-e512", k, 0, suf)
            if arm in T:
                rows.append((lab, k, T[arm]))
    for lab, k, v in rows:
        f.write(f"{lab} & ${k}\\times10^9$ & " + " & ".join(f"{100 * v[t]:.1f}" for t in TASKS) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ----------------------------------------------------------------------------- Figure 2: three axes
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3))
budget_shade = {1: "#a9c9ef", 2: "#6aa3e6", 4: "#3d86d9", 8: "#174f9a"}
for k in BUDGETS:
    ys = [stats(s, k)["loss"] for s in STORES]
    ax[0].plot([64, 128, 256, 512], ys, "o-", color=budget_shade[k])
    ax[0].text(590, ys[-1], f"{k}B", color=budget_shade[k], fontsize=6.5, va="center", bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax[0].axhline(dense[k][219.5e6], color=C["dense"], lw=0.8, ls=":")
ax[0].text(66, dense[1][219.5e6] + 0.006, "dotted: dense-220M", color=C["dense"], fontsize=6)
ax[0].set_xscale("log", base=2); ax[0].set_xticks([64, 128, 256, 512]); ax[0].set_xticklabels(["64", "128", "256", "512"])
ax[0].set_xlim(56, 760); ax[0].set_xlabel("experts per store layer (27.7M backbone)"); ax[0].set_ylabel("held-out loss (nats)")
ptitle(ax[0], "a", "Store size, four budgets")
toks = np.array(BUDGETS, dtype=float)


def direct(a, ys, label, color, dy=0.0):
    a.text(8.7, ys[-1] + dy, label, color=color, fontsize=6.3, va="center")


ys = [dense[k][27.7e6] for k in BUDGETS]; ax[1].plot(toks, ys, "s--", color=C["dense"]); direct(ax[1], ys, "dense-27.7M", C["dense"])
ys = [dense[k][219.5e6] for k in BUDGETS]; ax[1].plot(toks, ys, "s-", color=C["dense"]); direct(ax[1], ys, "dense-220M", C["dense"], +0.028)
ys = [stats("b27-e512", k)["loss"] for k in BUDGETS]; ax[1].plot(toks, ys, "o-", color=C["store"]); direct(ax[1], ys, "27.7M + hash-512", C["store"], -0.028)
g = [stats("b27-e512", k)["loss"] - dense[k][219.5e6] for k in BUDGETS]
ax[1].set_xscale("log", base=2); ax[1].set_xticks(BUDGETS); ax[1].set_xticklabels(["1B", "2B", "4B", "8B"]); ax[1].set_xlim(0.85, 40); ax[1].set_ylim(2.56, 3.24)
ax[1].set_xlabel("training tokens"); ptitle(ax[1], "b", "Training tokens")
ys = [dense[k][980.7e6] for k in BUDGETS]; ax[2].plot(toks, ys, "s-", color=C["dense"]); direct(ax[2], ys, "dense-1B", C["dense"], +0.022)
ys = [stats("b220-e512", k)["loss"] for k in BUDGETS]; ax[2].plot(toks, ys, "o-", color=C["store"]); direct(ax[2], ys, "220M + hash-512", C["store"], -0.026)
ys = [dense[k][405.6e6] for k in BUDGETS]; ax[2].plot(toks, ys, "s--", color=C["dense"]); direct(ax[2], ys, "dense-405M", C["dense"], +0.028)
ys = [stats("b100-e512", k)["loss"] for k in BUDGETS]; ax[2].plot(toks, ys, "o--", color=C["store"]); direct(ax[2], ys, "100M + hash-512", C["store"], +0.024)
ax[2].set_xscale("log", base=2); ax[2].set_xticks(BUDGETS); ax[2].set_xticklabels(["1B", "2B", "4B", "8B"]); ax[2].set_xlim(0.85, 40); ax[2].set_ylim(2.37, 2.9)
ax[2].set_xlabel("training tokens"); ptitle(ax[2], "c", "Larger backbones")
for a_ in ax: ygrid(a_)
fig.tight_layout(w_pad=0.8)
fig.savefig(os.path.join(OUT_F, "axes.pdf")); fig.savefig(os.path.join(OUT_F, "axes.png"), dpi=170); plt.close(fig)

# ----------------------------------------------------------------------------- Figure: capacity against residency (standalone)
fig, ax = plt.subplots(figsize=(5.4, 3.0))
A8, B8, c8, _ = FIT[8]
grid_n = np.logspace(np.log10(20e6), np.log10(1.4e9), 200)
ax.plot(grid_n, A8 + B8 * np.exp(-c8 * np.log(grid_n / gt.N0)), color="#9c9b96", lw=1.0, ls="--", label="fitted dense frontier")
ax.plot(sorted(dense[8]), [dense[8][n] for n in sorted(dense[8])], "o", color=C["dense"], ms=5, label="dense, fully resident (measured)")
for N, lab in [(27.7e6, "dense-27.7M"), (100.5e6, "dense-100M"), (219.5e6, "dense-220M"), (405.6e6, "dense-405M"), (980.7e6, "dense-1B")]:
    ax.text(N * 1.06, dense[8][N] + 0.014, lab, fontsize=6.5, color=C["dense"], zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.9))
for b in STORES + BACKS[1:]:
    s8 = stats(b, 8)
    ax.plot([s8["B"], s8["B"]], [dense[8][s8["B"]], s8["loss"]], color=C["store"], lw=1.0)
    ax.plot([s8["B"]], [s8["loss"]], "s", color=C["store"], ms=5)
    E_ = b.split("-e")[1]
    ax.text(s8["B"] * 0.94, s8["loss"], f"hash-{E_}", fontsize=6.5, color=C["store"], va="center", ha="right")
ax.plot([], [], "s", color=C["store"], label="ours: same resident backbone, growing NVMe store")
for b, dy in [("b27-e512", -0.012), ("b100-e512", -0.012), ("b220-e512", -0.012)]:
    s8 = stats(b, 8)
    ax.plot([s8["B"], s8["neq"]], [s8["loss"], s8["loss"]], color=C["store"], lw=0.7, ls=":")
    ax.plot([s8["neq"]], [s8["loss"]], "o", mfc="white", mec=C["store"], ms=5)
    ax.text(s8["neq"] * (0.92 if b == "b220-e512" else 1.0), s8["loss"] - 0.016, f"{s8['neq'] / s8['B']:.1f}$\\times$ fewer\nresident", fontsize=6.3, color=C["store"], va="top", ha="center", zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.9))
ax.plot([], [], "o", mfc="white", mec=C["store"], label="dense equivalent of the store")
ax.set_xscale("log"); ax.set_xlabel("resident backbone parameters (non-embedding)"); ax.set_ylabel("held-out loss (nats)"); ax.set_title("Capacity grows beyond the resident backbone", loc="left", fontweight="bold", pad=8); ax.text(1, 1.03, "$8\\times10^9$ training tokens", transform=ax.transAxes, ha="right", fontsize=7.5, color=GREY)
ax.set_xlim(17e6, 2.6e9); ax.set_ylim(2.33, 3.02); ax.legend(loc="upper right", fontsize=6.3)
ygrid(ax)
fig.tight_layout()
fig.savefig(os.path.join(OUT_F, "residency.pdf")); fig.savefig(os.path.join(OUT_F, "residency.png"), dpi=170); plt.close(fig)
F("residencyRatioTwoTwenty", f"{stats('b220-e512', 8)['neq'] / stats('b220-e512', 8)['B']:.1f}")
F("residencyRatioHundred", f"{stats('b100-e512', 8)['neq'] / stats('b100-e512', 8)['B']:.1f}")
F("residencyRatioTwentySeven", f"{stats('b27-e512', 8)['neq'] / stats('b27-e512', 8)['B']:.1f}")

# ----------------------------------------------------------------------------- Figure 3: serving (dense bf16 comparator; same-model ablation)
OFFLOAD = {"resident": 6.68, "prefetched int8": 7.58, "synchronous int8": 9.75}  # ms/step, 27.7M hash-512, attempt 3 (tables/offload.tex)
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.1))
labels = ["dense-1B\nbf16, resident", "220M + store\nNVMe int8"]
cols = [C["dense"], C["store"]]
vals = [im(B_, "peak_vram_bytes") / 1e9, im(S_, "peak_vram_bytes") / 1e9]
ax[0].bar([0, 1], vals, color=cols, width=0.6)
for i, v in enumerate(vals): ax[0].text(i, v + 0.05, f"{v:.2f} GB", ha="center", fontsize=6.5)
ax[0].text(0.5, max(vals) * 1.05, f"{vals[0] / vals[1]:.1f}$\\times$ less", ha="center", fontsize=7, color=C["store"])
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(labels, fontsize=6.5); ax[0].set_ylim(0, max(vals) * 1.25)
ptitle(ax[0], "a", "Peak GPU memory"); ax[0].set_ylabel("GB")
for i, arm in enumerate([B_, S_]):
    xs = [r[arm]["tok_s"] for r in reps]
    ax[1].scatter([i] * len(xs), xs, color=cols[i], s=16, zorder=3, edgecolors="white", linewidths=0.6)
    ax[1].hlines(st.mean(xs), i - 0.22, i + 0.22, color=cols[i], lw=2)
    ax[1].text(i + 0.27, st.mean(xs), f"{st.mean(xs):.0f}", va="center", fontsize=6.5)
ax[1].text(0.5, max(r[S_]["tok_s"] for r in reps) * 1.08, f"{st.mean(r[S_]['tok_s'] / r[B_]['tok_s'] for r in reps):.2f}$\\times$ faster, every node", ha="center", fontsize=7, color=C["store"])
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(labels, fontsize=6.5); ax[1].set_xlim(-0.6, 1.6); ax[1].set_ylabel("decode tok/s, batch 1")
ptitle(ax[1], "b", f"Throughput, {len(reps)} nodes"); ax[1].set_ylim(0, max(r[S_]["tok_s"] for r in reps) * 1.25)
names_ = list(OFFLOAD); vals_ = [OFFLOAD[n] for n in names_]
ax[2].bar(range(3), vals_, color=[C["dense"], C["store"], "#9c9b96"], width=0.6)
for i, v in enumerate(vals_): ax[2].text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=6.5)
ax[2].set_xticks(range(3)); ax[2].set_xticklabels(["experts\nresident", "NVMe,\nprefetched", "NVMe,\nsynchronous"], fontsize=6.5)
ax[2].set_ylabel("ms per decode step"); ax[2].set_ylim(0, max(vals_) * 1.25); ptitle(ax[2], "c", "Same model, 27.7M: overlap")
for a_ in ax: ygrid(a_)
fig.tight_layout(w_pad=1.0)
fig.savefig(os.path.join(OUT_F, "serving.pdf")); fig.savefig(os.path.join(OUT_F, "serving.png"), dpi=170); plt.close(fig)
F("offloadSyncMs", f"{OFFLOAD['synchronous int8']:.2f}"); F("offloadPrefMs", f"{OFFLOAD['prefetched int8']:.2f}"); F("offloadResMs", f"{OFFLOAD['resident']:.2f}")
F("offloadHiddenPct", f"{100 * (OFFLOAD['synchronous int8'] - OFFLOAD['prefetched int8']) / (OFFLOAD['synchronous int8'] - OFFLOAD['resident']):.0f}")
_s3 = J(os.path.join(SERVE, "offload_attempt3.json"))["configs"]
F("overlapMeasInt", f"{_s3['C_disk_int8']['tok_s'] / _s3['E_disk_int8_sync']['tok_s']:.2f}")          # int8 prefetched / int8 synchronous
F("overlapPredIntLo", f"{1 + _s3['C_disk_int8']['fetch_ms'] / _s3['A_resident']['step_ms']:.2f}")   # 1 + fetch/step with the prefetched arm's fetch
F("overlapPredIntHi", f"{1 + _s3['E_disk_int8_sync']['fetch_ms'] / _s3['A_resident']['step_ms']:.2f}")  # ... with the synchronous arm's fetch
F("overlapMeasBf", f"{_s3['B_disk_bf16']['tok_s'] / _s3['E_disk_int8_sync']['tok_s']:.2f}")
F("offloadratioint", f"{_s3['C_disk_int8']['tok_s'] / _s3['A_resident']['tok_s']:.3f}")
F("offloadratiobf", f"{_s3['B_disk_bf16']['tok_s'] / _s3['A_resident']['tok_s']:.3f}")

# ----------------------------------------------------------------------------- Figure 4: the two extensions
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.1), gridspec_kw={"width_ratios": [1.15, 1]})
kk = [k for k in BUDGETS if stats("b27-e512", k, 0, CONV)]
ax[0].axhline(0, color=C["store"], lw=1.2); ax[0].text(1.0, 0.0012, "token-id hash (fixed addresses)", color=C["store"], fontsize=6.5, va="bottom")
cv = [stats("b27-e512", k, 0, CONV)["loss"] - stats("b27-e512", k)["loss"] for k in kk]
ax[0].plot(kk, cv, "s", color=C["conv"], label="conventional router (address at the store layer)")
for a_, b_ in zip(kk[:-1], kk[1:]):
    ia, ib = kk.index(a_), kk.index(b_)
    ax[0].plot([a_, b_], [cv[ia], cv[ib]], "-" if b_ == 2 * a_ else "--", color=C["conv"], lw=1.4)
pv = [stats("b27-e512", k, 0, PREV)["loss"] - stats("b27-e512", k)["loss"] for k in BUDGETS]
ax[0].plot(BUDGETS, pv, "o-", color=C["prev"], label="previous-step router (address before layer 0)")
for k in BUDGETS:
    for suf, col in [(PREV, C["prev"]), (CONV, C["conv"])]:
        pass  # second seeds are reported in the seeds table, not drawn
ax[0].set_xscale("log", base=2); ax[0].set_xticks(BUDGETS); ax[0].set_xticklabels(["1B", "2B", "4B", "8B"])
ax[0].set_xlabel("training tokens"); ax[0].set_ylabel("loss $-$ hash-512 loss (nats)")
ax[0].set_ylim(-0.013, 0.046); ptitle(ax[0], "a", "Learned routing with lead time"); ax[0].legend(loc="upper right", fontsize=6)
cats = [("four experts,\nsummed (control)", LEVER_SUM, "#b3aee0"), ("four experts,\naveraged", LEVER_AVG, C["lever"])]
ax[1].axhline(0, color=C["store"], lw=1.2); ax[1].text(-0.45, 0.0006, "baseline: one expert per store layer", color=C["store"], fontsize=6.5, va="bottom")
for k_, dx_ in [(1, -0.13), (8, 0.13)]:
    if stats("b27-e512", k_, 0, LEVER_SUM) and stats("b27-e512", k_, 0, LEVER_AVG):
        ax[1].plot([dx_, 1 + dx_], [stats("b27-e512", k_, 0, LEVER_SUM)["loss"] - stats("b27-e512", k_)["loss"], stats("b27-e512", k_, 0, LEVER_AVG)["loss"] - stats("b27-e512", k_)["loss"]], "-", color=C["lever"], lw=0.8, zorder=1)
for i, (lab, suf, col) in enumerate(cats):
    for k, dx, mk in [(1, -0.13, "o"), (8, 0.13, "D")]:
        b0 = stats("b27-e512", k)
        s = stats("b27-e512", k, 0, suf)
        if s:
            d = s["loss"] - b0["loss"]
            ax[1].plot([i + dx], [d], mk, color=col, ms=7 if mk == "o" else 6)
            ax[1].text(i + dx, d + (0.0009 if d > 0 else -0.0009), f"{d:+.3f}", ha="center", va="bottom" if d > 0 else "top", fontsize=6)
            pass  # second seeds are reported in the seeds table, not drawn
if stats("b27-e512", 8, 0, LEVER_AVG):
    ax[1].plot([], [], "o", color="#52514e", label="1B tokens"); ax[1].plot([], [], "D", color="#52514e", ms=5, label="8B tokens")
ax[1].set_xticks(range(len(cats))); ax[1].set_xticklabels([c[0] for c in cats], fontsize=6.5); ax[1].set_xlim(-0.6, len(cats) - 0.4)
ax[1].set_ylim(-0.016, 0.0085); ax[1].set_ylabel("loss $-$ baseline (nats)"); ptitle(ax[1], "b", "Multi-expert combination")
if stats("b27-e512", 8, 0, LEVER_AVG):
    ax[1].legend(loc="upper right", fontsize=6)
for a_ in ax: ygrid(a_)
fig.tight_layout(w_pad=1.2)
fig.savefig(os.path.join(OUT_F, "extensions.pdf")); fig.savefig(os.path.join(OUT_F, "extensions.png"), dpi=170); plt.close(fig)

# ----------------------------------------------------------------------------- Figure: what the frontier looks like (section 7)
W = 6.6e9            # measured sustained read bandwidth of the NVMe drive used in this paper, bytes/s
W_PCIE5 = 14e9       # PCIe 5.0 consumer drive class (spec sheets: 14-15 GB/s sequential read)
V_TARGET = 20.0      # tok/s
NB_FRONTIER = 16e9   # int8 resident backbone
POOL = 1e12          # one 1 TB consumer drive at int8: 1T stored parameters
NS = 8               # store layers: every fifth layer of a 40-layer backbone (layers 5, 10, ..., 40)
E_LAYER = 8192       # blocks per store layer
D_FRONTIER, L_FRONTIER = 5120, 40  # width and depth assumed for a 16B backbone (embedding table and KV cache)
D_TRAIN = 14.8e12    # DeepSeek-V3's token budget, used for the exposure comparison
USD_PER_GB = 0.09
p_block = POOL / (NS * E_LAYER)
step_s = 1 / V_TARGET
r_lo, r_hi = min(stats(b, 8)["r"] for b in BACKS), max(stats(b, 8)["r"] for b in BACKS)
_grid = [(stats(b, 8)["P"] / stats(b, 8)["B"], stats(b, 8)["r"]) for b in STORES]
_slope = (_grid[-1][1] - _grid[0][1]) / np.log2(_grid[-1][0] / _grid[0][0])
def r_trend(x): return np.where(x <= _grid[-1][0], np.interp(np.log2(x), [np.log2(g[0]) for g in _grid], [g[1] for g in _grid]), _grid[-1][1] + _slope * np.log2(x / _grid[-1][0]))
def neq(P, r): return NB_FRONTIER * (P / NB_FRONTIER) ** r
R_200 = np.log(200e9 / NB_FRONTIER) / np.log(POOL / NB_FRONTIER)
r_avg8, r_base8 = stats("b27-e512", 8, 0, LEVER_AVG)["r"], stats("b27-e512", 8)["r"]
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.0), gridspec_kw={"width_ratios": [1.15, 1]})
rr = np.linspace(0.40, 0.90, 200)
ax[0].plot(rr, neq(POOL, rr) / 1e9, color=C["dense"], lw=1.4)
ax[0].axvspan(r_lo, r_hi, color="#e3f0fb", zorder=0); ax[0].text((r_lo + r_hi) / 2, 8, "token-id store,\nthis paper", ha="center", fontsize=6.3, color="#2a5d9c")
fr = [(0.51, "DeepSeekMoE-16B", 1.6, "right", -0.006), (0.59, "DeepSeekMoE-145B", 0.55, "center", 0.0), (0.65, "Ling-mini", 1.6, "right", -0.006), (0.67, "Qwen3-30B-A3B", 0.55, "left", 0.012), (0.83, "DeepSeek-V3", 0.6, "right", -0.008)]
for i_, (r_, lab, fy, ha, dx) in enumerate(fr):
    y = neq(POOL, r_) / 1e9
    ax[0].plot([r_], [y], "o", color=C["prev"], ms=4.5, mec="white", mew=0.6, label="frontier MoEs at their reported rates" if i_ == 0 else None)
    ax[0].text(r_ + dx, y * fy, lab, fontsize=5.8, color=C["prev"], ha=ha, zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.9))
ax[0].legend(loc="upper left", fontsize=6)
ax[0].set_yscale("log"); ax[0].set_ylim(6, 3000); ax[0].set_xlim(0.40, 0.92)
ax[0].set_xlabel("exchange rate $r$ (ours measured; frontier MoEs reported)"); ax[0].set_ylabel("dense-equivalent params (B)")
ptitle(ax[0], "a", "16B resident, 1T stored on one 1 TB drive")
for y_, lab in [(100, "100B"), (500, "500B")]:
    ax[0].axhline(y_, color="#9c9b96", lw=0.6, ls=":"); ax[0].text(0.405, y_ * 1.12, lab, fontsize=6, color="#9c9b96")
E = np.logspace(np.log10(256), np.log10(262144), 200)
qmax = W / V_TARGET
ax[1].fill_between(E, 1e-3, qmax / 1e6, color="#e3f0fb", zorder=0)
for k_, col, lab in [(1, C["store"], "1 block per store layer"), (4, C["lever"], "4 blocks per store layer")]:
    ax[1].plot(E, POOL * k_ / E / 1e6, color=col, label=lab)
ax[1].axhline(qmax / 1e6, color=C["dense"], ls=":", lw=0.9); ax[1].text(2.3e5, qmax / 1e6 * 0.72, f"measured drive, {W/1e9:.1f} GB/s", fontsize=5.8, color=C["dense"], ha="right", va="top", zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.9))
ax[1].axhline(W_PCIE5 / V_TARGET / 1e6, color=C["dense"], ls="--", lw=0.9); ax[1].text(2.3e5, W_PCIE5 / V_TARGET / 1e6 * 1.18, f"PCIe 5.0 drive, {W_PCIE5/1e9:.0f} GB/s", fontsize=5.8, color=C["dense"], ha="right", va="bottom", zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.9))
for E_, k_, col, txt, ha, fy in [(512, 1, C["store"], "512 (this paper)", "left", 1.45), (E_LAYER, 1, C["store"], f"{E_LAYER:,}: {POOL/E_LAYER/1e6:.0f} MB", "right", 0.6), (E_LAYER, 4, C["lever"], f"{4*POOL/E_LAYER/1e6:.0f} MB", "right", 1.55), (131072, 1, C["store"], "131k: 7.6 MB", "left", 1.45)]:
    q = POOL * k_ / E_ / 1e6
    ax[1].plot([E_], [q], "o", color=col, ms=4.5, mec="white", mew=0.7, zorder=5); ax[1].text(E_ * (1.3 if ha == "left" else 0.78), q * fy, txt, fontsize=6, ha=ha, color=col, zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.9))
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_ylim(2, 2e4); ax[1].set_xlabel("blocks per store layer, $E$ (8 store layers)"); ax[1].set_ylabel("MB per token")
ptitle(ax[1], "b", "Traffic for the 1 TB store (int8, 20 tok/s)"); ax[1].legend(loc="lower left", fontsize=6, handlelength=1.6, borderaxespad=0.4)
for a_ in ax: ygrid(a_)
fig.tight_layout(w_pad=1.2)
fig.savefig(os.path.join(OUT_F, "potential.pdf")); fig.savefig(os.path.join(OUT_F, "potential.png"), dpi=170); plt.close(fig)
# --- macros
GB = 1e9
F("envPoolT", f"{POOL/1e12:.0f}"); F("envDriveTB", f"{POOL/1e12:.0f}"); F("envDriveUSD", f"{POOL/1e9*USD_PER_GB:.0f}")
F("envExpansion", f"{POOL / NB_FRONTIER:g}"); F("envExpansionTrained", f"{_grid[-1][0]:.0f}")
F("envNS", f"{NS}"); F("envEperLayer", f"{E_LAYER:,}"); F("envEperLayerK", f"{E_LAYER//1024}k"); F("envBlockM", f"{p_block/1e6:.0f}")
F("envMBKOne", f"{NS * p_block / 1e6:.0f}"); F("envMBKFour", f"{4 * NS * p_block / 1e6:.0f}")
F("envGBsKOne", f"{NS * p_block * V_TARGET / 1e9:.1f}"); F("envGBsKFour", f"{4 * NS * p_block * V_TARGET / 1e9:.1f}")
# first-layer deadline: the first eighth of the payload is due at 5 ms of a 50 ms token, i.e. 1.25x the average rate
F("envGBsDeadlineKOne", f"{(NS * p_block / 8) / ((5 - 1) / L_FRONTIER * step_s) / 1e9:.1f}")    # store layer 5 runs at 1.25*(5-1) = 5 ms
F("envGBsDeadlineKFour", f"{(4 * NS * p_block / 8) / ((5 - 1) / L_FRONTIER * step_s) / 1e9:.1f}")
F("envTokSKFourMeasured", f"{W / (4 * NS * p_block):.1f}"); F("envTokSKTwoMeasured", f"{W / (2 * NS * p_block):.0f}")
F("envFirstDeadlineMs", f"{1000 * step_s * 5 / L_FRONTIER:.1f}"); F("envStepMs", f"{1000*step_s:.0f}")
F("envQKOne", f"{D_TRAIN * NS / POOL:.0f}"); F("envQKFour", f"{D_TRAIN * 4 * NS / POOL:.0f}")
F("envQOursKOne", f"{stats('b27-e512', 8)['D'] * 2 / stats('b27-e512', 8)['added']:.0f}" if 'D' in stats('b27-e512', 8) else f"{8e9 * 2 / (stats('b27-e512', 8)['P'] - stats('b27-e512', 8)['B']):.0f}")
F("envQOursKFour", f"{8e9 * 8 / (stats('b27-e512', 8)['P'] - stats('b27-e512', 8)['B']):.0f}")
F("envActivationKOne", f"{100 / E_LAYER:.4f}"); F("envActivationKFour", f"{100 * 4 / E_LAYER:.3f}"); F("envActivationVThree", f"{100 * 8 / 256:.1f}")
F("envEmin", f"{POOL * V_TARGET / W:,.0f}")
F("envTrainFlops", f"{6 * (NB_FRONTIER + 4 * NS * p_block) * D_TRAIN / 1e24:.1f}"); F("envActiveKFourB", f"{(NB_FRONTIER + 4 * NS * p_block)/1e9:.1f}")
F("envBackboneGBs", f"{NB_FRONTIER * V_TARGET / 1e9:.0f}")
F("envRTrend", f"{float(r_trend(POOL / NB_FRONTIER)):.2f}"); F("envRSlope", f"{-_slope:.3f}")
F("envNeqTrend", f"{neq(POOL, r_trend(POOL / NB_FRONTIER)) / 1e9:.0f}")
for tag, r_ in [("Lo", r_lo), ("Hi", r_hi), ("FrLo", 0.51), ("FrHi", 0.67), ("VThree", 0.83)]:
    F(f"envNeq{tag}", f"{neq(POOL, r_) / 1e9:.0f}")
for tag, P_ in [("OneT", 1e12), ("TwoT", 2e12)]:
    F(f"envNeq{tag}Lo", f"{neq(P_, r_lo)/1e9:.0f}"); F(f"envNeq{tag}Hi", f"{neq(P_, r_hi)/1e9:.0f}"); F(f"envNeq{tag}Trend", f"{neq(P_, r_trend(P_/NB_FRONTIER))/1e9:.0f}")
F("envDenseEqGPUs", f"{int(np.ceil(neq(POOL, r_lo) / 24e9))}"); F("envNeqRound", f"{50 * round(neq(POOL, r_lo) / 50e9):.0f}")
F("envRTwoHundred", f"{R_200:.2f}"); F("envNeqAvgRate", f"{neq(POOL, r_avg8)/1e9:.0f}"); F("envNeqConvRate", f"{neq(POOL, stats('b27-e512', 8, 0, ROUTER_CONV)['r'] if False else float(FK('rConvEight')))/1e9:.0f}")
F("envNeqLing", f"{neq(POOL, 0.65)/1e9:.0f}"); F("envNeqQwen", f"{neq(POOL, 0.67)/1e9:.0f}"); F("envNeqTrendRound", f"{neq(POOL, r_trend(POOL / NB_FRONTIER)) / 1e9:.0f}")
CTX, KV_HEADS, HEAD = 8192, 8, 128
F("envEmbGB", f"{128000 * D_FRONTIER / GB:.1f}"); F("envBackboneGB", f"{NB_FRONTIER / GB:.0f}")
F("envKV", f"{CTX * L_FRONTIER * KV_HEADS * HEAD * 2 * 2 / GB:.1f}")
for k_, ktag in [(1, ""), (4, "KFour")]:
    buf_ = 2 * k_ * NS * p_block          # int8 blocks in flight, double-buffered across the token
    deq_ = k_ * NS * p_block * 2          # bf16 working copies of the blocks a token executes
    F(f"envBuf{ktag}MB", f"{buf_ / 1e6:.0f}"); F(f"envDeq{ktag}MB", f"{deq_ / 1e6:.0f}")
    for ctx_, tag in [(2048, "TwoK"), (8192, "EightK"), (32768, "ThirtyTwoK")]:
        kvc = ctx_ * L_FRONTIER * KV_HEADS * HEAD * 2 * 2
        F(f"envResident{ktag}{tag}", f"{(NB_FRONTIER + 128000 * D_FRONTIER + kvc + buf_ + deq_ + 0.5 * GB) / GB:.1f}")
F("envResident", FK("envResidentEightK") if False else f"{(NB_FRONTIER + 128000 * D_FRONTIER + CTX * L_FRONTIER * KV_HEADS * HEAD * 4 + 2 * NS * p_block + NS * p_block * 2 + 0.5 * GB) / GB:.1f}")
F("rGridLo", f"{r_lo:.2f}"); F("rGridHi", f"{r_hi:.2f}")
F("rSixtyFourEight", f"{stats('b27-e64', 8)['r']:.2f}"); F("rSixtyFourOne", f"{stats('b27-e64', 1)['r']:.2f}")
F("envExpansionSixtyFour", f"{_grid[0][0]:.0f}")
F("vThreeExpertsPerLayer", "256")

# ----------------------------------------------------------------------------- same-model offload study (Appendix E)
# attempt 3 is the reported, same-node comparison; attempt 4 (faster node) supplies one ratio in the text
_a3 = J(os.path.join(SERVE, "offload_attempt3.json"))
_a4 = J(os.path.join(SERVE, "offload_attempt4.json"))["configs"]
F("offloadratiofast", f"{_a4['B_disk_bf16']['tok_s'] / _a4['A_resident']['tok_s']:.3f}")
with open(os.path.join(OUT_T, "offload.tex"), "w") as f:
    c = _a3["configs"]
    f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
    f.write("configuration (hash-512, 27.7M backbone) & tok/s & step (ms) & fetch (ms) & logit $\\Delta$ \\\\\n\\midrule\n")
    f.write(f"experts resident in GPU memory & {c['A_resident']['tok_s']:.1f} & {c['A_resident']['step_ms']:.2f} & -- & -- \\\\\n")
    f.write(f"NVMe bf16, prefetched & {c['B_disk_bf16']['tok_s']:.1f} & {c['B_disk_bf16']['step_ms']:.2f} & {c['B_disk_bf16']['fetch_ms']:.2f} & {_a3['logit_max_diff_bf16']:.1f} (bitwise) \\\\\n")
    f.write(f"NVMe int8, prefetched & {c['C_disk_int8']['tok_s']:.1f} & {c['C_disk_int8']['step_ms']:.2f} & {c['C_disk_int8']['fetch_ms']:.2f} & {_a3['logit_max_diff_int8']:.3f} \\\\\n")
    f.write(f"NVMe int8, synchronous (ablation) & {c['E_disk_int8_sync']['tok_s']:.1f} & {c['E_disk_int8_sync']['step_ms']:.2f} & {c['E_disk_int8_sync']['fetch_ms']:.2f} & -- \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ----------------------------------------------------------------------------- published frontier MoEs (Appendix B)
FRONTIER_MOE = [  # name, active B, total B, matched dense model B, training tokens, same training data?
    ("DeepSeekMoE-145B", 22.2, 144.6, 67.0, 245e9, True),
    ("Ling-mini-beta", 0.85, 17.5, 6.1, 1e12, True),
    ("DeepSeekMoE-16B", 2.8, 16.4, 6.9, 2e12, True),
    ("Qwen3-30B-A3B", 3.3, 30.5, 14.8, 36e12, True),
    ("DeepSeek-V3", 37.0, 671.0, 405.0, 14.8e12, False),
]
with open(os.path.join(OUT_T, "frontier.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrrrrl}\n\\toprule\nmodel & active & total & matched dense model & training tokens & $r$ & comparison \\\\\n\\midrule\n")
    for n_, a_, t_, c_, T_, same_ in FRONTIER_MOE:
        r_ = math.log(c_ / a_) / math.log(t_ / a_)
        f.write(f"{n_} & {a_:g}B & {t_:g}B & {c_:g}B & {T_ / 1e12:g}T & {r_:.2f} & {'same training data' if same_ else 'different data (benchmark parity)'} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ----------------------------------------------------------------------------- page-breaking versions of the two long appendix tables
for stem, caption, label in [
    ("seeds", "Configurations with a second initialization seed. Cross-platform repeats are not independent samples.", "tab:seeds"),
    ("tasksall", r"Zero-shot accuracy (\%) for scored checkpoints; standard errors are $0.5$--$1.2$ percentage points.", "tab:tasksall"),
]:
    src_ = open(os.path.join(OUT_T, f"{stem}.tex")).read()
    header, body = src_.split(r"\midrule", 1)
    header = header.replace("tabular", "longtable", 1)
    columns = header[header.index(r"\toprule"):]
    text = (header + "\\midrule\n\\endfirsthead\n" + columns + "\\midrule\n\\endhead\n\\midrule\n\\endfoot\n\\bottomrule\n"
            + r"\caption{" + caption + r"}\label{" + label + "}\\\\\n\\endlastfoot\n"
            + body.replace(r"\bottomrule", "").replace(r"\end{tabular}", r"\end{longtable}"))
    open(os.path.join(OUT_T, f"{stem}-long.tex"), "w").write("% Generated by analysis/make_assets.py.\n" + text)

# ----------------------------------------------------------------------------- write facts2.tex
with open(os.path.join(OUT_T, "facts2.tex"), "w") as f:
    for k, v in facts.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote {len(facts)} macros; seeds {len(seed_rows)}; serving clean reps {len(clean)}")
for k in ("perDoubling1", "perDoubling8", "gapTwoTwenty8", "rMin", "rMax", "HGridEight", "lawHLo", "lawHHi", "prevRetained", "deltaAvg1", "deltaAvg8", "isoSpeedNew", "isoVramFpNew", "envEmin", "envResident", "envNeqLo", "envNeqHi"):
    print(k, FK(k))
