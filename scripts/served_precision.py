"""Held-out loss of the two serving-comparison models through the weights that are actually served (Appendix E).

dense-1B: fp32 checkpoint and the same weights cast to bf16.
220M + store: fp32 checkpoint; experts round-tripped through the int8 store quantiser (per-column absmax, fp16
scales, the ExpertStore format); and that model cast to bf16. Same evaluation batches as every training result.

    python scripts/served_precision.py --dense-ckpt ckpt_tk-dense1B-1e9.pt --store-ckpt ckpt_tk-b220-e512-1e9-sn.pt \
        --root data/v3 --out served_precision.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch  # noqa: E402

from ckpt_load import load_ckpt_model, rows_fn  # noqa: E402
from evaluation import EvalSuite, default_splits  # noqa: E402
from serving import _dequant, _quant  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dense-ckpt", required=True)
ap.add_argument("--store-ckpt", required=True)
ap.add_argument("--root", default="data/v3")
ap.add_argument("--out", required=True)
a = ap.parse_args()
DEV = "cuda"
evals = EvalSuite(default_splits(a.root))
out = {"host": os.uname().nodename, "gpu": torch.cuda.get_device_name(0)}


def ev(model, rows):
    t0 = time.time()
    r = evals.evaluate(model, forward_fn=(lambda m, x, y: m(x, y, rows(x))) if rows else None, device=DEV)
    return {"main_heldout": r["main_heldout"]["loss"], "xd_lang": r["xd_lang"]["loss"], "xd_code": r["xd_code"]["loss"], "eval_s": time.time() - t0}


ck = torch.load(a.dense_ckpt, map_location="cpu", weights_only=False)
m, _, _, _ = load_ckpt_model(ck, DEV, name=os.path.basename(a.dense_ckpt))
out["dense1B_fp32"] = ev(m, None)
out["dense1B_bf16"] = ev(m.to(torch.bfloat16), None)
del m, ck
torch.cuda.empty_cache()

ck2 = torch.load(a.store_ckpt, map_location="cpu", weights_only=False)
m2, assign, layers, _ = load_ckpt_model(ck2, DEV, name=os.path.basename(a.store_ckpt))
rows = rows_fn(assign, layers)
out["store_fp32"] = ev(m2, rows)
with torch.no_grad():  # every (layer, expert, matrix) through the int8 store format, exactly as ExpertStore does
    for L in layers:
        bank = m2.layers[L].memory.bank
        for nm in ("gate", "up", "down"):
            W = getattr(bank, nm)
            for e in range(W.shape[0]):
                w = W[e].detach().float().cpu()
                pl, sc = _quant(w, "int8")
                W[e].copy_(_dequant(pl, sc, "int8", w.shape[0], w.shape[1]).to(W.device))
out["store_int8experts_fp32"] = ev(m2, rows)
out["store_int8experts_bf16"] = ev(m2.to(torch.bfloat16), rows)
json.dump(out, open(a.out, "w"), indent=2)
print(json.dumps(out, indent=2))
