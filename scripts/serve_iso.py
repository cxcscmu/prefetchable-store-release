"""Equal-quality serving comparison (Section 5.2, Figure 5, Tables 10-11): the dense-1B reference fully resident on
the GPU versus the 220M backbone with its hash-512 store served from an int8 NVMe store file with one-token-ahead
prefetch. Both are decoded at batch 1 with the same DecodeSession harness; the store's experts are the only weights
not resident.

    python scripts/serve_iso.py --dense-ckpt ckpt_tk-dense1B-1e9.pt --store-ckpt ckpt_tk-b220-e512-1e9-sn.pt \
        --store-dir /nvme/tmp --out iso_rep1.json                       # fp32 compute (Table 10 repeats)
    python scripts/serve_iso.py ... --dtype bf16 --out iso_bf16.json    # store model computes in bf16

Dense-1B is always measured both in fp32 and in bf16. The paper reports four --dtype bf16 repeats (main text)
and four --dtype fp32 repeats (appendix), each a separate job on an L40S node.
"""

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch  # noqa: E402

from serve_bench import CacheProvider, PowerMeter, banks_of, decode_run, load_model  # noqa: E402
from serving import ExpertStore, Prefetcher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dense-ckpt", required=True)
ap.add_argument("--store-ckpt", required=True)
ap.add_argument("--store-dir", required=True, help="directory on the NVMe drive for the temporary store file")
ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"], help="compute dtype of the store model (dense-1B is measured in both)")
ap.add_argument("--steps", type=int, default=1024)
ap.add_argument("--out", required=True)
a = ap.parse_args()
DEV = "cuda"


def loss_of(ck_path):
    """held-out loss from the result_<run>.json written next to the checkpoint by train.py, if present"""
    r = os.path.join(os.path.dirname(ck_path), os.path.basename(ck_path).replace("ckpt_", "result_").replace(".pt", ".json"))
    try:
        return json.load(open(r))["final_eval_on"]["main_heldout"]["loss"]
    except (OSError, KeyError, ValueError):
        return None


def dense_row(m, loss):
    torch.cuda.reset_peak_memory_stats()
    with PowerMeter() as pm:
        toks, step_ms, _, _ = decode_run(m, {}, [], a.steps, DEV)
    return {"tok_s": toks, "step_ms": step_ms, "watts": pm.watts(), "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            "resident_params": sum(p.numel() for p in m.parameters()), "store_bytes": 0, "loss": loss}


out = {"host": os.uname().nodename, "gpu": torch.cuda.get_device_name(0)}

# dense-1B, fully resident, measured in fp32 and then in bf16
ck = torch.load(a.dense_ckpt, map_location="cpu", weights_only=False)
m, _, _ = load_model(ck, DEV)
out["dense1B_fp32"] = dense_row(m, loss_of(a.dense_ckpt))
out["dense1B_bf16"] = dense_row(m.to(torch.bfloat16), loss_of(a.dense_ckpt))
print(json.dumps({k: out[k] for k in ("dense1B_fp32", "dense1B_bf16")}), flush=True)
del m, ck
torch.cuda.empty_cache()

# 220M backbone + hash-512 store, experts int8 on NVMe, prefetched
ck2 = torch.load(a.store_ckpt, map_location="cpu", weights_only=False)
m2, assign, arch = load_model(ck2, DEV, name=os.path.basename(a.store_ckpt))
layers = arch["moe_layers"]
banks = banks_of(m2, layers)
if a.dtype == "bf16":
    m2 = m2.to(torch.bfloat16)
sd = os.path.join(a.store_dir, "store_" + os.path.basename(a.store_ckpt).replace(".pt", ""))
os.makedirs(sd, exist_ok=True)
sp = sd + "/store_int8.bin"
t0 = time.time()
ExpertStore.write(sp, banks, "int8")
out["store_write_s"] = time.time() - t0
for L in layers:
    m2.layers[L].memory.bank.to("cpu")  # the experts leave the GPU; only the store file serves them
torch.cuda.empty_cache()
st = ExpertStore(sp).open()
pf = Prefetcher(st, prefetch=True, device=DEV)
prov = CacheProvider(DEV, pf)
torch.cuda.reset_peak_memory_stats()
with PowerMeter() as pm:
    toks, step_ms, _, fetch_ms = decode_run(m2, assign, layers, a.steps, DEV, provider=prov, prefetcher=pf)
out["b220_hash512_disk_int8"] = {
    "tok_s": toks, "step_ms": step_ms, "fetch_ms": fetch_ms, "watts": pm.watts(),
    "peak_vram_bytes": torch.cuda.max_memory_allocated(), "o_direct": st.direct,
    "resident_params": sum(p.numel() for n, p in m2.named_parameters() if "bank" not in n),
    "store_bytes": os.path.getsize(sp), "loss": loss_of(a.store_ckpt), "store_ckpt": os.path.basename(a.store_ckpt),
    "moe_layers": list(layers), "compute_dtype": a.dtype, "disk_dtype": "int8",
}
pf.stop()
st.close()
print(json.dumps({"b220_hash512_disk_int8": out["b220_hash512_disk_int8"]}), flush=True)
shutil.rmtree(sd, ignore_errors=True)
json.dump(out, open(a.out, "w"), indent=2)
