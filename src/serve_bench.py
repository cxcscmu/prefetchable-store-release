"""Same-model offload study (Appendix E, Table 12): decode the 27.7M + hash-512 model at batch 1 with its
experts resident in GPU memory, or streamed from an NVMe store file with one-token-ahead prefetch, or read
synchronously (the ablation that exposes the fetch).

    python src/serve_bench.py --ckpt ckpt_tk-b27-e512-1e9-sn.pt --store-dir /nvme/store --out serve_offload.json

Configurations (greedy decode, batch 1, --steps tokens after a warm-up):
  A_resident        experts in GPU memory
  B_disk_bf16       bf16 store on NVMe, prefetched      (logits must equal A bitwise)
  C_disk_int8       int8 store on NVMe, prefetched
  D_disk_int4       int4 store on NVMe, prefetched      (not reported in the paper)
  E_disk_int8_sync  int8 store, synchronous fetch

Also records a cold prefill: streaming every expert a 2048-token prompt touches versus the resident forward.
The other serving scripts (scripts/serve_iso.py, scripts/served_precision.py) import the pieces defined here.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import threading
import time

import torch

from serving import DecodeSession, ExpertStore, Prefetcher

L = None  # moe layers, from ckpt


def load_model(ck, dev, name=None):
    """Rebuild a train.py checkpoint -> (model, assign, arch)."""
    from ckpt_load import load_ckpt_model

    m, assign, layers, info = load_ckpt_model(ck, dev, name=name)
    return m, (assign or {}), ck["arch"]


def banks_of(m, layers):
    return {
        l: {nm: getattr(m.layers[l].memory.bank, nm).detach().cpu().clone() for nm in ("gate", "up", "down")}
        for l in layers
    }


class PowerMeter:
    def __init__(self):
        self.samples, self._stop = [], False
        self.t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.samples.append(float(out.stdout.strip().splitlines()[0]))
            except Exception:
                pass
            time.sleep(0.25)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self._stop = True
        self.t.join(timeout=2)

    def watts(self):
        return st.mean(self.samples) if self.samples else float("nan")


def decode_run(m, assign, layers, n_steps, dev, provider=None, prefetcher=None, warm=32, seed_tok=1):
    """Batch-1 greedy decode. Returns (tok_per_s, step_ms, logits_tail, fetch_ms)."""
    ses = DecodeSession(m, rows_fn=lambda xt: {l: assign[l][xt] for l in layers}, expert_provider=provider)
    x = torch.full((1, 1), seed_tok, dtype=torch.long, device=dev)
    fetch_times, tail = [], []
    if prefetcher is not None:
        prefetcher.request([(l, int(assign[l][x][0, 0])) for l in layers])
    t0 = None
    for i in range(n_steps + warm):
        if i == warm:
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
        if provider is not None:
            provider.arm()  # lazy take happens INSIDE the step,
        logits = ses.step(x)  # overlapping the pre-memory layers
        nxt = logits.argmax(-1, keepdim=True)
        if prefetcher is not None:
            fetch_times.append(prefetcher.last_fetch_s)
            prefetcher.request([(l, int(assign[l][nxt][0, 0])) for l in layers])
        x = nxt
        if i >= n_steps + warm - 5:
            tail.append(logits.float().cpu())
    if dev == "cuda":
        torch.cuda.synchronize()
    el = time.time() - t0
    return (
        n_steps / el,
        el / n_steps * 1e3,
        torch.stack(tail),
        st.mean(fetch_times[warm:]) * 1e3 if prefetcher is not None else None,
    )


class CacheProvider:
    """LAZY expert_provider: takes from the prefetcher on FIRST use inside a step, so the
    fetch overlaps the step's pre-memory layers. Attempt 1 took eagerly at the loop top --
    zero overlap, prefetch degenerated to sync (B/E measured 1.006). This is the fix."""

    def __init__(self, dev, prefetcher):
        self.dev, self.pf = dev, prefetcher
        self.cache, self._taken = {}, True

    def arm(self):
        self._taken = False

    def __call__(self, layer, eids):
        if not self._taken:
            self.cache = self.pf.take()
            self._taken = True
        gs = [self.cache[(layer, int(e))] for e in eids.tolist()]
        return (
            torch.stack([g[0] for g in gs]),
            torch.stack([g[1] for g in gs]),
            torch.stack([g[2] for g in gs]),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--out", default="serve_results.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    m, assign, arch = load_model(ck, dev)
    layers = arch["moe_layers"]
    banks = banks_of(m, layers)
    os.makedirs(a.store_dir, exist_ok=True)
    stores = {}
    for dt in ("bf16", "int8", "int4"):
        p = os.path.join(a.store_dir, f"store_{dt}.bin")
        if not os.path.exists(p + ".meta.json"):
            t0 = time.time()
            ExpertStore.write(p, banks, dt)
            print(
                json.dumps(
                    {"store_written": dt, "bytes": os.path.getsize(p), "sec": round(time.time() - t0, 1)}
                ),
                flush=True,
            )
        stores[dt] = p

    res = {"host": os.uname().nodename, "device": dev, "n_experts": arch["n_experts"], "configs": {}}

    # A0 (reference only): the training-path tiered-bmm compute. Attempt 2 showed this
    # carries heavy per-step overhead at batch 1, which HANDICAPPED the baseline and made
    # disk beat RAM (129%) -- a fairness confound, not a storage result.
    with PowerMeter() as pm:
        toks, step_ms, _, _ = decode_run(m, assign, layers, min(a.steps, 256), dev)
    res["configs"]["A0_grouped_ref"] = {"tok_s": toks, "step_ms": step_ms, "watts": pm.watts()}
    print(json.dumps({"A0_grouped_ref": res["configs"]["A0_grouped_ref"]}), flush=True)

    # A: FAIR resident baseline -- the IDENTICAL provider compute path as every disk
    # config, weights pre-resident on the device. The only variable left is storage.
    class RamProvider:
        def __init__(self, banks, dev):
            self.t = {l: {nm: b[nm].to(dev) for nm in ("gate", "up", "down")} for l, b in banks.items()}

        def arm(self):
            pass

        def __call__(self, layer, eids):
            b = self.t[layer]
            return (b["gate"][eids], b["up"][eids], b["down"][eids])

    ram = RamProvider(banks, dev)
    with PowerMeter() as pm:
        toks, step_ms, tail_A, _ = decode_run(m, assign, layers, a.steps, dev, provider=ram)
    res["configs"]["A_resident"] = {"tok_s": toks, "step_ms": step_ms, "watts": pm.watts()}
    print(json.dumps({"A_resident_FAIR": res["configs"]["A_resident"]}), flush=True)

    tails = {}
    for name, dt, prefetch in (
        ("B_disk_bf16", "bf16", True),
        ("C_disk_int8", "int8", True),
        ("D_disk_int4", "int4", True),
        ("E_disk_int8_sync", "int8", False),
    ):
        stt = ExpertStore(stores[dt]).open()
        pf = Prefetcher(stt, prefetch=prefetch, device=dev)
        prov = CacheProvider(dev, pf)
        with PowerMeter() as pm:
            toks, step_ms, tail, fetch_ms = decode_run(
                m, assign, layers, a.steps, dev, provider=prov, prefetcher=pf
            )
        res["configs"][name] = {
            "tok_s": toks,
            "step_ms": step_ms,
            "fetch_ms": fetch_ms,
            "watts": pm.watts(),
            "o_direct": stt.direct,
        }
        tails[name] = tail
        pf.stop()
        stt.close()
        print(json.dumps({name: res["configs"][name]}), flush=True)

    # bf16 store: logits must equal the resident model's (same dtype end to end)
    eq = bool(torch.allclose(tail_A, tails["B_disk_bf16"], atol=1e-5))
    res["logits_equal_bf16"] = eq
    res["logit_max_diff_bf16"] = float((tail_A - tails["B_disk_bf16"]).abs().max())
    res["logit_max_diff_int8"] = float((tail_A - tails["C_disk_int8"]).abs().max())

    # prefill: cold stream of all touched experts vs resident forward, T=2048
    x = torch.randint(0, arch["vocab"], (1, 2048), device=dev)
    rbl = {l: assign[l][x] for l in layers}
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        m(x, None, rbl)
    if dev == "cuda":
        torch.cuda.synchronize()
    res["prefill_resident_s"] = time.time() - t0
    stt = ExpertStore(stores["bf16"]).open()
    touched = sorted({(l, int(e)) for l in layers for e in torch.unique(rbl[l]).tolist()})
    t0 = time.time()
    for l, e in touched:
        stt.read_expert(l, e)
    res["prefill_cold_stream_s"] = time.time() - t0
    res["prefill_touched_experts"] = len(touched)
    stt.close()

    if dev == "cuda":
        res["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
    json.dump(res, open(a.out, "w"), indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
