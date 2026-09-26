# The Prefetchable Store

Code and results for *The Prefetchable Store: Decoupling Model Capacity from Fast Memory*.

A prefetchable store keeps most of a model's trained parameters on a commodity SSD as expert blocks addressed by
token id. Because every address is known before the first layer runs, each token's blocks are read from disk
while the model's early layers compute. This repository trains the dense reference models and the store models,
serves a store model from NVMe, and regenerates every table, figure and number in the paper from the included
result files.

## Layout

```
src/
  model.py           transformer backbone (RMSNorm, RoPE, grouped-query attention, SwiGLU, tied head)
  moe.py             the expert store (HashMoE), balanced token-id hashing, learned-routing controls
  train.py           trains any configuration in the paper (dense or store) at any token budget
  data_partition.py  train / held-out split of the corpus
  evaluation.py      held-out loss on fixed batches
  schedule.py        warmup + cosine learning-rate schedule
  serving.py         incremental decode, on-disk expert store, background prefetcher
  serve_bench.py     same-model offload study; decode harness used by the serving scripts
  ckpt_load.py       rebuild a model from a train.py checkpoint
  lm_eval_wrap.py    lm-evaluation-harness adapter
scripts/
  build_corpus.py    tokenised FineWeb-Edu corpus, split and held-out file (checked against the paper's SHA-256)
  paper_runs.txt     the train.py command for every training result in results/training
  score_tasks.py     zero-shot LAMBADA / PIQA / HellaSwag / ARC-Easy for one checkpoint
  serve_iso.py       equal-quality serving comparison: dense-1B resident vs 220M backbone + NVMe store
  served_precision.py held-out loss through the served (int8 / bf16) weights
results/
  training/          one result JSON per training run (losses on every held-out split, training curve, sizes)
  tasks/             zero-shot accuracies
  serving/           serving measurements
analysis/
  make_assets.py     generates every table, figure and in-text number of the paper from results/
  frontier_fit.py    dense-frontier fit, dense-equivalent size, overhead and exchange rate
tests/               fast CPU checks
```

## Setup

```bash
pip install -r requirements.txt
python -m pytest tests          # a few seconds on CPU
```

## Regenerating the paper's tables and figures from the included results

```bash
python analysis/make_assets.py      # analysis/out/tables/*.tex, analysis/out/figs/*.pdf
```

`analysis/out/tables/facts2.tex` holds every number quoted in the text as a LaTeX macro. The two schematic
diagrams (Figures 1 and 2) are drawings, not data, and are not generated here.

## Reproducing the results

**Data.** `python scripts/build_corpus.py --out data/v3` streams FineWeb-Edu (sample-10BT), tokenises it with the
Llama-2 tokenizer and writes 8.2e9 tokens, the sequence order and the held-out split. It verifies the files
against the SHA-256 of the corpus used for the paper.

**Training.** Each line of `scripts/paper_runs.txt` is one run:

```bash
python src/train.py --arm 6 --mult 8 --store-norm --root data/v3 --out out                  # 27.7M + hash-512, 8e9 tokens
torchrun --standalone --nproc_per_node=8 src/train.py --arm 6 --mult 8 --store-norm \
    --micro-batch 8 --accum 1 --root data/v3 --out out                                       # same run on 8 GPUs
```

The data-parallel run consumes the same sequences in the same order as the single-GPU run. Each run writes
`out/result_<name>.json` (the format in `results/training`) and a checkpoint.

| paper element | configurations (train.py flags) |
|---|---|
| dense reference frontier (all figures) | `--arm 1 2 3 4 12`, `--mult 1 2 4 8` |
| store grid, Table 1, Figures 3-4, Table 5 | `--arm 5 9 10 6` (64 to 512 experts), `--arm 7 8` (larger backbones), `--store-norm` |
| routers, Figure 6a, Table 7 | `--gran 4 --shared 1 --cap-factor 0` with `--router learned` or `--router prev`; Switch control `--router learned --cap-factor 0` |
| multi-expert combination, Figure 6b, Table 8 | `--gran 4 --full-width --indep-hash` with `--slot-avg` and `--slot-gate` |
| second seeds, Table 9 | `--init-seed 1` |

**Task scores.** `python scripts/score_tasks.py <ckpt> <name> <out.json>` (lm-evaluation-harness, zero-shot).

**Serving.** On a GPU node with a local NVMe drive:

```bash
python scripts/serve_iso.py --dense-ckpt out/ckpt_tk-dense1B-1e9.pt --store-ckpt out/ckpt_tk-b220-e512-1e9-sn.pt \
    --store-dir /nvme/tmp --dtype bf16 --out iso_bf16_rep1.json        # main text: four repeats; --dtype fp32 for the appendix
python scripts/served_precision.py --dense-ckpt ... --store-ckpt ... --root data/v3 --out served_precision.json
python src/serve_bench.py --ckpt out/ckpt_tk-b27-e512-1e9-sn.pt --store-dir /nvme/tmp --out offload.json
```

The paper's serving numbers were measured on NVIDIA L40S nodes with local NVMe drives (Tables 10-12).

## Notes on provenance

- The 1e9-token losses of the four smaller dense models (3.2020, 3.0239, 2.9229, 2.8714) come from their first
  training with the identical recipe, data and seed, before `train.py` gained multi-budget support; they are
  written into `make_assets.py`. The same-seed `train.py` reruns in `results/training` agree within 0.01 nats.
  Their task scores are in `results/tasks`.
- The same-model offload study (`results/serving/offload_attempt*.json`, Appendix E) was measured on an earlier
  training of the 27.7M + hash-512 configuration. All four measurement attempts are included; attempt 3 is the
  reported one.
- The cross-domain probe files (`xd_lang`, `xd_code`) are rebuilt from their sources but are not byte-identical
  to the originals. The paper reports only the in-domain held-out loss.
- Checkpoints are not included because of their size (up to 13 GB each).

## License

MIT (see `LICENSE`).
