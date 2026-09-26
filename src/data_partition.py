"""Train / held-out split of the corpus's 2048-token sequences (scripts/build_corpus.py writes the permutation).

    train      488,281 sequences (1.0e9 tokens) for the 1e9 budget; larger budgets append further sequences
    reserved    12,000 sequences, never trained on or scored
    eval        12,414 sequences, in-domain held-out evaluation

For budgets above 1e9 tokens the extra training sequences are taken from the permutation after the first
V2_SEQS entries, so the reserved and eval sequences never change and the first 1e9 tokens of every run are
byte-identical to the 1e9 run.
"""

from __future__ import annotations

import numpy as np

TRAIN_SEQS = 488_281  # 1.0e9 tokens at seq_len 2048
RESERVED_SEQS = 12_000
V2_SEQS = 512_695  # size of the first corpus block; train extensions come after it
SEQ_LEN = 2048


def partition(perm_path: str, train_seqs: int = TRAIN_SEQS) -> dict:
    perm = np.load(perm_path)
    if train_seqs < TRAIN_SEQS:
        raise ValueError(f"train_seqs {train_seqs:,} below the minimum {TRAIN_SEQS:,}")
    if len(np.unique(perm)) != len(perm):
        raise ValueError("permutation contains duplicates")
    if len(perm) < TRAIN_SEQS + RESERVED_SEQS + 1000:
        raise ValueError(f"corpus has {len(perm):,} sequences, too few for the split")

    train = perm[:TRAIN_SEQS]
    reserved = perm[TRAIN_SEQS : TRAIN_SEQS + RESERVED_SEQS]
    evl = perm[TRAIN_SEQS + RESERVED_SEQS : V2_SEQS]
    extra = train_seqs - TRAIN_SEQS
    if extra:
        if len(perm) - V2_SEQS < extra:
            raise ValueError(f"corpus holds {len(perm) - V2_SEQS:,} sequences past the first block; {extra:,} requested")
        train = np.concatenate([train, perm[V2_SEQS : V2_SEQS + extra]])

    parts = {"train": train, "reserved": reserved, "eval": evl}
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ov = np.intersect1d(parts[a], parts[b])
            if ov.size:
                raise ValueError(f"split leak: {a} and {b} share {ov.size:,} sequences")
    return parts


def write_eval_split(tokens_path: str, parts: dict, out_path: str) -> dict:
    """Write the eval sequences as a contiguous .bin of (SEQ_LEN + 1)-token records for EvalSuite."""
    import hashlib
    import os

    tok = np.memmap(tokens_path, dtype=np.uint16, mode="r")
    idx = parts["eval"]
    with open(out_path, "wb") as f:
        for i in idx:
            f.write(np.asarray(tok[i * SEQ_LEN : (i + 1) * SEQ_LEN + 1], dtype=np.uint16).tobytes())
    h = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
    return {"sequences": int(len(idx)), "bytes": os.path.getsize(out_path), "tokens": int(len(idx) * (SEQ_LEN + 1)), "sha256": h}
