"""Held-out evaluation on fixed batches: the same windows are scored for every model.

Splits: main_heldout (in-domain eval sequences) and two cross-domain probes, xd_lang (German web text) and
xd_code (Python). The paper reports main_heldout.

Convention: eval files store (seq_len + 1)-token records, but windows are sliced at stride seq_len, so most
windows straddle one record boundary. This is identical for every model, so all comparisons are unaffected, but
absolute losses are not comparable with other evaluation setups.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Split:
    name: str
    path: str
    kind: str  # "in_domain" | "cross_domain"
    min_tokens: int = 1_000_000


class EvalSuite:
    def __init__(self, splits, seq_len: int = 2048, batch: int = 16, max_batches: int = 64, seed: int = 0):
        self.seq_len, self.batch, self.max_batches = seq_len, batch, max_batches
        self.splits, self.data = [], {}
        problems = []
        for s in splits:
            if not os.path.exists(s.path):
                problems.append(f"{s.name}: missing file {s.path}")
                continue
            n_tok = os.path.getsize(s.path) // 2
            if n_tok < s.min_tokens:
                problems.append(f"{s.name}: {n_tok:,} tokens is below min_tokens={s.min_tokens:,}")
                continue
            self.splits.append(s)
            self.data[s.name] = np.memmap(s.path, dtype=np.uint16, mode="r")
        if problems:
            raise ValueError("evaluation splits unusable:\n  " + "\n  ".join(problems))
        self._batches = {s.name: self._fixed_batches(s.name) for s in self.splits}

    def _fixed_batches(self, name):
        arr = self.data[name]
        n_seq = len(arr) // (self.seq_len + 1)
        n = min(self.max_batches * self.batch, n_seq)
        idx = np.random.default_rng(0).permutation(n_seq)[:n]  # fixed across models
        return [idx[i : i + self.batch] for i in range(0, n, self.batch) if len(idx[i : i + self.batch]) == self.batch]

    @torch.no_grad()
    def evaluate(self, model, forward_fn=None, device="cpu") -> dict:
        """Token-weighted mean loss per split."""
        model.eval()
        out = {}
        for s in self.splits:
            arr, tot_loss, tot_tok = self.data[s.name], 0.0, 0
            for idx in self._batches[s.name]:
                L = self.seq_len
                seqs = np.stack([np.asarray(arr[i * L : (i + 1) * L + 1], dtype=np.int64) for i in idx])
                x = torch.from_numpy(seqs[:, :-1]).to(device)
                y = torch.from_numpy(seqs[:, 1:]).to(device)
                loss = forward_fn(model, x, y) if forward_fn else model(x, y)
                ntok = y.numel()
                tot_loss += float(loss) * ntok
                tot_tok += ntok
            m = tot_loss / tot_tok
            out[s.name] = {"loss": m, "ppl": math.exp(min(20, m)), "tokens": tot_tok, "kind": s.kind}
        model.train()
        return out


def default_splits(root="data/v3") -> list:
    return [
        Split("main_heldout", f"{root}/main_heldout.bin", "in_domain", 10_000_000),
        Split("xd_lang", f"{root}/xd_lang.bin", "cross_domain", 5_000_000),
        Split("xd_code", f"{root}/xd_code.bin", "cross_domain", 5_000_000),
    ]
