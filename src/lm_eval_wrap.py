"""lm-evaluation-harness (0.4.x) adapter for train.py checkpoints: zero-shot LAMBADA, PIQA, HellaSwag and ARC-Easy.

Only loglikelihood requests are implemented; the four tasks need nothing else. Store rows are supplied exactly as
in training: assign[token] per position for hash stores, placeholders for learned routers.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM


class StoreLM(LM):
    def __init__(self, model, tokenizer, assign=None, layers=(), device="cpu", max_len=2048, batch_size=8):
        super().__init__()
        self.m, self.tk = model.eval(), tokenizer
        self.assign = assign  # {layer: LongTensor[vocab]} or None for dense
        self.layers = list(layers)
        self.dev, self.max_len, self.bs = device, max_len, batch_size

    def _rows(self, x):
        if self.assign is None:  # dense (no layers) or a learned router (rows are placeholders)
            return {l: x for l in self.layers} if self.layers else None
        return {l: self.assign[l][x] for l in self.layers}

    @torch.no_grad()
    def _score(self, ctx_ids, cont_ids):
        """logprob sum of cont given ctx, and greedy flag. Truncates left to max_len."""
        ids = (ctx_ids + cont_ids)[-self.max_len :]
        n_cont = len(cont_ids)
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=self.dev)
        logits = self.m(x, None, self._rows(x))  # [1, T, V]
        lp = F.log_softmax(logits.float(), -1)[0]
        tgt = torch.tensor(ids[1:], device=self.dev)
        cont_lp = lp[-n_cont:].gather(-1, tgt[-n_cont:].unsqueeze(-1)).squeeze(-1)
        greedy = bool((lp[-n_cont:].argmax(-1) == tgt[-n_cont:]).all())
        return float(cont_lp.sum()), greedy

    def loglikelihood(self, requests: list[Instance]):
        out = []
        for req in requests:
            ctx, cont = req.args
            if ctx == "":
                # score cont against a lone BOS; encode() would add its own BOS on top
                ctx_ids = [self.tk.bos_token_id]
                whole = ctx_ids + self.tk.encode(cont, add_special_tokens=False)
            else:
                ctx_ids = self.tk.encode(ctx)
                whole = self.tk.encode(ctx + cont)
            # Split point is only valid if retokenizing ctx+cont kept ctx's tokens as a
            # prefix. If not, realign at the first divergence so no context token is
            # scored as continuation (and vice versa).
            k = min(len(ctx_ids), len(whole))
            i = 0
            while i < k and whole[i] == ctx_ids[i]:
                i += 1
            cont_ids = whole[i:]
            ctx_ids = whole[:i]
            if not cont_ids:  # tokenizer merge edge: re-split
                cont_ids = whole[-1:]
                ctx_ids = whole[:-1]
            out.append(self._score(ctx_ids, cont_ids))
        return out

    def loglikelihood_rolling(self, requests):
        out = []
        for req in requests:
            (text,) = req.args
            ids = self.tk.encode(text)[: self.max_len]
            x = torch.tensor([ids[:-1]], dtype=torch.long, device=self.dev)
            with torch.no_grad():
                logits = self.m(x, None, self._rows(x))
            lp = F.log_softmax(logits.float(), -1)[0]
            tgt = torch.tensor(ids[1:], device=self.dev)
            out.append(float(lp.gather(-1, tgt.unsqueeze(-1)).sum()))
        return out

    def generate_until(self, requests):
        raise NotImplementedError("the scored tasks are loglikelihood-only")


def load_ckpt_as_lm(ckpt_path, device="cpu", name=None):
    """Wrap a train.py checkpoint, or a dense checkpoint in the earlier {d, l, q, kv, hd, ff} arch format (the
    dense 1e9 reference models), as an lm-eval LM."""
    import os

    from transformers import AutoTokenizer

    from ckpt_load import load_ckpt_model
    from model import Config, Transformer

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["arch"]
    tk = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    if "moe_layers" in a:
        name = name or os.path.basename(ckpt_path).removeprefix("ckpt_").removesuffix(".pt")
        m, assign, layers, _ = load_ckpt_model(ck, device, name=name)
        return StoreLM(m, tk, assign, layers, device)
    cfg = Config(d_model=a["d"], n_layers=a["l"], n_q_heads=a["q"], n_kv_heads=a["kv"], head_dim=a["hd"], d_ff=a["ff"],
                 vocab=a["vocab"], seq_len=a["seq_len"], init_seed=0)
    m = Transformer(cfg, None)
    m.load_state_dict({k: (v.to(torch.float32) if v.is_floating_point() else v) for k, v in ck["state_dict_bf16"].items()})
    return StoreLM(m.to(device), tk, None, (), device)
