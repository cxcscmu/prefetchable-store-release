"""Build the training corpus (data/v3) from scratch.

  tokens_u16.bin              the first 8.2e9 tokens of FineWeb-Edu sample-10BT in stream order, Llama-2 tokenizer,
                              no special tokens, documents concatenated, uint16
  tokens_u16_doc_offsets.npy  document start offsets (last entry = number of tokens)
  seq_permutation.npy         order of the 2048-token sequences: a seed-0 shuffle of the first 512,695 sequences,
                              then a seed-1 shuffle of the rest (data_partition.py draws train / eval from it)
  main_heldout.bin            the eval sequences (the held-out split every loss in the paper is measured on)
  xd_lang.bin, xd_code.bin    10M-token cross-domain probes (German FineWeb-2, Python); not reported in the paper

tokens_u16.bin, tokens_u16_doc_offsets.npy, seq_permutation.npy and main_heldout.bin are checked against the
SHA-256 of the files the paper's runs used. The two cross-domain probes are rebuilt from their sources but are not
byte-identical to the originals (the original code probe mixed a small gated-repository sample into codeparrot).

    python scripts/build_corpus.py --out data/v3          # ~40 min on 16 cores, ~17 GB
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from data_partition import V2_SEQS, partition, write_eval_split  # noqa: E402

SEQ_LEN = 2048
TOKENIZER = "NousResearch/Llama-2-7b-hf"
EXPECTED = {
    "tokens_u16.bin": "e776fde67843376f4743b8e070a2af9145315b515b262e54003e5e6da38f2afa",
    "tokens_u16_doc_offsets.npy": "ccc0c10b020e9f69e63cf20e0a0f79c7358f25b03d148e91dbe33a196d29e92b",
    "seq_permutation.npy": "35960673eb0db5bdd2e6f51138d717c97963f0376ca679ae22889d35af416399",
    "main_heldout.bin": "23887c3297bff91b008f59e19a6ec9bc8469aeb7b8a70125940c3c66f9da0f75",
}


def sha256_file(p, chunk=1 << 26):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def stream_tokens(tok, texts, target, out_path, batch_docs=2048):
    """Tokenise documents in order and write them concatenated until `target` tokens; returns doc offsets."""
    offsets, n = [0], 0
    with open(out_path, "wb") as f:
        batch = []

        def flush(batch):
            nonlocal n
            for ids in tok(batch, add_special_tokens=False)["input_ids"]:
                arr = np.asarray(ids, dtype=np.uint16)[: target - n]
                f.write(arr.tobytes())
                n += len(arr)
                offsets.append(n)
                if n >= target:
                    return True
            return False

        for text in texts:
            batch.append(text)
            if len(batch) >= batch_docs:
                if flush(batch):
                    break
                batch = []
        else:
            if batch:
                flush(batch)
    assert n == target, f"stream exhausted at {n:,} of {target:,} tokens"
    return np.asarray(offsets, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/v3")
    ap.add_argument("--target", type=int, default=8_200_000_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    t0 = time.time()

    # 1. training stream
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    off = stream_tokens(tok, (r["text"] for r in ds), a.target, f"{a.out}/tokens_u16.bin")
    np.save(f"{a.out}/tokens_u16_doc_offsets.npy", off)

    # 2. sequence permutation
    n_seqs = a.target // SEQ_LEN
    perm = np.concatenate([np.random.default_rng(0).permutation(V2_SEQS).astype(np.int64),
                           np.random.default_rng(1).permutation(np.arange(V2_SEQS, n_seqs, dtype=np.int64))])
    np.save(f"{a.out}/seq_permutation.npy", perm)

    # 3. in-domain held-out split
    write_eval_split(f"{a.out}/tokens_u16.bin", partition(f"{a.out}/seq_permutation.npy"), f"{a.out}/main_heldout.bin")

    # 4. cross-domain probes (not reported in the paper)
    de = load_dataset("HuggingFaceFW/fineweb-2", "deu_Latn", split="train", streaming=True)
    np.save(f"{a.out}/xd_lang_doc_offsets.npy", stream_tokens(tok, (r["text"] for r in de), 10_000_000, f"{a.out}/xd_lang.bin"))
    py = load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True)
    np.save(f"{a.out}/xd_code_doc_offsets.npy", stream_tokens(tok, (r["content"] for r in py), 10_000_000, f"{a.out}/xd_code.bin"))

    shas = {k: sha256_file(f"{a.out}/{k}") for k in EXPECTED}
    bad = {k: v for k, v in shas.items() if v != EXPECTED[k]}
    json.dump({"tokenizer": TOKENIZER, "n_tokens": a.target, "n_sequences": int(n_seqs), "sha256": shas,
               "elapsed_s": round(time.time() - t0)}, open(f"{a.out}/corpus_meta.json", "w"), indent=2)
    if a.target == 8_200_000_000 and bad:
        raise SystemExit(f"checksum mismatch against the paper's corpus: {sorted(bad)}")
    print("corpus OK" if a.target == 8_200_000_000 else "corpus built (non-default size: checksums not compared)")


if __name__ == "__main__":
    main()
