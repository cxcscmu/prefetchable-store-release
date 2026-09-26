"""Zero-shot accuracy (LAMBADA, PIQA, HellaSwag, ARC-Easy) for one checkpoint with lm-evaluation-harness.

    python scripts/score_tasks.py ckpt_tk-b27-e512-1e9-sn.pt tk-b27-e512-1e9-sn tasks_tk-b27-e512-1e9-sn.json
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import lm_eval  # noqa: E402

from lm_eval_wrap import load_ckpt_as_lm  # noqa: E402

TASKS = ["lambada_openai", "piqa", "hellaswag", "arc_easy"]
ck, name, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
t0 = time.time()
lm = load_ckpt_as_lm(ck, os.environ.get("DEVICE", "cuda"), name=name)
res = lm_eval.simple_evaluate(model=lm, tasks=TASKS, verbosity="ERROR")
out = {name: {t: {k: v for k, v in res["results"][t].items() if isinstance(v, (int, float))} for t in TASKS}}
out[name]["wall_s"] = time.time() - t0
json.dump(out, open(out_path, "w"), indent=2)
print(json.dumps({t: out[name][t].get("acc,none") for t in TASKS}))
