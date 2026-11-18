"""Run GCR step 1 (baseline B1) on IKGQA (GoG's IKG-0/20/40/60/80% for WebQSP or CWQ).

Uses GCR's own step-1 metrics (src/utils/qa_utils.eval_path_result_w_ans: Hit, F1, path F1)
plus the faithful-path rate. ``--path_mode gcr_digraph`` reproduces GCR's original KG-Trie
construction (one relation per entity pair); ``multigraph`` keeps parallel relations.

CPU smoke test:
  python scripts/run_gcr_ikgqa.py --dataset webqsp --level 0.4 --limit 5 \
      --model_path rmanluo/GCR-Qwen2-0.5B-Instruct --dtype fp32
GPU (paper setting): --model_path rmanluo/GCR-Meta-Llama-3.1-8B-Instruct --dtype bf16
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from tqdm import tqdm  # noqa: E402

from cc_gcd.baselines.gcr import PATH_MODES, GCRRunner, faithful_rate  # noqa: E402
from cc_gcd.data.ikgqa import IKGQA, LEVELS, level_tag  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["webqsp", "cwq"], default="webqsp")
    ap.add_argument("--level", type=float, choices=LEVELS, default=0.4)
    ap.add_argument("--limit", type=int, default=None, help="first N of GoG's 1,000 questions")
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--path_mode", choices=sorted(PATH_MODES), default="gcr_digraph")
    ap.add_argument("--model_path", default="rmanluo/GCR-Qwen2-0.5B-Instruct")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--attn_implementation", default="sdpa")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "gcr_ikgqa"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    data = IKGQA(os.path.join(ROOT, "data", "processed", "ikgqa"), args.dataset)
    n = args.limit or len(data.removals(args.level))
    tag = f"{args.dataset}-{level_tag(args.level)}-n{n}-L{args.hops}-{args.path_mode}"
    out_dir = os.path.join(args.out, os.path.basename(args.model_path.rstrip("/")), tag)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out_dir, "args.json"), "w"), indent=2)

    runner = GCRRunner(args.model_path, k=args.k, dtype=args.dtype, attn_implementation=args.attn_implementation,
                       hops=args.hops)
    pred_file = os.path.join(out_dir, "predictions.jsonl")
    samples = itertools.islice(data.samples(args.level), n)
    runner.run(samples, pred_file, path_mode=args.path_mode, force=args.force, progress=lambda it: tqdm(it, total=n))

    from src.utils.qa_utils import eval_path_result_w_ans

    eval_path_result_w_ans(pred_file)
    print(json.dumps(faithful_rate([json.loads(line) for line in open(pred_file)])))


if __name__ == "__main__":
    main()
