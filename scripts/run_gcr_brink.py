"""Run GCR step 1 (baseline B1) on BRINK and score it with the official BRINK evaluator.

Writes to results/gcr_brink/<model>/<tag>/:
  predictions.jsonl  GCR format, plus n_trie_paths and the faithful-path counts
  brink_pred.json    BRINK evaluator input; raw_output = de-duplicated step-1 answers, comma separated
  gold.json          the matching subset of data/processed/brink/<dataset>/gold_<split>.json

CPU smoke test:
  python scripts/run_gcr_brink.py --dataset family --split "test[:5]" \
      --model_path rmanluo/GCR-Qwen2-0.5B-Instruct --dtype fp32
GPU (paper setting): --model_path rmanluo/GCR-Meta-Llama-3.1-8B-Instruct --dtype bf16
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from tqdm import tqdm  # noqa: E402

from cc_gcd.baselines.gcr import PATH_MODES, GCRRunner, faithful_rate, step1_answers  # noqa: E402
from cc_gcd.data.brink import BrinkKG  # noqa: E402

BRINK_EVAL = os.path.join(ROOT, "third_party", "brink", "evaluation", "evaluate_brink.py")


def parse_split(split: str):
    m = re.fullmatch(r"(\w+)(?:\[(\d*):(\d*)\])?", split)
    if not m:
        raise ValueError(f"bad split {split}")
    return m.group(1), int(m.group(2) or 0), int(m.group(3)) if m.group(3) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["family", "fb15k237"], required=True)
    ap.add_argument("--version", choices=["complete", "incomplete"], default="incomplete")
    ap.add_argument("--split", default="test")
    ap.add_argument("--hops", type=int, default=2, help="KG-Trie path length L (GCR default 2)")
    ap.add_argument("--max_fanout", type=int, default=None, help="cap edges per (node, relation) beyond the seed")
    ap.add_argument("--path_mode", choices=sorted(PATH_MODES), default="multigraph")
    ap.add_argument("--model_path", default="rmanluo/GCR-Qwen2-0.5B-Instruct")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--attn_implementation", default="sdpa")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "gcr_brink"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    split, lo, hi = parse_split(args.split)
    kg = BrinkKG(os.path.join(ROOT, "data", "processed", "brink", args.dataset), version=args.version, inverse=True)
    questions = list(kg.questions(split))[lo:hi]
    ids = {q["id"] for q in questions}

    tag = f"{args.dataset}-{args.version}-{args.split}-L{args.hops}-{args.path_mode}" + (
        f"-fan{args.max_fanout}" if args.max_fanout else "")
    out_dir = os.path.join(args.out, os.path.basename(args.model_path.rstrip("/")), tag)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out_dir, "args.json"), "w"), indent=2)

    runner = GCRRunner(args.model_path, k=args.k, dtype=args.dtype, attn_implementation=args.attn_implementation,
                       hops=args.hops)
    samples = (kg.sample(q, hops=args.hops, max_fanout=args.max_fanout) for q in questions)
    pred_file = os.path.join(out_dir, "predictions.jsonl")
    runner.run(samples, pred_file, path_mode=args.path_mode, force=args.force,
               progress=lambda it: tqdm(it, total=len(questions)))

    rows = [r for r in map(json.loads, open(pred_file)) if r["id"] in ids]
    pred = [{"id": r["id"], "raw_output": ", ".join(step1_answers(r["prediction"]))} for r in rows]
    json.dump(pred, open(os.path.join(out_dir, "brink_pred.json"), "w"), ensure_ascii=False, indent=1)
    gold = [g for g in json.load(open(os.path.join(kg.root, f"gold_{split}.json"))) if g["id"] in ids]
    json.dump(gold, open(os.path.join(out_dir, "gold.json"), "w"), ensure_ascii=False, indent=1)
    subprocess.run([sys.executable, BRINK_EVAL, "--gold", os.path.join(out_dir, "gold.json"),
                    "--pred", os.path.join(out_dir, "brink_pred.json")], check=True)
    print(json.dumps(faithful_rate(rows)))


if __name__ == "__main__":
    main()
