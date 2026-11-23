"""Paired bootstrap for E1: per-question metrics of each method vs a reference, on the same questions.

BRINK: per-question Hits@Any / Hits@Hard / F1 from the official evaluator's compute_example_metrics on
each method's brink_pred.json and gold.json. IKGQA: per-question Hit and answer F1 with GCR's qa_utils.
For every method, the mean difference to the reference, its 95% interval over 10,000 resamples of
questions, and the share of resamples with a difference <= 0 (one-sided p for "better").

    python scripts/e1_bootstrap.py brink --dataset fb15k237 --split "test" --ref B1
    python scripts/e1_bootstrap.py ikgqa --tag ikg80_eval300 --ref B1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def brink_per_question(method_dir: str):
    spec = importlib.util.spec_from_file_location("evb", os.path.join(ROOT, "third_party", "brink", "evaluation",
                                                                      "evaluate_brink.py"))
    evb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evb)
    gold = evb.load_gold(os.path.join(method_dir, "gold.json"))
    preds = evb.load_predictions(os.path.join(method_dir, "brink_pred.json")) if hasattr(evb, "load_predictions") else None
    if preds is None:
        raw = json.load(open(os.path.join(method_dir, "brink_pred.json")))
        preds = {p["id"]: evb.process_prediction(p["raw_output"]) for p in raw}
    out = {}
    for qid, g in gold.items():
        m = evb.compute_example_metrics(set(preds.get(qid, [])), g["answers"], g["hard_answer"])
        out[qid] = m
    return out


def ikgqa_per_question(method_dir: str):
    sys.path.insert(0, os.path.join(ROOT, "third_party", "graph-constrained-reasoning"))
    from src.utils.qa_utils import eval_f1, eval_hit
    from cc_gcd.baselines.gcr import step1_answers
    out = {}
    for line in open(os.path.join(method_dir, "predictions.jsonl")):
        r = json.loads(line)
        preds = r["prediction"] if isinstance(r["prediction"], list) else [r["prediction"]]
        ans = step1_answers(preds)
        gold = list(set(r["ground_truth"]))
        out[r["id"]] = {"Hit": float(eval_hit(" ".join(preds), gold)), "F1": eval_f1(ans, gold)[0] if ans else 0.0}
    return out


def bootstrap(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(1)
    return {"diff": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(means, 2.5)), 4),
                                                       round(float(np.percentile(means, 97.5)), 4)],
            "p_not_better": round(float((means <= 0).mean()), 4), "n": int(len(d))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", choices=["brink", "ikgqa"])
    ap.add_argument("--dataset", default="fb15k237")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default=None, help="ikgqa run tag, e.g. ikg80_eval300")
    ap.add_argument("--model", default="GCR-Qwen2-0.5B-Instruct")
    ap.add_argument("--ref", default="B1")
    args = ap.parse_args()
    if args.bench == "brink":
        base = os.path.join(ROOT, "data", "e1_brink", args.dataset, args.model, args.split.replace(":", "-"))
        per, metrics = brink_per_question, ("Hits@Any", "Hits@Hard", "F1")
        msfx = "" if args.model == "GCR-Qwen2-0.5B-Instruct" else f"_{args.model}"
        out = os.path.join(ROOT, "results", "e1_brink",
                           f"{args.dataset}_{args.split.replace(':', '-')}{msfx}_bootstrap_vs_{args.ref.replace(':', '_')}.json")
    else:
        base = os.path.join(ROOT, "data", "e1_ikgqa", args.model, args.tag)
        per, metrics = ikgqa_per_question, ("Hit", "F1")
        out = os.path.join(ROOT, "results", "e1_ikgqa", f"webqsp_{args.tag}_bootstrap_vs_{args.ref}.json")
    need = "gold.json" if args.bench == "brink" else "predictions.jsonl"
    methods = sorted(m for m in os.listdir(base) if os.path.exists(os.path.join(base, m, need))
                     and (args.bench == "ikgqa" or os.path.exists(os.path.join(base, m, "brink_pred.json"))))
    ref = per(os.path.join(base, args.ref.replace(":", "_").replace("/", "-")))
    res = {}
    for m in methods:
        pq = per(os.path.join(base, m))
        qs = sorted(set(pq) & set(ref))
        res[m] = {k: {"mean": round(float(np.mean([pq[q][k] for q in qs])), 4),
                      "vs_ref": bootstrap(np.array([pq[q][k] for q in qs]), np.array([ref[q][k] for q in qs]))}
                  for k in metrics}
        print(m, " ".join(f"{k} {v['mean']:.4f} ({v['vs_ref']['diff']:+.4f} [{v['vs_ref']['ci95'][0]:+.4f},"
                          f"{v['vs_ref']['ci95'][1]:+.4f}] p={v['vs_ref']['p_not_better']})" for k, v in res[m].items()))
    json.dump({"reference": args.ref, "methods": res}, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
