"""Run GCR step 1 (graph-constrained decoding, baseline B1) on BRINK.

Feeds BRINK questions with inverse-augmented L-hop subgraphs into GCR's own
prompt builder and decoding model (third_party/graph-constrained-reasoning),
then writes:
  predictions.jsonl  GCR format (10 "<PATH>...</PATH> # Answer: ..." strings per question)
  brink_pred.json    BRINK evaluator format; raw_output = de-duplicated step-1 answers, comma separated
and runs the official BRINK evaluator (third_party/brink/evaluation/evaluate_brink.py).

Example (CPU smoke test):
  python scripts/run_gcr_brink.py --dataset family --version incomplete --split "test[:5]" \
      --model_path rmanluo/GCR-Qwen2-0.5B-Instruct --dtype fp32 --attn_implementation sdpa
GPU (paper setting): --model_path rmanluo/GCR-Meta-Llama-3.1-8B-Instruct --dtype bf16
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GCR = os.path.join(ROOT, "third_party", "graph-constrained-reasoning")
BRINK = os.path.join(ROOT, "third_party", "brink")
sys.path.insert(0, ROOT)
sys.path.insert(0, GCR)

from tqdm import tqdm  # noqa: E402

from cc_gcd.data.brink import BrinkKG  # noqa: E402
from cc_gcd.data.paths import dfs_paths, dfs_paths_digraph  # noqa: E402


def load_gcr_prediction_fn():
    spec = importlib.util.spec_from_file_location("gcr_predict", os.path.join(GCR, "workflow", "predict_paths_and_answers.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.prediction


def parse_split(split: str):
    m = re.fullmatch(r"(\w+)(?:\[(\d*):(\d*)\])?", split)
    if not m:
        raise ValueError(f"bad split {split}")
    name, a, b = m.group(1), m.group(2), m.group(3)
    return name, int(a) if a else 0, int(b) if b else None


def path_validity(prediction, trie_paths):
    """Share of generated reasoning paths that are exact KG-Trie paths.

    GCR constrains decoding only between <PATH> and </PATH>; a beam that writes a
    path without opening <PATH> decodes freely and can produce paths that are not
    in the graph. This is the "faithful reasoning" rate of the GCR paper (Fig. 5).
    """
    valid_set = {" -> ".join([p[0][0]] + [x for h, r, t in p for x in (r, t)]) for p in trie_paths}
    preds = prediction if isinstance(prediction, list) else [prediction]
    n = valid = 0
    for p in preds:
        path = p.split("# Answer:\n")[0].split("# Reasoning Path:\n")[-1].strip()
        if not path:
            continue
        n += 1
        valid += path in valid_set
    return n, valid


def step1_answers(prediction):
    preds = prediction if isinstance(prediction, list) else [prediction]
    answers = []
    for p in preds:
        ans = p.split("# Answer:\n")[-1].strip()
        if ans and ans not in answers:
            answers.append(ans)
    return answers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["family", "fb15k237"], required=True)
    ap.add_argument("--version", choices=["complete", "incomplete"], default="incomplete")
    ap.add_argument("--split", default="test")
    ap.add_argument("--hops", type=int, default=2, help="KG-Trie path length L (GCR default 2)")
    ap.add_argument("--max_fanout", type=int, default=None, help="cap edges per (node, relation) beyond the seed")
    ap.add_argument("--path_mode", choices=["multigraph", "gcr_digraph"], default="multigraph")
    ap.add_argument("--model_path", default="rmanluo/GCR-Qwen2-0.5B-Instruct")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--generation_mode", default="group-beam")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--attn_implementation", default="sdpa")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "gcr_brink"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from src.llms.graph_constrained_decoding_model import GraphConstrainedDecodingModel
    from src.qa_prompt_builder import PathGenerationWithAnswerPromptBuilder

    split, lo, hi = parse_split(args.split)
    kg = BrinkKG(os.path.join(ROOT, "data", "processed", "brink", args.dataset), version=args.version, inverse=True)
    questions = list(kg.questions(split))[lo:hi]

    model_name = os.path.basename(args.model_path.rstrip("/"))
    tag = f"{args.dataset}-{args.version}-{args.split}-L{args.hops}-{args.path_mode}" + (f"-fan{args.max_fanout}" if args.max_fanout else "")
    out_dir = os.path.join(args.out, model_name, tag)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    gcr_args = argparse.Namespace(
        model_path=args.model_path, maximun_token=4096, max_new_tokens=args.max_new_tokens, dtype=args.dtype,
        quant="none", attn_implementation=args.attn_implementation, generation_mode=args.generation_mode,
        k=args.k, chat_model=True, use_assistant_model=False, assistant_model_path=None,
    )
    model = GraphConstrainedDecodingModel(gcr_args)
    model.prepare_for_inference()
    builder = PathGenerationWithAnswerPromptBuilder(model.tokenizer, "zero-shot", index_path_length=args.hops)
    predict = load_gcr_prediction_fn()
    enumerate_paths = dfs_paths if args.path_mode == "multigraph" else dfs_paths_digraph

    pred_file = os.path.join(out_dir, "predictions.jsonl")
    done = set()
    if os.path.exists(pred_file) and not args.force:
        with open(pred_file) as f:
            done = {json.loads(line)["id"] for line in f}
    with open(pred_file, "w" if args.force else "a") as fout:
        for q in tqdm(questions):
            if q["id"] in done:
                continue
            sample = kg.sample(q, hops=args.hops, max_fanout=args.max_fanout)
            sample["paths"] = enumerate_paths(sample["graph"], sample["q_entity"], args.hops)
            res = predict(sample, [], builder, model)
            if res is None:
                res = {"id": q["id"], "question": q["question"], "prediction": [], "ground_truth": sample["answer"]}
            res["n_trie_paths"] = len(sample["paths"])
            res["n_paths_generated"], res["n_paths_in_kg"] = path_validity(res["prediction"], sample["paths"])
            res["hard_answer"] = sample["hard_answer"]
            res.pop("input", None)
            fout.write(json.dumps(res, ensure_ascii=False) + "\n")
            fout.flush()

    ids = {q["id"] for q in questions}
    rows = [json.loads(line) for line in open(pred_file)]
    brink_pred = [{"id": r["id"], "raw_output": ", ".join(step1_answers(r["prediction"]))} for r in rows if r["id"] in ids]
    with open(os.path.join(out_dir, "brink_pred.json"), "w") as f:
        json.dump(brink_pred, f, ensure_ascii=False, indent=1)
    gold = [g for g in json.load(open(os.path.join(ROOT, "data", "processed", "brink", args.dataset, f"gold_{split}.json"))) if g["id"] in ids]
    gold_file = os.path.join(out_dir, "gold.json")
    with open(gold_file, "w") as f:
        json.dump(gold, f, ensure_ascii=False, indent=1)
    subprocess.run([sys.executable, os.path.join(BRINK, "evaluation", "evaluate_brink.py"), "--gold", gold_file,
                    "--pred", os.path.join(out_dir, "brink_pred.json")], check=True)
    gen = sum(r.get("n_paths_generated", 0) for r in rows if r["id"] in ids)
    ok = sum(r.get("n_paths_in_kg", 0) for r in rows if r["id"] in ids)
    print(json.dumps({"paths_generated": gen, "paths_in_kg": ok, "faithful_path_rate": ok / gen if gen else None}))


if __name__ == "__main__":
    main()
