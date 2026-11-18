"""Measure the ceiling of hard KG-Trie decoding (GCR) on RoG-WebQSP.

For each question, GCR builds its KG-Trie from all directed paths of length
<= L starting at the topic entities inside the question's retrieved subgraph
(src/qa_prompt_builder.py, JointReasoningPromptBuilder.get_graph_index).
A hard-masked decoder can only output answers that appear on those paths, so
the share of questions whose gold answer is reachable is an upper bound on
step-1 Hit (the empirical Cov(q) of Li & Song, Prop. 4).

Usage: python analysis/kg_trie_coverage.py --split test --L 2 --n 4
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool

GCR_ROOT = os.path.join(os.path.dirname(__file__), "..", "third_party", "graph-constrained-reasoning")
sys.path.insert(0, os.path.abspath(GCR_ROOT))

from datasets import load_dataset  # noqa: E402
from src.utils.graph_utils import build_graph, dfs  # noqa: E402

L = 2


def analyse(sample):
    g = build_graph(sample["graph"])
    q_entities = [e for e in sample["q_entity"]]
    answers = set(a.strip() for a in sample["a_entity"])
    paths = dfs(g, q_entities, L)
    reachable_nodes = set()
    endpoint_nodes = set()
    for p in paths:
        for h, _, t in p:
            reachable_nodes.add(h)
            reachable_nodes.add(t)
        endpoint_nodes.add(p[-1][2])
    in_graph = answers & set(g.nodes())
    covered = answers & endpoint_nodes
    return {
        "id": sample["id"],
        "n_answers": len(answers),
        "n_paths": len(paths),
        "n_triples": len(sample["graph"]),
        "q_in_graph": any(e in g for e in q_entities),
        "ans_in_graph": len(in_graph),
        "ans_covered": len(covered),
        "hit_ceiling": int(len(covered) > 0),
        "recall_ceiling": len(covered) / max(len(answers), 1),
        "no_trie": int(len(paths) == 0),
    }


def main():
    global L
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rmanluo/RoG-webqsp")
    ap.add_argument("--split", default="test")
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "kg_trie_coverage"))
    args = ap.parse_args()
    L = args.L

    ds = load_dataset(args.dataset, split=args.split)
    with Pool(args.n) as pool:
        rows = list(pool.imap(analyse, ds, chunksize=8))

    os.makedirs(args.out, exist_ok=True)
    name = f"{args.dataset.split('/')[-1]}_{args.split}_L{args.L}"
    with open(os.path.join(args.out, name + ".jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n = len(rows)
    mean = lambda k: sum(r[k] for r in rows) / n  # noqa: E731
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "L": args.L,
        "n_questions": n,
        "no_trie_rate": mean("no_trie"),
        "answer_in_subgraph_rate": sum(r["ans_in_graph"] > 0 for r in rows) / n,
        "hit_ceiling": mean("hit_ceiling"),
        "recall_ceiling": mean("recall_ceiling"),
        "mean_paths": mean("n_paths"),
        "median_paths": sorted(r["n_paths"] for r in rows)[n // 2],
        "mean_subgraph_triples": mean("n_triples"),
    }
    with open(os.path.join(args.out, name + "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
