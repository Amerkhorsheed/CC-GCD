"""Build IKGQA (GoG) incomplete versions of RoG-WebQSP / RoG-CWQ test subgraphs.

Input : third_party/GoG/data/<dataset>/data_with_ct_{0,0.2,0.4,0.6,0.8}.json (cloned by scripts/setup.sh)
        Hugging Face rmanluo/RoG-<dataset> test split
Output: data/processed/ikgqa/<dataset>/ikg{0,20,40,60,80}.jsonl, one line per question:
          id, rog_idx, removed_triples (GoG crucial triples present in the RoG subgraph),
          n_crucial (all GoG crucial triples), n_crucial_absent (not in the RoG subgraph)
        data/processed/ikgqa/<dataset>/stats.json and results/ikgqa_stats/<dataset>.json
Stats per level: deletions, and the hard KG-Trie ceiling (answer on a <= L-hop path from the
topic entities), both with GCR's DiGraph semantics and with a multigraph plus inverse edges.

Usage: python data_prep/prepare_ikgqa.py --dataset webqsp [--n 4]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from cc_gcd.data.ikgqa import LEVELS, ROG, level_tag, load_rog_test  # noqa: E402

GOG = os.path.join(ROOT, "third_party", "GoG", "data")
HOPS = (1, 2, 3)


def reach(graph, seeds, mode):
    """Min hop distance from seeds; mode in {'digraph', 'inverse'}."""
    adj = defaultdict(set)
    for h, r, t in graph:
        h, t = h.strip(), t.strip()
        adj[h].add(t)
        if mode == "inverse":
            adj[t].add(h)
    dist = {s: 0 for s in seeds}
    frontier = list(dist)
    for d in range(1, max(HOPS) + 1):
        nxt = []
        for u in frontier:
            for v in adj.get(u, ()):
                if v not in dist:
                    dist[v] = d
                    nxt.append(v)
        frontier = nxt
    return dist


def analyse(args):
    graph, q_entity, a_entity, removed = args
    removed_set = {tuple(t) for t in removed}
    out = {}
    for name, g in (("full", graph), ("ikg", [t for t in graph if tuple(t) not in removed_set])):
        for mode in ("digraph", "inverse"):
            dist = reach(g, q_entity, mode)
            dmin = min((dist[a] for a in a_entity if a in dist), default=None)
            for k in HOPS:
                out[f"{name}_{mode}_hit_within_{k}"] = int(dmin is not None and dmin <= k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(ROG), default="webqsp")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "processed", "ikgqa"))
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    rog = load_rog_test(args.dataset)
    out_dir = os.path.join(args.out, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    stats = {"dataset": args.dataset, "rog": ROG[args.dataset], "levels": {}}

    for p in LEVELS:
        src = os.path.join(GOG, args.dataset, f"data_with_ct_{p:g}.json")
        gog = json.load(open(src, encoding="utf-8"))
        recs, jobs = [], []
        id_mismatch = 0
        for s in gog:
            r = rog[s["idx_in_processed_samples"]]
            qid = s.get("QuestionId") or s.get("ID")
            id_mismatch += r["id"] != qid
            g = {tuple(t) for t in r["graph"]}
            crucial = [tuple(t) for t in s["crucial_triples"]]
            present = sorted({t for t in crucial if t in g})
            recs.append({
                "id": r["id"],
                "rog_idx": s["idx_in_processed_samples"],
                "removed_triples": [list(t) for t in present],
                "n_crucial": len(crucial),
                "n_crucial_absent": len(crucial) - len(present),
            })
            jobs.append((r["graph"], r["q_entity"], r["a_entity"], present))
        with open(os.path.join(out_dir, f"{level_tag(p)}.jsonl"), "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with Pool(args.n) as pool:
            rows = pool.map(analyse, jobs, chunksize=8)
        n = len(recs)
        lvl = {
            "questions": n,
            "rog_id_mismatch": id_mismatch,
            "questions_with_removal": sum(bool(r["removed_triples"]) for r in recs),
            "crucial_triples_total": sum(r["n_crucial"] for r in recs),
            "removed_from_rog_total": sum(len(r["removed_triples"]) for r in recs),
        }
        for key in rows[0]:
            lvl[key] = round(sum(r[key] for r in rows) / n, 4)
        stats["levels"][level_tag(p)] = lvl
        print(level_tag(p), json.dumps({k: v for k, v in lvl.items() if "within_3" not in k}))

    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    os.makedirs(os.path.join(ROOT, "results", "ikgqa_stats"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "ikgqa_stats", f"{args.dataset}.json"), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
