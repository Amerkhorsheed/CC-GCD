"""Build the KGE candidate table for a BRINK graph and measure oracle recall (plan sections 5.1, 5.3, 9.2).

For each generator mode (relation ranking x signature filter) this script
  1. precomputes top-R_max relations x top-k_max absent tails for every entity (GPU, once);
  2. on val and test questions, with L = 2 inverse-augmented hops, reports for a grid of (R, k):
       - hard-answer / any-answer reachability with observed edges only (hard KG-Trie ceiling)
         and with observed + candidate edges (oracle recall of the trie the decoder will see),
       - the share reachable only through candidate (absent) edges,
       - hard-triple recovery: the removed fact linking topic and hard answer is itself a candidate,
       - candidate edges added and trie walks (paths <= L) before / after, i.e. the search-space cost,
       - candidate precision against BRINK's complete graph (a lower bound: that graph is incomplete too);
  3. picks the configuration on val (best hard-triple recovery with R * k <= budget) and reports it on test.

Candidates come from the observed graph and a KGE trained on it only. Removed triples and question
answers are read for evaluation, never for building candidates.

    CUDA_VISIBLE_DEVICES=GPU-c5cd579e-52d6-31a3-0091-04ae52932736 \
        python scripts/build_candidates.py --dataset fb15k237
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from cc_gcd.data.brink import BrinkKG  # noqa: E402
from cc_gcd.graph.candidate_generator import CandidateGenerator  # noqa: E402
from cc_gcd.kge.complex_kge import ComplExScorer, augment  # noqa: E402

MODES = {"kge_prob+sig": ("kge_prob", True), "hybrid+sig": ("hybrid", True), "cooc+sig": ("cooc", True),
         "kge+sig": ("kge", True), "kge_prob": ("kge_prob", False)}
GRID_R = (1, 2, 3, 5, 10)
GRID_K = (1, 5, 10, 20)


def neighbours(adj, nodes):
    out = set()
    for u in nodes:
        out.update(v for _, v in adj.get(u, ()))
    return out


def hard_edges(q):
    """Removed facts linking a topic entity and the hard answer, oriented from the topic (inverse-augmented)."""
    seeds = set(q["q_entity"])
    out = set()
    for h, r, t in q["hard_triples"]:
        if h in seeds:
            out.add((h, r, t))
        if t in seeds:
            out.add((t, "~" + r, h))
    return sorted(out)


def evaluate(gen, kg, questions, removed_aug, R, k, cand_hops):
    T = gen.table
    ncand = (T["tail"][:, :R, :k] >= 0).sum((1, 2))
    e2id = gen.sc.e2id
    deg = defaultdict(int, {u: len(v) for u, v in kg.adj.items()})

    def nc(u):
        i = e2id.get(u)
        return int(ncand[i]) if i is not None else 0

    agg = defaultdict(float)
    for q in questions:
        seeds = set(q["q_entity"])
        hard, gold = set(q["hard_answer"]), set(q["a_entity"])
        c0 = [c for s in seeds for c in gen.candidates(s, R, k)]
        obs1 = neighbours(kg.adj, seeds)
        obs2 = obs1 | neighbours(kg.adj, obs1)
        aug1 = obs1 | {c[2] for c in c0}
        aug2 = aug1 | neighbours(kg.adj, aug1)
        n_edges = len(c0)
        hop2 = []
        if 2 in cand_hops:
            for u in aug1:
                cu = gen.candidates(u, R, k)
                hop2.extend(cu)
                aug2.update(c[2] for c in cu)
            n_edges += len(hop2)
        walks_obs = sum(1 + deg[v] for s in seeds for _, v in kg.adj.get(s, ()))
        if 2 in cand_hops:
            walks_aug = sum(1 + deg[v] + nc(v) for s in seeds for v in [v for _, v in kg.adj.get(s, ())] + [c[2] for c in c0 if c[0] == s])
        else:
            walks_aug = walks_obs + sum(1 + deg[c[2]] for c in c0)
        cands = {(c[0], c[1], c[2]) for c in c0}
        agg["hard_reach_observed"] += bool(hard & obs2)
        agg["hard_reach_augmented"] += bool(hard & aug2)
        agg["hard_reach_only_via_candidates"] += bool(hard & aug2) and not (hard & obs2)
        agg["any_reach_observed"] += bool(gold & obs2)
        agg["any_reach_augmented"] += bool(gold & aug2)
        agg["hard_triple_recovered"] += any(e in cands for e in hard_edges(q))
        agg["seed_candidates_true"] += sum((c[0], c[1], c[2]) in removed_aug for c in c0)
        agg["seed_candidates"] += len(c0)
        agg["candidate_edges"] += n_edges
        agg["walks_observed"] += walks_obs
        agg["walks_augmented"] += walks_aug
    n = len(questions)
    res = {key: round(agg[key] / n, 4) for key in ("hard_reach_observed", "hard_reach_augmented",
                                                   "hard_reach_only_via_candidates", "any_reach_observed",
                                                   "any_reach_augmented", "hard_triple_recovered")}
    res["seed_candidate_precision"] = round(agg["seed_candidates_true"] / max(agg["seed_candidates"], 1), 4)
    res["candidate_edges_per_question"] = round(agg["candidate_edges"] / n, 1)
    res["walks_observed_per_question"] = round(agg["walks_observed"] / n, 1)
    res["walks_augmented_per_question"] = round(agg["walks_augmented"] / n, 1)
    return res


def diagnostics(gen, questions):
    """Where hard triples are lost: relation ranking, tail ranking, or the signature filter."""
    rel_ranks, tail_ranks, filtered, oov = [], [], 0, 0
    for q in questions:
        best_rel, best_tail = None, None
        for u, r, v in hard_edges(q):
            if u not in gen.sc.e2id or v not in gen.sc.e2id or r not in gen.sc.r2id:
                oov += 1  # e.g. an answer that only occurs in facts missing from the observed graph
                continue
            rr = gen.relation_rank(u, r)
            tr = gen.tail_rank(u, r, v)
            filtered += tr < 0
            rr = 10 ** 6 if rr < 0 else rr
            tr = 10 ** 6 if tr < 0 else tr
            best_rel = rr if best_rel is None else min(best_rel, rr)
            best_tail = tr if best_tail is None else min(best_tail, tr)
        if best_rel is not None:
            rel_ranks.append(best_rel)
            tail_ranks.append(best_tail)
    rel_ranks, tail_ranks = np.array(rel_ranks), np.array(tail_ranks)
    return {
        "questions_with_hard_triple": len(rel_ranks),
        "hard_relation_in_top_R": {R: round(float((rel_ranks < R).mean()), 4) for R in GRID_R},
        # Rank of the hard answer among admissible absent tails, given the correct relation (filtered).
        "hard_tail_in_top_k_given_relation": {k: round(float((tail_ranks < k).mean()), 4) for k in (1, 5, 10, 20, 50, 100)},
        "hard_tail_mrr_given_relation": round(float(np.mean(1.0 / (1 + tail_ranks))), 4),
        "hard_triples_removed_by_signature_filter": int(filtered),
        "hard_triples_outside_kge_vocabulary": int(oov),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["family", "fb15k237", "wikidata"], required=True)
    ap.add_argument("--kge", default=None, help="model dir (default data/kge/brink-<dataset>)")
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    ap.add_argument("--budget", type=int, default=50, help="max R * k for the val-selected configuration")
    ap.add_argument("--R_max", type=int, default=10)
    ap.add_argument("--k_max", type=int, default=20)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--kge_tag", default="", help="alternate KGE, e.g. s1 -> data/kge/brink-<dataset>-s1")
    args = ap.parse_args()
    tag = args.dataset + (f"-{args.kge_tag}" if args.kge_tag else "")

    root = os.path.join(ROOT, "data", "processed", "brink", args.dataset)
    kg = BrinkKG(root, version="incomplete", inverse=True)
    kg_dir = BrinkKG(root, version="incomplete", inverse=False)
    removed = [tuple(x) for x in (line.rstrip("\n").split("\t") for line in open(os.path.join(root, "removed.tsv"))) if len(x) == 3]
    removed_aug = set(augment(removed))
    kge_dir = args.kge or os.path.join(ROOT, "data", "kge", f"brink-{tag}")
    scorer = ComplExScorer(kge_dir)
    qs = {s: list(kg.questions(s)) for s in ("val", "test")}

    report = {"dataset": f"brink-{args.dataset}", "graph": "kg_incomplete (observed) + inverse edges", "hops": 2,
              "kge": json.load(open(os.path.join(kge_dir, "metrics.json"))), "budget_R_times_k": args.budget,
              "modes": {}}
    for mode in args.modes:
        rel_rank, sig = MODES[mode]
        gen = CandidateGenerator(kg_dir.triples, scorer, rel_rank=rel_rank, signature_filter=sig,
                                 R_max=args.R_max, k_max=args.k_max)
        path = os.path.join(ROOT, "data", "candidates", f"brink-{tag}", f"{mode}.npz")
        t0 = time.time()
        if os.path.exists(path) and not args.rebuild:
            gen.load(path)
        else:
            gen.build()
            gen.save(path)
        build_s = round(time.time() - t0, 1)
        T = gen.table
        stored = T["tail"] >= 0
        h_ids = np.repeat(np.arange(gen.E), stored.sum((1, 2)))
        rr = np.broadcast_to(T["rel"][:, :, None], T["tail"].shape)[stored]
        tt = T["tail"][stored]
        ents, rels = scorer.entities, scorer.relations
        global_prec = float(np.mean([(ents[h], rels[r], ents[t]) in removed_aug for h, r, t in zip(h_ids, rr, tt)]))
        m = {"build_seconds": build_s, "stored_candidate_edges": int(stored.sum()),
             "stored_candidate_precision_vs_complete": round(global_prec, 5), "splits": {}}
        for split, questions in qs.items():
            grid = {f"R{R}_k{k}": evaluate(gen, kg, questions, removed_aug, R, k, {1}) for R in GRID_R for k in GRID_K}
            grid["R5_k10_hops12"] = evaluate(gen, kg, questions, removed_aug, 5, 10, {1, 2})
            m["splits"][split] = {"questions": len(questions), "grid_hop1": grid, "diagnostics": diagnostics(gen, questions)}
            print(mode, split, json.dumps(grid["R5_k10"]), flush=True)
        report["modes"][mode] = m

    # Selection on val only: best hard-triple recovery within the budget; ties -> fewer candidate edges.
    best = max(((mode, cfg) for mode in report["modes"] for cfg in report["modes"][mode]["splits"]["val"]["grid_hop1"]
                if cfg.startswith("R") and "hops" not in cfg
                and int(cfg[1:].split("_k")[0]) * int(cfg.split("_k")[1]) <= args.budget),
               key=lambda x: (report["modes"][x[0]]["splits"]["val"]["grid_hop1"][x[1]]["hard_triple_recovered"],
                              -report["modes"][x[0]]["splits"]["val"]["grid_hop1"][x[1]]["candidate_edges_per_question"]))
    report["selected_on_val"] = {"mode": best[0], "config": best[1],
                                 "val": report["modes"][best[0]]["splits"]["val"]["grid_hop1"][best[1]],
                                 "test": report["modes"][best[0]]["splits"]["test"]["grid_hop1"][best[1]]}
    print("selected:", json.dumps(report["selected_on_val"], indent=1))
    out = os.path.join(ROOT, "results", "candidates", f"brink-{tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
