"""KGE candidate generator on IKGQA (GoG incomplete WebQSP), per deletion level.

Graph. IKGQA deletes per question, and ~62% of a level's deleted triples also occur
in another question's RoG subgraph, so the KGE and the generator use
``IKGQA.kge_graph()``: the union of the 1,000 questions' subgraphs minus every triple
deleted at any level (``scripts/train_kge.py --dataset ikgqa-webqsp``). Candidates
for a question are further filtered against that question's own observed subgraph.

Split. IKGQA has only GoG's 1,000 test questions, so choosing the generator mode and
(R, k) on them would tune on test. Questions with index % 5 == 0 (200) form a dev set
for selection; numbers are reported on the other 800 ("eval"). The configuration chosen
on BRINK (hybrid, R = 10, k = 5) is reported on eval as an untuned transfer.

Per level IKG-0/20/40/60/80 and configuration, with L = 2 inverse-augmented hops:
  answer reachability with observed edges only (must equal ikg_inverse_hit_within_2 in
  results/ikgqa_stats) and with candidates added at hop 1; the share reachable only via
  candidates; recovery of the question's deleted triples that touch a topic entity (the only
  ones hop-1 candidates can restore); candidate edges and trie walks per question.

    CUDA_VISIBLE_DEVICES=GPU-c5cd579e-52d6-31a3-0091-04ae52932736 HF_HOME=$PWD/.hf \
        python scripts/build_candidates_ikgqa.py --dataset webqsp
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

from cc_gcd.data.ikgqa import IKGQA, LEVELS, level_tag  # noqa: E402
from cc_gcd.graph.candidate_generator import CandidateGenerator  # noqa: E402
from cc_gcd.kge.complex_kge import ComplExScorer  # noqa: E402

MODES = {"hybrid+sig": ("hybrid", True), "kge+sig": ("kge", True), "kge_prob+sig": ("kge_prob", True),
         "cooc+sig": ("cooc", True)}
GRID_R = (1, 3, 5, 10)
GRID_K = (1, 5, 10, 20)
BRINK_CONFIG = ("hybrid+sig", "R10_k5")


def adjacency(graph):
    adj = defaultdict(list)
    for h, r, t in graph:
        h, t = h.strip(), t.strip()
        adj[h].append((r, t))
        adj[t].append(("~" + r, h))
    return adj


def neighbours(adj, nodes):
    out = set()
    for u in nodes:
        out.update(v for _, v in adj.get(u, ()))
    return out


def seed_incident(removed, seeds):
    """Deleted triples touching a topic entity, oriented from it (inverse-augmented)."""
    out = set()
    for h, r, t in removed:
        h, t = h.strip(), t.strip()
        if h in seeds:
            out.add((h, r, t))
        if t in seeds:
            out.add((t, "~" + r, h))
    return out


def prepare(ik):
    """Per (question, level): seeds, answers, observed adjacency, 2-hop reach, deleted triples."""
    rows = []
    for p in LEVELS:
        for qi, s in enumerate(ik.samples(p)):
            adj = adjacency(s["graph"])
            seeds = {e.strip() for e in s["q_entity"]}
            obs1 = neighbours(adj, seeds)
            obs_edges = {(h, r, v) for h in seeds for r, v in adj.get(h, ())}
            rows.append({
                "qi": qi, "level": p, "seeds": seeds, "answers": {a.strip() for a in s["a_entity"]},
                "adj": adj, "obs2": obs1 | neighbours(adj, obs1) | seeds, "obs_edges": obs_edges,
                "nodes": set(adj),
                "deleted_seed": seed_incident(s["removed_triples"], seeds),
                "walks_obs": sum(1 + len(adj.get(v, ())) for h in seeds for _, v in adj.get(h, ())),
            })
    return rows


def evaluate(gen, rows, R, k, subset, local=None):
    agg = defaultdict(lambda: defaultdict(float))
    for row in rows:
        if (row["qi"] % 5 == 0) != (subset == "dev"):
            continue
        L = level_tag(row["level"])
        a = agg[L]
        if local is None:
            c = [x for s in row["seeds"] for x in gen.candidates(s, R, k)]
        else:
            c = [x for s in row["seeds"] for x in gen.candidates_from(s, local[(row["qi"], row["level"])].get(s), R, k)]
        c = [x for x in c if (x[0], x[1], x[2].strip()) not in row["obs_edges"]]
        tails = {x[2].strip() for x in c}
        aug2 = row["obs2"] | tails | neighbours(row["adj"], tails)
        hit_o = bool(row["answers"] & row["obs2"])
        hit_a = bool(row["answers"] & aug2)
        cand = {(x[0], x[1], x[2].strip()) for x in c}
        # Gain split: candidates that restore this question's own deleted facts vs other (hypothetical) edges.
        restored = {t for (_, _, t) in cand & row["deleted_seed"]}
        hit_r = bool(row["answers"] & (row["obs2"] | restored | neighbours(row["adj"], restored)))
        a["n"] += 1
        a["answer_reach_restored_only"] += hit_r
        a["answer_reach_observed"] += hit_o
        a["answer_reach_augmented"] += hit_a
        a["answer_reach_only_via_candidates"] += hit_a and not hit_o
        if row["deleted_seed"]:
            a["q_with_seed_deletion"] += 1
            a["seed_deletion_recovered_q"] += bool(row["deleted_seed"] & cand)
            a["seed_deleted_triples"] += len(row["deleted_seed"])
            a["seed_deleted_triples_recovered"] += len(row["deleted_seed"] & cand)
        a["candidate_edges"] += len(c)
        a["candidates_deleted_for_q"] += len(row["deleted_seed"] & cand)
        a["walks_observed"] += row["walks_obs"]
        a["walks_augmented"] += row["walks_obs"] + sum(1 + len(row["adj"].get(t, ())) for t in (x[2].strip() for x in c))
    out = {}
    for L, a in agg.items():
        n = a["n"]
        out[L] = {
            "questions": int(n),
            "answer_reach_observed": round(a["answer_reach_observed"] / n, 4),
            "answer_reach_augmented": round(a["answer_reach_augmented"] / n, 4),
            "answer_reach_only_via_candidates": round(a["answer_reach_only_via_candidates"] / n, 4),
            # Observed + only the candidates that are this question's deleted facts (restored evidence).
            "answer_reach_restored_only": round(a["answer_reach_restored_only"] / n, 4),
            "questions_with_seed_incident_deletion": int(a["q_with_seed_deletion"]),
            "seed_deletion_recovered_share_of_those_q": round(a["seed_deletion_recovered_q"] / max(a["q_with_seed_deletion"], 1), 4),
            "seed_deleted_triple_recall": round(a["seed_deleted_triples_recovered"] / max(a["seed_deleted_triples"], 1), 4),
            "candidate_precision_vs_own_deletions": round(a["candidates_deleted_for_q"] / max(a["candidate_edges"], 1), 4),
            "candidate_edges_per_question": round(a["candidate_edges"] / n, 1),
            "walks_observed_per_question": round(a["walks_observed"] / n, 1),
            "walks_augmented_per_question": round(a["walks_augmented"] / n, 1),
        }
    deleted_levels = [L for L in out if L != "ikg0"]
    out["mean_ikg20_80"] = {key: round(float(np.mean([out[L][key] for L in deleted_levels])), 4)
                            for key in ("answer_reach_observed", "answer_reach_augmented",
                                        "answer_reach_only_via_candidates", "answer_reach_restored_only",
                                        "seed_deletion_recovered_share_of_those_q",
                                        "seed_deleted_triple_recall", "candidate_edges_per_question")}
    return out


def local_tables(gen, rows):
    """Per (question, level, seed): candidates whose tails lie in the question's observed subgraph."""
    out = {}
    for row in rows:
        out[(row["qi"], row["level"])] = {s: gen.local_table(s, row["nodes"]) for s in row["seeds"]}
    return out


def diagnostics(gen, rows):
    """Where seed-incident deleted triples are lost (distinct triples over all levels, eval subset)."""
    trip = {t for row in rows if row["qi"] % 5 != 0 for t in row["deleted_seed"]}
    rel_ranks, tail_ranks, oov = [], [], 0
    for u, r, v in sorted(trip):
        if u not in gen.sc.e2id or v not in gen.sc.e2id or r not in gen.sc.r2id:
            oov += 1
            continue
        rr = gen.relation_rank(u, r)
        tr = gen.tail_rank(u, r, v)
        rel_ranks.append(10 ** 6 if rr < 0 else rr)
        tail_ranks.append(10 ** 6 if tr < 0 else tr)
    rel_ranks, tail_ranks = np.array(rel_ranks), np.array(tail_ranks)
    return {
        "distinct_seed_incident_deleted_triples": len(trip),
        "not_in_kge_vocabulary": int(oov),
        "relation_in_top_R": {R: round(float((rel_ranks < R).mean()), 4) for R in GRID_R},
        "tail_in_top_k_given_relation": {k: round(float((tail_ranks < k).mean()), 4) for k in (1, 5, 10, 20, 50, 100)},
        "tail_masked_by_signature_filter": int((tail_ranks >= 10 ** 6).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="webqsp", choices=["webqsp"])
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    ap.add_argument("--rel_pool", type=int, default=50, help="KGE rankings score only the top-N cooc relations per entity")
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--kge", default=None, help="model dir (default data/kge/ikgqa-<dataset>)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="first N questions only (debugging)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--variants", nargs="+", default=["global", "local"], choices=["global", "local"],
                    help="global: tails from all entities (stored table); local: tails restricted to the "
                         "question's observed subgraph at that level")
    args = ap.parse_args()

    ik = IKGQA(os.path.join(ROOT, "data", "processed", "ikgqa"), args.dataset)
    graph_path = os.path.join(ik.root, "kge_graph.tsv")
    if not os.path.exists(graph_path):
        with open(graph_path, "w", encoding="utf-8") as f:
            for t in ik.kge_graph():
                f.write("\t".join(t) + "\n")
    with open(graph_path, encoding="utf-8") as f:
        triples = [tuple(line.rstrip("\n").split("\t")) for line in f if line.strip()]
    kge_dir = args.kge or os.path.join(ROOT, "data", "kge", f"ikgqa-{args.dataset}")
    scorer = ComplExScorer(kge_dir, device=args.device)
    t0 = time.time()
    rows = [r for r in prepare(ik) if args.limit is None or r["qi"] < args.limit]
    seeds = sorted({s for row in rows for s in row["seeds"]})
    print(f"prepared {len(rows)} (question, level) rows, {len(seeds)} topic entities, {time.time() - t0:.0f}s", flush=True)

    report = {"dataset": f"ikgqa-{args.dataset}", "hops": 2, "candidate_hops": [1],
              "kge": {k: v for k, v in json.load(open(os.path.join(kge_dir, "metrics.json"))).items() if k != "history"},
              "kge_graph_triples": len(triples), "rel_pool": args.rel_pool, "budget_R_times_k": args.budget,
              "split": "dev = question index % 5 == 0 (200), eval = the other 800", "modes": {}}
    for variant in args.variants:
        key_modes = "modes" if variant == "global" else "modes_local"
        report[key_modes] = {}
        for mode in args.modes:
            rel_rank, sig = MODES[mode]
            gen = CandidateGenerator(triples, scorer, rel_rank=rel_rank, signature_filter=sig, rel_pool=args.rel_pool)
            t0 = time.time()
            local = None
            if variant == "global":
                path = os.path.join(args.out or os.path.join(ROOT, "data", "candidates", f"ikgqa-{args.dataset}"), f"{mode}.npz")
                if os.path.exists(path) and not args.rebuild:
                    gen.load(path)
                else:
                    gen.build(entities=seeds)
                    gen.save(path)
            else:
                local = local_tables(gen, rows)
            m = {"build_seconds": round(time.time() - t0, 1)}
            for subset in ("dev", "eval"):
                m[subset] = {f"R{R}_k{k}": evaluate(gen, rows, R, k, subset, local) for R in GRID_R for k in GRID_K}
            if variant == "global":
                m["diagnostics_eval"] = diagnostics(gen, rows)
            report[key_modes][mode] = m
            e = m["eval"]["R10_k5"]["mean_ikg20_80"]
            print(variant, mode, f"{m['build_seconds']}s", json.dumps(e), flush=True)

    def ok(cfg):
        R, k = cfg[1:].split("_k")
        return int(R) * int(k) <= args.budget

    fams = [k for k in ("modes", "modes_local") if k in report]
    best = max(((fam, mode, cfg) for fam in fams for mode in report[fam] for cfg in report[fam][mode]["dev"] if ok(cfg)),
               key=lambda x: (report[x[0]][x[1]]["dev"][x[2]]["mean_ikg20_80"]["answer_reach_augmented"],
                              -report[x[0]][x[1]]["dev"][x[2]]["mean_ikg20_80"]["candidate_edges_per_question"]))
    report["selected_on_dev"] = {"criterion": "mean IKG-20..80 answer_reach_augmented, R*k <= budget",
                                 "variant": "global" if best[0] == "modes" else "local", "mode": best[1],
                                 "config": best[2], "eval": report[best[0]][best[1]]["eval"][best[2]]}
    # Secondary view (not the pre-set criterion): reachability through restored deleted facts only.
    best_r = max(((fam, mode, cfg) for fam in fams for mode in report[fam] for cfg in report[fam][mode]["dev"] if ok(cfg)),
                 key=lambda x: (report[x[0]][x[1]]["dev"][x[2]]["mean_ikg20_80"]["answer_reach_restored_only"],
                                -report[x[0]][x[1]]["dev"][x[2]]["mean_ikg20_80"]["candidate_edges_per_question"]))
    report["selected_on_dev_restored"] = {"criterion": "mean IKG-20..80 answer_reach_restored_only, R*k <= budget",
                                          "variant": "global" if best_r[0] == "modes" else "local", "mode": best_r[1],
                                          "config": best_r[2], "eval": report[best_r[0]][best_r[1]]["eval"][best_r[2]]}
    if "modes" in report and BRINK_CONFIG[0] in report["modes"]:
        report["brink_config_untuned"] = {"mode": BRINK_CONFIG[0], "config": BRINK_CONFIG[1],
                                          "eval": report["modes"][BRINK_CONFIG[0]]["eval"][BRINK_CONFIG[1]]}
    out = os.path.join(args.out, "report.json") if args.out else os.path.join(ROOT, "results", "candidates", f"ikgqa-{args.dataset}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print("selected on dev:", best, json.dumps(report["selected_on_dev"]["eval"]["mean_ikg20_80"]))


if __name__ == "__main__":
    main()
