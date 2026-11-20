"""Fit the calibrated KGE prior pi(e) on BRINK (cc_gcd.calibration.kge_prior) and check it on BRINK's removals.

Calibration: candidates for the KGE's 2% MCAR holdout heads, from a generator that does not
observe the holdout; label = candidate is a holdout fact. Independent check: on the val/test
questions, the candidates the deployed generator (full observed graph) proposes at the topic
entities, scored with pi; BRINK's removed hard facts among them are true, the rest unknown.

    CUDA_VISIBLE_DEVICES=GPU-c5cd579e-52d6-31a3-0091-04ae52932736 python scripts/calibrate_kge_prior.py --dataset fb15k237
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from cc_gcd.calibration.kge_prior import calibrate, candidate_features, design  # noqa: E402
from cc_gcd.calibration.observation_model import calibration_metrics  # noqa: E402
from cc_gcd.data.brink import BrinkKG  # noqa: E402
from cc_gcd.graph.candidate_generator import CandidateGenerator  # noqa: E402
from cc_gcd.kge.complex_kge import ComplExScorer  # noqa: E402

MODES = {"family": "kge", "fb15k237": "hybrid", "wikidata": "hybrid"}  # generator modes selected on val (results/candidates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["family", "fb15k237", "wikidata"], default="fb15k237")
    ap.add_argument("--R", type=int, default=10)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--kge_tag", default="", help="alternate KGE, e.g. s1 -> data/kge/brink-<dataset>-s1")
    ap.add_argument("--seed", type=int, default=0, help="the KGE's training seed (its holdout split)")
    args = ap.parse_args()
    tag = args.dataset + (f"-{args.kge_tag}" if args.kge_tag else "")
    root = os.path.join(ROOT, "data", "processed", "brink", args.dataset)
    kg = BrinkKG(root, version="incomplete", inverse=False)
    kge_dir = os.path.join(ROOT, "data", "kge", f"brink-{tag}")
    prior, report = calibrate(kg.triples, kge_dir, rel_rank=MODES[args.dataset], R=args.R, k=args.k,
                              split_seed=args.seed)
    json.dump(prior.to_dict(), open(os.path.join(kge_dir, "pi_prior.json"), "w"))

    # Independent check on BRINK's removed hard facts (val + test topic entities).
    scorer = ComplExScorer(kge_dir)
    gen = CandidateGenerator(kg.triples, scorer, rel_rank=MODES[args.dataset], R_max=args.R, k_max=args.k)
    qs = list(kg.questions("val")) + list(kg.questions("test"))
    seeds = sorted({s for q in qs for s in q["q_entity"]})
    table = gen.build(entities=seeds)
    row = {int(e): j for j, e in enumerate(table["ent"])}
    ents, rels = scorer.entities, scorer.relations
    hard = set()
    for q in qs:
        s = set(q["q_entity"])
        for h, r, t in q["hard_triples"]:
            if h in s:
                hard.add((h, r, t))
            if t in s:
                hard.add((t, "~" + r, h))
    feats, lab = [], []
    for u in seeds:
        j = row.get(scorer.e2id.get(u, -1))
        if j is None:
            continue
        for f in candidate_features(table, j, args.R, args.k):
            feats.append(f)
            lab.append((u, rels[f[0]], ents[f[1]]) in hard)
    pi = prior.predict(design(feats))
    lab = np.array(lab, dtype=int)
    report["brink_removed_check"] = {k: v for k, v in calibration_metrics(pi, lab).items() if k != "reliability"}
    report["brink_removed_check"].update({"candidates": int(len(lab)), "removed_facts_among_them": int(lab.sum()),
                                          "mean_pi_removed": round(float(pi[lab == 1].mean()), 4),
                                          "mean_pi_other": round(float(pi[lab == 0].mean()), 4)})
    out = os.path.join(ROOT, "results", "kge_prior", f"brink-{tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w"), indent=1)
    t = report["test"]
    b = report["brink_removed_check"]
    print(f"{args.dataset}: calibration candidates {report['candidates']} (pos {report['positives']}, base "
          f"{report['base_rate']}) | test ECE {t['ece']} AUC {t['auc']} Brier {t['brier']} | BRINK removed: AUC "
          f"{b['auc']}, mean pi removed {b['mean_pi_removed']} vs other {b['mean_pi_other']}")


if __name__ == "__main__":
    main()
