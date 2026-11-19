"""Fit the KGE prior pi(e) for IKGQA-WebQSP (rank-free features: candidates there are subgraph-local).

Calibration as on BRINK (cc_gcd.calibration.kge_prior): candidates for a sample of the IKGQA
KGE's 2% holdout heads from a generator that does not observe the holdout; label = holdout fact.

    python scripts/calibrate_kge_prior_ikgqa.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from cc_gcd.calibration.kge_prior import calibrate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_heads", type=int, default=3000)
    ap.add_argument("--rel_rank", default="hybrid")
    args = ap.parse_args()
    graph = os.path.join(ROOT, "data", "processed", "ikgqa", "webqsp", "kge_graph.tsv")
    triples = [tuple(line.rstrip("\n").split("\t")) for line in open(graph, encoding="utf-8") if line.strip()]
    kge_dir = os.path.join(ROOT, "data", "kge", "ikgqa-webqsp")
    prior, report = calibrate(triples, kge_dir, rel_rank=args.rel_rank, R=10, k=20, rel_pool=50, features="rankfree",
                              max_heads=args.max_heads, device=args.device)
    json.dump(prior.to_dict(), open(os.path.join(kge_dir, "pi_prior.json"), "w"))
    os.makedirs(os.path.join(ROOT, "results", "kge_prior"), exist_ok=True)
    json.dump(report, open(os.path.join(ROOT, "results", "kge_prior", "ikgqa-webqsp.json"), "w"), indent=1)
    t = report["test"]
    print(f"ikgqa-webqsp: {report['candidates']} candidates, {report['positives']} positives (base {report['base_rate']}) "
          f"| test ECE {t['ece']} AUC {t['auc']} Brier {t['brier']}")


if __name__ == "__main__":
    main()
