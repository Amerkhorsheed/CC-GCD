"""Observation model c(e) on BRINK: mechanism-transfer matrix, leave-one-mechanism-out, BRINK's own missingness.

Reference graph: BRINK's observed (incomplete) graph. For each controlled mechanism
(``cc_gcd.calibration.missingness``) and seed, the reference triples are masked; the
label of a triple is whether it survived, and its features are computed on the masked
graph without it. Triples are split once into train / calibration / test (60/20/20).

Reported (plan 6.5), averaged over seeds:
  matrix   train on mechanism i (train split), isotonic on i (calibration split),
           test on mechanism j (test split); the result that counts is off-diagonal;
  lomo     train and calibrate on all mechanisms but j, test on j;
  bounds   constant c (SCAR, baseline B5) and the exact oracle c (B6) of mechanism j;
  level    diagnostic: each prediction shifted (one logit constant) to the target's true mean, vs the
           constant at that mean; isolates the per-edge shape of c from its level, which no
           estimator can identify from an incomplete graph (LOMO and BRINK only);
  brink    every model trained on the synthetic mechanisms, tested on BRINK's rule-based
           missingness (complete vs incomplete graph). BRINK is never used for training or
           calibration (plan 6.2), so this is a held-out, unseen mechanism.
Metrics: Brier, ECE (15 equal-mass bins), debiased l2-ECE, AUC, mean predicted vs observed.

    python scripts/observation_lomo.py --dataset fb15k237
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from cc_gcd.calibration.missingness import MECHANISMS, retention, sample  # noqa: E402
from cc_gcd.calibration.observation_model import (Encoded, ObservationModel, calibration_metrics, features,  # noqa: E402
                                                  match_level)
from cc_gcd.data.brink import BrinkKG  # noqa: E402

MODELS = ("scar", "relation", "logreg", "gbm")
KEYS = ("brier", "ece", "ece_debiased_l2", "auc", "mean_pred", "observed_rate")


def mean_std(rows):
    out = {}
    for k in KEYS:
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = round(float(v.mean()), 5)
        out[k + "_std"] = round(float(v.std()), 5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["family", "fb15k237"], default="fb15k237")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    root = os.path.join(ROOT, "data", "processed", "brink", args.dataset)
    ref = sorted(set(BrinkKG(root, version="incomplete", inverse=False).triples))
    complete = sorted(set(BrinkKG(root, version="complete", inverse=False).triples))
    train_q = list(BrinkKG(root, version="incomplete", inverse=False).questions("train"))
    enc = Encoded(ref)
    rng = np.random.default_rng(0)
    perm = rng.permutation(enc.n)
    tr, ca, te = perm[:int(0.6 * enc.n)], perm[int(0.6 * enc.n):int(0.8 * enc.n)], perm[int(0.8 * enc.n):]

    # BRINK's own missingness: reference = complete graph, observed = incomplete graph.
    enc_c = Encoded(complete)
    inc = set(ref)
    kept_nat = np.array([t in inc for t in complete])
    X_nat = features(enc_c, kept_nat)
    rels_c = sorted(enc_c.r2id, key=enc_c.r2id.get)
    rel_nat = np.array([enc.r2id.get(rels_c[i], -1) for i in enc_c.r], dtype=np.int64)
    assert (rel_nat >= 0).all(), "a complete-graph relation is missing from the incomplete graph"

    t0 = time.time()
    data = {}  # (mechanism, seed) -> (p_keep, kept, X)
    for m in MECHANISMS:
        p = retention(m, ref, questions=train_q, seed=0)
        for s in range(args.seeds):
            kept = sample(p, seed=1000 * s + MECHANISMS.index(m))
            data[(m, s)] = (p, kept, features(enc, kept))
    print(f"masked {len(MECHANISMS)} mechanisms x {args.seeds} seeds in {time.time() - t0:.0f}s", flush=True)

    report = {"dataset": f"brink-{args.dataset}", "reference": "kg_incomplete (observed graph)",
              "reference_triples": enc.n, "split": "random triples 60/20/20 (train / isotonic / test)",
              "seeds": args.seeds, "mechanisms": {}, "matrix": {}, "lomo": {}, "brink_missingness": {}}
    for m in MECHANISMS:
        p, kept, _ = data[(m, 0)]
        report["mechanisms"][m] = {"mean_retention": round(float(p.mean()), 4),
                                   "retention_min": round(float(p.min()), 4), "retention_max": round(float(p.max()), 4),
                                   "oracle_test": mean_std([calibration_metrics(data[(m, s)][0][te], data[(m, s)][1][te])
                                                            for s in range(args.seeds)])}

    def fit(kind, train_mechs, s):
        X = np.concatenate([data[(m, s)][2][tr] for m in train_mechs])
        y = np.concatenate([data[(m, s)][1][tr] for m in train_mechs]).astype(int)
        Xc = np.concatenate([data[(m, s)][2][ca] for m in train_mechs])
        yc = np.concatenate([data[(m, s)][1][ca] for m in train_mechs]).astype(int)
        rel = np.tile(enc.r[tr], len(train_mechs))
        relc = np.tile(enc.r[ca], len(train_mechs))
        return ObservationModel(kind, seed=s).fit(X, rel, y, Xc, relc, yc, enc.R)

    for kind in MODELS:
        t0 = time.time()
        mat = {mi: {mj: [] for mj in MECHANISMS} for mi in MECHANISMS}
        lomo = {mj: [] for mj in MECHANISMS}
        lomo_lvl = {mj: [] for mj in MECHANISMS}
        nat = {mi: [] for mi in list(MECHANISMS) + ["all"]}
        nat_lvl = []
        for s in range(args.seeds):
            for mi in MECHANISMS:
                model = fit(kind, [mi], s)
                for mj in MECHANISMS:
                    _, kept, X = data[(mj, s)]
                    mat[mi][mj].append(calibration_metrics(model.predict(X[te], enc.r[te]), kept[te]))
                nat[mi].append(calibration_metrics(model.predict(X_nat, rel_nat), kept_nat))
            for mj in MECHANISMS:
                model = fit(kind, [m for m in MECHANISMS if m != mj], s)
                _, kept, X = data[(mj, s)]
                pred = model.predict(X[te], enc.r[te])
                lomo[mj].append(calibration_metrics(pred, kept[te]))
                lomo_lvl[mj].append(calibration_metrics(match_level(pred, kept[te].mean()), kept[te]))
            model = fit(kind, list(MECHANISMS), s)
            pred = model.predict(X_nat, rel_nat)
            nat["all"].append(calibration_metrics(pred, kept_nat))
            nat_lvl.append(calibration_metrics(match_level(pred, kept_nat.mean()), kept_nat))
        report["matrix"][kind] = {mi: {mj: mean_std(v) for mj, v in row.items()} for mi, row in mat.items()}
        off = [x for mi in MECHANISMS for mj in MECHANISMS if mi != mj for x in mat[mi][mj]]
        diag = [x for m in MECHANISMS for x in mat[m][m]]
        report["matrix"][kind]["off_diagonal_mean"] = mean_std(off)
        report["matrix"][kind]["diagonal_mean"] = mean_std(diag)
        report["lomo"][kind] = {mj: mean_std(v) for mj, v in lomo.items()}
        report["lomo"][kind]["mean"] = mean_std([x for v in lomo.values() for x in v])
        report["brink_missingness"][kind] = {mi: mean_std(v) for mi, v in nat.items()}
        report.setdefault("level_matched", {})[kind] = {
            "lomo": {mj: mean_std(v) for mj, v in lomo_lvl.items()},
            "lomo_mean": mean_std([x for v in lomo_lvl.values() for x in v]),
            "brink_all": mean_std(nat_lvl)}
        o, l = report["matrix"][kind]["off_diagonal_mean"], report["lomo"][kind]["mean"]
        print(f"{kind:8s} off-diag ECE {o['ece']:.4f} Brier {o['brier']:.4f} AUC {o['auc']:.3f} | LOMO ECE {l['ece']:.4f} "
              f"AUC {l['auc']:.3f} | BRINK (all mechs) ECE {report['brink_missingness'][kind]['all']['ece']:.4f} "
              f"AUC {report['brink_missingness'][kind]['all']['auc']:.3f} | level-matched Brier LOMO "
              f"{report['level_matched'][kind]['lomo_mean']['brier']:.4f} BRINK {report['level_matched'][kind]['brink_all']['brier']:.5f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
    report["oracle_mean"] = mean_std([report["mechanisms"][m]["oracle_test"] for m in MECHANISMS])
    # Reliability tables are kept only for LOMO and BRINK to keep the file small.
    for kind in MODELS:
        for mi in MECHANISMS:
            for mj in MECHANISMS:
                report["matrix"][kind][mi][mj].pop("reliability", None)
    out = os.path.join(ROOT, "results", "observation", f"brink-{args.dataset}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
