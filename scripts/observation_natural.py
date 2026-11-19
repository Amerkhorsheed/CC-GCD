"""Observation model c(e) on natural missingness: Wikidata 2019-08-26 vs 2026-11-17 (plan 6.2, H3).

Rows: item-valued facts among the Wikidata items of BRINK-FB15k-237's entities
(``data_prep/wikidata_snapshots.py build-dump``). A fact true now and not ruled out as
"not yet true at T0" is labelled recorded (in the 2019 dump) or unrecorded. Facts present in
2019 but gone now stay in the observed 2019 graph for features but carry no label.
Features are the cross-fitted structural features of ``cc_gcd.calibration.observation_model``,
computed on the 2019 graph. Splits are by head entity (60/20/20), so test heads are unseen.

Reported:
  natural          c trained and tested on natural labels (constant, per-relation, logistic,
                   GBM, GBM without relation id), plus the level-matched diagnostic;
  synthetic        the fallback route: mask the 2019 graph with mechanisms A-C (D needs questions),
                   train c on the synthetic labels, test on the natural ones;
  to_brink         c trained on natural Wikidata labels (structural features only), tested on
                   BRINK-FB15k-237's rule-based missingness (complete vs incomplete graph);
  relations        recorded rate of the most frequent relations.

    python scripts/observation_natural.py --t0 2019-08-26
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from cc_gcd.calibration.missingness import retention, sample  # noqa: E402
from cc_gcd.calibration.observation_model import (Encoded, ObservationModel, calibration_metrics, features,  # noqa: E402
                                                  match_level)
from cc_gcd.data.brink import BrinkKG  # noqa: E402

NATURAL_MODELS = ("scar", "relation", "logreg", "gbm", "gbm_struct")
SYNTHETIC = ("A_mcar", "B_relation", "C_degree")


def head_split(heads, seed=0):
    """0 train / 1 calibration / 2 test, by a hash of the head entity."""
    out = np.empty(len(heads), dtype=np.int8)
    for i, h in enumerate(heads):
        u = int(hashlib.md5(f"{seed}:{h}".encode()).hexdigest()[:8], 16) / 2**32
        out[i] = 0 if u < 0.6 else (1 if u < 0.8 else 2)
    return out


def evaluate(p, y):
    m = calibration_metrics(p, y)
    m["level_matched"] = {k: v for k, v in calibration_metrics(match_level(p, y.mean()), y).items() if k != "reliability"}
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", default="2019-08-26")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    path = os.path.join(ROOT, "data", "processed", "wikidata", f"natural_t0_{args.t0}.tsv")
    rows = [line.rstrip("\n").split("\t") for line in open(path)][1:]
    ref = [(h, r, t) for h, r, t, _, _ in rows]
    kept = np.array([int(x[3]) for x in rows], dtype=bool)
    lab = np.array([int(x[4]) for x in rows], dtype=bool)
    rels = sorted({r for _, r, _ in ref})
    enc = Encoded(ref, relations=rels)
    t_start = time.time()
    X = features(enc, kept)
    split = head_split([h for h, _, _ in ref])
    tr, ca, te = (lab & (split == k) for k in (0, 1, 2))
    y = kept.astype(int)
    print(f"{enc.n} rows ({lab.sum()} labelled, recorded rate {kept[lab].mean():.4f}), {len(rels)} relations, "
          f"features {time.time() - t_start:.0f}s", flush=True)

    report = {"t0": args.t0, "rows": int(enc.n), "labelled": int(lab.sum()),
              "recorded_rate_labelled": round(float(kept[lab].mean()), 4),
              "split": "by head entity 60/20/20 (md5)", "test_rows": int(te.sum()), "natural": {}, "synthetic": {},
              "to_brink": {}}

    # ---- natural: train and test on natural labels
    for kind in NATURAL_MODELS:
        m = ObservationModel(kind).fit(X[tr], enc.r[tr], y[tr], X[ca], enc.r[ca], y[ca], enc.R)
        report["natural"][kind] = evaluate(m.predict(X[te], enc.r[te]), kept[te])
        r = report["natural"][kind]
        print(f"natural  {kind:10s} ECE {r['ece']:.4f} Brier {r['brier']:.4f} AUC {r['auc']:.3f} | level-matched Brier "
              f"{r['level_matched']['brier']:.4f}", flush=True)

    # ---- synthetic -> natural: mask the observed 2019 graph, learn c, test on natural labels
    obs_idx = np.where(kept)[0]
    ref_syn = [ref[i] for i in obs_idx]
    enc_syn = Encoded(ref_syn, relations=rels)
    split_syn = split[obs_idx]
    for kind in ("scar", "relation", "logreg", "gbm"):
        res = {}
        for mech in list(SYNTHETIC) + ["pooled"]:
            runs = []
            for s in range(args.seeds):
                parts = []
                for mname in (SYNTHETIC if mech == "pooled" else (mech,)):
                    keep = sample(retention(mname, ref_syn, seed=s), seed=100 * s + SYNTHETIC.index(mname))
                    parts.append((features(enc_syn, keep), keep.astype(int)))
                Xs = np.concatenate([p[0] for p in parts])
                ys = np.concatenate([p[1] for p in parts])
                rs = np.tile(enc_syn.r, len(parts))
                sp = np.tile(split_syn, len(parts))
                m = ObservationModel(kind, seed=s).fit(Xs[sp == 0], rs[sp == 0], ys[sp == 0], Xs[sp == 1], rs[sp == 1],
                                                       ys[sp == 1], enc.R)
                runs.append(evaluate(m.predict(X[te], enc.r[te]), kept[te]))
            res[mech] = {k: round(float(np.mean([r[k] for r in runs])), 5)
                         for k in ("brier", "ece", "ece_debiased_l2", "auc", "mean_pred", "observed_rate")}
            res[mech]["level_matched_brier"] = round(float(np.mean([r["level_matched"]["brier"] for r in runs])), 5)
        report["synthetic"][kind] = res
        print(f"synth->natural {kind:8s} " + " ".join(f"{k}: ECE {v['ece']:.3f} AUC {v['auc']:.3f}" for k, v in res.items()),
              flush=True)

    # ---- natural -> BRINK-FB15k-237 missingness (structural features; relation ids do not transfer)
    root = os.path.join(ROOT, "data", "processed", "brink", "fb15k237")
    complete = sorted(set(BrinkKG(root, version="complete", inverse=False).triples))
    inc = set(BrinkKG(root, version="incomplete", inverse=False).triples)
    enc_c = Encoded(complete)
    kept_b = np.array([t in inc for t in complete])
    Xb = features(enc_c, kept_b)
    for kind in ("scar", "logreg", "gbm_struct"):
        m = ObservationModel(kind).fit(X[tr], enc.r[tr], y[tr], X[ca], enc.r[ca], y[ca], enc.R)
        report["to_brink"][kind] = evaluate(m.predict(Xb, np.zeros(enc_c.n, dtype=np.int64)), kept_b)
        r = report["to_brink"][kind]
        print(f"natural->BRINK {kind:10s} ECE {r['ece']:.4f} AUC {r['auc']:.3f} | level-matched Brier "
              f"{r['level_matched']['brier']:.5f}", flush=True)

    # ---- relations
    cnt = Counter(enc.r[lab].tolist())
    rec = Counter(enc.r[lab & kept].tolist())
    report["relations"] = [{"relation": rels[r], "labelled": n, "recorded_rate": round(rec[r] / n, 4)}
                           for r, n in cnt.most_common(25)]
    out = os.path.join(ROOT, "results", "observation", f"wikidata-natural-{args.t0}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
