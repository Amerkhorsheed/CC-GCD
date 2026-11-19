"""Fit c(e) on Wikidata natural missingness (structural GBM, no relation id) and save it for decoding.

Trained on the train + calibration heads of scripts/observation_natural.py's head split (isotonic on
the calibration heads), saved to data/observation/wikidata_gbm_struct.joblib. Used by
scripts/run_e1_brink.py as ``CCe:wikidata_gbm_struct:<lambda>``.
"""

from __future__ import annotations

import os
import sys

import joblib
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from cc_gcd.calibration.observation_model import Encoded, ObservationModel, features  # noqa: E402
from observation_natural import head_split  # noqa: E402


def main(t0: str = "2019-08-26"):
    rows = [line.rstrip("\n").split("\t") for line in open(os.path.join(ROOT, "data", "processed", "wikidata",
                                                                       f"natural_t0_{t0}.tsv"))][1:]
    ref = [(h, r, t) for h, r, t, _, _ in rows]
    kept = np.array([int(x[3]) for x in rows], dtype=bool)
    lab = np.array([int(x[4]) for x in rows], dtype=bool)
    enc = Encoded(ref)
    X = features(enc, kept)
    split = head_split([h for h, _, _ in ref])
    tr, ca = lab & (split == 0), lab & (split == 1)
    m = ObservationModel("gbm_struct").fit(X[tr], enc.r[tr], kept[tr].astype(int), X[ca], enc.r[ca],
                                           kept[ca].astype(int), enc.R)
    out = os.path.join(ROOT, "data", "observation", "wikidata_gbm_struct.joblib")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    joblib.dump(m, out)
    print("saved", out, "train rate", round(float(kept[tr].mean()), 4))


if __name__ == "__main__":
    main()
