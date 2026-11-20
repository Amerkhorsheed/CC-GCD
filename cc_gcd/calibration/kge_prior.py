"""Calibrated plausibility prior pi(e) = P(T_e = 1 | x_e) for candidate edges (plan section 4.1).

pi is calibrated on candidates whose truth is known by construction. The KGE is trained on
98% of the observed graph (``complex_kge.train_complex``, seed-0 MCAR holdout of 2%), so the
holdout facts are true and unseen. A generator whose *observed* graph is only the 98% training
part proposes candidates for the holdout heads; a candidate is labelled true iff it is a holdout
fact. pi is therefore relative to the reference graph (plan 2: "relative coverage"): a true
but unrecorded fact counts as false, so pi is a lower bound on plausibility.

Features of a candidate (u, r, v): log P(v | u, r) under the KGE softmax, the raw score, log
tail rank and log relation rank in the generator's table. Model: logistic regression, then
isotonic regression on a separate, head-disjoint split; clipped to [1e-4, 1 - 1e-4].
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np

from cc_gcd.graph.candidate_generator import CandidateGenerator
from cc_gcd.kge.complex_kge import ComplExScorer, augment

Triple = Tuple[str, str, str]
PI_MIN, PI_MAX = 1e-4, 1 - 1e-4


def kge_holdout_split(triples: Sequence[Triple], valid_frac: float = 0.02, seed: int = 0):
    """Reproduces the KGE trainer's split exactly (train_complex): (train, holdout) canonical triples."""
    canon = sorted(set(map(tuple, triples)))
    rng = random.Random(seed)
    valid_idx = set(rng.sample(range(len(canon)), int(valid_frac * len(canon))))
    return [t for i, t in enumerate(canon) if i not in valid_idx], [t for i, t in enumerate(canon) if i in valid_idx]


def candidate_features(table: dict, j: int, R: int, k: int) -> List[Tuple[int, int, int, int, float, float]]:
    """(relation id, tail id, relation rank, tail rank, raw score, log p) for one table row."""
    out = []
    for a in range(min(R, table["rel"].shape[1])):
        ri = int(table["rel"][j, a])
        if ri < 0:
            continue
        for b in range(min(k, table["tail"].shape[2] if table["tail"].ndim == 3 else table["tail"].shape[1])):
            vi = int(table["tail"][j, a, b]) if table["tail"].ndim == 3 else int(table["tail"][a, b])
            if vi < 0:
                break
            sc = float(table["tail_score"][j, a, b]) if table["tail_score"].ndim == 3 else float(table["tail_score"][a, b])
            lp = float(table["tail_logp"][j, a, b]) if table["tail_logp"].ndim == 3 else float(table["tail_logp"][a, b])
            out.append((ri, vi, a, b, sc, lp))
    return out


def design(rows: Sequence[Tuple[int, int, int, int, float, float]], features: str = "all") -> np.ndarray:
    """"all": [log p, raw score, log1p tail rank, log1p relation rank]; "rankfree": [log p, raw score]
    (for subgraph-local candidates, whose ranks are not comparable with global ones)."""
    r = np.asarray(rows, dtype=float).reshape(-1, 6)
    if features == "rankfree":
        return np.column_stack([r[:, 5], r[:, 4]])
    return np.column_stack([r[:, 5], r[:, 4], np.log1p(r[:, 3]), np.log1p(r[:, 2])])


class KGEPrior:
    def fit(self, X: np.ndarray, y: np.ndarray, X_cal: np.ndarray, y_cal: np.ndarray) -> "KGEPrior":
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.lr = LogisticRegression(max_iter=2000, class_weight=None).fit((X - self.mu) / self.sd, y)
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(self._raw(X_cal), y_cal)
        self.base_rate = float(y.mean())
        return self

    def _raw(self, X: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba((X - self.mu) / self.sd)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self.iso.predict(self._raw(X)), PI_MIN, PI_MAX)

    def to_dict(self) -> dict:
        return {"features": getattr(self, "features", "all"), "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                "coef": self.lr.coef_.ravel().tolist(),
                "intercept": float(self.lr.intercept_[0]), "iso_x": self.iso.X_thresholds_.tolist(),
                "iso_y": self.iso.y_thresholds_.tolist(), "base_rate": self.base_rate}

    @classmethod
    def from_dict(cls, d: dict) -> "KGEPrior":
        self = cls()
        self.d = d
        return self

    def predict_from_dict(self, X: np.ndarray) -> np.ndarray:
        d = self.d
        z = ((X - np.array(d["mu"])) / np.array(d["sd"])) @ np.array(d["coef"]) + d["intercept"]
        p = 1 / (1 + np.exp(-z))
        return np.clip(np.interp(p, d["iso_x"], d["iso_y"]), PI_MIN, PI_MAX)


def head_bucket(h: str) -> float:
    return int(hashlib.md5(h.encode()).hexdigest()[:8], 16) / 2**32


def calibrate(triples: Sequence[Triple], kge_dir: str, rel_rank: str = "hybrid", R: int = 10, k: int = 20,
              rel_pool=None, features: str = "all", max_heads: int = 0, device: str = "cuda",
              seed: int = 0, split_seed: int = 0) -> Tuple[KGEPrior, Dict]:
    """Fit pi on holdout facts as described in the module docstring; returns (prior, report)."""
    from cc_gcd.calibration.observation_model import calibration_metrics
    train, hold = kge_holdout_split(triples, seed=split_seed)  # must match the KGE's training seed
    scorer = ComplExScorer(kge_dir, device=device)
    gen = CandidateGenerator(train, scorer, rel_rank=rel_rank, R_max=R, k_max=k, rel_pool=rel_pool)
    hold_aug = set(augment(hold))
    heads = sorted({h for h, _, _ in hold_aug})
    if max_heads and len(heads) > max_heads:
        heads = sorted(random.Random(seed).sample(heads, max_heads))
    table = gen.build(entities=heads)
    ents, rels = scorer.entities, scorer.relations
    X, y, grp = [], [], []
    for j, ui in enumerate(table["ent"]):
        u = ents[ui]
        rows = candidate_features(table, j, R, k)
        for row in rows:
            X.append(row)
            y.append((u, rels[row[0]], ents[row[1]]) in hold_aug)
            grp.append(head_bucket(u))
    X, y, grp = design(X, features), np.array(y, dtype=int), np.array(grp)
    tr, ca, te = grp < 0.6, (grp >= 0.6) & (grp < 0.8), grp >= 0.8
    prior = KGEPrior().fit(X[tr], y[tr], X[ca], y[ca])
    prior.features = features
    p = prior.predict(X[te])
    report = {"holdout_facts": len(hold), "heads": len(heads), "candidates": int(len(y)),
              "positives": int(y.sum()), "base_rate": round(float(y.mean()), 5), "R": R, "k": k,
              "rel_rank": rel_rank, "test": calibration_metrics(p, y[te]),
              "features": features, "heads_sampled": len(heads),
              "coef_order": ["log_p", "raw_score", "log1p_tail_rank", "log1p_rel_rank"][:X.shape[1]],
              "coef": [round(c, 4) for c in prior.lr.coef_.ravel()]}
    return prior, report
