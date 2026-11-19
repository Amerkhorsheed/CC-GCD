"""Observation model c(e) = P(O_e = 1 | T_e = 1, x_e) and the evidence score s(e) (plan sections 4 and 6).

c(e) is not identified from an incomplete graph alone (plan 6.1), so it is learned from
controlled masking of a reference graph (``cc_gcd.calibration.missingness``): for every
reference triple e (true by construction), the label is whether e survived masking.

Leakage. Features must not see e's own label. Removing only e is not enough: within one
masked graph every kept r-triple sees K - 1 other kept r-triples and every dropped one sees
K, so aggregate counts plus the relation identity separate kept from dropped perfectly
(a GBM reached AUC 0.999 under MCAR, where the oracle AUC is 0.5). Features are therefore
cross-fitted: triples are split into ``folds`` groups and fold k's features are computed on
the masked graph with *all* of fold k removed, so they depend on no label in their fold.
At deployment a candidate edge is absent anyway, which the 10% thinning mirrors.

Features (structural, plan 6.4; FB15k-237 has no types or provenance):
    relation frequency; head and tail degree; number of r-edges at the head and at the
    tail; whether the head / tail has another r-edge; distinct relations at head / tail;
    peer cardinality (Galarraga et al. 2017, Arnaout et al.): the mean number of r-edges
    of heads (tails) that have r, and the head's (tail's) count relative to it.
The GBM variant also sees the relation identity (a per-relation effect, as in baseline B3).

Estimators:
    "scar"        constant c = retention rate of the training mechanism (Elkan-Noto; B5).
    "relation"    per-relation retention, Beta-smoothed toward the global rate.
    "logreg"      logistic regression on standardized structural features.
    "gbm"         histogram gradient boosting (sklearn; LightGBM is not a dependency) on
                  structural features + relation identity.
Every estimator except "scar" is recalibrated with isotonic regression on a separate
split of the training mechanism, then clipped (plan 4.4) so that log(1 - c) stays finite.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Triple = Tuple[str, str, str]

FEATURES = ("log_rel_freq", "log_deg_h", "log_deg_t", "log_n_hr", "log_n_rt", "has_other_hr", "has_other_rt",
            "log_rels_h", "log_rels_t", "log_peer_card_h", "log_peer_card_t", "rel_card_h", "rel_card_t")

EPS, EPS0, ETA = 0.02, 0.02, 0.005  # clipping defaults (plan 4.4)


# ---------------------------------------------------------------------------- features
class Encoded:
    """Integer-encoded reference triples."""

    def __init__(self, triples: Sequence[Triple]):
        ents = sorted({e for h, _, t in triples for e in (h, t)})
        rels = sorted({r for _, r, _ in triples})
        self.e2id = {e: i for i, e in enumerate(ents)}
        self.r2id = {r: i for i, r in enumerate(rels)}
        self.h = np.array([self.e2id[h] for h, _, _ in triples], dtype=np.int64)
        self.r = np.array([self.r2id[r] for _, r, _ in triples], dtype=np.int64)
        self.t = np.array([self.e2id[t] for _, _, t in triples], dtype=np.int64)
        self.E, self.R, self.n = len(ents), len(rels), len(triples)


def _count(keys: np.ndarray, weights: np.ndarray, size: int) -> np.ndarray:
    return np.bincount(keys, weights=weights, minlength=size)


def _pair_counts(a: np.ndarray, b: np.ndarray, nb: int, kept: np.ndarray):
    """Counts of each (a, b) pair among kept rows, returned per row."""
    key = a * nb + b
    uniq, inv = np.unique(key, return_inverse=True)
    return np.bincount(inv, weights=kept, minlength=len(uniq))[inv]


def features(enc: Encoded, kept: np.ndarray, folds: int = 10, seed: int = 0) -> np.ndarray:
    """(n, len(FEATURES)) cross-fitted features: fold k on the observed (kept) graph minus all of fold k.

    ``folds=1`` gives the deployment form (graph minus the triple itself), which must not be used
    to build training labels (see the module docstring)."""
    if folds <= 1:
        return _features(enc, kept.astype(float), kept.astype(float))
    fold = np.random.default_rng(seed + 7).integers(0, folds, enc.n)
    X = np.empty((enc.n, len(FEATURES)))
    for k in range(folds):
        rows = fold == k
        graph = kept & ~rows
        X[rows] = _features(enc, graph.astype(float), np.zeros(enc.n))[rows]
    return X


def _features(enc: Encoded, g: np.ndarray, o: np.ndarray) -> np.ndarray:
    """Features of every triple on the graph ``g`` (0/1 per reference triple), minus each triple's own
    contribution ``o`` (o = g for the deployment form, 0 when the triple is already outside g)."""
    h, r, t, E, R = enc.h, enc.r, enc.t, enc.E, enc.R
    n_r = _count(r, g, R)[r] - o
    loop = (h == t).astype(float)  # a self-loop counts twice at its entity (1,539 on FB15k-237)
    deg = _count(h, g, E) + _count(t, g, E)
    deg_h, deg_t = deg[h] - o * (1 + loop), deg[t] - o * (1 + loop)
    n_hr = _pair_counts(h, r, R, g) - o
    n_rt = _pair_counts(t, r, R, g) - o

    # Distinct relations at an entity (either direction), without e: r drops out of h's set
    # when e was h's only r-edge.
    ent = np.concatenate([h, t])
    rel_dir = np.concatenate([r, r + R])  # outgoing r vs incoming r
    w = np.concatenate([g, g])
    key = ent * (2 * R) + rel_dir
    uniq, inv = np.unique(key, return_inverse=True)
    present = np.bincount(inv, weights=w, minlength=len(uniq)) > 0
    n_rels = np.bincount(uniq // (2 * R), weights=present.astype(float), minlength=E)
    rels_h = n_rels[h] - o * (n_hr == 0) - o * loop * (n_rt == 0)
    rels_t = n_rels[t] - o * (n_rt == 0) - o * loop * (n_hr == 0)

    # Peer cardinality: mean r-edges per head among heads that have r (and per tail), without e.
    heads_with_r = np.zeros(R)
    tails_with_r = np.zeros(R)
    uh, ih = np.unique(h * R + r, return_inverse=True)
    hr_cnt = np.bincount(ih, weights=g, minlength=len(uh))
    np.add.at(heads_with_r, uh % R, (hr_cnt > 0).astype(float))
    ut, it = np.unique(t * R + r, return_inverse=True)
    rt_cnt = np.bincount(it, weights=g, minlength=len(ut))
    np.add.at(tails_with_r, ut % R, (rt_cnt > 0).astype(float))
    hw = heads_with_r[r] - o * (n_hr == 0)
    tw = tails_with_r[r] - o * (n_rt == 0)
    peer_h = n_r / np.maximum(hw, 1.0)
    peer_t = n_r / np.maximum(tw, 1.0)

    X = np.stack([
        np.log1p(n_r), np.log1p(deg_h), np.log1p(deg_t), np.log1p(n_hr), np.log1p(n_rt),
        (n_hr > 0).astype(float), (n_rt > 0).astype(float), np.log1p(rels_h), np.log1p(rels_t),
        np.log1p(peer_h), np.log1p(peer_t), n_hr / np.maximum(peer_h, 1e-9), n_rt / np.maximum(peer_t, 1e-9),
    ], axis=1)
    return X


# ---------------------------------------------------------------------------- estimators
class ObservationModel:
    def __init__(self, kind: str = "gbm", eps: float = EPS, eps0: float = EPS0, prior_strength: float = 20.0,
                 seed: int = 0):
        if kind not in ("scar", "relation", "logreg", "gbm"):
            raise ValueError(kind)
        self.kind, self.eps, self.eps0, self.k, self.seed = kind, eps, eps0, prior_strength, seed
        self.iso = None

    def _raw(self, X: np.ndarray, rel: np.ndarray) -> np.ndarray:
        if self.kind == "scar":
            return np.full(len(X), self.rate)
        if self.kind == "relation":
            return self.rel_rate[rel] if rel.max(initial=-1) < len(self.rel_rate) else np.full(len(X), self.rate)
        if self.kind == "logreg":
            return self.clf.predict_proba((X - self.mu) / self.sd)[:, 1]
        return self.clf.predict_proba(np.column_stack([X, rel]))[:, 1]

    def fit(self, X: np.ndarray, rel: np.ndarray, y: np.ndarray, X_cal: np.ndarray, rel_cal: np.ndarray,
            y_cal: np.ndarray, n_rel: int) -> "ObservationModel":
        self.rate = float(y.mean())
        if self.kind == "relation":
            kept = np.bincount(rel, weights=y, minlength=n_rel)
            tot = np.bincount(rel, minlength=n_rel)
            self.rel_rate = (kept + self.k * self.rate) / (tot + self.k)
        elif self.kind == "logreg":
            from sklearn.linear_model import LogisticRegression
            self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
            self.clf = LogisticRegression(max_iter=2000, C=1.0).fit((X - self.mu) / self.sd, y)
        elif self.kind == "gbm":
            from sklearn.ensemble import HistGradientBoostingClassifier
            cat = np.zeros(X.shape[1] + 1, dtype=bool)
            cat[-1] = True
            self.clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                                      categorical_features=cat, early_stopping=True,
                                                      validation_fraction=0.1, random_state=self.seed)
            self.clf.fit(np.column_stack([X, rel]), y)
        if self.kind != "scar":
            from sklearn.isotonic import IsotonicRegression
            self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.iso.fit(self._raw(X_cal, rel_cal), y_cal)
        return self

    def predict(self, X: np.ndarray, rel: np.ndarray, clip: bool = True) -> np.ndarray:
        p = self._raw(X, rel)
        if self.iso is not None:
            p = self.iso.predict(p)
        return np.clip(p, self.eps, 1.0 - self.eps0) if clip else p


# ---------------------------------------------------------------------------- evidence score
def evidence_score(pi: np.ndarray, c: np.ndarray, observed: np.ndarray, f: float = 0.01,
                   eps: float = EPS, eps0: float = EPS0, eta: float = ETA) -> np.ndarray:
    """s(e) = log P(T_e = 1 | O_e, x_e) = log sigmoid(logit pi + Delta), with (plan 4.2-4.4)
        Delta_obs = log c - log f,   Delta_abs = log(1 - c) - log(1 - f),
    c clipped to [eps, 1 - eps0] and f to [eta, 1 - eta]. The decoder adds lambda * s(e) per edge.
    With f -> 0 an absent edge gets log(pi (1 - c) / (1 - pi c)): log pi as c -> 0 (the KGE prior,
    B4) and -> -inf as c -> 1 (the hard mask, B1)."""
    pi = np.clip(np.asarray(pi, dtype=float), 1e-12, 1 - 1e-12)
    c = np.clip(np.asarray(c, dtype=float), eps, 1.0 - eps0)
    f = float(np.clip(f, eta, 1.0 - eta))
    delta = np.where(observed, np.log(c) - np.log(f), np.log1p(-c) - np.log1p(-f))
    z = np.log(pi) - np.log1p(-pi) + delta
    return -np.logaddexp(0.0, -z)


# ---------------------------------------------------------------------------- level shift
def match_level(p: np.ndarray, rate: float, eps: float = EPS, eps0: float = EPS0) -> np.ndarray:
    """Shift logit(p) by one constant so that mean(p) equals ``rate`` (bisection), then clip.

    Separates the *level* of c (mechanism-specific, not identified from an incomplete graph; a
    declared coverage in the sense of Li & Song's c_R) from its per-edge *shape*. Using the target's
    true rate makes this a diagnostic, not an estimator."""
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6)) - np.log1p(-np.clip(p, 1e-6, 1 - 1e-6))
    lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(z + mid)))).mean() < rate:
            lo = mid
        else:
            hi = mid
    return np.clip(1 / (1 + np.exp(-(z + (lo + hi) / 2))), eps, 1 - eps0)


# ---------------------------------------------------------------------------- metrics
def calibration_metrics(p: np.ndarray, y: np.ndarray, bins: int = 15) -> Dict:
    """Brier, ECE (equal-mass bins), debiased l2-ECE (Kumar et al. 2019), AUC and a reliability table."""
    from sklearn.metrics import roc_auc_score
    p, y = np.asarray(p, float), np.asarray(y, float)
    order = np.argsort(p, kind="stable")
    groups = np.array_split(order, bins)
    ece = 0.0
    de2 = 0.0
    table: List[Dict] = []
    for g in groups:
        if len(g) == 0:
            continue
        pb, yb, w = p[g].mean(), y[g].mean(), len(g) / len(p)
        ece += w * abs(pb - yb)
        de2 += w * ((yb - pb) ** 2 - (yb * (1 - yb) / (len(g) - 1) if len(g) > 1 else 0.0))
        table.append({"mean_pred": round(float(pb), 4), "observed_rate": round(float(yb), 4), "n": int(len(g))})
    auc = float(roc_auc_score(y, p)) if 0 < y.mean() < 1 and p.std() > 0 else 0.5
    return {"brier": round(float(np.mean((p - y) ** 2)), 5), "ece": round(float(ece), 4),
            "ece_debiased_l2": round(float(np.sqrt(max(de2, 0.0))), 4), "auc": round(auc, 4),
            "mean_pred": round(float(p.mean()), 4), "observed_rate": round(float(y.mean()), 4),
            "reliability": table}
