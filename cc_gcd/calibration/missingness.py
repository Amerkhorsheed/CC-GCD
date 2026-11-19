"""Controlled missingness mechanisms for estimating the observation model c(e) (plan section 7.2).

Each mechanism maps a reference graph (canonical triples) to an exact per-triple
retention probability ``p_keep``; ``sample`` draws the masked graph from it. Because
``p_keep`` is exact, it is also the oracle c(e) of baseline B6: the best any estimator
of c can do under that mechanism.

    A  MCAR              every triple is dropped with the same probability.
    B  relation-selective a random subset of relations loses most of its triples.
    C  degree-biased     triples of low-degree heads (the long tail) are dropped more often.
    D  path-targeted     triples on <= 2-hop paths between topic entities and answers of
                         *training* questions are dropped more often (plan 6.5: test
                         questions are never used to build a mechanism).
    E  noisy graph       (false additions, for f) is not implemented yet: masking only
                         creates false negatives, so it cannot inform f (plan 6.3).

The reported result for c is always off-diagonal: calibrated on one mechanism and
tested on another (plan 6.5).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

Triple = Tuple[str, str, str]
MECHANISMS = ("A_mcar", "B_relation", "C_degree", "D_path")


def mcar(triples: Sequence[Triple], rate: float = 0.3, **_) -> np.ndarray:
    return np.full(len(triples), 1.0 - rate)


def relation_selective(triples: Sequence[Triple], frac_relations: float = 0.3, high: float = 0.6,
                       low: float = 0.1, seed: int = 0, **_) -> np.ndarray:
    rels = sorted({r for _, r, _ in triples})
    rng = np.random.default_rng(seed + 101)
    hit = set(rng.choice(rels, size=max(1, int(round(frac_relations * len(rels)))), replace=False).tolist())
    return np.array([1.0 - (high if r in hit else low) for _, r, _ in triples])


def degree_biased(triples: Sequence[Triple], max_rate: float = 0.6, min_rate: float = 0.02, **_) -> np.ndarray:
    deg: Dict[str, int] = defaultdict(int)
    for h, _, t in triples:
        deg[h] += 1
        deg[t] += 1
    d = np.array([deg[h] for h, _, _ in triples], dtype=float)
    # Degree percentile of the head among all triples' heads: low degree -> high drop rate.
    q = (np.argsort(np.argsort(d, kind="stable"), kind="stable") + 0.5) / len(d)
    return 1.0 - (min_rate + (max_rate - min_rate) * (1.0 - q))


def path_targeted(triples: Sequence[Triple], questions: Iterable[dict] = (), on_path: float = 0.6,
                  off_path: float = 0.05, hops: int = 2, **_) -> np.ndarray:
    """Triples on <= ``hops`` paths (inverse edges allowed) from a training question's topic
    entities to one of its answers."""
    nbr: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for i, (h, _, t) in enumerate(triples):
        nbr[h][t].append(i)
        nbr[t][h].append(i)
    on = np.zeros(len(triples), dtype=bool)
    for q in questions:
        for s in q["q_entity"]:
            ns = nbr.get(s)
            if not ns:
                continue
            for a in q["a_entity"]:
                if a == s or a not in nbr:
                    continue
                for i in ns.get(a, ()):  # 1 hop
                    on[i] = True
                if hops >= 2:  # 2 hops: s - m - a through a shared neighbour m
                    na = nbr[a]
                    small, big = (ns, na) if len(ns) <= len(na) else (na, ns)
                    for m in small.keys() & big.keys():
                        if m in (s, a):
                            continue
                        for i in ns[m] + na[m]:
                            on[i] = True
    return np.where(on, 1.0 - on_path, 1.0 - off_path)


BUILDERS = {"A_mcar": mcar, "B_relation": relation_selective, "C_degree": degree_biased, "D_path": path_targeted}


def retention(name: str, triples: Sequence[Triple], **kwargs) -> np.ndarray:
    """Exact retention probability p_keep (= oracle c) of every reference triple under a mechanism."""
    return BUILDERS[name](triples, **kwargs)


def sample(p_keep: np.ndarray, seed: int = 0) -> np.ndarray:
    """Boolean keep mask drawn from p_keep."""
    return np.random.default_rng(seed).random(len(p_keep)) < p_keep
