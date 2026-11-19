"""BRINK loader: global KG + per-question subgraphs in RoG/GCR format.

Files are produced by ``data_prep/prepare_brink.py`` under
``data/processed/brink/<dataset>/``. Entities are stored by canonical id
(Freebase MID for FB15k-237, numeric id for Family); ``entities.tsv`` maps
ids to unique display names, which is what the LLM and the KG-Trie see.

``BrinkKG.sample(...)`` returns a dict with the keys GCR's prompt builders
read (``id, question, q_entity, a_entity, answer, graph``) plus BRINK fields
(``hard_answer, hard_triples``). Inverse edges are added as ``~relation`` so
that a directed path search from the topic entity can follow facts in which
the topic entity is the tail (true for ~6% of FB15k-237 and ~40% of Family
hard triples).
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

Triple = Tuple[str, str, str]
INVERSE_PREFIX = "~"


def _read_tsv(path: str) -> List[List[str]]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n").split("\t") for line in f if line.strip()]


class BrinkKG:
    def __init__(self, root: str, version: str = "incomplete", inverse: bool = True):
        if version not in ("complete", "incomplete"):
            raise ValueError("version must be 'complete' or 'incomplete'")
        self.root = root
        self.version = version
        self.inverse = inverse
        self.name: Dict[str, str] = {eid: disp for eid, disp in _read_tsv(os.path.join(root, "entities.tsv"))}
        self.triples: List[Triple] = [tuple(t) for t in _read_tsv(os.path.join(root, f"kg_{version}.tsv"))]
        self.adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for h, r, t in self.triples:
            self.adj[h].append((r, t))
            if inverse:
                self.adj[t].append((INVERSE_PREFIX + r, h))

    # ------------------------------------------------------------------ questions
    def questions(self, split: str) -> Iterator[dict]:
        with open(os.path.join(self.root, f"questions_{split}.jsonl"), encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)

    # ------------------------------------------------------------------ subgraph
    def neighbourhood(self, seeds: Iterable[str], hops: int, max_fanout: Optional[int] = None,
                      extra_adj: Optional[Dict[str, List[Tuple[str, str]]]] = None) -> List[Triple]:
        """Edges on directed (inverse-augmented) paths of length <= hops from seeds.

        ``max_fanout`` caps the number of edges kept per (node, relation) at
        depth >= 1 (the seed's own edges are never capped), keeping the
        lexicographically first tails so the cap is deterministic.
        ``extra_adj`` adds edges (e.g. absent candidates) that the walk may also follow.
        """
        edges: set = set()
        frontier = set(seeds)
        seen = set(seeds)
        for depth in range(hops):
            nxt = set()
            for u in frontier:
                out = self.adj.get(u, []) + (extra_adj.get(u, []) if extra_adj else [])
                if max_fanout is not None and depth > 0:
                    by_rel: Dict[str, List[str]] = defaultdict(list)
                    for r, v in out:
                        by_rel[r].append(v)
                    out = [(r, v) for r, vs in by_rel.items() for v in sorted(vs)[:max_fanout]]
                for r, v in out:
                    edges.add((u, r, v))
                    if v not in seen:
                        nxt.add(v)
            seen |= nxt
            frontier = nxt
        return sorted(edges)

    def distances(self, seeds: Iterable[str], max_hops: int) -> Dict[str, int]:
        dist = {s: 0 for s in seeds}
        frontier = list(dist)
        for d in range(1, max_hops + 1):
            nxt = []
            for u in frontier:
                for _, v in self.adj.get(u, []):
                    if v not in dist:
                        dist[v] = d
                        nxt.append(v)
            frontier = nxt
        return dist

    def display(self, eid: str) -> str:
        return self.name.get(eid, eid)

    def sample(self, q: dict, hops: int = 2, max_fanout: Optional[int] = None,
               extra_edges: Optional[Iterable[Triple]] = None) -> dict:
        """One question in the dict format GCR's prompt builders expect (display names).

        ``extra_edges`` (canonical ids, inverse-augmented) are absent candidate edges from
        ``cc_gcd.graph.candidate_generator``; they join ``graph`` (baseline B1+) and are also
        listed under ``candidate_graph`` so decoders and metrics can tell them apart.
        """
        extra = sorted({tuple(e[:3]) for e in extra_edges}) if extra_edges is not None else None
        extra_adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for h, r, t in extra or ():
            extra_adj[h].append((r, t))
        graph = self.neighbourhood(q["q_entity"], hops, max_fanout, extra_adj or None)
        d = self.display
        out = {
            "id": q["id"],
            "question": q["question"],
            "q_entity": [d(e) for e in q["q_entity"]],
            "a_entity": [d(e) for e in q["a_entity"]],
            "answer": [d(e) for e in q["answer"]],
            "hard_answer": [d(e) for e in q["hard_answer"]],
            "hard_triples": [[d(h), r, d(t)] for h, r, t in q["hard_triples"]],
            "graph": [[d(h), r, d(t)] for h, r, t in graph],
        }
        if extra is not None:
            out["candidate_graph"] = [[d(h), r, d(t)] for h, r, t in extra]
        return out
