"""Path enumeration for KG-Trie construction.

GCR (src/utils/graph_utils.py) builds an ``nx.DiGraph`` with ``add_edge(h, t,
relation=r)``, so when an entity pair carries several relations only the last
one survives and paths through the others never enter the KG-Trie. On BRINK
this affects ~30% (FB15k-237) to ~38% (Family) of entity pairs in 2-hop
subgraphs, and half of the topic->hard-answer pairs in FB15k-237. ``dfs_paths``
keeps every relation (multigraph semantics); ``dfs_paths_digraph`` reproduces
GCR's original behaviour for ablations.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

Triple = Tuple[str, str, str]


def _adjacency(graph: Iterable[Sequence[str]]) -> Dict[str, List[Tuple[str, str]]]:
    adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for h, r, t in graph:
        adj[h.strip()].append((r.strip(), t.strip()))
    return adj


def dfs_paths(graph: Iterable[Sequence[str]], seeds: Iterable[str], max_len: int) -> List[List[Triple]]:
    """All directed paths of 1..max_len edges from the seeds, keeping parallel relations.

    Like GCR's ``dfs``, a path may revisit nodes; duplicates are removed.
    """
    adj = _adjacency(graph)
    out = set()

    def visit(node: str, path: Tuple[Triple, ...]):
        if len(path) == max_len:
            return
        for r, v in adj.get(node, []):
            new = path + ((node, r, v),)
            out.add(new)
            visit(v, new)

    for s in seeds:
        visit(s, ())
    return [list(p) for p in sorted(out)]


def dfs_paths_digraph(graph: Iterable[Sequence[str]], seeds: Iterable[str], max_len: int) -> List[List[Triple]]:
    """GCR's original behaviour: one relation per (h, t) pair, the last one seen wins."""
    last: Dict[Tuple[str, str], str] = {}
    for h, r, t in graph:
        last[(h.strip(), t.strip())] = r.strip()
    return dfs_paths([(h, r, t) for (h, t), r in last.items()], seeds, max_len)
