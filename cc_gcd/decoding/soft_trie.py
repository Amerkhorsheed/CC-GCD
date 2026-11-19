"""Soft KG-Trie and lookahead prefix bias for CC-GCD decoding (plan sections 4.5 and 5.1).

GCR constrains decoding between <PATH> and </PATH> with a trie of tokenized paths. CC-GCD
keeps that trie (ontology-violating paths would still be absent from it) but adds the
observed graph's absent candidate edges and scores every path:

    S(path) = sum over its edges of s(e),

with s(e) = 0 for observed edges and the method's evidence term for absent ones (B2: -1,
B4: log pi, CC-GCD: log sigmoid(logit pi + log(1 - c) - log(1 - f))). lambda scales S.

Lookahead bias (FUDGE-style). Every trie node stores V(node) = max over the completions
below it of lambda * S(path). When a beam at node u considers token a leading to child c,
the bias added to its log-probability is V(c) - V(u). The biases telescope, so a finished
path receives exactly lambda * S(path) - V(root): the path-level objective is unchanged up
to a per-question constant, but beams are steered as soon as a prefix commits to a branch
whose best completion is weak, instead of only when the triple completes (plan 4.5).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
from transformers import LogitsProcessor


class SoftTrie:
    """Token trie with per-node lookahead values; duck-types GCR's Trie.get for allowed tokens."""

    __slots__ = ("root",)

    def __init__(self, sequences: Sequence[Sequence[int]], scores: Sequence[float]):
        # node: [children dict, value]
        self.root: list = [{}, -float("inf")]
        for seq, s in zip(sequences, scores):
            node = self.root
            node[1] = max(node[1], s)
            for tok in seq:
                child = node[0].get(tok)
                if child is None:
                    child = node[0][tok] = [{}, -float("inf")]
                child[1] = max(child[1], s)
                node = child

    def node(self, prefix: Sequence[int]) -> Optional[list]:
        node = self.root
        for tok in prefix:
            node = node[0].get(tok)
            if node is None:
                return None
        return node

    def get(self, prefix: Sequence[int]) -> List[int]:
        node = self.node(prefix)
        return list(node[0].keys()) if node is not None else []

    def bias(self, prefix: Sequence[int]) -> Dict[int, float]:
        node = self.node(prefix)
        if node is None or not node[0]:
            return {}
        return {tok: child[1] - node[1] for tok, child in node[0].items()}


class PrefixBiasProcessor(LogitsProcessor):
    """Adds the lookahead bias inside an open <PATH> span."""

    def __init__(self, trie: SoftTrie, start_id: int, end_id: int):
        self.trie, self.start_id, self.end_id = trie, start_id, end_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        rows, cols, vals = [], [], []
        ids = input_ids.tolist()
        for i, seq in enumerate(ids):
            try:
                start = len(seq) - 1 - seq[::-1].index(self.start_id)
            except ValueError:
                continue
            if self.end_id in seq[start:]:
                continue  # path closed: unconstrained
            for tok, b in self.trie.bias(seq[start:]).items():
                if b != 0.0:
                    rows.append(i)
                    cols.append(tok)
                    vals.append(b)
        if rows:
            scores[torch.tensor(rows, device=scores.device), torch.tensor(cols, device=scores.device)] += \
                torch.tensor(vals, device=scores.device, dtype=scores.dtype)
        return scores
