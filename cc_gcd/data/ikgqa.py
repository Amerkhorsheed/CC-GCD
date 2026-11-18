"""IKGQA (GoG, EMNLP 2024) applied to RoG subgraphs.

GoG builds incomplete-KG settings IKG-20/40/60/80% for the first 1,000 test
questions of WebQSP and CWQ. For each question it binds the gold SPARQL query,
drops each (head, tail) entity pair on the bound answer paths with probability
p, and deletes every relation between that pair in both directions
("crucial triples"). The deletion is per question, not one global KG.

GCR (and CC-GCD) read the RoG subgraph of each question, so the IKG version of
a question is its RoG ``graph`` minus GoG's crucial triples (matched on the
label form ``crucial_triples``; GoG maps questions to RoG via
``idx_in_processed_samples``). ``data_prep/prepare_ikgqa.py`` stores only the
removed triples per question; this module re-applies them to the Hugging Face
RoG dataset so nothing large is duplicated on disk.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterator, List

LEVELS = (0.0, 0.2, 0.4, 0.6, 0.8)
ROG = {"webqsp": "rmanluo/RoG-webqsp", "cwq": "rmanluo/RoG-cwq"}


def load_rog_test(dataset: str):
    """RoG test split only (RoG-CWQ is ~3 GB with train; the test split is ~370 MB)."""
    from datasets import load_dataset

    return load_dataset(ROG[dataset], data_files={"test": "data/test-*.parquet"}, split="test",
                        verification_mode="no_checks")


def level_tag(p: float) -> str:
    return f"ikg{int(round(p * 100))}"


class IKGQA:
    def __init__(self, root: str, dataset: str = "webqsp"):
        self.root = os.path.join(root, dataset)
        self.dataset = dataset
        self._rog = None

    @property
    def rog(self):
        if self._rog is None:
            self._rog = load_rog_test(self.dataset)
        return self._rog

    def removals(self, p: float) -> List[dict]:
        with open(os.path.join(self.root, f"{level_tag(p)}.jsonl"), encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def samples(self, p: float) -> Iterator[dict]:
        """RoG-format samples (id, question, q_entity, a_entity, answer, graph) with IKG deletions applied."""
        for rec in self.removals(p):
            s = dict(self.rog[rec["rog_idx"]])
            removed = {tuple(t) for t in rec["removed_triples"]}
            s["graph"] = [t for t in s["graph"] if tuple(t) not in removed]
            s["ikg_level"] = p
            s["removed_triples"] = rec["removed_triples"]
            yield s
