"""Anonymized copy of the Wikidata-domain QA benchmark for the parametric-memory control (plan 7.3, E5).

Every entity name becomes a neutral token "entity N" (N from a seeded permutation), in the questions, the graph the
LLM reads and the gold answers; property labels stay. Entity ids (QIDs), the KG, the KGE, the candidates and pi are
unchanged, so the decoders see the same evidence and only the names the model could have memorized are gone.

    python data_prep/anonymize_wikidata_qa.py
"""

from __future__ import annotations

import json
import os
import random
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main(seed: int = 0):
    src = os.path.join(ROOT, "data", "processed", "brink", "wikidata")
    dst = os.path.join(ROOT, "data", "processed", "brink", "wikidata-anon")
    os.makedirs(dst, exist_ok=True)
    qids = [line.split("\t")[0] for line in open(os.path.join(src, "entities.tsv"), encoding="utf-8")]
    nums = list(range(10000, 10000 + len(qids)))
    random.Random(seed).shuffle(nums)
    anon = {q: f"entity {n}" for q, n in zip(qids, nums)}
    with open(os.path.join(dst, "entities.tsv"), "w", encoding="utf-8") as f:
        for q in qids:
            f.write(f"{q}\t{anon[q]}\n")
    for fn in ("kg_incomplete.tsv", "kg_complete.tsv", "removed.tsv", "stats.json"):
        shutil.copy(os.path.join(src, fn), os.path.join(dst, fn))
    for sp in ("val", "test"):
        qs = [json.loads(l) for l in open(os.path.join(src, f"questions_{sp}.jsonl"), encoding="utf-8")]
        gold = []
        with open(os.path.join(dst, f"questions_{sp}.jsonl"), "w", encoding="utf-8") as f:
            for q in qs:
                h, r, hard = q["hard_triples"][0]
                q["question"] = f"What is the {r} of {anon[h]}?"
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
                gold.append({"id": q["id"], "question": q["question"], "answers": sorted(anon[t] for t in q["a_entity"]),
                             "hard_answer": anon[hard], "hard_answer_raw": hard})
        json.dump(gold, open(os.path.join(dst, f"gold_{sp}.json"), "w"), ensure_ascii=False, indent=1)
    print("wrote", dst, len(anon), "entities anonymized")


if __name__ == "__main__":
    main()
