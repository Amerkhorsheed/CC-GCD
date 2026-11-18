"""Convert raw BRINK files (Hugging Face ZDZR/BRINK-*) into the CC-GCD layout.

Input  (download with data_prep/download_brink.sh):
    data/raw/brink/family/{knowledge_graph_complete,knowledge_graph_incomplete,train,val,test}.tsv
    data/raw/brink/fb15k237/{EntityID,Textlabel}/... (same file names)

Output (per dataset, under data/processed/brink/<dataset>/):
    entities.tsv            canonical id \t unique display name
    kg_complete.tsv         h \t r \t t (canonical ids)
    kg_incomplete.tsv       h \t r \t t
    removed.tsv             complete minus incomplete
    questions_<split>.jsonl id, question, q_entity, answer, a_entity, hard_answer,
                            hard_triples (removed triples linking topic and hard answer),
                            [rule, grounding] when BRINK provides them
    gold_<split>.json       BRINK evaluator format; hard_answer is the display label
                            (raw BRINK hard_answer kept as hard_answer_raw)
    stats.json              sizes and hard-KG-Trie reachability ceilings

Reachability uses inverse-augmented directed paths from the topic entities, the
search space of a KG-Trie built as in GCR plus inverse edges.

Usage: python data_prep/prepare_brink.py --dataset {family,fb15k237} [--max_hops 3]
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from cc_gcd.data.brink import BrinkKG  # noqa: E402

SPLITS = ("train", "val", "test")


def read_kg(path):
    with open(path, encoding="utf-8") as f:
        return [tuple(x.strip('"') for x in line.rstrip("\n").split("\t")) for line in f if line.strip()]


def read_questions(path):
    with open(path, encoding="utf-8") as f:
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    header, body = rows[0], rows[1:]
    out = []
    for r in body:
        d = dict(zip(header, r))
        for k in ("q_entity", "answer", "a_entity", "hard_answer", "rule", "grounding"):
            if k in d:
                d[k] = ast.literal_eval(d[k])
        out.append(d)
    return out


def display_names(id_triples, label_triples):
    """FB15k-237: align EntityID and Textlabel KGs line by line; make labels unique."""
    mid2label = {}
    for (h, r, t), (hl, rl, tl) in zip(id_triples, label_triples):
        assert r == rl, "EntityID and Textlabel KGs are not aligned"
        for mid, lab in ((h, hl), (t, tl)):
            old = mid2label.setdefault(mid, lab)
            assert old == lab, f"MID {mid} has two labels"
    counts = Counter(mid2label.values())
    return {mid: (lab if counts[lab] == 1 else f"{lab} ({mid})") for mid, lab in mid2label.items()}, mid2label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["family", "fb15k237"], required=True)
    ap.add_argument("--raw", default=os.path.join(ROOT, "data", "raw", "brink"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "processed", "brink"))
    ap.add_argument("--max_hops", type=int, default=3)
    args = ap.parse_args()

    out = os.path.join(args.out, args.dataset)
    os.makedirs(out, exist_ok=True)

    if args.dataset == "fb15k237":
        base = os.path.join(args.raw, "fb15k237", "EntityID")
        text = os.path.join(args.raw, "fb15k237", "Textlabel")
        kg_c = read_kg(os.path.join(base, "knowledge_graph_complete.tsv"))
        kg_i = read_kg(os.path.join(base, "knowledge_graph_incomplete.tsv"))
        names, labels = display_names(kg_c, read_kg(os.path.join(text, "knowledge_graph_complete.tsv")))
        qbase, qtext = base, text
    else:
        base = os.path.join(args.raw, "family")
        kg_c = read_kg(os.path.join(base, "knowledge_graph_complete.tsv"))
        kg_i = read_kg(os.path.join(base, "knowledge_graph_incomplete.tsv"))
        ents = {e for h, _, t in kg_c for e in (h, t)}
        names = labels = {e: e for e in ents}
        qbase = qtext = base

    set_c, set_i = set(kg_c), set(kg_i)
    assert set_i <= set_c, "incomplete KG is not a subset of the complete KG"
    removed = sorted(set_c - set_i)

    def write_kg(path, triples):
        with open(path, "w", encoding="utf-8") as f:
            for t in triples:
                f.write("\t".join(t) + "\n")

    with open(os.path.join(out, "entities.tsv"), "w", encoding="utf-8") as f:
        for eid in sorted(names):
            f.write(f"{eid}\t{names[eid]}\n")
    write_kg(os.path.join(out, "kg_complete.tsv"), sorted(set_c))
    write_kg(os.path.join(out, "kg_incomplete.tsv"), sorted(set_i))
    write_kg(os.path.join(out, "removed.tsv"), removed)

    by_pair = defaultdict(list)
    for h, r, t in removed:
        by_pair[(h, t)].append([h, r, t])
        by_pair[(t, h)].append([h, r, t])

    stats = {
        "dataset": args.dataset,
        "triples_complete": len(set_c),
        "triples_incomplete": len(set_i),
        "triples_removed": len(removed),
        "relations": len({r for _, r, _ in kg_c}),
        "entities": len(names),
        "ambiguous_labels_disambiguated": sum(1 for e in names if names[e] != labels.get(e, names[e])),
        "splits": {},
    }

    for split in SPLITS:
        qs_id = read_questions(os.path.join(qbase, f"{split}.tsv"))
        qs_tx = {q["id"]: q for q in read_questions(os.path.join(qtext, f"{split}.tsv"))}
        rows, gold, no_hard, ans_missing = [], [], 0, 0
        for q in qs_id:
            tx = qs_tx[q["id"]]
            hard = [[h, r, t] for qe in q["q_entity"] for a in q["hard_answer"] for (h, r, t) in by_pair.get((qe, a), [])]
            hard = [list(x) for x in sorted({tuple(x) for x in hard})]
            no_hard += not hard
            ans_missing += any(a not in names for a in q["a_entity"])
            row = {
                "id": q["id"],
                "question": tx["question"],
                "q_entity": q["q_entity"],
                "answer": q["answer"],
                "a_entity": q["a_entity"],
                "hard_answer": q["hard_answer"],
                "hard_triples": hard,
            }
            for k in ("rule", "grounding"):
                if k in q:
                    row[k] = q[k]
            rows.append(row)
            answers = sorted({names.get(a, a) for a in q["a_entity"]} | {labels.get(a, a) for a in q["a_entity"]} | set(map(str, tx["answer"])))
            gold.append({
                "id": q["id"],
                "question": tx["question"],
                "answers": answers,
                "hard_answer": labels.get(q["hard_answer"][0], q["hard_answer"][0]),
                "hard_answer_raw": tx["hard_answer"][0],
            })
        with open(os.path.join(out, f"questions_{split}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(os.path.join(out, f"gold_{split}.json"), "w", encoding="utf-8") as f:
            json.dump(gold, f, ensure_ascii=False, indent=1)
        stats["splits"][split] = {"questions": len(rows), "without_hard_triple": no_hard,
                                  "with_answer_not_in_kg": ans_missing,
                                  "hard_triple_topic_is_tail": sum(any(t[2] in r["q_entity"] for t in r["hard_triples"]) for r in rows)}

    # Hard KG-Trie ceilings: is the hard / any answer within <= k inverse-augmented hops of the topic?
    for version in ("complete", "incomplete"):
        kg = BrinkKG(out, version=version, inverse=True)
        kg_dir = BrinkKG(out, version=version, inverse=False)
        for split in ("val", "test"):
            qs = list(kg.questions(split))
            res = {}
            for label, g in (("inverse", kg), ("directed", kg_dir)):
                hard_k = Counter()
                any_k = Counter()
                for q in qs:
                    dist = g.distances(q["q_entity"], args.max_hops)
                    dh = min((dist[a] for a in q["hard_answer"] if a in dist), default=None)
                    da = min((dist[a] for a in q["a_entity"] if a in dist), default=None)
                    for k in range(1, args.max_hops + 1):
                        hard_k[k] += dh is not None and dh <= k
                        any_k[k] += da is not None and da <= k
                n = len(qs)
                res[label] = {f"hard_answer_within_{k}": round(hard_k[k] / n, 4) for k in range(1, args.max_hops + 1)}
                res[label].update({f"any_answer_within_{k}": round(any_k[k] / n, 4) for k in range(1, args.max_hops + 1)})
            sizes = sorted(len(kg.neighbourhood(q["q_entity"], 2)) for q in qs)
            res["inverse"]["subgraph_2hop_edges_median"] = sizes[len(sizes) // 2]
            res["inverse"]["subgraph_2hop_edges_p90"] = sizes[int(0.9 * len(sizes))]
            res["inverse"]["subgraph_2hop_edges_max"] = sizes[-1]
            stats.setdefault("reachability", {}).setdefault(version, {})[split] = res

    with open(os.path.join(out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
