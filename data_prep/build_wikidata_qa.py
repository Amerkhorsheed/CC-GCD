"""Wikidata-domain QA with natural missingness, in BRINK's format (data/processed/brink/wikidata/).

The plan's natural c(e) model is learned on Wikidata (2019-08-26 vs 2026-11-17); transferred to Freebase it
ranks missingness weakly. This builds a QA test in its own domain:

  KG (observed, "incomplete")  the 2019 graph over the 13,631 FB15k-237 entities (facts in 2019, including
                               those gone by 2026), from data/processed/wikidata/natural_t0_<t0>.tsv.
  "complete"                   the 2019 graph plus every labelled fact true now (not-yet-true facts excluded).
  Question                     "What is the <property> of <entity>?" for an (entity, property) pair whose
                               values true now include exactly one value missing in 2019 (the hard answer);
                               gold answers = all values true now.
  Splits                       by head entity with the same md5 split as scripts/observation_natural.py:
                               calibration heads -> val (dev), test heads -> test. The natural c(e) model
                               (scripts/fit_natural_c.py) never saw these heads' labels.

Entity names: BRINK-FB15k-237 display labels through the one-to-one QID <-> MID map. Property names: English
labels from the Wikidata API (one call per 50 properties; generic User-Agent).

    python data_prep/build_wikidata_qa.py --n_val 150 --n_test 400
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from observation_natural import head_split  # noqa: E402
from wikidata_snapshots import api  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", default="2019-08-26")
    ap.add_argument("--n_val", type=int, default=150)
    ap.add_argument("--n_test", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    wd = os.path.join(ROOT, "data", "processed", "wikidata")
    rows = [line.rstrip("\n").split("\t") for line in open(os.path.join(wd, f"natural_t0_{args.t0}.tsv"))][1:]
    q2m = {q: m for m, q in (line.rstrip("\n").split("\t") for line in open(os.path.join(wd, "fb15k237_qid.tsv")))}
    m2name = dict(line.rstrip("\n").split("\t") for line in
                  open(os.path.join(ROOT, "data", "processed", "brink", "fb15k237", "entities.tsv")))
    name = {q: m2name[m] for q, m in q2m.items() if m in m2name}

    props = sorted({r for _, r, _, _, _ in rows})
    plabel = {}
    for i in range(0, len(props), 50):
        ents = api({"action": "wbgetentities", "ids": "|".join(props[i:i + 50]), "props": "labels",
                    "languages": "en"}).get("entities", {})
        for p, e in ents.items():
            plabel[p] = e.get("labels", {}).get("en", {}).get("value", p)
    cnt = Counter(plabel.values())
    rel = {p: (plabel[p] if cnt[plabel[p]] == 1 else f"{plabel[p]} ({p})") for p in props}

    observed, true_now = set(), set()
    for h, p, t, in_t0, lab in rows:
        if h not in name or t not in name:
            continue
        e = (h, rel[p], t)
        if in_t0 == "1":
            observed.add(e)
        if lab == "1":
            true_now.add(e)
    complete = observed | true_now
    removed = complete - observed

    by_hr = defaultdict(set)
    for h, r, t in true_now:
        by_hr[(h, r)].add(t)
    split = dict(zip(sorted({h for h, _, _ in true_now}), head_split(sorted({h for h, _, _ in true_now}))))
    pools = {"val": [], "test": []}
    for (h, r), tails in sorted(by_hr.items()):
        missing = sorted(t for t in tails if (h, r, t) in removed)
        if len(missing) != 1:
            continue
        s = split[h]
        if s == 1:
            pools["val"].append((h, r, sorted(tails), missing[0]))
        elif s == 2:
            pools["test"].append((h, r, sorted(tails), missing[0]))
    rng = random.Random(args.seed)
    out = os.path.join(ROOT, "data", "processed", "brink", "wikidata")
    os.makedirs(out, exist_ok=True)

    def write_kg(path, triples):
        with open(path, "w", encoding="utf-8") as f:
            for x in sorted(triples):
                f.write("\t".join(x) + "\n")

    with open(os.path.join(out, "entities.tsv"), "w", encoding="utf-8") as f:
        for q in sorted(name):
            f.write(f"{q}\t{name[q]}\n")
    write_kg(os.path.join(out, "kg_incomplete.tsv"), observed)
    write_kg(os.path.join(out, "kg_complete.tsv"), complete)
    write_kg(os.path.join(out, "removed.tsv"), removed)
    stats = {"t0": args.t0, "entities": len(name), "properties": len(props), "observed_2019": len(observed),
             "true_now": len(true_now), "removed_true_now_missing_2019": len(removed), "splits": {}}
    for sp, n in (("val", args.n_val), ("test", args.n_test)):
        pool = pools[sp]
        pick = rng.sample(pool, min(n, len(pool)))
        qs, gold = [], []
        for i, (h, r, tails, hard) in enumerate(pick):
            qid = f"wd-{sp}-{i}"
            qs.append({"id": qid, "question": f"What is the {r} of {name[h]}?", "q_entity": [h], "answer": tails,
                       "a_entity": tails, "hard_answer": [hard], "hard_triples": [[h, r, hard]]})
            gold.append({"id": qid, "question": qs[-1]["question"], "answers": sorted(name[t] for t in tails),
                         "hard_answer": name[hard], "hard_answer_raw": hard})
        with open(os.path.join(out, f"questions_{sp}.jsonl"), "w", encoding="utf-8") as f:
            for q in qs:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        json.dump(gold, open(os.path.join(out, f"gold_{sp}.json"), "w"), ensure_ascii=False, indent=1)
        stats["splits"][sp] = {"pool": len(pool), "questions": len(qs),
                               "relations": Counter(r for _, r, _, _ in pick).most_common(10)}
    json.dump(stats, open(os.path.join(out, "stats.json"), "w"), indent=2)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
