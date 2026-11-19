"""Natural missingness from two Wikidata snapshots, for calibrating c(e) (plan sections 6.2, 7.1).

A fact true now (current Wikidata) and already true at T0 is *recorded* if it was in
Wikidata at T0 and *unrecorded* otherwise. c_nat(e) = P(in Wikidata at T0 | true at T0, x_e).
Full dumps do not fit on this machine, so both snapshots are read per entity from the
public API: the current claims (wbgetentities) and the entity's last revision at or before
T0 (prop=revisions, rvstart=T0). Both are genuine Wikidata states.

Universe: BRINK-FB15k-237's entities mapped to Wikidata via the Freebase ID property P646
(one-to-one mappings only), so natural missingness lives on the same entities as the BRINK
experiments. Facts are item-valued, non-deprecated statements (h, P, q) whose head is a
sampled universe entity and whose tail is a universe entity.

"Not yet true at T0" is not missingness. A current statement is dropped when a start time
(P580) or point in time (P585) qualifier is after T0, or when its tail item was created after
T0 (Wikidata item numbers are sequential; the last item created before T0 is found by binary
search on first-revision timestamps). Statements present at T0 but gone now are kept in the
observed T0 graph (they were observable) but get no label (they are not true by the current
snapshot). Remaining bias (plan 6.2): facts that are true but missing in both snapshots stay
invisible, and facts without time qualifiers may still have become true after T0.

Two routes. The dump route (used: the API throttles this connection to one call per 17-45 s):
    data_prep/wikidata_dump_stream.py streams the 2019-08-26 (archive.org) and current dumps
    python data_prep/wikidata_snapshots.py build-dump --t0 2019-08-26
The API route (per-entity revisions; needs an unthrottled connection), cached and resumable under
data/raw/wikidata/ and data/processed/wikidata/:
    python data_prep/wikidata_snapshots.py map          # MID -> QID (API search on P646)
    python data_prep/wikidata_snapshots.py threshold    # last QID created before T0
    python data_prep/wikidata_snapshots.py fetch --n 3000
    python data_prep/wikidata_snapshots.py build

Requests are sequential, use maxlag and back off on 429/503, and send a generic User-Agent.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

API = "https://www.wikidata.org/w/api.php"
UA = "cc-gcd-research/0.1 (academic knowledge-graph study; python urllib)"
RAW = os.path.join(ROOT, "data", "raw", "wikidata")
OUT = os.path.join(ROOT, "data", "processed", "wikidata")
TIME_QUALIFIERS = {"P580": "start", "P582": "end", "P585": "point"}


# ---------------------------------------------------------------------------- http
def get(url: str, params: dict, accept: str = "application/json", tries: int = 6) -> dict:
    q = urllib.parse.urlencode(params)
    for i in range(tries):
        req = urllib.request.Request(f"{url}?{q}", headers={"User-Agent": UA, "Accept": accept})
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                data = json.loads(r.read().decode("utf-8"))
            if isinstance(data, dict) and data.get("error", {}).get("code") == "maxlag":
                raise urllib.error.HTTPError(url, 503, "maxlag", {}, None)
            return data
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            wait = min(60, 2 ** i * 2)
            ra = getattr(e, "headers", None) and e.headers.get("Retry-After")
            if ra and ra.isdigit():
                if int(ra) > 120:  # the service asks us to stay away: stop instead of hammering it
                    raise RuntimeError(f"{url} asked to retry after {ra}s; stopping") from e
                wait = max(wait, int(ra))
            print(f"  retry {i + 1}/{tries} after {type(e).__name__}: {getattr(e, 'code', '')} (wait {wait}s)", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"request failed: {url}")


def api(params: dict) -> dict:
    return get(API, {"format": "json", "formatversion": 2, "maxlag": 5, **params})


# ---------------------------------------------------------------------------- claims
def qnum(qid: str) -> int:
    return int(qid[1:])


def parse_time(v) -> str:
    """Wikidata time '+2019-05-00T00:00:00Z' -> '2019-05-00' (precision-insensitive prefix compare)."""
    if not isinstance(v, dict) or "time" not in v:
        return ""
    return v["time"].lstrip("+")[:10]


def item_claims(entity: dict) -> list:
    """[(P, tail QID, rank, start, end, point)] for item-valued, non-deprecated statements."""
    out = []
    for prop, stmts in (entity.get("claims") or {}).items():
        for st in stmts:
            if st.get("rank") == "deprecated":
                continue
            snak = st.get("mainsnak", {})
            dv = snak.get("datavalue", {})
            if snak.get("snaktype") != "value" or dv.get("type") != "wikibase-entityid":
                continue
            val = dv.get("value", {})
            tail = val.get("id") or (f"Q{val['numeric-id']}" if val.get("entity-type") == "item" else None)
            if not tail or not tail.startswith("Q"):
                continue
            quals = {name: "" for name in TIME_QUALIFIERS.values()}
            for qp, name in TIME_QUALIFIERS.items():
                for qs in (st.get("qualifiers") or {}).get(qp, []):
                    t = parse_time(qs.get("datavalue", {}).get("value"))
                    if t and (not quals[name] or t > quals[name]):
                        quals[name] = t
            out.append([prop, tail, st.get("rank", "normal"), quals["start"], quals["end"], quals["point"]])
    return out


# ---------------------------------------------------------------------------- steps
def step_map(args):
    """MID -> QID through the API's search (haswbstatement:P646=..., OR'ed per batch), then each hit's
    own P646 values from its claims. (The SPARQL endpoint throttles this connection to one query per
    ~1,000 s, so it is not used.)"""
    from cc_gcd.data.brink import BrinkKG
    root = os.path.join(ROOT, "data", "processed", "brink", "fb15k237")
    kg = BrinkKG(root, version="complete", inverse=False)
    mids = sorted({e for h, _, t in kg.triples for e in (h, t)})
    want = set(mids)
    pairs = set()

    def search(batch):
        r = api({"action": "query", "list": "search", "srlimit": 50, "srprop": "",
                 "srsearch": "haswbstatement:" + "|".join(f"P646={m}" for m in batch)})
        hits = [x["title"] for x in r["query"]["search"]]
        if r["query"]["searchinfo"]["totalhits"] > len(hits) and len(batch) > 1:  # too many hits: split
            half = len(batch) // 2
            return search(batch[:half]) + search(batch[half:])
        return hits

    for i in range(0, len(mids), 40):
        qids = search(mids[i:i + 40])
        for j in range(0, len(qids), 50):
            ents = api({"action": "wbgetentities", "ids": "|".join(qids[j:j + 50]), "props": "claims"}).get("entities", {})
            for q, ent in ents.items():
                for st in (ent.get("claims") or {}).get("P646", []):
                    v = st.get("mainsnak", {}).get("datavalue", {}).get("value")
                    if isinstance(v, str) and v in want:
                        pairs.add((v, q))
        time.sleep(args.sleep)
        if (i // 40) % 25 == 0:
            print(f"  {min(i + 40, len(mids))}/{len(mids)} MIDs, {len(pairs)} pairs", flush=True)
    by_mid, by_q = Counter(m for m, _ in pairs), Counter(q for _, q in pairs)
    one = sorted({(m, q) for m, q in pairs if by_mid[m] == 1 and by_q[q] == 1})
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "fb15k237_qid.tsv"), "w") as f:
        for m, q in one:
            f.write(f"{m}\t{q}\n")
    stats = {"mids": len(mids), "pairs": len(pairs), "one_to_one": len(one),
             "mid_with_several_items": sum(1 for c in by_mid.values() if c > 1),
             "item_with_several_mids": sum(1 for c in by_q.values() if c > 1)}
    json.dump(stats, open(os.path.join(OUT, "map_stats.json"), "w"), indent=2)
    print(json.dumps(stats))


def first_revision_ts(qnum_: int):
    r = api({"action": "query", "prop": "revisions", "titles": f"Q{qnum_}", "rvlimit": 1, "rvdir": "newer",
             "rvprop": "timestamp"})
    page = r["query"]["pages"][0]
    if page.get("missing") or "revisions" not in page:
        return None
    return page["revisions"][0]["timestamp"]


def step_threshold(args):
    """Largest item number whose first revision is before T0 (probing nearby numbers over gaps)."""
    t0 = args.t0 + "T00:00:00Z"

    def ts_near(n):
        for d in range(0, 40):
            ts = first_revision_ts(n + d)
            time.sleep(0.2)
            if ts:
                return ts
        return None

    lo, hi = 1, 140_000_000
    while hi - lo > 1000:
        mid = (lo + hi) // 2
        ts = ts_near(mid)
        if ts is None or ts >= t0:
            hi = mid
        else:
            lo = mid
        print(f"  Q{mid}: {ts}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    res = {"t0": args.t0, "last_qnum_before_t0": lo, "first_qnum_after_t0": hi, "resolution": 1000}
    json.dump(res, open(os.path.join(OUT, f"qid_threshold_{args.t0}.json"), "w"), indent=2)
    print(json.dumps(res))


def sample_heads(n: int, seed: int):
    """Heads stratified by FB15k-237 degree quartile (equal numbers per quartile)."""
    from cc_gcd.data.brink import BrinkKG
    qmap = dict(line.rstrip("\n").split("\t") for line in open(os.path.join(OUT, "fb15k237_qid.tsv")))
    kg = BrinkKG(os.path.join(ROOT, "data", "processed", "brink", "fb15k237"), version="complete", inverse=False)
    deg = Counter()
    for h, _, t in kg.triples:
        deg[h] += 1
        deg[t] += 1
    mids = sorted(qmap, key=lambda m: (deg[m], m))
    rng = random.Random(seed)
    k = len(mids) // 4
    out = []
    for i in range(4):
        part = mids[i * k:(i + 1) * k] if i < 3 else mids[3 * k:]
        out += rng.sample(part, min(len(part), n // 4))
    return [qmap[m] for m in out], qmap


def step_fetch(args):
    heads, _ = sample_heads(args.n, args.seed)
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, f"claims_t0_{args.t0}.jsonl.gz")
    done = set()
    if os.path.exists(path):
        with gzip.open(path, "rt") as f:
            done = {json.loads(line)["qid"] for line in f}
    todo = [q for q in heads if q not in done]
    print(f"{len(heads)} sampled heads, {len(done)} cached, {len(todo)} to fetch", flush=True)
    t0 = args.t0 + "T00:00:00Z"
    with gzip.open(path, "at") as fout:
        for i in range(0, len(todo), 50):
            batch = todo[i:i + 50]
            cur = api({"action": "wbgetentities", "ids": "|".join(batch), "props": "claims"}).get("entities", {})
            for q in batch:
                ent = cur.get(q, {})
                rec = {"qid": q, "new_missing": "missing" in ent or "claims" not in ent,
                       "new_redirect": ent.get("id", q) != q, "new": item_claims(ent)}
                r = api({"action": "query", "prop": "revisions", "titles": q, "rvlimit": 1, "rvdir": "older",
                         "rvstart": t0, "rvprop": "content|timestamp|ids", "rvslots": "main"})
                page = r["query"]["pages"][0]
                revs = page.get("revisions") or []
                if revs:
                    rec["old_ts"] = revs[0]["timestamp"]
                    try:
                        old = json.loads(revs[0]["slots"]["main"]["content"])
                        rec["old"] = item_claims(old)
                        rec["old_redirect"] = "redirect" in old
                    except (KeyError, json.JSONDecodeError):
                        rec["old"], rec["old_error"] = [], True
                else:
                    rec["old"], rec["old_ts"] = [], None
                fout.write(json.dumps(rec) + "\n")
                time.sleep(args.sleep)
            fout.flush()
            print(f"  {min(i + 50, len(todo))}/{len(todo)} fetched", flush=True)


def step_build(args):
    _, qmap = sample_heads(args.n, args.seed)
    universe = set(qmap.values())
    thr = json.load(open(os.path.join(OUT, f"qid_threshold_{args.t0}.json")))["last_qnum_before_t0"]
    recs = [json.loads(line) for line in gzip.open(os.path.join(RAW, f"claims_t0_{args.t0}.jsonl.gz"), "rt")]
    t0 = args.t0
    stats = Counter()
    true_now, old_all = set(), set()
    drop = Counter()
    for rec in recs:
        q = rec["qid"]
        if rec.get("new_missing") or rec.get("new_redirect") or not rec.get("old_ts") or rec.get("old_redirect") \
                or rec.get("old_error"):
            stats["heads_skipped"] += 1
            continue
        stats["heads_used"] += 1
        for p, tail, rank, *_ in rec["old"]:
            if tail in universe and tail != q:
                old_all.add((q, p, tail))
        for p, tail, rank, start, end, point in rec["new"]:
            if tail not in universe or tail == q:
                continue
            if start and start > t0:
                drop["start_after_t0"] += 1
                continue
            if point and point > t0:
                drop["point_in_time_after_t0"] += 1
                continue
            if qnum(tail) > thr:
                drop["tail_created_after_t0"] += 1
                continue
            true_now.add((q, p, tail))
    labelled = sorted(true_now)
    old_only = sorted(old_all - true_now)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"natural_t0_{t0}.tsv")
    with open(path, "w") as f:
        f.write("head\trelation\ttail\tin_t0\tlabelled\n")
        for h, p, t in labelled:
            f.write(f"{h}\t{p}\t{t}\t{int((h, p, t) in old_all)}\t1\n")
        for h, p, t in old_only:  # observable at T0, not true now: features only
            f.write(f"{h}\t{p}\t{t}\t1\t0\n")
    rec_rate = sum((t in old_all) for t in labelled) / max(len(labelled), 1)
    per_p = Counter(p for _, p, _ in labelled)
    res = {"t0": t0, "snapshot_new_fetched": "2026-11-19", "universe_entities": len(universe), **stats,
           "labelled_facts_true_now": len(labelled), "recorded_at_t0": round(rec_rate, 4),
           "observable_at_t0_not_true_now": len(old_only), "dropped_not_yet_true": dict(drop),
           "relations_labelled": len(per_p), "top_relations": per_p.most_common(15),
           "last_qnum_before_t0": thr}
    json.dump(res, open(os.path.join(OUT, f"natural_t0_{t0}_stats.json"), "w"), indent=2)
    print(json.dumps(res, indent=1))


def step_build_dump(args):
    """Labels from two streamed dumps (data_prep/wikidata_dump_stream.py).

    Universe: items mapped one-to-one MID <-> QID in the new dump that also exist in the old dump
    carrying the same MID (an item without P646 at T0 is not captured by the old stream, so it is
    dropped rather than treated as unrecorded). Universe tails therefore existed at T0."""
    t0 = args.t0
    new = {r["qid"]: r for r in map(json.loads, gzip.open(os.path.join(RAW, f"dump_{args.new}.jsonl.gz"), "rt"))}
    old = {r["qid"]: r for r in map(json.loads, gzip.open(os.path.join(RAW, f"dump_{args.old}.jsonl.gz"), "rt"))}
    pairs = {(m, q) for q, r in new.items() for m in r["mids"]}
    by_mid, by_q = Counter(m for m, _ in pairs), Counter(q for _, q in pairs)
    one = {(m, q) for m, q in pairs if by_mid[m] == 1 and by_q[q] == 1}
    universe = {q for m, q in one if q in old and m in old[q]["mids"]}
    stats = Counter({"items_new_stream": len(new), "items_old_stream": len(old), "mid_qid_pairs_new": len(pairs),
                     "one_to_one_new": len(one), "universe": len(universe),
                     "one_to_one_not_in_old_with_same_mid": len(one) - len(universe)})
    drop = Counter()
    true_now, old_all = set(), set()
    for q in sorted(universe):
        for p, tail, *_ in old[q]["claims"]:
            if tail in universe and tail != q:
                old_all.add((q, p, tail))
        for p, tail, rank, start, end, point in new[q]["claims"]:
            if tail not in universe or tail == q:
                continue
            if start and start > t0:
                drop["start_after_t0"] += 1
                continue
            if point and point > t0:
                drop["point_in_time_after_t0"] += 1
                continue
            true_now.add((q, p, tail))
    # Properties essentially absent at T0 (schema change, not missingness): fewer than min_old_uses
    # uses inside the 2019 universe graph. Their facts are dropped from labels and features alike.
    uses = Counter(p for _, p, _ in old_all)
    rare = {p for _, p, _ in true_now | old_all if uses[p] < args.min_old_uses}
    drop["property_rare_at_t0"] = sum(1 for _, p, _ in true_now if p in rare)
    true_now = {x for x in true_now if x[1] not in rare}
    old_all = {x for x in old_all if x[1] not in rare}
    labelled = sorted(true_now)
    old_only = sorted(old_all - true_now)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"natural_t0_{t0}.tsv"), "w") as f:
        f.write("head\trelation\ttail\tin_t0\tlabelled\n")
        for h, p, t in labelled:
            f.write(f"{h}\t{p}\t{t}\t{int((h, p, t) in old_all)}\t1\n")
        for h, p, t in old_only:
            f.write(f"{h}\t{p}\t{t}\t1\t0\n")
    with open(os.path.join(OUT, "fb15k237_qid.tsv"), "w") as f:
        for m, q in sorted(one):
            if q in universe:
                f.write(f"{m}\t{q}\n")
    rec = sum(t in old_all for t in labelled) / max(len(labelled), 1)
    per_p = Counter(p for _, p, _ in labelled)
    res = {"t0": t0, "old_dump": args.old, "new_dump": args.new, "min_old_uses": args.min_old_uses,
           "properties_dropped_rare_at_t0": len(rare), **stats,
           "labelled_facts_true_now": len(labelled), "recorded_at_t0": round(rec, 4),
           "observable_at_t0_not_true_now": len(old_only), "dropped_not_yet_true": dict(drop),
           "relations_labelled": len(per_p), "top_relations": per_p.most_common(15)}
    json.dump(res, open(os.path.join(OUT, f"natural_t0_{t0}_stats.json"), "w"), indent=2)
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["map", "threshold", "fetch", "build", "build-dump"])
    ap.add_argument("--t0", default="2019-01-01")
    ap.add_argument("--old", default="20190826", help="build-dump: old dump tag (data/raw/wikidata/dump_<tag>.jsonl.gz)")
    ap.add_argument("--new", default="20261117", help="build-dump: new dump tag")
    ap.add_argument("--min_old_uses", type=int, default=20,
                    help="build-dump: drop properties with fewer uses than this in the T0 universe graph")
    ap.add_argument("--n", type=int, default=3000, help="sampled head entities (stratified by FB15k-237 degree)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.25)
    args = ap.parse_args()
    {"map": step_map, "threshold": step_threshold, "fetch": step_fetch, "build": step_build,
     "build-dump": step_build_dump}[args.step](args)


if __name__ == "__main__":
    main()
