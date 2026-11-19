"""Stream a Wikidata JSON dump and keep only items carrying one of BRINK-FB15k-237's Freebase IDs.

Full dumps (157 GB current, 55 GB for 2019-08-26) do not fit on this machine, so they are
streamed and filtered on the fly; only compact claim records are written:

    python data_prep/wikidata_dump_stream.py download URL \
      | pigz -dc \
      | python data_prep/wikidata_dump_stream.py filter --out data/raw/wikidata/dump_YYYYMMDD.jsonl.gz

``download`` writes the raw bytes to stdout and, when a connection drops, reconnects with an
HTTP Range request at the exact byte offset, so the decompressor downstream sees one
uninterrupted stream. ``filter`` rejects lines without a P646 (Freebase ID) key by substring
test, rejects lines whose /m/ strings are not ours by regex, and only then parses the JSON;
for kept items it stores the QID, its P646 values, and item-valued non-deprecated claims
with their time qualifiers (``wikidata_snapshots.item_claims``).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

UA = "cc-gcd-research/0.1 (academic knowledge-graph study)"
CHUNK = 4 << 20


def download(url: str) -> None:
    out = sys.stdout.buffer
    offset, total, fails = 0, None, 0
    t0 = time.time()
    last = t0
    while total is None or offset < total:
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": f"bytes={offset}-"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if total is None:
                    cr = r.headers.get("Content-Range")  # bytes 0-N/TOTAL
                    total = int(cr.rsplit("/", 1)[1]) if cr else int(r.headers["Content-Length"])
                    print(f"[download] {url} {total / 2**30:.1f} GiB", file=sys.stderr, flush=True)
                while True:
                    b = r.read(CHUNK)
                    if not b:
                        break
                    out.write(b)
                    offset += len(b)
                    fails = 0
                    if time.time() - last > 300:
                        last = time.time()
                        rate = offset / (last - t0) / 2**20
                        print(f"[download] {offset / 2**30:.1f}/{total / 2**30:.1f} GiB, {rate:.1f} MiB/s, "
                              f"eta {(total - offset) / max(rate, 1e-9) / 2**20 / 3600:.1f} h", file=sys.stderr, flush=True)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            fails += 1
            if fails > 30:
                raise
            print(f"[download] reconnect at {offset} after {type(e).__name__} ({fails})", file=sys.stderr, flush=True)
            time.sleep(min(60, 5 * fails))
    out.flush()
    print(f"[download] done {offset} bytes in {(time.time() - t0) / 3600:.2f} h", file=sys.stderr, flush=True)


def filter_items(out_path: str) -> None:
    from cc_gcd.data.brink import BrinkKG
    from wikidata_snapshots import item_claims

    kg = BrinkKG(os.path.join(ROOT, "data", "processed", "brink", "fb15k237"), version="complete", inverse=False)
    mids = {e.encode() for h, _, t in kg.triples for e in (h, t)}
    # The 2019 dump escapes slashes in strings ("\/m\/016pp7"); accept both forms.
    mid_re = re.compile(rb'"(\\?/m\\?/[0-9a-z_]+)"')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = kept = parsed = 0
    t0 = time.time()
    with gzip.open(out_path + ".part", "wt") as fout:
        for line in sys.stdin.buffer:
            n += 1
            if n % 10_000_000 == 0:
                print(f"[filter] {n / 1e6:.0f}M lines, {parsed} parsed, {kept} kept, {(time.time() - t0) / 60:.0f} min",
                      file=sys.stderr, flush=True)
            if b'"P646"' not in line or not line.startswith(b'{"type":"item"'):
                continue
            if not any(m.replace(b"\\", b"") in mids for m in mid_re.findall(line)):
                continue
            parsed += 1
            ent = json.loads(line.rstrip(b",\r\n"))
            p646 = [st.get("mainsnak", {}).get("datavalue", {}).get("value")
                    for st in (ent.get("claims") or {}).get("P646", []) if st.get("rank") != "deprecated"]
            ours = sorted({v for v in p646 if isinstance(v, str) and v.encode() in mids})
            if not ours:
                continue
            kept += 1
            fout.write(json.dumps({"qid": ent["id"], "mids": ours, "claims": item_claims(ent)}) + "\n")
    os.replace(out_path + ".part", out_path)
    print(f"[filter] done: {n} lines, {parsed} parsed, {kept} items kept, {(time.time() - t0) / 3600:.2f} h",
          file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("url")
    f = sub.add_parser("filter")
    f.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "download":
        download(args.url)
    else:
        filter_items(args.out)


if __name__ == "__main__":
    main()
