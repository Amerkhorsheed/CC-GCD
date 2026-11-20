"""Train ComplEx-N3 on BRINK's observed (incomplete) graph for the candidate generator.

Run on the RTX 3060 Ti only (the 3060 is reserved):
    CUDA_VISIBLE_DEVICES=GPU-c5cd579e-52d6-31a3-0091-04ae52932736 \
        python scripts/train_kge.py --dataset fb15k237

Output: data/kge/brink-<dataset>/{embeddings.npz, vocab.json, metrics.json};
with the default --out, metrics are also copied to results/kge/brink-<dataset>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from cc_gcd.data.brink import BrinkKG  # noqa: E402
from cc_gcd.kge.complex_kge import train_complex  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["family", "fb15k237", "wikidata", "ikgqa-webqsp"], required=True)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--reg_weight", type=float, default=5e-2, help="N3 weight")
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--amp", action="store_true", help="bf16 score matmul (large graphs)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--neg_sample", type=int, default=0, help="sampled softmax with this many shared negatives (0 = 1-vs-all)")
    args = ap.parse_args()

    if args.dataset.startswith("ikgqa-"):
        # Union of the IKGQA questions' RoG subgraphs minus every triple deleted at any level
        # (cc_gcd.data.ikgqa.IKGQA.kge_graph), written to kge_graph.tsv on first use.
        ds = args.dataset.split("-", 1)[1]
        path = os.path.join(ROOT, "data", "processed", "ikgqa", ds, "kge_graph.tsv")
        if not os.path.exists(path):
            from cc_gcd.data.ikgqa import IKGQA
            with open(path, "w", encoding="utf-8") as f:
                for t in IKGQA(os.path.join(ROOT, "data", "processed", "ikgqa"), ds).kge_graph():
                    f.write("\t".join(t) + "\n")
        with open(path, encoding="utf-8") as f:
            triples = [tuple(line.rstrip("\n").split("\t")) for line in f if line.strip()]
        name, graph = args.dataset, "IKGQA union of subgraphs minus all deleted triples (kge_graph.tsv)"
    else:
        triples = BrinkKG(os.path.join(ROOT, "data", "processed", "brink", args.dataset), version="incomplete",
                          inverse=False).triples
        name, graph = f"brink-{args.dataset}", "kg_incomplete.tsv (observed graph only)"
    out = args.out or os.path.join(ROOT, "data", "kge", name)
    metrics = train_complex(triples, out, dim=args.dim, epochs=args.epochs, lr=args.lr,
                            batch_size=args.batch_size, reg_weight=args.reg_weight, eval_every=args.eval_every,
                            seed=args.seed, amp=args.amp, device=args.device, neg_sample=args.neg_sample, log=lambda s: print(s, flush=True))
    metrics["dataset"], metrics["graph"] = name, graph
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))
    if args.out is None:
        os.makedirs(os.path.join(ROOT, "results", "kge"), exist_ok=True)
        with open(os.path.join(ROOT, "results", "kge", f"{name}.json"), "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
