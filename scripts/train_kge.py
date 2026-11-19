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
    ap.add_argument("--dataset", choices=["family", "fb15k237"], required=True)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--reg_weight", type=float, default=5e-2, help="N3 weight")
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    kg = BrinkKG(os.path.join(ROOT, "data", "processed", "brink", args.dataset), version="incomplete", inverse=False)
    out = args.out or os.path.join(ROOT, "data", "kge", f"brink-{args.dataset}")
    metrics = train_complex(kg.triples, out, dim=args.dim, epochs=args.epochs, lr=args.lr,
                            batch_size=args.batch_size, reg_weight=args.reg_weight, eval_every=args.eval_every,
                            seed=args.seed, log=lambda s: print(s, flush=True))
    metrics["dataset"] = f"brink-{args.dataset}"
    metrics["graph"] = "kg_incomplete.tsv (observed graph only)"
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))
    if args.out is None:
        os.makedirs(os.path.join(ROOT, "results", "kge"), exist_ok=True)
        with open(os.path.join(ROOT, "results", "kge", f"brink-{args.dataset}.json"), "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
