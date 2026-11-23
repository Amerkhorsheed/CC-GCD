"""E1 (crucial experiment, plan section 10) on BRINK: GCR step 1 with hard vs soft KG-Tries.

All methods share the model, the prompt, the beam settings (GCR: group beam search, k = 10) and
the candidate generator (val-selected: ``--cand_mode``, R, k, candidates at hop 1). Methods:

    B0           no KG (same prompt and beams, no trie): parametric-memory baseline
    B1           hard KG-Trie on the observed graph (GCR)
    B1+          hard KG-Trie on observed + candidate edges (no penalty)
    B2:x         soft: every absent edge costs x nats
    B4:l         soft: absent edge scores l * log pi(e) (calibrated KGE prior)
    CC:c:l       soft CC-GCD with constant c (B5): l * log sigmoid(logit pi + log(1-c) - log(1-f))
    CCe:name:l   soft CC-GCD with per-edge c(e) from ``data/observation/<name>.joblib`` (if present)
    B3:a/b/c/d   per-relation penalty, one value per relation-frequency quartile (rarest first)
    OR:x         upper bound (B6-style): true absent edges free, all others cost x nats

Observed edges score 0 in every method. Reported per method: the official BRINK metrics,
faithful-path rate, the share of generated KG paths that use an absent edge, and the share
of used absent edges that are BRINK-removed (true) facts.

    CUDA_VISIBLE_DEVICES=GPU-c5cd579e-52d6-31a3-0091-04ae52932736 HF_HOME=$PWD/.hf \
      python scripts/run_e1_brink.py --dataset fb15k237 --split "val[:150]" --methods B1 B1+ B2:3 B4:1 CC:0.9:1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from tqdm import tqdm  # noqa: E402

from cc_gcd.baselines.gcr import GCRRunner, faithful_rate, split_prediction, step1_answers  # noqa: E402
from cc_gcd.calibration.kge_prior import KGEPrior  # noqa: E402
from cc_gcd.calibration.observation_model import Encoded, features  # noqa: E402
from cc_gcd.data.brink import BrinkKG  # noqa: E402
from cc_gcd.graph.candidate_generator import CandidateGenerator  # noqa: E402
from cc_gcd.kge.complex_kge import ComplExScorer, augment  # noqa: E402

BRINK_EVAL = os.path.join(ROOT, "third_party", "brink", "evaluation", "evaluate_brink.py")
F_DEFAULT = 0.01


def parse_split(split):
    m = re.fullmatch(r"(\w+)(?:\[(\d*):(\d*)\])?", split)
    return m.group(1), int(m.group(2) or 0), int(m.group(3)) if m.group(3) else None


def log_sigmoid(z):
    return -np.logaddexp(0.0, -z)


def edge_scores(method: str, pi: np.ndarray, c_edge=None, truth=None, group=None) -> tuple:
    """(lambda, s per absent edge) for a method spec; None lambda = hard trie.

    B3:a/b/c/d  per-relation penalty, one value per relation-frequency quartile (``group``, 0 = rarest).
    OR:x        B6-style upper bound: absent edges that are true (``truth``, BRINK-removed facts) are free,
                all others cost x nats; i.e. evidence that knows which hypothetical edges are true."""
    name, *arg = method.split(":")
    if name in ("B0", "B1", "B1+"):
        return None, None
    if name == "B3":
        pens = np.array([float(x) for x in arg[0].split("/")])
        g = np.zeros(len(pi), dtype=int) if group is None else np.asarray(group, dtype=int)
        return 1.0, -pens[g]
    if name == "OR":
        t = np.zeros(len(pi), dtype=bool) if truth is None else np.asarray(truth, dtype=bool)
        return float(arg[0]), np.where(t, 0.0, -1.0)
    if name == "B2":
        return float(arg[0]), -np.ones_like(pi)
    if name == "B4":
        return float(arg[0]), np.log(pi)
    if name == "CC":
        c = float(arg[0])
        z = np.log(pi) - np.log1p(-pi) + math.log(1 - c) - math.log(1 - F_DEFAULT)
        return float(arg[1]), log_sigmoid(z)
    if name == "CCe":
        c = np.full(len(pi), 0.5) if c_edge is None else np.asarray(c_edge, dtype=float)
        if len(c) and np.isnan(c).any():  # guard: candidates are absent by construction, so this should not occur
            c = np.where(np.isnan(c), np.nanmean(c) if (~np.isnan(c)).any() else 0.5, c)
        c = np.clip(c, 0.02, 0.98)
        z = np.log(pi) - np.log1p(-pi) + np.log1p(-c) - math.log(1 - F_DEFAULT)
        return float(arg[1]), log_sigmoid(z)
    raise ValueError(method)


def candidate_c(model_name: str, observed, edges_by_q: dict) -> dict:
    """Per-edge c(e) for candidate edges: features of the canonical fact on the full observed graph
    (deployment form; the candidate is absent, so nothing has to be excluded), scored by a saved model."""
    import joblib
    model = joblib.load(os.path.join(ROOT, "data", "observation", f"{model_name}.joblib"))
    def canon(e):
        h, r, t = e
        return (t, r[1:], h) if r.startswith("~") else (h, r, t)
    obs = sorted(set(observed))
    obs_set = set(obs)
    new = sorted({canon(e) for es in edges_by_q.values() for e in es} - obs_set)
    enc = Encoded(obs + new)
    kept = np.array([True] * len(obs) + [False] * len(new))
    X = features(enc, kept, folds=1)[len(obs):]
    c = model.predict(X, np.zeros(len(new), dtype=np.int64))
    cmap = dict(zip(new, c))
    return {q: np.array([cmap.get(canon(e), np.nan) for e in es]) for q, es in edges_by_q.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["family", "fb15k237", "wikidata", "wikidata-anon"], default="fb15k237")
    ap.add_argument("--split", default="val[:150]")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--model_path", default="rmanluo/GCR-Qwen2-0.5B-Instruct")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--quant", default="none", choices=["none", "4bit", "8bit"],
                    help="bitsandbytes quantization (8B on a 12 GB card); changes logits, so results are labelled")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--cand_mode", default=None, help="default: the val-selected mode for the dataset")
    ap.add_argument("--cand_R", type=int, default=10)
    ap.add_argument("--cand_k", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--kge_tag", default="", help="alternate KGE/candidates/prior, e.g. s1 (second seed)")
    args = ap.parse_args()
    tag = args.dataset + (f"-{args.kge_tag}" if args.kge_tag else "")
    cand_mode = args.cand_mode or {"family": "kge+sig", "fb15k237": "hybrid+sig", "wikidata": "hybrid+sig",
                              "wikidata-anon": "hybrid+sig"}[args.dataset]

    split, lo, hi = parse_split(args.split)
    root = os.path.join(ROOT, "data", "processed", "brink", args.dataset)
    kg = BrinkKG(root, version="incomplete", inverse=True)
    kg_obs = BrinkKG(root, version="incomplete", inverse=False)
    questions = list(kg.questions(split))[lo:hi]
    ids = {q["id"] for q in questions}
    kge_dir = os.path.join(ROOT, "data", "kge", f"brink-{tag}")
    import torch
    scorer = ComplExScorer(kge_dir, device="cuda" if torch.cuda.is_available() else "cpu")
    rr, sig = cand_mode.split("+")[0], cand_mode.endswith("+sig")
    gen = CandidateGenerator(kg_obs.triples, scorer, rel_rank=rr, signature_filter=sig)
    gen.load(os.path.join(ROOT, "data", "candidates", f"brink-{tag}", f"{cand_mode}.npz"))
    prior = KGEPrior.from_dict(json.load(open(os.path.join(kge_dir, "pi_prior.json"))))
    removed = set(augment([tuple(x.split("\t")) for x in open(os.path.join(root, "removed.tsv")).read().splitlines() if x]))
    d = kg.display

    # Relation-frequency quartiles (inverse-augmented observed graph) for B3.
    from collections import Counter
    rc = Counter(r for _, r, _ in augment(kg_obs.triples))
    qs_ = np.quantile(np.array(list(rc.values())), [0.25, 0.5, 0.75])
    rel_group = {r: int(np.searchsorted(qs_, n, side="right")) for r, n in rc.items()}

    # Candidates per question (shared by all methods), with pi and truth.
    cands = {}
    for q in questions:
        c = gen.expand(q["q_entity"], kg.adj, args.cand_R, args.cand_k, {1}, 2)
        X = np.array([[x[6], x[3], math.log1p(x[5]), math.log1p(x[4])] for x in c]).reshape(-1, 4)
        pi = prior.predict_from_dict(X) if len(c) else np.zeros(0)
        disp = [(d(x[0]), x[1], d(x[2])) for x in c]
        cands[q["id"]] = {"edges": [x[:3] for x in c], "disp": disp, "pi": pi,
                          "true": {disp[i] for i, x in enumerate(c) if tuple(x[:3]) in removed},
                          "group": np.array([rel_group.get(x[1], 0) for x in c], dtype=int)}

    for name in {m.split(":")[1] for m in args.methods if m.startswith("CCe:")}:
        cc = candidate_c(name, kg_obs.triples, {q: v["edges"] for q, v in cands.items()})
        for q, v in cands.items():
            v.setdefault("c", {})[name] = cc[q]
        allc = np.concatenate([x for x in cc.values() if len(x)])
        print(f"c(e) from {name}: mean {np.nanmean(allc):.3f}, p10 {np.nanpercentile(allc, 10):.3f}, "
              f"p90 {np.nanpercentile(allc, 90):.3f}", flush=True)

    runner = GCRRunner(args.model_path, k=args.k, dtype=args.dtype, attn_implementation="sdpa", hops=2, quant=args.quant)
    gold_all = json.load(open(os.path.join(root, f"gold_{split}.json")))
    # Quantized (8B) runs get their own summary file; full-precision runs keep the original names.
    import torch as _t
    big = "8B" in args.model_path
    msfx = "" if (args.quant == "none" and not big) else "_" + os.path.basename(args.model_path) + (
        f"-{args.quant}" if args.quant != "none" else ("" if _t.cuda.is_available() else "-cpu"))
    summary_path = os.path.join(ROOT, "results", "e1_brink", f"{tag}_{args.split.replace(':', '-')}{msfx}.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    for method in args.methods:
        mtag = method.replace(":", "_").replace("/", "-")  # method name as one path component
        model_tag = os.path.basename(args.model_path) + ("" if args.quant == "none" else f"-{args.quant}")
        out_dir = os.path.join(ROOT, "data", "e1_brink", tag, model_tag,
                               args.split.replace(":", "-"), mtag)
        os.makedirs(out_dir, exist_ok=True)
        lam, _ = edge_scores(method, np.ones(1) * 0.5)
        use_cands = method not in ("B0", "B1")

        def samples():
            for q in questions:
                cq = cands[q["id"]]
                s = kg.sample(q, hops=2, extra_edges=cq["edges"] if use_cands else None)
                if method == "B0":  # no KG: same model, prompt and beams, unconstrained (plan 7.3)
                    s["no_kg"] = True
                if lam is not None:
                    c_edge = cq["c"][method.split(":")[1]] if method.startswith("CCe:") else None
                    truth = np.array([e in cq["true"] for e in cq["disp"]], dtype=bool)
                    _, sc = edge_scores(method, cq["pi"], c_edge, truth=truth, group=cq["group"])
                    s["edge_scores"] = {e: float(v) for e, v in zip(cq["disp"], sc)}
                yield s

        pred_file = os.path.join(out_dir, "predictions.jsonl")
        t0 = time.time()
        for _try in range(6):  # resume until every question has a prediction (e.g. after CUDA OOM)
            if runner.run(samples(), pred_file, force=args.force and _try == 0, lam=lam,
                   progress=lambda it: tqdm(it, total=len(questions), desc=method, mininterval=30)) == 0:
                break
            time.sleep(120)
        rows = [r for r in map(json.loads, open(pred_file)) if r["id"] in ids]
        pred = [{"id": r["id"], "raw_output": ", ".join(step1_answers(r["prediction"]))} for r in rows]
        json.dump(pred, open(os.path.join(out_dir, "brink_pred.json"), "w"), ensure_ascii=False)
        json.dump([g for g in gold_all if g["id"] in ids], open(os.path.join(out_dir, "gold.json"), "w"), ensure_ascii=False)
        ev = json.loads(subprocess.run([sys.executable, BRINK_EVAL, "--gold", os.path.join(out_dir, "gold.json"),
                                        "--pred", os.path.join(out_dir, "brink_pred.json")],
                                       capture_output=True, text=True, check=True).stdout)
        # Absent edges actually used in generated paths, and how many are BRINK-removed (true) facts.
        used = used_true = 0
        for r in rows:
            cq = cands[r["id"]]
            cset = set(cq["disp"])
            for p in (r["prediction"] if isinstance(r["prediction"], list) else [r["prediction"]]):
                path, _ = split_prediction(p)
                parts = [x.strip() for x in path.split(" -> ")]
                for i in range(0, len(parts) - 2, 2):
                    e = (parts[i], parts[i + 1], parts[i + 2])
                    if e in cset:
                        used += 1
                        used_true += e in cq["true"]
        fr = faithful_rate(rows)
        via = sum(r.get("n_paths_via_candidate", 0) for r in rows)
        summary[method] = {**{k: round(v, 4) if isinstance(v, float) else v for k, v in ev.items()}, **fr,
                           "paths_via_absent_edge": via,
                           "share_kg_paths_via_absent_edge": round(via / fr["paths_in_kg"], 4) if fr["paths_in_kg"] else None,
                           "absent_edges_used": used, "absent_edges_used_true": used_true,
                           "absent_edge_precision": round(used_true / used, 4) if used else None,
                           "questions": len(rows), "seconds": round(time.time() - t0, 1),
                           "candidates": f"{cand_mode} R{args.cand_R} k{args.cand_k} hop1", "model": args.model_path}
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        json.dump(summary, open(summary_path, "w"), indent=1)
        s = summary[method]
        print(f"{method:12s} Hits@Any {s.get('Hits@Any')} Hits@Hard {s.get('Hits@Hard')} F1 {s.get('F1')} | faithful "
              f"{s['faithful_path_rate']} | via absent {s['share_kg_paths_via_absent_edge']} | absent precision "
              f"{s['absent_edge_precision']} | {s['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
