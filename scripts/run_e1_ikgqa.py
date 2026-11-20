"""E1 on IKGQA-WebQSP: GCR step 1 with hard vs soft KG-Tries at one IKG level.

Same methods as scripts/run_e1_brink.py (B1, B1+, B2:x, B4:l, CC:c:l). Candidates are the
subgraph-local generator chosen for its balance on IKGQA (``--cand_mode`` local hybrid, R = 10,
k = 5, hop 1; results/candidates/ikgqa-webqsp.json), built from the leak-free union KGE. pi is the
rank-free IKGQA prior (results/kge_prior/ikgqa-webqsp.json). Observed edges get inverse
edges ``~r`` as on BRINK, and score 0. Questions: ``--subset dev`` (index % 5 == 0, used to tune)
or ``eval`` (the rest), first ``--limit``.

Metrics: GCR's step-1 Hit and answer F1 (src/utils/qa_utils), faithful-path rate, share of KG
paths through an absent edge, and the share of used absent edges that are this question's
deleted facts (true by construction).

    CUDA_VISIBLE_DEVICES=GPU-c5cd579e-52d6-31a3-0091-04ae52932736 HF_HOME=$PWD/.hf \
      python scripts/run_e1_ikgqa.py --level 0.8 --subset dev --limit 100 --methods B1 B1+ B4:1 CC:0.9:1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from tqdm import tqdm  # noqa: E402

from cc_gcd.baselines.gcr import GCR_ROOT, GCRRunner, faithful_rate, split_prediction, step1_answers  # noqa: E402
from cc_gcd.calibration.kge_prior import KGEPrior  # noqa: E402
from cc_gcd.data.ikgqa import IKGQA, level_tag  # noqa: E402
from cc_gcd.graph.candidate_generator import CandidateGenerator  # noqa: E402
from cc_gcd.kge.complex_kge import ComplExScorer  # noqa: E402
from run_e1_brink import candidate_c, edge_scores  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=float, default=0.8)
    ap.add_argument("--subset", choices=["dev", "eval"], default="dev")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--model_path", default="rmanluo/GCR-Qwen2-0.5B-Instruct")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--cand_mode", default="hybrid")
    ap.add_argument("--cand_R", type=int, default=10)
    ap.add_argument("--cand_k", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cand_device", default="cuda", help="device for the candidate pre-pass (cpu keeps the GPU free)")
    args = ap.parse_args()
    sys.path.insert(0, GCR_ROOT)
    from src.utils.qa_utils import eval_f1, eval_hit

    ik = IKGQA(os.path.join(ROOT, "data", "processed", "ikgqa"), "webqsp")
    samples = [s for qi, s in enumerate(ik.samples(args.level)) if (qi % 5 == 0) == (args.subset == "dev")][:args.limit]
    tag = f"{level_tag(args.level)}_{args.subset}{len(samples)}"

    # Candidates (local tables), computed once (cached on disk) and freed before the LLM is loaded.
    import pickle
    cache = os.path.join(ROOT, "data", "e1_ikgqa", f"cands_{tag}_{args.cand_mode}_R{args.cand_R}k{args.cand_k}.pkl")
    if os.path.exists(cache):
        cands = pickle.load(open(cache, "rb"))
    else:
        kge_dir = os.path.join(ROOT, "data", "kge", "ikgqa-webqsp")
        graph = os.path.join(ROOT, "data", "processed", "ikgqa", "webqsp", "kge_graph.tsv")
        triples = [tuple(line.rstrip("\n").split("\t")) for line in open(graph, encoding="utf-8") if line.strip()]
        scorer = ComplExScorer(kge_dir, device=args.cand_device)
        gen = CandidateGenerator(triples, scorer, rel_rank=args.cand_mode, rel_pool=50)
        prior = KGEPrior.from_dict(json.load(open(os.path.join(kge_dir, "pi_prior.json"))))
        cands = {}
        t0 = time.time()
        for s in samples:
            obs = {(h.strip(), r, t.strip()) for h, r, t in s["graph"]}
            nodes = {e for h, _, t in obs for e in (h, t)}
            seeds = [e.strip() for e in s["q_entity"]]
            c = [x for u in seeds for x in gen.candidates_from(u, gen.local_table(u, nodes), args.cand_R, args.cand_k)]
            c = [x for x in c if (x[0], x[1], x[2]) not in obs]
            X = np.array([[x[6], x[3]] for x in c]).reshape(-1, 2)
            pi = prior.predict_from_dict(X) if len(c) else np.zeros(0)
            deleted = set()
            for h, r, t in s["removed_triples"]:
                deleted.add((h.strip(), r, t.strip()))
                deleted.add((t.strip(), "~" + r, h.strip()))
            cands[s["id"]] = {"edges": [tuple(x[:3]) for x in c], "pi": pi,
                              "true": {tuple(x[:3]) for x in c if tuple(x[:3]) in deleted}}
        print(f"candidates for {len(samples)} questions in {time.time() - t0:.0f}s", flush=True)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        pickle.dump(cands, open(cache, "wb"))
        del gen, scorer
        torch.cuda.empty_cache()
    if args.methods == ["none"]:
        return
    for name in {m.split(":")[1] for m in args.methods if m.startswith("CCe:")}:
        graph = os.path.join(ROOT, "data", "processed", "ikgqa", "webqsp", "kge_graph.tsv")
        triples = [tuple(line.rstrip("\n").split("\t")) for line in open(graph, encoding="utf-8") if line.strip()]
        cc = candidate_c(name, triples, {q: v["edges"] for q, v in cands.items()})
        for q, v in cands.items():
            v.setdefault("c", {})[name] = cc[q]
        allc = np.concatenate([x for x in cc.values() if len(x)])
        print(f"c(e) from {name}: mean {np.nanmean(allc):.3f}, p10 {np.nanpercentile(allc, 10):.3f}, "
              f"p90 {np.nanpercentile(allc, 90):.3f}, missing {np.isnan(allc).mean():.3f}", flush=True)

    runner = GCRRunner(args.model_path, k=args.k, dtype="bf16", attn_implementation="sdpa", hops=2)
    summary_path = os.path.join(ROOT, "results", "e1_ikgqa", f"webqsp_{tag}.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    for method in args.methods:
        out_dir = os.path.join(ROOT, "data", "e1_ikgqa", os.path.basename(args.model_path), tag, method.replace(":", "_"))
        os.makedirs(out_dir, exist_ok=True)
        lam, _ = edge_scores(method, np.ones(1) * 0.5)
        use_cands = method != "B1"

        def gen_samples():
            for s in samples:
                cq = cands[s["id"]]
                g = [[h.strip(), r, t.strip()] for h, r, t in s["graph"]]
                g += [[t, "~" + r, h] for h, r, t in g]
                s2 = {k: s[k] for k in ("id", "question", "q_entity", "a_entity", "answer")}
                s2["q_entity"] = [e.strip() for e in s["q_entity"]]
                s2["graph"] = g + ([list(e) for e in cq["edges"]] if use_cands else [])
                if use_cands:
                    s2["candidate_graph"] = [list(e) for e in cq["edges"]]
                if lam is not None:
                    c_edge = cq["c"][method.split(":")[1]] if method.startswith("CCe:") else None
                    _, sc = edge_scores(method, cq["pi"], c_edge)
                    s2["edge_scores"] = {e: float(v) for e, v in zip(cq["edges"], sc)}
                yield s2

        pred_file = os.path.join(out_dir, "predictions.jsonl")
        t0 = time.time()
        runner.run(gen_samples(), pred_file, force=args.force, lam=lam,
                   progress=lambda it: tqdm(it, total=len(samples), desc=method, mininterval=30))
        ids = {s["id"] for s in samples}
        rows = [r for r in map(json.loads, open(pred_file)) if r["id"] in ids]
        hit = f1 = 0.0
        used = used_true = 0
        for r in rows:
            preds = r["prediction"] if isinstance(r["prediction"], list) else [r["prediction"]]
            answers = list(set(r["ground_truth"]))
            ans = step1_answers(preds)
            hit += eval_hit(" ".join(preds), answers)
            f1 += eval_f1(ans, answers)[0] if ans else 0.0
            cq = cands[r["id"]]
            cset = set(cq["edges"])
            for p in preds:
                path, _ = split_prediction(p)
                parts = [x.strip() for x in path.split(" -> ")]
                for i in range(0, len(parts) - 2, 2):
                    e = (parts[i], parts[i + 1], parts[i + 2])
                    if e in cset:
                        used += 1
                        used_true += e in cq["true"]
        n = max(len(rows), 1)
        fr = faithful_rate(rows)
        via = sum(r.get("n_paths_via_candidate", 0) for r in rows)
        summary[method] = {"Hit": round(hit / n, 4), "F1": round(f1 / n, 4), **fr,
                           "share_kg_paths_via_absent_edge": round(via / fr["paths_in_kg"], 4) if fr["paths_in_kg"] else None,
                           "absent_edges_used": used, "absent_edges_used_true": used_true,
                           "absent_edge_precision": round(used_true / used, 4) if used else None,
                           "questions": len(rows), "seconds": round(time.time() - t0, 1),
                           "candidates": f"local {args.cand_mode}+sig R{args.cand_R} k{args.cand_k} hop1"}
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        json.dump(summary, open(summary_path, "w"), indent=1)
        s = summary[method]
        print(f"{method:12s} Hit {s['Hit']} F1 {s['F1']} | faithful {s['faithful_path_rate']} | via absent "
              f"{s['share_kg_paths_via_absent_edge']} | absent precision {s['absent_edge_precision']} | {s['seconds']}s",
              flush=True)


if __name__ == "__main__":
    main()
