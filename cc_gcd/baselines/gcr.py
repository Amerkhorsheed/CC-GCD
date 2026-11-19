"""Baseline B1: GCR step 1 (graph-constrained decoding with a KG-Trie).

Wraps the official code in third_party/graph-constrained-reasoning so any
RoG-format sample (id, question, q_entity, a_entity, answer, graph) can be
decoded with the paper's settings (zero-shot prompt, group beam search, k=10).
The KG-Trie paths are passed in explicitly through the sample's ``paths`` key,
which GCR's prompt builder uses instead of its own DiGraph DFS.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Callable, Iterable, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GCR_ROOT = os.path.join(ROOT, "third_party", "graph-constrained-reasoning")
if GCR_ROOT not in sys.path:
    sys.path.insert(0, GCR_ROOT)

from cc_gcd.data.paths import dfs_paths, dfs_paths_digraph  # noqa: E402

PATH_MODES = {"multigraph": dfs_paths, "gcr_digraph": dfs_paths_digraph}


def path_string(path) -> str:
    return " -> ".join([path[0][0]] + [x for _, r, t in path for x in (r, t)])


def split_prediction(p: str):
    path = p.split("# Answer:\n")[0].split("# Reasoning Path:\n")[-1].strip()
    answer = p.split("# Answer:\n")[-1].strip()
    return path, answer


def path_validity(prediction, trie_paths) -> tuple:
    """(paths generated, paths that are exact KG-Trie paths).

    GCR constrains decoding only between <PATH> and </PATH>; a beam that writes a
    path without opening <PATH> decodes freely and may leave the graph. This is
    the faithful-reasoning rate of the GCR paper (Fig. 5).
    """
    valid = {path_string(p) for p in trie_paths}
    preds = prediction if isinstance(prediction, list) else [prediction]
    n = ok = 0
    for p in preds:
        path, _ = split_prediction(p)
        if path:
            n += 1
            ok += path in valid
    return n, ok


def step1_answers(prediction) -> List[str]:
    preds = prediction if isinstance(prediction, list) else [prediction]
    answers: List[str] = []
    for p in preds:
        _, ans = split_prediction(p)
        if ans and ans not in answers:
            answers.append(ans)
    return answers


def _load_prediction_fn() -> Callable:
    spec = importlib.util.spec_from_file_location(
        "gcr_predict", os.path.join(GCR_ROOT, "workflow", "predict_paths_and_answers.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.prediction


class GCRRunner:
    def __init__(self, model_path: str, k: int = 10, generation_mode: str = "group-beam", dtype: str = "bf16",
                 attn_implementation: str = "sdpa", max_new_tokens: int = 1024, hops: int = 2):
        from src.llms.graph_constrained_decoding_model import GraphConstrainedDecodingModel
        from src.qa_prompt_builder import PathGenerationWithAnswerPromptBuilder

        args = argparse.Namespace(
            model_path=model_path, maximun_token=4096, max_new_tokens=max_new_tokens, dtype=dtype, quant="none",
            attn_implementation=attn_implementation, generation_mode=generation_mode, k=k, chat_model=True,
            use_assistant_model=False, assistant_model_path=None,
        )
        self.model = GraphConstrainedDecodingModel(args)
        self.model.prepare_for_inference()
        self.hops = hops
        self.builder = PathGenerationWithAnswerPromptBuilder(self.model.tokenizer, "zero-shot", index_path_length=hops)
        self._predict = _load_prediction_fn()

    def run(self, samples: Iterable[dict], pred_file: str, path_mode: str = "multigraph", force: bool = False,
            progress: Optional[Callable] = None) -> None:
        enumerate_paths = PATH_MODES[path_mode]
        done = set()
        if os.path.exists(pred_file) and not force:
            with open(pred_file) as f:
                done = {json.loads(line)["id"] for line in f}
        samples = progress(samples) if progress else samples
        with open(pred_file, "w" if force else "a") as fout:
            for s in samples:
                if s["id"] in done:
                    continue
                s = dict(s)
                s["paths"] = enumerate_paths(s["graph"], s["q_entity"], self.hops)
                res = self._predict(s, [], self.builder, self.model)
                if res is None:  # GCR returns None when the trie is empty
                    res = {"id": s["id"], "question": s["question"], "prediction": [], "ground_truth": s["answer"],
                           "ground_truth_paths": []}
                res.pop("input", None)
                res["n_trie_paths"] = len(s["paths"])
                res["n_paths_generated"], res["n_paths_in_kg"] = path_validity(res["prediction"], s["paths"])
                if "candidate_graph" in s:  # B1+: separate paths that use an absent (candidate) edge
                    cand = {tuple(e) for e in s["candidate_graph"]}
                    via = [p for p in s["paths"] if any(tuple(e) in cand for e in p)]
                    res["n_candidate_edges"] = len(cand)
                    res["n_trie_paths_via_candidate"] = len(via)
                    res["n_paths_via_candidate"] = path_validity(res["prediction"], via)[1]
                for key in ("hard_answer", "ikg_level"):
                    if key in s:
                        res[key] = s[key]
                fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                fout.flush()


def faithful_rate(rows: List[dict]) -> dict:
    gen = sum(r.get("n_paths_generated", 0) for r in rows)
    ok = sum(r.get("n_paths_in_kg", 0) for r in rows)
    return {"paths_generated": gen, "paths_in_kg": ok, "faithful_path_rate": round(ok / gen, 4) if gen else None}
