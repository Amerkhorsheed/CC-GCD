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
                 attn_implementation: str = "sdpa", max_new_tokens: int = 1024, hops: int = 2, quant: str = "none"):
        from src.llms.graph_constrained_decoding_model import GraphConstrainedDecodingModel
        from src.qa_prompt_builder import PathGenerationWithAnswerPromptBuilder

        args = argparse.Namespace(
            model_path=model_path, maximun_token=4096, max_new_tokens=max_new_tokens, dtype=dtype, quant=quant,
            attn_implementation=attn_implementation, generation_mode=generation_mode, k=k, chat_model=True,
            use_assistant_model=False, assistant_model_path=None,
        )
        self.model = GraphConstrainedDecodingModel(args)
        gpu_mem = os.environ.get("CCGCD_MAX_GPU_MEM")  # e.g. "7GiB": bf16 8B on a 12 GB card, rest kept in RAM
        if gpu_mem:
            from transformers import AutoModelForCausalLM
            orig = AutoModelForCausalLM.from_pretrained.__func__

            def capped(cls, *a, **kw):
                kw.setdefault("max_memory", {0: gpu_mem, "cpu": os.environ.get("CCGCD_MAX_CPU_MEM", "24GiB")})
                return orig(cls, *a, **kw)
            AutoModelForCausalLM.from_pretrained = classmethod(capped)
            try:
                self.model.prepare_for_inference()
            finally:
                AutoModelForCausalLM.from_pretrained = classmethod(orig)
            print("[GCRRunner] device map:", sorted(set(getattr(self.model.model, "hf_device_map", {}).values()),
                                                   key=str), flush=True)
        else:
            self.model.prepare_for_inference()
        self.hops = hops
        self.builder = PathGenerationWithAnswerPromptBuilder(self.model.tokenizer, "zero-shot", index_path_length=hops)
        self._predict = _load_prediction_fn()

    def _predict_soft(self, s: dict, scores: List[float]) -> Optional[dict]:
        """GCR step 1 with the soft trie (cc_gcd.decoding.soft_trie): same prompt, same allowed tokens as
        GCR's KG-Trie, plus the lookahead prefix bias from the path scores."""
        from transformers import LogitsProcessorList
        from src.graph_constrained_decoding import GraphConstrainedDecoding
        from cc_gcd.decoding.soft_trie import PrefixBiasProcessor, SoftTrie

        if not s["paths"]:
            return None
        input_query, ground_paths, _ = self.builder.process_input(s, return_tire=False)
        tok = self.model.tokenizer
        start, end = self.builder.PATH_START_TOKEN, self.builder.PATH_END_TOKEN
        seqs = tok([f"{start}{path_string(p)}{end}" for p in s["paths"]], padding=False,
                   add_special_tokens=False).input_ids
        trie = SoftTrie(seqs, scores)
        sid, eid = tok.convert_tokens_to_ids(start), tok.convert_tokens_to_ids(end)
        prompt = self.model.prepare_model_prompt(input_query)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
        gcd = GraphConstrainedDecoding(tok, trie, sid, eid, False)
        res = self.model.model.generate(
            input_ids=enc.input_ids.to(self.model.model.device), attention_mask=enc.attention_mask.to(self.model.model.device),
            generation_config=self.model.generation_cfg, prefix_allowed_tokens_fn=gcd.allowed_tokens_fn,
            logits_processor=LogitsProcessorList([PrefixBiasProcessor(trie, sid, eid)]),
            return_dict_in_generate=True, pad_token_id=tok.eos_token_id)
        n = enc.input_ids.shape[1]
        pred = [tok.decode(r[n:], skip_special_tokens=True) for r in res.sequences]
        return {"id": s["id"], "question": s["question"], "prediction": pred if len(pred) > 1 else pred[0],
                "ground_truth": s["answer"], "ground_truth_paths": ground_paths}

    def _predict_free(self, s: dict) -> Optional[dict]:
        """B0: the same prompt and beams with no KG-Trie (no constrained decoding), for the parametric-memory
        control (plan 7.3)."""
        input_query, _, _ = self.builder.process_input({**s, "graph": []}, return_tire=False)
        tok = self.model.tokenizer
        prompt = self.model.prepare_model_prompt(input_query)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
        res = self.model.model.generate(
            input_ids=enc.input_ids.to(self.model.model.device), attention_mask=enc.attention_mask.to(self.model.model.device),
            generation_config=self.model.generation_cfg, return_dict_in_generate=True, pad_token_id=tok.eos_token_id)
        n = enc.input_ids.shape[1]
        pred = [tok.decode(r[n:], skip_special_tokens=True) for r in res.sequences]
        return {"id": s["id"], "question": s["question"], "prediction": pred if len(pred) > 1 else pred[0],
                "ground_truth": s["answer"], "ground_truth_paths": []}

    def run(self, samples: Iterable[dict], pred_file: str, path_mode: str = "multigraph", force: bool = False,
            progress: Optional[Callable] = None, lam: Optional[float] = None) -> int:
        """``lam`` set: soft decoding with path score lam * sum of sample["edge_scores"] (absent edges;
        observed edges score 0). ``lam`` None: GCR's hard KG-Trie.

        A generation that fails on a non-empty trie (GCR's generate_sentence swallows exceptions such as
        CUDA out-of-memory and returns None) is retried once after emptying the CUDA cache and, if it fails
        again, is *not* written, so a later call resumes it. Returns the number of such failures."""
        import time
        import torch
        enumerate_paths = PATH_MODES[path_mode]
        done = set()
        if os.path.exists(pred_file) and not force:
            with open(pred_file) as f:
                done = {json.loads(line)["id"] for line in f}
        samples = progress(samples) if progress else samples
        failed = 0
        with open(pred_file, "w" if force else "a") as fout:
            for s in samples:
                if s["id"] in done:
                    continue
                s = dict(s)
                s["paths"] = [] if s.get("no_kg") else enumerate_paths(s["graph"], s["q_entity"], self.hops)
                def attempt():
                    if s.get("no_kg"):
                        try:
                            return self._predict_free(s)
                        except torch.cuda.OutOfMemoryError:
                            return None
                    if lam is None:
                        return self._predict(s, [], self.builder, self.model)
                    es = s.get("edge_scores", {})
                    scores = [lam * sum(es.get(tuple(e), 0.0) for e in p) for p in s["paths"]]
                    try:
                        return self._predict_soft(s, scores)
                    except torch.cuda.OutOfMemoryError:
                        return None

                res = attempt()
                if res is None and (s["paths"] or s.get("no_kg")):
                    torch.cuda.empty_cache()
                    time.sleep(5)
                    res = attempt()
                    if res is None:
                        failed += 1
                        torch.cuda.empty_cache()
                        continue  # not written: resumed by the next call
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
        if failed:
            print(f"[GCRRunner] {failed} generations failed and were not written; rerun to resume", flush=True)
        return failed


def faithful_rate(rows: List[dict]) -> dict:
    gen = sum(r.get("n_paths_generated", 0) for r in rows)
    ok = sum(r.get("n_paths_in_kg", 0) for r in rows)
    return {"paths_generated": gen, "paths_in_kg": ok, "faithful_path_rate": round(ok / gen, 4) if gen else None}
