"""Adapt GCR-Qwen2-0.5B to Wikidata relations (LoRA, GCR's own SFT format) for the Wikidata-domain QA E1.

GCR-Qwen2-0.5B was fine-tuned on Freebase relation names and scores Hits@Any 0.7% on Wikidata labels. This
applies GCR's KG-specialization recipe (workflow/finetune_kg_specialized_llm.py: zero-shot path prompt, target
"# Reasoning Path:\\n<PATH>path</PATH>\\n# Answer:\\nanswer", loss on the completion only) to Wikidata:

  data   1-hop facts *recorded in the 2019 graph* (data/processed/brink/wikidata/kg_incomplete.tsv) whose head
         is a training head of the natural-c split (md5 split 0). Dev heads (split 1) and test heads (split 2),
         and every fact missing in 2019, are excluded, so neither the questions nor the hard answers leak.
         Question: "What is the <property> of <entity>?" (the benchmark's template).
  model  LoRA r = 16 on all attention and MLP projections, 1 epoch, merged into the base and saved to
         data/models/gcr-qwen2-0.5b-wikidata (RTX 3060 Ti).

    CUDA_VISIBLE_DEVICES=GPU-c5cd579e-52d6-31a3-0091-04ae52932736 HF_HOME=$PWD/.hf \\
        python scripts/finetune_wikidata_gcr.py --n 8000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
GCR_ROOT = os.path.join(ROOT, "third_party", "graph-constrained-reasoning")
sys.path.insert(0, GCR_ROOT)

from observation_natural import head_split  # noqa: E402

BASE = "rmanluo/GCR-Qwen2-0.5B-Instruct"
ANS_TEMPLATE = "# Reasoning Path:\n{reasoning_path}\n# Answer:\n{answer}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000, help="training examples (one path each)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation (effective batch = batch * accum)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "models", "gcr-qwen2-0.5b-wikidata"))
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from src.qa_prompt_builder import PathGenerationWithAnswerPromptBuilder

    root = os.path.join(ROOT, "data", "processed", "brink", "wikidata")
    name = dict(line.rstrip("\n").split("\t") for line in open(os.path.join(root, "entities.tsv"), encoding="utf-8"))
    observed = [tuple(line.rstrip("\n").split("\t")) for line in open(os.path.join(root, "kg_incomplete.tsv"),
                                                                       encoding="utf-8") if line.strip()]
    true_heads = sorted({json.loads(l)["q_entity"][0] for sp in ("val", "test")
                         for l in open(os.path.join(root, f"questions_{sp}.jsonl"))})
    # The benchmark's head split: md5 over the sorted heads of facts true now (build_wikidata_qa.py).
    wd = os.path.join(ROOT, "data", "processed", "wikidata", "natural_t0_2019-08-26.tsv")
    heads_now = sorted({r.split("\t")[0] for r in open(wd).read().splitlines()[1:] if r.split("\t")[4] == "1"})
    split = dict(zip(heads_now, head_split(heads_now)))
    assert all(split.get(h, 0) != 0 for h in true_heads), "a benchmark head is in the training split"
    by_hr = defaultdict(list)
    for h, r, t in observed:
        if split.get(h, 0) == 0 and h in name and t in name:  # training heads only (heads absent from 'now' count as train)
            by_hr[(h, r)].append(t)
    pairs = list(by_hr.items())
    random.shuffle(pairs)

    tok = AutoTokenizer.from_pretrained(BASE)
    builder = PathGenerationWithAnswerPromptBuilder(tok, "zero-shot", index_path_length=2)
    texts = []
    for (h, r), tails in pairs:
        for t in tails[:2]:
            prompt = builder.format_input_with_template(f"What is the {r} of {name[h]}?", [name[h]])
            path = f"{builder.PATH_START_TOKEN}{name[h]} -> {r} -> {name[t]}{builder.PATH_END_TOKEN}"
            p_ids = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                            add_generation_prompt=True)
            full = tok.apply_chat_template([{"role": "user", "content": prompt},
                                            {"role": "assistant", "content": ANS_TEMPLATE.format(
                                                reasoning_path=path, answer=name[t])}], tokenize=True)
            texts.append((full, len(p_ids)))
        if len(texts) >= args.n:
            break
    print(f"{len(texts)} training paths from {len({h for (h, _), _ in pairs})} training heads", flush=True)

    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16).cuda()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                             "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    steps = args.epochs * (len(texts) + args.batch * args.accum - 1) // (args.batch * args.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 30) * max(0.0, 1 - s / steps))
    model.train()
    t0, step = time.time(), 0
    micro = 0
    for _ in range(args.epochs):
        random.shuffle(texts)
        for i in range(0, len(texts), args.batch):
            b = texts[i:i + args.batch]
            L = max(len(x) for x, _ in b)
            ids = torch.full((len(b), L), tok.pad_token_id or tok.eos_token_id)
            lab = torch.full((len(b), L), -100)
            att = torch.zeros((len(b), L), dtype=torch.long)
            for j, (x, p) in enumerate(b):
                ids[j, :len(x)] = torch.tensor(x)
                lab[j, p:len(x)] = torch.tensor(x[p:])  # completion only
                att[j, :len(x)] = 1
            out = model(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=lab.cuda())
            (out.loss / args.accum).backward()
            micro += 1
            if micro % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 50 == 0:
                print(f"step {step}/{steps} loss {out.loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    model = model.merge_and_unload()
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    GenerationConfig.from_pretrained(BASE).save_pretrained(args.out)
    json.dump({"base": BASE, "examples": len(texts), "epochs": args.epochs, "lr": args.lr, "lora_r": 16,
               "batch": args.batch * args.accum,
               "train_seconds": round(time.time() - t0, 1), "data": "1-hop facts recorded in 2019, training heads only"},
              open(os.path.join(args.out, "cc_gcd_finetune.json"), "w"), indent=2)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
