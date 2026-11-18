# CC-GCD: Coverage-Calibrated Graph-Constrained Decoding

Research code for CC-GCD: KG-Trie constrained LLM decoding in which a missing
edge receives a calibrated, finite penalty derived from a per-edge observation
model, instead of a hard mask. Research plan and novelty review: see the
project doc (Claude Docs, "الخطة البحثية المصحّحة لـ CC-GCD").

Status (Nov 2026): baseline B1 (GCR) set up and smoke-tested; BRINK prepared.
No CC-GCD decoder yet.

## Layout

| Path | Contents |
| --- | --- |
| `cc_gcd/data/brink.py` | BRINK loader: global KG, inverse-augmented L-hop subgraphs in GCR/RoG format |
| `cc_gcd/data/paths.py` | KG-Trie path enumeration with parallel relations kept (GCR's DiGraph drops them) |
| `data_prep/download_brink.sh` | Download BRINK Family / FB15k-237 (and optionally Wikidata5m) from Hugging Face |
| `data_prep/prepare_brink.py` | Raw BRINK to `data/processed/brink/<dataset>/` + `stats.json` |
| `analysis/kg_trie_coverage.py` | Hard KG-Trie ceiling on RoG-WebQSP / RoG-CWQ |
| `scripts/setup.sh` | Pinned third-party clones (GCR 9518e8e, BRINK b95506d) + Python env |
| `scripts/run_gcr_brink.py` | GCR step 1 on BRINK, official BRINK evaluator, faithful-path rate |
| `scripts/reproduce_gcr_webqsp_gpu.sh` | Full two-step GCR reproduction on WebQSP (one GPU, no OpenAI key) |
| `results/` | Small committed summaries (coverage, BRINK stats) |

## Quick start

```bash
bash scripts/setup.sh            # or: bash scripts/setup.sh gpu
. .venv/bin/activate && export HF_HOME=$PWD/.hf
bash data_prep/download_brink.sh
python data_prep/prepare_brink.py --dataset family
python data_prep/prepare_brink.py --dataset fb15k237
python scripts/run_gcr_brink.py --dataset family --version incomplete --split "test[:5]" \
  --model_path rmanluo/GCR-Qwen2-0.5B-Instruct --dtype fp32 --attn_implementation sdpa
```

## Findings so far

**RoG-WebQSP test (1,628 q), hard KG-Trie, L = 2** (`results/kg_trie_coverage/`):
the gold answer lies on a reachable path for 95.6% of questions (answer recall 94.1%).
GCR reports Hit 92.6, so on the complete graph a soft decoder has at most ~3 points of headroom.

**BRINK** (`results/brink_stats/`), counts match the BRINK README exactly:

| | Family | FB15k-237 |
| --- | --- | --- |
| Triples complete / incomplete / removed | 17,615 / 15,785 / 1,830 | 204,087 / 198,183 / 5,904 |
| Test questions | 198 | 540 |
| Hard triple found for every test question | yes | yes |
| Hard triple has the topic entity as tail | 94 (47%) | 30 (6%) |
| Hard answer within 1 / 2 / 3 hops, incomplete KG, with inverse edges | 35% / 100% / 100% | 80% / 98% / 100% |
| Same, directed edges only (GCR default) | 9% / 93% / 99% | 44% / 87% / 98% |
| 2-hop subgraph edges, median / max (incomplete) | 459 / 1,532 | 7,522 / 127,549 |
| Entity pairs with >1 relation in 2-hop subgraphs | 38% | 30% |

Implications:
- BRINK keeps alternative evidence by design, so a hard trie can still reach the hard
  answer for almost every question. BRINK tests whether the decoder picks the right
  path, not whether it can reach the answer. Recall gains from soft constraints need
  settings without guaranteed alternative paths (GoG IKGQA, path-targeted masking).
- Without inverse edges, GCR's directed path search cannot reach 13% (FB15k-237) of
  hard answers within 2 hops, so inverse edges (`~relation`) are added by default.
- GCR builds `nx.DiGraph`, which keeps one relation per entity pair. On FB15k-237 this
  hits half of the topic-to-hard-answer pairs, so `cc_gcd/data/paths.py` enumerates paths
  on a multigraph; `--path_mode gcr_digraph` reproduces the original behaviour.
- In the Textlabel version BRINK gives `hard_answer` as a MID (e.g. `/m/03rk0`), so a text
  answer ("India") never scores Hits@Hard with the official evaluator. `gold_<split>.json`
  uses the label as `hard_answer` and keeps the original in `hard_answer_raw`.
- 483 FB15k-237 entities share a label with another entity (e.g. "Georgia"); display names
  are made unique as `label (MID)`.
- GCR constrains decoding only between `<PATH>` and `</PATH>`. In a 3-question smoke test with
  the 0.5B model on Family, 25 of 30 generated paths were real KG paths; the others were free
  text. `run_gcr_brink.py` reports this faithful-path rate.

## Reference numbers (GCR paper, WebQSP test)

| Setting | Hit | F1 |
| --- | --- | --- |
| Full GCR, Llama-3.1-8B + ChatGPT (Table 1) | 92.6 | 73.2 |
| Full GCR, Llama-3.1-8B + Llama-3.1-8B general (Table 4) | 90.24 | 71.19 |
| Full GCR, Llama-3.1-8B + Qwen2-7B general (Table 4) | 86.32 | 67.59 |
| Full GCR, Qwen2-0.5B KG-specialized + ChatGPT (Table 4) | 87.48 | 60.03 |
| Step 1 only, Llama-3.1-8B (Table 3) | n/a | 57.0 (P 58.0, R 70.1) |
