# CC-GCD: Coverage-Calibrated Graph-Constrained Decoding

Research code for CC-GCD: KG-Trie constrained LLM decoding in which a missing
edge receives a calibrated, finite penalty derived from a per-edge observation
model, instead of a hard mask. Research plan and novelty review: see the
project doc (Claude Docs, "الخطة البحثية المصحّحة لـ CC-GCD").

Status (Nov 2026): baseline B1 (GCR) set up and smoke-tested; BRINK prepared; KGE candidate
generator (plan section 5.1) built and measured on BRINK, wired into GCR as B1+. No CC-GCD decoder yet.

## Layout

| Path | Contents |
| --- | --- |
| `cc_gcd/data/brink.py` | BRINK loader: global KG, inverse-augmented L-hop subgraphs in GCR/RoG format |
| `cc_gcd/data/paths.py` | KG-Trie path enumeration with parallel relations kept (GCR's DiGraph drops them) |
| `data_prep/download_brink.sh` | Download BRINK Family / FB15k-237 (and optionally Wikidata5m) from Hugging Face |
| `data_prep/prepare_brink.py` | Raw BRINK to `data/processed/brink/<dataset>/` + `stats.json` |
| `cc_gcd/data/ikgqa.py` | IKGQA loader: GoG's per-question deletions applied to RoG subgraphs |
| `data_prep/prepare_ikgqa.py` | GoG IKG-0/20/40/60/80% for WebQSP / CWQ + hard-trie ceilings |
| `cc_gcd/baselines/gcr.py` | B1 wrapper around the official GCR code (explicit KG-Trie paths, faithful-path rate) |
| `scripts/run_gcr_ikgqa.py` | GCR step 1 on IKGQA with GCR's own Hit/F1 |
| `analysis/kg_trie_coverage.py` | Hard KG-Trie ceiling on RoG-WebQSP / RoG-CWQ |
| `scripts/setup.sh` | Pinned third-party clones (GCR 9518e8e, BRINK b95506d, GoG c0b2c36) + Python env |
| `scripts/run_gcr_brink.py` | GCR step 1 on BRINK, official BRINK evaluator, faithful-path rate |
| `cc_gcd/kge/complex_kge.py` | ComplEx-N3 (reciprocal, 1-vs-all) trained on the observed graph only; exported scorer |
| `cc_gcd/graph/candidate_generator.py` | Absent-edge candidates: top-R relations x top-k KGE tails per entity, signature filter |
| `scripts/train_kge.py` | Train the KGE on BRINK's incomplete graph (RTX 3060 Ti) |
| `scripts/build_candidates.py` | Candidate tables + oracle recall, hard-fact recovery and trie growth on val/test |
| `scripts/reproduce_gcr_webqsp_gpu.sh` | Full two-step GCR reproduction on WebQSP (one GPU, no OpenAI key) |
| `results/` | Small committed summaries (coverage, BRINK stats) |

## Quick start

```bash
bash scripts/setup.sh            # or: bash scripts/setup.sh gpu
. .venv/bin/activate && export HF_HOME=$PWD/.hf
bash data_prep/download_brink.sh
python data_prep/prepare_brink.py --dataset family
python data_prep/prepare_brink.py --dataset fb15k237
python data_prep/prepare_ikgqa.py --dataset webqsp     # and --dataset cwq (downloads ~370 MB)
python scripts/run_gcr_brink.py --dataset family --version incomplete --split "test[:5]" \
  --model_path rmanluo/GCR-Qwen2-0.5B-Instruct --dtype fp32 --attn_implementation sdpa

# KGE candidate generator (GPU work on the RTX 3060 Ti only)
export CUDA_VISIBLE_DEVICES=GPU-c5cd579e-52d6-31a3-0091-04ae52932736
python scripts/train_kge.py --dataset family --reg_weight 5e-3 --epochs 200   # ~20 s
python scripts/train_kge.py --dataset fb15k237 --reg_weight 1e-2              # ~9 min
python scripts/build_candidates.py --dataset family                          # ~1 min
python scripts/build_candidates.py --dataset fb15k237                        # ~15 min (mostly evaluation)
# B1+: hard trie over observed + candidate edges, no penalty
python scripts/run_gcr_brink.py --dataset fb15k237 --split "test[:5]" --candidates hybrid+sig --cand_R 10 --cand_k 5
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

**IKGQA** (GoG, EMNLP 2024; `results/ikgqa_stats/`). GoG binds each question's gold SPARQL
query, drops each entity pair on the bound answer paths with probability p, and deletes every
relation between the pair in both directions, per question. `data_prep/prepare_ikgqa.py`
applies these deletions to the question's RoG subgraph (what GCR reads); only 43% of GoG's
crucial triples appear in WebQSP's RoG subgraphs (71% for CWQ); the rest were never visible to GCR.
Hard KG-Trie ceiling = share of questions whose answer is on a <= 2-hop path from the topic entities:

| Level | WebQSP: questions with deletions | WebQSP: ceiling, directed (GCR) | WebQSP: with inverse edges | CWQ: questions with deletions | CWQ: ceiling, directed (GCR) | CWQ: with inverse edges |
| --- | --- | --- | --- | --- | --- | --- |
| Full (IKG-0) | 0 | 97.8% | 97.8% | 0 | 78.7% | 78.7% |
| IKG-20% | 481 | 92.4% | 95.0% | 534 | 67.8% | 72.3% |
| IKG-40% | 650 | 88.4% | 92.5% | 760 | 60.4% | 66.6% |
| IKG-60% | 823 | 80.5% | 87.9% | 901 | 50.9% | 61.4% |
| IKG-80% | 899 | 74.5% | 84.5% | 956 | 47.6% | 59.8% |

Each dataset: GoG's first 1,000 test questions, L = 2. Crucial triples present in RoG subgraphs: WebQSP 43%, CWQ 71%.
CWQ needs up to 4 hops, so even the full graph caps a 2-hop trie at 78.7%.

This is the headroom a soft decoder can target: at IKG-80% a hard trie cannot reach the
answer for a quarter of WebQSP questions and half of CWQ questions.

**GCR step 1, Qwen2-0.5B, RoG-WebQSP test[:200], CPU** (`results/gcr_webqsp/`): Hit 89.5, F1 37.2
(P 32.3, R 70.3), answer recall within reach of the hard trie on this subset 92.9% (Hit ceiling 95.0%).
1,688 of 2,000 generated paths (84.4%) are exact KG-Trie paths; nearly all others name entities
that are not in the question's subgraph, i.e. beams that left the constraint. This is a smoke-scale
check of the pipeline, not a reproduction: the paper reports no step-1-only number for 0.5B
(8B step 1: F1 57.0, P 58.0, R 70.1). The full 8B run needs `scripts/reproduce_gcr_webqsp_gpu.sh`.

**KGE candidate generator on BRINK** (`results/kge/`, `results/candidates/`). ComplEx-N3, dim 256,
trained on the incomplete graph with inverse relations; early stopping on a random 2% holdout of
observed triples (never BRINK's removed triples). Holdout filtered MRR: Family 0.688 (Hits@10 0.970),
FB15k-237 0.412 (Hits@10 0.619). For each entity the generator proposes the top-k absent tails of its
top-R relations; candidates enter the trie at hop 1 (from the topic entities). Mode and (R, k) were
chosen on val (best recovery with R x k <= 50); numbers below are test, L = 2, inverse edges:

| Test | Family | FB15k-237 |
| --- | --- | --- |
| Selected on val | relation rank `kge`, R = 10, k = 5 | `hybrid` (co-occurrence x KGE prob.), R = 10, k = 5 |
| Removed hard fact itself among the candidates | 61.6% | 54.4% |
| Hard answer reachable: observed only / with candidates | 100% / 100% | 97.96% / 99.44% |
| Reachable only through candidate edges | 0% | 1.48% |
| Any gold answer reachable: observed / with candidates | 100% / 100% | 98.7% / 99.8% |
| Candidate edges added per question | 50 | 47 |
| Trie walks per question (paths <= 2): observed -> with candidates | 682 -> 1,942 | 21,350 -> 37,147 |
| Candidates that are BRINK-removed facts (lower bound on precision) | 6.6% | 5.5% |

Diagnostics (test, `diagnostics` in the JSON):
- Relation choice is the bottleneck, not tail ranking. Given the right relation, the removed tail is in the
  KGE's top 10 for 97% (Family) and 87% (FB15k-237) of questions; the right relation is in the entity's
  top 5 for only 34-44% (Family) and 42% (FB15k-237, `hybrid`). Recovery still rises at R = 10, so R_max = 10
  may be limiting; the plan's default R = 5, k = 10 is dominated by R = 10, k = 5 on both graphs.
- Ranking relations by the softmax probability of the best absent tail (`kge_prob`) is the worst ranker
  (FB15k-237 recovery 0.385 vs 0.544 at R = 10, k = 5): BRINK removes facts of relations the entity
  already has, and the observed tails absorb the softmax mass. Raw scores (`kge`) and co-occurrence
  (`cooc`) do better; their combination (`hybrid`) is best on FB15k-237, `cooc` on Family test.
- The signature (domain/range) filter never removed a hard fact and helps when relations are ranked by
  KGE probability (0.385 vs 0.309 without it).
- Candidates at hop 2 as well (from every hop-1 node) add ~10,000 edges per FB15k-237 question
  (trie walks 44,000-70,000) for +0 to 0.4 points of reachability: hop 1 only is the default.
- BRINK keeps alternative paths, so reachability gains are small; BRINK measures whether the decoder
  picks the recovered fact. Reachability headroom is in IKGQA (next: a KGE over Freebase/RoG subgraphs).
- B1+ smoke test (GCR-Qwen2-0.5B, 3 Family questions, not a result): without a penalty the candidates
  dominate the trie (e.g. 971 of 1,202 trie paths use an absent edge) and 20 of 24 generated KG paths
  used one, which is why the absence penalty, not the generator alone, decides precision.

## Reference numbers (GCR paper, WebQSP test)

| Setting | Hit | F1 |
| --- | --- | --- |
| Full GCR, Llama-3.1-8B + ChatGPT (Table 1) | 92.6 | 73.2 |
| Full GCR, Llama-3.1-8B + Llama-3.1-8B general (Table 4) | 90.24 | 71.19 |
| Full GCR, Llama-3.1-8B + Qwen2-7B general (Table 4) | 86.32 | 67.59 |
| Full GCR, Qwen2-0.5B KG-specialized + ChatGPT (Table 4) | 87.48 | 60.03 |
| Step 1 only, Llama-3.1-8B (Table 3) | n/a | 57.0 (P 58.0, R 70.1) |
