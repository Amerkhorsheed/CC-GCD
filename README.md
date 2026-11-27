# CC-GCD: Coverage-Calibrated Graph-Constrained Decoding

Research code for CC-GCD: KG-Trie constrained LLM decoding in which a missing
edge receives a calibrated, finite penalty derived from a per-edge observation
model, instead of a hard mask. Research plan and novelty review: see the
project doc (Claude Docs, "الخطة البحثية المصحّحة لـ CC-GCD").

Status (Nov 2026): baseline B1 (GCR) set up and smoke-tested; BRINK and IKGQA-WebQSP prepared; KGE
candidate generator (plan section 5.1) built and measured on BRINK and IKGQA-WebQSP, wired into GCR
(BRINK) as B1+; observation model c(e) (plan section 6) tested on synthetic mechanisms and on natural missingness
from two Wikidata snapshots; calibrated KGE prior pi(e); CC-GCD decoder (soft KG-Trie with lookahead
prefix bias) and the E1 comparison with GCR-Qwen2-0.5B on the RTX 3060 Ti: soft decoding with priced
absent edges beats GCR's hard trie by ~1 point on BRINK-FB15k-237 and by 4-5 points (Hits@Hard) / 7-8 points
(F1) on a Wikidata QA benchmark with natural missingness, but the calibrated observation model c(e) never beats
a tuned KGE-prior penalty (B4) or a fixed penalty (B2/B3) on accuracy; with entity names anonymized (E5) the
KGE-prior penalty gains +16 Hits@Hard over the hard trie and beats CC-GCD (plan section 10: Reframe).

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
| `cc_gcd/kge/complex_kge.py` | ComplEx-N3 (reciprocal; 1-vs-all or sampled softmax) trained on the observed graph only; exported scorer |
| `cc_gcd/graph/candidate_generator.py` | Absent-edge candidates: top-R relations x top-k KGE tails per entity, signature filter |
| `scripts/train_kge.py` | Train the KGE on BRINK's incomplete graph or the IKGQA union graph (RTX 3060 Ti) |
| `scripts/build_candidates.py` | BRINK: candidate tables + oracle recall, hard-fact recovery and trie growth on val/test |
| `scripts/build_candidates_ikgqa.py` | IKGQA: global vs subgraph-local candidates per IKG level, dev/eval split |
| `cc_gcd/calibration/missingness.py` | Masking mechanisms A (MCAR), B (relation-selective), C (degree-biased), D (path-targeted, train questions) with exact retention = oracle c |
| `cc_gcd/calibration/observation_model.py` | c(e): cross-fitted structural features, SCAR / per-relation / logistic / GBM + isotonic + clipping; evidence score s(e) |
| `scripts/observation_lomo.py` | Mechanism-transfer matrix, leave-one-mechanism-out, BRINK's own missingness as held-out test (CPU, ~2 min) |
| `data_prep/wikidata_dump_stream.py` | Stream a Wikidata JSON dump (resumable Range download, pigz), keep items with our Freebase IDs |
| `data_prep/wikidata_snapshots.py` | Natural-missingness labels from the 2019-08-26 and current dumps (`build-dump`); API route kept for unthrottled networks |
| `scripts/observation_natural.py` | c(e) on natural labels (head-disjoint), synthetic -> natural, natural -> BRINK |
| `scripts/fit_natural_c.py` | Save the natural structural c(e) model used by CC-GCD with per-edge c |
| `cc_gcd/calibration/kge_prior.py` | Calibrated KGE prior pi(e) from the KGE's own MCAR holdout (logistic + isotonic; rank-free variant) |
| `scripts/calibrate_kge_prior.py`, `scripts/calibrate_kge_prior_ikgqa.py` | Fit pi on BRINK / IKGQA-WebQSP and check it on BRINK's removed facts |
| `cc_gcd/decoding/soft_trie.py` | Soft KG-Trie with per-node lookahead values; prefix bias as a LogitsProcessor |
| `data_prep/build_wikidata_qa.py` | Wikidata-domain QA (2019 graph, answers missing in 2019) in BRINK format |
| `data_prep/anonymize_wikidata_qa.py` | Anonymized copy of the Wikidata QA benchmark (entity names -> "entity N") for E5 |
| `scripts/run_8b_on_3060.sh` | 8B GCR (4-bit) on the RTX 3060 for the owner-permitted case, with a Qwen watchdog |
| `scripts/finetune_wikidata_gcr.py` | LoRA-adapt GCR-Qwen2-0.5B to Wikidata relations (GCR's SFT format, training heads only) |
| `scripts/e1_bootstrap.py` | Per-question metrics and paired bootstrap for E1 |
| `scripts/run_e1_brink.py`, `scripts/run_e1_ikgqa.py` | E1: B1, B1+, B2, B4, CC-GCD (constant c = B5, per-edge c) with shared model, beams and candidates |
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
# IKGQA-WebQSP: one leak-free KGE for all levels (sampled softmax: 599k entities do not fit 1-vs-all on 8 GB)
export HF_HOME=$PWD/.hf
python scripts/train_kge.py --dataset ikgqa-webqsp --dim 200 --batch_size 1024 --neg_sample 16384 --amp \
  --reg_weight 1e-2 --epochs 40 --eval_every 2                                 # ~30 min
python scripts/build_candidates_ikgqa.py --dataset webqsp                     # ~20 min
# Observation model c(e) (CPU)
python scripts/observation_lomo.py --dataset fb15k237                         # ~2 min
python scripts/observation_lomo.py --dataset family
# Natural missingness (two dump streams, ~3 h on this link), then c(e) on natural labels
python data_prep/wikidata_dump_stream.py download https://archive.org/download/wikidata-20190826-all.json.gz/wikidata-20190826-all.json.gz \
  | pigz -dc | python data_prep/wikidata_dump_stream.py filter --out data/raw/wikidata/dump_20190826.jsonl.gz
python data_prep/wikidata_dump_stream.py download https://ftpmirror.your.org/pub/wikimedia/dumps/wikidatawiki/entities/latest-all.json.gz \
  | pigz -dc | python data_prep/wikidata_dump_stream.py filter --out data/raw/wikidata/dump_20261117.jsonl.gz
python data_prep/wikidata_snapshots.py build-dump --t0 2019-08-26 --old 20190826 --new 20261117
python scripts/observation_natural.py --t0 2019-08-26 && python scripts/fit_natural_c.py
# Calibrated KGE prior pi(e)
python scripts/calibrate_kge_prior.py --dataset fb15k237
python scripts/calibrate_kge_prior_ikgqa.py --device cpu
# E1 (dev tuning on val[:150], then test)
python scripts/run_e1_brink.py --dataset fb15k237 --split "val[:150]" --methods B1 B1+ B2:3 B4:1 CC:0.9:1 CCe:wikidata_gbm_struct:1
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
  top 5 for only 32-34% (Family) and 42-44% (FB15k-237). Recovery still rises at R = 10, so R_max = 10
  may be limiting; the plan's default R = 5, k = 10 is dominated by R = 10, k = 5 on both graphs.
- Ranking relations by the softmax probability of the best absent tail (`kge_prob`) is the worst ranker
  (FB15k-237 recovery 0.385 vs 0.544 at R = 10, k = 5): BRINK removes facts of relations the entity
  already has, and the observed tails absorb the softmax mass. Raw scores (`kge`), co-occurrence
  (`cooc`) and their combination (`hybrid`) do better; val picks `kge` (Family) and `hybrid` (FB15k-237).
- `cooc` alone is tie-bound: every relation an entity already has scores 1, so for 20% (Family) and 40%
  (FB15k-237) of entities the 10th-best relation ties the best. Ties now go to the more frequent relation
  (deterministic, KGE-free); an earlier version broke them arbitrarily, which had moved `cooc` test recovery
  by up to 10 points. The val-selected configurations above did not change.
- The signature (domain/range) filter never removed a hard fact and helps when relations are ranked by
  KGE probability (0.385 vs 0.309 without it).
- Candidates at hop 2 as well (from every hop-1 node) add ~10,000 edges per FB15k-237 question
  (trie walks 44,000-70,000) for +0 to 0.4 points of reachability: hop 1 only is the default.
- BRINK keeps alternative paths, so reachability gains are small; BRINK measures whether the decoder
  picks the recovered fact. Reachability headroom is in IKGQA (next: a KGE over Freebase/RoG subgraphs).
- B1+ smoke test (GCR-Qwen2-0.5B, 3 Family questions, not a result): without a penalty the candidates
  dominate the trie (e.g. 971 of 1,202 trie paths use an absent edge) and 20 of 24 generated KG paths
  used one, which is why the absence penalty, not the generator alone, decides precision.

**KGE candidate generator on IKGQA-WebQSP** (`results/kge/ikgqa-webqsp.json`,
`results/candidates/ikgqa-webqsp.json`). IKGQA has no global KG and deletes per question, and 62% of a
level's deleted triples also occur in another question's RoG subgraph, so the KGE is trained on the union
of the 1,000 subgraphs minus every triple deleted at any level (1,740,574 triples, 598k entities, 10,352
relations with inverse; one model serves all levels). ComplEx-N3, dim 200, sampled softmax (16,384 shared
negatives) because 1-vs-all over 598k entities does not fit beside the desktop on the 8 GB card; holdout
filtered MRR 0.617 (Hits@10 0.678), 29 min. KGE rankings score the 50 relations with the highest
co-occurrence per topic entity. IKGQA has only test questions, so every fifth question (200) is a dev set
for selection and numbers are on the other 800. *Global* ranks tails over all entities; *local* ranks them
only over entities of the question's observed subgraph at that level (the evidence GCR already reads).
Candidates at hop 1; L = 2 with inverse edges. Observed-only reachability matches the committed
`ikg_inverse_hit_within_2` ceilings.

| Eval, 800 q | Answer reachable: observed -> with candidates, mean IKG-20..80 | same, IKG-80 | ... through restored deleted facts only, mean | Deleted topic fact recovered (q with one) | Candidate edges / q |
| --- | --- | --- | --- | --- | --- |
| Global `hybrid`, R = 10, k = 5 (BRINK choice, untuned) | 89.5% -> 90.3% | 83.9% -> 85.6% | 89.8% | 7.2% | 51 |
| Local `hybrid`, R = 10, k = 5 | 89.5% -> 92.5% | 83.9% -> 89.0% | 90.1% | 12.7% | 51 |
| Local `cooc`, R = 5, k = 10 (selected on dev, reachability) | 89.5% -> 93.7% | 83.9% -> 90.9% | 89.5% | 2.2% | 50 |
| Local `hybrid`, R = 10, k = 20 (4x budget) | 89.5% -> 94.6% | 83.9% -> 92.6% | 90.4% | 17.8% | 202 |

At IKG-0 no candidate changes reachability (97.9%). Local `hybrid`, R = 10, k = 5 grows trie walks per
question from 13,695 to 15,364 at IKG-80.

- Nearly all of the reachability gain is through hypothetical edges, not restored evidence: restoring the
  question's own deleted facts adds at most +0.9 points, and the configuration selected by reachability
  (`cooc`) restores almost none. Only 0.1-0.6% of candidates are the question's deleted facts. So on IKGQA a
  generator raises the hard-trie ceiling mainly by opening shortcuts, which is exactly where the absence
  penalty must decide; reachability alone is the wrong selection criterion for the generator (the JSON also
  reports the restored-evidence selection: local `kge`, R = 10, k = 1).
- Local beats global at the same budget (recovery 12.7% vs 7.2%): ranking 598k entities is much harder than
  ranking a subgraph's few thousand.
- Why recovery is low: 73% of deleted triples touch a topic entity, but 40% of those point to CVT nodes. Of
  2,751 CVT-tailed deleted facts, 28% have a CVT that occurs only in deleted triples (outside the KGE
  vocabulary), and the rest have median degree 2, so no embedding can say which CVT belongs to the topic
  (deleted tail in the global top 10 given the relation: 3.4%; named tails 15.8%, top 100: 42.5%). GoG also
  drops pairs on a path independently, so restoring the topic-CVT edge often still leaves the CVT-answer
  edge missing, and hop-1 candidates cannot restore hop-2 deletions.

**Observation model c(e) on BRINK** (`results/observation/`). Reference graph = BRINK's observed graph;
four controlled mechanisms (A MCAR, B relation-selective, C degree-biased, D path-targeted on *train*
questions; mean retention 0.69-0.75) give each triple an exact retention probability, which is also the
oracle c (B6). Label = the triple survived masking. Features are computed on the masked graph and are
cross-fitted (10 folds; fold k sees the graph without all of fold k): excluding only the triple itself
still leaks, because within one draw every kept r-triple sees K - 1 other kept r-triples and every dropped
one K, and a GBM with the relation id then reached AUC 0.995 under MCAR (oracle 0.5); cross-fitted, 0.495.
Isotonic on a separate split, clipping to [0.02, 0.98]. Three masking seeds; triples split 60/20/20.
BRINK's own rule-based missingness (complete vs incomplete) is never used to train or calibrate.

| FB15k-237 (Family) | Same mechanism: ECE / AUC | Other mechanism: ECE / AUC | Leave-one-mechanism-out: ECE / AUC | LOMO Brier, level matched | BRINK missingness: ECE / AUC |
| --- | --- | --- | --- | --- | --- |
| Constant c (SCAR, B5) | 0.007 / 0.50 | **0.033** / 0.50 (0.125) | **0.028** / 0.50 (0.102) | 0.2053 (**0.2123**) | 0.262 / 0.50 |
| Per relation | 0.007 / 0.66 | 0.133 / 0.49 (0.181) | 0.101 / 0.48 (0.129) | 0.2148 (0.2203) | 0.263 / **0.65** |
| Logistic regression | 0.008 / 0.63 | 0.101 / 0.50 (0.215) | 0.062 / 0.51 (0.106) | 0.2099 (0.2138) | 0.242 / 0.57 |
| GBM + relation id | 0.008 / **0.69** | 0.145 / 0.49 (0.206) | 0.060 / **0.61** (0.116) | **0.2001** (0.2187) | **0.098** / 0.52 |

Family values in parentheses where shown. Oracle AUC (mean of the four mechanisms): 0.71 (FB15k-237),
0.67 (Family). "Level matched" shifts each prediction by one logit constant to the target's true mean and
compares with the constant at that mean (0.2053 / 0.2123): it separates the per-edge shape of c from its
overall level, which no estimator can identify from an incomplete graph (plan 6.1); it uses the target's
true rate, so it is a diagnostic, not an estimator. On BRINK's missingness (retention 97.1% FB15k-237,
89.6% Family) the level-matched constant has Brier 0.02809 (0.0931) and no model beats it by more than 0.5%.

- Within a mechanism the features work: the GBM nearly reaches the oracle (B 0.783 vs 0.779, C 0.719 vs
  0.726, D 0.754 vs 0.823 AUC).
- Across mechanisms they do not. Trained on one mechanism and tested on another, every learned c is
  worse calibrated than the constant and ranks at chance or below (degree-biased -> path-targeted AUC 0.44,
  the reverse 0.41: C drops low-degree edges, D edges near question entities, often hubs). Trained on the
  three other mechanisms, the GBM keeps some ranking (AUC 0.61 on FB15k-237, 0.71 on B), and once the level
  is given it beats the constant by 2.5% Brier on FB15k-237, but not on Family.
- By the plan's stop rule (section 10: c's ECE on a held-out mechanism clearly worse than constant c, B5),
  c learned from these synthetic mechanisms fails clearly on FB15k-237 (off-diagonal and LOMO ECE 2-4x the
  constant's) and is no better than the constant on Family (LOMO ECE 0.106-0.129 vs 0.102). The level of c
  is the part that does not transfer; the per-edge shape transfers weakly at best.
- On BRINK's real missingness, models trained on the path-targeted mechanism rank best (AUC 0.66 on
  FB15k-237, 0.55 on Family), consistent with BRINK deleting rule heads around question paths; the GBM
  trained on all four mechanisms is closest in level (ECE 0.098) but ranks near chance.
- Implication: synthetic masking can test an estimator but not supply a transferable c. The plan's primary
  calibration source, natural missingness from temporal Wikidata snapshots (section 6.2), is the next
  required step before c can carry the decoder; until then B5 (constant c, i.e. a tuned level) is the
  honest default, and H2 rests on the level being set per graph, not learned.

**Natural missingness: Wikidata 2019-08-26 vs 2026-11-17** (`results/observation/wikidata-natural-*`).
The Wikidata API throttles this connection (429 with 17-45 s waits; the SPARQL endpoint asked for
1,000 s), so both JSON dumps (55 GB from archive.org, 157 GB current from a Wikimedia mirror) were
streamed and filtered on the fly to items carrying one of BRINK-FB15k-237's Freebase IDs (P646); only
2.6 MB + 1.5 MB of compact claims are stored. Universe: 13,631 entities mapped one-to-one in both
snapshots. Label: an item-valued fact between universe entities that is true now was recorded at T0 or
not. Dropped as "not yet true at T0": statements whose start time or point in time is after T0 (911), and
the facts of 229 properties with fewer than 20 uses in the 2019 universe graph (2,692; schema change,
not missingness). Facts present in 2019 but gone now stay in the 2019 graph for features (9,507) but get
no label. Result: 164,082 labelled facts, 78.95% recorded at T0, 134 properties.

| Head-disjoint test (32k facts) | ECE | Brier | AUC |
| --- | --- | --- | --- |
| Constant c (SCAR) | 0.031 | 0.166 | 0.50 |
| Per relation | 0.018 | 0.130 | 0.785 |
| Logistic regression | **0.011** | 0.138 | 0.766 |
| GBM + relation id | 0.013 | **0.099** | **0.887** |
| GBM, structural features only | 0.012 | 0.100 | 0.885 |

- On natural missingness per-edge c works: the GBM cuts Brier by 40% against constant c and 23% against
  per-relation c, and nearly all of it survives without the relation identity. Natural missingness is
  clustered: an entity with no recorded facts of a relation at T0 is missing more of them (the
  completeness signal of Galarraga et al., now on natural rather than synthetic deletions).
- c trained on synthetic masking of the same 2019 graph (A, B, C; D needs questions) does not transfer to
  the natural labels: AUC 0.47-0.59, ECE 0.06-0.21. The plan's preference for natural over synthetic
  calibration (section 6.2) is borne out.
- Natural Wikidata c transferred to BRINK-FB15k-237's rule-based missingness ranks weakly (AUC 0.60-0.62,
  structural features) and its level is off (Brier after level matching 0.0296-0.0307 vs 0.0281 for the
  constant). Natural c is a property of the graph and its editing history, not a universal constant.

**Calibrated KGE prior pi(e)** (`results/kge_prior/`). The KGE was trained on 98% of the observed graph,
so its 2% MCAR holdout is true and unseen. A generator that does not observe the holdout proposes
candidates for the holdout heads; label = holdout fact. Logistic regression on log P(t | h, r), raw
score, log tail rank and log relation rank (rank-free variant for IKGQA's subgraph-local candidates),
then isotonic on a head-disjoint split. pi is relative to the reference graph (a true but unrecorded fact
counts as false), so its values are small.

| | Calibration candidates (positive share) | Held-out heads: AUC / ECE | BRINK removed hard facts among candidates: AUC | mean pi removed vs other |
| --- | --- | --- | --- | --- |
| Family | 103,400 (0.44%) | 0.929 / 0.001 | 0.897 | 0.026 vs 0.004 |
| FB15k-237 | 821,886 (0.33%) | 0.925 / 0.0002 | 0.924 | 0.034 vs 0.004 |
| IKGQA-WebQSP (rank-free, 3,000 heads) | 490,270 (0.52%) | 0.948 / 0.0004 | - | - |

**CC-GCD decoder** (`cc_gcd/decoding/soft_trie.py`). GCR's KG-Trie is kept for the allowed tokens; the trie
also holds the generator's absent candidate edges, and every path gets S = sum of s(e) (s = 0 for observed
edges in every method, since the small, reference-relative pi would otherwise penalize observed edges;
the methods then differ only in the price of absent edges). Each trie node stores V = max lambda * S over
its completions, and a token moving from node u to child c gets the logit bias V(c) - V(u). The biases
telescope to lambda * S(path) minus a per-question constant (checked in a unit test), so the path
objective is exact while beams are steered as soon as a prefix commits to a weak branch (plan 4.5).
Absent-edge scores: B2 -x per edge; B4 lambda log pi; CC-GCD lambda log sigmoid(logit pi + log(1 - c) -
log(1 - f)), f = 0.01. With a constant c this is B4 plus a fixed per-edge shift, so only per-edge c can
separate CC-GCD from tuned B2 + B4 (plan risk "B3 equals CC-GCD"). B3 (474 tuned per-relation penalties)
needs hundreds of LLM runs per value and is not run; B6 (oracle c) has no per-edge truth on BRINK.

**E1, the crucial experiment (plan section 10)** (`results/e1_brink/`, `results/e1_ikgqa/`). GCR step 1 with
`rmanluo/GCR-Qwen2-0.5B-Instruct` (bf16, group beam search, k = 10) on the RTX 3060 Ti. The 8B checkpoint
needs ~16 GB in bf16 and does not fit the 8 GB card; 4-bit quantization would change the logits the
evidence is added to (plan 12.1), so it was not used. All methods share the model, prompt, beams and
candidates (BRINK: the val-selected generator per graph, hop 1; IKGQA: local hybrid, R = 10, k = 5).
Protocol fixed before testing: each family's setting chosen on dev by the target metric (BRINK: Hits@Hard,
then Hits@Any, then F1, on val[:150] of FB15k-237; IKGQA: Hit, then F1, on 100 dev questions at IKG-80),
then run once on test (BRINK-FB15k-237 test, 540 q; IKGQA eval, 300 q). Family test (198 q) and IKG-40
reuse the FB15k-237 / IKG-80 settings unchanged. Paired bootstrap over questions (10,000 resamples,
`scripts/e1_bootstrap.py`). Selected settings: B2 = 10 nats per absent edge, B4 lambda = 0.5 (BRINK) / 2
(IKGQA), CC-GCD c = 0.9 with lambda = 2 (BRINK) / 1 (IKGQA), per-edge c from the natural Wikidata model with
lambda = 2 (BRINK) / 1 (IKGQA).

| BRINK-FB15k-237 test (540 q) | Hits@Any | Hits@Hard | F1 | KG paths via an absent edge | ... of which BRINK-removed (true) |
| --- | --- | --- | --- | --- | --- |
| B1 hard trie (GCR) | 0.759 | 0.254 | 0.151 | 0% | - |
| B1+ candidates, no penalty | 0.732 | 0.244 | 0.123 | 42.6% | 8.5% |
| B2 fixed penalty (10 nats) | 0.767 | 0.261 | 0.152 | 1.3% | 23.6% |
| B4 KGE prior (lambda 0.5) | 0.767 | **0.267** | 0.140 | 25.1% | 9.9% |
| CC-GCD, constant c = 0.9 (B5) | 0.765 | 0.257 | 0.153 | 0.4% | **56.3%** |
| CC-GCD, per-edge c (natural Wikidata) | **0.770** | 0.265 | **0.153** | 1.4% | 29.5% |

Against B1 (paired bootstrap, 95% interval): CC-GCD with per-edge c +1.1 Hits@Any [+0.4, +2.0] (p = 0.002) and
+1.1 Hits@Hard [+0.4, +2.0] (p = 0.001); B2 +0.7 / +0.7 (p = 0.014 / 0.017); CC-GCD constant c +0.6 Hits@Any
(p = 0.047), F1 +0.12 (p = 0.013); B1+ -2.8 Hits@Any (p = 0.98 for "better") and -2.8 F1. Against the best
simple penalty, per-edge CC-GCD vs B2: +0.4 Hits@Any and Hits@Hard [-0.4, +1.1] (p = 0.22), F1 +0.11 (p = 0.10);
vs B4: Hits equal within noise, F1 +1.3 [+0.7, +2.0] (B4 buys its Hits@Hard with many absent edges and loses
F1).

| IKGQA-WebQSP (300 eval q) | IKG-80 Hit | IKG-80 F1 | IKG-40 Hit | IKG-40 F1 |
| --- | --- | --- | --- | --- |
| B1 hard trie | 0.707 | 0.231 | 0.803 | 0.294 |
| B1+ no penalty | 0.707 | 0.234 | 0.793 | 0.292 |
| B2 (10 nats) | 0.710 | 0.232 | 0.803 | 0.295 |
| B4 (lambda 2) | 0.707 | 0.232 | 0.803 | 0.295 |
| CC-GCD, c = 0.9 | **0.713** | **0.233** | **0.810** | **0.297** |
| CC-GCD, per-edge c | **0.713** | 0.233 | 0.807 | 0.295 |

IKGQA vs B1: CC-GCD (c = 0.9) Hit +0.7 at both levels (2 questions, p = 0.13) and F1 +0.26 / +0.27 [+0.03, +0.6]
(p = 0.008 / 0.005); vs B4 F1 +0.2 (p = 0.06 / 0.05). Per-edge natural c is no better than constant c here.
Family test (198 q, FB15k-237 settings transferred): B2 and both CC-GCD settings never take an absent edge and
equal B1 exactly (Hits@Any 0.591, Hits@Hard 0.237); B1+ and B4 lose Hits@Hard (0.187, 0.207).

Findings:
- Unpenalized candidates hurt (B1+ on every benchmark): with no price, 11-66% of generated KG paths run
  through hypothetical edges, of which only 6-15% are true.
- Pricing absent edges recovers and slightly exceeds the hard trie: on BRINK-FB15k-237 per-edge CC-GCD gains
  +1.1 points on both Hits metrics over GCR's hard trie (significant), and every soft method that keeps the
  number of absent edges small is at least as good as B1. The kept absent edges are much more often true than
  the candidate pool (CC-GCD constant c: 56% vs ~5% of candidates).
- The calibration itself is not what buys the gain. Per-edge CC-GCD does not beat a tuned fixed penalty (B2)
  on BRINK (+0.4, p = 0.22) and per-edge c is no better than constant c on IKGQA. All gains are about 1 point,
  well below the plan's go threshold (2 points over the best of B3/B4, out of mechanism). By the decision
  rule of plan section 10 this is **Reframe**, not Go: the soft trie with an explicit generator and a priced
  absence beats the hard trie, while the calibrated observation model does not separate from simple penalties
  with this model. Not Stop: CC-GCD does beat B1 with the same generator.
- Caveats: the main table uses the 0.5B model (the plan's main setting, the 8B GCR checkpoint, was run in 4-bit
  and in bf16 on a second machine; see below); B3 is approximated by four relation-frequency groups; B6 is an evidence oracle (true hypothetical
  edges free) rather than the true per-edge mechanism, which BRINK does not define; IKGQA's two levels share the
  same 300 questions. Two KGE seeds on BRINK agree (below). B1's few "absent edges used"
  on IKGQA come from beams that wrote a path without opening <PATH> (unconstrained) and matched a candidate.
- Next steps that could change the verdict: the 8B checkpoint now has all E1 methods in both 4-bit and bf16
  (below; same verdict); better evidence,
  since the B6 bound shows +7 Hits@Hard is available to the same decoder. On the Wikidata-domain QA benchmark below
  (natural missingness, in-domain model) the priced soft trie gains 4-5 points of Hits@Hard over B1, but the
  calibrated c(e) still does not beat the tuned KGE-prior penalty: the contribution to claim is the explicit
  generator + soft trie + calibrated pi, not the observation model (Reframe).

**Follow-up runs: upper bound, per-relation penalties, second seed, 8B, Wikidata-domain QA.**

- B6-style upper bound (BRINK-FB15k-237 test, 540 q): the same decoder with perfect evidence about the hypothetical
  edges (BRINK-removed candidates free, all other candidates 10 nats) reaches Hits@Any 0.802 (+4.3 over B1),
  Hits@Hard 0.324 (+7.0) and F1 0.161 (+1.0); 92% of the absent edges it uses are true. The generator and the
  soft trie can therefore deliver gains well above the plan's 2-point threshold: the bottleneck is the quality of
  the evidence (pi, c), not the decoding machinery.
- B3 (per-relation penalties): one penalty per relation-frequency quartile (inverse-augmented observed graph),
  coordinate search from 10 nats everywhere over {1, 3, 10} per quartile on FB15k-237 val[:150] (9 dev runs).
  No quartile improves on 10 nats (lowering the penalty for any quartile costs 0.7-3.3 points of Hits@Any or
  Hits@Hard on dev), so the selected B3 is the uniform B2:10; its test run reproduces B2:10 exactly (decoding is
  deterministic). Per-relation pricing adds nothing over one tuned penalty here.
- Second seed (KGE seed 1: its own 2% holdout, candidates, pi; holdout MRR 0.427, hard-fact recovery 57.2% vs 54.4%,
  pi AUC on BRINK removals 0.922 vs 0.924). FB15k-237 test, seed 1 / seed 0: B1+ 0.739 / 0.732 Hits@Any; B2:10
  0.769 / 0.767; CC-GCD c = 0.9 0.765 / 0.765; per-edge CC-GCD 0.772 / 0.770 (Hits@Hard 0.265 / 0.265). Per-edge
  CC-GCD vs B1 with seed 1: +1.3 Hits@Any (p = 0.0008), +1.1 Hits@Hard (p = 0.002); vs B2:10 +0.4 (p = 0.13). Both
  seeds give the same verdict. (Beam decoding is deterministic; question-level variance is in the bootstrap.)
- Reliability: GCR's generate_sentence swallows exceptions; a first seed-1 run lost 352 of 540 questions to CUDA
  out-of-memory as silent empty predictions. All other logs have zero OOM errors (checked); the runner now retries a
  failed generation once and never writes it as empty, and the E1 drivers resume until every question is answered.
- 8B (GCR-Meta-Llama-3.1-8B-Instruct), FB15k-237 test, all eight E1 methods. The 8 GB RTX 3060 Ti cannot hold it
  and the RTX 3060 is not reachable from the Claude app (its shells run with NoNewPrivs, so `su`/`sg`/`sudo` fail).
  - **4-bit NF4 on an RTX 5070 Ti Laptop (12 GB, second machine), all 540 questions**
    (`results/e1_brink/fb15k237_test_GCR-Meta-Llama-3.1-8B-Instruct-4bit.json`; paired bootstraps, 10,000
    resamples, against B1, B2:10, B4:0.5 and CC:0.9:2 beside it):

    | Method | Hits@Any | Hits@Hard | F1 | vs B1 |
    |---|---|---|---|---|
    | B1 hard trie | 0.833 | 0.228 | 0.193 | |
    | B1+ candidates, no penalty | 0.857 | **0.294** | 0.170 | Any +2.4 (p = 0.054), Hard +6.7 (p < 0.001), F1 -2.2 (worse, p < 0.001) |
    | B2:10 fixed penalty | 0.833 | 0.235 | 0.194 | Hard +0.7 (p = 0.015) |
    | B3 per-relation quartiles | 0.833 | 0.235 | 0.194 | identical to B2:10 (the dev-selected quartile penalties are all 10) |
    | B4:0.5 KGE prior | **0.876** | **0.294** | 0.186 | Any +4.3, Hard +6.7 (both p < 0.001), F1 -0.7 (n.s.) |
    | CC:0.9:2 constant c | 0.837 | 0.235 | 0.195 | Any +0.4 (n.s.), Hard +0.7 (p = 0.014), F1 +0.2 (p < 0.001) |
    | CCe per-edge c(e) | 0.837 | 0.239 | **0.196** | Any +0.4 (n.s.), Hard +1.1 (p = 0.002), F1 +0.3 (p < 0.001) |
    | OR:10 evidence oracle (B6) | 0.915 | 0.341 | 0.217 | Any +8.2, Hard +11.3, F1 +2.5 (all p < 0.001) |

    CCe vs B2:10: Hard +0.4 (p = 0.22), F1 +0.2 (p = 0.03). CCe vs B4: Any -3.9, Hard -5.6 (both p < 0.001),
    F1 +1.0 (p = 0.013). A first attempt stopped at question 207 of CC when the host's user rebooted it; the run
    resumed from the saved answers. (CC on those first 207 questions alone read 0.865 / 0.314 / 0.217; that was a
    subset, and only the full-set numbers above are comparable across methods.)
  - **bf16 (full precision) on the same RTX 5070 Ti, all 540 questions**
    (`results/e1_brink/fb15k237_test_GCR-Meta-Llama-3.1-8B-Instruct-bf16.json`, bootstraps vs B1, B2:10, B4:0.5,
    CC:0.9:2 beside it). The 15 GB of weights do not fit the 12 GB card, so 10.5 GiB stay on the GPU and the rest
    is kept in RAM and streamed to the GPU each step (`CCGCD_MAX_GPU_MEM`); all compute is on the GPU, ~38 s per
    question, 46 h for the eight methods. One generation that ran out of memory was redone with an 8 GiB cap.

    | Method | Hits@Any | Hits@Hard | F1 | vs B1 |
    |---|---|---|---|---|
    | B1 hard trie | 0.824 | 0.222 | 0.187 | |
    | B1+ candidates, no penalty | 0.854 | 0.293 | 0.171 | Any +3.0 (p = 0.02), Hard +7.0 (p < 0.001), F1 -1.6 (worse, p < 0.001) |
    | B2:10 fixed penalty | 0.824 | 0.231 | 0.189 | Hard +0.9 (p = 0.005) |
    | B3 per-relation quartiles | 0.824 | 0.231 | 0.189 | identical to B2:10 again |
    | B4:0.5 KGE prior | **0.870** | **0.291** | 0.183 | Any +4.6, Hard +6.9 (both p < 0.001), F1 -0.4 (n.s.) |
    | CC:0.9:2 constant c | 0.826 | 0.231 | 0.189 | Hard +0.9 (p = 0.004), F1 +0.2 (p = 0.02) |
    | CCe per-edge c(e) | 0.828 | 0.231 | **0.190** | Hard +0.9 (p = 0.005), F1 +0.2 (p = 0.009) |
    | OR:10 evidence oracle (B6) | 0.907 | 0.335 | 0.213 | Any +8.3, Hard +11.3, F1 +2.6 (all p < 0.001) |

    CCe vs B2:10: no difference (Hard 0.0, F1 +0.1, p = 0.10). CCe vs B4: Any -4.3, Hard -5.9 (both p < 0.001), F1
    +0.6 (p = 0.07). bf16 and 4-bit agree within about one point on every method and give the same ranking and the
    same significant differences, so the 4-bit table above is not a quantization artefact.
  - **fp32 on CPU, first 100 questions** (`..._test[-100]_GCR-Meta-Llama-3.1-8B-Instruct-cpu.json`, bootstraps vs
    B1, B2:10, B4:0.5): B1 0.89 / 0.31 / 0.226, B1+ 0.84 / 0.30 / 0.192, B2:10 0.89 / 0.32 / 0.231, B4:0.5 0.88 /
    0.33 / 0.209, CC:0.9:2 0.90 / 0.33 / 0.230, CCe 0.89 / 0.32 / 0.230, OR:10 0.97 / 0.43 / 0.254 (OR vs B1: Any +8,
    Hard +12, F1 +2.7, all p <= 0.001; CCe vs B1 F1 +0.4, p = 0.03; vs B2:10 none). B3 was stopped at 81 of 100 on
    the CPU and replaced by the full GPU run above. The 0.5B on the same questions: B1 0.78 / 0.28 / 0.165.
  - 4-bit vs fp32 on those 100 questions: 4-bit is about 2 points lower for every method, per-question Hits@Any
    agreement 0.97-0.98, same method ranking.
  - Reading: the 8B lifts every level by ~8-11 Hits@Any, and the room for priced absent edges grows with it: the
    evidence oracle adds 11.3 Hits@Hard at 8B vs 7.0 at 0.5B. The KGE-prior soft trie (B4) captures 6.7 of those
    11.3 points (vs 1.3 at 0.5B) without losing F1 significantly. The calibrated observation model (CC, CCe)
    stays conservative: small but significant gains over B1 on Hits@Hard and F1 and the best F1 short of the
    oracle, yet no hit gain over the tuned fixed penalty and well below B4 on hits. Pricing nothing (B1+) reaches
    B4's Hits@Hard but loses 2.2 F1. Verdict unchanged (Reframe): the contribution is the explicit generator +
    soft trie + KGE-calibrated pi; c(e) is a precision knob, not the source of the gain.
- Wikidata-domain QA (`data_prep/build_wikidata_qa.py`, BRINK format): the 2019-08-26 Wikidata graph over the
  13,631 entities (139,043 facts) as the observed KG; 34,546 facts true in 2026 but missing in 2019 as the
  natural "removed" set; "What is the <property> of <entity>?" for pairs with exactly one value missing in 2019
  (the hard answer), 150 dev / 400 test questions on the head split of the natural c(e) model (test heads unseen
  by it). KGE (CPU) holdout MRR 0.451; pi holdout AUC 0.934 and AUC 0.920 on the naturally missing facts. With
  the generator (hybrid, R = 10, k = 5, selected on dev) the hard answer becomes reachable for 90.8% of test
  questions instead of 80.5% (+10.3 points only through candidates), and the missing fact itself is a candidate
  for 39%: far more headroom than BRINK. The published GCR-Qwen2-0.5B was fine-tuned on Freebase relation names and
  writes Freebase-style paths on Wikidata labels (hard trie: Hits@Any 0.7% on dev), so it was adapted with GCR's own
  KG-specialization recipe (`scripts/finetune_wikidata_gcr.py`: LoRA r = 16, 1 epoch, 8,000 one-hop paths that were
  *recorded in 2019*, training heads only; no dev/test head and no missing-in-2019 fact is seen; 4 min on the 3060 Ti).
  This adds LLM training that the plan's main setting avoids, so these results are labelled as such.

**E1 on Wikidata-domain QA (natural missingness, adapted GCR-0.5B)** (`results/e1_brink/wikidata_*`). Same protocol:
settings chosen on 150 dev questions (B2 10 nats, B4 lambda 2, CC-GCD c = 0.9 lambda 1, per-edge c lambda 1), test on
400; candidates hybrid, R = 10, k = 5.

| Wikidata QA test (400 q) | Hits@Any | Hits@Hard | F1 | KG paths via an absent edge | ... of which true (missing in 2019) |
| --- | --- | --- | --- | --- | --- |
| B1 hard trie | 0.685 | 0.488 | 0.315 | 0% | - |
| B1+ no penalty | 0.720 | **0.545** | 0.332 | 63.2% | 38.8% |
| B2 (10 nats) | 0.723 | 0.528 | 0.346 | 39.9% | 51.7% |
| B4 KGE prior (lambda 2) | 0.723 | 0.528 | **0.397** | 42.7% | **65.9%** |
| CC-GCD, constant c = 0.9 | **0.728** | 0.535 | 0.391 | 45.8% | 60.0% |
| CC-GCD, per-edge c (natural, in domain) | 0.725 | 0.535 | 0.393 | 46.9% | 59.8% |

- Against B1, every method with candidates gains (paired bootstrap): CC-GCD +4.3 Hits@Any [+2.0, +6.8], +4.8 Hits@Hard
  [+2.5, +7.3], +7.5 F1 [+5.9, +9.2] (all p < 0.001); per-edge CC-GCD +4.0 / +4.8 / +7.8; B4 +3.8 / +4.0 / +8.2. On natural
  missingness with an in-domain model, H1 holds well above the 2-point bar. Even unpenalized candidates help here
  (B1+ +5.8 Hits@Hard), unlike on BRINK: the missing facts are real and the candidate pool contains them.
- Against the best simple baseline (B4), CC-GCD is within noise: +0.5 Hits@Any, +0.8 Hits@Hard (p = 0.17), F1 -0.7
  (n.s.); per-edge c is no better than constant c. Against B2, the pi-based methods gain +4.4 to +5.1 F1 (p < 0.001):
  the KGE plausibility pi is what buys precision; the observation model c adds nothing measurable. H2 fails again.
- Parametric memory (plan 7.3, E5; `data_prep/anonymize_wikidata_qa.py`, method B0 = same prompt and beams with no
  trie). With names, the no-KG B0 reaches Hits@Any 0.685 (= B1) and Hits@Hard 0.585, +9.8 over B1 (p < 0.001): the
  named benchmark partly measures pretraining memory, and the 2019 trie even steers away from facts the model
  knows. With every entity renamed "entity N" (KG, KGE, candidates and pi unchanged; B0 scores 0 by construction):

  | Wikidata QA test, anonymized (400 q) | Hits@Any | Hits@Hard | F1 | absent edges used that are true |
  | --- | --- | --- | --- | --- |
  | B1 hard trie | 0.480 | 0.245 | 0.173 | - |
  | B1+ no penalty | 0.468 | 0.273 | 0.154 | 10.0% |
  | B4 KGE prior (lambda 2) | **0.633** | **0.405** | **0.218** | 46.8% |
  | CC-GCD, constant c = 0.9 | 0.593 | 0.363 | 0.200 | 29.3% |
  | CC-GCD, per-edge c | 0.598 | 0.368 | 0.199 | 27.8% |

  Without names, the evidence carries the gain: B4 +15.3 Hits@Any [+11.3, +19.3] and +16.0 Hits@Hard [+12.0, +20.0]
  over B1, CC-GCD +11.3 / +11.8 and +11.8 / +12.3 (all p < 0.001), while unpenalized candidates do not help (only 10%
  of the absent edges they use are true). H5 holds: the soft-trie gains are not memory. But CC-GCD is significantly
  worse than B4 here (-3.5 to -4.3 points, p < 0.001): the calibrated KGE prior pi is the useful evidence; the
  observation model c makes the absence price worse, not better.

## Reference numbers (GCR paper, WebQSP test)

| Setting | Hit | F1 |
| --- | --- | --- |
| Full GCR, Llama-3.1-8B + ChatGPT (Table 1) | 92.6 | 73.2 |
| Full GCR, Llama-3.1-8B + Llama-3.1-8B general (Table 4) | 90.24 | 71.19 |
| Full GCR, Llama-3.1-8B + Qwen2-7B general (Table 4) | 86.32 | 67.59 |
| Full GCR, Qwen2-0.5B KG-specialized + ChatGPT (Table 4) | 87.48 | 60.03 |
| Step 1 only, Llama-3.1-8B (Table 3) | n/a | 57.0 (P 58.0, R 70.1) |
