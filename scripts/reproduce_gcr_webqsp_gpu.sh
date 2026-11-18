#!/usr/bin/env bash
# Full reproduction of GCR (Luo et al., ICML 2025) on RoG-WebQSP test (1,628 questions).
# Needs one GPU with >= 24 GB VRAM (8B model in bf16 + beam 10). No OpenAI key needed.
#
# Targets reported in the GCR paper (WebQSP test):
#   Step 1 only, Llama-3.1-8B ("GCR w/o General LLM", Table 3): F1 57.0 / P 58.0 / R 70.1
#   Full GCR, KG-specialized Llama-3.1-8B + general Qwen2-7B   (Table 4): Hit 86.32 / F1 67.59
#   Full GCR, KG-specialized Llama-3.1-8B + general Llama-3.1-8B (Table 4): Hit 90.24 / F1 71.19
#   Full GCR, + ChatGPT (Table 1): Hit 92.6 / F1 73.2  (needs OPENAI_API_KEY in .env; optional)
# Hard KG-Trie ceiling measured by analysis/kg_trie_coverage.py: Hit 95.6 / Recall 94.1.
#
# Usage (from the cc-gcd folder):  bash scripts/reproduce_gcr_webqsp_gpu.sh [general_llm]
#   general_llm defaults to Qwen/Qwen2-7B-Instruct (not gated). meta-llama/Llama-3.1-8B-Instruct
#   is gated: put HF_TOKEN=... in third_party/graph-constrained-reasoning/.env first.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GCR="$ROOT/third_party/graph-constrained-reasoning"
GENERAL_LLM="${1:-Qwen/Qwen2-7B-Instruct}"
SPLIT="${SPLIT:-test}"
OUT="$ROOT/results"

if [ ! -d "$GCR" ] || [ ! -d "$ROOT/.venv" ]; then
  bash "$ROOT/scripts/setup.sh" gpu
fi
. "$ROOT/.venv/bin/activate"
export HF_HOME="${HF_HOME:-$ROOT/.hf}"

ATTN=sdpa
python -c "import flash_attn" 2>/dev/null && ATTN=flash_attention_2

cd "$GCR"
export PYTHONPATH="$GCR" TOKENIZERS_PARALLELISM=false

# Step 1: graph-constrained decoding with the KG-specialized LLM (paper setting: L=2, k=10, group beam).
KG_LLM=rmanluo/GCR-Meta-Llama-3.1-8B-Instruct
KG_NAME=GCR-Meta-Llama-3.1-8B-Instruct
python workflow/predict_paths_and_answers.py \
  --data_path rmanluo --d RoG-webqsp --split "$SPLIT" \
  --index_path_length 2 --model_name "$KG_NAME" --model_path "$KG_LLM" \
  --k 10 --prompt_mode zero-shot --generation_mode group-beam \
  --attn_implementation "$ATTN" --dtype bf16 \
  --predict_path "$OUT/GenPaths"

PATHS="$OUT/GenPaths/RoG-webqsp/$KG_NAME/$SPLIT/zero-shot-group-beam-k10-index_len2/predictions.jsonl"

# Step 2: graph inductive reasoning with a general LLM over the 10 generated paths.
GEN_NAME="$(basename "$GENERAL_LLM")"
python workflow/predict_final_answer.py \
  --data_path rmanluo --d RoG-webqsp --split "$SPLIT" \
  --model_name "$GEN_NAME" --model_path "$GENERAL_LLM" \
  --reasoning_path "$PATHS" --add_path True -n 1 \
  --attn_implementation "$ATTN" --dtype bf16 \
  --predict_path "$OUT/KGQA"

echo "Step-1 metrics: see the 'Accuracy/Hit/F1' line above and $(dirname "$PATHS")/detailed_eval_result.jsonl"
echo "Step-2 metrics: see $OUT/KGQA/RoG-webqsp/$GEN_NAME/"
