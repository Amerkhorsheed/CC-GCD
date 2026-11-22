#!/bin/bash
# E1 with the 8B GCR checkpoint (4-bit NF4) on the RTX 3060, for Amer's own runs (machine rule 1: only while the
# shared Qwen server on :8080 is stopped). A watchdog stops the job the moment Qwen answers again.
#
# The 8B model does not fit the 8 GB RTX 3060 Ti (4-bit weights 5.2 GB + KV cache of 10 beams 3.1 GB + context);
# the 12 GB RTX 3060 holds it. Access to the 3060 needs the gpu3060 group in the running session: use `sg` if the
# system has it, otherwise start this from a fresh login (where `id -nG` lists gpu3060).
#
#   scripts/run_8b_on_3060.sh LOGFILE --dataset fb15k237 --split "test[:200]" \
#       --methods B1 B2:10 B4:0.5 CC:0.9:2 CCe:wikidata_gbm_struct:2
#
# Results: results/e1_brink/<dataset>_<split>_GCR-Meta-Llama-3.1-8B-Instruct-4bit.json
set -u
LOG=$1; shift
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPU3060=GPU-dd8fc9ad-ba55-9031-47f8-ee2beb8e2750
ARGS=""
for a in "$@"; do ARGS="$ARGS '${a//\'/\'\\\'\'}'"; done
if curl -s -m 2 http://127.0.0.1:8080/health > /dev/null; then echo "QWEN_RUNNING: not starting (rule 1)" | tee -a "$LOG"; exit 3; fi
INNER="cd '$ROOT' && . .venv/bin/activate && export HF_HOME='$ROOT/.hf' PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU3060 && python scripts/run_e1_brink.py --model_path rmanluo/GCR-Meta-Llama-3.1-8B-Instruct --quant 4bit $ARGS"
if command -v sg > /dev/null; then
  setsid sg gpu3060 -c "$INNER" >> "$LOG" 2>&1 &
elif id -nG | tr ' ' '\n' | grep -qx gpu3060; then
  setsid bash -c "$INNER" >> "$LOG" 2>&1 &
else
  echo "NO_GPU3060_ACCESS: this session lacks the gpu3060 group and sg is not installed; start from a fresh login" | tee -a "$LOG"
  exit 4
fi
PID=$!
while kill -0 $PID 2>/dev/null; do
  if curl -s -m 2 http://127.0.0.1:8080/health > /dev/null; then
    echo "QWEN_STARTED: stopping the 8B job" >> "$LOG"
    kill -- -$PID 2>/dev/null; sleep 5; kill -9 -- -$PID 2>/dev/null
    break
  fi
  sleep 60
done
wait $PID 2>/dev/null
echo "RUN8B_EXIT=$?" >> "$LOG"
