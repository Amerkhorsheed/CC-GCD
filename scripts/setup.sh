#!/usr/bin/env bash
# Clone pinned third-party code and create the Python environment.
#   bash scripts/setup.sh          # CPU torch (enough for data prep and 0.5B smoke tests)
#   bash scripts/setup.sh gpu      # CUDA torch + flash-attn
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p third_party

clone_pinned () {  # url dir commit
  if [ ! -d "third_party/$2" ]; then
    git clone -q "$1" "third_party/$2"
  fi
  git -C "third_party/$2" fetch -q --depth 1 origin "$3" 2>/dev/null || true
  git -C "third_party/$2" checkout -q "$3"
}
# Commits used for the results in results/ (Nov 2026).
clone_pinned https://github.com/RManLuo/graph-constrained-reasoning.git graph-constrained-reasoning 9518e8e
clone_pinned https://github.com/boschresearch/brink.git brink b95506d

if [ ! -d .venv ]; then
  uv venv -p 3.12 .venv
fi
. .venv/bin/activate
if [ "${1:-cpu}" = "gpu" ]; then
  uv pip install "torch==2.4.1"
else
  uv pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.4.1"
fi
uv pip install -r requirements.txt
if [ "${1:-cpu}" = "gpu" ]; then
  uv pip install flash-attn --no-build-isolation || echo "flash-attn not installed; use --attn_implementation sdpa"
fi
echo "Activate with: . .venv/bin/activate && export HF_HOME=$ROOT/.hf"
