#!/usr/bin/env bash
# Download BRINK (EACL 2026) from Hugging Face into data/raw/brink/.
# Family (~1 MB) and FB15k-237 (~110 MB, three ID variants). Wikidata5m (~1.4 GB) only with --wikidata5m.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw/brink"
HF=https://huggingface.co/datasets/ZDZR
FILES="knowledge_graph_complete.tsv knowledge_graph_incomplete.tsv train.tsv val.tsv test.tsv"

mkdir -p "$RAW/family"
for f in $FILES README.md; do
  curl -fsSL -o "$RAW/family/$f" "$HF/BRINK-family/resolve/main/$f"
done

for sub in EntityID Textlabel FB15k-237_PrivateID; do
  mkdir -p "$RAW/fb15k237/$sub"
  for f in $FILES; do
    curl -fsSL -o "$RAW/fb15k237/$sub/$f" "$HF/BRINK-FB15k-237/resolve/main/$sub/$f"
  done
done
curl -fsSL -o "$RAW/fb15k237/README.md" "$HF/BRINK-FB15k-237/resolve/main/README.md"

if [ "${1:-}" = "--wikidata5m" ]; then
  mkdir -p "$RAW/wikidata5m/kg" "$RAW/wikidata5m/qid"
  for f in entity.txt facts.tsv incomplete_facts.tsv relation.txt; do
    curl -fsSL -o "$RAW/wikidata5m/kg/$f" "$HF/BRINK-Wikidata5m/resolve/main/wikidata5m/kg/$f"
  done
  for s in train val test; do
    for f in ${s}_cleaned_final.tsv ${s}_cleaned_final_complete.tsv; do
      curl -fsSL -o "$RAW/wikidata5m/qid/$f" "$HF/BRINK-Wikidata5m/resolve/main/wikidata5m/qid/$f"
    done
  done
fi

wc -l "$RAW"/family/*.tsv "$RAW"/fb15k237/EntityID/*.tsv
