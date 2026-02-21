#!/usr/bin/env bash
# OCP v0.2.0 batch evaluation of all available Ollama models
set -e
export CUDA_VISIBLE_DEVICES=""

MODELS=(
  "ollama/minimax-m2.5:cloud"
  "ollama/kimi-k2:1t-cloud"
  "ollama/qwen3-coder:480b-cloud"
  "ollama/deepseek-v3.1:671b-cloud"
  "ollama/kimi-k2-thinking:cloud"
  "ollama/gemma3:27b-cloud"
  "ollama/gpt-oss:20b-cloud"
  "ollama/gpt-oss:120b-cloud"
  "ollama/glm-4.6:cloud"
  "ollama/cogito-2.1:671b-cloud"
  "ollama/devstral-2:123b-cloud"
  "ollama/nemotron-3-nano:30b-cloud"
  "ollama/rnj-1:8b-cloud"
  "ollama/minimax-m2.1:cloud"
  "ollama/gemini-3-flash-preview:latest"
  "ollama/qwen3-coder-next:cloud"
  "ollama/kimi-k2.5:cloud"
  "ollama/glm-5:cloud"
  "ollama/qwen3.5:cloud"
  "ollama/lfm2.5-thinking:latest"
  "ollama/ministral-3:14b-cloud"
  "ollama/phi4-mini:3.8b"
)

OUT_DIR="docs/results"
DONE=0
FAILED=0

for MODEL in "${MODELS[@]}"; do
  SLUG=$(echo "$MODEL" | sed 's|ollama/||' | tr ':/' '__')
  TS=$(date +%s)
  OUT="${OUT_DIR}/results_${SLUG}_v02_${TS}.json"
  echo "[$(date +%H:%M)] Running: $MODEL → $OUT"
  if ocp evaluate --model "$MODEL" --tests meta_cognition --sessions 2 --seed 42 --output "$OUT" 2>&1; then
    echo "  ✓ OK: $MODEL"
    DONE=$((DONE+1))
  else
    echo "  ✗ FAILED: $MODEL"
    FAILED=$((FAILED+1))
    rm -f "$OUT"
  fi
done

echo ""
echo "=== DONE: $DONE succeeded, $FAILED failed ==="
echo "Updating index.json..."
python3 -c "
import json, glob, os
files = sorted(glob.glob('${OUT_DIR}/results_*_v02_*.json'))
names = [os.path.basename(f) for f in files]
open('${OUT_DIR}/index.json','w').write(json.dumps(names, indent=2))
print(f'index.json updated: {len(names)} files')
"
