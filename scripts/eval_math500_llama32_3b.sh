#!/usr/bin/env bash
# Llama-3.2-3B on 8GB GPU: needs ~6.5GB for BF16 weights + KV.
# Close GPU-heavy apps (browser, IDE) if you hit CUDA OOM (~1.6GB is often used by the desktop).
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:-$HOME/huggingface/Llama-3.2-3B-Instruct/}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Conservative: 8GB laptop cannot run batch=500 like 0.6B.
python3 scripts/eval_math500.py "$MODEL" \
  --limit 0 \
  --max-tokens 512 \
  --max-model-len 2048 \
  --max-num-batched-tokens 2048 \
  --batch-size 4 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.98 \
  --embed-cpu-offload \
  --modes baseline,compress \
  --out results/math500_llama32_3b.json
