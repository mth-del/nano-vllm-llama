#!/usr/bin/env bash
# Qwen2.5-3B MATH-500：对比 GEMM backend (ref vs cuda)
set -euo pipefail

VENV="/root/autodl-tmp/nonallm"
REPO="/root/mth/code_space/nano-vllm-llama"
MODEL="/root/autodl-tmp/Qwen2.5-3B-Instruct"
DATA="/root/huggingface/MATH-500/test.jsonl"

source "${VENV}/bin/activate"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

cd "${REPO}"
python scripts/eval_math500.py "${MODEL}" \
  --data "${DATA}" \
  --modes baseline \
  --gemm-backends ref,cuda \
  --limit 50 \
  --max-tokens 512 \
  --batch-size 16 \
  --out results/math500_gemm_ref_cuda_50.json \
  "$@"
