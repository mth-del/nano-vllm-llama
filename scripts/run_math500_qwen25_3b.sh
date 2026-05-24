#!/usr/bin/env bash
# nano-vllm 虚拟环境 + Qwen2.5-3B MATH-500 评测
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
  "$@"
