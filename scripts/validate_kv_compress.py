#!/usr/bin/env python3
"""Smoke test: baseline vs kv_compress should produce identical tokens on short run."""

import gc
import sys

import torch

from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams


def run(model_path: str, kv_compress: bool, max_tokens: int = 16, seed: int = 42):
    torch.manual_seed(seed)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    prompt = "The capital of France is"
    llm = LLM(
        model_path,
        enforce_eager=True,
        kv_compress=kv_compress,
        kv_compress_n=1,
        tensor_parallel_size=1,
    )
    out = llm.generate([prompt], sp, use_tqdm=False)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return out[0]["token_ids"]


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not model_path:
        print("Usage: python scripts/validate_kv_compress.py <model_path>")
        sys.exit(1)
    base = run(model_path, False)
    comp = run(model_path, True)
    match = base == comp
    print(f"baseline tokens: {base}")
    print(f"compress tokens: {comp}")
    print(f"match: {match}")
    sys.exit(0 if match else 1)


if __name__ == "__main__":
    main()
