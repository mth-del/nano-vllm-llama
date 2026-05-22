#!/usr/bin/env python3
"""Compare unified GPU path vs CPU-P / GPU-D (pd_separation)."""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams


def run_once(model_path: str, pd: bool, seed: int) -> list[list[int]]:
    import gc
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        pd_separation=pd,
        max_model_len=512,
        gpu_memory_utilization=0.85 if pd else 0.9,
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "用一句话介绍你自己。"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    sp = SamplingParams(temperature=0.6, max_tokens=16)
    torch.manual_seed(seed)
    out = llm.generate([prompt], sp, use_tqdm=False)
    llm.exit()
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [o["token_ids"] for o in out]


def main():
    model_path = os.path.expanduser(
        sys.argv[1] if len(sys.argv) > 1 else "~/huggingface/Qwen3-0.6B/"
    )
    if not os.path.isdir(model_path):
        print(f"[Skip] model not found: {model_path}")
        return 0
    seed = 42
    baseline = run_once(model_path, pd=False, seed=seed)
    pd_out = run_once(model_path, pd=True, seed=seed)
    match = baseline == pd_out
    print("baseline token_ids:", baseline)
    print("pd_separation token_ids:", pd_out)
    print("match:", match)
    return 0 if match else 1


if __name__ == "__main__":
    raise SystemExit(main())
