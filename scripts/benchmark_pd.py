#!/usr/bin/env python3
"""Benchmark baseline vs pd_separation: TTFT, TPOT, throughput."""

import argparse
import gc
import os
import sys
import uuid
from dataclasses import dataclass
from time import perf_counter

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm import LLM, SamplingParams


@dataclass
class BenchResult:
    mode: str
    prompt_tokens: int
    output_tokens: int
    ttft_s: float
    prefill_s: float
    tpot_s: float
    total_s: float
    decode_tok_s: float
    end_to_end_tok_s: float
    prefill_steps: int
    decode_steps: int


@dataclass
class Scenario:
    name: str
    target_prompt_tokens: int
    max_tokens: int


SCENARIOS = [
    Scenario("short", 26, 64),
    Scenario("prompt512", 512, 256),
    Scenario("prompt2k", 2048, 256),
]


def make_prompt(tokenizer, target_prompt_tokens: int) -> str:
    chunk = (
        "请说明大模型推理中 prefill 与 decode 的差异、KV cache 分页、"
        "以及 FlashAttention 对吞吐和首 token 延迟的影响。"
        f" [uid={uuid.uuid4().hex[:8]}]"
    )
    content = chunk
    for _ in range(512):
        wrapped = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if len(tokenizer.encode(wrapped)) >= target_prompt_tokens:
            return wrapped
        content += chunk
    return wrapped


def bench_generate(llm, prompt: str, sp: SamplingParams) -> BenchResult:
    llm.add_request(prompt, sp)
    t0 = perf_counter()
    ttft = None
    prefill_steps = 0
    decode_steps = 0
    output_tokens = 0
    prefill_time = 0.0
    decode_time = 0.0

    while not llm.is_finished():
        t_step = perf_counter()
        outputs, num_tokens = llm.step()
        dt = perf_counter() - t_step
        if num_tokens > 0:
            prefill_steps += 1
            prefill_time += dt
        else:
            decode_steps += 1
            decode_time += dt
        for seq in list(llm.scheduler.running):
            if seq.num_completion_tokens > 0 and ttft is None:
                ttft = perf_counter() - t0
            output_tokens = max(output_tokens, seq.num_completion_tokens)
        for _, token_ids in outputs:
            output_tokens = max(output_tokens, len(token_ids))

    total = perf_counter() - t0
    if ttft is None:
        ttft = prefill_time if prefill_time > 0 else total
    decode_tokens = max(output_tokens - 1, 0)
    tpot = decode_time / decode_tokens if decode_tokens > 0 else 0.0

    mode = "pd_separation" if llm.config.pd_separation else "baseline"
    prompt_tokens = len(llm.tokenizer.encode(prompt))
    return BenchResult(
        mode=mode,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ttft_s=ttft,
        prefill_s=prefill_time,
        tpot_s=tpot,
        total_s=total,
        decode_tok_s=decode_tokens / decode_time if decode_tokens and decode_time > 0 else 0.0,
        end_to_end_tok_s=output_tokens / total if total > 0 else 0.0,
        prefill_steps=prefill_steps,
        decode_steps=decode_steps,
    )


def run_mode(
    model_path: str,
    pd: bool,
    prompt: str,
    sp: SamplingParams,
    max_model_len: int,
    warmup: bool,
    tokenizer=None,
) -> BenchResult:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        pd_separation=pd,
        max_model_len=max_model_len,
        max_num_batched_tokens=16384,
        gpu_memory_utilization=0.85 if pd else 0.9,
    )
    prompt_tokens = len(llm.tokenizer.encode(prompt))
    if warmup and tokenizer is not None:
        warm_prompt = make_prompt(tokenizer, 32)
        _ = bench_generate(llm, warm_prompt, SamplingParams(temperature=0.6, max_tokens=4))
    r = bench_generate(llm, prompt, sp)
    r.prompt_tokens = prompt_tokens
    llm.exit()
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return r


def fmt(r: BenchResult) -> str:
    return (
        f"{r.mode}: prompt={r.prompt_tokens}tok, out={r.output_tokens}tok | "
        f"TTFT={r.ttft_s*1000:.1f}ms (prefill={r.prefill_s*1000:.1f}ms), "
        f"TPOT={r.tpot_s*1000:.1f}ms, total={r.total_s:.2f}s | "
        f"decode={r.decode_tok_s:.1f}tok/s, e2e={r.end_to_end_tok_s:.1f}tok/s | "
        f"steps(P/D)={r.prefill_steps}/{r.decode_steps}"
    )


def run_scenario(
    model_path: str,
    scenario: Scenario,
    tokenizer,
    warmup: bool = True,
) -> tuple[BenchResult, BenchResult]:
    prompt = make_prompt(tokenizer, scenario.target_prompt_tokens)
    actual = len(tokenizer.encode(prompt))
    max_model_len = actual + scenario.max_tokens + 64
    sp = SamplingParams(temperature=0.6, max_tokens=scenario.max_tokens)
    print(f"\n--- {scenario.name}: prompt≈{scenario.target_prompt_tokens}tok (actual {actual}), max_tokens={scenario.max_tokens}, max_model_len={max_model_len} ---")
    print("[baseline]")
    base = run_mode(model_path, False, prompt, sp, max_model_len, warmup=warmup, tokenizer=tokenizer)
    print(fmt(base))
    print("[pd_separation]")
    pd = run_mode(model_path, True, prompt, sp, max_model_len, warmup=warmup, tokenizer=tokenizer)
    print(fmt(pd))
    return base, pd


def print_compare(base: BenchResult, pd: BenchResult):
    def r(a, b):
        return f"{a/b:.2f}x" if b > 0 else "n/a"

    print(
        f"  TTFT: {pd.ttft_s*1000:.0f} vs {base.ttft_s*1000:.0f} ms ({r(pd.ttft_s, base.ttft_s)}) | "
        f"TPOT: {pd.tpot_s*1000:.1f} vs {base.tpot_s*1000:.1f} ms ({r(pd.tpot_s, base.tpot_s)}) | "
        f"e2e: {pd.end_to_end_tok_s:.1f} vs {base.end_to_end_tok_s:.1f} tok/s ({pd.end_to_end_tok_s/base.end_to_end_tok_s:.0%})"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", nargs="?", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument(
        "--cases",
        default="short,prompt512,prompt2k",
        help="comma-separated: short,prompt512,prompt2k",
    )
    args = parser.parse_args()
    model_path = os.path.expanduser(args.model_path)
    if not os.path.isdir(model_path):
        print(f"[Skip] model not found: {model_path}")
        return 0

    case_map = {s.name: s for s in SCENARIOS}
    cases = [case_map[c.strip()] for c in args.cases.split(",") if c.strip() in case_map]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("=== nano-vllm PD benchmark (multi-scenario) ===")
    print(f"model: {model_path}, enforce_eager=True")

    summary: list[tuple[str, BenchResult, BenchResult]] = []
    for scenario in cases:
        base, pd = run_scenario(
            model_path,
            scenario,
            tokenizer,
            warmup=(scenario.target_prompt_tokens <= 64),
        )
        print("  compare:", end=" ")
        print_compare(base, pd)
        summary.append((scenario.name, base, pd))

    print("\n=== SUMMARY TABLE ===")
    print(f"{'case':<12} {'prompt':>6} {'out':>4} | {'TTFT_b':>8} {'TTFT_pd':>8} {'ratio':>6} | {'TPOT_b':>7} {'TPOT_pd':>7} | {'e2e_b':>7} {'e2e_pd':>7}")
    for name, base, pd in summary:
        print(
            f"{name:<12} {base.prompt_tokens:>6} {base.output_tokens:>4} | "
            f"{base.ttft_s*1000:>7.0f}ms {pd.ttft_s*1000:>7.0f}ms {pd.ttft_s/base.ttft_s:>5.2f}x | "
            f"{base.tpot_s*1000:>6.1f}ms {pd.tpot_s*1000:>6.1f}ms | "
            f"{base.end_to_end_tok_s:>6.1f} {pd.end_to_end_tok_s:>6.1f} tok/s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
