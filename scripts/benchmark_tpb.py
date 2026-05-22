#!/usr/bin/env python3
"""Benchmark store_kvcache TPB=1 vs TPB=4 on 2K/4K/8K prompts."""

import argparse
import gc
import json
import os
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from time import perf_counter

import torch
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nanovllm import LLM, SamplingParams


@dataclass
class BenchResult:
    tpb: int
    scenario: str
    target_prompt_tokens: int
    actual_prompt_tokens: int
    max_tokens: int
    prefill_s: float
    decode_s: float
    total_s: float
    prefill_tok_s: float
    decode_tok_s: float
    end_to_end_tok_s: float
    prefill_steps: int
    decode_steps: int


SCENARIOS = [
    ("2k", 2048),
    ("4k", 4096),
    ("8k", 8192),
]


def make_prompt(tokenizer, target_prompt_tokens: int) -> str:
    chunk = (
        "请说明大模型推理中 KV cache 写入、Triton kernel 的 grid/TPB，"
        "以及 prefill 与 decode 阶段 batch token 数 N 的差异。"
        f" [uid={uuid.uuid4().hex[:8]}]"
    )
    content = chunk
    for _ in range(1024):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if len(tokenizer.encode(text)) >= target_prompt_tokens:
            return text
        content += chunk
    return text


def run_once(
    model_path: str,
    tpb: int,
    scenario: str,
    target_prompt_tokens: int,
    max_tokens: int,
    enforce_eager: bool,
) -> BenchResult:
    os.environ["NANOVLLM_KV_TPB"] = str(tpb)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt = make_prompt(tokenizer, target_prompt_tokens)
    prompt_tokens = len(tokenizer.encode(prompt))
    max_model_len = prompt_tokens + max_tokens + 128
    sp = SamplingParams(temperature=0.6, max_tokens=max_tokens)

    llm = LLM(
        model_path,
        enforce_eager=enforce_eager,
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.90,
    )
    # 触发 Triton JIT，避免首轮把编译算进 TPB=1
    warm_sp = SamplingParams(temperature=0.6, max_tokens=4)
    llm.add_request(make_prompt(tokenizer, min(256, target_prompt_tokens)), warm_sp)
    while not llm.is_finished():
        llm.step()
    llm.exit()
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    llm = LLM(
        model_path,
        enforce_eager=enforce_eager,
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.90,
    )
    llm.add_request(prompt, sp)

    prefill_s = decode_s = 0.0
    prefill_steps = decode_steps = 0
    output_tokens = 0
    t0 = perf_counter()

    while not llm.is_finished():
        t_step = perf_counter()
        outputs, num_tokens = llm.step()
        dt = perf_counter() - t_step
        if num_tokens > 0:
            prefill_s += dt
            prefill_steps += 1
        else:
            decode_s += dt
            decode_steps += 1
        for seq in list(llm.scheduler.running):
            output_tokens = max(output_tokens, seq.num_completion_tokens)
        for _, token_ids in outputs:
            output_tokens = max(output_tokens, len(token_ids))

    total_s = perf_counter() - t0
    llm.exit()
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    decode_gen = max(output_tokens - 1, 0)
    total_gen = max(output_tokens, 0)
    return BenchResult(
        tpb=tpb,
        scenario=scenario,
        target_prompt_tokens=target_prompt_tokens,
        actual_prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        prefill_s=prefill_s,
        decode_s=decode_s,
        total_s=total_s,
        prefill_tok_s=prompt_tokens / prefill_s if prefill_s > 0 else 0.0,
        decode_tok_s=decode_gen / decode_s if decode_s > 0 else 0.0,
        end_to_end_tok_s=(prompt_tokens + total_gen) / total_s if total_s > 0 else 0.0,
        prefill_steps=prefill_steps,
        decode_steps=decode_steps,
    )


def run_subprocess(model_path: str, tpb: int, scenario: str, target: int, max_tokens: int, eager: bool) -> BenchResult:
    code = f"""
import json, sys
from dataclasses import asdict
sys.path.insert(0, {repr(ROOT)})
from scripts.benchmark_tpb import run_once, BenchResult
r = run_once({repr(model_path)}, {tpb}, {repr(scenario)}, {target}, {max_tokens}, {eager})
print("RESULT", json.dumps(asdict(r)))
"""
    env = os.environ.copy()
    env["NANOVLLM_KV_TPB"] = str(tpb)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "subprocess failed")
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            d = json.loads(line[len("RESULT ") :])
            return BenchResult(**d)
    raise RuntimeError(f"no RESULT in stdout:\n{proc.stdout}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark KV TPB=1 vs 4 on long prompts")
    parser.add_argument("model_path", nargs="?", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--max-tokens", type=int, default=128, help="decode length per request")
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    parser.add_argument("--no-eager", action="store_true", help="allow CUDA graphs")
    parser.add_argument("--in-process", action="store_true", help="no subprocess isolation")
    parser.add_argument("--tpb", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--scenarios", type=str, default="2k,4k,8k")
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model_path)
    if not os.path.isdir(model_path):
        print(f"[Skip] model not found: {model_path}")
        return 1

    eager = args.enforce_eager and not args.no_eager
    want = {s.strip() for s in args.scenarios.split(",")}
    scenarios = [(n, t) for n, t in SCENARIOS if n in want]
    runner = run_once if args.in_process else run_subprocess

    print("=== KV store_kvcache TPB benchmark ===")
    print(f"model={model_path}")
    print(f"tpb_list={args.tpb}, max_tokens={args.max_tokens}, enforce_eager={eager}\n")

    rows: list[BenchResult] = []
    for name, target in scenarios:
        print(f"--- scenario {name} (target prompt ~{target}) ---")
        for tpb in args.tpb:
            try:
                r = runner(model_path, tpb, name, target, args.max_tokens, eager)
                rows.append(r)
                speedup = ""
                print(
                    f"  TPB={tpb}: prompt={r.actual_prompt_tokens}, "
                    f"prefill={r.prefill_s:.3f}s ({r.prefill_tok_s:.1f} tok/s), "
                    f"decode={r.decode_s:.3f}s ({r.decode_tok_s:.1f} tok/s), "
                    f"total={r.total_s:.3f}s, e2e={r.end_to_end_tok_s:.1f} tok/s, "
                    f"steps prefill/decode={r.prefill_steps}/{r.decode_steps}"
                )
            except Exception as e:
                print(f"  TPB={tpb}: FAILED — {e}")
        print()

    # pairwise speedup TPB=4 vs 1 per scenario
    by_key = {(r.scenario, r.tpb): r for r in rows}
    print("=== TPB=4 vs TPB=1 (ratio >1 means TPB=4 faster) ===")
    for name, _ in scenarios:
        r1, r4 = by_key.get((name, 1)), by_key.get((name, 4))
        if not r1 or not r4:
            continue
        print(
            f"{name}: prefill {r1.prefill_s / r4.prefill_s:.2f}x, "
            f"decode {r1.decode_s / r4.decode_s:.2f}x, "
            f"total {r1.total_s / r4.total_s:.2f}x"
        )

    if args.output_json:
        out = os.path.expanduser(args.output_json)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump([asdict(r) for r in rows], f, indent=2)
        print(f"\nWrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
