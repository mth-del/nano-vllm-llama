#!/usr/bin/env python3
"""Benchmark fused MLP GEMM backends: old / ref / cuda on Qwen2.5-3B shapes."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import uuid
from dataclasses import asdict, dataclass
from time import perf_counter

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nanovllm import LLM, SamplingParams
from nanovllm.layers.activation import SiluAndMul
from nanovllm.ops.gemm import fused_gate_up_silu, gate_up_silu_reference


MICRO_SCENARIOS = [
    ("512", 512),
    ("2k", 2048),
    ("4k", 4096),
    ("8k", 8192),
]

E2E_SCENARIOS = [
    ("2k", 2048),
    ("4k", 4096),
]


@dataclass
class MicroResult:
    backend: str
    scenario: str
    num_tokens: int
    us_per_call: float
    layers: int
    ms_per_prefill_step: float


@dataclass
class E2EResult:
    backend: str
    scenario: str
    prompt_tokens: int
    max_tokens: int
    prefill_s: float
    prefill_tok_s: float
    run_index: int


def make_prompt(tokenizer, target_prompt_tokens: int) -> str:
    chunk = (
        "请说明大模型推理中 GEMM、MLP gate_up 与 SiLU 融合，"
        "以及 prefill 与 decode 阶段算子差异。"
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


def bench_us(fn, warm: int, iters: int) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (perf_counter() - t0) / iters * 1e6


def reset_gemm_state(backend: str) -> None:
    os.environ["NANOVLLM_GEMM_BACKEND"] = backend
    os.environ["NANOVLLM_FUSED_MLP"] = "1"
    import nanovllm.ops.gemm as gemm_mod

    gemm_mod._CUDA_AVAILABLE = None
    if backend == "cuda":
        from nanovllm.ops.cuda import load_fused_mlp_extension

        load_fused_mlp_extension.cache_clear()
        load_fused_mlp_extension()
        gemm_mod._CUDA_AVAILABLE = True


def run_micro(
    model_path: str,
    backends: list[str],
    warm: int,
    iters: int,
) -> list[MicroResult]:
    cfg = AutoConfig.from_pretrained(model_path)
    hs, inter, layers = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
    weight = torch.randn(2 * inter, hs, device="cuda", dtype=torch.bfloat16)
    act = SiluAndMul().cuda()

    rows: list[MicroResult] = []
    for scenario, n_tokens in MICRO_SCENARIOS:
        x = torch.randn(n_tokens, hs, device="cuda", dtype=torch.bfloat16)

        if "old" in backends:
            t_old = bench_us(lambda: act(F.linear(x, weight)), warm, iters)
            rows.append(
                MicroResult("old", scenario, n_tokens, t_old, layers, t_old * layers / 1e3)
            )

        if "ref" in backends:
            reset_gemm_state("ref")
            t_ref = bench_us(lambda: gate_up_silu_reference(x, weight), warm, iters)
            rows.append(
                MicroResult("ref", scenario, n_tokens, t_ref, layers, t_ref * layers / 1e3)
            )

        if "cuda" in backends:
            reset_gemm_state("cuda")
            t_cuda = bench_us(lambda: fused_gate_up_silu(x, weight), warm, iters)
            rows.append(
                MicroResult("cuda", scenario, n_tokens, t_cuda, layers, t_cuda * layers / 1e3)
            )

    return rows


def run_e2e_once(
    model_path: str,
    backend: str,
    prompt: str,
    prompt_tokens: int,
    max_tokens: int,
    max_model_len: int,
    enforce_eager: bool,
    gpu_util: float,
) -> tuple[float, float]:
    reset_gemm_state(backend)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    llm = LLM(
        model_path,
        enforce_eager=enforce_eager,
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_util,
    )
    sp = SamplingParams(temperature=0.6, max_tokens=max_tokens)
    llm.add_request(prompt, sp)

    prefill_s = 0.0
    while not llm.is_finished():
        t0 = perf_counter()
        _, num_tokens = llm.step()
        dt = perf_counter() - t0
        if num_tokens > 0:
            prefill_s += dt

    llm.exit()
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tps = prompt_tokens / prefill_s if prefill_s > 0 else 0.0
    return prefill_s, tps


def run_e2e(
    model_path: str,
    backends: list[str],
    max_tokens: int,
    e2e_runs: int,
    enforce_eager: bool,
    gpu_util: float,
    warmup_e2e: bool,
) -> list[E2EResult]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    rows: list[E2EResult] = []

    for scenario, target in E2E_SCENARIOS:
        prompt = make_prompt(tokenizer, target)
        prompt_tokens = len(tokenizer.encode(prompt))
        max_model_len = prompt_tokens + max_tokens + 128

        if warmup_e2e:
            for backend in backends:
                run_e2e_once(
                    model_path,
                    backend,
                    "warmup",
                    32,
                    4,
                    512,
                    enforce_eager,
                    gpu_util,
                )

        for backend in backends:
            for run_idx in range(e2e_runs):
                prefill_s, tps = run_e2e_once(
                    model_path,
                    backend,
                    prompt,
                    prompt_tokens,
                    max_tokens,
                    max_model_len,
                    enforce_eager,
                    gpu_util,
                )
                rows.append(
                    E2EResult(
                        backend=backend,
                        scenario=scenario,
                        prompt_tokens=prompt_tokens,
                        max_tokens=max_tokens,
                        prefill_s=prefill_s,
                        prefill_tok_s=tps,
                        run_index=run_idx,
                    )
                )

    return rows


def summarize_e2e(rows: list[E2EResult]) -> None:
    from collections import defaultdict

    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        groups[(r.scenario, r.backend)].append(r.prefill_s)

    print("\n=== E2E prefill summary (steady-state: use min or last run if cold start) ===")
    print(f"{'scenario':<8} {'backend':<8} {'runs(s)':<28} {'min_ms':>8} {'avg_ms':>8} {'tok/s@min':>10}")

    for scenario in sorted({k[0] for k in groups}):
        prompt_tokens = next(r.prompt_tokens for r in rows if r.scenario == scenario)
        for backend in sorted({k[1] for k in groups if k[0] == scenario}):
            times = groups[(scenario, backend)]
            t_min = min(times)
            t_avg = sum(times) / len(times)
            runs_str = ", ".join(f"{t * 1000:.1f}" for t in times)
            print(
                f"{scenario:<8} {backend:<8} [{runs_str}]"
                f" {t_min * 1000:8.1f} {t_avg * 1000:8.1f} {prompt_tokens / t_min:10.1f}"
            )

    print("\n=== Speedup (min prefill time, per scenario) ===")
    for scenario in sorted({k[0] for k in groups}):
        ref_t = min(groups[(scenario, "ref")], default=None) if (scenario, "ref") in groups else None
        cuda_t = min(groups[(scenario, "cuda")], default=None) if (scenario, "cuda") in groups else None
        old_t = min(groups[(scenario, "old")], default=None) if (scenario, "old") in groups else None
        if ref_t and cuda_t:
            pct = (1 - cuda_t / ref_t) * 100
            print(f"{scenario}: cuda vs ref = {ref_t / cuda_t:.3f}x ({pct:+.1f}% prefill time)")
        if old_t and cuda_t:
            pct = (1 - cuda_t / old_t) * 100
            print(f"{scenario}: cuda vs old = {old_t / cuda_t:.3f}x ({pct:+.1f}% prefill time)")


def summarize_micro(rows: list[MicroResult]) -> None:
    print("\n=== Micro gate_up+SiLU (μs per call) ===")
    print(f"{'scenario':<8} {'backend':<8} {'us/call':>10} {'36L ms':>10}")

    by_scenario: dict[str, dict[str, MicroResult]] = {}
    for r in rows:
        by_scenario.setdefault(r.scenario, {})[r.backend] = r

    for scenario, data in by_scenario.items():
        for backend in ("old", "ref", "cuda"):
            if backend in data:
                r = data[backend]
                print(f"{scenario:<8} {backend:<8} {r.us_per_call:10.1f} {r.ms_per_prefill_step:10.2f}")

    print("\n=== Micro speedup (cuda vs ref) ===")
    for scenario, data in by_scenario.items():
        if "ref" in data and "cuda" in data:
            ref_us = data["ref"].us_per_call
            cuda_us = data["cuda"].us_per_call
            pct = (1 - cuda_us / ref_us) * 100
            print(
                f"{scenario}: {ref_us / cuda_us:.3f}x ({pct:+.1f}%), "
                f"saved {data['ref'].ms_per_prefill_step - data['cuda'].ms_per_prefill_step:.2f} ms / 36 layers"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark GEMM / fused MLP backends")
    parser.add_argument(
        "model_path",
        nargs="?",
        default="/root/autodl-tmp/Qwen2.5-3B-Instruct",
    )
    parser.add_argument(
        "--backends",
        type=str,
        default="old,ref,cuda",
        help="comma-separated: old,ref,cuda",
    )
    parser.add_argument("--micro-only", action="store_true")
    parser.add_argument("--e2e-only", action="store_true")
    parser.add_argument("--micro-warm", type=int, default=15)
    parser.add_argument("--micro-iters", type=int, default=80)
    parser.add_argument("--e2e-runs", type=int, default=3, help="repeat count per backend/scenario")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    parser.add_argument("--no-eager", action="store_true")
    parser.add_argument("--no-warmup-e2e", action="store_true")
    parser.add_argument("--gpu-util", type=float, default=0.85)
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model_path)
    if not os.path.isdir(model_path):
        print(f"[Skip] model not found: {model_path}")
        return 1

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    eager = args.enforce_eager and not args.no_eager

    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        print(f"GPU: {torch.cuda.get_device_name(0)} (cap {cap[0]}.{cap[1]})")
    print(f"model={model_path}")
    print(f"backends={backends}, enforce_eager={eager}\n")

    micro_rows: list[MicroResult] = []
    e2e_rows: list[E2EResult] = []

    if not args.e2e_only:
        print("=== Running micro benchmark ===")
        micro_rows = run_micro(model_path, backends, args.micro_warm, args.micro_iters)
        summarize_micro(micro_rows)

    if not args.micro_only:
        print("\n=== Running E2E prefill benchmark ===")
        e2e_rows = run_e2e(
            model_path,
            [b for b in backends if b in ("ref", "cuda")],
            args.max_tokens,
            args.e2e_runs,
            eager,
            args.gpu_util,
            warmup_e2e=not args.no_warmup_e2e,
        )
        summarize_e2e(e2e_rows)

    if args.output_json:
        out = os.path.expanduser(args.output_json)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        payload = {
            "micro": [asdict(r) for r in micro_rows],
            "e2e": [asdict(r) for r in e2e_rows],
        }
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
