#!/usr/bin/env python3
"""Stress GPU KV pool: baseline vs PD (delayed GPU KV)."""

import argparse
import gc
import os
import subprocess
import sys
from time import perf_counter

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm import LLM, SamplingParams


def make_prompt(tokenizer, target_tokens: int, uid: str = "") -> str:
    chunk = "请简要说明 KV cache 分页如何提升 LLM 推理并发能力。"
    if uid:
        chunk += f" [{uid}]"
    content = chunk
    while True:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if len(tokenizer.encode(text)) >= target_tokens:
            return text
        content += chunk


def run_stress(model_path: str, pd: bool, num_reqs: int, prompt_tokens: int, max_tokens: int, gpu_util: float):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    import uuid
    prompt = make_prompt(tokenizer, prompt_tokens, uid=uuid.uuid4().hex[:8])
    sp = SamplingParams(temperature=0.6, max_tokens=max_tokens)
    prompt_len = len(tokenizer.encode(prompt))
    max_model_len = prompt_len + max_tokens + 64

    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        pd_separation=pd,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_util,
        max_num_seqs=max(num_reqs, 1),
    )
    blocks = llm.config.num_kvcache_blocks
    blocks_per_seq = (prompt_len + max_tokens + llm.config.kvcache_block_size - 1) // llm.config.kvcache_block_size

    max_prefill_ready = 0
    max_running = 0
    max_waiting = 0
    t0 = perf_counter()
    for i in range(num_reqs):
        p = make_prompt(tokenizer, prompt_tokens, uid=f"req{i}")
        llm.add_request(p, sp)
    while not llm.is_finished():
        llm.step()
        sch = llm.scheduler
        max_running = max(max_running, len(sch.running))
        max_waiting = max(max_waiting, len(sch.waiting))
        if pd:
            max_prefill_ready = max(max_prefill_ready, len(sch.prefill_ready))
    elapsed = perf_counter() - t0

    llm.exit()
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "kv_blocks": blocks,
        "wall_s": elapsed,
        "prompt_len": prompt_len,
        "blocks_per_seq": blocks_per_seq,
        "max_prefill_ready": max_prefill_ready,
        "max_running": max_running,
        "max_waiting": max_waiting,
    }


def run_stress_subprocess(model_path: str, pd: bool, num_reqs: int, prompt_tokens: int, max_tokens: int, gpu_util: float):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = f"""
import os, sys, json
sys.path.insert(0, {repr(root)})
from scripts.stress_kv_capacity import run_stress
r = run_stress({repr(model_path)}, {pd}, {num_reqs}, {prompt_tokens}, {max_tokens}, {gpu_util})
print("RESULT", json.dumps(r))
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "unknown error")
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            import json
            return json.loads(line[len("RESULT "):])
    raise RuntimeError(f"no RESULT in output: {proc.stdout}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", nargs="?", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--num-reqs", type=int, default=4)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--gpu-util", type=float, default=0.50)
    parser.add_argument("--in-process", action="store_true")
    args = parser.parse_args()
    model_path = os.path.expanduser(args.model_path)
    if not os.path.isdir(model_path):
        print(f"[Skip] model not found: {model_path}")
        return 0

    print("=== KV capacity stress (Phase 4) ===")
    print(
        f"num_reqs={args.num_reqs}, prompt≈{args.prompt_tokens}, max_tokens={args.max_tokens}, "
        f"gpu_util={args.gpu_util}\n"
    )

    runner = run_stress if args.in_process else run_stress_subprocess
    results = {}
    for name, pd in [("baseline", False), ("pd", True)]:
        try:
            r = runner(model_path, pd, args.num_reqs, args.prompt_tokens, args.max_tokens, args.gpu_util)
            results[name] = r
            print(
                f"{name}: kv_blocks={r['kv_blocks']}, ~blocks/seq={r['blocks_per_seq']}, "
                f"wall={r['wall_s']:.2f}s, max_running={r['max_running']}, "
                f"max_waiting={r['max_waiting']}, max_prefill_ready={r['max_prefill_ready']}"
            )
        except Exception as e:
            print(f"{name}: FAILED — {e}")
            results[name] = None

    b, p = results.get("baseline"), results.get("pd")
    if b and p:
        print(f"\nwall: pd/baseline = {p['wall_s']/b['wall_s']:.2f}x")
        print(
            f"max_running: baseline={b['max_running']}, pd={p['max_running']} "
            f"(实际同时在 GPU decode 的序列数)"
        )
        print(
            f"max_prefill_ready (PD only): {p['max_prefill_ready']} "
            f"(host 排队; baseline 用 max_waiting={b['max_waiting']} 粗看 GPU prefill 阻塞)"
        )
        cap = b["kv_blocks"] // max(b["blocks_per_seq"], 1)
        print(f"理论 KV 并发上限(粗算): ~{cap} 路 decode @ {b['blocks_per_seq']} blocks/seq")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
