#!/usr/bin/env python3
"""Evaluate nano-vllm on HuggingFaceH4/MATH-500 (test split)."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams

DEFAULT_DATA = os.path.expanduser("~/huggingface/MATH-500/test.jsonl")
DEFAULT_MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Put your final answer in \\boxed{{}}.\n\n{problem}"
)


def load_math500(path: str, limit: int | None, offset: int) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def last_boxed_only_string(string: str) -> str | None:
    idx = string.rfind("\\boxed")
    if idx < 0:
        return None
    i = idx
    num_left_braces = 0
    right_brace_idx = None
    while i < len(string):
        if string[i] == "{":
            num_left_braces += 1
        if string[i] == "}":
            right_brace_idx = i
            num_left_braces -= 1
            if num_left_braces == 0:
                break
        i += 1
    if right_brace_idx is None:
        return None
    retval = string[idx : right_brace_idx + 1]
    if retval.startswith("\\boxed "):
        return "\\boxed{" + retval[len("\\boxed ") :].strip() + "}"
    return retval


def remove_boxed(s: str) -> str:
    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left) : -1]
    return s


def extract_pred_answer(text: str) -> str:
    boxed = last_boxed_only_string(text)
    if boxed is not None:
        return remove_boxed(boxed).strip()
    # fallback: last line / last number-like span
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


def normalize_answer(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("$", "")
    return s.lower()


def is_correct(pred_text: str, gold: str) -> bool:
    pred = normalize_answer(extract_pred_answer(pred_text))
    ref = normalize_answer(gold)
    if not pred:
        return False
    return pred == ref or ref in pred or pred in ref


def build_prompts(rows: list[dict], tokenizer) -> list[str]:
    prompts = []
    for row in rows:
        user = PROMPT_TEMPLATE.format(problem=row["problem"])
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompts


def run_eval(
    model_path: str,
    data_path: str,
    *,
    kv_compress: bool,
    limit: int | None,
    offset: int,
    max_tokens: int,
    temperature: float,
    kv_compress_n: int,
    kv_compress_period: int,
    kv_compress_ratio: float,
    batch_size: int,
    max_num_seqs: int,
) -> dict:
    rows = load_math500(data_path, limit, offset)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        kv_compress=kv_compress,
        kv_compress_n=kv_compress_n,
        kv_compress_period=kv_compress_period,
        kv_compress_ratio=kv_compress_ratio,
        max_num_seqs=max_num_seqs,
    )
    llm.compress_step_count = 0
    sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    correct = 0
    results = []
    outputs = []
    t0 = time.perf_counter()
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        prompts = build_prompts(chunk, tokenizer)
        chunk_out = llm.generate(prompts, sp, use_tqdm=len(rows) <= batch_size)
        outputs.extend(chunk_out)
    elapsed = time.perf_counter() - t0
    compress_events = llm.compress_step_count if kv_compress else 0
    llm.exit()

    for row, out in zip(rows, outputs):
        ok = is_correct(out["text"], row["answer"])
        correct += int(ok)
        results.append({
            "unique_id": row.get("unique_id"),
            "subject": row.get("subject"),
            "correct": ok,
            "gold": row["answer"],
            "pred_extracted": extract_pred_answer(out["text"]),
            "completion": out["text"][:500],
        })

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    n = len(rows)

    return {
        "kv_compress": kv_compress,
        "kv_compress_period": kv_compress_period,
        "kv_compress_ratio": kv_compress_ratio,
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "elapsed_s": elapsed,
        "tok_per_s": sum(len(o["token_ids"]) for o in outputs) / elapsed if elapsed > 0 else 0,
        "compress_events": compress_events,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="MATH-500 eval for nano-vllm")
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--limit", type=int, default=50, help="0 = full 500")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--kv-compress-n", type=int, default=1)
    parser.add_argument("--kv-compress-period", type=int, default=0, help="1024 = blog periodic mode")
    parser.add_argument("--kv-compress-ratio", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16, help="prompts per generate() call")
    parser.add_argument("--max-num-seqs", type=int, default=0, help="scheduler cap, 0=auto")
    parser.add_argument("--repro-blog", action="store_true", help="period=1024 ratio=0.5 max_tokens=1024 limit=500")
    parser.add_argument("--modes", default="baseline,compress", help="baseline,compress")
    parser.add_argument("--out", default=None, help="json report path")
    args = parser.parse_args()

    if args.repro_blog:
        args.limit = 0
        args.max_tokens = 1024
        args.kv_compress_period = 1024
        args.kv_compress_ratio = 0.5
        if args.batch_size == 16:
            args.batch_size = 500
        if args.max_num_seqs == 0:
            args.max_num_seqs = 512

    model_path = os.path.expanduser(args.model)
    if not os.path.isdir(model_path):
        raise SystemExit(f"model not found: {model_path}")
    if not os.path.isfile(args.data):
        raise SystemExit(f"data not found: {args.data}")

    limit = None if args.limit == 0 else args.limit
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    reports = {}

    max_num_seqs = args.max_num_seqs or min(args.batch_size, 512)

    print(f"model={model_path}")
    print(f"data={args.data} limit={limit or 'all'} offset={args.offset}")
    print(
        f"max_tokens={args.max_tokens} batch_size={args.batch_size} "
        f"max_num_seqs={max_num_seqs} period={args.kv_compress_period} ratio={args.kv_compress_ratio}"
    )

    for mode in modes:
        kv = mode == "compress"
        if mode not in ("baseline", "compress"):
            print(f"skip unknown mode: {mode}")
            continue
        print(f"\n=== {mode} ===")
        rep = run_eval(
            model_path,
            args.data,
            kv_compress=kv,
            limit=limit,
            offset=args.offset,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            kv_compress_n=args.kv_compress_n,
            kv_compress_period=args.kv_compress_period if kv else 0,
            kv_compress_ratio=args.kv_compress_ratio,
            batch_size=args.batch_size,
            max_num_seqs=max_num_seqs,
        )
        reports[mode] = rep
        print(
            f"accuracy: {rep['correct']}/{rep['n']} = {rep['accuracy']:.2%}  "
            f"time: {rep['elapsed_s']:.1f}s  decode: {rep['tok_per_s']:.1f} tok/s  "
            f"compress_events: {rep.get('compress_events', 0)}"
        )

    if "baseline" in reports and "compress" in reports:
        b, c = reports["baseline"], reports["compress"]
        print("\n=== compare ===")
        print(f"accuracy  baseline {b['accuracy']:.2%}  compress {c['accuracy']:.2%}  delta {(c['accuracy']-b['accuracy'])*100:+.2f} pp")
        print(f"throughput baseline {b['tok_per_s']:.1f}  compress {c['tok_per_s']:.1f}  ratio {c['tok_per_s']/b['tok_per_s']:.2f}x")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        slim = {k: {kk: vv for kk, vv in v.items() if kk != "results"} for k, v in reports.items()}
        out_path.write_text(json.dumps(slim, indent=2), encoding="utf-8")
        print(f"report saved: {out_path}")


if __name__ == "__main__":
    main()
