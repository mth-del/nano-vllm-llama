"""Lazy-loaded CUDA extensions."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_CUDA_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def load_fused_mlp_extension():
    import torch
    from torch.utils.cpp_extension import load

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for nanovllm.ops.cuda")

    return load(
        name="nanovllm_fused_mlp",
        sources=[
            str(_CUDA_DIR / "fused_mlp.cpp"),
            str(_CUDA_DIR / "fused_mlp.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


def fused_gate_up_silu_cuda(x, weight):
    ext = load_fused_mlp_extension()
    return ext.fused_gate_up_silu_forward(x, weight)
