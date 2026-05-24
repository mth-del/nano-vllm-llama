"""GEMM and fused MLP kernels for prefill."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Triton tl.dot may fail to compile on newer GPUs (e.g. sm_120 / RTX 5090).
_TRITON_DOT_BROKEN: bool | None = None


_CUDA_AVAILABLE: bool | None = None


def _cuda_extension_available() -> bool:
    global _CUDA_AVAILABLE
    if _CUDA_AVAILABLE is not None:
        return _CUDA_AVAILABLE
    if not torch.cuda.is_available():
        _CUDA_AVAILABLE = False
        return False
    try:
        from nanovllm.ops.cuda import load_fused_mlp_extension

        load_fused_mlp_extension()
        _CUDA_AVAILABLE = True
    except Exception:
        _CUDA_AVAILABLE = False
    return _CUDA_AVAILABLE


def use_fused_mlp() -> bool:
    return os.environ.get("NANOVLLM_FUSED_MLP", "1") != "0"


def _gemm_backend() -> str:
    """auto | triton | ref"""
    return os.environ.get("NANOVLLM_GEMM_BACKEND", "auto").strip().lower()


def linear_gemm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Unified GEMM entry (cuBLAS via F.linear)."""
    return F.linear(x, weight, bias)


@triton.jit
def _fused_gate_up_silu_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wg_n,
    stride_wg_k,
    stride_wu_n,
    stride_wu_k,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """y = silu(x @ W_gate.T) * (x @ W_up.T); W_gate/W_up stacked in w_ptr."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    w_gate_base = w_ptr
    w_up_base = w_ptr + N * stride_wg_n

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)

        w_gate = tl.load(
            w_gate_base + offs_n[None, :] * stride_wg_n + offs_k[:, None] * stride_wg_k,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)

        w_up = tl.load(
            w_up_base + offs_n[None, :] * stride_wu_n + offs_k[:, None] * stride_wu_k,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)

        acc_gate += tl.dot(x, w_gate)
        acc_up += tl.dot(x, w_up)

    silu_gate = acc_gate * tl.sigmoid(acc_gate)
    out = (silu_gate * acc_up).to(tl.bfloat16)

    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        out,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _pick_block_size(M: int, N: int, K: int) -> tuple[int, int, int]:
    if M >= 4096:
        return 128, 128, 32
    if M >= 1024:
        return 64, 64, 32
    return 32, 64, 32


def gate_up_silu_reference(x: torch.Tensor, gate_up_weight: torch.Tensor) -> torch.Tensor:
    inter = gate_up_weight.shape[0] // 2
    gate_up = linear_gemm(x, gate_up_weight)
    gate, up = gate_up.split(inter, dim=-1)
    return F.silu(gate) * up


def _triton_unsupported() -> bool:
    global _TRITON_DOT_BROKEN
    if _TRITON_DOT_BROKEN is not None:
        return _TRITON_DOT_BROKEN
    if not torch.cuda.is_available():
        _TRITON_DOT_BROKEN = True
        return True
    major, _minor = torch.cuda.get_device_capability()
    # sm_120+ currently breaks Triton tl.dot matmul lowering in our stack.
    _TRITON_DOT_BROKEN = major >= 12
    return _TRITON_DOT_BROKEN


def _resolve_backend() -> str:
    backend = _gemm_backend()
    if backend in ("ref", "reference", "none"):
        return "ref"
    if backend == "cuda":
        return "cuda" if _cuda_extension_available() else "ref"
    if backend == "triton":
        return "triton"
    if backend != "auto":
        raise ValueError(f"Unknown NANOVLLM_GEMM_BACKEND={backend!r}")
    if _cuda_extension_available():
        return "cuda"
    return "ref" if _triton_unsupported() else "triton"


def _fused_gate_up_silu_triton(x: torch.Tensor, gate_up_weight: torch.Tensor) -> torch.Tensor:
    x2d = x.reshape(-1, x.shape[-1]).contiguous()
    weight = gate_up_weight.contiguous()
    M, K = x2d.shape
    inter = weight.shape[0] // 2
    N = inter

    y = torch.empty(M, N, device=x2d.device, dtype=x2d.dtype)
    block_m, block_n, block_k = _pick_block_size(M, N, K)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    _fused_gate_up_silu_kernel[grid](
        x2d,
        weight,
        y,
        M,
        N,
        K,
        x2d.stride(0),
        x2d.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(0),
        weight.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return y.reshape(*x.shape[:-1], N)


def _fused_gate_up_silu_cuda(x: torch.Tensor, gate_up_weight: torch.Tensor) -> torch.Tensor:
    from nanovllm.ops.cuda import fused_gate_up_silu_cuda

    return fused_gate_up_silu_cuda(x, gate_up_weight)


def fused_gate_up_silu(x: torch.Tensor, gate_up_weight: torch.Tensor) -> torch.Tensor:
    """Fused gate/up linear + SiLU*Mul. gate_up_weight: [2*inter, hidden]."""
    if not use_fused_mlp():
        return gate_up_silu_reference(x, gate_up_weight)

    assert gate_up_weight.ndim == 2
    assert x.shape[-1] == gate_up_weight.shape[1]
    assert gate_up_weight.shape[0] % 2 == 0

    backend = _resolve_backend()
    if backend == "ref":
        return gate_up_silu_reference(x, gate_up_weight)
    if backend == "cuda":
        return _fused_gate_up_silu_cuda(x, gate_up_weight)

    global _TRITON_DOT_BROKEN
    try:
        return _fused_gate_up_silu_triton(x, gate_up_weight)
    except RuntimeError:
        _TRITON_DOT_BROKEN = True
        return gate_up_silu_reference(x, gate_up_weight)
