#include "fused_mlp.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

// gate_up: [M, 2*N] row-major, out: [M, N]
// out[i, j] = silu(gate_up[i, j]) * gate_up[i, j + N]
__global__ void silu_mul_kernel(
    const __nv_bfloat16* __restrict__ gate_up,
    __nv_bfloat16* __restrict__ out,
    int M,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) {
        return;
    }
    int row = idx / N;
    int col = idx % N;
    int base = row * (2 * N);
    float g = __bfloat162float(gate_up[base + col]);
    float u = __bfloat162float(gate_up[base + N + col]);
    out[idx] = __float2bfloat16(silu_f(g) * u);
}

void launch_silu_mul(
    const void* gate_up_ptr,
    void* out_ptr,
    int M,
    int N,
    cudaStream_t stream
) {
    const auto* gate_up = static_cast<const __nv_bfloat16*>(gate_up_ptr);
    auto* out = static_cast<__nv_bfloat16*>(out_ptr);
    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    silu_mul_kernel<<<blocks, threads, 0, stream>>>(gate_up, out, M, N);
}
