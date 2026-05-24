#pragma once

#include <cuda_runtime.h>

void launch_silu_mul(
    const void* gate_up,
    void* out,
    int M,
    int N,
    cudaStream_t stream
);
