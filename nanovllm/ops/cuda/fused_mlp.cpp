#include "fused_mlp.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <vector>

torch::Tensor fused_gate_up_silu_forward(torch::Tensor x, torch::Tensor weight) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "weight must be bfloat16");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(x.size(-1) == weight.size(1), "K dimension mismatch");
    TORCH_CHECK(weight.size(0) % 2 == 0, "weight rows must be even");

    x = x.contiguous();
    weight = weight.contiguous();

    auto x2d = x.dim() == 1 ? x.unsqueeze(0) : x.view({-1, x.size(-1)});
    int64_t M = x2d.size(0);
    int64_t inter = weight.size(0) / 2;

    // Step 1: GEMM still uses cuBLAS via PyTorch (fast, sm_120 safe).
    auto gate_up = at::linear(x2d, weight);
    auto out = torch::empty({M, inter}, x.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_silu_mul(
        gate_up.data_ptr(),
        out.data_ptr(),
        static_cast<int>(M),
        static_cast<int>(inter),
        stream
    );

    if (x.dim() == 1) {
        return out.squeeze(0);
    }
    std::vector<int64_t> out_shape;
    for (int i = 0; i < x.dim() - 1; ++i) {
        out_shape.push_back(x.size(i));
    }
    out_shape.push_back(inter);
    return out.view(out_shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fused_gate_up_silu_forward",
        &fused_gate_up_silu_forward,
        "silu(x@W_gate.T) * (x@W_up.T) with cuBLAS GEMM + CUDA epilogue"
    );
}
