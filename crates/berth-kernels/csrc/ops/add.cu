// fp32 elementwise add. The 27B decode does not call the fork's add.cu for
// the residual: after the MLP, TransformerBlock.forward does `x += y` with
// both tensors fp32, and that is ATen CUDAFunctor_add<float> (64 launches
// per token, one per layer). The attention residual is fused into
// rms_norm_res_in instead.
//
// IEEE fp32 addition is a single rounding, so this matches torch.add on the
// same values. `c` may alias `a` (the in-place residual).

#include "check.cuh"
#include "ops/ops_state.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace {

constexpr int kThreads = 256;

// No __restrict__: the decode residual is in place (`c` aliases `a`).
__global__ void add_f32_kernel(const float* a, const float* b, float* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

}  // namespace

extern "C" int berth_ops_add_f32(const void* a, const void* b, void* c, int n, void* stream)
{
    try
    {
        berth_ops_clear();
        if (!a || !b || !c)
            throw std::runtime_error("add_f32: null device pointer");
        if (n < 1)
            throw std::runtime_error("add_f32: n must be positive");
        int blocks = (n + kThreads - 1) / kThreads;
        add_f32_kernel<<<blocks, kThreads, 0, (cudaStream_t) stream>>>(
            (const float*) a, (const float*) b, (float*) c, n);
        berth_cuda_ok(cudaGetLastError(), "add_f32 launch");
        berth_ops_set_launch(
            "add_f32 block=256 grid=" + std::to_string(blocks));
        return 0;
    }
    catch (const std::exception& err)
    {
        berth_ops_set_error(err.what());
        cudaGetLastError();
        return -1;
    }
}
