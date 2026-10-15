// deinterleave_qg_gr and mul_sigmoid__gr, without ATen.
//
// mul_sigmoid_kernel_h lives in activation_kernels.cuh. That header has no
// include guard, and the MLP kernels include it at global scope. The include
// is inside a namespace so the __global__ symbols do not collide at link.
// deinterleave_qg_kernel is defined in activation.cu (it includes ATen), so
// the device function is transcribed here. NUM_THREADS is 256, matching
// activation.cu.

#include "attn/error.h"
#include "check.cuh"

#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <cuda_runtime.h>

#include <cstdint>

// reduction.cuh uses NEG_INF_F16 (util.cuh). activation_kernels.cuh uses
// tanh_opt (compat.cuh). Neither header includes them. util.cuh also pulls
// cuBLAS types this file does not need, so the one macro is defined here.
#ifndef NEG_INF_F16
#define NEG_INF_F16 __ushort_as_half(0xFC00)
#endif

#include "reduction.cuh"
#include "compat.cuh"

namespace berth_attn_act {

#define NUM_THREADS 256
#define NUM_THREADS_P 1024
#define ACT_SILU 0
#define ACT_GELU 1
#define ACT_RELU2 2
#define ACT_SILU_OAI 3
#define ACT_RELU 4

#include "activation_kernels.cuh"

#undef ACT_RELU
#undef ACT_SILU_OAI
#undef ACT_RELU2
#undef ACT_GELU
#undef ACT_SILU
#undef NUM_THREADS_P
#undef NUM_THREADS

}  // namespace berth_attn_act

// activation.cu: per head, qg is [q head_dim | g head_dim]. uint4 covers 8 halves.

__global__ __launch_bounds__(256)
void berth_deinterleave_qg_kernel(
    const uint4* __restrict__ qg,
    uint4* __restrict__ q,
    uint4* __restrict__ g,
    const int hd8,
    const size_t n8)
{
    size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if (i >= n8) return;
    size_t d = i % hd8;
    size_t h = i / hd8;
    size_t src = h * 2 * hd8 + d;
    q[i] = qg[src];
    g[i] = qg[src + hd8];
}

extern "C" int berth_attn_deinterleave_qg(
    const void* qg,
    void* q,
    void* g,
    int head_dim,
    int64_t q_numel,
    void* stream)
{
    try {
        berth_attn_clear_error();
        berth_attn_check(qg && q && g, "deinterleave_qg: null pointer");
        berth_attn_check(head_dim > 0 && head_dim % 8 == 0, "deinterleave_qg: head_dim must be a positive multiple of 8");
        berth_attn_check(q_numel > 0 && q_numel % 8 == 0, "deinterleave_qg: q_numel must be a positive multiple of 8");
        berth_attn_check(q_numel % head_dim == 0, "deinterleave_qg: q_numel must be a multiple of head_dim");

        int hd8 = head_dim / 8;
        size_t n8 = (size_t) q_numel / 8;
        size_t blocks = (n8 + 256 - 1) / 256;
        berth_deinterleave_qg_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(
            (const uint4*) qg,
            (uint4*) q,
            (uint4*) g,
            hd8,
            n8);
        berth_cuda_ok(cudaGetLastError(), "deinterleave_qg");
        return 0;
    } catch (const std::exception& err) {
        berth_attn_set_error(err.what());
        return -1;
    }
}

extern "C" int berth_attn_mul_sigmoid(
    void* x,
    const void* y,
    int64_t numel,
    void* stream)
{
    try {
        berth_attn_clear_error();
        berth_attn_check(x && y, "mul_sigmoid: null pointer");
        berth_attn_check(numel > 0 && (numel % 2) == 0, "mul_sigmoid: numel must be positive and even");

        // activation.cu: blocks = ceil(numel / (2 * NUM_THREADS)), NUM_THREADS 256.
        size_t blocks = ((size_t) numel + 2 * 256 - 1) / (2 * 256);
        berth_attn_act::mul_sigmoid_kernel_h<<<blocks, 256, 0, (cudaStream_t) stream>>>(
            (half*) x,
            (const half*) y,
            (size_t) numel);
        berth_cuda_ok(cudaGetLastError(), "mul_sigmoid");
        return 0;
    } catch (const std::exception& err) {
        berth_attn_set_error(err.what());
        return -1;
    }
}
