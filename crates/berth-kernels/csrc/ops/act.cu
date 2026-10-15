// silu(gate) * up, half in and half out. Includes the fork's
// act_mul_kernel_h<ACT_SILU> (activation_kernels.cuh). The profile name is
// act_mul_kernel_h<0>. activation.cuh is the ATen launcher and is not included.
//
// GatedMLP on the 27B calls silu_mul(gate, up, out) with act_limit 0 and
// interm_dtype half. numel is intermediate_size (17408) per token. The graph
// path writes a separate half buffer; z may still alias gate or up.

#include "check.cuh"
#include "ops/ops_state.h"

#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "util.h"
#include "util.cuh"
#include "compat.cuh"
#include "reduction.cuh"

#define NUM_THREADS 256
#define NUM_THREADS_P 1024
#define ACT_SILU 0
#define ACT_GELU 1
#define ACT_RELU2 2
#define ACT_SILU_OAI 3
#define ACT_RELU 4

#include "activation_kernels.cuh"

extern "C" int berth_ops_silu_mul(
    const void* gate,
    const void* up,
    void* out,
    int64_t numel,
    float act_limit,
    void* stream)
{
    try
    {
        berth_ops_clear();
        if (!gate || !up || !out)
            throw std::runtime_error("silu_mul: null device pointer");
        if (numel < 2 || (numel & 1))
            throw std::runtime_error("silu_mul: numel must be positive and even");

        size_t blocks = CEIL_DIVIDE((size_t) numel, (size_t) 2 * NUM_THREADS);
        act_mul_kernel_h<ACT_SILU><<<blocks, NUM_THREADS, 0, (cudaStream_t) stream>>>(
            (const half*) gate,
            (const half*) up,
            (half*) out,
            act_limit,
            (size_t) numel);
        berth_cuda_ok(cudaGetLastError(), "silu_mul launch");
        berth_ops_set_launch(
            "act_mul_kernel_h<0> block=256 grid=" + std::to_string(blocks));
        return 0;
    }
    catch (const std::exception& err)
    {
        berth_ops_set_error(err.what());
        cudaGetLastError();
        return -1;
    }
}
