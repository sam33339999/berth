// In-place fp16 -inf fill of logits[limit, vocab).
//
// CustomSampler.forward writes logits[..., actual_vocab_size:] = -inf before
// any penalty, as fp16 0xFC00. This is not a fork kernel; the fork does the
// fill with a PyTorch slice assignment. limit >= vocab launches nothing.

#include "check.cuh"
#include "sample/sample_state.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace {

constexpr int kThreads = 256;

__global__ void mask_tail_f16_kernel(half* logits, int limit, int vocab)
{
    int idx = limit + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vocab)
        logits[idx] = __ushort_as_half(0xFC00);
}

}  // namespace

extern "C" int berth_sample_mask_tail_f16(void* logits, int limit, int vocab, void* stream)
{
    bool cuda_called = false;
    try
    {
        berth_sample_clear();
        if (!logits)
            throw std::runtime_error("mask_tail_f16: null device pointer");
        if (vocab < 1)
            throw std::runtime_error("mask_tail_f16: vocab must be positive");
        if (limit < 0 || limit > vocab)
            throw std::runtime_error("mask_tail_f16: limit out of range");
        if (limit >= vocab)
        {
            berth_sample_set_launch("mask_tail_f16 skipped");
            return 0;
        }
        int n = vocab - limit;
        int blocks = (n + kThreads - 1) / kThreads;
        cuda_called = true;
        mask_tail_f16_kernel<<<blocks, kThreads, 0, (cudaStream_t) stream>>>(
            (half*) logits, limit, vocab);
        berth_cuda_ok(cudaGetLastError(), "mask_tail_f16 launch");
        berth_sample_set_launch(
            "mask_tail_f16 grid=" + std::to_string(blocks) + " block=" + std::to_string(kThreads));
        return 0;
    }
    catch (const std::exception& err)
    {
        berth_sample_set_error(err.what());
        if (cuda_called)
            cudaGetLastError();
        return -1;
    }
}
