// Gumbel noise kernels from exllamav3_ext/generator/gumbel.cu.
//
// The serve's fused sampler does not launch these; the collapsed tail adds
// the same Philox noise inside fs_partial_argmax_kernel. They are linked so
// a kernel test can match gumbel_noise_f16 / gumbel_noise_f32 on their own.
// f16 blocks are CEIL_DIVIDE(CEIL_DIVIDE(size, 2), NUM_THREADS). f32 blocks
// are CEIL_DIVIDE(size, NUM_THREADS). NUM_THREADS is 1024.

#include "check.cuh"
#include "sample/sample_state.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#define CEIL_DIVIDE(x, size) (((x) + (size) - 1) / (size))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

#include "fork/gumbel_kernels.cuh"

namespace {

void check_gumbel(const char* what, const void* in_logits, void* out_logits, int size)
{
    if (!in_logits || !out_logits)
        throw std::runtime_error(std::string(what) + ": null device pointer");
    if (size < 1)
        throw std::runtime_error(std::string(what) + ": size must be positive");
}

}  // namespace

extern "C" int berth_sample_gumbel_f16(
    const void* in_logits,
    void* out_logits,
    int size,
    unsigned int random,
    void* stream)
{
    bool cuda_called = false;
    try
    {
        berth_sample_clear();
        check_gumbel("gumbel_noise_f16", in_logits, out_logits, size);
        int blocks = CEIL_DIVIDE(CEIL_DIVIDE(size, 2), NUM_THREADS);
        cuda_called = true;
        gumbel_noise_kernel_f16<<<blocks, NUM_THREADS, 0, (cudaStream_t) stream>>>(
            (const half*) in_logits,
            (half*) out_logits,
            size,
            random);
        berth_cuda_ok(cudaGetLastError(), "gumbel_noise_f16 launch");
        berth_sample_set_launch(
            "gumbel_noise_kernel_f16 grid=" + std::to_string(blocks)
            + " block=" + std::to_string(NUM_THREADS));
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

extern "C" int berth_sample_gumbel_f32(
    const void* in_logits,
    void* out_logits,
    int size,
    unsigned int random,
    void* stream)
{
    bool cuda_called = false;
    try
    {
        berth_sample_clear();
        check_gumbel("gumbel_noise_f32", in_logits, out_logits, size);
        int blocks = CEIL_DIVIDE(size, NUM_THREADS);
        cuda_called = true;
        gumbel_noise_kernel_f32<<<blocks, NUM_THREADS, 0, (cudaStream_t) stream>>>(
            (const float*) in_logits,
            (float*) out_logits,
            size,
            random);
        berth_cuda_ok(cudaGetLastError(), "gumbel_noise_f32 launch");
        berth_sample_set_launch(
            "gumbel_noise_kernel_f32 grid=" + std::to_string(blocks)
            + " block=" + std::to_string(NUM_THREADS));
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
