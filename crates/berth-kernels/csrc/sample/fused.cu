// Fused greedy / Gumbel argmax.
//
// The device code (anonymous namespace, through fs_finalize_kernel) is
// extracted from exllamav3_ext/generator/sampling_fused.cu. That file includes
// ATen, and sampling_fused.cuh does too, so the FUSED_SAMPLER_* macros are
// repeated here. Modes 0 and 1 are what the serve launches. Modes 2 and 3
// stay in the extract so the histogram kernels still compile; this launcher
// does not instantiate them.
//
// num_blocks = MIN(CEIL_DIVIDE(size, FS_THREADS * 4), FUSED_SAMPLER_MAX_BLOCKS)
// with FS_THREADS 256. The ValIdx workspace starts at bsz * MAX_BLOCKS floats
// (128, even, so an 8-byte slot stays aligned when the base is aligned).

#include "check.cuh"
#include "sample/sample_state.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <climits>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#define NEG_INF_F16 __ushort_as_half(0xFC00)
#include "reduction.cuh"

#define CEIL_DIVIDE(x, size) (((x) + (size) - 1) / (size))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

#define FUSED_SAMPLER_MAX_BLOCKS 128
#define FUSED_SAMPLER_HIST_BUCKETS 1024
#define FUSED_SAMPLER_HIST_STRIDE (2 * FUSED_SAMPLER_HIST_BUCKETS * 12 + 64)
#define FUSED_SAMPLER_F_TOPK 1
#define FUSED_SAMPLER_F_TOPP 2
#define FUSED_SAMPLER_F_MINP 4

#include "fork/sampling_fused_kernels.cuh"

static_assert(sizeof(ValIdx) == 8, "ValIdx must be 8 bytes");

extern "C" int berth_sample_fused(
    const void* logits,
    void* out_token,
    void* workspace,
    int dim,
    int size,
    int input_fp16,
    float inv_temp,
    unsigned int random,
    int mode,
    void* stream)
{
    bool cuda_called = false;
    try
    {
        berth_sample_clear();
        if (!logits || !out_token || !workspace)
            throw std::runtime_error("fused_sampler: null device pointer");
        if (dim < 1)
            throw std::runtime_error("fused_sampler: dim must be positive");
        if (size < 1 || size > dim)
            throw std::runtime_error("fused_sampler: invalid size bound");
        if (input_fp16 != 0 && input_fp16 != 1)
            throw std::runtime_error("fused_sampler: input_fp16 must be 0 or 1");
        if (mode != 0 && mode != 1)
            throw std::runtime_error("fused_sampler: mode must be 0 or 1");

        int num_blocks = MIN(CEIL_DIVIDE(size, FS_THREADS * 4), FUSED_SAMPLER_MAX_BLOCKS);
        float* ws_max = (float*) workspace;
        ValIdx* ws_vi = (ValIdx*) (ws_max + FUSED_SAMPLER_MAX_BLOCKS);
        dim3 grid(num_blocks, 1);
        cudaStream_t cu = (cudaStream_t) stream;
        const half* mask = nullptr;
        const uint32_t* bits = nullptr;
        const uint8_t* hist = nullptr;
        const float minp_log = 0.0f;
        const int filters = 0;
        const float inv_temp_filter = 1.0f;
        const bool mask_per_row = false;
        const int bits_words = 0;

        cuda_called = true;
        if (input_fp16)
        {
            if (mode == 0)
                fs_partial_argmax_kernel<half, 0><<<grid, FS_THREADS, 0, cu>>>(
                    (const half*) logits, mask, bits, ws_max, ws_vi, hist,
                    dim, size, num_blocks, mask_per_row, bits_words,
                    inv_temp, minp_log, filters, inv_temp_filter, random);
            else
                fs_partial_argmax_kernel<half, 1><<<grid, FS_THREADS, 0, cu>>>(
                    (const half*) logits, mask, bits, ws_max, ws_vi, hist,
                    dim, size, num_blocks, mask_per_row, bits_words,
                    inv_temp, minp_log, filters, inv_temp_filter, random);
        }
        else
        {
            if (mode == 0)
                fs_partial_argmax_kernel<float, 0><<<grid, FS_THREADS, 0, cu>>>(
                    (const float*) logits, mask, bits, ws_max, ws_vi, hist,
                    dim, size, num_blocks, mask_per_row, bits_words,
                    inv_temp, minp_log, filters, inv_temp_filter, random);
            else
                fs_partial_argmax_kernel<float, 1><<<grid, FS_THREADS, 0, cu>>>(
                    (const float*) logits, mask, bits, ws_max, ws_vi, hist,
                    dim, size, num_blocks, mask_per_row, bits_words,
                    inv_temp, minp_log, filters, inv_temp_filter, random);
        }
        berth_cuda_ok(cudaGetLastError(), "fs_partial_argmax launch");
        fs_finalize_kernel<<<1, FUSED_SAMPLER_MAX_BLOCKS, 0, cu>>>(
            ws_vi,
            (uint64_t*) out_token,
            num_blocks);
        berth_cuda_ok(cudaGetLastError(), "fs_finalize launch");

        const char* dtype = input_fp16 ? "half" : "float";
        berth_sample_set_launch(
            std::string("fs_partial_argmax_kernel<") + dtype + "," + std::to_string(mode) + ">"
            + " grid=(" + std::to_string(num_blocks) + ",1)"
            + " block=" + std::to_string(FS_THREADS)
            + "; fs_finalize_kernel grid=1 block=" + std::to_string(FUSED_SAMPLER_MAX_BLOCKS));
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
