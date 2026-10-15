// DRY penalty.
//
// dry_penalty_kernel is extracted from exllamav3_ext/generator/dry.cu (the
// file includes ATen above the kernel). One int32 scratch buffer holds the
// match-length workspace (vocab) and the row counter (1). The fork fills that
// buffer with -1 before the launch; 0 would penalize every token when
// allowed_length is 0. Grid is (scan_blocks, 1) with
// scan_blocks = MIN(MAX_SCAN_BLOCKS, MAX(1, CEIL_DIVIDE(m, NUM_THREADS))).
// breakers may be null. past_ids is required even when past_len is 0.

#include "check.cuh"
#include "sample/sample_state.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#define CEIL_DIVIDE(x, size) (((x) + (size) - 1) / (size))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

#include "fork/dry_kernel.cuh"

extern "C" int berth_sample_dry_penalty(
    const void* in_logits,
    void* out_logits,
    const void* past_ids,
    const void* breakers,
    void* scratch,
    int past_len,
    int vocab,
    int input_fp16,
    float multiplier,
    float base,
    int allowed_length,
    int range,
    int max_exponent,
    int match_cap,
    void* stream)
{
    bool cuda_called = false;
    try
    {
        berth_sample_clear();
        if (!in_logits || !out_logits || !past_ids || !scratch)
            throw std::runtime_error("dry_penalty: null device pointer");
        if (vocab < 1)
            throw std::runtime_error("dry_penalty: vocab must be positive");
        if (past_len < 0)
            throw std::runtime_error("dry_penalty: past_len must be non-negative");
        if (input_fp16 != 0 && input_fp16 != 1)
            throw std::runtime_error("dry_penalty: input_fp16 must be 0 or 1");

        int m = range > 0 ? MIN(past_len, range) : past_len;
        int scan_blocks = MIN(MAX_SCAN_BLOCKS, MAX(1, CEIL_DIVIDE(m, NUM_THREADS)));
        dim3 grid(scan_blocks, 1);
        cudaStream_t cu = (cudaStream_t) stream;
        int* workspace = (int*) scratch;
        unsigned int* counters = (unsigned int*) (workspace + vocab);
        size_t bytes = (size_t) (vocab + 1) * sizeof(int);

        cuda_called = true;
        berth_cuda_ok(cudaMemsetAsync(scratch, 0xFF, bytes, cu), "dry_penalty memset");
        if (input_fp16)
            dry_penalty_kernel<true><<<grid, NUM_THREADS, 0, cu>>>(
                in_logits,
                (float*) out_logits,
                (const int64_t*) past_ids,
                past_len,
                (const bool*) breakers,
                workspace,
                counters,
                vocab,
                multiplier,
                base,
                allowed_length,
                range,
                max_exponent,
                match_cap);
        else
            dry_penalty_kernel<false><<<grid, NUM_THREADS, 0, cu>>>(
                in_logits,
                (float*) out_logits,
                (const int64_t*) past_ids,
                past_len,
                (const bool*) breakers,
                workspace,
                counters,
                vocab,
                multiplier,
                base,
                allowed_length,
                range,
                max_exponent,
                match_cap);
        berth_cuda_ok(cudaGetLastError(), "dry_penalty launch");
        berth_sample_set_launch(
            std::string("dry_penalty_kernel<") + (input_fp16 ? "1" : "0") + ">"
            + " grid=" + std::to_string(scan_blocks) + "x1"
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
