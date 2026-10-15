// Repetition and presence/frequency penalties.
//
// Device kernels are apply_rep_pens_kernel and apply_pres_freq_pens_kernel
// from exllamav3_ext/generator/rep_pen.cu. That file includes ATen before the
// kernels, so build.rs extracts them into fork/rep_pen_kernels.cuh. The grid
// matches the fork's host launchers: CEIL_DIVIDE(vocab, BLOCK_VOCAB_SPAN)
// blocks of NUM_THREADS (1024). bsz is 1. Output is always fp32. A half input
// instantiates <true>, a float input <false>.

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

#include "fork/rep_pen_kernels.cuh"

namespace {

void check_pen(const char* what, const void* in_logits, void* out_logits, const void* past_ids, int past_len, int vocab)
{
    if (!in_logits || !out_logits || !past_ids)
        throw std::runtime_error(std::string(what) + ": null device pointer");
    if (vocab < 1)
        throw std::runtime_error(std::string(what) + ": vocab must be positive");
    if (past_len < 0)
        throw std::runtime_error(std::string(what) + ": past_len must be non-negative");
}

}  // namespace

extern "C" int berth_sample_apply_rep_pens(
    const void* in_logits,
    void* out_logits,
    const void* past_ids,
    int past_len,
    int vocab,
    int input_fp16,
    float rep_p,
    int sustain_range,
    int decay_range,
    void* stream)
{
    bool cuda_called = false;
    try
    {
        berth_sample_clear();
        check_pen("apply_rep_pens", in_logits, out_logits, past_ids, past_len, vocab);
        if (input_fp16 != 0 && input_fp16 != 1)
            throw std::runtime_error("apply_rep_pens: input_fp16 must be 0 or 1");
        int num_blocks = CEIL_DIVIDE(vocab, BLOCK_VOCAB_SPAN);
        cudaStream_t cu = (cudaStream_t) stream;
        cuda_called = true;
        if (input_fp16)
            apply_rep_pens_kernel<true><<<num_blocks, NUM_THREADS, 0, cu>>>(
                in_logits,
                (float*) out_logits,
                (const uint64_t*) past_ids,
                past_len,
                vocab,
                rep_p,
                sustain_range,
                decay_range);
        else
            apply_rep_pens_kernel<false><<<num_blocks, NUM_THREADS, 0, cu>>>(
                in_logits,
                (float*) out_logits,
                (const uint64_t*) past_ids,
                past_len,
                vocab,
                rep_p,
                sustain_range,
                decay_range);
        berth_cuda_ok(cudaGetLastError(), "apply_rep_pens launch");
        berth_sample_set_launch(
            std::string("apply_rep_pens_kernel<") + (input_fp16 ? "1" : "0") + ">"
            + " grid=" + std::to_string(num_blocks)
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

extern "C" int berth_sample_apply_pres_freq_pens(
    const void* in_logits,
    void* out_logits,
    const void* past_ids,
    int past_len,
    int vocab,
    int input_fp16,
    float pres_p,
    float freq_p,
    int sustain_range,
    int decay_range,
    void* stream)
{
    bool cuda_called = false;
    try
    {
        berth_sample_clear();
        check_pen("apply_pres_freq_pens", in_logits, out_logits, past_ids, past_len, vocab);
        if (input_fp16 != 0 && input_fp16 != 1)
            throw std::runtime_error("apply_pres_freq_pens: input_fp16 must be 0 or 1");
        int num_blocks = CEIL_DIVIDE(vocab, BLOCK_VOCAB_SPAN);
        cudaStream_t cu = (cudaStream_t) stream;
        cuda_called = true;
        if (input_fp16)
            apply_pres_freq_pens_kernel<true><<<num_blocks, NUM_THREADS, 0, cu>>>(
                in_logits,
                (float*) out_logits,
                (const uint64_t*) past_ids,
                past_len,
                vocab,
                pres_p,
                freq_p,
                sustain_range,
                decay_range);
        else
            apply_pres_freq_pens_kernel<false><<<num_blocks, NUM_THREADS, 0, cu>>>(
                in_logits,
                (float*) out_logits,
                (const uint64_t*) past_ids,
                past_len,
                vocab,
                pres_p,
                freq_p,
                sustain_range,
                decay_range);
        berth_cuda_ok(cudaGetLastError(), "apply_pres_freq_pens launch");
        berth_sample_set_launch(
            std::string("apply_pres_freq_pens_kernel<") + (input_fp16 ? "1" : "0") + ">"
            + " grid=" + std::to_string(num_blocks)
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
