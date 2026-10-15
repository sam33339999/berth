// Reconstruct and Hadamard launches for rows > 144.
// Device code is the fork extract (quant/reconstruct.cu, quant/hadamard.cu).
// Only K = 4, mul1 (cb 2) is instantiated: cbi = (4 - 1) + 16 = 19.

#include "prefill_lin/api.h"
#include "check.cuh"

#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

#include "util.h"
#include "util.cuh"
#include "ptx.cuh"
#include "exl3_dq.cuh"
#include "hadamard_inner.cuh"
#include "fork/prefill_reconstruct.cuh"
#include "fork/prefill_hadamard.cuh"

namespace {

void check_div(int value, int by, const char* what)
{
    if (value % by != 0)
        throw std::runtime_error(std::string("prefill reconstruct: ") + what + " must be a multiple of " + std::to_string(by));
}

}  // namespace

void berth_prefill_had_r_128(
    const void* input,
    void* output,
    const void* scale,
    int pre_scale,
    int rows,
    int cols,
    int fp32,
    cudaStream_t stream)
{
    check_div(cols, 128, "had columns");
    if (rows < 1) throw std::runtime_error("prefill had_r_128: rows must be positive");
    // hadamard.cu had_r_128: scale / sqrt(128). The call sites pass scale = 1.
    float r_scale = 1.0f * 0.088388347648f;
    dim3 block(32);
    dim3 grid(rows, cols / 128);
    const half* scale_ptr = (const half*) scale;
    if (!fp32)
    {
        const half* in = (const half*) input;
        half* out = (half*) output;
        if (pre_scale)
            had_hf_r_128_kernel<true, false><<<grid, block, 0, stream>>>(in, out, scale_ptr, r_scale);
        else
            had_hf_r_128_kernel<false, true><<<grid, block, 0, stream>>>(in, out, scale_ptr, r_scale);
    }
    else
    {
        const float* in = (const float*) input;
        float* out = (float*) output;
        if (pre_scale)
            had_ff_r_128_kernel<true, false><<<grid, block, 0, stream>>>(in, out, scale_ptr, r_scale);
        else
            had_ff_r_128_kernel<false, true><<<grid, block, 0, stream>>>(in, out, scale_ptr, r_scale);
    }
    berth_cuda_ok(cudaPeekAtLastError(), "prefill had_r_128");
}

void berth_prefill_reconstruct(
    void* unpacked,
    const void* packed,
    int size_k,
    int slice_n,
    int full_n,
    int n_offset,
    cudaStream_t stream)
{
    check_div(size_k, 16, "K");
    check_div(slice_n, 128, "slice N");
    check_div(full_n, 16, "full N");
    check_div(n_offset, 128, "n_offset");
    if (n_offset < 0 || n_offset + slice_n > full_n)
        throw std::runtime_error("prefill reconstruct: slice exceeds the trellis");
    int rows = size_k / 16;
    int packed_cols = full_n / 16;
    dim3 block(256);
    dim3 grid(slice_n / 128, rows);
    reconstruct_kernel<4, 2><<<grid, block, 0, stream>>>(
        (half*) unpacked,
        (const uint16_t*) packed,
        packed_cols,
        n_offset / 16);
    berth_cuda_ok(cudaPeekAtLastError(), "prefill reconstruct_kernel");
}

void berth_prefill_reconstruct_had(
    void* unpacked,
    const void* packed,
    const void* suh,
    const void* svh,
    int size_k,
    int slice_n,
    int full_n,
    int n_offset,
    cudaStream_t stream)
{
    check_div(size_k, 128, "K");
    check_div(slice_n, 128, "slice N");
    check_div(full_n, 16, "full N");
    check_div(n_offset, 128, "n_offset");
    if (n_offset < 0 || n_offset + slice_n > full_n)
        throw std::runtime_error("prefill reconstruct_had: slice exceeds the trellis");
    dim3 grid(slice_n / 128, size_k / 128);
    const half* svh_slice = (const half*) svh + n_offset;
    reconstruct_had_kernel<4, 2><<<grid, RH_THREADS, 0, stream>>>(
        (half*) unpacked,
        (const uint16_t*) packed,
        (const half*) suh,
        svh_slice,
        full_n / 16,
        n_offset / 16);
    berth_cuda_ok(cudaPeekAtLastError(), "prefill reconstruct_had_kernel");
}
