#pragma once

// Host launches for the prefill linear. Definitions live in this directory.
// Exceptions cross translation units the same way int8_sq.cu does.

#include <cuda_runtime.h>

#include <stddef.h>

void* berth_prefill_gemm_k4_mul1_f16(int shape_idx);
void* berth_prefill_gemm_k4_mul1_f32(int shape_idx);

// 1 = launched exl3_gemv_int8_sq_kernel<4, 2, ...>, 0 = declined. Throws on
// a CUDA error. out_grid / out_smem are set only on success.
int berth_prefill_try_int8_m2(
    const void* a,
    const void* trellis,
    void* c,
    int size_k,
    int size_n,
    int out_fp32,
    const void* suh,
    void* a_had,
    const void* svh,
    cudaStream_t stream,
    int* out_grid,
    size_t* out_smem);

// had_r_128. pre_scale != 0 selects the input sign vector; otherwise the
// output sign vector. fp32 selects had_ff (fp32 tensor) over had_hf.
void berth_prefill_had_r_128(
    const void* input,
    void* output,
    const void* scale,
    int pre_scale,
    int rows,
    int cols,
    int fp32,
    cudaStream_t stream);

// Plain reconstruct of a column slice. unpacked is (size_k, slice_n) fp16.
// packed is the full trellis. n_offset is a column in the unpacked weight.
void berth_prefill_reconstruct(
    void* unpacked,
    const void* packed,
    int size_k,
    int slice_n,
    int full_n,
    int n_offset,
    cudaStream_t stream);

// Fused suh/H/W/H/svh reconstruct. svh is the base vector; the slice reads
// svh + n_offset.
void berth_prefill_reconstruct_had(
    void* unpacked,
    const void* packed,
    const void* suh,
    const void* svh,
    int size_k,
    int slice_n,
    int full_n,
    int n_offset,
    cudaStream_t stream);

// hgemm_recon. Returns 1 when f16acc::gemm_kernel ran, 0 when cublasGemmEx ran.
// c's row stride is ldc (elements). N is b's column count, which can be a slice.
int berth_prefill_hgemm_recon(
    const void* a,
    const void* b,
    void* c,
    int size_m,
    int size_k,
    int size_n,
    int ldc,
    int out_fp32,
    cudaStream_t stream);
