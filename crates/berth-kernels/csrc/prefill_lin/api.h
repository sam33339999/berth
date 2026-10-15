#pragma once

// Host launches for the prefill linear. Definitions live in this directory.
// Exceptions cross translation units the same way int8_sq.cu does.

#include <cuda_runtime.h>

#include <stddef.h>

void* berth_prefill_gemm_k4_mul1_f16(int shape_idx);
void* berth_prefill_gemm_k4_mul1_f32(int shape_idx);

// Pure shape heuristic from quant/exl3_gemv.cu exl3_gemv_cfg. No CUDA.
// -1 ineligible, 0 narrow, 1 wide. narrow_coresident is occupancy * SMs.
// C linkage: the Rust test calls this without a CUDA context.
extern "C" int berth_exl3_gemv_cfg(
    int cc,
    int size_m,
    int size_k,
    int size_n,
    int k_bits,
    int cb,
    int mode,
    int narrow_coresident);

// 1 = launched exl3_gemv_kernel<4, fp32, 2, 1, 0, false>, 0 = declined so
// the caller can fall through to the coop GEMM. Throws on a CUDA error.
// Only the narrow m > 1 K=4 mul1 instances are linked. out_grid is set
// only on success.
extern "C" int berth_prefill_try_qtip(
    const void* a,
    const void* trellis,
    void* c,
    int size_m,
    int size_k,
    int size_n,
    int k_bits,
    int cb,
    int out_fp32,
    const void* suh,
    void* a_had,
    const void* svh,
    cudaStream_t stream,
    int* out_grid);

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
