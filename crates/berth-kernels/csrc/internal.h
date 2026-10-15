#pragma once

#include <stddef.h>

// Shared by the launch translation units. Not part of the Rust ABI.

#ifdef __cplusplus
extern "C" {
#endif

int berth_device_cc(int device);
int berth_device_sms(int device);
int* berth_device_locks(int device);

// 1 = launched `exl3_gemv_int8_sq_kernel`, 0 = this shape is not one of the
// linked instances or the sq geometry declined. Throws std::runtime_error on
// a CUDA failure. `out_grid` / `out_smem` are set only on success.
int berth_exl3_try_int8_sq(
    const void* a,
    const void* trellis,
    void* c,
    int size_m,
    int size_k,
    int size_n,
    int k_bits,
    int out_fp32,
    const void* suh,
    void* a_had,
    const void* svh,
    void* stream,
    int* out_grid,
    size_t* out_smem);

// `exl3_gemm_kernel<6, false, 2, EXL3_GEMM_SHAPE_idx>` or null.
void* berth_exl3_gemm_k6_mul1_fp16(int shape_idx);

#ifdef __cplusplus
}
#endif
