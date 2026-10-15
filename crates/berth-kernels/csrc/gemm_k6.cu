// Cooperative GEMM instances for lm_head: K = 6, mul1 codebook (cb 2), fp16
// output. exl3_gemm autotune may pick any compatible shape; the on-disk cache
// for this 27B (m = 1, K = 5120, N = 248320) selects shape 4.
//
// util.cuh (half4) and `namespace cg` have to be visible before the kernel
// header. That header does not include them itself; the fork's comp-unit
// .cu files include them first. quant/exl3_gemm.cuh is the ATen launcher and
// is not included.

#include "internal.h"

#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cooperative_groups.h>
#include <cstdint>

namespace cg = cooperative_groups;

#include "util.h"
#include "util.cuh"
#include "ptx.cuh"
#include "exl3_gemm_kernel.cuh"

extern "C" void* berth_exl3_gemm_k6_mul1_fp16(int shape_idx)
{
    switch (shape_idx)
    {
        case 1: return (void*) exl3_gemm_kernel<6, false, 2, EXL3_GEMM_SHAPE_1>;
        case 2: return (void*) exl3_gemm_kernel<6, false, 2, EXL3_GEMM_SHAPE_2>;
        case 3: return (void*) exl3_gemm_kernel<6, false, 2, EXL3_GEMM_SHAPE_3>;
        case 4: return (void*) exl3_gemm_kernel<6, false, 2, EXL3_GEMM_SHAPE_4>;
        default: return nullptr;
    }
}
