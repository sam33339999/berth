// exl3_mgemm_kernel<4, false, 2, shapes 1-4>. Attention fused qkv on the 27B
// (c_fp32 = 0) is the shape-4 instance. Shapes 1-3 stay linked so a coop
// autotune disk hit can resolve any of those tags. Shapes 5 and 6 are null
// in the fork's mgemm tables.
//
// util.cuh and `namespace cg` have to be visible before the kernel header.
// quant/exl3_gemm.cuh is the ATen launcher and is not included.

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

extern "C" void* berth_ops_mgemm_k4_mul1_f16(int shape_idx)
{
    switch (shape_idx)
    {
        case 1: return (void*) exl3_mgemm_kernel<4, false, 2, EXL3_GEMM_SHAPE_1>;
        case 2: return (void*) exl3_mgemm_kernel<4, false, 2, EXL3_GEMM_SHAPE_2>;
        case 3: return (void*) exl3_mgemm_kernel<4, false, 2, EXL3_GEMM_SHAPE_3>;
        case 4: return (void*) exl3_mgemm_kernel<4, false, 2, EXL3_GEMM_SHAPE_4>;
        default: return nullptr;
    }
}
