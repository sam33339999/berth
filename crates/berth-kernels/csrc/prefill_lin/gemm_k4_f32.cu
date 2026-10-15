// Cooperative GEMM for prefill rows: K = 4, mul1 (cb 2), fp32 output.
// Down, o_proj, and out_proj. Same six shapes as the fp16 unit.

#include "prefill_lin/api.h"

#include <cuda_fp16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include "util.h"
#include "util.cuh"
#include "ptx.cuh"
#include "exl3_gemm_kernel.cuh"

void* berth_prefill_gemm_k4_mul1_f32(int shape_idx)
{
    switch (shape_idx)
    {
        case 1: return (void*) exl3_gemm_kernel<4, true, 2, EXL3_GEMM_SHAPE_1>;
        case 2: return (void*) exl3_gemm_kernel<4, true, 2, EXL3_GEMM_SHAPE_2>;
        case 3: return (void*) exl3_gemm_kernel<4, true, 2, EXL3_GEMM_SHAPE_3>;
        case 4: return (void*) exl3_gemm_kernel<4, true, 2, EXL3_GEMM_SHAPE_4>;
        case 5: return (void*) exl3_gemm_kernel<4, true, 2, EXL3_GEMM_SHAPE_5>;
        case 6: return (void*) exl3_gemm_kernel<4, true, 2, EXL3_GEMM_SHAPE_6>;
        default: return nullptr;
    }
}
