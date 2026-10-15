// Cooperative GEMM for prefill rows: K = 4, mul1 (cb 2), fp16 output.
// exl3_gemm autotune may pick any compatible shape, including the dense
// multi-row shapes 5 and 6. The 100-token prefill profile selects shape 6
// (64, 32, 128, 4, 3). Shapes 5 and 6 are not in the mgemm tables.

#include "prefill_lin/api.h"

#include <cuda_fp16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include "util.h"
#include "util.cuh"
#include "ptx.cuh"
#include "exl3_gemm_kernel.cuh"

void* berth_prefill_gemm_k4_mul1_f16(int shape_idx)
{
    switch (shape_idx)
    {
        case 1: return (void*) exl3_gemm_kernel<4, false, 2, EXL3_GEMM_SHAPE_1>;
        case 2: return (void*) exl3_gemm_kernel<4, false, 2, EXL3_GEMM_SHAPE_2>;
        case 3: return (void*) exl3_gemm_kernel<4, false, 2, EXL3_GEMM_SHAPE_3>;
        case 4: return (void*) exl3_gemm_kernel<4, false, 2, EXL3_GEMM_SHAPE_4>;
        case 5: return (void*) exl3_gemm_kernel<4, false, 2, EXL3_GEMM_SHAPE_5>;
        case 6: return (void*) exl3_gemm_kernel<4, false, 2, EXL3_GEMM_SHAPE_6>;
        default: return nullptr;
    }
}
