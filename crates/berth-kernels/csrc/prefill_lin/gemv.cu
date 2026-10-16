// QTIP small-m GEMV for prefill rows 3..=8.
//
// The device code is quant/exl3_gemv_kernel.cuh (ATen-free: ptx.cuh,
// exl3_dq.cuh, exl3_kernel_map.cuh, hadamard_inner.cuh). Only the two
// instances the 27B narrow path needs are instantiated:
//   exl3_gemv_kernel<4, false, 2, 1, 0, false>  fp16 out
//   exl3_gemv_kernel<4, true,  2, 1, 0, false>  fp32 out
// K=4, mul1 (cb 2), mmode 1 (m > 1), cfg 0 (narrow), no smem staging.
// cfg 1, other K, other cb, and EXL3_GEMV_SMEM=1 decline so exl3_gemm
// can fall through to the cooperative kernel. Occupancy is measured at
// runtime (block 512, dynamic smem 0) and cached per device. It is not
// hardcoded.

#include "prefill_lin/api.h"
#include "internal.h"
#include "check.cuh"

#include <cuda_fp16.h>
#include <cooperative_groups.h>

#include "util.h"
#include "util.cuh"
#include "ptx.cuh"
#include "exl3_gemv_kernel.cuh"
#include "exl3_devctx.cuh"

#include <cstdlib>
#include <stdexcept>

namespace {

int gemv_env_mode()
{
    const char* env = std::getenv("EXL3_GEMV");
    if (!env) return 1;
    return std::atoi(env);
}

int gemv_env_smem()
{
    const char* env = std::getenv("EXL3_GEMV_SMEM");
    if (!env) return -1;
    return std::atoi(env);
}

void* narrow_kernel(bool fp32)
{
    if (fp32) return (void*) exl3_gemv_kernel<4, true, 2, 1, 0, false>;
    return (void*) exl3_gemv_kernel<4, false, 2, 1, 0, false>;
}

// blocks per SM, block 512, dynamic smem 0. 0 means not cached yet.
int g_occ[MAX_DEVICES][2] = {};

int occupancy(int device, bool fp32)
{
    int& slot = g_occ[device][fp32 ? 1 : 0];
    if (slot > 0) return slot;
    int blocks = 0;
    berth_cuda_ok(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, narrow_kernel(fp32), 512, 0),
        "exl3_gemv occupancy");
    if (blocks < 1)
        throw std::runtime_error("exl3_gemv occupancy is 0");
    slot = blocks;
    return blocks;
}

}  // namespace

// exl3_gemv_cfg (quant/exl3_gemv.cu). Host integer arithmetic only.
extern "C" int berth_exl3_gemv_cfg(
    int cc,
    int size_m,
    int size_k,
    int size_n,
    int K,
    int cb,
    int mode,
    int narrow_coresident)
{
    if (mode == 0) return -1;
    if (K < 2 || K > 4) return -1;
    if (K != 4 && cb == 0) return -1;
    if (size_m > EXL3_GEMV_MAX_M) return -1;
    if (size_k % 128 || size_n % 128) return -1;
    if (mode == 2) return size_n <= 8192 ? 0 : 1;
    if (mode == 3) return 0;
    if (mode == 4) return 1;

    if (K == 2) return size_n <= 8192 ? 0 : 1;
    if (K == 3 && cc == CC_ADA) return size_n <= 8192 ? 0 : 1;
    if (size_n / 32 <= narrow_coresident) return 0;
    if (size_k <= 2048 && size_n <= 8192) return 0;
    if (K == 3) return -1;
    if (size_n >= 8192 && size_k <= 4096) return 1;
    if (size_n >= 8192 && size_n <= 10240 && size_k <= 5120 && cc == CC_AMPERE) return 1;
    return -1;
}

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
    int* out_grid)
{
    // Free checks first. Anything this port cannot launch returns 0 before
    // a CUDA call, including cfg 1 (wide), which is not instantiated.
    if (!suh || !a_had || !svh) return 0;
    if (k_bits != 4 || cb != 2) return 0;
    if (size_m <= 1 || size_m > EXL3_GEMV_MAX_M) return 0;
    if (size_k % 128 || size_n % 128) return 0;
    int mode = gemv_env_mode();
    if (mode == 0) return 0;
    if (gemv_env_smem() == 1) return 0;

    int device = 0;
    berth_cuda_ok(cudaGetDevice(&device), "cudaGetDevice");
    if (device < 0 || device >= MAX_DEVICES)
        throw std::runtime_error("exl3_gemv: device index out of range");
    const bool fp32 = out_fp32 != 0;
    const int cc = berth_device_cc(device);
    const int num_sms = berth_device_sms(device);
    const int occ = occupancy(device, fp32);
    const int narrow_coresident = occ * num_sms;
    const int cfg = berth_exl3_gemv_cfg(
        cc, size_m, size_k, size_n, k_bits, cb, mode, narrow_coresident);
    if (cfg != 0) return 0;

    const int grid = MIN(size_n / 32, occ * num_sms);
    if (grid < 1) return 0;

    int* locks = berth_device_locks(device);
    const void* a_ptr = a;
    const void* b_ptr = trellis;
    void* c_ptr = c;
    const void* suh_ptr = suh;
    void* a_had_ptr = a_had;
    const void* svh_ptr = svh;
    void* kernel_args[] = {
        (void*) &a_ptr,
        (void*) &b_ptr,
        (void*) &c_ptr,
        (void*) &size_m,
        (void*) &size_k,
        (void*) &size_n,
        (void*) &locks,
        (void*) &suh_ptr,
        (void*) &a_had_ptr,
        (void*) &svh_ptr,
    };
    berth_cuda_ok(
        cudaLaunchCooperativeKernel(
            narrow_kernel(fp32), dim3(grid), dim3(512), kernel_args, 0, stream),
        "exl3_gemv");
    if (out_grid) *out_grid = grid;
    return 1;
}
