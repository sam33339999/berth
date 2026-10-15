// M = 2 int8 sq GEMV. exl3_gemv_int8_sq accepts m == 2 in plain mode
// (EXL3_INT8_GEMV=2). The decode shim links only M = 1 and refuses M = 2;
// this unit instantiates the M = 2 templates and does not change that path.
// Launch arithmetic matches exl3_gemv_int8_sq in quant/exl3_gemv_int8.cu.

#include "prefill_lin/api.h"
#include "internal.h"
#include "check.cuh"

#include <cuda_fp16.h>

#include "exl3_devctx.cuh"
#include "exl3_gemv_int8_kernel.cuh"
#include "util.h"

#include <map>
#include <set>
#include <utility>

namespace {

int* g_ws[MAX_DEVICES] = {};
std::set<void*> g_attr[MAX_DEVICES];
std::map<std::pair<void*, size_t>, int> g_occ[MAX_DEVICES];

void* sq_kernel(bool out_fp32)
{
    if (out_fp32) return (void*) exl3_gemv_int8_sq_kernel<4, 2, true, false>;
    return (void*) exl3_gemv_int8_sq_kernel<4, 2, false, false>;
}

int* workspace(int device, size_t ints_needed)
{
    constexpr size_t kCap = (16 * 1024 * 1024) / sizeof(int);
    if (ints_needed > kCap) return nullptr;
    if (!g_ws[device])
    {
        berth_cuda_ok(cudaMalloc(&g_ws[device], kCap * sizeof(int)), "prefill int8 workspace");
        berth_cuda_ok(cudaMemset(g_ws[device], 0, kCap * sizeof(int)), "prefill int8 workspace memset");
    }
    return g_ws[device];
}

}  // namespace

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
    size_t* out_smem)
{
    const bool residual = false;
    const int rows = 2;
    void* fn = sq_kernel(out_fp32 != 0);
    if (!fn) return 0;

    int device = 0;
    berth_cuda_ok(cudaGetDevice(&device), "cudaGetDevice");
    if (device < 0 || device >= MAX_DEVICES)
        throw std::runtime_error("prefill int8: device index out of range");
    const int num_sms = berth_device_sms(device);
    const int rows_max = gemv_int8_sq_rows_max(rows, residual);

    auto decomp = [&](int grid, int& ksplit, int& rows_per)
    {
        int rows_total = size_k / 16;
        int nb256 = size_n / 256;
        int r = CEIL_DIVIDE(rows_total * nb256, grid);
        rows_per = (MAX(r, MIN(2 * r, 32)) + 7) & ~7;
        rows_per = MAX(rows_per, SQ_MINROWS);
        rows_per = MIN(rows_per, rows_max);
        rows_per = MIN(rows_per, (rows_total + 7) & ~7);
        ksplit = CEIL_DIVIDE(rows_total, rows_per);
    };
    auto smem_for = [&](int rows_per) -> size_t
    {
        size_t stage = gemv_int8_stage_smem(4) ? (size_t) 8 * GEMV_STAGE_D * 16 * 4 * 4 : 0;
        return (size_t) rows_per * 16 * 2
            + (size_t) rows_per * 16 * 4 * rows * (residual ? 2 : 1)
            + stage
            + (size_t) 2 * rows * 128 * 4;
    };

    if (g_attr[device].find(fn) == g_attr[device].end())
    {
        berth_cuda_ok(
            cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_for(rows_max)),
            "prefill int8 max dynamic smem");
        berth_cuda_ok(
            cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared),
            "prefill int8 smem carveout");
        g_attr[device].insert(fn);
    }

    int ksplit = 0;
    int rows_per = 0;
    decomp(6 * num_sms, ksplit, rows_per);
    size_t smem_guess = smem_for(rows_per);
    auto occ_key = std::make_pair(fn, smem_guess);
    int maxb = 1;
    auto occ_it = g_occ[device].find(occ_key);
    if (occ_it != g_occ[device].end()) maxb = occ_it->second;
    else
    {
        berth_cuda_ok(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, fn, NUM_THREADS, smem_guess),
            "prefill int8 occupancy");
        g_occ[device][occ_key] = maxb;
    }
    int grid = MIN(MAX(maxb, 1) * num_sms, 1024);
    decomp(grid, ksplit, rows_per);
    size_t smem = smem_for(rows_per);
    if (ksplit > SQ_KSPLIT_CAP) return 0;
    if (size_n / 256 > SQ_COUNTERS_CAP) return 0;

    int pstride = size_n * (residual ? 2 : 1);
    int* ws = workspace(device, SQ_WS_RESERVED + (size_t) ksplit * rows * pstride);
    if (!ws) return 0;

    const half* a_ptr = (const half*) a;
    const uint16_t* b_ptr = (const uint16_t*) trellis;
    void* c_ptr = c;
    const half* suh_ptr = (const half*) suh;
    half* a_had_ptr = (half*) a_had;
    const half* svh_ptr = (const half*) svh;
    int size_m_arg = rows;

    void* args[] = {
        (void*) &a_ptr,
        (void*) &b_ptr,
        (void*) &c_ptr,
        (void*) &size_m_arg,
        (void*) &size_k,
        (void*) &size_n,
        (void*) &ws,
        (void*) &suh_ptr,
        (void*) &a_had_ptr,
        (void*) &svh_ptr,
    };

    cudaError_t err = cudaLaunchKernel(fn, dim3(grid), dim3(NUM_THREADS), args, smem, stream);
    if (err != cudaSuccess)
    {
        cudaGetLastError();
        throw std::runtime_error(std::string("prefill exl3_gemv_int8_sq M=2: ") + cudaGetErrorString(err));
    }
    if (out_grid) *out_grid = grid;
    if (out_smem) *out_smem = smem;
    return 1;
}
