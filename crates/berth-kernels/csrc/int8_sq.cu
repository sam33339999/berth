// int8 GEMV sq launcher. Mirrors exl3_gemv_int8_sq in
// quant/exl3_gemv_int8.cu (the m == 1 path profiled as
// exl3_gemv_int8_sq_kernel<4,1,{0,1},0>). Includes the kernel header only;
// the fork's exl3_gemv_int8.cuh pulls in ATen and is not included.

#include "internal.h"
#include "check.cuh"

#include <cuda_fp16.h>

#include "exl3_devctx.cuh"
#include "exl3_gemv_int8_kernel.cuh"

#include <map>
#include <set>
#include <utility>

namespace {

// 16 MiB, WORKSPACE_SIZE / sizeof(int). Fixed for the life of the process:
// the pointer is a kernel argument, and the counters at the front of the
// buffer are self-resetting, so it is zeroed once and never reallocated.
int* g_ws[MAX_DEVICES] = {};
std::set<void*> g_attr[MAX_DEVICES];
std::map<std::pair<void*, size_t>, int> g_occ[MAX_DEVICES];

void* sq_kernel(int k_bits, int rows, bool out_fp32)
{
    // Linked instances are the 27B decode pair: K = 4, one row, plain int8
    // (EXL3_INT8_GEMV=2, residual off). fp16 out is gate/up; fp32 out is
    // down / o_proj.
    if (k_bits != 4 || rows != 1) return nullptr;
    if (out_fp32) return (void*) exl3_gemv_int8_sq_kernel<4, 1, true, false>;
    return (void*) exl3_gemv_int8_sq_kernel<4, 1, false, false>;
}

int* workspace(int device, size_t ints_needed)
{
    constexpr size_t kCap = (16 * 1024 * 1024) / sizeof(int);
    if (ints_needed > kCap) return nullptr;
    if (!g_ws[device])
    {
        berth_cuda_ok(cudaMalloc(&g_ws[device], kCap * sizeof(int)), "int8 GEMV workspace");
        berth_cuda_ok(cudaMemset(g_ws[device], 0, kCap * sizeof(int)), "int8 GEMV workspace memset");
    }
    return g_ws[device];
}

}  // namespace

extern "C" int berth_exl3_try_int8_sq(
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
    size_t* out_smem)
{
    // Same gate as exl3_gemv_int8_sq: m == 1 in plain mode (m == 2 is also
    // accepted by the fork, but only the m == 1 instances are linked).
    if (size_m != 1) return 0;
    const bool residual = false;
    const int rows = size_m;
    void* fn = sq_kernel(k_bits, rows, out_fp32 != 0);
    if (!fn) return 0;

    int device = 0;
    berth_cuda_ok(cudaGetDevice(&device), "cudaGetDevice");
    if (device < 0 || device >= MAX_DEVICES)
        throw std::runtime_error("int8 GEMV: device index out of range");
    const int num_sms = berth_device_sms(device);
    const int rows_max = gemv_int8_sq_rows_max(rows, residual);

    // Mirror of the kernel's work decomposition (single-wave rule, half-wave floor).
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
        size_t stage = gemv_int8_stage_smem(k_bits) ? (size_t) 8 * GEMV_STAGE_D * 16 * k_bits * 4 : 0;
        return (size_t) rows_per * 16 * 2
            + (size_t) rows_per * 16 * 4 * rows * (residual ? 2 : 1)
            + stage
            + (size_t) 2 * rows * 128 * 4;
    };

    if (g_attr[device].find(fn) == g_attr[device].end())
    {
        berth_cuda_ok(
            cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_for(rows_max)),
            "int8 GEMV max dynamic smem");
        // Same carveout as the tensor-core kernels, so graph replay does not
        // reconfigure the SMs between the two families.
        berth_cuda_ok(
            cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared),
            "int8 GEMV smem carveout");
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
            "int8 GEMV occupancy");
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
    // The kernel's `size_m` argument is the real row count, not the template M.
    int size_m_arg = size_m;

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

    cudaError_t err = cudaLaunchKernel(fn, dim3(grid), dim3(NUM_THREADS), args, smem, (cudaStream_t) stream);
    if (err != cudaSuccess)
    {
        cudaGetLastError();
        throw std::runtime_error(std::string("exl3_gemv_int8_sq launch: ") + cudaGetErrorString(err));
    }
    if (out_grid) *out_grid = grid;
    if (out_smem) *out_smem = smem;
    return 1;
}
