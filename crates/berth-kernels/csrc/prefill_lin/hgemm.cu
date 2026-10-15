// Reconstruct-path GEMM: f16acc::gemm_kernel where hgemm_f16acc_try accepts
// the shape, otherwise cublasGemmEx.
//
// The extract opens `namespace f16acc` and does not close it (the host half
// of hgemm_f16acc.cu includes ATen). This file closes the namespace, then
// reimplements enabled / covered / worthwhile / launch without ATen.
// Ampere (this 3090) is not tuned_device (major == 12), so the launch is
// gemm_kernel<OUT_F32, 128, false>. The Blackwell tuned layouts are not
// instantiated; a shape the fork would send there fails closed.
//
// cuBLAS pin, matching hgemm.cu hgemm_gemmex_impl and the handle
// getCurrentCUDABlasHandle leaves behind on torch 2.13:
//   * libcublas.so.13 from the torch site-packages (see build.rs). The
//     toolkit lib64 also has that soname, so the binary's first RUNPATH
//     entry is OUT_DIR/cublas-torch (symlinks to torch's libcublas.so.13
//     and libcublasLt.so.13 only). libcudart.so.13 is resolved from the
//     toolkit entry after it.
//   * cublasGemmEx, CUBLAS_OP_N/N, swapped operands (row-major),
//     CUDA_R_16F inputs, lda = N and K, ldc = c's row stride,
//     alpha = 1.f, beta = 0.f, CUBLAS_COMPUTE_32F,
//     CUBLAS_GEMM_DEFAULT_TENSOR_OP.
//   * Pointer mode HOST. Workspace WORKSPACE_SIZE (16 MiB), the size
//     DevCtx::get_ws hands the fork. The buffer is per device and is not
//     reused for anything else.
//   * Math mode CUBLAS_DEFAULT_MATH. Torch 2.13's default fp32 matmul
//     precision is "highest", which sets allow_tf32 false, and the handle
//     pool then selects CUBLAS_DEFAULT_MATH. hgemm does not call
//     cublasSetMathMode and does not set
//     CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION (that flag is inside
//     torch's own gemm, not the fork's). The serve does not flip allow_tf32.

#include "prefill_lin/api.h"
#include "check.cuh"
#include "internal.h"

#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

#include "exl3_devctx.cuh"
#include "fork/prefill_f16acc.cuh"

}  // namespace f16acc

namespace {

void cublas_ok(cublasStatus_t status, const char* what)
{
    if (status == CUBLAS_STATUS_SUCCESS) return;
    throw std::runtime_error(std::string(what) + ": cublas status " + std::to_string((int) status));
}

struct BlasSlot {
    cublasHandle_t handle = nullptr;
    void* workspace = nullptr;
};

BlasSlot g_blas[MAX_DEVICES];
std::mutex g_blas_mu;

int g_enabled[MAX_DEVICES];
bool g_enabled_init = false;
std::mutex g_enabled_mu;

cublasHandle_t blas_handle(int device, cudaStream_t stream)
{
    std::lock_guard<std::mutex> lock(g_blas_mu);
    if (device < 0 || device >= MAX_DEVICES)
        throw std::runtime_error("prefill hgemm: device index out of range");
    BlasSlot& slot = g_blas[device];
    if (!slot.handle)
    {
        cublas_ok(cublasCreate(&slot.handle), "cublasCreate");
        cublas_ok(cublasSetMathMode(slot.handle, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
        berth_cuda_ok(cudaMalloc(&slot.workspace, WORKSPACE_SIZE), "cublas workspace");
    }
    cublas_ok(cublasSetStream(slot.handle, stream), "cublasSetStream");
    cublas_ok(cublasSetPointerMode(slot.handle, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
    cublas_ok(cublasSetWorkspace(slot.handle, slot.workspace, WORKSPACE_SIZE), "cublasSetWorkspace");
    return slot.handle;
}

float probe_ms(bool f16, int blocks, int iters, float* sink, cudaStream_t stream)
{
    auto run = [&]() {
        if (f16) f16acc::rate_kernel<true><<<blocks, 256, 0, stream>>>(iters, sink);
        else f16acc::rate_kernel<false><<<blocks, 256, 0, stream>>>(iters, sink);
    };
    run();
    cudaEvent_t e0, e1;
    berth_cuda_ok(cudaEventCreate(&e0), "f16acc probe event");
    berth_cuda_ok(cudaEventCreate(&e1), "f16acc probe event");
    berth_cuda_ok(cudaEventRecord(e0, stream), "f16acc probe record");
    for (int r = 0; r < 3; ++r) run();
    berth_cuda_ok(cudaEventRecord(e1, stream), "f16acc probe record");
    berth_cuda_ok(cudaEventSynchronize(e1), "f16acc probe sync");
    float ms = 0.f;
    berth_cuda_ok(cudaEventElapsedTime(&ms, e0, e1), "f16acc probe elapsed");
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    berth_cuda_ok(cudaGetLastError(), "f16acc probe");
    return ms;
}

bool f16acc_enabled(int device, cudaStream_t stream)
{
    std::lock_guard<std::mutex> lock(g_enabled_mu);
    if (!g_enabled_init)
    {
        for (int i = 0; i < MAX_DEVICES; ++i) g_enabled[i] = -1;
        g_enabled_init = true;
    }
    if (device < 0 || device >= MAX_DEVICES) return false;
    if (g_enabled[device] >= 0) return g_enabled[device] == 1;

    int on = 0;
    const char* env = std::getenv("EXL3_HGEMM_F16ACC");
    int major = 0;
    berth_cuda_ok(
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device),
        "compute capability");
    if (major < 8)
        on = 0;
    else if (env && std::strcmp(env, "0") == 0)
        on = 0;
    else if (env && std::strcmp(env, "1") == 0)
        on = 1;
    else
    {
        int sms = berth_device_sms(device);
        float* sink = nullptr;
        berth_cuda_ok(cudaMalloc(&sink, 256 * sizeof(float)), "f16acc probe sink");
        float t32 = probe_ms(false, sms * 4, 512, sink, stream);
        float t16 = probe_ms(true, sms * 4, 512, sink, stream);
        cudaFree(sink);
        on = (t32 > 0.f && t16 > 0.f && t32 / t16 >= 1.5f) ? 1 : 0;
    }
    g_enabled[device] = on;
    return on == 1;
}

bool aligned(const void* ptr, uintptr_t bytes)
{
    return ((uintptr_t) ptr & (bytes - 1)) == 0;
}

// Shape gate from f16acc::covered, for one unbatched matrix.
bool f16acc_covered(const void* a, const void* b, const void* c, int m, int n, int k, int ldc, int out_fp32, int major)
{
    if (major < 8) return false;
    if (m < 1 || n < 1 || k < 1) return false;
    if (k % f16acc::BK != 0 || n % f16acc::BN != 0) return false;
    if ((m + f16acc::BM - 1) / f16acc::BM > 65535) return false;
    if (ldc < n || (ldc % 2) != 0) return false;
    if (!aligned(a, 16) || !aligned(b, 16)) return false;
    if (!aligned(c, out_fp32 ? 8 : 4)) return false;
    return true;
}

// Non-Blackwell branch of f16acc::worthwhile. M < 384 is cuBLAS. Otherwise
// the grid must cover the SM count. Batch is 1 on this path.
bool f16acc_worthwhile(int m, int n, int sms)
{
    if (m < f16acc::MIN_ROWS) return false;
    long long blocks = (long long) ((m + f16acc::BM - 1) / f16acc::BM) * (n / f16acc::BN);
    return blocks >= sms;
}

// Fork's tuned_device && worthwhile branch. True when the fork would launch
// a TUNED=true kernel. Those instances are not linked.
bool f16acc_would_tune(int m, int n, int k, int sms, int major)
{
    if (major != 12) return false;
    if (!(m >= 64 && m >= f16acc::MIN_ROWS && n >= 1024 && k >= 512)) return false;
    // f16acc::narrow_tile, batch 1.
    bool narrow = true;
    if (!(n <= 1024 || m < 128))
    {
        if (n > 4096) narrow = false;
        else narrow = m < 512 || k > n || (m >= 1024 && m < 4096);
    }
    int tile_n = narrow ? 64 : 128;
    long long blocks = (long long) ((m + f16acc::BM - 1) / f16acc::BM) * (n / tile_n);
    return blocks * 2 >= sms;
}

template <bool OUT_F32>
void launch_f16acc(const half* a, const half* b, void* c, int m, int n, int k, int ldc, cudaStream_t stream)
{
    using CF = f16acc::Config<128, false>;
    auto kern = f16acc::gemm_kernel<OUT_F32, 128, false>;
    static std::once_flag once[MAX_DEVICES];
    int device = 0;
    berth_cuda_ok(cudaGetDevice(&device), "cudaGetDevice");
    if (device < 0 || device >= MAX_DEVICES)
        throw std::runtime_error("prefill f16acc: device index out of range");
    std::call_once(once[device], [&]() {
        berth_cuda_ok(
            cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) CF::SMEM_BYTES),
            "f16acc max dynamic smem");
    });
    dim3 grid(n / CF::BN, (m + CF::BM - 1) / CF::BM, 1);
    kern<<<grid, CF::THREADS, CF::SMEM_BYTES, stream>>>(a, b, c, m, n, k, ldc, 0, 0, 0);
    berth_cuda_ok(cudaPeekAtLastError(), "f16acc gemm_kernel");
}

void launch_cublas(const half* a, const half* b, void* c, int m, int n, int k, int ldc, int out_fp32, cudaStream_t stream)
{
    int device = 0;
    berth_cuda_ok(cudaGetDevice(&device), "cudaGetDevice");
    cublasHandle_t handle = blas_handle(device, stream);
    float alpha = 1.0f;
    float beta = 0.0f;
    cudaDataType_t c_type = out_fp32 ? CUDA_R_32F : CUDA_R_16F;
    cublas_ok(
        cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, k,
            &alpha, b, CUDA_R_16F, n,
                    a, CUDA_R_16F, k,
            &beta,  c, c_type, ldc,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "cublasGemmEx");
    berth_cuda_ok(cudaPeekAtLastError(), "cublasGemmEx");
}

}  // namespace

int berth_prefill_hgemm_recon(
    const void* a,
    const void* b,
    void* c,
    int size_m,
    int size_k,
    int size_n,
    int ldc,
    int out_fp32,
    cudaStream_t stream)
{
    int device = 0;
    berth_cuda_ok(cudaGetDevice(&device), "cudaGetDevice");
    int major = 0;
    berth_cuda_ok(
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device),
        "compute capability");
    int sms = berth_device_sms(device);
    bool covered = f16acc_covered(a, b, c, size_m, size_n, size_k, ldc, out_fp32, major);
    bool worthwhile = covered && f16acc_worthwhile(size_m, size_n, sms);
    if (covered && f16acc_would_tune(size_m, size_n, size_k, sms, major))
        throw std::runtime_error("prefill hgemm: Blackwell tuned f16acc layout is not linked");
    if (worthwhile && f16acc_enabled(device, stream))
    {
        if (out_fp32)
            launch_f16acc<true>((const half*) a, (const half*) b, c, size_m, size_n, size_k, ldc, stream);
        else
            launch_f16acc<false>((const half*) a, (const half*) b, c, size_m, size_n, size_k, ldc, stream);
        return 1;
    }
    launch_cublas((const half*) a, (const half*) b, c, size_m, size_n, size_k, ldc, out_fp32, stream);
    return 0;
}
