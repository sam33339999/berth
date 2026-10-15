// Dispatch for one EXL3 linear. Mirrors exl3_gemm_gr's order: int8 GEMV when
// mul1 and EXL3_INT8_GEMV allow it, then the QTIP GEMV gate (not linked), then
// cooperative exl3_gemm with coop autotune. Graph parameter recording is left
// to berth-cuda; the launch arguments match the fork.

#include "internal.h"
#include "check.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unistd.h>
#include <vector>

#include "coop_autotune.cuh"
#include "exl3_devctx.cuh"
#include "exl3_kernel_map.cuh"
#include "util.h"

namespace {

// quant/exl3_gemm_inner.cuh — dynamic smem passed to every cooperative GEMM.
constexpr size_t kGemmSmem = 90 * 1024;

// exl3_gemv_kernel.cuh. Not included: that header is the QTIP device code.
constexpr int kGemvMaxM = 8;

thread_local std::string g_err;
thread_local std::string g_launch;

struct DevSlot {
    int sms = 0;
    int cc = 0;
    int* locks = nullptr;
};

DevSlot g_dev[MAX_DEVICES];
std::mutex g_dev_mu;

// exl3_gemv_int8_mode re-reads the variable; its cache flag is never set.
// An empty string is atoi("") == 0, which disables the path, same as the fork.
int int8_mode()
{
    const char* value = std::getenv("EXL3_INT8_GEMV");
    return value ? std::atoi(value) : 2;
}

int int8_max_k(int device)
{
    // Cached on first use, same as the fork's function-local static.
    static const int from_env = [] {
        const char* value = std::getenv("EXL3_INT8_GEMV_MAX_K");
        return value ? std::atoi(value) : 0;
    }();
    if (from_env) return MIN(from_env, 8);
    int cc = berth_device_cc(device);
    return (cc == CC_HOPPER || cc == CC_BLACKWELL) ? 6 : 5;
}

uint64_t roundup_pow2(uint64_t x)
{
    if (x == 0) return 1;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x + 1;
}

int gemm_m_key(int size_m)
{
    if (size_m <= 16) return (int) MIN(roundup_pow2((uint64_t) size_m), (uint64_t) 16);
    if (size_m <= 31) return 32;
    if (size_m <= 63) return 64;
    if (size_m <= 127) return 128;
    return 512;
}

uint64_t gemm_hash(int size_m, int size_k, int size_n, int k_bits, int out_fp32, int device, int cc, int sms, int cb)
{
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) {
        h ^= v;
        h *= 1099511628211ull;
    };
    mix((uint64_t) gemm_m_key(size_m));
    mix((uint64_t) size_k);
    mix((uint64_t) size_n);
    mix((uint64_t) k_bits);
    mix(out_fp32 ? 1ull : 0ull);
    mix((uint64_t) device);
    mix((uint64_t) cc);
    mix((uint64_t) sms);
    mix((uint64_t) cb);
    return h;
}

const int kTileM[] = {EXL3_GEMM_TILESIZE_M};
const int kTileK[] = {EXL3_GEMM_TILESIZE_K};
const int kTileN[] = {EXL3_GEMM_TILESIZE_N};
const int kBlock[] = {EXL3_GEMM_BLOCKDIM};

bool shape_compat(int shape, int size_m, int size_k, int size_n)
{
    if (kTileM[shape] > 16 && size_m < kTileM[shape]) return false;
    return (size_k % kTileK[shape] == 0) && (size_n % kTileN[shape] == 0);
}

void describe_gemm(const CoopAutotuneLaunch& launch)
{
    int shape = launch.tag;
    g_launch = "exl3_gemm_kernel<6,0,2,"
        + std::to_string(kTileM[shape]) + ","
        + std::to_string(kTileK[shape]) + ","
        + std::to_string(kTileN[shape]) + ","
        + std::to_string(shape == 1 ? 6 : 4) + ","
        + std::to_string(shape == 1 ? 5 : 3) + ">"
        + " block=" + std::to_string(launch.block_dim)
        + " grid=" + std::to_string(launch.num_sms) + "x1x" + std::to_string(launch.concurrency)
        + " smem=" + std::to_string(kGemmSmem);
}

void launch_gemm(
    const void* a,
    const void* trellis,
    void* c,
    const void* suh,
    void* a_had,
    const void* svh,
    int size_m,
    int size_k,
    int size_n,
    int k_bits,
    int cb,
    int out_fp32,
    cudaStream_t stream)
{
    if (!(k_bits == 6 && cb == 2 && out_fp32 == 0))
    {
        throw std::runtime_error(
            "exl3_gemm: no linked kernel for K=" + std::to_string(k_bits)
            + " codebook=" + std::to_string(cb)
            + (out_fp32 ? " fp32" : " fp16")
            + " (this shim links K=6 mul1 fp16, shapes 1-4)");
    }

    int device = 0;
    berth_cuda_ok(cudaGetDevice(&device), "cudaGetDevice");
    if (device < 0 || device >= MAX_DEVICES) throw std::runtime_error("gemm: device index out of range");
    int sms = berth_device_sms(device);
    int cc = berth_device_cc(device);
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

    // exl3_gemm passes MAX(size_m, 2) so m == 1 shares the m == 2 tune entry.
    uint64_t key = gemm_hash(MAX(size_m, 2), size_k, size_n, k_bits, out_fp32, device, cc, sms, cb);

    CoopAutotuneLaunch tuned{};
    if (CoopKernelAutotuner::launch_locked(key, kernel_args, kGemmSmem, stream, &tuned))
    {
        describe_gemm(tuned);
        return;
    }

    std::vector<CoopAutotuneCandidate> candidates;
    for (int shape = 1; shape <= EXL3_GEMM_NUM_SHAPES; ++shape)
    {
        if (!shape_compat(shape, size_m, size_k, size_n)) continue;
        void* kernel = berth_exl3_gemm_k6_mul1_fp16(shape);
        if (!kernel) continue;
        int max_slices = MAX(size_k / kTileK[shape] * size_n / kTileN[shape], 1);
        int max_sms = MAX(MIN(max_slices, sms), 1);
        candidates.push_back({kernel, kBlock[shape], max_sms, 1, max_sms, shape});
    }
    if (candidates.empty())
        throw std::runtime_error("exl3_gemm autotune: no compatible linked shapes");

    tuned = CoopKernelAutotuner::launch(
        key, candidates, kernel_args, kGemmSmem, stream, (size_t) size_k * (size_t) size_n);
    describe_gemm(tuned);
}

void launch_linear(
    const void* a,
    const void* trellis,
    void* c,
    const void* suh,
    void* a_had,
    const void* svh,
    int size_m,
    int size_k,
    int size_n,
    int k_bits,
    int codebook,
    int out_fp32,
    cudaStream_t stream)
{
    if (!a || !trellis || !c || !suh || !a_had || !svh)
        throw std::runtime_error("exl3 linear: null device pointer");
    if (size_m < 1 || size_k < 1 || size_n < 1)
        throw std::runtime_error("exl3 linear: sizes must be positive");
    if (k_bits < 1 || k_bits > 8)
        throw std::runtime_error("exl3 linear: K must be in 1..=8");
    if (codebook < 0 || codebook > 2)
        throw std::runtime_error("exl3 linear: codebook must be 0 (default), 1 (mcg), or 2 (mul1)");
    if (size_k % 16 != 0)
        throw std::runtime_error("exl3 linear: size_k must be a multiple of 16");
    if (size_n % 128 != 0)
        throw std::runtime_error("exl3 linear: size_n must be a multiple of 128");

    // int8 GEMV. Same predicate as exl3_gemv_int8, before any QTIP / GEMM choice.
    if (codebook == 2 && int8_mode() != 0
        && size_n % 256 == 0 && size_k % 128 == 0)
    {
        int device = 0;
        berth_cuda_ok(cudaGetDevice(&device), "cudaGetDevice");
        if (k_bits >= 1 && k_bits <= int8_max_k(device))
        {
            if (int8_mode() == 1)
                throw std::runtime_error("EXL3_INT8_GEMV=1 residual sq kernels are not linked");
            if (size_m == 1)
            {
                int grid = 0;
                size_t smem = 0;
                int launched = berth_exl3_try_int8_sq(
                    a, trellis, c, size_m, size_k, size_n, k_bits, out_fp32,
                    suh, a_had, svh, (void*) stream, &grid, &smem);
                if (launched)
                {
                    g_launch = "exl3_gemv_int8_sq_kernel<4,1,"
                        + std::string(out_fp32 ? "1" : "0")
                        + ",0> grid=" + std::to_string(grid)
                        + " block=256 smem=" + std::to_string(smem);
                    return;
                }
                throw std::runtime_error(
                    "int8 sq declined this m=1 shape; cooperative int8 GEMV is not linked");
            }
            if (size_m == 2)
                throw std::runtime_error("int8 sq M=2 instance is not linked");
            // m > 2: the fork's int8 entry returns false and the caller continues.
        }
    }

    // QTIP GEMV (exl3_gemv_try_launch) is not linked. Decline only when its
    // free integer checks pass; the occupancy heuristic can still say no, and
    // this shim will not guess and run GEMM in that case.
    const char* gemv_env = std::getenv("EXL3_GEMV");
    int gemv_mode = gemv_env ? std::atoi(gemv_env) : 1;
    bool qtip = gemv_mode != 0
        && k_bits >= 2 && k_bits <= 4
        && !(k_bits != 4 && codebook == 0)
        && size_m <= kGemvMaxM
        && size_k % 128 == 0
        && size_n % 128 == 0;
    if (qtip)
        throw std::runtime_error("QTIP exl3_gemv is not linked; refusing to launch a different kernel");

    launch_gemm(a, trellis, c, suh, a_had, svh, size_m, size_k, size_n, k_bits, codebook, out_fp32, stream);
}

struct GpuFree {
    void* ptr = nullptr;
    ~GpuFree() { if (ptr) cudaFree(ptr); }
    void* alloc(size_t bytes)
    {
        berth_cuda_ok(cudaMalloc(&ptr, bytes), "smoke cudaMalloc");
        berth_cuda_ok(cudaMemset(ptr, 0, bytes), "smoke cudaMemset");
        return ptr;
    }
};

}  // namespace

extern "C" int berth_device_cc(int device)
{
    std::lock_guard<std::mutex> lock(g_dev_mu);
    if (device < 0 || device >= MAX_DEVICES) throw std::runtime_error("device index out of range");
    if (!g_dev[device].cc)
    {
        cudaDeviceProp prop{};
        berth_cuda_ok(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
        int cc = CC_OLD;
        if (prop.major >= 10) cc = CC_BLACKWELL;
        else if (prop.major >= 9) cc = CC_HOPPER;
        else if (prop.major >= 8 && prop.minor >= 9) cc = CC_ADA;
        else if (prop.major >= 8) cc = CC_AMPERE;
        g_dev[device].cc = cc;
    }
    return g_dev[device].cc;
}

extern "C" int berth_device_sms(int device)
{
    std::lock_guard<std::mutex> lock(g_dev_mu);
    if (device < 0 || device >= MAX_DEVICES) throw std::runtime_error("device index out of range");
    if (!g_dev[device].sms)
    {
        int sms = 0;
        berth_cuda_ok(
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device),
            "multiProcessorCount");
        g_dev[device].sms = sms;
    }
    return g_dev[device].sms;
}

extern "C" int* berth_device_locks(int device)
{
    std::lock_guard<std::mutex> lock(g_dev_mu);
    if (device < 0 || device >= MAX_DEVICES) throw std::runtime_error("device index out of range");
    if (!g_dev[device].locks)
    {
        size_t bytes = (MAX_TILES_C + MAX_BARRIERS * 2 + MOE_SCHED_INTS) * sizeof(int);
        berth_cuda_ok(cudaMalloc(&g_dev[device].locks, bytes), "gemm lock buffer");
        berth_cuda_ok(cudaMemset(g_dev[device].locks, 0, bytes), "gemm lock memset");
    }
    return g_dev[device].locks;
}

extern "C" uint64_t berth_exl3_gemm_tune_key(
    int size_m,
    int size_k,
    int size_n,
    int k_bits,
    int out_fp32,
    int device,
    int cc,
    int sms,
    int cb)
{
    // Same key exl3_gemm passes: m == 1 shares the m == 2 entry.
    return gemm_hash(MAX(size_m, 2), size_k, size_n, k_bits, out_fp32, device, cc, sms, cb);
}

extern "C" int berth_exl3_linear(
    const void* a,
    const void* trellis,
    void* c,
    const void* suh,
    void* a_had,
    const void* svh,
    int size_m,
    int size_k,
    int size_n,
    int k_bits,
    int codebook,
    int out_fp32,
    void* stream)
{
    try
    {
        g_err.clear();
        g_launch.clear();
        launch_linear(
            a, trellis, c, suh, a_had, svh,
            size_m, size_k, size_n, k_bits, codebook, out_fp32,
            (cudaStream_t) stream);
        return 0;
    }
    catch (const std::exception& err)
    {
        g_err = err.what();
        g_launch.clear();
        cudaGetLastError();
        return -1;
    }
}

extern "C" const char* berth_exl3_last_error(void)
{
    return g_err.c_str();
}

extern "C" const char* berth_exl3_last_launch(void)
{
    return g_launch.c_str();
}

extern "C" int berth_exl3_smoke_int8(int out_fp32)
{
    // The max-K gate is a process-lifetime cache. Set the serve's values
    // before the first int8 launch in this process.
    setenv("EXL3_INT8_GEMV", "2", 1);
    setenv("EXL3_INT8_GEMV_MAX_K", "5", 1);
    // Smallest sq shape: K = 4, m = 1, k % 128 == 0, n % 256 == 0.
    const int m = 1, k = 128, n = 256, bits = 4, cb = 2;
    size_t trellis_bytes = (size_t) (k / 16) * (n / 16) * (16 * bits) * sizeof(uint16_t);
    size_t c_bytes = (size_t) m * n * (out_fp32 ? sizeof(float) : sizeof(uint16_t));
    try
    {
        GpuFree a, trellis, c, suh, a_had, svh;
        int rc = berth_exl3_linear(
            a.alloc((size_t) m * k * sizeof(uint16_t)),
            trellis.alloc(trellis_bytes),
            c.alloc(c_bytes),
            suh.alloc((size_t) k * sizeof(uint16_t)),
            a_had.alloc((size_t) m * k * sizeof(uint16_t)),
            svh.alloc((size_t) n * sizeof(uint16_t)),
            m, k, n, bits, cb, out_fp32, nullptr);
        if (rc != 0) return rc;
        berth_cuda_ok(cudaDeviceSynchronize(), "int8 smoke sync");
        return 0;
    }
    catch (const std::exception& err)
    {
        g_err = err.what();
        return -1;
    }
}

extern "C" int berth_exl3_smoke_gemm(void)
{
    // Shapes 1-4 all divide k = 32 and n = 512. Redirect the tune cache so a
    // miss does not append to the service's coop_autotune_v1.bin.
    setenv("EXLLAMAV3_TUNE_CACHE", "/tmp/berth-kernels-tune-cache.bin", 1);
    const int m = 1, k = 32, n = 512, bits = 6, cb = 2;
    size_t trellis_bytes = (size_t) (k / 16) * (n / 16) * (16 * bits) * sizeof(uint16_t);
    try
    {
        GpuFree a, trellis, c, suh, a_had, svh;
        int rc = berth_exl3_linear(
            a.alloc((size_t) m * k * sizeof(uint16_t)),
            trellis.alloc(trellis_bytes),
            c.alloc((size_t) m * n * sizeof(uint16_t)),
            suh.alloc((size_t) k * sizeof(uint16_t)),
            a_had.alloc((size_t) m * k * sizeof(uint16_t)),
            svh.alloc((size_t) n * sizeof(uint16_t)),
            m, k, n, bits, cb, 0, nullptr);
        if (rc != 0) return rc;
        berth_cuda_ok(cudaDeviceSynchronize(), "gemm smoke sync");
        return 0;
    }
    catch (const std::exception& err)
    {
        g_err = err.what();
        return -1;
    }
}
