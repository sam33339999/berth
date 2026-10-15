// Prefill EXL3 linear, rows > 1.
//
// LinearEXL3.forward (modules/quant/exl3.py): rows <= 144, or
// no_reconstruct, calls BC_LinearEXL3::run_alloc. That reshapes to 2D and
// calls exl3_gemm (libtorch/linear.cpp). exl3_gemm tries the int8 sq GEMV
// first; plain mode (EXL3_INT8_GEMV=2) accepts m == 2. Anything larger, and
// a declined m == 2, continues to the cooperative kernel unless QTIP's free
// checks pass (m <= 8). QTIP is not linked here, same as the decode shim.
//
// rows > 144 calls reconstruct_hgemm. The fp16 weight is scratch for that
// call (exl3.py allocates `w` inside the function and drops it). It is not
// cached. Fused reconstruct (original-basis weights, no standalone Hadamard)
// runs when both dims are multiples of 128, EXL3_NO_FUSED_RECONSTRUCT is
// unset or "0", and rows >= 1024. The GEMM is hgemm_recon: f16acc when the
// shape is covered and worthwhile and the rate probe (or EXL3_HGEMM_F16ACC)
// says so, otherwise cublasGemmEx.
//
// The coop autotune key is berth_exl3_gemm_tune_key, the same hash exl3_gemm
// passes (max(m, 2), device, cc, SMs, codebook).

#include "prefill_lin/api.h"
#include "internal.h"
#include "check.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "coop_autotune.cuh"
#include "exl3_devctx.cuh"
#include "exl3_kernel_map.cuh"
#include "util.h"

extern "C" uint64_t berth_exl3_gemm_tune_key(
    int size_m, int size_k, int size_n, int k_bits, int out_fp32,
    int device, int cc, int sms, int cb);

namespace {

constexpr size_t kGemmSmem = 90 * 1024;
constexpr int kGemvMaxM = 8;
constexpr int kMaxSliceN = 32768;

thread_local std::string g_err;
thread_local std::string g_launch;

const int kTileK[] = {EXL3_GEMM_TILESIZE_K};
const int kTileN[] = {EXL3_GEMM_TILESIZE_N};
const int kTileM[] = {EXL3_GEMM_TILESIZE_M};
const int kBlock[] = {EXL3_GEMM_BLOCKDIM};

int int8_mode()
{
    const char* value = std::getenv("EXL3_INT8_GEMV");
    return value ? std::atoi(value) : 2;
}

int int8_max_k(int device)
{
    const char* value = std::getenv("EXL3_INT8_GEMV_MAX_K");
    int from_env = value ? std::atoi(value) : 0;
    if (from_env) return MIN(from_env, 8);
    int cc = berth_device_cc(device);
    return (cc == CC_HOPPER || cc == CC_BLACKWELL) ? 6 : 5;
}

bool shape_compat(int shape, int size_m, int size_k, int size_n)
{
    if (kTileM[shape] > 16 && size_m < kTileM[shape]) return false;
    return (size_k % kTileK[shape] == 0) && (size_n % kTileN[shape] == 0);
}

int sh_stages(int shape) { return shape == 1 ? 6 : 4; }
int frag_stages(int shape) { return shape == 1 ? 5 : 3; }

void describe_gemm(const CoopAutotuneLaunch& launch, int out_fp32)
{
    int shape = launch.tag;
    g_launch = "exl3_gemm_kernel<4,"
        + std::to_string(out_fp32) + ",2,"
        + std::to_string(kTileM[shape]) + ","
        + std::to_string(kTileK[shape]) + ","
        + std::to_string(kTileN[shape]) + ","
        + std::to_string(sh_stages(shape)) + ","
        + std::to_string(frag_stages(shape)) + ">"
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
    if (!(k_bits == 4 && cb == 2))
    {
        throw std::runtime_error(
            "prefill exl3_gemm: no linked kernel for K=" + std::to_string(k_bits)
            + " codebook=" + std::to_string(cb)
            + " (this shim links K=4 mul1, shapes 1-6)");
    }

    int device = 0;
    berth_cuda_ok(cudaGetDevice(&device), "cudaGetDevice");
    if (device < 0 || device >= MAX_DEVICES)
        throw std::runtime_error("prefill gemm: device index out of range");
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

    uint64_t key = berth_exl3_gemm_tune_key(
        size_m, size_k, size_n, k_bits, out_fp32, device, cc, sms, cb);

    CoopAutotuneLaunch tuned{};
    if (CoopKernelAutotuner::launch_locked(key, kernel_args, kGemmSmem, stream, &tuned))
    {
        describe_gemm(tuned, out_fp32);
        return;
    }

    std::vector<CoopAutotuneCandidate> candidates;
    for (int shape = 1; shape <= EXL3_GEMM_NUM_SHAPES; ++shape)
    {
        if (!shape_compat(shape, size_m, size_k, size_n)) continue;
        void* kernel = out_fp32
            ? berth_prefill_gemm_k4_mul1_f32(shape)
            : berth_prefill_gemm_k4_mul1_f16(shape);
        if (!kernel) continue;
        int max_slices = MAX(size_k / kTileK[shape] * size_n / kTileN[shape], 1);
        int max_sms = MAX(MIN(max_slices, sms), 1);
        candidates.push_back({kernel, kBlock[shape], max_sms, 1, max_sms, shape});
    }
    if (candidates.empty())
        throw std::runtime_error("prefill exl3_gemm autotune: no compatible linked shapes");

    tuned = CoopKernelAutotuner::launch(
        key, candidates, kernel_args, kGemmSmem, stream, (size_t) size_k * (size_t) size_n);
    describe_gemm(tuned, out_fp32);
}

bool no_fused_reconstruct()
{
    const char* env = std::getenv("EXL3_NO_FUSED_RECONSTRUCT");
    return env && std::strcmp(env, "0") != 0;
}

void launch_reconstruct(
    const void* a,
    const void* trellis,
    void* c,
    const void* suh,
    void* a_had,
    const void* svh,
    void* w_scratch,
    int size_m,
    int size_k,
    int size_n,
    int out_fp32,
    cudaStream_t stream)
{
    if (size_k % 128 != 0 || size_n % 128 != 0)
        throw std::runtime_error("prefill reconstruct: K and N must be multiples of 128");

    bool use_fused = !no_fused_reconstruct() && size_m >= 1024;
    const void* xh = use_fused ? a : a_had;
    if (!use_fused)
        berth_prefill_had_r_128(a, a_had, suh, 1, size_m, size_k, 0, stream);

    int kind = -1;
    for (int n0 = 0; n0 < size_n; n0 += kMaxSliceN)
    {
        int slice = MIN(kMaxSliceN, size_n - n0);
        if (use_fused)
            berth_prefill_reconstruct_had(w_scratch, trellis, suh, svh, size_k, slice, size_n, n0, stream);
        else
            berth_prefill_reconstruct(w_scratch, trellis, size_k, slice, size_n, n0, stream);
        char* c_col = (char*) c + (size_t) n0 * (out_fp32 ? sizeof(float) : sizeof(half));
        int this_kind = berth_prefill_hgemm_recon(
            xh, w_scratch, c_col, size_m, size_k, slice, size_n, out_fp32, stream);
        kind = this_kind;
    }
    if (!use_fused)
        berth_prefill_had_r_128(c, c, svh, 0, size_m, size_n, out_fp32, stream);

    g_launch = use_fused
        ? "reconstruct_had_kernel<4,2>"
        : "had_hf_r_128_kernel<1,0> + reconstruct_kernel<4,2>";
    g_launch += kind == 1
        ? (out_fp32 ? " + f16acc::gemm_kernel<1,128,false>" : " + f16acc::gemm_kernel<0,128,false>")
        : " + cublasGemmEx";
    if (!use_fused)
        g_launch += out_fp32 ? " + had_ff_r_128_kernel<0,1>" : " + had_hf_r_128_kernel<0,1>";
}

void launch_prefill(
    const void* a,
    const void* trellis,
    void* c,
    const void* suh,
    void* a_had,
    const void* svh,
    void* w_scratch,
    int size_m,
    int size_k,
    int size_n,
    int k_bits,
    int codebook,
    int out_fp32,
    cudaStream_t stream)
{
    if (!a || !trellis || !c || !suh || !a_had || !svh)
        throw std::runtime_error("prefill exl3 linear: null device pointer");
    if (size_m < 2)
        throw std::runtime_error("prefill exl3 linear: size_m must be > 1 (m = 1 is exl3_linear)");
    if (size_k < 1 || size_n < 1)
        throw std::runtime_error("prefill exl3 linear: sizes must be positive");
    if (k_bits < 1 || k_bits > 8)
        throw std::runtime_error("prefill exl3 linear: K must be in 1..=8");
    if (codebook < 0 || codebook > 2)
        throw std::runtime_error("prefill exl3 linear: codebook must be 0, 1, or 2");
    if (size_k % 16 != 0)
        throw std::runtime_error("prefill exl3 linear: size_k must be a multiple of 16");
    if (size_n % 128 != 0)
        throw std::runtime_error("prefill exl3 linear: size_n must be a multiple of 128");
    if (size_m > 144 && !w_scratch)
        throw std::runtime_error(
            "prefill exl3 linear: w_scratch (fp16, size_k * min(size_n, 32768)) is required when rows > 144");

    // exl3.py: rows past AUTO_RECONSTRUCT_THRESHOLD take reconstruct_hgemm.
    if (size_m > 144)
    {
        if (!(k_bits == 4 && codebook == 2))
            throw std::runtime_error("prefill reconstruct is linked for K=4 mul1 only");
        launch_reconstruct(
            a, trellis, c, suh, a_had, svh, w_scratch,
            size_m, size_k, size_n, out_fp32, stream);
        return;
    }

    int device = 0;
    berth_cuda_ok(cudaGetDevice(&device), "cudaGetDevice");
    bool int8_geometry = codebook == 2
        && size_n % 256 == 0
        && size_k % 128 == 0
        && k_bits >= 1
        && k_bits <= int8_max_k(device);
    int mode = int8_mode();
    if (mode != 0 && int8_geometry && size_m == 2)
    {
        if (mode == 1)
        {
            // Residual sq is m == 1 only. m == 2 falls through to QTIP / GEMM.
        }
        else if (k_bits != 4)
        {
            throw std::runtime_error("prefill int8 sq M=2 is linked for K=4 only");
        }
        else
        {
            int grid = 0;
            size_t smem = 0;
            int launched = berth_prefill_try_int8_m2(
                a, trellis, c, size_k, size_n, out_fp32,
                suh, a_had, svh, stream, &grid, &smem);
            if (launched)
            {
                g_launch = "exl3_gemv_int8_sq_kernel<4,2,"
                    + std::string(out_fp32 ? "1" : "0")
                    + ",0> grid=" + std::to_string(grid)
                    + " block=256 smem=" + std::to_string(smem);
                return;
            }
        }
    }

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

    launch_gemm(
        a, trellis, c, suh, a_had, svh,
        size_m, size_k, size_n, k_bits, codebook, out_fp32, stream);
}

}  // namespace

extern "C" int berth_prefill_exl3_linear(
    const void* a,
    const void* trellis,
    void* c,
    const void* suh,
    void* a_had,
    const void* svh,
    void* w_scratch,
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
        launch_prefill(
            a, trellis, c, suh, a_had, svh, w_scratch,
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

extern "C" const char* berth_prefill_last_error(void)
{
    return g_err.c_str();
}

extern "C" const char* berth_prefill_last_launch(void)
{
    return g_launch.c_str();
}
