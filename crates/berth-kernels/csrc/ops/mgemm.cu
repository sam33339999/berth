// ATen-free exl3_mgemm_gr. Argument order matches EXL3_MGEMM_ARGS. The
// autotune key matches mgemm_autotune_hash, including the sliced-mode xor.
// m = 1 is keyed as 1 (gemm_autotune_m_key), not max(m, 2).
//
// Decode on the 27B passes force_shape_idx = -1, so this launcher only
// autotunes. Indices, weights, and expert-range filtering stay null: the
// fused qkv / qkvz bundles do not use them. K = 4 mul1 is the linked pair
// (fp16 and fp32, shapes 1-4).

#include "internal.h"
#include "check.cuh"
#include "ops/ops_state.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "coop_autotune.cuh"
#include "exl3_devctx.cuh"
#include "exl3_kernel_map.cuh"
#include "util.h"

extern "C" void* berth_ops_mgemm_k4_mul1_f16(int shape_idx);
extern "C" void* berth_ops_mgemm_k4_mul1_f32(int shape_idx);

namespace {

constexpr size_t kGemmSmem = 90 * 1024;

const int kTileM[] = {EXL3_GEMM_TILESIZE_M};
const int kTileK[] = {EXL3_GEMM_TILESIZE_K};
const int kTileN[] = {EXL3_GEMM_TILESIZE_N};
const int kBlock[] = {EXL3_GEMM_BLOCKDIM};

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

int m_key(int size_m)
{
    if (size_m <= 16) return (int) MIN(roundup_pow2((uint64_t) size_m), (uint64_t) 16);
    if (size_m <= 31) return 32;
    if (size_m <= 63) return 64;
    if (size_m <= 127) return 128;
    return 512;
}

uint64_t mgemm_hash(
    int size_m,
    int size_k,
    int size_n,
    int k_bits,
    int c_fp32,
    int device,
    int cc,
    int sms,
    int cb,
    int bszm_in,
    int bszm_out)
{
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) {
        h ^= v;
        h *= 1099511628211ull;
    };
    mix((uint64_t) m_key(size_m));
    mix((uint64_t) size_k);
    mix((uint64_t) size_n);
    mix((uint64_t) k_bits);
    mix(c_fp32 ? 1ull : 0ull);
    mix((uint64_t) device);
    mix((uint64_t) cc);
    mix((uint64_t) sms);
    mix((uint64_t) cb);
    mix((uint64_t) MIN(bszm_in, 24));
    mix((uint64_t) MIN(bszm_out, 24));
    return h;
}

bool shape_compat(int shape, int size_m, int size_k, int size_n)
{
    if (kTileM[shape] > 16 && size_m < kTileM[shape]) return false;
    return (size_k % kTileK[shape] == 0) && (size_n % kTileN[shape] == 0);
}

int sh_stages(int shape)
{
    return shape == 1 ? 6 : 4;
}

int frag_stages(int shape)
{
    return shape == 1 ? 5 : 3;
}

void describe(const CoopAutotuneLaunch& launch, int c_fp32)
{
    int shape = launch.tag;
    std::string text = "exl3_mgemm_kernel<4,"
        + std::to_string(c_fp32) + ",2,"
        + std::to_string(kTileM[shape]) + ","
        + std::to_string(kTileK[shape]) + ","
        + std::to_string(kTileN[shape]) + ","
        + std::to_string(sh_stages(shape)) + ","
        + std::to_string(frag_stages(shape)) + ">"
        + " block=" + std::to_string(launch.block_dim)
        + " grid=" + std::to_string(launch.num_sms) + "x1x" + std::to_string(launch.concurrency)
        + " smem=" + std::to_string(kGemmSmem);
    berth_ops_set_launch(text);
}

int launch_mgemm(
    const void* a,
    const void* b_list,
    void* c,
    const void* suh_list,
    void* a_had,
    const void* svh_list,
    int size_m,
    int size_k,
    int size_n,
    int k_bits,
    int codebook,
    int c_fp32,
    int bszm_in,
    int bszm_out,
    const int* size_n_list,
    const void* c_ptrs,
    const int* n_stride_list,
    const int* had_src_list,
    int num_had_src,
    cudaStream_t stream)
{
    if (!a || !b_list || !c || !suh_list || !a_had || !svh_list)
        throw std::runtime_error("exl3_mgemm: null device pointer");
    if (size_m < 1 || size_k < 1 || size_n < 1 || bszm_in < 1 || bszm_out < 1)
        throw std::runtime_error("exl3_mgemm: sizes must be positive");
    if (size_k % 16 != 0)
        throw std::runtime_error("exl3_mgemm: size_k must be a multiple of 16");
    if (size_n % 128 != 0)
        throw std::runtime_error("exl3_mgemm: size_n must be a multiple of 128");
    if (k_bits != 4 || codebook != 2)
        throw std::runtime_error(
            "exl3_mgemm: no linked kernel for K=" + std::to_string(k_bits)
            + " codebook=" + std::to_string(codebook)
            + " (this shim links K=4 mul1, shapes 1-4)");
    if (c_fp32 != 0 && c_fp32 != 1)
        throw std::runtime_error("exl3_mgemm: c_fp32 must be 0 (fp16) or 1 (fp32)");

    bool sliced = had_src_list != nullptr;
    bool any_list = size_n_list || c_ptrs || n_stride_list || had_src_list;
    if (any_list)
    {
        if (!size_n_list || !c_ptrs || !n_stride_list || !had_src_list)
            throw std::runtime_error(
                "exl3_mgemm: sliced mode requires size_n_list, c_ptrs, n_stride_list, and had_src_list");
        if (num_had_src < 1)
            throw std::runtime_error("exl3_mgemm: sliced mode needs one suh entry per source");
        if (bszm_in != 1)
            throw std::runtime_error("exl3_mgemm: sliced mode is single-input");
    }
    else if (num_had_src != 0)
    {
        throw std::runtime_error("exl3_mgemm: num_had_src requires had_src_list");
    }

    int device = 0;
    berth_cuda_ok(cudaGetDevice(&device), "cudaGetDevice");
    if (device < 0 || device >= MAX_DEVICES)
        throw std::runtime_error("exl3_mgemm: device index out of range");
    int sms = berth_device_sms(device);
    int cc = berth_device_cc(device);
    int* locks = berth_device_locks(device);

    const half* a_ptr = (const half*) a;
    const uint16_t** b_ptr = (const uint16_t**) b_list;
    void* c_ptr = c;
    const half** suh_ptr = (const half**) suh_list;
    half* a_had_ptr = (half*) a_had;
    const half** svh_ptr = (const half**) svh_list;
    const int64_t* indices_ptr = nullptr;
    const half* weights_ptr = nullptr;
    int min_index = -1;
    int max_index = -1;
    int num_tokens = 1;
    const int* size_n_ptr = size_n_list;
    void** c_list_ptr = (void**) c_ptrs;
    const int* n_stride_ptr = n_stride_list;
    const int* had_src_ptr = had_src_list;
    int bszm = MAX(bszm_in, bszm_out);

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
        (void*) &indices_ptr,
        (void*) &weights_ptr,
        (void*) &bszm_in,
        (void*) &bszm_out,
        (void*) &min_index,
        (void*) &max_index,
        (void*) &num_tokens,
        (void*) &size_n_ptr,
        (void*) &c_list_ptr,
        (void*) &n_stride_ptr,
        (void*) &had_src_ptr,
        (void*) &num_had_src,
    };

    uint64_t key = mgemm_hash(
        size_m, size_k, size_n, k_bits, c_fp32, device, cc, sms, codebook, bszm_in, bszm_out);
    if (sliced) key ^= 0x9e3779b97f4a7c15ull;

    CoopAutotuneLaunch tuned{};
    if (CoopKernelAutotuner::launch_locked(key, kernel_args, kGemmSmem, stream, &tuned))
    {
        describe(tuned, c_fp32);
        return tuned.tag;
    }

    std::vector<CoopAutotuneCandidate> candidates;
    for (int shape = 1; shape <= EXL3_GEMM_NUM_SHAPES; ++shape)
    {
        if (!shape_compat(shape, size_m, size_k, size_n)) continue;
        void* kernel = c_fp32
            ? berth_ops_mgemm_k4_mul1_f32(shape)
            : berth_ops_mgemm_k4_mul1_f16(shape);
        if (!kernel) continue;
        int max_slices = MAX(size_k / kTileK[shape] * size_n / kTileN[shape], 1);
        int max_sms = MAX(MIN(max_slices, sms), 1);
        CoopAutotuneCandidate cand{};
        cand.kernel = kernel;
        cand.block_dim = kBlock[shape];
        cand.max_num_sms = max_sms;
        cand.max_concurrency = bszm;
        cand.total_sms = sms;
        cand.tag = shape;
        candidates.push_back(cand);
    }
    if (candidates.empty())
        throw std::runtime_error("exl3_mgemm autotune: no compatible linked shapes");

    tuned = CoopKernelAutotuner::launch(
        key,
        candidates,
        kernel_args,
        kGemmSmem,
        stream,
        (size_t) size_k * (size_t) size_n * (size_t) bszm);
    describe(tuned, c_fp32);
    return tuned.tag;
}

}  // namespace

extern "C" uint64_t berth_ops_mgemm_tune_key(
    int size_m,
    int size_k,
    int size_n,
    int k_bits,
    int c_fp32,
    int device,
    int cc,
    int sms,
    int cb,
    int bszm_in,
    int bszm_out,
    int sliced)
{
    uint64_t key = mgemm_hash(
        size_m, size_k, size_n, k_bits, c_fp32, device, cc, sms, cb, bszm_in, bszm_out);
    if (sliced) key ^= 0x9e3779b97f4a7c15ull;
    return key;
}

extern "C" int berth_ops_mgemm(
    const void* a,
    const void* b_list,
    void* c,
    const void* suh_list,
    void* a_had,
    const void* svh_list,
    int size_m,
    int size_k,
    int size_n,
    int k_bits,
    int codebook,
    int c_fp32,
    int bszm_in,
    int bszm_out,
    const int* size_n_list,
    const void* c_ptrs,
    const int* n_stride_list,
    const int* had_src_list,
    int num_had_src,
    void* stream)
{
    try
    {
        berth_ops_clear();
        return launch_mgemm(
            a, b_list, c, suh_list, a_had, svh_list,
            size_m, size_k, size_n, k_bits, codebook, c_fp32,
            bszm_in, bszm_out,
            size_n_list, c_ptrs, n_stride_list, had_src_list, num_had_src,
            (cudaStream_t) stream);
    }
    catch (const std::exception& err)
    {
        berth_ops_set_error(err.what());
        cudaGetLastError();
        return -1;
    }
}
