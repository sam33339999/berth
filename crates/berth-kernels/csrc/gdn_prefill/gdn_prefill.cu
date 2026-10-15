// Gated DeltaNet prefill host launches that are not Triton cubins.
//
// fused_op_2 is the fork's kernel (gdn.cu). The prefill translation unit does
// not include gdn_kernels.cuh: that header also defines the decode __global__
// kernels, and there is no relocatable device link. build.rs writes a private
// copy of the helpers plus gated_delta_net_fused_op_2_kernel into
// OUT_DIR/fork/gdn_prefill_kernels.cuh. The include sits in a namespace so
// those device functions do not collide with the decode TU.
//
// The transpose and the q/k/v pack are the ATen ops the fork runs around that
// kernel: qkv.transpose(1, 2).to(bfloat16).contiguous(), and the contiguous()
// copies input_guard makes of the strided q, k, v views. Both are pure
// round-to-nearest-even casts or byte copies.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

#include "check.cuh"

namespace prefill_kern {

using bfloat16 = __nv_bfloat16;

#include "fork/gdn_prefill_kernels.cuh"

}  // namespace prefill_kern

namespace {

thread_local std::string g_err;
thread_local std::string g_launch;

std::string describe(const char* name, dim3 grid, dim3 block, size_t smem)
{
    std::ostringstream out;
    out << name
        << " grid=" << grid.x << 'x' << grid.y << 'x' << grid.z
        << " block=" << block.x << 'x' << block.y << 'x' << block.z
        << " smem=" << smem;
    return out.str();
}

void clear_sticky()
{
    cudaGetLastError();
}

int ceil_div_int(int n, int d)
{
    return (n + d - 1) / d;
}

// qkv [B, S, F] f32 -> mixed [B, F, S] bf16, __float2bfloat16_rn.
// Fork: qkv.transpose(1, 2).to(torch.bfloat16).contiguous().
__global__ void qkv_transpose_bf16_kernel(
    const float* __restrict__ qkv,
    __nv_bfloat16* __restrict__ mixed,
    int batch,
    int seqlen,
    int features)
{
    const size_t n = static_cast<size_t>(batch) * static_cast<size_t>(seqlen) * static_cast<size_t>(features);
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x)
    {
        const int s = static_cast<int>(i % static_cast<size_t>(seqlen));
        const size_t bf = i / static_cast<size_t>(seqlen);
        const int f = static_cast<int>(bf % static_cast<size_t>(features));
        const int b = static_cast<int>(bf / static_cast<size_t>(features));
        const size_t src = (static_cast<size_t>(b) * static_cast<size_t>(seqlen) + static_cast<size_t>(s))
                * static_cast<size_t>(features)
            + static_cast<size_t>(f);
        mixed[i] = __float2bfloat16_rn(qkv[src]);
    }
}

// conv output is [B, S, F]. The q/k/v views are strided (row stride F).
// input_guard's contiguous() packs each view. q then k then v, feature offsets
// 0, k_dim, 2*k_dim.
__global__ void pack_qkv_kernel(
    const __nv_bfloat16* __restrict__ conv,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v,
    int batch,
    int seqlen,
    int features,
    int k_dim,
    int v_dim)
{
    const size_t n = static_cast<size_t>(batch) * static_cast<size_t>(seqlen) * static_cast<size_t>(features);
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x)
    {
        const int f = static_cast<int>(i % static_cast<size_t>(features));
        const size_t bs = i / static_cast<size_t>(features);
        const int s = static_cast<int>(bs % static_cast<size_t>(seqlen));
        const int b = static_cast<int>(bs / static_cast<size_t>(seqlen));
        const size_t row = static_cast<size_t>(b) * static_cast<size_t>(seqlen) + static_cast<size_t>(s);
        const __nv_bfloat16 value = conv[i];
        if (f < k_dim)
            q[row * static_cast<size_t>(k_dim) + static_cast<size_t>(f)] = value;
        else if (f < k_dim * 2)
            k[row * static_cast<size_t>(k_dim) + static_cast<size_t>(f - k_dim)] = value;
        else
            v[row * static_cast<size_t>(v_dim) + static_cast<size_t>(f - 2 * k_dim)] = value;
    }
}

dim3 stride_grid(size_t n)
{
    const int threads = 256;
    size_t blocks = (n + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads);
    if (blocks < 1)
        blocks = 1;
    if (blocks > 2048)
        blocks = 2048;
    return dim3(static_cast<unsigned>(blocks), 1, 1);
}

void launch_prepare(
    const void* qkv,
    void* mixed_qkv,
    const void* b,
    const void* a,
    const void* dt_bias,
    const void* a_log,
    void* beta,
    void* g,
    void* slot_dev,
    int slot_host,
    int batch,
    int seqlen,
    int features,
    int heads,
    float beta_scale,
    cudaStream_t stream)
{
    if (!qkv || !mixed_qkv || !b || !a || !dt_bias || !a_log || !beta || !g || !slot_dev)
        throw std::runtime_error("gdn_prefill_prepare: required pointer is null");
    if (batch != 1)
        throw std::runtime_error("gdn_prefill_prepare: batch must be 1");
    if (seqlen <= 0 || features <= 0 || heads <= 0)
        throw std::runtime_error("gdn_prefill_prepare: seqlen, features, and heads must be positive");
    if (heads > FUSED_OP_2_THREADS)
        throw std::runtime_error("gdn_prefill_prepare: too many heads");

    // gdn.cu gated_delta_net_fused_op_2, a_log bf16. The 27B checkpoint stores
    // A_log as bf16 and the prefill profile instantiates <__nv_bfloat16>.
    const int rows_per_block = FUSED_OP_2_THREADS / heads;
    const int threads = rows_per_block * heads;
    const int blocks = ceil_div_int(batch * seqlen, rows_per_block);
    dim3 fused_grid(blocks, 1, 1);
    dim3 fused_block(threads, 1, 1);
    clear_sticky();
    prefill_kern::gated_delta_net_fused_op_2_kernel<__nv_bfloat16><<<fused_grid, fused_block, 0, stream>>>(
        static_cast<const float*>(b),
        static_cast<const float*>(a),
        static_cast<const __nv_bfloat16*>(dt_bias),
        static_cast<const __nv_bfloat16*>(a_log),
        static_cast<__nv_bfloat16*>(beta),
        static_cast<float*>(g),
        batch,
        seqlen,
        heads,
        rows_per_block,
        beta_scale);
    berth_cuda_ok(cudaGetLastError(), "gated_delta_net_fused_op_2");

    const size_t n = static_cast<size_t>(batch) * static_cast<size_t>(seqlen) * static_cast<size_t>(features);
    dim3 grid = stride_grid(n);
    dim3 block(256, 1, 1);
    qkv_transpose_bf16_kernel<<<grid, block, 0, stream>>>(
        static_cast<const float*>(qkv),
        static_cast<__nv_bfloat16*>(mixed_qkv),
        batch,
        seqlen,
        features);
    berth_cuda_ok(cudaGetLastError(), "qkv_transpose_bf16");

    berth_cuda_ok(
        cudaMemcpyAsync(slot_dev, &slot_host, sizeof(int), cudaMemcpyHostToDevice, stream),
        "gdn_prefill slot");

    g_launch = describe("gated_delta_net_fused_op_2_kernel<bf16>", fused_grid, fused_block, 0);
}

void launch_pack(
    const void* conv,
    void* q,
    void* k,
    void* v,
    int batch,
    int seqlen,
    int features,
    int k_dim,
    int v_dim,
    cudaStream_t stream)
{
    if (!conv || !q || !k || !v)
        throw std::runtime_error("gdn_prefill_pack: required pointer is null");
    if (batch != 1 || seqlen <= 0 || features != k_dim * 2 + v_dim || k_dim <= 0 || v_dim <= 0)
        throw std::runtime_error("gdn_prefill_pack: bad shape");
    const size_t n = static_cast<size_t>(batch) * static_cast<size_t>(seqlen) * static_cast<size_t>(features);
    dim3 grid = stride_grid(n);
    dim3 block(256, 1, 1);
    clear_sticky();
    pack_qkv_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(conv),
        static_cast<__nv_bfloat16*>(q),
        static_cast<__nv_bfloat16*>(k),
        static_cast<__nv_bfloat16*>(v),
        batch,
        seqlen,
        features,
        k_dim,
        v_dim);
    berth_cuda_ok(cudaGetLastError(), "pack_qkv");
    g_launch = describe("pack_qkv", grid, block, 0);
}

template <typename Fn>
int guard(Fn fn)
{
    try
    {
        g_err.clear();
        g_launch.clear();
        fn();
        return 0;
    }
    catch (const std::exception& err)
    {
        g_err = err.what();
        g_launch.clear();
        return 1;
    }
    catch (...)
    {
        g_err = "gdn_prefill: unknown error";
        g_launch.clear();
        return 1;
    }
}

}  // namespace

extern "C" int berth_gdn_prefill_prepare(
    const void* qkv,
    void* mixed_qkv,
    const void* b,
    const void* a,
    const void* dt_bias,
    const void* a_log,
    void* beta,
    void* g,
    void* slot_dev,
    int slot_host,
    int batch,
    int seqlen,
    int features,
    int heads,
    float beta_scale,
    void* stream)
{
    return guard([&] {
        launch_prepare(
            qkv, mixed_qkv, b, a, dt_bias, a_log, beta, g, slot_dev, slot_host,
            batch, seqlen, features, heads, beta_scale, static_cast<cudaStream_t>(stream));
    });
}

extern "C" int berth_gdn_prefill_pack(
    const void* conv,
    void* q,
    void* k,
    void* v,
    int batch,
    int seqlen,
    int features,
    int k_dim,
    int v_dim,
    void* stream)
{
    return guard([&] {
        launch_pack(
            conv, q, k, v, batch, seqlen, features, k_dim, v_dim, static_cast<cudaStream_t>(stream));
    });
}

extern "C" int berth_gdn_prefill_zero(void* ptr, size_t bytes, void* stream)
{
    return guard([&] {
        if (!ptr && bytes != 0)
            throw std::runtime_error("gdn_prefill_zero: null pointer");
        if (bytes == 0)
            return;
        clear_sticky();
        berth_cuda_ok(
            cudaMemsetAsync(ptr, 0, bytes, static_cast<cudaStream_t>(stream)),
            "gdn_prefill_zero");
    });
}

extern "C" int berth_gdn_prefill_copy(void* dst, const void* src, size_t bytes, void* stream)
{
    return guard([&] {
        if (bytes == 0)
            return;
        if (!dst || !src)
            throw std::runtime_error("gdn_prefill_copy: null pointer");
        clear_sticky();
        berth_cuda_ok(
            cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream)),
            "gdn_prefill_copy");
    });
}

extern "C" const char* berth_gdn_prefill_last_error()
{
    return g_err.c_str();
}

extern "C" const char* berth_gdn_prefill_last_launch()
{
    return g_launch.c_str();
}
