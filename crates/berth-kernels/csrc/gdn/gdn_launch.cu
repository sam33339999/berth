// Gated DeltaNet decode kernels for one token, batch 1.
//
// The device code is the fork's. gdn.cu and norm.cu define it in translation
// units that include ATen, c10, and pybind11, so build.rs extracts the kernel
// line ranges into OUT_DIR/fork/ (FORK_EXTRACTS) and this file includes those.
// No fork source is stored in this repo.
//
// Only the instances measured on the 27B decode are instantiated:
// conv1d_update_kernel<1,0>, gated_delta_net_fused_op_3_kernel<bf16>,
// cuda_recurrent_gated_delta_rule_kernel_128<0,4,0>, and
// gated_rms_norm_kernel<32,half,bf16,float>. gdn_ba_gemv_kernel is not templated.
// Launchers below reproduce gdn_ba_gemv_gr, gated_delta_net_fused_op_3_gr,
// cuda_causal_conv1d_update_gr, cuda_recurrent_gated_delta_rule_gr, and
// gated_rms_norm_gr (grid, block, dynamic smem 0, argument order).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

#include <climits>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "check.cuh"
#include "util.h"
#include "util.cuh"

namespace {

using bfloat16 = __nv_bfloat16;

// The fork's kernels and their helpers, extracted from gdn.cu and norm.cu at
// build time (build.rs FORK_EXTRACTS). gdn_kernels.cuh carries gdn.cu's
// MAX_*_HEADS, SUBK, and FUSED_OP_*_THREADS defines.
#include "fork/norm_kernels.cuh"
#include "fork/gdn_kernels.cuh"

// Host launchers. Grid, block, dynamic smem, and argument order match
// gdn.cu / norm.cu for the 27B decode instances. Dynamic smem is 0.

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

// Drop a sticky runtime error from an earlier launch in this process.
// Called only after the host checks, so a rejected shape never touches CUDA.
void clear_sticky()
{
    cudaGetLastError();
}

void launch_ba_gemv(
    const void* x,
    const void* w_t,
    const void* bias,
    void* y,
    int rows,
    int k,
    int n,
    cudaStream_t stream)
{
    if (!x || !w_t || !y)
        throw std::runtime_error("gdn_ba_gemv: x, w_t, and y are required");
    if (rows <= 0 || k <= 0 || n <= 0)
        throw std::runtime_error("gdn_ba_gemv: rows, k, and n must be positive");
    if ((k & 1) != 0)
        throw std::runtime_error("gdn_ba_gemv: k must be even");

    dim3 grid(CEIL_DIVIDE(n, BA_GEMV_WARPS), rows, 1);
    dim3 block(BA_GEMV_WARPS * 32, 1, 1);
    clear_sticky();
    gdn_ba_gemv_kernel<<<grid, block, 0, stream>>>(
        static_cast<const half*>(x),
        static_cast<const half*>(w_t),
        static_cast<const half*>(bias),
        static_cast<float*>(y),
        k,
        n);
    berth_cuda_ok(cudaGetLastError(), "gdn_ba_gemv");
    g_launch = describe("gdn_ba_gemv_kernel", grid, block, 0);
}

void launch_conv1d(
    const void* x,
    void* conv_state,
    const void* slots,
    const void* weight,
    const void* bias,
    void* out,
    int bsz,
    int dim,
    int seqlen,
    int state_size,
    int k,
    cudaStream_t stream)
{
    if (!x || !conv_state || !weight || !out)
        throw std::runtime_error("conv1d_update: x, conv_state, weight, and out are required");
    if (bsz <= 0 || dim <= 0 || seqlen <= 0)
        throw std::runtime_error("conv1d_update: bsz, dim, and seqlen must be positive");
    if (k < 1 || k > CONV1D_MAX_K)
        throw std::runtime_error("conv1d_update: K must be in 1..=16");
    if (state_size < k)
        throw std::runtime_error("conv1d_update: state_size must be at least K");

    // activation=true, history=false: conv1d_update_kernel<1,0>. The other
    // three template instances are not linked.
    dim3 grid(CEIL_DIVIDE(dim, CONV1D_NUM_THREADS), bsz, 1);
    dim3 block(CONV1D_NUM_THREADS, 1, 1);
    clear_sticky();
    conv1d_update_kernel<true, false><<<grid, block, 0, stream>>>(
        static_cast<const bfloat16*>(x),
        static_cast<bfloat16*>(conv_state),
        static_cast<const int*>(slots),
        static_cast<const bfloat16*>(weight),
        static_cast<const bfloat16*>(bias),
        static_cast<bfloat16*>(out),
        dim,
        seqlen,
        state_size,
        k);
    berth_cuda_ok(cudaGetLastError(), "conv1d_update");
    g_launch = describe("conv1d_update_kernel<1,0>", grid, block, 0);
}

void launch_fused_op_3(
    const void* qkv,
    const void* ba,
    const void* dt_bias,
    const void* a_log,
    void* mixed_qkv,
    void* beta,
    void* g,
    int b,
    int s,
    int f,
    int h,
    float beta_scale,
    cudaStream_t stream)
{
    if (!qkv || !ba || !dt_bias || !a_log || !mixed_qkv || !beta || !g)
        throw std::runtime_error(
            "gated_delta_net_fused_op_3: qkv, ba, dt_bias, a_log, mixed_qkv, beta, and g are required");
    if (b <= 0 || s <= 0 || f <= 0 || h <= 0)
        throw std::runtime_error("gated_delta_net_fused_op_3: B, S, F, and H must be positive");

    const long long bs = static_cast<long long>(b) * static_cast<long long>(s);
    const long long total = bs * (static_cast<long long>(f) + static_cast<long long>(h));
    if (bs > INT_MAX || total > INT_MAX)
        throw std::runtime_error("gated_delta_net_fused_op_3: element count does not fit in int32");

    // a_log is bf16 on the 27B (checkpoint tensor A_log). The fp32 instance is
    // not linked: it would be a different specialization.
    const int BS = static_cast<int>(bs);
    const int blocks_i = CEIL_DIVIDE(static_cast<int>(total), FUSED_OP_3_THREADS);
    dim3 grid(blocks_i, 1, 1);
    dim3 block(FUSED_OP_3_THREADS, 1, 1);
    clear_sticky();
    gated_delta_net_fused_op_3_kernel<bfloat16><<<grid, block, 0, stream>>>(
        static_cast<const float*>(qkv),
        static_cast<const float*>(ba),
        static_cast<const bfloat16*>(dt_bias),
        static_cast<const bfloat16*>(a_log),
        static_cast<bfloat16*>(mixed_qkv),
        static_cast<bfloat16*>(beta),
        static_cast<float*>(g),
        BS,
        s,
        f,
        h,
        beta_scale);
    berth_cuda_ok(cudaGetLastError(), "gated_delta_net_fused_op_3");
    g_launch = describe("gated_delta_net_fused_op_3_kernel<bf16>", grid, block, 0);
}

void launch_recurrent(
    const void* mixed_qkv,
    const void* g,
    const void* beta,
    void* recurrent_state,
    void* core_attn_out,
    const void* slots,
    int bsz,
    int seqlen,
    int num_k_heads,
    int num_v_heads,
    int k_head_dim,
    int v_head_dim,
    int history_stride,
    cudaStream_t stream)
{
    if (!mixed_qkv || !g || !beta || !recurrent_state || !core_attn_out)
        throw std::runtime_error(
            "recurrent_gated_delta_rule: mixed_qkv, g, beta, recurrent_state, and core_attn_out are required");
    // v_split is 4 only for bsz == 1, both head dims 128, and num_v_heads <= 64.
    // Any other shape selects kernel_128<*,1> or the generic kernel.
    if (bsz != 1)
        throw std::runtime_error("recurrent_gated_delta_rule: bsz must be 1");
    if (k_head_dim != 128 || v_head_dim != 128)
        throw std::runtime_error("recurrent_gated_delta_rule: k_head_dim and v_head_dim must be 128");
    if (num_k_heads <= 0 || num_v_heads <= 0 || num_v_heads > 64 || num_v_heads % num_k_heads != 0)
        throw std::runtime_error(
            "recurrent_gated_delta_rule: num_v_heads must be a positive multiple of num_k_heads and at most 64");
    if (seqlen <= 0 || history_stride <= 0)
        throw std::runtime_error("recurrent_gated_delta_rule: seqlen and history_stride must be positive");

    constexpr int v_split = 4;
    dim3 grid(bsz, num_v_heads, v_split);
    dim3 block(MAX(k_head_dim, v_head_dim / v_split), SUBK, 1);
    // Host libm sqrtf, same as cuda_recurrent_gated_delta_rule_gr. --use_fast_math
    // does not rewrite this host call. D is nullptr: GDN has no Mamba2 skip.
    const float scale = 1.0f / sqrtf(static_cast<float>(k_head_dim));
    clear_sticky();
    cuda_recurrent_gated_delta_rule_kernel_128<false, 4><<<grid, block, 0, stream>>>(
        static_cast<const bfloat16*>(mixed_qkv),
        static_cast<const float*>(g),
        static_cast<const bfloat16*>(beta),
        static_cast<float*>(recurrent_state),
        static_cast<bfloat16*>(core_attn_out),
        bsz,
        seqlen,
        num_k_heads,
        num_v_heads,
        k_head_dim,
        v_head_dim,
        scale,
        static_cast<const int*>(slots),
        history_stride,
        nullptr);
    berth_cuda_ok(cudaGetLastError(), "recurrent_gated_delta_rule");
    g_launch = describe("cuda_recurrent_gated_delta_rule_kernel_128<0,4,0>", grid, block, 0);
}

void launch_gated_rms(
    const void* x,
    const void* w,
    void* y,
    const void* g,
    float epsilon,
    float constant_bias,
    int rows,
    int dim,
    int w_groups,
    int gate_first,
    int gate_act,
    cudaStream_t stream)
{
    if (!x || !w || !y || !g)
        throw std::runtime_error("gated_rms_norm: x, w, y, and g are required");
    if (rows <= 0)
        throw std::runtime_error("gated_rms_norm: rows must be positive");
    // dim > 256 selects the 1024-thread instance. dim not divisible by 4 is
    // rejected by gated_rms_norm_gr (vector width 4).
    if (dim <= 0 || dim > 256 || (dim & 3) != 0)
        throw std::runtime_error("gated_rms_norm: dim must be a positive multiple of 4 and at most 256");
    if (w_groups <= 0)
        throw std::runtime_error("gated_rms_norm: w_groups must be positive");
    if (gate_first != 0 && gate_first != 1)
        throw std::runtime_error("gated_rms_norm: gate_first must be 0 or 1");
    if (gate_act != 0 && gate_act != 1)
        throw std::runtime_error("gated_rms_norm: gate_act must be 0 (silu) or 1 (sigmoid)");

    dim3 grid(rows, 1, 1);
    dim3 block(32, 1, 1);
    clear_sticky();
    gated_rms_norm_kernel<32, half, bfloat16, float><<<grid, block, 0, stream>>>(
        static_cast<const bfloat16*>(x),
        static_cast<const bfloat16*>(w),
        static_cast<half*>(y),
        static_cast<const float*>(g),
        epsilon,
        rows,
        dim,
        constant_bias,
        w_groups,
        gate_first != 0,
        gate_act);
    berth_cuda_ok(cudaGetLastError(), "gated_rms_norm");
    g_launch = describe("gated_rms_norm_kernel<32,half,bf16,float>", grid, block, 0);
}

}  // namespace

template <typename F>
int berth_gdn_guard(F&& fn)
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
        g_err = "gdn: unknown error";
        g_launch.clear();
        return 1;
    }
}

extern "C" int berth_gdn_ba_gemv(
    const void* x,
    const void* w_t,
    const void* bias,
    void* y,
    int rows,
    int k,
    int n,
    void* stream)
{
    return berth_gdn_guard([&] {
        launch_ba_gemv(x, w_t, bias, y, rows, k, n, static_cast<cudaStream_t>(stream));
    });
}

extern "C" int berth_gdn_conv1d_update(
    const void* x,
    void* conv_state,
    const void* slots,
    const void* weight,
    const void* bias,
    void* out,
    int bsz,
    int dim,
    int seqlen,
    int state_size,
    int k,
    void* stream)
{
    return berth_gdn_guard([&] {
        launch_conv1d(
            x, conv_state, slots, weight, bias, out,
            bsz, dim, seqlen, state_size, k, static_cast<cudaStream_t>(stream));
    });
}

extern "C" int berth_gdn_fused_op_3(
    const void* qkv,
    const void* ba,
    const void* dt_bias,
    const void* a_log,
    void* mixed_qkv,
    void* beta,
    void* g,
    int b,
    int s,
    int f,
    int h,
    float beta_scale,
    void* stream)
{
    return berth_gdn_guard([&] {
        launch_fused_op_3(
            qkv, ba, dt_bias, a_log, mixed_qkv, beta, g,
            b, s, f, h, beta_scale, static_cast<cudaStream_t>(stream));
    });
}

extern "C" int berth_gdn_recurrent(
    const void* mixed_qkv,
    const void* g,
    const void* beta,
    void* recurrent_state,
    void* core_attn_out,
    const void* slots,
    int bsz,
    int seqlen,
    int num_k_heads,
    int num_v_heads,
    int k_head_dim,
    int v_head_dim,
    int history_stride,
    void* stream)
{
    return berth_gdn_guard([&] {
        launch_recurrent(
            mixed_qkv, g, beta, recurrent_state, core_attn_out, slots,
            bsz, seqlen, num_k_heads, num_v_heads, k_head_dim, v_head_dim,
            history_stride, static_cast<cudaStream_t>(stream));
    });
}

extern "C" int berth_gdn_gated_rms_norm(
    const void* x,
    const void* w,
    void* y,
    const void* g,
    float epsilon,
    float constant_bias,
    int rows,
    int dim,
    int w_groups,
    int gate_first,
    int gate_act,
    void* stream)
{
    return berth_gdn_guard([&] {
        launch_gated_rms(
            x, w, y, g, epsilon, constant_bias, rows, dim, w_groups,
            gate_first, gate_act, static_cast<cudaStream_t>(stream));
    });
}

extern "C" const char* berth_gdn_last_error(void)
{
    return g_err.c_str();
}

extern "C" const char* berth_gdn_last_launch(void)
{
    return g_launch.c_str();
}
