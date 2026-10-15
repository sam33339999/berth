// Gated DeltaNet decode kernels for one token, batch 1.
//
// The device functions below are the fork's, copied verbatim. They are not in
// a torch-free header: gdn.cu and norm.cu define them in translation units that
// include ATen, c10, and pybind11. Those headers are not included here. The
// copied ranges are from exllamav3_ext tree de35ffbe336bdc6d617e49e208cd7c730a789787:
//
//   gdn.cu 22-51     _sigmoid_fast_exp, trunc_bf16, as_float, softplus
//   gdn.cu 660-849   cuda_recurrent_gated_delta_rule_kernel_128
//   gdn.cu 1309-1406 conv1d_update_kernel
//   gdn.cu 1513-1561 gated_delta_net_fused_op_3_kernel
//   gdn.cu 1630-1669 gdn_ba_gemv_kernel
//   norm.cu 12-40, 60-68, 70-73, 75-83, 96-99, 101-108, 110-116, 126-136
//                    reduce and the vector helpers gated_rms_norm_kernel names
//                    (write_float4 is only the discarded fp32-output branch; the
//                    name is not dependent, so it has to be declared)
//   norm.cu 499-598  gated_rms_norm_kernel
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

// gdn.cu lines 17 and 20. The fused kernel does not use the threads macro;
// the launcher does. SUBK is the recurrent kernel's row split.
#define SUBK 4
#define FUSED_OP_3_THREADS 256

// gdn.cu lines 22-51
__device__ __forceinline__ float _sigmoid_fast_exp(float x)
{
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ bfloat16 trunc_bf16(float x)
{
    return __float2bfloat16_rn(x);
}

__device__ __forceinline__ float untrunc_bf16(bfloat16 x)
{
    return __bfloat162float(x);
}

__device__ __forceinline__ float as_float(bfloat16 x)
{
    return __bfloat162float(x);
}

__device__ __forceinline__ float as_float(float x)
{
    return x;
}

__device__ __forceinline__ float softplus(float x)  // beta=1.0, linear threshold=20.0
{
    if (x > 20.0f) return x;
    return log1pf(__expf(x));
}

// gdn.cu lines 660-849
template <bool save_history, int V_SPLIT, bool CHANNELWISE = false>
__global__ __launch_bounds__(128 * SUBK)
void cuda_recurrent_gated_delta_rule_kernel_128
(
                                                // k_head_dim = v_head_dim = 128
    const bfloat16* __restrict__ mixed_qkv,     // [bsz, seqlen, (k_dim + k_dim + v_dim)]
    const float* __restrict__ g,                // [bsz, seqlen, (group * num_k_heads)], or
                                                // [bsz, seqlen, heads, 128] log-decay per
                                                // k-channel when CHANNELWISE (KDA)
    const bfloat16* __restrict__ beta,          // [bsz, seqlen, (group * num_k_heads)]
    float* __restrict__ recurrent_state,        // [num_slots, max_history + 1, (group * num_k_heads), 128, 128]
    bfloat16* __restrict__ core_attn_out,       // [bsz, seqlen, num_v_heads, 128]
    const int bsz,
    const int seqlen,
    const int num_k_heads,
    const int num_v_heads,
    const int k_head_dim,
    const int v_head_dim,
    const float scale,
    const int* __restrict__ slots,              // [bsz]
    const int history_stride,                   // max_history + 1
    const float* __restrict__ D                 // unused, matches the generic kernel signature
)
{
    constexpr int HEAD_DIM = 128;
    constexpr int V_CHUNK_DIM = HEAD_DIM / V_SPLIT;
    constexpr int BTS = HEAD_DIM / SUBK;

    int group = num_v_heads / num_k_heads;
    constexpr size_t HEAD_STATE_SIZE = HEAD_DIM * HEAD_DIM;
    const size_t state_size = group * num_k_heads * HEAD_STATE_SIZE;
    const size_t slot_size = (size_t) history_stride * state_size;

    int bi = blockIdx.x;
    mixed_qkv +=        bi * seqlen * (3 * HEAD_DIM * num_k_heads + HEAD_DIM * (num_v_heads - num_k_heads));
    g +=                (size_t) bi * seqlen * (group * num_k_heads) * (CHANNELWISE ? HEAD_DIM : 1);
    beta +=             bi * seqlen * (group * num_k_heads);
    int state_slot = slots ? slots[bi] : bi;
    float* slot_state = recurrent_state + (size_t) state_slot * slot_size;
    float* final_state = slot_state;
    core_attn_out +=    bi * seqlen * num_v_heads * HEAD_DIM;

    int t = threadIdx.x;
    int bt = threadIdx.y;
    int lane = t % 32;
    int warp = t / 32;
    int head = blockIdx.y;
    int k_head = head / group;
    int v_chunk = blockIdx.z;
    int v_start = v_chunk * V_CHUNK_DIM;

    __shared__ float sh_red[2][HEAD_DIM / 32];
    __shared__ float sh_k[HEAD_DIM];
    __shared__ float sh_q[HEAD_DIM];
    // Per-k-slice partials, reduced in a fixed order by the consumer (see the generic kernel)
    __shared__ float sh_dot1[SUBK][HEAD_DIM];
    __shared__ float sh_dot2[SUBK][HEAD_DIM];
    __shared__ float sh_g[CHANNELWISE ? HEAD_DIM : 1];

    for (int s = 0; s < seqlen; ++s)
    {
        const bfloat16* gl_q = mixed_qkv + k_head * HEAD_DIM;
        const bfloat16* gl_k = mixed_qkv + (num_k_heads + k_head) * HEAD_DIM;
        const bfloat16* gl_v = mixed_qkv + (2 * num_k_heads * HEAD_DIM) + head * HEAD_DIM + v_start;
        bfloat16* out = core_attn_out + head * HEAD_DIM + v_start;

        float* gl_rs_r;
        float* gl_rs_w;
        if constexpr (save_history)
        {
            bool first = (s == 0);
            bool last = (s == seqlen - 1);
            float* history_r = first ? nullptr : slot_state + (size_t) s * state_size;
            float* history_w = last  ? final_state : slot_state + (size_t) (s + 1) * state_size;
            gl_rs_r = first ? final_state + head * HEAD_STATE_SIZE
                            : history_r   + head * HEAD_STATE_SIZE;
            gl_rs_w = history_w           + head * HEAD_STATE_SIZE;
        }
        else
        {
            gl_rs_r = final_state + head * HEAD_STATE_SIZE;
            gl_rs_w = gl_rs_r;
        }

        float q = __bfloat162float(gl_q[t]);
        float k = __bfloat162float(gl_k[t]);

        float sumq = q * q;
        float sumk = k * k;
        #pragma unroll
        for(int offset = 16; offset > 0; offset /= 2)
        {
            sumq += __shfl_xor_sync(0xffffffff, sumq, offset);
            sumk += __shfl_xor_sync(0xffffffff, sumk, offset);
        }
        if (lane == 0)
        {
            sh_red[0][warp] = sumq;
            sh_red[1][warp] = sumk;
        }
        __syncthreads();

        sumq = lane < HEAD_DIM / 32 ? sh_red[0][lane] : 0.0f;
        sumk = lane < HEAD_DIM / 32 ? sh_red[1][lane] : 0.0f;
        #pragma unroll
        for(int offset = 16; offset > 0; offset /= 2)
        {
            sumq += __shfl_xor_sync(0xffffffff, sumq, offset);
            sumk += __shfl_xor_sync(0xffffffff, sumk, offset);
        }

        q = q * rsqrtf(sumq + 1e-6f);
        k = k * rsqrtf(sumk + 1e-6f);
        sh_k[t] = k;
        sh_q[t] = q;
        if constexpr (CHANNELWISE)
            sh_g[t] = __expf(g[head * HEAD_DIM + t]);

        __syncthreads();

        if (t < V_CHUNK_DIM)
        {
            float sum = 0.0f;
            float* sh_k_rd = sh_k + bt * BTS;
            float* sh_g_rd = sh_g + bt * BTS;
            float* rs_rd = gl_rs_r + v_start + t + bt * BTS * HEAD_DIM;

            #pragma unroll
            for (int i = 0; i < HEAD_DIM / 8 / SUBK; ++i)
            {
                #pragma unroll
                for (int j = 0; j < 8; ++j, rs_rd += HEAD_DIM, sh_k_rd++, sh_g_rd++)
                {
                    if constexpr (CHANNELWISE)
                        // Decay folded per k-channel: kv_mem reads the decayed state
                        sum = sum + *sh_k_rd * *sh_g_rd * *rs_rd;
                    else
                        sum = sum + *sh_k_rd * *rs_rd;
                }
            }
            sh_dot1[bt][t] = sum;
        }
        __syncthreads();

        if (t < V_CHUNK_DIM)
        {
            float g_h = CHANNELWISE ? 1.0f : __expf(g[head]);
            float beta_h = __bfloat162float(beta[head]);
            // CHANNELWISE: sh_dot1 already read the decayed state, no head-wide factor
            float dot1 = 0.0f;
            #pragma unroll
            for (int s = 0; s < SUBK; ++s) dot1 += sh_dot1[s][t];
            float v = __bfloat162float(gl_v[t]) - dot1 * g_h;
            float v_out = 0.0f;
            float* sh_k_rd = sh_k + bt * BTS;
            float* sh_g_rd = sh_g + bt * BTS;
            float* sh_q_rd = sh_q + bt * BTS;
            float* rs_r = gl_rs_r + v_start + t + bt * BTS * HEAD_DIM;
            float* rs_w = gl_rs_w + v_start + t + bt * BTS * HEAD_DIM;

            #pragma unroll
            for (int i = 0; i < HEAD_DIM / 8 / SUBK; ++i)
            {
                #pragma unroll
                for (int j = 0; j < 8; ++j, rs_r += HEAD_DIM, rs_w += HEAD_DIM, sh_k_rd++, sh_g_rd++, sh_q_rd++)
                {
                    float state = *rs_r;
                    state = state * (CHANNELWISE ? *sh_g_rd : g_h) + *sh_k_rd * v * beta_h;
                    *rs_w = state;
                    v_out = v_out + *sh_q_rd * state;
                }
            }
            sh_dot2[bt][t] = v_out;
        }
        __syncthreads();

        if (t < V_CHUNK_DIM && bt == 0)
        {
            float v_out = 0.0f;
            #pragma unroll
            for (int s = 0; s < SUBK; ++s) v_out += sh_dot2[s][t];
            out[t] = __float2bfloat16_rz(v_out * scale);
        }

        mixed_qkv +=        2 * HEAD_DIM * num_k_heads + HEAD_DIM * num_v_heads;
        g +=                num_v_heads * (CHANNELWISE ? HEAD_DIM : 1);
        beta +=             num_v_heads;
        core_attn_out +=    num_v_heads * HEAD_DIM;
    }
}

// gdn.cu lines 1309-1406
#define CONV1D_MAX_K 16
#define CONV1D_NUM_THREADS 256

// Causal conv1d update for short decode steps. Equivalent to the triton kernel in
// modules/gated_delta_net_fn/conv1d.py but launchable in ~4us instead of ~50us of host time.
//
// out[b,s,d] = act(bias[d] + sum_k w[d,k] * in(b,d,s+k+1)) where the input sequence is the
// concatenation of conv_state[slot,d,0:K] and x[b,d,0:seqlen]. Without history, the last K
// inputs are written back to conv_state[slot,d,0:K]; with history, the last
// min(state_size, K+seqlen) inputs are written to the tail of the state buffer (rewindable).

template <bool ACT, bool HISTORY>
__global__ __launch_bounds__(CONV1D_NUM_THREADS)
void conv1d_update_kernel
(
    const bfloat16* __restrict__ x,           // (bsz, dim, seqlen)
    bfloat16* __restrict__ conv_state,        // (num_slots, dim, state_size)
    const int* __restrict__ slots,            // (bsz) or null (identity)
    const bfloat16* __restrict__ weight,      // (dim, K)
    const bfloat16* __restrict__ bias,        // (dim) or null
    bfloat16* __restrict__ out,               // (bsz, seqlen, dim)
    const int dim,
    const int seqlen,
    const int state_size,
    const int K
)
{
    int d = blockIdx.x * CONV1D_NUM_THREADS + threadIdx.x;
    if (d >= dim) return;
    int b = blockIdx.y;
    int slot = slots ? slots[b] : b;

    const bfloat16* x_d = x + ((size_t) b * dim + d) * seqlen;
    bfloat16* state_d = conv_state + ((size_t) slot * dim + d) * state_size;

    float w[CONV1D_MAX_K];
    #pragma unroll
    for (int k = 0; k < CONV1D_MAX_K; ++k)
        if (k < K) w[k] = __bfloat162float(weight[(size_t) d * K + k]);

    float bias_d = bias ? __bfloat162float(bias[d]) : 0.0f;

    // Previous window; win[K-1] slot is filled with the current input each step
    float old_state[CONV1D_MAX_K];
    float win[CONV1D_MAX_K];
    #pragma unroll
    for (int k = 0; k < CONV1D_MAX_K; ++k)
        if (k < K) old_state[k] = __bfloat162float(state_d[k]);
    #pragma unroll
    for (int k = 0; k < CONV1D_MAX_K - 1; ++k)
        if (k < K - 1) win[k] = old_state[k + 1];

    for (int s = 0; s < seqlen; ++s)
    {
        win[K - 1] = __bfloat162float(x_d[s]);

        float acc = bias_d;
        #pragma unroll
        for (int k = 0; k < CONV1D_MAX_K; ++k)
            if (k < K) acc = fmaf(w[k], win[k], acc);

        if constexpr (ACT)
            acc *= _sigmoid_fast_exp(acc);

        out[((size_t) b * seqlen + s) * dim + d] = __float2bfloat16_rn(acc);

        #pragma unroll
        for (int k = 0; k < CONV1D_MAX_K - 1; ++k)
            if (k < K - 1) win[k] = win[k + 1];
    }

    if constexpr (!HISTORY)
    {
        #pragma unroll
        for (int k = 0; k < CONV1D_MAX_K; ++k)
        {
            if (k < K)
            {
                int src_t = seqlen + k;
                float v = (src_t < K) ? old_state[src_t] : __bfloat162float(x_d[src_t - K]);
                state_d[k] = __float2bfloat16_rn(v);
            }
        }
    }
    else
    {
        int total = K + seqlen;
        int write_size = state_size < total ? state_size : total;
        int dst_start = state_size - write_size;
        int src_start = total - write_size;
        for (int j = 0; j < write_size; ++j)
        {
            int src_t = src_start + j;
            float v = (src_t < K) ? old_state[src_t] : __bfloat162float(x_d[src_t - K]);
            state_d[dst_start + j] = __float2bfloat16_rn(v);
        }
    }
}

// gdn.cu lines 1513-1561
// Split-projection (Qwen3.5) decode helper. Replaces the qkv transpose/cast done in Torch plus
// gated_delta_net_fused_op_2: mixed_qkv[b,f,s] = bf16(qkv[b,s,f]), and beta/g computed from the
// packed ba projection (b = ba[..,:H], a = ba[..,H:])

template <typename a_log_T>
__global__ void gated_delta_net_fused_op_3_kernel
(
    const float* __restrict__ in_qkv,           // [B,S,F]
    const float* __restrict__ in_ba,            // [B,S,2H]
    const bfloat16* __restrict__ in_dt_bias,    // [H]
    const a_log_T* __restrict__ in_a_log,       // [H]
    bfloat16* __restrict__ out_mixed_qkv,       // [B,F,S]
    bfloat16* __restrict__ out_beta,            // [B,S,H]
    float* __restrict__ out_g,                  // [B,S,H]
    const int BS,                               // B * S
    const int S,
    const int F,
    const int H,
    const float beta_scale
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int cast_elems = BS * F;

    if (idx < cast_elems)
    {
        int f = idx % F;
        int row = idx / F;                      // b * S + s
        int b = row / S;
        int s = row % S;
        out_mixed_qkv[((size_t) b * F + f) * S + s] = trunc_bf16(in_qkv[idx]);
    }
    else
    {
        idx -= cast_elems;
        if (idx >= BS * H) return;
        int h = idx % H;
        int row = idx / H;

        float bv = in_ba[(size_t) row * 2 * H + h];
        float av = in_ba[(size_t) row * 2 * H + H + h];
        float beta = _sigmoid_fast_exp(bv) * beta_scale;
        float dt_bias = as_float(in_dt_bias[h]);
        float gv = -softplus(av + dt_bias) * __expf(as_float(in_a_log[h]));

        out_beta[(size_t) row * H + h] = trunc_bf16(beta);
        out_g[(size_t) row * H + h] = gv;
    }
}

// gdn.cu lines 1630-1669
#define BA_GEMV_WARPS 8

__global__ __launch_bounds__(BA_GEMV_WARPS * 32)
void gdn_ba_gemv_kernel
(
    const half* __restrict__ x,                 // [rows, k]
    const half* __restrict__ w_t,               // [n, k]
    const half* __restrict__ bias,              // [n] or null
    float* __restrict__ y,                      // [rows, n]
    const int k,
    const int n
)
{
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * BA_GEMV_WARPS + warp;
    if (row >= n) return;
    int r = blockIdx.y;

    const half2* x2 = (const half2*) (x + (size_t) r * k);
    const half2* w2 = (const half2*) (w_t + (size_t) row * k);

    float sum = 0.0f;
    for (int j = lane; j < k / 2; j += 32)
    {
        float2 xf = __half22float2(x2[j]);
        float2 wf = __half22float2(w2[j]);
        sum = fmaf(xf.x, wf.x, sum);
        sum = fmaf(xf.y, wf.y, sum);
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane == 0)
    {
        if (bias) sum += __half2float(bias[row]);
        y[(size_t) r * n + row] = sum;
    }
}

// norm.cu lines 12-40
template <int num_threads>
__device__ inline float reduce(float sum, int warp_id, int lane_id)
{
    if constexpr (num_threads <= 32)
    {
        // Shuffle to sum across lanes
        __shared__ float sums[num_threads / 32];
        for(int offset = warpSize / 2; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);
        return sum;
    }
    else
    {
        // Shuffle to sum across lanes
        __shared__ float sums[num_threads / 32];
        for(int offset = warpSize / 2; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);
        if (lane_id == 0) sums[warp_id] = sum;
        __syncthreads();

        // Load partial sums from across warps, shuffle again across lanes
        #if defined(USE_ROCM)
            sum = lane_id < num_threads / 32 ? sums[lane_id] : 0.0f;
        #else
            sum = sums[lane_id];
        #endif
        for(int offset = warpSize / 2; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);

        return sum;
    }
}

// norm.cu lines 60-68
__device__ inline void read_bfloat164(float4& f4, const bfloat164* addr)
{
    bfloat164 h4;
    READ64(h4, addr);
    f4.x = __bfloat162float(__low2bfloat16(h4.x));
    f4.y = __bfloat162float(__high2bfloat16(h4.x));
    f4.z = __bfloat162float(__low2bfloat16(h4.y));
    f4.w = __bfloat162float(__high2bfloat16(h4.y));
}

// norm.cu lines 70-73
__device__ inline void read_float4(float4& f4, const float4* addr)
{
    READ128(f4, addr);
}

// norm.cu lines 75-83
__device__ inline void write_half4(const float4& f4, half4* addr)
{
    half4 h4
    (
        __halves2half2(__float2half_rn(f4.x), __float2half_rn(f4.y)),
        __halves2half2(__float2half_rn(f4.z), __float2half_rn(f4.w))
    );
    WRITE64(addr, h4);
}

// norm.cu lines 96-99
__device__ inline void write_float4(const float4& f4, float4* addr)
{
    WRITE128(addr, f4);
}

// norm.cu lines 101-108
__device__ inline float sum_sq4(float lsum, const float4& f4)
{
    lsum = fma(f4.x, f4.x, lsum);
    lsum = fma(f4.y, f4.y, lsum);
    lsum = fma(f4.z, f4.z, lsum);
    lsum = fma(f4.w, f4.w, lsum);
    return lsum;
}

// norm.cu lines 110-116
__device__ inline void apply4(float4& x4, const float4& w4, const float rmf)
{
    x4.x = x4.x * w4.x * rmf;
    x4.y = x4.y * w4.y * rmf;
    x4.z = x4.z * w4.z * rmf;
    x4.w = x4.w * w4.w * rmf;
}

// norm.cu lines 126-136
__device__ __forceinline__ float _silu(float x)
{
    float e     = __expf(-x);
    float recip = __fdividef(1.0f, 1.0f + e);
    return x * recip;
}

__device__ __forceinline__ float _sigmoid_f(float x)
{
    return __fdividef(1.0f, 1.0f + __expf(-x));
}

// norm.cu lines 499-598
template <int num_threads, typename output_t, typename weight_t, typename gate_t>
__global__ __launch_bounds__(num_threads)
void gated_rms_norm_kernel
(
    const bfloat16* __restrict__ x,
    const weight_t* __restrict__ w,
    output_t* __restrict__ y,
    const gate_t* __restrict__ g,
    const float epsilon,
    const int rows,
    const int dim,
    float constant_bias,
    const int w_groups,         // weight spans w_groups rows, cycled by row index (Mamba2 group norm)
    const bool gate_first,      // apply silu(g) before the norm instead of after (Mamba2 style)
    const int gate_act          // 0 = silu (GDN/Mamba2), 1 = sigmoid (KDA)
)
{
    #define _gate_fn(v) (gate_act == 1 ? _sigmoid_f(v) : _silu(v))
    constexpr bool output_fp32 = std::is_same_v<output_t, float>;
    constexpr bool output_fp16 = std::is_same_v<output_t, half>;
    constexpr bool weight_bf16 = std::is_same_v<weight_t, bfloat16>;
    constexpr bool gate_fp32   = std::is_same_v<gate_t,   float>;

    int t = threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row = blockIdx.x;
    const size_t row_off = (size_t) row * dim;   // 64-bit: rows * dim can exceed 2^31 elements

    int columns = dim / 4;
    const weight_t* w_row = w + (size_t) (row % w_groups) * dim;

    // Compute sum of squares
    float sum = 0.0f;
    for (int column = t; column < columns; column += num_threads)
    {
        float4 x4;
        read_bfloat164(x4, ((const bfloat164*) (x + row_off)) + column);
        if (gate_first)
        {
            float4 g4;
            if constexpr (gate_fp32)   read_float4   (g4, ((const float4*)    (g + row_off)) + column);
            else                       read_bfloat164(g4, ((const bfloat164*) (g + row_off)) + column);
            x4.x *= _gate_fn(g4.x);
            x4.y *= _gate_fn(g4.y);
            x4.z *= _gate_fn(g4.z);
            x4.w *= _gate_fn(g4.w);
        }
        sum = sum_sq4(sum, x4);
    }
    sum = reduce<num_threads>(sum, warp_id, lane_id);

    // Get norm
    float rmf = rsqrtf(sum / (float)dim + epsilon);

    // Normalize x, scaling by w * silu(g)
    for (int column = t; column < columns; column += num_threads)
    {
        float4 x4;
        float4 w4;
        float4 g4;

        read_bfloat164(x4, ((const bfloat164*) (x + row_off)) + column);
        if constexpr (weight_bf16) read_bfloat164(w4, ((const bfloat164*) w_row) + column);
        else                       read_float4   (w4, ((const float4*)    w_row) + column);
        if constexpr (gate_fp32)   read_float4   (g4, ((const float4*)    (g + row_off)) + column);
        else                       read_bfloat164(g4, ((const bfloat164*) (g + row_off)) + column);

        if (constant_bias != 0.0f)
        {
            w4.x += constant_bias;
            w4.y += constant_bias;
            w4.z += constant_bias;
            w4.w += constant_bias;
        }

        if (gate_first)
        {
            x4.x *= _gate_fn(g4.x);
            x4.y *= _gate_fn(g4.y);
            x4.z *= _gate_fn(g4.z);
            x4.w *= _gate_fn(g4.w);

            apply4(x4, w4, rmf);
        }
        else
        {
            apply4(x4, w4, rmf);

            x4.x *= _gate_fn(g4.x);
            x4.y *= _gate_fn(g4.y);
            x4.z *= _gate_fn(g4.z);
            x4.w *= _gate_fn(g4.w);
        }

        if constexpr (output_fp16) write_half4(x4, ((half4*) (y + row_off)) + column);
        if constexpr (output_fp32) write_float4(x4, ((float4*) (y + row_off)) + column);
    }
    #undef _gate_fn
}

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
