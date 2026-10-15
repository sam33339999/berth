// rope_gr for the profiled instance rope_kernel<2,1>: NEOX, bf16 q/k norm.
//
// The device function is defined in rope.cu, and rope.cuh includes ATen, so
// the kernel is transcribed here and instantiated only for that pair. Other
// rope modes and fp16 norms fail closed. reduction.cuh supplies
// warp_reduce_sum_f. ROPESTYLE_* match rope.cuh.

#include "attn/error.h"
#include "check.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

// reduction.cuh uses NEG_INF_F16 from util.cuh. Only the macro is required.
#ifndef NEG_INF_F16
#define NEG_INF_F16 __ushort_as_half(0xFC00)
#endif

#include "reduction.cuh"

#define MAX_NUM_THREADS 1024
#define MAX_ROTATE_DIMS 4

#define ROPESTYLE_NONE 0
#define ROPESTYLE_GPTJ 1
#define ROPESTYLE_NEOX 2

using bfloat16 = __nv_bfloat16;
using bfloat162 = __nv_bfloat162;

template <int rope_mode, bool norm_bf16>
__global__ void rope_kernel(
    const half* __restrict__ q,
    half* __restrict__ out_q,
    const half* __restrict__ k,
    half* __restrict__ out_k,
    const float* __restrict__ inv_freq,
    const int bsz,
    const int seq_len,
    const int num_heads_q,
    const int num_heads_k,
    const int head_dim,
    const int q_head_stride,
    const int k_head_stride,
    const int partial_head_dim,
    const int position,
    const uint32_t* __restrict__ positions,
    const uint32_t* __restrict__ position_ids,
    float attn_factor,
    const void* __restrict__ q_norm,
    const void* __restrict__ k_norm,
    const float norm_eps,
    const float norm_constant_bias,
    const bool inv_freq_table,
    const int inv_freq_stride,
    const float llama_4_scaling_beta,
    const int llama_4_scaling_original,
    int position_ids_stride,
    int rotate_dims,
    int rotate_offset)
{
    int batch = blockIdx.y;
    int token_pos = blockIdx.x;
    int t = threadIdx.x;
    int t_head = threadIdx.y;

    auto get_pos = [&] (int rdim) -> int {
        int pos = token_pos + position;
        if (positions)
            pos = token_pos + positions[batch];
        else if (position_ids) {
            int idx = (batch * seq_len + token_pos) * position_ids_stride;
            if (position_ids_stride > 1) idx += rdim;
            pos = position_ids[idx];
        }
        return pos;
    };

    auto get_sincos = [&] (int rdim, float& sin, float& cos) {
        int pos = get_pos(rdim);
        if (!inv_freq_table) {
            float fr = inv_freq[t];
            float pf = __int2float_rn(pos);
            sin = __sinf(fr * pf) * attn_factor;
            cos = __cosf(fr * pf) * attn_factor;
        } else {
            float fr = inv_freq[batch * inv_freq_stride + pos * partial_head_dim / 2 + t];
            sin = __sinf(fr) * attn_factor;
            cos = __cosf(fr) * attn_factor;
        }
    };

    // Llama 4 scaling: position-dependent scale on the full query head, queries only.
    float l4_scaling = 1.0f;
    if (llama_4_scaling_beta > 0.0f)
        l4_scaling = 1.0f + llama_4_scaling_beta * __logf(1.0f + (float) (get_pos(0) / llama_4_scaling_original));

    float sin_cache[MAX_ROTATE_DIMS];
    float cos_cache[MAX_ROTATE_DIMS];
    if (t < partial_head_dim / 2) {
        for (int rdim = 0; rdim < rotate_dims; ++rdim)
            get_sincos(rdim, sin_cache[rdim], cos_cache[rdim]);
    }

    __shared__ half norm_head[MAX_NUM_THREADS * 2];
    __shared__ float sums[MAX_NUM_THREADS / 32];

    int head_dim_pad = (head_dim + 63) / 64 * 64;

    for (int head_idx = blockIdx.z * blockDim.y + threadIdx.y;
         head_idx < num_heads_q + num_heads_k;
         head_idx += gridDim.z * blockDim.y) {
        const half* g_head_in_ptr = nullptr;
        half* g_head_out_ptr = nullptr;
        const void* norm_weight = nullptr;
        if (head_idx < num_heads_q) {
            g_head_in_ptr = q + ((batch * seq_len + token_pos) * num_heads_q + head_idx) * q_head_stride;
            g_head_out_ptr = out_q + ((batch * seq_len + token_pos) * num_heads_q + head_idx) * q_head_stride;
            norm_weight = q_norm;
        } else if (head_idx < num_heads_q + num_heads_k) {
            g_head_in_ptr = k + ((batch * seq_len + token_pos) * num_heads_k + head_idx - num_heads_q) * k_head_stride;
            g_head_out_ptr = out_k + ((batch * seq_len + token_pos) * num_heads_k + head_idx - num_heads_q) * k_head_stride;
            norm_weight = k_norm;
        }

        half* sh_head = norm_head + t_head * head_dim_pad;
        auto load_head = [&] () {
            if (t < head_dim / 2)
                ((half2*) sh_head)[t] = ((half2*) g_head_in_ptr)[t];
            else
                ((half2*) sh_head)[t] = {};
            __syncthreads();
        };
        auto store_head = [&] () {
            if (t < head_dim / 2)
                ((half2*) g_head_out_ptr)[t] = ((half2*) sh_head)[t];
            __syncthreads();
        };

        auto apply_rope = [&] () {
            for (int rdim = 0; rdim < rotate_dims; ++rdim) {
                int offset = rotate_offset + partial_head_dim * rdim;
                if (t < partial_head_dim / 2) {
                    float sin = sin_cache[rdim];
                    float cos = cos_cache[rdim];

                    if constexpr (rope_mode == ROPESTYLE_NEOX) {
                        float v1 = __half2float(sh_head[offset + t]);
                        float v2 = __half2float(sh_head[offset + t + partial_head_dim / 2]);
                        float r1 = v1 * cos - v2 * sin;
                        float r2 = v2 * cos + v1 * sin;
                        sh_head[offset + t] = __float2half_rn(r1);
                        sh_head[offset + t + partial_head_dim / 2] = __float2half_rn(r2);
                    } else if constexpr (rope_mode == ROPESTYLE_GPTJ) {
                        half2* tptr = (half2*) (sh_head + offset + t * 2);
                        half2 v = *tptr;
                        float v1 = __low2float(v);
                        float v2 = __high2float(v);
                        float r1 = v1 * cos - v2 * sin;
                        float r2 = v2 * cos + v1 * sin;
                        v = __floats2half2_rn(r1, r2);
                        *tptr = v;
                    }
                }
            }
            __syncthreads();
        };

        auto apply_norm = [&] () {
            half2* tptr = (half2*) (sh_head + t * 2);
            int lane_id = threadIdx.x % 32;
            int warp_id = threadIdx.x / 32;
            int warps = blockDim.x / 32;

            half2 v = *tptr;
            float v1 = __low2float(v);
            float v2 = __high2float(v);
            float sum = v1 * v1 + v2 * v2;
            sum = warp_reduce_sum_f(sum);
            if (lane_id == 0) sums[warps * t_head + warp_id] = sum;
            __syncthreads();

            sum = sums[warps * t_head];
            for (int i = 1; i < warps; ++i) sum += sums[warps * t_head + i];

            float rmf = rsqrtf(sum / (float) head_dim + norm_eps);
            v1 *= rmf;
            v2 *= rmf;

            // Lanes past head_dim / 2 hold padding and must not read the weight.
            if (t * 2 < head_dim) {
                if constexpr (norm_bf16) {
                    bfloat162* wptr = (bfloat162*) (((bfloat16*) norm_weight) + t * 2);
                    float2 w = __bfloat1622float2(*wptr);
                    w.x += norm_constant_bias;
                    w.y += norm_constant_bias;
                    v1 *= w.x;
                    v2 *= w.y;
                    v = __floats2half2_rn(v1, v2);
                } else {
                    half2 norm_constant_bias_h2 = __float2half2_rn(norm_constant_bias);
                    half2* wptr = (half2*) (((half*) norm_weight) + t * 2);
                    half2 w = __hadd2(*wptr, norm_constant_bias_h2);
                    v = __floats2half2_rn(v1, v2);
                    v = __hmul2(w, v);
                }
                *tptr = v;
            }
            __syncthreads();
        };

        // Whole block must call this (blockDim.y walks q and k together). K heads scale by 1.
        auto apply_l4_scaling = [&] (float scaling) {
            if (t < head_dim / 2) {
                half2* tptr = ((half2*) sh_head) + t;
                half2 v = *tptr;
                *tptr = __floats2half2_rn(__low2float(v) * scaling, __high2float(v) * scaling);
            }
            __syncthreads();
        };

        load_head();
        if (q_norm) apply_norm();
        apply_rope();
        if (l4_scaling != 1.0f) apply_l4_scaling(head_idx < num_heads_q ? l4_scaling : 1.0f);
        store_head();
    }
}

extern "C" int berth_attn_rope(
    const void* q,
    void* out_q,
    const void* k,
    void* out_k,
    const void* inv_freq,
    int bsz,
    int seq_len,
    int num_heads_q,
    int num_heads_k,
    int head_dim,
    int q_head_stride,
    int k_head_stride,
    int partial_head_dim,
    unsigned position,
    const void* positions,
    const void* position_ids,
    int rope_mode,
    float attn_factor,
    const void* q_norm,
    const void* k_norm,
    float norm_eps,
    float norm_constant_bias,
    int inv_freq_table,
    int inv_freq_stride,
    float llama_4_scaling_beta,
    int llama_4_scaling_original,
    int position_ids_stride,
    int rotate_dims,
    int rotate_offset,
    void* stream)
{
    try {
        berth_attn_clear_error();
        berth_attn_check(q && out_q && inv_freq, "rope: null q, out_q, or inv_freq");
        berth_attn_check(rope_mode == ROPESTYLE_NEOX, "rope: only NEOX (rope_mode 2) with bf16 norms is linked");
        berth_attn_check((q_norm == nullptr) == (k_norm == nullptr), "rope: q_norm and k_norm must both be set or both be null");
        berth_attn_check((k == nullptr) == (out_k == nullptr), "rope: k and out_k must both be set or both be null");
        berth_attn_check(num_heads_k == 0 || k != nullptr, "rope: k is required when num_heads_k > 0");
        berth_attn_check(bsz > 0 && seq_len > 0 && num_heads_q > 0, "rope: bsz, seq_len, and num_heads_q must be positive");
        berth_attn_check(num_heads_k >= 0, "rope: num_heads_k must be non-negative");
        berth_attn_check(head_dim > 0 && (head_dim % 2) == 0, "rope: head_dim must be a positive even number");
        berth_attn_check(q_head_stride > 0 && (num_heads_k == 0 || k_head_stride > 0), "rope: head strides must be positive");
        berth_attn_check(partial_head_dim > 0 && (partial_head_dim % 2) == 0, "rope: partial_head_dim must be a positive even number");
        berth_attn_check(rotate_dims > 0 && rotate_dims <= MAX_ROTATE_DIMS, "rope: rotate_dims out of range");
        berth_attn_check(
            rotate_dims == 1 || head_dim == partial_head_dim * rotate_dims,
            "rope: rotate_dims is inconsistent with inv_freq and head_dim");
        berth_attn_check(
            rotate_offset >= 0 && rotate_offset + partial_head_dim * rotate_dims <= head_dim,
            "rope: rotate_offset out of range");
        berth_attn_check(inv_freq_table == 0 || inv_freq_table == 1, "rope: inv_freq_table must be 0 or 1");
        berth_attn_check((positions == nullptr) || (position_ids == nullptr), "rope: positions and position_ids are mutually exclusive");
        berth_attn_check(position_ids == nullptr || position_ids_stride >= 1, "rope: position_ids_stride must be positive");

        int warps = ceil_div_int(head_dim / 2, 32);
        int thr = warps * 32;
        berth_attn_check(thr > 0 && thr <= MAX_NUM_THREADS, "rope: block is wider than 1024 threads");
        int heads = num_heads_q + num_heads_k;
        int parallel_heads = MAX_NUM_THREADS / thr;
        if (parallel_heads > heads) parallel_heads = heads;
        berth_attn_check(parallel_heads > 0, "rope: no parallel heads");
        int head_groups = ceil_div_int(heads, parallel_heads);
        if (seq_len * bsz >= 32) head_groups = 1;

        int head_dim_pad = (head_dim + 63) / 64 * 64;
        berth_attn_check(
            (long long) parallel_heads * head_dim_pad <= (long long) MAX_NUM_THREADS * 2,
            "rope: head tile does not fit the shared norm buffer");

        dim3 blocks(seq_len, bsz, head_groups);
        dim3 threads(thr, parallel_heads);

        bool table = inv_freq_table != 0;
        // The kernel parameter is int. rope_gr passes a uint32_t position; the
        // driver copies four bytes either way.
        int position_i = (int) position;

        rope_kernel<ROPESTYLE_NEOX, true><<<blocks, threads, 0, (cudaStream_t) stream>>>(
            (const half*) q,
            (half*) out_q,
            (const half*) k,
            (half*) out_k,
            (const float*) inv_freq,
            bsz,
            seq_len,
            num_heads_q,
            num_heads_k,
            head_dim,
            q_head_stride,
            k_head_stride,
            partial_head_dim,
            position_i,
            (const uint32_t*) positions,
            (const uint32_t*) position_ids,
            attn_factor,
            q_norm,
            k_norm,
            norm_eps,
            norm_constant_bias,
            table,
            inv_freq_stride,
            llama_4_scaling_beta,
            llama_4_scaling_original,
            position_ids_stride,
            rotate_dims,
            rotate_offset);
        berth_cuda_ok(cudaGetLastError(), "rope");
        return 0;
    } catch (const std::exception& err) {
        berth_attn_set_error(err.what());
        return -1;
    }
}
