// rope_gr for the profiled instance rope_kernel<2,1>: NEOX, bf16 q/k norm.
//
// The device function is defined in rope.cu, and rope.cuh includes ATen, so
// build.rs extracts it into OUT_DIR/fork/rope_kernel.cuh; this file launches
// only that pair. Other
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

#define ROPESTYLE_NONE 0
#define ROPESTYLE_GPTJ 1
#define ROPESTYLE_NEOX 2

using bfloat16 = __nv_bfloat16;
using bfloat162 = __nv_bfloat162;

// rope_kernel (and rope.cu's MAX_NUM_THREADS / MAX_ROTATE_DIMS) is the fork's,
// extracted at build time (build.rs FORK_EXTRACTS).
#include "fork/rope_kernel.cuh"

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
