// dequant_cache_paged_window for cq3 K and V.
// q_cache.cu:358. The ATen wrapper is not included; this launches
// dequant_cache_paged_kernel<3,3> from cache/q_cache_kernels.cuh.
//
// Prefill staging (EXL3_QC_STAGING=1, q_len >= 256) dequantizes the block
// table's pages, plus bonus_len rows past cache_seqlens, into a compact fp16
// scratch. compact_out is 1: page p of batch b lands at physical page
// (b * pages_per_seq + p). sliding_window is -1. compand_a is 0 on the 27B.
//
// The include is inside a namespace. The header has external-linkage
// __global__ symbols, and quant.cu already includes it in berth_attn_quant.

#include "attn/error.h"
#include "check.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>

namespace berth_attn_prefill_dequant {

#include "cache/q_cache_kernels.cuh"

}  // namespace berth_attn_prefill_dequant

extern "C" int berth_attn_prefill_dequant_window(
    const void* k_in,
    const void* k_scales,
    void* k_out,
    const void* v_in,
    const void* v_scales,
    void* v_out,
    const void* cache_seqlens,
    const void* block_table,
    int bsz,
    int pages_per_seq,
    int token_dim,
    int k_bits,
    int v_bits,
    int page_size,
    int bonus_len,
    float compand_a,
    void* stream)
{
    try {
        berth_attn_clear_error();
        berth_attn_check(
            k_in && k_scales && k_out && v_in && v_scales && v_out && cache_seqlens && block_table,
            "dequant_cache_paged_window: null pointer");
        berth_attn_check(page_size == CQ_PAGE_SIZE, "dequant_cache_paged_window: page_size must be 256");
        berth_attn_check(
            k_bits == 3 && v_bits == 3,
            "dequant_cache_paged_window: only cq3 K and V (bits 3,3) is linked");
        berth_attn_check(
            bsz > 0 && pages_per_seq > 0,
            "dequant_cache_paged_window: bsz and pages_per_seq must be positive");
        berth_attn_check(
            token_dim > 0 && token_dim % 32 == 0,
            "dequant_cache_paged_window: token_dim must be a positive multiple of 32");
        berth_attn_check(bonus_len >= 0, "dequant_cache_paged_window: bonus_len must be non-negative");

        // q_cache.cu:396-408. groups of 32, four groups per warp.
        int groups_per_token = token_dim / 32;
        int chunks_per_token = ceil_div_int(groups_per_token, 4);
        int chunks_per_seq = pages_per_seq * page_size * chunks_per_token;
        int num_blocks = ceil_div_int(chunks_per_seq, MAX_WARPS);
        int num_tb = ceil_div_int(num_blocks, ITER_PER_TB);
        int num_threads = 32 * chunks_per_seq;
        if (num_threads > 32 * MAX_WARPS) {
            num_threads = 32 * MAX_WARPS;
        }

        dim3 blocks(num_tb, bsz);
        dim3 threads(num_threads);

        const uint32_t* k_in_ptr = (const uint32_t*) k_in;
        const half* k_scales_ptr = (const half*) k_scales;
        half* k_out_ptr = (half*) k_out;
        const uint32_t* v_in_ptr = (const uint32_t*) v_in;
        const half* v_scales_ptr = (const half*) v_scales;
        half* v_out_ptr = (half*) v_out;
        const uint32_t* seqlens_ptr = (const uint32_t*) cache_seqlens;
        const uint32_t* table_ptr = (const uint32_t*) block_table;
        int pages = pages_per_seq;
        int groups = groups_per_token;
        int chunks = chunks_per_token;
        int sliding_window = -1;
        float ca = compand_a;
        int compact_out = 1;
        int bonus = bonus_len;

        auto kernel = berth_attn_prefill_dequant::dequant_cache_paged_kernel<3, 3>;
        void* kernel_args[] = {
            &k_in_ptr,
            &k_scales_ptr,
            &k_out_ptr,
            &v_in_ptr,
            &v_scales_ptr,
            &v_out_ptr,
            &seqlens_ptr,
            &table_ptr,
            &pages,
            &groups,
            &chunks,
            &sliding_window,
            &ca,
            &compact_out,
            &bonus,
        };
        berth_cuda_ok(
            cudaLaunchKernel((void*) kernel, blocks, threads, kernel_args, 0, (cudaStream_t) stream),
            "dequant_cache_paged_window");
        berth_cuda_ok(cudaGetLastError(), "dequant_cache_paged_window");
        return 0;
    } catch (const std::exception& err) {
        berth_attn_set_error(err.what());
        return -1;
    }
}
