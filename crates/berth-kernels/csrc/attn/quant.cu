// quant_cache_paged_gr for cq3 K and V. q_cache.cuh pulls ATen, so this file
// includes only cache/q_cache_kernels.cuh (lmq.cuh underneath) and launches
// quant_cache_paged_kernel<3,3>. The header instantiates every bitrate pair
// from 2 to 8; the launcher still rejects anything but 3,3.
//
// The include is inside a namespace. The header has external-linkage
// __global__ symbols, and a second translation unit that includes it at
// global scope would otherwise fail the link.

#include "attn/error.h"
#include "check.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>

namespace berth_attn_quant {

#include "cache/q_cache_kernels.cuh"

}  // namespace berth_attn_quant

extern "C" int berth_attn_quant_cache_paged(
    const void* k_in,
    void* k_out,
    void* k_scales,
    const void* v_in,
    void* v_out,
    void* v_scales,
    const void* cache_seqlens,
    const void* block_table,
    int bsz,
    int seq_len,
    int blocks_per_seq,
    int token_dim,
    int k_bits,
    int v_bits,
    int page_size,
    float compand_a,
    int in_contiguous,
    void* stream)
{
    try {
        berth_attn_clear_error();
        berth_attn_check(
            k_in && k_out && k_scales && v_in && v_out && v_scales && cache_seqlens && block_table,
            "quant_cache_paged: null pointer");
        berth_attn_check(page_size == CQ_PAGE_SIZE, "quant_cache_paged: page_size must be 256");
        berth_attn_check(k_bits == 3 && v_bits == 3, "quant_cache_paged: only cq3 K and V (bits 3,3) is linked");
        berth_attn_check(bsz > 0 && seq_len > 0 && blocks_per_seq > 0, "quant_cache_paged: bsz, seq_len, and blocks_per_seq must be positive");
        berth_attn_check(token_dim > 0 && token_dim % 32 == 0, "quant_cache_paged: token_dim must be a positive multiple of 32");
        berth_attn_check(in_contiguous == 0 || in_contiguous == 1, "quant_cache_paged: in_contiguous must be 0 or 1");

        // q_cache.cu: groups of 32, four groups per warp, MAX_WARPS warps per block.
        int groups_per_token = token_dim / 32;
        int chunks_per_token = ceil_div_int(groups_per_token, 4);
        int tb_per_token = ceil_div_int(chunks_per_token, MAX_WARPS);
        int tb_usage = ceil_div_int(chunks_per_token, tb_per_token);

        dim3 blocks(tb_per_token, seq_len, bsz);
        dim3 threads(32 * tb_usage);

        const half* k_in_ptr = (const half*) k_in;
        uint32_t* k_out_ptr = (uint32_t*) k_out;
        half* k_scales_ptr = (half*) k_scales;
        const half* v_in_ptr = (const half*) v_in;
        uint32_t* v_out_ptr = (uint32_t*) v_out;
        half* v_scales_ptr = (half*) v_scales;
        const uint32_t* seqlens_ptr = (const uint32_t*) cache_seqlens;
        const uint32_t* table_ptr = (const uint32_t*) block_table;
        int in_cont = in_contiguous;

        auto kernel = berth_attn_quant::quant_cache_paged_kernel_instances[k_bits - 2][v_bits - 2];
        void* kernel_args[] = {
            &k_in_ptr,
            &k_out_ptr,
            &k_scales_ptr,
            &v_in_ptr,
            &v_out_ptr,
            &v_scales_ptr,
            &seqlens_ptr,
            &table_ptr,
            &blocks_per_seq,
            &groups_per_token,
            &compand_a,
            &in_cont,
        };
        berth_cuda_ok(
            cudaLaunchKernel((void*) kernel, blocks, threads, kernel_args, 0, (cudaStream_t) stream),
            "quant_cache_paged");
        berth_cuda_ok(cudaGetLastError(), "quant_cache_paged");
        return 0;
    } catch (const std::exception& err) {
        berth_attn_set_error(err.what());
        return -1;
    }
}
