// Embedding lookup for model.language_model.embed_tokens.
//
// The checkpoint weight is BF16 [248320, 5120]. modules/embedding.py loads it
// with allow_bf16 (no fp16 conversion) into nn.Embedding, whose forward
// gathers rows in that dtype. qwen3_5.Embedding leaves out_dtype at the
// default torch.float, and to2 then widens the gather to fp32. normalize is
// false and multiplier is 1, so there is no extra scale. Indexed / multimodal
// embeddings are not on the text decode path.
//
// __bfloat162float is an exact widen (mantissa padded with zeros), the same
// bits as torch `.to(torch.float)` / F.embedding(...).float(). There is no
// exllamav3 binding for this op. `ids` must already lie in range; this kernel
// does not mask them.

#include "check.cuh"
#include "ops/ops_state.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace {

constexpr int kThreads = 256;

__global__ void embed_bf16_f32_kernel(
    const __nv_bfloat16* __restrict__ weight,
    const int* __restrict__ ids,
    float* __restrict__ out,
    int hidden)
{
    int token = blockIdx.x;
    const __nv_bfloat16* row = weight + (size_t) ids[token] * (size_t) hidden;
    float* dst = out + (size_t) token * (size_t) hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
        dst[i] = __bfloat162float(row[i]);
}

}  // namespace

extern "C" int berth_ops_embed_tokens(
    const void* weight,
    const int* ids,
    void* out,
    int hidden,
    int n,
    void* stream)
{
    try
    {
        berth_ops_clear();
        if (!weight || !ids || !out)
            throw std::runtime_error("embed_tokens: null device pointer");
        if (hidden < 1 || n < 1)
            throw std::runtime_error("embed_tokens: hidden and n must be positive");
        embed_bf16_f32_kernel<<<n, kThreads, 0, (cudaStream_t) stream>>>(
            (const __nv_bfloat16*) weight, ids, (float*) out, hidden);
        berth_cuda_ok(cudaGetLastError(), "embed_tokens launch");
        berth_ops_set_launch(
            "embed_bf16_f32 block=256 grid=" + std::to_string(n));
        return 0;
    }
    catch (const std::exception& err)
    {
        berth_ops_set_error(err.what());
        cudaGetLastError();
        return -1;
    }
}
