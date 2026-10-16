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
//
// The MTP input layer passes `out_dtype=torch.half` (`qwen3_5_mtp.py`).
// `to2` is `.to(torch.half)` (`util/tensor.py`). That cast on a bf16 value
// is bf16 -> fp32 (exact) then fp32 -> fp16 round-to-nearest-even, which is
// `__float2half(__bfloat162float(v))`. The f32 kernel above is unchanged;
// `embed_bf16_f16_kernel` is only that extra rounding.

#include "check.cuh"
#include "ops/ops_state.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
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

// bf16 weight row -> fp16. Equals torch `.half()` on bf16: the widen is
// exact and `__float2half` is round-to-nearest-even, matching PyTorch.
__global__ void embed_bf16_f16_kernel(
    const __nv_bfloat16* __restrict__ weight,
    const int* __restrict__ ids,
    __half* __restrict__ out,
    int hidden)
{
    int token = blockIdx.x;
    const __nv_bfloat16* row = weight + (size_t) ids[token] * (size_t) hidden;
    __half* dst = out + (size_t) token * (size_t) hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
        dst[i] = __float2half(__bfloat162float(row[i]));
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

extern "C" int berth_ops_embed_tokens_f16(
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
            throw std::runtime_error("embed_tokens_f16: null device pointer");
        if (hidden < 1 || n < 1)
            throw std::runtime_error("embed_tokens_f16: hidden and n must be positive");
        embed_bf16_f16_kernel<<<n, kThreads, 0, (cudaStream_t) stream>>>(
            (const __nv_bfloat16*) weight, ids, (__half*) out, hidden);
        berth_cuda_ok(cudaGetLastError(), "embed_tokens_f16 launch");
        berth_ops_set_launch(
            "embed_bf16_f16 block=256 grid=" + std::to_string(n));
        return 0;
    }
    catch (const std::exception& err)
    {
        berth_ops_set_error(err.what());
        cudaGetLastError();
        return -1;
    }
}
