// Device copies the MTP prefill uses. berth-cuda's ByteBuf has host copies
// only, so the shifted-hidden prefix and the embedding/hidden concat stay
// on the stream here.
//
// memcpy is cudaMemcpyAsync DeviceToDevice. The concat writes
// out[row, 0:H) = embedding and out[row, H:2H) = hidden norm, which is
// torch.cat((x, y), dim=-1) in Qwen3_5MTPInputLayer.forward.

#include "check.cuh"
#include "ops/ops_state.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace {

constexpr int kThreads = 256;

__global__ void cat_f16_last_kernel(
    const __half* __restrict__ embedding,
    const __half* __restrict__ hidden,
    __half* __restrict__ out,
    int width)
{
    int row = blockIdx.x;
    const __half* emb = embedding + (size_t) row * (size_t) width;
    const __half* hid = hidden + (size_t) row * (size_t) width;
    __half* dst = out + (size_t) row * (size_t) width * 2;
    for (int i = threadIdx.x; i < width; i += blockDim.x)
    {
        dst[i] = emb[i];
        dst[width + i] = hid[i];
    }
}

}  // namespace

extern "C" int berth_ops_memcpy_d2d(
    void* dst,
    const void* src,
    unsigned long long bytes,
    void* stream)
{
    try
    {
        berth_ops_clear();
        if (!dst || !src)
            throw std::runtime_error("memcpy_d2d: null device pointer");
        if (bytes == 0)
            throw std::runtime_error("memcpy_d2d: byte count is 0");
        berth_cuda_ok(
            cudaMemcpyAsync(dst, src, (size_t) bytes, cudaMemcpyDeviceToDevice, (cudaStream_t) stream),
            "memcpy_d2d");
        berth_ops_set_launch(
            "cudaMemcpyAsync DeviceToDevice bytes=" + std::to_string(bytes));
        return 0;
    }
    catch (const std::exception& err)
    {
        berth_ops_set_error(err.what());
        cudaGetLastError();
        return -1;
    }
}

extern "C" int berth_ops_cat_f16_last(
    const void* embedding,
    const void* hidden,
    void* out,
    int rows,
    int hidden_size,
    void* stream)
{
    try
    {
        berth_ops_clear();
        if (!embedding || !hidden || !out)
            throw std::runtime_error("cat_f16_last: null device pointer");
        if (rows < 1 || hidden_size < 1)
            throw std::runtime_error("cat_f16_last: rows and hidden must be positive");
        cat_f16_last_kernel<<<rows, kThreads, 0, (cudaStream_t) stream>>>(
            (const __half*) embedding, (const __half*) hidden, (__half*) out, hidden_size);
        berth_cuda_ok(cudaGetLastError(), "cat_f16_last launch");
        berth_ops_set_launch(
            "cat_f16_last block=256 grid=" + std::to_string(rows));
        return 0;
    }
    catch (const std::exception& err)
    {
        berth_ops_set_error(err.what());
        cudaGetLastError();
        return -1;
    }
}
