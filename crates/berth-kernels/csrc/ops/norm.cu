// RMSNorm for the 27B decode norms.
//
// The fork's device kernel is rms_norm_kernel in exllamav3_ext/norm.cu. That
// file includes ATen before the kernel, and norm.cuh is only the ATen
// declaration, so this TU cannot include it. The numeric path below follows
// that kernel for the two launches the decode actually makes:
//
//   rms_norm_kernel<0, float, half, bfloat16, float>  RES_NONE
//   rms_norm_kernel<2, float, half, bfloat16, float>  RES_IN
//
// fp16 weights are linked as well (same kernel, weight_t = half). Reduction,
// the fma sum of squares, rsqrtf(mean + eps) * scale, (x * (w + bias)) * rmf,
// and __float2half_rn stores match the fork. Block size is
// min(1024, round_up(dim/4, 32)), and dim must be a multiple of 4.
// w_groups is 1 and span_heads is false on these norms, so the weight is one
// row of `dim` elements.

#include "check.cuh"
#include "ops/ops_state.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <type_traits>

#include "util.h"
#include "util.cuh"

namespace {

constexpr int kNormThreads = 1024;

__device__ inline float reduce_dyn(float sum, int warp_id, int lane_id)
{
    __shared__ float sums[32];
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    int num_warps = blockDim.x / 32;
    if (num_warps == 1) return sum;
    if (lane_id == 0) sums[warp_id] = sum;
    __syncthreads();
    sum = lane_id < num_warps ? sums[lane_id] : 0.0f;
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    return sum;
}

__device__ inline float sum_sq4(float lsum, const float4& f4)
{
    lsum = fma(f4.x, f4.x, lsum);
    lsum = fma(f4.y, f4.y, lsum);
    lsum = fma(f4.z, f4.z, lsum);
    lsum = fma(f4.w, f4.w, lsum);
    return lsum;
}

template <int res_mode, typename weight_t>
__global__ __launch_bounds__(kNormThreads)
void rms_norm_kernel(
    const float* __restrict__ x,
    const weight_t* __restrict__ w,
    half* __restrict__ y,
    float* __restrict__ r,
    const float epsilon,
    const int dim,
    const float constant_bias,
    const float constant_scale)
{
    static_assert(res_mode == 0 || res_mode == 2, "rms_norm res_mode");
    constexpr bool weight_bf16 = std::is_same_v<weight_t, __nv_bfloat16>;
    static_assert(weight_bf16 || std::is_same_v<weight_t, half>, "rms_norm weight");

    int t = threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row = blockIdx.x;
    const size_t row_off = (size_t) row * (size_t) dim;
    int columns = dim / 4;

    auto load_w = [&](float4& w4, int column) {
        if constexpr (weight_bf16)
        {
            bfloat164 h4;
            READ64(h4, ((const bfloat164*) w) + column);
            w4.x = __bfloat162float(__low2bfloat16(h4.x));
            w4.y = __bfloat162float(__high2bfloat16(h4.x));
            w4.z = __bfloat162float(__low2bfloat16(h4.y));
            w4.w = __bfloat162float(__high2bfloat16(h4.y));
        }
        else
        {
            half4 h4;
            READ64(h4, ((const half4*) w) + column);
            w4.x = __half2float(__low2half(h4.x));
            w4.y = __half2float(__high2half(h4.x));
            w4.z = __half2float(__low2half(h4.y));
            w4.w = __half2float(__high2half(h4.y));
        }
        if (constant_bias != 0.0f)
        {
            w4.x += constant_bias;
            w4.y += constant_bias;
            w4.z += constant_bias;
            w4.w += constant_bias;
        }
    };

    auto apply_store = [&](float4& x4, int column, float rmf) {
        float4 w4;
        load_w(w4, column);
        x4.x = x4.x * w4.x * rmf;
        x4.y = x4.y * w4.y * rmf;
        x4.z = x4.z * w4.z * rmf;
        x4.w = x4.w * w4.w * rmf;
        half4 h4(
            __halves2half2(__float2half_rn(x4.x), __float2half_rn(x4.y)),
            __halves2half2(__float2half_rn(x4.z), __float2half_rn(x4.w)));
        WRITE64(((half4*) (y + row_off)) + column, h4);
    };

    auto add_resid = [&](float4& x4, int column) {
        float4 r4;
        READ128(r4, ((const float4*) (r + row_off)) + column);
        x4.x += r4.x;
        x4.y += r4.y;
        x4.z += r4.z;
        x4.w += r4.w;
        WRITE128(((float4*) (r + row_off)) + column, x4);
    };

    if (columns <= blockDim.x)
    {
        float4 x4 = {};
        float sum = 0.0f;
        if (t < columns)
        {
            READ128(x4, ((const float4*) (x + row_off)) + t);
            if constexpr (res_mode == 2) add_resid(x4, t);
            sum = sum_sq4(sum, x4);
        }
        sum = reduce_dyn(sum, warp_id, lane_id);
        float rmf = rsqrtf(sum / (float) dim + epsilon) * constant_scale;
        if (t < columns) apply_store(x4, t, rmf);
    }
    else
    {
        float sum = 0.0f;
        for (int column = t; column < columns; column += blockDim.x)
        {
            float4 x4;
            READ128(x4, ((const float4*) (x + row_off)) + column);
            if constexpr (res_mode == 2) add_resid(x4, column);
            sum = sum_sq4(sum, x4);
        }
        sum = reduce_dyn(sum, warp_id, lane_id);
        float rmf = rsqrtf(sum / (float) dim + epsilon) * constant_scale;
        for (int column = t; column < columns; column += blockDim.x)
        {
            float4 x4;
            // READ128's macro already ends in a semicolon, so the branches
            // need braces or the extra statement splits the if from the else.
            if constexpr (res_mode == 2) {
                READ128(x4, ((const float4*) (r + row_off)) + column);
            } else {
                READ128(x4, ((const float4*) (x + row_off)) + column);
            }
            apply_store(x4, column, rmf);
        }
    }
}

const char* weight_name(int weight_bf16)
{
    return weight_bf16 ? "bfloat16" : "half";
}

template <int res_mode, typename weight_t>
void launch_rms(
    const void* x,
    const void* w,
    void* y,
    void* r,
    int rows,
    int dim,
    float epsilon,
    float constant_bias,
    float constant_scale,
    int weight_bf16,
    cudaStream_t stream)
{
    int threads = MIN(kNormThreads, CEIL_DIVIDE(dim / 4, 32) * 32);
    rms_norm_kernel<res_mode, weight_t><<<rows, threads, 0, stream>>>(
        (const float*) x,
        (const weight_t*) w,
        (half*) y,
        (float*) r,
        epsilon,
        dim,
        constant_bias,
        constant_scale);
    berth_cuda_ok(cudaGetLastError(), "rms_norm launch");
    berth_ops_set_launch(
        std::string("rms_norm_kernel<") + (res_mode == 2 ? "2" : "0")
        + ",float,half," + weight_name(weight_bf16) + ",float>"
        + " block=" + std::to_string(threads)
        + " grid=" + std::to_string(rows));
}

void check_rms(const void* x, const void* w, void* y, void* r, int rows, int dim, int res_mode, int weight_bf16)
{
    if (!x || !w || !y)
        throw std::runtime_error("rms_norm: null device pointer");
    if (res_mode == 2 && !r)
        throw std::runtime_error("rms_norm: RES_IN requires a residual pointer");
    if (rows < 1 || dim < 1)
        throw std::runtime_error("rms_norm: rows and dim must be positive");
    if (dim % 4 != 0)
        throw std::runtime_error("rms_norm: dim must be a multiple of 4");
    if (weight_bf16 != 0 && weight_bf16 != 1)
        throw std::runtime_error("rms_norm: weight_bf16 must be 0 or 1");
}

}  // namespace

extern "C" int berth_ops_rms_norm(
    const void* x,
    const void* weight,
    void* y,
    int rows,
    int dim,
    float epsilon,
    float constant_bias,
    float constant_scale,
    int weight_bf16,
    void* stream)
{
    try
    {
        berth_ops_clear();
        check_rms(x, weight, y, nullptr, rows, dim, 0, weight_bf16);
        if (weight_bf16)
            launch_rms<0, __nv_bfloat16>(
                x, weight, y, nullptr, rows, dim, epsilon, constant_bias, constant_scale, 1,
                (cudaStream_t) stream);
        else
            launch_rms<0, half>(
                x, weight, y, nullptr, rows, dim, epsilon, constant_bias, constant_scale, 0,
                (cudaStream_t) stream);
        return 0;
    }
    catch (const std::exception& err)
    {
        berth_ops_set_error(err.what());
        cudaGetLastError();
        return -1;
    }
}

extern "C" int berth_ops_rms_norm_res_in(
    const void* x,
    const void* weight,
    void* y,
    void* residual,
    int rows,
    int dim,
    float epsilon,
    float constant_bias,
    float constant_scale,
    int weight_bf16,
    void* stream)
{
    try
    {
        berth_ops_clear();
        check_rms(x, weight, y, residual, rows, dim, 2, weight_bf16);
        if (weight_bf16)
            launch_rms<2, __nv_bfloat16>(
                x, weight, y, residual, rows, dim, epsilon, constant_bias, constant_scale, 1,
                (cudaStream_t) stream);
        else
            launch_rms<2, half>(
                x, weight, y, residual, rows, dim, epsilon, constant_bias, constant_scale, 0,
                (cudaStream_t) stream);
        return 0;
    }
    catch (const std::exception& err)
    {
        berth_ops_set_error(err.what());
        cudaGetLastError();
        return -1;
    }
}
