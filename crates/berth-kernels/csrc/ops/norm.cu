// RMSNorm for the 27B decode norms.
//
// The fork's device kernel is rms_norm_kernel in exllamav3_ext/norm.cu. That
// file includes ATen before the kernel, so build.rs extracts the kernel range
// into OUT_DIR/fork/norm_kernels.cuh and this file launches it as
// rms_norm_gr does, for the two launches the decode makes:
//
//   rms_norm_kernel<0, float, half, bfloat16, float>  RES_NONE
//   rms_norm_kernel<2, float, half, bfloat16, float>  RES_IN
//
// fp16 weights are linked as well (same kernel, weight_t = half). Block size
// is min(1024, round_up(dim/4, 32)), and dim must be a multiple of 4.
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

// rms_norm_kernel and its helpers are the fork's, extracted from norm.cu at
// build time (build.rs FORK_EXTRACTS). NUM_THREADS (1024) and RES_* come
// from that range too.
#include "fork/norm_kernels.cuh"

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
    // norm.cu rms_norm_gr: block sized to the row, one block per row.
    int threads = MIN(NUM_THREADS, CEIL_DIVIDE(dim / 4, 32) * 32);
    rms_norm_kernel<res_mode, float, half, weight_t, float><<<rows, threads, 0, stream>>>(
        (const float*) x,
        (const weight_t*) w,
        (half*) y,
        (float*) r,
        epsilon,
        rows,
        dim,
        constant_bias,
        constant_scale,
        1);
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
