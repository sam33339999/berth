// Batched GDN rewind. Kernel bodies are gdn.cu 1923–1960 (FORK_EXTRACTS,
// fork/gdn_rewind_kernels.cuh). ConvRewindJob / StateRewindJob match
// gdn.cuh 222–247. That header includes ATen, so the POD layout is
// declared here. Grid and block match batched_conv_rewind /
// batched_state_rewind (gdn.cu 1962–2010) without the CUDAGuard: the
// current device and the caller's stream, same as the other GDN launchers.
//
// REWIND_* and the batch structs sit just above the kernels in gdn.cu and
// are not part of the extract. CONV1D_MAX_K is 16.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>

#include "check.cuh"
#include "util.h"

using bfloat16 = __nv_bfloat16;

#define CONV1D_MAX_K 16
#define REWIND_MAX_JOBS 64
#define REWIND_CONV_THREADS 256
#define REWIND_STATE_THREADS 256

struct ConvRewindJob {
    uintptr_t src;
    uintptr_t dst;
    int dim;
    int cdim;
    int stride;
};

struct StateRewindJob {
    uintptr_t src;
    uintptr_t dst;
    int64_t num_elements;
};

struct ConvRewindJobBatch {
    ConvRewindJob jobs[REWIND_MAX_JOBS];
    int num_jobs;
};

struct StateRewindJobBatch {
    StateRewindJob jobs[REWIND_MAX_JOBS];
    int num_jobs;
};

static_assert(sizeof(ConvRewindJob) == 32, "ConvRewindJob must match gdn.cuh");
static_assert(offsetof(ConvRewindJob, src) == 0, "src");
static_assert(offsetof(ConvRewindJob, dst) == 8, "dst");
static_assert(offsetof(ConvRewindJob, dim) == 16, "dim");
static_assert(offsetof(ConvRewindJob, cdim) == 20, "cdim");
static_assert(offsetof(ConvRewindJob, stride) == 24, "stride");
static_assert(sizeof(StateRewindJob) == 24, "StateRewindJob must match gdn.cuh");
static_assert(offsetof(StateRewindJob, src) == 0, "src");
static_assert(offsetof(StateRewindJob, dst) == 8, "dst");
static_assert(offsetof(StateRewindJob, num_elements) == 16, "num_elements");

#include "fork/gdn_rewind_kernels.cuh"

namespace {

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

void clear_sticky()
{
    cudaGetLastError();
}

void launch_conv(const ConvRewindJob* jobs, int n, cudaStream_t stream)
{
    if (n < 0)
        throw std::runtime_error("batched_conv_rewind: num_jobs is negative");
    if (n == 0) return;
    if (!jobs)
        throw std::runtime_error("batched_conv_rewind: jobs are required");

    for (int base = 0; base < n; base += REWIND_MAX_JOBS)
    {
        const int count = MIN(n - base, REWIND_MAX_JOBS);
        ConvRewindJobBatch batch{};
        batch.num_jobs = count;
        int max_dim = 0;
        for (int i = 0; i < count; ++i)
        {
            batch.jobs[i] = jobs[base + i];
            if (batch.jobs[i].cdim < 0 || batch.jobs[i].cdim > CONV1D_MAX_K)
                throw std::runtime_error("batched_conv_rewind: cdim exceeds CONV1D_MAX_K");
            if (batch.jobs[i].dim < 0)
                throw std::runtime_error("batched_conv_rewind: dim is negative");
            max_dim = MAX(max_dim, batch.jobs[i].dim);
        }
        dim3 grid(CEIL_DIVIDE(max_dim, REWIND_CONV_THREADS), count, 1);
        dim3 block(REWIND_CONV_THREADS, 1, 1);
        clear_sticky();
        batched_conv_rewind_kernel<<<grid, block, 0, stream>>>(batch);
        berth_cuda_ok(cudaGetLastError(), "batched_conv_rewind");
        g_launch = describe("batched_conv_rewind_kernel", grid, block, 0);
    }
}

void launch_state(const StateRewindJob* jobs, int n, cudaStream_t stream)
{
    if (n < 0)
        throw std::runtime_error("batched_state_rewind: num_jobs is negative");
    if (n == 0) return;
    if (!jobs)
        throw std::runtime_error("batched_state_rewind: jobs are required");

    for (int base = 0; base < n; base += REWIND_MAX_JOBS)
    {
        const int count = MIN(n - base, REWIND_MAX_JOBS);
        StateRewindJobBatch batch{};
        batch.num_jobs = count;
        int64_t max_elems = 0;
        for (int i = 0; i < count; ++i)
        {
            batch.jobs[i] = jobs[base + i];
            if (batch.jobs[i].num_elements < 0 || (batch.jobs[i].num_elements % 4) != 0)
                throw std::runtime_error("batched_state_rewind: num_elements must be a multiple of 4");
            max_elems = MAX(max_elems, batch.jobs[i].num_elements);
        }
        if (max_elems / 4 > static_cast<int64_t>(INT_MAX))
            throw std::runtime_error("batched_state_rewind: num_elements/4 does not fit in int");
        const int n4 = static_cast<int>(max_elems / 4);
        dim3 grid(CEIL_DIVIDE(n4, REWIND_STATE_THREADS), count, 1);
        dim3 block(REWIND_STATE_THREADS, 1, 1);
        clear_sticky();
        batched_state_rewind_kernel<<<grid, block, 0, stream>>>(batch);
        berth_cuda_ok(cudaGetLastError(), "batched_state_rewind");
        g_launch = describe("batched_state_rewind_kernel", grid, block, 0);
    }
}

template <typename F>
int guard(F&& fn)
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
        g_err = "gdn rewind: unknown error";
        g_launch.clear();
        return 1;
    }
}

}  // namespace

extern "C" int berth_gdn_batched_conv_rewind(
    const ConvRewindJob* jobs,
    int n,
    void* stream)
{
    return guard([&] { launch_conv(jobs, n, static_cast<cudaStream_t>(stream)); });
}

extern "C" int berth_gdn_batched_state_rewind(
    const StateRewindJob* jobs,
    int n,
    void* stream)
{
    return guard([&] { launch_state(jobs, n, static_cast<cudaStream_t>(stream)); });
}

extern "C" const char* berth_gdn_rewind_last_error(void)
{
    return g_err.c_str();
}

extern "C" const char* berth_gdn_rewind_last_launch(void)
{
    return g_launch.c_str();
}
