#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

// Host-side check. The fork's `cuda_check` aborts the process; the shim
// returns the message through the C ABI instead.
inline void berth_cuda_ok(cudaError_t err, const char* what)
{
    if (err == cudaSuccess) return;
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}
