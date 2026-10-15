// Runtime-API launch used to prove a captured graph can see a kernel this
// crate did not launch through berth_cuda::Function. The host entry point is
// linked into a shared library against CUDA 13.3's libcudart.

#include <cuda_runtime.h>

extern "C" __global__ void berth_fill_runtime(int *out, int value) {
    *out = value;
}

extern "C" void berth_launch_fill_runtime(unsigned long long out, int value, void *stream) {
    berth_fill_runtime<<<1, 1, 0, (cudaStream_t)stream>>>((int *)out, value);
}
