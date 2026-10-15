// One thread touches the first and last byte of a 64 KiB dynamic allocation.
// Ampere refuses that launch until MAX_DYNAMIC_SHARED_SIZE_BYTES is raised
// above the 48 KiB default.

extern "C" __global__ void touch_dyn_smem(int *out) {
    extern __shared__ unsigned char smem[];
    if (threadIdx.x == 0) {
        smem[0] = 1;
        smem[65535] = 2;
        *out = (int)smem[0] + (int)smem[65535];
    }
}
