extern "C" __global__ void fill_i32(int *out, int value) {
    *out = value;
}
