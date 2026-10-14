// REJECT add_rn_f32_f16
// hgemm_f16acc.cu gates this on __CUDA_ARCH__ >= 1000. ptxas for sm_86 must refuse it.

__global__ void reject_add_rn_f32_f16(float* a, unsigned h) {
    asm volatile(
        "{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %2; "
        "add.rn.f32.f16 %0, lo, %0; add.rn.f32.f16 %1, hi, %1; }"
        : "+f"(a[0]), "+f"(a[1]) : "r"(h));
}
