// sm_86 probes. Each kernel is one instruction from doc/research_3090_decode.md
// sections 1–2, copied from the exllamav3 asm that actually ships.
// Compile with: nvcc -arch=sm_86 -cubin
//
// PROBE lines are the contract. crates/berth-ptx fails if this list drifts.

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ != 860
#error berth sm_86 probes must be compiled with -arch=sm_86
#endif

#include <stdint.h>

// PROBE mma_m16n8k8_f16
__global__ void probe_mma_m16n8k8_f16(unsigned* s) {
    unsigned c0 = s[0], c1 = s[1], a0 = s[2], a1 = s[3], b0 = s[4];
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3}, {%4}, {%5,%6};\n"
        : "=r"(c0), "=r"(c1) : "r"(a0), "r"(a1), "r"(b0), "r"(c0), "r"(c1));
    s[0] = c0; s[1] = c1;
}

// PROBE mma_m16n8k8_f32
__global__ void probe_mma_m16n8k8_f32(float* c, unsigned* ab) {
    float c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3];
    unsigned a0 = ab[0], a1 = ab[1], b0 = ab[2];
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
        : "=f"(c0), "=f"(c1), "=f"(c2), "=f"(c3)
        : "r"(a0), "r"(a1), "r"(b0), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
    c[0] = c0; c[1] = c1; c[2] = c2; c[3] = c3;
}

// PROBE mma_m16n8k16_f16
__global__ void probe_mma_m16n8k16_f16(unsigned* s) {
    unsigned c0 = s[0], c1 = s[1];
    unsigned a0 = s[2], a1 = s[3], a2 = s[4], a3 = s[5], b0 = s[6], b1 = s[7];
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
        : "=r"(c0), "=r"(c1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1));
    s[0] = c0; s[1] = c1;
}

// PROBE mma_m16n8k16_f32
__global__ void probe_mma_m16n8k16_f32(float* c, unsigned* ab) {
    float c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3];
    unsigned a0 = ab[0], a1 = ab[1], a2 = ab[2], a3 = ab[3], b0 = ab[4], b1 = ab[5];
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c0), "=f"(c1), "=f"(c2), "=f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
    c[0] = c0; c[1] = c1; c[2] = c2; c[3] = c3;
}

// PROBE mma_m16n8k32_s8
__global__ void probe_mma_m16n8k32_s8(int* c, unsigned* ab) {
    int d0 = c[0], d1 = c[1], d2 = c[2], d3 = c[3];
    unsigned a0 = ab[0], a1 = ab[1], a2 = ab[2], a3 = ab[3], b0 = ab[4], b1 = ab[5];
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(d0), "r"(d1), "r"(d2), "r"(d3));
    c[0] = d0; c[1] = d1; c[2] = d2; c[3] = d3;
}

// PROBE mma_m8n8k4_f32
// Ampere 上不是最佳路徑。exllamav3 ptx.cuh 標成 don't use，sm_86 仍要組得過。
__global__ void probe_mma_m8n8k4_f32(float* c, unsigned* ab) {
    float c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3];
    float c4 = c[4], c5 = c[5], c6 = c[6], c7 = c[7];
    unsigned a0 = ab[0], a1 = ab[1], b0 = ab[2], b1 = ab[3];
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
        "{%12,%13,%14,%15,%16,%17,%18,%19};\n"
        : "=f"(c0), "=f"(c1), "=f"(c2), "=f"(c3), "=f"(c4), "=f"(c5), "=f"(c6), "=f"(c7)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3), "f"(c4), "f"(c5), "f"(c6), "f"(c7));
    c[0] = c0; c[1] = c1; c[2] = c2; c[3] = c3;
    c[4] = c4; c[5] = c5; c[6] = c6; c[7] = c7;
}

// PROBE ldmatrix_x4
__global__ void probe_ldmatrix_x4(unsigned* s) {
    __shared__ __align__(16) unsigned buf[32];
    unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(buf));
    unsigned a0, a1, a2, a3;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(addr));
    s[0] = a0 ^ a1 ^ a2 ^ a3;
}

// PROBE ldmatrix_x4_trans
__global__ void probe_ldmatrix_x4_trans(unsigned* s) {
    __shared__ __align__(16) unsigned buf[32];
    unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(buf));
    unsigned a0, a1, a2, a3;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(addr));
    s[0] = a0 ^ a1 ^ a2 ^ a3;
}

// PROBE cp_async_cg
__global__ void probe_cp_async_cg(const void* g, unsigned* s) {
    __shared__ __align__(16) char buf[16];
    unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(buf));
    const int nbytes = 16;
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" :: "r"(smem), "l"(g), "n"(nbytes));
    s[0] = smem;
}

// PROBE cp_async_commit_group
__global__ void probe_cp_async_commit_group(const void* g) {
    __shared__ __align__(16) char buf[16];
    unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(buf));
    const int nbytes = 16;
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" :: "r"(smem), "l"(g), "n"(nbytes));
    asm volatile("cp.async.commit_group;\n" ::);
}

// PROBE cp_async_wait_group
__global__ void probe_cp_async_wait_group(const void* g) {
    __shared__ __align__(16) char buf[16];
    unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(buf));
    const int nbytes = 16;
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" :: "r"(smem), "l"(g), "n"(nbytes));
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group %0;\n" :: "n"(0));
}

// PROBE dp4a
__global__ void probe_dp4a(unsigned* s) {
    unsigned d, a = s[0], b = s[1], c = s[2];
    asm volatile("dp4a.s32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    s[0] = d;
}

// PROBE tanh_approx_f32
__global__ void probe_tanh_approx_f32(float* s) {
    float r, x = s[0];
    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    s[0] = r;
}

// PROBE lop3
__global__ void probe_lop3(unsigned* s) {
    unsigned x = s[0];
    asm volatile("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x));
    s[0] = x;
}

// PROBE shf
__global__ void probe_shf(unsigned* s) {
    unsigned d, lo = s[0], hi = s[1];
    asm volatile("shf.r.wrap.b32 %0, %1, %2, 16;" : "=r"(d) : "r"(lo), "r"(hi));
    s[0] = d;
}

// PROBE bfe_u32
__global__ void probe_bfe_u32(unsigned* s) {
    unsigned d, src = s[0];
    asm volatile("bfe.u32 %0, %1, 8, 16;" : "=r"(d) : "r"(src));
    s[0] = d;
}

// PROBE bfe_u64
__global__ void probe_bfe_u64(unsigned long long* s) {
    unsigned long long d, src = s[0];
    asm volatile("bfe.u64 %0, %1, 8, 16;" : "=l"(d) : "l"(src));
    s[0] = d;
}

// PROBE prmt
// 研究表列了 prmt。初始化當下 exllamav3_ext 的 .cu/.cuh 沒有 prmt 字面。
// 這條只證明 sm_86 的 ptxas 收得了這條指令。
__global__ void probe_prmt(unsigned* s) {
    unsigned d, a = s[0], b = s[1], c = s[2];
    asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    s[0] = d;
}

// PROBE ld_global_acquire
__global__ void probe_ld_global_acquire(int* lock, int* s) {
    int state;
    asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
    s[0] = state;
}

// PROBE st_global_release
__global__ void probe_st_global_release(unsigned* p, unsigned v) {
    asm volatile("st.global.release.sys.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

// PROBE red_relaxed
__global__ void probe_red_relaxed(int* lock, int val) {
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" :: "l"(lock), "r"(val));
}

// PROBE fence_acq_rel
__global__ void probe_fence_acq_rel() {
    asm volatile("fence.acq_rel.gpu;\n");
}

// PROBE ld_global_cv
__global__ void probe_ld_global_cv(unsigned* p, unsigned* s) {
    unsigned v;
    asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(v) : "l"(p));
    s[0] = v;
}

// PROBE st_global_wt
__global__ void probe_st_global_wt(unsigned* p, unsigned v) {
    asm volatile("st.global.wt.u32 [%0], %1;" :: "l"(p), "r"(v));
}

// PROBE nanosleep
__global__ void probe_nanosleep() {
    asm volatile("nanosleep.u32 32;");
}

// PROBE laneid
__global__ void probe_laneid(unsigned* s) {
    unsigned lane;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(lane));
    s[0] = lane;
}

// PROBE sm86_mad_lo
// codebook.cuh 的 __CUDA_ARCH__ == 860 路徑。本檔已用 #error 鎖死 860，
// 所以這條 mad 一定要進 cubin。
__global__ void probe_sm86_mad_lo(unsigned* s) {
    unsigned r, x = s[0];
    asm volatile(
        "{ .reg .u32 z,t;"
        "  mov.u32 t, %laneid;"
        "  sub.u32 z, t, t;"
        "  mad.lo.u32 %0, %1, %2, z;"
        "}"
        : "=r"(r) : "r"(x), "n"(0x83DCD12Du));
    s[0] = r;
}
