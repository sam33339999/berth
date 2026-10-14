//! Instruction catalog for the RTX 3090 (sm_86).
//!
//! `PROBE` and `REJECT` comments in `ptx/*.cu` are the strings ptxas is asked to
//! accept or refuse. This table records, for each of those ids, whether
//! cuda-oxide (NVIDIA/cuda-rust, surveyed at `bc46b1f`) has an intrinsic.
//! The survey lives in exllamav3 `doc/research_3090_decode.md` sections 1–2.

/// How cuda-oxide stood on 2026-10-12. Not a promise about a later commit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Oxide {
    /// A named intrinsic exists (`register_mma`, `wmma`, `cp.async`, atomics, …).
    Intrinsic,
    /// No dedicated intrinsic. The compiler may synthesize it. Not verified.
    Unverified,
    /// Not in cuda-oxide. A kernel that needs it uses `ptx_asm!`.
    Missing,
    /// Ordinary ALU plus an `__CUDA_ARCH__ == 860` switch. cuda-oxide expresses
    /// that switch with `--device-cfg` and `#[cfg]`, kept in step with `--arch`.
    DeviceCfg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Asm {
    /// `nvcc -arch=sm_86` must produce a cubin.
    Assemble,
    /// ptxas for sm_86 must reject the instruction.
    Reject,
}

#[derive(Clone, Copy, Debug)]
pub struct Op {
    pub id: &'static str,
    pub ptx: &'static str,
    pub asm: Asm,
    pub oxide: Oxide,
}

pub static OPS: &[Op] = &[
    Op { id: "mma_m16n8k8_f16", ptx: "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16", asm: Asm::Assemble, oxide: Oxide::Intrinsic },
    Op { id: "mma_m16n8k8_f32", ptx: "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32", asm: Asm::Assemble, oxide: Oxide::Intrinsic },
    Op { id: "mma_m16n8k16_f16", ptx: "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16", asm: Asm::Assemble, oxide: Oxide::Intrinsic },
    Op { id: "mma_m16n8k16_f32", ptx: "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", asm: Asm::Assemble, oxide: Oxide::Intrinsic },
    Op { id: "mma_m16n8k32_s8", ptx: "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32", asm: Asm::Assemble, oxide: Oxide::Intrinsic },
    Op { id: "mma_m8n8k4_f32", ptx: "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32", asm: Asm::Assemble, oxide: Oxide::Missing },
    Op { id: "ldmatrix_x4", ptx: "ldmatrix.sync.aligned.m8n8.x4.shared.b16", asm: Asm::Assemble, oxide: Oxide::Intrinsic },
    Op { id: "ldmatrix_x4_trans", ptx: "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16", asm: Asm::Assemble, oxide: Oxide::Intrinsic },
    Op { id: "cp_async_cg", ptx: "cp.async.cg.shared.global", asm: Asm::Assemble, oxide: Oxide::Intrinsic },
    Op { id: "cp_async_commit_group", ptx: "cp.async.commit_group", asm: Asm::Assemble, oxide: Oxide::Intrinsic },
    Op { id: "cp_async_wait_group", ptx: "cp.async.wait_group", asm: Asm::Assemble, oxide: Oxide::Intrinsic },
    Op { id: "dp4a", ptx: "dp4a.s32.u32", asm: Asm::Assemble, oxide: Oxide::Intrinsic },
    Op { id: "tanh_approx_f32", ptx: "tanh.approx.f32", asm: Asm::Assemble, oxide: Oxide::Intrinsic },
    Op { id: "lop3", ptx: "lop3.b32", asm: Asm::Assemble, oxide: Oxide::Unverified },
    Op { id: "shf", ptx: "shf.r.wrap.b32", asm: Asm::Assemble, oxide: Oxide::Unverified },
    Op { id: "bfe_u32", ptx: "bfe.u32", asm: Asm::Assemble, oxide: Oxide::Unverified },
    Op { id: "bfe_u64", ptx: "bfe.u64", asm: Asm::Assemble, oxide: Oxide::Unverified },
    Op { id: "prmt", ptx: "prmt.b32", asm: Asm::Assemble, oxide: Oxide::Intrinsic },
    Op { id: "ld_global_acquire", ptx: "ld.global.acquire.gpu.b32", asm: Asm::Assemble, oxide: Oxide::Intrinsic },
    Op { id: "st_global_release", ptx: "st.global.release.sys.u32", asm: Asm::Assemble, oxide: Oxide::Intrinsic },
    Op { id: "red_relaxed", ptx: "red.relaxed.gpu.global.add.s32", asm: Asm::Assemble, oxide: Oxide::Intrinsic },
    Op { id: "fence_acq_rel", ptx: "fence.acq_rel.gpu", asm: Asm::Assemble, oxide: Oxide::Intrinsic },
    Op { id: "ld_global_cv", ptx: "ld.global.cv.u32", asm: Asm::Assemble, oxide: Oxide::Missing },
    Op { id: "st_global_wt", ptx: "st.global.wt.u32", asm: Asm::Assemble, oxide: Oxide::Missing },
    Op { id: "nanosleep", ptx: "nanosleep.u32", asm: Asm::Assemble, oxide: Oxide::Intrinsic },
    Op { id: "laneid", ptx: "mov.u32 %laneid", asm: Asm::Assemble, oxide: Oxide::Intrinsic },
    Op { id: "sm86_mad_lo", ptx: "mad.lo.u32", asm: Asm::Assemble, oxide: Oxide::DeviceCfg },
    Op { id: "add_rn_f32_f16", ptx: "add.rn.f32.f16", asm: Asm::Reject, oxide: Oxide::Missing },
];

pub fn assemble_ids() -> impl Iterator<Item = &'static str> {
    OPS.iter().filter(|op| op.asm == Asm::Assemble).map(|op| op.id)
}

pub fn reject_ids() -> impl Iterator<Item = &'static str> {
    OPS.iter().filter(|op| op.asm == Asm::Reject).map(|op| op.id)
}
