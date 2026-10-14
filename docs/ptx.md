# sm_86 指令

來源是 exllamav3 `doc/research_3090_decode.md` 第 1–2 節，對到 2026-10-14 的程式。目錄在 `crates/berth-ptx`，組譯探针在 `ptx/sm86_probes.cu`。

`cargo test -p berth-ptx` 做兩件事：目錄和 `// PROBE` / `// REJECT` 註解一致，以及 `nvcc -arch=sm_86` 組得過探针、拒絕 `add.rn.f32.f16`。

cuda-oxide 的覆蓋率是調查 commit `bc46b1f`（2026-10-12）的結果。3090 在它宣稱的 sm_80–sm_100a 裡。它不支援 sm_75，也不支援 ROCm。

## 組得過

| id | 指令 | cuda-oxide |
|---|---|---|
| `mma_m16n8k8_f16` | `mma.sync.aligned.m16n8k8` f16 累加 | intrinsic |
| `mma_m16n8k8_f32` | 同上，f32 累加 | intrinsic |
| `mma_m16n8k16_f16` | `m16n8k16` f16 累加 | intrinsic |
| `mma_m16n8k16_f32` | 同上，f32 累加 | intrinsic |
| `mma_m16n8k32_s8` | `m16n8k32.s32.s8.s8.s32` | intrinsic |
| `mma_m8n8k4_f32` | `m8n8k4` f16→f32。Ampere 上不是最佳路徑 | 沒有 f16 版，只有 f64 |
| `ldmatrix_x4` | `ldmatrix.sync.aligned.m8n8.x4.shared.b16` | intrinsic |
| `ldmatrix_x4_trans` | 同上，`.trans` | intrinsic |
| `cp_async_cg` | `cp.async.cg.shared.global`，長度是立即值 | intrinsic |
| `cp_async_commit_group` | `cp.async.commit_group` | intrinsic |
| `cp_async_wait_group` | `cp.async.wait_group` | intrinsic |
| `dp4a` | `dp4a.s32.u32` | intrinsic |
| `tanh_approx_f32` | `tanh.approx.f32` | intrinsic |
| `lop3` | `lop3.b32`，lut `0x6a` | 沒有專用 intrinsic，未驗證 |
| `shf` | `shf.r.wrap.b32` | 同上 |
| `bfe_u32` | `bfe.u32` | 同上 |
| `bfe_u64` | `bfe.u64` | 同上 |
| `prmt` | `prmt.b32` | intrinsic。調查表有列；初始化時 exllamav3 的 `.cu`/`.cuh` 沒有 `prmt` 字面 |
| `ld_global_acquire` | `ld.global.acquire.gpu.b32` | intrinsic |
| `st_global_release` | `st.global.release.sys.u32` | intrinsic |
| `red_relaxed` | `red.relaxed.gpu.global.add.s32` | intrinsic |
| `fence_acq_rel` | `fence.acq_rel.gpu` | intrinsic |
| `ld_global_cv` | `ld.global.cv.u32` | 沒有 |
| `st_global_wt` | `st.global.wt.u32` | 沒有 |
| `nanosleep` | `nanosleep.u32` | intrinsic |
| `laneid` | `mov.u32 %laneid` | intrinsic |
| `sm86_mad_lo` | `mad.lo.u32`，只在 `__CUDA_ARCH__ == 860` | `--device-cfg`，不是缺 intrinsic |

`cp.async` 的長度運算元必須是立即值（`"n"(16)`）。用暫存器傳長度，ptxas 會報 arguments mismatch。

## 必須拒絕

| id | 指令 | 原因 |
|---|---|---|
| `add_rn_f32_f16` | `add.rn.f32.f16` | sm_100 才有。sm_86 的 ptxas 回 `Unexpected instruction types`。exllamav3 在 `__CUDA_ARCH__ >= 1000` 才發，否則拆成 half2→float 再加 |

## 這台機器用不到、所以沒有探针

- FP8、`wgmma`、`tcgen05`、TMA
- sm_75 的降級：`cp.async` 改同步寫 shared、`m16n8k16` 拆成兩次 `m16n8k8`、動態 shared 上限 64 KB
- `__CUDA_ARCH__ > 890` 的 GEMM barrier 分支。那是編譯期切換，不是一條 sm_86 要發的指令

## CPU

`this_host_is_avx2_without_avx512` 鎖定 i9-9900K：有 AVX2、FMA、F16C，沒有 AVX-512F。MoE 若有 CPU 路徑，走 AVX2，不走 VNNI / VBMI / AVX-512。
