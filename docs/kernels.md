# 27B decode 的 kernel

日期：2026-10-15。這份列出 Qwen3.8-27B 自回歸 decode 每一個 token 會呼叫到的 kernel，給 Rust 寫 shim 時對照。來源是 `~/qwen38-exl3-dflash2/exllamav3`，也就是 [cache.md](cache.md) 那支服務實際載入的 fork。讀原始碼時在 `community` 分支的 `355c6ee`；profile 時 fork 在 `feat/minicpm-v4.7` 的 `dc0e608`，只多了 MiniCPM-V 4.7 的 Python。兩者的 `exllamav3_ext` tree hash 都是 `de35ffb`，kernel 一樣。`~/exllamav3` 是另一個分支，kernel 檔案和這份不同，不要拿來對。

清單先讀原始碼追出來，再用 `nsys` 跑過一次對過，結果見「實測」。

## 這份 27B 的形狀

| 項目 | 值 |
|---|---|
| 層 | 64：48 GDN、16 full attention（每 4 層一個） |
| hidden / intermediate | 5120 / 17408 |
| attention | 24 Q head、4 KV head、head dim 256，`attn_output_gate` |
| GDN | 16 K head、48 V head、head dim 128，conv 4 |
| 權重 | EXL3 4 bpw，mul1 碼本。`lm_head` 6 bit |
| 詞彙 | 248320，`lm_head` 不和 embedding 共用 |

服務設 `EXL3_INT8_GEMV=2`、`EXL3_INT8_GEMV_MAX_K=5`、`EXL3_QC_STAGING=1`。K=4 的層走 int8 GEMV。`lm_head` 是 K=6，不走。

## 呼叫結構

每一層由一個 C++ 的 `BC_*` 物件跑完，各自捕捉成一張 graph：

- GDN：`BC_GatedDeltaNetSplit::run_bszN`
- full attention：`BC_Attention::run_gr`
- MLP：`BC_MLP::run_bsz1`

層和層之間仍由 Python 呼叫 norm 和殘差相加。一個 token 不是一張 graph，是 64 層 × 2 張，加上中間的 launch。

## EXL3 Linear

| 入口 | 檔案 | 什麼時候走 |
|---|---|---|
| `exl3_gemm_gr` | `quant/exl3_gemm.cu:124` | 所有 EXL3 Linear 的分派器 |
| `exl3_gemv_int8`、`exl3_gemv_int8_sq` | `quant/exl3_gemv_int8.cu`、`exl3_gemv_int8_kernel.cuh` | m=1、mul1、K≤5、`size_n % 256 == 0`、`size_k % 128 == 0`。骨架大多走這條 |
| `exl3_gemv_try_launch` | `quant/exl3_gemv.cu:92` | int8 不接手時的 QTIP 式 GEMV。27B decode 實測沒有走到 |
| 協作式 GEMM | `quant/exl3_gemm_kernel.cuh`、`coop_autotune.cu` | 上面兩條都不接手時。kernel 在執行時 autotune 選出。`lm_head`（K=6）走這條，用 `cudaLaunchCooperativeKernel` |
| `exl3_mgemm_gr` | `quant/exl3_gemm.cu:400` | GDN 的 qkv+z 合包、attention 的 qkv / qg 合包 |

各 K 和形狀的 template 實例在 `quant/comp_units/`，104 個檔。

## GDN（48 層）

| 入口 | 檔案 |
|---|---|
| `gated_delta_net_fused_op_3_gr` | `gdn.cu:1563` |
| `gdn_ba_gemv_gr`（b/a 寬讀取 GEMV） | `gdn.cu` |
| `cuda_causal_conv1d_update_gr` | `gdn.cu` |
| `cuda_recurrent_gated_delta_rule_gr` | `gdn.cu` |
| `gated_rms_norm_gr` | `norm.cu` |

q/k/v/z、o 的投影走上一節的 EXL3 Linear。

## Full attention（16 層）

| 入口 | 檔案 |
|---|---|
| `deinterleave_qg_gr` | `activation.cu` |
| `rope_gr`，q/k norm 融合在裡面 | `rope.cu` |
| `quant_cache_paged_gr`，新的 K/V 寫進 cq 頁 | `cache/q_cache.cu` |
| `_paged_attn_decode_split_kernel` | Triton，`modules/attention_fn/triton_paged.py:852` |
| `_paged_attn_decode_combine_kernel` | Triton，`triton_paged.py:999` |
| `mul_sigmoid__gr`，輸出 gate | `activation.cu` |

兩支 Triton kernel 在執行時用 `triton.compile` 編成 cubin（`bc_attn.py:64`），交給 C++ 的 `TritonKernel` launch。形狀和 cq bit 是 constexpr，split 數不是。`bc_attn.py` 檔頭註解說「split 數變了要重編」，和 `_configure` 的簽名對不上；以程式為準。splits_cap 的算法見 [runtime.md](runtime.md)。

## MLP（64 層）

| 入口 | 檔案 |
|---|---|
| `silu_mul_gr` | `activation.cu` |
| `copy2d_gr`，hidden 要 padding 時才有 | `add.cu` |

up 和 down 走 EXL3 Linear。SiLU 是獨立的 launch，沒有融進 GEMV。

## 層和層之間

| 入口 | 檔案 |
|---|---|
| `rms_norm`、`rms_norm_res_in`（殘差相加融進 norm） | `norm.cu` |
| `x += y`、embedding 查表 | ATen 自己的 kernel，不在 ext 裡 |

## 抽樣

服務的 sampler 是 RepP、PresFreqP、DRY、Temperature，全部在 GPU 上：

| 入口 | 檔案 |
|---|---|
| `apply_rep_pens`、`apply_pres_freq_pens` | `generator/rep_pen.cu` |
| `dry_penalty` | `generator/dry.cu` |
| `gumbel_noise_f32`、`fused_sampler` | `generator/gumbel.cu`、`generator/sampling_fused.cu` |

## 實測

2026-10-15，服務停著時跑的。`scripts/profile_27b_decode.py`：只載 target，不載 DFlash2 草稿；cq3；sampler 照服務預設（DRY 0.2，溫度 0.6）；prompt 約 180 token；暖機一個 job 之後，錄第二個 job 的 32 步 decode。環境變數和 `serve-host.sh` 一樣。`nsys` 加 `--cuda-graph-trace=node`，graph 裡的 kernel 才看得到。

`multi_processor_count` 讀到 82，compute capability 8.6。

每個 token：

| 項目 | 值 |
|---|---|
| 牆鐘時間 | 25.81 ms，約 38.7 token/s |
| GPU 忙碌 | 25.03 ms，97% |
| kernel 數 | 919：graph 裡 720，直接 launch 199 |
| `cudaGraphLaunch` | 128，64 層 × 2 張 |
| 改 graph 節點參數 | 38.5 次（`cudaGraphExecKernelNodeSetParams` 加 `cuGraphExecKernelNodeSetParams`） |
| 讀的權重 | 13.18 GB（decode 張量減掉 embedding），約 511 GB/s，3090 標稱 936 GB/s 的 55% |

時間佔比（32 步合計）：

| kernel | 每 token 次數 | 平均 | 佔比 | 是誰 |
|---|---|---|---|---|
| `exl3_gemv_int8_sq_kernel<4,1,0,0>` | 128 | 72.2 µs | 36.9% | fp16 輸出：MLP 的 gate、up |
| `exl3_gemv_int8_sq_kernel<4,1,1,0>` | 128 | 46.4 µs | 23.7% | fp32 輸出：MLP down、attention o_proj、GDN out_proj |
| `exl3_mgemm_kernel<4,1,…>` | 48 | 82.2 µs | 15.8% | GDN 的 qkv+z 合包 |
| `exl3_mgemm_kernel<4,0,…>` | 16 | 94.6 µs | 6.0% | attention 的 qkv 合包 |
| `exl3_gemm_kernel<6,…>` | 1 | 1225 µs | 4.9% | `lm_head` |
| `cuda_recurrent_gated_delta_rule_kernel_128` | 48 | 12.7 µs | 2.4% | GDN |
| `_paged_attn_decode_split_kernel` / `_combine_kernel` | 16 / 16 | 25.6 / 22.1 µs | 1.6% / 1.4% | Triton attention |
| `rms_norm_kernel`（兩種） | 64 + 65 | 4–5 µs | 2.4% | 層間 norm，多的一次是最後的 norm |
| 其他 GDN、attention、MLP 小 kernel | | | 約 4% | `gdn_ba_gemv`、`conv1d_update`、`gated_rms_norm`、`gated_delta_net_fused_op_3`、`rope`、`quant_cache_paged`、`deinterleave_qg`、`mul_sigmoid`、`act_mul_kernel_h` |
| ATen `CUDAFunctor_add<float>` | 64 | 2.2 µs | 0.6% | 殘差相加 |
| `dry_penalty_kernel` | 1 | 89.8 µs | 0.4% | DRY |
| `fs_partial_argmax_kernel`、`fs_finalize_kernel` | 1 + 1 | 5.0 / 1.9 µs | <0.1% | 抽樣 |

和讀原始碼時的推測對照：

- int8 GEMV 每 token 256 次，正好是 MLP 192 + GDN out_proj 48 + attention o_proj 16。gate 和 up 沒有合包
- 合包的 qkv 走 `exl3_mgemm`，不走 int8，佔 22%。這是 int8 以外最大的一塊
- `lm_head` 走協作式 GEMM，不是 GEMV。每 token 一次，1.2 ms
- 沒有 cuBLAS，`hgemm` 沒出現
- `copy2d` 沒出現，27B 的 hidden 不用 padding
- 服務預設 rep penalty 1.0、freq 0，所以 `apply_rep_pens`、`apply_pres_freq_pens` 沒有 launch。也沒有看到獨立的 `gumbel_noise_f32`；抽樣是 `fs_partial_argmax_kernel` 加 `fs_finalize_kernel`
- DFlash2 草稿的路徑沒有錄。服務實際每步會多跑草稿和驗證，那是另一份 profile

對照用的單條 Linear 輸出在 `testdata/exl3_ref/`，由 `scripts/dump_exl3_reference.py` 產生：layer 0 的 down、gate，layer 3 的 o_proj，和 `lm_head`，都是 m=1。同一份輸入跑兩次，輸出逐位元一致，所以 `berth-kernels` 可以要求完全相同。

## Rust 要自己重做的部分

這些不是 kernel，但 ext 裡的 kernel 依賴它們：

| 東西 | 檔案 | 作用 |
|---|---|---|
| `Graph` | `graph.cu`（221 行） | 捕捉後改 kernel 參數（`record_param`）：block table、seqlens、split 數 |
| `TritonKernel` | `triton_kernel.h` | 載 cubin，`cuLaunchKernel` |
| `DevCtx` | `quant/exl3_devctx.cu` | SM 數、GEMM 的 locks |
| GEMM autotune | `quant/coop_autotune.cu` | 照搬，或在建置時量好後固定 |

kernel 本體（`.cuh`）大多吃裸指標。外層 launcher 吃 `at::Tensor`，shim 要換掉的就是這一層。`.cu` 主要檔合計約 9600 行。

27B decode 不用 cuBLAS，profile 確認過。`hgemm_gr` 只在 fp16 headwise 或 full gate 時出現，27B 是 interleaved gate。

## 和 berth 規格的對位

[runtime.md](runtime.md) 已依這份清單改過：kernel 列表對上實際入口；attention 定為建置時預編的 Triton cubin，懲罰和抽樣留在 GPU。還沒對上的：

| 已經寫下來的 | 這支 fork 的現況 |
|---|---|
| [runtime.md](runtime.md)：Rust 擁有 CUDA graph 的 capture 和 launch | 現在是每層一張 graph，層之間是 Python |

## 下一輪要對的

- DFlash2 草稿加驗證的那一步，還沒錄 profile
- 這份只涵蓋 27B。另外三個目標要加的 kernel（MoE、QSA、PLE、視覺塔）見 [models.md](models.md)，還沒有清單
- `berth-plan` 的每 token 搬運量把整張 embedding（2.54 GB）算進去，實際只讀一列。估算要扣掉
