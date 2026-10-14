# 27B decode 的 kernel

日期：2026-10-15。這份列出 Qwen3.8-27B 自回歸 decode 每一個 token 會呼叫到的 kernel，給 Rust 寫 shim 時對照。來源是 `~/qwen38-exl3-dflash2/exllamav3`（`community` 分支，`355c6ee`），也就是 [cache.md](cache.md) 那支服務實際載入的 fork。`~/exllamav3` 是另一個分支，kernel 檔案和這份不同，不要拿來對。

清單是讀原始碼追出來的。當時服務佔著 GPU，沒有跑 `nsys`。哪條張量最後走哪支 GEMV，要等 profile 對過才算定。

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
| `exl3_gemv_try_launch` | `quant/exl3_gemv.cu:92` | int8 不接手時的 QTIP 式 GEMV。`lm_head` 可能走這條 |
| 協作式 GEMM | `quant/exl3_gemm_kernel.cuh`、`coop_autotune.cu` | 上面兩條都不接手時。kernel 在執行時 autotune 選出 |
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

## Rust 要自己重做的部分

這些不是 kernel，但 ext 裡的 kernel 依賴它們：

| 東西 | 檔案 | 作用 |
|---|---|---|
| `Graph` | `graph.cu`（221 行） | 捕捉後改 kernel 參數（`record_param`）：block table、seqlens、split 數 |
| `TritonKernel` | `triton_kernel.h` | 載 cubin，`cuLaunchKernel` |
| `DevCtx` | `quant/exl3_devctx.cu` | SM 數、GEMM 的 locks |
| GEMM autotune | `quant/coop_autotune.cu` | 照搬，或在建置時量好後固定 |

kernel 本體（`.cuh`）大多吃裸指標。外層 launcher 吃 `at::Tensor`，shim 要換掉的就是這一層。`.cu` 主要檔合計約 9600 行。

沒看到 27B decode 用 cuBLAS。`hgemm_gr` 只在 fp16 headwise 或 full gate 時出現，27B 是 interleaved gate。這一點也要等 profile 確認。

## 和 berth 規格的對位

[runtime.md](runtime.md) 已依這份清單改過：kernel 列表對上實際入口；attention 定為建置時預編的 Triton cubin，懲罰和抽樣留在 GPU。還沒對上的：

| 已經寫下來的 | 這支 fork 的現況 |
|---|---|
| [runtime.md](runtime.md)：Rust 擁有 CUDA graph 的 capture 和 launch | 現在是每層一張 graph，層之間是 Python |

## 下一輪要對的

- 用 `nsys` 跑一段短 decode，對每個 token 實際 launch 的 kernel 名稱，以及每條張量走的是 int8 GEMV、GEMV 還是 GEMM。`hgemm`（cuBLAS）有沒有出現也一起看。要等 `:8889` 的服務停下來才跑：它佔著約 22 GB VRAM，再開一個 27B 會 OOM，也會把服務拖垮。
- 82 個 SM 是 3090 的規格值，沒有在這台機器上讀 `multi_processor_count`。和上一項一起確認。
- 這份只涵蓋 27B。另外三個目標要加的 kernel（MoE、QSA、PLE、視覺塔）見 [models.md](models.md)，還沒有清單。
