# Runtime

日期：2026-10-14
機器：sam-msi。RTX 3090（sm_86，24 GB）、i9-9900K（AVX2、FMA、F16C，沒有 AVX-512）、32 GB DDR4-2667、PCIe 3.0。

## 分層

Rust 寫 host。已經在 exllamav3 量過的 kernel 用 nvcc 編成 cubin 或靜態庫，由 Rust 用 driver API launch。

「exllamav3」指 `~/qwen38-exl3-dflash2/exllamav3`（`community` 分支），也就是服務實際載入的那份 fork，不是 `~/exllamav3`。這些 kernel 的外層 launcher 吃 `at::Tensor`，Rust 這邊要寫一層吃裸指標的 shim。逐支清單見 [kernels.md](kernels.md)。

cuda-oxide（NVIDIA/cuda-rust，調查時的 commit `bc46b1f`，2026-10-12）用來核對 sm_86 指令能不能表達。它是早期專案，要 pinned nightly，API 還會變。缺的指令用 `ptx_asm!`。`__CUDA_ARCH__ == 860` 這類切換用 `--device-cfg` 配 `#[cfg]`，並且和 `--arch` 維持一致。

不要用 cuda-oxide 重寫 int8 GEMV、cq4 paged attention、GDN。那些 kernel 的速度在 device 碼裡。熱路徑是 inline PTX，最後都進 ptxas，decode 卡在頻寬。重寫會丟掉 wave 對齊、融合 norm 和 cq4。

新算子（QSA、gated residual、PLE conv）先寫 CUDA。形狀穩了再考慮搬進 cuda-oxide。

## Rust 擁有

- EXL3 的 safetensors 和 `config.json`，編成層計畫。格式見 [exl3.md](exl3.md)
- 每一塊 trellis 的泊位：VRAM、pinned RAM、SSD。搬運不改量化
- CUDA graph 的 capture 和 launch
- KV、GDN、PLE 的 checkpoint 和 rewind
- 重複懲罰和抽樣。懲罰是抽樣前對 logits 的 scatter，不進 forward，也不進 graph。懲罰和抽樣都留在 GPU，沿用 exllamav3 的 kernel（`rep_pen.cu`、`dry.cu`、`sampling_fused.cu`），由 Rust 在 graph 外面 launch。logits 不搬回 CPU，回到 host 的只有抽中的 token id
- 生成重複檢查。抽樣後看已接受的 token，規格見 [repetition.md](repetition.md)
- 草稿的提出、逐位置比對、退回。位置 i 的懲罰看得到同一塊裡已經接受的 token
- Flash-Next 的 n-gram hash，和按列從 SSD 預讀
- 對外的服務

## 留在 nvcc 的 kernel

- EXL3 GEMV / GEMM：int8 GEMV（m=1 時的 `sq` kernel）、QTIP 式 GEMV、協作式 GEMM，以及合包投影用的 `mgemm`。SiLU 是另一個 launch（`silu_mul`），殘差加 norm 是 `rms_norm_res_in`
- cq 頁寫入（`quant_cache_paged`）、RoPE
- GDN recurrent、causal conv1d update，和寬讀取 b/a GEMV

## decode attention：建置時預編的 Triton cubin

cq4 paged attention 原本列在上一節。實際上 decode attention 是 Triton：`_paged_attn_decode_split_kernel` 加 `_paged_attn_decode_combine_kernel`，執行時才編成 cubin。exllamav3 裡沒有對應的 nvcc kernel。

建置時用 Triton 把這兩支 kernel 預先編成 cubin，Rust 用 driver API 直接 launch，執行時不帶 Python。cubin 裡的 constexpr 是形狀和 cache 設定：head dim 256、24 Q head、4 KV head、`q_len`、K 和 V 的 cq bit。27B 自回歸只要一支 split 加一支 combine；換 cq bit 就再編一份。DFlash2 驗證窗的 `q_len` 不同，到時候另編。

split 數不進 cubin。`splits_cap` 只決定 grid 和 partial buffer 的大小，公式在 `bc_attn.py` 的 `_configure`：`min(2 × SM 數 / programs, 128)`。3090 有 82 個 SM，27B decode 的 programs 是 4，所以上限是 41，grid 是 4 × 41。實際的 split 數由 `libtorch/attention.cpp:388` 的 `split_config` 從 block table 寬度算出：`ceil((頁數 × 256 + 1) / 128)`，再夾到 41。超過約 5200 token 就固定是 41。它和 block table、seqlens 一樣是 kernel 參數，捕捉後改參數更新，不重編。

建置因此要有 Python 和 Triton。速度沿用現在的服務，不另寫 nvcc 版本。

骨架、熱專家、從 RAM 或 SSD 補上來的冷專家，走的是同一支 kernel。冷專家搬的是該 expert 的 `trellis`、`suh`、`svh`。檔案裡這幾塊連續排列，miss 時一次 DMA。不另做一種量化。

## Schedule

兩個權重後端、三種 mixer。不要做一個用 flag 切 dense、MoE、n-gram、草稿的 kernel。

```text
LayerPlan
  mixer:  Gdn | FullAttn | Qsa
  ffn:    DenseMlp | Moe { router, shared, experts }
  extra:  PleInject     # 只有 Flash-Next，第 2 層前面
```

視覺塔是 prefill，不進 decode graph。

## 順序

按架構搬，名稱是 exllamav3 fork 的 `arch_string`：

| 階段 | 架構 | 模型 |
|---|---|---|
| 1 | `Qwen3_5ForConditionalGeneration` | Qwen3.8-27B |
| 2 | `Qwen3_5MoeForConditionalGeneration` | Qwen3.6-35B-A3B |
| 3 | `MiniCPMV4_7ForConditionalGeneration` | MiniCPM-V 4.7 35B-A3B |
| 4 | `Qwen4ExpForConditionalGeneration` | Qwen3.8-Flash-Next |

1. Qwen3.8-27B，自回歸。鎖頻 1500 MHz 對上短 context 約 49 tok/s，262k、cq4、沒有草稿約 25 tok/s。decode KL 低於約 5e-4 當雜訊。對不上就先修 runtime。shim、graph、抽樣都在這一步對過，因為只有 27B 有正在跑的服務和量過的數字可以比。kernel 清單見 [kernels.md](kernels.md)
   - 同一條 target 加 frozen SAM。走通 accept、rewind、逐位置懲罰
   - DFlash2。target 在指定層留下 activation，草稿用自己的 cache
2. Qwen3.6-35B-A3B。GDN 和 attention 沿用第 1 步，GatedMLP 換成 MoE 加 shared expert。新的只有 MoE kernel 和專家排程。這台機器上只有它的 GGUF，EXL3 要先轉或下載
3. MiniCPM-V 4.7。文字堆疊和第 2 步同形，多一個 SigLIP2 視覺塔，只在 prefill 跑。這張卡上先整包進 VRAM。本機還沒有這份 checkpoint
4. Flash-Next 的三層泊位。QSA、hyper-connection、PLE n-gram 表。等 27B 的 rewind 是對的再做

CPU 上的 MoE 路徑只會是 AVX2+FMA+F16C。這顆 CPU 沒有 AVX-512，也不要把 miss 的 expert 改在 CPU 上算。
