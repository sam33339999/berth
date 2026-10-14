# Runtime

日期：2026-10-14
機器：sam-msi。RTX 3090（sm_86，24 GB）、i9-9900K（AVX2、FMA、F16C，沒有 AVX-512）、32 GB DDR4-2667、PCIe 3.0。

## 分層

Rust 寫 host。已經在 exllamav3 量過的 kernel 用 nvcc 編成 cubin 或靜態庫，由 Rust 用 driver API launch。

cuda-oxide（NVIDIA/cuda-rust，調查時的 commit `bc46b1f`，2026-10-12）用來核對 sm_86 指令能不能表達。它是早期專案，要 pinned nightly，API 還會變。缺的指令用 `ptx_asm!`。`__CUDA_ARCH__ == 860` 這類切換用 `--device-cfg` 配 `#[cfg]`，並且和 `--arch` 維持一致。

不要用 cuda-oxide 重寫 int8 GEMV、cq4 paged attention、GDN。那些 kernel 的速度在 device 碼裡。熱路徑是 inline PTX，最後都進 ptxas，decode 卡在頻寬。重寫會丟掉 wave 對齊、融合 norm 和 cq4。

新算子（QSA、gated residual、PLE conv）先寫 CUDA。形狀穩了再考慮搬進 cuda-oxide。

## Rust 擁有

- safetensors 和 `config.json`，編成層計畫
- 每一塊權重的泊位：VRAM、pinned RAM、SSD
- CUDA graph 的 capture 和 launch
- KV、GDN、PLE 的 checkpoint 和 rewind
- 重複懲罰和抽樣。懲罰是抽樣前對 logits 的 scatter，不進 forward，也不進 graph
- 草稿的提出、逐位置比對、退回。位置 i 的懲罰看得到同一塊裡已經接受的 token
- Flash-Next 的 n-gram hash，和按列從 SSD 預讀
- 對外的服務

## 留在 nvcc 的 kernel

- int8 GEMV：sq、multi、SiLU、fused norm
- cq4 paged attention
- GDN recurrent，和寬讀取 b/a GEMV

骨架和熱專家走這條路。冷專家是一段連續、好 DMA 的 bytes，搬上 GPU 再解。EXL3 trellis 適合常駐權重，不適合約 1.5 MB 的 PCIe 搬運。

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

1. Qwen3.8-27B，自回歸。鎖頻 1500 MHz 對上短 context 約 49 tok/s，262k、cq4、沒有草稿約 25 tok/s。decode KL 低於約 5e-4 當雜訊。對不上就先修 runtime。
2. 同一條 target 加 frozen SAM。走通 accept、rewind、逐位置懲罰。
3. DFlash2。target 在指定層留下 activation，草稿用自己的 cache。
4. MoE schedule。MiniCPM 在這張卡上先整包進 VRAM。
5. Flash-Next 的三層泊位。等 27B 的 rewind 是對的再做。

CPU 上的 MoE 路徑只會是 AVX2+FMA+F16C。這顆 CPU 沒有 AVX-512，也不要把 miss 的 expert 改在 CPU 上算。
