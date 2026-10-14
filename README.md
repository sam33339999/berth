# berth

這台機器（一張 RTX 3090、32 GB 記憶體）的推論 runtime。Rust 決定每一塊權重的泊位，CUDA kernel 負責計算。

泊位有三層：VRAM、pinned RAM、SSD。名字來自這裡。Qwen3.8-27B 整包在 VRAM。MiniCPM-V 4.7 35B-A3B 的 4-bit 也放得進這張卡。Qwen3.8-Flash-Next 的專家和 n-gram 表才需要三層一起用。

## 規格

- [docs/runtime.md](docs/runtime.md)：Rust 寫 host，已量過的 kernel 繼續用 nvcc
- [docs/models.md](docs/models.md)：Qwen3.8-27B、Qwen3.6-35B-A3B、MiniCPM-V 4.7 35B-A3B、Qwen3.8-Flash-Next
- [docs/memory.md](docs/memory.md)：三層泊位
- [docs/ptx.md](docs/ptx.md)：sm_86 要組過的指令，和 cuda-oxide 還沒有的那些

## 測試

```bash
cargo test -p berth-ptx
```

`ptx/sm86_probes.cu` 的每一條 `// PROBE` 都要被 `nvcc -arch=sm_86` 組進 cubin。`add.rn.f32.f16` 必須被 ptxas 拒絕。目錄在 `crates/berth-ptx`，和探针註解不一致時測試會失敗。

cuda-oxide 還沒有接進這個 crate。調查時的 commit 是 `bc46b1f`，它要 nightly。缺的指令以後用 `ptx_asm!` 補，見規格。
