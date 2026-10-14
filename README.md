# berth

這台機器（一張 RTX 3090、32 GB 記憶體）的推論 runtime。服務的權重是 exllamav3 的 EXL3。Rust 決定每一塊 trellis 的泊位，CUDA kernel 直接對那些張量做計算。

## 名稱

berth 是船隻靠岸時分配到的泊位。這個 runtime 第一件要決定的事，就是每塊權重停在哪一層：VRAM、pinned RAM，還是 SSD。

取這個名字是為了這條放置規則。Qwen3.8-27B 的 EXL3 整包停在 VRAM。MiniCPM-V 4.7 35B-A3B 在 4 bpw 也停得進這張卡。Qwen3.8-Flash-Next 的專家 trellis 和 n-gram 表才要三層一起排。家目錄裡沒有同名目錄，也避開了當時在看的 Strata。

## 規格

- [docs/exl3.md](docs/exl3.md)：服務格式。EXL3 trellis，含 n-gram 表和 KV cache
- [docs/runtime.md](docs/runtime.md)：Rust 寫 host，已量過的 kernel 繼續用 nvcc
- [docs/repetition.md](docs/repetition.md)：抽樣後檢查已接受的 token，串流裡送重複標記
- [docs/models.md](docs/models.md)：Qwen3.8-27B、Qwen3.6-35B-A3B、MiniCPM-V 4.7 35B-A3B、Qwen3.8-Flash-Next
- [docs/memory.md](docs/memory.md)：三層泊位
- [docs/cache.md](docs/cache.md)：多人共用一塊頁池時，前綴怎麼留
- [docs/ptx.md](docs/ptx.md)：sm_86 要組過的指令，和 cuda-oxide 還沒有的那些

## 測試

```bash
cargo test -p berth-ptx
```

`ptx/sm86_probes.cu` 的每一條 `// PROBE` 都要被 `nvcc -arch=sm_86` 組進 cubin。`add.rn.f32.f16` 必須被 ptxas 拒絕。目錄在 `crates/berth-ptx`，和探针註解不一致時測試會失敗。

cuda-oxide 還沒有接進這個 crate。調查時的 commit 是 `bc46b1f`，它要 nightly。缺的指令以後用 `ptx_asm!` 補，見規格。
