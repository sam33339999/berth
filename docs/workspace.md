# 工作區

日期：2026-10-15。這份寫 Rust 工作區怎麼切 crate、每個 crate 放什麼。純 Rust 的四個（`berth-exl3`、`berth-plan`、`berth-cache`、`berth-repetition`）已經實作並有測試。`berth-cuda` 和 `berth-kernels` 也有了，一條 EXL3 Linear 在 GPU 上和 fork 逐位元一致。`berth-engine` 只有那個對照測試，`berth` 還是骨架。

## 切法

照「需不需要 GPU」切。格式、層計畫、頁表、重複檢查是純 Rust，`cargo test` 不碰 GPU。會碰 GPU 的只有 `berth-cuda`、`berth-kernels`、`berth-engine` 三個。

服務常駐在 `:8889`，佔著 3090。純 Rust 的部分不用等 GPU 空出來就能寫、能測。

## 目錄

```text
crates/
  berth-ptx/          已有：sm_86 指令目錄，見 ptx.md
  berth-exl3/         純 Rust：讀 EXL3 checkpoint
  berth-plan/         純 Rust：LayerPlan 和泊位
  berth-cache/        純 Rust：頁表、hash、驅逐、checkpoint 的鍵
  berth-repetition/   純 Rust：生成重複檢查
  berth-cuda/         GPU：driver API、記憶體、stream、graph
  berth-kernels/      GPU：exllamav3 kernel 的 shim，加 Triton cubin
  berth-engine/       GPU：forward、decode 迴圈、抽樣
  berth/              binary：HTTP 服務和 CLI
```

## 每個 crate

| crate | 內容 | 規格 | 預計的外部依賴 |
|---|---|---|---|
| `berth-exl3` | `config.json` 和 `.safetensors`。每條 Linear 的 `trellis`、`suh` / `svh`、bias、mcg 或 mul1、K。打包的 `su` / `sv` 在這裡解開 | [exl3.md](exl3.md) | `serde_json`、`thiserror`。safetensors header 自己解，沒用 `safetensors` crate |
| `berth-plan` | 依 `arch_string` 建 `LayerPlan`。決定每塊張量停 VRAM、pinned RAM 還是 SSD | [runtime.md](runtime.md)、[memory.md](memory.md) | `thiserror`。`from_exl3` 把 checkpoint 轉成 `ModelShape` 和 `TensorBytes`，視覺塔（`model.visual.`）和 MTP（`mtp.`）不算進 decode |
| `berth-cache` | 滿頁的 blake2b-128 鏈式 hash、`ref_count`、以根的 `access_serial` 做樹狀驅逐、GDN checkpoint 的鍵、rewind 的記帳。只處理頁號，不碰 GPU 記憶體 | [cache.md](cache.md) | `blake2` |
| `berth-repetition` | `thinking` / `response` 兩個緩衝區、尾端和原樣檢查、標記字串、256 byte 和 64 條的上限 | [repetition.md](repetition.md) | 無 |
| `berth-cuda` | device 和 pinned 記憶體、stream、event。捕捉後能改 kernel 參數的 graph，取代 `graph.cu`。cubin 載入，取代 `TritonKernel`。SM 數和 GEMM locks，取代 `DevCtx` | [kernels.md](kernels.md) | `cuda-core =0.4.0`，見下一節 |
| `berth-kernels` | `build.rs` 用 nvcc 把 shim 和 fork 的 `.cuh` 編成靜態庫。`extern "C"` 宣告和安全包裝。執行時載入預編的 attention cubin | [kernels.md](kernels.md) | `cc` |
| `berth-engine` | 照 `LayerPlan` 跑 forward、捕捉 graph、decode 迴圈、GPU 上的懲罰和抽樣，把接受的 token 交給重複檢查。之後的草稿 accept / rewind 也在這裡 | [runtime.md](runtime.md) | 無 |
| `berth` | OpenAI 相容的 HTTP、SSE（含 `response.repetition_check`）、tokenizer、chat template、CLI | [cache.md](cache.md)、[repetition.md](repetition.md) | `axum`、`tokenizers`、`minijinja`、`clap` |

外部依賴等寫到那段程式時才加。

## 依賴方向

```text
berth
  └─ berth-engine
       ├─ berth-kernels ── berth-cuda
       ├─ berth-plan ───── berth-exl3
       ├─ berth-cache
       └─ berth-repetition
```

依賴只往下指。純 Rust 的四個不依賴 `berth-cuda`。

## berth-cuda 的底層

底下用 `cuda-core`，版本鎖死 `=0.4.0`。它是 `NVIDIA/cuda-rust` 的 host 端 runtime，和寫 device 碼的 cuda-oxide 同一個 repo（調查時是 `bc46b1f`）。host 端這幾個 crate 走 stable，不用 nightly。建置時 `cuda-bindings` 用 bindgen 讀 `cuda.h`，要 libclang。

它已經有的：device 記憶體、`PinnedHostBuffer`、stream、event、從檔案或位元組載 cubin、吃原始參數陣列的 `launch_kernel`、`multiprocessor_count`。

它沒有的：graph 只到 `begin_capture` / `end_capture`，拿回原始的 `CUgraph`。instantiate、launch、捕捉後改 kernel 參數（`cuGraphExecKernelNodeSetParams`），要在 `berth-cuda` 裡用 `cuda_core::sys` 自己包。這就是取代 `graph.cu` 的那一段。

cuda-core 自己寫明是早期研究專案，API 會改。`berth-cuda` 是隔開它的那一層：

- 公開 API 不露出 `cuda_core` 的任何型別。上層只看到 `berth-cuda` 自己的型別，例如 `DevicePtr`、`Stream`、`Event`、`Graph`、`Cubin`、`PinnedBuf`
- 錯誤轉成 `berth-cuda` 自己的錯誤型別，不讓 `DriverError` 漏出去
- 不 `pub use cuda_core::sys`。上層要原始呼叫，就在 `berth-cuda` 裡包一支

這樣 cuda-core 改版，只改 `berth-cuda`。要換回 `cudarc`，或直接用 bindgen 綁 driver API，影響也只在這個 crate 裡。`cudarc` 有沒有包「捕捉後改參數」沒有查過；先用 cuda-core 寫，以後要換再比。

## exllamav3 原始碼

不 vendor 進這個 repo。`berth-kernels` 的 `build.rs` 讀 `BERTH_EXL3_SRC`，預設 `~/qwen38-exl3-dflash2/exllamav3`，並檢查 `exllamav3/exllamav3_ext` 的 git tree hash 是 `de35ffb`；對不上就停。

檢查 tree hash，不檢查 commit。fork 只改 Python 時（例如 `355c6ee` 到 `dc0e608` 加了 MiniCPM-V 4.7），kernel 沒變，照常 build。

shim 放在 `crates/berth-kernels` 裡，只 include fork 的 `.cuh`。fork 外層吃 `at::Tensor` 的 launcher 不編進來。

## Triton cubin

不在 `build.rs` 裡產生。在 `build.rs` 叫 Python 很脆，也會讓每次 `cargo build` 都要有 Triton。

先跑一次 `scripts/build_triton_cubins.py`，cubin 放到固定目錄。`cargo build` 找不到就報錯，並說要跑哪支腳本。27B 自回歸只要 split 和 combine 兩支，見 [runtime.md](runtime.md)。

用服務那份 Python 跑，不載模型。腳本看到 `:8889` 有在聽就停，不碰 GPU：

```text
PYTHONPATH=~/qwen38-exl3-dflash2/exllamav3 \
TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu \
~/qwen38-exl3-dflash2/env/bin/python scripts/build_triton_cubins.py
```

預設寫到 `target/triton-cubins/`（第一個參數換目錄）：`_paged_attn_decode_split_kernel.cubin`、`_paged_attn_decode_combine_kernel.cubin`、`manifest.json`。`BERTH_27B_DIR` 和 `BERTH_EXL3_SRC` 沒設時用 `~/qwen38-exl3-dflash2` 底下的 checkpoint 和 fork。

## berth-kernels 的建置

`build.rs` 直接叫 nvcc，不經 `cc`。nvcc 是 `/usr/local/cuda-13.3/bin/nvcc`（`BERTH_NVCC` 可換）：fork 的 extension 是 CUDA 13.3 編的，`/usr/bin/nvcc` 是 12.4，不能用。旗標照 fork 的 `setup.py`：`-O3 --use_fast_math -lineinfo -arch=sm_86`。連 CUDA 13.3 的動態 `libcudart`。

除了 tree hash，`build.rs` 也檢查 fork 的 `exllamav3_ext` 底下沒有還沒 commit 的修改。tree hash 只看 HEAD，nvcc 編的是工作目錄。

協作式 GEMM 的 autotune 照 `quant/coop_autotune.cu` 重寫了一份不用 ATen 的（`csrc/autotune.cu`）。它讀寫服務共用的快取 `~/.cache/exllamav3/autotune/coop_autotune_v1.bin`，所以 `lm_head` 選到的設定和服務一樣。這是和 fork 最可能分岔的地方，fork 改了 autotune 就要跟著改。

連進來的只有 27B decode 用到的 kernel。其他形狀直接回錯誤，不退到別條路。

## 這台機器的 bindgen

`cuda-core` 底下的 `cuda-bindings` 建置時用 bindgen 讀 `cuda.h`。這台的 libclang 套件沒有附 clang 內建的標頭（`stddef.h` 等），所以 `.cargo/config.toml` 設了 `BINDGEN_EXTRA_CLANG_ARGS`，指向 gcc 15 的內建標頭。換機器時這行要改，或裝齊 clang 的 resource 目錄。

## GPU 測試

`berth-cuda`、`berth-kernels`、`berth-engine` 裡碰 GPU 的測試，只在 `BERTH_GPU_TESTS=1` 時跑。平常的 `cargo test` 不碰 GPU，服務在跑的時候也不會被拖垮。

`berth-ptx` 的測試只呼叫 nvcc 組譯，不碰 GPU，照常跑。

## 現在不開的

- `berth-ngram`：Flash-Next 的 n-gram hash 和 direct I/O 按列讀。第 4 階段再開
- MoE 的專家排程先放在 `berth-plan` 和 `berth-engine`，長大了再拆

## 第 1 階段（27B）的起手順序

1. ✅ `berth-exl3`：讀 27B 的 checkpoint，印出每條張量的 K 和形狀，和 Python 對照。不需要 GPU
2. ✅ `berth-plan`：建出 `Qwen3_5ForConditionalGeneration` 的 64 層計畫，整包停 VRAM
3. ✅ `berth-cache`、`berth-repetition`：照規格寫，加單元測試
4. ✅ `berth-cuda`、`berth-kernels`：shim int8 GEMV 和 `lm_head` 的協作式 GEMM，拿 `testdata/exl3_ref/` 的輸出逐位元對。`crates/berth-engine/tests/exl3_ref.rs`，`BERTH_GPU_TESTS=1` 時跑，4 條全部一致
5. `berth-engine`、`berth`：一個 token 跑通，再接成迴圈

## 第 4 步留下的

- `berth-cuda` 的 `Capture::record` 只記得到透過 `berth-cuda` 的 `Function` launch 的 kernel。`berth-kernels` 在 nvcc 編的程式裡用 runtime 直接 launch，那些節點現在記不到參數。第 5 步把整層捕捉成 graph 之前要補
- `berth-kernels` 還不依賴 `berth-cuda`，吃的是原始指標和 stream handle。`berth-engine` 接起兩邊
