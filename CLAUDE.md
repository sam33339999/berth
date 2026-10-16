# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 這是什麼

berth 是這台機器（一張 RTX 3090 24 GiB、sm_86）的 Rust 推論 runtime，目前服務 Qwen3.8-27B EXL3。最高原則：**和 fork `~/qwen38-exl3-dflash2/exllamav3`（正式服務 `:8889` 用的那份）逐位元一致**。每一步都拿 fork dump 出來的參考資料比對 digest，差一個 bit 就算錯。效能優化不能改變結果。

用中文回覆使用者。repo 的文件也都是中文；`docs/workspace.md` 是現況的權威說明，改了行為要一起更新它。

## 指令

```bash
cargo build --release -p berth
cargo test                                   # 不碰 GPU，服務在跑也能跑
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check

# GPU 測試：要 BERTH_GPU_TESTS=1，沒設會直接 pass（不是 skip 訊息）
BERTH_GPU_TESTS=1 cargo test --release --workspace
BERTH_GPU_TESTS=1 cargo test --release -p berth-engine --test prefill_ref -- --nocapture

# 本機試服務：一定換 port，8889 是正式服務
./target/release/berth --host 127.0.0.1 --port 18889
```

`BERTH_27B_DIR` 預設 `~/qwen38-exl3-dflash2/models/qwen38-27b-exl3`。

### 跑 GPU 工作之前

正式服務常佔著 3090。跑任何會配 GPU 記憶體的東西（GPU 測試、載模型、nsys、Python dump）之前，先看 `nvidia-smi` 和 `ss -ltn | grep :8889`，服務在跑就不要跑，並先問使用者。測試和試跑都不准綁 8889。

### Triton cubin 和 Python 腳本

Triton kernel 不在 `build.rs` 產生，要先用服務那份 Python 預編到 `target/triton-cubins/`（`build_triton_cubins.py`、`build_triton_fla_cubins.py` 寫到 `target/triton-cubins/fla`）。`scripts/dump_*.py` 產生 `testdata/` 的參考資料，要載 27B、要 GPU。環境照各腳本 docstring（`serve-host.sh` 的那組 export：`PYTHONPATH=~/qwen38-exl3-dflash2/exllamav3`、`TRITON_LIBCUDA_PATH`、`EXL3_*` 旗標），用 `~/qwen38-exl3-dflash2/env/bin/python`。git worktree 裡沒有 `target/triton-cubins`，要從主 repo symlink 過去。

## 架構

依賴只往下：`berth`（HTTP/CLI）→ `berth-engine` → `berth-kernels` / `berth-cuda` / `berth-plan`（→ `berth-exl3`）/ `berth-cache` / `berth-repetition`。只有 `berth-cuda`、`berth-kernels`、`berth-engine` 碰 GPU，其餘是純 Rust。

- **berth-cuda**：包住 `cuda-core =0.4.0`（早期專案，API 會改）。公開 API 不得露出任何 `cuda_core` 型別或錯誤，上層要原始呼叫就在這裡包一支。
- **berth-kernels**：`build.rs` 直接叫 `/usr/local/cuda-13.3/bin/nvcc`（不是 `/usr/bin/nvcc` 12.4），旗標照 fork 的 `setup.py`。fork 原始碼不進 repo：
  - 讀 `BERTH_EXL3_SRC`（預設 fork 路徑），檢查 `exllamav3_ext` 的 tree hash 是 `de35ffb`，且工作目錄乾淨，不符就停。
  - 和 ATen 寫在一起的 kernel 本體由 `FORK_EXTRACTS` 照行號抽到 `OUT_DIR/fork/*.cuh`；`csrc/` 只寫不吃 `at::Tensor` 的 launcher。不重寫 fork 已有的 kernel。
  - cuBLAS 必須是 torch 的 `libcublas.so.13`（13.1，在 `~/openwebui/.venv/.../nvidia/cu13/lib`），不是工具包的 13.6，兩者結果不同。`build.rs` 在 `target/cublas-torch/` 放 symlink，`.cargo/config.toml` 的 `$ORIGIN` rpath 讓執行檔找到它。
  - `.cargo/config.toml` 的 `BINDGEN_EXTRA_CLANG_ARGS` 是這台 libclang 缺標頭的 workaround。
- **berth-engine**：`Model::decode_step`（eager，不捕捉 graph）、`Model::prefill`、`Sampler`、`Engine::generate`。27B 是 GDN（linear attention）和 full attention 交錯的 64 層。要和 fork 一致的規則（細節在 `docs/workspace.md`）：
  - prefill 只跑 `prompt[..len-1]`，段落對齊 256、最多 2048、recurrent 模型切在最後一個頁邊界；最後一個 prompt token 走 decode。第 63 層 prefill 只跑 mixer。
  - decode 的 block table 寬度照每一步的當下序列長度算（`set_decode_width(position + 1)`：`ceil((position + 1) / 256)` 頁補到 16 的倍數）。`prompt + max_new_tokens` 只在沒跨 16 頁邊界時一致。不是 cache 頁數；attention split 數由寬度決定。
  - EXL3 linear 依列數走不同路徑：m=1 int8 GEMV、≤144 列協作式 GEMM、更多列 reconstruct 成 fp16 再走 cuBLAS 或 fork 的 f16acc GEMM。
  - Triton 依整數參數值特化（==1、%16==0、其他），每個組合都要預編 cubin，執行時照同規則挑。
- **berth**：OpenAI 相容 HTTP（`/v1/chat/completions` 含 SSE、`/v1/completions`），一次一個 GPU job。

### 參考資料與測試

`testdata/{exl3_ref,decode_ref,decode_seq,prefill_ref,kernels}` 是 fork 的輸出（多半是 blake2b digest），進 git。大型張量 dump（如 `scripts/dump_27b_prefill_layers.py`）寫到 `target/`，不進 git。測試的暫存檔放 `CARGO_TARGET_TMPDIR` 或 `target/`，不要放 `/tmp`（有 quota）。

## 工作方式

- 使用者常要求「用 grok 平行做」：每個任務開 `~/berth-wt/<name>` worktree 和 `grok/<name>` 分支。agent 的回報不可信（曾回報成功但測試根本沒連結），一律自己看 diff、自己跑測試。
- 只在使用者說 commit 時才 commit；commit 後清掉 worktree 和分支。commit 訊息照 `git log` 的風格：一句完整句子當標題（句點結尾），內文說明做了什麼、怎麼驗證和 fork 一致、還剩什麼。
- 不要動 fork、`~/qwen38-exl3-dflash2` 或其他專案的檔案。
