# TODO

最後更新：2026-10-16。

## 停在哪裡

**MiniCPM-V 4.7 的 MoE prefill 已經和 fork 逐位元一致，但還沒 commit。**

worktree 是 `~/berth-wt/minicpm`，分支 `grok/minicpm`。grok 的派工是 `~/berth-wt/minicpm-host/p2b_prefill_resume.md`，log 在 `p2b.ndjson`。它跑了 373 輪後正常結束，最終回報是空的。以下都是我自己驗證的結果。

- **改動還沒 commit**：
  - 改了 10 個檔，+775/−177：engine 的 attn、gdn、generate、model、moe、moe_plan、moe_prefill、prefill，加上 `berth-kernels/src/gdn_prefill.rs` 和 `docs/workspace.md`。
  - 新增 `tests/minicpm_prefill_ref.rs`、`tests/minicpm_moe_prefill_layer.rs`、`scripts/dump_minicpm_prefill_reference.py`。
  - 新增參考資料 `testdata/minicpm/prefill_ref/`。
- **GPU 驗證**（`BERTH_GPU_TESTS=1 cargo test --release --workspace --no-fail-fast`）：60 組全部 ok。
  - `minicpm_prefill_ref`：6、20、100、1500 四個 prompt 的每層 residual、cache、GDN state、logits digest 全部和 fork 一致。100 token 的 greedy 第一個 token 是 561。
  - `minicpm_moe_prefill_layer`：第 0 層（K=4）和第 10 層（K=3）在 5、19、99、1280、219 列時，selected、weights、output 都是 0 個元素不同。
  - 27B 全套和 MiniCPM decode 全套也都通過。
- **CPU**：`cargo test --workspace --no-fail-fast` 沒有失敗。
- **clippy 沒過**，`-D warnings` 下有 4 個 lint：
  - 3 個 `too_many_arguments`（8/7、9/7、9/7）。
  - 1 個 `the loop variable expert is only used to index counts`。
- **還沒涵蓋到的路徑**：這批 dump 的最大 expert count 是 1280，低於 batch recon 上限 2048，所以 per-expert 的 `index_add_` 路徑沒有被踩到。fork 在這條路徑上的累加順序本身就不確定。這點寫在 `docs/workspace.md`。
- **下一步**：
  1. 修掉 4 個 clippy lint。
  2. 我看過 diff 後 commit。
  3. 再跑一次全套 GPU。
  4. 照下面的步驟合進 main。

分支已有的 commit，從新到舊：

- `44f3fa6`：MiniCPM 的 q_len cubin 改照自己的 tile 載入。decode 已經在 GPU 上和 fork 一致：
  - `minicpm_linear`、`minicpm_moe_layer`、`minicpm_decode_ref` 通過。
  - `minicpm_decode_seq` 300 步，0 步不同。
  - 27B 的 GPU 測試全過。
- `367fd62`：fork 的 MiniCPM linear 與 decode 參考資料，加上兩個 dump 腳本的修正。
- `886c51f`：把 main 合進來，包含 MTP、DRY breaker 和 decode 寬度。我自己解了 26 處衝突，QTIP 小 m 路徑統一成一套。
- `de86272`：上一輪 grok 的 MoE prefill 半成品，因為輪數上限停在半路。

## 待辦

### MiniCPM

- [x] MoE prefill 在 GPU 上和 fork 逐位元一致：`minicpm_prefill_ref`、`minicpm_moe_prefill_layer`。
- [x] 全套回歸：27B 和 MiniCPM decode 都維持通過。
- [ ] 修掉 `moe_plan`、`moe_prefill` 的 4 個 clippy lint，然後 commit 到 `grok/minicpm`。
- [ ] 如果要涵蓋 `index_add_` 路徑（expert count 超過 2048），需要更長的 prompt 或特別挑過的輸入。fork 自己在這條路徑上也不確定，要先決定比對策略。
- [ ] MiniCPM 開服務實測：用 `--host 127.0.0.1 --port 18889` 試 chat。prompt 現在會走 prefill。
- [ ] 合進 main：
  - 刪 worktree 之前，先把 `~/berth-wt/minicpm/target/triton-cubins/minicpm-v4.7/`（含 `fla/`）複製到 `~/berth/target/triton-cubins/minicpm-v4.7/`。這份 cubin 只在 worktree 裡有。
  - 合併後刪掉 `grok/minicpm`、`grok/mc-kernels`、`grok/mc-shapes` 三個分支和它們的 worktree。後兩個已經合進 `grok/minicpm`。
- [ ] 視覺塔（SigLIP2 + merger）還沒派。規格在 `~/berth-wt/minicpm-host/p3_vision.md`。
- [ ] `testdata/minicpm` 約 14 MB，其中 `linear/` 存了原始 fp16 張量，佔 9.9 MB。要控制 repo 大小的話，可以改成只存 digest。
- [ ] fork 跑 MiniCPM dump 時，在 `~/.cache/exllamav3/autotune` 寫入了 MiniCPM 形狀的 autotune 條目。這是 fork 的正常行為，27B 原有的條目沒有變。之後 berth 讀的就是這份快取。

### MTP（已合進 main，`b6c1af8`）

- [ ] trace 還沒記錄 `block_in` 的 digest（`tests/mtp_ref.rs` 標了 TODO）。
- [ ] ≤17 token 的短 prompt：fork 的 prefill 會走 BC 路徑，berth 不會。這是本來就有的差異，`docs/mtp.md` 的「主持決定」有寫。
- [ ] MTP 開啟時記憶體比較吃緊，文件建議搭配較小的 `--cache-tokens`。實際能開多大還沒在服務上量過。
- [ ] 正式服務 `:8889` 還沒試 `--mtp-steps`。

### 已修好的既有問題（已合進 main，`014fd06`）

- DRY 的 sequence breaker 現在一律裝上 tokenizer 的預設名單。
- 非 MTP decode 的 block table 寬度改成每步依 `position + 1` 計算。
- 新增 `tests/generate_ref.rs`，用 4090 token 跨越 16 頁邊界的 prompt 驗證。兩項修正我都確認過：拿掉任一項，測試就會失敗。

### 環境與流程

- [ ] GPU 功率上限 240 W 是用 `sudo nvidia-smi -pl 240` 設的。persistence mode 沒開，驅動卸載後可能被重設。要保險的話跑 `sudo nvidia-smi -pm 1`。
- [ ] GPU 監看迴圈還在背景跑：pid 938329，腳本在 `/tmp/claude-1000/-home-sam-berth/3c0e0744-2a99-41f0-a207-9a296d144257/scratchpad/gpumon.sh`，log 寫到 `target/gpumon.log`。它不會常駐，用完可以 `kill 938329`。
- [ ] `crates/berth-ptx` 在 main 上本來就沒過 `cargo fmt --check`。
- [ ] `~/berth-test-py`（分支 `chorn/test-py`，`dddb3ee`，HTTP 量測 client）不是這次的工作，狀態不明，我沒動。
- 注意：grok 這幾輪的最終回報幾乎都是空的，也曾經沒跑 GPU 測試就結束。它交回來的東西一律要自己看 diff、自己跑測試。
