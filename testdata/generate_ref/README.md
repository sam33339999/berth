# testdata/generate_ref

fork（`~/qwen38-exl3-dflash2/exllamav3`，`exllamav3_ext` tree 前綴 `de35ffb`）的 27B 非 MTP generate 參考。不開草稿（`num_draft_tokens = 0`）。`meta.json` 有 6 個 case：prompt `4090` 和 `100`，sampler `greedy`、`sample`、`greedy_dry`。每個 case 32 個新 token。`logits_<prompt>_<sampler>_step0.bin` 是該 case 第一步（最後一個 prompt token 的 decode）的 logits，little-endian fp16、無 header、`[vocab]`。6 個檔共 2979840 bytes。欄位、dtype 和雜湊的位元組寫在 `scripts/dump_27b_generate_ref.py` 開頭的 docstring。

4090 在 32 步裡會把 decode block table 從 16 頁補到 32 頁。100 全程是 16 頁，用來讓 DRY 的 tokenizer 預設 sequence breaker 影響抽到的 token。`greedy` 是 `ArgmaxSampler`（penalty 全關）。`sample` 是正式服務 temperature 0.7 的建法。`greedy_dry` 是正式服務 temperature 0 的建法（DRY 開著，最後是 `SS_Argmax`）。

這些檔案只由該腳本寫出。GPU 必須空著：`:8889` 正式服務佔著 3090 時不要跑。腳本看到 8889 在聽會立刻停；`nvidia-smi` 的記憶體達到 4000 MiB 會等最多 20 分鐘再停。每個 case 先丟兩次相同 job 再錄第三次。fork 工作目錄要乾淨。不要設 `EXL3_BC_ATTN=0`，也不要設 `EXL3_FUSED_SAMPLER`。

再生時用 `scripts/serve-host.sh` 那組環境。`<torch nvidia libs>` 是這個 python 的 torch 底下 `nvidia/*/lib`：

```bash
ss -ltn | grep :8889 && exit 1
PY=/home/sam/qwen38-exl3-dflash2/env/bin/python
NVIDIA_LIB="$($PY -c 'import pathlib,torch;b=pathlib.Path(torch.__file__).resolve().parent.parent/"nvidia";print(":".join(str(p) for p in sorted(b.glob("*/lib")) if p.is_dir()))')"
export LD_LIBRARY_PATH="${NVIDIA_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu
export PYTHONPATH=/home/sam/qwen38-exl3-dflash2/exllamav3
export EXL3_HGEMM_F16ACC=auto EXL3_INT8_GEMV=2 EXL3_INT8_GEMV_MAX_K=5 EXL3_QC_STAGING=1
export PYTHONUNBUFFERED=1
$PY scripts/dump_27b_generate_ref.py
```
