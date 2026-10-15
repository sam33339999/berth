# testdata/mtp_ref

fork（`~/qwen38-exl3-dflash2/exllamav3`，`exllamav3_ext` tree 前綴 `de35ffb`）的 27B MTP speculative decode 參考。一個行程只建一個 `N`（`--num-draft`，整數 2..15）。`meta_N<N>.json` 頂層有 `num_draft_tokens`，`cases` 只有這個 `N` 的 8 筆（prompt `A` / `100` / `1500` / `8190`，sampler `greedy` 然後 `sample`）。`logits_<prompt>_N<N>_<sampler>_round0.bin` 是每個 case 第一個 verify round 的第 0 列 logits，little-endian fp16、無 header。一次呼叫 8 個檔，3973120 bytes，約 3.79 MiB。同一份位元組也寫到 `target/mtp-ref/`，檔名同樣帶這個 `N`。欄位、dtype、shape 和雜湊的位元組寫在 `scripts/dump_27b_mtp_reference.py` 開頭的 docstring。

這些檔案只由該腳本寫出。GPU 必須空著：`:8889` 正式服務佔著 3090 時不要跑。腳本看到 8889 在聽會立刻停；`nvidia-smi` 的記憶體達到 4000 MiB 會等最多 20 分鐘再停。

參考組是兩次呼叫，各一個行程：`--num-draft 4` 和 `--num-draft 2`。GDN graph 的 `(bsz, seqlen, history)` slot 把 conv 寬度和 history stride 烤進這個 `N` 的幾何；`q_len` 1 和 `N+1`、history 開和關，都先跑同一個 case 的兩次完整 warmup，再錄第三次。再生時用 `scripts/serve-host.sh` 那組環境。`<torch nvidia libs>` 是這個 python 的 torch 底下 `nvidia/*/lib`：

```bash
PYTHONPATH=/home/sam/qwen38-exl3-dflash2/exllamav3 \
TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu \
LD_LIBRARY_PATH=<torch nvidia libs>:$LD_LIBRARY_PATH \
EXL3_INT8_GEMV=2 EXL3_INT8_GEMV_MAX_K=5 EXL3_QC_STAGING=1 \
EXL3_HGEMM_F16ACC=auto \
/home/sam/qwen38-exl3-dflash2/env/bin/python \
    scripts/dump_27b_mtp_reference.py --num-draft 4 [META_DIR]

PYTHONPATH=/home/sam/qwen38-exl3-dflash2/exllamav3 \
TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu \
LD_LIBRARY_PATH=<torch nvidia libs>:$LD_LIBRARY_PATH \
EXL3_INT8_GEMV=2 EXL3_INT8_GEMV_MAX_K=5 EXL3_QC_STAGING=1 \
EXL3_HGEMM_F16ACC=auto \
/home/sam/qwen38-exl3-dflash2/env/bin/python \
    scripts/dump_27b_mtp_reference.py --num-draft 2 [META_DIR]
```

`--num-draft` 必填。`META_DIR` 預設 `testdata/mtp_ref`，只管 `meta_N<N>.json` 和這份 bin。`target/mtp-ref` 固定在 repo 的 `target/`。不要設 `EXL3_BC_ATTN=0`，也不要設 `EXL3_FUSED_SAMPLER`。fork 工作目錄要乾淨。
