# 服務格式：EXL3

berth 服務的權重是 exllamav3 轉出來的 EXL3。目錄裡是 `config.json`、tokenizer、一個或多個 `.safetensors`。不把同一份模型再轉成 GGUF、AWQ 或另一套整數格式來跑。

Strata 那類 Q2 / IQ 的體積（專家約 34 GB、n-gram 表約 29 GB）是另一種量化。那些數字不能拿來排這台機器的泊位。泊位裡的 bytes 是這份 EXL3 checkpoint 的 trellis 和 scale。

## 一條 Linear

`LinearEXL3` 存在 safetensors 裡的是：

| 張量 | dtype | 角色 |
|---|---|---|
| `trellis` | int16，三維 | 尾咬 trellis。最後一維是 `16 * K` 個 uint16；半整數 bitrate 再多 8 |
| `suh`、`svh` | fp16 | 輸入側、輸出側的 Hadamard scale。也接受打包的 `su`、`sv`（int16），載入時解開 |
| `bias` | fp16 | 可選 |
| `mcg` 或 `mul1` | 碼本 | 二選一。半整數 bitrate 必須是 mul1 |

bitrate `K` 由 `trellis.shape[-1] / 16` 讀出。整數是 1 到 8。半整數只有 1.5、2.5、3.5，而且只配 mul1。同一份模型裡每條 tensor 的 `K` 可以不同（`--bits`、`--head_bits`、`--recipe`）。

decode 直接把這三塊餵給已經量過的 kernel：常駐骨架和熱專家走 int8 GEMV（sq、multi、SiLU、fused norm），其餘走 EXL3 GEMM。Hadamard 在 kernel 裡用 `suh` / `svh` 做完，runtime 不先把權重解回 fp16。

## 轉換時分開量化的部分

這些仍是 EXL3，只是 bitrate 和檔案位置不同。預設來自 `convert.py`：

| 參數 | 預設 | 存放 |
|---|---|---|
| `--bits` | 目標平均 bpw | 各層 Linear |
| `--head_bits` | 6 | `lm_head` |
| `--mtp_bits` | 4，或 16 表示不量化 | MTP |
| `--vision_bits` | 驗證過的視覺塔 6，其餘 16 | 視覺塔 |
| `--embed_bits` | 16（fp16 原表） | token embedding。量化後是 256 寬一組的 mul1 ring，可放 RAM 或從磁碟讀 |
| `--ngram_bits` | `--bits` 四捨五入的整數，1 到 8 | 獨立的 `ngram_embedding.safetensors` |

n-gram 行格式是 `exl3_ngram_trellis`（`util/convert_ngram.py`）。一列 160 維，打包成 `(1 + 160 * K / 16)` 個 int16：第 0 個 word 是 fp16 scale 的位元，後面是 mul1 trellis。解碼是 `decode_mul1(state) * scale + head_bias[head]`。hash 和列號在 CPU，讀到的是這一段 EXL3 列，不是另一種表。

## 專家怎麼搬

MoE 的一個 expert 是三條 EXL3 Linear（gate、up、down），或已經融合的 `gate_up_proj` 加 `down_proj`。熱快取和冷拷貝搬的都是該 expert 的 `trellis` + `suh` + `svh`。

檔案裡一個 expert 的這幾塊要連續，miss 時一次 DMA 上 GPU，然後用同一支 GEMV。不為了 PCIe 另做一種量化。先前寫的「冷專家改成好 DMA 的裸 bytes、trellis 只留在常駐權重」不作數。

## KV

cache 沿用 exllamav3 的量化 cache（`-cq`，2 到 8 bit）。這台機器上 Qwen3.8-27B 已量過的部署選擇是 cq4。KV 量化不是權重格式，但它是同一套服務設定，不另發明一種 cache。
