# 模型

四個目標都走同一份 `LayerPlan`，權重都是 EXL3，見 [exl3.md](exl3.md)。下面的體積是 `參數數 × bpw / 8` 的 trellis 主體估算，不含 `suh` / `svh`。實作時以該 checkpoint 的張量為準。Strata 的 GGUF 體積不在這裡用。

## Qwen3.8-27B

Dense hybrid。約 48 層 GDN、16 層 full attention。EXL3 4 bpw 約 13.2 GB，每 token 整份重讀。沒有 MoE，沒有 n-gram 表。

這是第一個目標，直接連 exllamav3 已經量過的 kernel。權重整包在 VRAM。

驗收：鎖頻 1500 MHz，短 context 約 49 tok/s；262k、cq4、沒有草稿約 25 tok/s。

## Qwen3.6-35B-A3B

和 MiniCPM 的文字堆疊同一種形狀：GDN + full attention + routed experts + shared expert。Mixer 可以共用，FFN 換成 MoE。

視覺若是 Qwen 自己的塔，不要用 MiniCPM 的 `vision_tower` 去代。

## MiniCPM-V 4.7 35B-A3B

OpenBMB 的 checkpoint，不是 Qwen 的權重。外層類別 `MiniCPMV4_7ForConditionalGeneration`。README 致謝的語言骨幹是 Qwen3.6。`text_config.model_type` 是 `qwen3_5_moe_text`，因為 Qwen3.6-35B-A3B 和 Qwen3.5-35B-A3B 共用這個架構編號。

文字權重名稱：

- `model.language_model.layers.*.linear_attn.*`（GDN）
- 每四層一個 `self_attn`（第 3、7、11… 層）
- `mlp.experts.gate_up_proj`、`mlp.experts.down_proj`、`shared_expert`

視覺是另一套：`model.vision_tower`（SigLIP2）、圖 980、patch 14、第 6 層插入、canvas 切片、4×/16×。詞彙 248144（Qwen3.5-35B-A3B 是 248320）。草稿是另外的 DSpark，不在主 repo 裡。沒有 n-gram 表。

文字堆疊：40 層、hidden 2048、256 expert、每 token 開 8 個、intermediate 512、每層一個 shared expert。30 層 GDN、10 層 full attention。

| 部分 | 參數 | EXL3 4 bpw 估算 | 每 token |
|---|---|---|---|
| Routed experts | 32.2B | 16.1 GB | 8/256，約 0.50 GB |
| 骨架（GDN、attention、shared、router、embedding、lm_head） | 約 2.4B | 約 1.2 GB | 整份 |
| 視覺塔 | 約 0.5B | 約 0.25 GB | 有圖才跑 |
| 合計 | 35.2B | 約 17.6 GB | 真正計算約 3.5B |

「A3B」是約 2.4B 的常駐骨架，加上當下選中、每個 token 會換的約 1.0B expert。

EXL3 4 bpw 的主體估算約 17.6 GB。full attention 只有 10 層、2 個 KV head、head dim 256，fp16 KV 約 20 KB/token，128k 約 2.5 GB。加起來約 20 GB，放得進 24 GB。embedding 和 lm_head 的預設是 fp16（`--embed_bits 16`），約再多 1 GB 的表，加上同樣大小的 `lm_head` 若用預設 6 bit 則比 4 bpw 略大。GDN 狀態約 60 MB。

這張卡的預設是整包住 VRAM。RAM 只在桌面吃掉 VRAM、或專家用更高 bit 時當溢位。專家若留在 RAM，40 層 × 8 × 約 1.6 MB ≈ 500 MB/token，PCIe 3.0 上拷貝就要約 40 ms。

來源：<https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B>

## Qwen3.8-Flash-Next

48 層（36 GDN、12 QSA），512 expert、top-10，一個 shared expert。主體約 125B，每 token 約 6B。另加約 51B 的 n-gram 表：約 3.2 億列 × 160 維，只在第 2 層注入，8 個 bigram hash 加 8 個 trigram hash。

這張表是模型本體的 PLE。它不是草稿模型，也不是 frozen SAM 語料。

專家是 EXL3 Linear（或融合後的 `gate_up_proj` / `down_proj`），n-gram 是同一目錄裡的 `ngram_embedding.safetensors`（`exl3_ngram_trellis`，`--ngram_bits`）。主體 125B 在 4 bpw 約 62 GB，2 bpw 仍約 31 GB，都超過這台 32 GB 記憶體能 pin 的 18–20 GB。表的大小是 `51e9 × ngram_bits / 8` 加上每列的 scale 和 head bias。具體泊位看那一份 checkpoint 的 `K`，不看 Strata 的 Q2。

泊位見 [memory.md](memory.md)。MTP 約 6 GB，會和專家的 RAM 預算搶，這台先不要加。

QSA 的 KV 是 indexer 選中的 block。Qwen3.8-27B 的 cq3/cq4 結論不能直接搬過來。
