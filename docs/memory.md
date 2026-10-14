# 泊位

每一塊權重有一個泊位。decode 的速度由「這一 token 要從哪一層搬多少 bytes」決定，不是由語言決定。

## 這台機器的上限

- VRAM 24 GB。PCIe 3.0 實際約 12 GB/s。
- RAM 32 GB DDR4-2667。能 pin 來給 GPU 用的大約 18–20 GB。
- SSD 是 ADATA XPG SX8200 Pro。n-gram 用 direct I/O，不進 page cache，否則 29 GB 的表會把 pin 住的專家擠掉。
- i9-9900K 只有 AVX2。miss 的 expert 不要改在 CPU 上算。

## Qwen3.8-27B

整包在 VRAM。每 token 重讀約 13.2 GB 的常駐權重。沒有 RAM 專家，沒有 SSD 表。

## MiniCPM-V 4.7 35B-A3B 和 Qwen3.6-35B-A3B

4-bit 的 MiniCPM 整包放得進 VRAM，見 [models.md](models.md)。這是預設。

溢位時才分成：

- VRAM：骨架、熱專家快取、KV、GDN 狀態
- RAM：剩下的 routed experts，pin 住

視覺塔有圖時載入，不留在 decode 路徑上。

## Qwen3.8-Flash-Next

| 泊位 | 內容 |
|---|---|
| VRAM | 骨架、按頻率留下的熱專家快取、QSA 正在讀的 KV |
| RAM | 沒進快取的專家，pin 住。n-gram 只留很小的 staging buffer |
| SSD | n-gram 整表（direct I/O），和沒 pin 到的冷專家 |

不要每 token 從零搬進 48 層 × 10 個專家。那是約 700 MB，光 PCIe 就要約 60 ms。VRAM 裡的是一份熱快取：命中的直接算，沒中的才從 RAM 換進來。

n-gram 一個 token 只讀 16 列、大約數 KB。GPU 還在跑第 0、1 層時就把這 16 列讀進 pinned buffer。hash 和 token id 歷史留在 CPU。

下一層要哪些專家可以先猜、先搬。計算仍用這一層真正的 router 結果。

長 context 時 KV 可以放 RAM，GPU 只留正在讀的那一段，空出來的 VRAM 讓給熱專家。
