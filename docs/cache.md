# 多人與快取

日期：2026-10-15。這份是給下一輪一起看的。berth 還沒有自己的服務程式。今天實際在跑的 EXL3 服務是 `~/qwen38-exl3-dflash2`，下面的機制都從那份程式讀來。

## 現在這支服務

`scripts/serve-host.sh` 起一個進程，預設聽 `0.0.0.0:8889`。權重是 Qwen3.8-27B EXL3 加 DFlash2。一個 `Generator`、一塊頁表、一條 worker 佇列。

| 環境變數 | 預設 | 角色 |
|---|---|---|
| `CTX` | 262144 | 單條請求的上下文上限 |
| `CACHE_TOKENS` | 270336 | 頁池的 token 數。262144 再加 DFlash2 的驗證窗 |
| `CQ` | 3 | K 和 V 的量化 bit。2 到 8 |

`scripts/serve_openai.py` 把 `Cache` 建成 `CacheLayer_quant`，`max_batch_size = 1`。HTTP 用 `ThreadingHTTPServer`，很多人可以同時連上。GPU 一次只跑一個 job，下一個在佇列裡等。`/v1/chat/completions` 和 `/v1/completions` 都是這樣。

伺服器不存對話。每個請求自己帶完整的 `messages`，服務端用 tokenizer 的 chat template 渲成 token，做成一個 `Job`。

## 一塊頁池

頁大小是 256（`exllamav3/constants.py` 的 `PAGE_SIZE`）。270336 正好是 1056 頁。池子是這張卡上所有請求共用的，不是每人一份。

量化頁的形狀在 `cache/quant.py`。full attention 的 `token_dim = num_kv_heads * head_dim`。這份 27B 是 4 個 KV head、head dim 256，所以 `token_dim = 1024`。每一層、每一 token：

- K：`token_dim / 32 * k_bits` 個 int32
- V：同樣，用 `v_bits`
- K scale、V scale：各 `token_dim / 32` 個 fp16，每 32 個值一個 scale

16 層 full attention、cq3，約 14 KB/token，整池約 3.6 GB。cq4 約 18 KB/token，整池約 4.6 GB。GDN 層不進這塊量化 KV。DFlash2 另有一塊同樣 token 數的 draft cache，形狀跟草稿自己的 attention 走。

[exl3.md](exl3.md) 寫的部署選擇是 cq4。現在這支 serve 的預設是 cq3。兩邊都是同一套 `-cq`，bit 不同，頁裡的 bytes 不同。

## 什麼算命中

`generator/pagetable.py` 的 `PageTable`。滿頁才有內容 hash：blake2b-128，輸入是上一頁的 hash 再接這一頁 256 個 token 的 bytes。雜湊碰撞當成不會發生，查找時不再比對 token。

`Sequence.prepare` 只哈希 `(len(sequence_ids) - 1) // 256` 頁。最後不滿一頁的前綴沒有條目，每次都算。抽樣參數不進 hash：溫度、DRY、重複懲罰、think budget 變了，頁還是同一頁。

job 結束，`deallocate_pages` 把 `ref_count` 減到 0。頁留在 `unreferenced_pages`，hash 還在。下一個 job 的前綴對上就 `add_ref`，跳過那段 prefill。兩條請求的 token 前綴相同，就用同一串 `CachePage`，不管是不是同一個客戶端。

沒有 salt，沒有 workspace，沒有 per-user 配額。隔離單位是這一個進程。

## GDN 把命中截短

Qwen3.8-27B 的 caps 有 `recurrent_states`（`architecture/qwen3_5.py`）。可用的前綴停在「KV 頁還在，而且這一頁的 hash 上還有 GDN checkpoint」的最後一頁。後面那些只有 KV、沒有 checkpoint 的頁會被算進 `alloc_kv_only_pages`，然後從命中長度裡拿掉，接著重算。

checkpoint 在系統記憶體，`Generator` 預設 `recurrent_cache_size = 4 GiB`。這支 serve 沒有改這個數。間隔：

- 生成中，以及 prompt 尾端靠近生成邊界時：2048 token（模型的 `default_recurrent_checkpoint_interval`）
- 更前面的 prefill：32768 token（`recurrent_checkpoint_interval_pp`）
- prompt 的最後一個滿頁邊界會再存一次（`job.py` 裡 `recurrent_last_page`，間隔傳 `PAGE_SIZE`）

間隔都要是 256 的倍數。checkpoint 從已命中的前綴尾端起算，這樣恢復時 KV 位置和 GDN 狀態對得上。

非 recurrent 模型還會在最後一頁裡做部分 token 比對，把對得上的 KV 複製過來。這條路在 `recurrent_cache is not None` 時直接跳過。27B 走的是這條跳過的路。

GPU 頁被改寫時，若它的 hash 上還掛著 checkpoint，計數 `stashes_stranded` 加一。之後 `prune_stranded` 把對不上活頁的 checkpoint 丟掉。CPU 第二層預設關著（`cpu_cache_size = 0`）。頁一旦讓出去，就沒有地方把它載回來。

## 人多的時候頁怎麼讓

驅逐政策是 `"tree"`。順序：空頁、斷掉的鏈、然後還完整的樹。樹以根的 `access_serial` 由舊到新。同一棵樹從葉子往根收，所以同一條前綴被砍的時候，先掉的是尾巴。有人再次命中，根的 serial 會更新。

一條 262144 的上下文是 1024 頁。池子一共 1056 頁。兩條長對話沒有辦法同時留在 GPU 上。共用的系統提示和工具放在渲染結果的前面，而且每個人的 bytes 相同，這些頁才會合成一棵樹的根，尾巴被砍時還在。

`enable_defrag` 預設開。它只在沒有 referenced page 的空檔整理實體頁，不改變 hash 鏈的身份。

## 客戶端要守的

1. 只連這一個進程。再開一個 serve，就是另一塊空的頁表。
2. 每一輪重送完整 `messages`。模板、tools、`enable_thinking` 維持不變。`enable_thinking` 預設是開的，渲染出來的 token 會差一截。
3. 會變的字放在對話後面。時間戳、使用者名稱放在系統提示裡，整棵樹的 hash 從那一頁起就不同。
4. 助手訊息要原樣送回。思考那段也是 token。客戶端把它剝掉，命中就停在剝掉的那個位置。
5. 想讓好幾段對話同時留著，就讓它們各自的頁數加總仍進得了 1056 頁。共用前綴只佔一次。

## 怎麼看出有沒有命中

worker 結束時印 `prefill` 秒數。同一段對話的下一輪，這段應該短很多。它又回到整段 prompt 那麼久，就是前面的 token 變了，或那些頁已經被蓋掉。

`Job` 的結束事件裡有 `cached_pages` 和 `cached_tokens`。`serve_openai.py` 的 `_usage` 沒有把它們放進回給客戶端的 `usage`。現在對客戶端只能看到 `prompt_tokens` 和 `completion_tokens`。

重啟進程，頁表和 GDN checkpoint 一起沒有。沒有磁碟上的快取。

## 和 berth 規格的對位

| 已經寫下來的 | 這支 serve 的現況 |
|---|---|
| [exl3.md](exl3.md)：cache 沿用 `-cq`，27B 量過的選擇是 cq4 | 預設 `CQ=3` |
| [runtime.md](runtime.md)：cq4 paged attention 留在 nvcc，不重寫 | 頁的配置、hash、驅逐在 Python 的 `PageTable` |
| [runtime.md](runtime.md)：Rust 擁有 KV、GDN、PLE 的 checkpoint 和 rewind | checkpoint 的鍵是滿頁的 hash，間隔如上 |
| [memory.md](memory.md)：長 context 的 KV 可以放 RAM，GPU 只留正在讀的那段 | `cpu_cache_size = 0`，被驅逐的頁直接消失 |

berth 還沒有服務迴圈。第一個目標仍是 27B 自回歸，見 [runtime.md](runtime.md)。這份文件不改那個順序。

## 下一輪要對的

- 頁表這套 hash、樹狀驅逐、以及「沒有 GDN checkpoint 就不算命中」，是要搬進 Rust，還是服務先繼續叫這支 Python。規格寫的是 Rust 擁有 checkpoint 和 rewind，kernel 不重寫。
- cq3 和 cq4 哪一個是 27B 服務的預設。頁池的 bytes 差約 1 GB。
- 第一版服務是不是也是一個 slot、一塊 270336 token 的池。要同時 decode 兩個 job，就得先把單條上下文縮短，並把 `max_batch_size` 從 1 提高。現在的程式拒絕在 262144 上這樣做。
- 32 GB 記憶體裡，GDN checkpoint 已經預留 4 GB。要不要再開 CPU 頁層。開了，被擠出 GPU 的滿頁可以載回；GDN 狀態仍然得在，否則載回的 KV 也不能接著用。
- 這台機器上，相同前綴互相命中是預設。沒有單獨的開關把它們隔開。
