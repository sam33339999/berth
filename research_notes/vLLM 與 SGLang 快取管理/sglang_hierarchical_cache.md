# SGLang 在單一 GPU radix tree 之外的 KV 快取（截至 2026-10-14）

範圍：HiCache 分層、session／多輪、分散式 KV 與 prefill/decode（PD）分離、過載准入、多租戶隔離。GPU 內 RadixAttention 的樹形與 in-GPU scheduler 只寫到「外層如何接上」。另一位研究者負責樹本身。SGLang 官方 HiCache 文件沒有把這套階層拿來對照 vLLM 的 block hash；唯一相鄰敘述是把 LMCache 當成可替換的階層方案（`--enable-lmcache`）。外部論文 KVGov 則把未加鹽的 vLLM 與 SGLang prefix cache 一起當成 timing side channel 的對象。

版本錨點：HiCache 設計與 best practices、session、eviction、PD、hyperparameter 頁皆為 2026-09 至 2026-10 的線上文件。`schedule_batch.py` 為 2026-10-14 讀到的 `sgl-project/sglang` `main`。`HiRadixCache` 是 2025 部落格與 0.5.11 前後的實作類別；2026-08-11 的 Unified Radix Cache 與 2026-09-22 的移除 PR 說自 v0.5.19（PR #35081）起，服務路徑改由 `UnifiedRadixCache` 承接。下文若數字來自 `HiRadixCache` 原始碼快照，會單獨標成可能已被 unified 路徑取代。

## HiCache 的層級、寫入、載回與汰換

### Takeaway

HiCache 把 GPU radix 從「只活在 HBM」擴成三層：L1 GPU、L2 host（兩者都是單一 inference instance 私有），以及可選的 L3 儲存（只有 backend 設成同一 namespace 才能跨 instance）。請求先在本地樹做 local match，未命中的連續前綴再向 L3 prefetch 進 L2，最後用 load-back 把 host 上的 KV 拉回 GPU 才算 prefill。GPU→較慢層的寫入政策是 `write_through`、`write_through_selective`、`write_back`；L1/L2 的受害者選擇沿用 radix eviction（外加 session 軟保護），L3 的淘汰交給儲存 backend，樹不常駐 L3 metadata。

### Cited Findings

- RadixAttention 用閒置 GPU 記憶體重用 prefix KV；HiCache 把同一想法延伸到 host 與分散式儲存。文件明寫類比 CPU：GPU 為 L1、host 為 L2、分散式儲存為 L3，並可接 Mooncake、3FS、NIXL、AIBrix KVCache。 — [HiCache design](https://docs.sglang.io/docs/advanced_features/hicache_design)
- 層級範圍（同一頁的 Tier Sharing Scope）：L1 是該 instance 的 GPU HBM，不跨 instance。L2 是該 instance 程序擁有的 host DRAM，連同一台機器上的兩個 instance 也不互讀。L3 backend 可以是 `file`、`mooncake`、`hf3fs`、`nixl`、`aibrix` 或自訂；`file` 預設寫 `/tmp/hicache`（`SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR` 可改），除非該路徑是共享掛載，否則只在節點內。`mooncake` / `hf3fs` / `nixl` / `aibrix` 只有在每個 instance 指向同一 namespace 與設定時才是叢集範圍。調大 `--hicache-ratio` 或 `--hicache-size` 只放大自己的 L2，不能把多機 host 記憶體合成一個 L2。 — [HiCache design](https://docs.sglang.io/docs/advanced_features/hicache_design)；同段摘要亦見 [best practices](https://docs.sglang.io/docs/advanced_features/hicache_best_practices)
- 2025-09-10 LMSYS 文章把索引結構叫做 HiRadixTree：「acts as a page table for referencing KV caches residing locally in GPU and CPU memory」，cache controller 負責在 GPU pool、CPU pool，以及「disks and remote memory」之間 load 與 backup。 — [LMSYS HiCache blog, 2025-09-10](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/)
- 現行設計文件：HiRadixTree 的每個節點是一段連續 token 的 KV，並記錄這段在 GPU、CPU、L3 或其中多層。本地層有精確位址。為了少同步，樹不保存、也不持續同步 L3 metadata；存取 L3 時即時問 backend 是否存在、在哪台機器。 — [HiCache design](https://docs.sglang.io/docs/advanced_features/hicache_design)
- 三個主操作是 local match、prefetch、write-back。新請求先搜本地 L1/L2；本地沒有的部分嘗試從 L3 prefetch；prefetch 之後所需 KV 都進 GPU 再算；prefill 完成後再考慮把新資料寫進 L2 或 L3。 — [HiCache design](https://docs.sglang.io/docs/advanced_features/hicache_design)
- Local match：從根沿 token prefix 走。`page_size > 1` 時以 page 為粒度。比對停在節點中間就 split。回傳的是一段連續前綴，「the first part residing in L1 and the latter part in L2」。這一步不拷貝資料。 — [HiCache design](https://docs.sglang.io/docs/advanced_features/hicache_design)
- L3 prefetch 觸發：local match 之後，對 L1/L2 都沒有的部分查 L3 metadata；L3 命中長度超過門檻才 prefetch。預設門檻 256 tokens，可改。三個停止策略，旗標 `--hicache-storage-prefetch-policy {best_effort,wait_complete,timeout}`：`best_effort` 是 GPU 已能開始 prefill 就停，不等；`wait_complete` 等全部 prefetch 完；`timeout` 在完成或逾時時停。文件建議生產用 `timeout`。`timeout` 公式是 `min(prefetch_timeout_max, prefetch_timeout_base + prefetch_timeout_per_ki_token * num_token_to_fetch / 1024)`，預設 `prefetch_timeout_base=2` 秒、`prefetch_timeout_per_ki_token=0.1` 秒／1024 tokens、`prefetch_timeout_max=30` 秒。停掉之後，已經拉回來的資料加上本地資料一起做 prefill。 — [HiCache design](https://docs.sglang.io/docs/advanced_features/hicache_design)
- 多 rank（例如 TP）在 prefetch 用 `all_reduce(op=min)`，讓大家看到相同的 L3 hit 長度；prefetch 結束或中止後再 `all_reduce(op=min)`，共識實際取回的 prefix 長度。 — [HiCache design](https://docs.sglang.io/docs/advanced_features/hicache_design)
- 寫回三政策，旗標 `--hicache-write-policy {write_back,write_through,write_through_selective}`。`write_through`：每次存取立刻寫下一層，頻寬夠時快取效益最強。`write_through_selective`：命中次數超過門檻才寫，只備份熱資料。`write_back`：只在從較快層被驅逐時才寫下一層。L2→L3 只傳 L3 還沒有的資料。 — [HiCache design](https://docs.sglang.io/docs/advanced_features/hicache_design)；政策的一句話版本亦見 [2025-09-10 blog](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/)
- GPU 命中但 CPU miss 的反方向（CPU hit、GPU miss）走 layer-wise overlap：算第 N 層時同時把第 N+1 層的 KV 從 CPU 載入。外部儲存則是命中後機會式 prefetch 進 host，因為儲存延遲高且不穩；作者表示對 GPU Direct Storage 保持開放。CPU–GPU 除了 `cudaMemcpyAsync`，還有 GPU-assisted I/O kernel，部落格寫「up to 3× higher throughput」。Host pool 可改成 page-first，以便對 L3 做較大、零拷貝的傳輸；部落格寫典型部署可到 2×。 — [2025-09-10 blog](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/)；overlap 與 kernel 亦寫在 [HiCache design](https://docs.sglang.io/docs/advanced_features/hicache_design)
- 傳輸旗標：`--hicache-io-backend {direct,kernel}`（`direct` 是 CUDA copy，`kernel` 是 GPU-assisted，文件建議 kernel）。`--hicache-mem-layout {layer_first,page_first,page_first_direct}`。GPU 計算自然是 layer-first。`page_first` / `page_first_direct` 把同一 page 的 KV 放在連續記憶體，整頁當一個物件零拷貝進 L3。`page_first` 從 L2 回 GPU 要按「每層一個 token」搬；`page_first_direct` 把一頁裡同一層的 token 排在一起，可按 page-layer 聚合。Best practices：`page_first` 只相容 `kernel`，配 `direct` 會自動改 `layer_first`；`page_first_direct` 是給 `direct` 的。 — [HiCache design](https://docs.sglang.io/docs/advanced_features/hicache_design)；[best practices](https://docs.sglang.io/docs/advanced_features/hicache_best_practices)
- MLA：每個 TP rank 都持有完整且相同的 token KV，因此只讓一個 rank 發起 write-back，避免重複存。MHA 則是每 rank 持有 `1/tp_size`。 — [HiCache design](https://docs.sglang.io/docs/advanced_features/hicache_design)
- 異質 TP：GQA/MHA 若不同部署的 TP（例如 4 與 8）共用同一 L3 namespace，在 `--hicache-storage-backend-extra-config` 設 `tp_lcm_size` 為各 TP 的最小公倍數。Mooncake 加 `page_head` layout 時會依這個值切 head shard。大家 TP 相同就不必設。 — [best practices](https://docs.sglang.io/docs/advanced_features/hicache_best_practices)
- 容量旗標：`--enable-hierarchical-cache` 才算打開 HiCache。`--hicache-ratio` 是 host pool 相對 device pool 的倍數，必須大於 1（實作要求 host KV 大於 device KV）。`--hicache-size` 以 GB 指定 host pool，且覆寫 ratio；單位是 `1GB = 1e9` bytes，而且是每個 rank。例：`--hicache-size 30` 在 8 rank 上總共 240GB，且仍必須大於該 rank 的 device KV。`--page-size` 是每頁 token 數；頁越大，metadata 與 I/O 越好，但只命中頁的一部分時 hit rate 較差。 — [HiCache design](https://docs.sglang.io/docs/advanced_features/hicache_design)
- L3 介面類別是 `HiCacheStorage`。文件與部落格都說接新 backend 只要 `get`、`exists`/`exist`、`set`；排程與同步由 cache controller 做。內建列舉：`--hicache-storage-backend {file,mooncake,hf3fs,nixl,aibrix,dynamic}`。`dynamic` 用 extra-config 的 `backend_name`、`module_path`、`class_name`，可選 `interface_v1` 控制 `batch_get_v1` / `batch_set_v1`。註冊點是 `BackendFactory`。另有 `--enable-lmcache` 與 `--lmcache-config-file`，文件稱 LMCache 是 HiCache 的替代方案，不是第 N 個內建 tier。Storage backend 可在不重啟的情況下用 HTTP admin attach/detach。 — [HiCache design](https://docs.sglang.io/docs/advanced_features/hicache_design)；[best practices](https://docs.sglang.io/docs/advanced_features/hicache_best_practices)；[2025-09-10 blog](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/)
- 類別演進。2025 部落格與 GitHub 上的 `python/sglang/srt/mem_cache/hiradix_cache.py` 是 `class HiRadixCache(RadixCache)`，內含 `HiCacheController`、`ongoing_write_through`、`ongoing_load_back`、`ongoing_prefetch`、`ongoing_backup`。該快照裡 `write_through_threshold` 在政策為 `write_through` 時是 1，否則是 2；`load_back_threshold = 10`（旁邊有 “todo: more loading policies”）。`write_backup` 經 `cache_controller.write` 做 device→host；host 滿了就先 `evict_host` 再寫。`writing_check` 等 DMA 的 `finish_event`，完成後若開了 storage 再 `write_backup_storage`。`load_back` 把 `node.host_value` 拷回 device indices。 — [hiradix_cache.py（GitHub 索引中的 main 快照）](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/hiradix_cache.py)；[issue #19212，2026-02-24，呼叫鏈 `init_load_back` → `load_back` → `evict` → `write_backup` → `evict_host`](https://github.com/sgl-project/sglang/issues/19212)
- NVIDIA Dynamo 對 `HiRadixCache` 的 KV event（需要 SGLang ≥ 0.5.11，該版含「emit KV events for L2 host cache insertions」；Dynamo 1.3.0 映像帶 SGLang 0.5.15）把層間遷移記成：fresh prefill `store(GPU)`；GPU→host 的 DMA 完成後 `store(CPU_PINNED)`；GPU 被驅逐但 host 還在則 `remove(GPU)`；host 也被驅逐則 `remove(CPU_PINNED)`；host→GPU 的 `load_back` 是 `store(GPU)`；L3→host prefetch 是 `store(CPU_PINNED)`。`store(新層)` 先於 `remove(舊層)`。`CPU_PINNED` 是線上字串，概念名是 Host。 — [Dynamo HiCache](https://docs.nvidia.com/dynamo/knowledge-base/modular-components/backends/sg-lang/hi-cache)
- 2026-08-11 LMSYS／2026-10 的 SGLang 視覺化文章改寫成：HiCache 延伸的是 Unified Radix Cache，同一棵樹橫跨 GPU L1、host L2、外部 L3。`UnifiedRadixCache` 做 pool orchestration，`UnifiedTreeCore` 做 match/split/insert/lock/evict。節點上每個 component（FULL、SWA、MAMBA）一個 slot。FULL 與 SWA 存 page index，MAMBA 在前綴端點存 checkpoint，重用時 copy-on-write。輔助 pool（壓縮 KV、indexer、compressor）當 sidecar，跟著所屬 component 的 index 與傳輸走。DeepSeek-V4 是 FULL+SWA（3 個 sidecar 跟 FULL、2 個跟 SWA），Kimi-K3 是 FULL+MAMBA，Inkling 三者都有。安全重用邊界是「所有 component 都接受的最深邊界」。 — [SGLang unified radix blog](https://www.sglang.io/blog/unified-radix-cache)；源文 [LMSYS, 2026-08-11](https://www.lmsys.org/blog/2026-08-11-unified-radix-cache/)
- PR #40787（2026-09-22，作者 hnyls2002）寫：「HiCache has gone through `UnifiedRadixCache` since #35081 (first released in v0.5.19), so `HiRadixCache` has no callers」。本筆記沒有再確認該 PR 是否已合併、檔案是否已從 main 刪除。Eviction 文件說 `tlru`「Requires the unified radix cache (the default tree)」，所以 2026-10-12 的文件已把 unified tree 當預設。 — [PR #40787](https://github.com/sgl-project/sglang/pull/40787)；[eviction policies](https://docs.sglang.io/docs/advanced_features/radix_eviction_policy)
- GPU／radix 受害者：只考慮 evictable leaves（KV 在、沒被 in-flight lock、沒有仍握有 KV 的子節點）。根永不驅逐。分數最低者先走；葉子走了，父節點可能變成葉子，所以是從尖端走回根。政策只評分，不能 pin；其他人都清掉仍不夠時，受保護節點還是會被趕走。旗標 `--radix-eviction-policy`。CLI 可選：`lru`（預設，最久沒用）、`lfu`（命中最少，再比 LRU）、`slru`（先清 probationary，再段內 LRU）、`priority`（請求 priority 最低者，再 LRU）、`tlru`（先清對話尾端裡下一輪 prefill 用不到的 “TEL-safe” 以外 token）。平手都退回 LRU。`fifo` / `mru` / `filo` 在 registry 裡，但不在 CLI，只給 out-of-tree 實驗。Hit count 不會因為 chunked-prefill 的步驟、已驅逐節點，或 write-back HiCache 而增加。`priority` 留下插入該前綴的請求裡最高的 priority；沒開 priority scheduling 時大家都是 0，等同 `lru`。`slru` 的 `protected_threshold` 預設 2。`tlru` 要 `threshold` 與 `next_prompt_estimate`，且 threshold 必須大於估計值，否則退化成 LRU 並在啟動時被拒。論文是 arXiv:2510.15152，論文用 block，設定時要乘 `--page-size`。 — [eviction policies，文件頁 2026-10-12](https://docs.sglang.io/docs/advanced_features/radix_eviction_policy)
- Host 層：session 文件寫 device 與 host eviction 用同一套 session 偏好。`write_back` 在 host 壓力下會先收回「GPU 上仍有副本」的 host 複本（PR #33777，2026-08-06 提出、2026-08-07 合併進 main）。PR #26728 說明 load-back 在 `cache_controller.load()` 之前就要握住 host lock，因為 write-back 壓力下 device eviction 或 load 可能先觸發 host eviction。Issue #12875（2025-11-08）曾指出 `HiRadixCache.evict` 在 prefetch 之後才呼叫，host 不夠時 `prefetch_from_storage` 會失敗，因為 GPU 上的節點不能從 host 被 evict。Issue #19212（2026-02-24，commit `feb041f4e`）則是 `write_back` 下 `evict()` 走到 `evict_host()` 時改了樹，造成 `AssertionError: parent does not have child key`。 — [session radix](https://docs.sglang.io/docs/advanced_features/session_radix_cache)；[PR #33777](https://github.com/sgl-project/sglang/pull/33777)；[PR #26728](https://github.com/sgl-project/sglang/pull/26728)；[issue #12875](https://github.com/sgl-project/sglang/issues/12875)；[issue #19212](https://github.com/sgl-project/sglang/issues/19212)
- L3 淘汰：unified 文章寫「Session retention applies to L1/L2; the storage backend manages L3 eviction。」HiRadixTree 不持續保存 L3 metadata。 — [unified radix blog](https://www.sglang.io/blog/unified-radix-cache)；[HiCache design](https://docs.sglang.io/docs/advanced_features/hicache_design)
- 2026-10-08 的路線圖仍把「同一 supernode 內跨 engine 共享 L1/L2」列為未完成工作（NVLink、AMD UALoE、遠端 Grace DRAM／EGM）。這與「今天的 L2 不跨 instance」一致，不能把它寫成已上線。 — [issue #43098](https://github.com/sgl-project/sglang/issues/43098)

### Inferences

- 對 scheduler 來說，外層不是第二棵平行的樹，而是同一條前綴路徑上的駐留位元：節點可以同時標成在 L1、L2、L3。GPU radix 的 match／lock／evict 仍是熱路徑；HiCache 加上的是「L1 miss 但 L2/L3 hit 時先搬再算」以及「什麼時候把新 KV 往下備份」。
- `write_through` 用頻寬換「慢層盡量熱」；`write_back` 讓 L2/L3 只接住快被趕出 GPU 的資料，所以慢層容量效益較高，但命中次數統計在 write-back 下不增加，`lfu`/`slru` 會變鈍。`write_through_selective` 介在中間。
- 跨使用者、跨機器的重用只可能發生在 L3，而且 key 的 namespace（下面的 `extra_key` / `cache_salt` / LoRA）必須一併進儲存鍵，否則會變成跨租戶共用。設計文件說 L3 可以跨 SGLang instance 共享，同時 `RadixKey` 的註解說 `extra_key` 與 `cache_salt` 會 namespace tree 與 storage（見下一節）。
- `HiRadixCache` 快照裡的 threshold（write-through 1 次、其他政策 2 次、`load_back_threshold = 10`）描述的是 unified 之前的控制面。2026-10 的設計文件只保留三個具名政策與 256-token prefetch 門檻，沒有重述這三個整數。報告不應把 1/2/10 寫成 v0.5.19 之後的保證。

### Gaps

- 沒有在 2026-10-14 重讀 `UnifiedRadixCache` 的 write／load-back 原始碼，因此不能證明 `write_through_threshold` 與 `load_back_threshold` 被原樣搬過去。
- 各 L3 backend（Mooncake、3FS、NIXL、AIBrix、file）自己的淘汰與 TTL 沒有在 HiCache 設計頁展開。只知道「L3 eviction 歸 backend」。
- Best practices 的 PD+HF3FS 範例同時寫 `--hicache-ratio 2` 與 `--hicache-size 0`，但設計頁說一旦設定 `--hicache-size` 就覆寫 ratio，且必須大於 device KV pool。`0` 是「忽略、改走 ratio」還是「不配置 host」沒有找到解釋。 — 兩頁並陳：[best practices 範例](https://docs.sglang.io/docs/advanced_features/hicache_best_practices)、[design 對 size 的約束](https://docs.sglang.io/docs/advanced_features/hicache_design)
- PR #40787 的合併狀態未再查。

## 多輪、session，以及什麼會打壞重用

### Takeaway

OpenAI 相容伺服器不會自動記住對話。客戶端每一輪都要送進完整、token 化後能接上前綴的 prompt，radix 才會命中；`session_id` 只登記「這段 KV 仍被哪個會話參考」，不補歷史、也不 pin 記憶體。預設同一進程、同一模型權重下，相同 token 前綴跨請求共享，包含不同呼叫者，除非用 `cache_salt`、`extra_key` 或 LoRA 的 `lora_id` 把樹分開。取樣參數不進入 radix key；影像則會，因為每個多模態 item 的 hash 會變成 pad token 插進序列。

### Cited Findings

- Radix cache 預設開啟。`--disable-radix-cache` 的說明是「Disable RadixAttention for prefix caching」，預設 `False`。權重被 `/update_weights` 更新時會自動打 `/flush_cache`，把 radix cache 清掉。 — [server arguments 快照中的 `--disable-radix-cache`](https://docs.sglang.io/docs/advanced_features/server_arguments)；[native API：`/flush_cache`](https://docs.sglang.io/basic_usage/native_api.html)
- Session-aware radix 只由 `UnifiedRadixCache` 實作，旗標 `--enable-session-radix-cache`。應用要在同一段會話的每個請求帶同一個頂層 `session_id`。文件原句：「The ID labels cache references only; it does not append or reconstruct conversation context, so each request must still contain the intended prompt。」請求結束時，SGLang 自動把可重用的 cache leaves 登記到這個 `session_id`，不需要 `/open_session`。`POST /close_session` 只拿掉該 session 的參考，不立刻釋放 KV；KV 仍可重用，並回到一般淘汰順序。參考是軟保護，不是 pin；把未參考的 KV 清光仍不夠時，被參考的 KV 還是會被驅逐。 — [session radix，文件頁 2026-09-23](https://docs.sglang.io/docs/advanced_features/session_radix_cache)
- 同一頁的淘汰順序，device 與 host 相同。FULL：先未參考節點，再依 session 參考數由少到多，然後才是設定的政策（例如 LRU）；保護的是從登記葉子到根的可重用前綴。SWA：兩輪 LRU，先未參考再已參考；保護的是滑動窗加上 page 對齊的尾端。Mamba：同樣兩輪 LRU，保護登記葉子上的 state。Cascade：趕掉內部 FULL 節點會連帶 SWA 與 Mamba；趕 SWA 會連帶 Mamba；只趕 Mamba 不影響另外兩個。趕葉子會清掉該葉子上所有 component。L3 不吃這套 session 保護。 — [session radix](https://docs.sglang.io/docs/advanced_features/session_radix_cache)；L3 句見 [unified radix blog](https://www.sglang.io/blog/unified-radix-cache)
- 2026-10-14 的 `main` `schedule_batch.py`：`Req` 同時有 `extra_key`、`cache_salt`、`lora_id`、`session_id`。若 `lora_id is not None`，則 `extra_key = (extra_key or "") + lora_id`。`cache_salt` 另存，不在這一行被拼進去。GitHub 上 `RadixKey` 的註解（索引到的 `radix_cache.py`）寫：`extra_key`「Namespaces the tree and storage; omitted from KV events」；`cache_salt`「Namespaces the tree, storage and KV events」。`match_prefix` 的文件字串寫邏輯 namespace 由 token id 序列加上 `RadixKey.extra_key` 決定。 — [schedule_batch.py on main](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py)；[radix_cache.py 索引片段](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)
- Issue #24568（2026-05-07，commit `9e1336d`）說當時 OpenAI 路徑的 `OpenAIServingBase._compute_extra_key()` 把 `cache_salt` 與 `extra_key` 接在一起，in-memory radix 有依 salt 隔離（實驗裡不同 salt 的 `cached_tokens` 行為被作者用來證明隔離），但發出的 `BlockStored.block_hashes` 不含 `extra_key`，token 相同、salt 不同的頁 hash 相同。欄位當時有 `block_hashes`、`parent_block_hash`、`token_ids`、`block_size`、`lora_id`、`medium`，沒有 namespace。這是 event／外部索引的洞，不是單機 prefix match 算錯。2026-10 的 `RadixKey` 註解若仍準確，`cache_salt` 已被單獨拿去 namespace KV events，`extra_key` 仍不進 event。本筆記沒有重跑該 issue。 — [issue #24568](https://github.com/sgl-project/sglang/issues/24568)
- Issue #24638（2026-05-08）報告：base 模型請求若把 `extra_key` 設成某個已載入 LoRA 的 `lora_id`，會重用該 LoRA 算出的 radix KV，確定性輸出因此改變。作者歸因就是上面的字串拼接，以及 `RadixKey.child_key()` 只看拼完的 `extra_key`。2026-10-14 的 `Req.__init__` 仍是 `(extra_key or "") + lora_id`。較早的 PR #7216 曾用獨立的 `LoRARadixCache`，並註明當時 LoRA+radix 尚不支援 `page_size > 1`、非 FCFS schedule、以及 hierarchical cache。那條限制是否在 2026-10 仍在，沒有用現行測試確認。 — [issue #24638](https://github.com/sgl-project/sglang/issues/24638)；[PR #7216](https://github.com/sgl-project/sglang/pull/7216)
- 多模態：`MultimodalDataItem` 的註解寫「Each item has its own hash and pad_value, enabling per-image RadixAttention caching。」`pad_value = MM_PAD_SHIFT_VALUE + (hash % 2^30)`，`MM_PAD_SHIFT_VALUE = 1_000_000`。`build_padded_input_ids` 用這個 pad 蓋過影像 token 區間，所以 radix 看到的 token 序列含影像身份。hash 來自 feature 或 precomputed embeddings（`resolve_multimodal_item_hash`）。不同影像位元組會得到不同 hash，前綴在該處岔開；相同 hash 才能沿同一條 radix 路徑重用。 — [schedule_batch.py on main](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py)
- 2026-02-17 的 server arguments 快照有 `--enable-prefix-mm-cache`：「Enable prefix multimodal cache. Currently only supports mm-only。」預設 `False`。這是另一個開關，不是上面的 pad-token 路徑。2026-10 的 server arguments 頁沒有在本輪研究裡逐行核對這個旗標是否還在。 — [2026-02-17 server arguments 源頁](https://docs.sglang.io/_sources/advanced_features/server_arguments.md)
- 取樣參數存在 `Req.sampling_params`，不出現在上述 `extra_key`／`cache_salt` 的組鍵程式裡。PD rebootstrap 組 payload 時會帶上 temperature、top_p、top_k 等，但那是重算請求的取樣設定，不是 radix 鍵的一部分；同一份 payload 仍單獨帶 `extra_key` 與 `cache_salt`。 — [schedule_batch.py，`build_rebootstrap_payload`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py)
- HTTP PD 下的 Responses API：前景 `POST /v1/responses`（含 `stream=true`）可以，但 store、`previous_response_id`、background、依 id 取回或取消都不支援。原因是 prefill 與 decode 是不同進程，process-local Responses store 無法共用；文件要求「Send explicit conversation history」。內建 `web_search`、`code_interpreter` 也不支援，因為 router 對每個 HTTP 請求只派一對 prefill/decode。PR #39122 讓 `--enable-response-store` 在 PD 下被拒，並對 stateful 請求回 HTTP 400；舊版不能假設一定會在准入時拒絕。 — [PD disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)
- 排程側若要多吃共享前綴，hyperparameter 文件建議 `--schedule-policy lpm`（longest prefix match）：把請求重排以提高 cache hit，但排程開銷較大。這不改變鍵，只改變誰先被算。 — [hyperparameter tuning，2026-10-12](https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning)
- EAGLE 使用 `RadixKey.is_bigram`，鍵的形狀與一般 token 序列不同，因此與非 bigram 路徑不是同一套前綴。 — [radix_cache.py 索引中的 `RadixKey` 註解](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)

### Inferences

- 多輪「自動快取」的真正條件是：客戶端把系統提示、歷史與新一輪串成穩定前綴再送來。OpenAI chat 若每次都重送完整 `messages`，且 chat template 穩定，就會命中。伺服器不會用 `session_id` 或 API 會話去補上一次的輸出。
- 取樣本身不使快取失效。它只會在「上一輪抽出的 token 被拼進下一輪」時間接改寫前綴。溫度不同但提示 token 相同，仍應共享 KV。
- 換模型或換權重：另一個 server 進程的 L1/L2 本來就不共享；同一進程在 `/update_weights` 時會 flush。L3 只有在 namespace、頁大小、TP 切分（`tp_lcm_size`）與模型權重都相容時才可能重用。文件沒有寫「權重版本自動進 L3 鍵」。
- 換 LoRA 應走不同的 `extra_key`。現行拼接方式讓使用者自填的 `extra_key` 有機會冒充 `lora_id`，這是隔離缺陷，不是功能。
- 影像快取失效的單位是 item hash，不是「有沒有圖片」這個布林值。同一張圖、同一處理結果可以跟文字前綴一起重用；重新編碼導致 hash 變了就從該圖的 pad 區間開始 miss。

### Gaps

- 沒有讀到 OpenAI HTTP 層在 2026-10 是否仍把 `cache_salt` 拼進 `extra_key`，還是已經填進 `Req.cache_salt`。`Req` 兩個欄位都在，但 serving adapter 的接線沒有核對。
- 沒有官方文件寫「temperature／top_p 不參與鍵」。上一節的推論來自 `Req` 組鍵程式沒有讀取 `sampling_params`，不是產品文件的保證。
- `--enable-prefix-mm-cache` 與 pad-token radix 的分工，在 2026-10 文件裡沒有對上。
- 沒有找到「API key 自動成為 cache namespace」的官方行為。見多租戶一節。
- `input_embeds` 路徑在 retract 時會丟掉已生成的 `output_ids` 並整段重跑（見下節）。它與 radix 鍵的關係沒有另找文件。

## 分散式 KV：Mooncake、NIXL，以及 prefill/decode 分離

### Takeaway

PD 分離把「算出 KV」與「拿著 KV 逐 token 解碼」拆到不同進程。傳輸引擎是 Mooncake（RDMA／RoCE、可選 NVLink、AWS 上可選 EFA）或 NIXL（預設 UCX，也可 LIBFABRIC），Ascend 另有 memfabric 與 Mooncake 路徑。預設是 decode 在 prefill 結束前就預留目的 KV slot，prefill 用傳輸引擎把 KV 寫進 decode 的 GPU pool；算完之後 KV 住在 decode 端。這條 P→D 傳輸不是 HiCache。HiCache 是另一條軸：可以只開在 prefill 上讓多個 prefill 經 L3 共享，或讓 decode 非同步把 KV 卸載回 L3，供下一輪 prefill 重用。較新的 `prefill_complete` 配置政策明確不支援 decode radix 與 decode HiCache。

### Cited Findings

- 動機：prefill 算完整個輸入，decode 被 KV 容量綁住。合一引擎裡，新的 prefill batch 會打斷 decode，DP attention 也可能一邊 prefill、一邊 decode。PD 把兩段拆開。目前文件寫支援的傳輸引擎是 Mooncake 與 NIXL（同頁後面還有 ASCEND）。 — [PD disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)
- 角色旗標：`--disaggregation-mode prefill|decode`。Router：`python -m sglang_router.launch_router --pd-disaggregation --prefill <url> --decode <url>`。Mooncake 需要 `uv pip install mooncake-transfer-engine`。IB 裝置 `--disaggregation-ib-device` 可以是共用清單（`mlx5_0` 或 `mlx5_0,mlx5_1`）、per-GPU JSON，或 JSON 檔路徑。Mooncake 整合說明寫這條路徑基於 SGLang PR #4654 與 #4880，KV 以 Transfer Engine 在節點間用 RDMA 搬 block。 — [PD disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)；[Mooncake PD 說明](https://kvcache-ai.github.io/Mooncake/deployment/integrations/sglang/pd-disaggregation.html)
- Decode 配置政策。預設（文件稱為既有行為，`early`）是 decode 在 prefill 完成前就為該請求保留目的 KV slot，長 prefill 會佔著這些 slot。`--disaggregation-decode-allocation-policy prefill_complete` 必須 prefill 與 decode 兩邊一起開：prefill 不必等目的 slot 就能算，接受最後一段 prefill 結果後才發 readiness；decode 那時才保留 slot 並做既有的 KV transfer handshake。它改的是每請求的保留時機，不改全域 KV pool 的大小。代價是 KV 在 prefill 上留更久，而且 chunked prefill 與 KV 傳輸的重疊被拿掉。Chunked prefill 不會因為 decode 還沒准入就讓出。傳輸成功後 prefill 清掉 room 的 readiness 與 transfer 狀態再通知 decode；retraction recovery 可以重用同一個 bootstrap room。 — [PD disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)
- `prefill_complete` 的第一版實作要求：Mooncake（RDMA 或 `mooncake_tcp`）、Python bootstrap registry、attention TP/CP 與 pipeline size 為 1（attention DP 可以，但每個 DP rank 的 attention TP 必須是 1）。不支援 staging buffer、執行期 PD 角色切換、PD multiplexing、encoder disaggregation、decoupled speculation、HiSparse、decode radix cache、decode HiCache，以及非 DSA 的輔助 state pool。Prefill HiCache 在這個政策下只限 L2、`write_back`、且沒有 storage backend。不相容的組合會在驗證時失敗。兩邊必須同一 revision、同一政策。 — [PD disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)
- 逾時：prefill 側 `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` 預設 300 秒，是等目的 KV index；`prefill_complete` 下這個計時改從最後一段 prefill 結果之後開始。Decode 側 `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` 預設 300 秒。文件說若可接受較高平均 TTFT，可設成 600。Queue：`SGLANG_DISAGGREGATION_QUEUE_SIZE` 預設 4；設成 1 就是 FCFS 一個一個傳。執行緒池 `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE` 預設 `int(0.75 * cpu_count) // 8`，再夾在 4 與 12 之間，單位是每個 TP rank。`SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES` 預設 0（關閉），用來把同步的 all-layer 傳輸切成有序子批；custom memory pool 的 layerwise 路徑不變。 — [PD disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)
- NVLink：`SGLANG_MOONCAKE_CUSTOM_MEM_POOL` 可為 `NVLINK`（或 `True`）、`BAREX`、`INTRA_NODE_NVLINK`。NVL72 建議 `NVLINK` 加 `MC_FORCE_MNNVL=True`。A100/H20/H100 這類節點內 NVLink 用 `INTRA_NODE_NVLINK` 加 `MC_INTRANODE_NVLINK=true`。輔助資料仍走 TCP。 — [PD disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)
- AWS EFA：套件 `mooncake-transfer-engine-efa-cuda13==0.3.13.post1`，`MOONCAKE_PROTOCOL=efa`。libfabric EFA 目前不能傳 CUDA VMM。因此不要把 custom mem pool 設成 `NVLINK` 或 `BAREX`（`INTRA_NODE_NVLINK` 可以，因為用預設 CUDA allocator），不要開 `expandable_segments:True`，不要設 `MC_FORCE_MNNVL` 或 `NCCL_MNNVL_ENABLE`。EFA 的 KV 使用 CUDA VMM，post-capture KV sizing 會被自動關掉。 — [PD disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)
- 異質 TP 的 GPU staging buffer：prefill 與 decode 的 attention TP 不同時，KV layout 不同。Staging 在 prefill 把 head slice gather 成連續 buffer，整批 RDMA，再在 decode scatter 進頁。文件寫相對預設的 per-token slice「2–5x throughput improvement」，高併發下與同質 TP 基線差距在約 5% 以內。環境變數 `SGLANG_DISAGG_STAGING_BUFFER`（預設 false）與 `SGLANG_DISAGG_STAGING_POOL_SIZE_MB`（預設 4096，decode 側 ring）。兩邊 TP 相同即使打開也會 bypass。給非 MLA（GQA/MHA）；DeepSeek-V2/V3 這類 MLA 不應開。混合 SWA 模型可以用不同 attention TP，Mooncake 或 NIXL 都以 token 與 KV-head slice 傳 SWA KV；staging 只帶 full-attention KV。此配置要用 `--pp-size 1`、不要 `--enable-unified-memory`，且不支援 MXFP8 KV。 — [PD disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)
- NIXL：`--disaggregation-transfer-backend nixl`。預設插件是 UCX。`SGLANG_DISAGGREGATION_NIXL_BACKEND` 可改，文件舉例 `LIBFABRIC`，並說「UCX (default), LIBFABRIC, or any installed NIXL plugin」。NIXL 同時也是 HiCache 的 L3 backend（可接 3FS、GPU Direct Storage、S3 相容物件儲存）。PD 傳輸用的 NIXL 與 HiCache L3 用的 NIXL 是兩條整合，文件沒有說開了其中一條就等於另一條。 — [PD disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)；[HiCache design 的 NIXL L3](https://docs.sglang.io/docs/advanced_features/hicache_design)
- HiCache 與 PD 的兩種接法（best practices）：(1) 只在 prefill 開 HiCache，讓 prefill instance 之間經 L3 共享，文件說適合 system prompt；(2) prefill 開 HiCache，decode 開非同步卸載，讓下一輪 prefill 重用 decode 寫出的 KV，適合多輪。Decode 旗標是 `--disaggregation-decode-enable-offload-kvcache`。範例的傳輸 backend 仍是 `mooncake`，L3 範例是 `hf3fs`，write policy `write_through`，prefetch `wait_complete`，layout `page_first_direct`，io `direct`。Mooncake 的 HiCache quick start 把同一旗標解釋成「writes the decode worker's outputs back into Mooncake」。 — [best practices](https://docs.sglang.io/docs/advanced_features/hicache_best_practices)；[Mooncake HiCache quick start](https://kvcache-ai.github.io/Mooncake/deployment/integrations/sglang/hicache-quick-start.html)
- 設計頁的較短說法：PD 經 Mooncake TransferEngine；HiCache 可以在 prefill 與 decode 都開；開在 decode 時，decode 輸出也會寫回 L3。 — [HiCache design](https://docs.sglang.io/docs/advanced_features/hicache_design)
- Mooncake 純傳輸頻寬（不是 HiCache 命中）：1P1D，Qwen3-235B-A22B-Instruct-2507 BF16，SGLang `main` commit `43124cdd`、Mooncake `6e3a11b9`，皆 2026-07-15 拉下。KV backend 是 Mooncake Transfer Engine，RoCE v2，四條 400GbE（匯總線速 200 GB/s）。HiCache 與 radix 都關掉，chunked prefill 也關掉（`--disable-radix-cache`、`--chunked-prefill-size -1`）。峰值 164.3 GB/s，線速利用率 82.2%。Prompt 32768 tokens 的 KV 是 12.62 GB，傳輸 76.78 ms，佔該次 TTFT 的 2.4%。表上其他點：128 tokens／49 MB／2.07 ms／23.79 GB/s；1024／394 MB／3.96 ms／99.47 GB/s；8192／3.15 GB／20.31 ms／155.26 GB/s；16384／6.31 GB／39.14 ms／161.19 GB/s。Mooncake 另一頁寫 PD+Mooncake「~30% lower ITL」且 throughput 相當，細節指向同一份 benchmark。 — [Mooncake SGLang PD benchmark](https://kvcache-ai.github.io/Mooncake/performance/sglang/sglang-benchmark-results-v1.html)
- Dynamo 若工人共享 Mooncake 這類外部池，KV router 會依層（GPU／Host／External）打分，而不是只看 device overlap；SGLang 0.5.11 起才有 L2 host 的 KV event。這是路由器側的附加，不是 SGLang 自己的 scheduler。 — [Dynamo HiCache](https://docs.nvidia.com/dynamo/knowledge-base/modular-components/backends/sg-lang/hi-cache)

### Inferences

- Prefill 結束後，服務中的活動 KV 在 decode worker 的 GPU pool。Prefill 端是否還留一份，取決於有沒有 HiCache write-back 到 L2/L3，以及 `prefill_complete` 是否把 KV 留在 optimistic-prefill inflight queue 直到 decode 給出目的 index。沒開 HiCache 又已經傳完，prefill 端不該被假設還能當下一輪的快取。
- 多輪要跨 PD 重用，靠的是 decode 卸載進 L3（或客戶端把變長歷史重送到一個仍握有該前綴的 prefill／L3），不是 P→D 那一次 RDMA。P→D 是請求內的搬運；L3 是請求間的共用。
- `prefill_complete` 與「decode 上也開 radix／HiCache」互斥。要多輪 L3 卸載就不能同時用這份文件裡的 `prefill_complete` 限制集。

### Gaps

- 沒有為「傳輸完成後 prefill GPU 上的 KV 何時釋放」找到一張與 HiCache 無關的狀態圖。`prefill_complete` 只描述 room 狀態與 inflight queue。
- Ascend 段在抓取時被截斷，只確認有 `disaggregation-transfer-backend ascend`（memfabric，`ASCEND_MF_STORE_URL`）以及 `ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE`。細節不足以寫進操作說明。
- Mooncake 頁的「~30% lower ITL」沒有在本輪打開它指向的細節表，硬體與 SGLang 版本未核對，不要和 2026-07-15 的頻寬實驗混成同一組數字。
- 沒有 SGLang 原生、獨立於 Mooncake／NIXL／ASCEND 的第四種 PD KV 傳輸在這份 PD 文件的開頭清單裡。文件首段只點名 Mooncake 與 NIXL。

## 過載時 GPU 放不下所有運行中請求的 KV

### Takeaway

KV pool 是開機時用 `--mem-fraction-static` 劃出來的固定池，不是請求來了才向 CUDA 現要。Decode 發現下一步放不進池，就 retract：依輸出長度（或 priority）從較不值得留下的請求開始，釋放其 device KV，而且明確不插回 radix tree。合一／非 decode 模式因此是之後重算。Decode disaggregation 才會把 KV `backup_kv_cache` 到 host，之後 `load_kv_cache` 免重算；host pool 耗盡就中止請求。真正的 CUDA OOM 則要靠降低 `--chunked-prefill-size`、`--max-running-requests` 或 `--mem-fraction-static`，retract 取代不了 activation OOM。

### Cited Findings

- 記憶體帳：總使用 ≈ 權重 + KV pool + CUDA graph buffers + activations。`--mem-fraction-static = (權重 + KV pool) / GPU 容量`。其餘留給 activation 與 CUDA graph。文件經驗是留 5–8 GB 給 activation。啟動日誌形如 `max_total_num_tokens=665690, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=4096, context_len=65536, available_gpu_mem=13.50 GB`（範例時間 2025-08-11）。`available_gpu_mem` 在 5–8 GB 視為合適；10–20 GB 代表還可以把 static fraction 調高以增大 KV pool；太低則後續容易 OOM。 — [hyperparameter tuning](https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning)
- 同一頁：prefill OOM 就把 `--chunked-prefill-size` 降到 4096 或 2048，省記憶體但長提示變慢。Decode OOM 就降低 `--max-running-requests`。把 `--mem-fraction-static` 降到 0.8 或 0.7 會縮小 KV pool，prefill 與 decode 都不容易 OOM，但併發與峰值吞吐下降。`token usage > 0.9` 視為池用得滿。若經常 `< 0.9` 且 `#queue-req > 0`，代表准入太保守（常見原因是 `max_new_tokens` 很大但很早 EOS），可把 `--schedule-conservativeness` 降到約 0.3。若經常看到 `KV cache pool is full. Retract requests. #retracted_reqs: 1, #new_token_ratio: 0.9998 -> 1.0000`，把它升到約 1.3。大約每分鐘一次以內的 retract 被認為可以接受。健康的 `#queue-req` 區間寫成 100–2000。 — [hyperparameter tuning](https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning)
- 2026-10-14 `main` 的 `ScheduleBatch.check_decode_mem`：下一步 decode 需要的 token 數交給 allocator 的 `check_decode_capacity`。註解寫 allocator 擁有容量門（含它自己的 eviction 與每步保留）；retract 迴圈收斂在同一次檢查上，避免配置失敗直接當致命錯誤。 — [schedule_batch.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py)
- `retract_decode`：KV 不夠時撤回 decode 請求。順序預設用 `length_key = (len(output_ids), -len(origin_input_ids))`，排序後從尾端 pop，也就是較少已生成 token 的先被撤回；輸出長度相同時，較長的原始輸入先被撤回。`retraction_policy == "priority"` 時改把 priority 放在這個鍵前面，方向受 `schedule_low_priority_values_first` 影響。註解仍寫「TODO(lsyin): improve retraction policy for radix cache」。Beam group 不能 retract，直接 abort（HTTP 500，「Beam groups cannot be retracted」）。至少留一個請求；若連最後一個都放不下，abort 該請求而不是讓 scheduler 崩潰，訊息是「Out of memory even after retracting all other requests in the decode batch」。 — [schedule_batch.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py)
- `release_req` 的註解與程式：decode disaggregation 且 `offload_kv` 為真時，先 `backup_kv_cache(...)`，註解寫「offloaded to host so it can be restored later without recompute（見 `resume_retracted_reqs`／`load_kv_cache`）」。會重算的呼叫者（「PD true-retraction rebootstrap」）傳 `offload_kv=False`，省掉這次 device→host。接著一律 `release_kv_cache(req, tree_cache, checkpoint=False)`，旁邊的 TODO 寫 preempted 請求「we may want to insert into the tree」，現在沒有插回去。再 `evict_from_tree_cache`，額外清掉 `remaining_req_count * SGLANG_RETRACT_DECODE_STEPS` 個 token。然後 `req.reset_for_retract()`。`retract_decode` 裡的註解：「release memory and don't insert into the tree because we need the space instantly」。 — [schedule_batch.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py)
- Backup 失敗：device KV 已經放掉，請求不能恢復。若 `disaggregation_decode_retraction_backup == "host_pool"`，訊息是「Retraction host KV pool exhausted. Aborting the request.」（HTTP 500）。否則「Retracted under decode memory pressure without a KV backup. Retry later.」（HTTP 503）。`offload_kv_cache` 用 `token_to_kv_pool_allocator.get_cpu_copy` 把 token KV（以及需要時的 Mamba state）放進 `RetractionBackup`；`load_kv_cache` 再載回並清掉 backup。 — [schedule_batch.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py)
- 非 decode 模式不會進 `backup_kv_cache`（條件是 `disaggregation_mode == "decode"`）。`reset_for_retract` 把 `prefix_len`、`last_node`、`num_matched_prefix_tokens` 清成 0，設 `is_retracted = True`。若請求帶 `input_embeds`，已生成的 `output_ids` 會被丟掉，因為重 prefill 時無法把原始 embeddings 和新生 token id 混在同一個 KV 形狀裡，於是整段重跑。 — [schedule_batch.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py)
- PD 的重算路徑 `build_rebootstrap_payload`：請原本的 prefill worker「recompute this request's prefix KV under the current weights」。`max_new_tokens` 被改成 1，拿掉 stop／grammar／`min_new_tokens`，只為了重建前綴再交出一個 handoff token；真正已經送出的邊界 token 留在 decode 側覆寫。TODO 寫多模態還沒支援：payload 只有 `input_ids`，丟掉 image／audio／video，重算出的前綴 KV 不會等於原請求。 — [schedule_batch.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py)
- 較早的 PD PR #7196 已描述 decode server 在記憶體緊時把請求撤下並把 KV offload 到 CPU，之後再載回，且恢復被撤回的請求優先於新請求。這與 2026-10-14 原始碼的 decode backup 路徑一致；PR 本身的合併版本號本輪沒有再對。 — [PR #7196](https://github.com/sgl-project/sglang/pull/7196)
- Issue #26248（2026-05-25）記錄合一路徑上 retract 之後重 prefill 出現 `#cached-token: 0`，並把原因歸到 `release_kv_cache(..., is_insert=False)` 沒把前綴插回樹，再加上 `evict_from_tree_cache` 還會再趕其他前綴。該 issue 的症狀是 decode 吞吐從約 429 tok/s 掉到約 20–38 tok/s。這是使用者對當時程式的解讀與一次運行紀錄，不是官方基準；它和現在「不插入樹」的註解相符，但不能拿來代表 decode-disagg 的 host backup 路徑。 — [issue #26248](https://github.com/sgl-project/sglang/issues/26248)
- Issue #8463（2025-07-28，SGLang 0.4.8，DeepSeek-R1-0528，8×H20）報告 pool 滿並印出 retract 之後服務不再產出，GPU 利用率掉到 0，`/health` 仍是 200。這是故障報告，不是預期規格。 — [issue #8463](https://github.com/sgl-project/sglang/issues/8463)
- Issue #26265（2026-05-25）描述當時 prefill 配置失敗會變成未捕捉的 `RuntimeError` 再 `SIGQUIT`，而 decode 已有 retract。提案包含 `retry_prefill_on_kv_oom`。本輪沒有核對 2026-10-14 的 prefill 失敗是否已改成撤回准入。 — [issue #26265](https://github.com/sgl-project/sglang/issues/26265)
- v0.5.8 的 scheduler 在 retract 後把請求放回佇列（`_add_request_to_queue(req, is_retracted=True)`），並把 `new_token_ratio` 提高，使後續准入少預留未來 decode token。2026-10 的文件仍用 `new_token_ratio` 這條警告當操作指標。 — [v0.5.8 scheduler.py](https://raw.githubusercontent.com/sgl-project/sglang/v0.5.8/python/sglang/srt/managers/scheduler.py)；[hyperparameter tuning](https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning)

### Inferences

- 「Swap」與「重算」在 2026-10-14 的 main 上是分叉，不是單一政策。只有 `disaggregation_mode == "decode"` 且 backup 成功才是 host swap。Prefill 角色、合一引擎、backup 失敗、beam、以及 `input_embeds` 都更接近放棄 device KV 再重算或直接 abort。
- Retract 釋放的是運行中請求握著的 KV，不只是閒置 prefix。因為它不把這段插回樹，同一實例上其他請求也無法接上這段剛被撤下的前綴；HiCache 的 L2 不會自動變成 retract 的避難所，除非走 decode backup 或原本的 write-back 已經把前綴寫下去。
- `--schedule-conservativeness` 是在 retract 發生前少放人進來；`--max-running-requests` 是硬頂同時運行數；chunked prefill 是單次 prefill 的 activation／暫存上限。三者都不增加「已在跑的長 decode」能共享的 HBM，那個要靠更大的 KV pool、更早的 retract，或 HiCache 讓閒置前綴不要佔著 L1。

### Gaps

- `SGLANG_RETRACT_DECODE_STEPS` 的預設值沒有在讀到的片段裡。
- `--retraction-policy` 是否就是 CLI 上對應 `get_schedule().retraction_policy` 的旗標，沒有在 server arguments 頁核對。原始碼只顯示值 `"priority"` 會改排序。
- `disaggregation_decode_retraction_backup` 除了 `"host_pool"` 以外還接受哪些值，沒有讀到定義點。
- Prefill 階段池耗盡在 main 上是撤回、排隊還是仍會打死進程，沒有重讀 scheduler 事件迴圈。#26265 只代表 2026-05 的行為。

## 多租戶隔離與安全

### Takeaway

SGLang 沒有把 API key 自動做成快取隔離邊界。同一 inference instance 的 L1/L2 radix 預設共享 token 前綴；L3 在同一 backend namespace 裡還會跨 instance 共享。官方提供的隔離槓桿是呼叫者自己傳的 `cache_salt` 與 `extra_key`，以及 LoRA 的 `lora_id`（後者被拼進 `extra_key`）。不設這些的多租戶部署會讓租戶 A 的系統提示或長前綴被租戶 B 命中，並留下可量測的 TTFT 差。KVGov（2026-08）把這件事做成對 SGLang radix 的重建攻擊，不是 SGLang 已內建的防護。

### Cited Findings

- L1/L2 不跨 instance，L3 可以。因此「一台機器上兩個租戶各一個進程」仍不共享 host KV；「許多租戶打進同一個 server 進程」則共享該進程的樹。跨進程要共享必須走到 L3，而且 namespace 相同。 — [HiCache design，Tier Sharing Scope](https://docs.sglang.io/docs/advanced_features/hicache_design)
- `cache_salt` 與 `extra_key` 被 `RadixKey` 註解描述為 tree 與 storage 的 namespace。`cache_salt` 還進入 KV events；`extra_key` 不進入。In-memory 隔離在 #24568 的實驗敘述裡成立。Event hash 在該 issue 的版本不帶 salt，外部依 event 建索引的路由器可能把不同租戶的區塊看成同一個。 — [radix_cache.py 索引](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)；[issue #24568](https://github.com/sgl-project/sglang/issues/24568)
- LoRA 隔離依賴 `lora_id` 被拼進 `extra_key`。#24638 證明呼叫者控制的 `extra_key` 可以冒充這個 namespace，使 base 模型讀到 LoRA 的 KV。這是跨權重正確性問題，也是多租戶問題：知道或猜到 `lora_id` 的客戶可以構造 `extra_key`。2026-10-14 的拼接程式還在。 — [schedule_batch.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py)；[issue #24638](https://github.com/sgl-project/sglang/issues/24638)
- Session 參考不是安全邊界。它讓仍被 `session_id` 指到的前綴較晚被驅逐，而且文件沒有說 `session_id` 會阻止其他請求 match 同一條 token 路徑。共享前綴在 `/close_session` 之後仍留下其他 session 的參考。 — [session radix](https://docs.sglang.io/docs/advanced_features/session_radix_cache)
- KVGov，arXiv:2608.09225v1，2026-08-11：多租戶下 prefix cache 共享造成 timing side channel。PROMPTPEEK、EarlyBird、InputSnatch 對未保護的 vLLM 與 SGLang「reach up to 100% attack success rate」，成功率隨架構與提示結構變化。文中寫 PROMPTPEEK 在 SGLang radix-tree cache 上 ASR 99–100%，方法是探 TTFT。作者提出的防禦是每 principal 的 HMAC salt 種進 block-hash chain；對 SGLang 的對應描述是「Per-principal root node, giving disjoint trees」，而不是 SGLang 已經實作的功能。SGLang 這邊被點名的機制是「Longest matching token prefix by tree traversal」，命中訊號是 matched prefix 長度，比 vLLM 的 block 量化更細。 — [arXiv:2608.09225](https://arxiv.org/pdf/2608.09225v1)
- 權重更新會 flush radix。這能避免新權重誤用舊 KV，但不是租戶隔離。 — [native API `/flush_cache`](https://docs.sglang.io/basic_usage/native_api.html)
- PD 文件拒絕用 process-local response store 在 prefill 與 decode 之間共享對話，理由是兩個進程的歷史會不一致。那是正確性限制，不是租戶隔離功能。 — [PD disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)

### Inferences

- 若產品把多個 API key 打到同一個 SGLang 進程且不傳 `cache_salt`，共享系統提示會被故意共用，私有長前綴也會被共用。要隔離就得讓 salt 跟 principal 綁定，並且確認 L3 鍵含這個 salt（`RadixKey` 註解說 storage 也會被 namespace；#24568 顯示 2026-05 的對外 block hash 曾漏掉它）。
- 只隔離記憶體裡的樹不夠。任何消費 KV event 的路由器若仍用未加鹽的 hash，會在排程層把租戶湊在一起，即使 GPU 上的 `RadixKey` 已經分開。
- `session_id` 不能拿來當租戶 ID。它影響的是淘汰順序。

### Gaps

- 沒有找到 SGLang 官方文件說 gateway 或 API key 會自動設定 `cache_salt`。
- 沒有重跑 PROMPTPEEK，也沒有找到 SGLang 在 2026-10 合併了 KVGov 式的 per-principal root。論文是攻擊與提案，不是 release note。
- `extra_key` 不進 KV event 這個註解，與 #24568 之後是否只修了 `cache_salt`、沒修 `extra_key`，沒有用一筆 2026-10 的 event payload 驗證。

## 量化結果與 2025–2026 的行為變更

### Takeaway

2024 的 RadixAttention 只覆蓋 GPU。2025-09 的 HiCache 把 host 與外接儲存做成可插拔的 L3，並給出最高約 6× 吞吐與最高約 80% TTFT 下降，但那是作者摘要，圖本身沒有在本輪逐點抄出。2026-08 的 Unified Radix Cache 把混合注意力接上同一棵樹與 HiCache，多輪加 500 GiB Mooncake 時 DeepSeek-V4-Flash 的有效輸入吞吐從 L1 的 9.4K 到 L1+L2+L3 的 145.5K tok/s。2026-07 的 Mooncake PD 數字是純傳輸，radix 與 HiCache 都關著，不能當成快取命中收益。

### Cited Findings

- 2025-09-10，Zhiqiang Xie，LMSYS。作者測量：「up to 6× throughput improvement and up to 80% reduction in TTFT」。社群引言：Novita AI 在 Qwen3-Coder-480B 的 coding agent，對話常超過 25K tokens、約 8 輪；接 DeepSeek 3FS KVStore 後，平均 TTFT 降 56%，吞吐加倍，hit rate 從 40% 到 80%。Ant Group 在 DeepSeek-R1-671B、PD 分離、一般 QA 的線上抽樣上，接 Mooncake 後，cache hit 相對完整重算的 TTFT 平均降 84%。GPU I/O kernel「up to 3×」；page-first 加零拷貝「up to 2×」。重現指令：DeepSeek-R1、8×H20-3e、`--page-size 64`、`--chunked-prefill-size 6144`、`--mem-fraction-static 0.85`、`--enable-hierarchical-cache`、`--hicache-ratio 2`、`--hicache-io-backend kernel`、`--hicache-mem-layout page_first`、`--hicache-storage-backend hf3fs`、`--hicache-storage-prefetch-policy wait_complete`，資料 `loogle_wiki_qa.json`。另一組是 Qwen3-235B-A22B-Instruct-2507、8×H800、8 條 mlx5、Mooncake RDMA、`hicache-ratio 2`、prefetch `timeout`，`bench_multiturn.py` 用 `--num-clients 80 --num-rounds 10 --request-length 2048 --output-length 1` 來模擬 PD。圖的逐點座標只在圖片裡，本輪沒有數位化。 — [LMSYS, 2025-09-10](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/)
- Unified Radix Cache 多輪，源文 2026-08-11，視覺化頁轉述並標「Source: LMSYS Figure 4」。有效輸入吞吐的定義是「total full prompt length ÷ wall-clock time」，命中的前綴 token 也算進分子。DeepSeek-V4-Flash，FULL+SWA，4×H200 TP4，48 clients，60 rounds，每輪 4096 in + 16 out：L1 9.4K、L1+L2 14.3K、L1+L2+L3 145.5K tok/s；L3 是額外 500 GiB Mooncake；最終 hit rate 約 98%，平均 TTFT 低於 9 秒。相對 L1 的倍數在視覺化頁寫成 15.5×。Inkling-Small，FULL+SWA+MAMBA，8×H200 TP8，64 clients，30 rounds，1216 in + 64 out：15.5K／21.1K／67.1K tok/s，L3 hit rate 96.8%，平均 TTFT 1.23 秒。Hit rate 定義是該輪「cached prefix tokens ÷ complete prompt tokens」。視覺化頁註明曲線是從原圖近似讀出的。 — [SGLang unified radix blog](https://www.sglang.io/blog/unified-radix-cache)；[LMSYS, 2026-08-11](https://www.lmsys.org/blog/2026-08-11-unified-radix-cache/)
- 同一篇的 session-aware：相對「HiRadixCache + LRU」，SWE-bench 上 Unified Radix Cache 加 session-aware eviction，DeepSeek-V4-Pro TP8 的 TTFT 在 batch 128 降 11.0%（裝置 hit 42%→51%）、batch 256 降 2.9%；Qwen3.5-397B-A17B TP8 在 batch 32 降 13.5%（裝置 hit 5%→34%）、batch 64 降 16.6%（device+host hit 58%→67%）。視覺化頁把 DeepSeek 的 11.0% 寫成圖上的 −11%。 — [unified radix blog](https://www.sglang.io/blog/unified-radix-cache)
- Rust tree core。原型 PR #29074，只 L1，200 輪、每輪 100 in + 100 out、6 次、同 GPU 依序跑：SWA gpt-oss-20b TP2 全部輪次 TTFT −38%、第 176–200 輪 −42%；full attention Qwen3-32B TP2 −10%、最後 25 輪 −18%；hybrid SSM Qwen3-Next-80B-A3B TP4 −5%、最後 25 輪 −7%。其後合併的 Rust TreeCore 是 PR #32710，支援 FULL／SWA／MAMBA 與 HiCache。Session-aware 仍用 Python tree core。 — [unified radix blog](https://www.sglang.io/blog/unified-radix-cache)；[PR #29074](https://github.com/sgl-project/sglang/pull/29074)；[PR #32710](https://github.com/sgl-project/sglang/pull/32710)
- Mooncake PD 傳輸，2026-07-15，數字見上一節（峰值 164.3 GB/s、82.2%、32768 tokens 的傳輸佔 TTFT 2.4%）。實驗明示關掉 HiCache 與 radix。 — [Mooncake benchmark](https://kvcache-ai.github.io/Mooncake/performance/sglang/sglang-benchmark-results-v1.html)
- 異質 TP staging 的「2–5x」與「within ~5%」沒有附硬體、模型與日期，只出現在 PD 文件的設計敘述裡。 — [PD disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)
- 時間線（只列本輪對上的一次來源）：RadixAttention 論文 arXiv:2312.07104 被 2025-09 HiCache 文當成 GPU-only 前身。2025-09-10 HiCache 公開（L1/L2/L3、Mooncake、3FS、NIXL、file）。2025-10 PR #11506 把 FIFO／MRU／FILO 放進政策 registry；2026-10 的 eviction 文件說這三個不在 CLI。2025-11 issue #12875，`HiRadixCache.evict` 太晚。2026-02-24 issue #19212，`write_back` 在 `evict_host` 斷言失敗。2026-05 issue #24568（event hash 忽略 salt）、#24638（`extra_key` 冒充 LoRA）、#26248（retract 後重 prefill 幾乎沒有 cached token）、#26265（prefill 池耗盡打死 scheduler）。2026-06 PR #26728，load-back 期間的 host lock。2026-07-15 Mooncake 164.3 GB/s 傳輸測量。2026-08-06/07 PR #33777，write-back 優先收回重複的 host 複本。2026-08-11 Unified Radix Cache 文章與 KVGov 論文。2026-09-22 PR #40787 準備刪除已無呼叫者的 `HiRadixCache`（自 v0.5.19／#35081 起由 `UnifiedRadixCache` 接手）。2026-09-23 session 文件。2026-10 的文件頁包含 tier 私有性、`tlru`、runtime attach/detach、`tp_lcm_size`、`prefill_complete` 與其對 decode HiCache 的禁止項。2026-10-08 路線圖仍把跨 engine 的 L1/L2 共享列為未來工作。SGLang 0.5.11 起才對 L2 插入發 KV event（Dynamo 文件；Dynamo 1.3.0 帶 0.5.15）。 — 各連結見上文對應節

### Inferences

- 報告裡凡是「HiCache 讓多輪變快」的數字，都要註明層級與工作負載。9.4K→145.5K 是「長而重複的多輪 + 500 GiB L3」相對「只有 GPU」的有效輸入吞吐，不是 decode 生成吞吐，也不是 2025-09 那組 6× 的同一實驗。
- 164.3 GB/s 證明的是 P→D 搬運能吃到 RoCE 線速的八成，不是 prefix 重用。該次還把 radix 關掉，避免快取讓傳輸量變小。
- 2026 年相對 2025-09 部落格的結構變化是：混合注意力的樹從「每種組合一個 cache 類別」收成 `UnifiedRadixCache`；session 從「沒有」變成可選的軟保護；PD 出現延後配置 slot 的政策，但那個政策把 decode 側快取關掉。只複述 2025-09 的 HiRadixTree 會漏掉這些限制。

### Gaps

- 2025-09 部落格兩張基準圖（3FS 與 Mooncake）的座標沒有表格化，不能在不看圖的情況下引用中間點。
- Unified 文章的 hit-rate 曲線被作者標成「approximate samples read from the source figure」。只有文中寫出的端點（約 98%、96.8%、TTFT 上界、吞吐三欄）適合作為硬數字。
- 第三方 HiQCache（INT8 L2，宣稱 Qwen3-8B 上 host KV 1.78×）不是 SGLang 上游，本筆記不把它算進官方階層。
- 沒有核對 2026-10-14 當天的最新 release tag 與 `main` 的差距。原始碼引述是 `main`，文件引述是 docs.sglang.io 當日頁面，兩者可能差幾個未發佈的 commit。
