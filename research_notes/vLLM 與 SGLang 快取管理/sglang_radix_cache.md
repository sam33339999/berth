# SGLang 的 in-GPU Radix KV Cache（RadixAttention）

取材截止 2026-10-14。論文與 2024-01 LMSYS 部落格描述的是歷史設計；類別名、旗標與預設值以當日 GitHub `main` 與 docs.sglang.io 為準。行號會隨 `main` 移動。HiCache 只在樹的呼叫邊界帶到，不展開分層傳輸。

## RadixAttention 是什麼？radix tree 如何存 prefix、如何 match，節點裡有什麼？

### Takeaway
RadixAttention 是 runtime 自動重用 KV 的機制：請求結束後不丟 KV，而把 token 序列當 key、KV pool 的 index 當 value，放進 CPU 上的 radix tree；新請求從 root 走最長相同 prefix，只 prefill 未命中的後段。2026-10 的預設樹已是 `UnifiedRadixCache`（`python/sglang/srt/mem_cache/registry.py` 的 fallback），經典 `RadixCache` / `TreeNode` 仍在，並負責模擬用的 waiting-queue tree。

### Cited Findings
- 論文定義：不同於做完一次 generation 就丟掉 KV，系統把 prompt 與 generation 結果的 KV 留在 radix tree，做 prefix search、reuse、insertion、eviction；並實作 LRU 與 cache-aware scheduling。樹在 CPU，GPU 上的 KV 是 non-contiguous paged layout，而且「each page is equivalent to one token」。與 continuous batching、paged attention、tensor parallelism 相容；沒有 hit 時 overhead 可忽略。— [Zheng et al., arXiv:2312.07104v2](https://arxiv.org/html/2312.07104v2)
- Radix tree 的邊可以標一整段 token，而不只單一元素，所以比逐 token 的 trie 省節點。前端把完整 prompt 送給 runtime，由 runtime 做 prefix matching。— [Zheng et al., arXiv:2312.07104v2](https://arxiv.org/html/2312.07104v2)；同圖解見 [LMSYS 2024-01-17](https://www.lmsys.org/blog/2024-01-17-sglang/)
- `RadixKey`（`radix_cache.py`）的 slot 是 `token_ids`、`extra_key`、`cache_salt`、`is_bigram`、`limit`。`extra_key` 註解寫「Namespaces the tree and storage; KV events see only its LoRA name。」`cache_salt` 註解寫「Namespaces the tree, storage and KV events。」`page_aligned()` 在 `page_size > 1` 時把長度切到 page 的整數倍。`match()` / `match_at()` 的共同 prefix 長度也會向下對齊 `page_size`。子節點的 dict key 由 `child_key_at()` 產生：`cache_salt is not None` 時是 `((extra_key, cache_salt), plain)`，否則 `extra_key is None` 時只有 token plain，不然是 `(extra_key, plain)`。`plain` 在 `page_size == 1` 時是一個 token id（bigram 模式則是相鄰 token pair），否則是一整頁 token。— [radix_cache.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)
- `TreeNode.__init__` 欄位：`children`（`defaultdict(TreeNode)`）、`parent`、`key: RadixKey`、`value: Optional[torch.Tensor]`（KV index，不是 KV tensor 本體）、`lock_ref = 0`、`last_access_time = time.monotonic()`、`creation_time`、`hit_count = 0`、`hash_value`（每頁 hash）、`event_hash_value`（給外部 KV event 的 namespace-aware hash）、`priority`、`id`。`evicted` 為 `value is None`。排序用 `__lt__` 比 `last_access_time`。— [radix_cache.py L250–L278](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)
- `reset()` 建 root：`priority=-sys.maxsize`、空 `RadixKey`、`value=[]`、`lock_ref = 1`。root 因此不會被當成可驅逐葉。— 同檔 `reset`
- `match_prefix` 的 docstring：最長 cache prefix 由 token id 序列加上可選 `extra_key` 決定；相同前導 token 但 `extra_key` 不同的項目「are intentionally kept disjoint and never share prefix nodes」，用途含隔離不同 LoRA / adapter，以及「different sampling salt, cache version, or retrieval augmentation context」。`page_size > 1` 時先把 key 長度截成 page 倍數；空 key 回傳 root。命中若停在某個節點儲存區段的中間，「the node is split once to expose a precise boundary」。同時更新 eviction 用的 access timestamp。回傳 `MatchResult.device_prefix_len` 與 `last_device_node`。`disable` 或空 key 直接回傳快取的空結果。— 同檔 `match_prefix`
- `insert` 同樣先 page-align；`disable` 時 `InsertResult(prefix_len=0)`。`_insert_cache` 用 `req.extra_key` 與 `req.cache_salt` 組 `RadixKey`，把 request row 上的 KV index 插入樹，再 `free_segment` 掉與樹重複的 index（註解：「Radix Cache takes one ref in memory pool」）。`checkpoint()` 對未完成請求做同樣插入，並可 `split_prompt=True`：在 prompt 邊界再插一次，讓「eviction can drop the output KV without the prompt」。`page_size > 1` 時，不滿一頁的尾巴留在 request row、不進樹；`cache_protected_len` 標出樹擁有的部分。— 同檔 `insert` / `_insert_cache` / `checkpoint`
- 預設實作不再是單一 `RadixCache`。`registry.py` 模組說明：`--radix-cache-backend` 未設定時走 built-in selection。該鏈在 LMCache、external linker、pure SWA、FlexKV 等特例之後，`return create_unified_radix_cache(ctx)`。`create_unified_radix_cache` 的 docstring 是「Initialize a UnifiedRadixCache with proper components and optional HiCache。」元件從 `ComponentType.FULL` 起，hybrid SWA / SSM 再加對應元件。— [registry.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/registry.py)
- 2026-09-20 合併的 PR #40313 寫：`SWARadixCache` 與 `MambaRadixCache` 已不可達，hybrid 走 `UnifiedRadixCache`；「`RadixCache` stays, it is still selected for non-hybrid models」。兩天後的 PR #40780（2026-09-22）寫：unified radix tree 已是所有情況的預設（#35081），並移除 `SGLANG_ENABLE_UNIFIED_RADIX_TREE`。當日 `main` 的 `registry.py` fallback 是 `create_unified_radix_cache`，與 #40780 一致，與 #40313「non-hybrid 仍選 RadixCache」的說法衝突；應以 `registry.py` 為準。`RadixCache` 類別仍在，`SchedulePolicy` 用 `RadixCache.create_simulated()` 做 in-batch 模擬樹，`PureSWARadixCache` 仍被 pure-SWA 路徑引用。— [PR #40313](https://github.com/sgl-project/sglang/pull/40313)；[PR #40780](https://github.com/sgl-project/sglang/pull/40780)；[registry.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/registry.py)；[schedule_policy.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_policy.py)
- `UnifiedRadixCache` 官方說明：一個樹加上可插拔元件。每個節點存一段 token span，以及每個元件一個 slot。FULL 與 SWA slot 放 KV page index；MAMBA 在 prefix 終點放 checkpoint。視窗外的 SWA 可單獨驅逐，留下空 slot（tombstone）。Python 仍擁有 `req_to_token_pool` 與 KV allocator；樹的 match / insert / lock / evict 可選 Rust core（見後）。— [SGLang blog, Unified Radix Cache, 2026-10-12](https://www.sglang.io/blog/unified-radix-cache)
- HiCache 開啟時，同一棵樹的 match 會先對本機層（GPU L1、host L2）走 prefix；`page_size > 1` 時以 page 為粒度，中途結束會 split。這是 `enable_hierarchical_cache` 的延伸，不是 in-GPU radix 的預設路徑（該旗標預設 `False`）。`HiRadixCache` 類別已在 2026-09-22 的 PR #40787 刪除，理由是 HiCache 自 #35081（v0.5.19）改走 `UnifiedRadixCache`。— [HiCache design doc](https://docs.sglang.io/advanced_features/hicache_design.html)；[PR #40787](https://github.com/sgl-project/sglang/pull/40787)

### Inferences
- 節點的 `value` 是 pool index，KV tensor 在 `token_to_kv_pool_allocator` 指向的 GPU pool。樹本身是 CPU 上的拓撲與引用計數，這點從 2024 論文到 2026 程式沒有改。
- Match 是從 root 依 `child_key(page_size)` 往下走；第一個對不齊的 token（或 page）就是命中終點。部分重疊會 split，讓後續請求能接在精確邊界上，而不是把整段邊複製一份。
- `lock_ref` 是「正在使用這條 prefix 的請求數」，不是 LRU 分數。root 固定為 1，所以驅逐只會從葉往上吃，不會拆掉仍有後代或仍被 in-flight 請求握住的節點。

### Gaps
- `_match_prefix_helper` / `_split_node` / `_insert_helper` 的逐行迴圈未完整抄錄；split 與 timestamp 更新以 `match_prefix` docstring 與 `child_key` 的 key 格式為準。
- `UnifiedRadixCache.match_prefix` 在 `params.disable == True` 時是否與 `RadixCache` 一樣直接回空，未逐行核對。
- Rust tree core 與 Python `RadixCache` 的節點欄位是否一一對應，只看到 PR 摘要（lock、LRU list、deferred free），沒有對到穩定 release tag。

## Cache-aware scheduling 和 cache-blind continuous batching 差在哪？優化什麼？類別與政策？

### Takeaway
執行緒仍是單一 `Scheduler` 的 continuous batching：running batch 裡的 decode 與 waiting queue 裡的 prefill 共用一個 KV pool。差別在 waiting queue 的排序。`SchedulePolicy` 把政策分成 `CacheAwarePolicy`（`lpm`、`dfs-weight`、`hrrn`、`shortest-prefill-first`）與 `CacheAgnosticPolicy`（`fcfs`、`lof`、`random`、`routing-key`）。論文要的是 cache hit rate，演算法是 longest-shared-prefix-first；2026-10 文件上的預設卻是 cache-blind 的 `fcfs`。即使 `fcfs`，prefix 仍可被 match 後跳過 prefill，只是不為了 hit rate 重排。

### Cited Findings
- 論文把 cache hit rate 定義為 cached prompt tokens / prompt tokens。Waiting queue 若在無關請求之間頻繁切換會 thrashing。Batch 設定下「sort the requests by matched prefix length and prioritize requests with longer matched prefixes instead of using a first-come, first-served schedule」。演算法名為 longest-shared-prefix-first，附於 continuous / contiguous batching。Theorem 3.1：離線、cache 容量 ≥ 最長請求時，以 DFS 走請求的 radix tree 可得到最佳 hit rate，且 longest-shared-prefix-first 等價於該 DFS。實務上輸出長度不可預測，證明假設不完全成立。— [arXiv:2312.07104v2，Cache-aware scheduling 與 Theorem 3.1](https://arxiv.org/html/2312.07104v2)
- 論文同時寫明：正在 running batch 使用的節點不能驅逐；每個節點有 reference counter，為 0 才可驅逐。Cache 與 running request 共用同一個 memory pool，不為 cache 另劃固定池；waiting 請求夠多時，系統會為了更大的 batch 把已 cache 的 token 全部驅逐。— 同論文 RadixAttention 小節
- 當日 `main` 的類別就在 `schedule_policy.py`：`class SchedulePolicy`，內含 `CacheAwarePolicy` 與 `CacheAgnosticPolicy`。`Scheduler` 以 `SchedulePolicy(self.schedule_policy, self.tree_cache, self.enable_hierarchical_cache, ...)` 建立它。沒有名為 ContinuousBatchingScheduler 的獨立類別。— [schedule_policy.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_policy.py)；[scheduler.py 約 L1394](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py)
- `calc_priority`：cache-aware 政策先 `_compute_prefix_matches`，再依政策排序。`lpm` 呼叫 `_sort_by_longest_prefix`，key 是 `-r.num_matched_prefix_tokens`（暫時降優先的 rid 放 `+inf`）。`dfs-weight` 呼叫 `tree_cache.dfs_weight_order(last_node)`，註解為 depth-first search weighting。`hrrn` 的註解把 classic HRRN 寫成 `ratio = 1 + wait / est_prefill_time`，實作用已處理的 prefill token 當時間：`ratio_delta = waited_tokens / uncached`，`uncached = len(origin_input_ids) - num_matched_prefix_tokens`（host hit 也算 cached）。未 cache 長度 ≤ 0 的請求立刻排走。`shortest-prefill-first` 依未 cache 的 prefill 工作量由短到長，log 字樣是「Experimental」。— 同檔 `calc_priority` 至 `_sort_by_hrrn`
- In-batch prefix caching（`_compute_prefix_matches`）：若請求在既有樹上的 prefix 短於門檻，改對 `waiting_queue_radix_tree`（`RadixCache.create_simulated()`）做 match。若模擬樹上已有足夠長的相同 prefix，該請求被放進 `temporary_deprioritized`，讓「only one of them」先跑，把共享 prefix 寫進真的樹。註解舉例：引擎跑久之後，像 `"the"` 這種過短 prefix 不該觸發這個邏輯。key 帶 `extra_key` 與 `cache_salt`。— 同檔
- Cache-agnostic 路徑：註解寫「Cache-agnostic policies sort without matching」，但若 `tree_cache.supports_fast_match_prefix()` 且不是 disagg decode，仍會對每個 waiting request `match_kv_cache`，讓 load snapshot 看得到 `num_matched_prefix_tokens`。`fcfs` 本身不重排；若 `--enable-priority-scheduling`，改走 `_sort_by_priority_and_fcfs`。`lof` 依 `max_new_tokens` 由長到短，可再被 priority 當第一鍵。`random` 與 `routing-key`（依 running batch 裡 routing key 的頻率）不看樹。— 同檔 `calc_priority`
- 佇列過長時，`lpm` 與 `hrrn` 若 `len(waiting_queue) > WAITING_QUEUE_PREFIX_MATCH_MAX`，`_determine_active_policy` 改回 `fcfs`，註解：「Turn off the expensive prefix matching and sorting」。樹被 disable 時，cache-aware 政策除了 `shortest-prefill-first`（直接報錯，因為它需要 prefix caching）以外，被改成 `fcfs`。— 同檔 `_determine_active_policy`、`_validate_and_adjust_policy`
- 文件（2026-10-14）`--schedule-policy` 預設 `fcfs`，列出的選項是 `lpm`、`random`、`fcfs`、`dfs-weight`、`lof`、`priority`、`routing-key`。— [Server Arguments](https://docs.sglang.io/docs/advanced_features/server_arguments)
- 歷史預設不同：v0.3.6（2024-11-22）的 `ServerArgs.schedule_policy` 預設是 `"lpm"`，choices 只有 `lpm`、`random`、`fcfs`、`dfs-weight`。2025-07 的一份 `server_args.py` 快照已是 `schedule_policy: str = "fcfs"`、`page_size: int = 1`。— [v0.3.6 server_args.py](https://github.com/sgl-project/sglang/blob/v0.3.6/python/sglang/srt/server_args.py)；[2025-07 server_args 鏡像](https://developer.sourcefind.cn/codes/zhaoyu6/sglang/-/blob/b5dd5e87416acc05576e286b826b62e97264a0fb/python/sglang/srt/server_args.py)

### Inferences
- Cache-blind 的 `fcfs` 仍是 continuous batching，而且通常仍會吃 radix hit 來縮短 prefill；它沒做的是「為了提高命中率而打亂到達順序」。Cache-aware 的 `lpm` 才對應論文的 longest-prefix-first。`dfs-weight` 更接近 Theorem 3.1 的 DFS 順序，而不是只比單一請求的命中長度。
- `hrrn` 與 `shortest-prefill-first` 的目標不是 hit rate。前者用「等了多久 / 還要 prefill 多少」逼近回應比，cache 只影響分母；後者讓短的未命中 prefill 先走。多使用者同時到達時，`fcfs` 保等待公平，`lpm` 會讓共享長 system prompt 的請求插隊。
- 論文「cache 與 running request 共用 pool，必要時把 cache 全部讓給 batch」仍是排程壓力的方向，但 2026 的 pool 是事先用 `mem_fraction_static` 算出大小的 KV pool，不是論文所說的不預先配置固定池。

### Gaps
- `WAITING_QUEUE_PREFIX_MATCH_MAX`、`IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD`、`IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD` 的數值這次沒讀到。
- 文件把 `priority` 列為 `--schedule-policy` 的選項，但 `CacheAwarePolicy` / `CacheAgnosticPolicy` 沒有 `priority` 這個值；priority 是另一個旗標 `--enable-priority-scheduling`。反之，程式裡的 `hrrn` 與 `shortest-prefill-first` 沒有出現在該文件的 choices。哪一份是 CLI parser 的最終清單，未再對 `argparse` 註冊處。
- `fcfs` 在 `supports_fast_match_prefix()` 為假時，prefix match 改由 `Scheduler` 哪一個函式補做，未定位。

## 驅逐：LRU 還是別的？誰被保護？粒度是 page 還是 token？

### Takeaway
預設仍是 LRU，但是 2026 的可選政策已多於論文的單一 LRU。只驅逐「有 KV、`lock_ref` 為 0、沒有仍握有 KV 的子節點」的葉；root 不驅逐。Session 參考與政策分數都不是 pin。粒度是節點上的一整段 page-aligned index：`page_size == 1` 時一段可以是多個 token，`page_size > 1` 時 match、insert、未滿頁的尾巴都以 page 為單位。

### Cited Findings
- 論文：LRU，先驅逐 least recently used 的葉，祖先要等自己變成葉才會被驅逐。Running batch 使用中的節點（reference counter > 0）不驅逐。— [arXiv:2312.07104v2](https://arxiv.org/html/2312.07104v2)
- 官方文件：pool 滿了才依政策回收 prefix。`--radix-eviction-policy` 預設 `lru`。只考慮 evictable leaves：「nodes whose KV is present, unlocked (no in-flight request holds them), and not shadowed by a child that still holds KV. The root is never evictable.」最低分先走；葉被刪後，父節點若變成葉就重新進入候選，所以是從枝端走向 root。政策只打分，不決定要釋放多少，也不能 pin：「a node protected by the policy is still evicted if reclaiming everything else is not enough.」平手時回到 least-recently-used。— [Radix Cache Eviction Policies](https://docs.sglang.io/docs/advanced_features/radix_eviction_policy)
- 同一頁列出的政策：`lru`（最久未用）、`lfu`（hit 最少，再比 recency）、`slru`（probationary 先於 protected；`protected_threshold` 預設 2）、`priority`（插入該 prefix 的請求 priority 最低者先走；多請求到達同一節點時保留最高 priority；沒開 priority scheduling 時大家都是 0，等價 LRU）、`tlru`（Tail-Optimized LRU，arXiv:2510.15152；先丟對話裡下一輪 prefill 用不到的尾端，再回到 recency；需要 unified tree）。`fifo`、`mru`、`filo` 在 policy registry 裡，但不在命令列，只給 out-of-tree 實驗。Hit count 計算「被之後的請求 match 到的次數」，chunked-prefill 的中間步、已驅逐節點、write-back HiCache 不增加。— 同頁
- Server arguments 表（2026-10-14）的 `--radix-eviction-policy` choices 只寫 `lru`、`lfu`、`slru`、`priority`，預設 `lru`。這與上一頁把 `tlru` 當成可用 CLI 值衝突。— [Server Arguments](https://docs.sglang.io/docs/advanced_features/server_arguments)；對照 [eviction policy 頁](https://docs.sglang.io/docs/advanced_features/radix_eviction_policy)
- `RadixCache.evict`：`disable` 則空結果。對 `evictable_leaves` 以 `eviction_strategy.get_priority(node)` 做 heap，pop 最低者，`free_segment(x.value)`，`num_evicted += len(x.value)`，`_delete_leaf`。若父節點已無 children 且 `parent.lock_ref == 0`，把父節點推進 heap。註解：「Tree values are page-aligned copies of a kv row.」— [radix_cache.py `evict`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)
- `inc_lock_ref` 的契約（unified 元件說明）：請求開始使用某段 cache prefix 時鎖住該節點與其祖先，避免依賴中的節點被驅逐。解鎖必須重放同一張 receipt。— [unified_cache components 說明，GitHub tree 頁](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/mem_cache/unified_cache/components)
- Session 參考是軟保護，不是 pin。文件原句：「Session references are soft protection, not memory pins. Referenced KV can still be evicted when reclaiming unreferenced KV is insufficient.」FULL 的順序是：無 session 參考的節點、參考數較少的節點、然後才是設定的政策（例如 LRU）。SWA 與 Mamba 是兩輪 LRU：先無參考，不夠再動有參考的。`/close_session` 只拿掉參考，KV 回到一般驅逐順序，不立刻釋放。此功能「implemented only by `UnifiedRadixCache`」，要 `--enable-session-radix-cache`。— [Session-Aware Radix Cache](https://docs.sglang.io/docs/advanced_features/session_radix_cache)
- `--page-size` 文件定義為「The number of tokens in a page」，預設 `1`。HiCache 文件補充：較大 page 降低 metadata 與儲存 I/O 成本，但只匹配到 page 的一部分時 hit rate 會下降。— [Server Arguments](https://docs.sglang.io/docs/advanced_features/server_arguments)；[HiCache design](https://docs.sglang.io/advanced_features/hicache_design.html)

### Inferences
- 「保護」有三層，強度不同：(1) `lock_ref > 0` 與非葉、root，根本不進受害者集合；(2) session reference 與 `slru` / `priority` / `tlru` 只改變分數，pool 真的不夠仍會被清掉；(3) `checkpoint` 的 prompt/output split 讓輸出葉可以先死、prompt 祖先留下來，這是拓撲而不是 pin。
- 驅逐單位是樹節點的 `value` 整段，不是每次一個 token。`page_size == 1` 時這段仍可含多個 token（radix 邊是變長序列）。`page_size > 1` 時，不滿一頁的 token 根本不在樹上，所以也不會被這次 leaf eviction 代表。

### Gaps
- `get_eviction_strategy` 各政策的分數公式（`lru` 用 `last_access_time` 以外是否還加 `creation_time`）未讀 `evict_policy.py`。
- Unified tree 上 FULL / SWA / Mamba 的 cascade（拆內部 FULL 節點會連帶清 SWA 與 Mamba）來自 session 文件，未對到 `evict` 原始迴圈。
- `tlru` 是否已通過當日 CLI parser，兩份官方文件不一致，未跑 `--help`。

## 不同使用者的請求如何交錯？預設是否跨使用者共享 prefix？有沒有隔離？

### Takeaway
同一個 engine scheduler 上，所有使用者的請求進同一條 waiting queue、同一個 running batch、同一棵 radix tree、同一個 GPU KV pool。Token prefix 相同且 `extra_key` / `cache_salt` 都相同（預設都是空）時，跨使用者共享 KV 是預設行為。`session_id` 不隔離內容，只在打開 session radix 時影響驅逐順序。要隔離必須顯式給不同的 `extra_key` 或 `cache_salt`（LoRA、cache version、salt）。

### Cited Findings
- 論文的範例就是兩段 chat、一批 few-shot、以及 self-consistency 抽樣共用一棵樹：第二個 chat 會把第一個 chat 的 system prompt 節點 split 出來共享。前端每次送的是完整 prompt，不是增量 delta。— [LMSYS 2024-01-17，Figure 4 步驟 (4)](https://www.lmsys.org/blog/2024-01-17-sglang/)；[論文](https://arxiv.org/html/2312.07104v2)
- `match_prefix` docstring 寫明：不同 `extra_key` 即使 token 相同也不共享節點。列出的用途是 LoRA / adapter，以及呼叫端故意不共享的 sampling salt、cache version、retrieval context。`_check_compatible` 在 `extra_key` 或 `cache_salt` 不同時直接 `ValueError`。子節點 key 把兩者都編進去。— [radix_cache.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)
- `_compute_prefix_matches` 與 `_insert_cache` 都把 `req.extra_key` 和 `req.cache_salt` 放進 `RadixKey`。兩者都空時，child key 只剩 token id，因此不同請求只要 token 相同就撞進同一節點。— [schedule_policy.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_policy.py)；[radix_cache.py `_insert_cache`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)
- `session_id` 的文件：「The ID labels cache references only; it does not append or reconstruct conversation context, so each request must still contain the intended prompt.」同一 session 要每次自己帶完整 prompt。請求結束才把可重用的 leaf 登記到該 `session_id`。這不是 namespace。— [Session-Aware Radix Cache](https://docs.sglang.io/docs/advanced_features/session_radix_cache)
- `Scheduler` 只有在 `enable_session_radix_cache` 時才 `ensure_session_generation` / `open_radix_session` / `release_radix_session`。註解：「Radix-native sessions use only the top-level session_id.」— [scheduler.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py)
- 交錯發生在 `SchedulePolicy.calc_priority` 對單一 `waiting_queue` 排序，以及 `Scheduler` 的 running batch。`max_running_requests` 預設 `None`，文件語意是 running request 數量上限，不是 per-user 上限。— [Server Arguments](https://docs.sglang.io/docs/advanced_features/server_arguments)
- PR #24210 提案環境變數 `SGLANG_RADIX_DISABLE_REUSE`：在 `Req.__init__` 把 per-request UUID 附加到 `extra_key`，藉由既有 namespace 禁止跨請求命中，但同一請求內的 chunked prefill 仍用同一把 key。該 PR 正文也寫 `--disable-radix-cache` 會把實作從 `RadixCache` / `SWARadixCache` 換成 `ChunkCache` / `SWAChunkCache`。這段描述的是 PR 動機當時的程式，不是 2026-10-14 `registry.py` 的行為。未在這次擷取中確認該 PR 已在 `main`。— [PR #24210](https://github.com/sgl-project/sglang/pull/24210)
- 多 replica 時，router 側另有 approximate radix tree（以字元而非 token id 記錄，避免 tokenization），用來把請求送到 prefix 較長的 worker；那棵樹會自己做 LRU，不保證與 GPU 上的真樹一致。這是跨 engine 的路由，不是單機 KV 隔離。— [sgl-model-gateway cache_aware.rs 檔頭](https://raw.githubusercontent.com/sgl-project/sglang/main/sgl-model-gateway/src/policies/cache_aware.rs)

### Inferences
- 預設沒有 per-user、per-tenant 的 cache 隔離。共享 system prompt 或相同 few-shot 的使用者會命中彼此留下的 KV。這是命中率的來源，也表示隔離必須由呼叫端放進 `extra_key` 或 `cache_salt`，而不是靠 `session_id`。
- 不同使用者的 decode 步驟在同一個 running batch 裡交錯；誰的 prefix 留在 GPU 上，取決於驅逐政策與 `lock_ref`，不取決於 user id。`lpm` 會進一步讓「跟目前樹上長 prefix 相同」的使用者先 prefill。

### Gaps
- 公開 HTTP / OpenAI 相容 API 裡 `cache_salt` 與 `extra_key` 的 JSON 欄位名、預設值，沒有從 protocol 檔確認；只確認到 `Req.cache_salt` 與 `Req.extra_key` 會進樹。
- 沒有找到官方的多租戶隱私模型或「預設不跨使用者共享」的開關說明。`SGLANG_RADIX_DISABLE_REUSE` 是否已合入未核對。
- Data-parallel 多 scheduler replica 之間，單機樹是否完全不相通（除了 router 的近似樹與 HiCache L2/L3），未讀 DP controller。

## 多輪、few-shot、平行抽樣如何重用節點？和 DSL 的 fork/join 是什麼關係？

### Takeaway
重用條件始終是「已在樹上的 token prefix」。多輪對話要客戶端重送含歷史的完整 prompt，runtime 才 match 到上一輪的 prompt+輸出。Few-shot 是共享例子那段邊，在第一個相異 token split。平行抽樣 / self-consistency 是同一條 prompt 邊上長出多個生成後代。2024 DSL 的 `fork` 會先把共享 prefix 當 hint 送進樹，再送各分支；今日的 HTTP serving 不再靠這個 hint，只要完整 token 序列相同就會撞上同一節點。

### Cited Findings
- DSL：`fork`「creates parallel forks of the prompt state」，`join` 再併回。程式可以鏈多次 generation，並用 `fork` 做平行副本；不同程式實例也常共享 system prompt。論文 Figure 3 / 部落格 Figure 4 的九個步驟覆蓋：多輪 chat 把新 turn 接在舊節點後、第二個 session 為了共享 system prompt 而 split、few-shot batch 在共同例子處 split、self-consistency 對同一問題再抽樣並為了騰空間驅逐別的葉。— [arXiv:2312.07104v2](https://arxiv.org/html/2312.07104v2)；[LMSYS 2024-01-17](https://www.lmsys.org/blog/2024-01-17-sglang/)
- 論文特別寫 frontend hint：「During the execution of the fork primitive, the frontend sends the prefix first as a hint, ensuring the prefix is correctly inserted into the tree. It then sends the remaining prompts.」這是 2024 前端與 runtime 的共同設計，用來簡化排程。— [arXiv:2312.07104v2](https://arxiv.org/html/2312.07104v2)
- 今日 session 文件要求每個 turn 自己帶 `FULL_PROMPT_FOR_THIS_TURN`；`session_id` 不重建歷史。因此多輪命中依賴「這次 prompt 的前綴仍在樹上等價於上一輪插入的 prompt+輸出」，而不是 session 物件裡存了一份對話。— [Session-Aware Radix Cache](https://docs.sglang.io/docs/advanced_features/session_radix_cache)
- 生成過程中的插入走 `RadixCache.checkpoint`：把 `full_untruncated_fill_ids` 的 page-aligned 前段寫入樹，必要時在 prompt 邊界 split，然後 `unlock` 舊節點、`lock` 新的 `last_node`。未完成請求的 KV 因此能被下一個共享該 prefix 的請求（另一輪、另一個 fork、另一個使用者）match 到，而不必等請求完全結束。— [radix_cache.py `checkpoint`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)
- In-batch 模擬樹讓「同時在 waiting queue、彼此共享但 GPU 樹上還沒有」的請求只放行一條去把 prefix 寫入，其餘暫時降優先。這就是 few-shot 或相同 system prompt 同時湧入時的排程對應物，不再需要 DSL hint。— [schedule_policy.py `_compute_prefix_matches`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_policy.py)
- 部落格給的歷史吞吐：Llama-7B 與 Mixtral-8x7B 上，相對 Guidance、vLLM v0.2.5、TGI v1.3.0，「up to 5 times higher throughput」，工作負載含 few-shot（MMLU、HellaSwag）、多輪 chat、ReAct、Tree-of-Thought。論文 v2 摘要（2024-06-06）寫的是 up to 6.4×，對象與數字都比部落格廣，不應混成同一個結果。— [LMSYS 2024-01-17](https://www.lmsys.org/blog/2024-01-17-sglang/)；[arXiv abstract](https://arxiv.org/html/2312.07104v2)
- 2026-10-12 部落格的多輪數字屬於 unified tree + 分層儲存，不是 2024 論文的 in-GPU 實驗：有 Mooncake L3 時多輪 hit rate 接近 98%；session-aware eviction 相對 HiRadixCache + LRU，SWE-bench TTFT 在 DeepSeek-V4-Pro 降到多 11.0%、在 Qwen3.5-397B-A17B 降到多 16.6%。— [Unified Radix Cache blog](https://www.sglang.io/blog/unified-radix-cache)

### Inferences
- `fork` / `join` 是 2024 前端語言的控制流。現在的 serving 路徑用「完整 prompt + radix match」達到同一種共享：平行抽樣是多個請求帶相同 prompt prefix，生成 token 不同後自然分岔。沒有證據顯示 HTTP server 仍會發送論文裡的 frontend hint。
- 多輪要命中，歷史必須原樣出現在新 prompt 的前段，而且中間沒被 LRU 掉。`session_id` 只讓這段歷史比別人更晚被驅逐，不保證還在。

### Gaps
- `sampling_params` 的 `n`（一次請求要多個 completion）是在 scheduler 裡拆成多個 `Req` 共享 `last_node`，還是只在取樣層分叉，未讀到對應程式。
- DSL interpreter 是否還在現行 sglang 套件裡發送 fork hint，未查 frontend 套件。

## 重要旗標：mem_fraction_static、max_running_requests、page_size、disable_radix_cache、schedule_policy，以及 2024–2026 的替代品

### Takeaway
2026-10-14 文件：`--disable-radix-cache` 預設關（也就是 radix 預設開），`--schedule-policy` 預設 `fcfs`，`--page-size` 預設 1，`--radix-eviction-policy` 預設 `lru`，`--mem-fraction-static` 與 `--max-running-requests` 預設都不設上限式常數。和 2024 年底相比，排程預設從 `lpm` 改成 `fcfs`，驅逐從只有 LRU 變成一組政策，樹的本體從 `RadixCache` 換成預設 `UnifiedRadixCache`。`HiRadixCache` 與 `SGLANG_ENABLE_UNIFIED_RADIX_TREE` 已退出。

### Cited Findings
- `--mem-fraction-static`：靜態配置佔 GPU 記憶體的比例，對象是 model weights 與 KV cache pool。看到 OOM 就調小。未設定時以 `(GPU memory - reserved memory) / GPU memory` 計算；偵測不到 GPU 記憶體則 0.88。預設欄位值是 `None`。— [Server Arguments，2026-10-14](https://docs.sglang.io/docs/advanced_features/server_arguments)
- `--max-running-requests`：同時 running 的請求上限，預設 `None`。同表還有 `--max-queued-requests`（預設 `None`；disaggregation 下忽略）與 `--max-prefill-tokens`（預設 16384，實際上限是它與模型 context 的較大者）。— 同頁
- `--page-size`：一頁的 token 數，預設 1。— 同頁
- `--schedule-policy`：預設 `fcfs`。文件列出的選項見上一節。另有 `--enable-priority-scheduling`（預設 False）：整數 priority 較大的請求預設先排。`--schedule-conservativeness` 在 2024-11 與 2025-07 的 `server_args` 預設 1.0，語意是越大越保守；這次 server arguments 摘錄沒有再看到該列，不把它當成 2026-10 文件仍公開的旗標。— [Server Arguments](https://docs.sglang.io/docs/advanced_features/server_arguments)；[v0.3.6](https://github.com/sgl-project/sglang/blob/v0.3.6/python/sglang/srt/server_args.py)
- `--disable-radix-cache`：文件寫「Disable RadixAttention for prefix caching.」，預設 False。— [Server Arguments](https://docs.sglang.io/docs/advanced_features/server_arguments)
- `--radix-eviction-policy` 預設 `lru`。參數走 `--radix-eviction-policy-config` JSON。`lru` / `lfu` / `priority` 不吃參數。— [eviction policy](https://docs.sglang.io/docs/advanced_features/radix_eviction_policy)；[Server Arguments](https://docs.sglang.io/docs/advanced_features/server_arguments)
- `--enable-session-radix-cache`：只對 `UnifiedRadixCache` 有效。預設未在該頁寫成 True。— [Session-Aware Radix Cache](https://docs.sglang.io/docs/advanced_features/session_radix_cache)
- `--enable-hierarchical-cache` 預設 False。相關但屬 HiCache：`--hicache-ratio` 預設 2.0、`--hicache-write-policy` 預設 `write_through`（另有 `write_back`、`write_through_selective`）、`--hicache-storage-backend` 預設 None。`--disaggregation-decode-enable-radix-cache` 預設 False，只在 PD decode server 上快取 prefix。— [Server Arguments](https://docs.sglang.io/docs/advanced_features/server_arguments)
- `--radix-cache-backend`：未設定就走 built-in 鏈；設定則必須是已 `register_radix_cache_backend` 的名字。Rust unified tree 在 PR #29074 是 opt-in：`--radix-cache-backend rust_unified_tree`，且該 PR 自己寫「the default cache path is unchanged」，當時只做 L1。— [registry.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/registry.py)；[PR #29074](https://github.com/sgl-project/sglang/pull/29074)
- 關掉 radix 時的類別：`chunk_cache.py` 的 docstring 仍寫「ChunkCache is used when radix cache is disabled」，包含一般 chunked prefill 與未開 decode radix 的 PD decode 側，且 `disable` property 恆為 True、`match_prefix` 不做 prefix matching。但當日 `registry.py` 的 `disable_radix_cache` 分支，除了 pure SWA（且不是 host-pool retraction）回 `PureSWARadixCache` 以外，是 `return create_unified_radix_cache(ctx)`，註解說 disabled cache 不發布 storage、host-pool retraction 要靠 UnifiedRadixCache。兩份程式並存，工廠才是當日選擇器。— [chunk_cache.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/chunk_cache.py)；[registry.py L85–L146](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/registry.py)
- 已退役、不應再當成現行開關：`SGLANG_ENABLE_UNIFIED_RADIX_TREE`（#40780，2026-09-22 刪除，unified 已是預設）；`HiRadixCache`（#40787，同日刪除）；`SWARadixCache` 與 `MambaRadixCache`（#40313，2026-09-20 刪除）。v0.3.6 的 `--schedule-policy` 預設 `lpm` 也已被文件上的 `fcfs` 取代。— 上述 PR 與 [v0.3.6 server_args.py](https://github.com/sgl-project/sglang/blob/v0.3.6/python/sglang/srt/server_args.py)

### Inferences
- `mem_fraction_static` 決定整池有多大，間接決定樹上能留多少 prefix；它不改變 match 或 LRU。`max_running_requests` 限制同時握有 `lock_ref` 的請求數，因此也限制「不可驅逐」的 KV 量。兩者都不是 cache 政策。
- 想回到論文的排程行為要顯式 `--schedule-policy lpm`。只開著預設 `fcfs` 仍有 radix reuse，但沒有 longest-prefix 重排。
- 多輪在記憶體壓力下想少被別人擠掉，現行旗標是 `--enable-session-radix-cache` 加穩定的 `session_id`，而不是再找一個 2024 的 pin API。

### Gaps
- `mem_fraction_static` 的 reserved memory 公式（與 chunked prefill、CUDA graph 的關係）沒有從當日 `main` 的函式引用，只採用文件的一句話。
- `--enable-session-radix-cache` 的預設布林值沒有出現在摘到的 server arguments 表裡。
- `ChunkCache` 是否仍有 `registry.py` 以外的呼叫者，未全庫搜尋。

## 2025–2026 的實作和 2024 RadixAttention 論文差在哪？

### Takeaway
論文的不變量還在：CPU radix tree、GPU paged KV、最長 prefix match、命中中段就 split、葉優先 LRU、running 請求用 reference count 擋住驅逐、cache 與 running request 搶同一個 pool。已改變的是：預設排程從 longest-prefix 變成 FCFS；page 可以大於 1 token；驅逐政策變成一組；節點多了 `extra_key`、`cache_salt`、`hit_count`、`priority`、hash；預設樹是帶 FULL/SWA/Mamba 槽的 `UnifiedRadixCache`；HiCache 接在這棵樹上而不是獨立的 `HiRadixCache`；多輪有軟性的 session 參考；DSL fork hint 不再是 serving 的必要路徑。

### Cited Findings
- 論文頁大小固定等於一個 token；程式 `--page-size` 預設仍是 1，但 `RadixKey.page_aligned` 與 `match_prefix` 在 `page_size > 1` 時只快取完整頁。— [論文](https://arxiv.org/html/2312.07104v2)；[radix_cache.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)；[Server Arguments](https://docs.sglang.io/docs/advanced_features/server_arguments)
- 論文只有 LRU，而且說沒有另劃固定 cache pool。程式有具名的 KV pool（`token_to_kv_pool_allocator`），大小跟 `mem_fraction_static` 走；驅逐是把葉的 index `free_segment` 還給 allocator。政策除 LRU 外還有文件化的 `lfu`、`slru`、`priority`、`tlru`。— [論文](https://arxiv.org/html/2312.07104v2)；[radix_cache.py `evict`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)；[eviction docs](https://docs.sglang.io/docs/advanced_features/radix_eviction_policy)
- 論文的 cache-aware 排程是預設敘事（對比 FCFS）。v0.3.6（2024-11）預設字串仍是 `lpm`。2026-10-14 文件預設是 `fcfs`，`lpm` 與 `dfs-weight` 仍可選，另外多了程式內的 `hrrn` 與 experimental `shortest-prefill-first`，以及佇列過長時 `lpm`/`hrrn` 退回 `fcfs`。— [v0.3.6](https://github.com/sgl-project/sglang/blob/v0.3.6/python/sglang/srt/server_args.py)；[Server Arguments](https://docs.sglang.io/docs/advanced_features/server_arguments)；[schedule_policy.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_policy.py)
- 論文沒有 per-request namespace。程式用 `extra_key` 與 `cache_salt` 把相同 token 拆成互不相交的子樹，並在 in-batch 模擬樹裡一併帶上。— [radix_cache.py `match_prefix` 與 `child_key_at`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)
- 論文的前端 hint 只服務 DSL `fork`。現行多輪文件要求每次重送完整 prompt；session 只登記可重用 leaf 的軟參考。— [論文](https://arxiv.org/html/2312.07104v2)；[session 文件](https://docs.sglang.io/docs/advanced_features/session_radix_cache)
- Hybrid 與分層：論文只處理 full attention 的 KV（另說多模態可延伸到 image token）。2026-10 的預設樹是 `UnifiedRadixCache`，FULL / SWA / Mamba 分槽，視窗外 SWA 可獨立成 tombstone。HiCache 預設關閉；打開後同一棵樹 match 到 host 與 storage。`HiRadixCache` 已於 2026-09-22 刪除。Rust 實作 PR #29074 自述不是預設路徑。部落格稱 L1 Rust prototype #29074 相對 Python tree，SWA TTFT 降 38%、full attention 降 10%、hybrid SSM 降 5%；同一篇開頭又寫「Rust tree core lowers SWA TTFT by up to 42%」。兩個數字都出自該文，沒有在這裡對實驗設定。— [registry.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/registry.py)；[Unified Radix Cache blog](https://www.sglang.io/blog/unified-radix-cache)；[PR #40787](https://github.com/sgl-project/sglang/pull/40787)；[PR #29074](https://github.com/sgl-project/sglang/pull/29074)；[HiCache design](https://docs.sglang.io/advanced_features/hicache_design.html)
- `Scheduler` 在 `enable_hierarchical_cache` 時才碰 `tree_cache.cache_controller`、host load 與 storage prefetch。In-GPU radix 不依賴這條路徑。— [scheduler.py 對 `enable_hierarchical_cache` 的分支](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py)

### Inferences
- 若只談「許多併發請求共用 system prompt / few-shot」的 GPU 行為，2024 的心智模型仍然可用：match 最長邊、鎖住 running path、LRU 從葉回收。不能沿用的是預設排程名稱、page 一定等於 1 token、以及「沒有命名空間所以所有使用者必然共享」。
- `RadixCache` 沒有消失，但它不再是文件與工廠所描述的預設產品類別。讀 2026 的行為應從 `registry.py` → `UnifiedRadixCache` 看起，再用 `radix_cache.py` 理解 FULL-attention、模擬樹與 pure SWA 仍共享的演算法。

### Gaps
- 沒有把當日 `main` 對到某個 release tag 或 commit SHA，類別選擇可能在筆記寫成後再變。
- 論文 Appendix Algorithm 1 的偽碼沒有逐行對上 `Scheduler` 的 prefill/decode 迴圈，只對上了 `SchedulePolicy` 的排序。
- 部落格 42% 與 38% 的 SWA TTFT 差異沒有另找實驗報告對帳。
