# vLLM block-level GPU KV cache（PagedAttention 到 V1，截至 2026-10-14）

範圍只含單一 engine 內、用來同時服務多個 request 的 block 級 KV cache：切塊、block table、分配／釋放／共享、記憶體耗盡時的 preemption，以及 2024–2026 相對論文設計的演進。模型品質與無關功能不在此。文件頁若標 “latest developer preview”，日期以頁尾或檢索日為準，可能領先 stable。`main` 排程器是 2026-10-14 當下抓到的原始碼，不是某個 release tag。

## PagedAttention 如何把 KV cache 切成 block，block table 如何把 logical token 映到 physical block？

### Takeaway
PagedAttention 把每個 sequence 的 KV 切成固定 token 數的 KV block，logical block 可以不連續地對到 GPU 上的 physical block；block table 是這張對照表，attention kernel 用它做定址，而不是要求整段 KV 佔一塊連續顯存。

### Cited Findings
- 「PagedAttention divides the request’s KV cache into blocks, each of which can contain the attention keys and values of a fixed number of tokens.」Block 不必連續存放；作者把 blocks 比成 pages、tokens 比成 bytes、requests 比成 processes。內部碎片用較小 block 加 on-demand 分配減輕，外部碎片因為所有 block 同大小而消除，並能以 block 為粒度共享。— [Kwon et al., arXiv:2309.06180, SOSP 2023](https://arxiv.org/html/2309.06180)
- 一個 token 在每一層、每一個 attention head 都有一組 key/value。論文註腳：可以全部放在同一個 KV block，或讓不同 head／layer 各有 block、各有 block table；兩種設計「have no performance difference」，實作選後者。Block 內是固定數量 token 的 key 與 value，記為 KV block size \(B\)。Key block \(K_j=(k_{(j-1)B+1},\ldots,k_{jB})\)。Attention 改成對 block 的計算，kernel「identifies and fetches different KV blocks separately」。— [Kwon et al., §4.1](https://arxiv.org/html/2309.06180)
- KV cache manager 的類比是 virtual memory。Request 的 KV 是一串 logical KV blocks，「filled from left to right」。最後一塊未填滿的位置留給後續生成。GPU worker 上的 block engine 先分配一整段連續 GPU DRAM，再切成 physical KV blocks（swap 時 CPU RAM 同樣做）。KV block manager 維護 block tables，「the mapping between logical and physical KV blocks」。每個 entry 記錄該 logical block 對應的 physical blocks，以及「the number of filled positions」。Logical／physical 分離讓 KV 可以長大，而不必事先為最大長度保留。— [Kwon et al., §4.2](https://arxiv.org/html/2309.06180)
- 論文走讀（block 示意為 4 token）：7-token prompt 只映射前 2 個 logical block（0、1）到 physical block 7 與 1。Prefill 用一般 self-attention，前 4 個 token 的 KV 放 logical block 0，後 3 個放 logical block 1，剩 1 個 slot。下一步 decode 仍寫進最後一塊並更新 “#filled”。再下一步最後一塊滿了，才分配新的 physical block 3。浪費被限制在「one block」之內，因為「a new physical block is only allocated when all previous blocks are full」。相鄰 logical block 在實體顯存不必相鄰。— [Kwon et al., §4.3](https://arxiv.org/html/2309.06180)
- 既有系統把一個 request 的 KV 放在連續 tensor，並按最大長度預先配置，造成 reserved slots、internal fragmentation、external fragmentation。論文實驗裡既有系統只有 20.4%–38.2% 的 KV cache 記憶體用來放實際 token state。— [Kwon et al., §1 與 §3.1](https://arxiv.org/html/2309.06180)
- 現行 V1 文件把 block 定義成「number of tokens inside a block」。單一 memory pool 切成相同 page size 的 block。文件中的 page size 在分組語境是 \(\text{num_layers}\times\text{block_size}\times\text{kv_hidden_size}\)；程式裡的 `KVCacheSpec.page_size_bytes` 則是 \(\text{block_size}\times\text{kv_hidden_size}\)。`KVCacheGroupSpec` 代表「a group of model layers that share the same KV cache block table」，在 manager 裡視為一層。只有 full attention 的模型：pattern 是全部 full attention，因此只有一個 group，「the block table is shared by all layers」。— [Hybrid KV Cache Manager, 頁尾 2026-10-13，基於 commit 458e74](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager.html)；[KVCacheGroupSpec 說明見同頁所引介面](https://docs.vllm.ai/en/latest/api/vllm/v1/kv_cache_interface.html)
- `--block-size`：「Size of a contiguous cache block in number of tokens. Accepts None (meaning "use default"). After construction, always int.」未在 latest engine-args 頁給出數字預設。v0.25.1 的 `CacheConfig` 寫 `DEFAULT_BLOCK_SIZE: ClassVar[int] = 16`，`block_size is None` 時設成這個值；使用者有指定則 `user_specified_block_size = True`。更早的 CUDA platform 在 `block_size is None` 時寫死 `cache_config.block_size = 16`。Attention backend 可以改這個值：例如未指定時 FLASHMLA 會「Setting kv cache block size to 64」；使用者指定的 block size 與 backend 不相容會直接報錯，而不是悄悄覆蓋。— [Engine arguments, latest，檢索 2026-10-14](https://docs.vllm.ai/en/latest/configuration/engine_args/)；[CacheConfig, v0.25.1 API, 2026-07-13](https://docs.vllm.ai/en/v0.25.1/api/vllm/config/cache/)；[CUDA platform, v0.11.0 API](https://docs.vllm.ai/en/v0.11.0/api/vllm/platforms/cuda.html)；[PR #33600 / #35122 對 block size 與 backend 的規則](https://github.com/vllm-project/vllm/pull/33600)
- Prefix caching 的 hash 粒度可以細於實體 block：`--prefix-match-unit`「equals to the `hash_block_size`」，「controls matching granularity only, not how often states are stored」，且每個 KV group 的 `block_size` 必須能被它整除。— [Engine arguments, `--prefix-match-unit`](https://docs.vllm.ai/en/latest/configuration/engine_args/)

### Inferences
- Token 位置 \(t\) 的 logical block 是 \(\lfloor t/B\rfloor\)、block 內 offset 是 \(t \bmod B\)；physical 位址再查該 request 的 block table。這是論文 block table 與 “#filled” 的直接後果，不是 2026 原始碼裡另外抄到的函式簽名。
- 論文實作傾向「每 head／每 layer 各自的 block table」；V1 改成按 attention 類型把多層收成 `KVCacheGroup`，同組共用一張 block table，再對到同一塊 page-size 記憶體池。定址抽象還在，實體佈局已經不是論文註腳的第二種實作。

### Gaps
- 未逐行核對 2026-10-14 `main` 上 `PagedAttention` kernel 的 `block_tables` tensor shape，以及每個 backend（FlashAttention、FlashInfer、MLA）是否仍用「physical block id + offset」同一種 slot mapping。
- Latest engine-args 沒有寫死 `--block-size` 的數字；`DEFAULT_BLOCK_SIZE = 16` 來自 v0.25.1 API（2026-07），backend 仍可能在啟動時改掉。不能把「CUDA 上永遠是 16」當成 2026-10 的不變事實。

## Block manager / KV cache manager 如何在 concurrent sequences 之間分配、釋放、共享 block（continuous batching、beam search、parallel sampling）？

### Takeaway
現行路徑是 V1 `Scheduler` 每個 step 呼叫 `KVCacheManager.allocate_slots()`；block 從全域 `BlockPool` 的 free queue 取出，用 `ref_cnt` 共享，request 結束且 `ref_cnt` 歸零才回到 free queue。跨 request 的共享主要是 automatic prefix caching（只快取滿塊）。論文裡 beam search／parallel sampling 的 copy-on-write 是另一套、以 sequence group 為單位的歷史機制，V1 排程器原始碼已不再以 sequence group 做這件事。

### Cited Findings
- 每個 decoding iteration：先選一批 candidate sequences，為新需要的 logical block 分配 physical block，再把本步的 input tokens 接成一個 sequence 送進模型。Prompt-phase request 送全部 token，generation-phase request 只送最新 token。PagedAttention kernel 依 logical block 讀舊 KV，把新 KV 寫進 physical block。Request 結束後 KV blocks「can be freed to store the KV cache of other requests」。— [Kwon et al., §4.3](https://arxiv.org/html/2309.06180)
- 論文的 parallel sampling：同 prompt 的多個 sample 在 prompt phase 只留一份 KV；兩邊的 logical block 0、1 都映到同一組 physical block，physical block 的 reference count 為 2。生成階段要寫入且 `ref_cnt > 1` 時做 block 粒度 copy-on-write：分配新 physical block、複製、把 ref count 降到 1。最後一個 logical block 走 CoW，其餘 prompt block 共享。— [Kwon et al., §4.4](https://arxiv.org/html/2309.06180)
- Beam search：除了 prompt，候選之間還會共享後續 block，共享型態隨 step 變，類似 OS process tree。離開 top-k 的候選釋放 logical blocks，physical block 的 reference count 降到 0 才真正 free。CoW 只發生在「newly generated tokens are within an old shared block」。多個 sequence 組成 sequence group，因為可能共享記憶體而「always preempted or rescheduled together」。— [Kwon et al., §4.4–§4.5](https://arxiv.org/html/2309.06180)
- V1 prefix caching 的資料結構（簡化）：`KVCacheBlock` 有不可變 `block_id`、滿塊才指定且驅逐時清掉的 `block_hash`、`ref_cnt`，以及 free queue 的 doubly linked list 指標 `prev_free_block`／`next_free_block`。初始化時一次配置全部 `KVCacheBlock` 當 block pool，避免執行期建 Python 物件。元件是：Block Pool、只存 head/tail 指標的 Free Block Queue、hash → block IDs 的 cache、request ID → block IDs。— [Automatic Prefix Caching 設計文件，頁尾 2026-06-23](https://docs.vllm.ai/en/latest/design/prefix_caching.html)
- 新 request：`get_computed_blocks()` 對 prompt token 做 hash 查已算過的滿塊；`allocate_slots()` 先算還要幾塊，不夠就返回；接著 touch 命中塊（`ref_cnt += 1`，若原本沒人用就從 free queue 移除以免被驅逐）；再從 free queue 頭部 pop 新塊，若頭部是 cached block 就先驅逐（從 cache 拿掉，別人不能再 reuse）；若新分配的塊已經滿，立刻放進 cache，讓同 batch 的其他 request 能用。Running request 則只分配增量塊，token 寫進既有與新塊，滿了才 cache。— [同上](https://docs.vllm.ai/en/latest/design/prefix_caching.html)
- 「We only cache full blocks.」Hash 是 `hash(tuple[components])`：parent hash、本塊 token tuple、extra hashes（LoRA IDs、多模態輸入 hash、`cache_salt`）。文件舉例 block size 4：第三塊只對上 2/4 個 token 就不算命中。Request 結束時，若沒有其他 request 在用（`ref_cnt == 0`）就釋放；釋放時以相反順序加到 free queue 尾端，因為最後一塊 hash 了更多 token、較不容易被 reuse，應先被驅逐。— [同上](https://docs.vllm.ai/en/latest/design/prefix_caching.html)
- Eviction 是 LRU：free queue 頭部是 least recently used。若它仍是 cached block，就 pop、從 cache 刪掉 block ID、清掉 block hash，之後不能再被 prefix hit。— [同上](https://docs.vllm.ai/en/latest/design/prefix_caching.html)
- V1 的 block table 是 append-only。同一 hash 可能暫時有兩份 physical block（文件的 Request 2 用了新塊 3，而不是像 V0 那樣偵測到重複後改指回塊 1）。「This duplication will be eliminated when the request is freed.」— [同上](https://docs.vllm.ai/en/latest/design/prefix_caching.html)
- 2026-10-14 `main` 的 `vllm/v1/core/sched/scheduler.py`：`Scheduler` 持有一個 `KVCacheManager(...)`，`enable_caching=self.cache_config.enable_prefix_caching`。註解寫明沒有獨立的 decoding phase／prefill phase；每步只是讓 `num_computed_tokens` 追上 `num_tokens_with_spec`，這套迴圈同時覆蓋 chunked prefill、prefix caching、speculative decoding。Running queue 先被排，分配失敗才搶佔別人。— [scheduler.py on main](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)
- `allocate_slots` 的文件化參數（stable API，2026-10-03）：`num_lookahead_tokens` 給「spec decode proposers with kv-cache such as eagle」；`num_external_computed_tokens` 是「not cached by vLLM but cached by the connector」；`delay_cache_blocks` 給 P/D，「skip caching the blocks」因為 KV transfer 會在未來 step 完成；`num_encoder_tokens` 給 encoder-decoder（例如 Whisper）的 cross-attention，decoder-only 應為 0；`full_sequence_must_fit` 當 admission gate；`reserved_blocks` 要求留下空塊給其他 in-flight sequences。未驗證的 draft token 不進 cache：「we only cache the verified tokens (by capping the number at `request.num_tokens`)」。— [kv_cache_manager API, stable, 2026-10-03](https://docs.vllm.ai/en/stable/api/vllm/v1/core/kv_cache_manager/)
- V1 三層：`KVCacheManager` 是 scheduler 介面；`KVCacheCoordinator` 分 `KVCacheCoordinatorNoPrefixCache`（關 prefix cache）、`UnitaryKVCacheCoordinator`（只有一個 KV group）、`HybridKVCacheCoordinator`（恰好兩組，且必須有一組 full attention；其他情況「are not implemented」，可關 prefix caching 退回 NoPrefixCache）。底下是 per-group `SingleTypeKVCacheManager`（full attention、sliding window、Mamba）。標準部署裡 `BlockPool` 在 coordinator 建立，各 `SingleTypeKVCacheManager` 共享。— [Hybrid KV Cache Manager, 2026-10-13](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager.html)
- Prefix caching 設計文件把 SGLang 只當作「也廣泛使用 prefix caching」的開源框架例子，沒有比較兩者的 block 分配器。— [Prefix caching 設計文件](https://docs.vllm.ai/en/latest/design/prefix_caching.html)

### Inferences
- Continuous batching 在 V1 不是另一個 allocator，而是 `Scheduler.schedule()` 每步混排 running（多為 decode 或未完成的 prefill chunk）與 waiting，並只向同一個 `BlockPool` 要塊。
- 論文的 beam／parallel sampling 共享靠「同一個 request 裡多條 sequence 的 block table 指到同一 physical block + CoW」。V1 跨 request 共享靠滿塊 hash 與 `ref_cnt`。2026-10-14 的 scheduler 原始碼沒有 sequence group，也沒有 beam 的 fork/CoW 路徑；檔內唯一 “copy-on-write” 註解是在講 local prefix hit 的 sub-block tail 不要和更長的 remote connector hit 競態，不是 beam search。
- V0 會在發現重複 hash 時改寫 block table；V1 明確不改，所以並發下同一內容可能短暫佔兩份 GPU block，直到 request free。這會讓「共享」比論文的 refcount 理想情況更費記憶體。

### Gaps
- 未在 2026-10 的 `SamplingParams.n`（平行採樣）路徑上確認：多個 sample 是變成多個 `Request` 只靠 prefix cache 共享 prompt，還是仍有 in-request CoW。Scheduler 原始碼看不到後者。
- `BlockPool` 的完整方法（`get_new_blocks`、`free_blocks`、`cache_full_blocks`）這次沒有整檔讀完，行為依設計文件與 `allocate_slots` docstring，而不是逐行對上 2026-10-14 的 `block_pool.py`。
- Hybrid coordinator「恰好兩組」的限制，文件自己標成 early stage（commit 458e74）；超過兩種 attention 類型時 prefix caching 是否直接被關掉，要以該 commit 之後的程式為準，這裡沒有再追。

## GPU 記憶體耗盡時怎麼辦：preemption、swapping、recomputation、priority？近期 vLLM 的預設？

### Takeaway
論文與 V0 有 swapping（GPU block 整段搬到 CPU）與 recomputation 兩種恢復，預設在單 sequence 時走 recomputation。V1 作為現在唯一的 engine，預設是 recompute：搶佔時直接 free GPU blocks、把 `num_computed_tokens` 清零、請求回到 waiting queue 重算。V1 排程器沒有 swap preemption；CPU 上的 KV 是另外的 offload／KV connector，不是論文那個 CPU block allocator。

### Cited Findings
- 排程政策論文採用 FCFS。「the earliest arrived requests are served first and the latest requests are preempted first.」因為一個 sequence 的所有 block 會一起被用到，evict 是 all-or-nothing。Sequence group 一起被搶佔或重新排程。— [Kwon et al., §4.5](https://arxiv.org/html/2309.06180)
- Swapping：被趕出的 block 複製到 CPU。除了 GPU block allocator 還有 CPU block allocator。GPU physical block 用盡時選一批 sequence 把 KV 轉到 CPU。搶佔發生後「stops accepting new requests until all preempted sequences are completed」；有 request 完成、空出塊之後，才把被搶佔 sequence 的塊搬回。CPU swap 的塊數「never exceeds the number of total physical blocks in the GPU RAM」。— [Kwon et al., §4.5](https://arxiv.org/html/2309.06180)
- Recomputation：被搶佔的 sequence 再排程時重算 KV。Decode 已經產生的 token 可以接回 prompt，在一次 prompt-phase iteration 重算所有位置，所以「recomputation latency can be significantly lower than the original latency」。— [Kwon et al., §4.5](https://arxiv.org/html/2309.06180)
- 論文消融：小 block 時 swapping 因為大量小傳輸吃不滿 PCIe，overhead 高；recomputation 不依賴 block，overhead 跨 block size 大致持平，且「never higher than 20% of swapping’s latency」。中等 block size（文中 16 到 64）兩者接近，但 swapping 對 block size 更敏感。— [Kwon et al., §7.3](https://arxiv.org/html/2309.06180)
- V0 `vllm/core/scheduler.py` 的 `_preempt`：使用者沒指定時，`get_max_num_running_seqs() == 1` 用 `PreemptionMode.RECOMPUTE`，否則（例如 beam search）用 `SWAP`，因為「recomputation is not currently supported」於多 sequence。註解承認這會讓多 sequence 的 group 隱性優先於單 sequence，因為 swapped sequences 優先於 waiting sequences。— [vllm.core.scheduler API，頁面標 2025-07-24](https://docs.vllm.ai/api/vllm/core/scheduler.html)
- 舊版 engine args：`--preemption-mode` 可選 `recompute`、`swap`、`None`，預設 `None`，意思是「We use recomputation by default since it incurs lower overhead than swapping」，但多 sequence 仍改走 swapping。同期 `--swap-space`「Size of the CPU swap space per GPU (in GiB). Default: 4」。— [舊版 Engine Arguments](https://docs.vllm.ai/configuration/engine_args.html)
- V1 調優文件（頁尾 2026-08-20）：KV 不夠時可以 preempt，被搶佔的 request「are recomputed when sufficient KV cache space becomes available again」。警告文是 `PreemptionMode.RECOMPUTE`。原句：「In vLLM V1, the default preemption mode is `RECOMPUTE` rather than `SWAP`, as recomputation has lower overhead in the V1 architecture.」緩解手段是提高 `gpu_memory_utilization`、降低 `max_num_seqs` 或 `max_num_batched_tokens`、加大 tensor／pipeline parallel。可用 Prometheus 看 preemption 次數。— [Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/)
- 2026-10-14 `main` 的 `Scheduler.schedule`：`allocate_slots` 回 `None` 就搶佔。`policy == PRIORITY` 時受害者是 `max(self.running, key=lambda r: (r.priority, r.arrival_time))`，也就是 priority 值較大（較不優先）且到達較晚者。否則 `preempted_req = self.running[-1]`，FCFS 從 running 尾端 pop。`_preempt_request` 呼叫 `_free_request_blocks`、`encoder_cache_manager.free`，狀態改 `RequestStatus.PREEMPTED`，`num_computed_tokens = 0`，清掉 `spec_token_ids`，`num_preemptions += 1`，再 `waiting.prepend_request`。沒有把 KV 拷到 CPU swap 的分支。— [scheduler.py on main](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)
- `--scheduling-policy` 預設 `fcfs`。「"fcfs" means first come first served」；`"priority"` 是「lower value means earlier handling」，同 priority 再比到達時間。— [Engine arguments, latest](https://docs.vllm.ai/en/latest/configuration/engine_args/)
- `--watermark` 預設 `0.0`：「Fraction of total KV cache blocks to keep free ... when admitting waiting or preempted requests」。0.0「disables the watermark」。目的是避免 GPU 很緊時反覆 eviction／preemption。`allocate_slots` 只在「there's at least one request already scheduled」且 request 狀態是 `WAITING` 或 `PREEMPTED` 時套用 watermark。— [Engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/)；[allocate_slots docstring, 2026-10-03](https://docs.vllm.ai/en/stable/api/vllm/v1/core/kv_cache_manager/)
- `--scheduler-reserve-full-isl` 預設 `True`：准入前檢查「the full input sequence length fits in the KV cache」，不是只檢查第一個 chunk，避免 chunked prefill 過度准入造成 thrashing。— [Engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/)
- 2025-09 的官方技術文：V1「may attempt recompute preemption (swap preemption was supported in V0)」；不夠塊就 `kv_cache_manager.free`，把塊還給 block pool。— [Inside vLLM, 2025-09-05](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)
- 論文的「搶佔後停止收新 request，直到被搶佔者全部跑完」與現行碼不一致：V1 把被搶佔者 prepend 回 waiting，下一個 step 可以再和其他 waiting request 一起搶塊。這是程式行為，不是文件裡的一句政策宣言。— [scheduler.py `_preempt_request`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)；對照 [Kwon et al., §4.5](https://arxiv.org/html/2309.06180)

### Inferences
- 2026 年 10 月若沒有另接 KV connector／offload，記憶體耗盡的預設恢復就是丟掉該 request 的 GPU block 並重算，不是 swap。調優文件仍印著 `PreemptionMode.RECOMPUTE` 這個 V0 列舉名字，但 latest engine-args 已沒有 `--preemption-mode` 或 `--swap-space`；那兩個旗標應視為 V0 歷史介面。
- FCFS 下受害者是 `running[-1]`。Running list 的尾端是最近被放進去、因而在 FCFS 意義上較晚的 request，對得上論文「latest requests are preempted first」，但是 V1 的 waiting queue 在 prefix-cache miss 時會 `continue` 而不是 `break`，註解寫明這「do not strictly follow the FCFS scheduling policy」。搶佔選擇與准入順序不是同一個 FCFS。
- Watermark 預設關閉。它不是論文的 swap watermark（舊 `BlockSpaceManager` 的 swapping watermark 預設 0.01，見下方 V0 API）。

### Gaps
- 沒有量到「重算 vs 2026 offload connector」的延遲數字；論文 §7.3 的 20% 是 2023 年 swapping vs recompute，不能套到 `OffloadingConnector`。
- `running[-1]` 是否永遠等於「最晚到達」取決於 running list 的插入順序；這次沒有把 `schedule()` 前半段（誰被 append 進 `self.running`）整段讀完。Priority 路徑的 key 則是原始碼寫明的。
- 被搶佔後 prefix cache 是否還留得住：`_free_request_blocks` 的實作沒有展開。若只是 `ref_cnt -= 1` 且塊曾被 cache，滿塊可能仍在 free queue 裡被之後的 `get_computed_blocks` 命中，重算量會小於「整個 prompt 重算」。這是從 prefix caching 設計推論，不是 `_free_request_blocks` 的逐行確認。

## V1 engine 和原來的 block manager 差在哪？類別、模組、設定旗標？

### Takeaway
V0 的 `vllm.core` scheduler／block manager 與 swapping 已在 v0.11.0 從程式庫移除；2026 年 10 月的引擎就是 V1。KV 管理從 `BlockSpaceManager` 換成 `KVCacheManager` + `BlockPool` + per-group `SingleTypeKVCacheManager`。Prefix caching 與 chunked prefill 在 V1 改為能開就開；`gpu_memory_utilization` 預設 0.92。

### Cited Findings
- v0.11.0 release：「This release completes the removal of V0 engine. V0 engine code including AsyncLLMEngine, LLMEngine, MQLLMEngine, all attention backends, and related components have been removed. V1 is the only engine in the codebase now.」同一則也列了 KV cache CPU offloading with LRU。其後版本已到 v0.31.0（2026-10-02）與 v0.32.0rc0（2026-10-12）。— [Release v0.11.0](https://github.com/vllm-project/vllm/releases/tag/v0.11.0)；[release 列表，頁面 2026-10-12](https://releasealert.dev/github/vllm-project/vllm)
- Stable 的 V1 指南仍寫「V1 is now enabled by default」且可用 `VLLM_USE_V1=0` 關掉。這與「V0 程式已刪除」的 release note 衝突；指南比較像棄用過程中的文字，不該當成 2026-10 main 仍能切回 V0。— [vLLM V1 guide, stable](https://docs.vllm.ai/en/stable/usage/v1_guide.html)；對照 [v0.11.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.11.0)
- V1 為什麼重寫：保留模型、GPU kernel、工具，重做 scheduler、KV cache manager、worker、sampler、API server。目標包含「Require zero configs by enabling features/optimizations by default」。— [vLLM V1 guide, stable](https://docs.vllm.ai/en/stable/usage/v1_guide.html)
- 現行類別與模組（2026-10-14 `main` scheduler 的 import 與建構）：
  - `vllm.v1.core.sched.scheduler.Scheduler`，latest engine-args 寫明這是 `--scheduler-cls` 的預設。
  - `vllm.v1.core.kv_cache_manager.KVCacheManager`
  - `vllm.v1.core.kv_cache_utils.KVCacheBlock`
  - `vllm.v1.kv_cache_interface.KVCacheConfig`、`MambaSpec`
  - Coordinator／`SingleTypeKVCacheManager`／`BlockPool` 在設計文件：`vllm/v1/core/kv_cache_coordinator.py`、`single_type_kv_cache_manager.py`、`block_pool.py`
  - Encoder 側是另一個 manager：`EncoderCacheManager` 或 encoder-decoder 的 `EncoderDecoderCacheManager`，不是 KV `BlockPool`。
  — [scheduler.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)；[Hybrid KV Cache Manager](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager.html)；[Engine arguments `--scheduler-cls`](https://docs.vllm.ai/en/latest/configuration/engine_args/)
- V0 歷史類別：`vllm.core.scheduler.Scheduler._preempt`，以及 `vllm.core.block_manager` 的 block manager。舊 docstring：`block_size`、`num_gpu_blocks`、`num_cpu_blocks`、swapping 用的 `watermark` 預設 0.01、`enable_caching` 預設 False。Speculative decoding 用 lookahead slots 存 proposal token 的 KV。— [vllm.core.scheduler](https://docs.vllm.ai/api/vllm/core/scheduler.html)；[vllm.core.block_manager](https://docs.vllm.ai/api/vllm/core/block_manager.html)
- 旗標（latest engine-args，檢索 2026-10-14，除非另註）：
  - `--gpu-memory-utilization` 預設 `0.92`。「per-instance limit」；同一張 GPU 上兩個 vLLM instance 要各自設，例如各 0.5。非 GPU 裝置時控制對應裝置記憶體。文件寫啟動時用這個比例預先配置 GPU cache。— [Engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/)；[Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/)
  - `--kv-cache-memory-bytes`：預設 None，依 `gpu_memory_utilization` 推 KV 大小；若設定則「ignores gpu_memory_utilization」。— [Engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/)
  - `--block-size`：見上一節，建構前可以是 None，之後必為 int。v0.25.1 的類別預設常數是 16。
  - `--enable-prefix-caching`：latest 頁只寫「Whether to enable prefix caching」，沒印布林預設。Stable cache API（2026-10-03）寫「Enabled by default for V1。」較舊的 config 頁寫「Disabled by default for V0. Enabled by default for V1.」— [cache API, stable](https://docs.vllm.ai/en/stable/api/vllm/config/cache.html)；[舊 config API](https://docs.vllm.ai/api/vllm/config.html)
  - `--prefix-caching-hash-algo` 預設 `sha256`。選項還有 `sha256_cbor`、`xxhash`、`xxhash_cbor`。文件：v0.11 起預設 sha256，因為先前 hash「was not guaranteed to be collision-free」。`xxhash` 被標成非密碼學安全，多租戶有碰撞導致「undefined behavior or even leak private information」的理論風險。— [Prefix caching 設計文件](https://docs.vllm.ai/en/latest/design/prefix_caching.html)；[Engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/)
  - `--num-gpu-blocks-override`：覆蓋 profile 出來的 `num_gpu_blocks`，「Used for testing preemption.」
  - `--max-num-seqs`：`SchedulerConfig` 欄位「mainly for convenience when testing」，正式值在 `EngineArgs.create_engine_config` 設。它同時是 model runner 的 per-request buffer／CUDA graph 上限，以及預設的 RUNNING 准入上限。
  - `--max-num-active-seqs`：預設 None，表示沿用 `max_num_seqs`。設了只縮小 RUNNING 准入，不縮小 runner／graph。
  - `--scheduling-policy` 預設 `fcfs`。
  - `--watermark` 預設 `0.0`。
  - `--scheduler-reserve-full-isl` 預設 `True`。
  - `--disable-hybrid-kv-cache-manager`：True 時即使有 full attention 與 sliding window，也「allocate the same size of KV cache for all attention layers」。預設 None，再依環境決定。
  - `--kv-offloading-size`：GiB，預設 None 表示不做 KV offload；TP>1 時是各 rank 加總。`--kv-offloading-backend` 預設 `native`，另一選項 `lmcache`。
  - 舊的 `--preemption-mode`、`--swap-space`（預設 4 GiB）出現在 2025-07 前後的 engine args，不在 2026-10-14 latest 頁的 CacheConfig／SchedulerConfig 節錄裡。
  — [Engine arguments, latest](https://docs.vllm.ai/en/latest/configuration/engine_args/)
- V1 prefix caching 相對於 V0 的行為差：block table append-only，不在命中時把新塊重綁到舊塊；V0 會 free 掉重複塊並改 block table。— [Prefix caching 設計文件](https://docs.vllm.ai/en/latest/design/prefix_caching.html)
- 分散式：論文仍是單一 centralized KV cache manager，GPU workers 共用 logical→physical 對照；每個 worker 的 physical block ID 相同，但只存自己那些 attention heads 的 KV 切片。Scheduler 每步把 token IDs 和 per-request block table 廣播給 workers。— [Kwon et al., §4.6](https://arxiv.org/html/2309.06180)

### Inferences
- 「V1 是否為預設」在 2026-10 應該回答成：V1 是程式庫裡唯一的 engine，不是還能用環境變數切回的預設選項。Stable 指南上的 `VLLM_USE_V1=0` 與 release note 衝突，以 v0.11.0 已刪除 V0、其後又發了 v0.29–v0.31 為準。
- 記憶體預算的現行旋鈕是 `gpu_memory_utilization=0.92`（或直接給 `--kv-cache-memory-bytes`），不再是論文的「GPU 塊數 vs 等量 CPU swap 塊」。Profile 先扣掉權重與啟動時量到的峰值，剩下的才變成 `num_gpu_blocks`；`--num-gpu-blocks-override` 是測試 preemption 的人工上限。

### Gaps
- `EngineArgs.create_engine_config` 裡 `max_num_seqs` 的實際數字沒有從官方 API 頁讀到。2026-10-03 的論壇帖描述 v0.30 預設 `max_num_seqs` 1024 會讓某個 Mamba 模型在 733 個 cache block 時起不來，這是使用者回報，不是 `SchedulerConfig` 的常數。— [vLLM forum](https://discuss.vllm.ai/t/booting-14-current-open-models-on-vllm-v0-29-v0-30-what-broke-with-the-defaults/3025)
- 未確認 `--disable-hybrid-kv-cache-manager` 在 None 時的現行預設布林值（文件說依環境決定）。
- 論文的「單一 scheduler 對所有 TP worker 發同一張 block table」是否在 2026 的 DCP／PCP（`decode_context_parallel_size`、`prefill_context_parallel_size` 會乘進 scheduler block size）下仍然一字不差，這次只看到 scheduler 把這兩個 world size 傳進 `KVCacheManager`，沒有追 worker 端切片。

## 同一個 engine process 裡，不同使用者的 request 如何隔離或交錯？什麼是共享的，什麼是 per-request？

### Takeaway
一個 engine process（V1 的 EngineCore scheduler）裡，所有使用者的 request 共用同一塊 GPU block pool 與同一張 prefix-cache hash 表，並在同一步 forward 裡交錯。隔離單位是 `Request`（自己的 token、block table、sampling、priority），不是使用者或 process。預設沒有 per-user 記憶體配額；要隔開 prefix cache 得用 `cache_salt`。

### Cited Findings
- 論文：集中式 scheduler 協調分散的 GPU workers；KV cache manager 管的是 worker 上的 physical KV。Kernel「only see a list of physical block IDs for each sequence and do not need to handle sharing patterns across sequences。」不同 decoding 偏好的 request 可以同時在一個 batch。— [Kwon et al., §4 與 §4.4](https://arxiv.org/html/2309.06180)
- V1 scheduler 的共享狀態：一個 `self.requests: dict[str, Request]`、一個 `waiting` queue、一個 `running: list[Request]`、一個 `KVCacheManager`（其下一個 block pool）。Per-request 狀態在 `Request`：`request_id`、status、`num_computed_tokens`、`num_prompt_tokens`、`priority`、`arrival_time`、`spec_token_ids`、`num_preemptions`，以及 manager 裡 request ID → block IDs。— [scheduler.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)；[Prefix caching 設計文件](https://docs.vllm.ai/en/latest/design/prefix_caching.html)
- 交錯方式：V1「There's no decoding phase nor prefill phase」。Chunked prefill 在 V1「enabled by default whenever possible」。有 chunked prefill 時「prioritizes decode requests」：先把待處理的 decode 放進 batch，`max_num_batched_tokens` 還有餘額才排 prefill；塞不下就把 prefill 切塊。— [scheduler.py 註解](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)；[Optimization and Tuning, 2026-08-20](https://docs.vllm.ai/en/latest/configuration/optimization/)
- `gpu_memory_utilization`「only applies to the current vLLM instance. It does not matter if you have another vLLM instance running on the same GPU.」實例之間不共享這份預算；實例內部的 request 則共享該實例算出的那一池 KV block。— [Engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/)
- Prefix cache 預設跨 request 共用，包含不同使用者的相同前綴。隔離要顯式加鹽：「By including a `cache_salt` in the request, this value is injected into the hash of the first block, ensuring that only requests with the same salt can reuse cached KV blocks.」目的寫明是防 timing attack，「infer cached content by observing latency differences」。`cache_salt` 不是記憶體配額，只限制誰能 reuse 塊。— [Prefix caching 設計文件](https://docs.vllm.ai/en/latest/design/prefix_caching.html)
- 多模態 hash 把前端 image processor 的 image hash 放進 extra hash，避免 placeholder token 相同但圖不同時誤用 KV。— [同上](https://docs.vllm.ai/en/latest/design/prefix_caching.html)
- LoRA 是另一種排程隔離：scheduler 會讓本步的 LoRA 數不超過 `max_loras`，超過就 skip 該 request，不是另外一池 KV。— [scheduler.py 約 L941](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)
- Data parallelism 是複本而不是同一池：調優文件說 DP「replicates the entire model across multiple GPU sets and processes different batches」。`--max-num-queued-reqs` 在 API server process 跨 DP rank 計數；`--max-num-seqs` 是 per data-parallel rank。— [Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/)；[Engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/)
- 進程模型：至少 1 個 API server、1 個 engine core（scheduler 在這裡）、N 個 GPU worker。Scheduler 與 KV manager 在 engine core，不在每個使用者連線裡。— [Optimization and Tuning, “CPU Resources for GPU Deployments”](https://docs.vllm.ai/en/latest/configuration/optimization/)

### Inferences
- 共享的是：GPU KV block pool、prefix-cache hash map、free queue、本步的 batched forward、（TP 下）同一組 block ID。Per-request 的是：token 序列、block table、已計算 token 數、sampling、priority、encoder cache 佔用、speculative token。
- 沒有文件化的 per-user KV 配額。一個很長的 request 可以透過 FCFS／priority 搶佔把別的使用者擠去重算。`max_num_seqs`、`max_num_active_seqs`、`max_num_batched_tokens`、`watermark`、`scheduler_reserve_full_isl` 是全域閥，不是租戶閥。
- `cache_salt` 相同的使用者會共享物理塊；鹽不同則相同字串也不命中，於是各存一份。這是隱私隔離，會降低複用、增加顯存壓力。

### Gaps
- 未查 API 層是否還有別的租戶鍵（例如 organization id）會自動進 block hash。設計文件只點名 `cache_salt`、LoRA ID、多模態 hash。
- Engine core 與 worker 之間 block table 的序列化格式（`SchedulerOutput` 裡的 `req_to_new_blocks`）沒有展開，所以「worker 是否會自己再分配一塊」只依論文 §4.6 與 connector 的分工來理解：一般路徑是 scheduler 分配、worker 照表讀寫。

## 2024–2026 有哪些改動會碰到 block pool：chunked prefill、speculative decoding、encoder-decoder／多模態、disaggregated prefill/decode？

### Takeaway
這些功能都還坐在同一套 paged block 上，但改變「何時佔塊、塊裡算不算 cache、KV 身在哪一個 process」。Chunked prefill 讓長 prompt 分段佔塊並與 decode 同 batch；speculative decoding 會多留 lookahead 塊且只快取已驗證 token；多模態的 encoder cache 是另一個預算，encoder-decoder 才把 encoder token 算進 KV slot；disaggregated prefill 把 KV 留在各 instance 自己的 paged pool，再用 KV connector 在 pool 之間搬。

### Cited Findings
- Chunked prefill：V1 盡可能預設開啟。長 prefill 切成不超過 `max_num_batched_tokens` 的塊，並與 decode 同一 batch。Decode 優先。`max_num_batched_tokens == max_model_len` 時「almost the equivalent to the V0 default scheduling policy (except that it still prioritizes decodes)」。關掉 chunked prefill 時，`max_num_batched_tokens` 必須大於 `max_model_len`，否則可能在啟動崩潰。— [Optimization and Tuning, 2026-08-20](https://docs.vllm.ai/en/latest/configuration/optimization/)
- 與 block pool 的交互：`--scheduler-reserve-full-isl` 預設 True，准入時要求整段 input 放得進 KV，而不是只看第一個 chunk，避免「over-admission and KV cache thrashing」。`allocate_slots(..., full_sequence_must_fit=...)` 的註解相同。Waiting／preempted 的 watermark 也是為了不要在塊很緊時反覆搶佔。— [Engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/)；[allocate_slots docstring](https://docs.vllm.ai/en/stable/api/vllm/v1/core/kv_cache_manager/)
- Mamba／hybrid 與 chunk 邊界：scheduler 的 `_mamba_block_aligned_split` 會把 prefill chunk 剪到 Mamba state 必須被 cache 的位置。註解：「In "align" cache mode reusable SSM states are materialized at block boundaries」。`--mamba-cache-mode` 在 latest args 被標成將於 v0.33.0 移除且「has no effect: Mamba prefix caching follows --enable-prefix-caching」。`--prefix-cache-retention-interval` 預設 0，只留語意 checkpoint。— [scheduler.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)；[Engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/)
- Speculative decoding：`num_lookahead_tokens` 用來為 eagle 這類自帶 KV 的 proposer 多配 slot。文件寫新 token 裡含未驗證 draft 時「only cache the verified tokens」。EAGLE 會丟掉最後一個 matched block，「drop the last matched block to force recompute the last block to get the required hidden states」。2026-10-14 scheduler 有 `use_eagle_block_drop`；關掉時會警告「EAGLE trailing prefix-cache block dropping is disabled」。Multi-module MTP 若 chunk 結尾距 prefill 結束小於 `num_prefill_lookahead`，會把邊界往前推，避免 draft 模組 KV「permanently polluting」。— [single_type_kv_cache_manager API, 2026-10-12](https://docs.vllm.ai/en/latest/api/vllm/v1/core/single_type_kv_cache_manager/)；[allocate_slots docstring](https://docs.vllm.ai/en/stable/api/vllm/v1/core/kv_cache_manager/)；[scheduler.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)
- 多模態與 encoder：processor cache 與 IPC cache（`mm_processor_cache_gb`，調優文件寫預設 4 GiB）快取的是處理後的多模態 tensor，不是 attention KV block。KV prefix hash 另把 image hash 放進 extra hashes。Encoder compute 有獨立預算 `max_num_encoder_input_tokens`；scheduler 在 encoder cache 耗盡時會跳過 request。`num_encoder_tokens` 只在 encoder-decoder 的 cross-attention 向 KV manager 要塊。KV connector 路徑「Encoder-decoder models are not currently supported」。— [Optimization and Tuning, Multi-Modal Caching](https://docs.vllm.ai/en/latest/configuration/optimization/)；[Prefix caching 設計文件](https://docs.vllm.ai/en/latest/design/prefix_caching.html)；[scheduler.py 建構 encoder manager 與 connector assert](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)
- Hybrid sliding window 不另開記憶體池：單一 pool、同一 page size。Full attention 為所有 token 留塊；sliding window 只留最近 `sliding_window_size` 個 token，超出的塊在 free 時進入同一條 LRU。Prefix hit 取 full attention 與 sliding window 命中的交集。KV sharing（例如一層直接用另一層的 KV）時，manager 不為 sharing 層另配塊。`--disable-hybrid-kv-cache-manager` 會讓 sliding window 也按 full attention 的長度配塊。— [Hybrid KV Cache Manager, 2026-10-13](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager.html)
- Disaggregated prefill：實驗功能，「subject to change」，「DOES NOT improve throughput」。兩個 vLLM instance，prefill 與 decode 分開，connector 把 prefill KV「transfer ... from prefill instance to decode instance」。程式在 `vllm/distributed/kv_transfer`。每個 process 各有 connector：scheduler connector 決定傳哪些 token，worker connector 執行傳輸，可層層搭配 attention。支援的 connector 含 Example、LMCache、Nixl、Mooncake、MoRI-IO、MultiConnector、Offloading、FlexKV。— [Disaggregated Prefilling, 頁尾 2026-10-03](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
- KV 仍然進各 instance 的 paged pool，而不是換成一個不分塊的大 buffer。官方文寫 decode 端 `start_load_kv`「loads KV from the external server and injects it into paged memory」；prefill 端 `wait_for_save` 等到 KV 上傳。`allocate_slots` 的 `delay_cache_blocks` 就是給「blocks used in a KV transfer which will complete in a future step」。`num_external_computed_tokens` 表示塊在 connector 而不在本地 prefix cache。— [Inside vLLM, 2025-09-05](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)；[allocate_slots docstring](https://docs.vllm.ai/en/stable/api/vllm/v1/core/kv_cache_manager/)
- Offload 到 CPU 不是 V0 swap preemption：`--kv-transfer-config` 的 `OffloadingConnector` 可另定 CPU `block_size` 與 `cpu_bytes_to_use`。`--kv-offloading-size` 預設不開。Scheduler 在 consumer、且有 overlapping batches 時會 `defer_block_free`，避免下一步的 load 寫進尚未寫完的 freed block。— [Disaggregated Prefilling](https://docs.vllm.ai/en/latest/features/disagg_prefill/)；[Engine arguments `--kv-offloading-size`](https://docs.vllm.ai/en/latest/configuration/engine_args/)；[scheduler.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)
- vLLM-Ascend 的 `PreemptOffloadConnector` 是外掛：沒設定時「falls back to the original recompute behavior」，而且「only intended for Decode nodes in P/D disaggregation」，不是上游預設。— [vLLM Ascend preempt offload guide](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/preempt_offload_connector.html)

### Inferences
- 2024–2026 沒有換掉「固定大小 physical block + per-request block table」這層。變的是：誰來決定一塊算不算 prefix（滿塊、hash、EAGLE 丟最後一塊、Mamba 只在邊界留 state）、一塊要留多久（sliding window 提早 free）、以及塊的位元組在 prefill process、decode process 還是 CPU offload。
- Chunked prefill 會讓一個 request 在跑完 prompt 之前就佔著已經算完的那些塊，同時 decode request 也佔塊。`scheduler_reserve_full_isl=True` 是為了這個交互才預設打開的准入閘，不是 PagedAttention 論文的一部分。
- 多模態「快取」有兩份，不要混：`mm_processor_cache_gb` 是 CPU／shared-memory 上的前處理 tensor；GPU KV 仍是 block pool，只是 hash 多了 image id。Encoder-decoder 的 cross-attention 才會向同一個 KV manager 再要 `num_encoder_tokens` 塊。

### Gaps
- 未讀 `NixlConnector` 是否要求兩邊 `block_size` 相同，或傳輸時會重切塊。文件只保證 KV 被注入 decode 端的 paged memory。
- EAGLE「丟掉最後一個 matched block」的規則來自 2026-10-12 的 `find_longest_cache_hit` docstring 與 scheduler 的 `use_eagle_block_drop`；不同 speculator（n-gram、Medusa）是否也佔 lookahead 塊，沒有逐一查。
- Hybrid 設計文件自己警告內容基於 commit 458e74 且「things may change」。頁尾是 2026-10-13，但是文內的分組啟發式不能當成 v0.32 已凍結。
- Q4 2026 roadmap 寫到「Implement extensible KV cache support」與關掉 Model Runner V1，代表 block 管理介面仍在動；這份筆記對上的是 2026-10-14 的文件與 `main` scheduler，不是 roadmap 實作完成之後的狀態。— [Roadmap issue #61024, 2026-10-10](https://github.com/vllm-project/vllm/issues/61024)
