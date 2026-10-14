# vLLM Automatic Prefix Caching：跨請求 prefix KV 的准入、查找與驅逐（截至 2026-10-14）

範圍只含跨請求重用已算過的 prefix KV。PagedAttention block pool、preemption、一般 block 分配不在此處展開；block 只在 hash、pin（reference count）與驅逐時出現。以下「latest」指 2026-10-14 抓到的 developer preview（`docs.vllm.ai/en/latest`）。V0 engine 的預設與 hash 已標為過時。官方 APC 功能頁沒有 TTFT 或 hit rate 數字；量化只引用論文或官方 blog 裡寫明的數字。

## APC 是什麼？block 怎麼 hash？兩個請求要共享一塊必須相同什麼？

### Takeaway
Automatic Prefix Caching（APC）把已算完的 full KV block 放進一張以內容 hash 為鍵的表；新請求若從 prompt 起點起有相同的 hash chain，就直接 pin 那些物理 block、跳過對應 prefill。hash 不是「這個 block 自己的 token」，而是 parent hash、本 block 的完整 token ids，再加上 LoRA / multimodal / `cache_salt` / prompt embeds 等 extra keys。只快取滿 block；尾端不滿的 block 不能被下一筆請求重用。

### Cited Findings
- APC 快取既有 query 的 KV，新 query 若與某筆既有 query 共享 prefix，就重用該段 KV、跳過共享部分的計算。官方功能頁要讀者另看設計頁。 — [Automatic Prefix Caching（stable features）](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)
- vLLM 用 hash，不用樹。每個 block 的身份是「本 block tokens + 它前面的 prefix tokens」。設計頁給的元件是：parent hash、本 block 的 token tuple（降低碰撞）、extra hashes（LoRA、multimodal input hashes、多租戶 `cache_salt`）。原文：「We only cache full blocks.」 — [Automatic Prefix Caching（latest design）](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- 設計頁的多模態例子（**假設** block size 16，不是文件宣告的全域預設）：`[IMG]` 變成 placeholder token，extra hash 放 frontend image processor 產生的 image hash。同一段 placeholder、不同圖片會得到不同 block hash。parent hash 再把差異傳給後續 block。 — [同上](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- v0.28.0 的 `hash_block_tokens`：若沒有 parent 就用 `NONE_HASH`；對 `(parent_block_hash, token id tuple, extra_keys)` 做 hash。docstring 寫整個 tuple 當作 block 的 hash key；不滿的 block 不 hash。 — [kv_cache_utils v0.28.0](https://docs.vllm.ai/en/v0.28.0/api/vllm/v1/core/kv_cache_utils/)
- v0.18.0 原始碼：`need_extra_keys` 在有 `mm_features`、`lora_request` 或 `cache_salt` 時為真。`cache_salt` 只在 `start_token_idx == 0` 注入第一個 block。LoRA extra key 當時是 `lora_name`（不是權重內容）。另有 `_gen_prompt_embeds_extra_hash_keys`。 — [kv_cache_utils.py @ v0.18.0](https://github.com/vllm-project/vllm/blob/v0.18.0/vllm/v1/core/kv_cache_utils.py)
- 更早的 API 文件把 LoRA extra key 寫成 `lora_int_id`，與 v0.18 的 `lora_name` 不同。這段 `lora_int_id` 行為已被後續版本取代。 — [kv_cache_utils（舊 API 站）](https://docs.vllm.ai/api/vllm/v1/core/kv_cache_utils.html)
- v0.31.0（2026-10-05）changelog：prefix-cache extra keys 依來源加 tag，避免 LoRA name 與 `cache_salt` 撞鍵（#51899）；block hash 納入 LoRA path（#59335）。只靠 name 時，unload 再以同名載入另一個 adapter、或 `load_inplace` 換 path，會重用舊 adapter 算出的 KV，直到該 block 被驅逐。同 path 原地改檔仍無法偵測。 — [vLLM 0.31.0 release notes 轉載](https://newreleases.io/project/pypi/vllm/release/0.31.0)；[PR #59335](https://github.com/vllm-project/vllm/pull/59335)
- 自 v0.11 起預設 hash 演算法是 `sha256`，用來處理先前不能保證無碰撞的問題。`--prefix-caching-hash-algo`：`sha256`（預設，pickle 後 SHA-256，跨 Python / vLLM 版本不保證可重現）、`sha256_cbor`（canonical CBOR，文件建議用於跨環境確定性快取）、`xxhash` 與 `xxhash_cbor`（需可選套件 `xxhash`）。文件對非密碼學 hash 的原文包含：「can cause undefined behavior or even leak private information in multi-tenant environments.」 — [latest design](https://docs.vllm.ai/en/latest/design/prefix_caching/)；[CLI serve](https://docs.vllm.ai/en/latest/cli/serve/)
- pickle SHA-256 實作：「The input is serialized using pickle before hashing」且「does not use a hash seed」。`sha256_cbor` 用 `cbor2.dumps(..., canonical=True)`。 — [hashing.py](https://github.com/vllm-project/vllm/blob/main/vllm/utils/hashing.py)
- Hybrid 模型：block pool 以類似 `(block_hash, group_id)` 的鍵分別快取各 KV group；新請求的可重用 prefix 是各 group hit 的交集。full attention 以外的 group 不會因為 token 相同就自動共享另一 group 的 block。 — [Hybrid KV Cache Manager](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)
- `prefix_match_unit` 是 prefix-cache key 的最細 token 邊界，可以細於實體 block（文件舉例 32 vs hybrid 的 1024），但每個 group 的 `block_size` 必須能被它整除。未設定時，預設是所有可 prefix-cache 的 KV group block size 的 GCD；單一 group 時就等於該 block size。Mamba `--mamba-cache-mode align` 下 GCD 等於 block size，因此沒有 sub-block 邊界。 — [latest features](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)；[CacheConfig](https://docs.vllm.ai/en/latest/api/vllm/config/cache/)
- 設計頁範例（block size 4，僅說明用）：14 個 prompt token 裡前 10 個與舊請求相同，只命中前 2 個 full block（8 tokens），因為第 3 塊只對上 4 個 token 中的 2 個。 — [latest design](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- v1 的 block table 是 append-only。同一 hash 可能暫時有兩份滿 block（設計頁稱為 duplicated blocks）；v0 會改寫 block table 指回舊 block，v1 不允許，重複要等請求 free 才消除。v0 的改寫行為已過時。 — [latest design](https://docs.vllm.ai/en/latest/design/prefix_caching/)

### Inferences
- 兩筆請求要共享某一塊，至少要同時滿足：hash 演算法與序列化方式相同、從第一個 block 起的 parent chain 相同、該塊 token ids 完全相同且已滿、extra keys 相同（第一塊的 `cache_salt`、LoRA 身份且 v0.31 起含 path、重疊區塊的 multimodal hash、prompt-embeds hash）。temperature、top_p、seed、structured output schema 不在設計頁列出的 hash 元件裡，因此它們本身不會讓「相同 prompt prefix」miss；會讓後續生成 token 分叉，下一輪對話的 prefix 才跟著分叉。
- 第一個 block 有一個 token 不同，後面所有 block 的 parent hash 都變，等於整段 prefix miss。這是 chain hash 的直接結果，不是另外一套規則。
- 官方設計頁仍寫「LoRA IDs」，與 v0.31 changelog 的「依來源 tag + LoRA path」不完全同詞。寫報告時應以 0.31.0 changelog / PR #59335 為 2026-10 的實作，設計頁句子視為概念層說明。

### Gaps
- 未核到 2026-10-14 `main` 上 `generate_block_hash_extra_keys` 的逐行原始碼，因此 extra key 的 tuple 形狀（tag 字串、path 欄位順序）沒有逐欄引用，只引用到 v0.18 行為與 v0.31 changelog。
- `CacheConfig.block_size` 在 latest API 預設是 `None`（由平台決定）。設計頁的 16 與 4 都是例子。沒有找到一份 2026-10 官方文件說「所有後端預設都是 16」，所以不把 16 寫成預設。
- prompt embeds 的 extra key 是「哪些 byte、涵蓋哪些 token」沒有在 2026-10 功能頁寫清；只確認 v0.18 有這個函式，且 `--enable-prompt-embeds` 預設關。 — [Prompt Embedding Inputs](https://docs.vllm.ai/en/latest/features/prompt_embeds/)

## APC 預設開嗎？哪些旗標與隔離 API？

### Takeaway
截至 2026-10-13 的 latest `CacheConfig`，`enable_prefix_caching` 欄位預設是 `True`。CLI 同時有 `--enable-prefix-caching` 與 `--no-enable-prefix-caching`，文件把 CLI 預設寫成 `None`（沒指定就用 config 預設）。V0「預設關閉」已過時。hash 演算法預設 `sha256`。`cache_salt` 是 per-request、opt-in，不是 server flag。

### Cited Findings
- latest `CacheConfig`：`enable_prefix_caching: bool = True`，說明就是 “Whether to enable prefix caching.” `prefix_caching_hash_algo` 預設 `"sha256"`。 — [cache config latest](https://docs.vllm.ai/en/latest/api/vllm/config/cache/)；[config index](https://docs.vllm.ai/en/latest/api/vllm/config/index.html)
- CLI：`--enable-prefix-caching` / `--no-enable-prefix-caching`，說明 “Whether to enable prefix caching.”，Default: `None`。`--prefix-caching-hash-algo` 選項為 `sha256`、`sha256_cbor`、`xxhash`、`xxhash_cbor`，Default: `sha256`。 — [vllm serve CLI](https://docs.vllm.ai/en/latest/cli/serve/)
- 舊 API 文件寫：「Disabled by default for V0. Enabled by default for V1.」這句的 V0 半段已過時；V1 半段與今天的 `bool = True` 一致。 — [舊 config API](https://docs.vllm.ai/api/vllm/config.html)
- vLLM Ascend 文件（2026-10 附近）寫：V1 的 on-chip prefix caching 預設開，除非加 `--no-enable-prefix-caching`。 — [KV Cache Pool Guide（Ascend）](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/KV_Cache_Pool_Guide.html)
- stable 功能頁仍教使用者「Set `enable_prefix_caching=True`」，沒有寫「已經是預設」。這與 latest `CacheConfig` 的預設 True 並陳，不是互相否定的實測。 — [stable features](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)
- `cache_salt` 是請求欄位，不是啟動旗標。OpenAI chat / completions / responses、pooling（embeddings、classification、scoring），以及 Anthropic `/v1/messages` 都接受。不傳則所有請求可互享 prefix。 — [Security（v0.31.0）](https://docs.vllm.ai/en/v0.31.0/usage/security/)
- 其他會改變「hit 落在哪」的旗標：`--prefix-match-unit`、`--mamba-cache-mode align`、`--enable-mamba-shared-prefix-checkpoint`（預設關）、`--mamba-block-size`（只在 prefix caching 開啟時可設，且須為 8 的倍數）。 — [latest features](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)；[CLI](https://docs.vllm.ai/en/latest/cli/serve/)
- `VLLM_SERVER_DEV_MODE=1` 才暴露 `POST` 類的 `/reset_prefix_cache`（以及 `/reset_mm_cache`、`/reset_encoder_cache`）。安全文件把它列為未強制 API key、可打斷服務的開發端點，不應在 production 開。 — [Security](https://docs.vllm.ai/en/v0.31.0/usage/security/)
- 針對 v0.28.0 的第三方操作指南把 hit 指標寫成 token counter：`vllm:prefix_cache_hits_total` 與 `vllm:prefix_cache_queries_total`，區間命中率是兩者增量相除。這不是 vLLM 設計頁的定義。 — [Tensorfuse，對照 v0.28.0](https://tensorfuse.io/docs/guides/inference/prefix-caching)

### Inferences
- 2026-10 的實務開關是「預設開，用 `--no-enable-prefix-caching` 關掉」，不是舊教學文的「預設關、要加 `--enable-prefix-caching`」。功能頁沒改寫，容易讓操作員以為還要顯式打開。
- CLI default `None` 加上 dataclass default `True`，合理讀法是：命令列沒寫時沿用 `CacheConfig` 的 True。沒有在這次研究裡跑一個二進位來印 engine config，所以這是對兩份文件的對讀，不是啟動日誌。

### Gaps
- 沒有查到 `enable_prefix_caching=None` 在 argparse 與 `CacheConfig.__post_init__` 之間的最新原始碼路徑，因此不能引用「None 在哪一行被改成 True」。
- Prometheus 指標的現行名稱以 Tensorfuse 對 v0.28.0 的描述為準；沒有對到 0.31 / latest 的 metrics 文件，0.31 是否改名未知。舊部落格裡的 `vllm:gpu_prefix_cache_hit_rate` 不應再當成現行 gauge。

## 沒人引用的 prefix block 如何被驅逐？referenced 與 cached 差在哪？

### Takeaway
V1 的正常路徑是：`ref_cnt > 0` 的 block 不在 free queue，分配器不會把它拿去給別人。`ref_cnt` 降到 0 才進 free queue，但 hash 還在，所以仍是 cached、可被後來的請求 hit。真正的驅逐發生在分配器從 free queue 頭部取到一塊「仍帶 hash」的 block 時：把它從 hash map 拔掉，之後不能再 hit。順序是 LRU，而且 free 時把請求的 block 依反序放進 queue 尾端，讓較長、較不可能被重用的尾塊先被驅逐。

### Cited Findings
- `KVCacheBlock` 有不可變的 `block_id`、滿塊才指定且驅逐時清掉的 `block_hash`、以及 `ref_cnt`（「The number of requests using this block now」）。free queue 用 block 內的雙向指標，不用另外一條 Python deque。 — [latest design](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- 新請求：`get_computed_blocks()` 對 prompt tokens hash 並查 cache。接著 `allocate_slots()`：(1) 不夠就返回；(2) touch 已算過的 block：`ref_cnt += 1`，若先前沒有別的請求在用，就從 free queue 移除，避免被驅逐；(3) 從 free queue 頭部拿新 block，若頭部是 cached block，同時驅逐，使其他請求不能再重用；(4) 若新分配的 block 已經滿，立刻放進 cache，同一 batch 的其他請求可以重用。 — [同上](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- 已在跑的請求：只走 `allocate_slots()`，設計頁沒有再呼叫 `get_computed_blocks()`。滿塊才 `cache`。 — [同上](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- Free：請求結束時，只有 `ref_cnt` 降到 0 的 block 才釋放。它們以**反序**加到 free queue 尾端。原文：「the last block of a request must hash more tokens and is less likely to be reused by other requests. As a result, it should be evicted first.」仍被其他請求使用的 block 不進 free queue。 — [同上](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- Eviction (LRU) 三步：從 free queue 頭部彈出（文件稱為 least recently used）；從 cache block map 移除 block id；清掉 block hash。 — [同上](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- 設計頁 Time 6：hit 到的 block 先被 touch、移出 free queue，然後才從新的頭部分配；被分配到、原本 cached 的頭部 block 在這一步被驅逐。 — [同上](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- latest `BlockPool` 說明：`free_block_queue` 以驅逐順序存放 free blocks；`cached_block_hash_to_block` 用 hash 找 cached block。`hash_block_size` 是算 hash 的粒度；各 group 實體 block 可以是它的倍數。 — [block_pool latest](https://docs.vllm.ai/en/latest/api/vllm/v1/core/block_pool/)
- 舊版（v0.16）`evict_blocks(block_ids)` 寫明：只處理目前有 hash 的 block；`ref_cnt > 0` 的 block **不會**從 pool 釋放，只從 prefix-cache hash table 移除。這是明示驅逐 API，不是上面的 LRU 分配路徑。2026-10 的 block_pool 頁面擷取沒有再出現這段函式，不能當成 latest 仍公開同一 API。 — [block_pool v0.16.0](https://docs.vllm.ai/en/v0.16.0/api/vllm/v1/core/block_pool/)

### Inferences
- 「referenced」= `ref_cnt > 0`，正在被至少一個請求使用，不在 free queue，正常 LRU 碰不到。「cached」= `block_hash` 仍在 hash map 裡。兩者獨立：可以又 referenced 又 cached（使用中且可被別的請求共享），也可以 cached 但 `ref_cnt == 0`（在 free queue 裡待命，直到被 LRU 摘掉 hash）。hash 被清掉之後就不再是 cached，即使那塊實體記憶體立刻被新請求拿去覆寫。
- 正常分配路徑下，不能在另一個請求 `ref_cnt > 0` 時把該 block 的 KV 搶走。能發生的是：明示把 hash 摘掉（若 v0.16 那種 API 還在）讓新請求 miss，但持有者仍用著同一塊實體記憶體。latest 是否仍保留這個 API 未核實。
- free 的反序是在 LRU queue 上偏置「先丟長 prefix 的尾巴、後丟共享的前段」，不是第二套獨立政策。

### Gaps
- 沒有 latest `block_pool.py` 的 `touch` / `free` 原始碼行號，反序與 LRU 的敘述以設計頁為準。
- v0 的 `LRUEvictor`（`OrderedDict.popitem(last=False)`）是另一套資料結構。設計頁描述的 V1 free queue 才是現行機制；不要把 v0 evictor 寫成 2026 的實作。
- 多租戶下誰的 cached-but-unreferenced block 先死，文件只給反序 + LRU，沒有 per-tenant quota 或 priority。

## 如何隔離租戶？文件警告哪些洩漏與正確性問題？

### Takeaway
預設不隔離：沒有 `cache_salt` 時，任何 token prefix 相同的請求都能互用 KV。隔離方式是每個請求帶一個秘密 `cache_salt`，它被混進**第一個** KV block 的 hash，之後整條 chain 都不同。這擋的是 TTFT 時序側信道（CVE-2025-46570），不是加密。salt 可被猜到就等於沒隔離。另外，multimodal 的 client `uuid` 若衝突，會讓不同媒體共用同一快取身份；`xxhash` 碰撞在理論上可能洩內容或造成未定義行為。

### Cited Findings
- 安全文件：「Prefix caching reuses KV cache blocks across requests that share a common prompt prefix. In multi-tenant deployments, that reuse is a timing side channel (CVE-2025-46570).」攻擊者用 TTFT 差判斷猜的 prefix 是否已在別人的快取裡。文件寫研究顯示這個訊號在 prefix 只有 8 tokens 時幾乎可完美區分（ROC AUC 0.99），並指向論文。 — [Security](https://docs.vllm.ai/en/v0.31.0/usage/security/)；advisory [GHSA-4qjh-9fv9-r85r](https://github.com/vllm-project/vllm/security/advisories/GHSA-4qjh-9fv9-r85r)；論文 [Leaking Secrets from Prefix Caches](https://arxiv.org/html/2411.18191v1)
- `cache_salt` 混進第一個 KV block 的 hash，只有相同 salt 能共享。設計頁同一句：注入第一塊，「only requests with the same salt can reuse cached KV blocks」，目的是阻止用延遲差推論快取內容。 — [Security](https://docs.vllm.ai/en/v0.31.0/usage/security/)；[latest design](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- 選 salt：當成秘密。可猜到（使用者名稱、帳號 ID）就仍能對該租戶做時序攻擊。文件舉例用夠長的亂數（例如 43 個 base64 字元、256 bits）。粒度可以是 per-user（互不推論）或 per-group（同一組織內允許互享）。不傳 salt = 全部請求可互享，只適合單租戶或在乎命中率多於 prefix 隱私。加 salt 會降低命中率，因為只有同 salt 能重用。 — [Security](https://docs.vllm.ai/en/v0.31.0/usage/security/)
- 多模態 part 的可選 `uuid` 會取代「對媒體位元組做 hash」成為快取身份。多租戶下兩個呼叫者對不同媒體給同一個 `uuid`：先到的媒體留下，後到的被靜默丟掉（完整性）；故意重用別人 `uuid` 的人會拿到基於別人媒體的輸出（機密性）。這涵蓋 multimodal processor cache、encoder output cache，以及 prefix cache block hashes。`cache_salt` 是額外防線：即使 uuid 相撞，不同 salt 仍不能共享 prefix KV blocks。`cache_salt`「is opt-in and not passed by default.」 — [Security](https://docs.vllm.ai/en/v0.31.0/usage/security/)
- `xxhash` / `xxhash_cbor` 不是 FIPS 核准演算法；FIPS 環境應選 `sha256` 或 `sha256_cbor`。非密碼學 hash 的碰撞警告見設計頁（未定義行為或在多租戶洩漏私人資訊）。 — [Security](https://docs.vllm.ai/en/v0.31.0/usage/security/)；[latest design](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- 舊設計頁曾寫：「Cache isolation is not supported in engine V0.」V0 無 salt 是過時限制；現行設計頁已描述 salt，且不再把這句當現況。 — [舊 V1 設計鏡像](https://docs.vllm.ai/design/v1/prefix_caching.html)
- GHSA-2823-qmq8-rwvj（轉載，2026-10-06）：vLLM ≤ 0.25.1 對 `cache_salt` 只檢查非空字串。在 LMCache-MP connector 上，含 `@`、`/`、`\`、NUL 或長於 128 字元的 salt 會讓下游 `IPCCacheServerKey` 丟出未被捕獲的 `ValueError`，`EngineCore` 把未捕獲例外當致命錯誤、整個 engine 進程死掉。轉載寫修復版本是 ≥ 0.30.0。這是可用性問題，不是時序洩漏本身。 — [GHSA-2823-qmq8-rwvj 轉載](https://psirt.com/advisory/GHSA-2823-qmq8-rwvj)
- 節點間 KV transfer 預設不安全，安全文件要求放在隔離網路；這是傳輸層，不是 prefix hash 隔離。 — [Security](https://docs.vllm.ai/en/v0.31.0/usage/security/)

### Inferences
- salt 不是加密、也不把 KV 從 GPU 抹掉。它只讓 hash 對不上。同一進程裡，不同 salt 的 KV 仍佔同一塊 GPU prefix cache，只是不能互相 hit。忘記在多租戶每個請求帶 salt，就回到 CVE-2025-46570 的預設行為。
- 用可預測的 user id 當 salt，文件明確說擋不住會猜 salt 的攻擊者。
- LoRA name 重用造成的錯 KV（#59335 修 path）是正確性問題，和 salt 的隱私問題不同：即使開了 salt，同 salt 同 name 不同權重仍可能在修補前共享錯 KV。

### Gaps
- 沒有打開 GHSA-2823-qmq8-rwvj 的 GitHub 原文，修復是否只在 LMCache connector 路徑、0.31 是否仍接受任意字元的 salt，這次沒有用原始碼確認。轉載說升級到 ≥ 0.30.0。
- 文件沒有寫 salt 是否也進入 offload tier 的 key，除了 LMCache-MP 這則 advisory 顯示 salt 會被送進 connector lookup。
- 沒有官方說法主張 `cache_salt` 能防止同一 GPU 上的實體記憶體被其他租戶的進程直接讀取。它只隔離 hash 查找與時序訊號。

## APC 與 chunked prefill、LoRA、speculative decoding、structured outputs / sampling 如何交互？哪些會讓 hit 消失？

### Takeaway
查找發生在請求第一次被排程時，對整段 prompt 的滿 block 做 hash，不是只對第一個 chunk。請求變成 running 之後，設計頁不再重查 prefix cache。因此 chunked prefill 不會把「已經命中的前綴」切碎，但也不會在後續 chunk 裡撿到別的請求剛寫入的新 hit。LoRA 身份（v0.31 起含 path）與 multimodal hash 會讓 token 相同卻 miss。`prompt_logprobs` 會跳過讀快取。speculative decoding（EAGLE/MTP）不改 hash 輸入，但會縮短可用 hit，hybrid 模型上甚至曾造成正確性錯誤或大段重算。structured outputs 與 temperature 這類 sampling 參數不在 hash 元件裡，沒有官方文件說它們會破壞 prompt prefix hit。

### Cited Findings
- 新請求才呼叫 `get_computed_blocks()`，對該請求的 prompt tokens 做 hash。running 請求的路徑只有 `allocate_slots()` 與「滿了就 cache」。同一 batch 裡，剛滿的 block 會立刻進入 cache，供其他請求重用。 — [latest design](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- 功能頁限制：APC 只縮短 prefill，不縮短 decode。答案很長、或新 query 完全不共享 prefix 時沒有收益。原文也說 APC 一般不會降低 vLLM 效能，但沒有附數字。 — [stable features](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)
- 2025-06 的 vLLM 論壇回覆（非官方文件，且寫「as of now」）：V1 開 chunked prefill 時，長 prompt 只有第一個 chunk 走 prefix-cache 查找；之後的 chunk 被當成 running request，scheduler 不再重查。作者認為這是當時 mainline 的設計，不是 bug。這與設計頁「只有新請求呼叫 `get_computed_blocks`」一致，但論壇把「第一次查找」說成「只覆蓋第一個 chunk」；設計頁寫的是 hash「the prompt tokens」，也就是准入時看整段 prompt 的滿 block。兩份材料對「後續 chunk 不再查找」一致，對「第一次查找涵蓋多長」用詞不同。 — [vLLM forum](https://discuss.vllm.ai/t/should-vllm-consider-prefix-caching-when-chunked-prefill-is-enabled/903)
- LoRA：不同 adapter 不得共享 KV。v0.18 用 `lora_name`；v0.31.0 起 hash 含 path，並把 extra key 依來源 tag，避免 name 與 `cache_salt` 撞鍵。同名重指到別的 path 是會錯共享的已知問題，#59335 修的是 path 不同的情況，不是同 path 改檔。 — [kv_cache_utils v0.18.0](https://github.com/vllm-project/vllm/blob/v0.18.0/vllm/v1/core/kv_cache_utils.py)；[0.31.0 notes](https://newreleases.io/project/pypi/vllm/release/0.31.0)；[PR #59335](https://github.com/vllm-project/vllm/pull/59335)
- 動態載入 LoRA 預設關，要 `--enable-lora` 且 `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`。安全文件說這不是安全操作。它會改變「這個名字現在對應哪份權重」，因此和 prefix hash 的 LoRA 身份直接相關。 — [Security](https://docs.vllm.ai/en/v0.31.0/usage/security/)
- `skip_reading_prefix_cache`：2026-06 的 PR #44592 說 `KVCacheManager.get_computed_blocks()` 已有這個 guard；`prompt_logprobs` 或顯式 `skip_reading_prefix_cache=True` 應完整重算，否則跳過的 prefill 不會產生那些 token 的 logprob。該 PR 修的是 `OffloadingConnector` 仍去查 CPU offload。v0.31 changelog 另列「`skip_reading_prefix_cache` honored for connector hits（#57269）」。讀快取被跳過，不表示不寫入；mintlify 寫「不讀但仍寫」，但那是非官方生成站，只當旁證。 — [PR #44592](https://github.com/vllm-project/vllm/pull/44592)；[0.31.0 notes](https://newreleases.io/project/pypi/vllm/release/0.31.0)
- EAGLE/MTP 會在 hit 路徑丟掉最後一個 matched block，讓 drafter 重算最後一段 hidden state。在 dense attention 上這通常是一個較小的 block；在 hybrid GDN/Mamba 上對齊單位可以大到 1648 tokens，等於每次 hit 多算一大段。Issue #53670（2026-08-25）在一個 Qwen3.8 hybrid layout 上量到：無 spec 時 warm lookup hit 31360 tokens、尾端重算 640；dflash K=7 時 hit 29664、尾端重算 2336，c=8 吞吐 206 vs 無 spec 的 327（約 −37%）。關掉 drop 的消融把吞吐拉回 322。這是 issue 作者的量測，不是 vLLM 官方 benchmark。 — [Issue #53670](https://github.com/vllm-project/vllm/issues/53670)
- Hybrid + prefix caching + MTP 曾有正確性 bug：Mamba group 與 full attention 的 hit 長度不一致時，中間 token 沒進 full attention。PR #46281、#45614 描述 Qwen3.5/3.6 這類模型在 APC 與 MTP 同時開啟時輸出偏離 APC-off。功能頁現在的 `--enable-mamba-shared-prefix-checkpoint` 預設關，只在 prefix caching 開、Mamba group 上有 EAGLE/MTP、`--prefix-match-unit` 小於 Mamba block、且非 multi-module MTP 時生效；它在共享 prefix 的接點多存一份 Mamba checkpoint，讓結束在 block 內部的共享 prefix 仍可重用。 — [latest features](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)；[PR #45614](https://github.com/vllm-project/vllm/pull/45614)；[PR #46281](https://github.com/vllm-project/vllm/pull/46281)
- 更早的回歸（PR #33270，2026-01）：gpt-oss + eagle 在 0.14.0 因 hit 長度被減到 0，prefix cache hit rate 變成 0%，作者觀察到約 20% 效能回歸。這是已修的歷史 bug，不應寫成現行必然行為。 — [PR #33270](https://github.com/vllm-project/vllm/pull/33270)
- SWA：官方 DeepSeek-V4.1-Flash blog（2026-10-07）寫，若要在每個可能的 hit 邊界都存 SWA KV，儲存會超過 global KV 的 10 倍。他們的 SWA bounded replay 改為只快取 global KV，hit 長度為 H 時重跑 `[H−128, H)` 重建 SWA。這改變的是 SWA prefix「存什麼」，不是把 APC 關掉。同一篇的約 30% TTFT、以及約 100K throughput 時 TTFT 降近 70%，是 replay 與 kernel 優化合在一起，不能算成 APC 本身的收益。 — [vLLM blog 2026-10-07](https://vllm.ai/blog/2026-10-07-deepseek-v41-flash)
- 設計頁列出的 hash 元件沒有 sampling 參數，也沒有 grammar / JSON schema。structured outputs 作用在 decode 的 logit mask。這次沒有找到官方文件說換 schema 或換 temperature 會讓相同 prompt prefix miss。

### Inferences
- 會破壞或縮短 hit 的條件：token 或 extra key 不同（LoRA、圖片 hash、salt、prompt embeds）、block 沒滿、`prompt_logprobs` / `skip_reading_prefix_cache`、EAGLE 的 last-block drop、hybrid group 交集比單一 group 短、`prefix_match_unit` 與 Mamba align 對不齊。不會因為 chunked prefill 本身就把已命中的前綴作廢。
- 會讓「看起來像同一段文字」仍然 miss 的常見情況：chat template 或工具定義有一個 token 差在前段、圖片 placeholder 相同但 image hash 不同、同名 LoRA 換了 path（0.31 之前）、salt 不同。
- sampling 參數不同仍可共享 prompt KV；它們只影響命中之後新生成的 token，以及下一輪把那些 token 接進 prefix 之後的 hash。

### Gaps
- 沒有官方文件直接寫「structured outputs 與 APC 相容」。上面是 hash 元件清單的負面結論，不是測試結果。
- chunked prefill「後續 chunk 不再查找」之後有沒有在 2026 下半年改掉，設計頁（latest，頁尾日期 2026-06-23，索引抓取日 2026-10-14）仍是舊流程。沒有對到 scheduler 原始碼確認 0.31 是否仍如此。
- `skip_reading_prefix_cache` 在 GPU prefix cache 上是「不讀但仍寫」還是「完全不參與」，官方設計頁沒寫。PR #44592 與 mintlify 支持「不讀」；寫入行為未用原始碼核實。

## 收益數字：只引用論文與官方 benchmark 裡出現的數

### Takeaway
官方 APC 功能頁只有「更高吞吐、更低延遲」，沒有 hit rate 或 TTFT。可以引用的數字來自分開的論文 PrefixBench-H100，以及官方 blog 裡**不是純 APC** 的 KV 傳輸實驗。不要把社群部落格的 TTFT 前後對比寫進報告。

### Cited Findings
- 官方功能頁的兩個工作負載（長文件重複問、多輪對話）只說能避免重複 prefill，帶來「much higher throughput and much lower latency」，沒有數字。並寫明不加速 decode。 — [stable features](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)
- PrefixBench-H100（arXiv:2609.19657，2026-09，第三方論文，不是 vLLM 官方）：8192-token 共享 prefix 時，vLLM p50 TTFT 從無重用的 345.1 ms 降到有重用的 53.0 ms，約 6.5 倍；TensorRT-LLM 從 306.1 ms 到 58.2 ms。prefix 長度 1024 / 2048 / 4096 / 8192 的近似重用比例約 65% / 78% / 87% / 92%。工作集達到容量 1.7 倍時 hit rate 約 47–53%；13.2 倍時 vLLM hit rate 5.6%、TensorRT-LLM 6.6%，p50 TTFT 接近 1.4 秒。表內一列：128 個 prefix、容量比 13.2×，vLLM hit 5.6%、p50 1.421 s。另一組：vLLM KV 未封頂（≥ 1.25M tok）hit 68.2%、p50 0.052 s；封頂 56.3k tok 時 hit 3.6%、p50 0.318 s。 — [arXiv HTML](https://arxiv.org/html/2609.19657)；[alphaXiv 摘要](https://www.alphaxiv.org/abs/2609.19657)
- 同一論文寫輸出長度對 TTFT 幾乎沒影響：三種輸出長度下 p50 TTFT 最多差 1.2 ms（vLLM 39.7–40.9 ms），prefix-cache hit rate 在 TensorRT-LLM 為 0.859，vLLM 上「equivalent」。這支持「APC 不動 decode 長度」的官方定性說法，但是論文設定內的觀察。 — [arXiv HTML](https://arxiv.org/html/2609.19657)
- 官方 blog（2026-10-12）的 1.6× / 3.3× **不是** APC hit 的收益。實驗在 prefill 節點用 `--no-enable-prefix-caching` 關掉本地 prefix cache，改測雙向 KV 傳輸。Qwen3-32B：關傳輸時 TTFT 從 2k 的 158 ms 升到 24k 的 778 ms；開傳輸時從 122 ms 到 460 ms（最長 prompt 約 1.6×）。GLM-5-FP8：關為 2k 的 500 ms 到 20k 的 2334 ms；開為 358 ms 到 702 ms，長 prompt 最多約 3.3× 低於重算路徑。KV 傳輸本身約 30–130 ms（Qwen3-32B）或 29–110 ms（GLM-5-FP8）。 — [vLLM blog 2026-10-12](https://vllm.ai/blog/2026-10-12-bidirectional-kvxfer-multiturn-agentic-workload)
- DeepSeek-V4.1 blog 的「SWA 若按每個 hit 邊界儲存會 >10× global KV」是儲存成本，不是加速比。同篇 TTFT 百分比是多項優化合計。 — [vLLM blog 2026-10-07](https://vllm.ai/blog/2026-10-07-deepseek-v41-flash)

### Inferences
- 論文裡的收益高度依賴「共享 prefix 還放得進 KV 容量」。容量不夠時 hit rate 掉到個位數，TTFT 回到重算的水準。這和 V1「未引用的 cached block 走 LRU」一致：壓力大時 prefix 留不住。
- 官方沒有發布「APC 開啟、hit rate 0% 時吞吐下降不到 1%」這類數字。2024 年第三方比較曾寫 vLLM APC 在無共享 prefix 時吞吐下降約 36.7%，那是 V0 / Python hash 時代的舊量測，不應與 2026 預設 `sha256` + V1 混為一談，所以不列入可引用的現行數字。

### Gaps
- 沒有 vLLM 官方 APC benchmark 的 hit rate 或 TTFT。報告若需要「官方數字」，這項是空的。
- PrefixBench 的硬體、模型與 vLLM 版本要回論文正文才能寫死；這次只用了 HTML 裡的表與摘要句，沒有逐段核對實驗設定。
- Issue #53670 的 −37% 是 hybrid + speculative decoding 的作者量測，不是 APC 一般收益，不應放進「APC 加速」表。

## 2025–2026：KV connector、CPU offload、disaggregated prefill 如何改 prefix KV 的存放與重用

### Takeaway
GPU 上的 hash 表仍是第一層。connector 用同一套 block hash 把已算完的 block 放到 CPU、檔案系統、LMCache 或另一個 vLLM 實例；查找順序是本地 GPU prefix cache，然後外部層，最後才重算。外部命中會把 KV 搬回 GPU。不滿的 block 仍然不是快取單位。`cpu_offload_gb` 是權重卸載，不是 prefix KV。

### Cited Findings
- `OffloadingConnector`「extends the prefix cache by offloading completed KV blocks」到較慢但較大的層（CPU，以及可選的次級層），命中後再提升回 GPU。GPU↔CPU 用 `cudaMemcpyAsync`，與計算重疊。v0.24 指南寫目前支援 CUDA、ROCm、XPU。預設 spec 是 `CPUOffloadingSpec`（pinned host memory）；`TieringOffloadingSpec` 是 CPU 再加次級層，範例 `eviction_policy` 為 `lru`，次級可以是 filesystem 或經 NIXL 的 P2P。範例用 `--kv-transfer-config`，`kv_connector=OffloadingConnector`，`kv_role=kv_both`，`block_size` 64 或 16、`cpu_bytes_to_use` 由使用者指定。connector 的 `block_size` 不必等於模型的 attention block size。 — [KV Offloading Usage Guide v0.24.0](https://docs.vllm.ai/en/v0.24.0/features/kv_offloading_usage/)；較早擷取的 v0.25.1 同題頁面內容一致：[v0.25.1](https://docs.vllm.ai/en/v0.25.1/features/kv_offloading_usage/)
- latest disaggregated prefill 文件把 `OffloadingConnector` 列為 9 種 connector 之一，並寫明 disaggregated prefill **不**提高吞吐。LMCache 有兩條路：進程內 `LMCacheConnectorV1`（NIXL 傳輸範例）與多進程 `LMCacheMPConnector`（獨立 lmcache server，多個 vLLM 實例共享）。`MultiConnector` 可並排 `NixlConnector` 與其他 connector。 — [Disaggregated Prefilling](https://docs.vllm.ai/en/latest/features/disagg_prefill/)；[LMCache examples](https://docs.vllm.ai/en/latest/examples/disaggregated/lmcache/)
- vLLM 的 LMCache 範例頁：進程內模式用環境變數或 `LMCACHE_CONFIG_FILE`；MP 模式是建議的跨實例共享方式。單機 CPU/磁碟卸載有捷徑 `--kv-offloading-backend lmcache` 與 `--kv-offloading-size`（GiB）。 — [LMCache examples](https://docs.vllm.ai/en/latest/examples/disaggregated/lmcache/)
- LMCache MP 的 disaggregated prefill：prefill 路徑的 KV 來源順序是 local GPU cache → LMCache → recompute。`NixlConnector` 負責這一次請求的 prefill→decode 交接；`LMCacheMPConnector` 負責卸載/載入。文件寫 vllm-project/vllm#46865 之前，`MultiConnector` 下 offload 會靜默不觸發。 — [LMCache MP disaggregated prefill](https://docs.lmcache.ai/mp/disaggregated_prefill.html)
- LMCache lazy offload 要求 `enable_prefix_caching=True`，因為驅逐偵測讀的是 block hash；prefix caching 關了就沒有這些 hash。預設關。`EVICTION_AWARE` 在 GPU block 接近驅逐頭時才送出 store。 — [LMCache lazy offload](https://docs.lmcache.ai/mp/lazy_offload.html)
- latest `SimpleCPUOffloadConnector`：若發現 prefix caching 關閉，會停用 CPU offload，log 寫「it requires prefix caching」。它自述是 best-effort cache，丟掉一次 save 只是下次 miss。驅逐沿用 BlockPool LRU。 — [simple_cpu_offload_connector](https://docs.vllm.ai/en/latest/api/vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector/)
- latest `OffloadingConnector`：不會對「自己沒卸載、而該 group 又 prefix-cacheable」的範圍提供 token；超過那個 group 自己已快取的 prefix 就會留下沒寫入的 KV。hybrid 若 `tokens_per_block` 不能被 hash 粒度整除，錯誤訊息要求開 `--enable-prefix-caching` 來對齊。 — [offloading_connector](https://docs.vllm.ai/en/latest/api/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector/)；[offloading config](https://docs.vllm.ai/en/latest/api/vllm/distributed/kv_transfer/kv_connector/v1/offloading/config/)
- Offloading scheduler：GPU prefix caching 開啟時，用一個集合記住已載入的 chunk，避免重複 load。 — [offloading scheduler（stable）](https://docs.vllm.ai/en/stable/api/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler/)
- Ascend：NPU prefix miss 但 CPU 有資料時，非同步載回 NPU。文件說這是建在上游 `OffloadingConnector` 上，`CPUOffloadingSpec` / `TieringOffloadingSpec` 註冊成 NPU 實作。MooncakeStore 是另一條 KV pool：可當 LMCache 的遠端後端，或用 `MooncakeStoreConnector`；on-chip prefix caching 仍是 V1 預設，兩者並用。 — [Ascend CPU offload](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/kv_cache_cpu_offload.html)；[Ascend KV pool](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/KV_Cache_Pool_Guide.html)
- 雙向 KV 傳輸（官方 blog，2026-10-12）：多輪時若 prefill 節點已經把本地 prefix 趕走，或 router 選到沒有這段 KV 的 prefill，就從 decode 節點把保留的 KV 送回，而不是整段重算。實驗故意在 prefill 加 `--no-enable-prefix-caching`。數字見上一節。傳輸層在該實驗是 AWS EFA。 — [vLLM blog 2026-10-12](https://vllm.ai/blog/2026-10-12-bidirectional-kvxfer-multiturn-agentic-workload)
- v0.31.0 另有 `release_kv_cache_memory()` 與 `POST /release_kv_cache_memory`（#44890）。changelog 沒有寫它與 prefix hash 的交互細節。 — [0.31.0 notes](https://newreleases.io/project/pypi/vllm/release/0.31.0)
- CLI 的 `--cpu-offload-gb` 是把模型權重放到 CPU、每步再搬上 GPU。它不是 prefix KV offload。 — [CLI serve](https://docs.vllm.ai/en/latest/cli/serve/)

### Inferences
- 外部層重用的是「已經算完、且 hash 對得上」的 full block，不是一套新的前綴演算法。GPU LRU 把一塊摘掉 hash 之後，CPU/LMCache 仍可能有同一 hash；下一次請求先 miss GPU，再從 connector 載回。lazy offload 反過來說：沒有 GPU prefix hash，就無法知道哪一塊快被驅逐、該送去外層。
- 跨節點重用（NIXL、LMCache MP、decode→prefill 回傳）讓 prefix KV 離開單一進程的 GPU pool。安全文件仍把節點間 KV 傳輸視為預設不加密，要靠網路隔離，而不是靠 `cache_salt`。
- `cache_salt`、LoRA path、multimodal hash 若有進 block hash，外部 store 也會跟著分鍵；0.31 改了 extra key 之後，舊的外部 KV 會全部 miss。PR #59335 自己也說 connector 與 offload tier 會自動吃到新 hash，因此升級會使舊快取失效。

### Gaps
- 沒有對到 2026-10-14 的單一一份「KV Offloading Usage Guide」latest 頁；使用細節引用的是 v0.24/v0.25 指南，API 行為引用的是 latest connector 頁。兩者之間的預設 `block_size`、`eviction_policy` 是否仍是文件範例那些值，未再核對。
- Mooncake 在上游 CUDA vLLM 與 Ascend 插件的支援矩陣沒有逐項核對。這裡只記錄 Ascend 文件寫的接法。
- NixlConnector 本身是「這一筆請求的 prefill KV 交給 decode」，不是跨使用者的 prefix 快取。只有它和 LMCache / offload 疊在一起時，才同時存在「跨請求重用」與「跨節點搬運」。文件沒有把這兩個生命週期畫在同一張圖裡。
