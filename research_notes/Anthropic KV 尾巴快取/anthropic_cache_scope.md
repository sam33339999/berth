# Anthropic 多租戶 prompt cache 的範圍、淘汰與隔離

擷取日：2026-10-14。範圍是服務端／多租戶合約：誰的請求能命中誰寫入的 prefix、TTL 與淘汰保證、失效條件，以及 2025–2026 改變命中行為的公開機制。請求級 `cache_control` 語法與「tail cache」用語不在本文展開。未公開的 GPU scheduler、分頁塊大小、radix 樹，文件沒寫的就不重建。

現行合約以 2026-10-14 抓到的英文 Prompt caching 頁為準。`docs.claude.com` 仍停在 2024 年底到 2025 年初的敘述（只列到約 Claude Opus 4.1／Haiku 3、只有組織級隔離），不能拿來描述現況。

## 誰能對同一前綴產生 cache hit？

### Takeaway

第一方合約不是「同一支 API key」，也不是「任何使用該模型的客戶」。命中條件是：隔離邊界內、prompt 到 breakpoint 的 cryptographic hash 完全相同、同一模型，而且讀取請求有參加 caching。Claude API、Claude Platform on AWS、Microsoft Foundry 的邊界是 workspace；Amazon Bedrock 與 Google Cloud 在 Anthropic 文件裡仍是 organization。跨 organization 永不共享。

### Cited Findings

- 官方隱私／範圍原句（Prompt caching，Cache storage and sharing，2026-10-14 抓取）：「Caches are isolated between organizations. Different organizations never share caches, even if they use identical prompts. Caches are also isolated per workspace within an organization on the Claude API, Claude Platform on AWS, and Microsoft Foundry; Bedrock and Google Cloud use organization-level isolation only.」同頁警告框把這條寫成 workspace-level isolation，並點名後兩個平台維持 organization-level。FAQ 再寫一次：「On the Claude API, Claude Platform on AWS, and Microsoft Foundry, caches are isolated per workspace within an organization. On Bedrock and Google Cloud, caches are isolated per organization. In every case, caches are never shared across organizations, even for identical prompts.」— [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 同一 FAQ 的 key 原句：「Cache keys are generated using a cryptographic hash of the prompts up to the cache control point. This means only requests with identical prompts can access a specific cache.」Exact matching：「Cache hits require 100% identical prompt segments, including all text and images up to and including the block marked with cache control.」— [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- Workspaces 頁對齊同一邊界，並寫明 API key 預設綁在單一 workspace：「Prompt caches are also isolated per workspace on the Claude API, Claude Platform on AWS, and Microsoft Foundry. On Amazon Bedrock and Google Cloud, prompt caches are isolated per organization.」— [Workspaces](https://platform.claude.com/docs/en/manage-claude/workspaces)
- 法文文件仍標日期：「À compter du 5 février 2026，les caches sont également isolés par espace de travail…」搜尋索引裡的英文頁也曾出現 “As of February 5, 2026…”。2026-10-14 直接抓到的英文正文警告框沒有這句日期。2026-02-05 的 release notes 當天條目是 Claude Opus 4.6、effort、compaction、data residency，沒有 cache isolation。— [法文 Prompt caching](https://platform.claude.com/docs/fr/build-with-claude/prompt-caching)；[Release notes](https://platform.claude.com/docs/en/release-notes/overview)
- 2024 年底到 2025-02 的舊約是組織級、沒有 workspace。Wayback 2025-02-25：「Organization Isolation: Caches are isolated between organizations. Different organizations never share caches, even if they use identical prompts.」並要求讀寫時「the same block must be marked with cache_control」。`docs.claude.com` 現行殘頁仍寫：「Caches are organization-specific. Users within the same organization can access the same cache if they use identical prompts, but caches are not shared across different organizations.」— [Wayback 2025-02-25](https://web.archive.org/web/20250225142138/https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)；[docs.claude.com 殘頁](https://docs.claude.com/en/docs/build-with-claude/prompt-caching)
- Cache diagnostics 的 fingerprint「are scoped to your organization and workspace」；「The previous request must have been made with an API key from the same organization and workspace。」此功能「Claude API only: Not available on Amazon Bedrock or Google Cloud.」— [Cache diagnostics](https://platform.claude.com/docs/en/build-with-claude/cache-diagnostics)
- Google 自己的 Claude prompt caching 頁（頁面標 2026-10-13）寫的是 project，不是 Anthropic 的 organization：「Caches are unique to your Google Cloud project and cannot be used by other projects.」並說 Claude 只對有開 caching 的請求算 hash；Google 把這些 hash 當 User Metadata／Service Data，不當 Cloud DPA 下的 Customer Data，「Google does not use these hashes for any other purpose.」可向支援要求按 project 關掉 explicit caching，關掉後帶 caching 的請求會被拒絕。— [Google Cloud Claude prompt caching](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/claude/prompt-caching)
- AWS 產品頁：「Your cache is specific to your account and comprises the internal model state representing your prompts.」這是 account，不是 Anthropic 文件裡的 organization，也不是 workspace。使用者指南確認 Bedrock 上的 Claude 同時有 Implicit（不放 cache control 也嘗試重用合格 prefix）與 Explicit，但該頁擷取內容沒有重述隔離主體。— [Amazon Bedrock prompt caching（產品頁）](https://aws.amazon.com/bedrock/prompt-caching/)；[Bedrock user guide](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)
- 第一方 API 是 opt-in。沒有 top-level 或 block 級 `cache_control` 時「nothing is cached」，每個請求都付全額 input。Bedrock 的 implicit caching 是 AWS 文件另寫的行為，不適用這句。— [Mid-conversation system messages](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages)
- Cache 依模型分開。Cache diagnostics 的 `model_changed`：「The cache is per-model.」Anthropic 2026-09-08 文章：「the prompt cache is pinned to a specific model」且「prompt cache reads must be byte-exact across the full span of the prompt.」— [Cache diagnostics](https://platform.claude.com/docs/en/build-with-claude/cache-diagnostics)；[Reducing cost and improving performance](https://claude.com/resources/articles/reducing-cost-and-improving-performance-with-claude-platform)
- 2026-08-27：personal key 與 service account key 可以綁單一 workspace，或「work on admin endpoints and across any workspace」。Workspace API key 仍是 legacy。文件沒有說 cache 改成 per-key。— [Release notes，2026-08-27](https://platform.claude.com/docs/en/release-notes/overview)

### Inferences

- 同一 workspace 內，不同 API key、不同使用者，只要前綴 hash、模型與會被寫進 prompt 的參數一致，就可以互打。隔離單位不是 key。
- 跨 workspace 的 key 若能選 workspace，合理預期打到的是該次請求所屬 workspace 的 cache，而不是一把 key 共用的池。這是從「per workspace」推出的，文件沒有單獨寫 cross-workspace key 的 cache 路由。
- Google「project」與 Anthropic「organization」、AWS「account」與 Anthropic「organization」沒有被對成同一個 ID。在 Bedrock／Google Cloud 上，不能假設 Anthropic workspace 仍是邊界；Anthropic 明文說這兩個平台沒有 workspace 隔離。

### Gaps

- Release notes 沒有獨立條目記載 2026-02-05 的 workspace 隔離切換。日期只出現在部分語系／搜尋索引，英文現頁警告框未附日期。切換是否分批、舊 organization 級條目何時清空，沒有寫。
- 沒有公開句子說 cache 也依 region、`inference_geo` 或資料落地切開。Data residency（2026-02-05 上線）與 cache 放置的關係未說明。
- Google project 與 Anthropic organization 衝突時，以哪一個為準，兩邊文件都沒有對照表。
- AWS user guide 擷取段沒有「account」或「organization」隔離句；account 只出現在產品頁。Implicit cache 是否與 explicit cache 同一隔離域，沒有寫。

## 使用者 A 的 cache write 會不會降低使用者 B 的延遲或費用？

### Takeaway

會，但只在上一節的隔離邊界內，而且 B 的請求必須自己帶 caching、前綴 byte-exact、模型相同、條目還在最小 TTL 內。A 付 write 溢價，B 付自己的 cache-read 價並少做 prefill。跨 organization 不行。跨 workspace 只在 Bedrock／Google Cloud 的 organization 級合約下才可能，而且跟 Google 的 project 句衝突。

### Cited Findings

- 機制是 prefix 重用，不是把 A 的回應複製給 B：「If found, it uses the cached version, reducing processing time and costs. Otherwise, it processes the full prompt and caches the prefix once the response begins.」「Prompt caching has no effect on output token generation. The response you receive is identical to what you would get if prompt caching were not used.」— [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 計費拆開。5 分鐘 write 是 base input 的 1.25 倍，1 小時 write 是 2 倍，read 一般是 0.1 倍；Claude Opus 5.5 與 Claude Sonnet 5.5 的 read 是 0.05 倍，Claude Fable 5.1 與 Claude Mythos 5.1 是 0.025 倍。命中刷新不另收費。— [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 「cache hits are not deducted against your rate limit.」這句寫在 1 小時 cache 的使用時機，沒有另設「只對寫入者本人」的但書。— [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 並行：條目要等第一個 response 開始才可供後續請求命中。「If you need cache hits for parallel requests, wait for the first response before sending subsequent requests.」— [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- Message Batches：「because asynchronous batch requests can be processed concurrently and in any order, cache hits are provided on a best-effort basis.」— [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 2026-09-08 文章把節省描述成讀回 KV state，並把共享條件寫成 byte-identical、同一模型、同一 effort：「Subagents and branches only share the parent’s cache when the fork’s prefix is byte-identical, on the same model, and using the same effort.」— [Reducing cost and improving performance](https://claude.com/resources/articles/reducing-cost-and-improving-performance-with-claude-platform)
- Claude Code 部落格的 “globally cached” 是產品 harness 的分段標籤（靜態 system prompt 與 tools 放最前面），不是「全體 API 客戶共用一塊 KV」。同文強調 prefix match，任一前綴變動就讓後面失效。頁面日期 April 30, 2026。— [Lessons from building Claude Code](https://claude.com/blog/lessons-from-building-claude-code-prompt-caching-is-everything)
- 舊組織級句子仍在 `docs.claude.com`：「Users within the same organization can access the same cache if they use identical prompts.」這句在現行 Claude API 上被 workspace 句子收窄。— [docs.claude.com 殘頁](https://docs.claude.com/en/docs/build-with-claude/prompt-caching)

### Inferences

- A 的 write 不會改 B 的輸出分布；只讓 B 少付 input、少等 prefill。A 自己的那一次通常更貴（1.25x 或 2x），收益在邊界內之後的 hit。
- 同一 workspace 裡，A 與 B 不必同一對話、不必同一 key。相同的工具定義加 system prompt 就足以讓 B 讀到 A 寫入的前綴。這是 isolation 句加上 exact hash 的直接結果。
- 不同 organization 的相同 system prompt 不會互相省錢。把「靜態 system prompt 全球快取」讀成跨客戶共享，超出第一方合約。

### Gaps

- 沒有第一方數字說明同一 workspace 內跨使用者命中的實際比例，或這類共享是否被容量政策故意拆開。
- 沒有寫 B 的 hit 是否計入 A 的用量，或 read 費用是否可能出在寫入者帳單。公開計費欄位在每個 response 的 `usage` 上，看起來是請求者自付，但沒有跨主體計費的明文。
- Bedrock implicit caching 是否讓「沒放 marker 的 B」吃到「有放 marker 的 A」所寫的條目，user guide 擷取段沒有說。

## 淘汰與容量：LRU、上限，還是只有 TTL？

### Takeaway

公開合約是最小 TTL，不是 LRU，也沒有最大 cache 容量。預設至少 5 分鐘，`ttl: "1h"` 至少 1 小時；每次命中免費刷新。過期後「迅速但非立即」刪除。沒有手動清除。保證的是最短存活；容量壓力能否提前趕走條目，文件沉默。Batches 的命中被明文寫成 best-effort。

### Cited Findings

- 「By default, the cache has a 5-minute lifetime. The cache is refreshed for no additional cost each time the cached content is used.」計時「from the start of the request that writes or reads the cache entry, not from the end of its response.」若串流生成花 4 分鐘，5 分鐘條目只剩約 1 分鐘可給下一個請求。— [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- Data retention：「Cached entries have a minimum lifetime of 5 minutes (standard) or 1 hour (extended), after which they are promptly, though not immediately, deleted.」存放物是「KV (key-value) cache representations and cryptographic hashes」，「held in memory only and are not stored at rest.」— [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- FAQ：「The cache's default minimum lifetime (TTL) is 5 minutes.」「Currently, there's no way to manually clear the cache. Cached prefixes automatically expire after a minimum of 5 minutes of inactivity.」— [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 目前唯一 cache type 是 `ephemeral`。1 小時與 5 分鐘可混用，但較長 TTL 的 breakpoint 必須出現在較短 TTL 之前。— [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 範例程式註解把「evicted」用在「not being used for more than 5 minutes」，沒有另給 LRU 或容量閾值。— [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 5 分鐘與 1 小時「behave the same with respect to latency.」TTL 換的是存活與 write 單價，不是另一種更快的讀取路徑。— [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 低於模型最小長度會靜默不快取，不報錯：`cache_creation_input_tokens` 與 `cache_read_input_tokens` 都是 0。這不是淘汰，是從未寫入。現行門檻（Claude API、Claude Platform on AWS、Google Cloud、Microsoft Foundry；Bedrock 另見 AWS 表）：512（Fable 5.1、Mythos 5.1、Opus 5.5、Opus 5、Sonnet 5.5、Fable 5、Mythos 5、Haiku 5.5）、1024（Opus 4.8、Sonnet 5、Sonnet 4.6、Sonnet 4.5、以及若干已退役仍留在 Bedrock／Google Cloud 的型號）、2048（Mythos Preview、Opus 4.7、Haiku 3.5）、4096（Opus 4.6、Opus 4.5、Haiku 4.5）。— [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 寫入只發生在 breakpoint。Lookback 最多往回看 20 個 block（breakpoint 自己算第一個）；Claude API 上連續 `tool_use` 或連續 `tool_result` 各算一個位置。找不到就停止，不會把穩定前綴自動補寫進去。— [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)

### Inferences

- 「minimum lifetime」加上「not immediately deleted」表示 5 分鐘／1 小時是地板，不是準點刪除的上限。文件沒有承諾最長只活這麼久。
- 範例裡的 evict 指的是閒置超過 TTL，不能讀成已公開 LRU。
- 容量不足時是否提前淘汰，沒有保證也沒有否認。把「最小 TTL」說成「記憶體壓力下仍保證活滿」，超出原文。

### Gaps

- 沒有最大條目數、最大 token、每 workspace 配額、每模型池大小。
- 沒有 LRU、LFU、TTL-only 或隨機淘汰的政策名。
- 沒有寫過期刪除的延遲上界（「not immediately」沒有秒數）。
- 沒有寫同一前綴同時有 5 分鐘與 1 小時條目時，命中選哪一筆，除了混用 TTL 的計費位置 A/B/C。

## 哪些變更會使 cache 失效？

### Takeaway

失效是前綴階層，不是整包一律重算。順序固定為 `tools` → `system` → `messages`。某一層改了，該層與之後的層失效，更早的 breakpoint 仍可命中。官方表列了工具 schema、citations、`tool_choice`、圖片有無、thinking、effort；沒有把 `temperature` 列進表。模型本身是另一個 cache 命名空間。

### Cited Findings

- 階層與表（✘ 為失效，✓ 為仍有效），2026-10-14 Prompt caching「What invalidates the cache」：
  - 工具定義（名稱、描述、parameters）改動：tools、system、messages 全失效。
  - Web search 開關、citations 開關、`speed: "fast"` 與標準速度切換：tools 仍在，system 與 messages 失效。Citations 與 web search 是因為它們改寫 system prompt。
  - `tool_choice`：只失效 messages。
  - 在 prompt 任何位置加入或移除圖片：只失效 messages。圖片內容本身也在 exact match 裡（「all text and images」）。
  - Thinking 設定（mode，以及 extended 的 `budget_tokens`）會被 render 進 prompt，messages 一定失效；若該模型把設定畫在 tools／system 之前，那兩層也失效。
  - `output_config.effort` 與 thinking 相同；明確設成該模型預設值等同省略，不失效。支援 per-message effort 的模型若把 effort 改動放在 `messages` 裡的 `role: "system"`，已快取前綴維持有效。
  - 帶 thinking 時送入非 tool-result 的 user 內容：tools／system 仍在。Opus 4.5+、Sonnet 4.6+、Haiku 5.5 預設保留 thinking，messages 仍有效。更早的 Opus／Sonnet，以及到 Haiku 4.5 為止的 Haiku，會剝掉先前 thinking blocks，其後 messages 從 cache 移除。
  - 收到方丟掉讀不了的 thinking block（文件點名 Fable 5.1、Mythos 5.1、Opus 5.5、Sonnet 5.5、Haiku 5.5）：從該 block 起 messages 前綴改變。原樣回傳、模型讀得懂的 block 則保持。
  — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 工具頁另外寫：`disable_parallel_tool_use` 改動只失效 messages。`defer_loading` 的工具不進 system 前綴；工具搜尋把定義以 `tool_reference` 追加，前綴維持。Server tool（web search、web fetch、code execution）在請求已有 caching 時，會在 server tool result 上自動放一個 5 分鐘 breakpoint，即使呼叫者設的是 1 小時。— [Tool use with prompt caching](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-use-with-prompt-caching)
- Thinking blocks 不能自己掛 `cache_control`，但出現在先前 assistant turn 時會隨其他內容被快取，讀取時算 input tokens。Signature 沒有被單獨列成 cache key 欄位；要保持命中必須把 block 原樣送回。— [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 2026-09-28：Claude Sonnet 5.5 的 thinking blocks「work only in the account that produced them, or in an account linked to it.」別的 account 送回時，API 在模型看到之前丟掉該 block，請求仍成功。這會落到上面的 dropped-thinking 規則。— [Release notes，2026-09-28](https://platform.claude.com/docs/en/release-notes/overview)
- Citations 的子 block 不能直接標 `cache_control`；要快取的是上層 document block。空 text block 不能快取。— [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 改 top-level `system` 會改 hash，system 與之後的 messages 都 miss。中途指令應追加 `{"role": "system"}`，讓已快取前綴不變。改或刪已送出的 mid-conversation system message，會從該點往後失效。— [Mid-conversation system messages](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages)
- 改 `tools` 會讓整段對話的 cache 失效。`inline-tools-2026-09-15` 允許把新工具放進 mid-conversation system message 的 `tool_addition`，前提是 `tools` 陣列與第一次完全相同。例外：`tools` 裡沒有任何非 deferred 工具時，第一次用這種方式定義工具會整段 miss。— [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)；[Release notes，2026-09-22](https://platform.claude.com/docs/en/release-notes/overview)
- Breakpoint 本身：寫入只發生在標了 `cache_control` 的位置。前綴任一 block 變動，該 breakpoint 的累積 hash 就不同。Lookback 只找先前請求已經寫過的位置，不把沒標過的穩定內容補進 cache。2025-01-15 起，設一個 breakpoint 會自動讀最長的先前已快取前綴；這取代了 2025-02 Wayback 仍寫的「讀寫必須標在同一 block」。— [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)；[Release notes，2025-01-15](https://platform.claude.com/docs/en/release-notes/overview)
- `tool_use` 的 JSON key 順序不穩定（文件點名 Swift、Go）會破壞 cache。System 用字串或 typed block 陣列是不同形狀；這點沒有寫在失效表，但 hash 覆蓋到 breakpoint 的整個前綴。— [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 故障排查清單要求各次呼叫的 `tool_choice`、圖片使用、thinking 設定、`output_config.effort` 保持一致，並確認 breakpoint 位置相同。— [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- Cache diagnostics（Claude API）只回報最早分歧：`model_changed`、`system_changed`、`tools_changed`（含增刪、重排、`input_schema` JSON 非決定性序列化）、`messages_changed`（改、重排、刪除較早訊息，或 assistant／`tool_result` 重送時序列化不同）。`previous_message_not_found` 不是「內容變了」的證據。比較的是請求結構，與該次是否真的 hit 無關。— [Cache diagnostics](https://platform.claude.com/docs/en/build-with-claude/cache-diagnostics)
- Beta header 沒有被列成獨立失效列。Prompt caching 自 2024-12-17 起不再需要 beta header。1 小時 TTL 自 2025-08-13 起不再需要 beta header。會改寫 prompt 的 header（例如 inline tools、per-message effort）是透過內容變動失效，不是 header 字串本身有一條公開規則。— [Release notes](https://platform.claude.com/docs/en/release-notes/overview)；[Prompt caching FAQ](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)

### Inferences

- 失效表是「哪些頂層參數會被 render 進前綴」的清單，不是全部 API 欄位的封閉集合。表上沒有的欄位不能反推「一定不影響 hash」。
- 改 breakpoint 位置若不改內容，仍可能 miss：新位置若沒有先前寫入，lookback 20 格內又碰不到舊寫入，就會重寫。這是 miss，不是 TTL 淘汰。
- 圖片「有無」失效 messages，是表上的結構規則；圖片像素不同則由 exact match 處理。兩條都要成立才會 hit。

### Gaps

- `temperature`、`top_p`、`top_k`、`max_tokens` 不在失效表，也沒有句子說它們在或不在 cache key。部分新模型直接拒絕非預設取樣參數（例如 2026-05-28 的 Opus 4.8），所以那些模型上這題無實務差異。
- Beta header 字串本身是否進入 hash，沒有寫。只送 `cache-diagnosis-2026-04-07` 與否，文件沒有說會不會拆 cache。
- Thinking block 的 `signature` 位元組若改一個字元，是否算 messages 前綴變動：只能從「byte-exact／原樣回傳」推論，沒有單獨句子。
- 文件圖像、PDF、document block 的內部正規化（空白、media type、cache control 從子 block 提升到父 block）對 hash 的影響，除了 2025-05-01「cache control 必須在 `tool_result` 與 `document.source` 的父 content block」之外，沒有完整清單。

## 2025–2026 改變命中行為的機制

### Takeaway

這段期間公開、而且會改「誰命中、命中多久、如何解釋 miss」的項目是：1 小時 TTL、breakpoint 自動回讀、workspace 隔離、automatic caching、cache diagnostics、以及 ZDR 下仍把 KV 與 hash 留在記憶體直到 TTL。沒有找到名為 cache priority 的 prompt-cache 參數。`service_tier: "priority"` 是服務層級，不是 cache 淘汰優先權。

### Cited Findings

- 1 小時 TTL：Anthropic 官方貼文 2025-05-22 宣布，在 5 分鐘之外提供 1 小時，並重述長 prompt「costs by up to 90%」「latency by up to 85%」。Release notes 2025-08-13：「The 1-hour cache duration for prompt caching no longer requires a beta header.」現行 schema 只有 `ttl: "5m" | "1h"`，預設 `5m`。— [Anthropic 貼文](https://x.com/AnthropicAI/status/1925633128174899453)；[Release notes，2025-08-13](https://platform.claude.com/docs/en/release-notes/overview)；[Messages API](https://platform.claude.com/docs/en/api/messages)
- 有人主張 2026-03-06 起預設 TTL 從 1 小時被默默改回 5 分鐘。第一方時間線相反：預設一直是 5 分鐘，1 小時從 2025-05 起是 opt-in。該文的 `ttl: 3600` 也與現行 `"5m"`／`"1h"` 字串不符。— 與 [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) 及 2025-05-22 貼文矛盾；不採該則次級來源。
- 2025-01-15：設 breakpoint 後自動讀最長的先前已快取前綴。2025-01-10：Batches 的 caching 支援被優化以提高命中率，沒有公布前後數字。2025-03-13 文章：Claude 3.7 Sonnet 上 cache read 不再計入 ITPM；並重複「自動讀最長前綴」。現行文件把「cache hits are not deducted against your rate limit」寫成一般規則，不再限 3.7。— [Release notes](https://platform.claude.com/docs/en/release-notes/overview)；[Token-saving updates，2025-03-13](https://claude.com/blog/token-saving-updates)
- Automatic caching：2026-02-19 上線，當時寫「Available on the Claude API and Microsoft Foundry (preview)」。2026-10-14 文件改成除了 legacy Amazon Bedrock（Opus 4.6 與更早）以外的平台都可用；該 legacy 整合對 top-level `cache_control` 回 400，須改用 explicit breakpoint。自動 breakpoint 佔 4 個名額之一；最後一個 block 已有不同 TTL 會 400；沒有合格 block 則靜默跳過。— [Release notes，2026-02-19](https://platform.claude.com/docs/en/release-notes/overview)；[Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- Workspace 隔離見第一節。2026-02-05 這個日期出現在法文頁與搜尋索引，沒有出現在當天 release notes。
- Cache diagnostics：2026-05-13 public beta，header `cache-diagnosis-2026-04-07`。2026-09-09 起只有請求帶 `diagnostics` 物件才存 fingerprint。2026-09-23 脫離 beta，不再需要該 header；沒帶 `diagnostics` 時欄位為 `null`。只在 Claude API。Fingerprint 只有 hash 與 token 估計，不含原文，範圍是 organization 加 workspace，保留時間有限。`previous_message_not_found` 的常見原因包含沒存 fingerprint、不同 workspace，或隔太久。— [Release notes](https://platform.claude.com/docs/en/release-notes/overview)；[Cache diagnostics](https://platform.claude.com/docs/en/build-with-claude/cache-diagnostics)
- ZDR，以 2026-10-14 Prompt caching 的 Data retention 為準：「Prompt caching (both automatic and explicit) is ZDR eligible. Anthropic does not store the raw text of your prompts or Claude's responses.」KV 與 cryptographic hash 只在記憶體、不落盤，活滿最小 TTL 後迅速但非立即刪除，並沿用 organization／workspace 隔離。同頁指向 API and data retention。搜尋索引中的該表寫：prompts 與 outputs 不存，KV 與 hash「held in memory for the cache TTL and promptly deleted after expiry.」— [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)；[API and data retention](https://platform.claude.com/docs/en/manage-claude/api-and-data-retention)
- 另一頁 Zero Data Retention 的搜尋摘錄仍把 prompt caching 稱為 beta，並說「may be suitable」給需要 ZDR 型承諾的客戶，因為不存原文、條目至少 5 或 60 分鐘、組織間隔離。用詞舊於現頁的「ZDR eligible」與 workspace 隔離。以現頁為準，舊頁可能未更新。— [Zero Data Retention](https://platform.claude.com/docs/en/build-with-claude/zero-data-retention)
- 模型級 ZDR 例外不是 cache 功能例外：2026-09-01，Claude Fable 5.1 與 Claude Mythos 5.1「require 30-day data retention and aren't available under zero data retention unless expressly authorized」。Claude Fable 5 更早也有同樣限制。— [Release notes，2026-09-01](https://platform.claude.com/docs/en/release-notes/overview)
- 沒有 cache priority 欄位。Messages API 的 `cache_control` 只有 `type: "ephemeral"` 與可選 `ttl`。`service_tier` 的 `"priority"` 是請求服務層，文件沒有把它連到 cache 命中或淘汰。— [Messages API](https://platform.claude.com/docs/en/api/messages)
- 其他會改命中、但不是獨立 cache 產品的行為：2026-05-28 Opus 4.8 最小可快取長度降到 1024；2026-09-01 起 per-message effort 與 turn-scoped system message（`clear_at: "next_user_message"`）用來在改指令時保住前綴；2026-09-22 inline tools。Claude Code 客戶端文件（頁面日期 2026-08-14）說訂閱上 cache 約一小時、API key 上約五分鐘，可用 `ENABLE_PROMPT_CACHING_1H=1` 改成一小時。這是該客戶端選 TTL，不是另一套多租戶隔離。— [Release notes](https://platform.claude.com/docs/en/release-notes/overview)；[Maximizing the value of your Claude Code sessions](https://claude.com/resources/articles/maximizing-the-value-of-your-claude-code-sessions)
- Pre-warming：`max_tokens: 0` 會把 prompt 讀進模型、在 breakpoint 寫 cache、不生成輸出。Placeholder 使用者訊息不能當 breakpoint，否則正式請求對不上。Thinking 與 `output_config.effort` 必須與正式流量相同。Release notes 擷取範圍內沒有這項的上線日。— [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)

### Inferences

- ZDR 沒有關掉 cache。它承諾的是原文不落盤；KV 與 hash 仍在記憶體裡活一個最小 TTL，而且同一 workspace（或 Bedrock／Google Cloud 上的 organization）內的相同前綴仍可讀到。ZDR 不是 per-request 隔離。
- Automatic caching 不改變隔離域或 TTL，只改變 breakpoint 放哪。放在每請求都變的最後一個 block 上時，命中率會變差，這是文件自己舉的陷阱。
- Cache diagnostics 解釋的是「這次請求結構和上次哪裡不同」，不是「條目被 TTL 或容量丟掉」。結構相同仍可能因過期而 `cache_read_input_tokens = 0`。

### Gaps

- 「cache priority」在 platform docs、release notes 與 `cache_control` schema 都沒找到。若它是未公開的內部排程參數，沒有第一方描述。
- Zero Data Retention 專頁與 Prompt caching 現頁用詞不一致（beta／60 分鐘／只寫組織隔離，對上 ZDR eligible／5 分鐘或 1 小時／workspace）。沒有逐頁版本戳可證明專頁已下線。
- Pre-warming 的上線日期不在已讀的 release notes 摘錄裡。
- 2026-02-05 workspace 隔離沒有 changelog 條目，也沒有遷移對客戶命中率的第一方數字。

## 已公布的命中率、TTFT 與價格

### Takeaway

Anthropic 沒有公布服務級 hit rate。有日期的數字是價格倍率，以及 2024 年啟動案例的 TTFT／成本降幅；那些案例仍掛在 2025-08-14 的文章頁上。2026 年的價格變動以 release notes 為準。次級部落格的「命中率從 38% 回到 92%」不是 Anthropic 數字。

### Cited Findings

- 啟動宣傳，文章頁日期 August 14, 2025，內文更新戳 December 17, 2024（GA；Bedrock 與 Vertex 當時為 preview）。正文：「reducing costs by up to 90% and latency by up to 85% for long prompts.」早期客戶表：
  - Chat with a book，100,000 token cached prompt：TTFT 11.5s → 2.4s（−79%），成本 −90%。
  - Many-shot，10,000 token prompt：1.6s → 1.1s（−31%），成本 −86%。
  - Multi-turn，10 輪加長 system prompt：約 10s → 約 2.5s（−75%），成本 −53%。
  同頁當時寫 cache write 比 base input 貴 25%，cache read 為 base input 的 10%。表的腳註方法沒有在摘錄中展開。— [Prompt caching with Claude](https://claude.com/blog/prompt-caching)
- 同一功能在 2024-12-17 release notes 寫的是「latency by up to 80%」，不是 85%。與部落格的 85% 並存，Anthropic 沒有解釋差額。— [Release notes，2024-12-17](https://platform.claude.com/docs/en/release-notes/overview)
- 2025-05-22 宣布 1 小時 TTL 時再次使用「up to 90%」成本與「up to 85%」延遲，沒有新的 TTFT 表。— [Anthropic 貼文](https://x.com/AnthropicAI/status/1925633128174899453)
- 2026-10-14 價格表的倍率：5 分鐘 write 1.25x，1 小時 write 2x，read 一般 0.1x。註腳：Fable 5.1 與 Mythos 5.1 的 hit／refresh 是 0.025x；Opus 5.5 與 Sonnet 5.5 是 0.05x。表上絕對價例：Opus 5.5 base input $4、5 分鐘 write $5、1 小時 write $8、hit $0.20／MTok；Sonnet 5.5 base $2、5 分鐘 write $2.50、1 小時 write $4、hit $0.10／MTok；Fable 5.1 base $10、hit $0.25／MTok。倍率可與 Batch、long context、data residency 疊加。— [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 2026-10-07：「lowered the price of prompt cache reads on Claude Sonnet 5.5 from $0.20 USD to $0.10 USD per million tokens: 0.05x the base input price instead of 0.1x. Cache writes and all other prices are unchanged.」— [Release notes，2026-10-07](https://platform.claude.com/docs/en/release-notes/overview)
- 2026-09-01：Fable 5.1 與 Mythos 5.1 的 cache read 為 $0.25／MTok（0.025x），write 不變。— [Release notes，2026-09-01](https://platform.claude.com/docs/en/release-notes/overview)
- 2024-12-17 的 80% 與部落格 85% 都是「up to」，不是保證的 TTFT。現行文件只說長文件「generally see improved time-to-first-token」，5 分鐘與 1 小時延遲表現相同，沒有 2026 年的新 TTFT 表。— [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- Claude Code 文說他們對 hit rate 做告警，甚至低到會宣 SEV，但沒有給百分比。頁面日期 April 30, 2026。— [Lessons from building Claude Code](https://claude.com/blog/lessons-from-building-claude-code-prompt-caching-is-everything)

### Inferences

- 90% 成本降幅對得上「read ≈ 0.1x」的穩態，不是單次請求的帳單。Write 溢價在第一次。0.05x 與 0.025x 的型號穩態降幅會大於 90%，但 Anthropic 沒有為這兩個倍率重發案例表。
- 11.5s → 2.4s 是 2024 年長 prompt 案例，模型與硬體都已更換。不能當作 2026 年 Opus 5.5 的 TTFT。

### Gaps

- 沒有 Anthropic 公布的生產環境 hit rate、workspace 共享率，或容量導致的非 TTL miss 比例。
- 啟動表的對照組（是否含網路、輸出長度、哪個模型）只在未完整擷取的腳註。85% 與 80% 兩個「up to」沒有對齊。
- AWS 產品頁也寫 Bedrock「up to 90%」成本與「up to 85%」延遲。那是 AWS 行銷句，不是具名基準，本文不把它算進 Anthropic 數字。

## 第一方是否描述內部 KV 佈局？

### Takeaway

沒有。第一方只說記憶體中的 KV 表示加上 cryptographic prefix hash、exact prefix match、以及最小 TTL。沒有 paged blocks、radix tree、block 大小或淘汰資料結構。公開合約就是隔離邊界內的 TTL 加精確前綴。第三方把 Anthropic 說成 vLLM／SGLang 同類實作，不是第一方架構。

### Cited Findings

- 「KV (key-value) cache representations and cryptographic hashes of cached content are held in memory only and are not stored at rest.」— [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 「Cache keys are generated using a cryptographic hash of the prompts up to the cache control point.」寫入「exactly one cache entry: a hash of the prefix ending at that block.」讀取先算 breakpoint 的 prefix hash，沒中再逐 block 往回，最多 20 個位置。— [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 2026-09-08 文章把 prefill 稱為貴的那一段，prompt caching「saves that state (the key–value, or KV, cache)」；相同前綴就讀回而不是重算。條件是模型固定、byte-exact、TTL 內。沒有描述頁、樹或 hash 的區塊粒度。— [Reducing cost and improving performance](https://claude.com/resources/articles/reducing-cost-and-improving-performance-with-claude-platform)
- Claude Code 說明文（2026-08-14）：請求若以前一次伺服器剛看過的相同 tokens 開頭，那段開頭的 state 可以留著，只 prefill 後面的部分。仍是 prefix state，不是資料結構。— [Maximizing the value of your Claude Code sessions](https://claude.com/resources/articles/maximizing-the-value-of-your-claude-code-sessions)
- Google 只多說 hash 是 fingerprint，且他們不把 hash 當 Customer Data。沒有 KV 佈局。— [Google Cloud Claude prompt caching](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/claude/prompt-caching)
- AWS 產品頁說 cache「comprises the internal model state representing your prompts.」User guide 描述 checkpoint 與約 20 個 content block 的回看，與 Anthropic 的 20-block lookback 同形，仍然不是記憶體配置。— [Bedrock 產品頁](https://aws.amazon.com/bedrock/prompt-caching/)；[Bedrock user guide](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)

查過、沒有第一方 KV 佈局的來源：

- [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)（含 FAQ、Data retention、invalidation、1 小時 TTL、automatic caching）
- [Workspaces](https://platform.claude.com/docs/en/manage-claude/workspaces)
- [Cache diagnostics](https://platform.claude.com/docs/en/build-with-claude/cache-diagnostics)
- [API and data retention](https://platform.claude.com/docs/en/manage-claude/api-and-data-retention) 與 [Zero Data Retention](https://platform.claude.com/docs/en/build-with-claude/zero-data-retention)
- [Release notes](https://platform.claude.com/docs/en/release-notes/overview)（2024-12 至 2026-10 與 caching 相關條目）
- [Messages API `cache_control` schema](https://platform.claude.com/docs/en/api/messages)
- [Reducing cost and improving performance](https://claude.com/resources/articles/reducing-cost-and-improving-performance-with-claude-platform)、[Prompt caching with Claude](https://claude.com/blog/prompt-caching)、[Lessons from building Claude Code](https://claude.com/blog/lessons-from-building-claude-code-prompt-caching-is-everything)
- Bedrock 與 Google Cloud 的 caching 頁，如上

第三方（kunwar.page、Spheron、Analytics Vidhya、AI Wiki）把 Anthropic 的 prompt caching 類比成 vLLM prefix cache 或 SGLang RadixAttention，並自行描述 LRU。那些句子沒有 Anthropic 出處。不採用。

### Inferences

- 「cryptographic hash of the prefix」加上「只在 breakpoint 寫一筆」比較像粗粒度前綴鍵，而不是文件所說的逐 token radix。這仍是合約行為，不是記憶體佈局。20-block lookback 是 API 搜尋窗口，不是 GPU page size。
- 多租戶隔離寫在 hash 的存取範圍（workspace 或 organization），不是寫在「不同客戶的 KV 頁不能相鄰」。頁級隔離沒有公開。

### Gaps

- 沒有第一方的 block size、page size、radix、prefix tree、跨 replica 是否共槽、或 region 內有幾份副本。
- 沒有寫 hash 是否含模型 ID、workspace ID、TTL 種類、thinking 設定；這些只以「改了就 miss」的行為出現。
- 工程部落格沒有服務端論文或 scheduler 文章描述 Claude API 的 KV 池。已查的第一方工程文停留在客戶端如何排 prefix。
