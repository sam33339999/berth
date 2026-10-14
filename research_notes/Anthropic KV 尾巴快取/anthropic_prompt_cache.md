# Anthropic prompt caching 機制（官方 API，截至 2026-10-14）

範圍是 Claude API、Amazon Bedrock、Google Cloud（Vertex / Gemini Enterprise Agent Platform）上公開文件描述的 prompt caching。不是 GPU radix tree 實作推測。官方沒有名為 tail cache、suffix cache 或 conversation-end cache 的獨立產品或計費層。文件裡和「尾巴」最接近的用語是 varying suffix、last cacheable block、incremental caching，以及 lookback。快取單位始終是從 prompt 開頭到 breakpoint 的 prefix，不是任意後綴。

主要頁面：<https://platform.claude.com/docs/en/build-with-claude/prompt-caching>（`docs.anthropic.com` 同路徑會導向此頁）。檢索日 2026-10-14。

## Claude API、Bedrock、Vertex 的請求欄位、breakpoint 與放置順序是什麼？

### Takeaway
現行 Claude API 用 `cache_control`，`type` 只有 `"ephemeral"`，可加 `ttl`。啟用方式有兩種：請求頂層單一 `cache_control`（automatic caching），或在 content block 上最多 4 個 explicit breakpoint。Prefix 順序固定為 `tools` → `system` → `messages`。Bedrock 另有 implicit caching 與 `cachePoint`；Vertex 沿用 Anthropic 的 `cache_control`，但快取綁 Google Cloud project。

### Cited Findings
- 兩種啟用方式：「Automatic caching: Add a single `cache_control` field at the top level of your request. The system automatically applies the cache breakpoint to the last cacheable block and moves it forward as conversations grow.」以及「Explicit cache breakpoints: Place `cache_control` directly on individual content blocks」。頂層範例為 `"cache_control": {"type": "ephemeral"}`，也可 `"ttl": "1h"`。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 「Currently, "ephemeral" is the only supported cache type, which by default has a 5-minute lifetime.」Beta Messages schema 寫 `ttl` 可為 `"5m"` 或 `"1h"`，「Defaults to `5m`」。指南範例通常省略 `ttl`（即 5 分鐘）或設 `"1h"`，沒有要求再送舊的 beta header。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)；[Beta API](https://platform.claude.com/docs/en/api/beta)
- 「You can define up to 4 cache breakpoints」。Automatic caching「uses one of the 4 available breakpoint slots」。邊界：最後一塊已有相同 TTL 的 explicit `cache_control` 時 automatic 是 no-op；TTL 不同回 400；已有 4 個 block-level breakpoint 再加頂層欄位回 400；最後一塊不能當 automatic target 時「silently walks backward」，都沒有就「caching is skipped」。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 「Prompt caching references the entire prompt: `tools`, `system`, and `messages` (in that order), up to and including the block designated with `cache_control`.」「Cache prefixes are created in the following order: `tools`, `system`, then `messages`.」 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 可快取：`tools` 定義、`system` array 的 content blocks、`messages` 裡 user 與 assistant 的 text、user turn 的 images 與 documents、`tool_use` 與 `tool_result`。「Each of these elements can be cached, either automatically or by marking them with `cache_control`.」工具斷點放在 `tools` array 最後一個 tool。`mcp_toolset`、computer use、browser use 要把 `cache_control` 放在 toolset entry 上，API 套到展開後的最後一個 tool。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)；[Tool use with prompt caching](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-use-with-prompt-caching)
- 不能直接標：thinking blocks「cannot be cached directly with `cache_control`」，但出現在先前 assistant turn 時可隨其他內容一起被寫入，讀回時計入 input tokens。Citations 的 sub-content 不能直接標，要標頂層 document block。Empty text blocks 不能快取。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 沒有任何 `cache_control`（頂層或 block）就完全不快取：「Prompt caching is opt-in: if a request has no `cache_control` field (automatic or an explicit breakpoint), nothing is cached and every request pays the regular input token price」。 — [Mid-conversation system messages](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages)
- Explicit breakpoint 的寫入規則：「Marking a block with `cache_control` writes exactly one cache entry: a hash of the prefix ending at that block. The system does not write entries for any earlier position.」讀取會從該 breakpoint「walks backward one block at a time」找先前寫過的 entry。「The lookback window is 20 blocks」，「counting the breakpoint itself as the first」。Claude API 上連續 `tool_use` 算一個 position，連續 `tool_result` 也算一個。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 混用 TTL：「Cache entries with longer TTL must appear before shorter TTLs (that is, a 1-hour cache entry must appear before any 5-minute cache entries).」 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- Automatic caching「is available on every platform except the legacy Amazon Bedrock (Opus 4.6 and earlier) integration. On that integration, the API returns a 400 error for a top-level `cache_control` field」。1-hour TTL 文件列在 Claude API、Amazon Bedrock、legacy Bedrock（Opus 4.6 and earlier）、Claude Platform on AWS、Google Cloud、Microsoft Foundry。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- Bedrock 有兩種：「Implicit Prompt Caching」不需要 cache control，「best effort. Repeating an identical prompt doesn't guarantee a cache hit」。Explicit 用 cache checkpoint。Anthropic 模型在 Bedrock「support both Implicit Prompt Caching and Explicit Prompt Caching」。Converse 的 marker 是獨立 content block `{"cachePoint": {"type": "default"}}`，1 小時為 `"ttl": "1h"`。InvokeModel（Claude）用 `cache_control: {"type": "ephemeral"}`，1 小時同樣加 `"ttl": "1h"`。只支援 5 分鐘的模型若帶 `ttl` 可能 `ValidationException`。處理順序同樣是 `tools` → `system` → `messages`。Claude 的 simplified cache management 也是「looking back up to approximately 20 content blocks」。 — [Bedrock prompt caching](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)
- Bedrock 只支援 on-demand。「It is not supported with the batch inference API.」Cross-region inference 在高需求時「may lead to increased cache writes」。 — [Bedrock prompt caching](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)
- Vertex / Gemini Enterprise Agent Platform：「Caches are unique to your Google Cloud project and cannot be used by other projects.」行為指向 Anthropic 文件。預設 five-minute TTL，`"ttl": "1h"` 可延長。「The cache lifetime is refreshed each time the cached content is accessed.」1 小時「isn't supported」於 Google Cloud 上的 Claude 3.7 Sonnet、Claude 3.5 Sonnet v2、Claude 3.5 Sonnet、Claude 3 Opus。可向支援申請在特定 project 完全停用 explicit caching，停用後帶 caching 的請求會被拒絕。頁面最後更新 2026-10-13 UTC。 — [Vertex prompt caching](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/prompt-caching)

### Inferences
- Claude API 的 automatic caching 不是「不放任何 marker 就自動快取」。必須有頂層或 block 上的 `cache_control`。Bedrock implicit caching 才是不放 marker 的 best-effort prefix reuse，而且官方不保證 hit。
- Explicit `cache_control` 必須掛在 block 上，因此 `system` 與 message content 要用 array of blocks，不能用純字串。頂層 automatic caching 則允許 `system` 仍是字串，由伺服器把斷點放到最後一個可快取 block。
- 官方沒有第五種 cache `type`，也沒有獨立的 tail / suffix tier。`ttl` 只有 5 分鐘與 1 小時兩檔。

### Gaps
- 未在非 beta 的 Messages schema 頁逐欄複核 `cache_control`；`"5m" | "1h"` 的列舉出自 Beta API reference，與指南範例一致，但是否所有 GA endpoint 都接受顯式 `"5m"` 未另開參考頁核對。
- Bedrock implicit caching 對 Claude 的最低 token、TTL 與計費是否與 explicit checkpoint 完全相同，模型卡層級的句子沒有整表抄出；userguide 只明確把 token 下限、checkpoint 上限與 TTL 列在 Explicit 表。
- Vertex 頁沒有逐模型列出最低 token 或 usage 欄位名，只說行為同 Anthropic 文件。

## cache hit、cache write、cache miss 如何區分，usage 欄位與價格倍率是什麼？

### Takeaway
沒有名為 cache miss 的 usage 欄位。Hit 記在 `cache_read_input_tokens`，write 記在 `cache_creation_input_tokens`，並用 `cache_creation.ephemeral_5m_input_tokens` 與 `ephemeral_1h_input_tokens` 拆 TTL。`input_tokens` 只算最後一個 breakpoint 之後、未進快取的 token。5 分鐘 write 為 1.25 倍 base input，1 小時 write 為 2 倍；read 多數模型 0.1 倍，Opus 5.5 與 Sonnet 5.5 為 0.05 倍，Fable 5.1 與 Mythos 5.1 為 0.025 倍。

### Cited Findings
- 流程：有 caching 時先檢查 breakpoint 為止的 prefix 是否已在近期查詢的快取中；有則用快取；否則處理完整 prompt，「caches the prefix once the response begins」。並發時「a cache entry only becomes available after the first response begins」。要讓平行請求 hit，必須等第一個 response 開始。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- `usage`（streaming 在 `message_start`）：`cache_creation_input_tokens` 是「Number of tokens written to the cache when creating a new entry」；`cache_read_input_tokens` 是「Number of tokens retrieved from the cache」；`input_tokens` 是「tokens after the last cache breakpoint」。公式：`total_input_tokens = cache_read_input_tokens + cache_creation_input_tokens + input_tokens`。低於最低長度時兩個 cache 欄位都是 0，請求仍成功、不報錯。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 1 小時範例同時有 read 與 write，並拆開 TTL：`cache_creation.ephemeral_5m_input_tokens` 與 `ephemeral_1h_input_tokens`。「the current `cache_creation_input_tokens` field equals the sum of the values in the `cache_creation` object.」 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 混 TTL 的計費位置：`A` 是最高 cache hit 的 token 數（沒有 hit 則 0）；`B` 是 `A` 之後最高的 1-hour `cache_control`（沒有則等於 `A`）；`C` 是最後一個 `cache_control`。收費為：`A` 的 cache read、`(B - A)` 的 1-hour write、`(C - B)` 的 5-minute write。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 倍率：「5-minute cache write tokens are 1.25 times the base input tokens price」；「1-hour cache write tokens are 2 times」；「Cache read tokens are 0.1 times the base input tokens price」，註腳例外：Fable 5.1 與 Mythos 5.1 的 hits and refreshes 為 0.025 倍；Opus 5.5 與 Sonnet 5.5 為 0.05 倍。「All other models use the standard 0.1x multiplier.」FAQ 另寫 read 為「10% of base input token price, or 2.5% on Claude Fable 5.1 and Claude Mythos 5.1, and 5% on Claude Opus 5.5 and Claude Sonnet 5.5」。Breakpoint 本身不加價。Output「has no effect」於 output token generation，回應與不快取時相同。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 2026-10-14 價格表（USD / MTok），欄位為 base input、5m write、1h write、cache hits and refreshes、output。Fable 5.1 與 Mythos 5.1：$10 / $12.50 / $20 / $0.25 / $50。Fable 5 與 Mythos 5：$10 / $12.50 / $20 / $1 / $50。Opus 5.5：$4 / $5 / $8 / $0.20 / $20。Opus 5、Opus 4.8、4.7、4.6、4.5：$5 / $6.25 / $10 / $0.50 / $25。Opus 4.1 與 Opus 4（已 retired，例外平台仍列）：$15 / $18.75 / $30 / $1.50 / $75。Sonnet 5.5：$2 / $2.50 / $4 / $0.10 / $10。Sonnet 5：$2 / $2.50 / $4 / $0.20 / $10。Sonnet 4.6、4.5、Sonnet 4：$3 / $3.75 / $6 / $0.30 / $15。Haiku 5.5 提示 ≤100,000 tokens：$0.10 / $0.125 / $0.20 / $0.01 / $0.50；超過 100,000：$0.50 / $0.625 / $1 / $0.05 / $2.50。Haiku 4.5：$1 / $1.25 / $2 / $0.10 / $5。Haiku 3.5：$0.80 / $1 / $1.60 / $0.08 / $4。倍率可與 Batch API、long context pricing、data residency 疊加。「On a model priced by prompt length, cache reads and cache writes count toward a request's prompt length.」 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- ITPM：`input_tokens` 與 `cache_creation_input_tokens` 計入；`cache_read_input_tokens`「Do NOT count towards ITPM for most models」。例外：「Claude Haiku 3.5 ... also counts `cache_read_input_tokens` toward ITPM」。同頁又寫沒有 † 的模型「are billed at a reduced rate (10% of base input token price)」。 — [Rate limits](https://platform.claude.com/docs/en/api/rate-limits)
- Priority Tier 容量折算與帳單倍率不是同一件事：cache read 算 0.1 token per token，5 分鐘 write 算 1.25，1 小時 write 算 2.00。 — [Service tiers](https://platform.claude.com/docs/en/api/service-tiers)
- Bedrock Converse 欄位名不同：`cacheReadInputTokens`、`cacheWriteInputTokens`、`cacheDetails`（寫入所用 ttl）。`inputTokens` 只代表「non-cached input tokens」。公式：`total input tokens = inputTokens + cacheReadInputTokens + cacheWriteInputTokens`。Explicit「doesn't guarantee that an eligible request results in a cache hit」。計費細節指向 Bedrock pricing，不在 userguide 重列 Claude 單價。 — [Bedrock prompt caching](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)
- Vertex 計費句（2026-10-13）：5 分鐘 write「25% more expensive than base input tokens」；1 小時 write「100% more expensive」；「Cache read tokens are 90% cheaper than base input tokens.」未列出 0.05 倍或 0.025 倍例外。 — [Vertex prompt caching](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/prompt-caching)
- 同一次請求可以同時 read 舊 prefix 並 write 新尾段。多輪範例：`cache_read_input_tokens` 是到上一輪為止的 token，`cache_creation_input_tokens` 是新的 assistant 與 user turn，`input_tokens` 是新 user message 且「will be minimal」。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- Context window 仍把三段都算進去：「the input count is split across `input_tokens`, `cache_read_input_tokens`, and `cache_creation_input_tokens`, and all three count toward the window.」「Cached prompt prefixes still occupy the context window」；caching 改的是價格，不是是否佔視窗。 — [Context windows](https://platform.claude.com/docs/en/build-with-claude/context-windows)

### Inferences
- 「Miss」在文件裡是行為描述，不是計費桶。低於門檻或沒放 `cache_control` 時，token 整段落在 `input_tokens`，cache 兩欄為 0。有 breakpoint、長度夠、但 hash 不合時，是 write（`cache_creation_input_tokens` > 0，read 為 0），不是免費的 miss。Lookback 命中較早的 entry 時，同一回應會同時有 read 與 write。
- Rate limits 頁的「一律 10%」與 prompt caching 價格表的 0.05 倍、0.025 倍衝突。計價應以 prompt caching 表為準；rate limits 頁的 10% 句看起來沒有跟上 5.5 / 5.1 的 read 折扣。Priority Tier 的 0.1 是容量權重，不能當成這些新模型的帳單單價。

### Gaps
- 未核對 Bedrock 價目表上各 Claude 模型的 cache read / write 美元單價是否與 Claude API 表逐項相同。
- `usage.iterations[]` 在串流或 fallback 範例中會再拆每輪的 cache 欄位，但 prompt caching 主文沒有定義何時出現、與頂層 usage 如何加總。未當作快取語意的必要欄位。
- 文件沒有寫 cache hit 的延遲 SLA，只有「generally see improved time-to-first-token」，以及 2024 公告的個案數字（見歷史題）。

## 各模型最低可快取前綴是多少，什麼改動會讓 hit 失敗？

### Takeaway
門檻按模型而不是按「Haiku / Sonnet / Opus」家族一刀切。現行 Claude API 從 512 到 4,096 tokens。不夠長就靜默不快取。Hit 要求到 breakpoint 為止的 prefix 100% 相同，包含文字與圖片；`tools` 變動會讓其後全部失效。Bedrock 對 Opus 4.7 寫 4,096，Anthropic 主文寫 2,048，兩邊衝突。

### Cited Findings
- Claude API、Claude Platform on AWS、Google Cloud、Microsoft Foundry 的最低長度（Anthropic 主文，並寫「These minimums apply on every platform where each model is available」）：512 tokens：Fable 5.1、Mythos 5.1、Opus 5.5、Opus 5、Sonnet 5.5、Fable 5、Mythos 5、Haiku 5.5。2,048：Mythos Preview、Opus 4.7。4,096：Opus 4.6、Opus 4.5。1,024：Opus 4.8、Sonnet 5、Sonnet 4.6、Sonnet 4.5、Opus 4.1、Opus 4、Sonnet 4。4,096：Haiku 4.5。2,048：Haiku 3.5。「Shorter prompts cannot be cached, even if marked with `cache_control`」，不回錯誤。同段又說 Bedrock 的下限、失敗行為與 usage 欄位名改看 AWS 文件。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- Bedrock explicit 表（與上表多數一致，衝突見下）：Haiku 5.5、Sonnet 5.5、Opus 5.5、Fable 5.1、Mythos 5.1、Fable 5、Mythos 5、Opus 5 皆 512，最多 4 checkpoint，TTL 5 分鐘與 1 小時，欄位 `system`、`messages`、`tools`。Opus 4.8 與 Sonnet 5、Sonnet 4.6、Sonnet 4.5 為 1,024。Opus 4.7、Opus 4.6、Opus 4.5、Haiku 4.5 為 4,096。Claude 3.7 Sonnet 與 Claude 3.5 Sonnet v2 為 1,024、最多 4、只列 5 minutes。下限是 checkpoint 前整個 prefix（含 `tools`、`system`、`messages`）的累計 token，「There is no minimum number of tokens required between cache checkpoints」。不夠長時「inference still succeeds, but your prefix isn't cached」。 — [Bedrock prompt caching](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)
- 「Cache hits require 100% identical prompt segments, including all text and images up to and including the block marked with cache control.」Cache diagnostics 頁寫 prefix 必須與近期請求「byte-for-byte identical」。Mid-conversation 頁寫 hash 順序是 `tools`，then `system`，then `messages`，「A cache hit requires the prefix to match a recent request exactly, byte for byte, up to the cache breakpoint.」 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)；[Cache diagnostics](https://platform.claude.com/docs/en/build-with-claude/cache-diagnostics)；[Mid-conversation system messages](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages)
- 失效表（✘ 為該層失效）。改 tool definitions（names, descriptions, parameters）：tools、system、messages 全失效。Web search toggle、citations toggle、`speed: "fast"` 與 standard 切換：system 與 messages 失效，tools 仍有效。`tool_choice`、images（「Adding/removing images anywhere in the prompt」）：只失效 messages。Thinking parameters（mode，以及 extended mode 的 `budget_tokens`）「always invalidates message blocks」；會把設定 render 進 prompt，因此部分模型連 tools 與 system 一起失效。`output_config.effort` 相同；明確設成該模型預設值「is equivalent to omitting it and does not invalidate」。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 工具頁另列：改 `disable_parallel_tool_use` 只失效 messages cache。Web search、web fetch 的啟用或停用失效 system 與 messages。Code execution 的 container state「is independent of prompt cache」。`defer_loading` 的工具不進 system-prompt prefix；tool search 發現後以 `tool_reference` 追加在對話裡，「The prefix is untouched」。 — [Tool use with prompt caching](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-use-with-prompt-caching)
- Thinking blocks 與非 tool-result user 內容：Opus 4.5+、Sonnet 4.6+、Haiku 5.5 預設保留 thinking blocks，messages cache 仍有效。更早的 Opus/Sonnet 以及到 Haiku 4.5 為止，會把先前 thinking blocks 從 context 拿掉，其後 messages 也移出快取。Fable 5.1、Mythos 5.1、Opus 5.5、Sonnet 5.5、Haiku 5.5 若 API drop 掉接收模型讀不了的 thinking block，從該 block 起 prefix 改變。`drop_block`（`prefix_mismatch_behavior`）時「the prompt cache restarts at the edit」。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)；[Preserved thinking](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking)
- 改頂層 `system`，「even appending a sentence, produces a different hash」，system 與其後 cached messages 都 miss。支援的模型可改 append `{"role": "system"}`，讓既有 prefix 仍 hit。此能力列在 Fable 5.1、Mythos 5.1、Fable 5、Mythos 5、Opus 5.5、Opus 4.8、Opus 5、Sonnet 5.5、Haiku 5.5；「This feature is not available on Claude Sonnet 5.」`inline-tools-2026-09-15` 可用 `tool_addition` 追加工具而不改 `tools` array，prefix 仍相符；例外是 `tools` 裡沒有 non-deferred tool 時，第一次這樣定義「costs one full cache miss」。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)；[Mid-conversation system messages](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages)
- 其他會弄破 hit 的官方排查項：`tool_choice`、是否有圖、thinking 設定、`output_config.effort` 在兩次呼叫間不一致；`tool_use` content 的 JSON key 順序不穩定（文件點名 Swift、Go 可能打亂）；breakpoint 放在每次都變的 block（timestamp、這次的 user message）上，因為 write 只發生在 breakpoint，lookback「does not find stable content behind the breakpoint」。Cache diagnostics 可比較上一則 `id`，指出分歧在「the model, the system prompt, the tools, or the message history」；範例 `diagnostics.cache_miss_reason.type` 為 `system_changed`。`cache-diagnosis-2026-04-07`「is no longer required」。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)；[Cache diagnostics](https://platform.claude.com/docs/en/build-with-claude/cache-diagnostics)
- Context editing：tool result clearing「Invalidates cached prompt prefixes when content is cleared」，之後可重寫新 prefix。Thinking block clearing：保留則 cache 仍在；清除則「the cache is invalidated at the point where clearing occurs」。Beta header 範例為 `context-management-2025-06-27`。 — [Context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing)
- Compaction：可在 compaction block 上放 `cache_control`。若沒有另外的 system breakpoint，摘要會讓已快取的 system prompt 一起失效。建議在 system prompt 結尾再放一個 breakpoint，compaction 後只 write 摘要、system 仍 read。On-demand 頁寫「`cache_control` on the block places a breakpoint after the summary.」門檻式 compaction 範例 beta 為 `compact-2026-01-12`。 — [Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction)；[Compaction on demand](https://platform.claude.com/docs/en/build-with-claude/compaction-on-demand)
- Task budget：伺服器注入的 countdown marker「does not match across requests」。若客戶端每輪改 `task_budget.remaining`，會讓包含該值的 cache prefix 失效。 — [Task budgets](https://platform.claude.com/docs/en/build-with-claude/task-budgets)

### Inferences
- 「Byte-level prefix change」是官方標準：不是模糊相似，也不是只比對 system prompt。圖片的有無會動到 messages 層；工具定義的任何欄位變動從 tools 層整段作廢。
- Beta header 本身沒有被列成一張「換 header 就失效」的總表。會失效的是 header 所代表的行為改變了實際 render 進 prefix 的內容（thinking、effort、tool 定義、token-efficient 類的請求形狀）。現行文件明確點名的 header 是 `inline-tools-2026-09-15`（用來避免改 `tools`）、以及已不必再送的 `cache-diagnosis-2026-04-07`。
- Opus 4.7 的 2,048（Anthropic，且聲稱各平台適用）與 4,096（Bedrock 表）不能併成一個數字。在 Bedrock 上應先信 AWS 表，直到 Anthropic 改掉「every platform」那句或 AWS 改表。

### Gaps
- 2024-08 公開 beta 當時各模型的最低 token（常見二次資料寫 Sonnet/Opus 1,024、Haiku 2,048）沒有在本次抓到的現行頁或新聞稿正文裡出現，故不記成已證實的歷史門檻。現行頁對已 retired 的 Haiku 3.5（2,048）與 Opus 4 / Sonnet 4（1,024）仍有數字。
- Cache diagnostics 的完整 `cache_miss_reason.type` 列舉沒有整頁抄完，只確認到文件會區分 model、system、tools、messages，以及範例 `system_changed`。
- 未證實 `inference_geo` 或 `service_tier` 是否進入 cache key。Diagnostics 把 model 列為可比對的分歧點，因此換模型應視為 miss，但主文沒有寫「cache key 包含 model id」這句。

## 快取的範圍是全域、組織、workspace、API key，還是終端使用者？隔離與隱私怎麼寫？

### Takeaway
不是全域，也不是 per end user 或 per API key（文件沒有這兩種隔離）。Claude API、Claude Platform on AWS、Microsoft Foundry 是 workspace 隔離，同一 workspace 內相同 prefix 可以共用，含不同終端使用者經由同一 workspace 發出的請求。Bedrock 與 Google Cloud 在 Anthropic 文件中是 organization 隔離；Vertex 自己寫的是 Google Cloud project。快取鍵是到 breakpoint 的 cryptographic hash。KV 與 hash 只在記憶體，不落盤。

### Cited Findings
- 「Prompt caching uses workspace-level isolation. Caches are isolated per workspace ... This applies to the Claude API, Claude Platform on AWS, and Microsoft Foundry; Bedrock and Google Cloud maintain organization-level cache isolation.」「Different organizations never share caches, even if they use identical prompts.」 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- Workspaces 頁複述：Claude API、Claude Platform on AWS、Microsoft Foundry 的 prompt caches「isolated per workspace」。Amazon Bedrock 與 Google Cloud「isolated per organization」。API key「are scoped to a specific workspace」。 — [Workspaces](https://platform.claude.com/docs/en/manage-claude/workspaces)
- FAQ：「Cache keys are generated using a cryptographic hash of the prompts up to the cache control point. This means only requests with identical prompts can access a specific cache.」隔離敘述與上段相同，並寫「caches are never shared across organizations」。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 隱私與保留：「Prompt caching (both automatic and explicit) is ZDR eligible. Anthropic does not store the raw text of your prompts or Claude's responses.」「KV (key-value) cache representations and cryptographic hashes of cached content are held in memory only and are not stored at rest.」生命期至少 5 分鐘或 1 小時，「after which they are promptly, though not immediately, deleted.」FAQ：「there's no way to manually clear the cache.」 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 舊版公開文件（`docs.claude.com` 檢索摘要，模型名單停在 Opus 4.1 / Haiku 3 一代）只寫「Caches are organization-specific. Users within the same organization can access the same cache if they use identical prompts」，沒有 workspace 層。 — [舊版 Prompt caching FAQ 摘要](https://docs.claude.com/en/docs/build-with-claude/prompt-caching)
- AWS 行銷頁：「Your cache is specific to your account and comprises the internal model state representing your prompts.」 — [Amazon Bedrock prompt caching](https://aws.amazon.com/bedrock/prompt-caching/)
- Vertex：「Caches are unique to your Google Cloud project and cannot be used by other projects.」Hash 被 Google 當成 User Metadata / Service Data，不是 Cloud DPA 下的 Customer Data；「Google does not use these hashes for any other purpose.」 — [Vertex prompt caching](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/prompt-caching)

### Inferences
- 兩個終端使用者若打到同一個 Claude API workspace（實務上常是同一把或同一 workspace 的多把 API key）且 prefix 完全相同，文件允許他們 hit 同一筆快取。隔離單位不是 end user。文件沒有寫 per API key 的第二層隔離；key 只是被綁在 workspace 上。
- Bedrock 的「account」（AWS 行銷頁）與 Anthropic 的「organization」不是同一個詞。不能把 Bedrock 快取說成 Anthropic workspace，也不能說成跨 AWS account 共用。
- Vertex 的 project 隔離比 Anthropic 寫的「Google Cloud organization-level」更細。若一個 GCP organization 有多個 project，應以 Vertex 頁的 project 為準，否則可能高估 hit。

### Gaps
- 現行英文主文沒有寫 workspace 隔離的生效日。檢索摘要裡的 zh-CN 頁曾出現「從 2026 年 2 月 5 日開始」，但本次抓下的英文全文沒有這句，故不把該日期當成已核對事實。
- 未找到官方句子說明同一 workspace 的兩把 API key 是否共享快取；上面的共用結論是從「隔離單位是 workspace、不是 key」推出來的。
- ZDR 頁本身沒有整頁重讀，只引用 prompt caching 頁的「ZDR eligible」與不存 raw text。

## 5 分鐘與 1 小時 TTL 何時起算、hit 會不會刷新、過期後怎樣？

### Takeaway
只有兩檔：預設 5 分鐘，以及 opt-in 的 1 小時。TTL 從「寫入或讀取該 entry 的請求開始」起算，不是從回應結束起算，生成時間算在窗內。每次使用以不再收費的方式刷新。過期後不能手動刪；下次是新的 write。沒有第三檔 TTL。

### Cited Findings
- 「By default, the cache has a 5-minute lifetime. The cache is refreshed for no additional cost each time the cached content is used.」「The lifetime is measured from the start of the request that writes or reads the cache entry, not from the end of its response.」例子：回應串流 4 分鐘，則後續請求必須在該回應完成後約 1 分鐘內開始，才還在 5 分鐘窗內。FAQ「When does the cache lifetime start?」複述同一規則。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 1 小時寫法：`"cache_control": {"type": "ephemeral", "ttl": "1h"}`，write 為 2 倍 base input。若流量「more frequently than every 5 minutes」，應繼續用 5 分鐘，因為 hit「will continue to be refreshed at no additional charge」。1 小時適合間隔大於 5 分鐘但小於 1 小時、在意延遲、或想利用 cache hit 不扣 rate limit 的情況。兩種 TTL「behave the same with respect to latency」。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 過期：「Cached entries have a minimum lifetime of 5 minutes (standard) or 1 hour (extended), after which they are promptly, though not immediately, deleted.」FAQ：「Cached prefixes automatically expire after a minimum of 5 minutes of inactivity.」「Currently, there's no way to manually clear the cache.」 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- Bedrock：「The cache has a Time To Live (TTL), which resets with each successful cache hit.」「If no cache hits occur within the TTL window, your cache expires.」預設省略 `ttl` 就是 5 分鐘。 — [Bedrock prompt caching](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)
- Vertex：「By default, the cache has a five-minute lifetime」。`"ttl": "1h"` 可延長，且「refreshed each time the cached content is accessed」。上題所列舊模型沒有 1 小時。 — [Vertex prompt caching](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/prompt-caching)
- 多輪範例把 lookback hit 同時稱為「a cache hit (and also a cache refresh!) if they are hit within 5 minutes」。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)

### Inferences
- Refresh 不是再付一次 write。使用既有 entry 只付該次 read（依模型 0.1 / 0.05 / 0.025 倍），並把 TTL 從這次請求的起點重算。
- 過期後的下一筆請求若仍帶 breakpoint 且長度夠，就是一筆新的 cache write，不是讀到殘留資料。文件說刪除「not immediately」，但沒有承諾過期後仍可 hit。
- 長回應會吃掉 5 分鐘窗。官方建議改 1 小時，而不是假設 TTL 從最後一個 token 才開始。

### Gaps
- 「Promptly, though not immediately」沒有上限秒數。過期後的競態是否仍可能 hit，文件沒寫。
- 1 小時 entry 被 5 分鐘 read 刷新時，剩餘 TTL 是維持 1 小時還是被較短 read 改寫，主文沒有單獨句子。只寫「refreshed each time the cached content is used」以及 5 分鐘與 1 小時是不同 entry。未把「讀取會把 1 小時降成 5 分鐘」記成事實。
- 第三方 2026-04 文章聲稱預設 TTL 在 2026-03-06 從 1 小時被悄悄改成 5 分鐘，並使用 `ttl: 3600`。這與 2024 公告以來官方「預設 5 分鐘、1 小時要顯式 opt-in」以及現行 `"1h"` 字串矛盾，不採信。

## 多輪對話官方要怎麼放 breakpoint？有沒有對話尾端或 incremental cache？

### Takeaway
官方做法是快取「到目前為止的穩定 prefix」，不是快取後綴。多輪時把 breakpoint 放在每一輪最後一個穩定 block，或用頂層 automatic caching 讓斷點跟著最後一個可快取 block 往前移。先前輪次不必再標 `cache_control`，20-block lookback 會找到舊 write，並順便 refresh。對話每輪若新增少於 20 個 block，單一尾端 breakpoint 就夠；超過就要預先再放一個較早的 breakpoint。Varying suffix（timestamp、本輪使用者原文）必須留在 breakpoint 之後。

### Cited Findings
- Automatic caching 多輪表：Request 1 在最後一個 User 上寫入全部；Request 2 讀到上一個 User，只 write 新的 assistant 與 user；Request 3 類推。「The cache breakpoint automatically moves to the last cacheable block in each request, so you don't need to update any `cache_control` markers as the conversation grows.」 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- Explicit 多輪：「During each turn, the final block of the final message is marked with `cache_control` so the conversation can be incrementally cached. The system automatically looks up and uses the longest previously cached sequence of blocks for follow-up messages. That is, blocks that were previously marked with a `cache_control` block are later not marked with this, but they will still be considered a cache hit (and also a cache refresh!) if they are hit within 5 minutes.」System 上另放一個 breakpoint，以便超過 5 分鐘被清掉後下一筆請求能把 system 再寫回去。預期 usage：`input_tokens` 只有新 user message，`cache_creation_input_tokens` 是新的 assistant 與 user turns，`cache_read_input_tokens` 是到上一輪為止。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- FAQ：「In most cases, a single cache breakpoint at the end of your static content is sufficient.」需要多個 breakpoint 的情況包含：對話成長讓 breakpoint 比上一次 write 多了 20 個 block 以上、不同區段更新頻率不同、或要控制成本。Lookback 範例：turn 3 有 35 blocks、斷點在 block 35 時，20 個 position 的視窗看不到 block 15 的舊 entry，「there is no cache hit」。要在 block 15 再放一個 breakpoint，才會開第二個 lookback 視窗。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 「For a prompt with a varying suffix (timestamps, per-request context, the incoming message), place the breakpoint at the end of the static prefix, not on the varying block.」Automatic caching 若把斷點放在每筆都變的最後一塊，會每次 write、永不 read。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 四個 breakpoint 的官方範例順序就是層級：tools 的最後一個定義、system 裡很少變的 instructions、RAG documents、最後一則 user message（「to enable incremental caching of the conversation as it progresses」）。只追加新輪且把第四個斷點移到新的最後一則時，前三段 read，新的 user 與上一則 assistant write。任一 breakpoint 的內容改變會讓該段及其後失效，較早的段仍有效。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 靜態前綴可預熱：`max_tokens: 0`，breakpoint 放在與正式請求共用的最後一塊（通常是 system 或 tools），不要放在 placeholder user message（`"warmup"` 即可，但必須有非空白內容）。否則 entry「is keyed to the placeholder」。回應 `content` 為 []，`stop_reason: "max_tokens"`，usage 仍完整。這是 cache write，output token 計費為 0。5 分鐘快取要至少每 5 分鐘再預熱一次。`max_tokens: 0` 在 `stream: true`、extended thinking、structured outputs、`tool_choice` 為 `tool` 或 `any`、以及 Message Batches 裡會被拒。文件說這取代了過去 `max_tokens: 1` 的暖機做法。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- Thinking 與工具回合：不能在 thinking block 上標 `cache_control`，但把 thinking、`tool_use`、`tool_result` 傳回去時，它們會成為下一筆 request prefix 的一部分。範例把 `cache_control` 放在 `tool_result` 上。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)

### Inferences
- 「Incremental / 對話尾端快取」是放置策略的名字，不是另一種 cache。被快取的仍是從 `tools` 起到該尾端 block 的整段 prefix。下一次請求的新 user 文字若在 breakpoint 之後，就只以 `input_tokens` 計價，不會單獨形成可被別人命中的 suffix entry。
- 若每輪使用者文字都不同，又不想把該文字寫進快取，不應使用「永遠標在最後一塊」的 automatic caching。應在穩定 prefix 結尾放 explicit breakpoint，讓 varying suffix 留在後面。這是官方針對 automatic caching 的明確陷阱，也是和「tail cache」這個暱稱最容易混淆的地方。
- 20-block 視窗是官方限制，不是實作猜測。平行工具呼叫在 Claude API 上被折成一個 position，所以單靠工具數量不一定會把上一輪擠出視窗。

### Gaps
- 「Approximately 20」出現在 Bedrock 的 simplified cache management；Claude API 主文寫的是「at most 20 positions」。兩邊用詞不完全相同，未證實 Bedrock 的 20 是否也把連續 tool blocks 折成一個 position。
- Automatic caching 上線日期不在本次抓到的 changelog 正文裡。

## 從 2024-08 公開上線到 2026-10，文件記載了哪些變更、棄用欄位與模型例外？

### Takeaway
2024-08-14 公開 beta，當時必須 `anthropic-beta: prompt-caching-2024-07-31`。2024-12-17 起 GA，不再走 `client.beta.prompt_caching`。1 小時快取以 beta header `extended-cache-ttl-2025-04-11` 進入 enum，現行指南只要求 `ttl: "1h"`。2026 年的文件新增頂層 automatic caching、workspace 隔離、cache diagnostics、`max_tokens: 0` 預熱，以及依模型下調的最低 token 與 read 倍率。沒有官方的第三 TTL，也沒有把預設從 5 分鐘改成別的值。

### Cited Findings
- 新聞稿：「Update: As of December 17, 2024, prompt caching is Generally Available on the Anthropic API. Prompt caching is also available in preview in Amazon Bedrock and on Google Cloud's Vertex AI.」正文仍寫公開 beta 可用於 Claude 3.5 Sonnet、Claude 3 Opus、Claude 3 Haiku，並給出最高約 90% 成本與 85% 延遲的個案（100,000 token 書籍：TTFT 11.5s 到 2.4s，成本 -90%）。 — [Prompt caching with Claude](https://anthropic.com/news/prompt-caching)
- 2024-08-14 Anthropic 員工貼文：啟用方式是 `"cache_control": {"type": "ephemeral"}` 加上 `"anthropic-beta": "prompt-caching-2024-07-31"`。 — [Alex Albert，2024-08-14](https://x.com/alexalbert__/status/1823751995901272068)
- 2024-11-01 的文件快照仍標 Prompt caching (beta)，並要求該 header。 — [Wayback，2024-11-01](https://web.archive.org/web/20241101181804/https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- 現行 FAQ：`AttributeError: 'Beta' object has no attribute 'prompt_caching'` 是因為「Prompt caching no longer requires the beta prefix」。應使用 `client.messages.create`，不要用 `client.beta.prompt_caching.messages.create`。TypeScript 同理，不再用 `client.beta.promptCaching.messages.create`。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- `prompt-caching-2024-07-31` 與 `extended-cache-ttl-2025-04-11` 仍出現在 beta header 列舉裡。現行 prompt caching 指南的 1 小時範例沒有再要求 `extended-cache-ttl-2025-04-11`。 — [Beta headers 列舉見 API beta 文件](https://platform.claude.com/docs/en/api/beta)；[Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 舊 `docs.claude.com` 摘要仍寫「only available」於 Opus 4.1、Opus 4、Sonnet 4.5、Sonnet 4、Sonnet 3.7、Haiku 4.5、Haiku 3.5、Haiku 3、Opus 3（deprecated），且隱私段只有 organization。現行平台頁改寫「supported on all active Claude models」，並加上 workspace 隔離與 automatic caching。 — [舊版 FAQ 摘要](https://docs.claude.com/en/docs/build-with-claude/prompt-caching)；[Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 模型例外（現行，不是 2024 清單）：Sonnet 5 沒有 mid-conversation system messages。Opus 4.5+、Sonnet 4.6+、Haiku 5.5 在非 tool-result user 內容出現時仍保留 thinking cache；更早模型會剝掉。Bedrock 上 Claude 3.7 Sonnet 與 3.5 Sonnet v2 的表只列 5 分鐘 TTL。Vertex 明確說 1 小時不支援 3.7 Sonnet、3.5 Sonnet v2、3.5 Sonnet、3 Opus。Haiku 3.5 的 cache read 仍計入 ITPM。Haiku 5.5 有 100,000 token 上下兩種單價，cache read 與 write 計入 prompt length。Read 倍率例外是 Fable 5.1 / Mythos 5.1（0.025 倍）與 Opus 5.5 / Sonnet 5.5（0.05 倍）。 — 見前幾節所列同一批官方頁
- 預熱從社群常用的 `max_tokens: 1` 變成官方的 `max_tokens: 0`。Cache diagnostics 的 `cache-diagnosis-2026-04-07`「no longer required」。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)；[Cache diagnostics](https://platform.claude.com/docs/en/build-with-claude/cache-diagnostics)
- 價格表把 Opus 4.1、Opus 4、Sonnet 4、Haiku 3.5 標成 retired，並註明仍在 Bedrock 或 Google Cloud 的例外。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)

### Inferences
- `anthropic-beta: prompt-caching-2024-07-31` 是已退出必填的啟動 header，不是現行開關。留在 beta enum 不代表還要送。`extended-cache-ttl-2025-04-11` 同樣更像 2025-04 的舊入口；現行契約是 `ttl` 欄位。
- 最低 token 與 read 倍率已經按模型世代拆開，不能再用「Sonnet 1024、Haiku 2048、read 一律 0.1 倍」描述 2026-10 的 API。
- 隔離從舊文件的 organization-only 變成 Claude API 的 workspace，是文件層級的行為變更。生效日未在英文正文核到。

### Gaps
- 沒有官方 changelog 頁給出 automatic caching、`max_tokens: 0`、workspace 隔離、0.05 倍 / 0.025 倍 read 各自的 GA 日期。
- `extended-cache-ttl-2025-04-11` 是否已正式棄用、送了會不會 400，指南沒寫。只知道現行範例不需要它。
- 2024 beta 的最低 token 與當時是否已有 `cache_creation` 拆分，未用當時全文複核。

## 哪些快取是伺服器自動做的，哪些必須呼叫端放 breakpoint？

### Takeaway
Claude API 預設不快取。呼叫端必須放頂層或 block 級 `cache_control`。一旦啟用，伺服器會做三件自動的事：把頂層斷點放在最後一個可快取 block、從 breakpoint 往回找最多 20 個 position 的舊 write、以及在 server tool 迴圈裡自動加一個 5 分鐘斷點。Bedrock 另外對 Claude 提供不放 marker 的 implicit caching，但是 best effort。這都不是獨立的 tail-cache 產品。

### Cited Findings
- Opt-in 句見前：沒有 `cache_control` 就「nothing is cached」。 — [Mid-conversation system messages](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages)
- 頂層 automatic caching 由系統「applies the cache breakpoint to the last cacheable block and moves it forward」。它仍佔 4 個 slot 之一，定價、最低 token、順序與 20-block lookback 與 explicit 相同。Legacy Bedrock（Opus 4.6 及更早）對頂層欄位回 400。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- Lookback 是讀取側自動行為，不是自動寫入：「Cache writes happen only at your breakpoint.」系統不會為 breakpoint 之前的每個 block 各寫一筆。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- Server tools：「When your request has prompt caching enabled and Claude uses a server tool such as web search, web fetch, or code execution, the API automatically places a cache breakpoint on the server tool result before running the next iteration of the agentic loop.」這個自動斷點「always uses the default 5-minute TTL, independent of any TTL you set」，所以全用 `ttl: "1h"` 時 usage 仍可能出現 `ephemeral_5m_input_tokens`。請求若完全沒有 `cache_control`，不會得到這個自動斷點。 — [Tool use with prompt caching](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-use-with-prompt-caching)
- Bedrock implicit：「automatically attempts to reuse eligible prompt prefixes without requiring cache controls」，「best effort」，不保證相同 prompt 一定 hit。Anthropic 模型在 Bedrock 同時支援 implicit 與 explicit。Explicit 的 simplified 模式仍要呼叫端放一個 checkpoint 在靜態內容結尾。 — [Bedrock prompt caching](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)
- Vertex 寫的「Caching automatically occurs when subsequent requests contain the identical text, images, and `cache_control` parameter」指的是已放 `cache_control` 之後的 hit，不是無 marker 的 implicit caching。 — [Vertex prompt caching](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/prompt-caching)
- Message Batches：可以用 prompt caching，但「cache hits are provided on a best-effort basis」，因為批次可能並行且順序不定。建議先用一筆 1 小時 cache 寫入共用 prefix，完成後再送其餘請求。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 文件沒有出現 tail cache、suffix cache、conversation-end cache 這類產品名。與尾端有關的官方句子是 varying suffix 應放在 breakpoint 之外、automatic breakpoint 移到 last cacheable block、以及「incrementally cached」。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)

### Inferences
- 若另一位研究者在找「tail cache」，官方對應物是這三個機制的組合：自動把斷點移到對話最後一個可快取 block、20-position lookback 讓舊的尾端 write 在下一輪變成 hit、以及 server-tool 迴圈裡自動標在 tool result 上的 5 分鐘斷點。三者都仍然是 prefix cache。沒有「只快取最後 N 個 token、前面不算」的計費層。
- Claude API 與 Bedrock 的「automatic」不是同一件事。前者是頂層 `cache_control`。後者的 implicit 可以完全不放 marker，但 hit 不保證，而且 Anthropic 主文說 Opus 4.6 及更早的 legacy Bedrock 不接受頂層 `cache_control`。
- 官方來源沒有把這套機制對比到 vLLM 或 SGLang。本次不引入該對比。

### Gaps
- Server-tool 自動斷點是否也佔用「最多 4 個 breakpoint」的名額，工具頁沒有寫。只寫它一定是 5 分鐘 TTL，且會出現在 `ephemeral_5m_input_tokens`。
- Bedrock implicit caching 是否對所有現行 Claude 模型開啟、最低 token 是否等於 explicit 表，userguide 只說 Anthropic 模型兩者都支援，沒有逐列 implicit 的失敗條件。
