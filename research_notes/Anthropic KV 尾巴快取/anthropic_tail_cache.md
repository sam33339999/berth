# Anthropic「尾巴快取」與對話末端的 cache hit

截至 2026-10-14，Anthropic 官方沒有名為 tail cache、suffix cache、ephemeral tail、尾巴快取、尾部快取或尾快取的功能。搜尋官方 prompt caching 文件、Claude Code 文件、cookbook，以及中文技術文，都沒有把「對話末端」做成一套獨立於 prefix 的 KV。最接近的官方機制仍是 prompt caching：在 prefix 上放 `cache_control` breakpoint（含 automatic caching 自動把斷點移到最後一個可快取 block）。社群說的「尾巴」多半是這個移動寫入點的暱稱，不是第二個 cache。

官方行為、社群暱稱、推論分層寫在各節。推論只出現在 Inferences。

## 「tail cache / suffix cache / 尾巴快取」這個名字存不存在？

### Takeaway

不存在這個官方名稱。官方名稱是 prompt caching，實作是 prefix hash，加上 explicit breakpoint 或頂層 automatic caching。中英文社群用 moving tail checkpoint、rolling breakpoint、messages-tail、動態後綴來描述「把斷點打在不斷變長的對話末端」，指的是同一套 prefix cache 的擺放位置。

### Cited Findings

- 官方 prompt caching 頁只定義兩種啟用方式。Automatic caching：請求頂層一個 `cache_control`，「The system automatically applies the cache breakpoint to the last cacheable block and moves it forward as conversations grow.」Explicit cache breakpoints：把 `cache_control` 放在個別 content block。沒有 tail、suffix、rolling 或 conversation cache 這類第二機制的欄位。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 快取對象是整段 prefix，不是末端片段。「Prompt caching references the entire prompt: `tools`, `system`, and `messages` (in that order), up to and including the block designated with `cache_control`.」 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 唯一的 cache type 欄位值是 `ephemeral`。預設 TTL 5 分鐘；`{"type": "ephemeral", "ttl": "1h"}` 為 1 小時。寫入只發生在 breakpoint。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 以「尾巴快取」「尾部快取」「尾快取」「尾巴 cache」搜尋，沒有對上 Anthropic 產品頁或文件標題。最接近的中文工程用語來自 AWS 中國部落格，談的是 Bedrock 上自行擺放的「單 moving tail checkpoint／單尾部 checkpoint／當前請求尾部 moving checkpoint」，不是 Anthropic API 欄位。 — [在 Amazon Bedrock 上为 Claude 应用设计稳健的 Prompt Cache 策略](https://aws.amazon.com/cn/blogs/china/amazon-bedrock-claude-application-design-prompt-cache-policy/)
- 知乎文章把請求拆成「靜態前綴」與「動態後綴」（使用者訊息、助手回應、工具輸出），並說啟用自動快取後「緩存斷點會隨著對話的增長自動推進」。用的是後綴／斷點，不是尾巴快取這個產品名。 — [LLM中的提示缓存！Claude如何实现92%缓存命中率的案例研究](https://zhuanlan.zhihu.com/p/2017269864117609145)
- V2EX 貼文把歷史訊息稱為 append-only 段，並寫「它每輪都在長尾部」。這是作者自己的分段，不是官方名詞。 — [Harness 工程分享：我們怎麼把 LLM Agent 的 prompt cache 命中率做到 90%+](https://www.v2ex.com/t/1212780)
- Claude Code GitHub issue #58103 提案 `settings.json` 的 `"at": "messagesTail"`。作者明說 breakpoint 放置是內部的、使用者無法設定，`cachingPolicy` 並不存在。這是功能請求裡的假想鍵名。 — [Feature request: user-configurable cache breakpoint hierarchy](https://github.com/anthropics/claude-code/issues/58103)
- 2026-10-13 的 Medium 文用 “tail rent” 指長 session 每一輪重送成長中的對話、即使 cache read 仍累積的費用，不是一種 cache 實作。 — [Where the Tokens Actually Go](https://pub.towardsai.net/where-the-tokens-actually-go-cache-rent-effort-levels-and-routing-in-a-long-claude-code-session-4d1d2a1625df)

### Inferences

- 使用者聽到的「尾巴 cache」最可能是這三件事的混稱：（1）automatic caching 把 breakpoint 推到最後一個可快取 block；（2）呼叫端自己在最後一則 user／tool_result 上打 `cache_control`；（3）AWS／社群文章裡的 moving tail checkpoint。三者都還是 prefix 寫入點，不是末端獨立 KV。

### Gaps

- 沒有搜到 Anthropic 員工把內部 scheduler 稱作 tail cache 或 suffix cache 的貼文或演講逐字稿。Code w/ Claude 2026「Caching, harnesses, and advisors」場次頁面沒有公開逐字稿，無法核對口語用詞。 — [Session page](https://claude.com/code-with-claude/session/sf-caching-harnesses-and-advisors-building-on-claude-at-github-scale)

## 除了 prefix，有沒有第二套快取（最新訊息、workspace tail、prefix 變了仍能命中）？

### Takeaway

沒有。命中條件是從請求開頭到某個已寫入 breakpoint 的 prefix 位元組完全相同。改 tools 或 system 會讓其後全部失效。多個 breakpoint 只是多個由短到長的 prefix 條目，較短的仍可在後面內容變動時命中，但末端內容不能在前面 prefix 已變時單獨命中。

### Cited Findings

- 寫入規則：「Cache writes happen only at your breakpoint. Marking a block with `cache_control` writes exactly one cache entry: a hash of the prefix ending at that block. The system does not write entries for any earlier position.」hash 是累積的，breakpoint 之前任一 block 改變，下一次該點的 hash 就不同。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 讀取規則：先算 breakpoint 處的 prefix hash；沒有條目就逐 block 往回找「先前請求已經寫過的條目」，不是找「看起來穩定的內容」。每個 breakpoint 最多查 20 個 position，breakpoint 自己算第一個。視窗內沒有就停，或改從下一個 explicit breakpoint 再查。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- Claude API 上，連續的 `tool_use` block 算一個 position，連續的 `tool_result` block 也算一個 position。因此單輪大量平行工具呼叫不會單靠 block 數量把上一筆寫入擠出視窗。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 失效階層是 `tools` → `system` → `messages`。「Changes at each level invalidate that level and all subsequent levels.」改 tool definitions 使整段失效；改 `tool_choice`、增刪圖片只使 messages cache 失效，tools 與 system 仍可命中。這是較短 prefix 還在，不是末端獨立 cache。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 官方對「breakpoint 打在每輪都變的最後一塊」的警告：lookback 不會把 breakpoint 後面的穩定內容找出來快取。automatic caching 會踩同一個陷阱，因為它把斷點放在最後一個可快取 block。有每輪變化的 suffix（timestamp、本次使用者訊息）時，斷點要放在靜態 prefix 的最後一塊，而不是那塊會變的內容上。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- Claude Code 文件寫明：「The API caches by matching the start of each request, called the prefix... The match is exact, so a change anywhere in the prefix recomputes everything after it. There is no per-file or per-segment caching.」正常一輪的 prefix 就是上一整個請求，只有最新一輪是新的。 — [How Claude Code uses prompt caching](https://code.claude.com/docs/en/prompt-caching)
- 隔離範圍：KV 表示與內容的 cryptographic hash 只在記憶體、不落盤。條目至少活 5 分鐘或 1 小時，之後儘快刪除。組織之間隔離；在 Claude API、Claude Platform on AWS、Microsoft Foundry 上，同一組織的 workspace 之間也隔離。相同 prompt 不會跨組織命中。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 舊的 docs.claude.com FAQ 曾寫 cache key「using a cryptographic hash of the prompts up to the cache control point」，且「Caches are organization-specific」。與現行「workspace 也隔離」相比，舊文沒有 workspace 這一句。以 2026-10 的 platform 頁為準。 — [舊版 FAQ 鏡像](https://docs.claude.com/en/docs/build-with-claude/prompt-caching)；[現行 Data retention](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- SDK issue #1547 曾回報：只在 system block 放 explicit breakpoint 時，伺服器仍對 user 內容多寫一筆 cache write。Anthropic 回覆這是伺服器端 bug，已修；修後 warm call 的 `cache_creation` 為 0，與「writes happen only at your breakpoint」一致。這否定了「斷點之後還有隱藏尾端寫入」的現行行為。 — [anthropic-sdk-python#1547](https://github.com/anthropics/anthropic-sdk-python/issues/1547)

### Inferences

- 「prefix 改了、最新幾則訊息仍命中」與文件矛盾。若 tools／system 變了，messages 的 hash 依賴前面內容，不可能單獨命中。多 breakpoint 能保住的是變更點之前的 prefix。

### Gaps

- 沒有官方數字說明 workspace 內不同 API key 是否共享同一 cache 條目。文件只寫到 organization，以及 Claude API 等平台上的 workspace 隔離。

## 最後一個 breakpoint 要怎麼放，下一輪才會命中整段先前對話？社群的「尾巴」是不是這個？

### Takeaway

是。下一輪要命中「到上一輪為止的整段對話」，本輪必須在「下一輪仍會原樣重送」的最後一個 block 上留下寫入。成長中的對話把 `cache_control` 放在該請求最後一個 block 即可，前提是每一輪新增少於 20 個 position，lookback 才找得到上一筆寫入。這就是社群說的尾部／rolling breakpoint，不是另一個 cache。

### Cited Findings

- 官方 takeaway 原文：「Place `cache_control` on the last block whose prefix is identical across the requests you want to share a cache. In a growing conversation the final block works as long as each turn adds fewer than 20 blocks: earlier content never changes, so the next request's lookback finds the prior write.」 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- Automatic caching 的多輪表：Request 1 把 breakpoint 放在 User(2)，整段寫入。Request 2 讀到 User(2)，只把 Asst(2)+User(3) 寫入。Request 3 讀到 User(3)，只寫 Asst(3)+User(4)。「The cache breakpoint automatically moves to the last cacheable block in each request.」 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- Lookback 範例：Turn 1 在 block 10 寫入。Turn 2 breakpoint 在 block 15，往回找到 block 10，只重算 11–15，並在 15 寫新條目。Turn 3 breakpoint 在 block 35，只查 35 到 16 共 20 個 position，block 15 在窗外，整段 miss。要在 block 15 再放一個 breakpoint，才會開第二個 lookback 視窗。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 「The lookback can only find entries that earlier requests already wrote. If a growing conversation pushes your breakpoint 20 or more blocks past the last write, the lookback window misses it. Add a second breakpoint closer to that position from the start so a write accumulates there before you need it.」最多 4 個 breakpoint；第 5 個 explicit breakpoint 會 400。頂層 automatic caching 佔用其中一格。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 計費空間：`cache_read_input_tokens` 是 breakpoint 之前已快取的 token；`cache_creation_input_tokens` 是 breakpoint 之前這次寫入的 token；`input_tokens` 只計最後一個 breakpoint 之後、不進 cache 的 token。`total_input_tokens = cache_read_input_tokens + cache_creation_input_tokens + input_tokens`。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- TTL 從「寫入或讀取該條目的請求開始」算，不是從回應結束算。若串流 4 分鐘，下一次要重用同一 prefix 的請求必須在該回應完成後約 1 分鐘內開始，才還在 5 分鐘窗內。命中會免費刷新 TTL。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 短於模型最小長度不會快取，也不報錯。現行 Claude API 等平台：Fable 5.1、Mythos 5.1、Opus 5.5、Opus 5、Sonnet 5.5、Fable 5、Mythos 5、Haiku 5.5 為 512；Mythos Preview 與 Opus 4.7 為 2048；Opus 4.6 與 Opus 4.5 為 4096；Opus 4.8、Sonnet 5、Sonnet 4.6、Sonnet 4.5 等為 1024；Haiku 4.5 為 4096。兩邊 usage 計數都是 0 就代表沒進 cache。Bedrock 最小長度以 AWS 文件為準。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- Cookbook 的 explicit 範例把 `cache_control` 只放在最後一則 user message 的 text block 上，註解是 “Return messages with cache_control on the last user message.” Automatic 範例則只在 `messages.create` 頂層加 `cache_control`，不手動移動。 — [Prompt caching through the Claude API](https://platform.claude.com/cookbook/misc-prompt-caching)
- AWS 中文文把同一策略叫做「單 moving tail checkpoint」：清掉歷史上的舊 cachePoint，在當前最後一個 user message 或 toolResult 末尾放新的 cachePoint。並寫「長對話本身不是問題。某一輪突然新增過多 content blocks，才是單尾部 checkpoint 的主要風險。」穩健預設是三個 checkpoint：system 末尾、tools 末尾、當前請求尾部 moving checkpoint。這是 Bedrock 呼叫端策略，用的是 cache checkpoint，不是 Anthropic 的 tail cache 產品。 — [AWS 中國部落格](https://aws.amazon.com/cn/blogs/china/amazon-bedrock-claude-application-design-prompt-cache-policy/)
- 舊鏡像把 lookback 寫成「up to 20 blocks before each explicit breakpoint」，並建議 “Always set an explicit cache breakpoint at the end of your conversation”。現行平台頁改成「最多 20 個 position，含 breakpoint 自己」，並加上連續 tool block 合併計數。兩份都支持「斷點打在對話末端」，計數細節以現行頁為準。 — [舊版 Prompt caching](https://docs.claude.com/en/docs/build-with-claude/prompt-caching)；[現行 Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)

### Inferences

- 社群「尾巴 cache 管理命中」描述的是：本輪在末端寫一筆 prefix，下一輪 lookback 從新的末端往回 20 個 position 撞上這一筆，於是整段舊對話以 `cache_read_input_tokens` 計費，只有新增長的一段是 write 或 uncached。命中的是舊 prefix，不是一套尾端索引。

### Gaps

- 官方沒有公開「20 個 position」裡 system／tools 的 block 是否與 messages 用同一套計數，只給了 messages 裡 tool_use／tool_result 的合併規則，以及 10／15／35 block 的教學例。

## Claude Code、Agent SDK、Messages API 會不會自動在訊息列表尾端插 breakpoint？

### Takeaway

Messages API 不會在沒有 `cache_control` 時自動快取。有頂層 `cache_control` 時，伺服器把斷點放在最後一個合格 block，並隨對話前移。另外，已啟用 prompt caching 時，server tool 的結果會在同一請求的下一輪 agentic loop 之前自動被加上 5 分鐘 breakpoint。Claude Code 自己送 `cache_control`，並在對話中途的 system context block 上做標記；gateway 拒絕時會把標記挪到最後一則對話訊息。

### Cited Findings

- 頂層 automatic caching 的斷點目標是最後一個可快取 block。邊界：最後一塊已有相同 TTL 的 explicit `cache_control` 則 automatic 是 no-op；TTL 不同則 400；已經有 4 個 explicit breakpoint 則 400；最後一塊不合格就默默往前找到最近的合格 block，找不到就跳過快取。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- Server tool：「When your request has prompt caching enabled and Claude uses a server tool such as web search, web fetch, or code execution, the API automatically places a cache breakpoint on the server tool result before running the next iteration of the agentic loop.」這個自動斷點一律是預設 5 分鐘 TTL，與呼叫端設的 1 小時 TTL 無關，寫入出現在 `cache_creation.ephemeral_5m_input_tokens`。請求裡至少要已有一個 `cache_control`，否則不會加。 — [Tool use with prompt caching](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-use-with-prompt-caching)
- Thinking block 不能直接標 `cache_control`，但出現在先前 assistant turn、並隨後續請求（常見是帶回 tool result）重送時，會跟其他內容一起進 prefix。文件並寫，在帶 tool result 的後續呼叫裡，thinking「get cached as part of the request content」，且「This caching behavior occurs even without explicit `cache_control` markers」。此句與「沒有 `cache_control` 就完全不快取」並陳，適用範圍是 thinking 被原樣帶回、且模型保留 thinking 的路徑，不是一條通用的隱藏 tail cache。Opus 4.5+、Sonnet 4.6+、Haiku 5.5 預設保留 thinking，非 tool-result 的新 user 內容不會因此拆掉舊 thinking；更早的 Opus／Sonnet 與 Haiku 4.5 及之前會剝掉，messages cache 從那些 thinking 之後失效。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- Claude Code：「Claude Code handles prompt caching for you, unless you disable it.」它重送 system prompt、專案上下文、先前訊息與工具結果，新內容接在末端。層級由穩到變：system prompt（核心指示與工具定義）、project context（CLAUDE.md、auto memory、未限定範圍的規則）、conversation。改對話層仍可命中前兩層；改 system prompt 則全部重算。 — [How Claude Code uses prompt caching](https://code.claude.com/docs/en/prompt-caching)
- Claude Code「appends system context mid-conversation, such as file-change notices, and marks that block for caching」；設 `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS` 則該 block 不快取。Gateway 若回 400 並點名 `cache_control`，Claude Code 會重送，把標記從該 block 挪到「your last conversation message」，並在該對話之後保持這個位置。對話仍快取，被挪走的 block 以未快取 input 計費。 — [How Claude Code uses prompt caching](https://code.claude.com/docs/en/prompt-caching)
- Advisor 工具是例外：「its definition sits after the cache breakpoint, so enabling or disabling `/advisor` keeps the cached prefix intact.」這表示 Claude Code 的主 breakpoint 不是永遠黏在整個請求的絕對最後一個位元組；至少 advisor 定義被放在斷點之後。 — [How Claude Code uses prompt caching](https://code.claude.com/docs/en/prompt-caching)
- Plan mode 與 skill 把指示當成對話訊息附加，所以已快取的 prefix 不變。檔案變更不回寫舊的 read，而是附加 `<system-reminder>`。 — [How Claude Code uses prompt caching](https://code.claude.com/docs/en/prompt-caching)
- TTL 由 Claude Code 依請求分桶。主對話用 `promptCacheTtl` 或 `CLAUDE_CODE_PROMPT_CACHE_TTL`（`5m` 或 `1h`）；其餘用 `subagentPromptCacheTtl` 或 `CLAUDE_CODE_SUBAGENT_PROMPT_CACHE_TTL`。未自訂時，訂閱方案內用量的主對話要 1 小時；超出後改扣 usage credits 則主對話降為 5 分鐘。API key 預設主對話 5 分鐘，要 1 小時須自行設定。`usage.cache_creation.ephemeral_1h_input_tokens` 與 `ephemeral_5m_input_tokens` 可區分。 — [How Claude Code uses prompt caching](https://code.claude.com/docs/en/prompt-caching)
- 關閉開關：`DISABLE_PROMPT_CACHING`，以及依 alias 的 `DISABLE_PROMPT_CACHING_HAIKU`／`SONNET`／`OPUS`／`FABLE`。Subagent 有自己的 system prompt 與工具集，第一個請求讀不到 parent cache。 — [How Claude Code uses prompt caching](https://code.claude.com/docs/en/prompt-caching)
- Agent SDK 沒有獨立的 tail-cache API。Claude Code 文件把 SDK 指到同一套 cache token 與 TTL 設定。 — [How Claude Code uses prompt caching](https://code.claude.com/docs/en/prompt-caching)
- 中途 system 訊息：把 `{"role": "system"}` 附加在 `messages` 末端，不要改頂層 `system`，已快取 prefix 仍匹配。頂層 automatic caching 在選斷點時會跳過 turn-scoped system 訊息。清掉 turn-scoped 訊息的那一次請求，可重用的 prefix 停在它前面的 user turn，中間那一個 assistant turn 會重算。 — [Mid-conversation system messages](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages)

### Inferences

- Claude Code「自動處理 caching」是客戶端放置 `cache_control` 加上 API 的 prefix 匹配，不是伺服器在 prefix 改變後仍命中一段 session tail。Gateway fallback「標記改放到最後一則對話訊息」是文件裡最明確的「尾端 breakpoint」行為，而且只在 gateway 拒絕原本標記時發生。

### Gaps

- 公開文件沒有列出 Claude Code 預設的四個 breakpoint 各自落在哪個 block。Issue #58103 也說放置不透明。因此不能斷言每一輪都一定標在最後一則 user 或 assistant 上；能斷言的是 file-change notice 會被標記，advisor 定義在斷點之後，以及 gateway 400 時改標到最後一則對話訊息。

## Context editing、compaction、memory、clear_tool_uses／clear_thinking 是為了保住尾端命中嗎？它們使什麼失效？

### Takeaway

這些功能是在刪或摘要歷史，不是在維持一個尾端 cache。它們會從被改寫的那個點打斷 prefix。設計目標是：能命中的前綴盡量不要動，摘要請求用同一 prefix 再附加一則新訊息（cache-safe fork），清工具結果時一次清夠多，讓隨後的新 prefix 值得重新寫入。

### Cited Findings

- Context editing 官方定位是長對話的細粒度清理；多數情況應優先用 server-side compaction。策略是 `clear_tool_uses_20250919` 與 `clear_thinking_20251015`，beta header `context-management-2025-06-27`。編輯在伺服器、進模型之前發生；客戶端仍保留完整未改歷史。 — [Context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing)
- 與 prompt caching 的交互，官方原文：Tool result clearing「Invalidates cached prompt prefixes when content is cleared.」建議用 `clear_at_least` 清到值得打斷 cache 的量。當次付出 cache write，其後請求可重用新的 prefix。Thinking block clearing：思考塊留在上下文時「the prompt cache is preserved」；被清掉時「the cache is invalidated at the point where clearing occurs.」 — [Context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing)
- `clear_tool_uses_20250919` 依時間順序清最舊的 tool result，換成 placeholder。預設只清結果；`clear_tool_inputs: true` 連 tool call 參數一起清。範例欄位：`trigger`（`input_tokens`）、`keep`（`tool_uses`）、`clear_at_least`、`exclude_tools`。 — [Context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing)
- Claude Code 的 `/compact`「replaces your message history with a summary」，因此對話層 prefix 對不上，這是設計如此。System prompt 層仍重用，除非這次是「resume 時仍帶著原本會變的舊 system prompt」，第一次 compact 才會換成現行 prompt 並重建該層。專案上下文從磁碟重載，只有 CLAUDE.md 與 memory 與 session 開始時相同才命中。 — [How Claude Code uses prompt caching](https://code.claude.com/docs/en/prompt-caching)
- Compact 的摘要請求本身是 cache-safe 的讀取：同一 system prompt、tools、history，把摘要指示當成最後一則 user message 附加。Cache 還熱時，這次請求讀 prefix，費用主要在生成摘要。超過 TTL 之後沒有 cache，摘要請求把整段歷史當未快取 input。Compact 之後下一輪只為短摘要重建對話 cache。 — [How Claude Code uses prompt caching](https://code.claude.com/docs/en/prompt-caching)
- Anthropic 工程文把同一做法稱為 cache-safe forking：「use the exact same system prompt, user context, system context, and tool definitions as the parent conversation. We prepend the parent's conversation messages, then append the compaction prompt as a new user message at the end.」另開一支「summarize this」、不帶 tools 的請求會對不上 prefix，這是他們點名的成本陷阱。 — [Lessons from building Claude Code: Prompt caching is everything](https://claude.com/blog/lessons-from-building-claude-code-prompt-caching-is-everything)
- `/rewind` 把對話截回較早一輪。剩下的歷史就是當時建 cache 的內容，system 與 project context 不變，所以命中較早的條目。之後每一輪都讀過那個 prefix，所以即使原輪已超過 TTL，條目仍被刷新。`/recap` 只把摘要附加為指令輸出，不取代歷史，prefix 保持。 — [How Claude Code uses prompt caching](https://code.claude.com/docs/en/prompt-caching)
- 在 1 小時 TTL 上閒置時，Claude Code 可能在 cache 到期前自動 compact。`idleCompaction: false` 可關（v2.1.290+）。 — [How Claude Code uses prompt caching](https://code.claude.com/docs/en/prompt-caching)
- 移除一批過舊圖片／PDF 會改到那些訊息，下一請求從最早被改的訊息起重算對話。 — [How Claude Code uses prompt caching](https://code.claude.com/docs/en/prompt-caching)
- Memory：工具表把 Memory 與 Text editor、Bash 同列為 standard client tool，擷取到的文字沒有給它專用的 cache 規則。Claude Code 的 auto memory 在 project context 層，於 session 開始或 `/clear`、`/compact` 之後才換；session 中途改 CLAUDE.md 不打破當下 cache，但也不生效。 — [Tool use with prompt caching](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-use-with-prompt-caching)；[How Claude Code uses prompt caching](https://code.claude.com/docs/en/prompt-caching)
- 知乎文的「緩存安全分叉」與官方 cache-safe forking 相同：壓縮時不截斷 prefix，保持 system、tools、對話結構，把壓縮請求附加成新訊息。 — [LLM中的提示缓存](https://zhuanlan.zhihu.com/p/2017269864117609145)

### Inferences

- `clear_at_least` 與 cache-safe fork 都是在承認 prefix 命中很脆：中間一改，從該點到尾端都要重寫。它們不是為了讓「尾端」在 prefix 變化後繼續命中，而是為了少改、或一次改完再建立新 prefix。

### Gaps

- 沒有讀到 Memory tool（`memory` 這個 server／client tool）自身的讀寫是否插入會改 hash 的系統區塊。只能確認工具表未把它列成會單獨使 system cache 失效的開關，以及 Claude Code auto memory 屬於 project context 層。

## 內部 KV 的命中怎麼算？有沒有不是純 prefix hash 的部分？

### Takeaway

官方公開的命中就是累積 prefix 的 cryptographic hash，再加 20-position lookback 去找先前 breakpoint 寫下的條目。沒有官方文件描述 radix tree、block pool 或 session KV 可以在 prefix 不同時命中對話尾端。Cache diagnostics 的 fingerprint 是除錯用的 hash 比較，文件寫明與 cache 是否命中無關。

### Cited Findings

- 現行頁：breakpoint 寫入「a hash of the prefix ending at that block」；讀取先比對該 hash，再往回找已寫入的條目。隔離與保留見 Data retention：記憶體中的 KV 與 cryptographic hashes，組織與（指定平台上）workspace 隔離。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 舊 FAQ 同一機制的說法：「Cache keys are generated using a cryptographic hash of the prompts up to the cache control point. This means only requests with identical prompts can access a specific cache.」 — [舊版 Prompt caching](https://docs.claude.com/en/docs/build-with-claude/prompt-caching)
- Cache diagnostics（beta）：帶 header 時 API 依回應 `id` 存輕量 fingerprint。下次請求帶上一個 `id`，API 重建 fingerprint 並回傳第一個分歧點。Fingerprint 只有 hash 與 token 估計，不含原文，範圍是組織與 workspace。「The comparison is about request structure, independent of whether the cache actually hit.」它不是第二條命中路徑。 — [Cache diagnostics](https://platform.claude.com/docs/en/build-with-claude/cache-diagnostics)
- 模型是 cache 的一部分。Claude Code：換模型即使內容相同也整段重算。多數模型換 effort 也整段重算；Opus 5.5、Sonnet 5.5、Haiku 5.5、Fable 5.1 在 API key 或 Claude 訂閱上，換 effort 預設保留 cache。Fast mode 的請求 header 屬於 cache key，第一次打開會整段 miss；之後只改速度設定則速度不在 cache key 裡。 — [How Claude Code uses prompt caching](https://code.claude.com/docs/en/prompt-caching)
- API 失效表補充：`speed: "fast"` 與標準速度切換使 system 與 messages 失效。`output_config.effort` 一定使 messages 失效；明確設成該模型預設值等於省略，不失效。支援 per-message effort 時，把 effort 放在 `messages` 裡的 `role: "system"` 訊息，cached prefix 保持。`inline-tools-2026-09-15` 可用 `tool_addition` 在中途加工具而不改 `tools` 陣列，prefix 仍匹配；例外是 `tools` 裡還沒有任何 non-deferred tool 時，第一次這樣定義會整段 miss 一次。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 並行請求：cache 條目要等第一個回應開始之後才可用。 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- 第三方對 vLLM／SGLang radix、固定 block hash 的說明，不是 Anthropic 文件。Morph 文自己也說「No vendor doc explains why the match has to be a prefix」，接著才描述開源引擎。不能把它寫成 Anthropic 內部實作。 — [Prompt Caching: How It Works](https://www.morphllm.com/prompt-caching)

### Inferences

- 20-position lookback 是唯一偏離「只比對你標的那一個 hash」的官方規則，但它仍只接受某個更短 prefix 的 hash 命中，而且該 hash 必須是更早的 breakpoint 寫過的。它不是 suffix match，也不能在前綴不同時撿回對話尾端。
- 推論：若 Anthropic 內部用 block 或 radix 實作，對外契約仍被收斂成「只在 breakpoint 寫入、只按累積 prefix 命中」。公開材料不足以描述 block 大小或多租戶 pool。

### Gaps

- 沒有 Anthropic 論文或員工帖公開 radix、block size、session pool 或多租戶 KV 的命中演算法。Code w/ Claude 2026 該場沒有可用逐字稿。YouTube 對 radix attention 的講解是開源引擎，不是 Anthropic 架構說明。
