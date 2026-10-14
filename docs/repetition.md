# 生成重複檢查

日期：2026-10-15

berth 在抽樣之後檢查已經接受的 token。發現重複，就在串流裡送標記。呼叫端決定要不要中止這一次，再送一筆新的請求。

vLLM、llama.cpp、ollama 不會送這種標記。檢查做在這個引擎裡，呼叫端才看得到思考階段的迴圈。

開關和事件的名字用 berth 自己的。不必照 Grok 的 `x-grok-doom-loop-check`。對照表在文末，給之後一起分析用。第一版只認本文件的名字。

## 引擎做的事

每個已經接受的 token 進該頻道的緩衝區。引擎立刻看這個緩衝區。

思考區的 token 在寫出來的當下就檢查。引擎不等回答開始，也不等這一次生成結束。

第一次發現時，引擎在串流中途送事件。之後同一筆請求又有新標記，事件再送一次，裡面是累積的整組標記。生成結束時，同一組標記再附在完成事件上。沒有發現，就不送空事件。

請求沒帶某個開關，該檢查不跑。

## 引擎不做的事

引擎不重抽。重抽是呼叫端把連線中止，再送一筆新請求。Grok 客戶端預設最多重抽 2 次；那是呼叫端的預算，不是引擎的停止條件。預算用完之後，引擎仍要照實送標記。

引擎不看工具名稱和參數。同一次回覆裡工具名稱加參數完全相同，是呼叫端的 `IdenticalToolCallRun`：先提醒，再停止。

引擎不判斷目標有沒有進展。那是呼叫端的 `GoalTracker`，沒有進展就暫停。

引擎不另外叫一個模型來審查很長的思考。那是呼叫端的 `loop_review`，預設關。它是旁邊的決策呼叫，不是這份檢查。

重複懲罰仍是抽樣前對 logits 的 scatter，見 [runtime.md](runtime.md)。它改的是下一個 token 的分數。這份檢查看的是已經寫出的 token。兩件事都在 host，都不進 forward，也不進 CUDA graph。

## 放在哪

抽樣器每接受一個 token，就把它交給檢查。被退回的草稿 token 不進緩衝區。同一塊草稿裡，位置 i 已經接受的 token 要看得到，這和重複懲罰的可見範圍相同。

檢查要知道這個 token 屬於哪個頻道。頻道是生成狀態，不是從文字猜的。

- `thinking`：模型給自己看的推理 token
- `response`：使用者看得到的回答 token

沒有思考頻道的模型，接受的 token 都算 `response`。

兩個頻道各有一個緩衝區。一段重複不要跨頻道連起來算。

## 兩種檢查

### 尾端重複

開關：`x-berth-tail-check`

值是十進位的回看長度，單位是 token。沒帶這個開關，尾端檢查不跑。

提案預設 1024。呼叫端送出的值落在 512 到 4096 就用它。落在外面，引擎用 4096。這和 Grok 客戶端對視窗的夾法相同，只是名字不同。

引擎只看該頻道最近這一段。尾巴同一小段不斷出現時，送出：

```text
tail_repetition:<t>@<channel>
```

`t` 是整數。同一筆生成可以同時有好幾個 `t`。數字越小，重複越密。Grok 客戶端把較小的 `t` 當成較強的證據，並把每一個命中的 `t` 都送出來。

`t` 的比較公式，Grok 客戶端沒有寫。客戶端把標記當成不透明字串。berth 要自己定公式。未定的部分見文末，不要在公式定下來之前假裝 `t` 就是週期。

### 原樣重複

開關：`x-berth-exact-repeat-check`

值是最短長度，單位是 token。沒帶這個開關，原樣檢查不跑。提案預設 64。

同一串 token 要原樣、連續出現，而且長度不短於這個值，引擎才報。標記：

```text
exact_repetition:<長度>x<次數>@<channel>
```

`<長度>` 是這一串有幾個 token。`<次數>` 是它連續出現幾次。例如 `exact_repetition:64x3@thinking`：64 個 token 的那一串，在思考區連續出現 3 次。

Grok 客戶端的註解寫的是這個意思：an exact token sequence repeated consecutively。這一種比尾端重複明確。

## 低樣式

Grok 的尾端開關同時涵蓋一個低樣式檢查。標記是 `low_logprob@<channel>`。客戶端註解寫的是 degenerate low-entropy generation，也就是這一段 token 的種類變太少。它和尾端重複共用回看長度。公式同樣不在客戶端裡。

berth 第一版不做這個檢查，也不送替代名字。分析時要決定做不做，再決定標記叫 `low_diversity` 還是沿用 `low_logprob`。

## 串流

中途事件的提案：

```text
event: response.repetition_check
data: {"type":"response.repetition_check","repetition_check":{"triggers":["tail_repetition:4@thinking"]}}
```

下一次發現再送，`triggers` 是累積的整組，不是只有新的那條。

完成時附在同一筆回應上：

```text
"repetition_check": {"triggers": ["tail_repetition:4@thinking", "tail_repetition:2@thinking"]}
```

`response.completed` 和 `response.incomplete` 都要能帶這欄。沒有發現就省略這欄。

標記只放標籤，不放 token 原文。單條標籤長度上限先定 256 byte。一筆請求最多留 64 條，超出的丟掉。這兩個上限對齊 Grok 客戶端的收集器，避免標記本身無界。

## 呼叫端可以怎麼用

這是 Grok 客戶端現在的政策，不是 berth 引擎裡的分支。寫出來是為了讓標記夠用。

客戶端只對一種標記中止並重抽：`tail_repetition:<t>@thinking`，而且 `t` 在 2 到 64。預設最多重抽 2 次。`t` 的上限是客戶端自己的設定，不放進請求。

下面三種只記錄：

- `exact_repetition`，不論在哪個頻道
- 回答頻道上的 `tail_repetition`
- `low_logprob`

中止發生在中途事件之後的下一筆串流事件。若標記一直到完成事件才出現，客戶端仍拒絕這一次結果，再重抽。思考區的尾端重複因此抓得到，不必等到回答開始。

舊的客戶端自行掃描 token 的檢查已經拿掉。回報欄位留空。註解是 server-detected, client-resampled。

berth 自己的呼叫端可以採不同政策。引擎的合約只到「送對標記」。

## Grok 對照

客戶端原始碼在這台機器的 grok-build。偵測公式不在那裡。

- `crates/codegen/xai-grok-sampling-types/src/doom_loop.rs`：標記文法和開關常數
- `crates/codegen/xai-grok-config/src/remote_settings.rs`：視窗、門檻、重抽次數的夾法
- `crates/codegen/xai-grok-sampler/src/doom_loop.rs`：收集器
- `crates/codegen/xai-grok-sampler/src/stream/responses.rs`：中途中止

| 角色 | berth 提案 | Grok 客戶端現名 |
|---|---|---|
| 尾端檢查的開關 | `x-berth-tail-check` | `x-grok-doom-loop-check` |
| 原樣檢查的開關 | `x-berth-exact-repeat-check` | `x-grok-exact-repetition-check` |
| 中途事件 | `response.repetition_check` | `response.doom_loop_check` |
| 完成事件上的欄位 | `repetition_check` | `doom_loop_check` |
| 尾端標記 | `tail_repetition:<t>@<channel>` | 同左 |
| 原樣標記 | `exact_repetition:<長度>x<次數>@<channel>` | 同左 |
| 低樣式標記 | 第一版不送 | `low_logprob@<channel>` |

Grok 客戶端只有在回復政策打開時才送那兩個開關。開關不在，伺服器不報。berth 同樣把「沒帶開關」當成關。

現有的 Grok 客戶端只認它自己的名字。berth 用上表左欄時，那個客戶端不會重抽。要讓它重抽，之後再加一層名字相容，或者改客戶端。第一版不加相容層。

## 待分析

1. `t` 到底量什麼。觀察只有三條：它是整數；較小代表較密；同一次生成可以同時出現多個值。線上樣本同時有 `tail_repetition:4@response` 和 `tail_repetition:2@response`。假設是「`t` 是尾巴的週期，短週期成立時長週期也成立」。這只是假設。
2. 低樣式要不要做。不做的話，種類變少但字面不完全相同的思考迴圈，第一版抓不到。
3. 原樣重複的「連續」要不要允許中間夾一個不同 token。客戶端註解寫的是 consecutively，提案不允許夾。
4. 回看長度是只數該頻道的 token，還是數這一次生成的全部 token。提案是各頻道各算。伺服器原始碼沒有證實這一點。
5. 名字是否維持上表左欄。
