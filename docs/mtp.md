# MTP

日期：2026-10-15。這份是 27B 上可設定草稿長度的 MTP speculative decoding 設計。後面的實作照這份寫。對 fork `~/qwen38-exl3-dflash2/exllamav3` 逐位元一致。沒開 MTP 的路徑（`max_history = 0`、現有 kernel 參數、cubin 類別、block table 寬度）保持不動。

fork 路徑相對 `~/qwen38-exl3-dflash2/exllamav3/exllamav3/`。ext 在 `exllamav3_ext/`。行號對 tree `de35ffb`。

27B 文字塔：64 層，每 3 層 GDN 接 1 層 full attention，第 63 層是 full attention。`hidden = 5120`，`vocab = 248320`，`intermediate = 17408`，`head_dim = 256`，`num_q_heads = 24`，`num_kv_heads = 4`（group 6）。GDN：`linear_conv_kernel_dim = 4`，`linear_num_key_heads = 16`，`linear_num_value_heads = 48`，key/value head dim 128，`fdim_qkv = 10240`。`rms_norm_eps = 1e-6`。`partial_rotary_factor = 0.25`，`rope_theta = 1e7`，mrope sections `[11, 11, 10]`。`interleaved_gate`，gate 是 swish，所以 `BC_Attention.gate_mode = 3`（`modules/attention_fn/bc_attn.py` 161–162）。量化：`bits = 4`、`head_bits = 6`、`mtp_bits = 4`（checkpoint `config.json` 的 `quantization_config`）。不是 QSA。`PAGE_SIZE = 256`。

BC 上限：`MAX_BSZ = 8`、`MAX_QLEN = 16`（`bc_attn.py` 52–53）。`BC_GatedMLP` 的 `MAX_BSZN = 8`（`modules/mlp.py` 15）。草稿步數 `K` 的驗證列數是 `n = K + 1`。`n <= 16` 才走 `BC_Attention`。`n <= 8`（`K <= 7`）才走 `BC_GatedMLP::run_bszN`。`K = 8..=15` 的 MLP 落到分開的 `Linear.forward`。預設 `K = 4`（`architecture/qwen3_5_mtp.py` 171，`default_draft_size`）。

## 主持決定（優先於下文）

- **GDN state 只配一份**，形狀由 `max_history` 決定：MTP 關時 `max_history = 0`，和現在完全相同；MTP 開時 `max_history = N`。不要另外再配一份窄的。所有 GDN launch（decode、prefill conv、FLA、history）的 `history_stride` 和 `state_size` 都從這份 state 的 `max_history` 算出：`history_stride = max_history + 1`、`state_size = conv_k + max_history`。`max_history = 0` 時傳進去的值和現在一字不差（`1` 與 `4`），所以非 MTP 的 launch 參數不變。conv prefill 的 Triton cubin 依 `state_size` 選；`state_size = 4` 的那支不變。這樣第 7 節「另配一塊寬的 state」改成這條。
- **Triton decode split/combine 要為 `q_len = 2..=16` 每一個都預編**（`q_len = 1` 的位元組不變）。驗證用 `K + 1`，接受後的 MTP refresh 用 `A - 1`（`1..=K`），所以 `K = 4` 就要 `q_len = 1..=5`。`--mtp-steps` 上限 15。
- **記憶體**：`N = 4` 時 GDN recurrent state 從 48 × 3.1 MB 變 5 倍（約 0.75 GB），加上 MTP 權重（約 0.2 GB）與 draft cache。預設 270336 token 的 cache 載入後已是 22.3 GB，長 context 時可能放不下。`Engine::load` 在 MTP 開時照常配置；配不下就回清楚的錯誤。文件要寫明 MTP 開時建議較小的 `--cache-tokens`。
- **短 prompt（≤ 17 token）的 prefill chunk 只有 ≤ 16 列**，fork 這時 target 也走 BC 路徑（`attn.py` 1067、`gated_delta_net.py` 1034），berth 現有的 prefill 不走。這是非 MTP 也存在的既有差異，這輪不修，只在第 9 節的 prompt 裡避開 ≤ 17 token（單 token prompt 除外，它根本不 prefill）。

## 1. 參考設定

Parity 的物件是 `mtp-host/common.md` 規定的那一組，把 serve 的 cache 建法和 acceptance check 的 MTP 頭合在一起。兩支腳本各自都不是這組。

`scripts/serve_openai.py` 333–350 載的是另一個 draft 目錄（DFlash），cache 是 `CacheLayer_quant`，`k_bits = v_bits = args.cq`，`--cq` 預設 3（同檔 1003）。`max_history` 取 draft 的 `default_draft_size`，沒有就 4（336）。`max_batch_size = 1`。draft cache 同樣 cq、同樣 `max_num_tokens`，不傳 `max_history`。`Generator` 不傳 `num_draft_tokens`，所以用 draft cap 的 `default_draft_size`（`generator/generator.py` 164–167）。

`~/qwen38-exl3-dflash2/scripts/acceptance_check.py` 28–56 在 `--draft mtp` 時用 `Model.from_config(tcfg, component="mtp")`，`max_history = default_draft_size`（預設 4），`Generator` 也不傳 `num_draft_tokens`。`Cache(...)` 沒給 `layer_type`，`cache/cache.py` 142 的預設是 `CacheLayer_fp16`。

要對的是這組：

- target：`Cache(model, max_num_tokens=..., layer_type=CacheLayer_quant, k_bits=3, v_bits=3, max_history=N, max_batch_size=1)`
- draft model：`Model.from_config(target_cfg, component="mtp")`，再 `attach_to`（載入時 fork 自己綁；見 `qwen3_5_mtp.py` 193–220）
- draft cache：`Cache(draft_model, max_num_tokens=同一個, layer_type=CacheLayer_quant, k_bits=3, v_bits=3, max_batch_size=1)`。MTP 頭沒有 recurrent layer（`generator.py` 162 斷言 draft 不能有 `recurrent_states`），`max_history` 留 0
- `Generator(model, cache, tokenizer, draft_model=..., draft_cache=..., num_draft_tokens=N)`。`dynamic_draft_tokens` 留預設 `False`（`generator.py` 46、174）。`N > 0` 時 `num_draft_tokens` 就是 `N`，不走到 `default_draft_size`

`max_history` 的文件寫「等於 draft token 數」（`cache/cache.py` 123–125）。`N = 2` 的參考要把 `max_history` 設成 2，不能沿用 acceptance check 寫死的 4。

dump 在 import 前設和 `scripts/dump_27b_decode_sequence.py` 45–69 一樣的環境：

```text
PYTHONPATH=/home/sam/qwen38-exl3-dflash2/exllamav3
TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu
LD_LIBRARY_PATH=<torch nvidia libs>:$LD_LIBRARY_PATH
EXL3_INT8_GEMV=2
EXL3_INT8_GEMV_MAX_K=5
EXL3_QC_STAGING=1
EXL3_HGEMM_F16ACC=auto
```

用 `~/qwen38-exl3-dflash2/env/bin/python`。腳本開頭 `os.environ.setdefault` 這四個 `EXL3_*`（同檔 66–69），這樣裸跑也跟 `scripts/serve-host.sh` 一樣。`:8889` 有在聽、或 GPU 記憶體已經到 4000 MiB，就拒絕，跟那支 dump 一樣。不要設 `EXL3_BC_ATTN=0`。BC 預設開，serve 走的就是這條。

每一輪要記的東西在第 9 節。hook 的方式也在那裡：照 `scripts/dump_27b_prefill_layers.py` 69–100，把方法換成 wrapper，forward 回來再 clone。

Graph：`BC_Attention::run`、`BC_GatedDeltaNetSplit::run_bszN`、`BC_GatedMLP::run_bszN` 對每個 `(bsz, q_len)` 或 `(bsz, seqlen, history)` slot，第一次 eager、第二次 capture、之後 replay（`attention.cpp` 的 `run`，`gated_delta_net.cpp` 326–370，`dump_27b_decode_sequence.py` 15–28 的說明）。berth 的 decode 是 eager，不捕捉（`docs/workspace.md`）。dump 照舊丟掉每個 slot 的前兩次相同 job，記錄的是 replay。berth 對那些 digest。graph 不是第二條要實作的數值路徑。

## 2. 帶 MTP 的 prefill

`Job.prefill`（`generator/job.py` 1218–1494）一次 `iterate` 跑一個 chunk。token 邊界和現在的 `prefill::chunk_bounds`（`crates/berth-engine/src/prefill.rs` 288）是同一條：

- `prefill_start = seq.kv_position`
- `prefill_end = align_down(start + max_chunk_size, 256)`，再 `min(end, len(sequence_ids) - 1)`
- `max_chunk_size` 是 2048（`generator.py` 34）
- recurrent 模型再切最後一頁：`last_page_b = (len - 1) // 256 * 256`，`start < last_page_b <= end` 時把 `end` 收到 `last_page_b`（1321–1328）

部分頁的 `best_match` 只在 `recurrent_cache is None` 時跑（1282）。27B 有 recurrent cache，這段是死的。裡面的 MTP 夾限（1295–1298）也跟著死。chunk 的 token 邊界不因 MTP 改變。100 token 仍是 `[0, 99)`，1500 仍是 `[0, 1280)` 再 `[1280, 1499)`。

MTP 把 `last_tokens_only = 1`，呼叫 `model.forward`，不呼叫 `model.prefill`（1408–1414）。`model.forward` 走 `forward_ls`（`model/model.py` 215–235，`model/model_ls.py` 317–330）。`forward_ls` 不設 `params["prefill"]`。`prefill_ls`（298–314）才會在最後一個 KV module 設 `prefill` 然後 break。所以 MTP prefill 和 berth 現在的 `Model::prefill`（`model.rs` 的 `prefill` / `prefill_with`，註解 1425–1427）差在這三點：

- 第 63 層的 MLP 有跑。`TransformerBlock.forward` 只在 `params["prefill"]` 時於 attention 之後直接 `return x`（`modules/transformer.py` 160–161）。這條 flag 沒設，第 63 層走完 mixer、residual、MLP。
- 最後一個 RMSNorm（target 的 `model.language_model.norm`，就在 lm_head 前面；`qwen3_5_mtp.py` 216–219 把它的 key 放進 `draft_verifier_params["export_state_norm_keys"]`）對每一列都跑。`last_tokens_only` 只在進入 `logits_output` module 時把 `x` 切成最後一列（`model_ls.py` 326–327），切點在 final norm 之後。norm 的 export 是 `y.half()`（`modules/rmsnorm.py` 169–173）。final norm 的 `out_dtype` 本來就是 fp16，export 是每一列的 post-norm hidden，形狀 `[1, chunk_rows, 5120]`。
- lm_head 只吃最後一列。`Job.prefill` 不接 `forward` 的回傳（1412）。那一列的 logits 算出來就丟掉，後面沒有人讀。cache、GDN state、export 都不依賴它。berth 的 MTP prefill 不必發這次 `lm_head`。

`params` 還帶 `draft_verifier_params`（1406–1407）、`block_table = seq.block_index_tensor`（整條序列的頁，不補到 16 的倍數）、`cache_seqlens = [prefill_start]`、target cache、`recurrent_states`。沒有 `recurrent_history`。`advance_recurrent_states`（`model.py` 235，`cache/recurrent_util.py` 66–74）因此把 `last_history` 設成 0。

第 0–62 層的 linear 路徑只看列數，不看 MTP。一個 chunk 的列數和現在一樣，所以：

- 列數 `> 8`：MLP 不是 `BC_GatedMLP::run_bszN`（`mlp.py` 740）。27B 的 `multi_gu` 沒建（下面第 4 節的 `use_mgemm`），`列數 > 32` 走兩次 `Linear.forward`（`mlp.py` 744–750）。`<= 144` 列是 `exl3_gemm`，`> 144` 列是 reconstruct。對應 berth 的 `Exl3Linear::forward_rows`（`device.rs` 210）。
- 列數 `> 16`：GDN 和 attention 都不進 BC（`gated_delta_net.py` 1034–1038，`attn.py` 1067–1070）。GDN 走 prefill 那條：`seqlen >= 48` 且 history 關時是 FLA `chunk_gated_delta_rule`，只讀寫 `recurrent_state[slot, 0]`（`gated_delta_net_fn/gated_delta_rule.py` 166–190）。conv 在 `seqlen > 32` 時走 Triton（`conv1d.py` 9、385–417），`history = false` 只讀寫每個 channel 的前 `conv_k` 個元素（`conv1d.py` 51–67、89–91：輸出的 load mask 是 `src_t < conv_kernel_size`，寫回起點是 0、長度是 `conv_kernel_size`）。位址的 stride 是 `state_size = conv_state.shape[-1]`。
- attention prefill 的 cubin 類別看 `num_pages_per_seq` 的 Triton 整數類（`== 1`、`% 16 == 0`、其他；berth `attn_prefill.rs` 的 `int_class`）和 split。split 只在 `pages * 256 + q_len >= 8192` 時打開（`attn_prefill.rs` 303–333）。頁數來自 `Sequence.prepare`：`total_pages = ceil((len(sequence_ids) + max_rq_tokens) / 256)`（`generator/pagetable.py` 231–238）。MTP 把預設 `max_rq_tokens` 從 `max_new_tokens + 1` 改成 `max_new_tokens + 1 + N`（`job.py` 1118–1120）。berth 現在的 `prefill_block_pages`（`prefill.rs` 276）沒有 `+ N`。

所以第 0–62 層的 mixer 演算法和現在的 prefill 相同，但有兩個會讓舊 digest 不能直接當 MTP prefill 的預期值：

- conv 的 `state_size` 是 Triton constexpr（`conv1d.py` 26，`BLOCK_STATE = next_pow2(state_size)`）。`max_history = N` 時 `state_size = 4 + N`，和 berth 現在為 `state_size = 4` 編的 cubin（`gdn_prefill.rs` 69–71，load 時 `require_i32(..., "state_size", 4)`）不是同一個類別。輸出公式只讀每個 channel 的前 4 個元素，前 4 個元素的值一樣時輸出一樣；launch 必須用寬 buffer 的 stride。舊 cubin 不能套在寬 layout 上。
- 頁數的類別或 8192 split 門檻一變，attention prefill cubin 就變。`max_new_tokens = 1` 時：100 和 1500 加 `N = 2` 或 `4` 頁數不變（100：102→106，都是 1 頁；1500：1502→1506，都是 6 頁）。8190：`8190 + 2 = 8192` 是 32 頁（`div16`），加 `N >= 2` 變成 33 頁（`other`）。8400 本來就是 33 頁，加 4 仍是 33。

舊的第 0–62 層 activation digest，在頁數類別和 split 決策沒變的 prompt 上仍然是預期值（100、1500，`max_new = 1`，`N = 2` 或 `4`）。conv state 的尾端不要拿去比。8190 要在 MTP 寬度下重 dump。參考腳本對第 0–62 層再 hash 一次，不要假設舊檔。

### MTP 頭吃這個 chunk

export 拿到之後（`job.py` 1426–1446）：

- 第一個 chunk，`seq.mtp_carry_hidden` 是 `None`（`pagetable.py` 228），carry 用 `zeros_like(target_hidden[:, :1, :])`，形狀 `[1, 1, 5120]`
- `shifted = cat(carry, target_hidden[:, :-1], dim=1)`，列數等於 chunk 列數。第 0 列是上一 chunk 的最後一列（或 0），第 `i > 0` 列是這一 chunk 第 `i - 1` 列的 post-norm
- `seq.mtp_carry_hidden = target_hidden[:, -1:, :].clone()`
- `job.mtp_last_hidden` 指到同一個 tensor

然後 `draft_model.prefill`，`input_ids` 是這段 chunk 的 id，`target_hidden = shifted`，`cache` 是 draft cache，`cache_seqlens = [prefill_start]`，`block_table` 是 target 的 `seq.block_index_tensor`（同一組頁號，另一塊儲存；`generator.py` 160 要求兩個 cache 的 `max_num_tokens` 相同）。

MTP module 順序（`qwen3_5_mtp.py` 32–165）：`Qwen3_5MTPInputLayer`，然後一層 `TransformerBlock`（`mtp.layers.0`，full attention，`interleaved_gate = True`，dense `GatedMLP`，`qbits_key = "mtp_bits"`），然後 `mtp.norm`。`last_kv_module_idx` 在 append `mtp.norm` 之前設成 `len(modules) - 1`（155），所以最後一個 KV module 是那個 `TransformerBlock`，不是 norm。

`model.prefill` → `prefill_ls` 在這個 block 上設 `prefill = True` 然後 break。block 跑完 attention 就 `return x`（`transformer.py` 160–161），MLP 和 `mtp.norm` 都不跑。prompt prefill 和接受後的 refresh 都是這條。dense MLP 只在草稿的 `forward`（1 列）裡跑。

一個 n 列的 MTP prefill（n 是 chunk 列數，通常遠大於 16）實際跑的是：

1. `Qwen3_5MTPInputLayer.forward`（`modules/arch_specific/qwen3_5_mtp.py` 88–114）。`pre_fc_norm_hidden(target_hidden)` → fp16。target embedding 的 `forward`，`out_dtype = fp16`（104；embedding 本體是 target 的，`attach_to` 借來）。`pre_fc_norm_embedding`。`cat((emb, hidden), dim=-1)`，最後一維 `10240`。`mtp.fc`：`Linear`，`2H → H`，`out_dtype = fp32`，`qbits_key = "mtp_bits"` 所以 K = 4（`qwen3_5_mtp.py` 33–43）。
2. block 的 `input_layernorm`（fp16），然後 attention。`n > 16` 不進 `BC_Attention`（`attn.py` 1067–1070），走 `project_qkv`、q/k norm、RoPE、`attn_dispatch` 的 prefill kernel，寫 draft cache 的 cq3。寫入起點是 `cache_seqlens = prefill_start`，頁表是上面那條未補 16 的 block table。`gate_mode` 會是 3（這個 Attention 建構時 `interleaved_gate = True`，`qwen3_5_mtp.py` 80）。
3. 到此 return。沒有 `post_attention` MLP，沒有 `mtp.norm`。

`mtp.fc` 的列數等於 chunk 列數，走 `Linear.forward` 的列數分流，和 target MLP 同一套（第 4 節）。幾百列是 reconstruct 或 `<= 144` 的 coop GEMM，不是 `m <= 8` 的 GEMV。

chunk 結束後 `seq.kv_position = prefill_end`（1455）。prefill 涵蓋 `ids[:-1]`，所以整段 prompt prefill 結束時 `mtp_last_hidden` 是倒數第二個 prompt token 的 post-norm。第一個草稿步把最後一個 prompt token（還沒進 cache）和這塊 hidden 配在一起（`generator.py` 1330 的註解，配對方式在第 6 節）。

單 token prompt：`is_prefill_done` 一開始就是真（`kv_position == 0 == len - 1`，`job.py` 367–368）。`mtp_last_hidden` 仍是 `None`。第一輪草稿直接 `return None`（`generator.py` 740–743），第一輪 `iterate_gen` 是普通的 1 token forward。

## 3. 草稿回合

`iterate_draftmodel_mtp_gen`（`generator.py` 703–796）。一個 job 只能有一條 sequence（739）。

Block table 寬度，token 數是 `max(job.get_max_seq_len() + num_draft_tokens + 1)`（712）。`get_max_seq_len`（`job.py` 371–378）在 prefill 完成後回傳 `len(sequence_ids)`，含那個還沒進 cache 的 token；prefill 沒完成回傳 0。頁數 `ceil(tokens / 256)`，再補到 16 的倍數（718–721）。多出來的欄位先清 0。`cache_seqlens[batch] = seq.kv_position`（731），只加在這塊 staging tensor 上，不加 `seq.kv_position`。每步結束 `cache_seqlens += 1`（777）。

`batch_ids` 一開始是 `get_input_ids_list()`（747，不帶 draft、不加進 cache）：`sequence_ids[kv_position:]`，平常就是那 1 個還沒進 cache 的 id。`temp_hidden` 是 `mtp_last_hidden`，fp16 `[1, 1, 5120]`。

迴圈 `idx in 0..K-1`（`window = num_draft_tokens`，757；`dynamic_draft` 預設關，calibrator 是 `None`，不會提前縮窗，759–787 的 confidence 分支不進）：

```text
draft_model.forward(batch_ids, {
    target_hidden: temp_hidden,
    attn_mode: "flash_attn",
    block_table, cache: draft_cache, cache_seqlens,
})
batch_state = lm_head.prepare_for_device(那個輸出)   # 773
new_ids = draft_model.sample_from_state(batch_state) # 774
batch_ids = new_ids
cache_seqlens += 1
temp_hidden = batch_state                            # 778
```

`sample_from_state`（`qwen3_5_mtp.py` 231–248）再 `prepare_for_device` 一次，然後 target `lm_head.forward`，`torch.argmax`。`export_draft_conf` 只有 calibrator 開才設。logits 不裁。

`batch_state` 是 `mtp.norm` 的 fp16 輸出經過 `Module.prepare_for_device`（`modules/module.py` 84–92）。單 GPU 上 tensor 已經在 lm_head 的 device，這個呼叫原樣返回，不改值、不改 dtype。下一階的 hidden 是這塊，不是 logits。

1 列 MTP forward（`model.forward`，沒有 `prefill` flag）跑完整個頭：

1. input layer，和上面相同，列數 1。`mtp.fc` 是 K = 4、m = 1、fp32 out 的 `Linear`。m = 1、mul1、K ≤ 5 走 int8 GEMV（第 4 節）。
2. `input_layernorm`，attention。`bsz = 1`、`seqlen = 1` ≤ 16，`EXL3_BC_ATTN` 開、沒有 non-causal span，走 `BC_Attention::run`（`attn.py` 1067–1074）。cq3，`quant_cache_paged` 寫 **1** 列，位置是當下的 `cache_seqlens`，也就是 `kv_position + idx`（`attention.cpp` 564–569，`q_len` 引數是 1）。頁表是上面補到 16 的 draft 寬度。`gate_mode = 3`。kernel 順序和 target 的 1 列 decode 相同：sliced qkv mgemm、`deinterleave_qg`、fused q/k norm + RoPE、`quant_cache_paged`、Triton decode split/combine、`mul_sigmoid`、`o_proj`。berth 已經有這條的 target 版（`attn.rs` 模組註解 1–16）。MTP 頭是另一組權重（`mtp.layers.0.self_attn.*`），同一套 kernel。
3. `post_attention_layernorm`，dense `GatedMLP`。m = 1 ≤ 8，走 `BC_GatedMLP::run_bszN`（`mlp.py` 740–742）。27B 形狀的 `multi_gu` 沒建時是兩次 `exl3_gemm` + `silu_mul` + down `exl3_gemm`（`mlp.cpp` 62–90）。m = 1、K = 4 的 gate/up/down 走 int8 GEMV。
4. `mtp.norm`，fp16 輸出。這就是下一階的 hidden。

Argmax 用 target `lm_head`，1 列。`head_bits = 6`，K = 6。int8 GEMV 的 K 上限是 `EXL3_INT8_GEMV_MAX_K = 5`（`exl3_gemv_int8.cu` 245），QTIP GEMV 要 K = 2..4（`exl3_gemv.cu` 49）。兩條都拒絕，落到 coop GEMM。berth 的 1 列 `lm_head` 已經是這條（`Exl3Linear::forward`，`device.rs`；decode 的 `Model::project`）。草稿 argmax 叫同一支，權重是 target 的 `lm_head`，不是 MTP 自己的。

草稿的 K 步在 draft cache 寫下位置 `kv_position .. kv_position + K - 1`。不寫 `kv_position + K`。

## 4. 驗證 forward

`iterate_gen`（`generator.py` 943–1348）帶 `draft_tokens`，形狀 `[1, K]`。

輸入 id：`get_input_ids_list(draft_tokens, add_to_cache=True)`（996，`job.py` 381–406）。`sequence_ids[kv_position:]` 是還沒進 cache 的 token，平常 1 個，再接 `d1..dK`。`q_len = K + 1`。`add_to_cache=True` 在 forward **之前**把這 `K + 1` 個 id 寫進 target 頁，並把 `page.kv_position` 加上 `K + 1`。`seq.kv_position` 這裡不動。

`cache_seqlens` 的 staging 在這次推進之前抄好（969–978），forward 看到的是推進前的 kv 長度。

Block table 的 token 寬度（950–957）：先 `max(get_max_seq_len() + num_draft_tokens)`，`draft_tokens is not None` 時再加 `draft_tokens.shape[-1]`，也就是 `K`。頁數 `ceil(/ 256)`，再補到 16 的倍數（965–966）。`K == N` 時，草稿寬度是 `get_max_seq_len + N + 1` 個 token，驗證寬度是 `get_max_seq_len + N + K = get_max_seq_len + 2N`。補頁之前兩者差 `K - 1` 個 token。多出來的 0 欄位 kernel 不讀（963–964 的註解：讀取範圍由 cache length 限制）。

`recurrent_history = draft_tokens is not None`（1036），驗證是 `True`。有 draft model 時再併入 `draft_verifier_params`（1039–1040），final norm 把全部 n 列 export 出來。

`n = K + 1 <= 16` 時，每一層走 BC，不走 prefill kernel。順序如下。berth 已有的都是 `n = 1` 的 decode，或列數很大的 prefill。

### Embedding

target `prepare_inputs` 把 n 個 id gather 成 fp32 `[1, n, 5120]`。berth 的 `embed_ids`（`model.rs` 920）已經吃 n。decode 的 `embed_token` 是 n = 1 的那條。驗證用 `embed_ids`。

### 每一層 GDN

`BC_GatedDeltaNetSplit::run_bszN(..., history=true)`（`gated_delta_net.py` 1030–1045）。27B 不是 KDA，不是 fused qkvz。`R = bsz * seqlen = n <= 16 <= 32`，而且 sliced qkv+z bundle 在（`EXL3_QKV_SLICE` 預設開，`gated_delta_net.py` 17–18），所以 `use_qkvz`（`gated_delta_net.cpp` 228–229）。

`run_bszN_gr`（216–324）的 launch 順序：

1. `exl3_mgemm_gr`，qkv 和 z 一次 sliced bundle，fp16 中間、寫進 slot 的 qkv / z static（234–242）。K = 4，mul1。
2. `gdn_ba_gemv_gr`，b/a 投影，fp16 權重 × fp16 輸入 → fp32（278）。
3. `gated_delta_net_fused_op_3_gr`（280–287）。
4. `cuda_causal_conv1d_update_gr(..., activation=true, history=true)`（290–301）。實例 `conv1d_update_kernel<true, true>`（`gdn.cu` 1482–1485）。
5. `cuda_recurrent_gated_delta_rule_gr(..., history=true)`（303–317）。27B：`k_head_dim = v_head_dim = 128`，`bsz = 1`，`num_v_heads = 48 <= 64`，所以 `v_split = 4`（928）。實例 `cuda_recurrent_gated_delta_rule_kernel_128<true, 4>`（995–999）。grid `(bsz, num_v_heads, v_split) = (1, 48, 4)`。`history_stride = recurrent_state.size(1) = max_history + 1`（933）。
6. gated RMSNorm（319）。
7. `o_proj` 的 `exl3_gemm_gr`，fp32 out（321）。

history 怎麼寫（`kernel_128`，`gdn.cu` 728–737）：`final_state` 是 index 0。`s` 從 0 到 `seqlen - 1`：

- `s == 0`：讀 index 0（forward 前的 state）
- `s == seqlen - 1`：寫 index 0
- 否則：讀 index `s`，寫 index `s + 1`

n 個 token 之後：index 0 是 n 個都吃完的 state（蓋掉 forward 前的那份，沒有別的 slot 留著它）。index `k`（`k = 1 .. n - 1`）是吃完 k 個 token 的 state。`advance_recurrent_states` 把 `last_history` 設成 `seqlen - 1`（`recurrent_util.py` 74）。

conv 的 history 寫在第 6 節，和 rewind 放一起。

berth 現在：`gdn_launch.cu` 只實例 `conv1d_update_kernel<true, false>` 和 `kernel_128<false, 4>`。`gdn.rs` 的 decode launch 把 `history_stride` 寫死 1（約 1166）。`FORK_EXTRACTS` 的 `gdn.cu` 660–849 已經含 `save_history` 的模板本體，1309–1406 已經含 conv 的兩個 history 分支。缺的是 `<true, 4>` 和 `<true, true>` 的實例，以及會把 `history`、`history_stride`、`state_size` 傳進去的 launcher。prefill 的 FLA cubin 是另一條，列數 `> 16` 才用，驗證用不到。

### 每一層 attention

`BC_Attention::run`，`q_len = n`，regime 0（不是 QSA）。不是 prefill kernel。`R = n <= 16 <= 32`，sliced qkv bundle 在，`use_qkv`（`attention.cpp` 436）。`run_gr`（396–840）對 27B 的 `gate_mode = 3`：

1. sliced `exl3_mgemm_gr`，q（含 interleaved gate）、k、v（437–450）
2. `deinterleave_qg_gr`（449）
3. fused q/k RMSNorm + RoPE，in place（549–561）
4. `quant_cache_paged_gr`，寫 **n** 列 cq3 的 K 和 V，起點 `cache_seqlens`（566–569）。`q_len` 是這次的 n。非量化的 Triton `k_update` 這條不走（`quant_cache` 為真）
5. decode split kernel，再 combine。`split_config`（388–394）：

```text
bound = bt_width * page_size + q_len
num_splits = max(1, min(splits_cap, ceil(bound / (4 * block_n))))
split_len = ceil(ceil(bound / num_splits) / block_n) * block_n
```

`bc_attn.py` `_configure`（263–310）：`hd_pad = 256`，`block_n = max(16, 8192/256) = 32`。`block_m = next_pow2(q_len)`，`block_h = max(16/block_m, 1)`，`block_rows = block_m * block_h`，`h_blocks = cdiv(group=6, block_h)`，`programs = bsz * kv_heads * h_blocks`。3090 有 82 個 SM，`target = 164`，`splits_cap = min(target / programs, 128)`。constexpr 含 `q_len`、`kv_append_len = q_len`、`BLOCK_M/H/ROWS/N`。`FINAL` 在 BC 的 consts 裡寫死 `False`（308），`run_gr` 永遠先 split 再 combine（752–811）。eager 的 `paged_attn_triton_decode` 在 `num_splits == 1` 時會跳過 combine；fork 的註解說 `split_config` 讓兩邊的數值相同。驗證的 `num_splits` 在有頁的表上大於 1，兩條都會 combine。

6. `mul_sigmoid`（`gate_mode == 3`，825–826）
7. `o_proj` 的 `exl3_gemm_gr`，fp32 out（834–837）

`q_len > 1` 仍是這一家 Triton decode split/combine，用新的 `q_len` 重編，不是 prefill kernel。tile 和 `q_len` 的關係（`block_rows` 在這些點都是 16）：

| `q_len` | `block_m` | `block_h` | `h_blocks` | `programs`（bsz 1、kv 4） | `splits_cap` |
|---|---|---|---|---|---|
| 1 | 1 | 16 | 1 | 4 | 41 |
| 2 | 2 | 8 | 1 | 4 | 41 |
| 3、4 | 4 | 4 | 2 | 8 | 20 |
| 5–8 | 8 | 2 | 3 | 12 | 13 |
| 9–16 | 16 | 1 | 6 | 24 | 6 |

`q_len` 本身是 constexpr。3 和 4 的 tile 相同，cubin 仍是兩份。5–8、9–16 同樣。berth 現在只編 `q_len = 1`（`scripts/build_triton_cubins.py` 49–50）。`K = 4` 要 `q_len = 5`，`K = 2` 要 `q_len = 3`。`--mtp-steps` 放到 15 就要 `q_len = 1..=16` 裡會用到的每一個。

16 頁、`q_len = 1`：`bound = 4097`，`num_splits = min(41, ceil(4097/128)) = 33`，`split_len = 128`。這就是現在 decode 的數。16 頁、`q_len = 5`：`bound = 4101`，`num_splits = min(13, 33) = 13`，`split_len = ceil(ceil(4101/13)/32)*32 = 320`。partial buffer 的大小跟著 `programs`、`splits_cap`、`block_rows` 變，`q_len = 1` 的那塊不能拿來跑 `q_len = 5`。

berth 已有：`quant_cache_paged` 的 `seq_len` 是 runtime 引數（`csrc/attn/quant.cu` 35，grid 的 y 維就是 `seq_len`），decode 傳 1。`split_config` 在 `berth-kernels` 的 attn 包裝裡已經吃 `q_len`。缺的是每個 `q_len` 的 split/combine cubin，和一條 `q_len = n` 的 launch（投影、RoPE、gate、`o_proj` 的列數跟著 n 走；sliced mgemm 的 `R <= 32` 在 `n <= 16` 仍然成立）。prefill 的 cubin 家族（`attn_prefill`）不要拿來頂這條。

### 每一層 MLP

`GatedMLP.forward`（`mlp.py` 721–791）。`bsz * q_len <= MAX_BSZN`（8）時 `BC_GatedMLP::run_bszN`。`K = 2`（n = 3）和 `K = 4`（n = 5）都在裡面。`K >= 8`（n ≥ 9）不在：`multi_gu` 是 `None` 時走 gate、up 兩次 `Linear.forward`、`silu_mul`、down（744–750）。n ≤ 16 仍小於 144，是 `exl3_gemm`，不是 reconstruct。

`use_mgemm`（`model/config.py` 61–77）：mul1 為假就融合。mul1 為真時，`K >= K_thr` 或 `out_features < n_threshold` 才融合。`EXL3_INT8_GEMV > 0` 把預設 `K_thr` 設 6、`n_threshold` 設 8192（30–32）；環境沒有 `EXL3_MGEMM_K_THRESHOLD` 時，`K_thr` 再改成 `exl3_gemv_int8_max_k(device) + 1`（71–76），這組環境是 6。27B MLP gate 的 N = 17408，不小於 8192，K = 4 小於 6，所以 `multi_gu` 不建。`run_bszN_gr` 走未融合支（`mlp.cpp` 62–90）：gate `exl3_gemm_gr`、up `exl3_gemm_gr`、可選 bias、`silu_mul_gr`（`act_limit` 是 27B 的值）、down `exl3_gemm_gr`、可選 bias。和 `mlp.rs` 開頭的註解同一條。MTP 頭的 MLP 也是 `GatedMLP`、`mtp_bits = 4`、intermediate 17408，同一條未融合支；它在驗證裡不跑，只在 1 列草稿裡跑。

`exl3_gemm` 裡小 m 的分流（`quant/exl3_gemm.cu` 192 起）：

- mul1 且 `EXL3_INT8_GEMV > 0`：先試 int8。mode 預設 2（環境設了 2；mode 1 才是 residual）。SQ kernel 在 plain mode 接受 `m <= 2`（`exl3_gemv_int8.cu` 127、250–258）。`m > 1` 接著直接 `return false`（260），不進 coop int8。K 必須 `<= exl3_gemv_int8_max_k`（245），這組環境是 5。所以 m = 1 和 m = 2、K ≤ 5、N 是 256 的倍數、K 維是 128 的倍數：`exl3_gemv_int8_sq_kernel`。
- 然後 `exl3_gemv_try_launch`（`exl3_gemv.cu`）。`EXL3_GEMV` 沒設就是 heuristic mode 1（29–33）。硬條件：K = 2..4、`m <= 8`（`EXL3_GEMV_MAX_M`，`exl3_gemv_kernel.cuh` 30）、k 和 n 都是 128 的倍數。`exl3_gemv_cfg`（46–71）：`size_n / 32 <= occupancy * SM 數` 用 narrow；否則 `k <= 2048 且 n <= 8192` 用 narrow；否則 `n >= 8192 且 k <= 4096` 用 wide；否則 Ampere 上 `n` 在 `[8192, 10240]` 且 `k <= 5120` 用 wide；否則拒絕。拒絕的形狀進 coop autotune GEMM（`exl3_gemm.cu` 的 autotune key 用 `max(size_m, 2)`）。
- occupancy 是測出來的。gate/up（k = 5120，n = 17408，`n/32 = 544`）和 down（k = 17408，n = 5120，fp32 out，`n/32 = 160`）在 m = 3 和 m = 5 會不會進 QTIP，見文末「未確認」。

berth：`Exl3Linear::forward` 只做 m = 1（合格就 int8）。`prefill_lin.rs` 的 `exl3_linear` 做 `rows > 1`：m = 2 走 int8 sq，其餘 coop。QTIP（int8 拒絕之後的 `m <= 8`）是明確拒絕，不換 kernel（`prefill_lin.rs` 94–95）。m = 3 和 m = 5 如果 fork 發的是 `exl3_gemv_kernel`，berth 現在對不上，缺的是那支 GEMV，不是改呼叫 `forward_rows`。如果 heuristic 拒絕、走 coop，`forward_rows` / `exl3_linear` 已經是同一條。`K >= 8` 的驗證 MLP（n ≥ 9）過了 `EXL3_GEMV_MAX_M`，一定是 coop，berth 已有。

### Final norm、export、lm_head

final RMSNorm 跑在全部 n 列上，fp16。key 在 `export_state_norm_keys` 裡，所以 `export_states[-1]` 是這 n 列（`rmsnorm.py` 169–173）。`last_tokens_only` 在驗證沒設，lm_head 吃全部 n 列。

lm_head 是 K = 6。m = 1 和 m = n 都是 coop GEMM，autotune key `max(m, 2)`（`exl3_gemm.cu` 255）。int8 只收 K ≤ 5（`exl3_gemv_int8.cu` 245、260），QTIP 只收 K = 2..4（`exl3_gemv.cu` 49），所以 K = 6 兩條都不進。berth 的 1 列已經是 decode shim 的 coop。n 列（2..=144）走 `Exl3Linear::forward_n`：K = 6 改呼叫同一支 `berth_kernels::exl3_linear`（`size_m = rows`，`a_had` 是呼叫端的 n 列），不走只連結 K = 4 的 prefill shim。

logits 形狀 `[1, n, 248320]` fp16。抽樣逐列切 `[1, 1, vocab]`。

## 5. 抽樣與接受

`draft_tokens is None` 且 logits 只有 1 列時走批次的單 token 路徑（`generator.py` 1108–1151）。MTP 的驗證不走這條。單 token prompt 的第一輪沒有草稿，走這條；`accepted_lengths` 記 1，第 6 節的 refresh 因 `accepted_length > 1` 不跑。

有草稿時是一條 sequence 裡的位置迴圈（1156–1259）。`i` 從 0 到 `n - 1`：

1. `token_logits = job_logits[:, i:i+1, :]`
2. `receive_logits`（`job.py` 560–596）：沒有 forced token 時 `sampler.forward(logits, past_ids, rng.randint(0, 2**32 - 1), ...)`。greedy 也抽一個 `randint`。
3. `receive_sample`（599–1016）把這個 token 收進 sequence：`new_tokens += 1`（632），`sequence_ids.append`（638），`seq.kv_position += 1`（640）。stop token、`max_new_tokens`、filter、stop string 的檢查都在這之後。

接受條件（1220–1224），只在 `i < n - 1`（還有對應的 draft）時看：

```text
sampled != draft_tokens[j, i]  或  is_checkpoint_boundary()
```

任一為真：`reject_remainder(job, j, i, batch_states)` 然後 break。都不真：`accepted_draft_tokens += 1`，`accepted_length += 1`，然後 `prepare_logit_mask` 和 `prepare_sampling_past_ids`（1240–1242）。`accepted_length` 從 1 起算，加上接受的 draft 數。對上的 draft 位置是 `i`，發出的 token 數是 `i + 1`（含這個位置的 sample）。最後一列 `i = K` 沒有 draft 可比，迴圈自然結束。全部接受時 `accepted_length = K + 1`，`rejected` 維持 0。

對不上的那個 sample 已經被 `receive_sample` 收進 sequence，它就是這一輪發出的 token。被拒的是它後面的 draft 位置。

視窗裡的 EOS / stop / `max_new_tokens`（1187–1202，`receive_sample` 829–835）：`receive_sample` 回 `eos`，break，**不**呼叫 `reject_remainder`。`rejected` 維持迴圈開頭的 0（1161）。未解的 draft 只計進 `rejected_draft_tokens`（`n - 1 - i`）。cache 的 kv 長度不回捲。因為 `rejected` 仍是 0，1244–1246 還是呼叫 `batch_states[j].rewind(0)`：GDN 把 conv 尾端抄回開頭，recurrent state 不退（`gated_delta_net.py` 113–121）。`accepted_length` 停在 `i + 1`。1301–1333 在 `accepted_length > 1` 時仍 refresh draft cache，然後 `deallocate_pages`（1339–1342）。stop token 先於 `max_new_tokens`（829 在 834 之前）。純函數 `accept_window` 仍回 `Stop` / `Length`，不回 `Rewind`；GPU 迴圈用 `accept_row`，`rewind(0)` 是呼叫端的事。

requeue（`new_tokens > max_rq_tokens - num_draft_tokens`，`job.py` 633）：`reject_remainder` 然後 break（1181–1185）。banned-string 的 checkpoint rewind 在 `receive_sample` 裡做完，`checkpoint_rewound` 讓迴圈把 `rejected` 設成 -1、不再 rewind（1208–1214）。berth 這輪不實作 banned string、filter、requeue；參考 dump 也不開它們。

`rejected == 0`（全部接受，或 EOS break 沒有改 `rejected`）：`batch_states[j].rewind(0)`（1245–1246）。

### Sampler 狀態

`prepare_sampling_past_ids`（`job.py` 1546–1562）把 `sequence_ids` 的尾端補上 device。迴圈前叫一次（1067），每接受一個 draft 再叫一次。penalties 和 DRY 看到的是已經接受的 token，含 prompt。對不上的那個 token 也已經 append，所以它在 history 裡；沒抽到的後續位置不在。

RNG：每個走到的位置在 `receive_logits` 抽一次 `randint`，含 greedy。break 之後的位置不抽。

berth `Sampler::sample`（`sampler.rs` 383）已經是「一列 logits → 抽一個 token → append 到 history」，而且 greedy 也先 `randint_u32`（418–419）。驗證時對每一個走到的位置叫一次，break 之後不要再叫。past ids 用它自己的 history，不必另做一份 `prepare_sampling_past_ids`。不用改 `sample` 的內部順序。`Engine::generate` 現在每輪只叫一次，MTP 分支要改成對 n 列逐列叫，並在拒絕、EOS、`max_new_tokens`、checkpoint 邊界處停。

### `is_checkpoint_boundary`

`job.py` 1577–1594。`seq_pos = seq.kv_position`，`prompt_len = len(sequence_ids) - 1`。這個檢查發生在 `receive_sample` 之後，所以 `seq_pos` 已經含剛收下的 token，而且 `seq_pos == prompt_len`（append 和 `kv_position += 1` 成對）。

```text
override 有設:  seq_pos % override == 0
否則，seq_pos >= prompt_len - max_chunk_size * 2:
    (seq_pos - cached_pages * 256) % recurrent_checkpoint_interval == 0
否則:
    (seq_pos - cached_pages * 256) % recurrent_checkpoint_interval_pp == 0
```

`recurrent_checkpoint_interval` 預設是架構的 `default_recurrent_checkpoint_interval`，27B 是 2048（`architecture/qwen3_5.py` 486，`generator.py` 239）。必須是頁的倍數。`interval_pp` 預設 32768（`generator.py` 44）。

生成中 `seq_pos == prompt_len`，所以 `seq_pos >= prompt_len - 4096` 始終成立，走 2048 那條。berth 沒有 prefix cache，`cached_pages` 維持 0（fork 只在 prefill 吃到已經滿的頁時加，`job.py` 1258；我們不復活頁）。公式是 `seq_pos % 2048 == 0`。

邊界上即使 draft 對上了也拒絕並 rewind。token 本身留下。`maybe_stash_recurrent`（1614）要求 `kv_position % 256 == 0`；`iterate` 在 prefill 之後、草稿之前呼叫 `recurrent_checkpoint`（`generator.py` 533–535）。stash 的是停下來之後、下一輪 prefill 結束時的頁對齊位置。測試選長度時不要踩 2048，除非那一輪就是要測邊界。

## 6. Rewind

`reject_remainder`（`generator.py` 1081–1100）。`num_rejected = n - 1 - i`。0 就直接返回。

GDN：`batch_states[j].rewind(num_rejected)`。`GDNState.rewind`（`gated_delta_net.py` 113–121）對每一層收 job，然後 `position -= num_tokens`，`last_history = 0`。

state 的來源 index（`rewind_state_job` 273–285）：`last_history + 1 - num_tokens`。驗證後 `last_history = n - 1`，所以來源是 `n - num_rejected`。拒絕發生在位置 `i` 時 `num_rejected = n - 1 - i`，來源 index = `i + 1`。index `i + 1`（範圍 `1 .. n - 1`）是吃完 `i + 1` 個 token 的 state，也就是留下的那些 token 之後的 state。`num_tokens == 0` 時不發 state job（276）。

conv（`rewind_conv_job` 251–270）：`last_history == 0` 就不做。否則 `p = state_size - num_tokens`，把 `[p - conv_k, p)` 抄到 `[:conv_k]`。`state_size = conv_k + max_history`。

全部接受的 `rewind(0)`：state job 跳過，index 0 已經是最終 state。`last_history = n - 1 > 0`，conv job 仍跑：`p = state_size`，把尾巴的 `conv_k` 個元素抄到前面。下一階 history 關閉的 kernel 只讀 `state[0:conv_k]`，這份抄寫就是為了它。

conv 在 history 打開時怎麼寫（`gdn.cu` 1316–1318、1393–1404）：最新的 `min(state_size, conv_k + seqlen)` 個輸入寫進 buffer 的尾巴。驗證 `seqlen = K + 1`、`max_history = K`、`conv_k = 4`：`state_size = 4 + K`，邏輯序列長度 `4 + K + 1`，寫滿整個 buffer，丟掉最舊的那一個。以 `K = 4` 為例，寫入是 `old[1], old[2], old[3], x[0]..x[4]`。拒絕在 `i = 0`（留下 1 個 token，`num_rejected = 4`）時 `p = 4`，抄回 `[0:4) = old[1], old[2], old[3], x[0]`，就是只吃了 `x[0]` 之後的 conv 窗。

attention：`reject_remainder` 只減 `page.kv_position`（1092–1099），不擦 cq3。`get_input_ids_list` 事先加了 `K + 1`，這裡減掉 `num_rejected`，淨增加 `i + 1`，和 `receive_sample` 對 `seq.kv_position` 的增加次數相同。被拒的列留在頁裡，下一次寫同一個 cursor 時蓋掉。

### Draft cache 怎麼對齊

草稿 K 步寫的是 draft cache 的 `P .. P+K-1`，`P` 是這輪開始時的 `seq.kv_position`。target 驗證寫的是 target cache 的 `P .. P+K`。頁號相同，儲存是兩塊（`generator.py` 1091 的註解）。

接受後的 refresh（1301–1333），跳過 `checkpoint_rewound` 的 job：

```text
target_hidden = export_states[-1]          # final norm 的 n 列
若 accepted_length > 1:
    draft_model.prefill(
        ids = batch_ids[:, 1:accepted_length],          # d1 .. d_{A-1}
        cache_seqlens = p_cache_seqlens + 1,            # P + 1
        target_hidden = target_hidden[:, :accepted_length - 1],
        block_table = 這一輪的驗證表,
        cache = draft_cache,
    )
job.mtp_last_hidden = target_hidden[:, A-1:A].clone()
```

`batch_ids` 是 `[還沒進 cache 的 token, d1..dK]`。`A = accepted_length`。

這次 `prefill` 和 prompt 時一樣，停在 `TransformerBlock` 的 attention，不跑 MTP MLP，不跑 `mtp.norm`。它只重寫 draft cache。列數是 `A - 1`，`A - 1 <= K`。`K = 4` 時最多 4 列，`<= 16`，attention 走 `BC_Attention` 而不是長 prefill kernel；`A - 1 > 8` 的 MLP 反正被 `prefill` flag 跳過。`mtp.fc` 仍跑，列數 `A - 1`，小 m 分流見第 4 節。

位置 `P`（草稿第 0 步，用的是上一輪真實的 target state）留著。refresh 從 `P + 1` 寫到 `P + A - 1`，id 是 `d1 .. d_{A-1}`，hidden 是 export 的第 `0 .. A-2` 列。配對和 prompt 的 shift 一樣：位置上的 id 是當前 token，hidden 是前一個 token 的 post-norm。export 第 `j` 列是「吃完驗證輸入第 `j` 個 token」的 state，logits 第 `j` 列用它預測下一個 token。`mtp_last_hidden = export[A - 1]` 就是預測「最後一個收下的 token」的那塊 state。下一輪草稿的輸入 id 是 `sequence_ids[kv_position:]` 的那一個，也就是最後收下的 token，和這塊 state 配成 `(token, 它前一個 token的 post-norm)`。

`A = 1`（一個 draft 都沒收下）不做 refresh。下一輪草稿的 `cache_seqlens` 已經前進 1，寫在 `P + 1`，蓋掉第一個投機 slot。

## 7. 記憶體配置

`GDNLayerState`（`gated_delta_net.py` 182–190）：

- `conv_state`：bf16 `[max_batch, fdim_qkv, conv_k + max_history]`。27B 是 `[1, 10240, 4 + N]`
- `recurrent_state`：fp32 `[max_batch, max_history + 1, num_v_heads, k_head_dim, v_head_dim]`，`[1, N + 1, 48, 128, 128]`

`max_history = 0` 就是 berth 現在的形狀（`gdn.rs` 67–72 的註解）。history 關閉的 kernel 只碰 index 0 和 conv 的前 `conv_k` 個元素，數值不讀尾巴。可是 stride 是 buffer 的最後一維：conv 的 channel stride、recurrent 的 `history_stride`。把現有 buffer 加寬、又沿用 `history_stride = 1` 和 `state_size = 4` 的 launch，位址會錯。

非 MTP 路徑維持 `max_history = 0`：現有 decode / prefill 的 buffer、launch 參數、cubin 選擇、block table 寬度都不改。MTP 另配一塊寬的 state，只有 MTP 的 generator 用。prefill（history 關）和驗證（history 開）都走這塊。conv prefill 的 Triton cubin 要為 `state_size = 4 + N`、`history = false` 另編一份；FLA 的 recurrent cubin 只看到 index 0 那塊 `[1, 48, 128, 128]`，slot 0 的位址不因 `history_stride` 改變，沿用現有 cubin。`seqlen <= 32` 的 conv 在 fork 走 CUDA `conv1d_update_kernel`（runtime `state_size`），不走那份 Triton constexpr。

Draft cache：和 target 一樣多的 token、cq3、`max_batch_size = 1`、沒有 recurrent state。頁號用 target page table 的整數，內容是 MTP attention 自己的 K/V。`K` 不改變 draft cache 的頁數公式，改變的是 target 的 `max_rq_tokens`（多 `N` 個 token）以及每一輪草稿 / 驗證 block table 的寬度。

## 8. Engine 計畫

新檔 `crates/berth-engine/src/mtp.rs`，在 `lib.rs` 加 `pub mod mtp`。`MtpHead` 載這些權重，prefix 是 checkpoint 裡的 39 個 `mtp.*`（`berth-plan` `from_exl3/bytes.rs` 20–26；現在 `from_exl3` 把它們排除在 decode 計畫之外，MTP 路徑另載，不把它們塞進現有 `ModelPlan`）：

- `mtp.pre_fc_norm_hidden`、`mtp.pre_fc_norm_embedding`：RMSNorm，fp16 輸出，`constant_bias = 1`
- `mtp.fc`：EXL3，K = 4，`10240 → 5120`，fp32 輸出
- `mtp.layers.0.*`：一層 full attention（q/k norm、interleaved gate、cq3 寫 draft cache）加 dense GatedMLP（`mtp_bits = 4`）
- `mtp.norm`：RMSNorm，fp16 輸出
- embedding 和 `lm_head` 借用 target 的，不另載

`MtpHead` 的方法：

- `prefill_chunk(ids, hidden, cache_seqlens)`：input layer + attention。不跑 MLP，不跑 `mtp.norm`。prompt chunk（列數通常 `> 16`）走 attention prefill；refresh（列數 `A - 1 <= K <= 15`）走 `BC_Attention`。
- `draft_step(id, hidden, cache_seqlens) -> (next_id, norm_out)`：1 列完整 forward，再 target `lm_head` argmax。`norm_out` 是 `mtp.norm` 的 fp16，給下一階。

target 側加新函數，不改現有函數的參數：

- `Model::prefill_for_mtp`：token 邊界仍是 `chunk_bounds`。第 0–62 層走現有 prefill。第 63 層跑完 mixer 和 MLP。final norm 對每一列 export。不發 lm_head。回傳每一 chunk 的 export，讓 `MtpHead::prefill_chunk` 吃 shifted hidden。block table 頁數是 `prefill_block_pages` 再加上 `N` 個 token：`ceil((prompt_len + max_new_tokens + 1 + N) / 256)`，不補 16。
- `Model::verify_step(ids: &[u32]) -> (logits [n, vocab] fp16, exported [n, hidden] fp16)`。`n = ids.len() = K + 1`。`recurrent_history = true`。block table 用驗證寬度（第 4 節），補到 16。
- GDN：新的 state 物件，`max_history = K`。新的 launch 函數傳 `history` 和真實的 `history_stride` / `state_size`。`rewind(num_tokens)` 照第 6 節的 index。現有 decode 的 `history_stride: 1` 呼叫不動。
- attention：`q_len = n` 的 launch，`quant_cache_paged` 的 `seq_len = n`。rewind 只是把 kv 長度退回接受的位置，不另寫 kernel。被拒的 cq3 列留著。

`Engine::generate`（`generate.rs` 136）在 `mtp_steps == 0` 時保持現在的形狀：`prefill` 吃 `prompt[..len-1]`，`decode_step` 吃最後一個 prompt token，然後一列一列 `sample`。`mtp_steps > 0` 時：

1. `set_prefill_pages` 用加上 `N` 的頁數。`set_decode_width` 的現有呼叫是 `prompt + max_new`，驗證和草稿的寬度每輪不一樣，在循環裡另算，不拿這個欄位硬塞。
2. `prefill_for_mtp`，接著每個 chunk `MtpHead::prefill_chunk`。單 token prompt 跳過，`mtp_last_hidden` 留空。
3. 若 `mtp_last_hidden` 是空的（單 token prompt 的第一輪），走一次現有的 1 token `decode_step` + `sample`，把那一列 final norm 留成 `mtp_last_hidden`。這次 `recurrent_history` 是假，寬 state 只有 index 0 被寫。
4. 否則草稿 K 步，再 `verify_step`，再第 5 節的接受迴圈，再 rewind，再 refresh，更新 `mtp_last_hidden`。發出的 token 仍透過 `on_token`。EOS / `max_new_tokens` 在視窗中間達到就停，不呼叫 `reject_remainder`（cache 不回捲），但仍 `rewind(0)`，`accepted > 1` 時 refresh。剩下的 draft 不算進輸出。checkpoint 邊界（`kv_position % 2048 == 0`，`cached_pages = 0`）視同拒絕。

`Engine::load` 加一個 `mtp_steps: u32`。0 就是現在的載入，不配 draft cache、不配寬 state。`> 0` 才載 `MtpHead` 和 draft cache。上限 15（`q_len = 16`）。`K >= 8` 要走 MLP 的 `Linear.forward` 路徑，文件裡寫清楚，不要假設 `run_bszN` 涵蓋所有 `n <= 16`。

`crates/berth` 只加一個旗標。`main.rs` 的 `Args` 加 `--mtp-steps`，預設 0，傳進 `Engine::load`。`ServeConfig`（`serve.rs` 38）如果要讓 HTTP 用同一個值，加一個欄位轉手，不改請求處理。其他 agent 正在改 `main.rs`、`lib.rs`、`serve.rs`，這三處只加這個 `u32`，不動旁邊的旗標。

CUDA graph 不影響 parity 要做的事：berth 保持 eager。dump 記錄的是 fork 在每個 slot warmup 之後的 replay。不要為了對齊去捕捉 graph，也不要關掉 BC。

## 9. 測試與參考

新腳本 `scripts/dump_27b_mtp_reference.py`。環境和第 1 節一樣。物件是第 1 節的 parity 組，`max_num_tokens` 用 4096，和 `dump_27b_decode_sequence.py` 的 cache 同一量級。每個 `(prompt, N, sampler)` 先跑掉兩個相同的短 job（eager，再 capture），再跑要記錄的那次。`q_len` 是 3 和 5 兩種 slot，各暖各的。

Prompt：

- 單 token `"A"`（id 32），沒有 prefill，第一輪是 1 token forward
- 100 和 1500，頁數類別在 `N = 2` 和 `4` 時不變，第 0–62 層可以和現有 prefill digest 對
- 一個會改頁數類別的長度：8190，`max_new_tokens` 至少讓生成碰到，用來確認寬 block table。這一條的第 0–62 層 digest **不要**拿舊檔當預期值
- 長度不要落在 2048 的倍數上，除非另外加一條專測 checkpoint 邊界

`num_draft_tokens`：4，以及 2。`max_history` 跟這個數走。

Sampler：一組 greedy（`ArgmaxSampler`，penalties 關）。一組採樣：temperature 0.7、seed 固定、`SamplerConfig::serve_defaults` 的 penalty（`rep_penalty = 1`、presence/frequency 0、`dry_multiplier = 0.2`、`dry_base = 1.75`、`dry_allowed_length = 6`、`dry_range = 4096`）。兩組都要，採樣那組才走完 RNG、DRY 和 past ids。`max_new_tokens` 取 32，讓 `K = 4` 至少有幾輪拒絕和幾輪多收。

每一輪記錄：

- 草稿 id，`K` 個 `u32`
- 驗證 logits 每一列的 blake2b-256（fp16 little-endian，`[1, 1, 248320]`）
- `accepted_length`
- 這一輪實際發出的 token
- rewind 之後：每個 GDN 層的 conv state（整塊，含尾巴）和 recurrent state（含 history index）的 blake2b，以及 target cache 裡這一輪碰到的 cq3 頁的 blake2b。draft cache 在 refresh 之後同樣 hash 被寫過的列

結束時記完整 token 串，和 greedy / 採樣各自的 `Finish` 原因。

hook：照 `dump_27b_prefill_layers.py` 69–100。

- `model.forward` 包一層。`recurrent_history` 為真時，把 logits clone 出來（fp16），這就是驗證的 n 列。prefill 的 `last_tokens_only` forward 不要記 logits。
- `Generator.iterate_draftmodel_mtp_gen` 包一層，回傳的 `draft_ids` clone 下來。
- `iterate` 回來之後、下一輪開始之前，從 job 的 `recurrent_state` 抄 GDN tensor，從 target / draft cache 抄碰到的頁。rewind 和 refresh 在 `iterate_gen` 返回前就做完，所以 `iterate` 回來時狀態已經是退過、refresh 過的。
- 第 0–62 層在 MTP prefill 時照現有 prefill dump 的方式包 `TransformerBlock`，寫進這份參考，和舊 digest 並列。頁數類別沒變的 prompt 應該相同。

輸出放 `target/mtp-ref/`，不進 git。進 git 的是 digest 和 id 的 json，放 `testdata/mtp_ref/`，張量本體不進。

berth GPU 測試：`crates/berth-engine/tests/mtp_ref.rs`，`BERTH_GPU_TESTS=1` 才跑。沒設環境變數就直接成功，和現有 GPU 測試一樣。testdata 不在時，開了旗標要失敗並寫「先跑 `scripts/dump_27b_mtp_reference.py`」。比對每一輪的草稿 id、logits digest、`accepted_length`、發出的 token、rewind 後的 state / cache digest，以及整段 token 串。另有一條沒開 MTP 的生成，確認和現有 `decode_seq` / `generate` 的 digest 相同。

Kernel 測試，Python 參考的寫法和 `crates/berth-kernels/tests/attn_prefill_ref.py` 一樣，Rust 側在 `BERTH_GPU_TESTS` 後面：

- GDN `run_bszN`，`seqlen = n`（至少 3 和 5），`history = true`。比 conv 輸出、recurrent index 0 和 index `1..n-1`、gated norm 之前的 core attn。
- rewind：已知 history 的 state，`rewind(R)` 和 `rewind(0)` 之後 index 0 和 conv 前 `conv_k` 個元素。index 公式用第 6 節的例子當 CPU 測試，不碰 GPU。
- attention `q_len = n`（3 和 5），cq3，比 `quant_cache_paged` 寫進的 n 列和 split/combine 的輸出。`num_splits` / `split_len` 用第 4 節的 16 頁例子做 CPU 測試。

這些 GPU 測試這份文件的任務不跑。

## 10. 實作階段

四段。每一段可以獨立交給一個 worktree。現有函數只加呼叫點；新行為放新函數。`attn.rs`、`gdn.rs`、`model.rs`、`attn_prefill.rs`、`gdn_prefill.rs` 另外有人在改，MTP 不要重排它們。

**P1，kernel。** 檔案：`crates/berth-kernels/csrc/gdn/`（新的 history launcher，不改現有 `<false>` 呼叫）、`build.rs` 的 `FORK_EXTRACTS`、`scripts/build_triton_cubins.py`（decode cubin 的 `q_len`）、`csrc/attn` 一條 `q_len = n` 的包裝。rewind 的抽出範圍：`gdn.cu` 1923–1960 兩支 kernel。`ConvRewindJob` / `StateRewindJob` 在 `gdn.cuh` 222–247，但 `gdn.cuh` 開頭就有 `at::Tensor`，整檔不能 include；結構是 POD，和 kernel 一起抽，或在 csrc 裡照欄位宣告。host wrapper `gdn.cu` 1962–2010 和 `bindings.cpp` 223–224 吃 ATen，不抽。`kernel_128` 和 `conv1d_update_kernel` 的本體已經在現有 extract 裡，這段只加實例。conv prefill 的 `state_size = 4 + N` cubin 也在這段，由 `build_triton_fla_cubins.py` 的新類別產生，不改 `state_size = 4` 的檢查。QTIP 那條若「未確認」的結論是 fork 會發 `exl3_gemv_kernel`，GEMV 的實例也放這段；結論是 coop，這段就不要做。CPU：`cargo build`、`cargo test`（不設 `BERTH_GPU_TESTS`）、`cargo clippy -p berth-kernels --all-targets -- -D warnings`。rewind 的 index 算術做成純 Rust 測試。

**P1 已完成。** 只動 `crates/berth-kernels/**` 與兩支 cubin 腳本；腳本沒有執行。既有 launcher、q_len 1 與 state_size 4 的 cubin 名稱不變。

- `FORK_EXTRACTS`：`gdn.cu` 1923–1960 → `fork/gdn_rewind_kernels.cuh`（`batched_conv_rewind_kernel`）。host wrapper 1962–2010 與 `gdn.cuh` 不抽。
- 新實例：`conv1d_update_kernel<true,true>`、`cuda_recurrent_gated_delta_rule_kernel_128<true,4>`、`exl3_gemv_kernel<4,false,2,1,0,false>`、`exl3_gemv_kernel<4,true,2,1,0,false>`。
- `extern "C"`：`berth_gdn_conv1d_update_history`、`berth_gdn_recurrent_history`、`berth_gdn_batched_conv_rewind`、`berth_gdn_batched_state_rewind`、`berth_gdn_rewind_last_error`、`berth_gdn_rewind_last_launch`、`berth_exl3_gemv_cfg`、`berth_prefill_try_qtip`。
- Rust：`conv1d_update_history`、`recurrent_gated_delta_rule_history`、`batched_conv_rewind`、`batched_state_rewind`、`rewind_last_launch`、`rewind_conv_addrs`、`rewind_state_addrs`、`exl3_gemv_cfg`、`decode_qlen_tile`、`qlen_partial_bytes`、`PagedDecode::launch_qlen`、`GdnPrefill::run_with_state_size`。
- cubin（腳本會產生、本次未跑）：`_paged_attn_decode_{split,combine}_kernel_q{2..=16}.cubin`；`conv_combined_bs{bs}[_div16]_ss{5..=19}.cubin`、`conv_{output,state}[_div16]_ss{5..=19}.cubin`。
- CPU 測試：`rewind_job_layout`、`rewind_addrs_match_section6`、`rewind_and_history_reject_before_cuda`、`exl3_gemv_cfg_matches_27b_shapes`、`qlen_constexpr_table_for_82_sms`、`split_config_16_pages_qlen_5`、`conv_key_for_keeps_state_size_4`。

**P2，MTP 頭和 target prefill export。** 檔案：新的 `crates/berth-engine/src/mtp.rs`，`model.rs` 只加 `prefill_for_mtp`，`gdn.rs` 只加寬 state 的配置函數（現有 buffer 和 launch 不動）。draft cache 用現有的 cq3 頁配置再配一份。CPU：`mtp.fc` 的形狀、`chunk_bounds` 在 `N` 底下不變、`prefill_block_pages` 加 `N` 之後的頁數（含 8190 從 32 頁變 33 頁）、carry / shift 的列數。`cargo test -p berth-engine`。不跑 GPU。

**P2 已完成。** 公開入口是 `Model::prefill_mtp`（不是這段原稿的 `prefill_for_mtp`）。`max_history == 0` 時 GDN 的 `history_stride` 仍是 1、`state_size` 仍是 4，prefill 仍走 `state_size = 4` 的 cubin。`prefill_block_pages` 沒改。沒跑 GPU。P2 當時 <= 16 列回 `Error::Input`。P3 改走 `forward_qlen`。

- `gdn_state_launch`、`GdnLayer::set_max_history`、`Model::set_max_history`。
- `AttnLayer::load_prefixed`、`AttnLayer::set_decode_pages`、`MlpLayer::load_prefixed`。
- `MtpHead::load`、`MtpHead::prefill_chunk`、`MtpHead::draft_step`、`MtpHead::draft_norm`。
- `Model::prefill_mtp`、`Model::prefill_mtp_with`。`export_norm` 關掉時，prefill 的最後一層仍只跑 mixer。
- `prefill_block_pages_mtp`。8190、`max_new_tokens = 1`、`N = 2` 是 33 頁；`prefill_block_pages(8190, 1)` 仍是 32。
- `shifted_row`、`next_carry_row`。
- kernel：`rms_norm_f16`（half in / half out / bf16 weight）、`embed_tokens_f16`（bf16 → fp32 → fp16）、`memcpy_d2d`、`cat_f16_last`。
- CPU 測試：`gdn_state_launch_matches_max_history`、`mtp_tensor_list_has_39_names`、`mtp_27b_tensors_match_the_checkpoint`、`shifted_hidden_rows`、`chunk_bounds_ignore_draft_width`、`prefill_pages_8190_grows_by_draft_tokens`。

**P3，驗證、草稿、接受、rewind。** 檔案：`generate.rs` 的 MTP 分支，`model.rs` 的 `verify_step`，`gdn.rs` 的 `rewind`，`attn.rs` 只加 `q_len = n` 的呼叫（kv rewind 是長度，不是新 kernel），`sampler.rs` 不改。`crates/berth` 的 `--mtp-steps`。CPU：接受條件、EOS 不 rewind、`kv_position % 2048 == 0` 停、`rewind` 的 slot、草稿 / 驗證 block table 寬度。用假的 logits 列，不載模型。`cargo test -p berth-engine` 和 `cargo test -p berth`。

**P3 已完成。** `mtp_steps == 0` 的 prefill / decode / generate、launch 參數、cubin 名稱、block table 寬度都沒改。沒跑 GPU，不宣稱和 fork 逐位元一致。`q_len` 2..=16 的 cubin 這輪沒有編。

- `AttnLayer::qlen_scratch`、`AttnQlenScratch`、`AttnLayer::forward_qlen`、`AttnLayer::check_qlen_pages`。
- `GdnLayer::history_scratch`、`GdnHistoryScratch`、`GdnLayer::forward_history`、`GdnLayer::rewind`、`GdnLayer::rewind_jobs`、`GdnLayer::clear_history_cursor`、`GdnLayer::check_history_rows`。`last_history` 在 history forward 成功寫完 conv 和 recurrent 之後設成 `rows - 1`。
- `MlpLayer::forward_n`（`n == 1` 走 `forward`，其餘走 `prefill`）、`Exl3Linear::forward_n`（`lm_head` 的 n 列，不配 vocab 寬的 `RowScratch`）。
- `Model::prepare_verify`、`Model::set_decode_pages`、`Model::verify_step`、`Model::rewind_mtp`。export / logit 列的拷貝是 `copy_verify_export_prefix`、`copy_verify_export_row`、`copy_verify_logit_row`。
- `MtpHead::refresh`。`prefill_chunk` 在 1..=16 列改走 `qlen_block`（`AttnLayer::forward_qlen`）；超過 16 列仍是 `AttnLayer::prefill`。
- 純函數：`padded_block_pages`、`draft_block_pages`、`verify_block_pages`、`nodraft_block_pages`、`is_checkpoint_boundary`、`accept_row`、`accept_window`、`rewind_plan`。GPU 迴圈逐列叫 `accept_row`，不叫 `accept_window`。
- `Engine::load` 加 `mtp_steps`。0 不載 MTP、不配寬 state、不配 draft cache、不配 verify scratch。1..=15 才載。`generate` 在 `mtp_steps > 0` 時走草稿、驗證、接受、rewind、refresh。
- CLI：`--mtp-steps` / `BERTH_MTP_STEPS`，預設 0，`check_config` 和 `Engine::load` 都拒絕 16 以上。
- CPU 測試：`accept_window_matches_draft`、`accept_window_rejects_mismatch`、`accept_window_accepts_all`、`accept_window_eos_does_not_rewind`、`accept_window_max_new_tokens_stops`、`accept_window_checkpoint_rejects`、`rewind_plan_matches_section6_k4`、`draft_and_verify_pages_match_section3`、`mtp_steps_parses_0_4_and_15`、`mtp_steps_rejects_16`。

P4 接了測試用 trace（`MtpTrace`、`Engine::generate_traced`）和 `tests/mtp_ref.rs`。`mtp_steps == 0` 不經過 trace。trace 沒啟用時不多做 device→host 複製，也不清 cq3。`block_in` 還沒抓。

P5 修了三處，N=2 和 N=4 的 16 個 case 和 fork 逐位元一致（logit 第 0 列 max abs diff 0）。沒有改 kernel，沒有改 `mtp_steps == 0` 的 launch、cubin、block table 寬度。

- n 列 `lm_head`（K=6）走 decode shim 的 coop GEMM，見第 4 節。
- EOS / `max_new_tokens` 不呼叫 `reject_remainder`，但仍 `rewind(0)`，`accepted > 1` 時 refresh，見第 5 節。
- `Engine::load` 在 `mtp_steps > 0` 時裝上 `dry_default_sequence_breaker_tokens`（`custom.py` 883–890）。`mtp_steps == 0` 維持 null mask。

P5 之後的非 MTP 對齊：`Engine::load` 不論 `mtp_steps` 都裝上同一份預設 breaker。`mtp_steps == 0` 的 decode 寬度改成每步 `position + 1`（fork 的當下 `len(sequence_ids)`），不再用整段 `prompt + max_new_tokens`。prefill 頁數與 MTP 分支的寬度沒改。

單 token prompt 的第一個 `decode_step` 沿用既有 1-token quant：`blocks_per_seq` 是 `num_pages`，split 用 `decode_width`。cache 剛好只有 `prompt + max_new` 時，投機窗 `K + 1` 可能在 `max_new_tokens` 之前回 `ContextFull`，不會退回 1-token decode。

**P4，參考和 GPU 測試。** 檔案：`scripts/dump_27b_mtp_reference.py`、`crates/berth-engine/tests/mtp_ref.rs`、`crates/berth-kernels/tests/` 裡第 9 節的三支。CPU：新測試在沒設 `BERTH_GPU_TESTS` 時直接成功；腳本能被 Python 語法檢查（`python3 -m py_compile`，不 import torch）。不跑 dump，不跑 GPU。

## 未確認

1. （已解決，靜態查證）QTIP narrow 會開。fork 的 `exllamav3_ext.cpython-313-x86_64-linux-gnu.so` 用 `cuobjdump --dump-resource-usage` 看：`exl3_gemv_kernel<4, fp32, 2(mul1), 1(m>1), 0(narrow), false>` 是 REG 64、SHARED 16384，block 512 threads。每個 SM 的暫存器 65536 / (512 × 64) = 2 個 block（threads 1536 允許 3、smem 也允許，暫存器先卡），`narrow_coresident = 2 × 82 = 164`。所以在 m = 3..8、K = 4、mul1、`EXL3_GEMV` 未設（mode 1）時：
   - `n = 5120` 的投影（`n / 32 = 160 <= 164`）走 QTIP narrow：MLP down（k = 17408）、attention `o_proj`（k = 6144）、GDN `o_proj`（k = 6144）、`mtp.fc`（k = 10240，refresh 的 m = A - 1）。grid `min(n / 32, occupancy × SM) = 160`，cooperative launch，block 512。
   - gate / up（n = 17408，`544 > 164`，k = 5120 過不了後面的條件）回 -1，走 coop GEMM。
   - 只有 `exl3_gemm` 會試 QTIP（`exl3_gemm.cu` 239）；sliced `exl3_mgemm` 不試。m = 1 與 2 先被 int8 sq 接走（K ≤ 5）。m ≥ 9 超過 `EXL3_GEMV_MAX_M`，是 coop。
   P1 要實例化這支 GEMV（fp16 與 fp32 輸出都要，`cb = 2`、`mmode = 1`、cfg 0；cfg 1 這組形狀用不到），launcher 照 `exl3_gemv.cu` 92–170 的 occupancy 與 cooperative launch，並讓 `exl3_linear`（`prefill_lin.rs`）在 m = 3..8 時照同一條 `exl3_gemv_cfg` 判斷，取代現在的明確拒絕。occupancy 執行時用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 算，不寫死 2。

2. graph replay 和 eager warmup 在新的 `(q_len, history)` slot 上是否逐位元相同。現有 decode dump 把兩者當同一件事，丟掉前兩次 job（`dump_27b_decode_sequence.py` 15–28）。MTP dump 沿用這個協議，記錄下來的 digest 是權威。berth 不捕捉 graph。若 replay 和 eager 不一致，會在和第 9 節的 digest 比較時露出來，那時再讀 `BC_*::run` 的 capture 分支，不在這份先假設。
