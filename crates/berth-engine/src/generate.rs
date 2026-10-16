//! One request, end to end: feed the prompt, then sample until a stop.
//!
//! A prompt longer than one token is the fork's prefill: [`Model::prefill`]
//! on every id except the last (`generator/job.py` `Job.prefill` forwards
//! `ids[:-1]`), then [`Model::decode_step`] of that last id. A one-token
//! prompt is only the decode step.
//!
//! Prompt tokens occupy positions `0..n-1` and are not sampled. The logits
//! of the last prompt token are the first sample. Each later sample is fed
//! back at the next position. Generation stops on a stop token, on
//! `max_new_tokens`, on [`Flow::Cancel`], or when the next token would not
//! fit in the cache. The stop token is reported.
//!
//! `mtp_steps == 0` keeps that loop. `mtp_steps > 0` drafts, verifies, and
//! rewinds (`docs/mtp.md` section 8). The zero path does not load the MTP head
//! and does not call [`MtpTrace`]. A traced MTP generate copies device state
//! only when [`MtpTrace::active`] is true.

use std::path::Path;

use berth_exl3::Checkpoint;
use berth_kernels::ops;
use berth_plan::build_plan;
use berth_plan::from_exl3::model_shape;

use crate::device::ptr;
use crate::model::Model;
use crate::mtp::{
    accept_row, draft_block_pages, gdn_snaps, nodraft_block_pages, union_pages, verify_block_pages,
    AcceptStep, MtpChunkEvent, MtpDraftSnap, MtpGdnSnap, MtpHead, MtpPageSnap, MtpPrefillSnap,
    MtpRound, MtpTrace, MtpVerifySnap, RowSample,
};
use crate::sampler::{Sampler, SamplerConfig};
use crate::{Error, Gpu, Result};

/// What the caller asks for.
#[derive(Clone, Debug)]
pub struct GenerateRequest {
    /// Prompt token ids, already templated and tokenized.
    pub prompt: Vec<u32>,
    pub max_new_tokens: u32,
    /// Generation ends after any of these is sampled (it is still reported).
    pub stop_tokens: Vec<u32>,
    pub sampler: SamplerConfig,
}

/// Returned by the per-token callback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flow {
    Continue,
    /// Stop now, e.g. the client closed the connection.
    Cancel,
}

/// Why generation ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// A stop token was sampled.
    Stop,
    /// `max_new_tokens` reached.
    Length,
    /// The callback returned [`Flow::Cancel`].
    Cancelled,
    /// The context (`max_tokens` of the cache) is full.
    ContextFull,
}

/// Counts for the response's `usage`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

/// The loaded model, its sampler, and the device. One request at a time.
///
/// `Send`, so the server can move it onto its own thread. Fields stay private
/// and are ordinary owned buffers; nothing here is `unsafe impl Send`.
pub struct Engine {
    gpu: Gpu,
    model: Model,
    sampler: Sampler,
    vocab: u32,
    max_tokens: u32,
    /// Stop ids the serve would put on a job. See [`Self::default_stop_tokens`].
    stop_tokens: Vec<u32>,
    /// Draft tokens per round. 0 leaves the one-token loop.
    mtp_steps: u32,
    /// MTP head and draft cache. Absent when `mtp_steps` is 0.
    head: Option<MtpHead>,
    /// fp16 `[hidden]`. Last target final-norm row the next draft reads.
    mtp_hidden: Option<berth_cuda::ByteBuf>,
    /// fp16 `[16, hidden]`. Export prefix copied before refresh.
    mtp_refresh: Option<berth_cuda::ByteBuf>,
}

/// `Engine` moves to the server thread. A non-`Send` field fails this build
/// instead of being papered over with `unsafe impl Send`.
const _: fn(&Engine) = assert_engine_send;

fn assert_engine_send<T: Send>(_: &T) {}

/// Observer for one MTP generate. `held` is the last prefill chunk, which
/// shares an iterate with the verify that follows it.
struct TraceObs<'a> {
    trace: &'a mut dyn MtpTrace,
    held: &'a mut Option<MtpPrefillSnap>,
}

impl Engine {
    /// Open the checkpoint at `model_dir` and load it with a `max_tokens` cache.
    /// `cubin_dir` holds the Triton attention cubins (`target/triton-cubins`).
    ///
    /// `mtp_steps == 0` is the current load: no MTP head, no widened GDN
    /// history, no draft cache. `1..=15` loads the head and calls
    /// [`Model::set_max_history`] then [`Model::prepare_verify`].
    ///
    /// # Errors
    ///
    /// `mtp_steps` is above 15, or the checkpoint or the device load fails.
    pub fn load(
        model_dir: &Path,
        cubin_dir: &Path,
        max_tokens: u32,
        mtp_steps: u32,
    ) -> Result<Self> {
        if mtp_steps > 15 {
            return Err(Error::Input(format!(
                "mtp-steps must be 0..=15 (got {mtp_steps})"
            )));
        }
        let stop_tokens = read_default_stop_tokens(model_dir)?;
        // Fork's `SS_DRY` with no explicit breaker set resolves the tokenizer
        // default at sample time (`custom.py` 1008–1010). Only the MTP load
        // installs that mask. `mtp_steps == 0` leaves the sampler's null mask.
        let dry_breakers = if mtp_steps == 0 {
            None
        } else {
            Some(dry_default_breaker_ids(model_dir)?)
        };
        let ckpt = Checkpoint::open(model_dir)?;
        let shape = model_shape(&ckpt.config).map_err(|err| Error::Weights(err.to_string()))?;
        let plan = build_plan(&shape).map_err(|err| Error::Weights(err.to_string()))?;
        let gpu = Gpu::new()?;
        let mut model = Model::load_with_cubins(&gpu, &ckpt, &plan, max_tokens, cubin_dir)?;
        let mut sampler = Sampler::new(&gpu, plan.vocab_size, max_tokens)?;
        if let Some(ids) = &dry_breakers {
            sampler.set_sequence_breakers(&gpu, ids)?;
        }
        let (head, mtp_hidden, mtp_refresh) = if mtp_steps == 0 {
            (None, None, None)
        } else {
            model.set_max_history(&gpu, mtp_steps)?;
            model.prepare_verify(&gpu)?;
            let head = MtpHead::load(&gpu, &ckpt, &shape, max_tokens, cubin_dir)?;
            let row = hidden_row_bytes(model.hidden_size())?;
            let wide = row
                .checked_mul(16)
                .ok_or_else(|| Error::Input("mtp refresh buffer overflows".into()))?;
            (Some(head), Some(gpu.zeroed(row)?), Some(gpu.zeroed(wide)?))
        };
        Ok(Self {
            gpu,
            model,
            sampler,
            vocab: plan.vocab_size,
            max_tokens,
            stop_tokens,
            mtp_steps,
            head,
            mtp_hidden,
            mtp_refresh,
        })
    }

    /// Vocabulary size of the loaded model.
    pub fn vocab_size(&self) -> u32 {
        self.vocab
    }

    /// Cache length in tokens; prompt plus output must fit.
    pub fn max_tokens(&self) -> u32 {
        self.max_tokens
    }

    /// Stop token ids, in the serve's order.
    ///
    /// `serve_openai.py` sets `stop_ids` from `model.config.eos_token_id_list`
    /// after `Tokenizer` has extended it, then appends `tokenizer.eos_token_id`
    /// when that id is missing. The list is `config.json`'s `eos_token_id`
    /// (top level, else `text_config`), plus the `tokenizer_config.json`
    /// `eos_token` piece when its id is non-zero and new, plus every
    /// `generation_config.json` `eos_token_id`, plus the tokenizer's own eos
    /// id (`tokenizer.py` `get_default_token_id`, default 2 only when the
    /// config and the tokenizer config both lack one).
    pub fn default_stop_tokens(&self) -> &[u32] {
        &self.stop_tokens
    }

    /// Run one request. Every sampled token, including a final stop token,
    /// goes to `on_token` as soon as it is sampled.
    ///
    /// Per-sequence state is cleared first: attention cache length, GDN conv
    /// and recurrent state, and [`Sampler::reset`] with this prompt. A cancel
    /// reported for a token wins over that token also being a stop token.
    /// A stop token wins over `max_new_tokens` when the same token would trip
    /// both, matching `Job.receive_token` (stop check, then the length check).
    pub fn generate(
        &mut self,
        request: &GenerateRequest,
        on_token: &mut dyn FnMut(u32) -> Flow,
    ) -> Result<(Finish, Usage)> {
        if self.mtp_steps == 0 {
            return self.generate_plain(request, on_token);
        }
        let mut trace = NopTrace;
        self.generate_traced(request, on_token, &mut trace)
    }

    /// [`Self::generate`] with an MTP observer.
    ///
    /// `mtp_steps == 0` ignores `trace` and runs the plain loop. `mtp_steps > 0`
    /// is the same draft / verify / accept path as [`Self::generate`]. Device
    /// state is copied only when `trace.active()` is true, after each fork
    /// `iterate` (a prefill-only chunk, or a verify after rewind and refresh).
    ///
    /// # Errors
    ///
    /// The same errors as [`Self::generate`].
    pub fn generate_traced(
        &mut self,
        request: &GenerateRequest,
        on_token: &mut dyn FnMut(u32) -> Flow,
        trace: &mut dyn MtpTrace,
    ) -> Result<(Finish, Usage)> {
        if self.mtp_steps == 0 {
            return self.generate_plain(request, on_token);
        }
        let prompt_len = self.check_request(request)?;
        self.model.reset_sequence(&self.gpu)?;
        // The reference zeroes KV between jobs. `reset_sequence` clears GDN
        // and the cache length, not the cq3 bytes a whole-page digest reads.
        // The no-op trace is inactive, so a normal generate does not do this.
        if trace.active() {
            self.zero_kv_for_trace()?;
        }
        self.sampler
            .reset(&self.gpu, &request.sampler, &request.prompt)?;
        self.generate_mtp(request, on_token, prompt_len, trace)
    }

    /// The `mtp_steps == 0` loop. No trace parameter, so it cannot copy for one.
    fn generate_plain(
        &mut self,
        request: &GenerateRequest,
        on_token: &mut dyn FnMut(u32) -> Flow,
    ) -> Result<(Finish, Usage)> {
        let prompt_len = self.check_request(request)?;
        self.model.reset_sequence(&self.gpu)?;
        // The fork sizes the decode block table from the job's whole length
        // (`Job.get_max_seq_len`: prompt plus max_new_tokens).
        self.model
            .set_decode_width(prompt_len.saturating_add(request.max_new_tokens));
        // Prefill sees the job's allocated pages, not the pages written so
        // far and not the 16-padded decode width. `max_rq_tokens` is
        // `max_new_tokens + 1` when the generator has no draft.
        let prefill_pages = crate::prefill::prefill_block_pages(prompt_len, request.max_new_tokens);
        self.model.set_prefill_pages(prefill_pages)?;
        self.sampler
            .reset(&self.gpu, &request.sampler, &request.prompt)?;

        let max_tokens = self.max_tokens;
        let Engine {
            gpu,
            model,
            sampler,
            ..
        } = self;
        feed_prompt(model, gpu, &request.prompt)?;

        let mut completion = 0u32;
        loop {
            let token = sampler.sample(gpu, model.logits_mut()?)?;
            completion += 1;
            let usage = Usage {
                prompt_tokens: prompt_len,
                completion_tokens: completion,
            };
            if on_token(token) == Flow::Cancel {
                return Ok((Finish::Cancelled, usage));
            }
            if request.stop_tokens.contains(&token) {
                return Ok((Finish::Stop, usage));
            }
            if completion >= request.max_new_tokens {
                return Ok((Finish::Length, usage));
            }
            // The token just sampled is decoded at this position to produce
            // the next logits. `prompt_len + completion - 1` is that index:
            // the first sample comes from the last prompt position and is
            // fed back at `prompt_len`.
            let Some(next_pos) = prompt_len.checked_add(completion - 1) else {
                return Ok((Finish::ContextFull, usage));
            };
            if next_pos >= max_tokens {
                return Ok((Finish::ContextFull, usage));
            }
            model.decode_step(gpu, token, next_pos)?;
            model.synchronize(gpu)?;
        }
    }

    fn check_request(&self, request: &GenerateRequest) -> Result<u32> {
        if request.prompt.is_empty() {
            return Err(Error::Input("prompt is empty".into()));
        }
        if request.max_new_tokens == 0 {
            return Err(Error::Input("max_new_tokens must be at least 1".into()));
        }
        let prompt_len = u32::try_from(request.prompt.len())
            .map_err(|_| Error::Input("prompt is longer than u32::MAX".into()))?;
        if prompt_len > self.max_tokens {
            return Err(Error::Input(format!(
                "prompt of {prompt_len} tokens does not fit the {}-token cache",
                self.max_tokens
            )));
        }
        for &token in &request.prompt {
            if token >= self.vocab {
                return Err(Error::Input(format!(
                    "token {token} is past the vocabulary ({})",
                    self.vocab
                )));
            }
        }
        Ok(prompt_len)
    }

    fn generate_mtp(
        &mut self,
        request: &GenerateRequest,
        on_token: &mut dyn FnMut(u32) -> Flow,
        prompt_len: u32,
        trace: &mut dyn MtpTrace,
    ) -> Result<(Finish, Usage)> {
        let mut completion = 0u32;
        let mut held = None;
        let mut obs = TraceObs {
            trace,
            held: &mut held,
        };
        match self.prime_prompt(request, on_token, prompt_len, &mut completion, &mut obs)? {
            Prime::Done(finish) => Ok((finish, usage(prompt_len, completion))),
            Prime::Live(cursor) => self.speculate(
                request,
                on_token,
                prompt_len,
                cursor,
                &mut completion,
                &mut obs,
            ),
        }
    }

    fn prime_prompt(
        &mut self,
        request: &GenerateRequest,
        on_token: &mut dyn FnMut(u32) -> Flow,
        prompt_len: u32,
        completion: &mut u32,
        obs: &mut TraceObs<'_>,
    ) -> Result<Prime> {
        if prompt_len == 1 {
            return self.prime_one_token(request, on_token, completion, obs.trace);
        }
        let pages = crate::prefill::prefill_block_pages_mtp(
            prompt_len,
            request.max_new_tokens,
            self.mtp_steps,
        );
        let prefix =
            &request.prompt[..usize::try_from(prompt_len).expect("prompt len fits usize") - 1];
        let capture = obs.trace.active();
        let carry = {
            let Engine {
                gpu, model, head, ..
            } = self;
            let head = head
                .as_mut()
                .ok_or_else(|| Error::Input("mtp head is not loaded".into()))?;
            let carried = if capture {
                model.prefill_mtp_traced(
                    gpu,
                    prefix,
                    head,
                    pages,
                    None,
                    None,
                    Some(&mut |event| {
                        match event {
                            MtpChunkEvent::Round(round) => obs.trace.on_round(&round),
                            MtpChunkEvent::Hold(prefill) => *obs.held = Some(prefill),
                        }
                        Ok(())
                    }),
                )?
            } else {
                model.prefill_mtp(gpu, prefix, head, pages)?
            };
            carried.ok_or_else(|| Error::Input("mtp prefill returned no carry".into()))?
        };
        self.mtp_hidden = Some(carry);
        let last = usize::try_from(prompt_len).expect("prompt len fits usize") - 1;
        Ok(Prime::Live(LiveCursor {
            pending: request.prompt[last],
            kv: prompt_len - 1,
        }))
    }

    fn prime_one_token(
        &mut self,
        request: &GenerateRequest,
        on_token: &mut dyn FnMut(u32) -> Flow,
        completion: &mut u32,
        trace: &mut dyn MtpTrace,
    ) -> Result<Prime> {
        let pages = nodraft_block_pages(1, self.mtp_steps)?;
        // quant_cache on the existing 1-token path keeps blocks_per_seq at
        // num_pages while the split reads decode_width. This step calls
        // decode_step, so it inherits that. forward_qlen passes one width.
        self.model.set_decode_pages(pages)?;
        self.model.decode_step(&self.gpu, request.prompt[0], 0)?;
        self.model.synchronize(&self.gpu)?;
        // The sampler overwrites this row. The trace wants the forward's logits.
        let logit_row = if trace.active() {
            Some(self.copy_decode_logits()?)
        } else {
            None
        };
        self.copy_final_norm()?;
        let token = {
            let Engine {
                gpu,
                model,
                sampler,
                ..
            } = self;
            sampler.sample(gpu, model.logits_mut()?)?
        };
        *completion += 1;
        let flow = on_token(token);
        if let Some(row) = logit_row {
            let round = self.trace_bootstrap(row, vec![token])?;
            trace.on_round(&round);
        }
        if flow == Flow::Cancel {
            return Ok(Prime::Done(Finish::Cancelled));
        }
        if request.stop_tokens.contains(&token) {
            return Ok(Prime::Done(Finish::Stop));
        }
        if *completion >= request.max_new_tokens {
            return Ok(Prime::Done(Finish::Length));
        }
        Ok(Prime::Live(LiveCursor {
            pending: token,
            kv: 1,
        }))
    }

    fn speculate(
        &mut self,
        request: &GenerateRequest,
        on_token: &mut dyn FnMut(u32) -> Flow,
        prompt_len: u32,
        mut cursor: LiveCursor,
        completion: &mut u32,
        obs: &mut TraceObs<'_>,
    ) -> Result<(Finish, Usage)> {
        let k = self.mtp_steps;
        let n = k + 1;
        let max_tokens = self.max_tokens;
        loop {
            // K+1 tokens have to fit. A cache sized to prompt + max_new can
            // end the request before max_new_tokens. No 1-token fallback.
            if cursor
                .kv
                .checked_add(n)
                .map(|end| end > max_tokens)
                .unwrap_or(true)
            {
                return Ok((Finish::ContextFull, usage(prompt_len, *completion)));
            }
            let drafts = self.draft_k(cursor.pending, cursor.kv, k)?;
            let cur_len = cursor.kv.checked_add(1).expect("kv + n fit in u32");
            let pages = verify_block_pages(cur_len, k, k)?;
            self.model.set_decode_pages(pages)?;
            let mut ids = Vec::with_capacity(drafts.len() + 1);
            ids.push(cursor.pending);
            ids.extend_from_slice(&drafts);
            self.model.verify_step(&self.gpu, &ids, cursor.kv, pages)?;
            let logit_rows = if obs.trace.active() {
                Some(self.model.copy_verify_logit_rows(&self.gpu, n)?)
            } else {
                None
            };
            let round = VerifyRound {
                request,
                drafts: &drafts,
                kv: cursor.kv,
                n,
                pages,
            };
            let mut emitted = Vec::new();
            match self.walk_verify(&round, on_token, completion, &mut emitted)? {
                Walk::Done { finish, accepted } => {
                    if let Some(rows) = logit_rows {
                        let refresh = match accepted {
                            Some(kept) if kept > 1 => Some((
                                cursor.kv.checked_add(1).ok_or_else(|| {
                                    Error::Input("refresh start overflows".into())
                                })?,
                                kept - 1,
                            )),
                            _ => None,
                        };
                        self.emit_verify(
                            obs.trace,
                            obs.held,
                            verify_snap(cursor.kv, n, true, Some(drafts), rows),
                            emitted,
                            refresh,
                        )?;
                    }
                    return Ok((finish, usage(prompt_len, *completion)));
                }
                Walk::Next {
                    cursor: next,
                    accepted,
                } => {
                    if let Some(rows) = logit_rows {
                        let refresh = if accepted > 1 {
                            Some((
                                cursor.kv.checked_add(1).ok_or_else(|| {
                                    Error::Input("refresh start overflows".into())
                                })?,
                                accepted - 1,
                            ))
                        } else {
                            None
                        };
                        self.emit_verify(
                            obs.trace,
                            obs.held,
                            verify_snap(cursor.kv, n, true, Some(drafts), rows),
                            emitted,
                            refresh,
                        )?;
                    }
                    cursor = next;
                }
            }
        }
    }

    fn draft_k(&mut self, pending: u32, kv: u32, k: u32) -> Result<Vec<u32>> {
        let cur_len = kv
            .checked_add(1)
            .ok_or_else(|| Error::Input("draft cur_len overflows".into()))?;
        let pages = draft_block_pages(cur_len, k)?;
        let mut drafts = Vec::with_capacity(usize::try_from(k).unwrap_or(0));
        let mut id = pending;
        for idx in 0..k {
            let position = kv
                .checked_add(idx)
                .ok_or_else(|| Error::Input("draft position overflows".into()))?;
            let token = self.one_draft(id, position, pages)?;
            drafts.push(token);
            self.copy_draft_norm()?;
            id = token;
        }
        Ok(drafts)
    }

    fn one_draft(&mut self, id: u32, position: u32, pages: u32) -> Result<u32> {
        let Engine {
            gpu,
            model,
            head,
            mtp_hidden,
            ..
        } = self;
        let head = head
            .as_mut()
            .ok_or_else(|| Error::Input("mtp head is not loaded".into()))?;
        let hidden = mtp_hidden
            .as_ref()
            .ok_or_else(|| Error::Input("mtp hidden is not allocated".into()))?;
        head.draft_step(gpu, model, id, hidden, position, pages)
    }

    fn copy_draft_norm(&mut self) -> Result<()> {
        let Engine {
            gpu,
            head,
            mtp_hidden,
            ..
        } = self;
        let src = head
            .as_ref()
            .ok_or_else(|| Error::Input("mtp head is not loaded".into()))?
            .draft_norm();
        let dst = mtp_hidden
            .as_mut()
            .ok_or_else(|| Error::Input("mtp hidden is not allocated".into()))?;
        copy_row(gpu, src, dst, "mtp draft norm")
    }

    fn copy_final_norm(&mut self) -> Result<()> {
        let row = hidden_row_bytes(self.model.hidden_size())?;
        let Engine {
            gpu,
            model,
            mtp_hidden,
            ..
        } = self;
        let src = model.final_norm()?;
        let dst = mtp_hidden
            .as_mut()
            .ok_or_else(|| Error::Input("mtp hidden is not allocated".into()))?;
        if src.num_bytes() < row || dst.num_bytes() < row {
            return Err(Error::Input(
                "mtp hidden row is shorter than the final norm".into(),
            ));
        }
        let n = u64::try_from(row)
            .map_err(|_| Error::Input("mtp hidden row does not fit in u64".into()))?;
        // SAFETY: the two buffers are distinct device rows. The copy is
        // ordered before the sampler reads logits on this stream.
        unsafe {
            ops::memcpy_d2d(ptr(dst), ptr(src), n, gpu.raw_stream())?;
        }
        Ok(())
    }

    fn sample_verify_row(&mut self, index: u32) -> Result<u32> {
        self.model.copy_verify_logit_row(&self.gpu, index)?;
        let Engine {
            gpu,
            model,
            sampler,
            ..
        } = self;
        sampler.sample(gpu, model.verify_logit_row_mut()?)
    }

    fn walk_verify(
        &mut self,
        round: &VerifyRound<'_>,
        on_token: &mut dyn FnMut(u32) -> Flow,
        completion: &mut u32,
        emitted: &mut Vec<u32>,
    ) -> Result<Walk> {
        for i in 0..round.n {
            let token = self.sample_verify_row(i)?;
            let step = classify_row(round, token, i, *completion)?;
            *completion += 1;
            emitted.push(token);
            if on_token(token) == Flow::Cancel {
                return Ok(Walk::Done {
                    finish: Finish::Cancelled,
                    accepted: None,
                });
            }
            let rejected = match step {
                AcceptStep::Continue => None,
                AcceptStep::Halt => {
                    // `generator.py` 1192–1202 breaks without calling
                    // `reject_remainder`, so `rejected` stays 0. The
                    // fallthrough at 1244–1246 still runs `rewind(0)`.
                    // Cache length is not rolled back. Refresh still runs
                    // when more than the anchor was kept (1301–1333).
                    let accepted = i + 1;
                    self.model.rewind_mtp(&self.gpu, 0)?;
                    if accepted > 1 {
                        self.refresh_draft(round.drafts, accepted, round.kv, round.pages)?;
                    }
                    self.store_export_row(accepted - 1)?;
                    return Ok(Walk::Done {
                        finish: finish_halt(round.request, token),
                        accepted: Some(accepted),
                    });
                }
                AcceptStep::Reject { num_rejected } => Some(num_rejected),
                AcceptStep::AcceptAll => Some(0),
            };
            if let Some(num_rejected) = rejected {
                let accepted = round.n.checked_sub(num_rejected).ok_or_else(|| {
                    Error::Input("rewind rejected more tokens than the verify window".into())
                })?;
                let next = self.finish_accept(num_rejected, token, round)?;
                return Ok(Walk::Next {
                    cursor: next,
                    accepted,
                });
            }
        }
        Err(Error::Input(
            "verify window ended without a decision".into(),
        ))
    }

    fn finish_accept(
        &mut self,
        num_rejected: u32,
        token: u32,
        round: &VerifyRound<'_>,
    ) -> Result<LiveCursor> {
        self.model.rewind_mtp(&self.gpu, num_rejected)?;
        let accepted = round.n.checked_sub(num_rejected).ok_or_else(|| {
            Error::Input("rewind rejected more tokens than the verify window".into())
        })?;
        if accepted == 0 {
            return Err(Error::Input(
                "rewind rejected the whole verify window".into(),
            ));
        }
        if accepted > 1 {
            self.refresh_draft(round.drafts, accepted, round.kv, round.pages)?;
        }
        self.store_export_row(accepted - 1)?;
        let kv = round
            .kv
            .checked_add(accepted)
            .ok_or_else(|| Error::Input("verify cache length overflows".into()))?;
        Ok(LiveCursor { pending: token, kv })
    }

    fn refresh_draft(&mut self, drafts: &[u32], accepted: u32, kv: u32, pages: u32) -> Result<()> {
        let rows = accepted - 1;
        let ids = drafts
            .get(..usize::try_from(rows).unwrap_or(usize::MAX))
            .ok_or_else(|| {
                Error::Input("refresh ids are shorter than the accepted drafts".into())
            })?;
        {
            let Engine {
                gpu,
                model,
                mtp_refresh,
                ..
            } = self;
            let dst = mtp_refresh
                .as_mut()
                .ok_or_else(|| Error::Input("mtp refresh buffer is not allocated".into()))?;
            model.copy_verify_export_prefix(gpu, dst, rows)?;
        }
        let start = kv
            .checked_add(1)
            .ok_or_else(|| Error::Input("refresh start overflows".into()))?;
        let Engine {
            gpu,
            model,
            head,
            mtp_refresh,
            ..
        } = self;
        let head = head
            .as_mut()
            .ok_or_else(|| Error::Input("mtp head is not loaded".into()))?;
        let hidden = mtp_refresh
            .as_ref()
            .ok_or_else(|| Error::Input("mtp refresh buffer is not allocated".into()))?;
        head.refresh(gpu, model, ids, hidden, start, pages)
    }

    fn store_export_row(&mut self, index: u32) -> Result<()> {
        let Engine {
            gpu,
            model,
            mtp_hidden,
            ..
        } = self;
        let dst = mtp_hidden
            .as_mut()
            .ok_or_else(|| Error::Input("mtp hidden is not allocated".into()))?;
        model.copy_verify_export_row(gpu, dst, index)
    }

    fn zero_kv_for_trace(&mut self) -> Result<()> {
        self.model.zero_attn_pages(&self.gpu)?;
        if let Some(head) = self.head.as_mut() {
            head.zero_cache(&self.gpu)?;
        }
        Ok(())
    }

    fn copy_decode_logits(&self) -> Result<Vec<u8>> {
        let logits = self.model.logits()?;
        let mut bytes = vec![0u8; logits.num_bytes()];
        logits.copy_to_host(&self.gpu.stream, &mut bytes)?;
        Ok(bytes)
    }

    fn trace_bootstrap(&self, logit_row: Vec<u8>, emitted: Vec<u32>) -> Result<MtpRound> {
        let (gdn, target_pages, draft_refresh) = self.trace_tail(&[(0, 1)], None)?;
        Ok(MtpRound {
            prefill: None,
            verify: Some(MtpVerifySnap {
                kv_pos: 0,
                q_len: 1,
                recurrent_history: false,
                draft_ids: None,
                logit_rows: vec![logit_row],
            }),
            emitted,
            gdn,
            target_pages,
            draft_refresh,
        })
    }

    fn emit_verify(
        &mut self,
        trace: &mut dyn MtpTrace,
        held: &mut Option<MtpPrefillSnap>,
        verify: MtpVerifySnap,
        emitted: Vec<u32>,
        refresh: Option<(u32, u32)>,
    ) -> Result<()> {
        let mut spans = Vec::new();
        if let Some(prefill) = held.as_ref() {
            spans.push((prefill.start, prefill.end));
        }
        let end = verify
            .kv_pos
            .checked_add(verify.q_len)
            .ok_or_else(|| Error::Input("verify span end overflows".into()))?;
        spans.push((verify.kv_pos, end));
        let (gdn, target_pages, draft_refresh) = self.trace_tail(&spans, refresh)?;
        trace.on_round(&MtpRound {
            prefill: held.take(),
            verify: Some(verify),
            emitted,
            gdn,
            target_pages,
            draft_refresh,
        });
        Ok(())
    }

    fn trace_tail(
        &self,
        spans: &[(u32, u32)],
        refresh: Option<(u32, u32)>,
    ) -> Result<(Vec<MtpGdnSnap>, Vec<MtpPageSnap>, Option<MtpDraftSnap>)> {
        let gdn = gdn_snaps(self.model.gdn_state(&self.gpu)?);
        let target_pages = self
            .model
            .mtp_target_pages(&self.gpu, &union_pages(spans))?;
        let draft_refresh = match refresh {
            Some((start, length)) => {
                let head = self
                    .head
                    .as_ref()
                    .ok_or_else(|| Error::Input("mtp head is not loaded".into()))?;
                Some(head.draft_span(&self.gpu, start, length)?)
            }
            None => None,
        };
        Ok((gdn, target_pages, draft_refresh))
    }
}

/// [`MtpTrace`] that stays inactive. [`Engine::generate`] uses this so the
/// MTP loop has one path and still does no extra device-to-host copy.
struct NopTrace;

impl MtpTrace for NopTrace {}

fn verify_snap(
    kv_pos: u32,
    q_len: u32,
    recurrent_history: bool,
    draft_ids: Option<Vec<u32>>,
    logit_rows: Vec<Vec<u8>>,
) -> MtpVerifySnap {
    MtpVerifySnap {
        kv_pos,
        q_len,
        recurrent_history,
        draft_ids,
        logit_rows,
    }
}

/// Where the MTP loop is after the prompt is in the cache.
enum Prime {
    Done(Finish),
    Live(LiveCursor),
}

/// Token not yet in the cache, and the cache length it will be written at.
struct LiveCursor {
    pending: u32,
    kv: u32,
}

/// One verify window, either finished or ready for another round.
enum Walk {
    /// `accepted` is set when the fork's `rejected == 0` path ran `rewind(0)`.
    /// `None` is a caller cancel, which does not rewind.
    Done {
        finish: Finish,
        accepted: Option<u32>,
    },
    /// `accepted` is the tokens kept from this window, including the anchor.
    Next { cursor: LiveCursor, accepted: u32 },
}

/// Borrowed inputs of one verify accept loop.
struct VerifyRound<'a> {
    request: &'a GenerateRequest,
    drafts: &'a [u32],
    kv: u32,
    n: u32,
    pages: u32,
}

fn usage(prompt_len: u32, completion: u32) -> Usage {
    Usage {
        prompt_tokens: prompt_len,
        completion_tokens: completion,
    }
}

fn finish_halt(request: &GenerateRequest, token: u32) -> Finish {
    if request.stop_tokens.contains(&token) {
        Finish::Stop
    } else {
        Finish::Length
    }
}

fn classify_row(
    round: &VerifyRound<'_>,
    token: u32,
    i: u32,
    completion: u32,
) -> Result<AcceptStep> {
    let seq_pos = round
        .kv
        .checked_add(i)
        .and_then(|pos| pos.checked_add(1))
        .ok_or_else(|| Error::Input("verify seq_pos overflows".into()))?;
    let hits_length = completion
        .checked_add(1)
        .map(|next| next >= round.request.max_new_tokens)
        .unwrap_or(true);
    let row = RowSample {
        token,
        is_stop: round.request.stop_tokens.contains(&token),
        hits_length,
        seq_pos,
    };
    let draft = if i + 1 < round.n {
        Some(round.drafts[usize::try_from(i).expect("row index fits usize")])
    } else {
        None
    };
    accept_row(i, round.n, draft, &row)
}

fn hidden_row_bytes(hidden: i32) -> Result<usize> {
    let width = usize::try_from(hidden).unwrap_or(0);
    width
        .checked_mul(2)
        .ok_or_else(|| Error::Input("mtp hidden row overflows".into()))
}

fn copy_row(
    gpu: &Gpu,
    src: &berth_cuda::ByteBuf,
    dst: &mut berth_cuda::ByteBuf,
    what: &str,
) -> Result<()> {
    let row = src.num_bytes();
    if row == 0 || dst.num_bytes() < row {
        return Err(Error::Input(format!(
            "{what} does not fit the hidden buffer"
        )));
    }
    let n = u64::try_from(row).map_err(|_| Error::Input(format!("{what} does not fit in u64")))?;
    // SAFETY: the buffers do not overlap. `row` is the source length and
    // fits in the destination. Both stay allocated until `gpu.stream` finishes.
    unsafe {
        ops::memcpy_d2d(ptr(dst), ptr(src), n, gpu.raw_stream())?;
    }
    Ok(())
}

/// Prompt tokens at positions `0..prompt.len()`.
///
/// A longer prompt is `Job.prefill` (`generator/job.py`): [`Model::prefill`]
/// on `prompt[..len - 1]`, then one [`Model::decode_step`] of the last id at
/// position `len - 1`. A one-token prompt is only that decode step. The step
/// is synchronized so the logits are finished before sampling.
fn feed_prompt(model: &mut Model, gpu: &Gpu, prompt: &[u32]) -> Result<()> {
    if prompt.len() > 1 {
        model.prefill(gpu, &prompt[..prompt.len() - 1])?;
    }
    let last = *prompt.last().expect("prompt is non-empty");
    let position = u32::try_from(prompt.len() - 1).expect("prompt length fits u32");
    model.decode_step(gpu, last, position)?;
    model.synchronize(gpu)?;
    Ok(())
}

/// Stop ids in the order `serve_openai.py` builds `self.stop_ids`.
///
/// Fork:
/// * `model/config.py` reads `eos_token_id`, else `text_config->eos_token_id`,
///   and stores a list with `None` entries removed. A list's first element,
///   before that filter, is `eos_token_id`.
/// * `tokenizer/tokenizer.py` resolves `tokenizer_config.json` `eos_token`
///   through `token_to_id`. A non-zero id missing from the list is appended
///   (`if e:`). `generation_config.json` `eos_token_id` is appended the same
///   way. The tokenizer's own `eos_token_id` stays at the config value when
///   the config had one; otherwise it is the resolved piece, or `2` when the
///   tokenizer config has no `eos_token` key.
/// * `serve_openai.py` copies `eos_token_id_list` and appends
///   `tokenizer.eos_token_id` when it is not `None` and not already listed.
fn read_default_stop_tokens(model_dir: &Path) -> Result<Vec<u32>> {
    let config = read_required_json(&model_dir.join("config.json"))?;
    let (mut list, config_first) = config_eos(&config)?;
    let tokenizer = read_optional_json(&model_dir.join("tokenizer_config.json"))?;
    let pieces = piece_map(tokenizer.as_ref())?;
    let piece_id = match tokenizer_eos_piece(tokenizer.as_ref()) {
        Some(piece) => lookup_piece(model_dir, piece, &pieces)?,
        None => None,
    };
    // `if e:` in `tokenizer.py`. Id 0 is a real token but it is skipped here.
    if let Some(id) = piece_id {
        if id != 0 {
            push_new(&mut list, id);
        }
    }
    if let Some(generation) = read_optional_json(&model_dir.join("generation_config.json"))? {
        append_generation_eos(&mut list, &generation)?;
    }
    let serve_eos = match config_first {
        Some(id) => Some(id),
        None if tokenizer_eos_key_absent(tokenizer.as_ref()) => Some(2),
        None => piece_id,
    };
    if let Some(id) = serve_eos {
        push_new(&mut list, id);
    }
    Ok(list)
}

fn push_new(list: &mut Vec<u32>, id: u32) {
    if !list.contains(&id) {
        list.push(id);
    }
}

fn config_eos(root: &serde_json::Value) -> Result<(Vec<u32>, Option<u32>)> {
    let Some(value) = first_present(root, &["eos_token_id", "text_config->eos_token_id"]) else {
        return Ok((Vec::new(), None));
    };
    match value {
        serde_json::Value::Array(items) if items.is_empty() => Ok((Vec::new(), None)),
        serde_json::Value::Array(items) => {
            let first = if items[0].is_null() {
                None
            } else {
                Some(json_u32(&items[0], "eos_token_id")?)
            };
            let mut list = Vec::new();
            for (index, item) in items.iter().enumerate() {
                if item.is_null() {
                    continue;
                }
                list.push(json_u32(item, &format!("eos_token_id[{index}]"))?);
            }
            Ok((list, first))
        }
        other => {
            let id = json_u32(other, "eos_token_id")?;
            Ok((vec![id], Some(id)))
        }
    }
}

fn append_generation_eos(list: &mut Vec<u32>, root: &serde_json::Value) -> Result<()> {
    let Some(value) = root.get("eos_token_id") else {
        return Ok(());
    };
    // `if ee:` is false for null, for 0, and for an empty list.
    if value.is_null() {
        return Ok(());
    }
    match value {
        serde_json::Value::Array(items) if items.is_empty() => Ok(()),
        serde_json::Value::Array(items) => {
            for (index, item) in items.iter().enumerate() {
                if item.is_null() {
                    continue;
                }
                let id = json_u32(
                    item,
                    &format!("generation_config.json eos_token_id[{index}]"),
                )?;
                push_new(list, id);
            }
            Ok(())
        }
        serde_json::Value::Number(number)
            if number.as_u64() == Some(0) || number.as_i64() == Some(0) =>
        {
            Ok(())
        }
        other => {
            push_new(
                list,
                json_u32(other, "generation_config.json eos_token_id")?,
            );
            Ok(())
        }
    }
}

fn tokenizer_eos_key_absent(config: Option<&serde_json::Value>) -> bool {
    match config {
        None => true,
        Some(value) => value.get("eos_token").is_none(),
    }
}

fn tokenizer_eos_piece(config: Option<&serde_json::Value>) -> Option<&str> {
    let value = config?.get("eos_token")?;
    match value {
        serde_json::Value::String(text) => Some(text.as_str()),
        serde_json::Value::Object(map) => map.get("content").and_then(serde_json::Value::as_str),
        _ => None,
    }
}

fn piece_map(config: Option<&serde_json::Value>) -> Result<std::collections::HashMap<String, u32>> {
    let mut map = std::collections::HashMap::new();
    let Some(config) = config else {
        return Ok(map);
    };
    let Some(decoder) = config.get("added_tokens_decoder") else {
        return Ok(map);
    };
    if decoder.is_null() {
        return Ok(map);
    }
    let obj = decoder.as_object().ok_or_else(|| {
        Error::Weights("tokenizer_config.json added_tokens_decoder is not an object".into())
    })?;
    for (key, spec) in obj {
        let id = key.parse::<u32>().map_err(|_| {
            Error::Weights(format!(
                "added_tokens_decoder key `{key}` is not a u32 token id"
            ))
        })?;
        let content = spec
            .get("content")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                Error::Weights(format!("added_tokens_decoder[{key}] has no string content"))
            })?;
        match map.entry(content.to_string()) {
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(id);
            }
            std::collections::hash_map::Entry::Occupied(slot) if *slot.get() == id => {}
            std::collections::hash_map::Entry::Occupied(slot) => {
                return Err(Error::Weights(format!(
                    "added token `{content}` is id {} and {id}",
                    slot.get()
                )));
            }
        }
    }
    Ok(map)
}

/// `Tokenizer.token_to_id` for a special piece.
///
/// `added_tokens_decoder` in `tokenizer_config.json` is the table the 27B
/// uses (`<|im_end|>` is 248046). A piece that is only in `tokenizer.json`'s
/// `added_tokens` array is resolved from that file, which is what the
/// HuggingFace tokenizer reads.
fn lookup_piece(
    model_dir: &Path,
    piece: &str,
    known: &std::collections::HashMap<String, u32>,
) -> Result<Option<u32>> {
    if let Some(id) = known.get(piece) {
        return Ok(Some(*id));
    }
    let Some(value) = read_optional_json(&model_dir.join("tokenizer.json"))? else {
        return Ok(None);
    };
    let Some(added) = value
        .get("added_tokens")
        .and_then(serde_json::Value::as_array)
    else {
        return Ok(None);
    };
    for item in added {
        let Some(content) = item.get("content").and_then(serde_json::Value::as_str) else {
            continue;
        };
        if content != piece {
            continue;
        }
        let id = item.get("id").ok_or_else(|| {
            Error::Weights(format!("tokenizer.json added token `{piece}` has no id"))
        })?;
        return Ok(Some(json_u32(id, "tokenizer.json added token id")?));
    }
    Ok(None)
}

fn first_present<'a>(root: &'a serde_json::Value, keys: &[&str]) -> Option<&'a serde_json::Value> {
    keys.iter().find_map(|key| dig(root, key))
}

fn dig<'a>(root: &'a serde_json::Value, key: &str) -> Option<&'a serde_json::Value> {
    let mut cur = root;
    for part in key.split("->") {
        match cur.get(part) {
            Some(serde_json::Value::Null) | None => return None,
            Some(next) => cur = next,
        }
    }
    Some(cur)
}

fn json_u32(value: &serde_json::Value, what: &str) -> Result<u32> {
    if let Some(number) = value.as_u64() {
        return u32::try_from(number)
            .map_err(|_| Error::Weights(format!("{what} {number} does not fit in u32")));
    }
    if let Some(number) = value.as_i64() {
        return u32::try_from(number)
            .map_err(|_| Error::Weights(format!("{what} {number} is not a token id")));
    }
    if let Some(number) = value.as_f64() {
        let max = f64::from(u32::MAX);
        // `1.0` is a JSON float. An integral value in range is still a token id.
        if number.is_finite() && number.fract() == 0.0 && (0.0..=max).contains(&number) {
            let text = format!("{number:.0}");
            return text
                .parse::<u32>()
                .map_err(|_| Error::Weights(format!("{what} {number} does not fit in u32")));
        }
    }
    Err(Error::Weights(format!("{what} is not an integer token id")))
}

fn read_required_json(path: &Path) -> Result<serde_json::Value> {
    match read_optional_json(path)? {
        Some(value) => Ok(value),
        None => Err(Error::Read {
            name: path.display().to_string(),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "file not found"),
        }),
    }
}

/// Token ids `SS_DRY` treats as sequence breakers when none were passed.
///
/// `dry_default_sequence_breaker_tokens` (`custom.py` 883–890): every piece
/// from `Tokenizer.get_id_to_piece_list(include_special_tokens=True)` that
/// contains one of `.,!?<>[](){}\n\t"`, plus every `extended_id_to_piece`
/// id. Pieces for the base vocab are `get_fixed_vocab` (`tokenizer.py`
/// 597–617), which decodes each id (prefix-stripped when `" t"` is not one
/// token). The returned ids are sorted and unique.
fn dry_default_breaker_ids(model_dir: &Path) -> Result<Vec<u32>> {
    let path = model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&path)
        .map_err(|err| Error::Weights(format!("tokenizer.json: {err}")))?;
    let raw = tokenizer.get_vocab_size(true);
    let pieces = fixed_vocab_pieces(&tokenizer, raw)?;
    let mut ids = std::collections::BTreeSet::new();
    for (index, piece) in pieces.iter().enumerate() {
        if piece_has_dry_breaker(piece) {
            let id = u32::try_from(index)
                .map_err(|_| Error::Weights("tokenizer vocab does not fit in u32".into()))?;
            ids.insert(id);
        }
    }
    for id in extended_breaker_ids(model_dir, &tokenizer)? {
        ids.insert(id);
    }
    Ok(ids.into_iter().collect())
}

/// `.,!?<>[](){}\n\t"`. Each entry is one character, so a substring check is
/// a character check.
fn piece_has_dry_breaker(piece: &str) -> bool {
    piece.chars().any(|ch| {
        matches!(
            ch,
            '.' | ','
                | '!'
                | '?'
                | '<'
                | '>'
                | '['
                | ']'
                | '('
                | ')'
                | '{'
                | '}'
                | '\n'
                | '\t'
                | '"'
        )
    })
}

/// `Tokenizer.get_fixed_vocab` (`tokenizer.py` 597–617).
fn fixed_vocab_pieces(tokenizer: &tokenizers::Tokenizer, raw: usize) -> Result<Vec<String>> {
    let probe = tokenizer
        .encode(" t", false)
        .map_err(|err| Error::Weights(format!("tokenizer encode: {err}")))?;
    let probe_ids = probe.get_ids();
    if probe_ids.len() == 1 {
        // `decode([test_id])` defaults to skip_special_tokens=True (`tokenizer.py` 602).
        let decoded = tokenizer
            .decode(&[probe_ids[0]], true)
            .map_err(|err| Error::Weights(format!("tokenizer decode: {err}")))?;
        // Python `len(" t")` is 2 code points.
        if decoded.chars().count() == 2 {
            return decode_vocab(tokenizer, raw, None, false);
        }
    }
    let space = tokenizer
        .encode(" ", false)
        .map_err(|err| Error::Weights(format!("tokenizer encode: {err}")))?;
    let prefix_id = space
        .get_ids()
        .first()
        .copied()
        .ok_or_else(|| Error::Weights("tokenizer encoded a space to zero tokens".into()))?;
    // `decode([prefix_id])` defaults to skipping special tokens.
    let prefix_piece = tokenizer
        .decode(&[prefix_id], true)
        .map_err(|err| Error::Weights(format!("tokenizer decode: {err}")))?;
    let prefix_len = prefix_piece.chars().count();
    let paired = decode_vocab(tokenizer, raw, Some(prefix_id), true)?;
    Ok(paired
        .into_iter()
        .map(|text| text.chars().skip(prefix_len).collect())
        .collect())
}

fn decode_vocab(
    tokenizer: &tokenizers::Tokenizer,
    raw: usize,
    prefix: Option<u32>,
    skip_special_tokens: bool,
) -> Result<Vec<String>> {
    let mut out = Vec::with_capacity(raw);
    let mut owned = Vec::new();
    let mut index = 0usize;
    while index < raw {
        let end = (index + 4096).min(raw);
        owned.clear();
        for token in index..end {
            let id = u32::try_from(token)
                .map_err(|_| Error::Weights("tokenizer vocab does not fit in u32".into()))?;
            match prefix {
                Some(prefix_id) => owned.push(vec![prefix_id, id]),
                None => owned.push(vec![id]),
            }
        }
        let view: Vec<&[u32]> = owned.iter().map(Vec::as_slice).collect();
        let decoded = tokenizer
            .decode_batch(&view, skip_special_tokens)
            .map_err(|err| Error::Weights(format!("tokenizer decode: {err}")))?;
        out.extend(decoded);
        index = end;
    }
    Ok(out)
}

/// Ids in `Tokenizer.extended_id_to_piece` after `__init__` (`tokenizer.py`
/// 78–201). `added_tokens.json` is applied first, then
/// `tokenizer_config.json`'s `added_tokens_decoder` overwrites the same
/// piece. Bos and eos then come from `config.json` (`text_config` included,
/// `model/config.py` 135–141) or, when that id is absent, from
/// `get_default_token_id` (`tokenizer.py` 129–148). A null tokenizer-config
/// value stays unresolved; a missing key falls back to bos 1 / eos 2. Unk is
/// the BPE `unk_token` string (`tokenizer.py` 120–122), not `config.json`.
/// An id is inserted only when its piece is non-empty (189–201).
fn extended_breaker_ids(model_dir: &Path, tokenizer: &tokenizers::Tokenizer) -> Result<Vec<u32>> {
    let mut ordered: Vec<(String, u32)> = Vec::new();
    if let Some(added) = read_optional_json(&model_dir.join("added_tokens.json"))? {
        let obj = added
            .as_object()
            .ok_or_else(|| Error::Weights("added_tokens.json is not an object".into()))?;
        for (piece, value) in obj {
            upsert_piece(
                &mut ordered,
                piece.clone(),
                json_u32(value, "added_tokens.json id")?,
            );
        }
    }
    let tok_cfg = read_optional_json(&model_dir.join("tokenizer_config.json"))?;
    if let Some(decoder) = tok_cfg
        .as_ref()
        .and_then(|value| value.get("added_tokens_decoder"))
        .filter(|value| !value.is_null())
    {
        let obj = decoder
            .as_object()
            .ok_or_else(|| Error::Weights("added_tokens_decoder is not an object".into()))?;
        for (key, spec) in obj {
            let id = key.parse::<u32>().map_err(|_| {
                Error::Weights(format!(
                    "added_tokens_decoder key `{key}` is not a u32 token id"
                ))
            })?;
            let content = spec
                .get("content")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    Error::Weights(format!("added_tokens_decoder[{key}] has no string content"))
                })?;
            upsert_piece(&mut ordered, content.to_string(), id);
        }
    }
    // `{v: k for k, v in piece_to_id.items()}`: later pairs overwrite the piece.
    let mut id_to_piece = std::collections::HashMap::<u32, String>::new();
    for (piece, id) in &ordered {
        id_to_piece.insert(*id, piece.clone());
    }
    let mut ids: Vec<u32> = id_to_piece.keys().copied().collect();
    if let Some(piece) = model_unk_piece(tokenizer) {
        if !piece.is_empty() {
            if let Some(id) = tokenizer.token_to_id(&piece) {
                ids.push(id);
            }
        }
    }
    let model_cfg = read_optional_json(&model_dir.join("config.json"))?;
    let src = ControlSrc {
        tokenizer,
        id_to_piece: &id_to_piece,
        model_cfg: model_cfg.as_ref(),
        tok_cfg: tok_cfg.as_ref(),
    };
    push_resolved_control(
        &mut ids,
        &src,
        "bos_token",
        &["bos_token_id", "text_config->bos_token_id"],
        Some(1),
    )?;
    push_resolved_control(
        &mut ids,
        &src,
        "eos_token",
        &["eos_token_id", "text_config->eos_token_id"],
        Some(2),
    )?;
    Ok(ids)
}

struct ControlSrc<'a> {
    tokenizer: &'a tokenizers::Tokenizer,
    id_to_piece: &'a std::collections::HashMap<u32, String>,
    model_cfg: Option<&'a serde_json::Value>,
    tok_cfg: Option<&'a serde_json::Value>,
}

/// Python dict assignment: an existing piece keeps its slot and updates the id.
fn upsert_piece(ordered: &mut Vec<(String, u32)>, piece: String, id: u32) {
    if let Some(slot) = ordered.iter_mut().find(|(have, _)| *have == piece) {
        slot.1 = id;
    } else {
        ordered.push((piece, id));
    }
}

/// `m.unk_token` for BPE / WordPiece / WordLevel (`tokenizer.py` 120–122).
/// Unigram reads `unk_id` from `tokenizer.json`; this checkpoint is BPE.
fn model_unk_piece(tokenizer: &tokenizers::Tokenizer) -> Option<String> {
    match tokenizer.get_model() {
        tokenizers::ModelWrapper::BPE(model) => model.unk_token.clone(),
        tokenizers::ModelWrapper::WordPiece(model) => Some(model.unk_token.clone()),
        tokenizers::ModelWrapper::WordLevel(model) => Some(model.unk_token.clone()),
        tokenizers::ModelWrapper::Unigram(_) => None,
    }
}

/// `get_default_token_id` plus the bos/eos insert (`tokenizer.py` 129–201).
fn push_resolved_control(
    ids: &mut Vec<u32>,
    src: &ControlSrc<'_>,
    tok_key: &str,
    model_keys: &[&str],
    fallback: Option<u32>,
) -> Result<()> {
    let from_model = src
        .model_cfg
        .and_then(|cfg| first_present(cfg, model_keys))
        .map(|value| config_token_id(value, tok_key))
        .transpose()?
        .flatten();
    // A resolved model id wins. Null / absent falls through to the tokenizer
    // config, and only a missing key there uses `fallback` (`tokenizer.py` 129–148).
    let id = if let Some(id) = from_model {
        Some(id)
    } else {
        match control_id_from_tokenizer_config(src.tok_cfg, src.tokenizer, tok_key)? {
            Some(resolved) => Some(resolved),
            None if tokenizer_key_absent(src.tok_cfg, tok_key) => fallback,
            None => None,
        }
    };
    let Some(id) = id else {
        return Ok(());
    };
    let piece = src
        .id_to_piece
        .get(&id)
        .cloned()
        .or_else(|| src.tokenizer.id_to_token(id));
    if piece.as_deref().is_some_and(|text| !text.is_empty()) {
        ids.push(id);
    }
    Ok(())
}

/// A list uses its first element, as `Config.__init__` does for `eos_token_id`.
/// Null or an empty list leaves the id unresolved.
fn config_token_id(value: &serde_json::Value, what: &str) -> Result<Option<u32>> {
    match value {
        serde_json::Value::Array(items) => match items.first() {
            Some(item) if !item.is_null() => Ok(Some(json_u32(item, what)?)),
            _ => Ok(None),
        },
        serde_json::Value::Null => Ok(None),
        other => Ok(Some(json_u32(other, what)?)),
    }
}

fn tokenizer_key_absent(config: Option<&serde_json::Value>, key: &str) -> bool {
    match config {
        None => true,
        Some(value) => !value.as_object().is_some_and(|map| map.contains_key(key)),
    }
}

fn control_id_from_tokenizer_config(
    config: Option<&serde_json::Value>,
    tokenizer: &tokenizers::Tokenizer,
    key: &str,
) -> Result<Option<u32>> {
    let Some(config) = config else {
        return Ok(None);
    };
    if !config.as_object().is_some_and(|map| map.contains_key(key)) {
        return Ok(None);
    }
    let Some(value) = config.get(key).filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let piece = match value {
        serde_json::Value::String(text) => Some(text.as_str()),
        serde_json::Value::Object(map) => map.get("content").and_then(serde_json::Value::as_str),
        _ => None,
    };
    let Some(piece) = piece else {
        return Ok(None);
    };
    Ok(tokenizer.token_to_id(piece))
}

fn read_optional_json(path: &Path) -> Result<Option<serde_json::Value>> {
    match std::fs::read(path) {
        Ok(bytes) => {
            let value: serde_json::Value = serde_json::from_slice(&bytes)
                .map_err(|err| Error::Weights(format!("{}: {err}", path.display())))?;
            if !value.is_object() {
                return Err(Error::Weights(format!(
                    "{}: expected a JSON object",
                    path.display()
                )));
            }
            Ok(Some(value))
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(Error::Read {
            name: path.display().to_string(),
            source,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicU64, Ordering};

    static FIXTURES: AtomicU64 = AtomicU64::new(0);

    struct Tmp {
        path: std::path::PathBuf,
    }

    impl Tmp {
        fn write(files: &[(&str, &str)]) -> Self {
            let n = FIXTURES.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("berth-stop-{}-{n}", std::process::id()));
            std::fs::create_dir_all(&path).unwrap();
            for (name, body) in files {
                std::fs::write(path.join(name), body).unwrap();
            }
            Self { path }
        }
    }

    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn stops(dir: &Path) -> Vec<u32> {
        read_default_stop_tokens(dir).unwrap()
    }

    #[test]
    fn serve_order_is_config_then_tokenizer_piece_then_generation() {
        let dir = Tmp::write(&[
            ("config.json", r#"{"text_config": {"eos_token_id": 10}}"#),
            (
                "tokenizer_config.json",
                r#"{"eos_token": "<|im_end|>", "added_tokens_decoder": {"20": {"content": "<|im_end|>"}}}"#,
            ),
            ("generation_config.json", r#"{"eos_token_id": [30, 10]}"#),
        ]);
        assert_eq!(stops(&dir.path), vec![10, 20, 30]);
    }

    #[test]
    fn top_level_eos_wins_over_text_config() {
        let dir = Tmp::write(&[(
            "config.json",
            r#"{"eos_token_id": 1, "text_config": {"eos_token_id": 2}}"#,
        )]);
        assert_eq!(stops(&dir.path), vec![1]);
    }

    #[test]
    fn missing_eos_falls_back_to_tokenizer_default_2() {
        let dir = Tmp::write(&[("config.json", "{}")]);
        assert_eq!(stops(&dir.path), vec![2]);
    }

    #[test]
    fn zero_piece_id_is_not_appended_by_the_tokenizer_update() {
        // `if e:` skips 0. The config already supplied 10, so the serve's
        // own eos id is 10 and 0 never enters the list.
        let dir = Tmp::write(&[
            ("config.json", r#"{"eos_token_id": 10}"#),
            (
                "tokenizer_config.json",
                r#"{"eos_token": "<eos>", "added_tokens_decoder": {"0": {"content": "<eos>"}}}"#,
            ),
        ]);
        assert_eq!(stops(&dir.path), vec![10]);
    }

    #[test]
    fn zero_piece_id_is_the_serve_eos_when_config_has_none() {
        let dir = Tmp::write(&[
            ("config.json", "{}"),
            (
                "tokenizer_config.json",
                r#"{"eos_token": "<eos>", "added_tokens_decoder": {"0": {"content": "<eos>"}}}"#,
            ),
        ]);
        assert_eq!(stops(&dir.path), vec![0]);
    }

    #[test]
    fn eos_token_object_uses_its_content_field() {
        let dir = Tmp::write(&[
            ("config.json", r#"{"eos_token_id": 4}"#),
            (
                "tokenizer_config.json",
                r#"{"eos_token": {"content": "</s>"}, "added_tokens_decoder": {"8": {"content": "</s>"}}}"#,
            ),
        ]);
        assert_eq!(stops(&dir.path), vec![4, 8]);
    }

    #[test]
    fn generation_config_integer_is_appended() {
        let dir = Tmp::write(&[
            ("config.json", r#"{"eos_token_id": 4}"#),
            ("generation_config.json", r#"{"eos_token_id": 9}"#),
        ]);
        assert_eq!(stops(&dir.path), vec![4, 9]);
    }

    #[test]
    fn generation_config_float_integer_is_appended() {
        let dir = Tmp::write(&[
            ("config.json", r#"{"eos_token_id": 4}"#),
            ("generation_config.json", r#"{"eos_token_id": 9.0}"#),
        ]);
        assert_eq!(stops(&dir.path), vec![4, 9]);
    }

    #[test]
    fn qwen38_27b_stop_tokens_match_the_serve() {
        let dir = std::env::var_os("BERTH_27B_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from("/home/sam/qwen38-exl3-dflash2/models/qwen38-27b-exl3")
            });
        if !dir.join("generation_config.json").is_file() {
            eprintln!("skipping: {} has no generation_config.json", dir.display());
            return;
        }
        assert_eq!(stops(&dir), vec![248_044, 248_046]);
    }

    #[test]
    fn qwen38_27b_dry_breakers_match_mtp_meta() {
        let dir = std::env::var_os("BERTH_27B_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from("/home/sam/qwen38-exl3-dflash2/models/qwen38-27b-exl3")
            });
        if !dir.join("tokenizer.json").is_file() {
            eprintln!("skipping: {} has no tokenizer.json", dir.display());
            return;
        }
        let got = dry_default_breaker_ids(&dir).unwrap();
        for name in ["meta_N4.json", "meta_N2.json"] {
            let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../testdata/mtp_ref")
                .join(name);
            let bytes = std::fs::read(&path).unwrap();
            let meta: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            let expect: Vec<u32> = meta["samplers"]["sample"]["dry_breaker_token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap() as u32)
                .collect();
            let got_set: std::collections::BTreeSet<u32> = got.iter().copied().collect();
            let expect_set: std::collections::BTreeSet<u32> = expect.iter().copied().collect();
            let missing: Vec<u32> = expect_set.difference(&got_set).copied().take(16).collect();
            let extra: Vec<u32> = got_set.difference(&expect_set).copied().take(16).collect();
            assert!(
                missing.is_empty() && extra.is_empty() && got.len() == expect.len(),
                "{name}: got {} expect {} missing {missing:?} extra {extra:?}",
                got.len(),
                expect.len()
            );
        }
    }
}
