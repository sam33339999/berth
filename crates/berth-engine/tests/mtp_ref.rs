//! MTP generate against `testdata/mtp_ref`. GPU-gated like `generate`.
//!
//! One load per `N` in `{4, 2}` (`BERTH_MTP_REF_N` keeps one). Each meta has
//! eight cases; `BERTH_MTP_REF_CASE=<prompt>_<sampler>` keeps one. The trace
//! hashes each iterate as it arrives and drops the bytes. The first field
//! that differs from the meta panics. Round 0 of the bins is the first
//! verify row, and its max abs diff is printed before that round's fields.
//!
//! ```text
//! BERTH_GPU_TESTS=1 cargo test --release -p berth-engine --test mtp_ref -- --nocapture --test-threads=1
//! ```

mod common;

use std::path::{Path, PathBuf};

use berth_engine::mtp::{
    MtpCq3Snap, MtpDraftSnap, MtpGdnSnap, MtpPageSnap, MtpPrefillSnap, MtpRound, MtpTrace,
    MtpVerifySnap,
};
use berth_engine::{Engine, Finish, Flow, GenerateRequest, SamplerConfig};
use blake2::digest::consts::U32;
use blake2::digest::Digest;
use blake2::Blake2b;
use serde_json::Value;

const MAX_NEW: u32 = 32;

struct Layers<'a> {
    attn: &'a [u32],
    gdn: &'a [u32],
    draft: &'a [u32],
}

#[test]
fn mtp_ref_matches_fork_rounds() {
    if std::env::var("BERTH_GPU_TESTS").as_deref() != Ok("1") {
        eprintln!("skipping mtp_ref: set BERTH_GPU_TESTS=1");
        return;
    }
    let model_dir = common::gpu_ready().unwrap_or_else(|why| panic!("mtp_ref: {why}"));
    let _slot = common::full_model_slot().unwrap_or_else(|why| panic!("mtp_ref: {why}"));
    let cubin = cubin_dir();
    if !cubin.is_dir() {
        panic!("cubin dir {} is missing", cubin.display());
    }

    let only_n = env_n();
    let only_case = env_case();
    let ns = match only_n {
        Some(n) => vec![n],
        None => vec![4, 2],
    };
    let mut ran = 0u32;
    for n in ns {
        let meta = load_meta(n);
        check_sampler_bits(&meta);
        let cache_tokens = u32_of(&meta["cache_tokens"], "cache_tokens");
        let attn_layers = layer_ids(&meta, "attn");
        let gdn_layers = layer_ids(&meta, "gdn");
        let draft_layers = layer_ids(&meta, "draft_attn");
        eprintln!("mtp_ref N={n} loading cache_tokens={cache_tokens}");
        let mut engine = Engine::load(Path::new(&model_dir), &cubin, cache_tokens, n)
            .unwrap_or_else(|err| panic!("N={n} Engine::load: {err}"));
        let cases = meta["cases"]
            .as_array()
            .unwrap_or_else(|| panic!("N={n} cases is not an array"));
        for case in cases {
            let prompt = str_of(&case["prompt"], "prompt").to_string();
            let sampler_name = str_of(&case["sampler"], "sampler").to_string();
            if let Some((want_prompt, want_sampler)) = &only_case {
                if &prompt != want_prompt || &sampler_name != want_sampler {
                    continue;
                }
            }
            ran += 1;
            run_case(
                &mut engine,
                &meta,
                case,
                n,
                &prompt,
                &sampler_name,
                &Layers {
                    attn: &attn_layers,
                    gdn: &gdn_layers,
                    draft: &draft_layers,
                },
            );
            eprintln!("mtp_ref N={n} case={prompt}_{sampler_name} ok");
        }
    }
    if ran == 0 {
        panic!("BERTH_MTP_REF_CASE matched no case in the selected meta");
    }
}

fn run_case(
    engine: &mut Engine,
    meta: &Value,
    case: &Value,
    n: u32,
    prompt: &str,
    sampler_name: &str,
    layers: &Layers<'_>,
) {
    let label = format!("{prompt}_{sampler_name}");
    let ids = prompt_ids(meta, prompt);
    let prompt_len = u32_of(&case["prompt_len"], "prompt_len");
    if ids.len() != usize::try_from(prompt_len).unwrap_or(usize::MAX) {
        panic!(
            "N={n} case={label} prompt_len {prompt_len} but prompts.{prompt} has {} ids",
            ids.len()
        );
    }
    if u32_of(&case["num_draft_tokens"], "num_draft_tokens") != n {
        panic!("N={n} case={label} num_draft_tokens is not {n}");
    }
    let rounds = case["rounds"]
        .as_array()
        .unwrap_or_else(|| panic!("N={n} case={label} rounds is not an array"));
    eprintln!(
        "mtp_ref N={n} case={label} generating {} rounds",
        rounds.len()
    );

    let mut trace = ActiveTrace {
        n,
        case_name: label.clone(),
        rounds: Vec::new(),
    };
    let request = GenerateRequest {
        prompt: ids,
        max_new_tokens: MAX_NEW,
        stop_tokens: Vec::new(),
        sampler: sampler_config(sampler_name),
    };
    let mut tokens = Vec::new();
    // A launch error can land after earlier iterates were already traced.
    // Those rounds are compared first. The error is the failure only when
    // every captured round matches.
    let generated = engine.generate_traced(
        &request,
        &mut |token| {
            tokens.push(token);
            Flow::Continue
        },
        &mut trace,
    );
    if let Err(err) = &generated {
        eprintln!(
            "mtp_ref N={n} case={label} generate stopped after {} rounds: {err}",
            trace.rounds.len()
        );
    }

    for (index, expect) in rounds.iter().enumerate() {
        let got = trace.rounds.get(index);
        let round_i = u32_of(&expect["i"], "round.i");
        let loc = Loc {
            n,
            case_name: label.clone(),
            round: round_i,
        };
        let Some(got) = got else {
            match &generated {
                Err(err) => mismatch(&loc, "generate", None, "round completed", err.to_string()),
                Ok(_) => mismatch(
                    &loc,
                    "rounds",
                    None,
                    format!("{} rounds", rounds.len()),
                    format!("missing round {index}"),
                ),
            }
        };
        if u64::from(round_i) != u64::try_from(index).unwrap_or(u64::MAX) {
            mismatch(
                &loc,
                "rounds.i",
                None,
                index.to_string(),
                round_i.to_string(),
            );
        }
        if verify_index(expect) == Some(0) {
            print_logit_diff(&loc, case, got);
        }
        compare_round(&loc, expect, got, layers);
    }
    if trace.rounds.len() > rounds.len() {
        let loc = Loc {
            n,
            case_name: label.clone(),
            round: u32::try_from(rounds.len()).unwrap_or(u32::MAX),
        };
        mismatch(
            &loc,
            "rounds",
            None,
            format!("{} rounds", rounds.len()),
            format!("{} rounds", trace.rounds.len()),
        );
    }

    let (finish, _usage) = match generated {
        Ok(done) => done,
        Err(err) => {
            let loc = Loc {
                n,
                case_name: label.clone(),
                round: u32::try_from(trace.rounds.len()).unwrap_or(u32::MAX),
            };
            mismatch(&loc, "generate", None, "finished", err.to_string());
        }
    };
    let loc = Loc {
        n,
        case_name: label,
        round: u32::try_from(rounds.len()).unwrap_or(u32::MAX),
    };
    let expect_tokens = u32_list(&case["output_tokens"], "output_tokens");
    if tokens != expect_tokens {
        mismatch(
            &loc,
            "output_tokens",
            None,
            format!("{expect_tokens:?}"),
            format!("{tokens:?}"),
        );
    }
    let expect_eos = str_of(&case["eos_reason"], "eos_reason");
    let actual_eos = eos_reason(finish);
    if actual_eos != expect_eos {
        mismatch(&loc, "eos_reason", None, expect_eos, actual_eos);
    }
}

fn compare_round(loc: &Loc, expect: &Value, got: &DigestedRound, layers: &Layers<'_>) {
    compare_prefill(
        loc,
        &expect["prefill"],
        &expect["draft_prefill"],
        got.prefill.as_ref(),
        layers.draft,
    );
    compare_verify_head(loc, &expect["verify"], got.verify.as_ref());
    compare_draft_ids(loc, &expect["verify"], got.verify.as_ref());
    compare_logit_rows(loc, &expect["verify"], got.verify.as_ref());
    let expect_emitted = u32_list(&expect["emitted"], "emitted");
    if got.emitted != expect_emitted {
        mismatch(
            loc,
            "emitted",
            None,
            format!("{expect_emitted:?}"),
            format!("{:?}", got.emitted),
        );
    }
    compare_gdn(loc, &expect["gdn"], &got.gdn, layers.gdn);
    compare_pages(loc, &expect["target_pages"], &got.target_pages, layers.attn);
    compare_refresh(
        loc,
        &expect["draft_refresh"],
        got.draft_refresh.as_ref(),
        layers.draft,
    );
}

fn compare_prefill(
    loc: &Loc,
    expect: &Value,
    expect_draft: &Value,
    got: Option<&DigestedPrefill>,
    draft_layers: &[u32],
) {
    match (expect, got) {
        (Value::Null, None) => {
            if !expect_draft.is_null() {
                mismatch(loc, "draft_prefill", None, "null", "object");
            }
        }
        (Value::Null, Some(_)) => mismatch(loc, "prefill", None, "null", "object"),
        (_, None) => mismatch(loc, "prefill", None, "object", "null"),
        (_, Some(got)) => {
            same_u32(
                loc,
                "prefill.start",
                None,
                u32_of(&expect["start"], "prefill.start"),
                got.start,
            );
            same_u32(
                loc,
                "prefill.end",
                None,
                u32_of(&expect["end"], "prefill.end"),
                got.end,
            );
            same_str(
                loc,
                "prefill.export",
                None,
                str_of(&expect["export"], "prefill.export"),
                &got.export,
            );
            compare_draft_span(
                loc,
                "draft_prefill",
                expect_draft,
                Some(&got.draft_prefill),
                draft_layers,
            );
        }
    }
}

fn compare_verify_head(loc: &Loc, expect: &Value, got: Option<&DigestedVerify>) {
    match (expect, got) {
        (Value::Null, None) => {}
        (Value::Null, Some(_)) => mismatch(loc, "verify", None, "null", "object"),
        (_, None) => mismatch(loc, "verify", None, "object", "null"),
        (_, Some(got)) => {
            same_u32(
                loc,
                "verify.kv_pos",
                None,
                u32_of(&expect["kv_pos"], "verify.kv_pos"),
                got.kv_pos,
            );
            same_u32(
                loc,
                "verify.q_len",
                None,
                u32_of(&expect["q_len"], "verify.q_len"),
                got.q_len,
            );
            let expect_hist = bool_of(&expect["recurrent_history"], "verify.recurrent_history");
            if expect_hist != got.recurrent_history {
                mismatch(
                    loc,
                    "verify.recurrent_history",
                    None,
                    expect_hist.to_string(),
                    got.recurrent_history.to_string(),
                );
            }
        }
    }
}

fn compare_draft_ids(loc: &Loc, expect_verify: &Value, got: Option<&DigestedVerify>) {
    let expect = if expect_verify.is_null() {
        None
    } else {
        json_ids(&expect_verify["draft_ids"])
    };
    let actual = got.and_then(|verify| verify.draft_ids.clone());
    if expect != actual {
        mismatch(
            loc,
            "draft_ids",
            None,
            format!("{expect:?}"),
            format!("{actual:?}"),
        );
    }
}

fn compare_logit_rows(loc: &Loc, expect_verify: &Value, got: Option<&DigestedVerify>) {
    let expect = if expect_verify.is_null() {
        Vec::new()
    } else {
        str_list(&expect_verify["logit_rows"], "logit_rows")
    };
    let actual = got
        .map(|verify| verify.logit_rows.clone())
        .unwrap_or_default();
    if expect.len() != actual.len() {
        mismatch(
            loc,
            "logit_rows",
            None,
            format!("{} rows", expect.len()),
            format!("{} rows", actual.len()),
        );
    }
    for (index, (want, have)) in expect.iter().zip(actual.iter()).enumerate() {
        if want != have {
            mismatch(
                loc,
                &format!("logit_rows[{index}]"),
                None,
                *want,
                have.as_str(),
            );
        }
    }
}

fn compare_gdn(loc: &Loc, expect: &Value, got: &[DigestedGdn], gdn_layers: &[u32]) {
    let expect = expect.as_array().unwrap_or_else(|| {
        panic!(
            "N={} case={} round={} gdn is not an array",
            loc.n, loc.case_name, loc.round
        )
    });
    if expect.len() != got.len() || expect.len() != gdn_layers.len() {
        mismatch(
            loc,
            "gdn",
            None,
            format!("{} layers", expect.len()),
            format!("{} layers", got.len()),
        );
    }
    for (index, (want, have)) in expect.iter().zip(got.iter()).enumerate() {
        let layer = gdn_layers[index];
        if have.layer != layer {
            mismatch(
                loc,
                "gdn.layer",
                Some(layer),
                layer.to_string(),
                have.layer.to_string(),
            );
        }
        same_str(
            loc,
            "gdn.conv",
            Some(layer),
            str_of(&want["conv"], "gdn.conv"),
            &have.conv,
        );
        same_str(
            loc,
            "gdn.recurrent",
            Some(layer),
            str_of(&want["recurrent"], "gdn.recurrent"),
            &have.recurrent,
        );
    }
}

fn compare_pages(loc: &Loc, expect: &Value, got: &[DigestedPage], attn_layers: &[u32]) {
    let expect = expect.as_array().unwrap_or_else(|| {
        panic!(
            "N={} case={} round={} target_pages is not an array",
            loc.n, loc.case_name, loc.round
        )
    });
    let want_logical: Vec<u32> = expect
        .iter()
        .map(|page| u32_of(&page["logical"], "target_pages.logical"))
        .collect();
    let have_logical: Vec<u32> = got.iter().map(|page| page.logical).collect();
    if want_logical != have_logical {
        mismatch(
            loc,
            "target_pages.logical",
            None,
            format!("{want_logical:?}"),
            format!("{have_logical:?}"),
        );
    }
    for (want, have) in expect.iter().zip(got.iter()) {
        compare_cq3_arrays(
            loc,
            &format!("target_pages[{}]", have.logical),
            want,
            &have.layers,
            attn_layers,
        );
    }
}

fn compare_refresh(loc: &Loc, expect: &Value, got: Option<&DigestedDraft>, draft_layers: &[u32]) {
    compare_draft_span(loc, "draft_refresh", expect, got, draft_layers);
}

fn compare_draft_span(
    loc: &Loc,
    field: &str,
    expect: &Value,
    got: Option<&DigestedDraft>,
    draft_layers: &[u32],
) {
    match (expect, got) {
        (Value::Null, None) => {}
        (Value::Null, Some(_)) => mismatch(loc, field, None, "null", "object"),
        (_, None) => mismatch(loc, field, None, "object", "null"),
        (_, Some(got)) => {
            same_u32(
                loc,
                &format!("{field}.start"),
                None,
                u32_of(&expect["start"], field),
                got.start,
            );
            same_u32(
                loc,
                &format!("{field}.length"),
                None,
                u32_of(&expect["length"], field),
                got.length,
            );
            compare_cq3_arrays(loc, field, expect, &got.layers, draft_layers);
        }
    }
}

fn compare_cq3_arrays(
    loc: &Loc,
    field: &str,
    expect: &Value,
    got: &[DigestedCq3],
    layer_ids: &[u32],
) {
    let qk = str_list(&expect["qk"], field);
    let qv = str_list(&expect["qv"], field);
    let sk = str_list(&expect["sk"], field);
    let sv = str_list(&expect["sv"], field);
    if qk.len() != got.len() || qk.len() != layer_ids.len() {
        mismatch(
            loc,
            field,
            None,
            format!("{} layers", qk.len()),
            format!("{} layers", got.len()),
        );
    }
    for (index, layer) in layer_ids.iter().copied().enumerate() {
        if got[index].layer != layer {
            mismatch(
                loc,
                &format!("{field}.layer"),
                Some(layer),
                layer.to_string(),
                got[index].layer.to_string(),
            );
        }
        same_str(
            loc,
            &format!("{field}.qk"),
            Some(layer),
            qk[index],
            &got[index].qk,
        );
        same_str(
            loc,
            &format!("{field}.qv"),
            Some(layer),
            qv[index],
            &got[index].qv,
        );
        same_str(
            loc,
            &format!("{field}.sk"),
            Some(layer),
            sk[index],
            &got[index].sk,
        );
        same_str(
            loc,
            &format!("{field}.sv"),
            Some(layer),
            sv[index],
            &got[index].sv,
        );
    }
}

fn print_logit_diff(loc: &Loc, case: &Value, got: &DigestedRound) {
    let name = str_of(&case["logits_bin"], "logits_bin");
    let path = ref_dir().join(name);
    let bin = std::fs::read(&path).unwrap_or_else(|err| {
        panic!(
            "{} ({err}). 先跑 `scripts/dump_27b_mtp_reference.py`",
            path.display()
        );
    });
    let row = got
        .verify
        .as_ref()
        .and_then(|verify| verify.logit_row0.as_deref());
    let Some(row) = row else {
        eprintln!(
            "N={} case={} round={} verify index 0 logit row 0 max abs diff: no row (bin {} bytes)",
            loc.n,
            loc.case_name,
            loc.round,
            bin.len()
        );
        return;
    };
    let n = row.len().min(bin.len()) / 2;
    let mut max = 0.0f32;
    for index in 0..n {
        let off = index * 2;
        let got_v = f16_to_f32(u16::from_le_bytes([row[off], row[off + 1]]));
        let exp_v = f16_to_f32(u16::from_le_bytes([bin[off], bin[off + 1]]));
        max = max.max(abs_diff(got_v, exp_v));
    }
    eprintln!(
        "N={} case={} round={} verify index 0 logit row 0 max abs diff {max} (row {} bytes, bin {} bytes, {name})",
        loc.n,
        loc.case_name,
        loc.round,
        row.len(),
        bin.len()
    );
}

/// IEEE binary16 to f32. No `half` crate.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from((bits >> 10) & 0x1f);
    let frac = u32::from(bits & 0x3ff);
    let bits32 = match exp {
        0 if frac == 0 => sign,
        0 => {
            let mut frac = frac;
            let mut shift = 0u32;
            while frac & 0x400 == 0 {
                frac <<= 1;
                shift += 1;
            }
            frac &= 0x3ff;
            // Unbiased exponent of a subnormal is -14 - shift. f32 bias is 127.
            sign | ((113 - shift) << 23) | (frac << 13)
        }
        31 => sign | 0x7f80_0000 | (frac << 13),
        _ => sign | ((exp + (127 - 15)) << 23) | (frac << 13),
    };
    f32::from_bits(bits32)
}

fn abs_diff(left: f32, right: f32) -> f32 {
    if left.to_bits() == right.to_bits() {
        return 0.0;
    }
    if left.is_nan() || right.is_nan() {
        return f32::INFINITY;
    }
    (left - right).abs()
}

fn verify_index(round: &Value) -> Option<u64> {
    round["verify"]["index"].as_u64()
}

struct Loc {
    n: u32,
    case_name: String,
    round: u32,
}

fn mismatch(
    loc: &Loc,
    field: &str,
    layer: Option<u32>,
    expected: impl AsRef<str>,
    actual: impl AsRef<str>,
) -> ! {
    let layer = match layer {
        Some(layer) => layer.to_string(),
        None => "-".to_string(),
    };
    panic!(
        "N={} case={} round={} field={field} layer={layer}: expected {} actual {}",
        loc.n,
        loc.case_name,
        loc.round,
        expected.as_ref(),
        actual.as_ref()
    );
}

fn same_u32(loc: &Loc, field: &str, layer: Option<u32>, expected: u32, actual: u32) {
    if expected != actual {
        mismatch(loc, field, layer, expected.to_string(), actual.to_string());
    }
}

fn same_str(loc: &Loc, field: &str, layer: Option<u32>, expected: &str, actual: &str) {
    if expected != actual {
        mismatch(loc, field, layer, expected, actual);
    }
}

struct ActiveTrace {
    n: u32,
    case_name: String,
    rounds: Vec<DigestedRound>,
}

impl MtpTrace for ActiveTrace {
    fn active(&self) -> bool {
        true
    }

    fn on_round(&mut self, round: &MtpRound) {
        let digested = digest_round(round);
        let prefill = match &digested.prefill {
            Some(prefill) => format!("{}..{}", prefill.start, prefill.end),
            None => "null".to_string(),
        };
        let q_len = match &digested.verify {
            Some(verify) => verify.q_len.to_string(),
            None => "null".to_string(),
        };
        eprintln!(
            "mtp_ref N={} case={} captured round {} prefill={prefill} q_len={q_len} emitted={}",
            self.n,
            self.case_name,
            self.rounds.len(),
            digested.emitted.len()
        );
        self.rounds.push(digested);
    }
}

struct DigestedRound {
    prefill: Option<DigestedPrefill>,
    verify: Option<DigestedVerify>,
    emitted: Vec<u32>,
    gdn: Vec<DigestedGdn>,
    target_pages: Vec<DigestedPage>,
    draft_refresh: Option<DigestedDraft>,
}

struct DigestedPrefill {
    start: u32,
    end: u32,
    export: String,
    draft_prefill: DigestedDraft,
}

struct DigestedVerify {
    kv_pos: u32,
    q_len: u32,
    recurrent_history: bool,
    draft_ids: Option<Vec<u32>>,
    logit_rows: Vec<String>,
    /// Raw fp16 row 0, for the first-verify max-abs print.
    logit_row0: Option<Vec<u8>>,
}

struct DigestedGdn {
    layer: u32,
    conv: String,
    recurrent: String,
}

struct DigestedPage {
    logical: u32,
    layers: Vec<DigestedCq3>,
}

struct DigestedDraft {
    start: u32,
    length: u32,
    layers: Vec<DigestedCq3>,
}

struct DigestedCq3 {
    layer: u32,
    qk: String,
    qv: String,
    sk: String,
    sv: String,
}

fn digest_round(round: &MtpRound) -> DigestedRound {
    DigestedRound {
        prefill: round.prefill.as_ref().map(digest_prefill),
        verify: round.verify.as_ref().map(digest_verify),
        emitted: round.emitted.clone(),
        gdn: round.gdn.iter().map(digest_gdn).collect(),
        target_pages: round.target_pages.iter().map(digest_page).collect(),
        draft_refresh: round.draft_refresh.as_ref().map(digest_draft),
    }
}

fn digest_prefill(prefill: &MtpPrefillSnap) -> DigestedPrefill {
    DigestedPrefill {
        start: prefill.start,
        end: prefill.end,
        export: blake2b_256(&prefill.export),
        draft_prefill: digest_draft(&prefill.draft_prefill),
    }
}

fn digest_verify(verify: &MtpVerifySnap) -> DigestedVerify {
    DigestedVerify {
        kv_pos: verify.kv_pos,
        q_len: verify.q_len,
        recurrent_history: verify.recurrent_history,
        draft_ids: verify.draft_ids.clone(),
        logit_rows: verify
            .logit_rows
            .iter()
            .map(|row| blake2b_256(row))
            .collect(),
        logit_row0: verify.logit_rows.first().cloned(),
    }
}

fn digest_gdn(gdn: &MtpGdnSnap) -> DigestedGdn {
    DigestedGdn {
        layer: gdn.layer,
        conv: blake2b_256(&gdn.conv),
        recurrent: blake2b_256(&gdn.recurrent),
    }
}

fn digest_page(page: &MtpPageSnap) -> DigestedPage {
    DigestedPage {
        logical: page.logical,
        layers: page.layers.iter().map(digest_cq3).collect(),
    }
}

fn digest_draft(draft: &MtpDraftSnap) -> DigestedDraft {
    DigestedDraft {
        start: draft.start,
        length: draft.length,
        layers: draft.layers.iter().map(digest_cq3).collect(),
    }
}

fn digest_cq3(snap: &MtpCq3Snap) -> DigestedCq3 {
    DigestedCq3 {
        layer: snap.layer,
        qk: blake2b_256(&snap.qk),
        qv: blake2b_256(&snap.qv),
        sk: blake2b_256(&snap.sk),
        sv: blake2b_256(&snap.sv),
    }
}

fn blake2b_256(bytes: &[u8]) -> String {
    let mut hasher = Blake2b::<U32>::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        out.push(char::from(HEX[usize::from(byte >> 4)]));
        out.push(char::from(HEX[usize::from(byte & 0xf)]));
    }
    out
}

fn sampler_config(name: &str) -> SamplerConfig {
    match name {
        "greedy" => SamplerConfig::greedy(),
        "sample" => SamplerConfig::serve_defaults(0.7, 42),
        other => panic!("sampler {other} is not greedy or sample"),
    }
}

/// The live sampler bits the dump recorded. A mismatch here is the test
/// setup, so it fails before a generate.
fn check_sampler_bits(meta: &Value) {
    let greedy = &meta["samplers"]["greedy"]["collapsed_steps"][0];
    same_hex(
        "greedy inv_temp",
        "3f800000",
        str_of(&greedy["inv_temp_f32_bits"], "greedy inv_temp"),
    );
    let greedy_cfg = SamplerConfig::greedy();
    assert_eq!(greedy_cfg.temperature.to_bits(), 0, "greedy temperature");
    assert_eq!(
        greedy_cfg.dry_multiplier.to_bits(),
        0,
        "greedy dry_multiplier"
    );
    assert_eq!(greedy_cfg.seed, 0, "greedy seed");
    assert_eq!(
        u32_of(&meta["samplers"]["greedy"]["seed"], "greedy seed"),
        0
    );

    let sample = &meta["samplers"]["sample"];
    let cfg = SamplerConfig::serve_defaults(0.7, 42);
    let inv = (1.0f64 / f64::from(0.7f32)) as f32;
    let steps = sample["collapsed_steps"]
        .as_array()
        .unwrap_or_else(|| panic!("sample collapsed_steps is not an array"));
    let dry = steps
        .iter()
        .find(|step| step["type"] == "SS_DRY")
        .unwrap_or_else(|| panic!("sample stack has no SS_DRY"));
    let fused = steps
        .iter()
        .find(|step| step["type"] == "SS_Fused")
        .unwrap_or_else(|| panic!("sample stack has no SS_Fused"));
    same_hex(
        "dry_multiplier",
        str_of(&dry["dry_multiplier_f32_bits"], "dry_multiplier"),
        &hex_f32(cfg.dry_multiplier),
    );
    same_hex(
        "dry_base",
        str_of(&dry["dry_base_f32_bits"], "dry_base"),
        &hex_f32(cfg.dry_base),
    );
    same_hex(
        "sample inv_temp",
        str_of(&fused["inv_temp_f32_bits"], "sample inv_temp"),
        &hex_f32(inv),
    );
    assert_eq!(
        u32_of(&dry["dry_allowed_length"], "dry_allowed_length"),
        cfg.dry_allowed_length
    );
    assert_eq!(u32_of(&dry["dry_range"], "dry_range"), cfg.dry_range);
    assert_eq!(cfg.rep_penalty.to_bits(), 1.0f32.to_bits());
    assert_eq!(cfg.presence_penalty.to_bits(), 0);
    assert_eq!(cfg.frequency_penalty.to_bits(), 0);
    assert_eq!(cfg.seed, 42);
    assert_eq!(u32_of(&sample["seed"], "sample seed"), 42);
}

fn hex_f32(value: f32) -> String {
    format!("{:08x}", value.to_bits())
}

fn same_hex(what: &str, expected: &str, actual: &str) {
    assert_eq!(expected, actual, "{what} f32 bits");
}

fn eos_reason(finish: Finish) -> &'static str {
    match finish {
        Finish::Length => "max_new_tokens",
        Finish::Stop => "stop",
        Finish::Cancelled => "cancelled",
        Finish::ContextFull => "context_full",
    }
}

fn load_meta(n: u32) -> Value {
    let path = ref_dir().join(format!("meta_N{n}.json"));
    let bytes = std::fs::read(&path).unwrap_or_else(|_| {
        panic!(
            "{} 不在。先跑 `scripts/dump_27b_mtp_reference.py`",
            path.display()
        );
    });
    let meta: Value =
        serde_json::from_slice(&bytes).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    if meta["schema_version"] != 1 {
        panic!("{} schema_version is not 1", path.display());
    }
    if meta["num_draft_tokens"] != n {
        panic!("{} num_draft_tokens is not {n}", path.display());
    }
    meta
}

fn prompt_ids(meta: &Value, name: &str) -> Vec<u32> {
    let ids = meta["prompts"][name]
        .as_array()
        .unwrap_or_else(|| panic!("prompts.{name} is not an array"));
    ids.iter().map(|id| u32_of(id, "prompt id")).collect()
}

fn layer_ids(meta: &Value, key: &str) -> Vec<u32> {
    let layers = meta["layers"][key]
        .as_array()
        .unwrap_or_else(|| panic!("layers.{key} is not an array"));
    layers
        .iter()
        .map(|pair| u32_of(&pair[0], "layer id"))
        .collect()
}

fn u32_list(value: &Value, what: &str) -> Vec<u32> {
    let items = value
        .as_array()
        .unwrap_or_else(|| panic!("{what} is not an array"));
    items.iter().map(|item| u32_of(item, what)).collect()
}

fn str_list<'a>(value: &'a Value, what: &str) -> Vec<&'a str> {
    let items = value
        .as_array()
        .unwrap_or_else(|| panic!("{what} is not an array"));
    items.iter().map(|item| str_of(item, what)).collect()
}

fn json_ids(value: &Value) -> Option<Vec<u32>> {
    if value.is_null() {
        None
    } else {
        Some(u32_list(value, "draft_ids"))
    }
}

fn u32_of(value: &Value, what: &str) -> u32 {
    let n = value
        .as_u64()
        .unwrap_or_else(|| panic!("{what} is not an integer"));
    u32::try_from(n).unwrap_or_else(|_| panic!("{what} does not fit in u32"))
}

fn str_of<'a>(value: &'a Value, what: &str) -> &'a str {
    value
        .as_str()
        .unwrap_or_else(|| panic!("{what} is not a string"))
}

fn bool_of(value: &Value, what: &str) -> bool {
    value
        .as_bool()
        .unwrap_or_else(|| panic!("{what} is not a bool"))
}

fn env_n() -> Option<u32> {
    let Ok(raw) = std::env::var("BERTH_MTP_REF_N") else {
        return None;
    };
    let n: u32 = raw
        .parse()
        .unwrap_or_else(|_| panic!("BERTH_MTP_REF_N must be an integer, got {raw}"));
    if n != 2 && n != 4 {
        panic!("BERTH_MTP_REF_N must be 2 or 4, got {n}");
    }
    Some(n)
}

fn env_case() -> Option<(String, String)> {
    let Ok(raw) = std::env::var("BERTH_MTP_REF_CASE") else {
        return None;
    };
    let (prompt, sampler) = raw.rsplit_once('_').unwrap_or_else(|| {
        panic!("BERTH_MTP_REF_CASE must be <prompt>_<sampler>, got {raw}");
    });
    if sampler != "greedy" && sampler != "sample" {
        panic!("BERTH_MTP_REF_CASE sampler must be greedy or sample, got {raw}");
    }
    Some((prompt.to_string(), sampler.to_string()))
}

fn ref_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/mtp_ref")
}

fn cubin_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/triton-cubins")
}
