//! Spec checks for `docs/repetition.md`.

use berth_repetition::{
    format_completion_field, format_mid_stream_event, parse_exact_min_tokens, parse_tail_lookback,
    Channel, CheckConfig, NoTail, Outcome, RepetitionCheck, Sampled, TailDetector, TriggerSet,
    EXACT_REPEAT_CHECK_HEADER, MAX_LABEL_BYTES, MAX_TRIGGERS, MAX_WINDOW,
    PROPOSED_EXACT_MIN_TOKENS, PROPOSED_TAIL_LOOKBACK, TAIL_CHECK_HEADER, TAIL_LOOKBACK_MAX,
    TAIL_LOOKBACK_MIN,
};

fn accept(token: u32) -> Sampled {
    Sampled::Accepted(token)
}

fn reject(token: u32) -> Sampled {
    Sampled::Rejected(token)
}

fn feed(check: &mut RepetitionCheck<impl TailDetector>, channel: Channel, tokens: &[u32]) {
    for token in tokens {
        let _ = check.observe(channel, accept(*token));
    }
}

/// Records every window it is shown and returns a fixed set of `t` values.
struct ScriptedTail {
    answer: Vec<u32>,
    calls: Vec<(Channel, Vec<u32>)>,
}

impl TailDetector for ScriptedTail {
    fn detect(&mut self, channel: Channel, window: &[u32]) -> Vec<u32> {
        self.calls.push((channel, window.to_vec()));
        self.answer.clone()
    }
}

/// Returns one fixed set of `t` values and does not store the windows.
struct FixedTail {
    answer: Vec<u32>,
}

impl TailDetector for FixedTail {
    fn detect(&mut self, _channel: Channel, _window: &[u32]) -> Vec<u32> {
        self.answer.clone()
    }
}

struct PanicTail;

impl TailDetector for PanicTail {
    fn detect(&mut self, _channel: Channel, _window: &[u32]) -> Vec<u32> {
        panic!("tail detector runs only when x-berth-tail-check is enabled");
    }
}

fn block(len: u32) -> Vec<u32> {
    (0..len).collect()
}

#[test]
fn channels_do_not_share_a_buffer() {
    let mut check = RepetitionCheck::new(
        CheckConfig::new(None, Some(PROPOSED_EXACT_MIN_TOKENS)),
        PanicTail,
    );
    let block = block(PROPOSED_EXACT_MIN_TOKENS);
    feed(&mut check, Channel::Thinking, &block);
    feed(&mut check, Channel::Response, &block);
    assert!(
        check.triggers().is_empty(),
        "one copy on each channel is not a run"
    );
    assert!(check.completion_field().is_none());
    assert!(check.mid_stream_event().is_none());

    feed(&mut check, Channel::Thinking, &block);
    assert_eq!(
        check.triggers(),
        ["exact_repetition:64x2@thinking".to_string()]
    );
    assert!(check
        .triggers()
        .iter()
        .all(|label| !label.contains("@response")));
}

#[test]
fn rejected_draft_tokens_are_not_stored_and_do_not_split_a_run() {
    let mut check = RepetitionCheck::new(CheckConfig::new(None, Some(4)), PanicTail);
    let block = block(4);

    feed(&mut check, Channel::Response, &block);
    let skipped = check.observe(Channel::Response, reject(999));
    assert_eq!(
        skipped,
        Outcome {
            new_trigger: false,
            mid_stream_event: None
        }
    );
    assert_eq!(check.accepted_len(Channel::Response), 4);
    feed(&mut check, Channel::Response, &block);

    assert_eq!(
        check.triggers(),
        ["exact_repetition:4x2@response".to_string()],
        "the rejected id is dropped, so the two blocks sit next to each other"
    );

    let mut broken = RepetitionCheck::new(CheckConfig::new(None, Some(4)), PanicTail);
    feed(&mut broken, Channel::Response, &block);
    feed(&mut broken, Channel::Response, &[999]);
    feed(&mut broken, Channel::Response, &block);
    assert!(
        broken.triggers().is_empty(),
        "an accepted token between copies is not consecutive"
    );

    // The token that would have completed the second copy was rejected.
    let mut short = RepetitionCheck::new(CheckConfig::new(None, Some(4)), PanicTail);
    feed(&mut short, Channel::Response, &block);
    feed(&mut short, Channel::Response, &[0, 1, 2]);
    let outcome = short.observe(Channel::Response, reject(3));
    assert!(!outcome.new_trigger);
    assert!(short.triggers().is_empty());
    assert_eq!(short.accepted_len(Channel::Response), 7);
}

#[test]
fn exact_repetition_doc_example_64x3_thinking() {
    let mut check = RepetitionCheck::new(
        CheckConfig::new(None, Some(PROPOSED_EXACT_MIN_TOKENS)),
        PanicTail,
    );
    let block = block(64);
    feed(&mut check, Channel::Thinking, &block);
    assert!(check.triggers().is_empty());

    let mut saw_two = false;
    let mut second_event = None;
    for token in block.iter() {
        let outcome = check.observe(Channel::Thinking, accept(*token));
        if outcome.new_trigger {
            saw_two = true;
            second_event = outcome.mid_stream_event;
            assert_eq!(
                check.triggers(),
                ["exact_repetition:64x2@thinking".to_string()]
            );
        }
    }
    assert!(saw_two, "the second copy is reported when it completes");
    assert_eq!(
        second_event.as_deref(),
        Some(
            "event: response.repetition_check\n\
             data: {\"type\":\"response.repetition_check\",\"repetition_check\":{\"triggers\":[\"exact_repetition:64x2@thinking\"]}}\n"
        )
    );

    let mut third = Outcome {
        new_trigger: false,
        mid_stream_event: None,
    };
    for (index, token) in block.iter().enumerate() {
        let outcome = check.observe(Channel::Thinking, accept(*token));
        if index + 1 < block.len() {
            assert!(!outcome.new_trigger, "the same count is not sent again");
        }
        third = outcome;
    }
    assert!(third.new_trigger, "a higher count is sent again");
    assert_eq!(
        check.triggers(),
        ["exact_repetition:64x3@thinking".to_string()]
    );
    assert_eq!(
        third.mid_stream_event.as_deref(),
        Some(
            "event: response.repetition_check\n\
             data: {\"type\":\"response.repetition_check\",\"repetition_check\":{\"triggers\":[\"exact_repetition:64x3@thinking\"]}}\n"
        )
    );
    assert_eq!(
        check.completion_field().as_deref(),
        Some("\"repetition_check\": {\"triggers\": [\"exact_repetition:64x3@thinking\"]}")
    );
}

#[test]
fn non_consecutive_repeat_does_not_fire() {
    let mut check = RepetitionCheck::new(CheckConfig::new(None, Some(3)), PanicTail);
    // Same block twice, with one different token between. The proposal does not allow that.
    feed(&mut check, Channel::Response, &[1, 2, 3, 9, 1, 2, 3]);
    assert!(check.triggers().is_empty());

    // The same block twice, far apart, still not consecutive.
    feed(&mut check, Channel::Thinking, &[4, 5, 6, 7, 8, 4, 5, 6]);
    assert!(check.triggers().is_empty());

    // A real adjacent pair still fires, and it does not invent low_logprob.
    feed(&mut check, Channel::Response, &[1, 2, 3]);
    assert_eq!(
        check.triggers(),
        ["exact_repetition:3x2@response".to_string()]
    );
    assert!(check
        .triggers()
        .iter()
        .all(|label| { !label.starts_with("low_logprob") && !label.starts_with("low_diversity") }));
}

#[test]
fn identical_tokens_do_not_emit_low_logprob() {
    let mut check = RepetitionCheck::new(CheckConfig::new(Some(512), None), NoTail);
    feed(&mut check, Channel::Thinking, &[7, 7, 7, 7, 7, 7, 7, 7]);
    assert!(
        check.triggers().is_empty(),
        "tail formula is not shipped; NoTail reports nothing, and low_logprob is not a v1 label"
    );
}

#[test]
fn tail_lookback_clamp_table() {
    let cases = [
        ("512", Some(512)),
        ("1024", Some(1024)),
        ("4096", Some(4096)),
        ("0512", Some(512)),
        (" 2048 ", Some(2048)),
        ("2048\n", Some(2048)),
        ("0", Some(TAIL_LOOKBACK_MAX)),
        ("511", Some(TAIL_LOOKBACK_MAX)),
        ("4097", Some(TAIL_LOOKBACK_MAX)),
        ("100000", Some(TAIL_LOOKBACK_MAX)),
        ("000", Some(TAIL_LOOKBACK_MAX)),
        (&"9".repeat(40), Some(TAIL_LOOKBACK_MAX)),
        ("", None),
        ("   ", None),
        ("abc", None),
        ("1024.0", None),
        ("-1", None),
        ("+512", None),
        ("512 tokens", None),
        ("5_12", None),
        ("0x200", None),
        ("１024", None),
    ];
    for (value, expect) in cases {
        assert_eq!(parse_tail_lookback(value), expect, "tail value {value:?}");
    }

    assert_eq!(TAIL_LOOKBACK_MIN, 512);
    assert_eq!(TAIL_LOOKBACK_MAX, 4096);
    assert_eq!(PROPOSED_TAIL_LOOKBACK, 1024);
    assert_eq!(MAX_WINDOW, 4096);

    let missing = CheckConfig::parse_headers([("content-type", "application/json")]);
    assert!(!missing.tail_enabled());
    assert_eq!(
        missing.tail_lookback(),
        None,
        "a missing header is off, not 1024"
    );

    let header = CheckConfig::parse_headers([(TAIL_CHECK_HEADER, "511")]);
    assert_eq!(header.tail_lookback(), Some(4096));

    let cased = CheckConfig::parse_headers([("X-Berth-Tail-Check", "1024")]);
    assert_eq!(cased.tail_lookback(), Some(1024));

    let first_wins =
        CheckConfig::parse_headers([(TAIL_CHECK_HEADER, "1024"), (TAIL_CHECK_HEADER, "2048")]);
    assert_eq!(first_wins.tail_lookback(), Some(1024));

    let malformed_first =
        CheckConfig::parse_headers([(TAIL_CHECK_HEADER, "nope"), (TAIL_CHECK_HEADER, "1024")]);
    assert_eq!(malformed_first.tail_lookback(), None);
}

#[test]
fn exact_minimum_malformed_is_off_and_missing_is_off() {
    assert_eq!(parse_exact_min_tokens("64"), Some(64));
    assert_eq!(parse_exact_min_tokens(" 1 "), Some(1));
    assert_eq!(parse_exact_min_tokens("4097"), Some(4097));
    assert_eq!(parse_exact_min_tokens("00064"), Some(64));
    assert_eq!(
        parse_exact_min_tokens("0"),
        None,
        "zero is not a block length"
    );
    assert_eq!(parse_exact_min_tokens("00"), None);
    assert_eq!(parse_exact_min_tokens(""), None);
    assert_eq!(parse_exact_min_tokens("-64"), None);
    assert_eq!(parse_exact_min_tokens("+64"), None);
    assert_eq!(parse_exact_min_tokens("64.0"), None);
    assert_eq!(parse_exact_min_tokens(&"9".repeat(40)), None);
    assert_eq!(PROPOSED_EXACT_MIN_TOKENS, 64);

    let missing = CheckConfig::parse_headers([(TAIL_CHECK_HEADER, "1024")]);
    assert!(missing.exact_min_tokens().is_none());
    assert_eq!(missing.tail_lookback(), Some(1024));

    let both = CheckConfig::parse_headers([
        ("X-BERTH-EXACT-REPEAT-CHECK", "32"),
        (TAIL_CHECK_HEADER, "4097"),
    ]);
    assert_eq!(both.exact_min_tokens(), Some(32));
    assert_eq!(both.tail_lookback(), Some(4096));

    let constructed = CheckConfig::new(Some(100), Some(0));
    assert_eq!(constructed.tail_lookback(), Some(4096));
    assert_eq!(constructed.exact_min_tokens(), None);
}

#[test]
fn lookback_counts_tokens_of_that_channel_only() {
    let mut check = RepetitionCheck::new(
        CheckConfig::new(Some(TAIL_LOOKBACK_MIN), None),
        ScriptedTail {
            answer: vec![],
            calls: vec![],
        },
    );
    for token in 0..600u32 {
        let _ = check.observe(Channel::Response, accept(token));
    }
    for token in 0..10u32 {
        let _ = check.observe(Channel::Thinking, accept(token));
    }

    let calls = &check.detector_mut().calls;
    let response: Vec<_> = calls
        .iter()
        .filter(|(channel, _)| *channel == Channel::Response)
        .collect();
    let thinking: Vec<_> = calls
        .iter()
        .filter(|(channel, _)| *channel == Channel::Thinking)
        .collect();
    assert_eq!(response.len(), 600);
    assert_eq!(thinking.len(), 10);

    let (_, last_response) = response.last().unwrap();
    assert_eq!(last_response.len(), 512);
    assert_eq!(last_response.first().copied(), Some(600 - 512));
    assert_eq!(last_response.last().copied(), Some(599));

    let (_, last_thinking) = thinking.last().unwrap();
    assert_eq!(last_thinking, &(0..10).collect::<Vec<_>>());
    assert_eq!(check.accepted_len(Channel::Response), MAX_WINDOW.min(600));
    assert_eq!(check.accepted_len(Channel::Thinking), 10);
}

#[test]
fn buffer_drops_accepted_tokens_past_the_window() {
    let mut check = RepetitionCheck::new(CheckConfig::new(None, Some(3_000)), PanicTail);
    for token in 0..5_000u32 {
        let _ = check.observe(Channel::Response, accept(token));
    }
    assert_eq!(check.accepted_len(Channel::Response), MAX_WINDOW);
    assert!(
        check.triggers().is_empty(),
        "two copies of a 3000-token block do not fit in 4096"
    );

    let mut off = RepetitionCheck::new(CheckConfig::disabled(), PanicTail);
    let _ = off.observe(Channel::Response, accept(1));
    let _ = off.observe(Channel::Response, reject(2));
    assert_eq!(off.accepted_len(Channel::Response), 0);
    assert!(off.triggers().is_empty());
}

#[test]
fn label_byte_cap_and_count_cap() {
    let mut set = TriggerSet::new();
    let oversized = "x".repeat(MAX_LABEL_BYTES + 1);
    assert!(!set.insert(&oversized));
    assert!(set.is_empty(), "an oversized label does not take a slot");

    let maxed = "y".repeat(MAX_LABEL_BYTES);
    assert_eq!(maxed.len(), MAX_LABEL_BYTES);
    assert!(set.insert(&maxed));

    let wide = "字".repeat(100);
    assert!(wide.len() > MAX_LABEL_BYTES);
    assert!(!set.insert(&wide));

    assert!(!set.insert(&maxed), "duplicate");
    assert_eq!(set.len(), 1);

    for index in 1..MAX_TRIGGERS {
        assert!(set.insert(&format!("label-{index}")));
    }
    assert_eq!(set.len(), MAX_TRIGGERS);
    assert!(!set.insert("one-past-cap"));
    assert_eq!(set.as_slice().last().map(String::as_str), Some("label-63"));
    assert_eq!(
        set.as_slice().first().map(String::as_str),
        Some(maxed.as_str())
    );

    let mut check = RepetitionCheck::new(
        CheckConfig::new(Some(512), None),
        ScriptedTail {
            answer: (0..80).collect(),
            calls: vec![],
        },
    );
    let outcome = check.observe(Channel::Thinking, accept(1));
    assert!(outcome.new_trigger);
    assert_eq!(check.triggers().len(), MAX_TRIGGERS);
    assert_eq!(check.triggers()[0], "tail_repetition:0@thinking");
    assert_eq!(check.triggers()[63], "tail_repetition:63@thinking");

    let again = check.observe(Channel::Thinking, accept(2));
    assert!(
        !again.new_trigger,
        "labels past the cap are not a new trigger"
    );
    assert!(again.mid_stream_event.is_none());
    assert_eq!(check.triggers().len(), MAX_TRIGGERS);
}

#[test]
fn triggers_accumulate_and_reemit_the_full_set() {
    let mut check = RepetitionCheck::new(
        CheckConfig::new(Some(1024), None),
        ScriptedTail {
            answer: vec![4],
            calls: vec![],
        },
    );
    let first = check.observe(Channel::Thinking, accept(1));
    assert!(first.new_trigger);
    let spec_sse = "\
event: response.repetition_check
data: {\"type\":\"response.repetition_check\",\"repetition_check\":{\"triggers\":[\"tail_repetition:4@thinking\"]}}
";
    assert_eq!(first.mid_stream_event.as_deref(), Some(spec_sse));
    assert_eq!(check.mid_stream_event().as_deref(), Some(spec_sse));

    let repeat = check.observe(Channel::Thinking, accept(2));
    assert!(!repeat.new_trigger);
    assert!(repeat.mid_stream_event.is_none());
    assert_eq!(check.triggers(), ["tail_repetition:4@thinking".to_string()]);

    check.detector_mut().answer = vec![4, 4, 2];
    let second = check.observe(Channel::Thinking, accept(3));
    assert!(second.new_trigger);
    let event = second.mid_stream_event.expect("new t");
    assert!(event.starts_with("event: response.repetition_check\ndata: "));
    assert!(event.contains("[\"tail_repetition:4@thinking\",\"tail_repetition:2@thinking\"]"));
    assert_eq!(
        check.triggers(),
        [
            "tail_repetition:4@thinking".to_string(),
            "tail_repetition:2@thinking".to_string(),
        ]
    );
    assert_eq!(
        check.completion_field().as_deref(),
        Some(
            "\"repetition_check\": {\"triggers\": [\"tail_repetition:4@thinking\", \"tail_repetition:2@thinking\"]}"
        )
    );
}

#[test]
fn sse_and_completion_bytes_match_the_spec_examples() {
    let one = ["tail_repetition:4@thinking"];
    let two = ["tail_repetition:4@thinking", "tail_repetition:2@thinking"];

    assert_eq!(
        format_mid_stream_event(&one).as_deref(),
        Some(
            "event: response.repetition_check\n\
             data: {\"type\":\"response.repetition_check\",\"repetition_check\":{\"triggers\":[\"tail_repetition:4@thinking\"]}}\n"
        )
    );
    assert_eq!(
        format_completion_field(&two).as_deref(),
        Some(
            "\"repetition_check\": {\"triggers\": [\"tail_repetition:4@thinking\", \"tail_repetition:2@thinking\"]}"
        )
    );
    assert!(format_mid_stream_event(&[] as &[&str]).is_none());
    assert!(format_completion_field(&[] as &[&str]).is_none());

    let escaped = format_mid_stream_event(&["a\"b\\c"]).unwrap();
    assert!(escaped.contains(r#"["a\"b\\c"]"#));
}

#[test]
fn mock_tail_detector_emits_tail_repetition_4_thinking() {
    let mut check = RepetitionCheck::new(
        CheckConfig::parse_headers([(TAIL_CHECK_HEADER, "1024")]),
        ScriptedTail {
            answer: vec![4],
            calls: vec![],
        },
    );
    assert_eq!(check.config().tail_lookback(), Some(1024));
    assert!(check.config().exact_min_tokens().is_none());

    let outcome = check.observe(Channel::Thinking, accept(11));
    assert!(outcome.new_trigger);
    assert_eq!(check.triggers(), ["tail_repetition:4@thinking".to_string()]);
    assert_eq!(
        outcome.mid_stream_event.as_deref(),
        Some(
            "event: response.repetition_check\n\
             data: {\"type\":\"response.repetition_check\",\"repetition_check\":{\"triggers\":[\"tail_repetition:4@thinking\"]}}\n"
        )
    );

    let calls = &check.detector_mut().calls;
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, Channel::Thinking);
    assert_eq!(calls[0].1, vec![11]);

    // An obvious period still produces no tail label unless the detector says so.
    check.detector_mut().answer.clear();
    feed(&mut check, Channel::Response, &[1, 2, 3, 1, 2, 3, 1, 2, 3]);
    assert!(check
        .triggers()
        .iter()
        .all(|label| !label.starts_with("tail_repetition") || label.ends_with("@thinking")));
    assert!(!check
        .triggers()
        .iter()
        .any(|label| label.contains("low_logprob") || label.contains("@response")));
}

#[test]
fn exact_and_tail_on_one_token_insert_exact_first() {
    let mut check = RepetitionCheck::new(
        CheckConfig::parse_headers([(EXACT_REPEAT_CHECK_HEADER, "1"), (TAIL_CHECK_HEADER, "512")]),
        ScriptedTail {
            answer: vec![4],
            calls: vec![],
        },
    );
    // No `t` until the second copy, so both labels are born on the same token.
    check.detector_mut().answer.clear();
    let outcome = check.observe(Channel::Response, accept(5));
    assert!(!outcome.new_trigger);
    check.detector_mut().answer = vec![4];
    let outcome = check.observe(Channel::Response, accept(5));
    assert!(outcome.new_trigger);
    assert_eq!(
        check.triggers(),
        [
            "exact_repetition:1x2@response".to_string(),
            "tail_repetition:4@response".to_string(),
        ]
    );
}

#[test]
fn headers_parse_from_a_mixed_list() {
    let config = CheckConfig::parse_headers([
        ("accept", "*/*"),
        ("x-berth-exact-repeat-check", "64"),
        ("x-berth-tail-check", "4096"),
    ]);
    assert_eq!(config.exact_min_tokens(), Some(64));
    assert_eq!(config.tail_lookback(), Some(4096));
}

#[test]
fn exact_run_to_64x70_keeps_one_label_and_a_later_tail_still_fits() {
    let mut check = RepetitionCheck::new(
        CheckConfig::new(Some(TAIL_LOOKBACK_MIN), Some(PROPOSED_EXACT_MIN_TOKENS)),
        FixedTail { answer: Vec::new() },
    );
    let block = block(PROPOSED_EXACT_MIN_TOKENS);
    for copy in 1..=70u32 {
        feed(&mut check, Channel::Thinking, &block);
        if copy == 1 {
            assert!(check.triggers().is_empty());
        } else {
            assert_eq!(
                check.triggers(),
                [format!("exact_repetition:64x{copy}@thinking")]
            );
        }
    }

    check.detector_mut().answer = vec![4];
    let outcome = check.observe(Channel::Thinking, accept(u32::MAX));
    assert!(outcome.new_trigger);
    assert_eq!(
        check.triggers(),
        [
            "exact_repetition:64x70@thinking".to_string(),
            "tail_repetition:4@thinking".to_string(),
        ]
    );
    let event = outcome.mid_stream_event.expect("the tail label is new");
    assert!(event.contains("[\"exact_repetition:64x70@thinking\",\"tail_repetition:4@thinking\"]"));
}

#[test]
fn exact_rewrite_still_fits_when_the_trigger_list_is_full() {
    let mut check = RepetitionCheck::new(
        CheckConfig::new(Some(TAIL_LOOKBACK_MIN), Some(4)),
        FixedTail { answer: Vec::new() },
    );
    let block = block(4);
    feed(&mut check, Channel::Response, &block);
    feed(&mut check, Channel::Response, &block);
    assert_eq!(
        check.triggers(),
        ["exact_repetition:4x2@response".to_string()]
    );

    check.detector_mut().answer = (0..63).collect();
    let filled = check.observe(Channel::Response, accept(block[0]));
    assert!(filled.new_trigger);
    assert_eq!(check.triggers().len(), MAX_TRIGGERS);
    assert_eq!(check.triggers()[0], "exact_repetition:4x2@response");
    assert_eq!(check.triggers()[1], "tail_repetition:0@response");
    assert_eq!(check.triggers()[63], "tail_repetition:62@response");
    let tails = check.triggers()[1..].to_vec();

    let mut rewritten = Outcome {
        new_trigger: false,
        mid_stream_event: None,
    };
    for token in &block[1..] {
        rewritten = check.observe(Channel::Response, accept(*token));
    }
    assert!(
        rewritten.new_trigger,
        "raising the count sends the set again"
    );
    assert_eq!(check.triggers().len(), MAX_TRIGGERS);
    assert_eq!(check.triggers()[0], "exact_repetition:4x3@response");
    assert_eq!(&check.triggers()[1..], &tails[..]);
    let event = rewritten.mid_stream_event.expect("rewrite is a change");
    assert!(event.contains("exact_repetition:4x3@response"));
    assert!(!event.contains("exact_repetition:4x2@response"));
    assert!(event.contains("tail_repetition:0@response"));
    assert!(event.contains("tail_repetition:62@response"));
    assert!(
        event.find("exact_repetition:4x3@response").unwrap()
            < event.find("tail_repetition:0@response").unwrap()
    );

    check.detector_mut().answer = vec![999];
    let dropped = check.observe(Channel::Response, accept(u32::MAX));
    assert!(!dropped.new_trigger);
    assert!(dropped.mid_stream_event.is_none());
    assert_eq!(check.triggers().len(), MAX_TRIGGERS);
    assert_eq!(check.triggers()[0], "exact_repetition:4x3@response");
}

#[test]
fn same_exact_length_on_the_other_channel_is_a_separate_label() {
    let mut check = RepetitionCheck::new(
        CheckConfig::new(None, Some(PROPOSED_EXACT_MIN_TOKENS)),
        PanicTail,
    );
    let block = block(PROPOSED_EXACT_MIN_TOKENS);
    feed(&mut check, Channel::Thinking, &block);
    feed(&mut check, Channel::Thinking, &block);
    feed(&mut check, Channel::Response, &block);
    feed(&mut check, Channel::Response, &block);
    assert_eq!(
        check.triggers(),
        [
            "exact_repetition:64x2@thinking".to_string(),
            "exact_repetition:64x2@response".to_string(),
        ]
    );

    feed(&mut check, Channel::Thinking, &block);
    assert_eq!(
        check.triggers(),
        [
            "exact_repetition:64x3@thinking".to_string(),
            "exact_repetition:64x2@response".to_string(),
        ]
    );
}

#[test]
fn different_exact_lengths_each_keep_one_label() {
    let mut check = RepetitionCheck::new(CheckConfig::new(None, Some(2)), PanicTail);
    feed(&mut check, Channel::Response, &[1, 2, 1, 2]);
    assert_eq!(
        check.triggers(),
        ["exact_repetition:2x2@response".to_string()]
    );
    feed(&mut check, Channel::Response, &[3, 4, 5, 6, 3, 4, 5, 6]);
    assert_eq!(
        check.triggers(),
        [
            "exact_repetition:2x2@response".to_string(),
            "exact_repetition:4x2@response".to_string(),
        ]
    );
}

#[test]
fn smaller_exact_count_does_not_downgrade() {
    let mut check = RepetitionCheck::new(CheckConfig::new(None, Some(8)), PanicTail);
    let first = block(8);
    let second: Vec<u32> = (100..108).collect();
    feed(&mut check, Channel::Thinking, &first);
    feed(&mut check, Channel::Thinking, &first);
    feed(&mut check, Channel::Thinking, &first);
    assert_eq!(
        check.triggers(),
        ["exact_repetition:8x3@thinking".to_string()]
    );

    let mut raised = false;
    for _ in 0..2 {
        for token in &second {
            raised |= check.observe(Channel::Thinking, accept(*token)).new_trigger;
        }
    }
    assert!(!raised);
    assert_eq!(
        check.triggers(),
        ["exact_repetition:8x3@thinking".to_string()]
    );
    let event = check.mid_stream_event().expect("the larger count stays");
    assert!(event.contains("exact_repetition:8x3@thinking"));
    assert!(!event.contains("exact_repetition:8x2@thinking"));

    let mut alone = RepetitionCheck::new(CheckConfig::new(None, Some(8)), PanicTail);
    feed(&mut alone, Channel::Thinking, &second);
    feed(&mut alone, Channel::Thinking, &second);
    assert_eq!(
        alone.triggers(),
        ["exact_repetition:8x2@thinking".to_string()],
        "the later block would have been 8x2 on its own"
    );
}
