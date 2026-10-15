//! Scripted-backend tests. No GPU and no bind on port 8889.

use std::io::{Read, Write};
use std::net::{Shutdown, SocketAddr, TcpStream};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use berth::{
    load_assets, serve, Assets, Backend, Finish, Flow, GenerateRequest, SamplerConfig, ServeConfig,
    Usage, DEFAULT_CHAT_MAX_TOKENS, DEFAULT_COMPLETION_MAX_TOKENS, DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_NAME,
};
use serde_json::{json, Value};

fn assets() -> Arc<Assets> {
    static ASSETS: OnceLock<Arc<Assets>> = OnceLock::new();
    Arc::clone(ASSETS.get_or_init(|| {
        Arc::new(
            load_assets(Path::new(DEFAULT_MODEL_DIR))
                .expect("load the 27B tokenizer and chat template"),
        )
    }))
}

struct Shared {
    tokens: Vec<u32>,
    hold: Duration,
    delay: Duration,
    max_tokens: u32,
    active: AtomicUsize,
    max_active: AtomicUsize,
    cancelled: AtomicBool,
    starts: Mutex<Vec<Instant>>,
    ends: Mutex<Vec<Instant>>,
    requests: Mutex<Vec<GenerateRequest>>,
}

struct Fake {
    shared: Arc<Shared>,
}

impl Backend for Fake {
    fn vocab_size(&self) -> u32 {
        262_144
    }

    fn max_tokens(&self) -> u32 {
        self.shared.max_tokens
    }

    fn generate(
        &mut self,
        request: &GenerateRequest,
        on_token: &mut dyn FnMut(u32) -> Flow,
    ) -> berth_engine::Result<(Finish, Usage)> {
        self.shared
            .requests
            .lock()
            .expect("requests")
            .push(request.clone());
        self.shared
            .starts
            .lock()
            .expect("starts")
            .push(Instant::now());
        let active = self.shared.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.shared.max_active.fetch_max(active, Ordering::SeqCst);
        if !self.shared.hold.is_zero() {
            thread::sleep(self.shared.hold);
        }
        let mut produced = 0u32;
        let mut finish = Finish::Length;
        for &token in &self.shared.tokens {
            if produced >= request.max_new_tokens {
                finish = Finish::Length;
                break;
            }
            if !self.shared.delay.is_zero() {
                thread::sleep(self.shared.delay);
            }
            let flow = on_token(token);
            produced += 1;
            if flow == Flow::Cancel {
                self.shared.cancelled.store(true, Ordering::SeqCst);
                finish = Finish::Cancelled;
                break;
            }
            if request.stop_tokens.contains(&token) {
                finish = Finish::Stop;
                break;
            }
        }
        self.shared.active.fetch_sub(1, Ordering::SeqCst);
        self.shared.ends.lock().expect("ends").push(Instant::now());
        Ok((
            finish,
            Usage {
                prompt_tokens: request.prompt.len() as u32,
                completion_tokens: produced,
            },
        ))
    }
}

fn shared(tokens: Vec<u32>, hold: Duration, delay: Duration) -> Arc<Shared> {
    Arc::new(Shared {
        tokens,
        hold,
        delay,
        max_tokens: 8192,
        active: AtomicUsize::new(0),
        max_active: AtomicUsize::new(0),
        cancelled: AtomicBool::new(false),
        starts: Mutex::new(Vec::new()),
        ends: Mutex::new(Vec::new()),
        requests: Mutex::new(Vec::new()),
    })
}

fn start(fake: Arc<Shared>) -> berth::ServeHandle {
    let backend = Fake {
        shared: Arc::clone(&fake),
    };
    serve(
        assets(),
        Box::new(backend),
        ServeConfig {
            host: "127.0.0.1".into(),
            port: 0,
            model_name: DEFAULT_MODEL_NAME.into(),
            chat_max_tokens: DEFAULT_CHAT_MAX_TOKENS,
            completion_max_tokens: DEFAULT_COMPLETION_MAX_TOKENS,
        },
    )
    .expect("bind ephemeral port")
}

struct RawResponse {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

fn http(
    addr: SocketAddr,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: &[u8],
) -> RawResponse {
    let mut sock = TcpStream::connect(addr).expect("connect");
    sock.set_nodelay(true).expect("nodelay");
    sock.set_read_timeout(Some(Duration::from_secs(10)))
        .expect("timeout");
    write_request(&mut sock, method, path, headers, body);
    read_response(&mut sock)
}

fn write_request(
    sock: &mut TcpStream,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: &[u8],
) {
    let mut head = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for (name, value) in headers {
        head.push_str(name);
        head.push_str(": ");
        head.push_str(value);
        head.push_str("\r\n");
    }
    head.push_str("\r\n");
    sock.write_all(head.as_bytes()).expect("write head");
    sock.write_all(body).expect("write body");
}

fn read_response(sock: &mut TcpStream) -> RawResponse {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    loop {
        if buf.windows(4).any(|window| window == b"\r\n\r\n") {
            break;
        }
        match sock.read(&mut tmp) {
            Ok(0) => panic!("eof before headers: {}", String::from_utf8_lossy(&buf)),
            Ok(n) => buf.extend_from_slice(&tmp[..n]),
            Err(err) => panic!("read headers: {err}"),
        }
    }
    let split = buf
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .expect("headers");
    let header_text = String::from_utf8_lossy(&buf[..split]).into_owned();
    let mut body = buf[split + 4..].to_vec();
    let mut lines = header_text.split("\r\n");
    let status_line = lines.next().expect("status");
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .expect("code")
        .parse()
        .expect("status number");
    let mut headers = Vec::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            headers.push((name.trim().to_ascii_lowercase(), value.trim().to_string()));
        }
    }
    let content_length = headers.iter().find_map(|(name, value)| {
        (name == "content-length").then(|| value.parse::<usize>().expect("length"))
    });
    let chunked = headers.iter().any(|(name, value)| {
        name == "transfer-encoding" && value.to_ascii_lowercase().contains("chunked")
    });
    if let Some(len) = content_length {
        while body.len() < len {
            match sock.read(&mut tmp) {
                Ok(0) => break,
                Ok(n) => body.extend_from_slice(&tmp[..n]),
                Err(err) => panic!("body: {err}"),
            }
        }
        body.truncate(len);
    } else if chunked {
        body = read_chunked(sock, body);
    } else {
        loop {
            match sock.read(&mut tmp) {
                Ok(0) => break,
                Ok(n) => body.extend_from_slice(&tmp[..n]),
                Err(err)
                    if err.kind() == std::io::ErrorKind::WouldBlock
                        || err.kind() == std::io::ErrorKind::TimedOut =>
                {
                    break;
                }
                Err(err) => panic!("body: {err}"),
            }
        }
    }
    RawResponse {
        status,
        headers,
        body,
    }
}

fn read_chunked(sock: &mut TcpStream, mut pending: Vec<u8>) -> Vec<u8> {
    let mut out = Vec::new();
    let mut tmp = [0u8; 8192];
    loop {
        while !pending.windows(2).any(|window| window == b"\r\n") {
            match sock.read(&mut tmp) {
                Ok(0) => return out,
                Ok(n) => pending.extend_from_slice(&tmp[..n]),
                Err(err) => panic!(
                    "chunk size: {err} pending {}",
                    String::from_utf8_lossy(&pending)
                ),
            }
        }
        let line_end = pending
            .windows(2)
            .position(|window| window == b"\r\n")
            .expect("crlf");
        let line = String::from_utf8_lossy(&pending[..line_end])
            .trim()
            .to_string();
        let size = usize::from_str_radix(line.split(';').next().unwrap_or("").trim(), 16)
            .unwrap_or_else(|_| panic!("chunk size {line}"));
        pending.drain(..line_end + 2);
        if size == 0 {
            return out;
        }
        while pending.len() < size + 2 {
            match sock.read(&mut tmp) {
                Ok(0) => break,
                Ok(n) => pending.extend_from_slice(&tmp[..n]),
                Err(err) => panic!("chunk data: {err}"),
            }
        }
        assert!(pending.len() >= size + 2, "short chunk");
        out.extend_from_slice(&pending[..size]);
        pending.drain(..size + 2);
    }
}

fn header<'a>(response: &'a RawResponse, name: &str) -> &'a str {
    response
        .headers
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.as_str())
        .unwrap_or("")
}

struct SseFrame {
    event: Option<String>,
    data: Option<String>,
}

fn sse_frames(body: &str) -> Vec<SseFrame> {
    body.split("\n\n")
        .filter_map(|block| {
            let block = block.trim_start_matches('\n');
            if block.is_empty() {
                return None;
            }
            let mut event = None;
            let mut data = None;
            for line in block.split('\n') {
                if let Some(rest) = line.strip_prefix("event:") {
                    event = Some(rest.trim().to_string());
                } else if let Some(rest) = line.strip_prefix("data:") {
                    data = Some(rest.trim().to_string());
                }
            }
            Some(SseFrame { event, data })
        })
        .collect()
}

fn think_tokens() -> Vec<u32> {
    let mut ids = assets()
        .encode("<think>\nwave 👋\n</think>\n\nOK", false)
        .expect("encode scripted text");
    ids.push(assets().token_id("<|im_end|>").expect("im_end"));
    ids
}

#[test]
fn chat_template_matches_the_serve_render() {
    let fixture: Value = serde_json::from_str(include_str!("data/two_message_chat.json")).unwrap();
    let messages = fixture["messages"].as_array().unwrap();
    let rendered = assets()
        .render_chat(messages, None, &serde_json::Map::new())
        .expect("render");
    assert_eq!(rendered, fixture["rendered"].as_str().unwrap());
}

#[test]
fn chat_nonstream_and_stream_split_reasoning_and_keep_utf8() {
    let tokens = think_tokens();
    let token_len = tokens.len() as u64;
    let state = shared(tokens, Duration::ZERO, Duration::ZERO);
    let server = start(Arc::clone(&state));
    let addr = server.addr();
    assert_ne!(addr.port(), 8889);

    let health = http(addr, "GET", "/health", &[], b"");
    assert_eq!(health.status, 200);
    assert_eq!(health.body, br#"{"status":"ok"}"#);

    let models = http(addr, "GET", "/v1/models", &[], b"");
    assert_eq!(models.status, 200);
    let models: Value = serde_json::from_slice(&models.body).unwrap();
    assert_eq!(models["data"][0]["id"], DEFAULT_MODEL_NAME);
    assert_eq!(models["data"][0]["max_model_len"], 8192);

    let messages = json!([
        {"role": "system", "content": "You are berth."},
        {"role": "user", "content": "Count to two."}
    ]);
    let body = json!({
        "model": "unit-test",
        "messages": messages,
        "max_tokens": 64,
    });
    let response = http(
        addr,
        "POST",
        "/v1/chat/completions",
        &[("content-type", "application/json")],
        body.to_string().as_bytes(),
    );
    assert_eq!(
        response.status,
        200,
        "{}",
        String::from_utf8_lossy(&response.body)
    );
    let completion: Value = serde_json::from_slice(&response.body).expect("json");
    assert_eq!(completion["object"], "chat.completion");
    assert_eq!(completion["model"], "unit-test");
    assert!(completion["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert_eq!(
        completion["id"].as_str().unwrap().len(),
        "chatcmpl-".len() + 24
    );
    assert_eq!(completion["choices"][0]["finish_reason"], "stop");
    assert_eq!(
        completion["choices"][0]["message"]["reasoning_content"],
        "<think>\nwave 👋\n"
    );
    assert_eq!(completion["choices"][0]["message"]["content"], "OK");
    assert!(completion.get("stats").is_none());
    assert!(!String::from_utf8_lossy(&response.body).contains('\u{FFFD}'));
    let prompt = assets()
        .encode(
            assets()
                .render_chat(messages.as_array().unwrap(), None, &serde_json::Map::new())
                .unwrap()
                .as_str(),
            false,
        )
        .unwrap();
    assert_eq!(completion["usage"]["prompt_tokens"], prompt.len() as u64);
    assert_eq!(completion["usage"]["completion_tokens"], token_len);
    assert_eq!(
        completion["usage"]["total_tokens"],
        prompt.len() as u64 + token_len
    );

    let recorded = state.requests.lock().unwrap().clone();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].prompt, prompt);
    assert_eq!(recorded[0].max_new_tokens, 64);
    assert_eq!(recorded[0].stop_tokens, assets().eos_ids());
    assert_eq!(recorded[0].sampler, SamplerConfig::serve_defaults(0.0, 0));

    let stream_body = json!({
        "model": "unit-test",
        "messages": messages,
        "max_tokens": 64,
        "stream": true,
    });
    let streamed = http(
        addr,
        "POST",
        "/v1/chat/completions",
        &[("content-type", "application/json")],
        stream_body.to_string().as_bytes(),
    );
    assert_eq!(
        streamed.status,
        200,
        "{}",
        String::from_utf8_lossy(&streamed.body)
    );
    assert!(header(&streamed, "content-type").starts_with("text/event-stream"));
    assert_eq!(header(&streamed, "cache-control"), "no-cache");
    assert_eq!(header(&streamed, "x-accel-buffering"), "no");
    let text = String::from_utf8(streamed.body.clone()).expect("utf-8 body");
    assert!(!text.contains('\u{FFFD}'), "{text}");
    let frames = sse_frames(&text);
    assert!(frames
        .iter()
        .any(|frame| frame.data.as_deref() == Some("[DONE]")));
    let data: Vec<Value> = frames
        .iter()
        .filter(|frame| frame.event.is_none())
        .filter_map(|frame| frame.data.as_ref())
        .filter(|data| data.as_str() != "[DONE]")
        .map(|data| serde_json::from_str(data).expect(data))
        .collect();
    assert!(data.len() >= 3, "{text}");
    assert_eq!(data[0]["object"], "chat.completion.chunk");
    assert_eq!(data[0]["choices"][0]["delta"]["role"], "assistant");
    assert!(data[0]["choices"][0]["finish_reason"].is_null());
    let mut reasoning = String::new();
    let mut content = String::new();
    let mut saw_emoji = false;
    for chunk in &data {
        assert_eq!(chunk["object"], "chat.completion.chunk");
        assert_eq!(chunk["id"], data[0]["id"]);
        let delta = &chunk["choices"][0]["delta"];
        if let Some(piece) = delta["reasoning_content"].as_str() {
            assert!(!piece.contains('\u{FFFD}'));
            if piece.contains('👋') {
                saw_emoji = true;
            }
            reasoning.push_str(piece);
        }
        if let Some(piece) = delta["content"].as_str() {
            assert!(!piece.contains('\u{FFFD}'));
            content.push_str(piece);
        }
    }
    assert!(saw_emoji, "emoji was split or dropped: {text}");
    assert_eq!(reasoning, "<think>\nwave 👋\n");
    assert_eq!(content, "OK");
    let last = data.last().unwrap();
    assert_eq!(last["choices"][0]["delta"], json!({}));
    assert_eq!(last["choices"][0]["finish_reason"], "stop");
    assert_eq!(last["usage"]["prompt_tokens"], prompt.len() as u64);
    assert_eq!(last["usage"]["completion_tokens"], token_len);
    assert_eq!(
        last["usage"]["total_tokens"],
        prompt.len() as u64 + token_len
    );
    assert!(frames.last().unwrap().data.as_deref() == Some("[DONE]"));
}

#[test]
fn exact_repeat_header_emits_the_repetition_event_and_completion_field() {
    let state = shared(vec![0, 1, 2, 3, 0, 1, 2, 3], Duration::ZERO, Duration::ZERO);
    let server = start(Arc::clone(&state));
    let addr = server.addr();
    let headers = [
        ("content-type", "application/json"),
        ("x-berth-exact-repeat-check", "4"),
        ("x-berth-tail-check", "1024"),
    ];
    let stream_body = json!({
        "model": "unit-test",
        "prompt": "loop",
        "max_tokens": 32,
        "stream": true,
    });
    let streamed = http(
        addr,
        "POST",
        "/v1/completions",
        &headers,
        stream_body.to_string().as_bytes(),
    );
    assert_eq!(
        streamed.status,
        200,
        "{}",
        String::from_utf8_lossy(&streamed.body)
    );
    let text = String::from_utf8(streamed.body).unwrap();
    assert!(text.contains("event: response.repetition_check"), "{text}");
    assert!(text.contains("exact_repetition:4x2@response"), "{text}");
    assert!(!text.contains("tail_repetition"), "{text}");
    assert!(text.contains("data: [DONE]\n"));
    let frames = sse_frames(&text);
    let event = frames
        .iter()
        .find(|frame| frame.event.as_deref() == Some("response.repetition_check"))
        .expect("mid-stream event");
    let payload: Value = serde_json::from_str(event.data.as_deref().unwrap()).unwrap();
    assert_eq!(payload["type"], "response.repetition_check");
    assert_eq!(
        payload["repetition_check"]["triggers"],
        json!(["exact_repetition:4x2@response"])
    );
    let chunks: Vec<Value> = frames
        .iter()
        .filter(|frame| frame.event.is_none() && frame.data.as_deref() != Some("[DONE]"))
        .map(|frame| serde_json::from_str(frame.data.as_deref().unwrap()).unwrap())
        .collect();
    assert!(chunks
        .iter()
        .all(|chunk| chunk["object"] == "text_completion"));
    let last = chunks.last().unwrap();
    assert_eq!(last["choices"][0]["text"], "");
    assert_eq!(last["choices"][0]["finish_reason"], "length");
    assert_eq!(
        last["repetition_check"]["triggers"],
        json!(["exact_repetition:4x2@response"])
    );
    assert_eq!(last["usage"]["completion_tokens"], 8);

    let body = json!({
        "model": "unit-test",
        "prompt": [1, 2, 3],
        "max_tokens": 16,
        "temperature": 0.7,
        "repetition_penalty": 1.1,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.3,
        "frequency_range": 128,
        "dry_multiplier": 0.5,
        "dry_base": 1.5,
        "dry_allowed_length": 3,
        "dry_range": 100,
        "seed": 99,
    });
    let response = http(
        addr,
        "POST",
        "/v1/completions",
        &headers,
        body.to_string().as_bytes(),
    );
    assert_eq!(
        response.status,
        200,
        "{}",
        String::from_utf8_lossy(&response.body)
    );
    let completion: Value = serde_json::from_slice(&response.body).unwrap();
    assert_eq!(completion["object"], "text_completion");
    assert!(completion["id"].as_str().unwrap().starts_with("cmpl-"));
    assert_eq!(
        completion["repetition_check"]["triggers"],
        json!(["exact_repetition:4x2@response"])
    );
    assert_eq!(completion["usage"]["prompt_tokens"], 3);
    assert_eq!(completion["usage"]["completion_tokens"], 8);
    assert_eq!(completion["usage"]["total_tokens"], 11);
    let requests = state.requests.lock().unwrap();
    let sampler = &requests[1].sampler;
    assert_eq!(sampler.temperature, 0.7f64 as f32);
    assert_eq!(sampler.rep_penalty, 1.1f64 as f32);
    assert_eq!(sampler.frequency_penalty, 0.2f64 as f32);
    assert_eq!(sampler.presence_penalty, 0.3f64 as f32);
    assert_eq!(sampler.penalty_range, 128);
    assert_eq!(sampler.dry_multiplier, 0.5f64 as f32);
    assert_eq!(sampler.dry_base, 1.5f64 as f32);
    assert_eq!(sampler.dry_allowed_length, 3);
    assert_eq!(sampler.dry_range, 100);
    assert_eq!(sampler.seed, 99);
    assert_eq!(requests[1].prompt, vec![1, 2, 3]);
    assert_eq!(requests[1].max_new_tokens, 16);
}

#[test]
fn disconnect_cancels_the_in_flight_generation() {
    let tokens = assets().encode("Hi", false).expect("hi").repeat(40);
    let state = shared(tokens, Duration::ZERO, Duration::from_millis(30));
    let server = start(Arc::clone(&state));
    let addr = server.addr();
    let body = json!({
        "model": "unit-test",
        "prompt": "x",
        "max_tokens": 100,
        "stream": true,
    })
    .to_string();
    let mut sock = TcpStream::connect(addr).expect("connect");
    sock.set_nodelay(true).unwrap();
    sock.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
    write_request(
        &mut sock,
        "POST",
        "/v1/completions",
        &[("content-type", "application/json")],
        body.as_bytes(),
    );
    let mut buf = [0u8; 1024];
    let _ = sock.read(&mut buf);
    sock.shutdown(Shutdown::Both).ok();
    drop(sock);

    let deadline = Instant::now() + Duration::from_secs(3);
    while !state.cancelled.load(Ordering::SeqCst) && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(20));
    }
    assert!(
        state.cancelled.load(Ordering::SeqCst),
        "generation was not cancelled"
    );
    let health = http(addr, "GET", "/v1/health", &[], b"");
    assert_eq!(health.status, 200);
    assert_eq!(health.body, br#"{"status":"ok"}"#);
}

#[test]
fn two_concurrent_requests_run_one_at_a_time() {
    let tokens = assets().encode("Hi", false).expect("hi");
    let state = shared(tokens, Duration::from_millis(200), Duration::ZERO);
    let server = start(Arc::clone(&state));
    let addr = server.addr();
    let barrier = Arc::new(Barrier::new(2));
    let mut handles = Vec::new();
    for _ in 0..2 {
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            barrier.wait();
            let body = json!({
                "prompt": "queued",
                "max_tokens": 8,
            })
            .to_string();
            http(
                addr,
                "POST",
                "/v1/completions",
                &[("content-type", "application/json")],
                body.as_bytes(),
            )
        }));
    }
    let responses: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect();
    for response in &responses {
        assert_eq!(
            response.status,
            200,
            "{}",
            String::from_utf8_lossy(&response.body)
        );
    }
    assert_eq!(state.max_active.load(Ordering::SeqCst), 1);
    let starts = state.starts.lock().unwrap().clone();
    let ends = state.ends.lock().unwrap().clone();
    assert_eq!(starts.len(), 2);
    assert_eq!(ends.len(), 2);
    assert!(
        starts[1] >= ends[0],
        "requests overlapped: {starts:?} {ends:?}"
    );
}
