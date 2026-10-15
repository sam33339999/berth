//! HTTP routes, the single backend thread, and the request queue.
//!
//! One generation runs at a time. The backend thread pulls jobs from an
//! unbounded queue. Dropping the response body sets a cancel flag; the next
//! token returns [`Flow::Cancel`].

use std::convert::Infallible;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::thread::JoinHandle;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::extract::rejection::JsonRejection;
use axum::extract::{DefaultBodyLimit, Json, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use axum::routing::{get, post};
use axum::Router;
use berth_engine::Finish;
use berth_repetition::CheckConfig;
use bytes::Bytes;
use serde_json::{json, Map, Value};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::{Stream, StreamExt};

use crate::assets::Assets;
use crate::think::split_nonstream;
use crate::turn::{send_outbound, Outbound, Turn};
use crate::{Backend, Error, GenerateRequest, Result, SamplerConfig, Usage, MAX_BODY};

/// Bind address and the sampling defaults that are not per request.
#[derive(Clone, Debug)]
pub struct ServeConfig {
    pub host: String,
    pub port: u16,
    pub model_name: String,
    pub chat_max_tokens: u32,
    pub completion_max_tokens: u32,
}

pub struct ServeHandle {
    addr: SocketAddr,
    shutdown: Option<oneshot::Sender<()>>,
    http: Option<JoinHandle<()>>,
    worker: Option<JoinHandle<()>>,
}

impl ServeHandle {
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }
}

impl Drop for ServeHandle {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(http) = self.http.take() {
            let _ = http.join();
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

struct AppState {
    assets: Arc<Assets>,
    jobs: std::sync::mpsc::Sender<Job>,
    max_tokens: u32,
    model_name: String,
    chat_max_tokens: u32,
    completion_max_tokens: u32,
}

impl Clone for AppState {
    fn clone(&self) -> Self {
        Self {
            assets: Arc::clone(&self.assets),
            jobs: self.jobs.clone(),
            max_tokens: self.max_tokens,
            model_name: self.model_name.clone(),
            chat_max_tokens: self.chat_max_tokens,
            completion_max_tokens: self.completion_max_tokens,
        }
    }
}

struct Job {
    chat: bool,
    in_think: bool,
    skip_special: bool,
    stop_strings: Vec<String>,
    repetition: CheckConfig,
    request: GenerateRequest,
    cancel: Arc<AtomicBool>,
    tokens: mpsc::Sender<Outbound>,
    done: oneshot::Sender<TurnOutcome>,
}

struct TurnOutcome {
    finish: Finish,
    usage: Usage,
    triggers: Vec<String>,
    raw: String,
    stop_string: bool,
    error: Option<String>,
}

struct Prepared {
    chat: bool,
    in_think: bool,
    skip_special: bool,
    stop_strings: Vec<String>,
    repetition: CheckConfig,
    request: GenerateRequest,
    stream: bool,
    model: String,
}

/// Bind `config` and serve `backend` until the returned handle is dropped.
///
/// The HTTP runtime lives on its own thread so callers can block on sockets
/// without sharing that runtime. `backend` is moved to a second thread.
pub fn serve(
    assets: Arc<Assets>,
    backend: Box<dyn Backend>,
    config: ServeConfig,
) -> Result<ServeHandle> {
    let max_tokens = backend.max_tokens();
    let (job_tx, job_rx) = std::sync::mpsc::channel::<Job>();
    let worker_assets = Arc::clone(&assets);
    let worker = std::thread::Builder::new()
        .name("berth-backend".into())
        .spawn(move || worker_loop(backend, worker_assets, job_rx))
        .map_err(|err| Error::new(format!("spawn backend thread: {err}")))?;

    let state = AppState {
        assets,
        jobs: job_tx,
        max_tokens,
        model_name: config.model_name,
        chat_max_tokens: config.chat_max_tokens,
        completion_max_tokens: config.completion_max_tokens,
    };
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let host = config.host;
    let port = config.port;
    let http = std::thread::Builder::new()
        .name("berth-http".into())
        .spawn(move || http_thread(host, port, state, shutdown_rx, ready_tx))
        .map_err(|err| Error::new(format!("spawn http thread: {err}")))?;

    match ready_rx.recv() {
        Ok(Ok(addr)) => Ok(ServeHandle {
            addr,
            shutdown: Some(shutdown_tx),
            http: Some(http),
            worker: Some(worker),
        }),
        Ok(Err(err)) => {
            drop(shutdown_tx);
            let _ = http.join();
            let _ = worker.join();
            Err(err)
        }
        Err(_) => {
            drop(shutdown_tx);
            let _ = http.join();
            let _ = worker.join();
            Err(Error::new("http thread exited before binding"))
        }
    }
}

fn http_thread(
    host: String,
    port: u16,
    state: AppState,
    shutdown_rx: oneshot::Receiver<()>,
    ready_tx: std::sync::mpsc::Sender<std::result::Result<SocketAddr, Error>>,
) {
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            let _ = ready_tx.send(Err(Error::new(format!("tokio runtime: {err}"))));
            return;
        }
    };
    runtime.block_on(async move {
        let listener = match tokio::net::TcpListener::bind((host.as_str(), port)).await {
            Ok(listener) => listener,
            Err(err) => {
                let _ = ready_tx.send(Err(Error::new(format!("bind {host}:{port}: {err}"))));
                return;
            }
        };
        let addr = match listener.local_addr() {
            Ok(addr) => addr,
            Err(err) => {
                let _ = ready_tx.send(Err(Error::new(err.to_string())));
                return;
            }
        };
        let _ = ready_tx.send(Ok(addr));
        let app = router(state);
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await;
    });
}

fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(completions))
        .fallback(not_found)
        .layer(DefaultBodyLimit::max(MAX_BODY))
        .with_state(state)
}

fn worker_loop(
    mut backend: Box<dyn Backend>,
    assets: Arc<Assets>,
    jobs: std::sync::mpsc::Receiver<Job>,
) {
    while let Ok(job) = jobs.recv() {
        let outcome = execute(&mut *backend, &assets, &job);
        let Job { tokens, done, .. } = job;
        let _ = done.send(outcome);
        drop(tokens);
    }
}

fn execute(backend: &mut dyn Backend, assets: &Assets, job: &Job) -> TurnOutcome {
    let mut turn = Turn::new(
        job.chat,
        job.in_think,
        job.skip_special,
        &job.request.stop_tokens,
        &job.stop_strings,
        job.repetition,
    );
    let generated = backend.generate(&job.request, &mut |id| {
        if job.cancel.load(Ordering::SeqCst) {
            return crate::Flow::Cancel;
        }
        let flow = turn.on_token(assets.tokenizer(), id);
        for piece in turn.drain_pieces() {
            if send_outbound(&job.tokens, &job.cancel, piece) == crate::Flow::Cancel {
                return crate::Flow::Cancel;
            }
        }
        if job.cancel.load(Ordering::SeqCst) {
            return crate::Flow::Cancel;
        }
        flow
    });
    for piece in turn.finish(assets.tokenizer()) {
        if send_outbound(&job.tokens, &job.cancel, piece) == crate::Flow::Cancel {
            break;
        }
    }
    match generated {
        Ok((finish, usage)) => TurnOutcome {
            finish,
            usage,
            triggers: turn.triggers().to_vec(),
            raw: turn.raw().to_string(),
            stop_string: turn.stop_string(),
            error: None,
        },
        Err(err) => {
            eprintln!("berth: generate: {err}");
            TurnOutcome {
                finish: Finish::Cancelled,
                usage: Usage {
                    prompt_tokens: job.request.prompt.len() as u32,
                    completion_tokens: 0,
                },
                triggers: turn.triggers().to_vec(),
                raw: turn.raw().to_string(),
                stop_string: turn.stop_string(),
                error: Some(err.to_string()),
            }
        }
    }
}

async fn health() -> Response {
    json_response(StatusCode::OK, &json!({"status": "ok"}))
}

async fn models(State(state): State<AppState>) -> Response {
    json_response(
        StatusCode::OK,
        &json!({
            "object": "list",
            "data": [{
                "id": state.model_name,
                "object": "model",
                "created": unix_now(),
                "owned_by": "local",
                "max_model_len": state.max_tokens,
            }],
        }),
    )
}

async fn not_found() -> Response {
    error_response(StatusCode::NOT_FOUND, "not found", "invalid_request_error")
}

async fn chat(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> Response {
    let value = match body {
        Ok(Json(value)) => value,
        Err(err) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("invalid JSON: {err}"),
                "invalid_request_error",
            )
        }
    };
    let Some(map) = value.as_object() else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "invalid JSON: expected an object",
            "invalid_request_error",
        );
    };
    match prepare_chat(&state, &headers, map) {
        Ok(prepared) => dispatch(state, prepared).await,
        Err(message) => error_response(StatusCode::BAD_REQUEST, &message, "invalid_request_error"),
    }
}

async fn completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> Response {
    let value = match body {
        Ok(Json(value)) => value,
        Err(err) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("invalid JSON: {err}"),
                "invalid_request_error",
            )
        }
    };
    let Some(map) = value.as_object() else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "invalid JSON: expected an object",
            "invalid_request_error",
        );
    };
    match prepare_completion(&state, &headers, map) {
        Ok(prepared) => dispatch(state, prepared).await,
        Err(message) => error_response(StatusCode::BAD_REQUEST, &message, "invalid_request_error"),
    }
}

async fn dispatch(state: AppState, prepared: Prepared) -> Response {
    let Prepared {
        chat,
        in_think,
        skip_special,
        stop_strings,
        repetition,
        request,
        stream,
        model,
    } = prepared;
    let cancel = Arc::new(AtomicBool::new(false));
    let (token_tx, token_rx) = mpsc::channel(1);
    let (done_tx, done_rx) = oneshot::channel();
    let job = Job {
        chat,
        in_think,
        skip_special,
        stop_strings,
        repetition,
        request,
        cancel: Arc::clone(&cancel),
        tokens: token_tx,
        done: done_tx,
    };
    if state.jobs.send(job).is_err() {
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server is shutting down",
            "server_error",
        );
    }
    let meta = Meta {
        id: new_id(if chat { "chatcmpl-" } else { "cmpl-" }),
        created: unix_now(),
        model,
        chat,
    };
    if stream {
        stream_response(meta, token_rx, done_rx, cancel)
    } else {
        json_completion(meta, token_rx, done_rx, cancel).await
    }
}

struct Meta {
    id: String,
    created: u64,
    model: String,
    chat: bool,
}

async fn json_completion(
    meta: Meta,
    mut token_rx: mpsc::Receiver<Outbound>,
    done_rx: oneshot::Receiver<TurnOutcome>,
    cancel: Arc<AtomicBool>,
) -> Response {
    let _guard = CancelFlag(cancel);
    while token_rx.recv().await.is_some() {}
    let outcome = match done_rx.await {
        Ok(outcome) => outcome,
        Err(_) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "generation dropped",
                "server_error",
            )
        }
    };
    if let Some(err) = outcome.error {
        return error_response(StatusCode::INTERNAL_SERVER_ERROR, &err, "server_error");
    }
    if outcome.finish == Finish::Cancelled && !outcome.stop_string {
        return Response::builder()
            .status(StatusCode::NO_CONTENT)
            .body(Body::empty())
            .expect("no-content response");
    }
    json_response(StatusCode::OK, &completion_body(&meta, &outcome))
}

fn stream_response(
    meta: Meta,
    token_rx: mpsc::Receiver<Outbound>,
    done_rx: oneshot::Receiver<TurnOutcome>,
    cancel: Arc<AtomicBool>,
) -> Response {
    let (out_tx, out_rx) = mpsc::channel::<Bytes>(8);
    let flag = Arc::clone(&cancel);
    tokio::spawn(async move {
        if meta.chat
            && out_tx
                .send(Bytes::from(data_frame(&chat_chunk(
                    &meta,
                    json!({"role": "assistant"}),
                    None,
                    None,
                    &[],
                ))))
                .await
                .is_err()
        {
            return;
        }
        let mut token_rx = token_rx;
        while let Some(piece) = token_rx.recv().await {
            let frame = frame_for(&meta, piece);
            if out_tx.send(Bytes::from(frame)).await.is_err() {
                return;
            }
        }
        let Ok(outcome) = done_rx.await else {
            return;
        };
        if outcome.finish == Finish::Cancelled && !outcome.stop_string {
            return;
        }
        if let Some(err) = outcome.error.as_deref() {
            let payload = json!({"error": {"message": err, "type": "server_error"}});
            let _ = out_tx.send(Bytes::from(data_frame(&payload))).await;
            let _ = out_tx.send(Bytes::from("data: [DONE]\n\n")).await;
            return;
        }
        let final_chunk = if meta.chat {
            chat_chunk(
                &meta,
                json!({}),
                Some(finish_reason(&outcome)),
                Some(usage_json(outcome.usage)),
                &outcome.triggers,
            )
        } else {
            completion_chunk(
                &meta,
                "",
                Some(finish_reason(&outcome)),
                Some(usage_json(outcome.usage)),
                &outcome.triggers,
            )
        };
        let _ = out_tx.send(Bytes::from(data_frame(&final_chunk))).await;
        let _ = out_tx.send(Bytes::from("data: [DONE]\n\n")).await;
    });
    let stream = ReceiverStream::new(out_rx).map(Ok::<Bytes, Infallible>);
    let stream = FlagStream {
        inner: stream,
        flag,
    };
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream; charset=utf-8")
        .header("cache-control", "no-cache")
        .header("connection", "keep-alive")
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(stream))
        .expect("sse response")
}

fn frame_for(meta: &Meta, piece: Outbound) -> String {
    match piece {
        Outbound::Repetition(event) => {
            if event.ends_with("\n\n") {
                event
            } else {
                format!("{event}\n")
            }
        }
        Outbound::Reasoning(text) => data_frame(&chat_chunk(
            meta,
            json!({"reasoning_content": text}),
            None,
            None,
            &[],
        )),
        Outbound::Content(text) => {
            data_frame(&chat_chunk(meta, json!({"content": text}), None, None, &[]))
        }
        Outbound::Text(text) => data_frame(&completion_chunk(meta, &text, None, None, &[])),
    }
}

fn completion_body(meta: &Meta, outcome: &TurnOutcome) -> Value {
    let mut payload = if meta.chat {
        let (reasoning, content) = split_nonstream(&outcome.raw);
        let mut message = Map::new();
        message.insert("role".into(), Value::String("assistant".into()));
        message.insert(
            "content".into(),
            if content.is_empty() {
                Value::Null
            } else {
                Value::String(content)
            },
        );
        message.insert(
            "reasoning_content".into(),
            match reasoning {
                Some(text) => Value::String(text),
                None => Value::Null,
            },
        );
        json!({
            "id": meta.id,
            "object": "chat.completion",
            "created": meta.created,
            "model": meta.model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason(outcome),
            }],
            "usage": usage_json(outcome.usage),
        })
    } else {
        json!({
            "id": meta.id,
            "object": "text_completion",
            "created": meta.created,
            "model": meta.model,
            "choices": [{
                "index": 0,
                "text": outcome.raw,
                "finish_reason": finish_reason(outcome),
            }],
            "usage": usage_json(outcome.usage),
        })
    };
    if !outcome.triggers.is_empty() {
        payload["repetition_check"] = json!({"triggers": outcome.triggers});
    }
    payload
}

fn chat_chunk(
    meta: &Meta,
    delta: Value,
    finish: Option<&str>,
    usage: Option<Value>,
    triggers: &[String],
) -> Value {
    let mut payload = json!({
        "id": meta.id,
        "object": "chat.completion.chunk",
        "created": meta.created,
        "model": meta.model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish,
        }],
    });
    if let Some(usage) = usage {
        payload["usage"] = usage;
    }
    if !triggers.is_empty() {
        payload["repetition_check"] = json!({"triggers": triggers});
    }
    payload
}

fn completion_chunk(
    meta: &Meta,
    text: &str,
    finish: Option<&str>,
    usage: Option<Value>,
    triggers: &[String],
) -> Value {
    let mut payload = json!({
        "id": meta.id,
        "object": "text_completion",
        "created": meta.created,
        "model": meta.model,
        "choices": [{
            "index": 0,
            "text": text,
            "finish_reason": finish,
        }],
    });
    if let Some(usage) = usage {
        payload["usage"] = usage;
    }
    if !triggers.is_empty() {
        payload["repetition_check"] = json!({"triggers": triggers});
    }
    payload
}

fn finish_reason(outcome: &TurnOutcome) -> &'static str {
    if outcome.stop_string {
        return "stop";
    }
    match outcome.finish {
        Finish::Length | Finish::ContextFull => "length",
        Finish::Stop | Finish::Cancelled => "stop",
    }
}

fn usage_json(usage: Usage) -> Value {
    json!({
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.prompt_tokens.saturating_add(usage.completion_tokens),
    })
}

fn data_frame(value: &Value) -> String {
    format!(
        "data: {}\n\n",
        serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string())
    )
}

fn prepare_chat(
    state: &AppState,
    headers: &HeaderMap,
    body: &Map<String, Value>,
) -> std::result::Result<Prepared, String> {
    let kwargs = template_kwargs(body);
    let tools = request_tools(body);
    let messages = match body.get("messages") {
        None => &[][..],
        Some(Value::Array(messages)) => messages.as_slice(),
        Some(_) => return Err("messages must be an array".to_string()),
    };
    let rendered = state
        .assets
        .render_chat(messages, tools.as_ref(), kwargs)
        .map_err(|err| err.to_string())?;
    let prompt = state
        .assets
        .encode(&rendered, false)
        .map_err(|err| err.to_string())?;
    let in_think = !matches!(kwargs.get("enable_thinking"), Some(Value::Bool(false)));
    let prepared = prepare_common(
        state,
        headers,
        body,
        prompt,
        state.chat_max_tokens,
        true,
        in_think,
    )?;
    Ok(prepared)
}

fn prepare_completion(
    state: &AppState,
    headers: &HeaderMap,
    body: &Map<String, Value>,
) -> std::result::Result<Prepared, String> {
    let prompt_value = body
        .get("prompt")
        .ok_or_else(|| "prompt must be a string or a token-id list".to_string())?;
    let prompt = prompt_ids(&state.assets, prompt_value)?;
    prepare_common(
        state,
        headers,
        body,
        prompt,
        state.completion_max_tokens,
        false,
        false,
    )
}

fn prepare_common(
    state: &AppState,
    headers: &HeaderMap,
    body: &Map<String, Value>,
    prompt: Vec<u32>,
    default_max: u32,
    chat: bool,
    in_think: bool,
) -> std::result::Result<Prepared, String> {
    if prompt.len() as u32 >= state.max_tokens {
        return Err(format!(
            "prompt of {} tokens exceeds the {} token cache",
            prompt.len(),
            state.max_tokens
        ));
    }
    let room = state.max_tokens - prompt.len() as u32;
    let requested = max_new_tokens(body, default_max)?;
    let max_new_tokens = requested.min(room);
    if max_new_tokens == 0 {
        return Err(format!(
            "prompt of {} tokens exceeds the {} token cache",
            prompt.len(),
            state.max_tokens
        ));
    }
    let (extra_ids, stop_strings) = classify_stops(&state.assets, body.get("stop"))?;
    let mut stop_tokens = state.assets.eos_ids().to_vec();
    for id in extra_ids {
        if !stop_tokens.contains(&id) {
            stop_tokens.push(id);
        }
    }
    let sampler = sampler_config(body)?;
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .filter(|model| !model.is_empty())
        .unwrap_or(&state.model_name)
        .to_string();
    Ok(Prepared {
        chat,
        in_think,
        skip_special: skip_special_tokens(body),
        stop_strings,
        repetition: repetition_config(headers),
        request: GenerateRequest {
            prompt,
            max_new_tokens,
            stop_tokens,
            sampler,
        },
        stream: bool_flag(body, "stream"),
        model,
    })
}

fn repetition_config(headers: &HeaderMap) -> CheckConfig {
    let pairs: Vec<(String, String)> = headers
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.as_str().to_string(), value.to_string()))
        })
        .collect();
    CheckConfig::parse_headers(
        pairs
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str())),
    )
}

fn sampler_config(body: &Map<String, Value>) -> std::result::Result<SamplerConfig, String> {
    let temperature = temperature(body)?;
    let seed = match body.get("seed") {
        None | Some(Value::Null) => 0,
        Some(Value::Number(number)) => number_u64(number).ok_or_else(|| "bad seed".to_string())?,
        Some(_) => return Err("bad seed".to_string()),
    };
    let mut sampler = SamplerConfig::serve_defaults(temperature, seed);
    sampler.rep_penalty = opt_f32(body, "repetition_penalty", sampler.rep_penalty)?;
    sampler.frequency_penalty = opt_f32(body, "frequency_penalty", sampler.frequency_penalty)?;
    sampler.presence_penalty = opt_f32(body, "presence_penalty", sampler.presence_penalty)?;
    sampler.penalty_range = opt_u32(body, "frequency_range", sampler.penalty_range)?;
    sampler.dry_multiplier = opt_f32(body, "dry_multiplier", sampler.dry_multiplier)?;
    sampler.dry_base = opt_f32(body, "dry_base", sampler.dry_base)?;
    sampler.dry_allowed_length = opt_u32(body, "dry_allowed_length", sampler.dry_allowed_length)?;
    sampler.dry_range = opt_u32(body, "dry_range", sampler.dry_range)?;
    Ok(sampler)
}

/// `int(body.get("max_tokens") or default)`: missing, null, and 0 use `default`.
fn max_new_tokens(body: &Map<String, Value>, default_max: u32) -> std::result::Result<u32, String> {
    match body.get("max_tokens") {
        None | Some(Value::Null) => Ok(default_max),
        Some(Value::String(text)) if text.is_empty() => Ok(default_max),
        Some(Value::Number(number)) => {
            let value = number_u32(number).ok_or_else(|| "bad max_tokens".to_string())?;
            if value == 0 {
                Ok(default_max)
            } else {
                Ok(value)
            }
        }
        Some(_) => Err("bad max_tokens".to_string()),
    }
}

/// `float(body.get("temperature") or 0)`.
fn temperature(body: &Map<String, Value>) -> std::result::Result<f32, String> {
    match body.get("temperature") {
        None | Some(Value::Null) => Ok(0.0),
        Some(Value::String(text)) if text.is_empty() => Ok(0.0),
        Some(Value::Number(number)) => {
            let value = number
                .as_f64()
                .ok_or_else(|| "bad temperature".to_string())?;
            if value == 0.0 {
                Ok(0.0)
            } else {
                Ok(value as f32)
            }
        }
        Some(_) => Err("bad temperature".to_string()),
    }
}

fn opt_f32(body: &Map<String, Value>, key: &str, default: f32) -> std::result::Result<f32, String> {
    match body.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::String(text)) if text.is_empty() => Ok(default),
        Some(Value::Number(number)) => number
            .as_f64()
            .map(|value| value as f32)
            .ok_or_else(|| format!("bad {key}")),
        Some(_) => Err(format!("bad {key}")),
    }
}

fn opt_u32(body: &Map<String, Value>, key: &str, default: u32) -> std::result::Result<u32, String> {
    match body.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::String(text)) if text.is_empty() => Ok(default),
        Some(Value::Number(number)) => number_u32(number).ok_or_else(|| format!("bad {key}")),
        Some(_) => Err(format!("bad {key}")),
    }
}

fn number_u32(number: &serde_json::Number) -> Option<u32> {
    if let Some(value) = number.as_u64() {
        return u32::try_from(value).ok();
    }
    let value = number.as_f64()?;
    if value.is_finite() && value >= 0.0 && value <= u32::MAX as f64 && value.fract() == 0.0 {
        Some(value as u32)
    } else {
        None
    }
}

fn number_u64(number: &serde_json::Number) -> Option<u64> {
    if let Some(value) = number.as_u64() {
        return Some(value);
    }
    let value = number.as_f64()?;
    if value.is_finite() && value >= 0.0 && value <= u64::MAX as f64 && value.fract() == 0.0 {
        Some(value as u64)
    } else {
        None
    }
}

fn classify_stops(
    assets: &Assets,
    value: Option<&Value>,
) -> std::result::Result<(Vec<u32>, Vec<String>), String> {
    let mut ids = Vec::new();
    let mut strings = Vec::new();
    let Some(value) = value else {
        return Ok((ids, strings));
    };
    match value {
        Value::Null => Ok((ids, strings)),
        Value::String(text) => {
            push_stop_string(assets, &mut ids, &mut strings, text);
            Ok((ids, strings))
        }
        Value::Array(items) => {
            for item in items {
                match item {
                    Value::String(text) => push_stop_string(assets, &mut ids, &mut strings, text),
                    Value::Number(number) => {
                        let id = number_u32(number)
                            .ok_or_else(|| "stop token id is out of range".to_string())?;
                        ids.push(id);
                    }
                    _ => return Err("stop entries must be strings or token ids".to_string()),
                }
            }
            Ok((ids, strings))
        }
        _ => Err("stop must be a string or an array".to_string()),
    }
}

fn push_stop_string(assets: &Assets, ids: &mut Vec<u32>, strings: &mut Vec<String>, text: &str) {
    if text.is_empty() {
        return;
    }
    strings.push(text.to_string());
    if let Ok(encoded) = assets.encode(text, false) {
        if encoded.len() == 1 {
            ids.push(encoded[0]);
        }
    }
}

fn prompt_ids(assets: &Assets, prompt: &Value) -> std::result::Result<Vec<u32>, String> {
    match prompt {
        Value::String(text) => assets.encode(text, false).map_err(|err| err.to_string()),
        Value::Array(items) => {
            match items.first() {
                Some(Value::Number(_)) => {
                    let mut ids = Vec::with_capacity(items.len());
                    for item in items {
                        let number = item.as_number().ok_or_else(|| {
                            "prompt must be a string or a token-id list".to_string()
                        })?;
                        ids.push(number_u32(number).ok_or_else(|| {
                            "prompt must be a string or a token-id list".to_string()
                        })?);
                    }
                    Ok(ids)
                }
                Some(Value::Array(_)) => prompt_ids(assets, &items[0]),
                _ => Err("prompt must be a string or a token-id list".to_string()),
            }
        }
        _ => Err("prompt must be a string or a token-id list".to_string()),
    }
}

fn template_kwargs(body: &Map<String, Value>) -> &Map<String, Value> {
    body.get("chat_template_kwargs")
        .and_then(Value::as_object)
        .unwrap_or_else(|| empty_map())
}

fn empty_map() -> &'static Map<String, Value> {
    use std::sync::OnceLock;
    static EMPTY: OnceLock<Map<String, Value>> = OnceLock::new();
    EMPTY.get_or_init(Map::new)
}

fn request_tools(body: &Map<String, Value>) -> Option<Value> {
    if body.get("tool_choice").and_then(Value::as_str) == Some("none") {
        return None;
    }
    let tools = body.get("tools")?;
    let present = match tools {
        Value::Null | Value::Bool(false) => false,
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
        Value::String(text) => !text.is_empty(),
        Value::Number(number) => number.as_f64().map(|value| value != 0.0).unwrap_or(true),
        Value::Bool(true) => true,
    };
    present.then(|| tools.clone())
}

fn skip_special_tokens(body: &Map<String, Value>) -> bool {
    match body.get("skip_special_tokens") {
        Some(Value::Bool(value)) => *value,
        _ => true,
    }
}

fn bool_flag(body: &Map<String, Value>, key: &str) -> bool {
    matches!(body.get(key), Some(Value::Bool(true)))
}

fn json_response(status: StatusCode, value: &Value) -> Response {
    let body = serde_json::to_vec(value).unwrap_or_else(|_| b"{}".to_vec());
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Body::from(body))
        .expect("json response")
}

fn error_response(status: StatusCode, message: &str, kind: &str) -> Response {
    json_response(
        status,
        &json!({"error": {"message": message, "type": kind}}),
    )
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn new_id(prefix: &str) -> String {
    let mut bytes = [0u8; 12];
    if let Ok(mut file) = std::fs::File::open("/dev/urandom") {
        use std::io::Read;
        let _ = file.read_exact(&mut bytes);
    } else {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        bytes.copy_from_slice(&nanos.to_le_bytes()[..12]);
    }
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut id = String::with_capacity(prefix.len() + 24);
    id.push_str(prefix);
    for byte in bytes {
        id.push(HEX[(byte >> 4) as usize] as char);
        id.push(HEX[(byte & 0xf) as usize] as char);
    }
    id
}

struct CancelFlag(Arc<AtomicBool>);

impl Drop for CancelFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

struct FlagStream<S> {
    inner: S,
    flag: Arc<AtomicBool>,
}

impl<S> Drop for FlagStream<S> {
    fn drop(&mut self) {
        self.flag.store(true, Ordering::SeqCst);
    }
}

impl<S: Stream + Unpin> Stream for FlagStream<S> {
    type Item = S::Item;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.get_mut().inner).poll_next(cx)
    }
}
