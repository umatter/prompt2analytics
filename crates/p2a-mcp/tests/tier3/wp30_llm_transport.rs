//! T4-WP30 MCP-path tests: the providers' stream parsers (UTF-8 across reads, NDJSON objects
//! across reads, stream errors and cut-off streams), request parameters, model lists, the chat
//! routes' error codes and SSE events, and the prompt sent once. Every provider is a stub on
//! 127.0.0.1 inside the test process. Owned by T4-WP30.
#![cfg(feature = "llm")]

use crate::common::*;
use axum::body::Body;
use futures::StreamExt;
use p2a_mcp::llm::{
    AnthropicProvider, LlmError, LlmProvider, LlmSettings, Message, OllamaProvider, OpenAIProvider,
    ProviderConfig, ProviderType, StreamChunk, ToolExecutor,
};
use serde_json::{Value, json};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// The request bodies a stub received, in order.
type Bodies = Arc<Mutex<Vec<Value>>>;

/// A provider stub on loopback that answers request n on `path` with `answers[n]` (the last
/// repeats): its byte chunks, 10 ms apart, so each arrives as a read of its own. Records every
/// request body.
async fn chunked_stub(path: &'static str, answers: Vec<Vec<Vec<u8>>>) -> (SocketAddr, Bodies) {
    let answers = Arc::new(answers);
    let bodies = Bodies::default();
    let (a, b) = (answers.clone(), bodies.clone());
    let router = axum::Router::new().route(
        path,
        axum::routing::post(move |axum::Json(body): axum::Json<Value>| {
            let (a, b) = (a.clone(), b.clone());
            async move {
                let n = {
                    let mut g = b.lock().unwrap();
                    g.push(body);
                    g.len() - 1
                };
                let chunks = a[n.min(a.len() - 1)].clone();
                let stream = futures::stream::iter(chunks).then(|chunk| async move {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    Ok::<_, std::convert::Infallible>(chunk)
                });
                (
                    [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
                    Body::from_stream(stream),
                )
            }
        }),
    );
    (spawn_stub(router).await, bodies)
}

/// `bytes` cut after every byte `at`, so each such byte ends a read.
fn cut_after(bytes: &[u8], at: u8) -> Vec<Vec<u8>> {
    let mut out = vec![Vec::new()];
    for &b in bytes {
        out.last_mut().unwrap().push(b);
        if b == at {
            out.push(Vec::new());
        }
    }
    out.retain(|c| !c.is_empty());
    out
}

/// A provider of `t` on the stub at `addr`. The IP-literal host never reaches the egress
/// resolver (as in `wp53_llm_routes.rs`).
fn provider(t: ProviderType, addr: SocketAddr) -> Box<dyn LlmProvider> {
    let settings = LlmSettings::from_limits(&p2a_mcp::tools::limits::Limits::off(), None);
    let config = ProviderConfig {
        provider_type: t,
        base_url: Some(format!("http://{addr}")),
        api_key: Some("wp30-key".into()),
        model: "stub".into(),
        ..ProviderConfig::default()
    };
    match t {
        ProviderType::Ollama => Box::new(OllamaProvider::with_settings(config, &settings)),
        ProviderType::OpenAI => Box::new(OpenAIProvider::with_settings(config, &settings)),
        ProviderType::Anthropic => Box::new(AnthropicProvider::with_settings(config, &settings)),
    }
}

/// Runs every tool, recording its name and arguments; answers "ok-<name>".
#[derive(Default, Clone)]
struct Recorder(Arc<Mutex<Vec<(String, Value)>>>);
#[async_trait::async_trait]
impl ToolExecutor for Recorder {
    async fn execute(&self, name: &str, arguments: Value) -> Result<String, String> {
        self.0.lock().unwrap().push((name.to_string(), arguments));
        Ok(format!("ok-{name}"))
    }
}

/// One streamed chat from "go": the reply, and the text of every `Text` chunk in order.
async fn stream_chat(
    p: &dyn LlmProvider,
    tools: &Recorder,
    interpret: bool,
) -> (Result<Message, LlmError>, Vec<String>) {
    let texts = Arc::new(Mutex::new(Vec::new()));
    let t = texts.clone();
    let reply = tokio::time::timeout(
        Duration::from_secs(30),
        p.chat_stream(
            &[Message::user("go")],
            &[],
            tools,
            interpret,
            Box::new(move |chunk| {
                if let StreamChunk::Text { content } = chunk {
                    t.lock().unwrap().push(content);
                }
            }),
        ),
    )
    .await
    .expect("no answer within 30 s");
    let texts = texts.lock().unwrap().clone();
    (reply, texts)
}

// ---------- Anthropic and OpenAI stream bodies ----------

/// Anthropic's server-sent events: `event: <type>` and `data: <json>` per event.
fn anthropic_sse(events: &[Value]) -> String {
    events
        .iter()
        .map(|e| format!("event: {}\ndata: {e}\n\n", e["type"].as_str().unwrap()))
        .collect()
}

/// Anthropic's events for a text answer, up to and including `message_stop` when `complete`.
fn anthropic_text(text: &str, complete: bool) -> Vec<Value> {
    let mut events = vec![
        json!({"type": "message_start", "message": {"id": "m", "role": "assistant", "content": []}}),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        json!({"type": "ping"}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}}),
    ];
    if complete {
        events.push(json!({"type": "message_stop"}));
    }
    events
}

/// Anthropic's events for one tool call whose input arrives as `input` (JSON text).
fn anthropic_tool(name: &str, input: &str) -> Vec<Value> {
    vec![
        json!({"type": "message_start", "message": {"id": "m", "role": "assistant", "content": []}}),
        json!({"type": "content_block_start", "index": 0,
               "content_block": {"type": "tool_use", "id": "t1", "name": name, "input": {}}}),
        json!({"type": "content_block_delta", "index": 0,
               "delta": {"type": "input_json_delta", "partial_json": input}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}}),
        json!({"type": "message_stop"}),
    ]
}

/// OpenAI's server-sent events: one `data:` line per chunk, then `data: [DONE]` when `done`.
fn openai_sse(chunks: &[Value], done: bool) -> String {
    let mut out: String = chunks.iter().map(|c| format!("data: {c}\n\n")).collect();
    if done {
        out.push_str("data: [DONE]\n\n");
    }
    out
}

/// OpenAI's chunks for a text answer, the last with its finish reason when `finish`.
fn openai_text(text: &str, finish: bool) -> Vec<Value> {
    let mut chunks = vec![
        json!({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]}),
        json!({"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]}),
    ];
    if finish {
        chunks.push(json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}));
    }
    chunks
}

/// OpenAI's chunks for one tool call whose arguments arrive as `arguments`.
fn openai_tool(name: &str, arguments: &str) -> Vec<Value> {
    vec![
        json!({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "c1", "type": "function",
               "function": {"name": name, "arguments": ""}}]}, "finish_reason": null}]}),
        json!({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0,
               "function": {"arguments": arguments}}]}, "finish_reason": null}]}),
        json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
    ]
}

// ---------- LLM-8: a character split across two reads ----------

/// The text delta "größe" and the tool arguments `{"y":"größe"}`, each cut inside `ö` (after its
/// first byte, 0xC3), reach the answer, every text chunk and the tool call whole, on Anthropic's
/// and OpenAI's streams.
#[tokio::test]
async fn a_character_split_across_reads_reaches_the_answer_whole() {
    let word = "größe";
    let arguments = json!({"y": word}).to_string();
    let bodies = [
        (
            ProviderType::Anthropic,
            "/v1/messages",
            anthropic_sse(&anthropic_text(word, true)),
            anthropic_sse(&anthropic_tool("describe_dataset", &arguments)),
        ),
        (
            ProviderType::OpenAI,
            "/chat/completions",
            openai_sse(&openai_text(word, true), true),
            openai_sse(&openai_tool("describe_dataset", &arguments), true),
        ),
    ];
    for (t, path, text, tool) in bodies {
        let (addr, _) = chunked_stub(path, vec![cut_after(text.as_bytes(), 0xC3)]).await;
        let (reply, texts) =
            stream_chat(provider(t, addr).as_ref(), &Recorder::default(), true).await;
        let reply = reply.unwrap_or_else(|e| panic!("{t}: {e}"));
        assert_eq!(reply.content, word, "{t}");
        assert_eq!(texts.concat(), word, "{t}");

        let (addr, _) = chunked_stub(path, vec![cut_after(tool.as_bytes(), 0xC3)]).await;
        let tools = Recorder::default();
        let (reply, _) = stream_chat(provider(t, addr).as_ref(), &tools, false).await;
        reply.unwrap_or_else(|e| panic!("{t}: {e}"));
        assert_eq!(
            *tools.0.lock().unwrap(),
            [("describe_dataset".to_string(), json!({"y": word}))],
            "{t}"
        );
    }
}

// ---------- LLM-11: Ollama objects across reads ----------

/// Ollama's NDJSON: an object cut across two reads is kept (text and tool calls), the tool calls of
/// every object are kept in order, and the last object needs no line break.
#[tokio::test]
async fn ollama_keeps_objects_that_span_reads_and_every_tool_call() {
    let first = format!(
        "{}\n",
        json!({"message": {"role": "assistant", "content": "HELLO-PART "}, "done": false})
    );
    let second = format!(
        "{}\n",
        json!({"message": {"role": "assistant", "content": "WORLD-PART"}, "done": false})
    );
    let done = json!({"message": {"role": "assistant", "content": ""}, "done": true, "done_reason": "stop"})
        .to_string();
    let (head, tail) = second.as_bytes().split_at(20);
    let chunks = vec![
        first.into_bytes(),
        head.to_vec(),
        tail.to_vec(),
        done.clone().into_bytes(),
    ];
    let (addr, _) = chunked_stub("/api/chat", vec![chunks]).await;
    let (reply, texts) = stream_chat(
        provider(ProviderType::Ollama, addr).as_ref(),
        &Recorder::default(),
        true,
    )
    .await;
    let reply = reply.unwrap();
    assert_eq!(reply.content, "HELLO-PART WORLD-PART");
    assert_eq!(texts.concat(), "HELLO-PART WORLD-PART");

    let call = |name: &str| {
        format!(
            "{}\n",
            json!({"message": {"role": "assistant", "content": "",
                   "tool_calls": [{"function": {"name": name, "arguments": {"k": name}}}]}, "done": false})
        )
        .into_bytes()
    };
    let (addr, _) = chunked_stub(
        "/api/chat",
        vec![vec![
            call("first_tool"),
            call("second_tool"),
            done.into_bytes(),
        ]],
    )
    .await;
    let tools = Recorder::default();
    let (reply, _) =
        stream_chat(provider(ProviderType::Ollama, addr).as_ref(), &tools, false).await;
    reply.unwrap();
    assert_eq!(
        *tools.0.lock().unwrap(),
        [
            ("first_tool".to_string(), json!({"k": "first_tool"})),
            ("second_tool".to_string(), json!({"k": "second_tool"}))
        ]
    );
}

// ---------- LLM-18: stream errors and cut-off streams ----------

/// Whether `reply` failed with the variant whose debug text starts with `want` (the variant name
/// and its fields), and its message carries none of the provider's own text.
fn failed_with(t: ProviderType, reply: Result<Message, LlmError>, want: &str) {
    let e = match reply {
        Ok(m) => panic!(
            "{t}: answered {:?} instead of failing with {want}",
            m.content
        ),
        Err(e) => e,
    };
    let debug = format!("{e:?}");
    assert!(debug.starts_with(want), "{t}: {debug}");
    let text = e.to_string();
    assert!(
        !text.contains("Overloaded") && !text.contains("STUB-SECRET"),
        "{t}: {text}"
    );
}

/// A provider's error event in the middle of an answer fails the call with the provider's error
/// type (`unknown` when it has none of the expected shape), never with its message text.
#[tokio::test]
async fn a_stream_error_event_fails_the_call() {
    let mut anthropic = anthropic_sse(&anthropic_text("half a sen", false)[..4]);
    anthropic.push_str(&format!(
        "event: error\ndata: {}\n\n",
        json!({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
    ));
    let mut openai = openai_sse(&openai_text("half a sen", false), false);
    openai.push_str(&format!(
        "data: {}\n\n",
        json!({"error": {"type": "server_error", "message": "STUB-SECRET"}})
    ));
    let ollama = format!(
        "{}\n{}\n",
        json!({"message": {"role": "assistant", "content": "half a sen"}, "done": false}),
        json!({"error": "STUB-SECRET went wrong"})
    );
    for (t, path, body, want) in [
        (
            ProviderType::Anthropic,
            "/v1/messages",
            anthropic,
            r#"StreamFailed { provider: "Anthropic", kind: "overloaded_error" }"#,
        ),
        (
            ProviderType::OpenAI,
            "/chat/completions",
            openai,
            r#"StreamFailed { provider: "OpenAI", kind: "server_error" }"#,
        ),
        (
            ProviderType::Ollama,
            "/api/chat",
            ollama,
            r#"StreamFailed { provider: "Ollama", kind: "unknown" }"#,
        ),
    ] {
        let (addr, _) = chunked_stub(path, vec![vec![body.into_bytes()]]).await;
        let (reply, _) = stream_chat(provider(t, addr).as_ref(), &Recorder::default(), true).await;
        failed_with(t, reply, want);
    }
}

/// A stream that ends without its end marker (Anthropic's `message_stop`, OpenAI's `[DONE]` or a
/// finish reason, Ollama's `done: true`) fails as incomplete instead of returning the half answer.
/// OpenAI's finish reason alone, without `[DONE]`, is a complete answer.
#[tokio::test]
async fn a_cut_off_stream_fails_as_incomplete() {
    let ollama = format!(
        "{}\n",
        json!({"message": {"role": "assistant", "content": "half a sen"}, "done": false})
    );
    for (t, path, body) in [
        (
            ProviderType::Anthropic,
            "/v1/messages",
            anthropic_sse(&anthropic_text("half a sen", false)),
        ),
        (
            ProviderType::OpenAI,
            "/chat/completions",
            openai_sse(&openai_text("half a sen", false), false),
        ),
        (ProviderType::Ollama, "/api/chat", ollama),
    ] {
        let (addr, _) = chunked_stub(path, vec![vec![body.into_bytes()]]).await;
        let (reply, _) = stream_chat(provider(t, addr).as_ref(), &Recorder::default(), true).await;
        failed_with(t, reply, "StreamIncomplete {");
    }
    let finished = openai_sse(&openai_text("a whole sentence.", true), false);
    let (addr, _) = chunked_stub("/chat/completions", vec![vec![finished.into_bytes()]]).await;
    let (reply, _) = stream_chat(
        provider(ProviderType::OpenAI, addr).as_ref(),
        &Recorder::default(),
        true,
    )
    .await;
    assert_eq!(reply.unwrap().content, "a whole sentence.");
}

// ---------- the chat routes over HTTP ----------

/// `POST /api/llm/chat/stream` with `body` on `app`: every SSE event's JSON, in order. The stream
/// ends at its `done` or `error` event.
async fn stream_events(app: &axum::Router, body: Value) -> Vec<Value> {
    use tower::ServiceExt;
    let req = axum::http::Request::post("/api/llm/chat/stream")
        .header(axum::http::header::HOST, "localhost")
        .header(axum::http::header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let read = async {
        let resp = app.clone().oneshot(req).await.unwrap();
        axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap()
    };
    let bytes = tokio::time::timeout(Duration::from_secs(30), read)
        .await
        .expect("no end of the stream within 30 s");
    String::from_utf8(bytes.to_vec())
        .unwrap()
        .lines()
        .filter_map(|l| l.strip_prefix("data: ").or_else(|| l.strip_prefix("data:")))
        .map(|d| serde_json::from_str(d).unwrap())
        .collect()
}

/// The provider settings of a chat request to the stub of `t` at `addr`.
fn provider_json(t: ProviderType, addr: SocketAddr) -> Value {
    json!({
        "provider_type": t.to_string().to_lowercase(),
        "base_url": format!("http://{addr}"),
        "api_key": "wp30-key",
        "model": "stub",
    })
}

/// The last event of a stream, which must be an `error`: its message and code.
fn stream_error(events: &[Value]) -> (String, String) {
    let last = events.last().expect("no events");
    assert_eq!(last["type"], "error", "{events:?}");
    (
        last["error"].as_str().unwrap_or_default().to_string(),
        last["code"].as_str().unwrap_or_default().to_string(),
    )
}

/// LLM-18 through the stream route: an Ollama error object and a cut-off Ollama stream end the
/// stream with an `error` event coded `upstream_error`, never with a `done` carrying half an answer.
#[tokio::test]
async fn the_stream_route_reports_a_failed_or_cut_off_provider_stream() {
    let (app, _) = http_app().await;
    let half = json!({"message": {"role": "assistant", "content": "half a sen"}, "done": false});
    for (body, want) in [
        (
            format!("{half}\n{}\n", json!({"error": "STUB-SECRET went wrong"})),
            "Ollama reported an error during its answer (unknown); try again",
        ),
        (
            format!("{half}\n"),
            "Ollama's answer ended before it was complete; try again",
        ),
    ] {
        let (addr, _) = chunked_stub("/api/chat", vec![vec![body.into_bytes()]]).await;
        let sid = new_session(&app).await;
        let events = stream_events(
            &app,
            json!({"session_id": sid, "message": "go",
                   "provider": provider_json(ProviderType::Ollama, addr)}),
        )
        .await;
        assert!(!events.iter().any(|e| e["type"] == "done"), "{events:?}");
        let (error, code) = stream_error(&events);
        assert_eq!((error.as_str(), code.as_str()), (want, "upstream_error"));
    }
}

// ---------- LLM-21: an OpenAI-compatible endpoint keeps its fields ----------

/// A non-default OpenAI endpoint gets `max_tokens` and the configured temperature for any model,
/// streaming or not; the reasoning-model rule of OpenAI's own endpoint is a unit test of the
/// request builder, since no test may reach that endpoint.
#[tokio::test]
async fn an_openai_compatible_endpoint_gets_max_tokens_and_the_temperature() {
    let settings = LlmSettings::from_limits(&p2a_mcp::tools::limits::Limits::off(), None);
    let plain = openai_sse(&openai_text("ok", true), true);
    for streaming in [false, true] {
        let answer = if streaming {
            plain.clone().into_bytes()
        } else {
            json!({"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"}]})
            .to_string()
            .into_bytes()
        };
        let (addr, bodies) = chunked_stub("/chat/completions", vec![vec![answer]]).await;
        let config = ProviderConfig {
            provider_type: ProviderType::OpenAI,
            base_url: Some(format!("http://{addr}")),
            api_key: Some("wp30-key".into()),
            model: "o4-mini".into(),
            temperature: Some(0.7),
            max_tokens: Some(4096),
            ..ProviderConfig::default()
        };
        let p = OpenAIProvider::with_settings(config, &settings);
        if streaming {
            let (reply, _) = stream_chat(&p, &Recorder::default(), true).await;
            reply.unwrap();
        } else {
            p.chat(&[Message::user("go")], &[], &Recorder::default(), true)
                .await
                .unwrap();
        }
        let body = &bodies.lock().unwrap()[0];
        assert_eq!(body["max_tokens"], 4096, "streaming={streaming}: {body}");
        assert!(body.get("max_completion_tokens").is_none(), "{body}");
        // 0.7 as an f32 on the wire.
        let t = body["temperature"].as_f64().unwrap();
        assert!((t - 0.7).abs() < 1e-6, "{body}");
    }
}

// ---------- LLM-19, LLM-25: model lists ----------

/// A stub that answers `GET path` with `body` and records each request's headers.
async fn models_stub(
    path: &'static str,
    body: Value,
) -> (SocketAddr, Arc<Mutex<Vec<axum::http::HeaderMap>>>) {
    let seen: Arc<Mutex<Vec<axum::http::HeaderMap>>> = Arc::default();
    let s = seen.clone();
    let router = axum::Router::new().route(
        path,
        axum::routing::get(move |headers: axum::http::HeaderMap| {
            let (s, body) = (s.clone(), body.clone());
            async move {
                s.lock().unwrap().push(headers);
                axum::Json(body)
            }
        }),
    );
    (spawn_stub(router).await, seen)
}

/// Anthropic's list comes from its models endpoint with the caller's key; an OpenAI-compatible
/// endpoint's list keeps every id (the default endpoint's chat filter is a unit test).
#[tokio::test]
async fn the_model_lists_come_from_the_providers() {
    let (addr, seen) = models_stub(
        "/v1/models",
        json!({"data": [{"id": "claude-opus-4-1", "type": "model"},
                        {"id": "claude-haiku-4-5", "type": "model"}], "has_more": false}),
    )
    .await;
    let models = provider(ProviderType::Anthropic, addr)
        .list_models()
        .await
        .unwrap();
    assert_eq!(models, ["claude-opus-4-1", "claude-haiku-4-5"]);
    let headers = &seen.lock().unwrap()[0];
    assert_eq!(headers["x-api-key"], "wp30-key");
    assert_eq!(headers["anthropic-version"], "2023-06-01");

    let ids = [
        "gpt-4o",
        "o4-mini",
        "text-embedding-3-small",
        "whisper-1",
        "tts-1",
        "dall-e-3",
        "meta-llama/llama-3.1-8b-instruct",
    ];
    let data: Vec<Value> = ids
        .iter()
        .map(|id| json!({"id": id, "object": "model"}))
        .collect();
    let (addr, _) = models_stub("/models", json!({"object": "list", "data": data})).await;
    let models = provider(ProviderType::OpenAI, addr)
        .list_models()
        .await
        .unwrap();
    assert_eq!(models, ids);
}

/// `POST /api/llm/models` lists the caller's provider (here an Ollama stub), and refuses a
/// provider the egress policy refuses with 400 `invalid_provider`. The route is registered at the
/// integration; this router has only it, on the app's state.
#[tokio::test]
async fn the_models_route_lists_the_callers_provider() {
    let (_, state) = http_app().await;
    let app = axum::Router::new()
        .route(
            "/api/llm/models",
            axum::routing::post(p2a_mcp::transport::llm_routes::llm_list_models_for),
        )
        .with_state(state);
    let (addr, _) = models_stub(
        "/api/tags",
        json!({"models": [{"name": "llama3.2:3b"}, {"name": "qwen2.5:7b"}]}),
    )
    .await;
    let (status, body) = post_json(
        &app,
        "/api/llm/models",
        json!({"provider": provider_json(ProviderType::Ollama, addr)}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["data"]["provider"], "ollama", "{body}");
    assert_eq!(body["data"]["models"], json!(["llama3.2:3b", "qwen2.5:7b"]));

    let (status, body) = post_json(
        &app,
        "/api/llm/models",
        json!({"provider": {"provider_type": "ollama", "base_url": "ftp://127.0.0.1:1", "model": "m"}}),
    )
    .await;
    assert_eq!(
        (status.as_u16(), body["code"].as_str()),
        (400, Some("invalid_provider")),
        "{body}"
    );
}

// ---------- error codes ----------

/// A refused provider is 400 `invalid_provider` on `/api/llm/chat` and the stream's error code; a
/// provider's failure is 502 `upstream_error`; a stream on an unknown session ends with
/// `not_found`.
#[tokio::test]
async fn the_llm_routes_give_every_refusal_a_code() {
    let (app, _) = http_app().await;
    let sid = new_session(&app).await;
    let refused = json!({"provider_type": "ollama", "base_url": "ftp://127.0.0.1:1", "model": "m"});
    let (status, body) = post_json(
        &app,
        "/api/llm/chat",
        json!({"session_id": sid, "message": "go", "provider": refused}),
    )
    .await;
    assert_eq!(
        (status.as_u16(), body["code"].as_str()),
        (400, Some("invalid_provider")),
        "{body}"
    );
    let events = stream_events(
        &app,
        json!({"session_id": sid, "message": "go", "provider": refused}),
    )
    .await;
    assert_eq!(stream_error(&events).1, "invalid_provider");

    // A provider that answers 500: the caller's 502.
    let failing = axum::Router::new().route(
        "/api/chat",
        axum::routing::post(|| async {
            (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "STUB-SECRET")
        }),
    );
    let addr = spawn_stub(failing).await;
    let (status, body) = post_json(
        &app,
        "/api/llm/chat",
        json!({"session_id": sid, "message": "go",
               "provider": provider_json(ProviderType::Ollama, addr)}),
    )
    .await;
    assert_eq!(
        (status.as_u16(), body["code"].as_str()),
        (502, Some("upstream_error")),
        "{body}"
    );
    assert!(!body.to_string().contains("STUB-SECRET"), "{body}");
    let events = stream_events(
        &app,
        json!({"session_id": sid, "message": "go",
               "provider": provider_json(ProviderType::Ollama, addr)}),
    )
    .await;
    assert_eq!(stream_error(&events).1, "upstream_error");

    let events = stream_events(
        &app,
        json!({"session_id": "no-such-session", "message": "go",
               "provider": provider_json(ProviderType::Ollama, addr)}),
    )
    .await;
    assert_eq!(stream_error(&events).1, "not_found");
    // the non-streaming route's start-of-request lookup answers with the session routes' code
    let (status, body) = post_json(
        &app,
        "/api/llm/chat",
        json!({"session_id": "no-such-session", "message": "go",
               "provider": provider_json(ProviderType::Ollama, addr)}),
    )
    .await;
    assert_eq!(
        (status.as_u16(), body["code"].as_str()),
        (404, Some("not_found")),
        "{body}"
    );
}

// ---------- tool errors ----------

/// A tool that answers with an error result (a dataset that does not exist) ends with
/// `tool_end.status` "error" and reaches the model as an error (the next request's tool message
/// starts "Error: "); a successful call has "success". The provider is an Ollama stub, the one
/// provider the chat routes take over plain HTTP.
#[tokio::test]
async fn a_tools_error_result_is_an_error_to_the_model_and_the_client() {
    let (app, _) = http_app().await;
    let sid = new_session(&app).await;
    for (tool, args, status) in [
        (
            "describe_dataset",
            json!({"dataset": "no_such_dataset"}),
            "error",
        ),
        ("list_datasets", json!({}), "success"),
    ] {
        let call = format!(
            "{}\n",
            json!({"message": {"role": "assistant", "content": "",
                   "tool_calls": [{"function": {"name": tool, "arguments": args}}]}, "done": true})
        );
        let done = format!(
            "{}\n",
            json!({"message": {"role": "assistant", "content": "done"}, "done": true})
        );
        let (addr, bodies) = chunked_stub(
            "/api/chat",
            vec![vec![call.into_bytes()], vec![done.into_bytes()]],
        )
        .await;
        let events = stream_events(
            &app,
            json!({"session_id": sid, "message": "go",
                   "provider": provider_json(ProviderType::Ollama, addr)}),
        )
        .await;
        let end = events
            .iter()
            .find(|e| e["type"] == "tool_end")
            .unwrap_or_else(|| panic!("{tool}: no tool_end in {events:?}"));
        assert_eq!(end["status"], status, "{tool}: {end}");
        assert_eq!(events.last().unwrap()["type"], "done", "{events:?}");
        let bodies = bodies.lock().unwrap();
        assert_eq!(bodies.len(), 2, "{tool}");
        let tool_message = bodies[1]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["role"] == "tool")
            .unwrap_or_else(|| panic!("{tool}: no tool message in {}", bodies[1]))
            .clone();
        let content = tool_message["content"].as_str().unwrap();
        assert_eq!(
            content.starts_with("Error: "),
            status == "error",
            "{tool}: {content}"
        );
    }
}
