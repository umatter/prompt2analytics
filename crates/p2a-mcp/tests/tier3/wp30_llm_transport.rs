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
