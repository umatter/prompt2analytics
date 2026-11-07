//! T3-WP53 MCP-path tests: the one LLM tool loop over every provider path, tool results replayed
//! as history, the conversation routes, /api/llm/chat with a conversation, the Dioxus client's
//! routes, API_ENDPOINTS.md against the app, and the `database` feature. Owned by T3-WP53.
use crate::common::*;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use std::time::Duration;
use tower::ServiceExt;

/// One request (Host: localhost); `session` sets X-P2A-Session-Id. The status, the body as JSON
/// (a non-JSON body as a JSON string, an empty one as Null) and the body's length in bytes.
async fn send(
    app: &axum::Router,
    method: &str,
    uri: &str,
    session: Option<&str>,
    body: Option<Value>,
) -> (StatusCode, Value, usize) {
    let mut req = Request::builder()
        .method(method)
        .uri(uri)
        .header("host", "localhost");
    if let Some(s) = session {
        req = req.header("x-p2a-session-id", s);
    }
    let req = match body {
        Some(b) => req
            .header("content-type", "application/json")
            .body(Body::from(b.to_string()))
            .unwrap(),
        None => req.body(Body::empty()).unwrap(),
    };
    let resp = tokio::time::timeout(Duration::from_secs(30), app.clone().oneshot(req))
        .await
        .unwrap_or_else(|_| panic!("{method} {uri}: no response within 30 s"))
        .unwrap();
    let status = resp.status();
    let bytes = tokio::time::timeout(
        Duration::from_secs(30),
        axum::body::to_bytes(resp.into_body(), usize::MAX),
    )
    .await
    .unwrap_or_else(|_| panic!("{method} {uri}: no body within 30 s"))
    .unwrap();
    let value = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes)
            .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()))
    };
    (status, value, bytes.len())
}

/// axum's answer when no route has the path (404, empty body) or the method (405).
fn unregistered(status: StatusCode, len: usize) -> bool {
    status == StatusCode::METHOD_NOT_ALLOWED || (status == StatusCode::NOT_FOUND && len == 0)
}

/// Whether this build compiles the feature named in API_ENDPOINTS.md's Feature column.
fn compiled(feature: &str) -> bool {
    match feature {
        "http" => true,
        "llm" => cfg!(feature = "llm"),
        "db" => cfg!(feature = "db"),
        "websocket" => cfg!(feature = "websocket"),
        other => panic!("API_ENDPOINTS.md names an unknown feature `{other}`"),
    }
}

#[cfg(feature = "llm")]
mod providers {
    use super::*;
    use p2a_mcp::llm::{
        AnthropicProvider, LlmProvider, LlmSettings, Message, OllamaProvider, OpenAIProvider,
        ProviderConfig, ProviderType, StreamChunk, ToolExecutor,
    };
    use std::net::SocketAddr;
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Copy, Debug)]
    pub(super) enum Kind {
        Ollama,
        OpenAI,
        Anthropic,
    }
    pub(super) const KINDS: [Kind; 3] = [Kind::Ollama, Kind::OpenAI, Kind::Anthropic];

    /// One scripted model turn: tool calls (name, arguments), or a final text.
    pub(super) enum Turn {
        Calls(Vec<(String, Value)>),
        Text(String),
    }
    pub(super) fn call1(name: &str, args: Value) -> Turn {
        Turn::Calls(vec![(name.to_string(), args)])
    }
    pub(super) fn text(t: &str) -> Turn {
        Turn::Text(t.to_string())
    }

    /// The id the OpenAI and Anthropic stubs give every tool call: a provider that repeats ids.
    const DUP: &str = "dup";

    /// The provider's answer to `turn`: (content type, body). Shapes as the parsers read them.
    fn answer(kind: Kind, streaming: bool, turn: &Turn) -> (&'static str, String) {
        match (kind, streaming) {
            (Kind::Ollama, _) => {
                let message = match turn {
                    Turn::Text(s) => json!({"role": "assistant", "content": s}),
                    Turn::Calls(cs) => json!({"role": "assistant", "content": "", "tool_calls":
                        cs.iter().map(|(n, a)| json!({"function": {"name": n, "arguments": a}})).collect::<Vec<_>>()}),
                };
                (
                    "application/x-ndjson",
                    format!("{}\n", json!({"message": message, "done": true})),
                )
            }
            (Kind::OpenAI, false) => {
                let message = match turn {
                    Turn::Text(s) => json!({"role": "assistant", "content": s}),
                    Turn::Calls(cs) => json!({"role": "assistant", "content": null, "tool_calls":
                        cs.iter().map(|(n, a)| json!({"id": DUP, "type": "function",
                            "function": {"name": n, "arguments": a.to_string()}})).collect::<Vec<_>>()}),
                };
                (
                    "application/json",
                    json!({"choices": [{"message": message}]}).to_string(),
                )
            }
            (Kind::OpenAI, true) => {
                let mut out = String::new();
                match turn {
                    Turn::Text(s) => out.push_str(&format!("data: {}\n\n",
                        json!({"choices": [{"index": 0, "delta": {"content": s}, "finish_reason": null}]}))),
                    Turn::Calls(cs) => for (i, (n, a)) in cs.iter().enumerate() {
                        out.push_str(&format!("data: {}\n\n", json!({"choices": [{"index": 0, "delta": {"tool_calls":
                            [{"index": i, "id": DUP, "type": "function", "function": {"name": n, "arguments": a.to_string()}}]},
                            "finish_reason": null}]})));
                    },
                }
                out.push_str("data: [DONE]\n\n");
                ("text/event-stream", out)
            }
            (Kind::Anthropic, false) => {
                let content = match turn {
                    Turn::Text(s) => json!([{"type": "text", "text": s}]),
                    Turn::Calls(cs) => Value::Array(cs.iter().map(|(n, a)|
                        json!({"type": "tool_use", "id": DUP, "name": n, "input": a})).collect()),
                };
                (
                    "application/json",
                    json!({"content": content, "stop_reason": "end_turn"}).to_string(),
                )
            }
            (Kind::Anthropic, true) => {
                let mut events = Vec::new();
                match turn {
                    Turn::Text(s) => {
                        events.push(json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}));
                        events.push(json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": s}}));
                        events.push(json!({"type": "content_block_stop", "index": 0}));
                    }
                    Turn::Calls(cs) => {
                        for (i, (n, a)) in cs.iter().enumerate() {
                            events.push(json!({"type": "content_block_start", "index": i,
                            "content_block": {"type": "tool_use", "id": DUP, "name": n, "input": {}}}));
                            events.push(json!({"type": "content_block_delta", "index": i,
                            "delta": {"type": "input_json_delta", "partial_json": a.to_string()}}));
                            events.push(json!({"type": "content_block_stop", "index": i}));
                        }
                    }
                }
                events.push(json!({"type": "message_stop"}));
                let body = events
                    .iter()
                    .map(|e| format!("event: {}\ndata: {e}\n\n", e["type"].as_str().unwrap()))
                    .collect();
                ("text/event-stream", body)
            }
        }
    }

    /// A provider stub on loopback: answers request n with script[n] (the last repeats) and records
    /// every request body.
    pub(super) struct Stub {
        pub(super) addr: SocketAddr,
        pub(super) bodies: Arc<Mutex<Vec<Value>>>,
    }
    pub(super) async fn stub(kind: Kind, streaming: bool, script: Vec<Turn>) -> Stub {
        let path = match kind {
            Kind::Ollama => "/api/chat",
            Kind::OpenAI => "/chat/completions",
            Kind::Anthropic => "/v1/messages",
        };
        let script = Arc::new(script);
        let bodies = Arc::new(Mutex::new(Vec::new()));
        let (s, b) = (script.clone(), bodies.clone());
        let router = axum::Router::new().route(
            path,
            axum::routing::post(move |axum::Json(body): axum::Json<Value>| {
                let (s, b) = (s.clone(), b.clone());
                async move {
                    let n = {
                        let mut g = b.lock().unwrap();
                        g.push(body);
                        g.len() - 1
                    };
                    let (ct, text) = answer(kind, streaming, &s[n.min(s.len() - 1)]);
                    ([(axum::http::header::CONTENT_TYPE, ct)], text)
                }
            }),
        );
        Stub {
            addr: spawn_stub(router).await,
            bodies,
        }
    }

    /// A provider of `kind` pointed at `addr`. The IP-literal host never reaches the egress
    /// resolver (T0-WP08's llm_egress.rs uses the same seam for Ollama).
    fn provider(
        kind: Kind,
        addr: SocketAddr,
        max_tool_iterations: Option<usize>,
    ) -> Box<dyn LlmProvider> {
        let settings = LlmSettings::from_limits(&p2a_mcp::tools::limits::Limits::off(), None);
        let provider_type = match kind {
            Kind::Ollama => ProviderType::Ollama,
            Kind::OpenAI => ProviderType::OpenAI,
            Kind::Anthropic => ProviderType::Anthropic,
        };
        let config = ProviderConfig {
            provider_type,
            base_url: Some(format!("http://{addr}")),
            api_key: Some("wp53-key".into()),
            model: "stub".into(),
            max_tool_iterations,
            ..ProviderConfig::default()
        };
        match kind {
            Kind::Ollama => Box::new(OllamaProvider::with_settings(config, &settings)),
            Kind::OpenAI => Box::new(OpenAIProvider::with_settings(config, &settings)),
            Kind::Anthropic => Box::new(AnthropicProvider::with_settings(config, &settings)),
        }
    }

    /// Runs every tool, recording its name; answers "ok-<name>".
    #[derive(Default, Clone)]
    struct Recorder(Arc<Mutex<Vec<String>>>);
    #[async_trait::async_trait]
    impl ToolExecutor for Recorder {
        async fn execute(&self, name: &str, _arguments: Value) -> Result<String, String> {
            self.0.lock().unwrap().push(name.to_string());
            Ok(format!("ok-{name}"))
        }
    }

    pub(super) struct Run {
        pub(super) message: Message,
        pub(super) chunks: Vec<StreamChunk>,
        pub(super) executed: Vec<String>,
        pub(super) bodies: Vec<Value>,
    }

    /// One chat on `kind`, streaming or not, from the user message "go".
    pub(super) async fn run(
        kind: Kind,
        streaming: bool,
        script: Vec<Turn>,
        interpret: bool,
        max_iter: Option<usize>,
    ) -> Run {
        let s = stub(kind, streaming, script).await;
        let p = provider(kind, s.addr, max_iter);
        let tools = Recorder::default();
        let chunks = Arc::new(Mutex::new(Vec::new()));
        let msgs = [Message::user("go")];
        let c = chunks.clone();
        let fut = async {
            if streaming {
                p.chat_stream(
                    &msgs,
                    &[],
                    &tools,
                    interpret,
                    Box::new(move |ch| c.lock().unwrap().push(ch)),
                )
                .await
            } else {
                p.chat(&msgs, &[], &tools, interpret).await
            }
        };
        let message = tokio::time::timeout(Duration::from_secs(30), fut)
            .await
            .unwrap_or_else(|_| panic!("{kind:?} streaming={streaming}: no answer within 30 s"))
            .unwrap_or_else(|e| panic!("{kind:?} streaming={streaming}: {e}"));
        let chunks = chunks.lock().unwrap().clone();
        let executed = tools.0.lock().unwrap().clone();
        let bodies = s.bodies.lock().unwrap().clone();
        Run {
            message,
            chunks,
            executed,
            bodies,
        }
    }

    /// (request part, n) of a loop id `call_<12 lowercase hex>_<n>`.
    pub(super) fn parse_id(id: &str) -> Option<(String, usize)> {
        let (req, n) = id.strip_prefix("call_")?.rsplit_once('_')?;
        let hex = req.len() == 12
            && req
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
        if !hex {
            return None;
        }
        Some((req.to_string(), n.parse().ok()?))
    }

    /// LLM-17 over /api/llm/chat (the finding's failure scenario): the interpret=false answer is
    /// sent back as history, and the provider then receives the result right after its call.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_interpret_false_message_replays_as_history() {
        let (app, _state) = http_app().await;
        let sid = new_session(&app).await;
        let s = stub(
            Kind::Ollama,
            false,
            vec![call1("list_datasets", json!({})), text("next answer")],
        )
        .await;
        let provider = json!({"provider_type": "ollama", "base_url": format!("http://{}", s.addr), "model": "stub"});
        let (status, first) = post_json(
            &app,
            "/api/llm/chat",
            json!({"session_id": sid, "message": "list", "interpret": false, "provider": provider}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{first}");
        let returned = first["data"]["message"].clone();
        let result = returned["tool_results"][0]["content"]
            .as_str()
            .expect("a tool result")
            .to_string();
        let (status, second) = post_json(
            &app,
            "/api/llm/chat",
            json!({"session_id": sid, "message": "next",
            "provider": provider, "history": [{"role": "user", "content": "list"}, returned]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{second}");
        let sent = s.bodies.lock().unwrap()[1]["messages"].clone();
        let roles: Vec<&str> = sent
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["role"].as_str().unwrap())
            .collect();
        assert_eq!(
            roles,
            ["system", "user", "assistant", "tool", "user"],
            "{sent}"
        );
        assert_eq!(
            sent[2]["tool_calls"][0]["function"]["name"],
            "list_datasets"
        );
        assert_eq!(sent[3]["content"], result);
    }
}
