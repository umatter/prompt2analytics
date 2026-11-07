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

    fn tool_call_ids(chunks: &[StreamChunk]) -> Vec<String> {
        chunks
            .iter()
            .filter_map(|c| match c {
                StreamChunk::ToolCall { tool_call } => Some(tool_call.id.clone()),
                _ => None,
            })
            .collect()
    }
    fn tool_result_ids(chunks: &[StreamChunk]) -> Vec<String> {
        chunks
            .iter()
            .filter_map(|c| match c {
                StreamChunk::ToolResult { tool_result } => Some(tool_result.tool_call_id.clone()),
                _ => None,
            })
            .collect()
    }
    /// The call ids and result ids a later request replays (OpenAI tool_calls / tool messages,
    /// Anthropic tool_use / tool_result blocks; Ollama sends no ids).
    fn replayed_ids(kind: Kind, body: &Value) -> (Vec<String>, Vec<String>) {
        let msgs = body["messages"].as_array().cloned().unwrap_or_default();
        let s = |v: &Value| v.as_str().unwrap().to_string();
        match kind {
            Kind::OpenAI => (
                msgs.iter()
                    .filter(|m| m["role"] == "assistant")
                    .flat_map(|m| m["tool_calls"].as_array().cloned().unwrap_or_default())
                    .map(|c| s(&c["id"]))
                    .collect(),
                msgs.iter()
                    .filter(|m| m["role"] == "tool")
                    .map(|m| s(&m["tool_call_id"]))
                    .collect(),
            ),
            Kind::Anthropic => {
                let blocks = |role: &str, ty: &str, key: &str| -> Vec<String> {
                    msgs.iter()
                        .filter(|m| m["role"] == role)
                        .flat_map(|m| m["content"].as_array().cloned().unwrap_or_default())
                        .filter(|b| b["type"] == ty)
                        .map(|b| s(&b[key]))
                        .collect()
                };
                (
                    blocks("assistant", "tool_use", "id"),
                    blocks("user", "tool_result", "tool_use_id"),
                )
            }
            Kind::Ollama => (Vec::new(), Vec::new()),
        }
    }

    /// LLM-20: every provider path numbers its tool calls call_<request>_<n>, unique within the
    /// request, whatever the provider sent (the OpenAI/Anthropic stubs repeat "dup"; Ollama sends
    /// none and was numbered call_0 on every iteration).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_every_provider_path_assigns_request_unique_tool_ids() {
        for kind in KINDS {
            for streaming in [false, true] {
                let tag = format!("{kind:?} streaming={streaming}");
                let r = run(
                    kind,
                    streaming,
                    vec![call1("t1", json!({})), call1("t2", json!({})), text("done")],
                    true,
                    None,
                )
                .await;
                assert_eq!(r.message.content, "done", "{tag}");
                assert_eq!(r.executed, ["t1", "t2"], "{tag}");
                let (calls, results) = if streaming {
                    (tool_call_ids(&r.chunks), tool_result_ids(&r.chunks))
                } else {
                    replayed_ids(kind, &r.bodies[2])
                };
                if !streaming && matches!(kind, Kind::Ollama) {
                    // Ollama's wire carries no ids; test_tool_ids_differ_between_requests covers this path.
                    assert!(calls.is_empty() && results.is_empty(), "{tag}");
                    continue;
                }
                let parsed: Vec<_> = calls.iter().map(|id| parse_id(id)).collect();
                assert!(
                    parsed.iter().all(Option::is_some),
                    "{tag}: ids {calls:?} are not call_<request>_<n>"
                );
                let parsed: Vec<_> = parsed.into_iter().flatten().collect();
                assert_eq!(
                    parsed.iter().map(|p| p.1).collect::<Vec<_>>(),
                    [1, 2],
                    "{tag}: {calls:?}"
                );
                assert_eq!(
                    parsed[0].0, parsed[1].0,
                    "{tag}: one request part per request"
                );
                assert_eq!(results, calls, "{tag}: each result carries its call's id");
            }
        }
    }

    /// LLM-20: a request gets its own id space, so ids never repeat within a conversation.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_tool_ids_differ_between_requests() {
        let mut parts = Vec::new();
        for _ in 0..2 {
            let r = run(
                Kind::Ollama,
                false,
                vec![call1("t1", json!({}))],
                false,
                None,
            )
            .await;
            assert_eq!(r.executed, ["t1"]);
            let id = r
                .message
                .tool_calls
                .as_ref()
                .expect("interpret=false returns the calls")[0]
                .id
                .clone();
            assert_eq!(
                r.message.tool_results.as_ref().expect("and their results")[0].tool_call_id,
                id
            );
            let (req, n) =
                parse_id(&id).unwrap_or_else(|| panic!("{id} is not call_<request>_<n>"));
            assert_eq!(n, 1);
            parts.push(req);
        }
        assert_ne!(parts[0], parts[1]);
    }

    /// LLM-20: a tool call is announced on the stream only when it runs. The second, exact repeat is
    /// refused before it runs (REPEATED_CALLS_NOTE); the Ollama transport used to announce it anyway.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_tool_calls_are_announced_only_when_executed() {
        for kind in KINDS {
            let r = run(
                kind,
                true,
                vec![call1("t", json!({"a": 1})), call1("t", json!({"a": 1}))],
                true,
                None,
            )
            .await;
            let announced = r
                .chunks
                .iter()
                .filter(|c| matches!(c, StreamChunk::ToolCall { .. }))
                .count();
            assert_eq!(r.bodies.len(), 2, "{kind:?}");
            assert_eq!(r.executed, ["t"], "{kind:?}");
            assert_eq!(
                announced,
                r.executed.len(),
                "{kind:?}: {announced} ToolCall chunks for {} executed calls",
                r.executed.len()
            );
            assert_eq!(
                r.message.content,
                "Analysis detected a repeated tool call pattern and stopped. Please try rephrasing your request.",
                "{kind:?}"
            );
        }
    }

    /// Guard: all six paths stop at the iteration cap (3 model calls, 3 tool runs, the cap note).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_every_provider_path_stops_at_the_iteration_cap() {
        for kind in KINDS {
            for streaming in [false, true] {
                let script = (1..=6)
                    .map(|n| call1(&format!("t{n}"), json!({})))
                    .collect();
                let r = run(kind, streaming, script, true, Some(3)).await;
                assert_eq!(
                    (r.bodies.len(), r.executed.len()),
                    (3, 3),
                    "{kind:?} streaming={streaming}"
                );
                assert_eq!(
                    r.message.content,
                    "Analysis reached the maximum number of tool calls. Here are the results gathered so far."
                );
            }
        }
    }
}

mod features {
    use super::*;
    const MANIFEST: &str = include_str!("../../Cargo.toml");

    /// WT-8: p2a-mcp's p2a-core dependency leaves `database` (duckdb-sys) off; p2a-mcp's own
    /// `database` feature passes it through and is in `default`, `full` and `embedded`.
    #[test]
    fn test_manifest_passes_the_database_feature_through() {
        let line = |prefix: &str| {
            MANIFEST
                .lines()
                .find(|l| l.trim_start().starts_with(prefix))
                .unwrap_or_else(|| panic!("crates/p2a-mcp/Cargo.toml has no `{prefix}` line"))
        };
        let core = line("p2a-core =");
        assert!(core.contains("default-features = false"), "{core}");
        assert!(
            !core.contains("\"full\"") && !core.contains("\"database\""),
            "{core}"
        );
        for f in [
            "\"visualization\"",
            "\"forecasting\"",
            "\"file-formats\"",
            "\"spectral-analysis\"",
        ] {
            assert!(core.contains(f), "{core} lacks {f}");
        }
        assert_eq!(
            line("database =").trim(),
            r#"database = ["p2a-core/database"]"#
        );
        for feature in ["default =", "full =", "embedded ="] {
            let l = line(feature);
            assert!(l.contains("\"database\""), "{l}");
        }
    }

    /// WT-8: the seven SQL tools exist exactly when `database` is compiled.
    #[test]
    fn test_database_tools_follow_the_database_feature() {
        let routed: Vec<String> = p2a_mcp::server::AnalyticsServer::new()
            .router_tools()
            .into_iter()
            .map(|t| t.name.to_string())
            .collect();
        let present: Vec<&str> = p2a_mcp::tools::registry::OPTIONAL_TOOLS
            .iter()
            .copied()
            .filter(|n| routed.iter().any(|r| r == n))
            .collect();
        if cfg!(feature = "database") {
            assert_eq!(present.len(), 7, "{present:?}");
        } else {
            assert!(
                present.is_empty(),
                "built without `database`, yet routed: {present:?}"
            );
        }
    }

    /// The server instructions name the SQL tools only in builds that have them.
    #[test]
    fn test_instructions_follow_the_database_feature() {
        use rmcp::ServerHandler;
        let text = p2a_mcp::server::AnalyticsServer::new()
            .get_info()
            .instructions
            .unwrap();
        assert_eq!(
            text.contains("db_sqlite_query"),
            cfg!(feature = "database"),
            "{text}"
        );
        assert!(text.starts_with("prompt2analytics is a local data analytics engine. Use 'load_dataset' to load a CSV or Parquet file"));
    }
}

#[cfg(feature = "db")]
mod conversations {
    use super::*;

    /// A live session with one conversation ("t1") holding the user message "hi".
    pub(super) async fn seeded() -> (axum::Router, String, String) {
        let (app, _state) = http_app().await;
        let sid = new_session(&app).await;
        let (s, conv, _) = send(
            &app,
            "POST",
            "/api/conversations",
            None,
            Some(json!({"session_id": sid, "title": "t1"})),
        )
        .await;
        assert_eq!(s, StatusCode::CREATED, "{conv}");
        let cid = conv["data"]["id"].as_str().unwrap().to_string();
        let (s, m, _) = send(
            &app,
            "POST",
            &format!("/api/conversations/{cid}/messages"),
            Some(&sid),
            Some(json!({"role": "user", "content": "hi"})),
        )
        .await;
        assert_eq!(s, StatusCode::CREATED, "{m}");
        (app, sid, cid)
    }

    /// MCP-15: GET .../full answers as GET /api/conversations/{id}, with the same ownership check.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_full_is_an_alias_of_get_conversation() {
        let (app, sid, cid) = seeded().await;
        let (s, full, len) = send(
            &app,
            "GET",
            &format!("/api/conversations/{cid}/full"),
            Some(&sid),
            None,
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{len} bytes: {full}");
        let (_, plain, _) = send(
            &app,
            "GET",
            &format!("/api/conversations/{cid}"),
            Some(&sid),
            None,
        )
        .await;
        assert_eq!(full, plain);
        assert_eq!(
            (
                full["data"]["conversation"]["id"].as_str(),
                full["data"]["messages"][0]["content"].as_str()
            ),
            (Some(cid.as_str()), Some("hi"))
        );
        let stranger = "00000000-0000-4000-8000-000000000000";
        let (s, body, _) = send(
            &app,
            "GET",
            &format!("/api/conversations/{cid}/full"),
            Some(stranger),
            None,
        )
        .await;
        assert_eq!(
            (s, body["code"].as_str()),
            (StatusCode::NOT_FOUND, Some("not_found"))
        );
    }

    /// MCP-15: the web UI archives with PUT {is_archived}; PUT applies title and is_archived.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_put_updates_title_and_archive_state() {
        let (app, sid, cid) = seeded().await;
        let uri = format!("/api/conversations/{cid}");
        let put = |body: Value| send(&app, "PUT", &uri, Some(&sid), Some(body));
        let (s, b, _) = put(json!({"is_archived": true})).await;
        assert_eq!(s, StatusCode::OK, "{b}");
        assert_eq!(
            (b["data"]["is_archived"].clone(), b["data"]["title"].clone()),
            (json!(true), json!("t1"))
        );
        let (s, b, _) = put(json!({"title": "t2", "is_archived": false})).await;
        assert_eq!(s, StatusCode::OK, "{b}");
        assert_eq!(
            (b["data"]["is_archived"].clone(), b["data"]["title"].clone()),
            (json!(false), json!("t2"))
        );
        let (s, b, _) = put(json!({"title": "t3"})).await;
        assert_eq!(
            (
                s,
                b["data"]["title"].clone(),
                b["data"]["is_archived"].clone()
            ),
            (StatusCode::OK, json!("t3"), json!(false))
        );
        let (_, list, _) = send(
            &app,
            "GET",
            &format!("/api/conversations?session_id={sid}"),
            Some(&sid),
            None,
        )
        .await;
        assert_eq!(list["data"][0]["title"], "t3");
        let (s, b, _) = put(json!({})).await;
        assert_eq!(
            (s, b["code"].as_str()),
            (StatusCode::BAD_REQUEST, Some("invalid_arguments"))
        );
        assert!(
            b["error"]
                .as_str()
                .unwrap()
                .contains("send `title`, `is_archived` or both"),
            "{b}"
        );
        let (s, b, _) = put(json!({"archived": true})).await;
        assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY, "{b}");
        assert!(
            b.as_str().unwrap().contains("unknown field `archived`"),
            "{b}"
        );
    }

    /// MCP-15: POST .../messages/clear answers as DELETE .../messages.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_messages_clear_is_an_alias_of_delete_messages() {
        let (app, sid, cid) = seeded().await;
        let (s, b, len) = send(
            &app,
            "POST",
            &format!("/api/conversations/{cid}/messages/clear"),
            Some(&sid),
            Some(json!({})),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{len} bytes: {b}");
        assert_eq!(b["data"]["deleted_count"], 1);
        let (_, msgs, _) = send(
            &app,
            "GET",
            &format!("/api/conversations/{cid}/messages"),
            Some(&sid),
            None,
        )
        .await;
        assert_eq!(msgs["data"], json!([]));
        let (s, b, _) = send(
            &app,
            "DELETE",
            &format!("/api/conversations/{cid}/messages"),
            Some(&sid),
            None,
        )
        .await;
        assert_eq!(
            (s, b["data"]["deleted_count"].clone()),
            (StatusCode::OK, json!(0))
        );
    }
}
