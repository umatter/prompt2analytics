//! Integration tests for the LLM egress guard, the upstream error and retry
//! policy, the key policy, the tool allowlist, the context budget and the
//! chat-loop bounds (T0-WP08).
//!
//! Upstreams are stub servers on loopback; the app is built with
//! `build_app` and driven with `tower::ServiceExt::oneshot`. Run with
//! `P2A_LLM_ALLOWED_HOSTS`, `P2A_LIMITS` and `P2A_LIMIT_OVERRIDES` unset:
//! the tests rely on the Off defaults.
#![cfg(all(feature = "http", feature = "llm"))]

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use axum::routing::{any, get, post};
use p2a_mcp::config::ServerConfig;
use p2a_mcp::llm::egress::{EgressPolicy, GuardedResolver, Lookup, guarded_client_builder};
use p2a_mcp::llm::{
    LlmError, LlmProvider, LlmSettings, Message, OllamaProvider, ProviderConfig, ProviderType,
    RetryConfig, ToolExecutor, active_chat_slots, send_with_retry,
};
use p2a_mcp::tools::limits::Limits;
use p2a_mcp::transport::http::{AppState, build_app};
use reqwest::dns::Resolve;
use serde_json::{Value, json};
use tower::ServiceExt;

/// Bind `127.0.0.1:0` and serve `router` on it in the background.
async fn spawn_stub(router: Router) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    addr
}

/// A lookup that answers from a fixed table and counts its calls.
fn fake_lookup(calls: Arc<AtomicUsize>) -> Lookup {
    Arc::new(move |host: String| {
        calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            let ips: Vec<IpAddr> = match host.as_str() {
                "mixed.test" => vec![
                    "93.184.216.34".parse().unwrap(),
                    "10.0.0.1".parse().unwrap(),
                ],
                "mapped.test" => vec!["::ffff:127.0.0.1".parse().unwrap()],
                "public.test" => vec!["93.184.216.34".parse().unwrap()],
                "localhost" | "internal.test" => vec!["127.0.0.1".parse().unwrap()],
                _ => vec![],
            };
            Ok(ips)
        })
    })
}

async fn resolve(r: &GuardedResolver, n: &str) -> Result<Vec<SocketAddr>, String> {
    r.resolve(n.parse().unwrap())
        .await
        .map(|a| a.collect())
        .map_err(|e| e.to_string())
}

fn open_policy() -> Arc<EgressPolicy> {
    Arc::new(EgressPolicy {
        allowed_hosts: None,
        allow_loopback: true,
    })
}

#[tokio::test]
async fn resolver_rejects_names_resolving_to_internal_addresses() {
    let calls = Arc::new(AtomicUsize::new(0));
    let open = open_policy();
    let r = GuardedResolver::with_lookup(open.clone(), false, fake_lookup(calls.clone()));
    let mixed = resolve(&r, "mixed.test").await.unwrap_err();
    assert!(
        !mixed.contains("10.0.0.1"),
        "never echo resolved internal addresses: {mixed}"
    );
    assert!(resolve(&r, "mapped.test").await.is_err());
    assert!(resolve(&r, "localhost").await.is_err());
    assert!(resolve(&r, "nothing.test").await.is_err());
    assert_eq!(
        resolve(&r, "public.test").await.unwrap(),
        vec!["93.184.216.34:0".parse::<SocketAddr>().unwrap()]
    );
    let lb = GuardedResolver::with_lookup(open, true, fake_lookup(calls.clone()));
    assert_eq!(
        resolve(&lb, "localhost").await.unwrap(),
        vec!["127.0.0.1:0".parse::<SocketAddr>().unwrap()]
    );
    assert!(
        resolve(&lb, "mixed.test").await.is_err(),
        "the Ollama loopback exemption covers loopback only"
    );
    let before = calls.load(Ordering::SeqCst);
    let listed = Arc::new(EgressPolicy {
        allowed_hosts: Some(vec!["api.openai.com".into()]),
        allow_loopback: false,
    });
    assert!(
        resolve(
            &GuardedResolver::with_lookup(listed, false, fake_lookup(calls.clone())),
            "public.test"
        )
        .await
        .is_err()
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        before,
        "a host off the allowlist is never looked up"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn guarded_client_never_connects_to_a_denied_address() {
    let hits = Arc::new(AtomicUsize::new(0));
    let h = hits.clone();
    let addr = spawn_stub(Router::new().route(
        "/ping",
        get(move || {
            let h = h.clone();
            async move {
                h.fetch_add(1, Ordering::SeqCst);
                "pong"
            }
        }),
    ))
    .await;
    let open = open_policy();
    let calls = Arc::new(AtomicUsize::new(0));
    let url = format!("http://internal.test:{}/ping", addr.port());
    let denied = guarded_client_builder(GuardedResolver::with_lookup(
        open.clone(),
        false,
        fake_lookup(calls.clone()),
    ))
    .build()
    .unwrap();
    let err = denied.get(&url).send().await.unwrap_err();
    let mapped = LlmError::from(err);
    assert!(matches!(mapped, LlmError::EgressDenied(_)), "{mapped}");
    assert_eq!(hits.load(Ordering::SeqCst), 0);
    // Positive control: the same client shape connects when the resolver
    // admits loopback, so the denial above came from the resolver.
    let allowed =
        guarded_client_builder(GuardedResolver::with_lookup(open, true, fake_lookup(calls)))
            .build()
            .unwrap();
    assert!(
        allowed
            .get(&url)
            .send()
            .await
            .unwrap()
            .status()
            .is_success()
    );
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

// ---------------------------------------------------------------------------
// Task 2: redirects, upstream errors, retries, timeouts, response caps
// ---------------------------------------------------------------------------

fn off_settings() -> LlmSettings {
    LlmSettings::from_limits(&Limits::off(), None)
}

/// An Ollama provider pointed at a stub. Off allows loopback for Ollama, and
/// an IP-literal host never reaches the resolver.
fn ollama(addr: SocketAddr, s: &LlmSettings) -> OllamaProvider {
    OllamaProvider::with_settings(
        ProviderConfig {
            provider_type: ProviderType::Ollama,
            base_url: Some(format!("http://{addr}")),
            model: "stub".into(),
            ..ProviderConfig::default()
        },
        s,
    )
}

/// A tool executor that refuses every call.
struct NoTools;

#[async_trait::async_trait]
impl ToolExecutor for NoTools {
    async fn execute(&self, _name: &str, _arguments: Value) -> Result<String, String> {
        Err("no tools".into())
    }
}

/// A tool executor that answers every call with "ok".
struct EchoTools;

#[async_trait::async_trait]
impl ToolExecutor for EchoTools {
    async fn execute(&self, _name: &str, _arguments: Value) -> Result<String, String> {
        Ok("ok".into())
    }
}

fn counter() -> Arc<AtomicUsize> {
    Arc::new(AtomicUsize::new(0))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn redirects_are_not_followed() {
    let after_hits = counter();
    let a = after_hits.clone();
    let addr = spawn_stub(
        Router::new()
            .route(
                "/api/chat",
                post(|| async {
                    (
                        StatusCode::TEMPORARY_REDIRECT,
                        [(header::LOCATION, "/after")],
                    )
                }),
            )
            .route(
                "/after",
                any(move || {
                    let a = a.clone();
                    async move {
                        a.fetch_add(1, Ordering::SeqCst);
                        r#"{"message":{"role":"assistant","content":"ok"},"done":true}"#
                    }
                }),
            ),
    )
    .await;
    let err = ollama(addr, &off_settings())
        .chat(&[Message::user("hi")], &[], &NoTools, true)
        .await
        .unwrap_err();
    assert!(
        matches!(err, LlmError::Upstream { status: 307, .. }),
        "{err}"
    );
    assert_eq!(after_hits.load(Ordering::SeqCst), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn upstream_error_bodies_are_not_reflected() {
    let addr = spawn_stub(Router::new().route(
        "/api/chat",
        post(|| async {
            (
                StatusCode::UNAUTHORIZED,
                "UPSTREAM_BODY_MARKER internal detail",
            )
        }),
    ))
    .await;
    let err = ollama(addr, &off_settings())
        .chat(&[Message::user("hi")], &[], &NoTools, true)
        .await
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "Ollama returned HTTP 401; check the API key"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn server_errors_are_retried_at_most_three_times() {
    let hits = counter();
    let h = hits.clone();
    let addr = spawn_stub(Router::new().route(
        "/api/chat",
        post(move || {
            let h = h.clone();
            async move {
                h.fetch_add(1, Ordering::SeqCst);
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    [(header::RETRY_AFTER, "0")],
                    "UPSTREAM_BODY_MARKER",
                )
            }
        }),
    ))
    .await;
    let err = ollama(addr, &off_settings())
        .chat(&[Message::user("hi")], &[], &NoTools, true)
        .await
        .unwrap_err();
    assert!(
        matches!(err, LlmError::Upstream { status: 503, .. }),
        "{err}"
    );
    assert!(!err.to_string().contains("UPSTREAM_BODY_MARKER"));
    assert_eq!(hits.load(Ordering::SeqCst), 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn completion_timeout_is_not_retried() {
    let hits = counter();
    let h = hits.clone();
    let addr = spawn_stub(Router::new().route(
        "/api/chat",
        post(move || {
            let h = h.clone();
            async move {
                h.fetch_add(1, Ordering::SeqCst);
                std::future::pending::<()>().await;
                "never"
            }
        }),
    ))
    .await;
    let s = LlmSettings {
        timeout: Duration::from_millis(300),
        ..off_settings()
    };
    let err = ollama(addr, &s)
        .chat(&[Message::user("hi")], &[], &NoTools, true)
        .await
        .unwrap_err();
    assert!(matches!(err, LlmError::Timeout), "{err}");
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn connect_errors_are_retried_before_send() {
    // A port that was free a moment ago: the listener closes on drop.
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let client = guarded_client_builder(GuardedResolver::new(open_policy(), true))
        .build()
        .unwrap();
    let n = AtomicUsize::new(0);
    let cfg = RetryConfig {
        max_retries: 2,
        initial_delay_ms: 1,
        max_delay_ms: 5,
        backoff_factor: 2.0,
    };
    let res = send_with_retry(
        || {
            n.fetch_add(1, Ordering::SeqCst);
            client.post(format!("http://127.0.0.1:{port}/api/chat"))
        },
        &cfg,
    )
    .await;
    assert!(matches!(res, Err(LlmError::NetworkError(_))), "{res:?}");
    assert_eq!(n.load(Ordering::SeqCst), 3);
    assert_eq!(RetryConfig::default().max_retries, 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn egress_denial_is_not_retried() {
    let hits = counter();
    let h = hits.clone();
    let addr = spawn_stub(Router::new().route(
        "/api/chat",
        post(move || {
            let h = h.clone();
            async move {
                h.fetch_add(1, Ordering::SeqCst);
                "{}"
            }
        }),
    ))
    .await;
    let client = guarded_client_builder(GuardedResolver::with_lookup(
        open_policy(),
        false,
        fake_lookup(counter()),
    ))
    .build()
    .unwrap();
    let n = AtomicUsize::new(0);
    let cfg = RetryConfig {
        max_retries: 2,
        initial_delay_ms: 1,
        max_delay_ms: 5,
        backoff_factor: 2.0,
    };
    let url = format!("http://internal.test:{}/api/chat", addr.port());
    let res = send_with_retry(
        || {
            n.fetch_add(1, Ordering::SeqCst);
            client.post(&url)
        },
        &cfg,
    )
    .await;
    assert!(matches!(res, Err(LlmError::EgressDenied(_))), "{res:?}");
    assert_eq!(n.load(Ordering::SeqCst), 1);
    assert_eq!(hits.load(Ordering::SeqCst), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn oversized_upstream_body_is_refused() {
    let addr = spawn_stub(
        Router::new().route("/api/chat", post(|| async { vec![b'x'; 9 * 1024 * 1024] })),
    )
    .await;
    let err = ollama(addr, &off_settings())
        .chat(&[Message::user("hi")], &[], &NoTools, true)
        .await
        .unwrap_err();
    assert!(matches!(err, LlmError::ResponseTooLarge(_)), "{err}");
}

// ---------------------------------------------------------------------------
// The HTTP app: helpers
// ---------------------------------------------------------------------------

async fn app() -> (Router, AppState) {
    build_app(&ServerConfig::default()).await.unwrap()
}

fn post_json(uri: &str, body: &Value) -> Request<Body> {
    Request::post(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn send_json(app: &Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let resp = app.clone().oneshot(post_json(uri, &body)).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

async fn new_session(app: &Router) -> String {
    let (s, b) = send_json(app, "/api/sessions", json!({})).await;
    assert_eq!(s, StatusCode::CREATED, "{b}");
    b["data"]["session_id"].as_str().unwrap().to_string()
}

/// The provider block of a chat request pointing at an Ollama stub.
fn stub_provider(addr: SocketAddr) -> Value {
    json!({"provider_type": "ollama", "base_url": format!("http://{addr}"), "model": "stub"})
}

// ---------------------------------------------------------------------------
// Task 4: the tool allowlist of the LLM executors
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn llm_executor_refuses_internal_and_unknown_tools() {
    let (app, _state) = app().await;
    let sid = new_session(&app).await;
    let addr = spawn_stub(Router::new().route(
        "/api/chat",
        post(|| async {
            axum::Json(json!({
                "message": {"role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": "server_stats", "arguments": {}}},
                    {"function": {"name": "definitely_not_a_tool", "arguments": {}}},
                    {"function": {"name": "hypothesis_poisson", "arguments": {}}}
                ]},
                "done": true
            }))
        }),
    ))
    .await;
    let (status, body) = send_json(
        &app,
        "/api/llm/chat",
        json!({"session_id": sid, "message": "go", "interpret": false,
               "provider": stub_provider(addr)}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let r = body["data"]["message"]["tool_results"]
        .as_array()
        .unwrap_or_else(|| panic!("tool_results in {body}"));
    assert!(
        r[0]["is_error"].as_bool().unwrap()
            && r[0]["content"]
                .as_str()
                .unwrap()
                .contains("is not available to the assistant"),
        "{}",
        r[0]
    );
    assert!(
        r[1]["is_error"].as_bool().unwrap()
            && r[1]["content"].as_str().unwrap().contains("Unknown tool"),
        "{}",
        r[1]
    );
    // Dispatched (it fails on its own arguments), not refused.
    let c2 = r[2]["content"].as_str().unwrap();
    assert!(
        !c2.contains("is not available to the assistant") && !c2.contains("Unknown tool"),
        "{c2}"
    );
}

// ---------------------------------------------------------------------------
// Task 5b: the bounded context on the chat paths
// ---------------------------------------------------------------------------

type Recorded = Arc<std::sync::Mutex<Vec<Value>>>;

/// An Ollama stub that records each request body and answers "ok".
async fn recording_stub() -> (SocketAddr, Recorded) {
    let recorded: Recorded = Arc::default();
    let r = recorded.clone();
    let addr = spawn_stub(Router::new().route(
        "/api/chat",
        post(move |axum::Json(body): axum::Json<Value>| {
            let r = r.clone();
            async move {
                r.lock().unwrap().push(body);
                axum::Json(json!({"message": {"role": "assistant", "content": "ok"}, "done": true}))
            }
        }),
    ))
    .await;
    (addr, recorded)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn system_prompt_dataset_block_is_bounded() {
    let (app, _state) = app().await;
    let sid = new_session(&app).await;
    let mut csv = (0..3000)
        .map(|j| format!("c{j}"))
        .collect::<Vec<_>>()
        .join(",");
    for i in 0..3 {
        csv.push('\n');
        csv.push_str(
            &(0..3000)
                .map(|j| format!("{:.3}", i as f64 * 0.37 + j as f64 * 0.011))
                .collect::<Vec<_>>()
                .join(","),
        );
    }
    let (status, body) = send_json(
        &app,
        "/api/tools/create_dataset",
        json!({"session_id": sid, "arguments": {"name": "wide", "csv_content": csv}}),
    )
    .await;
    assert!(status.is_success(), "{body}");

    let (addr, recorded) = recording_stub().await;
    let (status, body) = send_json(
        &app,
        "/api/llm/chat",
        json!({"session_id": sid, "message": "hi", "provider": stub_provider(addr)}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let recorded = recorded.lock().unwrap();
    let sys = recorded[0]["messages"][0]["content"].as_str().unwrap();
    let start = sys.find("<<<DATASETS").expect("an opening delimiter");
    let end = sys.find("DATASETS>>>").expect("a closing delimiter");
    let block = &sys[start..end];
    // The context budget plus the opening delimiter line.
    assert!(block.len() <= 65_536 + 32, "{}", block.len());
    assert!(block.contains("\"c0\"") && block.contains("and 2940 more columns"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn long_history_is_compacted_before_sending() {
    let (app, _state) = app().await;
    let sid = new_session(&app).await;
    // 300 messages of 2,000 ASCII bytes each: 600 KB > 524,288.
    let history: Vec<Value> = (0..300)
        .map(|i| {
            json!({
                "role": if i % 2 == 0 { "user" } else { "assistant" },
                "content": format!("{i:03}:{}", "h".repeat(1996)),
            })
        })
        .collect();
    let (addr, recorded) = recording_stub().await;
    let (status, body) = send_json(
        &app,
        "/api/llm/chat",
        json!({"session_id": sid, "message": "now", "history": history,
               "provider": stub_provider(addr)}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let recorded = recorded.lock().unwrap();
    let msgs = recorded[0]["messages"].as_array().unwrap();
    let hist = &msgs[1..msgs.len() - 1];
    let bytes: usize = hist
        .iter()
        .map(|m| m["content"].as_str().unwrap_or("").len())
        .sum();
    assert!(bytes <= 524_288, "{bytes}");
    assert!(
        hist[0]["content"]
            .as_str()
            .unwrap()
            .starts_with("[Conversation context from earlier in this session:]")
    );
    assert_eq!(
        hist.last().unwrap()["content"],
        history.last().unwrap()["content"]
    );
    // The current message is never compacted.
    assert_eq!(msgs.last().unwrap()["content"], "now");
}

// ---------------------------------------------------------------------------
// Task 6: iteration cap, chat slots, disconnect
// ---------------------------------------------------------------------------

/// An Ollama answer that calls one tool.
fn tool_call_answer(name: &str) -> Value {
    json!({"message": {"role": "assistant", "content": "",
                       "tool_calls": [{"function": {"name": name, "arguments": {}}}]},
           "done": true})
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn provider_honours_iteration_cap() {
    let hits = counter();
    let h = hits.clone();
    // Request n calls tool `t{n}`, so loop detection never fires.
    let addr = spawn_stub(Router::new().route(
        "/api/chat",
        post(move || {
            let n = h.fetch_add(1, Ordering::SeqCst) + 1;
            async move { axum::Json(tool_call_answer(&format!("t{n}"))) }
        }),
    ))
    .await;
    let s = LlmSettings {
        max_tool_iterations: 3,
        ..off_settings()
    };
    let p = OllamaProvider::with_settings(
        ProviderConfig {
            provider_type: ProviderType::Ollama,
            base_url: Some(format!("http://{addr}")),
            model: "stub".into(),
            max_tool_iterations: Some(1000),
            ..ProviderConfig::default()
        },
        &s,
    );
    let m = p
        .chat(&[Message::user("go")], &[], &EchoTools, true)
        .await
        .unwrap();
    assert!(
        m.content.contains("maximum number of tool calls"),
        "{}",
        m.content
    );
    assert_eq!(hits.load(Ordering::SeqCst), 3);
}

/// Poll `cond` every 10 ms; the timeout only guards against a hang.
async fn eventually(what: &str, cond: impl Fn() -> bool) {
    tokio::time::timeout(Duration::from_secs(10), async {
        while !cond() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("{what}"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_task_stops_after_client_disconnects() {
    use futures::StreamExt;

    let (app, _state) = app().await;
    let sid = new_session(&app).await;
    let hits = counter();
    let release = Arc::new(tokio::sync::Notify::new());
    let (h, r) = (hits.clone(), release.clone());
    // Request 1 calls list_datasets; request 2 never answers.
    let addr = spawn_stub(Router::new().route(
        "/api/chat",
        post(move || {
            let (h, r) = (h.clone(), r.clone());
            async move {
                if h.fetch_add(1, Ordering::SeqCst) + 1 >= 2 {
                    r.notified().await;
                }
                format!("{}\n", tool_call_answer("list_datasets"))
            }
        }),
    ))
    .await;

    let resp = app
        .clone()
        .oneshot(post_json(
            "/api/llm/chat/stream",
            &json!({"session_id": sid, "message": "go", "interpret": true,
                    "provider": stub_provider(addr)}),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let mut stream = resp.into_body().into_data_stream();
    let mut text = String::new();
    tokio::time::timeout(Duration::from_secs(10), async {
        while !text.contains("\"type\":\"tool_end\"") {
            let chunk = stream.next().await.expect("more events").unwrap();
            text.push_str(&String::from_utf8_lossy(&chunk));
        }
    })
    .await
    .unwrap_or_else(|_| panic!("a tool_end event; got {text}"));
    eventually("the second provider request", || {
        hits.load(Ordering::SeqCst) == 2
    })
    .await;

    assert_eq!(active_chat_slots(&sid), 1);
    drop(stream); // the client goes away
    eventually("the chat task must end when the client disconnects", || {
        active_chat_slots(&sid) == 0
    })
    .await;
    release.notify_waiters();
    // The task is gone; nobody sends request 3.
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(hits.load(Ordering::SeqCst), 2);
}
