//! The public HTTP start functions, `transport::http::serve` and
//! `start_http_transport`, install the request limits of the configuration
//! they serve: an embedder that starts an open public server through them
//! gets the hosted floor (the catalogue, no free-form SQL, no server LLM
//! keys, no loopback Ollama) with its `limits.overrides` applied (PR-18).
//!
//! One process serves one set of request limits: while a server runs, a
//! start whose limits differ is refused instead of replacing them under it.
//!
//! `serve` installs the limits and the process-wide policies (allocation
//! ceiling, session directories) for the whole process, and the tests of one
//! binary run in parallel. So every server in this binary that starts serves
//! the open public configuration of [`open_public`], whose limits are
//! [`expected`]; the other configurations here are refused before they
//! install anything, and a test that starts another configuration belongs in
//! a binary of its own. None of these tests touches a file, so the session
//! directories stay unused.
#![cfg(feature = "http")]

use std::net::SocketAddr;

use p2a_mcp::config::{ServerConfig, TransportType};
use p2a_mcp::tools::limits::{self, Family, Limits};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// The overrides of every configuration served here: the lag cap raised from
/// the hosted 1,000 to 1,500, so a test sees both the hosted catalogue and the
/// override applied.
fn overrides() -> Vec<String> {
    vec!["lags=1500".to_string()]
}

/// An open public configuration (no token, `--allow-unauthenticated`) on
/// `addr`, with [`overrides`]; everything else at its default.
fn open_public(addr: SocketAddr) -> ServerConfig {
    let mut c = ServerConfig {
        transport: TransportType::Http,
        ..Default::default()
    };
    c.http.addr = addr;
    c.http.allow_unauthenticated = true;
    c.limits.overrides = overrides();
    c
}

/// The limits an [`open_public`] server must run under.
fn expected() -> Limits {
    Limits::hosted().with_overrides(&overrides()).unwrap()
}

/// A local HTTP configuration on `addr` (loopback, no token, no `--limits
/// hosted`, no overrides): its limits are the off profile.
fn local(addr: SocketAddr) -> ServerConfig {
    let mut c = ServerConfig {
        transport: TransportType::Http,
        ..Default::default()
    };
    c.http.addr = addr;
    c
}

/// An [`open_public`] server started through bind + serve on a free loopback
/// port and already answering, so it holds its limits: its address, the
/// sender that stops it and its task.
async fn running_open_public() -> (
    SocketAddr,
    tokio::sync::oneshot::Sender<()>,
    tokio::task::JoinHandle<p2a_mcp::transport::TransportResult<()>>,
) {
    let cfg = open_public("127.0.0.1:0".parse().unwrap());
    let listener = p2a_mcp::transport::http::bind(&cfg).await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(p2a_mcp::transport::http::serve(listener, cfg, async {
        let _ = stopped.await;
    }));
    let (status, created) = request(addr, "POST", "/api/sessions", Some(&json!({}))).await;
    assert!((200..300).contains(&status), "{status} {created}");
    (addr, stop, server)
}

/// Asserts that `e` is the refusal of a start whose limits differ from those
/// of a running server, and that the installed limits are still [`expected`].
fn assert_refused_for_other_limits(e: &p2a_mcp::transport::TransportError, what: &str) {
    assert!(
        e.to_string()
            .contains("running under different request limits"),
        "{what}: {e}"
    );
    assert_eq!(
        *limits::current(),
        expected(),
        "{what} replaced the limits of the running open public server"
    );
}

/// One HTTP/1.1 request to `addr` (the connection closed after the answer):
/// the status code and the JSON body (`Null` when the body is not JSON). The
/// answer comes only once the server is serving, which is after its startup.
async fn request(addr: SocketAddr, method: &str, uri: &str, body: Option<&Value>) -> (u16, Value) {
    let body = body.map(Value::to_string).unwrap_or_default();
    let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let request = format!(
        "{method} {uri} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes()).await.unwrap();
    let mut answer = String::new();
    stream.read_to_string(&mut answer).await.unwrap();
    let (head, json) = answer.split_once("\r\n\r\n").unwrap();
    let status = head
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("no status line in {head:?}"));
    (status, serde_json::from_str(json).unwrap_or(Value::Null))
}

/// Checks a server serving [`open_public`] on `addr`: a lag above the
/// overridden cap is refused naming that cap, a lag between the hosted cap
/// and the override runs, and the installed limits are [`expected`].
async fn assert_serves_open_public_limits(addr: SocketAddr) {
    let (status, created) = request(addr, "POST", "/api/sessions", Some(&json!({}))).await;
    assert!((200..300).contains(&status), "{status} {created}");
    let sid = created["data"]["session_id"].as_str().unwrap().to_string();
    let call = |lag_max: u64| {
        let arguments = json!({"lag_max": lag_max, "ar": [0.5], "ma": [0.3]});
        json!({"session_id": sid, "arguments": arguments})
    };

    let uri = "/api/tools/timeseries_arma_to_ma";
    let (status, refused) = request(addr, "POST", uri, Some(&call(2000))).await;
    assert_eq!(status, 422, "{refused}");
    assert_eq!(refused["code"], "limit_exceeded", "{refused}");
    assert_eq!(
        refused["details"],
        json!({"parameter": "lag_max", "requested": "2000", "limit": "1500"})
    );

    let (status, ran) = request(addr, "POST", uri, Some(&call(1200))).await;
    assert_eq!(status, 200, "{ran}");
    assert_eq!(ran["success"], true, "{ran}");

    // The server answered, so its startup, limits included, is done.
    let installed = limits::current();
    assert_eq!(*installed, expected(), "the server runs without its limits");
    assert!(!installed.sql_freeform, "free-form SQL is on");
    assert!(!installed.llm_server_keys, "server LLM keys are on");
    assert!(!installed.llm_allow_loopback, "loopback Ollama is on");
    assert_eq!(installed.cap(Family::Lags), Some(1500));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn serve_installs_the_limits_of_an_open_public_config() {
    let cfg = open_public("127.0.0.1:0".parse().unwrap());
    let listener = p2a_mcp::transport::http::bind(&cfg).await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(p2a_mcp::transport::http::serve(listener, cfg, async {
        let _ = stopped.await;
    }));

    assert_serves_open_public_limits(addr).await;

    stop.send(()).unwrap();
    server.await.unwrap().unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn start_http_transport_installs_the_limits_of_an_open_public_config() {
    // `start_http_transport` binds the configured address and never returns
    // while it serves. A free port is found by binding port 0 and releasing
    // it; if the port is taken again before the server binds it, the bind
    // fails and another port is tried.
    for _ in 0..20 {
        let free = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = free.local_addr().unwrap();
        drop(free);
        let cfg = open_public(addr);
        let server =
            tokio::spawn(async move { p2a_mcp::transport::http::start_http_transport(&cfg).await });
        // Wait (bounded) until the address accepts connections or the start
        // has failed.
        let mut bound = false;
        for _ in 0..3000 {
            if server.is_finished() {
                break;
            }
            if tokio::net::TcpStream::connect(addr).await.is_ok() {
                bound = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        if !bound {
            assert!(server.is_finished(), "{addr} neither bound nor refused");
            let e = server
                .await
                .unwrap()
                .expect_err("start_http_transport returned while serving");
            assert!(e.to_string().contains("Failed to bind"), "{e}");
            continue;
        }
        assert_serves_open_public_limits(addr).await;
        server.abort();
        return;
    }
    panic!("no free port found for start_http_transport");
}

#[tokio::test]
async fn serve_refuses_an_unknown_limit_override() {
    // Fails before the limits or the process-wide policies are installed, so
    // this configuration does not change what the other tests see.
    let mut cfg = open_public("127.0.0.1:0".parse().unwrap());
    cfg.limits.overrides = vec!["no_such_limit=1".to_string()];
    let listener = p2a_mcp::transport::http::bind(&cfg).await.unwrap();
    let e = p2a_mcp::transport::http::serve(listener, cfg, std::future::ready(()))
        .await
        .expect_err("serve started with an unknown limit override");
    assert!(e.to_string().contains("no_such_limit"), "{e}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn serve_refuses_other_limits_while_an_open_public_server_runs() {
    let (addr, stop, server) = running_open_public().await;

    let cfg = local("127.0.0.1:0".parse().unwrap());
    assert_eq!(limits::resolve(&cfg), Ok(Limits::off()));
    let listener = p2a_mcp::transport::http::bind(&cfg).await.unwrap();
    let e = p2a_mcp::transport::http::serve(listener, cfg, std::future::ready(()))
        .await
        .expect_err("serve started a local configuration next to an open public server");
    assert_refused_for_other_limits(&e, "serve of a local configuration");

    // The open public server still runs under its limits.
    assert_serves_open_public_limits(addr).await;
    stop.send(()).unwrap();
    server.await.unwrap().unwrap();
}
