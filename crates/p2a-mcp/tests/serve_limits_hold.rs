//! How long a server holds the process-wide request limits (PR-18): while
//! any connection it accepted, or any websocket session upgraded from one,
//! is still served, not only while `serve` runs. axum serves each connection
//! in a task of its own and an upgraded websocket outside it, so a
//! connection outlives an aborted `start_http_transport`, and a websocket
//! session outlives a graceful shutdown. Until the last of them closes, a
//! start of other limits is refused and the old connections keep the open
//! public server's limits; after it, such a start is accepted.
//!
//! Each test ends by serving a local configuration, which installs the off
//! profile process-wide, so the tests of this binary run one at a time
//! ([`one_at_a_time`]) and nothing else here serves anything. They live
//! apart from `serve_limits.rs`, whose tests serve open public servers in
//! parallel and would hold the limits whatever the server under test did.
#![cfg(feature = "http")]

use std::net::SocketAddr;
use std::time::Duration;

use p2a_mcp::config::{ServerConfig, TransportType};
use p2a_mcp::tools::limits::{self, Limits};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Serialises the tests of this binary: each installs the off profile at
/// its end.
static ONE_AT_A_TIME: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

async fn one_at_a_time() -> tokio::sync::MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().await
}

/// The overrides of the open public configuration: the lag cap raised from
/// the hosted 1,000 to 1,500.
fn overrides() -> Vec<String> {
    vec!["lags=1500".to_string()]
}

/// An open public configuration (no token, `--allow-unauthenticated`) on
/// `addr`, with [`overrides`].
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

/// The limits an [`open_public`] server runs under.
fn expected() -> Limits {
    Limits::hosted().with_overrides(&overrides()).unwrap()
}

/// A local configuration on a free loopback port: the off profile.
fn local() -> ServerConfig {
    let mut c = ServerConfig {
        transport: TransportType::Http,
        ..Default::default()
    };
    c.http.addr = "127.0.0.1:0".parse().unwrap();
    c
}

/// The bound on every wait for the server; a liveness guard, not a timing
/// assertion.
const PATIENCE: Duration = Duration::from_secs(60);

/// Whether `e` is the refusal of a start whose limits differ from those of a
/// running server.
fn is_conflict(e: &p2a_mcp::transport::TransportError) -> bool {
    e.to_string()
        .contains("running under different request limits")
}

/// Serve [`local`] once and assert that it is refused because a server of
/// other limits still holds them, and that the installed limits are still
/// [`expected`].
async fn assert_local_refused(what: &str) {
    let cfg = local();
    let listener = p2a_mcp::transport::http::bind(&cfg).await.unwrap();
    let e = p2a_mcp::transport::http::serve(listener, cfg, std::future::ready(()))
        .await
        .expect_err(what);
    assert!(is_conflict(&e), "{what}: {e}");
    assert_eq!(
        *limits::current(),
        expected(),
        "{what}: the limits of the open public server were replaced"
    );
}

/// Serve [`local`] until it is accepted (it shuts down at once), which is
/// once nothing of the open public server holds the limits any more. The
/// refusals before that must be the conflict; the wait is bounded.
async fn assert_local_accepted_once_released(what: &str) {
    let give_up = std::time::Instant::now() + PATIENCE;
    loop {
        let cfg = local();
        let listener = p2a_mcp::transport::http::bind(&cfg).await.unwrap();
        match p2a_mcp::transport::http::serve(listener, cfg, std::future::ready(())).await {
            Ok(()) => break,
            Err(e) => assert!(is_conflict(&e), "{what}: {e}"),
        }
        assert!(
            std::time::Instant::now() < give_up,
            "{what}: the hold was never released"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(*limits::current(), Limits::off(), "{what}");
}

/// One HTTP/1.1 request to `addr` on a connection of its own, closed after
/// the answer: the status code and the JSON body.
async fn request(addr: SocketAddr, method: &str, uri: &str, body: Option<&Value>) -> (u16, Value) {
    let mut connection = KeepAlive::open(addr).await;
    connection.request(method, uri, body).await
}

/// A client connection kept open between requests (HTTP/1.1 keep-alive).
struct KeepAlive(tokio::net::TcpStream);

impl KeepAlive {
    async fn open(addr: SocketAddr) -> Self {
        Self(tokio::net::TcpStream::connect(addr).await.unwrap())
    }

    /// One request on this connection, which stays open: the status code
    /// and the JSON body (`Null` when it is not JSON).
    async fn request(&mut self, method: &str, uri: &str, body: Option<&Value>) -> (u16, Value) {
        tokio::time::timeout(PATIENCE, self.exchange(method, uri, body))
            .await
            .unwrap_or_else(|_| panic!("no answer to {method} {uri}"))
    }

    async fn exchange(&mut self, method: &str, uri: &str, body: Option<&Value>) -> (u16, Value) {
        let body = body.map(Value::to_string).unwrap_or_default();
        let request = format!(
            "{method} {uri} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\n\r\n{body}",
            body.len()
        );
        self.0.write_all(request.as_bytes()).await.unwrap();
        let mut answer = Vec::new();
        let head_end = loop {
            if let Some(i) = answer.windows(4).position(|w| w == b"\r\n\r\n") {
                break i + 4;
            }
            self.read_more(&mut answer).await;
        };
        let head = String::from_utf8_lossy(&answer[..head_end]).into_owned();
        let length: usize = head
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse().unwrap())
            })
            .unwrap_or_else(|| panic!("no Content-Length in {head:?}"));
        while answer.len() < head_end + length {
            self.read_more(&mut answer).await;
        }
        let status = head
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("no status line in {head:?}"));
        let json = &answer[head_end..head_end + length];
        (status, serde_json::from_slice(json).unwrap_or(Value::Null))
    }

    async fn read_more(&mut self, answer: &mut Vec<u8>) {
        let mut chunk = [0u8; 4096];
        let n = self.0.read(&mut chunk).await.unwrap();
        assert!(n > 0, "the server closed the connection");
        answer.extend_from_slice(&chunk[..n]);
    }
}

/// The session id in the answer to a `POST /api/sessions`.
fn session_id(status: u16, created: &Value) -> String {
    assert!((200..300).contains(&status), "{status} {created}");
    created["data"]["session_id"].as_str().unwrap().to_string()
}

/// The arguments of a `timeseries_arma_to_ma` call with `lag_max`.
fn lags(lag_max: u64) -> Value {
    json!({"lag_max": lag_max, "ar": [0.5], "ma": [0.3]})
}

/// Above the overridden lag cap of [`expected`]; the off profile runs it.
const OVER_CAP: u64 = 2000;

/// Asserts the answer of an HTTP tool call above the lag cap: refused,
/// naming the overridden cap.
fn assert_lag_refused(status: u16, answer: &Value, what: &str) {
    assert_eq!(status, 422, "{what}: {answer}");
    assert_eq!(answer["code"], "limit_exceeded", "{what}: {answer}");
    assert_eq!(answer["details"]["limit"], "1500", "{what}: {answer}");
}

/// Start [`open_public`] through `start_http_transport` on a free loopback
/// port and wait (bounded) until it accepts connections: its address and its
/// task. `start_http_transport` binds the configured address, so a port
/// taken again before it binds is retried.
async fn start_http_transport_open_public() -> (
    SocketAddr,
    tokio::task::JoinHandle<p2a_mcp::transport::TransportResult<()>>,
) {
    for _ in 0..20 {
        let free = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = free.local_addr().unwrap();
        drop(free);
        let cfg = open_public(addr);
        let server =
            tokio::spawn(async move { p2a_mcp::transport::http::start_http_transport(&cfg).await });
        let give_up = std::time::Instant::now() + PATIENCE;
        while !server.is_finished() && std::time::Instant::now() < give_up {
            if tokio::net::TcpStream::connect(addr).await.is_ok() {
                return (addr, server);
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(server.is_finished(), "{addr} neither bound nor refused");
        let e = server
            .await
            .unwrap()
            .expect_err("start_http_transport returned while serving");
        assert!(e.to_string().contains("Failed to bind"), "{e}");
    }
    panic!("no free port found for start_http_transport");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_aborted_start_http_transport_holds_its_limits_until_its_last_connection_closes() {
    let _alone = one_at_a_time().await;
    let (addr, server) = start_http_transport_open_public().await;
    let mut open = KeepAlive::open(addr).await;
    let (status, created) = open
        .request("POST", "/api/sessions", Some(&json!({})))
        .await;
    let sid = session_id(status, &created);
    let call = json!({"session_id": sid, "arguments": lags(OVER_CAP)});
    let uri = "/api/tools/timeseries_arma_to_ma";
    let (status, answer) = open.request("POST", uri, Some(&call)).await;
    assert_lag_refused(status, &answer, "before the abort");

    // start_http_transport has no shutdown signal, so aborting its task is
    // the only way to stop it in-process. It stops accepting connections;
    // the open one is still served.
    server.abort();
    assert!(server.await.unwrap_err().is_cancelled());
    assert_local_refused("serve of a local configuration while a connection is open").await;
    let (status, answer) = open.request("POST", uri, Some(&call)).await;
    assert_lag_refused(status, &answer, "on the open connection after the abort");

    drop(open);
    assert_local_accepted_once_released("after the last connection closed").await;
}

#[cfg(feature = "websocket")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_websocket_session_holds_its_servers_limits_after_a_graceful_shutdown() {
    use futures::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;

    type Client = tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >;

    /// The next text message of `ws`, as JSON.
    async fn next_json(ws: &mut Client) -> Value {
        loop {
            let frame = tokio::time::timeout(PATIENCE, ws.next())
                .await
                .expect("no message from the server")
                .expect("the websocket ended")
                .unwrap();
            if let Message::Text(text) = frame {
                return serde_json::from_str(&text).unwrap();
            }
        }
    }

    /// Send a lag call above the cap on `ws` and assert it is refused.
    async fn assert_ws_lag_refused(ws: &mut Client, what: &str) {
        let call = json!({"type": "tool", "id": "lag", "name": "timeseries_arma_to_ma",
            "args": lags(OVER_CAP)});
        ws.send(Message::Text(call.to_string().into()))
            .await
            .unwrap();
        let answer = next_json(ws).await;
        assert_eq!(answer["type"], "error", "{what}: {answer}");
        assert_eq!(answer["code"], "limit_exceeded", "{what}: {answer}");
    }

    let _alone = one_at_a_time().await;
    let mut cfg = open_public("127.0.0.1:0".parse().unwrap());
    cfg.http.enable_ws = true;
    let listener = p2a_mcp::transport::http::bind(&cfg).await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(p2a_mcp::transport::http::serve(listener, cfg, async {
        let _ = stopped.await;
    }));
    let (status, created) = request(addr, "POST", "/api/sessions", Some(&json!({}))).await;
    let sid = session_id(status, &created);
    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws"))
        .await
        .unwrap();
    let auth = json!({"type": "auth", "session_id": sid});
    ws.send(Message::Text(auth.to_string().into()))
        .await
        .unwrap();
    assert_eq!(
        next_json(&mut ws).await,
        json!({"type": "auth_result", "success": true})
    );
    assert_ws_lag_refused(&mut ws, "before the shutdown").await;

    // A graceful shutdown waits for the HTTP connections, but an upgraded
    // websocket has left its connection, so serve returns while the
    // session still runs.
    stop.send(()).unwrap();
    tokio::time::timeout(PATIENCE, server)
        .await
        .expect("serve did not return after its shutdown signal")
        .unwrap()
        .unwrap();
    assert_local_refused("serve of a local configuration while a websocket session runs").await;
    assert_ws_lag_refused(&mut ws, "on the session after the shutdown").await;

    ws.close(None).await.unwrap();
    drop(ws);
    assert_local_accepted_once_released("after the websocket session closed").await;
}
