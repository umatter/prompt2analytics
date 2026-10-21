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

use axum::Router;
use axum::routing::get;
use p2a_mcp::llm::LlmError;
use p2a_mcp::llm::egress::{EgressPolicy, GuardedResolver, Lookup, guarded_client_builder};
use reqwest::dns::Resolve;

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
