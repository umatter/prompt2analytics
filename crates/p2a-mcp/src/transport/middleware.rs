//! HTTP middleware: CORS, bearer-token authentication, the request body
//! limit and the local-mode Host check, plus the order in which they wrap
//! the router.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Request, State},
    http::{HeaderName, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
};
use tower_http::cors::{AllowHeaders, Any, CorsLayer};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::trace::TraceLayer;

use super::http::ApiResponse;
use crate::config::{DeploymentMode, ResourceProfile, ServerConfig};

/// Request headers a browser client may send cross-origin: the bearer token,
/// JSON bodies, `Accept` and the session header. A `*` wildcard would not
/// cover `Authorization`, so they are listed.
fn allowed_request_headers() -> [HeaderName; 4] {
    [
        header::AUTHORIZATION,
        header::CONTENT_TYPE,
        header::ACCEPT,
        HeaderName::from_static("x-p2a-session-id"),
    ]
}

/// Build the CORS layer from configuration.
///
/// - `cors_permissive`: allow any origin and method, and mirror the request
///   headers (dev only — do NOT use for a public deployment).
/// - `cors_origins` set: allow exactly those origins with the headers from
///   [`allowed_request_headers`] (recommended for deploys).
/// - neither: a restrictive default `CorsLayer` (no cross-origin access).
///
/// Both configured variants expose `Retry-After`, so browser clients can
/// honour the governor's back-off hints.
pub(crate) fn build_cors(config: &ServerConfig) -> CorsLayer {
    if config.http.cors_permissive {
        CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(AllowHeaders::mirror_request())
            .expose_headers([header::RETRY_AFTER])
    } else if !config.http.cors_origins.is_empty() {
        let origins: Vec<_> = config
            .http
            .cors_origins
            .iter()
            .filter_map(|s| s.parse().ok())
            .collect();
        CorsLayer::new()
            .allow_origin(origins)
            .allow_methods(Any)
            .allow_headers(allowed_request_headers())
            .expose_headers([header::RETRY_AFTER])
    } else {
        CorsLayer::new()
    }
}

/// The environment variable that sets [`max_request_body_bytes`], in MiB.
pub(crate) const MAX_HTTP_BODY_VAR: &str = "P2A_MAX_HTTP_BODY_MB";

/// Maximum accepted HTTP request body size, in bytes.
///
/// Caps memory use from oversized requests (e.g. large base64 dataset uploads)
/// on a publicly reachable backend. Defaults to 32 MiB; override with the
/// `P2A_MAX_HTTP_BODY_MB` environment variable. This only limits request
/// bodies and does not affect SSE/streaming responses. The routes that parse
/// a JSON body take one up to this size, except in the hosted profile
/// ([`parsed_body_bytes`]).
pub(crate) fn max_request_body_bytes() -> usize {
    const DEFAULT_MB: usize = 32;
    let mb = std::env::var(MAX_HTTP_BODY_VAR)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&mb| mb > 0)
        .unwrap_or(DEFAULT_MB);
    mb * 1024 * 1024
}

/// The largest body the hosted profile parses: 2 MiB, axum's own default for
/// its body extractors, which the hosted server has always applied.
///
/// A route parses its whole body (a tool call's arguments into
/// `serde_json::Value`) before the tool governor's memory admission sees the
/// call, and the parsed value can take much more memory than the body. The
/// hosted profile keeps the 2 MiB limit until that parse is bounded; raising
/// it to the body cap is a separate decision.
const HOSTED_PARSED_BODY_BYTES: usize = 2 * 1024 * 1024;

/// The largest body the routes that parse one (axum's `Json`, `Bytes`,
/// `String` and `Form` extractors) accept: the body cap `body_cap`, or in the
/// hosted profile [`HOSTED_PARSED_BODY_BYTES`] when that is smaller.
fn parsed_body_bytes(profile: ResourceProfile, body_cap: usize) -> usize {
    match profile {
        ResourceProfile::Off => body_cap,
        ResourceProfile::Hosted => body_cap.min(HOSTED_PARSED_BODY_BYTES),
    }
}

/// What building the router logs in the hosted profile when the body cap
/// `body_cap` is above [`HOSTED_PARSED_BODY_BYTES`]: an operator who raised
/// `P2A_MAX_HTTP_BODY_MB` learns that HTTP JSON bodies stay at the hosted
/// limit, which no setting raises. `None` when the two limits are the same.
fn hosted_parse_limit_note(profile: ResourceProfile, body_cap: usize) -> Option<String> {
    let parsed = parsed_body_bytes(profile, body_cap);
    (profile == ResourceProfile::Hosted && parsed < body_cap).then(|| {
        format!(
            "hosted profile: HTTP JSON request bodies are limited to {parsed} bytes; \
             {MAX_HTTP_BODY_VAR} ({body_cap} bytes) still bounds declared body lengths \
             and websocket messages"
        )
    })
}

/// Apply global middleware to the fully-composed router.
///
/// Must run AFTER all routes (including nested ones) are registered so the
/// layers cover every route. In axum each `.layer` call wraps everything
/// before it, so the order below is innermost first: the body limits, the
/// bearer token, the local-mode Host check, then CORS and tracing outermost.
/// With CORS outside the other layers, preflight requests are answered
/// before authentication, and 401, 413 and 421 responses carry CORS headers
/// a browser can read.
///
/// There are two body limits. `RequestBodyLimitLayer` holds every body to
/// the cap ([`max_request_body_bytes`]): a body declared longer answers 413
/// (plain text `length limit exceeded`) before it is read, and one without a
/// declared length is cut off at the cap. axum's body extractors apply a
/// limit of their own, 2 MiB unless `DefaultBodyLimit` sets it, and refuse a
/// longer body with 413 (`Failed to buffer the request body: length limit
/// exceeded`); it is set to [`parsed_body_bytes`], so outside the hosted
/// profile the cap is the limit of every route. In the hosted profile with a
/// cap above 2 MiB, building the router logs the lower JSON limit once
/// ([`hosted_parse_limit_note`]).
pub(crate) fn apply_global_middleware(router: Router, config: &ServerConfig) -> Router {
    // Shared bearer token, if configured. `None` disables the check (loopback
    // development only; the startup guard forbids this on a public bind).
    let token_state: Option<Arc<str>> = config.http.access_token.as_deref().map(Arc::from);
    let body_cap = max_request_body_bytes();
    let parsed = parsed_body_bytes(config.resource_profile(), body_cap);
    if let Some(note) = hosted_parse_limit_note(config.resource_profile(), body_cap) {
        tracing::info!("{note}");
    }
    let router = router
        .layer(DefaultBodyLimit::max(parsed))
        .layer(RequestBodyLimitLayer::new(body_cap))
        .layer(middleware::from_fn_with_state(
            token_state,
            require_access_token,
        ));
    // DNS-rebinding guard, local mode only.
    let router = if config.deployment_mode() == DeploymentMode::Local {
        router.layer(middleware::from_fn(require_local_host))
    } else {
        router
    };
    router
        .layer(build_cors(config))
        .layer(TraceLayer::new_for_http())
}

/// Middleware enforcing a shared bearer token on all routes except `/health`
/// and CORS preflight (`OPTIONS`). When no token is configured the request is
/// passed through unchanged.
pub(crate) async fn require_access_token(
    State(expected): State<Option<Arc<str>>>,
    req: Request,
    next: Next,
) -> Response {
    use axum::http::Method;

    // Health checks and CORS preflight are always allowed through.
    if req.method() == Method::OPTIONS || req.uri().path() == "/health" {
        return next.run(req).await;
    }

    let Some(expected) = expected.as_deref() else {
        return next.run(req).await;
    };

    let provided = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::trim);

    let authorized = provided
        .map(|p| constant_time_eq(p.as_bytes(), expected.as_bytes()))
        .unwrap_or(false);

    if authorized {
        next.run(req).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            [(header::WWW_AUTHENTICATE, "Bearer")],
            "missing or invalid bearer token",
        )
            .into_response()
    }
}

/// Length-checked, constant-time byte comparison to avoid leaking the token
/// via response timing.
pub(crate) fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Whether `host` (a Host header or URI authority) names this machine:
/// `localhost` (any case) or a loopback IP literal, with or without a port.
pub(crate) fn is_local_host(host: &str) -> bool {
    let name = if let Some(rest) = host.strip_prefix('[') {
        // Bracketed IPv6 literal: "[::1]" or "[::1]:8080".
        match rest.split_once(']') {
            Some((inside, _)) => inside,
            None => return false,
        }
    } else {
        match host.rsplit_once(':') {
            Some((name, port)) if !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit()) => {
                name
            }
            _ => host,
        }
    };
    if name.eq_ignore_ascii_case("localhost") {
        return true;
    }
    name.parse::<std::net::IpAddr>()
        .is_ok_and(|ip| ip.is_loopback())
}

/// DNS-rebinding guard for a local server: a request addressed to any host
/// other than `localhost` or a loopback address gets 421. A request without
/// a host (not a browser) passes.
pub(crate) async fn require_local_host(req: Request, next: Next) -> Response {
    let host = req.uri().host().map(str::to_string).or_else(|| {
        req.headers()
            .get(header::HOST)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string)
    });
    match host {
        Some(h) if !is_local_host(&h) => (
            StatusCode::MISDIRECTED_REQUEST,
            Json(ApiResponse::<()>::error_with_code(
                "This server only accepts requests addressed to localhost. To serve other \
                 hostnames, configure an access token or --allow-unauthenticated.",
                "misdirected",
                None,
            )),
        )
            .into_response(),
        _ => next.run(req).await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_host_detection() {
        for ok in [
            "localhost:8081",
            "LOCALHOST",
            "127.0.0.1:8080",
            "[::1]:8080",
            "127.0.0.5",
            "[::1]",
        ] {
            assert!(is_local_host(ok), "{ok}");
        }
        for bad in [
            "evil.example",
            "localhost.evil.example",
            "0.0.0.0:8080",
            "127.0.0.1.nip.io",
            "",
            "[::]:80",
        ] {
            assert!(!is_local_host(bad), "{bad}");
        }
    }

    #[test]
    fn parsed_bodies_follow_the_cap_except_above_2_mib_in_the_hosted_profile() {
        const MIB: usize = 1024 * 1024;
        assert_eq!(parsed_body_bytes(ResourceProfile::Off, 32 * MIB), 32 * MIB);
        assert_eq!(parsed_body_bytes(ResourceProfile::Off, MIB), MIB);
        assert_eq!(
            parsed_body_bytes(ResourceProfile::Hosted, 32 * MIB),
            2 * MIB
        );
        assert_eq!(parsed_body_bytes(ResourceProfile::Hosted, MIB), MIB);
        // axum's own extractor default, which the hosted server applied
        // before the limit was set explicitly.
        assert_eq!(HOSTED_PARSED_BODY_BYTES, 2_097_152);
    }

    /// A shared buffer the log lines of a test subscriber go to.
    #[derive(Clone, Default)]
    struct Captured(Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for Captured {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// In the hosted profile, a body cap above 2 MiB does not raise the JSON
    /// body limit, and building the router says so once: the note names
    /// both limits and the variable that sets the cap. With the two limits
    /// equal, or outside the hosted profile, there is nothing to say.
    #[test]
    fn a_hosted_router_logs_a_json_body_limit_below_the_cap_once() {
        const MIB: usize = 1024 * 1024;
        let note = hosted_parse_limit_note(ResourceProfile::Hosted, 32 * MIB)
            .expect("a hosted cap above 2 MiB is noted");
        for part in [
            "2097152 bytes",
            "33554432 bytes",
            MAX_HTTP_BODY_VAR,
            "websocket",
        ] {
            assert!(note.contains(part), "{part}: {note}");
        }
        assert_eq!(
            hosted_parse_limit_note(ResourceProfile::Hosted, 2 * MIB),
            None
        );
        assert_eq!(hosted_parse_limit_note(ResourceProfile::Hosted, MIB), None);
        assert_eq!(
            hosted_parse_limit_note(ResourceProfile::Off, 32 * MIB),
            None
        );

        let captured = Captured::default();
        let writer = captured.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_max_level(tracing::Level::INFO)
            .with_writer(move || writer.clone())
            .finish();
        // Another test thread may register the note's callsite while no
        // subscriber is in place, which caches it as disabled: with a second
        // dispatcher registered, recompute every callsite's interest with
        // this subscriber in place (as websocket.rs's log test does).
        let _second = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
        let _guard = tracing::subscriber::set_default(subscriber);
        tracing::callsite::rebuild_interest_cache();
        let log = || String::from_utf8(captured.0.lock().unwrap().clone()).unwrap();

        let mut config = ServerConfig::default();
        config.limits.profile = Some(ResourceProfile::Hosted);
        assert_eq!(config.resource_profile(), ResourceProfile::Hosted);
        let _router = apply_global_middleware(Router::new(), &config);
        // The cap this test process runs with: the default 32 MiB unless its
        // environment sets another.
        match hosted_parse_limit_note(ResourceProfile::Hosted, max_request_body_bytes()) {
            Some(note) => assert_eq!(log().matches(&note).count(), 1, "{}", log()),
            None => assert!(!log().contains("hosted profile"), "{}", log()),
        }

        let before = log();
        config.limits.profile = None;
        assert_eq!(config.resource_profile(), ResourceProfile::Off);
        let _router = apply_global_middleware(Router::new(), &config);
        assert_eq!(log(), before);
    }

    #[test]
    fn constant_time_eq_compares_bytes() {
        assert!(constant_time_eq(b"token", b"token"));
        assert!(!constant_time_eq(b"token", b"tokem"));
        assert!(!constant_time_eq(b"token", b"tok"));
    }
}
