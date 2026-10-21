//! HTTP middleware: CORS, bearer-token authentication, the request body
//! limit and the local-mode Host check, plus the order in which they wrap
//! the router.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Request, State},
    http::{HeaderName, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
};
use tower_http::cors::{AllowHeaders, Any, CorsLayer};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::trace::TraceLayer;

use super::http::ApiResponse;
use crate::config::{DeploymentMode, ServerConfig};

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

/// Maximum accepted HTTP request body size, in bytes.
///
/// Caps memory use from oversized requests (e.g. large base64 dataset uploads)
/// on a publicly reachable backend. Defaults to 32 MiB; override with the
/// `P2A_MAX_HTTP_BODY_MB` environment variable. This only limits request
/// bodies and does not affect SSE/streaming responses.
pub(crate) fn max_request_body_bytes() -> usize {
    const DEFAULT_MB: usize = 32;
    let mb = std::env::var("P2A_MAX_HTTP_BODY_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&mb| mb > 0)
        .unwrap_or(DEFAULT_MB);
    mb * 1024 * 1024
}

/// Apply global middleware to the fully-composed router.
///
/// Must run AFTER all routes (including nested ones) are registered so the
/// layers cover every route. In axum each `.layer` call wraps everything
/// before it, so the order below is innermost first: the body limit, the
/// bearer token, the local-mode Host check, then CORS and tracing outermost.
/// With CORS outside the other layers, preflight requests are answered
/// before authentication, and 401, 413 and 421 responses carry CORS headers
/// a browser can read.
pub(crate) fn apply_global_middleware(router: Router, config: &ServerConfig) -> Router {
    // Shared bearer token, if configured. `None` disables the check (loopback
    // development only; the startup guard forbids this on a public bind).
    let token_state: Option<Arc<str>> = config.http.access_token.as_deref().map(Arc::from);
    let router = router
        .layer(RequestBodyLimitLayer::new(max_request_body_bytes()))
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
    fn constant_time_eq_compares_bytes() {
        assert!(constant_time_eq(b"token", b"token"));
        assert!(!constant_time_eq(b"token", b"tokem"));
        assert!(!constant_time_eq(b"token", b"tok"));
    }
}
