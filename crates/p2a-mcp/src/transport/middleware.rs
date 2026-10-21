//! HTTP middleware: CORS, bearer-token authentication and the request body
//! limit, plus the order in which they wrap the router.

use std::sync::Arc;

use axum::{
    Router,
    extract::{Request, State},
    http::{StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
};
use tower_http::cors::{Any, CorsLayer};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::trace::TraceLayer;

use crate::config::ServerConfig;

/// Build the CORS layer from configuration.
///
/// - `cors_permissive`: allow any origin/method/header (dev only — do NOT use
///   for a public deployment).
/// - `cors_origins` set: allow exactly those origins (recommended for deploys).
/// - neither: a restrictive default `CorsLayer` (no cross-origin access).
pub(crate) fn build_cors(config: &ServerConfig) -> CorsLayer {
    if config.http.cors_permissive {
        CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any)
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
            .allow_headers(Any)
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
/// layers cover every route. CORS is outermost so preflight `OPTIONS` is
/// handled before the body-size limit.
pub(crate) fn apply_global_middleware(router: Router, config: &ServerConfig) -> Router {
    // Shared bearer token, if configured. `None` disables the check (loopback
    // development only; the startup guard forbids this on a public bind).
    let token_state: Option<Arc<str>> = config.http.access_token.as_deref().map(Arc::from);
    router
        // CORS stays outermost so preflight is handled and error responses
        // (including 401s from the auth layer below) carry CORS headers.
        .layer(build_cors(config))
        .layer(middleware::from_fn_with_state(
            token_state,
            require_access_token,
        ))
        .layer(RequestBodyLimitLayer::new(max_request_body_bytes()))
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
