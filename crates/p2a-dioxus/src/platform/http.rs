//! Platform-agnostic HTTP client abstraction
//!
//! - Web: Uses web_sys fetch API
//! - Native: Uses reqwest

use std::collections::HashMap;
use std::sync::{PoisonError, RwLock};

// ============================================================================
// Backend access token (bearer token for `P2A_ACCESS_TOKEN` servers)
// ============================================================================

/// Process-wide backend access token, published by `Settings::load`/`save`.
///
/// Only `ApiClient` and the chat stream read it, and they only send it to the
/// backend; the generic `HttpClient` implementations never add it, so the
/// token cannot reach any other URL. (On wasm32 std's RwLock is the
/// single-threaded implementation, which is fine here.)
static BACKEND_ACCESS_TOKEN: RwLock<Option<String>> = RwLock::new(None);

/// Trim a token and accept it only if it is non-empty and every character is
/// visible ASCII (`'!'..='~'`), so it can be sent in an HTTP header and
/// cannot inject a second header.
pub fn normalize_access_token(raw: &str) -> Option<String> {
    let token = raw.trim();
    (!token.is_empty() && token.chars().all(|c| ('!'..='~').contains(&c)))
        .then(|| token.to_string())
}

/// Publish the backend access token; an empty or invalid token clears it.
///
/// An invalid one (for example from a hand-edited settings file) is logged
/// as a warning, never with its value, so that requests going out without a
/// token while Settings still shows one are explained.
pub fn set_backend_access_token(token: Option<&str>) {
    let normalized = token.and_then(normalize_access_token);
    if normalized.is_none() && token.is_some_and(|t| !t.trim().is_empty()) {
        tracing::warn!(
            "Backend access token ignored: it may contain only visible ASCII characters"
        );
    }
    *BACKEND_ACCESS_TOKEN
        .write()
        .unwrap_or_else(PoisonError::into_inner) = normalized;
}

/// The published backend access token, if any.
pub fn backend_access_token() -> Option<String> {
    BACKEND_ACCESS_TOKEN
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .clone()
}

/// The `Authorization: Bearer <token>` header pair for a token.
pub fn authorization_header(token: Option<&str>) -> Option<(String, String)> {
    token.map(|t| ("Authorization".to_string(), format!("Bearer {t}")))
}

/// User-facing message for a non-2xx backend response. A 401 is explained
/// as a missing or rejected access token; other statuses keep the plain
/// `HTTP error: <status>` form.
pub fn http_status_message(status: u16, token_sent: bool) -> String {
    match (status, token_sent) {
        (401, false) => "The backend requires an access token (HTTP 401). Enter it in Settings under Backend access token.".to_string(),
        (401, true) => "The backend rejected the access token (HTTP 401). Check Settings under Backend access token.".to_string(),
        _ => format!("HTTP error: {status}"),
    }
}

/// How much of a non-JSON error body a message keeps, in bytes.
const ERROR_BODY_BYTES: usize = 300;

/// A server's error as the user sees it: `<message> (<code>)`, or the
/// message alone when there is no code. The one formatter for REST and SSE
/// errors.
pub fn error_text(message: &str, code: Option<&str>) -> String {
    match code.filter(|c| !c.is_empty()) {
        Some(code) => format!("{message} ({code})"),
        None => message.to_string(),
    }
}

/// A non-2xx backend answer: its status, and the server's `error` text and
/// `code` when the body is the JSON error envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiError {
    pub status: u16,
    pub code: Option<String>,
    pub message: String,
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&error_text(&self.message, self.code.as_deref()))
    }
}

impl From<ApiError> for String {
    fn from(e: ApiError) -> Self {
        e.to_string()
    }
}

/// The error envelope of the JSON routes: `{"success":false,"error":..,"code":..}`.
#[derive(serde::Deserialize)]
struct ErrorBody {
    error: Option<String>,
    #[serde(default)]
    code: Option<String>,
}

/// Read a non-2xx answer: the JSON envelope gives the message and code; a
/// 401 keeps the access-token explanation of [`http_status_message`]; a
/// plain-text body (axum's rejections, a proxy page) is kept, cut to
/// [`ERROR_BODY_BYTES`] on a char boundary, and a 413 is explained as a
/// request too large for the server; an empty body gives `HTTP error: <status>`.
pub fn api_error(status: u16, body: &str, token_sent: bool) -> ApiError {
    let envelope = serde_json::from_str::<ErrorBody>(body)
        .ok()
        .and_then(|b| b.error.map(|error| (error, b.code)));
    let code = envelope.as_ref().and_then(|(_, code)| code.clone());
    let message = if status == 401 {
        http_status_message(status, token_sent)
    } else if let Some((error, _)) = envelope {
        p2a_text::ellipsize_bytes(&error, ERROR_BODY_BYTES, "...").into_owned()
    } else {
        let text = body.trim();
        let text = p2a_text::ellipsize_bytes(text, ERROR_BODY_BYTES, "...");
        match (status, text.is_empty()) {
            (_, true) => format!("HTTP error: {status}"),
            (413, false) => {
                format!("The request is too large for this server (HTTP 413: {text})")
            }
            (_, false) => format!("HTTP {status}: {text}"),
        }
    };
    ApiError {
        status,
        code,
        message,
    }
}

/// HTTP response wrapper
#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub status: u16,
    pub body: String,
}

impl HttpResponse {
    pub fn is_ok(&self) -> bool {
        (200..300).contains(&self.status)
    }
}

/// HTTP client error
#[derive(Debug, Clone)]
pub struct HttpError(pub String);

impl std::fmt::Display for HttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for HttpError {}

/// Platform-agnostic HTTP client trait
#[allow(async_fn_in_trait)]
pub trait HttpClient {
    /// Perform a GET request
    async fn get(&self, url: &str) -> Result<HttpResponse, HttpError>;

    /// Perform a POST request with JSON body
    async fn post(&self, url: &str, body: &str) -> Result<HttpResponse, HttpError>;

    /// Perform a PUT request with JSON body
    async fn put(&self, url: &str, body: &str) -> Result<HttpResponse, HttpError>;

    /// Perform a DELETE request
    async fn delete(&self, url: &str) -> Result<HttpResponse, HttpError>;

    /// Perform a request with full control
    async fn request(
        &self,
        url: &str,
        method: &str,
        body: Option<&str>,
        headers: &HashMap<String, String>,
    ) -> Result<HttpResponse, HttpError>;
}

// ============================================================================
// Web implementation (WASM with fetch API)
// ============================================================================

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use wasm_bindgen::JsCast;
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{Request, RequestInit, RequestMode, Response};

    /// Web HTTP client using fetch API
    pub struct WebHttpClient;

    impl WebHttpClient {
        pub fn new() -> Self {
            Self
        }

        async fn fetch_internal(
            &self,
            url: &str,
            method: &str,
            body: Option<&str>,
            headers: &HashMap<String, String>,
        ) -> Result<HttpResponse, HttpError> {
            let opts = RequestInit::new();
            opts.set_method(method);
            opts.set_mode(RequestMode::Cors);

            if let Some(body_str) = body {
                opts.set_body(&JsValue::from_str(body_str));
            }

            let request = Request::new_with_str_and_init(url, &opts)
                .map_err(|e| HttpError(format!("Failed to create request: {e:?}")))?;

            // Set default Content-Type for requests with body
            if body.is_some() {
                request
                    .headers()
                    .set("Content-Type", "application/json")
                    .map_err(|e| HttpError(format!("Failed to set Content-Type: {e:?}")))?;
            }

            // Set custom headers
            for (key, value) in headers {
                request
                    .headers()
                    .set(key, value)
                    .map_err(|e| HttpError(format!("Failed to set header {}: {e:?}", key)))?;
            }

            let window =
                web_sys::window().ok_or_else(|| HttpError("No window object".to_string()))?;
            let resp_value = JsFuture::from(window.fetch_with_request(&request))
                .await
                .map_err(|e| HttpError(format!("Fetch error: {e:?}")))?;

            let response: Response = resp_value
                .dyn_into()
                .map_err(|_| HttpError("Response is not a Response object".to_string()))?;

            let status = response.status();

            let text = JsFuture::from(
                response
                    .text()
                    .map_err(|e| HttpError(format!("Failed to get text: {e:?}")))?,
            )
            .await
            .map_err(|e| HttpError(format!("Failed to read response: {e:?}")))?;

            let body = text
                .as_string()
                .ok_or_else(|| HttpError("Response is not a string".to_string()))?;

            Ok(HttpResponse { status, body })
        }
    }

    impl Default for WebHttpClient {
        fn default() -> Self {
            Self::new()
        }
    }

    impl HttpClient for WebHttpClient {
        async fn get(&self, url: &str) -> Result<HttpResponse, HttpError> {
            self.fetch_internal(url, "GET", None, &HashMap::new()).await
        }

        async fn post(&self, url: &str, body: &str) -> Result<HttpResponse, HttpError> {
            self.fetch_internal(url, "POST", Some(body), &HashMap::new())
                .await
        }

        async fn put(&self, url: &str, body: &str) -> Result<HttpResponse, HttpError> {
            self.fetch_internal(url, "PUT", Some(body), &HashMap::new())
                .await
        }

        async fn delete(&self, url: &str) -> Result<HttpResponse, HttpError> {
            self.fetch_internal(url, "DELETE", None, &HashMap::new())
                .await
        }

        async fn request(
            &self,
            url: &str,
            method: &str,
            body: Option<&str>,
            headers: &HashMap<String, String>,
        ) -> Result<HttpResponse, HttpError> {
            self.fetch_internal(url, method, body, headers).await
        }
    }
}

// ============================================================================
// Native implementation (reqwest)
// ============================================================================

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::*;

    /// Native HTTP client using reqwest
    pub struct NativeHttpClient {
        client: reqwest::Client,
    }

    impl NativeHttpClient {
        pub fn new() -> Self {
            Self {
                client: reqwest::Client::new(),
            }
        }
    }

    impl Default for NativeHttpClient {
        fn default() -> Self {
            Self::new()
        }
    }

    impl HttpClient for NativeHttpClient {
        async fn get(&self, url: &str) -> Result<HttpResponse, HttpError> {
            let response = self
                .client
                .get(url)
                .send()
                .await
                .map_err(|e| HttpError(format!("Request failed: {}", e)))?;

            let status = response.status().as_u16();
            let body = response
                .text()
                .await
                .map_err(|e| HttpError(format!("Failed to read response: {}", e)))?;

            Ok(HttpResponse { status, body })
        }

        async fn post(&self, url: &str, body: &str) -> Result<HttpResponse, HttpError> {
            let response = self
                .client
                .post(url)
                .header("Content-Type", "application/json")
                .body(body.to_string())
                .send()
                .await
                .map_err(|e| HttpError(format!("Request failed: {}", e)))?;

            let status = response.status().as_u16();
            let body = response
                .text()
                .await
                .map_err(|e| HttpError(format!("Failed to read response: {}", e)))?;

            Ok(HttpResponse { status, body })
        }

        async fn put(&self, url: &str, body: &str) -> Result<HttpResponse, HttpError> {
            let response = self
                .client
                .put(url)
                .header("Content-Type", "application/json")
                .body(body.to_string())
                .send()
                .await
                .map_err(|e| HttpError(format!("Request failed: {}", e)))?;

            let status = response.status().as_u16();
            let body = response
                .text()
                .await
                .map_err(|e| HttpError(format!("Failed to read response: {}", e)))?;

            Ok(HttpResponse { status, body })
        }

        async fn delete(&self, url: &str) -> Result<HttpResponse, HttpError> {
            let response = self
                .client
                .delete(url)
                .send()
                .await
                .map_err(|e| HttpError(format!("Request failed: {}", e)))?;

            let status = response.status().as_u16();
            let body = response
                .text()
                .await
                .map_err(|e| HttpError(format!("Failed to read response: {}", e)))?;

            Ok(HttpResponse { status, body })
        }

        async fn request(
            &self,
            url: &str,
            method: &str,
            body: Option<&str>,
            headers: &HashMap<String, String>,
        ) -> Result<HttpResponse, HttpError> {
            let method = reqwest::Method::from_bytes(method.as_bytes())
                .map_err(|e| HttpError(format!("Invalid method: {}", e)))?;

            let mut request = self.client.request(method, url);

            if let Some(body_str) = body {
                request = request
                    .header("Content-Type", "application/json")
                    .body(body_str.to_string());
            }

            for (key, value) in headers {
                request = request.header(key, value);
            }

            let response = request
                .send()
                .await
                .map_err(|e| HttpError(format!("Request failed: {}", e)))?;

            let status = response.status().as_u16();
            let body = response
                .text()
                .await
                .map_err(|e| HttpError(format!("Failed to read response: {}", e)))?;

            Ok(HttpResponse { status, body })
        }
    }
}

// ============================================================================
// Platform-specific type aliases
// ============================================================================

#[cfg(target_arch = "wasm32")]
pub type PlatformHttpClient = web::WebHttpClient;

#[cfg(not(target_arch = "wasm32"))]
pub type PlatformHttpClient = native::NativeHttpClient;

/// Create a new platform-appropriate HTTP client
pub fn create_http_client() -> PlatformHttpClient {
    PlatformHttpClient::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_access_token_rules() {
        assert_eq!(
            normalize_access_token("  abc.DEF-123_~+/=  ").as_deref(),
            Some("abc.DEF-123_~+/=")
        );
        for bad in ["", "   ", "abc\r\nX-Other: 1", "tök", "a b"] {
            assert_eq!(normalize_access_token(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn bearer_header_pair() {
        assert_eq!(
            authorization_header(Some("s3cret")),
            Some(("Authorization".into(), "Bearer s3cret".into()))
        );
        assert_eq!(authorization_header(None), None);
    }

    #[test]
    fn status_401_explains_the_token() {
        assert!(http_status_message(401, false).contains("requires an access token"));
        assert!(http_status_message(401, true).contains("rejected the access token"));
        assert_eq!(http_status_message(503, false), "HTTP error: 503");
    }

    #[test]
    fn api_error_reads_the_servers_body() {
        let e = api_error(
            503,
            r#"{"success":false,"error":"Maximum number of sessions reached","code":"capacity"}"#,
            false,
        );
        assert_eq!(e.status, 503);
        assert_eq!(e.message, "Maximum number of sessions reached");
        assert_eq!(e.code.as_deref(), Some("capacity"));
        assert_eq!(
            e.to_string(),
            "Maximum number of sessions reached (capacity)"
        );

        let e = api_error(
            410,
            r#"{"success":false,"error":"Session expired","code":"session_expired"}"#,
            false,
        );
        assert_eq!(String::from(e), "Session expired (session_expired)");

        let e = api_error(
            400,
            r#"{"success":false,"error":"Missing X-P2A-Session-Id header","code":"invalid_arguments"}"#,
            false,
        );
        assert_eq!(e.code.as_deref(), Some("invalid_arguments"));
        assert!(e.message.contains("X-P2A-Session-Id"), "{e}");

        // A JSON error without a code keeps the message alone.
        let e = api_error(404, r#"{"success":false,"error":"Not found"}"#, false);
        assert_eq!((e.message.as_str(), e.code), ("Not found", None));
    }

    #[test]
    fn api_error_explains_plain_text_bodies() {
        for body in [
            "length limit exceeded",
            "Failed to buffer the request body: length limit exceeded",
        ] {
            let e = api_error(413, body, false);
            assert_eq!(e.code, None);
            assert!(e.message.contains("too large"), "{e}");
            assert!(e.message.contains(body), "{e}");
        }
        // axum's plain-text rejection of a malformed JSON body.
        let e = api_error(
            422,
            "Failed to deserialize the JSON body into the target type: missing field `name`",
            false,
        );
        assert_eq!(
            e.message,
            "HTTP 422: Failed to deserialize the JSON body into the target type: missing field `name`"
        );
        assert_eq!(api_error(502, "", false).message, "HTTP error: 502");
        assert_eq!(api_error(502, "  \n", false).message, "HTTP error: 502");

        // A long HTML page is cut on a char boundary.
        let html = format!("<html><body>{}</body></html>", "é".repeat(2048));
        let e = api_error(500, &html, false);
        assert!(
            e.message.len() <= "HTTP 500: ".len() + 300,
            "{}",
            e.message.len()
        );
        assert!(e.message.starts_with("HTTP 500: <html><body>é"), "{e}");
    }

    #[test]
    fn api_error_keeps_the_token_messages_for_401() {
        assert!(
            api_error(401, "", false)
                .message
                .contains("requires an access token")
        );
        assert!(
            api_error(401, "Unauthorized", true)
                .message
                .contains("rejected the access token")
        );
    }

    #[test]
    fn error_text_appends_the_code() {
        assert_eq!(
            error_text("Session expired", Some("session_expired")),
            "Session expired (session_expired)"
        );
        assert_eq!(error_text("Session expired", None), "Session expired");
        assert_eq!(error_text("Session expired", Some("")), "Session expired");
    }

    /// Records the message of every WARN event on the thread it is the
    /// default subscriber of.
    #[derive(Clone, Default)]
    struct WarnCapture(std::sync::Arc<std::sync::Mutex<Vec<String>>>);

    impl WarnCapture {
        fn take(&self) -> Vec<String> {
            std::mem::take(&mut *self.0.lock().unwrap())
        }
    }

    struct MessageVisitor<'a>(&'a mut String);

    impl tracing::field::Visit for MessageVisitor<'_> {
        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            if field.name() == "message" {
                *self.0 = format!("{value:?}");
            }
        }
    }

    impl tracing::Subscriber for WarnCapture {
        fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
            true
        }
        fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }
        fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
        fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
        fn event(&self, event: &tracing::Event<'_>) {
            if *event.metadata().level() == tracing::Level::WARN {
                let mut message = String::new();
                event.record(&mut MessageVisitor(&mut message));
                self.0.lock().unwrap().push(message);
            }
        }
        fn enter(&self, _: &tracing::span::Id) {}
        fn exit(&self, _: &tracing::span::Id) {}
    }

    #[test]
    fn backend_token_store_round_trip() {
        // the only test that touches the global
        let warnings = WarnCapture::default();
        tracing::subscriber::with_default(warnings.clone(), || {
            set_backend_access_token(Some("  t0k  "));
            assert_eq!(backend_access_token().as_deref(), Some("t0k"));
            assert!(warnings.take().is_empty());

            // A non-empty but invalid token is cleared, with a warning that
            // does not carry the value.
            set_backend_access_token(Some("bad\ntoken"));
            assert_eq!(backend_access_token(), None);
            let warned = warnings.take();
            assert_eq!(warned.len(), 1, "{warned:?}");
            assert!(
                warned[0].contains("Backend access token ignored"),
                "{warned:?}"
            );
            assert!(!warned[0].contains("bad"), "{warned:?}");

            // No token at all is not worth a warning.
            for none in [None, Some(""), Some("   ")] {
                set_backend_access_token(none);
                assert_eq!(backend_access_token(), None);
            }
            assert!(warnings.take().is_empty());
        });
    }
}
