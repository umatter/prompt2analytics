//! Platform-agnostic HTTP client abstraction
//!
//! - Web: Uses web_sys fetch API
//! - Native: Uses reqwest

use std::collections::HashMap;

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
    use wasm_bindgen::prelude::*;
    use wasm_bindgen::JsCast;
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

            let window = web_sys::window().ok_or_else(|| HttpError("No window object".to_string()))?;
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
