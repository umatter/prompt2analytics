//! Retry policy and response handling for LLM provider calls.
//!
//! Only failures that cannot have reached the provider, or that the provider
//! asks to be retried, are retried: connect errors raised before the request
//! was sent, HTTP 429 and HTTP 5xx. A timed-out completion is never sent
//! again, since the provider may still be working on (and billing) it.
//! Upstream bodies are never put into an error: [`upstream_error`] logs at
//! most 512 characters at `debug` and returns a fixed message.

use super::LlmError;
use serde::de::DeserializeOwned;
use std::time::Duration;

/// The largest response body read from a provider (8 MiB).
pub(crate) const MAX_UPSTREAM_BODY_BYTES: usize = 8 * 1024 * 1024;

/// How much of an error body is read for the debug log.
const ERROR_BODY_READ_BYTES: usize = 4_096;

/// How many characters of an error body are logged.
const ERROR_BODY_LOG_CHARS: usize = 512;

/// Configuration for retry behavior on transient errors.
pub struct RetryConfig {
    /// Maximum number of retries after the first attempt (default: 2, so at
    /// most 3 attempts)
    pub max_retries: u32,
    /// Delay before the first retry in milliseconds (default: 1000)
    pub initial_delay_ms: u64,
    /// Maximum delay in milliseconds, also the cap on `Retry-After`
    /// (default: 30000)
    pub max_delay_ms: u64,
    /// Backoff multiplier applied after each wait (default: 2.0)
    pub backoff_factor: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 2,
            initial_delay_ms: 1000,
            max_delay_ms: 30_000,
            backoff_factor: 2.0,
        }
    }
}

/// Send an HTTP request, retrying transient failures.
///
/// The `request_builder_fn` closure is called for every attempt because
/// `RequestBuilder` is consumed by `.send()`.
///
/// Retried, while attempts remain:
/// - connect errors (including connect timeouts), which happen before the
///   request is sent;
/// - HTTP 429 and 5xx, honouring an integer `Retry-After` (capped at
///   `max_delay_ms`) and otherwise backing off exponentially.
///
/// Not retried:
/// - an egress refusal ([`LlmError::EgressDenied`]);
/// - a timeout after the request was sent ([`LlmError::Timeout`]);
/// - any other transport error ([`LlmError::NetworkError`], without the URL);
/// - success, redirects and other 4xx, and a 429/5xx on the last attempt:
///   the response is returned, and the caller turns a non-success status
///   into [`LlmError::Upstream`] with [`upstream_error`].
pub async fn send_with_retry<F>(
    request_builder_fn: F,
    config: &RetryConfig,
) -> Result<reqwest::Response, LlmError>
where
    F: Fn() -> reqwest::RequestBuilder,
{
    let attempts = config.max_retries.saturating_add(1);
    let mut delay_ms = config.initial_delay_ms;
    let mut attempt: u32 = 0;

    loop {
        attempt += 1;
        let last = attempt >= attempts;

        let wait_ms = match request_builder_fn().send().await {
            Err(e) => {
                if let Some(denied) = super::egress::find_egress_error(&e) {
                    return Err(LlmError::EgressDenied(denied.0.clone()));
                }
                if e.is_connect() {
                    if last {
                        return Err(LlmError::NetworkError(e.without_url().to_string()));
                    }
                    delay_ms
                } else if e.is_timeout() {
                    return Err(LlmError::Timeout);
                } else {
                    return Err(LlmError::NetworkError(e.without_url().to_string()));
                }
            }
            Ok(response) => {
                let status = response.status();
                let retryable = status.as_u16() == 429 || status.is_server_error();
                if !retryable || last {
                    return Ok(response);
                }
                let retry_after = response
                    .headers()
                    .get(reqwest::header::RETRY_AFTER)
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.trim().parse::<u64>().ok());
                // Drop the response without reading its body.
                drop(response);
                match retry_after {
                    Some(secs) => secs.saturating_mul(1000).min(config.max_delay_ms),
                    None => delay_ms,
                }
            }
        };

        tracing::info!(
            attempt = attempt + 1,
            delay_ms = wait_ms,
            "Retrying LLM provider request"
        );
        tokio::time::sleep(Duration::from_millis(wait_ms)).await;
        delay_ms = ((delay_ms as f64) * config.backoff_factor) as u64;
        delay_ms = delay_ms.min(config.max_delay_ms);
    }
}

/// Turn a non-success provider response into [`LlmError::Upstream`].
///
/// Reads at most 4,096 bytes of the body and logs the first 512 characters
/// at `debug`; the body never reaches the caller.
pub(crate) async fn upstream_error(
    provider: &'static str,
    mut response: reqwest::Response,
) -> LlmError {
    let status = response.status().as_u16();
    let mut body: Vec<u8> = Vec::new();
    while body.len() < ERROR_BODY_READ_BYTES {
        match response.chunk().await {
            Ok(Some(chunk)) => {
                let take = (ERROR_BODY_READ_BYTES - body.len()).min(chunk.len());
                body.extend_from_slice(&chunk[..take]);
            }
            _ => break,
        }
    }
    let preview: String = String::from_utf8_lossy(&body)
        .chars()
        .take(ERROR_BODY_LOG_CHARS)
        .collect();
    tracing::debug!(provider, status, body = %preview, "LLM provider returned an error status");
    LlmError::Upstream { provider, status }
}

/// Read a provider's JSON body, refusing it past `cap` bytes.
pub(crate) async fn read_json_capped<T: DeserializeOwned>(
    mut response: reqwest::Response,
    cap: usize,
) -> Result<T, LlmError> {
    if response
        .content_length()
        .is_some_and(|len| len > cap as u64)
    {
        return Err(LlmError::ResponseTooLarge(cap));
    }
    let mut body: Vec<u8> = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        if body.len() + chunk.len() > cap {
            return Err(LlmError::ResponseTooLarge(cap));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(serde_json::from_slice(&body)?)
}

/// Count `received` more streamed bytes against the response cap.
pub(crate) fn count_streamed(total: &mut usize, received: usize) -> Result<(), LlmError> {
    *total = total.saturating_add(received);
    if *total > MAX_UPSTREAM_BODY_BYTES {
        Err(LlmError::ResponseTooLarge(MAX_UPSTREAM_BODY_BYTES))
    } else {
        Ok(())
    }
}
