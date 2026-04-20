//! Retry logic for transient API errors (429, 5xx, network).

use super::LlmError;
use std::time::Duration;

/// Configuration for retry behavior on transient errors.
pub struct RetryConfig {
    /// Maximum number of retries (default: 3)
    pub max_retries: u32,
    /// Initial delay in milliseconds (default: 1000)
    pub initial_delay_ms: u64,
    /// Maximum delay in milliseconds (default: 30000)
    pub max_delay_ms: u64,
    /// Backoff multiplier (default: 2.0)
    pub backoff_factor: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay_ms: 1000,
            max_delay_ms: 30000,
            backoff_factor: 2.0,
        }
    }
}

/// Send an HTTP request with retry on transient errors.
///
/// The `request_builder_fn` closure is called each retry because `RequestBuilder`
/// is consumed by `.send()`.
///
/// Retries on:
/// - 429 (rate limited): respects Retry-After header
/// - 500, 502, 503, 504 (server errors): exponential backoff
/// - Network/timeout errors: exponential backoff
///
/// Does NOT retry on:
/// - Other 4xx errors (client errors)
/// - Successful responses (even with error bodies - caller handles those)
pub async fn send_with_retry<F>(
    request_builder_fn: F,
    config: &RetryConfig,
) -> Result<reqwest::Response, LlmError>
where
    F: Fn() -> reqwest::RequestBuilder,
{
    let mut last_error: Option<LlmError> = None;
    let mut delay_ms = config.initial_delay_ms;

    for attempt in 0..=config.max_retries {
        if attempt > 0 {
            tracing::info!(
                attempt = attempt,
                delay_ms = delay_ms,
                "Retrying API request"
            );
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
        }

        match request_builder_fn().send().await {
            Ok(response) => {
                let status = response.status();

                if status.is_success() || status.is_redirection() {
                    return Ok(response);
                }

                if status.as_u16() == 429 {
                    // Rate limited - check for Retry-After header
                    let retry_after = response
                        .headers()
                        .get("retry-after")
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.parse::<u64>().ok());

                    let body = response.text().await.unwrap_or_default();
                    last_error = Some(LlmError::ApiError(format!("Rate limited (429): {}", body)));

                    if attempt < config.max_retries {
                        delay_ms = if let Some(seconds) = retry_after {
                            seconds * 1000
                        } else {
                            (delay_ms as f64 * config.backoff_factor) as u64
                        };
                        delay_ms = delay_ms.min(config.max_delay_ms);
                        continue;
                    }
                } else if status.is_server_error() {
                    // 5xx - retry with backoff
                    let body = response.text().await.unwrap_or_default();
                    last_error = Some(LlmError::ApiError(format!(
                        "Server error ({}): {}",
                        status, body
                    )));

                    if attempt < config.max_retries {
                        delay_ms = ((delay_ms as f64) * config.backoff_factor) as u64;
                        delay_ms = delay_ms.min(config.max_delay_ms);
                        continue;
                    }
                } else {
                    // Other status codes (4xx except 429) - don't retry, return as-is
                    return Ok(response);
                }
            }
            Err(e) => {
                // Network/timeout error - retry
                last_error = Some(LlmError::NetworkError(e.to_string()));

                if attempt < config.max_retries {
                    delay_ms = ((delay_ms as f64) * config.backoff_factor) as u64;
                    delay_ms = delay_ms.min(config.max_delay_ms);
                    continue;
                }
            }
        }
    }

    Err(last_error
        .unwrap_or_else(|| LlmError::NetworkError("Unknown error after retries".to_string())))
}
