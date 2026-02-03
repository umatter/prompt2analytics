//! Audit logging for MCP tool calls.
//!
//! Provides comprehensive logging of all tool invocations for security auditing
//! and privacy verification. Each log entry includes:
//! - Timestamp (ISO 8601)
//! - Session ID
//! - Tool name
//! - Arguments (sanitized to exclude file contents and large data)
//! - Success/failure status
//! - Execution duration
//! - Result summary (truncated)

use chrono::{DateTime, Utc};
use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::config::AuditConfig;

/// Audit log entry for a single tool invocation.
#[derive(Debug, Clone, Serialize)]
pub struct AuditEntry {
    /// ISO 8601 timestamp
    pub timestamp: DateTime<Utc>,
    /// Session identifier
    pub session_id: String,
    /// Name of the tool invoked
    pub tool_name: String,
    /// Sanitized arguments (file contents and large data removed)
    pub arguments: serde_json::Value,
    /// Whether the tool call succeeded
    pub success: bool,
    /// Execution time in milliseconds
    pub duration_ms: u64,
    /// Truncated result summary (max 500 chars)
    pub result_summary: String,
    /// Client IP address if available
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_ip: Option<String>,
}

/// Thread-safe audit logger.
pub struct AuditLogger {
    writer: Arc<Mutex<Option<BufWriter<File>>>>,
    enabled: bool,
}

impl AuditLogger {
    /// Create a new audit logger from configuration.
    pub fn new(config: &AuditConfig) -> std::io::Result<Self> {
        if !config.enabled {
            return Ok(Self {
                writer: Arc::new(Mutex::new(None)),
                enabled: false,
            });
        }

        let path = Path::new(&config.path);
        let file = OpenOptions::new().create(true).append(true).open(path)?;

        tracing::info!(path = %config.path, "Audit logging enabled");

        Ok(Self {
            writer: Arc::new(Mutex::new(Some(BufWriter::new(file)))),
            enabled: true,
        })
    }

    /// Create a disabled audit logger.
    pub fn disabled() -> Self {
        Self {
            writer: Arc::new(Mutex::new(None)),
            enabled: false,
        }
    }

    /// Check if audit logging is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Log a tool call.
    pub async fn log(&self, entry: AuditEntry) {
        if !self.enabled {
            return;
        }

        let mut guard = self.writer.lock().await;
        if let Some(ref mut writer) = *guard {
            // Serialize to JSON with newline
            if let Ok(json) = serde_json::to_string(&entry) {
                if let Err(e) = writeln!(writer, "{}", json) {
                    tracing::warn!(error = %e, "Failed to write audit log entry");
                } else if let Err(e) = writer.flush() {
                    tracing::warn!(error = %e, "Failed to flush audit log");
                }
            }
        }
    }

    /// Create an audit entry for a tool call.
    pub fn create_entry(
        session_id: &str,
        tool_name: &str,
        arguments: &serde_json::Value,
        success: bool,
        duration_ms: u64,
        result: &str,
        client_ip: Option<String>,
    ) -> AuditEntry {
        AuditEntry {
            timestamp: Utc::now(),
            session_id: session_id.to_string(),
            tool_name: tool_name.to_string(),
            arguments: sanitize_arguments(arguments),
            success,
            duration_ms,
            result_summary: truncate_result(result, 500),
            client_ip,
        }
    }
}

/// Sanitize arguments to remove sensitive or large data.
fn sanitize_arguments(args: &serde_json::Value) -> serde_json::Value {
    match args {
        serde_json::Value::Object(map) => {
            let mut sanitized = serde_json::Map::new();
            for (key, value) in map {
                // Skip or truncate potentially sensitive/large fields
                let sanitized_value = match key.as_str() {
                    // Skip file contents entirely
                    "content" | "file_content" | "data" | "csv_data" => {
                        serde_json::Value::String("[REDACTED]".to_string())
                    }
                    // Skip binary data
                    "base64" | "image" | "binary" => {
                        serde_json::Value::String("[BINARY REDACTED]".to_string())
                    }
                    // Truncate long strings
                    _ if value.is_string() => {
                        let s = value.as_str().unwrap();
                        if s.len() > 200 {
                            serde_json::Value::String(format!("{}...[truncated]", &s[..200]))
                        } else {
                            value.clone()
                        }
                    }
                    // Recursively sanitize nested objects
                    _ if value.is_object() => sanitize_arguments(value),
                    // Pass through other values
                    _ => value.clone(),
                };
                sanitized.insert(key.clone(), sanitized_value);
            }
            serde_json::Value::Object(sanitized)
        }
        _ => args.clone(),
    }
}

/// Truncate result to a maximum length.
fn truncate_result(result: &str, max_len: usize) -> String {
    if result.len() <= max_len {
        result.to_string()
    } else {
        format!("{}...[truncated]", &result[..max_len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_arguments() {
        let args = serde_json::json!({
            "dataset": "test.csv",
            "content": "very long file content that should be redacted",
            "column": "price",
            "nested": {
                "data": "sensitive data"
            }
        });

        let sanitized = sanitize_arguments(&args);

        assert_eq!(sanitized["dataset"], "test.csv");
        assert_eq!(sanitized["content"], "[REDACTED]");
        assert_eq!(sanitized["column"], "price");
        assert_eq!(sanitized["nested"]["data"], "[REDACTED]");
    }

    #[test]
    fn test_truncate_result() {
        let short = "short result";
        assert_eq!(truncate_result(short, 100), short);

        let long = "a".repeat(600);
        let truncated = truncate_result(&long, 500);
        assert!(truncated.ends_with("...[truncated]"));
        assert!(truncated.len() < 520);
    }

    #[test]
    fn test_audit_entry_serialization() {
        let entry = AuditEntry {
            timestamp: Utc::now(),
            session_id: "test-session".to_string(),
            tool_name: "load_dataset".to_string(),
            arguments: serde_json::json!({"path": "/data/test.csv"}),
            success: true,
            duration_ms: 150,
            result_summary: "Loaded dataset with 1000 rows".to_string(),
            client_ip: Some("127.0.0.1".to_string()),
        };

        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("load_dataset"));
        assert!(json.contains("test-session"));
    }
}
