//! Audit logging for MCP tool calls.
//!
//! One JSON line per executed tool call, for security auditing (for example
//! investigating tool use an LLM was steered into). Each entry records:
//! - the timestamp (ISO 8601) and the session id;
//! - the origin of the call (`http`, `llm_tool`, `websocket` or `stdio`);
//! - the tool name and its arguments, sanitised by [`sanitize_arguments`]:
//!   keys that carry data values ([`REDACT_KEYS`], such as `csv_content` or
//!   `filter_value`) are replaced by their size, other strings are cut to 200
//!   bytes, arrays to 20 elements and nesting to 4 levels;
//! - whether it succeeded, its outcome (`ok`, `tool_error`, `cancelled` or a
//!   refusal code such as `limit_exceeded`) and its duration;
//! - the size of its result ([`ResultSummary`]), never the result itself.

use chrono::{DateTime, Utc};
use serde::Serialize;
use serde_json::Value;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::config::AuditConfig;

/// Argument keys whose values are data, not parameters: they are replaced by
/// their size at any depth (`"[redacted: 59 bytes]"` for a string,
/// `"[redacted: 3 items]"` for an array, `"[redacted]"` otherwise).
pub const REDACT_KEYS: &[&str] = &[
    "content",
    "file_content",
    "csv_content",
    "csv_data",
    "data",
    "rows",
    "text",
    "value",
    "values",
    "old_value",
    "new_value",
    "filter_value",
    "replacement",
    "base64",
    "image",
    "binary",
];

/// Longest string kept in an entry, in bytes (cut at a char boundary).
const MAX_STRING_BYTES: usize = 200;
/// Array elements kept in an entry.
const MAX_ARRAY_ITEMS: usize = 20;
/// Deepest nesting kept; the top-level object is level 1.
const MAX_DEPTH: usize = 4;

/// The size of a tool result: content items, text bytes and image data
/// bytes. The audit log records this instead of any output.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ResultSummary {
    /// Content items in the result.
    pub content_items: usize,
    /// Bytes of text content.
    pub text_bytes: usize,
    /// Bytes of (base64) image data.
    pub image_bytes: usize,
}

impl ResultSummary {
    /// The size of a governed call's result.
    #[cfg(feature = "http")]
    pub(crate) fn of_tool_result(result: &crate::transport::http::ToolResult) -> Self {
        use crate::transport::http::ContentItem;
        let mut summary = Self {
            content_items: result.content.len(),
            ..Self::default()
        };
        for item in &result.content {
            match item {
                ContentItem::Text { text } => summary.text_bytes += text.len(),
                ContentItem::Image { data, .. } => summary.image_bytes += data.len(),
            }
        }
        summary
    }

    /// The size of an MCP tool result (stdio).
    pub(crate) fn of_call_result(result: &rmcp::model::CallToolResult) -> Self {
        use rmcp::model::RawContent;
        let mut summary = Self {
            content_items: result.content.len(),
            ..Self::default()
        };
        for item in &result.content {
            match &item.raw {
                RawContent::Text(text) => summary.text_bytes += text.text.len(),
                RawContent::Image(image) => summary.image_bytes += image.data.len(),
                _ => {}
            }
        }
        summary
    }
}

/// Audit log entry for a single tool invocation.
#[derive(Debug, Clone, Serialize)]
pub struct AuditEntry {
    /// ISO 8601 timestamp
    pub timestamp: DateTime<Utc>,
    /// Session identifier (`stdio` on the stdio transport)
    pub session_id: String,
    /// Where the call came from: `http`, `llm_tool`, `websocket` or `stdio`
    pub origin: &'static str,
    /// Name of the tool invoked
    pub tool_name: String,
    /// Sanitised arguments (see [`sanitize_arguments`])
    pub arguments: Value,
    /// Whether the tool call succeeded (`outcome == "ok"`)
    pub success: bool,
    /// `ok`, `tool_error`, `cancelled`, or a refusal code such as
    /// `limit_exceeded`, `unknown_tool` or `deadline_exceeded`
    pub outcome: &'static str,
    /// Execution time in milliseconds
    pub duration_ms: u64,
    /// The size of the result (never its content)
    pub result: ResultSummary,
}

impl AuditEntry {
    /// An entry stamped now. `sanitized_arguments` must come from
    /// [`sanitize_arguments`] (callers sanitise before the arguments move into
    /// the tool call, so a large `csv_content` is never cloned).
    pub fn new(
        session_id: &str,
        origin: &'static str,
        tool_name: &str,
        sanitized_arguments: Value,
        outcome: &'static str,
        duration_ms: u64,
        result: ResultSummary,
    ) -> Self {
        Self {
            timestamp: Utc::now(),
            session_id: session_id.to_string(),
            origin,
            tool_name: tool_name.to_string(),
            arguments: sanitized_arguments,
            success: outcome == "ok",
            outcome,
            duration_ms,
            result,
        }
    }
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
}

/// Sanitise tool arguments for the audit log: [`REDACT_KEYS`] are replaced
/// by their size at any depth; other strings longer than 200 bytes are cut
/// at a char boundary and marked with their length; arrays keep their first
/// 20 elements plus a `"[+n more]"` marker; objects and arrays below level 4
/// (the top level is level 1) become `"[nested]"`. Numbers, booleans and
/// null pass through. SQL text and regex patterns are kept (truncated): they
/// are what an operator needs to investigate injected tool use.
pub fn sanitize_arguments(args: &Value) -> Value {
    sanitize_value(args, 1)
}

fn sanitize_value(value: &Value, level: usize) -> Value {
    match value {
        Value::Object(map) => {
            if level > MAX_DEPTH {
                return Value::String("[nested]".to_string());
            }
            let sanitized = map
                .iter()
                .map(|(key, v)| {
                    let v = if REDACT_KEYS.contains(&key.as_str()) {
                        redact(v)
                    } else {
                        sanitize_value(v, level + 1)
                    };
                    (key.clone(), v)
                })
                .collect();
            Value::Object(sanitized)
        }
        Value::Array(items) => {
            if level > MAX_DEPTH {
                return Value::String("[nested]".to_string());
            }
            let mut kept: Vec<Value> = items
                .iter()
                .take(MAX_ARRAY_ITEMS)
                .map(|v| sanitize_value(v, level + 1))
                .collect();
            if items.len() > MAX_ARRAY_ITEMS {
                kept.push(Value::String(format!(
                    "[+{} more]",
                    items.len() - MAX_ARRAY_ITEMS
                )));
            }
            Value::Array(kept)
        }
        Value::String(s) => Value::String(truncate(s)),
        other => other.clone(),
    }
}

/// The size-only stand-in for a redacted value.
fn redact(value: &Value) -> Value {
    Value::String(match value {
        Value::String(s) => format!("[redacted: {} bytes]", s.len()),
        Value::Array(items) => format!("[redacted: {} items]", items.len()),
        _ => "[redacted]".to_string(),
    })
}

/// `s` cut to at most [`MAX_STRING_BYTES`] at a char boundary, with its
/// original length appended when it was cut.
fn truncate(s: &str) -> String {
    if s.len() <= MAX_STRING_BYTES {
        return s.to_string();
    }
    let head = &s[..floor_char_boundary(s, MAX_STRING_BYTES)];
    format!("{head}...[truncated {} bytes]", s.len())
}

/// The largest char boundary of `s` at or below `max` (`str::floor_char_boundary`
/// is newer than the MSRV).
fn floor_char_boundary(s: &str, max: usize) -> usize {
    if max >= s.len() {
        return s.len();
    }
    let mut i = max;
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
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
        assert_eq!(sanitized["content"], "[redacted: 46 bytes]");
        assert_eq!(sanitized["column"], "price");
        assert_eq!(sanitized["nested"]["data"], "[redacted: 14 bytes]");
    }

    #[test]
    fn sanitize_redacts_csv_content_and_cell_values() {
        let args = serde_json::json!({"name": "pii",
            "csv_content": "email,income\nalice@example.org,51000\nbob@example.org,73000\n",
            "filter_value": "alice@example.org", "column": "email"});
        let s = sanitize_arguments(&args);
        assert_eq!(s["csv_content"], "[redacted: 59 bytes]");
        assert_eq!(s["filter_value"], "[redacted: 17 bytes]");
        assert_eq!(s["column"], "email");
        assert!(!s.to_string().contains("alice@example.org"));
    }

    #[test]
    fn sanitize_truncates_on_a_char_boundary() {
        // 'é' spans bytes 199..201; 211 bytes in all.
        let long = format!("{}é{}", "a".repeat(199), "b".repeat(10));
        let s = sanitize_arguments(&serde_json::json!({ "title": long }));
        assert_eq!(
            s["title"],
            format!("{}...[truncated 211 bytes]", "a".repeat(199))
        );
        // A 3-byte character straddling the cut (box drawing, as in polars tables).
        let table = format!("{}│{}", "x".repeat(199), "y".repeat(5));
        let s = sanitize_arguments(&serde_json::json!({ "title": table }));
        assert_eq!(
            s["title"],
            format!("{}...[truncated 207 bytes]", "x".repeat(199))
        );
    }

    #[test]
    fn sanitize_bounds_arrays_and_depth() {
        let s = sanitize_arguments(&serde_json::json!({"x": (0..100).collect::<Vec<i32>>(),
            "a": {"b": {"c": {"d": {"e": 1}}}}}));
        assert_eq!(s["x"].as_array().unwrap().len(), 21);
        assert_eq!(s["x"][20], "[+80 more]");
        // top-level object = level 1; a container at level 5+ becomes "[nested]"
        assert_eq!(s["a"]["b"]["c"]["d"], "[nested]");
        // scalars and short strings pass through
        let s = sanitize_arguments(&serde_json::json!({"n": 3, "ok": true, "z": null}));
        assert_eq!(s, serde_json::json!({"n": 3, "ok": true, "z": null}));
    }

    #[test]
    fn sanitize_redacts_by_type_at_any_depth() {
        let s = sanitize_arguments(&serde_json::json!({
            "operations": [{"value": 51000}, {"values": ["a", "b", "c"]}],
            "rows": [[1, 2], [3, 4]],
            "query": "SELECT * FROM t WHERE x = 1",
        }));
        assert_eq!(s["operations"][0]["value"], "[redacted]");
        assert_eq!(s["operations"][1]["values"], "[redacted: 3 items]");
        assert_eq!(s["rows"], "[redacted: 2 items]");
        assert_eq!(s["query"], "SELECT * FROM t WHERE x = 1");
        // a non-object top level is sanitised the same way
        let long = "q".repeat(300);
        assert_eq!(
            sanitize_arguments(&Value::String(long)),
            format!("{}...[truncated 300 bytes]", "q".repeat(200))
        );
    }

    #[test]
    fn floor_char_boundary_steps_back_inside_a_character() {
        assert_eq!(floor_char_boundary("aé", 2), 1);
        assert_eq!(floor_char_boundary("aé", 3), 3);
        assert_eq!(floor_char_boundary("abc", 10), 3);
        assert_eq!(floor_char_boundary("", 0), 0);
    }

    #[test]
    fn entry_records_sizes_not_content() {
        let e = AuditEntry::new(
            "s1",
            "http",
            "head_dataset",
            sanitize_arguments(&serde_json::json!({"dataset": "pii"})),
            "ok",
            3,
            ResultSummary {
                content_items: 1,
                text_bytes: 600,
                image_bytes: 0,
            },
        );
        let j = serde_json::to_value(&e).unwrap();
        assert_eq!(j["origin"], "http");
        assert_eq!(j["outcome"], "ok");
        assert_eq!(j["success"], true);
        assert_eq!(
            j["result"],
            serde_json::json!({"content_items": 1, "text_bytes": 600, "image_bytes": 0})
        );
        assert!(j.get("result_summary").is_none() && j.get("client_ip").is_none());
        let refused = AuditEntry::new(
            "s1",
            "stdio",
            "x",
            Value::Null,
            "limit_exceeded",
            0,
            ResultSummary::default(),
        );
        assert!(!refused.success);
    }

    #[test]
    fn test_audit_entry_serialization() {
        let entry = AuditEntry::new(
            "test-session",
            "llm_tool",
            "load_dataset",
            sanitize_arguments(&serde_json::json!({"path": "/data/test.csv"})),
            "ok",
            150,
            ResultSummary::default(),
        );

        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("load_dataset"));
        assert!(json.contains("test-session"));
        assert!(json.contains("\"origin\":\"llm_tool\""));
    }

    /// Words that mark a request field as a likely carrier of data values.
    const PAYLOAD_WORDS: &[&str] = &[
        "content",
        "csv",
        "value",
        "values",
        "replacement",
        "rows",
        "text",
        "data",
    ];

    /// Fields whose names contain a payload word but that hold parameters
    /// (column names, formats, counts), reviewed one by one.
    const NOT_PAYLOAD: &[&str] = &["content_type", "value_name", "value_vars"];

    /// Collects every string or array property of `schema`, following
    /// `$ref`s into `$defs`/`definitions` (each definition once).
    fn collect_properties(
        schema: &Value,
        root: &Value,
        seen: &mut std::collections::HashSet<String>,
        out: &mut std::collections::BTreeSet<String>,
    ) {
        if let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
            if seen.insert(reference.to_string()) {
                let name = reference.rsplit('/').next().unwrap_or_default();
                for defs in ["$defs", "definitions"] {
                    if let Some(target) = root.get(defs).and_then(|d| d.get(name)) {
                        collect_properties(target, root, seen, out);
                    }
                }
            }
            return;
        }
        if let Some(props) = schema.get("properties").and_then(Value::as_object) {
            for (name, prop) in props {
                if is_string_or_array(prop, root) {
                    out.insert(name.clone());
                }
                collect_properties(prop, root, seen, out);
            }
        }
        for key in ["items", "additionalProperties"] {
            if let Some(sub) = schema.get(key) {
                collect_properties(sub, root, seen, out);
            }
        }
        for key in ["anyOf", "oneOf", "allOf", "prefixItems"] {
            for sub in schema.get(key).and_then(Value::as_array).into_iter().flatten() {
                collect_properties(sub, root, seen, out);
            }
        }
    }

    fn is_string_or_array(prop: &Value, root: &Value) -> bool {
        let is = |t: &Value| matches!(t.as_str(), Some("string" | "array"));
        match prop.get("type") {
            Some(Value::Array(types)) => return types.iter().any(is),
            Some(t) => return is(t),
            None => {}
        }
        if let Some(reference) = prop.get("$ref").and_then(Value::as_str) {
            let name = reference.rsplit('/').next().unwrap_or_default();
            return ["$defs", "definitions"].iter().any(|defs| {
                root.get(*defs)
                    .and_then(|d| d.get(name))
                    .is_some_and(|t| t.get("enum").is_none() && is_string_or_array(t, root))
            });
        }
        ["anyOf", "oneOf"].iter().any(|key| {
            prop.get(*key)
                .and_then(Value::as_array)
                .is_some_and(|subs| subs.iter().any(|s| is_string_or_array(s, root)))
        })
    }

    #[cfg(feature = "http")]
    #[test]
    fn payload_keys_cover_request_schemas() {
        let mut uncovered = std::collections::BTreeSet::new();
        let mut payload_fields = std::collections::BTreeSet::new();
        for tool in crate::server::AnalyticsServer::new().list_tools() {
            let mut names = std::collections::BTreeSet::new();
            let mut seen = std::collections::HashSet::new();
            collect_properties(&tool.input_schema, &tool.input_schema, &mut seen, &mut names);
            for name in names {
                let payload_like = name.split('_').any(|w| PAYLOAD_WORDS.contains(&w));
                if !payload_like {
                    continue;
                }
                payload_fields.insert(name.clone());
                if !REDACT_KEYS.contains(&name.as_str()) && !NOT_PAYLOAD.contains(&name.as_str()) {
                    uncovered.insert(format!("{}.{name}", tool.name));
                }
            }
        }
        // The walk reaches top-level and nested request fields.
        for known in ["csv_content", "filter_value", "value_vars", "replacement"] {
            assert!(
                payload_fields.contains(known),
                "{known} not found: {payload_fields:?}"
            );
        }
        assert!(
            uncovered.is_empty(),
            "classify these fields in REDACT_KEYS or NOT_PAYLOAD: {uncovered:#?}"
        );
    }
}
