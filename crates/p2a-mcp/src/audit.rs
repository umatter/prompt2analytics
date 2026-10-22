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
//!
//! Entries go through a bounded queue to one writer thread per file
//! ([`AuditLogger::new`] returns a handle to the existing writer for a path),
//! so tool calls never do file IO on a runtime worker. The file is created
//! with mode 0600 and rotates by size ([`RotationPolicy`], `P2A_AUDIT_MAX_MB`,
//! default 50 MiB, `0` disables), keeping `<path>.1` to `<path>.3`. Entries
//! still queued when the process is killed are lost.

use chrono::{DateTime, Utc};
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex as StdMutex, PoisonError, Weak};
use tokio::sync::{mpsc, oneshot};

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

/// Default size at which the audit log rotates, in MiB (`P2A_AUDIT_MAX_MB`).
const DEFAULT_ROTATE_MB: u64 = 50;
/// Rotated files kept (`<path>.1` .. `<path>.3`).
const DEFAULT_KEEP: usize = 3;
/// Entries queued for the writer thread before `log` waits.
const QUEUE_CAPACITY: usize = 1024;

/// When the audit log rotates and how many old files it keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RotationPolicy {
    /// Rotate before a write would take the file above this size (`None`:
    /// never rotate).
    pub max_bytes: Option<u64>,
    /// Rotated files kept as `<path>.1` (newest) to `<path>.<keep>`.
    pub keep: usize,
}

impl RotationPolicy {
    /// The policy for a `P2A_AUDIT_MAX_MB` value: unset means 50 MiB, `0`
    /// disables rotation, anything unparsable is logged and gives the
    /// default. Three old files are kept.
    pub fn from_env_value(value: Option<&str>) -> Self {
        let mb = match value.map(str::trim) {
            None | Some("") => DEFAULT_ROTATE_MB,
            Some(v) => v.parse::<u64>().unwrap_or_else(|_| {
                tracing::warn!(
                    value = v,
                    "P2A_AUDIT_MAX_MB is not a whole number of MiB; rotating the audit log at \
                     {DEFAULT_ROTATE_MB} MiB"
                );
                DEFAULT_ROTATE_MB
            }),
        };
        Self {
            max_bytes: (mb > 0).then(|| mb.saturating_mul(1 << 20)),
            keep: DEFAULT_KEEP,
        }
    }
}

/// A message to the writer thread.
enum Msg {
    Entry(Box<AuditEntry>),
    /// Reply once every earlier entry is written and flushed.
    Flush(oneshot::Sender<()>),
}

/// The handle side of one file's writer thread. The thread stops after the
/// last handle drops and the queue is drained.
struct Sink {
    tx: mpsc::Sender<Msg>,
    /// Set once the writer is found gone (warned once).
    closed_warned: AtomicBool,
}

/// One writer per absolute path in the process: every `AuditLogger` for the
/// same file shares its thread, file handle and rotation state.
static WRITERS: LazyLock<StdMutex<HashMap<PathBuf, Weak<Sink>>>> =
    LazyLock::new(|| StdMutex::new(HashMap::new()));

/// Audit logger: hands entries to a dedicated writer thread through a
/// bounded queue, so callers never do file IO.
pub struct AuditLogger {
    sink: Option<Arc<Sink>>,
}

impl AuditLogger {
    /// A logger for `config` (disabled unless `config.enabled`), rotating
    /// at `P2A_AUDIT_MAX_MB`. Returns a handle to the process's one writer
    /// for that file when there already is one.
    pub fn new(config: &AuditConfig) -> std::io::Result<Self> {
        let policy =
            RotationPolicy::from_env_value(std::env::var("P2A_AUDIT_MAX_MB").ok().as_deref());
        Self::with_rotation(config, policy)
    }

    /// A logger for `config` with an explicit rotation policy. The first
    /// configuration of a path wins: a later logger for the same file shares
    /// the existing writer and its policy. Opening the file (create, append,
    /// mode 0600 on Unix) is the one synchronous step, at startup.
    pub fn with_rotation(config: &AuditConfig, policy: RotationPolicy) -> std::io::Result<Self> {
        if !config.enabled {
            return Ok(Self::disabled());
        }
        let path = std::path::absolute(&config.path)?;
        let mut writers = WRITERS.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(sink) = writers.get(&path).and_then(Weak::upgrade) {
            return Ok(Self { sink: Some(sink) });
        }
        let file = open_log(&path)?;
        let size = file.metadata()?.len();
        let (tx, rx) = mpsc::channel(QUEUE_CAPACITY);
        let writer = Writer {
            path: path.clone(),
            file: BufWriter::new(file),
            size,
            policy,
            io_errors: 0,
        };
        std::thread::Builder::new()
            .name("p2a-audit".to_string())
            .spawn(move || writer.run(rx))?;
        let sink = Arc::new(Sink {
            tx,
            closed_warned: AtomicBool::new(false),
        });
        writers.retain(|_, w| w.strong_count() > 0);
        writers.insert(path, Arc::downgrade(&sink));
        tracing::info!(path = %config.path, "Audit logging enabled");
        Ok(Self { sink: Some(sink) })
    }

    /// Create a disabled audit logger.
    pub fn disabled() -> Self {
        Self { sink: None }
    }

    /// Check if audit logging is enabled.
    pub fn is_enabled(&self) -> bool {
        self.sink.is_some()
    }

    /// Queue one entry for the writer thread. Waits while the queue is full
    /// (back-pressure); never touches the file itself.
    pub async fn log(&self, entry: AuditEntry) {
        let Some(sink) = &self.sink else {
            return;
        };
        if sink.tx.send(Msg::Entry(Box::new(entry))).await.is_err()
            && !sink.closed_warned.swap(true, Ordering::Relaxed)
        {
            tracing::warn!("the audit log writer has stopped; audit entries are being dropped");
        }
    }

    /// Resolves after every entry queued before it is written and flushed.
    pub async fn flush(&self) {
        let Some(sink) = &self.sink else {
            return;
        };
        let (reply, done) = oneshot::channel();
        if sink.tx.send(Msg::Flush(reply)).await.is_ok() {
            let _ = done.await;
        }
    }

    /// True when both loggers write through the same writer thread (and so
    /// the same file).
    pub fn shares_writer_with(&self, other: &AuditLogger) -> bool {
        matches!((&self.sink, &other.sink), (Some(a), Some(b)) if Arc::ptr_eq(a, b))
    }
}

/// Opens the log for appending, creating it private to the owner (0600) on
/// Unix.
fn open_log(path: &Path) -> std::io::Result<File> {
    let mut options = OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)
}

/// `<path>.<k>`.
fn numbered(path: &Path, k: usize) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(format!(".{k}"));
    PathBuf::from(name)
}

/// The writer thread's state for one file.
struct Writer {
    path: PathBuf,
    file: BufWriter<File>,
    /// Bytes in the current file.
    size: u64,
    policy: RotationPolicy,
    io_errors: u64,
}

impl Writer {
    /// Writes queued messages until every handle is gone and the queue is
    /// drained; flushes whenever the queue runs empty.
    fn run(mut self, mut rx: mpsc::Receiver<Msg>) {
        while let Some(msg) = rx.blocking_recv() {
            self.handle(msg);
            while let Ok(msg) = rx.try_recv() {
                self.handle(msg);
            }
            self.flush();
        }
        self.flush();
    }

    fn handle(&mut self, msg: Msg) {
        match msg {
            Msg::Entry(entry) => self.write_entry(&entry),
            Msg::Flush(reply) => {
                self.flush();
                let _ = reply.send(());
            }
        }
    }

    fn write_entry(&mut self, entry: &AuditEntry) {
        let mut line = match serde_json::to_string(entry) {
            Ok(line) => line,
            Err(e) => {
                tracing::warn!(error = %e, "could not serialise an audit entry");
                return;
            }
        };
        line.push('\n');
        let len = line.len() as u64;
        if let Some(max) = self.policy.max_bytes
            && self.size > 0
            && self.size.saturating_add(len) > max
        {
            self.rotate();
        }
        match self.file.write_all(line.as_bytes()) {
            Ok(()) => self.size = self.size.saturating_add(len),
            Err(e) => self.io_error(&e),
        }
    }

    fn flush(&mut self) {
        if let Err(e) = self.file.flush() {
            self.io_error(&e);
        }
    }

    /// `<path>.<keep>` is removed, `.k` becomes `.k+1`, the current file
    /// becomes `.1` and a new file is opened. A failed rename or open is
    /// logged and writing continues in the file at hand, so no entry is
    /// dropped.
    fn rotate(&mut self) {
        self.flush();
        let keep = self.policy.keep;
        let moved = if keep == 0 {
            std::fs::remove_file(&self.path)
        } else {
            match std::fs::remove_file(numbered(&self.path, keep)) {
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => self.io_error(&e),
                _ => {}
            }
            for k in (1..keep).rev() {
                match std::fs::rename(numbered(&self.path, k), numbered(&self.path, k + 1)) {
                    Err(e) if e.kind() != std::io::ErrorKind::NotFound => self.io_error(&e),
                    _ => {}
                }
            }
            std::fs::rename(&self.path, numbered(&self.path, 1))
        };
        if let Err(e) = moved {
            self.io_error(&e);
            return;
        }
        match open_log(&self.path) {
            Ok(file) => {
                self.file = BufWriter::new(file);
                self.size = 0;
            }
            Err(e) => self.io_error(&e),
        }
    }

    /// Counts IO errors; warns on the first and every 1000th.
    fn io_error(&mut self, e: &std::io::Error) {
        self.io_errors += 1;
        if self.io_errors == 1 || self.io_errors.is_multiple_of(1000) {
            tracing::warn!(
                error = %e,
                errors = self.io_errors,
                path = %self.path.display(),
                "audit log write failed"
            );
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

    fn cfg_in(dir: &tempfile::TempDir) -> AuditConfig {
        AuditConfig {
            enabled: true,
            path: dir.path().join("audit.log").to_string_lossy().into_owned(),
        }
    }

    fn entry(i: usize) -> AuditEntry {
        AuditEntry::new(
            &format!("s{i}"),
            "http",
            "list_datasets",
            serde_json::json!({ "i": i }),
            "ok",
            1,
            ResultSummary::default(),
        )
    }

    #[test]
    fn rotation_policy_from_env_value() {
        let mib = 1024 * 1024;
        assert_eq!(
            RotationPolicy::from_env_value(None).max_bytes,
            Some(50 * mib)
        );
        assert_eq!(
            RotationPolicy::from_env_value(Some("10")).max_bytes,
            Some(10 * mib)
        );
        assert_eq!(RotationPolicy::from_env_value(Some("0")).max_bytes, None);
        // unparsable: warn and use the default
        assert_eq!(
            RotationPolicy::from_env_value(Some("ten")).max_bytes,
            Some(50 * mib)
        );
        assert_eq!(RotationPolicy::from_env_value(None).keep, 3);
    }

    #[tokio::test]
    async fn writer_rotates_and_keeps_three_files() {
        let dir = tempfile::tempdir().unwrap();
        let logger = AuditLogger::with_rotation(
            &cfg_in(&dir),
            RotationPolicy {
                max_bytes: Some(4096),
                keep: 3,
            },
        )
        .unwrap();
        for i in 0..200 {
            logger.log(entry(i)).await;
        }
        logger.flush().await;

        let file = |k: usize| match k {
            0 => dir.path().join("audit.log"),
            k => dir.path().join(format!("audit.log.{k}")),
        };
        for k in 0..=3 {
            let len = std::fs::metadata(file(k)).unwrap().len();
            assert!(len > 0 && len <= 4096, "file {k}: {len} bytes");
        }
        assert!(!file(4).exists());
        let mut ids = Vec::new();
        for k in [3, 2, 1, 0] {
            for line in std::fs::read_to_string(file(k)).unwrap().lines() {
                let v: Value = serde_json::from_str(line).unwrap();
                let id = v["session_id"].as_str().unwrap();
                ids.push(id[1..].parse::<usize>().unwrap());
            }
        }
        assert_eq!(ids.last(), Some(&199));
        assert!(ids.len() < 200, "old files were not dropped");
        assert!(
            ids.windows(2).all(|w| w[1] == w[0] + 1),
            "not consecutive: {ids:?}"
        );
    }

    #[tokio::test]
    async fn same_path_shares_one_writer() {
        let dir = tempfile::tempdir().unwrap();
        let a = AuditLogger::new(&cfg_in(&dir)).unwrap();
        let b = AuditLogger::new(&cfg_in(&dir)).unwrap();
        assert!(a.shares_writer_with(&b));
        let other_dir = tempfile::tempdir().unwrap();
        let c = AuditLogger::new(&cfg_in(&other_dir)).unwrap();
        assert!(!a.shares_writer_with(&c));
        assert!(!AuditLogger::disabled().shares_writer_with(&AuditLogger::disabled()));

        a.log(entry(1)).await;
        b.log(entry(2)).await;
        a.flush().await;
        let text = std::fs::read_to_string(dir.path().join("audit.log")).unwrap();
        let ids: Vec<String> = text
            .lines()
            .map(|l| serde_json::from_str::<Value>(l).unwrap()["session_id"].to_string())
            .collect();
        assert_eq!(ids, ["\"s1\"", "\"s2\""]);
    }

    #[tokio::test]
    async fn disabled_logger_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = cfg_in(&dir);
        cfg.enabled = false;
        let logger = AuditLogger::new(&cfg).unwrap();
        assert!(!logger.is_enabled());
        logger.log(entry(1)).await;
        logger.flush().await;
        assert!(!dir.path().join("audit.log").exists());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn audit_file_is_private() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let logger = AuditLogger::new(&cfg_in(&dir)).unwrap();
        logger.log(entry(1)).await;
        logger.flush().await;
        let mode = std::fs::metadata(dir.path().join("audit.log"))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600);
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
