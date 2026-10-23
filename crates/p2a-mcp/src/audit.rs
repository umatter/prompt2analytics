//! Audit logging for MCP tool calls.
//!
//! One JSON line per executed tool call, for security auditing (for example
//! investigating tool use an LLM was steered into). Each entry records:
//! - the timestamp (ISO 8601) and the session id;
//! - the origin of the call (`http`, `llm_tool`, `websocket` or `stdio`);
//! - the tool name, cut to 200 bytes, and its arguments, sanitised by
//!   [`sanitize_arguments`]: keys that carry data values ([`REDACT_KEYS`],
//!   such as `csv_content` or `filter_value`) are replaced by their size,
//!   other strings and object keys are cut to 200 bytes, arrays to 20
//!   elements, objects to 64 keys, nesting to 4 levels and the whole to about
//!   16 KiB, so an entry stays small whatever the request carried; arguments
//!   that are not an object (positional values, which every call path
//!   refuses) are replaced by their size as a whole. The
//!   parameters the tool's schema declares are taken first (`query` and
//!   `pattern` ahead of the rest), so the key cap and the size bound fall on
//!   the other keys before them; a value passed under a serde alias (such as
//!   `entity` for a panel tool's `entity_var`) is not declared and ranks
//!   with those other keys;
//! - whether it succeeded, its outcome (`ok`, `tool_error`, `cancelled` or a
//!   refusal code such as `limit_exceeded`) and its duration;
//! - the size of its result ([`ResultSummary`]), never the result itself.
//!
//! Entries go through a bounded queue to one writer thread per file
//! ([`AuditLogger::new`] returns a handle to the existing writer for a path),
//! so tool calls never do file IO on a runtime worker. The file is created
//! with mode 0600 and rotates by size ([`RotationPolicy`], `P2A_AUDIT_MAX_MB`,
//! default 50 MiB, `0` disables), keeping `<path>.1` to `<path>.3`. When the
//! file is removed or moved away outside the server, the next entry or flush
//! starts a new file at the path, which also takes the entries not yet
//! written out, and the numbered files are left as they are; that new file
//! is created exclusively, so a symlink planted at the path is never
//! followed. The writer flushes whenever its queue runs empty, and the stdio
//! transport and `transport::http::serve` write out every entry queued by the
//! time they return (a call still running after its deadline answer is
//! logged when its work ends); entries still queued when the process is
//! killed are lost.

use chrono::{DateTime, Utc};
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write, WriterPanicked};
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

/// Declared parameters taken ahead of every other argument: the SQL text,
/// regex patterns and search queries an operator needs to investigate
/// injected tool use (`query`: `db_sqlite_query`, `db_duckdb_query`,
/// `db_query_file`, `search_tools`; `pattern`: `str_regex_replace`,
/// `str_regex_extract`, `str_regex_count`, `str_split`).
const KEEP_FIRST_KEYS: &[&str] = &["query", "pattern"];

/// Longest string kept in an entry, in bytes (cut at a char boundary).
const MAX_STRING_BYTES: usize = 200;
/// Array elements kept in an entry.
const MAX_ARRAY_ITEMS: usize = 20;
/// Keys kept per object, above the parameter count of the widest tool
/// request (a test checks this).
const MAX_OBJECT_KEYS: usize = 64;
/// Deepest nesting kept; the top-level object is level 1.
const MAX_DEPTH: usize = 4;
/// Bytes the sanitised arguments may take in an entry's JSON, apart from
/// the markers that count what was left out once it is spent.
const MAX_ARGUMENT_BYTES: usize = 16 * 1024;

/// The size of a tool result: content items, text bytes, image data bytes
/// and the bytes of any other content kind. The audit log records this
/// instead of any output, and the response cap (`response_bytes`) measures
/// [`output_bytes`](Self::output_bytes), so the two cannot disagree.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ResultSummary {
    /// Content items in the result.
    pub content_items: usize,
    /// Bytes of text content.
    pub text_bytes: usize,
    /// Bytes of (base64) image data.
    pub image_bytes: usize,
    /// Bytes of the other MCP content kinds (stdio only; the HTTP result
    /// type has none): embedded resource text or (base64) blob, (base64)
    /// audio data, and a resource link's JSON. No tool returns these today,
    /// so the field is left out of an entry while it is zero.
    #[serde(skip_serializing_if = "is_zero")]
    pub other_bytes: usize,
}

fn is_zero(n: &usize) -> bool {
    *n == 0
}

impl ResultSummary {
    /// Bytes of output in the result, whatever its kind: what the
    /// `response_bytes` budget is checked against.
    pub fn output_bytes(&self) -> usize {
        self.text_bytes
            .saturating_add(self.image_bytes)
            .saturating_add(self.other_bytes)
    }

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

    /// The size of an MCP tool result (stdio). Every content kind is
    /// counted; the match has no catch-all, so a kind a newer rmcp adds is a
    /// compile error here rather than output the cap does not see.
    pub(crate) fn of_call_result(result: &rmcp::model::CallToolResult) -> Self {
        use rmcp::model::{RawContent, ResourceContents};
        let mut summary = Self {
            content_items: result.content.len(),
            ..Self::default()
        };
        for item in &result.content {
            match &item.raw {
                RawContent::Text(text) => summary.text_bytes += text.text.len(),
                RawContent::Image(image) => summary.image_bytes += image.data.len(),
                RawContent::Audio(audio) => summary.other_bytes += audio.data.len(),
                RawContent::Resource(embedded) => {
                    summary.other_bytes += match &embedded.resource {
                        ResourceContents::TextResourceContents { text, .. } => text.len(),
                        ResourceContents::BlobResourceContents { blob, .. } => blob.len(),
                    }
                }
                // Metadata only (uri, name, title, icons ...): its JSON is
                // what the client receives.
                RawContent::ResourceLink(link) => {
                    summary.other_bytes += serde_json::to_vec(link).map_or(0, |json| json.len());
                }
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
    /// Name of the tool invoked, cut to 200 bytes like an argument string
    /// (an unknown tool's name is whatever the caller sent)
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
            tool_name: truncate(tool_name),
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

/// Opens the log for appending at startup, creating it private to the owner
/// (0600) on Unix.
fn open_log(path: &Path) -> std::io::Result<File> {
    log_options(OpenOptions::new().create(true)).open(path)
}

/// Creates a new log for appending, private to the owner (0600) on Unix.
/// Fails with `AlreadyExists` when anything is at `path`, a symlink included
/// (`O_CREAT | O_EXCL` does not follow it).
fn create_log(path: &Path) -> std::io::Result<File> {
    log_options(OpenOptions::new().create_new(true)).open(path)
}

fn log_options(options: &mut OpenOptions) -> &mut OpenOptions {
    options.append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options
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
            Msg::Entry(entry) => {
                #[cfg(test)]
                gate::wait(&self.path);
                self.write_entry(&entry);
            }
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
        self.reopen_if_removed();
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

    /// Writes the buffered entries out to the file at the log's path: when
    /// the file was removed or moved away since they were written, to a new
    /// file there ([`reopen_if_removed`](Self::reopen_if_removed)).
    fn flush(&mut self) {
        if !self.file.buffer().is_empty() {
            self.reopen_if_removed();
        }
        self.flush_buffer();
    }

    /// Writes the buffer out to the file at hand.
    fn flush_buffer(&mut self) {
        if let Err(e) = self.file.flush() {
            self.io_error(&e);
        }
    }

    /// Opens a new file when nothing is at the log's path any more (an
    /// operator's `rm`, or a log rotator that moved the file without
    /// creating a new one), so later entries do not go to a file that can no
    /// longer be found; the entries still buffered go to the new file too
    /// ([`reopen`](Self::reopen)). One `lstat` per entry and one per flush of
    /// a non-empty buffer; a file or symlink at the path is left as it is.
    fn reopen_if_removed(&mut self) {
        let gone = std::fs::symlink_metadata(&self.path)
            .is_err_and(|e| e.kind() == std::io::ErrorKind::NotFound);
        if gone && self.reopen() {
            tracing::warn!(
                path = %self.path.display(),
                "the audit log was removed or moved away; writing to a new file at its path"
            );
        }
    }

    /// Makes a new file at `self.path`, which the caller found empty, the
    /// file at hand, and writes into it the entries the old file's buffer
    /// still held: after an `rm` they would be lost with the old file, and
    /// after a move they are kept at the path with the entries that follow.
    /// The file is created exclusively: whatever appeared at the path since
    /// the check, a symlink included, is refused rather than opened or
    /// followed. On failure the error is counted, writing (the buffered
    /// entries included) continues in the file at hand and the size count
    /// restarts, so the next rotation waits for another `max_bytes`.
    fn reopen(&mut self) -> bool {
        match create_log(&self.path) {
            Ok(file) => {
                let old = std::mem::replace(&mut self.file, BufWriter::new(file));
                // `into_parts` hands back the buffer without writing it out.
                let (_, unwritten) = old.into_parts();
                let unwritten = unwritten.unwrap_or_else(WriterPanicked::into_inner);
                self.size = 0;
                match self.file.write_all(&unwritten) {
                    Ok(()) => self.size = unwritten.len() as u64,
                    Err(e) => self.io_error(&e),
                }
                true
            }
            Err(e) => {
                self.io_error(&e);
                self.size = 0;
                false
            }
        }
    }

    /// `<path>.<keep>` is removed, `.k` becomes `.k+1`, the current file
    /// (its buffer written out first) becomes `.1` and a new file is opened.
    /// When nothing is at the path any more (the file was removed or moved
    /// away outside the server), the numbered files are left as they are and
    /// a new file is opened, which takes the entries still buffered. Any
    /// other failure is logged and writing continues in the file at hand;
    /// the size count restarts, so the next attempt waits for another
    /// `max_bytes` instead of shifting the numbered files again on every
    /// entry. Buffered entries stay in a removed file only when no new file
    /// can be created at the path, which is logged.
    fn rotate(&mut self) {
        use std::io::ErrorKind::NotFound;
        match std::fs::symlink_metadata(&self.path) {
            Ok(_) => {}
            Err(e) if e.kind() == NotFound => {
                self.reopen();
                return;
            }
            Err(e) => {
                self.io_error(&e);
                self.size = 0;
                return;
            }
        }
        // What the current file was sent goes with it to `.1`.
        self.flush_buffer();
        let keep = self.policy.keep;
        let moved = if keep == 0 {
            std::fs::remove_file(&self.path)
        } else {
            match std::fs::remove_file(numbered(&self.path, keep)) {
                Err(e) if e.kind() != NotFound => self.io_error(&e),
                _ => {}
            }
            for k in (1..keep).rev() {
                match std::fs::rename(numbered(&self.path, k), numbered(&self.path, k + 1)) {
                    Err(e) if e.kind() != NotFound => self.io_error(&e),
                    _ => {}
                }
            }
            std::fs::rename(&self.path, numbered(&self.path, 1))
        };
        match moved {
            Err(e) if e.kind() != NotFound => {
                self.io_error(&e);
                self.size = 0;
            }
            // Moved, or gone since the check: a new file either way.
            _ => {
                self.reopen();
            }
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

/// Test hook: while a test holds a log's path, its writer waits before each
/// entry, so the test decides when queued entries are written.
#[cfg(test)]
pub(crate) mod gate {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Condvar, LazyLock, Mutex, PoisonError};

    /// `true` while the gate is closed.
    type Shared = Arc<(Mutex<bool>, Condvar)>;

    static GATES: LazyLock<Mutex<HashMap<PathBuf, Shared>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));

    /// A closed gate for one log path; dropping it opens the gate.
    pub(crate) struct Held {
        path: PathBuf,
        shared: Shared,
    }

    /// Closes the gate of the log at `path` until the guard drops.
    #[must_use = "the gate opens when this guard drops"]
    pub(crate) fn hold(path: &Path) -> Held {
        let path = std::path::absolute(path).expect("an absolute path");
        let shared: Shared = Arc::new((Mutex::new(true), Condvar::new()));
        GATES
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(path.clone(), Arc::clone(&shared));
        Held { path, shared }
    }

    impl Drop for Held {
        fn drop(&mut self) {
            GATES
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .remove(&self.path);
            let (closed, opened) = &*self.shared;
            *closed.lock().unwrap_or_else(PoisonError::into_inner) = false;
            opened.notify_all();
        }
    }

    /// Waits while the gate of the log at `path` (absolute) is closed.
    pub(super) fn wait(path: &Path) {
        let shared = GATES
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(path)
            .cloned();
        if let Some(shared) = shared {
            let (closed, opened) = &*shared;
            let mut closed = closed.lock().unwrap_or_else(PoisonError::into_inner);
            while *closed {
                closed = opened.wait(closed).unwrap_or_else(PoisonError::into_inner);
            }
        }
    }
}

/// Sanitise tool arguments for the audit log. `declared` names the tool's
/// top-level parameters, the `properties` of its input schema
/// ([`AnalyticsServer::declared_parameters`](crate::server::AnalyticsServer::declared_parameters);
/// empty for an unknown tool).
///
/// [`REDACT_KEYS`] are replaced by their size at any depth; other strings
/// and object keys longer than 200 bytes are cut at a char boundary and
/// marked with their length; arrays keep their first 20 elements plus a
/// `"[+n more]"` marker, objects at most 64 keys plus a `"[+n more keys]":
/// null` entry; objects and arrays below level 4 (the top level is level 1)
/// become `"[nested]"`. Numbers, booleans and null pass through. The result
/// takes at most 16 KiB as JSON, apart from the markers: once that is spent,
/// the remaining elements are left out and counted by the same markers.
///
/// The top-level keys are taken in this order: the declared `query` and
/// `pattern`, the other declared parameters that are single values (or are
/// redacted to one), the declared objects and arrays (by name within each
/// group), then the keys the schema does not declare, in the order the
/// arguments map holds them (sorted by key, or as sent in builds where a
/// dependency enables serde_json's `preserve_order`, as the `db` feature
/// does). The key cap and the byte bound therefore fall on undeclared keys
/// before any declared parameter, and SQL text and regex patterns are kept
/// (truncated): they are what an operator needs to investigate injected
/// tool use. The ordering covers the parameters the schema declares: a
/// value passed under a serde alias (such as `entity` for a panel tool's
/// `entity_var`) is not declared and ranks with the undeclared keys. Nested
/// objects keep the map's order. A key cut to the name of a key already
/// kept, or spelled like the `"[+n more keys]"` marker, is left out and
/// counted by the marker, so no kept value is overwritten and the marker is
/// always the sanitiser's own.
///
/// Arguments that are not an object are replaced by their size like a
/// [`REDACT_KEYS`] value (`"[redacted: 5 items]"` for an array): positional
/// values have no names to redact by, and every call path refuses them
/// (`AnalyticsServer::call_tool_with_session`), so their entry needs only
/// their shape. `null`, arguments left out, stays `null`.
pub fn sanitize_arguments(args: &Value, declared: &[&str]) -> Value {
    let mut budget = MAX_ARGUMENT_BYTES;
    match args {
        // A top-level object always fits: it needs 3 bytes.
        Value::Object(map) => sanitize_object(map, 1, declared, &mut budget).unwrap_or(Value::Null),
        Value::Null => Value::Null,
        other => redact(other),
    }
}

/// The sanitised `value`, charged to `budget` (bytes of JSON left), or
/// `None`, with `budget` unchanged, when it does not fit.
fn sanitize_value(value: &Value, level: usize, budget: &mut usize) -> Option<Value> {
    match value {
        Value::Object(map) if level <= MAX_DEPTH => sanitize_object(map, level, &[], budget),
        Value::Array(items) if level <= MAX_DEPTH => {
            *budget = budget.checked_sub(3)?;
            let mut kept = Vec::new();
            for v in items.iter().take(MAX_ARRAY_ITEMS) {
                let Some(v) = sanitize_value(v, level + 1, budget) else {
                    break;
                };
                kept.push(v);
            }
            if kept.len() < items.len() {
                let marker = format!("[+{} more]", items.len() - kept.len());
                *budget = budget.saturating_sub(json_string_len(&marker) + 1);
                kept.push(Value::String(marker));
            }
            Some(Value::Array(kept))
        }
        Value::Object(_) | Value::Array(_) => scalar(Value::String("[nested]".to_string()), budget),
        Value::String(s) => scalar(Value::String(truncate(s)), budget),
        other => scalar(other.clone(), budget),
    }
}

/// The sanitised object `map` at `level`, charged to `budget` like
/// [`sanitize_value`]. The keys in `declared` come first, in [`key_rank`]
/// order and by name within a rank; the others follow in map order (sorted
/// by key, or the order the request was sent in where serde_json's
/// `preserve_order` is on, as it is in `db` builds). At most
/// [`MAX_OBJECT_KEYS`] are kept, and the map is walked lazily, so a wide
/// object's keys are never collected.
fn sanitize_object(
    map: &serde_json::Map<String, Value>,
    level: usize,
    declared: &[&str],
    budget: &mut usize,
) -> Option<Value> {
    // Braces and a separator.
    *budget = budget.checked_sub(3)?;
    // The declared keys present, at most the schema's parameter count, by
    // rank and then by name; the rest in map order.
    let mut head: Vec<(&String, &Value)> = declared
        .iter()
        .filter_map(|name| map.get_key_value(*name))
        .collect();
    head.sort_by_key(|&(key, v)| (key_rank(key, v), key));
    let rest = map
        .iter()
        .filter(|(key, _)| !declared.contains(&key.as_str()));
    let mut kept = serde_json::Map::new();
    for (key, v) in head.into_iter().chain(rest) {
        if kept.len() == MAX_OBJECT_KEYS {
            break;
        }
        let name = truncate(key);
        // Left out and counted: a second key cut to a kept name (its value
        // would replace the first) and a key spelled like the marker (the
        // marker would replace it, or it would pass for one).
        if kept.contains_key(&name) || is_keys_marker(&name) {
            continue;
        }
        // The key, its colon, then the value; nothing is charged unless both
        // fit.
        let Some(mut left) = budget.checked_sub(json_string_len(&name) + 1) else {
            break;
        };
        let v = if REDACT_KEYS.contains(&key.as_str()) {
            scalar(redact(v), &mut left)
        } else {
            sanitize_value(v, level + 1, &mut left)
        };
        let Some(v) = v else {
            break;
        };
        *budget = left;
        kept.insert(name, v);
    }
    let left_out = map.len() - kept.len();
    if left_out > 0 {
        let marker = format!("[+{left_out} more keys]");
        *budget = budget.saturating_sub(json_string_len(&marker) + 6);
        kept.insert(marker, Value::Null);
    }
    Some(Value::Object(kept))
}

/// Where a declared top-level parameter comes in the sanitising order,
/// lowest first: a [`KEEP_FIRST_KEYS`] parameter, another parameter that is
/// a single value or is redacted to one, an object or array. The keys the
/// tool does not declare come after all of them.
fn key_rank(key: &str, value: &Value) -> u8 {
    if KEEP_FIRST_KEYS.contains(&key) {
        0
    } else if REDACT_KEYS.contains(&key) || !(value.is_object() || value.is_array()) {
        1
    } else {
        2
    }
}

/// True for a key spelled like the `"[+n more keys]"` marker.
fn is_keys_marker(name: &str) -> bool {
    name.starts_with("[+") && name.ends_with(" more keys]")
}

/// `value` (a string, number, boolean or null) when it and a separator fit
/// in `budget`, which is charged for them.
fn scalar(value: Value, budget: &mut usize) -> Option<Value> {
    let len = match &value {
        Value::String(s) => json_string_len(s),
        Value::Number(n) => n.to_string().len(),
        Value::Bool(true) | Value::Null => 4,
        Value::Bool(false) => 5,
        // Not reached: containers are charged element by element.
        container => serde_json::to_string(container).map_or(0, |s| s.len()),
    };
    *budget = budget.checked_sub(len + 1)?;
    Some(value)
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

/// Bytes `s` takes as a JSON string as serde_json writes it: the quotes, a
/// two-byte escape for `"`, `\` and `\b \t \n \f \r`, `\u00XX` for the other
/// control characters, every other byte as it is.
fn json_string_len(s: &str) -> usize {
    2 + s
        .bytes()
        .map(|b| match b {
            b'"' | b'\\' | b'\n' | b'\r' | b'\t' | 0x08 | 0x0c => 2,
            0..=0x1f => 6,
            _ => 1,
        })
        .sum::<usize>()
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

        let sanitized = sanitize_arguments(&args, &[]);

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
        let s = sanitize_arguments(&args, &[]);
        assert_eq!(s["csv_content"], "[redacted: 59 bytes]");
        assert_eq!(s["filter_value"], "[redacted: 17 bytes]");
        assert_eq!(s["column"], "email");
        assert!(!s.to_string().contains("alice@example.org"));
    }

    #[test]
    fn sanitize_truncates_on_a_char_boundary() {
        // 'é' spans bytes 199..201; 211 bytes in all.
        let long = format!("{}é{}", "a".repeat(199), "b".repeat(10));
        let s = sanitize_arguments(&serde_json::json!({ "title": long }), &[]);
        assert_eq!(
            s["title"],
            format!("{}...[truncated 211 bytes]", "a".repeat(199))
        );
        // A 3-byte character straddling the cut (box drawing, as in polars tables).
        let table = format!("{}│{}", "x".repeat(199), "y".repeat(5));
        let s = sanitize_arguments(&serde_json::json!({ "title": table }), &[]);
        assert_eq!(
            s["title"],
            format!("{}...[truncated 207 bytes]", "x".repeat(199))
        );
    }

    #[test]
    fn sanitize_bounds_arrays_and_depth() {
        let s = sanitize_arguments(
            &serde_json::json!({"x": (0..100).collect::<Vec<i32>>(),
            "a": {"b": {"c": {"d": {"e": 1}}}}}),
            &[],
        );
        assert_eq!(s["x"].as_array().unwrap().len(), 21);
        assert_eq!(s["x"][20], "[+80 more]");
        // top-level object = level 1; a container at level 5+ becomes "[nested]"
        assert_eq!(s["a"]["b"]["c"]["d"], "[nested]");
        // scalars and short strings pass through
        let s = sanitize_arguments(&serde_json::json!({"n": 3, "ok": true, "z": null}), &[]);
        assert_eq!(s, serde_json::json!({"n": 3, "ok": true, "z": null}));
    }

    #[test]
    fn sanitize_redacts_by_type_at_any_depth() {
        let s = sanitize_arguments(
            &serde_json::json!({
            "operations": [{"value": 51000}, {"values": ["a", "b", "c"]}],
            "rows": [[1, 2], [3, 4]],
            "query": "SELECT * FROM t WHERE x = 1",
            }),
            &[],
        );
        assert_eq!(s["operations"][0]["value"], "[redacted]");
        assert_eq!(s["operations"][1]["values"], "[redacted: 3 items]");
        assert_eq!(s["rows"], "[redacted: 2 items]");
        assert_eq!(s["query"], "SELECT * FROM t WHERE x = 1");
    }

    #[test]
    fn sanitize_redacts_positional_arguments() {
        // Positional (array) arguments have no names to redact by: the
        // `csv_content` or filter `value` they carry would be logged in
        // clear. The governed call path refuses them; their entry records
        // only their size.
        let filter = serde_json::json!(["small", "cat", "eq", "SECRET-ARR", "f2"]);
        assert_eq!(sanitize_arguments(&filter, &[]), "[redacted: 5 items]");
        let create = serde_json::json!(["pii", "email,income\nalice@example.org,51000\n"]);
        assert_eq!(
            sanitize_arguments(&create, &["name", "csv_content"]),
            "[redacted: 2 items]"
        );
        // Any other top level that is not an object, likewise.
        assert_eq!(
            sanitize_arguments(&Value::String("q".repeat(300)), &[]),
            "[redacted: 300 bytes]"
        );
        assert_eq!(
            sanitize_arguments(&serde_json::json!(51000), &[]),
            "[redacted]"
        );
        assert_eq!(sanitize_arguments(&Value::Bool(true), &[]), "[redacted]");
        // Arguments left out stay null.
        assert_eq!(sanitize_arguments(&Value::Null, &[]), Value::Null);
    }

    #[test]
    fn entry_cuts_a_long_tool_name_on_a_char_boundary() {
        // An unknown tool's name is the caller's string. '€' (3 bytes)
        // spans bytes 199..202; 205 bytes in all.
        let name = format!("{}€{}", "t".repeat(199), "u".repeat(3));
        let e = AuditEntry::new(
            "stdio",
            "stdio",
            &name,
            Value::Null,
            "unknown_tool",
            0,
            ResultSummary::default(),
        );
        assert_eq!(
            e.tool_name,
            format!("{}...[truncated 205 bytes]", "t".repeat(199))
        );
        let short = AuditEntry::new(
            "s",
            "http",
            "head_dataset",
            Value::Null,
            "ok",
            0,
            ResultSummary::default(),
        );
        assert_eq!(short.tool_name, "head_dataset");
    }

    #[test]
    fn sanitize_bounds_object_keys() {
        let wide: serde_json::Map<String, Value> = (0..1000)
            .map(|i| (format!("k{i:04}"), Value::from(i)))
            .collect();
        let long_key = "K".repeat(10_000);
        let mut args = serde_json::Map::new();
        args.insert("wide".to_string(), Value::Object(wide));
        args.insert(long_key, Value::from(1));
        let s = sanitize_arguments(&Value::Object(args), &[]);

        let kept = s["wide"].as_object().unwrap();
        assert_eq!(kept.len(), MAX_OBJECT_KEYS + 1);
        let marker = format!("[+{} more keys]", 1000 - MAX_OBJECT_KEYS);
        assert_eq!(kept.get(&marker), Some(&Value::Null));
        assert_eq!(kept["k0000"], 0);
        // The 10 KB key is cut like a value and keeps its value.
        let cut = format!("{}...[truncated 10000 bytes]", "K".repeat(200));
        assert_eq!(s[&cut], 1, "{s}");
        assert!(s.as_object().unwrap().keys().all(|k| k.len() < 250));
    }

    /// `"a00"` to `"a63"`: 64 small keys that sort before every parameter.
    fn padding_keys() -> serde_json::Map<String, Value> {
        (0..64)
            .map(|i| (format!("a{i:02}"), Value::from(0)))
            .collect()
    }

    /// The count in an object's `"[+n more keys]"` marker (there must be
    /// exactly one) and the number of the other keys.
    fn left_out_and_kept(s: &Value) -> (usize, usize) {
        let keys = s.as_object().unwrap().keys();
        let (markers, kept): (Vec<&String>, Vec<&String>) = keys.partition(|k| is_keys_marker(k));
        assert_eq!(markers.len(), 1, "{s}");
        let count = &markers[0]["[+".len()..markers[0].len() - " more keys]".len()];
        (count.parse().unwrap(), kept.len())
    }

    #[test]
    fn declared_parameters_are_kept_ahead_of_padding() {
        // 64 undeclared keys sorting before `query` would fill the key cap.
        let mut args = padding_keys();
        args.insert("query".into(), "select 1".into());
        let args = Value::Object(args);
        let s = sanitize_arguments(&args, &["query"]);
        assert_eq!(s["query"], "select 1", "{s}");
        assert_eq!(left_out_and_kept(&s), (1, MAX_OBJECT_KEYS));
        // Without the declared list (a call site passing `&[]`) the padding,
        // first in key order and inserted first, takes the cap. The e2e tests
        // in tests/limits_e2e.rs send these arguments through each call site.
        let undeclared = sanitize_arguments(&args, &[]);
        assert!(undeclared.get("query").is_none(), "{undeclared}");
        assert_eq!(left_out_and_kept(&undeclared), (1, MAX_OBJECT_KEYS));
        // One undeclared key with about 24 KB of escapes would fill the
        // byte budget.
        let args = serde_json::json!({"a": vec!["\u{1}".repeat(200); 20], "query": "select 1"});
        let s = sanitize_arguments(&args, &["query"]);
        assert_eq!(s["query"], "select 1", "{s}");
        let a = s["a"].as_array().unwrap();
        assert!(a.len() < 20 && a.last().unwrap().as_str().unwrap().starts_with("[+"));
        assert!(s.to_string().len() <= MAX_ARGUMENT_BYTES + 256);
    }

    #[test]
    fn padding_cannot_push_the_sql_of_a_query_tool_out() {
        let server = crate::server::AnalyticsServer::new();
        let declared = server.declared_parameters("db_sqlite_query");
        assert!(
            ["db_path", "query"].iter().all(|p| declared.contains(p)),
            "{declared:?}"
        );
        // Both paddings at once: a 20 x 20 array of 200-byte strings (about
        // 80 KB) and 64 small keys, all sorting before `db_path` and `query`.
        let mut args = padding_keys();
        let row = Value::Array(vec![Value::String("p".repeat(200)); 20]);
        args.insert("a".into(), Value::Array(vec![row; 20]));
        args.insert("db_path".into(), "x.db".into());
        args.insert("query".into(), "SELECT 1".into());
        let total = args.len();
        let s = sanitize_arguments(&Value::Object(args), &declared);
        assert_eq!(s["query"], "SELECT 1", "{s}");
        assert_eq!(s["db_path"], "x.db", "{s}");
        // The marker counts every padding key that was left out.
        let (left_out, kept) = left_out_and_kept(&s);
        assert!(left_out > 0);
        assert_eq!(left_out + kept, total);
        assert!(s.to_string().len() <= MAX_ARGUMENT_BYTES + 256);
    }

    #[test]
    fn declared_single_values_come_before_declared_containers() {
        // 8000 short strings (about 48 KB) in a parameter that sorts first.
        let cube = serde_json::json!(vec![vec![vec!["abc"; 20]; 20]; 20]);
        let args = serde_json::json!({"columns": cube, "name": "out"});
        let s = sanitize_arguments(&args, &["columns", "name"]);
        assert_eq!(s["name"], "out", "{s}");
        assert!(s["columns"].is_array());
        assert!(s.to_string().len() <= MAX_ARGUMENT_BYTES + 256);
    }

    #[test]
    fn keys_cut_to_one_name_or_spelled_like_the_marker_are_counted() {
        // Two 300-byte keys with the same first 200 bytes cut to one name.
        let head = "k".repeat(200);
        let mut args = serde_json::Map::new();
        args.insert(format!("{head}{}", "a".repeat(100)), Value::from(1));
        args.insert(format!("{head}{}", "b".repeat(100)), Value::from(2));
        let s = sanitize_arguments(&Value::Object(args), &[]);
        let mut expected = serde_json::Map::new();
        expected.insert(format!("{head}...[truncated 300 bytes]"), Value::from(1));
        expected.insert("[+1 more keys]".into(), Value::Null);
        assert_eq!(s, Value::Object(expected));
        // A caller's key can neither replace the marker nor pass for one.
        let s = sanitize_arguments(
            &serde_json::json!({"[+1 more keys]": "x", "[+9 more keys]": null, "b": 1}),
            &[],
        );
        assert_eq!(s, serde_json::json!({"b": 1, "[+2 more keys]": null}));
    }

    #[test]
    fn sanitized_arguments_stay_within_the_byte_budget() {
        // 64 x 64 objects of 200-byte keys and values: about 1.7 MB as
        // JSON, 4.9 MB for the control-character values (6 bytes each).
        for leaf in ["v".repeat(200), "\u{1}".repeat(200)] {
            let inner: serde_json::Map<String, Value> = (0..64)
                .map(|i| (format!("{i:0>200}"), Value::String(leaf.clone())))
                .collect();
            let outer: serde_json::Map<String, Value> = (0..64)
                .map(|i| (format!("o{i:02}"), Value::Object(inner.clone())))
                .collect();
            let input = Value::Object(outer);
            assert!(input.to_string().len() > 1_000_000);
            let s = sanitize_arguments(&input, &[]);
            let len = s.to_string().len();
            assert!(
                len <= MAX_ARGUMENT_BYTES + 256,
                "{len} bytes of sanitised arguments"
            );
            // What was left out is counted, at the top level too.
            let top = s.as_object().unwrap();
            assert!(
                top.keys()
                    .any(|k| k.starts_with("[+") && k.ends_with(" more keys]")),
                "{:?}",
                top.keys().collect::<Vec<_>>()
            );
            assert!(top.contains_key("o00"));
        }
        // Wide arrays nested to the depth limit are bounded the same way.
        let mut v = Value::String("a".repeat(200));
        for _ in 0..4 {
            v = Value::Array(vec![v; 20]);
        }
        let len = sanitize_arguments(&serde_json::json!({ "x": v }), &[])
            .to_string()
            .len();
        assert!(len <= MAX_ARGUMENT_BYTES + 256, "{len} bytes");
    }

    #[test]
    fn json_string_len_counts_quotes_and_escapes() {
        for s in [
            "",
            "abc",
            "é│",
            "a\"b\\c",
            "\n\r\t\u{8}\u{c}",
            "\u{0}\u{1f}\u{7f}",
        ] {
            assert_eq!(
                json_string_len(s),
                serde_json::to_string(s).unwrap().len(),
                "{s:?}"
            );
        }
    }

    #[cfg(feature = "http")]
    #[test]
    fn key_cap_keeps_every_parameter_of_a_tool_request() {
        let widest = crate::server::AnalyticsServer::new()
            .list_tools()
            .into_iter()
            .map(|t| {
                let n = t
                    .input_schema
                    .get("properties")
                    .and_then(Value::as_object)
                    .map_or(0, serde_json::Map::len);
                (n, t.name)
            })
            .max()
            .unwrap();
        assert!(
            widest.0 <= MAX_OBJECT_KEYS,
            "{} has {} parameters, above MAX_OBJECT_KEYS",
            widest.1,
            widest.0
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
    fn call_result_summary_counts_every_content_kind() {
        use rmcp::model::{
            Annotated, CallToolResult, Content, RawAudioContent, RawContent, RawResource,
            ResourceContents,
        };
        let link = RawResource::new("file:///data/a.csv", "a.csv");
        let link_bytes = serde_json::to_vec(&link).unwrap().len();
        let result = CallToolResult::success(vec![
            Content::text("abc"),
            Content::image("aGVsbG8=", "image/png"),
            Content::embedded_text("str:///t", "x".repeat(300)),
            Content::resource(ResourceContents::BlobResourceContents {
                uri: "str:///b".into(),
                mime_type: None,
                blob: "y".repeat(40),
                meta: None,
            }),
            Annotated::new(
                RawContent::Audio(RawAudioContent {
                    data: "z".repeat(16),
                    mime_type: "audio/wav".into(),
                }),
                None,
            ),
            Content::resource_link(link),
        ]);
        let s = ResultSummary::of_call_result(&result);
        assert_eq!(
            (s.content_items, s.text_bytes, s.image_bytes),
            (6, 3, 8),
            "{s:?}"
        );
        assert_eq!(s.other_bytes, 300 + 40 + 16 + link_bytes, "{s:?}");
        assert_eq!(s.output_bytes(), 3 + 8 + s.other_bytes);
        // Recorded in the entry once it is non-zero.
        let j = serde_json::to_value(s).unwrap();
        assert_eq!(j["other_bytes"], s.other_bytes);
    }

    #[test]
    fn entry_records_sizes_not_content() {
        let e = AuditEntry::new(
            "s1",
            "http",
            "head_dataset",
            sanitize_arguments(&serde_json::json!({"dataset": "pii"}), &[]),
            "ok",
            3,
            ResultSummary {
                content_items: 1,
                text_bytes: 600,
                ..ResultSummary::default()
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
            sanitize_arguments(&serde_json::json!({"path": "/data/test.csv"}), &[]),
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

    /// The numbers of the `s<i>` session ids in `path`, in order.
    fn ids_in(path: &Path) -> Vec<usize> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|l| {
                let v: Value = serde_json::from_str(l).unwrap();
                v["session_id"].as_str().unwrap()[1..].parse().unwrap()
            })
            .collect()
    }

    fn writer_at(path: &Path, policy: RotationPolicy) -> Writer {
        Writer {
            path: path.to_path_buf(),
            file: BufWriter::new(open_log(path).unwrap()),
            size: 0,
            policy,
            io_errors: 0,
        }
    }

    #[test]
    fn rotation_leaves_the_numbered_files_when_the_log_is_gone() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.log");
        for k in 1..=3 {
            std::fs::write(numbered(&path, k), format!("old {k}\n")).unwrap();
        }
        let policy = RotationPolicy {
            max_bytes: Some(1 << 20),
            keep: 3,
        };
        let mut writer = writer_at(&path, policy);
        writer.write_entry(&entry(0));
        writer.flush();
        // Removed (or moved away) outside the server when a rotation is due.
        std::fs::remove_file(&path).unwrap();
        writer.rotate();
        for i in 1..4 {
            writer.write_entry(&entry(i));
        }
        writer.flush();
        for k in 1..=3 {
            assert_eq!(
                std::fs::read_to_string(numbered(&path, k)).unwrap(),
                format!("old {k}\n"),
                "{}.{k}",
                path.display()
            );
        }
        assert_eq!(ids_in(&path), [1, 2, 3]);
    }

    #[test]
    fn entries_buffered_when_the_log_is_removed_go_to_the_new_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.log");
        let policy = RotationPolicy {
            max_bytes: None,
            keep: 3,
        };
        let mut writer = writer_at(&path, policy);
        // Removed with an entry still buffered; the next entry finds the
        // path empty.
        writer.write_entry(&entry(0));
        assert!(!writer.file.buffer().is_empty());
        std::fs::remove_file(&path).unwrap();
        writer.write_entry(&entry(1));
        writer.flush();
        assert_eq!(ids_in(&path), [0, 1]);
        assert_eq!(writer.size, std::fs::metadata(&path).unwrap().len());
        // Removed between the last entry of a burst and its flush.
        writer.write_entry(&entry(2));
        std::fs::remove_file(&path).unwrap();
        writer.flush();
        assert_eq!(ids_in(&path), [2]);
        assert_eq!(writer.size, std::fs::metadata(&path).unwrap().len());
        assert_eq!(writer.io_errors, 0);
    }

    #[test]
    fn a_failing_rotation_waits_for_another_max_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.log");
        // `.1` is a non-empty directory: it can be neither removed nor
        // replaced, so every rotation fails (and not with NotFound).
        std::fs::create_dir(numbered(&path, 1)).unwrap();
        std::fs::write(numbered(&path, 1).join("keep"), "x").unwrap();
        let line = serde_json::to_string(&entry(0)).unwrap().len() as u64 + 1;
        let policy = RotationPolicy {
            max_bytes: Some(3 * line + line / 2),
            keep: 1,
        };
        let mut writer = writer_at(&path, policy);
        for i in 0..10 {
            writer.write_entry(&entry(i));
        }
        writer.flush();
        // Due at entries 3, 6 and 9 (two failed steps each), not at every
        // entry from 3 on.
        assert_eq!(writer.io_errors, 6);
        assert_eq!(ids_in(&path), (0..10).collect::<Vec<_>>());
        assert!(numbered(&path, 1).join("keep").exists());
    }

    #[tokio::test]
    async fn a_removed_log_is_recreated_and_the_old_files_kept() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.log");
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
        let rotated: Vec<String> = (1..=3)
            .map(|k| std::fs::read_to_string(numbered(&path, k)).unwrap())
            .collect();

        std::fs::remove_file(&path).unwrap();
        for i in 200..203 {
            logger.log(entry(i)).await;
        }
        logger.flush().await;

        for (k, text) in (1..=3).zip(&rotated) {
            assert_eq!(&std::fs::read_to_string(numbered(&path, k)).unwrap(), text);
        }
        assert_eq!(ids_in(&path), [200, 201, 202]);
    }

    #[cfg(unix)]
    #[test]
    fn a_new_log_never_follows_a_symlink_planted_at_the_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.log");
        let target = dir.path().join("elsewhere");
        std::fs::write(&target, "untouched\n").unwrap();
        let policy = RotationPolicy {
            max_bytes: None,
            keep: 3,
        };
        let mut writer = writer_at(&path, policy);
        writer.write_entry(&entry(0));
        writer.flush();
        // The log is removed, and a symlink appears at its path between the
        // writer's check and its open of a new file.
        std::fs::remove_file(&path).unwrap();
        std::os::unix::fs::symlink(&target, &path).unwrap();
        assert!(!writer.reopen());
        assert_eq!(writer.io_errors, 1);
        // Later entries go to the file at hand; the link is left as it is.
        writer.write_entry(&entry(1));
        writer.flush();
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "untouched\n");
        assert!(
            std::fs::symlink_metadata(&path)
                .unwrap()
                .file_type()
                .is_symlink()
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
            for sub in schema
                .get(key)
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
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
            collect_properties(
                &tool.input_schema,
                &tool.input_schema,
                &mut seen,
                &mut names,
            );
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
