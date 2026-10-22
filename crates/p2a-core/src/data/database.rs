//! Database connectivity for SQLite and DuckDB.
//!
//! Provides functions to execute SQL queries and load results as Polars DataFrames.

use crate::errors::EconError;
use crate::resource::{self, Budget};
use polars::prelude::*;
use std::borrow::Cow;
use std::path::Path;
use std::sync::{Arc, OnceLock, mpsc};
use std::time::{Duration, Instant};
use thiserror::Error;

/// Validate that a string is a safe SQL identifier (table name, column name, etc.).
/// Only allows alphanumeric characters and underscores.
fn validate_identifier(name: &str) -> Result<(), DatabaseError> {
    if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') {
        Ok(())
    } else {
        Err(DatabaseError::UnsupportedType(format!(
            "Invalid SQL identifier: '{}'. Only alphanumeric characters and underscores are allowed.",
            name
        )))
    }
}

/// Database-related errors.
#[derive(Debug, Error)]
pub enum DatabaseError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("DuckDB error: {0}")]
    DuckDB(#[from] duckdb::Error),

    #[error("Polars error: {0}")]
    Polars(#[from] PolarsError),

    #[error("Unsupported column type: {0}")]
    UnsupportedType(String),

    #[error("Database file not found: {0}")]
    FileNotFound(String),

    #[error("Query rejected: {0}")]
    Forbidden(String),

    #[error("Query stopped: {0}")]
    Interrupted(String),
}

/// Validate that a SQL statement is a read-only query.
///
/// Accepts `SELECT`, `WITH` (CTE), `EXPLAIN`, `PRAGMA` (metadata), `DESCRIBE`,
/// and `SHOW`. Anything else — `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`,
/// `CREATE`, `ATTACH`, `COPY`, `INSTALL`, `LOAD`, … — is rejected.
///
/// This is a keyword-level filter, not a full SQL parser. The first keyword is
/// found with the statement lexer of [`ensure_single_statement`], which skips
/// comments as DuckDB does (nested `/* */`, `--` up to an LF or CR), so a
/// comment cannot hide the keyword the engine parses.
///
/// For SQLite it is defence in depth only: the enforcement is the engine
/// configuration (read-only, `query_only`, no ATTACH, schema PRAGMAs only)
/// together with [`ensure_single_statement`]. The DuckDB connections lock the
/// engine too (external access off, extensions off, locked configuration,
/// read-only files), but the file-query connection can write the one file it
/// may read, so for DuckDB this check and the further lexical checks of
/// `validate_duckdb_query` are what keep that file read-only.
pub fn validate_read_only_query(query: &str) -> Result<(), DatabaseError> {
    const ALLOWED: &[&str] = &["select", "with", "explain", "pragma", "describe", "show"];

    let first_token = match SqlLexer::new(query).next_token()? {
        Some((_, SqlToken::Word(w))) => w.to_ascii_lowercase(),
        Some(_) => String::new(),
        None => return Err(DatabaseError::Forbidden("empty SQL statement".to_string())),
    };
    if !ALLOWED.contains(&first_token.as_str()) {
        return Err(DatabaseError::Forbidden(format!(
            "statement starting with `{}` is not allowed; only read-only queries \
             (SELECT, WITH, EXPLAIN, PRAGMA, DESCRIBE, SHOW) may be executed",
            first_token
        )));
    }
    Ok(())
}

/// [`validate_read_only_query`] plus the checks a DuckDB statement needs. The
/// file-query connection may write the one file it may read (DuckDB's
/// `allowed_paths` grants both), so the statement must not be able to write:
/// `EXPLAIN ANALYZE` (also spelled `ANALYSE`, and the parenthesised option
/// form) executes the statement it explains, and a `COPY` keyword anywhere
/// outside strings, quoted identifiers and comments is refused as well.
fn validate_duckdb_query(body: &str) -> Result<(), DatabaseError> {
    validate_read_only_query(body)?;
    let mut lexer = SqlLexer::new(body);
    let mut after_explain = false;
    while let Some((_, token)) = lexer.next_token()? {
        if after_explain
            && match token {
                SqlToken::Word(w) => {
                    w.eq_ignore_ascii_case("analyze") || w.eq_ignore_ascii_case("analyse")
                }
                SqlToken::Byte(b) => b == b'(',
                SqlToken::Quoted => false,
            }
        {
            return Err(DatabaseError::Forbidden(
                "EXPLAIN ANALYZE (or EXPLAIN with options) is not allowed: it executes the \
                 statement it explains; use plain EXPLAIN"
                    .to_string(),
            ));
        }
        after_explain = matches!(token, SqlToken::Word(w) if w.eq_ignore_ascii_case("explain"));
        if matches!(token, SqlToken::Word(w) if w.eq_ignore_ascii_case("copy")) {
            return Err(DatabaseError::Forbidden(
                "COPY is not allowed in a read-only query (quote the name if it is a column)"
                    .to_string(),
            ));
        }
    }
    Ok(())
}

/// The first keyword of `query`, lowercased: its first code token if that is
/// a word, otherwise (or for text the lexer refuses) the empty string.
fn first_keyword(query: &str) -> String {
    match SqlLexer::new(query).next_token() {
        Ok(Some((_, SqlToken::Word(w)))) => w.to_ascii_lowercase(),
        _ => String::new(),
    }
}

/// Refuse any input that is not exactly one SQL statement.
///
/// The text may end in `;`, further `;`, whitespace and comments, but nothing
/// else may follow the first top-level `;`. Strings (`'..'`, `E'..'`), quoted
/// identifiers, dollar quotes (`$$..$$`, `$tag$..$tag$`) and comments (`--`
/// up to an LF or CR, nested `/* */`) are skipped with DuckDB's rules, so a
/// `;` inside them does not count. Unterminated strings or comments, empty input and NUL bytes are
/// refused as well.
pub fn ensure_single_statement(query: &str) -> Result<(), DatabaseError> {
    single_statement_body(query).map(|_| ())
}

const MULTIPLE_STATEMENTS: &str =
    "only one SQL statement may be run per call; remove everything after the first ';'";
const UNTERMINATED: &str = "unterminated string literal, quoted identifier or comment";

/// Bytes that may continue an identifier (DuckDB: letters, digits, `_`, `$`,
/// and any non-ASCII byte).
fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'$' || b >= 0x80
}

/// The single statement in `query`, without its terminating `;` and anything
/// after it. The text is not trimmed: a trailing line comment stays part of
/// the body (callers that wrap the body put a newline after it).
fn single_statement_body(query: &str) -> Result<&str, DatabaseError> {
    let bytes = query.as_bytes();
    if bytes.contains(&0) {
        return Err(DatabaseError::Forbidden(
            "the query contains a NUL byte".to_string(),
        ));
    }
    let unterminated = || DatabaseError::Forbidden(UNTERMINATED.to_string());

    let mut lexer = SqlLexer::new(query);
    let mut has_code = false;
    let end = loop {
        match lexer.next_token()? {
            Some((i, SqlToken::Byte(b';'))) => break i,
            Some(_) => has_code = true,
            None if has_code => return Ok(query),
            None => return Err(DatabaseError::Forbidden("empty SQL statement".to_string())),
        }
    };

    // After the first top-level `;`: only blanks, `;` and terminated comments.
    let n = bytes.len();
    let mut j = end + 1;
    while j < n {
        let b = bytes[j];
        let next = bytes.get(j + 1).copied();
        match b {
            b';' => j += 1,
            b if b.is_ascii_whitespace() => j += 1,
            b'-' if next == Some(b'-') => j = skip_line_comment(bytes, j + 2),
            b'/' if next == Some(b'*') => {
                j = skip_block_comment(bytes, j + 2).ok_or_else(unterminated)?;
            }
            _ => return Err(DatabaseError::Forbidden(MULTIPLE_STATEMENTS.to_string())),
        }
    }
    if !has_code {
        return Err(DatabaseError::Forbidden("empty SQL statement".to_string()));
    }
    Ok(&query[..end])
}

/// Index just past the closing `'` of a string whose body starts at `i`.
/// `''` stays inside the string; in an escape string (`E'..'`) a backslash
/// also skips the next byte. `None` if the string is unterminated.
fn skip_string(bytes: &[u8], mut i: usize, escape: bool) -> Option<usize> {
    while i < bytes.len() {
        match bytes[i] {
            b'\\' if escape => i += 2,
            b'\'' if bytes.get(i + 1) == Some(&b'\'') => i += 2,
            b'\'' => return Some(i + 1),
            _ => i += 1,
        }
    }
    None
}

/// Index just past the closing `"` of a quoted identifier whose body starts at `i`.
fn skip_quoted_identifier(bytes: &[u8], mut i: usize) -> Option<usize> {
    while i < bytes.len() {
        match bytes[i] {
            b'"' if bytes.get(i + 1) == Some(&b'"') => i += 2,
            b'"' => return Some(i + 1),
            _ => i += 1,
        }
    }
    None
}

/// Length of the dollar-quote delimiter (`$$` or `$tag$`) at the start of `s`,
/// which begins with `$`; `None` if `s` does not start a dollar quote (`$1`).
fn dollar_tag_len(s: &[u8]) -> Option<usize> {
    let tag_start = |b: u8| b.is_ascii_alphabetic() || b == b'_' || b >= 0x80;
    let tag_byte = |b: u8| b.is_ascii_alphanumeric() || b == b'_' || b >= 0x80;
    match s.get(1) {
        Some(b'$') => Some(2),
        Some(&b) if tag_start(b) => {
            let mut k = 2;
            while k < s.len() && tag_byte(s[k]) {
                k += 1;
            }
            (s.get(k) == Some(&b'$')).then_some(k + 1)
        }
        _ => None,
    }
}

/// Whether `b` ends a `--` line comment. DuckDB's scanner (libpg_query
/// `scan.l`, `newline [\n\r]`) ends it at a CR as well as at an LF.
fn is_newline(b: u8) -> bool {
    b == b'\n' || b == b'\r'
}

/// Index of the byte after the newline (LF or CR, see [`is_newline`]) ending a
/// line comment whose body starts at `i`, or the end of the input.
fn skip_line_comment(bytes: &[u8], i: usize) -> usize {
    let i = i.min(bytes.len());
    match bytes[i..].iter().position(|&b| is_newline(b)) {
        Some(p) => i + p + 1,
        None => bytes.len(),
    }
}

/// Index just past the `*/` closing a (nested) block comment whose body starts
/// at `i`; `None` if it is unterminated.
fn skip_block_comment(bytes: &[u8], mut i: usize) -> Option<usize> {
    let mut depth = 1usize;
    while i < bytes.len() {
        match (bytes[i], bytes.get(i + 1)) {
            (b'/', Some(b'*')) => {
                depth += 1;
                i += 2;
            }
            (b'*', Some(b'/')) => {
                depth -= 1;
                i += 2;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => i += 1,
        }
    }
    None
}

/// One code token of SQL text: a word (keyword, identifier or number), a
/// quoted item (string, escape string, quoted identifier, dollar quote), or
/// any other single byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SqlToken<'a> {
    Word(&'a str),
    Quoted,
    Byte(u8),
}

/// Splits SQL text into [`SqlToken`]s with DuckDB's lexical rules, skipping
/// whitespace and comments (`--` up to an LF or CR, nested `/* */`). Every
/// check that looks at the code of a query goes through it, so a comment hides
/// the same text from the checks as from the engine.
struct SqlLexer<'a> {
    query: &'a str,
    i: usize,
}

impl<'a> SqlLexer<'a> {
    fn new(query: &'a str) -> Self {
        Self { query, i: 0 }
    }

    /// The next code token and the byte offset it starts at, or `None` at the
    /// end of the text. Unterminated strings, quoted identifiers and comments
    /// are refused, and so is a quote after a digit and `E` (see below).
    fn next_token(&mut self) -> Result<Option<(usize, SqlToken<'a>)>, DatabaseError> {
        let bytes = self.query.as_bytes();
        let unterminated = || DatabaseError::Forbidden(UNTERMINATED.to_string());
        while self.i < bytes.len() {
            let i = self.i;
            let b = bytes[i];
            let next = bytes.get(i + 1).copied();
            let token = match b {
                b'-' if next == Some(b'-') => {
                    self.i = skip_line_comment(bytes, i + 2);
                    continue;
                }
                b'/' if next == Some(b'*') => {
                    self.i = skip_block_comment(bytes, i + 2).ok_or_else(unterminated)?;
                    continue;
                }
                b if b.is_ascii_whitespace() => {
                    self.i += 1;
                    continue;
                }
                b'\'' => {
                    let after_e = i >= 1 && matches!(bytes[i - 1], b'E' | b'e');
                    // DuckDB's `realfail1` rule gives the trailing E/e back after a
                    // numeric literal, so `1E'..'` and `1.E'..'` lex there as a
                    // number followed by an escape string. The byte before the E is
                    // then a digit or `.`, and the quote still starts an escape
                    // string whose `\'` the engine would honour. The lexer cannot
                    // tell that apart from a plain string cheaply, so it fails
                    // closed (over-rejection is acceptable per the plan).
                    if after_e && i >= 2 && (bytes[i - 2].is_ascii_digit() || bytes[i - 2] == b'.')
                    {
                        return Err(DatabaseError::Forbidden(
                            "a quote after a digit and `E` is ambiguous (DuckDB reads it as an \
                             escape string); separate the number from the string"
                                .to_string(),
                        ));
                    }
                    // An escape string (`E'..'`) when the E/e is a standalone token,
                    // i.e. not the tail of an identifier.
                    let escape = after_e && (i < 2 || !is_ident_byte(bytes[i - 2]));
                    self.i = skip_string(bytes, i + 1, escape).ok_or_else(unterminated)?;
                    SqlToken::Quoted
                }
                b'"' => {
                    self.i = skip_quoted_identifier(bytes, i + 1).ok_or_else(unterminated)?;
                    SqlToken::Quoted
                }
                b'$' if i == 0 || !is_ident_byte(bytes[i - 1]) => match dollar_tag_len(&bytes[i..])
                {
                    Some(len) => {
                        let tag = &bytes[i..i + len];
                        let start = i + len;
                        let close = bytes[start..]
                            .windows(len)
                            .position(|w| w == tag)
                            .ok_or_else(unterminated)?;
                        self.i = start + close + len;
                        SqlToken::Quoted
                    }
                    None => {
                        self.i += 1;
                        SqlToken::Byte(b'$')
                    }
                },
                b if is_ident_byte(b) => {
                    // Every non-ASCII byte is an identifier byte, so a word
                    // never starts or ends inside a UTF-8 sequence.
                    let len = bytes[i..].iter().take_while(|&&c| is_ident_byte(c)).count();
                    self.i = i + len;
                    SqlToken::Word(&self.query[i..i + len])
                }
                _ => {
                    self.i += 1;
                    SqlToken::Byte(b)
                }
            };
            return Ok(Some((i, token)));
        }
        Ok(None)
    }
}

/// Bounds for one SQL query. The default has none: the engine lockdown
/// (read-only, no file access, no extensions, locked configuration) always
/// applies, but rows, bytes, memory, threads and time are unbounded.
#[derive(Debug, Clone, Default)]
pub struct QueryOptions {
    /// Rows kept; also the engine-side `LIMIT` (+1) for `SELECT`/`WITH` statements.
    pub max_rows: Option<usize>,
    /// Estimated result bytes kept; for SQLite also the largest value (`SQLITE_LIMIT_LENGTH`).
    pub max_bytes: Option<u64>,
    /// Wall-clock limit for the query. The query also stops when the installed
    /// [`resource`] budget is cancelled or past its deadline.
    pub timeout: Option<Duration>,
    /// DuckDB `memory_limit` in MiB (spilling to disk is then disabled). SQLite ignores it.
    pub memory_limit_mb: Option<u64>,
    /// DuckDB worker threads. SQLite ignores it.
    pub threads: Option<u32>,
}

/// Result of a database query.
#[derive(Debug)]
pub struct QueryResult {
    /// The resulting DataFrame
    pub dataframe: DataFrame,
    /// Number of rows returned
    pub rows: usize,
    /// Column names
    pub columns: Vec<String>,
}

impl std::fmt::Display for QueryResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Query Result")?;
        writeln!(f, "============")?;
        writeln!(f, "Rows returned: {}", self.rows)?;
        writeln!(f, "Columns: {}", self.columns.join(", "))?;
        writeln!(f)?;
        writeln!(f, "Preview (first 5 rows):")?;
        write!(f, "{}", self.dataframe.head(Some(5)))
    }
}

/// The result of a bounded query (`query_*_with`).
#[derive(Debug)]
pub struct QueryOutcome {
    /// The rows that were kept.
    pub result: QueryResult,
    /// `true` when at least one further row existed, or a row did not fit the
    /// byte budget, so `result` holds only the first rows.
    pub truncated: bool,
    /// Number of rows kept (equal to `result.rows`).
    pub rows_returned: usize,
}

/// Rows collected column by column, stopping at the row and byte budgets.
struct RowCollector<V> {
    columns: Vec<Vec<V>>,
    rows: usize,
    bytes: u64,
    max_rows: Option<usize>,
    max_bytes: Option<u64>,
    truncated: bool,
}

impl<V> RowCollector<V> {
    fn new(n_columns: usize, opts: &QueryOptions) -> Self {
        Self {
            columns: (0..n_columns).map(|_| Vec::new()).collect(),
            rows: 0,
            bytes: 0,
            max_rows: opts.max_rows,
            max_bytes: opts.max_bytes,
            truncated: false,
        }
    }

    /// Called when the engine produced one more row, before it is read:
    /// `false` (and the result marked truncated) when the row budget is used up.
    fn wants_row(&mut self) -> bool {
        if self.max_rows.is_some_and(|max| self.rows >= max) {
            self.truncated = true;
            return false;
        }
        true
    }

    /// Keep the values in `row` (one per column, estimated at `row_bytes`),
    /// draining it. `false` (and the result marked truncated) when the row
    /// would exceed the byte budget; it is then not kept.
    fn push(&mut self, row: &mut Vec<V>, row_bytes: u64) -> bool {
        let total = self.bytes.saturating_add(row_bytes);
        if self.max_bytes.is_some_and(|max| total > max) {
            self.truncated = true;
            return false;
        }
        for (column, value) in self.columns.iter_mut().zip(row.drain(..)) {
            column.push(value);
        }
        self.bytes = total;
        self.rows += 1;
        true
    }
}

/// Assemble the collected columns into a [`QueryOutcome`], converting each
/// column by consuming it (peak memory: the values plus one converted column).
fn finish_outcome<V>(
    collector: RowCollector<V>,
    column_names: Vec<String>,
    to_series: impl Fn(&str, Vec<V>) -> Series,
) -> Result<QueryOutcome, DatabaseError> {
    let RowCollector {
        columns,
        rows,
        truncated,
        ..
    } = collector;
    let columns_vec: Vec<Column> = column_names
        .iter()
        .zip(columns)
        .map(|(name, values)| to_series(name, values).into())
        .collect();
    let dataframe = DataFrame::new(columns_vec)?;
    Ok(QueryOutcome {
        result: QueryResult {
            dataframe,
            rows,
            columns: column_names,
        },
        truncated,
        rows_returned: rows,
    })
}

// ============================================================================
// Deadlines, cancellation and memory growth
// ============================================================================

/// How often the DuckDB watchdog checks its stop conditions.
const WATCH_POLL: Duration = Duration::from_millis(20);
/// Rows between two budget checks in the row loops.
const CHECK_EVERY_ROWS: usize = 1024;
/// SQLite virtual-machine instructions between two progress-handler calls.
const SQLITE_PROGRESS_OPS: i32 = 1000;
const MIB: usize = 1024 * 1024;

/// The text of a failed budget check, without the "Computation stopped:" prefix.
fn budget_message(err: EconError) -> String {
    match err {
        EconError::Cancelled(message) => message,
        other => other.to_string(),
    }
}

/// [`resource::checkpoint`] as a [`DatabaseError::Interrupted`].
fn checkpoint() -> Result<(), DatabaseError> {
    resource::checkpoint().map_err(|e| DatabaseError::Interrupted(budget_message(e)))
}

/// The RSS guard of a query: `(RSS at the start, allowed growth)` in bytes,
/// the growth being twice the memory limit. DuckDB allocates materialised
/// results and string data outside `memory_limit`, so this is the only bound
/// on them. Active only with a memory limit and where RSS is readable (Linux).
fn rss_guard(opts: &QueryOptions) -> Option<(usize, usize)> {
    let mb = opts.memory_limit_mb?;
    let start = crate::memory::get_process_memory()?.rss_bytes;
    let growth = usize::try_from(mb)
        .unwrap_or(usize::MAX)
        .saturating_mul(2 * MIB);
    Some((start, growth))
}

/// Why a running query must stop, if it must: its own time limit, the
/// installed budget (cancelled or past its deadline), or RSS growth past the
/// guard, checked in that order.
fn stop_reason(
    started: Instant,
    timeout: Option<Duration>,
    budget: Option<&Budget>,
    rss_guard: Option<(usize, usize)>,
) -> Option<String> {
    if let Some(limit) = timeout
        && started.elapsed() >= limit
    {
        return Some(format!(
            "the query exceeded its time limit of {:.1} s",
            limit.as_secs_f64()
        ));
    }
    if let Some(budget) = budget
        && let Err(e) = budget.check()
    {
        return Some(budget_message(e));
    }
    if let Some((start, growth)) = rss_guard
        && let Some(now) = crate::memory::get_process_memory()
        && now.rss_bytes > start.saturating_add(growth)
    {
        return Some(format!(
            "the server's memory grew by more than {} MiB while the query ran \
             (twice its {} MiB memory limit)",
            growth / MIB,
            growth / MIB / 2
        ));
    }
    None
}

/// A watchdog thread that interrupts a running DuckDB query once
/// [`stop_reason`] gives a reason. It re-issues the interrupt on every poll
/// until it is dropped, because an interrupt that arrives before DuckDB starts
/// executing is lost. Dropping it stops and joins the thread.
///
/// A polling thread rather than one timer armed at the deadline: a cancelled
/// budget has no wake-up signal, and polling [`Budget::check`] sees both.
struct QueryWatch {
    reason: Arc<OnceLock<String>>,
    stop: Option<mpsc::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl QueryWatch {
    /// Arm a watchdog for the query about to run on `conn`; `None` when
    /// nothing can stop it (no timeout, no installed budget, no RSS guard).
    /// Fails closed when a timeout or budget applies and no thread can start.
    fn arm(conn: &duckdb::Connection, opts: &QueryOptions) -> Result<Option<Self>, DatabaseError> {
        let budget = resource::current_budget();
        let rss = rss_guard(opts);
        let timeout = opts.timeout;
        if timeout.is_none() && budget.is_none() && rss.is_none() {
            return Ok(None);
        }
        let required = timeout.is_some() || budget.is_some();
        let handle = conn.interrupt_handle();
        let reason = Arc::new(OnceLock::new());
        let seen = Arc::clone(&reason);
        let (stop, stopped) = mpsc::channel::<()>();
        let started = Instant::now();
        let spawned = std::thread::Builder::new()
            .name("p2a-sql-watchdog".to_string())
            .spawn(move || {
                while let Err(mpsc::RecvTimeoutError::Timeout) = stopped.recv_timeout(WATCH_POLL) {
                    if seen.get().is_none()
                        && let Some(why) = stop_reason(started, timeout, budget.as_ref(), rss)
                    {
                        let _ = seen.set(why);
                    }
                    if seen.get().is_some() {
                        handle.interrupt();
                    }
                }
            });
        match spawned {
            Ok(thread) => Ok(Some(Self {
                reason,
                stop: Some(stop),
                thread: Some(thread),
            })),
            Err(e) if required => Err(DatabaseError::Interrupted(format!(
                "could not start the query watchdog: {e}"
            ))),
            Err(_) => Ok(None),
        }
    }

    /// Why the watchdog stopped the query, once it has.
    fn reason(&self) -> Option<&str> {
        self.reason.get().map(String::as_str)
    }
}

impl Drop for QueryWatch {
    fn drop(&mut self) {
        drop(self.stop.take());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Map an engine error to [`DatabaseError::Interrupted`] when a stop reason
/// was recorded: the error is then the interrupt that reason caused.
fn stopped_by(reason: Option<&str>, err: DatabaseError) -> DatabaseError {
    match reason {
        Some(why) => DatabaseError::Interrupted(why.to_string()),
        None => err,
    }
}

// ============================================================================
// SQLite Support
// ============================================================================

/// Read-only schema PRAGMAs that an open [`SqliteConnection`] admits. Every
/// other PRAGMA is refused by the authorizer, because SQLite applies many
/// PRAGMAs while it *compiles* them — inside `prepare`, before
/// `Statement::readonly()` can be checked — and some of them change
/// process-global state (`hard_heap_limit`, `temp_store_directory`, …) on a
/// read-only connection and need no valid database.
const SQLITE_SCHEMA_PRAGMAS: &[&str] = &[
    "table_info",
    "table_xinfo",
    "table_list",
    "index_list",
    "index_info",
    "index_xinfo",
    "foreign_key_list",
    "database_list",
    "collation_list",
];

/// A locked SQLite connection together with the slot its authorizer writes the
/// name of a refused PRAGMA into (so the caller can turn the resulting
/// "not authorized" error into a clear [`DatabaseError::Forbidden`]).
struct SqliteConnection {
    conn: rusqlite::Connection,
    denied_pragma: Arc<OnceLock<String>>,
}

/// The path handed to SQLite's `open`. The bundled build compiles
/// `-DSQLITE_USE_URI`, so SQLite parses any filename that *begins with*
/// `file:` as a URI (`mode=`, `vfs=`, `immutable=`, `cache=`), whether or not
/// `SQLITE_OPEN_URI` is passed. A relative path is prefixed with `.` so it can
/// never begin with `file:`; an absolute path (what the MCP jail always passes)
/// cannot begin with `file:` already and is left unchanged.
fn sqlite_open_path(path: &Path) -> Cow<'_, Path> {
    if path.is_relative() {
        Cow::Owned(Path::new(".").join(path))
    } else {
        Cow::Borrowed(path)
    }
}

/// Open a SQLite database for queries: read-only, never as a `file:` URI (so no
/// `mode=`/`vfs=` parameters, see [`sqlite_open_path`]), no attached databases,
/// `query_only` on, an authorizer that admits only the read-only schema PRAGMAs
/// ([`SQLITE_SCHEMA_PRAGMAS`]), and, when a byte budget is set, no value larger
/// than that budget.
fn open_sqlite_locked(path: &Path, opts: &QueryOptions) -> Result<SqliteConnection, DatabaseError> {
    use rusqlite::hooks::{AuthAction, AuthContext, Authorization};
    use rusqlite::limits::Limit;
    let flags =
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX;
    let conn = rusqlite::Connection::open_with_flags(sqlite_open_path(path).as_ref(), flags)?;
    conn.set_limit(Limit::SQLITE_LIMIT_ATTACHED, 0)?;
    if let Some(bytes) = opts.max_bytes {
        // A larger value could never be returned within the byte budget anyway.
        let len = i32::try_from(bytes).unwrap_or(i32::MAX);
        conn.set_limit(Limit::SQLITE_LIMIT_LENGTH, len)?;
    }
    conn.execute_batch("PRAGMA query_only = ON")?;

    // Arm the authorizer only after the setup PRAGMA above, which it would
    // otherwise refuse. From here on only the schema PRAGMAs are allowed.
    let denied_pragma: Arc<OnceLock<String>> = Arc::new(OnceLock::new());
    let denied = Arc::clone(&denied_pragma);
    conn.authorizer(Some(move |ctx: AuthContext<'_>| match ctx.action {
        AuthAction::Pragma { pragma_name, .. }
            if SQLITE_SCHEMA_PRAGMAS.contains(&pragma_name.to_ascii_lowercase().as_str()) =>
        {
            Authorization::Allow
        }
        AuthAction::Pragma { pragma_name, .. } => {
            let _ = denied.set(pragma_name.to_string());
            Authorization::Deny
        }
        _ => Authorization::Allow,
    }));

    Ok(SqliteConnection {
        conn,
        denied_pragma,
    })
}

/// Map a `prepare`/`query` error from a connection with a denied PRAGMA to a
/// clear [`DatabaseError::Forbidden`], leaving other errors untouched.
fn denied_pragma_error(
    denied: &OnceLock<String>,
    fallback: impl FnOnce() -> DatabaseError,
) -> DatabaseError {
    match denied.get() {
        Some(name) => DatabaseError::Forbidden(format!(
            "PRAGMA {name} is not allowed; only schema PRAGMAs \
             (table_info, table_list, index_list, ...) may be run"
        )),
        None => fallback(),
    }
}

/// Execute a SQL query against a SQLite database and return results as a DataFrame.
///
/// # Arguments
/// * `db_path` - Path to the SQLite database file
/// * `query` - SQL query to execute
///
/// # Returns
/// QueryResult containing the DataFrame and metadata
pub fn query_sqlite(db_path: impl AsRef<Path>, query: &str) -> Result<QueryResult, DatabaseError> {
    query_sqlite_with(db_path, query, &QueryOptions::default()).map(|o| o.result)
}

/// [`query_sqlite`] with bounds: rows are streamed and collection stops at
/// `opts.max_rows` or the estimated `opts.max_bytes` (`truncated` is then set);
/// no single value may exceed `opts.max_bytes`. `memory_limit_mb` and
/// `threads` apply to DuckDB only.
pub fn query_sqlite_with(
    db_path: impl AsRef<Path>,
    query: &str,
    opts: &QueryOptions,
) -> Result<QueryOutcome, DatabaseError> {
    checkpoint()?;
    let body = single_statement_body(query)?;
    validate_read_only_query(body)?;
    let path = db_path.as_ref();

    if !path.exists() {
        return Err(DatabaseError::FileNotFound(path.display().to_string()));
    }

    // Read-only, query_only, no ATTACH, schema PRAGMAs only. rusqlite's prepare
    // compiles only the first statement; the lexer above already refused any
    // second one.
    let SqliteConnection {
        conn,
        denied_pragma,
    } = open_sqlite_locked(path, opts)?;

    // The progress handler interrupts the statement at the time limit or when
    // the installed budget is cancelled or past its deadline.
    let reason: Arc<OnceLock<String>> = Arc::new(OnceLock::new());
    let budget = resource::current_budget();
    if opts.timeout.is_some() || budget.is_some() {
        let seen = Arc::clone(&reason);
        let timeout = opts.timeout;
        let started = Instant::now();
        conn.progress_handler(
            SQLITE_PROGRESS_OPS,
            Some(move || {
                if seen.get().is_some() {
                    return true;
                }
                match stop_reason(started, timeout, budget.as_ref(), None) {
                    Some(why) => {
                        let _ = seen.set(why);
                        true
                    }
                    None => false,
                }
            }),
        );
    }
    let stopped = |e: rusqlite::Error| {
        // A PRAGMA refused by the authorizer surfaces as a "not authorized"
        // error during prepare; name it.
        denied_pragma_error(&denied_pragma, || {
            // A value over SQLITE_LIMIT_LENGTH surfaces as the bare "string or
            // blob too big"; name the server's result size limit instead.
            if opts.max_bytes.is_some()
                && e.sqlite_error_code() == Some(rusqlite::ErrorCode::TooBig)
            {
                let b = opts.max_bytes.unwrap();
                return DatabaseError::Forbidden(format!(
                    "a value is too big for this server's result size limit of {b} bytes"
                ));
            }
            stopped_by(reason.get().map(String::as_str), e.into())
        })
    };

    let mut stmt = conn.prepare(body).map_err(stopped)?;
    if !stmt.readonly() {
        return Err(DatabaseError::Forbidden(
            "the statement would modify the database; only read-only queries are allowed"
                .to_string(),
        ));
    }

    let column_count = stmt.column_count();
    let column_names: Vec<String> = stmt.column_names().iter().map(|s| s.to_string()).collect();

    // SQLite streams: each `next` steps the statement once.
    let mut collector = RowCollector::new(column_count, opts);
    let mut row_values = Vec::with_capacity(column_count);
    let mut rows = stmt.query([]).map_err(stopped)?;
    while let Some(row) = rows.next().map_err(stopped)? {
        if !collector.wants_row() {
            break;
        }
        if collector.rows % CHECK_EVERY_ROWS == CHECK_EVERY_ROWS - 1 {
            checkpoint()?;
            if let Some(why) = reason.get() {
                return Err(DatabaseError::Interrupted(why.clone()));
            }
        }
        let mut row_bytes = 0u64;
        for i in 0..column_count {
            let value: rusqlite::types::Value = row.get_ref(i).map_err(stopped)?.into();
            row_bytes = row_bytes.saturating_add(sqlite_value_bytes(&value));
            row_values.push(value);
        }
        if !collector.push(&mut row_values, row_bytes) {
            break;
        }
    }
    drop(rows);

    finish_outcome(collector, column_names, sqlite_column_to_series)
}

/// Estimated size of one SQLite value: numbers 8 bytes, text and blobs their length.
fn sqlite_value_bytes(value: &rusqlite::types::Value) -> u64 {
    use rusqlite::types::Value;
    match value {
        Value::Null => 0,
        Value::Integer(_) | Value::Real(_) => 8,
        Value::Text(s) => s.len() as u64,
        Value::Blob(b) => b.len() as u64,
    }
}

/// One SQLite result column as a Series, typed by its first non-null value.
fn sqlite_column_to_series(name: &str, values: Vec<rusqlite::types::Value>) -> Series {
    use rusqlite::types::{Type, Value};
    let col_type = values
        .iter()
        .find_map(|v| match v {
            Value::Null => None,
            v => Some(v.data_type()),
        })
        .unwrap_or(Type::Text);

    match col_type {
        Type::Integer => {
            let values: Vec<Option<i64>> = values
                .into_iter()
                .map(|v| match v {
                    Value::Integer(i) => Some(i),
                    _ => None,
                })
                .collect();
            Series::new(name.into(), values)
        }
        Type::Real => {
            let values: Vec<Option<f64>> = values
                .into_iter()
                .map(|v| match v {
                    Value::Real(f) => Some(f),
                    Value::Integer(i) => Some(i as f64),
                    _ => None,
                })
                .collect();
            Series::new(name.into(), values)
        }
        Type::Text | Type::Blob => {
            let values: Vec<Option<String>> = values
                .into_iter()
                .map(|v| match v {
                    Value::Text(s) => Some(s),
                    Value::Blob(b) => Some(String::from_utf8_lossy(&b).into_owned()),
                    Value::Null => None,
                    Value::Integer(i) => Some(i.to_string()),
                    Value::Real(f) => Some(f.to_string()),
                })
                .collect();
            Series::new(name.into(), values)
        }
        Type::Null => {
            // All nulls - default to string
            let values: Vec<Option<String>> = vec![None; values.len()];
            Series::new(name.into(), values)
        }
    }
}

/// List all tables in a SQLite database.
pub fn list_sqlite_tables(db_path: impl AsRef<Path>) -> Result<Vec<String>, DatabaseError> {
    let path = db_path.as_ref();

    if !path.exists() {
        return Err(DatabaseError::FileNotFound(path.display().to_string()));
    }

    let conn = open_sqlite_locked(path, &QueryOptions::default())?.conn;
    let mut stmt =
        conn.prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")?;

    let tables: Vec<String> = stmt
        .query_map([], |row| row.get(0))?
        .filter_map(|r| r.ok())
        .collect();

    Ok(tables)
}

/// Get schema information for a SQLite table.
pub fn sqlite_table_schema(
    db_path: impl AsRef<Path>,
    table_name: &str,
) -> Result<Vec<(String, String)>, DatabaseError> {
    let path = db_path.as_ref();

    if !path.exists() {
        return Err(DatabaseError::FileNotFound(path.display().to_string()));
    }

    let conn = open_sqlite_locked(path, &QueryOptions::default())?.conn;
    validate_identifier(table_name)?;
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({})", table_name))?;

    let schema: Vec<(String, String)> = stmt
        .query_map([], |row| {
            Ok((row.get::<_, String>(1)?, row.get::<_, String>(2)?))
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(schema)
}

// ============================================================================
// DuckDB Support
// ============================================================================

/// What a DuckDB connection is opened for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DuckDbKind {
    /// A database file, opened read-only.
    File,
    /// A fresh `:memory:` database.
    Memory,
    /// The in-memory connection of [`query_file_with_duckdb`]; external access
    /// is disabled and the configuration locked after `allowed_paths` is set.
    FileQuery,
}

/// The startup configuration of every DuckDB connection. The order matters:
/// `temp_directory` can no longer be changed once external access is off, and
/// `lock_configuration` must come last.
fn duckdb_config(opts: &QueryOptions, kind: DuckDbKind) -> Result<duckdb::Config, DatabaseError> {
    let mut config = duckdb::Config::default();
    if let Some(mb) = opts.memory_limit_mb {
        // No temp directory: buffer-managed operators cannot spill past the limit.
        config = config
            .with("memory_limit", format!("{mb}MiB"))?
            .with("temp_directory", "")?;
    }
    if let Some(n) = opts.threads {
        config = config.threads(i64::from(n))?;
    }
    config = config
        .enable_autoload_extension(false)?
        .with("allow_community_extensions", "false")?;
    if kind == DuckDbKind::File {
        config = config.access_mode(duckdb::AccessMode::ReadOnly)?;
    }
    if kind != DuckDbKind::FileQuery {
        config = config
            .enable_external_access(false)?
            .with("lock_configuration", "true")?;
    }
    Ok(config)
}

/// Open a DuckDB database (`:memory:` or a file, the latter read-only) with
/// external access, extension install/load and reconfiguration disabled by the
/// engine itself.
fn open_duckdb_locked(
    path: &Path,
    opts: &QueryOptions,
) -> Result<duckdb::Connection, DatabaseError> {
    if path.to_string_lossy() == ":memory:" {
        let config = duckdb_config(opts, DuckDbKind::Memory)?;
        Ok(duckdb::Connection::open_in_memory_with_flags(config)?)
    } else {
        let config = duckdb_config(opts, DuckDbKind::File)?;
        Ok(duckdb::Connection::open_with_flags(path, config)?)
    }
}

/// `file` as a single-quoted SQL string literal (`'` doubled).
fn quoted_file_literal(file: &Path) -> Result<String, DatabaseError> {
    let s = file
        .to_str()
        .ok_or_else(|| DatabaseError::Forbidden("the file path is not valid UTF-8".to_string()))?;
    Ok(format!("'{}'", s.replace('\'', "''")))
}

/// An in-memory DuckDB connection that can read exactly `file` and nothing
/// else: `allowed_paths` names the file, then external access is disabled and
/// the configuration locked, so ATTACH, COPY and the file readers refuse every
/// other path.
///
/// The engine leaves `file` itself writable: `allowed_paths` grants write
/// access as well as read. What keeps it read-only are the lexical checks run
/// before the statement reaches this connection ([`ensure_single_statement`],
/// [`validate_read_only_query`] and `validate_duckdb_query`: one statement,
/// a read-only first keyword, no `EXPLAIN ANALYZE`, no `COPY`).
fn open_file_query_connection(
    file: &Path,
    opts: &QueryOptions,
) -> Result<duckdb::Connection, DatabaseError> {
    let literal = quoted_file_literal(file)?;
    let config = duckdb_config(opts, DuckDbKind::FileQuery)?;
    let conn = duckdb::Connection::open_in_memory_with_flags(config)?;
    conn.execute_batch(&format!("SET allowed_paths = [{literal}]"))?;
    conn.execute_batch("SET enable_external_access = false")?;
    conn.execute_batch("SET lock_configuration = true")?;
    Ok(conn)
}

/// Execute a SQL query against a DuckDB database and return results as a DataFrame.
///
/// # Arguments
/// * `db_path` - Path to the DuckDB database file (use ":memory:" for in-memory)
/// * `query` - SQL query to execute
///
/// # Returns
/// QueryResult containing the DataFrame and metadata
pub fn query_duckdb(db_path: impl AsRef<Path>, query: &str) -> Result<QueryResult, DatabaseError> {
    query_duckdb_with(db_path, query, &QueryOptions::default()).map(|o| o.result)
}

/// [`query_duckdb`] with bounds. A `SELECT`/`WITH` statement is wrapped in
/// `LIMIT max_rows + 1` so DuckDB itself stops producing rows; collection stops
/// at `opts.max_rows` or the estimated `opts.max_bytes` (`truncated` is then
/// set). `memory_limit_mb` and `threads` configure the engine.
pub fn query_duckdb_with(
    db_path: impl AsRef<Path>,
    query: &str,
    opts: &QueryOptions,
) -> Result<QueryOutcome, DatabaseError> {
    checkpoint()?;
    let body = single_statement_body(query)?;
    validate_duckdb_query(body)?;
    let path = db_path.as_ref();
    let path_str = path.to_string_lossy();

    // Allow :memory: for in-memory databases
    if path_str != ":memory:" && !path.exists() {
        return Err(DatabaseError::FileNotFound(path.display().to_string()));
    }

    // Locked by the engine: no file access, no extensions, no reconfiguration;
    // file databases are also read-only.
    let conn = open_duckdb_locked(path, opts)?;
    run_duckdb_query(&conn, body, opts)
}

/// List all tables in a DuckDB database.
pub fn list_duckdb_tables(db_path: impl AsRef<Path>) -> Result<Vec<String>, DatabaseError> {
    let path = db_path.as_ref();
    let path_str = path.to_string_lossy();

    if path_str != ":memory:" && !path.exists() {
        return Err(DatabaseError::FileNotFound(path.display().to_string()));
    }

    let conn = open_duckdb_locked(path, &QueryOptions::default())?;
    let mut stmt = conn
        .prepare("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'")?;

    let mut tables = Vec::new();
    let mut rows = stmt.query([])?;

    while let Some(row) = rows.next()? {
        let name: String = row.get(0)?;
        tables.push(name);
    }

    Ok(tables)
}

/// Get schema information for a DuckDB table.
pub fn duckdb_table_schema(
    db_path: impl AsRef<Path>,
    table_name: &str,
) -> Result<Vec<(String, String)>, DatabaseError> {
    let path = db_path.as_ref();
    let path_str = path.to_string_lossy();

    if path_str != ":memory:" && !path.exists() {
        return Err(DatabaseError::FileNotFound(path.display().to_string()));
    }

    let conn = open_duckdb_locked(path, &QueryOptions::default())?;
    let query = "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ? ORDER BY ordinal_position";
    let mut stmt = conn.prepare(query)?;

    let mut schema = Vec::new();
    let mut rows = stmt.query([table_name])?;

    while let Some(row) = rows.next()? {
        let col_name: String = row.get(0)?;
        let col_type: String = row.get(1)?;
        schema.push((col_name, col_type));
    }

    Ok(schema)
}

/// Execute a DuckDB query directly on a Parquet/CSV file without loading into database.
/// DuckDB can query files directly, which is very powerful for analytics.
pub fn query_file_with_duckdb(
    file_path: impl AsRef<Path>,
    query: &str,
) -> Result<QueryResult, DatabaseError> {
    query_file_with_duckdb_with(file_path.as_ref(), query, &QueryOptions::default())
        .map(|o| o.result)
}

/// [`query_file_with_duckdb`] with the bounds of [`query_duckdb_with`]. The
/// query can read `file` (as `{file}`) and no other path.
pub fn query_file_with_duckdb_with(
    file: &Path,
    query: &str,
    opts: &QueryOptions,
) -> Result<QueryOutcome, DatabaseError> {
    checkpoint()?;
    // Replace the placeholder with the file path as a quoted literal (`'` doubled).
    // The path cannot open a second statement from inside the literal; the lexer
    // checks the substituted text.
    let literal = quoted_file_literal(file)?;
    let full_query = query.replace("{file}", &literal);
    let body = single_statement_body(&full_query)?;
    validate_duckdb_query(body)?;

    if !file.exists() {
        return Err(DatabaseError::FileNotFound(file.display().to_string()));
    }

    // In-memory DuckDB that can read exactly this file (allowed_paths), with
    // external access otherwise disabled and the configuration locked.
    let conn = open_file_query_connection(file, opts)?;
    run_duckdb_query(&conn, body, opts)
}

/// The SQL actually sent to DuckDB. duckdb-rs materialises a whole result
/// before the first row can be read, so stopping the row loop early bounds
/// nothing inside the engine: with a row budget a `SELECT`/`WITH` statement is
/// wrapped in `LIMIT max_rows + 1` (the extra row tells whether more existed).
/// Other statements (`EXPLAIN`, `PRAGMA`, `DESCRIBE`, `SHOW`) cannot be
/// wrapped and are sent unchanged. The newline before `)` keeps a trailing
/// line comment in `body` harmless.
fn bounded_duckdb_sql<'a>(body: &'a str, opts: &QueryOptions) -> Cow<'a, str> {
    match opts.max_rows {
        Some(n) if matches!(first_keyword(body).as_str(), "select" | "with") => {
            Cow::Owned(format!(
                "SELECT * FROM (\n{body}\n) AS p2a_bounded LIMIT {}",
                n.saturating_add(1)
            ))
        }
        _ => Cow::Borrowed(body),
    }
}

/// Run one (already validated) statement on a locked DuckDB connection and
/// collect its rows within the bounds of `opts`.
fn run_duckdb_query(
    conn: &duckdb::Connection,
    body: &str,
    opts: &QueryOptions,
) -> Result<QueryOutcome, DatabaseError> {
    let sql = bounded_duckdb_sql(body, opts);
    let watch = QueryWatch::arm(conn, opts)?;
    let reason = || watch.as_ref().and_then(QueryWatch::reason);
    let stopped = |e: duckdb::Error| stopped_by(reason(), e.into());

    let mut stmt = conn.prepare(&sql).map_err(stopped)?;

    // Execute query first - DuckDB requires this before accessing column info.
    // This materialises the whole result; the watchdog interrupts it.
    let mut rows = stmt.query([]).map_err(stopped)?;

    // Get column information after execution
    let column_count = rows.as_ref().map(|r| r.column_count()).unwrap_or(0);
    let column_names: Vec<String> = rows
        .as_ref()
        .map(|r| r.column_names().iter().map(|s| s.to_string()).collect())
        .unwrap_or_default();

    let mut collector = RowCollector::new(column_count, opts);
    let mut row_values = Vec::with_capacity(column_count);
    while let Some(row) = rows.next().map_err(stopped)? {
        if !collector.wants_row() {
            break;
        }
        if collector.rows % CHECK_EVERY_ROWS == CHECK_EVERY_ROWS - 1 {
            checkpoint()?;
            if let Some(why) = reason() {
                return Err(DatabaseError::Interrupted(why.to_string()));
            }
        }
        let mut row_bytes = 0u64;
        for i in 0..column_count {
            let (value, bytes) = duckdb_owned_value(row.get_ref(i).map_err(stopped)?);
            row_bytes = row_bytes.saturating_add(bytes);
            row_values.push(value);
        }
        if !collector.push(&mut row_values, row_bytes) {
            break;
        }
    }
    drop(rows);
    drop(stmt);
    drop(watch);

    finish_outcome(collector, column_names, duckdb_column_to_series)
}

/// One DuckDB value read *without owning* the container types the converter
/// discards, plus its estimated byte size: booleans and numbers 8 bytes, text
/// and binary values their length.
///
/// [`duckdb_column_to_series`] keeps only Null, the signed integers, the two
/// floats, text and blobs; everything else (LIST, STRUCT, MAP, ARRAY, UNION,
/// ENUM, DECIMAL, TIMESTAMP, the huge and unsigned integers, …) it renders as
/// an empty string. Reading those through [`duckdb::types::ValueRef`] and
/// replacing them with an empty [`Value::Text`] here, instead of
/// `row.get`-ing an owned [`Value`], means a LIST of millions of elements is
/// never allocated in Rust (outside DuckDB's `memory_limit`, in a conversion
/// the watchdog cannot interrupt) only to be thrown away. The output column is
/// identical. Text is decoded losslessly (`from_utf8_lossy`) so a non-UTF-8
/// byte sequence cannot panic.
fn duckdb_owned_value(value: duckdb::types::ValueRef<'_>) -> (duckdb::types::Value, u64) {
    use duckdb::types::{Value, ValueRef};
    match value {
        ValueRef::Null => (Value::Null, 0),
        ValueRef::Boolean(b) => (Value::Boolean(b), 8),
        ValueRef::TinyInt(i) => (Value::TinyInt(i), 8),
        ValueRef::SmallInt(i) => (Value::SmallInt(i), 8),
        ValueRef::Int(i) => (Value::Int(i), 8),
        ValueRef::BigInt(i) => (Value::BigInt(i), 8),
        ValueRef::Float(f) => (Value::Float(f), 8),
        ValueRef::Double(f) => (Value::Double(f), 8),
        ValueRef::Text(b) => {
            let s = String::from_utf8_lossy(b).into_owned();
            let n = s.len() as u64;
            (Value::Text(s), n)
        }
        ValueRef::Blob(b) => (Value::Blob(b.to_vec()), b.len() as u64),
        // Every other variant renders as "" in the converter; do not
        // materialise it, and count it as zero bytes.
        _ => (Value::Text(String::new()), 0),
    }
}

/// One DuckDB result column as a Series, typed by its first non-null value.
/// Booleans, the four signed integer types and the two float types keep their
/// type; everything else becomes text (blobs lossily, other types as "").
fn duckdb_column_to_series(name: &str, values: Vec<duckdb::types::Value>) -> Series {
    use duckdb::types::Value;
    #[derive(Clone, Copy)]
    enum Kind {
        Bool,
        Int,
        Float,
        Text,
    }
    let kind = values
        .iter()
        .find_map(|v| match v {
            Value::Null => None,
            Value::Boolean(_) => Some(Kind::Bool),
            Value::TinyInt(_) | Value::SmallInt(_) | Value::Int(_) | Value::BigInt(_) => {
                Some(Kind::Int)
            }
            Value::Float(_) | Value::Double(_) => Some(Kind::Float),
            _ => Some(Kind::Text),
        })
        .unwrap_or(Kind::Text);

    match kind {
        Kind::Bool => {
            let values: Vec<Option<bool>> = values
                .into_iter()
                .map(|v| match v {
                    Value::Boolean(b) => Some(b),
                    _ => None,
                })
                .collect();
            Series::new(name.into(), values)
        }
        Kind::Int => {
            let values: Vec<Option<i64>> = values
                .into_iter()
                .map(|v| match v {
                    Value::TinyInt(i) => Some(i64::from(i)),
                    Value::SmallInt(i) => Some(i64::from(i)),
                    Value::Int(i) => Some(i64::from(i)),
                    Value::BigInt(i) => Some(i),
                    _ => None,
                })
                .collect();
            Series::new(name.into(), values)
        }
        Kind::Float => {
            let values: Vec<Option<f64>> = values
                .into_iter()
                .map(|v| match v {
                    Value::Float(f) => Some(f64::from(f)),
                    Value::Double(f) => Some(f),
                    Value::TinyInt(i) => Some(f64::from(i)),
                    Value::SmallInt(i) => Some(f64::from(i)),
                    Value::Int(i) => Some(f64::from(i)),
                    Value::BigInt(i) => Some(i as f64),
                    _ => None,
                })
                .collect();
            Series::new(name.into(), values)
        }
        Kind::Text => {
            // Default to string for text, blob, and unknown types
            let values: Vec<Option<String>> = values
                .into_iter()
                .map(|v| match v {
                    Value::Text(s) => Some(s),
                    Value::Blob(b) => Some(String::from_utf8_lossy(&b).into_owned()),
                    Value::Null => None,
                    Value::Boolean(b) => Some(b.to_string()),
                    Value::TinyInt(i) => Some(i.to_string()),
                    Value::SmallInt(i) => Some(i.to_string()),
                    Value::Int(i) => Some(i.to_string()),
                    Value::BigInt(i) => Some(i.to_string()),
                    Value::Float(f) => Some(f.to_string()),
                    Value::Double(f) => Some(f.to_string()),
                    _ => Some(String::new()),
                })
                .collect();
            Series::new(name.into(), values)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{self, Budget, CancelToken};
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    fn write_file(dir: &Path, name: &str, body: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, body).unwrap();
        p
    }

    fn dir_names(dir: &Path) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        v.sort();
        v
    }

    fn i64_col(df: &DataFrame, name: &str) -> Vec<Option<i64>> {
        df.column(name)
            .unwrap()
            .as_materialized_series()
            .i64()
            .unwrap()
            .into_iter()
            .collect()
    }

    /// `t(x INTEGER)` holding 0..=999.
    fn sqlite_fixture(dir: &Path) -> PathBuf {
        let db = dir.join("t.sqlite");
        rusqlite::Connection::open(&db)
            .unwrap()
            .execute_batch(
                "CREATE TABLE t(x INTEGER); \
                 WITH RECURSIVE c(n) AS (SELECT 0 UNION ALL SELECT n + 1 FROM c WHERE n < 999) \
                 INSERT INTO t SELECT n FROM c;",
            )
            .unwrap();
        db
    }

    // ------------------------------------------------------------------
    // Single statement (LLM-1)
    // ------------------------------------------------------------------

    #[test]
    fn single_statement_accepts_one_statement() {
        for q in [
            "SELECT 1",
            "SELECT 1;",
            "SELECT 1; -- done",
            "SELECT 1;;",
            "SELECT 1 /* c */ ;  \n",
            r"SELECT 'a;b'",
            r"SELECT 'it''s; fine'",
            r"SELECT E'it\'s; fine'",
            "SELECT $$a;b$$",
            "SELECT $tag$a;$$b$tag$",
            r#"SELECT "a;b" FROM t"#,
            r#"SELECT "a""b;" FROM t"#,
            "SELECT /* ; */ 1",
            "SELECT /* outer /* inner ; */ still ; */ 2",
            "SELECT 1 -- ; trailing",
            // DuckDB ends a line comment at CR as well as LF
            "SELECT 1; -- done\r  \r\n",
            "SELECT 1 -- c\r",
            "SELECT 1; -- c\r/* still \n a comment */ \r",
            "SELECT $1",
            "WITH x AS (SELECT 1) SELECT * FROM x",
        ] {
            assert!(
                ensure_single_statement(q).is_ok(),
                "expected {q:?} to be accepted"
            );
        }
        assert_eq!(
            single_statement_body("SELECT 1; -- done").unwrap(),
            "SELECT 1"
        );
        assert_eq!(
            single_statement_body("SELECT 'a;b';").unwrap(),
            "SELECT 'a;b'"
        );
        assert_eq!(
            single_statement_body("SELECT 1 -- c").unwrap(),
            "SELECT 1 -- c"
        );
    }

    #[test]
    fn single_statement_rejects_a_second_statement() {
        for q in [
            "SELECT 1; SELECT 2",
            "SELECT 1;\nATTACH 'x.duckdb' AS e",
            "SELECT 1 -- c\n; SELECT 2",
            // a line comment ends at a lone CR, before and after the terminator
            "SELECT 1 -- c\r; SELECT 2",
            "SELECT 1; -- c\rSELECT 2",
            // E-string: DuckDB sees two statements
            r"SELECT E'\''; ATTACH 'x' AS e; --'",
            // nested comment: DuckDB sees two statements
            r"SELECT 1 /* /* */ ' */ ; ATTACH 'x' AS e; -- '",
            "SELECT $$ x $$; SELECT 2",
            // `$` inside an identifier is not a quote
            "SELECT a$b$ FROM t; SELECT 2",
        ] {
            let err = ensure_single_statement(q).unwrap_err();
            assert!(matches!(err, DatabaseError::Forbidden(_)), "{q:?}: {err}");
            assert!(
                err.to_string().contains("only one SQL statement"),
                "{q:?}: {err}"
            );
        }
    }

    #[test]
    fn single_statement_rejects_unterminated_empty_and_nul() {
        for (q, needle) in [
            ("", "empty"),
            ("   ", "empty"),
            (";", "empty"),
            ("-- only a comment", "empty"),
            ("/* c */ ;", "empty"),
            ("SELECT 'abc", "unterminated"),
            ("SELECT /* x", "unterminated"),
            ("SELECT $$x", "unterminated"),
            (r#"SELECT "abc"#, "unterminated"),
            ("SELECT 1\0", "NUL"),
            // DuckDB reads `1E'..'` / `1.E'..'` as a number plus an escape
            // string; the lexer fails closed rather than mis-read the quote.
            (r"SELECT 1E'\''; ATTACH 'x' AS e", "escape string"),
            (r"SELECT 1.E'\''", "escape string"),
        ] {
            let err = ensure_single_statement(q).unwrap_err();
            assert!(
                matches!(err, DatabaseError::Forbidden(_)) && err.to_string().contains(needle),
                "{q:?}: {err}"
            );
        }
    }

    #[test]
    fn query_duckdb_refuses_multi_statement_and_creates_no_file() {
        let outside = tempfile::tempdir().unwrap();
        let target = outside.path().join("evil.duckdb");
        let q = format!(
            "SELECT 1; ATTACH '{}' AS e; CREATE TABLE e.s AS SELECT 'x' AS v",
            target.display()
        );
        let err = query_duckdb(":memory:", &q).unwrap_err();
        assert!(matches!(err, DatabaseError::Forbidden(_)), "{err}");
        assert!(!target.exists(), "a second statement reached the engine");
    }

    #[test]
    fn query_sqlite_refuses_trailing_statement() {
        let dir = tempfile::tempdir().unwrap();
        let db = sqlite_fixture(dir.path());
        assert!(matches!(
            query_sqlite(&db, "SELECT x FROM t; SELECT 2").unwrap_err(),
            DatabaseError::Forbidden(_)
        ));
    }

    #[test]
    fn query_file_refuses_trailing_statement() {
        let dir = tempfile::tempdir().unwrap();
        let csv = write_file(dir.path(), "in.csv", "a,b\n1,2\n3,4\n");
        let out = dir.path().join("copied.csv");
        let q = format!(
            "SELECT * FROM {{file}}; COPY (SELECT 1) TO '{}'",
            out.display()
        );
        assert!(matches!(
            query_file_with_duckdb(&csv, &q).unwrap_err(),
            DatabaseError::Forbidden(_)
        ));
        assert!(!out.exists());
    }

    #[test]
    fn query_file_refuses_a_statement_after_a_cr_terminated_comment() {
        // DuckDB ends the comment at the CR; the allowed file itself stays
        // writable by the engine (Fact 5), so the lexer must refuse this.
        let dir = tempfile::tempdir().unwrap();
        let original = b"a,b\n1,2\n3,4\n";
        let csv = dir.path().join("in.csv");
        std::fs::write(&csv, original).unwrap();
        let q =
            "SELECT 1 -- c\r; COPY (SELECT 9 AS a) TO {file} (HEADER false, USE_TMP_FILE false)";
        let err = query_file_with_duckdb_with(&csv, q, &rows(10)).unwrap_err();
        assert!(matches!(err, DatabaseError::Forbidden(_)), "{err}");
        assert_eq!(std::fs::read(&csv).unwrap(), original);
    }

    #[test]
    fn query_file_keeps_the_allowed_file_read_only() {
        // The allowed file stays writable by the engine (Fact 5). The keyword
        // check must see through nested block comments as DuckDB does, and
        // EXPLAIN ANALYZE executes the statement it explains. Both with and
        // without a row budget (only SELECT/WITH are wrapped in a LIMIT).
        let write = "COPY (SELECT 9 AS a) TO {file} (HEADER false, USE_TMP_FILE false)";
        let queries = [
            format!("/* /* */ SELECT */ {write}"),
            format!("/* /* */ SHOW */ {write}"),
            format!("EXPLAIN ANALYZE {write}"),
            format!("EXPLAIN ANALYSE {write}"),
            format!("EXPLAIN (ANALYZE) {write}"),
            format!("EXPLAIN /* x */ ANALYZE {write}"),
        ];
        let dir = tempfile::tempdir().unwrap();
        let original = b"a,b\n1,2\n3,4\n";
        let csv = dir.path().join("in.csv");
        for opts in [QueryOptions::default(), rows(10)] {
            for q in &queries {
                std::fs::write(&csv, original).unwrap();
                let err = query_file_with_duckdb_with(&csv, q, &opts).unwrap_err();
                assert!(matches!(err, DatabaseError::Forbidden(_)), "{q}: {err}");
                assert_eq!(std::fs::read(&csv).unwrap(), original, "{q}");
            }
        }
        // Plain EXPLAIN and a quoted `copy` are still read-only queries.
        assert!(query_file_with_duckdb(&csv, "EXPLAIN SELECT * FROM {file}").is_ok());
        let r = query_file_with_duckdb(&csv, "SELECT 'copy' AS \"copy\" FROM {file}").unwrap();
        assert_eq!(r.rows, 2);
    }

    #[test]
    fn duckdb_refuses_explain_analyze_and_copy() {
        for q in [
            "EXPLAIN ANALYZE SELECT 1",
            "explain /* a */ analyse SELECT 1",
            "EXPLAIN (FORMAT json) SELECT 1",
            "SELECT 1 AS copy",
        ] {
            let err = query_duckdb(":memory:", q).unwrap_err();
            assert!(matches!(err, DatabaseError::Forbidden(_)), "{q}: {err}");
        }
        assert!(query_duckdb(":memory:", "EXPLAIN SELECT 1").is_ok());
    }

    // ------------------------------------------------------------------
    // Locked DuckDB connections (LLM-1)
    // ------------------------------------------------------------------

    #[test]
    fn locked_duckdb_connection_settings() {
        let s = |c: &duckdb::Connection, name: &str| -> String {
            c.query_row(
                &format!("SELECT current_setting('{name}')::VARCHAR"),
                [],
                |r| r.get(0),
            )
            .unwrap()
        };
        let conn = open_duckdb_locked(Path::new(":memory:"), &QueryOptions::default()).unwrap();
        assert_eq!(s(&conn, "enable_external_access"), "false");
        assert_eq!(s(&conn, "lock_configuration"), "true");
        assert_eq!(s(&conn, "autoload_known_extensions"), "false");
        assert_eq!(s(&conn, "autoinstall_known_extensions"), "false");
        assert_eq!(s(&conn, "allow_community_extensions"), "false");
        let opts = QueryOptions {
            memory_limit_mb: Some(64),
            threads: Some(1),
            ..Default::default()
        };
        let conn = open_duckdb_locked(Path::new(":memory:"), &opts).unwrap();
        assert_eq!(s(&conn, "memory_limit"), "64.0 MiB");
        assert_eq!(s(&conn, "threads"), "1");
        assert_eq!(s(&conn, "temp_directory"), "");
    }

    #[test]
    fn locked_duckdb_refuses_file_access_extensions_and_reconfiguration() {
        let outside = tempfile::tempdir().unwrap();
        let secret = write_file(outside.path(), "secret.csv", "k,v\nsecret,42\n");
        let o = outside.path().display();
        let conn = open_duckdb_locked(Path::new(":memory:"), &QueryOptions::default()).unwrap();
        for sql in [
            format!("ATTACH '{o}/evil.duckdb' AS e"),
            format!("COPY (SELECT 1) TO '{o}/copied.csv'"),
            format!("SELECT * FROM read_csv_auto('{}')", secret.display()),
            format!("SELECT * FROM '{}'", secret.display()),
        ] {
            let err = conn.execute_batch(&sql).unwrap_err().to_string();
            assert!(err.contains("disabled by configuration"), "{sql}: {err}");
        }
        assert!(
            conn.execute_batch("SET enable_external_access = true")
                .unwrap_err()
                .to_string()
                .contains("locked")
        );
        assert!(conn.execute_batch("INSTALL httpfs").is_err());
        assert!(conn.execute_batch("LOAD httpfs").is_err());
        assert_eq!(dir_names(outside.path()), vec!["secret.csv"]);
    }

    #[test]
    fn query_duckdb_cannot_read_files_by_literal_path() {
        // single statement, passes the keyword check
        let outside = tempfile::tempdir().unwrap();
        let secret = write_file(outside.path(), "secret.csv", "k,v\nsecret,42\n");
        for q in [
            format!("SELECT * FROM '{}'", secret.display()),
            format!("SELECT * FROM read_csv_auto('{}')", secret.display()),
        ] {
            let err = query_duckdb(":memory:", &q).unwrap_err();
            assert!(
                err.to_string().contains("disabled by configuration"),
                "{q}: {err}"
            );
        }
    }

    #[test]
    fn locked_duckdb_file_database_is_read_only_and_cannot_attach() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("main.duckdb");
        duckdb::Connection::open(&db)
            .unwrap()
            .execute_batch("CREATE TABLE t AS SELECT range AS x FROM range(3)")
            .unwrap();
        let other = dir.path().join("other.duckdb");
        duckdb::Connection::open(&other)
            .unwrap()
            .execute_batch("CREATE TABLE s AS SELECT 'v' AS secret")
            .unwrap();
        let r = query_duckdb(&db, "SELECT count(*) AS n FROM t").unwrap();
        assert_eq!(i64_col(&r.dataframe, "n"), vec![Some(3)]);
        let conn = open_duckdb_locked(&db, &QueryOptions::default()).unwrap();
        assert!(conn.execute_batch("CREATE TABLE z AS SELECT 1").is_err());
        let err = conn
            .execute_batch(&format!("ATTACH '{}' AS e (READ_ONLY)", other.display()))
            .unwrap_err()
            .to_string();
        assert!(err.contains("disabled by configuration"), "{err}");
    }

    #[test]
    fn file_query_reads_only_the_named_file() {
        let dir = tempfile::tempdir().unwrap();
        let csv = write_file(dir.path(), "in.csv", "a,b\n1,2\n3,4\n");
        let other = write_file(dir.path(), "other.csv", "k,v\nsecret,42\n");
        let r = query_file_with_duckdb(&csv, "SELECT count(*) AS n FROM {file}").unwrap();
        assert_eq!(i64_col(&r.dataframe, "n"), vec![Some(2)]);
        let r = query_file_with_duckdb(&csv, "SELECT * FROM read_csv_auto({file}, header = true)")
            .unwrap();
        assert_eq!(r.rows, 2); // reader options still work
        let err = query_file_with_duckdb(
            &csv,
            &format!("SELECT * FROM read_csv_auto('{}')", other.display()),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("disabled by configuration"),
            "{err}"
        );
        let conn = open_file_query_connection(&csv, &QueryOptions::default()).unwrap();
        let d = dir.path().display();
        for sql in [
            format!("ATTACH '{d}/evil.duckdb' AS e"),
            format!("COPY (SELECT 1) TO '{d}/copied.csv'"),
            format!("SELECT * FROM glob('{d}/*')"),
        ] {
            let err = conn.execute_batch(&sql).unwrap_err().to_string();
            assert!(err.contains("disabled by configuration"), "{sql}: {err}");
        }
        assert!(
            conn.execute_batch("SET enable_external_access = true")
                .unwrap_err()
                .to_string()
                .contains("locked")
        );
        assert_eq!(dir_names(dir.path()), vec!["in.csv", "other.csv"]);
    }

    #[test]
    fn file_query_reads_parquet_with_extension_autoload_disabled() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("in.parquet");
        let mut df =
            df!("x" => [1i64, 2, 3, 4, 5], "y" => [0.52f64, 1.38, 2.61, 3.47, 4.55]).unwrap();
        ParquetWriter::new(std::fs::File::create(&path).unwrap())
            .finish(&mut df)
            .unwrap();
        let r = query_file_with_duckdb(&path, "SELECT count(*) AS n FROM {file}").unwrap();
        assert_eq!(i64_col(&r.dataframe, "n"), vec![Some(5)]);
    }

    // ------------------------------------------------------------------
    // Locked SQLite connections (LLM-1)
    // ------------------------------------------------------------------

    #[test]
    fn locked_sqlite_refuses_attach_and_non_schema_pragmas() {
        let dir = tempfile::tempdir().unwrap();
        let db = sqlite_fixture(dir.path());
        let other = dir.path().join("other.sqlite");
        rusqlite::Connection::open(&other)
            .unwrap()
            .execute_batch("CREATE TABLE s(secret TEXT); INSERT INTO s VALUES ('v');")
            .unwrap();
        let SqliteConnection { conn, .. } =
            open_sqlite_locked(&db, &QueryOptions::default()).unwrap();
        let err = conn
            .execute_batch(&format!("ATTACH '{}' AS o", other.display()))
            .unwrap_err()
            .to_string();
        assert!(err.contains("too many attached databases"), "{err}");
        // A schema PRAGMA is admitted; every other PRAGMA, even read-only
        // `query_only`, is refused by the authorizer.
        assert!(conn.execute_batch("PRAGMA table_info('t')").is_ok());
        assert!(conn.execute_batch("PRAGMA query_only").is_err());
    }

    #[test]
    fn sqlite_relative_paths_are_never_parsed_as_file_uris() {
        // The bundled SQLite is built with -DSQLITE_USE_URI, so any name that
        // begins with `file:` is URI-parsed. A relative path is prefixed with
        // `.` so it cannot; an absolute path is left unchanged.
        let uri_like = Path::new("file:t.sqlite?mode=memory");
        let opened = sqlite_open_path(uri_like);
        assert!(!opened.to_string_lossy().starts_with("file:"), "{opened:?}");
        assert!(opened.starts_with("."));
        let abs = Path::new("/tmp/file:t.sqlite?mode=memory");
        assert_eq!(sqlite_open_path(abs), Cow::Borrowed(abs));
        // Behaviourally: a relative `file:…:memory:` name must open as a plain
        // (nonexistent) file, not as an in-memory URI database.
        let err = open_sqlite_locked(
            Path::new("file::memory:?cache=shared"),
            &QueryOptions::default(),
        );
        assert!(
            err.is_err(),
            "a relative file: URI opened as a URI database"
        );
    }

    #[test]
    fn sqlite_refuses_side_effecting_pragmas_but_keeps_schema_pragmas() {
        let dir = tempfile::tempdir().unwrap();
        let db = sqlite_fixture(dir.path());
        // Vectors that do no harm even if the authorizer regresses: a
        // `hard_heap_limit` of 0 and an empty `temp_store_directory` would both
        // change process-global SQLite state for every later query.
        for q in [
            "PRAGMA hard_heap_limit = 0",
            "PRAGMA temp_store_directory = ''",
        ] {
            let err = query_sqlite(&db, q).unwrap_err();
            assert!(
                matches!(err, DatabaseError::Forbidden(_))
                    && err.to_string().contains("not allowed"),
                "{q}: {err}"
            );
        }
        // Read-only schema PRAGMAs and their table-valued forms still work.
        assert!(query_sqlite(&db, "PRAGMA table_info('t')").is_ok());
        assert!(query_sqlite(&db, "SELECT * FROM pragma_table_info('t')").is_ok());
        assert!(sqlite_table_schema(&db, "t").is_ok());
    }

    #[test]
    fn query_sqlite_refuses_statements_that_write() {
        let dir = tempfile::tempdir().unwrap();
        let db = sqlite_fixture(dir.path());
        // starts with WITH, so the keyword allowlist admits it
        let err = query_sqlite(
            &db,
            "WITH v AS (SELECT 1 AS x) INSERT INTO t SELECT x FROM v",
        )
        .unwrap_err();
        assert!(
            matches!(err, DatabaseError::Forbidden(_)) && err.to_string().contains("read-only"),
            "{err}"
        );
    }

    // ------------------------------------------------------------------
    // Bounded results (LLM-2)
    // ------------------------------------------------------------------

    fn rows(max: usize) -> QueryOptions {
        QueryOptions {
            max_rows: Some(max),
            ..Default::default()
        }
    }

    #[test]
    fn duckdb_row_limit_truncates_in_query_order() {
        let o = query_duckdb_with(
            ":memory:",
            "SELECT range AS x FROM range(1000) ORDER BY x DESC",
            &rows(100),
        )
        .unwrap();
        assert!(o.truncated);
        assert_eq!(o.rows_returned, 100);
        assert_eq!(o.result.dataframe.height(), 100);
        let x = i64_col(&o.result.dataframe, "x");
        assert_eq!((x[0], x[99]), (Some(999), Some(900)));
    }

    #[test]
    fn duckdb_row_limit_stops_the_engine_not_just_the_copy() {
        // Without the engine-side LIMIT this materialises 10^13 rows inside
        // DuckDB. The backstop so a broken wrapper fails cleanly instead of
        // exhausting memory is the RSS guard (memory_limit_mb below; it
        // interrupts at 128 MiB of growth on Linux) plus the timeout.
        let opts = QueryOptions {
            max_rows: Some(5),
            memory_limit_mb: Some(64),
            timeout: Some(Duration::from_secs(30)),
            ..Default::default()
        };
        let o = query_duckdb_with(
            ":memory:",
            "SELECT range AS x FROM range(10000000000000)",
            &opts,
        )
        .unwrap();
        assert!(o.truncated);
        assert_eq!(
            i64_col(&o.result.dataframe, "x"),
            vec![Some(0), Some(1), Some(2), Some(3), Some(4)]
        );
    }

    #[test]
    fn duckdb_exact_fit_is_not_truncated() {
        let o = query_duckdb_with(":memory:", "SELECT range AS x FROM range(100);", &rows(100))
            .unwrap();
        assert!(!o.truncated);
        assert_eq!(o.rows_returned, 100);
    }

    #[test]
    fn duckdb_byte_limit_truncates() {
        // estimate: 1000 bytes per VARCHAR value of length 1000 -> exactly 50 rows fit in 50_000
        let opts = QueryOptions {
            max_bytes: Some(50_000),
            ..Default::default()
        };
        let o = query_duckdb_with(
            ":memory:",
            "SELECT repeat('x', 1000) AS s FROM range(1000)",
            &opts,
        )
        .unwrap();
        assert!(o.truncated);
        assert_eq!(o.rows_returned, 50);
    }

    #[test]
    fn duckdb_container_values_are_not_materialised() {
        // A LIST of a million elements must not be allocated in Rust just to be
        // rendered as "". The byte budget would admit it (0 bytes), so a row is
        // returned; its value is the empty string, and no OOM occurs.
        let o = query_duckdb_with(
            ":memory:",
            "SELECT list(range) AS l FROM range(1000000)",
            &QueryOptions {
                max_bytes: Some(1000),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(o.rows_returned, 1);
        let l = o
            .result
            .dataframe
            .column("l")
            .unwrap()
            .as_materialized_series()
            .str()
            .unwrap()
            .get(0);
        assert_eq!(l, Some(""));
    }

    #[test]
    fn duckdb_unbounded_default_returns_everything() {
        let r = query_duckdb(":memory:", "SELECT range AS x FROM range(5000)").unwrap();
        assert_eq!(r.rows, 5000);
    }

    #[test]
    fn duckdb_bounded_metadata_statements_still_work() {
        for q in [
            "EXPLAIN SELECT 1",
            "PRAGMA database_list",
            "DESCRIBE SELECT 1 AS a",
            "SHOW TABLES",
            "SELECT 1 -- trailing comment",
            "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 50) SELECT n FROM r",
        ] {
            assert!(query_duckdb_with(":memory:", q, &rows(10)).is_ok(), "{q}");
        }
    }

    #[test]
    fn file_query_row_limit() {
        let dir = tempfile::tempdir().unwrap();
        let body: String = std::iter::once("a\n".to_string())
            .chain((0..1000).map(|i| format!("{i}\n")))
            .collect();
        let csv = write_file(dir.path(), "big.csv", &body);
        let o = query_file_with_duckdb_with(&csv, "SELECT * FROM {file}", &rows(10)).unwrap();
        assert!(o.truncated);
        assert_eq!(o.rows_returned, 10);
    }

    #[test]
    fn sqlite_row_and_byte_limits() {
        let dir = tempfile::tempdir().unwrap();
        let db = sqlite_fixture(dir.path());
        let o = query_sqlite_with(&db, "SELECT x FROM t ORDER BY x", &rows(10)).unwrap();
        assert!(o.truncated);
        assert_eq!(i64_col(&o.result.dataframe, "x")[9], Some(9));
        let o = query_sqlite_with(&db, "SELECT x FROM t", &rows(1000)).unwrap();
        assert!(!o.truncated);
        assert_eq!(o.rows_returned, 1000);
        // 8 bytes per INTEGER -> 100 rows fit in 800 bytes
        let o = query_sqlite_with(
            &db,
            "SELECT x FROM t",
            &QueryOptions {
                max_bytes: Some(800),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(o.truncated);
        assert_eq!(o.rows_returned, 100);
        let err = query_sqlite_with(
            &db,
            "SELECT randomblob(20000) AS b",
            &QueryOptions {
                max_bytes: Some(10_000),
                ..Default::default()
            },
        )
        .unwrap_err();
        // Forbidden, naming the server's result size limit, not the bare
        // "string or blob too big".
        assert!(
            matches!(err, DatabaseError::Forbidden(_)),
            "expected Forbidden: {err}"
        );
        let msg = err.to_string();
        assert!(msg.contains("too big") && msg.contains("10000"), "{err}");
    }

    // ------------------------------------------------------------------
    // Deadlines, cancellation and memory growth (LLM-2)
    // ------------------------------------------------------------------
    // None asserts wall-clock time except the release-only bound. Each query is
    // finite even if the watchdog is broken, so a regression fails instead of hanging.

    fn heavy_duckdb() -> &'static str {
        // ~7 s single-threaded in release, much longer in debug
        "SELECT count(*) AS n FROM range(20000) a, range(20000) b WHERE a.range + b.range = 3"
    }

    #[test]
    fn duckdb_query_stops_at_timeout() {
        let opts = QueryOptions {
            timeout: Some(Duration::from_millis(200)),
            threads: Some(1),
            ..Default::default()
        };
        let t0 = Instant::now();
        let err = query_duckdb_with(":memory:", heavy_duckdb(), &opts).unwrap_err();
        assert!(
            matches!(err, DatabaseError::Interrupted(_)) && err.to_string().contains("time limit"),
            "{err}"
        );
        if !cfg!(debug_assertions) {
            assert!(t0.elapsed() < Duration::from_secs(5));
        }
    }

    #[test]
    fn duckdb_query_stops_when_budget_cancelled() {
        let token = CancelToken::new();
        let canceller = {
            let t = token.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(200));
                t.cancel();
            })
        };
        let opts = QueryOptions {
            threads: Some(1),
            ..Default::default()
        };
        let err = resource::with_budget(Budget::new(None, token), || {
            query_duckdb_with(":memory:", heavy_duckdb(), &opts)
        })
        .unwrap_err();
        canceller.join().unwrap();
        assert!(matches!(err, DatabaseError::Interrupted(_)), "{err}");
    }

    #[test]
    fn duckdb_query_stops_at_budget_deadline() {
        let budget = Budget::new(
            Some(Instant::now() + Duration::from_millis(200)),
            CancelToken::new(),
        );
        let opts = QueryOptions {
            threads: Some(1),
            ..Default::default()
        };
        let err = resource::with_budget(budget, || {
            query_duckdb_with(":memory:", heavy_duckdb(), &opts)
        })
        .unwrap_err();
        assert!(matches!(err, DatabaseError::Interrupted(_)), "{err}");
    }

    #[test]
    fn expired_budget_refuses_before_opening() {
        let budget = Budget::new(
            Some(Instant::now() - Duration::from_millis(1)),
            CancelToken::new(),
        );
        let err = resource::with_budget(budget, || {
            query_duckdb_with(":memory:", "SELECT 1", &QueryOptions::default())
        })
        .unwrap_err();
        assert!(matches!(err, DatabaseError::Interrupted(_)), "{err}");
    }

    #[test]
    fn sqlite_query_stops_at_timeout_and_on_cancel() {
        let dir = tempfile::tempdir().unwrap();
        let db = sqlite_fixture(dir.path());
        let q = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 500000000) \
                 SELECT count(*) FROM r";
        let err = query_sqlite_with(
            &db,
            q,
            &QueryOptions {
                timeout: Some(Duration::from_millis(200)),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(
            matches!(err, DatabaseError::Interrupted(_)) && err.to_string().contains("time limit"),
            "{err}"
        );
        let token = CancelToken::new();
        let t = token.clone();
        let h = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(200));
            t.cancel();
        });
        let err = resource::with_budget(Budget::new(None, token), || {
            query_sqlite_with(&db, q, &QueryOptions::default())
        })
        .unwrap_err();
        h.join().unwrap();
        assert!(matches!(err, DatabaseError::Interrupted(_)), "{err}");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn duckdb_rss_guard_stops_untracked_result_growth() {
        // No max_rows: DuckDB materialises ~200 MB outside memory_limit; the guard
        // interrupts once RSS has grown by more than 2 x 32 MiB. Parallel tests can
        // only make it fire earlier.
        let opts = QueryOptions {
            memory_limit_mb: Some(32),
            threads: Some(1),
            ..Default::default()
        };
        let err = query_duckdb_with(
            ":memory:",
            "SELECT repeat(range::VARCHAR, 64) AS s FROM range(500000)",
            &opts,
        )
        .unwrap_err();
        assert!(
            matches!(err, DatabaseError::Interrupted(_)) && err.to_string().contains("memory"),
            "{err}"
        );
    }

    #[test]
    fn test_validate_read_only_allows_select() {
        assert!(validate_read_only_query("SELECT * FROM t").is_ok());
        assert!(validate_read_only_query("  select 1").is_ok());
        assert!(validate_read_only_query("WITH x AS (SELECT 1) SELECT * FROM x").is_ok());
        assert!(validate_read_only_query("EXPLAIN SELECT 1").is_ok());
        assert!(validate_read_only_query("PRAGMA table_info('t')").is_ok());
        assert!(validate_read_only_query("DESCRIBE t").is_ok());
        assert!(validate_read_only_query("SHOW TABLES").is_ok());
    }

    #[test]
    fn test_validate_read_only_rejects_mutations() {
        for q in [
            "INSERT INTO t VALUES (1)",
            "UPDATE t SET x=1",
            "DELETE FROM t",
            "DROP TABLE t",
            "ALTER TABLE t ADD COLUMN x INT",
            "CREATE TABLE t (x INT)",
            "ATTACH 'evil.db' AS e",
            "COPY t TO 'out.csv'",
            "INSTALL 'httpfs'",
            "LOAD 'httpfs'",
            "",
            "   ",
        ] {
            assert!(
                validate_read_only_query(q).is_err(),
                "expected {:?} to be rejected",
                q
            );
        }
    }

    #[test]
    fn test_validate_read_only_strips_comments() {
        assert!(validate_read_only_query("-- a comment\nSELECT 1").is_ok());
        assert!(validate_read_only_query("/* block */ SELECT 1").is_ok());
        assert!(validate_read_only_query("-- hide\nDROP TABLE t").is_err());
        // a line comment ends at CR too, as in DuckDB's scanner
        assert!(validate_read_only_query("-- a comment\rSELECT 1").is_ok());
        assert!(validate_read_only_query("-- hide\rDROP TABLE t").is_err());
        assert_eq!(first_keyword("-- c\rshow tables"), "show");
        // block comments nest, as in DuckDB's scanner
        assert_eq!(first_keyword("/* a /* b */ c */ show tables"), "show");
        assert!(validate_read_only_query("/* /* */ SELECT */ DROP TABLE t").is_err());
        assert!(matches!(
            bounded_duckdb_sql("/* /* */ x */ SELECT 1", &rows(10)),
            Cow::Owned(_)
        ));
    }

    #[test]
    fn test_sqlite_query() {
        // A fresh temporary directory: a fixed path would race between parallel runs.
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_p2a.db");

        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute("CREATE TABLE test (id INTEGER, name TEXT, value REAL)", [])
            .unwrap();
        conn.execute("INSERT INTO test VALUES (1, 'Alice', 10.5)", [])
            .unwrap();
        conn.execute("INSERT INTO test VALUES (2, 'Bob', 20.3)", [])
            .unwrap();
        drop(conn);

        let result = query_sqlite(&db_path, "SELECT * FROM test").unwrap();
        assert_eq!(result.rows, 2);
        assert_eq!(result.columns.len(), 3);
    }

    #[test]
    fn test_duckdb_query() {
        // Use in-memory DuckDB
        let conn = duckdb::Connection::open_in_memory().unwrap();
        conn.execute(
            "CREATE TABLE test (id INTEGER, name VARCHAR, value DOUBLE)",
            [],
        )
        .unwrap();
        conn.execute("INSERT INTO test VALUES (1, 'Alice', 10.5)", [])
            .unwrap();
        conn.execute("INSERT INTO test VALUES (2, 'Bob', 20.3)", [])
            .unwrap();

        let outcome =
            run_duckdb_query(&conn, "SELECT * FROM test", &QueryOptions::default()).unwrap();
        assert!(!outcome.truncated);
        assert_eq!(outcome.result.rows, 2);
        assert_eq!(outcome.result.columns.len(), 3);
    }
}
