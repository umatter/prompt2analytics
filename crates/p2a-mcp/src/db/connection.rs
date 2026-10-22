//! SurrealDB connection management
//!
//! Provides connection to embedded SurrealDB with RocksDB or in-memory backend.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

use surrealdb::Surreal;
use surrealdb::engine::local::{Db, Mem, RocksDb};
use thiserror::Error;

/// Database connection errors
#[derive(Debug, Error)]
pub enum DbError {
    #[error("Connection error: {0}")]
    Connection(String),

    #[error("Query error: {0}")]
    Query(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    /// A client-posted message, title or settings string longer than
    /// `message_bytes`. The text names the size and the limit.
    #[error("{0}")]
    TooLarge(String),

    /// A count or byte budget of the conversation store is used up
    /// (`messages_per_conversation`, `conversations_per_session`,
    /// `session_db_bytes`). The text names the budget.
    #[error("{0}")]
    QuotaExceeded(String),

    /// The process's resident memory is above the soft limit.
    #[error("the server is low on memory; try again shortly")]
    MemoryPressure,
}

impl From<surrealdb::Error> for DbError {
    fn from(e: surrealdb::Error) -> Self {
        DbError::Query(e.to_string())
    }
}

/// Bounds on what the conversation store keeps. The server resolves them
/// from its resource profile (`tools::limits`) and hands them over with
/// [`DbConnection::with_limits`]; `None` means unlimited. The default is the
/// `Off` profile: 2 MiB per message and nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PersistenceLimits {
    /// Bytes of one client-posted message, title or settings string; longer
    /// server-produced text is truncated to it.
    pub message_bytes: Option<usize>,
    /// Conversations one session may hold.
    pub conversations_per_session: Option<usize>,
    /// Messages one conversation may hold.
    pub messages_per_conversation: Option<usize>,
    /// Bytes one session's conversation history may hold (titles, message
    /// contents, tool-call arguments, results and errors).
    pub session_db_bytes: Option<u64>,
    /// Writes are refused while the process's resident memory is above this.
    pub rss_soft_limit_bytes: Option<u64>,
}

impl Default for PersistenceLimits {
    fn default() -> Self {
        Self {
            message_bytes: Some(2_097_152),
            conversations_per_session: None,
            messages_per_conversation: None,
            session_db_bytes: None,
            rss_soft_limit_bytes: None,
        }
    }
}

/// Database connection wrapper
///
/// Supports both RocksDB (persistent) and in-memory backends.
pub struct DbConnection {
    db: Arc<Surreal<Db>>,
    limits: PersistenceLimits,
    /// Bytes each session's conversation history stores, as counted by the
    /// `session_db_bytes` budget. An entry is seeded from the store on first
    /// use (a read of every stored message of the session), lowered by what
    /// a deleted or cleared conversation held, and dropped when the
    /// session's rows are deleted. Only touched when the budget is set;
    /// shared by clones.
    session_bytes: Arc<Mutex<HashMap<String, u64>>>,
}

/// Appended to text the store shortens.
const TRUNCATION_SUFFIX: &str = "\n[truncated]";

/// `s` cut at a char boundary so that, with [`TRUNCATION_SUFFIX`] appended,
/// it is at most `limit` bytes; `s` itself when it already fits. A `limit`
/// smaller than the suffix gives the empty string.
pub(crate) fn truncate_to(s: &str, limit: usize) -> String {
    if s.len() <= limit {
        return s.to_string();
    }
    let Some(room) = limit.checked_sub(TRUNCATION_SUFFIX.len()) else {
        return String::new();
    };
    let mut cut = room;
    while !s.is_char_boundary(cut) {
        cut -= 1;
    }
    let mut out = String::with_capacity(cut + TRUNCATION_SUFFIX.len());
    out.push_str(&s[..cut]);
    out.push_str(TRUNCATION_SUFFIX);
    out
}

impl DbConnection {
    /// Connect to a RocksDB database at the given path
    pub async fn connect_rocksdb(path: impl AsRef<Path>) -> Result<Self, DbError> {
        let path = path.as_ref();
        tracing::info!("Connecting to SurrealDB RocksDB at {:?}", path);

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                DbError::Connection(format!("Failed to create database directory: {}", e))
            })?;
        }

        let db = Surreal::new::<RocksDb>(path)
            .await
            .map_err(|e| DbError::Connection(e.to_string()))?;

        Self::setup(db).await
    }

    /// Connect to an in-memory database (for testing)
    pub async fn connect_memory() -> Result<Self, DbError> {
        tracing::info!("Connecting to SurrealDB in-memory");

        let db = Surreal::new::<Mem>(())
            .await
            .map_err(|e| DbError::Connection(e.to_string()))?;

        Self::setup(db).await
    }

    /// Common setup for both connection types
    async fn setup(db: Surreal<Db>) -> Result<Self, DbError> {
        // Use namespace and database
        db.use_ns("p2a").use_db("analytics").await?;

        // Run schema setup
        Self::setup_schema(&db).await?;

        tracing::info!("SurrealDB connected and schema initialized");

        Ok(Self {
            db: Arc::new(db),
            limits: PersistenceLimits::default(),
            session_bytes: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// This connection with `limits` applied to every later write.
    pub fn with_limits(mut self, limits: PersistenceLimits) -> Self {
        self.limits = limits;
        self
    }

    /// The limits this connection applies.
    pub fn limits(&self) -> &PersistenceLimits {
        &self.limits
    }

    // ==================== Limits ====================

    /// `MemoryPressure` while the process's resident memory is above
    /// `rss_soft_limit_bytes` (never when the limit is unset or the platform
    /// does not report the process memory).
    pub(crate) fn check_memory(&self) -> Result<(), DbError> {
        let Some(limit) = self.limits.rss_soft_limit_bytes else {
            return Ok(());
        };
        match p2a_core::get_process_memory() {
            Some(mem) if u64::try_from(mem.rss_bytes).unwrap_or(u64::MAX) > limit => {
                Err(DbError::MemoryPressure)
            }
            _ => Ok(()),
        }
    }

    /// `TooLarge` when a client-posted `what` ("message", "title", a
    /// settings field) of `len` bytes is longer than `message_bytes`; `per`
    /// names the unit the limit applies to ("message", "title", "setting").
    pub(crate) fn check_message_bytes(
        &self,
        what: &str,
        per: &str,
        len: usize,
    ) -> Result<(), DbError> {
        match self.limits.message_bytes {
            Some(limit) if len > limit => Err(DbError::TooLarge(format!(
                "{what} is {len} bytes; this server accepts at most {limit} bytes per {per}"
            ))),
            _ => Ok(()),
        }
    }

    /// Server-produced text cut to `message_bytes` (see [`truncate_to`]).
    pub(crate) fn truncate_server_text(&self, text: &str) -> String {
        match self.limits.message_bytes {
            Some(limit) => truncate_to(text, limit),
            None => text.to_string(),
        }
    }

    fn session_bytes_map(&self) -> MutexGuard<'_, HashMap<String, u64>> {
        self.session_bytes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Run `f` on the session's byte counter under its lock, seeding the
    /// counter from the store first when it is missing. No lock is held
    /// across the query; a counter another write seeded in the meantime wins.
    async fn with_session_bytes<R>(
        &self,
        session_id: &str,
        f: impl FnOnce(&mut u64) -> R,
    ) -> Result<R, DbError> {
        let mut seed = None;
        loop {
            {
                let mut map = self.session_bytes_map();
                if let Some(stored) = seed {
                    map.entry(session_id.to_string()).or_insert(stored);
                }
                if let Some(used) = map.get_mut(session_id) {
                    return Ok(f(used));
                }
            }
            seed = Some(self.session_stored_bytes(session_id).await?);
        }
    }

    /// Count `n` more bytes against the session's `session_db_bytes`
    /// budget, or refuse with `QuotaExceeded` when they would pass it. The
    /// check and the increment happen under one lock. A no-op without the
    /// budget.
    pub(crate) async fn reserve_session_bytes(
        &self,
        session_id: &str,
        n: u64,
    ) -> Result<(), DbError> {
        let Some(limit) = self.limits.session_db_bytes else {
            return Ok(());
        };
        self.with_session_bytes(session_id, |used| {
            if used.saturating_add(n) > limit {
                return Err(DbError::QuotaExceeded(format!(
                    "this session already stores {used} bytes of conversation history and this \
                     server keeps at most {limit} bytes per session; delete a conversation first"
                )));
            }
            *used += n;
            Ok(())
        })
        .await?
    }

    /// Server-produced `text` cut to the bytes the session has left under
    /// `session_db_bytes` (possibly to the empty string) and counted; never
    /// refused. `text` unchanged without the budget.
    pub(crate) async fn fit_session_bytes(
        &self,
        session_id: &str,
        text: String,
    ) -> Result<String, DbError> {
        let Some(limit) = self.limits.session_db_bytes else {
            return Ok(text);
        };
        self.with_session_bytes(session_id, |used| {
            let room = usize::try_from(limit.saturating_sub(*used)).unwrap_or(usize::MAX);
            let stored = if text.len() > room {
                truncate_to(&text, room)
            } else {
                text
            };
            *used = used.saturating_add(stored.len() as u64);
            stored
        })
        .await
    }

    /// Drop the session's byte counter (its rows were deleted); a later
    /// write re-seeds it from the store.
    pub(crate) fn forget_session_bytes(&self, session_id: &str) {
        if self.limits.session_db_bytes.is_some() {
            self.session_bytes_map().remove(session_id);
        }
    }

    /// Whether the session's byte counter is seeded.
    pub(crate) fn has_session_bytes(&self, session_id: &str) -> bool {
        self.limits.session_db_bytes.is_some() && self.session_bytes_map().contains_key(session_id)
    }

    /// Subtract `n` deleted bytes from the session's counter, if it is
    /// seeded (an unseeded counter reads the store when it is seeded).
    pub(crate) fn release_session_bytes(&self, session_id: &str, n: u64) {
        if self.limits.session_db_bytes.is_some()
            && let Some(used) = self.session_bytes_map().get_mut(session_id)
        {
            *used = used.saturating_sub(n);
        }
    }

    /// Whether the `session_db_bytes` budget is set.
    pub(crate) fn tracks_session_bytes(&self) -> bool {
        self.limits.session_db_bytes.is_some()
    }

    /// The session's byte counter as it stands (`None` when not seeded).
    #[cfg(test)]
    pub(crate) fn session_bytes_entry(&self, session_id: &str) -> Option<u64> {
        self.session_bytes_map().get(session_id).copied()
    }

    /// Initialize database schema
    async fn setup_schema(db: &Surreal<Db>) -> Result<(), DbError> {
        // Define schema using SurrealQL
        // Using DEFINE TABLE IF NOT EXISTS pattern for idempotency
        let schema = r#"
            -- Conversations table
            DEFINE TABLE IF NOT EXISTS conversations SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS id ON conversations TYPE string;
            DEFINE FIELD IF NOT EXISTS session_id ON conversations TYPE string;
            DEFINE FIELD IF NOT EXISTS title ON conversations TYPE string;
            DEFINE FIELD IF NOT EXISTS created_at ON conversations TYPE datetime;
            DEFINE FIELD IF NOT EXISTS updated_at ON conversations TYPE datetime;
            DEFINE FIELD IF NOT EXISTS is_archived ON conversations TYPE bool DEFAULT false;
            DEFINE FIELD IF NOT EXISTS message_count ON conversations TYPE int DEFAULT 0;
            DEFINE FIELD IF NOT EXISTS last_message_preview ON conversations TYPE option<string>;
            DEFINE INDEX IF NOT EXISTS idx_conversations_session ON conversations COLUMNS session_id;
            DEFINE INDEX IF NOT EXISTS idx_conversations_id ON conversations COLUMNS id UNIQUE;

            -- Messages table
            DEFINE TABLE IF NOT EXISTS messages SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS id ON messages TYPE string;
            DEFINE FIELD IF NOT EXISTS conversation_id ON messages TYPE string;
            DEFINE FIELD IF NOT EXISTS role ON messages TYPE string;
            DEFINE FIELD IF NOT EXISTS content ON messages TYPE string;
            DEFINE FIELD IF NOT EXISTS created_at ON messages TYPE datetime;
            DEFINE FIELD IF NOT EXISTS token_count ON messages TYPE option<int>;
            DEFINE FIELD IF NOT EXISTS model ON messages TYPE option<string>;
            DEFINE FIELD IF NOT EXISTS finish_reason ON messages TYPE option<string>;
            DEFINE INDEX IF NOT EXISTS idx_messages_conversation ON messages COLUMNS conversation_id;
            DEFINE INDEX IF NOT EXISTS idx_messages_id ON messages COLUMNS id UNIQUE;

            -- Tool calls table
            DEFINE TABLE IF NOT EXISTS tool_calls SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS id ON tool_calls TYPE string;
            DEFINE FIELD IF NOT EXISTS message_id ON tool_calls TYPE string;
            DEFINE FIELD IF NOT EXISTS conversation_id ON tool_calls TYPE string;
            DEFINE FIELD IF NOT EXISTS tool_name ON tool_calls TYPE string;
            DEFINE FIELD IF NOT EXISTS arguments ON tool_calls TYPE string;
            DEFINE FIELD IF NOT EXISTS result ON tool_calls TYPE option<string>;
            DEFINE FIELD IF NOT EXISTS error ON tool_calls TYPE option<string>;
            DEFINE FIELD IF NOT EXISTS status ON tool_calls TYPE string DEFAULT 'pending';
            DEFINE FIELD IF NOT EXISTS started_at ON tool_calls TYPE datetime;
            DEFINE FIELD IF NOT EXISTS completed_at ON tool_calls TYPE option<datetime>;
            DEFINE FIELD IF NOT EXISTS duration_ms ON tool_calls TYPE option<int>;
            DEFINE INDEX IF NOT EXISTS idx_tool_calls_message ON tool_calls COLUMNS message_id;
            DEFINE INDEX IF NOT EXISTS idx_tool_calls_id ON tool_calls COLUMNS id UNIQUE;

            -- Sessions table
            DEFINE TABLE IF NOT EXISTS sessions SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS id ON sessions TYPE string;
            DEFINE FIELD IF NOT EXISTS user_id ON sessions TYPE option<string>;
            DEFINE FIELD IF NOT EXISTS created_at ON sessions TYPE datetime;
            DEFINE FIELD IF NOT EXISTS last_accessed ON sessions TYPE datetime;
            DEFINE FIELD IF NOT EXISTS global_seed ON sessions TYPE option<int>;
            DEFINE FIELD IF NOT EXISTS active_conversation_id ON sessions TYPE option<string>;
            DEFINE FIELD IF NOT EXISTS dataset_names ON sessions TYPE array DEFAULT [];
            DEFINE INDEX IF NOT EXISTS idx_sessions_id ON sessions COLUMNS id UNIQUE;
            DEFINE INDEX IF NOT EXISTS idx_sessions_user ON sessions COLUMNS user_id;

            -- Settings table
            DEFINE TABLE IF NOT EXISTS settings SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS id ON settings TYPE string;
            DEFINE FIELD IF NOT EXISTS session_id ON settings TYPE string;
            DEFINE FIELD IF NOT EXISTS provider ON settings TYPE string DEFAULT 'ollama';
            DEFINE FIELD IF NOT EXISTS model ON settings TYPE string DEFAULT 'llama3.1';
            DEFINE FIELD IF NOT EXISTS base_url ON settings TYPE option<string>;
            DEFINE FIELD IF NOT EXISTS temperature ON settings TYPE float DEFAULT 0.7;
            DEFINE FIELD IF NOT EXISTS max_tokens ON settings TYPE int DEFAULT 4096;
            DEFINE FIELD IF NOT EXISTS system_prompt ON settings TYPE option<string>;
            DEFINE FIELD IF NOT EXISTS updated_at ON settings TYPE datetime;
            DEFINE INDEX IF NOT EXISTS idx_settings_session ON settings COLUMNS session_id UNIQUE;

            -- Dataset metadata table
            DEFINE TABLE IF NOT EXISTS dataset_meta SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS id ON dataset_meta TYPE string;
            DEFINE FIELD IF NOT EXISTS session_id ON dataset_meta TYPE string;
            DEFINE FIELD IF NOT EXISTS name ON dataset_meta TYPE string;
            DEFINE FIELD IF NOT EXISTS source_path ON dataset_meta TYPE option<string>;
            DEFINE FIELD IF NOT EXISTS source_type ON dataset_meta TYPE string;
            DEFINE FIELD IF NOT EXISTS row_count ON dataset_meta TYPE int;
            DEFINE FIELD IF NOT EXISTS column_count ON dataset_meta TYPE int;
            DEFINE FIELD IF NOT EXISTS column_names ON dataset_meta TYPE array<string>;
            DEFINE FIELD IF NOT EXISTS loaded_at ON dataset_meta TYPE datetime;
            DEFINE FIELD IF NOT EXISTS file_size_bytes ON dataset_meta TYPE option<int>;
            DEFINE INDEX IF NOT EXISTS idx_dataset_meta_session ON dataset_meta COLUMNS session_id;
            DEFINE INDEX IF NOT EXISTS idx_dataset_meta_id ON dataset_meta COLUMNS id UNIQUE;
        "#;

        db.query(schema).await?;
        Ok(())
    }

    /// Get a reference to the database for queries
    pub fn db(&self) -> &Surreal<Db> {
        &self.db
    }

    /// Check if the database is healthy
    pub async fn health_check(&self) -> Result<bool, DbError> {
        // Try a simple query
        self.db.query("SELECT * FROM sessions LIMIT 1").await?;
        Ok(true)
    }

    /// Get database statistics
    pub async fn stats(&self) -> Result<DbStats, DbError> {
        let mut result = self
            .db
            .query(
                r#"
                RETURN {
                    sessions: (SELECT count() FROM sessions GROUP ALL)[0].count ?? 0,
                    conversations: (SELECT count() FROM conversations GROUP ALL)[0].count ?? 0,
                    messages: (SELECT count() FROM messages GROUP ALL)[0].count ?? 0
                }
            "#,
            )
            .await?;

        #[derive(serde::Deserialize, Default)]
        struct StatsResult {
            sessions: Option<i64>,
            conversations: Option<i64>,
            messages: Option<i64>,
        }

        let stats: Option<StatsResult> = result.take(0)?;
        let stats = stats.unwrap_or_default();

        Ok(DbStats {
            session_count: stats.sessions.unwrap_or(0) as u32,
            conversation_count: stats.conversations.unwrap_or(0) as u32,
            message_count: stats.messages.unwrap_or(0) as u32,
        })
    }
}

impl Clone for DbConnection {
    fn clone(&self) -> Self {
        Self {
            db: Arc::clone(&self.db),
            limits: self.limits,
            session_bytes: Arc::clone(&self.session_bytes),
        }
    }
}

/// Database statistics
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct DbStats {
    pub session_count: u32,
    pub conversation_count: u32,
    pub message_count: u32,
}
