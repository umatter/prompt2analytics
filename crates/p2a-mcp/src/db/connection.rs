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
    /// Bytes one session's conversation history and dataset metadata may
    /// hold (titles, message contents, tool-call arguments, results and
    /// errors, and each dataset metadata row's texts and fixed fields).
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
    /// Bytes each session's conversation history and dataset metadata
    /// rows store, as counted by the `session_db_bytes` budget. An entry is
    /// seeded from the store on first use (a read of every stored message
    /// and dataset metadata row of the session), lowered by the bytes a
    /// conversation delete or clear or a dataset metadata delete actually
    /// removed, and dropped when the session's rows are deleted. Only
    /// touched when the budget is set; shared by clones.
    session_bytes: Arc<Mutex<HashMap<String, u64>>>,
    /// One lock per conversation whose rows a call is changing (and per
    /// session creating a conversation or changing its dataset metadata),
    /// with the number of calls holding or waiting for it (see
    /// [`StoreTurn`]); shared by clones.
    store_turns: StoreTurns,
    /// Conversations deleted recently, for
    /// [`delete_recent_orphans`](Self::delete_recent_orphans); shared by
    /// clones.
    deleted_conversations: Arc<Mutex<DeletedConversations>>,
}

/// The conversations deleted since the last orphan sweep (`current`) and
/// during the one before (`previous`).
#[derive(Debug, Default)]
struct DeletedConversations {
    current: std::collections::HashSet<String>,
    previous: std::collections::HashSet<String>,
    /// More were deleted than are remembered; the next sweep is a full one.
    overflowed: bool,
}

/// Recently deleted conversations remembered for the orphan sweep; past
/// this the next sweep is a full one
/// ([`delete_orphans`](DbConnection::delete_orphans)).
pub(crate) const MAX_REMEMBERED_DELETES: usize = 100_000;

type StoreTurns = Arc<Mutex<HashMap<String, (Arc<tokio::sync::Mutex<()>>, usize)>>>;

/// One call's turn at changing a conversation's rows (a conversation delete or
/// clear, a message add or content update, a title or archive change, or a
/// tool-call create, update or status change:
/// [`conversation_turn`](DbConnection::conversation_turn)), or at creating a
/// conversation in a session ([`session_turn`](DbConnection::session_turn)),
/// or at changing a session's dataset metadata rows
/// ([`dataset_meta_turn`](DbConnection::dataset_meta_turn)); the next call's
/// turn starts when it is dropped. It is `Send` and `'static`, so the task
/// of [`run_to_end`] can own it.
///
/// Two deletes of one conversation that run at the same time conflict in the
/// store: the in-memory engine hands both the same rows (neither fails), so
/// each would release their bytes from the `session_db_bytes` counter, and
/// RocksDB fails one of them with a write conflict. Taking turns makes the
/// later call find nothing instead. Every write of an existing conversation's
/// row takes its turn, so the counter update after a message add or content
/// update never loses a write conflict to another write of that row in this
/// process. Message adds and content updates and tool-call creates, updates
/// and status changes take it too, so none commits on a row while a delete or
/// clear deletes it (on RocksDB that would fail the delete or clear whole with
/// a write conflict). A message add holds the turn from its
/// `messages_per_conversation` check until the message is written, and a
/// conversation creation holds its session's turn from the
/// `conversations_per_session` check until the row is written, so concurrent
/// writes cannot pass a cap together. Both checks count the rows in the store
/// under the turn (`count_messages`, `count_conversations`): the
/// conversation's stored `message_count` is a display counter, recounted under
/// the turn after each message add or content update (and written by an add
/// refused at the cap when it differs from the rows), and never what a cap is
/// checked against. A message add, a content update, a title change and a
/// tool-call write (create, update, result or error) count their bytes against
/// `session_db_bytes` and write them in the conversation's turn, so a delete
/// or clear never runs between the count and the write. The byte counter is
/// per process, and so are the turns. An entry leaves the map with its last
/// holder or waiter, also when a waiting call is cancelled. A turn ends and
/// leaves its entry in one step under the map's lock, so a call that holds a
/// turn never finds the call before it still counted in the map.
///
/// Every call that takes a turn waits for it in its caller and then moves it
/// into a task of its own ([`run_to_end`]), which holds it until the store
/// has finished the writes made in it. A call dropped while it waits for its
/// turn (an HTTP client that disconnects before the turn comes) leaves the
/// queue having done nothing, so the calls queued for a turn are never more
/// than the callers still waiting. The store finishes a query it was sent
/// even when the caller is dropped, so a turn that ended with the caller
/// would let the next call count the rows before the dropped call's row is
/// committed, or run its transaction alongside the dropped call's write (on
/// RocksDB, one of the two then fails on a write conflict).
pub(crate) struct StoreTurn {
    turns: StoreTurns,
    key: String,
    guard: Option<tokio::sync::OwnedMutexGuard<()>>,
}

impl Drop for StoreTurn {
    fn drop(&mut self) {
        let mut turns = self
            .turns
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // Let the next call in before giving up the entry, under the map's
        // lock: the next call's turn may start at once on another thread,
        // and any look at the map after that waits for this lock, so it
        // finds this turn gone. Releasing the guard only wakes the next
        // waiter and takes no lock on the map, so nothing waits on itself.
        self.guard.take();
        if let Some((_, users)) = turns.get_mut(&self.key) {
            *users = users.saturating_sub(1);
            if *users == 0 {
                turns.remove(&self.key);
            }
        }
    }
}

/// Move `turn`, the [`StoreTurn`] the caller waited for, and `work`, the
/// writes the call makes in it, into a task of their own, and wait for it.
/// The task starts when this is called, not when the returned future is
/// first polled, so the turn passes from the caller to the task with no
/// await in between, and it ends once `work` has finished.
///
/// Waiting for the turn in the caller means that a call dropped while it
/// waits does nothing. The store runs a query it was sent to its end even
/// when the caller is dropped (its engine runs each query in a task).
/// Running the call in a task as well keeps the turn held until the store
/// has finished, and makes sure the steps after its writes still run (the
/// conversation's counter update after a message add, the release of a
/// delete's bytes from the `session_db_bytes` counter), even when the
/// caller (an HTTP client that disconnects) is dropped. A task that panicked
/// or was cancelled (the runtime shutting down) is a `Query` error.
pub(super) fn run_to_end<T: Send + 'static>(
    turn: StoreTurn,
    work: impl Future<Output = Result<T, DbError>> + Send + 'static,
) -> impl Future<Output = Result<T, DbError>> + Send {
    let task = tokio::spawn(async move {
        let done = work.await;
        drop(turn);
        done
    });
    async move {
        task.await
            .map_err(|e| DbError::Query(format!("conversation store task failed: {e}")))?
    }
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
    let Some(cut) = truncation_cut(s, limit) else {
        return String::new();
    };
    let mut out = String::with_capacity(cut + TRUNCATION_SUFFIX.len());
    out.push_str(&s[..cut]);
    out.push_str(TRUNCATION_SUFFIX);
    out
}

/// `text` cut as [`truncate_to`] cuts it, in its own buffer: a text that
/// fits is left as it is, and one that does not is shortened and given the
/// suffix without a copy.
pub(crate) fn truncate_in_place(text: &mut String, limit: usize) {
    if text.len() <= limit {
        return;
    }
    match truncation_cut(text, limit) {
        Some(cut) => {
            text.truncate(cut);
            text.push_str(TRUNCATION_SUFFIX);
        }
        None => text.clear(),
    }
}

/// How many bytes of `s`, which is longer than `limit`, are kept in front
/// of [`TRUNCATION_SUFFIX`]: the most that fit beside it, at a char
/// boundary. `None` when `limit` has no room for the suffix (nothing is
/// kept).
fn truncation_cut(s: &str, limit: usize) -> Option<usize> {
    let mut cut = limit.checked_sub(TRUNCATION_SUFFIX.len())?;
    while !s.is_char_boundary(cut) {
        cut -= 1;
    }
    Some(cut)
}

/// The error a transaction's failed statement reported, from the
/// [`take_errors`](surrealdb::Response::take_errors) of its response; `None`
/// when no statement failed. When a statement inside `BEGIN`/`COMMIT`
/// fails, the store replaces the results of the other statements, before
/// and after the failing one, with `QueryNotExecuted` and keeps the cause
/// only at the failing statement, so this is the lowest-indexed error that
/// is not `QueryNotExecuted` (the lowest-indexed error when all are).
/// [`Response::check`](surrealdb::Response::check) returns the first error
/// by index instead, which hides the cause behind a `QueryNotExecuted`
/// whenever a statement other than the first fails. A failed commit gives
/// every statement a `QueryNotExecutedDetail` that carries the cause, and a
/// transaction cancelled or timed out between statements gives the
/// statements that ran `QueryCancelled` and the rest `QueryCancelled` or
/// `QueryTimedout`, so a timeout can come back as `QueryCancelled` (this
/// server sets no query or transaction timeout); this returns the first of
/// them as is. `take_errors` returns a `HashMap`, so the index, not the
/// order of iteration, decides.
pub(super) fn failed_statement(
    errors: HashMap<usize, surrealdb::Error>,
) -> Option<surrealdb::Error> {
    errors
        .into_iter()
        .min_by_key(|(index, error)| {
            let not_executed = matches!(
                error,
                surrealdb::Error::Db(surrealdb::error::Db::QueryNotExecuted)
            );
            (not_executed, *index)
        })
        .map(|(_, error)| error)
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
            store_turns: Arc::new(Mutex::new(HashMap::new())),
            deleted_conversations: Arc::new(Mutex::new(DeletedConversations::default())),
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

    /// Server-produced `text` cut to `message_bytes` in its own buffer (see
    /// [`truncate_in_place`]); a text that fits is left as it is.
    pub(crate) fn truncate_server_text_in_place(&self, text: &mut String) {
        if let Some(limit) = self.limits.message_bytes {
            truncate_in_place(text, limit);
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
                    "this session already stores {used} bytes of conversation history and \
                     dataset metadata, and this server keeps at most {limit} bytes per session; \
                     delete a conversation first"
                )));
            }
            *used += n;
            Ok(())
        })
        .await?
    }

    /// Fit server-produced `texts`, of a write that replaces stored texts of
    /// `replaced` bytes, in place: cut in order (see [`truncate_in_place`]) to the
    /// bytes the session has left under `session_db_bytes` plus `replaced`
    /// (each possibly to the empty string), and count them; never refused.
    /// Without the budget `texts` are left as they are. The counter holds
    /// the replaced texts too until the caller, once the write has replaced
    /// them, releases their bytes
    /// ([`release_session_bytes`](Self::release_session_bytes)); a write that
    /// fails leaves both counted. Call it, the write and the release in the
    /// conversation's [`StoreTurn`], so that no delete or clear releases the
    /// replaced texts in between.
    pub(crate) async fn fit_session_texts(
        &self,
        session_id: &str,
        texts: &mut [&mut String],
        replaced: u64,
    ) -> Result<(), DbError> {
        let Some(limit) = self.limits.session_db_bytes else {
            return Ok(());
        };
        self.with_session_bytes(session_id, |used| {
            let mut room = limit.saturating_sub(*used).saturating_add(replaced);
            for text in texts.iter_mut() {
                truncate_in_place(text, usize::try_from(room).unwrap_or(usize::MAX));
                room = room.saturating_sub(text.len() as u64);
                *used = used.saturating_add(text.len() as u64);
            }
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

    /// Drop the byte counter of every session not in `with_row` (an orphan
    /// sweep deleted rows of sessions without a session row); a later write
    /// re-seeds it from the store.
    pub(crate) fn forget_session_bytes_except(&self, with_row: &std::collections::HashSet<String>) {
        if self.limits.session_db_bytes.is_some() {
            self.session_bytes_map()
                .retain(|session_id, _| with_row.contains(session_id));
        }
    }

    /// Whether the session's byte counter is seeded.
    pub(crate) fn has_session_bytes(&self, session_id: &str) -> bool {
        self.limits.session_db_bytes.is_some() && self.session_bytes_map().contains_key(session_id)
    }

    /// Subtract `n` deleted bytes from the session's counter, if it is
    /// seeded (an unseeded counter reads the store when it is seeded).
    /// `n` must be what the caller's own statements removed during the
    /// conversation's [`StoreTurn`], so that no row is released twice.
    pub(crate) fn release_session_bytes(&self, session_id: &str, n: u64) {
        if self.limits.session_db_bytes.is_some()
            && let Some(used) = self.session_bytes_map().get_mut(session_id)
        {
            *used = used.saturating_sub(n);
        }
    }

    /// Wait for this call's turn at changing conversation
    /// `conversation_id`'s rows (see [`StoreTurn`]).
    pub(crate) async fn conversation_turn(&self, conversation_id: &str) -> StoreTurn {
        self.turn(conversation_id.to_string()).await
    }

    /// Wait for this call's turn at creating a conversation in session
    /// `session_id` (see [`StoreTurn`]). Conversation ids are UUIDs, so
    /// this key is never a conversation's.
    pub(crate) async fn session_turn(&self, session_id: &str) -> StoreTurn {
        self.turn(format!("session:{session_id}")).await
    }

    /// Wait for this call's turn at changing session `session_id`'s dataset
    /// metadata rows (see [`StoreTurn`]): a save reads the row it replaces
    /// and counts the bytes it adds, a delete releases the bytes it
    /// removed, and neither may run between another's read and write.
    pub(crate) async fn dataset_meta_turn(&self, session_id: &str) -> StoreTurn {
        self.turn(format!("datasets:{session_id}")).await
    }

    async fn turn(&self, key: String) -> StoreTurn {
        let lock = {
            let mut turns = self
                .store_turns
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let (lock, users) = turns
                .entry(key.clone())
                .or_insert_with(|| (Arc::new(tokio::sync::Mutex::new(())), 0));
            *users += 1;
            Arc::clone(lock)
        };
        // Registered before waiting, so a cancelled wait still leaves.
        let mut turn = StoreTurn {
            turns: Arc::clone(&self.store_turns),
            key,
            guard: None,
        };
        turn.guard = Some(lock.lock_owned().await);
        turn
    }

    fn deleted_conversations_map(&self) -> MutexGuard<'_, DeletedConversations> {
        self.deleted_conversations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Remember that conversation `id` is being deleted, so the next two
    /// orphan sweeps look for rows left behind for it.
    pub(crate) fn note_conversation_deleted(&self, id: &str) {
        let mut deleted = self.deleted_conversations_map();
        deleted.current.insert(id.to_string());
        if deleted.current.len() + deleted.previous.len() > MAX_REMEMBERED_DELETES {
            deleted.current.clear();
            deleted.previous.clear();
            deleted.overflowed = true;
        }
    }

    /// The conversations deleted since the sweep before the last one, and
    /// the start of the next sweep's memory; `None` when more were deleted
    /// than are remembered (the caller sweeps in full).
    pub(crate) fn take_deleted_conversations(&self) -> Option<std::collections::HashSet<String>> {
        let mut deleted = self.deleted_conversations_map();
        let current = std::mem::take(&mut deleted.current);
        let mut ids = std::mem::replace(&mut deleted.previous, current);
        if std::mem::take(&mut deleted.overflowed) {
            return None;
        }
        ids.extend(deleted.previous.iter().cloned());
        Some(ids)
    }

    /// Conversations and sessions with a store turn held or awaited.
    #[cfg(test)]
    pub(crate) fn store_turns_in_use(&self) -> usize {
        self.store_turns
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    /// The calls holding or waiting for the store turn `key` (a conversation
    /// id, `session:<id>` or `datasets:<id>`); 0 when none is.
    #[cfg(test)]
    pub(crate) fn store_turn_users(&self, key: &str) -> usize {
        self.store_turns
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(key)
            .map_or(0, |(_, users)| *users)
    }

    /// The lock on the store turns' map, held until the returned guard is
    /// dropped: meanwhile no turn ends and no call joins one.
    #[cfg(test)]
    pub(crate) fn hold_store_turns(&self) -> impl Sized + '_ {
        self.store_turns
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Whether a call holds the store turn `key` (a conversation id, or
    /// `session:<id>`), as opposed to only being registered for it:
    /// [`turn`](Self::turn) counts a call in `users` before it waits for the
    /// lock.
    #[cfg(test)]
    pub(crate) fn store_turn_held(&self, key: &str) -> bool {
        self.store_turns
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(key)
            .is_some_and(|(lock, _)| lock.try_lock().is_err())
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
            DEFINE INDEX IF NOT EXISTS idx_tool_calls_conversation ON tool_calls COLUMNS conversation_id;
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
            store_turns: Arc::clone(&self.store_turns),
            deleted_conversations: Arc::clone(&self.deleted_conversations),
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
