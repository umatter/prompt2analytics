//! Session persistence operations

use std::collections::HashSet;

use chrono::{DateTime, Utc};
use surrealdb::RecordId;
use surrealdb::sql::Datetime;

use super::connection::{DbConnection, DbError};
use super::models::{DbSession, Settings, key_string};

/// One row of `SELECT count() ... GROUP ALL`.
#[derive(serde::Deserialize)]
struct CountRow {
    count: i64,
}

/// How much of the messages and tool calls an orphan sweep looks at.
#[derive(Debug, Clone, Copy)]
enum OrphanScan {
    /// Every conversation that no longer exists (the gaps of the
    /// conversation indexes between those that do; see [`OwnerGap`]).
    Full,
    /// The rows of recently deleted conversations.
    Recent,
}

/// Orphaned rows the sweep reads and deletes per statement. SurrealDB keeps
/// every row a statement reads until the statement ends, so deleting all
/// the orphans of a gap in one statement would hold all of them at once.
pub(super) const ORPHAN_BATCH: usize = 32;

/// One batch of a gap's orphans: the rows read (at most [`ORPHAN_BATCH`])
/// and, of those, the rows deleted.
#[derive(Debug, Clone, Copy)]
struct Batch {
    read: usize,
    deleted: usize,
}

/// How the batches of one gap ended ([`drain_gap`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GapEnd {
    /// A batch came back short: it held the last rows of the gap that
    /// match.
    Drained,
    /// A full batch of which no row could be deleted; reading again would
    /// read the same rows.
    Stuck,
    /// The batches the gap's count allows were read, and the last was full.
    Bound,
}

/// Delete the orphans of one gap of `table`, a batch at a time
/// (`next_batch` reads up to [`ORPHAN_BATCH`] of them and deletes what it
/// can), until a batch comes back short; the rows deleted, and how it ended.
///
/// The gap held `rows` rows when it was counted, and each batch is deleted
/// before the next is read, so the rows that match only decrease and
/// `rows.div_ceil(ORPHAN_BATCH) + 1` batches drain the gap. Two things keep
/// it from draining: rows that match written into the gap after its count,
/// and a full batch of which nothing could be deleted (conversations that
/// `delete_conversation` cannot find by the key read for them), which every
/// further read would return again. This then stops, at the bound or at
/// once, and logs it; what is left waits for the next full sweep, at the
/// next start (later sweeps follow recent deletes only).
async fn drain_gap<F, Fut>(
    table: &str,
    rows: usize,
    mut next_batch: F,
) -> Result<(usize, GapEnd), DbError>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<Batch, DbError>>,
{
    let mut deleted = 0usize;
    for _ in 0..=rows.div_ceil(ORPHAN_BATCH) {
        let batch = next_batch().await?;
        deleted += batch.deleted;
        if batch.read < ORPHAN_BATCH {
            return Ok((deleted, GapEnd::Drained));
        }
        if batch.deleted == 0 {
            tracing::warn!(
                table,
                "orphan sweep could not delete a whole batch of orphaned rows; \
                 the rest of their gap waits for the next full sweep"
            );
            return Ok((deleted, GapEnd::Stuck));
        }
    }
    tracing::warn!(
        table,
        "orphan sweep stopped at its batch bound; remaining orphans wait for the next full sweep"
    );
    Ok((deleted, GapEnd::Bound))
}

/// A conversation the orphan sweep deletes: its id and its session.
#[derive(serde::Deserialize)]
struct ConversationOwner {
    id: RecordId,
    session_id: String,
}

/// A range of owner values (`session_id` or `conversation_id`): those after
/// one owner and before another, neither included. Owner columns are
/// strings, which their index orders by their bytes, as `str` does, so such
/// a range is a range of the index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OwnerGap<'a> {
    /// The owner the range follows; `None`: it starts at the empty string
    /// (the least string), included, so that even the range of every owner
    /// is a range of the index rather than a table scan.
    after: Option<&'a str>,
    /// The owner the range ends before; `None`: no end.
    before: Option<&'a str>,
}

/// The `$lo` and `$hi` of [`OwnerGap::condition`], bound as query
/// parameters.
#[derive(serde::Serialize)]
struct GapBounds {
    lo: String,
    hi: String,
}

impl OwnerGap<'_> {
    /// `owner` in the range, as a SurrealQL condition on `$lo` and `$hi`
    /// ([`bounds`](Self::bounds)).
    fn condition(&self, owner: &str) -> String {
        let lower = match self.after {
            Some(_) => format!("{owner} > $lo"),
            None => format!("{owner} >= $lo"),
        };
        match self.before {
            Some(_) => format!("{lower} AND {owner} < $hi"),
            None => lower,
        }
    }

    /// The parameters of [`condition`](Self::condition).
    fn bounds(&self) -> GapBounds {
        GapBounds {
            lo: self.after.unwrap_or_default().to_string(),
            hi: self.before.unwrap_or_default().to_string(),
        }
    }
}

/// The owners in `known` that bound the gaps of an owner index, sorted by
/// their bytes (the order of the index).
///
/// An owner with a NUL byte is left out: SurrealDB cannot encode such a
/// string into an index key (as a bound it would fail the query), and for
/// the same reason no row of an indexed owner column holds it, so the gaps
/// on its two sides merge without taking in any row it owns. The server's
/// session and conversation ids are UUIDs.
fn sorted_owners(known: &HashSet<String>) -> Vec<&str> {
    let mut owners: Vec<&str> = known
        .iter()
        .map(String::as_str)
        .filter(|owner| !owner.contains('\0'))
        .collect();
    owners.sort_unstable();
    owners
}

/// The gaps between the sorted `owners`, in order: gap `i` ends before
/// `owners[i]` and gap `i + 1` starts after it (one gap, every owner, when
/// there are none). Every row whose owner is not in `owners` lies in exactly
/// one gap.
fn owner_gaps<'a>(owners: &[&'a str]) -> Vec<OwnerGap<'a>> {
    let mut gaps = Vec::with_capacity(owners.len() + 1);
    let mut after = None;
    for &owner in owners {
        gaps.push(OwnerGap {
            after,
            before: Some(owner),
        });
        after = Some(owner);
    }
    gaps.push(OwnerGap {
        after,
        before: None,
    });
    gaps
}

/// The text of a message row that the `session_db_bytes` budget counts.
/// Other fields of the row are ignored.
#[derive(serde::Deserialize)]
pub(super) struct MessageText {
    content: String,
}

/// The text of a tool-call row that the `session_db_bytes` budget counts.
/// Other fields of the row are ignored.
#[derive(serde::Deserialize)]
pub(super) struct ToolCallText {
    arguments: String,
    result: Option<String>,
    error: Option<String>,
}

/// UTF-8 bytes of the messages' contents.
pub(super) fn message_text_bytes(rows: &[MessageText]) -> u64 {
    rows.iter()
        .map(|m| m.content.len() as u64)
        .fold(0u64, u64::saturating_add)
}

/// UTF-8 bytes of the tool calls' arguments, results and errors.
pub(super) fn tool_call_text_bytes(rows: &[ToolCallText]) -> u64 {
    let len = |s: &str| s.len() as u64;
    rows.iter()
        .map(|t| {
            len(&t.arguments)
                .saturating_add(t.result.as_deref().map_or(0, len))
                .saturating_add(t.error.as_deref().map_or(0, len))
        })
        .fold(0u64, u64::saturating_add)
}

impl DbConnection {
    // ==================== Session Operations ====================

    /// Create or update a session in the database
    pub async fn upsert_session(&self, session: &DbSession) -> Result<DbSession, DbError> {
        let result: Option<DbSession> = self
            .db()
            .upsert(session.id.clone())
            .content(session.clone())
            .await?;

        result.ok_or_else(|| DbError::Query("Failed to upsert session".to_string()))
    }

    /// Get a session by ID
    pub async fn get_session(&self, id: &str) -> Result<Option<DbSession>, DbError> {
        let session: Option<DbSession> = self.db().select(("sessions", id)).await?;
        Ok(session)
    }

    /// Update session's last_accessed timestamp
    pub async fn touch_session(&self, id: &str) -> Result<(), DbError> {
        let id_owned = id.to_string();
        self.db()
            .query("UPDATE sessions SET last_accessed = time::now() WHERE id = $id")
            .bind(("id", RecordId::from(("sessions", id_owned.as_str()))))
            .await?;
        Ok(())
    }

    /// Set a session row's `last_accessed` to `at` if the row exists; never
    /// creates one. Whether the row existed.
    pub async fn refresh_session_last_accessed(
        &self,
        id: &str,
        at: DateTime<Utc>,
    ) -> Result<bool, DbError> {
        let mut result = self
            .db()
            .query("UPDATE $id SET last_accessed = $at RETURN AFTER")
            .bind(("id", RecordId::from(("sessions", id))))
            .bind(("at", Datetime::from(at)))
            .await?;
        let updated: Vec<DbSession> = result.take(0)?;
        Ok(!updated.is_empty())
    }

    /// Set a session row's `last_accessed` to `at`. A missing row is
    /// created.
    pub async fn set_session_last_accessed(
        &self,
        id: &str,
        at: DateTime<Utc>,
    ) -> Result<(), DbError> {
        if !self.refresh_session_last_accessed(id, at).await? {
            let mut row = DbSession::new(id.to_string(), None);
            row.last_accessed = Datetime::from(at);
            self.upsert_session(&row).await?;
        }
        Ok(())
    }

    /// Update session's active conversation
    pub async fn set_active_conversation(
        &self,
        session_id: &str,
        conversation_id: Option<&str>,
    ) -> Result<(), DbError> {
        let session_id_owned = session_id.to_string();
        self.db()
            .query("UPDATE sessions SET active_conversation_id = $conv_id, last_accessed = time::now() WHERE id = $id")
            .bind(("id", RecordId::from(("sessions", session_id_owned.as_str()))))
            .bind(("conv_id", conversation_id.map(|s| s.to_string())))
            .await?;
        Ok(())
    }

    /// Update session's dataset names
    pub async fn update_session_datasets(
        &self,
        session_id: &str,
        dataset_names: Vec<String>,
    ) -> Result<(), DbError> {
        let session_id_owned = session_id.to_string();
        self.db()
            .query("UPDATE sessions SET dataset_names = $names, last_accessed = time::now() WHERE id = $id")
            .bind(("id", RecordId::from(("sessions", session_id_owned.as_str()))))
            .bind(("names", dataset_names))
            .await?;
        Ok(())
    }

    /// Update session's global seed
    pub async fn update_session_seed(
        &self,
        session_id: &str,
        seed: Option<i64>,
    ) -> Result<(), DbError> {
        let session_id_owned = session_id.to_string();
        self.db()
            .query("UPDATE sessions SET global_seed = $seed, last_accessed = time::now() WHERE id = $id")
            .bind(("id", RecordId::from(("sessions", session_id_owned.as_str()))))
            .bind(("seed", seed))
            .await?;
        Ok(())
    }

    /// List all sessions
    pub async fn list_sessions(&self) -> Result<Vec<DbSession>, DbError> {
        let mut result = self
            .db()
            .query("SELECT * FROM sessions ORDER BY last_accessed DESC")
            .await?;

        let sessions: Vec<DbSession> = result.take(0)?;
        Ok(sessions)
    }

    /// Delete a session and all associated data
    pub async fn delete_session(&self, id: &str) -> Result<(), DbError> {
        // The byte counter goes with the session: dropping it first spares
        // each conversation delete below from summing what it releases.
        self.forget_session_bytes(id);
        // Get all conversations for this session
        let conversations = self.list_conversations(id).await?;

        // Delete each conversation (which also deletes messages and tool
        // calls). One that is already gone was deleted concurrently (the
        // removal listener and DELETE /api/sessions/{id} can both run).
        for conv in conversations {
            match self.delete_conversation(&conv.id_string()).await {
                Ok(()) | Err(DbError::NotFound(_)) => {}
                Err(e) => return Err(e),
            }
        }

        // Delete settings
        let _: Option<Settings> = self.db().delete(("settings", id)).await?;

        // Delete dataset metadata. A failed delete is an error: the session
        // row below must not go while its dataset metadata stays.
        let id_owned = id.to_string();
        self.db()
            .query("DELETE FROM dataset_meta WHERE session_id = $id")
            .bind(("id", id_owned))
            .await?
            .check()?;

        // Delete session
        let _: Option<DbSession> = self.db().delete(("sessions", id)).await?;
        self.forget_session_bytes(id);

        Ok(())
    }

    /// Ids of the session rows whose `last_accessed` is before `cutoff`.
    pub async fn session_ids_last_accessed_before(
        &self,
        cutoff: DateTime<Utc>,
    ) -> Result<Vec<String>, DbError> {
        let mut result = self
            .db()
            .query("SELECT * FROM sessions WHERE last_accessed < $cutoff")
            .bind(("cutoff", Datetime::from(cutoff)))
            .await?;
        let rows: Vec<DbSession> = result.take(0)?;
        Ok(rows.iter().map(DbSession::id_string).collect())
    }

    /// Number of conversations the session holds.
    pub async fn count_conversations(&self, session_id: &str) -> Result<usize, DbError> {
        let mut result = self
            .db()
            .query("SELECT count() FROM conversations WHERE session_id = $session_id GROUP ALL")
            .bind(("session_id", session_id.to_string()))
            .await?;
        let count: Option<CountRow> = result.take(0)?;
        Ok(count.map_or(0, |c| usize::try_from(c.count).unwrap_or(0)))
    }

    /// Bytes the session's conversations store: titles, message contents
    /// and tool-call arguments, results and errors, as UTF-8 bytes (summed
    /// here; SurrealQL's `string::len` counts characters). Seeds the
    /// `session_db_bytes` counter.
    ///
    /// It reads every stored message once, one conversation at a time (each
    /// through its index); a deleted or cleared conversation releases its
    /// own bytes instead, so this runs only when a counter is first seeded.
    pub async fn session_stored_bytes(&self, session_id: &str) -> Result<u64, DbError> {
        let conversations = self.list_conversations(session_id).await?;
        let mut total = conversations
            .iter()
            .map(|c| c.title.len() as u64)
            .fold(0u64, u64::saturating_add);
        for conv in &conversations {
            let text = self.conversation_text_bytes(&conv.id_string()).await?;
            total = total.saturating_add(text);
        }
        Ok(total)
    }

    /// Bytes the messages and tool calls of conversation `conversation_id`
    /// store (contents, arguments, results and errors; the title not
    /// included), as UTF-8 bytes.
    pub(crate) async fn conversation_text_bytes(
        &self,
        conversation_id: &str,
    ) -> Result<u64, DbError> {
        let mut result = self
            .db()
            .query("SELECT content FROM messages WHERE conversation_id = $id")
            .query("SELECT arguments, result, error FROM tool_calls WHERE conversation_id = $id")
            .bind(("id", conversation_id.to_string()))
            .await?;
        let contents: Vec<MessageText> = result.take(0)?;
        let tool_texts: Vec<ToolCallText> = result.take(1)?;
        Ok(message_text_bytes(&contents).saturating_add(tool_call_text_bytes(&tool_texts)))
    }

    /// Delete the rows that no session or conversation owns any more:
    /// conversations, settings and dataset metadata whose session row is
    /// missing, and messages and tool calls whose conversation is missing.
    /// Rows written after the call started are left alone, so a session or
    /// conversation being created concurrently cannot lose its first rows.
    /// Returns the number of rows deleted.
    ///
    /// Of the rows that stay, only the session rows and the conversation rows
    /// are read: the conversation rows one session per statement, for their
    /// ids (SurrealDB lists record ids only by reading the records). No
    /// message, tool call, settings or dataset metadata row of an owner that
    /// exists is read: the owners that exist split each owner index into
    /// gaps ([`OwnerGap`]), a gap is counted from the index keys alone, and
    /// only the rows of a gap that holds some are read, [`ORPHAN_BATCH`] per
    /// statement, to delete them. So the sweep holds at most one session's
    /// conversation rows, or one batch of orphans, at a time.
    ///
    /// A store without orphans costs one pair of index counts per table
    /// (they read every key of the index, not the rows); each gap with
    /// orphans adds about two pairs per halving of the owners
    /// ([`gaps_with_rows`](Self::gaps_with_rows)). Reading every conversation
    /// row costs more than that, so a running server does this once and then
    /// calls [`delete_recent_orphans`](Self::delete_recent_orphans).
    pub async fn delete_orphans(&self) -> Result<usize, DbError> {
        self.delete_orphans_in(OrphanScan::Full).await
    }

    /// [`delete_orphans`](Self::delete_orphans) without looking at every
    /// conversation: only the messages and tool calls of the conversations
    /// deleted since the call before the last one are deleted, through the
    /// conversation index (skipping a conversation that still exists).
    /// Messages and tool calls are written only to an existing conversation,
    /// so a running server orphans them only with a write that races the
    /// conversation's delete or a delete that stopped half way, and both
    /// leave the conversation's id behind; remembering it for two calls also
    /// finds a racing write that lands after the next call started. Orphans
    /// from before the process started (a RocksDB store after a crash) need
    /// one full sweep. When more conversations were deleted than are
    /// remembered, this sweeps in full.
    pub async fn delete_recent_orphans(&self) -> Result<usize, DbError> {
        self.delete_orphans_in(OrphanScan::Recent).await
    }

    async fn delete_orphans_in(&self, scan: OrphanScan) -> Result<usize, DbError> {
        let started = Datetime::from(Utc::now());
        let session_ids: HashSet<String> = {
            let mut result = self.db().query("SELECT * FROM sessions").await?;
            let rows: Vec<DbSession> = result.take(0)?;
            rows.iter().map(DbSession::id_string).collect()
        };
        let mut deleted = 0usize;

        // Conversations (with their messages and tool calls) of sessions
        // without a row: only those are read, a batch at a time.
        for (gap, rows) in self
            .gaps_with_rows("conversations", "session_id", &session_ids)
            .await?
        {
            let (n, _) = drain_gap("conversations", rows, || {
                self.delete_orphaned_conversations(gap, &started)
            })
            .await?;
            deleted += n;
        }

        // Settings and dataset metadata of sessions without a row.
        for (table, stamp) in [("settings", "updated_at"), ("dataset_meta", "loaded_at")] {
            deleted += self
                .delete_unowned(table, "session_id", stamp, &session_ids, &started)
                .await?;
        }

        // Messages and tool calls of conversations that no longer exist.
        let recent = match scan {
            OrphanScan::Full => None,
            OrphanScan::Recent => self.take_deleted_conversations(),
        };
        match recent {
            Some(ids) => {
                for id in &ids {
                    match self.get_conversation(id).await {
                        // It still exists; its rows stay.
                        Ok(_) => continue,
                        Err(DbError::NotFound(_)) => {}
                        Err(e) => return Err(e),
                    }
                    deleted += self.delete_rows_of_deleted_conversation(id).await?;
                }
            }
            None => {
                // Read after the deletes above, so every conversation left
                // that is older than `started` has a session in
                // `session_ids`. One that does not (a session whose row was
                // written after `session_ids` was read) is younger than
                // `started`, and so are its messages and tool calls, which
                // the `stamp` condition keeps.
                let conversation_ids = self.conversation_ids_of(&session_ids).await?;
                for (table, stamp) in [("messages", "created_at"), ("tool_calls", "started_at")] {
                    deleted += self
                        .delete_unowned(
                            table,
                            "conversation_id",
                            stamp,
                            &conversation_ids,
                            &started,
                        )
                        .await?;
                }
            }
        }

        if deleted > 0 {
            tracing::info!(deleted, "Deleted orphaned conversation-store rows");
        }
        Ok(deleted)
    }

    /// Read up to [`ORPHAN_BATCH`] conversations in `gap` of the session
    /// index that were written before `started`, and delete each with its
    /// messages and tool calls. One that is already gone, or that
    /// `delete_conversation` cannot find by its key, is read but not
    /// deleted.
    async fn delete_orphaned_conversations(
        &self,
        gap: OwnerGap<'_>,
        started: &Datetime,
    ) -> Result<Batch, DbError> {
        let mut result = self
            .db()
            .query(format!(
                "SELECT id, session_id FROM conversations \
                 WHERE {} AND created_at < $started LIMIT {ORPHAN_BATCH}",
                gap.condition("session_id")
            ))
            .bind(gap.bounds())
            .bind(("started", started.clone()))
            .await?;
        let orphans: Vec<ConversationOwner> = result.take(0)?;
        let mut batch = Batch {
            read: orphans.len(),
            deleted: 0,
        };
        for conv in orphans {
            // The session has no row any more; its counter goes too.
            self.forget_session_bytes(&conv.session_id);
            match self.delete_conversation(&key_string(&conv.id)).await {
                Ok(()) => batch.deleted += 1,
                Err(DbError::NotFound(_)) => {}
                Err(e) => return Err(e),
            }
        }
        Ok(batch)
    }

    /// Delete the messages and tool calls left for deleted conversation `id`
    /// (through the conversation indexes) and return how many there were.
    /// Conversation ids are never reused, so every such row is an orphan.
    async fn delete_rows_of_deleted_conversation(&self, id: &str) -> Result<usize, DbError> {
        let mut result = self
            .db()
            .query("SELECT count() FROM messages WHERE conversation_id = $id GROUP ALL")
            .query("DELETE FROM messages WHERE conversation_id = $id")
            .query("SELECT count() FROM tool_calls WHERE conversation_id = $id GROUP ALL")
            .query("DELETE FROM tool_calls WHERE conversation_id = $id")
            .bind(("id", id.to_string()))
            .await?
            .check()?;
        let messages: Option<CountRow> = result.take(0)?;
        let tool_calls: Option<CountRow> = result.take(2)?;
        Ok([messages, tool_calls]
            .into_iter()
            .flatten()
            .map(|c| usize::try_from(c.count).unwrap_or(0))
            .sum())
    }

    /// The id of every conversation of the sessions in `session_ids`, read
    /// one session per statement through the session index, keeping only
    /// the ids. SurrealDB keeps every row a statement reads until the
    /// statement ends, so one statement over all conversations would hold
    /// every title at once; this holds one session's conversation rows at a
    /// time (what listing the session's conversations holds as well).
    async fn conversation_ids_of(
        &self,
        session_ids: &HashSet<String>,
    ) -> Result<HashSet<String>, DbError> {
        let mut ids = HashSet::new();
        // A session id with a NUL byte owns no conversation (`sorted_owners`).
        for session_id in session_ids.iter().filter(|id| !id.contains('\0')) {
            let mut result = self
                .db()
                .query("SELECT VALUE id FROM conversations WHERE session_id = $session_id")
                .bind(("session_id", session_id.clone()))
                .await?;
            let rows: Vec<RecordId> = result.take(0)?;
            ids.extend(rows.iter().map(key_string));
        }
        Ok(ids)
    }

    /// The gaps between the owners in `known` ([`owner_gaps`]) that hold
    /// rows of `table`, each with its number of rows, found through the
    /// index on `owner` without reading a row.
    ///
    /// A span of neighbouring gaps holds such rows when its range of the
    /// index holds more rows than the owners inside it own
    /// ([`unowned_rows`](Self::unowned_rows)). The whole index is one span,
    /// so a table without orphans costs one check; a span with some is
    /// halved until single gaps remain, about two checks per level of the
    /// halving for each gap that holds orphans.
    async fn gaps_with_rows<'a>(
        &self,
        table: &str,
        owner: &str,
        known: &'a HashSet<String>,
    ) -> Result<Vec<(OwnerGap<'a>, usize)>, DbError> {
        let owners = sorted_owners(known);
        let gaps = owner_gaps(&owners);
        let mut found = Vec::new();
        // Spans of gaps (first, last), the leftmost on top.
        let mut spans = vec![(0, gaps.len() - 1)];
        while let Some((first, last)) = spans.pop() {
            let span = OwnerGap {
                after: gaps[first].after,
                before: gaps[last].before,
            };
            // The owners between gap `first` and gap `last`.
            let inside = &owners[first..last];
            let rows = self.unowned_rows(table, owner, span, inside).await?;
            if rows == 0 {
                continue;
            }
            if first == last {
                found.push((span, rows));
                continue;
            }
            let mid = first + (last - first) / 2;
            spans.push((mid + 1, last));
            spans.push((first, mid));
        }
        Ok(found)
    }

    /// The rows of `table` in the range `span` of `owner` values that none
    /// of the owners `inside` it owns: the rows in the range less the rows
    /// of those owners, both counted in one transaction from the index keys
    /// (SurrealDB counts a `SELECT count() ... GROUP ALL` whose condition is
    /// a range of one index, or its column `IN` an array, by the keys,
    /// without reading a row). `inside` must be the distinct owners of the
    /// range that are to be kept.
    async fn unowned_rows(
        &self,
        table: &str,
        owner: &str,
        span: OwnerGap<'_>,
        inside: &[&str],
    ) -> Result<usize, DbError> {
        let inside: Vec<String> = inside.iter().map(|o| o.to_string()).collect();
        let mut result = self
            .db()
            .query("BEGIN TRANSACTION")
            .query(format!(
                "SELECT count() FROM {table} WHERE {} GROUP ALL",
                span.condition(owner)
            ))
            .query(format!(
                "SELECT count() FROM {table} WHERE {owner} IN $inside GROUP ALL"
            ))
            .query("COMMIT TRANSACTION")
            .bind(span.bounds())
            .bind(("inside", inside))
            .await?
            .check()?;
        let count =
            |row: Option<CountRow>| row.map_or(0, |c| usize::try_from(c.count).unwrap_or(0));
        let in_range = count(result.take(0)?);
        let owned = count(result.take(1)?);
        Ok(in_range.saturating_sub(owned))
    }

    /// Delete the rows of `table` written before `started` (by their `stamp`
    /// column) whose `owner` column (`session_id` or `conversation_id`) names
    /// none of `known`, and return how many there were. The gaps between the
    /// owners in `known` that hold rows are found through the index on
    /// `owner` ([`gaps_with_rows`](Self::gaps_with_rows)); only the rows in
    /// those gaps are read, [`ORPHAN_BATCH`] per statement, to delete them.
    async fn delete_unowned(
        &self,
        table: &str,
        owner: &str,
        stamp: &str,
        known: &HashSet<String>,
        started: &Datetime,
    ) -> Result<usize, DbError> {
        let mut deleted = 0usize;
        for (gap, rows) in self.gaps_with_rows(table, owner, known).await? {
            let select = format!(
                "SELECT VALUE id FROM {table} WHERE {} AND {stamp} < $started \
                 LIMIT {ORPHAN_BATCH}",
                gap.condition(owner)
            );
            let (n, _) =
                drain_gap(table, rows, || self.delete_selected(&select, gap, started)).await?;
            deleted += n;
        }
        Ok(deleted)
    }

    /// Delete the rows whose ids `select` returns (a `SELECT VALUE id` on
    /// `$lo`, `$hi` and `$started`, at most [`ORPHAN_BATCH`] rows) in one
    /// statement.
    async fn delete_selected(
        &self,
        select: &str,
        gap: OwnerGap<'_>,
        started: &Datetime,
    ) -> Result<Batch, DbError> {
        let mut result = self
            .db()
            .query(select)
            .bind(gap.bounds())
            .bind(("started", started.clone()))
            .await?;
        let ids: Vec<RecordId> = result.take(0)?;
        let read = ids.len();
        if read > 0 {
            self.db()
                .query("DELETE $ids")
                .bind(("ids", ids))
                .await?
                .check()?;
        }
        Ok(Batch {
            read,
            deleted: read,
        })
    }

    /// Clean up sessions older than the given number of days
    pub async fn cleanup_old_sessions(&self, days: u32) -> Result<u32, DbError> {
        let cutoff = Datetime::from(Utc::now() - chrono::Duration::days(days as i64));

        // Get old sessions
        let mut result = self
            .db()
            .query("SELECT id FROM sessions WHERE last_accessed < $cutoff")
            .bind(("cutoff", cutoff))
            .await?;

        #[derive(serde::Deserialize)]
        struct IdOnly {
            id: RecordId,
        }

        let old_sessions: Vec<IdOnly> = result.take(0)?;
        let count = old_sessions.len() as u32;

        // Delete each old session
        for session in old_sessions {
            let id_str = session.id.key().to_string();
            if let Err(e) = self.delete_session(&id_str).await {
                tracing::warn!("Failed to delete old session {}: {}", id_str, e);
            }
        }

        if count > 0 {
            tracing::info!("Cleaned up {} old sessions", count);
        }

        Ok(count)
    }

    // ==================== Settings Operations ====================

    /// Get or create settings for a session
    pub async fn get_or_create_settings(&self, session_id: &str) -> Result<Settings, DbError> {
        // Try to get existing settings
        let existing: Option<Settings> = self.db().select(("settings", session_id)).await?;

        if let Some(settings) = existing {
            return Ok(settings);
        }

        // Create default settings
        let settings = Settings::default_for_session(session_id.to_string());
        let created: Option<Settings> = self
            .db()
            .create(settings.id.clone())
            .content(settings)
            .await?;

        created.ok_or_else(|| DbError::Query("Failed to create settings".to_string()))
    }

    /// Update settings for a session
    pub async fn update_settings(&self, settings: &Settings) -> Result<Settings, DbError> {
        for (what, value) in [
            ("provider", Some(&settings.provider)),
            ("model", Some(&settings.model)),
            ("base_url", settings.base_url.as_ref()),
            ("system_prompt", settings.system_prompt.as_ref()),
        ] {
            self.check_message_bytes(what, "setting", value.map_or(0, String::len))?;
        }
        let mut updated_settings = settings.clone();
        updated_settings.updated_at = Datetime::from(Utc::now());
        let session_id = settings.session_id.clone();

        let result: Option<Settings> = self
            .db()
            .update(("settings", &session_id))
            .content(updated_settings)
            .await?;

        result.ok_or_else(|| DbError::NotFound("Settings not found".to_string()))
    }

    /// Update specific settings fields
    pub async fn patch_settings(
        &self,
        session_id: &str,
        provider: Option<&str>,
        model: Option<&str>,
        temperature: Option<f64>,
        max_tokens: Option<i32>,
    ) -> Result<Settings, DbError> {
        for (what, value) in [("provider", provider), ("model", model)] {
            self.check_message_bytes(what, "setting", value.map_or(0, str::len))?;
        }
        // Build update SET clauses dynamically
        let mut set_parts = vec!["updated_at = time::now()".to_string()];

        if provider.is_some() {
            set_parts.push("provider = $provider".to_string());
        }
        if model.is_some() {
            set_parts.push("model = $model".to_string());
        }
        if temperature.is_some() {
            set_parts.push("temperature = $temperature".to_string());
        }
        if max_tokens.is_some() {
            set_parts.push("max_tokens = $max_tokens".to_string());
        }

        let query = format!(
            "UPDATE settings SET {} WHERE id = $id RETURN AFTER",
            set_parts.join(", ")
        );

        let session_id_owned = session_id.to_string();
        let mut result = self
            .db()
            .query(&query)
            .bind((
                "id",
                RecordId::from(("settings", session_id_owned.as_str())),
            ))
            .bind(("provider", provider.map(|s| s.to_string())))
            .bind(("model", model.map(|s| s.to_string())))
            .bind(("temperature", temperature))
            .bind(("max_tokens", max_tokens))
            .await?;

        let settings: Option<Settings> = result.take(0)?;
        settings.ok_or_else(|| DbError::NotFound("Settings not found".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn known(ids: &[&str]) -> HashSet<String> {
        ids.iter().map(|id| id.to_string()).collect()
    }

    /// The gaps run in byte order from the empty string up, each bounded by
    /// the owners on its two sides; without owners there is one gap, and
    /// every gap is a range (no gap lacks a lower bound).
    #[test]
    fn owner_gaps_cover_everything_between_the_known_owners() {
        let ids = known(&["b", "a", "ab"]);
        let owners = sorted_owners(&ids);
        assert_eq!(owners, vec!["a", "ab", "b"]);
        let gaps = owner_gaps(&owners);
        let pairs: Vec<_> = gaps.iter().map(|g| (g.after, g.before)).collect();
        assert_eq!(
            pairs,
            vec![
                (None, Some("a")),
                (Some("a"), Some("ab")),
                (Some("ab"), Some("b")),
                (Some("b"), None),
            ]
        );
        assert_eq!(gaps[0].condition("o"), "o >= $lo AND o < $hi");
        assert_eq!(gaps[1].condition("o"), "o > $lo AND o < $hi");
        assert_eq!(gaps[3].condition("o"), "o > $lo");
        let first = gaps[0].bounds();
        assert_eq!((first.lo.as_str(), first.hi.as_str()), ("", "a"));

        let all = owner_gaps(&[]);
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].condition("o"), "o >= $lo");
        assert_eq!(all[0].bounds().lo, "");
    }

    /// What [`drain`] saw: the rows deleted, how the gap ended, the batches
    /// read and the warnings logged.
    #[derive(Debug)]
    struct Drain {
        deleted: usize,
        end: GapEnd,
        batches: usize,
        warnings: Vec<String>,
    }

    /// `drain_gap` over batches that read `reads[i]` rows (a full batch
    /// past the end of `reads`) and delete all they read, except batch
    /// `stuck`, which deletes nothing.
    async fn drain(rows: usize, reads: &[usize], stuck: Option<usize>) -> Drain {
        let logs = super::super::tests::Logs::default();
        let _default = tracing::subscriber::set_default(logs.subscriber());
        let mut batches = 0usize;
        let (deleted, end) = drain_gap("t", rows, || {
            let i = batches;
            batches += 1;
            let read = reads.get(i).copied().unwrap_or(ORPHAN_BATCH);
            let deleted = if stuck == Some(i) { 0 } else { read };
            std::future::ready(Ok(Batch { read, deleted }))
        })
        .await
        .unwrap();
        let warnings = logs
            .text()
            .lines()
            .filter(|line| line.contains("WARN"))
            .map(String::from)
            .collect();
        Drain {
            deleted,
            end,
            batches,
            warnings,
        }
    }

    /// A gap is read until a batch comes back short, and no further than
    /// its count allows (one batch more than its rows fill): a gap that
    /// still returns full batches then is left with a warning.
    #[tokio::test]
    async fn drain_gap_stops_at_a_short_batch_or_its_bound() {
        let b = ORPHAN_BATCH;
        for (rows, reads) in [
            (2 * b + 5, vec![b, b, 5]),
            (0, vec![0]),
            // Exactly full batches: the empty read after them ends the gap.
            (2 * b, vec![b, b, 0]),
        ] {
            let d = drain(rows, &reads, None).await;
            let read: usize = reads.iter().sum();
            assert_eq!(
                (d.deleted, d.end, d.batches),
                (read, GapEnd::Drained, reads.len())
            );
            assert!(d.warnings.is_empty(), "{:?}", d.warnings);
        }
        // Rows that keep coming (written after the count): 2 + 1 batches for
        // a count of b + 1, 1 for an empty count, then the bound.
        for (rows, batches) in [(b + 1, 3), (0, 1)] {
            let d = drain(rows, &[], None).await;
            assert_eq!(
                (d.deleted, d.end, d.batches),
                (batches * b, GapEnd::Bound, batches)
            );
            assert_eq!(d.warnings.len(), 1, "{:?}", d.warnings);
            assert!(d.warnings[0].contains("batch bound"), "{}", d.warnings[0]);
            assert!(d.warnings[0].contains("table=\"t\""), "{}", d.warnings[0]);
        }
    }

    /// A full batch of which nothing could be deleted ends the gap at once,
    /// with a warning, instead of being read again; a short one ends it as
    /// drained.
    #[tokio::test]
    async fn drain_gap_stops_at_a_full_batch_it_could_not_delete() {
        let b = ORPHAN_BATCH;
        let d = drain(10 * b, &[b, b, b], Some(1)).await;
        assert_eq!((d.deleted, d.end, d.batches), (b, GapEnd::Stuck, 2));
        assert_eq!(d.warnings.len(), 1, "{:?}", d.warnings);
        assert!(
            d.warnings[0].contains("could not delete"),
            "{}",
            d.warnings[0]
        );

        let d = drain(10 * b, &[3], Some(0)).await;
        assert_eq!((d.deleted, d.end, d.batches), (0, GapEnd::Drained, 1));
        assert!(d.warnings.is_empty(), "{:?}", d.warnings);
    }

    /// An owner with a NUL byte is no bound (it could not be bound to a
    /// query); the gaps around it merge.
    #[test]
    fn owner_gaps_skip_an_owner_with_a_nul_byte() {
        let ids = known(&["a", "b\0c", "d"]);
        let owners = sorted_owners(&ids);
        assert_eq!(owners, vec!["a", "d"]);
        let pairs: Vec<_> = owner_gaps(&owners)
            .iter()
            .map(|g| (g.after, g.before))
            .collect();
        assert_eq!(
            pairs,
            vec![(None, Some("a")), (Some("a"), Some("d")), (Some("d"), None)]
        );
    }
}
