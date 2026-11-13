//! Dataset metadata CRUD operations

use surrealdb::types::SurrealValue;
use tracing;

use super::connection::{DbConnection, DbError, run_to_end};
use super::models::DatasetMeta;

/// The most column names one dataset's metadata row keeps (the dataset's
/// first ones; `column_count` keeps its width).
///
/// The names are for display (the dataset sidebar and inspector, the chat
/// input's suggestions). Measured on SurrealDB 2.4, the store's work on a
/// row was quadratic in the length of an array field: `column_names TYPE
/// array<string>` implicitly defines `column_names[*]`, and the schema check
/// of a SCHEMAFULL row (surrealdb-core 2.4 `cleanup_table_fields`) looked
/// each element's path up in a list of every element's path. The embedded
/// engine runs each query in a task on the server's async runtime, whoever
/// awaits it, so an unbounded list let one wide dataset hold a runtime worker
/// for minutes. The bound stays on SurrealDB 4: the names are for display.
pub const MAX_STORED_COLUMN_NAMES: usize = 1_000;

/// The most dataset metadata rows one session keeps when the server sets a
/// `session_db_bytes` budget: a save under a new name past them is refused
/// with `QuotaExceeded`, while a save that replaces a row is not. Without a
/// budget there is no bound. The bytes budget alone would admit tens of
/// thousands of small rows, each of which the dataset sidebar lists.
pub const MAX_DATASET_META_ROWS_PER_SESSION: usize = 1_000;

/// The most bytes of column names one dataset's metadata row keeps: a few
/// long names stop the list earlier.
pub const MAX_STORED_COLUMN_NAME_BYTES: usize = 64 * 1024;

/// The longest prefix of `names` within [`MAX_STORED_COLUMN_NAMES`] names
/// and [`MAX_STORED_COLUMN_NAME_BYTES`] bytes.
fn stored_column_names(names: &[String]) -> Vec<String> {
    let mut bytes = 0usize;
    names
        .iter()
        .take(MAX_STORED_COLUMN_NAMES)
        .take_while(|name| {
            bytes = bytes.saturating_add(name.len());
            bytes <= MAX_STORED_COLUMN_NAME_BYTES
        })
        .cloned()
        .collect()
}

/// The bytes a dataset metadata row counts against its session's
/// `session_db_bytes` budget for its fields other than texts: its id, its
/// counts, its timestamp and its file size.
const DATASET_META_FIXED_BYTES: u64 = 128;

/// The texts of a stored dataset metadata row that the `session_db_bytes`
/// budget counts. Other fields of the row are ignored.
#[derive(SurrealValue)]
struct DatasetMetaText {
    session_id: String,
    name: String,
    source_path: Option<String>,
    source_type: String,
    column_names: Vec<String>,
    #[surreal(default)]
    na_strings: Option<Vec<String>>,
}

impl DatasetMetaText {
    fn bytes(&self) -> u64 {
        dataset_meta_row_bytes(
            &self.session_id,
            &self.name,
            self.source_path.as_deref(),
            &self.source_type,
            &self.column_names,
            self.na_strings.as_deref(),
        )
    }
}

/// The bytes of the dataset metadata rows `rows`.
fn dataset_meta_text_bytes(rows: &[DatasetMetaText]) -> u64 {
    rows.iter()
        .map(DatasetMetaText::bytes)
        .fold(0u64, u64::saturating_add)
}

/// The bytes one dataset metadata row counts against its session's
/// `session_db_bytes` budget: the UTF-8 bytes of its session id, name,
/// source path and type, stored column names and missing-value markers, and
/// [`DATASET_META_FIXED_BYTES`]. The one definition for both sides of the
/// counter, the saves that count a row and the seed and deletes that read
/// what is stored.
fn dataset_meta_row_bytes(
    session_id: &str,
    name: &str,
    source_path: Option<&str>,
    source_type: &str,
    column_names: &[String],
    na_strings: Option<&[String]>,
) -> u64 {
    let len = |s: &str| s.len() as u64;
    [session_id, name, source_type]
        .into_iter()
        .chain(source_path)
        .chain(column_names.iter().map(String::as_str))
        .chain(na_strings.unwrap_or_default().iter().map(String::as_str))
        .map(len)
        .fold(DATASET_META_FIXED_BYTES, u64::saturating_add)
}

/// The bytes `meta`'s row counts ([`dataset_meta_row_bytes`]), as it is
/// (call it on the [`stored_meta`] copy).
fn meta_bytes(meta: &DatasetMeta) -> u64 {
    dataset_meta_row_bytes(
        &meta.session_id,
        &meta.name,
        meta.source_path.as_deref(),
        &meta.source_type,
        &meta.column_names,
        meta.na_strings.as_deref(),
    )
}

/// `meta` as its row is stored: its column names cut to
/// [`stored_column_names`], every other field as it is.
fn stored_meta(meta: &DatasetMeta) -> DatasetMeta {
    DatasetMeta {
        id: meta.id.clone(),
        session_id: meta.session_id.clone(),
        name: meta.name.clone(),
        source_path: meta.source_path.clone(),
        source_type: meta.source_type.clone(),
        row_count: meta.row_count,
        column_count: meta.column_count,
        column_names: stored_column_names(&meta.column_names),
        loaded_at: meta.loaded_at,
        file_size_bytes: meta.file_size_bytes,
        na_strings: meta.na_strings.clone(),
    }
}

impl DbConnection {
    // ==================== Dataset Meta Operations ====================

    /// Save dataset metadata (upsert based on session_id + name). The row
    /// keeps the first column names only, at most
    /// [`MAX_STORED_COLUMN_NAMES`] of them and
    /// [`MAX_STORED_COLUMN_NAME_BYTES`] bytes; `column_count` is kept as
    /// given.
    ///
    /// With a `session_db_bytes` budget, the row counts against it beside
    /// the session's conversation history (the bytes of its texts, kept
    /// column names included, and 128 for its other fields): a new row, or
    /// the bytes a replacing row adds, that would pass it is refused with
    /// `QuotaExceeded` and nothing is written; the bytes a smaller
    /// replacing row frees are released once it is written. The saves and
    /// deletes of one session's rows take turns
    /// ([`dataset_meta_turn`](DbConnection::dataset_meta_turn)), in a task a
    /// dropped caller does not stop ([`run_to_end`]), so two saves under one
    /// name cannot both create a row or count it twice.
    pub async fn save_dataset_meta(&self, meta: &DatasetMeta) -> Result<DatasetMeta, DbError> {
        let given_names = meta.column_names.len();
        let stored = stored_meta(meta);
        tracing::info!(
            session_id = %stored.session_id,
            name = %stored.name,
            column_names_len = stored.column_names.len(),
            given_column_names_len = given_names,
            "[DB_SAVE_META] Starting save_dataset_meta"
        );
        tracing::debug!(
            column_names = ?stored.column_names,
            "[DB_SAVE_META] Column names to save"
        );

        let db = self.clone();
        let turn = db.dataset_meta_turn(&stored.session_id).await;
        run_to_end(turn, async move { db.save_stored_meta(stored).await }).await
    }

    /// [`save_dataset_meta`](Self::save_dataset_meta) of the row `stored`
    /// (its names already cut), in its session's dataset metadata turn.
    async fn save_stored_meta(&self, mut stored: DatasetMeta) -> Result<DatasetMeta, DbError> {
        // Check if a dataset with the same name exists in this session
        let existing = self
            .get_dataset_meta(&stored.session_id, &stored.name)
            .await?;

        // Under a budget a session keeps a bounded number of rows; a row
        // that replaces one under its name adds none.
        if existing.is_none() && self.tracks_session_bytes() {
            self.check_dataset_meta_rows(&stored.session_id).await?;
        }

        // Only what the row adds over the row it replaces counts; what it
        // frees is released once it is written.
        let (added, freed) = {
            let new = meta_bytes(&stored);
            let old = existing.as_ref().map_or(0, meta_bytes);
            (new.saturating_sub(old), old.saturating_sub(new))
        };
        self.reserve_session_bytes(&stored.session_id, added)
            .await?;
        let session_id = stored.session_id.clone();

        let saved = if let Some(existing_meta) = existing {
            tracing::info!(
                existing_id = %existing_meta.id_string(),
                existing_column_names_len = existing_meta.column_names.len(),
                "[DB_SAVE_META] Found existing record, will update"
            );

            // Update existing record
            stored.id = existing_meta.id.clone();
            tracing::info!(
                update_column_names_len = stored.column_names.len(),
                "[DB_SAVE_META] Content being sent to UPDATE"
            );

            let result: Option<DatasetMeta> = self
                .db()
                .update(("dataset_meta", existing_meta.id_string().as_str()))
                .content(stored)
                .await?;

            match &result {
                Some(updated) => {
                    tracing::info!(
                        updated_id = %updated.id_string(),
                        updated_column_names_len = updated.column_names.len(),
                        "[DB_SAVE_META] UPDATE returned successfully"
                    );
                }
                None => {
                    tracing::error!("[DB_SAVE_META] UPDATE returned None!");
                }
            }

            result.ok_or_else(|| DbError::Query("Failed to update dataset meta".to_string()))?
        } else {
            tracing::info!(
                new_id = %stored.id_string(),
                "[DB_SAVE_META] No existing record, will create new"
            );

            // Create new record
            let created: Option<DatasetMeta> =
                self.db().create(stored.id.clone()).content(stored).await?;

            match &created {
                Some(new_meta) => {
                    tracing::info!(
                        created_id = %new_meta.id_string(),
                        created_column_names_len = new_meta.column_names.len(),
                        "[DB_SAVE_META] CREATE returned successfully"
                    );
                }
                None => {
                    tracing::error!("[DB_SAVE_META] CREATE returned None!");
                }
            }

            created.ok_or_else(|| DbError::Query("Failed to create dataset meta".to_string()))?
        };
        self.release_session_bytes(&session_id, freed);
        Ok(saved)
    }

    /// Get all datasets for a session
    pub async fn get_datasets_for_session(
        &self,
        session_id: &str,
    ) -> Result<Vec<DatasetMeta>, DbError> {
        tracing::info!(
            session_id = %session_id,
            "[DB_GET_DATASETS] Querying datasets for session"
        );

        let session_id_owned = session_id.to_string();
        let mut result = self
            .db()
            .query(
                "SELECT * FROM dataset_meta WHERE session_id = $session_id ORDER BY loaded_at DESC",
            )
            .bind(("session_id", session_id_owned))
            .await?;

        let datasets: Vec<DatasetMeta> = result.take(0)?;

        tracing::info!(
            num_datasets = datasets.len(),
            "[DB_GET_DATASETS] Query returned datasets"
        );

        // One debug event per row: a session's listing logs two info lines
        // whatever its number of datasets.
        for (idx, ds) in datasets.iter().enumerate() {
            tracing::debug!(
                idx = idx,
                id = %ds.id_string(),
                name = %ds.name,
                column_names_len = ds.column_names.len(),
                column_names = ?ds.column_names,
                "[DB_GET_DATASETS] Dataset record from SurrealDB"
            );
        }

        Ok(datasets)
    }

    /// Get a specific dataset by session_id and name
    pub async fn get_dataset_meta(
        &self,
        session_id: &str,
        name: &str,
    ) -> Result<Option<DatasetMeta>, DbError> {
        tracing::debug!(
            session_id = %session_id,
            name = %name,
            "[DB_GET_META] Looking up specific dataset"
        );

        let session_id_owned = session_id.to_string();
        let name_owned = name.to_string();
        let mut result = self
            .db()
            .query("SELECT * FROM dataset_meta WHERE session_id = $session_id AND name = $name")
            .bind(("session_id", session_id_owned))
            .bind(("name", name_owned))
            .await?;

        let datasets: Vec<DatasetMeta> = result.take(0)?;
        let found = datasets.into_iter().next();

        match &found {
            Some(meta) => {
                tracing::debug!(
                    id = %meta.id_string(),
                    column_names = ?meta.column_names,
                    column_names_len = meta.column_names.len(),
                    "[DB_GET_META] Found dataset"
                );
            }
            None => {
                tracing::debug!("[DB_GET_META] Dataset not found");
            }
        }

        Ok(found)
    }

    /// Get a dataset meta by ID
    pub async fn get_dataset_meta_by_id(&self, id: &str) -> Result<DatasetMeta, DbError> {
        let meta: Option<DatasetMeta> = self.db().select(("dataset_meta", id)).await?;
        meta.ok_or_else(|| DbError::NotFound(format!("Dataset meta not found: {}", id)))
    }

    /// Delete dataset metadata by session_id and name. With a
    /// `session_db_bytes` budget, the bytes of the rows it deleted are
    /// released. Runs in the session's dataset metadata turn, in a task a
    /// dropped caller does not stop ([`run_to_end`]).
    pub async fn delete_dataset_meta(&self, session_id: &str, name: &str) -> Result<(), DbError> {
        let db = self.clone();
        let (session_id, name) = (session_id.to_string(), name.to_string());
        let turn = db.dataset_meta_turn(&session_id).await;
        run_to_end(turn, async move {
            let mut result = db
                .db()
                .query(
                    "DELETE FROM dataset_meta WHERE session_id = $session_id AND name = $name \
                     RETURN BEFORE",
                )
                .bind(("session_id", session_id.clone()))
                .bind(("name", name))
                .await?;
            let deleted: Vec<DatasetMetaText> = result.take(0)?;
            db.release_session_bytes(&session_id, dataset_meta_text_bytes(&deleted));
            Ok(())
        })
        .await
    }

    /// Delete all dataset metadata for a session and return how many rows
    /// there were. With a `session_db_bytes` budget, their bytes are
    /// released. Runs in the session's dataset metadata turn, in a task a
    /// dropped caller does not stop ([`run_to_end`]).
    pub async fn delete_all_dataset_meta_for_session(
        &self,
        session_id: &str,
    ) -> Result<u32, DbError> {
        let db = self.clone();
        let session_id = session_id.to_string();
        let turn = db.dataset_meta_turn(&session_id).await;
        run_to_end(turn, async move {
            let mut result = db
                .db()
                .query("DELETE FROM dataset_meta WHERE session_id = $session_id RETURN BEFORE")
                .bind(("session_id", session_id.clone()))
                .await?;
            let deleted: Vec<DatasetMetaText> = result.take(0)?;
            db.release_session_bytes(&session_id, dataset_meta_text_bytes(&deleted));
            Ok(u32::try_from(deleted.len()).unwrap_or(u32::MAX))
        })
        .await
    }

    /// Bytes the session's dataset metadata rows count against its
    /// `session_db_bytes` budget ([`dataset_meta_row_bytes`]); part of the
    /// counter's seed
    /// ([`session_stored_bytes`](Self::session_stored_bytes)).
    pub(crate) async fn dataset_meta_bytes(&self, session_id: &str) -> Result<u64, DbError> {
        let mut result = self
            .db()
            .query(
                "SELECT session_id, name, source_path, source_type, column_names, \
                 na_strings FROM dataset_meta WHERE session_id = $session_id",
            )
            .bind(("session_id", session_id.to_string()))
            .await?;
        let rows: Vec<DatasetMetaText> = result.take(0)?;
        Ok(dataset_meta_text_bytes(&rows))
    }

    /// Refuse with `QuotaExceeded` when the session already keeps
    /// [`MAX_DATASET_META_ROWS_PER_SESSION`] dataset metadata rows.
    async fn check_dataset_meta_rows(&self, session_id: &str) -> Result<(), DbError> {
        #[derive(Debug, SurrealValue)]
        struct Count {
            count: u64,
        }
        let mut result = self
            .db()
            .query("SELECT count() FROM dataset_meta WHERE session_id = $session_id GROUP ALL")
            .bind(("session_id", session_id.to_string()))
            .await?;
        let count: Option<Count> = result.take(0)?;
        let rows = count.map_or(0, |c| c.count);
        if rows >= MAX_DATASET_META_ROWS_PER_SESSION as u64 {
            return Err(DbError::QuotaExceeded(format!(
                "this session already keeps the metadata of {rows} datasets, and this server \
                 keeps at most {MAX_DATASET_META_ROWS_PER_SESSION} per session; remove a \
                 dataset, or start a new session"
            )));
        }
        Ok(())
    }

    /// Update the source path for a dataset (useful when file moves). With a
    /// `session_db_bytes` budget, the bytes a longer path adds count against
    /// it (`QuotaExceeded` when they would pass it), and those a shorter one
    /// frees are released once it is written. Runs in the session's dataset
    /// metadata turn, in a task a dropped caller does not stop
    /// ([`run_to_end`]).
    pub async fn update_dataset_source_path(
        &self,
        session_id: &str,
        name: &str,
        source_path: &str,
    ) -> Result<DatasetMeta, DbError> {
        let db = self.clone();
        let (session_id, name, source_path) = (
            session_id.to_string(),
            name.to_string(),
            source_path.to_string(),
        );
        let turn = db.dataset_meta_turn(&session_id).await;
        run_to_end(turn, async move {
            let not_found = || {
                DbError::NotFound(format!(
                    "Dataset meta not found: session={}, name={}",
                    session_id, name
                ))
            };
            // The stored path's length is needed only for the session's
            // byte counter; without a budget the update alone finds the row
            // (or nothing, which is `NotFound`).
            let (mut reserved, mut freed) = (0, 0);
            if db.tracks_session_bytes() {
                let Some(existing) = db.get_dataset_meta(&session_id, &name).await? else {
                    return Err(not_found());
                };
                let old = existing.source_path.as_deref().map_or(0, str::len);
                reserved = source_path.len().saturating_sub(old) as u64;
                db.reserve_session_bytes(&session_id, reserved).await?;
                freed = old.saturating_sub(source_path.len()) as u64;
            }

            let mut result = db
                .db()
                .query(
                    r#"UPDATE dataset_meta SET
                    source_path = $source_path
                WHERE session_id = $session_id AND name = $name RETURN AFTER"#,
                )
                .bind(("session_id", session_id.clone()))
                .bind(("name", name.clone()))
                .bind(("source_path", source_path))
                .await?;

            let mut updated: Vec<DatasetMeta> = result.take(0)?;
            // The (session_id, name) index is unique, so at most one row
            // matched; the UPDATE has committed by now either way.
            debug_assert!(
                updated.len() <= 1,
                "{} dataset metadata rows matched session={session_id}, name={name}",
                updated.len()
            );
            let updated = updated.pop().ok_or_else(not_found)?;
            db.release_session_bytes(&session_id, freed);
            Ok(updated)
        })
        .await
    }
}
