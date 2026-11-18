//! The session's dataset list in the store, kept in step with its dataset
//! map after every tool call.
//!
//! `GET /api/sessions/{id}/datasets` (the dataset sidebar of the web app)
//! lists the session's dataset metadata rows. The rows of the loaders
//! (`load_dataset`, `upload_dataset`, `create_dataset`) are written from
//! their result text, with the source a reload reads
//! ([`capture_loader_metadata`], on the blocking pool:
//! `routes_tools::capture_dataset_metadata_off_runtime`). Every other tool
//! that makes a dataset (generated data, a filter, a join, an import) wrote
//! none, so its dataset was in the session but not in the list. After each
//! call, the tool route and the LLM tool loop both call
//! [`sync_dataset_meta`], which reads the session's own map, not a tool's
//! text:
//!
//! - a dataset with no row gets one: its name, row and column counts and
//!   column names, no source path, and the source type
//!   [`SOURCE_TYPE_GENERATED`] or [`SOURCE_TYPE_DERIVED`], which a reload
//!   skips with a note (no file holds the dataset);
//! - a row whose counts or names differ from the dataset is updated; when
//!   a tool other than a loader changed the dataset in the call (a filter
//!   that overwrites a loaded dataset), the row loses its source, since the
//!   file no longer holds what the session does;
//! - a dataset that left the map during the call (a session import that
//!   replaces the session's datasets) loses its row.
//!
//! A call that changed no dataset costs two passes over the map's names
//! (a [`DatasetSnapshot`] before and after) and no store query; one that
//! did costs one listing query and, per changed row, a read and a write. No frame is
//! serialised. Datasets that leave the map for other reasons (a server
//! restart, eviction) keep their rows, which a reload reads.
//!
//! Two calls in one session can overlap where the session runs more than
//! one call at a time (limits off: local servers, the desktop app; a hosted
//! session runs one, but a call refused as `session_busy` is synced too).
//! The snapshot shows what changed during the call, not who changed it, so
//! a dataset a loader put in the map while another call ran looks to that
//! call like one it made. Two rules keep the loader's row:
//!
//! - a call during which a loader call also ran (one was running when the
//!   call took its snapshot, or one started since: [`LoaderCalls`]) clears
//!   no source and deletes no loaded dataset's row; it only adds rows and
//!   corrects counts and names. A loaded dataset such a call changed in
//!   place keeps its source until a later call changes it again;
//! - each save and delete is made only over the row the call listed, read
//!   again in the store's turn for the session's rows, so a row another call
//!   wrote after the listing is left to that write.
//!
//! What the sync does not see: a tool that replaces a loaded dataset with
//! one of the same shape and column names leaves its row, and its source,
//! as they were.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

use p2a_core::data::Dataset;

use super::routes_tools::METADATA_TOOLS;
use crate::db::{DatasetMeta, DbConnection, DbError, SOURCE_TYPE_DERIVED, SOURCE_TYPE_GENERATED};
use crate::session::{LoaderCall, LoaderCalls, Session};

/// A dataset's row and column counts and a hash of its column names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Shape {
    rows: usize,
    cols: usize,
    names: u64,
}

impl Shape {
    fn of(dataset: &Dataset) -> Self {
        let mut hasher = DefaultHasher::new();
        for column in dataset.df().get_columns() {
            column.name().as_str().hash(&mut hasher);
        }
        Self {
            rows: dataset.nrows(),
            cols: dataset.ncols(),
            names: hasher.finish(),
        }
    }
}

/// The shapes of a session's datasets by name.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Shapes(HashMap<String, Shape>);

impl Shapes {
    async fn of(session: &Session) -> Self {
        let datasets = session.datasets.read().await;
        Self(
            datasets
                .iter()
                .map(|(name, dataset)| (name.clone(), Shape::of(dataset)))
                .collect(),
        )
    }
}

/// A session's datasets before a tool call, which [`sync_dataset_meta`]
/// compares with after it to tell what the call changed, and the session's
/// dataset-loader calls at that moment ([`LoaderCalls`]), to tell whether a
/// loader ran while the call did. A snapshot taken for a loader counts that
/// call as running until it is dropped, after the call's sync.
#[derive(Debug)]
pub(crate) struct DatasetSnapshot {
    shapes: Shapes,
    /// Loader calls started when the snapshot was taken.
    loaders_started: u64,
    /// Another loader call was running when the snapshot was taken.
    loader_running: bool,
    /// This call's own count as a running loader.
    _loader: Option<LoaderCall>,
}

impl DatasetSnapshot {
    /// The shapes of `session`'s datasets now, before a call of `tool`.
    pub(crate) async fn of(session: &Session, tool: &str) -> Self {
        let calls = &session.loader_calls;
        let (loader, loaders_started, others) = if METADATA_TOOLS.contains(&tool) {
            let (call, started) = calls.start();
            (Some(call), started, calls.running().saturating_sub(1))
        } else {
            (None, calls.started(), calls.running())
        };
        Self {
            shapes: Shapes::of(session).await,
            loaders_started,
            loader_running: others > 0,
            _loader: loader,
        }
    }

    /// Whether another loader call ran at some time between this snapshot
    /// and now: one was running when it was taken, or one started since.
    fn loader_overlapped(&self, session: &Session) -> bool {
        self.loader_running || session.loader_calls.started() != self.loaders_started
    }
}

/// What one [`sync_dataset_meta`] wrote: the names whose rows it saved and
/// those whose rows it deleted, each sorted.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct SyncOutcome {
    pub saved: Vec<String>,
    pub deleted: Vec<String>,
}

/// One live dataset as its row would store it.
struct Live {
    shape: Shape,
    row_count: i32,
    column_count: i32,
    column_names: Vec<String>,
}

/// The source type of a row a call of `tool` adds.
fn made_by(tool: &str) -> &'static str {
    if tool == "generate_random_data" {
        SOURCE_TYPE_GENERATED
    } else {
        SOURCE_TYPE_DERIVED
    }
}

/// Bring the dataset metadata rows of `session_id` in the store in step with
/// `session`'s dataset map after a call of `tool`; `before` is the map's
/// [`DatasetSnapshot`] taken before the call, and `loader` says whether the
/// call's own rows were written from its result text (a successful call of
/// a loader the store captures), whose source fields are kept. See the
/// module documentation for the rules. Store failures are logged, a
/// `session_db_bytes` refusal at warn; the call's answer is never changed.
pub(crate) async fn sync_dataset_meta(
    db: &DbConnection,
    session: &Session,
    session_id: &str,
    tool: &str,
    before: &DatasetSnapshot,
    loader: bool,
) -> SyncOutcome {
    let mut outcome = SyncOutcome::default();
    let after = Shapes::of(session).await;
    if after == before.shapes && !loader {
        return outcome;
    }

    // The live datasets as rows would store them, read under one lock.
    let live: HashMap<String, Live> = {
        let datasets = session.datasets.read().await;
        datasets
            .iter()
            .map(|(name, dataset)| {
                let names = dataset.column_names();
                (
                    name.clone(),
                    Live {
                        shape: Shape::of(dataset),
                        row_count: i32::try_from(dataset.nrows()).unwrap_or(i32::MAX),
                        column_count: i32::try_from(dataset.ncols()).unwrap_or(i32::MAX),
                        column_names: crate::db::stored_column_names(&names),
                    },
                )
            })
            .collect()
    };

    let stored: HashMap<String, DatasetMeta> = match db.get_datasets_for_session(session_id).await {
        Ok(rows) => rows
            .into_iter()
            .map(|row| (row.name.clone(), row))
            .collect(),
        Err(e) => {
            tracing::warn!(
                tool,
                session_id,
                error = %e,
                "[DATASET_SYNC] The session's dataset rows could not be read"
            );
            return outcome;
        }
    };

    // A loader that ran during this call may have put a dataset in the map
    // and stored its row: this call did not change that dataset, so it
    // clears no source (read after the listing, so that a loader's row in
    // it is covered; a write after the listing is caught by the saves'
    // own check).
    let overlapped = !loader && before.loader_overlapped(session);

    let mut names: Vec<&String> = live.keys().collect();
    names.sort_unstable();
    for name in names {
        let live = &live[name];
        // A dataset this call made or changed, other than a loader's own.
        let changed_by_tool = !loader && before.shapes.0.get(name) != Some(&live.shape);
        let listed = stored.get(name);
        let meta = match listed {
            None => DatasetMeta::new(
                session_id.to_string(),
                name.clone(),
                made_by(tool).to_string(),
                live.row_count,
                live.column_count,
                live.column_names.clone(),
            ),
            Some(row)
                if row.row_count == live.row_count
                    && row.column_count == live.column_count
                    && row.column_names == live.column_names =>
            {
                continue;
            }
            Some(row) => {
                let mut meta = row.clone();
                meta.row_count = live.row_count;
                meta.column_count = live.column_count;
                meta.column_names = live.column_names.clone();
                if changed_by_tool && !overlapped {
                    // No file holds the dataset any more.
                    meta.source_path = None;
                    meta.file_size_bytes = None;
                    meta.na_strings = None;
                    meta.source_type = made_by(tool).to_string();
                }
                meta
            }
        };
        // Written only over the row listed above: a row another call wrote
        // since is newer than this call's view of it.
        match db.save_dataset_meta_unless_changed(&meta, listed).await {
            Ok(Some(_)) => outcome.saved.push(name.clone()),
            Ok(None) => skipped(tool, session_id, name),
            Err(e) => log_failure(tool, session_id, name, "saved", &e),
        }
    }

    // Datasets the call removed from the map; while a loader ran, only
    // rows a tool made (a loader's row may be its new dataset's).
    let mut removed: Vec<&DatasetMeta> = before
        .shapes
        .0
        .keys()
        .filter(|name| !live.contains_key(*name))
        .filter_map(|name| stored.get(name))
        .filter(|row| !overlapped || row.made_in_session())
        .collect();
    removed.sort_unstable_by(|a, b| a.name.cmp(&b.name));
    for row in removed {
        match db.delete_dataset_meta_unless_changed(row).await {
            Ok(true) => outcome.deleted.push(row.name.clone()),
            Ok(false) => skipped(tool, session_id, &row.name),
            Err(e) => log_failure(tool, session_id, &row.name, "deleted", &e),
        }
    }

    if !outcome.saved.is_empty() || !outcome.deleted.is_empty() {
        tracing::info!(
            tool,
            session_id,
            saved = outcome.saved.len(),
            deleted = outcome.deleted.len(),
            "[DATASET_SYNC] Dataset rows updated"
        );
    }
    outcome
}

fn skipped(tool: &str, session_id: &str, dataset: &str) {
    tracing::debug!(
        tool,
        session_id,
        dataset,
        "[DATASET_SYNC] A dataset row changed since it was read; left to the newer write"
    );
}

fn log_failure(tool: &str, session_id: &str, dataset: &str, what: &str, e: &DbError) {
    if matches!(e, DbError::QuotaExceeded(_)) {
        tracing::warn!(
            tool,
            session_id,
            dataset,
            error = %e,
            "[DATASET_SYNC] A dataset row was not {what}"
        );
    } else {
        tracing::error!(
            tool,
            session_id,
            dataset,
            error = %e,
            "[DATASET_SYNC] A dataset row was not {what}"
        );
    }
}

/// Store the dataset metadata of a successful `tool_name` call
/// (`load_dataset`, `upload_dataset` or `create_dataset`) in session
/// `session_id`: the dataset's name, dimensions and column names, read
/// from the call's `result_text`, and its source, read from the
/// `path` and `filename` fields of `arguments`, and the `na_strings` of a
/// `load_dataset` call, which a reload reads the file with. The work is linear in
/// `result_text`, so the tool route and the LLM tool loop both run it
/// on the blocking pool
/// (`routes_tools::capture_dataset_metadata_off_runtime`).
///
/// At info level a capture logs a fixed set of events with counts; the
/// result text (its first 2,000 bytes), the column lines and the column
/// names are at debug (the first line and the Dimensions line stay at
/// info), and column lines without a dtype are one warning with their
/// count. A save that fails is logged and nothing else happens: a
/// refusal by the session's `session_db_bytes` budget (`QuotaExceeded`)
/// at warn, any other failure at error.
pub(crate) async fn capture_loader_metadata(
    db: std::sync::Arc<DbConnection>,
    session_id: &str,
    tool_name: &str,
    arguments: &serde_json::Value,
    result_text: &str,
) {
    use std::path::PathBuf;

    #[cfg(test)]
    super::routes_tools::capture_threads::record(session_id);

    // Entry point: the tool, the session and the result's length.
    tracing::info!(
        tool = tool_name,
        session = session_id,
        result_len = result_text.len(),
        "[METADATA_CAPTURE] Starting metadata capture"
    );

    // The result text, cut at 2,000 bytes.
    tracing::debug!(
        result_text = %p2a_text::truncate_bytes(result_text, 2_000),
        result_len = result_text.len(),
        "[METADATA_CAPTURE] Full result text"
    );

    // Parse result to extract dataset info
    // Format: "Successfully loaded dataset 'name'\n\nDimensions: X rows x Y columns\n\nColumns:..."
    let first_line = result_text.lines().next();
    tracing::debug!(
        first_line = ?first_line,
        "[METADATA_CAPTURE] First line of result"
    );

    let dataset_name = first_line.and_then(|line| {
        let start = line.find('\'');
        let end = line.rfind('\'');
        tracing::debug!(
            line = %line,
            start_quote_pos = ?start,
            end_quote_pos = ?end,
            "[METADATA_CAPTURE] Parsing dataset name from first line"
        );
        match (start, end) {
            (Some(s), Some(e)) if s < e => {
                let name = line[s + 1..e].to_string();
                tracing::info!(
                    extracted_name = %name,
                    "[METADATA_CAPTURE] Extracted dataset name"
                );
                Some(name)
            }
            _ => {
                tracing::warn!(
                    "[METADATA_CAPTURE] Could not extract dataset name - quote positions invalid"
                );
                None
            }
        }
    });

    // Find dimensions line
    let dimensions_line = result_text
        .lines()
        .find(|line| line.trim().starts_with("Dimensions:"));
    tracing::info!(
        dimensions_line = ?dimensions_line,
        "[METADATA_CAPTURE] Found dimensions line"
    );

    let dimensions = dimensions_line.and_then(|line| {
        // "Dimensions: X rows x Y columns"
        let parts: Vec<&str> = line.split_whitespace().collect();
        tracing::debug!(
            parts = ?parts,
            parts_len = parts.len(),
            "[METADATA_CAPTURE] Dimensions line parts"
        );
        if parts.len() >= 5 {
            let rows = parts[1].parse::<i32>().ok();
            let cols = parts[4].parse::<i32>().ok();
            tracing::info!(
                rows = ?rows,
                cols = ?cols,
                "[METADATA_CAPTURE] Parsed dimensions"
            );
            match (rows, cols) {
                (Some(r), Some(c)) => Some((r, c)),
                _ => {
                    tracing::warn!("[METADATA_CAPTURE] Could not parse row/col numbers");
                    None
                }
            }
        } else {
            tracing::warn!(
                parts_len = parts.len(),
                "[METADATA_CAPTURE] Dimensions line has insufficient parts (need >= 5)"
            );
            None
        }
    });

    // Extract column names from "Columns:" section
    let columns_start = result_text.find("Columns:");
    tracing::info!(
        columns_start_pos = ?columns_start,
        "[METADATA_CAPTURE] Looking for 'Columns:' section"
    );

    let column_names: Vec<String> = if let Some(start) = columns_start {
        let columns_section = &result_text[start..];
        tracing::info!(
            total_lines_in_section = columns_section.lines().count(),
            "[METADATA_CAPTURE] Lines in Columns section"
        );
        tracing::debug!(
            first_5_lines = ?columns_section.lines().take(5).collect::<Vec<_>>(),
            "[METADATA_CAPTURE] First lines of the Columns section"
        );

        // Column lines with no '(' before their dtype, logged once.
        let mut unparsed_lines = 0usize;
        let parsed_columns: Vec<String> = columns_section
            .lines()
            .skip(1) // Skip "Columns:" line
            .enumerate()
            .filter_map(|(idx, line)| {
                let trimmed_line = line.trim();
                let starts_with_dash = trimmed_line.starts_with('-');

                if !starts_with_dash {
                    if !trimmed_line.is_empty() {
                        tracing::debug!(
                            line_idx = idx,
                            line = %line,
                            trimmed = %trimmed_line,
                            "[METADATA_CAPTURE] Skipping line (doesn't start with '-')"
                        );
                    }
                    return None;
                }

                // "  - column_name (dtype): X nulls (Y%)"
                let after_dash = trimmed_line.trim_start_matches('-').trim();
                let paren_pos = after_dash.find('(');

                tracing::debug!(
                    line_idx = idx,
                    original_line = %line,
                    after_dash = %after_dash,
                    paren_pos = ?paren_pos,
                    "[METADATA_CAPTURE] Parsing column line"
                );

                match paren_pos {
                    Some(pos) => {
                        let col_name = after_dash[..pos].trim().to_string();
                        tracing::debug!(
                            extracted_column = %col_name,
                            "[METADATA_CAPTURE] Extracted column name"
                        );
                        Some(col_name)
                    }
                    None => {
                        unparsed_lines += 1;
                        tracing::debug!(
                            line = %line,
                            "[METADATA_CAPTURE] Column line has no '(' - cannot extract name"
                        );
                        None
                    }
                }
            })
            .collect();
        if unparsed_lines > 0 {
            tracing::warn!(
                unparsed_lines,
                "[METADATA_CAPTURE] Column line has no '(' - cannot extract name"
            );
        }

        tracing::info!(
            num_columns_parsed = parsed_columns.len(),
            "[METADATA_CAPTURE] Finished parsing columns"
        );
        tracing::debug!(
            columns = ?parsed_columns,
            "[METADATA_CAPTURE] Parsed column names"
        );
        parsed_columns
    } else {
        tracing::warn!("[METADATA_CAPTURE] No 'Columns:' section found in result");
        Vec::new()
    };

    // The file the dataset was loaded from, which a reload loads again.
    // Only load_dataset reads a `path`; upload_dataset and create_dataset
    // ignore one a caller adds, so it is not their source.
    let source_path = arguments
        .get("path")
        .filter(|_| tool_name == "load_dataset")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let file_size_bytes = source_path
        .as_deref()
        .and_then(|p| jailed_file_size(session_id, p));

    // The missing-value markers load_dataset read the file with, which a reload
    // reads it with again (the call succeeded, so they passed the request limits).
    let na_strings: Option<Vec<String>> = arguments
        .get("na_strings")
        .filter(|_| tool_name == "load_dataset")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        });

    // Determine source type
    let source_type = if tool_name == "upload_dataset" {
        arguments
            .get("filename")
            .and_then(|v| v.as_str())
            .map(PathBuf::from)
            .and_then(|p| p.extension().map(|e| e.to_string_lossy().to_string()))
            .unwrap_or_else(|| "unknown".to_string())
    } else if tool_name == "create_dataset" {
        "inline_csv".to_string()
    } else {
        source_path
            .as_ref()
            .map(PathBuf::from)
            .and_then(|p| p.extension().map(|e| e.to_string_lossy().to_string()))
            .unwrap_or_else(|| "unknown".to_string())
    };

    tracing::info!(
        dataset_name = ?dataset_name,
        dimensions = ?dimensions,
        num_columns = column_names.len(),
        source_type = %source_type,
        source_path = ?source_path,
        "[METADATA_CAPTURE] Final parsed metadata summary"
    );

    // Build and save metadata
    match (dataset_name.as_ref(), dimensions.as_ref()) {
        (Some(name), Some((rows, cols))) => {
            let mut meta = DatasetMeta::new(
                session_id.to_string(),
                name.clone(),
                source_type,
                *rows,
                *cols,
                column_names,
            );
            meta.source_path = source_path;
            meta.file_size_bytes = file_size_bytes;
            meta.na_strings = na_strings;

            tracing::info!(
                meta_id = %meta.id_string(),
                meta_session_id = %meta.session_id,
                meta_name = %meta.name,
                meta_source_type = %meta.source_type,
                meta_row_count = meta.row_count,
                meta_column_count = meta.column_count,
                meta_column_names_len = meta.column_names.len(),
                "[METADATA_CAPTURE] DatasetMeta object before save"
            );

            match db.save_dataset_meta(&meta).await {
                Ok(saved_meta) => {
                    tracing::info!(
                        dataset = %name,
                        session_id = %session_id,
                        saved_id = %saved_meta.id_string(),
                        saved_column_names_len = saved_meta.column_names.len(),
                        "[METADATA_CAPTURE] Successfully saved dataset metadata"
                    );
                }
                // The session's budget is full: an expected refusal.
                Err(e @ crate::db::DbError::QuotaExceeded(_)) => {
                    tracing::warn!(
                        dataset = %name,
                        session_id = %session_id,
                        error = %e,
                        "[METADATA_CAPTURE] Failed to save dataset metadata"
                    );
                }
                Err(e) => {
                    tracing::error!(
                        dataset = %name,
                        session_id = %session_id,
                        error = %e,
                        "[METADATA_CAPTURE] Failed to save dataset metadata"
                    );
                }
            }
        }
        _ => {
            tracing::warn!(
                dataset_name = ?dataset_name,
                dimensions = ?dimensions,
                "[METADATA_CAPTURE] Could not parse dataset info from result - missing name or dimensions"
            );
        }
    }
}

/// The size of a dataset's source file for its stored metadata, read only
/// through the path jail inside the session's scope: `None` for a path
/// outside the data root or a file that cannot be read. Synchronous, so the
/// scope is never held across an `.await`.
fn jailed_file_size(session_id: &str, path: &str) -> Option<i64> {
    let _scope = crate::path_jail::enter_session_scope(session_id);
    let checked = crate::path_jail::validate_data_path(path).ok()?;
    let len = std::fs::metadata(checked).ok()?.len();
    Some(i64::try_from(len).unwrap_or(i64::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn frame(rows: usize, columns: &[&str]) -> Dataset {
        let cols: Vec<Column> = columns
            .iter()
            .enumerate()
            .map(|(j, name)| {
                let values: Vec<f64> = (0..rows).map(|i| i as f64 * 1.1 + j as f64).collect();
                Column::new((*name).into(), values)
            })
            .collect();
        Dataset::new(DataFrame::new(cols).unwrap())
    }

    async fn db() -> DbConnection {
        DbConnection::connect_memory().await.unwrap()
    }

    async fn rows(db: &DbConnection, sid: &str) -> HashMap<String, DatasetMeta> {
        db.get_datasets_for_session(sid)
            .await
            .unwrap()
            .into_iter()
            .map(|r| (r.name.clone(), r))
            .collect()
    }

    /// A dataset a tool adds gets a row with its counts and names and no
    /// source; a call that changes nothing writes nothing.
    #[tokio::test]
    async fn a_new_dataset_gets_a_row_and_an_unchanged_map_writes_none() {
        let db = db().await;
        let session = Session::new("s-sync".to_string(), None);
        let sid = session.id.clone();
        let before = DatasetSnapshot::of(&session, "generate_random_data").await;
        session
            .datasets
            .write()
            .await
            .insert("g".into(), frame(7, &["x", "y"]));
        let out =
            sync_dataset_meta(&db, &session, &sid, "generate_random_data", &before, false).await;
        assert_eq!(out.saved, ["g"]);
        let stored = rows(&db, &sid).await;
        let g = &stored["g"];
        assert_eq!((g.row_count, g.column_count), (7, 2));
        assert_eq!(g.column_names, ["x", "y"]);
        assert_eq!(g.source_path, None);
        assert_eq!(g.source_type, SOURCE_TYPE_GENERATED);
        assert!(g.made_in_session());

        let before = DatasetSnapshot::of(&session, "ols_regression").await;
        let out = sync_dataset_meta(&db, &session, &sid, "ols_regression", &before, false).await;
        assert_eq!(out, SyncOutcome::default());
    }

    /// A loaded dataset a tool changes in place loses its source; one a
    /// loader wrote keeps it, its counts corrected from the map.
    #[tokio::test]
    async fn a_changed_loaded_dataset_loses_its_source_unless_a_loader_changed_it() {
        let db = db().await;
        let session = Session::new("s-sync".to_string(), None);
        let sid = session.id.clone();
        for (name, loader, kept) in [("a", true, true), ("b", false, false)] {
            let mut meta = DatasetMeta::new(
                sid.clone(),
                name.into(),
                "csv".into(),
                9,
                2,
                vec!["x".into(), "y".into()],
            );
            meta.source_path = Some(format!("{name}.csv"));
            meta.na_strings = Some(vec!["-99".into()]);
            db.save_dataset_meta(&meta).await.unwrap();
            let tool = if loader {
                "load_dataset"
            } else {
                "munge_filter"
            };
            let before = DatasetSnapshot::of(&session, tool).await;
            session
                .datasets
                .write()
                .await
                .insert(name.into(), frame(4, &["x", "y (cm)"]));
            let out = sync_dataset_meta(&db, &session, &sid, tool, &before, loader).await;
            assert_eq!(out.saved, [name], "{name}");
            let row = &rows(&db, &sid).await[name];
            assert_eq!((row.row_count, row.column_count), (4, 2), "{name}");
            assert_eq!(row.column_names, ["x", "y (cm)"], "{name}");
            assert_eq!(row.source_path.is_some(), kept, "{name}");
            assert_eq!(row.na_strings.is_some(), kept, "{name}");
            assert_eq!(row.made_in_session(), !kept, "{name}");
        }
    }

    /// A dataset the call removes loses its row; one that was never in the
    /// map before the call (a stored row awaiting a reload) keeps it.
    #[tokio::test]
    async fn a_removed_dataset_loses_its_row_and_a_stored_one_stays() {
        let db = db().await;
        let session = Session::new("s-sync".to_string(), None);
        let sid = session.id.clone();
        let stored_only = DatasetMeta::new(sid.clone(), "old".into(), "csv".into(), 3, 1, vec![]);
        db.save_dataset_meta(&stored_only).await.unwrap();
        let before = DatasetSnapshot::of(&session, "munge_select").await;
        session
            .datasets
            .write()
            .await
            .insert("d".into(), frame(5, &["x"]));
        sync_dataset_meta(&db, &session, &sid, "munge_select", &before, false).await;

        let before = DatasetSnapshot::of(&session, "session_import").await;
        session.datasets.write().await.clear();
        let out = sync_dataset_meta(&db, &session, &sid, "session_import", &before, false).await;
        assert_eq!(out.deleted, ["d"]);
        let stored = rows(&db, &sid).await;
        assert!(!stored.contains_key("d"));
        assert!(stored.contains_key("old"));
    }

    /// The race of two calls in one session: a call that started before a
    /// loader put its dataset in the map, and syncs after the loader stored
    /// the dataset's row, leaves the loader's source alone; it did not
    /// change the dataset. The loader's row is written from its result text,
    /// whose column names can differ from the dataset's (a name with a
    /// parenthesis), so the other call sees a row that differs from the map.
    #[tokio::test]
    async fn a_call_overlapping_a_loader_keeps_the_loaded_source() {
        let db = db().await;
        let session = Session::new("s-sync".to_string(), None);
        let sid = session.id.clone();
        // B, any other tool, takes its snapshot first.
        let b_before = DatasetSnapshot::of(&session, "ols_regression").await;
        // A, load_dataset, starts, loads the file and stores its row.
        let a_before = DatasetSnapshot::of(&session, "load_dataset").await;
        session
            .datasets
            .write()
            .await
            .insert("sales".into(), frame(4, &["x", "y (cm)"]));
        let mut meta = DatasetMeta::new(
            sid.clone(),
            "sales".into(),
            "csv".into(),
            4,
            2,
            vec!["x".into(), "y".into()],
        );
        meta.source_path = Some("sales.csv".into());
        db.save_dataset_meta(&meta).await.unwrap();
        // B's sync runs before A's.
        sync_dataset_meta(&db, &session, &sid, "ols_regression", &b_before, false).await;
        let row = &rows(&db, &sid).await["sales"];
        assert_eq!(row.source_path.as_deref(), Some("sales.csv"));
        assert_eq!(row.source_type, "csv");
        assert!(!row.made_in_session());
        sync_dataset_meta(&db, &session, &sid, "load_dataset", &a_before, true).await;
        drop((a_before, b_before));
        let row = &rows(&db, &sid).await["sales"];
        assert_eq!(row.source_path.as_deref(), Some("sales.csv"));
        assert_eq!(row.column_names, ["x", "y (cm)"]);

        // Once the loader has finished, a tool that changes the dataset in
        // place still takes its source away.
        let before = DatasetSnapshot::of(&session, "munge_filter").await;
        session
            .datasets
            .write()
            .await
            .insert("sales".into(), frame(2, &["x", "y (cm)"]));
        let out = sync_dataset_meta(&db, &session, &sid, "munge_filter", &before, false).await;
        assert_eq!(out.saved, ["sales"]);
        let row = &rows(&db, &sid).await["sales"];
        assert_eq!(row.source_path, None);
        assert_eq!(row.source_type, SOURCE_TYPE_DERIVED);
    }

    /// The same when the loader's call started first: it was running when
    /// the other call took its snapshot.
    #[tokio::test]
    async fn a_call_started_while_a_loader_runs_keeps_the_loaded_source() {
        let db = db().await;
        let session = Session::new("s-sync".to_string(), None);
        let sid = session.id.clone();
        let a_before = DatasetSnapshot::of(&session, "load_dataset").await;
        let b_before = DatasetSnapshot::of(&session, "describe_dataset").await;
        session
            .datasets
            .write()
            .await
            .insert("sales".into(), frame(4, &["x", "y (cm)"]));
        let mut meta = DatasetMeta::new(
            sid.clone(),
            "sales".into(),
            "csv".into(),
            4,
            2,
            vec!["x".into(), "y".into()],
        );
        meta.source_path = Some("sales.csv".into());
        db.save_dataset_meta(&meta).await.unwrap();
        // B's sync runs before A's own.
        sync_dataset_meta(&db, &session, &sid, "describe_dataset", &b_before, false).await;
        let row = &rows(&db, &sid).await["sales"];
        assert_eq!(row.source_path.as_deref(), Some("sales.csv"));
        assert!(!row.made_in_session());
        sync_dataset_meta(&db, &session, &sid, "load_dataset", &a_before, true).await;
    }

    /// The conditional writes the sync makes: a save or delete over a row
    /// that changed since it was read writes nothing.
    #[tokio::test]
    async fn a_row_changed_since_it_was_read_is_left_to_the_newer_write() {
        let db = db().await;
        let sid = "s-sync".to_string();
        let mut loaded = DatasetMeta::new(
            sid.clone(),
            "d".into(),
            "csv".into(),
            4,
            1,
            vec!["x".into()],
        );
        loaded.source_path = Some("d.csv".into());
        let derived = DatasetMeta::new(
            sid.clone(),
            "d".into(),
            SOURCE_TYPE_DERIVED.into(),
            2,
            1,
            vec!["x".into()],
        );
        // Read as absent, then written by another call: no save.
        db.save_dataset_meta(&loaded).await.unwrap();
        assert!(
            db.save_dataset_meta_unless_changed(&derived, None)
                .await
                .unwrap()
                .is_none()
        );
        let row = rows(&db, &sid).await["d"].clone();
        assert_eq!(row.source_path.as_deref(), Some("d.csv"));
        // Read, then rewritten by another call: no save, no delete.
        let mut newer = row.clone();
        newer.row_count = 5;
        db.save_dataset_meta(&newer).await.unwrap();
        assert!(
            db.save_dataset_meta_unless_changed(&derived, Some(&row))
                .await
                .unwrap()
                .is_none()
        );
        assert!(!db.delete_dataset_meta_unless_changed(&row).await.unwrap());
        assert_eq!(rows(&db, &sid).await["d"].row_count, 5);
        // Unchanged since it was read: written, then deleted.
        let current = rows(&db, &sid).await["d"].clone();
        let mut update = current.clone();
        update.row_count = 3;
        assert!(
            db.save_dataset_meta_unless_changed(&update, Some(&current))
                .await
                .unwrap()
                .is_some()
        );
        let current = rows(&db, &sid).await["d"].clone();
        assert_eq!(current.row_count, 3);
        assert!(
            db.delete_dataset_meta_unless_changed(&current)
                .await
                .unwrap()
        );
        assert!(rows(&db, &sid).await.is_empty());
    }

    #[test]
    fn metadata_file_size_goes_through_the_jail() {
        assert_eq!(
            jailed_file_size("00000000-0000-4000-8000-000000000000", "/etc/passwd"),
            None
        );
    }
}
