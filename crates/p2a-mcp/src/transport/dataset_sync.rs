//! The session's dataset list in the store, kept in step with its dataset
//! map after every tool call.
//!
//! `GET /api/sessions/{id}/datasets` (the dataset sidebar of the web app)
//! lists the session's dataset metadata rows. The rows of the loaders
//! (`load_dataset`, `upload_dataset`, `create_dataset`) are written from
//! their result text, with the source a reload reads
//! (`routes_tools::capture_dataset_metadata_off_runtime`). Every other tool
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
//! did costs one listing query and a write per changed row. No frame is
//! serialised. Datasets that leave the map for other reasons (a server
//! restart, eviction) keep their rows, which a reload reads.
//!
//! What the sync does not see: a tool that replaces a loaded dataset with
//! one of the same shape and column names leaves its row, and its source,
//! as they were.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

use p2a_core::data::Dataset;

use crate::db::{DatasetMeta, DbConnection, DbError, SOURCE_TYPE_DERIVED, SOURCE_TYPE_GENERATED};
use crate::session::Session;

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

/// The shapes of a session's datasets by name, taken before a tool call so
/// that [`sync_dataset_meta`] can tell what the call changed.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct DatasetSnapshot(HashMap<String, Shape>);

impl DatasetSnapshot {
    /// The shapes of `session`'s datasets now.
    pub(crate) async fn of(session: &Session) -> Self {
        let datasets = session.datasets.read().await;
        Self(
            datasets
                .iter()
                .map(|(name, dataset)| (name.clone(), Shape::of(dataset)))
                .collect(),
        )
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
    let after = DatasetSnapshot::of(session).await;
    if after == *before && !loader {
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

    let mut names: Vec<&String> = live.keys().collect();
    names.sort_unstable();
    for name in names {
        let live = &live[name];
        // A dataset this call made or changed, other than a loader's own.
        let changed_by_tool = !loader && before.0.get(name) != Some(&live.shape);
        let meta = match stored.get(name) {
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
                if changed_by_tool {
                    // No file holds the dataset any more.
                    meta.source_path = None;
                    meta.file_size_bytes = None;
                    meta.na_strings = None;
                    meta.source_type = made_by(tool).to_string();
                }
                meta
            }
        };
        match db.save_dataset_meta(&meta).await {
            Ok(_) => outcome.saved.push(name.clone()),
            Err(e) => log_failure(tool, session_id, name, "saved", &e),
        }
    }

    // Datasets the call removed from the map.
    let mut removed: Vec<&String> = before
        .0
        .keys()
        .filter(|name| !live.contains_key(*name) && stored.contains_key(*name))
        .collect();
    removed.sort_unstable();
    for name in removed {
        match db.delete_dataset_meta(session_id, name).await {
            Ok(()) => outcome.deleted.push(name.clone()),
            Err(e) => log_failure(tool, session_id, name, "deleted", &e),
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
        let before = DatasetSnapshot::of(&session).await;
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

        let before = DatasetSnapshot::of(&session).await;
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
            let before = DatasetSnapshot::of(&session).await;
            session
                .datasets
                .write()
                .await
                .insert(name.into(), frame(4, &["x", "y (cm)"]));
            let tool = if loader {
                "load_dataset"
            } else {
                "munge_filter"
            };
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
        let before = DatasetSnapshot::of(&session).await;
        session
            .datasets
            .write()
            .await
            .insert("d".into(), frame(5, &["x"]));
        sync_dataset_meta(&db, &session, &sid, "munge_select", &before, false).await;

        let before = DatasetSnapshot::of(&session).await;
        session.datasets.write().await.clear();
        let out = sync_dataset_meta(&db, &session, &sid, "session_import", &before, false).await;
        assert_eq!(out.deleted, ["d"]);
        let stored = rows(&db, &sid).await;
        assert!(!stored.contains_key("d"));
        assert!(stored.contains_key("old"));
    }
}
