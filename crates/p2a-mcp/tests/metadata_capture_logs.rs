//! What the dataset-metadata capture after a `load_dataset`,
//! `upload_dataset` or `create_dataset` call logs, and at which level.
//!
//! This is a test binary of its own, with one test, for the reason given in
//! `dataset_meta_logs.rs`: `tracing` caches each callsite's interest for the
//! whole process when a thread first reaches it, so a subscriber set as one
//! test's thread default can miss events whose callsite another test reached
//! first. Here the one subscriber is the global default before any callsite
//! of the crate is reached, and every thread sees it.
//!
//! Run with `cargo test -p p2a-mcp --features full --test
//! metadata_capture_logs`. CI does not: its feature set for p2a-mcp leaves
//! out `db`, which compiles this file to an empty test binary.
#![cfg(all(feature = "db", feature = "llm", feature = "http"))]

use std::sync::{Arc, Mutex};

use p2a_mcp::db::{DbConnection, PersistenceLimits};
use p2a_mcp::transport::llm_routes::StreamingToolExecutor;

/// What the global subscriber writes.
#[derive(Clone, Default)]
struct Logs(Arc<Mutex<Vec<u8>>>);

impl Logs {
    /// The lines written so far.
    fn lines(&self) -> Vec<String> {
        String::from_utf8_lossy(&self.0.lock().unwrap())
            .lines()
            .map(String::from)
            .collect()
    }
}

impl std::io::Write for Logs {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Logs {
    type Writer = Logs;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

/// The text of a successful `create_dataset` call for dataset `name` with
/// `width` columns `wcol00000_q`, ..., followed by `malformed` column lines
/// that have no dtype in parentheses.
fn created(name: &str, width: usize, malformed: usize) -> String {
    let columns: Vec<String> = (0..width)
        .map(|i| format!("  - wcol{i:05}_q (f64)"))
        .chain((0..malformed).map(|i| format!("  - broken line {i}")))
        .collect();
    format!(
        "Successfully created dataset '{name}'\n\nDimensions: 2 rows x {width} columns\n\n\
         Columns:\n{}",
        columns.join("\n")
    )
}

/// The levels as the fmt layer writes them, after the padding.
const LEVELS: [&str; 5] = ["TRACE ", "DEBUG ", "INFO ", "WARN ", "ERROR "];

/// The level `line` starts with, if it starts with one.
fn level_of(line: &str) -> Option<&'static str> {
    let line = line.trim_start();
    LEVELS.into_iter().find(|level| line.starts_with(level))
}

/// The events among `lines`, with their level: each is a line that starts
/// with a level and the lines after it that do not. The fmt layer writes a
/// field value that holds newlines (the result text, for one) on lines of
/// their own, with no level in front. Lines before the first level are left
/// out.
fn events(lines: &[String]) -> Vec<(&'static str, &[String])> {
    let heads: Vec<(usize, &'static str)> = lines
        .iter()
        .enumerate()
        .filter_map(|(i, line)| level_of(line).map(|level| (i, level)))
        .collect();
    heads
        .iter()
        .enumerate()
        .map(|(k, &(start, level))| {
            let end = heads.get(k + 1).map_or(lines.len(), |&(next, _)| next);
            (level, &lines[start..end])
        })
        .collect()
}

/// Every line of the events at `level` ("INFO ", "WARN ", ...) among
/// `lines`, the lines that continue an event included.
fn at<'a>(lines: &'a [String], level: &str) -> Vec<&'a String> {
    events(lines)
        .into_iter()
        .filter(|&(event_level, _)| event_level == level)
        .flat_map(|(_, event)| event)
        .collect()
}

/// Every line of the `[METADATA_CAPTURE]` events at `level` among `lines`.
fn capture_lines<'a>(lines: &'a [String], level: &str) -> Vec<&'a String> {
    events(lines)
        .into_iter()
        .filter(|&(event_level, event)| {
            event_level == level && event[0].contains("[METADATA_CAPTURE]")
        })
        .flat_map(|(_, event)| event)
        .collect()
}

/// A capture logs the same number of info lines for a dataset of 3 columns
/// and one of 300, with counts and no column name on them (the names are on
/// a debug line), and reads nothing back after a save. A line counts at the
/// level of the event it belongs to, so a multi-line value at info, such as
/// the result text, shows here. Column lines it cannot read are one warning
/// with their count. A save the session's `session_db_bytes` budget refuses
/// is a warning, not an error.
#[tokio::test]
async fn the_capture_logs_counts_at_info_and_a_refused_save_as_a_warning() {
    let logs = Logs::default();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(logs.clone())
        .with_env_filter(tracing_subscriber::EnvFilter::new("p2a_mcp=debug"))
        .with_ansi(false)
        .without_time()
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("this test binary sets the only subscriber");
    let none = serde_json::json!({"path": null, "filename": null});

    let db = Arc::new(DbConnection::connect_memory().await.unwrap());
    let capture = |session: &'static str, text: String| {
        let db = db.clone();
        let none = none.clone();
        async move {
            StreamingToolExecutor::capture_dataset_metadata_static(
                db,
                session,
                "create_dataset",
                &none,
                &text,
            )
            .await;
        }
    };

    let mark = logs.lines().len();
    capture("narrow", created("narrow", 3, 0)).await;
    let narrow = logs.lines()[mark..].to_vec();
    let mark = logs.lines().len();
    capture("wide", created("wide", 300, 0)).await;
    let wide = logs.lines()[mark..].to_vec();

    for (session, width) in [("narrow", 3), ("wide", 300)] {
        let rows = db.get_datasets_for_session(session).await.unwrap();
        assert_eq!(rows.len(), 1, "{session}: no metadata row");
        assert_eq!(rows[0].column_names.len(), width, "{session}");
    }
    let (narrow_info, wide_info) = (
        capture_lines(&narrow, "INFO "),
        capture_lines(&wide, "INFO "),
    );
    assert_eq!(
        narrow_info.len(),
        wide_info.len(),
        "info lines grow with the width:\n{}",
        wide_info
            .iter()
            .map(|line| line.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    );
    // Every line of every info event, the lines that continue one included.
    let named: Vec<_> = at(&wide, "INFO ")
        .into_iter()
        .filter(|line| line.contains("wcol"))
        .collect();
    assert!(named.is_empty(), "column names on info lines: {named:#?}");
    assert!(
        wide_info.iter().any(
            |line| line.contains("[METADATA_CAPTURE] Finished parsing columns")
                && line.contains("num_columns_parsed=300")
        ),
        "no count on the info line of the parse"
    );
    assert!(
        at(&wide, "DEBUG ")
            .iter()
            .any(|line| line.contains("wcol00299_q")),
        "no column names on a debug line"
    );
    let read_back: Vec<_> = narrow
        .iter()
        .chain(&wide)
        .filter(|line| line.contains("read back"))
        .collect();
    assert!(
        read_back.is_empty(),
        "the save was read back: {read_back:#?}"
    );

    // Column lines without a dtype: one warning, with their count.
    let mark = logs.lines().len();
    capture("malformed", created("malformed", 2, 50)).await;
    let malformed = logs.lines()[mark..].to_vec();
    let unreadable: Vec<_> = capture_lines(&malformed, "WARN ")
        .into_iter()
        .filter(|line| line.contains("Column line has no '('"))
        .collect();
    assert_eq!(unreadable.len(), 1, "{unreadable:#?}");
    assert!(
        unreadable[0].contains("unparsed_lines=50"),
        "{}",
        unreadable[0]
    );
    let rows = db.get_datasets_for_session("malformed").await.unwrap();
    assert_eq!(rows[0].column_names, ["wcol00000_q", "wcol00001_q"]);

    // A full session budget refuses the row: a warning, no error.
    let full = Arc::new(DbConnection::connect_memory().await.unwrap().with_limits(
        PersistenceLimits {
            session_db_bytes: Some(64),
            ..Default::default()
        },
    ));
    let mark = logs.lines().len();
    StreamingToolExecutor::capture_dataset_metadata_static(
        full.clone(),
        "refused",
        "create_dataset",
        &none,
        &created("refused", 3, 0),
    )
    .await;
    let refused = logs.lines()[mark..].to_vec();
    assert!(
        full.get_datasets_for_session("refused")
            .await
            .unwrap()
            .is_empty(),
        "the budget did not refuse the row"
    );
    assert!(
        capture_lines(&refused, "WARN ")
            .iter()
            .any(|line| line.contains("[METADATA_CAPTURE] Failed to save dataset metadata")),
        "no warning for the refused save:\n{}",
        refused.join("\n")
    );
    let errors = capture_lines(&refused, "ERROR ");
    assert!(
        errors.is_empty(),
        "the refused save is an error: {errors:#?}"
    );
}
