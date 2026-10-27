//! What saving and listing a dataset's metadata logs at info level and at
//! debug level.
//!
//! This is a test binary of its own, with one test, because `tracing`
//! caches each callsite's interest for the whole process when a thread first
//! reaches it, and while just one dispatcher is registered it takes that
//! interest from the reaching thread's default dispatcher alone. In the
//! library's test binary, other tests save and list dataset metadata on their
//! own threads with no subscriber; a callsite one of them reached first stayed
//! disabled for a subscriber set as this test's thread default, so the test
//! failed now and then. Here the one subscriber is the global default before
//! any callsite of the crate is reached, and every thread sees it.
//!
//! Run with `cargo test -p p2a-mcp --features full --test dataset_meta_logs`.
//! CI does not: its feature set for p2a-mcp leaves out `db`, which compiles
//! this file to an empty test binary.
#![cfg(feature = "db")]

use std::sync::{Arc, Mutex};

use p2a_mcp::db::{DatasetMeta, DbConnection};

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

/// Saving a dataset's metadata twice (a create, then an update of the same
/// row) and listing its session logs the counts of its column names at info
/// level, and the names themselves only at debug: a wide dataset's names
/// are up to `MAX_STORED_COLUMN_NAME_BYTES` on one line.
#[tokio::test]
async fn dataset_meta_logs_its_column_names_only_at_debug() {
    let logs = Logs::default();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(logs.clone())
        .with_env_filter(tracing_subscriber::EnvFilter::new("p2a_mcp=debug"))
        .with_ansi(false)
        .without_time()
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("this test binary sets the only subscriber");

    let db = DbConnection::connect_memory().await.unwrap();
    let names: Vec<String> = (0..40).map(|i| format!("logcol{i:03}_q")).collect();
    let meta = DatasetMeta::new("s1".into(), "logged".into(), "csv".into(), 3, 40, names);
    db.save_dataset_meta(&meta).await.unwrap();
    db.save_dataset_meta(&meta).await.unwrap();
    assert_eq!(db.get_datasets_for_session("s1").await.unwrap().len(), 1);

    let lines = logs.lines();
    let at = |level: &str| -> Vec<&String> {
        lines
            .iter()
            .filter(|line| line.trim_start().starts_with(level))
            .collect()
    };
    let (info, debug) = (at("INFO "), at("DEBUG "));
    for message in [
        "[DB_SAVE_META] Starting save_dataset_meta",
        "[DB_SAVE_META] CREATE returned successfully",
        "[DB_SAVE_META] Found existing record, will update",
        "[DB_SAVE_META] Content being sent to UPDATE",
        "[DB_SAVE_META] UPDATE returned successfully",
        "[DB_GET_DATASETS] Dataset record from SurrealDB",
    ] {
        assert!(
            info.iter().any(|line| line.contains(message)),
            "no {message:?} on an info line:\n{}",
            lines.join("\n")
        );
    }
    let named: Vec<_> = info.iter().filter(|line| line.contains("logcol")).collect();
    assert!(named.is_empty(), "column names on info lines: {named:#?}");
    assert!(
        info.iter().any(
            |line| line.contains("[DB_SAVE_META] Starting save_dataset_meta")
                && line.contains("column_names_len=40")
                && line.contains("given_column_names_len=40")
        ),
        "no counts on the info line of the save"
    );
    assert!(
        debug.iter().any(|line| line.contains("logcol039_q")),
        "no column names on a debug line"
    );
}
