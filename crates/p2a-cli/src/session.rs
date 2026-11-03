//! Session recording and management for reproducibility

use crate::memory::{MemoryTracker, estimate_dataset_memory};
use chrono::{DateTime, Utc};
use p2a_core::{DataLoader, Dataset};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

/// Metadata about a loaded dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMeta {
    /// Dataset name/identifier
    pub name: String,
    /// Original file path
    pub source_path: PathBuf,
    /// File format (csv, parquet, excel, stata, sas)
    pub format: String,
    /// Number of rows
    pub nrows: usize,
    /// Column names
    pub columns: Vec<String>,
    /// When the dataset was loaded
    pub loaded_at: DateTime<Utc>,
    /// Strings read as missing values when the CSV file was loaded (`data load
    /// --na-strings`), used again on every reload; `None` means the loader's default
    /// (`NA` and the empty field). Absent in older session files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub na_strings: Option<Vec<String>>,
}

/// A recorded command execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandRecord {
    /// Unique identifier for this command
    pub id: String,
    /// Timestamp of execution
    pub timestamp: DateTime<Utc>,
    /// The full command line (for display/comments)
    pub command_line: String,
    /// Command category (data, reg, panel, etc.)
    pub category: String,
    /// Subcommand name (load, ols, fe, etc.)
    pub subcommand: String,
    /// Structured arguments for reconstruction
    pub arguments: serde_json::Value,
    /// The command-portion argv (category, subcommand, and args) as actually
    /// invoked: without the program name, without the global flags typed before
    /// the category and without `--session`; global flags typed later are kept.
    /// Preferred over `arguments` for faithful, injection-safe script export.
    #[serde(default)]
    pub argv: Vec<String>,
    /// Referenced datasets by name
    pub dataset_refs: Vec<String>,
    /// Output file paths (for viz commands)
    pub output_files: Vec<PathBuf>,
    /// Whether the command succeeded
    pub success: bool,
    /// Optional error message if failed
    pub error: Option<String>,
    /// Execution duration in milliseconds
    pub duration_ms: u64,
}

/// A complete analysis session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Session identifier
    pub id: String,
    /// Session title/description
    pub title: Option<String>,
    /// When the session started
    pub created_at: DateTime<Utc>,
    /// When the session was last modified
    pub updated_at: DateTime<Utc>,
    /// Working directory when session was created
    pub working_dir: PathBuf,
    /// Datasets loaded during this session
    pub datasets: HashMap<String, DatasetMeta>,
    /// Chronological list of executed commands
    pub commands: Vec<CommandRecord>,
    /// p2a version used
    pub version: String,
}

impl Session {
    /// Create a new session
    pub fn new() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            title: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            working_dir: std::env::current_dir().unwrap_or_default(),
            datasets: HashMap::new(),
            commands: Vec::new(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Load an existing session from a file
    pub fn load(path: &PathBuf) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let session: Session = serde_json::from_str(&content)?;
        Ok(session)
    }

    /// Save the session to a file
    pub fn save(&self, path: &PathBuf) -> anyhow::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Record a command execution
    pub fn record_command(&mut self, record: CommandRecord) {
        self.updated_at = Utc::now();
        self.commands.push(record);
    }

    /// Register a loaded dataset
    pub fn register_dataset(&mut self, meta: DatasetMeta) {
        self.updated_at = Utc::now();
        self.datasets.insert(meta.name.clone(), meta);
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

/// Reload a dataset from its source file, with the missing-value strings it was
/// loaded with
fn reload_dataset(meta: &DatasetMeta) -> anyhow::Result<Dataset> {
    let df = match meta.format.as_str() {
        "csv" => crate::commands::data::load_csv_with_markers(
            &meta.source_path,
            meta.na_strings.as_deref(),
        )?,
        "parquet" => DataLoader::load_parquet(&meta.source_path)?,
        "xlsx" | "xls" | "xlsb" => DataLoader::load_excel(&meta.source_path, None)?,
        "dta" => DataLoader::load_stata(&meta.source_path)?,
        "sas7bdat" => DataLoader::load_sas(&meta.source_path)?,
        fmt => anyhow::bail!("Unsupported format for reload: {}", fmt),
    };
    Ok(Dataset::new(df))
}

/// Manages session state during CLI execution
pub struct SessionManager {
    /// Path to the session file
    path: PathBuf,
    /// The session being recorded
    session: Session,
    /// In-memory dataset store (actual data)
    datasets: HashMap<String, Dataset>,
    /// Timer for command duration tracking
    command_start: Option<Instant>,
    /// Estimated memory of the datasets in `datasets`, against `--max-memory`
    memory: MemoryTracker,
    /// The estimate counted for each dataset name (a replaced dataset is counted once)
    memory_by_name: HashMap<String, usize>,
    /// The `--max-memory` value in MiB (0 = no limit)
    max_memory_mb: usize,
    /// Whether this invocation has printed its memory warning
    memory_warned: bool,
}

impl SessionManager {
    /// Create a new session manager, loading existing session if file exists
    pub fn new(path: PathBuf) -> anyhow::Result<Self> {
        let (session, datasets) = if path.exists() {
            let session = Session::load(&path)?;

            // Reload datasets from their original source files
            let mut datasets = HashMap::new();
            for (name, meta) in &session.datasets {
                match reload_dataset(meta) {
                    Ok(ds) => {
                        datasets.insert(name.clone(), ds);
                    }
                    Err(e) => {
                        eprintln!("Warning: Could not reload dataset '{}': {}", name, e);
                    }
                }
            }
            (session, datasets)
        } else {
            (Session::new(), HashMap::new())
        };

        Ok(Self {
            path,
            session,
            datasets,
            command_start: None,
            memory: MemoryTracker::unlimited(),
            memory_by_name: HashMap::new(),
            max_memory_mb: 0,
            memory_warned: false,
        })
    }

    /// Applies `--max-memory` (MiB; 0 = no limit): counts the datasets already in memory
    /// (reloaded from the session file) and prints the warning line if they exceed it.
    /// Called once by `main`.
    pub fn set_memory_limit(&mut self, max_memory_mb: usize) {
        self.max_memory_mb = max_memory_mb;
        self.memory = MemoryTracker::new(max_memory_mb);
        self.memory_by_name.clear();
        let estimates: Vec<(String, usize)> = self
            .datasets
            .iter()
            .map(|(name, ds)| (name.clone(), estimate_dataset_memory(ds)))
            .collect();
        for (name, bytes) in estimates {
            self.count_bytes(name, bytes);
        }
        self.emit_memory_warning();
    }

    /// Counts `bytes` for the dataset `name`, replacing the estimate of an earlier dataset
    /// of that name. The new estimate is added before the old one is removed, so a
    /// replacement neither counts twice nor re-arms a warning already taken.
    fn count_bytes(&mut self, name: String, bytes: usize) {
        self.memory.add_bytes(bytes);
        if let Some(old) = self.memory_by_name.insert(name, bytes) {
            self.memory.remove_bytes(old);
        }
    }

    /// The memory warning, at most once per invocation, with the flag that set the limit.
    fn memory_warning(&mut self) -> Option<String> {
        if self.memory_warned {
            return None;
        }
        let w = self
            .memory
            .take_warning()
            .map(|w| format!("{w} (--max-memory {})", self.max_memory_mb))?;
        self.memory_warned = true;
        Some(w)
    }

    /// Prints the memory warning on stderr, if there is one.
    fn emit_memory_warning(&mut self) {
        if let Some(w) = self.memory_warning() {
            eprintln!("Warning: {w}");
        }
    }

    /// Start timing a command
    pub fn start_command(&mut self) {
        self.command_start = Some(Instant::now());
    }

    /// Record a completed command
    pub fn record_command(
        &mut self,
        category: &str,
        subcommand: &str,
        command_line: &str,
        arguments: serde_json::Value,
        dataset_refs: Vec<String>,
        output_files: Vec<PathBuf>,
        success: bool,
        error: Option<String>,
    ) {
        let duration_ms = self
            .command_start
            .map(|start| start.elapsed().as_millis() as u64)
            .unwrap_or(0);

        let record = CommandRecord {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            command_line: command_line.to_string(),
            category: category.to_string(),
            subcommand: subcommand.to_string(),
            arguments,
            argv: Vec::new(),
            dataset_refs,
            output_files,
            success,
            error,
            duration_ms,
        };

        self.session.record_command(record);
        self.command_start = None;
    }

    /// Record a command invocation from its command-portion argv (category,
    /// subcommand, and arguments), recorded without the program name, without the
    /// global flags typed before the category and without `--session`; global
    /// flags typed later are kept. The recorded subcommand is the first token
    /// after the category that is not a global flag or a global flag's value.
    ///
    /// This is the central recording path used by `main` for every analytics
    /// command, so reproducibility does not depend on each of the ~150 handlers
    /// remembering to record itself.
    pub fn record_invocation(&mut self, argv: Vec<String>, success: bool, error: Option<String>) {
        if argv.is_empty() {
            return;
        }
        let duration_ms = self
            .command_start
            .map(|start| start.elapsed().as_millis() as u64)
            .unwrap_or(0);
        let category = argv.first().cloned().unwrap_or_default();
        let mut i = 1;
        while let Some(width) = argv.get(i).and_then(|tok| crate::global_flag_width(tok)) {
            i += width;
        }
        let subcommand = argv.get(i).cloned().unwrap_or_default();
        let command_line = crate::output::shell_join(&argv);

        let record = CommandRecord {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            command_line,
            category,
            subcommand,
            arguments: serde_json::Value::Null,
            argv,
            dataset_refs: Vec::new(),
            output_files: Vec::new(),
            success,
            error,
            duration_ms,
        };

        self.session.record_command(record);
        self.command_start = None;
    }

    /// Register a loaded dataset; `na_strings` is the `--na-strings` list of a CSV
    /// load (`None`: the loader's default), stored so that reloads read the file
    /// the same way
    pub fn register_dataset(
        &mut self,
        name: &str,
        source_path: PathBuf,
        format: &str,
        dataset: &Dataset,
        na_strings: Option<Vec<String>>,
    ) {
        let df = dataset.df();
        let meta = DatasetMeta {
            name: name.to_string(),
            source_path,
            format: format.to_string(),
            nrows: df.height(),
            columns: df
                .get_column_names()
                .iter()
                .map(|s| s.to_string())
                .collect(),
            loaded_at: Utc::now(),
            na_strings,
        };
        self.session.register_dataset(meta);
    }

    /// Store a dataset in memory
    pub fn store_dataset(&mut self, name: String, dataset: Dataset) {
        self.count_bytes(name.clone(), estimate_dataset_memory(&dataset));
        self.datasets.insert(name, dataset);
        self.emit_memory_warning();
    }

    /// Get a dataset by name
    pub fn get_dataset(&self, name: &str) -> Option<&Dataset> {
        self.datasets.get(name)
    }

    /// List all loaded datasets
    pub fn list_datasets(&self) -> Vec<&String> {
        self.datasets.keys().collect()
    }

    /// Save the session to disk
    pub fn save(&self) -> anyhow::Result<()> {
        self.session.save(&self.path)
    }

    /// Get reference to the session
    pub fn session(&self) -> &Session {
        &self.session
    }
}

/// In-memory dataset store for non-session mode
pub struct DatasetStore {
    datasets: HashMap<String, Dataset>,
}

impl DatasetStore {
    pub fn new() -> Self {
        Self {
            datasets: HashMap::new(),
        }
    }

    pub fn insert(&mut self, name: String, dataset: Dataset) {
        self.datasets.insert(name, dataset);
    }

    pub fn get(&self, name: &str) -> Option<&Dataset> {
        self.datasets.get(name)
    }

    pub fn list(&self) -> Vec<&String> {
        self.datasets.keys().collect()
    }

    pub fn remove(&mut self, name: &str) -> Option<Dataset> {
        self.datasets.remove(name)
    }
}

impl Default for DatasetStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod record_invocation_tests {
    use super::SessionManager;

    fn recorded_subcommand(argv: &[&str]) -> String {
        let dir = tempfile::tempdir().unwrap();
        let mut manager = SessionManager::new(dir.path().join("s.json")).unwrap();
        manager.start_command();
        manager.record_invocation(argv.iter().map(|s| s.to_string()).collect(), true, None);
        let record = &manager.session().commands[0];
        assert_eq!(record.category, argv[0]);
        record.subcommand.clone()
    }

    #[test]
    fn subcommand_skips_globals_typed_after_the_category() {
        assert_eq!(recorded_subcommand(&["data", "load", "f.csv"]), "load");
        assert_eq!(
            recorded_subcommand(&["data", "-F", "json", "load", "f.csv"]),
            "load"
        );
        assert_eq!(
            recorded_subcommand(&["data", "--format=json", "--quiet", "load"]),
            "load"
        );
        assert_eq!(recorded_subcommand(&["reg", "-Fjson", "ols", "d"]), "ols");
        assert_eq!(recorded_subcommand(&["smoke-test"]), "");
    }
}

#[cfg(test)]
mod dataset_meta_tests {
    use super::DatasetMeta;

    #[test]
    fn na_strings_round_trip_and_old_sessions_parse() {
        let old = r#"{"name":"d","source_path":"d.csv","format":"csv","nrows":3,"columns":["x"],"loaded_at":"2026-01-01T00:00:00Z"}"#;
        let meta: DatasetMeta = serde_json::from_str(old).unwrap();
        assert_eq!(meta.na_strings, None);
        assert!(!serde_json::to_string(&meta).unwrap().contains("na_strings"));
        let with = DatasetMeta {
            na_strings: Some(vec![".".into(), "NA".into()]),
            ..meta
        };
        let back: DatasetMeta =
            serde_json::from_str(&serde_json::to_string(&with).unwrap()).unwrap();
        assert_eq!(
            back.na_strings,
            Some(vec![".".to_string(), "NA".to_string()])
        );
    }
}

#[cfg(test)]
mod memory_limit_tests {
    use super::*;
    use crate::memory::estimate_dataset_memory;
    use polars::prelude::*;

    fn two_float_columns(rows: usize) -> Dataset {
        let x: Vec<f64> = (0..rows).map(|i| i as f64 + 0.5).collect();
        let y: Vec<f64> = (0..rows).map(|i| 2.0 * i as f64 + 0.25).collect();
        Dataset::new(df! { "x" => x, "y" => y }.unwrap())
    }

    #[test]
    fn memory_limit_warns_once_and_counts_replacements_once() {
        let dir = tempfile::tempdir().unwrap();
        let mut m = SessionManager::new(dir.path().join("s.json")).unwrap();
        m.set_memory_limit(1);
        assert_eq!(m.memory_warning(), None);
        let bytes = estimate_dataset_memory(&two_float_columns(70_000));
        assert_eq!(bytes, 1_232_000); // 70,000 x 2 x 8 bytes + 10 %
        m.count_bytes("a".into(), bytes);
        assert_eq!(
            m.memory_warning().as_deref(),
            Some("estimated dataset memory 1.17 MB exceeds the limit of 1.00 MB (--max-memory 1)")
        );
        m.count_bytes("a".into(), bytes); // replaced, not added
        assert_eq!(m.memory.current_usage(), 1_232_000);
        m.count_bytes("b".into(), bytes);
        assert_eq!(m.memory.current_usage(), 2_464_000);
        assert_eq!(
            m.memory_warning(),
            None,
            "at most one warning per invocation"
        );
    }

    #[test]
    fn memory_limit_zero_never_warns() {
        let dir = tempfile::tempdir().unwrap();
        let mut m = SessionManager::new(dir.path().join("s.json")).unwrap();
        m.set_memory_limit(0);
        m.count_bytes("a".into(), usize::MAX / 2);
        assert_eq!(m.memory_warning(), None);
    }
}
