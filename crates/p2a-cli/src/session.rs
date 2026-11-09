//! Session recording and management for reproducibility

use crate::memory::{MemoryTracker, estimate_dataset_memory};
use chrono::{DateTime, Utc};
use p2a_core::{DataLoader, Dataset};
use serde::{Deserialize, Serialize};
use std::cell::{OnceCell, RefCell};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
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
    /// Written by this session (munge result, data generate) to
    /// `sidecar_dir(session)/<file>.parquet`; `source_path` is then relative to the session
    /// file's directory. Absent in older session files and for loaded files.
    #[serde(default, skip_serializing_if = "is_false")]
    pub derived: bool,
}

/// serde: omit `derived` when it is false.
fn is_false(b: &bool) -> bool {
    !*b
}

/// The longest file name (in bytes, before ".parquet") a derived dataset is stored under.
const MAX_SIDECAR_NAME: usize = 200;

/// `<dir>/<session file name>.datasets`: where a session keeps the datasets it derived.
pub fn sidecar_dir(session_path: &Path) -> PathBuf {
    let mut name = session_path
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    name.push(".datasets");
    session_path.with_file_name(name)
}

/// The file a derived dataset is stored in: every byte of `name` outside `[a-z0-9_-]`
/// (uppercase ASCII letters included) as `%XX`, plus ".parquet". The map is injective,
/// and stays injective under case-insensitive comparison (the encoded name has no
/// uppercase letter outside the hex digits of an escape, whose case never varies), so
/// two names never share a file on a case-insensitive filesystem; no name leaves the
/// directory. A name that would encode to a Windows device name (`con`, `nul`, `com1`,
/// ...) has its first byte escaped too.
fn sidecar_file_name(name: &str) -> anyhow::Result<String> {
    if name.is_empty() {
        anyhow::bail!("a dataset name cannot be empty");
    }
    let mut out = String::with_capacity(name.len());
    for b in name.bytes() {
        if b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-' {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    if is_windows_device_name(&out) {
        out = format!("%{:02X}{}", out.as_bytes()[0], &out[1..]);
    }
    if out.len() > MAX_SIDECAR_NAME {
        anyhow::bail!(
            "the dataset name '{}' is too long to store ({} bytes encoded, at most {MAX_SIDECAR_NAME})",
            p2a_text::ellipsize(name, 40),
            out.len()
        );
    }
    out.push_str(".parquet");
    Ok(out)
}

/// Whether `stem` (lowercase, no extension) is a name Windows reserves for a device.
fn is_windows_device_name(stem: &str) -> bool {
    matches!(stem, "con" | "prn" | "aux" | "nul")
        || ((stem.starts_with("com") || stem.starts_with("lpt"))
            && stem.len() == 4
            && matches!(stem.as_bytes()[3], b'1'..=b'9'))
}

/// The file of a derived dataset entry, checked: its `source_path` must be exactly
/// `<session file name>.datasets/<file>.parquet`, as [`SessionManager::persist_dataset`]
/// writes it. A hand-edited or foreign session file that points anywhere else is refused,
/// so neither a reload nor a replacing `data load` reads or removes a file outside the
/// session's own directory of derived datasets.
fn derived_file(meta: &DatasetMeta, session_path: &Path) -> anyhow::Result<PathBuf> {
    use std::path::Component;
    let dir = sidecar_dir(session_path);
    let expected_dir = dir.file_name().unwrap_or_default();
    let parts: Vec<Component> = meta.source_path.components().collect();
    match parts.as_slice() {
        [Component::Normal(d), Component::Normal(f)]
            if *d == expected_dir
                && f.to_str()
                    .is_some_and(|f| f.ends_with(".parquet") && f.len() > ".parquet".len()) =>
        {
            Ok(dir.join(f))
        }
        _ => anyhow::bail!(
            "the session entry points outside {} ({}); refusing to use it",
            expected_dir.to_string_lossy(),
            meta.source_path.display()
        ),
    }
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
/// loaded with. A derived dataset's file is checked by [`derived_file`].
fn reload_dataset(meta: &DatasetMeta, session_path: &Path) -> anyhow::Result<Dataset> {
    if meta.derived {
        let df = DataLoader::load_parquet(derived_file(meta, session_path)?)?;
        return Ok(Dataset::new(df));
    }
    // the markers apply to CSV only; `data load` refuses them for any other format
    let na = match meta.format.as_str() {
        "csv" => meta.na_strings.as_deref(),
        _ => None,
    };
    Ok(crate::commands::data::load_with_markers(
        &meta.source_path,
        na,
    )?)
}

/// Estimated memory of the datasets in memory, against `--max-memory`.
struct MemoryState {
    /// The running total
    tracker: MemoryTracker,
    /// The estimate counted for each dataset name (a replaced dataset is counted once)
    by_name: HashMap<String, usize>,
    /// The `--max-memory` value in MiB (0 = no limit)
    max_mb: usize,
    /// Whether this invocation has printed its memory warning
    warned: bool,
}

/// Manages session state during CLI execution
pub struct SessionManager {
    /// Path to the session file
    path: PathBuf,
    /// The session being recorded
    session: Session,
    /// The datasets by name, each read from its file the first time a command asks for
    /// it (`None`: the reload failed and was reported)
    datasets: HashMap<String, OnceCell<Option<Dataset>>>,
    /// Timer for command duration tracking
    command_start: Option<Instant>,
    /// Memory accounting; charged when a dataset is loaded or stored
    memory: RefCell<MemoryState>,
}

impl SessionManager {
    /// Create a new session manager, loading existing session if file exists
    /// The session file is read now; its datasets are read when a command first asks for
    /// them ([`SessionManager::get_dataset`]).
    pub fn new(path: PathBuf) -> anyhow::Result<Self> {
        let session = if path.exists() {
            Session::load(&path)?
        } else {
            Session::new()
        };
        let datasets = session
            .datasets
            .keys()
            .map(|name| (name.clone(), OnceCell::new()))
            .collect();

        Ok(Self {
            path,
            session,
            datasets,
            command_start: None,
            memory: RefCell::new(MemoryState {
                tracker: MemoryTracker::unlimited(),
                by_name: HashMap::new(),
                max_mb: 0,
                warned: false,
            }),
        })
    }

    /// Applies `--max-memory` (MiB; 0 = no limit): with a limit, loads the session's
    /// datasets, counts them and prints the warning line if they exceed it; datasets
    /// stored later are counted when they are stored. Called once by `main`.
    pub fn set_memory_limit(&mut self, max_memory_mb: usize) {
        self.memory.borrow_mut().max_mb = max_memory_mb;
        if max_memory_mb > 0 {
            // a limit applies to the whole session: load every dataset so the estimate
            // counts them all (without a limit they stay unloaded until a command reads
            // them). The loads print no warning of their own; one follows for the total.
            let warned = std::mem::replace(&mut self.memory.borrow_mut().warned, true);
            let names: Vec<String> = self.datasets.keys().cloned().collect();
            for name in &names {
                let _ = self.get_dataset(name);
            }
            self.memory.borrow_mut().warned = warned;
        }
        let estimates: Vec<(String, usize)> = self
            .datasets
            .iter()
            .filter_map(|(name, cell)| {
                let ds = cell.get()?.as_ref()?;
                Some((name.clone(), estimate_dataset_memory(ds)))
            })
            .collect();
        {
            let mut m = self.memory.borrow_mut();
            m.tracker = MemoryTracker::new(max_memory_mb);
            m.by_name.clear();
        }
        for (name, bytes) in estimates {
            self.count_bytes(name, bytes);
        }
        self.emit_memory_warning();
    }

    /// Counts `bytes` for the dataset `name`, replacing the estimate of an earlier dataset
    /// of that name. The new estimate is added before the old one is removed, so a
    /// replacement neither counts twice nor re-arms a warning already taken.
    fn count_bytes(&self, name: String, bytes: usize) {
        let mut m = self.memory.borrow_mut();
        m.tracker.add_bytes(bytes);
        if let Some(old) = m.by_name.insert(name, bytes) {
            m.tracker.remove_bytes(old);
        }
    }

    /// The memory warning, at most once per invocation, with the flag that set the limit.
    fn memory_warning(&self) -> Option<String> {
        let mut m = self.memory.borrow_mut();
        if m.warned {
            return None;
        }
        let max_mb = m.max_mb;
        let w = m
            .tracker
            .take_warning()
            .map(|w| format!("{w} (--max-memory {max_mb})"))?;
        m.warned = true;
        Some(w)
    }

    /// Prints the memory warning on stderr, if there is one.
    fn emit_memory_warning(&self) {
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
        self.remove_derived_copy(name);
        let meta = dataset_meta(name, source_path, format, dataset, na_strings, false);
        self.session.register_dataset(meta);
    }

    /// Removes the stored file of `name` when the session entry it is about to replace
    /// was derived. A file already gone is fine; any other failure is a warning.
    fn remove_derived_copy(&self, name: &str) {
        let Some(old) = self.session.datasets.get(name) else {
            return;
        };
        if !old.derived {
            return;
        }
        let file = match derived_file(old, &self.path) {
            Ok(file) => file,
            Err(e) => {
                eprintln!("Warning: not removing the stored copy of '{name}': {e}");
                return;
            }
        };
        match std::fs::remove_file(file) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => eprintln!("Warning: could not remove the stored copy of '{name}': {e}"),
        }
    }

    /// Writes `dataset` as Parquet into [`sidecar_dir`] (a temporary file, then a rename),
    /// registers it (derived, "parquet", a path relative to the session file's directory),
    /// then keeps it in memory. An error leaves the session unchanged.
    pub fn persist_dataset(&mut self, name: &str, dataset: Dataset) -> anyhow::Result<()> {
        let stored = (|| -> anyhow::Result<PathBuf> {
            let file = sidecar_file_name(name)?;
            let dir = sidecar_dir(&self.path);
            std::fs::create_dir_all(&dir)?;
            let tmp = dir.join(format!(".{file}.{}.tmp", uuid::Uuid::new_v4()));
            let written = dataset
                .to_parquet(&tmp)
                .map_err(anyhow::Error::from)
                .and_then(|()| std::fs::rename(&tmp, dir.join(&file)).map_err(Into::into));
            if let Err(e) = written {
                let _ = std::fs::remove_file(&tmp);
                return Err(e);
            }
            let dir_name = dir.file_name().map(PathBuf::from).unwrap_or_default();
            Ok(dir_name.join(file))
        })()
        .map_err(|e| anyhow::anyhow!("Could not store dataset '{name}' in the session: {e}"))?;
        let meta = dataset_meta(name, stored, "parquet", &dataset, None, true);
        self.session.register_dataset(meta);
        self.store_dataset(name.to_string(), dataset);
        Ok(())
    }

    /// Store a dataset in memory
    pub fn store_dataset(&mut self, name: String, dataset: Dataset) {
        self.count_bytes(name.clone(), estimate_dataset_memory(&dataset));
        self.datasets.insert(name, OnceCell::from(Some(dataset)));
        self.emit_memory_warning();
    }

    /// Get a dataset by name, reading it from its file the first time it is asked for. A
    /// file that cannot be read is reported once (`Warning: Could not reload dataset ..`)
    /// and the dataset is then absent.
    pub fn get_dataset(&self, name: &str) -> Option<&Dataset> {
        let cell = self.datasets.get(name)?;
        cell.get_or_init(|| {
            let meta = self.session.datasets.get(name)?;
            match reload_dataset(meta, &self.path) {
                Ok(ds) => {
                    self.count_bytes(name.to_string(), estimate_dataset_memory(&ds));
                    self.emit_memory_warning();
                    Some(ds)
                }
                Err(e) => {
                    eprintln!("Warning: Could not reload dataset '{}': {}", name, e);
                    None
                }
            }
        })
        .as_ref()
    }

    /// The names of the session's datasets (registered or stored), loaded or not
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

/// The session entry of a dataset.
fn dataset_meta(
    name: &str,
    source_path: PathBuf,
    format: &str,
    dataset: &Dataset,
    na_strings: Option<Vec<String>>,
    derived: bool,
) -> DatasetMeta {
    let df = dataset.df();
    DatasetMeta {
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
        derived,
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
        assert_eq!(m.memory.borrow().tracker.current_usage(), 1_232_000);
        m.count_bytes("b".into(), bytes);
        assert_eq!(m.memory.borrow().tracker.current_usage(), 2_464_000);
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

#[cfg(test)]
mod derived_dataset_tests {
    use super::*;
    use polars::prelude::*;

    fn sample() -> Dataset {
        Dataset::new(
            df! {
                "i" => [Some(1i64), None, Some(3)],
                "f" => [Some(0.5), Some(1.5), None],
                "s" => [Some("a"), None, Some("c")],
                "b" => [Some(true), Some(false), None],
            }
            .unwrap(),
        )
    }

    #[test]
    fn sidecar_names_are_encoded_and_bounded() {
        assert_eq!(sidecar_file_name("g_sorted").unwrap(), "g_sorted.parquet");
        assert_eq!(
            sidecar_file_name("my data/ü").unwrap(),
            "my%20data%2F%C3%BC.parquet"
        );
        assert_eq!(sidecar_file_name("..").unwrap(), "%2E%2E.parquet");
        assert_eq!(sidecar_file_name("Sales").unwrap(), "%53ales.parquet");
        // distinct under case folding (macOS APFS, NTFS)
        for (a, b) in [("Sales", "sales"), ("AB", "ab"), ("aB", "Ab")] {
            assert_ne!(
                sidecar_file_name(a).unwrap().to_lowercase(),
                sidecar_file_name(b).unwrap().to_lowercase()
            );
        }
        assert_eq!(sidecar_file_name("con").unwrap(), "%63on.parquet");
        assert_eq!(sidecar_file_name("com1").unwrap(), "%63om1.parquet");
        assert_eq!(sidecar_file_name("com").unwrap(), "com.parquet");
        assert_eq!(sidecar_file_name("console").unwrap(), "console.parquet");
        assert_ne!(
            sidecar_file_name("con").unwrap(),
            sidecar_file_name("%63on").unwrap()
        );
        assert!(sidecar_file_name("").is_err());
        assert!(sidecar_file_name(&"x".repeat(201)).is_err());
        assert_eq!(sidecar_file_name(&"x".repeat(200)).unwrap().len(), 208);
    }

    #[test]
    fn sidecar_dir_sits_next_to_the_session_file() {
        assert_eq!(
            sidecar_dir(Path::new("a/b/s.json")),
            PathBuf::from("a/b/s.json.datasets")
        );
        assert_eq!(
            sidecar_dir(Path::new("s.json")),
            PathBuf::from("s.json.datasets")
        );
    }

    #[test]
    fn persisted_dataset_reloads_with_types_and_nulls() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.json");
        let mut m = SessionManager::new(path.clone()).unwrap();
        m.persist_dataset("d", sample()).unwrap();
        m.save().unwrap();
        let meta = &m.session().datasets["d"];
        assert!(
            meta.derived
                && meta.format == "parquet"
                && meta.source_path == Path::new("s.json.datasets/d.parquet")
        );
        let again = SessionManager::new(path).unwrap();
        assert!(
            again
                .get_dataset("d")
                .unwrap()
                .df()
                .equals_missing(sample().df())
        );
    }

    #[test]
    fn loading_over_a_derived_dataset_removes_its_file() {
        let dir = tempfile::tempdir().unwrap();
        let mut m = SessionManager::new(dir.path().join("s.json")).unwrap();
        m.persist_dataset("d", sample()).unwrap();
        let file = dir.path().join("s.json.datasets/d.parquet");
        assert!(file.exists());
        m.register_dataset("d", dir.path().join("d.csv"), "csv", &sample(), None);
        assert!(!file.exists());
        assert!(!m.session().datasets["d"].derived);
    }

    /// A derived entry whose path leaves the session's directory of derived datasets is
    /// neither removed by a replacing load nor read by a reload.
    #[test]
    fn derived_entries_outside_the_sidecar_dir_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.json");
        let outside = dir.path().join("outside.parquet");
        sample().to_parquet(&outside).unwrap();
        for bad in [
            PathBuf::from("../outside.parquet"),
            PathBuf::from("outside.parquet"),
            outside.clone(),
            PathBuf::from("s.json.datasets/../outside.parquet"),
            PathBuf::from("other.datasets/outside.parquet"),
            PathBuf::from("s.json.datasets/x.csv"),
        ] {
            let mut m = SessionManager::new(path.clone()).unwrap();
            m.persist_dataset("d", sample()).unwrap();
            m.session.datasets.get_mut("d").unwrap().source_path = bad.clone();
            m.save().unwrap();
            let again = SessionManager::new(path.clone()).unwrap();
            assert!(again.get_dataset("d").is_none(), "{bad:?} was read");
            let mut again = SessionManager::new(path.clone()).unwrap();
            again.register_dataset("d", dir.path().join("d.csv"), "csv", &sample(), None);
            assert!(outside.exists(), "{bad:?} removed the outside file");
        }
        // the path persist_dataset writes is accepted
        let mut m = SessionManager::new(dir.path().join("t.json")).unwrap();
        m.persist_dataset("d", sample()).unwrap();
        let meta = &m.session().datasets["d"];
        assert_eq!(
            derived_file(meta, &dir.path().join("t.json")).unwrap(),
            dir.path().join("t.json.datasets/d.parquet")
        );
    }

    #[test]
    fn old_session_files_parse_without_derived() {
        let old = r#"{"name":"d","source_path":"d.csv","format":"csv","nrows":3,"columns":["x"],"loaded_at":"2026-01-01T00:00:00Z"}"#;
        let meta: DatasetMeta = serde_json::from_str(old).unwrap();
        assert!(!meta.derived);
        assert!(!serde_json::to_string(&meta).unwrap().contains("derived"));
    }

    /// Datasets are read when first used: a session whose files include a missing one
    /// opens, the intact dataset loads, and the missing one is absent (warned once).
    #[test]
    fn datasets_load_on_first_use() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.json");
        let a = dir.path().join("a.csv");
        let b = dir.path().join("b.csv");
        std::fs::write(&a, "x,y\n1,2.5\n3,4.5\n").unwrap();
        std::fs::write(&b, "x\n1\n").unwrap();
        let mut m = SessionManager::new(path.clone()).unwrap();
        m.register_dataset("a", a.clone(), "csv", &sample(), None);
        m.register_dataset("b", b.clone(), "csv", &sample(), None);
        m.save().unwrap();
        std::fs::remove_file(&b).unwrap();
        let again = SessionManager::new(path).unwrap();
        assert!(
            again.datasets.values().all(|c| c.get().is_none()),
            "nothing read yet"
        );
        assert_eq!(again.get_dataset("a").unwrap().df().height(), 2);
        assert!(again.get_dataset("b").is_none());
        assert!(again.get_dataset("b").is_none()); // the cell holds the failure: no second read
        assert!(again.datasets["b"].get().is_some_and(Option::is_none));
        let mut names = again.list_datasets();
        names.sort();
        assert_eq!(names, ["a", "b"]);
    }
}
