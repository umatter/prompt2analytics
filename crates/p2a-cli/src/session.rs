//! Session recording and management for reproducibility

use crate::memory::{MemoryTracker, estimate_dataset_memory};
use chrono::{DateTime, Utc};
use p2a_core::{DataLoader, Dataset};
use serde::{Deserialize, Serialize};
use std::cell::{OnceCell, RefCell};
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Metadata about a loaded dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMeta {
    /// Dataset name/identifier
    pub name: String,
    /// The file the dataset was loaded from, as an absolute path (session files written by
    /// older p2a versions may hold a path relative to [`Session::working_dir`]); for a
    /// derived dataset, the stored copy relative to the session file's directory
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
    /// What value-labelled columns of a Stata file load as (`data load --value-labels`),
    /// used again on every reload; `None` means their codes (the default). Absent in older
    /// session files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value_labels: Option<String>,
    /// Written by this session (munge result, data generate) to
    /// `sidecar_dir(session)/<file>.parquet`; `source_path` is then relative to the session
    /// file's directory. Absent in older session files and for loaded files.
    #[serde(default, skip_serializing_if = "is_false")]
    pub derived: bool,
    /// Size of the source file in bytes at `data load`; a reload warns when it differs.
    /// `None` for derived datasets, older files and a file whose size could not be read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_bytes: Option<u64>,
    /// Modification time of the source file at `data load`; a reload warns when it
    /// differs. `None` as for `source_bytes`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_modified: Option<DateTime<Utc>>,
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
/// directory. A name that would encode to a Windows device name (`con`, `nul`, `com0` to
/// `com9`, `lpt0` to `lpt9`, ...) has its first byte escaped too.
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
            && stem.as_bytes()[3].is_ascii_digit())
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
            "the stored copy of dataset '{}' must be under {}/ (found {}); was the session file renamed or copied? Re-run the commands that created it",
            meta.name,
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
    /// Not filled any more (recorded as `null`); [`CommandRecord::argv`] is the
    /// canonical record of a command, and `script export` reads only `argv` (or this field
    /// for records older than `argv`). Still written, because older p2a versions require it.
    #[serde(default)]
    pub arguments: serde_json::Value,
    /// The command-portion argv (category, subcommand, and args) as actually
    /// invoked: without the program name, without the global flags typed before
    /// the category and without `--session`; global flags typed later are kept.
    /// Preferred over `arguments` for faithful, injection-safe script export.
    #[serde(default)]
    pub argv: Vec<String>,
    /// Not filled any more (recorded empty); see [`CommandRecord::arguments`]
    #[serde(default)]
    pub dataset_refs: Vec<String>,
    /// Not filled any more (recorded empty); see [`CommandRecord::arguments`]
    #[serde(default)]
    pub output_files: Vec<PathBuf>,
    /// The working directory of the invocation (`None` in older files); an exported
    /// script changes into it before replaying the command
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<PathBuf>,
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
    /// The working directory of the session's first command; older files' relative
    /// dataset paths are read against it
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

    /// Save the session to a file, atomically ([`write_atomically`]).
    pub fn save(&self, path: &PathBuf) -> anyhow::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        write_atomically(path, content.as_bytes())?;
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

/// What identifies the contents of a session file between two reads: length,
/// modification time and, on Unix, device and inode (a rename over the file changes them).
/// Off Unix it is length and modification time only: on a file system with coarse
/// modification times (FAT and exFAT, 2 s), two parallel saves that write the same length
/// within one tick look unchanged, and the later one can drop the other's commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Fingerprint {
    len: u64,
    modified: Option<std::time::SystemTime>,
    #[cfg(unix)]
    dev: u64,
    #[cfg(unix)]
    ino: u64,
}

impl Fingerprint {
    fn of(meta: &std::fs::Metadata) -> Self {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        Self {
            len: meta.len(),
            modified: meta.modified().ok(),
            #[cfg(unix)]
            dev: meta.dev(),
            #[cfg(unix)]
            ino: meta.ino(),
        }
    }
}

/// Reads a session file through one handle, so the fingerprint describes exactly the bytes
/// read: `(None, None)` when the file does not exist, `(None, Some(_))` when it holds only
/// whitespace, the parsed session otherwise.
fn read_session_file(path: &Path) -> anyhow::Result<(Option<Session>, Option<Fingerprint>)> {
    use std::io::Read as _;
    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((None, None)),
        Err(e) => return Err(e.into()),
    };
    let fingerprint = Fingerprint::of(&file.metadata()?);
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    if content.trim().is_empty() {
        return Ok((None, Some(fingerprint)));
    }
    Ok((Some(serde_json::from_str(&content)?), Some(fingerprint)))
}

/// `<session file>.lock`, beside the session file: the file [`SessionManager::save`] locks.
/// Created at the first save and never removed.
fn lock_path(session_path: &Path) -> PathBuf {
    let mut name = session_path
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    name.push(".lock");
    session_path.with_file_name(name)
}

/// Writes `bytes` to `path` so that a reader sees the old file or the new one, never a
/// part: a temporary file `.<name>.<uuid>.tmp` in the target's directory (with the
/// existing file's permissions), then a rename over the target. A symbolic link is
/// followed (the file it points to is replaced, the link stays). The temporary file is
/// removed on any error. No fsync: the rename is what makes the write atomic for readers.
/// A dangling link is followed too ([`dangling_link_target`]): the file it names is created.
/// The target's directory must be writable, even when the target file itself is.
pub(crate) fn write_atomically(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write as _;
    let target = match std::fs::symlink_metadata(path) {
        Ok(m) if m.file_type().is_symlink() => match std::fs::canonicalize(path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => dangling_link_target(path)?,
            Err(e) => return Err(e),
        },
        _ => path.to_path_buf(),
    };
    let name = target.file_name().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{} is not a file name", target.display()),
        )
    })?;
    let dir = match target.parent() {
        Some(d) if !d.as_os_str().is_empty() => d.to_path_buf(),
        _ => PathBuf::from("."),
    };
    let tmp = dir.join(format!(
        ".{}.{}.tmp",
        name.to_string_lossy(),
        uuid::Uuid::new_v4()
    ));
    let written = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)?;
        file.write_all(bytes)?;
        if let Ok(old) = std::fs::metadata(&target) {
            file.set_permissions(old.permissions())?;
        }
        drop(file);
        std::fs::rename(&tmp, &target)
    })();
    if written.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    written
}

/// The most links [`dangling_link_target`] follows, as Linux's limit for one path lookup.
const MAX_LINK_HOPS: usize = 40;

/// The file a dangling symbolic link at `link` finally names, found as `std::fs::write`
/// finds it: each link is read in turn (a relative target against the link's own
/// directory) until a path that does not exist. More than [`MAX_LINK_HOPS`] links is the
/// loop error the operating system gives.
fn dangling_link_target(link: &Path) -> std::io::Result<PathBuf> {
    let mut path = link.to_path_buf();
    for _ in 0..MAX_LINK_HOPS {
        match std::fs::symlink_metadata(&path) {
            Ok(m) if m.file_type().is_symlink() => {
                let next = std::fs::read_link(&path)?;
                path = match path.parent() {
                    Some(dir) if next.is_relative() => dir.join(next),
                    _ => next,
                };
            }
            _ => return Ok(path),
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        format!(
            "{}: more than {MAX_LINK_HOPS} levels of symbolic links",
            link.display()
        ),
    ))
}

/// Reload a dataset from its source file, with the missing-value strings it was
/// loaded with, and the warnings of [`reload_warnings`]. A derived dataset's file is
/// checked by [`derived_file`]. An absolute source path is read as stored; a relative one
/// (older session files) against `working_dir` when the file exists there, else against
/// the current directory.
fn reload_dataset(
    meta: &DatasetMeta,
    session_path: &Path,
    working_dir: &Path,
) -> anyhow::Result<(Dataset, Vec<String>)> {
    if meta.derived {
        let df = DataLoader::load_parquet(derived_file(meta, session_path)?)?;
        return Ok((Dataset::new(df), Vec::new()));
    }
    let path = if meta.source_path.is_absolute() || working_dir.as_os_str().is_empty() {
        meta.source_path.clone()
    } else {
        let in_wd = working_dir.join(&meta.source_path);
        if in_wd.is_file() {
            in_wd
        } else {
            meta.source_path.clone()
        }
    };
    let file = std::fs::metadata(&path).ok();
    // the markers apply to CSV only; `data load` refuses them for any other format
    let na = match meta.format.as_str() {
        "csv" => meta.na_strings.as_deref(),
        _ => None,
    };
    let value_labels = match meta.value_labels.as_deref() {
        Some(v) => Some(
            v.parse::<p2a_core::data::ValueLabels>()
                .map_err(|e| anyhow::anyhow!(crate::output::core_message(e)))?,
        ),
        None => None,
    };
    let dataset = crate::commands::data::load_with_markers(&path, na, value_labels)?;
    let warnings = reload_warnings(meta, &path, file.as_ref(), &dataset);
    Ok((dataset, warnings))
}

/// Why a reloaded dataset may not match the recording: its file's size or modification
/// time differ from those recorded at `data load`, or the data have other rows or
/// columns than recorded. One line each; empty when nothing differs.
fn reload_warnings(
    meta: &DatasetMeta,
    path: &Path,
    file: Option<&std::fs::Metadata>,
    dataset: &Dataset,
) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(file) = file {
        let size_differs = meta.source_bytes.is_some_and(|b| b != file.len());
        let time_differs = meta
            .source_modified
            .is_some_and(|t| !file.modified().is_ok_and(|m| DateTime::<Utc>::from(m) == t));
        if size_differs || time_differs {
            out.push(format!(
                "the file of dataset '{}' ({}) changed since it was loaded into the session (size or modification time differ); results may not match the recorded ones",
                meta.name,
                path.display()
            ));
        }
    }
    let df = dataset.df();
    let same_columns = df.width() == meta.columns.len()
        && df
            .get_column_names()
            .iter()
            .zip(&meta.columns)
            .all(|(a, b)| a.as_str() == b);
    if df.height() != meta.nrows || !same_columns {
        out.push(format!(
            "dataset '{}' now has {} rows and {} columns; the session recorded {} rows and {} columns",
            meta.name,
            df.height(),
            df.width(),
            meta.nrows,
            meta.columns.len()
        ));
    }
    out
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

/// Manages session state during CLI execution.
///
/// Parallel invocations on one session file all record their commands: a save locks
/// `<session>.lock`, reads the file again when another invocation wrote it since this one
/// read it, appends this invocation's commands and its registered datasets to what is
/// there, and replaces the file atomically. A command does not see datasets that a
/// parallel command registers after it started.
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
    /// What [`SessionManager::save`] merges into the file
    saved: RefCell<SaveState>,
}

/// What a save needs to merge this invocation into a file that another invocation wrote.
struct SaveState {
    /// The file as this invocation last read or wrote it: `None` when there was no file;
    /// `Some(None)` after a merge (the file then differs from the session in memory, so
    /// the next save reads it again)
    file: Option<Option<Fingerprint>>,
    /// Commands of `session.commands` already in the file
    base_commands: usize,
    /// Datasets this invocation registered since the last save; their entries replace
    /// the file's
    touched: BTreeSet<String>,
}

impl SessionManager {
    /// Create a new session manager, loading existing session if file exists
    /// The session file is read now; its datasets are read when a command first asks for
    /// them ([`SessionManager::get_dataset`]).
    /// An empty or blank file (made by `mktemp`, say) is a new session.
    pub fn new(path: PathBuf) -> anyhow::Result<Self> {
        let (session, fingerprint) = read_session_file(&path)?;
        let session = session.unwrap_or_default();
        let saved = RefCell::new(SaveState {
            file: fingerprint.map(Some),
            base_commands: session.commands.len(),
            touched: BTreeSet::new(),
        });
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
            saved,
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
            cwd: std::env::current_dir().ok(),
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
        let source_path = std::path::absolute(&source_path).unwrap_or(source_path);
        let mut meta = dataset_meta(name, source_path, format, dataset, na_strings, false);
        if let Ok(file) = std::fs::metadata(&meta.source_path) {
            meta.source_bytes = Some(file.len());
            meta.source_modified = file.modified().ok().map(DateTime::<Utc>::from);
        }
        self.session.register_dataset(meta);
        self.saved.get_mut().touched.insert(name.to_string());
    }

    /// Records the `--value-labels` choice of the loaded dataset `name` (see
    /// [`DatasetMeta::value_labels`]); a name the session does not hold is ignored.
    pub fn set_value_labels(&mut self, name: &str, value_labels: &str) {
        if let Some(meta) = self.session.datasets.get_mut(name) {
            meta.value_labels = Some(value_labels.to_string());
        }
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
        self.saved.get_mut().touched.insert(name.to_string());
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
            match reload_dataset(meta, &self.path, &self.session.working_dir) {
                Ok((ds, warnings)) => {
                    for w in warnings {
                        eprintln!("Warning: {w}");
                    }
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

    /// Save the session to disk, under an exclusive lock on [`lock_path`]: when another
    /// invocation wrote the file since this one read it, this invocation's commands are
    /// appended to the file's and the datasets it registered replace the file's entries
    /// (the file's id, title, creation time, working directory and version are kept). The
    /// file is replaced atomically, so the session file's directory must be writable (the
    /// lock file and the temporary file are created there); an error names the file that
    /// could not be created. A file system without locking saves without the lock, after a
    /// warning.
    pub fn save(&self) -> anyhow::Result<()> {
        let lock_file = lock_path(&self.path);
        let could_not_lock =
            |e: std::io::Error| anyhow::anyhow!("could not lock {}: {e}", lock_file.display());
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&lock_file)
            .map_err(could_not_lock)?;
        match lock.lock() {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::Unsupported => eprintln!(
                "Warning: the file system does not support file locks; saving {} without one",
                self.path.display()
            ),
            Err(e) => return Err(could_not_lock(e)),
        }
        let mut state = self.saved.borrow_mut();
        let (on_disk, now) = match std::fs::metadata(&self.path) {
            Ok(m) => (true, Some(Fingerprint::of(&m))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => (false, None),
            Err(e) => return Err(e.into()),
        };
        let unchanged = match state.file {
            None => !on_disk,
            Some(Some(fp)) => now == Some(fp),
            Some(None) => false,
        };
        let merged = if unchanged {
            None
        } else {
            read_session_file(&self.path)?
                .0
                .map(|disk| self.merge_into(disk, &state))
        };
        let content = serde_json::to_string_pretty(merged.as_ref().unwrap_or(&self.session))?;
        write_atomically(&self.path, content.as_bytes()).map_err(|e| {
            anyhow::anyhow!("could not save session file {}: {e}", self.path.display())
        })?;
        state.file = if merged.is_some() {
            Some(None)
        } else {
            Some(
                std::fs::metadata(&self.path)
                    .ok()
                    .map(|m| Fingerprint::of(&m)),
            )
        };
        state.base_commands = self.session.commands.len();
        state.touched.clear();
        drop(lock); // unlocks
        Ok(())
    }

    /// `disk` with this invocation's commands since the last save appended and the
    /// datasets it registered since then replacing their entries.
    fn merge_into(&self, mut disk: Session, state: &SaveState) -> Session {
        let start = state.base_commands.min(self.session.commands.len());
        disk.commands
            .extend(self.session.commands[start..].iter().cloned());
        for name in &state.touched {
            if let Some(meta) = self.session.datasets.get(name) {
                disk.datasets.insert(name.clone(), meta.clone());
            }
        }
        disk.updated_at = Utc::now();
        disk
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
        value_labels: None,
        derived,
        source_bytes: None,
        source_modified: None,
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
        assert_eq!(bytes, 1_120_000); // 70,000 x 2 x 8 bytes
        m.count_bytes("a".into(), bytes);
        assert_eq!(
            m.memory_warning().as_deref(),
            Some(
                "estimated dataset memory 1.07 MiB exceeds the limit of 1.00 MiB (--max-memory 1)"
            )
        );
        m.count_bytes("a".into(), bytes); // replaced, not added
        assert_eq!(m.memory.borrow().tracker.current_usage(), 1_120_000);
        m.count_bytes("b".into(), bytes);
        assert_eq!(m.memory.borrow().tracker.current_usage(), 2_240_000);
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
        assert_eq!(sidecar_file_name("com0").unwrap(), "%63om0.parquet");
        assert_eq!(sidecar_file_name("lpt0").unwrap(), "%6Cpt0.parquet");
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
        // the refusal names the expected directory and the path found
        let mut meta = dataset_meta("d", PathBuf::new(), "parquet", &sample(), None, true);
        for bad in [
            "other.datasets/outside.parquet",
            "../outside.parquet",
            "s.json.datasets/x.csv",
        ] {
            meta.source_path = PathBuf::from(bad);
            assert_eq!(
                derived_file(&meta, Path::new("s.json"))
                    .unwrap_err()
                    .to_string(),
                format!(
                    "the stored copy of dataset 'd' must be under s.json.datasets/ (found {bad}); was the session file renamed or copied? Re-run the commands that created it"
                )
            );
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
    fn old_records_without_cwd_or_legacy_fields_parse() {
        let old = r#"{"id":"r","timestamp":"2026-01-01T00:00:00Z","command_line":"data list","category":"data","subcommand":"list","argv":["data","list"],"success":true,"error":null,"duration_ms":1}"#;
        let record: CommandRecord = serde_json::from_str(old).unwrap();
        assert_eq!(record.cwd, None);
        assert!(record.arguments.is_null() && record.dataset_refs.is_empty());
        let written = serde_json::to_value(&record).unwrap();
        for key in ["arguments", "dataset_refs", "output_files"] {
            assert!(
                written.get(key).is_some(),
                "{key} is written for older p2a versions"
            );
        }
        assert!(written.get("cwd").is_none());
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

#[cfg(test)]
mod save_tests {
    use super::*;

    /// A manager on `path` with one recorded command and a registered dataset `name`.
    fn record_one(m: &mut SessionManager, cmd: &str, name: &str, dir: &Path) {
        m.start_command();
        m.record_invocation(vec!["data".into(), cmd.into()], true, None);
        let ds = Dataset::new(polars::df! { "x" => [1.5, 2.25] }.unwrap());
        m.register_dataset(name, dir.join(format!("{name}.csv")), "csv", &ds, None);
    }

    fn commands(path: &Path) -> Vec<String> {
        let s = Session::load(&path.to_path_buf()).unwrap();
        s.commands.iter().map(|c| c.subcommand.clone()).collect()
    }

    #[test]
    fn save_merges_concurrent_invocations() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.json");
        let mut first = SessionManager::new(path.clone()).unwrap();
        first.record_invocation(vec!["data".into(), "original".into()], true, None);
        first.save().unwrap();

        let mut a = SessionManager::new(path.clone()).unwrap();
        let mut b = SessionManager::new(path.clone()).unwrap();
        record_one(&mut a, "a", "da", dir.path());
        record_one(&mut b, "b", "db", dir.path());
        a.save().unwrap();
        b.save().unwrap();

        assert_eq!(commands(&path), ["original", "a", "b"]);
        let s = Session::load(&path).unwrap();
        let mut names: Vec<&String> = s.datasets.keys().collect();
        names.sort();
        assert_eq!(names, ["da", "db"]);
        assert_eq!(s.id, first.session().id);
    }

    #[test]
    fn save_merges_into_a_file_created_meanwhile() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.json");
        let mut a = SessionManager::new(path.clone()).unwrap();
        let mut b = SessionManager::new(path.clone()).unwrap();
        record_one(&mut a, "a", "da", dir.path());
        record_one(&mut b, "b", "db", dir.path());
        a.save().unwrap();
        b.save().unwrap();
        assert_eq!(commands(&path), ["a", "b"]);
        let s = Session::load(&path).unwrap();
        assert_eq!(s.id, a.session().id);
        assert_eq!(s.created_at, a.session().created_at);
    }

    /// Two saves of one manager write its commands once (the second save re-reads the
    /// file a merge wrote and adds nothing twice).
    #[test]
    fn repeated_saves_do_not_duplicate_commands() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.json");
        let mut a = SessionManager::new(path.clone()).unwrap();
        let mut b = SessionManager::new(path.clone()).unwrap();
        record_one(&mut a, "a", "da", dir.path());
        record_one(&mut b, "b", "db", dir.path());
        a.save().unwrap();
        b.save().unwrap();
        b.save().unwrap();
        a.save().unwrap();
        record_one(&mut a, "c", "dc", dir.path());
        a.save().unwrap();
        assert_eq!(commands(&path), ["a", "b", "c"]);
    }

    #[test]
    fn save_is_atomic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.json");
        let mut m = SessionManager::new(path.clone()).unwrap();
        record_one(&mut m, "a", "da", dir.path());
        m.save().unwrap();
        let mut names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        assert_eq!(names, ["s.json", "s.json.lock"]);

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let before = std::fs::read(&path).unwrap();
            let mut m = SessionManager::new(path.clone()).unwrap();
            record_one(&mut m, "b", "db", dir.path());
            std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o555)).unwrap();
            let probe = dir.path().join("probe");
            let root = std::fs::write(&probe, b"x").is_ok();
            let saved = if root {
                let _ = std::fs::remove_file(&probe);
                None
            } else {
                Some(m.save())
            };
            std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
            if let Some(saved) = saved {
                assert!(
                    saved.is_err(),
                    "a save into a read-only directory must fail"
                );
                assert_eq!(std::fs::read(&path).unwrap(), before);
            }
        }
    }

    #[test]
    fn empty_or_blank_file_is_a_new_session() {
        let dir = tempfile::tempdir().unwrap();
        for (i, body) in ["", "\n  \n"].iter().enumerate() {
            let path = dir.path().join(format!("s{i}.json"));
            std::fs::write(&path, body).unwrap();
            let mut m = SessionManager::new(path.clone()).unwrap();
            assert!(m.session().commands.is_empty());
            assert!(
                Session::load(&path).is_err(),
                "load keeps refusing {body:?}"
            );
            assert!(read_session_file(&path).unwrap().0.is_none());
            record_one(&mut m, "a", "da", dir.path());
            m.save().unwrap();
            assert_eq!(commands(&path), ["a"]);
        }
    }

    /// The rule-5 shortcut: a file unchanged since `new` (length, modification time,
    /// inode) is not read again; the save uses the session in memory.
    #[test]
    fn unchanged_file_is_not_reread() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.json");
        let mut m = SessionManager::new(path.clone()).unwrap();
        record_one(&mut m, "a", "da", dir.path());
        m.save().unwrap();

        let mut m = SessionManager::new(path.clone()).unwrap();
        let meta = std::fs::metadata(&path).unwrap();
        let garbage = vec![b'!'; meta.len() as usize];
        let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        use std::io::Write as _;
        (&f).write_all(&garbage).unwrap();
        f.set_modified(meta.modified().unwrap()).unwrap();
        drop(f);
        record_one(&mut m, "b", "db", dir.path());
        m.save().unwrap();
        assert_eq!(commands(&path), ["a", "b"]);
    }

    #[cfg(unix)]
    #[test]
    fn save_keeps_a_symlinked_session_file() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real");
        std::fs::create_dir(&real).unwrap();
        let mut m = SessionManager::new(real.join("s.json")).unwrap();
        record_one(&mut m, "a", "da", dir.path());
        m.save().unwrap();
        let link = dir.path().join("s.json");
        std::os::unix::fs::symlink(real.join("s.json"), &link).unwrap();
        let mut m = SessionManager::new(link.clone()).unwrap();
        record_one(&mut m, "b", "db", dir.path());
        m.save().unwrap();
        assert!(
            std::fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(commands(&real.join("s.json")), ["a", "b"]);
        assert!(dir.path().join("s.json.lock").exists());
    }

    /// A dangling link is followed as `std::fs::write` follows it: the save creates the
    /// file the link names, and the link stays. Relative targets resolve against the
    /// link's directory, through a chain of links.
    #[cfg(unix)]
    #[test]
    fn save_creates_the_target_of_a_dangling_symlink() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real");
        std::fs::create_dir(&real).unwrap();
        let link = dir.path().join("s.json");
        std::os::unix::fs::symlink("hop.json", &link).unwrap();
        std::os::unix::fs::symlink("real/s.json", dir.path().join("hop.json")).unwrap();
        let mut m = SessionManager::new(link.clone()).unwrap();
        record_one(&mut m, "a", "da", dir.path());
        m.save().unwrap();
        assert!(
            std::fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(commands(&real.join("s.json")), ["a"]);
        assert_eq!(commands(&link), ["a"]);
    }

    /// A save that cannot create its lock or temporary file names the file it could not
    /// create, not only the operating system's error.
    #[cfg(unix)]
    #[test]
    fn a_failed_save_names_the_file() {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = tempfile::tempdir().unwrap();
        let ro = dir.path().join("ro");
        std::fs::create_dir(&ro).unwrap();
        let path = ro.join("s.json");
        let mut m = SessionManager::new(path.clone()).unwrap();
        record_one(&mut m, "a", "da", dir.path());
        std::fs::set_permissions(&ro, std::fs::Permissions::from_mode(0o555)).unwrap();
        if std::fs::File::create(ro.join("probe")).is_ok() {
            // running as root: the directory is writable regardless
            std::fs::set_permissions(&ro, std::fs::Permissions::from_mode(0o755)).unwrap();
            return;
        }
        let err = m.save().unwrap_err().to_string();
        std::fs::set_permissions(&ro, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(err.starts_with("could not lock "), "{err}");
        assert!(
            err.contains(&lock_path(&path).display().to_string()),
            "{err}"
        );

        // With the lock file in place, the temporary file is what fails.
        std::fs::write(lock_path(&path), b"").unwrap();
        std::fs::set_permissions(&ro, std::fs::Permissions::from_mode(0o555)).unwrap();
        let err = m.save().unwrap_err().to_string();
        std::fs::set_permissions(&ro, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(err.starts_with("could not save session file "), "{err}");
        assert!(err.contains(&path.display().to_string()), "{err}");
    }
}

#[cfg(test)]
mod reload_tests {
    use super::*;

    fn csv(path: &Path, rows: usize) -> Dataset {
        let mut s = String::from("x,y\n");
        for i in 0..rows {
            s.push_str(&format!("{}.5,{}.25\n", i, 2 * i + 1));
        }
        std::fs::write(path, s).unwrap();
        crate::commands::data::load_with_markers(path, None, None).unwrap()
    }

    #[test]
    fn register_dataset_stores_an_absolute_path_and_the_file_stamp() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("d.csv");
        let ds = csv(&file, 3);
        let mut m = SessionManager::new(dir.path().join("s.json")).unwrap();
        m.register_dataset("d", file.clone(), "csv", &ds, None);
        let meta = &m.session().datasets["d"];
        let fs_meta = std::fs::metadata(&file).unwrap();
        assert_eq!(meta.source_path, file);
        assert_eq!(meta.source_bytes, Some(fs_meta.len()));
        assert_eq!(
            meta.source_modified,
            Some(DateTime::<Utc>::from(fs_meta.modified().unwrap()))
        );

        let rel = PathBuf::from(format!("p2a_wp33_absent_{}.csv", std::process::id()));
        m.register_dataset("e", rel.clone(), "csv", &ds, None);
        let meta = &m.session().datasets["e"];
        assert_eq!(meta.source_path, std::path::absolute(&rel).unwrap());
        assert!(meta.source_path.is_absolute());
        assert_eq!((meta.source_bytes, meta.source_modified), (None, None));
    }

    #[test]
    fn legacy_relative_path_resolves_against_working_dir() {
        let dir = tempfile::tempdir().unwrap();
        csv(&dir.path().join("d.csv"), 3);
        let session = format!(
            r#"{{"id":"x","title":null,"created_at":"2026-01-01T00:00:00Z","updated_at":"2026-01-01T00:00:00Z","working_dir":{},"datasets":{{"d":{{"name":"d","source_path":"d.csv","format":"csv","nrows":3,"columns":["x","y"],"loaded_at":"2026-01-01T00:00:00Z"}}}},"commands":[],"version":"0.4.0"}}"#,
            serde_json::to_string(dir.path()).unwrap()
        );
        let path = dir.path().join("s.json");
        std::fs::write(&path, session).unwrap();
        let m = SessionManager::new(path).unwrap();
        assert_eq!(m.get_dataset("d").unwrap().df().height(), 3);
    }

    #[test]
    fn changed_source_is_reported_once() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("d.csv");
        let ds = csv(&file, 3);
        let path = dir.path().join("s.json");
        let mut m = SessionManager::new(path.clone()).unwrap();
        m.register_dataset("d", file.clone(), "csv", &ds, None);
        let meta = m.session().datasets["d"].clone();
        let wd = m.session().working_dir.clone();

        let (_, warnings) = reload_dataset(&meta, &path, &wd).unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");

        csv(&file, 4);
        let (again, warnings) = reload_dataset(&meta, &path, &wd).unwrap();
        assert_eq!(again.df().height(), 4, "the data are still used");
        assert_eq!(
            warnings,
            [
                format!(
                    "the file of dataset 'd' ({}) changed since it was loaded into the session (size or modification time differ); results may not match the recorded ones",
                    file.display()
                ),
                "dataset 'd' now has 4 rows and 2 columns; the session recorded 3 rows and 2 columns"
                    .to_string(),
            ]
        );

        // a derived dataset is not stamped and gives none
        m.persist_dataset("e", ds).unwrap();
        let meta = m.session().datasets["e"].clone();
        assert_eq!((meta.source_bytes, meta.source_modified), (None, None));
        let (_, warnings) = reload_dataset(&meta, &path, &wd).unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    #[test]
    fn record_invocation_records_the_working_directory() {
        let dir = tempfile::tempdir().unwrap();
        let mut m = SessionManager::new(dir.path().join("s.json")).unwrap();
        m.record_invocation(vec!["data".into(), "list".into()], true, None);
        assert_eq!(
            m.session().commands[0].cwd,
            Some(std::env::current_dir().unwrap())
        );
    }
}
