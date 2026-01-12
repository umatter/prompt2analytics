//! Session recording and management for reproducibility

use chrono::{DateTime, Utc};
use p2a_core::{Dataset, DataLoader};
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

/// Reload a dataset from its source file
fn reload_dataset(meta: &DatasetMeta) -> anyhow::Result<Dataset> {
    let df = match meta.format.as_str() {
        "csv" => DataLoader::load_csv(&meta.source_path)?,
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
        })
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
            dataset_refs,
            output_files,
            success,
            error,
            duration_ms,
        };

        self.session.record_command(record);
        self.command_start = None;
    }

    /// Register a loaded dataset
    pub fn register_dataset(&mut self, name: &str, source_path: PathBuf, format: &str, dataset: &Dataset) {
        let df = dataset.df();
        let meta = DatasetMeta {
            name: name.to_string(),
            source_path,
            format: format.to_string(),
            nrows: df.height(),
            columns: df.get_column_names().iter().map(|s| s.to_string()).collect(),
            loaded_at: Utc::now(),
        };
        self.session.register_dataset(meta);
    }

    /// Store a dataset in memory
    pub fn store_dataset(&mut self, name: String, dataset: Dataset) {
        self.datasets.insert(name, dataset);
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
