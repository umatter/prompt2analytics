//! Data management tool handlers.
//!
//! This module defines data loading, export, and inspection tools
//! using the `#[tool_router(router = data_router)]` pattern.
//!
//! Note: Cleaning session tools remain in server.rs due to their complexity.

use std::io::Write;
use std::path::{Path, PathBuf};

use rmcp::{
    ErrorData as McpError,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content},
    tool, tool_router,
};

use crate::path_jail;
use crate::server::AnalyticsServer;
use crate::tools::handlers::utils::CappedWriter;
use crate::tools::requests::data::*;

use p2a_core::data::{DataLoader, Dataset, DatasetInfo};
use p2a_core::polars::prelude::{AnyValue, Column, CsvWriter, DataFrame, ParquetWriter, SerWriter};
use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};

/// Rows serialised between two cancellation checkpoints in a JSON export.
const CHECKPOINT_ROWS: usize = 1 << 16;

/// The file formats `export_dataset` writes.
#[derive(Clone, Copy)]
enum ExportKind {
    Csv,
    Parquet,
    Json,
}

/// Why an export could not be written.
enum ExportError {
    /// The output passed the export_bytes budget.
    TooLarge,
    /// The writer failed (the detail may still name the file).
    Failed(String),
}

fn export_limit_message(cap: Option<u64>) -> String {
    format!(
        "The export is larger than this server's export limit of {} bytes; export fewer rows or \
         columns and try again.",
        cap.unwrap_or_default()
    )
}

/// Write `df` as `kind` to the new file `tmp` through a buffered writer
/// capped at `cap` bytes.
fn write_export(
    df: &DataFrame,
    kind: ExportKind,
    tmp: &Path,
    cap: Option<u64>,
) -> Result<(), ExportError> {
    let file = std::fs::File::create(tmp).map_err(|e| ExportError::Failed(e.to_string()))?;
    let mut w = CappedWriter::new(std::io::BufWriter::new(file), cap);
    let result = match kind {
        ExportKind::Csv => CsvWriter::new(&mut w)
            .finish(&mut df.clone())
            .map_err(|e| e.to_string()),
        ExportKind::Parquet => ParquetWriter::new(&mut w)
            .finish(&mut df.clone())
            .map(|_| ())
            .map_err(|e| e.to_string()),
        ExportKind::Json => {
            serde_json::to_writer_pretty(&mut w, &RecordsJson(df)).map_err(|e| e.to_string())
        }
    };
    let result = result.and_then(|()| w.flush().map_err(|e| e.to_string()));
    if w.exceeded() {
        return Err(ExportError::TooLarge);
    }
    result.map_err(ExportError::Failed)
}

/// A data frame serialised in the records layout of
/// `Dataset::to_json_string` (an array with one object per row, keys in
/// column order) without building it in memory. Values map as there, except
/// that text cells are plain JSON strings (the legacy path formatted them
/// with polars' `Display`, which wraps them in literal quote characters).
struct RecordsJson<'a>(&'a DataFrame);

impl Serialize for RecordsJson<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::Error as _;
        let df = self.0;
        let columns = df.get_columns();
        let mut seq = serializer.serialize_seq(Some(df.height()))?;
        for row in 0..df.height() {
            if row % CHECKPOINT_ROWS == 0 {
                p2a_core::resource::checkpoint().map_err(S::Error::custom)?;
            }
            seq.serialize_element(&RecordRow { columns, row })?;
        }
        seq.end()
    }
}

/// One row of [`RecordsJson`].
struct RecordRow<'a> {
    columns: &'a [Column],
    row: usize,
}

impl Serialize for RecordRow<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.columns.len()))?;
        for col in self.columns {
            map.serialize_entry(col.name().as_str(), &CellJson(col.get(self.row).ok()))?;
        }
        map.end()
    }
}

/// One cell of [`RecordsJson`]; `None` (a failed read) is null.
struct CellJson<'a>(Option<AnyValue<'a>>);

impl Serialize for CellJson<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match &self.0 {
            None | Some(AnyValue::Null) => s.serialize_none(),
            Some(AnyValue::Boolean(b)) => s.serialize_bool(*b),
            Some(AnyValue::Int8(v)) => s.serialize_i64(i64::from(*v)),
            Some(AnyValue::Int16(v)) => s.serialize_i64(i64::from(*v)),
            Some(AnyValue::Int32(v)) => s.serialize_i64(i64::from(*v)),
            Some(AnyValue::Int64(v)) => s.serialize_i64(*v),
            Some(AnyValue::UInt8(v)) => s.serialize_u64(u64::from(*v)),
            Some(AnyValue::UInt16(v)) => s.serialize_u64(u64::from(*v)),
            Some(AnyValue::UInt32(v)) => s.serialize_u64(u64::from(*v)),
            Some(AnyValue::UInt64(v)) => s.serialize_u64(*v),
            // serde_json writes non-finite floats as null, as the legacy export did.
            Some(AnyValue::Float32(v)) => s.serialize_f64(f64::from(*v)),
            Some(AnyValue::Float64(v)) => s.serialize_f64(*v),
            Some(AnyValue::String(v)) => s.serialize_str(v),
            Some(AnyValue::StringOwned(v)) => s.serialize_str(v.as_str()),
            Some(other) => s.serialize_str(&format!("{other}")),
        }
    }
}

/// `text` (an error from a loader or writer) with the absolute forms of
/// `path` and its folder replaced by their [`path_jail::display_path`] forms,
/// so no server path reaches the caller.
fn without_server_paths(text: &str, path: &Path) -> String {
    let mut out = text.replace(&path.display().to_string(), &path_jail::display_path(path));
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        out = out.replace(
            &parent.display().to_string(),
            &path_jail::display_path(parent),
        );
    }
    out
}

#[tool_router(router = data_router, vis = "pub")]
impl AnalyticsServer {
    // ========================================================================
    // Dataset Loading/Saving Tools
    // ========================================================================

    /// List all currently loaded datasets.
    #[tool(
        description = "List all currently loaded datasets with their basic information (name, dimensions, column types)."
    )]
    pub async fn list_datasets(&self) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        if datasets.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No datasets currently loaded. Use the 'load_dataset' tool to load a data file.",
            )]));
        }

        let mut result = String::from("Loaded Datasets:\n\n");
        for (id, dataset) in datasets.iter() {
            let info: DatasetInfo = dataset.into();
            result.push_str(&format!(
                "- **{}**: {} rows x {} columns\n",
                id, info.nrows, info.ncols
            ));
            result.push_str("  Columns: ");
            let col_summary: Vec<String> = info
                .columns
                .iter()
                .map(|c| format!("{} ({})", c.name, c.dtype))
                .collect();
            result.push_str(&col_summary.join(", "));
            result.push_str("\n\n");
        }

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    /// Load a dataset from a file.
    #[tool(
        description = "Load a dataset from a file. Supports CSV, Parquet, Excel (xlsx, xls, xlsb, ods), Stata (dta), and SAS (sas7bdat) formats. Returns dataset information including dimensions and column types."
    )]
    pub async fn load_dataset(
        &self,
        Parameters(request): Parameters<LoadDatasetRequest>,
    ) -> Result<CallToolResult, McpError> {
        let path = match path_jail::validate_data_path(&request.path) {
            Ok(p) => p,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Refused to load dataset: {}",
                    e
                ))]));
            }
        };

        // The configurable maximum file size: checked on disk before any
        // parsing, only when the limits set a load_file_bytes budget (hosted
        // 64 MiB, none when limits are off, P2A_LIMIT_OVERRIDES in any mode).
        let limits = crate::tools::limits::current();
        if limits.load_file_bytes.is_some() {
            let size = match std::fs::metadata(&path) {
                Ok(meta) => meta.len(),
                Err(_) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Failed to load dataset: File not found: {}",
                        path_jail::display_path(&path)
                    ))]));
                }
            };
            if let Err(v) = limits.check_budget(
                "load_dataset",
                "load_file_bytes",
                size,
                limits.load_file_bytes,
            ) {
                tracing::info!(
                    tool = "load_dataset",
                    override_key = v.override_key,
                    requested = %v.requested,
                    limit = %v.limit,
                    "request refused by a named budget"
                );
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "{} Load a smaller file, or split it and load the parts.",
                    v.message
                ))]));
            }
        }

        // Load the dataset
        let dataset = match DataLoader::load(&path) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to load dataset: {}",
                    without_server_paths(&e.to_string(), &path)
                ))]));
            }
        };

        // Generate an ID for the dataset
        let id = request.name.unwrap_or_else(|| {
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("dataset")
                .to_string()
        });

        // Get info before moving
        let info: DatasetInfo = (&dataset).into();

        // Track memory usage
        {
            let mut profiler = self.memory_profiler.write().await;
            profiler.track_dataset(&id, &dataset);
        }

        // Store the dataset
        let mut datasets = self.datasets.write().await;
        datasets.insert(id.clone(), dataset);

        let result = format!(
            "Successfully loaded dataset '{}'\n\n\
             Dimensions: {} rows x {} columns\n\n\
             Columns:\n{}",
            id,
            info.nrows,
            info.ncols,
            info.columns
                .iter()
                .map(|c| format!(
                    "  - {} ({}): {} nulls ({:.1}%)",
                    c.name,
                    c.dtype,
                    c.null_count,
                    if info.nrows > 0 {
                        (c.null_count as f64 / info.nrows as f64) * 100.0
                    } else {
                        0.0
                    }
                ))
                .collect::<Vec<_>>()
                .join("\n")
        );

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    /// Export a dataset to a file.
    #[tool(
        description = "Export a loaded dataset to a file. Supports CSV, Parquet, and JSON formats. The format is determined by the file extension or can be explicitly specified."
    )]
    pub async fn export_dataset(
        &self,
        Parameters(request): Parameters<ExportDatasetRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let path = match path_jail::validate_data_path_for_write(&request.path) {
            Ok(p) => p,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Refused to export dataset: {}",
                    e
                ))]));
            }
        };

        // Determine format from extension or explicit parameter
        let format = request
            .format
            .as_deref()
            .or_else(|| path.extension().and_then(|ext| ext.to_str()));

        let (kind, label) = match format {
            Some("csv") => (ExportKind::Csv, "CSV"),
            Some("parquet") => (ExportKind::Parquet, "Parquet"),
            Some("json") => (ExportKind::Json, "JSON"),
            Some(other) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unsupported format '{}'. Use 'csv', 'parquet', or 'json'.",
                    other
                ))]));
            }
            None => {
                return Ok(CallToolResult::error(vec![Content::text(
                    "Could not determine output format. Please specify a format or use a file extension.",
                )]));
            }
        };

        // Written to a temporary file in the validated folder, capped at the
        // export_bytes budget, then renamed into place (path_jail checks the
        // session storage quota before the rename).
        let cap = crate::tools::limits::current().export_bytes;
        let written = path_jail::write_atomically(&path, |tmp| {
            write_export(dataset.df(), kind, tmp, cap).map_err(|e| match e {
                ExportError::TooLarge => export_limit_message(cap),
                ExportError::Failed(detail) => format!(
                    "Failed to export to {label}: {}",
                    without_server_paths(&without_server_paths(&detail, tmp), &path)
                ),
            })
        });
        if let Err(e) = written {
            return Ok(CallToolResult::error(vec![Content::text(e)]));
        }

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Successfully exported dataset '{}' to {label} file:\n  {}\n  {} rows x {} columns",
            request.dataset,
            path_jail::display_path(&path),
            dataset.nrows(),
            dataset.ncols()
        ))]))
    }

    /// Upload and load a dataset from base64-encoded content.
    #[tool(
        description = "Upload and load a dataset from base64-encoded content. Useful when the client cannot access the filesystem directly. Supports CSV and Parquet formats."
    )]
    pub async fn upload_dataset(
        &self,
        Parameters(request): Parameters<UploadDatasetRequest>,
    ) -> Result<CallToolResult, McpError> {
        use base64::Engine;

        // Decode base64
        let decoded = match base64::engine::general_purpose::STANDARD.decode(&request.content) {
            Ok(d) => d,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to decode base64 content: {}",
                    e
                ))]));
            }
        };

        // Determine format from filename extension
        let path = PathBuf::from(&request.filename);
        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("csv");

        // Load based on format
        let dataset = match extension.to_lowercase().as_str() {
            "csv" => {
                // Convert bytes to string and use from_csv_string
                let csv_str = match String::from_utf8(decoded.clone()) {
                    Ok(s) => s,
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Failed to decode CSV as UTF-8: {}",
                            e
                        ))]));
                    }
                };
                match DataLoader::from_csv_string(&csv_str) {
                    Ok(df) => Dataset::new(df),
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Failed to parse CSV: {}",
                            e
                        ))]));
                    }
                }
            }
            "parquet" => {
                // Write to temp file and load (Parquet is binary format)
                let temp_dir = std::env::temp_dir();
                let temp_path = temp_dir.join(format!("upload_{}.parquet", uuid::Uuid::new_v4()));

                if let Err(e) =
                    std::fs::File::create(&temp_path).and_then(|mut f| f.write_all(&decoded))
                {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Failed to write temp file: {}",
                        e
                    ))]));
                }

                let result = DataLoader::load(&temp_path);
                let _ = std::fs::remove_file(&temp_path); // Clean up temp file

                match result {
                    Ok(ds) => ds,
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Failed to parse Parquet: {}",
                            e
                        ))]));
                    }
                }
            }
            other => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unsupported file format '{}'. Upload supports CSV and Parquet only.",
                    other
                ))]));
            }
        };

        // Generate ID
        let id = request.name.unwrap_or_else(|| {
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("uploaded")
                .to_string()
        });

        // Get info
        let info: DatasetInfo = (&dataset).into();

        // Store
        let mut datasets = self.datasets.write().await;
        datasets.insert(id.clone(), dataset);

        let result = format!(
            "Successfully uploaded dataset '{}'\n\n\
             Source: {} ({} bytes decoded)\n\
             Dimensions: {} rows x {} columns\n\n\
             Columns:\n{}",
            id,
            request.filename,
            decoded.len(),
            info.nrows,
            info.ncols,
            info.columns
                .iter()
                .map(|c| format!("  - {} ({})", c.name, c.dtype))
                .collect::<Vec<_>>()
                .join("\n")
        );

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    /// Create a dataset from inline CSV content.
    #[tool(
        description = "Create a dataset directly from CSV content provided as a string. Useful for creating small datasets without file I/O."
    )]
    pub async fn create_dataset(
        &self,
        Parameters(request): Parameters<CreateDatasetRequest>,
    ) -> Result<CallToolResult, McpError> {
        // from_csv_string returns DataFrame, wrap in Dataset
        let df = match DataLoader::from_csv_string(&request.csv_content) {
            Ok(df) => df,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to parse CSV content: {}",
                    e
                ))]));
            }
        };
        let dataset = Dataset::new(df);

        let info: DatasetInfo = (&dataset).into();

        let mut datasets = self.datasets.write().await;
        datasets.insert(request.name.clone(), dataset);

        let result = format!(
            "Successfully created dataset '{}'\n\n\
             Dimensions: {} rows x {} columns\n\n\
             Columns:\n{}",
            request.name,
            info.nrows,
            info.ncols,
            info.columns
                .iter()
                .map(|c| format!("  - {} ({})", c.name, c.dtype))
                .collect::<Vec<_>>()
                .join("\n")
        );

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    // ========================================================================
    // Dataset Inspection Tools
    // ========================================================================

    /// Describe a loaded dataset with summary statistics.
    #[tool(
        description = "Get summary statistics for a loaded dataset including column types, null counts, and basic statistics for numeric columns."
    )]
    pub async fn describe_dataset(
        &self,
        Parameters(request): Parameters<DescribeDatasetRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let info: DatasetInfo = dataset.into();
        let result = format!(
            "Dataset: {}\n\
             Dimensions: {} rows x {} columns\n\n\
             Columns:\n{}",
            request.dataset,
            info.nrows,
            info.ncols,
            info.columns
                .iter()
                .map(|c| format!(
                    "  - {} ({}): {} nulls ({:.1}%)",
                    c.name,
                    c.dtype,
                    c.null_count,
                    if info.nrows > 0 {
                        (c.null_count as f64 / info.nrows as f64) * 100.0
                    } else {
                        0.0
                    }
                ))
                .collect::<Vec<_>>()
                .join("\n")
        );

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    /// Preview the first rows of a dataset.
    #[tool(
        description = "Preview the first N rows of a loaded dataset (default: 5 rows). Returns data in a formatted table."
    )]
    pub async fn head_dataset(
        &self,
        Parameters(request): Parameters<HeadDatasetRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let n = request.n.unwrap_or(5);
        let df = dataset.df();
        let preview = df.head(Some(n));

        let result = format!(
            "Dataset: {} (showing first {} of {} rows)\n\n{}",
            request.dataset,
            n.min(df.height()),
            df.height(),
            preview
        );

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p2a_core::polars::prelude::df;
    use serde_json::json;

    #[test]
    fn path_schemas_say_relative_to_data_directory() {
        for schema in [
            serde_json::to_string(&schemars::schema_for!(LoadDatasetRequest)).unwrap(),
            serde_json::to_string(&schemars::schema_for!(ExportDatasetRequest)).unwrap(),
        ] {
            assert!(
                schema.contains("relative to the server's data directory"),
                "{schema}"
            );
        }
    }

    #[test]
    fn records_json_matches_to_json_string_except_string_quoting() {
        let df = df! {
            "f" => [Some(1.5), Some(f64::NAN), None],
            "i" => [Some(1i64), None, Some(3)],
            "s" => [Some("a"), None, Some("c\"q")],
            "b" => [Some(true), Some(false), None],
            "g" => [Some(0.1f32), None, Some(2.0f32)]
        }
        .unwrap();
        let mut legacy: serde_json::Value =
            serde_json::from_str(&Dataset::new(df.clone()).to_json_string().unwrap()).unwrap();
        let streamed: serde_json::Value =
            serde_json::from_slice(&serde_json::to_vec_pretty(&RecordsJson(&df)).unwrap()).unwrap();
        // Strings: plain JSON strings now (legacy wrapped them in literal quotes).
        let s: Vec<serde_json::Value> = streamed
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["s"].clone())
            .collect();
        assert_eq!(s, vec![json!("a"), json!(null), json!("c\"q")]);
        // Everything else: identical to the legacy export.
        let rows_old = legacy.as_array_mut().unwrap();
        let rows_new = streamed.as_array().unwrap();
        assert_eq!(rows_new.len(), rows_old.len());
        for (row_new, row_old) in rows_new.iter().zip(rows_old) {
            let mut row_new = row_new.clone();
            row_new.as_object_mut().unwrap().remove("s");
            row_old.as_object_mut().unwrap().remove("s");
            assert_eq!(&row_new, row_old);
        }
    }
}
