//! Data management tool handlers.
//!
//! This module defines data loading, export, and inspection tools
//! using the `#[tool_router(router = data_router)]` pattern.
//!
//! Note: Cleaning session tools remain in server.rs due to their complexity.

use std::io::Write;
use std::path::PathBuf;

use rmcp::{
    ErrorData as McpError,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content},
    tool, tool_router,
};

use crate::path_jail;
use crate::server::AnalyticsServer;
use crate::tools::requests::data::*;

use p2a_core::data::{DataLoader, Dataset, DatasetInfo};

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

        // Load the dataset
        let dataset = match DataLoader::load(&path) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to load dataset: {}",
                    e
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

        let path = match path_jail::validate_data_path(&request.path) {
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

        let result_msg = match format {
            Some("csv") => match dataset.to_csv(&path) {
                Ok(()) => format!(
                    "Successfully exported dataset '{}' to CSV file:\n  {}\n  {} rows x {} columns",
                    request.dataset,
                    path.display(),
                    dataset.nrows(),
                    dataset.ncols()
                ),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Failed to export to CSV: {}",
                        e
                    ))]));
                }
            },
            Some("parquet") => match dataset.to_parquet(&path) {
                Ok(()) => format!(
                    "Successfully exported dataset '{}' to Parquet file:\n  {}\n  {} rows x {} columns",
                    request.dataset,
                    path.display(),
                    dataset.nrows(),
                    dataset.ncols()
                ),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Failed to export to Parquet: {}",
                        e
                    ))]));
                }
            },
            Some("json") => {
                // to_json_string() returns String, so we write it to file manually
                match dataset.to_json_string() {
                    Ok(json_str) => {
                        if let Err(e) = std::fs::write(&path, json_str) {
                            return Ok(CallToolResult::error(vec![Content::text(format!(
                                "Failed to write JSON file: {}",
                                e
                            ))]));
                        }
                        format!(
                            "Successfully exported dataset '{}' to JSON file:\n  {}\n  {} rows x {} columns",
                            request.dataset,
                            path.display(),
                            dataset.nrows(),
                            dataset.ncols()
                        )
                    }
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Failed to export to JSON: {}",
                            e
                        ))]));
                    }
                }
            }
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

        Ok(CallToolResult::success(vec![Content::text(result_msg)]))
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
