//! Request types for data management tools (load, export, describe, etc.).

use schemars::JsonSchema;
use serde::Deserialize;

// ============================================================================
// Dataset Loading/Saving Requests
// ============================================================================

/// Request to load a dataset from a file.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct LoadDatasetRequest {
    /// Path to the data file
    #[schemars(
        description = "Path of the data file, relative to the server's data directory (for example 'sales.csv' or 'raw/sales.csv'). An absolute path must lie inside that directory. Supports CSV, Parquet, Excel (xlsx, xls, xlsb, ods), Stata (dta), and SAS (sas7bdat) formats."
    )]
    pub path: String,

    /// Optional name/identifier for the dataset
    #[schemars(
        description = "Optional name to identify this dataset. If not provided, the filename will be used."
    )]
    pub name: Option<String>,

    /// Strings read as missing values (CSV only).
    #[serde(default)]
    #[schemars(
        description = "Strings read as missing values when the data is CSV; default [\"NA\", \"\"] (R's read.csv). Unquoted empty fields are always missing. Pass [] to keep the text NA as a value. Not allowed for non-CSV files."
    )]
    pub na_strings: Option<Vec<String>>,

    /// Declared file type (the web UI's File Type selector).
    #[schemars(
        description = "Optional declared file type: 'auto' (default), 'csv', 'parquet', 'excel', 'stata' or 'sas'. The file is read by its extension; a declared type that the extension contradicts is refused, so a mislabelled file is never parsed."
    )]
    pub file_type: Option<String>,
}

/// Request to upload and load a dataset from base64-encoded content.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct UploadDatasetRequest {
    /// Base64-encoded file content
    #[schemars(description = "The file content encoded as base64")]
    pub content: String,

    /// Original filename (used to determine format and default name)
    #[schemars(description = "Original filename including extension (e.g., 'data.csv')")]
    pub filename: String,

    /// Optional name/identifier for the dataset
    #[schemars(
        description = "Optional name to identify this dataset. If not provided, the filename will be used."
    )]
    pub name: Option<String>,

    /// Declared file type (the web UI's File Type selector).
    #[schemars(
        description = "Optional declared file type: 'auto' (default), 'csv', 'parquet', 'excel', 'stata' or 'sas'. The file is read by its extension; a declared type that the extension contradicts is refused, so a mislabelled file is never parsed."
    )]
    pub file_type: Option<String>,
}

/// Request to create a dataset from inline CSV content.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CreateDatasetRequest {
    /// Name/identifier for the dataset
    #[schemars(description = "Name to identify this dataset (e.g., 'my_data')")]
    pub name: String,

    /// CSV content as plain text
    #[schemars(description = "CSV content with headers in first row (e.g., 'x,y\\n1,2\\n3,4')")]
    pub csv_content: String,

    /// Strings read as missing values in `csv_content`.
    #[serde(default)]
    #[schemars(
        description = "Strings read as missing values; default [\"NA\", \"\"] (R's read.csv). Unquoted empty fields are always missing. Pass [] to keep the text NA as a value."
    )]
    pub na_strings: Option<Vec<String>>,
}

/// Request to export a dataset to a file.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExportDatasetRequest {
    /// Name/ID of the dataset to export
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Output file path
    #[schemars(
        description = "Where to save the file, relative to the server's data directory (for example 'results.csv'). The format is determined by extension: .csv, .parquet, .json. An existing file with the same name is replaced."
    )]
    pub path: String,

    /// Output format (optional, inferred from extension if not specified)
    #[schemars(
        description = "Output format: 'csv', 'parquet', or 'json'. If not specified, inferred from file extension."
    )]
    pub format: Option<String>,
}

// ============================================================================
// Dataset Inspection Requests
// ============================================================================

/// Request to describe a loaded dataset.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DescribeDatasetRequest {
    /// Name/ID of the dataset to describe
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,
}

/// Request to preview rows from a dataset.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct HeadDatasetRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Number of rows to return (default: 5)
    #[schemars(description = "Number of rows to return. Default is 5.")]
    pub n: Option<usize>,
}

// ============================================================================
// Data Quality and Cleaning Requests
// ============================================================================

// ============================================================================
// Cleaning Session Requests
// ============================================================================

// ============================================================================
// Descriptive Statistics Requests
// ============================================================================
