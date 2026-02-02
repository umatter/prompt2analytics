//! Request types for utility tools (seed management, reports, session export/import).

use schemars::JsonSchema;
use serde::Deserialize;

// ============================================================================
// Seed Management Requests
// ============================================================================

/// Request to set the global random seed for ML reproducibility.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SetSeedRequest {
    /// The random seed value
    #[schemars(description = "The random seed value. Set to null/omit to clear the global seed.")]
    pub seed: Option<u64>,
}

/// Request to get the current global seed.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetSeedRequest {}

// ============================================================================
// Random Data Generation Requests
// ============================================================================

/// Column specification for random data generation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ColumnSpecInput {
    /// Name of the column
    #[schemars(description = "Name of the column to generate.")]
    pub name: String,

    /// Distribution type and parameters
    #[schemars(
        description = "Distribution specification. Must include 'type' field and distribution-specific parameters."
    )]
    pub distribution: serde_json::Value,
}

/// Request to generate random data.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GenerateRandomDataRequest {
    /// Number of rows to generate
    #[schemars(description = "Number of rows to generate.")]
    pub n_rows: usize,

    /// Column specifications
    #[schemars(
        description = "Array of column specifications. Each must have 'name' and 'distribution' fields. Distribution types: 'uniform' (min, max), 'normal' (mean, std), 'binomial' (n, p), 'poisson' (lambda), 'exponential' (rate), 'bernoulli' (p), 'categorical' (categories, optional weights), 'uniform_int' (min, max), 'sequence' (start), 'constant' (value), 'constant_string' (value)."
    )]
    pub columns: Vec<ColumnSpecInput>,

    /// Random seed for reproducibility
    #[schemars(description = "Optional random seed for reproducible results.")]
    pub seed: Option<u64>,

    /// Name to assign to the generated dataset
    #[schemars(description = "Name to assign to the generated dataset. Defaults to 'generated'.")]
    pub name: Option<String>,
}

// ============================================================================
// Report Generation Requests
// ============================================================================

/// A section in the HTML report.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReportSectionInput {
    /// Section title
    #[schemars(description = "Title for this section of the report.")]
    pub title: String,

    /// Content items for the section
    #[schemars(description = "Content items to include in this section.")]
    pub content: Vec<ReportContentInput>,
}

/// Content item for a report section.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReportContentInput {
    /// Type of content: 'text', 'code', 'table', 'chart', or 'stats'
    #[schemars(
        description = "Type of content: 'text' (paragraph), 'code' (code block), 'table' (data table), 'chart' (base64 image), or 'stats' (key-value pairs)."
    )]
    pub content_type: String,

    /// Text content (for text and code types)
    #[schemars(description = "Text content for 'text' or 'code' types.")]
    pub text: Option<String>,

    /// Programming language (for code blocks)
    #[schemars(description = "Programming language for code block syntax highlighting.")]
    pub language: Option<String>,

    /// Table headers (for table type)
    #[schemars(description = "Column headers for table content.")]
    pub headers: Option<Vec<String>>,

    /// Table rows (for table type) - each row is a list of cell values
    #[schemars(description = "Table rows, where each row is a list of string values.")]
    pub rows: Option<Vec<Vec<String>>>,

    /// Table caption
    #[schemars(description = "Caption for the table.")]
    pub caption: Option<String>,

    /// Base64-encoded chart image (for chart type)
    #[schemars(description = "Base64-encoded PNG image data for chart content.")]
    pub image_base64: Option<String>,

    /// Chart title
    #[schemars(description = "Title for the chart.")]
    pub chart_title: Option<String>,

    /// Chart caption
    #[schemars(description = "Caption for the chart.")]
    pub chart_caption: Option<String>,

    /// Key-value statistics (for stats type)
    #[schemars(
        description = "Key-value pairs for statistics display. Format: [[key, value], ...]"
    )]
    pub stats: Option<Vec<Vec<String>>>,
}

/// Request to generate an HTML report.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GenerateReportRequest {
    /// Report title
    #[schemars(description = "Title for the report.")]
    pub title: String,

    /// Report subtitle (optional)
    #[schemars(description = "Optional subtitle or description for the report.")]
    pub subtitle: Option<String>,

    /// Author name (optional)
    #[schemars(description = "Optional author name.")]
    pub author: Option<String>,

    /// Report sections
    #[schemars(description = "Sections to include in the report.")]
    pub sections: Vec<ReportSectionInput>,
}

// ============================================================================
// Session Export/Import Requests
// ============================================================================

/// Request to export the current analysis session.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExportSessionRequest {
    /// Path to save the session file
    #[schemars(
        description = "File path to save the session (JSON format). If not provided, returns session data as string."
    )]
    pub file_path: Option<String>,

    /// Whether to include dataset data (default: true)
    #[schemars(
        description = "Include full dataset data. If false, only metadata and file paths are saved."
    )]
    pub include_data: Option<bool>,
}

/// Request to import a previously exported session.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ImportSessionRequest {
    /// Path to the session file to import
    #[schemars(description = "File path to the session JSON file to import.")]
    pub file_path: String,

    /// Whether to merge with existing session (default: false, replaces)
    #[schemars(description = "If true, merges with existing datasets instead of replacing.")]
    pub merge: Option<bool>,
}
