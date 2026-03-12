//! Request types for munging tools.

use schemars::JsonSchema;
use serde::Deserialize;

/// Request to batch process multiple datasets with the same operation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct BatchProcessRequest {
    /// Names/IDs of datasets to process
    #[schemars(description = "List of dataset names to process. Each must be previously loaded.")]
    pub datasets: Vec<String>,

    /// Operation to perform on each dataset
    #[schemars(
        description = "Operation to perform: 'describe' (summary stats), 'correlation' (correlation matrix), or 'ols' (regression)."
    )]
    pub operation: String,

    /// Columns to analyze (optional, defaults to all numeric for describe/correlation)
    #[schemars(
        description = "List of column names to analyze. For 'ols', first column is dependent variable."
    )]
    pub columns: Option<Vec<String>>,

    /// Whether to return combined summary across all datasets
    #[schemars(description = "If true, also returns an aggregated summary across all datasets.")]
    pub combine_results: Option<bool>,
}

/// Request to compare the same columns across multiple datasets.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CompareDatasetRequest {
    /// Names/IDs of datasets to compare
    #[schemars(description = "List of dataset names to compare. Each must be previously loaded.")]
    pub datasets: Vec<String>,

    /// Columns to compare
    #[schemars(description = "List of column names to compare across datasets.")]
    pub columns: Vec<String>,

    /// Type of comparison
    #[schemars(
        description = "Comparison type: 'summary' (side-by-side stats), 'correlation' (correlation differences), or 'distribution' (distribution comparison)."
    )]
    pub comparison_type: Option<String>,
}

/// Request to filter rows in a dataset.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FilterDatasetRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset to filter.")]
    pub dataset: String,

    /// Column to filter on
    #[schemars(description = "Name of the column to filter on.")]
    pub column: String,

    /// Comparison operator
    #[schemars(
        description = "Comparison operator: 'eq', 'ne', 'gt', 'ge', 'lt', 'le', 'contains', 'starts_with', 'ends_with'."
    )]
    pub op: String,

    /// Value to compare against
    #[schemars(
        description = "Value to compare against (as string, will be parsed based on column type)."
    )]
    pub value: String,

    /// Optional name for the resulting dataset
    #[schemars(
        description = "Optional name for the filtered result. If not provided, overwrites the source dataset."
    )]
    pub result_name: Option<String>,
}

/// Request to select columns from a dataset.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SelectColumnsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Columns to select
    #[schemars(description = "List of column names to keep.")]
    pub columns: Vec<String>,

    /// Optional name for the resulting dataset
    #[schemars(
        description = "Optional name for the result. If not provided, overwrites the source dataset."
    )]
    pub result_name: Option<String>,
}

/// Request to drop columns from a dataset.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DropColumnsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Columns to drop
    #[schemars(description = "List of column names to drop.")]
    pub columns: Vec<String>,

    /// Optional name for the resulting dataset
    #[schemars(
        description = "Optional name for the result. If not provided, overwrites the source dataset."
    )]
    pub result_name: Option<String>,
}

/// Request to rename columns in a dataset.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RenameColumnsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Mapping of old names to new names
    #[schemars(
        description = "Mapping of old column names to new names as pairs: [[\"old1\", \"new1\"], [\"old2\", \"new2\"]]."
    )]
    pub renames: Vec<Vec<String>>,

    /// Optional name for the resulting dataset
    #[schemars(
        description = "Optional name for the result. If not provided, overwrites the source dataset."
    )]
    pub result_name: Option<String>,
}

/// Request to sort a dataset.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SortDatasetRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Columns to sort by
    #[schemars(description = "List of column names to sort by.")]
    pub by: Vec<String>,

    /// Sort in descending order
    #[schemars(description = "If true, sort in descending order. Default is ascending.")]
    pub descending: Option<bool>,

    /// Optional name for the resulting dataset
    #[schemars(
        description = "Optional name for the result. If not provided, overwrites the source dataset."
    )]
    pub result_name: Option<String>,
}

/// Request to join two datasets.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct JoinDatasetsRequest {
    /// Name/ID of the left dataset
    #[schemars(description = "Name or ID of the left dataset.")]
    pub left: String,

    /// Name/ID of the right dataset
    #[schemars(description = "Name or ID of the right dataset.")]
    pub right: String,

    /// Columns to join on (from left dataset)
    #[schemars(description = "Column names from the left dataset to join on.")]
    pub left_on: Vec<String>,

    /// Columns to join on (from right dataset)
    #[schemars(
        description = "Column names from the right dataset to join on. If not provided, uses left_on."
    )]
    pub right_on: Option<Vec<String>>,

    /// Type of join
    #[schemars(description = "Join type: 'left', 'right', 'inner', or 'full'. Default is 'left'.")]
    pub join_type: Option<String>,

    /// Suffix for duplicate column names
    #[schemars(
        description = "Suffix to add to duplicate column names from the right dataset. Default is '_right'."
    )]
    pub suffix: Option<String>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the joined result.")]
    pub result_name: Option<String>,
}

/// Request to concatenate datasets.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ConcatDatasetsRequest {
    /// Names/IDs of datasets to concatenate
    #[schemars(description = "List of dataset names to concatenate vertically (row-bind).")]
    pub datasets: Vec<String>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the concatenated result.")]
    pub result_name: Option<String>,
}

/// Request to group and aggregate a dataset.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GroupByRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Columns to group by
    #[schemars(description = "Column names to group by.")]
    pub by: Vec<String>,

    /// Aggregation specifications
    #[schemars(
        description = "Aggregation specs as [[\"column\", \"function\"], ...]. Functions: 'count', 'sum', 'mean', 'median', 'min', 'max', 'std', 'var', 'first', 'last'."
    )]
    pub aggs: Vec<Vec<String>>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the grouped result.")]
    pub result_name: Option<String>,
}

/// Request to compute value counts for a column.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ValueCountsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to count values in
    #[schemars(description = "Column name to compute value counts for.")]
    pub column: String,

    /// Whether to normalize to percentages
    #[schemars(description = "If true, return percentages instead of counts.")]
    pub normalize: Option<bool>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to pivot a dataset from long to wide format.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PivotDatasetRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Index columns (will remain as rows)
    #[schemars(description = "Column names to use as index (will remain as rows).")]
    pub index: Vec<String>,

    /// Column whose values become new column names
    #[schemars(description = "Column whose values become new column names.")]
    pub on: String,

    /// Column containing values to fill the new columns
    #[schemars(description = "Column containing values to fill the new columns.")]
    pub values: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the pivoted result.")]
    pub result_name: Option<String>,
}

/// Request to melt a dataset from wide to long format.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MeltDatasetRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// ID columns to keep as-is
    #[schemars(description = "Column names to keep as identifier variables.")]
    pub id_vars: Vec<String>,

    /// Value columns to unpivot
    #[schemars(description = "Column names to unpivot into rows.")]
    pub value_vars: Vec<String>,

    /// Name for the variable column
    #[schemars(description = "Name for the new variable column. Default is 'variable'.")]
    pub variable_name: Option<String>,

    /// Name for the value column
    #[schemars(description = "Name for the new value column. Default is 'value'.")]
    pub value_name: Option<String>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the melted result.")]
    pub result_name: Option<String>,
}

/// Request to drop rows with null values.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DropNaRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Columns to check for nulls
    #[schemars(
        description = "Column names to check for nulls. If not provided, checks all columns."
    )]
    pub columns: Option<Vec<String>>,

    /// How to drop rows
    #[schemars(
        description = "How to drop: 'any' (drop if any null) or 'all' (drop only if all null). Default is 'any'."
    )]
    pub how: Option<String>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to fill null values.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FillNaRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Columns to fill nulls in
    #[schemars(description = "Column names to fill nulls in. If not provided, fills all columns.")]
    pub columns: Option<Vec<String>>,

    /// Fill strategy
    #[schemars(
        description = "Fill strategy: 'mean', 'median', 'mode', 'forward', 'backward', or a constant value."
    )]
    pub strategy: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to remove duplicate rows.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DeduplicateRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Columns to check for duplicates
    #[schemars(
        description = "Column names to check for duplicates. If not provided, checks all columns."
    )]
    pub columns: Option<Vec<String>>,

    /// Which duplicate to keep
    #[schemars(
        description = "Which duplicate to keep: 'first', 'last', or 'none'. Default is 'first'."
    )]
    pub keep: Option<String>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to trim whitespace from string columns.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct TrimRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Columns to trim
    #[schemars(description = "Column names to trim. If not provided, trims all string columns.")]
    pub columns: Option<Vec<String>>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to convert a string column to lowercase.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ToLowercaseRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to convert
    #[schemars(description = "Name of the string column to convert to lowercase.")]
    pub column: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to convert a string column to uppercase.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ToUppercaseRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to convert
    #[schemars(description = "Name of the string column to convert to uppercase.")]
    pub column: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to replace exact values in a column.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReplaceValueRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to modify
    #[schemars(description = "Name of the column to modify.")]
    pub column: String,

    /// Value to find
    #[schemars(description = "Exact value to search for and replace.")]
    pub old_value: String,

    /// Replacement value
    #[schemars(description = "Value to replace matches with.")]
    pub new_value: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to replace substrings matching a regex pattern.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RegexReplaceRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to modify
    #[schemars(description = "Name of the string column to modify.")]
    pub column: String,

    /// Regex pattern
    #[schemars(description = "Regular expression pattern to match.")]
    pub pattern: String,

    /// Replacement string
    #[schemars(description = "Replacement string. Use $1, $2, etc. for capture groups.")]
    pub replacement: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to extract substrings matching a regex pattern.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RegexExtractRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to extract from
    #[schemars(description = "Name of the string column to extract from.")]
    pub column: String,

    /// Regex pattern with capture groups
    #[schemars(
        description = "Regular expression pattern. Use capture groups () to specify what to extract."
    )]
    pub pattern: String,

    /// Name for the new column
    #[schemars(description = "Name for the new column containing extracted values.")]
    pub new_column: String,

    /// Which capture group to extract
    #[schemars(
        description = "Which capture group to extract: 0 = entire match, 1 = first group, etc. Default is 1."
    )]
    pub group: Option<usize>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to count regex matches in each row.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RegexCountRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to search in
    #[schemars(description = "Name of the string column to search in.")]
    pub column: String,

    /// Regex pattern
    #[schemars(description = "Regular expression pattern to count matches for.")]
    pub pattern: String,

    /// Name for the new count column
    #[schemars(description = "Name for the new column containing match counts.")]
    pub new_column: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to split a string column into multiple columns.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct StrSplitRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to split
    #[schemars(description = "Name of the string column to split.")]
    pub column: String,

    /// Pattern to split on
    #[schemars(description = "Pattern to split on (regex supported). E.g., ',' or '\\s+'.")]
    pub pattern: String,

    /// Maximum number of splits
    #[schemars(
        description = "Maximum number of splits. If not provided, splits on all occurrences."
    )]
    pub max_splits: Option<usize>,

    /// Prefix for new column names
    #[schemars(
        description = "Prefix for new column names. Creates columns named prefix_0, prefix_1, etc."
    )]
    pub prefix: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to concatenate multiple string columns.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct StrConcatRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Columns to concatenate
    #[schemars(description = "Names of the string columns to concatenate.")]
    pub columns: Vec<String>,

    /// Name for the new column
    #[schemars(description = "Name for the new concatenated column.")]
    pub new_column: String,

    /// Separator between values
    #[schemars(description = "Separator to insert between values. Default is empty string.")]
    pub separator: Option<String>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to get string lengths.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct StrLengthRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to measure
    #[schemars(description = "Name of the string column to measure lengths for.")]
    pub column: String,

    /// Name for the new length column
    #[schemars(description = "Name for the new column containing string lengths.")]
    pub new_column: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to extract a substring.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct StrSubstringRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to extract from
    #[schemars(description = "Name of the string column.")]
    pub column: String,

    /// Start index
    #[schemars(description = "Start index (0-based). Negative values count from end.")]
    pub start: i64,

    /// Length to extract
    #[schemars(description = "Number of characters to extract. If not provided, extracts to end.")]
    pub length: Option<usize>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to create lag or lead columns.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct LagLeadRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to shift
    #[schemars(description = "Column name to create lag/lead for.")]
    pub column: String,

    /// Number of periods to shift
    #[schemars(
        description = "Number of periods to shift. Positive for lag, negative for lead (or use 'direction')."
    )]
    pub periods: i64,

    /// Direction: 'lag' or 'lead'
    #[schemars(
        description = "Direction: 'lag' (shift forward) or 'lead' (shift backward). Default is 'lag'."
    )]
    pub direction: Option<String>,

    /// Columns to group by (for panel data)
    #[schemars(description = "Optional group-by columns for panel data (e.g., ['firm_id']).")]
    pub group_by: Option<Vec<String>>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to standardize or normalize columns.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct StandardizeRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Columns to transform
    #[schemars(description = "Column names to standardize/normalize.")]
    pub columns: Vec<String>,

    /// Method: 'standardize' or 'normalize'
    #[schemars(
        description = "Method: 'standardize' (z-score) or 'normalize' (0-1 range). Default is 'standardize'."
    )]
    pub method: Option<String>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to bin a continuous variable.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct BinColumnRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to bin
    #[schemars(description = "Column name to bin.")]
    pub column: String,

    /// Binning strategy
    #[schemars(
        description = "Binning strategy: 'uniform' (equal width), 'quantile' (equal frequency), or 'custom'."
    )]
    pub strategy: String,

    /// Number of bins or custom breaks
    #[schemars(
        description = "Number of bins (for uniform/quantile) or list of break points (for custom)."
    )]
    pub bins: Vec<f64>,

    /// Optional labels for bins
    #[schemars(description = "Optional labels for the bins.")]
    pub labels: Option<Vec<String>>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to one-hot encode a categorical column.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct OneHotEncodeRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to encode
    #[schemars(description = "Categorical column name to one-hot encode.")]
    pub column: String,

    /// Whether to drop the first category
    #[schemars(
        description = "If true, drop first category to avoid multicollinearity. Default is false."
    )]
    pub drop_first: Option<bool>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to compute differences or percent changes.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DiffRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to compute differences for
    #[schemars(description = "Column name to compute differences for.")]
    pub column: String,

    /// Number of periods
    #[schemars(description = "Number of periods for difference. Default is 1.")]
    pub periods: Option<i64>,

    /// Type of difference
    #[schemars(
        description = "Type: 'diff' (absolute difference) or 'pct_change' (percent change). Default is 'diff'."
    )]
    pub diff_type: Option<String>,

    /// Columns to group by (for panel data)
    #[schemars(description = "Optional group-by columns for panel data.")]
    pub group_by: Option<Vec<String>>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to sample rows from a dataset.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SampleDatasetRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Number of rows to sample
    #[schemars(description = "Number of rows to sample.")]
    pub n: usize,

    /// Whether to sample with replacement
    #[schemars(description = "If true, sample with replacement. Default is false.")]
    pub replace: Option<bool>,

    /// Random seed for reproducibility
    #[schemars(description = "Random seed for reproducible sampling.")]
    pub seed: Option<u64>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to cast column(s) to a different data type.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CastColumnRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Single column to cast
    #[schemars(description = "Name of the column to cast. Use this for single-column casting.")]
    pub column: Option<String>,

    /// Target data type for single column
    #[schemars(
        description = "Target data type: 'int' (i64), 'float' (f64), 'string' (Utf8), 'bool' (Boolean)."
    )]
    pub dtype: Option<String>,

    /// Batch cast: list of [column, dtype] pairs
    #[schemars(
        description = "Batch mode: list of [column, dtype] pairs, e.g. [[\"col1\", \"float\"], [\"col2\", \"int\"]]. Use this to cast multiple columns at once."
    )]
    pub casts: Option<Vec<Vec<String>>>,

    /// Optional name for the resulting dataset
    #[schemars(
        description = "Optional name for the result. If not provided, overwrites the source dataset."
    )]
    pub result_name: Option<String>,
}

/// Request to create a new column by computation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MutateColumnRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Name for the new column
    #[schemars(description = "Name for the new column.")]
    pub new_column: String,

    /// Expression type
    #[schemars(
        description = "Expression type: 'arithmetic' (e.g., col1 + col2), 'function' (e.g., log(col)), or 'constant'."
    )]
    pub expr_type: String,

    /// Left operand (column name for arithmetic)
    #[schemars(
        description = "Left operand: column name for arithmetic, column for function, or constant value."
    )]
    pub left: String,

    /// Operator (for arithmetic: '+', '-', '*', '/')
    #[schemars(
        description = "Operator for arithmetic: '+', '-', '*', '/'. For function: function name ('log', 'exp', 'sqrt', 'abs', 'square')."
    )]
    pub operator: Option<String>,

    /// Right operand (column name for arithmetic)
    #[schemars(description = "Right operand: column name for arithmetic expressions.")]
    pub right: Option<String>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}
