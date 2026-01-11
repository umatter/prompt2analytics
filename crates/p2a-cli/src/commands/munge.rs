//! Data munging commands for filtering, joining, reshaping, and cleaning data.

use clap::{Subcommand, ValueEnum};
use p2a_core::data::munging::{
    filter, select, drop_columns, rename, sort, mutate, sample,
    drop_na, fill_na, deduplicate, FillStrategy,
    left_join, right_join, inner_join, full_join, concat,
    group_by, value_counts, AggFn, AggSpec,
    pivot, melt,
    lag, lead, diff, pct_change, standardize, normalize, bin, one_hot_encode,
    BinStrategy, MutateExpr, ArithOp,
};
use p2a_core::Dataset;

use crate::output::{print_error, print_message, OutputFormat};
use crate::session::SessionManager;

#[derive(Clone, ValueEnum)]
pub enum JoinType {
    Left,
    Right,
    Inner,
    Full,
}

#[derive(Clone, ValueEnum)]
pub enum FillMethod {
    Mean,
    Median,
    Constant,
    Forward,
    Backward,
    Zero,
}

#[derive(Clone, ValueEnum)]
pub enum BinMethod {
    EqualWidth,
    Quantile,
}

#[derive(Subcommand)]
pub enum MungeCommands {
    /// Filter rows based on a condition
    Filter {
        /// Dataset name
        dataset: String,

        /// Column to filter on
        #[arg(short, long)]
        column: String,

        /// Comparison operator (eq, ne, gt, ge, lt, le, contains, startswith, endswith)
        #[arg(short, long)]
        op: String,

        /// Value to compare against
        #[arg(short, long)]
        value: String,

        /// Name for the resulting dataset
        #[arg(short, long)]
        name: Option<String>,
    },

    /// Select specific columns
    Select {
        /// Dataset name
        dataset: String,

        /// Columns to select
        #[arg(short, long, num_args = 1..)]
        columns: Vec<String>,

        /// Name for the resulting dataset
        #[arg(short, long)]
        name: Option<String>,
    },

    /// Drop columns from a dataset
    Drop {
        /// Dataset name
        dataset: String,

        /// Columns to drop
        #[arg(short, long, num_args = 1..)]
        columns: Vec<String>,

        /// Name for the resulting dataset
        #[arg(short, long)]
        name: Option<String>,
    },

    /// Rename columns
    Rename {
        /// Dataset name
        dataset: String,

        /// Column renames in OLD=NEW format
        #[arg(short, long, num_args = 1..)]
        renames: Vec<String>,

        /// Name for the resulting dataset
        #[arg(short, long)]
        name: Option<String>,
    },

    /// Sort by columns
    Sort {
        /// Dataset name
        dataset: String,

        /// Columns to sort by
        #[arg(short, long, num_args = 1..)]
        by: Vec<String>,

        /// Sort in descending order
        #[arg(short, long)]
        desc: bool,

        /// Name for the resulting dataset
        #[arg(short, long)]
        name: Option<String>,
    },

    /// Create or compute a new column
    Mutate {
        /// Dataset name
        dataset: String,

        /// Name of the new column
        #[arg(short, long)]
        new_col: String,

        /// Expression: copy:COL, constant:VALUE, add:COL1:COL2, sub:COL1:COL2, mul:COL1:COL2, div:COL1:COL2
        #[arg(short, long)]
        expr: String,

        /// Name for the resulting dataset
        #[arg(short, long)]
        name: Option<String>,
    },

    /// Take a random sample of rows
    Sample {
        /// Dataset name
        dataset: String,

        /// Number of rows to sample
        #[arg(short, long)]
        n: usize,

        /// Sample with replacement
        #[arg(long)]
        replace: bool,

        /// Random seed for reproducibility
        #[arg(long)]
        seed: Option<u64>,

        /// Name for the resulting dataset
        #[arg(short = 'o', long)]
        name: Option<String>,
    },

    /// Join two datasets
    Join {
        /// Left dataset name
        left: String,

        /// Right dataset name
        right: String,

        /// Join key columns
        #[arg(short, long, num_args = 1..)]
        on: Vec<String>,

        /// Right key columns (if different from left)
        #[arg(long, num_args = 1..)]
        right_on: Option<Vec<String>>,

        /// Join type
        #[arg(short = 't', long, default_value = "left")]
        join_type: JoinType,

        /// Suffix for duplicate column names
        #[arg(long, default_value = "_right")]
        suffix: String,

        /// Name for the resulting dataset
        #[arg(short, long)]
        name: Option<String>,
    },

    /// Concatenate datasets vertically
    Concat {
        /// Dataset names to concatenate
        #[arg(num_args = 2..)]
        datasets: Vec<String>,

        /// Name for the resulting dataset
        #[arg(short, long)]
        name: Option<String>,
    },

    /// Group by and aggregate
    GroupBy {
        /// Dataset name
        dataset: String,

        /// Columns to group by
        #[arg(short, long, num_args = 1..)]
        by: Vec<String>,

        /// Aggregations in COLUMN:FUNCTION format (sum, mean, count, min, max, std, var, first, last, median)
        #[arg(short, long, num_args = 1..)]
        aggs: Vec<String>,

        /// Name for the resulting dataset
        #[arg(short, long)]
        name: Option<String>,
    },

    /// Count value frequencies
    ValueCounts {
        /// Dataset name
        dataset: String,

        /// Column to count
        #[arg(short, long)]
        column: String,

        /// Name for the resulting dataset
        #[arg(short, long)]
        name: Option<String>,
    },

    /// Pivot from long to wide format
    Pivot {
        /// Dataset name
        dataset: String,

        /// Index columns (row identifiers)
        #[arg(short, long, num_args = 1..)]
        index: Vec<String>,

        /// Column whose values become new column names
        #[arg(short, long)]
        on: String,

        /// Column containing values
        #[arg(short, long)]
        values: String,

        /// Name for the resulting dataset
        #[arg(short, long)]
        name: Option<String>,
    },

    /// Melt from wide to long format
    Melt {
        /// Dataset name
        dataset: String,

        /// ID columns to keep
        #[arg(short, long, num_args = 1..)]
        id_vars: Vec<String>,

        /// Value columns to unpivot
        #[arg(short, long, num_args = 1..)]
        value_vars: Vec<String>,

        /// Name for the variable column
        #[arg(long, default_value = "variable")]
        var_name: String,

        /// Name for the value column
        #[arg(long, default_value = "value")]
        val_name: String,

        /// Name for the resulting dataset
        #[arg(short, long)]
        name: Option<String>,
    },

    /// Drop rows with missing values
    DropNa {
        /// Dataset name
        dataset: String,

        /// Columns to check (all if not specified)
        #[arg(short, long, num_args = 1..)]
        columns: Option<Vec<String>>,

        /// How to drop: any (default) or all
        #[arg(long, default_value = "any")]
        how: String,

        /// Name for the resulting dataset
        #[arg(short, long)]
        name: Option<String>,
    },

    /// Fill missing values
    FillNa {
        /// Dataset name
        dataset: String,

        /// Fill method
        #[arg(short, long)]
        method: FillMethod,

        /// Columns to fill (all if not specified)
        #[arg(short, long, num_args = 1..)]
        columns: Option<Vec<String>>,

        /// Constant value (for constant method)
        #[arg(long)]
        value: Option<f64>,

        /// Name for the resulting dataset
        #[arg(short, long)]
        name: Option<String>,
    },

    /// Remove duplicate rows
    Deduplicate {
        /// Dataset name
        dataset: String,

        /// Columns to consider for duplicates (all if not specified)
        #[arg(short, long, num_args = 1..)]
        subset: Option<Vec<String>>,

        /// Which duplicate to keep: first (default), last, or none
        #[arg(long, default_value = "first")]
        keep: String,

        /// Name for the resulting dataset
        #[arg(short, long)]
        name: Option<String>,
    },

    /// Create lag of a column
    Lag {
        /// Dataset name
        dataset: String,

        /// Column to lag
        #[arg(short, long)]
        column: String,

        /// Number of periods to lag
        #[arg(short, long, default_value = "1")]
        periods: usize,

        /// Group by columns (for panel data)
        #[arg(short, long, num_args = 1..)]
        group_by: Option<Vec<String>>,

        /// Name for the resulting dataset
        #[arg(short, long)]
        name: Option<String>,
    },

    /// Create lead of a column
    Lead {
        /// Dataset name
        dataset: String,

        /// Column to lead
        #[arg(short, long)]
        column: String,

        /// Number of periods to lead
        #[arg(short, long, default_value = "1")]
        periods: usize,

        /// Group by columns (for panel data)
        #[arg(short, long, num_args = 1..)]
        group_by: Option<Vec<String>>,

        /// Name for the resulting dataset
        #[arg(short, long)]
        name: Option<String>,
    },

    /// Compute difference of a column
    Diff {
        /// Dataset name
        dataset: String,

        /// Column to difference
        #[arg(short, long)]
        column: String,

        /// Number of periods
        #[arg(short, long, default_value = "1")]
        periods: usize,

        /// Compute percentage change instead
        #[arg(long)]
        pct: bool,

        /// Name for the resulting dataset
        #[arg(short, long)]
        name: Option<String>,
    },

    /// Standardize (z-score) columns
    Standardize {
        /// Dataset name
        dataset: String,

        /// Columns to standardize
        #[arg(short, long, num_args = 1..)]
        columns: Vec<String>,

        /// Normalize to 0-1 instead of z-score
        #[arg(long)]
        normalize: bool,

        /// Name for the resulting dataset
        #[arg(short, long)]
        name: Option<String>,
    },

    /// Bin a continuous column
    Bin {
        /// Dataset name
        dataset: String,

        /// Column to bin
        #[arg(short, long)]
        column: String,

        /// Binning method
        #[arg(short, long, default_value = "equal-width")]
        method: BinMethod,

        /// Number of bins
        #[arg(short, long, default_value = "10")]
        n_bins: usize,

        /// Name for the resulting dataset
        #[arg(short = 'o', long)]
        name: Option<String>,
    },

    /// One-hot encode a categorical column
    OneHot {
        /// Dataset name
        dataset: String,

        /// Column to encode
        #[arg(short, long)]
        column: String,

        /// Drop first category (avoid multicollinearity)
        #[arg(long)]
        drop_first: bool,

        /// Name for the resulting dataset
        #[arg(short, long)]
        name: Option<String>,
    },
}

pub fn execute(
    cmd: &MungeCommands,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    match cmd {
        MungeCommands::Filter { dataset, column, op, value, name } => {
            execute_filter(dataset, column, op, value, name.as_deref(), format, session)
        }
        MungeCommands::Select { dataset, columns, name } => {
            execute_select(dataset, columns, name.as_deref(), format, session)
        }
        MungeCommands::Drop { dataset, columns, name } => {
            execute_drop(dataset, columns, name.as_deref(), format, session)
        }
        MungeCommands::Rename { dataset, renames, name } => {
            execute_rename(dataset, renames, name.as_deref(), format, session)
        }
        MungeCommands::Sort { dataset, by, desc, name } => {
            execute_sort(dataset, by, *desc, name.as_deref(), format, session)
        }
        MungeCommands::Mutate { dataset, new_col, expr, name } => {
            execute_mutate(dataset, new_col, expr, name.as_deref(), format, session)
        }
        MungeCommands::Sample { dataset, n, replace, seed, name } => {
            execute_sample(dataset, *n, *replace, *seed, name.as_deref(), format, session)
        }
        MungeCommands::Join { left, right, on, right_on, join_type, suffix, name } => {
            execute_join(left, right, on, right_on.as_ref(), join_type, suffix, name.as_deref(), format, session)
        }
        MungeCommands::Concat { datasets, name } => {
            execute_concat(datasets, name.as_deref(), format, session)
        }
        MungeCommands::GroupBy { dataset, by, aggs, name } => {
            execute_group_by(dataset, by, aggs, name.as_deref(), format, session)
        }
        MungeCommands::ValueCounts { dataset, column, name } => {
            execute_value_counts(dataset, column, name.as_deref(), format, session)
        }
        MungeCommands::Pivot { dataset, index, on, values, name } => {
            execute_pivot(dataset, index, on, values, name.as_deref(), format, session)
        }
        MungeCommands::Melt { dataset, id_vars, value_vars, var_name, val_name, name } => {
            execute_melt(dataset, id_vars, value_vars, var_name, val_name, name.as_deref(), format, session)
        }
        MungeCommands::DropNa { dataset, columns, how, name } => {
            execute_drop_na(dataset, columns.as_ref(), how, name.as_deref(), format, session)
        }
        MungeCommands::FillNa { dataset, method, columns, value, name } => {
            execute_fill_na(dataset, method, columns.as_ref(), *value, name.as_deref(), format, session)
        }
        MungeCommands::Deduplicate { dataset, subset, keep, name } => {
            execute_deduplicate(dataset, subset.as_ref(), keep, name.as_deref(), format, session)
        }
        MungeCommands::Lag { dataset, column, periods, group_by, name } => {
            execute_lag(dataset, column, *periods, group_by.as_ref(), name.as_deref(), format, session)
        }
        MungeCommands::Lead { dataset, column, periods, group_by, name } => {
            execute_lead(dataset, column, *periods, group_by.as_ref(), name.as_deref(), format, session)
        }
        MungeCommands::Diff { dataset, column, periods, pct, name } => {
            execute_diff(dataset, column, *periods, *pct, name.as_deref(), format, session)
        }
        MungeCommands::Standardize { dataset, columns, normalize, name } => {
            execute_standardize(dataset, columns, *normalize, name.as_deref(), format, session)
        }
        MungeCommands::Bin { dataset, column, method, n_bins, name } => {
            execute_bin(dataset, column, method, *n_bins, name.as_deref(), format, session)
        }
        MungeCommands::OneHot { dataset, column, drop_first, name } => {
            execute_one_hot(dataset, column, *drop_first, name.as_deref(), format, session)
        }
    }
}

/// Helper macro to get a dataset, clone it, and handle errors
macro_rules! get_dataset_clone {
    ($name:expr, $session:expr, $format:expr) => {
        match $session {
            Some(ref mgr) => match mgr.get_dataset($name) {
                Some(ds) => ds.clone(),
                None => {
                    print_error(&format!("Dataset '{}' not found", $name), $format);
                    return Ok(());
                }
            },
            None => {
                print_error("No session active. Use --session <file> to enable dataset storage.", $format);
                return Ok(());
            }
        }
    };
}

fn store_result(
    result: Dataset,
    output_name: Option<&str>,
    source_name: &str,
    operation: &str,
    session: Option<&mut SessionManager>,
    format: &OutputFormat,
) -> anyhow::Result<()> {
    let name = output_name
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("{}_{}", source_name, operation));

    let nrows = result.nrows();
    let ncols = result.ncols();

    if let Some(mgr) = session {
        mgr.store_dataset(name.clone(), result);
    }

    print_message(
        &format!("Created dataset '{}' ({} rows x {} columns)", name, nrows, ncols),
        format,
    );

    Ok(())
}

fn execute_filter(
    dataset: &str,
    column: &str,
    op: &str,
    value: &str,
    output_name: Option<&str>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let ds = get_dataset_clone!(dataset, session, format);

    match filter(&ds, column, op, value) {
        Ok(result) => store_result(result, output_name, dataset, "filtered", session, format),
        Err(e) => {
            print_error(&format!("Filter failed: {}", e), format);
            Ok(())
        }
    }
}

fn execute_select(
    dataset: &str,
    columns: &[String],
    output_name: Option<&str>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let ds = get_dataset_clone!(dataset, session, format);

    let cols: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();

    match select(&ds, &cols) {
        Ok(result) => store_result(result, output_name, dataset, "selected", session, format),
        Err(e) => {
            print_error(&format!("Select failed: {}", e), format);
            Ok(())
        }
    }
}

fn execute_drop(
    dataset: &str,
    columns: &[String],
    output_name: Option<&str>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let ds = get_dataset_clone!(dataset, session, format);

    let cols: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();

    match drop_columns(&ds, &cols) {
        Ok(result) => store_result(result, output_name, dataset, "dropped", session, format),
        Err(e) => {
            print_error(&format!("Drop failed: {}", e), format);
            Ok(())
        }
    }
}

fn execute_rename(
    dataset: &str,
    renames: &[String],
    output_name: Option<&str>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let ds = get_dataset_clone!(dataset, session, format);

    // Parse renames from OLD=NEW format
    let mut rename_map: Vec<(&str, &str)> = Vec::new();
    for r in renames {
        let parts: Vec<&str> = r.split('=').collect();
        if parts.len() != 2 {
            print_error(&format!("Invalid rename format '{}'. Use OLD=NEW", r), format);
            return Ok(());
        }
        rename_map.push((parts[0], parts[1]));
    }

    match rename(&ds, &rename_map) {
        Ok(result) => store_result(result, output_name, dataset, "renamed", session, format),
        Err(e) => {
            print_error(&format!("Rename failed: {}", e), format);
            Ok(())
        }
    }
}

fn execute_sort(
    dataset: &str,
    by: &[String],
    desc: bool,
    output_name: Option<&str>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let ds = get_dataset_clone!(dataset, session, format);

    let cols: Vec<&str> = by.iter().map(|s| s.as_str()).collect();
    let descending: Vec<bool> = vec![desc; cols.len()];

    match sort(&ds, &cols, &descending) {
        Ok(result) => store_result(result, output_name, dataset, "sorted", session, format),
        Err(e) => {
            print_error(&format!("Sort failed: {}", e), format);
            Ok(())
        }
    }
}

fn execute_mutate(
    dataset: &str,
    new_col: &str,
    expr: &str,
    output_name: Option<&str>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let ds = get_dataset_clone!(dataset, session, format);

    // Parse expression: copy:COL, constant:VALUE, add:COL1:COL2, etc.
    let parts: Vec<&str> = expr.split(':').collect();
    if parts.is_empty() {
        print_error("Invalid expression format", format);
        return Ok(());
    }

    let mutate_expr = match parts[0] {
        "copy" if parts.len() == 2 => MutateExpr::Copy(parts[1].to_string()),
        "constant" if parts.len() == 2 => MutateExpr::Constant(parts[1].to_string()),
        "add" if parts.len() == 3 => MutateExpr::Arithmetic(parts[1].to_string(), ArithOp::Add, parts[2].to_string()),
        "sub" if parts.len() == 3 => MutateExpr::Arithmetic(parts[1].to_string(), ArithOp::Sub, parts[2].to_string()),
        "mul" if parts.len() == 3 => MutateExpr::Arithmetic(parts[1].to_string(), ArithOp::Mul, parts[2].to_string()),
        "div" if parts.len() == 3 => MutateExpr::Arithmetic(parts[1].to_string(), ArithOp::Div, parts[2].to_string()),
        _ => {
            print_error("Invalid expression. Use: copy:COL, constant:VALUE, add:COL1:COL2, sub:COL1:COL2, mul:COL1:COL2, div:COL1:COL2", format);
            return Ok(());
        }
    };

    match mutate(&ds, new_col, mutate_expr) {
        Ok(result) => store_result(result, output_name, dataset, "mutated", session, format),
        Err(e) => {
            print_error(&format!("Mutate failed: {}", e), format);
            Ok(())
        }
    }
}

fn execute_sample(
    dataset: &str,
    n: usize,
    replace: bool,
    seed: Option<u64>,
    output_name: Option<&str>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let ds = get_dataset_clone!(dataset, session, format);

    match sample(&ds, Some(n), None, replace, seed) {
        Ok(result) => store_result(result, output_name, dataset, "sampled", session, format),
        Err(e) => {
            print_error(&format!("Sample failed: {}", e), format);
            Ok(())
        }
    }
}

fn execute_join(
    left: &str,
    right: &str,
    on: &[String],
    right_on: Option<&Vec<String>>,
    join_type: &JoinType,
    suffix: &str,
    output_name: Option<&str>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    // Get both datasets first
    let left_ds = match session.as_ref().and_then(|s| s.get_dataset(left)) {
        Some(ds) => ds.clone(),
        None => {
            print_error(&format!("Dataset '{}' not found", left), format);
            return Ok(());
        }
    };
    let right_ds = match session.as_ref().and_then(|s| s.get_dataset(right)) {
        Some(ds) => ds.clone(),
        None => {
            print_error(&format!("Dataset '{}' not found", right), format);
            return Ok(());
        }
    };

    let on_cols: Vec<&str> = on.iter().map(|s| s.as_str()).collect();
    let right_on_cols: Option<Vec<&str>> = right_on.map(|v| v.iter().map(|s| s.as_str()).collect());
    let right_on_ref: Option<&[&str]> = right_on_cols.as_ref().map(|v| v.as_slice());

    let result = match join_type {
        JoinType::Left => left_join(&left_ds, &right_ds, &on_cols, right_on_ref, Some(suffix)),
        JoinType::Right => right_join(&left_ds, &right_ds, &on_cols, right_on_ref, Some(suffix)),
        JoinType::Inner => inner_join(&left_ds, &right_ds, &on_cols, right_on_ref, Some(suffix)),
        JoinType::Full => full_join(&left_ds, &right_ds, &on_cols, right_on_ref, Some(suffix)),
    };

    match result {
        Ok(ds) => store_result(ds, output_name, left, "joined", session, format),
        Err(e) => {
            print_error(&format!("Join failed: {}", e), format);
            Ok(())
        }
    }
}

fn execute_concat(
    datasets: &[String],
    output_name: Option<&str>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    if datasets.len() < 2 {
        print_error("At least 2 datasets required for concatenation", format);
        return Ok(());
    }

    // Collect all datasets
    let mut ds_vec: Vec<Dataset> = Vec::new();
    for name in datasets {
        match session.as_ref().and_then(|s| s.get_dataset(name)) {
            Some(ds) => ds_vec.push(ds.clone()),
            None => {
                print_error(&format!("Dataset '{}' not found", name), format);
                return Ok(());
            }
        }
    }

    let ds_refs: Vec<&Dataset> = ds_vec.iter().collect();

    match concat(&ds_refs) {
        Ok(result) => store_result(result, output_name, &datasets[0], "concat", session, format),
        Err(e) => {
            print_error(&format!("Concat failed: {}", e), format);
            Ok(())
        }
    }
}

fn execute_group_by(
    dataset: &str,
    by: &[String],
    aggs: &[String],
    output_name: Option<&str>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let ds = get_dataset_clone!(dataset, session, format);

    let by_cols: Vec<&str> = by.iter().map(|s| s.as_str()).collect();

    // Parse aggregations from COLUMN:FUNCTION format
    let mut agg_specs: Vec<AggSpec> = Vec::new();
    for agg in aggs {
        let parts: Vec<&str> = agg.split(':').collect();
        if parts.len() != 2 {
            print_error(&format!("Invalid aggregation format '{}'. Use COLUMN:FUNCTION", agg), format);
            return Ok(());
        }
        let col = parts[0];
        let func = match parts[1].to_lowercase().as_str() {
            "sum" => AggFn::Sum,
            "mean" | "avg" => AggFn::Mean,
            "count" => AggFn::Count,
            "min" => AggFn::Min,
            "max" => AggFn::Max,
            "std" => AggFn::Std,
            "var" => AggFn::Var,
            "first" => AggFn::First,
            "last" => AggFn::Last,
            "median" => AggFn::Median,
            _ => {
                print_error(&format!("Unknown aggregation function '{}'", parts[1]), format);
                return Ok(());
            }
        };
        agg_specs.push(AggSpec::new(col, func));
    }

    match group_by(&ds, &by_cols, &agg_specs) {
        Ok(result) => store_result(result, output_name, dataset, "grouped", session, format),
        Err(e) => {
            print_error(&format!("Group by failed: {}", e), format);
            Ok(())
        }
    }
}

fn execute_value_counts(
    dataset: &str,
    column: &str,
    output_name: Option<&str>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let ds = get_dataset_clone!(dataset, session, format);

    match value_counts(&ds, column) {
        Ok(result) => store_result(result, output_name, dataset, "counts", session, format),
        Err(e) => {
            print_error(&format!("Value counts failed: {}", e), format);
            Ok(())
        }
    }
}

fn execute_pivot(
    dataset: &str,
    index: &[String],
    on: &str,
    values: &str,
    output_name: Option<&str>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let ds = get_dataset_clone!(dataset, session, format);

    let index_cols: Vec<&str> = index.iter().map(|s| s.as_str()).collect();

    match pivot(&ds, &index_cols, on, values) {
        Ok(result) => store_result(result, output_name, dataset, "pivoted", session, format),
        Err(e) => {
            print_error(&format!("Pivot failed: {}", e), format);
            Ok(())
        }
    }
}

fn execute_melt(
    dataset: &str,
    id_vars: &[String],
    value_vars: &[String],
    var_name: &str,
    val_name: &str,
    output_name: Option<&str>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let ds = get_dataset_clone!(dataset, session, format);

    let id_cols: Vec<&str> = id_vars.iter().map(|s| s.as_str()).collect();
    let val_cols: Vec<&str> = value_vars.iter().map(|s| s.as_str()).collect();

    match melt(&ds, &id_cols, &val_cols, var_name, val_name) {
        Ok(result) => store_result(result, output_name, dataset, "melted", session, format),
        Err(e) => {
            print_error(&format!("Melt failed: {}", e), format);
            Ok(())
        }
    }
}

fn execute_drop_na(
    dataset: &str,
    columns: Option<&Vec<String>>,
    how: &str,
    output_name: Option<&str>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let ds = get_dataset_clone!(dataset, session, format);

    let cols: Option<Vec<&str>> = columns.map(|v| v.iter().map(|s| s.as_str()).collect());
    let cols_ref: Option<&[&str]> = cols.as_ref().map(|v| v.as_slice());

    match drop_na(&ds, cols_ref, how) {
        Ok(result) => store_result(result, output_name, dataset, "dropna", session, format),
        Err(e) => {
            print_error(&format!("Drop NA failed: {}", e), format);
            Ok(())
        }
    }
}

fn execute_fill_na(
    dataset: &str,
    method: &FillMethod,
    columns: Option<&Vec<String>>,
    value: Option<f64>,
    output_name: Option<&str>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let ds = get_dataset_clone!(dataset, session, format);

    let cols: Option<Vec<&str>> = columns.map(|v| v.iter().map(|s| s.as_str()).collect());
    let cols_ref: Option<&[&str]> = cols.as_ref().map(|v| v.as_slice());

    let strategy = match method {
        FillMethod::Mean => FillStrategy::Mean,
        FillMethod::Median => FillStrategy::Median,
        FillMethod::Constant => {
            let val = value.unwrap_or(0.0);
            FillStrategy::Constant(val.to_string())
        }
        FillMethod::Forward => FillStrategy::Forward,
        FillMethod::Backward => FillStrategy::Backward,
        FillMethod::Zero => FillStrategy::Zero,
    };

    match fill_na(&ds, cols_ref, strategy) {
        Ok(result) => store_result(result, output_name, dataset, "fillna", session, format),
        Err(e) => {
            print_error(&format!("Fill NA failed: {}", e), format);
            Ok(())
        }
    }
}

fn execute_deduplicate(
    dataset: &str,
    subset: Option<&Vec<String>>,
    keep: &str,
    output_name: Option<&str>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let ds = get_dataset_clone!(dataset, session, format);

    let cols: Option<Vec<&str>> = subset.map(|v| v.iter().map(|s| s.as_str()).collect());
    let cols_ref: Option<&[&str]> = cols.as_ref().map(|v| v.as_slice());

    match deduplicate(&ds, cols_ref, keep) {
        Ok(result) => store_result(result, output_name, dataset, "dedup", session, format),
        Err(e) => {
            print_error(&format!("Deduplicate failed: {}", e), format);
            Ok(())
        }
    }
}

fn execute_lag(
    dataset: &str,
    column: &str,
    periods: usize,
    group_by_cols: Option<&Vec<String>>,
    output_name: Option<&str>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let ds = get_dataset_clone!(dataset, session, format);

    let groups: Option<Vec<&str>> = group_by_cols.map(|v| v.iter().map(|s| s.as_str()).collect());
    let groups_ref: Option<&[&str]> = groups.as_ref().map(|v| v.as_slice());

    match lag(&ds, column, periods, groups_ref) {
        Ok(result) => store_result(result, output_name, dataset, "lagged", session, format),
        Err(e) => {
            print_error(&format!("Lag failed: {}", e), format);
            Ok(())
        }
    }
}

fn execute_lead(
    dataset: &str,
    column: &str,
    periods: usize,
    group_by_cols: Option<&Vec<String>>,
    output_name: Option<&str>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let ds = get_dataset_clone!(dataset, session, format);

    let groups: Option<Vec<&str>> = group_by_cols.map(|v| v.iter().map(|s| s.as_str()).collect());
    let groups_ref: Option<&[&str]> = groups.as_ref().map(|v| v.as_slice());

    match lead(&ds, column, periods, groups_ref) {
        Ok(result) => store_result(result, output_name, dataset, "lead", session, format),
        Err(e) => {
            print_error(&format!("Lead failed: {}", e), format);
            Ok(())
        }
    }
}

fn execute_diff(
    dataset: &str,
    column: &str,
    periods: usize,
    pct: bool,
    output_name: Option<&str>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let ds = get_dataset_clone!(dataset, session, format);

    let result = if pct {
        pct_change(&ds, column, periods)
    } else {
        diff(&ds, column, periods)
    };

    match result {
        Ok(ds) => store_result(ds, output_name, dataset, if pct { "pct_change" } else { "diff" }, session, format),
        Err(e) => {
            print_error(&format!("Diff failed: {}", e), format);
            Ok(())
        }
    }
}

fn execute_standardize(
    dataset: &str,
    columns: &[String],
    do_normalize: bool,
    output_name: Option<&str>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let ds = get_dataset_clone!(dataset, session, format);

    let cols: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();

    let result = if do_normalize {
        normalize(&ds, &cols)
    } else {
        standardize(&ds, &cols)
    };

    match result {
        Ok(ds) => store_result(ds, output_name, dataset, if do_normalize { "normalized" } else { "standardized" }, session, format),
        Err(e) => {
            print_error(&format!("Standardize failed: {}", e), format);
            Ok(())
        }
    }
}

fn execute_bin(
    dataset: &str,
    column: &str,
    method: &BinMethod,
    n_bins: usize,
    output_name: Option<&str>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let ds = get_dataset_clone!(dataset, session, format);

    let strategy = match method {
        BinMethod::EqualWidth => BinStrategy::EqualWidth(n_bins),
        BinMethod::Quantile => BinStrategy::Quantile(n_bins),
    };

    match bin(&ds, column, strategy, None) {
        Ok(result) => store_result(result, output_name, dataset, "binned", session, format),
        Err(e) => {
            print_error(&format!("Bin failed: {}", e), format);
            Ok(())
        }
    }
}

fn execute_one_hot(
    dataset: &str,
    column: &str,
    drop_first: bool,
    output_name: Option<&str>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let ds = get_dataset_clone!(dataset, session, format);

    match one_hot_encode(&ds, column, drop_first) {
        Ok(result) => store_result(result, output_name, dataset, "onehot", session, format),
        Err(e) => {
            print_error(&format!("One-hot encode failed: {}", e), format);
            Ok(())
        }
    }
}
