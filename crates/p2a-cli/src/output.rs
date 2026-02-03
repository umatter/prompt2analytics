//! Output formatting and validation for CLI results

use clap::ValueEnum;
use p2a_core::Dataset;
use serde::Serialize;
use std::fmt::Display;

/// Output format options
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum OutputFormat {
    /// Human-readable text output
    #[default]
    Text,
    /// JSON output for programmatic consumption
    Json,
    /// Formatted table output
    Table,
}

/// Format a result for output
pub fn format_output<T: Serialize + Display>(value: &T, format: &OutputFormat) -> String {
    match format {
        OutputFormat::Text => value.to_string(),
        OutputFormat::Json => {
            serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
        }
        OutputFormat::Table => value.to_string(), // TODO: Use tabled for structured data
    }
}

/// Print formatted output
pub fn print_output<T: Serialize + Display>(value: &T, format: &OutputFormat) {
    println!("{}", format_output(value, format));
}

/// Format a simple message
pub fn print_message(msg: &str, format: &OutputFormat) {
    match format {
        OutputFormat::Json => {
            println!("{}", serde_json::json!({"message": msg}));
        }
        _ => println!("{}", msg),
    }
}

/// Format an error message
pub fn print_error(msg: &str, format: &OutputFormat) {
    match format {
        OutputFormat::Json => {
            eprintln!("{}", serde_json::json!({"error": msg}));
        }
        _ => eprintln!("Error: {}", msg),
    }
}

/// Format dataset summary for display
pub fn format_dataset_summary(
    name: &str,
    nrows: usize,
    ncols: usize,
    columns: &[String],
    format: &OutputFormat,
) -> String {
    match format {
        OutputFormat::Json => serde_json::to_string_pretty(&serde_json::json!({
            "name": name,
            "rows": nrows,
            "columns": ncols,
            "column_names": columns,
        }))
        .unwrap(),
        OutputFormat::Table | OutputFormat::Text => {
            let mut out = String::new();
            out.push_str(&format!("Dataset: {}\n", name));
            out.push_str(&format!("  Rows: {}\n", nrows));
            out.push_str(&format!("  Columns: {}\n", ncols));
            out.push_str(&format!("  Column names: {}\n", columns.join(", ")));
            out
        }
    }
}

/// Format regression results for display
pub fn format_regression_results(
    method: &str,
    coefficients: &[(String, f64, f64, f64, f64)], // name, coef, se, t, p
    r_squared: f64,
    adj_r_squared: f64,
    n_obs: usize,
    format: &OutputFormat,
) -> String {
    match format {
        OutputFormat::Json => {
            let coef_json: Vec<_> = coefficients
                .iter()
                .map(|(name, coef, se, t, p)| {
                    serde_json::json!({
                        "name": name,
                        "coefficient": coef,
                        "std_error": se,
                        "t_value": t,
                        "p_value": p,
                    })
                })
                .collect();

            serde_json::to_string_pretty(&serde_json::json!({
                "method": method,
                "coefficients": coef_json,
                "r_squared": r_squared,
                "adj_r_squared": adj_r_squared,
                "n_observations": n_obs,
            }))
            .unwrap()
        }
        OutputFormat::Table | OutputFormat::Text => {
            let mut out = String::new();
            out.push_str(&format!("\n{} Results\n", method));
            out.push_str(&"=".repeat(60));
            out.push_str("\n\n");

            // Header
            out.push_str(&format!(
                "{:<15} {:>12} {:>12} {:>10} {:>10}\n",
                "Variable", "Coefficient", "Std. Error", "t-value", "p-value"
            ));
            out.push_str(&"-".repeat(60));
            out.push('\n');

            // Coefficients
            for (name, coef, se, t, p) in coefficients {
                let sig = if *p < 0.001 {
                    "***"
                } else if *p < 0.01 {
                    "**"
                } else if *p < 0.05 {
                    "*"
                } else if *p < 0.1 {
                    "."
                } else {
                    ""
                };
                out.push_str(&format!(
                    "{:<15} {:>12.6} {:>12.6} {:>10.3} {:>10.4}{}\n",
                    name, coef, se, t, p, sig
                ));
            }

            out.push_str(&"-".repeat(60));
            out.push('\n');
            out.push_str(&format!("R-squared: {:.6}\n", r_squared));
            out.push_str(&format!("Adj. R-squared: {:.6}\n", adj_r_squared));
            out.push_str(&format!("Observations: {}\n", n_obs));
            out.push_str("\nSignif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1\n");

            out
        }
    }
}

// =============================================================================
// Input Validation Helpers
// =============================================================================

/// Validate that a column exists in the dataset.
/// Returns an error message with similar column suggestions if not found.
pub fn validate_column_exists(
    dataset: &Dataset,
    col_name: &str,
    purpose: &str,
) -> Result<(), String> {
    let columns: Vec<String> = dataset
        .df()
        .get_column_names()
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    if columns.iter().any(|c| c == col_name) {
        return Ok(());
    }

    // Find similar columns using edit distance
    let suggestions = find_similar_columns(&columns, col_name, 3);

    let mut msg = format!(
        "Column '{}' not found (for {}).",
        col_name, purpose
    );

    if !suggestions.is_empty() {
        msg.push_str("\n  Did you mean one of these?");
        for s in &suggestions {
            msg.push_str(&format!("\n    - {}", s));
        }
    }

    msg.push_str(&format!(
        "\n  Available columns: {}",
        if columns.len() > 10 {
            format!("{}, ... ({} total)", columns[..10].join(", "), columns.len())
        } else {
            columns.join(", ")
        }
    ));

    Err(msg)
}

/// Validate that multiple columns exist in the dataset.
pub fn validate_columns_exist(
    dataset: &Dataset,
    col_names: &[&str],
    purpose: &str,
) -> Result<(), String> {
    for col_name in col_names {
        validate_column_exists(dataset, col_name, purpose)?;
    }
    Ok(())
}

/// Validate that a numeric parameter is within a valid range.
pub fn validate_range<T: PartialOrd + Display>(
    value: T,
    min: Option<T>,
    max: Option<T>,
    param_name: &str,
) -> Result<(), String> {
    if let Some(min_val) = min {
        if value < min_val {
            return Err(format!(
                "Invalid {}: {} is below minimum allowed value ({}).",
                param_name, value, min_val
            ));
        }
    }
    if let Some(max_val) = max {
        if value > max_val {
            return Err(format!(
                "Invalid {}: {} exceeds maximum allowed value ({}).",
                param_name, value, max_val
            ));
        }
    }
    Ok(())
}

/// Validate confidence level is between 0 and 1 (exclusive).
pub fn validate_confidence_level(level: f64) -> Result<(), String> {
    if level <= 0.0 || level >= 1.0 {
        return Err(format!(
            "Invalid confidence level: {:.4}. Must be between 0 and 1 (exclusive).\n  \
            Common values: 0.90, 0.95, 0.99",
            level
        ));
    }
    Ok(())
}

/// Validate significance level (alpha) is between 0 and 1 (exclusive).
pub fn validate_significance_level(alpha: f64) -> Result<(), String> {
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(format!(
            "Invalid significance level (alpha): {:.4}. Must be between 0 and 1 (exclusive).\n  \
            Common values: 0.01, 0.05, 0.10",
            alpha
        ));
    }
    Ok(())
}

/// Validate minimum sample size for an analysis.
pub fn validate_sample_size(
    n_obs: usize,
    min_required: usize,
    analysis_type: &str,
) -> Result<(), String> {
    if n_obs < min_required {
        return Err(format!(
            "Insufficient sample size for {}: {} observations provided, but at least {} required.",
            analysis_type, n_obs, min_required
        ));
    }
    Ok(())
}

/// Validate that positive integer parameter is greater than zero.
pub fn validate_positive(value: usize, param_name: &str) -> Result<(), String> {
    if value == 0 {
        return Err(format!(
            "Invalid {}: value must be greater than 0.",
            param_name
        ));
    }
    Ok(())
}

/// Find similar column names using Levenshtein distance.
fn find_similar_columns(columns: &[String], target: &str, max_results: usize) -> Vec<String> {
    let mut scored: Vec<(String, usize)> = columns
        .iter()
        .map(|c| (c.clone(), levenshtein_distance(c, target)))
        .filter(|(_, dist)| *dist <= 3) // Only suggestions within 3 edits
        .collect();

    scored.sort_by_key(|(_, dist)| *dist);
    scored
        .into_iter()
        .take(max_results)
        .map(|(name, _)| name)
        .collect()
}

/// Compute Levenshtein edit distance between two strings.
fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();
    let a: Vec<char> = a_lower.chars().collect();
    let b: Vec<char> = b_lower.chars().collect();

    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }

    let mut prev_row: Vec<usize> = (0..=b.len()).collect();
    let mut curr_row = vec![0; b.len() + 1];

    for (i, ca) in a.iter().enumerate() {
        curr_row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr_row[j + 1] = (prev_row[j + 1] + 1)
                .min(curr_row[j] + 1)
                .min(prev_row[j] + cost);
        }
        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[b.len()]
}
