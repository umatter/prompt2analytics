//! Output formatting for CLI results

use clap::ValueEnum;
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
        OutputFormat::Json => serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string()),
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
        OutputFormat::Json => {
            serde_json::to_string_pretty(&serde_json::json!({
                "name": name,
                "rows": nrows,
                "columns": ncols,
                "column_names": columns,
            }))
            .unwrap()
        }
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
            out.push_str("\n");

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
            out.push_str("\n");
            out.push_str(&format!("R-squared: {:.6}\n", r_squared));
            out.push_str(&format!("Adj. R-squared: {:.6}\n", adj_r_squared));
            out.push_str(&format!("Observations: {}\n", n_obs));
            out.push_str("\nSignif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1\n");

            out
        }
    }
}
