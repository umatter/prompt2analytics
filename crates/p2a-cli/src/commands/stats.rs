//! Statistical tests and descriptive statistics commands

use clap::{Subcommand, ValueEnum};
use p2a_core::{
    // T-tests
    one_sample_t_test, two_sample_t_test, paired_t_test, Alternative,
    // Chi-squared tests
    run_chisq_gof, run_chisq_independence,
    // Other tests
    run_shapiro_wilk, run_fisher_test,
    wilcoxon_rank_sum, wilcoxon_signed_rank, WilcoxonConfig,
    run_bartlett_test, run_kruskal_test,
    // Descriptive
    run_fivenum, run_iqr, run_mad,
    // ACF/PACF (raw data versions)
    acf, pacf, AcfType,
};

use crate::output::{print_error, OutputFormat};
use crate::session::SessionManager;

#[derive(Clone, ValueEnum)]
pub enum TTestAlternative {
    TwoSided,
    Less,
    Greater,
}

impl From<TTestAlternative> for Alternative {
    fn from(val: TTestAlternative) -> Self {
        match val {
            TTestAlternative::TwoSided => Alternative::TwoSided,
            TTestAlternative::Less => Alternative::Less,
            TTestAlternative::Greater => Alternative::Greater,
        }
    }
}

#[derive(Subcommand)]
pub enum StatsCommands {
    /// One-sample t-test
    TTestOne {
        /// Dataset name
        dataset: String,

        /// Column to test
        #[arg(short, long)]
        col: String,

        /// Hypothesized mean (null hypothesis)
        #[arg(short, long, default_value = "0.0")]
        mu: f64,

        /// Alternative hypothesis
        #[arg(short, long, default_value = "two-sided")]
        alternative: TTestAlternative,

        /// Confidence level
        #[arg(long, default_value = "0.95")]
        conf_level: f64,
    },

    /// Two-sample t-test (independent samples)
    TTestTwo {
        /// Dataset name
        dataset: String,

        /// First column
        #[arg(long)]
        col1: String,

        /// Second column
        #[arg(long)]
        col2: String,

        /// Alternative hypothesis
        #[arg(short, long, default_value = "two-sided")]
        alternative: TTestAlternative,

        /// Assume equal variances (use pooled variance)
        #[arg(long)]
        equal_var: bool,
    },

    /// Paired t-test
    TTestPaired {
        /// Dataset name
        dataset: String,

        /// First column (before/treatment)
        #[arg(long)]
        col1: String,

        /// Second column (after/control)
        #[arg(long)]
        col2: String,

        /// Alternative hypothesis
        #[arg(short, long, default_value = "two-sided")]
        alternative: TTestAlternative,
    },

    /// Chi-squared goodness-of-fit test
    ChisqGof {
        /// Dataset name
        dataset: String,

        /// Column with observed counts
        #[arg(short, long)]
        observed: String,
    },

    /// Chi-squared test of independence
    ChisqIndep {
        /// Dataset name
        dataset: String,

        /// Row variable column
        #[arg(long)]
        row_var: String,

        /// Column variable column
        #[arg(long)]
        col_var: String,
    },

    /// Fisher's exact test for 2x2 tables
    Fisher {
        /// Dataset name
        dataset: String,

        /// Row variable column
        #[arg(long)]
        row_var: String,

        /// Column variable column
        #[arg(long)]
        col_var: String,
    },

    /// Shapiro-Wilk normality test
    Shapiro {
        /// Dataset name
        dataset: String,

        /// Column to test
        #[arg(short, long)]
        col: String,
    },

    /// Wilcoxon rank-sum test (Mann-Whitney U)
    Wilcoxon {
        /// Dataset name
        dataset: String,

        /// First column
        #[arg(long)]
        col1: String,

        /// Second column
        #[arg(long)]
        col2: String,

        /// Paired test (signed-rank) instead of rank-sum
        #[arg(long)]
        paired: bool,

        /// Alternative hypothesis
        #[arg(short, long, default_value = "two-sided")]
        alternative: TTestAlternative,
    },

    /// Bartlett's test for homogeneity of variances
    Bartlett {
        /// Dataset name
        dataset: String,

        /// Value column
        #[arg(short, long)]
        col: String,

        /// Grouping column
        #[arg(short, long)]
        group: String,
    },

    /// Kruskal-Wallis rank-sum test
    Kruskal {
        /// Dataset name
        dataset: String,

        /// Value column
        #[arg(short, long)]
        col: String,

        /// Grouping column
        #[arg(short, long)]
        group: String,
    },

    /// Five-number summary (min, Q1, median, Q3, max)
    Fivenum {
        /// Dataset name
        dataset: String,

        /// Column to summarize
        #[arg(short, long)]
        col: String,
    },

    /// Interquartile range
    Iqr {
        /// Dataset name
        dataset: String,

        /// Column to compute IQR
        #[arg(short, long)]
        col: String,
    },

    /// Median Absolute Deviation
    Mad {
        /// Dataset name
        dataset: String,

        /// Column to compute MAD
        #[arg(short, long)]
        col: String,
    },

    /// Autocorrelation function
    Acf {
        /// Dataset name
        dataset: String,

        /// Time series column
        #[arg(short, long)]
        col: String,

        /// Maximum lag
        #[arg(long, default_value = "20")]
        lag_max: usize,

        /// Compute partial ACF instead
        #[arg(long)]
        partial: bool,
    },
}

pub fn execute(
    cmd: &StatsCommands,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    match cmd {
        StatsCommands::TTestOne { dataset, col, mu, alternative, conf_level } => {
            execute_t_test_one(dataset, col, *mu, alternative.clone(), *conf_level, format, session)
        }
        StatsCommands::TTestTwo { dataset, col1, col2, alternative, equal_var } => {
            execute_t_test_two(dataset, col1, col2, alternative.clone(), *equal_var, format, session)
        }
        StatsCommands::TTestPaired { dataset, col1, col2, alternative } => {
            execute_t_test_paired(dataset, col1, col2, alternative.clone(), format, session)
        }
        StatsCommands::ChisqGof { dataset, observed } => {
            execute_chisq_gof(dataset, observed, format, session)
        }
        StatsCommands::ChisqIndep { dataset, row_var, col_var } => {
            execute_chisq_indep(dataset, row_var, col_var, format, session)
        }
        StatsCommands::Fisher { dataset, row_var, col_var } => {
            execute_fisher(dataset, row_var, col_var, format, session)
        }
        StatsCommands::Shapiro { dataset, col } => {
            execute_shapiro(dataset, col, format, session)
        }
        StatsCommands::Wilcoxon { dataset, col1, col2, paired, alternative } => {
            execute_wilcoxon(dataset, col1, col2, *paired, alternative.clone(), format, session)
        }
        StatsCommands::Bartlett { dataset, col, group } => {
            execute_bartlett(dataset, col, group, format, session)
        }
        StatsCommands::Kruskal { dataset, col, group } => {
            execute_kruskal(dataset, col, group, format, session)
        }
        StatsCommands::Fivenum { dataset, col } => {
            execute_fivenum(dataset, col, format, session)
        }
        StatsCommands::Iqr { dataset, col } => {
            execute_iqr(dataset, col, format, session)
        }
        StatsCommands::Mad { dataset, col } => {
            execute_mad(dataset, col, format, session)
        }
        StatsCommands::Acf { dataset, col, lag_max, partial } => {
            execute_acf(dataset, col, *lag_max, *partial, format, session)
        }
    }
}

/// Helper to extract a column as Vec<f64>
fn extract_column(dataset: &p2a_core::Dataset, col: &str) -> Result<Vec<f64>, String> {
    let df = dataset.df();
    let column = df
        .column(col)
        .map_err(|e| format!("Column '{}' not found: {}", col, e))?;
    let f64_col = column
        .f64()
        .map_err(|e| format!("Column '{}' must be numeric: {}", col, e))?;
    Ok(f64_col.into_no_null_iter().collect())
}

fn execute_t_test_one(
    dataset_name: &str,
    col: &str,
    mu: f64,
    alternative: TTestAlternative,
    conf_level: f64,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            let data = match extract_column(ds, col) {
                Ok(d) => d,
                Err(e) => { print_error(&e, format); return Ok(()); }
            };

            let alt: Alternative = alternative.into();
            match one_sample_t_test(&data, mu, alt, conf_level) {
                Ok(result) => {
                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "test": "One-sample t-test",
                                "t_statistic": result.t_statistic,
                                "df": result.df,
                                "p_value": result.p_value,
                                "ci_lower": result.conf_int_lower,
                                "ci_upper": result.conf_int_upper,
                                "mean": result.estimate,
                                "null_value": mu,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nOne-sample t-test");
                            println!("{}", "=".repeat(40));
                            println!("H0: mean = {:.4}", mu);
                            println!("\nt = {:.4}, df = {:.1}, p-value = {:.6}",
                                result.t_statistic, result.df, result.p_value);
                            println!("{:.0}% CI: [{:.4}, {:.4}]",
                                conf_level * 100.0, result.conf_int_lower, result.conf_int_upper);
                            println!("Sample mean: {:.4}", result.estimate);
                        }
                    }
                }
                Err(e) => print_error(&format!("t-test failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_t_test_two(
    dataset_name: &str,
    col1: &str,
    col2: &str,
    alternative: TTestAlternative,
    equal_var: bool,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            let data1 = match extract_column(ds, col1) {
                Ok(d) => d,
                Err(e) => { print_error(&e, format); return Ok(()); }
            };
            let data2 = match extract_column(ds, col2) {
                Ok(d) => d,
                Err(e) => { print_error(&e, format); return Ok(()); }
            };

            let alt: Alternative = alternative.into();
            match two_sample_t_test(&data1, &data2, 0.0, alt, equal_var, 0.95) {
                Ok(result) => {
                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "test": if equal_var { "Two-sample t-test (equal variances)" } else { "Welch's t-test" },
                                "t_statistic": result.t_statistic,
                                "df": result.df,
                                "p_value": result.p_value,
                                "ci_lower": result.conf_int_lower,
                                "ci_upper": result.conf_int_upper,
                                "mean_difference": result.estimate,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            let test_name = if equal_var { "Two-sample t-test (equal variances)" } else { "Welch's t-test" };
                            println!("\n{}", test_name);
                            println!("{}", "=".repeat(45));
                            println!("t = {:.4}, df = {:.2}, p-value = {:.6}",
                                result.t_statistic, result.df, result.p_value);
                            println!("95% CI for difference: [{:.4}, {:.4}]",
                                result.conf_int_lower, result.conf_int_upper);
                            println!("Mean difference: {:.4}", result.estimate);
                        }
                    }
                }
                Err(e) => print_error(&format!("t-test failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_t_test_paired(
    dataset_name: &str,
    col1: &str,
    col2: &str,
    alternative: TTestAlternative,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            let data1 = match extract_column(ds, col1) {
                Ok(d) => d,
                Err(e) => { print_error(&e, format); return Ok(()); }
            };
            let data2 = match extract_column(ds, col2) {
                Ok(d) => d,
                Err(e) => { print_error(&e, format); return Ok(()); }
            };

            let alt: Alternative = alternative.into();
            match paired_t_test(&data1, &data2, 0.0, alt, 0.95) {
                Ok(result) => {
                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "test": "Paired t-test",
                                "t_statistic": result.t_statistic,
                                "df": result.df,
                                "p_value": result.p_value,
                                "ci_lower": result.conf_int_lower,
                                "ci_upper": result.conf_int_upper,
                                "mean_difference": result.estimate,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nPaired t-test");
                            println!("{}", "=".repeat(40));
                            println!("t = {:.4}, df = {:.1}, p-value = {:.6}",
                                result.t_statistic, result.df, result.p_value);
                            println!("95% CI: [{:.4}, {:.4}]",
                                result.conf_int_lower, result.conf_int_upper);
                            println!("Mean difference: {:.4}", result.estimate);
                        }
                    }
                }
                Err(e) => print_error(&format!("Paired t-test failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_chisq_gof(
    dataset_name: &str,
    observed_col: &str,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            match run_chisq_gof(ds, observed_col, None) {
                Ok(result) => {
                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "test": "Chi-squared goodness-of-fit",
                                "chi_squared": result.statistic,
                                "df": result.df,
                                "p_value": result.p_value,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nChi-squared Goodness-of-Fit Test");
                            println!("{}", "=".repeat(40));
                            println!("X-squared = {:.4}, df = {}, p-value = {:.6}",
                                result.statistic, result.df, result.p_value);
                        }
                    }
                }
                Err(e) => print_error(&format!("Chi-squared test failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_chisq_indep(
    dataset_name: &str,
    row_var: &str,
    col_var: &str,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            // run_chisq_independence(dataset, row_col, col_col, correct)
            match run_chisq_independence(ds, row_var, col_var, true) {
                Ok(result) => {
                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "test": "Chi-squared test of independence",
                                "chi_squared": result.statistic,
                                "df": result.df,
                                "p_value": result.p_value,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nChi-squared Test of Independence");
                            println!("{}", "=".repeat(40));
                            println!("X-squared = {:.4}, df = {}, p-value = {:.6}",
                                result.statistic, result.df, result.p_value);
                        }
                    }
                }
                Err(e) => print_error(&format!("Chi-squared test failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_fisher(
    dataset_name: &str,
    row_var: &str,
    col_var: &str,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    use p2a_core::FisherAlternative;

    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            // run_fisher_test(dataset, row_col, col_col, alternative, conf_level)
            match run_fisher_test(ds, row_var, col_var, FisherAlternative::TwoSided, Some(0.95)) {
                Ok(result) => {
                    let (ci_lower, ci_upper) = result.odds_ratio_ci.unwrap_or((f64::NAN, f64::NAN));
                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "test": "Fisher's exact test",
                                "p_value": result.p_value,
                                "odds_ratio": result.odds_ratio,
                                "ci_lower": ci_lower,
                                "ci_upper": ci_upper,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nFisher's Exact Test for 2x2 Table");
                            println!("{}", "=".repeat(40));
                            println!("p-value = {:.6}", result.p_value);
                            println!("Odds ratio = {:.4}", result.odds_ratio);
                            if result.odds_ratio_ci.is_some() {
                                println!("95% CI: [{:.4}, {:.4}]", ci_lower, ci_upper);
                            }
                        }
                    }
                }
                Err(e) => print_error(&format!("Fisher's test failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_shapiro(
    dataset_name: &str,
    col: &str,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            // run_shapiro_wilk(dataset, column)
            match run_shapiro_wilk(ds, col) {
                Ok(result) => {
                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "test": "Shapiro-Wilk normality test",
                                "W": result.w_statistic,
                                "p_value": result.p_value,
                                "n": result.n,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nShapiro-Wilk Normality Test");
                            println!("{}", "=".repeat(40));
                            println!("W = {:.6}, p-value = {:.6}", result.w_statistic, result.p_value);
                            println!("n = {}", result.n);
                            if result.p_value < 0.05 {
                                println!("\nConclusion: Data is significantly non-normal (p < 0.05)");
                            } else {
                                println!("\nConclusion: Cannot reject normality");
                            }
                        }
                    }
                }
                Err(e) => print_error(&format!("Shapiro-Wilk test failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_wilcoxon(
    dataset_name: &str,
    col1: &str,
    col2: &str,
    paired: bool,
    alternative: TTestAlternative,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            let data1 = match extract_column(ds, col1) {
                Ok(d) => d,
                Err(e) => { print_error(&e, format); return Ok(()); }
            };
            let data2 = match extract_column(ds, col2) {
                Ok(d) => d,
                Err(e) => { print_error(&e, format); return Ok(()); }
            };

            let config = WilcoxonConfig::default();
            let alt: Alternative = alternative.into();

            let result = if paired {
                wilcoxon_signed_rank(&data1, Some(&data2), 0.0, alt, &config)
            } else {
                wilcoxon_rank_sum(&data1, &data2, 0.0, alt, &config)
            };

            match result {
                Ok(res) => {
                    let test_name = if paired { "Wilcoxon signed-rank test" } else { "Wilcoxon rank-sum test" };
                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "test": test_name,
                                "statistic": res.statistic,
                                "p_value": res.p_value,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\n{}", test_name);
                            println!("{}", "=".repeat(40));
                            println!("W = {:.4}, p-value = {:.6}", res.statistic, res.p_value);
                        }
                    }
                }
                Err(e) => print_error(&format!("Wilcoxon test failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_bartlett(
    dataset_name: &str,
    col: &str,
    group_col: &str,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            // run_bartlett_test(dataset, response_col, factor_col)
            match run_bartlett_test(ds, col, group_col) {
                Ok(result) => {
                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "test": "Bartlett's test for homogeneity of variances",
                                "statistic": result.statistic,
                                "df": result.df,
                                "p_value": result.p_value,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nBartlett's Test for Homogeneity of Variances");
                            println!("{}", "=".repeat(50));
                            println!("K-squared = {:.4}, df = {}, p-value = {:.6}",
                                result.statistic, result.df, result.p_value);
                        }
                    }
                }
                Err(e) => print_error(&format!("Bartlett's test failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_kruskal(
    dataset_name: &str,
    col: &str,
    group_col: &str,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            // run_kruskal_test(dataset, value_col, group_col)
            match run_kruskal_test(ds, col, group_col) {
                Ok(result) => {
                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "test": "Kruskal-Wallis rank sum test",
                                "statistic": result.statistic,
                                "df": result.df,
                                "p_value": result.p_value,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nKruskal-Wallis Rank Sum Test");
                            println!("{}", "=".repeat(40));
                            println!("H = {:.4}, df = {}, p-value = {:.6}",
                                result.statistic, result.df, result.p_value);
                        }
                    }
                }
                Err(e) => print_error(&format!("Kruskal-Wallis test failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_fivenum(
    dataset_name: &str,
    col: &str,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            let data = match extract_column(ds, col) {
                Ok(d) => d,
                Err(e) => { print_error(&e, format); return Ok(()); }
            };

            match run_fivenum(&data) {
                Ok(result) => {
                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "min": result.minimum,
                                "q1": result.lower_hinge,
                                "median": result.median,
                                "q3": result.upper_hinge,
                                "max": result.maximum,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nFive-Number Summary for '{}'", col);
                            println!("{}", "=".repeat(35));
                            println!("Min:    {:.4}", result.minimum);
                            println!("Q1:     {:.4}", result.lower_hinge);
                            println!("Median: {:.4}", result.median);
                            println!("Q3:     {:.4}", result.upper_hinge);
                            println!("Max:    {:.4}", result.maximum);
                        }
                    }
                }
                Err(e) => print_error(&format!("fivenum failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_iqr(
    dataset_name: &str,
    col: &str,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            let data = match extract_column(ds, col) {
                Ok(d) => d,
                Err(e) => { print_error(&e, format); return Ok(()); }
            };

            match run_iqr(&data, None) {
                Ok(iqr_val) => {
                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "iqr": iqr_val,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("IQR({}): {:.4}", col, iqr_val);
                        }
                    }
                }
                Err(e) => print_error(&format!("IQR failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_mad(
    dataset_name: &str,
    col: &str,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            let data = match extract_column(ds, col) {
                Ok(d) => d,
                Err(e) => { print_error(&e, format); return Ok(()); }
            };

            match run_mad(&data, None, None) {
                Ok(mad_val) => {
                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "mad": mad_val,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("MAD({}): {:.4}", col, mad_val);
                        }
                    }
                }
                Err(e) => print_error(&format!("MAD failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_acf(
    dataset_name: &str,
    col: &str,
    lag_max: usize,
    partial: bool,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            let data = match extract_column(ds, col) {
                Ok(d) => d,
                Err(e) => { print_error(&e, format); return Ok(()); }
            };

            if partial {
                match pacf(&data, Some(lag_max)) {
                    Ok(result) => {
                        match format {
                            OutputFormat::Json => {
                                let json = serde_json::json!({
                                    "type": "PACF",
                                    "lags": result.lags,
                                    "values": result.values,
                                });
                                println!("{}", serde_json::to_string_pretty(&json)?);
                            }
                            _ => {
                                println!("\nPartial Autocorrelation Function");
                                println!("{}", "=".repeat(40));
                                for (lag, &v) in result.lags.iter().zip(result.values.iter()) {
                                    println!("  Lag {}: {:.4}", lag, v);
                                }
                            }
                        }
                    }
                    Err(e) => print_error(&format!("PACF failed: {}", e), format),
                }
            } else {
                // acf(data, lag_max, acf_type, demean, na_rm)
                match acf(&data, Some(lag_max), AcfType::Correlation, true, true) {
                    Ok(result) => {
                        match format {
                            OutputFormat::Json => {
                                let json = serde_json::json!({
                                    "type": "ACF",
                                    "lags": result.lags,
                                    "values": result.values,
                                });
                                println!("{}", serde_json::to_string_pretty(&json)?);
                            }
                            _ => {
                                println!("\nAutocorrelation Function");
                                println!("{}", "=".repeat(40));
                                for (lag, &v) in result.lags.iter().zip(result.values.iter()) {
                                    println!("  Lag {}: {:.4}", lag, v);
                                }
                            }
                        }
                    }
                    Err(e) => print_error(&format!("ACF failed: {}", e), format),
                }
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}
