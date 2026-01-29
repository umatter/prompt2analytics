//! Statistical tests and descriptive statistics commands

use clap::{Subcommand, ValueEnum};
use p2a_core::{
    AcfType,
    Alternative,
    BoxTestType,
    CcfType,
    WilcoxonConfig,
    // ACF/PACF/CCF (raw data versions)
    acf,
    // T-tests
    one_sample_t_test,
    pacf,
    paired_t_test,
    run_bartlett_test,
    // Box test (Ljung-Box)
    run_box_test,
    run_ccf,
    // Chi-squared tests
    run_chisq_gof,
    run_chisq_independence,
    run_fisher_test,
    // Descriptive
    run_fivenum,
    run_friedman_test,
    run_iqr,
    run_kruskal_test,
    // Kolmogorov-Smirnov test
    run_ks_test,
    run_mad,
    // ANOVA
    run_one_way_anova,
    // Power analysis
    run_power_t_test,
    // Phillips-Perron test
    run_pp_test,
    // Other tests
    run_shapiro_wilk,
    // Tukey HSD
    run_tukey_hsd,
    run_two_way_anova,
    two_sample_t_test,
    wilcoxon_rank_sum,
    wilcoxon_signed_rank,
};

use crate::output::{OutputFormat, print_error};
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

    /// One-way ANOVA
    Anova {
        /// Dataset name
        dataset: String,

        /// Response (dependent) variable column
        #[arg(short, long)]
        response: String,

        /// Factor (grouping) variable column
        #[arg(short, long)]
        factor: String,
    },

    /// Two-way ANOVA
    AnovaTwo {
        /// Dataset name
        dataset: String,

        /// Response (dependent) variable column
        #[arg(short, long)]
        response: String,

        /// First factor (grouping) variable column
        #[arg(long)]
        factor_a: String,

        /// Second factor (grouping) variable column
        #[arg(long)]
        factor_b: String,

        /// Include interaction term
        #[arg(long)]
        interaction: bool,
    },

    /// Tukey's HSD post-hoc test
    Tukey {
        /// Dataset name
        dataset: String,

        /// Response (dependent) variable column
        #[arg(short, long)]
        response: String,

        /// Factor (grouping) variable column
        #[arg(short, long)]
        factor: String,

        /// Confidence level
        #[arg(long, default_value = "0.95")]
        conf_level: f64,
    },

    /// Friedman rank sum test for blocked data
    Friedman {
        /// Dataset name
        dataset: String,

        /// Value column (measurements)
        #[arg(short, long)]
        value: String,

        /// Group/treatment column
        #[arg(short, long)]
        group: String,

        /// Block column (e.g., subject ID)
        #[arg(short, long)]
        block: String,
    },

    /// Partial autocorrelation function
    Pacf {
        /// Dataset name
        dataset: String,

        /// Time series column
        #[arg(short, long)]
        col: String,

        /// Maximum lag
        #[arg(long, default_value = "20")]
        lag_max: usize,
    },

    /// Cross-correlation function
    Ccf {
        /// Dataset name
        dataset: String,

        /// First time series column
        #[arg(long)]
        x: String,

        /// Second time series column
        #[arg(long)]
        y: String,

        /// Maximum lag
        #[arg(long, default_value = "20")]
        lag_max: usize,
    },

    /// Ljung-Box or Box-Pierce test for autocorrelation
    BoxTest {
        /// Dataset name
        dataset: String,

        /// Time series column
        #[arg(short, long)]
        col: String,

        /// Maximum lag to test
        #[arg(long, default_value = "10")]
        lag: usize,

        /// Use Box-Pierce instead of Ljung-Box
        #[arg(long)]
        box_pierce: bool,

        /// Degrees of freedom to subtract (for ARMA residuals)
        #[arg(long, default_value = "0")]
        fitdf: usize,
    },

    /// Kolmogorov-Smirnov two-sample test
    KsTest {
        /// Dataset name
        dataset: String,

        /// First sample column
        #[arg(long)]
        x: String,

        /// Second sample column
        #[arg(long)]
        y: String,

        /// Alternative hypothesis
        #[arg(short, long, default_value = "two-sided")]
        alternative: TTestAlternative,
    },

    /// Phillips-Perron unit root test
    PpTest {
        /// Dataset name
        dataset: String,

        /// Time series column
        #[arg(short, long)]
        col: String,

        /// Use short truncation lag
        #[arg(long)]
        lshort: bool,
    },

    /// Power analysis for t-test
    PowerT {
        /// Sample size per group (leave unset to solve for it)
        #[arg(short, long)]
        n: Option<f64>,

        /// True difference in means (leave unset to solve for it)
        #[arg(short, long)]
        delta: Option<f64>,

        /// Standard deviation
        #[arg(long, default_value = "1.0")]
        sd: f64,

        /// Significance level
        #[arg(long, default_value = "0.05")]
        sig_level: f64,

        /// Desired power (leave unset to solve for it)
        #[arg(short, long)]
        power: Option<f64>,

        /// Test type: one-sample, two-sample, or paired
        #[arg(long, default_value = "two-sample")]
        test_type: PowerTestType,

        /// Alternative: two-sided or one-sided
        #[arg(long, default_value = "two-sided")]
        alternative: PowerAlt,
    },
}

#[derive(Clone, ValueEnum)]
pub enum PowerTestType {
    OneSample,
    TwoSample,
    Paired,
}

#[derive(Clone, ValueEnum)]
pub enum PowerAlt {
    TwoSided,
    OneSided,
}

pub fn execute(
    cmd: &StatsCommands,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    match cmd {
        StatsCommands::TTestOne {
            dataset,
            col,
            mu,
            alternative,
            conf_level,
        } => execute_t_test_one(
            dataset,
            col,
            *mu,
            alternative.clone(),
            *conf_level,
            format,
            session,
        ),
        StatsCommands::TTestTwo {
            dataset,
            col1,
            col2,
            alternative,
            equal_var,
        } => execute_t_test_two(
            dataset,
            col1,
            col2,
            alternative.clone(),
            *equal_var,
            format,
            session,
        ),
        StatsCommands::TTestPaired {
            dataset,
            col1,
            col2,
            alternative,
        } => execute_t_test_paired(dataset, col1, col2, alternative.clone(), format, session),
        StatsCommands::ChisqGof { dataset, observed } => {
            execute_chisq_gof(dataset, observed, format, session)
        }
        StatsCommands::ChisqIndep {
            dataset,
            row_var,
            col_var,
        } => execute_chisq_indep(dataset, row_var, col_var, format, session),
        StatsCommands::Fisher {
            dataset,
            row_var,
            col_var,
        } => execute_fisher(dataset, row_var, col_var, format, session),
        StatsCommands::Shapiro { dataset, col } => execute_shapiro(dataset, col, format, session),
        StatsCommands::Wilcoxon {
            dataset,
            col1,
            col2,
            paired,
            alternative,
        } => execute_wilcoxon(
            dataset,
            col1,
            col2,
            *paired,
            alternative.clone(),
            format,
            session,
        ),
        StatsCommands::Bartlett {
            dataset,
            col,
            group,
        } => execute_bartlett(dataset, col, group, format, session),
        StatsCommands::Kruskal {
            dataset,
            col,
            group,
        } => execute_kruskal(dataset, col, group, format, session),
        StatsCommands::Fivenum { dataset, col } => execute_fivenum(dataset, col, format, session),
        StatsCommands::Iqr { dataset, col } => execute_iqr(dataset, col, format, session),
        StatsCommands::Mad { dataset, col } => execute_mad(dataset, col, format, session),
        StatsCommands::Acf {
            dataset,
            col,
            lag_max,
            partial,
        } => execute_acf(dataset, col, *lag_max, *partial, format, session),
        StatsCommands::Anova {
            dataset,
            response,
            factor,
        } => execute_anova(dataset, response, factor, format, session),
        StatsCommands::AnovaTwo {
            dataset,
            response,
            factor_a,
            factor_b,
            interaction,
        } => execute_anova_two(
            dataset,
            response,
            factor_a,
            factor_b,
            *interaction,
            format,
            session,
        ),
        StatsCommands::Tukey {
            dataset,
            response,
            factor,
            conf_level,
        } => execute_tukey(dataset, response, factor, *conf_level, format, session),
        StatsCommands::Friedman {
            dataset,
            value,
            group,
            block,
        } => execute_friedman(dataset, value, group, block, format, session),
        StatsCommands::Pacf {
            dataset,
            col,
            lag_max,
        } => execute_pacf(dataset, col, *lag_max, format, session),
        StatsCommands::Ccf {
            dataset,
            x,
            y,
            lag_max,
        } => execute_ccf(dataset, x, y, *lag_max, format, session),
        StatsCommands::BoxTest {
            dataset,
            col,
            lag,
            box_pierce,
            fitdf,
        } => execute_box_test(dataset, col, *lag, *box_pierce, *fitdf, format, session),
        StatsCommands::KsTest {
            dataset,
            x,
            y,
            alternative,
        } => execute_ks_test(dataset, x, y, alternative.clone(), format, session),
        StatsCommands::PpTest {
            dataset,
            col,
            lshort,
        } => execute_pp_test(dataset, col, *lshort, format, session),
        StatsCommands::PowerT {
            n,
            delta,
            sd,
            sig_level,
            power,
            test_type,
            alternative,
        } => execute_power_t(
            *n,
            *delta,
            *sd,
            *sig_level,
            *power,
            test_type.clone(),
            alternative.clone(),
            format,
        ),
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
                Err(e) => {
                    print_error(&e, format);
                    return Ok(());
                }
            };

            let alt: Alternative = alternative.into();
            match one_sample_t_test(&data, mu, alt, conf_level) {
                Ok(result) => match format {
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
                        println!(
                            "\nt = {:.4}, df = {:.1}, p-value = {:.6}",
                            result.t_statistic, result.df, result.p_value
                        );
                        println!(
                            "{:.0}% CI: [{:.4}, {:.4}]",
                            conf_level * 100.0,
                            result.conf_int_lower,
                            result.conf_int_upper
                        );
                        println!("Sample mean: {:.4}", result.estimate);
                    }
                },
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
                Err(e) => {
                    print_error(&e, format);
                    return Ok(());
                }
            };
            let data2 = match extract_column(ds, col2) {
                Ok(d) => d,
                Err(e) => {
                    print_error(&e, format);
                    return Ok(());
                }
            };

            let alt: Alternative = alternative.into();
            match two_sample_t_test(&data1, &data2, 0.0, alt, equal_var, 0.95) {
                Ok(result) => match format {
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
                        let test_name = if equal_var {
                            "Two-sample t-test (equal variances)"
                        } else {
                            "Welch's t-test"
                        };
                        println!("\n{}", test_name);
                        println!("{}", "=".repeat(45));
                        println!(
                            "t = {:.4}, df = {:.2}, p-value = {:.6}",
                            result.t_statistic, result.df, result.p_value
                        );
                        println!(
                            "95% CI for difference: [{:.4}, {:.4}]",
                            result.conf_int_lower, result.conf_int_upper
                        );
                        println!("Mean difference: {:.4}", result.estimate);
                    }
                },
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
                Err(e) => {
                    print_error(&e, format);
                    return Ok(());
                }
            };
            let data2 = match extract_column(ds, col2) {
                Ok(d) => d,
                Err(e) => {
                    print_error(&e, format);
                    return Ok(());
                }
            };

            let alt: Alternative = alternative.into();
            match paired_t_test(&data1, &data2, 0.0, alt, 0.95) {
                Ok(result) => match format {
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
                        println!(
                            "t = {:.4}, df = {:.1}, p-value = {:.6}",
                            result.t_statistic, result.df, result.p_value
                        );
                        println!(
                            "95% CI: [{:.4}, {:.4}]",
                            result.conf_int_lower, result.conf_int_upper
                        );
                        println!("Mean difference: {:.4}", result.estimate);
                    }
                },
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
        Some(ds) => match run_chisq_gof(ds, observed_col, None) {
            Ok(result) => match format {
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
                    println!(
                        "X-squared = {:.4}, df = {}, p-value = {:.6}",
                        result.statistic, result.df, result.p_value
                    );
                }
            },
            Err(e) => print_error(&format!("Chi-squared test failed: {}", e), format),
        },
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
                Ok(result) => match format {
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
                        println!(
                            "X-squared = {:.4}, df = {}, p-value = {:.6}",
                            result.statistic, result.df, result.p_value
                        );
                    }
                },
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
            match run_fisher_test(
                ds,
                row_var,
                col_var,
                FisherAlternative::TwoSided,
                Some(0.95),
            ) {
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
                Ok(result) => match format {
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
                        println!(
                            "W = {:.6}, p-value = {:.6}",
                            result.w_statistic, result.p_value
                        );
                        println!("n = {}", result.n);
                        if result.p_value < 0.05 {
                            println!("\nConclusion: Data is significantly non-normal (p < 0.05)");
                        } else {
                            println!("\nConclusion: Cannot reject normality");
                        }
                    }
                },
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
                Err(e) => {
                    print_error(&e, format);
                    return Ok(());
                }
            };
            let data2 = match extract_column(ds, col2) {
                Ok(d) => d,
                Err(e) => {
                    print_error(&e, format);
                    return Ok(());
                }
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
                    let test_name = if paired {
                        "Wilcoxon signed-rank test"
                    } else {
                        "Wilcoxon rank-sum test"
                    };
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
                Ok(result) => match format {
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
                        println!(
                            "K-squared = {:.4}, df = {}, p-value = {:.6}",
                            result.statistic, result.df, result.p_value
                        );
                    }
                },
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
                Ok(result) => match format {
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
                        println!(
                            "H = {:.4}, df = {}, p-value = {:.6}",
                            result.statistic, result.df, result.p_value
                        );
                    }
                },
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
                Err(e) => {
                    print_error(&e, format);
                    return Ok(());
                }
            };

            match run_fivenum(&data) {
                Ok(result) => match format {
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
                },
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
                Err(e) => {
                    print_error(&e, format);
                    return Ok(());
                }
            };

            match run_iqr(&data, None) {
                Ok(iqr_val) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "iqr": iqr_val,
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("IQR({}): {:.4}", col, iqr_val);
                    }
                },
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
                Err(e) => {
                    print_error(&e, format);
                    return Ok(());
                }
            };

            match run_mad(&data, None, None) {
                Ok(mad_val) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "mad": mad_val,
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("MAD({}): {:.4}", col, mad_val);
                    }
                },
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
                Err(e) => {
                    print_error(&e, format);
                    return Ok(());
                }
            };

            if partial {
                match pacf(&data, Some(lag_max)) {
                    Ok(result) => match format {
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
                    },
                    Err(e) => print_error(&format!("PACF failed: {}", e), format),
                }
            } else {
                // acf(data, lag_max, acf_type, demean, na_rm)
                match acf(&data, Some(lag_max), AcfType::Correlation, true, true) {
                    Ok(result) => match format {
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
                    },
                    Err(e) => print_error(&format!("ACF failed: {}", e), format),
                }
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_anova(
    dataset_name: &str,
    response: &str,
    factor: &str,
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
        Some(ds) => match run_one_way_anova(ds, response, factor) {
            Ok(result) => match format {
                OutputFormat::Json => {
                    let json = serde_json::json!({
                        "test": "One-way ANOVA",
                        "response": result.response_var,
                        "factor": result.factor_var,
                        "ss_between": result.ss_between,
                        "ss_within": result.ss_within,
                        "ss_total": result.ss_total,
                        "df_between": result.df_between,
                        "df_within": result.df_within,
                        "ms_between": result.ms_between,
                        "ms_within": result.ms_within,
                        "f_statistic": result.f_statistic,
                        "p_value": result.p_value,
                        "eta_squared": result.eta_squared,
                        "n_groups": result.n_groups,
                        "n_obs": result.n_obs,
                    });
                    println!("{}", serde_json::to_string_pretty(&json)?);
                }
                _ => {
                    println!("\nOne-Way ANOVA");
                    println!("{}", "=".repeat(60));
                    println!(
                        "Response: {}  |  Factor: {}",
                        result.response_var, result.factor_var
                    );
                    println!("N = {}  |  Groups = {}", result.n_obs, result.n_groups);
                    println!();
                    println!(
                        "{:>12} {:>12} {:>8} {:>12} {:>10} {:>10}",
                        "Source", "SS", "DF", "MS", "F", "Pr(>F)"
                    );
                    println!("{}", "-".repeat(66));
                    println!(
                        "{:>12} {:>12.4} {:>8} {:>12.4} {:>10.4} {:>10.6}",
                        "Between",
                        result.ss_between,
                        result.df_between,
                        result.ms_between,
                        result.f_statistic,
                        result.p_value
                    );
                    println!(
                        "{:>12} {:>12.4} {:>8} {:>12.4}",
                        "Within", result.ss_within, result.df_within, result.ms_within
                    );
                    println!(
                        "{:>12} {:>12.4} {:>8}",
                        "Total", result.ss_total, result.df_total
                    );
                    println!();
                    println!("Effect size: eta-squared = {:.4}", result.eta_squared);
                }
            },
            Err(e) => print_error(&format!("ANOVA failed: {}", e), format),
        },
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_anova_two(
    dataset_name: &str,
    response: &str,
    factor_a: &str,
    factor_b: &str,
    interaction: bool,
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
        Some(ds) => match run_two_way_anova(ds, response, factor_a, factor_b, interaction) {
            Ok(result) => match format {
                OutputFormat::Json => {
                    let json = serde_json::json!({
                        "test": "Two-way ANOVA",
                        "response": result.response_var,
                        "factor_a": result.factor_a,
                        "factor_b": result.factor_b,
                        "with_interaction": result.with_interaction,
                        "ss_a": result.ss_a,
                        "ss_b": result.ss_b,
                        "ss_ab": result.ss_ab,
                        "ss_error": result.ss_error,
                        "ss_total": result.ss_total,
                        "f_a": result.f_a,
                        "f_b": result.f_b,
                        "f_ab": result.f_ab,
                        "p_a": result.p_a,
                        "p_b": result.p_b,
                        "p_ab": result.p_ab,
                        "n_obs": result.n_obs,
                    });
                    println!("{}", serde_json::to_string_pretty(&json)?);
                }
                _ => {
                    println!("\nTwo-Way ANOVA");
                    println!("{}", "=".repeat(70));
                    println!(
                        "Response: {}  |  Factors: {}, {}",
                        result.response_var, result.factor_a, result.factor_b
                    );
                    println!(
                        "N = {}  |  Interaction: {}",
                        result.n_obs,
                        if result.with_interaction { "Yes" } else { "No" }
                    );
                    println!();
                    println!(
                        "{:>15} {:>12} {:>6} {:>12} {:>10} {:>10}",
                        "Source", "SS", "DF", "MS", "F", "Pr(>F)"
                    );
                    println!("{}", "-".repeat(70));
                    println!(
                        "{:>15} {:>12.4} {:>6} {:>12.4} {:>10.4} {:>10.6}",
                        &result.factor_a,
                        result.ss_a,
                        result.df_a,
                        result.ms_a,
                        result.f_a,
                        result.p_a
                    );
                    println!(
                        "{:>15} {:>12.4} {:>6} {:>12.4} {:>10.4} {:>10.6}",
                        &result.factor_b,
                        result.ss_b,
                        result.df_b,
                        result.ms_b,
                        result.f_b,
                        result.p_b
                    );
                    if result.with_interaction {
                        if let (Some(ss_ab), Some(df_ab), Some(ms_ab), Some(f_ab), Some(p_ab)) = (
                            result.ss_ab,
                            result.df_ab,
                            result.ms_ab,
                            result.f_ab,
                            result.p_ab,
                        ) {
                            println!(
                                "{:>15} {:>12.4} {:>6} {:>12.4} {:>10.4} {:>10.6}",
                                "Interaction", ss_ab, df_ab, ms_ab, f_ab, p_ab
                            );
                        }
                    }
                    println!(
                        "{:>15} {:>12.4} {:>6} {:>12.4}",
                        "Error", result.ss_error, result.df_error, result.ms_error
                    );
                    println!(
                        "{:>15} {:>12.4} {:>6}",
                        "Total", result.ss_total, result.df_total
                    );
                }
            },
            Err(e) => print_error(&format!("Two-way ANOVA failed: {}", e), format),
        },
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_tukey(
    dataset_name: &str,
    response: &str,
    factor: &str,
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
        Some(ds) => match run_tukey_hsd(ds, response, factor, conf_level) {
            Ok((anova_result, tukey_result)) => match format {
                OutputFormat::Json => {
                    let comparisons: Vec<_> = tukey_result
                        .comparisons
                        .iter()
                        .map(|c| {
                            serde_json::json!({
                                "group1": c.group1,
                                "group2": c.group2,
                                "diff": c.diff,
                                "ci_lower": c.ci_lower,
                                "ci_upper": c.ci_upper,
                                "p_adj": c.p_adj,
                            })
                        })
                        .collect();
                    let json = serde_json::json!({
                        "test": "Tukey HSD",
                        "response": tukey_result.response_var,
                        "factor": tukey_result.factor_var,
                        "conf_level": tukey_result.conf_level,
                        "mse": tukey_result.mse,
                        "df": tukey_result.df,
                        "f_statistic": anova_result.f_statistic,
                        "anova_p_value": anova_result.p_value,
                        "comparisons": comparisons,
                    });
                    println!("{}", serde_json::to_string_pretty(&json)?);
                }
                _ => {
                    println!("\nTukey's HSD Test");
                    println!("{}", "=".repeat(70));
                    println!(
                        "Response: {}  |  Factor: {}",
                        tukey_result.response_var, tukey_result.factor_var
                    );
                    println!("Confidence level: {:.0}%", tukey_result.conf_level * 100.0);
                    println!("MSE = {:.4}, df = {}", tukey_result.mse, tukey_result.df);
                    println!();
                    println!(
                        "{:>20} {:>10} {:>12} {:>12} {:>10}",
                        "Comparison", "Diff", "CI Lower", "CI Upper", "p (adj)"
                    );
                    println!("{}", "-".repeat(70));
                    for c in &tukey_result.comparisons {
                        let stars = c.significance.stars();
                        println!(
                            "{:>10}-{:<10} {:>10.4} {:>12.4} {:>12.4} {:>10.4} {}",
                            c.group1, c.group2, c.diff, c.ci_lower, c.ci_upper, c.p_adj, stars
                        );
                    }
                }
            },
            Err(e) => print_error(&format!("Tukey HSD failed: {}", e), format),
        },
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_friedman(
    dataset_name: &str,
    value_col: &str,
    group_col: &str,
    block_col: &str,
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
        Some(ds) => match run_friedman_test(ds, value_col, group_col, block_col) {
            Ok(result) => match format {
                OutputFormat::Json => {
                    let json = serde_json::json!({
                        "test": "Friedman rank sum test",
                        "statistic": result.statistic,
                        "df": result.df,
                        "p_value": result.p_value,
                        "n_blocks": result.n_blocks,
                        "n_treatments": result.n_treatments,
                        "treatment_names": result.treatment_names,
                        "rank_sums": result.rank_sums,
                        "mean_ranks": result.mean_ranks,
                    });
                    println!("{}", serde_json::to_string_pretty(&json)?);
                }
                _ => {
                    println!("\nFriedman Rank Sum Test");
                    println!("{}", "=".repeat(50));
                    println!(
                        "Blocks: {}  |  Treatments: {}",
                        result.n_blocks, result.n_treatments
                    );
                    println!();
                    println!(
                        "Chi-squared = {:.4}, df = {}, p-value = {:.6}",
                        result.statistic, result.df, result.p_value
                    );
                    println!();
                    println!("Mean ranks by treatment:");
                    for (name, rank) in result.treatment_names.iter().zip(result.mean_ranks.iter())
                    {
                        println!("  {}: {:.4}", name, rank);
                    }
                }
            },
            Err(e) => print_error(&format!("Friedman test failed: {}", e), format),
        },
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_pacf(
    dataset_name: &str,
    col: &str,
    lag_max: usize,
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
                Err(e) => {
                    print_error(&e, format);
                    return Ok(());
                }
            };

            match pacf(&data, Some(lag_max)) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "type": "PACF",
                            "lags": result.lags,
                            "values": result.values,
                            "n_obs": result.n_obs,
                            "confidence_bound": result.confidence_bound,
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nPartial Autocorrelation Function");
                        println!("{}", "=".repeat(40));
                        println!("Observations: {}", result.n_obs);
                        println!("95% CI: +/- {:.4}", result.confidence_bound);
                        println!();
                        for (lag, &v) in result.lags.iter().zip(result.values.iter()) {
                            let sig = if v.abs() > result.confidence_bound {
                                "*"
                            } else {
                                ""
                            };
                            println!("  Lag {:2}: {:>7.4} {}", lag, v, sig);
                        }
                    }
                },
                Err(e) => print_error(&format!("PACF failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_ccf(
    dataset_name: &str,
    x_col: &str,
    y_col: &str,
    lag_max: usize,
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
        Some(ds) => match run_ccf(ds, x_col, y_col, Some(lag_max), CcfType::Correlation) {
            Ok(result) => match format {
                OutputFormat::Json => {
                    let json = serde_json::json!({
                        "type": "CCF",
                        "x_series": result.x_series,
                        "y_series": result.y_series,
                        "lags": result.lags,
                        "values": result.values,
                        "n_obs": result.n_obs,
                    });
                    println!("{}", serde_json::to_string_pretty(&json)?);
                }
                _ => {
                    println!("\nCross-Correlation Function");
                    println!("{}", "=".repeat(45));
                    if let (Some(x), Some(y)) = (&result.x_series, &result.y_series) {
                        println!("X: {}  |  Y: {}", x, y);
                    }
                    println!("Observations: {}", result.n_obs);
                    println!();
                    for (lag, &v) in result.lags.iter().zip(result.values.iter()) {
                        println!("  Lag {:3}: {:>7.4}", lag, v);
                    }
                }
            },
            Err(e) => print_error(&format!("CCF failed: {}", e), format),
        },
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_box_test(
    dataset_name: &str,
    col: &str,
    lag: usize,
    box_pierce: bool,
    fitdf: usize,
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
            let test_type = if box_pierce {
                BoxTestType::BoxPierce
            } else {
                BoxTestType::LjungBox
            };
            match run_box_test(ds, col, Some(lag), test_type, fitdf) {
                Ok(result) => {
                    let test_name = format!("{} test", result.test_type);
                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "test": test_name,
                                "statistic": result.statistic,
                                "df": result.df,
                                "p_value": result.p_value,
                                "lag": result.lag,
                                "fitdf": result.fitdf,
                                "n_obs": result.n_obs,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\n{}", test_name);
                            println!("{}", "=".repeat(45));
                            println!(
                                "X-squared = {:.4}, df = {}, p-value = {:.6}",
                                result.statistic, result.df, result.p_value
                            );
                            println!("Lag = {}, fitdf = {}", result.lag, result.fitdf);
                        }
                    }
                }
                Err(e) => print_error(&format!("Box test failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_ks_test(
    dataset_name: &str,
    x_col: &str,
    y_col: &str,
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
            let alt: Alternative = alternative.into();
            match run_ks_test(ds, x_col, y_col, alt) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "test": result.test_name,
                            "statistic": result.statistic,
                            "p_value": result.p_value,
                            "alternative": format!("{:?}", result.alternative),
                            "exact": result.exact,
                            "n": result.n,
                            "n_2": result.n_2,
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\n{}", result.test_name);
                        println!("{}", "=".repeat(50));
                        println!(
                            "D = {:.6}, p-value = {:.6}",
                            result.statistic, result.p_value
                        );
                        println!("Alternative: {:?}", result.alternative);
                        if let Some(n2) = result.n_2 {
                            println!("Sample sizes: n = {}, m = {}", result.n, n2);
                        } else {
                            println!("Sample size: n = {}", result.n);
                        }
                    }
                },
                Err(e) => print_error(&format!("K-S test failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_pp_test(
    dataset_name: &str,
    col: &str,
    lshort: bool,
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
        Some(ds) => match run_pp_test(ds, col, lshort) {
            Ok(result) => match format {
                OutputFormat::Json => {
                    let json = serde_json::json!({
                        "test": "Phillips-Perron unit root test",
                        "statistic": result.statistic,
                        "truncation_lag": result.truncation_lag,
                        "p_value": result.p_value,
                        "n_obs": result.n_obs,
                        "lshort": result.lshort,
                    });
                    println!("{}", serde_json::to_string_pretty(&json)?);
                }
                _ => {
                    println!("\nPhillips-Perron Unit Root Test");
                    println!("{}", "=".repeat(50));
                    println!(
                        "Dickey-Fuller Z(tau) = {:.4}, Truncation lag = {}, p-value = {:.4}",
                        result.statistic, result.truncation_lag, result.p_value
                    );
                    println!("Observations: {}", result.n_obs);
                    if result.p_value < 0.05 {
                        println!("\nConclusion: Reject unit root (series appears stationary)");
                    } else {
                        println!(
                            "\nConclusion: Cannot reject unit root (series may be non-stationary)"
                        );
                    }
                }
            },
            Err(e) => print_error(&format!("Phillips-Perron test failed: {}", e), format),
        },
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_power_t(
    n: Option<f64>,
    delta: Option<f64>,
    sd: f64,
    sig_level: f64,
    power: Option<f64>,
    test_type: PowerTestType,
    alternative: PowerAlt,
    format: &OutputFormat,
) -> anyhow::Result<()> {
    let test_type_str = match test_type {
        PowerTestType::OneSample => "one.sample",
        PowerTestType::TwoSample => "two.sample",
        PowerTestType::Paired => "paired",
    };
    let alt_str = match alternative {
        PowerAlt::TwoSided => "two.sided",
        PowerAlt::OneSided => "one.sided",
    };

    match run_power_t_test(
        n,
        delta,
        Some(sd),
        Some(sig_level),
        power,
        test_type_str,
        alt_str,
    ) {
        Ok(result) => match format {
            OutputFormat::Json => {
                let json = serde_json::json!({
                    "method": result.method,
                    "n": result.n,
                    "delta": result.delta,
                    "sd": result.sd,
                    "sig_level": result.sig_level,
                    "power": result.power,
                    "alternative": format!("{:?}", result.alternative),
                    "test_type": format!("{:?}", result.test_type),
                    "note": result.note,
                });
                println!("{}", serde_json::to_string_pretty(&json)?);
            }
            _ => {
                println!("\n{}", result.method);
                println!("{}", "=".repeat(50));
                println!("              n = {:.4}", result.n);
                println!("          delta = {:.4}", result.delta);
                println!("             sd = {:.4}", result.sd);
                println!("      sig.level = {:.4}", result.sig_level);
                println!("          power = {:.4}", result.power);
                println!("    alternative = {:?}", result.alternative);
                if let Some(note) = &result.note {
                    println!("\nNOTE: {}", note);
                }
            }
        },
        Err(e) => print_error(&format!("Power analysis failed: {}", e), format),
    }
    Ok(())
}
