//! Regression analysis commands

use clap::{Subcommand, ValueEnum};
use p2a_core::{run_ols, run_ols_clustered, run_diagnostics, LinearEstimator};
use p2a_core::{run_quantreg, run_loess};
use p2a_core::regression::CovarianceType;

use crate::output::{format_regression_results, print_error, OutputFormat};
use crate::session::SessionManager;

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum RobustSE {
    Standard,
    HC0,
    HC1,
    HC2,
    HC3,
}

impl From<RobustSE> for CovarianceType {
    fn from(val: RobustSE) -> Self {
        match val {
            RobustSE::Standard => CovarianceType::Standard,
            RobustSE::HC0 => CovarianceType::HC0,
            RobustSE::HC1 => CovarianceType::HC1,
            RobustSE::HC2 => CovarianceType::HC2,
            RobustSE::HC3 => CovarianceType::HC3,
        }
    }
}

#[derive(Subcommand)]
pub enum RegressionCommands {
    /// Ordinary Least Squares regression
    Ols {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,

        /// Include intercept
        #[arg(long, default_value = "true")]
        intercept: bool,

        /// Robust standard errors type
        #[arg(short, long, default_value = "hc1")]
        robust: RobustSE,
    },

    /// Clustered standard errors regression
    Clustered {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,

        /// Cluster variable column
        #[arg(long)]
        cluster: String,

        /// Include intercept
        #[arg(long, default_value = "true")]
        intercept: bool,
    },

    /// Run regression diagnostics
    Diagnostics {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,
    },

    /// Quantile regression
    Quantreg {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,

        /// Quantile (tau) value (default: 0.5 = median)
        #[arg(short = 't', long, default_value = "0.5")]
        tau: f64,

        /// Include intercept
        #[arg(long, default_value = "true")]
        intercept: bool,
    },

    /// LOESS (local polynomial regression)
    Loess {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable column (single predictor)
        #[arg(short = 'x', long)]
        indep_var: String,

        /// Span (smoothing parameter, 0-1, default: 0.75)
        #[arg(long, default_value = "0.75")]
        span: f64,

        /// Polynomial degree (1 = local linear, 2 = local quadratic)
        #[arg(long, default_value = "2")]
        degree: usize,
    },
}

pub fn execute(
    cmd: &RegressionCommands,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    match cmd {
        RegressionCommands::Ols {
            dataset,
            dep_var,
            indep_vars,
            intercept,
            robust,
        } => execute_ols(dataset, dep_var, indep_vars, *intercept, *robust, format, session),
        RegressionCommands::Clustered {
            dataset,
            dep_var,
            indep_vars,
            cluster,
            intercept,
        } => execute_clustered(dataset, dep_var, indep_vars, cluster, *intercept, format, session),
        RegressionCommands::Diagnostics {
            dataset,
            dep_var,
            indep_vars,
        } => execute_diagnostics(dataset, dep_var, indep_vars, format, session),
        RegressionCommands::Quantreg {
            dataset,
            dep_var,
            indep_vars,
            tau,
            intercept: _,
        } => execute_quantreg(dataset, dep_var, indep_vars, *tau, format, session),
        RegressionCommands::Loess {
            dataset,
            dep_var,
            indep_var,
            span,
            degree,
        } => execute_loess(dataset, dep_var, indep_var, *span, *degree, format, session),
    }
}

fn execute_ols(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    intercept: bool,
    robust: RobustSE,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error(
                "No session active. Use --session <file> to enable dataset storage.",
                format,
            );
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            let x_cols: Vec<&str> = indep_vars.iter().map(|s| s.as_str()).collect();
            let cov_type: CovarianceType = robust.into();

            match run_ols(ds, dep_var, &x_cols, intercept, cov_type) {
                Ok(result) => {
                    // Build coefficient table
                    let coeffs = result.coefficients();
                    let ses = result.std_errors();
                    let p_vals = result.p_values();

                    // Calculate t-values manually
                    let t_vals: Vec<f64> = coeffs.iter()
                        .zip(ses.iter())
                        .map(|(c, s)| if *s > 0.0 { c / s } else { 0.0 })
                        .collect();

                    let mut var_names = Vec::new();
                    if intercept {
                        var_names.push("(Intercept)".to_string());
                    }
                    var_names.extend(indep_vars.iter().cloned());

                    let coef_table: Vec<(String, f64, f64, f64, f64)> = var_names
                        .iter()
                        .enumerate()
                        .map(|(i, name)| {
                            (
                                name.clone(),
                                coeffs[i],
                                ses[i],
                                t_vals[i],
                                p_vals[i],
                            )
                        })
                        .collect();

                    let output = format_regression_results(
                        "OLS Regression",
                        &coef_table,
                        result.r_squared(),
                        result.adj_r_squared(),
                        result.n_obs(),
                        format,
                    );
                    println!("{}", output);
                }
                Err(e) => {
                    print_error(&format!("Regression failed: {}", e), format);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

fn execute_clustered(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    cluster: &str,
    intercept: bool,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error(
                "No session active. Use --session <file> to enable dataset storage.",
                format,
            );
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            let x_cols: Vec<&str> = indep_vars.iter().map(|s| s.as_str()).collect();

            // run_ols_clustered takes cluster1 and optional cluster2 for two-way clustering
            match run_ols_clustered(ds, dep_var, &x_cols, cluster, None) {
                Ok(result) => {
                    // OlsClusteredResult has .ols field which implements LinearEstimator
                    let coeffs = result.ols.coefficients();
                    let ses = result.ols.std_errors();
                    let p_vals = result.ols.p_values();
                    let t_vals: Vec<f64> = coeffs.iter()
                        .zip(ses.iter())
                        .map(|(c, s)| if *s > 0.0 { c / s } else { 0.0 })
                        .collect();

                    // Intercept is always included by run_ols_clustered
                    let mut var_names = vec!["(Intercept)".to_string()];
                    var_names.extend(indep_vars.iter().cloned());

                    let coef_table: Vec<(String, f64, f64, f64, f64)> = var_names
                        .iter()
                        .enumerate()
                        .map(|(i, name)| {
                            (
                                name.clone(),
                                coeffs[i],
                                ses[i],
                                t_vals[i],
                                p_vals[i],
                            )
                        })
                        .collect();

                    let output = format_regression_results(
                        &format!("Clustered OLS (cluster: {})", cluster),
                        &coef_table,
                        result.ols.r_squared(),
                        result.ols.adj_r_squared(),
                        result.ols.n_obs(),
                        format,
                    );
                    println!("{}", output);
                }
                Err(e) => {
                    print_error(&format!("Clustered regression failed: {}", e), format);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

fn execute_diagnostics(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error(
                "No session active. Use --session <file> to enable dataset storage.",
                format,
            );
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            let x_cols: Vec<&str> = indep_vars.iter().map(|s| s.as_str()).collect();

            // run_diagnostics takes 3 args (no intercept param)
            match run_diagnostics(ds, dep_var, &x_cols) {
                Ok(diag) => {
                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "n_obs": diag.n_obs,
                                "n_params": diag.n_params,
                                "jarque_bera": diag.jarque_bera.as_ref().map(|jb| {
                                    serde_json::json!({
                                        "statistic": jb.statistic,
                                        "p_value": jb.p_value,
                                    })
                                }),
                                "breusch_pagan": diag.breusch_pagan.as_ref().map(|bp| {
                                    serde_json::json!({
                                        "statistic": bp.statistic,
                                        "p_value": bp.p_value,
                                    })
                                }),
                                "durbin_watson": diag.durbin_watson.as_ref().map(|dw| dw.statistic),
                                "vif": diag.vif.as_ref().map(|vifs| {
                                    vifs.iter().map(|v| serde_json::json!({
                                        "variable": v.variable,
                                        "vif": v.vif,
                                    })).collect::<Vec<_>>()
                                }),
                                "condition_number": diag.condition_number,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nRegression Diagnostics");
                            println!("{}", "=".repeat(50));
                            println!("Observations: {}, Parameters: {}", diag.n_obs, diag.n_params);

                            if let Some(ref jb) = diag.jarque_bera {
                                println!("\nJarque-Bera Test (Normality):");
                                println!("  Statistic: {:.4}", jb.statistic);
                                println!("  P-value: {:.4}", jb.p_value);
                            }

                            if let Some(ref bp) = diag.breusch_pagan {
                                println!("\nBreusch-Pagan Test (Heteroskedasticity):");
                                println!("  Statistic: {:.4}", bp.statistic);
                                println!("  P-value: {:.4}", bp.p_value);
                            }

                            if let Some(ref dw) = diag.durbin_watson {
                                println!("\nDurbin-Watson Statistic: {:.4}", dw.statistic);
                                println!("  {}", dw.interpretation);
                            }

                            if let Some(ref vifs) = diag.vif {
                                println!("\nVariance Inflation Factors:");
                                for vif_result in vifs {
                                    println!("  {}: {:.4}", vif_result.variable, vif_result.vif);
                                }
                            }

                            if let Some(cond) = diag.condition_number {
                                println!("\nCondition Number: {:.4}", cond);
                            }
                        }
                    }
                }
                Err(e) => {
                    print_error(&format!("Diagnostics failed: {}", e), format);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

fn execute_quantreg(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    tau: f64,
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
            let x_cols: Vec<&str> = indep_vars.iter().map(|s| s.as_str()).collect();

            match run_quantreg(ds, dep_var, &x_cols, tau) {
                Ok(result) => {
                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "method": "Quantile Regression",
                                "tau": result.tau,
                                "coefficients": result.coefficients.iter().map(|c| {
                                    serde_json::json!({
                                        "name": c.name,
                                        "estimate": c.estimate,
                                        "std_error": c.std_error,
                                        "t_value": c.t_value,
                                        "p_value": c.p_value,
                                        "ci_lower_95": c.ci_lower_95,
                                        "ci_upper_95": c.ci_upper_95,
                                    })
                                }).collect::<Vec<_>>(),
                                "n_obs": result.n_obs,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nQuantile Regression (tau = {:.2})", result.tau);
                            println!("{}", "=".repeat(60));

                            println!("\nCoefficients:");
                            println!("{:<15} {:>12} {:>12} {:>10} {:>10}",
                                "Variable", "Coef", "Std Err", "t", "P>|t|");
                            println!("{}", "-".repeat(60));

                            for c in &result.coefficients {
                                let sig = if c.p_value < 0.001 { "***" }
                                         else if c.p_value < 0.01 { "**" }
                                         else if c.p_value < 0.05 { "*" }
                                         else { "" };
                                println!("{:<15} {:>12.6} {:>12.6} {:>10.4} {:>9.4} {}",
                                    c.name, c.estimate, c.std_error,
                                    c.t_value, c.p_value, sig);
                            }

                            println!("\n---");
                            println!("Observations: {}", result.n_obs);
                        }
                    }
                }
                Err(e) => print_error(&format!("Quantile regression failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_loess(
    dataset_name: &str,
    dep_var: &str,
    indep_var: &str,
    span: f64,
    degree: usize,
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
            match run_loess(ds, dep_var, indep_var, span, degree, false) {
                Ok(result) => {
                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "method": "LOESS",
                                "span": result.span,
                                "degree": result.degree,
                                "n_obs": result.n_obs,
                                "enp": result.enp,
                                "rss": result.rss,
                                "fitted_values_sample": result.fitted.iter().take(10).collect::<Vec<_>>(),
                                "residuals_sample": result.residuals.iter().take(10).collect::<Vec<_>>(),
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nLOESS (Local Polynomial Regression)");
                            println!("{}", "=".repeat(50));
                            println!("\nSpan: {:.4}", result.span);
                            println!("Degree: {}", result.degree);
                            println!("Observations: {}", result.n_obs);
                            println!("Effective params: {:.2}", result.enp);

                            println!("\nFitted values (first 10):");
                            for (i, fitted) in result.fitted.iter().take(10).enumerate() {
                                println!("  {}: {:.6}", i + 1, fitted);
                            }
                        }
                    }
                }
                Err(e) => print_error(&format!("LOESS failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}
