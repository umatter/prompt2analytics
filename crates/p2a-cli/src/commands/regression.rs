//! Regression analysis commands

use clap::{Subcommand, ValueEnum};
use p2a_core::regression::CovarianceType;
use p2a_core::{LinearEstimator, run_diagnostics, run_ols, run_ols_clustered};
use p2a_core::{run_loess, run_quantreg};
// Additional regression methods
use ndarray::Array1;
use p2a_core::traits::t_test_p_value;
use p2a_core::{
    model_asymptotic, model_exponential_decay, model_exponential_growth, model_logistic_growth,
    model_michaelis_menten, model_power, run_nls,
};
use p2a_core::{run_gls, run_line, run_smooth_spline, run_step, run_supsmu, run_vcov_hac};

use crate::output::{OutputFormat, format_regression_results, print_error};
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
    #[command(after_help = "\
EXAMPLES:
    # Simple regression
    p2a reg ols mydata -y price -x sqft

    # Multiple regression with HC1 robust SEs
    p2a reg ols mydata -y price -x sqft bedrooms bathrooms --robust hc1

    # Without intercept
    p2a reg ols mydata -y price -x sqft --intercept false
")]
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
    #[command(after_help = "\
EXAMPLES:
    # Cluster by firm
    p2a reg clustered mydata -y revenue -x employees --cluster firm_id

    # Two-way clustering (firm and year)
    p2a reg clustered mydata -y revenue -x employees --cluster firm_id --cluster2 year
")]
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
    #[command(after_help = "\
EXAMPLES:
    # Full diagnostics (Jarque-Bera, Breusch-Pagan, Durbin-Watson, VIF)
    p2a reg diagnostics mydata -y price -x sqft bedrooms bathrooms
")]
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
    #[command(after_help = "\
EXAMPLES:
    # Median regression (tau = 0.5)
    p2a reg quantreg mydata -y price -x sqft bedrooms

    # 90th percentile regression
    p2a reg quantreg mydata -y price -x sqft bedrooms -t 0.9
")]
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
    #[command(after_help = "\
EXAMPLES:
    # Default LOESS (span=0.75, degree=2)
    p2a reg loess mydata -y price -x sqft

    # More local fit with lower span
    p2a reg loess mydata -y price -x sqft --span 0.3 --degree 1
")]
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

    /// Nonlinear Least Squares regression
    #[command(after_help = "\
EXAMPLES:
    # Exponential decay: y = a * exp(-b * x)
    p2a reg nls mydata -y response -x time --model exponential_decay --start 10,0.5

    # Michaelis-Menten: y = Vmax * x / (Km + x)
    p2a reg nls mydata -y rate -x substrate --model michaelis_menten --start 100,5

    # Logistic growth: y = K / (1 + exp(-r * (x - x0)))
    p2a reg nls mydata -y population -x time --model logistic_growth --start 1000,0.5,10
")]
    Nls {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable column (single predictor)
        #[arg(short = 'x', long)]
        indep_var: String,

        /// Model type: exponential_decay, exponential_growth, michaelis_menten, logistic_growth, power, asymptotic
        #[arg(long, default_value = "exponential_decay")]
        model: String,

        /// Starting values for parameters (comma-separated)
        #[arg(long, default_value = "1.0,0.1")]
        start: String,
    },

    /// Generalized Least Squares regression
    #[command(after_help = "\
EXAMPLES:
    # Auto-detect AR(1) correlation
    p2a reg gls mydata -y gdp -x investment exports --correlation ar1_auto

    # Fixed AR(1) with rho=0.7
    p2a reg gls mydata -y gdp -x investment --correlation ar1 --rho 0.7
")]
    Gls {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,

        /// Correlation structure: ar1, ar1_auto, compound_symmetry, identity
        #[arg(long, default_value = "ar1_auto")]
        correlation: String,

        /// Correlation parameter (rho for ar1/cs, ignored for ar1_auto)
        #[arg(long)]
        rho: Option<f64>,

        /// Include intercept
        #[arg(long, default_value = "true")]
        intercept: bool,
    },

    /// Stepwise regression model selection
    #[command(after_help = "\
EXAMPLES:
    # Bidirectional stepwise with AIC
    p2a reg step mydata -y price -x sqft bedrooms bathrooms age pool

    # Forward selection with BIC
    p2a reg step mydata -y price -x sqft bedrooms bathrooms --direction forward --use-bic

    # Backward elimination keeping sqft always
    p2a reg step mydata -y price -x sqft bedrooms bathrooms --direction backward --required-vars sqft
")]
    Step {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Candidate predictor variables (upper scope)
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,

        /// Variables always included (lower scope)
        #[arg(long, num_args = 0..)]
        required_vars: Vec<String>,

        /// Direction: forward, backward, or both (default: both)
        #[arg(long, default_value = "both")]
        direction: String,

        /// Use BIC instead of AIC
        #[arg(long, default_value = "false")]
        use_bic: bool,

        /// Include intercept
        #[arg(long, default_value = "true")]
        intercept: bool,
    },

    /// Smoothing spline regression
    #[command(after_help = "\
EXAMPLES:
    # Auto-select smoothing via cross-validation
    p2a reg smooth-spline mydata -y response -x time

    # Fixed degrees of freedom
    p2a reg smooth-spline mydata -y response -x time --df 5
")]
    SmoothSpline {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable column (single predictor)
        #[arg(short = 'x', long)]
        indep_var: String,

        /// Equivalent degrees of freedom (if not set, uses cross-validation)
        #[arg(long)]
        df: Option<f64>,
    },

    /// Friedman's SuperSmoother
    #[command(after_help = "\
EXAMPLES:
    # Auto-select span via cross-validation
    p2a reg supsmu mydata -y response -x time

    # Fixed span with bass smoothing
    p2a reg supsmu mydata -y response -x time --span 0.05 --bass 5
")]
    Supsmu {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable column (single predictor)
        #[arg(short = 'x', long)]
        indep_var: String,

        /// Fixed span (0-1), if not set uses cross-validation
        #[arg(long)]
        span: Option<f64>,

        /// Bass parameter (0-10, higher = smoother)
        #[arg(long, default_value = "0")]
        bass: f64,
    },

    /// Tukey's resistant line (robust regression)
    #[command(after_help = "\
EXAMPLES:
    # Basic resistant line
    p2a reg line mydata -y price -x sqft

    # With additional polishing iterations
    p2a reg line mydata -y price -x sqft --iter 3
")]
    Line {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable column (single predictor)
        #[arg(short = 'x', long)]
        indep_var: String,

        /// Number of polishing iterations
        #[arg(long, default_value = "1")]
        iter: usize,
    },

    /// HAC (Newey-West) standard errors for OLS
    #[command(after_help = "\
EXAMPLES:
    # Auto-select bandwidth
    p2a reg hac mydata -y returns -x market_return

    # Fixed lag length
    p2a reg hac mydata -y returns -x market_return smb hml --lags 4
")]
    Hac {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,

        /// Number of lags (if not set, uses automatic bandwidth)
        #[arg(long)]
        lags: Option<usize>,

        /// Include intercept
        #[arg(long, default_value = "true")]
        intercept: bool,
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
        } => execute_ols(
            dataset, dep_var, indep_vars, *intercept, *robust, format, session,
        ),
        RegressionCommands::Clustered {
            dataset,
            dep_var,
            indep_vars,
            cluster,
            intercept,
        } => execute_clustered(
            dataset, dep_var, indep_vars, cluster, *intercept, format, session,
        ),
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
        RegressionCommands::Nls {
            dataset,
            dep_var,
            indep_var,
            model,
            start,
        } => execute_nls(dataset, dep_var, indep_var, model, start, format, session),
        RegressionCommands::Gls {
            dataset,
            dep_var,
            indep_vars,
            correlation,
            rho,
            intercept,
        } => execute_gls(
            dataset,
            dep_var,
            indep_vars,
            correlation,
            *rho,
            *intercept,
            format,
            session,
        ),
        RegressionCommands::Step {
            dataset,
            dep_var,
            indep_vars,
            required_vars,
            direction,
            use_bic,
            intercept,
        } => execute_step(
            dataset,
            dep_var,
            indep_vars,
            required_vars,
            direction,
            *use_bic,
            *intercept,
            format,
            session,
        ),
        RegressionCommands::SmoothSpline {
            dataset,
            dep_var,
            indep_var,
            df,
        } => execute_smooth_spline(dataset, dep_var, indep_var, *df, format, session),
        RegressionCommands::Supsmu {
            dataset,
            dep_var,
            indep_var,
            span,
            bass,
        } => execute_supsmu(dataset, dep_var, indep_var, *span, *bass, format, session),
        RegressionCommands::Line {
            dataset,
            dep_var,
            indep_var,
            iter,
        } => execute_line(dataset, dep_var, indep_var, *iter, format, session),
        RegressionCommands::Hac {
            dataset,
            dep_var,
            indep_vars,
            lags,
            intercept,
        } => execute_hac(
            dataset, dep_var, indep_vars, *lags, *intercept, format, session,
        ),
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
                    let t_vals: Vec<f64> = coeffs
                        .iter()
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
                        .map(|(i, name)| (name.clone(), coeffs[i], ses[i], t_vals[i], p_vals[i]))
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
    _intercept: bool,
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
                    let t_vals: Vec<f64> = coeffs
                        .iter()
                        .zip(ses.iter())
                        .map(|(c, s)| if *s > 0.0 { c / s } else { 0.0 })
                        .collect();

                    // Intercept is always included by run_ols_clustered
                    let mut var_names = vec!["(Intercept)".to_string()];
                    var_names.extend(indep_vars.iter().cloned());

                    let coef_table: Vec<(String, f64, f64, f64, f64)> = var_names
                        .iter()
                        .enumerate()
                        .map(|(i, name)| (name.clone(), coeffs[i], ses[i], t_vals[i], p_vals[i]))
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
                Ok(diag) => match format {
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
                        println!(
                            "Observations: {}, Parameters: {}",
                            diag.n_obs, diag.n_params
                        );

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
                },
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
                Ok(result) => match format {
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
                        println!(
                            "{:<15} {:>12} {:>12} {:>10} {:>10}",
                            "Variable", "Coef", "Std Err", "t", "P>|t|"
                        );
                        println!("{}", "-".repeat(60));

                        for c in &result.coefficients {
                            let sig = if c.p_value < 0.001 {
                                "***"
                            } else if c.p_value < 0.01 {
                                "**"
                            } else if c.p_value < 0.05 {
                                "*"
                            } else {
                                ""
                            };
                            println!(
                                "{:<15} {:>12.6} {:>12.6} {:>10.4} {:>9.4} {}",
                                c.name, c.estimate, c.std_error, c.t_value, c.p_value, sig
                            );
                        }

                        println!("\n---");
                        println!("Observations: {}", result.n_obs);
                    }
                },
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
        Some(ds) => match run_loess(ds, dep_var, indep_var, span, degree, false) {
            Ok(result) => match format {
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
            },
            Err(e) => print_error(&format!("LOESS failed: {}", e), format),
        },
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_nls(
    dataset_name: &str,
    dep_var: &str,
    indep_var: &str,
    model_name: &str,
    start_str: &str,
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
            // Parse starting values
            let start_vals: Vec<f64> = start_str
                .split(',')
                .filter_map(|s| s.trim().parse::<f64>().ok())
                .collect();

            if start_vals.is_empty() {
                print_error(
                    "Invalid starting values. Use comma-separated numbers.",
                    format,
                );
                return Ok(());
            }

            let start = Array1::from_vec(start_vals.clone());

            // Select model function and parameter names
            let (model_fn, param_names): (fn(&Array1<f64>, &Array1<f64>) -> f64, Vec<&str>) =
                match model_name.to_lowercase().as_str() {
                    "exponential_decay" | "exp_decay" => {
                        if start_vals.len() != 2 {
                            print_error(
                                "exponential_decay requires 2 starting values: a, b (y = a * exp(-b * x))",
                                format,
                            );
                            return Ok(());
                        }
                        (model_exponential_decay, vec!["a", "b"])
                    }
                    "exponential_growth" | "exp_growth" => {
                        if start_vals.len() != 2 {
                            print_error(
                                "exponential_growth requires 2 starting values: a, b (y = a * exp(b * x))",
                                format,
                            );
                            return Ok(());
                        }
                        (model_exponential_growth, vec!["a", "b"])
                    }
                    "michaelis_menten" | "mm" => {
                        if start_vals.len() != 2 {
                            print_error(
                                "michaelis_menten requires 2 starting values: Vmax, Km (y = Vmax * x / (Km + x))",
                                format,
                            );
                            return Ok(());
                        }
                        (model_michaelis_menten, vec!["Vmax", "Km"])
                    }
                    "logistic_growth" | "logistic" => {
                        if start_vals.len() != 3 {
                            print_error(
                                "logistic_growth requires 3 starting values: K, r, x0 (y = K / (1 + exp(-r * (x - x0))))",
                                format,
                            );
                            return Ok(());
                        }
                        (model_logistic_growth, vec!["K", "r", "x0"])
                    }
                    "power" => {
                        if start_vals.len() != 2 {
                            print_error(
                                "power requires 2 starting values: a, b (y = a * x^b)",
                                format,
                            );
                            return Ok(());
                        }
                        (model_power, vec!["a", "b"])
                    }
                    "asymptotic" => {
                        if start_vals.len() != 3 {
                            print_error(
                                "asymptotic requires 3 starting values: a, b, c (y = a - b * exp(-c * x))",
                                format,
                            );
                            return Ok(());
                        }
                        (model_asymptotic, vec!["a", "b", "c"])
                    }
                    _ => {
                        print_error(
                            &format!(
                                "Unknown model: {}. Options: exponential_decay, exponential_growth, michaelis_menten, logistic_growth, power, asymptotic",
                                model_name
                            ),
                            format,
                        );
                        return Ok(());
                    }
                };

            match run_nls(ds, dep_var, indep_var, model_fn, &start, &param_names) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": "Nonlinear Least Squares",
                            "model": model_name,
                            "converged": result.converged,
                            "iterations": result.iterations,
                            "parameters": result.param_names.iter().zip(result.coefficients.iter())
                                .map(|(name, val)| serde_json::json!({
                                    "name": name,
                                    "estimate": val,
                                })).collect::<Vec<_>>(),
                            "std_errors": result.std_errors,
                            "rss": result.rss,
                            "n_obs": result.n_obs,
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nNonlinear Least Squares");
                        println!("{}", "=".repeat(50));
                        println!("Model: {}", model_name);
                        println!("Converged: {}", result.converged);
                        println!("Iterations: {}", result.iterations);

                        println!("\nParameters:");
                        println!("{:<12} {:>15} {:>15}", "Parameter", "Estimate", "Std Error");
                        println!("{}", "-".repeat(45));

                        for (i, name) in result.param_names.iter().enumerate() {
                            let se = if i < result.std_errors.len() {
                                format!("{:.6}", result.std_errors[i])
                            } else {
                                "N/A".to_string()
                            };
                            println!("{:<12} {:>15.6} {:>15}", name, result.coefficients[i], se);
                        }

                        println!("\n---");
                        println!("RSS: {:.6}", result.rss);
                        println!("Observations: {}", result.n_obs);
                    }
                },
                Err(e) => print_error(&format!("NLS failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_gls(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    correlation: &str,
    rho: Option<f64>,
    intercept: bool,
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
            // Extract y column
            let df = ds.df();
            let y_series = match df.column(dep_var) {
                Ok(s) => s,
                Err(_) => {
                    print_error(&format!("Column '{}' not found", dep_var), format);
                    return Ok(());
                }
            };
            let y: Vec<f64> = y_series
                .f64()
                .map(|ca| ca.into_no_null_iter().collect())
                .unwrap_or_default();

            // Build design matrix with intercept if requested
            let n = y.len();
            let mut x_data: Vec<f64> = Vec::new();
            let n_cols = if intercept {
                indep_vars.len() + 1
            } else {
                indep_vars.len()
            };

            for i in 0..n {
                if intercept {
                    x_data.push(1.0);
                }
                for var in indep_vars {
                    if let Ok(col) = df.column(var) {
                        if let Ok(ca) = col.f64() {
                            if let Some(v) = ca.get(i) {
                                x_data.push(v);
                            } else {
                                x_data.push(0.0);
                            }
                        }
                    }
                }
            }

            match run_gls(&y, &x_data, n_cols, correlation, rho) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": "GLS Regression",
                            "correlation": result.correlation,
                            "correlation_param": result.correlation_param,
                            "coefficients": result.coefficients,
                            "std_errors": result.std_errors,
                            "t_values": result.t_values,
                            "p_values": result.p_values,
                            "r_squared": result.r_squared,
                            "adj_r_squared": result.adj_r_squared,
                            "sigma": result.sigma,
                            "aic": result.aic,
                            "bic": result.bic,
                            "n_obs": result.n_obs,
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nGeneralized Least Squares Regression");
                        println!("{}", "=".repeat(60));
                        println!("Correlation: {}", result.correlation);
                        if let Some(rho_val) = result.correlation_param {
                            println!("Rho: {:.4}", rho_val);
                        }
                        println!("\nCoefficients:");
                        println!(
                            "{:<15} {:>12} {:>12} {:>10} {:>10}",
                            "Variable", "Coef", "Std Err", "t", "P>|t|"
                        );
                        println!("{}", "-".repeat(60));

                        let mut var_names = Vec::new();
                        if intercept {
                            var_names.push("(Intercept)".to_string());
                        }
                        var_names.extend(indep_vars.iter().cloned());

                        for (i, name) in var_names.iter().enumerate() {
                            if i < result.coefficients.len() {
                                let sig = if result.p_values[i] < 0.001 {
                                    "***"
                                } else if result.p_values[i] < 0.01 {
                                    "**"
                                } else if result.p_values[i] < 0.05 {
                                    "*"
                                } else {
                                    ""
                                };
                                println!(
                                    "{:<15} {:>12.6} {:>12.6} {:>10.4} {:>9.4} {}",
                                    name,
                                    result.coefficients[i],
                                    result.std_errors[i],
                                    result.t_values[i],
                                    result.p_values[i],
                                    sig
                                );
                            }
                        }

                        println!("\n---");
                        println!("R-squared: {:.4}", result.r_squared);
                        println!("Adj R-squared: {:.4}", result.adj_r_squared);
                        println!("Sigma: {:.4}", result.sigma);
                        println!("AIC: {:.4}, BIC: {:.4}", result.aic, result.bic);
                        println!("Observations: {}", result.n_obs);
                    }
                },
                Err(e) => print_error(&format!("GLS failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_step(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    required_vars: &[String],
    direction: &str,
    use_bic: bool,
    intercept: bool,
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
            let scope_lower: Vec<&str> = required_vars.iter().map(|s| s.as_str()).collect();
            let scope_upper: Vec<&str> = indep_vars.iter().map(|s| s.as_str()).collect();

            match run_step(
                ds,
                dep_var,
                &scope_lower,
                &scope_upper,
                direction,
                use_bic,
                intercept,
            ) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": "Stepwise Regression",
                            "direction": format!("{}", result.direction),
                            "criterion": result.criterion_name,
                            "initial_variables": result.initial_variables,
                            "final_variables": result.final_variables,
                            "n_steps": result.n_steps,
                            "final_aic": result.final_model.aic,
                            "final_r_squared": result.final_model.r_squared,
                            "steps": result.steps.iter().map(|s| {
                                serde_json::json!({
                                    "step": s.step,
                                    "action": s.action,
                                    "df": s.df,
                                    "rss": s.rss,
                                    "criterion": s.criterion,
                                })
                            }).collect::<Vec<_>>(),
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nStepwise Model Selection");
                        println!("{}", "=".repeat(60));
                        println!("Direction: {}", result.direction);
                        println!("Criterion: {} (k = {:.2})", result.criterion_name, result.k);

                        println!("\nStep History:");
                        println!(
                            "{:>5} {:>20} {:>6} {:>12} {:>12}",
                            "Step", "Action", "Df", "RSS", &result.criterion_name
                        );
                        println!("{}", "-".repeat(60));

                        for step in &result.steps {
                            let action = step.action.as_deref().unwrap_or("<initial>");
                            println!(
                                "{:>5} {:>20} {:>6} {:>12.4} {:>12.4}",
                                step.step, action, step.df, step.rss, step.criterion
                            );
                        }

                        println!("\nInitial model: {}", result.initial_variables.join(" + "));
                        println!("Final model: {}", result.final_variables.join(" + "));
                        println!("Total steps: {}", result.n_steps);
                        println!(
                            "\nFinal model R-squared: {:.4}",
                            result.final_model.r_squared
                        );
                    }
                },
                Err(e) => print_error(&format!("Stepwise selection failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_smooth_spline(
    dataset_name: &str,
    dep_var: &str,
    indep_var: &str,
    df: Option<f64>,
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
        Some(ds) => match run_smooth_spline(ds, indep_var, dep_var, df) {
            Ok(result) => match format {
                OutputFormat::Json => {
                    let json = serde_json::json!({
                        "method": "Smoothing Spline",
                        "df": result.df,
                        "spar": result.spar,
                        "lambda": result.lambda,
                        "cv_crit": result.cv_crit,
                        "n_obs": result.n_obs,
                        "n_knots": result.n_knots,
                        "fitted_sample": result.y.iter().take(10).collect::<Vec<_>>(),
                    });
                    println!("{}", serde_json::to_string_pretty(&json)?);
                }
                _ => {
                    println!("\nSmoothing Spline");
                    println!("{}", "=".repeat(50));
                    println!("Equivalent df: {:.4}", result.df);
                    println!("Smoothing param (spar): {:.4}", result.spar);
                    println!("Lambda: {:.6e}", result.lambda);
                    if let Some(cv) = result.cv_crit {
                        println!("CV criterion: {:.6}", cv);
                    }
                    println!("Observations: {}", result.n_obs);
                    println!("Knots: {}", result.n_knots);

                    println!("\nFitted values (first 10):");
                    for (i, (x, y)) in result.x.iter().zip(result.y.iter()).take(10).enumerate() {
                        println!("  {}: x={:.4}, y={:.6}", i + 1, x, y);
                    }
                }
            },
            Err(e) => print_error(&format!("Smoothing spline failed: {}", e), format),
        },
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_supsmu(
    dataset_name: &str,
    dep_var: &str,
    indep_var: &str,
    span: Option<f64>,
    bass: f64,
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
            // Extract x and y columns
            let df = ds.df();
            let x: Vec<f64> = match df.column(indep_var) {
                Ok(s) => s
                    .f64()
                    .map(|ca| ca.into_no_null_iter().collect())
                    .unwrap_or_default(),
                Err(_) => {
                    print_error(&format!("Column '{}' not found", indep_var), format);
                    return Ok(());
                }
            };
            let y: Vec<f64> = match df.column(dep_var) {
                Ok(s) => s
                    .f64()
                    .map(|ca| ca.into_no_null_iter().collect())
                    .unwrap_or_default(),
                Err(_) => {
                    print_error(&format!("Column '{}' not found", dep_var), format);
                    return Ok(());
                }
            };

            match run_supsmu(&x, &y, None, span, false, bass) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": "SuperSmoother",
                            "bass": result.bass,
                            "periodic": result.periodic,
                            "n": result.n,
                            "fitted_sample": result.y.iter().take(10).collect::<Vec<_>>(),
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nFriedman's SuperSmoother");
                        println!("{}", "=".repeat(50));
                        println!("Bass parameter: {:.1}", result.bass);
                        println!("Observations: {}", result.n);

                        println!("\nSmoothed values (first 10):");
                        for (i, (x_val, y_val)) in
                            result.x.iter().zip(result.y.iter()).take(10).enumerate()
                        {
                            println!("  {}: x={:.4}, y={:.6}", i + 1, x_val, y_val);
                        }
                    }
                },
                Err(e) => print_error(&format!("SuperSmoother failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_line(
    dataset_name: &str,
    dep_var: &str,
    indep_var: &str,
    iter: usize,
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
            // Extract x and y columns
            let df = ds.df();
            let x: Vec<f64> = match df.column(indep_var) {
                Ok(s) => s
                    .f64()
                    .map(|ca| ca.into_no_null_iter().collect())
                    .unwrap_or_default(),
                Err(_) => {
                    print_error(&format!("Column '{}' not found", indep_var), format);
                    return Ok(());
                }
            };
            let y: Vec<f64> = match df.column(dep_var) {
                Ok(s) => s
                    .f64()
                    .map(|ca| ca.into_no_null_iter().collect())
                    .unwrap_or_default(),
                Err(_) => {
                    print_error(&format!("Column '{}' not found", dep_var), format);
                    return Ok(());
                }
            };

            match run_line(&x, &y, Some(iter)) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": "Tukey's Resistant Line",
                            "intercept": result.intercept,
                            "slope": result.slope,
                            "n": result.n,
                            "iterations": result.iter,
                            "residuals_sample": result.residuals.iter().take(10).collect::<Vec<_>>(),
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nTukey's Resistant Line");
                        println!("{}", "=".repeat(50));
                        println!("Intercept: {:.6}", result.intercept);
                        println!("Slope: {:.6}", result.slope);
                        println!("Observations: {}", result.n);
                        println!("Polishing iterations: {}", result.iter);

                        println!("\nResiduals (first 10):");
                        for (i, resid) in result.residuals.iter().take(10).enumerate() {
                            println!("  {}: {:.6}", i + 1, resid);
                        }
                    }
                },
                Err(e) => print_error(&format!("Resistant line failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_hac(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    lags: Option<usize>,
    intercept: bool,
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

            // Run HAC estimation directly using the Dataset-based API
            match run_vcov_hac(ds, dep_var, &x_cols, lags, Some("bartlett"), false) {
                Ok(hac_result) => {
                    // Also run OLS for R-squared
                    let ols_result = run_ols(ds, dep_var, &x_cols, true, CovarianceType::Standard);
                    let (r_squared, n_obs_display) = match &ols_result {
                        Ok(ols) => (ols.r_squared, ols.n_obs),
                        Err(_) => (f64::NAN, hac_result.n_obs),
                    };

                    // Compute t-values and p-values from coefficients and HAC standard errors
                    let coefficients: Vec<f64> = hac_result.coefficients.to_vec();
                    let std_errors: Vec<f64> = hac_result.std_errors.to_vec();
                    let t_values: Vec<f64> = coefficients
                        .iter()
                        .zip(std_errors.iter())
                        .map(|(c, s)| if *s > 0.0 { c / s } else { 0.0 })
                        .collect();

                    let df = (hac_result.n_obs - hac_result.n_params) as f64;
                    let p_values: Vec<f64> =
                        t_values.iter().map(|&t| t_test_p_value(t, df)).collect();

                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "method": "OLS with HAC Standard Errors",
                                "kernel": "Bartlett (Newey-West)",
                                "bandwidth": hac_result.bandwidth,
                                "coefficients": coefficients,
                                "std_errors": std_errors,
                                "t_values": t_values,
                                "p_values": p_values,
                                "n_obs": n_obs_display,
                                "r_squared": r_squared,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nOLS with HAC (Newey-West) Standard Errors");
                            println!("{}", "=".repeat(60));
                            println!("Kernel: Bartlett");
                            println!("Bandwidth: {}", hac_result.bandwidth);

                            println!("\nCoefficients:");
                            println!(
                                "{:<15} {:>12} {:>12} {:>10} {:>10}",
                                "Variable", "Coef", "HAC SE", "t", "P>|t|"
                            );
                            println!("{}", "-".repeat(60));

                            let mut var_names = Vec::new();
                            if intercept {
                                var_names.push("(Intercept)".to_string());
                            }
                            var_names.extend(indep_vars.iter().cloned());

                            for (i, name) in var_names.iter().enumerate() {
                                if i < coefficients.len() {
                                    let sig = if p_values[i] < 0.001 {
                                        "***"
                                    } else if p_values[i] < 0.01 {
                                        "**"
                                    } else if p_values[i] < 0.05 {
                                        "*"
                                    } else {
                                        ""
                                    };
                                    println!(
                                        "{:<15} {:>12.6} {:>12.6} {:>10.4} {:>9.4} {}",
                                        name,
                                        coefficients[i],
                                        std_errors[i],
                                        t_values[i],
                                        p_values[i],
                                        sig
                                    );
                                }
                            }

                            println!("\n---");
                            println!("R-squared: {:.4}", r_squared);
                            println!("Observations: {}", n_obs_display);
                        }
                    }
                }
                Err(e) => print_error(&format!("HAC estimation failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}
