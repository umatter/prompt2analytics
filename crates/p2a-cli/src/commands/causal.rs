//! Causal inference commands

use clap::Subcommand;
use p2a_core::{
    run_iv2sls, run_did,
    run_synthetic_control, SynthConfig, PredictorSpec, VOptimization,
    run_ipw_treatment, run_doubly_robust, run_mediation_analysis,
    run_rd, run_fuzzy_rd,
    IpwConfig, DoublyRobustConfig, MediationConfig, RdConfig,
    Estimand, DRMethod, KernelType, BandwidthMethod, VceType,
};

use crate::output::{format_regression_results, print_error, OutputFormat};
use crate::session::SessionManager;

#[derive(Subcommand)]
pub enum CausalCommands {
    /// Two-Stage Least Squares (Instrumental Variables)
    Iv {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Exogenous variable columns
        #[arg(long, num_args = 1..)]
        exog: Vec<String>,

        /// Endogenous variable columns
        #[arg(long, num_args = 1..)]
        endog: Vec<String>,

        /// Instrument columns
        #[arg(long, num_args = 1..)]
        instruments: Vec<String>,
    },

    /// Difference-in-Differences
    Did {
        /// Dataset name
        dataset: String,

        /// Outcome variable column
        #[arg(short = 'y', long)]
        outcome: String,

        /// Treatment indicator column
        #[arg(long)]
        treat: String,

        /// Post-treatment period indicator column
        #[arg(long)]
        post: String,

        /// Control variable columns (optional)
        #[arg(short = 'x', long, num_args = 0..)]
        controls: Vec<String>,
    },

    /// Synthetic Control Method (Abadie et al.)
    Synth {
        /// Dataset name
        dataset: String,

        /// Outcome variable column
        #[arg(short = 'y', long)]
        outcome: String,

        /// Unit identifier column (e.g., "state", "country")
        #[arg(long)]
        unit: String,

        /// Time period column (must be integer)
        #[arg(long)]
        time: String,

        /// Name/ID of the treated unit
        #[arg(long)]
        treated: String,

        /// Treatment time (first post-treatment period)
        #[arg(long)]
        treatment_time: i64,

        /// Predictor columns (will use pre-treatment mean for matching)
        #[arg(short = 'p', long, num_args = 1..)]
        predictors: Vec<String>,

        /// V optimization method: "datadriven" (default), "equal"
        #[arg(long, default_value = "datadriven")]
        v_method: String,

        /// Run placebo tests for inference
        #[arg(long)]
        placebos: bool,
    },

    /// Inverse Probability Weighting treatment effect
    Ipw {
        /// Dataset name
        dataset: String,

        /// Outcome variable column
        #[arg(short = 'y', long)]
        outcome: String,

        /// Treatment indicator column
        #[arg(short = 't', long)]
        treatment: String,

        /// Covariate columns
        #[arg(short = 'x', long, num_args = 1..)]
        covariates: Vec<String>,

        /// Estimand: "ate" (default) or "att"
        #[arg(long, default_value = "ate")]
        estimand: String,

        /// Trimming threshold for propensity scores (default: 0.05)
        #[arg(long, default_value = "0.05")]
        trim: f64,

        /// Number of bootstrap replications (default: 999)
        #[arg(long, default_value = "999")]
        bootstrap: usize,
    },

    /// Doubly robust treatment effect (AIPW)
    DoublyRobust {
        /// Dataset name
        dataset: String,

        /// Outcome variable column
        #[arg(short = 'y', long)]
        outcome: String,

        /// Treatment indicator column
        #[arg(short = 't', long)]
        treatment: String,

        /// Covariate columns
        #[arg(short = 'x', long, num_args = 1..)]
        covariates: Vec<String>,

        /// Method: "aipw" (default), "ipw", or "regression"
        #[arg(long, default_value = "aipw")]
        method: String,

        /// Estimand: "ate" (default) or "att"
        #[arg(long, default_value = "ate")]
        estimand: String,

        /// Trimming threshold (default: 0.05)
        #[arg(long, default_value = "0.05")]
        trim: f64,

        /// Number of bootstrap replications (default: 999)
        #[arg(long, default_value = "999")]
        bootstrap: usize,
    },

    /// Causal mediation analysis
    Mediation {
        /// Dataset name
        dataset: String,

        /// Outcome variable column
        #[arg(short = 'y', long)]
        outcome: String,

        /// Treatment indicator column
        #[arg(short = 't', long)]
        treatment: String,

        /// Mediator variable column
        #[arg(short = 'm', long)]
        mediator: String,

        /// Covariate columns
        #[arg(short = 'x', long, num_args = 0..)]
        covariates: Vec<String>,

        /// Number of bootstrap replications (default: 999)
        #[arg(long, default_value = "999")]
        bootstrap: usize,
    },

    /// Regression Discontinuity (sharp design)
    Rd {
        /// Dataset name
        dataset: String,

        /// Outcome variable column
        #[arg(short = 'y', long)]
        outcome: String,

        /// Running variable column
        #[arg(short = 'r', long)]
        running: String,

        /// Cutoff value
        #[arg(short = 'c', long)]
        cutoff: f64,

        /// Polynomial order (default: 1 = local linear)
        #[arg(short = 'p', long, default_value = "1")]
        poly_order: usize,

        /// Kernel: "triangular" (default), "uniform", "epanechnikov"
        #[arg(long, default_value = "triangular")]
        kernel: String,

        /// Bandwidth selection: "mserd" (default), "msetwo", "cerrd"
        #[arg(long, default_value = "mserd")]
        bwselect: String,
    },

    /// Fuzzy Regression Discontinuity
    FuzzyRd {
        /// Dataset name
        dataset: String,

        /// Outcome variable column
        #[arg(short = 'y', long)]
        outcome: String,

        /// Running variable column
        #[arg(short = 'r', long)]
        running: String,

        /// Treatment indicator column
        #[arg(short = 't', long)]
        treatment: String,

        /// Cutoff value
        #[arg(short = 'c', long)]
        cutoff: f64,

        /// Polynomial order (default: 1)
        #[arg(short = 'p', long, default_value = "1")]
        poly_order: usize,

        /// Kernel: "triangular" (default), "uniform", "epanechnikov"
        #[arg(long, default_value = "triangular")]
        kernel: String,
    },
}

pub fn execute(
    cmd: &CausalCommands,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    match cmd {
        CausalCommands::Iv {
            dataset,
            dep_var,
            exog,
            endog,
            instruments,
        } => execute_iv(dataset, dep_var, exog, endog, instruments, format, session),
        CausalCommands::Did {
            dataset,
            outcome,
            treat,
            post,
            controls,
        } => execute_did(dataset, outcome, treat, post, controls, format, session),
        CausalCommands::Synth {
            dataset,
            outcome,
            unit,
            time,
            treated,
            treatment_time,
            predictors,
            v_method,
            placebos,
        } => execute_synth(
            dataset, outcome, unit, time, treated, *treatment_time,
            predictors, v_method, *placebos, format, session
        ),
        CausalCommands::Ipw {
            dataset,
            outcome,
            treatment,
            covariates,
            estimand,
            trim,
            bootstrap,
        } => execute_ipw(dataset, outcome, treatment, covariates, estimand, *trim, *bootstrap, format, session),
        CausalCommands::DoublyRobust {
            dataset,
            outcome,
            treatment,
            covariates,
            method,
            estimand,
            trim,
            bootstrap,
        } => execute_dr(dataset, outcome, treatment, covariates, method, estimand, *trim, *bootstrap, format, session),
        CausalCommands::Mediation {
            dataset,
            outcome,
            treatment,
            mediator,
            covariates,
            bootstrap,
        } => execute_mediation(dataset, outcome, treatment, mediator, covariates, *bootstrap, format, session),
        CausalCommands::Rd {
            dataset,
            outcome,
            running,
            cutoff,
            poly_order,
            kernel,
            bwselect,
        } => execute_rd(dataset, outcome, running, *cutoff, *poly_order, kernel, bwselect, format, session),
        CausalCommands::FuzzyRd {
            dataset,
            outcome,
            running,
            treatment,
            cutoff,
            poly_order,
            kernel,
        } => execute_fuzzy_rd(dataset, outcome, running, treatment, *cutoff, *poly_order, kernel, format, session),
    }
}

fn execute_iv(
    dataset_name: &str,
    dep_var: &str,
    exog: &[String],
    endog: &[String],
    instruments: &[String],
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
            let exog_refs: Vec<&str> = exog.iter().map(|s| s.as_str()).collect();
            let endog_refs: Vec<&str> = endog.iter().map(|s| s.as_str()).collect();
            let inst_refs: Vec<&str> = instruments.iter().map(|s| s.as_str()).collect();

            match run_iv2sls(ds, dep_var, &exog_refs, &endog_refs, &inst_refs, true) {
                Ok(result) => {
                    // IVResult uses fields, not methods
                    let coeffs = &result.coefficients;
                    let ses = &result.std_errors;
                    let p_vals = &result.p_values;
                    let t_vals = &result.t_stats;

                    let var_names = &result.variables;

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
                        "2SLS / IV Regression",
                        &coef_table,
                        result.r_squared,
                        result.r_squared, // IVResult has no adj_r_squared
                        result.n_obs,
                        format,
                    );
                    println!("{}", output);

                    // Print first-stage diagnostics
                    match format {
                        OutputFormat::Json => {}
                        _ => {
                            if !result.first_stage_f_stats.is_empty() {
                                println!("\nFirst-stage diagnostics:");
                                for (i, f_stat) in result.first_stage_f_stats.iter().enumerate() {
                                    let endog_name = if i < endog.len() { &endog[i] } else { "Unknown" };
                                    println!("  {} F-statistic: {:.4}", endog_name, f_stat);
                                }
                                if result.strong_instruments {
                                    println!("  Instruments are strong (all F > 10)");
                                } else {
                                    println!("  Warning: Some instruments may be weak (F < 10)");
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    print_error(&format!("2SLS failed: {}", e), format);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

fn execute_did(
    dataset_name: &str,
    outcome: &str,
    treat: &str,
    post: &str,
    controls: &[String],
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
            let control_refs: Vec<&str> = controls.iter().map(|s| s.as_str()).collect();
            let controls_opt = if control_refs.is_empty() {
                None
            } else {
                Some(control_refs.as_slice())
            };

            match run_did(ds, outcome, treat, post, controls_opt) {
                Ok(result) => {
                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "method": "Difference-in-Differences",
                                "att": result.att,
                                "std_error": result.std_error,
                                "t_stat": result.t_stat,
                                "p_value": result.p_value,
                                "n_obs": result.n_obs,
                                "treated_pre_mean": result.treated_pre_mean,
                                "treated_post_mean": result.treated_post_mean,
                                "control_pre_mean": result.control_pre_mean,
                                "control_post_mean": result.control_post_mean,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nDifference-in-Differences Results");
                            println!("{}", "=".repeat(50));
                            println!("\nAverage Treatment Effect on Treated (ATT): {:.6}", result.att);
                            println!("Standard Error: {:.6}", result.std_error);
                            println!("t-statistic: {:.4}", result.t_stat);
                            println!("p-value: {:.4}", result.p_value);
                            println!("\nObservations: {}", result.n_obs);
                            println!("\nGroup means:");
                            println!("  Pre-treatment (treated): {:.4}", result.treated_pre_mean);
                            println!("  Post-treatment (treated): {:.4}", result.treated_post_mean);
                            println!("  Pre-treatment (control): {:.4}", result.control_pre_mean);
                            println!("  Post-treatment (control): {:.4}", result.control_post_mean);
                        }
                    }
                }
                Err(e) => {
                    print_error(&format!("DiD failed: {}", e), format);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

fn execute_synth(
    dataset_name: &str,
    outcome: &str,
    unit_col: &str,
    time_col: &str,
    treated_unit: &str,
    treatment_time: i64,
    predictors: &[String],
    v_method: &str,
    run_placebos: bool,
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
            // Build predictor specs (using pre-treatment mean for all)
            let predictor_specs: Vec<PredictorSpec> = predictors
                .iter()
                .map(|col| PredictorSpec::new(col))
                .collect();

            // Parse V optimization method
            let v_opt = match v_method.to_lowercase().as_str() {
                "equal" => VOptimization::Equal,
                _ => VOptimization::DataDriven,
            };

            let config = SynthConfig {
                treated_unit: treated_unit.to_string(),
                treatment_time,
                optimization_window: None,
                v_method: v_opt,
                tolerance: 1e-6,
                max_iter: 1000,
                run_placebos,
                weight_threshold: 0.001,
            };

            match run_synthetic_control(ds, outcome, unit_col, time_col, &predictor_specs, config) {
                Ok(result) => {
                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "method": "Synthetic Control",
                                "treated_unit": result.treated_unit,
                                "treatment_time": result.treatment_time,
                                "unit_weights": result.unit_weights,
                                "predictor_balance": result.predictor_balance.iter().map(|b| {
                                    serde_json::json!({
                                        "predictor": b.predictor,
                                        "treated_value": b.treated_value,
                                        "synthetic_value": b.synthetic_value,
                                        "percent_diff": b.percent_diff,
                                    })
                                }).collect::<Vec<_>>(),
                                "pre_treatment_mspe": result.pre_treatment_mspe,
                                "pre_treatment_rmspe": result.pre_treatment_rmspe,
                                "treatment_effects": result.treatment_effects.iter().map(|e| {
                                    serde_json::json!({
                                        "time": e.time,
                                        "actual": e.actual,
                                        "synthetic": e.synthetic,
                                        "effect": e.effect,
                                    })
                                }).collect::<Vec<_>>(),
                                "average_effect": result.average_effect,
                                "cumulative_effect": result.cumulative_effect,
                                "placebo_results": result.placebo_results.as_ref().map(|p| {
                                    serde_json::json!({
                                        "treated_rank": p.treated_rank,
                                        "p_value": p.p_value,
                                        "n_units": p.n_units,
                                    })
                                }),
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("{}", result);
                        }
                    }
                }
                Err(e) => {
                    print_error(&format!("Synthetic Control failed: {}", e), format);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

fn execute_ipw(
    dataset_name: &str,
    outcome: &str,
    treatment: &str,
    covariates: &[String],
    estimand: &str,
    trim: f64,
    bootstrap: usize,
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
            let cov_refs: Vec<&str> = covariates.iter().map(|s| s.as_str()).collect();
            let est = match estimand.to_lowercase().as_str() {
                "att" => Estimand::ATT,
                _ => Estimand::ATE,
            };
            let config = IpwConfig {
                estimand: est,
                trim,
                bootstrap,
                normalized: true,
                seed: None,
            };

            match run_ipw_treatment(ds, outcome, treatment, &cov_refs, config) {
                Ok(result) => {
                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "method": "IPW Treatment Effect",
                                "estimand": format!("{:?}", result.estimand),
                                "effect": result.effect,
                                "std_error": result.std_error,
                                "ci_lower": result.ci_lower,
                                "ci_upper": result.ci_upper,
                                "t_stat": result.t_stat,
                                "p_value": result.p_value,
                                "n_obs": result.n_obs,
                                "n_treated": result.n_treated,
                                "n_control": result.n_control,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nIPW Treatment Effect ({:?})", result.estimand);
                            println!("{}", "=".repeat(50));
                            println!("Effect: {:.6}", result.effect);
                            println!("Std Error: {:.6}", result.std_error);
                            println!("95% CI: [{:.6}, {:.6}]", result.ci_lower, result.ci_upper);
                            println!("t-stat: {:.4}, p-value: {:.4}", result.t_stat, result.p_value);
                            println!("N: {} (treated: {}, control: {})", result.n_obs, result.n_treated, result.n_control);
                        }
                    }
                }
                Err(e) => print_error(&format!("IPW failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_dr(
    dataset_name: &str,
    outcome: &str,
    treatment: &str,
    covariates: &[String],
    method: &str,
    estimand: &str,
    trim: f64,
    bootstrap: usize,
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
            let cov_refs: Vec<&str> = covariates.iter().map(|s| s.as_str()).collect();
            let est = match estimand.to_lowercase().as_str() {
                "att" => Estimand::ATT,
                _ => Estimand::ATE,
            };
            let dr_method = match method.to_lowercase().as_str() {
                "ipw" => DRMethod::IPW,
                "regression" => DRMethod::Regression,
                _ => DRMethod::AIPW,
            };
            let config = DoublyRobustConfig {
                method: dr_method,
                estimand: est,
                trim,
                bootstrap,
                seed: None,
            };

            match run_doubly_robust(ds, outcome, treatment, &cov_refs, config) {
                Ok(result) => {
                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "method": format!("{:?}", result.method),
                                "estimand": format!("{:?}", result.estimand),
                                "effect": result.effect,
                                "std_error": result.std_error,
                                "ci_lower": result.ci_lower,
                                "ci_upper": result.ci_upper,
                                "t_stat": result.t_stat,
                                "p_value": result.p_value,
                                "n_obs": result.n_obs,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nDoubly Robust Treatment Effect ({:?}, {:?})", result.method, result.estimand);
                            println!("{}", "=".repeat(50));
                            println!("Effect: {:.6}", result.effect);
                            println!("Std Error: {:.6}", result.std_error);
                            println!("95% CI: [{:.6}, {:.6}]", result.ci_lower, result.ci_upper);
                            println!("t-stat: {:.4}, p-value: {:.4}", result.t_stat, result.p_value);
                            println!("N: {}", result.n_obs);
                        }
                    }
                }
                Err(e) => print_error(&format!("Doubly robust failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_mediation(
    dataset_name: &str,
    outcome: &str,
    treatment: &str,
    mediator: &str,
    covariates: &[String],
    bootstrap: usize,
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
            let cov_refs: Vec<&str> = covariates.iter().map(|s| s.as_str()).collect();
            let config = MediationConfig {
                bootstrap,
                trim: 0.05,
                seed: None,
            };

            match run_mediation_analysis(ds, outcome, treatment, mediator, &cov_refs, config) {
                Ok(result) => {
                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "method": "Causal Mediation Analysis",
                                "total_effect": result.total_effect,
                                "direct_effect": result.direct_effect,
                                "indirect_effect": result.indirect_effect,
                                "proportion_mediated": result.proportion_mediated,
                                "se_total": result.se_total,
                                "se_direct": result.se_direct,
                                "se_indirect": result.se_indirect,
                                "p_total": result.p_total,
                                "p_direct": result.p_direct,
                                "p_indirect": result.p_indirect,
                                "n_obs": result.n_obs,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nCausal Mediation Analysis");
                            println!("{}", "=".repeat(50));
                            println!("Total Effect (ATE): {:.6} (SE: {:.6}, p: {:.4})",
                                result.total_effect, result.se_total, result.p_total);
                            println!("Direct Effect (NDE): {:.6} (SE: {:.6}, p: {:.4})",
                                result.direct_effect, result.se_direct, result.p_direct);
                            println!("Indirect Effect (NIE): {:.6} (SE: {:.6}, p: {:.4})",
                                result.indirect_effect, result.se_indirect, result.p_indirect);
                            println!("Proportion Mediated: {:.2}%", result.proportion_mediated * 100.0);
                            println!("N: {}", result.n_obs);
                        }
                    }
                }
                Err(e) => print_error(&format!("Mediation analysis failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_rd(
    dataset_name: &str,
    outcome: &str,
    running: &str,
    cutoff: f64,
    poly_order: usize,
    kernel: &str,
    bwselect: &str,
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
            let kern = match kernel.to_lowercase().as_str() {
                "uniform" => KernelType::Uniform,
                "epanechnikov" => KernelType::Epanechnikov,
                _ => KernelType::Triangular,
            };
            let bw = match bwselect.to_lowercase().as_str() {
                "msetwo" => BandwidthMethod::MseTwo,
                "cerrd" => BandwidthMethod::CerRd,
                _ => BandwidthMethod::MseRd,
            };
            let config = RdConfig {
                p: poly_order,
                q: None,
                h: None,
                b: None,
                rho: 1.0,
                kernel: kern,
                bwselect: bw,
                vce: VceType::Nn,
                nnmatch: 3,
                level: 0.95,
                scaleregul: 1.0,
            };

            match run_rd(ds, outcome, running, cutoff, config) {
                Ok(result) => {
                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "method": "Sharp RD",
                                "outcome": result.outcome,
                                "running_var": result.running_var,
                                "cutoff": result.cutoff,
                                "tau_robust": result.tau_robust,
                                "se_robust": result.se_robust,
                                "p_robust": result.p_robust,
                                "ci_robust": result.ci_robust,
                                "n_left": result.n_left,
                                "n_right": result.n_right,
                                "h_left": result.h_left,
                                "h_right": result.h_right,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nRegression Discontinuity Estimation");
                            println!("{}", "=".repeat(50));
                            println!("Outcome: {}, Running var: {}", result.outcome, result.running_var);
                            println!("Cutoff: {:.4}", result.cutoff);
                            println!("\nRobust RD Estimate: {:.6}", result.tau_robust);
                            println!("Std Error: {:.6}", result.se_robust);
                            println!("p-value: {:.4}", result.p_robust);
                            println!("95% CI: [{:.6}, {:.6}]", result.ci_robust.0, result.ci_robust.1);
                            println!("\nN left: {}, N right: {}", result.n_left, result.n_right);
                            println!("Bandwidth h: left={:.4}, right={:.4}", result.h_left, result.h_right);
                        }
                    }
                }
                Err(e) => print_error(&format!("RD estimation failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_fuzzy_rd(
    dataset_name: &str,
    outcome: &str,
    running: &str,
    treatment: &str,
    cutoff: f64,
    poly_order: usize,
    kernel: &str,
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
            let kern = match kernel.to_lowercase().as_str() {
                "uniform" => KernelType::Uniform,
                "epanechnikov" => KernelType::Epanechnikov,
                _ => KernelType::Triangular,
            };
            let config = RdConfig {
                p: poly_order,
                q: None,
                h: None,
                b: None,
                rho: 1.0,
                kernel: kern,
                bwselect: BandwidthMethod::MseRd,
                vce: VceType::Nn,
                nnmatch: 3,
                level: 0.95,
                scaleregul: 1.0,
            };

            match run_fuzzy_rd(ds, outcome, running, treatment, cutoff, config) {
                Ok(result) => {
                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "method": "Fuzzy RD",
                                "tau_fuzzy": result.tau_fuzzy,
                                "se_fuzzy": result.se_fuzzy,
                                "p_fuzzy": result.p_fuzzy,
                                "ci_fuzzy": result.ci_fuzzy,
                                "treatment": result.treatment,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nFuzzy Regression Discontinuity Estimation");
                            println!("{}", "=".repeat(50));
                            println!("Treatment: {}", result.treatment);
                            println!("\nFuzzy RD Estimate (LATE): {:.6}", result.tau_fuzzy);
                            println!("Std Error: {:.6}", result.se_fuzzy);
                            println!("p-value: {:.4}", result.p_fuzzy);
                            println!("95% CI: [{:.6}, {:.6}]", result.ci_fuzzy.0, result.ci_fuzzy.1);
                        }
                    }
                }
                Err(e) => print_error(&format!("Fuzzy RD estimation failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}
