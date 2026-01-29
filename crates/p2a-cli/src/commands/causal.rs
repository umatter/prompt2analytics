//! Causal inference commands

use clap::Subcommand;
use p2a_core::{
    AttEstimationMethod,
    BandwidthMethod,
    ComparisonGroup,
    DRMethod,
    DistanceMethod,
    DoublyRobustConfig,
    Estimand,
    GModel,
    GsynthConfig,
    GsynthEstimator,
    GsynthForce,
    IpwConfig,
    KernelType,
    MatchMethod,
    MediationConfig,
    PredictorSpec,
    QModel,
    RdConfig,
    SEMethod,
    StaggeredDidConfig,
    StdRegConfig,
    StdRegEstimand,
    StdRegModel,
    SynthConfig,
    TmleConfig,
    VOptimization,
    VceType,
    // Propensity Score Matching
    match_it,
    run_did,
    run_doubly_robust,
    run_fuzzy_rd,
    // Generalized Synthetic Control
    run_gsynth,
    run_ipw_treatment,
    run_iv2sls,
    run_mediation_analysis,
    run_rd,
    // Staggered DiD (Callaway-Sant'Anna)
    run_staggered_did,
    // Regression Standardization / G-computation
    run_stdreg,
    run_synthetic_control,
    // TMLE
    tmle,
};

use crate::output::{OutputFormat, format_regression_results, print_error};
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

    /// Staggered Difference-in-Differences (Callaway-Sant'Anna)
    StaggeredDid {
        /// Dataset name
        dataset: String,

        /// Outcome variable column
        #[arg(short = 'y', long)]
        outcome: String,

        /// Treatment timing column (period when unit was first treated; 0 or negative = never treated)
        #[arg(long)]
        treatment_time: String,

        /// Time period column
        #[arg(long)]
        time: String,

        /// Unit identifier column
        #[arg(long)]
        unit: String,

        /// Covariate columns for conditional parallel trends (optional)
        #[arg(short = 'x', long, num_args = 0..)]
        covariates: Vec<String>,

        /// Comparison group: "nevertreated" (default) or "notyettreated"
        #[arg(long, default_value = "nevertreated")]
        comparison: String,

        /// Estimation method: "or" (outcome regression), "ipw", or "dr" (doubly robust, default)
        #[arg(long, default_value = "or")]
        method: String,

        /// Number of bootstrap replications for SEs (default: 999)
        #[arg(long, default_value = "999")]
        bootstrap: usize,
    },

    /// Targeted Maximum Likelihood Estimation (TMLE)
    Tmle {
        /// Dataset name
        dataset: String,

        /// Outcome variable column
        #[arg(short = 'y', long)]
        outcome: String,

        /// Treatment indicator column (binary 0/1)
        #[arg(short = 't', long)]
        treatment: String,

        /// Covariate columns
        #[arg(short = 'x', long, num_args = 1..)]
        covariates: Vec<String>,

        /// Outcome model: "logistic" (default) or "linear"
        #[arg(long, default_value = "logistic")]
        qmodel: String,

        /// Propensity score truncation lower bound (default: 0.01)
        #[arg(long, default_value = "0.01")]
        truncate_lower: f64,

        /// Propensity score truncation upper bound (default: 0.99)
        #[arg(long, default_value = "0.99")]
        truncate_upper: f64,
    },

    /// Generalized Synthetic Control (gsynth)
    Gsynth {
        /// Dataset name
        dataset: String,

        /// Outcome variable column
        #[arg(short = 'y', long)]
        outcome: String,

        /// Treatment indicator column (0/1)
        #[arg(short = 'd', long)]
        treatment: String,

        /// Unit identifier column
        #[arg(long)]
        unit: String,

        /// Time period column
        #[arg(long)]
        time: String,

        /// Covariate columns (optional)
        #[arg(short = 'x', long, num_args = 0..)]
        covariates: Vec<String>,

        /// Number of latent factors (0 = auto-select via cross-validation)
        #[arg(long, default_value = "0")]
        n_factors: usize,

        /// Fixed effects: "unit", "time", "twoway" (default), or "none"
        #[arg(long, default_value = "twoway")]
        force: String,

        /// Estimator: "ife" (interactive FE, default) or "mc" (matrix completion)
        #[arg(long, default_value = "ife")]
        estimator: String,

        /// Compute bootstrap standard errors
        #[arg(long)]
        bootstrap_se: bool,

        /// Number of bootstrap replications (default: 200)
        #[arg(long, default_value = "200")]
        n_bootstrap: usize,
    },

    /// Propensity Score Matching (MatchIt)
    Matching {
        /// Dataset name
        dataset: String,

        /// Treatment indicator column (binary 0/1)
        #[arg(short = 't', long)]
        treatment: String,

        /// Covariate columns for matching
        #[arg(short = 'x', long, num_args = 1..)]
        covariates: Vec<String>,

        /// Matching method: "nn" (nearest neighbor, default), "cem" (coarsened exact), "full", or "subclass"
        #[arg(long, default_value = "nn")]
        method: String,

        /// Distance metric: "logit" (default), "probit", "mahalanobis", or "euclidean"
        #[arg(long, default_value = "logit")]
        distance: String,

        /// Matching ratio for nearest neighbor (default: 1)
        #[arg(long, default_value = "1")]
        ratio: usize,

        /// Caliper for nearest neighbor (in SD units of propensity score; optional)
        #[arg(long)]
        caliper: Option<f64>,

        /// Sample with replacement (for nearest neighbor)
        #[arg(long)]
        replace: bool,

        /// Number of subclasses (for subclass method, default: 5)
        #[arg(long, default_value = "5")]
        n_subclasses: usize,
    },

    /// Regression Standardization / G-computation (stdReg)
    Stdreg {
        /// Dataset name
        dataset: String,

        /// Outcome variable column
        #[arg(short = 'y', long)]
        outcome: String,

        /// Treatment indicator column (binary 0/1)
        #[arg(short = 't', long)]
        treatment: String,

        /// Covariate columns
        #[arg(short = 'x', long, num_args = 1..)]
        covariates: Vec<String>,

        /// Outcome model: "linear" (default), "logistic", or "poisson"
        #[arg(long, default_value = "linear")]
        model: String,

        /// Estimand: "ate" (default), "att", "atc", or "levels"
        #[arg(long, default_value = "ate")]
        estimand: String,

        /// SE method: "bootstrap" (default), "delta", or "sandwich"
        #[arg(long, default_value = "bootstrap")]
        se_method: String,

        /// Number of bootstrap replications (default: 999)
        #[arg(long, default_value = "999")]
        bootstrap: usize,

        /// Include treatment-covariate interactions in outcome model
        #[arg(long)]
        interactions: bool,
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
            dataset,
            outcome,
            unit,
            time,
            treated,
            *treatment_time,
            predictors,
            v_method,
            *placebos,
            format,
            session,
        ),
        CausalCommands::Ipw {
            dataset,
            outcome,
            treatment,
            covariates,
            estimand,
            trim,
            bootstrap,
        } => execute_ipw(
            dataset, outcome, treatment, covariates, estimand, *trim, *bootstrap, format, session,
        ),
        CausalCommands::DoublyRobust {
            dataset,
            outcome,
            treatment,
            covariates,
            method,
            estimand,
            trim,
            bootstrap,
        } => execute_dr(
            dataset, outcome, treatment, covariates, method, estimand, *trim, *bootstrap, format,
            session,
        ),
        CausalCommands::Mediation {
            dataset,
            outcome,
            treatment,
            mediator,
            covariates,
            bootstrap,
        } => execute_mediation(
            dataset, outcome, treatment, mediator, covariates, *bootstrap, format, session,
        ),
        CausalCommands::Rd {
            dataset,
            outcome,
            running,
            cutoff,
            poly_order,
            kernel,
            bwselect,
        } => execute_rd(
            dataset,
            outcome,
            running,
            *cutoff,
            *poly_order,
            kernel,
            bwselect,
            format,
            session,
        ),
        CausalCommands::FuzzyRd {
            dataset,
            outcome,
            running,
            treatment,
            cutoff,
            poly_order,
            kernel,
        } => execute_fuzzy_rd(
            dataset,
            outcome,
            running,
            treatment,
            *cutoff,
            *poly_order,
            kernel,
            format,
            session,
        ),
        CausalCommands::StaggeredDid {
            dataset,
            outcome,
            treatment_time,
            time,
            unit,
            covariates,
            comparison,
            method,
            bootstrap,
        } => execute_staggered_did(
            dataset,
            outcome,
            treatment_time,
            time,
            unit,
            covariates,
            comparison,
            method,
            *bootstrap,
            format,
            session,
        ),
        CausalCommands::Tmle {
            dataset,
            outcome,
            treatment,
            covariates,
            qmodel,
            truncate_lower,
            truncate_upper,
        } => execute_tmle(
            dataset,
            outcome,
            treatment,
            covariates,
            qmodel,
            *truncate_lower,
            *truncate_upper,
            format,
            session,
        ),
        CausalCommands::Gsynth {
            dataset,
            outcome,
            treatment,
            unit,
            time,
            covariates,
            n_factors,
            force,
            estimator,
            bootstrap_se,
            n_bootstrap,
        } => execute_gsynth(
            dataset,
            outcome,
            treatment,
            unit,
            time,
            covariates,
            *n_factors,
            force,
            estimator,
            *bootstrap_se,
            *n_bootstrap,
            format,
            session,
        ),
        CausalCommands::Matching {
            dataset,
            treatment,
            covariates,
            method,
            distance,
            ratio,
            caliper,
            replace,
            n_subclasses,
        } => execute_matching(
            dataset,
            treatment,
            covariates,
            method,
            distance,
            *ratio,
            caliper.as_ref().copied(),
            *replace,
            *n_subclasses,
            format,
            session,
        ),
        CausalCommands::Stdreg {
            dataset,
            outcome,
            treatment,
            covariates,
            model,
            estimand,
            se_method,
            bootstrap,
            interactions,
        } => execute_stdreg(
            dataset,
            outcome,
            treatment,
            covariates,
            model,
            estimand,
            se_method,
            *bootstrap,
            *interactions,
            format,
            session,
        ),
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
                        .map(|(i, name)| (name.clone(), coeffs[i], ses[i], t_vals[i], p_vals[i]))
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
                                    let endog_name = if i < endog.len() {
                                        &endog[i]
                                    } else {
                                        "Unknown"
                                    };
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
                Ok(result) => match format {
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
                        println!(
                            "\nAverage Treatment Effect on Treated (ATT): {:.6}",
                            result.att
                        );
                        println!("Standard Error: {:.6}", result.std_error);
                        println!("t-statistic: {:.4}", result.t_stat);
                        println!("p-value: {:.4}", result.p_value);
                        println!("\nObservations: {}", result.n_obs);
                        println!("\nGroup means:");
                        println!("  Pre-treatment (treated): {:.4}", result.treated_pre_mean);
                        println!(
                            "  Post-treatment (treated): {:.4}",
                            result.treated_post_mean
                        );
                        println!("  Pre-treatment (control): {:.4}", result.control_pre_mean);
                        println!(
                            "  Post-treatment (control): {:.4}",
                            result.control_post_mean
                        );
                    }
                },
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
                Ok(result) => match format {
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
                },
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
                Ok(result) => match format {
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
                        println!(
                            "t-stat: {:.4}, p-value: {:.4}",
                            result.t_stat, result.p_value
                        );
                        println!(
                            "N: {} (treated: {}, control: {})",
                            result.n_obs, result.n_treated, result.n_control
                        );
                    }
                },
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
                Ok(result) => match format {
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
                        println!(
                            "\nDoubly Robust Treatment Effect ({:?}, {:?})",
                            result.method, result.estimand
                        );
                        println!("{}", "=".repeat(50));
                        println!("Effect: {:.6}", result.effect);
                        println!("Std Error: {:.6}", result.std_error);
                        println!("95% CI: [{:.6}, {:.6}]", result.ci_lower, result.ci_upper);
                        println!(
                            "t-stat: {:.4}, p-value: {:.4}",
                            result.t_stat, result.p_value
                        );
                        println!("N: {}", result.n_obs);
                    }
                },
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
                Ok(result) => match format {
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
                        println!(
                            "Total Effect (ATE): {:.6} (SE: {:.6}, p: {:.4})",
                            result.total_effect, result.se_total, result.p_total
                        );
                        println!(
                            "Direct Effect (NDE): {:.6} (SE: {:.6}, p: {:.4})",
                            result.direct_effect, result.se_direct, result.p_direct
                        );
                        println!(
                            "Indirect Effect (NIE): {:.6} (SE: {:.6}, p: {:.4})",
                            result.indirect_effect, result.se_indirect, result.p_indirect
                        );
                        println!(
                            "Proportion Mediated: {:.2}%",
                            result.proportion_mediated * 100.0
                        );
                        println!("N: {}", result.n_obs);
                    }
                },
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
                Ok(result) => match format {
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
                        println!(
                            "Outcome: {}, Running var: {}",
                            result.outcome, result.running_var
                        );
                        println!("Cutoff: {:.4}", result.cutoff);
                        println!("\nRobust RD Estimate: {:.6}", result.tau_robust);
                        println!("Std Error: {:.6}", result.se_robust);
                        println!("p-value: {:.4}", result.p_robust);
                        println!(
                            "95% CI: [{:.6}, {:.6}]",
                            result.ci_robust.0, result.ci_robust.1
                        );
                        println!("\nN left: {}, N right: {}", result.n_left, result.n_right);
                        println!(
                            "Bandwidth h: left={:.4}, right={:.4}",
                            result.h_left, result.h_right
                        );
                    }
                },
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
                Ok(result) => match format {
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
                        println!(
                            "95% CI: [{:.6}, {:.6}]",
                            result.ci_fuzzy.0, result.ci_fuzzy.1
                        );
                    }
                },
                Err(e) => print_error(&format!("Fuzzy RD estimation failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_staggered_did(
    dataset_name: &str,
    outcome: &str,
    treatment_time: &str,
    time_col: &str,
    unit_col: &str,
    covariates: &[String],
    comparison: &str,
    method: &str,
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
            let covariates_opt = if cov_refs.is_empty() {
                None
            } else {
                Some(cov_refs.as_slice())
            };

            let comparison_group = match comparison.to_lowercase().as_str() {
                "notyettreated" | "notyet" | "nyt" => ComparisonGroup::NotYetTreated,
                _ => ComparisonGroup::NeverTreated,
            };

            let estimation_method = match method.to_lowercase().as_str() {
                "ipw" => AttEstimationMethod::IPW,
                "dr" | "doublyrobust" | "aipw" => AttEstimationMethod::DoublyRobust,
                _ => AttEstimationMethod::OutcomeRegression,
            };

            let config = StaggeredDidConfig {
                comparison_group,
                estimation_method,
                bootstrap,
                ..Default::default()
            };

            match run_staggered_did(
                ds,
                outcome,
                treatment_time,
                time_col,
                unit_col,
                covariates_opt,
                config,
            ) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": "Callaway-Sant'Anna Staggered DiD",
                            "comparison_group": format!("{}", result.config.comparison_group),
                            "estimation_method": format!("{}", result.config.estimation_method),
                            "overall_att": {
                                "att": result.overall_att.att,
                                "std_error": result.overall_att.std_error,
                                "ci_lower": result.overall_att.ci_lower,
                                "ci_upper": result.overall_att.ci_upper,
                                "p_value": result.overall_att.p_value,
                            },
                            "event_study": result.event_study.iter().map(|e| {
                                serde_json::json!({
                                    "relative_time": e.key,
                                    "att": e.att,
                                    "std_error": e.std_error,
                                    "ci_lower": e.ci_lower,
                                    "ci_upper": e.ci_upper,
                                    "p_value": e.p_value,
                                })
                            }).collect::<Vec<_>>(),
                            "cohorts": result.cohorts,
                            "n_obs": result.n_obs,
                            "n_treated": result.n_treated,
                            "n_never_treated": result.n_never_treated,
                            "pretrend_test": result.pretrend_test.as_ref().map(|pt| {
                                serde_json::json!({
                                    "chi2": pt.chi2,
                                    "df": pt.df,
                                    "p_value": pt.p_value,
                                })
                            }),
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("{}", result);
                    }
                },
                Err(e) => print_error(&format!("Staggered DiD failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_tmle(
    dataset_name: &str,
    outcome: &str,
    treatment: &str,
    covariates: &[String],
    qmodel: &str,
    truncate_lower: f64,
    truncate_upper: f64,
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

            let q_model = match qmodel.to_lowercase().as_str() {
                "linear" => QModel::Linear,
                _ => QModel::Logistic,
            };

            let config = TmleConfig {
                q_model,
                g_model: GModel::Logistic,
                truncate_ps: (truncate_lower, truncate_upper),
                ..Default::default()
            };

            match tmle(ds, outcome, treatment, &cov_refs, config) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": "Targeted Maximum Likelihood Estimation (TMLE)",
                            "ate": result.ate,
                            "ate_se": result.ate_se,
                            "ci_lower": result.ate_ci_lower,
                            "ci_upper": result.ate_ci_upper,
                            "z_stat": result.z_stat,
                            "p_value": result.ate_p_value,
                            "significance": format!("{}", result.significance),
                            "fluctuation_coef": result.fluctuation_coef,
                            "n_obs": result.n_obs,
                            "n_treated": result.n_treated,
                            "n_control": result.n_control,
                            "n_truncated": result.n_truncated,
                            "q_model_converged": result.q_model_converged,
                            "g_model_converged": result.g_model_converged,
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nTargeted Maximum Likelihood Estimation (TMLE)");
                        println!("{}", "=".repeat(50));
                        println!("ATE: {:.6}", result.ate);
                        println!("Std Error: {:.6}", result.ate_se);
                        println!(
                            "95% CI: [{:.6}, {:.6}]",
                            result.ate_ci_lower, result.ate_ci_upper
                        );
                        println!(
                            "Z-stat: {:.4}, p-value: {:.4}{}",
                            result.z_stat,
                            result.ate_p_value,
                            result.significance.stars()
                        );
                        println!(
                            "\nFluctuation coefficient (epsilon): {:.6}",
                            result.fluctuation_coef
                        );
                        println!(
                            "N: {} (treated: {}, control: {})",
                            result.n_obs, result.n_treated, result.n_control
                        );
                        if result.n_truncated > 0 {
                            println!("Propensity scores truncated: {}", result.n_truncated);
                        }
                        println!(
                            "Q-model converged: {}, G-model converged: {}",
                            result.q_model_converged, result.g_model_converged
                        );
                    }
                },
                Err(e) => print_error(&format!("TMLE failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_gsynth(
    dataset_name: &str,
    outcome: &str,
    treatment: &str,
    unit_col: &str,
    time_col: &str,
    covariates: &[String],
    n_factors: usize,
    force: &str,
    estimator: &str,
    bootstrap_se: bool,
    n_bootstrap: usize,
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

            let gsynth_force = match force.to_lowercase().as_str() {
                "unit" => GsynthForce::Unit,
                "time" => GsynthForce::Time,
                "none" => GsynthForce::None,
                _ => GsynthForce::TwoWay,
            };

            let gsynth_estimator = match estimator.to_lowercase().as_str() {
                "mc" | "matrixcompletion" => GsynthEstimator::MatrixCompletion,
                _ => GsynthEstimator::Ife,
            };

            let config = GsynthConfig {
                n_factors,
                cross_validate: n_factors == 0,
                force: gsynth_force,
                estimator: gsynth_estimator,
                bootstrap_se,
                n_bootstrap,
                ..Default::default()
            };

            match run_gsynth(
                ds, outcome, treatment, unit_col, time_col, &cov_refs, config,
            ) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": "Generalized Synthetic Control (gsynth)",
                            "att": result.att,
                            "att_se": result.att_se,
                            "att_ci": result.att_ci,
                            "p_value": result.p_value,
                            "n_treated": result.n_treated,
                            "n_control": result.n_control,
                            "n_pre_periods": result.n_pre_periods,
                            "n_post_periods": result.n_post_periods,
                            "n_factors": result.n_factors,
                            "estimator": format!("{:?}", result.estimator),
                            "force": format!("{:?}", result.force),
                            "dynamic_effects": result.dynamic_effects.iter().map(|(t, e)| {
                                serde_json::json!({ "time": t, "effect": e })
                            }).collect::<Vec<_>>(),
                            "unit_effects": result.unit_effects.iter().map(|ue| {
                                serde_json::json!({
                                    "unit": ue.unit,
                                    "treatment_time": ue.treatment_time,
                                    "att": ue.att,
                                    "se": ue.se,
                                })
                            }).collect::<Vec<_>>(),
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("{}", result);
                    }
                },
                Err(e) => print_error(&format!("gsynth failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_matching(
    dataset_name: &str,
    treatment: &str,
    covariates: &[String],
    method: &str,
    distance: &str,
    ratio: usize,
    caliper: Option<f64>,
    replace: bool,
    n_subclasses: usize,
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

            let match_method = match method.to_lowercase().as_str() {
                "cem" | "coarsened" | "coarsenedexact" => MatchMethod::CoarsenedExact {
                    cutpoints: None,
                    n_bins: Some(4),
                },
                "full" | "optimal" => MatchMethod::Full {
                    min_ratio: 0.5,
                    max_ratio: 2.0,
                },
                "subclass" | "stratification" => MatchMethod::Subclass { n_subclasses },
                _ => MatchMethod::NearestNeighbor {
                    ratio,
                    caliper,
                    replace,
                },
            };

            let distance_method = match distance.to_lowercase().as_str() {
                "probit" => DistanceMethod::Probit,
                "mahalanobis" => DistanceMethod::Mahalanobis,
                "euclidean" => DistanceMethod::Euclidean,
                _ => DistanceMethod::Logit,
            };

            match match_it(
                ds,
                treatment,
                &cov_refs,
                match_method,
                Some(distance_method),
            ) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": format!("{}", result.method),
                            "distance": format!("{}", result.distance),
                            "n_obs": result.n_obs,
                            "n_treated": result.n_treated,
                            "n_control": result.n_control,
                            "n_matched_treated": result.n_matched_treated,
                            "n_matched_control": result.n_matched_control,
                            "n_discarded_treated": result.n_discarded_treated,
                            "n_discarded_control": result.n_discarded_control,
                            "effective_sample_size": result.effective_sample_size,
                            "caliper_used": result.caliper_used,
                            "balance_before": {
                                "mean_abs_std_diff": result.balance_before.mean_abs_std_diff,
                                "max_abs_std_diff": result.balance_before.max_abs_std_diff,
                                "n_imbalanced": result.balance_before.n_imbalanced,
                            },
                            "balance_after": {
                                "mean_abs_std_diff": result.balance_after.mean_abs_std_diff,
                                "max_abs_std_diff": result.balance_after.max_abs_std_diff,
                                "n_imbalanced": result.balance_after.n_imbalanced,
                            },
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("{}", result);
                        println!("\nDetailed Balance After Matching:");
                        println!("{}", result.balance_after);
                    }
                },
                Err(e) => print_error(&format!("Matching failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_stdreg(
    dataset_name: &str,
    outcome: &str,
    treatment: &str,
    covariates: &[String],
    model: &str,
    estimand: &str,
    se_method: &str,
    bootstrap: usize,
    interactions: bool,
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

            let model_type = match model.to_lowercase().as_str() {
                "logistic" | "logit" => StdRegModel::Logistic,
                "poisson" => StdRegModel::Poisson,
                _ => StdRegModel::Linear,
            };

            let stdreg_estimand = match estimand.to_lowercase().as_str() {
                "att" => StdRegEstimand::ATT,
                "atc" => StdRegEstimand::ATC,
                "levels" => StdRegEstimand::Levels,
                _ => StdRegEstimand::ATE,
            };

            let se_meth = match se_method.to_lowercase().as_str() {
                "delta" => SEMethod::Delta,
                "sandwich" | "robust" => SEMethod::Sandwich,
                _ => SEMethod::Bootstrap,
            };

            let config = StdRegConfig {
                model_type,
                estimand: stdreg_estimand,
                se_method: se_meth,
                n_bootstrap: bootstrap,
                include_interactions: interactions,
                ..Default::default()
            };

            match run_stdreg(ds, outcome, treatment, &cov_refs, config) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": "Regression Standardization (G-computation)",
                            "estimand": format!("{}", result.estimand),
                            "model_type": format!("{}", result.model_type),
                            "se_method": format!("{}", result.se_method),
                            "ate": result.ate,
                            "se": result.se,
                            "ci_lower": result.ci_lower,
                            "ci_upper": result.ci_upper,
                            "z_stat": result.z_stat,
                            "p_value": result.p_value,
                            "significance": format!("{}", result.significance),
                            "ey1": result.ey1,
                            "ey0": result.ey0,
                            "ey1_se": result.ey1_se,
                            "ey0_se": result.ey0_se,
                            "risk_ratio": result.risk_ratio,
                            "risk_ratio_ci": result.risk_ratio_ci,
                            "odds_ratio": result.odds_ratio,
                            "odds_ratio_ci": result.odds_ratio_ci,
                            "nnt": result.nnt,
                            "n_obs": result.n_obs,
                            "n_treated": result.n_treated,
                            "n_control": result.n_control,
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nRegression Standardization / G-computation");
                        println!("{}", "=".repeat(50));
                        println!("Estimand: {}", result.estimand);
                        println!("Model: {}", result.model_type);
                        println!("SE Method: {}", result.se_method);
                        println!();
                        println!(
                            "Treatment Effect ({:?}): {:.6}",
                            result.estimand, result.ate
                        );
                        println!("Std Error: {:.6}", result.se);
                        println!("95% CI: [{:.6}, {:.6}]", result.ci_lower, result.ci_upper);
                        println!(
                            "Z-stat: {:.4}, p-value: {:.4}{}",
                            result.z_stat,
                            result.p_value,
                            result.significance.stars()
                        );
                        println!();
                        println!("Potential Outcomes:");
                        println!("  E[Y(1)] = {:.6} (SE: {:.6})", result.ey1, result.ey1_se);
                        println!("  E[Y(0)] = {:.6} (SE: {:.6})", result.ey0, result.ey0_se);
                        if let Some(rr) = result.risk_ratio {
                            println!("  Risk Ratio: {:.4}", rr);
                            if let Some((lo, hi)) = result.risk_ratio_ci {
                                println!("  RR 95% CI: [{:.4}, {:.4}]", lo, hi);
                            }
                        }
                        if let Some(or) = result.odds_ratio {
                            println!("  Odds Ratio: {:.4}", or);
                            if let Some((lo, hi)) = result.odds_ratio_ci {
                                println!("  OR 95% CI: [{:.4}, {:.4}]", lo, hi);
                            }
                        }
                        if let Some(nnt) = result.nnt {
                            println!("  Number Needed to Treat: {:.2}", nnt);
                        }
                        println!();
                        println!(
                            "N: {} (treated: {}, control: {})",
                            result.n_obs, result.n_treated, result.n_control
                        );
                    }
                },
                Err(e) => print_error(&format!("Regression standardization failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}
