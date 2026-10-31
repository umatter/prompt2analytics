//! Discrete choice model commands

use clap::Subcommand;
use p2a_core::data::columns::{MissingSummary, is_numeric_dtype};
use p2a_core::econometrics::{OrderedConfig, run_ordered_model_with};
use p2a_core::{
    GlmFamily, HurdleType, MixedLogitConfig, OrderedModelType, RandomDistribution,
    RandomParameterSpec, run_conditional_logit, run_feglm, run_hurdle, run_logit, run_mixed_logit,
    run_multinom, run_negbin, run_probit, run_zinb, run_zip,
};

use crate::output::{
    OutputFormat, format_regression_results, missing_json, print_error, print_missing_note,
};
use crate::session::SessionManager;

#[derive(Subcommand)]
pub enum DiscreteCommands {
    /// Logit (logistic regression)
    #[command(after_help = "\
EXAMPLES:
    # Binary logit
    p2a discrete logit mydata -y employed -x age education income
")]
    Logit {
        /// Dataset name
        dataset: String,

        /// Dependent variable column (binary 0/1)
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,
    },

    /// Probit regression
    #[command(after_help = "\
EXAMPLES:
    # Binary probit
    p2a discrete probit mydata -y employed -x age education income
")]
    Probit {
        /// Dataset name
        dataset: String,

        /// Dependent variable column (binary 0/1)
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,
    },

    /// Ordered Logit (proportional odds model)
    #[command(after_help = "\
EXAMPLES:
    # Ordered logit for satisfaction ratings (1-5)
    p2a discrete o-logit mydata -y satisfaction -x age income service_quality

    # A text outcome needs its order, lowest category first
    p2a discrete o-logit mydata -y rating -x age income --levels low,mid,high
")]
    OLogit {
        /// Dataset name
        dataset: String,

        /// Dependent variable column (ordered categorical, e.g., 1,2,3,4,5)
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,

        /// Ordinal order of a text outcome, lowest first (e.g. --levels low,mid,high)
        #[arg(long, value_delimiter = ',', num_args = 1..)]
        levels: Option<Vec<String>>,
    },

    /// Ordered Probit
    #[command(after_help = "\
EXAMPLES:
    # Ordered probit; a text outcome needs its order, lowest category first
    p2a discrete o-probit mydata -y rating -x age income --levels low,mid,high
")]
    OProbit {
        /// Dataset name
        dataset: String,

        /// Dependent variable column (ordered categorical)
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,

        /// Ordinal order of a text outcome, lowest first (e.g. --levels low,mid,high)
        #[arg(long, value_delimiter = ',', num_args = 1..)]
        levels: Option<Vec<String>>,
    },

    /// Multinomial Logit
    #[command(after_help = "\
EXAMPLES:
    # Multinomial choice model
    p2a discrete mlogit mydata -y transport_mode -x income distance age
")]
    Mlogit {
        /// Dataset name
        dataset: String,

        /// Dependent variable column (unordered categorical)
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,

        /// Reference category (default: lowest value)
        #[arg(long)]
        base: Option<String>,
    },

    /// Negative Binomial regression (count data with overdispersion)
    #[command(after_help = "\
EXAMPLES:
    # Count data with overdispersion
    p2a discrete negbin mydata -y doctor_visits -x age income chronic_conditions
")]
    Negbin {
        /// Dataset name
        dataset: String,

        /// Dependent variable column (non-negative integer counts)
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,
    },

    /// Poisson regression (count data)
    Poisson {
        /// Dataset name
        dataset: String,

        /// Dependent variable column (non-negative integer counts)
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,
    },

    /// Zero-Inflated Poisson (ZIP) model
    #[command(after_help = "\
EXAMPLES:
    # ZIP with same covariates for both parts
    p2a discrete zip mydata -y patents -x rd_spending firm_size

    # ZIP with different covariates for zero-inflation
    p2a discrete zip mydata -y patents -x rd_spending firm_size -z age industry
")]
    Zip {
        /// Dataset name
        dataset: String,

        /// Dependent variable column (non-negative counts with excess zeros)
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns for count model
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,

        /// Independent variable columns for zero-inflation model (defaults to same as count model)
        #[arg(short = 'z', long, num_args = 1..)]
        zero_vars: Option<Vec<String>>,
    },

    /// Zero-Inflated Negative Binomial (ZINB) model
    Zinb {
        /// Dataset name
        dataset: String,

        /// Dependent variable column (non-negative counts with excess zeros and overdispersion)
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns for count model
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,

        /// Independent variable columns for zero-inflation model (defaults to same as count model)
        #[arg(short = 'z', long, num_args = 1..)]
        zero_vars: Option<Vec<String>>,
    },

    /// Hurdle model (two-part: binary + truncated count)
    Hurdle {
        /// Dataset name
        dataset: String,

        /// Dependent variable column (non-negative counts)
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns for count model
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,

        /// Independent variable columns for binary (hurdle) model (defaults to same as count model)
        #[arg(short = 'z', long, num_args = 1..)]
        zero_vars: Option<Vec<String>>,

        /// Count distribution: poisson or negbin (default: poisson)
        #[arg(long, default_value = "poisson")]
        dist: String,
    },

    /// Conditional Logit (McFadden's choice model for panel/choice data)
    #[command(after_help = "\
EXAMPLES:
    # Discrete choice model for mode choice
    p2a discrete clogit mydata --choice-id trip_id --alt-id mode -y chosen -x travel_time cost
")]
    Clogit {
        /// Dataset name
        dataset: String,

        /// Choice situation ID column (groups alternatives)
        #[arg(long)]
        choice_id: String,

        /// Alternative ID column
        #[arg(long)]
        alt_id: String,

        /// Choice indicator column (1 = chosen, 0 = not chosen)
        #[arg(short = 'y', long)]
        choice: String,

        /// Alternative-specific variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,

        /// Reference alternative (default: first alternative)
        #[arg(long)]
        reference: Option<String>,
    },

    /// Mixed Logit / Random Parameters Logit
    MixedLogit {
        /// Dataset name
        dataset: String,

        /// Choice situation ID column
        #[arg(long)]
        choice_id: String,

        /// Alternative ID column
        #[arg(long)]
        alt_id: String,

        /// Choice indicator column (1 = chosen, 0 = not chosen)
        #[arg(short = 'y', long)]
        choice: String,

        /// Variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,

        /// Variables to treat as random (with normal distribution)
        #[arg(long, num_args = 1..)]
        random_vars: Option<Vec<String>>,

        /// Number of simulation draws (default: 500)
        #[arg(long, default_value = "500")]
        n_draws: usize,

        /// Column identifying the decision maker for repeated choices (mlogit panel = TRUE)
        #[arg(long)]
        individual: Option<String>,
    },
}

pub fn execute(
    cmd: &DiscreteCommands,
    format: &OutputFormat,
    _quiet: bool,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    match cmd {
        DiscreteCommands::Logit {
            dataset,
            dep_var,
            indep_vars,
        } => execute_logit(dataset, dep_var, indep_vars, format, session),
        DiscreteCommands::Probit {
            dataset,
            dep_var,
            indep_vars,
        } => execute_probit(dataset, dep_var, indep_vars, format, session),
        DiscreteCommands::OLogit {
            dataset,
            dep_var,
            indep_vars,
            levels,
        } => execute_ordered(
            dataset,
            dep_var,
            indep_vars,
            levels.as_deref(),
            OrderedModelType::Logit,
            format,
            session,
        ),
        DiscreteCommands::OProbit {
            dataset,
            dep_var,
            indep_vars,
            levels,
        } => execute_ordered(
            dataset,
            dep_var,
            indep_vars,
            levels.as_deref(),
            OrderedModelType::Probit,
            format,
            session,
        ),
        DiscreteCommands::Mlogit {
            dataset,
            dep_var,
            indep_vars,
            base,
        } => execute_mlogit(
            dataset,
            dep_var,
            indep_vars,
            base.as_deref(),
            format,
            session,
        ),
        DiscreteCommands::Negbin {
            dataset,
            dep_var,
            indep_vars,
        } => execute_negbin(dataset, dep_var, indep_vars, format, session),
        DiscreteCommands::Poisson {
            dataset,
            dep_var,
            indep_vars,
        } => execute_poisson(dataset, dep_var, indep_vars, format, session),
        DiscreteCommands::Zip {
            dataset,
            dep_var,
            indep_vars,
            zero_vars,
        } => execute_zip(
            dataset,
            dep_var,
            indep_vars,
            zero_vars.as_deref(),
            format,
            session,
        ),
        DiscreteCommands::Zinb {
            dataset,
            dep_var,
            indep_vars,
            zero_vars,
        } => execute_zinb(
            dataset,
            dep_var,
            indep_vars,
            zero_vars.as_deref(),
            format,
            session,
        ),
        DiscreteCommands::Hurdle {
            dataset,
            dep_var,
            indep_vars,
            zero_vars,
            dist,
        } => execute_hurdle(
            dataset,
            dep_var,
            indep_vars,
            zero_vars.as_deref(),
            dist,
            format,
            session,
        ),
        DiscreteCommands::Clogit {
            dataset,
            choice_id,
            alt_id,
            choice,
            indep_vars,
            reference,
        } => execute_clogit(
            dataset,
            choice_id,
            alt_id,
            choice,
            indep_vars,
            reference.as_deref(),
            format,
            session,
        ),
        DiscreteCommands::MixedLogit {
            dataset,
            choice_id,
            alt_id,
            choice,
            indep_vars,
            random_vars,
            n_draws,
            individual,
        } => execute_mixed_logit(
            dataset,
            choice_id,
            alt_id,
            choice,
            indep_vars,
            random_vars.as_deref(),
            *n_draws,
            individual.as_deref(),
            format,
            session,
        ),
    }
}

/// `Converged: Yes|No (k iterations)`, then the warnings, one per line.
fn print_convergence(converged: bool, iterations: usize, warnings: &[String]) {
    println!(
        "Converged: {} ({} iterations)",
        if converged { "Yes" } else { "No" },
        iterations
    );
    if !warnings.is_empty() {
        println!("Warnings:");
        for w in warnings {
            println!("  - {w}");
        }
    }
}

/// Insert `missing` (when rows were dropped) and `warnings` (when there are any) into a
/// JSON object.
fn insert_notes(json: &mut serde_json::Value, missing: &MissingSummary, warnings: &[String]) {
    if let Some(object) = json.as_object_mut() {
        if let Some(m) = missing_json(missing) {
            object.insert("missing".to_string(), m);
        }
        if !warnings.is_empty() {
            object.insert("warnings".to_string(), serde_json::json!(warnings));
        }
    }
}

/// `output` (a `format_regression_results` JSON document) with `missing` inserted when
/// rows were dropped.
fn with_missing(output: String, missing: &MissingSummary) -> String {
    let Some(m) = missing_json(missing) else {
        return output;
    };
    match serde_json::from_str::<serde_json::Value>(&output) {
        Ok(mut json) => {
            if let Some(object) = json.as_object_mut() {
                object.insert("missing".to_string(), m);
            }
            serde_json::to_string_pretty(&json).unwrap_or(output)
        }
        Err(_) => output,
    }
}

fn execute_logit(
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

            // run_logit takes 3 args: dataset, y_col, x_cols
            match run_logit(ds, dep_var, &x_cols) {
                Ok(result) => {
                    // DiscreteResult uses fields, not methods
                    let coeffs = &result.coefficients;
                    let ses = &result.std_errors;
                    let p_vals = &result.p_values;
                    let z_vals = &result.z_stats;

                    // Build variable names from the result
                    let var_names = &result.variables;

                    let coef_table: Vec<(String, f64, f64, f64, f64)> = var_names
                        .iter()
                        .enumerate()
                        .map(|(i, name)| (name.clone(), coeffs[i], ses[i], z_vals[i], p_vals[i]))
                        .collect();

                    let output = format_regression_results(
                        "Logit Regression (MLE)",
                        &coef_table,
                        result.pseudo_r_squared,
                        result.pseudo_r_squared, // Same for now
                        result.n_obs,
                        format,
                    );
                    match format {
                        OutputFormat::Json => {
                            println!("{}", with_missing(output, &result.missing));
                        }
                        _ => {
                            println!("{}", output);
                            println!("\nLog-likelihood: {:.4}", result.log_likelihood);
                            println!("Null log-likelihood: {:.4}", result.log_likelihood_null);
                            println!("Iterations: {}", result.iterations);
                            print_missing_note(&result.missing, format);
                        }
                    }
                }
                Err(e) => {
                    print_error(&format!("Logit failed: {}", e), format);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

fn execute_probit(
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

            // run_probit takes 3 args: dataset, y_col, x_cols
            match run_probit(ds, dep_var, &x_cols) {
                Ok(result) => {
                    // DiscreteResult uses fields, not methods
                    let coeffs = &result.coefficients;
                    let ses = &result.std_errors;
                    let p_vals = &result.p_values;
                    let z_vals = &result.z_stats;

                    let var_names = &result.variables;

                    let coef_table: Vec<(String, f64, f64, f64, f64)> = var_names
                        .iter()
                        .enumerate()
                        .map(|(i, name)| (name.clone(), coeffs[i], ses[i], z_vals[i], p_vals[i]))
                        .collect();

                    let output = format_regression_results(
                        "Probit Regression (MLE)",
                        &coef_table,
                        result.pseudo_r_squared,
                        result.pseudo_r_squared,
                        result.n_obs,
                        format,
                    );
                    match format {
                        OutputFormat::Json => {
                            println!("{}", with_missing(output, &result.missing));
                        }
                        _ => {
                            println!("{}", output);
                            println!("\nLog-likelihood: {:.4}", result.log_likelihood);
                            println!("Null log-likelihood: {:.4}", result.log_likelihood_null);
                            println!("Iterations: {}", result.iterations);
                            print_missing_note(&result.missing, format);
                        }
                    }
                }
                Err(e) => {
                    print_error(&format!("Probit failed: {}", e), format);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

/// o-logit / o-probit: `run_ordered_model_with` with the given link and `--levels`.
fn execute_ordered(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    levels: Option<&[String]>,
    link: OrderedModelType,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let (method, title) = match link {
        OrderedModelType::Logit => ("Ordered Logit", "Ordered Logit (Proportional Odds) Model"),
        OrderedModelType::Probit => ("Ordered Probit", "Ordered Probit Model"),
    };
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };
    let Some(ds) = dataset else {
        print_error(&format!("Dataset '{}' not found", dataset_name), format);
        return Ok(());
    };
    let x_cols: Vec<&str> = indep_vars.iter().map(|s| s.as_str()).collect();
    let config = OrderedConfig {
        link,
        levels: levels.map(<[String]>::to_vec),
        ..OrderedConfig::default()
    };
    let result = match run_ordered_model_with(ds, dep_var, &x_cols, &config) {
        Ok(r) => r,
        Err(e) => {
            let mut message = format!("{method} failed: {e}");
            let text_outcome = levels.is_none()
                && ds
                    .df()
                    .column(dep_var)
                    .is_ok_and(|c| !is_numeric_dtype(c.dtype(), true));
            if text_outcome {
                message.push_str(
                    " Give the order with --levels, lowest category first (e.g. --levels \
                     low,mid,high).",
                );
            }
            print_error(&message, format);
            return Ok(());
        }
    };
    match format {
        OutputFormat::Json => {
            let mut json = serde_json::json!({
                "method": method,
                "variables": result.variables,
                "coefficients": result.coefficients,
                "std_errors": result.std_errors,
                "z_stats": result.z_stats,
                "p_values": result.p_values,
                "categories": result.categories,
                "thresholds": result.thresholds,
                "threshold_std_errors": result.threshold_std_errors,
                "n_obs": result.n_obs,
                "log_likelihood": result.log_likelihood,
                "aic": result.aic,
                "converged": result.converged,
                "iterations": result.iterations,
            });
            insert_notes(&mut json, &result.missing, &result.warnings);
            println!("{}", serde_json::to_string_pretty(&json)?);
        }
        _ => {
            println!("\n{title}");
            println!("{}", "=".repeat(60));

            println!("\nCoefficients:");
            println!(
                "{:<15} {:>12} {:>12} {:>10} {:>10}",
                "Variable", "Coef", "Std Err", "z", "P>|z|"
            );
            println!("{}", "-".repeat(60));

            for (i, var) in result.variables.iter().enumerate() {
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
                    var,
                    result.coefficients[i],
                    result.std_errors[i],
                    result.z_stats[i],
                    result.p_values[i],
                    sig
                );
            }

            println!("\nThresholds (cutpoints):");
            for (i, (cut, se)) in result
                .thresholds
                .iter()
                .zip(&result.threshold_std_errors)
                .enumerate()
            {
                println!(
                    "  {}|{}: {:.6} (SE: {:.6})",
                    result.categories[i],
                    result.categories[i + 1],
                    cut,
                    se
                );
            }

            println!("\n---");
            println!("Observations: {}", result.n_obs);
            println!("Log-likelihood: {:.4}", result.log_likelihood);
            println!("AIC: {:.4}", result.aic);
            print_convergence(result.converged, result.iterations, &result.warnings);
            print_missing_note(&result.missing, format);
        }
    }
    Ok(())
}

fn execute_mlogit(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    _base: Option<&str>,
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

            match run_multinom(ds, dep_var, &x_cols, None) {
                Ok(result) => {
                    match format {
                        OutputFormat::Json => {
                            let mut json = serde_json::json!({
                                "method": "Multinomial Logit",
                                "categories": result.categories,
                                "reference_category": result.reference_category,
                                "variables": result.variables,
                                "coefficients": result.coefficients,
                                "std_errors": result.std_errors,
                                "z_stats": result.z_stats,
                                "p_values": result.p_values,
                                "n_obs": result.n_obs,
                                "log_likelihood": result.log_likelihood,
                                "aic": result.aic,
                            });
                            insert_notes(&mut json, &result.missing, &[]);
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nMultinomial Logit Model");
                            println!("{}", "=".repeat(60));
                            println!("Reference category: {}", result.reference_category);
                            println!("Categories: {:?}", result.categories);

                            // coefficients is Vec<Vec<f64>> - [category][variable]
                            for (cat_idx, coefs) in result.coefficients.iter().enumerate() {
                                let cat = &result.categories[cat_idx + 1]; // Skip reference
                                println!(
                                    "\n--- Category: {} (vs {}) ---",
                                    cat, result.reference_category
                                );
                                println!(
                                    "{:<15} {:>12} {:>12} {:>10} {:>10}",
                                    "Variable", "Coef", "Std Err", "z", "P>|z|"
                                );
                                println!("{}", "-".repeat(60));

                                for (i, var) in result.variables.iter().enumerate() {
                                    if i < coefs.len()
                                        && cat_idx < result.p_values.len()
                                        && i < result.p_values[cat_idx].len()
                                    {
                                        let sig = if result.p_values[cat_idx][i] < 0.001 {
                                            "***"
                                        } else if result.p_values[cat_idx][i] < 0.01 {
                                            "**"
                                        } else if result.p_values[cat_idx][i] < 0.05 {
                                            "*"
                                        } else {
                                            ""
                                        };
                                        println!(
                                            "{:<15} {:>12.6} {:>12.6} {:>10.4} {:>9.4} {}",
                                            var,
                                            coefs[i],
                                            result.std_errors[cat_idx][i],
                                            result.z_stats[cat_idx][i],
                                            result.p_values[cat_idx][i],
                                            sig
                                        );
                                    }
                                }
                            }

                            println!("\n---");
                            println!("Observations: {}", result.n_obs);
                            println!("Log-likelihood: {:.4}", result.log_likelihood);
                            println!("AIC: {:.4}", result.aic);
                            print_missing_note(&result.missing, format);
                        }
                    }
                }
                Err(e) => print_error(&format!("Multinomial Logit failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_negbin(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
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

            match run_negbin(ds, dep_var, &x_cols, None) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let mut json = serde_json::json!({
                            "method": "Negative Binomial",
                            "variables": result.variables,
                            "coefficients": result.coefficients,
                            "std_errors": result.std_errors,
                            "z_stats": result.z_stats,
                            "p_values": result.p_values,
                            "theta": result.theta,
                            "theta_std_error": result.theta_std_error,
                            "n_obs": result.n_obs,
                            "log_likelihood": result.log_likelihood,
                            "aic": result.aic,
                            "converged": result.converged,
                            "iterations": result.iterations,
                        });
                        insert_notes(&mut json, &result.missing, &result.warnings);
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nNegative Binomial Regression");
                        println!("{}", "=".repeat(60));

                        println!("\nCoefficients:");
                        println!(
                            "{:<15} {:>12} {:>12} {:>10} {:>10}",
                            "Variable", "Coef", "Std Err", "z", "P>|z|"
                        );
                        println!("{}", "-".repeat(60));

                        for (i, var) in result.variables.iter().enumerate() {
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
                                var,
                                result.coefficients[i],
                                result.std_errors[i],
                                result.z_stats[i],
                                result.p_values[i],
                                sig
                            );
                        }

                        println!("\n---");
                        println!(
                            "Theta (dispersion): {:.6} (SE: {:.6})",
                            result.theta, result.theta_std_error
                        );
                        println!("Observations: {}", result.n_obs);
                        println!("Log-likelihood: {:.4}", result.log_likelihood);
                        println!("AIC: {:.4}", result.aic);
                        print_convergence(result.converged, result.iterations, &result.warnings);
                        print_missing_note(&result.missing, format);
                    }
                },
                Err(e) => print_error(&format!("Negative Binomial failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_poisson(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
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

            // Use FEGLM with Poisson family for simple Poisson regression
            match run_feglm(ds, dep_var, &x_cols, &[], GlmFamily::Poisson, None) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": "Poisson Regression",
                            "variables": result.variables,
                            "coefficients": result.coefficients,
                            "std_errors": result.std_errors,
                            "z_stats": result.z_stats,
                            "p_values": result.p_values,
                            "n_obs": result.n_obs,
                            "log_likelihood": result.log_likelihood,
                            "aic": result.aic,
                            "dispersion": result.dispersion,
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nPoisson Regression");
                        println!("{}", "=".repeat(60));

                        println!("\nCoefficients:");
                        println!(
                            "{:<15} {:>12} {:>12} {:>10} {:>10}",
                            "Variable", "Coef", "Std Err", "z", "P>|z|"
                        );
                        println!("{}", "-".repeat(60));

                        for (i, var) in result.variables.iter().enumerate() {
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
                                var,
                                result.coefficients[i],
                                result.std_errors[i],
                                result.z_stats[i],
                                result.p_values[i],
                                sig
                            );
                        }

                        println!("\n---");
                        println!("Observations: {}", result.n_obs);
                        println!("Log-likelihood: {:.4}", result.log_likelihood);
                        println!("AIC: {:.4}", result.aic);
                        println!("Dispersion: {:.4}", result.dispersion);
                    }
                },
                Err(e) => print_error(&format!("Poisson regression failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_zip(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    zero_vars: Option<&[String]>,
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
            let z_cols: Option<Vec<&str>> =
                zero_vars.map(|zv| zv.iter().map(|s| s.as_str()).collect());

            match run_zip(ds, dep_var, &x_cols, z_cols.as_deref()) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let mut json = serde_json::json!({
                            "method": "Zero-Inflated Poisson (ZIP)",
                            "count_variables": result.count_variables,
                            "count_coefficients": result.count_coefficients,
                            "count_std_errors": result.count_std_errors,
                            "count_z_stats": result.count_z_stats,
                            "count_p_values": result.count_p_values,
                            "zero_variables": result.zero_variables,
                            "zero_coefficients": result.zero_coefficients,
                            "zero_std_errors": result.zero_std_errors,
                            "zero_z_stats": result.zero_z_stats,
                            "zero_p_values": result.zero_p_values,
                            "theta_std_error": result.theta_std_error,
                            "log_theta_std_error": result.log_theta_std_error,
                            "n_obs": result.n_obs,
                            "n_zeros": result.n_zeros,
                            "log_likelihood": result.log_likelihood,
                            "aic": result.aic,
                            "converged": result.converged,
                            "iterations": result.iterations,
                        });
                        insert_notes(&mut json, &result.missing, &result.warnings);
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nZero-Inflated Poisson (ZIP) Model");
                        println!("{}", "=".repeat(60));
                        println!(
                            "Zeros: {} ({:.1}%)",
                            result.n_zeros,
                            100.0 * result.n_zeros as f64 / result.n_obs as f64
                        );

                        println!("\nCount Model Coefficients:");
                        println!(
                            "{:<15} {:>12} {:>12} {:>10} {:>10}",
                            "Variable", "Coef", "Std Err", "z", "P>|z|"
                        );
                        println!("{}", "-".repeat(60));

                        for (i, var) in result.count_variables.iter().enumerate() {
                            let sig = if result.count_p_values[i] < 0.001 {
                                "***"
                            } else if result.count_p_values[i] < 0.01 {
                                "**"
                            } else if result.count_p_values[i] < 0.05 {
                                "*"
                            } else {
                                ""
                            };
                            println!(
                                "{:<15} {:>12.6} {:>12.6} {:>10.4} {:>9.4} {}",
                                var,
                                result.count_coefficients[i],
                                result.count_std_errors[i],
                                result.count_z_stats[i],
                                result.count_p_values[i],
                                sig
                            );
                        }

                        println!("\nZero-Inflation Model (logit):");
                        println!(
                            "{:<15} {:>12} {:>12} {:>10} {:>10}",
                            "Variable", "Coef", "Std Err", "z", "P>|z|"
                        );
                        println!("{}", "-".repeat(60));

                        for (i, var) in result.zero_variables.iter().enumerate() {
                            let sig = if result.zero_p_values[i] < 0.001 {
                                "***"
                            } else if result.zero_p_values[i] < 0.01 {
                                "**"
                            } else if result.zero_p_values[i] < 0.05 {
                                "*"
                            } else {
                                ""
                            };
                            println!(
                                "{:<15} {:>12.6} {:>12.6} {:>10.4} {:>9.4} {}",
                                var,
                                result.zero_coefficients[i],
                                result.zero_std_errors[i],
                                result.zero_z_stats[i],
                                result.zero_p_values[i],
                                sig
                            );
                        }

                        println!("\n---");
                        println!("Observations: {}", result.n_obs);
                        println!("Log-likelihood: {:.4}", result.log_likelihood);
                        println!("AIC: {:.4}", result.aic);
                        print_convergence(result.converged, result.iterations, &result.warnings);
                        print_missing_note(&result.missing, format);
                    }
                },
                Err(e) => print_error(&format!("ZIP failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_zinb(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    zero_vars: Option<&[String]>,
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
            let z_cols: Option<Vec<&str>> =
                zero_vars.map(|zv| zv.iter().map(|s| s.as_str()).collect());

            match run_zinb(ds, dep_var, &x_cols, z_cols.as_deref()) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let mut json = serde_json::json!({
                            "method": "Zero-Inflated Negative Binomial (ZINB)",
                            "count_variables": result.count_variables,
                            "count_coefficients": result.count_coefficients,
                            "count_std_errors": result.count_std_errors,
                            "count_z_stats": result.count_z_stats,
                            "count_p_values": result.count_p_values,
                            "zero_variables": result.zero_variables,
                            "zero_coefficients": result.zero_coefficients,
                            "zero_std_errors": result.zero_std_errors,
                            "zero_z_stats": result.zero_z_stats,
                            "zero_p_values": result.zero_p_values,
                            "theta": result.theta,
                            "theta_std_error": result.theta_std_error,
                            "log_theta_std_error": result.log_theta_std_error,
                            "n_obs": result.n_obs,
                            "n_zeros": result.n_zeros,
                            "log_likelihood": result.log_likelihood,
                            "aic": result.aic,
                            "converged": result.converged,
                            "iterations": result.iterations,
                        });
                        insert_notes(&mut json, &result.missing, &result.warnings);
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nZero-Inflated Negative Binomial (ZINB) Model");
                        println!("{}", "=".repeat(60));
                        println!(
                            "Zeros: {} ({:.1}%)",
                            result.n_zeros,
                            100.0 * result.n_zeros as f64 / result.n_obs as f64
                        );

                        println!("\nCount Model Coefficients:");
                        println!(
                            "{:<15} {:>12} {:>12} {:>10} {:>10}",
                            "Variable", "Coef", "Std Err", "z", "P>|z|"
                        );
                        println!("{}", "-".repeat(60));

                        for (i, var) in result.count_variables.iter().enumerate() {
                            let sig = if result.count_p_values[i] < 0.001 {
                                "***"
                            } else if result.count_p_values[i] < 0.01 {
                                "**"
                            } else if result.count_p_values[i] < 0.05 {
                                "*"
                            } else {
                                ""
                            };
                            println!(
                                "{:<15} {:>12.6} {:>12.6} {:>10.4} {:>9.4} {}",
                                var,
                                result.count_coefficients[i],
                                result.count_std_errors[i],
                                result.count_z_stats[i],
                                result.count_p_values[i],
                                sig
                            );
                        }

                        println!("\nZero-Inflation Model (logit):");
                        println!(
                            "{:<15} {:>12} {:>12} {:>10} {:>10}",
                            "Variable", "Coef", "Std Err", "z", "P>|z|"
                        );
                        println!("{}", "-".repeat(60));

                        for (i, var) in result.zero_variables.iter().enumerate() {
                            let sig = if result.zero_p_values[i] < 0.001 {
                                "***"
                            } else if result.zero_p_values[i] < 0.01 {
                                "**"
                            } else if result.zero_p_values[i] < 0.05 {
                                "*"
                            } else {
                                ""
                            };
                            println!(
                                "{:<15} {:>12.6} {:>12.6} {:>10.4} {:>9.4} {}",
                                var,
                                result.zero_coefficients[i],
                                result.zero_std_errors[i],
                                result.zero_z_stats[i],
                                result.zero_p_values[i],
                                sig
                            );
                        }

                        println!("\n---");
                        match (
                            result.theta,
                            result.theta_std_error,
                            result.log_theta_std_error,
                        ) {
                            (Some(theta), Some(se), Some(log_se)) => println!(
                                "Theta (dispersion): {:.6} (SE: {:.6}), log(theta) SE: {:.6}",
                                theta, se, log_se
                            ),
                            (Some(theta), _, _) => println!("Theta (dispersion): {:.6}", theta),
                            _ => {}
                        }
                        println!("Observations: {}", result.n_obs);
                        println!("Log-likelihood: {:.4}", result.log_likelihood);
                        println!("AIC: {:.4}", result.aic);
                        print_convergence(result.converged, result.iterations, &result.warnings);
                        print_missing_note(&result.missing, format);
                    }
                },
                Err(e) => print_error(&format!("ZINB failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_hurdle(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    zero_vars: Option<&[String]>,
    dist: &str,
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
            let z_cols: Option<Vec<&str>> =
                zero_vars.map(|zv| zv.iter().map(|s| s.as_str()).collect());

            let hurdle_type = match dist.to_lowercase().as_str() {
                "negbin" | "nb" => HurdleType::NegBin,
                _ => HurdleType::Poisson,
            };

            match run_hurdle(ds, dep_var, &x_cols, z_cols.as_deref(), hurdle_type) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let mut json = serde_json::json!({
                            "method": format!("Hurdle {:?}", result.model_type),
                            "binary_variables": result.binary_variables,
                            "binary_coefficients": result.binary_coefficients,
                            "binary_std_errors": result.binary_std_errors,
                            "binary_z_stats": result.binary_z_stats,
                            "binary_p_values": result.binary_p_values,
                            "count_variables": result.count_variables,
                            "count_coefficients": result.count_coefficients,
                            "count_std_errors": result.count_std_errors,
                            "count_z_stats": result.count_z_stats,
                            "count_p_values": result.count_p_values,
                            "theta": result.theta,
                            "n_obs": result.n_obs,
                            "n_zeros": result.n_zeros,
                            "n_positive": result.n_positive,
                            "log_likelihood": result.log_likelihood,
                            "aic": result.aic,
                            "converged": result.converged,
                            "iterations": result.iterations,
                        });
                        insert_notes(&mut json, &result.missing, &[]);
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\n{} Model", result.model_type);
                        println!("{}", "=".repeat(60));
                        println!("Zeros: {}, Positive: {}", result.n_zeros, result.n_positive);

                        println!("\nBinary Part (logit: y > 0):");
                        println!(
                            "{:<15} {:>12} {:>12} {:>10} {:>10}",
                            "Variable", "Coef", "Std Err", "z", "P>|z|"
                        );
                        println!("{}", "-".repeat(60));

                        for (i, var) in result.binary_variables.iter().enumerate() {
                            let sig = if result.binary_p_values[i] < 0.001 {
                                "***"
                            } else if result.binary_p_values[i] < 0.01 {
                                "**"
                            } else if result.binary_p_values[i] < 0.05 {
                                "*"
                            } else {
                                ""
                            };
                            println!(
                                "{:<15} {:>12.6} {:>12.6} {:>10.4} {:>9.4} {}",
                                var,
                                result.binary_coefficients[i],
                                result.binary_std_errors[i],
                                result.binary_z_stats[i],
                                result.binary_p_values[i],
                                sig
                            );
                        }

                        println!("\nCount Part (truncated):");
                        println!(
                            "{:<15} {:>12} {:>12} {:>10} {:>10}",
                            "Variable", "Coef", "Std Err", "z", "P>|z|"
                        );
                        println!("{}", "-".repeat(60));

                        for (i, var) in result.count_variables.iter().enumerate() {
                            let sig = if result.count_p_values[i] < 0.001 {
                                "***"
                            } else if result.count_p_values[i] < 0.01 {
                                "**"
                            } else if result.count_p_values[i] < 0.05 {
                                "*"
                            } else {
                                ""
                            };
                            println!(
                                "{:<15} {:>12.6} {:>12.6} {:>10.4} {:>9.4} {}",
                                var,
                                result.count_coefficients[i],
                                result.count_std_errors[i],
                                result.count_z_stats[i],
                                result.count_p_values[i],
                                sig
                            );
                        }

                        println!("\n---");
                        if let Some(theta) = result.theta {
                            println!("Theta (dispersion): {:.4}", theta);
                        }
                        println!("Observations: {}", result.n_obs);
                        println!("Log-likelihood: {:.4}", result.log_likelihood);
                        println!("AIC: {:.4}", result.aic);
                        print_convergence(result.converged, result.iterations, &[]);
                        print_missing_note(&result.missing, format);
                    }
                },
                Err(e) => print_error(&format!("Hurdle model failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_clogit(
    dataset_name: &str,
    choice_id: &str,
    alt_id: &str,
    choice: &str,
    indep_vars: &[String],
    reference: Option<&str>,
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

            match run_conditional_logit(ds, choice_id, alt_id, choice, &x_cols, reference) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": "Conditional Logit (McFadden)",
                            "alternatives": result.alternatives,
                            "reference_alternative": result.reference_alternative,
                            "n_choice_situations": result.n_choice_situations,
                            "n_alternatives": result.n_alternatives,
                            "alt_specific_vars": result.alt_specific_vars,
                            "beta": result.beta,
                            "beta_std_errors": result.beta_std_errors,
                            "beta_z_stats": result.beta_z_stats,
                            "beta_p_values": result.beta_p_values,
                            "log_likelihood": result.log_likelihood,
                            "pseudo_r_squared": result.pseudo_r_squared,
                            "aic": result.aic,
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nConditional Logit (McFadden's Choice Model)");
                        println!("{}", "=".repeat(60));
                        println!("Choice situations: {}", result.n_choice_situations);
                        println!(
                            "Alternatives: {} (reference: {})",
                            result.n_alternatives, result.reference_alternative
                        );

                        println!("\nCoefficients:");
                        println!(
                            "{:<15} {:>12} {:>12} {:>10} {:>10}",
                            "Variable", "Coef", "Std Err", "z", "P>|z|"
                        );
                        println!("{}", "-".repeat(60));

                        for (i, var) in result.alt_specific_vars.iter().enumerate() {
                            let sig = if result.beta_p_values[i] < 0.001 {
                                "***"
                            } else if result.beta_p_values[i] < 0.01 {
                                "**"
                            } else if result.beta_p_values[i] < 0.05 {
                                "*"
                            } else {
                                ""
                            };
                            println!(
                                "{:<15} {:>12.6} {:>12.6} {:>10.4} {:>9.4} {}",
                                var,
                                result.beta[i],
                                result.beta_std_errors[i],
                                result.beta_z_stats[i],
                                result.beta_p_values[i],
                                sig
                            );
                        }

                        println!("\n---");
                        println!("Log-likelihood: {:.4}", result.log_likelihood);
                        println!("Pseudo R-squared: {:.4}", result.pseudo_r_squared);
                        println!("AIC: {:.4}", result.aic);
                        println!(
                            "Converged: {} ({} iterations)",
                            if result.converged { "Yes" } else { "No" },
                            result.iterations
                        );
                    }
                },
                Err(e) => print_error(&format!("Conditional Logit failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_mixed_logit(
    dataset_name: &str,
    choice_id: &str,
    alt_id: &str,
    choice: &str,
    indep_vars: &[String],
    random_vars: Option<&[String]>,
    n_draws: usize,
    individual: Option<&str>,
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

            // Build random parameter specs
            let random_specs: Vec<RandomParameterSpec> = match random_vars {
                Some(vars) => vars
                    .iter()
                    .map(|v| RandomParameterSpec {
                        name: v.clone(),
                        distribution: RandomDistribution::Normal,
                    })
                    .collect(),
                None => x_cols
                    .iter()
                    .map(|v| RandomParameterSpec {
                        name: v.to_string(),
                        distribution: RandomDistribution::Normal,
                    })
                    .collect(),
            };

            let config = MixedLogitConfig {
                n_draws,
                individual_col: individual.map(str::to_string),
                ..Default::default()
            };

            match run_mixed_logit(
                ds,
                choice_id,
                alt_id,
                choice,
                &x_cols,
                &random_specs,
                Some(config),
            ) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let mut json = serde_json::json!({
                            "method": "Mixed Logit (Random Parameters)",
                            "n_choice_situations": result.n_choice_situations,
                            "n_alternatives": result.n_alternatives,
                            "n_draws": result.n_draws,
                            "n_units": result.n_units,
                            "n_individuals": result.n_individuals,
                            "draw_scheme": result.draw_scheme,
                            "variable_names": result.variable_names,
                            "distributions": result.distributions.iter()
                                .map(|d| format!("{:?}", d)).collect::<Vec<_>>(),
                            "means": result.means,
                            "std_devs": result.std_devs,
                            "mean_std_errors": result.mean_std_errors,
                            "mean_z_stats": result.mean_z_stats,
                            "mean_p_values": result.mean_p_values,
                            "std_dev_std_errors": result.std_dev_std_errors,
                            "std_dev_z_stats": result.std_dev_z_stats,
                            "std_dev_p_values": result.std_dev_p_values,
                            "log_likelihood": result.log_likelihood,
                            "aic": result.aic,
                            "converged": result.converged,
                            "iterations": result.iterations,
                            "max_abs_gradient": result.max_abs_gradient,
                        });
                        insert_notes(&mut json, &MissingSummary::default(), &result.warnings);
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nMixed Logit (Random Parameters Logit)");
                        println!("{}", "=".repeat(70));
                        println!("Choice situations: {}", result.n_choice_situations);
                        if let Some(n) = result.n_individuals {
                            println!("Individuals: {}", n);
                        }
                        println!("Alternatives: {}", result.n_alternatives);
                        println!("Simulation draws: {}", result.n_draws);
                        println!("Draws: {}", result.draw_scheme);

                        println!("\nRandom Parameters:");
                        println!(
                            "{:<15} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
                            "Variable", "Mean", "Std Err", "Std.Dev", "SD SE", "Dist", "P>|z|"
                        );
                        println!("{}", "-".repeat(81));

                        for (i, var) in result.variable_names.iter().enumerate() {
                            let dist_str = format!("{:?}", result.distributions[i]);
                            let (sd, sd_se) =
                                if result.distributions[i] == RandomDistribution::Fixed {
                                    ("-".to_string(), "-".to_string())
                                } else {
                                    (
                                        format!("{:.4}", result.std_devs[i]),
                                        format!("{:.4}", result.std_dev_std_errors[i]),
                                    )
                                };
                            let sig = if result.mean_p_values[i] < 0.001 {
                                "***"
                            } else if result.mean_p_values[i] < 0.01 {
                                "**"
                            } else if result.mean_p_values[i] < 0.05 {
                                "*"
                            } else {
                                ""
                            };
                            println!(
                                "{:<15} {:>10.4} {:>10.4} {:>10} {:>10} {:>10} {:>9.4} {}",
                                var,
                                result.means[i],
                                result.mean_std_errors[i],
                                sd,
                                sd_se,
                                dist_str,
                                result.mean_p_values[i],
                                sig
                            );
                        }

                        println!("\n---");
                        println!("Log-likelihood: {:.4}", result.log_likelihood);
                        let pseudo_r2 = 1.0 - result.log_likelihood / result.log_likelihood_null;
                        println!("Pseudo R-squared: {:.4}", pseudo_r2);
                        println!("AIC: {:.4}", result.aic);
                        print_convergence(result.converged, result.iterations, &result.warnings);
                    }
                },
                Err(e) => print_error(&format!("Mixed Logit failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}
