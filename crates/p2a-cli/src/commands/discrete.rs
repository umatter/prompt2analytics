//! Discrete choice model commands

use clap::{Subcommand, ValueEnum};
use p2a_core::data::columns::{ColumnRequest, MissingSummary, NaPolicy, is_numeric_dtype};
use p2a_core::econometrics::discrete::conditional_logit::{
    MlogitConfig, run_conditional_logit_with,
};
use p2a_core::econometrics::discrete::types::DiscreteResult;
use p2a_core::econometrics::glm::{GlmSpec, fit_glm};
use p2a_core::econometrics::{OrderedConfig, run_ordered_model_with};
use p2a_core::linalg::solve::aliasing_message;
use p2a_core::traits::convergence::ResultNotes;
use p2a_core::traits::estimator::normal_p_value_two_sided;
use p2a_core::{
    HurdleType, MixedLogitConfig, OrderedModelType, RandomDistribution, RandomParameterSpec,
    run_hurdle, run_logit, run_mixed_logit, run_multinom, run_negbin, run_probit, run_zinb,
    run_zip,
};

use crate::output::{
    OutputFormat, add_notes, add_result_notes, format_regression_results, missing_json,
    print_error, print_missing_note, print_notes, print_result_notes, regression_results_json,
};
use crate::session::SessionManager;

#[derive(Subcommand)]
pub enum DiscreteCommands {
    /// Logit (logistic regression)
    #[command(after_help = "\
EXAMPLES:
    # Binary logit
    p2a --session s.json discrete logit mydata -y employed -x age education income
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
    p2a --session s.json discrete probit mydata -y employed -x age education income
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
    p2a --session s.json discrete o-logit mydata -y satisfaction -x age income service_quality

    # A text outcome needs its order, lowest category first
    p2a --session s.json discrete o-logit mydata -y rating -x age income --levels low,mid,high
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
    p2a --session s.json discrete o-probit mydata -y rating -x age income --levels low,mid,high
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
    p2a --session s.json discrete mlogit mydata -y transport_mode -x income distance age
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
    p2a --session s.json discrete negbin mydata -y doctor_visits -x age income chronic_conditions
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
    p2a --session s.json discrete zip mydata -y patents -x rd_spending firm_size

    # ZIP with different covariates for zero-inflation
    p2a --session s.json discrete zip mydata -y patents -x rd_spending firm_size -z age industry
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

        /// Count distribution: poisson (default) or negbin (aliases nb, negative-binomial)
        #[arg(long, value_enum, ignore_case = true, default_value = "poisson")]
        dist: HurdleDist,
    },

    /// Conditional Logit (McFadden's choice model for panel/choice data)
    #[command(after_help = "\
EXAMPLES:
    # Discrete choice model for mode choice
    p2a --session s.json discrete clogit mydata --choice-id trip_id --alt-id mode -y chosen -x travel_time cost
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

        /// Reference alternative (mlogit's reflevel); default: the first alternative in sorted
        /// order
        #[arg(long, visible_alias = "reflevel")]
        reference: Option<String>,

        /// Alternative-specific constants (true/false; default true), as mlogit's default;
        /// false fits mlogit(choice ~ x | 0)
        #[arg(
            long,
            value_name = "BOOL",
            num_args = 0..=1,
            default_value_t = true,
            default_missing_value = "true",
            action = clap::ArgAction::Set,
            value_parser = clap::builder::BoolishValueParser::new()
        )]
        asc: bool,
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

/// Count distribution of `discrete hurdle --dist` (the core's `HurdleType` has no `FromStr`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum HurdleDist {
    /// Truncated Poisson count part.
    Poisson,
    /// Truncated negative binomial count part.
    #[value(name = "negbin", aliases = ["nb", "negative-binomial", "negative_binomial"])]
    NegBin,
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
            *dist,
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
            asc,
        } => execute_clogit(
            dataset,
            choice_id,
            alt_id,
            choice,
            indep_vars,
            reference.as_deref(),
            *asc,
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

/// The logit/probit report: the coefficient table (aliased rows `NaN` in text, `null` in JSON),
/// the log-likelihoods, the convergence line and every core warning, then T2's missing-value note.
/// JSON adds `log_likelihood`, `log_likelihood_null`, `aic`, `separation`, `aliased`, `missing`
/// (when rows were dropped) and the notes (`converged`, `iterations`, `warnings`).
fn print_binary(
    method: &str,
    result: &DiscreteResult,
    format: &OutputFormat,
) -> anyhow::Result<()> {
    let coef_table: Vec<(String, f64, f64, f64, f64)> = result
        .variables
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let at = |v: &[f64]| v.get(i).copied().unwrap_or(f64::NAN);
            (
                name.clone(),
                at(&result.coefficients),
                at(&result.std_errors),
                at(&result.z_stats),
                at(&result.p_values),
            )
        })
        .collect();
    match format {
        OutputFormat::Json => {
            let mut json = regression_results_json(
                method,
                &coef_table,
                result.pseudo_r_squared,
                result.pseudo_r_squared,
                result.n_obs,
            );
            if let Some(object) = json.as_object_mut() {
                object.insert("log_likelihood".into(), result.log_likelihood.into());
                object.insert(
                    "log_likelihood_null".into(),
                    result.log_likelihood_null.into(),
                );
                object.insert("aic".into(), result.aic.into());
                object.insert("separation".into(), result.separation.into());
                object.insert("aliased".into(), serde_json::json!(result.aliased));
                if let Some(m) = missing_json(&result.missing) {
                    object.insert("missing".into(), m);
                }
            }
            add_result_notes(&mut json, result);
            println!("{}", serde_json::to_string_pretty(&json)?);
        }
        _ => {
            let output = format_regression_results(
                method,
                &coef_table,
                result.pseudo_r_squared,
                result.pseudo_r_squared,
                result.n_obs,
                format,
            );
            println!("{}", output);
            println!("\nLog-likelihood: {:.4}", result.log_likelihood);
            println!("Null log-likelihood: {:.4}", result.log_likelihood_null);
            print_result_notes(result, format);
            print_missing_note(&result.missing, format);
        }
    }
    Ok(())
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
                    print_binary("Logit Regression (MLE)", &result, format)?;
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
                    print_binary("Probit Regression (MLE)", &result, format)?;
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
            crate::output::add_missing_and_warnings(&mut json, &result.missing, &result.warnings);
            add_notes(
                &mut json,
                Some((result.converged, result.iterations)),
                &result.warnings,
            );
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
    base: Option<&str>,
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

            match run_multinom(ds, dep_var, &x_cols, base) {
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
                                "separation": result.separation,
                            });
                            crate::output::add_missing_and_warnings(
                                &mut json,
                                &result.missing,
                                &[],
                            );
                            add_result_notes(&mut json, &result);
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
                            print_result_notes(&result, format);
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
                        crate::output::add_missing_and_warnings(
                            &mut json,
                            &result.missing,
                            &result.warnings,
                        );
                        add_result_notes(&mut json, &result);
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

/// `discrete poisson`: glm(family = poisson) through the shared GLM, rows with a missing
/// outcome or regressor dropped jointly (glm's na.omit).
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
    let Some(ds) = dataset else {
        print_error(&format!("Dataset '{}' not found", dataset_name), format);
        return Ok(());
    };
    let x_cols: Vec<&str> = indep_vars.iter().map(|s| s.as_str()).collect();
    let fitted = ColumnRequest::new(ds.df())
        .numeric(dep_var)
        .numerics(&x_cols)
        .na(NaPolicy::Omit)
        .min_rows(x_cols.len() + 2, "Poisson GLM")
        .extract()
        .and_then(|cols| {
            let x = cols.matrix(&x_cols, true)?;
            let y = cols.numeric_array(dep_var);
            let fit = fit_glm(x.data.view(), y.view(), &GlmSpec::poisson())?;
            Ok((cols.missing().clone(), x.column_names, fit))
        });
    let (missing, variables, fit) = match fitted {
        Ok(v) => v,
        Err(e) => {
            print_error(&format!("Poisson regression failed: {}", e), format);
            return Ok(());
        }
    };
    let z: Vec<f64> = fit
        .coefficients
        .iter()
        .zip(fit.std_errors.iter())
        .map(|(b, se)| {
            if se.is_finite() && *se > 0.0 {
                b / se
            } else {
                f64::NAN
            }
        })
        .collect();
    let p: Vec<f64> = z.iter().map(|z| normal_p_value_two_sided(*z)).collect();
    let aliased: Vec<String> = fit
        .aliased
        .iter()
        .filter_map(|&j| variables.get(j).cloned())
        .collect();
    let mut warnings = fit.warnings.clone();
    if let Some(msg) = aliasing_message(&fit.aliased, &variables)
        && !warnings.contains(&msg)
    {
        warnings.push(msg);
    }
    let n_obs = fit.fitted.len();
    let converged = Some((fit.converged, fit.iterations));
    match format {
        OutputFormat::Json => {
            let mut json = serde_json::json!({
                "method": "Poisson Regression",
                "variables": variables,
                "coefficients": fit.coefficients.to_vec(),
                "std_errors": fit.std_errors.to_vec(),
                "z_stats": z,
                "p_values": p,
                "n_obs": n_obs,
                "log_likelihood": fit.log_likelihood,
                "aic": fit.aic,
                "dispersion": fit.dispersion,
                "deviance": fit.deviance,
                "null_deviance": fit.null_deviance,
                "df_residual": fit.df_residual,
                "aliased": aliased,
            });
            crate::output::add_missing_and_warnings(&mut json, &missing, &[]);
            add_notes(&mut json, converged, &warnings);
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

            for (i, var) in variables.iter().enumerate() {
                let sig = if p[i] < 0.001 {
                    "***"
                } else if p[i] < 0.01 {
                    "**"
                } else if p[i] < 0.05 {
                    "*"
                } else {
                    ""
                };
                println!(
                    "{:<15} {:>12.6} {:>12.6} {:>10.4} {:>9.4} {}",
                    var, fit.coefficients[i], fit.std_errors[i], z[i], p[i], sig
                );
            }

            println!("\n---");
            println!("Observations: {}", n_obs);
            println!("Log-likelihood: {:.4}", fit.log_likelihood);
            println!("AIC: {:.4}", fit.aic);
            println!("Deviance: {:.4}", fit.deviance);
            println!("Null deviance: {:.4}", fit.null_deviance);
            println!("Dispersion: {:.4}", fit.dispersion);
            print_notes(converged, &warnings, format);
            print_missing_note(&missing, format);
        }
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
                        crate::output::add_missing_and_warnings(
                            &mut json,
                            &result.missing,
                            &result.warnings,
                        );
                        add_result_notes(&mut json, &result);
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
                        crate::output::add_missing_and_warnings(
                            &mut json,
                            &result.missing,
                            &result.warnings,
                        );
                        add_result_notes(&mut json, &result);
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
    dist: HurdleDist,
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

            let hurdle_type = match dist {
                HurdleDist::Poisson => HurdleType::Poisson,
                HurdleDist::NegBin => HurdleType::NegBin,
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
                        crate::output::add_missing_and_warnings(
                            &mut json,
                            &result.missing,
                            &result.warnings,
                        );
                        add_result_notes(&mut json, &result);
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
                        print_notes(None, &result.warnings, format);
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

#[allow(clippy::too_many_arguments)]
fn execute_clogit(
    dataset_name: &str,
    choice_id: &str,
    alt_id: &str,
    choice: &str,
    indep_vars: &[String],
    reference: Option<&str>,
    asc: bool,
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
    let Some(ds) = dataset else {
        print_error(&format!("Dataset '{}' not found", dataset_name), format);
        return Ok(());
    };
    let x_cols: Vec<&str> = indep_vars.iter().map(|s| s.as_str()).collect();
    let mut config = MlogitConfig::default();
    config.asc = asc;
    config.reflevel = reference.map(str::to_string);
    let result = match run_conditional_logit_with(ds, choice_id, alt_id, choice, &x_cols, &config) {
        Ok(r) => r,
        Err(e) => {
            print_error(&format!("Conditional Logit failed: {}", e), format);
            return Ok(());
        }
    };
    // every coefficient in mlogit's order: the constants (with --asc), then the generic ones
    let names = result.coefficient_names();
    let coefs = result.coefficients();
    let ses = result.std_errors();
    let rows: Vec<(String, f64, f64, f64, f64)> = names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let b = coefs.get(i).copied().unwrap_or(f64::NAN);
            let se = ses.get(i).copied().unwrap_or(f64::NAN);
            let z = if se.is_finite() && se > 0.0 {
                b / se
            } else {
                f64::NAN
            };
            (name.clone(), b, se, z, normal_p_value_two_sided(z))
        })
        .collect();
    match format {
        OutputFormat::Json => {
            let coefficients: Vec<serde_json::Value> = rows
                .iter()
                .map(|(name, b, se, z, p)| {
                    serde_json::json!({
                        "name": name,
                        "coefficient": b,
                        "std_error": se,
                        "z_value": z,
                        "p_value": p,
                    })
                })
                .collect();
            let mut json = serde_json::json!({
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
                "coefficients": coefficients,
                "asc": result.asc,
                "separation": result.separation,
                "log_likelihood": result.log_likelihood,
                "pseudo_r_squared": result.pseudo_r_squared,
                "aic": result.aic,
            });
            add_result_notes(&mut json, &result);
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
                "{:<20} {:>12} {:>12} {:>10} {:>10}",
                "Variable", "Coef", "Std Err", "z", "P>|z|"
            );
            println!("{}", "-".repeat(65));

            for (name, b, se, z, p) in &rows {
                let sig = if *p < 0.001 {
                    "***"
                } else if *p < 0.01 {
                    "**"
                } else if *p < 0.05 {
                    "*"
                } else {
                    ""
                };
                println!(
                    "{:<20} {:>12.6} {:>12.6} {:>10.4} {:>9.4} {}",
                    name, b, se, z, p, sig
                );
            }

            println!("\n---");
            println!("Log-likelihood: {:.4}", result.log_likelihood);
            println!("Pseudo R-squared: {:.4}", result.pseudo_r_squared);
            println!("AIC: {:.4}", result.aic);
            println!(
                "Alternative-specific constants: {}",
                if result.asc { "yes" } else { "no" }
            );
            println!(
                "Converged: {} ({} iterations)",
                if result.converged { "Yes" } else { "No" },
                result.iterations
            );
            print_notes(None, &result.notes_warnings(), format);
        }
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
                        crate::output::add_missing_and_warnings(
                            &mut json,
                            &MissingSummary::default(),
                            &result.warnings,
                        );
                        add_notes(
                            &mut json,
                            Some((result.converged, result.iterations)),
                            &result.warnings,
                        );
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
