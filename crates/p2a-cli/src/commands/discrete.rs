//! Discrete choice model commands

use clap::Subcommand;
use p2a_core::{run_logit, run_probit, run_ordered_logit, run_ordered_probit, run_multinom, run_negbin};

use crate::output::{format_regression_results, print_error, OutputFormat};
use crate::session::SessionManager;

#[derive(Subcommand)]
pub enum DiscreteCommands {
    /// Logit (logistic regression)
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
    OLogit {
        /// Dataset name
        dataset: String,

        /// Dependent variable column (ordered categorical, e.g., 1,2,3,4,5)
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,
    },

    /// Ordered Probit
    OProbit {
        /// Dataset name
        dataset: String,

        /// Dependent variable column (ordered categorical)
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,
    },

    /// Multinomial Logit
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
}

pub fn execute(
    cmd: &DiscreteCommands,
    format: &OutputFormat,
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
        } => execute_ologit(dataset, dep_var, indep_vars, format, session),
        DiscreteCommands::OProbit {
            dataset,
            dep_var,
            indep_vars,
        } => execute_oprobit(dataset, dep_var, indep_vars, format, session),
        DiscreteCommands::Mlogit {
            dataset,
            dep_var,
            indep_vars,
            base,
        } => execute_mlogit(dataset, dep_var, indep_vars, base.as_deref(), format, session),
        DiscreteCommands::Negbin {
            dataset,
            dep_var,
            indep_vars,
        } => execute_negbin(dataset, dep_var, indep_vars, format, session),
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
                        .map(|(i, name)| {
                            (
                                name.clone(),
                                coeffs[i],
                                ses[i],
                                z_vals[i],
                                p_vals[i],
                            )
                        })
                        .collect();

                    let output = format_regression_results(
                        "Logit Regression (MLE)",
                        &coef_table,
                        result.pseudo_r_squared,
                        result.pseudo_r_squared, // Same for now
                        result.n_obs,
                        format,
                    );
                    println!("{}", output);

                    // Print additional info
                    match format {
                        OutputFormat::Json => {}
                        _ => {
                            println!("\nLog-likelihood: {:.4}", result.log_likelihood);
                            println!("Null log-likelihood: {:.4}", result.log_likelihood_null);
                            println!("Iterations: {}", result.iterations);
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
                        .map(|(i, name)| {
                            (
                                name.clone(),
                                coeffs[i],
                                ses[i],
                                z_vals[i],
                                p_vals[i],
                            )
                        })
                        .collect();

                    let output = format_regression_results(
                        "Probit Regression (MLE)",
                        &coef_table,
                        result.pseudo_r_squared,
                        result.pseudo_r_squared,
                        result.n_obs,
                        format,
                    );
                    println!("{}", output);

                    match format {
                        OutputFormat::Json => {}
                        _ => {
                            println!("\nLog-likelihood: {:.4}", result.log_likelihood);
                            println!("Null log-likelihood: {:.4}", result.log_likelihood_null);
                            println!("Iterations: {}", result.iterations);
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

fn execute_ologit(
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

            match run_ordered_logit(ds, dep_var, &x_cols) {
                Ok(result) => {
                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "method": "Ordered Logit",
                                "variables": result.variables,
                                "coefficients": result.coefficients,
                                "std_errors": result.std_errors,
                                "z_stats": result.z_stats,
                                "p_values": result.p_values,
                                "thresholds": result.thresholds,
                                "n_obs": result.n_obs,
                                "log_likelihood": result.log_likelihood,
                                "aic": result.aic,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nOrdered Logit (Proportional Odds) Model");
                            println!("{}", "=".repeat(60));

                            println!("\nCoefficients:");
                            println!("{:<15} {:>12} {:>12} {:>10} {:>10}",
                                "Variable", "Coef", "Std Err", "z", "P>|z|");
                            println!("{}", "-".repeat(60));

                            for (i, var) in result.variables.iter().enumerate() {
                                let sig = if result.p_values[i] < 0.001 { "***" }
                                         else if result.p_values[i] < 0.01 { "**" }
                                         else if result.p_values[i] < 0.05 { "*" }
                                         else { "" };
                                println!("{:<15} {:>12.6} {:>12.6} {:>10.4} {:>9.4} {}",
                                    var, result.coefficients[i], result.std_errors[i],
                                    result.z_stats[i], result.p_values[i], sig);
                            }

                            println!("\nThresholds (cutpoints):");
                            for (i, cut) in result.thresholds.iter().enumerate() {
                                println!("  Threshold{}: {:.4}", i + 1, cut);
                            }

                            println!("\n---");
                            println!("Observations: {}", result.n_obs);
                            println!("Log-likelihood: {:.4}", result.log_likelihood);
                            println!("AIC: {:.4}", result.aic);
                        }
                    }
                }
                Err(e) => print_error(&format!("Ordered Logit failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_oprobit(
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

            match run_ordered_probit(ds, dep_var, &x_cols) {
                Ok(result) => {
                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "method": "Ordered Probit",
                                "variables": result.variables,
                                "coefficients": result.coefficients,
                                "std_errors": result.std_errors,
                                "z_stats": result.z_stats,
                                "p_values": result.p_values,
                                "thresholds": result.thresholds,
                                "n_obs": result.n_obs,
                                "log_likelihood": result.log_likelihood,
                                "aic": result.aic,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nOrdered Probit Model");
                            println!("{}", "=".repeat(60));

                            println!("\nCoefficients:");
                            println!("{:<15} {:>12} {:>12} {:>10} {:>10}",
                                "Variable", "Coef", "Std Err", "z", "P>|z|");
                            println!("{}", "-".repeat(60));

                            for (i, var) in result.variables.iter().enumerate() {
                                let sig = if result.p_values[i] < 0.001 { "***" }
                                         else if result.p_values[i] < 0.01 { "**" }
                                         else if result.p_values[i] < 0.05 { "*" }
                                         else { "" };
                                println!("{:<15} {:>12.6} {:>12.6} {:>10.4} {:>9.4} {}",
                                    var, result.coefficients[i], result.std_errors[i],
                                    result.z_stats[i], result.p_values[i], sig);
                            }

                            println!("\nThresholds:");
                            for (i, cut) in result.thresholds.iter().enumerate() {
                                println!("  Threshold{}: {:.4}", i + 1, cut);
                            }

                            println!("\n---");
                            println!("Observations: {}", result.n_obs);
                            println!("Log-likelihood: {:.4}", result.log_likelihood);
                            println!("AIC: {:.4}", result.aic);
                        }
                    }
                }
                Err(e) => print_error(&format!("Ordered Probit failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
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
                            let json = serde_json::json!({
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
                                println!("\n--- Category: {} (vs {}) ---", cat, result.reference_category);
                                println!("{:<15} {:>12} {:>12} {:>10} {:>10}",
                                    "Variable", "Coef", "Std Err", "z", "P>|z|");
                                println!("{}", "-".repeat(60));

                                for (i, var) in result.variables.iter().enumerate() {
                                    if i < coefs.len() && cat_idx < result.p_values.len() && i < result.p_values[cat_idx].len() {
                                        let sig = if result.p_values[cat_idx][i] < 0.001 { "***" }
                                                 else if result.p_values[cat_idx][i] < 0.01 { "**" }
                                                 else if result.p_values[cat_idx][i] < 0.05 { "*" }
                                                 else { "" };
                                        println!("{:<15} {:>12.6} {:>12.6} {:>10.4} {:>9.4} {}",
                                            var, coefs[i],
                                            result.std_errors[cat_idx][i],
                                            result.z_stats[cat_idx][i],
                                            result.p_values[cat_idx][i], sig);
                                    }
                                }
                            }

                            println!("\n---");
                            println!("Observations: {}", result.n_obs);
                            println!("Log-likelihood: {:.4}", result.log_likelihood);
                            println!("AIC: {:.4}", result.aic);
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
                Ok(result) => {
                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "method": "Negative Binomial",
                                "variables": result.variables,
                                "coefficients": result.coefficients,
                                "std_errors": result.std_errors,
                                "z_stats": result.z_stats,
                                "p_values": result.p_values,
                                "theta": result.theta,
                                "n_obs": result.n_obs,
                                "log_likelihood": result.log_likelihood,
                                "aic": result.aic,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nNegative Binomial Regression");
                            println!("{}", "=".repeat(60));

                            println!("\nCoefficients:");
                            println!("{:<15} {:>12} {:>12} {:>10} {:>10}",
                                "Variable", "Coef", "Std Err", "z", "P>|z|");
                            println!("{}", "-".repeat(60));

                            for (i, var) in result.variables.iter().enumerate() {
                                let sig = if result.p_values[i] < 0.001 { "***" }
                                         else if result.p_values[i] < 0.01 { "**" }
                                         else if result.p_values[i] < 0.05 { "*" }
                                         else { "" };
                                println!("{:<15} {:>12.6} {:>12.6} {:>10.4} {:>9.4} {}",
                                    var, result.coefficients[i], result.std_errors[i],
                                    result.z_stats[i], result.p_values[i], sig);
                            }

                            println!("\n---");
                            println!("Theta (dispersion): {:.4}", result.theta);
                            println!("Observations: {}", result.n_obs);
                            println!("Log-likelihood: {:.4}", result.log_likelihood);
                            println!("AIC: {:.4}", result.aic);
                        }
                    }
                }
                Err(e) => print_error(&format!("Negative Binomial failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}
