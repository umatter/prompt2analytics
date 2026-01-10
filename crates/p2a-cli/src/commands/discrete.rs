//! Discrete choice model commands

use clap::Subcommand;
use p2a_core::{run_logit, run_probit};

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
