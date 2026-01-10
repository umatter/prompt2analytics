//! Causal inference commands

use clap::Subcommand;
use p2a_core::{run_iv2sls, run_did};

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
