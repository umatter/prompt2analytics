//! Panel data estimation commands

use clap::Subcommand;
use p2a_core::{run_fixed_effects, run_random_effects, run_hausman_test, run_hdfe};
use p2a_core::regression::CovarianceType;

use crate::output::{format_regression_results, print_error, OutputFormat};
use crate::session::SessionManager;

#[derive(Subcommand)]
pub enum PanelCommands {
    /// Fixed Effects estimation
    Fe {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,

        /// Entity (group) column for fixed effects
        #[arg(long)]
        entity: String,
    },

    /// Random Effects estimation
    Re {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,

        /// Entity (group) column
        #[arg(long)]
        entity: String,
    },

    /// Hausman test (Fixed vs Random Effects)
    Hausman {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,

        /// Entity (group) column
        #[arg(long)]
        entity: String,
    },

    /// High-dimensional Fixed Effects
    Hdfe {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,

        /// Fixed effect columns
        #[arg(long, num_args = 1..)]
        fe: Vec<String>,
    },
}

pub fn execute(
    cmd: &PanelCommands,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    match cmd {
        PanelCommands::Fe {
            dataset,
            dep_var,
            indep_vars,
            entity,
        } => execute_fe(dataset, dep_var, indep_vars, entity, format, session),
        PanelCommands::Re {
            dataset,
            dep_var,
            indep_vars,
            entity,
        } => execute_re(dataset, dep_var, indep_vars, entity, format, session),
        PanelCommands::Hausman {
            dataset,
            dep_var,
            indep_vars,
            entity,
        } => execute_hausman(dataset, dep_var, indep_vars, entity, format, session),
        PanelCommands::Hdfe {
            dataset,
            dep_var,
            indep_vars,
            fe,
        } => execute_hdfe(dataset, dep_var, indep_vars, fe, format, session),
    }
}

fn execute_fe(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    entity: &str,
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

            // run_fixed_effects takes 4 args: dataset, y_col, x_cols, entity_col
            match run_fixed_effects(ds, dep_var, &x_cols, entity) {
                Ok(result) => {
                    // PanelResult uses fields, not methods
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
                        "Fixed Effects (Within) Estimation",
                        &coef_table,
                        result.r_squared,
                        result.adj_r_squared,
                        result.n_obs,
                        format,
                    );
                    println!("{}", output);

                    match format {
                        OutputFormat::Json => {}
                        _ => {
                            println!("\nNumber of groups: {}", result.n_groups);
                            println!("F-statistic: {:.4} (p-value: {:.4})", result.f_stat, result.f_p_value);
                        }
                    }
                }
                Err(e) => {
                    print_error(&format!("Fixed Effects failed: {}", e), format);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

fn execute_re(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    entity: &str,
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

            // run_random_effects takes 4 args: dataset, y_col, x_cols, entity_col
            match run_random_effects(ds, dep_var, &x_cols, entity) {
                Ok(result) => {
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
                        "Random Effects (GLS) Estimation",
                        &coef_table,
                        result.r_squared,
                        result.adj_r_squared,
                        result.n_obs,
                        format,
                    );
                    println!("{}", output);

                    match format {
                        OutputFormat::Json => {}
                        _ => {
                            println!("\nNumber of groups: {}", result.n_groups);
                            println!("F-statistic: {:.4} (p-value: {:.4})", result.f_stat, result.f_p_value);
                        }
                    }
                }
                Err(e) => {
                    print_error(&format!("Random Effects failed: {}", e), format);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

fn execute_hausman(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    entity: &str,
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

            match run_hausman_test(ds, dep_var, &x_cols, entity) {
                Ok(result) => {
                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "test": "Hausman",
                                "chi2_statistic": result.chi2_statistic,
                                "p_value": result.p_value,
                                "df": result.df,
                                "recommendation": result.recommendation,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nHausman Test Results");
                            println!("{}", "=".repeat(50));
                            println!("H0: Random Effects is consistent");
                            println!("H1: Fixed Effects is preferred");
                            println!("\nChi-squared statistic: {:.4}", result.chi2_statistic);
                            println!("Degrees of freedom: {}", result.df);
                            println!("P-value: {:.4}", result.p_value);
                            println!("\nRecommendation: {}", result.recommendation);
                        }
                    }
                }
                Err(e) => {
                    print_error(&format!("Hausman test failed: {}", e), format);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

fn execute_hdfe(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    fe_cols: &[String],
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
            let fe_refs: Vec<&str> = fe_cols.iter().map(|s| s.as_str()).collect();

            // run_hdfe takes 6 args: dataset, y_col, x_cols, fe_cols, config: Option<HdfeConfig>, cov_type
            match run_hdfe(ds, dep_var, &x_cols, &fe_refs, None, CovarianceType::HC1) {
                Ok(result) => {
                    // HdfeResult uses fields, not methods
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
                        &format!("High-Dimensional FE ({})", fe_cols.join(", ")),
                        &coef_table,
                        result.r_squared_within,
                        result.adj_r_squared_within,
                        result.n_obs,
                        format,
                    );
                    println!("{}", output);

                    match format {
                        OutputFormat::Json => {}
                        _ => {
                            println!("\nFixed effect dimensions:");
                            for (dim, count) in result.fe_dimensions.iter().zip(result.fe_counts.iter()) {
                                println!("  {}: {} levels", dim, count);
                            }
                            println!("\nF-statistic: {:.4} (p-value: {:.4})", result.f_stat, result.f_p_value);
                        }
                    }
                }
                Err(e) => {
                    print_error(&format!("HDFE failed: {}", e), format);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}
