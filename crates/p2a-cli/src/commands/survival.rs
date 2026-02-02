//! Survival analysis commands

use clap::Subcommand;
use p2a_core::{
    AftConfig, AftDistribution, CoxConfig, TiesMethod, log_rank_test, run_aft, run_competing_risks,
    run_cox_ph, run_kaplan_meier,
};

use crate::output::{OutputFormat, print_error};
use crate::session::SessionManager;

#[derive(Subcommand)]
pub enum SurvivalCommands {
    /// Kaplan-Meier survival curve estimation
    #[command(after_help = "\
EXAMPLES:
    # Basic survival curve
    p2a --session s.json survival km mydata -t time -e status

    # Stratified by treatment group
    p2a --session s.json survival km mydata -t time -e status -g treatment
")]
    Km {
        /// Dataset name
        dataset: String,

        /// Time variable column (duration/time to event)
        #[arg(short = 't', long)]
        time: String,

        /// Event indicator column (1 = event, 0 = censored)
        #[arg(short = 'e', long)]
        event: String,

        /// Optional grouping variable for stratified analysis
        #[arg(short = 'g', long)]
        group: Option<String>,

        /// Confidence level (default 0.95)
        #[arg(long, default_value = "0.95")]
        conf: f64,
    },

    /// Log-rank test comparing survival curves between groups
    #[command(after_help = "\
EXAMPLES:
    p2a --session s.json survival log-rank mydata -t time -e status -g treatment
")]
    LogRank {
        /// Dataset name
        dataset: String,

        /// Time variable column
        #[arg(short = 't', long)]
        time: String,

        /// Event indicator column
        #[arg(short = 'e', long)]
        event: String,

        /// Grouping variable column
        #[arg(short = 'g', long)]
        group: String,
    },

    /// Cox Proportional Hazards regression
    #[command(after_help = "\
EXAMPLES:
    # Cox PH with robust standard errors
    p2a --session s.json survival cox mydata -t time -e status -x age treatment stage --robust

    # Using Efron ties handling
    p2a --session s.json survival cox mydata -t time -e status -x age bmi --ties efron
")]
    Cox {
        /// Dataset name
        dataset: String,

        /// Time variable column
        #[arg(short = 't', long)]
        time: String,

        /// Event indicator column
        #[arg(short = 'e', long)]
        event: String,

        /// Covariate columns
        #[arg(short = 'x', long, num_args = 1..)]
        covariates: Vec<String>,

        /// Ties handling method: "breslow" (default), "efron"
        #[arg(long, default_value = "breslow")]
        ties: String,

        /// Use robust (sandwich) standard errors
        #[arg(long)]
        robust: bool,
    },

    /// Accelerated Failure Time model
    #[command(after_help = "\
EXAMPLES:
    # Weibull AFT (default)
    p2a --session s.json survival aft mydata -t time -e status -x age treatment

    # Log-normal distribution
    p2a --session s.json survival aft mydata -t time -e status -x age --dist lognormal
")]
    Aft {
        /// Dataset name
        dataset: String,

        /// Time variable column
        #[arg(short = 't', long)]
        time: String,

        /// Event indicator column
        #[arg(short = 'e', long)]
        event: String,

        /// Covariate columns
        #[arg(short = 'x', long, num_args = 1..)]
        covariates: Vec<String>,

        /// Distribution: "weibull" (default), "exponential", "lognormal", "loglogistic"
        #[arg(long, default_value = "weibull")]
        dist: String,
    },

    /// Competing risks analysis (cumulative incidence)
    #[command(after_help = "\
EXAMPLES:
    # Event types: 0=censored, 1=death, 2=relapse
    p2a --session s.json survival competing-risks mydata -t time -e event_type
")]
    CompetingRisks {
        /// Dataset name
        dataset: String,

        /// Time variable column
        #[arg(short = 't', long)]
        time: String,

        /// Event type column (0 = censored, 1,2,... = event types)
        #[arg(short = 'e', long)]
        event_type: String,

        /// Confidence level (default 0.95)
        #[arg(long, default_value = "0.95")]
        conf: f64,
    },
}

pub fn execute(
    cmd: &SurvivalCommands,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    match cmd {
        SurvivalCommands::Km {
            dataset,
            time,
            event,
            group,
            conf,
        } => execute_km(
            dataset,
            time,
            event,
            group.as_deref(),
            *conf,
            format,
            session,
        ),
        SurvivalCommands::LogRank {
            dataset,
            time,
            event,
            group,
        } => execute_log_rank(dataset, time, event, group, format, session),
        SurvivalCommands::Cox {
            dataset,
            time,
            event,
            covariates,
            ties,
            robust,
        } => execute_cox(
            dataset, time, event, covariates, ties, *robust, format, session,
        ),
        SurvivalCommands::Aft {
            dataset,
            time,
            event,
            covariates,
            dist,
        } => execute_aft(dataset, time, event, covariates, dist, format, session),
        SurvivalCommands::CompetingRisks {
            dataset,
            time,
            event_type,
            conf,
        } => execute_competing_risks(dataset, time, event_type, *conf, format, session),
    }
}

fn execute_km(
    dataset_name: &str,
    time_col: &str,
    event_col: &str,
    group_col: Option<&str>,
    conf_level: f64,
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
            match run_kaplan_meier(ds, time_col, event_col, group_col, conf_level) {
                Ok(results) => {
                    // Results is a Vec<KaplanMeierResult>, one per group
                    for result in &results {
                        match format {
                            OutputFormat::Json => {
                                let json = serde_json::json!({
                                    "method": "Kaplan-Meier",
                                    "group": result.group,
                                    "n_obs": result.n_obs,
                                    "total_events": result.total_events,
                                    "total_censored": result.total_censored,
                                    "median_survival": result.median_survival,
                                    "times": result.times,
                                    "survival": result.survival,
                                    "std_errors": result.std_errors,
                                });
                                println!("{}", serde_json::to_string_pretty(&json)?);
                            }
                            _ => {
                                println!("\nKaplan-Meier Survival Estimates");
                                println!("{}", "=".repeat(50));
                                if let Some(ref group) = result.group {
                                    println!("Group: {}", group);
                                }
                                println!("Observations: {}", result.n_obs);
                                println!("Events: {}", result.total_events);
                                println!("Censored: {}", result.total_censored);
                                if let Some(median) = result.median_survival {
                                    println!("Median survival time: {:.4}", median);
                                } else {
                                    println!("Median survival time: not reached");
                                }
                                println!("\nSurvival table (first 10 time points):");
                                println!("{:<12} {:<12} {:<12}", "Time", "Survival", "Std.Error");
                                println!("{}", "-".repeat(36));
                                let n = result.times.len().min(10);
                                for i in 0..n {
                                    println!(
                                        "{:<12.4} {:<12.4} {:<12.4}",
                                        result.times[i], result.survival[i], result.std_errors[i]
                                    );
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    print_error(&format!("Kaplan-Meier failed: {}", e), format);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

fn execute_log_rank(
    dataset_name: &str,
    time_col: &str,
    event_col: &str,
    group_col: &str,
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
        Some(ds) => match log_rank_test(ds, time_col, event_col, group_col) {
            Ok(result) => match format {
                OutputFormat::Json => {
                    let json = serde_json::json!({
                        "method": "Log-rank test",
                        "chi_squared": result.chi_squared,
                        "degrees_of_freedom": result.df,
                        "p_value": result.p_value,
                        "groups": result.groups,
                        "n_per_group": result.n_per_group,
                        "events_per_group": result.events_per_group,
                    });
                    println!("{}", serde_json::to_string_pretty(&json)?);
                }
                _ => {
                    println!("\nLog-rank Test Results");
                    println!("{}", "=".repeat(50));
                    println!("Chi-squared statistic: {:.4}", result.chi_squared);
                    println!("Degrees of freedom: {}", result.df);
                    println!("p-value: {:.6}", result.p_value);
                    println!("Number of groups: {}", result.groups.len());
                    if result.p_value < 0.05 {
                        println!(
                            "\nConclusion: Significant difference in survival curves (p < 0.05)"
                        );
                    } else {
                        println!("\nConclusion: No significant difference in survival curves");
                    }
                }
            },
            Err(e) => {
                print_error(&format!("Log-rank test failed: {}", e), format);
            }
        },
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

fn execute_cox(
    dataset_name: &str,
    time_col: &str,
    event_col: &str,
    covariates: &[String],
    ties: &str,
    robust: bool,
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
            let cov_refs: Vec<&str> = covariates.iter().map(|s| s.as_str()).collect();

            let ties_method = match ties.to_lowercase().as_str() {
                "efron" => TiesMethod::Efron,
                _ => TiesMethod::Breslow,
            };

            let config = CoxConfig {
                ties: ties_method,
                max_iter: 25,
                tolerance: 1e-9,
                robust_se: robust,
            };

            match run_cox_ph(ds, time_col, event_col, &cov_refs, Some(config)) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": "Cox Proportional Hazards",
                            "variables": result.variables,
                            "coefficients": result.coefficients,
                            "std_errors": result.std_errors,
                            "hazard_ratios": result.hazard_ratios,
                            "z_stats": result.z_stats,
                            "p_values": result.p_values,
                            "log_likelihood": result.log_likelihood,
                            "concordance": result.concordance,
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nCox Proportional Hazards Model");
                        println!("{}", "=".repeat(60));
                        println!("Log-likelihood: {:.4}", result.log_likelihood);
                        println!("Concordance: {:.4}", result.concordance);
                        println!(
                            "\n{:<15} {:<10} {:<10} {:<10} {:<10} {:<10}",
                            "Variable", "Coef", "HR", "Std.Err", "z", "p-value"
                        );
                        println!("{}", "-".repeat(60));
                        for i in 0..result.variables.len() {
                            println!(
                                "{:<15} {:<10.4} {:<10.4} {:<10.4} {:<10.4} {:<10.4}",
                                result.variables[i],
                                result.coefficients[i],
                                result.hazard_ratios[i],
                                result.std_errors[i],
                                result.z_stats[i],
                                result.p_values[i]
                            );
                        }
                    }
                },
                Err(e) => {
                    print_error(&format!("Cox PH failed: {}", e), format);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

fn execute_aft(
    dataset_name: &str,
    time_col: &str,
    event_col: &str,
    covariates: &[String],
    dist: &str,
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
            let cov_refs: Vec<&str> = covariates.iter().map(|s| s.as_str()).collect();

            let distribution = match dist.to_lowercase().as_str() {
                "exponential" => AftDistribution::Exponential,
                "lognormal" => AftDistribution::LogNormal,
                "loglogistic" => AftDistribution::LogLogistic,
                _ => AftDistribution::Weibull,
            };

            let config = AftConfig {
                distribution,
                max_iter: 100,
                tolerance: 1e-9,
            };

            match run_aft(ds, time_col, event_col, &cov_refs, Some(config)) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": "Accelerated Failure Time",
                            "distribution": format!("{}", result.distribution),
                            "n_obs": result.n_obs,
                            "n_events": result.n_events,
                            "variables": result.variables,
                            "coefficients": result.coefficients,
                            "std_errors": result.std_errors,
                            "z_stats": result.z_stats,
                            "p_values": result.p_values,
                            "log_likelihood": result.log_likelihood,
                            "aic": result.aic,
                            "scale": result.scale,
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nAccelerated Failure Time Model ({})", result.distribution);
                        println!("{}", "=".repeat(60));
                        println!(
                            "Observations: {}, Events: {}",
                            result.n_obs, result.n_events
                        );
                        println!("Log-likelihood: {:.4}", result.log_likelihood);
                        println!("AIC: {:.4}", result.aic);
                        println!("Scale parameter: {:.4}", result.scale);
                        println!(
                            "\n{:<15} {:<10} {:<10} {:<10} {:<10}",
                            "Variable", "Coef", "Std.Err", "z", "p-value"
                        );
                        println!("{}", "-".repeat(55));
                        for i in 0..result.variables.len() {
                            println!(
                                "{:<15} {:<10.4} {:<10.4} {:<10.4} {:<10.4}",
                                result.variables[i],
                                result.coefficients[i],
                                result.std_errors[i],
                                result.z_stats[i],
                                result.p_values[i]
                            );
                        }
                    }
                },
                Err(e) => {
                    print_error(&format!("AFT failed: {}", e), format);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

fn execute_competing_risks(
    dataset_name: &str,
    time_col: &str,
    event_type_col: &str,
    conf_level: f64,
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
        Some(ds) => match run_competing_risks(ds, time_col, event_type_col, conf_level) {
            Ok(result) => match format {
                OutputFormat::Json => {
                    let json = serde_json::json!({
                        "method": "Competing Risks Analysis",
                        "n_obs": result.n_obs,
                        "n_censored": result.n_censored,
                        "event_types": result.event_types,
                        "cumulative_incidence": result.cifs.iter().map(|ci| {
                            serde_json::json!({
                                "event_type": ci.event_type,
                                "times": ci.times,
                                "incidence": ci.incidence,
                            })
                        }).collect::<Vec<_>>(),
                    });
                    println!("{}", serde_json::to_string_pretty(&json)?);
                }
                _ => {
                    println!("\nCompeting Risks Analysis");
                    println!("{}", "=".repeat(50));
                    println!("Observations: {}", result.n_obs);
                    println!("Censored: {}", result.n_censored);
                    println!("Event types: {:?}", result.event_types);
                    for ci in &result.cifs {
                        println!("\nEvent type {} cumulative incidence:", ci.event_type);
                        println!("{:<12} {:<12}", "Time", "CIF");
                        println!("{}", "-".repeat(24));
                        let n = ci.times.len().min(10);
                        for i in 0..n {
                            println!("{:<12.4} {:<12.4}", ci.times[i], ci.incidence[i]);
                        }
                    }
                }
            },
            Err(e) => {
                print_error(&format!("Competing risks analysis failed: {}", e), format);
            }
        },
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}
