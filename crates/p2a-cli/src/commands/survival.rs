//! Survival analysis commands

use clap::Subcommand;
use p2a_core::econometrics::{KmConfType, KmConfig, run_kaplan_meier_with};
use p2a_core::{
    AftConfig, AftDistribution, CoxConfig, CoxSeType, TiesMethod, log_rank_test, run_aft,
    run_competing_risks, run_cox_ph,
};

use crate::output::{
    OutputFormat, add_notes, add_result_notes, missing_json, parse_core_option, print_error,
    print_missing_note, print_result_notes,
};
use crate::session::SessionManager;

/// "95" for 0.95, "97.5" for 0.975.
fn percent(conf_level: f64) -> String {
    let pct = conf_level * 100.0;
    if (pct - pct.round()).abs() < 1e-9 {
        format!("{pct:.0}")
    } else {
        format!("{pct}")
    }
}

/// `{:.4}`, or "NA" for NaN or `None` (R's NA).
fn na4(v: Option<f64>) -> String {
    match v {
        Some(x) if !x.is_nan() => format!("{x:.4}"),
        _ => "NA".to_string(),
    }
}

/// How the competing-risks standard errors are computed.
const CR_SE_METHOD: &str = "delta method (as survfit)";

/// Adds `"missing"` to a JSON object when rows were dropped.
fn with_missing(
    mut json: serde_json::Value,
    missing: &p2a_core::data::columns::MissingSummary,
) -> serde_json::Value {
    if let Some(m) = missing_json(missing) {
        json["missing"] = m;
    }
    json
}

#[derive(Subcommand)]
pub enum SurvivalCommands {
    /// Kaplan-Meier survival curve estimation
    #[command(after_help = "\
EXAMPLES:
    # Basic survival curve
    p2a --session s.json survival km mydata -t time -e status

    # Stratified by treatment group
    p2a --session s.json survival km mydata -t time -e status -g treatment

    # Log-log confidence intervals (R survfit conf.type log-log)
    p2a --session s.json survival km mydata -t time -e status --conf-type log-log
")]
    Km {
        /// Dataset name
        dataset: String,

        /// Time variable column (duration/time to event)
        #[arg(short = 't', long)]
        time: String,

        /// Status column, coded as R's Surv(): 0/1 (1 = event), 1/2 (2 = event) or
        /// TRUE/FALSE
        #[arg(short = 'e', long)]
        event: String,

        /// Optional grouping variable for stratified analysis
        #[arg(short = 'g', long)]
        group: Option<String>,

        /// Confidence level (default 0.95)
        #[arg(long, default_value = "0.95")]
        conf: f64,

        /// Confidence interval type, as R survfit's conf.type: log (default), log-log, plain
        #[arg(long = "conf-type", default_value = "log")]
        conf_type: String,
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

        /// Status column, coded as R's Surv(): 0/1 (1 = event), 1/2 (2 = event) or
        /// TRUE/FALSE
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

        /// Status column, coded as R's Surv(): 0/1 (1 = event), 1/2 (2 = event) or
        /// TRUE/FALSE
        #[arg(short = 'e', long)]
        event: String,

        /// Covariate columns
        #[arg(short = 'x', long, num_args = 1..)]
        covariates: Vec<String>,

        /// Ties handling method, as coxph's ties: "efron" (default) or "breslow"
        #[arg(long, default_value_t = TiesMethod::default(), value_parser = parse_core_option::<TiesMethod>)]
        ties: TiesMethod,

        /// Use robust (Lin-Wei sandwich) standard errors, as coxph(robust = TRUE)
        #[arg(long)]
        robust: bool,

        /// Cluster column for clustered sandwich standard errors, as coxph's cluster()
        /// (implies --robust)
        #[arg(long)]
        cluster: Option<String>,
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

        /// Status column, coded as R's Surv(): 0/1 (1 = event), 1/2 (2 = event) or
        /// TRUE/FALSE
        #[arg(short = 'e', long)]
        event: String,

        /// Covariate columns
        #[arg(short = 'x', long, num_args = 1..)]
        covariates: Vec<String>,

        /// Distribution, as survreg's dist: "weibull" (default), "exponential", "lognormal",
        /// "loglogistic"
        #[arg(long, default_value_t = AftDistribution::default(), value_parser = parse_core_option::<AftDistribution>)]
        dist: AftDistribution,
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
    _quiet: bool,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    match cmd {
        SurvivalCommands::Km {
            dataset,
            time,
            event,
            group,
            conf,
            conf_type,
        } => execute_km(
            dataset,
            time,
            event,
            group.as_deref(),
            *conf,
            conf_type,
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
            cluster,
        } => execute_cox(
            dataset,
            time,
            event,
            covariates,
            *ties,
            *robust,
            cluster.as_deref(),
            format,
            session,
        ),
        SurvivalCommands::Aft {
            dataset,
            time,
            event,
            covariates,
            dist,
        } => execute_aft(dataset, time, event, covariates, *dist, format, session),
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
    conf_type: &str,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let conf_type = match conf_type.parse::<KmConfType>() {
        Ok(t) => t,
        Err(e) => {
            print_error(&format!("Kaplan-Meier failed: {e}"), format);
            return Ok(());
        }
    };
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

    let Some(ds) = dataset else {
        print_error(&format!("Dataset '{}' not found", dataset_name), format);
        return Ok(());
    };
    let mut config = KmConfig::default();
    config.conf_level = conf_level;
    config.conf_type = conf_type;
    let results = match run_kaplan_meier_with(ds, time_col, event_col, group_col, &config) {
        Ok(results) => results,
        Err(e) => {
            print_error(&format!("Kaplan-Meier failed: {}", e), format);
            return Ok(());
        }
    };

    // One document or block per group (CLI-38: still one JSON document per group)
    for result in &results {
        match format {
            OutputFormat::Json => {
                let json = with_missing(
                    serde_json::json!({
                        "method": "Kaplan-Meier",
                        "group": result.group,
                        "n_obs": result.n_obs,
                        "total_events": result.total_events,
                        "total_censored": result.total_censored,
                        "conf_level": result.conf_level,
                        "conf_type": result.conf_type.to_string(),
                        "median_survival": result.median_survival,
                        "median_ci": [result.median_ci.0, result.median_ci.1],
                        "times": result.times,
                        "n_at_risk": result.n_at_risk,
                        "n_events": result.n_events,
                        "n_censored": result.n_censored,
                        "survival": result.survival,
                        "std_errors": result.std_errors,
                        "ci_lower": result.ci_lower,
                        "ci_upper": result.ci_upper,
                    }),
                    &result.missing,
                );
                let mut json = json;
                add_notes(&mut json, None, &[]);
                println!("{}", serde_json::to_string_pretty(&json)?);
            }
            _ => {
                let pct = percent(result.conf_level);
                println!("\nKaplan-Meier Survival Estimates");
                println!("{}", "=".repeat(50));
                if let Some(ref group) = result.group {
                    println!("Group: {}", group);
                }
                println!("Observations: {}", result.n_obs);
                println!("Events: {}", result.total_events);
                println!("Censored: {}", result.total_censored);
                println!(
                    "Confidence intervals: {pct}%, conf.type = {}",
                    result.conf_type
                );
                let limits = format!(
                    "({pct}% CI: {} - {})",
                    na4(result.median_ci.0),
                    na4(result.median_ci.1)
                );
                match result.median_survival {
                    Some(median) => println!("Median survival time: {median:.4} {limits}"),
                    None => println!("Median survival time: not reached {limits}"),
                }
                println!("\nSurvival table (first 10 time points):");
                println!(
                    "{:<12} {:<12} {:<12} {:<12} {:<12}",
                    "Time", "Survival", "Std.Error", "Lower", "Upper"
                );
                println!("{}", "-".repeat(60));
                let n = result.times.len().min(10);
                for i in 0..n {
                    println!(
                        "{:<12.4} {:<12.4} {:<12} {:<12} {:<12}",
                        result.times[i],
                        result.survival[i],
                        na4(Some(result.std_errors[i])),
                        na4(Some(result.ci_lower[i])),
                        na4(Some(result.ci_upper[i]))
                    );
                }
            }
        }
    }
    // The rows dropped from the whole call, once after the last group.
    if let Some(first) = results.first() {
        print_missing_note(&first.missing, format);
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
                    let json = with_missing(
                        serde_json::json!({
                            "method": "Log-rank test",
                            "chi_squared": result.chi_squared,
                            "degrees_of_freedom": result.df,
                            "p_value": result.p_value,
                            "groups": result.groups,
                            "n_per_group": result.n_per_group,
                            "events_per_group": result.events_per_group,
                            "expected_per_group": result.expected_per_group,
                            "obs_minus_exp": result.obs_minus_exp,
                            "variance": result.variance,
                        }),
                        &result.missing,
                    );
                    let mut json = json;
                    add_notes(&mut json, None, &[]);
                    println!("{}", serde_json::to_string_pretty(&json)?);
                }
                _ => {
                    println!("\nLog-rank Test Results");
                    println!("{}", "=".repeat(50));
                    println!("Chi-squared statistic: {:.4}", result.chi_squared);
                    println!("Degrees of freedom: {}", result.df);
                    println!(
                        "\n{:<15} {:>8} {:>8} {:>10}",
                        "Group", "N", "Events", "Expected"
                    );
                    for (i, group) in result.groups.iter().enumerate() {
                        println!(
                            "{:<15} {:>8} {:>8} {:>10.2}",
                            group,
                            result.n_per_group[i],
                            result.events_per_group[i],
                            result.expected_per_group[i]
                        );
                    }
                    let excluded: Vec<&str> = result
                        .groups
                        .iter()
                        .zip(&result.expected_per_group)
                        .filter(|(_, e)| **e <= 0.0)
                        .map(|(g, _)| g.as_str())
                        .collect();
                    if !excluded.is_empty() {
                        println!(
                            "Groups with no expected events: {} (excluded from the test, as R survdiff)",
                            excluded.join(", ")
                        );
                    }
                    println!();
                    println!("p-value: {:.6}", result.p_value);
                    println!("Number of groups: {}", result.groups.len());
                    if result.p_value < 0.05 {
                        println!(
                            "\nConclusion: Significant difference in survival curves (p < 0.05)"
                        );
                    } else {
                        println!("\nConclusion: No significant difference in survival curves");
                    }
                    print_missing_note(&result.missing, format);
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
    ties: TiesMethod,
    robust: bool,
    cluster: Option<&str>,
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

    let Some(ds) = dataset else {
        print_error(&format!("Dataset '{}' not found", dataset_name), format);
        return Ok(());
    };
    let cov_refs: Vec<&str> = covariates.iter().map(|s| s.as_str()).collect();
    let mut config = CoxConfig::default();
    config.ties = ties;
    config.robust_se = robust;
    config.cluster = cluster.map(str::to_string);

    let result = match run_cox_ph(ds, time_col, event_col, &cov_refs, Some(config)) {
        Ok(result) => result,
        Err(e) => {
            print_error(&format!("Cox PH failed: {}", e), format);
            return Ok(());
        }
    };
    let robust_se = result.se_type != CoxSeType::Model;
    match format {
        OutputFormat::Json => {
            let mut json = with_missing(
                serde_json::json!({
                    "method": "Cox Proportional Hazards",
                    "n_obs": result.n_obs,
                    "n_events": result.n_events,
                    "ties": TiesMethod::choices().canonical(result.ties_method),
                    "se_type": result.se_type,
                    "cluster": result.cluster,
                    "n_clusters": result.n_clusters,
                    "variables": result.variables,
                    "coefficients": result.coefficients,
                    "std_errors": result.std_errors,
                    "naive_std_errors": result.naive_std_errors,
                    "hazard_ratios": result.hazard_ratios,
                    "z_stats": result.z_stats,
                    "p_values": result.p_values,
                    "log_likelihood": result.log_likelihood,
                    "concordance": result.concordance,
                    "robust_score_test": result.robust_score_test,
                    "robust_score_p_value": result.robust_score_p_value,
                    "aliased": result.aliased,
                }),
                &result.missing,
            );
            add_result_notes(&mut json, &result);
            println!("{}", serde_json::to_string_pretty(&json)?);
        }
        _ => {
            println!("\nCox Proportional Hazards Model");
            println!("{}", "=".repeat(60));
            println!(
                "Observations: {}, Events: {}",
                result.n_obs, result.n_events
            );
            println!("Ties: {}", result.ties_method);
            println!("Standard errors: {}", result.se_type);
            if let (Some(c), Some(g)) = (&result.cluster, result.n_clusters) {
                println!("Clusters: {g} (levels of {c})");
            }
            println!("Log-likelihood: {:.4}", result.log_likelihood);
            println!("Concordance: {:.4}", result.concordance);
            let se_header = if robust_se { "Robust SE" } else { "Std.Err" };
            println!(
                "\n{:<15} {:<10} {:<10} {:<10} {:<10} {:<10}",
                "Variable", "Coef", "HR", se_header, "z", "p-value"
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
            if let (Some(t), Some(p)) = (result.robust_score_test, result.robust_score_p_value) {
                println!(
                    "\nRobust score test: {} on {} df, p = {}",
                    na4(Some(t)),
                    result.df(),
                    na4(Some(p))
                );
            }
            print_result_notes(&result, format);
            print_missing_note(&result.missing, format);
        }
    }
    Ok(())
}

fn execute_aft(
    dataset_name: &str,
    time_col: &str,
    event_col: &str,
    covariates: &[String],
    distribution: AftDistribution,
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

    let Some(ds) = dataset else {
        print_error(&format!("Dataset '{}' not found", dataset_name), format);
        return Ok(());
    };
    let cov_refs: Vec<&str> = covariates.iter().map(|s| s.as_str()).collect();
    let mut config = AftConfig::default();
    config.distribution = distribution;

    let result = match run_aft(ds, time_col, event_col, &cov_refs, Some(config)) {
        Ok(result) => result,
        Err(e) => {
            print_error(&format!("AFT failed: {}", e), format);
            return Ok(());
        }
    };
    match format {
        OutputFormat::Json => {
            let mut json = with_missing(
                serde_json::json!({
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
                    "bic": result.bic,
                    "scale": result.scale,
                    "log_scale_se": result.log_scale_se,
                    "scale_fixed": result.scale_fixed,
                    "aliased": result.aliased,
                }),
                &result.missing,
            );
            add_result_notes(&mut json, &result);
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
            println!("BIC: {:.4}", result.bic);
            if result.scale_fixed {
                println!("Scale fixed at 1 (exponential)");
            } else {
                println!("Scale parameter: {:.4}", result.scale);
            }
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
            if let (Some(se), Some((z, p))) = (result.log_scale_se, result.log_scale_test()) {
                println!(
                    "{:<15} {:<10.4} {:<10.4} {:<10.4} {:<10.4}",
                    "Log(scale)",
                    result.scale.ln(),
                    se,
                    z,
                    p
                );
            }
            print_result_notes(&result, format);
            print_missing_note(&result.missing, format);
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
                    let n_events_by_type: serde_json::Map<String, serde_json::Value> = result
                        .event_types
                        .iter()
                        .map(|k| {
                            let n = result.n_events_by_type.get(k).copied().unwrap_or(0);
                            (k.to_string(), serde_json::json!(n))
                        })
                        .collect();
                    let mut json = with_missing(
                        serde_json::json!({
                            "method": "Competing Risks Analysis",
                            "n_obs": result.n_obs,
                            "n_censored": result.n_censored,
                            "event_types": result.event_types,
                            "n_events_by_type": n_events_by_type,
                            "conf_level": result.conf_level,
                            "se_method": CR_SE_METHOD,
                            "cumulative_incidence": result.cifs.iter().map(|ci| {
                                serde_json::json!({
                                    "event_type": ci.event_type,
                                    "times": ci.times,
                                    "incidence": ci.incidence,
                                    "std_errors": ci.std_errors,
                                    "ci_lower": ci.ci_lower,
                                    "ci_upper": ci.ci_upper,
                                })
                            }).collect::<Vec<_>>(),
                        }),
                        &result.missing,
                    );
                    add_notes(&mut json, None, &[]);
                    println!("{}", serde_json::to_string_pretty(&json)?);
                }
                _ => {
                    println!("\nCompeting Risks Analysis");
                    println!("{}", "=".repeat(50));
                    println!("Observations: {}", result.n_obs);
                    println!("Censored: {}", result.n_censored);
                    println!("Event types: {:?}", result.event_types);
                    println!("Confidence level: {}%", percent(result.conf_level));
                    println!("Standard errors: {CR_SE_METHOD}; intervals on the log scale");
                    for ci in &result.cifs {
                        println!("\nEvent type {} cumulative incidence:", ci.event_type);
                        println!(
                            "{:<12} {:<12} {:<12} {:<12} {:<12}",
                            "Time", "CIF", "SE", "Lower", "Upper"
                        );
                        println!("{}", "-".repeat(60));
                        let n = ci.times.len().min(10);
                        for i in 0..n {
                            println!(
                                "{:<12.4} {:<12.4} {:<12} {:<12} {:<12}",
                                ci.times[i],
                                ci.incidence[i],
                                na4(Some(ci.std_errors[i])),
                                na4(Some(ci.ci_lower[i])),
                                na4(Some(ci.ci_upper[i]))
                            );
                        }
                    }
                    print_missing_note(&result.missing, format);
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
