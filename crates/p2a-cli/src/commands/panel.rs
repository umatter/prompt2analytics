//! Panel data estimation commands

use clap::Subcommand;
use clap::builder::{PossibleValuesParser, TypedValueParser};
use p2a_core::regression::CovarianceType;
use p2a_core::run_arellano_bond;
use p2a_core::{
    ClusterAdjustment, PanelConfig, PanelResult, PanelVcov, TestDistribution,
    run_fixed_effects_with, run_hausman_test, run_hdfe, run_random_effects_with,
};
use p2a_core::{GlmFamily, run_feglm};
use p2a_core::{PanelGlsModel, run_panel_gls};
use p2a_core::{PanelModel, PanelUnitRootConfig, PanelUnitRootTest, run_panel_unit_root};
use p2a_core::{PvcmType, run_pvcm};

use crate::output::{
    CoefRow, OutputFormat, RegressionReport, StatKind, format_regression_report, print_error,
    se_label, sorted_object,
};
use crate::session::SessionManager;

#[derive(Subcommand)]
pub enum PanelCommands {
    /// Fixed Effects estimation
    #[command(after_help = "\
EXAMPLES:
    # Entity fixed effects
    p2a panel fe mydata -y revenue -x employees capital --entity firm_id

    # Entity-clustered (Arellano) standard errors
    p2a panel fe mydata -y revenue -x employees capital --entity firm_id --vcov cluster
")]
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

        /// Standard errors: classical (default, as plm), cluster (entity-clustered Arellano, Stata small-sample factor), cluster-hc0, cluster-hc1
        #[arg(long, default_value = "classical", value_parser = panel_vcov_parser())]
        vcov: PanelVcov,
    },

    /// Random Effects estimation
    #[command(after_help = "\
EXAMPLES:
    # Random effects with GLS
    p2a panel re mydata -y revenue -x employees capital --entity firm_id

    # Entity-clustered (Arellano) standard errors
    p2a panel re mydata -y revenue -x employees capital --entity firm_id --vcov cluster
")]
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

        /// Standard errors: classical (default, as plm), cluster (entity-clustered Arellano, Stata small-sample factor), cluster-hc0, cluster-hc1
        #[arg(long, default_value = "classical", value_parser = panel_vcov_parser())]
        vcov: PanelVcov,
    },

    /// Hausman test (Fixed vs Random Effects)
    #[command(after_help = "\
EXAMPLES:
    # Test FE vs RE (H0: RE consistent)
    p2a panel hausman mydata -y revenue -x employees capital --entity firm_id
")]
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
    #[command(after_help = "\
EXAMPLES:
    # Two-way fixed effects (firm + year)
    p2a panel hdfe mydata -y revenue -x employees --fe firm_id year

    # Three-way fixed effects
    p2a panel hdfe mydata -y wage -x experience --fe worker_id firm_id year
")]
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

    /// Fixed Effects GLM (logit, probit, poisson with HDFE)
    #[command(after_help = "\
EXAMPLES:
    # Logit with fixed effects
    p2a panel feglm mydata -y employed -x age education --fe industry year --family logit

    # Poisson FE for count data
    p2a panel feglm mydata -y patents -x rd_spending --fe firm_id --family poisson
")]
    Feglm {
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

        /// GLM family: "logit" (default), "probit", "poisson", "gaussian"
        #[arg(long, default_value = "logit")]
        family: String,
    },

    /// Arellano-Bond GMM estimation for dynamic panels
    #[command(after_help = "\
EXAMPLES:
    # Dynamic panel with lagged dependent variable
    p2a panel gmm mydata -y investment -x sales cash_flow --entity firm_id --time year
")]
    Gmm {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,

        /// Entity (group) column for panel structure
        #[arg(long)]
        entity: String,

        /// Time column for panel structure
        #[arg(long)]
        time: String,
    },

    /// Panel Variable Coefficients Model (Swamy estimator)
    #[command(after_help = "\
EXAMPLES:
    # Variable coefficients (within model)
    p2a panel pvcm mydata -y gdp -x investment exports --entity country --model within

    # Random coefficient model
    p2a panel pvcm mydata -y gdp -x investment exports --entity country --model random
")]
    Pvcm {
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

        /// Model type: "within" (default) or "random"
        #[arg(long, default_value = "within")]
        model: String,
    },

    /// Pooled Mean Group estimator
    #[command(after_help = "\
EXAMPLES:
    # PMG for heterogeneous panels
    p2a panel pmg mydata -y consumption -x income wealth --entity country
")]
    Pmg {
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

    /// Panel GLS (Feasible GLS)
    #[command(after_help = "\
EXAMPLES:
    # FGLS with FE
    p2a panel gls mydata -y revenue -x employees --entity firm_id --time year --model fe

    # First-difference estimation
    p2a panel gls mydata -y revenue -x employees --entity firm_id --time year --model fd
")]
    Gls {
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

        /// Time column
        #[arg(long)]
        time: String,

        /// Model type: "fe" (fixed effects), "pooling", or "fd" (first-difference)
        #[arg(long, default_value = "fe")]
        model: String,
    },

    /// Panel unit root tests (LLC, IPS, Hadri)
    #[command(after_help = "\
EXAMPLES:
    # Levin-Lin-Chu test
    p2a panel unit-root mydata -v gdp --unit country --time year --test llc

    # Im-Pesaran-Shin test with 2 lags
    p2a panel unit-root mydata -v gdp --unit country --time year --test ips --lags 2
")]
    UnitRoot {
        /// Dataset name
        dataset: String,

        /// Variable to test for unit root
        #[arg(short = 'v', long)]
        var: String,

        /// Unit (entity) column
        #[arg(long)]
        unit: String,

        /// Time column
        #[arg(long)]
        time: String,

        /// Test type: "llc" (Levin-Lin-Chu), "ips" (Im-Pesaran-Shin), "hadri", or "fisher"
        #[arg(long, default_value = "llc")]
        test: String,

        /// Number of lags (optional, auto-selected if not specified)
        #[arg(long)]
        lags: Option<usize>,
    },
}

pub fn execute(
    cmd: &PanelCommands,
    format: &OutputFormat,
    _quiet: bool,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    match cmd {
        PanelCommands::Fe {
            dataset,
            dep_var,
            indep_vars,
            entity,
            vcov,
        } => execute_fe(dataset, dep_var, indep_vars, entity, *vcov, format, session),
        PanelCommands::Re {
            dataset,
            dep_var,
            indep_vars,
            entity,
            vcov,
        } => execute_re(dataset, dep_var, indep_vars, entity, *vcov, format, session),
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
        PanelCommands::Feglm {
            dataset,
            dep_var,
            indep_vars,
            fe,
            family,
        } => execute_feglm(dataset, dep_var, indep_vars, fe, family, format, session),
        PanelCommands::Gmm {
            dataset,
            dep_var,
            indep_vars,
            entity,
            time,
        } => execute_gmm(dataset, dep_var, indep_vars, entity, time, format, session),
        PanelCommands::Pvcm {
            dataset,
            dep_var,
            indep_vars,
            entity,
            model,
        } => execute_pvcm(dataset, dep_var, indep_vars, entity, model, format, session),
        PanelCommands::Pmg {
            dataset,
            dep_var,
            indep_vars,
            entity,
        } => execute_pmg(dataset, dep_var, indep_vars, entity, format, session),
        PanelCommands::Gls {
            dataset,
            dep_var,
            indep_vars,
            entity,
            time,
            model,
        } => execute_gls(
            dataset, dep_var, indep_vars, entity, time, model, format, session,
        ),
        PanelCommands::UnitRoot {
            dataset,
            var,
            unit,
            time,
            test,
            lags,
        } => execute_unit_root(dataset, var, unit, time, test, lags, format, session),
    }
}

fn execute_fe(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    entity: &str,
    vcov: PanelVcov,
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
            let config = PanelConfig { vcov };

            match run_fixed_effects_with(ds, dep_var, &x_cols, entity, &config) {
                Ok(result) => {
                    let se_type = result.vcov_type.to_string();
                    let se_type_json = panel_se_json(result.vcov_type, entity);
                    let mut extra_json = serde_json::Map::new();
                    extra_json.insert("n_groups".to_string(), result.n_groups.into());
                    extra_json.insert("f_statistic".to_string(), result.f_stat.into());
                    extra_json.insert("f_p_value".to_string(), result.f_p_value.into());
                    let report = RegressionReport {
                        method: "Fixed Effects (Within) Estimation",
                        coefficients: panel_rows(&result),
                        statistic: stat_kind(result.test_distribution),
                        r_squared: ("R-squared (within)", result.r_squared),
                        adj_r_squared: ("Adj. R-squared (within)", result.adj_r_squared),
                        n_obs: result.n_obs,
                        se_type: &se_type,
                        se_type_json: &se_type_json,
                        r_squared_type: "within",
                        extra_json,
                        warnings: &result.warnings,
                    };
                    println!("{}", format_regression_report(&report, format));

                    match format {
                        OutputFormat::Json => {}
                        _ => {
                            println!("\nNumber of groups: {}", result.n_groups);
                            print_panel_f_line(&result);
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
    vcov: PanelVcov,
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
            let config = PanelConfig { vcov };

            match run_random_effects_with(ds, dep_var, &x_cols, entity, &config) {
                Ok(result) => {
                    let se_type = result.vcov_type.to_string();
                    let se_type_json = panel_se_json(result.vcov_type, entity);
                    let theta_range = theta_range(&result);
                    let mut extra_json = serde_json::Map::new();
                    extra_json.insert("n_groups".to_string(), result.n_groups.into());
                    extra_json.insert("f_statistic".to_string(), result.f_stat.into());
                    extra_json.insert("f_p_value".to_string(), result.f_p_value.into());
                    if let Some(df) = result.wald_df {
                        extra_json.insert("wald_df".to_string(), df.into());
                    }
                    let optional = [
                        ("wald_chi2", result.wald_chi2),
                        ("wald_p_value", result.wald_p_value),
                        ("sigma_u", result.sigma_u),
                        ("sigma_e", result.sigma_e),
                        ("theta", result.theta),
                        ("theta_min", theta_range.map(|(lo, _)| lo)),
                        ("theta_max", theta_range.map(|(_, hi)| hi)),
                    ];
                    for (key, value) in optional {
                        if let Some(value) = value {
                            extra_json.insert(key.to_string(), value.into());
                        }
                    }
                    let report = RegressionReport {
                        method: "Random Effects (GLS) Estimation",
                        coefficients: panel_rows(&result),
                        statistic: stat_kind(result.test_distribution),
                        r_squared: ("R-squared", result.r_squared),
                        adj_r_squared: ("Adj. R-squared", result.adj_r_squared),
                        n_obs: result.n_obs,
                        se_type: &se_type,
                        se_type_json: &se_type_json,
                        r_squared_type: "cor(y*, X*b)^2 (plm)",
                        extra_json,
                        warnings: &result.warnings,
                    };
                    println!("{}", format_regression_report(&report, format));

                    match format {
                        OutputFormat::Json => {}
                        _ => {
                            println!("\nNumber of groups: {}", result.n_groups);
                            if let (Some(w), Some(df), Some(p)) =
                                (result.wald_chi2, result.wald_df, result.wald_p_value)
                            {
                                let label = match result.vcov_type {
                                    PanelVcov::Classical => "Wald chi-squared",
                                    PanelVcov::Arellano(_) => "Wald chi-squared (clustered)",
                                };
                                println!("{label}: {w:.4} on {df} DF (p-value: {p:.4})");
                            } else {
                                print_panel_f_line(&result);
                            }
                            if let (Some(u), Some(e), Some(theta)) =
                                (result.sigma_u, result.sigma_e, result.theta)
                            {
                                match theta_range {
                                    None => println!(
                                        "sigma_u: {u:.4}, sigma_e: {e:.4}, theta: {theta:.4}"
                                    ),
                                    Some((lo, hi)) => println!(
                                        "sigma_u: {u:.4}, sigma_e: {e:.4}, theta (mean): \
                                         {theta:.4}, range {lo:.4}-{hi:.4}"
                                    ),
                                }
                            }
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
                Ok(result) => match format {
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
                },
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
                    let method = format!("High-Dimensional FE ({})", fe_cols.join(", "));
                    let se_type = se_label(result.cov_type);
                    let fe_dimensions: Vec<serde_json::Value> = result
                        .fe_dimensions
                        .iter()
                        .zip(&result.fe_counts)
                        .map(|(name, levels)| {
                            sorted_object([
                                ("name".to_string(), name.clone().into()),
                                ("levels".to_string(), (*levels).into()),
                            ])
                        })
                        .collect();
                    let mut extra_json = serde_json::Map::new();
                    // n_groups: the levels of the first --fe column.
                    extra_json.insert(
                        "n_groups".to_string(),
                        result.fe_counts.first().copied().unwrap_or(0).into(),
                    );
                    extra_json.insert("f_statistic".to_string(), result.f_stat.into());
                    extra_json.insert("f_p_value".to_string(), result.f_p_value.into());
                    extra_json.insert("fe_dimensions".to_string(), fe_dimensions.into());
                    let report = RegressionReport {
                        method: &method,
                        coefficients: CoefRow::from_slices(
                            &result.variables,
                            &result.coefficients,
                            &result.std_errors,
                            &result.t_stats,
                            &result.p_values,
                        ),
                        statistic: StatKind::T,
                        r_squared: ("R-squared (within)", result.r_squared_within),
                        adj_r_squared: ("Adj. R-squared (within)", result.adj_r_squared_within),
                        n_obs: result.n_obs,
                        se_type: &se_type,
                        se_type_json: result.cov_type.short_label(),
                        r_squared_type: "within",
                        extra_json,
                        warnings: &[],
                    };
                    println!("{}", format_regression_report(&report, format));

                    match format {
                        OutputFormat::Json => {}
                        _ => {
                            println!("\nFixed effect dimensions:");
                            for (dim, count) in
                                result.fe_dimensions.iter().zip(result.fe_counts.iter())
                            {
                                println!("  {}: {} levels", dim, count);
                            }
                            println!(
                                "\nF-statistic: {:.4} (p-value: {:.4})",
                                result.f_stat, result.f_p_value
                            );
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

/// `--vcov` values, parsed into [`PanelVcov`]; clap lists them in its error.
fn panel_vcov_parser() -> impl TypedValueParser<Value = PanelVcov> {
    PossibleValuesParser::new(["classical", "cluster", "cluster-hc0", "cluster-hc1"])
        .try_map(|s: String| s.parse::<PanelVcov>().map_err(|e| e.to_string()))
}

/// JSON `se_type` of a panel covariance: `classical`, `cluster:<entity>`
/// (Stata factor), `cluster-hc0:<entity>` or `cluster-hc1:<entity>`.
fn panel_se_json(vcov: PanelVcov, entity: &str) -> String {
    match vcov {
        PanelVcov::Classical => "classical".to_string(),
        PanelVcov::Arellano(ClusterAdjustment::Stata) => format!("cluster:{entity}"),
        PanelVcov::Arellano(ClusterAdjustment::HC0) => format!("cluster-hc0:{entity}"),
        PanelVcov::Arellano(ClusterAdjustment::HC1) => format!("cluster-hc1:{entity}"),
    }
}

/// t for Student t inference, z for normal inference (random effects).
fn stat_kind(dist: TestDistribution) -> StatKind {
    match dist {
        TestDistribution::StudentT => StatKind::T,
        TestDistribution::Normal => StatKind::Z,
    }
}

/// The coefficient rows of a panel result.
fn panel_rows(result: &PanelResult) -> Vec<CoefRow> {
    CoefRow::from_slices(
        &result.variables,
        &result.coefficients,
        &result.std_errors,
        &result.t_stats,
        &result.p_values,
    )
}

/// Smallest and largest θ_i of an unbalanced random-effects fit; `None` when
/// the panel is balanced (one θ).
fn theta_range(result: &PanelResult) -> Option<(f64, f64)> {
    if result.theta_by_group.is_empty() {
        return None;
    }
    let g = &result.theta_by_group;
    Some((
        g.iter().copied().fold(f64::INFINITY, f64::min),
        g.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    ))
}

/// The F line after a panel table. Classical: `F-statistic: F (p-value: p)`.
/// Under the entity-clustered covariance the statistic is a Wald F on
/// (K, G - 1) degrees of freedom, as plm, and the line says so. A NaN F (a
/// singular clustered covariance) prints nothing.
fn print_panel_f_line(result: &PanelResult) {
    if result.f_stat.is_nan() {
        return;
    }
    match result.vcov_type {
        PanelVcov::Classical => println!(
            "F-statistic: {:.4} (p-value: {:.4})",
            result.f_stat, result.f_p_value
        ),
        PanelVcov::Arellano(_) => {
            let (df1, df2) = result.f_df();
            println!(
                "F-statistic (Wald, clustered): {:.4} on {df1} and {df2} DF (p-value: {:.4})",
                result.f_stat, result.f_p_value
            );
        }
    }
}

fn execute_feglm(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    fe_cols: &[String],
    family_str: &str,
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
            let fe_refs: Vec<&str> = fe_cols.iter().map(|s| s.as_str()).collect();

            let family = match family_str.to_lowercase().as_str() {
                "probit" => GlmFamily::Probit,
                "poisson" => GlmFamily::Poisson,
                "gaussian" => GlmFamily::Gaussian,
                _ => GlmFamily::Logit,
            };

            match run_feglm(ds, dep_var, &x_cols, &fe_refs, family, None) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": format!("FEGLM ({})", result.family),
                            "family": format!("{:?}", result.family),
                            "dep_var": result.dep_var,
                            "variables": result.variables,
                            "coefficients": result.coefficients,
                            "std_errors": result.std_errors,
                            "z_stats": result.z_stats,
                            "p_values": result.p_values,
                            "n_obs": result.n_obs,
                            "log_likelihood": result.log_likelihood,
                            "deviance": result.deviance,
                            "fe_dimensions": result.fe_dimensions,
                            "fe_counts": result.fe_counts,
                            "converged": result.converged,
                            "iterations": result.iterations,
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nFEGLM: {} with Fixed Effects", result.family);
                        println!("{}", "=".repeat(60));

                        println!("\nCoefficients:");
                        println!(
                            "{:<20} {:>12} {:>12} {:>10} {:>10}",
                            "Variable", "Coef", "Std Err", "z", "P>|z|"
                        );
                        println!("{}", "-".repeat(66));

                        for (i, var) in result.variables.iter().enumerate() {
                            let sig = if result.p_values[i] < 0.001 {
                                "***"
                            } else if result.p_values[i] < 0.01 {
                                "**"
                            } else if result.p_values[i] < 0.05 {
                                "*"
                            } else if result.p_values[i] < 0.1 {
                                "."
                            } else {
                                ""
                            };
                            println!(
                                "{:<20} {:>12.6} {:>12.6} {:>10.4} {:>9.4} {}",
                                var,
                                result.coefficients[i],
                                result.std_errors[i],
                                result.z_stats[i],
                                result.p_values[i],
                                sig
                            );
                        }

                        println!("\n---");
                        println!("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1");

                        println!("\nModel Info:");
                        println!("  Observations: {}", result.n_obs);
                        println!("  Log-Likelihood: {:.4}", result.log_likelihood);
                        println!("  Deviance: {:.4}", result.deviance);
                        println!(
                            "  Converged: {} ({} iterations)",
                            result.converged, result.iterations
                        );

                        println!("\nFixed Effects:");
                        for (dim, count) in result.fe_dimensions.iter().zip(&result.fe_counts) {
                            println!("  {}: {} levels", dim, count);
                        }
                    }
                },
                Err(e) => print_error(&format!("FEGLM failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_gmm(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    entity: &str,
    time: &str,
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

            match run_arellano_bond(ds, dep_var, &x_cols, entity, time) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": "Arellano-Bond GMM",
                            "transform": format!("{}", result.transform),
                            "step": format!("{}", result.step),
                            "dep_var": result.dep_var,
                            "variables": result.variables,
                            "coefficients": result.coefficients,
                            "std_errors": result.std_errors,
                            "z_stats": result.z_stats,
                            "p_values": result.p_values,
                            "n_obs": result.n_obs,
                            "n_groups": result.n_groups,
                            "n_instruments": result.n_instruments,
                            "sargan_statistic": result.sargan_statistic,
                            "sargan_df": result.sargan_df,
                            "sargan_p_value": result.sargan_p_value,
                            "ar1_statistic": result.ar1_statistic,
                            "ar1_p_value": result.ar1_p_value,
                            "ar2_statistic": result.ar2_statistic,
                            "ar2_p_value": result.ar2_p_value,
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nArellano-Bond GMM Estimation");
                        println!("{}", "=".repeat(70));
                        println!("Transform: {}  Step: {}", result.transform, result.step);
                        println!(
                            "Observations: {}  Groups: {}  Instruments: {}",
                            result.n_obs, result.n_groups, result.n_instruments
                        );

                        println!("\nCoefficients:");
                        println!(
                            "{:<20} {:>12} {:>12} {:>10} {:>10}",
                            "Variable", "Coef", "Std Err", "z", "P>|z|"
                        );
                        println!("{}", "-".repeat(66));

                        for (i, var) in result.variables.iter().enumerate() {
                            let sig = if result.p_values[i] < 0.001 {
                                "***"
                            } else if result.p_values[i] < 0.01 {
                                "**"
                            } else if result.p_values[i] < 0.05 {
                                "*"
                            } else if result.p_values[i] < 0.1 {
                                "."
                            } else {
                                ""
                            };
                            println!(
                                "{:<20} {:>12.6} {:>12.6} {:>10.4} {:>9.4} {}",
                                var,
                                result.coefficients[i],
                                result.std_errors[i],
                                result.z_stats[i],
                                result.p_values[i],
                                sig
                            );
                        }

                        println!("\n---");
                        println!("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1");

                        println!("\nDiagnostic Tests:");
                        println!(
                            "  Sargan test: chi2({}) = {:.4}, p = {:.4}",
                            result.sargan_df, result.sargan_statistic, result.sargan_p_value
                        );
                        println!(
                            "  AR(1): z = {:.4}, p = {:.4}",
                            result.ar1_statistic, result.ar1_p_value
                        );
                        println!(
                            "  AR(2): z = {:.4}, p = {:.4}",
                            result.ar2_statistic, result.ar2_p_value
                        );
                    }
                },
                Err(e) => print_error(&format!("GMM estimation failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_pvcm(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    entity: &str,
    model_str: &str,
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

            let model = match model_str.to_lowercase().as_str() {
                "random" => PvcmType::Random,
                _ => PvcmType::Within,
            };

            match run_pvcm(ds, dep_var, &x_cols, entity, model) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": "PVCM",
                            "model_type": format!("{:?}", result.model_type),
                            "dep_var": result.dep_var,
                            "variables": result.variables,
                            "coefficients": result.coefficients,
                            "std_errors": result.std_errors,
                            "t_stats": result.t_stats,
                            "p_values": result.p_values,
                            "n_obs": result.n_obs,
                            "n_entities": result.n_entities,
                            "homogeneity_stat": result.homogeneity_stat,
                            "homogeneity_pvalue": result.homogeneity_pvalue,
                            "individual_coefficients": result.individual_coefficients,
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nVariable Coefficients Model (PVCM)");
                        println!("{}", "=".repeat(70));
                        println!(
                            "Model: {:?}  Observations: {}  Entities: {}",
                            result.model_type, result.n_obs, result.n_entities
                        );

                        println!("\nOverall Coefficients:");
                        println!(
                            "{:<20} {:>12} {:>12} {:>10} {:>10}",
                            "Variable", "Coef", "Std Err", "t", "P>|t|"
                        );
                        println!("{}", "-".repeat(66));

                        for (i, var) in result.variables.iter().enumerate() {
                            let sig = if result.p_values[i] < 0.001 {
                                "***"
                            } else if result.p_values[i] < 0.01 {
                                "**"
                            } else if result.p_values[i] < 0.05 {
                                "*"
                            } else if result.p_values[i] < 0.1 {
                                "."
                            } else {
                                ""
                            };
                            println!(
                                "{:<20} {:>12.6} {:>12.6} {:>10.4} {:>9.4} {}",
                                var,
                                result.coefficients[i],
                                result.std_errors[i],
                                result.t_stats[i],
                                result.p_values[i],
                                sig
                            );
                        }

                        println!("\n---");
                        println!("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1");

                        println!("\nHomogeneity Test:");
                        println!(
                            "  Chi-squared = {:.4}, p = {:.4}",
                            result.homogeneity_stat, result.homogeneity_pvalue
                        );
                    }
                },
                Err(e) => print_error(&format!("PVCM estimation failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_pmg(
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
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            let x_cols: Vec<&str> = indep_vars.iter().map(|s| s.as_str()).collect();

            match p2a_core::run_pmg(ds, dep_var, &x_cols, entity) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": "PMG (Pooled Mean Group)",
                            "dep_var": result.dep_var,
                            "variables": result.variables,
                            "coefficients": result.coefficients,
                            "std_errors": result.std_errors,
                            "t_stats": result.t_stats,
                            "p_values": result.p_values,
                            "n_obs": result.n_obs,
                            "n_entities": result.n_entities,
                            "individual_coefficients": result.individual_coefficients,
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nPooled Mean Group Estimator (PMG)");
                        println!("{}", "=".repeat(70));
                        println!(
                            "Observations: {}  Entities: {}",
                            result.n_obs, result.n_entities
                        );

                        println!("\nMean Group Coefficients:");
                        println!(
                            "{:<20} {:>12} {:>12} {:>10} {:>10}",
                            "Variable", "Coef", "Std Err", "t", "P>|t|"
                        );
                        println!("{}", "-".repeat(66));

                        for (i, var) in result.variables.iter().enumerate() {
                            let sig = if result.p_values[i] < 0.001 {
                                "***"
                            } else if result.p_values[i] < 0.01 {
                                "**"
                            } else if result.p_values[i] < 0.05 {
                                "*"
                            } else if result.p_values[i] < 0.1 {
                                "."
                            } else {
                                ""
                            };
                            println!(
                                "{:<20} {:>12.6} {:>12.6} {:>10.4} {:>9.4} {}",
                                var,
                                result.coefficients[i],
                                result.std_errors[i],
                                result.t_stats[i],
                                result.p_values[i],
                                sig
                            );
                        }

                        println!("\n---");
                        println!("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1");
                    }
                },
                Err(e) => print_error(&format!("PMG estimation failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_gls(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    entity: &str,
    time: &str,
    model_str: &str,
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

            let model = match model_str.to_lowercase().as_str() {
                "pooling" => Some(PanelGlsModel::Pooling),
                "fd" | "first-difference" => Some(PanelGlsModel::FirstDifference),
                _ => Some(PanelGlsModel::FixedEffects),
            };

            match run_panel_gls(ds, dep_var, &x_cols, entity, time, model) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": "Panel GLS",
                            "model": format!("{}", result.model),
                            "dep_var": result.dep_var,
                            "variables": result.variables,
                            "coefficients": result.coefficients,
                            "std_errors": result.std_errors,
                            "t_stats": result.t_stats,
                            "p_values": result.p_values,
                            "r_squared": result.r_squared,
                            "n_obs": result.n_obs,
                            "n_groups": result.n_groups,
                            "n_periods": result.n_periods,
                            "sigma": result.sigma,
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nPanel GLS (FGLS) Estimation");
                        println!("{}", "=".repeat(70));
                        println!("Model: {}", result.model);
                        println!(
                            "Observations: {}  Groups: {}  Time periods: {}",
                            result.n_obs, result.n_groups, result.n_periods
                        );

                        println!("\nCoefficients:");
                        println!(
                            "{:<20} {:>12} {:>12} {:>10} {:>10}",
                            "Variable", "Coef", "Std Err", "t", "P>|t|"
                        );
                        println!("{}", "-".repeat(66));

                        for (i, var) in result.variables.iter().enumerate() {
                            let sig = if result.p_values[i] < 0.001 {
                                "***"
                            } else if result.p_values[i] < 0.01 {
                                "**"
                            } else if result.p_values[i] < 0.05 {
                                "*"
                            } else if result.p_values[i] < 0.1 {
                                "."
                            } else {
                                ""
                            };
                            println!(
                                "{:<20} {:>12.6} {:>12.6} {:>10.4} {:>9.4} {}",
                                var,
                                result.coefficients[i],
                                result.std_errors[i],
                                result.t_stats[i],
                                result.p_values[i],
                                sig
                            );
                        }

                        println!("\n---");
                        println!("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1");

                        println!("\nModel Fit:");
                        println!("  R-squared: {:.4}", result.r_squared);
                        println!("  Residual std. error: {:.4}", result.sigma);
                    }
                },
                Err(e) => print_error(&format!("Panel GLS failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_unit_root(
    dataset_name: &str,
    var: &str,
    unit: &str,
    time: &str,
    test_str: &str,
    lags: &Option<usize>,
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
            let test_type = match test_str.to_lowercase().as_str() {
                "ips" => PanelUnitRootTest::IPS,
                "hadri" => PanelUnitRootTest::Hadri,
                "fisher" => PanelUnitRootTest::Fisher,
                _ => PanelUnitRootTest::LLC,
            };

            let config = PanelUnitRootConfig {
                test_type,
                model: PanelModel::Intercept,
                lags: *lags,
                max_lags: None,
            };

            match run_panel_unit_root(ds, var, unit, time, config) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "test_type": format!("{}", result.test_type),
                            "model": format!("{}", result.model),
                            "statistic": result.statistic,
                            "p_value": result.p_value,
                            "null_hypothesis": result.null_hypothesis,
                            "alternative_hypothesis": result.alternative_hypothesis,
                            "n_panels": result.n_panels,
                            "avg_time_periods": result.avg_time_periods,
                            "lags_used": result.lags_used,
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nPanel Unit Root Test: {}", result.test_type);
                        println!("{}", "=".repeat(70));
                        println!("Model: {}", result.model);
                        println!(
                            "Panels: {}  Avg. time periods: {:.1}  Lags: {}",
                            result.n_panels, result.avg_time_periods, result.lags_used
                        );

                        println!("\nH0: {}", result.null_hypothesis);
                        println!("H1: {}", result.alternative_hypothesis);

                        println!("\nTest statistic: {:.4}", result.statistic);
                        println!("P-value: {:.4}", result.p_value);

                        let sig = if result.p_value < 0.01 {
                            "*** Reject H0 at 1% level"
                        } else if result.p_value < 0.05 {
                            "** Reject H0 at 5% level"
                        } else if result.p_value < 0.10 {
                            "* Reject H0 at 10% level"
                        } else {
                            "Fail to reject H0"
                        };
                        println!("\nConclusion: {}", sig);
                    }
                },
                Err(e) => print_error(&format!("Panel unit root test failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}
