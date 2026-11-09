//! Panel data estimation commands

use clap::Subcommand;
use clap::builder::{PossibleValuesParser, TypedValueParser};
use p2a_core::econometrics::panel_unit_root::LagSelection;
use p2a_core::regression::CovarianceType;
use p2a_core::{
    ClusterAdjustment, PanelConfig, PanelResult, PanelVcov, TestDistribution,
    run_fixed_effects_with, run_hausman_test, run_hdfe, run_random_effects_with,
};
use p2a_core::{GlmFamily, run_feglm};
use p2a_core::{GmmConfig, GmmStep, GmmTransform, run_gmm};
use p2a_core::{PanelGlsModel, run_panel_gls};
use p2a_core::{PanelModel, PanelUnitRootConfig, PanelUnitRootTest, run_panel_unit_root};
use p2a_core::{PvcmType, run_pvcm};

use crate::output::{
    CoefRow, OutputFormat, RegressionReport, StatKind, add_notes, add_result_notes,
    format_regression_report, missing_json, parse_core_option, print_error, print_missing_note,
    print_notes, print_result_notes, se_label, sorted_object,
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

        /// GLM family: logit (default; also binomial), probit, poisson, gaussian
        #[arg(long, value_enum, ignore_case = true, default_value = "logit")]
        family: FeglmFamily,
    },

    /// Arellano-Bond difference GMM for dynamic panels (as plm::pgmm)
    #[command(after_help = "\
EXAMPLES:
    # Dynamic panel with lagged dependent variable (two-step, Windmeijer SEs)
    p2a panel gmm mydata -y investment -x sales cash_flow --entity firm_id --time year

    # One-step with Arellano-Bond (1991) classical SEs
    p2a panel gmm mydata -y investment -x sales --entity firm_id --time year --step onestep --vcov classical

    # Two lags of y, instrument lags 2 to 4, collapsed instruments
    p2a panel gmm mydata -y investment -x sales --entity firm_id --time year --lags 2 --max-lag 4 --collapse
")]
    Gmm {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Strictly exogenous regressors (the lagged dependent variable is added by --lags)
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,

        /// Entity (group) column for panel structure
        #[arg(long)]
        entity: String,

        /// Time column (whole numbers; lags are steps of 1)
        #[arg(long)]
        time: String,

        /// Lags of the dependent variable
        #[arg(long, default_value_t = 1)]
        lags: usize,

        /// One-step or two-step GMM
        #[arg(long, value_enum, ignore_case = true, default_value = "twostep")]
        step: GmmStepArg,

        /// Shallowest lag of y used as instrument
        #[arg(long = "min-lag", default_value_t = 2)]
        min_lag: usize,

        /// Deepest lag of y used as instrument (default: all available)
        #[arg(long = "max-lag")]
        max_lag: Option<usize>,

        /// One instrument column per lag depth
        #[arg(long)]
        collapse: bool,

        /// robust (one-step robust, two-step Windmeijer) or classical
        #[arg(long, value_enum, ignore_case = true, default_value = "robust")]
        vcov: GmmVcovArg,
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

        /// Model type: within (default) or random (Swamy; also swamy)
        #[arg(long, value_enum, ignore_case = true, default_value = "within")]
        model: PvcmModel,
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

        /// Model type: fe (fixed effects, default), fd (first-difference) or pooling
        #[arg(long, value_enum, ignore_case = true, default_value = "fe")]
        model: PanelGlsModelArg,
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

        /// Test: llc (Levin-Lin-Chu; also levinlin), ips (Im-Pesaran-Shin), madwu
        /// (Maddala-Wu Fisher; also fisher) or hadri, as plm::purtest's test
        #[arg(long, value_parser = parse_core_option::<PanelUnitRootTest>, default_value = "llc")]
        test: PanelUnitRootTest,

        /// Deterministic terms: none, intercept (default) or trend, as purtest's exo
        #[arg(long, value_parser = parse_core_option::<PanelModel>, default_value = "intercept")]
        model: PanelModel,

        /// Lag order of every unit (default floor(T_max^(1/3)) for every unit;
        /// plm::purtest's default is SIC with pmax 10); see --lag-selection
        #[arg(long)]
        lags: Option<usize>,

        /// Lag order per unit: fixed (needs --lags), aic, sic or hall, as
        /// plm::purtest(lags = ..)
        #[arg(long, value_parser = lag_selection_arg)]
        lag_selection: Option<String>,

        /// Largest lag the criteria try (default 10, purtest's pmax); needs --lag-selection
        /// aic, sic or hall
        #[arg(long, value_name = "N")]
        pmax: Option<usize>,
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
        } => execute_feglm(
            dataset,
            dep_var,
            indep_vars,
            fe,
            family.core(),
            format,
            session,
        ),
        PanelCommands::Gmm {
            dataset,
            dep_var,
            indep_vars,
            entity,
            time,
            lags,
            step,
            min_lag,
            max_lag,
            collapse,
            vcov,
        } => {
            let mut config = GmmConfig::default();
            config.transform = GmmTransform::Difference;
            config.step = match step {
                GmmStepArg::Onestep => GmmStep::OneStep,
                GmmStepArg::Twostep => GmmStep::TwoStep,
            };
            config.max_lag = *max_lag;
            config.min_lag = *min_lag;
            config.collapse = *collapse;
            config.robust = *vcov == GmmVcovArg::Robust;
            execute_gmm(
                dataset, dep_var, indep_vars, entity, time, *lags, config, format, session,
            )
        }
        PanelCommands::Pvcm {
            dataset,
            dep_var,
            indep_vars,
            entity,
            model,
        } => execute_pvcm(
            dataset,
            dep_var,
            indep_vars,
            entity,
            model.core(),
            format,
            session,
        ),
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
            dataset,
            dep_var,
            indep_vars,
            entity,
            time,
            model.core(),
            format,
            session,
        ),
        PanelCommands::UnitRoot {
            dataset,
            var,
            unit,
            time,
            test,
            model,
            lags,
            lag_selection,
            pmax,
        } => {
            let lag_selection = match lag_selection {
                None => None,
                Some(name) => match LagSelection::from_option(name, *lags) {
                    Ok(rule) => Some(rule),
                    Err(e) => {
                        print_error(&format!("Panel unit root test failed: {e}"), format);
                        return Ok(());
                    }
                },
            };
            if pmax.is_some() && matches!(lag_selection, None | Some(LagSelection::Fixed(_))) {
                print_error(
                    "--pmax applies to --lag-selection aic, sic or hall only",
                    format,
                );
                return Ok(());
            }
            let mut config = PanelUnitRootConfig::default();
            config.test_type = *test;
            config.model = *model;
            config.lags = *lags;
            config.lag_selection = lag_selection;
            config.pmax = *pmax;
            execute_unit_root(dataset, var, unit, time, config, format, session)
        }
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
            let mut config = PanelConfig::default();
            config.vcov = vcov;

            match run_fixed_effects_with(ds, dep_var, &x_cols, entity, &config) {
                Ok(result) => {
                    let se_type = result.vcov_type.to_string();
                    let se_type_json = panel_se_json(result.vcov_type, entity);
                    let mut extra_json = serde_json::Map::new();
                    extra_json.insert("n_groups".to_string(), result.n_groups.into());
                    extra_json.insert("f_statistic".to_string(), result.f_stat.into());
                    extra_json.insert("f_p_value".to_string(), result.f_p_value.into());
                    extra_json.insert("dropped".to_string(), result.dropped.clone().into());
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
            let mut config = PanelConfig::default();
            config.vcov = vcov;

            match run_random_effects_with(ds, dep_var, &x_cols, entity, &config) {
                Ok(result) => {
                    let se_type = result.vcov_type.to_string();
                    let se_type_json = panel_se_json(result.vcov_type, entity);
                    let theta_range = theta_range(&result);
                    let mut extra_json = serde_json::Map::new();
                    extra_json.insert("n_groups".to_string(), result.n_groups.into());
                    extra_json.insert("f_statistic".to_string(), result.f_stat.into());
                    extra_json.insert("f_p_value".to_string(), result.f_p_value.into());
                    extra_json.insert("dropped".to_string(), result.dropped.clone().into());
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
                        let json = sorted_object([
                            ("test".to_string(), "Hausman".into()),
                            ("chi2_statistic".to_string(), result.chi2_statistic.into()),
                            ("p_value".to_string(), result.p_value.into()),
                            ("df".to_string(), result.df.into()),
                            (
                                "recommendation".to_string(),
                                result.recommendation.clone().into(),
                            ),
                            ("vcov_method".to_string(), result.vcov_method.clone().into()),
                            ("warnings".to_string(), result.warnings.clone().into()),
                        ]);
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
                        println!("Covariance: {}", result.vcov_method);
                        println!("\nRecommendation: {}", result.recommendation);
                        if !result.warnings.is_empty() {
                            println!("\nWarnings:");
                            for w in &result.warnings {
                                println!("  - {w}");
                            }
                        }
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
                    extra_json.insert("dropped".to_string(), result.dropped.clone().into());
                    extra_json.insert("converged".to_string(), result.converged.into());
                    extra_json.insert("iterations".to_string(), result.iterations.into());
                    if let Some(missing) = missing_json(&result.missing) {
                        extra_json.insert("missing".to_string(), missing);
                    }
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
                        warnings: &result.warnings,
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
                            print_missing_note(&result.missing, format);
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
    family: GlmFamily,
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
                            "dropped": result.dropped,
                        });
                        let mut json = with_missing(json, &result.missing);
                        add_result_notes(&mut json, &result);
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
                        print_result_notes(&result, format);
                        print_missing_note(&result.missing, format);
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
    lags: usize,
    config: GmmConfig,
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

            match run_gmm(ds, dep_var, &x_cols, entity, time, lags, Some(config)) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let mut json = serde_json::json!({
                            "method": "Arellano-Bond GMM",
                            "transform": format!("{}", result.transform),
                            "step": format!("{}", result.step),
                            "dep_var": result.dep_var,
                            "variables": result.variables,
                            "coefficients": result.coefficients,
                            "std_errors": result.std_errors,
                            "se_type": result.se_type,
                            "z_stats": result.z_stats,
                            "p_values": result.p_values,
                            "n_obs": result.n_obs,
                            "n_groups": result.n_groups,
                            "n_instruments": result.n_instruments,
                            "lags": result.lags,
                            "min_lag": result.min_lag,
                            "max_lag": result.max_lag,
                            "collapse": result.collapse,
                            "sigma2": result.sigma2,
                            "sargan_statistic": result.sargan_statistic,
                            "sargan_df": result.sargan_df,
                            "sargan_p_value": result.sargan_p_value,
                            "sargan_homoskedastic": result.sargan_homoskedastic,
                            "sargan_homoskedastic_p_value": result.sargan_homoskedastic_p_value,
                            "one_step_weight_rank": result.one_step_weight_rank,
                            "two_step_weight_rank": result.two_step_weight_rank,
                            "ar1_statistic": result.ar1_statistic,
                            "ar1_p_value": result.ar1_p_value,
                            "ar2_statistic": result.ar2_statistic,
                            "ar2_p_value": result.ar2_p_value,
                            "warnings": result.warnings,
                        });
                        if let Some(missing) = missing_json(&result.missing) {
                            json["missing"] = missing;
                        }
                        add_notes(&mut json, None, &result.warnings);
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
                        println!("Std. errors: {}", result.se_type);

                        println!("\nCoefficients:");
                        println!(
                            "{:<20} {:>12} {:>12} {:>10} {:>10}",
                            "Variable", "Coef", "Std Err", "z", "P>|z|"
                        );
                        println!("{}", "-".repeat(66));

                        for (i, var) in result.variables.iter().enumerate() {
                            println!(
                                "{:<20} {:>12.6} {:>12.6} {:>10.4} {:>9.4} {}",
                                var,
                                result.coefficients[i],
                                result.std_errors[i],
                                result.z_stats[i],
                                result.p_values[i],
                                significance_mark(result.p_values[i])
                            );
                        }

                        println!("\n---");
                        println!("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1");

                        println!("\nDiagnostic Tests:");
                        println!(
                            "  Sargan/Hansen test: chi2({}) = {:.4}, p = {:.4}",
                            result.sargan_df, result.sargan_statistic, result.sargan_p_value
                        );
                        if let (Some(stat), Some(p)) = (
                            result.sargan_homoskedastic,
                            result.sargan_homoskedastic_p_value,
                        ) {
                            println!(
                                "  Sargan test (homoskedastic, one-step weights): chi2({}) = \
                                 {stat:.4}, p = {p:.4}",
                                result.sargan_df
                            );
                        }
                        println!(
                            "  AR(1): z = {:.4}, p = {:.4}",
                            result.ar1_statistic, result.ar1_p_value
                        );
                        println!(
                            "  AR(2): z = {:.4}, p = {:.4}",
                            result.ar2_statistic, result.ar2_p_value
                        );
                        print_warnings(&result.warnings);
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
    model: PvcmType,
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
                        let mut json = with_missing(json, &result.missing);
                        add_result_notes(&mut json, &result);
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
                        print_result_notes(&result, format);
                        print_missing_note(&result.missing, format);
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
                        let mut json = with_missing(json, &result.missing);
                        add_result_notes(&mut json, &result);
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
                        print_result_notes(&result, format);
                        print_missing_note(&result.missing, format);
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
    model: PanelGlsModel,
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

            match run_panel_gls(ds, dep_var, &x_cols, entity, time, Some(model)) {
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
                        let mut json = with_missing(json, &result.missing);
                        add_result_notes(&mut json, &result);
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
                        // the missing-value note is one of the warnings; it prints last
                        print_notes(
                            None,
                            &without_missing_note(&result.warnings, &result.missing),
                            format,
                        );
                        print_missing_note(&result.missing, format);
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
    config: PanelUnitRootConfig,
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
        Some(ds) => match run_panel_unit_root(ds, var, unit, time, config) {
            Ok(result) => match format {
                OutputFormat::Json => {
                    let mut json = serde_json::json!({
                        "test_type": format!("{}", result.test_type),
                        "model": format!("{}", result.model),
                        "statistic": result.statistic,
                        "p_value": result.p_value,
                        "null_hypothesis": result.null_hypothesis,
                        "alternative_hypothesis": result.alternative_hypothesis,
                        "n_panels": result.n_panels,
                        "avg_time_periods": result.avg_time_periods,
                        "lags_used": result.lags_used,
                        "warnings": result.warnings,
                    });
                    if let Some(stats) = &result.unit_statistics {
                        json["unit_statistics"] = serde_json::json!(stats);
                    }
                    if let Some(p) = &result.unit_p_values {
                        json["unit_p_values"] = serde_json::json!(p);
                    }
                    json["unit_ids"] = serde_json::json!(result.unit_ids);
                    json["unit_lags"] = serde_json::json!(result.unit_lags);
                    json["lag_selection"] = serde_json::json!(result.lag_selection);
                    json["dfcor"] = serde_json::json!(result.dfcor);
                    let mut json = with_missing(json, &result.missing);
                    add_result_notes(&mut json, &result);
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

                    println!("Lags: {}", result.lag_selection);
                    if !result.unit_lags.is_empty() {
                        let joined: Vec<String> =
                            result.unit_lags.iter().map(|p| p.to_string()).collect();
                        println!("Unit lags: {}", joined.join(", "));
                    }

                    println!("\nH0: {}", result.null_hypothesis);
                    println!("H1: {}", result.alternative_hypothesis);

                    println!("\nTest statistic: {:.4}", result.statistic);
                    println!("P-value: {:.4}", result.p_value);

                    print_warnings(&result.warnings);

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
        },
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

/// `json` with `"missing"` added when rows were dropped.
fn with_missing(
    mut json: serde_json::Value,
    missing: &p2a_core::data::columns::MissingSummary,
) -> serde_json::Value {
    if let Some(m) = missing_json(missing) {
        json["missing"] = m;
    }
    json
}

/// A `Warnings:` block (nothing when there are none); the missing-value note is one of
/// them, as in `panel fe` and `panel re`.
fn print_warnings(warnings: &[String]) {
    if warnings.is_empty() {
        return;
    }
    println!("\nWarnings:");
    for w in warnings {
        println!("  - {w}");
    }
}

/// `***`, `**`, `*`, `.` or nothing for a p-value.
fn significance_mark(p: f64) -> &'static str {
    if p < 0.001 {
        "***"
    } else if p < 0.01 {
        "**"
    } else if p < 0.05 {
        "*"
    } else if p < 0.1 {
        "."
    } else {
        ""
    }
}

/// `warnings` without the missing-value note, for a caller that prints that note last.
fn without_missing_note(
    warnings: &[String],
    missing: &p2a_core::data::columns::MissingSummary,
) -> Vec<String> {
    let note = missing.note();
    warnings
        .iter()
        .filter(|w| Some(w.as_str()) != note.as_deref())
        .cloned()
        .collect()
}

/// `--lag-selection`: fixed or a criterion name [`LagSelection`] accepts; `--lags` is
/// checked against it when the command runs.
fn lag_selection_arg(s: &str) -> Result<String, String> {
    if p2a_core::options::normalize(s) == "fixed" {
        return Ok(s.to_string());
    }
    LagSelection::from_option(s, None)
        .map(|_| s.to_string())
        .map_err(|_| {
            format!(
                "unknown lag selection '{s}'; use one of: {}",
                LagSelection::names_help()
            )
        })
}

/// `panel feglm --family`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum FeglmFamily {
    #[value(alias = "binomial")]
    Logit,
    Probit,
    Poisson,
    Gaussian,
}

impl FeglmFamily {
    fn core(self) -> GlmFamily {
        match self {
            FeglmFamily::Logit => GlmFamily::Logit,
            FeglmFamily::Probit => GlmFamily::Probit,
            FeglmFamily::Poisson => GlmFamily::Poisson,
            FeglmFamily::Gaussian => GlmFamily::Gaussian,
        }
    }
}

/// `panel pvcm --model`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum PvcmModel {
    Within,
    #[value(alias = "swamy")]
    Random,
}

impl PvcmModel {
    fn core(self) -> PvcmType {
        match self {
            PvcmModel::Within => PvcmType::Within,
            PvcmModel::Random => PvcmType::Random,
        }
    }
}

/// `panel gls --model`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum PanelGlsModelArg {
    #[value(aliases = ["fixed-effects", "within"])]
    Fe,
    #[value(aliases = ["first-difference", "first_difference"])]
    Fd,
    Pooling,
}

impl PanelGlsModelArg {
    fn core(self) -> PanelGlsModel {
        match self {
            PanelGlsModelArg::Fe => PanelGlsModel::FixedEffects,
            PanelGlsModelArg::Fd => PanelGlsModel::FirstDifference,
            PanelGlsModelArg::Pooling => PanelGlsModel::Pooling,
        }
    }
}

/// `panel gmm --step`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum GmmStepArg {
    Onestep,
    Twostep,
}

/// `panel gmm --vcov`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum GmmVcovArg {
    Robust,
    Classical,
}
