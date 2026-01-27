//! Validation runner for Rust vs R comparison
//! Usage: run-validation --method ols --data file.csv -y dep_var -x x1 x2 x3

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use ndarray::Array2;
use p2a_core::{
    run_ols, run_fixed_effects, run_logit, run_probit, run_random_effects,
    run_diagnostics, run_ols_clustered, run_hausman_test, run_hdfe, run_feglm,
    run_iv2sls, run_did, run_ipw_treatment, run_doubly_robust, run_mediation_analysis,
    run_arima, run_mstl, run_stl, run_holt_winters, run_ar,
    run_one_way_anova, run_two_way_anova, one_sample_t_test, two_sample_t_test,
    run_shapiro_wilk, run_chisq_gof, run_kaplan_meier, run_cox_ph,
    DataLoader, Dataset, LinearEstimator,
    regression::CovarianceType,
    econometrics::{GlmFamily, IpwConfig, DoublyRobustConfig, Estimand, DRMethod, MediationConfig},
    forecasting::SeasonalType,
    stats::Alternative,
    ml::{kmeans, pca, dbscan, hierarchical, Linkage},
    data::munging::{sort, filter, group_by, AggSpec, AggFn, select, standardize, lag, lead, diff},
};
use serde_json::json;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Method {
    // Regression
    Ols,
    OlsHc0,
    OlsHc1,
    OlsHc2,
    OlsHc3,
    OlsClustered,
    Diagnostics,
    // Panel
    PanelFe,
    PanelRe,
    Hausman,
    Hdfe,
    Feglm,
    // Discrete choice
    Logit,
    Probit,
    // Causal inference
    Iv2sls,
    Did,
    Ipw,
    DoublyRobust,
    Mediation,
    // Survival
    KaplanMeier,
    CoxPh,
    // Time series / Forecasting
    Arima,
    Mstl,
    Stl,
    HoltWinters,
    Ar,
    // Stats tests
    AnovaOneway,
    AnovaTwoway,
    TTestOneSample,
    TTestTwoSample,
    ShapiroWilk,
    ChisqGof,
    // ML
    Kmeans,
    Pca,
    Dbscan,
    Hierarchical,
    // Data munging
    Sort,
    Filter,
    GroupBy,
    Select,
    Standardize,
    Lag,
    Lead,
    Diff,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Robust {
    Standard,
    Hc0,
    Hc1,
    Hc2,
    Hc3,
}

impl From<Robust> for CovarianceType {
    fn from(val: Robust) -> Self {
        match val {
            Robust::Standard => CovarianceType::Standard,
            Robust::Hc0 => CovarianceType::HC0,
            Robust::Hc1 => CovarianceType::HC1,
            Robust::Hc2 => CovarianceType::HC2,
            Robust::Hc3 => CovarianceType::HC3,
        }
    }
}

#[derive(Parser)]
#[command(name = "run-validation", about = "Run validation methods")]
struct Args {
    /// Method to run
    #[arg(short, long)]
    method: Method,

    /// Path to CSV data file
    #[arg(short, long)]
    data: PathBuf,

    /// Dependent variable
    #[arg(short = 'y', long)]
    dep_var: Option<String>,

    /// Independent variables
    #[arg(short = 'x', long, num_args = 1..)]
    indep_vars: Option<Vec<String>>,

    /// Entity variable (for panel data)
    #[arg(short = 'e', long)]
    entity_var: Option<String>,

    /// Time variable (for panel data / time series)
    #[arg(short = 't', long)]
    time_var: Option<String>,

    /// Cluster variable
    #[arg(long)]
    cluster_var: Option<String>,

    /// Treatment variable (for causal inference)
    #[arg(long)]
    treatment_var: Option<String>,

    /// Post variable (for DiD)
    #[arg(long)]
    post_var: Option<String>,

    /// Instrument variables (for IV)
    #[arg(long, num_args = 1..)]
    instruments: Option<Vec<String>>,

    /// Endogenous variables (for IV)
    #[arg(long, num_args = 1..)]
    endog_vars: Option<Vec<String>>,

    /// Mediator variable (for mediation)
    #[arg(long)]
    mediator_var: Option<String>,

    /// Event variable (for survival analysis)
    #[arg(long)]
    event_var: Option<String>,

    /// Factor variable (for ANOVA)
    #[arg(long)]
    factor_var: Option<String>,

    /// Second factor variable (for two-way ANOVA)
    #[arg(long)]
    factor2_var: Option<String>,

    /// Fixed effects columns (for HDFE/FEGLM)
    #[arg(long, num_args = 1..)]
    fe_cols: Option<Vec<String>>,

    /// Number of clusters (for k-means)
    #[arg(short = 'k', long, default_value = "3")]
    k: usize,

    /// Number of components (for PCA)
    #[arg(short = 'n', long)]
    n_components: Option<usize>,

    /// Random seed
    #[arg(short = 's', long, default_value = "42")]
    seed: u64,

    /// Group variable (for group_by)
    #[arg(short = 'g', long)]
    group_var: Option<String>,

    /// Sort column
    #[arg(long)]
    sort_col: Option<String>,

    /// Robust SE type
    #[arg(short = 'r', long, default_value = "hc1")]
    robust: Robust,

    /// ARIMA p (AR order)
    #[arg(long, default_value = "1")]
    arima_p: usize,

    /// ARIMA d (differencing order)
    #[arg(long, default_value = "1")]
    arima_d: usize,

    /// ARIMA q (MA order)
    #[arg(long, default_value = "1")]
    arima_q: usize,

    /// Seasonal period (for STL, MSTL, Holt-Winters)
    #[arg(long, default_value = "12")]
    period: usize,

    /// Number of lags
    #[arg(long, default_value = "1")]
    lags: usize,

    /// Dry run - only print what would be executed
    #[arg(long)]
    dry_run: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.dry_run {
        println!("Dry run mode - would execute:");
        println!("  Method: {:?}", args.method);
        println!("  Data: {}", args.data.display());
        if let Some(ref y) = args.dep_var {
            println!("  Dependent var: {}", y);
        }
        if let Some(ref x) = args.indep_vars {
            println!("  Independent vars: {:?}", x);
        }
        return Ok(());
    }

    // Load data
    let dataset = DataLoader::load(&args.data)
        .context("Failed to load data file")?;
    let n = dataset.df().height();

    // Run method
    let result = match args.method {
        // Regression
        Method::Ols => run_ols_method(&dataset, &args, CovarianceType::Standard)?,
        Method::OlsHc0 => run_ols_method(&dataset, &args, CovarianceType::HC0)?,
        Method::OlsHc1 => run_ols_method(&dataset, &args, CovarianceType::HC1)?,
        Method::OlsHc2 => run_ols_method(&dataset, &args, CovarianceType::HC2)?,
        Method::OlsHc3 => run_ols_method(&dataset, &args, CovarianceType::HC3)?,
        Method::OlsClustered => run_ols_clustered_method(&dataset, &args)?,
        Method::Diagnostics => run_diagnostics_method(&dataset, &args)?,
        // Panel
        Method::PanelFe => run_panel_fe_method(&dataset, &args)?,
        Method::PanelRe => run_panel_re_method(&dataset, &args)?,
        Method::Hausman => run_hausman_method(&dataset, &args)?,
        Method::Hdfe => run_hdfe_method(&dataset, &args)?,
        Method::Feglm => run_feglm_method(&dataset, &args)?,
        // Discrete choice
        Method::Logit => run_logit_method(&dataset, &args)?,
        Method::Probit => run_probit_method(&dataset, &args)?,
        // Causal inference
        Method::Iv2sls => run_iv2sls_method(&dataset, &args)?,
        Method::Did => run_did_method(&dataset, &args)?,
        Method::Ipw => run_ipw_method(&dataset, &args)?,
        Method::DoublyRobust => run_doubly_robust_method(&dataset, &args)?,
        Method::Mediation => run_mediation_method(&dataset, &args)?,
        // Survival
        Method::KaplanMeier => run_kaplan_meier_method(&dataset, &args)?,
        Method::CoxPh => run_cox_ph_method(&dataset, &args)?,
        // Time series / Forecasting
        Method::Arima => run_arima_method(&dataset, &args)?,
        Method::Mstl => run_mstl_method(&dataset, &args)?,
        Method::Stl => run_stl_method(&dataset, &args)?,
        Method::HoltWinters => run_holt_winters_method(&dataset, &args)?,
        Method::Ar => run_ar_method(&dataset, &args)?,
        // Stats tests
        Method::AnovaOneway => run_anova_oneway_method(&dataset, &args)?,
        Method::AnovaTwoway => run_anova_twoway_method(&dataset, &args)?,
        Method::TTestOneSample => run_ttest_one_sample_method(&dataset, &args)?,
        Method::TTestTwoSample => run_ttest_two_sample_method(&dataset, &args)?,
        Method::ShapiroWilk => run_shapiro_wilk_method(&dataset, &args)?,
        Method::ChisqGof => run_chisq_gof_method(&dataset, &args)?,
        // ML
        Method::Kmeans => run_kmeans_method(&dataset, &args)?,
        Method::Pca => run_pca_method(&dataset, &args)?,
        Method::Dbscan => run_dbscan_method(&dataset, &args)?,
        Method::Hierarchical => run_hierarchical_method(&dataset, &args)?,
        // Munging methods
        Method::Sort => run_sort_method(&dataset, &args)?,
        Method::Filter => run_filter_method(&dataset, &args)?,
        Method::GroupBy => run_group_by_method(&dataset, &args)?,
        Method::Select => run_select_method(&dataset, &args)?,
        Method::Standardize => run_standardize_method(&dataset, &args)?,
        Method::Lag => run_lag_method(&dataset, &args)?,
        Method::Lead => run_lead_method(&dataset, &args)?,
        Method::Diff => run_diff_method(&dataset, &args)?,
    };

    // Output JSON
    let output = json!({
        "method": format!("{:?}", args.method).to_lowercase(),
        "dataset": args.data.file_name().and_then(|s| s.to_str()).unwrap_or("unknown"),
        "n": n,
        "results": result
    });

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

// ========== Helper functions ==========

fn extract_array(dataset: &Dataset, cols: &[String]) -> Result<Array2<f64>> {
    let df = dataset.df();
    let n_rows = df.height();
    let n_cols = cols.len();
    let mut data_vec = Vec::with_capacity(n_rows * n_cols);

    for row_idx in 0..n_rows {
        for col_name in cols {
            let col = df.column(col_name)
                .map_err(|e| anyhow::anyhow!("Column {} not found: {}", col_name, e))?;
            let val = col.f64()
                .map_err(|e| anyhow::anyhow!("Column {} not f64: {}", col_name, e))?
                .get(row_idx)
                .ok_or_else(|| anyhow::anyhow!("Null value at row {}", row_idx))?;
            data_vec.push(val);
        }
    }

    Array2::from_shape_vec((n_rows, n_cols), data_vec)
        .context("Failed to create array")
}

fn extract_column(dataset: &Dataset, col: &str) -> Result<Vec<f64>> {
    let df = dataset.df();
    let column = df.column(col)
        .map_err(|e| anyhow::anyhow!("Column {} not found: {}", col, e))?;
    Ok(column.f64()
        .map_err(|e| anyhow::anyhow!("Column {} not f64: {}", col, e))?
        .into_no_null_iter()
        .collect())
}

// ========== Regression Methods ==========

fn run_ols_method(dataset: &Dataset, args: &Args, cov_type: CovarianceType) -> Result<serde_json::Value> {
    let dep_var = args.dep_var.as_ref().context("dep_var required")?;
    let indep_vars: Vec<&str> = args
        .indep_vars
        .as_ref()
        .context("indep_vars required")?
        .iter()
        .map(|s| s.as_str())
        .collect();

    let result = run_ols(dataset, dep_var, &indep_vars, true, cov_type)
        .context("OLS failed")?;

    let var_names = result.variable_names();
    let coeffs: HashMap<String, f64> = var_names
        .iter()
        .zip(result.coefficients().iter())
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    let std_errs: HashMap<String, f64> = var_names
        .iter()
        .zip(result.std_errors().iter())
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    let t_stats = result.t_stats();
    let p_vals = result.p_values();
    let t_map: HashMap<String, f64> = var_names
        .iter()
        .zip(t_stats.iter())
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    let p_map: HashMap<String, f64> = var_names
        .iter()
        .zip(p_vals.iter())
        .map(|(k, v)| (k.clone(), *v))
        .collect();

    Ok(json!({
        "coefficients": coeffs,
        "std_errors": std_errs,
        "t_values": t_map,
        "p_values": p_map,
        "r_squared": result.r_squared(),
        "adj_r_squared": result.adj_r_squared(),
        "residual_std_error": result.residual_std_error(),
        "n_obs": result.n_obs()
    }))
}

fn run_ols_clustered_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let dep_var = args.dep_var.as_ref().context("dep_var required")?;
    let indep_vars: Vec<&str> = args
        .indep_vars
        .as_ref()
        .context("indep_vars required")?
        .iter()
        .map(|s| s.as_str())
        .collect();
    let cluster_var = args.cluster_var.as_ref().context("cluster_var required for clustered SE")?;

    let result = run_ols_clustered(dataset, dep_var, &indep_vars, cluster_var, None)
        .context("OLS clustered failed")?;

    let var_names = result.ols.variable_names();
    let coeffs: HashMap<String, f64> = var_names
        .iter()
        .zip(result.ols.coefficients().iter())
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    let std_errs: HashMap<String, f64> = var_names
        .iter()
        .zip(result.ols.std_errors().iter())
        .map(|(k, v)| (k.clone(), *v))
        .collect();

    Ok(json!({
        "coefficients": coeffs,
        "std_errors": std_errs,
        "r_squared": result.ols.r_squared(),
        "n_obs": result.ols.n_obs()
    }))
}

fn run_diagnostics_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let dep_var = args.dep_var.as_ref().context("dep_var required")?;
    let indep_vars: Vec<&str> = args
        .indep_vars
        .as_ref()
        .context("indep_vars required")?
        .iter()
        .map(|s| s.as_str())
        .collect();

    let result = run_diagnostics(dataset, dep_var, &indep_vars)
        .context("Diagnostics failed")?;

    let jb = result.jarque_bera.as_ref();
    let bp = result.breusch_pagan.as_ref();

    Ok(json!({
        "jarque_bera": {
            "statistic": jb.map(|t| t.statistic),
            "p_value": jb.map(|t| t.p_value)
        },
        "breusch_pagan": {
            "statistic": bp.map(|t| t.statistic),
            "p_value": bp.map(|t| t.p_value)
        },
        "durbin_watson": result.durbin_watson.as_ref().map(|dw| dw.statistic),
        "condition_number": result.condition_number
    }))
}

// ========== Panel Methods ==========

fn run_panel_fe_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let dep_var = args.dep_var.as_ref().context("dep_var required")?;
    let indep_vars: Vec<&str> = args
        .indep_vars
        .as_ref()
        .context("indep_vars required")?
        .iter()
        .map(|s| s.as_str())
        .collect();
    let entity_var = args.entity_var.as_ref().context("entity_var required for panel_fe")?;

    let result = run_fixed_effects(dataset, dep_var, &indep_vars, entity_var)
        .context("Panel FE failed")?;

    let coeffs: HashMap<String, f64> = result.variables
        .iter()
        .zip(result.coefficients.iter())
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    let std_errs: HashMap<String, f64> = result.variables
        .iter()
        .zip(result.std_errors.iter())
        .map(|(k, v)| (k.clone(), *v))
        .collect();

    Ok(json!({
        "coefficients": coeffs,
        "std_errors": std_errs,
        "r_squared": result.r_squared,
        "n_obs": result.n_obs
    }))
}

fn run_panel_re_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let dep_var = args.dep_var.as_ref().context("dep_var required")?;
    let indep_vars: Vec<&str> = args
        .indep_vars
        .as_ref()
        .context("indep_vars required")?
        .iter()
        .map(|s| s.as_str())
        .collect();
    let entity_var = args.entity_var.as_ref().context("entity_var required for panel_re")?;

    let result = run_random_effects(dataset, dep_var, &indep_vars, entity_var)
        .context("Panel RE failed")?;

    let coeffs: HashMap<String, f64> = result.variables
        .iter()
        .zip(result.coefficients.iter())
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    let std_errs: HashMap<String, f64> = result.variables
        .iter()
        .zip(result.std_errors.iter())
        .map(|(k, v)| (k.clone(), *v))
        .collect();

    Ok(json!({
        "coefficients": coeffs,
        "std_errors": std_errs,
        "r_squared": result.r_squared,
        "n_obs": result.n_obs,
        "theta": result.theta
    }))
}

fn run_hausman_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let dep_var = args.dep_var.as_ref().context("dep_var required")?;
    let indep_vars: Vec<&str> = args
        .indep_vars
        .as_ref()
        .context("indep_vars required")?
        .iter()
        .map(|s| s.as_str())
        .collect();
    let entity_var = args.entity_var.as_ref().context("entity_var required for hausman")?;

    let result = run_hausman_test(dataset, dep_var, &indep_vars, entity_var)
        .context("Hausman test failed")?;

    Ok(json!({
        "chi2_statistic": result.chi2_statistic,
        "p_value": result.p_value,
        "df": result.df,
        "recommendation": result.recommendation
    }))
}

fn run_hdfe_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let dep_var = args.dep_var.as_ref().context("dep_var required")?;
    let indep_vars: Vec<&str> = args
        .indep_vars
        .as_ref()
        .context("indep_vars required")?
        .iter()
        .map(|s| s.as_str())
        .collect();
    let fe_cols: Vec<&str> = args
        .fe_cols
        .as_ref()
        .context("fe_cols required for hdfe")?
        .iter()
        .map(|s| s.as_str())
        .collect();

    let result = run_hdfe(dataset, dep_var, &indep_vars, &fe_cols, None, CovarianceType::HC1)
        .context("HDFE failed")?;

    let coeffs: HashMap<String, f64> = result.variables
        .iter()
        .zip(result.coefficients.iter())
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    let std_errs: HashMap<String, f64> = result.variables
        .iter()
        .zip(result.std_errors.iter())
        .map(|(k, v)| (k.clone(), *v))
        .collect();

    Ok(json!({
        "coefficients": coeffs,
        "std_errors": std_errs,
        "r_squared_within": result.r_squared_within,
        "n_obs": result.n_obs,
        "fe_dimensions": result.fe_dimensions,
        "fe_counts": result.fe_counts
    }))
}

fn run_feglm_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let dep_var = args.dep_var.as_ref().context("dep_var required")?;
    let indep_vars: Vec<&str> = args
        .indep_vars
        .as_ref()
        .context("indep_vars required")?
        .iter()
        .map(|s| s.as_str())
        .collect();
    let fe_cols: Vec<&str> = args
        .fe_cols
        .as_ref()
        .context("fe_cols required for feglm")?
        .iter()
        .map(|s| s.as_str())
        .collect();

    let result = run_feglm(dataset, dep_var, &indep_vars, &fe_cols, GlmFamily::Logit, None)
        .context("FEGLM failed")?;

    let coeffs: HashMap<String, f64> = result.variables
        .iter()
        .zip(result.coefficients.iter())
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    let std_errs: HashMap<String, f64> = result.variables
        .iter()
        .zip(result.std_errors.iter())
        .map(|(k, v)| (k.clone(), *v))
        .collect();

    Ok(json!({
        "coefficients": coeffs,
        "std_errors": std_errs,
        "log_likelihood": result.log_likelihood,
        "pseudo_r_squared": result.pseudo_r_squared,
        "n_obs": result.n_obs,
        "fe_dimensions": result.fe_dimensions
    }))
}

// ========== Discrete Choice ==========

fn run_logit_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let dep_var = args.dep_var.as_ref().context("dep_var required")?;
    let indep_vars: Vec<&str> = args
        .indep_vars
        .as_ref()
        .context("indep_vars required")?
        .iter()
        .map(|s| s.as_str())
        .collect();

    let result = run_logit(dataset, dep_var, &indep_vars)
        .context("Logit failed")?;

    let coeffs: HashMap<String, f64> = result.variables
        .iter()
        .zip(result.coefficients.iter())
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    let std_errs: HashMap<String, f64> = result.variables
        .iter()
        .zip(result.std_errors.iter())
        .map(|(k, v)| (k.clone(), *v))
        .collect();

    Ok(json!({
        "coefficients": coeffs,
        "std_errors": std_errs,
        "log_likelihood": result.log_likelihood,
        "pseudo_r_squared": result.pseudo_r_squared
    }))
}

fn run_probit_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let dep_var = args.dep_var.as_ref().context("dep_var required")?;
    let indep_vars: Vec<&str> = args
        .indep_vars
        .as_ref()
        .context("indep_vars required")?
        .iter()
        .map(|s| s.as_str())
        .collect();

    let result = run_probit(dataset, dep_var, &indep_vars)
        .context("Probit failed")?;

    let coeffs: HashMap<String, f64> = result.variables
        .iter()
        .zip(result.coefficients.iter())
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    let std_errs: HashMap<String, f64> = result.variables
        .iter()
        .zip(result.std_errors.iter())
        .map(|(k, v)| (k.clone(), *v))
        .collect();

    Ok(json!({
        "coefficients": coeffs,
        "std_errors": std_errs,
        "log_likelihood": result.log_likelihood,
        "pseudo_r_squared": result.pseudo_r_squared
    }))
}

// ========== Causal Inference ==========

fn run_iv2sls_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let dep_var = args.dep_var.as_ref().context("dep_var required")?;
    let indep_vars: Vec<&str> = args
        .indep_vars
        .as_ref()
        .context("indep_vars required (exogenous)")?
        .iter()
        .map(|s| s.as_str())
        .collect();
    let endog_vars: Vec<&str> = args
        .endog_vars
        .as_ref()
        .context("endog_vars required for IV")?
        .iter()
        .map(|s| s.as_str())
        .collect();
    let instruments: Vec<&str> = args
        .instruments
        .as_ref()
        .context("instruments required for IV")?
        .iter()
        .map(|s| s.as_str())
        .collect();

    let result = run_iv2sls(dataset, dep_var, &indep_vars, &endog_vars, &instruments, true)
        .context("IV 2SLS failed")?;

    let coeffs: HashMap<String, f64> = result.variables
        .iter()
        .zip(result.coefficients.iter())
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    let std_errs: HashMap<String, f64> = result.variables
        .iter()
        .zip(result.std_errors.iter())
        .map(|(k, v)| (k.clone(), *v))
        .collect();

    Ok(json!({
        "coefficients": coeffs,
        "std_errors": std_errs,
        "r_squared": result.r_squared,
        "n_obs": result.n_obs
    }))
}

fn run_did_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let dep_var = args.dep_var.as_ref().context("dep_var required")?;
    let treatment_var = args.treatment_var.as_ref().context("treatment_var required for DiD")?;
    let post_var = args.post_var.as_ref().context("post_var required for DiD")?;
    let controls: Option<Vec<&str>> = args.indep_vars.as_ref().map(|v| v.iter().map(|s| s.as_str()).collect());

    let result = run_did(dataset, dep_var, treatment_var, post_var, controls.as_deref())
        .context("DiD failed")?;

    Ok(json!({
        "att": result.att,
        "std_error": result.std_error,
        "t_stat": result.t_stat,
        "p_value": result.p_value,
        "r_squared": result.r_squared,
        "n_obs": result.n_obs
    }))
}

fn run_ipw_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let dep_var = args.dep_var.as_ref().context("dep_var required")?;
    let treatment_var = args.treatment_var.as_ref().context("treatment_var required for IPW")?;
    let covariates: Vec<&str> = args
        .indep_vars
        .as_ref()
        .context("indep_vars required for IPW")?
        .iter()
        .map(|s| s.as_str())
        .collect();

    let config = IpwConfig {
        estimand: Estimand::ATE,
        ..Default::default()
    };

    let result = run_ipw_treatment(dataset, dep_var, treatment_var, &covariates, config)
        .context("IPW failed")?;

    Ok(json!({
        "effect": result.effect,
        "std_error": result.std_error,
        "t_stat": result.t_stat,
        "p_value": result.p_value,
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
        "n_obs": result.n_obs
    }))
}

fn run_doubly_robust_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let dep_var = args.dep_var.as_ref().context("dep_var required")?;
    let treatment_var = args.treatment_var.as_ref().context("treatment_var required for DR")?;
    let covariates: Vec<&str> = args
        .indep_vars
        .as_ref()
        .context("indep_vars required for DR")?
        .iter()
        .map(|s| s.as_str())
        .collect();

    let config = DoublyRobustConfig {
        estimand: Estimand::ATE,
        method: DRMethod::AIPW,
        ..Default::default()
    };

    let result = run_doubly_robust(dataset, dep_var, treatment_var, &covariates, config)
        .context("Doubly robust failed")?;

    Ok(json!({
        "effect": result.effect,
        "std_error": result.std_error,
        "t_stat": result.t_stat,
        "p_value": result.p_value,
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
        "n_obs": result.n_obs
    }))
}

fn run_mediation_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let dep_var = args.dep_var.as_ref().context("dep_var required")?;
    let treatment_var = args.treatment_var.as_ref().context("treatment_var required for mediation")?;
    let mediator_var = args.mediator_var.as_ref().context("mediator_var required for mediation")?;
    let covariates: Vec<&str> = args
        .indep_vars
        .as_ref()
        .map(|v| v.iter().map(|s| s.as_str()).collect())
        .unwrap_or_default();

    let config = MediationConfig::default();
    let result = run_mediation_analysis(dataset, dep_var, treatment_var, mediator_var, &covariates, config)
        .context("Mediation failed")?;

    Ok(json!({
        "total_effect": result.total_effect,
        "direct_effect": result.direct_effect,
        "indirect_effect": result.indirect_effect,
        "proportion_mediated": result.proportion_mediated,
        "n_obs": result.n_obs
    }))
}

// ========== Survival Analysis ==========

fn run_kaplan_meier_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let time_var = args.time_var.as_ref().context("time_var required for KM")?;
    let event_var = args.event_var.as_ref().context("event_var required for KM")?;
    let group_var = args.group_var.as_ref().map(|s| s.as_str());

    let results = run_kaplan_meier(dataset, time_var, event_var, group_var, 0.95)
        .context("Kaplan-Meier failed")?;

    // Use the first result (overall if no grouping)
    let result = results.first().context("No Kaplan-Meier results")?;

    Ok(json!({
        "n_obs": result.n_obs,
        "total_events": result.total_events,
        "median_survival": result.median_survival,
        "n_times": result.times.len()
    }))
}

fn run_cox_ph_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let time_var = args.time_var.as_ref().context("time_var required for Cox PH")?;
    let event_var = args.event_var.as_ref().context("event_var required for Cox PH")?;
    let indep_vars: Vec<&str> = args
        .indep_vars
        .as_ref()
        .context("indep_vars required for Cox PH")?
        .iter()
        .map(|s| s.as_str())
        .collect();

    let result = run_cox_ph(dataset, time_var, event_var, &indep_vars, None)
        .context("Cox PH failed")?;

    let coeffs: HashMap<String, f64> = result.variables
        .iter()
        .zip(result.coefficients.iter())
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    let std_errs: HashMap<String, f64> = result.variables
        .iter()
        .zip(result.std_errors.iter())
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    let hazard_ratios: HashMap<String, f64> = result.variables
        .iter()
        .zip(result.hazard_ratios.iter())
        .map(|(k, v)| (k.clone(), *v))
        .collect();

    Ok(json!({
        "coefficients": coeffs,
        "std_errors": std_errs,
        "hazard_ratios": hazard_ratios,
        "log_likelihood": result.log_likelihood,
        "n_obs": result.n_obs,
        "n_events": result.n_events
    }))
}

// ========== Time Series / Forecasting ==========

fn run_arima_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let col = args.dep_var.as_ref()
        .or_else(|| args.indep_vars.as_ref().and_then(|v| v.first()))
        .context("dep_var or indep_vars[0] required for ARIMA")?;

    let result = run_arima(dataset, col, args.arima_p, args.arima_d, args.arima_q)
        .context("ARIMA failed")?;

    Ok(json!({
        "p": result.p,
        "d": result.d,
        "q": result.q,
        "ar_coeffs": result.ar_coeffs,
        "ma_coeffs": result.ma_coeffs,
        "intercept": result.intercept,
        "ssr": result.ssr,
        "aic": result.aic,
        "n_obs": result.n_obs
    }))
}

fn run_mstl_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let col = args.dep_var.as_ref()
        .or_else(|| args.indep_vars.as_ref().and_then(|v| v.first()))
        .context("dep_var or indep_vars[0] required for MSTL")?;

    let periods = vec![args.period];
    let result = run_mstl(dataset, col, &periods)
        .context("MSTL failed")?;

    Ok(json!({
        "periods": result.periods,
        "n_obs": result.n_obs,
        "trend_length": result.trend.len(),
        "n_seasonal": result.seasonal.len()
    }))
}

fn run_stl_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let col = args.dep_var.as_ref()
        .or_else(|| args.indep_vars.as_ref().and_then(|v| v.first()))
        .context("dep_var or indep_vars[0] required for STL")?;

    let result = run_stl(dataset, col, args.period, false)
        .context("STL failed")?;

    Ok(json!({
        "period": result.period,
        "n_obs": result.x.len(),
        "seasonal_strength": result.seasonal_strength
    }))
}

fn run_holt_winters_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let col = args.dep_var.as_ref()
        .or_else(|| args.indep_vars.as_ref().and_then(|v| v.first()))
        .context("dep_var or indep_vars[0] required for Holt-Winters")?;

    let result = run_holt_winters(dataset, col, args.period, SeasonalType::Additive, None, None, None)
        .context("Holt-Winters failed")?;

    Ok(json!({
        "alpha": result.alpha,
        "beta": result.beta,
        "gamma": result.gamma,
        "n_obs": result.n_obs,
        "sse": result.sse,
        "period": result.period
    }))
}

fn run_ar_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let col = args.dep_var.as_ref()
        .or_else(|| args.indep_vars.as_ref().and_then(|v| v.first()))
        .context("dep_var or indep_vars[0] required for AR")?;

    let data = extract_column(dataset, col)?;
    let result = run_ar(&data)
        .context("AR failed")?;

    Ok(json!({
        "order": result.order,
        "ar_coefficients": result.ar,
        "var_pred": result.var_pred,
        "aic": result.aic,
        "n_obs": result.n_obs
    }))
}

// ========== Stats Tests ==========

fn run_anova_oneway_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let response = args.dep_var.as_ref().context("dep_var required for ANOVA")?;
    let factor = args.factor_var.as_ref()
        .or_else(|| args.group_var.as_ref())
        .context("factor_var or group_var required for ANOVA")?;

    let result = run_one_way_anova(dataset, response, factor)
        .context("One-way ANOVA failed")?;

    Ok(json!({
        "f_statistic": result.f_statistic,
        "p_value": result.p_value,
        "df_between": result.df_between,
        "df_within": result.df_within,
        "ss_between": result.ss_between,
        "ss_within": result.ss_within
    }))
}

fn run_anova_twoway_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let response = args.dep_var.as_ref().context("dep_var required for ANOVA")?;
    let factor_a = args.factor_var.as_ref().context("factor_var required for two-way ANOVA")?;
    let factor_b = args.factor2_var.as_ref().context("factor2_var required for two-way ANOVA")?;

    let result = run_two_way_anova(dataset, response, factor_a, factor_b, true)
        .context("Two-way ANOVA failed")?;

    Ok(json!({
        "f_a": result.f_a,
        "p_a": result.p_a,
        "f_b": result.f_b,
        "p_b": result.p_b,
        "f_ab": result.f_ab,
        "p_ab": result.p_ab
    }))
}

fn run_ttest_one_sample_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let col = args.dep_var.as_ref()
        .or_else(|| args.indep_vars.as_ref().and_then(|v| v.first()))
        .context("dep_var or indep_vars[0] required for t-test")?;

    let data = extract_column(dataset, col)?;
    let result = one_sample_t_test(&data, 0.0, Alternative::TwoSided, 0.95)
        .context("One-sample t-test failed")?;

    Ok(json!({
        "t_statistic": result.t_statistic,
        "p_value": result.p_value,
        "df": result.df,
        "conf_int_lower": result.conf_int_lower,
        "conf_int_upper": result.conf_int_upper,
        "estimate": result.estimate
    }))
}

fn run_ttest_two_sample_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let indep_vars = args.indep_vars.as_ref().context("indep_vars required for two-sample t-test")?;
    if indep_vars.len() < 2 {
        return Err(anyhow::anyhow!("Two columns required for two-sample t-test"));
    }

    let x = extract_column(dataset, &indep_vars[0])?;
    let y = extract_column(dataset, &indep_vars[1])?;
    let result = two_sample_t_test(&x, &y, 0.0, Alternative::TwoSided, false, 0.95)
        .context("Two-sample t-test failed")?;

    Ok(json!({
        "t_statistic": result.t_statistic,
        "p_value": result.p_value,
        "df": result.df,
        "conf_int_lower": result.conf_int_lower,
        "conf_int_upper": result.conf_int_upper,
        "estimate": result.estimate
    }))
}

fn run_shapiro_wilk_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let col = args.dep_var.as_ref()
        .or_else(|| args.indep_vars.as_ref().and_then(|v| v.first()))
        .context("dep_var or indep_vars[0] required for Shapiro-Wilk")?;

    let result = run_shapiro_wilk(dataset, col)
        .context("Shapiro-Wilk failed")?;

    Ok(json!({
        "w_statistic": result.w_statistic,
        "p_value": result.p_value,
        "n": result.n
    }))
}

fn run_chisq_gof_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let col = args.dep_var.as_ref()
        .or_else(|| args.indep_vars.as_ref().and_then(|v| v.first()))
        .context("dep_var or indep_vars[0] required for chi-squared")?;

    let result = run_chisq_gof(dataset, col, None)
        .context("Chi-squared GOF failed")?;

    Ok(json!({
        "statistic": result.statistic,
        "p_value": result.p_value,
        "df": result.df,
        "test_name": result.test_name
    }))
}

// ========== ML Methods ==========

fn run_kmeans_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let indep_vars = args
        .indep_vars
        .as_ref()
        .context("indep_vars required for kmeans")?;

    let data = extract_array(dataset, indep_vars)?;

    let result = kmeans(data.view(), args.k, Some(100), Some(1e-6), Some(10), Some(args.seed))
        .map_err(|e| anyhow::anyhow!("K-means failed: {}", e))?;

    Ok(json!({
        "k": args.k,
        "inertia": result.inertia,
        "n_iterations": result.n_iterations,
        "centers": result.centroids.as_slice()
    }))
}

fn run_pca_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let indep_vars = args
        .indep_vars
        .as_ref()
        .context("indep_vars required for pca")?;

    let data = extract_array(dataset, indep_vars)?;

    let result = pca(data.view(), args.n_components, false)
        .map_err(|e| anyhow::anyhow!("PCA failed: {}", e))?;

    Ok(json!({
        "n_components": result.n_components,
        "explained_variance": result.explained_variance.as_slice(),
        "explained_variance_ratio": result.explained_variance_ratio.as_slice(),
        "total_variance": result.total_variance,
        "components": result.components.as_slice()
    }))
}

fn run_dbscan_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let indep_vars = args
        .indep_vars
        .as_ref()
        .context("indep_vars required for dbscan")?;

    let data = extract_array(dataset, indep_vars)?;

    let eps = 0.5;
    let min_samples = 5;
    let result = dbscan(data.view(), eps, min_samples)
        .map_err(|e| anyhow::anyhow!("DBSCAN failed: {}", e))?;

    Ok(json!({
        "n_clusters": result.n_clusters,
        "n_noise": result.n_noise
    }))
}

fn run_hierarchical_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let indep_vars = args
        .indep_vars
        .as_ref()
        .context("indep_vars required for hierarchical")?;

    let data = extract_array(dataset, indep_vars)?;

    let result = hierarchical(data.view(), Some(args.k), Linkage::Ward, None)
        .map_err(|e| anyhow::anyhow!("Hierarchical failed: {}", e))?;

    Ok(json!({
        "n_clusters": result.n_clusters,
        "labels": result.labels
    }))
}

// ========== Munging Methods ==========

fn run_sort_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let sort_col = args.sort_col.as_ref()
        .or_else(|| args.indep_vars.as_ref().and_then(|v| v.first()))
        .context("sort_col or indep_vars required for sort")?;

    let result = sort(dataset, &[sort_col.as_str()], &[false])
        .map_err(|e| anyhow::anyhow!("Sort failed: {}", e))?;

    Ok(json!({
        "n_rows": result.nrows(),
        "n_cols": result.ncols(),
        "sort_column": sort_col
    }))
}

fn run_filter_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let filter_col = args.indep_vars.as_ref()
        .and_then(|v| v.first())
        .context("indep_vars required for filter")?;

    // Get median value for the filter
    let col = dataset.df().column(filter_col)
        .map_err(|e| anyhow::anyhow!("Column {} not found: {}", filter_col, e))?;
    let values: Vec<f64> = col.f64()
        .map_err(|e| anyhow::anyhow!("Column {} not f64: {}", filter_col, e))?
        .into_no_null_iter()
        .collect();

    let mut sorted = values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];

    // Filter rows where value > median
    let result = filter(dataset, filter_col, "gt", &median.to_string())
        .map_err(|e| anyhow::anyhow!("Filter failed: {}", e))?;

    Ok(json!({
        "n_rows_before": dataset.nrows(),
        "n_rows_after": result.nrows(),
        "filter_column": filter_col,
        "threshold": median
    }))
}

fn run_group_by_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let group_col = args.group_var.as_ref()
        .or_else(|| args.entity_var.as_ref())
        .context("group_var or entity_var required for group_by")?;

    let agg_col = args.dep_var.as_ref()
        .or_else(|| args.indep_vars.as_ref().and_then(|v| v.first()))
        .context("dep_var or indep_vars required for group_by aggregation")?;

    let agg_specs = vec![AggSpec::new(agg_col, AggFn::Mean)];

    let result = group_by(dataset, &[group_col.as_str()], &agg_specs)
        .map_err(|e| anyhow::anyhow!("Group by failed: {}", e))?;

    Ok(json!({
        "n_groups": result.nrows(),
        "group_column": group_col,
        "agg_column": agg_col
    }))
}

fn run_select_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let cols: Vec<&str> = args.indep_vars.as_ref()
        .context("indep_vars required for select")?
        .iter().map(|s| s.as_str()).collect();

    let result = select(dataset, &cols)
        .map_err(|e| anyhow::anyhow!("Select failed: {}", e))?;

    Ok(json!({
        "n_rows": result.nrows(),
        "n_cols": result.ncols()
    }))
}

fn run_standardize_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let cols: Vec<&str> = args.indep_vars.as_ref()
        .context("indep_vars required for standardize")?
        .iter().map(|s| s.as_str()).collect();

    let result = standardize(dataset, &cols)
        .map_err(|e| anyhow::anyhow!("Standardize failed: {}", e))?;

    Ok(json!({
        "n_rows": result.nrows(),
        "n_cols": result.ncols()
    }))
}

fn run_lag_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let col = args.indep_vars.as_ref()
        .and_then(|v| v.first())
        .context("indep_vars required for lag")?;

    let result = lag(dataset, col, args.lags, None)
        .map_err(|e| anyhow::anyhow!("Lag failed: {}", e))?;

    Ok(json!({
        "n_rows": result.nrows(),
        "n_cols": result.ncols()
    }))
}

fn run_lead_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let col = args.indep_vars.as_ref()
        .and_then(|v| v.first())
        .context("indep_vars required for lead")?;

    let result = lead(dataset, col, args.lags, None)
        .map_err(|e| anyhow::anyhow!("Lead failed: {}", e))?;

    Ok(json!({
        "n_rows": result.nrows(),
        "n_cols": result.ncols()
    }))
}

fn run_diff_method(dataset: &Dataset, args: &Args) -> Result<serde_json::Value> {
    let col = args.indep_vars.as_ref()
        .and_then(|v| v.first())
        .context("indep_vars required for diff")?;

    let result = diff(dataset, col, args.lags)
        .map_err(|e| anyhow::anyhow!("Diff failed: {}", e))?;

    Ok(json!({
        "n_rows": result.nrows(),
        "n_cols": result.ncols()
    }))
}
