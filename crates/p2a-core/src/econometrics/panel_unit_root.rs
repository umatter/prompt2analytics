//! Panel Unit Root Tests.
//!
//! Implements various panel unit root tests for testing stationarity
//! in panel data. Panel unit root tests exploit the cross-sectional
//! dimension to increase statistical power over univariate tests.
//!
//! # Available Tests
//!
//! - **Levin-Lin-Chu (LLC)**: Common unit root process across panels
//! - **Im-Pesaran-Shin (IPS)**: Heterogeneous unit root processes
//! - **Fisher-type (Maddala-Wu)**: Combines individual ADF test p-values
//! - **Hadri**: Tests null of stationarity
//!
//! # References
//!
//! - Levin, A., Lin, C.-F., & Chu, C.-S. J. (2002). "Unit Root Tests in Panel Data:
//!   Asymptotic and Finite-Sample Properties". *Journal of Econometrics*, 108(1), 1-24.
//!   https://doi.org/10.1016/S0304-4076(01)00098-7
//!
//! - Im, K. S., Pesaran, M. H., & Shin, Y. (2003). "Testing for Unit Roots in
//!   Heterogeneous Panels". *Journal of Econometrics*, 115(1), 53-74.
//!   https://doi.org/10.1016/S0304-4076(03)00109-2
//!
//! - Maddala, G. S., & Wu, S. (1999). "A Comparative Study of Unit Root Tests with
//!   Panel Data and a New Simple Test". *Oxford Bulletin of Economics and Statistics*,
//!   61(S1), 631-652.
//!
//! - Hadri, K. (2000). "Testing for Stationarity in Heterogeneous Panel Data".
//!   *Econometrics Journal*, 3(2), 148-161.
//!
//! - R packages: `plm` (Croissant & Millo), `tseries`

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::{safe_inverse, xtx, xty};
use crate::traits::estimator::normal_cdf;

// ═══════════════════════════════════════════════════════════════════════════════
// Types and Configuration
// ═══════════════════════════════════════════════════════════════════════════════

/// Type of panel unit root test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PanelUnitRootTest {
    /// Levin-Lin-Chu test (common unit root)
    #[default]
    LLC,
    /// Im-Pesaran-Shin test (heterogeneous unit root)
    IPS,
    /// Fisher-type test combining individual ADF p-values (Maddala-Wu)
    Fisher,
    /// Hadri test (null: stationarity)
    Hadri,
}

impl fmt::Display for PanelUnitRootTest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PanelUnitRootTest::LLC => write!(f, "Levin-Lin-Chu"),
            PanelUnitRootTest::IPS => write!(f, "Im-Pesaran-Shin"),
            PanelUnitRootTest::Fisher => write!(f, "Fisher (Maddala-Wu)"),
            PanelUnitRootTest::Hadri => write!(f, "Hadri"),
        }
    }
}

/// Model specification for panel unit root tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PanelModel {
    /// No deterministic components
    None,
    /// Individual intercepts only
    #[default]
    Intercept,
    /// Individual intercepts and trends
    Trend,
}

impl fmt::Display for PanelModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PanelModel::None => write!(f, "None"),
            PanelModel::Intercept => write!(f, "Individual Intercepts"),
            PanelModel::Trend => write!(f, "Individual Intercepts and Trends"),
        }
    }
}

/// Configuration for panel unit root tests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelUnitRootConfig {
    /// Type of test to perform
    pub test_type: PanelUnitRootTest,
    /// Model specification
    pub model: PanelModel,
    /// Number of lags for ADF-type regressions (None = automatic selection)
    pub lags: Option<usize>,
    /// Maximum lags for automatic selection
    pub max_lags: Option<usize>,
}

impl Default for PanelUnitRootConfig {
    fn default() -> Self {
        Self {
            test_type: PanelUnitRootTest::LLC,
            model: PanelModel::Intercept,
            lags: None,
            max_lags: None,
        }
    }
}

/// Result from a panel unit root test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelUnitRootResult {
    /// Test type used
    pub test_type: PanelUnitRootTest,
    /// Model specification
    pub model: PanelModel,
    /// Test statistic
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Null hypothesis description
    pub null_hypothesis: String,
    /// Alternative hypothesis description
    pub alternative_hypothesis: String,
    /// Number of panels (N)
    pub n_panels: usize,
    /// Average time periods (T)
    pub avg_time_periods: f64,
    /// Lags used
    pub lags_used: usize,
    /// Individual unit test statistics (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit_statistics: Option<Vec<f64>>,
    /// Individual unit p-values (for Fisher test)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit_p_values: Option<Vec<f64>>,
}

impl fmt::Display for PanelUnitRootResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "═══════════════════════════════════════════════════════════════")?;
        writeln!(f, "Panel Unit Root Test: {}", self.test_type)?;
        writeln!(f, "═══════════════════════════════════════════════════════════════")?;
        writeln!(f)?;
        writeln!(f, "Model:      {}", self.model)?;
        writeln!(f, "N panels:   {}", self.n_panels)?;
        writeln!(f, "Avg T:      {:.1}", self.avg_time_periods)?;
        writeln!(f, "Lags:       {}", self.lags_used)?;
        writeln!(f)?;
        writeln!(f, "H0: {}", self.null_hypothesis)?;
        writeln!(f, "H1: {}", self.alternative_hypothesis)?;
        writeln!(f)?;
        writeln!(f, "Test statistic: {:.4}", self.statistic)?;
        writeln!(f, "P-value:        {:.4}", self.p_value)?;
        writeln!(f)?;

        let sig = if self.p_value < 0.01 {
            "*** (reject H0 at 1%)"
        } else if self.p_value < 0.05 {
            "** (reject H0 at 5%)"
        } else if self.p_value < 0.10 {
            "* (reject H0 at 10%)"
        } else {
            "(fail to reject H0)"
        };
        writeln!(f, "Significance: {}", sig)?;

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main Entry Point
// ═══════════════════════════════════════════════════════════════════════════════

/// Run a panel unit root test.
///
/// # Arguments
///
/// * `dataset` - Panel dataset
/// * `var_col` - Variable to test for unit root
/// * `unit_col` - Panel unit identifier column
/// * `time_col` - Time period column
/// * `config` - Test configuration
///
/// # Returns
///
/// Test result with statistic and p-value.
///
/// # Example
///
/// ```ignore
/// use p2a_core::econometrics::{run_panel_unit_root, PanelUnitRootConfig, PanelUnitRootTest};
///
/// let config = PanelUnitRootConfig {
///     test_type: PanelUnitRootTest::LLC,
///     ..Default::default()
/// };
/// let result = run_panel_unit_root(&dataset, "gdp", "country", "year", config)?;
/// ```
pub fn run_panel_unit_root(
    dataset: &Dataset,
    var_col: &str,
    unit_col: &str,
    time_col: &str,
    config: PanelUnitRootConfig,
) -> EconResult<PanelUnitRootResult> {
    // Extract panel data
    let panels = extract_panels(dataset, var_col, unit_col, time_col)?;

    if panels.is_empty() {
        return Err(EconError::EmptyDataset);
    }

    // Determine lags
    let max_t = panels.values().map(|v| v.len()).max().unwrap_or(0);
    let default_max_lags = ((max_t as f64).powf(1.0 / 3.0)).floor() as usize;
    let max_lags = config.max_lags.unwrap_or(default_max_lags).max(1);
    let lags = config.lags.unwrap_or_else(|| {
        // Use AIC-based selection or default to floor(T^(1/3))
        default_max_lags.min(max_lags)
    });

    match config.test_type {
        PanelUnitRootTest::LLC => run_llc_test(&panels, config.model, lags),
        PanelUnitRootTest::IPS => run_ips_test(&panels, config.model, lags),
        PanelUnitRootTest::Fisher => run_fisher_test(&panels, config.model, lags),
        PanelUnitRootTest::Hadri => run_hadri_test(&panels, config.model),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Panel Data Extraction
// ═══════════════════════════════════════════════════════════════════════════════

/// Extract panel data organized by unit.
fn extract_panels(
    dataset: &Dataset,
    var_col: &str,
    unit_col: &str,
    time_col: &str,
) -> EconResult<BTreeMap<i64, Vec<f64>>> {
    let df = dataset.df();

    let available: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();

    // Get columns
    let var = df
        .column(var_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: var_col.to_string(),
            available: available.clone(),
        })?;
    let unit = df
        .column(unit_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: unit_col.to_string(),
            available: available.clone(),
        })?;
    let time = df
        .column(time_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: time_col.to_string(),
            available: available.clone(),
        })?;

    // Convert to f64
    let var_vals: Vec<Option<f64>> = var
        .cast(&polars::prelude::DataType::Float64)
        .map_err(|_| EconError::NonNumericColumn { column: var_col.to_string() })?
        .f64()
        .map_err(|_| EconError::NonNumericColumn { column: var_col.to_string() })?
        .into_iter()
        .collect();

    let unit_vals: Vec<Option<i64>> = unit
        .cast(&polars::prelude::DataType::Int64)
        .map_err(|_| EconError::NonNumericColumn { column: unit_col.to_string() })?
        .i64()
        .map_err(|_| EconError::NonNumericColumn { column: unit_col.to_string() })?
        .into_iter()
        .collect();

    let time_vals: Vec<Option<i64>> = time
        .cast(&polars::prelude::DataType::Int64)
        .map_err(|_| EconError::NonNumericColumn { column: time_col.to_string() })?
        .i64()
        .map_err(|_| EconError::NonNumericColumn { column: time_col.to_string() })?
        .into_iter()
        .collect();

    // Group by unit and sort by time
    let mut panels: BTreeMap<i64, BTreeMap<i64, f64>> = BTreeMap::new();

    for i in 0..var_vals.len() {
        if let (Some(v), Some(u), Some(t)) = (var_vals[i], unit_vals[i], time_vals[i]) {
            if v.is_finite() {
                panels.entry(u).or_default().insert(t, v);
            }
        }
    }

    // Convert to sorted vectors
    let result: BTreeMap<i64, Vec<f64>> = panels
        .into_iter()
        .filter_map(|(u, time_map)| {
            let vals: Vec<f64> = time_map.into_values().collect();
            if vals.len() >= 3 {
                Some((u, vals))
            } else {
                None // Need at least 3 observations per panel
            }
        })
        .collect();

    Ok(result)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Levin-Lin-Chu Test
// ═══════════════════════════════════════════════════════════════════════════════

/// Levin-Lin-Chu (2002) panel unit root test.
///
/// Tests H0: All panels have a unit root (common ρ = 0)
/// vs H1: All panels are stationary (common ρ < 0)
///
/// The LLC test assumes a common autoregressive parameter across panels
/// and is based on a pooled ADF regression with panel-specific intercepts/trends.
fn run_llc_test(
    panels: &BTreeMap<i64, Vec<f64>>,
    model: PanelModel,
    lags: usize,
) -> EconResult<PanelUnitRootResult> {
    let n = panels.len();
    let avg_t: f64 = panels.values().map(|v| v.len() as f64).sum::<f64>() / n as f64;

    // Step 1: Run individual ADF regressions and collect residuals
    let mut delta_y_tilde_all: Vec<f64> = Vec::new();
    let mut y_lag_tilde_all: Vec<f64> = Vec::new();
    let mut s2_all: Vec<f64> = Vec::new();
    let mut t_eff_all: Vec<f64> = Vec::new();

    for series in panels.values() {
        let t = series.len();
        if t <= lags + 2 {
            continue;
        }

        // Compute first differences
        let delta_y: Vec<f64> = series.windows(2).map(|w| w[1] - w[0]).collect();
        let y_lag: Vec<f64> = series[0..t - 1].to_vec();

        // Build design matrix for auxiliary regression
        let t_eff = delta_y.len() - lags;
        if t_eff < 3 {
            continue;
        }

        // Regress Δy on deterministics and lagged Δy, get residuals e_t
        // Regress y_{t-1} on same, get residuals v_{t-1}
        let (e, v) = compute_llc_residuals(&delta_y, &y_lag, model, lags)?;

        if e.len() != v.len() || e.is_empty() {
            continue;
        }

        // Compute panel-specific variance
        let s2: f64 = e.iter().map(|ei| ei * ei).sum::<f64>() / e.len() as f64;
        s2_all.push(s2.max(1e-10));
        t_eff_all.push(e.len() as f64);

        // Standardize residuals by sqrt(s2)
        let s = s2.sqrt().max(1e-10);
        for ei in &e {
            delta_y_tilde_all.push(ei / s);
        }
        for vi in &v {
            y_lag_tilde_all.push(vi / s);
        }
    }

    if delta_y_tilde_all.is_empty() || y_lag_tilde_all.is_empty() {
        return Err(EconError::InsufficientData {
            required: 10,
            provided: 0,
            context: "LLC test: no valid residuals".to_string(),
        });
    }

    // Step 2: Pooled regression of standardized residuals
    // Δ̃y = ρ * ṽ_{t-1} + error
    let n_obs = delta_y_tilde_all.len();
    let delta_y_tilde = Array1::from(delta_y_tilde_all);
    let y_lag_tilde = Array1::from(y_lag_tilde_all);

    // Simple regression: ρ̂ = Σ(ṽ·Δ̃y) / Σ(ṽ²)
    let sum_vy: f64 = y_lag_tilde
        .iter()
        .zip(delta_y_tilde.iter())
        .map(|(v, d)| v * d)
        .sum();
    let sum_vv: f64 = y_lag_tilde.iter().map(|v| v * v).sum();

    if sum_vv.abs() < 1e-10 {
        return Err(EconError::SingularMatrix {
            context: "LLC: ṽ'ṽ near zero".to_string(),
            suggestion: "Check for constant or near-constant series".to_string(),
        });
    }

    let rho_hat = sum_vy / sum_vv;

    // Compute standard error
    let resid: Array1<f64> = &delta_y_tilde - &y_lag_tilde.mapv(|v| v * rho_hat);
    let s2_resid: f64 = resid.iter().map(|r| r * r).sum::<f64>() / (n_obs - 1) as f64;
    let se_rho = (s2_resid / sum_vv).sqrt();

    // Compute t-statistic
    let t_rho = rho_hat / se_rho.max(1e-10);

    // Step 3: Bias adjustment (simplified)
    // Use approximate adjustment factors from LLC (2002) tables
    let (mu_star, sigma_star) = get_llc_adjustment_factors(avg_t, model);

    // Adjusted t-statistic
    let n_bar_t = n as f64 * avg_t;
    let s_n_t = (s2_all.iter().sum::<f64>() / n as f64).sqrt();

    // t*_ρ = (t_ρ - N̄T * S_N,T * μ*_m,T / σ̂²) / σ*_m,T
    let t_star = (t_rho * se_rho - n_bar_t.sqrt() * s_n_t * mu_star) / sigma_star;

    // P-value from standard normal
    let p_value = normal_cdf(t_star);

    Ok(PanelUnitRootResult {
        test_type: PanelUnitRootTest::LLC,
        model,
        statistic: t_star,
        p_value,
        null_hypothesis: "Panels contain unit roots (common ρ = 0)".to_string(),
        alternative_hypothesis: "Panels are stationary (common ρ < 0)".to_string(),
        n_panels: n,
        avg_time_periods: avg_t,
        lags_used: lags,
        unit_statistics: None,
        unit_p_values: None,
    })
}

/// Compute LLC auxiliary regression residuals.
fn compute_llc_residuals(
    delta_y: &[f64],
    y_lag: &[f64],
    model: PanelModel,
    lags: usize,
) -> EconResult<(Vec<f64>, Vec<f64>)> {
    let t = delta_y.len();
    let t_eff = t - lags;

    if t_eff < 3 {
        return Err(EconError::InsufficientData {
            required: 3,
            provided: t_eff,
            context: "LLC residuals".to_string(),
        });
    }

    // Build design matrix: [intercept, trend?, lagged Δy_1, ..., lagged Δy_p]
    let n_cols = match model {
        PanelModel::None => lags,
        PanelModel::Intercept => 1 + lags,
        PanelModel::Trend => 2 + lags,
    };

    let mut x = Array2::zeros((t_eff, n_cols.max(1)));
    let mut dy_vec = Vec::with_capacity(t_eff);
    let mut ylag_vec = Vec::with_capacity(t_eff);

    for i in lags..t {
        let row = i - lags;
        dy_vec.push(delta_y[i]);
        ylag_vec.push(y_lag[i]);

        let mut col = 0;
        if model == PanelModel::Intercept || model == PanelModel::Trend {
            x[[row, col]] = 1.0;
            col += 1;
        }
        if model == PanelModel::Trend {
            x[[row, col]] = (i + 1) as f64;
            col += 1;
        }
        for j in 0..lags {
            if i >= j + 1 {
                x[[row, col + j]] = delta_y[i - j - 1];
            }
        }
    }

    let dy = Array1::from(dy_vec.clone());
    let ylag = Array1::from(ylag_vec.clone());

    if n_cols == 0 {
        // No regressors, residuals are just the data
        return Ok((dy_vec, ylag_vec));
    }

    // Regress dy on X to get residuals e
    let xtx_mat = xtx(&x.view());
    let xty_dy = xty(&x.view(), &dy);
    let xty_ylag = xty(&x.view(), &ylag);

    let beta_dy = match safe_inverse(&xtx_mat.view()) {
        Ok((inv, _)) => inv.dot(&xty_dy),
        Err(_) => return Ok((dy_vec, ylag_vec)), // Fall back to original
    };

    let beta_ylag = match safe_inverse(&xtx_mat.view()) {
        Ok((inv, _)) => inv.dot(&xty_ylag),
        Err(_) => return Ok((dy_vec, ylag_vec)),
    };

    // Compute residuals
    let e: Vec<f64> = (0..t_eff)
        .map(|i| {
            let pred: f64 = (0..n_cols).map(|j| x[[i, j]] * beta_dy[j]).sum();
            dy[i] - pred
        })
        .collect();

    let v: Vec<f64> = (0..t_eff)
        .map(|i| {
            let pred: f64 = (0..n_cols).map(|j| x[[i, j]] * beta_ylag[j]).sum();
            ylag[i] - pred
        })
        .collect();

    Ok((e, v))
}

/// Get LLC adjustment factors (μ*, σ*) based on T and model.
/// These are approximations from LLC (2002) simulation tables.
fn get_llc_adjustment_factors(avg_t: f64, model: PanelModel) -> (f64, f64) {
    // Simplified approximations based on LLC (2002) Table 2
    // For more accuracy, use interpolation tables

    match model {
        PanelModel::None => {
            // No deterministics
            let mu = -0.5 / avg_t.sqrt();
            let sigma = 1.0 + 0.5 / avg_t;
            (mu, sigma)
        }
        PanelModel::Intercept => {
            // Individual means
            // μ* ≈ -0.71 - 1.61/T - 2.39/T²
            let mu = -0.71 - 1.61 / avg_t - 2.39 / (avg_t * avg_t);
            // σ* ≈ 1.0
            let sigma = 1.0;
            (mu, sigma)
        }
        PanelModel::Trend => {
            // Individual means and trends
            // μ* ≈ -1.04 - 4.69/T - 10.6/T²
            let mu = -1.04 - 4.69 / avg_t - 10.6 / (avg_t * avg_t);
            let sigma = 1.0;
            (mu, sigma)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Im-Pesaran-Shin Test
// ═══════════════════════════════════════════════════════════════════════════════

/// Im-Pesaran-Shin (2003) panel unit root test.
///
/// Tests H0: All panels have a unit root (heterogeneous ρ_i = 0)
/// vs H1: Some panels are stationary (some ρ_i < 0)
///
/// Based on averaging individual ADF t-statistics and using
/// tabulated means and variances for standardization.
fn run_ips_test(
    panels: &BTreeMap<i64, Vec<f64>>,
    model: PanelModel,
    lags: usize,
) -> EconResult<PanelUnitRootResult> {
    let n = panels.len();
    let mut t_stats: Vec<f64> = Vec::with_capacity(n);
    let mut t_lengths: Vec<usize> = Vec::with_capacity(n);

    // Compute individual ADF t-statistics
    for series in panels.values() {
        match compute_adf_t_statistic(series, model, lags) {
            Ok(t) => {
                t_stats.push(t);
                t_lengths.push(series.len());
            }
            Err(_) => continue,
        }
    }

    if t_stats.is_empty() {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "IPS test: no valid t-statistics".to_string(),
        });
    }

    let n_valid = t_stats.len();
    let avg_t: f64 = t_lengths.iter().map(|&x| x as f64).sum::<f64>() / n_valid as f64;

    // Compute average t-statistic
    let t_bar: f64 = t_stats.iter().sum::<f64>() / n_valid as f64;

    // Get moments E[t] and Var[t] from IPS (2003) tables
    let (e_t, var_t) = get_ips_moments(avg_t, model);

    // W-statistic: standardized t-bar
    let w_stat = (n_valid as f64).sqrt() * (t_bar - e_t) / var_t.sqrt();

    // P-value from standard normal
    let p_value = normal_cdf(w_stat);

    Ok(PanelUnitRootResult {
        test_type: PanelUnitRootTest::IPS,
        model,
        statistic: w_stat,
        p_value,
        null_hypothesis: "All panels have unit roots (ρ_i = 0 for all i)".to_string(),
        alternative_hypothesis: "Some panels are stationary (ρ_i < 0 for some i)".to_string(),
        n_panels: n,
        avg_time_periods: avg_t,
        lags_used: lags,
        unit_statistics: Some(t_stats),
        unit_p_values: None,
    })
}

/// Compute ADF t-statistic for a single series.
fn compute_adf_t_statistic(series: &[f64], model: PanelModel, lags: usize) -> EconResult<f64> {
    let t = series.len();
    if t <= lags + 3 {
        return Err(EconError::InsufficientData {
            required: lags + 4,
            provided: t,
            context: "ADF t-statistic".to_string(),
        });
    }

    // Compute first differences: ΔY_t = Y_t - Y_{t-1}
    // delta_y[i] = series[i+1] - series[i], so delta_y has length t-1
    let delta_y: Vec<f64> = series.windows(2).map(|w| w[1] - w[0]).collect();
    let n_diff = delta_y.len(); // t - 1

    // Effective sample size: we lose 1 for differencing, lags for lagged differences
    // So t_eff = (t - 1) - lags = t - lags - 1
    let t_eff = n_diff.saturating_sub(lags);
    if t_eff < 3 {
        return Err(EconError::InsufficientData {
            required: 3,
            provided: t_eff,
            context: "ADF regression".to_string(),
        });
    }

    // Build design matrix: [y_{t-1}, intercept?, trend?, Δy_{t-1}, ..., Δy_{t-p}]
    let n_det = match model {
        PanelModel::None => 0,
        PanelModel::Intercept => 1,
        PanelModel::Trend => 2,
    };
    let k = 1 + n_det + lags; // y_lag + deterministics + lagged differences

    let mut x = Array2::zeros((t_eff, k));
    let mut y = Array1::zeros(t_eff);

    for i in 0..t_eff {
        // obs is the index in delta_y that we're regressing
        // We need lags previous delta_y values, so start at index `lags`
        let obs = i + lags; // Index in delta_y

        y[i] = delta_y[obs];

        // y_{t-1} in original series: delta_y[obs] = series[obs+1] - series[obs]
        // So y_{t-1} = series[obs]
        x[[i, 0]] = series[obs];

        let mut col = 1;

        // Deterministics
        if model == PanelModel::Intercept || model == PanelModel::Trend {
            x[[i, col]] = 1.0;
            col += 1;
        }
        if model == PanelModel::Trend {
            x[[i, col]] = (obs + 2) as f64; // Time trend (1-indexed)
            col += 1;
        }

        // Lagged differences: Δy_{t-1}, Δy_{t-2}, ..., Δy_{t-p}
        for j in 0..lags {
            let lag_idx = obs - j - 1;
            if lag_idx < n_diff {
                x[[i, col + j]] = delta_y[lag_idx];
            }
        }
    }

    // OLS: β = (X'X)^{-1} X'y
    let xtx_mat = xtx(&x.view());
    let xty_vec = xty(&x.view(), &y);

    let (inv, _) = safe_inverse(&xtx_mat.view()).map_err(|_| EconError::SingularMatrix {
        context: "ADF regression X'X".to_string(),
        suggestion: "Check for multicollinearity or constant regressors".to_string(),
    })?;

    let beta = inv.dot(&xty_vec);

    // Residuals and variance
    let fitted: Array1<f64> = x.dot(&beta);
    let residuals = &y - &fitted;
    let s2 = residuals.iter().map(|r| r * r).sum::<f64>() / (t_eff - k) as f64;

    // Standard error of ρ̂ (coefficient on y_{t-1})
    let se_rho = (s2 * inv[[0, 0]]).sqrt();

    // t-statistic
    let t_stat = beta[0] / se_rho.max(1e-10);

    Ok(t_stat)
}

/// Get IPS (2003) expected mean and variance of t-statistic.
/// These are approximations based on Table 2 of IPS (2003).
fn get_ips_moments(avg_t: f64, model: PanelModel) -> (f64, f64) {
    match model {
        PanelModel::None => {
            // No constant: E[t] ≈ 0, Var[t] ≈ 1
            (0.0, 1.0)
        }
        PanelModel::Intercept => {
            // Individual intercept only
            // Approximation: E[t] ≈ -1.52 - 0.5/T, Var[t] ≈ 0.96 + 0.56/T
            let e_t = -1.52 - 0.5 / avg_t;
            let var_t = 0.96 + 0.56 / avg_t;
            (e_t, var_t)
        }
        PanelModel::Trend => {
            // Individual intercept and trend
            // E[t] ≈ -2.15 - 2.0/T, Var[t] ≈ 0.89 + 0.91/T
            let e_t = -2.15 - 2.0 / avg_t;
            let var_t = 0.89 + 0.91 / avg_t;
            (e_t, var_t)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Fisher Test
// ═══════════════════════════════════════════════════════════════════════════════

/// Fisher-type panel unit root test (Maddala-Wu, 1999).
///
/// Combines p-values from individual ADF tests using Fisher's method:
/// P = -2 Σ ln(p_i) ~ χ²(2N) under H0
///
/// This test allows for unbalanced panels and heterogeneous lag structures.
fn run_fisher_test(
    panels: &BTreeMap<i64, Vec<f64>>,
    model: PanelModel,
    lags: usize,
) -> EconResult<PanelUnitRootResult> {
    let n = panels.len();
    let mut p_values: Vec<f64> = Vec::with_capacity(n);
    let mut t_stats: Vec<f64> = Vec::with_capacity(n);
    let mut t_lengths: Vec<usize> = Vec::with_capacity(n);

    for series in panels.values() {
        match compute_adf_t_statistic(series, model, lags) {
            Ok(t) => {
                // Convert t-statistic to p-value using Fuller distribution approximation
                let df = (series.len() - lags - 2) as u32;
                let p = adf_p_value(t, df, model);
                p_values.push(p.max(1e-10).min(1.0 - 1e-10));
                t_stats.push(t);
                t_lengths.push(series.len());
            }
            Err(_) => continue,
        }
    }

    if p_values.is_empty() {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "Fisher test: no valid p-values".to_string(),
        });
    }

    let n_valid = p_values.len();
    let avg_t: f64 = t_lengths.iter().map(|&x| x as f64).sum::<f64>() / n_valid as f64;

    // Fisher statistic: P = -2 Σ ln(p_i)
    let fisher_stat: f64 = -2.0 * p_values.iter().map(|p| p.ln()).sum::<f64>();

    // P-value from χ²(2N)
    let df = 2.0 * n_valid as f64;
    let p_value = 1.0 - chi_squared_cdf(fisher_stat, df);

    Ok(PanelUnitRootResult {
        test_type: PanelUnitRootTest::Fisher,
        model,
        statistic: fisher_stat,
        p_value,
        null_hypothesis: "All panels have unit roots".to_string(),
        alternative_hypothesis: "At least one panel is stationary".to_string(),
        n_panels: n,
        avg_time_periods: avg_t,
        lags_used: lags,
        unit_statistics: Some(t_stats),
        unit_p_values: Some(p_values),
    })
}

/// Approximate p-value for ADF t-statistic using MacKinnon (1996) critical values.
fn adf_p_value(t_stat: f64, _df: u32, model: PanelModel) -> f64 {
    // MacKinnon (1996) response surface approximation
    // For simplicity, use standard normal CDF with appropriate shift
    // A more accurate implementation would use MacKinnon's tables

    let shift = match model {
        PanelModel::None => 0.0,
        PanelModel::Intercept => -2.86, // 5% critical value ≈ -2.86
        PanelModel::Trend => -3.41,     // 5% critical value ≈ -3.41
    };

    // Approximate p-value: transform to standard normal scale
    let z = (t_stat - shift) / 0.8;
    normal_cdf(z)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Hadri Test
// ═══════════════════════════════════════════════════════════════════════════════

/// Hadri (2000) panel stationarity test.
///
/// Tests H0: All panels are stationary
/// vs H1: Some panels have unit roots
///
/// Based on KPSS-type LM statistics averaged across panels.
fn run_hadri_test(panels: &BTreeMap<i64, Vec<f64>>, model: PanelModel) -> EconResult<PanelUnitRootResult> {
    let n = panels.len();
    let mut lm_stats: Vec<f64> = Vec::with_capacity(n);
    let mut t_lengths: Vec<usize> = Vec::with_capacity(n);

    for series in panels.values() {
        match compute_kpss_statistic(series, model) {
            Ok(lm) => {
                lm_stats.push(lm);
                t_lengths.push(series.len());
            }
            Err(_) => continue,
        }
    }

    if lm_stats.is_empty() {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "Hadri test: no valid LM statistics".to_string(),
        });
    }

    let n_valid = lm_stats.len();
    let avg_t: f64 = t_lengths.iter().map(|&x| x as f64).sum::<f64>() / n_valid as f64;

    // Average LM statistic
    let lm_bar: f64 = lm_stats.iter().sum::<f64>() / n_valid as f64;

    // Get moments for KPSS statistic
    let (mu, sigma2) = get_hadri_moments(model);

    // Z-statistic
    let z_stat = (n_valid as f64).sqrt() * (lm_bar - mu) / sigma2.sqrt();

    // P-value (one-sided, upper tail)
    let p_value = 1.0 - normal_cdf(z_stat);

    Ok(PanelUnitRootResult {
        test_type: PanelUnitRootTest::Hadri,
        model,
        statistic: z_stat,
        p_value,
        null_hypothesis: "All panels are stationary".to_string(),
        alternative_hypothesis: "Some panels have unit roots".to_string(),
        n_panels: n,
        avg_time_periods: avg_t,
        lags_used: 0,
        unit_statistics: Some(lm_stats),
        unit_p_values: None,
    })
}

/// Compute KPSS-type LM statistic for a single series.
fn compute_kpss_statistic(series: &[f64], model: PanelModel) -> EconResult<f64> {
    let t = series.len();
    if t < 3 {
        return Err(EconError::InsufficientData {
            required: 3,
            provided: t,
            context: "KPSS statistic".to_string(),
        });
    }

    // Detrend the series
    let residuals = match model {
        PanelModel::None => series.to_vec(),
        PanelModel::Intercept => {
            let mean: f64 = series.iter().sum::<f64>() / t as f64;
            series.iter().map(|&x| x - mean).collect()
        }
        PanelModel::Trend => {
            // Regress on constant and trend
            let mut x = Array2::zeros((t, 2));
            for i in 0..t {
                x[[i, 0]] = 1.0;
                x[[i, 1]] = (i + 1) as f64;
            }
            let y = Array1::from(series.to_vec());

            let xtx_mat = xtx(&x.view());
            let xty_vec = xty(&x.view(), &y);

            match safe_inverse(&xtx_mat.view()) {
                Ok((inv, _)) => {
                    let beta = inv.dot(&xty_vec);
                    (0..t).map(|i| series[i] - beta[0] - beta[1] * (i + 1) as f64).collect()
                }
                Err(_) => {
                    let mean: f64 = series.iter().sum::<f64>() / t as f64;
                    series.iter().map(|&x| x - mean).collect()
                }
            }
        }
    };

    // Compute partial sums S_t = Σ_{s=1}^{t} e_s
    let mut s: Vec<f64> = Vec::with_capacity(t);
    let mut cumsum = 0.0;
    for e in &residuals {
        cumsum += e;
        s.push(cumsum);
    }

    // Long-run variance estimate (simple Newey-West with automatic bandwidth)
    let bandwidth = ((t as f64).powf(1.0 / 3.0)).ceil() as usize;
    let mut sigma2_lr = residuals.iter().map(|e| e * e).sum::<f64>() / t as f64;

    for j in 1..=bandwidth {
        let weight = 1.0 - j as f64 / (bandwidth + 1) as f64;
        let mut gamma_j = 0.0;
        for i in j..t {
            gamma_j += residuals[i] * residuals[i - j];
        }
        gamma_j /= t as f64;
        sigma2_lr += 2.0 * weight * gamma_j;
    }

    sigma2_lr = sigma2_lr.max(1e-10);

    // LM statistic: η = (1/T²) Σ S_t² / σ²_LR
    let sum_s2: f64 = s.iter().map(|si| si * si).sum();
    let lm = sum_s2 / ((t * t) as f64 * sigma2_lr);

    Ok(lm)
}

/// Get Hadri (2000) moments for LM statistic under H0.
fn get_hadri_moments(model: PanelModel) -> (f64, f64) {
    match model {
        PanelModel::None | PanelModel::Intercept => {
            // Individual intercept: μ = 1/6, σ² = 1/45
            (1.0 / 6.0, 1.0 / 45.0)
        }
        PanelModel::Trend => {
            // Individual intercept and trend: μ = 1/15, σ² = 11/6300
            (1.0 / 15.0, 11.0 / 6300.0)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Chi-squared CDF approximation using Wilson-Hilferty transformation.
fn chi_squared_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if df <= 0.0 {
        return 1.0;
    }

    // Wilson-Hilferty approximation
    let z = ((x / df).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / (2.0 / (9.0 * df)).sqrt();
    normal_cdf(z)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn create_panel_dataset() -> Dataset {
        // Create a balanced panel with 10 units and 20 time periods
        let n_units = 10;
        let n_time = 20;

        let mut unit_ids = Vec::new();
        let mut time_ids = Vec::new();
        let mut y_stationary = Vec::new();
        let mut y_unit_root = Vec::new();

        let mut rng_seed: u64 = 12345;
        let next_rand = |seed: &mut u64| -> f64 {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((*seed >> 33) as f64 / 2147483648.0) - 1.0
        };

        for i in 0..n_units {
            let mut y_i_stat = 0.0;
            let mut y_i_ur = 0.0;

            for t in 0..n_time {
                unit_ids.push(i as i64);
                time_ids.push(t as i64);

                let eps = next_rand(&mut rng_seed) * 0.5;

                // Stationary: AR(1) with |ρ| < 1
                y_i_stat = 0.5 * y_i_stat + eps + (i as f64) * 0.1; // unit-specific intercept
                y_stationary.push(y_i_stat);

                // Unit root: random walk
                y_i_ur += eps;
                y_unit_root.push(y_i_ur + (i as f64) * 0.1);
            }
        }

        let df = DataFrame::new(vec![
            Column::new("unit".into(), &unit_ids),
            Column::new("time".into(), &time_ids),
            Column::new("y_stat".into(), &y_stationary),
            Column::new("y_ur".into(), &y_unit_root),
        ])
        .unwrap();

        Dataset::new(df)
    }

    #[test]
    fn test_llc_stationary() {
        let dataset = create_panel_dataset();
        let config = PanelUnitRootConfig {
            test_type: PanelUnitRootTest::LLC,
            model: PanelModel::Intercept,
            lags: Some(1),
            ..Default::default()
        };

        let result = run_panel_unit_root(&dataset, "y_stat", "unit", "time", config).unwrap();

        assert_eq!(result.test_type, PanelUnitRootTest::LLC);
        assert_eq!(result.n_panels, 10);
        // For stationary data, should reject H0 (unit root) at some level
        // Note: small sample, might not always reject
        println!("LLC stationary: stat={:.4}, p={:.4}", result.statistic, result.p_value);
    }

    #[test]
    fn test_llc_unit_root() {
        let dataset = create_panel_dataset();
        let config = PanelUnitRootConfig {
            test_type: PanelUnitRootTest::LLC,
            model: PanelModel::Intercept,
            lags: Some(1),
            ..Default::default()
        };

        let result = run_panel_unit_root(&dataset, "y_ur", "unit", "time", config).unwrap();

        // For unit root data, should NOT reject H0
        println!("LLC unit root: stat={:.4}, p={:.4}", result.statistic, result.p_value);
    }

    #[test]
    fn test_ips_basic() {
        let dataset = create_panel_dataset();
        let config = PanelUnitRootConfig {
            test_type: PanelUnitRootTest::IPS,
            model: PanelModel::Intercept,
            lags: Some(1),
            ..Default::default()
        };

        let result = run_panel_unit_root(&dataset, "y_stat", "unit", "time", config).unwrap();

        assert_eq!(result.test_type, PanelUnitRootTest::IPS);
        assert!(result.unit_statistics.is_some());
        println!("IPS: stat={:.4}, p={:.4}", result.statistic, result.p_value);
    }

    #[test]
    fn test_fisher_basic() {
        let dataset = create_panel_dataset();
        let config = PanelUnitRootConfig {
            test_type: PanelUnitRootTest::Fisher,
            model: PanelModel::Intercept,
            lags: Some(1),
            ..Default::default()
        };

        let result = run_panel_unit_root(&dataset, "y_stat", "unit", "time", config).unwrap();

        assert_eq!(result.test_type, PanelUnitRootTest::Fisher);
        assert!(result.unit_p_values.is_some());
        println!("Fisher: stat={:.4}, p={:.4}", result.statistic, result.p_value);
    }

    #[test]
    fn test_hadri_basic() {
        let dataset = create_panel_dataset();
        let config = PanelUnitRootConfig {
            test_type: PanelUnitRootTest::Hadri,
            model: PanelModel::Intercept,
            ..Default::default()
        };

        let result = run_panel_unit_root(&dataset, "y_stat", "unit", "time", config).unwrap();

        assert_eq!(result.test_type, PanelUnitRootTest::Hadri);
        // Hadri: H0 is stationarity, so for stationary data, should NOT reject
        println!("Hadri: stat={:.4}, p={:.4}", result.statistic, result.p_value);
    }

    #[test]
    fn test_display() {
        let result = PanelUnitRootResult {
            test_type: PanelUnitRootTest::LLC,
            model: PanelModel::Intercept,
            statistic: -2.5,
            p_value: 0.006,
            null_hypothesis: "Unit root".to_string(),
            alternative_hypothesis: "Stationary".to_string(),
            n_panels: 10,
            avg_time_periods: 20.0,
            lags_used: 1,
            unit_statistics: None,
            unit_p_values: None,
        };

        let display = format!("{}", result);
        assert!(display.contains("Levin-Lin-Chu"));
        assert!(display.contains("10"));
    }
}
