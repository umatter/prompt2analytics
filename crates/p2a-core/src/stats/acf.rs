//! Autocorrelation and Cross-Correlation Functions.
//!
//! Provides ACF, PACF, and CCF computation for time series analysis.
//!
//! # References
//!
//! - Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
//!   *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
//! - Brockwell, P. J., & Davis, R. A. (1991). *Time Series: Theory and Methods*
//!   (2nd ed.). Springer.
//! - Durbin, J. (1960). "The Fitting of Time-Series Models". *Revue de
//!   l'Institut International de Statistique*, 28(3), 233-244.
//! - R `stats::acf`: <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/acf.html>

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{NaPolicy, numeric_column};
use crate::errors::{EconError, EconResult};
use crate::traits::estimator::z_critical;

/// Refuse a series with NaN (`NullValues`) or infinite values (`NonFiniteValues`), as R's
/// `na.fail` refuses `NA`, naming the series.
pub(crate) fn refuse_non_finite(x: &[f64], name: &str) -> EconResult<()> {
    let nan = x.iter().filter(|v| v.is_nan()).count();
    if nan > 0 {
        return Err(EconError::NullValues {
            column: name.to_string(),
            count: nan,
        });
    }
    let inf = x.iter().filter(|v| v.is_infinite()).count();
    if inf > 0 {
        return Err(EconError::NonFiniteValues {
            column: name.to_string(),
            count: inf,
        });
    }
    Ok(())
}

/// Type of autocorrelation/covariance to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AcfType {
    /// Sample autocorrelation function (normalized to [-1, 1])
    #[default]
    Correlation,
    /// Sample autocovariance function
    Covariance,
    /// Partial autocorrelation function (via Durbin-Levinson)
    Partial,
}

crate::option_enum!(
    AcfType,
    crate::options::Choices {
        option: "ACF type",
        values: &[
            (&["correlation", "acf", "cor"], AcfType::Correlation),
            (
                &["covariance", "cov", "acvf", "autocovariance"],
                AcfType::Covariance
            ),
            (&["partial", "pacf"], AcfType::Partial),
        ],
    },
    no_display
);

impl fmt::Display for AcfType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AcfType::Correlation => write!(f, "Autocorrelation"),
            AcfType::Covariance => write!(f, "Autocovariance"),
            AcfType::Partial => write!(f, "Partial Autocorrelation"),
        }
    }
}

/// Type of cross-correlation/covariance to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CcfType {
    /// Sample cross-correlation function
    #[default]
    Correlation,
    /// Sample cross-covariance function
    Covariance,
}

crate::option_enum!(
    CcfType,
    crate::options::Choices {
        option: "CCF type",
        values: &[
            (&["correlation", "cor", "ccf"], CcfType::Correlation),
            (&["covariance", "cov"], CcfType::Covariance),
        ],
    },
    no_display
);

impl fmt::Display for CcfType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CcfType::Correlation => write!(f, "Cross-Correlation"),
            CcfType::Covariance => write!(f, "Cross-Covariance"),
        }
    }
}

/// Result from ACF computation.
///
/// Contains autocorrelation or autocovariance values for lags 0 to lag_max.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcfResult {
    /// Type of function computed
    pub acf_type: AcfType,
    /// Lag values (0 to lag_max)
    pub lags: Vec<i32>,
    /// ACF/ACVF values for each lag
    pub values: Vec<f64>,
    /// Number of observations used
    pub n_obs: usize,
    /// Series name (if from dataset)
    pub series_name: Option<String>,
    /// 95% confidence interval bounds (±value) assuming white noise
    /// Only for correlation type
    pub confidence_bound: Option<f64>,
    /// Whether data was demeaned
    pub demeaned: bool,
    /// Whether adjusted denominator (n-k instead of n) was used
    pub adjusted: bool,
}

impl fmt::Display for AcfResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} Function Results", self.acf_type)?;
        writeln!(f, "==============================================")?;
        if let Some(ref name) = self.series_name {
            writeln!(f, "Series: {}", name)?;
        }
        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(f, "Max Lag: {}", self.lags.last().unwrap_or(&0))?;
        writeln!(f, "Demeaned: {}", if self.demeaned { "Yes" } else { "No" })?;
        writeln!(
            f,
            "Adjusted: {}",
            if self.adjusted { "Yes (n-k)" } else { "No (n)" }
        )?;
        writeln!(f)?;

        // Print values in a compact format
        writeln!(f, "Lag    Value")?;
        writeln!(f, "---    -----")?;
        for (lag, value) in self.lags.iter().zip(self.values.iter()) {
            writeln!(f, "{:3}    {:.4}", lag, value)?;
        }

        if let Some(bound) = self.confidence_bound {
            writeln!(f)?;
            writeln!(f, "95% CI for white noise: ±{:.4}", bound)?;
        }

        Ok(())
    }
}

/// Result from PACF computation.
///
/// Contains partial autocorrelation values for lags 1 to lag_max.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PacfResult {
    /// Lag values (1 to lag_max)
    pub lags: Vec<i32>,
    /// PACF values for each lag
    pub values: Vec<f64>,
    /// Number of observations used
    pub n_obs: usize,
    /// Series name (if from dataset)
    pub series_name: Option<String>,
    /// 95% confidence interval bounds (±value) assuming white noise
    pub confidence_bound: f64,
}

impl fmt::Display for PacfResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Partial Autocorrelation Function Results")?;
        writeln!(f, "==============================================")?;
        if let Some(ref name) = self.series_name {
            writeln!(f, "Series: {}", name)?;
        }
        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(f, "Max Lag: {}", self.lags.last().unwrap_or(&0))?;
        writeln!(f)?;

        writeln!(f, "Lag    PACF")?;
        writeln!(f, "---    ----")?;
        for (lag, value) in self.lags.iter().zip(self.values.iter()) {
            let sig = if value.abs() > self.confidence_bound {
                "*"
            } else {
                ""
            };
            writeln!(f, "{:3}    {:.4} {}", lag, value, sig)?;
        }

        writeln!(f)?;
        writeln!(f, "95% CI for white noise: ±{:.4}", self.confidence_bound)?;
        writeln!(f, "* indicates significant at 5% level")?;

        Ok(())
    }
}

/// Result from CCF computation.
///
/// Contains cross-correlation or cross-covariance values for lags -lag_max to +lag_max.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CcfResult {
    /// Type of function computed
    pub ccf_type: CcfType,
    /// Lag values (-lag_max to +lag_max)
    pub lags: Vec<i32>,
    /// CCF/CCVF values for each lag
    pub values: Vec<f64>,
    /// Number of observations used
    pub n_obs: usize,
    /// X series name
    pub x_series: Option<String>,
    /// Y series name
    pub y_series: Option<String>,
    /// 95% confidence interval bounds (±value) assuming white noise
    pub confidence_bound: Option<f64>,
}

impl fmt::Display for CcfResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} Function Results", self.ccf_type)?;
        writeln!(f, "==============================================")?;
        if let (Some(x), Some(y)) = (&self.x_series, &self.y_series) {
            writeln!(f, "X series: {}", x)?;
            writeln!(f, "Y series: {}", y)?;
        }
        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(f)?;

        writeln!(f, "Lag    Value")?;
        writeln!(f, "---    -----")?;
        for (lag, value) in self.lags.iter().zip(self.values.iter()) {
            writeln!(f, "{:4}   {:.4}", lag, value)?;
        }

        if let Some(bound) = self.confidence_bound {
            writeln!(f)?;
            writeln!(f, "95% CI for white noise: ±{:.4}", bound)?;
        }

        Ok(())
    }
}

/// Compute the sample autocovariance function.
///
/// # Formula
///
/// γ̂(k) = (1/n) Σ_{t=1}^{n-|k|} (x_{t+|k|} - x̄)(x_t - x̄)
///
/// If `adjusted` is true, divides by (n-|k|) instead of n.
///
/// # Arguments
///
/// * `x` - Time series data
/// * `lag_max` - Maximum lag to compute. If None, uses min(10*log10(n), n-1)
/// * `demean` - Whether to subtract the mean before computing
/// * `adjusted` - If true, divide by (n-k) instead of n
///
/// # References
///
/// Brockwell & Davis (1991), Section 1.4
fn autocovariance(x: &[f64], lag_max: usize, demean: bool, adjusted: bool) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }

    let mean = if demean {
        x.iter().sum::<f64>() / n as f64
    } else {
        0.0
    };

    let mut acvf = Vec::with_capacity(lag_max + 1);

    for k in 0..=lag_max {
        let mut sum = 0.0;
        for t in 0..(n - k) {
            sum += (x[t + k] - mean) * (x[t] - mean);
        }
        // Divide by n or (n-k)
        let divisor = if adjusted { (n - k) as f64 } else { n as f64 };
        acvf.push(sum / divisor);
    }

    acvf
}

/// Compute the sample autocorrelation function.
///
/// # Formula
///
/// ρ̂(k) = γ̂(k) / γ̂(0)
///
/// # Arguments
///
/// * `x` - Time series data
/// * `lag_max` - Maximum lag to compute
/// * `acf_type` - Type of ACF to compute (Correlation, Covariance, or Partial)
/// * `demean` - Whether to subtract the mean
/// * `adjusted` - Whether to use (n-k) denominator
///
/// # References
///
/// Box et al. (2015), Chapter 2
pub fn acf(
    x: &[f64],
    lag_max: Option<usize>,
    acf_type: AcfType,
    demean: bool,
    adjusted: bool,
) -> EconResult<AcfResult> {
    refuse_non_finite(x, "x")?;
    let n = x.len();

    if n < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n,
            context: "ACF computation".to_string(),
        });
    }

    // Default lag_max: min(10 * log10(n), n-1), matching R behavior
    let max_lag = lag_max.unwrap_or_else(|| {
        let default = (10.0 * (n as f64).log10()).floor() as usize;
        default.min(n - 1)
    });

    if max_lag >= n {
        return Err(EconError::InvalidSpecification {
            message: format!("lag_max ({}) must be less than n ({})", max_lag, n),
        });
    }

    let values = match acf_type {
        AcfType::Covariance => autocovariance(x, max_lag, demean, adjusted),
        AcfType::Correlation => {
            let acvf = autocovariance(x, max_lag, demean, adjusted);
            let var = acvf[0];
            if var.is_nan() || var <= 0.0 {
                return Err(EconError::InvalidSpecification {
                    message: "Variance is zero or negative; cannot compute autocorrelation"
                        .to_string(),
                });
            }
            acvf.into_iter().map(|g| g / var).collect()
        }
        AcfType::Partial => {
            // Compute PACF via Durbin-Levinson algorithm
            let acf_vals = {
                let acvf = autocovariance(x, max_lag, demean, adjusted);
                let var = acvf[0];
                if var.is_nan() || var <= 0.0 {
                    return Err(EconError::InvalidSpecification {
                        message: "Variance is zero or negative; cannot compute PACF".to_string(),
                    });
                }
                acvf.into_iter().map(|g| g / var).collect::<Vec<_>>()
            };
            let pacf = durbin_levinson(&acf_vals)?;
            // Insert 1.0 at lag 0 for consistency (R includes lag 0 = 1 for ACF)
            let mut result = vec![1.0];
            result.extend(pacf);
            result
        }
    };

    // 95% confidence bound for white noise: ±qnorm(0.975)/√n, as R's plot.acf
    let confidence_bound = if matches!(acf_type, AcfType::Correlation | AcfType::Partial) {
        Some(z_critical(0.95)? / (n as f64).sqrt())
    } else {
        None
    };

    Ok(AcfResult {
        acf_type,
        lags: (0..=max_lag as i32).collect(),
        values,
        n_obs: n,
        series_name: None,
        confidence_bound,
        demeaned: demean,
        adjusted,
    })
}

/// Compute the partial autocorrelation function using the Durbin-Levinson algorithm.
///
/// # Formula (Durbin-Levinson recursion)
///
/// φₙ,ₙ = [ρ(n) - Σₖ₌₁ⁿ⁻¹ φₙ₋₁,ₖ ρ(n-k)] / [1 - Σₖ₌₁ⁿ⁻¹ φₙ₋₁,ₖ ρ(k)]
/// φₙ,ₖ = φₙ₋₁,ₖ - φₙ,ₙ × φₙ₋₁,ₙ₋ₖ  for 1 ≤ k ≤ n-1
///
/// # Arguments
///
/// * `x` - Time series data
/// * `lag_max` - Maximum lag to compute
///
/// # References
///
/// - Durbin, J. (1960). "The Fitting of Time-Series Models"
/// - Brockwell & Davis (1991), Algorithm 8.2.1
pub fn pacf(x: &[f64], lag_max: Option<usize>) -> EconResult<PacfResult> {
    refuse_non_finite(x, "x")?;
    let n = x.len();

    if n < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n,
            context: "PACF computation".to_string(),
        });
    }

    // Default lag_max: min(10 * log10(n), n-1)
    let max_lag = lag_max.unwrap_or_else(|| {
        let default = (10.0 * (n as f64).log10()).floor() as usize;
        default.min(n - 1)
    });

    if max_lag >= n {
        return Err(EconError::InvalidSpecification {
            message: format!("lag_max ({}) must be less than n ({})", max_lag, n),
        });
    }

    // First compute ACF
    let acvf = autocovariance(x, max_lag, true, false);
    let var = acvf[0];
    if var.is_nan() || var <= 0.0 {
        return Err(EconError::InvalidSpecification {
            message: "Variance is zero or negative; cannot compute PACF".to_string(),
        });
    }
    let acf_vals: Vec<f64> = acvf.into_iter().map(|g| g / var).collect();

    // Compute PACF via Durbin-Levinson
    let pacf_vals = durbin_levinson(&acf_vals)?;

    // 95% confidence bound: ±qnorm(0.975)/√n, as R's plot.acf
    let confidence_bound = z_critical(0.95)? / (n as f64).sqrt();

    Ok(PacfResult {
        lags: (1..=max_lag as i32).collect(),
        values: pacf_vals,
        n_obs: n,
        series_name: None,
        confidence_bound,
    })
}

/// Durbin-Levinson algorithm for computing PACF from ACF.
///
/// Given autocorrelations ρ(0), ρ(1), ..., ρ(p), computes the partial
/// autocorrelations φ₁₁, φ₂₂, ..., φₚₚ.
///
/// # References
///
/// Brockwell & Davis (1991), Algorithm 8.2.1
fn durbin_levinson(acf: &[f64]) -> EconResult<Vec<f64>> {
    let p = acf.len() - 1; // acf[0] = ρ(0) = 1
    if p == 0 {
        return Ok(vec![]);
    }

    let mut pacf = Vec::with_capacity(p);
    let mut phi: Vec<f64> = vec![0.0; p];

    // φ₁₁ = ρ(1)
    phi[0] = acf[1];
    pacf.push(phi[0]);

    for n in 2..=p {
        // Compute φₙ,ₙ
        let mut num = acf[n];
        let mut den = 1.0;

        for k in 1..n {
            num -= phi[k - 1] * acf[n - k];
            den -= phi[k - 1] * acf[k];
        }

        if den.abs() < 1e-15 {
            // Denominator near zero indicates numerical issues
            return Err(EconError::Internal(
                "Durbin-Levinson: denominator near zero".to_string(),
            ));
        }

        let phi_nn = num / den;
        pacf.push(phi_nn);

        // Update φₙ,ₖ for k = 1, ..., n-1
        let phi_prev = phi.clone();
        for k in 1..n {
            phi[k - 1] = phi_prev[k - 1] - phi_nn * phi_prev[n - k - 1];
        }
        phi[n - 1] = phi_nn;
    }

    Ok(pacf)
}

/// Compute the sample cross-covariance function.
///
/// # Formula
///
/// γ̂ₓᵧ(k) = (1/n) Σ_{t=1}^{n-|k|} (x_{t+k} - x̄)(y_t - ȳ)  for k ≥ 0
/// γ̂ₓᵧ(k) = (1/n) Σ_{t=1}^{n+k} (x_t - x̄)(y_{t-k} - ȳ)  for k < 0
///
/// # Arguments
///
/// * `x` - First time series
/// * `y` - Second time series
/// * `lag_max` - Maximum lag (computes for -lag_max to +lag_max)
///
/// # References
///
/// R `ccf` documentation
fn cross_covariance(x: &[f64], y: &[f64], lag_max: usize) -> Vec<(i32, f64)> {
    let n = x.len();
    assert_eq!(n, y.len(), "x and y must have the same length");

    let x_mean = x.iter().sum::<f64>() / n as f64;
    let y_mean = y.iter().sum::<f64>() / n as f64;

    let mut ccvf = Vec::with_capacity(2 * lag_max + 1);

    // Negative lags: correlation between x_t and y_{t+|k|}
    for k in (1..=lag_max).rev() {
        let mut sum = 0.0;
        for t in 0..(n - k) {
            sum += (x[t] - x_mean) * (y[t + k] - y_mean);
        }
        ccvf.push((-(k as i32), sum / n as f64));
    }

    // Lag 0
    let mut sum = 0.0;
    for t in 0..n {
        sum += (x[t] - x_mean) * (y[t] - y_mean);
    }
    ccvf.push((0, sum / n as f64));

    // Positive lags: correlation between x_{t+k} and y_t
    for k in 1..=lag_max {
        let mut sum = 0.0;
        for t in 0..(n - k) {
            sum += (x[t + k] - x_mean) * (y[t] - y_mean);
        }
        ccvf.push((k as i32, sum / n as f64));
    }

    ccvf
}

/// Compute the sample cross-correlation function.
///
/// # Formula
///
/// ρ̂ₓᵧ(k) = γ̂ₓᵧ(k) / √(γ̂ₓₓ(0) × γ̂ᵧᵧ(0))
///
/// The lag k value estimates the correlation between x_{t+k} and y_t.
/// - Positive lag k: x leads y (x_{t+k} is correlated with y_t)
/// - Negative lag k: y leads x (x_t is correlated with y_{t-k})
///
/// # Arguments
///
/// * `x` - First time series
/// * `y` - Second time series
/// * `lag_max` - Maximum lag (computes for -lag_max to +lag_max)
/// * `ccf_type` - Type of CCF (Correlation or Covariance)
///
/// # References
///
/// R `ccf` documentation
pub fn ccf(
    x: &[f64],
    y: &[f64],
    lag_max: Option<usize>,
    ccf_type: CcfType,
) -> EconResult<CcfResult> {
    refuse_non_finite(x, "x")?;
    refuse_non_finite(y, "y")?;
    let n = x.len();

    if n != y.len() {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "x and y must have the same length: x has {}, y has {}",
                n,
                y.len()
            ),
        });
    }

    if n < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n,
            context: "CCF computation".to_string(),
        });
    }

    // Default lag_max: min(10 * log10(n), n-1)
    let max_lag = lag_max.unwrap_or_else(|| {
        let default = (10.0 * (n as f64).log10()).floor() as usize;
        default.min(n - 1)
    });

    if max_lag >= n {
        return Err(EconError::InvalidSpecification {
            message: format!("lag_max ({}) must be less than n ({})", max_lag, n),
        });
    }

    let ccvf = cross_covariance(x, y, max_lag);

    let (lags, values): (Vec<i32>, Vec<f64>) = match ccf_type {
        CcfType::Covariance => ccvf.into_iter().unzip(),
        CcfType::Correlation => {
            // Compute variances of x and y
            let var_x = autocovariance(x, 0, true, false)[0];
            let var_y = autocovariance(y, 0, true, false)[0];

            if var_x.is_nan() || var_x <= 0.0 || var_y.is_nan() || var_y <= 0.0 {
                return Err(EconError::InvalidSpecification {
                    message: "One or both series have zero variance".to_string(),
                });
            }

            let normalizer = (var_x * var_y).sqrt();
            ccvf.into_iter()
                .map(|(lag, cov)| (lag, cov / normalizer))
                .unzip()
        }
    };

    // 95% confidence bound for white noise: ±qnorm(0.975)/√n, as R's plot.acf
    let confidence_bound = if matches!(ccf_type, CcfType::Correlation) {
        Some(z_critical(0.95)? / (n as f64).sqrt())
    } else {
        None
    };

    Ok(CcfResult {
        ccf_type,
        lags,
        values,
        n_obs: n,
        x_series: None,
        y_series: None,
        confidence_bound,
    })
}

// ============================================================================
// Dataset-based convenience functions
// ============================================================================

/// Extract a numeric column from a dataset.
fn extract_column(dataset: &Dataset, col_name: &str) -> EconResult<Vec<f64>> {
    numeric_column(dataset.df(), col_name, NaPolicy::Fail).map(|(v, _)| v)
}

/// Run ACF on a dataset column.
///
/// # Arguments
///
/// * `dataset` - Dataset containing the time series
/// * `column` - Name of the column to analyze
/// * `lag_max` - Maximum lag (optional)
/// * `acf_type` - Type of ACF (Correlation, Covariance, or Partial)
///
/// # Example
///
/// ```ignore
/// let result = run_acf(&dataset, "returns", Some(20), AcfType::Correlation)?;
/// println!("{}", result);
/// ```
pub fn run_acf(
    dataset: &Dataset,
    column: &str,
    lag_max: Option<usize>,
    acf_type: AcfType,
) -> EconResult<AcfResult> {
    let x = extract_column(dataset, column)?;
    let mut result = acf(&x, lag_max, acf_type, true, false)?;
    result.series_name = Some(column.to_string());
    Ok(result)
}

/// Run PACF on a dataset column.
///
/// # Arguments
///
/// * `dataset` - Dataset containing the time series
/// * `column` - Name of the column to analyze
/// * `lag_max` - Maximum lag (optional)
///
/// # Example
///
/// ```ignore
/// let result = run_pacf(&dataset, "returns", Some(20))?;
/// println!("{}", result);
/// ```
pub fn run_pacf(dataset: &Dataset, column: &str, lag_max: Option<usize>) -> EconResult<PacfResult> {
    let x = extract_column(dataset, column)?;
    let mut result = pacf(&x, lag_max)?;
    result.series_name = Some(column.to_string());
    Ok(result)
}

/// Run CCF on two dataset columns.
///
/// # Arguments
///
/// * `dataset` - Dataset containing the time series
/// * `x_col` - Name of the first (X) column
/// * `y_col` - Name of the second (Y) column
/// * `lag_max` - Maximum lag (optional)
/// * `ccf_type` - Type of CCF (Correlation or Covariance)
///
/// # Example
///
/// ```ignore
/// let result = run_ccf(&dataset, "x", "y", Some(10), CcfType::Correlation)?;
/// println!("{}", result);
/// ```
pub fn run_ccf(
    dataset: &Dataset,
    x_col: &str,
    y_col: &str,
    lag_max: Option<usize>,
    ccf_type: CcfType,
) -> EconResult<CcfResult> {
    let x = extract_column(dataset, x_col)?;
    let y = extract_column(dataset, y_col)?;
    let mut result = ccf(&x, &y, lag_max, ccf_type)?;
    result.x_series = Some(x_col.to_string());
    result.y_series = Some(y_col.to_string());
    Ok(result)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use p2a_validation::{Check, Tol};

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_acf_white_noise() {
        // White noise should have ACF(0) = 1, ACF(k) ≈ 0 for k > 0
        let x = vec![0.1, -0.3, 0.2, -0.1, 0.4, -0.2, 0.1, -0.3, 0.2, 0.1];
        let result = acf(&x, Some(5), AcfType::Correlation, true, false).unwrap();

        // ACF(0) should be 1
        assert!(approx_eq(result.values[0], 1.0, 1e-10));
        // ACF(k) for k > 0 should be small (within confidence bounds for white noise)
        for &v in &result.values[1..] {
            assert!(v.abs() < 1.0, "ACF values should be in [-1, 1]");
        }
    }

    #[test]
    fn test_acf_autocovariance() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = acf(&x, Some(2), AcfType::Covariance, true, false).unwrap();

        // Manually compute: mean = 3, variance = (4+1+0+1+4)/5 = 2.0
        // ACVF(0) = variance = 2.0
        assert!(approx_eq(result.values[0], 2.0, 1e-10));

        // ACVF(1) = (1*(-2) + (-1)*(-1) + 0*0 + 1*1) / 5 = (-2+1+0+1)/5 = 0
        // Wait, let's recalculate: x - mean = [-2, -1, 0, 1, 2]
        // ACVF(1) = sum of (x[t+1]-mean)(x[t]-mean) for t=0..3, divided by 5
        // = ((-1)*(-2) + 0*(-1) + 1*0 + 2*1) / 5 = (2 + 0 + 0 + 2) / 5 = 0.8
        assert!(approx_eq(result.values[1], 0.8, 1e-10));
    }

    #[test]
    fn test_acf_adjusted() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let unadj = acf(&x, Some(2), AcfType::Covariance, true, false).unwrap();
        let adj = acf(&x, Some(2), AcfType::Covariance, true, true).unwrap();

        // Adjusted divides by (n-k) instead of n
        // For ACVF(1): unadjusted uses /5, adjusted uses /4
        assert!(approx_eq(adj.values[1], unadj.values[1] * 5.0 / 4.0, 1e-10));
    }

    #[test]
    fn test_pacf_basic() {
        // Test that PACF computes values within the expected range
        // and that PACF(1) equals ACF(1) for any series (this is a mathematical property)
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let acf_result = acf(&x, Some(5), AcfType::Correlation, true, false).unwrap();
        let pacf_result = pacf(&x, Some(5)).unwrap();

        // PACF(1) should equal ACF(1) exactly (mathematical property of Durbin-Levinson)
        assert!(
            (pacf_result.values[0] - acf_result.values[1]).abs() < 1e-10,
            "PACF(1) = {} should equal ACF(1) = {}",
            pacf_result.values[0],
            acf_result.values[1]
        );

        // All PACF values should be in [-1, 1]
        for (i, &v) in pacf_result.values.iter().enumerate() {
            assert!(
                v.abs() <= 1.0,
                "PACF({}) = {} should be in [-1, 1]",
                i + 1,
                v
            );
        }
    }

    #[test]
    fn test_ccf_lag_zero() {
        // CCF at lag 0 should equal correlation coefficient
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];

        let result = ccf(&x, &y, Some(2), CcfType::Correlation).unwrap();

        // Find lag 0 value
        let lag0_idx = result.lags.iter().position(|&l| l == 0).unwrap();
        let ccf_0 = result.values[lag0_idx];

        // Compute Pearson correlation manually
        let n = x.len() as f64;
        let x_mean = x.iter().sum::<f64>() / n;
        let y_mean = y.iter().sum::<f64>() / n;
        let cov: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - x_mean) * (b - y_mean))
            .sum::<f64>()
            / n;
        let var_x: f64 = x.iter().map(|a| (a - x_mean).powi(2)).sum::<f64>() / n;
        let var_y: f64 = y.iter().map(|b| (b - y_mean).powi(2)).sum::<f64>() / n;
        let r = cov / (var_x.sqrt() * var_y.sqrt());

        assert!(
            approx_eq(ccf_0, r, 1e-10),
            "CCF(0) = {} should equal r = {}",
            ccf_0,
            r
        );
    }

    #[test]
    fn test_ccf_symmetry() {
        // CCF(x, y, k) should NOT equal CCF(x, y, -k) in general
        // But CCF(x, y, k) = CCF(y, x, -k)
        let x = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let y = vec![0.5, 1.0, 2.0, 3.0, 2.5];

        let ccf_xy = ccf(&x, &y, Some(2), CcfType::Correlation).unwrap();
        let ccf_yx = ccf(&y, &x, Some(2), CcfType::Correlation).unwrap();

        // CCF(x, y, k) = CCF(y, x, -k)
        for (i, &lag) in ccf_xy.lags.iter().enumerate() {
            let j = ccf_yx.lags.iter().position(|&l| l == -lag).unwrap();
            assert!(
                approx_eq(ccf_xy.values[i], ccf_yx.values[j], 1e-10),
                "CCF(x,y,{}) = {} should equal CCF(y,x,{}) = {}",
                lag,
                ccf_xy.values[i],
                -lag,
                ccf_yx.values[j]
            );
        }
    }

    #[test]
    fn test_default_lag_max() {
        let x: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
        let result = acf(&x, None, AcfType::Correlation, true, false).unwrap();

        // Default should be floor(10 * log10(100)) = floor(20) = 20
        assert_eq!(result.lags.len(), 21); // 0 to 20
    }

    #[test]
    fn test_confidence_bounds() {
        let x: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
        let result = acf(&x, Some(10), AcfType::Correlation, true, false).unwrap();

        // 95% CI: ±qnorm(0.975)/√100
        let expected = z_critical(0.95).unwrap() / 10.0;
        assert!(approx_eq(result.confidence_bound.unwrap(), expected, 1e-10));
    }

    #[cfg(feature = "spectral-analysis")]
    fn spectrum_refusals(ds: &Dataset, col: &str) -> Vec<(&'static str, EconError)> {
        use crate::stats::{SpectrumConfig, run_spectrum, run_spectrum_ar};
        vec![
            (
                "spectrum",
                run_spectrum(ds, col, SpectrumConfig::default()).unwrap_err(),
            ),
            (
                "spectrum_ar",
                run_spectrum_ar(ds, col, None, None).unwrap_err(),
            ),
        ]
    }

    #[cfg(not(feature = "spectral-analysis"))]
    fn spectrum_refusals(_: &Dataset, _: &str) -> Vec<(&'static str, EconError)> {
        Vec::new()
    }

    /// RF-34: the ACF, PACF, CCF, Box, PP and spectrum dataset readers refuse a column with
    /// missing values (R's na.action = na.fail) and name the number of missing cells, R's
    /// sum(is.na(col)): the rows minus the non-missing count the case records (case
    /// stat_two_sample_na; x, xi and yi have missing cells, id has none). They reported 1.
    #[test]
    fn test_validate_na_ts_refusals_report_null_count() {
        use crate::stats::{BoxTestType, run_box_test, run_pp_test};
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_two_sample_na");
        let rows = ds.nrows();
        let mut check = Check::new(&case);
        for (col, key) in [("x", "n_x"), ("xi", "n_xi"), ("yi", "n_yi")] {
            let mut refusals: Vec<(&str, EconError)> = vec![
                (
                    "acf",
                    run_acf(&ds, col, Some(3), AcfType::Correlation).unwrap_err(),
                ),
                ("pacf", run_pacf(&ds, col, Some(3)).unwrap_err()),
                (
                    "ccf",
                    run_ccf(&ds, col, "id", Some(3), CcfType::Correlation).unwrap_err(),
                ),
                (
                    "box",
                    run_box_test(&ds, col, Some(3), BoxTestType::LjungBox, 0).unwrap_err(),
                ),
                ("pp", run_pp_test(&ds, col, false).unwrap_err()),
            ];
            refusals.extend(spectrum_refusals(&ds, col));
            for (reader, e) in refusals {
                match e {
                    EconError::NullValues { column, count } => {
                        assert_eq!(column, col, "{reader}");
                        // R: sum(!is.na(col)) = rows - sum(is.na(col))
                        check.scalar(key, (rows - count) as f64, Tol::exact());
                    }
                    other => panic!("{reader} {col}: expected NullValues, got {other}"),
                }
            }
        }
        check.finish();
    }

    /// TOOL-11: NaN or ±Inf gave all-NaN results (acf, pacf) and a panic in the MCP ccf handler. R: na.fail.
    #[test]
    fn test_acf_rejects_non_finite() {
        let ok = [1.0, 2.3, 1.7, 3.1, 2.2, 1.4, 2.8];
        let mut nan = ok;
        nan[2] = f64::NAN;
        let mut inf = ok;
        inf[4] = f64::NEG_INFINITY;
        for t in [AcfType::Correlation, AcfType::Covariance, AcfType::Partial] {
            assert!(
                matches!(acf(&nan, Some(2), t, true, false), Err(EconError::NullValues { ref column, count: 1 }) if column == "x")
            );
            assert!(
                matches!(acf(&inf, Some(2), t, true, false), Err(EconError::NonFiniteValues { ref column, count: 1 }) if column == "x")
            );
        }
        assert!(matches!(
            pacf(&nan, Some(2)),
            Err(EconError::NullValues { .. })
        ));
        for t in [CcfType::Correlation, CcfType::Covariance] {
            assert!(
                matches!(ccf(&ok, &nan, Some(2), t), Err(EconError::NullValues { ref column, .. }) if column == "y")
            );
            assert!(
                matches!(ccf(&inf, &ok, Some(2), t), Err(EconError::NonFiniteValues { ref column, .. }) if column == "x")
            );
        }
    }

    /// TOOL-11: a NaN cell in a dataset column is refused with the column name (the only allowed inline NaN fixture, T2 2.2);
    /// integer columns are accepted.
    #[test]
    fn test_run_acf_refuses_nan_cell() {
        let ds = Dataset::new(
            polars::prelude::df! {
                "x" => [1.0, 2.3, f64::NAN, 3.1, 2.2, 1.4, 2.8],
                "y" => [0.4, 1.1, 0.9, 1.8, 1.2, 0.7, 1.5],
                "i" => [3i64, 1, 4, 1, 5, 9, 2],
            }
            .unwrap(),
        );
        assert!(
            matches!(run_acf(&ds, "x", Some(2), AcfType::Correlation), Err(EconError::NullValues { ref column, count: 1 }) if column == "x")
        );
        assert!(matches!(
            run_pacf(&ds, "x", Some(2)),
            Err(EconError::NullValues { .. })
        ));
        assert!(
            matches!(run_ccf(&ds, "y", "x", Some(2), CcfType::Correlation), Err(EconError::NullValues { ref column, .. }) if column == "x")
        );
        assert!(run_acf(&ds, "i", Some(2), AcfType::Correlation).is_ok());
    }

    /// STAT-62 (acf), TOOL-11, R7: acf, autocovariance, pacf and ccf pinned to R; the white-noise bound is qnorm(0.975)/sqrt(n).
    #[test]
    fn test_validate_acf_pacf_ccf_t3() {
        let (case, ds) = crate::test_support::load_case("stat_ts_acf_spline_t3");
        let a = run_acf(&ds, "x", None, AcfType::Correlation).unwrap();
        let cv = run_acf(&ds, "x", None, AcfType::Covariance).unwrap();
        let p = run_pacf(&ds, "x", None).unwrap();
        let c = run_ccf(&ds, "x", "y", Some(5), CcfType::Correlation).unwrap();
        let lags: Vec<f64> = c.lags.iter().map(|&l| l as f64).collect();
        Check::new(&case)
            .vector("acf", &a.values, Tol::stat(64))
            .scalar("acf_lag_max", *a.lags.last().unwrap() as f64, Tol::exact())
            .vector("acvf", &cv.values, Tol::stat(64))
            .vector("pacf", &p.values, Tol::stat(64))
            .vector("ccf", &c.values, Tol::stat(64))
            .vector("ccf_lags", &lags, Tol::exact())
            .scalar("acf_ci_bound", a.confidence_bound.unwrap(), Tol::stat(64))
            .finish();
        assert_eq!(p.confidence_bound, a.confidence_bound.unwrap());
        assert_eq!(c.confidence_bound, a.confidence_bound);
    }

    #[test]
    fn test_acf_type_parses_aliases_and_rejects_unknown() {
        assert_eq!(AcfType::choices().validate(), Ok(()));
        assert_eq!(CcfType::choices().validate(), Ok(()));
        assert_eq!("PACF".parse::<AcfType>().unwrap(), AcfType::Partial);
        assert_eq!("cov".parse::<AcfType>().unwrap(), AcfType::Covariance);
        assert_eq!(
            "correlation".parse::<CcfType>().unwrap(),
            CcfType::Correlation
        );
        assert!("partial".parse::<CcfType>().is_err());
        assert!("spectral".parse::<AcfType>().is_err());
        assert_eq!(AcfType::Partial.to_string(), "Partial Autocorrelation"); // existing Display kept (no_display)
    }
}
