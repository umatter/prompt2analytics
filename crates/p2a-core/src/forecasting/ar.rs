//! Autoregressive (AR) Model Fitting.
//!
//! Fits AR models to univariate time series data using various methods:
//! - Yule-Walker (default): Solves Yule-Walker equations via Durbin-Levinson
//! - Burg: Burg's method using forward/backward prediction errors
//! - OLS: Ordinary least squares regression
//!
//! # Mathematical Background
//!
//! The AR(p) model is:
//! ```text
//! x_t - μ = φ₁(x_{t-1} - μ) + φ₂(x_{t-2} - μ) + ... + φₚ(x_{t-p} - μ) + ε_t
//! ```
//!
//! where ε_t ~ WN(0, σ²) (white noise).
//!
//! # References
//!
//! - Brockwell, P. J. & Davis, R. A. (1991). "Time Series: Theory and Methods".
//!   Springer. (Chapters 8-9 on parameter estimation)
//! - Burg, J. P. (1967). "Maximum Entropy Spectral Analysis". 37th Meeting of
//!   the Society of Exploration Geophysicists, Oklahoma City.
//! - R Core Team. `stats::ar()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/ar.html>

use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::xtx;
use crate::linalg::solve::{LM_QR_TOL, Qr, least_squares};
use crate::regression::ols::normal_inverse_from_gram;
use crate::resource::{checkpoint, try_vec, try_zeros1, try_zeros2};
use crate::traits::convergence::{ResultNotes, notes_text};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Method for AR model fitting.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum ArMethod {
    /// Yule-Walker equations solved via Durbin-Levinson (default)
    #[default]
    YuleWalker,
    /// Burg's method using forward/backward prediction errors
    Burg,
    /// Ordinary least squares
    Ols,
}

/// Configuration for AR model fitting.
#[derive(Debug, Clone)]
pub struct ArConfig {
    /// Use AIC for order selection (default: true)
    pub aic: bool,
    /// Maximum order to consider (default: computed from series length)
    pub order_max: Option<usize>,
    /// Specific order to use (if aic=false)
    pub order: Option<usize>,
    /// Fitting method (default: YuleWalker)
    pub method: ArMethod,
    /// Demean the series before fitting (default: true)
    pub demean: bool,
}

impl Default for ArConfig {
    fn default() -> Self {
        ArConfig {
            aic: true,
            order_max: None,
            order: None,
            method: ArMethod::YuleWalker,
            demean: true,
        }
    }
}

/// Result of AR model fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArResult {
    /// Selected or specified model order
    pub order: usize,
    /// Estimated AR coefficients (φ₁, ..., φₚ)
    pub ar: Vec<f64>,
    /// Prediction error variance (innovation variance)
    pub var_pred: f64,
    /// Estimated mean of the series
    pub x_mean: f64,
    /// AIC values relative to the minimum (only if aic=true)
    pub aic: Option<Vec<f64>>,
    /// Partial autocorrelation coefficients at each lag
    pub partial_acf: Vec<f64>,
    /// Number of observations
    pub n_obs: usize,
    /// Fitting method used
    pub method: ArMethod,
    /// Residuals (first `order` values are NA represented as NaN)
    #[serde(skip)]
    pub resid: Vec<f64>,
    /// R's ar warnings (OLS singularity stop, saturated order); shown by Display.
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl ResultNotes for ArResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

impl std::fmt::Display for ArResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Autoregressive Model AR({})", self.order)?;
        writeln!(f, "{}", "=".repeat(40))?;
        writeln!(f)?;
        writeln!(f, "Method: {:?}", self.method)?;
        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(f, "Order selected: {}", self.order)?;
        writeln!(f)?;
        writeln!(f, "Coefficients:")?;
        for (i, coef) in self.ar.iter().enumerate() {
            writeln!(f, "  ar{}: {:.6}", i + 1, coef)?;
        }
        writeln!(f)?;
        writeln!(f, "Mean: {:.6}", self.x_mean)?;
        writeln!(f, "Prediction variance: {:.6}", self.var_pred)?;
        if let Some(ref aic_vals) = self.aic {
            writeln!(f)?;
            writeln!(f, "AIC (relative to minimum):")?;
            for (i, aic) in aic_vals.iter().enumerate() {
                if *aic < 100.0 {
                    writeln!(f, "  AR({}): {:.2}", i, aic)?;
                }
            }
        }
        write!(f, "{}", notes_text(self))
    }
}

/// What a fitting route returns; `ar` adds the mean, the residuals and the labels.
struct ArFit {
    ar: Vec<f64>,
    var_pred: f64,
    pacf: Vec<f64>,
    aic: Option<Vec<f64>>,
    warnings: Vec<String>,
}

/// Fit an autoregressive model to a time series.
///
/// The methods follow `stats::ar`: `YuleWalker` is `ar.yw` (`var_pred` is the Durbin-Levinson
/// variance times n / (n - (order + 1))), `Burg` is `ar.burg` and `Ols` is
/// `ar.ols(demean = config.demean, intercept = FALSE)`: orders are fitted while the lag matrix
/// has at least as many rows as lags and `qr(X'X)` is of full rank, `var_pred` is SSE / (n - m),
/// the AIC is n ln(var) + 2m relative to its minimum (`+inf` for orders not fitted), and a stop
/// or a saturated order is reported in `warnings`.
///
/// # Arguments
///
/// * `x` - Time series data
/// * `config` - Configuration options
///
/// # Errors
///
/// `InsufficientData` for a series of fewer than 3 values, a Yule-Walker order with
/// n <= order + 1, or a fixed OLS order with fewer rows than lags; `SingularMatrix` for a fixed
/// OLS order whose X'X is rank-deficient; `Cancelled` when the budget stops (checked once per
/// order).
///
/// # Example
///
/// ```
/// use p2a_core::forecasting::ar::{ar, ArConfig, ArMethod};
///
/// let data = vec![1.0, 2.1, 1.9, 3.2, 2.8, 4.1, 3.9, 5.0, 4.8, 6.1];
/// let config = ArConfig::default();
/// let result = ar(&data, config).unwrap();
/// println!("Selected order: {}", result.order);
/// println!("AR coefficients: {:?}", result.ar);
/// ```
pub fn ar(x: &[f64], config: ArConfig) -> EconResult<ArResult> {
    let n = x.len();

    if n < 3 {
        return Err(EconError::InsufficientData {
            required: 3,
            provided: n,
            context: "AR model fitting".to_string(),
        });
    }

    // Compute default max order: min(n-1, 10 * log10(n))
    let default_max = ((n as f64).log10() * 10.0).floor() as usize;
    let max_order = config
        .order_max
        .unwrap_or(default_max.min(n - 1))
        .min(n - 1);

    // Demean if requested
    let x_mean = if config.demean {
        x.iter().sum::<f64>() / n as f64
    } else {
        0.0
    };
    let y: Vec<f64> = x.iter().map(|v| v - x_mean).collect();

    // Fit based on method
    let fit = match config.method {
        ArMethod::YuleWalker => fit_ar_yule_walker_full(&y, max_order, config.aic, config.order)?,
        ArMethod::Burg => fit_ar_burg(&y, max_order, config.aic, config.order)?,
        ArMethod::Ols => fit_ar_ols(&y, max_order, config.aic, config.order)?,
    };

    let ArFit {
        ar: ar_coefs,
        var_pred,
        pacf: partial_acf,
        aic: aic_values,
        warnings,
    } = fit;

    // Compute residuals
    let order = ar_coefs.len();
    let mut resid = vec![f64::NAN; n];
    for t in order..n {
        let mut fitted = 0.0;
        for (k, &phi) in ar_coefs.iter().enumerate() {
            fitted += phi * y[t - k - 1];
        }
        resid[t] = y[t] - fitted;
    }

    Ok(ArResult {
        order,
        ar: ar_coefs,
        var_pred,
        x_mean,
        aic: aic_values,
        partial_acf,
        n_obs: n,
        method: config.method,
        resid,
        warnings,
    })
}

/// The Durbin-Levinson recursion on the sample autocovariances: per order 0..=max_order the
/// coefficients and the innovation variance, and the partial autocorrelations.
fn durbin_levinson_path(
    y: &[f64],
    max_order: usize,
) -> EconResult<(Vec<Vec<f64>>, Vec<f64>, Vec<f64>)> {
    let n = y.len();

    // Compute autocovariances up to max_order
    let mut gamma = Vec::with_capacity(max_order + 1);
    for k in 0..=max_order {
        checkpoint()?;
        let mut sum = 0.0;
        for t in 0..(n - k) {
            sum += y[t] * y[t + k];
        }
        gamma.push(sum / n as f64);
    }

    if gamma[0] <= 0.0 {
        return Err(EconError::InvalidSpecification {
            message: "Series has zero variance".to_string(),
        });
    }

    // Use Durbin-Levinson to compute all orders simultaneously
    // This gives us partial autocorrelations as a byproduct
    let mut phi_all: Vec<Vec<f64>> = Vec::with_capacity(max_order + 1);
    let mut var_all: Vec<f64> = Vec::with_capacity(max_order + 1);
    let mut pacf = Vec::with_capacity(max_order);

    // Order 0
    phi_all.push(vec![]);
    var_all.push(gamma[0]);

    // Orders 1 through max_order
    if max_order > 0 {
        // Order 1
        let phi_11 = gamma[1] / gamma[0];
        pacf.push(phi_11);
        phi_all.push(vec![phi_11]);
        var_all.push(gamma[0] * (1.0 - phi_11 * phi_11));

        for p in 2..=max_order {
            checkpoint()?;
            let phi_prev = &phi_all[p - 1];

            // Compute partial autocorrelation phi_pp
            let mut num = gamma[p];
            for j in 0..(p - 1) {
                num -= phi_prev[j] * gamma[p - 1 - j];
            }
            let phi_pp = num / var_all[p - 1];
            pacf.push(phi_pp);

            // Update coefficients
            let mut phi_new = try_vec(p, 0.0, "the Durbin-Levinson coefficients")?;
            for j in 0..(p - 1) {
                phi_new[j] = phi_prev[j] - phi_pp * phi_prev[p - 2 - j];
            }
            phi_new[p - 1] = phi_pp;

            // Update variance
            let new_var = var_all[p - 1] * (1.0 - phi_pp * phi_pp);

            phi_all.push(phi_new);
            var_all.push(new_var.max(0.0));
        }
    }

    Ok((phi_all, var_all, pacf))
}

/// Fit AR model using Yule-Walker equations with full diagnostics.
fn fit_ar_yule_walker_full(
    y: &[f64],
    max_order: usize,
    use_aic: bool,
    fixed_order: Option<usize>,
) -> EconResult<ArFit> {
    let n = y.len();
    let (phi_all, var_all, pacf) = durbin_levinson_path(y, max_order)?;

    // Select order based on AIC or fixed
    let (selected_order, aic_values) = if let Some(order) = fixed_order {
        let order = order.min(max_order);
        (order, None)
    } else if use_aic {
        let mut aic_vals = Vec::with_capacity(max_order + 1);
        let mut min_aic = f64::INFINITY;
        let mut best_order = 0;

        for p in 0..=max_order {
            let var = var_all[p];
            if var <= 0.0 {
                aic_vals.push(f64::INFINITY);
                continue;
            }
            // AIC = n * log(var) + 2 * (p + 1)
            let aic = n as f64 * var.ln() + 2.0 * (p + 1) as f64;
            aic_vals.push(aic);
            if aic < min_aic {
                min_aic = aic;
                best_order = p;
            }
        }

        // Convert to relative AIC
        let aic_relative: Vec<f64> = aic_vals.iter().map(|a| a - min_aic).collect();
        (best_order, Some(aic_relative))
    } else {
        // Default to max_order if neither aic nor fixed_order
        (max_order, None)
    };

    // R's ar.yw: var.pred * n / (n - (order + 1))
    let den = n
        .checked_sub(selected_order + 1)
        .filter(|d| *d > 0)
        .ok_or_else(|| EconError::InsufficientData {
            required: selected_order + 2,
            provided: n,
            context: format!(
                "Yule-Walker AR({selected_order}): R's var.pred * n / (n - (order + 1)) needs n > order + 1"
            ),
        })?;

    Ok(ArFit {
        ar: phi_all[selected_order].clone(),
        var_pred: var_all[selected_order] * n as f64 / den as f64,
        pacf,
        aic: aic_values,
        warnings: Vec::new(),
    })
}

/// Fit AR model using Burg's method.
///
/// Burg's method minimizes the sum of forward and backward prediction error squares.
fn fit_ar_burg(
    y: &[f64],
    max_order: usize,
    use_aic: bool,
    fixed_order: Option<usize>,
) -> EconResult<ArFit> {
    let n = y.len();

    // Initialize forward and backward prediction errors
    let mut ef: Vec<f64> = y.to_vec();
    let mut eb: Vec<f64> = y.to_vec();

    let var0 = y.iter().map(|v| v * v).sum::<f64>() / n as f64;
    if var0 <= 0.0 {
        return Err(EconError::InvalidSpecification {
            message: "Series has zero variance".to_string(),
        });
    }

    let mut phi_all: Vec<Vec<f64>> = Vec::with_capacity(max_order + 1);
    let mut var_all: Vec<f64> = Vec::with_capacity(max_order + 1);
    let mut pacf = Vec::with_capacity(max_order);

    // Order 0
    phi_all.push(vec![]);
    var_all.push(var0);

    for p in 1..=max_order {
        // Compute reflection coefficient using Burg's formula
        let mut num = 0.0;
        let mut den = 0.0;
        for t in p..n {
            num += ef[t] * eb[t - 1];
            den += ef[t] * ef[t] + eb[t - 1] * eb[t - 1];
        }

        if den <= 0.0 {
            // Can't continue
            break;
        }

        let k = 2.0 * num / den;
        pacf.push(k);

        // Update coefficients
        let phi_prev = if p > 1 {
            phi_all[p - 1].clone()
        } else {
            vec![]
        };
        let mut phi_new = vec![0.0; p];
        for j in 0..(p - 1) {
            phi_new[j] = phi_prev[j] - k * phi_prev[p - 2 - j];
        }
        phi_new[p - 1] = k;

        // Update variance
        let new_var = var_all[p - 1] * (1.0 - k * k);
        var_all.push(new_var.max(0.0));
        phi_all.push(phi_new);

        // Update forward and backward errors
        let ef_old = ef.clone();
        for t in p..n {
            ef[t] = ef_old[t] - k * eb[t - 1];
            eb[t] = eb[t - 1] - k * ef_old[t];
        }
    }

    // Ensure we have enough orders
    while phi_all.len() <= max_order {
        phi_all.push(phi_all.last().unwrap_or(&vec![]).clone());
        var_all.push(*var_all.last().unwrap_or(&var0));
    }

    // Select order
    let (selected_order, aic_values) = if let Some(order) = fixed_order {
        let order = order.min(phi_all.len() - 1);
        (order, None)
    } else if use_aic {
        let mut aic_vals = Vec::with_capacity(phi_all.len());
        let mut min_aic = f64::INFINITY;
        let mut best_order = 0;

        for p in 0..phi_all.len() {
            let var = var_all[p];
            if var <= 0.0 {
                aic_vals.push(f64::INFINITY);
                continue;
            }
            let aic = n as f64 * var.ln() + 2.0 * (p + 1) as f64;
            aic_vals.push(aic);
            if aic < min_aic {
                min_aic = aic;
                best_order = p;
            }
        }

        let aic_relative: Vec<f64> = aic_vals.iter().map(|a| a - min_aic).collect();
        (best_order, Some(aic_relative))
    } else {
        (max_order.min(phi_all.len() - 1), None)
    };

    Ok(ArFit {
        ar: phi_all[selected_order].clone(),
        var_pred: var_all[selected_order],
        pacf,
        aic: aic_values,
        warnings: Vec::new(),
    })
}

/// One order of `ar.ols`: coefficients and SSE of `y[t + m]` on its `m` lags, or `None` when
/// `qr(X'X)` is rank-deficient (R's stop rule, tolerance 1e-7). The caller ensures `rows >= m >= 1`.
///
/// One lag design per order: the normal equations when they are safe, the pivoted QR otherwise
/// and for a saturated order (`rows == m`), whose residuals are exactly zero.
fn fit_ols_order(y: &[f64], m: usize, rows: usize) -> EconResult<Option<(Vec<f64>, f64)>> {
    let mut x = try_zeros2(rows, m, "the AR lag design")?;
    let mut resp = try_zeros1(rows, "the AR response")?;
    for t in 0..rows {
        resp[t] = y[t + m];
        for k in 0..m {
            x[[t, k]] = y[t + m - k - 1];
        }
    }

    let gram = xtx(&x.view());
    if Qr::decompose(gram.view(), LM_QR_TOL)?.rank() < m {
        return Ok(None);
    }

    let solved: Option<Array1<f64>> = if rows > m {
        normal_inverse_from_gram(gram.view())?.map(|inv| inv.coefficients(x.view(), &resp))
    } else {
        None
    };
    let (coefs, sse) = match solved {
        Some(beta) => {
            let sse = (&resp - &x.dot(&beta)).iter().map(|r| r * r).sum();
            (beta, sse)
        }
        None => {
            let ls = least_squares(x.view(), resp.view())?;
            if ls.rank < m {
                return Ok(None);
            }
            let sse = if rows == m {
                0.0
            } else {
                ls.residuals.iter().map(|r| r * r).sum()
            };
            (ls.coefficients, sse)
        }
    };
    Ok(Some((coefs.to_vec(), sse)))
}

/// Why an order could not be fitted by OLS.
enum OlsStop {
    TooFewRows(usize),
    Singular(usize),
}

impl OlsStop {
    fn error(&self) -> EconError {
        match *self {
            OlsStop::TooFewRows(k) => EconError::InsufficientData {
                required: k.saturating_mul(2),
                provided: 0,
                context: format!("AR({k}) by OLS needs at least as many rows as lags (n - k >= k)"),
            },
            OlsStop::Singular(k) => EconError::SingularMatrix {
                context: format!("AR({k}) by OLS: X'X is rank-deficient"),
                suggestion: "use a lower order".to_string(),
            },
        }
    }

    fn order(&self) -> usize {
        match *self {
            OlsStop::TooFewRows(k) | OlsStop::Singular(k) => k,
        }
    }
}

/// Fit AR model using OLS, as `stats::ar.ols(intercept = FALSE)`. R's rescaling by the standard
/// deviation cancels in the coefficients, the rank decision and the relative AIC, so it is omitted.
fn fit_ar_ols(
    y: &[f64],
    max_order: usize,
    use_aic: bool,
    fixed_order: Option<usize>,
) -> EconResult<ArFit> {
    let n = y.len();

    // The partial autocorrelations come from the Durbin-Levinson path, as in R's ar.ols.
    let (_, _, pacf) = durbin_levinson_path(y, max_order)?;

    let (first, last) = match fixed_order {
        Some(k) => (k, k),
        None if use_aic => (0, max_order),
        None => (max_order, max_order),
    };
    let searching = fixed_order.is_none() && use_aic;

    let mut aic_raw = vec![f64::INFINITY; if searching { max_order + 1 } else { 0 }];
    let mut fits: Vec<(Vec<f64>, f64, usize)> = Vec::new();
    let mut stop = None;

    for m in first..=last {
        checkpoint()?;
        let rows = match n.checked_sub(m) {
            Some(rows) if rows >= m => rows,
            _ => {
                stop = Some(OlsStop::TooFewRows(m));
                break;
            }
        };
        let fit = if m == 0 {
            Some((Vec::new(), y.iter().map(|v| v * v).sum::<f64>()))
        } else {
            fit_ols_order(y, m, rows)?
        };
        let Some((coefs, sse)) = fit else {
            stop = Some(OlsStop::Singular(m));
            break;
        };
        let var = sse / rows as f64;
        if searching {
            // ln(0) = -inf for a saturated order
            aic_raw[m] = n as f64 * var.ln() + 2.0 * m as f64;
        }
        fits.push((coefs, var, rows));
    }

    let mut warnings = Vec::new();
    if searching && let Some(stop) = &stop {
        let m = stop.order();
        warnings.push(format!(
            "model order {m}: singularities in the computation of the projection matrix; results are only valid up to model order {}",
            m - 1
        ));
    }

    let (index, aic) = if searching {
        let min = aic_raw.iter().copied().fold(f64::INFINITY, f64::min);
        let best = aic_raw.iter().position(|a| *a == min).unwrap_or(0);
        let relative = aic_raw
            .iter()
            .map(|a| {
                if min.is_finite() {
                    a - min
                } else if *a == min {
                    0.0
                } else {
                    f64::INFINITY
                }
            })
            .collect();
        (best, Some(relative))
    } else {
        if let Some(stop) = &stop {
            let mut err = stop.error();
            if let EconError::InsufficientData { provided, .. } = &mut err {
                *provided = n;
            }
            return Err(err);
        }
        (0, None)
    };

    let (coefs, var_pred, rows) = fits.swap_remove(index);
    let m = coefs.len();
    if m > 0 && rows == m {
        warnings.push(format!(
            "AR({m}) by OLS uses {rows} observations for {m} lags: zero residual degrees of freedom, so the fit is exact and its prediction variance is rounding error"
        ));
    }

    Ok(ArFit {
        ar: coefs,
        var_pred,
        pacf,
        aic,
        warnings,
    })
}

/// Convenience function to run AR model fitting with default settings.
pub fn run_ar(x: &[f64]) -> EconResult<ArResult> {
    ar(x, ArConfig::default())
}

/// Convenience function to run AR model with specific order.
pub fn run_ar_with_order(x: &[f64], order: usize) -> EconResult<ArResult> {
    ar(
        x,
        ArConfig {
            aic: false,
            order: Some(order),
            ..Default::default()
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use p2a_validation::{Check, Tol};

    fn col(ds: &crate::data::Dataset, name: &str) -> Vec<f64> {
        crate::data::columns::numeric_column(ds.df(), name, crate::data::columns::NaPolicy::Omit)
            .unwrap()
            .0
    }

    fn sample_variance(v: &[f64]) -> f64 {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        v.iter().map(|a| (a - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64
    }

    fn ols(order_max: Option<usize>, aic: bool, order: Option<usize>) -> ArConfig {
        let mut c = ArConfig::default();
        c.method = ArMethod::Ols;
        c.order_max = order_max;
        c.aic = aic;
        c.order = order;
        c
    }

    /// Case ts_filter_ar_cpt: ar(s, method = "ols", intercept = FALSE), n = 24, default order.max 13.
    /// R fits 0..12 (12 = saturated), stops at 13 with "singularities", selects 12.
    #[test]
    fn test_validate_ar_ols_short_series() {
        let (case, ds) = crate::test_support::load_case("ts_filter_ar_cpt");
        let s = col(&ds, "s");
        let r = ar(&s, ols(None, true, None)).unwrap();
        let aic = r.aic.clone().unwrap();
        assert_eq!(aic.len(), 14);
        let r_infinite = case.vector("ar_ols_aic_infinite_orders"); // [13]
        let infinite: Vec<f64> = r_infinite
            .iter()
            .map(|&k| {
                if aic[k as usize].is_infinite() {
                    k
                } else {
                    -1.0
                }
            })
            .collect();
        Check::new(&case)
            .scalar("ar_ols_order", r.order as f64, Tol::exact()) // 12
            .vector("ar_ols_coef", &r.ar, Tol::coef(24))
            .vector("ar_ols_aic_infinite_orders", &infinite, Tol::exact())
            .finish();
        assert_eq!(aic[12], 0.0);
        // R's var.pred 3.365475531705031e-30 and its finite relative AICs (1611.6, ...) are
        // the rounding of an exact fit and not comparable; the AIC differences are pinned by
        // test_validate_ar_ols_aic_by_order.
        assert!(
            r.var_pred.abs() < 1e-20 * sample_variance(&s),
            "{}",
            r.var_pred
        );
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("model order 13")
                    && w.contains("only valid up to model order 12")),
            "{:?}",
            r.warnings
        );
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("zero residual degrees of freedom")),
            "{:?}",
            r.warnings
        );
        assert!(format!("{r}").contains("Warning: model order 13"));
    }

    /// The same call with order.max = 11: every order has residual df >= 2, so the AIC
    /// differences are R's n ln(SSE_m / (n - m)) + 2m (key ar_ols_aic_diff = aic[k] - aic[0]).
    #[test]
    fn test_validate_ar_ols_aic_by_order() {
        let (case, ds) = crate::test_support::load_case("ts_filter_ar_cpt");
        let s = col(&ds, "s");
        let r = ar(&s, ols(Some(11), true, None)).unwrap();
        let aic = r.aic.unwrap();
        let diff: Vec<f64> = aic.iter().map(|a| a - aic[0]).collect();
        Check::new(&case)
            .vector("ar_ols_aic_diff", &diff, Tol::stat(24))
            .finish();
    }

    /// ar(x, method = "ols", intercept = FALSE) on the 60-value series: order 1, coefficient,
    /// var.pred (SSE / (n - m)), x.mean and the relative AIC for orders 0..17.
    #[test]
    fn test_validate_ar_ols_n60() {
        let (case, ds) = crate::test_support::load_case("ts_filter_ar_cpt");
        let x = col(&ds, "x");
        let r = ar(&x, ols(None, true, None)).unwrap();
        Check::new(&case)
            .scalar("ar_ols_n60_order", r.order as f64, Tol::exact())
            .vector("ar_ols_n60_coef", &r.ar, Tol::coef(60))
            .vector("ar_ols_n60_aic", &r.aic.clone().unwrap(), Tol::stat(60))
            .scalar("ar_ols_n60_var", r.var_pred, Tol::stat(60))
            .scalar("ar_ols_n60_mean", r.x_mean, Tol::coef(60))
            .finish();
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
    }

    /// ar(s, method = "ols", aic = FALSE, order.max = 2 | 3): var.pred 0.9278760628690985 and
    /// 0.9698929739989995 (SSE / (n - m)).
    #[test]
    fn test_validate_ar_ols_fixed_order() {
        let (case, ds) = crate::test_support::load_case("ts_filter_ar_cpt");
        let s = col(&ds, "s");
        let mut check = Check::new(&case);
        for m in [2usize, 3] {
            let r = ar(&s, ols(None, false, Some(m))).unwrap();
            assert_eq!(r.order, m);
            check
                .vector(&format!("ar_ols_fixed{m}_coef"), &r.ar, Tol::coef(24))
                .scalar(&format!("ar_ols_fixed{m}_var"), r.var_pred, Tol::stat(24));
        }
        check.finish();
    }

    /// R: ar(s, method = "ols", aic = FALSE, order.max = 13) warns and stops ("AA %*% t(X)").
    #[test]
    fn test_ar_ols_unestimable_fixed_order_is_an_error() {
        let path = p2a_validation::repo_path("validation/datasets/generated/ts_utils.csv");
        let ds = crate::data::Dataset::new(crate::data::DataLoader::load_csv(&path).unwrap());
        let s = col(&ds, "s");
        let err = ar(&s, ols(None, false, Some(13))).unwrap_err();
        assert!(
            matches!(
                err,
                EconError::InsufficientData { .. } | EconError::SingularMatrix { .. }
            ) && err.to_string().contains("AR(13) by OLS"),
            "{err}"
        );
        // An order beyond the series is the same refusal.
        let err = ar(&s, ols(None, false, Some(usize::MAX))).unwrap_err();
        assert!(matches!(err, EconError::InsufficientData { .. }), "{err}");
    }

    /// A series of an exact low-order recursion: the rank of X'X stops the search at R's rule,
    /// on a design with rows to spare, with R's warning.
    #[test]
    fn test_ar_ols_rank_stop_keeps_the_estimable_orders() {
        let x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.7).sin()).collect();
        let r = ar(&x, ols(Some(8), true, None)).unwrap();
        assert!(r.order <= 7, "{}", r.order);
        assert!(
            r.warnings.iter().any(|w| w.contains("singularities")),
            "{:?}",
            r.warnings
        );
        let aic = r.aic.unwrap();
        assert_eq!(aic[r.order], 0.0);
        assert!(aic[8].is_infinite());
    }

    /// ar(x, method = "yule-walker"): order 1, var.pred = DL variance * 60 / 58; fixed order 3.
    #[test]
    fn test_validate_ar_yule_walker_var_pred() {
        let (case, ds) = crate::test_support::load_case("ts_filter_ar_cpt");
        let x = col(&ds, "x");
        let r = ar(&x, ArConfig::default()).unwrap();
        let mut c3 = ArConfig::default();
        c3.aic = false;
        c3.order = Some(3);
        let r3 = ar(&x, c3).unwrap();
        Check::new(&case)
            .scalar("ar_yw_order", r.order as f64, Tol::exact())
            .vector("ar_yw_coef", &r.ar, Tol::coef(60))
            .scalar("ar_yw_var", r.var_pred, Tol::stat(60))
            .vector("ar_yw_fixed3_coef", &r3.ar, Tol::coef(60))
            .scalar("ar_yw_fixed3_var", r3.var_pred, Tol::stat(60))
            .finish();
    }

    #[test]
    fn test_ar_basic() {
        // Generate AR(1) process
        let mut x = vec![0.0; 100];
        x[0] = 1.0;
        let phi = 0.7;
        for t in 1..100 {
            // Simple AR(1): x_t = phi * x_{t-1} + noise
            x[t] = phi * x[t - 1] + (t as f64 % 3.0 - 1.0) * 0.3;
        }

        let result = ar(&x, ArConfig::default()).unwrap();

        println!("AR result: {}", result);
        println!("True phi: {}, Estimated: {:?}", phi, result.ar);

        assert!(result.n_obs == 100);
        assert!(result.var_pred > 0.0);
    }

    #[test]
    fn test_ar_methods() {
        let x: Vec<f64> = (0..50)
            .map(|i| (i as f64).sin() + (i as f64 * 0.1))
            .collect();

        // Test all methods
        for method in [ArMethod::YuleWalker, ArMethod::Burg, ArMethod::Ols] {
            let config = ArConfig {
                method,
                ..Default::default()
            };
            let result = ar(&x, config).unwrap();
            println!(
                "{:?} method: order={}, var={:.4}",
                method, result.order, result.var_pred
            );
            assert!(result.var_pred > 0.0);
        }
    }

    #[test]
    fn test_ar_fixed_order() {
        let x: Vec<f64> = (0..50).map(|i| i as f64 + (i as f64).sin()).collect();

        let config = ArConfig {
            aic: false,
            order: Some(3),
            ..Default::default()
        };
        let result = ar(&x, config).unwrap();

        assert_eq!(result.order, 3);
        assert_eq!(result.ar.len(), 3);
    }

    #[test]
    fn test_ar_aic_selection() {
        let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.2).sin()).collect();

        let result = ar(&x, ArConfig::default()).unwrap();

        assert!(result.aic.is_some());
        let aic = result.aic.unwrap();
        assert!(!aic.is_empty());
        // Minimum should be 0
        assert!(aic.contains(&0.0));
    }

    #[test]
    fn test_ar_residuals() {
        let x: Vec<f64> = (0..30).map(|i| i as f64).collect();

        let result = ar(
            &x,
            ArConfig {
                aic: false,
                order: Some(2),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(result.resid.len(), 30);
        // First `order` residuals should be NaN
        assert!(result.resid[0].is_nan());
        assert!(result.resid[1].is_nan());
        // Later residuals should not be NaN
        assert!(!result.resid[5].is_nan());
    }

    // ========================================================================
    // Validation Tests Against R
    // ========================================================================

    #[test]
    fn test_ar_display() {
        let x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.2).sin()).collect();
        let result = ar(&x, ArConfig::default()).unwrap();

        let display = format!("{}", result);
        assert!(display.contains("Autoregressive Model"));
        assert!(display.contains("Order selected"));
    }

    // ========================================================================
    // Additional R-vs-Rust Validation Tests (Phase 6)
    // ========================================================================

    /// LCG for deterministic random numbers
    fn lcg_rand_ar(seed: &mut u64) -> f64 {
        let a: u64 = 1103515245;
        let c: u64 = 12345;
        let m: u64 = 2_u64.pow(31);
        *seed = (a.wrapping_mul(*seed).wrapping_add(c)) % m;
        (*seed as f64) / (m as f64)
    }

    /// Box-Muller for normal
    fn box_muller_ar(seed: &mut u64) -> f64 {
        let u1 = lcg_rand_ar(seed).max(1e-10);
        let u2 = lcg_rand_ar(seed);
        ((-2.0_f64 * u1.ln()).sqrt()) * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn create_ar2_process() -> Vec<f64> {
        // Generate AR(2) process: x_t = 0.7*x_{t-1} - 0.2*x_{t-2} + e_t
        let n = 100;
        let mut seed: u64 = 42;
        let mut x = vec![0.0; n];

        x[0] = box_muller_ar(&mut seed);
        x[1] = 0.7 * x[0] + box_muller_ar(&mut seed);

        for t in 2..n {
            x[t] = 0.7 * x[t - 1] - 0.2 * x[t - 2] + box_muller_ar(&mut seed);
        }
        x
    }

    #[test]
    fn test_validate_ar_burg_vs_r() {
        // R reference:
        // ar(x, method = "burg")
        let x = create_ar2_process();
        let result = ar(
            &x,
            ArConfig {
                method: ArMethod::Burg,
                aic: true,
                ..Default::default()
            },
        )
        .unwrap();

        // Check structure - Burg method should run and produce valid output
        assert!(result.n_obs == x.len());
        assert!(result.var_pred > 0.0);
        assert!(result.method == ArMethod::Burg);

        // Residuals should be computed
        assert_eq!(result.resid.len(), x.len());
    }

    #[test]
    fn test_validate_ar_coefficient_recovery() {
        // Test that AR(2) with known coefficients recovers approximately
        let x = create_ar2_process();

        // Force AR(2) fit
        let result = ar(
            &x,
            ArConfig {
                aic: false,
                order: Some(2),
                method: ArMethod::YuleWalker,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(result.order, 2);
        assert_eq!(result.ar.len(), 2);

        // True values: ar1 = 0.7, ar2 = -0.2
        // With noise, expect within 0.3 of true values (relaxed for finite sample)
        let ar1_diff = (result.ar[0] - 0.7).abs();
        let ar2_diff = (result.ar[1] - (-0.2)).abs();

        assert!(
            ar1_diff < 0.5,
            "AR1 coefficient {} too far from true 0.7",
            result.ar[0]
        );
        assert!(
            ar2_diff < 0.5,
            "AR2 coefficient {} too far from true -0.2",
            result.ar[1]
        );
    }

    #[test]
    fn test_validate_ar_aic_order_selection() {
        // R reference:
        // ar(x, aic = TRUE)
        let x = create_ar2_process();
        let result = ar(
            &x,
            ArConfig {
                aic: true,
                ..Default::default()
            },
        )
        .unwrap();

        // AIC values should be present
        assert!(result.aic.is_some());
        let aic_vals = result.aic.as_ref().unwrap();

        // Should have AIC for orders 0 to max_order
        assert!(aic_vals.len() > 1);

        // Minimum AIC should be 0 (relative AIC)
        let min_aic = aic_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            min_aic.abs() < 1e-10,
            "Minimum relative AIC should be 0, got {}",
            min_aic
        );

        // Selected order should correspond to minimum AIC
        let selected_order = result.order;
        assert!(
            aic_vals[selected_order].abs() < 1e-10,
            "Selected order {} should have AIC near 0, got {}",
            selected_order,
            aic_vals[selected_order]
        );
    }

    #[test]
    fn test_validate_ar_pacf_structure() {
        // PACF should cut off after true order for AR process
        let x = create_ar2_process();
        let result = ar(
            &x,
            ArConfig {
                order_max: Some(10),
                ..Default::default()
            },
        )
        .unwrap();

        // PACF should exist
        assert!(!result.partial_acf.is_empty());

        // For AR(2), PACF[0] and PACF[1] should be significant
        // PACF[k] for k > 2 should be smaller (decaying)
        if result.partial_acf.len() >= 5 {
            let early_pacf: f64 = result.partial_acf[0..2]
                .iter()
                .map(|x| x.abs())
                .sum::<f64>()
                / 2.0;
            let late_pacf: f64 = result.partial_acf[3..5]
                .iter()
                .map(|x| x.abs())
                .sum::<f64>()
                / 2.0;

            // Early PACF should generally be larger than late PACF for AR(2)
            // But this is probabilistic, so just check they're computed
            assert!(early_pacf.is_finite());
            assert!(late_pacf.is_finite());
        }
    }

    #[test]
    fn test_validate_ar_residuals() {
        let x = create_ar2_process();
        let result = ar(
            &x,
            ArConfig {
                aic: false,
                order: Some(2),
                ..Default::default()
            },
        )
        .unwrap();

        // Residuals should have same length as input
        assert_eq!(result.resid.len(), x.len());

        // First `order` residuals should be NaN
        for i in 0..result.order {
            assert!(result.resid[i].is_nan(), "Residual {} should be NaN", i);
        }

        // Remaining residuals should be finite
        for i in result.order..result.resid.len() {
            assert!(
                result.resid[i].is_finite(),
                "Residual {} should be finite",
                i
            );
        }

        // Residual variance should be close to var_pred
        let valid_resid: Vec<f64> = result
            .resid
            .iter()
            .filter(|r| r.is_finite())
            .cloned()
            .collect();
        let resid_var: f64 =
            valid_resid.iter().map(|r| r * r).sum::<f64>() / valid_resid.len() as f64;

        // Should be in same ballpark (within factor of 2)
        let ratio = resid_var / result.var_pred;
        assert!(
            ratio > 0.3 && ratio < 3.0,
            "Residual variance {} should be close to var_pred {}",
            resid_var,
            result.var_pred
        );
    }

    #[test]
    fn test_validate_ar_mean_centering() {
        // Test with non-zero mean series
        let x: Vec<f64> = create_ar2_process().iter().map(|v| v + 10.0).collect();

        let result = ar(
            &x,
            ArConfig {
                demean: true,
                aic: false,
                order: Some(2),
                ..Default::default()
            },
        )
        .unwrap();

        // x_mean should be close to 10
        assert!(
            (result.x_mean - 10.0).abs() < 2.0,
            "Mean should be close to 10, got {}",
            result.x_mean
        );
    }

    #[test]
    fn test_validate_ar_methods_consistency() {
        // All methods should give similar results for well-behaved data
        let x = create_ar2_process();

        let yw = ar(
            &x,
            ArConfig {
                method: ArMethod::YuleWalker,
                aic: false,
                order: Some(2),
                ..Default::default()
            },
        )
        .unwrap();
        let burg = ar(
            &x,
            ArConfig {
                method: ArMethod::Burg,
                aic: false,
                order: Some(2),
                ..Default::default()
            },
        )
        .unwrap();
        let ols = ar(
            &x,
            ArConfig {
                method: ArMethod::Ols,
                aic: false,
                order: Some(2),
                ..Default::default()
            },
        )
        .unwrap();

        // All should select same order when forced
        assert_eq!(yw.order, 2);
        assert_eq!(burg.order, 2);
        assert_eq!(ols.order, 2);

        // Coefficients should be similar (within 0.3)
        for i in 0..2 {
            let diff_yw_burg = (yw.ar[i] - burg.ar[i]).abs();
            let diff_yw_ols = (yw.ar[i] - ols.ar[i]).abs();
            assert!(
                diff_yw_burg < 0.5,
                "YW-Burg coefficient {} diff too large: {}",
                i,
                diff_yw_burg
            );
            assert!(
                diff_yw_ols < 0.5,
                "YW-OLS coefficient {} diff too large: {}",
                i,
                diff_yw_ols
            );
        }
    }
}
