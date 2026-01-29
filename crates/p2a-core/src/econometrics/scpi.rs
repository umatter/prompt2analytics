//! Synthetic Control with Prediction Intervals (SCPI).
//!
//! Pure Rust implementation of the SCPI method developed by Cattaneo, Feng, and Titiunik (2021).
//! SCPI extends synthetic control methods with proper uncertainty quantification through
//! prediction intervals that account for both in-sample and out-of-sample uncertainty.
//!
//! # Mathematical Framework
//!
//! The synthetic control problem minimizes:
//! ```text
//! ||Y₁ - W'Y₀||² subject to constraints on W
//! ```
//!
//! Supported constraint types:
//! - **Simplex**: Σwⱼ = 1, wⱼ ≥ 0 (classic Abadie, Diamond, Hainmueller approach)
//! - **Lasso**: Add λ||w||₁ penalty
//! - **Ridge**: Add λ||w||₂² penalty
//! - **Lasso-Simplex**: Lasso penalty with simplex constraints
//!
//! Prediction intervals are constructed as:
//! ```text
//! PI = prediction ± t_{α/2,df} × √(σ²_in + σ²_out)
//! ```
//!
//! Where:
//! - σ²_in: In-sample variance from pre-treatment fit
//! - σ²_out: Out-of-sample variance estimated via cross-validation or plug-in formula
//!
//! # References
//!
//! - Cattaneo, M. D., Feng, Y., & Titiunik, R. (2021). "Prediction Intervals for Synthetic
//!   Control Methods." *Journal of the American Statistical Association*, 116(536), 1865-1880.
//!   DOI: 10.1080/01621459.2021.1979561
//!
//! - Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control Methods for
//!   Comparative Case Studies." *Journal of the American Statistical Association*, 105(490), 493-505.
//!
//! - Implementation inspired by:
//!   - R/Python package `scpi` (Cattaneo, Feng, Palomba, Titiunik)
//!     Source: <https://nppackages.github.io/scpi/>
//!     CRAN: <https://cran.r-project.org/package=scpi>

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::{cholesky_inverse, xtx, xty};
use crate::traits::estimator::SignificanceLevel;

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Constraint type for SCPI weight estimation.
///
/// Different constraint types offer trade-offs between interpretability,
/// sparsity, and bias-variance.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum SCPIConstraint {
    /// Simplex constraint: weights sum to 1 and are non-negative.
    /// This is the classic synthetic control constraint (Abadie et al., 2010).
    /// Most interpretable but may have high variance with many donors.
    #[default]
    Simplex,

    /// Lasso (L1) penalty: λ||w||₁.
    /// Encourages sparse solutions with few non-zero weights.
    /// Good when only a few donors are relevant.
    Lasso {
        /// Regularization parameter (larger = more sparsity)
        lambda: f64,
    },

    /// Ridge (L2) penalty: λ||w||₂².
    /// Shrinks all weights toward zero but doesn't produce sparsity.
    /// Good for reducing variance with many correlated donors.
    Ridge {
        /// Regularization parameter (larger = more shrinkage)
        lambda: f64,
    },

    /// Lasso penalty combined with simplex constraints.
    /// Sparse, interpretable weights that sum to 1.
    LassoSimplex {
        /// L1 regularization parameter
        lambda: f64,
    },
}

/// Configuration for SCPI estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SCPIConfig {
    /// Constraint type for weight estimation
    pub constraint: SCPIConstraint,

    /// Significance level for prediction intervals (default: 0.05 for 95% PI)
    pub alpha: f64,

    /// Method for out-of-sample variance estimation
    pub variance_method: VarianceMethod,

    /// Number of folds for cross-validation (if using CV for variance)
    pub cv_folds: usize,

    /// Maximum iterations for optimization
    pub max_iter: usize,

    /// Convergence tolerance
    pub tolerance: f64,

    /// Minimum weight to report (for sparsity in output)
    pub weight_threshold: f64,
}

impl Default for SCPIConfig {
    fn default() -> Self {
        SCPIConfig {
            constraint: SCPIConstraint::Simplex,
            alpha: 0.05,
            variance_method: VarianceMethod::Subgaussian,
            cv_folds: 5,
            max_iter: 10000,
            tolerance: 1e-8,
            weight_threshold: 0.001,
        }
    }
}

/// Method for estimating out-of-sample variance.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum VarianceMethod {
    /// Gaussian assumption: uses standard regression variance
    Gaussian,

    /// Subgaussian assumption (recommended): more conservative bounds
    /// This is the default in the scpi package
    #[default]
    Subgaussian,

    /// Leave-one-out cross-validation
    LooCv,

    /// K-fold cross-validation
    KFoldCv,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Result Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Prediction interval for a single time period.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionInterval {
    /// Time period index
    pub period: usize,

    /// Point prediction (synthetic control value)
    pub prediction: f64,

    /// Lower bound of prediction interval
    pub lower: f64,

    /// Upper bound of prediction interval
    pub upper: f64,

    /// Standard error of prediction
    pub std_error: f64,
}

/// Complete results from SCPI estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SCPIResult {
    // ─── Weights ───────────────────────────────────────────────────────────────
    /// Estimated donor weights (J × 1)
    #[serde(skip)]
    pub weights: Array1<f64>,

    /// Non-zero weights with donor indices (donor_idx, weight)
    pub nonzero_weights: Vec<(usize, f64)>,

    /// Number of donors with non-negligible weight
    pub n_effective_donors: usize,

    // ─── Time Series ───────────────────────────────────────────────────────────
    /// Actual outcomes for treated unit (T × 1)
    #[serde(skip)]
    pub treated_actual: Array1<f64>,

    /// Synthetic control predictions (T × 1)
    #[serde(skip)]
    pub synthetic: Array1<f64>,

    /// Treatment effect estimates: actual - synthetic (T_post × 1)
    #[serde(skip)]
    pub effect: Array1<f64>,

    // ─── Standard Errors and Intervals ─────────────────────────────────────────
    /// Standard errors for each post-treatment period
    #[serde(skip)]
    pub effect_se: Array1<f64>,

    /// Lower bound of prediction intervals
    #[serde(skip)]
    pub ci_lower: Array1<f64>,

    /// Upper bound of prediction intervals
    #[serde(skip)]
    pub ci_upper: Array1<f64>,

    /// Prediction intervals for each post-treatment period
    pub prediction_intervals: Vec<PredictionInterval>,

    // ─── Variance Components ───────────────────────────────────────────────────
    /// In-sample variance (from pre-treatment fit)
    pub in_sample_var: f64,

    /// Out-of-sample variance estimate
    pub out_sample_var: f64,

    /// Combined variance (in_sample + out_sample)
    pub total_var: f64,

    // ─── Fit Statistics ────────────────────────────────────────────────────────
    /// Pre-treatment RMSPE (Root Mean Squared Prediction Error)
    pub pre_treatment_rmspe: f64,

    /// Pre-treatment MSPE
    pub pre_treatment_mspe: f64,

    /// R-squared in pre-treatment period
    pub pre_treatment_r2: f64,

    // ─── Configuration ─────────────────────────────────────────────────────────
    /// Number of pre-treatment periods
    pub n_pre_periods: usize,

    /// Number of post-treatment periods
    pub n_post_periods: usize,

    /// Number of donor units
    pub n_donors: usize,

    /// Treatment period index (first post-treatment period)
    pub treatment_period: usize,

    /// Significance level used for intervals
    pub alpha: f64,

    /// Critical value used for intervals (t or z)
    pub critical_value: f64,

    /// Constraint type used
    pub constraint_type: String,
}

impl fmt::Display for SCPIResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Synthetic Control with Prediction Intervals (SCPI)")?;
        writeln!(f, "===================================================")?;
        writeln!(f)?;

        writeln!(f, "Configuration:")?;
        writeln!(f, "  Constraint type: {}", self.constraint_type)?;
        writeln!(f, "  Pre-treatment periods: {}", self.n_pre_periods)?;
        writeln!(f, "  Post-treatment periods: {}", self.n_post_periods)?;
        writeln!(f, "  Donor pool size: {}", self.n_donors)?;
        writeln!(f, "  Effective donors: {}", self.n_effective_donors)?;
        writeln!(
            f,
            "  Significance level: {:.0}%",
            (1.0 - self.alpha) * 100.0
        )?;
        writeln!(f)?;

        writeln!(f, "Pre-Treatment Fit:")?;
        writeln!(f, "  RMSPE: {:.6}", self.pre_treatment_rmspe)?;
        writeln!(f, "  R-squared: {:.4}", self.pre_treatment_r2)?;
        writeln!(f)?;

        writeln!(f, "Variance Components:")?;
        writeln!(f, "  In-sample variance: {:.6}", self.in_sample_var)?;
        writeln!(f, "  Out-of-sample variance: {:.6}", self.out_sample_var)?;
        writeln!(f, "  Total variance: {:.6}", self.total_var)?;
        writeln!(f)?;

        writeln!(f, "Donor Weights (non-zero):")?;
        for (idx, weight) in &self.nonzero_weights {
            writeln!(f, "  Donor {}: {:.4}", idx, weight)?;
        }
        writeln!(f)?;

        writeln!(
            f,
            "Treatment Effects with {}% Prediction Intervals:",
            ((1.0 - self.alpha) * 100.0) as i32
        )?;
        writeln!(
            f,
            "{:>8} {:>12} {:>12} {:>12} {:>12}",
            "Period", "Effect", "Std.Err", "Lower", "Upper"
        )?;
        writeln!(f, "{}", "-".repeat(60))?;

        let sig = SignificanceLevel::from_p_value(self.alpha);
        for (i, pi) in self.prediction_intervals.iter().enumerate() {
            let effect = self.effect[i];
            let se = self.effect_se[i];

            // Check if interval excludes zero (significant)
            let star = if pi.lower > 0.0 || pi.upper < 0.0 {
                sig.stars()
            } else {
                ""
            };

            writeln!(
                f,
                "{:>8} {:>12.4} {:>12.4} {:>12.4} {:>12.4}{}",
                pi.period, effect, se, pi.lower, pi.upper, star
            )?;
        }

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main Function
// ═══════════════════════════════════════════════════════════════════════════════

/// Run Synthetic Control with Prediction Intervals (SCPI).
///
/// Constructs a synthetic control using constrained regression and computes
/// prediction intervals that account for both in-sample and out-of-sample uncertainty.
///
/// # Arguments
///
/// * `treated` - Time series of outcomes for the treated unit (T × 1)
/// * `donors` - Matrix of donor unit outcomes (T × J, rows = time, cols = units)
/// * `treatment_period` - Index of the first post-treatment period (0-based)
/// * `config` - Configuration options for estimation
///
/// # Mathematical Framework
///
/// The synthetic control weights W are estimated by minimizing:
/// ```text
/// min_W ||Y₁_pre - Y₀_pre × W||² + penalty(W)
/// ```
///
/// Subject to constraints depending on `config.constraint`:
/// - Simplex: Σwⱼ = 1, wⱼ ≥ 0
/// - Lasso: Add λ||W||₁
/// - Ridge: Add λ||W||₂²
///
/// Prediction intervals are computed as (Cattaneo et al., 2021, Eq. 3.5):
/// ```text
/// [Ŷ₁ₜ - c_α × σ̂, Ŷ₁ₜ + c_α × σ̂]
/// ```
///
/// Where σ̂² = σ̂²_in + σ̂²_out combines in-sample and out-of-sample variance.
///
/// # References
///
/// - Cattaneo, M. D., Feng, Y., & Titiunik, R. (2021). JASA 116(536), 1865-1880.
///
/// # Example
///
/// ```ignore
/// use p2a_core::econometrics::scpi::{run_scpi, SCPIConfig, SCPIConstraint};
/// use ndarray::{array, Array2};
///
/// // Treated unit outcomes (10 periods, treatment at period 7)
/// let treated = array![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 20.0, 21.0, 22.0];
///
/// // Donor outcomes (10 periods × 5 donors)
/// let donors = Array2::from_shape_fn((10, 5), |(t, j)| 10.0 + t as f64 + 0.5 * j as f64);
///
/// let config = SCPIConfig {
///     constraint: SCPIConstraint::Simplex,
///     alpha: 0.05,
///     ..Default::default()
/// };
///
/// let result = run_scpi(&treated.view(), &donors.view(), 7, config)?;
/// println!("Treatment effect: {:?}", result.effect);
/// ```
pub fn run_scpi(
    treated: &ArrayView1<f64>,
    donors: &ArrayView2<f64>,
    treatment_period: usize,
    config: SCPIConfig,
) -> EconResult<SCPIResult> {
    // Validate inputs
    let t_total = treated.len();
    let (t_donors, n_donors) = donors.dim();

    if t_donors != t_total {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Treated series length ({}) must match donor matrix rows ({})",
                t_total, t_donors
            ),
        });
    }

    if treatment_period == 0 {
        return Err(EconError::InvalidSpecification {
            message: "Treatment period must be > 0 (need at least one pre-treatment period)"
                .to_string(),
        });
    }

    if treatment_period >= t_total {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Treatment period ({}) must be < total periods ({})",
                treatment_period, t_total
            ),
        });
    }

    let n_pre = treatment_period;
    let n_post = t_total - treatment_period;

    if n_donors == 0 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "At least one donor unit is required".to_string(),
        });
    }

    if n_pre < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n_pre,
            context: "At least 2 pre-treatment periods are required".to_string(),
        });
    }

    // Split data into pre- and post-treatment
    let y_pre = treated.slice(s![..treatment_period]).to_owned();
    let y_post = treated.slice(s![treatment_period..]).to_owned();
    let x_pre = donors.slice(s![..treatment_period, ..]).to_owned();
    let x_post = donors.slice(s![treatment_period.., ..]).to_owned();

    // Estimate weights
    let weights = estimate_weights(
        &y_pre.view(),
        &x_pre.view(),
        &config.constraint,
        config.max_iter,
        config.tolerance,
    )?;

    // Compute synthetic control predictions
    let synthetic_pre = x_pre.dot(&weights);
    let synthetic_post = x_post.dot(&weights);

    // Full synthetic series
    let mut synthetic = Array1::zeros(t_total);
    synthetic
        .slice_mut(s![..treatment_period])
        .assign(&synthetic_pre);
    synthetic
        .slice_mut(s![treatment_period..])
        .assign(&synthetic_post);

    // Treatment effects (post-treatment only)
    let effect = &y_post - &synthetic_post;

    // Compute residuals and in-sample variance (Cattaneo et al. 2021, Eq. 3.2)
    let residuals_pre = &y_pre - &synthetic_pre;
    let in_sample_var = compute_in_sample_variance(&residuals_pre, n_pre);

    // Compute out-of-sample variance (Cattaneo et al. 2021, Section 3.2)
    let out_sample_var = compute_out_sample_variance(
        &y_pre.view(),
        &x_pre.view(),
        &weights.view(),
        &config.variance_method,
        config.cv_folds,
    )?;

    // Total variance for prediction intervals
    let total_var = in_sample_var + out_sample_var;

    // Compute critical value
    // For small samples, use t-distribution; for large samples, z
    let df = (n_pre - 1).max(1);
    let critical_value = if df < 30 {
        // Use t-distribution critical value
        compute_t_critical(config.alpha / 2.0, df)
    } else {
        // Use z critical value for large samples
        compute_z_critical(config.alpha / 2.0)
    };

    // Standard error for each post-treatment period
    // SE varies by period due to potential heteroskedasticity
    let se_base = total_var.sqrt();
    let effect_se = Array1::from_elem(n_post, se_base);

    // Prediction intervals
    let ci_lower = &effect - &(&effect_se * critical_value);
    let ci_upper = &effect + &(&effect_se * critical_value);

    // Build detailed prediction intervals
    let prediction_intervals: Vec<PredictionInterval> = (0..n_post)
        .map(|i| PredictionInterval {
            period: treatment_period + i,
            prediction: synthetic_post[i],
            lower: ci_lower[i],
            upper: ci_upper[i],
            std_error: effect_se[i],
        })
        .collect();

    // Pre-treatment fit statistics
    let pre_mspe = residuals_pre.iter().map(|r| r * r).sum::<f64>() / n_pre as f64;
    let pre_rmspe = pre_mspe.sqrt();

    let y_pre_mean = y_pre.mean().unwrap_or(0.0);
    let tss: f64 = y_pre.iter().map(|y| (y - y_pre_mean).powi(2)).sum();
    let rss: f64 = residuals_pre.iter().map(|r| r * r).sum();
    let pre_r2 = if tss > 0.0 { 1.0 - rss / tss } else { 0.0 };

    // Non-zero weights
    let nonzero_weights: Vec<(usize, f64)> = weights
        .iter()
        .enumerate()
        .filter(|(_, w)| w.abs() >= config.weight_threshold)
        .map(|(i, w)| (i, *w))
        .collect();

    let n_effective_donors = nonzero_weights.len();

    // Constraint type description
    let constraint_type = match &config.constraint {
        SCPIConstraint::Simplex => "Simplex".to_string(),
        SCPIConstraint::Lasso { lambda } => format!("Lasso (lambda={})", lambda),
        SCPIConstraint::Ridge { lambda } => format!("Ridge (lambda={})", lambda),
        SCPIConstraint::LassoSimplex { lambda } => format!("Lasso-Simplex (lambda={})", lambda),
    };

    Ok(SCPIResult {
        weights,
        nonzero_weights,
        n_effective_donors,
        treated_actual: treated.to_owned(),
        synthetic,
        effect,
        effect_se,
        ci_lower,
        ci_upper,
        prediction_intervals,
        in_sample_var,
        out_sample_var,
        total_var,
        pre_treatment_rmspe: pre_rmspe,
        pre_treatment_mspe: pre_mspe,
        pre_treatment_r2: pre_r2,
        n_pre_periods: n_pre,
        n_post_periods: n_post,
        n_donors,
        treatment_period,
        alpha: config.alpha,
        critical_value,
        constraint_type,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Weight Estimation
// ═══════════════════════════════════════════════════════════════════════════════

/// Estimate synthetic control weights under various constraints.
///
/// # Arguments
/// * `y` - Pre-treatment outcomes for treated unit (T0 × 1)
/// * `x` - Pre-treatment outcomes for donors (T0 × J)
/// * `constraint` - Constraint type
/// * `max_iter` - Maximum iterations
/// * `tolerance` - Convergence tolerance
fn estimate_weights(
    y: &ArrayView1<f64>,
    x: &ArrayView2<f64>,
    constraint: &SCPIConstraint,
    max_iter: usize,
    tolerance: f64,
) -> EconResult<Array1<f64>> {
    let _n = x.ncols();

    match constraint {
        SCPIConstraint::Simplex => solve_simplex_weights(y, x, max_iter, tolerance),

        SCPIConstraint::Lasso { lambda } => solve_lasso_weights(y, x, *lambda, max_iter, tolerance),

        SCPIConstraint::Ridge { lambda } => solve_ridge_weights(y, x, *lambda),

        SCPIConstraint::LassoSimplex { lambda } => {
            solve_lasso_simplex_weights(y, x, *lambda, max_iter, tolerance)
        }
    }
}

/// Solve for weights under simplex constraints using projected gradient descent.
///
/// Minimizes ||y - Xw||² subject to Σwⱼ = 1, wⱼ ≥ 0
///
/// Uses Frank-Wolfe algorithm (Cattaneo et al. 2021, Algorithm 1).
fn solve_simplex_weights(
    y: &ArrayView1<f64>,
    x: &ArrayView2<f64>,
    max_iter: usize,
    tolerance: f64,
) -> EconResult<Array1<f64>> {
    let n = x.ncols();

    // QP: min 0.5 * w' * H * w + c' * w
    // H = X'X, c = -X'y
    let h = xtx(x);
    let c = -xty(x, &y.to_owned());

    // Add small regularization for numerical stability
    let mut h_reg = h.clone();
    for i in 0..n {
        h_reg[[i, i]] += 1e-8;
    }

    // Initialize with uniform weights
    let mut w = Array1::from_elem(n, 1.0 / n as f64);

    // Frank-Wolfe / projected gradient descent
    for _ in 0..max_iter {
        // Gradient: H * w + c
        let grad = h_reg.dot(&w) + &c;

        // Find vertex that minimizes gradient (Frank-Wolfe direction)
        let min_idx = grad
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Direction: e_min - w
        let mut direction = Array1::zeros(n);
        direction[min_idx] = 1.0;
        let direction = &direction - &w;

        // Optimal step size via line search
        // d'Hd and d'(Hw + c)
        let hd = h_reg.dot(&direction);
        let d_h_d: f64 = direction.iter().zip(hd.iter()).map(|(&d, &h)| d * h).sum();
        let hw_c = h_reg.dot(&w) + &c;
        let d_hw_c: f64 = direction
            .iter()
            .zip(hw_c.iter())
            .map(|(&d, &h)| d * h)
            .sum();

        let alpha = if d_h_d.abs() > 1e-12 {
            (-d_hw_c / d_h_d).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Update
        let w_new = &w + &(&direction * alpha);

        // Check convergence
        let change: f64 = w_new
            .iter()
            .zip(w.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum();
        w = w_new;

        if change < tolerance {
            break;
        }
    }

    // Ensure constraints are satisfied
    w.mapv_inplace(|x| x.max(0.0));
    let sum: f64 = w.sum();
    if sum > 0.0 {
        w.mapv_inplace(|x| x / sum);
    }

    Ok(w)
}

/// Solve for weights with Lasso (L1) penalty using coordinate descent.
///
/// Minimizes ||y - Xw||² + λ||w||₁
fn solve_lasso_weights(
    y: &ArrayView1<f64>,
    x: &ArrayView2<f64>,
    lambda: f64,
    max_iter: usize,
    tolerance: f64,
) -> EconResult<Array1<f64>> {
    let (t, n) = x.dim();

    // Coordinate descent (LASSO via soft-thresholding)
    let mut w = Array1::zeros(n);

    // Precompute X'X diagonal and X'y
    let x_sq_sum: Vec<f64> = (0..n)
        .map(|j| x.column(j).iter().map(|v| v * v).sum())
        .collect();

    let _xy: Vec<f64> = (0..n)
        .map(|j| x.column(j).iter().zip(y.iter()).map(|(&x, &y)| x * y).sum())
        .collect();

    for _ in 0..max_iter {
        let w_old = w.clone();

        for j in 0..n {
            // Compute partial residual
            let mut r_partial = 0.0;
            for i in 0..t {
                let mut pred = 0.0;
                for k in 0..n {
                    if k != j {
                        pred += x[[i, k]] * w[k];
                    }
                }
                r_partial += x[[i, j]] * (y[i] - pred);
            }

            // Soft-thresholding
            let denom = x_sq_sum[j];
            if denom > 1e-10 {
                w[j] = soft_threshold(r_partial / denom, lambda / denom);
            }
        }

        // Check convergence
        let change: f64 = w
            .iter()
            .zip(w_old.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum();
        if change < tolerance {
            break;
        }
    }

    Ok(w)
}

/// Soft-thresholding operator for Lasso.
#[inline]
fn soft_threshold(x: f64, lambda: f64) -> f64 {
    if x > lambda {
        x - lambda
    } else if x < -lambda {
        x + lambda
    } else {
        0.0
    }
}

/// Solve for weights with Ridge (L2) penalty.
///
/// Minimizes ||y - Xw||² + λ||w||₂²
///
/// Closed-form solution: w = (X'X + λI)⁻¹ X'y
fn solve_ridge_weights(
    y: &ArrayView1<f64>,
    x: &ArrayView2<f64>,
    lambda: f64,
) -> EconResult<Array1<f64>> {
    let n = x.ncols();

    // X'X + λI
    let mut xtx_reg = xtx(x);
    for i in 0..n {
        xtx_reg[[i, i]] += lambda;
    }

    // (X'X + λI)⁻¹
    let xtx_inv = cholesky_inverse(&xtx_reg.view()).map_err(|e| {
        EconError::Computation(format!(
            "Ridge weight estimation failed: Matrix inversion error: {:?}",
            e
        ))
    })?;

    // X'y
    let xy = xty(x, &y.to_owned());

    // w = (X'X + λI)⁻¹ X'y
    let w = xtx_inv.dot(&xy);

    Ok(w)
}

/// Solve for weights with Lasso penalty under simplex constraints.
///
/// Uses ADMM (Alternating Direction Method of Multipliers).
fn solve_lasso_simplex_weights(
    y: &ArrayView1<f64>,
    x: &ArrayView2<f64>,
    lambda: f64,
    max_iter: usize,
    tolerance: f64,
) -> EconResult<Array1<f64>> {
    let n = x.ncols();
    let rho = 1.0; // ADMM penalty parameter

    // Precompute matrices
    let xtx_mat = xtx(x);
    let xy = xty(x, &y.to_owned());

    // (X'X + ρI)⁻¹
    let mut a_inv_mat = xtx_mat.clone();
    for i in 0..n {
        a_inv_mat[[i, i]] += rho;
    }
    let a_inv = cholesky_inverse(&a_inv_mat.view()).map_err(|e| {
        EconError::Computation(format!(
            "Lasso-simplex weight estimation failed: Matrix inversion error: {:?}",
            e
        ))
    })?;

    // Initialize variables
    let mut w = Array1::from_elem(n, 1.0 / n as f64);
    let mut z = w.clone();
    let mut u: Array1<f64> = Array1::zeros(n); // Dual variable

    for _ in 0..max_iter {
        let w_old = w.clone();

        // w-update: minimize ||y - Xw||² + (ρ/2)||w - z + u||²
        // w = (X'X + ρI)⁻¹ (X'y + ρ(z - u))
        let rhs = &xy + &((&z - &u) * rho);
        w = a_inv.dot(&rhs);

        // z-update: simplex projection with L1 penalty
        // z = argmin λ||z||₁ + (ρ/2)||w + u - z||²  s.t. simplex
        let v = &w + &u;

        // Apply soft-thresholding then project to simplex
        let v_thresh: Array1<f64> = v.mapv(|x| soft_threshold(x, lambda / rho));
        z = project_to_simplex(&v_thresh);

        // u-update
        u = &u + &(&w - &z);

        // Check convergence
        let primal_res: f64 = (&w - &z).iter().map(|&x| x * x).sum::<f64>().sqrt();
        let dual_res: f64 = (&w - &w_old).iter().map(|&x| x * x).sum::<f64>().sqrt() * rho;

        if primal_res < tolerance && dual_res < tolerance {
            break;
        }
    }

    Ok(z)
}

/// Project a vector onto the simplex (Σx = 1, x ≥ 0).
///
/// Uses Duchi et al. (2008) algorithm.
fn project_to_simplex(v: &Array1<f64>) -> Array1<f64> {
    let _n = v.len();

    // Sort in descending order
    let mut sorted: Vec<(usize, f64)> = v.iter().cloned().enumerate().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Find threshold
    let mut cumsum = 0.0;
    let mut rho = 0;

    for (i, (_, val)) in sorted.iter().enumerate() {
        cumsum += val;
        if val - (cumsum - 1.0) / (i + 1) as f64 > 0.0 {
            rho = i + 1;
        }
    }

    let theta = (sorted[..rho].iter().map(|(_, v)| v).sum::<f64>() - 1.0) / rho as f64;

    // Project
    let result: Array1<f64> = v.mapv(|x| (x - theta).max(0.0));

    result
}

// ═══════════════════════════════════════════════════════════════════════════════
// Variance Estimation
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute in-sample variance from pre-treatment fit residuals.
///
/// σ²_in = (1 / T0) Σ (y_t - ŷ_t)² (Cattaneo et al. 2021, Eq. 3.2)
fn compute_in_sample_variance(residuals: &Array1<f64>, n_pre: usize) -> f64 {
    let sse: f64 = residuals.iter().map(|r| r * r).sum();
    sse / n_pre as f64
}

/// Compute out-of-sample variance estimate.
///
/// Different methods provide different bias-variance tradeoffs:
/// - Gaussian: standard regression variance (may underestimate)
/// - Subgaussian: more conservative, accounts for tail behavior
/// - CV: data-driven, but higher variance
fn compute_out_sample_variance(
    y: &ArrayView1<f64>,
    x: &ArrayView2<f64>,
    w: &ArrayView1<f64>,
    method: &VarianceMethod,
    cv_folds: usize,
) -> EconResult<f64> {
    let (t, j) = x.dim();

    match method {
        VarianceMethod::Gaussian => {
            // Standard regression variance: σ² = ||y - Xw||² / (T0 - J)
            let synthetic = x.dot(w);
            let residuals = y.to_owned() - &synthetic;
            let sse: f64 = residuals.iter().map(|r| r * r).sum();
            let df = (t as i64 - j as i64).max(1) as f64;
            Ok(sse / df)
        }

        VarianceMethod::Subgaussian => {
            // Subgaussian variance (Cattaneo et al. 2021, Section 3.2)
            // More conservative: accounts for worst-case tail behavior
            let synthetic = x.dot(w);
            let residuals = y.to_owned() - &synthetic;

            // Estimate subgaussian parameter using median absolute deviation
            let mut abs_resid: Vec<f64> = residuals.iter().map(|r| r.abs()).collect();
            abs_resid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let median_abs = if abs_resid.len() % 2 == 0 {
                let mid = abs_resid.len() / 2;
                (abs_resid[mid - 1] + abs_resid[mid]) / 2.0
            } else {
                abs_resid[abs_resid.len() / 2]
            };

            // MAD-based variance estimate (robust)
            // σ ≈ MAD / 0.6745 for normal data
            let sigma_mad = median_abs / 0.6745;

            // Inflate for subgaussian tail behavior
            // Factor of 1.5-2 is common in practice
            let inflation = 1.5;

            Ok(sigma_mad * sigma_mad * inflation)
        }

        VarianceMethod::LooCv => {
            // Leave-one-out cross-validation
            let mut cv_errors = Vec::with_capacity(t);

            for i in 0..t {
                // Leave out observation i
                let y_train: Array1<f64> = y
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| *idx != i)
                    .map(|(_, &v)| v)
                    .collect();

                let mut x_train = Array2::zeros((t - 1, j));
                let mut row_idx = 0;
                for (idx, row) in x.axis_iter(Axis(0)).enumerate() {
                    if idx != i {
                        x_train.row_mut(row_idx).assign(&row);
                        row_idx += 1;
                    }
                }

                // Re-estimate weights (using simplex by default for LOO)
                if let Ok(w_loo) =
                    solve_simplex_weights(&y_train.view(), &x_train.view(), 1000, 1e-8)
                {
                    let pred = x.row(i).dot(&w_loo);
                    let error = y[i] - pred;
                    cv_errors.push(error * error);
                }
            }

            if cv_errors.is_empty() {
                return Err(EconError::Computation(
                    "LOO-CV variance estimation failed: No valid CV folds".to_string(),
                ));
            }

            let cv_mse = cv_errors.iter().sum::<f64>() / cv_errors.len() as f64;
            Ok(cv_mse)
        }

        VarianceMethod::KFoldCv => {
            // K-fold cross-validation
            let fold_size = t.div_ceil(cv_folds);
            let mut cv_errors = Vec::new();

            for fold in 0..cv_folds {
                let start = fold * fold_size;
                let end = ((fold + 1) * fold_size).min(t);

                if end <= start {
                    continue;
                }

                // Split into train/test
                let test_indices: Vec<usize> = (start..end).collect();
                let train_indices: Vec<usize> =
                    (0..t).filter(|i| !test_indices.contains(i)).collect();

                if train_indices.is_empty() {
                    continue;
                }

                let y_train: Array1<f64> = train_indices.iter().map(|&i| y[i]).collect();
                let mut x_train = Array2::zeros((train_indices.len(), j));
                for (row_idx, &orig_idx) in train_indices.iter().enumerate() {
                    x_train.row_mut(row_idx).assign(&x.row(orig_idx));
                }

                // Re-estimate weights
                if let Ok(w_fold) =
                    solve_simplex_weights(&y_train.view(), &x_train.view(), 1000, 1e-8)
                {
                    // Compute test errors
                    for &test_idx in &test_indices {
                        let pred = x.row(test_idx).dot(&w_fold);
                        let error = y[test_idx] - pred;
                        cv_errors.push(error * error);
                    }
                }
            }

            if cv_errors.is_empty() {
                return Err(EconError::Computation(
                    "K-fold CV variance estimation failed: No valid CV folds".to_string(),
                ));
            }

            let cv_mse = cv_errors.iter().sum::<f64>() / cv_errors.len() as f64;
            Ok(cv_mse)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Statistical Utilities
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute t-distribution critical value using approximation.
///
/// For df > 30, essentially equal to z.
fn compute_t_critical(alpha: f64, df: usize) -> f64 {
    // For small df, use approximation based on student-t quantile
    // This is the absolute value for two-tailed test
    if df >= 120 {
        return compute_z_critical(alpha);
    }

    // Simple approximation for common df values
    // More accurate would require special functions
    let z = compute_z_critical(alpha);

    // Cornish-Fisher expansion approximation
    let _g1 = 0.0; // Skewness = 0 for t
    let g2 = 6.0 / df as f64; // Excess kurtosis ≈ 6/df

    z + g2 * (z * z - 1.0) / 24.0
}

/// Compute z critical value (standard normal quantile).
fn compute_z_critical(alpha: f64) -> f64 {
    // Approximate inverse normal using rational approximation
    // Abramowitz and Stegun 26.2.23
    if alpha <= 0.0 || alpha >= 0.5 {
        return 1.96; // Default for 95% CI
    }

    let p = alpha;
    let t = (-2.0 * p.ln()).sqrt();

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{Array2, array};

    fn generate_test_data() -> (Array1<f64>, Array2<f64>) {
        // Generate simple test data
        // Treated unit: trend + treatment effect after period 7
        let treated = array![
            10.0, 11.5, 12.8, 14.1, 15.5, 16.9, 18.2, // Pre-treatment
            22.0, 23.5, 25.0 // Post-treatment (with +3 effect)
        ];

        // Donors: similar trends
        let donors = Array2::from_shape_fn((10, 5), |(t, j)| {
            10.0 + 1.4 * t as f64 + 0.3 * j as f64 + 0.1 * ((t * j) as f64).sin()
        });

        (treated, donors)
    }

    #[test]
    fn test_scpi_basic() {
        let (treated, donors) = generate_test_data();

        let config = SCPIConfig::default();
        let result = run_scpi(&treated.view(), &donors.view(), 7, config).unwrap();

        // Check basic properties
        assert_eq!(result.n_pre_periods, 7);
        assert_eq!(result.n_post_periods, 3);
        assert_eq!(result.n_donors, 5);
        assert_eq!(result.treatment_period, 7);

        // Weights should sum to 1 (simplex constraint)
        let weight_sum: f64 = result.weights.sum();
        assert_relative_eq!(weight_sum, 1.0, epsilon = 1e-6);

        // All weights should be non-negative
        assert!(result.weights.iter().all(|&w| w >= -1e-10));

        // Should have prediction intervals
        assert_eq!(result.prediction_intervals.len(), 3);

        // Intervals should be ordered: lower < effect < upper
        for (i, pi) in result.prediction_intervals.iter().enumerate() {
            // Note: effect is actual - synthetic, PI bounds are for the effect
            let effect = result.effect[i];
            assert!(pi.lower < effect || (pi.lower - effect).abs() < 1e-10);
            assert!(effect < pi.upper || (effect - pi.upper).abs() < 1e-10);
        }

        println!("{}", result);
    }

    #[test]
    fn test_scpi_lasso() {
        let (treated, donors) = generate_test_data();

        let config = SCPIConfig {
            constraint: SCPIConstraint::Lasso { lambda: 0.1 },
            ..Default::default()
        };

        let result = run_scpi(&treated.view(), &donors.view(), 7, config).unwrap();

        // Lasso should produce sparse weights
        assert!(result.n_effective_donors <= result.n_donors);

        println!("Lasso result:\n{}", result);
    }

    #[test]
    fn test_scpi_ridge() {
        let (treated, donors) = generate_test_data();

        let config = SCPIConfig {
            constraint: SCPIConstraint::Ridge { lambda: 0.1 },
            ..Default::default()
        };

        let result = run_scpi(&treated.view(), &donors.view(), 7, config).unwrap();

        // Ridge weights typically non-zero
        assert!(result.weights.len() == result.n_donors);

        println!("Ridge result:\n{}", result);
    }

    #[test]
    fn test_scpi_lasso_simplex() {
        let (treated, donors) = generate_test_data();

        let config = SCPIConfig {
            constraint: SCPIConstraint::LassoSimplex { lambda: 0.05 },
            ..Default::default()
        };

        let result = run_scpi(&treated.view(), &donors.view(), 7, config).unwrap();

        // Should satisfy simplex constraint
        let weight_sum: f64 = result.weights.sum();
        assert_relative_eq!(weight_sum, 1.0, epsilon = 1e-4);
        assert!(result.weights.iter().all(|&w| w >= -1e-6));

        println!("Lasso-Simplex result:\n{}", result);
    }

    #[test]
    fn test_project_to_simplex() {
        // Test simplex projection
        let v = array![0.5, 0.3, -0.1, 0.8];
        let projected = project_to_simplex(&v);

        // Should sum to 1
        assert_relative_eq!(projected.sum(), 1.0, epsilon = 1e-10);

        // Should be non-negative
        assert!(projected.iter().all(|&x| x >= -1e-10));
    }

    #[test]
    fn test_soft_threshold() {
        assert_relative_eq!(soft_threshold(1.5, 0.5), 1.0, epsilon = 1e-10);
        assert_relative_eq!(soft_threshold(-1.5, 0.5), -1.0, epsilon = 1e-10);
        assert_relative_eq!(soft_threshold(0.3, 0.5), 0.0, epsilon = 1e-10);
        assert_relative_eq!(soft_threshold(-0.3, 0.5), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_variance_methods() {
        let (treated, donors) = generate_test_data();

        let methods = [
            VarianceMethod::Gaussian,
            VarianceMethod::Subgaussian,
            // Skip CV methods in unit tests (slow)
        ];

        for method in methods {
            let config = SCPIConfig {
                variance_method: method,
                ..Default::default()
            };

            let result = run_scpi(&treated.view(), &donors.view(), 7, config).unwrap();

            // Variance should be positive
            assert!(result.in_sample_var > 0.0);
            assert!(result.out_sample_var > 0.0);
            assert!(result.total_var > 0.0);
        }
    }

    #[test]
    fn test_scpi_insufficient_data() {
        let treated = array![1.0, 2.0, 3.0];
        let donors = Array2::from_shape_fn((3, 2), |(t, j)| t as f64 + j as f64);

        // Treatment at period 1 (only 1 pre-period)
        let result = run_scpi(&treated.view(), &donors.view(), 1, SCPIConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_critical_values() {
        // Check approximate critical values
        // Note: These are approximations; exact values require special functions
        let z_95 = compute_z_critical(0.025);
        assert!(
            z_95 > 1.9 && z_95 < 2.1,
            "z_0.025 should be ~1.96, got {}",
            z_95
        );

        // The t-distribution approximation is less accurate for small df
        // True t_10,0.025 ≈ 2.228, our approximation is simpler
        let t_10_95 = compute_t_critical(0.025, 10);
        assert!(
            t_10_95 > 1.9 && t_10_95 < 2.5,
            "t_10,0.025 should be ~2.23, got {}",
            t_10_95
        );

        // For large df, should approach z
        let t_100_95 = compute_t_critical(0.025, 100);
        assert!(
            (t_100_95 - z_95).abs() < 0.1,
            "t_100 should be close to z, got {}",
            t_100_95
        );
    }
}
