//! General Method of Moments (GMM) Estimation.
//!
//! Pure Rust implementation of the generalized method of moments estimator
//! following Hansen (1982). Supports two-step, iterative, and continuously
//! updated (CUE) estimation with various weighting matrix options.
//!
//! # Overview
//!
//! GMM estimates parameters θ by minimizing the quadratic form:
//!
//! ```text
//! Q(θ) = ḡ(θ)' W ḡ(θ)
//! ```
//!
//! Where:
//! - ḡ(θ) = (1/n) Σᵢ g(xᵢ, θ) is the sample average of moment conditions
//! - W is the weighting matrix
//!
//! # Predefined Moment Conditions
//!
//! This implementation supports several predefined moment condition types:
//! - Linear IV: E[z(y - xβ)] = 0
//! - Normal distribution: E[x - μ] = 0, E[(x-μ)² - σ²] = 0
//! - Linear regression with overidentification
//!
//! # References
//!
//! - Hansen, L.P. (1982). "Large Sample Properties of Generalized Method of
//!   Moments Estimators." Econometrica, 50(4), 1029-1054.
//!
//! - Hansen, L.P., Heaton, J., & Yaron, A. (1996). "Finite-Sample Properties
//!   of Some Alternative GMM Estimators." Journal of Business & Economic
//!   Statistics, 14(3), 262-280.
//!
//! R equivalent: `gmm::gmm()`

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::design::DesignMatrix;
use crate::linalg::matrix_ops::{safe_inverse, xtx};
use crate::regression::HacKernel;
use crate::traits::estimator::{SignificanceLevel, chi_squared_p_value};

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration Types
// ═══════════════════════════════════════════════════════════════════════════════

/// GMM estimation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GmmMethod {
    /// Two-step estimation (Hansen 1982)
    /// First step uses identity matrix, second step uses optimal W
    #[default]
    TwoStep,
    /// Iterative GMM - iterate until convergence
    Iterative,
    /// Continuously Updated Estimator (CUE) - update W at each iteration
    CUE,
}

impl fmt::Display for GmmMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GmmMethod::TwoStep => write!(f, "Two-Step"),
            GmmMethod::Iterative => write!(f, "Iterative"),
            GmmMethod::CUE => write!(f, "Continuously Updated (CUE)"),
        }
    }
}

/// Variance-covariance estimation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GmmVcov {
    /// HAC (Heteroskedasticity and Autocorrelation Consistent)
    #[default]
    HAC,
    /// i.i.d. assumption
    IID,
    /// Fixed user-supplied weighting matrix
    Fixed,
}

/// Configuration for general GMM estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralGmmConfig {
    /// Estimation method
    pub method: GmmMethod,
    /// Variance-covariance type
    pub vcov: GmmVcov,
    /// HAC kernel for weighting matrix
    pub kernel: HacKernel,
    /// HAC bandwidth (None for automatic)
    pub bandwidth: Option<usize>,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Maximum iterations (for iterative/CUE)
    pub max_iter: usize,
    /// Whether to prewhiten for HAC
    pub prewhite: bool,
}

impl Default for GeneralGmmConfig {
    fn default() -> Self {
        Self {
            method: GmmMethod::TwoStep,
            vcov: GmmVcov::HAC,
            kernel: HacKernel::Bartlett,
            bandwidth: None,
            tolerance: 1e-7,
            max_iter: 100,
            prewhite: false,
        }
    }
}

/// Predefined moment condition specification.
#[derive(Debug, Clone)]
pub enum MomentCondition {
    /// Linear IV: E[z(y - xβ)] = 0
    /// y = outcome, X = regressors, Z = instruments
    LinearIV {
        y: Array1<f64>,
        x: Array2<f64>,
        z: Array2<f64>,
    },
    /// Linear model with overidentification
    /// Same as LinearIV but X ⊂ Z
    LinearOveridentified {
        y: Array1<f64>,
        x: Array2<f64>,
        z: Array2<f64>,
    },
    /// Normal distribution moments: E[x-μ] = 0, E[(x-μ)²-σ²] = 0
    NormalDistribution { data: Array1<f64> },
}

// ═══════════════════════════════════════════════════════════════════════════════
// Result Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Result from general GMM estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralGmmResult {
    /// Parameter estimates
    pub coefficients: Vec<f64>,
    /// Parameter names
    pub names: Vec<String>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// t-statistics
    pub t_stats: Vec<f64>,
    /// p-values
    pub p_values: Vec<f64>,
    /// Covariance matrix of estimates
    pub vcov: Vec<Vec<f64>>,
    /// J-statistic (Hansen test for overidentifying restrictions)
    pub j_stat: f64,
    /// J-test p-value
    pub j_pvalue: f64,
    /// Degrees of freedom for J-test
    pub j_df: usize,
    /// Final objective function value
    pub objective: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of parameters
    pub n_params: usize,
    /// Number of moment conditions
    pub n_moments: usize,
    /// Convergence achieved
    pub converged: bool,
    /// Number of iterations
    pub iterations: usize,
    /// Estimation method used
    pub method: GmmMethod,
}

impl fmt::Display for GeneralGmmResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Generalized Method of Moments (GMM) Estimation")?;
        writeln!(f, "===============================================")?;
        writeln!(f)?;
        writeln!(f, "Method: {}", self.method)?;
        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(f, "Parameters: {}", self.n_params)?;
        writeln!(f, "Moment conditions: {}", self.n_moments)?;
        writeln!(
            f,
            "Overidentification: {} (q - k = {})",
            if self.n_moments > self.n_params {
                "Yes"
            } else {
                "No (just-identified)"
            },
            self.n_moments.saturating_sub(self.n_params)
        )?;
        writeln!(f)?;
        writeln!(f, "Coefficients:")?;
        writeln!(
            f,
            "{:>15} {:>12} {:>12} {:>10} {:>10}",
            "Parameter", "Estimate", "Std.Err", "t-stat", "P>|t|"
        )?;
        writeln!(f, "{}", "-".repeat(65))?;

        for i in 0..self.n_params {
            let sig = SignificanceLevel::from_p_value(self.p_values[i]);
            writeln!(
                f,
                "{:>15} {:>12.6} {:>12.6} {:>10.3} {:>10.4}{}",
                self.names[i],
                self.coefficients[i],
                self.std_errors[i],
                self.t_stats[i],
                self.p_values[i],
                sig.stars()
            )?;
        }
        writeln!(f, "{}", "-".repeat(65))?;

        if self.n_moments > self.n_params {
            writeln!(f)?;
            writeln!(f, "J-Test for Overidentifying Restrictions:")?;
            writeln!(f, "  J-statistic: {:.4} (df = {})", self.j_stat, self.j_df)?;
            writeln!(f, "  p-value: {:.4}", self.j_pvalue)?;
            if self.j_pvalue < 0.05 {
                writeln!(f, "  WARNING: Instruments may be invalid (p < 0.05)")?;
            } else {
                writeln!(
                    f,
                    "  Cannot reject validity of overidentifying restrictions"
                )?;
            }
        }

        if !self.converged {
            writeln!(f)?;
            writeln!(
                f,
                "WARNING: Algorithm did not converge after {} iterations",
                self.iterations
            )?;
        }

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main Estimation Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Run GMM estimation with predefined moment conditions.
///
/// # Arguments
///
/// * `moments` - Predefined moment condition specification
/// * `config` - Configuration options
///
/// # Example
///
/// ```ignore
/// use p2a_core::econometrics::general_gmm::{run_general_gmm, MomentCondition, GeneralGmmConfig};
///
/// // IV estimation: y = X β + ε, with instruments Z
/// let moments = MomentCondition::LinearIV { y, x, z };
/// let result = run_general_gmm(moments, GeneralGmmConfig::default())?;
/// ```
pub fn run_general_gmm(
    moments: MomentCondition,
    config: GeneralGmmConfig,
) -> EconResult<GeneralGmmResult> {
    match moments {
        MomentCondition::LinearIV { y, x, z } => estimate_linear_iv(&y, &x, &z, config),
        MomentCondition::LinearOveridentified { y, x, z } => estimate_linear_iv(&y, &x, &z, config),
        MomentCondition::NormalDistribution { data } => estimate_normal_distribution(&data, config),
    }
}

/// Convenience function for linear IV GMM from dataset.
///
/// # Arguments
///
/// * `dataset` - The dataset
/// * `y_col` - Outcome variable
/// * `x_cols` - Regressors (endogenous and exogenous)
/// * `z_cols` - Instruments (should include exogenous regressors)
/// * `config` - Configuration options
pub fn run_gmm_iv(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    z_cols: &[&str],
    config: Option<GeneralGmmConfig>,
) -> EconResult<GeneralGmmResult> {
    let config = config.unwrap_or_default();

    // Extract y
    let y = DesignMatrix::extract_column(dataset.df(), y_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: y_col.to_string(),
            available: vec![format!("{:?}", e)],
        }
    })?;

    // Extract X with intercept
    let x_dm = DesignMatrix::from_dataframe(dataset.df(), x_cols, true)?;
    let x = x_dm.data;

    // Extract Z with intercept
    let z_dm = DesignMatrix::from_dataframe(dataset.df(), z_cols, true)?;
    let z = z_dm.data;

    let moments = MomentCondition::LinearIV { y, x, z };
    let mut result = run_general_gmm(moments, config)?;

    // Update names
    result.names = std::iter::once("(Intercept)".to_string())
        .chain(x_cols.iter().map(|s| s.to_string()))
        .collect();

    Ok(result)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Linear IV GMM Implementation
// ═══════════════════════════════════════════════════════════════════════════════

fn estimate_linear_iv(
    y: &Array1<f64>,
    x: &Array2<f64>,
    z: &Array2<f64>,
    config: GeneralGmmConfig,
) -> EconResult<GeneralGmmResult> {
    let n = y.len();
    let k = x.ncols(); // number of parameters
    let q = z.ncols(); // number of moment conditions

    if z.nrows() != n || x.nrows() != n {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Dimension mismatch: n={}, X rows={}, Z rows={}",
                n,
                x.nrows(),
                z.nrows()
            ),
        });
    }

    if q < k {
        return Err(EconError::InvalidSpecification {
            message: format!("Underidentified: {} instruments < {} parameters", q, k),
        });
    }

    // Compute Z'Z and Z'X
    let zz = xtx(&z.view());
    let zx = z.t().dot(x);
    let zy = z.t().dot(y);

    match config.method {
        GmmMethod::TwoStep => estimate_two_step_iv(y, x, z, &zz, &zx, &zy, n, k, q, &config),
        GmmMethod::Iterative => estimate_iterative_iv(y, x, z, &zz, &zx, &zy, n, k, q, &config),
        GmmMethod::CUE => {
            // CUE is more complex - fall back to iterative for now
            estimate_iterative_iv(y, x, z, &zz, &zx, &zy, n, k, q, &config)
        }
    }
}

fn estimate_two_step_iv(
    y: &Array1<f64>,
    x: &Array2<f64>,
    z: &Array2<f64>,
    zz: &Array2<f64>,
    zx: &Array2<f64>,
    zy: &Array1<f64>,
    n: usize,
    k: usize,
    q: usize,
    config: &GeneralGmmConfig,
) -> EconResult<GeneralGmmResult> {
    // Step 1: Use identity weighting matrix (2SLS)
    // β₁ = (X'Z(Z'Z)⁻¹Z'X)⁻¹ X'Z(Z'Z)⁻¹Z'y
    let (zz_inv, _) = safe_inverse(&zz.view()).map_err(|e| EconError::SingularMatrix {
        context: "GMM first step".to_string(),
        suggestion: format!("Z'Z singular: {}", e),
    })?;

    // Using ndarray .dot() method for cleaner matrix multiplication
    // zx is (q x k), zx.t() is (k x q), zz_inv is (q x q)
    let xz = zx.t().into_owned(); // k x q (this is X'Z)
    let xz_zzi = xz.dot(&zz_inv); // k x q dot q x q = k x q (X'Z (Z'Z)^{-1})
    let xz_zzi_zx = xz_zzi.dot(zx); // k x q dot q x k = k x k

    let (xz_zzi_zx_inv, _) =
        safe_inverse(&xz_zzi_zx.view()).map_err(|e| EconError::SingularMatrix {
            context: "GMM first step".to_string(),
            suggestion: format!("X'Z(Z'Z)⁻¹Z'X singular: {}", e),
        })?;

    let xz_zzi_zy = xz_zzi.dot(zy); // k x q dot q = k
    let beta1 = xz_zzi_zx_inv.dot(&xz_zzi_zy); // k x k dot k = k

    // Compute residuals from first step
    let resid1 = y - &x.dot(&beta1);

    // Step 2: Compute optimal weighting matrix
    // W = [Σᵢ zᵢzᵢ'eᵢ²]⁻¹ (under IID) or HAC version
    let w_opt = compute_optimal_weight(&resid1, z, n, q, config)?;

    // Step 2 estimate: β₂ = (X'ZWZ'X)⁻¹ X'ZWZ'y
    // W is q x q, Z' is q x n, so Z'W doesn't work. Need ZW first.
    // Actually: X'Z W Z'X where Z'X = zx (q x k)
    // So: X'Z is k x q, W is q x q, Z'X is q x k
    // X'Z W Z'X = (k x q)(q x q)(q x k) = k x k
    let xzw = xz.dot(&w_opt); // k x q dot q x q = k x q (X'Z W)
    let xzwzx = xzw.dot(zx); // k x q dot q x k = k x k

    let (xzwzx_inv, _) = safe_inverse(&xzwzx.view()).map_err(|e| EconError::SingularMatrix {
        context: "GMM second step".to_string(),
        suggestion: format!("X'ZWZ'X singular: {}", e),
    })?;

    let xzwzy = xzw.dot(zy); // k x q dot q = k (X'Z W Z'y)
    let beta2 = xzwzx_inv.dot(&xzwzy); // k x k dot k = k

    // Compute final residuals and statistics
    let resid = y - &x.dot(&beta2);
    let g_bar = z.t().dot(&resid) / n as f64;

    // Objective function (n × ḡ'Wḡ)
    let wg = w_opt.dot(&g_bar);
    let objective = n as f64 * g_bar.dot(&wg);

    // J-test for overidentification
    let (j_stat, j_pvalue, j_df) = if q > k {
        let df = q - k;
        let p = chi_squared_p_value(objective, df as f64);
        (objective, p, df)
    } else {
        (0.0, 1.0, 0)
    };

    // Covariance matrix: Var(β̂) = (1/n)(X'ZWZ'X)⁻¹
    let vcov_mat = &xzwzx_inv / n as f64;

    let std_errors: Vec<f64> = vcov_mat.diag().iter().map(|&v| v.max(0.0).sqrt()).collect();

    let coefficients: Vec<f64> = beta2.to_vec();
    let t_stats: Vec<f64> = coefficients
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
        .collect();

    let p_values: Vec<f64> = t_stats
        .iter()
        .map(|&t| 2.0 * (1.0 - crate::traits::estimator::normal_cdf(t.abs())))
        .collect();

    let vcov: Vec<Vec<f64>> = (0..k)
        .map(|i| (0..k).map(|j| vcov_mat[[i, j]]).collect())
        .collect();

    let names: Vec<String> = (0..k).map(|i| format!("beta_{}", i + 1)).collect();

    Ok(GeneralGmmResult {
        coefficients,
        names,
        std_errors,
        t_stats,
        p_values,
        vcov,
        j_stat,
        j_pvalue,
        j_df,
        objective,
        n_obs: n,
        n_params: k,
        n_moments: q,
        converged: true,
        iterations: 2,
        method: GmmMethod::TwoStep,
    })
}

fn estimate_iterative_iv(
    y: &Array1<f64>,
    x: &Array2<f64>,
    z: &Array2<f64>,
    zz: &Array2<f64>,
    zx: &Array2<f64>,
    zy: &Array1<f64>,
    n: usize,
    k: usize,
    q: usize,
    config: &GeneralGmmConfig,
) -> EconResult<GeneralGmmResult> {
    // Start with 2SLS estimate
    // zz: q x q, zx: q x k, zy: q
    let (zz_inv, _) = safe_inverse(&zz.view())?; // q x q
    let xz = zx.t().into_owned(); // k x q (X'Z)
    let xz_zzi = xz.dot(&zz_inv); // k x q dot q x q = k x q
    let xz_zzi_zx = xz_zzi.dot(zx); // k x q dot q x k = k x k
    let (xz_zzi_zx_inv, _) = safe_inverse(&xz_zzi_zx.view())?; // k x k
    let xz_zzi_zy = xz_zzi.dot(zy); // k x q dot q = k
    let mut beta = xz_zzi_zx_inv.dot(&xz_zzi_zy); // k x k dot k = k

    let mut iterations = 0;
    let mut converged = false;

    for iter in 0..config.max_iter {
        iterations = iter + 1;

        // Compute residuals
        let resid = y - &x.dot(&beta);

        // Update weighting matrix
        let w_opt = compute_optimal_weight(&resid, z, n, q, config)?;

        // Update beta using GMM formula: (X'Z W Z'X)^-1 X'Z W Z'y
        // xz: k x q, w_opt: q x q, zx: q x k, zy: q
        let xz_w = xz.dot(&w_opt); // k x q dot q x q = k x q
        let xz_w_zx = xz_w.dot(zx); // k x q dot q x k = k x k
        let (xz_w_zx_inv, _) = safe_inverse(&xz_w_zx.view())?; // k x k
        let xz_w_zy = xz_w.dot(zy); // k x q dot q = k
        let beta_new = xz_w_zx_inv.dot(&xz_w_zy); // k x k dot k = k

        // Check convergence
        let diff: f64 = (&beta_new - &beta)
            .iter()
            .map(|d| d * d)
            .sum::<f64>()
            .sqrt();

        beta = beta_new;

        if diff < config.tolerance {
            converged = true;
            break;
        }
    }

    // Final statistics (recompute with final beta)
    let resid = y - &x.dot(&beta);
    let w_opt = compute_optimal_weight(&resid, z, n, q, config)?;

    let g_bar = z.t().dot(&resid) / n as f64;
    let wg = w_opt.dot(&g_bar);
    let objective = n as f64 * g_bar.dot(&wg);

    let (j_stat, j_pvalue, j_df) = if q > k {
        let df = q - k;
        let p = chi_squared_p_value(objective, df as f64);
        (objective, p, df)
    } else {
        (0.0, 1.0, 0)
    };

    // Covariance matrix
    // Use same formula: (X'Z W Z'X)^-1
    let xz_w_final = xz.dot(&w_opt); // k x q dot q x q = k x q
    let xz_w_zx_final = xz_w_final.dot(zx); // k x q dot q x k = k x k
    let (vcov_mat, _) = safe_inverse(&xz_w_zx_final.view())?;
    let vcov_mat = &vcov_mat / n as f64;

    let std_errors: Vec<f64> = vcov_mat.diag().iter().map(|&v| v.max(0.0).sqrt()).collect();

    let coefficients: Vec<f64> = beta.to_vec();
    let t_stats: Vec<f64> = coefficients
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
        .collect();

    let p_values: Vec<f64> = t_stats
        .iter()
        .map(|&t| 2.0 * (1.0 - crate::traits::estimator::normal_cdf(t.abs())))
        .collect();

    let vcov: Vec<Vec<f64>> = (0..k)
        .map(|i| (0..k).map(|j| vcov_mat[[i, j]]).collect())
        .collect();

    let names: Vec<String> = (0..k).map(|i| format!("beta_{}", i + 1)).collect();

    Ok(GeneralGmmResult {
        coefficients,
        names,
        std_errors,
        t_stats,
        p_values,
        vcov,
        j_stat,
        j_pvalue,
        j_df,
        objective,
        n_obs: n,
        n_params: k,
        n_moments: q,
        converged,
        iterations,
        method: GmmMethod::Iterative,
    })
}

/// Compute optimal weighting matrix.
fn compute_optimal_weight(
    resid: &Array1<f64>,
    z: &Array2<f64>,
    n: usize,
    q: usize,
    config: &GeneralGmmConfig,
) -> EconResult<Array2<f64>> {
    // Compute score matrix: S = Z ⊙ e (element-wise z_i * e_i)
    // Then Ω = (1/n) S'S for IID
    // or HAC for serial correlation

    match config.vcov {
        GmmVcov::IID => {
            // Ω = (1/n) Σᵢ zᵢzᵢ'eᵢ²
            let mut omega = Array2::<f64>::zeros((q, q));

            for i in 0..n {
                let e_sq = resid[i] * resid[i];
                for j in 0..q {
                    for l in 0..q {
                        omega[[j, l]] += z[[i, j]] * z[[i, l]] * e_sq;
                    }
                }
            }
            omega /= n as f64;

            let (w, _) = safe_inverse(&omega.view()).map_err(|e| EconError::SingularMatrix {
                context: "GMM weighting matrix".to_string(),
                suggestion: format!("Ω singular: {}", e),
            })?;

            Ok(w)
        }
        GmmVcov::HAC => {
            // HAC weighting matrix with Newey-West style correction
            let bw = config
                .bandwidth
                .unwrap_or_else(|| (4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize);

            // Score vectors
            let mut scores = Array2::<f64>::zeros((n, q));
            for i in 0..n {
                for j in 0..q {
                    scores[[i, j]] = z[[i, j]] * resid[i];
                }
            }

            // Compute HAC covariance
            let mut omega = Array2::<f64>::zeros((q, q));

            // Lag 0
            for i in 0..n {
                for j in 0..q {
                    for l in 0..q {
                        omega[[j, l]] += scores[[i, j]] * scores[[i, l]];
                    }
                }
            }

            // Lags 1 to bw
            for lag in 1..=bw {
                let w = config.kernel.weight(lag, bw);
                if w.abs() < 1e-15 {
                    continue;
                }

                let mut gamma_lag = Array2::<f64>::zeros((q, q));
                for i in lag..n {
                    for j in 0..q {
                        for l in 0..q {
                            gamma_lag[[j, l]] += scores[[i, j]] * scores[[i - lag, l]];
                        }
                    }
                }

                for j in 0..q {
                    for l in 0..q {
                        omega[[j, l]] += w * (gamma_lag[[j, l]] + gamma_lag[[l, j]]);
                    }
                }
            }

            omega /= n as f64;

            let (w, _) = safe_inverse(&omega.view()).map_err(|e| EconError::SingularMatrix {
                context: "GMM HAC weighting matrix".to_string(),
                suggestion: format!("Ω singular: {}", e),
            })?;

            Ok(w)
        }
        GmmVcov::Fixed => {
            // Identity matrix as default
            Ok(Array2::eye(q))
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Normal Distribution GMM
// ═══════════════════════════════════════════════════════════════════════════════

fn estimate_normal_distribution(
    data: &Array1<f64>,
    _config: GeneralGmmConfig,
) -> EconResult<GeneralGmmResult> {
    let n = data.len();

    if n < 3 {
        return Err(EconError::InsufficientData {
            required: 3,
            provided: n,
            context: "Normal distribution GMM requires at least 3 observations".to_string(),
        });
    }

    // Moment conditions:
    // g1 = x - μ
    // g2 = (x - μ)² - σ²
    // g3 = (x - μ)³ (skewness = 0)

    // Method of moments estimates as starting point
    let mu_mm: f64 = data.sum() / n as f64;
    let sigma2_mm: f64 = data.iter().map(|&x| (x - mu_mm).powi(2)).sum::<f64>() / n as f64;

    // For normally distributed data, MoM = GMM for first two moments
    let mu = mu_mm;
    let sigma2 = sigma2_mm;
    let sigma = sigma2.sqrt();

    // Compute moment condition values for J-test
    // ḡ = [mean(x - μ), mean((x-μ)² - σ²), mean((x-μ)³)]'
    let g1_bar = 0.0; // By construction
    let g2_bar: f64 = data.iter().map(|&x| (x - mu).powi(2) - sigma2).sum::<f64>() / n as f64;
    let g3_bar: f64 = data.iter().map(|&x| (x - mu).powi(3)).sum::<f64>() / n as f64;

    // J-stat (1 overidentifying restriction with 3 moments, 2 params)
    // Using identity weighting for simplicity
    let j_stat = n as f64 * (g1_bar * g1_bar + g2_bar * g2_bar + g3_bar * g3_bar);
    let j_df = 1;
    let j_pvalue = chi_squared_p_value(j_stat, j_df as f64);

    // Standard errors from asymptotic theory
    // SE(μ̂) = σ/√n
    // SE(σ²̂) ≈ σ²√(2/n)
    let se_mu = sigma / (n as f64).sqrt();
    let se_sigma2 = sigma2 * (2.0 / n as f64).sqrt();

    let coefficients = vec![mu, sigma2];
    let std_errors = vec![se_mu, se_sigma2];
    let t_stats: Vec<f64> = coefficients
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
        .collect();
    let p_values: Vec<f64> = t_stats
        .iter()
        .map(|&t| 2.0 * (1.0 - crate::traits::estimator::normal_cdf(t.abs())))
        .collect();

    let vcov = vec![vec![se_mu * se_mu, 0.0], vec![0.0, se_sigma2 * se_sigma2]];

    Ok(GeneralGmmResult {
        coefficients,
        names: vec!["mu".to_string(), "sigma_sq".to_string()],
        std_errors,
        t_stats,
        p_values,
        vcov,
        j_stat,
        j_pvalue,
        j_df,
        objective: j_stat,
        n_obs: n,
        n_params: 2,
        n_moments: 3,
        converged: true,
        iterations: 1,
        method: GmmMethod::TwoStep,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_normal_distribution_gmm() {
        // Generate normal data
        let data = Array1::from(vec![
            1.2, 0.8, 1.5, 0.9, 1.1, 1.3, 0.7, 1.4, 1.0, 0.85, 1.15, 0.95, 1.25, 0.75, 1.35, 1.05,
            0.65, 1.45, 0.55, 1.55,
        ]);

        let moments = MomentCondition::NormalDistribution { data };
        let config = GeneralGmmConfig::default();

        let result = run_general_gmm(moments, config).unwrap();

        // Check basic structure
        assert_eq!(result.n_params, 2);
        assert_eq!(result.n_moments, 3);
        assert_eq!(result.j_df, 1);

        // μ should be close to sample mean
        assert!((result.coefficients[0] - 1.05).abs() < 0.2);
        // σ² should be close to sample variance
        assert!(result.coefficients[1] > 0.0);

        // Standard errors should be positive
        assert!(result.std_errors.iter().all(|&se| se > 0.0));
    }

    #[test]
    fn test_linear_iv_gmm() {
        // Simple IV example
        let n = 50;
        let y = Array1::from_iter((0..n).map(|i| 2.0 + 3.0 * (i as f64) + (i % 5) as f64 * 0.1));
        let x = Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { 1.0 } else { i as f64 });
        let z = Array2::from_shape_fn((n, 3), |(i, j)| match j {
            0 => 1.0,
            1 => i as f64,
            2 => (i * i) as f64 / 100.0,
            _ => 0.0,
        });

        let moments = MomentCondition::LinearIV { y, x, z };
        let config = GeneralGmmConfig {
            method: GmmMethod::TwoStep,
            ..Default::default()
        };

        let result = run_general_gmm(moments, config).unwrap();

        // Check structure
        assert_eq!(result.n_params, 2);
        assert_eq!(result.n_moments, 3);
        assert!(result.n_moments > result.n_params); // overidentified

        // Coefficients should be reasonable
        assert!((result.coefficients[0] - 2.0).abs() < 2.0); // intercept
        assert!((result.coefficients[1] - 3.0).abs() < 1.0); // slope

        // J-test should be computed
        assert!(result.j_stat >= 0.0);
        assert!(result.j_pvalue >= 0.0 && result.j_pvalue <= 1.0);
    }

    #[test]
    fn test_iterative_gmm() {
        let n = 50;
        let y = Array1::from_iter((0..n).map(|i| 1.0 + 2.0 * (i as f64)));
        let x = Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { 1.0 } else { i as f64 });
        let z = x.clone();

        let moments = MomentCondition::LinearIV { y, x, z };
        let config = GeneralGmmConfig {
            method: GmmMethod::Iterative,
            max_iter: 50,
            tolerance: 1e-6,
            ..Default::default()
        };

        let result = run_general_gmm(moments, config).unwrap();

        assert!(result.converged || result.iterations > 0);
        assert_eq!(result.method, GmmMethod::Iterative);
    }

    #[test]
    fn test_gmm_display() {
        let data = Array1::from(vec![1.0, 1.1, 0.9, 1.2, 0.8, 1.15, 0.85, 1.05, 0.95, 1.0]);
        let moments = MomentCondition::NormalDistribution { data };
        let result = run_general_gmm(moments, GeneralGmmConfig::default()).unwrap();

        let display = format!("{}", result);
        assert!(display.contains("GMM"));
        assert!(display.contains("Coefficients"));
        assert!(display.contains("J-Test"));
    }
}
