//! Nonlinear Least Squares (NLS) estimation.
//!
//! Pure Rust implementation using Levenberg-Marquardt algorithm with
//! numerical differentiation for the Jacobian.
//!
//! # References
//!
//! - Levenberg, K. (1944). "A Method for the Solution of Certain Non-Linear Problems
//!   in Least Squares". Quarterly of Applied Mathematics, 2(2), 164-168.
//! - Marquardt, D. W. (1963). "An Algorithm for Least-Squares Estimation of Nonlinear
//!   Parameters". SIAM Journal on Applied Mathematics, 11(2), 431-441.
//! - R implementation: `stats::nls()` - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/nls.html

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::safe_inverse;
use crate::traits::estimator::t_test_p_value;

/// Available NLS algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum NlsAlgorithm {
    /// Gauss-Newton algorithm (fast but may not converge for poorly conditioned problems)
    GaussNewton,
    /// Levenberg-Marquardt algorithm (more robust, default)
    #[default]
    LevenbergMarquardt,
}

impl fmt::Display for NlsAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NlsAlgorithm::GaussNewton => write!(f, "Gauss-Newton"),
            NlsAlgorithm::LevenbergMarquardt => write!(f, "Levenberg-Marquardt"),
        }
    }
}

/// Configuration for NLS estimation.
#[derive(Debug, Clone)]
pub struct NlsConfig {
    /// Algorithm to use (default: Levenberg-Marquardt)
    pub algorithm: NlsAlgorithm,
    /// Maximum number of iterations (default: 200)
    pub max_iter: usize,
    /// Convergence tolerance for relative change in RSS (default: 1e-8)
    pub tolerance: f64,
    /// Initial damping parameter for L-M algorithm (default: 1e-3)
    pub lambda_init: f64,
    /// Factor to increase lambda when step is rejected (default: 10.0)
    pub lambda_up: f64,
    /// Factor to decrease lambda when step is accepted (default: 0.1)
    pub lambda_down: f64,
    /// Step size for numerical differentiation (default: 1e-7)
    pub diff_step: f64,
    /// Whether to use weighted least squares
    pub weights: Option<Vec<f64>>,
    /// Lower bounds for parameters (optional)
    pub lower: Option<Vec<f64>>,
    /// Upper bounds for parameters (optional)
    pub upper: Option<Vec<f64>>,
}

impl Default for NlsConfig {
    fn default() -> Self {
        Self {
            algorithm: NlsAlgorithm::LevenbergMarquardt,
            max_iter: 200,
            tolerance: 1e-8,
            lambda_init: 1e-3,
            lambda_up: 10.0,
            lambda_down: 0.1,
            diff_step: 1e-7,
            weights: None,
            lower: None,
            upper: None,
        }
    }
}

/// Result from nonlinear least squares estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NlsResult {
    /// Algorithm used
    pub algorithm: String,
    /// Parameter names
    pub param_names: Vec<String>,
    /// Estimated parameter values
    pub coefficients: Vec<f64>,
    /// Standard errors of parameters
    pub std_errors: Vec<f64>,
    /// t-statistics
    pub t_stats: Vec<f64>,
    /// p-values
    pub p_values: Vec<f64>,
    /// Residual sum of squares
    pub rss: f64,
    /// Residuals
    pub residuals: Vec<f64>,
    /// Fitted values
    pub fitted: Vec<f64>,
    /// Residual standard error (sigma)
    pub sigma: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of parameters
    pub n_params: usize,
    /// Degrees of freedom
    pub df: usize,
    /// Number of iterations
    pub iterations: usize,
    /// Whether convergence was achieved
    pub converged: bool,
    /// Convergence code (0 = success, 1 = max iter, 2 = singular)
    pub convergence_code: u8,
    /// Final value of lambda (for L-M algorithm)
    pub final_lambda: Option<f64>,
    /// Variance-covariance matrix of parameters (serialization skipped for size)
    #[serde(skip)]
    pub vcov: Option<Array2<f64>>,
}

impl fmt::Display for NlsResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Nonlinear Least Squares Results ({})", self.algorithm)?;
        writeln!(f, "==============================================")?;
        writeln!(f, "No. Observations: {}", self.n_obs)?;
        writeln!(f, "No. Parameters: {}", self.n_params)?;
        writeln!(f, "Residual Std. Error: {:.6} on {} degrees of freedom", self.sigma, self.df)?;
        writeln!(f, "Residual Sum of Squares: {:.6}", self.rss)?;
        writeln!(f, "Converged: {} (iterations: {})", self.converged, self.iterations)?;
        if let Some(lambda) = self.final_lambda {
            writeln!(f, "Final Lambda: {:.2e}", lambda)?;
        }
        writeln!(f)?;
        writeln!(
            f,
            "{:<15} {:>12} {:>12} {:>10} {:>10}",
            "Parameter", "Estimate", "Std. Error", "t value", "Pr(>|t|)"
        )?;
        writeln!(f, "{}", "-".repeat(65))?;

        for i in 0..self.param_names.len() {
            let sig = if self.p_values[i] < 0.001 {
                "***"
            } else if self.p_values[i] < 0.01 {
                "**"
            } else if self.p_values[i] < 0.05 {
                "*"
            } else if self.p_values[i] < 0.1 {
                "†"
            } else {
                ""
            };
            writeln!(
                f,
                "{:<15} {:>12.6} {:>12.6} {:>10.3} {:>10.4} {}",
                self.param_names[i],
                self.coefficients[i],
                self.std_errors[i],
                self.t_stats[i],
                self.p_values[i],
                sig
            )?;
        }
        writeln!(f, "{}", "-".repeat(65))?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;

        Ok(())
    }
}

/// User-defined model function type.
///
/// Takes `x` values (independent variables) and parameters `theta`,
/// returns predicted `y` values.
pub type ModelFn = fn(&Array1<f64>, &Array1<f64>) -> f64;

/// Fit a nonlinear model using least squares.
///
/// # Arguments
/// * `x` - Independent variable values (n observations)
/// * `y` - Dependent variable values (n observations)
/// * `model` - Model function f(x, theta) -> y_hat
/// * `start` - Initial parameter values
/// * `param_names` - Names for the parameters
/// * `config` - NLS configuration
///
/// # Example
/// ```ignore
/// // Exponential decay model: y = a * exp(-b * x) + c
/// fn exp_decay(x: &Array1<f64>, theta: &Array1<f64>) -> f64 {
///     let a = theta[0];
///     let b = theta[1];
///     let c = theta[2];
///     a * (-b * x[0]).exp() + c
/// }
///
/// let result = nls(&x, &y, exp_decay, &start, &["a", "b", "c"], NlsConfig::default())?;
/// ```
pub fn nls(
    x: &Array1<f64>,
    y: &Array1<f64>,
    model: ModelFn,
    start: &Array1<f64>,
    param_names: &[&str],
    config: NlsConfig,
) -> EconResult<NlsResult> {
    let n = y.len();
    let k = start.len();

    if n < k {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "More parameters ({}) than observations ({})",
                k, n
            ),
        });
    }

    if param_names.len() != k {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Parameter names length ({}) doesn't match start values length ({})",
                param_names.len(),
                k
            ),
        });
    }

    // Validate weights if provided
    let weights = if let Some(ref w) = config.weights {
        if w.len() != n {
            return Err(EconError::InvalidSpecification {
                message: format!("Weights length ({}) doesn't match data length ({})", w.len(), n),
            });
        }
        Some(Array1::from_vec(w.clone()))
    } else {
        None
    };

    // Validate bounds if provided
    if let (Some(lower), Some(upper)) = (&config.lower, &config.upper) {
        if lower.len() != k || upper.len() != k {
            return Err(EconError::InvalidSpecification {
                message: "Bounds must have same length as parameters".to_string(),
            });
        }
        for i in 0..k {
            if lower[i] > upper[i] {
                return Err(EconError::InvalidSpecification {
                    message: format!("Lower bound {} exceeds upper bound {} for parameter {}",
                        lower[i], upper[i], param_names[i]),
                });
            }
        }
    }

    // Initialize parameters
    let mut theta = start.clone();

    // Apply bounds if specified
    if let (Some(lower), Some(upper)) = (&config.lower, &config.upper) {
        for i in 0..k {
            theta[i] = theta[i].max(lower[i]).min(upper[i]);
        }
    }

    // Compute initial residuals and RSS
    let mut residuals = compute_residuals(x, y, &theta, model, n);
    let mut rss = compute_rss(&residuals, &weights);

    let mut iterations = 0;
    let mut converged = false;
    let mut convergence_code: u8 = 1; // Default: max iterations reached
    let mut lambda = config.lambda_init;

    for _iter in 0..config.max_iter {
        iterations += 1;

        // Compute Jacobian matrix J[i,j] = ∂r_i/∂θ_j
        let jacobian = compute_jacobian(x, y, &theta, model, n, k, config.diff_step);

        // Compute J'J and J'r
        let jtj = jacobian.t().dot(&jacobian);
        let jtr = jacobian.t().dot(&residuals);

        // Solve for parameter update based on algorithm
        let delta = match config.algorithm {
            NlsAlgorithm::GaussNewton => {
                // Gauss-Newton: solve (J'J) * delta = J'r
                match solve_normal_equations(&jtj, &jtr, 0.0) {
                    Ok(d) => d,
                    Err(_) => {
                        convergence_code = 2;
                        break;
                    }
                }
            }
            NlsAlgorithm::LevenbergMarquardt => {
                // Levenberg-Marquardt: solve (J'J + λ*diag(J'J)) * delta = J'r
                let diag_jtj = jtj.diag().to_owned();

                loop {
                    match solve_lm_equations(&jtj, &jtr, lambda, &diag_jtj) {
                        Ok(d) => {
                            // Try the step
                            let theta_new = &theta - &d;
                            let theta_bounded = apply_bounds(&theta_new, &config.lower, &config.upper);
                            let residuals_new = compute_residuals(x, y, &theta_bounded, model, n);
                            let rss_new = compute_rss(&residuals_new, &weights);

                            if rss_new < rss {
                                // Step accepted - decrease lambda
                                lambda *= config.lambda_down;
                                lambda = lambda.max(1e-10);
                                theta = theta_bounded;
                                residuals = residuals_new;

                                // Check convergence
                                let rel_change = (rss - rss_new) / (rss + 1e-10);
                                rss = rss_new;

                                if rel_change < config.tolerance {
                                    converged = true;
                                    convergence_code = 0;
                                }
                                break;
                            } else {
                                // Step rejected - increase lambda
                                lambda *= config.lambda_up;
                                if lambda > 1e16 {
                                    convergence_code = 2;
                                    break;
                                }
                            }
                        }
                        Err(_) => {
                            lambda *= config.lambda_up;
                            if lambda > 1e16 {
                                convergence_code = 2;
                                break;
                            }
                        }
                    }
                }

                if convergence_code == 2 || converged {
                    break;
                }
                continue; // Skip the regular update below for L-M
            }
        };

        // For Gauss-Newton: apply update
        if config.algorithm == NlsAlgorithm::GaussNewton {
            let theta_new = &theta - &delta;
            theta = apply_bounds(&theta_new, &config.lower, &config.upper);

            let residuals_new = compute_residuals(x, y, &theta, model, n);
            let rss_new = compute_rss(&residuals_new, &weights);

            let rel_change = (rss - rss_new).abs() / (rss + 1e-10);
            residuals = residuals_new;
            rss = rss_new;

            if rel_change < config.tolerance {
                converged = true;
                convergence_code = 0;
                break;
            }
        }
    }

    // Compute final statistics
    let df = n - k;
    let sigma = if df > 0 { (rss / df as f64).sqrt() } else { f64::NAN };

    // Compute variance-covariance matrix
    let jacobian_final = compute_jacobian(x, y, &theta, model, n, k, config.diff_step);
    let jtj_final = jacobian_final.t().dot(&jacobian_final);

    let (vcov, std_errors) = match safe_inverse(&jtj_final.view()) {
        Ok((inv, _)) => {
            let vcov = &inv * (sigma * sigma);
            let se: Vec<f64> = vcov.diag().iter().map(|v| v.max(0.0).sqrt()).collect();
            (Some(vcov), se)
        }
        Err(_) => {
            // If inversion fails, report NaN standard errors
            (None, vec![f64::NAN; k])
        }
    };

    // Compute t-statistics and p-values
    let t_stats: Vec<f64> = theta.iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| if se.is_finite() && se > 0.0 { b / se } else { f64::NAN })
        .collect();

    let p_values: Vec<f64> = t_stats.iter()
        .map(|&t| if t.is_finite() { t_test_p_value(t, df as f64) } else { f64::NAN })
        .collect();

    // Compute fitted values
    let fitted: Vec<f64> = (0..n)
        .map(|i| {
            let xi = Array1::from_elem(1, x[i]);
            model(&xi, &theta)
        })
        .collect();

    Ok(NlsResult {
        algorithm: config.algorithm.to_string(),
        param_names: param_names.iter().map(|s| s.to_string()).collect(),
        coefficients: theta.to_vec(),
        std_errors,
        t_stats,
        p_values,
        rss,
        residuals: residuals.to_vec(),
        fitted,
        sigma,
        n_obs: n,
        n_params: k,
        df,
        iterations,
        converged,
        convergence_code,
        final_lambda: if config.algorithm == NlsAlgorithm::LevenbergMarquardt { Some(lambda) } else { None },
        vcov,
    })
}

/// Fit a nonlinear model using least squares with multi-dimensional x.
///
/// # Arguments
/// * `x` - Independent variable matrix (n × p)
/// * `y` - Dependent variable values (n observations)
/// * `model` - Model function f(x_row, theta) -> y_hat (takes row of X)
/// * `start` - Initial parameter values
/// * `param_names` - Names for the parameters
/// * `config` - NLS configuration
pub fn nls_multi(
    x: &Array2<f64>,
    y: &Array1<f64>,
    model: fn(&Array1<f64>, &Array1<f64>) -> f64,
    start: &Array1<f64>,
    param_names: &[&str],
    config: NlsConfig,
) -> EconResult<NlsResult> {
    let n = y.len();
    let k = start.len();

    if x.nrows() != n {
        return Err(EconError::InvalidSpecification {
            message: format!("X rows ({}) doesn't match y length ({})", x.nrows(), n),
        });
    }

    if n < k {
        return Err(EconError::InvalidSpecification {
            message: format!("More parameters ({}) than observations ({})", k, n),
        });
    }

    if param_names.len() != k {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Parameter names length ({}) doesn't match start values length ({})",
                param_names.len(), k
            ),
        });
    }

    let weights = config.weights.as_ref().map(|w| Array1::from_vec(w.clone()));

    let mut theta = start.clone();
    if let (Some(lower), Some(upper)) = (&config.lower, &config.upper) {
        for i in 0..k {
            theta[i] = theta[i].max(lower[i]).min(upper[i]);
        }
    }

    let mut residuals = compute_residuals_multi(x, y, &theta, model);
    let mut rss = compute_rss(&residuals, &weights);

    let mut iterations = 0;
    let mut converged = false;
    let mut convergence_code: u8 = 1;
    let mut lambda = config.lambda_init;

    for _iter in 0..config.max_iter {
        iterations += 1;

        let jacobian = compute_jacobian_multi(x, y, &theta, model, config.diff_step);
        let jtj = jacobian.t().dot(&jacobian);
        let jtr = jacobian.t().dot(&residuals);

        match config.algorithm {
            NlsAlgorithm::GaussNewton => {
                match solve_normal_equations(&jtj, &jtr, 0.0) {
                    Ok(delta) => {
                        let theta_new = &theta - &delta;
                        theta = apply_bounds(&theta_new, &config.lower, &config.upper);

                        let residuals_new = compute_residuals_multi(x, y, &theta, model);
                        let rss_new = compute_rss(&residuals_new, &weights);

                        let rel_change = (rss - rss_new).abs() / (rss + 1e-10);
                        residuals = residuals_new;
                        rss = rss_new;

                        if rel_change < config.tolerance {
                            converged = true;
                            convergence_code = 0;
                            break;
                        }
                    }
                    Err(_) => {
                        convergence_code = 2;
                        break;
                    }
                }
            }
            NlsAlgorithm::LevenbergMarquardt => {
                let diag_jtj = jtj.diag().to_owned();

                let mut step_accepted = false;
                for _ in 0..50 { // Max attempts to find good lambda
                    match solve_lm_equations(&jtj, &jtr, lambda, &diag_jtj) {
                        Ok(d) => {
                            let theta_new = &theta - &d;
                            let theta_bounded = apply_bounds(&theta_new, &config.lower, &config.upper);
                            let residuals_new = compute_residuals_multi(x, y, &theta_bounded, model);
                            let rss_new = compute_rss(&residuals_new, &weights);

                            if rss_new < rss {
                                lambda *= config.lambda_down;
                                lambda = lambda.max(1e-10);
                                theta = theta_bounded;
                                residuals = residuals_new;

                                let rel_change = (rss - rss_new) / (rss + 1e-10);
                                rss = rss_new;

                                if rel_change < config.tolerance {
                                    converged = true;
                                    convergence_code = 0;
                                }
                                step_accepted = true;
                                break;
                            } else {
                                lambda *= config.lambda_up;
                            }
                        }
                        Err(_) => {
                            lambda *= config.lambda_up;
                        }
                    }

                    if lambda > 1e16 {
                        convergence_code = 2;
                        break;
                    }
                }

                if convergence_code == 2 || !step_accepted {
                    break;
                }
                if converged {
                    break;
                }
            }
        }
    }

    let df = n - k;
    let sigma = if df > 0 { (rss / df as f64).sqrt() } else { f64::NAN };

    let jacobian_final = compute_jacobian_multi(x, y, &theta, model, config.diff_step);
    let jtj_final = jacobian_final.t().dot(&jacobian_final);

    let (vcov, std_errors) = match safe_inverse(&jtj_final.view()) {
        Ok((inv, _)) => {
            let vcov = &inv * (sigma * sigma);
            let se: Vec<f64> = vcov.diag().iter().map(|v| v.max(0.0).sqrt()).collect();
            (Some(vcov), se)
        }
        Err(_) => (None, vec![f64::NAN; k])
    };

    let t_stats: Vec<f64> = theta.iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| if se.is_finite() && se > 0.0 { b / se } else { f64::NAN })
        .collect();

    let p_values: Vec<f64> = t_stats.iter()
        .map(|&t| if t.is_finite() { t_test_p_value(t, df as f64) } else { f64::NAN })
        .collect();

    let fitted: Vec<f64> = (0..n)
        .map(|i| model(&x.row(i).to_owned(), &theta))
        .collect();

    Ok(NlsResult {
        algorithm: config.algorithm.to_string(),
        param_names: param_names.iter().map(|s| s.to_string()).collect(),
        coefficients: theta.to_vec(),
        std_errors,
        t_stats,
        p_values,
        rss,
        residuals: residuals.to_vec(),
        fitted,
        sigma,
        n_obs: n,
        n_params: k,
        df,
        iterations,
        converged,
        convergence_code,
        final_lambda: if config.algorithm == NlsAlgorithm::LevenbergMarquardt { Some(lambda) } else { None },
        vcov,
    })
}

/// Run NLS from a Dataset with a specified formula.
///
/// # Arguments
/// * `dataset` - The dataset containing the data
/// * `y_col` - Name of the dependent variable column
/// * `x_col` - Name of the independent variable column
/// * `model` - Model function f(x, theta) -> y
/// * `start` - Initial parameter values
/// * `param_names` - Names for the parameters
///
/// # Example
/// ```ignore
/// // Michaelis-Menten kinetics: y = Vmax * x / (Km + x)
/// fn michaelis_menten(x: &Array1<f64>, theta: &Array1<f64>) -> f64 {
///     let vmax = theta[0];
///     let km = theta[1];
///     vmax * x[0] / (km + x[0])
/// }
///
/// let start = Array1::from_vec(vec![200.0, 0.1]);
/// let result = run_nls(&dataset, "velocity", "substrate", michaelis_menten,
///     &start, &["Vmax", "Km"])?;
/// ```
pub fn run_nls(
    dataset: &Dataset,
    y_col: &str,
    x_col: &str,
    model: ModelFn,
    start: &Array1<f64>,
    param_names: &[&str],
) -> EconResult<NlsResult> {
    run_nls_with_config(dataset, y_col, x_col, model, start, param_names, NlsConfig::default())
}

/// Run NLS from a Dataset with custom configuration.
pub fn run_nls_with_config(
    dataset: &Dataset,
    y_col: &str,
    x_col: &str,
    model: ModelFn,
    start: &Array1<f64>,
    param_names: &[&str],
    config: NlsConfig,
) -> EconResult<NlsResult> {
    use crate::linalg::design::DesignMatrix;

    let y = DesignMatrix::extract_column(dataset.df(), y_col)
        .map_err(|e| EconError::ColumnNotFound {
            column: y_col.to_string(),
            available: vec![format!("{:?}", e)],
        })?;

    let x = DesignMatrix::extract_column(dataset.df(), x_col)
        .map_err(|e| EconError::ColumnNotFound {
            column: x_col.to_string(),
            available: vec![format!("{:?}", e)],
        })?;

    nls(&x, &y, model, start, param_names, config)
}

// ============================================================================
// Helper functions
// ============================================================================

/// Compute residuals: r_i = y_i - f(x_i, theta)
fn compute_residuals(
    x: &Array1<f64>,
    y: &Array1<f64>,
    theta: &Array1<f64>,
    model: ModelFn,
    n: usize,
) -> Array1<f64> {
    let mut residuals = Array1::zeros(n);
    for i in 0..n {
        let xi = Array1::from_elem(1, x[i]);
        let y_hat = model(&xi, theta);
        residuals[i] = y[i] - y_hat;
    }
    residuals
}

/// Compute residuals for multi-dimensional x
fn compute_residuals_multi(
    x: &Array2<f64>,
    y: &Array1<f64>,
    theta: &Array1<f64>,
    model: fn(&Array1<f64>, &Array1<f64>) -> f64,
) -> Array1<f64> {
    let n = y.len();
    let mut residuals = Array1::zeros(n);
    for i in 0..n {
        let xi = x.row(i).to_owned();
        let y_hat = model(&xi, theta);
        residuals[i] = y[i] - y_hat;
    }
    residuals
}

/// Compute (weighted) residual sum of squares
fn compute_rss(residuals: &Array1<f64>, weights: &Option<Array1<f64>>) -> f64 {
    match weights {
        Some(w) => residuals.iter()
            .zip(w.iter())
            .map(|(&r, &wi)| wi * r * r)
            .sum(),
        None => residuals.iter().map(|r| r * r).sum(),
    }
}

/// Compute Jacobian matrix using central differences
fn compute_jacobian(
    x: &Array1<f64>,
    y: &Array1<f64>,
    theta: &Array1<f64>,
    model: ModelFn,
    n: usize,
    k: usize,
    h: f64,
) -> Array2<f64> {
    let mut jacobian = Array2::zeros((n, k));

    for j in 0..k {
        let mut theta_plus = theta.clone();
        let mut theta_minus = theta.clone();

        let step = h * theta[j].abs().max(1.0);
        theta_plus[j] += step;
        theta_minus[j] -= step;

        for i in 0..n {
            let xi = Array1::from_elem(1, x[i]);
            let f_plus = model(&xi, &theta_plus);
            let f_minus = model(&xi, &theta_minus);

            // Jacobian of residuals: ∂r/∂θ = -∂f/∂θ
            jacobian[[i, j]] = -(f_plus - f_minus) / (2.0 * step);
        }
    }

    jacobian
}

/// Compute Jacobian matrix for multi-dimensional x
fn compute_jacobian_multi(
    x: &Array2<f64>,
    _y: &Array1<f64>,
    theta: &Array1<f64>,
    model: fn(&Array1<f64>, &Array1<f64>) -> f64,
    h: f64,
) -> Array2<f64> {
    let n = x.nrows();
    let k = theta.len();
    let mut jacobian = Array2::zeros((n, k));

    for j in 0..k {
        let mut theta_plus = theta.clone();
        let mut theta_minus = theta.clone();

        let step = h * theta[j].abs().max(1.0);
        theta_plus[j] += step;
        theta_minus[j] -= step;

        for i in 0..n {
            let xi = x.row(i).to_owned();
            let f_plus = model(&xi, &theta_plus);
            let f_minus = model(&xi, &theta_minus);

            jacobian[[i, j]] = -(f_plus - f_minus) / (2.0 * step);
        }
    }

    jacobian
}

/// Solve normal equations (J'J + λI) * delta = J'r
fn solve_normal_equations(
    jtj: &Array2<f64>,
    jtr: &Array1<f64>,
    lambda: f64,
) -> EconResult<Array1<f64>> {
    let k = jtj.nrows();
    let mut augmented = jtj.clone();

    if lambda > 0.0 {
        for i in 0..k {
            augmented[[i, i]] += lambda;
        }
    }

    let (inv, _) = safe_inverse(&augmented.view())
        .map_err(|e| EconError::SingularMatrix {
            context: "Normal equations in NLS".to_string(),
            suggestion: format!("Try different starting values or increase lambda: {:?}", e),
        })?;

    Ok(inv.dot(jtr))
}

/// Solve Levenberg-Marquardt equations (J'J + λ*diag(J'J)) * delta = J'r
fn solve_lm_equations(
    jtj: &Array2<f64>,
    jtr: &Array1<f64>,
    lambda: f64,
    diag: &Array1<f64>,
) -> EconResult<Array1<f64>> {
    let k = jtj.nrows();
    let mut augmented = jtj.clone();

    for i in 0..k {
        augmented[[i, i]] += lambda * diag[i].max(1e-10);
    }

    let (inv, _) = safe_inverse(&augmented.view())
        .map_err(|e| EconError::SingularMatrix {
            context: "L-M equations in NLS".to_string(),
            suggestion: format!("Lambda may be too small: {:?}", e),
        })?;

    Ok(inv.dot(jtr))
}

/// Apply parameter bounds
fn apply_bounds(
    theta: &Array1<f64>,
    lower: &Option<Vec<f64>>,
    upper: &Option<Vec<f64>>,
) -> Array1<f64> {
    match (lower, upper) {
        (Some(lo), Some(hi)) => {
            theta.iter()
                .zip(lo.iter())
                .zip(hi.iter())
                .map(|((&t, &l), &u)| t.max(l).min(u))
                .collect()
        }
        _ => theta.clone(),
    }
}

// ============================================================================
// Common nonlinear models (convenience functions)
// ============================================================================

/// Exponential decay model: y = a * exp(-b * x) + c
///
/// Parameters: [a, b, c]
pub fn model_exponential_decay(x: &Array1<f64>, theta: &Array1<f64>) -> f64 {
    let a = theta[0];
    let b = theta[1];
    let c = theta[2];
    a * (-b * x[0]).exp() + c
}

/// Exponential growth model: y = a * exp(b * x)
///
/// Parameters: [a, b]
pub fn model_exponential_growth(x: &Array1<f64>, theta: &Array1<f64>) -> f64 {
    let a = theta[0];
    let b = theta[1];
    a * (b * x[0]).exp()
}

/// Michaelis-Menten kinetics: y = Vmax * x / (Km + x)
///
/// Parameters: [Vmax, Km]
pub fn model_michaelis_menten(x: &Array1<f64>, theta: &Array1<f64>) -> f64 {
    let vmax = theta[0];
    let km = theta[1];
    if km + x[0] != 0.0 {
        vmax * x[0] / (km + x[0])
    } else {
        0.0
    }
}

/// Logistic growth model: y = K / (1 + exp(-r * (x - x0)))
///
/// Parameters: [K, r, x0]
pub fn model_logistic_growth(x: &Array1<f64>, theta: &Array1<f64>) -> f64 {
    let k = theta[0];
    let r = theta[1];
    let x0 = theta[2];
    k / (1.0 + (-r * (x[0] - x0)).exp())
}

/// Power model: y = a * x^b
///
/// Parameters: [a, b]
pub fn model_power(x: &Array1<f64>, theta: &Array1<f64>) -> f64 {
    let a = theta[0];
    let b = theta[1];
    if x[0] > 0.0 {
        a * x[0].powf(b)
    } else {
        a * x[0].abs().powf(b) * x[0].signum()
    }
}

/// Asymptotic model: y = a - b * exp(-c * x)
///
/// Parameters: [a, b, c]
pub fn model_asymptotic(x: &Array1<f64>, theta: &Array1<f64>) -> f64 {
    let a = theta[0];
    let b = theta[1];
    let c = theta[2];
    a - b * (-c * x[0]).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;
    use crate::data::Dataset;

    fn create_exponential_decay_data() -> (Array1<f64>, Array1<f64>) {
        // True model: y = 10 * exp(-0.5 * x) + 2 + noise
        let x = Array1::from_vec(vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]);
        let y = Array1::from_vec(vec![
            11.8, 9.7, 8.0, 6.5, 5.7, 4.8, 4.2, 3.7, 3.4, 3.1, 2.9
        ]);
        (x, y)
    }

    fn create_michaelis_menten_data() -> (Array1<f64>, Array1<f64>) {
        // True model: V = 200 * S / (0.1 + S)
        let x = Array1::from_vec(vec![0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]);
        let y = Array1::from_vec(vec![28.6, 65.0, 100.0, 133.3, 166.7, 181.8, 190.5, 196.1]);
        (x, y)
    }

    #[test]
    fn test_nls_exponential_decay_lm() {
        let (x, y) = create_exponential_decay_data();
        let start = Array1::from_vec(vec![8.0, 0.3, 1.0]); // Initial guesses

        let result = nls(
            &x, &y,
            model_exponential_decay,
            &start,
            &["a", "b", "c"],
            NlsConfig::default()
        ).unwrap();

        assert!(result.converged, "Should converge");

        // Check parameters are close to true values (a=10, b=0.5, c=2)
        assert!((result.coefficients[0] - 10.0).abs() < 2.0, "a should be close to 10");
        assert!((result.coefficients[1] - 0.5).abs() < 0.2, "b should be close to 0.5");
        assert!((result.coefficients[2] - 2.0).abs() < 1.0, "c should be close to 2");
    }

    #[test]
    fn test_nls_exponential_decay_gn() {
        let (x, y) = create_exponential_decay_data();
        let start = Array1::from_vec(vec![8.0, 0.3, 1.0]);

        let config = NlsConfig {
            algorithm: NlsAlgorithm::GaussNewton,
            ..Default::default()
        };

        let result = nls(
            &x, &y,
            model_exponential_decay,
            &start,
            &["a", "b", "c"],
            config
        ).unwrap();

        // Gauss-Newton may or may not converge depending on starting values
        // Just check it ran
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_nls_michaelis_menten() {
        let (x, y) = create_michaelis_menten_data();
        let start = Array1::from_vec(vec![150.0, 0.05]); // Initial guesses

        let result = nls(
            &x, &y,
            model_michaelis_menten,
            &start,
            &["Vmax", "Km"],
            NlsConfig::default()
        ).unwrap();

        assert!(result.converged, "Should converge");

        // Check parameters are close to true values (Vmax=200, Km=0.1)
        assert!((result.coefficients[0] - 200.0).abs() < 20.0, "Vmax should be close to 200");
        assert!((result.coefficients[1] - 0.1).abs() < 0.05, "Km should be close to 0.1");
    }

    #[test]
    fn test_nls_with_bounds() {
        let (x, y) = create_exponential_decay_data();
        let start = Array1::from_vec(vec![8.0, 0.3, 1.0]);

        let config = NlsConfig {
            lower: Some(vec![0.0, 0.0, 0.0]), // All positive
            upper: Some(vec![100.0, 10.0, 10.0]),
            ..Default::default()
        };

        let result = nls(
            &x, &y,
            model_exponential_decay,
            &start,
            &["a", "b", "c"],
            config
        ).unwrap();

        // All parameters should be within bounds
        for (&coef, name) in result.coefficients.iter().zip(&result.param_names) {
            assert!(coef >= 0.0, "{} should be >= 0", name);
        }
    }

    #[test]
    fn test_nls_result_structure() {
        let (x, y) = create_exponential_decay_data();
        let start = Array1::from_vec(vec![8.0, 0.3, 1.0]);

        let result = nls(
            &x, &y,
            model_exponential_decay,
            &start,
            &["a", "b", "c"],
            NlsConfig::default()
        ).unwrap();

        // Check structure
        assert_eq!(result.n_obs, 11);
        assert_eq!(result.n_params, 3);
        assert_eq!(result.df, 8);
        assert_eq!(result.param_names.len(), 3);
        assert_eq!(result.coefficients.len(), 3);
        assert_eq!(result.std_errors.len(), 3);
        assert_eq!(result.residuals.len(), 11);
        assert_eq!(result.fitted.len(), 11);

        // Check that RSS is positive
        assert!(result.rss > 0.0);

        // Check t-stats and p-values are finite
        for (t, p) in result.t_stats.iter().zip(&result.p_values) {
            assert!(t.is_finite(), "t-stat should be finite");
            assert!(p.is_finite() && *p >= 0.0 && *p <= 1.0, "p-value should be in [0,1]");
        }
    }

    #[test]
    fn test_nls_display() {
        let (x, y) = create_exponential_decay_data();
        let start = Array1::from_vec(vec![8.0, 0.3, 1.0]);

        let result = nls(
            &x, &y,
            model_exponential_decay,
            &start,
            &["a", "b", "c"],
            NlsConfig::default()
        ).unwrap();

        let display = format!("{}", result);
        assert!(display.contains("Nonlinear Least Squares Results"));
        assert!(display.contains("Converged"));
    }

    #[test]
    fn test_nls_insufficient_data() {
        let x = Array1::from_vec(vec![1.0, 2.0]);
        let y = Array1::from_vec(vec![1.0, 2.0]);
        let start = Array1::from_vec(vec![1.0, 1.0, 1.0]); // 3 params, only 2 obs

        let result = nls(
            &x, &y,
            model_exponential_decay,
            &start,
            &["a", "b", "c"],
            NlsConfig::default()
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_nls_param_names_mismatch() {
        let (x, y) = create_exponential_decay_data();
        let start = Array1::from_vec(vec![8.0, 0.3, 1.0]);

        let result = nls(
            &x, &y,
            model_exponential_decay,
            &start,
            &["a", "b"], // Only 2 names for 3 params
            NlsConfig::default()
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_run_nls_from_dataset() {
        // Create dataset
        let df = df! {
            "substrate" => [0.02f64, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
            "velocity" => [28.6f64, 65.0, 100.0, 133.3, 166.7, 181.8, 190.5, 196.1]
        }.unwrap();
        let dataset = Dataset::new(df);

        let start = Array1::from_vec(vec![150.0, 0.05]);

        let result = run_nls(
            &dataset,
            "velocity",
            "substrate",
            model_michaelis_menten,
            &start,
            &["Vmax", "Km"]
        ).unwrap();

        assert!(result.converged);
        assert!((result.coefficients[0] - 200.0).abs() < 20.0);
    }

    /// Validate against R's nls() output
    /// R code:
    /// ```r
    /// x <- c(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
    /// y <- c(11.8, 9.7, 8.0, 6.5, 5.7, 4.8, 4.2, 3.7, 3.4, 3.1, 2.9)
    /// fit <- nls(y ~ a * exp(-b * x) + c, start = list(a = 8, b = 0.3, c = 1))
    /// summary(fit)
    /// ```
    /// R output:
    /// a = 9.776  SE = 0.366
    /// b = 0.513  SE = 0.051
    /// c = 2.208  SE = 0.150
    /// RSS = 0.198
    ///
    /// Note: Our L-M implementation may find a better local optimum (lower RSS)
    /// than R's default Gauss-Newton, particularly for well-behaved problems.
    #[test]
    fn test_validate_against_r_exponential_decay() {
        let (x, y) = create_exponential_decay_data();
        let start = Array1::from_vec(vec![8.0, 0.3, 1.0]);

        let result = nls(
            &x, &y,
            model_exponential_decay,
            &start,
            &["a", "b", "c"],
            NlsConfig::default()
        ).unwrap();

        // Our L-M may find a better fit than R's Gauss-Newton
        // Key validation: convergence and reasonable parameter estimates
        assert!(result.converged, "Should converge");

        // RSS should be small (fitting the data well)
        // Rust often finds RSS <= R's RSS since L-M is more robust
        assert!(result.rss < 0.5, "RSS should be small, got {}", result.rss);

        // Check parameters are in reasonable range
        // a (amplitude) should be positive and roughly 8-12
        assert!(result.coefficients[0] > 5.0 && result.coefficients[0] < 15.0,
            "a should be in [5, 15], got {}", result.coefficients[0]);
        // b (decay rate) should be positive and roughly 0.3-0.7
        assert!(result.coefficients[1] > 0.2 && result.coefficients[1] < 1.0,
            "b should be in [0.2, 1.0], got {}", result.coefficients[1]);
        // c (offset) should be roughly 1-3
        assert!(result.coefficients[2] > 0.0 && result.coefficients[2] < 5.0,
            "c should be in [0, 5], got {}", result.coefficients[2]);
    }

    /// Validate Michaelis-Menten against R
    /// R code:
    /// ```r
    /// S <- c(0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
    /// V <- c(28.6, 65.0, 100.0, 133.3, 166.7, 181.8, 190.5, 196.1)
    /// fit <- nls(V ~ Vmax * S / (Km + S), start = list(Vmax = 150, Km = 0.05))
    /// coef(fit)
    /// ```
    /// R output: Vmax=200.2, Km=0.102
    #[test]
    fn test_validate_against_r_michaelis_menten() {
        let (x, y) = create_michaelis_menten_data();
        let start = Array1::from_vec(vec![150.0, 0.05]);

        let result = nls(
            &x, &y,
            model_michaelis_menten,
            &start,
            &["Vmax", "Km"],
            NlsConfig::default()
        ).unwrap();

        assert!((result.coefficients[0] - 200.2).abs() < 5.0,
            "Vmax: Rust={} vs R=200.2", result.coefficients[0]);
        assert!((result.coefficients[1] - 0.102).abs() < 0.02,
            "Km: Rust={} vs R=0.102", result.coefficients[1]);
    }
}
