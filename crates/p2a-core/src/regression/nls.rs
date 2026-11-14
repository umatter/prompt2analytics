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
//! - R implementation: `stats::nls()` - <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/nls.html>

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::safe_inverse;
use crate::linalg::solve::{SingularPolicy, sym_inverse};
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
    /// Observation weights (R's `weights`): the fit minimises `sum w (y - f)^2`. One per
    /// observation, finite and non-negative; a zero weight drops the observation from the
    /// residual degrees of freedom, as in R.
    pub weights: Option<Vec<f64>>,
    /// Lower bounds for the parameters, one per parameter (`-inf` for none); usable without
    /// `upper` (R's `algorithm = "port"`).
    pub lower: Option<Vec<f64>>,
    /// Upper bounds for the parameters, one per parameter (`+inf` for none); usable without
    /// `lower`.
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
        writeln!(
            f,
            "Residual Std. Error: {:.6} on {} degrees of freedom",
            self.sigma, self.df
        )?;
        writeln!(f, "Residual Sum of Squares: {:.6}", self.rss)?;
        writeln!(
            f,
            "Converged: {} (iterations: {})",
            self.converged, self.iterations
        )?;
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

    let setup = check_nls_inputs(n, k, param_names, &config)?;
    let weights = setup.weights.clone();
    let bounds = &setup.bounds;

    // Initialize parameters inside the bounds
    let mut theta = apply_bounds(start, bounds);

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

        // Compute J'J and J'r, on sqrt(w) r with weights
        let (jtj, jtr) = normal_terms(jacobian, &residuals, &setup, &theta)?;

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
                            let theta_bounded = apply_bounds(&theta_new, bounds);
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
            theta = apply_bounds(&theta_new, bounds);

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
    let df = setup.n_used - k;
    let sigma = if df > 0 {
        (rss / df as f64).sqrt()
    } else {
        f64::NAN
    };

    // Compute variance-covariance matrix
    let jacobian_final = compute_jacobian(x, y, &theta, model, n, k, config.diff_step);
    let jtj_final = final_cross_product(jacobian_final, &setup);

    // A rank-deficient J'J at the solution (R's nls stops with "singular gradient") has no
    // covariance: NaN standard errors rather than a generalized inverse.
    let (vcov, std_errors) = match sym_inverse(
        jtj_final.view(),
        SingularPolicy::Error,
        "J'J at the solution",
    ) {
        Ok(s) => {
            let vcov = &s.inverse * (sigma * sigma);
            let se: Vec<f64> = vcov.diag().iter().map(|v| v.max(0.0).sqrt()).collect();
            (Some(vcov), se)
        }
        Err(_) => {
            // If inversion fails, report NaN standard errors
            (None, vec![f64::NAN; k])
        }
    };

    // Compute t-statistics and p-values
    let t_stats: Vec<f64> = theta
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| {
            if se.is_finite() && se > 0.0 {
                b / se
            } else {
                f64::NAN
            }
        })
        .collect();

    let p_values: Vec<f64> = t_stats
        .iter()
        .map(|&t| {
            if t.is_finite() {
                t_test_p_value(t, df as f64)
            } else {
                f64::NAN
            }
        })
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
        final_lambda: if config.algorithm == NlsAlgorithm::LevenbergMarquardt {
            Some(lambda)
        } else {
            None
        },
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

    let setup = check_nls_inputs(n, k, param_names, &config)?;
    let weights = setup.weights.clone();
    let bounds = &setup.bounds;

    let mut theta = apply_bounds(start, bounds);

    let mut residuals = compute_residuals_multi(x, y, &theta, model);
    let mut rss = compute_rss(&residuals, &weights);

    let mut iterations = 0;
    let mut converged = false;
    let mut convergence_code: u8 = 1;
    let mut lambda = config.lambda_init;

    for _iter in 0..config.max_iter {
        iterations += 1;

        let jacobian = compute_jacobian_multi(x, y, &theta, model, config.diff_step);
        let (jtj, jtr) = normal_terms(jacobian, &residuals, &setup, &theta)?;

        match config.algorithm {
            NlsAlgorithm::GaussNewton => match solve_normal_equations(&jtj, &jtr, 0.0) {
                Ok(delta) => {
                    let theta_new = &theta - &delta;
                    theta = apply_bounds(&theta_new, bounds);

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
            },
            NlsAlgorithm::LevenbergMarquardt => {
                let diag_jtj = jtj.diag().to_owned();

                let mut step_accepted = false;
                for _ in 0..50 {
                    // Max attempts to find good lambda
                    match solve_lm_equations(&jtj, &jtr, lambda, &diag_jtj) {
                        Ok(d) => {
                            let theta_new = &theta - &d;
                            let theta_bounded = apply_bounds(&theta_new, bounds);
                            let residuals_new =
                                compute_residuals_multi(x, y, &theta_bounded, model);
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

    let df = setup.n_used - k;
    let sigma = if df > 0 {
        (rss / df as f64).sqrt()
    } else {
        f64::NAN
    };

    let jacobian_final = compute_jacobian_multi(x, y, &theta, model, config.diff_step);
    let jtj_final = final_cross_product(jacobian_final, &setup);

    // A rank-deficient J'J at the solution (R's nls stops with "singular gradient") has no
    // covariance: NaN standard errors rather than a generalized inverse.
    let (vcov, std_errors) = match sym_inverse(
        jtj_final.view(),
        SingularPolicy::Error,
        "J'J at the solution",
    ) {
        Ok(s) => {
            let vcov = &s.inverse * (sigma * sigma);
            let se: Vec<f64> = vcov.diag().iter().map(|v| v.max(0.0).sqrt()).collect();
            (Some(vcov), se)
        }
        Err(_) => (None, vec![f64::NAN; k]),
    };

    let t_stats: Vec<f64> = theta
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| {
            if se.is_finite() && se > 0.0 {
                b / se
            } else {
                f64::NAN
            }
        })
        .collect();

    let p_values: Vec<f64> = t_stats
        .iter()
        .map(|&t| {
            if t.is_finite() {
                t_test_p_value(t, df as f64)
            } else {
                f64::NAN
            }
        })
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
        final_lambda: if config.algorithm == NlsAlgorithm::LevenbergMarquardt {
            Some(lambda)
        } else {
            None
        },
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
    run_nls_with_config(
        dataset,
        y_col,
        x_col,
        model,
        start,
        param_names,
        NlsConfig::default(),
    )
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

    let y = DesignMatrix::extract_column(dataset.df(), y_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: y_col.to_string(),
            available: vec![format!("{:?}", e)],
        }
    })?;

    let x = DesignMatrix::extract_column(dataset.df(), x_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: x_col.to_string(),
            available: vec![format!("{:?}", e)],
        }
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
        Some(w) => residuals
            .iter()
            .zip(w.iter())
            .map(|(&r, &wi)| wi * r * r)
            .sum(),
        None => residuals.iter().map(|r| r * r).sum(),
    }
}

/// Compute Jacobian matrix using central differences
fn compute_jacobian(
    x: &Array1<f64>,
    _y: &Array1<f64>,
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

    // inverse-ok: a Levenberg-Marquardt trial step; the residual-sum check accepts or rejects it
    let (inv, _) = safe_inverse(&augmented.view()).map_err(|e| EconError::SingularMatrix {
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

    // inverse-ok: a Levenberg-Marquardt trial step; the residual-sum check accepts or rejects it
    let (inv, _) = safe_inverse(&augmented.view()).map_err(|e| EconError::SingularMatrix {
        context: "L-M equations in NLS".to_string(),
        suggestion: format!("Lambda may be too small: {:?}", e),
    })?;

    Ok(inv.dot(jtr))
}

/// The checked weights and bounds of an NLS fit.
struct NlsSetup {
    /// The weights, if any
    weights: Option<Array1<f64>>,
    /// Their square roots, which scale the residuals and Jacobian rows
    sqrt_w: Option<Array1<f64>>,
    /// Observations with a non-zero weight (all of them without weights)
    n_used: usize,
    /// (lower, upper) per parameter, infinite where a side is unbounded; `None` without bounds
    bounds: Option<(Vec<f64>, Vec<f64>)>,
}

/// Checks the inputs of `nls` and `nls_multi`: sizes, parameter names, weights (one per
/// observation, finite, non-negative, enough of them non-zero) and bounds (one per parameter,
/// not NaN, lower <= upper; either side may be given alone).
fn check_nls_inputs(
    n: usize,
    k: usize,
    param_names: &[&str],
    config: &NlsConfig,
) -> EconResult<NlsSetup> {
    let invalid = |message: String| EconError::InvalidSpecification { message };
    if n < k {
        return Err(invalid(format!(
            "More parameters ({}) than observations ({})",
            k, n
        )));
    }
    if param_names.len() != k {
        return Err(invalid(format!(
            "Parameter names length ({}) doesn't match start values length ({})",
            param_names.len(),
            k
        )));
    }
    let (weights, sqrt_w, n_used) = match &config.weights {
        None => (None, None, n),
        Some(w) => {
            if w.len() != n {
                return Err(invalid(format!(
                    "Weights length ({}) doesn't match data length ({})",
                    w.len(),
                    n
                )));
            }
            if let Some(i) = w.iter().position(|v| !(v.is_finite() && *v >= 0.0)) {
                return Err(invalid(format!(
                    "weights must be finite and non-negative; weight {} is {}",
                    i + 1,
                    w[i]
                )));
            }
            let n_used = w.iter().filter(|v| **v > 0.0).count();
            if n_used < k {
                return Err(invalid(format!(
                    "{n_used} observations have a positive weight, fewer than the {k} parameters"
                )));
            }
            let w = Array1::from_vec(w.clone());
            let sw = w.mapv(f64::sqrt);
            (Some(w), Some(sw), n_used)
        }
    };
    let side = |b: &Option<Vec<f64>>, name: &str, none: f64| -> EconResult<Vec<f64>> {
        match b {
            None => Ok(vec![none; k]),
            Some(v) if v.len() != k => Err(invalid(format!(
                "{name} bounds have {} values; the model has {k} parameters",
                v.len()
            ))),
            Some(v) if v.iter().any(|b| b.is_nan()) => {
                Err(invalid(format!("{name} bounds must not be NaN")))
            }
            Some(v) => Ok(v.clone()),
        }
    };
    let bounds = if config.lower.is_none() && config.upper.is_none() {
        None
    } else {
        let lo = side(&config.lower, "Lower", f64::NEG_INFINITY)?;
        let hi = side(&config.upper, "Upper", f64::INFINITY)?;
        for i in 0..k {
            if lo[i] > hi[i] {
                return Err(invalid(format!(
                    "Lower bound {} exceeds upper bound {} for parameter {}",
                    lo[i], hi[i], param_names[i]
                )));
            }
        }
        Some((lo, hi))
    };
    Ok(NlsSetup {
        weights,
        sqrt_w,
        n_used,
        bounds,
    })
}

/// `J'J` and `J'r` of one iteration, with `J` and `r` scaled by `sqrt(w)` under weights. A
/// parameter at a bound whose descent direction points outside is held there (its row and
/// column of `J'J` become the unit vector, its `J'r` zero), so the step moves the others.
fn normal_terms(
    mut jacobian: Array2<f64>,
    residuals: &Array1<f64>,
    setup: &NlsSetup,
    theta: &Array1<f64>,
) -> EconResult<(Array2<f64>, Array1<f64>)> {
    let (mut jtj, mut jtr) = match &setup.sqrt_w {
        None => (jacobian.t().dot(&jacobian), jacobian.t().dot(residuals)),
        Some(sw) => {
            for (mut row, s) in jacobian.rows_mut().into_iter().zip(sw.iter()) {
                row *= *s;
            }
            let r = residuals * sw;
            (jacobian.t().dot(&jacobian), jacobian.t().dot(&r))
        }
    };
    if let Some((lo, hi)) = &setup.bounds {
        let k = theta.len();
        for j in 0..k {
            // the step is theta - delta with delta ~ (J'J)^-1 J'r: J'r > 0 lowers theta_j
            let held = (theta[j] <= lo[j] && jtr[j] > 0.0) || (theta[j] >= hi[j] && jtr[j] < 0.0);
            if held {
                for i in 0..k {
                    jtj[[i, j]] = 0.0;
                    jtj[[j, i]] = 0.0;
                }
                jtj[[j, j]] = 1.0;
                jtr[j] = 0.0;
            }
        }
    }
    Ok((jtj, jtr))
}

/// `J'J` at the solution, on `sqrt(w) J` under weights (R's gradient of the weighted model).
fn final_cross_product(mut jacobian: Array2<f64>, setup: &NlsSetup) -> Array2<f64> {
    if let Some(sw) = &setup.sqrt_w {
        for (mut row, s) in jacobian.rows_mut().into_iter().zip(sw.iter()) {
            row *= *s;
        }
    }
    jacobian.t().dot(&jacobian)
}

/// Clamp the parameters into their bounds (unchanged without bounds).
fn apply_bounds(theta: &Array1<f64>, bounds: &Option<(Vec<f64>, Vec<f64>)>) -> Array1<f64> {
    match bounds {
        Some((lo, hi)) => theta
            .iter()
            .zip(lo.iter())
            .zip(hi.iter())
            .map(|((&t, &l), &u)| t.max(l).min(u))
            .collect(),
        None => theta.clone(),
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
    use crate::data::Dataset;
    use polars::prelude::*;

    fn create_exponential_decay_data() -> (Array1<f64>, Array1<f64>) {
        // True model: y = 10 * exp(-0.5 * x) + 2 + noise
        let x = Array1::from_vec(vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]);
        let y = Array1::from_vec(vec![11.8, 9.7, 8.0, 6.5, 5.7, 4.8, 4.2, 3.7, 3.4, 3.1, 2.9]);
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
            &x,
            &y,
            model_exponential_decay,
            &start,
            &["a", "b", "c"],
            NlsConfig::default(),
        )
        .unwrap();

        assert!(result.converged, "Should converge");

        // Check parameters are close to true values (a=10, b=0.5, c=2)
        assert!(
            (result.coefficients[0] - 10.0).abs() < 2.0,
            "a should be close to 10"
        );
        assert!(
            (result.coefficients[1] - 0.5).abs() < 0.2,
            "b should be close to 0.5"
        );
        assert!(
            (result.coefficients[2] - 2.0).abs() < 1.0,
            "c should be close to 2"
        );
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
            &x,
            &y,
            model_exponential_decay,
            &start,
            &["a", "b", "c"],
            config,
        )
        .unwrap();

        // Gauss-Newton may or may not converge depending on starting values
        // Just check it ran
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_nls_michaelis_menten() {
        let (x, y) = create_michaelis_menten_data();
        let start = Array1::from_vec(vec![150.0, 0.05]); // Initial guesses

        let result = nls(
            &x,
            &y,
            model_michaelis_menten,
            &start,
            &["Vmax", "Km"],
            NlsConfig::default(),
        )
        .unwrap();

        assert!(result.converged, "Should converge");

        // Check parameters are close to true values (Vmax=200, Km=0.1)
        assert!(
            (result.coefficients[0] - 200.0).abs() < 20.0,
            "Vmax should be close to 200"
        );
        assert!(
            (result.coefficients[1] - 0.1).abs() < 0.05,
            "Km should be close to 0.1"
        );
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
            &x,
            &y,
            model_exponential_decay,
            &start,
            &["a", "b", "c"],
            config,
        )
        .unwrap();

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
            &x,
            &y,
            model_exponential_decay,
            &start,
            &["a", "b", "c"],
            NlsConfig::default(),
        )
        .unwrap();

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
            assert!(
                p.is_finite() && *p >= 0.0 && *p <= 1.0,
                "p-value should be in [0,1]"
            );
        }
    }

    #[test]
    fn test_nls_display() {
        let (x, y) = create_exponential_decay_data();
        let start = Array1::from_vec(vec![8.0, 0.3, 1.0]);

        let result = nls(
            &x,
            &y,
            model_exponential_decay,
            &start,
            &["a", "b", "c"],
            NlsConfig::default(),
        )
        .unwrap();

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
            &x,
            &y,
            model_exponential_decay,
            &start,
            &["a", "b", "c"],
            NlsConfig::default(),
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_nls_param_names_mismatch() {
        let (x, y) = create_exponential_decay_data();
        let start = Array1::from_vec(vec![8.0, 0.3, 1.0]);

        let result = nls(
            &x,
            &y,
            model_exponential_decay,
            &start,
            &["a", "b"], // Only 2 names for 3 params
            NlsConfig::default(),
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_run_nls_from_dataset() {
        // Create dataset
        let df = df! {
            "substrate" => [0.02f64, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
            "velocity" => [28.6f64, 65.0, 100.0, 133.3, 166.7, 181.8, 190.5, 196.1]
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let start = Array1::from_vec(vec![150.0, 0.05]);

        let result = run_nls(
            &dataset,
            "velocity",
            "substrate",
            model_michaelis_menten,
            &start,
            &["Vmax", "Km"],
        )
        .unwrap();

        assert!(result.converged);
        assert!((result.coefficients[0] - 200.0).abs() < 20.0);
    }

    fn nls_case_columns(ds: &Dataset, x: &str, y: &str) -> (Array1<f64>, Array1<f64>) {
        use crate::data::columns::{NaPolicy, numeric_column};
        let num = |c: &str| Array1::from(numeric_column(ds.df(), c, NaPolicy::Fail).unwrap().0);
        (num(x), num(y))
    }

    /// Case `nls_expdecay`: stats::nls(y ~ a * exp(-b * x) + c), start (8, 0.3, 1). Iterative: R
    /// stops at relative offset 1e-7, p2a's Levenberg-Marquardt at a 1e-8 relative change in RSS;
    /// p2a's SEs come from a forward-difference Jacobian (step 1e-7).
    #[test]
    fn test_validate_nls_expdecay() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("nls_expdecay");
        let (x, y) = nls_case_columns(&ds, "x", "y");
        let r = nls(
            &x,
            &y,
            model_exponential_decay,
            &ndarray::array![8.0, 0.3, 1.0],
            &["a", "b", "c"],
            NlsConfig::default(),
        )
        .unwrap();
        assert!(r.converged);
        Check::new(&case)
            .named(
                "coef",
                &r.param_names,
                &r.coefficients,
                Tol::iterative(1e-6),
            )
            .named("se", &r.param_names, &r.std_errors, Tol::iterative(1e-5))
            .named("t", &r.param_names, &r.t_stats, Tol::iterative(1e-5))
            .named("p", &r.param_names, &r.p_values, Tol::iterative(1e-3))
            .scalar("rss", r.rss, Tol::iterative(1e-8))
            .scalar("sigma", r.sigma, Tol::iterative(1e-8))
            .scalar("df", r.df as f64, Tol::exact())
            .finish();
    }

    /// Case `nls_mm`: stats::nls(V ~ Vmax * S / (Km + S)), start (150, 0.05); iterative as
    /// `test_validate_nls_expdecay`.
    #[test]
    fn test_validate_nls_michaelis_menten() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("nls_mm");
        let (x, y) = nls_case_columns(&ds, "S", "V");
        let r = nls(
            &x,
            &y,
            model_michaelis_menten,
            &ndarray::array![150.0, 0.05],
            &["Vmax", "Km"],
            NlsConfig::default(),
        )
        .unwrap();
        assert!(r.converged);
        Check::new(&case)
            .named(
                "coef",
                &r.param_names,
                &r.coefficients,
                Tol::iterative(1e-6),
            )
            .named("se", &r.param_names, &r.std_errors, Tol::iterative(1e-5))
            .named("t", &r.param_names, &r.t_stats, Tol::iterative(1e-5))
            .named("p", &r.param_names, &r.p_values, Tol::iterative(1e-3))
            .scalar("rss", r.rss, Tol::iterative(1e-8))
            .scalar("sigma", r.sigma, Tol::iterative(1e-8))
            .scalar("df", r.df as f64, Tol::exact())
            .finish();
    }

    fn nls_t4(case_id: &str) -> (p2a_validation::Case, Array1<f64>, Array1<f64>, Vec<f64>) {
        use crate::data::columns::{NaPolicy, numeric_column};
        let (case, ds) = crate::test_support::load_case(case_id);
        let c = |n: &str| numeric_column(ds.df(), n, NaPolicy::Fail).unwrap().0;
        (case, Array1::from(c("x")), Array1::from(c("y")), c("w"))
    }

    /// Case `nls_weighted_t4` (REG-48): stats::nls(y ~ a exp(-b x) + c, weights = w), variance
    /// rising with x. Iterative as `test_validate_nls_expdecay`, with the RSS change tolerance
    /// at 1e-12.
    #[test]
    fn test_validate_nls_weighted() {
        use p2a_validation::{Check, Tol};
        let (case, x, y, w) = nls_t4("nls_weighted_t4");
        let mut config = NlsConfig::default();
        config.weights = Some(w);
        config.tolerance = 1e-12;
        let r = nls(
            &x,
            &y,
            model_exponential_decay,
            &ndarray::array![8.0, 0.3, 1.0],
            &["a", "b", "c"],
            config,
        )
        .unwrap();
        assert!(r.converged);
        Check::new(&case)
            .named(
                "w_coef",
                &r.param_names,
                &r.coefficients,
                Tol::iterative(1e-6),
            )
            .named("w_se", &r.param_names, &r.std_errors, Tol::iterative(1e-5))
            .named("w_t", &r.param_names, &r.t_stats, Tol::iterative(1e-5))
            .named("w_p", &r.param_names, &r.p_values, Tol::iterative(1e-3))
            .scalar("w_rss", r.rss, Tol::iterative(1e-8))
            .scalar("w_sigma", r.sigma, Tol::iterative(1e-8))
            .scalar("w_df", r.df as f64, Tol::exact())
            .vector("w_fitted", &r.fitted, Tol::iterative(1e-6))
            .vector("w_residuals", &r.residuals, Tol::iterative(1e-5))
            .finish();
    }

    /// Case `nls_weighted_t4` (REG-48): a lower bound alone on c (weighted, as the R case) and
    /// an upper bound alone on b (unweighted), as nls(algorithm = "port"); both optima lie on
    /// the bound.
    #[test]
    fn test_validate_nls_one_sided_bounds() {
        use p2a_validation::{Check, Tol};
        let (case, x, y, w) = nls_t4("nls_weighted_t4");
        let names = ["a", "b", "c"];
        let mut lo = NlsConfig::default();
        lo.weights = Some(w);
        lo.tolerance = 1e-12;
        lo.lower = Some(vec![f64::NEG_INFINITY, f64::NEG_INFINITY, 2.5]);
        let rl = nls(
            &x,
            &y,
            model_exponential_decay,
            &ndarray::array![8.0, 0.3, 2.5],
            &names,
            lo,
        )
        .unwrap();
        let mut up = NlsConfig::default();
        up.tolerance = 1e-12;
        up.upper = Some(vec![f64::INFINITY, 0.45, f64::INFINITY]);
        let ru = nls(
            &x,
            &y,
            model_exponential_decay,
            &ndarray::array![8.0, 0.3, 1.0],
            &names,
            up,
        )
        .unwrap();
        Check::new(&case)
            .named(
                "lower_coef",
                &rl.param_names,
                &rl.coefficients,
                Tol::iterative(1e-6),
            )
            .scalar("lower_rss", rl.rss, Tol::iterative(1e-8))
            .named(
                "upper_coef",
                &ru.param_names,
                &ru.coefficients,
                Tol::iterative(1e-6),
            )
            .scalar("upper_rss", ru.rss, Tol::iterative(1e-8))
            .finish();
    }

    /// nls_multi checks its inputs as nls does: a short weights or bounds vector is an error,
    /// not a panic; negative or non-finite weights are refused.
    #[test]
    fn test_nls_multi_rejects_short_weights_and_bounds() {
        fn lin(x: &Array1<f64>, t: &Array1<f64>) -> f64 {
            t[0] + t[1] * x[0]
        }
        let x = Array2::from_shape_fn((6, 1), |(i, _)| i as f64);
        let y = ndarray::array![0.1, 1.2, 1.9, 3.1, 4.0, 5.2];
        let start = ndarray::array![0.0, 1.0];
        let run = |c: NlsConfig| nls_multi(&x, &y, lin, &start, &["a", "b"], c);
        for c in [
            NlsConfig {
                weights: Some(vec![1.0; 3]),
                ..Default::default()
            },
            NlsConfig {
                lower: Some(vec![0.0]),
                ..Default::default()
            },
            NlsConfig {
                upper: Some(vec![9.0]),
                ..Default::default()
            },
            NlsConfig {
                weights: Some(vec![1.0, -1.0, 1.0, 1.0, 1.0, 1.0]),
                ..Default::default()
            },
            NlsConfig {
                weights: Some(vec![1.0, f64::NAN, 1.0, 1.0, 1.0, 1.0]),
                ..Default::default()
            },
        ] {
            assert!(matches!(
                run(c),
                Err(EconError::InvalidSpecification { .. })
            ));
        }
    }
}
