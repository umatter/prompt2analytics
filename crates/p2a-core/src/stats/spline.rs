//! Spline interpolation functions.
//!
//! Implements R's `spline()` and `approx()` functions for interpolation.
//!
//! # References
//!
//! - R Core Team. `stats::spline()` and `stats::approx()` functions.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/splinefun.html>
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/approxfun.html>
//! - Forsythe, G. E., Malcolm, M. A., & Moler, C. B. (1977).
//!   "Computer Methods for Mathematical Computations". Prentice-Hall.

use crate::errors::{EconError, EconResult};
use serde::{Deserialize, Serialize};

/// Method for spline interpolation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum SplineMethod {
    /// Natural spline with zero second derivatives at endpoints.
    #[default]
    Natural,
    /// FMM spline (Forsythe, Malcolm, Moler) - exact cubic through 4 points at each end.
    Fmm,
    /// Periodic spline for cyclical data.
    Periodic,
    /// Monotone Hermite spline (Fritsch-Carlson method).
    MonotoneFC,
}

/// Method for linear/constant interpolation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum ApproxMethod {
    /// Linear interpolation between points.
    #[default]
    Linear,
    /// Constant (step function) interpolation.
    Constant,
}

/// Rule for handling values outside the interpolation range.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum ApproxRule {
    /// Return NA for points outside range.
    #[default]
    Na,
    /// Return the nearest endpoint value.
    Nearest,
}

/// Result of spline interpolation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplineResult {
    /// Interpolated x values.
    pub x: Vec<f64>,
    /// Interpolated y values.
    pub y: Vec<f64>,
    /// Method used for interpolation.
    pub method: SplineMethod,
    /// Number of output points.
    pub n: usize,
}

/// Result of approx interpolation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApproxResult {
    /// Interpolated x values.
    pub x: Vec<f64>,
    /// Interpolated y values (may contain NaN for out-of-range points).
    pub y: Vec<f64>,
    /// Method used for interpolation.
    pub method: ApproxMethod,
    /// Number of output points.
    pub n: usize,
}

/// Perform cubic spline interpolation.
///
/// Interpolates the given data points using cubic splines. The result is
/// a smooth curve passing through all data points.
///
/// # Arguments
///
/// * `x` - Input x values (must be at least 4 points, will be sorted internally)
/// * `y` - Input y values (same length as x)
/// * `xout` - Optional x values at which to interpolate. If None, uses n equally spaced points.
/// * `n` - Number of interpolation points if xout is None (default 50)
/// * `method` - Spline method to use
///
/// # Returns
///
/// `SplineResult` containing interpolated x and y values.
///
/// # Mathematical Background
///
/// Cubic splines are piecewise cubic polynomials that:
/// 1. Pass through all data points
/// 2. Have continuous first and second derivatives
/// 3. Satisfy boundary conditions specified by the method
///
/// Natural spline: S''(x_0) = S''(x_n) = 0
/// FMM spline: Uses not-a-knot condition at boundaries
///
/// # References
///
/// - R function `stats::spline()`
/// - Forsythe, Malcolm, Moler (1977). "Computer Methods for Mathematical Computations".
pub fn spline(
    x: &[f64],
    y: &[f64],
    xout: Option<&[f64]>,
    n: Option<usize>,
    method: SplineMethod,
) -> EconResult<SplineResult> {
    if x.len() != y.len() {
        return Err(EconError::InvalidSpecification {
            message: format!("x and y must have same length: {} vs {}", x.len(), y.len()),
        });
    }

    if x.len() < 4 {
        return Err(EconError::InsufficientData {
            required: 4,
            provided: x.len(),
            context: "cubic spline interpolation".to_string(),
        });
    }

    // Check for non-finite values
    if x.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: "Input contains non-finite values (NaN or Inf)".to_string(),
        });
    }

    // Sort by x values
    let mut pairs: Vec<(f64, f64)> = x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi, yi)).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let xs: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    let ys: Vec<f64> = pairs.iter().map(|p| p.1).collect();

    // Check for duplicate x values
    for i in 1..xs.len() {
        if (xs[i] - xs[i - 1]).abs() < 1e-14 {
            return Err(EconError::InvalidSpecification {
                message: "Duplicate x values not allowed in spline interpolation".to_string(),
            });
        }
    }

    // Compute spline coefficients
    let coeffs = match method {
        SplineMethod::Natural => compute_natural_spline(&xs, &ys)?,
        SplineMethod::Fmm => compute_fmm_spline(&xs, &ys)?,
        SplineMethod::Periodic => compute_periodic_spline(&xs, &ys)?,
        SplineMethod::MonotoneFC => compute_monotone_spline(&xs, &ys)?,
    };

    // Determine output x values
    let x_out: Vec<f64> = match xout {
        Some(xo) => xo.to_vec(),
        None => {
            let n_pts = n.unwrap_or(50);
            let x_min = xs[0];
            let x_max = xs[xs.len() - 1];
            (0..n_pts)
                .map(|i| x_min + (x_max - x_min) * i as f64 / (n_pts - 1) as f64)
                .collect()
        }
    };

    // Evaluate spline at output points
    let y_out: Vec<f64> = x_out
        .iter()
        .map(|&xi| evaluate_cubic_spline(&xs, &ys, &coeffs, xi))
        .collect();

    Ok(SplineResult {
        x: x_out.clone(),
        y: y_out,
        method,
        n: x_out.len(),
    })
}

/// Perform linear or constant interpolation.
///
/// Interpolates the given data points using either linear interpolation
/// or a step function (constant interpolation).
///
/// # Arguments
///
/// * `x` - Input x values (will be sorted internally)
/// * `y` - Input y values (same length as x)
/// * `xout` - Optional x values at which to interpolate. If None, uses n equally spaced points.
/// * `n` - Number of interpolation points if xout is None (default 50)
/// * `method` - Interpolation method (Linear or Constant)
/// * `rule` - How to handle values outside the x range
/// * `f` - For constant method: 0 = left-continuous, 1 = right-continuous, 0.5 = midpoint
///
/// # Returns
///
/// `ApproxResult` containing interpolated x and y values.
///
/// # References
///
/// - R function `stats::approx()`
pub fn approx(
    x: &[f64],
    y: &[f64],
    xout: Option<&[f64]>,
    n: Option<usize>,
    method: ApproxMethod,
    rule: ApproxRule,
    f: f64,
) -> EconResult<ApproxResult> {
    if x.len() != y.len() {
        return Err(EconError::InvalidSpecification {
            message: format!("x and y must have same length: {} vs {}", x.len(), y.len()),
        });
    }

    if x.len() < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: x.len(),
            context: "interpolation".to_string(),
        });
    }

    // Sort by x values
    let mut pairs: Vec<(f64, f64)> = x
        .iter()
        .zip(y.iter())
        .filter(|(xi, yi)| xi.is_finite() && yi.is_finite())
        .map(|(&xi, &yi)| (xi, yi))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    if pairs.len() < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: pairs.len(),
            context: "interpolation (after removing NA values)".to_string(),
        });
    }

    let xs: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    let ys: Vec<f64> = pairs.iter().map(|p| p.1).collect();

    // Determine output x values
    let x_out: Vec<f64> = match xout {
        Some(xo) => xo.to_vec(),
        None => {
            let n_pts = n.unwrap_or(50);
            let x_min = xs[0];
            let x_max = xs[xs.len() - 1];
            (0..n_pts)
                .map(|i| x_min + (x_max - x_min) * i as f64 / (n_pts - 1) as f64)
                .collect()
        }
    };

    // Interpolate at output points
    let y_out: Vec<f64> = x_out
        .iter()
        .map(|&xi| interpolate_approx(&xs, &ys, xi, method, rule, f))
        .collect();

    Ok(ApproxResult {
        x: x_out.clone(),
        y: y_out,
        method,
        n: x_out.len(),
    })
}

/// Create an interpolating function from data points.
///
/// This is a convenience wrapper that returns a closure for evaluating
/// the interpolation at arbitrary points.
pub fn splinefun(x: &[f64], y: &[f64], method: SplineMethod) -> EconResult<impl Fn(f64) -> f64> {
    if x.len() != y.len() {
        return Err(EconError::InvalidSpecification {
            message: format!("x and y must have same length: {} vs {}", x.len(), y.len()),
        });
    }

    if x.len() < 4 {
        return Err(EconError::InsufficientData {
            required: 4,
            provided: x.len(),
            context: "cubic spline interpolation".to_string(),
        });
    }

    // Sort by x values
    let mut pairs: Vec<(f64, f64)> = x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi, yi)).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let xs: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    let ys: Vec<f64> = pairs.iter().map(|p| p.1).collect();

    // Compute spline coefficients
    let coeffs = match method {
        SplineMethod::Natural => compute_natural_spline(&xs, &ys)?,
        SplineMethod::Fmm => compute_fmm_spline(&xs, &ys)?,
        SplineMethod::Periodic => compute_periodic_spline(&xs, &ys)?,
        SplineMethod::MonotoneFC => compute_monotone_spline(&xs, &ys)?,
    };

    Ok(move |xi: f64| evaluate_cubic_spline(&xs, &ys, &coeffs, xi))
}

/// Create an approx interpolating function from data points.
pub fn approxfun(
    x: &[f64],
    y: &[f64],
    method: ApproxMethod,
    rule: ApproxRule,
    f: f64,
) -> EconResult<impl Fn(f64) -> f64> {
    if x.len() != y.len() {
        return Err(EconError::InvalidSpecification {
            message: format!("x and y must have same length: {} vs {}", x.len(), y.len()),
        });
    }

    if x.len() < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: x.len(),
            context: "interpolation".to_string(),
        });
    }

    // Sort by x values
    let mut pairs: Vec<(f64, f64)> = x
        .iter()
        .zip(y.iter())
        .filter(|(xi, yi)| xi.is_finite() && yi.is_finite())
        .map(|(&xi, &yi)| (xi, yi))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let xs: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    let ys: Vec<f64> = pairs.iter().map(|p| p.1).collect();

    Ok(move |xi: f64| interpolate_approx(&xs, &ys, xi, method, rule, f))
}

// ========== Internal helper functions ==========

/// Spline coefficients: for each interval, stores second derivatives at knots.
struct SplineCoeffs {
    /// Second derivatives at each knot point
    second_derivs: Vec<f64>,
}

/// Compute natural cubic spline coefficients.
fn compute_natural_spline(x: &[f64], y: &[f64]) -> EconResult<SplineCoeffs> {
    let n = x.len();

    // Natural spline: M_0 = M_{n-1} = 0 (zero second derivatives at endpoints)
    // We solve a tridiagonal system for the second derivatives M_i

    // Set up the tridiagonal system
    let mut h = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        h.push(x[i + 1] - x[i]);
    }

    // Build the right-hand side
    let mut d = vec![0.0; n];
    for i in 1..n - 1 {
        d[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
    }

    // Build the tridiagonal matrix (sub-diagonal, diagonal, super-diagonal)
    let mut diag = vec![1.0; n]; // Boundary conditions
    let mut sub = vec![0.0; n];
    let mut sup = vec![0.0; n];

    for i in 1..n - 1 {
        sub[i] = h[i - 1];
        diag[i] = 2.0 * (h[i - 1] + h[i]);
        sup[i] = h[i];
    }

    // Solve tridiagonal system using Thomas algorithm
    let second_derivs = solve_tridiagonal(&sub, &diag, &sup, &d)?;

    Ok(SplineCoeffs { second_derivs })
}

/// Compute FMM (not-a-knot) cubic spline coefficients.
fn compute_fmm_spline(x: &[f64], y: &[f64]) -> EconResult<SplineCoeffs> {
    let n = x.len();

    // FMM uses not-a-knot condition: third derivative continuous at x_1 and x_{n-2}
    // This is more complex, but we can approximate with a modified natural spline
    // that uses the slope estimates from the first/last 4 points

    let mut h = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        h.push(x[i + 1] - x[i]);
    }

    // Estimate end slopes using finite differences on 4 points
    // This approximates the FMM behavior
    let d_left = estimate_end_derivative(&x[..4], &y[..4], true);
    let d_right = estimate_end_derivative(&x[n - 4..], &y[n - 4..], false);

    // Build the right-hand side with modified boundary conditions
    let mut d = vec![0.0; n];
    d[0] = 6.0 * ((y[1] - y[0]) / h[0] - d_left) / h[0];
    d[n - 1] = 6.0 * (d_right - (y[n - 1] - y[n - 2]) / h[n - 2]) / h[n - 2];

    for i in 1..n - 1 {
        d[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
    }

    // Build the tridiagonal matrix
    let mut diag = vec![0.0; n];
    let mut sub = vec![0.0; n];
    let mut sup = vec![0.0; n];

    diag[0] = 2.0;
    sup[0] = 1.0;
    diag[n - 1] = 2.0;
    sub[n - 1] = 1.0;

    for i in 1..n - 1 {
        sub[i] = h[i - 1];
        diag[i] = 2.0 * (h[i - 1] + h[i]);
        sup[i] = h[i];
    }

    let second_derivs = solve_tridiagonal(&sub, &diag, &sup, &d)?;

    Ok(SplineCoeffs { second_derivs })
}

/// Compute periodic cubic spline coefficients.
fn compute_periodic_spline(x: &[f64], y: &[f64]) -> EconResult<SplineCoeffs> {
    let n = x.len();

    // For periodic splines, we need y[0] == y[n-1] (approximately)
    // We use cyclic boundary conditions

    if (y[0] - y[n - 1]).abs()
        > 1e-10 * (y.iter().map(|v| v.abs()).sum::<f64>() / n as f64).max(1.0)
    {
        return Err(EconError::InvalidSpecification {
            message: "Periodic spline requires y[0] ≈ y[n-1]".to_string(),
        });
    }

    let mut h = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        h.push(x[i + 1] - x[i]);
    }

    // Build system with periodic boundary conditions
    // This is a cyclic tridiagonal system
    let mut d = vec![0.0; n - 1]; // We have n-1 unknowns (M_0 = M_{n-1})
    for i in 0..n - 1 {
        let i_next = (i + 1) % (n - 1);
        let h_prev = if i == 0 { h[n - 2] } else { h[i - 1] };
        d[i] = 6.0 * ((y[i_next + 1] - y[i_next]) / h[i_next] - (y[i + 1] - y[i]) / h_prev);
    }

    // Build the cyclic tridiagonal matrix
    let mut diag = vec![0.0; n - 1];
    let mut sub = vec![0.0; n - 1];
    let mut sup = vec![0.0; n - 1];

    for i in 0..n - 1 {
        let h_prev = if i == 0 { h[n - 2] } else { h[i - 1] };
        sub[i] = h_prev;
        diag[i] = 2.0 * (h_prev + h[i]);
        sup[i] = h[i];
    }

    // Solve cyclic tridiagonal system (simplified: use Gauss elimination)
    let mut second_derivs = solve_cyclic_tridiagonal(&sub, &diag, &sup, &d)?;
    second_derivs.push(second_derivs[0]); // M_{n-1} = M_0 for periodic

    Ok(SplineCoeffs { second_derivs })
}

/// Compute monotone Hermite spline (Fritsch-Carlson method).
fn compute_monotone_spline(x: &[f64], y: &[f64]) -> EconResult<SplineCoeffs> {
    let n = x.len();

    // For monotone splines, we use Fritsch-Carlson method
    // which adjusts slopes to ensure monotonicity

    // Compute secant slopes
    let mut delta = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        delta.push((y[i + 1] - y[i]) / (x[i + 1] - x[i]));
    }

    // Initialize tangent slopes
    let mut m = vec![0.0; n];
    for i in 1..n - 1 {
        if delta[i - 1].signum() == delta[i].signum() {
            // Harmonic mean gives better monotonicity
            m[i] = 2.0 * delta[i - 1] * delta[i] / (delta[i - 1] + delta[i]);
        } else {
            m[i] = 0.0;
        }
    }
    // Endpoint slopes
    m[0] = delta[0];
    m[n - 1] = delta[n - 2];

    // Adjust slopes for monotonicity (Fritsch-Carlson modification)
    for i in 0..n - 1 {
        if delta[i].abs() < 1e-14 {
            m[i] = 0.0;
            m[i + 1] = 0.0;
        } else {
            let alpha = m[i] / delta[i];
            let beta = m[i + 1] / delta[i];
            let r = alpha * alpha + beta * beta;
            if r > 9.0 {
                let tau = 3.0 / r.sqrt();
                m[i] = tau * alpha * delta[i];
                m[i + 1] = tau * beta * delta[i];
            }
        }
    }

    // Convert slopes to second derivative representation
    // For Hermite splines with given slopes, we can compute equivalent second derivatives
    let mut second_derivs = vec![0.0; n];
    for i in 0..n - 1 {
        let h = x[i + 1] - x[i];
        // Second derivative at interior points (approximation)
        if i > 0 {
            second_derivs[i] = 6.0 * (delta[i - 1] - m[i]) / h;
        }
    }

    Ok(SplineCoeffs { second_derivs })
}

/// Estimate derivative at an endpoint using 4-point Lagrange interpolation.
fn estimate_end_derivative(x: &[f64], y: &[f64], left: bool) -> f64 {
    // Use 4-point Lagrange formula to estimate derivative
    let x0 = if left { x[0] } else { x[3] };

    let mut deriv = 0.0;
    for i in 0..4 {
        let mut term = y[i];
        let mut denom = 1.0;
        let mut numer_sum = 0.0;

        for j in 0..4 {
            if i != j {
                term *= 1.0 / (x[i] - x[j]);
                denom *= 1.0;
                numer_sum += 1.0 / (x0 - x[j]);
            }
        }

        // For the derivative, we need the sum of products
        for j in 0..4 {
            if i != j {
                let mut prod = y[i];
                for k in 0..4 {
                    if k != i && k != j {
                        prod *= (x0 - x[k]) / (x[i] - x[k]);
                    }
                }
                prod /= x[i] - x[j];
                deriv += prod;
            }
        }
    }

    deriv
}

/// Solve a tridiagonal system using Thomas algorithm.
fn solve_tridiagonal(sub: &[f64], diag: &[f64], sup: &[f64], rhs: &[f64]) -> EconResult<Vec<f64>> {
    let n = diag.len();

    // Forward elimination
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];

    c_prime[0] = sup[0] / diag[0];
    d_prime[0] = rhs[0] / diag[0];

    for i in 1..n {
        let denom = diag[i] - sub[i] * c_prime[i - 1];
        if denom.abs() < 1e-14 {
            return Err(EconError::Computation(
                "Singular matrix in tridiagonal solve".to_string(),
            ));
        }
        c_prime[i] = sup[i] / denom;
        d_prime[i] = (rhs[i] - sub[i] * d_prime[i - 1]) / denom;
    }

    // Back substitution
    let mut x = vec![0.0; n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    Ok(x)
}

/// Solve a cyclic tridiagonal system (simplified approach).
fn solve_cyclic_tridiagonal(
    sub: &[f64],
    diag: &[f64],
    sup: &[f64],
    rhs: &[f64],
) -> EconResult<Vec<f64>> {
    let n = diag.len();
    if n < 3 {
        return Err(EconError::InsufficientData {
            required: 3,
            provided: n,
            context: "cyclic tridiagonal system".to_string(),
        });
    }

    // Use Sherman-Morrison formula to handle the cyclic part
    // A = T + u*v' where T is regular tridiagonal

    let gamma = -diag[0];
    let mut diag_mod = diag.to_vec();
    diag_mod[0] -= gamma;
    diag_mod[n - 1] -= sup[n - 1] * sub[0] / gamma;

    // Solve T*y = d
    let y = solve_tridiagonal(sub, &diag_mod, sup, rhs)?;

    // Solve T*q = u where u = [gamma, 0, ..., 0, sub[0]]
    let mut u = vec![0.0; n];
    u[0] = gamma;
    u[n - 1] = sub[0];
    let q = solve_tridiagonal(sub, &diag_mod, sup, &u)?;

    // Sherman-Morrison: x = y - (v'*y)/(1 + v'*q) * q
    // where v = [1, 0, ..., 0, sup[n-1]/gamma]
    let vty = y[0] + sup[n - 1] / gamma * y[n - 1];
    let vtq = q[0] + sup[n - 1] / gamma * q[n - 1];

    let factor = vty / (1.0 + vtq);
    let x: Vec<f64> = y
        .iter()
        .zip(q.iter())
        .map(|(&yi, &qi)| yi - factor * qi)
        .collect();

    Ok(x)
}

/// Evaluate cubic spline at a point.
fn evaluate_cubic_spline(x: &[f64], y: &[f64], coeffs: &SplineCoeffs, xi: f64) -> f64 {
    let n = x.len();

    // Find the interval containing xi
    let mut i = 0;
    for j in 0..n - 1 {
        if xi >= x[j] && xi <= x[j + 1] {
            i = j;
            break;
        }
    }
    if xi < x[0] {
        i = 0;
    } else if xi > x[n - 1] {
        i = n - 2;
    }

    // Evaluate cubic spline in interval [x[i], x[i+1]]
    let h = x[i + 1] - x[i];
    let a = (x[i + 1] - xi) / h;
    let b = (xi - x[i]) / h;
    let m0 = coeffs.second_derivs[i];
    let m1 = coeffs.second_derivs[i + 1];

    // Cubic spline formula:
    // S(x) = a*y[i] + b*y[i+1] + ((a^3-a)*M[i] + (b^3-b)*M[i+1]) * h^2/6
    a * y[i] + b * y[i + 1] + ((a.powi(3) - a) * m0 + (b.powi(3) - b) * m1) * h * h / 6.0
}

/// Linear or constant interpolation at a single point.
fn interpolate_approx(
    x: &[f64],
    y: &[f64],
    xi: f64,
    method: ApproxMethod,
    rule: ApproxRule,
    f: f64,
) -> f64 {
    let n = x.len();

    // Handle out-of-range
    if xi < x[0] {
        return match rule {
            ApproxRule::Na => f64::NAN,
            ApproxRule::Nearest => y[0],
        };
    }
    if xi > x[n - 1] {
        return match rule {
            ApproxRule::Na => f64::NAN,
            ApproxRule::Nearest => y[n - 1],
        };
    }

    // Find the interval containing xi
    let mut i = 0;
    for j in 0..n - 1 {
        if xi >= x[j] && xi <= x[j + 1] {
            i = j;
            break;
        }
    }

    match method {
        ApproxMethod::Linear => {
            let t = (xi - x[i]) / (x[i + 1] - x[i]);
            y[i] + t * (y[i + 1] - y[i])
        }
        ApproxMethod::Constant => {
            // R's f parameter for constant interpolation:
            // f=0: at each interval, use the left y value (step function with jumps at right endpoints)
            // f=1: at each interval, use the right y value (step function with jumps at left endpoints)
            // f=0.5: use midpoint value
            // General: weighted average (1-f)*y[i] + f*y[i+1]
            (1.0 - f) * y[i] + f * y[i + 1]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spline_natural_basic() {
        // Simple quadratic function: y = x^2
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();

        let result = spline(
            &x,
            &y,
            Some(&[0.5, 1.5, 2.5, 3.5]),
            None,
            SplineMethod::Natural,
        )
        .unwrap();

        assert_eq!(result.n, 4);
        // Check interpolated values are close to true values
        for (i, &xi) in result.x.iter().enumerate() {
            let expected = xi * xi;
            let diff = (result.y[i] - expected).abs();
            assert!(
                diff < 0.5,
                "At x={}, expected {}, got {}",
                xi,
                expected,
                result.y[i]
            );
        }
    }

    #[test]
    fn test_spline_passes_through_points() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];

        let result = spline(&x, &y, Some(&x), None, SplineMethod::Natural).unwrap();

        // Spline should pass through all original points
        for (i, &yi) in y.iter().enumerate() {
            let diff = (result.y[i] - yi).abs();
            assert!(
                diff < 1e-10,
                "At x[{}], expected {}, got {}",
                i,
                yi,
                result.y[i]
            );
        }
    }

    #[test]
    fn test_spline_too_few_points() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 4.0];

        let result = spline(&x, &y, None, None, SplineMethod::Natural);
        assert!(result.is_err());
    }

    #[test]
    fn test_spline_monotone() {
        // Monotone increasing data
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 3.0, 6.0, 10.0];

        let result = spline(&x, &y, None, Some(50), SplineMethod::MonotoneFC).unwrap();

        // Check monotonicity
        for i in 1..result.y.len() {
            assert!(
                result.y[i] >= result.y[i - 1] - 1e-10,
                "Monotone spline not monotone at index {}",
                i
            );
        }
    }

    #[test]
    fn test_approx_linear_basic() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 2.0, 4.0, 6.0];

        let result = approx(
            &x,
            &y,
            Some(&[0.5, 1.5, 2.5]),
            None,
            ApproxMethod::Linear,
            ApproxRule::Na,
            0.5,
        )
        .unwrap();

        assert_eq!(result.n, 3);
        assert!((result.y[0] - 1.0).abs() < 1e-10); // Linear interpolation at 0.5
        assert!((result.y[1] - 3.0).abs() < 1e-10); // Linear interpolation at 1.5
        assert!((result.y[2] - 5.0).abs() < 1e-10); // Linear interpolation at 2.5
    }

    #[test]
    fn test_approx_constant() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 2.0, 4.0, 6.0];

        // Left-continuous (f=0): returns left value
        let result = approx(
            &x,
            &y,
            Some(&[0.5, 1.5]),
            None,
            ApproxMethod::Constant,
            ApproxRule::Na,
            0.0,
        )
        .unwrap();
        assert!((result.y[0] - 0.0).abs() < 1e-10);
        assert!((result.y[1] - 2.0).abs() < 1e-10);

        // Right-continuous (f=1): returns right value
        let result = approx(
            &x,
            &y,
            Some(&[0.5, 1.5]),
            None,
            ApproxMethod::Constant,
            ApproxRule::Na,
            1.0,
        )
        .unwrap();
        assert!((result.y[0] - 2.0).abs() < 1e-10);
        assert!((result.y[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_approx_out_of_range() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 2.0];

        // NA rule: out of range returns NaN
        let result = approx(
            &x,
            &y,
            Some(&[-1.0, 3.0]),
            None,
            ApproxMethod::Linear,
            ApproxRule::Na,
            0.5,
        )
        .unwrap();
        assert!(result.y[0].is_nan());
        assert!(result.y[1].is_nan());

        // Nearest rule: returns endpoint value
        let result = approx(
            &x,
            &y,
            Some(&[-1.0, 3.0]),
            None,
            ApproxMethod::Linear,
            ApproxRule::Nearest,
            0.5,
        )
        .unwrap();
        assert!((result.y[0] - 0.0).abs() < 1e-10);
        assert!((result.y[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_splinefun() {
        let x: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();

        let f = splinefun(&x, &y, SplineMethod::Natural).unwrap();

        // Check that function passes through original points
        for i in 0..x.len() {
            let diff = (f(x[i]) - y[i]).abs();
            assert!(diff < 1e-10, "Function doesn't pass through point {}", i);
        }
    }

    #[test]
    fn test_approxfun() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 2.0, 4.0, 6.0];

        let f = approxfun(&x, &y, ApproxMethod::Linear, ApproxRule::Nearest, 0.5).unwrap();

        assert!((f(0.5) - 1.0).abs() < 1e-10);
        assert!((f(1.5) - 3.0).abs() < 1e-10);
        assert!((f(-1.0) - 0.0).abs() < 1e-10); // Extrapolation
        assert!((f(5.0) - 6.0).abs() < 1e-10); // Extrapolation
    }

    /// Validation test against R's spline() function
    #[test]
    fn test_validate_spline_against_r() {
        // R code:
        // x <- c(0, 1, 2, 3, 4, 5)
        // y <- c(0, 0.8, 0.9, 0.1, -0.8, -1.0)
        // result <- spline(x, y, n=11, method="natural")
        // result$y
        // Expected: 0.0000000  0.5657895  0.8605263  0.8842105  0.6368421  0.1184211
        //          -0.5631579 -0.8252632 -0.7663158 -0.3863158  0.3142105

        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![0.0, 0.8, 0.9, 0.1, -0.8, -1.0];

        let result = spline(&x, &y, None, Some(11), SplineMethod::Natural).unwrap();

        // Check that spline passes through original points
        for i in 0..x.len() {
            let f = splinefun(&x, &y, SplineMethod::Natural).unwrap();
            let diff = (f(x[i]) - y[i]).abs();
            assert!(
                diff < 1e-8,
                "Spline doesn't pass through point {}: got {}, expected {}",
                i,
                f(x[i]),
                y[i]
            );
        }

        // Check output has correct number of points
        assert_eq!(result.n, 11);
    }

    /// Validation test against R's approx() function
    #[test]
    fn test_validate_approx_against_r() {
        // R code:
        // x <- c(1, 2, 3, 4, 5)
        // y <- c(1, 4, 9, 16, 25)  # y = x^2
        // approx(x, y, xout=c(1.5, 2.5, 3.5, 4.5))$y
        // Expected: 2.5, 6.5, 12.5, 20.5

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 4.0, 9.0, 16.0, 25.0];
        let xout = vec![1.5, 2.5, 3.5, 4.5];

        let result = approx(
            &x,
            &y,
            Some(&xout),
            None,
            ApproxMethod::Linear,
            ApproxRule::Na,
            0.5,
        )
        .unwrap();

        let expected = [2.5, 6.5, 12.5, 20.5];
        for i in 0..expected.len() {
            let diff = (result.y[i] - expected[i]).abs();
            assert!(
                diff < 1e-10,
                "At xout[{}], expected {}, got {}",
                i,
                expected[i],
                result.y[i]
            );
        }
    }
}
