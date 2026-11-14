//! Linearly Constrained Optimization.
//!
//! This module implements the adaptive barrier algorithm for minimizing a function
//! subject to linear inequality constraints (ui %*% theta - ci >= 0).
//!
//! # Mathematical Background
//!
//! Solves the constrained optimization problem:
//!
//! minimize f(θ)  subject to Uθ - c ≥ 0
//!
//! using R's adaptive logarithmic barrier (Lange): at outer iterate θₖ the inner problem is
//!
//! minimize R(θ, θₖ) = f(θ) - μ Σᵢ [(uᵢ'θₖ - cᵢ) log(uᵢ'θ - cᵢ) - uᵢ'θ]
//!
//! with μ fixed (default 1e-4), solved by BFGS (with a gradient) or Nelder-Mead.
//!
//! ## Algorithm
//!
//! 1. Start with a point θ₀ in the interior of the feasible region
//! 2. Solve the barrier subproblem re-centred at the current iterate
//! 3. Stop when the barrier objective changes by less than `(1e-3 + |R|) outer_eps`, when the
//!    objective increases (code 11) or when the outer iterations run out (code 7)
//!
//! # References
//!
//! - Fiacco, A.V., & McCormick, G.P. (1968). *Nonlinear Programming: Sequential
//!   Unconstrained Minimization Techniques*. Wiley. Reprinted by SIAM, 1990.
//!   ISBN: 978-0898712544. The foundational work on barrier methods.
//!
//! - Gill, P.E., Murray, W., & Wright, M.H. (1981). *Practical Optimization*.
//!   Academic Press. ISBN: 978-0122839528. Chapter 5 on constrained optimization.
//!
//! - Nocedal, J., & Wright, S.J. (2006). *Numerical Optimization* (2nd ed.).
//!   Springer. ISBN: 978-0387303031. Chapter 19 on interior-point methods.
//!
//! - Lange, K. (2010). *Numerical Analysis for Statisticians* (2nd ed.). Springer.
//!   ISBN: 978-1441959447. Chapter 13 on constrained optimization.
//!
//! R equivalent: `stats::constrOptim()`

use crate::errors::{EconError, EconResult};
use serde::{Deserialize, Serialize};

/// Result of constrained optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstrOptimResult {
    /// Optimal parameter values
    pub par: Vec<f64>,
    /// Objective function value at `par` (without the barrier)
    pub value: f64,
    /// Number of function evaluations
    pub counts_fn: usize,
    /// Number of gradient evaluations
    pub counts_grad: usize,
    /// Convergence code as R's `constrOptim`: 0 success, 1 the last inner optimisation reached
    /// its iteration limit, 7 the outer iterations ran out, 11 the objective increased at an
    /// outer iteration
    pub convergence: i32,
    /// Convergence message
    pub message: Option<String>,
    /// The barrier term at the optimum: the last inner value of the barrier objective minus
    /// `value` (R's `barrier.value`)
    pub barrier_value: f64,
    /// Number of outer iterations
    pub outer_iterations: usize,
}

/// Configuration for constrained optimization (R's `constrOptim` defaults).
#[derive(Debug, Clone)]
pub struct ConstrOptimConfig {
    /// Barrier term multiplier (R's `mu`, default 1e-4); fixed across outer iterations. Must be
    /// finite and positive (p2a has no `fnscale`, so a maximisation is written as minimising -f)
    pub mu: f64,
    /// Maximum outer iterations (R's `outer.iterations`, default 100)
    pub outer_iterations: usize,
    /// Outer convergence tolerance (R's `outer.eps`, default 1e-5): stop when the barrier
    /// objective changes by less than `(1e-3 + |r|) outer_eps`. Must be finite and positive
    pub outer_eps: f64,
    /// Inner optimization method. BFGS needs a gradient (as R's `method = "BFGS"` without
    /// `grad`, a call without one is refused; Nelder-Mead is the gradient-free method).
    /// L-BFGS-B is refused: it needs finite values outside the region, which the barrier
    /// does not give
    pub method: OptimMethod,
    /// Maximum function evaluations per inner iteration
    pub max_fn_evals: usize,
    /// Gradient-norm tolerance of the inner BFGS (Nelder-Mead stops at R's `reltol`,
    /// `sqrt(.Machine$double.eps)`)
    pub inner_tol: f64,
}

impl Default for ConstrOptimConfig {
    fn default() -> Self {
        ConstrOptimConfig {
            mu: 1e-4,
            outer_iterations: 100,
            outer_eps: 1e-5,
            method: OptimMethod::BFGS,
            max_fn_evals: 1000,
            inner_tol: 1e-8,
        }
    }
}

/// Optimization method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimMethod {
    /// Nelder-Mead simplex (gradient-free)
    NelderMead,
    /// BFGS quasi-Newton method
    BFGS,
    /// L-BFGS-B limited memory BFGS. Not available: `constr_optim` refuses it
    LBFGSB,
}

/// Minimize a function subject to linear inequality constraints.
///
/// The feasible region is defined by: `ui %*% theta - ci >= 0`
///
/// The algorithm uses an adaptive barrier approach that adds a logarithmic
/// penalty term to the objective function to enforce the constraints.
///
/// # Arguments
/// * `theta0` - Initial parameter values (must satisfy constraints)
/// * `f` - Objective function to minimize: fn(&[f64]) -> f64
/// * `grad` - Optional gradient function: fn(&[f64]) -> Vec<f64>
/// * `ui` - Constraint matrix (k x p), where k = number of constraints, p = number of parameters
/// * `ci` - Constraint vector (length k)
/// * `config` - Optimization configuration
///
/// # Returns
/// A `ConstrOptimResult` containing the optimal solution
///
/// # Example
/// ```
/// use p2a_core::stats::constroptim::{constr_optim, ConstrOptimConfig};
///
/// // Minimize x^2 + y^2 subject to x + y >= 1
/// let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
/// let grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
///
/// let ui = vec![vec![1.0, 1.0]]; // x + y >= 1
/// let ci = vec![1.0];
/// let theta0 = vec![1.0, 1.0]; // Feasible starting point
///
/// let result = constr_optim(&theta0, f, Some(grad), &ui, &ci, ConstrOptimConfig::default()).unwrap();
/// println!("Optimal: {:?}, Value: {}", result.par, result.value);
/// ```
pub fn constr_optim<F, G>(
    theta0: &[f64],
    f: F,
    grad: Option<G>,
    ui: &[Vec<f64>],
    ci: &[f64],
    config: ConstrOptimConfig,
) -> EconResult<ConstrOptimResult>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let invalid = |message: String| EconError::InvalidSpecification { message };
    let p = theta0.len();
    let k = ci.len();

    // Validate inputs
    if ui.len() != k {
        return Err(invalid(format!(
            "ui has {} rows but ci has {} elements",
            ui.len(),
            k
        )));
    }
    for (i, row) in ui.iter().enumerate() {
        if row.len() != p {
            return Err(invalid(format!(
                "ui row {} has {} elements but theta has {} elements",
                i,
                row.len(),
                p
            )));
        }
    }
    if !(config.mu.is_finite() && config.mu > 0.0) {
        return Err(invalid(format!(
            "constrOptim mu must be finite and positive, got {}",
            config.mu
        )));
    }
    if !(config.outer_eps.is_finite() && config.outer_eps > 0.0) {
        return Err(invalid(format!(
            "constrOptim outer_eps must be finite and positive, got {}",
            config.outer_eps
        )));
    }
    // R's inner methods: BFGS needs the gradient (R's explicit method = "BFGS" without one
    // fails). p2a has no bounded optimiser, and L-BFGS-B needs finite values outside the
    // region, which the barrier does not give. Both are refused rather than replaced.
    let use_bfgs = match (config.method, grad.is_some()) {
        (OptimMethod::NelderMead, _) => false,
        (OptimMethod::BFGS, true) => true,
        (OptimMethod::BFGS, false) => {
            return Err(invalid(
                "constrOptim method BFGS needs a gradient: pass grad, or use method Nelder-Mead"
                    .to_string(),
            ));
        }
        (OptimMethod::LBFGSB, _) => {
            return Err(invalid(
                "constrOptim method L-BFGS-B is not available (it needs finite values outside \
                 the feasible region, which the barrier does not give): use BFGS with a \
                 gradient, or Nelder-Mead"
                    .to_string(),
            ));
        }
    };

    // Check that initial point satisfies constraints
    let margin0 = compute_constraint_margin(theta0, ui, ci);
    if margin0.iter().any(|&m| m <= 0.0) {
        return Err(invalid(
            "Initial value is not feasible (does not satisfy constraints)".to_string(),
        ));
    }
    // R's nmmin stops on a non-finite objective at the start value.
    if !f(theta0).is_finite() {
        return Err(invalid(
            "function cannot be evaluated at initial parameters".to_string(),
        ));
    }

    // mu > 0 (checked above), so R's sign(mu) is 1
    let mu = config.mu;
    // R's adaptive barrier (Lange): R(theta, theta_old) = f(theta)
    //   - mu sum[(u_i theta_old - c_i) log(u_i theta - c_i) - u_i theta],
    // NaN outside the region, so the inner optimiser rejects the point.
    let barrier_objective = |x: &[f64], old_margin: &[f64]| -> f64 {
        let mut bar = 0.0;
        for ((row, &c), &g_old) in ui.iter().zip(ci).zip(old_margin) {
            let u_theta: f64 = row.iter().zip(x).map(|(&u, &t)| u * t).sum();
            let g = u_theta - c;
            if g < 0.0 {
                return f64::NAN;
            }
            bar += g_old * g.ln() - u_theta;
        }
        if !bar.is_finite() {
            bar = f64::NEG_INFINITY;
        }
        f(x) - mu * bar
    };

    let mut theta = theta0.to_vec();
    let mut obj = f(&theta);
    let mut r = barrier_objective(&theta, &margin0);
    let mut obj_old = obj;
    let mut par = theta.clone();
    let mut inner_code = 0;
    let mut fn_count = 0;
    let mut grad_count = 0;
    let mut outer_iter = 0;

    for i in 1..=config.outer_iterations {
        crate::resource::checkpoint()?;
        outer_iter = i;
        obj_old = obj;
        let r_old = r;
        let old_margin = compute_constraint_margin(&theta, ui, ci);
        let fun = |x: &[f64]| barrier_objective(x, &old_margin);

        let (new_par, value, fn_evals, grad_evals, code) = match (use_bfgs, grad.as_ref()) {
            (true, Some(g)) => {
                // dR = grad f - mu sum_i u_i ((u_i theta_old - c_i) / (u_i theta - c_i) - 1)
                let d_r = |x: &[f64]| {
                    let mut grad_vec = g(x);
                    let margin = compute_constraint_margin(x, ui, ci);
                    for ((row, &m), &m_old) in ui.iter().zip(&margin).zip(&old_margin) {
                        let w = m_old / m - 1.0;
                        for (gv, &u) in grad_vec.iter_mut().zip(row) {
                            *gv -= mu * u * w;
                        }
                    }
                    grad_vec
                };
                bfgs_with_grad(&theta, fun, d_r, config.max_fn_evals, config.inner_tol)
            }
            _ => nelder_mead(&theta, fun, config.max_fn_evals, NM_RELTOL),
        };
        fn_count += fn_evals;
        grad_count += grad_evals;
        par = new_par;
        inner_code = code;
        r = value;

        // R: stop when the barrier objective no longer moves (theta is not advanced, the inner
        // result is returned); otherwise advance and stop if the objective got worse.
        if r.is_finite()
            && r_old.is_finite()
            && (r - r_old).abs() < (1e-3 + r.abs()) * config.outer_eps
        {
            break;
        }
        theta = par.clone();
        obj = f(&theta);
        if obj > obj_old {
            break;
        }
    }

    let (convergence, message) = if obj > obj_old {
        (
            11,
            format!("Objective function increased at outer iteration {outer_iter}"),
        )
    } else if outer_iter == config.outer_iterations {
        (
            7,
            "Barrier algorithm ran out of iterations and did not converge".to_string(),
        )
    } else if inner_code != 0 {
        (
            inner_code,
            "the inner optimiser reached its iteration limit".to_string(),
        )
    } else {
        (0, "Converged".to_string())
    };

    let value = f(&par);
    Ok(ConstrOptimResult {
        par,
        value,
        counts_fn: fn_count,
        counts_grad: grad_count,
        convergence,
        message: Some(message),
        barrier_value: r - value,
        outer_iterations: outer_iter,
    })
}

/// Compute constraint margin: ui %*% theta - ci
fn compute_constraint_margin(theta: &[f64], ui: &[Vec<f64>], ci: &[f64]) -> Vec<f64> {
    ui.iter()
        .zip(ci.iter())
        .map(|(row, &c)| {
            row.iter()
                .zip(theta.iter())
                .map(|(&u, &t)| u * t)
                .sum::<f64>()
                - c
        })
        .collect()
}

/// R's `optim` default `reltol`, `sqrt(.Machine$double.eps)`, the Nelder-Mead stopping rule.
const NM_RELTOL: f64 = 1.490_116_119_384_765_6e-8;

/// R's nmmin (src/appl/optim.c): non-finite function values become big.
const NM_BIG: f64 = 1.0e35;

/// Nelder-Mead simplex optimization (gradient-free). Returns (par, value, function evaluations,
/// gradient evaluations, code): code 0 when the simplex's value range fell within `tol` relative to
/// the best value (R's `reltol` rule), 1 at the iteration limit.
fn nelder_mead<F>(
    x0: &[f64],
    f_user: F,
    max_evals: usize,
    tol: f64,
) -> (Vec<f64>, f64, usize, usize, i32)
where
    F: Fn(&[f64]) -> f64,
{
    let f = |x: &[f64]| {
        let v = f_user(x);
        if v.is_finite() { v } else { NM_BIG }
    };
    let n = x0.len();
    let mut fn_evals = 0;

    // Initialize simplex
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());

    for i in 0..n {
        let mut vertex = x0.to_vec();
        vertex[i] += if vertex[i].abs() > 1e-10 {
            0.05 * vertex[i]
        } else {
            0.00025
        };
        simplex.push(vertex);
    }

    // Evaluate function at all vertices
    let mut values: Vec<f64> = simplex
        .iter()
        .map(|v| {
            fn_evals += 1;
            f(v)
        })
        .collect();

    // Nelder-Mead parameters
    let alpha = 1.0; // Reflection
    let gamma = 2.0; // Expansion
    let rho = 0.5; // Contraction
    let sigma = 0.5; // Shrink

    for _ in 0..max_evals {
        // Sort vertices by function value
        let mut indices: Vec<usize> = (0..=n).collect();
        indices.sort_by(|&a, &b| values[a].total_cmp(&values[b]));

        // R's nmmin: stop when the simplex's value range is within reltol of the best value
        let f_range = values[indices[n]] - values[indices[0]];
        if f_range <= tol * (values[indices[0]].abs() + tol) {
            return (
                simplex[indices[0]].clone(),
                values[indices[0]],
                fn_evals,
                0,
                0,
            );
        }

        // Compute centroid (excluding worst point)
        let mut centroid = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                centroid[j] += simplex[indices[i]][j];
            }
        }
        for c in &mut centroid {
            *c /= n as f64;
        }

        // Reflection
        let worst_idx = indices[n];
        let reflected: Vec<f64> = centroid
            .iter()
            .zip(simplex[worst_idx].iter())
            .map(|(&c, &w)| c + alpha * (c - w))
            .collect();

        fn_evals += 1;
        let f_reflected = f(&reflected);

        if f_reflected < values[indices[0]] {
            // Try expansion
            let expanded: Vec<f64> = centroid
                .iter()
                .zip(reflected.iter())
                .map(|(&c, &r)| c + gamma * (r - c))
                .collect();

            fn_evals += 1;
            let f_expanded = f(&expanded);

            if f_expanded < f_reflected {
                simplex[worst_idx] = expanded;
                values[worst_idx] = f_expanded;
            } else {
                simplex[worst_idx] = reflected;
                values[worst_idx] = f_reflected;
            }
        } else if f_reflected < values[indices[n - 1]] {
            // Accept reflection
            simplex[worst_idx] = reflected;
            values[worst_idx] = f_reflected;
        } else {
            // Contraction
            let contracted: Vec<f64> = if f_reflected < values[worst_idx] {
                // Outside contraction
                centroid
                    .iter()
                    .zip(reflected.iter())
                    .map(|(&c, &r)| c + rho * (r - c))
                    .collect()
            } else {
                // Inside contraction
                centroid
                    .iter()
                    .zip(simplex[worst_idx].iter())
                    .map(|(&c, &w)| c + rho * (w - c))
                    .collect()
            };

            fn_evals += 1;
            let f_contracted = f(&contracted);

            if f_contracted < values[worst_idx].min(f_reflected) {
                simplex[worst_idx] = contracted;
                values[worst_idx] = f_contracted;
            } else {
                // Shrink
                let best = simplex[indices[0]].clone();
                for i in 1..=n {
                    for j in 0..n {
                        simplex[indices[i]][j] =
                            best[j] + sigma * (simplex[indices[i]][j] - best[j]);
                    }
                    fn_evals += 1;
                    values[indices[i]] = f(&simplex[indices[i]]);
                }
            }
        }
    }

    // Return best point found
    let mut best_idx = 0;
    for i in 1..=n {
        if values[i] < values[best_idx] {
            best_idx = i;
        }
    }
    (simplex[best_idx].clone(), values[best_idx], fn_evals, 0, 1)
}

/// BFGS optimization with gradient. Returns (par, value, function evaluations, gradient
/// evaluations, code): code 0 when the gradient norm fell below `tol` or no step along the search
/// direction decreases the function, 1 at the iteration limit.
fn bfgs_with_grad<F, G>(
    x0: &[f64],
    f: F,
    grad: G,
    max_evals: usize,
    tol: f64,
) -> (Vec<f64>, f64, usize, usize, i32)
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut fn_evals = 0;
    let mut grad_evals = 0;

    // Initialize inverse Hessian approximation as identity
    let mut h_inv: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();

    fn_evals += 1;
    let mut fx = f(&x);
    grad_evals += 1;
    let mut gx = grad(&x);
    let mut h_is_identity = true;

    for _ in 0..max_evals {
        // Check gradient convergence
        let grad_norm: f64 = gx.iter().map(|&g| g * g).sum::<f64>().sqrt();
        if grad_norm < tol {
            return (x, fx, fn_evals, grad_evals, 0);
        }

        // Compute search direction: p = -H_inv * g
        let mut p = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                p[i] -= h_inv[i][j] * gx[j];
            }
        }

        // Line search (simple backtracking)
        let mut alpha = 1.0;
        let c1 = 1e-4;
        let mut x_new = x.clone();
        let mut found_valid_step = false;

        let dot: f64 = gx.iter().zip(p.iter()).map(|(&g, &d)| g * d).sum();

        for _ in 0..30 {
            for i in 0..n {
                x_new[i] = x[i] + alpha * p[i];
            }
            fn_evals += 1;
            let fx_new = f(&x_new);

            // Armijo condition - only accept finite values that satisfy sufficient decrease
            if fx_new.is_finite() && fx_new <= fx + c1 * alpha * dot {
                found_valid_step = true;
                break;
            }
            // Reject and continue backtracking
            alpha *= 0.5;
        }

        // No step decreases the function. Along the steepest descent direction that makes x a
        // minimum to working precision; otherwise restart from the steepest descent direction
        // (repeating the search from the same state would fail the same way).
        if !found_valid_step {
            if h_is_identity {
                return (x, fx, fn_evals, grad_evals, 0);
            }
            for (i, row) in h_inv.iter_mut().enumerate() {
                for (j, h) in row.iter_mut().enumerate() {
                    *h = if i == j { 1.0 } else { 0.0 };
                }
            }
            h_is_identity = true;
            continue;
        }

        // Update position
        let s: Vec<f64> = (0..n).map(|i| x_new[i] - x[i]).collect();
        x = x_new;

        // Get new gradient
        grad_evals += 1;
        let gx_new = grad(&x);
        let y: Vec<f64> = (0..n).map(|i| gx_new[i] - gx[i]).collect();

        fn_evals += 1;
        fx = f(&x);
        gx = gx_new;

        // BFGS update
        let sy: f64 = s.iter().zip(y.iter()).map(|(&si, &yi)| si * yi).sum();
        if sy.abs() < 1e-10 {
            continue;
        }

        // H_inv = (I - sy' / sy) * H_inv * (I - ys' / sy) + ss' / sy
        let mut h_y = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                h_y[i] += h_inv[i][j] * y[j];
            }
        }

        let yhy: f64 = y.iter().zip(h_y.iter()).map(|(&yi, &hyi)| yi * hyi).sum();

        for i in 0..n {
            for j in 0..n {
                h_inv[i][j] +=
                    ((sy + yhy) * s[i] * s[j] - h_y[i] * s[j] - s[i] * h_y[j]) / (sy * sy);
            }
        }
        h_is_identity = false;
    }

    (x, fx, fn_evals, grad_evals, 1)
}

/// Run constrained optimization (convenience wrapper).
pub fn run_constr_optim<F, G>(
    theta0: &[f64],
    f: F,
    grad: Option<G>,
    ui: &[Vec<f64>],
    ci: &[f64],
    config: Option<ConstrOptimConfig>,
) -> EconResult<ConstrOptimResult>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    constr_optim(theta0, f, grad, ui, ci, config.unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_constr_optim_refuses_non_finite_start_value() {
        // R's nmmin stops with "function cannot be evaluated at initial parameters"; NaN used
        // to enter the simplex sort.
        let f = |t: &[f64]| {
            if t[0] < 1.0 {
                f64::NAN
            } else {
                (t[0] - 2.0).powi(2) + 0.1 * t[1].powi(2)
            }
        };
        let r = constr_optim(
            &[0.5, 0.5],
            f,
            None::<fn(&[f64]) -> Vec<f64>>,
            &[vec![1.0, 0.0]],
            &[0.0],
            ConstrOptimConfig {
                method: OptimMethod::NelderMead,
                ..Default::default()
            },
        );
        let e = r.unwrap_err().to_string();
        assert!(
            e.contains("cannot be evaluated at initial parameters"),
            "{e}"
        );
    }

    #[test]
    fn test_constr_optim_basic() {
        // Minimize x^2 + y^2 subject to x + y >= 1
        // Optimal is at x = y = 0.5, with value = 0.5
        let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];

        let ui = vec![vec![1.0, 1.0]]; // x + y >= 1
        let ci = vec![1.0];
        let theta0 = vec![1.0, 1.0]; // Feasible starting point

        let result = constr_optim(
            &theta0,
            f,
            Some(grad),
            &ui,
            &ci,
            ConstrOptimConfig::default(),
        )
        .unwrap();

        // Check constraint is satisfied (x + y >= 1)
        assert!(result.par[0] + result.par[1] >= 0.99, "Constraint violated");

        // Check we improved from starting point (f(1,1) = 2)
        assert!(result.value < 2.0, "Should improve from starting point");

        // Check result is reasonable (optimal is 0.5)
        // Barrier methods may not reach exact optimum but should be close
        assert!(
            result.value < 1.0,
            "Value should be less than 1.0, got {}",
            result.value
        );
    }

    #[test]
    fn test_constr_optim_multiple_constraints() {
        // Minimize -x - y subject to x >= 0, y >= 0, x + y <= 1
        // Optimal is at any point on x + y = 1 where x,y > 0
        // Optimal value is -1 (minimizing -x - y when x + y = 1)
        let f = |x: &[f64]| -x[0] - x[1];
        let grad = |_x: &[f64]| vec![-1.0, -1.0];

        let ui = vec![
            vec![1.0, 0.0],   // x >= 0
            vec![0.0, 1.0],   // y >= 0
            vec![-1.0, -1.0], // -x - y >= -1 (x + y <= 1)
        ];
        let ci = vec![0.0, 0.0, -1.0];
        let theta0 = vec![0.3, 0.3];

        let result = constr_optim(
            &theta0,
            f,
            Some(grad),
            &ui,
            &ci,
            ConstrOptimConfig::default(),
        )
        .unwrap();

        // Check all constraints are satisfied
        assert!(result.par[0] >= -0.01, "x should be non-negative");
        assert!(result.par[1] >= -0.01, "y should be non-negative");
        assert!(
            result.par[0] + result.par[1] <= 1.01,
            "x + y should be <= 1"
        );

        // Check we improved from starting point (f(0.3, 0.3) = -0.6)
        assert!(result.value < -0.5, "Should improve from starting point");

        // Optimal value is -1; should be reasonably close
        assert!(
            result.value <= -0.8,
            "Value should be close to optimal -1.0, got {}",
            result.value
        );
    }

    #[test]
    fn test_constr_optim_nelder_mead() {
        // Test without gradient
        let f = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);

        let ui = vec![vec![1.0, 0.0], vec![0.0, 1.0]]; // x >= 0, y >= 0
        let ci = vec![0.0, 0.0];
        let theta0 = vec![0.5, 0.5];

        let config = ConstrOptimConfig {
            method: OptimMethod::NelderMead,
            ..Default::default()
        };

        let result =
            constr_optim::<_, fn(&[f64]) -> Vec<f64>>(&theta0, f, None, &ui, &ci, config).unwrap();

        // Unconstrained optimum is (1, 2), which is feasible
        assert_relative_eq!(result.par[0], 1.0, epsilon = 0.1);
        assert_relative_eq!(result.par[1], 2.0, epsilon = 0.1);
    }

    #[test]
    fn test_constr_optim_infeasible_start() {
        let f = |x: &[f64]| x[0].powi(2);
        let ui = vec![vec![1.0]]; // x >= 1
        let ci = vec![1.0];
        let theta0 = vec![0.0]; // x = 0 doesn't satisfy x >= 1

        let result = constr_optim::<_, fn(&[f64]) -> Vec<f64>>(
            &theta0,
            f,
            None,
            &ui,
            &ci,
            ConstrOptimConfig::default(),
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_constraint_margin() {
        let theta = vec![2.0, 3.0];
        let ui = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let ci = vec![1.0, 2.0, 4.0];

        let margin = compute_constraint_margin(&theta, &ui, &ci);

        // x >= 1: 2 - 1 = 1
        // y >= 2: 3 - 2 = 1
        // x + y >= 4: 5 - 4 = 1
        assert_relative_eq!(margin[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(margin[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(margin[2], 1.0, epsilon = 1e-10);
    }

    // =========================================================================
    // Validation tests against R
    // =========================================================================

    // Case stat_constroptim_legacy (validation/r/cases/stats_power_loglin.R): R's constrOptim
    // on the same four problems. Outer iterations are a path quantity (R 3 on each): checked as
    // a property only.

    use p2a_validation::{Case, Check, Tol};

    /// STAT-62's rule for iterative methods: 1e-4 relative at R's solution.
    fn iter() -> Tol {
        Tol::iterative(1e-4)
    }

    #[test]
    fn test_validate_constr_optim_quadratic() {
        // R: constrOptim(c(1, 1), function(x) x[1]^2 + x[2]^2,
        //                grad = function(x) c(2*x[1], 2*x[2]),
        //                ui = matrix(c(1, 1), nrow = 1), ci = 1)
        let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
        let ui = vec![vec![1.0, 1.0]];
        let ci = vec![1.0];
        let r = constr_optim(
            &[1.0, 1.0],
            f,
            Some(grad),
            &ui,
            &ci,
            ConstrOptimConfig::default(),
        )
        .unwrap();
        assert!(r.outer_iterations <= 100 && r.barrier_value.is_finite());
        let case = Case::load("stat_constroptim_legacy");
        Check::new(&case)
            .vector("quadratic_par", &r.par, iter())
            .scalar("quadratic_value", r.value, iter())
            .scalar(
                "quadratic_convergence",
                f64::from(r.convergence),
                Tol::exact(),
            )
            .finish();
    }

    #[test]
    fn test_validate_constr_optim_linear_three_constraints() {
        // R: constrOptim(c(0.3, 0.3), function(x) -x[1] - x[2], grad = function(x) c(-1, -1),
        //                ui = matrix(c(1, 0, 0, 1, -1, -1), nrow = 3, byrow = TRUE),
        //                ci = c(0, 0, -1))
        let f = |x: &[f64]| -x[0] - x[1];
        let grad = |_x: &[f64]| vec![-1.0, -1.0];
        let ui = vec![
            vec![1.0, 0.0],   // x >= 0
            vec![0.0, 1.0],   // y >= 0
            vec![-1.0, -1.0], // x + y <= 1
        ];
        let ci = vec![0.0, 0.0, -1.0];
        let r = constr_optim(
            &[0.3, 0.3],
            f,
            Some(grad),
            &ui,
            &ci,
            ConstrOptimConfig::default(),
        )
        .unwrap();
        assert!(r.outer_iterations <= 100 && r.barrier_value.is_finite());
        let case = Case::load("stat_constroptim_legacy");
        Check::new(&case)
            .vector("lp3_par", &r.par, iter())
            .scalar("lp3_value", r.value, iter())
            .scalar("lp3_convergence", f64::from(r.convergence), Tol::exact())
            .finish();
    }

    #[test]
    fn test_validate_constr_optim_nelder_mead() {
        // R: constrOptim(c(0.5, 0.5), function(x) (x[1] - 1)^2 + (x[2] - 2)^2, grad = NULL,
        //                ui = diag(2), ci = c(0, 0))
        let f = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
        let ui = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let ci = vec![0.0, 0.0];
        let config = ConstrOptimConfig {
            method: OptimMethod::NelderMead,
            ..Default::default()
        };
        let r = constr_optim::<_, fn(&[f64]) -> Vec<f64>>(&[0.5, 0.5], f, None, &ui, &ci, config)
            .unwrap();
        assert!(r.outer_iterations <= 100 && r.barrier_value.is_finite());
        // The value at the optimum is 0 up to the inner tolerance (R 7.8e-13): rounding-level, so it
        // is checked as small, not against R's digits.
        assert!(r.value < 1e-9, "{}", r.value);
        let case = Case::load("stat_constroptim_legacy");
        Check::new(&case)
            .vector("nelder_mead_par", &r.par, iter())
            .scalar(
                "nelder_mead_convergence",
                f64::from(r.convergence),
                Tol::exact(),
            )
            .finish();
    }

    /// R's codes: 7 when the outer iterations run out (even if the last one converged), and 11
    /// when the objective increases between outer iterates (here an objective that grows with
    /// each call). No point is projected: the result is the last inner solution.
    #[test]
    fn test_constr_optim_codes_seven_and_eleven() {
        let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
        let mut cfg = ConstrOptimConfig::default();
        cfg.outer_iterations = 1;
        let r = constr_optim(&[1.0, 1.0], f, Some(grad), &[vec![1.0, 1.0]], &[1.0], cfg).unwrap();
        assert_eq!(r.convergence, 7, "{r:?}");
        assert_eq!(r.outer_iterations, 1);
        assert!(r.par[0] + r.par[1] > 1.0);

        let calls = std::cell::Cell::new(0usize);
        let growing = |x: &[f64]| {
            calls.set(calls.get() + 1);
            x[0].powi(2) + x[1].powi(2) + 0.1 * calls.get() as f64
        };
        let mut nm = ConstrOptimConfig::default();
        nm.method = OptimMethod::NelderMead;
        let r = constr_optim::<_, fn(&[f64]) -> Vec<f64>>(
            &[1.0, 1.0],
            growing,
            None,
            &[vec![1.0, 1.0]],
            &[1.0],
            nm,
        )
        .unwrap();
        assert_eq!(r.convergence, 11, "{r:?}");
        assert_eq!(r.outer_iterations, 1);
        assert!(r.message.unwrap().contains("increased"));
    }

    /// The inner methods p2a cannot run as asked are refused, naming the method and what is
    /// missing: BFGS without a gradient (as R's `method = "BFGS"` without `grad`) and L-BFGS-B
    /// (with or without one). Nothing is substituted. The methods that run (BFGS with the
    /// gradient, Nelder-Mead without one) reach R's solution of the quadratic problem; a zero,
    /// negative or non-finite `mu` and a non-positive `outer_eps` are refused by name.
    #[test]
    fn test_validate_constr_optim_inner_methods() {
        let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
        let ui = vec![vec![1.0, 1.0]];
        let ci = vec![1.0];
        let case = Case::load("stat_constroptim_legacy");
        let spec = |r: EconResult<ConstrOptimResult>| match r {
            Err(EconError::InvalidSpecification { message }) => message,
            other => panic!("expected InvalidSpecification, got {other:?}"),
        };
        let no_grad = None::<fn(&[f64]) -> Vec<f64>>;

        let m = spec(constr_optim(
            &[1.0, 1.0],
            f,
            no_grad,
            &ui,
            &ci,
            ConstrOptimConfig::default(),
        ));
        assert!(m.contains("BFGS") && m.contains("needs a gradient"), "{m}");
        for g in [None, Some(grad)] {
            let mut cfg = ConstrOptimConfig::default();
            cfg.method = OptimMethod::LBFGSB;
            let m = spec(constr_optim(&[1.0, 1.0], f, g, &ui, &ci, cfg));
            assert!(m.contains("L-BFGS-B is not available"), "{m}");
        }
        for mu in [0.0, -1e-4, f64::NAN, f64::INFINITY] {
            let mut cfg = ConstrOptimConfig::default();
            cfg.mu = mu;
            let m = spec(constr_optim(&[1.0, 1.0], f, Some(grad), &ui, &ci, cfg));
            assert!(m.contains("mu must be finite and positive"), "{m}");
        }
        for eps in [0.0, -1e-5, f64::NAN] {
            let mut cfg = ConstrOptimConfig::default();
            cfg.outer_eps = eps;
            let m = spec(constr_optim(&[1.0, 1.0], f, Some(grad), &ui, &ci, cfg));
            assert!(m.contains("outer_eps must be finite and positive"), "{m}");
        }

        let b = constr_optim(
            &[1.0, 1.0],
            f,
            Some(grad),
            &ui,
            &ci,
            ConstrOptimConfig::default(),
        )
        .unwrap();
        let mut nm = ConstrOptimConfig::default();
        nm.method = OptimMethod::NelderMead;
        let n = constr_optim(&[1.0, 1.0], f, no_grad, &ui, &ci, nm).unwrap();
        for p in [&b.par, &n.par] {
            for (a, r) in p.iter().zip(case.vector("quadratic_par")) {
                assert!(iter().allows(*a, r), "{p:?}");
            }
        }
        // the defaults are R's
        let d = ConstrOptimConfig::default();
        assert_eq!((d.mu, d.outer_iterations, d.outer_eps), (1e-4, 100, 1e-5));
    }

    #[test]
    fn test_validate_constr_optim_binding_constraint() {
        // R: constrOptim(2, function(x) x^2, grad = function(x) 2 * x, ui = matrix(1), ci = 1)
        let f = |x: &[f64]| x[0].powi(2);
        let grad = |x: &[f64]| vec![2.0 * x[0]];
        let ui = vec![vec![1.0]];
        let ci = vec![1.0];
        let r = constr_optim(
            &[2.0],
            f,
            Some(grad),
            &ui,
            &ci,
            ConstrOptimConfig::default(),
        )
        .unwrap();
        assert!(r.outer_iterations <= 100 && r.barrier_value.is_finite());
        let case = Case::load("stat_constroptim_legacy");
        Check::new(&case)
            .vector("binding_par", &r.par, iter())
            .scalar("binding_value", r.value, iter())
            .scalar(
                "binding_convergence",
                f64::from(r.convergence),
                Tol::exact(),
            )
            .finish();
    }
}
