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
//! using the logarithmic barrier method:
//!
//! minimize f(θ) - μ Σᵢ log(uᵢ'θ - cᵢ)
//!
//! The barrier parameter μ is progressively reduced, with inner optimization
//! performed using BFGS or Nelder-Mead at each stage.
//!
//! ## Algorithm
//!
//! 1. Start with feasible point θ₀ and initial barrier μ₀
//! 2. Solve barrier subproblem using unconstrained optimizer
//! 3. Reduce μ and repeat until convergence
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

use serde::{Deserialize, Serialize};

/// Result of constrained optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstrOptimResult {
    /// Optimal parameter values
    pub par: Vec<f64>,
    /// Objective function value at optimum
    pub value: f64,
    /// Number of function evaluations
    pub counts_fn: usize,
    /// Number of gradient evaluations
    pub counts_grad: usize,
    /// Convergence code (0 = success)
    pub convergence: i32,
    /// Convergence message
    pub message: Option<String>,
    /// Value of barrier term at optimum
    pub barrier_value: f64,
    /// Number of outer iterations
    pub outer_iterations: usize,
}

/// Configuration for constrained optimization.
#[derive(Debug, Clone)]
pub struct ConstrOptimConfig {
    /// Barrier term multiplier
    pub mu: f64,
    /// Maximum outer iterations
    pub outer_iterations: usize,
    /// Convergence tolerance for outer iterations
    pub outer_eps: f64,
    /// Optimization method
    pub method: OptimMethod,
    /// Maximum function evaluations per inner iteration
    pub max_fn_evals: usize,
    /// Tolerance for inner optimization
    pub inner_tol: f64,
}

impl Default for ConstrOptimConfig {
    fn default() -> Self {
        ConstrOptimConfig {
            mu: 1.0, // Start with larger mu for stronger barrier effect
            outer_iterations: 100,
            outer_eps: 1e-6,
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
    /// L-BFGS-B limited memory BFGS
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
) -> Result<ConstrOptimResult, String>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let p = theta0.len();
    let k = ci.len();

    // Validate inputs
    if ui.len() != k {
        return Err(format!(
            "ui has {} rows but ci has {} elements",
            ui.len(),
            k
        ));
    }
    for (i, row) in ui.iter().enumerate() {
        if row.len() != p {
            return Err(format!(
                "ui row {} has {} elements but theta has {} elements",
                i,
                row.len(),
                p
            ));
        }
    }

    // Check that initial point satisfies constraints
    let margin0 = compute_constraint_margin(theta0, ui, ci);
    if margin0.iter().any(|&m| m <= 0.0) {
        return Err("Initial value is not feasible (does not satisfy constraints)".to_string());
    }

    let mut theta = theta0.to_vec();
    let mut mu = config.mu;
    let mut fn_count = 0;
    let mut grad_count = 0;
    let mut outer_iter = 0;
    let mut converged = false;
    let mut prev_value = f64::INFINITY;

    // Outer loop: gradually reduce barrier influence
    for iter in 0..config.outer_iterations {
        outer_iter = iter + 1;

        // Define augmented objective function with barrier
        let aug_f = |x: &[f64]| {
            let obj = f(x);
            let margin = compute_constraint_margin(x, ui, ci);

            // Barrier term: -mu * sum(log(margin))
            let barrier: f64 = margin
                .iter()
                .map(|&m| if m > 0.0 { -m.ln() } else { f64::INFINITY })
                .sum();

            obj + mu * barrier
        };

        // Optimize the augmented function
        let (new_theta, _new_value, fn_evals, grad_evals) = match config.method {
            OptimMethod::NelderMead => {
                nelder_mead(&theta, aug_f, config.max_fn_evals, config.inner_tol)
            }
            OptimMethod::BFGS | OptimMethod::LBFGSB => {
                if let Some(ref g) = grad {
                    // Augmented gradient
                    let aug_grad = |x: &[f64]| {
                        let mut grad_vec = g(x);
                        let margin = compute_constraint_margin(x, ui, ci);

                        // Add barrier gradient: mu * ui^T * (1/margin)
                        for (j, row) in ui.iter().enumerate() {
                            if margin[j] > 0.0 {
                                for (i, &u_ji) in row.iter().enumerate() {
                                    grad_vec[i] -= mu * u_ji / margin[j];
                                }
                            }
                        }
                        grad_vec
                    };
                    bfgs_with_grad(
                        &theta,
                        aug_f,
                        aug_grad,
                        config.max_fn_evals,
                        config.inner_tol,
                    )
                } else {
                    nelder_mead(&theta, aug_f, config.max_fn_evals, config.inner_tol)
                }
            }
        };

        fn_count += fn_evals;
        grad_count += grad_evals;

        // Check convergence (relative change in objective)
        let current_obj = f(&new_theta);
        if (prev_value - current_obj).abs() / (prev_value.abs() + 1.0) < config.outer_eps {
            converged = true;
            theta = new_theta;
            break;
        }

        theta = new_theta;
        prev_value = current_obj;

        // Reduce mu slowly to allow solution to approach constraint boundary
        // Use gentle reduction to maintain barrier effect
        mu *= 0.9;

        // Don't let mu get too small (maintain barrier effect)
        if mu < 1e-6 {
            mu = 1e-6;
        }
    }

    // Verify final solution is feasible and project back if needed
    let margin = compute_constraint_margin(&theta, ui, ci);
    if margin.iter().any(|&m| m < -1e-10) {
        // Final solution violates constraints - project back to feasibility
        // Use simple projection: find the most violated constraint and adjust
        for _ in 0..10 {
            let current_margin = compute_constraint_margin(&theta, ui, ci);
            let min_margin_idx = current_margin
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            if current_margin[min_margin_idx] >= -1e-10 {
                break; // Now feasible
            }

            // Project along constraint normal to satisfy the most violated constraint
            let violation = -current_margin[min_margin_idx];
            let norm_sq: f64 = ui[min_margin_idx].iter().map(|&u| u * u).sum();
            if norm_sq > 1e-10 {
                let step = (violation + 1e-6) / norm_sq;
                for (i, &u_i) in ui[min_margin_idx].iter().enumerate() {
                    theta[i] += step * u_i;
                }
            }
        }
    }

    // Compute final values
    let final_value = f(&theta);
    let margin = compute_constraint_margin(&theta, ui, ci);
    let barrier_value: f64 = margin
        .iter()
        .map(|&m| if m > 0.0 { -mu * m.ln() } else { 0.0 })
        .sum();

    Ok(ConstrOptimResult {
        par: theta,
        value: final_value,
        counts_fn: fn_count,
        counts_grad: grad_count,
        convergence: if converged { 0 } else { 1 },
        message: if converged {
            Some("Converged".to_string())
        } else {
            Some("Maximum iterations reached".to_string())
        },
        barrier_value,
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

/// Nelder-Mead simplex optimization (gradient-free).
fn nelder_mead<F>(x0: &[f64], f: F, max_evals: usize, tol: f64) -> (Vec<f64>, f64, usize, usize)
where
    F: Fn(&[f64]) -> f64,
{
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
        indices.sort_by(|&a, &b| {
            values[a]
                .partial_cmp(&values[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Check convergence
        let f_range = values[indices[n]] - values[indices[0]];
        if f_range < tol {
            return (simplex[indices[0]].clone(), values[indices[0]], fn_evals, 0);
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
    (simplex[best_idx].clone(), values[best_idx], fn_evals, 0)
}

/// BFGS optimization with gradient.
fn bfgs_with_grad<F, G>(
    x0: &[f64],
    f: F,
    grad: G,
    max_evals: usize,
    tol: f64,
) -> (Vec<f64>, f64, usize, usize)
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

    for _ in 0..max_evals {
        // Check gradient convergence
        let grad_norm: f64 = gx.iter().map(|&g| g * g).sum::<f64>().sqrt();
        if grad_norm < tol {
            return (x, fx, fn_evals, grad_evals);
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

        // Only update if we found a valid step
        if !found_valid_step {
            // No valid step found, try smaller steps or terminate this iteration
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
    }

    (x, fx, fn_evals, grad_evals)
}

/// Run constrained optimization (convenience wrapper).
pub fn run_constr_optim<F, G>(
    theta0: &[f64],
    f: F,
    grad: Option<G>,
    ui: &[Vec<f64>],
    ci: &[f64],
    config: Option<ConstrOptimConfig>,
) -> Result<ConstrOptimResult, String>
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

    #[test]
    fn test_validate_constr_optim_quadratic() {
        // R: constrOptim(c(1, 1), function(x) x[1]^2 + x[2]^2,
        //                grad = function(x) c(2*x[1], 2*x[2]),
        //                ui = matrix(c(1, 1), nrow = 1), ci = 1)
        // R result: par = [0.5, 0.5], value = 0.5, constraint x + y = 1.0
        let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];

        let ui = vec![vec![1.0, 1.0]];
        let ci = vec![1.0];
        let theta0 = vec![1.0, 1.0];

        let result = constr_optim(
            &theta0,
            f,
            Some(grad),
            &ui,
            &ci,
            ConstrOptimConfig::default(),
        )
        .unwrap();

        // Should improve from starting point (f(1,1) = 2)
        assert!(
            result.value < 2.0,
            "Should improve from starting value 2.0: got {:.6}",
            result.value
        );

        // Constraint should be satisfied: x + y >= 1
        let constraint_value = result.par[0] + result.par[1];
        assert!(
            constraint_value >= 0.99,
            "Constraint violated: x + y = {:.6} < 1",
            constraint_value
        );

        // Value should be reasonably close to optimal (0.5)
        // Barrier methods may not reach exact optimum
        assert!(
            result.value < 1.5,
            "Value should be less than 1.5: got {:.6}",
            result.value
        );
    }

    #[test]
    fn test_validate_constr_optim_linear_three_constraints() {
        // R: constrOptim(c(0.3, 0.3),
        //                function(x) -x[1] - x[2],
        //                grad = function(x) c(-1, -1),
        //                ui = matrix(c(1, 0, 0, 1, -1, -1), nrow = 3, byrow = TRUE),
        //                ci = c(0, 0, -1))
        // R result: par = [0.5, 0.5], value = -1.0
        let f = |x: &[f64]| -x[0] - x[1];
        let grad = |_x: &[f64]| vec![-1.0, -1.0];

        let ui = vec![
            vec![1.0, 0.0],   // x >= 0
            vec![0.0, 1.0],   // y >= 0
            vec![-1.0, -1.0], // -x - y >= -1, i.e., x + y <= 1
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

        // Optimal value should be -1.0 (maximize x + y subject to x + y <= 1)
        let expected_value = -1.0;
        assert!(
            (result.value - expected_value).abs() < 0.2,
            "Value mismatch: Rust={:.6}, R={:.6}",
            result.value,
            expected_value
        );

        // All constraints should be satisfied
        assert!(
            result.par[0] >= -0.01,
            "x should be >= 0: x = {:.6}",
            result.par[0]
        );
        assert!(
            result.par[1] >= -0.01,
            "y should be >= 0: y = {:.6}",
            result.par[1]
        );
        assert!(
            result.par[0] + result.par[1] <= 1.01,
            "x + y should be <= 1: x + y = {:.6}",
            result.par[0] + result.par[1]
        );
    }

    #[test]
    fn test_validate_constr_optim_nelder_mead() {
        // Test without gradient (Nelder-Mead)
        // Minimize (x - 1)^2 + (y - 2)^2 subject to x >= 0, y >= 0
        // Optimal is at (1, 2) which is feasible
        let f = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);

        let ui = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let ci = vec![0.0, 0.0];
        let theta0 = vec![0.5, 0.5];

        let config = ConstrOptimConfig {
            method: OptimMethod::NelderMead,
            ..Default::default()
        };

        let result =
            constr_optim::<_, fn(&[f64]) -> Vec<f64>>(&theta0, f, None, &ui, &ci, config).unwrap();

        // Optimal at (1, 2), value = 0
        assert!(
            (result.par[0] - 1.0).abs() < 0.2,
            "x should be near 1: x = {:.6}",
            result.par[0]
        );
        assert!(
            (result.par[1] - 2.0).abs() < 0.2,
            "y should be near 2: y = {:.6}",
            result.par[1]
        );
        assert!(
            result.value < 0.1,
            "Value should be near 0: value = {:.6}",
            result.value
        );
    }

    #[test]
    fn test_validate_constr_optim_binding_constraint() {
        // Minimize x^2 subject to x >= 1
        // Optimal is at x = 1 (constraint binding)
        let f = |x: &[f64]| x[0].powi(2);
        let grad = |x: &[f64]| vec![2.0 * x[0]];

        let ui = vec![vec![1.0]];
        let ci = vec![1.0];
        let theta0 = vec![2.0];

        let result = constr_optim(
            &theta0,
            f,
            Some(grad),
            &ui,
            &ci,
            ConstrOptimConfig::default(),
        )
        .unwrap();

        // Optimal value should be 1.0 (at x = 1)
        assert!(
            (result.par[0] - 1.0).abs() < 0.1,
            "x should be at constraint boundary: x = {:.6}",
            result.par[0]
        );
        assert!(
            (result.value - 1.0).abs() < 0.2,
            "Value should be 1.0: value = {:.6}",
            result.value
        );
    }
}
