//! The damped Newton step of the ordered (`ordered_models`) and zero-inflated
//! (`count_models`) maximum-likelihood fits, their convergence test, and the checks
//! around them.

use ndarray::{Array1, Array2};

use crate::errors::EconResult;
use crate::linalg::matrix_ops::cholesky;
use crate::resource;

/// The m x m matrices a Newton fit holds at once: the Hessian and its damped copy, and
/// at the end, inside `cholesky_inverse` (the covariance), the faer copy, the factor,
/// the identity and the solution. (A step holds fewer: the factor of the damped copy,
/// its faer copy and the returned one.) Check the working set against the allocation
/// ceiling before building it.
pub(super) const DENSE_INVERSE_COPIES: usize = 6;

/// Step halvings before a line search gives up.
pub(super) const MAX_HALVINGS: usize = 30;

/// Levenberg damping attempts when the information is not positive definite.
pub(super) const MAX_DAMPING: usize = 20;

/// Refuse a fit whose [`DENSE_INVERSE_COPIES`] m x m matrices exceed the allocation
/// ceiling (`ResourceLimit`), before any of them is built.
pub(super) fn ensure_dense_inverse_fits(m: usize, what: &str) -> EconResult<()> {
    let cells = resource::checked_len(&[m, m, DENSE_INVERSE_COPIES], what)?;
    resource::check_alloc::<f64>(cells, what)
}

/// A step of [`newton_step`].
#[derive(Debug, Clone)]
pub(super) struct NewtonStep {
    /// `(-H + lambda I)^-1 g`.
    pub(super) delta: Array1<f64>,
    /// The Levenberg damping the step was solved with; 0 when `-H` is positive definite.
    pub(super) lambda: f64,
}

impl NewtonStep {
    /// The convergence test of the ordered and zero-inflated fits at the point where the
    /// step was computed (`grad` the score there, `par` the parameters, `ll` the
    /// log-likelihood, `ll_prev` the one before the last accepted step). All of:
    /// - the Newton decrement `g' delta` is at most `1e-12 (1 + |ll|)`;
    /// - the log-likelihood changed by less than `ll_tol (|ll_prev| + ll_tol)`;
    /// - every step component is at most `1e-10 (1 + |par_k|)`;
    /// - after damping (`lambda > 0`), every score component is below `1e-8 (1 + |ll|)`.
    ///
    /// The decrement and the log-likelihood change do not depend on the units of a
    /// regressor (Newton's method is affine-invariant); the largest score component
    /// does (x times s: its component times s). The step test is relative only for
    /// parameters above 1 in absolute value, so a change of units that makes a
    /// coefficient large tightens it, which can cost an iteration. A damped step shrinks
    /// with lambda, and so do its components and its decrement: away from an optimum,
    /// tiny damped steps would pass the first three tests, so a damped step also needs a
    /// small score. At a regular optimum `-H` is positive definite and the step is not
    /// damped.
    pub(super) fn converged(
        &self,
        grad: &[f64],
        par: &[f64],
        ll: f64,
        ll_prev: f64,
        ll_tol: f64,
    ) -> bool {
        let decrement: f64 = grad.iter().zip(&self.delta).map(|(g, d)| g * d).sum();
        let rel_step = self
            .delta
            .iter()
            .zip(par)
            .fold(0.0f64, |m, (d, p)| m.max(d.abs() / (1.0 + p.abs())));
        let small_score =
            self.lambda == 0.0 || grad.iter().all(|g| g.abs() < 1e-8 * (1.0 + ll.abs()));
        decrement <= 1e-12 * (1.0 + ll.abs())
            && (ll - ll_prev).abs() < ll_tol * (ll_prev.abs() + ll_tol)
            && rel_step <= 1e-10
            && small_score
    }
}

/// The Newton step `(-H + lambda I)^-1 g`, solved with the Cholesky factor of
/// `-H + lambda I`: `lambda = 0` first; when that matrix is not positive definite,
/// `lambda = 1e-8 max(1, max diag(-H))`, ten times larger on each of up to
/// [`MAX_DAMPING`] attempts (Levenberg). `None` if every attempt fails. `work` holds the
/// damped copy.
pub(super) fn newton_step(
    hess: &Array2<f64>,
    grad: &[f64],
    work: &mut Array2<f64>,
) -> Option<NewtonStep> {
    let m = grad.len();
    let max_diag = (0..m).fold(1.0f64, |acc, i| acc.max(-hess[[i, i]]));
    let mut lambda = 0.0;
    for attempt in 0..=MAX_DAMPING {
        work.zip_mut_with(hess, |w, h| *w = -h);
        for i in 0..m {
            work[[i, i]] += lambda;
        }
        if let Ok(l) = cholesky(&work.view()) {
            return Some(NewtonStep {
                delta: solve_with_factor(&l, grad),
                lambda,
            });
        }
        lambda = if attempt == 0 {
            1e-8 * max_diag
        } else {
            lambda * 10.0
        };
    }
    None
}

/// `x` with `L L' x = b`, by forward and back substitution on the lower triangle of `l`.
fn solve_with_factor(l: &Array2<f64>, b: &[f64]) -> Array1<f64> {
    let m = b.len();
    let mut x = Array1::from(b.to_vec());
    for i in 0..m {
        let s = (0..i).fold(x[i], |s, j| s - l[[i, j]] * x[j]);
        x[i] = s / l[[i, i]];
    }
    for i in (0..m).rev() {
        let s = (i + 1..m).fold(x[i], |s, j| s - l[[j, i]] * x[j]);
        x[i] = s / l[[i, i]];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// The solve with the factor is the step through the inverse.
    #[test]
    fn test_newton_step_solves_with_the_cholesky_factor() {
        // -H = [[4, 1, 0.5], [1, 3, 0.2], [0.5, 0.2, 2]] (positive definite)
        let hess = array![[-4.0, -1.0, -0.5], [-1.0, -3.0, -0.2], [-0.5, -0.2, -2.0]];
        let grad = [1.0, -2.0, 0.5];
        let mut work = Array2::zeros((3, 3));
        let step = newton_step(&hess, &grad, &mut work).unwrap();
        // positive definite: not damped
        assert_eq!(step.lambda, 0.0);
        let neg = hess.mapv(|h| -h);
        let residual = neg.dot(&step.delta) - Array1::from(grad.to_vec());
        assert!(
            residual.iter().all(|r| r.abs() < 1e-14),
            "residual {residual}"
        );
    }

    /// An indefinite -H is damped until it is positive definite, and the step says by
    /// how much.
    #[test]
    fn test_newton_step_damps_an_indefinite_information() {
        let hess = array![[-1.0, 0.0], [0.0, 1e-12]];
        let mut work = Array2::zeros((2, 2));
        let step = newton_step(&hess, &[1.0, 1.0], &mut work).unwrap();
        // lambda = 1e-8 max(1, max diag(-H)) = 1e-8, the first damping attempt
        assert_eq!(step.lambda, 1e-8);
        // the first coordinate is (almost) the Newton step, the second is damped
        let delta = &step.delta;
        assert!((delta[0] - 1.0 / (1.0 + 1e-8)).abs() < 1e-15, "{delta}");
        assert!(
            (delta[1] - 1.0 / (1e-8 - 1e-12)).abs() < 1e-6 * delta[1],
            "{delta}"
        );
    }

    /// A damped step shrinks with lambda, and so do its components and the decrement:
    /// away from an optimum, tiny damped steps pass the step, decrement and
    /// log-likelihood tests. -H = diag(1, -3e5) is damped to lambda = 1e6 (the 15th
    /// attempt); the score 5e-5 then gives steps below 1e-10 and a decrement near 6e-15,
    /// and the log-likelihood (-1000) has not changed. The score, five times
    /// 1e-8 (1 + |ll|), says the point is not an optimum.
    #[test]
    fn test_damped_step_converges_only_with_a_small_score() {
        let hess = array![[-1.0, 0.0], [0.0, 3e5]];
        let mut work = Array2::zeros((2, 2));
        let par = [0.5, -0.5];
        let ll = -1000.0;
        let grad = [5e-5, 5e-5];
        let step = newton_step(&hess, &grad, &mut work).unwrap();
        assert!(step.lambda > 3e5 && step.lambda < 1e7, "{}", step.lambda);
        assert!(step.delta.iter().all(|d| d.abs() < 1e-10), "{}", step.delta);
        assert!(!step.converged(&grad, &par, ll, ll, 1e-12));
        // the same step, undamped: the decrement, log-likelihood and step tests
        // decide, and they pass
        let undamped = NewtonStep {
            lambda: 0.0,
            ..step.clone()
        };
        assert!(undamped.converged(&grad, &par, ll, ll, 1e-12));
        // damped, with a score below 1e-8 (1 + |ll|): converged
        let small = [1e-6, 1e-6];
        let step = newton_step(&hess, &small, &mut work).unwrap();
        assert!(step.lambda > 0.0);
        assert!(step.converged(&small, &par, ll, ll, 1e-12));
        // the log-likelihood change still counts
        assert!(!step.converged(&small, &par, ll, ll - 1e-6, 1e-12));
    }
}
