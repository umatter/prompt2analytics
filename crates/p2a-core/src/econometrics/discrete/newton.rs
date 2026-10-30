//! The damped Newton step of the ordered (`ordered_models`) and zero-inflated
//! (`count_models`) maximum-likelihood fits, and the checks around it.

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

/// The Newton step `(-H + lambda I)^-1 g`, solved with the Cholesky factor of
/// `-H + lambda I`: `lambda = 0` first; when that matrix is not positive definite,
/// `lambda = 1e-8 max(1, max diag(-H))`, ten times larger on each of up to
/// [`MAX_DAMPING`] attempts (Levenberg). `None` if every attempt fails. `work` holds the
/// damped copy.
pub(super) fn newton_step(
    hess: &Array2<f64>,
    grad: &[f64],
    work: &mut Array2<f64>,
) -> Option<Array1<f64>> {
    let m = grad.len();
    let max_diag = (0..m).fold(1.0f64, |acc, i| acc.max(-hess[[i, i]]));
    let mut lambda = 0.0;
    for attempt in 0..=MAX_DAMPING {
        work.zip_mut_with(hess, |w, h| *w = -h);
        for i in 0..m {
            work[[i, i]] += lambda;
        }
        if let Ok(l) = cholesky(&work.view()) {
            return Some(solve_with_factor(&l, grad));
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
        let neg = hess.mapv(|h| -h);
        let residual = neg.dot(&step) - Array1::from(grad.to_vec());
        assert!(
            residual.iter().all(|r| r.abs() < 1e-14),
            "residual {residual}"
        );
    }

    /// An indefinite -H is damped until it is positive definite.
    #[test]
    fn test_newton_step_damps_an_indefinite_information() {
        let hess = array![[-1.0, 0.0], [0.0, 1e-12]];
        let mut work = Array2::zeros((2, 2));
        let step = newton_step(&hess, &[1.0, 1.0], &mut work).unwrap();
        // lambda = 1e-8: the first coordinate is (almost) the Newton step, the second
        // is damped
        assert!((step[0] - 1.0 / (1.0 + 1e-8)).abs() < 1e-15, "{step}");
        assert!(
            (step[1] - 1.0 / (1e-8 - 1e-12)).abs() < 1e-6 * step[1],
            "{step}"
        );
    }
}
