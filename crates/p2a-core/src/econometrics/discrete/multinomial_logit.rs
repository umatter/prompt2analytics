//! Multinomial Logit Model.
//!
//! # Mathematical Background
//!
//! For J outcome categories, the multinomial logit model specifies:
//!
//! P(y_i = j | X_i) = exp(X_i'beta_j) / sum_k exp(X_i'beta_k)
//!
//! For identification, the reference category (j=0) has beta_0 = 0.
//!
//! # References
//!
//! - McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior.
//!
//! R equivalent: `nnet::multinom()`

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, MissingSummary, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::solve::{LM_QR_TOL, Qr};
use crate::resource;
use crate::traits::convergence::{Convergence, ResultNotes, deviance_change, notes_text};
use crate::traits::estimator::normal_p_value_two_sided;

use super::conditional_logit::{
    certainly_full_rank, check_newton_control, covariance_from_information,
    equilibrated_newton_step,
};
use super::newton::{MAX_HALVINGS, ensure_dense_inverse_fits};

/// Newton iterations of [`run_multinom`].
const MAX_ITER: usize = 100;
/// Tolerance of the Newton-decrement rule of [`run_multinom`].
const TOLERANCE: f64 = 1e-10;

/// Most outcome categories a multinomial logit accepts.
///
/// A method-level cap, the analogue of `nnet::multinom`'s `MaxNWts`, active in
/// every mode: each category adds a block of K parameters, and the Newton step
/// builds and inverts a dense ((J - 1) K)^2 Hessian, so a continuous column
/// passed as the outcome would otherwise create one category per observation.
const MAX_OUTCOME_CATEGORIES: usize = 100;

/// Category probabilities of one observation by log-sum-exp (RF-32).
///
/// `xb[j]` is the linear predictor of non-reference category `j + 1`; the reference
/// category has 0. With `m = max(0, max_j xb_j)` and
/// `s = exp(-m) + sum_j exp(xb_j - m)`: `probs[0] = exp(-m) / s` and
/// `probs[j + 1] = exp(xb_j - m) / s`. Returns `ln sum_c exp(eta_c) = m + ln s`, so
/// `ln p_c = eta_c - (return value)`. No overflow for any finite `xb`; this replaces
/// the clamp `exp(x'b).min(1e10)`, which biased the score and the likelihood once a
/// linear predictor exceeded ln(1e10) = 23.03.
fn softmax_into(xb: &[f64], probs: &mut [f64]) -> f64 {
    let m = xb.iter().fold(0.0f64, |m, &v| m.max(v));
    let e0 = (-m).exp();
    probs[0] = e0;
    let mut s = e0;
    for (p, &v) in probs[1..].iter_mut().zip(xb) {
        let e = (v - m).exp();
        *p = e;
        s += e;
    }
    for p in probs.iter_mut() {
        *p /= s;
    }
    m + s.ln()
}

/// Result from multinomial logit regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultinomResult {
    /// Dependent variable name
    pub dep_var: String,
    /// Variable names (including intercept)
    pub variables: Vec<String>,
    /// Outcome categories (reference category is first)
    pub categories: Vec<String>,
    /// Reference category
    pub reference_category: String,
    /// Coefficients for each non-reference category (J-1 sets of K coefficients)
    /// Organized as: coefficients[category_idx][variable_idx]
    pub coefficients: Vec<Vec<f64>>,
    /// Standard errors (same structure as coefficients)
    pub std_errors: Vec<Vec<f64>>,
    /// Z-statistics
    pub z_stats: Vec<Vec<f64>>,
    /// P-values (two-sided)
    pub p_values: Vec<Vec<f64>>,
    /// Log-likelihood at convergence
    pub log_likelihood: f64,
    /// Log-likelihood of null model (intercept only)
    pub log_likelihood_null: f64,
    /// McFadden's pseudo R-squared
    pub pseudo_r_squared: f64,
    /// AIC
    pub aic: f64,
    /// BIC
    pub bic: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Whether converged
    pub converged: bool,
    /// Number of observations
    pub n_obs: usize,
    /// Counts by category
    pub category_counts: Vec<usize>,
    /// Rows dropped because of a missing outcome or regressor (`nnet::multinom`: `na.omit`).
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
    /// Some fitted probability is numerically 0 or 1 (within `10 eps` of the bound): no
    /// maximum likelihood estimate exists.
    #[serde(default)]
    pub separation: bool,
    /// Convergence record of the Newton iteration.
    #[serde(default)]
    pub convergence: Convergence,
    /// Warnings (separation, convergence, singular information), without the
    /// missing-value note.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

impl ResultNotes for MultinomResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }

    fn notes_convergence(&self) -> Option<Convergence> {
        Some(self.convergence.clone())
    }
}

impl fmt::Display for MultinomResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Multinomial Logit Regression")?;
        writeln!(f, "============================")?;
        writeln!(f, "Dependent variable: {}", self.dep_var)?;
        writeln!(f, "Reference category: {}", self.reference_category)?;
        writeln!(
            f,
            "N = {}, Categories = {}",
            self.n_obs,
            self.categories.len()
        )?;
        writeln!(f)?;

        // Print coefficients for each category vs reference
        for (cat_idx, cat) in self.categories.iter().skip(1).enumerate() {
            writeln!(f, "--- {} vs {} ---", cat, self.reference_category)?;
            writeln!(
                f,
                "{:<15} {:>10} {:>10} {:>10} {:>10}",
                "Variable", "Coef", "Std.Err", "z", "P>|z|"
            )?;
            writeln!(
                f,
                "{:-<15} {:-<10} {:-<10} {:-<10} {:-<10}",
                "", "", "", "", ""
            )?;
            for (var_idx, var) in self.variables.iter().enumerate() {
                writeln!(
                    f,
                    "{:<15} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
                    var,
                    self.coefficients[cat_idx][var_idx],
                    self.std_errors[cat_idx][var_idx],
                    self.z_stats[cat_idx][var_idx],
                    self.p_values[cat_idx][var_idx]
                )?;
            }
            writeln!(f)?;
        }

        writeln!(f, "Log-Likelihood: {:.4}", self.log_likelihood)?;
        writeln!(f, "Pseudo R-squared: {:.4}", self.pseudo_r_squared)?;
        writeln!(f, "AIC: {:.4}, BIC: {:.4}", self.aic, self.bic)?;
        write!(f, "{}", notes_text(self))?;
        if let Some(note) = self.missing.note() {
            writeln!(f, "Missing values: {note}")?;
        }
        Ok(())
    }
}

/// Run multinomial logit regression.
///
/// # Arguments
///
/// * `dataset` - The dataset
/// * `y_col` - Name of the categorical dependent variable (text, integer, float or
///   Boolean labels)
/// * `x_cols` - Names of independent variables
/// * `reference` - Optional reference category (default: the first category in R
///   `factor()` order: numbers numerically, text by byte)
///
/// # Returns
///
/// `MultinomResult` containing coefficient estimates and statistics.
///
/// # Missing values
///
/// Rows with a missing outcome or regressor are dropped (as `nnet::multinom`'s
/// `na.omit`) and reported in `missing`.
///
/// # Errors
///
/// An outcome with more than 100 distinct values is refused with
/// `InvalidSpecification` (a method-level cap, like `nnet::multinom`'s `MaxNWts`).
/// Regressors that are linear combinations of the others are refused with
/// `SingularMatrix` naming them (`nnet::multinom` returns an arbitrary split).
///
/// # Convergence
///
/// Newton-Raphson on the multinomial-logit likelihood, the MLE that `nnet::multinom`
/// approximates by BFGS. The step is solved on the equilibrated information (Cholesky
/// first); the iteration stops when the Newton decrement on `glm.fit`'s
/// relative-deviance scale is below 1e-10, then takes that last full step (at most 100
/// iterations), and one more full step to polish. A line search that finds no increase
/// stops it as not converged. When some observation's fitted probability of its own
/// category is numerically 0 or 1 (separation: no MLE exists), the fit is
/// flagged (`separation`), warned and reported as not converged. A singular information
/// at the estimates gives `NaN` standard errors and a warning. Non-convergence is
/// reported in the result, never as an error.
///
/// # References
///
/// - McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior.
///
/// R equivalent: `nnet::multinom()`
pub fn run_multinom(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    reference: Option<&str>,
) -> EconResult<MultinomResult> {
    let df = dataset.df();

    // The outcome as a factor and X from the same rows: a row with a missing outcome or
    // regressor is dropped (nnet::multinom: na.omit). Categories follow R factor() order
    // (numbers numerically, text by byte).
    let cols = ColumnRequest::new(df)
        .factor(y_col)
        .numerics(x_cols)
        .na(NaPolicy::Omit)
        .extract()?;
    let y_factor = cols.factor(y_col);
    let n = cols.n();
    let mut categories: Vec<String> = y_factor.levels.clone();

    let j = categories.len(); // Number of categories
    if j < 2 {
        return Err(EconError::InvalidSpecification {
            message: "Multinomial logit requires at least 2 categories".to_string(),
        });
    }
    if j > MAX_OUTCOME_CATEGORIES {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Outcome column '{y_col}' has {j} distinct values; multinomial logit supports \
                 at most {MAX_OUTCOME_CATEGORIES} outcome categories (like nnet::multinom's \
                 MaxNWts limit). Check that '{y_col}' is a categorical outcome, not a \
                 continuous variable, or recode it into fewer categories."
            ),
        });
    }

    // Set reference category
    let ref_cat = reference
        .map(|s| s.to_string())
        .unwrap_or_else(|| categories[0].clone());
    let Some(ref_idx) = categories.iter().position(|c| c == &ref_cat) else {
        return Err(EconError::InvalidSpecification {
            message: format!("Reference category '{}' not found in data", ref_cat),
        });
    };

    // Reorder categories with reference first
    categories.swap(0, ref_idx);

    // Category index of each factor level (the levels with the reference moved first):
    // the swap exchanges levels 0 and ref_idx.
    let level_to_cat: Vec<usize> = (0..y_factor.levels.len())
        .map(|i| {
            if i == ref_idx {
                0
            } else if i == 0 {
                ref_idx
            } else {
                i
            }
        })
        .collect();

    // Category index of each observation, and the counts per category
    let mut y_idx = resource::try_vec(n, 0usize, "multinomial logit outcome")?;
    let mut category_counts = vec![0usize; j];
    for (yi, &code) in y_idx.iter_mut().zip(&y_factor.codes) {
        *yi = level_to_cat[code as usize];
        category_counts[*yi] += 1;
    }

    // Design matrix on the kept rows
    let dm = cols.matrix(x_cols, true)?;
    let x = dm.view();
    let k = x.ncols();

    // Variable names
    let mut var_names = vec!["(Intercept)".to_string()];
    var_names.extend(x_cols.iter().map(|s| s.to_string()));

    // Aliased regressors are refused (no finite, identified MLE). A design the
    // normal-equations test certifies is of full rank needs no QR.
    if !certainly_full_rank(x)? {
        let qr = Qr::decompose(x, LM_QR_TOL)?;
        let aliased = qr.aliased();
        if !aliased.is_empty() {
            let list: Vec<&str> = aliased.iter().map(|&a| var_names[a].as_str()).collect();
            return Err(EconError::SingularMatrix {
                context: format!(
                    "multinomial logit: regressors {} are linear combinations of the other columns \
                     (rank {} of {k})",
                    list.join(", "),
                    qr.rank()
                ),
                suggestion:
                    "drop the listed regressors (nnet::multinom returns an arbitrary split \
                             of their coefficients)"
                        .to_string(),
            });
        }
    }

    let n_cats = j - 1; // Non-reference categories
    let n_params = n_cats * k;
    ensure_dense_inverse_fits(n_params, "multinomial logit Hessian")?;
    check_newton_control(MAX_ITER, TOLERANCE)?;
    let model = MultinomModel {
        x,
        y_idx: &y_idx,
        n_cats,
    };

    // Newton-Raphson iteration
    let what = "multinomial logit information matrix";
    let mut beta = vec![0.0; n_params];
    let mut warnings: Vec<String> = Vec::new();
    let mut converged = false;
    let mut iterations = 0;
    let mut final_change = f64::NAN;
    // The derivatives at the current `beta` when a trial step already computed them (the
    // full step of the line search is evaluated with its derivatives, which the next
    // iteration then reuses; the values are the ones a fresh pass gives, bit for bit).
    let mut at_beta: Option<MultinomDerivatives> = None;
    while iterations < MAX_ITER {
        resource::checkpoint()?;
        let MultinomDerivatives { ll, grad, info, .. } = match at_beta.take() {
            Some(known) => known,
            None => model.derivatives(&beta)?,
        };
        let (d, lambda2) = equilibrated_newton_step(&info, &grad, what, &mut warnings)?;
        drop(info);
        let dev = -2.0 * ll;
        final_change = deviance_change(dev - lambda2, dev);
        let trial = |t: f64| -> Vec<f64> { beta.iter().zip(&d).map(|(b, s)| b + t * s).collect() };
        if final_change < TOLERANCE {
            // The last full step, then one more from the new point: with 35 parameters
            // (disc_multinom_mn8) one step leaves the iterate 5e-10 off the optimum.
            let full = trial(1.0);
            let at_full = model.derivatives(&full)?;
            if at_full.ll >= ll - 1e-12 * ll.abs() {
                beta = full;
                let (d2, _) =
                    equilibrated_newton_step(&at_full.info, &at_full.grad, what, &mut warnings)?;
                let polish: Vec<f64> = beta.iter().zip(&d2).map(|(b, s)| b + s).collect();
                let at_polish = model.derivatives(&polish)?;
                if at_polish.ll >= at_full.ll - 1e-12 * at_full.ll.abs() {
                    beta = polish;
                    at_beta = Some(at_polish);
                } else {
                    at_beta = Some(at_full);
                }
            }
            converged = true;
            break;
        }
        let mut t = 1.0;
        let mut accepted = None;
        for h in 0..=MAX_HALVINGS {
            let cand = trial(t);
            if h == 0 {
                // The full step, usually accepted: its derivatives serve the next iteration.
                let at_cand = model.derivatives(&cand)?;
                if at_cand.ll > ll {
                    accepted = Some((cand, Some(at_cand)));
                    break;
                }
            } else if model.log_likelihood(&cand)? > ll {
                accepted = Some((cand, None));
                break;
            }
            t *= 0.5;
        }
        let Some((cand, at_cand)) = accepted else {
            warnings.push(format!(
                "line search failed at iteration {}: no step along the Newton direction \
                 increases the log-likelihood; the estimates are not at the optimum",
                iterations + 1
            ));
            break;
        };
        beta = cand;
        at_beta = at_cand;
        iterations += 1;
    }

    let MultinomDerivatives {
        ll: log_likelihood,
        info: info_final,
        n_extreme,
        ..
    } = match at_beta.take() {
        Some(known) => known,
        None => model.derivatives(&beta)?,
    };

    // Separation: the fitted probability of some observation's own category numerically
    // 0 or 1 (glm.fit's rule; in the binary case the two probabilities are each other's
    // complement). A small probability of a category the observation did not choose is
    // an ordinary tail (disc_multinom_sep5). No MLE exists under separation, so the fit
    // is reported as not converged even when the decrement rule was met. The count comes
    // from the pass of the final derivatives.
    let separation = n_extreme > 0;
    if separation {
        warnings.push(format!(
            "Separation: fitted probabilities are numerically 0 or 1 for {n_extreme} of {n} \
             observations; the regressors (almost) perfectly predict '{y_col}', so the maximum \
             likelihood estimates do not exist and the coefficients and standard errors are not \
             reliable"
        ));
    }
    let separated_stop = converged && separation;
    let converged = converged
        && !separation
        && log_likelihood.is_finite()
        && beta.iter().all(|b| b.is_finite());
    let convergence = Convergence::new(
        converged,
        iterations,
        MAX_ITER,
        "Newton decrement (relative deviance change)",
        if separated_stop {
            f64::NAN
        } else {
            final_change
        },
        TOLERANCE,
    );
    if separated_stop {
        warnings.push(format!(
            "multinomial logit stopped after {iterations} iterations without a maximum \
             likelihood estimate (separation); reported as not converged"
        ));
    } else if let Some(note) = convergence.note("multinomial logit") {
        warnings.push(note);
    }

    // Null model log-likelihood
    let log_likelihood_null: f64 = category_counts
        .iter()
        .map(|&count| count as f64 * (count as f64 / n as f64).ln())
        .sum();

    // Pseudo R-squared
    let pseudo_r_squared = 1.0 - (log_likelihood / log_likelihood_null);

    // Information criteria
    let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;
    let bic = -2.0 * log_likelihood + (n_params as f64) * (n as f64).ln();

    // Standard errors from the inverse information
    let vcov = covariance_from_information(&info_final, what, &mut warnings)?;
    drop(info_final);

    // Extract coefficients, SEs, z-stats, p-values
    let mut coefficients = Vec::with_capacity(n_cats);
    let mut std_errors = Vec::with_capacity(n_cats);
    let mut z_stats = Vec::with_capacity(n_cats);
    let mut p_values = Vec::with_capacity(n_cats);

    for cat_idx in 0..n_cats {
        let mut cat_coefs = Vec::with_capacity(k);
        let mut cat_ses = Vec::with_capacity(k);
        let mut cat_zs = Vec::with_capacity(k);
        let mut cat_ps = Vec::with_capacity(k);

        for kk in 0..k {
            let idx = cat_idx * k + kk;
            let coef = beta[idx];
            let se = vcov
                .as_ref()
                .map_or(f64::NAN, |v| v[[idx, idx]].max(0.0).sqrt());
            let z = if se > 0.0 && se.is_finite() {
                coef / se
            } else {
                f64::NAN
            };
            let p = normal_p_value_two_sided(z);

            cat_coefs.push(coef);
            cat_ses.push(se);
            cat_zs.push(z);
            cat_ps.push(p);
        }

        coefficients.push(cat_coefs);
        std_errors.push(cat_ses);
        z_stats.push(cat_zs);
        p_values.push(cat_ps);
    }

    Ok(MultinomResult {
        dep_var: y_col.to_string(),
        variables: var_names,
        categories,
        reference_category: ref_cat,
        coefficients,
        std_errors,
        z_stats,
        p_values,
        log_likelihood,
        log_likelihood_null,
        pseudo_r_squared,
        aic,
        bic,
        iterations,
        converged,
        n_obs: n,
        category_counts,
        missing: cols.missing().clone(),
        separation,
        convergence,
        warnings,
    })
}

/// The data of one multinomial-logit fit: design, outcome categories (0 the reference)
/// and the number of non-reference categories. `beta` is laid out category by category
/// (`beta[c * k + v]`).
struct MultinomModel<'a> {
    x: ndarray::ArrayView2<'a, f64>,
    y_idx: &'a [usize],
    n_cats: usize,
}

impl MultinomModel<'_> {
    /// Linear predictors of row `i` into `xb`, then the probabilities into `probs`;
    /// returns the log-sum-exp.
    fn row_probs(&self, beta: &[f64], i: usize, xb: &mut [f64], probs: &mut [f64]) -> f64 {
        let k = self.x.ncols();
        let row = self.x.row(i);
        for (c, v) in xb.iter_mut().enumerate() {
            *v = row
                .iter()
                .zip(&beta[c * k..(c + 1) * k])
                .map(|(a, b)| a * b)
                .sum();
        }
        softmax_into(xb, probs)
    }

    /// The log-likelihood `sum_i (eta_{i, y_i} - logsumexp(eta_i))`.
    fn log_likelihood(&self, beta: &[f64]) -> EconResult<f64> {
        let mut xb = vec![0.0; self.n_cats];
        let mut probs = vec![0.0; self.n_cats + 1];
        let mut ll = 0.0;
        for (i, &y) in self.y_idx.iter().enumerate() {
            if i % 256 == 0 {
                resource::checkpoint()?;
            }
            let lse = self.row_probs(beta, i, &mut xb, &mut probs);
            ll += if y == 0 { 0.0 } else { xb[y - 1] } - lse;
        }
        Ok(ll)
    }

    /// Log-likelihood, score `X'(Y - P)` and information
    /// `sum_i (diag(p_i) - p_i p_i') (x) x_i x_i'` (non-reference categories), and the
    /// observations whose fitted probability of their own category is within `10 eps` of
    /// 0 or 1, in one pass.
    fn derivatives(&self, beta: &[f64]) -> EconResult<MultinomDerivatives> {
        let eps10 = 10.0 * f64::EPSILON;
        let mut n_extreme = 0;
        let k = self.x.ncols();
        let m = self.n_cats * k;
        let mut xb = vec![0.0; self.n_cats];
        let mut probs = vec![0.0; self.n_cats + 1];
        let mut ll = 0.0;
        let mut grad = Array1::<f64>::zeros(m);
        let mut info = resource::try_zeros2(m, m, "multinomial logit Hessian")?;
        // The information row by row: the same sums in the same order as indexing it,
        // without the per-element index checks.
        let Some(inf) = info.as_slice_mut() else {
            return Err(EconError::Computation(
                "multinomial logit: information matrix not in standard layout".into(),
            ));
        };
        for (i, &y) in self.y_idx.iter().enumerate() {
            if i % 256 == 0 {
                resource::checkpoint()?;
            }
            let lse = self.row_probs(beta, i, &mut xb, &mut probs);
            ll += if y == 0 { 0.0 } else { xb[y - 1] } - lse;
            let p_own = probs[y];
            n_extreme += usize::from(p_own < eps10 || p_own > 1.0 - eps10);
            let row = self.x.row(i);
            for c in 0..self.n_cats {
                let pc = probs[c + 1];
                let resid = if y == c + 1 { 1.0 } else { 0.0 } - pc;
                for (kk, &xv) in row.iter().enumerate() {
                    grad[c * k + kk] += xv * resid;
                }
                for c2 in c..self.n_cats {
                    let w = if c == c2 {
                        pc * (1.0 - pc)
                    } else {
                        -pc * probs[c2 + 1]
                    };
                    if w == 0.0 {
                        continue;
                    }
                    for kk in 0..k {
                        let wx = w * row[kk];
                        let l0 = if c == c2 { kk } else { 0 };
                        let at = (c * k + kk) * m + c2 * k;
                        for (cell, &xl) in inf[at + l0..at + k].iter_mut().zip(row.iter().skip(l0))
                        {
                            *cell += wx * xl;
                        }
                    }
                }
            }
        }
        // mirror the upper triangle
        for a in 0..m {
            for b in (a + 1)..m {
                info[[b, a]] = info[[a, b]];
            }
        }
        Ok(MultinomDerivatives {
            ll,
            grad,
            info,
            n_extreme,
        })
    }
}

/// What [`MultinomModel::derivatives`] computes at one `beta`.
struct MultinomDerivatives {
    ll: f64,
    grad: Array1<f64>,
    info: Array2<f64>,
    /// Observations whose fitted probability of their own category is within `10 eps` of
    /// 0 or 1.
    n_extreme: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::design::DesignMatrix;
    use crate::linalg::matrix_ops::safe_inverse;
    use crate::resource::{self, Budget, CancelToken};
    use p2a_validation::{Check, Tol};
    use polars::prelude::*;

    fn flat(v: &[Vec<f64>]) -> Vec<f64> {
        v.iter().flatten().copied().collect()
    }

    /// "<category>:<variable>", the names the case files use (reference category excluded).
    fn flat_names(r: &MultinomResult) -> Vec<String> {
        r.categories[1..]
            .iter()
            .flat_map(|c| r.variables.iter().map(move |v| format!("{c}:{v}")))
            .collect()
    }

    /// Every number run_multinom reports, against the tight nnet::multinom fit of `case`.
    fn pin_multinom(case: &p2a_validation::Case, r: &MultinomResult) {
        let n = r.n_obs;
        let names = flat_names(r);
        let counts: Vec<f64> = r.category_counts.iter().map(|&c| c as f64).collect();
        Check::new(case)
            .scalar("n", n as f64, Tol::exact())
            .named("counts", &r.categories, &counts, Tol::exact())
            .named("coef_tight", &names, &flat(&r.coefficients), Tol::coef(n))
            .named("se_tight", &names, &flat(&r.std_errors), Tol::se(n))
            .named("z_tight", &names, &flat(&r.z_stats), Tol::stat(n))
            .named("p_tight", &names, &flat(&r.p_values), Tol::p_value(n))
            .scalar("loglik", r.log_likelihood, Tol::stat(n))
            .scalar("null_loglik", r.log_likelihood_null, Tol::stat(n))
            .scalar("aic", r.aic, Tol::stat(n))
            .scalar("bic", r.bic, Tol::stat(n))
            .finish();
        // Not pinned: loglik_pkg (nnet at its own optimum; equals loglik to 1e-12),
        // max_abs_xb (a property of the case), start_hessian_*
        // (test_validate_multinom_start_hessian_mn8).
        assert!(r.converged && r.iterations < 50, "{r}");
    }

    /// RF-31: an ordinary 8-category fit (35 parameters). Its Hessian is negative
    /// definite, so the old Newton step went through faer 0.22's full-pivot LU, which
    /// inverted it wrongly (coefficients of 2e11, log-likelihood -inf).
    #[test]
    fn test_validate_multinom_mn8() {
        let (case, ds) = crate::test_support::load_case("disc_multinom_mn8");
        let r = run_multinom(&ds, "y", &["x1", "x2", "x3", "x4"], None).unwrap();
        pin_multinom(&case, &r);
    }

    /// The first matrix run_multinom's Newton loop inverted on disc_multinom_mn8 (the
    /// Hessian at beta = 0, built in the same summation order): well conditioned, not
    /// positive definite, and inverted to rounding by safe_inverse's LU fallback.
    #[test]
    fn test_validate_multinom_start_hessian_mn8() {
        use crate::linalg::matrix_ops::{cholesky, condition_number, eig_symmetric};
        let (case, ds) = crate::test_support::load_case("disc_multinom_mn8");
        let dm = DesignMatrix::from_dataframe(ds.df(), &["x1", "x2", "x3", "x4"], true).unwrap();
        let x = dm.view();
        let (n, k) = x.dim();
        let n_cats = case.names("counts").len() - 1;
        let p = 1.0 / (n_cats + 1) as f64; // every probability at beta = 0
        let mut h = Array2::<f64>::zeros((n_cats * k, n_cats * k));
        for i in 0..n {
            for c in 0..n_cats {
                for c2 in 0..n_cats {
                    let w = if c == c2 { p * (1.0 - p) } else { -p * p };
                    for kk in 0..k {
                        for ll in 0..k {
                            h[[c * k + kk, c2 * k + ll]] -= w * x[[i, kk]] * x[[i, ll]];
                        }
                    }
                }
            }
        }
        assert!(
            cholesky(&h.view()).is_err(),
            "the start Hessian must not be PD"
        );
        let kappa = condition_number(&h.view()).unwrap();
        let (eigen, _) = eig_symmetric(&h.view()).unwrap();
        let max_eigen = eigen.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Check::new(&case)
            .scalar("start_hessian_kappa", kappa, Tol::stat(n))
            .scalar("start_hessian_max_eigen", max_eigen, Tol::stat(n))
            .finish();
        let (inv, w) = safe_inverse(&h.view()).unwrap();
        assert_eq!(w, None);
        let prod = h.dot(&inv);
        let mut r = 0.0f64;
        for (idx, v) in prod.indexed_iter() {
            let d = (v - if idx.0 == idx.1 { 1.0 } else { 0.0 }).abs();
            r = if d.is_nan() { f64::INFINITY } else { r.max(d) };
        }
        assert!(r <= 1e-12, "max |H X - I| = {r}");
    }

    /// Control: 4 overlapping categories, 9 parameters.
    #[test]
    fn test_validate_multinom_j4() {
        let (case, ds) = crate::test_support::load_case("disc_multinom_j4");
        let r = run_multinom(&ds, "y", &["x1", "x2"], None).unwrap();
        pin_multinom(&case, &r);
    }

    /// RF-32: 5 well-separated categories; at the MLE a linear predictor exceeds
    /// ln(1e10) = 23.03, where the old `exp(x'b).min(1e10)` clamp stopped the fit at
    /// a log-likelihood above the true maximum.
    #[test]
    fn test_validate_multinom_sep5() {
        let (case, ds) = crate::test_support::load_case("disc_multinom_sep5");
        let r = run_multinom(&ds, "y", &["x"], None).unwrap();
        pin_multinom(&case, &r);
    }

    /// Every coefficient and SE of `a` equals that of `b` (same categories in the same
    /// order) to rounding.
    fn assert_same_fit(a: &MultinomResult, b: &MultinomResult) {
        let tol = Tol::new(1e-12, 1e-14);
        let (ca, cb) = (flat(&a.coefficients), flat(&b.coefficients));
        let (sa, sb) = (flat(&a.std_errors), flat(&b.std_errors));
        assert_eq!(ca.len(), cb.len());
        for (i, (x, y)) in ca.iter().zip(&cb).chain(sa.iter().zip(&sb)).enumerate() {
            assert!(tol.allows(*x, *y), "[{i}] {x} vs {y}");
        }
    }

    /// A missing outcome is dropped (nnet::multinom's na.omit), not a phantom "" category.
    #[test]
    fn test_validate_multinom_na_outcome_dropped() {
        let (case, ds) = crate::test_support::load_case("disc_choice_na");
        let r = run_multinom(&ds, "g", &["x", "w"], None).unwrap();
        assert_eq!(r.categories, ["A", "B", "C"]);
        assert_eq!(r.reference_category, "A");
        assert_eq!(r.n_obs as f64, case.scalar("multinom_n"));
        let counts: Vec<f64> = r.category_counts.iter().map(|&c| c as f64).collect();
        assert_eq!(counts, case.vector("multinom_counts"));
        assert_eq!(r.missing.n_dropped, 2);
        assert_eq!(r.missing.by_column, vec![("g".to_string(), 2)]);
        let complete = Dataset::new(ds.df().drop_nulls::<String>(Some(&["g".into()])).unwrap());
        let cc = run_multinom(&complete, "g", &["x", "w"], None).unwrap();
        assert_same_fit(&r, &cc);
        assert_eq!(
            r.to_string().lines().last().unwrap(),
            "Missing values: 2 of 300 rows dropped because of missing values (g: 2)"
        );
    }

    /// Integer outcome labels are ordered numerically, as R's factor() (2 < 10 < 33).
    #[test]
    fn test_validate_multinom_integer_outcome_level_order() {
        let (case, ds) = crate::test_support::load_case("disc_choice_na");
        let r = run_multinom(&ds, "k", &["x", "w"], None).unwrap();
        let levels: Vec<String> = case
            .vector("k_levels")
            .iter()
            .map(|v| format!("{v}"))
            .collect();
        assert_eq!(r.categories, levels);
        assert_eq!(r.categories, ["2", "10", "33"]);
        assert_eq!(r.reference_category, "2");
        let g = run_multinom(&ds, "g", &["x", "w"], None).unwrap();
        assert_same_fit(&r, &g);
    }

    #[test]
    fn test_softmax_into_log_sum_exp() {
        // (1) Ordinary predictors.
        let mut p = [0.0; 3];
        let lse = softmax_into(&[0.3, -1.2], &mut p);
        let e = [1.0, 0.3f64.exp(), (-1.2f64).exp()];
        let s: f64 = e.iter().sum();
        for (pc, ec) in p.iter().zip(e) {
            assert!((pc - ec / s).abs() <= 1e-15, "{p:?}");
        }
        assert!((p.iter().sum::<f64>() - 1.0).abs() <= 1e-15, "{p:?}");
        assert!((lse - s.ln()).abs() <= 1e-15, "{lse}");
        // (2) exp(800) overflows; log-sum-exp does not.
        let lse = softmax_into(&[800.0, 799.0], &mut p);
        let want = [0.0, 0.7310585786300049, 0.2689414213699951];
        for (pc, wc) in p.iter().zip(want) {
            assert!((pc - wc).abs() <= 1e-15, "{p:?}");
        }
        assert!((lse - 800.3132616875182).abs() <= 1e-12, "{lse}");
        // (3) Above ln(1e10): the clamp gave p[1] = 1e10 / (1 + 1e10) = 1 - 1e-10.
        let mut p = [0.0; 2];
        softmax_into(&[30.0], &mut p);
        assert!((p[1] - 0.9999999999999065).abs() <= 1e-16, "{p:?}");
        assert!((p[0] - 9.3576229688393e-14).abs() <= 1e-27, "{p:?}");
    }

    fn create_multinomial_dataset() -> Dataset {
        let df = df! {
            "y" => ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C", "A", "B"],
            "x" => [1.0, 2.0, 1.5, 4.0, 5.0, 4.5, 7.0, 8.0, 9.0, 8.5, 2.5, 5.5]
        }
        .unwrap();
        Dataset::new(df)
    }

    fn cancelled() -> Budget {
        let t = CancelToken::new();
        t.cancel();
        Budget::new(None, t)
    }

    /// A continuous outcome with `n` distinct values.
    fn continuous_y_dataset(n: usize) -> Dataset {
        let y: Vec<f64> = (0..n).map(|i| i as f64 + 0.5).collect();
        let x: Vec<f64> = (0..n)
            .map(|i| ((i * 7) % 13) as f64 + 0.1 * i as f64)
            .collect();
        Dataset::new(df! { "y" => y, "x" => x }.unwrap())
    }

    #[test]
    fn test_multinom_refuses_more_than_100_categories() {
        let err = run_multinom(&continuous_y_dataset(101), "y", &["x"], None).unwrap_err();
        match err {
            EconError::InvalidSpecification { message } => {
                assert!(
                    message.contains("'y'") && message.contains("101") && message.contains("100"),
                    "{message}"
                )
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    /// The category cap admits 100 categories and refuses 101. The cap is checked
    /// before the first budget checkpoint, so under a cancelled budget 100
    /// categories get past it to the Newton loop (`Cancelled`), and 101 do not.
    /// A full fit of 100 one-row categories runs 35 Newton iterations on 198
    /// parameters (32.7 s in an opt-level 0 build, as CI's), so its behaviour is
    /// checked on 20 categories in
    /// test_multinom_one_row_per_category_does_not_converge.
    #[test]
    fn test_multinom_accepts_100_categories() {
        let run = |n| {
            resource::with_budget(cancelled(), || {
                run_multinom(&continuous_y_dataset(n), "y", &["x"], None)
            })
            .unwrap_err()
        };
        let e = run(100);
        assert!(
            matches!(e, EconError::Cancelled(_)),
            "100 categories: {e:?}"
        );
        let e = run(101);
        assert!(
            matches!(e, EconError::InvalidSpecification { .. }),
            "101 categories: {e:?}"
        );
    }

    /// One row per category is complete separation, so no MLE exists. Before Tier 1
    /// the 100-category fit was all NaN and reported `Converged: true (1 iterations)`
    /// (RF-31). The estimates must be finite and `converged` false: the iteration
    /// runs out of steps or stops at the first step that comes from the
    /// pseudoinverse (singular information), never applying it. Measured on 20
    /// categories (38 parameters): with pseudoinverse steps applied, the fit
    /// reported `converged = true` with a log-likelihood of -1.5e18.
    #[test]
    fn test_multinom_one_row_per_category_does_not_converge() {
        let r = run_multinom(&continuous_y_dataset(20), "y", &["x"], None).unwrap();
        assert_eq!(r.categories.len(), 20);
        assert!(
            flat(&r.coefficients).iter().all(|b| b.is_finite()) && r.log_likelihood.is_finite(),
            "non-finite estimates: LL {}, converged {} after {} iterations",
            r.log_likelihood,
            r.converged,
            r.iterations
        );
        assert!(
            !r.converged,
            "no MLE exists, yet converged after {} iterations (LL {}, null {})",
            r.iterations, r.log_likelihood, r.log_likelihood_null
        );
    }

    #[test]
    fn test_multinom_stops_when_cancelled() {
        let ds = create_multinomial_dataset();
        let e = resource::with_budget(cancelled(), || run_multinom(&ds, "y", &["x"], None))
            .unwrap_err();
        assert!(matches!(e, EconError::Cancelled(_)), "{e:?}");
    }

    #[test]
    fn test_multinom_hessian_respects_alloc_ceiling() {
        // n_params = 2 * 2 = 4; working set 4*4*6*8 = 768 bytes > 512
        let ds = create_multinomial_dataset();
        let e = resource::with_alloc_ceiling(Some(512), || run_multinom(&ds, "y", &["x"], None))
            .unwrap_err();
        assert!(matches!(e, EconError::ResourceLimit(_)), "{e:?}");
    }

    #[test]
    fn test_multinom_basic() {
        let dataset = create_multinomial_dataset();
        let result = run_multinom(&dataset, "y", &["x"], None).unwrap();

        assert_eq!(result.n_obs, 12);
        assert_eq!(result.categories.len(), 3);
        assert_eq!(result.coefficients.len(), 2);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_multinom_with_reference() {
        let dataset = create_multinomial_dataset();
        let result = run_multinom(&dataset, "y", &["x"], Some("B")).unwrap();

        assert_eq!(result.reference_category, "B");
    }

    // ==========================================================================
    // R Validation Tests
    // ==========================================================================

    #[test]
    fn test_multinom_four_categories_shapes() {
        // R reference with 4 categories
        // R: multinom(y ~ x) with 4 categories

        let n = 400;
        let mut y_vec: Vec<&str> = Vec::with_capacity(n);
        let mut x_vec: Vec<f64> = Vec::with_capacity(n);

        for i in 0..n {
            let x: f64 = (i as f64 / 100.0) - 2.0;
            x_vec.push(x);

            // Ensure all 4 categories are represented
            // Use deterministic assignment based on x ranges
            let cat = match i % 4 {
                0 => "1",
                1 => "2",
                2 => "3",
                _ => "4",
            };
            y_vec.push(cat);
        }

        let df = df! {
            "y" => y_vec,
            "x" => x_vec
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_multinom(&dataset, "y", &["x"], None).unwrap();

        // Structure checks
        assert_eq!(result.n_obs, n);
        assert_eq!(result.categories.len(), 4);
        assert_eq!(result.coefficients.len(), 3); // 3 non-reference categories

        // Each category has 2 coefficients (intercept, x)
        for coef_vec in &result.coefficients {
            assert_eq!(coef_vec.len(), 2);
        }

        // Log-likelihood should be finite
        assert!(result.log_likelihood.is_finite());
        assert!(result.converged && !result.separation, "{result}");
    }

    #[test]
    fn test_multinom_overlapping_thresholds_converge() {
        // Test on data with moderate separation
        let n = 300;
        let mut y_vec: Vec<&str> = Vec::with_capacity(n);
        let mut x_vec: Vec<f64> = Vec::with_capacity(n);

        for i in 0..n {
            let x: f64 = (i as f64 / 75.0) - 2.0;
            x_vec.push(x);

            // Moderate separation with some overlap
            let noise = (i as f64 * 0.3).sin() * 0.5;
            let cat = if x + noise < -0.5 {
                "Low"
            } else if x + noise < 0.8 {
                "Medium"
            } else {
                "High"
            };
            y_vec.push(cat);
        }

        let df = df! {
            "y" => y_vec,
            "x" => x_vec
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_multinom(&dataset, "y", &["x"], None).unwrap();

        // Should produce valid results (may or may not fully converge)
        assert!(result.iterations > 0, "Should have iterated");
        assert!(
            result.log_likelihood.is_finite(),
            "Log-likelihood should be finite"
        );

        // Pseudo R-squared should be positive for data with some signal
        assert!(
            result.pseudo_r_squared > 0.0,
            "Pseudo R-squared should be positive: {}",
            result.pseudo_r_squared
        );
        assert!(result.converged && !result.separation, "{result}");
    }

    // ==========================================================================
    // nnet::multinom on overlapping data (disc_multinom_overlap)
    // ==========================================================================

    fn flat3(r: &MultinomResult) -> (Vec<String>, Vec<f64>, Vec<f64>) {
        (flat_names(r), flat(&r.coefficients), flat(&r.std_errors))
    }

    /// `to = from * scale` as a Float64 column.
    fn with_scaled(ds: &Dataset, from: &str, to: &str, scale: f64) -> Dataset {
        crate::regression::ols::na_fixtures::with_derived(ds, to, from, |v| v * scale)
    }

    /// Newton-polished nnet::multinom(y ~ x1 + x2), reference A.
    #[test]
    fn test_validate_multinom_overlap() {
        let (case, ds) = crate::test_support::load_case("disc_multinom_overlap");
        let r = run_multinom(&ds, "y", &["x1", "x2"], None).unwrap();
        let (names, coef, se) = flat3(&r);
        let counts: Vec<f64> = r.category_counts.iter().map(|&c| c as f64).collect();
        Check::new(&case)
            .scalar("n", r.n_obs as f64, Tol::exact())
            .named("counts", &r.categories, &counts, Tol::exact())
            .named("coef_tight", &names, &coef, Tol::iterative(1e-8))
            .named("se_tight", &names, &se, Tol::iterative(1e-6))
            .scalar("loglik", r.log_likelihood, Tol::stat(300))
            .scalar("aic", r.aic, Tol::stat(300))
            .scalar("null_loglik", r.log_likelihood_null, Tol::stat(300))
            // nnet at its defaults (8.3e-6 off the MLE)
            .named("coef_pkg", &names, &coef, Tol::iterative(1e-4))
            .finish();
        assert_eq!(r.reference_category, "A");
        assert!(r.converged && !r.separation && r.warnings.is_empty(), "{r}");
        assert!(r.to_string().contains("Converged: yes ("), "{r}");
    }

    /// Newton's method is affine invariant: rescaling x1 by 1e-6 or 1e6 changes the
    /// iteration count by at most one and the coefficients only by the scale.
    #[test]
    fn test_validate_multinom_convergence_is_scale_free() {
        let (case, ds) = crate::test_support::load_case("disc_multinom_overlap");
        let base = run_multinom(&ds, "y", &["x1", "x2"], None).unwrap();
        for scale in [1e-6, 1e6] {
            let s = with_scaled(&ds, "x1", "x1s", scale);
            let r = run_multinom(&s, "y", &["x1s", "x2"], None).unwrap();
            assert!(r.converged, "scale {scale}: {:?}", r.warnings);
            assert!(
                r.iterations.abs_diff(base.iterations) <= 1,
                "scale {scale}: {} vs {}",
                r.iterations,
                base.iterations
            );
            let (names, coef, _) = flat3(&r);
            let names: Vec<String> = names.iter().map(|n| n.replace(":x1s", ":x1")).collect();
            let back: Vec<f64> = names
                .iter()
                .zip(&coef)
                .map(|(n, c)| if n.ends_with(":x1") { c * scale } else { *c })
                .collect();
            Check::new(&case)
                .named("coef_tight", &names, &back, Tol::iterative(1e-8))
                .finish();
        }
    }

    /// x1b = 2 x1: an error naming it (nnet returns an arbitrary split).
    #[test]
    fn test_validate_multinom_aliased_regressor_is_an_error() {
        let (case, ds) = crate::test_support::load_case("disc_multinom_overlap");
        assert_eq!(case.scalar("n"), 300.0);
        let s = with_scaled(&ds, "x1", "x1b", 2.0);
        match run_multinom(&s, "y", &["x1", "x1b", "x2"], None) {
            Err(EconError::SingularMatrix { context, .. }) => {
                assert!(context.contains("x1b"), "{context}")
            }
            other => panic!("expected SingularMatrix, got {other:?}"),
        }
    }

    /// A < B < C in x: complete separation is flagged and warned.
    #[test]
    fn test_multinom_complete_separation_warns() {
        let r = run_multinom(&create_multinomial_dataset(), "y", &["x"], None).unwrap();
        assert!(r.separation);
        assert!(!r.converged);
        assert!(
            r.warnings.iter().any(|w| w.starts_with("Separation:")),
            "{:?}",
            r.warnings
        );
    }
}
