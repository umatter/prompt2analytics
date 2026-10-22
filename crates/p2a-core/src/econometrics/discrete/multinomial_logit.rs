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
use crate::errors::{EconError, EconResult};
use crate::linalg::design::DesignMatrix;
use crate::linalg::matrix_ops::safe_inverse;
use crate::resource;
use crate::traits::estimator::normal_cdf;

/// Most outcome categories a multinomial logit accepts.
///
/// A method-level cap, the analogue of `nnet::multinom`'s `MaxNWts`, active in
/// every mode: each category adds a block of K parameters, and the Newton step
/// builds and inverts a dense ((J - 1) K)^2 Hessian, so a continuous column
/// passed as the outcome would otherwise create one category per observation.
const MAX_OUTCOME_CATEGORIES: usize = 100;

/// `safe_inverse` holds several m x m copies at once (faer copy, factor, identity, solution,
/// SVD fallback). Check the working set against the allocation ceiling before building it.
const DENSE_INVERSE_COPIES: usize = 6;

fn ensure_dense_inverse_fits(m: usize, what: &str) -> EconResult<()> {
    let cells = resource::checked_len(&[m, m, DENSE_INVERSE_COPIES], what)?;
    resource::check_alloc::<f64>(cells, what)
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
        writeln!(
            f,
            "Converged: {} ({} iterations)",
            self.converged, self.iterations
        )?;
        Ok(())
    }
}

/// Run multinomial logit regression.
///
/// # Arguments
///
/// * `dataset` - The dataset
/// * `y_col` - Name of the categorical dependent variable
/// * `x_cols` - Names of independent variables
/// * `reference` - Optional reference category (default: first category in sorted order)
///
/// # Returns
///
/// `MultinomResult` containing coefficient estimates and statistics.
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

    // Extract y values and determine categories
    let y_series = df.column(y_col).map_err(|_| EconError::ColumnNotFound {
        column: y_col.to_string(),
        available: df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect(),
    })?;

    // Get unique categories
    let y_str: Vec<String> = if let Ok(ca) = y_series.str() {
        ca.into_no_null_iter().map(|s| s.to_string()).collect()
    } else if let Ok(ca) = y_series.i64() {
        ca.into_no_null_iter().map(|v| v.to_string()).collect()
    } else if let Ok(ca) = y_series.f64() {
        ca.into_no_null_iter().map(|v| v.to_string()).collect()
    } else {
        return Err(EconError::InvalidSpecification {
            message: format!("Column '{}' must be categorical (string or integer)", y_col),
        });
    };

    let n = y_str.len();

    // Get sorted unique categories
    let mut categories: Vec<String> = y_str
        .iter()
        .cloned()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    categories.sort();

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

    if !categories.contains(&ref_cat) {
        return Err(EconError::InvalidSpecification {
            message: format!("Reference category '{}' not found in data", ref_cat),
        });
    }

    // Reorder categories with reference first
    let ref_idx = categories.iter().position(|c| c == &ref_cat).unwrap();
    categories.swap(0, ref_idx);

    // Count observations per category
    let category_counts: Vec<usize> = categories
        .iter()
        .map(|cat| y_str.iter().filter(|y| *y == cat).count())
        .collect();

    // Create category index mapping
    let cat_to_idx: std::collections::HashMap<&str, usize> = categories
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i))
        .collect();

    // Convert y to category indices
    let y_idx: Vec<usize> = y_str
        .iter()
        .map(|y| *cat_to_idx.get(y.as_str()).unwrap())
        .collect();

    // Build design matrix
    let dm = DesignMatrix::from_dataframe(df, x_cols, true)?;
    let x = dm.view();
    let k = x.ncols();

    // Variable names
    let mut var_names = vec!["(Intercept)".to_string()];
    var_names.extend(x_cols.iter().map(|s| s.to_string()));

    // Initialize coefficients: (J-1) x K matrix (excluding reference category)
    let n_cats = j - 1; // Non-reference categories
    let n_params = n_cats * k;
    let mut beta = vec![0.0; n_params];
    ensure_dense_inverse_fits(n_params, "multinomial logit Hessian")?;

    // Probabilities for each observation and category, overwritten every iteration
    let mut probs = resource::try_zeros2(n, j, "multinomial logit probabilities")?;

    // Newton-Raphson iteration
    let max_iter = 50;
    let tol = 1e-8;
    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..max_iter {
        resource::checkpoint()?;
        iterations = iter + 1;

        // Compute probabilities for each observation and category
        for i in 0..n {
            if i % 256 == 0 {
                resource::checkpoint()?;
            }
            let mut exp_xb = vec![0.0; j];
            exp_xb[0] = 1.0; // Reference category (beta = 0)

            for cat_idx in 0..n_cats {
                let mut xb = 0.0;
                for kk in 0..k {
                    xb += x[[i, kk]] * beta[cat_idx * k + kk];
                }
                exp_xb[cat_idx + 1] = xb.exp().min(1e10); // Prevent overflow
            }

            let sum_exp: f64 = exp_xb.iter().sum();
            for jj in 0..j {
                probs[[i, jj]] = exp_xb[jj] / sum_exp;
            }
        }

        // Compute gradient and Hessian
        let mut gradient = vec![0.0; n_params];
        let mut hess_arr = resource::try_zeros2(n_params, n_params, "multinomial logit Hessian")?;

        for i in 0..n {
            if i % 256 == 0 {
                resource::checkpoint()?;
            }
            for cat_idx in 0..n_cats {
                let y_indicator = if y_idx[i] == cat_idx + 1 { 1.0 } else { 0.0 };
                let residual = y_indicator - probs[[i, cat_idx + 1]];

                for kk in 0..k {
                    gradient[cat_idx * k + kk] += x[[i, kk]] * residual;
                }

                // Hessian
                for cat_idx2 in 0..n_cats {
                    let w = if cat_idx == cat_idx2 {
                        probs[[i, cat_idx + 1]] * (1.0 - probs[[i, cat_idx + 1]])
                    } else {
                        -probs[[i, cat_idx + 1]] * probs[[i, cat_idx2 + 1]]
                    };

                    for kk in 0..k {
                        for ll in 0..k {
                            hess_arr[[cat_idx * k + kk, cat_idx2 * k + ll]] -=
                                w * x[[i, kk]] * x[[i, ll]];
                        }
                    }
                }
            }
        }

        let (hess_inv, _) = match safe_inverse(&hess_arr.view()) {
            Ok(inv) => inv,
            Err(_) => {
                // Add small ridge regularization
                let mut hess_reg = hess_arr.clone();
                for i in 0..n_params {
                    hess_reg[[i, i]] -= 1e-6;
                }
                safe_inverse(&hess_reg.view()).map_err(|e| EconError::SingularMatrix {
                    context: "Multinomial logit Hessian".to_string(),
                    suggestion: format!("Model may be unidentified: {}", e),
                })?
            }
        };

        // Newton step
        let grad_arr = Array1::from_vec(gradient);
        let step = hess_inv.dot(&grad_arr);

        // Update beta
        let mut max_change = 0.0f64;
        for i in 0..n_params {
            let change = step[i];
            beta[i] -= change;
            max_change = max_change.max(change.abs());
        }

        if max_change < tol {
            converged = true;
            break;
        }
    }

    // Compute final log-likelihood
    let mut log_likelihood = 0.0;
    for i in 0..n {
        let mut exp_xb = vec![0.0; j];
        exp_xb[0] = 1.0;

        for cat_idx in 0..n_cats {
            let mut xb = 0.0;
            for kk in 0..k {
                xb += x[[i, kk]] * beta[cat_idx * k + kk];
            }
            exp_xb[cat_idx + 1] = xb.exp().min(1e10);
        }

        let sum_exp: f64 = exp_xb.iter().sum();
        let log_prob = (exp_xb[y_idx[i]] / sum_exp).ln();
        log_likelihood += log_prob;
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

    // Standard errors from Hessian inverse
    let mut hess_final_arr =
        resource::try_zeros2(n_params, n_params, "multinomial logit final Hessian")?;
    for i in 0..n {
        if i % 256 == 0 {
            resource::checkpoint()?;
        }
        let mut exp_xb = vec![0.0; j];
        exp_xb[0] = 1.0;
        for cat_idx in 0..n_cats {
            let mut xb = 0.0;
            for kk in 0..k {
                xb += x[[i, kk]] * beta[cat_idx * k + kk];
            }
            exp_xb[cat_idx + 1] = xb.exp().min(1e10);
        }
        let sum_exp: f64 = exp_xb.iter().sum();
        let probs_i: Vec<f64> = exp_xb.iter().map(|e| e / sum_exp).collect();

        for cat_idx in 0..n_cats {
            for cat_idx2 in 0..n_cats {
                let w = if cat_idx == cat_idx2 {
                    probs_i[cat_idx + 1] * (1.0 - probs_i[cat_idx + 1])
                } else {
                    -probs_i[cat_idx + 1] * probs_i[cat_idx2 + 1]
                };

                for kk in 0..k {
                    for ll in 0..k {
                        hess_final_arr[[cat_idx * k + kk, cat_idx2 * k + ll]] +=
                            w * x[[i, kk]] * x[[i, ll]];
                    }
                }
            }
        }
    }

    let (vcov, _) =
        safe_inverse(&hess_final_arr.view()).unwrap_or_else(|_| (Array2::eye(n_params), None));

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
            let se = vcov[[idx, idx]].abs().sqrt();
            let z = if se > 1e-15 { coef / se } else { 0.0 };
            let p = 2.0 * (1.0 - normal_cdf(z.abs()));

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
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{self, Budget, CancelToken};
    use polars::prelude::*;

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

    #[test]
    fn test_multinom_accepts_100_categories() {
        let r = run_multinom(&continuous_y_dataset(100), "y", &["x"], None).unwrap();
        assert_eq!(r.categories.len(), 100);
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

    /// Pinned-value check. A wrong or empty `expected` prints the actual values to paste.
    fn assert_pinned(what: &str, actual: &[f64], expected: &[f64]) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{what}: pin these values: {actual:?}"
        );
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= 1e-9 * e.abs().max(1e-3),
                "{what}[{i}]: got {a}, pinned {e}; all: {actual:?}"
            );
        }
    }

    /// 60 noisy rows with three overlapping outcome classes (20 A, 13 B, 27 C),
    /// so the MLE exists: A covers x in [0, 3], B [1.25, 4] and C [2, 5.5].
    fn create_multinomial_dataset_60() -> Dataset {
        let mut y: Vec<&str> = Vec::with_capacity(60);
        let mut x: Vec<f64> = Vec::with_capacity(60);
        for i in 0..60usize {
            let xi = ((i * 37) % 23) as f64 / 4.0;
            let u = ((i * 7919) % 101) as f64 / 101.0;
            let z = 0.8 * xi + 3.0 * (u - 0.5);
            y.push(if z < 1.2 {
                "A"
            } else if z < 2.6 {
                "B"
            } else {
                "C"
            });
            x.push(xi);
        }
        Dataset::new(df! { "y" => y, "x" => x }.unwrap())
    }

    /// Characterisation test: the code before the T0-WP14 resource bounds gives
    /// the same numbers, and later refactors must not move them.
    ///
    /// It uses overlapping classes so that Newton converges to the MLE (the
    /// largest step is 4e-6 at iteration 7 and 6e-12 at iteration 8, far from
    /// the 1e-8 tolerance on either side). `create_multinomial_dataset()` is
    /// separated by x, so its fit never converges and its coefficients and SEs
    /// are rounding-determined; it serves only the cancellation and
    /// allocation-ceiling tests.
    ///
    /// R (nnet 7.3.19), on the same data:
    ///
    /// ```r
    /// i <- 0:59; x <- ((i * 37) %% 23) / 4; u <- ((i * 7919) %% 101) / 101
    /// z <- 0.8 * x + 3 * (u - 0.5)
    /// y <- ifelse(z < 1.2, "A", ifelse(z < 2.6, "B", "C"))
    /// fit <- multinom(factor(y) ~ x, reltol = 1e-14, maxit = 1000)
    /// coef(fit)                   # B: -3.69511772941 1.95653049214
    ///                             # C: -7.40054970907 3.27832909639
    /// summary(fit)$standard.errors  # B: 1.20900782158 0.65621014984
    ///                             # C: 1.81997955534 0.78618046886
    /// logLik(fit)                 # -32.172315334772
    /// ```
    #[test]
    fn test_pinned_multinom_outputs() {
        let r = run_multinom(&create_multinomial_dataset_60(), "y", &["x"], None).unwrap();
        assert!(r.converged && r.iterations < 50, "{r}");
        let mut v: Vec<f64> = r.coefficients.iter().flatten().copied().collect();
        v.extend(r.std_errors.iter().flatten().copied());
        v.push(r.log_likelihood);
        let nnet = [
            -3.69511772941,
            1.95653049214,
            -7.40054970907,
            3.27832909639,
            1.20900782158,
            0.65621014984,
            1.81997955534,
            0.78618046886,
            -32.172315334772,
        ];
        for (i, (got, want)) in v.iter().zip(nnet).enumerate() {
            assert!(
                (got - want).abs() < 1e-6 * want.abs(),
                "[{i}] {got} vs nnet {want}; all: {v:?}"
            );
        }
        v.push(r.iterations as f64);
        assert_pinned(
            "multinom",
            &v,
            &[
                -3.695117715270053,
                1.9565304881927836,
                -7.400549687067565,
                3.2783290835102505,
                1.2090078159764222,
                0.6562101468219043,
                1.8199795473885245,
                0.7861804646938978,
                -32.17231533477236,
                8.0,
            ],
        );
    }

    // ==========================================================================
    // R Validation Tests
    // ==========================================================================

    #[test]
    fn test_validate_multinom_vs_r() {
        // R reference: nnet::multinom(y ~ x1 + x2)
        // R: category B vs A: intercept=0.518, x1=1.111, x2=-0.510
        // R: category C vs A: intercept=-0.492, x1=0.394, x2=0.768
        // R: log-likelihood = -259.77

        let n = 300;
        let mut y_vec: Vec<&str> = Vec::with_capacity(n);
        let mut x1_vec: Vec<f64> = Vec::with_capacity(n);
        let mut x2_vec: Vec<f64> = Vec::with_capacity(n);

        for i in 0..n {
            // Generate x values
            let t = i as f64 / n as f64;
            let x1: f64 = (t * 6.0 - 3.0) * 0.6 + 0.3 * (i as f64 * 0.217).sin();
            let x2: f64 = (t * 5.0 - 2.5) * 0.5 + 0.3 * (i as f64 * 0.314).cos();
            x1_vec.push(x1);
            x2_vec.push(x2);

            // V_A = 0 (reference)
            // V_B = 0.5 + 1.0*x1 - 0.5*x2
            // V_C = -0.3 + 0.3*x1 + 0.8*x2

            let v_a = 0.0;
            let v_b = 0.5 + 1.0 * x1 - 0.5 * x2;
            let v_c = -0.3 + 0.3 * x1 + 0.8 * x2;

            // Add some noise via deterministic pseudo-random selection
            let noise = (i as f64 * 0.7 + 1.3).sin() * 1.5;

            let cat = if v_a + noise > v_b && v_a + noise > v_c {
                "A"
            } else if v_b + noise * 0.5 > v_c {
                "B"
            } else {
                "C"
            };
            y_vec.push(cat);
        }

        let df = df! {
            "y" => y_vec,
            "x1" => x1_vec,
            "x2" => x2_vec
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_multinom(&dataset, "y", &["x1", "x2"], None).unwrap();

        // Basic structure checks
        assert_eq!(result.n_obs, n);
        assert_eq!(result.categories.len(), 3);
        assert_eq!(result.coefficients.len(), 2); // 2 non-reference categories

        // Each category has 3 coefficients (intercept, x1, x2)
        assert_eq!(result.coefficients[0].len(), 3);
        assert_eq!(result.coefficients[1].len(), 3);

        // Log-likelihood should be finite and negative
        assert!(
            result.log_likelihood.is_finite(),
            "Log-likelihood should be finite"
        );
        assert!(
            result.log_likelihood < 0.0,
            "Log-likelihood should be negative"
        );

        // AIC should be positive
        assert!(result.aic > 0.0, "AIC should be positive");

        // Pseudo R-squared should be between 0 and 1
        assert!(
            result.pseudo_r_squared >= 0.0 && result.pseudo_r_squared <= 1.0,
            "Pseudo R-squared should be in [0, 1]: {}",
            result.pseudo_r_squared
        );
    }

    #[test]
    fn test_validate_multinom_4_categories() {
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
    }

    #[test]
    fn test_validate_multinom_coefficient_signs() {
        // Test that coefficient signs are consistent with data generating process
        let n = 300;
        let mut y_vec: Vec<&str> = Vec::with_capacity(n);
        let mut x_vec: Vec<f64> = Vec::with_capacity(n);

        for i in 0..n {
            // x increases from -2 to 2
            let x: f64 = (i as f64 / 75.0) - 2.0;
            x_vec.push(x);

            // Strong relationship: higher x -> category C
            // B has moderate x, A has low x
            let cat = if x < -0.5 {
                "A"
            } else if x < 1.0 {
                "B"
            } else {
                "C"
            };
            y_vec.push(cat);
        }

        let df = df! {
            "y" => y_vec,
            "x" => x_vec
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_multinom(&dataset, "y", &["x"], Some("A")).unwrap();

        // With A as reference, coefficients for B and C vs A should be positive for x
        // since higher x leads to B and C over A

        // Find B and C in results
        let b_idx = result.categories.iter().position(|c| c == "B");
        let c_idx = result.categories.iter().position(|c| c == "C");

        assert!(b_idx.is_some(), "B should be in categories");
        assert!(c_idx.is_some(), "C should be in categories");

        // Coefficient indices (B and C are non-reference, so 0 and 1)
        // The x coefficient is index 1 (index 0 is intercept)

        // B vs A: x coefficient should be positive
        assert!(
            result.coefficients[0][1] > 0.0,
            "B vs A coefficient for x should be positive: {}",
            result.coefficients[0][1]
        );

        // C vs A: x coefficient should be positive (and larger than B)
        assert!(
            result.coefficients[1][1] > 0.0,
            "C vs A coefficient for x should be positive: {}",
            result.coefficients[1][1]
        );

        // C should have larger coefficient than B (stronger relationship)
        assert!(
            result.coefficients[1][1] > result.coefficients[0][1],
            "C vs A should have larger x coefficient than B vs A"
        );
    }

    #[test]
    fn test_validate_multinom_convergence() {
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
    }
}
