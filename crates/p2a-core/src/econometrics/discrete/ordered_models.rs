//! Ordered Logit/Probit (Proportional Odds Model).
//!
//! # Mathematical Background
//!
//! For ordered outcomes y in {1, 2, ..., J}, the model is:
//!
//! P(y <= j | X) = F(alpha_j - X'beta)
//!
//! where F is the logistic CDF for logit, or normal CDF for probit.
//!
//! # References
//!
//! - McCullagh, P. (1980). Regression models for ordinal data.
//!
//! R equivalent: `MASS::polr()`

use ndarray::{Array1, Array2};
use polars::prelude::{DataFrame, DataType};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{self, ColumnRequest, Columns, Factor, MissingSummary, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::safe_inverse;
use crate::resource;
use crate::traits::estimator::{logistic_cdf, logistic_pdf, normal_cdf, normal_pdf};

/// Most outcome categories an ordered logit/probit accepts.
///
/// A method-level cap, the analogue of `nnet::multinom`'s `MaxNWts`, active in
/// every mode: each category adds a threshold parameter to a dense Hessian
/// that is rebuilt and inverted every Newton iteration, so a continuous column
/// passed as the outcome would otherwise create one category per observation.
const MAX_OUTCOME_CATEGORIES: usize = 100;

/// `safe_inverse` holds several m x m copies at once (faer copy, factor, identity or solution,
/// residual product, SVD fallback). Check the working set against the allocation ceiling
/// before building it.
const DENSE_INVERSE_COPIES: usize = 6;

fn ensure_dense_inverse_fits(m: usize, what: &str) -> EconResult<()> {
    let cells = resource::checked_len(&[m, m, DENSE_INVERSE_COPIES], what)?;
    resource::check_alloc::<f64>(cells, what)
}

/// Largest absolute component of a Newton step, or `None` if any component is not
/// finite. (The fold `max_change.max(change.abs())` it replaces dropped NaN, so an
/// all-NaN step counted as converged, RF-31.)
fn max_abs_step(step: &Array1<f64>) -> Option<f64> {
    let mut m = 0.0f64;
    for s in step {
        if !s.is_finite() {
            return None;
        }
        m = m.max(s.abs());
    }
    Some(m)
}

/// Type of ordered discrete choice model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderedModelType {
    Logit,
    Probit,
}

impl fmt::Display for OrderedModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderedModelType::Logit => write!(f, "Ordered Logit"),
            OrderedModelType::Probit => write!(f, "Ordered Probit"),
        }
    }
}

/// Options of [`run_ordered_model_with`] (implemented by T2-WP10).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderedConfig {
    /// Link function (default logit).
    pub link: OrderedModelType,
    /// ordinal order of a text outcome (required for String outcomes)
    pub levels: Option<Vec<String>>,
    /// Maximum Newton iterations (default 100).
    pub max_iter: usize,
    /// Convergence tolerance on the relative change of the log-likelihood (default 1e-10).
    pub tol: f64,
}

impl Default for OrderedConfig {
    fn default() -> Self {
        OrderedConfig {
            link: OrderedModelType::Logit,
            levels: None,
            max_iter: 100,
            tol: 1e-10,
        }
    }
}

/// Result from ordered logit/probit regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderedResult {
    /// Model type
    pub model_type: OrderedModelType,
    /// Dependent variable name
    pub dep_var: String,
    /// Variable names (excluding intercept - absorbed into thresholds)
    pub variables: Vec<String>,
    /// Ordered categories
    pub categories: Vec<String>,
    /// Coefficient estimates (beta)
    pub coefficients: Vec<f64>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// Z-statistics
    pub z_stats: Vec<f64>,
    /// P-values
    pub p_values: Vec<f64>,
    /// Threshold (cut-point) estimates (alpha_1, alpha_2, ..., alpha_{j-1})
    pub thresholds: Vec<f64>,
    /// Threshold standard errors
    pub threshold_std_errors: Vec<f64>,
    /// Log-likelihood at convergence
    pub log_likelihood: f64,
    /// Log-likelihood of null model
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
    /// Rows dropped because of missing values in the outcome or a regressor (R polr: na.omit).
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
    /// Estimation problems in plain words (non-convergence, singular information).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

impl fmt::Display for OrderedResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.model_type)?;
        writeln!(f, "========================")?;
        writeln!(f, "Dependent variable: {}", self.dep_var)?;
        writeln!(
            f,
            "N = {}, Categories = {}",
            self.n_obs,
            self.categories.len()
        )?;
        writeln!(f)?;

        writeln!(f, "Coefficients:")?;
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
        for (i, var) in self.variables.iter().enumerate() {
            writeln!(
                f,
                "{:<15} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
                var, self.coefficients[i], self.std_errors[i], self.z_stats[i], self.p_values[i]
            )?;
        }
        writeln!(f)?;

        writeln!(f, "Thresholds:")?;
        for (i, threshold) in self.thresholds.iter().enumerate() {
            writeln!(
                f,
                "  {}|{}: {:.4} (SE: {:.4})",
                self.categories[i],
                self.categories[i + 1],
                threshold,
                self.threshold_std_errors[i]
            )?;
        }
        writeln!(f)?;

        writeln!(f, "Log-Likelihood: {:.4}", self.log_likelihood)?;
        writeln!(f, "Pseudo R-squared: {:.4}", self.pseudo_r_squared)?;
        writeln!(f, "AIC: {:.4}, BIC: {:.4}", self.aic, self.bic)?;
        writeln!(
            f,
            "Converged: {} ({} iterations)",
            self.converged, self.iterations
        )?;
        if !self.warnings.is_empty() {
            writeln!(f, "Warnings:")?;
            for w in &self.warnings {
                writeln!(f, "  - {w}")?;
            }
        }
        if let Some(note) = self.missing.note() {
            writeln!(f, "Missing values: {note}")?;
        }
        Ok(())
    }
}

/// Run ordered logit regression (proportional odds model).
///
/// R equivalent: `MASS::polr()`. The estimates are not yet polr's: the beta score
/// has the wrong sign and the Hessian is an outer-product approximation, so the fit
/// does not reach the MLE (ECON-4, T2-WP10), and the standard errors are a 0.1
/// placeholder (ECON-5).
///
/// # Convergence
///
/// Newton steps on the information matrix (the negated Hessian approximation plus
/// a 1e-6 ridge, positive definite), at most 100. The inverse takes the Cholesky
/// path at any conditioning, so ill-conditioning that comes from the scale of a
/// regressor (values in the thousands, say) neither stops nor truncates the
/// iteration. `converged == true` implies that every coefficient, every threshold
/// and the log-likelihood are finite, and that the last step came from a regular
/// (not pseudo-) inverse with largest absolute (clamped) change below `1e-8`. A
/// non-finite score or step ends the iteration with `converged = false`, and that
/// step is not applied. So does a step from the pseudoinverse, which
/// `safe_inverse` returns only when Cholesky fails in floating point (the ridge
/// lost to rounding) and the condition number exceeds `1e10`. Non-convergence is
/// reported in the result, never as an error.
///
/// # Errors
///
/// As [`run_ordered_model_with`] with its defaults: among others, an outcome with
/// more than 100 distinct values is refused with `InvalidSpecification` (a
/// method-level cap on the outcome categories), and so is a text outcome, whose
/// order needs [`OrderedConfig::levels`]. Rows with missing values are dropped.
pub fn run_ordered_logit(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
) -> EconResult<OrderedResult> {
    let config = OrderedConfig {
        link: OrderedModelType::Logit,
        ..OrderedConfig::default()
    };
    run_ordered_model_with(dataset, y_col, x_cols, &config)
}

/// Run ordered probit regression.
///
/// R equivalent: `MASS::polr(method = "probit")`. The estimates are not yet
/// polr's (ECON-4, ECON-5; T2-WP10). Iteration and convergence are as for
/// [`run_ordered_logit`]: Newton steps on the positive definite information
/// matrix (Cholesky at any conditioning); `converged == true` implies finite
/// estimates and log-likelihood and a last step from a regular inverse; a
/// non-finite step, or a pseudoinverse step (Cholesky failed in floating point
/// and cond > 1e10), ends the iteration, not converged and not applied.
///
/// # Errors
///
/// As [`run_ordered_model_with`] with its defaults: among others, an outcome with
/// more than 100 distinct values is refused with `InvalidSpecification` (a
/// method-level cap on the outcome categories), and so is a text outcome, whose
/// order needs [`OrderedConfig::levels`]. Rows with missing values are dropped.
pub fn run_ordered_probit(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
) -> EconResult<OrderedResult> {
    let config = OrderedConfig {
        link: OrderedModelType::Probit,
        ..OrderedConfig::default()
    };
    run_ordered_model_with(dataset, y_col, x_cols, &config)
}

/// Ordered logit or probit with the options of [`OrderedConfig`] (R `MASS::polr`).
///
/// The outcome and the regressors are read together and rows with a missing value in
/// any of them are dropped (R `na.omit`), reported in [`OrderedResult::missing`].
/// The categories follow the outcome's data type: numbers by value (2 < 10,
/// -2 < -1), an Enum in its declared order, booleans FALSE < TRUE, dates by time.
/// A text or categorical outcome has no such order and needs
/// [`OrderedConfig::levels`], lowest category first.
///
/// # Errors
///
/// `InvalidSpecification` when `max_iter` is 0, `tol` is not positive and finite,
/// `levels` is empty, repeats a label, misses an observed label or names one without
/// observations, when no regressor is given, when a text outcome comes without
/// `levels`, and when the outcome has fewer than 2 or more than 100 categories (a
/// method-level cap). Column errors (`ColumnNotFound`, `NonNumericColumn`,
/// `NonFiniteValues`) name the column.
pub fn run_ordered_model_with(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    config: &OrderedConfig,
) -> EconResult<OrderedResult> {
    let model_type = config.link;
    if config.max_iter == 0 {
        return Err(EconError::InvalidSpecification {
            message: "max_iter must be at least 1".to_string(),
        });
    }
    if !(config.tol.is_finite() && config.tol > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!("tol must be a positive, finite number (got {})", config.tol),
        });
    }
    if let Some(levels) = &config.levels {
        check_levels(levels, y_col)?;
    }
    if x_cols.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "At least one predictor is required".to_string(),
        });
    }

    let df = dataset.df();
    let cols = ColumnRequest::new(df)
        .factor(y_col)
        .numerics(x_cols)
        .na(NaPolicy::Omit)
        .extract()?;
    let factor = ordered_factor(&cols, df, y_col, config.levels.as_deref())?;
    let n = cols.n();
    let categories = factor.levels.clone();
    let category_counts = factor.counts();

    let j = categories.len();
    if j < 2 {
        return Err(EconError::InvalidSpecification {
            message: "Ordered model requires at least 2 categories".to_string(),
        });
    }
    if j > MAX_OUTCOME_CATEGORIES {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Outcome column '{y_col}' has {j} distinct values; ordered logit/probit \
                 supports at most {MAX_OUTCOME_CATEGORIES} outcome categories (like \
                 nnet::multinom's MaxNWts limit). Check that '{y_col}' is a categorical \
                 outcome, not a continuous variable, or recode it into fewer categories."
            ),
        });
    }

    let y_idx: Vec<usize> = factor.codes.iter().map(|&c| c as usize).collect();

    // Design matrix (no intercept: it is absorbed into the thresholds)
    let dm = cols.matrix(x_cols, false);
    let x = dm.view();
    let k = x.ncols();

    let var_names: Vec<String> = x_cols.iter().map(|s| s.to_string()).collect();

    // Number of thresholds = J - 1
    let n_thresholds = j - 1;
    let n_params = k + n_thresholds;

    // Initialize: thresholds evenly spaced, beta = 0
    let mut theta = vec![0.0; n_params];

    // Initialize thresholds based on marginal proportions
    let mut cum_prop = 0.0;
    for t in 0..n_thresholds {
        cum_prop += category_counts[t] as f64 / n as f64;
        let init_thresh = match model_type {
            OrderedModelType::Logit => (cum_prop / (1.0 - cum_prop.min(0.999))).ln(),
            OrderedModelType::Probit => {
                // Approximate inverse normal
                let p = cum_prop.max(0.001).min(0.999);
                let t_approx = ((2.0 * p - 1.0).abs()).sqrt();
                (2.0 * p - 1.0).signum() * t_approx * 1.5
            }
        };
        theta[k + t] = init_thresh;
    }

    // CDF function
    let cdf_fn: fn(f64) -> f64 = match model_type {
        OrderedModelType::Logit => logistic_cdf,
        OrderedModelType::Probit => normal_cdf,
    };

    let pdf_fn: fn(f64) -> f64 = match model_type {
        OrderedModelType::Logit => logistic_pdf,
        OrderedModelType::Probit => normal_pdf,
    };

    // Newton-Raphson iteration
    let max_iter = 100;
    let tol = 1e-8;
    let mut converged = false;
    let mut iterations = 0;
    let hessian_what = match model_type {
        OrderedModelType::Logit => "ordered logit Hessian",
        OrderedModelType::Probit => "ordered probit Hessian",
    };
    ensure_dense_inverse_fits(n_params, hessian_what)?;
    // Positions and values of the non-zero score entries of one observation:
    // the k slopes and at most two thresholds.
    let mut score_idx: Vec<usize> = Vec::with_capacity(k + 2);
    let mut score_val: Vec<f64> = Vec::with_capacity(k + 2);

    for iter in 0..max_iter {
        resource::checkpoint()?;
        iterations = iter + 1;

        // Extract current parameters
        let beta: Vec<f64> = theta[..k].to_vec();
        let alpha: Vec<f64> = theta[k..].to_vec();

        // Compute gradient and Hessian
        let mut gradient = vec![0.0; n_params];
        let mut hess = resource::try_zeros2(n_params, n_params, hessian_what)?;

        for i in 0..n {
            if i % 256 == 0 {
                resource::checkpoint()?;
            }
            let yi = y_idx[i];

            // Compute X'beta
            let mut xb = 0.0;
            for kk in 0..k {
                xb += x[[i, kk]] * beta[kk];
            }

            // P(Y <= j) = F(alpha_j - X'beta)
            let p_low = if yi == 0 {
                0.0
            } else {
                cdf_fn(alpha[yi - 1] - xb)
            };
            let p_high = if yi == j - 1 {
                1.0
            } else {
                cdf_fn(alpha[yi] - xb)
            };
            let p_i = (p_high - p_low).max(1e-15);

            let f_low = if yi == 0 {
                0.0
            } else {
                pdf_fn(alpha[yi - 1] - xb)
            };
            let f_high = if yi == j - 1 {
                0.0
            } else {
                pdf_fn(alpha[yi] - xb)
            };

            // Gradient w.r.t. beta
            let grad_beta = (f_high - f_low) / p_i;
            for kk in 0..k {
                gradient[kk] += x[[i, kk]] * grad_beta;
            }

            // Gradient w.r.t. thresholds
            if yi > 0 {
                gradient[k + yi - 1] -= f_low / p_i;
            }
            if yi < j - 1 {
                gradient[k + yi] += f_high / p_i;
            }

            // Hessian (approximate using outer product), over the non-zero
            // score entries only
            score_idx.clear();
            score_val.clear();
            for kk in 0..k {
                score_idx.push(kk);
                score_val.push(x[[i, kk]] * grad_beta);
            }
            if yi > 0 {
                score_idx.push(k + yi - 1);
                score_val.push(-f_low / p_i);
            }
            if yi < j - 1 {
                score_idx.push(k + yi);
                score_val.push(f_high / p_i);
            }
            subtract_scaled_outer(&mut hess, &score_idx, &score_val, p_i);
        }

        // A non-finite score ends the iteration, not converged.
        if !gradient.iter().all(|g| g.is_finite()) {
            break;
        }

        // Information matrix: the negated Hessian (approximation) plus a 1e-6 ridge.
        // It is positive definite, so safe_inverse takes its Cholesky path at any
        // conditioning. (The negative definite Hessian itself always fails
        // Cholesky, and for cond > 1e10, as with a regressor in the thousands,
        // safe_inverse answers it with the pseudoinverse.)
        let mut info = hess;
        info.mapv_inplace(|v| -v);
        for i in 0..n_params {
            info[[i, i]] += 1e-6;
        }

        let (info_inv, singular) = match safe_inverse(&info.view()) {
            Ok(inv) => inv,
            Err(_) => {
                // Fall back to gradient descent step
                let step_size = 0.1;
                for i in 0..n_params {
                    theta[i] += step_size * gradient[i];
                }
                continue;
            }
        };
        drop(info);

        // Newton step: theta += I^-1 g (the same iterate as theta -= H^-1 g)
        let grad_arr = Array1::from_vec(gradient);
        let step = info_inv.dot(&grad_arr);
        drop(info_inv);

        // A non-finite step ends the iteration unapplied and not converged; so does
        // one from the pseudoinverse, which safe_inverse returns only when Cholesky
        // fails in floating point despite the ridge and cond exceeds 1e10.
        let Some(max_step) = max_abs_step(&step) else {
            break;
        };
        if singular.is_some() {
            break;
        }

        // Update with step size control: each component clamped to [-2, 2]
        for i in 0..n_params {
            theta[i] += step[i].clamp(-2.0, 2.0);
        }
        // The largest clamped change, max_i |clamp(step_i)| = min(max_i |step_i|, 2)
        let max_change = max_step.min(2.0);

        // Ensure thresholds are ordered
        for t in 1..n_thresholds {
            if theta[k + t] <= theta[k + t - 1] {
                theta[k + t] = theta[k + t - 1] + 0.1;
            }
        }

        if max_change < tol {
            converged = true;
            break;
        }
    }

    // Final parameters
    let beta: Vec<f64> = theta[..k].to_vec();
    let alpha: Vec<f64> = theta[k..].to_vec();

    // Compute log-likelihood
    let mut log_likelihood = 0.0;
    for i in 0..n {
        let yi = y_idx[i];
        let mut xb = 0.0;
        for kk in 0..k {
            xb += x[[i, kk]] * beta[kk];
        }

        let p_low = if yi == 0 {
            0.0
        } else {
            cdf_fn(alpha[yi - 1] - xb)
        };
        let p_high = if yi == j - 1 {
            1.0
        } else {
            cdf_fn(alpha[yi] - xb)
        };
        let p_i = (p_high - p_low).max(1e-15);
        log_likelihood += p_i.ln();
    }
    let converged = converged && log_likelihood.is_finite() && theta.iter().all(|t| t.is_finite());

    // Null model: thresholds only
    let log_likelihood_null: f64 = category_counts
        .iter()
        .map(|&c| c as f64 * (c as f64 / n as f64).ln())
        .sum();

    let pseudo_r_squared = 1.0 - (log_likelihood / log_likelihood_null);
    let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;
    let bic = -2.0 * log_likelihood + (n_params as f64) * (n as f64).ln();

    // Standard errors (simplified - using observed information)
    let std_errors: Vec<f64> = beta.iter().map(|_| 0.1).collect(); // Placeholder
    let z_stats: Vec<f64> = beta
        .iter()
        .zip(&std_errors)
        .map(|(b, se)| if *se > 1e-15 { b / se } else { 0.0 })
        .collect();
    let p_values: Vec<f64> = z_stats
        .iter()
        .map(|z| 2.0 * (1.0 - normal_cdf(z.abs())))
        .collect();

    let threshold_std_errors: Vec<f64> = alpha.iter().map(|_| 0.1).collect(); // Placeholder

    Ok(OrderedResult {
        model_type,
        dep_var: y_col.to_string(),
        variables: var_names,
        categories,
        coefficients: beta,
        std_errors,
        z_stats,
        p_values,
        thresholds: alpha,
        threshold_std_errors,
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
        warnings: Vec::new(),
    })
}

/// `levels` must be non-empty and name each label once.
fn check_levels(levels: &[String], y_col: &str) -> EconResult<()> {
    if levels.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "`levels` is empty; give every category of '{y_col}' once, lowest first"
            ),
        });
    }
    let mut seen: HashSet<&str> = HashSet::with_capacity(levels.len());
    for l in levels {
        if !seen.insert(l.as_str()) {
            return Err(EconError::InvalidSpecification {
                message: format!("`levels` lists '{}' more than once", short_label(l)),
            });
        }
    }
    Ok(())
}

/// The outcome's categories in model order: the data type's order, or `levels`.
/// A text or categorical outcome needs `levels`; a level of `levels` without
/// observations is refused (polr cannot place the cut points next to it).
fn ordered_factor(
    cols: &Columns,
    df: &DataFrame,
    y_col: &str,
    levels: Option<&[String]>,
) -> EconResult<Factor> {
    let factor = cols.factor(y_col);
    let Some(levels) = levels else {
        let dtype = columns::column(df, y_col)?.dtype();
        if matches!(dtype, DataType::String | DataType::Categorical(..)) {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "Outcome column '{y_col}' is text with labels {}; ordered logit/probit \
                     cannot guess their order. Pass the order as `levels`, lowest category \
                     first, every label once.",
                    label_list(&factor.levels)
                ),
            });
        }
        return Ok(factor.clone());
    };
    let ordered = factor.with_level_order(levels)?;
    if let Some((label, _)) = ordered
        .levels
        .iter()
        .zip(ordered.counts())
        .find(|(_, count)| *count == 0)
    {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "level '{}' in `levels` has no observations in column '{y_col}'; polr \
                 cannot estimate the cut points next to an empty category. Remove it from \
                 `levels`.",
                short_label(label)
            ),
        });
    }
    Ok(ordered)
}

/// At most 20 labels, each shortened, comma-separated.
fn label_list(labels: &[String]) -> String {
    const SHOWN: usize = 20;
    let mut parts: Vec<String> = labels.iter().take(SHOWN).map(|l| short_label(l)).collect();
    if labels.len() > SHOWN {
        parts.push(format!("... ({} in all)", labels.len()));
    }
    parts.join(", ")
}

/// A label of at most 40 characters (longer ones end in "...").
fn short_label(label: &str) -> String {
    const MAX_CHARS: usize = 40;
    if label.chars().count() <= MAX_CHARS {
        label.to_string()
    } else {
        format!("{}...", label.chars().take(MAX_CHARS).collect::<String>())
    }
}

/// Subtract `g g' / p_i` from `hess` for a score `g` that is zero except at the
/// positions `idx`, where it takes the values `val`.
///
/// Only the non-zero pairs are visited. The terms the dense double loop would
/// add for the other pairs are exact zeros, so the result is bit-identical.
fn subtract_scaled_outer(hess: &mut Array2<f64>, idx: &[usize], val: &[f64], p_i: f64) {
    for (&a, &g_a) in idx.iter().zip(val) {
        for (&b, &g_b) in idx.iter().zip(val) {
            hess[[a, b]] -= g_a * g_b / p_i;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{self, Budget, CancelToken};
    use crate::test_support::load_case;
    use ndarray::array;
    use p2a_validation::{Case, Check, Tol};
    use polars::prelude::*;

    /// Case data for tests that are not R-pinned (not registered; see the plan's
    /// test-placement rule).
    fn ordered_data() -> Dataset {
        load_case("disc_ordered").1
    }

    /// Rows 1-12 of `disc_ordered` (7, 3 and 2 rows of categories 1, 2, 3).
    fn ordered_head() -> Dataset {
        Dataset::new(ordered_data().df().head(Some(12)))
    }

    fn cfg(link: OrderedModelType, levels: Option<&[&str]>) -> OrderedConfig {
        OrderedConfig {
            link,
            levels: levels.map(|l| l.iter().map(|s| s.to_string()).collect()),
            ..OrderedConfig::default()
        }
    }

    fn counts(r: &OrderedResult) -> Vec<f64> {
        r.category_counts.iter().map(|&c| c as f64).collect()
    }

    fn with_column(ds: &Dataset, col: Column) -> Dataset {
        let mut df = ds.df().clone();
        df.with_column(col).unwrap();
        Dataset::new(df)
    }

    fn invalid_message<T: std::fmt::Debug>(r: EconResult<T>) -> String {
        match r {
            Err(EconError::InvalidSpecification { message }) => message,
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    /// The fit with polr's level order (the case's `counts` names, the sorted labels
    /// c001, c002, ...): the outcome is text, which needs `levels` (ECON-19).
    fn fit_j(case: &Case, ds: &Dataset, link: OrderedModelType) -> OrderedResult {
        let config = OrderedConfig {
            link,
            levels: Some(case.names("counts").to_vec()),
            ..OrderedConfig::default()
        };
        run_ordered_model_with(ds, "y", &["x"], &config).unwrap()
    }

    /// RF-31: with 16 or more categories the Newton step went through faer 0.22's
    /// full-pivot LU, which returned NaN, and the NaN fit was reported as converged.
    /// The estimates are not polr's until ECON-4 is fixed (T2-WP10; deviation in
    /// validation/methods/discrete_newton.json), so this pins n, the category counts
    /// and the null log-likelihood, and requires finite estimates and an honest
    /// convergence flag.
    fn pin_ordered(case: &Case, ds: &Dataset) {
        for (link, r) in [
            ("logit", fit_j(case, ds, OrderedModelType::Logit)),
            ("probit", fit_j(case, ds, OrderedModelType::Probit)),
        ] {
            let n = r.n_obs;
            let finite = r
                .coefficients
                .iter()
                .chain(&r.thresholds)
                .all(|v| v.is_finite())
                && r.log_likelihood.is_finite();
            assert!(finite, "{link}: non-finite estimates\n{r}");
            // The MLE maximises the likelihood: no finite iterate can lie above polr's.
            let ll_max = case.scalar(&format!("{link}_loglik"));
            assert!(
                r.log_likelihood <= ll_max + 1e-8 * ll_max.abs(),
                "{link}: {} > polr {ll_max}",
                r.log_likelihood
            );
            assert!(
                !r.converged || r.log_likelihood >= r.log_likelihood_null,
                "{link}: converged below the null model\n{r}"
            );
            // deviation ECON-4: coefficients, thresholds and log-likelihood vs
            // <link>_coef/_zeta/_loglik (T2-WP10)
            // deviation ECON-5: standard errors are the 0.1 placeholder (T2-WP10)
            assert_eq!(r.thresholds.len(), r.categories.len() - 1);
            Check::new(case)
                .scalar("n", n as f64, Tol::exact())
                .named("counts", &r.categories, &counts(&r), Tol::exact())
                .scalar("null_loglik", r.log_likelihood_null, Tol::stat(n))
                .finish();
        }
    }

    /// Control: 5 categories (5 parameters), below the size where the old LU failed.
    #[test]
    fn test_validate_ordered_j5() {
        let (case, ds) = crate::test_support::load_case("disc_polr_j5");
        pin_ordered(&case, &ds);
    }

    /// `ds` with its column `x` multiplied by `scale` (outcome `y` unchanged).
    fn with_x_scaled(ds: &Dataset, scale: f64) -> Dataset {
        let y: Vec<String> = ds
            .column("y")
            .unwrap()
            .str()
            .unwrap()
            .into_no_null_iter()
            .map(String::from)
            .collect();
        let x: Vec<f64> = ds
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .map(|v| v * scale)
            .collect();
        Dataset::new(df! { "y" => y, "x" => x }.unwrap())
    }

    /// The fit must not depend on the scale of a regressor. The step used to invert
    /// the negative definite Hessian, which always takes safe_inverse's fallback,
    /// and that answers cond > 1e10 with the pseudoinverse. With the rule that a
    /// pseudoinverse step ends the fit, disc_polr_j5 with x times 1e4 returned its
    /// starting point after one iteration (coefficient 0, thresholds at their start
    /// values, the null log-likelihood). On the information matrix (Cholesky) the
    /// scaled fit is the unscaled one with the coefficient divided by the scale, to
    /// 1.1e-10 relative (logit) and 7.9e-11 (probit) measured at both scales: the
    /// 1e-6 ridge is not scale-free. The fits are the non-converged iterates of
    /// ECON-4. Each scaled fit is also checked against polr's case: n, the category
    /// counts and the null log-likelihood do not depend on the scale of x.
    #[test]
    fn test_validate_ordered_j5_regressor_scale() {
        let (case, ds) = crate::test_support::load_case("disc_polr_j5");
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * b.abs();
        for (link, model) in [
            ("logit", OrderedModelType::Logit),
            ("probit", OrderedModelType::Probit),
        ] {
            let base = fit_j(&case, &ds, model);
            for scale in [1e4, 1e6] {
                let r = fit_j(&case, &with_x_scaled(&ds, scale), model);
                Check::new(&case)
                    .scalar("n", r.n_obs as f64, Tol::exact())
                    .named("counts", &r.categories, &counts(&r), Tol::exact())
                    .scalar("null_loglik", r.log_likelihood_null, Tol::stat(r.n_obs))
                    .finish();
                let want = base.coefficients[0] / scale;
                assert!(
                    r.iterations > 1 && close(r.coefficients[0], want),
                    "{link}, x * {scale:e}: coefficient {} after {} iterations, want {want}",
                    r.coefficients[0],
                    r.iterations
                );
                for (t, (a, b)) in r.thresholds.iter().zip(&base.thresholds).enumerate() {
                    assert!(
                        close(*a, *b),
                        "{link}, x * {scale:e}: threshold {t} {a} vs {b}"
                    );
                }
                assert!(
                    close(r.log_likelihood, base.log_likelihood),
                    "{link}, x * {scale:e}: LL {} vs {}",
                    r.log_likelihood,
                    base.log_likelihood
                );
            }
        }
    }

    #[test]
    fn test_validate_ordered_j16() {
        let (case, ds) = crate::test_support::load_case("disc_polr_j16");
        pin_ordered(&case, &ds);
    }

    #[test]
    fn test_validate_ordered_j30() {
        let (case, ds) = crate::test_support::load_case("disc_polr_j30");
        pin_ordered(&case, &ds);
    }

    // ---- category order, levels and missing values (ECON-19, ECON-12) ----------------

    /// 13 integer levels -2..10 are ordered by value (R factor()), not as strings
    /// ("-1" < "-2", "10" < "2").
    #[test]
    fn test_validate_ordered_logit_integer_levels_in_numeric_order() {
        let (case, ds) = load_case("disc_ordered");
        let r = run_ordered_logit(&ds, "y_wide", &["x1", "x2"]).unwrap();
        let want: Vec<String> = case
            .vector("wide_levels")
            .iter()
            .map(|&v| format!("{}", v as i64))
            .collect();
        assert_eq!(r.categories, want);
        Check::new(&case)
            .vector("wide_counts", &counts(&r), Tol::exact())
            .finish();
    }

    /// An Enum outcome keeps its declared order (low < mid < high), which is not the
    /// alphabetical one.
    #[test]
    fn test_validate_ordered_logit_enum_outcome_uses_declared_order() {
        let (case, ds) = load_case("disc_ordered");
        let declared = FrozenCategories::new(["low", "mid", "high"]).unwrap();
        let y_enum = ds
            .column("y_label")
            .unwrap()
            .as_materialized_series()
            .strict_cast(&DataType::from_frozen_categories(declared))
            .unwrap()
            .with_name("y_enum".into());
        let ds = with_column(&ds, y_enum.into_column());
        let r = run_ordered_logit(&ds, "y_enum", &["x1", "x2"]).unwrap();
        assert_eq!(r.categories, ["low", "mid", "high"]);
        Check::new(&case)
            .vector("logit_counts", &counts(&r), Tol::exact())
            .finish();
    }

    /// Rows with a missing outcome or regressor are dropped jointly (polr: na.omit)
    /// and reported; a missing outcome is not a category.
    #[test]
    fn test_validate_ordered_logit_missing_rows_dropped() {
        let (case, ds) = load_case("disc_ordered");
        let r = run_ordered_logit(&ds, "y_na", &["x1_na", "x2"]).unwrap();
        assert_eq!(
            r.missing.by_column,
            [("y_na".to_string(), 4), ("x1_na".to_string(), 3)]
        );
        assert_eq!(r.missing.n_total, 400);
        assert_eq!(r.categories, ["1", "2", "3"]);
        let text = r.to_string();
        assert!(
            text.contains("Missing values: 7 of 400 rows dropped"),
            "{text}"
        );
        Check::new(&case)
            .scalar("na_n_used", r.n_obs as f64, Tol::exact())
            .scalar("na_n_dropped", r.missing.n_dropped as f64, Tol::exact())
            .scalar(
                "na_missing_y",
                r.missing.by_column[0].1 as f64,
                Tol::exact(),
            )
            .scalar(
                "na_missing_x1",
                r.missing.by_column[1].1 as f64,
                Tol::exact(),
            )
            .finish();
    }

    /// A text outcome has no order the program could know: an error that lists the
    /// observed labels and asks for `levels`, instead of an alphabetical fit.
    #[test]
    fn test_ordered_text_outcome_requires_levels() {
        let ds = ordered_data();
        let message = invalid_message(run_ordered_logit(&ds, "y_label", &["x1", "x2"]));
        for part in ["'y_label'", "high", "low", "mid", "levels"] {
            assert!(message.contains(part), "{part}: {message}");
        }
    }

    #[test]
    fn test_ordered_levels_are_validated() {
        let ds = ordered_data();
        let fit = |levels: &[&str]| {
            run_ordered_model_with(
                &ds,
                "y_label",
                &["x1", "x2"],
                &cfg(OrderedModelType::Logit, Some(levels)),
            )
        };
        // An observed label missing from `levels` (the message names the label).
        let message = invalid_message(fit(&["low", "mid"]));
        assert!(message.contains("'high'"), "{message}");
        let message = invalid_message(fit(&["low", "mid", "top", "high"]));
        assert!(
            message.contains("'top'") && message.contains("no observations"),
            "{message}"
        );
        let message = invalid_message(fit(&["low", "mid", "mid", "high"]));
        assert!(
            message.contains("'mid'") && message.contains("more than once"),
            "{message}"
        );
        let message = invalid_message(fit(&[]));
        assert!(message.contains("levels"), "{message}");

        let r = fit(&["low", "mid", "high"]).unwrap();
        assert_eq!(r.categories, ["low", "mid", "high"]);
        let integer = run_ordered_logit(&ds, "y", &["x1", "x2"]).unwrap();
        assert_eq!(r.category_counts, integer.category_counts);
    }

    /// An Int64 outcome and the same values as Float64 are the same model.
    #[test]
    fn test_na_ordered_float_outcome_equals_int64() {
        let ds = ordered_data();
        let y_f64 = ds
            .column("y")
            .unwrap()
            .as_materialized_series()
            .cast(&DataType::Float64)
            .unwrap()
            .with_name("y_f64".into());
        let ds = with_column(&ds, y_f64.into_column());
        let a = run_ordered_logit(&ds, "y", &["x1", "x2"]).unwrap();
        let b = run_ordered_logit(&ds, "y_f64", &["x1", "x2"]).unwrap();
        assert_eq!(a.categories, b.categories);
        assert_eq!(a.category_counts, b.category_counts);
        assert_eq!(a.coefficients, b.coefficients);
        assert_eq!(a.thresholds, b.thresholds);
    }

    #[test]
    fn test_na_ordered_column_errors() {
        let ds = ordered_data();
        match run_ordered_logit(&ds, "y", &["y_label"]) {
            Err(EconError::NonNumericColumn { column }) => assert_eq!(column, "y_label"),
            other => panic!("expected NonNumericColumn, got {other:?}"),
        }
        for (y, x) in [("nope", "x1"), ("y", "nope")] {
            match run_ordered_logit(&ds, y, &[x]) {
                Err(EconError::ColumnNotFound { column, .. }) => assert_eq!(column, "nope"),
                other => panic!("expected ColumnNotFound, got {other:?}"),
            }
        }
        invalid_message(run_ordered_logit(&ds, "y", &[]));
    }

    // ---- resource bounds (T0-WP14) ------------------------------------------------------

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
    fn test_ordered_refuses_more_than_100_categories() {
        let err = run_ordered_logit(&continuous_y_dataset(101), "y", &["x"]).unwrap_err();
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

    /// 100 categories of one row each. Before Tier 1 the fit was finite, but every
    /// Newton step used a wrong full-pivot LU inverse (RF-31). It must stay finite
    /// and end before the iteration limit, not converged: here the information
    /// matrix fails Cholesky in floating point despite its ridge, safe_inverse
    /// answers with the pseudoinverse, and that step ends the fit (measured: step
    /// 18). With pseudoinverse steps applied the fit runs all 100 iterations, not
    /// converged (measured), so the iteration count is what this test can see.
    #[test]
    fn test_ordered_accepts_100_categories() {
        let r = run_ordered_logit(&continuous_y_dataset(100), "y", &["x"]).unwrap();
        assert_eq!(r.categories.len(), 100);
        assert!(
            r.coefficients
                .iter()
                .chain(&r.thresholds)
                .all(|v| v.is_finite())
                && r.log_likelihood.is_finite(),
            "non-finite estimates: LL {}, converged {} after {} iterations",
            r.log_likelihood,
            r.converged,
            r.iterations
        );
        assert!(
            !r.converged && r.iterations < 100,
            "expected a stop on a pseudoinverse step: converged {} after {} iterations \
             (LL {}, null {})",
            r.converged,
            r.iterations,
            r.log_likelihood,
            r.log_likelihood_null
        );
    }

    #[test]
    fn test_ordered_max_abs_step_rejects_non_finite() {
        // The old fold `max_change.max(change.abs())` dropped NaN (f64::max), so the
        // all-NaN steps of RF-31 counted as converged.
        assert_eq!(max_abs_step(&array![0.5, -2.0]), Some(2.0));
        assert_eq!(max_abs_step(&Array1::zeros(0)), Some(0.0));
        assert_eq!(max_abs_step(&array![1.0, f64::NAN]), None);
        assert_eq!(max_abs_step(&array![f64::INFINITY]), None);
        assert_eq!(max_abs_step(&array![f64::NAN, 1.0]), None);
    }

    #[test]
    fn test_ordered_stops_when_cancelled() {
        let ds = ordered_head();
        let e = resource::with_budget(cancelled(), || run_ordered_logit(&ds, "y", &["x1", "x2"]))
            .unwrap_err();
        assert!(matches!(e, EconError::Cancelled(_)), "{e:?}");
    }

    #[test]
    fn test_ordered_hessian_respects_alloc_ceiling() {
        // n_params = 2 + 2 = 4; working set 4*4*6*8 = 768 bytes > 512, while the
        // 12-row columns (96 bytes) and design (192 bytes) fit.
        let ds = ordered_head();
        let e =
            resource::with_alloc_ceiling(Some(512), || run_ordered_logit(&ds, "y", &["x1", "x2"]))
                .unwrap_err();
        match e {
            EconError::ResourceLimit(m) => assert!(m.contains("Hessian"), "{m}"),
            other => panic!("expected ResourceLimit, got {other:?}"),
        }
    }

    #[test]
    fn test_sparse_outer_product_matches_dense_loop() {
        let n_params = 9;
        let idx = [0usize, 1, 2, 5, 6];
        let val = [0.31, -1.7, 2.25, -0.42, 0.9];
        let p_i = 0.37;
        let mut base = Array2::<f64>::from_shape_fn((n_params, n_params), |(a, b)| {
            0.01 * (a * 3 + b) as f64 - 0.2
        });
        let mut dense = base.clone();
        let mut g = vec![0.0; n_params];
        for (&i, &v) in idx.iter().zip(&val) {
            g[i] = v;
        }
        for a in 0..n_params {
            for b in 0..n_params {
                dense[[a, b]] -= g[a] * g[b] / p_i;
            }
        }
        subtract_scaled_outer(&mut base, &idx, &val, p_i);
        assert_eq!(base, dense); // bitwise: skipped entries would only subtract +-0.0
    }
}
