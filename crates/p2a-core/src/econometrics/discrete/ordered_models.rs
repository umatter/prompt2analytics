//! Ordered Logit/Probit (Proportional Odds Model).
//!
//! # Mathematical Background
//!
//! For ordered outcomes y in {1, 2, ..., J}, the model is (polr's parameterisation):
//!
//! P(y <= j | X) = F(zeta_j - X'beta),   zeta_1 < ... < zeta_{J-1}
//!
//! where F is the logistic CDF for logit, or normal CDF for probit. The fit is the
//! maximum-likelihood estimate by Newton-Raphson on the analytic score and observed
//! Hessian; standard errors come from the inverse observed information.
//!
//! # References
//!
//! - McCullagh, P. (1980). Regression models for ordinal data.
//!
//! R equivalent: `MASS::polr()`

use ndarray::{Array1, Array2, ArrayView2};
use polars::prelude::{DataFrame, DataType};
use serde::{Deserialize, Serialize};
use statrs::distribution::{Continuous, ContinuousCDF, Normal};
use std::collections::HashSet;
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{self, ColumnRequest, Columns, Factor, MissingSummary, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::cholesky_inverse;
use crate::resource;
use crate::traits::estimator::logistic_cdf;

use super::newton::{MAX_HALVINGS, ensure_dense_inverse_fits, newton_step};

/// Most outcome categories an ordered logit/probit accepts.
///
/// A method-level cap, the analogue of `nnet::multinom`'s `MaxNWts`, active in
/// every mode: each category adds a threshold parameter to a dense Hessian
/// that is rebuilt and inverted every Newton iteration, so a continuous column
/// passed as the outcome would otherwise create one category per observation.
const MAX_OUTCOME_CATEGORIES: usize = 100;

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

/// Options of [`run_ordered_model_with`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderedConfig {
    /// Link function (default logit).
    pub link: OrderedModelType,
    /// Category order, lowest first (required for a text or categorical outcome).
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
/// R equivalent: `MASS::polr()`: `P(y <= j) = F(zeta_j - x'beta)` with the logistic
/// `F`. The estimates, standard errors and log-likelihood are polr's maximum
/// likelihood fit (`validation/econometrics/ordered.md`).
///
/// # Convergence
///
/// Newton-Raphson on the analytic score and observed Hessian, at most 100 iterations,
/// from the null-model MLE (beta = 0, cut points at `F^-1` of the cumulative category
/// shares). A step is halved until the cut points stay increasing and the
/// log-likelihood does not fall; where the observed information is not positive
/// definite the step is damped (Levenberg). The fit has converged when the Newton
/// decrement `g'(-H)^-1 g` is at most `1e-12 (1 + |ll|)`, the relative change of the
/// log-likelihood below `1e-10` and every component of the Newton step at most
/// `1e-10 (1 + |theta_k|)`. Unlike the largest score component, none of these depends
/// on the units of a regressor, so neither the estimates (Newton's method is
/// affine-invariant) nor the convergence flag do. `converged == true` implies finite
/// estimates and log-likelihood. A non-finite
/// step, a failed line search and the iteration limit end the fit with
/// `converged = false` and a message in `warnings`, never with an error. Standard
/// errors are the square roots of the diagonal of the inverse observed information;
/// they are NaN, with a warning, when the information is not positive definite.
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
/// R equivalent: `MASS::polr(method = "probit")`: the model of [`run_ordered_logit`]
/// with the standard normal `F`. Estimation, convergence and standard errors are as
/// for [`run_ordered_logit`].
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

    // Design matrix (no intercept: it is absorbed into the thresholds)
    let dm = cols.matrix(x_cols, false);
    let x = dm.view();
    let k = x.ncols();

    let var_names: Vec<String> = x_cols.iter().map(|s| s.to_string()).collect();
    let codes = factor.codes.as_slice();

    // Parameters: theta = (beta_1..beta_k, zeta_1..zeta_{J-1})
    let n_thresholds = j - 1;
    let n_params = k + n_thresholds;
    let hessian_what = match model_type {
        OrderedModelType::Logit => "ordered logit Hessian",
        OrderedModelType::Probit => "ordered probit Hessian",
    };
    ensure_dense_inverse_fits(n_params, hessian_what)?;
    let mut hess = resource::try_zeros2(n_params, n_params, hessian_what)?;
    let mut work = resource::try_zeros2(n_params, n_params, hessian_what)?;
    let mut grad = resource::try_vec(n_params, 0.0, "ordered model score")?;

    // Start at the null-model MLE: beta = 0, zeta_j = F^-1(P_j) with P_j the
    // cumulative category shares, so the first log-likelihood is the null one.
    let mut theta = resource::try_vec(n_params, 0.0, "ordered model parameters")?;
    let mut cumulative = 0usize;
    for t in 0..n_thresholds {
        cumulative += category_counts[t];
        theta[k + t] = link_quantile(cumulative as f64 / n as f64, model_type);
    }
    let mut trial = theta.clone();

    let mut ll = ordered_loglik(
        &x,
        codes,
        j,
        &theta,
        model_type,
        Some(&mut grad),
        Some(&mut hess),
    )?;
    let mut ll_prev = f64::NAN;
    let mut warnings: Vec<String> = Vec::new();
    let mut converged = false;
    let mut stopped = false;
    let mut iterations = 0;
    for iter in 1..=config.max_iter {
        resource::checkpoint()?;
        iterations = iter;
        if !(ll.is_finite()
            && grad.iter().all(|g| g.is_finite())
            && hess.iter().all(|h| h.is_finite()))
        {
            warnings.push(format!(
                "the log-likelihood, score or Hessian is not finite at iteration {iter}; \
                 estimation stopped"
            ));
            stopped = true;
            break;
        }
        let Some(delta) = newton_step(&hess, &grad, &mut work) else {
            warnings.push(format!(
                "the observed information is not positive definite at iteration {iter}, \
                 even with damping; estimation stopped"
            ));
            stopped = true;
            break;
        };
        // A non-finite step is not applied (RF-31: it never counts as converged).
        if max_abs_step(&delta).is_none() {
            warnings.push(format!(
                "the Newton step is not finite at iteration {iter}; estimation stopped"
            ));
            stopped = true;
            break;
        }
        // The Newton decrement g'(-H)^-1 g (twice the gain the step predicts) does not
        // depend on the units of a regressor; max |g| does (x times s: its score
        // component times s).
        let decrement: f64 = grad.iter().zip(&delta).map(|(g, d)| g * d).sum();
        let rel_step = delta
            .iter()
            .zip(&theta)
            .fold(0.0f64, |m, (d, t)| m.max(d.abs() / (1.0 + t.abs())));
        if iter > 1
            && decrement <= 1e-12 * (1.0 + ll.abs())
            && (ll - ll_prev).abs() < config.tol * (ll_prev.abs() + config.tol)
            && rel_step <= 1e-10
        {
            converged = true;
            break;
        }

        // Step halving: the cut points must stay increasing and the log-likelihood
        // must not fall.
        let mut accepted = false;
        let mut s = 1.0;
        for _ in 0..=MAX_HALVINGS {
            for ((t, th), d) in trial.iter_mut().zip(&theta).zip(&delta) {
                *t = th + s * d;
            }
            if trial[k..].windows(2).all(|w| w[0] < w[1]) {
                // the log-likelihood alone; score and Hessian once, at the accepted point
                let ll_trial = ordered_loglik(&x, codes, j, &trial, model_type, None, None)?;
                if ll_trial.is_finite() && ll_trial >= ll - 1e-12 * ll.abs() {
                    accepted = true;
                    break;
                }
            }
            s *= 0.5;
        }
        if !accepted {
            warnings.push(format!(
                "the line search could not increase the log-likelihood at iteration {iter}; \
                 estimation stopped"
            ));
            stopped = true;
            break;
        }
        std::mem::swap(&mut theta, &mut trial);
        ll_prev = ll;
        ll = ordered_loglik(
            &x,
            codes,
            j,
            &theta,
            model_type,
            Some(&mut grad),
            Some(&mut hess),
        )?;
    }
    if !converged && !stopped {
        let max_score = grad.iter().fold(0.0f64, |m, g| m.max(g.abs()));
        warnings.push(format!(
            "Newton-Raphson did not converge in {} iterations (max |score| = {max_score:.3e}); \
             the estimates may be unreliable (separation, or a category with very few \
             observations)",
            config.max_iter
        ));
    }
    let log_likelihood = ll;
    let converged = converged && log_likelihood.is_finite() && theta.iter().all(|t| t.is_finite());

    // Standard errors from the inverse observed information at the final estimate.
    hess.mapv_inplace(|h| -h);
    let covariance = if hess.iter().all(|h| h.is_finite()) {
        cholesky_inverse(&hess.view()).ok()
    } else {
        None
    };
    let se_all: Vec<f64> = match &covariance {
        Some(v) => (0..n_params).map(|i| v[[i, i]].sqrt()).collect(),
        None => {
            warnings.push(
                "the observed information is not positive definite; standard errors are \
                 not available"
                    .to_string(),
            );
            vec![f64::NAN; n_params]
        }
    };
    drop(covariance);

    let beta: Vec<f64> = theta[..k].to_vec();
    let alpha: Vec<f64> = theta[k..].to_vec();
    let std_errors: Vec<f64> = se_all[..k].to_vec();
    let threshold_std_errors: Vec<f64> = se_all[k..].to_vec();
    let z_stats: Vec<f64> = beta.iter().zip(&std_errors).map(|(b, se)| b / se).collect();
    let p_values: Vec<f64> = z_stats
        .iter()
        .map(|z| 2.0 * Normal::standard().sf(z.abs()))
        .collect();

    // Null model: thresholds only
    let log_likelihood_null: f64 = category_counts
        .iter()
        .map(|&c| c as f64 * (c as f64 / n as f64).ln())
        .sum();

    let pseudo_r_squared = 1.0 - (log_likelihood / log_likelihood_null);
    let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;
    let bic = -2.0 * log_likelihood + (n_params as f64) * (n as f64).ln();

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
        warnings,
    })
}

/// Log-likelihood of the ordered model at `theta = (beta, zeta)`, with its score in
/// `grad` and its observed Hessian in `hess` when they are given (polr's model,
/// `P(y <= j) = F(zeta_j - x'beta)`). `NEG_INFINITY` when some observation has
/// probability 0 (then `grad` and `hess` are incomplete).
///
/// Per observation in category `c` (0-based) with `u = zeta_c - eta` (`+inf` for the
/// last category) and `l = zeta_{c-1} - eta` (`-inf` for the first),
/// `p = F(u) - F(l)` and, with `f`, `f'` zero at infinite bounds,
/// `dp = (x (f(l) - f(u)), -f(l) at zeta_{c-1}, f(u) at zeta_c)` and
/// `d2p = (x x' (f'(u) - f'(l)); -x f'(u) at (beta, zeta_c); x f'(l) at
/// (beta, zeta_{c-1}); f'(u) at (zeta_c, zeta_c); -f'(l) at (zeta_{c-1}, zeta_{c-1}))`:
/// `ll += ln p`, `grad += dp / p`, `hess += d2p / p - dp dp' / p^2`, accumulated over
/// the slopes and the (at most two) cut points of the observation only.
fn ordered_loglik(
    x: &ArrayView2<f64>,
    codes: &[u32],
    n_cat: usize,
    theta: &[f64],
    link: OrderedModelType,
    mut grad: Option<&mut [f64]>,
    mut hess: Option<&mut Array2<f64>>,
) -> EconResult<f64> {
    let k = x.ncols();
    let (beta, zeta) = theta.split_at(k);
    if let Some(g) = grad.as_deref_mut() {
        g.fill(0.0);
    }
    if let Some(h) = hess.as_deref_mut() {
        h.fill(0.0);
    }
    let mut idx: Vec<usize> = Vec::with_capacity(k + 2);
    let mut dp: Vec<f64> = Vec::with_capacity(k + 2);
    let mut ll = 0.0;
    for (i, (row, &code)) in x.outer_iter().zip(codes).enumerate() {
        if i % 256 == 0 {
            resource::checkpoint()?;
        }
        let c = code as usize;
        let eta: f64 = row.iter().zip(beta).map(|(a, b)| a * b).sum();
        let upper = (c + 1 < n_cat).then(|| zeta[c] - eta);
        let lower = (c > 0).then(|| zeta[c - 1] - eta);
        let p = interval_probability(lower, upper, link);
        if p.is_nan() || p <= 0.0 {
            return Ok(f64::NEG_INFINITY);
        }
        ll += p.ln();
        if grad.is_none() && hess.is_none() {
            continue;
        }
        let (fu, fpu) = upper.map_or((0.0, 0.0), |u| link_density(u, link));
        let (fl, fpl) = lower.map_or((0.0, 0.0), |l| link_density(l, link));

        idx.clear();
        dp.clear();
        for (a, &xa) in row.iter().enumerate() {
            idx.push(a);
            dp.push(xa * (fl - fu));
        }
        if c > 0 {
            idx.push(k + c - 1);
            dp.push(-fl);
        }
        if c + 1 < n_cat {
            idx.push(k + c);
            dp.push(fu);
        }
        if let Some(g) = grad.as_deref_mut() {
            for (&a, &d) in idx.iter().zip(&dp) {
                g[a] += d / p;
            }
        }

        if let Some(h) = hess.as_deref_mut() {
            let slope = (fpu - fpl) / p;
            for (a, &xa) in row.iter().enumerate() {
                for (b, &xb) in row.iter().enumerate() {
                    h[[a, b]] += xa * xb * slope;
                }
            }
            if c + 1 < n_cat {
                let jc = k + c;
                for (a, &xa) in row.iter().enumerate() {
                    let v = -xa * fpu / p;
                    h[[a, jc]] += v;
                    h[[jc, a]] += v;
                }
                h[[jc, jc]] += fpu / p;
            }
            if c > 0 {
                let jl = k + c - 1;
                for (a, &xa) in row.iter().enumerate() {
                    let v = xa * fpl / p;
                    h[[a, jl]] += v;
                    h[[jl, a]] += v;
                }
                h[[jl, jl]] -= fpl / p;
            }
            let p2 = p * p;
            for (&a, &da) in idx.iter().zip(&dp) {
                for (&b, &db) in idx.iter().zip(&dp) {
                    h[[a, b]] -= da * db / p2;
                }
            }
        }
    }
    Ok(ll)
}

/// `F(u) - F(l)` for `l < u` (`None` = infinite bound), as `S(l) - S(u)` with the upper
/// tail `S = 1 - F` when `l > 0`, so that neither difference loses its digits.
fn interval_probability(lower: Option<f64>, upper: Option<f64>, link: OrderedModelType) -> f64 {
    match (lower, upper) {
        (None, Some(u)) => link_cdf(u, link),
        (Some(l), None) => link_sf(l, link),
        (Some(l), Some(u)) if l > 0.0 => link_sf(l, link) - link_sf(u, link),
        (Some(l), Some(u)) => link_cdf(u, link) - link_cdf(l, link),
        (None, None) => 1.0,
    }
}

/// `F(u)`.
fn link_cdf(u: f64, link: OrderedModelType) -> f64 {
    match link {
        OrderedModelType::Logit => logistic_cdf(u),
        OrderedModelType::Probit => Normal::standard().cdf(u),
    }
}

/// `1 - F(u)`, computed in the upper tail.
fn link_sf(u: f64, link: OrderedModelType) -> f64 {
    match link {
        OrderedModelType::Logit => link_cdf(-u, link),
        OrderedModelType::Probit => Normal::standard().sf(u),
    }
}

/// The density `f(u)` and its derivative `f'(u)` at a finite `u`.
fn link_density(u: f64, link: OrderedModelType) -> (f64, f64) {
    match link {
        OrderedModelType::Logit => {
            let e = (-u.abs()).exp();
            let f = e / ((1.0 + e) * (1.0 + e));
            (f, f * (1.0 - 2.0 * link_cdf(u, link)))
        }
        OrderedModelType::Probit => {
            let f = Normal::standard().pdf(u);
            (f, -u * f)
        }
    }
}

/// `F^-1(p)` for `0 < p < 1`.
fn link_quantile(p: f64, link: OrderedModelType) -> f64 {
    match link {
        OrderedModelType::Logit => (p / (1.0 - p)).ln(),
        OrderedModelType::Probit => Normal::standard().inverse_cdf(p),
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::design::DesignMatrix;
    use crate::resource::{self, Budget, CancelToken};
    use crate::test_support::load_case;
    use ndarray::array;
    use p2a_validation::{Case, Check, Tol};
    use polars::prelude::*;

    /// Case data for tests that are not R-pinned (these tests are not registered).
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

    fn zeta_names(r: &OrderedResult) -> Vec<String> {
        r.categories
            .windows(2)
            .map(|w| format!("{}|{}", w[0], w[1]))
            .collect()
    }

    fn counts(r: &OrderedResult) -> Vec<f64> {
        r.category_counts.iter().map(|&c| c as f64).collect()
    }

    /// Every quantity of one polr fit; `p` is "logit_" or "probit_".
    fn check_polr(case: &Case, p: &str, r: &OrderedResult) {
        let n = r.n_obs;
        assert!(
            r.converged,
            "{p}: not converged after {} iterations: {:?}",
            r.iterations, r.warnings
        );
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
        let zn = zeta_names(r);
        let k = |s: &str| format!("{p}{s}");
        Check::new(case)
            .named(
                &k("coef"),
                &r.variables,
                &r.coefficients,
                Tol::iterative(1e-6),
            )
            .named(&k("zeta"), &zn, &r.thresholds, Tol::iterative(1e-6))
            .named(
                &k("se_coef"),
                &r.variables,
                &r.std_errors,
                Tol::iterative(1e-5),
            )
            .named(
                &k("se_zeta"),
                &zn,
                &r.threshold_std_errors,
                Tol::iterative(1e-5),
            )
            .named(&k("z"), &r.variables, &r.z_stats, Tol::iterative(1e-5))
            // p of an iterative estimate: d ln p / d ln z ~ z^2 (<= 46 here) times the z
            // tolerance
            .named(&k("p"), &r.variables, &r.p_values, Tol::iterative(2e-3))
            .scalar(&k("loglik"), r.log_likelihood, Tol::stat(n))
            .scalar(&k("loglik_null"), r.log_likelihood_null, Tol::stat(n))
            .scalar(&k("pseudo_r2"), r.pseudo_r_squared, Tol::stat(n))
            .scalar(&k("aic"), r.aic, Tol::stat(n))
            .scalar(&k("bic"), r.bic, Tol::stat(n))
            .vector(&k("counts"), &counts(r), Tol::exact())
            .scalar(&k("n"), n as f64, Tol::exact())
            // polr's own BFGS optimum (reltol 1e-15, max |grad| ~ 2e-6): a looser
            // cross-check
            .named(
                &k("coef_pkg"),
                &r.variables,
                &r.coefficients,
                Tol::iterative(1e-4),
            )
            .named(&k("zeta_pkg"), &zn, &r.thresholds, Tol::iterative(1e-4))
            .named(
                &k("se_coef_pkg"),
                &r.variables,
                &r.std_errors,
                Tol::iterative(1e-4),
            )
            .finish();
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
    /// The fit is now polr's maximum-likelihood fit, so this pins the coefficient, the cut
    /// points, their standard errors and the log-likelihood of both links, besides n,
    /// the category counts and the null log-likelihood. The case values are polr's
    /// optimum (BFGS, reltol = 1e-14): coefficients within 8.6e-9, cut points within
    /// 1.2e-7, SEs within 1.8e-6 relative of the polished MLE (measured in R as
    /// validation/econometrics/discrete_newton.md describes), inside the tolerances below.
    fn pin_ordered(case: &Case, ds: &Dataset) {
        for (link, model) in [
            ("logit", OrderedModelType::Logit),
            ("probit", OrderedModelType::Probit),
        ] {
            let r = fit_j(case, ds, model);
            let n = r.n_obs;
            assert!(
                r.converged && r.warnings.is_empty(),
                "{link}: {:?}\n{r}",
                r.warnings
            );
            let zn = zeta_names(&r);
            let k = |s: &str| format!("{link}_{s}");
            Check::new(case)
                .named(
                    &k("coef"),
                    &r.variables,
                    &r.coefficients,
                    Tol::iterative(1e-6),
                )
                .named(&k("zeta"), &zn, &r.thresholds, Tol::iterative(1e-6))
                .named(
                    &k("se_coef"),
                    &r.variables,
                    &r.std_errors,
                    Tol::iterative(1e-5),
                )
                .named(
                    &k("se_zeta"),
                    &zn,
                    &r.threshold_std_errors,
                    Tol::iterative(1e-5),
                )
                .scalar(&k("loglik"), r.log_likelihood, Tol::stat(n))
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

    /// The fit must not depend on the scale of a regressor. An earlier step inverted
    /// the negative definite Hessian through the general inverse, whose fallback
    /// answered cond > 1e10 with the pseudoinverse: disc_polr_j5 with x times 1e4
    /// returned its starting point. Newton's method is affine-invariant, so the iterates
    /// on the scaled data are the unscaled ones up to rounding: the coefficient divided
    /// by the scale, the cut points and the log-likelihood equal, to 1e-9 relative. So
    /// is the convergence test (the Newton decrement), and the scaled fit stops at the
    /// same iteration; a test on max |score| (the slope's component times the scale)
    /// ran to the iteration limit at 1e12.
    /// Each scaled fit is also checked against polr's case: the cut points and the
    /// log-likelihood, n, the category counts and the null log-likelihood do not depend
    /// on the scale of x.
    #[test]
    fn test_validate_ordered_j5_regressor_scale() {
        let (case, ds) = crate::test_support::load_case("disc_polr_j5");
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * b.abs();
        for (link, model) in [
            ("logit", OrderedModelType::Logit),
            ("probit", OrderedModelType::Probit),
        ] {
            let base = fit_j(&case, &ds, model);
            for scale in [1e4, 1e6, 1e12] {
                let r = fit_j(&case, &with_x_scaled(&ds, scale), model);
                assert!(
                    r.converged,
                    "{link}, x * {scale:e}: not converged: {:?}",
                    r.warnings
                );
                // the iterates are the unscaled ones, and so is the stopping iteration
                assert_eq!(r.iterations, base.iterations, "{link}, x * {scale:e}");
                Check::new(&case)
                    .scalar("n", r.n_obs as f64, Tol::exact())
                    .named("counts", &r.categories, &counts(&r), Tol::exact())
                    .scalar("null_loglik", r.log_likelihood_null, Tol::stat(r.n_obs))
                    .named(
                        &format!("{link}_zeta"),
                        &zeta_names(&r),
                        &r.thresholds,
                        Tol::iterative(1e-6),
                    )
                    .scalar(
                        &format!("{link}_loglik"),
                        r.log_likelihood,
                        Tol::stat(r.n_obs),
                    )
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

    // ---- polr's fit: estimates, standard errors and statistics -----------------------

    #[test]
    fn test_validate_ordered_logit_disc_ordered() {
        let (case, ds) = load_case("disc_ordered");
        let r = run_ordered_logit(&ds, "y", &["x1", "x2"]).unwrap();
        assert_eq!(r.categories, ["1", "2", "3"]);
        check_polr(&case, "logit_", &r);
    }

    #[test]
    fn test_validate_ordered_probit_disc_ordered() {
        let (case, ds) = load_case("disc_ordered");
        let r = run_ordered_probit(&ds, "y", &["x1", "x2"]).unwrap();
        assert_eq!(r.categories, ["1", "2", "3"]);
        check_polr(&case, "probit_", &r);
    }

    /// The text labels in their ordinal order give the integer fit; in R's alphabetical
    /// order (high < low < mid) they give R's fit of that (different) model.
    #[test]
    fn test_validate_ordered_logit_text_outcome_with_levels() {
        let (case, ds) = load_case("disc_ordered");
        let fit = |levels: &[&str]| {
            run_ordered_model_with(
                &ds,
                "y_label",
                &["x1", "x2"],
                &cfg(OrderedModelType::Logit, Some(levels)),
            )
            .unwrap()
        };
        let r = fit(&["low", "mid", "high"]);
        assert!(r.converged, "{:?}", r.warnings);
        Check::new(&case)
            .named(
                "logit_coef",
                &r.variables,
                &r.coefficients,
                Tol::iterative(1e-6),
            )
            // the labels differ (low|mid vs 1|2): compare in order
            .vector("logit_zeta", &r.thresholds, Tol::iterative(1e-6))
            .named(
                "logit_se_coef",
                &r.variables,
                &r.std_errors,
                Tol::iterative(1e-5),
            )
            .scalar("logit_loglik", r.log_likelihood, Tol::stat(r.n_obs))
            .finish();
        let a = fit(&["high", "low", "mid"]);
        assert!(a.converged, "{:?}", a.warnings);
        Check::new(&case)
            .named(
                "alpha_labels_coef",
                &a.variables,
                &a.coefficients,
                Tol::iterative(1e-6),
            )
            .named(
                "alpha_labels_zeta",
                &zeta_names(&a),
                &a.thresholds,
                Tol::iterative(1e-6),
            )
            .scalar("alpha_labels_loglik", a.log_likelihood, Tol::stat(a.n_obs))
            .finish();
    }

    /// The private score and Hessian are the derivatives of the log-likelihood (central
    /// differences, step 1e-6), for both links, away from the optimum.
    #[test]
    fn test_ordered_analytic_derivatives_match_central_differences() {
        let ds = ordered_data();
        let df = ds.df();
        let cols = ColumnRequest::new(df).factor("y").extract().unwrap();
        let codes = cols.factor("y").codes.clone();
        let dm = DesignMatrix::from_dataframe(df, &["x1", "x2"], false).unwrap();
        let x = dm.view();
        let theta = [0.3, -0.2, -0.5, 0.9];
        let m = theta.len();
        let h = 1e-6;
        for link in [OrderedModelType::Logit, OrderedModelType::Probit] {
            let mut g = vec![0.0; m];
            let mut hess = Array2::zeros((m, m));
            ordered_loglik(&x, &codes, 3, &theta, link, Some(&mut g), Some(&mut hess)).unwrap();
            for kk in 0..m {
                let (mut up, mut dn) = (theta, theta);
                up[kk] += h;
                dn[kk] -= h;
                let (mut g_up, mut g_dn) = (vec![0.0; m], vec![0.0; m]);
                let ll_up =
                    ordered_loglik(&x, &codes, 3, &up, link, Some(&mut g_up), None).unwrap();
                let ll_dn =
                    ordered_loglik(&x, &codes, 3, &dn, link, Some(&mut g_dn), None).unwrap();
                let fd = (ll_up - ll_dn) / (2.0 * h);
                assert!(
                    (fd - g[kk]).abs() <= 1e-6 * (1.0 + g[kk].abs()),
                    "{link:?}: score[{kk}] {} vs central difference {fd}",
                    g[kk]
                );
                for a in 0..m {
                    let fd = (g_up[a] - g_dn[a]) / (2.0 * h);
                    assert!(
                        (fd - hess[[a, kk]]).abs() <= 1e-5 * (1.0 + hess[[a, kk]].abs()),
                        "{link:?}: Hessian[{a}, {kk}] {} vs central difference {fd}",
                        hess[[a, kk]]
                    );
                }
            }
        }
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
        assert!(r.converged, "{:?}", r.warnings);
        let n = r.n_obs;
        Check::new(&case)
            .vector("wide_counts", &counts(&r), Tol::exact())
            .named(
                "wide_coef",
                &r.variables,
                &r.coefficients,
                Tol::iterative(1e-6),
            )
            .named(
                "wide_zeta",
                &zeta_names(&r),
                &r.thresholds,
                Tol::iterative(1e-6),
            )
            .named(
                "wide_se_coef",
                &r.variables,
                &r.std_errors,
                Tol::iterative(1e-5),
            )
            .scalar("wide_loglik", r.log_likelihood, Tol::stat(n))
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
            .named(
                "logit_coef",
                &r.variables,
                &r.coefficients,
                Tol::iterative(1e-6),
            )
            // the labels differ (low|mid vs 1|2): compare in order
            .vector("logit_zeta", &r.thresholds, Tol::iterative(1e-6))
            .scalar("logit_loglik", r.log_likelihood, Tol::stat(r.n_obs))
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
            .named(
                "na_coef",
                &r.variables,
                &r.coefficients,
                Tol::iterative(1e-6),
            )
            .named(
                "na_zeta",
                &zeta_names(&r),
                &r.thresholds,
                Tol::iterative(1e-6),
            )
            .named(
                "na_se_coef",
                &r.variables,
                &r.std_errors,
                Tol::iterative(1e-5),
            )
            .scalar("na_loglik", r.log_likelihood, Tol::stat(r.n_obs))
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

    /// `max_iter` and `tol` are validated; the message names the field.
    #[test]
    fn test_ordered_config_is_validated() {
        let ds = ordered_data();
        let bad = [
            (
                OrderedConfig {
                    max_iter: 0,
                    ..OrderedConfig::default()
                },
                "max_iter",
            ),
            (
                OrderedConfig {
                    tol: f64::NAN,
                    ..OrderedConfig::default()
                },
                "tol",
            ),
            (
                OrderedConfig {
                    tol: 0.0,
                    ..OrderedConfig::default()
                },
                "tol",
            ),
            (
                OrderedConfig {
                    tol: -1e-8,
                    ..OrderedConfig::default()
                },
                "tol",
            ),
        ];
        for (config, field) in bad {
            let message = invalid_message(run_ordered_model_with(&ds, "y", &["x1", "x2"], &config));
            assert!(message.contains(field), "{field}: {message}");
        }
    }

    /// An iteration limit the fit cannot meet ends it with `converged = false` and a
    /// warning, not an error.
    #[test]
    fn test_ordered_iteration_limit_is_reported() {
        let ds = ordered_data();
        let config = OrderedConfig {
            max_iter: 2,
            ..OrderedConfig::default()
        };
        let r = run_ordered_model_with(&ds, "y", &["x1", "x2"], &config).unwrap();
        assert!(!r.converged, "{r}");
        assert_eq!(r.iterations, 2);
        assert!(
            r.warnings.iter().any(|w| w.contains("did not converge")),
            "{:?}",
            r.warnings
        );
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

    /// 100 categories of one row each (T0's cap fixture) have an MLE: the old loop (wrong
    /// score sign) stopped on a pseudoinverse step at iteration 18 instead. R (MASS
    /// 7.3.60.0.1): `i <- 0:99; x <- ((i * 7) %% 13) + 0.1 * i; yv <- i + 0.5;
    /// MASS::polr(factor(yv) ~ x, method = "logistic", control = list(reltol = 1e-15,
    /// maxit = 100000))` (convergence 0), polished by Newton steps on the analytic score
    /// (`.polish` and `.polr_grad` of validation/r/cases/discrete.R) to max |grad|
    /// 8.8e-13: coefficient 0.27537780315599802, log-likelihood -437.01698120738473. A
    /// cap fixture, not a validation dataset: the values stay inline.
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
            r.converged && r.iterations < 100,
            "converged {} after {} iterations: {:?}",
            r.converged,
            r.iterations,
            r.warnings
        );
        let coef = 0.275_377_803_155_998;
        assert!(
            (r.coefficients[0] - coef).abs() <= 1e-6 * coef,
            "coefficient {} vs polr {coef}",
            r.coefficients[0]
        );
        let ll = -437.01698120738473_f64;
        assert!(
            (r.log_likelihood - ll).abs() <= 1e-9 * ll.abs(),
            "log-likelihood {} vs polr {ll}",
            r.log_likelihood
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
}
