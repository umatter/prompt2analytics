//! Count data models: Negative Binomial, Zero-Inflated, and Hurdle models.
//!
//! # Models Included
//!
//! - **Negative Binomial**: For overdispersed count data (glm.nb)
//! - **Zero-Inflated Poisson (ZIP)**: For excess zeros with Poisson counts
//! - **Zero-Inflated Negative Binomial (ZINB)**: For excess zeros with overdispersion
//! - **Hurdle models**: Two-part models (binary + truncated count)
//!
//! # References
//!
//! - Cameron, A.C. & Trivedi, P.K. (2013). Regression Analysis of Count Data.
//!
//! R equivalents: `MASS::glm.nb()`, `pscl::zeroinfl()`, `pscl::hurdle()`

use ndarray::{Array1, Array2, ArrayView2};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, MissingSummary, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::{cholesky_inverse, safe_inverse};
use crate::resource;
use crate::traits::estimator::{logistic_cdf, normal_cdf};

// ============================================================================
// Negative Binomial Regression
// ============================================================================

/// Result from negative binomial regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NegBinResult {
    /// Dependent variable name
    pub dep_var: String,
    /// Variable names
    pub variables: Vec<String>,
    /// Coefficient estimates
    pub coefficients: Vec<f64>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// Z-statistics
    pub z_stats: Vec<f64>,
    /// P-values
    pub p_values: Vec<f64>,
    /// Overdispersion parameter theta
    pub theta: f64,
    /// Standard error of theta
    pub theta_std_error: f64,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// Null log-likelihood
    pub log_likelihood_null: f64,
    /// McFadden's pseudo R-squared
    pub pseudo_r_squared: f64,
    /// AIC
    pub aic: f64,
    /// BIC
    pub bic: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of iterations
    pub iterations: usize,
    /// Whether converged
    pub converged: bool,
    /// Mean of y
    pub y_mean: f64,
    /// Variance of y
    pub y_var: f64,
    /// Rows dropped because of missing values in the outcome or a regressor (R: na.omit).
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
    /// Estimation problems in plain words (non-convergence, no overdispersion).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

impl fmt::Display for NegBinResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Negative Binomial Regression")?;
        writeln!(f, "============================")?;
        writeln!(f, "Dependent variable: {}", self.dep_var)?;
        writeln!(f, "N = {}", self.n_obs)?;
        writeln!(f)?;
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
        writeln!(
            f,
            "Theta: {:.4} (SE: {:.4})",
            self.theta, self.theta_std_error
        )?;
        writeln!(f, "Log-Likelihood: {:.4}", self.log_likelihood)?;
        writeln!(f, "Pseudo R-squared: {:.4}", self.pseudo_r_squared)?;
        writeln!(f, "AIC: {:.4}, BIC: {:.4}", self.aic, self.bic)?;
        writeln!(
            f,
            "Converged: {} ({} iterations)",
            self.converged, self.iterations
        )?;
        write_notes(f, &self.warnings, &self.missing)
    }
}

/// The `Warnings:` block and the `Missing values:` line of a result's Display, each
/// only when non-empty.
fn write_notes(
    f: &mut fmt::Formatter<'_>,
    warnings: &[String],
    missing: &MissingSummary,
) -> fmt::Result {
    if !warnings.is_empty() {
        writeln!(f, "Warnings:")?;
        for w in warnings {
            writeln!(f, "  - {w}")?;
        }
    }
    if let Some(note) = missing.note() {
        writeln!(f, "Missing values: {note}")?;
    }
    Ok(())
}

/// Largest theta the NB and ZINB fits use. Without overdispersion the likelihood keeps
/// rising towards the Poisson limit theta = inf.
const THETA_MAX: f64 = 1e8;

/// A final theta above this means no overdispersion: the NB2 variance
/// `mu + mu^2 / theta` is then within `1e-6 mu^2` of the Poisson variance. (The theta
/// score is at rounding level near [`THETA_MAX`], so where theta stops between the two
/// depends on rounding; R's glm.nb stops with "iteration limit reached".)
const THETA_POISSON: f64 = 1e6;

/// The warning for a theta above [`THETA_POISSON`].
fn no_overdispersion_warning(theta: f64) -> String {
    format!(
        "theta = {theta:.3e}: the data show no overdispersion beyond the Poisson model, \
         which fits as well (R glm.nb warns 'iteration limit reached')"
    )
}

/// Digamma function (derivative of log-gamma).
fn digamma(x: f64) -> f64 {
    use statrs::function::gamma::digamma as statrs_digamma;
    statrs_digamma(x)
}

/// Trigamma function (second derivative of log-gamma), for `x > 0`: the recurrence
/// `psi1(x) = psi1(x + 1) + 1/x^2` up to `x >= 10`, then the asymptotic series to the
/// `x^-11` term (relative error below 1e-12). NaN for `x <= 0` or a non-finite `x`.
fn trigamma(x: f64) -> f64 {
    if !(x > 0.0 && x.is_finite()) {
        return f64::NAN;
    }
    let mut result = 0.0;
    let mut xx = x;
    while xx < 10.0 {
        result += 1.0 / (xx * xx);
        xx += 1.0;
    }
    let inv = 1.0 / xx;
    let inv2 = inv * inv;
    // 1/x + 1/(2x^2) + 1/(6x^3) - 1/(30x^5) + 1/(42x^7) - 1/(30x^9) + 5/(66x^11)
    let series = inv2
        * (1.0 / 6.0
            + inv2
                * (-1.0 / 30.0 + inv2 * (1.0 / 42.0 + inv2 * (-1.0 / 30.0 + inv2 * (5.0 / 66.0)))));
    result + inv + inv2 / 2.0 + inv * series
}

/// `exp(eta)` with `eta` capped at 700 (glm.fit's guard against overflow).
fn exp_link(eta: f64) -> f64 {
    eta.min(700.0).exp()
}

/// `eta = X beta` and `mu = exp(eta)` into the buffers.
fn linear_predictor(x: &ArrayView2<f64>, beta: &Array1<f64>, eta: &mut [f64], mu: &mut [f64]) {
    for ((row, e), m) in x.outer_iter().zip(eta.iter_mut()).zip(mu.iter_mut()) {
        *e = row.dot(beta);
        *m = exp_link(*e);
    }
}

/// `X' W X` and `X' W z` of a weighted least-squares step.
fn weighted_cross_products(
    x: &ArrayView2<f64>,
    w: &[f64],
    z: &[f64],
) -> EconResult<(Array2<f64>, Array1<f64>)> {
    let k = x.ncols();
    let mut xtwx = resource::try_zeros2(k, k, "IRLS cross-product matrix")?;
    let mut xtwz = Array1::zeros(k);
    for (i, row) in x.outer_iter().enumerate() {
        if i % 4096 == 0 {
            resource::checkpoint()?;
        }
        for a in 0..k {
            let wa = w[i] * row[a];
            xtwz[a] += wa * z[i];
            for b in 0..=a {
                xtwx[[a, b]] += wa * row[b];
            }
        }
    }
    for a in 0..k {
        for b in 0..a {
            xtwx[[b, a]] = xtwx[[a, b]];
        }
    }
    Ok((xtwx, xtwz))
}

/// The NB2 IRLS weight `mu theta / (theta + mu)`; the Poisson weight `mu` for an
/// infinite theta.
fn nb_weight(mu: f64, theta: f64) -> f64 {
    if theta.is_infinite() {
        mu
    } else {
        mu * theta / (theta + mu)
    }
}

/// Work vectors of the NB fit, one entry per observation.
struct NbWork {
    eta: Vec<f64>,
    mu: Vec<f64>,
    w: Vec<f64>,
    z: Vec<f64>,
}

impl NbWork {
    fn new(n: usize) -> EconResult<NbWork> {
        Ok(NbWork {
            eta: resource::try_vec(n, 0.0, "negative binomial linear predictor")?,
            mu: resource::try_vec(n, 0.0, "negative binomial fitted means")?,
            w: resource::try_vec(n, 0.0, "negative binomial IRLS weights")?,
            z: resource::try_vec(n, 0.0, "negative binomial working response")?,
        })
    }
}

/// The error for an X'WX that is not positive definite.
fn nb_singular() -> EconError {
    EconError::SingularMatrix {
        context: "negative binomial IRLS (X'WX)".to_string(),
        suggestion: "check for collinear regressors".to_string(),
    }
}

/// Fisher scoring for the coefficients of the NB2 log-link model at a fixed `theta`
/// (`f64::INFINITY`: Poisson), glm.fit's IRLS: weights `W = mu theta / (theta + mu)`,
/// working response `z = eta + (y - mu) / mu`. Starts from `beta`, or from
/// `mu = y + 0.1` (glm's Poisson start) when `beta` is `None`. Iterates until
/// `max |delta beta| < 1e-12 (1 + max |beta|)`, at most 100 times. `work.eta` and
/// `work.mu` hold the fitted values on return.
fn irls_beta(
    x: &ArrayView2<f64>,
    y: &[f64],
    beta: Option<&Array1<f64>>,
    theta: f64,
    work: &mut NbWork,
) -> EconResult<Array1<f64>> {
    let mut beta = match beta {
        Some(b) => {
            linear_predictor(x, b, &mut work.eta, &mut work.mu);
            Some(b.clone())
        }
        None => {
            for ((e, m), &yi) in work.eta.iter_mut().zip(work.mu.iter_mut()).zip(y) {
                *m = (yi + 0.1).max(0.1);
                *e = m.ln();
            }
            None
        }
    };
    for _ in 0..100 {
        resource::checkpoint()?;
        for i in 0..y.len() {
            let m = work.mu[i].max(f64::MIN_POSITIVE);
            work.w[i] = nb_weight(m, theta);
            work.z[i] = work.eta[i] + (y[i] - m) / m;
        }
        let (xtwx, xtwz) = weighted_cross_products(x, &work.w, &work.z)?;
        let inv = cholesky_inverse(&xtwx.view()).map_err(|_| nb_singular())?;
        let new_beta = inv.dot(&xtwz);
        if new_beta.iter().any(|b| !b.is_finite()) {
            return Err(nb_singular());
        }
        let scale = 1.0 + new_beta.iter().fold(0.0f64, |m, b| m.max(b.abs()));
        let change = beta.as_ref().map_or(f64::INFINITY, |old| {
            new_beta
                .iter()
                .zip(old)
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()))
        });
        linear_predictor(x, &new_beta, &mut work.eta, &mut work.mu);
        beta = Some(new_beta);
        if change < 1e-12 * scale {
            break;
        }
    }
    Ok(beta.unwrap_or_else(|| Array1::zeros(x.ncols())))
}

/// `ln Gamma(theta + y) - ln Gamma(theta)`.
fn ln_gamma_ratio(theta: f64, y: f64) -> f64 {
    use statrs::function::gamma::ln_gamma;
    ln_gamma(theta + y) - ln_gamma(theta)
}

/// The NB2 log-likelihood at `theta` and the fitted means.
fn nb_loglik(theta: f64, mu: &[f64], y: &[f64]) -> f64 {
    use statrs::function::gamma::ln_gamma;
    y.iter()
        .zip(mu)
        .map(|(&yi, &mi)| {
            let log_mu = if yi == 0.0 { 0.0 } else { yi * mi.ln() };
            ln_gamma_ratio(theta, yi) - ln_gamma(yi + 1.0) + theta * theta.ln() + log_mu
                - (theta + yi) * (theta + mi).ln()
        })
        .sum()
}

/// MASS `theta.ml`'s score of theta at fixed `mu` (signs as MASS).
fn theta_score(theta: f64, mu: &[f64], y: &[f64]) -> f64 {
    y.iter()
        .zip(mu)
        .map(|(&yi, &mi)| {
            digamma(theta + yi) - digamma(theta) + theta.ln() + 1.0
                - (theta + mi).ln()
                - (yi + theta) / (mi + theta)
        })
        .sum()
}

/// MASS `theta.ml`'s (observed) information of theta at fixed `mu` (signs as MASS).
fn theta_information(theta: f64, mu: &[f64], y: &[f64]) -> f64 {
    y.iter()
        .zip(mu)
        .map(|(&yi, &mi)| {
            -trigamma(theta + yi) + trigamma(theta) - 1.0 / theta + 2.0 / (mi + theta)
                - (yi + theta) / ((mi + theta) * (mi + theta))
        })
        .sum()
}

/// The ML theta at fixed `mu`: Newton steps on `phi = ln theta` with MASS theta.ml's
/// score `s` and information `I` (`delta = s / (theta I - s)` where the profile is
/// concave in `phi`, else `sign(s)`), clamped to +-2 and halved while the profile
/// log-likelihood falls by more than its rounding (`1e-12 |ll|`). Stops when
/// `|delta| < 1e-12`, after 100 steps, when no halving helps, or at [`THETA_MAX`].
fn theta_newton(theta: f64, mu: &[f64], y: &[f64]) -> EconResult<f64> {
    let mut theta = theta.min(THETA_MAX);
    let mut ll = nb_loglik(theta, mu, y);
    for _ in 0..100 {
        resource::checkpoint()?;
        let s = theta_score(theta, mu, y);
        let info = theta_information(theta, mu, y);
        if !(s.is_finite() && info.is_finite() && ll.is_finite()) {
            break;
        }
        let curvature = theta * info - s;
        let mut delta = if curvature > 0.0 {
            s / curvature
        } else {
            s.signum()
        };
        delta = delta.clamp(-2.0, 2.0);
        if delta.abs() < 1e-12 {
            break;
        }
        let mut accepted = false;
        for _ in 0..30 {
            let candidate = (theta.ln() + delta).exp().min(THETA_MAX);
            let ll_candidate = nb_loglik(candidate, mu, y);
            // not lower beyond rounding: near the optimum a step changes the
            // log-likelihood by less than its rounding error
            if ll_candidate.is_finite() && ll_candidate >= ll - 1e-12 * ll.abs() {
                theta = candidate;
                ll = ll_candidate;
                accepted = true;
                break;
            }
            delta *= 0.5;
        }
        if !accepted || theta >= THETA_MAX {
            break;
        }
    }
    Ok(theta)
}

/// Run negative binomial regression (NB2, log link).
///
/// R equivalent: `MASS::glm.nb()`. The coefficients come from Fisher scoring (IRLS)
/// at fixed theta, theta from Newton steps on MASS `theta.ml`'s score and information
/// at fixed fitted means, alternating until theta changes by less than 1e-10 relative
/// and the log-likelihood by less than 1e-12 relative (at most 100 rounds), as glm.nb
/// does. `initial_theta` is the starting value of theta (glm.nb's `init.theta`); without
/// it the start is a Poisson fit and MASS's moment estimate `n / sum((y/mu - 1)^2)`.
/// Standard errors of the coefficients are `(X'WX)^-1` with `W = mu theta/(theta + mu)`
/// (summary.negbin), `SE(theta) = 1/sqrt(information)` at the optimum (MASS
/// `SE.theta`).
///
/// Rows with a missing outcome or regressor are dropped (R `na.omit`) and reported in
/// `missing`. Data without overdispersion drive theta towards its bound 1e8; a final
/// theta above 1e6 gives `converged = false`, `theta_std_error = NaN` and a warning.
///
/// # Errors
///
/// `InvalidSpecification` for an `initial_theta` that is not positive and finite;
/// column errors (`ColumnNotFound`, `NonNumericColumn`, `NonFiniteValues`) name the
/// column; `InsufficientData` below 2 complete rows; `SingularMatrix` for collinear
/// regressors.
pub fn run_negbin(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    initial_theta: Option<f64>,
) -> EconResult<NegBinResult> {
    if let Some(t) = initial_theta
        && !(t.is_finite() && t > 0.0)
    {
        return Err(EconError::InvalidSpecification {
            message: format!("init_theta must be a positive, finite number (got {t})"),
        });
    }
    let df = dataset.df();
    let cols = ColumnRequest::new(df)
        .numeric(y_col)
        .numerics(x_cols)
        .na(NaPolicy::Omit)
        .min_rows(2, "negative binomial regression")
        .extract()?;
    let y = cols.numeric(y_col);
    let n = cols.n();
    let dm = cols.matrix(x_cols, true);
    let x = dm.view();
    let k = x.ncols();

    let y_mean = y.iter().sum::<f64>() / n as f64;
    let y_var = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>() / (n - 1) as f64;

    let mut var_names = vec!["(Intercept)".to_string()];
    var_names.extend(x_cols.iter().map(|s| s.to_string()));

    let mut work = NbWork::new(n)?;

    // Start: Poisson (or NB at init_theta) coefficients, then theta.
    let start_theta = initial_theta.unwrap_or(f64::INFINITY);
    let mut beta = irls_beta(&x, y, None, start_theta, &mut work)?;
    let mut theta = match initial_theta {
        Some(t) => t,
        None => {
            let ss: f64 = y
                .iter()
                .zip(&work.mu)
                .map(|(&yi, &mi)| (yi / mi - 1.0).powi(2))
                .sum();
            (n as f64 / ss).min(THETA_MAX)
        }
    };
    if !(theta.is_finite() && theta > 0.0) {
        theta = THETA_MAX;
    }

    let mut warnings: Vec<String> = Vec::new();
    let mut converged = false;
    let mut iterations = 0;
    let mut ll = nb_loglik(theta, &work.mu, y);
    for round in 1..=100 {
        resource::checkpoint()?;
        iterations = round;
        beta = irls_beta(&x, y, Some(&beta), theta, &mut work)?;
        let theta_new = theta_newton(theta, &work.mu, y)?;
        let ll_new = nb_loglik(theta_new, &work.mu, y);
        let theta_change = (theta_new - theta).abs() / theta;
        let ll_change = (ll_new - ll).abs();
        theta = theta_new;
        ll = ll_new;
        if theta_change < 1e-10 && ll_change < 1e-12 * ll.abs() {
            converged = true;
            break;
        }
        if theta >= THETA_MAX {
            break;
        }
    }
    if !converged && theta < THETA_MAX {
        warnings.push("the beta/theta alternation did not converge in 100 rounds".to_string());
    }
    // The coefficients at the final theta.
    beta = irls_beta(&x, y, Some(&beta), theta, &mut work)?;
    let log_likelihood = nb_loglik(theta, &work.mu, y);

    // Null log-likelihood (unchanged; ECON-64, Tier 4)
    let log_likelihood_null = n as f64
        * (theta * (theta / (theta + y_mean)).ln() + y_mean * (y_mean / (theta + y_mean)).ln());

    // Standard errors from the expected information (X'WX)^-1 (summary.negbin)
    for (wi, &mi) in work.w.iter_mut().zip(&work.mu) {
        *wi = nb_weight(mi, theta);
    }
    let (info_matrix, _) = weighted_cross_products(&x, &work.w, &work.z)?;
    let std_errors: Vec<f64> = match cholesky_inverse(&info_matrix.view()) {
        Ok(v) => (0..k).map(|i| v[[i, i]].sqrt()).collect(),
        Err(_) => {
            warnings.push(
                "X'WX is not positive definite at the estimate; standard errors are not \
                 available"
                    .to_string(),
            );
            vec![f64::NAN; k]
        }
    };
    let z_stats: Vec<f64> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| b / se)
        .collect();
    let p_values: Vec<f64> = z_stats.iter().map(|&z| two_sided_p(z)).collect();

    let pseudo_r_squared = 1.0 - log_likelihood / log_likelihood_null;
    let n_params = k + 1; // beta + theta
    let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;
    let bic = -2.0 * log_likelihood + (n_params as f64) * (n as f64).ln();

    let mut converged = converged && log_likelihood.is_finite() && theta.is_finite();
    let theta_std_error = if theta > THETA_POISSON {
        converged = false;
        warnings.push(no_overdispersion_warning(theta));
        f64::NAN
    } else {
        let info = theta_information(theta, &work.mu, y);
        if info > 0.0 {
            1.0 / info.sqrt()
        } else {
            warnings.push(
                "the information of theta is not positive at the estimate; SE(theta) is not \
                 available"
                    .to_string(),
            );
            f64::NAN
        }
    };

    Ok(NegBinResult {
        dep_var: y_col.to_string(),
        variables: var_names,
        coefficients: beta.to_vec(),
        std_errors,
        z_stats,
        p_values,
        theta,
        theta_std_error,
        log_likelihood,
        log_likelihood_null,
        pseudo_r_squared,
        aic,
        bic,
        n_obs: n,
        iterations,
        converged,
        y_mean,
        y_var,
        missing: cols.missing().clone(),
        warnings,
    })
}

/// `2 P(Z > |z|)` from the upper tail (X-17).
fn two_sided_p(z: f64) -> f64 {
    2.0 * Normal::standard().sf(z.abs())
}

// ============================================================================
// Zero-Inflated Models
// ============================================================================

/// Type of zero-inflated model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZeroInflatedType {
    Poisson,
    NegBin,
}

impl fmt::Display for ZeroInflatedType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZeroInflatedType::Poisson => write!(f, "Zero-Inflated Poisson (ZIP)"),
            ZeroInflatedType::NegBin => write!(f, "Zero-Inflated Negative Binomial (ZINB)"),
        }
    }
}

/// Result from zero-inflated count model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroInflResult {
    /// Model type
    pub model_type: ZeroInflatedType,
    /// Dependent variable name
    pub dep_var: String,
    /// Count model variable names
    pub count_variables: Vec<String>,
    /// Zero-inflation model variable names
    pub zero_variables: Vec<String>,
    /// Count model coefficients
    pub count_coefficients: Vec<f64>,
    /// Count model standard errors
    pub count_std_errors: Vec<f64>,
    /// Count model z-statistics
    pub count_z_stats: Vec<f64>,
    /// Count model p-values
    pub count_p_values: Vec<f64>,
    /// Zero-inflation coefficients (logit)
    pub zero_coefficients: Vec<f64>,
    /// Zero-inflation standard errors
    pub zero_std_errors: Vec<f64>,
    /// Zero-inflation z-statistics
    pub zero_z_stats: Vec<f64>,
    /// Zero-inflation p-values
    pub zero_p_values: Vec<f64>,
    /// Overdispersion parameter (for ZINB only)
    pub theta: Option<f64>,
    /// Log-likelihood
    pub log_likelihood: f64,
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
    /// Number of zeros
    pub n_zeros: usize,
    /// SE of log(theta) from the inverse Hessian (ZINB only; pscl's SE.logtheta).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub log_theta_std_error: Option<f64>,
    /// theta * SE(log theta) (delta method; ZINB only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub theta_std_error: Option<f64>,
    /// Rows dropped because of missing values in the outcome or a regressor (R: na.omit).
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
    /// Estimation problems in plain words (non-convergence, quasi-separation).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

impl fmt::Display for ZeroInflResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.model_type)?;
        writeln!(f, "===================================")?;
        writeln!(f, "Dependent variable: {}", self.dep_var)?;
        writeln!(
            f,
            "N = {}, Zeros = {} ({:.1}%)",
            self.n_obs,
            self.n_zeros,
            100.0 * self.n_zeros as f64 / self.n_obs as f64
        )?;
        writeln!(f)?;

        writeln!(f, "Count Model Coefficients:")?;
        writeln!(
            f,
            "{:<15} {:>10} {:>10} {:>10} {:>10}",
            "Variable", "Coef", "Std.Err", "z", "P>|z|"
        )?;
        for (i, var) in self.count_variables.iter().enumerate() {
            writeln!(
                f,
                "{:<15} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
                var,
                self.count_coefficients[i],
                self.count_std_errors[i],
                self.count_z_stats[i],
                self.count_p_values[i]
            )?;
        }
        writeln!(f)?;

        writeln!(f, "Zero-Inflation Model (logit):")?;
        for (i, var) in self.zero_variables.iter().enumerate() {
            writeln!(
                f,
                "{:<15} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
                var,
                self.zero_coefficients[i],
                self.zero_std_errors[i],
                self.zero_z_stats[i],
                self.zero_p_values[i]
            )?;
        }
        writeln!(f)?;

        if let Some(theta) = self.theta {
            writeln!(
                f,
                "Theta: {:.4} (SE: {:.4}), log(theta): {:.4} (SE: {:.4})",
                theta,
                self.theta_std_error.unwrap_or(f64::NAN),
                theta.ln(),
                self.log_theta_std_error.unwrap_or(f64::NAN)
            )?;
        }
        writeln!(f, "Log-Likelihood: {:.4}", self.log_likelihood)?;
        writeln!(f, "AIC: {:.4}, BIC: {:.4}", self.aic, self.bic)?;
        writeln!(
            f,
            "Converged: {} ({} iterations)",
            self.converged, self.iterations
        )?;
        write_notes(f, &self.warnings, &self.missing)
    }
}

/// Run Zero-Inflated Poisson (ZIP) model.
///
/// R equivalent: `pscl::zeroinfl(y ~ x | z, dist = "poisson")`, logit zero part;
/// `z_cols` defaults to `x_cols` (pscl's default). The ZIP EM runs to convergence for
/// starting values; then the joint log-likelihood is maximised by damped Newton steps
/// with pscl's analytic gradient and a central-difference Hessian, converging on the
/// gradient, the log-likelihood and the Newton step. Standard errors come from the
/// inverse Hessian at the estimate.
///
/// Rows with a missing outcome or regressor (count or zero part) are dropped (R
/// `na.omit`) and reported in `missing`. Non-convergence, a Hessian that is not
/// negative definite and fitted zero-inflation probabilities numerically 0 or 1
/// (quasi-separation) are reported in `warnings`, never as an error.
///
/// # Errors
///
/// `InvalidSpecification` when the outcome has no zero count (pscl: "minimum count is
/// not zero"); column errors (`ColumnNotFound`, `NonNumericColumn`, `NonFiniteValues`)
/// name the column.
pub fn run_zip(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    z_cols: Option<&[&str]>,
) -> EconResult<ZeroInflResult> {
    run_zeroinfl(dataset, y_col, x_cols, z_cols, ZeroInflatedType::Poisson)
}

/// Run Zero-Inflated Negative Binomial (ZINB) model.
///
/// R equivalent: `pscl::zeroinfl(y ~ x | z, dist = "negbin")`. As [`run_zip`], with the
/// NB2 count likelihood and `ln theta` estimated jointly (starting from the ZIP EM
/// values and theta = 1). `log_theta_std_error` is pscl's `SE.logtheta`,
/// `theta_std_error = theta * SE(ln theta)` (delta method). A final theta above 1e6
/// (no overdispersion) gives `converged = false`, NaN theta SEs and a warning.
///
/// # Errors
///
/// As [`run_zip`].
pub fn run_zinb(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    z_cols: Option<&[&str]>,
) -> EconResult<ZeroInflResult> {
    run_zeroinfl(dataset, y_col, x_cols, z_cols, ZeroInflatedType::NegBin)
}

/// Newton iterations of the joint zero-inflated fit.
const ZI_MAX_ITER: usize = 200;

/// Step halvings before a line search gives up.
const MAX_HALVINGS: usize = 30;

/// Levenberg damping attempts when the information is not positive definite.
const MAX_DAMPING: usize = 20;

/// The m x m matrices a Newton fit holds at once: the Hessian, its damped copy, and
/// inside `cholesky_inverse` the faer copy, the factor, the identity and the solution
/// (the covariance). Check the working set against the allocation ceiling before
/// building it (a file-private copy of the ordered models' check, T0 convention).
const DENSE_INVERSE_COPIES: usize = 6;

fn ensure_dense_inverse_fits(m: usize, what: &str) -> EconResult<()> {
    let cells = resource::checked_len(&[m, m, DENSE_INVERSE_COPIES], what)?;
    resource::check_alloc::<f64>(cells, what)
}

/// The Newton step `(-H + lambda I)^-1 g`: `lambda = 0` first; when that matrix is not
/// positive definite, `lambda = 1e-8 max(1, max diag(-H))`, ten times larger on each of
/// up to [`MAX_DAMPING`] attempts (Levenberg). `None` if every attempt fails. `work`
/// holds the damped copy.
fn newton_step(hess: &Array2<f64>, grad: &[f64], work: &mut Array2<f64>) -> Option<Array1<f64>> {
    let m = grad.len();
    let max_diag = (0..m).fold(1.0f64, |acc, i| acc.max(-hess[[i, i]]));
    let mut lambda = 0.0;
    for attempt in 0..=MAX_DAMPING {
        work.zip_mut_with(hess, |w, h| *w = -h);
        for i in 0..m {
            work[[i, i]] += lambda;
        }
        if let Ok(inv) = cholesky_inverse(&work.view()) {
            return Some(inv.dot(&ndarray::ArrayView1::from(grad)));
        }
        lambda = if attempt == 0 {
            1e-8 * max_diag
        } else {
            lambda * 10.0
        };
    }
    None
}

/// `ln(1 + e^t)` without overflow or cancellation.
fn softplus(t: f64) -> f64 {
    if t > 0.0 {
        t + (-t).exp().ln_1p()
    } else {
        t.exp().ln_1p()
    }
}

/// The logistic CDF `1 / (1 + e^-t)`, without overflow.
fn logistic(t: f64) -> f64 {
    if t >= 0.0 {
        1.0 / (1.0 + (-t).exp())
    } else {
        let e = t.exp();
        e / (1.0 + e)
    }
}

/// `ln(e^a + e^b)`.
fn log_add_exp(a: f64, b: f64) -> f64 {
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    if hi == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    hi + (lo - hi).exp().ln_1p()
}

/// The two GLMs of the zero-inflated starting values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GlmFamily {
    /// Poisson, log link.
    Poisson,
    /// Binomial, logit link (the response may be a probability, as in pscl's EM).
    Logit,
}

/// glm.fit's IRLS for a Poisson-log or binomial-logit model with prior weights, used
/// for the zero-inflated starting values and EM steps. glm.fit's guards: the logit mean
/// within `[eps, 1 - eps]` and `d mu / d eta >= eps`, the Poisson `eta <= 700`; rows
/// with prior weight 0 do not enter. Starts from `start`, or from glm's `mustart`
/// (`y + 0.1`, `(y + 0.5) / 2`). Stops when the coefficients change by less than
/// `1e-10 (1 + max |b|)`, after `max_iter` iterations, or at a singular or non-finite
/// update, which keeps the previous coefficients (zeros if there are none).
fn glm_irls(
    x: &ArrayView2<f64>,
    response: &[f64],
    prior: Option<&[f64]>,
    family: GlmFamily,
    start: Option<&Array1<f64>>,
    max_iter: usize,
) -> EconResult<Array1<f64>> {
    let n = response.len();
    let k = x.ncols();
    let mut eta = resource::try_vec(n, 0.0, "GLM linear predictor")?;
    let mut w = resource::try_vec(n, 0.0, "GLM weights")?;
    let mut z = resource::try_vec(n, 0.0, "GLM working response")?;
    let mut beta = match start {
        Some(b) => {
            for (e, row) in eta.iter_mut().zip(x.outer_iter()) {
                *e = row.dot(b);
            }
            Some(b.clone())
        }
        None => {
            for (e, &r) in eta.iter_mut().zip(response) {
                *e = match family {
                    GlmFamily::Poisson => (r + 0.1).max(0.1).ln(),
                    GlmFamily::Logit => {
                        let m = (r + 0.5) / 2.0;
                        (m / (1.0 - m)).ln()
                    }
                };
            }
            None
        }
    };
    let eps = f64::EPSILON;
    for _ in 0..max_iter {
        resource::checkpoint()?;
        for i in 0..n {
            let pw = prior.map_or(1.0, |p| p[i]);
            let (mu, mu_eta, var) = match family {
                GlmFamily::Poisson => {
                    let m = eta[i].min(700.0).exp().max(f64::MIN_POSITIVE);
                    (m, m, m)
                }
                GlmFamily::Logit => {
                    let m = logistic(eta[i]).clamp(eps, 1.0 - eps);
                    let e = (-eta[i].abs()).exp();
                    let d = (e / ((1.0 + e) * (1.0 + e))).max(eps);
                    (m, d, m * (1.0 - m))
                }
            };
            if pw > 0.0 {
                w[i] = pw * mu_eta * mu_eta / var;
                z[i] = eta[i] + (response[i] - mu) / mu_eta;
            } else {
                w[i] = 0.0;
                z[i] = 0.0;
            }
        }
        let (xtwx, xtwz) = weighted_cross_products(x, &w, &z)?;
        let Ok(inv) = cholesky_inverse(&xtwx.view()) else {
            break;
        };
        let new_beta = inv.dot(&xtwz);
        if new_beta.iter().any(|b| !b.is_finite()) {
            break;
        }
        let change = beta.as_ref().map_or(f64::INFINITY, |b| {
            new_beta
                .iter()
                .zip(b)
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()))
        });
        let scale = 1.0 + new_beta.iter().fold(0.0f64, |m, b| m.max(b.abs()));
        for (e, row) in eta.iter_mut().zip(x.outer_iter()) {
            *e = row.dot(&new_beta);
        }
        beta = Some(new_beta);
        if change < 1e-10 * scale {
            break;
        }
    }
    Ok(beta.unwrap_or_else(|| Array1::zeros(k)))
}

/// The design and data of a zero-inflated fit.
struct ZiData<'a> {
    x: ArrayView2<'a, f64>,
    z: ArrayView2<'a, f64>,
    y: &'a [f64],
    dist: ZeroInflatedType,
}

/// Log-likelihood of the zero-inflated model at `par = (beta, gamma[, ln theta])`, with
/// pscl's analytic gradient (`gradPoisson` / `gradNegBin`, logit link) in `grad`.
///
/// In log space: `ln pi = -softplus(-eta_z)`, `ln(1 - pi) = -softplus(eta_z)`,
/// `ln f0 = -mu` (Poisson) or `theta (ln theta - ln(theta + mu))` (NB). A zero count adds
/// `logaddexp(ln pi, ln(1 - pi) + ln f0)`, a positive one `ln(1 - pi) + ln f(y)`. With
/// `q = (1 - pi) f0 / dens0 = logistic(ln f0 - eta_z)`, the scores of a zero are
/// `-q mu` (Poisson) or `-q theta mu / (mu + theta)` (NB) on the count part,
/// `(1 - q) - pi` on the zero part and `theta q (ln theta - ln(mu + theta) + 1 -
/// theta / (mu + theta))` on `ln theta`; of a positive count `y - mu` or
/// `y - mu (y + theta) / (mu + theta)`, `-pi`, and `theta (digamma(y + theta) -
/// digamma(theta) + ln theta - ln(mu + theta) + 1 - (y + theta) / (mu + theta))`.
fn zi_loglik(data: &ZiData<'_>, par: &[f64], mut grad: Option<&mut [f64]>) -> EconResult<f64> {
    use statrs::function::gamma::ln_gamma;
    let kx = data.x.ncols();
    let kz = data.z.ncols();
    let beta = &par[..kx];
    let gamma = &par[kx..kx + kz];
    let negbin = data.dist == ZeroInflatedType::NegBin;
    let (theta, log_theta) = if negbin {
        let lt = par[kx + kz];
        (lt.exp(), lt)
    } else {
        (f64::INFINITY, f64::INFINITY)
    };
    let lg_theta = if negbin { ln_gamma(theta) } else { 0.0 };
    let dg_theta = if negbin { digamma(theta) } else { 0.0 };
    if let Some(g) = grad.as_deref_mut() {
        g.fill(0.0);
    }
    let mut ll = 0.0;
    for (i, ((xr, zr), &yi)) in data
        .x
        .outer_iter()
        .zip(data.z.outer_iter())
        .zip(data.y)
        .enumerate()
    {
        if i % 1024 == 0 {
            resource::checkpoint()?;
        }
        let eta_c = xr
            .iter()
            .zip(beta)
            .map(|(a, b)| a * b)
            .sum::<f64>()
            .min(700.0);
        let mu = eta_c.exp();
        let eta_z: f64 = zr.iter().zip(gamma).map(|(a, b)| a * b).sum();
        let ln_pi = -softplus(-eta_z);
        let ln_1m_pi = -softplus(eta_z);
        let pi = logistic(eta_z);
        let ln_theta_mu = if negbin { (theta + mu).ln() } else { 0.0 };
        let (s_count, s_zero, s_theta);
        if yi < 0.5 {
            let ln_f0 = if negbin {
                theta * (log_theta - ln_theta_mu)
            } else {
                -mu
            };
            ll += log_add_exp(ln_pi, ln_1m_pi + ln_f0);
            let q = logistic(ln_f0 - eta_z);
            s_count = if negbin {
                -q * theta * mu / (mu + theta)
            } else {
                -q * mu
            };
            s_zero = (1.0 - q) - pi;
            s_theta = if negbin {
                theta * q * (log_theta - ln_theta_mu + 1.0 - theta / (mu + theta))
            } else {
                0.0
            };
        } else {
            let ln_f = if negbin {
                ln_gamma(yi + theta) - lg_theta - ln_gamma(yi + 1.0)
                    + theta * log_theta
                    + yi * eta_c
                    - (theta + yi) * ln_theta_mu
            } else {
                -mu + yi * eta_c - ln_gamma(yi + 1.0)
            };
            ll += ln_1m_pi + ln_f;
            s_count = if negbin {
                yi - mu * (yi + theta) / (mu + theta)
            } else {
                yi - mu
            };
            s_zero = -pi;
            s_theta = if negbin {
                theta
                    * (digamma(yi + theta) - dg_theta + log_theta - ln_theta_mu + 1.0
                        - (yi + theta) / (mu + theta))
            } else {
                0.0
            };
        }
        if let Some(g) = grad.as_deref_mut() {
            for (gk, &xk) in g[..kx].iter_mut().zip(xr.iter()) {
                *gk += s_count * xk;
            }
            for (gk, &zk) in g[kx..kx + kz].iter_mut().zip(zr.iter()) {
                *gk += s_zero * zk;
            }
            if negbin {
                g[kx + kz] += s_theta;
            }
        }
    }
    Ok(ll)
}

/// The symmetrised central-difference Hessian of the analytic gradient of
/// [`zi_loglik`] at `par` (step `1e-5 max(1, |par_k|)`), into `hess`.
fn zi_hessian(
    data: &ZiData<'_>,
    par: &[f64],
    hess: &mut Array2<f64>,
    g_up: &mut [f64],
    g_dn: &mut [f64],
    shifted: &mut [f64],
) -> EconResult<()> {
    let m = par.len();
    for kk in 0..m {
        let h = 1e-5 * par[kk].abs().max(1.0);
        shifted.copy_from_slice(par);
        shifted[kk] = par[kk] + h;
        zi_loglik(data, shifted, Some(g_up))?;
        shifted[kk] = par[kk] - h;
        zi_loglik(data, shifted, Some(g_dn))?;
        for a in 0..m {
            hess[[a, kk]] = (g_up[a] - g_dn[a]) / (2.0 * h);
        }
    }
    for a in 0..m {
        for b in 0..a {
            let v = 0.5 * (hess[[a, b]] + hess[[b, a]]);
            hess[[a, b]] = v;
            hess[[b, a]] = v;
        }
    }
    Ok(())
}

/// The ZIP EM of pscl (`EM = TRUE`) to convergence, for starting values: E-step
/// `w_i = pi_i / (pi_i + (1 - pi_i) e^-mu_i)` for zeros (0 otherwise), M-steps a
/// logistic IRLS with response `w` and a Poisson IRLS with weights `1 - w`, each from
/// the current values. Stops when the relative log-likelihood change is below 1e-8,
/// after 500 rounds, or when the log-likelihood would fall or not be finite (keeping
/// the previous round). Returns `(beta, gamma)`.
fn zip_em(data: &ZiData<'_>, zero: &[f64]) -> EconResult<(Array1<f64>, Array1<f64>)> {
    let n = data.y.len();
    let kx = data.x.ncols();
    let kz = data.z.ncols();
    let poisson = ZiData {
        x: data.x,
        z: data.z,
        y: data.y,
        dist: ZeroInflatedType::Poisson,
    };
    let mut beta = glm_irls(&data.x, data.y, None, GlmFamily::Poisson, None, 25)?;
    let mut gamma = glm_irls(&data.z, zero, None, GlmFamily::Logit, None, 25)?;
    let mut par = resource::try_vec(kx + kz, 0.0, "zero-inflated parameters")?;
    let pack = |par: &mut [f64], b: &Array1<f64>, g: &Array1<f64>| {
        for (p, v) in par.iter_mut().zip(b.iter().chain(g.iter())) {
            *p = *v;
        }
    };
    pack(&mut par, &beta, &gamma);
    let mut ll = zi_loglik(&poisson, &par, None)?;
    if !ll.is_finite() {
        gamma.fill(0.0);
        pack(&mut par, &beta, &gamma);
        ll = zi_loglik(&poisson, &par, None)?;
    }
    let mut w = resource::try_vec(n, 0.0, "EM zero-state probabilities")?;
    let mut count_weight = resource::try_vec(n, 0.0, "EM count weights")?;
    for _ in 0..500 {
        resource::checkpoint()?;
        for (i, ((xr, zr), &yi)) in data
            .x
            .outer_iter()
            .zip(data.z.outer_iter())
            .zip(data.y)
            .enumerate()
        {
            w[i] = if yi < 0.5 {
                // pi / (pi + (1 - pi) e^-mu) = logistic(eta_z + mu)
                let mu = xr.dot(&beta).min(700.0).exp();
                logistic(zr.dot(&gamma) + mu)
            } else {
                0.0
            };
            count_weight[i] = 1.0 - w[i];
        }
        let gamma_new = glm_irls(&data.z, &w, None, GlmFamily::Logit, Some(&gamma), 25)?;
        let beta_new = glm_irls(
            &data.x,
            data.y,
            Some(&count_weight),
            GlmFamily::Poisson,
            Some(&beta),
            25,
        )?;
        pack(&mut par, &beta_new, &gamma_new);
        let ll_new = zi_loglik(&poisson, &par, None)?;
        if !(ll_new.is_finite() && ll_new >= ll) {
            break;
        }
        let rel = (ll_new - ll).abs() / ll.abs();
        beta = beta_new;
        gamma = gamma_new;
        ll = ll_new;
        if rel < 1e-8 {
            break;
        }
    }
    Ok((beta, gamma))
}

fn run_zeroinfl(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    z_cols: Option<&[&str]>,
    model_type: ZeroInflatedType,
) -> EconResult<ZeroInflResult> {
    let df = dataset.df();
    // Use the same covariates for zero-inflation if not specified (pscl's default)
    let zero_cols = z_cols.unwrap_or(x_cols);
    let cols = ColumnRequest::new(df)
        .numeric(y_col)
        .numerics(x_cols)
        .numerics(zero_cols)
        .na(NaPolicy::Omit)
        .extract()?;
    let y = cols.numeric(y_col);
    let n = cols.n();
    let dm_count = cols.matrix(x_cols, true);
    let dm_zero = cols.matrix(zero_cols, true);
    let x_count = dm_count.view();
    let x_zero = dm_zero.view();
    let k_count = x_count.ncols();
    let k_zero = x_zero.ncols();

    // Zeros keep the rule y < 0.5 (ECON-58, Tier 4).
    let mut zero = resource::try_vec(n, 0.0, "zero indicator")?;
    for (z, &yi) in zero.iter_mut().zip(y) {
        *z = if yi < 0.5 { 1.0 } else { 0.0 };
    }
    let n_zeros = zero.iter().filter(|&&z| z > 0.5).count();
    if n_zeros == 0 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "a zero-inflated model needs at least one zero count in '{y_col}' \
                 (pscl::zeroinfl: minimum count is not zero)"
            ),
        });
    }

    let mut count_vars = vec!["(Intercept)".to_string()];
    count_vars.extend(x_cols.iter().map(|s| s.to_string()));
    let mut zero_vars = vec!["(Intercept)".to_string()];
    zero_vars.extend(zero_cols.iter().map(|s| s.to_string()));

    let data = ZiData {
        x: x_count,
        z: x_zero,
        y,
        dist: model_type,
    };
    let negbin = model_type == ZeroInflatedType::NegBin;
    let n_params = k_count + k_zero + usize::from(negbin);
    let hessian_what = match model_type {
        ZeroInflatedType::Poisson => "zero-inflated Poisson Hessian",
        ZeroInflatedType::NegBin => "zero-inflated negative binomial Hessian",
    };
    ensure_dense_inverse_fits(n_params, hessian_what)?;
    let mut hess = resource::try_zeros2(n_params, n_params, hessian_what)?;
    let mut work = resource::try_zeros2(n_params, n_params, hessian_what)?;
    let mut grad = resource::try_vec(n_params, 0.0, "zero-inflated score")?;
    let mut g_up = resource::try_vec(n_params, 0.0, "zero-inflated score")?;
    let mut g_dn = resource::try_vec(n_params, 0.0, "zero-inflated score")?;
    let mut shifted = resource::try_vec(n_params, 0.0, "zero-inflated parameters")?;

    // Starting values: the ZIP EM to convergence; ZINB starts there with theta = 1.
    let (beta0, gamma0) = zip_em(&data, &zero)?;
    let mut par = resource::try_vec(n_params, 0.0, "zero-inflated parameters")?;
    for (p, v) in par.iter_mut().zip(beta0.iter().chain(gamma0.iter())) {
        *p = *v;
    }
    let log_theta_max = THETA_MAX.ln();
    let mut trial = par.clone();

    let mut warnings: Vec<String> = Vec::new();
    let mut converged = false;
    let mut stopped = false;
    let mut iterations = 0;
    let mut ll = zi_loglik(&data, &par, Some(&mut grad))?;
    let mut ll_prev = f64::NAN;
    for iter in 1..=ZI_MAX_ITER {
        resource::checkpoint()?;
        iterations = iter;
        zi_hessian(&data, &par, &mut hess, &mut g_up, &mut g_dn, &mut shifted)?;
        if !(ll.is_finite()
            && grad.iter().all(|g| g.is_finite())
            && hess.iter().all(|h| h.is_finite()))
        {
            warnings.push(format!(
                "the log-likelihood, gradient or Hessian is not finite at iteration {iter}; \
                 estimation stopped"
            ));
            stopped = true;
            break;
        }
        let Some(delta) = newton_step(&hess, &grad, &mut work) else {
            warnings.push(format!(
                "the Hessian is not negative definite at iteration {iter}, even with \
                 damping; estimation stopped"
            ));
            stopped = true;
            break;
        };
        if delta.iter().any(|d| !d.is_finite()) {
            warnings.push(format!(
                "the Newton step is not finite at iteration {iter}; estimation stopped"
            ));
            stopped = true;
            break;
        }
        let max_grad = grad.iter().fold(0.0f64, |m, g| m.max(g.abs()));
        let rel_step = delta
            .iter()
            .zip(&par)
            .fold(0.0f64, |m, (d, p)| m.max(d.abs() / (1.0 + p.abs())));
        if iter > 1
            && max_grad < 1e-8 * (1.0 + ll.abs())
            && (ll - ll_prev).abs() < 1e-12 * ll_prev.abs()
            && rel_step <= 1e-10
        {
            converged = true;
            break;
        }

        let mut accepted = false;
        let mut s = 1.0;
        for _ in 0..=MAX_HALVINGS {
            for ((t, p), d) in trial.iter_mut().zip(&par).zip(&delta) {
                *t = p + s * d;
            }
            if negbin {
                trial[n_params - 1] = trial[n_params - 1].min(log_theta_max);
            }
            let ll_trial = zi_loglik(&data, &trial, None)?;
            if ll_trial.is_finite() && ll_trial >= ll - 1e-12 * ll.abs() {
                accepted = true;
                break;
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
        std::mem::swap(&mut par, &mut trial);
        ll_prev = ll;
        ll = zi_loglik(&data, &par, Some(&mut grad))?;
    }
    if !converged && !stopped {
        let max_grad = grad.iter().fold(0.0f64, |m, g| m.max(g.abs()));
        warnings.push(format!(
            "Newton's method did not converge in {ZI_MAX_ITER} iterations (max |gradient| = \
             {max_grad:.3e}); the estimates may be unreliable"
        ));
    }
    let log_likelihood = ll;
    let mut converged =
        converged && log_likelihood.is_finite() && par.iter().all(|p| p.is_finite());

    // glm.fit's test for fitted probabilities numerically 0 or 1 (pscl's EM suppresses it).
    let tiny = 10.0 * f64::EPSILON;
    let separated = x_zero.outer_iter().any(|zr| {
        let pi = logistic(
            zr.iter()
                .zip(&par[k_count..k_count + k_zero])
                .map(|(a, b)| a * b)
                .sum(),
        );
        pi.min(1.0 - pi) < tiny
    });
    if separated {
        warnings.push(
            "fitted zero-inflation probabilities numerically 0 or 1 occurred (quasi-separation \
             in the zero part); its estimates and standard errors are not reliable"
                .to_string(),
        );
    }

    // Standard errors from the inverse Hessian at the final estimate.
    zi_hessian(&data, &par, &mut hess, &mut g_up, &mut g_dn, &mut shifted)?;
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
                "the Hessian is not negative definite at the estimate; standard errors are \
                 not available"
                    .to_string(),
            );
            vec![f64::NAN; n_params]
        }
    };
    drop(covariance);

    let beta: Vec<f64> = par[..k_count].to_vec();
    let gamma: Vec<f64> = par[k_count..k_count + k_zero].to_vec();
    let count_std_errors: Vec<f64> = se_all[..k_count].to_vec();
    let zero_std_errors: Vec<f64> = se_all[k_count..k_count + k_zero].to_vec();
    let count_z_stats: Vec<f64> = beta
        .iter()
        .zip(&count_std_errors)
        .map(|(b, se)| b / se)
        .collect();
    let count_p_values: Vec<f64> = count_z_stats.iter().map(|&z| two_sided_p(z)).collect();
    let zero_z_stats: Vec<f64> = gamma
        .iter()
        .zip(&zero_std_errors)
        .map(|(g, se)| g / se)
        .collect();
    let zero_p_values: Vec<f64> = zero_z_stats.iter().map(|&z| two_sided_p(z)).collect();

    let (theta, log_theta_std_error, theta_std_error) = if negbin {
        let theta = par[n_params - 1].exp();
        let se_log = se_all[n_params - 1];
        let mut se_theta = theta * se_log;
        let mut se_log = se_log;
        if theta > THETA_POISSON {
            converged = false;
            se_log = f64::NAN;
            se_theta = f64::NAN;
            warnings.push(no_overdispersion_warning(theta));
        }
        (Some(theta), Some(se_log), Some(se_theta))
    } else {
        (None, None, None)
    };

    let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;
    let bic = -2.0 * log_likelihood + (n_params as f64) * (n as f64).ln();

    Ok(ZeroInflResult {
        model_type,
        dep_var: y_col.to_string(),
        count_variables: count_vars,
        zero_variables: zero_vars,
        count_coefficients: beta,
        count_std_errors,
        count_z_stats,
        count_p_values,
        zero_coefficients: gamma,
        zero_std_errors,
        zero_z_stats,
        zero_p_values,
        theta,
        log_likelihood,
        aic,
        bic,
        iterations,
        converged,
        n_obs: n,
        n_zeros,
        log_theta_std_error,
        theta_std_error,
        missing: cols.missing().clone(),
        warnings,
    })
}

// ============================================================================
// Hurdle Models
// ============================================================================

/// Type of hurdle model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HurdleType {
    Poisson,
    NegBin,
}

impl fmt::Display for HurdleType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HurdleType::Poisson => write!(f, "Hurdle Poisson"),
            HurdleType::NegBin => write!(f, "Hurdle Negative Binomial"),
        }
    }
}

/// Result from hurdle model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HurdleResult {
    /// Model type
    pub model_type: HurdleType,
    /// Dependent variable name
    pub dep_var: String,
    /// Binary part variable names
    pub binary_variables: Vec<String>,
    /// Binary part coefficients
    pub binary_coefficients: Vec<f64>,
    /// Binary part standard errors
    pub binary_std_errors: Vec<f64>,
    /// Binary part z-statistics
    pub binary_z_stats: Vec<f64>,
    /// Binary part p-values
    pub binary_p_values: Vec<f64>,
    /// Count part variable names
    pub count_variables: Vec<String>,
    /// Count part coefficients
    pub count_coefficients: Vec<f64>,
    /// Count part standard errors
    pub count_std_errors: Vec<f64>,
    /// Count part z-statistics
    pub count_z_stats: Vec<f64>,
    /// Count part p-values
    pub count_p_values: Vec<f64>,
    /// Overdispersion parameter (for NegBin only)
    pub theta: Option<f64>,
    /// Total log-likelihood
    pub log_likelihood: f64,
    /// Binary part log-likelihood
    pub ll_binary: f64,
    /// Count part log-likelihood
    pub ll_count: f64,
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
    /// Number of zeros
    pub n_zeros: usize,
    /// Number of positive observations
    pub n_positive: usize,
    /// Rows dropped because of missing values in the outcome or a regressor (R: na.omit).
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
}

impl fmt::Display for HurdleResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.model_type)?;
        writeln!(f, "================================")?;
        writeln!(f, "Dependent variable: {}", self.dep_var)?;
        writeln!(
            f,
            "N = {}, Zeros = {}, Positive = {}",
            self.n_obs, self.n_zeros, self.n_positive
        )?;
        writeln!(f)?;

        writeln!(f, "Binary Part (logit: y > 0):")?;
        writeln!(
            f,
            "{:<15} {:>10} {:>10} {:>10} {:>10}",
            "Variable", "Coef", "Std.Err", "z", "P>|z|"
        )?;
        for (i, var) in self.binary_variables.iter().enumerate() {
            writeln!(
                f,
                "{:<15} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
                var,
                self.binary_coefficients[i],
                self.binary_std_errors[i],
                self.binary_z_stats[i],
                self.binary_p_values[i]
            )?;
        }
        writeln!(f)?;

        writeln!(f, "Count Part (truncated):")?;
        for (i, var) in self.count_variables.iter().enumerate() {
            writeln!(
                f,
                "{:<15} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
                var,
                self.count_coefficients[i],
                self.count_std_errors[i],
                self.count_z_stats[i],
                self.count_p_values[i]
            )?;
        }
        writeln!(f)?;

        if let Some(theta) = self.theta {
            writeln!(f, "Theta: {:.4}", theta)?;
        }
        writeln!(f, "Log-Likelihood: {:.4}", self.log_likelihood)?;
        writeln!(f, "AIC: {:.4}, BIC: {:.4}", self.aic, self.bic)?;
        writeln!(
            f,
            "Converged: {} ({} iterations)",
            self.converged, self.iterations
        )?;
        write_notes(f, &[], &self.missing)
    }
}

/// Run hurdle model for count data with excess zeros.
///
/// R equivalent: `pscl::hurdle()`. Rows with a missing outcome or regressor (count or
/// binary part) are dropped from both parts (R `na.omit`) and reported in `missing`.
/// The estimates are not yet pscl's for the NB hurdle theta and the count-part
/// standard errors (ECON-16).
pub fn run_hurdle(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    z_cols: Option<&[&str]>,
    model_type: HurdleType,
) -> EconResult<HurdleResult> {
    let df = dataset.df();
    let binary_cols = z_cols.unwrap_or(x_cols);
    let cols = ColumnRequest::new(df)
        .numeric(y_col)
        .numerics(x_cols)
        .numerics(binary_cols)
        .na(NaPolicy::Omit)
        .extract()?;
    let y = cols.numeric(y_col);
    let n = cols.n();

    // Create binary indicator
    let y_binary: Vec<f64> = y
        .iter()
        .map(|&yi| if yi > 0.0 { 1.0 } else { 0.0 })
        .collect();

    // Separate positive observations
    let positive_indices: Vec<usize> = y
        .iter()
        .enumerate()
        .filter(|(_, yi)| **yi > 0.0)
        .map(|(i, _)| i)
        .collect();

    let n_zeros = n - positive_indices.len();
    let n_positive = positive_indices.len();

    if n_positive < 3 {
        return Err(EconError::InsufficientData {
            required: 3,
            provided: n_positive,
            context: "Hurdle model requires at least 3 positive observations".to_string(),
        });
    }

    // Build design matrices
    let x_full = cols.matrix(x_cols, true).data;
    let k_count = x_full.ncols();

    let x_binary = cols.matrix(binary_cols, true).data;
    let k_binary = x_binary.ncols();

    // Part 1: Binary logit model
    let (beta_binary, ll_binary, converged_binary, iter_binary) =
        fit_logit_model(&y_binary, &x_binary, 50, 1e-8)?;

    // Compute binary standard errors
    let pi_hat: Vec<f64> = (0..n)
        .map(|i| {
            let xb: f64 = (0..k_binary)
                .map(|j| x_binary[[i, j]] * beta_binary[j])
                .sum();
            logistic_cdf(xb)
        })
        .collect();

    let binary_info = compute_logit_information(&x_binary.view(), &pi_hat);
    let binary_vcov = match safe_inverse(&binary_info.view()) {
        Ok((inv, _)) => inv,
        Err(_) => Array2::eye(k_binary) * 1e-6,
    };
    let binary_std_errors: Vec<f64> = (0..k_binary)
        .map(|i| binary_vcov[[i, i]].max(1e-10).sqrt())
        .collect();

    // Part 2: Truncated count model (positive y only)
    let y_positive: Vec<f64> = positive_indices.iter().map(|&i| y[i]).collect();
    let x_positive: Array2<f64> = Array2::from_shape_fn((n_positive, k_count), |(i, j)| {
        x_full[[positive_indices[i], j]]
    });

    let (beta_count, theta, ll_count, converged_count, iter_count) =
        fit_truncated_count_model(&y_positive, &x_positive, model_type)?;

    // Compute count standard errors
    let count_info =
        compute_truncated_count_information(&y_positive, &x_positive.view(), &beta_count, theta);
    let count_vcov = match safe_inverse(&count_info.view()) {
        Ok((inv, _)) => inv,
        Err(_) => Array2::eye(k_count) * 1e-6,
    };
    let count_std_errors: Vec<f64> = (0..k_count)
        .map(|i| count_vcov[[i, i]].max(1e-10).sqrt())
        .collect();

    // Z-statistics and p-values
    let binary_z_stats: Vec<f64> = beta_binary
        .iter()
        .zip(binary_std_errors.iter())
        .map(|(b, se)| b / se)
        .collect();
    let binary_p_values: Vec<f64> = binary_z_stats
        .iter()
        .map(|&z| 2.0 * (1.0 - normal_cdf(z.abs())))
        .collect();

    let count_z_stats: Vec<f64> = beta_count
        .iter()
        .zip(count_std_errors.iter())
        .map(|(b, se)| b / se)
        .collect();
    let count_p_values: Vec<f64> = count_z_stats
        .iter()
        .map(|&z| 2.0 * (1.0 - normal_cdf(z.abs())))
        .collect();

    let log_likelihood = ll_binary + ll_count;
    let n_params = k_binary
        + k_count
        + if model_type == HurdleType::NegBin {
            1
        } else {
            0
        };
    let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;
    let bic = -2.0 * log_likelihood + (n_params as f64) * (n as f64).ln();

    let mut binary_variables = vec!["(Intercept)".to_string()];
    binary_variables.extend(binary_cols.iter().map(|s| s.to_string()));

    let mut count_variables = vec!["(Intercept)".to_string()];
    count_variables.extend(x_cols.iter().map(|s| s.to_string()));

    let converged = converged_binary && converged_count;
    let iterations = iter_binary + iter_count;

    Ok(HurdleResult {
        model_type,
        dep_var: y_col.to_string(),
        binary_variables,
        binary_coefficients: beta_binary,
        binary_std_errors,
        binary_z_stats,
        binary_p_values,
        count_variables,
        count_coefficients: beta_count,
        count_std_errors,
        count_z_stats,
        count_p_values,
        theta: if model_type == HurdleType::NegBin {
            Some(theta)
        } else {
            None
        },
        log_likelihood,
        ll_binary,
        ll_count,
        aic,
        bic,
        iterations,
        converged,
        n_obs: n,
        n_zeros,
        n_positive,
        missing: cols.missing().clone(),
    })
}

// Helper functions for hurdle models

fn fit_logit_model(
    y: &[f64],
    x: &Array2<f64>,
    max_iter: usize,
    tol: f64,
) -> EconResult<(Vec<f64>, f64, bool, usize)> {
    let n = y.len();
    let k = x.ncols();
    let mut beta = vec![0.0; k];
    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..max_iter {
        iterations = iter + 1;

        let mut ll = 0.0;
        let mut gradient = vec![0.0; k];
        let mut hessian = vec![vec![0.0; k]; k];

        for i in 0..n {
            let xb: f64 = (0..k).map(|j| x[[i, j]] * beta[j]).sum();
            let pi = logistic_cdf(xb);
            let yi = y[i];

            ll += yi * pi.max(1e-15).ln() + (1.0 - yi) * (1.0 - pi).max(1e-15).ln();

            let error = yi - pi;
            for j in 0..k {
                gradient[j] += error * x[[i, j]];
            }

            let w = pi * (1.0 - pi);
            for j in 0..k {
                for l in 0..k {
                    hessian[j][l] -= w * x[[i, j]] * x[[i, l]];
                }
            }
        }

        let hess_arr = Array2::from_shape_fn((k, k), |(i, j)| hessian[i][j]);
        let delta = match safe_inverse(&hess_arr.view()) {
            Ok((inv, _)) => {
                let grad_arr: Array1<f64> = gradient.iter().cloned().collect();
                let d = inv.dot(&grad_arr);
                d.iter().map(|&x| -x).collect::<Vec<f64>>()
            }
            Err(_) => gradient.iter().map(|&g| 0.01 * g).collect(),
        };

        let mut max_change = 0.0f64;
        for j in 0..k {
            beta[j] += delta[j];
            max_change = max_change.max(delta[j].abs());
        }

        if max_change < tol {
            converged = true;
            break;
        }
    }

    // Final log-likelihood
    let mut ll = 0.0;
    for i in 0..n {
        let xb: f64 = (0..k).map(|j| x[[i, j]] * beta[j]).sum();
        let pi = logistic_cdf(xb);
        ll += y[i] * pi.max(1e-15).ln() + (1.0 - y[i]) * (1.0 - pi).max(1e-15).ln();
    }

    Ok((beta, ll, converged, iterations))
}

fn compute_logit_information(x: &ArrayView2<f64>, pi: &[f64]) -> Array2<f64> {
    let n = pi.len();
    let k = x.ncols();
    let mut info = Array2::zeros((k, k));

    for i in 0..n {
        let w = pi[i] * (1.0 - pi[i]);
        for j in 0..k {
            for l in 0..k {
                info[[j, l]] += w * x[[i, j]] * x[[i, l]];
            }
        }
    }

    info
}

fn fit_truncated_count_model(
    y: &[f64],
    x: &Array2<f64>,
    model_type: HurdleType,
) -> EconResult<(Vec<f64>, f64, f64, bool, usize)> {
    let n = y.len();
    let k = x.ncols();

    let y_mean = y.iter().sum::<f64>() / n as f64;
    let mut beta: Array1<f64> = Array1::zeros(k);
    beta[0] = y_mean.max(0.1).ln();
    let mut theta = 1.0;

    let max_iter = 50;
    let tol = 1e-6;
    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..max_iter {
        iterations = iter + 1;

        let xb = x.dot(&beta);
        let mu: Vec<f64> = xb.iter().map(|&v| v.exp()).collect();

        // Compute gradient
        let mut gradient = Array1::zeros(k);
        let mut weights = Array1::zeros(n);

        for i in 0..n {
            let yi = y[i];
            let mui = mu[i];

            let (score_i, weight_i) = match model_type {
                HurdleType::Poisson => {
                    let p0 = (-mui).exp();
                    let adj = mui * p0 / (1.0 - p0);
                    (yi - mui - adj, mui + adj * (1.0 + adj / (1.0 - p0)))
                }
                HurdleType::NegBin => {
                    let ratio = theta / (theta + mui);
                    let p0 = ratio.powf(theta);
                    let score_nb = (yi - mui) * ratio;
                    let adj = p0 / (1.0 - p0) * mui * ratio;
                    (score_nb - adj, mui * ratio * (1.0 + adj / (1.0 - p0)))
                }
            };

            for j in 0..k {
                gradient[j] += score_i * x[[i, j]];
            }
            weights[i] = weight_i;
        }

        // Compute Hessian
        let mut wx = x.to_owned();
        for i in 0..n {
            let w_sqrt = weights[i].sqrt();
            for j in 0..k {
                wx[[i, j]] *= w_sqrt;
            }
        }
        let neg_hessian = wx.t().dot(&wx);

        let delta = match safe_inverse(&neg_hessian.view()) {
            Ok((inv, _)) => inv.dot(&gradient),
            Err(_) => gradient.mapv(|g| 0.01 * g),
        };

        let max_change = delta.iter().map(|&d| d.abs()).fold(0.0f64, f64::max);
        beta = &beta + &delta;

        // Update theta for NegBin
        if model_type == HurdleType::NegBin && iter % 2 == 0 {
            let mu: Vec<f64> = x.dot(&beta).iter().map(|&v| v.exp()).collect();
            let y_var: f64 =
                y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>() / (n - 1) as f64;
            let mu_mean: f64 = mu.iter().sum::<f64>() / n as f64;
            let excess_var = y_var - mu_mean;
            if excess_var > 0.0 {
                let new_theta = mu_mean.powi(2) / excess_var;
                theta = (0.7 * theta + 0.3 * new_theta.clamp(0.1, 100.0)).clamp(0.01, 1000.0);
            }
        }

        if max_change < tol {
            converged = true;
            break;
        }
    }

    // Final log-likelihood
    let mu: Vec<f64> = x.dot(&beta).iter().map(|&v| v.exp()).collect();

    let ll = match model_type {
        HurdleType::Poisson => truncated_poisson_loglik(y, &mu),
        HurdleType::NegBin => truncated_negbin_loglik(y, &mu, theta),
    };

    Ok((beta.to_vec(), theta, ll, converged, iterations))
}

fn compute_truncated_count_information(
    y: &[f64],
    x: &ArrayView2<f64>,
    beta: &[f64],
    _theta: f64,
) -> Array2<f64> {
    let n = y.len();
    let k = x.ncols();

    let mu: Vec<f64> = (0..n)
        .map(|i| {
            let xb: f64 = (0..k).map(|j| x[[i, j]] * beta[j]).sum();
            xb.exp()
        })
        .collect();

    let mut info = Array2::zeros((k, k));
    for i in 0..n {
        let mui = mu[i];
        let p0 = (-mui).exp();
        let weight = mui + mui * p0 / (1.0 - p0) * (1.0 + mui * p0 / (1.0 - p0));

        for j in 0..k {
            for l in 0..k {
                info[[j, l]] += weight * x[[i, j]] * x[[i, l]];
            }
        }
    }

    info
}

fn truncated_poisson_loglik(y: &[f64], mu: &[f64]) -> f64 {
    use statrs::function::gamma::ln_gamma;
    let n = y.len();
    let mut ll = 0.0;

    for i in 0..n {
        let yi = y[i];
        let mui = mu[i];
        let log_py = -mui + yi * mui.ln() - ln_gamma(yi + 1.0);
        let log_p_positive = (1.0 - (-mui).exp()).ln();
        ll += log_py - log_p_positive;
    }

    ll
}

fn truncated_negbin_loglik(y: &[f64], mu: &[f64], theta: f64) -> f64 {
    use statrs::function::gamma::ln_gamma;
    let n = y.len();
    let mut ll = 0.0;

    for i in 0..n {
        let yi = y[i];
        let mui = mu[i];
        let log_pnb = ln_gamma(yi + theta) - ln_gamma(theta) - ln_gamma(yi + 1.0)
            + theta * (theta / (theta + mui)).ln()
            + yi * (mui / (theta + mui)).ln();
        let p0 = (theta / (theta + mui)).powf(theta);
        let log_p_positive = (1.0 - p0).ln();
        ll += log_pnb - log_p_positive;
    }

    ll
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{self, Budget, CancelToken};
    use crate::test_support::load_case;
    use p2a_validation::{Check, Tol};
    use polars::prelude::*;

    /// Case data for tests that are not R-pinned (not registered).
    fn negbin_data() -> Dataset {
        load_case("disc_negbin").1
    }

    /// Case data for tests that are not R-pinned (not registered).
    fn zi_data() -> Dataset {
        load_case("disc_zeroinfl").1
    }

    /// A text outcome (`y_label`) with numeric regressors (not registered).
    fn ordered_text_data() -> Dataset {
        load_case("disc_ordered").1
    }

    fn with_column(ds: &Dataset, col: Column) -> Dataset {
        let mut df = ds.df().clone();
        df.with_column(col).unwrap();
        Dataset::new(df)
    }

    /// Every quantity of one zeroinfl fit; `p` is "zip_" or "zinb_".
    fn check_zi(case: &p2a_validation::Case, p: &str, r: &ZeroInflResult) {
        let n = r.n_obs;
        assert!(r.converged, "{p}: {:?}", r.warnings);
        assert!(r.warnings.is_empty(), "{p}: {:?}", r.warnings);
        let k = |s: &str| format!("{p}{s}");
        Check::new(case)
            .named(
                &k("count"),
                &r.count_variables,
                &r.count_coefficients,
                Tol::iterative(1e-6),
            )
            .named(
                &k("zero"),
                &r.zero_variables,
                &r.zero_coefficients,
                Tol::iterative(1e-6),
            )
            .vector(&k("se_count"), &r.count_std_errors, Tol::iterative(1e-5))
            .vector(&k("se_zero"), &r.zero_std_errors, Tol::iterative(1e-5))
            .vector(&k("z_count"), &r.count_z_stats, Tol::iterative(1e-5))
            .vector(&k("z_zero"), &r.zero_z_stats, Tol::iterative(1e-5))
            // p of an iterative estimate: d ln p / d ln z ~ z^2 (<= 130 here) times the z
            // tolerance
            .vector(&k("p_count"), &r.count_p_values, Tol::iterative(2e-3))
            .vector(&k("p_zero"), &r.zero_p_values, Tol::iterative(2e-3))
            .scalar(&k("loglik"), r.log_likelihood, Tol::stat(n))
            .scalar(&k("aic"), r.aic, Tol::stat(n))
            .scalar(&k("bic"), r.bic, Tol::stat(n))
            .scalar(&k("n"), n as f64, Tol::exact())
            .scalar(&k("n_zeros"), r.n_zeros as f64, Tol::exact())
            // pscl's own BFGS optimum (max |grad| ~ 7e-6): a looser cross-check
            .named(
                &k("count_pkg"),
                &r.count_variables,
                &r.count_coefficients,
                Tol::iterative(1e-4),
            )
            .named(
                &k("zero_pkg"),
                &r.zero_variables,
                &r.zero_coefficients,
                Tol::iterative(1e-4),
            )
            .vector(
                &k("se_count_pkg"),
                &r.count_std_errors,
                Tol::iterative(1e-4),
            )
            .vector(&k("se_zero_pkg"), &r.zero_std_errors, Tol::iterative(1e-4))
            .finish();
    }

    fn cancelled() -> Budget {
        let t = CancelToken::new();
        t.cancel();
        Budget::new(None, t)
    }

    fn create_count_dataset() -> Dataset {
        let df = df! {
            "y" => [0.0, 1.0, 0.0, 2.0, 3.0, 1.0, 5.0, 4.0, 7.0, 8.0, 2.0, 6.0],
            "x" => [1.0, 2.0, 1.5, 3.0, 4.0, 2.5, 5.0, 4.5, 6.0, 7.0, 3.5, 5.5]
        }
        .unwrap();
        Dataset::new(df)
    }

    /// Underdispersed data (variance below the mean): the likelihood rises towards the
    /// Poisson limit. R's glm.nb returns theta 106936.84 with "iteration limit reached"
    /// (T2-WP10 Appendix B.3); here theta runs to its bound and the fit is flagged.
    #[test]
    fn test_negbin_without_overdispersion_is_flagged() {
        let dataset = create_count_dataset();
        let r = run_negbin(&dataset, "y", &["x"], None).unwrap();
        assert_eq!(r.n_obs, 12);
        assert!(!r.converged, "{r}");
        assert!(r.theta > 1e6, "theta {}", r.theta);
        assert!(
            r.theta_std_error.is_nan(),
            "SE(theta) {}",
            r.theta_std_error
        );
        assert!(
            r.warnings.iter().any(|w| w.contains("overdispersion")),
            "{:?}",
            r.warnings
        );
    }

    // ---- negative binomial against MASS::glm.nb (disc_negbin) -----------------------

    #[test]
    fn test_validate_negbin_theta_disc_negbin() {
        let (case, ds) = load_case("disc_negbin");
        let r = run_negbin(&ds, "y", &["x1", "x2"], None).unwrap();
        assert!(r.converged, "{:?}", r.warnings);
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
        let n = r.n_obs;
        // deviation ECON-64 (T4): log_likelihood_null and pseudo_r_squared not pinned
        Check::new(&case)
            .named("coef", &r.variables, &r.coefficients, Tol::iterative(1e-6))
            .named("se", &r.variables, &r.std_errors, Tol::iterative(1e-5))
            .named("z", &r.variables, &r.z_stats, Tol::iterative(1e-5))
            // p of an iterative estimate: d ln p / d ln z ~ z^2 times the z tolerance
            .named("p", &r.variables, &r.p_values, Tol::iterative(2e-3))
            .scalar("theta", r.theta, Tol::iterative(1e-6))
            .scalar("se_theta", r.theta_std_error, Tol::iterative(1e-5))
            .scalar("loglik", r.log_likelihood, Tol::stat(n))
            .scalar("aic", r.aic, Tol::stat(n))
            .scalar("bic", r.bic, Tol::stat(n))
            .scalar("y_mean", r.y_mean, Tol::coef(n))
            .scalar("y_var", r.y_var, Tol::coef(n))
            .scalar("n", n as f64, Tol::exact())
            // glm.nb's own optimum (cross-check)
            .named(
                "coef_pkg",
                &r.variables,
                &r.coefficients,
                Tol::iterative(1e-6),
            )
            .scalar("theta_pkg", r.theta, Tol::iterative(1e-6))
            // glm.nb evaluates SE.theta one theta.ml step early
            .scalar("se_theta_pkg", r.theta_std_error, Tol::iterative(1e-4))
            .finish();
    }

    /// `init_theta` is a starting value (glm.nb's init.theta): the fit is the same MLE.
    /// It must be positive and finite.
    #[test]
    fn test_validate_negbin_init_theta_is_a_starting_value() {
        let (case, ds) = load_case("disc_negbin");
        for init in [50.0, 0.2] {
            let r = run_negbin(&ds, "y", &["x1", "x2"], Some(init)).unwrap();
            assert!(r.converged, "init {init}: {:?}", r.warnings);
            Check::new(&case)
                .scalar("theta", r.theta, Tol::iterative(1e-6))
                .scalar("theta_init50_pkg", r.theta, Tol::iterative(1e-6))
                .named("coef", &r.variables, &r.coefficients, Tol::iterative(1e-6))
                .finish();
        }
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            match run_negbin(&ds, "y", &["x1", "x2"], Some(bad)) {
                Err(EconError::InvalidSpecification { message }) => {
                    assert!(message.contains("init_theta"), "{message}")
                }
                other => panic!("init {bad}: expected InvalidSpecification, got {other:?}"),
            }
        }
    }

    /// Rows with a missing outcome or regressor are dropped jointly (glm.nb: na.omit).
    #[test]
    fn test_validate_negbin_missing_rows_dropped() {
        let (case, ds) = load_case("disc_negbin");
        let r = run_negbin(&ds, "y_na", &["x1_na", "x2"], None).unwrap();
        assert!(r.converged, "{:?}", r.warnings);
        assert_eq!(
            r.missing.by_column,
            [("y_na".to_string(), 3), ("x1_na".to_string(), 2)]
        );
        let n = r.n_obs;
        Check::new(&case)
            .named(
                "na_coef",
                &r.variables,
                &r.coefficients,
                Tol::iterative(1e-6),
            )
            .named("na_se", &r.variables, &r.std_errors, Tol::iterative(1e-5))
            .scalar("na_theta", r.theta, Tol::iterative(1e-6))
            .scalar("na_se_theta", r.theta_std_error, Tol::iterative(1e-5))
            .scalar("na_loglik", r.log_likelihood, Tol::stat(n))
            .scalar("na_n_used", n as f64, Tol::exact())
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

    /// R's trigamma at the points the NB theta information needs. The series is
    /// accurate to 2.3e-13 relative (T2-WP10 Appendix B.4); the one it replaced was off
    /// by 4.5e-7 at 6.
    #[test]
    fn test_validate_negbin_trigamma_disc_negbin() {
        let (case, _) = load_case("disc_negbin");
        let values: Vec<f64> = case
            .vector("trigamma_x")
            .iter()
            .map(|&x| trigamma(x))
            .collect();
        Check::new(&case)
            .vector("trigamma", &values, Tol::new(1e-12, 0.0))
            .finish();
    }

    #[test]
    fn test_negbin_stops_when_cancelled() {
        let ds = negbin_data();
        let e = resource::with_budget(cancelled(), || run_negbin(&ds, "y", &["x1", "x2"], None))
            .unwrap_err();
        assert!(matches!(e, EconError::Cancelled(_)), "{e:?}");
    }

    #[test]
    fn test_zip_basic() {
        let df = df! {
            "y" => [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 0.0, 5.0, 4.0],
            "x" => [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 3.5, 6.0, 4.5, 7.0, 6.5]
        }
        .unwrap();
        let dataset = Dataset::new(df);
        let result = run_zip(&dataset, "y", &["x"], None).unwrap();

        assert_eq!(result.n_obs, 12);
        assert_eq!(result.model_type, ZeroInflatedType::Poisson);
        assert!(result.theta.is_none());
    }

    #[test]
    fn test_zinb_basic() {
        let df = df! {
            "y" => [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 0.0, 5.0, 4.0],
            "x" => [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 3.5, 6.0, 4.5, 7.0, 6.5]
        }
        .unwrap();
        let dataset = Dataset::new(df);
        let result = run_zinb(&dataset, "y", &["x"], None).unwrap();

        assert_eq!(result.model_type, ZeroInflatedType::NegBin);
        assert!(result.theta.is_some());
    }

    /// The 12-row fixture of `test_zip_basic` separates the zero part: pscl reports zero
    /// coefficients 94.70464, -25.39549 and logLik -8.370407 (`zeroinfl(y ~ x)`, BFGS;
    /// T2-WP10 Appendix B.3) as converged. The supremum is higher (the EM alone reaches
    /// -8.368208), so the fit must be at least pscl's and flag the separation.
    #[test]
    fn test_zip_separated_zero_part_is_flagged() {
        let df = df! {
            "y" => [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 0.0, 5.0, 4.0],
            "x" => [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 3.5, 6.0, 4.5, 7.0, 6.5]
        }
        .unwrap();
        let r = run_zip(&Dataset::new(df), "y", &["x"], None).unwrap();
        assert_eq!(r.n_obs, 12);
        assert_eq!(r.n_zeros, 7);
        assert!(
            r.warnings.iter().any(|w| w.contains("numerically 0 or 1")),
            "{:?}",
            r.warnings
        );
        assert!(r.log_likelihood.is_finite(), "{r}");
        assert!(r.log_likelihood >= -8.370407, "{r}");
    }

    // ---- zero-inflated Poisson and NB against pscl::zeroinfl (disc_zeroinfl) --------

    #[test]
    fn test_validate_zip_disc_zeroinfl() {
        let (case, ds) = load_case("disc_zeroinfl");
        let r = run_zip(&ds, "y", &["x1", "x2"], Some(&["x1", "x2"])).unwrap();
        check_zi(&case, "zip_", &r);
        assert!(r.theta.is_none() && r.theta_std_error.is_none());
        assert!(r.log_theta_std_error.is_none());
    }

    #[test]
    fn test_validate_zinb_disc_zeroinfl() {
        let (case, ds) = load_case("disc_zeroinfl");
        let r = run_zinb(&ds, "y_nb", &["x1", "x2"], Some(&["x1", "x2"])).unwrap();
        check_zi(&case, "zinb_", &r);
        let theta = r.theta.unwrap();
        Check::new(&case)
            .scalar("zinb_theta", theta, Tol::iterative(1e-6))
            .scalar("zinb_theta_pkg", theta, Tol::iterative(1e-4))
            .scalar(
                "zinb_se_log_theta",
                r.log_theta_std_error.unwrap(),
                Tol::iterative(1e-5),
            )
            .scalar(
                "zinb_se_theta",
                r.theta_std_error.unwrap(),
                Tol::iterative(1e-5),
            )
            .scalar(
                "zinb_se_log_theta_pkg",
                r.log_theta_std_error.unwrap(),
                Tol::iterative(1e-4),
            )
            .finish();
    }

    /// The zero part with its own regressors (`y ~ x1 + x2 | x1`).
    #[test]
    fn test_validate_zip_zero_part_regressors() {
        let (case, ds) = load_case("disc_zeroinfl");
        let r = run_zip(&ds, "y", &["x1", "x2"], Some(&["x1"])).unwrap();
        assert!(r.converged, "{:?}", r.warnings);
        Check::new(&case)
            .named(
                "zip_z1_count",
                &r.count_variables,
                &r.count_coefficients,
                Tol::iterative(1e-6),
            )
            .named(
                "zip_z1_zero",
                &r.zero_variables,
                &r.zero_coefficients,
                Tol::iterative(1e-6),
            )
            .vector("zip_z1_se_zero", &r.zero_std_errors, Tol::iterative(1e-5))
            .scalar("zip_z1_loglik", r.log_likelihood, Tol::stat(r.n_obs))
            .finish();
    }

    /// Rows with a missing outcome or regressor are dropped jointly (zeroinfl: na.omit);
    /// the zero part defaults to the count regressors.
    #[test]
    fn test_validate_zip_missing_rows_dropped() {
        let (case, ds) = load_case("disc_zeroinfl");
        let r = run_zip(&ds, "y_na", &["x1", "x2_na"], None).unwrap();
        assert!(r.converged, "{:?}", r.warnings);
        assert_eq!(
            r.missing.by_column,
            [("y_na".to_string(), 3), ("x2_na".to_string(), 2)]
        );
        Check::new(&case)
            .named(
                "zip_na_count",
                &r.count_variables,
                &r.count_coefficients,
                Tol::iterative(1e-6),
            )
            .named(
                "zip_na_zero",
                &r.zero_variables,
                &r.zero_coefficients,
                Tol::iterative(1e-6),
            )
            .vector("zip_na_se_count", &r.count_std_errors, Tol::iterative(1e-5))
            .scalar("zip_na_loglik", r.log_likelihood, Tol::stat(r.n_obs))
            .scalar("zip_na_n_used", r.n_obs as f64, Tol::exact())
            .scalar("zip_na_n_dropped", r.missing.n_dropped as f64, Tol::exact())
            .scalar(
                "zip_na_missing_y",
                r.missing.by_column[0].1 as f64,
                Tol::exact(),
            )
            .scalar(
                "zip_na_missing_x2",
                r.missing.by_column[1].1 as f64,
                Tol::exact(),
            )
            .finish();
    }

    /// pscl::hurdle drops the same rows (na.omit); the hurdle estimates are not pinned
    /// (ECON-16, Tier 4) but equal the fit on the complete rows.
    #[test]
    fn test_validate_hurdle_missing_rows_dropped() {
        let (case, ds) = load_case("disc_zeroinfl");
        let r = run_hurdle(&ds, "y_na", &["x1", "x2_na"], None, HurdleType::Poisson).unwrap();
        assert_eq!(r.missing.n_dropped, 5);
        Check::new(&case)
            .scalar("hurdle_na_n_used", r.n_obs as f64, Tol::exact())
            .finish();
        let complete = Dataset::new(
            ds.df()
                .drop_nulls(Some(&["y_na".to_string(), "x2_na".to_string()]))
                .unwrap(),
        );
        let c = run_hurdle(
            &complete,
            "y_na",
            &["x1", "x2_na"],
            None,
            HurdleType::Poisson,
        )
        .unwrap();
        assert_eq!(r.binary_coefficients, c.binary_coefficients);
        assert_eq!(r.count_coefficients, c.count_coefficients);
    }

    /// An outcome without a zero count has no zero part to fit (pscl: "minimum count is
    /// not zero").
    #[test]
    fn test_zeroinfl_rejects_outcome_without_zeros() {
        let ds = zi_data();
        let y_pos =
            (ds.column("y_nb").unwrap().as_materialized_series() + 1).with_name("y_pos".into());
        let ds = with_column(&ds, y_pos.into_column());
        for r in [
            run_zip(&ds, "y_pos", &["x1", "x2"], None),
            run_zinb(&ds, "y_pos", &["x1", "x2"], None),
        ] {
            match r {
                Err(EconError::InvalidSpecification { message }) => {
                    assert!(message.contains("zero"), "{message}")
                }
                other => panic!("expected InvalidSpecification, got {other:?}"),
            }
        }
    }

    /// An Int64 outcome and the same values as Float64 give the same fit.
    #[test]
    fn test_na_count_models_float_outcome_equals_int64() {
        let as_f64 = |ds: &Dataset| {
            let y = ds
                .column("y")
                .unwrap()
                .as_materialized_series()
                .cast(&DataType::Float64)
                .unwrap()
                .with_name("y_f64".into());
            with_column(ds, y.into_column())
        };
        let ds = as_f64(&negbin_data());
        let a = run_negbin(&ds, "y", &["x1", "x2"], None).unwrap();
        let b = run_negbin(&ds, "y_f64", &["x1", "x2"], None).unwrap();
        assert_eq!(a.coefficients, b.coefficients);
        let ds = as_f64(&zi_data());
        let a = run_zip(&ds, "y", &["x1", "x2"], None).unwrap();
        let b = run_zip(&ds, "y_f64", &["x1", "x2"], None).unwrap();
        assert_eq!(a.count_coefficients, b.count_coefficients);
        assert_eq!(a.zero_coefficients, b.zero_coefficients);
    }

    /// A text outcome is not a count: NonNumericColumn (before, a non-strict cast read
    /// every label as 0).
    #[test]
    fn test_na_count_models_text_outcome_is_non_numeric() {
        let ds = ordered_text_data();
        let errors = [
            run_negbin(&ds, "y_label", &["x1"], None).map(|_| ()),
            run_zip(&ds, "y_label", &["x1"], None).map(|_| ()),
            run_hurdle(&ds, "y_label", &["x1"], None, HurdleType::Poisson).map(|_| ()),
        ];
        for e in errors {
            match e {
                Err(EconError::NonNumericColumn { column }) => assert_eq!(column, "y_label"),
                other => panic!("expected NonNumericColumn, got {other:?}"),
            }
        }
    }

    /// pscl's analytic gradient (the private `zi_loglik`) is the derivative of the
    /// log-likelihood: central differences (step 1e-6) at a point away from the optimum,
    /// for both distributions.
    #[test]
    fn test_zeroinfl_gradient_matches_central_differences() {
        let ds = zi_data();
        let cols = ColumnRequest::new(ds.df())
            .numerics(&["y_nb", "x1", "x2"])
            .extract()
            .unwrap();
        let x = cols.matrix(&["x1", "x2"], true);
        let y = cols.numeric("y_nb");
        let h = 1e-6;
        for (dist, par) in [
            (
                ZeroInflatedType::Poisson,
                vec![0.5, 0.2, -0.2, -0.3, 0.4, 0.1],
            ),
            (
                ZeroInflatedType::NegBin,
                vec![0.5, 0.2, -0.2, -0.3, 0.4, 0.1, 1.5f64.ln()],
            ),
        ] {
            let data = ZiData {
                x: x.view(),
                z: x.view(),
                y,
                dist,
            };
            let mut g = vec![0.0; par.len()];
            zi_loglik(&data, &par, Some(&mut g)).unwrap();
            for k in 0..par.len() {
                let (mut up, mut dn) = (par.clone(), par.clone());
                up[k] += h;
                dn[k] -= h;
                let fd = (zi_loglik(&data, &up, None).unwrap()
                    - zi_loglik(&data, &dn, None).unwrap())
                    / (2.0 * h);
                assert!(
                    (fd - g[k]).abs() <= 1e-6 * (1.0 + g[k].abs()),
                    "{dist:?}: gradient[{k}] {} vs central difference {fd}",
                    g[k]
                );
            }
        }
    }

    #[test]
    fn test_zeroinfl_stops_when_cancelled() {
        let ds = zi_data();
        let e = resource::with_budget(cancelled(), || run_zip(&ds, "y", &["x1", "x2"], None))
            .unwrap_err();
        assert!(matches!(e, EconError::Cancelled(_)), "{e:?}");
    }

    #[test]
    fn test_hurdle_poisson() {
        let df = df! {
            "y" => [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 3.0, 4.0, 2.0, 5.0, 3.0],
            "x" => [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 3.5, 5.0, 6.0, 4.5, 7.0, 5.5]
        }
        .unwrap();
        let dataset = Dataset::new(df);
        let result = run_hurdle(&dataset, "y", &["x"], None, HurdleType::Poisson).unwrap();

        assert_eq!(result.n_obs, 12);
        assert_eq!(result.model_type, HurdleType::Poisson);
        assert!(result.theta.is_none());
    }

    #[test]
    fn test_hurdle_negbin() {
        let df = df! {
            "y" => [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 3.0, 4.0, 2.0, 5.0, 3.0],
            "x" => [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 3.5, 5.0, 6.0, 4.5, 7.0, 5.5]
        }
        .unwrap();
        let dataset = Dataset::new(df);
        let result = run_hurdle(&dataset, "y", &["x"], None, HurdleType::NegBin).unwrap();

        assert_eq!(result.model_type, HurdleType::NegBin);
        assert!(result.theta.is_some());
    }

    // ==========================================================================
    // R Validation Tests
    // ==========================================================================

    #[test]
    fn test_validate_hurdle_poisson_structure() {
        // Hurdle Poisson validation
        // R reference: pscl::hurdle(y ~ x | x, dist="poisson")

        let x: Vec<f64> = (0..150).map(|i| (i as f64 * 0.033) % 5.0).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| {
                // Binary hurdle: ~40% zeros
                if i % 5 < 2 {
                    0.0
                } else {
                    // Truncated Poisson (positive counts only)
                    1.0 + (xi * 0.5).round().min(5.0)
                }
            })
            .collect();

        let n_zeros = y.iter().filter(|&&yi| yi < 0.5).count();
        let n_positive = y.len() - n_zeros;

        let df = df! {
            "y" => y,
            "x" => x
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_hurdle(&dataset, "y", &["x"], None, HurdleType::Poisson).unwrap();

        // Structure checks
        assert_eq!(result.n_obs, 150);
        assert_eq!(result.model_type, HurdleType::Poisson);
        assert!(result.theta.is_none());
        assert_eq!(result.n_zeros, n_zeros);
        assert_eq!(result.n_positive, n_positive);

        // Both parts should have coefficients
        assert!(
            !result.binary_coefficients.is_empty(),
            "Should have binary coefficients"
        );
        assert!(
            !result.count_coefficients.is_empty(),
            "Should have count coefficients"
        );

        // Log-likelihood should be sum of both parts
        assert!(
            result.log_likelihood.is_finite(),
            "Log-likelihood should be finite"
        );
        assert!(result.ll_binary.is_finite(), "Binary LL should be finite");
        assert!(result.ll_count.is_finite(), "Count LL should be finite");
    }

    #[test]
    fn test_validate_hurdle_negbin_structure() {
        // Hurdle Negative Binomial validation
        // R reference: pscl::hurdle(y ~ x | x, dist="negbin")

        let x: Vec<f64> = (0..150).map(|i| (i as f64 * 0.033) % 5.0).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| {
                if i % 3 == 0 {
                    0.0
                } else {
                    1.0 + (xi * 0.8).round().min(8.0)
                }
            })
            .collect();

        let df = df! {
            "y" => y,
            "x" => x
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_hurdle(&dataset, "y", &["x"], None, HurdleType::NegBin).unwrap();

        // Structure checks
        assert_eq!(result.n_obs, 150);
        assert_eq!(result.model_type, HurdleType::NegBin);
        assert!(result.theta.is_some(), "NegBin hurdle should have theta");
        assert!(result.theta.unwrap() > 0.0, "Theta should be positive");

        // Log-likelihood checks
        assert!(
            result.log_likelihood.is_finite(),
            "Log-likelihood should be finite"
        );
    }
}
