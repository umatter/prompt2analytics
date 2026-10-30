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

use super::newton::{MAX_HALVINGS, ensure_dense_inverse_fits, newton_step};

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

/// Smallest `init_theta` [`run_negbin`] starts from. Far below it the theta score and
/// information cannot be evaluated (trigamma(theta) = 1/theta^2 overflows below 1e-154).
const INIT_THETA_MIN: f64 = 1e-8;

/// The warning for a theta above [`THETA_POISSON`]: `hint` names the model that fits as
/// well.
fn no_overdispersion_warning(theta: f64, hint: &str) -> String {
    format!("theta = {theta:.3e}: the data show no overdispersion beyond the Poisson model; {hint}")
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

/// The log-likelihood the coefficient IRLS monitors: NB2 at `theta`, Poisson for an
/// infinite `theta`.
fn irls_loglik(theta: f64, mu: &[f64], y: &[f64]) -> f64 {
    use statrs::function::gamma::ln_gamma;
    if theta.is_infinite() {
        y.iter()
            .zip(mu)
            .map(|(&yi, &mi)| {
                let log_mu = if yi == 0.0 { 0.0 } else { yi * mi.ln() };
                log_mu - mi - ln_gamma(yi + 1.0)
            })
            .sum()
    } else {
        nb_loglik(theta, mu, y)
    }
}

/// Fisher scoring for the coefficients of the NB2 log-link model at a fixed `theta`
/// (`f64::INFINITY`: Poisson), glm.fit's IRLS: weights `W = mu theta / (theta + mu)`,
/// working response `z = eta + (y - mu) / mu`. Starts from `beta`, or from
/// `mu = y + 0.1` (glm's Poisson start) when `beta` is `None`. Iterates at most 100
/// times, until the coefficients change by less than `1e-12 (1 + max |beta|)` or the
/// log-likelihood by at most `1e-12 (|ll| + 0.1)` (glm.fit's relative test, on the
/// log-likelihood). The second test is the one an ill-conditioned X'WX (polynomial
/// terms, nearly collinear regressors) passes: there the solve moves the coefficients
/// by rounding noise in the directions the data determine poorly, which does not
/// change the likelihood. [`IrlsStop`] says whether either test passed and holds the
/// last step. `work.eta` and `work.mu` hold the fitted values on return.
fn irls_beta(
    x: &ArrayView2<f64>,
    y: &[f64],
    beta: Option<&Array1<f64>>,
    theta: f64,
    work: &mut NbWork,
) -> EconResult<(Array1<f64>, IrlsStop)> {
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
    let mut ll = match beta {
        Some(_) => irls_loglik(theta, &work.mu, y),
        None => f64::NAN,
    };
    let mut stop = IrlsStop {
        converged: false,
        last_step: None,
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
        stop.last_step = beta.as_ref().map(|old| &new_beta - old);
        let change = stop.last_step.as_ref().map_or(f64::INFINITY, |step| {
            step.iter().fold(0.0f64, |m, d| m.max(d.abs()))
        });
        linear_predictor(x, &new_beta, &mut work.eta, &mut work.mu);
        beta = Some(new_beta);
        let ll_new = irls_loglik(theta, &work.mu, y);
        let ll_change = (ll_new - ll).abs();
        ll = ll_new;
        if change < 1e-12 * scale || ll_change <= 1e-12 * (ll_new.abs() + 0.1) {
            stop.converged = true;
            break;
        }
    }
    Ok((beta.unwrap_or_else(|| Array1::zeros(x.ncols())), stop))
}

/// How [`irls_beta`] ended.
#[derive(Debug, Clone)]
struct IrlsStop {
    /// One of the two stopping tests passed.
    converged: bool,
    /// The coefficient change of the last step; `None` when that step was the first
    /// one from glm's start (no coefficients before it).
    last_step: Option<Array1<f64>>,
}

/// Smallest drop of a linear predictor in the last IRLS step that counts as a fitted
/// mean still being driven to 0 (by more than 1%; [`run_negbin`]'s no-MLE test).
const ETA_FALLING: f64 = 0.01;

impl IrlsStop {
    /// Whether the last step lowered the linear predictor `x_i' beta` of row `xi` by
    /// more than [`ETA_FALLING`]; `true` when there was no step from earlier
    /// coefficients.
    fn lowered(&self, xi: ndarray::ArrayView1<f64>) -> bool {
        self.last_step
            .as_ref()
            .is_none_or(|step| xi.dot(step) < -ETA_FALLING)
    }
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
/// The flag is `false` when the score, the information or the log-likelihood is not
/// finite at a theta the steps reach, so that theta could not be updated from there
/// (the returned theta is that one).
fn theta_newton(theta: f64, mu: &[f64], y: &[f64]) -> EconResult<(f64, bool)> {
    let mut theta = theta.min(THETA_MAX);
    let mut ll = nb_loglik(theta, mu, y);
    for _ in 0..100 {
        resource::checkpoint()?;
        let s = theta_score(theta, mu, y);
        let info = theta_information(theta, mu, y);
        if !(s.is_finite() && info.is_finite() && ll.is_finite()) {
            return Ok((theta, false));
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
    Ok((theta, true))
}

/// Run negative binomial regression (NB2, log link).
///
/// R equivalent: `MASS::glm.nb()`. The coefficients come from Fisher scoring (IRLS)
/// at fixed theta, theta from Newton steps on MASS `theta.ml`'s score and information
/// at fixed fitted means, alternating until theta changes by less than 1e-10 relative
/// and the log-likelihood by less than 1e-12 relative (at most 100 rounds), as glm.nb
/// does. `initial_theta` is the starting value of theta (glm.nb's `init.theta`), in
/// `[1e-8, 1e6]`; without it the start is a Poisson fit and MASS's moment estimate
/// `n / sum((y/mu - 1)^2)`, moved into that range. Standard errors of the coefficients
/// are `(X'WX)^-1` with `W = mu theta/(theta + mu)` (summary.negbin),
/// `SE(theta) = 1/sqrt(information)` at the optimum (MASS `SE.theta`).
///
/// Rows with a missing outcome or regressor are dropped (R `na.omit`) and reported in
/// `missing`. Data without overdispersion drive theta towards its bound 1e8; a final
/// theta above 1e6 gives `converged = false`, `theta_std_error = NaN` and a warning.
/// An outcome without a finite MLE, such as all zeros or zero counts a regressor
/// separates, gives `converged = false` and a warning that counts the zero counts whose
/// linear predictor the last step of the final IRLS still lowered by more than 0.01
/// (their fitted means by more than 1%): without a finite MLE each step lowers it by
/// about 1, at any n, while the means themselves need not reach glm.fit's `10 eps`
/// before the IRLS stops. Fitted means below `10 eps` at a converged fit (a zero count
/// at an extreme regressor value, which adds nothing to the likelihood) are a note in
/// `warnings` that counts the rows, and `converged` is unchanged. A theta that cannot
/// be updated and a coefficient IRLS that does not settle also give
/// `converged = false` and a warning.
///
/// # Errors
///
/// `InvalidSpecification` for an `initial_theta` outside `[1e-8, 1e6]` (or not a
/// number): below it the theta score cannot be evaluated, above it the data show no
/// overdispersion the alternation could move from. Column errors (`ColumnNotFound`,
/// `NonNumericColumn`, `NonFiniteValues`) name the column; `InsufficientData` below 2
/// complete rows; `SingularMatrix` for collinear regressors.
pub fn run_negbin(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    initial_theta: Option<f64>,
) -> EconResult<NegBinResult> {
    if let Some(t) = initial_theta
        && !(INIT_THETA_MIN..=THETA_POISSON).contains(&t)
    {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "init_theta must be between 1e-8 and 1e6 (got {t:e}): outside that range \
                 the theta iteration cannot move from its start. Omit init_theta to start \
                 from the moment estimate."
            ),
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
    let (mut beta, _) = irls_beta(&x, y, None, start_theta, &mut work)?;
    let mut theta = match initial_theta {
        Some(t) => t,
        None => {
            let ss: f64 = y
                .iter()
                .zip(&work.mu)
                .map(|(&yi, &mi)| (yi / mi - 1.0).powi(2))
                .sum();
            // moved into init_theta's range; an undefined estimate (0/0) starts at
            // its upper end
            let t = n as f64 / ss;
            if t.is_nan() {
                THETA_POISSON
            } else {
                t.clamp(INIT_THETA_MIN, THETA_POISSON)
            }
        }
    };

    let mut warnings: Vec<String> = Vec::new();
    let mut converged = false;
    let mut theta_stuck = false;
    let mut iterations = 0;
    let mut ll = nb_loglik(theta, &work.mu, y);
    for round in 1..=100 {
        resource::checkpoint()?;
        iterations = round;
        beta = irls_beta(&x, y, Some(&beta), theta, &mut work)?.0;
        let (theta_new, updated) = theta_newton(theta, &work.mu, y)?;
        if !updated {
            warnings.push(format!(
                "theta could not be updated in round {round}: its score or information is \
                 not finite at theta = {theta_new:.3e}; the estimates are not reliable"
            ));
            theta = theta_new;
            theta_stuck = true;
            break;
        }
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
    if !converged && !theta_stuck && theta < THETA_MAX {
        warnings.push("the beta/theta alternation did not converge in 100 rounds".to_string());
    }
    // The coefficients at the final theta.
    let (beta, irls) = irls_beta(&x, y, Some(&beta), theta, &mut work)?;
    // An outcome without a finite MLE (all counts 0, or zero counts that a regressor
    // separates): the IRLS drives the fitted means of those zero counts to 0. The working
    // response of such a row is eta - 1, so each step lowers its linear predictor by
    // about 1, at any n. Their means are no test: the IRLS stops on its log-likelihood
    // test, whose threshold grows with |ll|, so on larger data they are still above
    // glm.fit's 10 eps when it stops. At a regular fit the last step moves every row by
    // rounding and theta's last change, also a zero count at an extreme regressor value
    // (which adds nothing to the likelihood); its tiny means are a note.
    let n_falling = y
        .iter()
        .zip(x.outer_iter())
        .filter(|&(&yi, xi)| yi == 0.0 && irls.lowered(xi))
        .count();
    let no_mle = n_falling > 0;
    let tiny = 10.0 * f64::EPSILON;
    let n_tiny = work.mu.iter().filter(|&&m| m < tiny).count();
    let rows = |k: usize| if k == 1 { "row" } else { "rows" };
    if no_mle {
        warnings.push(format!(
            "the coefficient IRLS was still driving the fitted means of {n_falling} \
             zero-count {} to 0 (its last step lowered their linear predictor by more \
             than {ETA_FALLING}): the outcome has no finite maximum-likelihood estimate \
             (all counts 0, or zero counts that a regressor separates); the estimates are \
             not reliable",
            rows(n_falling)
        ));
    } else if !irls.converged {
        warnings.push(
            "the IRLS for the coefficients did not settle in 100 iterations at the final \
             theta: neither the coefficients nor the log-likelihood stopped changing"
                .to_string(),
        );
    } else if n_tiny > 0 {
        warnings.push(format!(
            "fitted means numerically 0 occurred for {n_tiny} {} (glm.fit: 'fitted rates \
             numerically 0') at converged coefficients; check those rows for extreme \
             regressor values",
            rows(n_tiny)
        ));
    }
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

    let mut converged =
        converged && irls.converged && !no_mle && log_likelihood.is_finite() && theta.is_finite();
    let theta_std_error = if theta > THETA_POISSON {
        converged = false;
        warnings.push(no_overdispersion_warning(
            theta,
            "the Poisson model fits as well (R glm.nb warns 'iteration limit reached')",
        ));
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
/// with pscl's analytic gradient and the analytic Hessian. The fit has converged when
/// the Newton decrement `g'(-H)^-1 g` is at most `1e-12 (1 + |ll|)`, the
/// log-likelihood changes by less than `1e-12 (|ll| + 1e-12)` and every component of
/// the Newton step is at most `1e-10 (1 + |par_k|)`; unlike the largest gradient
/// component, none of these depends on the units of a regressor. Standard errors come
/// from the inverse Hessian at the estimate; like the estimates and the convergence
/// flag, they do not depend on the units of a regressor.
///
/// Rows with a missing outcome or regressor (count or zero part) are dropped (R
/// `na.omit`) and reported in `missing`. Non-convergence and a Hessian that is not
/// negative definite are reported in `warnings`, never as an error, and so are
/// fitted zero-inflation probabilities numerically 0 or 1 (glm.fit's diagnostic, with
/// the number of rows: quasi-separation in the zero part, or rows with extreme
/// regressor values), which leave `converged` unchanged.
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
/// (no overdispersion) gives `converged = false`, NaN theta SEs and a warning; the
/// count and zero-part SEs then come from their own block of the Hessian (the
/// Poisson-limit information, where the ln theta direction is singular).
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

/// `ln(1 + e^t)` without overflow or cancellation.
fn softplus(t: f64) -> f64 {
    if t > 0.0 {
        t + (-t).exp().ln_1p()
    } else {
        t.exp().ln_1p()
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
                    let m = logistic_cdf(eta[i]).clamp(eps, 1.0 - eps);
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
/// pscl's analytic gradient (`gradPoisson` / `gradNegBin`, logit link) in `grad` and the
/// analytic Hessian in `hess` when they are given.
///
/// In log space: `ln pi = -softplus(-eta_z)`, `ln(1 - pi) = -softplus(eta_z)`,
/// `L = ln f0 = -mu` (Poisson) or `theta (ln theta - ln(theta + mu))` (NB). A zero count
/// adds `logaddexp(ln pi, ln(1 - pi) + L)`, a positive one `ln(1 - pi) + ln f(y)`.
///
/// Derivatives per observation in `c = eta_c` (count part), `z = eta_z` (zero part) and
/// `t = ln theta`. A zero, with `q = (1 - pi) f0 / dens0 = logistic(L - eta_z)`:
/// first `q L_c`, `1 - q - pi`, `q L_t`; second
/// `q L_ab + q (1 - q) L_a L_b` for `a, b` in `{c, t}`, `-q (1 - q) L_a` for `(z, a)`
/// and `q (1 - q) - pi (1 - pi)` for `(z, z)`, where `L_c = -mu`, `L_cc = -mu`
/// (Poisson) or `L_c = -theta mu / (theta + mu)`, `L_cc = -theta^2 mu / (theta + mu)^2`,
/// `L_t = theta (ln theta - ln(theta + mu) + mu / (theta + mu))`,
/// `L_ct = -theta mu^2 / (theta + mu)^2`, `L_tt = L_t + theta mu^2 / (theta + mu)^2`
/// (NB). A positive count: `y - mu`, `-mu` (Poisson) or `y - mu (y + theta) / (mu +
/// theta)`, `-theta mu (y + theta) / (mu + theta)^2` (NB) in `c`; `-pi`,
/// `-pi (1 - pi)` in `z` (no cross term); in `t` (NB) `theta D` with
/// `D = digamma(y + theta) - digamma(theta) + ln theta + 1 - ln(mu + theta) -
/// (y + theta) / (mu + theta)`, `(c, t)`: `-theta mu (mu - y) / (mu + theta)^2`, and
/// `(t, t)`: `theta D + theta^2 (trigamma(y + theta) - trigamma(theta) + 1 / theta -
/// 1 / (mu + theta) - (mu - y) / (mu + theta)^2)`. The parameter derivatives are these
/// times the regressors (`x` for `c`, `z` for `z`).
fn zi_loglik(
    data: &ZiData<'_>,
    par: &[f64],
    mut grad: Option<&mut [f64]>,
    mut hess: Option<&mut Array2<f64>>,
) -> EconResult<f64> {
    use statrs::function::gamma::ln_gamma;
    let kx = data.x.ncols();
    let kz = data.z.ncols();
    let it = kx + kz;
    let beta = &par[..kx];
    let gamma = &par[kx..it];
    let negbin = data.dist == ZeroInflatedType::NegBin;
    let (theta, log_theta) = if negbin {
        let lt = par[it];
        (lt.exp(), lt)
    } else {
        (f64::INFINITY, f64::INFINITY)
    };
    let lg_theta = if negbin { ln_gamma(theta) } else { 0.0 };
    let dg_theta = if negbin { digamma(theta) } else { 0.0 };
    let tg_theta = if negbin && hess.is_some() {
        trigamma(theta)
    } else {
        0.0
    };
    if let Some(g) = grad.as_deref_mut() {
        g.fill(0.0);
    }
    if let Some(h) = hess.as_deref_mut() {
        h.fill(0.0);
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
        let ln_theta_mu = if negbin { (theta + mu).ln() } else { 0.0 };
        if grad.is_none() && hess.is_none() {
            ll += if yi < 0.5 {
                let ln_f0 = if negbin {
                    theta * (log_theta - ln_theta_mu)
                } else {
                    -mu
                };
                log_add_exp(ln_pi, ln_1m_pi + ln_f0)
            } else if negbin {
                ln_1m_pi + ln_gamma(yi + theta) - lg_theta - ln_gamma(yi + 1.0)
                    + theta * log_theta
                    + yi * eta_c
                    - (theta + yi) * ln_theta_mu
            } else {
                ln_1m_pi - mu + yi * eta_c - ln_gamma(yi + 1.0)
            };
            continue;
        }
        let pi = logistic_cdf(eta_z);
        let pi_var = pi * (1.0 - pi);
        // first derivatives in (c, z, t), then second ones
        let (s_c, s_z, s_t);
        let (h_cc, h_cz, h_zz, h_ct, h_zt, h_tt);
        if yi < 0.5 {
            let (l_c, l_cc, l_t, l_ct, l_tt);
            if negbin {
                let r = theta / (theta + mu);
                let mu_r2 = mu * (1.0 - r) * theta / (theta + mu); // theta mu^2 / (theta + mu)^2
                l_c = -r * mu;
                l_cc = -r * r * mu;
                l_t = theta * (log_theta - ln_theta_mu + (1.0 - r));
                l_ct = -mu_r2;
                l_tt = l_t + mu_r2;
            } else {
                l_c = -mu;
                l_cc = -mu;
                l_t = 0.0;
                l_ct = 0.0;
                l_tt = 0.0;
            }
            let ln_f0 = if negbin {
                theta * (log_theta - ln_theta_mu)
            } else {
                -mu
            };
            ll += log_add_exp(ln_pi, ln_1m_pi + ln_f0);
            let q = logistic_cdf(ln_f0 - eta_z);
            let qq = q * (1.0 - q);
            s_c = q * l_c;
            s_z = (1.0 - q) - pi;
            s_t = q * l_t;
            h_cc = q * l_cc + qq * l_c * l_c;
            h_ct = q * l_ct + qq * l_c * l_t;
            h_tt = q * l_tt + qq * l_t * l_t;
            h_cz = -qq * l_c;
            h_zt = -qq * l_t;
            h_zz = qq - pi_var;
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
            s_z = -pi;
            h_zz = -pi_var;
            h_cz = 0.0;
            h_zt = 0.0;
            if negbin {
                let tm = mu + theta;
                s_c = yi - mu * (yi + theta) / tm;
                h_cc = -theta * mu * (yi + theta) / (tm * tm);
                let d = digamma(yi + theta) - dg_theta + log_theta + 1.0
                    - ln_theta_mu
                    - (yi + theta) / tm;
                s_t = theta * d;
                h_ct = -theta * mu * (mu - yi) / (tm * tm);
                h_tt = if hess.is_some() {
                    theta * d
                        + theta
                            * theta
                            * (trigamma(yi + theta) - tg_theta + 1.0 / theta
                                - 1.0 / tm
                                - (mu - yi) / (tm * tm))
                } else {
                    0.0
                };
            } else {
                s_c = yi - mu;
                h_cc = -mu;
                s_t = 0.0;
                h_ct = 0.0;
                h_tt = 0.0;
            }
        }
        if let Some(g) = grad.as_deref_mut() {
            for (gk, &xk) in g[..kx].iter_mut().zip(xr.iter()) {
                *gk += s_c * xk;
            }
            for (gk, &zk) in g[kx..it].iter_mut().zip(zr.iter()) {
                *gk += s_z * zk;
            }
            if negbin {
                g[it] += s_t;
            }
        }
        if let Some(h) = hess.as_deref_mut() {
            // the lower triangle; mirrored below
            for (a, &xa) in xr.iter().enumerate() {
                for (b, &xb) in xr.iter().take(a + 1).enumerate() {
                    h[[a, b]] += h_cc * xa * xb;
                }
            }
            for (a, &za) in zr.iter().enumerate() {
                let ra = kx + a;
                if h_cz != 0.0 {
                    for (b, &xb) in xr.iter().enumerate() {
                        h[[ra, b]] += h_cz * za * xb;
                    }
                }
                for (b, &zb) in zr.iter().take(a + 1).enumerate() {
                    h[[ra, kx + b]] += h_zz * za * zb;
                }
            }
            if negbin {
                for (b, &xb) in xr.iter().enumerate() {
                    h[[it, b]] += h_ct * xb;
                }
                if h_zt != 0.0 {
                    for (b, &zb) in zr.iter().enumerate() {
                        h[[it, kx + b]] += h_zt * zb;
                    }
                }
                h[[it, it]] += h_tt;
            }
        }
    }
    if let Some(h) = hess {
        let m = h.nrows();
        for a in 0..m {
            for b in 0..a {
                h[[b, a]] = h[[a, b]];
            }
        }
    }
    Ok(ll)
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
    let mut ll = zi_loglik(&poisson, &par, None, None)?;
    if !ll.is_finite() {
        gamma.fill(0.0);
        pack(&mut par, &beta, &gamma);
        ll = zi_loglik(&poisson, &par, None, None)?;
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
                logistic_cdf(zr.dot(&gamma) + mu)
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
        let ll_new = zi_loglik(&poisson, &par, None, None)?;
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
    // `grad` and `hess` always belong to `par`: evaluated together, at the start and at
    // each accepted point.
    let mut ll = zi_loglik(&data, &par, Some(&mut grad), Some(&mut hess))?;
    let mut ll_prev = f64::NAN;
    for iter in 1..=ZI_MAX_ITER {
        resource::checkpoint()?;
        iterations = iter;
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
        // The Newton decrement g'(-H)^-1 g (twice the gain the step predicts) does not
        // depend on the units of a regressor; max |g| does (x times s: its score
        // component times s).
        let decrement: f64 = grad.iter().zip(&delta).map(|(g, d)| g * d).sum();
        let rel_step = delta
            .iter()
            .zip(&par)
            .fold(0.0f64, |m, (d, p)| m.max(d.abs() / (1.0 + p.abs())));
        if iter > 1
            && decrement <= 1e-12 * (1.0 + ll.abs())
            && (ll - ll_prev).abs() < 1e-12 * (ll_prev.abs() + 1e-12)
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
            let ll_trial = zi_loglik(&data, &trial, None, None)?;
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
        ll = zi_loglik(&data, &par, Some(&mut grad), Some(&mut hess))?;
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

    // glm.fit's diagnostic for fitted probabilities numerically 0 or 1 (pscl's EM
    // suppresses it). One row with an extreme zero-part regressor value is enough, so it
    // is a note to check the rows, not a verdict on the fit.
    let tiny = 10.0 * f64::EPSILON;
    let n_extreme = x_zero
        .outer_iter()
        .filter(|zr| {
            let pi = logistic_cdf(
                zr.iter()
                    .zip(&par[k_count..k_count + k_zero])
                    .map(|(a, b)| a * b)
                    .sum(),
            );
            pi.min(1.0 - pi) < tiny
        })
        .count();
    if n_extreme > 0 {
        let rows = if n_extreme == 1 { "row" } else { "rows" };
        warnings.push(format!(
            "fitted zero-inflation probabilities numerically 0 or 1 occurred for {n_extreme} \
             {rows} (possible quasi-separation in the zero part, or rows with extreme \
             regressor values); check those rows"
        ));
    }

    // Standard errors from the inverse Hessian at the final estimate (`hess` is at
    // `par`). With theta at the Poisson limit (ZINB) the ln theta direction of -H is
    // singular; the count and zero parts then take their SEs from their own block,
    // the Poisson-limit information, and the theta SEs are NaN. -H is negated in place
    // and passed as a view, so the working set stays at DENSE_INVERSE_COPIES.
    let theta_at_limit = negbin && par[n_params - 1].exp() > THETA_POISSON;
    let m_se = if theta_at_limit {
        n_params - 1
    } else {
        n_params
    };
    hess.mapv_inplace(|h| -h);
    let information = hess.slice(ndarray::s![..m_se, ..m_se]);
    let covariance = if information.iter().all(|h| h.is_finite()) {
        cholesky_inverse(&information).ok()
    } else {
        None
    };
    let mut se_all: Vec<f64> = match &covariance {
        Some(v) => (0..m_se).map(|i| v[[i, i]].sqrt()).collect(),
        None => {
            warnings.push(
                "the Hessian is not negative definite at the estimate; standard errors are \
                 not available"
                    .to_string(),
            );
            vec![f64::NAN; m_se]
        }
    };
    drop(covariance);
    se_all.resize(n_params, f64::NAN);

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
        if theta_at_limit {
            converged = false;
            se_log = f64::NAN;
            se_theta = f64::NAN;
            warnings.push(no_overdispersion_warning(
                theta,
                "the zero-inflated Poisson model (pscl::zeroinfl with dist = 'poisson') fits \
                 as well. The count and zero-part standard errors are its information; those \
                 of theta are not available",
            ));
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
    /// Poisson limit. R (MASS 7.3.60.0.1): `MASS::glm.nb(y ~ x, data = data.frame(y = c(0,
    /// 1, 0, 2, 3, 1, 5, 4, 7, 8, 2, 6), x = c(1, 2, 1.5, 3, 4, 2.5, 5, 4.5, 6, 7, 3.5,
    /// 5.5)))` returns theta 106936.84 with "iteration limit reached"; here theta runs to
    /// its bound and the fit is flagged.
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

    /// NB counts of size 2 (seeded) on two ill-conditioned designs: `y ~ x + x^2 + x^3`
    /// with `x ~ U(10, 20)` (columns up to 8000) and `y ~ x1 + x2` with
    /// `x2 = x1 + 1e-4 noise`. The last IRLS steps move the coefficients by rounding
    /// noise in the directions the data determine poorly. Both fits are regular, so
    /// they must be reported as converged, without warnings. (R's glm.nb on the cubic
    /// sample, written to CSV, converges to the same coefficients, coefficient SEs and
    /// theta within 2e-10 relative.)
    #[test]
    fn test_negbin_ill_conditioned_designs_converge() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use rand_distr::{Distribution, Gamma, Normal as NormalDist, Poisson, Uniform};
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        let nb = |mu: f64, rng: &mut ChaCha8Rng| -> f64 {
            let lambda = Gamma::new(2.0, mu / 2.0).unwrap().sample(rng);
            Poisson::new(lambda.max(1e-300)).unwrap().sample(rng)
        };
        let unif = Uniform::new(10.0, 20.0);
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for _ in 0..400 {
            let xi: f64 = unif.sample(&mut rng);
            let t = xi - 15.0;
            let mu = (0.4 + 0.1 * t - 0.01 * t * t + 0.002 * t * t * t).exp();
            y.push(nb(mu, &mut rng));
            x.push(xi);
        }
        let x2: Vec<f64> = x.iter().map(|v| v * v).collect();
        let x3: Vec<f64> = x.iter().map(|v| v * v * v).collect();
        let cubic = Dataset::new(df! { "y" => y, "x" => x, "x2" => x2, "x3" => x3 }.unwrap());

        let normal = NormalDist::new(0.0, 1.0).unwrap();
        let (mut x1, mut x2, mut y) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..400 {
            let a: f64 = normal.sample(&mut rng);
            let e: f64 = normal.sample(&mut rng);
            y.push(nb((0.3 + 0.5 * a).exp(), &mut rng));
            x1.push(a);
            x2.push(a + 1e-4 * e);
        }
        let collinear = Dataset::new(df! { "y" => y, "x1" => x1, "x2" => x2 }.unwrap());

        for (ds, xs) in [
            (&cubic, &["x", "x2", "x3"][..]),
            (&collinear, &["x1", "x2"][..]),
        ] {
            let r = run_negbin(ds, "y", xs, None).unwrap();
            assert!(r.converged, "{xs:?}: {r}");
            assert!(r.warnings.is_empty(), "{xs:?}: {:?}", r.warnings);
            assert!(r.theta > 0.5 && r.theta < 10.0, "{xs:?}: {r}");
            assert!(
                r.std_errors.iter().all(|s| s.is_finite() && *s > 0.0),
                "{xs:?}: {r}"
            );
        }
    }

    /// An all-zero outcome has no MLE (the coefficients run to -inf, the fitted means
    /// to 0): the fit must say so instead of reporting `converged = true` for whatever
    /// theta the alternation stopped at. Also with few rows, where the log-likelihood
    /// (-sum mu) stops changing while the means are still above glm.fit's 10 eps.
    #[test]
    fn test_negbin_all_zero_outcome_is_not_converged() {
        for n in [30, 5] {
            let x: Vec<f64> = (0..n)
                .map(|i| (i as f64 * 0.37).sin() + 0.01 * i as f64)
                .collect();
            let ds = Dataset::new(df! { "y" => vec![0.0; n], "x" => x }.unwrap());
            let r = run_negbin(&ds, "y", &["x"], None).unwrap();
            assert!(!r.converged, "n = {n}: {r}");
            // every row is a zero count the IRLS still drives to 0
            let all_rows = format!("of {n} zero-count rows");
            assert!(
                r.warnings.iter().any(|w| w.contains("IRLS")
                    && w.contains("maximum-likelihood")
                    && w.contains(&all_rows)),
                "n = {n}: {:?}",
                r.warnings
            );
        }
    }

    /// Zero counts that a regressor separates (every row with d = 1 has y = 0) have no
    /// finite MLE either: the coefficient of d runs to -inf, each IRLS step lowering the
    /// linear predictor of those rows by about 1. The fit must not be reported as
    /// converged, although theta (from the d = 0 rows) settles.
    #[test]
    fn test_negbin_separating_regressor_is_not_converged() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use rand_distr::{Distribution, Gamma, Poisson};
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let (mut y, mut d, mut x) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..90 {
            let di = f64::from(u8::from(i % 3 == 0));
            let xi = (i as f64 * 0.37).sin();
            let lambda = Gamma::new(2.0, (0.5 + 0.3 * xi).exp() / 2.0)
                .unwrap()
                .sample(&mut rng);
            let count: f64 = Poisson::new(lambda.max(1e-300)).unwrap().sample(&mut rng);
            y.push(if di == 1.0 { 0.0 } else { count });
            d.push(di);
            x.push(xi);
        }
        let ds = Dataset::new(df! { "y" => y, "d" => d, "x" => x }.unwrap());
        let r = run_negbin(&ds, "y", &["d", "x"], None).unwrap();
        assert!(!r.converged, "{r}");
        assert!(r.coefficients[1] < -10.0, "{r}");
        // the 30 rows with d = 1, not the zero counts among the others
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("maximum-likelihood") && w.contains("of 30 zero-count rows")),
            "{:?}",
            r.warnings
        );
    }

    /// Separated zero counts on more rows. The coefficient IRLS stops on its
    /// log-likelihood test, whose threshold grows with |ll|, so on 5000 rows it stops
    /// while the separated rows' fitted means are still above glm.fit's 10 eps (here
    /// 5e-14 to 1e-13): a verdict that needed such means reported this fit as
    /// converged (3 rounds), without a warning. The last IRLS step still lowers the
    /// linear predictor of each separated zero count by about 1, at any n. NB counts of
    /// size 2 (seeded), `mu = exp(0.5 + 0.1 x)`; a dummy on every 50th row, whose 100
    /// rows all have y = 0.
    #[test]
    fn test_negbin_separation_is_flagged_on_large_data() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use rand_distr::{Distribution, Gamma, Normal as NormalDist, Poisson};
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let normal = NormalDist::new(0.0, 1.0).unwrap();
        let (mut y, mut d, mut x) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..5000 {
            let di = f64::from(u8::from(i % 50 == 0));
            let xi: f64 = normal.sample(&mut rng);
            let lambda = Gamma::new(2.0, (0.5 + 0.1 * xi).exp() / 2.0)
                .unwrap()
                .sample(&mut rng);
            let count: f64 = Poisson::new(lambda.max(1e-300)).unwrap().sample(&mut rng);
            y.push(if di == 1.0 { 0.0 } else { count });
            d.push(di);
            x.push(xi);
        }
        let ds = Dataset::new(df! { "y" => y, "d" => d, "x" => x }.unwrap());
        let r = run_negbin(&ds, "y", &["d", "x"], None).unwrap();
        assert_eq!(r.n_obs, 5000);
        assert!(!r.converged, "{r}");
        assert!(r.coefficients[1] < -10.0, "{r}");
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("maximum-likelihood") && w.contains("of 100 zero-count rows")),
            "{:?}",
            r.warnings
        );
    }

    /// One zero count at an extreme regressor value has a fitted mean far below 10 eps
    /// (x1 = -80: eta about -38) but contributes nothing to the likelihood, so the fit is
    /// the one without that row and converges. R (MASS 7.3.60.0.1): glm.nb on
    /// negbin.csv plus the row (y = 0, x1 = -80, x2 = 0.1) converges to theta 1.602585,
    /// with glm.fit's "fitted rates numerically 0" warning from its Poisson start. The
    /// tiny mean is a note, not a missing MLE: the coefficients have stopped moving.
    #[test]
    fn test_negbin_high_leverage_zero_row_converges() {
        let ds = negbin_data();
        let col = |name: &str| -> Vec<f64> {
            ds.column(name)
                .unwrap()
                .as_materialized_series()
                .cast(&DataType::Float64)
                .unwrap()
                .f64()
                .unwrap()
                .into_no_null_iter()
                .collect()
        };
        let (mut y, mut x1, mut x2) = (col("y"), col("x1"), col("x2"));
        y.push(0.0);
        x1.push(-80.0);
        x2.push(0.1);
        let lev = Dataset::new(df! { "y" => y, "x1" => x1, "x2" => x2 }.unwrap());
        let base = run_negbin(&ds, "y", &["x1", "x2"], None).unwrap();
        let r = run_negbin(&lev, "y", &["x1", "x2"], None).unwrap();
        assert_eq!(r.n_obs, 401);
        assert!(r.converged, "{r}");
        // the note, and nothing else
        assert_eq!(r.warnings.len(), 1, "{:?}", r.warnings);
        assert!(
            r.warnings[0].contains("numerically 0")
                && r.warnings[0].contains("occurred for 1 row (")
                && r.warnings[0].contains("at converged coefficients"),
            "{:?}",
            r.warnings
        );
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-6 * b.abs();
        for (i, (a, b)) in r.coefficients.iter().zip(&base.coefficients).enumerate() {
            assert!(close(*a, *b), "coefficient {i}: {a} vs {b}");
        }
        assert!(
            close(r.theta, base.theta),
            "theta {} vs {}",
            r.theta,
            base.theta
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

    /// `init_theta` is a starting value (glm.nb's init.theta): the fit is the same MLE,
    /// from either end of the accepted range [1e-8, 1e6]. A start outside it is refused:
    /// below 1e-154 trigamma(theta) overflows, so theta could not move, and at 1e8 the
    /// theta score is at rounding level, so the alternation stopped where it started.
    #[test]
    fn test_validate_negbin_init_theta_is_a_starting_value() {
        let (case, ds) = load_case("disc_negbin");
        for init in [50.0, 0.2, 1e-8, 1e6] {
            let r = run_negbin(&ds, "y", &["x1", "x2"], Some(init)).unwrap();
            assert!(r.converged, "init {init}: {:?}", r.warnings);
            assert!(r.warnings.is_empty(), "init {init}: {:?}", r.warnings);
            Check::new(&case)
                .scalar("theta", r.theta, Tol::iterative(1e-6))
                .scalar("theta_init50_pkg", r.theta, Tol::iterative(1e-6))
                .named("coef", &r.variables, &r.coefficients, Tol::iterative(1e-6))
                .finish();
        }
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e-300, 1e-9, 2e6, 1e300] {
            match run_negbin(&ds, "y", &["x1", "x2"], Some(bad)) {
                Err(EconError::InvalidSpecification { message }) => assert!(
                    message.contains("init_theta")
                        && message.contains("1e-8")
                        && message.contains("1e6"),
                    "{message}"
                ),
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

    /// The intercept-only model, glm.nb(y ~ 1).
    #[test]
    fn test_validate_negbin_intercept_only_disc_negbin() {
        let (case, ds) = load_case("disc_negbin");
        let r = run_negbin(&ds, "y", &[], None).unwrap();
        assert!(r.converged, "{r}");
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
        assert_eq!(r.variables, ["(Intercept)"]);
        Check::new(&case)
            .named(
                "int_coef",
                &r.variables,
                &r.coefficients,
                Tol::iterative(1e-6),
            )
            .named("int_se", &r.variables, &r.std_errors, Tol::iterative(1e-5))
            .scalar("int_theta", r.theta, Tol::iterative(1e-6))
            .scalar("int_se_theta", r.theta_std_error, Tol::iterative(1e-5))
            .scalar("int_loglik", r.log_likelihood, Tol::stat(r.n_obs))
            .finish();
    }

    /// R's trigamma at the points the NB theta information needs (`trigamma_x` of the
    /// `disc_negbin` case, validation/r/cases/discrete.R), to 1e-12 relative. The series
    /// it replaced stopped after `1/(30 x^5)` and was off by 4.5e-7 relative at 6 (R:
    /// `x <- 6; abs(1/x + 1/(2*x^2) + 1/(6*x^3) - 1/(30*x^5) - trigamma(x)) /
    /// trigamma(x)`).
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

    /// theta_newton says when it cannot move: at theta = 1e-300 trigamma(theta)
    /// overflows, so the information is not finite and theta is not updated.
    #[test]
    fn test_theta_newton_reports_a_theta_it_cannot_update() {
        let (mu, y) = ([1.0, 2.0, 0.5], [0.0, 3.0, 1.0]);
        let (theta, updated) = theta_newton(1e-300, &mu, &y).unwrap();
        assert!(!updated && theta == 1e-300, "{theta} {updated}");
        let (theta, updated) = theta_newton(1.0, &mu, &y).unwrap();
        assert!(
            updated && theta.is_finite() && theta > 0.0,
            "{theta} {updated}"
        );
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

    /// The 12-row fixture of `test_zip_basic` separates the zero part. R (pscl 1.5.9):
    /// `pscl::zeroinfl(y ~ x, data = data.frame(y = c(0, 0, 0, 0, 0, 1, 2, 0, 3, 0, 5,
    /// 4), x = c(1, 1.5, 2, 2.5, 3, 4, 5, 3.5, 6, 4.5, 7, 6.5)))` (BFGS) reports zero
    /// coefficients 94.70464, -25.39549 and logLik -8.370407 as converged. The supremum
    /// is higher (with `control = zeroinfl.control(EM = TRUE)` the logLik is -8.368208),
    /// so the fit must be at least pscl's and flag the separation.
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

    /// One zero count at an extreme regressor value (x1 = -90: logit(pi) about -71)
    /// has a fitted zero-inflation probability far below 10 eps, but the fit is regular:
    /// the warning is glm.fit's diagnostic with the row count, the fit converges and
    /// equals the one without that row.
    #[test]
    fn test_zinb_extreme_row_is_a_note() {
        let ds = zi_data();
        let col = |name: &str| -> Vec<f64> {
            ds.column(name)
                .unwrap()
                .as_materialized_series()
                .cast(&DataType::Float64)
                .unwrap()
                .f64()
                .unwrap()
                .into_no_null_iter()
                .collect()
        };
        let (mut y, mut x1, mut x2) = (col("y_nb"), col("x1"), col("x2"));
        y.push(0.0);
        x1.push(-90.0);
        x2.push(0.0);
        let ext = Dataset::new(df! { "y_nb" => y, "x1" => x1, "x2" => x2 }.unwrap());
        let base = run_zinb(&ds, "y_nb", &["x1", "x2"], None).unwrap();
        let r = run_zinb(&ext, "y_nb", &["x1", "x2"], None).unwrap();
        assert!(r.converged, "{r}");
        assert_eq!(r.warnings.len(), 1, "{:?}", r.warnings);
        let w = &r.warnings[0];
        assert!(
            w.contains("numerically 0 or 1 occurred for 1 row")
                && w.contains("possible quasi-separation")
                && !w.contains("not reliable"),
            "{w}"
        );
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-6 * b.abs();
        for (what, a, b) in [
            ("count", &r.count_coefficients, &base.count_coefficients),
            ("zero", &r.zero_coefficients, &base.zero_coefficients),
        ] {
            for (i, (u, v)) in a.iter().zip(b.iter()).enumerate() {
                assert!(close(*u, *v), "{what}[{i}]: {u} vs {v}");
            }
        }
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

    /// The zero part on its intercept alone: zeroinfl(y ~ x1 + x2 | 1).
    #[test]
    fn test_validate_zip_intercept_only_zero_part() {
        let (case, ds) = load_case("disc_zeroinfl");
        let r = run_zip(&ds, "y", &["x1", "x2"], Some(&[])).unwrap();
        assert!(r.converged, "{:?}", r.warnings);
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
        assert_eq!(r.zero_variables, ["(Intercept)"]);
        Check::new(&case)
            .named(
                "zip_z0_count",
                &r.count_variables,
                &r.count_coefficients,
                Tol::iterative(1e-6),
            )
            .named(
                "zip_z0_zero",
                &r.zero_variables,
                &r.zero_coefficients,
                Tol::iterative(1e-6),
            )
            .vector("zip_z0_se_count", &r.count_std_errors, Tol::iterative(1e-5))
            .vector("zip_z0_se_zero", &r.zero_std_errors, Tol::iterative(1e-5))
            .scalar("zip_z0_loglik", r.log_likelihood, Tol::stat(r.n_obs))
            .finish();
    }

    /// ZINB on counts without overdispersion: the positive counts of the case's `y` are
    /// replaced by the rounded count mean `exp(0.6 + 0.25 x1 - 0.3 x2)` of its R code
    /// (at least 1), so they vary less than Poisson counts. theta runs to its bound and
    /// the fit is flagged (not converged, NaN theta SEs, a warning that names the
    /// zero-inflated Poisson model), and the count and zero parts are the ZIP fit, with
    /// the SEs of its information instead of NaN.
    #[test]
    fn test_zinb_without_overdispersion_is_flagged() {
        let ds = zi_data();
        let col = |name: &str| -> Vec<f64> {
            ds.column(name)
                .unwrap()
                .as_materialized_series()
                .cast(&DataType::Float64)
                .unwrap()
                .f64()
                .unwrap()
                .into_no_null_iter()
                .collect()
        };
        let (y, x1, x2) = (col("y"), col("x1"), col("x2"));
        let y_under: Vec<f64> = y
            .iter()
            .zip(x1.iter().zip(&x2))
            .map(|(&yi, (&a, &b))| {
                if yi == 0.0 {
                    0.0
                } else {
                    (0.6 + 0.25 * a - 0.3 * b).exp().round().max(1.0)
                }
            })
            .collect();
        let ds = with_column(&ds, Column::new("y_under".into(), y_under));
        let zip = run_zip(&ds, "y_under", &["x1", "x2"], None).unwrap();
        assert!(zip.converged && zip.warnings.is_empty(), "{zip}");
        let r = run_zinb(&ds, "y_under", &["x1", "x2"], None).unwrap();
        assert!(!r.converged, "{r}");
        assert!(r.theta.unwrap() > 1e6, "{r}");
        assert!(r.theta_std_error.unwrap().is_nan(), "{r}");
        assert!(r.log_theta_std_error.unwrap().is_nan(), "{r}");
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("overdispersion") && w.contains("pscl::zeroinfl")),
            "{:?}",
            r.warnings
        );
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-5 * b.abs();
        for (what, a, b) in [
            ("count", &r.count_coefficients, &zip.count_coefficients),
            ("zero", &r.zero_coefficients, &zip.zero_coefficients),
            ("se_count", &r.count_std_errors, &zip.count_std_errors),
            ("se_zero", &r.zero_std_errors, &zip.zero_std_errors),
        ] {
            for (i, (u, v)) in a.iter().zip(b.iter()).enumerate() {
                assert!(close(*u, *v), "{what}[{i}]: ZINB {u} vs ZIP {v}\n{r}");
            }
        }
        assert!(
            close(r.log_likelihood, zip.log_likelihood),
            "LL {} vs {}",
            r.log_likelihood,
            zip.log_likelihood
        );
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

    /// The fit must not depend on the units of a regressor (income, population): with
    /// x1 multiplied by `s` (count and zero part), the x1 coefficients and SEs are the
    /// unscaled ones divided by `s`, everything else, z and the log-likelihood
    /// included, is unchanged, and the fit converges at the same iteration. A Hessian by
    /// central differences with a step relative to the parameter value did not have this
    /// property (its z values moved by 10% at s = 1e5), nor did a convergence test on
    /// max |gradient| (the x1 component is multiplied by `s`: ZIP at s = 1e10 ran to the
    /// iteration limit). Each scaled fit is also checked against pscl's case, x1
    /// rescaled.
    #[test]
    fn test_validate_zeroinfl_regressor_scale_disc_zeroinfl() {
        let (case, ds) = load_case("disc_zeroinfl");
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-7 * b.abs();
        for (p, dist, y) in [
            ("zip_", ZeroInflatedType::Poisson, "y"),
            ("zinb_", ZeroInflatedType::NegBin, "y_nb"),
        ] {
            let fit = |ds: &Dataset| {
                run_zeroinfl(ds, y, &["x1", "x2"], Some(&["x1", "x2"]), dist).unwrap()
            };
            let base = fit(&ds);
            for s in [1e4, 1e6, 1e10, 1e12] {
                let x1 =
                    (ds.column("x1").unwrap().as_materialized_series() * s).with_name("x1".into());
                let r = fit(&with_column(&ds, x1.into_column()));
                assert!(r.converged, "{p} x1 * {s:e}: {:?}", r.warnings);
                // the iterates are the unscaled ones, and so is the stopping iteration
                assert_eq!(r.iterations, base.iterations, "{p} x1 * {s:e}");
                assert!(r.warnings.is_empty(), "{p} x1 * {s:e}: {:?}", r.warnings);
                // x1 is the second column of both parts
                let unscale = |v: &[f64]| -> Vec<f64> {
                    v.iter()
                        .enumerate()
                        .map(|(i, &b)| if i == 1 { b * s } else { b })
                        .collect()
                };
                let pairs = [
                    (
                        "count",
                        unscale(&r.count_coefficients),
                        &base.count_coefficients,
                    ),
                    (
                        "zero",
                        unscale(&r.zero_coefficients),
                        &base.zero_coefficients,
                    ),
                    (
                        "se_count",
                        unscale(&r.count_std_errors),
                        &base.count_std_errors,
                    ),
                    (
                        "se_zero",
                        unscale(&r.zero_std_errors),
                        &base.zero_std_errors,
                    ),
                    ("z_count", r.count_z_stats.clone(), &base.count_z_stats),
                    ("z_zero", r.zero_z_stats.clone(), &base.zero_z_stats),
                ];
                for (what, scaled, unscaled) in &pairs {
                    for (i, (a, b)) in scaled.iter().zip(unscaled.iter()).enumerate() {
                        assert!(close(*a, *b), "{p}{what}[{i}], x1 * {s:e}: {a} vs {b}");
                    }
                }
                assert!(
                    close(r.log_likelihood, base.log_likelihood),
                    "{p} x1 * {s:e}: LL {} vs {}",
                    r.log_likelihood,
                    base.log_likelihood
                );
                if let (Some(a), Some(b)) = (r.log_theta_std_error, base.log_theta_std_error) {
                    assert!(close(a, b), "{p} x1 * {s:e}: SE(log theta) {a} vs {b}");
                }
                let k = |q: &str| format!("{p}{q}");
                Check::new(&case)
                    .named(
                        &k("count"),
                        &r.count_variables,
                        &unscale(&r.count_coefficients),
                        Tol::iterative(1e-6),
                    )
                    .named(
                        &k("zero"),
                        &r.zero_variables,
                        &unscale(&r.zero_coefficients),
                        Tol::iterative(1e-6),
                    )
                    .vector(
                        &k("se_count"),
                        &unscale(&r.count_std_errors),
                        Tol::iterative(1e-5),
                    )
                    .vector(
                        &k("se_zero"),
                        &unscale(&r.zero_std_errors),
                        Tol::iterative(1e-5),
                    )
                    .vector(&k("z_count"), &r.count_z_stats, Tol::iterative(1e-5))
                    .vector(&k("z_zero"), &r.zero_z_stats, Tol::iterative(1e-5))
                    .scalar(&k("loglik"), r.log_likelihood, Tol::stat(r.n_obs))
                    .finish();
            }
        }
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

    /// Column errors name the column, for the outcome, the count regressors and the
    /// zero-part (hurdle: binary-part) regressors of every count model: an absent column
    /// is ColumnNotFound, a text regressor with a numeric outcome NonNumericColumn.
    #[test]
    fn test_na_count_models_column_errors() {
        let (nb, zi, text) = (negbin_data(), zi_data(), ordered_text_data());
        let poisson = HurdleType::Poisson;
        let not_found = [
            (
                "negbin y",
                run_negbin(&nb, "nope", &["x1"], None).map(|_| ()),
            ),
            (
                "negbin x",
                run_negbin(&nb, "y", &["nope"], None).map(|_| ()),
            ),
            ("zip y", run_zip(&zi, "nope", &["x1"], None).map(|_| ())),
            ("zip x", run_zip(&zi, "y", &["nope"], None).map(|_| ())),
            (
                "zip z",
                run_zip(&zi, "y", &["x1"], Some(&["nope"])).map(|_| ()),
            ),
            ("zinb x", run_zinb(&zi, "y_nb", &["nope"], None).map(|_| ())),
            (
                "zinb z",
                run_zinb(&zi, "y_nb", &["x1"], Some(&["nope"])).map(|_| ()),
            ),
            (
                "hurdle y",
                run_hurdle(&zi, "nope", &["x1"], None, poisson).map(|_| ()),
            ),
            (
                "hurdle x",
                run_hurdle(&zi, "y", &["nope"], None, poisson).map(|_| ()),
            ),
            (
                "hurdle z",
                run_hurdle(&zi, "y", &["x1"], Some(&["nope"]), poisson).map(|_| ()),
            ),
        ];
        for (what, r) in not_found {
            match r {
                Err(EconError::ColumnNotFound { column, .. }) => {
                    assert_eq!(column, "nope", "{what}")
                }
                other => panic!("{what}: expected ColumnNotFound, got {other:?}"),
            }
        }
        // `y` of disc_ordered is numeric (1, 2, 3), `y_label` its text labels
        let non_numeric = [
            (
                "negbin x",
                run_negbin(&text, "y", &["y_label"], None).map(|_| ()),
            ),
            ("zip x", run_zip(&text, "y", &["y_label"], None).map(|_| ())),
            (
                "zip z",
                run_zip(&text, "y", &["x1"], Some(&["y_label"])).map(|_| ()),
            ),
            (
                "zinb x",
                run_zinb(&text, "y", &["y_label"], None).map(|_| ()),
            ),
            (
                "hurdle x",
                run_hurdle(&text, "y", &["y_label"], None, poisson).map(|_| ()),
            ),
            (
                "hurdle z",
                run_hurdle(&text, "y", &["x1"], Some(&["y_label"]), poisson).map(|_| ()),
            ),
        ];
        for (what, r) in non_numeric {
            match r {
                Err(EconError::NonNumericColumn { column }) => {
                    assert_eq!(column, "y_label", "{what}")
                }
                other => panic!("{what}: expected NonNumericColumn, got {other:?}"),
            }
        }
    }

    /// pscl's analytic gradient and the analytic Hessian (the private `zi_loglik`) are
    /// the derivatives of the log-likelihood: central differences (step 1e-6) of the
    /// log-likelihood and of the gradient at a point away from the optimum, for both
    /// distributions, with the zero part on all count regressors and on fewer. The
    /// log-likelihood alone equals the one computed with the derivatives.
    #[test]
    fn test_zeroinfl_analytic_derivatives_match_central_differences() {
        let ds = zi_data();
        let cols = ColumnRequest::new(ds.df())
            .numerics(&["y_nb", "x1", "x2"])
            .extract()
            .unwrap();
        let x = cols.matrix(&["x1", "x2"], true);
        let z1 = cols.matrix(&["x1"], true);
        let y = cols.numeric("y_nb");
        let h = 1e-6;
        for (dist, z, par) in [
            (
                ZeroInflatedType::Poisson,
                &x,
                vec![0.5, 0.2, -0.2, -0.3, 0.4, 0.1],
            ),
            (
                ZeroInflatedType::NegBin,
                &x,
                vec![0.5, 0.2, -0.2, -0.3, 0.4, 0.1, 1.5f64.ln()],
            ),
            (
                ZeroInflatedType::NegBin,
                &z1,
                vec![0.5, 0.2, -0.2, -0.3, 0.4, 0.7f64.ln()],
            ),
        ] {
            let data = ZiData {
                x: x.view(),
                z: z.view(),
                y,
                dist,
            };
            let m = par.len();
            let mut g = vec![0.0; m];
            let mut hess = Array2::zeros((m, m));
            let ll = zi_loglik(&data, &par, Some(&mut g), Some(&mut hess)).unwrap();
            let ll_only = zi_loglik(&data, &par, None, None).unwrap();
            assert!(
                (ll - ll_only).abs() <= 1e-12 * ll.abs(),
                "{dist:?}: {ll} vs {ll_only}"
            );
            for k in 0..m {
                let (mut up, mut dn) = (par.clone(), par.clone());
                up[k] += h;
                dn[k] -= h;
                let (mut g_up, mut g_dn) = (vec![0.0; m], vec![0.0; m]);
                let fd = (zi_loglik(&data, &up, Some(&mut g_up), None).unwrap()
                    - zi_loglik(&data, &dn, Some(&mut g_dn), None).unwrap())
                    / (2.0 * h);
                assert!(
                    (fd - g[k]).abs() <= 1e-6 * (1.0 + g[k].abs()),
                    "{dist:?}: gradient[{k}] {} vs central difference {fd}",
                    g[k]
                );
                for a in 0..m {
                    let fd = (g_up[a] - g_dn[a]) / (2.0 * h);
                    assert!(
                        (fd - hess[[a, k]]).abs() <= 1e-5 * (1.0 + hess[[a, k]].abs()),
                        "{dist:?}: Hessian[{a}, {k}] {} vs central difference {fd}",
                        hess[[a, k]]
                    );
                }
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
