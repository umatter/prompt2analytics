//! Shared traits for econometric estimators.
//!
//! This module provides the `LinearEstimator` trait that unifies the interface
//! across OLS, panel data, IV, and other linear estimators.
//!
//! It also holds the shared distribution helpers: upper-tail p-values ([`t_test_p_value`],
//! [`chi_squared_p_value`], [`f_test_p_value`], [`normal_p_value_two_sided`]) and quantiles
//! ([`normal_quantile`], [`z_critical`], [`t_quantile`], [`t_critical`],
//! [`chi_squared_quantile`]). Tails are computed directly in log space (no `1 - cdf`, which
//! rounded every p-value below about 1e-16 to 0), and quantiles by a safeguarded Newton
//! iteration on the log tail. Against R 4.3.3: `t_quantile` within 3.6e-13 relative for df >= 1,
//! `chi_squared_quantile` within 8.0e-13 for df <= 1e6, `t_test_p_value` within 3.2e-13 and
//! `chi_squared_p_value` within 1.1e-13, `f_test_p_value` within 6.1e-13 (7.7e-10 near F = 1
//! with both df near 1e6).
//!
//! statrs's inverse and survival functions for these distributions are not used, because:
//! `StudentsT::inverse_cdf` does not terminate for very large df (and at q = 1e-100, df = 1000);
//! it panics for q < 1e-154; it is 6.1e-5 off at df = 1e6; `ChiSquared::sf` returns 1 for
//! x < 1.1e-15 at any df; `ChiSquared::inverse_cdf` returns a coarse bisection point when the
//! density overflows (e.g. qchisq(0.025, 318)); and `FisherSnedecor::sf` loses accuracy far in
//! the tail. statrs still provides `ln_gamma` and the normal distribution; the two-sided normal
//! tail is the chi-square(1) upper tail of z^2, because statrs's `erfc` is only accurate to
//! about 5e-11 relative (2 * pnorm(-1.96) came out 4.5e-11 off).

use crate::errors::{EconError, EconResult};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use statrs::distribution::{Continuous, ContinuousCDF, Normal};

/// Significance level indicators for p-values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignificanceLevel {
    /// p > 0.10
    NotSignificant,
    /// p < 0.10 (†)
    TenPercent,
    /// p < 0.05 (*)
    FivePercent,
    /// p < 0.01 (**)
    OnePercent,
    /// p < 0.001 (***)
    TenthPercent,
}

impl SignificanceLevel {
    /// Determine significance level from a p-value.
    pub fn from_p_value(p: f64) -> Self {
        if p < 0.001 {
            Self::TenthPercent
        } else if p < 0.01 {
            Self::OnePercent
        } else if p < 0.05 {
            Self::FivePercent
        } else if p < 0.10 {
            Self::TenPercent
        } else {
            Self::NotSignificant
        }
    }

    /// Get the star notation for this significance level.
    pub fn stars(&self) -> &'static str {
        match self {
            Self::NotSignificant => "",
            Self::TenPercent => "†",
            Self::FivePercent => "*",
            Self::OnePercent => "**",
            Self::TenthPercent => "***",
        }
    }

    /// Get a short code for this significance level (for CSV export).
    pub fn code(&self) -> &'static str {
        match self {
            Self::NotSignificant => "",
            Self::TenPercent => "p<0.10",
            Self::FivePercent => "p<0.05",
            Self::OnePercent => "p<0.01",
            Self::TenthPercent => "p<0.001",
        }
    }
}

impl std::fmt::Display for SignificanceLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.stars())
    }
}

/// Core trait for linear regression-type estimators.
///
/// This trait provides a unified interface for accessing estimation results
/// and computing derived statistics. Implementors only need to provide the
/// core results; derived statistics are computed via default implementations.
pub trait LinearEstimator {
    /// Get the estimated coefficients.
    fn coefficients(&self) -> &Array1<f64>;

    /// Get the standard errors of the coefficients.
    fn std_errors(&self) -> &Array1<f64>;

    /// Get the residuals (y - X*beta).
    fn residuals(&self) -> &Array1<f64>;

    /// Get the variance-covariance matrix of the coefficients.
    fn vcov_matrix(&self) -> &Array2<f64>;

    /// Get the names of the variables (coefficients).
    fn variable_names(&self) -> &[String];

    /// Get the degrees of freedom for t-tests.
    fn degrees_of_freedom(&self) -> usize;

    /// Get the number of observations.
    fn n_obs(&self) -> usize;

    /// Compute t-statistics: beta / se(beta)
    fn t_stats(&self) -> Array1<f64> {
        let coefs = self.coefficients();
        let ses = self.std_errors();
        coefs / ses
    }

    /// Compute two-sided p-values from t-statistics with `degrees_of_freedom()` df.
    ///
    /// Goes through [`t_test_p_value`]: a NaN t-statistic (for example from a NaN or
    /// zero standard error) gives a NaN p-value, an infinite one gives 0, and df = 0
    /// gives NaN. It never panics.
    fn p_values(&self) -> Array1<f64> {
        let df = self.degrees_of_freedom() as f64;
        self.t_stats().mapv(|t| t_test_p_value(t, df))
    }

    /// Get significance levels for each coefficient.
    fn significance(&self) -> Vec<SignificanceLevel> {
        self.p_values()
            .iter()
            .map(|&p| SignificanceLevel::from_p_value(p))
            .collect()
    }

    /// Compute confidence intervals at the given level (e.g., 0.95 for 95%).
    fn confidence_intervals(&self, level: f64) -> Vec<(f64, f64)> {
        let coefs = self.coefficients();
        let ses = self.std_errors();
        let df = self.degrees_of_freedom() as f64;

        if !df.is_finite() || df <= 0.0 || !level.is_finite() || !(0.0 < level && level < 1.0) {
            return vec![(f64::NAN, f64::NAN); coefs.len()];
        }

        let alpha = 1.0 - level;
        let t_crit = match t_quantile(1.0 - alpha / 2.0, df) {
            Ok(q) => q,
            Err(_) => return vec![(f64::NAN, f64::NAN); coefs.len()],
        };

        coefs
            .iter()
            .zip(ses.iter())
            .map(|(&b, &se)| (b - t_crit * se, b + t_crit * se))
            .collect()
    }

    /// Compute 95% confidence intervals.
    fn confidence_intervals_95(&self) -> Vec<(f64, f64)> {
        self.confidence_intervals(0.95)
    }

    /// Compute R-squared (coefficient of determination).
    /// R² = 1 - SSR/SST where SSR = sum(residuals²), SST = sum((y - mean(y))²)
    /// Note: Default implementation returns NAN. Subclasses should override.
    fn r_squared(&self) -> f64 {
        // Default: return NAN since we need original y values to compute SST
        f64::NAN
    }

    /// Compute adjusted R-squared: `1 - (1 - R²)(n - 1)/df`, with
    /// df = `degrees_of_freedom()` = n - k, where k counts every coefficient including
    /// the intercept (R `summary.lm`). Assumes a model with an intercept. NaN when
    /// df = 0, and NaN whenever `r_squared()` is (its default).
    fn adj_r_squared(&self) -> f64 {
        let r2 = self.r_squared();
        let n = self.n_obs() as f64;
        let df = self.degrees_of_freedom();

        if df == 0 {
            return f64::NAN;
        }

        1.0 - (1.0 - r2) * (n - 1.0) / df as f64
    }

    /// Residual standard error: sqrt(SSR / (n - k))
    fn residual_std_error(&self) -> f64 {
        let residuals = self.residuals();
        let df = self.degrees_of_freedom() as f64;

        if df <= 0.0 {
            return f64::NAN;
        }

        let ssr: f64 = residuals.iter().map(|&r| r * r).sum();
        (ssr / df).sqrt()
    }

    /// Gaussian log-likelihood at the ML error variance σ̂² = SSR/n (R `logLik` for `lm`).
    fn log_likelihood(&self) -> f64 {
        let residuals = self.residuals();
        let n = self.n_obs() as f64;
        let sigma2: f64 = residuals.iter().map(|&r| r * r).sum::<f64>() / n;

        if sigma2 <= 0.0 {
            return f64::NAN;
        }

        -n / 2.0 * (1.0 + (2.0 * std::f64::consts::PI * sigma2).ln())
    }

    /// Akaike Information Criterion: AIC = 2(k + 1) - 2 ln L, with k coefficients.
    ///
    /// The error variance counts as a parameter, as in R's `AIC` for `lm`.
    /// Implementors with another parameter count override this.
    fn aic(&self) -> f64 {
        let p = self.coefficients().len() as f64 + 1.0;
        let ll = self.log_likelihood();
        2.0 * p - 2.0 * ll
    }

    /// Bayesian Information Criterion: BIC = (k + 1) ln n - 2 ln L, with k coefficients.
    ///
    /// The error variance counts as a parameter, as in R's `BIC` for `lm`.
    /// Implementors with another parameter count override this.
    fn bic(&self) -> f64 {
        let p = self.coefficients().len() as f64 + 1.0;
        let n = self.n_obs() as f64;
        let ll = self.log_likelihood();
        p * n.ln() - 2.0 * ll
    }
}

/// Check that a level or probability is finite and strictly between 0 and 1.
pub fn validate_level(level: f64, name: &str) -> EconResult<()> {
    if level.is_finite() && level > 0.0 && level < 1.0 {
        Ok(())
    } else {
        Err(EconError::InvalidSpecification {
            message: format!("{name} must be strictly between 0 and 1, got {level}"),
        })
    }
}

/// Standard normal quantile Phi^-1(p), 0 < p < 1 (statrs, Boost erf_inv).
pub fn normal_quantile(p: f64) -> EconResult<f64> {
    validate_level(p, "probability")?;
    Ok(Normal::standard().inverse_cdf(p))
}

/// Two-sided normal critical value Phi^-1((1 + level) / 2).
///
/// ```
/// use p2a_core::traits::z_critical;
/// assert!((z_critical(0.95).unwrap() - 1.959963984540054).abs() < 1e-12);
/// ```
pub fn z_critical(level: f64) -> EconResult<f64> {
    validate_level(level, "level")?;
    normal_quantile((1.0 + level) / 2.0)
}

/// Student t quantile (R `qt`). df > 0; df = +inf or > 1e20 gives the normal quantile (as R).
///
/// ```
/// use p2a_core::traits::t_quantile;
/// assert!((t_quantile(0.975, 10.0).unwrap() - 2.2281388519862739).abs() < 1e-12);
/// ```
pub fn t_quantile(p: f64, df: f64) -> EconResult<f64> {
    validate_level(p, "probability")?;
    if df.is_nan() || df <= 0.0 {
        return Err(EconError::InvalidSpecification {
            message: format!("degrees of freedom must be positive, got {df}"),
        });
    }
    if p == 0.5 {
        return Ok(0.0);
    }
    let z = normal_quantile(p)?;
    if df > 1e20 {
        return Ok(z);
    }
    if df >= 1000.0 && z * z <= 0.005 * df {
        return Ok(tails::cornish_fisher_t(z, df));
    }
    let x = tails::t_lower_quantile(p.min(1.0 - p), df).map_err(convergence_error)?;
    Ok(if p < 0.5 { x } else { -x })
}

/// Chi-square quantile (R `qchisq`); df finite and > 0.
pub fn chi_squared_quantile(p: f64, df: f64) -> EconResult<f64> {
    validate_level(p, "probability")?;
    if !df.is_finite() || df <= 0.0 {
        return Err(EconError::InvalidSpecification {
            message: format!("degrees of freedom must be finite and positive, got {df}"),
        });
    }
    tails::chi_squared_quantile(p, df).map_err(convergence_error)
}

/// Two-sided normal p-value 2 * Phi(-|z|) = P(chi2_1 > z^2); NaN for NaN.
pub fn normal_p_value_two_sided(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    tails::normal_two_sided(z.abs())
}

/// Two-sided p-value of a t statistic, 2 P(T_df > |t|), computed as an upper tail (no `1 - cdf`).
pub fn t_test_p_value(t_stat: f64, df: f64) -> f64 {
    if !df.is_finite() || df <= 0.0 || t_stat.is_nan() {
        return f64::NAN;
    }
    if t_stat.is_infinite() {
        return 0.0;
    }
    if t_stat == 0.0 {
        return 1.0;
    }
    let at = t_stat.abs();
    let p = if df >= 1000.0 && at * at <= 0.004 * df {
        tails::normal_two_sided(tails::z_of_t(at, df))
    } else {
        match tails::t_ln_lower_tail(at, at.ln(), df) {
            Ok(l) => 2.0 * l.exp(),
            Err(_) => f64::NAN,
        }
    };
    p.min(1.0)
}

/// Upper-tail p-value P(chi2_df > chi2).
pub fn chi_squared_p_value(chi2: f64, df: f64) -> f64 {
    if !df.is_finite() || df <= 0.0 || chi2.is_nan() || chi2 < 0.0 {
        return f64::NAN;
    }
    if chi2.is_infinite() {
        return 0.0;
    }
    if chi2 == 0.0 {
        return 1.0;
    }
    tails::chi2_ln_tail(chi2, df, false).map_or(f64::NAN, f64::exp)
}

/// Upper-tail p-value P(F_{df1, df2} > f_stat).
pub fn f_test_p_value(f_stat: f64, df1: f64, df2: f64) -> f64 {
    if !df1.is_finite()
        || df1 <= 0.0
        || !df2.is_finite()
        || df2 <= 0.0
        || f_stat.is_nan()
        || f_stat < 0.0
    {
        return f64::NAN;
    }
    if f_stat.is_infinite() {
        return 0.0;
    }
    if f_stat == 0.0 {
        return 1.0;
    }
    tails::f_ln_upper_tail(f_stat, df1, df2).map_or(f64::NAN, f64::exp)
}

/// Critical value for a two-sided t test at significance `alpha` (NaN for invalid input, as before).
pub fn t_critical(alpha: f64, df: f64) -> f64 {
    if !df.is_finite() || df <= 0.0 || !alpha.is_finite() || !(0.0 < alpha && alpha < 1.0) {
        return f64::NAN;
    }
    t_quantile(1.0 - alpha / 2.0, df).unwrap_or(f64::NAN)
}

fn convergence_error(e: tails::NoConvergence) -> EconError {
    EconError::ConvergenceFailure {
        iterations: e.iterations,
        last_change: e.last_change,
        suggestion: format!(
            "{} did not converge; the degrees of freedom are likely beyond the supported range \
             (about 1e12, see the p2a_core::traits::estimator docs); if not, please report the arguments",
            e.what
        ),
    }
}

/// Log-space tail probabilities and quantile solvers behind the helpers above.
///
/// - The incomplete beta continued fraction and the incomplete gamma series and continued fraction
///   are statrs 0.18's algorithms, with the prefactor kept in logs so that no tail underflows before
///   1e-308.
/// - Quantiles use a safeguarded Newton iteration on ln(tail) in u = ln|x|.
/// - Student's t with df >= 1000 uses the four-term Cornish-Fisher expansion (Fisher 1925; A&S 26.7.5)
///   where z^2 <= 0.005 df. There its truncation error is below double precision, verified against R
///   up to df = 1e12.
/// - Iteration caps bound the supported df. The chi-square series needs about 5 sqrt(df) terms near
///   the centre and reaches its 1e7 cap at df of about 4e12 (the continued fraction stays below its
///   cap there); the F continued fraction was checked to df1, df2 = 1e12. Up to df = 1e12 every helper
///   converges. Beyond the caps the p-value helpers return NaN and the quantile helpers
///   ConvergenceFailure (chi-square p-values and quantiles near the centre for df >= 5e12).
mod tails {
    use statrs::distribution::{ContinuousCDF, Normal};
    use statrs::function::gamma::ln_gamma;

    const LN_2: f64 = std::f64::consts::LN_2;
    const LN_HALF: f64 = -std::f64::consts::LN_2;
    const LN_PI: f64 = 1.144_729_885_849_400_2;
    const LN_SQRT_PI: f64 = 0.572_364_942_924_700_1;
    const EPS: f64 = 1.110_223_024_625_156_5e-16; // 2^-53, statrs F64_PREC
    const FPMIN: f64 = f64::MIN_POSITIVE / EPS;
    const CF_MAX: usize = 1_000_000;
    const SERIES_MAX: usize = 10_000_000;
    const NEWTON_MAX: usize = 200;
    const STEP_CLAMP: f64 = 5.0;
    const STEP_TOL: f64 = 1e-14;
    const STEP_ACCEPT: f64 = 1e-8;
    const EXP_MAX: f64 = 709.78;

    /// An iteration that hit its cap (beyond the supported df range, see the module docs; reported as
    /// ConvergenceFailure / NaN).
    #[derive(Debug, Clone, Copy)]
    pub(super) struct NoConvergence {
        pub(super) what: &'static str,
        pub(super) iterations: usize,
        pub(super) last_change: f64,
    }
    type R = Result<f64, NoConvergence>;

    fn exp_or_inf(u: f64) -> f64 {
        if u < EXP_MAX { u.exp() } else { f64::INFINITY }
    }
    fn normal_quantile(p: f64) -> f64 {
        Normal::standard().inverse_cdf(p)
    }
    /// lnGamma(z) minus Stirling's leading terms; truncation < 1e-21 for z >= 100.
    fn stirling_tail(z: f64) -> f64 {
        let z2 = z * z;
        (1.0 / 12.0 - (1.0 / 360.0 - (1.0 / 1260.0 - 1.0 / (1680.0 * z2)) / z2) / z2) / z
    }
    /// lnGamma(a + b) - lnGamma(a); without cancellation for a >= 100 (callers pass b <= a there).
    fn ln_gamma_diff(a: f64, b: f64) -> f64 {
        if a < 100.0 {
            ln_gamma(a + b) - ln_gamma(a)
        } else {
            (a - 0.5) * (b / a).ln_1p() + b * (a + b).ln() - b + stirling_tail(a + b)
                - stirling_tail(a)
        }
    }
    fn ln_beta_prefactor(a: f64, b: f64) -> f64 {
        let (big, small) = if a >= b { (a, b) } else { (b, a) };
        ln_gamma_diff(big, small) - ln_gamma(small)
    }
    /// ln I_x(a, b) for x < (a + 1)/(a + b + 2): modified Lentz continued fraction (statrs
    /// `checked_beta_reg`), prefactor in logs; ln_y = ln(1 - x).
    fn ln_beta_cf(a: f64, b: f64, x: f64, ln_x: f64, ln_y: f64) -> R {
        let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
        let mut c = 1.0;
        let mut d = 1.0 - qab * x / qap;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        d = 1.0 / d;
        let mut h = d;
        let mut m = 0usize;
        loop {
            m += 1;
            if m > CF_MAX {
                return Err(NoConvergence {
                    what: "incomplete beta continued fraction",
                    iterations: CF_MAX,
                    last_change: f64::NAN,
                });
            }
            let mf = m as f64;
            let m2 = 2.0 * mf;
            let mut aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d;
            if d.abs() < FPMIN {
                d = FPMIN;
            }
            c = 1.0 + aa / c;
            if c.abs() < FPMIN {
                c = FPMIN;
            }
            d = 1.0 / d;
            h *= d * c;
            aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d;
            if d.abs() < FPMIN {
                d = FPMIN;
            }
            c = 1.0 + aa / c;
            if c.abs() < FPMIN {
                c = FPMIN;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() <= EPS {
                break;
            }
        }
        Ok(ln_beta_prefactor(a, b) + a * ln_x + b * ln_y + h.ln() - a.ln())
    }
    /// ln I_x(a, b) with y = 1 - x; both logs are supplied so that neither side loses digits.
    fn ln_beta_reg(a: f64, b: f64, x: f64, ln_x: f64, y: f64, ln_y: f64) -> R {
        if ln_x == f64::NEG_INFINITY {
            return Ok(f64::NEG_INFINITY);
        }
        if ln_y == f64::NEG_INFINITY {
            return Ok(0.0);
        }
        if x < (a + 1.0) / (a + b + 2.0) {
            ln_beta_cf(a, b, x, ln_x, ln_y)
        } else {
            Ok((-ln_beta_cf(b, a, y, ln_y, ln_x)?.exp()).ln_1p())
        }
    }
    /// (h, ln h, w, ln w, ln(1 + r)) for h = 1/(1 + r), w = r/(1 + r), without cancellation.
    fn odds_split(r: f64, ln_r: f64) -> (f64, f64, f64, f64, f64) {
        if r <= 1.0 {
            let l1p = r.ln_1p();
            (1.0 / (1.0 + r), -l1p, r / (1.0 + r), ln_r - l1p, l1p)
        } else {
            let s = if r == f64::INFINITY {
                (-ln_r).exp()
            } else {
                1.0 / r
            };
            let l1ps = s.ln_1p();
            (
                s / (1.0 + s),
                -ln_r - l1ps,
                1.0 / (1.0 + s),
                -l1ps,
                ln_r + l1ps,
            )
        }
    }
    /// r = ax^2 / df and ln r without overflow (ax may be +inf with a finite ln_ax).
    fn ratio_and_log(ax: f64, ln_ax: f64, df: f64) -> (f64, f64) {
        let r = ax * (ax / df);
        if r.is_finite() && r > 0.0 {
            (r, r.ln())
        } else {
            let ln_r = 2.0 * ln_ax - df.ln();
            (
                if ln_r < 709.0 {
                    ln_r.exp()
                } else {
                    f64::INFINITY
                },
                ln_r,
            )
        }
    }
    /// ln P(T_df <= -ax) for ax > 0.
    pub(super) fn t_ln_lower_tail(ax: f64, ln_ax: f64, df: f64) -> R {
        let (r, ln_r) = ratio_and_log(ax, ln_ax, df);
        let (h, ln_h, w, ln_w, _) = odds_split(r, ln_r);
        Ok(LN_HALF + ln_beta_reg(0.5 * df, 0.5, h, ln_h, w, ln_w)?)
    }
    fn t_ln_pdf(ax: f64, ln_ax: f64, df: f64) -> f64 {
        let (r, ln_r) = ratio_and_log(ax, ln_ax, df);
        let l1p = odds_split(r, ln_r).4;
        let a = 0.5 * df;
        ln_gamma_diff(a, 0.5) - 0.5 * (df.ln() + LN_PI) - (a + 0.5) * l1p
    }
    /// Cornish-Fisher expansion of the t quantile in powers of 1/df (four terms).
    pub(super) fn cornish_fisher_t(z: f64, df: f64) -> f64 {
        let z2 = z * z;
        let g1 = (z2 + 1.0) * z / 4.0;
        let g2 = ((5.0 * z2 + 16.0) * z2 + 3.0) * z / 96.0;
        let g3 = (((3.0 * z2 + 19.0) * z2 + 17.0) * z2 - 15.0) * z / 384.0;
        let g4 = ((((79.0 * z2 + 776.0) * z2 + 1482.0) * z2 - 1920.0) * z2 - 945.0) * z / 92160.0;
        z + (g1 + (g2 + (g3 + g4 / df) / df) / df) / df
    }
    fn cornish_fisher_t_deriv(z: f64, df: f64) -> f64 {
        let z2 = z * z;
        let d1 = (3.0 * z2 + 1.0) / 4.0;
        let d2 = ((25.0 * z2 + 48.0) * z2 + 3.0) / 96.0;
        let d3 = (((21.0 * z2 + 95.0) * z2 + 51.0) * z2 - 15.0) / 384.0;
        let d4 = ((((711.0 * z2 + 5432.0) * z2 + 7410.0) * z2 - 5760.0) * z2 - 945.0) / 92160.0;
        1.0 + (d1 + (d2 + (d3 + d4 / df) / df) / df) / df
    }
    /// z with cornish_fisher_t(z, df) = at (at >= 0), by Newton from at (1 - 1/(4 df)) / sqrt(1 + at^2/(2 df)).
    pub(super) fn z_of_t(at: f64, df: f64) -> f64 {
        let val = 1.0 / (4.0 * df);
        let mut z = at * (1.0 - val) / (1.0 + at * at * 2.0 * val).sqrt();
        for _ in 0..20 {
            let step = (cornish_fisher_t(z, df) - at) / cornish_fisher_t_deriv(z, df);
            z -= step;
            if step.abs() <= 1e-16 * z.max(1.0) {
                break;
            }
        }
        z
    }
    /// Safeguarded Newton for ln_tail(u) = lq; ln_tail is monotone (increasing for a lower tail),
    /// d ln_tail / du = +-exp(u + ln_dens(u) - ln_tail(u)). Steps are clamped to 5 and kept inside the
    /// bracket the iterates establish (bisection otherwise). The result may exceed 709 (overflow).
    /// `first` is ln_tail(u) at the start when the caller has it already (saves one evaluation).
    fn solve_log_tail(
        mut u: f64,
        mut first: Option<f64>,
        lq: f64,
        ln_tail: impl Fn(f64) -> R,
        ln_dens: impl Fn(f64) -> f64,
        increasing: bool,
    ) -> R {
        let (mut lo, mut hi) = (f64::NEG_INFINITY, f64::INFINITY);
        let mut last = f64::INFINITY;
        for _ in 0..NEWTON_MAX {
            let lt = match first.take() {
                Some(lt) => lt,
                None => ln_tail(u)?,
            };
            let g = lt - lq;
            if g == 0.0 {
                return Ok(u);
            }
            let root_below = (g > 0.0) == increasing;
            if root_below {
                hi = u;
            } else {
                lo = u;
            }
            let mut un = f64::NAN;
            if lt.is_finite() {
                let mut slope = (u + ln_dens(u) - lt).exp();
                if !increasing {
                    slope = -slope;
                }
                if slope != 0.0 && slope.is_finite() {
                    un = u - g / slope;
                }
            }
            if un.is_finite() && (un - u).abs() <= STEP_TOL * u.abs().max(1.0) {
                return Ok(un);
            }
            if !un.is_finite() || (un - u).abs() > STEP_CLAMP {
                un = if root_below {
                    u - STEP_CLAMP
                } else {
                    u + STEP_CLAMP
                };
            }
            if !(lo < un && un < hi) {
                un = 0.5 * (lo + hi);
            }
            last = (un - u).abs();
            u = un;
            if u < -746.0 && root_below {
                return Ok(f64::NEG_INFINITY);
            }
        }
        if last <= STEP_ACCEPT * u.abs().max(1.0) {
            Ok(u)
        } else {
            Err(NoConvergence {
                what: "quantile iteration",
                iterations: NEWTON_MAX,
                last_change: last,
            })
        }
    }
    /// x < 0 with P(T_df <= x) = q, 0 < q < 1/2. Start: the best of the tail asymptote, the normal
    /// quantile and the Cornish-Fisher value (statrs StudentsT::inverse_cdf is not used: it hangs and
    /// panics, see the module docs).
    pub(super) fn t_lower_quantile(q: f64, df: f64) -> R {
        let lq = q.ln();
        let a = 0.5 * df;
        let z = -normal_quantile(q).abs();
        let tail = |u: f64| t_ln_lower_tail(exp_or_inf(u), u, df);
        let dens = |u: f64| t_ln_pdf(exp_or_inf(u), u, df);
        let cf = cornish_fisher_t(z, df);
        // Away from the far tail and from df < 3 the Cornish-Fisher value is close to the
        // root, so Newton starts from it without comparing the other starts (each comparison
        // costs a full incomplete-beta tail). The stopping rule is the same, so is the root.
        if cf.is_finite() && cf < 0.0 && q >= 1e-8 && df >= 3.0 {
            let u = solve_log_tail((-cf).ln(), None, lq, tail, dens, false)?;
            return Ok(-exp_or_inf(u));
        }
        let mut starts = [
            (LN_HALF + ln_gamma(a + 0.5) - ln_gamma(a + 1.0) - LN_SQRT_PI + a * df.ln() - lq) / df,
            (-z).ln(),
            f64::NAN,
        ];
        let n_starts = if cf.is_finite() && cf < 0.0 {
            starts[2] = (-cf).ln();
            3
        } else {
            2
        };
        // The tail at the chosen start is the Newton iteration's first evaluation.
        let mut best = starts[0];
        let mut best_lt = tail(best)?;
        let mut best_g = (best_lt - lq).abs();
        for &u0 in &starts[1..n_starts] {
            let lt0 = tail(u0)?;
            let g0 = (lt0 - lq).abs();
            if g0 < best_g {
                best = u0;
                best_lt = lt0;
                best_g = g0;
            }
        }
        let u = solve_log_tail(best, Some(best_lt), lq, tail, dens, false)?;
        Ok(-exp_or_inf(u))
    }
    /// ln P(a, y) by the series y^a e^-y / Gamma(a + 1) * sum_k y^k / prod_{j<=k} (a + j); y < a + 1.
    fn ln_gamma_lower_series(a: f64, y: f64) -> R {
        let (mut term, mut total) = (1.0_f64, 1.0_f64);
        let mut k = 0usize;
        loop {
            k += 1;
            if k > SERIES_MAX {
                return Err(NoConvergence {
                    what: "incomplete gamma series",
                    iterations: SERIES_MAX,
                    last_change: term / total,
                });
            }
            term *= y / (a + k as f64);
            total += term;
            if term <= total * 1e-17 {
                break;
            }
        }
        Ok(a * y.ln() - y - ln_gamma(a + 1.0) + total.ln())
    }
    /// ln Q(a, y) by the statrs (Cephes) continued fraction, prefactor in logs; y >= a + 1.
    fn ln_gamma_upper_cf(a: f64, y: f64) -> R {
        const BIG: f64 = 4_503_599_627_370_496.0;
        const BIG_INV: f64 = 2.220_446_049_250_313e-16;
        let mut yy = 1.0 - a;
        let mut z = y + yy + 1.0;
        let mut c = 0.0;
        let (mut pkm2, mut qkm2) = (1.0, y);
        let (mut pkm1, mut qkm1) = (y + 1.0, z * y);
        let mut ans = pkm1 / qkm1;
        let mut k = 0usize;
        loop {
            k += 1;
            if k > CF_MAX {
                return Err(NoConvergence {
                    what: "incomplete gamma continued fraction",
                    iterations: CF_MAX,
                    last_change: f64::NAN,
                });
            }
            yy += 1.0;
            z += 2.0;
            c += 1.0;
            let yc = yy * c;
            let pk = pkm1 * z - pkm2 * yc;
            let qk = qkm1 * z - qkm2 * yc;
            pkm2 = pkm1;
            pkm1 = pk;
            qkm2 = qkm1;
            qkm1 = qk;
            if pk.abs() > BIG {
                pkm2 *= BIG_INV;
                pkm1 *= BIG_INV;
                qkm2 *= BIG_INV;
                qkm1 *= BIG_INV;
            }
            if qk != 0.0 {
                let r = pk / qk;
                let t = ((ans - r) / r).abs();
                ans = r;
                if t <= 1e-15 {
                    break;
                }
            }
        }
        Ok(a * y.ln() - y - ln_gamma(a) + ans.ln())
    }
    /// ln P(chi2_df <= x) (lower) or ln P(chi2_df > x).
    pub(super) fn chi2_ln_tail(x: f64, df: f64, lower: bool) -> R {
        let a = 0.5 * df;
        let y = 0.5 * x;
        if y <= 0.0 {
            return Ok(if lower { f64::NEG_INFINITY } else { 0.0 });
        }
        if y < a + 1.0 {
            let lp = ln_gamma_lower_series(a, y)?;
            Ok(if lower { lp } else { (-lp.exp()).ln_1p() })
        } else {
            let lqv = ln_gamma_upper_cf(a, y)?;
            Ok(if lower { (-lqv.exp()).ln_1p() } else { lqv })
        }
    }
    /// 2 P(Z > z) for z >= 0 as the chi-square(1) upper tail of z^2 (accurate to the
    /// chi-square tail's ~1e-13 relative; statrs `erfc` is not). NaN if the series or continued
    /// fraction hit its cap (beyond the supported df range, see the module docs).
    pub(super) fn normal_two_sided(z: f64) -> f64 {
        if z == 0.0 {
            return 1.0;
        }
        let x = z * z;
        if !x.is_finite() {
            return 0.0;
        }
        chi2_ln_tail(x, 1.0, false).map_or(f64::NAN, |l| l.exp().min(1.0))
    }
    fn chi2_ln_pdf(u: f64, df: f64) -> f64 {
        let a = 0.5 * df;
        (a - 1.0) * u - 0.5 * exp_or_inf(u) - a * LN_2 - ln_gamma(a)
    }
    /// Chi-square quantile: Wilson-Hilferty start (the small-x series start in the lower tail of small
    /// df), Newton on ln(tail) in u = ln x.
    pub(super) fn chi_squared_quantile(p: f64, df: f64) -> R {
        let a = 0.5 * df;
        let lower = p <= 0.5;
        let q = if lower { p } else { 1.0 - p };
        let z = normal_quantile(p);
        let c = 2.0 / (9.0 * df);
        let w = 1.0 - c + z * c.sqrt();
        let u_small = LN_2 + (p.ln() + ln_gamma(a + 1.0)) / a;
        let u0 = if w <= 0.0 || (lower && u_small < LN_2) {
            u_small
        } else {
            df.ln() + 3.0 * w.ln()
        };
        if lower && u0 < -700.0 {
            return Ok(u0.exp()); // P(a, y) = y^a / Gamma(a + 1) (1 + O(y)) with y < 1e-304
        }
        let u = solve_log_tail(
            u0,
            None,
            q.ln(),
            |uu| chi2_ln_tail(exp_or_inf(uu), df, lower),
            |uu| chi2_ln_pdf(uu, df),
            lower,
        )?;
        Ok(if u == f64::NEG_INFINITY {
            0.0
        } else {
            exp_or_inf(u)
        })
    }
    /// ln P(F_{df1, df2} > f) = ln I_x(df2/2, df1/2), x = df2/(df2 + df1 f).
    pub(super) fn f_ln_upper_tail(f: f64, df1: f64, df2: f64) -> R {
        let mut r = df1 * (f / df2);
        let ln_r = if r.is_finite() && r > 0.0 {
            r.ln()
        } else {
            let l = df1.ln() + f.ln() - df2.ln();
            r = if l < 709.0 { l.exp() } else { f64::INFINITY };
            l
        };
        let (x, ln_x, y, ln_y, _) = odds_split(r, ln_r);
        ln_beta_reg(0.5 * df2, 0.5 * df1, x, ln_x, y, ln_y)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Handing solve_log_tail the tail value at its start only skips that evaluation: the
        /// quantile is the same, bit for bit, as when the iteration evaluates it itself.
        #[test]
        fn solve_log_tail_given_first_value_matches_evaluating_it() {
            for &q in &[1e-300_f64, 1e-20, 1e-4, 0.005, 0.025, 0.1, 0.3, 0.4999] {
                for &df in &[0.5, 1.0, 3.0, 10.0, 94.0, 999.0, 1e5] {
                    let lq = q.ln();
                    let tail = |u: f64| t_ln_lower_tail(exp_or_inf(u), u, df);
                    let dens = |u: f64| t_ln_pdf(exp_or_inf(u), u, df);
                    let u0 = normal_quantile(q).abs().ln();
                    let evaluated = solve_log_tail(u0, None, lq, tail, dens, false);
                    let given = solve_log_tail(u0, Some(tail(u0).unwrap()), lq, tail, dens, false);
                    assert_eq!(
                        evaluated.map(f64::to_bits).ok(),
                        given.map(f64::to_bits).ok(),
                        "q = {q}, df = {df}"
                    );
                }
            }
        }
    }
}

/// Standard normal CDF (for Probit).
pub fn normal_cdf(x: f64) -> f64 {
    Normal::standard().cdf(x)
}

/// Standard normal density (for Probit).
pub fn normal_pdf(x: f64) -> f64 {
    Normal::standard().pdf(x)
}

/// Logistic CDF (for Logit)
pub fn logistic_cdf(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Logistic PDF (for Logit)
pub fn logistic_pdf(x: f64) -> f64 {
    let p = logistic_cdf(x);
    p * (1.0 - p)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `t_quantile` inverts the t tail: on a grid of probabilities and degrees of freedom,
    /// the lower tail at -|q| equals min(p, 1 - p) to a relative 1e-12 (the lower tail of
    /// a positive quantile's mirror image, so p > 1/2 is checked through 1 - p).
    #[test]
    fn test_t_quantile_inverts_the_tail_on_a_grid() {
        let ps = [
            1e-12, 1e-8, 1e-4, 0.01, 0.025, 0.1, 0.4, 0.6, 0.9, 0.975, 0.995, 0.9999,
        ];
        let dfs = [0.5, 1.0, 2.5, 3.0, 10.0, 30.0, 99.0, 500.0, 999.0, 5000.0];
        let mut worst = (0.0_f64, 0.0, 0.0);
        for &p in &ps {
            for &df in &dfs {
                let q = t_quantile(p, df).unwrap();
                let ax = q.abs();
                let lt = tails::t_ln_lower_tail(ax, ax.ln(), df).unwrap().exp();
                let rel = (lt / p.min(1.0 - p) - 1.0).abs();
                if rel > worst.0 {
                    worst = (rel, p, df);
                }
            }
        }
        assert!(worst.0 <= 1e-12, "worst {worst:?}");
    }

    #[test]
    fn test_significance_level() {
        assert_eq!(
            SignificanceLevel::from_p_value(0.0001),
            SignificanceLevel::TenthPercent
        );
        assert_eq!(
            SignificanceLevel::from_p_value(0.005),
            SignificanceLevel::OnePercent
        );
        assert_eq!(
            SignificanceLevel::from_p_value(0.03),
            SignificanceLevel::FivePercent
        );
        assert_eq!(
            SignificanceLevel::from_p_value(0.08),
            SignificanceLevel::TenPercent
        );
        assert_eq!(
            SignificanceLevel::from_p_value(0.15),
            SignificanceLevel::NotSignificant
        );
    }

    #[test]
    fn test_stars() {
        assert_eq!(SignificanceLevel::TenthPercent.stars(), "***");
        assert_eq!(SignificanceLevel::OnePercent.stars(), "**");
        assert_eq!(SignificanceLevel::FivePercent.stars(), "*");
        assert_eq!(SignificanceLevel::TenPercent.stars(), "†");
        assert_eq!(SignificanceLevel::NotSignificant.stars(), "");
    }

    #[test]
    fn test_t_test_p_value() {
        // t = 0 should give p = 1
        let p = t_test_p_value(0.0, 10.0);
        assert!((p - 1.0).abs() < 1e-10);

        // Large t should give small p
        let p = t_test_p_value(10.0, 100.0);
        assert!(p < 0.001);
    }

    #[test]
    fn test_p_value_helpers_survive_nonfinite_df() {
        // Regression guard: StudentsT::new used to panic here on NaN/Inf df.
        assert!(t_test_p_value(1.5, f64::NAN).is_nan());
        assert!(t_test_p_value(1.5, f64::INFINITY).is_nan());
        assert!(t_test_p_value(1.5, f64::NEG_INFINITY).is_nan());
        assert!(t_test_p_value(1.5, -1.0).is_nan());

        assert!(chi_squared_p_value(2.0, f64::NAN).is_nan());
        assert!(chi_squared_p_value(2.0, f64::INFINITY).is_nan());

        assert!(f_test_p_value(2.0, f64::NAN, 10.0).is_nan());
        assert!(f_test_p_value(2.0, 10.0, f64::INFINITY).is_nan());

        assert!(t_critical(0.05, f64::NAN).is_nan());
        assert!(t_critical(0.05, f64::INFINITY).is_nan());
        // invalid alpha
        assert!(t_critical(0.0, 10.0).is_nan());
        assert!(t_critical(1.0, 10.0).is_nan());
        assert!(t_critical(f64::NAN, 10.0).is_nan());
    }

    #[test]
    fn test_logistic_cdf() {
        assert!((logistic_cdf(0.0) - 0.5).abs() < 1e-10);
        assert!(logistic_cdf(10.0) > 0.99);
        assert!(logistic_cdf(-10.0) < 0.01);
    }

    #[test]
    fn test_normal_cdf() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-10);
        assert!(normal_cdf(3.0) > 0.99);
        assert!(normal_cdf(-3.0) < 0.01);
    }

    use crate::regression::{CovarianceType, OlsResult, run_ols};
    use ndarray::array;
    use p2a_validation::{Check, Tol};

    /// Delegates only the required methods (and `r_squared`) to an `OlsResult`, so every
    /// other method runs its trait default.
    struct DefaultsOnly<'a>(&'a OlsResult);

    impl LinearEstimator for DefaultsOnly<'_> {
        fn coefficients(&self) -> &Array1<f64> {
            LinearEstimator::coefficients(self.0)
        }
        fn std_errors(&self) -> &Array1<f64> {
            LinearEstimator::std_errors(self.0)
        }
        fn residuals(&self) -> &Array1<f64> {
            LinearEstimator::residuals(self.0)
        }
        fn vcov_matrix(&self) -> &Array2<f64> {
            LinearEstimator::vcov_matrix(self.0)
        }
        fn variable_names(&self) -> &[String] {
            &self.0.variable_names
        }
        fn degrees_of_freedom(&self) -> usize {
            self.0.df_resid
        }
        fn n_obs(&self) -> usize {
            self.0.n_obs
        }
        fn r_squared(&self) -> f64 {
            self.0.r_squared
        }
    }

    /// Fixed arrays for the NaN and degrees-of-freedom cases.
    struct Fixed {
        b: Array1<f64>,
        se: Array1<f64>,
        e: Array1<f64>,
        v: Array2<f64>,
        names: Vec<String>,
        df: usize,
        n: usize,
        r2: f64,
    }

    impl Fixed {
        /// Four coefficients, n = 10, df = 6, R² = 0.5; residuals with noise.
        fn new(b: Array1<f64>, se: Array1<f64>) -> Fixed {
            Fixed {
                b,
                se,
                e: array![0.3, -0.1, 0.25, -0.4, 0.05, 0.12, -0.2, 0.33, -0.15, -0.2],
                v: Array2::zeros((4, 4)),
                names: ["(Intercept)", "x1", "x2", "x3"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                df: 6,
                n: 10,
                r2: 0.5,
            }
        }
    }

    impl LinearEstimator for Fixed {
        fn coefficients(&self) -> &Array1<f64> {
            &self.b
        }
        fn std_errors(&self) -> &Array1<f64> {
            &self.se
        }
        fn residuals(&self) -> &Array1<f64> {
            &self.e
        }
        fn vcov_matrix(&self) -> &Array2<f64> {
            &self.v
        }
        fn variable_names(&self) -> &[String] {
            &self.names
        }
        fn degrees_of_freedom(&self) -> usize {
            self.df
        }
        fn n_obs(&self) -> usize {
            self.n
        }
        fn r_squared(&self) -> f64 {
            self.r2
        }
    }

    /// The trait defaults on `lm(inv ~ value + capital)` (case `ols_grunfeld`) give what
    /// summary.lm, confint, logLik, AIC and BIC give (REG-3).
    #[test]
    fn test_validate_linear_estimator_defaults_grunfeld() {
        let (case, ds) = crate::test_support::load_case("ols_grunfeld");
        let r = run_ols(
            &ds,
            "inv",
            &["value", "capital"],
            true,
            CovarianceType::Standard,
        )
        .unwrap();
        let d = DefaultsOnly(&r);
        let n = d.n_obs();
        let names = d.variable_names();
        let (lo, hi): (Vec<f64>, Vec<f64>) = d.confidence_intervals_95().into_iter().unzip();
        Check::new(&case)
            .named("t_classical", names, &d.t_stats().to_vec(), Tol::stat(n))
            .named(
                "p_classical",
                names,
                &d.p_values().to_vec(),
                Tol::p_value(n),
            )
            .named("ci95_lower_classical", names, &lo, Tol::stat(n))
            .named("ci95_upper_classical", names, &hi, Tol::stat(n))
            .scalar("adj_r2", d.adj_r_squared(), Tol::stat(n))
            .scalar("sigma", d.residual_std_error(), Tol::stat(n))
            .scalar("log_lik", d.log_likelihood(), Tol::stat(n))
            .scalar("aic", d.aic(), Tol::stat(n))
            .scalar("bic", d.bic(), Tol::stat(n))
            .finish();
    }

    /// A NaN or zero standard error gives the `t_test_p_value` convention, never a panic
    /// (the core half of CLI-6).
    #[test]
    fn test_default_p_values_are_nan_safe() {
        let f = Fixed::new(array![1.0, 2.0, 3.0, 0.0], array![0.5, f64::NAN, 0.0, 0.0]);
        let p = f.p_values();
        assert_eq!(p.len(), 4);
        // t = 2
        assert!((p[0] - t_test_p_value(2.0, 6.0)).abs() <= 1e-15, "{p:?}");
        // t = 2 / NaN
        assert!(p[1].is_nan(), "{p:?}");
        // t = 3 / 0 = +inf
        assert_eq!(p[2], 0.0, "{p:?}");
        // t = 0 / 0
        assert!(p[3].is_nan(), "{p:?}");
    }

    /// Adjusted R² divides by the residual df, which already counts the intercept.
    #[test]
    fn test_default_adj_r_squared_uses_residual_df() {
        let mut f = Fixed::new(array![1.0, 2.0, 3.0, 0.5], array![0.5, 0.4, 0.3, 0.2]);
        // 1 - (1 - 0.5) * 9 / 6
        assert!(
            (f.adj_r_squared() - 0.25).abs() <= 1e-15,
            "{}",
            f.adj_r_squared()
        );
        f.df = 0;
        assert!(f.adj_r_squared().is_nan(), "{}", f.adj_r_squared());
    }

    // ---------------------------------------------------------------------
    // Distribution helpers against R (case stat_quantile_types)
    // ---------------------------------------------------------------------

    /// Relative 1e-11 and no absolute floor: X-17 is about p-values below 1e-12, which
    /// `Tol::p_value`'s abs = 1e-12 would accept as 0. Measured maxima of the algorithm
    /// against R: 6.1e-13 (F), 3.6e-13 (qt), 8.0e-13 (qchisq), 1.5e-13 (pt).
    const TAIL: Tol = Tol::new(1e-11, 0.0);

    /// The id is an argument so that every registered test names "stat_quantile_types" in
    /// its own body (validation_status.py rule registered-test-loads-no-case).
    fn case_of(id: &str) -> p2a_validation::Case {
        crate::test_support::load_case(id).0
    }

    /// The HC1 t of `value` in `ols_grunfeld` and t = 13.5, both at df = 197, give p-values
    /// near 1e-40 and 1e-30 that `1 - cdf` rounded to 0. (The Tier 3 design's text paired
    /// t = 13.5 with 2.165754515305109e-40, which is the p-value of the first t.)
    #[test]
    fn test_validate_t_p_value_tail_is_not_zero() {
        let case = case_of("stat_quantile_types");
        let (t, df, p) = (
            case.vector("pt2_t"),
            case.vector("pt2_df"),
            case.vector("pt2_p"),
        );
        for i in 0..2 {
            let got = t_test_p_value(t[i], df[i]);
            assert!(got > 0.0, "t = {}, df = {}: p = {got}", t[i], df[i]);
            assert!(
                TAIL.allows(got, p[i]),
                "t = {}, df = {}: rust {got:e}, R {:e}",
                t[i],
                df[i],
                p[i]
            );
        }
    }

    #[test]
    fn test_validate_p_value_helpers_stat_quantile_types() {
        let c = case_of("stat_quantile_types");
        let (tt, tdf) = (c.vector("pt2_t"), c.vector("pt2_df"));
        let t_vals: Vec<f64> = tt
            .iter()
            .zip(&tdf)
            .map(|(&t, &df)| t_test_p_value(t, df))
            .collect();
        let (cx, cdf) = (c.vector("pchisq_x"), c.vector("pchisq_df"));
        let c_vals: Vec<f64> = cx
            .iter()
            .zip(&cdf)
            .map(|(&x, &df)| chi_squared_p_value(x, df))
            .collect();
        let (fx, f1, f2) = (c.vector("pf_x"), c.vector("pf_d1"), c.vector("pf_d2"));
        let f_vals: Vec<f64> = (0..fx.len())
            .map(|i| f_test_p_value(fx[i], f1[i], f2[i]))
            .collect();
        Check::new(&c)
            .vector("pt2_p", &t_vals, TAIL)
            .vector("pchisq_p", &c_vals, TAIL)
            .vector("pf_p", &f_vals, TAIL)
            .finish();
    }

    #[test]
    fn test_validate_t_critical_large_df_matches_r() {
        let case = case_of("stat_quantile_types");
        let q = case.vector("qt_q");
        let at197 = t_critical(0.05, 197.0);
        assert!(TAIL.allows(at197, q[1]), "df 197: rust {at197}, R {}", q[1]);
        let at1e6 = t_critical(0.05, 1e6);
        assert!(TAIL.allows(at1e6, q[4]), "df 1e6: rust {at1e6}, R {}", q[4]);
    }

    /// statrs `StudentsT::inverse_cdf` never returned at these very large df.
    #[test]
    fn test_validate_t_critical_terminates_for_huge_df() {
        let case = case_of("stat_quantile_types");
        let q = case.vector("qt_q");
        let a = t_critical(0.05, 2e7);
        assert!(TAIL.allows(a, q[5]), "df 2e7: rust {a}, R {}", q[5]);
        let b = t_critical(0.05, 1e9);
        assert!(TAIL.allows(b, q[6]), "df 1e9: rust {b}, R {}", q[6]);
    }

    /// One coefficient with df = 1e6, so the default CI uses qt(0.975, 1e6).
    struct LargeDf {
        b: Array1<f64>,
        se: Array1<f64>,
        e: Array1<f64>,
        v: Array2<f64>,
        names: Vec<String>,
    }

    impl LinearEstimator for LargeDf {
        fn coefficients(&self) -> &Array1<f64> {
            &self.b
        }
        fn std_errors(&self) -> &Array1<f64> {
            &self.se
        }
        fn residuals(&self) -> &Array1<f64> {
            &self.e
        }
        fn vcov_matrix(&self) -> &Array2<f64> {
            &self.v
        }
        fn variable_names(&self) -> &[String] {
            &self.names
        }
        fn degrees_of_freedom(&self) -> usize {
            1_000_000
        }
        fn n_obs(&self) -> usize {
            1_000_001
        }
    }

    #[test]
    fn test_validate_default_confidence_intervals_large_df() {
        let case = case_of("stat_quantile_types");
        let q = case.vector("qt_q");
        let fit = LargeDf {
            b: array![1.0],
            se: array![0.5],
            e: array![0.0],
            v: Array2::from_elem((1, 1), 0.25),
            names: vec!["b".to_string()],
        };
        let ci = fit.confidence_intervals(0.95)[0];
        assert!(TAIL.allows(ci.0, 1.0 - 0.5 * q[4]), "{ci:?}");
        assert!(TAIL.allows(ci.1, 1.0 + 0.5 * q[4]), "{ci:?}");
    }

    #[test]
    fn test_validate_normal_helpers_stat_quantile_types() {
        let c = case_of("stat_quantile_types");
        let z: Vec<f64> = c
            .vector("pnorm2_z")
            .iter()
            .map(|&z| normal_p_value_two_sided(z))
            .collect();
        let qn: Vec<f64> = c
            .vector("qnorm_p")
            .iter()
            .map(|&p| normal_quantile(p).unwrap())
            .collect();
        let zc: Vec<f64> = c
            .vector("zcrit_level")
            .iter()
            .map(|&l| z_critical(l).unwrap())
            .collect();
        Check::new(&c)
            .vector("pnorm2_p", &z, TAIL)
            .vector("qnorm_q", &qn, TAIL)
            .vector("zcrit_z", &zc, TAIL)
            .finish();
    }

    #[test]
    fn test_validate_quantile_helpers_stat_quantile_types() {
        let c = case_of("stat_quantile_types");
        let (tp, tdf) = (c.vector("qt_p"), c.vector("qt_df"));
        let qt: Vec<f64> = tp
            .iter()
            .zip(&tdf)
            .map(|(&p, &df)| t_quantile(p, df).unwrap())
            .collect();
        let (cp, cdf) = (c.vector("qchisq_p"), c.vector("qchisq_df"));
        let qc: Vec<f64> = cp
            .iter()
            .zip(&cdf)
            .map(|(&p, &df)| chi_squared_quantile(p, df).unwrap())
            .collect();
        Check::new(&c)
            .vector("qt_q", &qt, TAIL)
            .vector("qchisq_q", &qc, TAIL)
            .finish();
    }

    fn spec_message<T: std::fmt::Debug>(r: EconResult<T>) -> String {
        match r {
            Err(EconError::InvalidSpecification { message }) => message,
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    #[test]
    fn test_quantile_helpers_reject_bad_arguments() {
        for p in [0.0, 1.0, -0.1, 1.5, f64::NAN, f64::INFINITY] {
            let m = spec_message(normal_quantile(p));
            assert!(m.contains("probability"), "{p}: {m}");
        }
        for l in [0.0, 1.0, 95.0, f64::NAN] {
            let m = spec_message(z_critical(l));
            assert!(m.contains("level"), "{l}: {m}");
        }
        for df in [0.0, -1.0, f64::NAN] {
            let m = spec_message(t_quantile(0.975, df));
            assert!(m.contains("degrees of freedom"), "{df}: {m}");
        }
        assert!(spec_message(t_quantile(1.5, 10.0)).contains("probability"));
        for df in [f64::INFINITY, 0.0, f64::NAN] {
            let m = spec_message(chi_squared_quantile(0.5, df));
            assert!(m.contains("degrees of freedom"), "{df}: {m}");
        }
        assert!(validate_level(0.95, "conf_level").is_ok());
        assert_eq!(
            spec_message(validate_level(1.0, "conf_level")),
            "conf_level must be strictly between 0 and 1, got 1"
        );
        assert_eq!(t_quantile(0.5, 3.0).unwrap(), 0.0);
        assert_eq!(
            t_quantile(0.975, f64::INFINITY).unwrap(),
            normal_quantile(0.975).unwrap()
        );
    }

    #[test]
    fn test_p_value_helpers_edge_values() {
        assert_eq!(t_test_p_value(0.0, 5.0), 1.0);
        assert_eq!(t_test_p_value(f64::INFINITY, 5.0), 0.0);
        assert_eq!(t_test_p_value(1e200, 5.0), 0.0);
        assert_eq!(chi_squared_p_value(0.0, 3.0), 1.0);
        assert_eq!(chi_squared_p_value(f64::INFINITY, 3.0), 0.0);
        assert_eq!(f_test_p_value(0.0, 2.0, 3.0), 1.0);
        assert_eq!(f_test_p_value(1e300, 10.0, 10.0), 0.0);
        assert!(normal_p_value_two_sided(f64::NAN).is_nan());
        assert_eq!(normal_p_value_two_sided(0.0), 1.0);
        for t in [-1e3, -5.0, -1.0, 0.3, 4.0, 1e6] {
            for df in [0.5, 3.0, 1e3, 1e8] {
                let p = t_test_p_value(t, df);
                assert!((0.0..=1.0).contains(&p), "t = {t}, df = {df}: {p}");
            }
        }
    }

    /// SplitMix64, inline so the test needs no RNG crate.
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        /// Uniform on [0, 1).
        fn unif(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.unif()
        }
    }

    /// The chi-square helpers' documented df range: converged up to 1e12, NaN or
    /// ConvergenceFailure near the centre beyond the series cap.
    #[test]
    fn test_chi_squared_helpers_df_range() {
        // Inside the documented range: finite, and just below 1/2 at the mean.
        let p = chi_squared_p_value(1e12, 1e12);
        assert!(p > 0.49 && p < 0.5, "{p}");
        // Beyond the series cap (df >= 5e12, near the centre): NaN and ConvergenceFailure.
        assert!(chi_squared_p_value(1e14, 1e14).is_nan());
        assert!(matches!(
            chi_squared_quantile(0.5, 1e14),
            Err(EconError::ConvergenceFailure { .. })
        ));
        // Far from the centre the continued fraction still converges at df = 1e14.
        let far = chi_squared_p_value(1.1e14, 1e14);
        assert_eq!(far, 0.0, "{far}");
    }

    /// Seeded draws: lower-tail p in [1e-300, 0.5), upper-tail p in (0.5, 1 - 1e-15], and df
    /// from 0.1 to 1e12 (t) and 0.03 to 1e5 (chi-square): the quantile solvers return, and the
    /// tails of their answers give p back.
    #[test]
    fn test_distribution_helpers_seeded_round_trip() {
        let rel = |a: f64, b: f64| ((a - b) / b).abs();
        let mut rng = SplitMix64(20261021);
        // Lower-tail q down to 1e-300; upper-tail q from 1e-15, so that 1 - q stays below 1.
        let (lo, lo_upper, hi) = (1e-300_f64.ln(), 1e-15_f64.ln(), 0.5_f64.ln());
        let mut done = 0;
        while done < 1000 {
            let upper = rng.unif() >= 0.5;
            let q = rng.range(if upper { lo_upper } else { lo }, hi).exp();
            let p = if upper { 1.0 - q } else { q };
            if p == 0.5 || p >= 1.0 {
                continue;
            }
            done += 1;
            let df = 10f64.powf(rng.range(-1.0, 12.0));
            let x = t_quantile(p, df).unwrap();
            if x.is_finite() && x != 0.0 {
                let back = t_test_p_value(x, df) / 2.0;
                assert!(
                    rel(back, p.min(1.0 - p)) <= 1e-10,
                    "t: p = {p:e}, df = {df:e}, x = {x:e}, tail {back:e}"
                );
            } else {
                assert!(df < 1.0, "t: p = {p:e}, df = {df:e}, x = {x}");
            }
            let dfc = 10f64.powf(rng.range(-1.5, 5.0));
            let c = chi_squared_quantile(p, dfc).unwrap();
            assert!(
                c.is_finite() && c >= 0.0,
                "chi2: p = {p:e}, df = {dfc:e}, x = {c}"
            );
            if c >= 1e-290 {
                let lower = p <= 0.5;
                let back = tails::chi2_ln_tail(c, dfc, lower).unwrap().exp();
                let target = if lower { p } else { 1.0 - p };
                assert!(
                    rel(back, target) <= 1e-8,
                    "chi2: p = {p:e}, df = {dfc:e}, x = {c:e}, tail {back:e}"
                );
            }
        }
    }
}
