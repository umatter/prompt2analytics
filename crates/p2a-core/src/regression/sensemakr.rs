//! Sensitivity analysis for unmeasured confounding (sensemakr).
//!
//! This module implements sensitivity analysis for linear regression to assess
//! robustness to unmeasured confounding. The key outputs include:
//!
//! - **Robustness Value (RV)**: The minimum strength of association between an
//!   unobserved confounder and both treatment and outcome needed to explain away
//!   the observed effect.
//!
//! - **Partial R-squared**: Measures of how strongly a confounder would need to
//!   be related to treatment and outcome.
//!
//! - **Bias bounds**: Adjusted estimates under different confounding scenarios.
//!
//! # Mathematical Background
//!
//! ## Partial R-squared of treatment with outcome
//!
//! The partial R-squared of the treatment D with the outcome Y given covariates X is:
//!
//! R²_{Y~D|X} = t² / (t² + df)
//!
//! where t is the t-statistic and df is the residual degrees of freedom.
//!
//! ## Robustness Value (RV)
//!
//! The robustness value for a reduction of q*100% of the effect (q=1 means nullify) is:
//!
//! RV_q = 0.5 * (sqrt(f⁴ + 4f²) - f²)
//!
//! where f = t_q / sqrt(df) and t_q is the t-statistic that would correspond to
//! reducing the effect by a factor of q.
//!
//! For q=1 (nullifying the effect):
//! - f = |t| / sqrt(df)
//!
//! For statistical significance at level α (`robustness_value(t, dof, q, alpha)`):
//! - f_c = |qt(α/2, df - 1)| / sqrt(df - 1) and f_qa = q|t| / sqrt(df) - f_c
//! - RV_{q,α} = 0.5 * (sqrt(f_qa⁴ + 4f_qa²) - f_qa²) for f_qa > 0 (0 otherwise), or the
//!   extreme robustness value (f_q² - f_c²) / (1 + f_q²) when f_q > 1/f_c
//!
//! Every statistic here uses the classical OLS t-statistic (`CovarianceType::Standard`).
//!
//! ## Bias-adjusted estimate
//!
//! Under a confounding scenario with partial R-squared R²_{Y~Z|X,D} and R²_{D~Z|X}:
//!
//! bias = se * sqrt(R²_{Y~Z|X,D} * R²_{D~Z|X}) * sqrt(df / (1 - R²_{D~Z|X}))
//!
//! The adjusted estimate is: β_adj = β - sign(β) * bias, its standard error
//! sqrt((1 - R²_{Y~Z|X,D}) / (1 - R²_{D~Z|X})) * se * sqrt(df / (df - 1))
//! ([`adjusted_se_ovb`]), and its t-statistic (β_adj - h0) / se_adj.
//!
//! ## Benchmark bounds
//!
//! A confounder kd times as strongly related to the treatment and ky times as strongly to
//! the outcome as an observed covariate X_j is bounded by [`ovb_partial_r2_bound`]
//! (Cinelli & Hazlett 2020, Section 6.1); the bounds table reports the adjusted estimate,
//! SE, t against h0 = (1 - q) β and the (1 - α) interval with qt(α/2, df), as
//! `sensemakr::sensemakr()` does.
//!
//! # References
//!
//! - Cinelli, C. & Hazlett, C. (2020). "Making Sense of Sensitivity: Extending
//!   Omitted Variable Bias". Journal of the Royal Statistical Society: Series B,
//!   82(1), 39-67. <https://doi.org/10.1111/rssb.12348>
//!
//! - sensemakr R package: <https://carloscinelli.com/sensemakr/>
//!   Cinelli, C., Ferwerda, J., & Hazlett, C. (2020). sensemakr: Sensitivity
//!   Analysis Tools for Regression Models. R package version 0.1.4.
//!   <https://CRAN.R-project.org/package=sensemakr>
//!
//! R equivalent: `sensemakr::sensemakr()`

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use std::collections::HashMap;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::design::DesignMatrix;
use crate::linalg::solve::{LM_QR_TOL, least_squares};
use crate::regression::ols::{CovarianceType, OlsResult, normal_equations_inverse, run_ols};
use crate::traits::t_critical;

/// Result of sensitivity analysis for a single treatment variable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensemakrResult {
    // ═══════════════════════════════════════════════════════════════════════
    // Treatment information
    // ═══════════════════════════════════════════════════════════════════════
    /// Name of the treatment variable
    pub treatment: String,
    /// Point estimate (coefficient)
    pub estimate: f64,
    /// Standard error
    pub std_error: f64,
    /// t-statistic
    pub t_statistic: f64,
    /// Degrees of freedom
    pub df: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // Core sensitivity statistics
    // ═══════════════════════════════════════════════════════════════════════
    /// Partial R² of treatment with outcome (R²_{Y~D|X})
    pub partial_r2_yd: f64,
    /// Robustness value for nullifying the effect (q=1)
    pub robustness_value: f64,
    /// Robustness value for statistical significance at given alpha
    pub robustness_value_alpha: f64,
    /// Proportion of effect to reduce (q parameter)
    pub rv_q: f64,
    /// Significance level used for RV_alpha
    pub rv_alpha: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // Additional sensitivity measures
    // ═══════════════════════════════════════════════════════════════════════
    /// Sensitivity statistics (partial R², RV) for each covariate as benchmark
    pub benchmark_bounds: Vec<SensitivityBound>,

    // ═══════════════════════════════════════════════════════════════════════
    // Contour data for plotting
    // ═══════════════════════════════════════════════════════════════════════
    /// Contour data for sensitivity plots
    pub contour_data: Option<ContourData>,

    /// Non-fatal warnings. Populated, for example, when benchmark bounds fall
    /// back to an approximation because the design matrix was unavailable
    /// (the OLS-only `sensemakr` entry point); `run_sensemakr` computes the
    /// exact benchmark partial R² and leaves this empty.
    #[serde(default)]
    pub warnings: Vec<String>,
}

/// Sensitivity bounds computed using a benchmark covariate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityBound {
    /// Name of the benchmark covariate
    pub benchmark: String,
    /// Partial R² of benchmark with outcome (R²_{Y~X_j|X_{-j}})
    pub r2_yd_benchmark: f64,
    /// Partial R² of benchmark with treatment (R²_{D~X_j|X_{-j}})
    pub r2_xz_benchmark: f64,
    /// Multiplier applied to get bound (kd)
    pub kd: f64,
    /// Multiplier applied to get bound (ky)
    pub ky: f64,
    /// Bound on R²_{Y~Z|D,X} implied by the benchmark (sensemakr's `r2yz.dx`; set to 1 when
    /// the implied value exceeds 1, with a warning)
    pub r2_yd_bound: f64,
    /// Bound on R²_{D~Z|X} implied by the benchmark (sensemakr's `r2dz.x`)
    pub r2_xz_bound: f64,
    /// Bias from this confounding scenario
    pub bias: f64,
    /// Adjusted estimate under this confounding scenario
    pub adjusted_estimate: f64,
    /// Bias-adjusted standard error (sensemakr's `adjusted_se`)
    pub adjusted_se: f64,
    /// Adjusted t-statistic against h0 = (1 - q) · estimate (sensemakr's `adjusted_t`)
    pub adjusted_t: f64,
    /// Lower bound of the (1 - alpha) CI for the adjusted estimate (qt(alpha/2, df))
    pub adjusted_ci_lower: f64,
    /// Upper bound of the (1 - alpha) CI for the adjusted estimate
    pub adjusted_ci_upper: f64,
}

/// Data for sensitivity contour plots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContourData {
    /// Grid of R²_{Y~U|X,D} values (x-axis)
    pub r2_yd_values: Vec<f64>,
    /// Grid of R²_{D~U|X} values (y-axis)
    pub r2_xz_values: Vec<f64>,
    /// Matrix of adjusted estimates (row = r2_xz, col = r2_yd)
    pub adjusted_estimates: Vec<Vec<f64>>,
    /// Matrix of t-statistics of the adjusted estimates against 0, with sensemakr's adjusted SE
    pub adjusted_t_stats: Vec<Vec<f64>>,
}

impl std::fmt::Display for SensemakrResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Sensitivity Analysis (sensemakr)")?;
        writeln!(f, "=================================")?;
        writeln!(f)?;
        writeln!(f, "Treatment Variable: {}", self.treatment)?;
        writeln!(f)?;
        writeln!(f, "Outcome: Coefficient of {}", self.treatment)?;
        writeln!(f, "  Estimate: {:>12.6}", self.estimate)?;
        writeln!(f, "  Std. Error: {:>10.6}", self.std_error)?;
        writeln!(f, "  t-value: {:>13.4}", self.t_statistic)?;
        writeln!(f, "  DF: {:>17.0}", self.df)?;
        writeln!(f)?;
        writeln!(f, "Sensitivity Statistics:")?;
        writeln!(f, "  Partial R²(Y~D|X): {:>8.4}", self.partial_r2_yd)?;
        writeln!(
            f,
            "  H0: q = {}, reduce = {:.0}%",
            self.rv_q,
            self.rv_q * 100.0
        )?;
        writeln!(f)?;
        writeln!(f, "Robustness Values:")?;
        writeln!(f, "  RV(q={}): {:>12.4}", self.rv_q, self.robustness_value)?;
        writeln!(
            f,
            "  RV(q={}, α={:.2}): {:>6.4}",
            self.rv_q, self.rv_alpha, self.robustness_value_alpha
        )?;
        writeln!(f)?;

        if !self.benchmark_bounds.is_empty() {
            writeln!(
                f,
                "Bounds on confounding required to explain away the estimate:"
            )?;
            writeln!(
                f,
                "{:>20} {:>10} {:>10} {:>12} {:>12} {:>12}",
                "Benchmark", "kd", "ky", "R²(Y~U|X,D)", "R²(D~U|X)", "Adj. Est."
            )?;
            writeln!(f, "{:-<82}", "")?;
            for bound in &self.benchmark_bounds {
                writeln!(
                    f,
                    "{:>20} {:>10.1} {:>10.1} {:>12.4} {:>12.4} {:>12.4}",
                    p2a_text::ellipsize(&bound.benchmark, 20),
                    bound.kd,
                    bound.ky,
                    bound.r2_yd_bound,
                    bound.r2_xz_bound,
                    bound.adjusted_estimate
                )?;
            }
        }

        if let Some(c) = &self.contour_data {
            writeln!(f)?;
            writeln!(
                f,
                "Contour data: {} x {} grid of adjusted estimates and t-statistics (rows \
                 R²(D~Z|X) {}..{}, columns R²(Y~Z|D,X) {}..{}); serialise the result to read it.",
                c.r2_xz_values.len(),
                c.r2_yd_values.len(),
                c.r2_xz_values.first().copied().unwrap_or(f64::NAN),
                c.r2_xz_values.last().copied().unwrap_or(f64::NAN),
                c.r2_yd_values.first().copied().unwrap_or(f64::NAN),
                c.r2_yd_values.last().copied().unwrap_or(f64::NAN),
            )?;
        }

        writeln!(f)?;
        writeln!(f, "Interpretation:")?;
        writeln!(
            f,
            "  An unobserved confounder would need to explain at least {:.1}% of the",
            self.robustness_value * 100.0
        )?;
        writeln!(
            f,
            "  residual variance of both the treatment and the outcome to fully"
        )?;
        writeln!(f, "  account for the estimated effect.")?;

        for warning in &self.warnings {
            writeln!(f)?;
            writeln!(f, "⚠ WARNING: {}", warning)?;
        }

        Ok(())
    }
}

/// Compute partial R² from t-statistic and degrees of freedom.
///
/// R²_{Y~D|X} = t² / (t² + df)
///
/// # Arguments
/// * `t_stat` - The t-statistic for the coefficient
/// * `df` - Residual degrees of freedom
///
/// # References
/// Cinelli & Hazlett (2020), Equation (3)
pub fn partial_r2(t_stat: f64, df: f64) -> f64 {
    if df <= 0.0 || t_stat.is_nan() {
        return f64::NAN;
    }
    let t2 = t_stat.powi(2);
    t2 / (t2 + df)
}

/// Compute robustness value (RV) for a given reduction factor q.
///
/// The RV is the minimum strength of confounding needed to reduce the estimate
/// by a factor of q (q=1 means completely nullify).
///
/// RV_q = 0.5 * (sqrt(f⁴ + 4f²) - f²)
///
/// where f = |t_adj| / sqrt(df) and t_adj accounts for the reduction.
///
/// # Arguments
/// * `t_stat` - The t-statistic
/// * `df` - Residual degrees of freedom
/// * `q` - Proportion of the effect to be explained away (0 < q <= 1)
///
/// # References
/// Cinelli & Hazlett (2020), Proposition 2
pub fn robustness_value(t_stat: f64, df: f64, q: f64) -> f64 {
    if df <= 0.0 || q <= 0.0 {
        return f64::NAN;
    }

    // For nullifying (q=1): f = |t| / sqrt(df)
    // For partial reduction: f = q * |t| / sqrt(df)
    let f = q * t_stat.abs() / df.sqrt();

    // RV = 0.5 * (sqrt(f^4 + 4f^2) - f^2)
    let f2 = f.powi(2);
    let f4 = f.powi(4);
    0.5 * ((f4 + 4.0 * f2).sqrt() - f2)
}

/// Compute robustness value for statistical significance.
///
/// RV_{q,α} is the minimum strength of confounding (partial R² with both treatment and
/// outcome) that would make the bias-adjusted estimate, reduced by q·100%, not significant
/// at level α. As `sensemakr::robustness_value(t, dof, q, alpha)`:
///
/// - f_q = q|t| / sqrt(df), f_c = |qt(α/2, df - 1)| / sqrt(df - 1) (0 at α = 1),
///   f_qa = f_q - f_c;
/// - RV = 0 when f_qa <= 0 (already not significant at that reduction);
/// - RV = (f_q² - f_c²) / (1 + f_q²) when f_c > 0 and f_q > 1/f_c (the extreme
///   robustness value, where a confounder explaining all the residual variance of the
///   outcome is needed);
/// - otherwise RV = 2 / (1 + sqrt(1 + 4/f_qa²)).
///
/// # Arguments
/// * `t_stat` - The classical OLS t-statistic
/// * `df` - Residual degrees of freedom (must exceed 1)
/// * `q` - Proportion of effect to explain away (non-negative)
/// * `alpha` - Significance level, 0 < alpha <= 1
///
/// Returns NaN for inputs outside those ranges.
///
/// # References
/// Cinelli & Hazlett (2020), Section 4.1 and Appendix
pub fn robustness_value_alpha(t_stat: f64, df: f64, q: f64, alpha: f64) -> f64 {
    if !(t_stat.is_finite() && df.is_finite() && q.is_finite() && alpha.is_finite())
        || df <= 1.0
        || q < 0.0
        || !(alpha > 0.0 && alpha <= 1.0)
    {
        return f64::NAN;
    }
    let fq = q * t_stat.abs() / df.sqrt();
    let fc = if alpha == 1.0 {
        0.0
    } else {
        t_critical(alpha, df - 1.0) / (df - 1.0).sqrt()
    };
    let fqa = fq - fc;
    if fqa.is_nan() || fqa <= 0.0 {
        return if fqa.is_nan() { f64::NAN } else { 0.0 };
    }
    if fc > 0.0 && fq > 1.0 / fc {
        return (fq * fq - fc * fc) / (1.0 + fq * fq);
    }
    2.0 / (1.0 + (1.0 + 4.0 / (fqa * fqa)).sqrt())
}

/// Compute the bias from a hypothetical unobserved confounder.
///
/// bias = se * sqrt(R²_{Y~U|X,D}) * sqrt(R²_{D~U|X} * df / (1 - R²_{D~U|X}))
///
/// # Arguments
/// * `se` - Standard error of the treatment coefficient
/// * `r2_yd` - Partial R² of the confounder with outcome given treatment and covariates
/// * `r2_xz` - Partial R² of the confounder with treatment given covariates
/// * `df` - Residual degrees of freedom
///
/// # References
/// Cinelli & Hazlett (2020), Equation (9)
pub fn confounding_bias(se: f64, r2_yd: f64, r2_xz: f64, df: f64) -> f64 {
    if df <= 0.0 || r2_xz >= 1.0 || r2_yd < 0.0 || r2_xz < 0.0 {
        return f64::NAN;
    }

    // bias = se * sqrt(r2_yd) * sqrt(r2_xz * df / (1 - r2_xz))
    se * r2_yd.sqrt() * (r2_xz * df / (1.0 - r2_xz)).sqrt()
}

/// Compute the adjusted estimate under a confounding scenario.
///
/// # Arguments
/// * `estimate` - Original coefficient estimate
/// * `se` - Standard error
/// * `r2_yd` - Hypothesized partial R² of confounder with outcome
/// * `r2_xz` - Hypothesized partial R² of confounder with treatment
/// * `df` - Residual degrees of freedom
///
/// # Returns
/// The adjusted coefficient, which is the original estimate minus the bias
/// in the direction of the original estimate's sign.
pub fn adjusted_estimate(estimate: f64, se: f64, r2_yd: f64, r2_xz: f64, df: f64) -> f64 {
    let bias = confounding_bias(se, r2_yd, r2_xz, df);

    // Subtract bias in the direction of the estimate
    if estimate >= 0.0 {
        estimate - bias
    } else {
        estimate + bias
    }
}

/// Compute the adjusted standard error under a confounding scenario with no association
/// between the confounder and the treatment (R²_{D~Z|X} = 0).
///
/// se_adj = se * sqrt(1 - r2_yd) * sqrt(df / (df - 1)), which is
/// [`adjusted_se_ovb`]`(se, r2_yd, 0.0, df)`; use that function for a confounder that is
/// also associated with the treatment.
///
/// # References
/// sensemakr R package, `adjusted_se.numeric`
pub fn adjusted_se(se: f64, r2_yd: f64, df: f64) -> f64 {
    adjusted_se_ovb(se, r2_yd, 0.0, df)
}

/// Compute the bias-adjusted standard error of the treatment coefficient, as sensemakr's
/// `adjusted_se(se, dof, r2dz.x, r2yz.dx)`:
///
/// se_adj = sqrt((1 - R²_{Y~Z|D,X}) / (1 - R²_{D~Z|X})) * se * sqrt(df / (df - 1))
///
/// # Arguments
/// * `se` - Standard error of the treatment coefficient (classical OLS)
/// * `r2yz_dx` - Partial R² of the confounder with the outcome given treatment and covariates
/// * `r2dz_x` - Partial R² of the confounder with the treatment given the covariates
/// * `df` - Residual degrees of freedom
///
/// Returns NaN for df <= 1, a partial R² outside [0, 1], or R²_{D~Z|X} = 1.
///
/// # References
/// Cinelli & Hazlett (2020), Section 4; sensemakr R package, `adjusted_se.numeric`
pub fn adjusted_se_ovb(se: f64, r2yz_dx: f64, r2dz_x: f64, df: f64) -> f64 {
    let in_unit = |r: f64| (0.0..=1.0).contains(&r);
    if df.is_nan() || df <= 1.0 || !in_unit(r2yz_dx) || !in_unit(r2dz_x) || r2dz_x >= 1.0 {
        return f64::NAN;
    }
    ((1.0 - r2yz_dx) / (1.0 - r2dz_x)).sqrt() * se * (df / (df - 1.0)).sqrt()
}

/// Bounds on the strength of a confounder Z that is `kd` times as strongly associated with
/// the treatment and `ky` times as strongly with the outcome as an observed covariate X_j,
/// as sensemakr's `ovb_partial_r2_bound(r2dxj.x, r2yxj.dx, kd, ky)` (Cinelli & Hazlett 2020,
/// Section 6):
///
/// - R²_{D~Z|X} = kd R²_{D~X_j|X_-j} / (1 - R²_{D~X_j|X_-j}), which must be below 1;
/// - with r = R²_{D~X_j|X_-j}, R²_{Z~X_j|X_-j,D} = kd r² / ((1 - kd r)(1 - r)), which must be
///   below 1;
/// - R²_{Y~Z|D,X} = ((sqrt(ky) + sqrt(R²_{Z~X_j})) / sqrt(1 - R²_{Z~X_j}))² ·
///   R²_{Y~X_j|D,X_-j} / (1 - R²_{Y~X_j|D,X_-j}), set to 1 when it exceeds 1.
///
/// # Arguments
/// * `r2dxj_x` - Partial R² of the benchmark with the treatment, R²_{D~X_j|X_-j}
/// * `r2yxj_dx` - Partial R² of the benchmark with the outcome, R²_{Y~X_j|D,X_-j}
/// * `kd` - Multiplier on the treatment side (finite, > 0)
/// * `ky` - Multiplier on the outcome side (finite, >= 0)
///
/// # Returns
/// `(r2dz_x, r2yz_dx, clamped)`, `clamped` true when R²_{Y~Z|D,X} was set to 1.
///
/// # Errors
/// `InvalidSpecification` for a multiplier outside its range, a partial R² outside [0, 1),
/// or an impossible kd (an implied partial R² of 1 or more).
pub fn ovb_partial_r2_bound(
    r2dxj_x: f64,
    r2yxj_dx: f64,
    kd: f64,
    ky: f64,
) -> EconResult<(f64, f64, bool)> {
    if !(kd.is_finite() && kd > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!("kd must be a finite number above 0, got {kd}"),
        });
    }
    if !(ky.is_finite() && ky >= 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!("ky must be a finite number of at least 0, got {ky}"),
        });
    }
    let partial = |r: f64| r.is_finite() && (0.0..1.0).contains(&r);
    if !(partial(r2dxj_x) && partial(r2yxj_dx)) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "benchmark partial R² must lie in [0, 1), got R²(D~X_j|X) = {r2dxj_x}, \
                 R²(Y~X_j|D,X) = {r2yxj_dx}"
            ),
        });
    }
    let r2dz_x = kd * (r2dxj_x / (1.0 - r2dxj_x));
    if r2dz_x >= 1.0 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "kd = {kd} implies R²(D~Z|X) = {r2dz_x} >= 1, which is impossible; try a lower kd"
            ),
        });
    }
    let r2zxj = kd * r2dxj_x * r2dxj_x / ((1.0 - kd * r2dxj_x) * (1.0 - r2dxj_x));
    if r2zxj.is_nan() || r2zxj >= 1.0 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "kd = {kd} implies R²(Z~X_j|X,D) = {r2zxj} >= 1, which is impossible; try a lower kd"
            ),
        });
    }
    let r2yz_dx =
        ((ky.sqrt() + r2zxj.sqrt()) / (1.0 - r2zxj).sqrt()).powi(2) * (r2yxj_dx / (1.0 - r2yxj_dx));
    if r2yz_dx > 1.0 {
        Ok((r2dz_x, 1.0, true))
    } else {
        Ok((r2dz_x, r2yz_dx, false))
    }
}

/// Compute partial R² of a covariate with outcome, residualized on other covariates.
///
/// This computes R²_{Y~X_j|X_{-j}} by running auxiliary regressions.
fn partial_r2_covariate_outcome(
    y: &Array1<f64>,
    x: &Array2<f64>,
    covariate_idx: usize,
    name: &str,
) -> EconResult<f64> {
    let n = x.nrows();
    let k = x.ncols();

    if covariate_idx >= k {
        return Err(EconError::InvalidSpecification {
            message: format!("Covariate index {} out of bounds", covariate_idx),
        });
    }

    // Regress y on all covariates except the one at covariate_idx
    // Then get residuals and compute partial R² from the t-stat

    if k == 1 {
        // Only one covariate - partial R² is total R²
        let y_mean = y.mean().unwrap_or(0.0);
        let tss: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
        let x_j = x.column(covariate_idx);

        // Regress y on x_j alone
        let xtx_j = x_j.dot(&x_j);
        let xty_j = x_j.dot(y);
        let beta_j = xty_j / xtx_j;
        let residuals: Array1<f64> = y
            .iter()
            .zip(x_j.iter())
            .map(|(&yi, &xi)| yi - beta_j * xi)
            .collect();
        let rss: f64 = residuals.iter().map(|&r| r.powi(2)).sum();

        return Ok(if tss > 1e-10 { 1.0 - rss / tss } else { 0.0 });
    }

    // X_{-j}: X without column j
    let k_minus_1 = k - 1;
    let mut x_minus_j = Array2::<f64>::zeros((n, k_minus_1));
    for (dst, src) in (0..k).filter(|&j| j != covariate_idx).enumerate() {
        x_minus_j.column_mut(dst).assign(&x.column(src));
    }

    // Residuals of y and of X_j on X_{-j}. An aliased X_{-j} column (which the pivoted QR,
    // lm.fit, decides), or X_j in the span of X_{-j}, leaves the partial R² undefined.
    let collinear = || EconError::SingularMatrix {
        context: format!("benchmark covariate '{name}' is collinear with the other covariates"),
        suggestion: "drop the collinear covariate or choose another benchmark".to_string(),
    };
    // Both regressions share X_{-j}, so they share one solve decision (OLS's: the normal
    // equations on a full-rank, well-conditioned X_{-j}, the QR otherwise).
    let x_j = x.column(covariate_idx).to_owned();
    let (y_resid, x_j_resid) = match normal_equations_inverse(x_minus_j.view())? {
        Some(normal) => {
            let xm = x_minus_j.view();
            let y_resid = y - &xm.dot(&normal.coefficients(xm, y));
            let x_j_resid = &x_j - &xm.dot(&normal.coefficients(xm, &x_j));
            (y_resid, x_j_resid)
        }
        None => {
            let fit_y = least_squares(x_minus_j.view(), y.view())?;
            if fit_y.rank < k_minus_1 {
                return Err(collinear());
            }
            let fit_x = least_squares(x_minus_j.view(), x_j.view())?;
            (fit_y.residuals, fit_x.residuals)
        }
    };
    let x_scale = x_j.dot(&x_j).sqrt();
    if x_j_resid.dot(&x_j_resid).sqrt() <= LM_QR_TOL * x_scale {
        return Err(collinear());
    }

    // Partial R² = correlation between residuals squared
    let y_resid_mean = y_resid.mean().unwrap_or(0.0);
    let x_j_resid_mean = x_j_resid.mean().unwrap_or(0.0);

    let cov: f64 = y_resid
        .iter()
        .zip(x_j_resid.iter())
        .map(|(&y, &x)| (y - y_resid_mean) * (x - x_j_resid_mean))
        .sum();
    let var_y: f64 = y_resid.iter().map(|&y| (y - y_resid_mean).powi(2)).sum();
    let var_x: f64 = x_j_resid
        .iter()
        .map(|&x| (x - x_j_resid_mean).powi(2))
        .sum();

    if var_y < 1e-10 || var_x < 1e-10 {
        return Ok(0.0);
    }

    let r = cov / (var_y.sqrt() * var_x.sqrt());
    Ok(r.powi(2))
}

/// Compute partial R² of a covariate with treatment, residualized on other covariates.
///
/// This computes R²_{D~X_j|X_{-j}} by running auxiliary regressions.
fn partial_r2_covariate_treatment(
    treatment: &Array1<f64>,
    x: &Array2<f64>,
    covariate_idx: usize,
    name: &str,
) -> EconResult<f64> {
    // Same as partial_r2_covariate_outcome but with treatment instead of y
    partial_r2_covariate_outcome(treatment, x, covariate_idx, name)
}

/// sensemakr's statistics use the classical OLS t: the partial R² identity t² / (t² + df),
/// the robustness values and the bias formula do not hold for a heteroskedasticity-robust t.
fn require_classical_fit(ols_result: &OlsResult) -> EconResult<()> {
    if ols_result.cov_type != CovarianceType::Standard {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "sensemakr needs the classical OLS fit (CovarianceType::Standard), got {}: \
                 its partial R², robustness values and bias bounds are defined for the \
                 classical t-statistic",
                ols_result.cov_type
            ),
        });
    }
    Ok(())
}

/// Run sensitivity analysis from an existing OLS result.
///
/// This is the main entry point when you already have OLS results.
///
/// # Arguments
/// * `ols_result` - Result from a previous OLS regression with `CovarianceType::Standard`
///   (any other covariance type is an `InvalidSpecification`)
/// * `treatment` - Name of the treatment variable to analyze
/// * `benchmark_covariates` - Optional list of covariates to use as sensitivity benchmarks
/// * `kd` - Multiplier for treatment benchmark partial R² (default 1.0)
/// * `ky` - Multiplier for outcome benchmark partial R² (default = kd)
/// * `q` - Proportion of effect to reduce (default 1.0 = nullify)
/// * `alpha` - Significance level (default 0.05)
///
/// # Example
/// ```ignore
/// let ols_result = run_ols(&dataset, "y", &["treatment", "x1", "x2"], true, CovarianceType::Standard)?;
/// let sens = sensemakr(&ols_result, "treatment", Some(&["x1", "x2"]), None, None, 1.0, 0.05)?;
/// println!("{}", sens);
/// ```
pub fn sensemakr(
    ols_result: &OlsResult,
    treatment: &str,
    benchmark_covariates: Option<&[&str]>,
    kd: Option<f64>,
    ky: Option<f64>,
    q: f64,
    alpha: f64,
) -> EconResult<SensemakrResult> {
    // No design matrix available here, so benchmark bounds fall back to the
    // approximation (disclosed via a warning on the result). Callers that have
    // the data should use `run_sensemakr`, which computes exact bounds.
    sensemakr_impl(
        ols_result,
        treatment,
        benchmark_covariates,
        None,
        kd,
        ky,
        q,
        alpha,
    )
}

/// Internal sensitivity-analysis implementation.
///
/// `benchmark_treatment_r2`, when supplied, maps each benchmark covariate name
/// to the exact partial R² of the treatment on that covariate given the others
/// (R²_{D~X_j|X_{-j}}). When it is `None` (or a benchmark is missing from it),
/// the benchmark's treatment partial R² is approximated and a warning is added.
#[allow(clippy::too_many_arguments)]
fn sensemakr_impl(
    ols_result: &OlsResult,
    treatment: &str,
    benchmark_covariates: Option<&[&str]>,
    benchmark_treatment_r2: Option<&HashMap<String, f64>>,
    kd: Option<f64>,
    ky: Option<f64>,
    q: f64,
    alpha: f64,
) -> EconResult<SensemakrResult> {
    crate::traits::estimator::validate_level(alpha, "alpha")?;
    if !(q.is_finite() && q >= 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "q (the fraction of the estimate to explain away) must be finite and at least 0, got {q}"
            ),
        });
    }
    // Find treatment in variable names
    let treatment_idx = ols_result
        .variable_names
        .iter()
        .position(|name| name == treatment)
        .ok_or_else(|| EconError::ColumnNotFound {
            column: treatment.to_string(),
            available: ols_result.variable_names.clone(),
        })?;

    require_classical_fit(ols_result)?;
    let coef = &ols_result.coefficients[treatment_idx];
    let estimate = coef.estimate;
    let se = coef.std_error;
    let t_stat = coef.t_value;
    let df = ols_result.df_resid as f64;

    // Compute core sensitivity statistics
    let partial_r2_yd = partial_r2(t_stat, df);
    let rv = robustness_value(t_stat, df, q);
    let rv_alpha = robustness_value_alpha(t_stat, df, q, alpha);

    // Compute benchmark bounds if covariates provided
    let mut benchmark_bounds = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    let mut approx_benchmarks: Vec<String> = Vec::new();

    if let Some(benchmarks) = benchmark_covariates {
        let kd = kd.unwrap_or(1.0);
        let ky = ky.unwrap_or(kd);
        // the multipliers are checked even when no benchmark matches
        ovb_partial_r2_bound(0.0, 0.0, kd, ky)?;

        for &benchmark_name in benchmarks {
            // Find benchmark in variable names (skip intercept typically)
            if let Some(_bench_idx) = ols_result
                .variable_names
                .iter()
                .position(|name| name == benchmark_name)
            {
                // We need the original data to compute benchmark partial R²
                // For now, we'll use a simplified approach based on the t-statistics
                // In a full implementation, we'd store X in OlsResult or pass it here

                // Find benchmark coefficient
                if let Some(bench_coef) = ols_result
                    .coefficients
                    .iter()
                    .find(|c| c.name == benchmark_name)
                {
                    // Benchmark partial R² with the OUTCOME (R²_{Y~X_j|X_{-j}}),
                    // recovered from the benchmark's t-statistic.
                    let r2_benchmark_y = partial_r2(bench_coef.t_value, df);

                    // Benchmark partial R² with the TREATMENT (R²_{D~X_j|X_{-j}}).
                    // Use the exact value computed from the design matrix when the
                    // caller supplied it (run_sensemakr); otherwise fall back to a
                    // crude 0.5 * R²_{Y~X_j} approximation and flag it.
                    let r2_benchmark_d =
                        match benchmark_treatment_r2.and_then(|m| m.get(benchmark_name)) {
                            Some(&r2) => r2,
                            None => {
                                approx_benchmarks.push(benchmark_name.to_string());
                                r2_benchmark_y * 0.5
                            }
                        };

                    // Cinelli and Hazlett's bounds from the benchmark's partial R²
                    let (r2_xz_bound, r2_yd_bound, clamped) =
                        ovb_partial_r2_bound(r2_benchmark_d, r2_benchmark_y, kd, ky).map_err(
                            |e| match e {
                                EconError::InvalidSpecification { message } => {
                                    EconError::InvalidSpecification {
                                        message: format!("benchmark '{benchmark_name}': {message}"),
                                    }
                                }
                                e => e,
                            },
                        )?;
                    if clamped {
                        warnings.push(format!(
                            "the implied R²(Y~Z|D,X) of benchmark '{benchmark_name}' (kd = {kd}, \
                             ky = {ky}) exceeds 1; set to 1, as sensemakr does"
                        ));
                    }

                    // Bias-adjusted estimate, SE, t (h0 = (1 - q) estimate) and CI, as the
                    // bounds table of sensemakr() reports them (reduce = TRUE)
                    let bias = confounding_bias(se, r2_yd_bound, r2_xz_bound, df);
                    let adj_est = if estimate >= 0.0 {
                        estimate - bias
                    } else {
                        estimate + bias
                    };
                    let adj_se = adjusted_se_ovb(se, r2_yd_bound, r2_xz_bound, df);
                    let h0 = estimate * (1.0 - q);
                    let adj_t = (adj_est - h0) / adj_se;
                    let t_crit = t_critical(alpha, df);

                    benchmark_bounds.push(SensitivityBound {
                        benchmark: benchmark_name.to_string(),
                        r2_yd_benchmark: r2_benchmark_y,
                        r2_xz_benchmark: r2_benchmark_d,
                        kd,
                        ky,
                        r2_yd_bound,
                        r2_xz_bound,
                        bias,
                        adjusted_estimate: adj_est,
                        adjusted_se: adj_se,
                        adjusted_t: adj_t,
                        adjusted_ci_lower: adj_est - t_crit * adj_se,
                        adjusted_ci_upper: adj_est + t_crit * adj_se,
                    });
                }
            }
        }
    }

    if !approx_benchmarks.is_empty() {
        warnings.push(format!(
            "Benchmark bounds for [{}] use an approximation (R²_(D~X_j) ≈ 0.5 · R²_(Y~X_j)) \
             because the design matrix was not available; call run_sensemakr(dataset, ...) for \
             exact benchmark bounds.",
            approx_benchmarks.join(", ")
        ));
    }

    Ok(SensemakrResult {
        treatment: treatment.to_string(),
        estimate,
        std_error: se,
        t_statistic: t_stat,
        df,
        partial_r2_yd,
        robustness_value: rv,
        robustness_value_alpha: rv_alpha,
        rv_q: q,
        rv_alpha: alpha,
        benchmark_bounds,
        contour_data: None,
        warnings,
    })
}

/// Run sensitivity analysis directly from a dataset.
///
/// This is a convenience function that runs OLS first, then sensitivity analysis.
///
/// # Arguments
/// * `dataset` - The dataset containing the data
/// * `y_col` - Name of the outcome variable
/// * `treatment_col` - Name of the treatment variable
/// * `covariate_cols` - Names of the control covariates
/// * `benchmark_covariates` - Optional covariates to use as benchmarks
/// * `kd` - Multiplier for treatment benchmark partial R² (default 1.0)
/// * `ky` - Multiplier for outcome benchmark partial R² (default = kd)
/// * `q` - Proportion of effect to reduce (default 1.0)
/// * `alpha` - Significance level (default 0.05)
///
/// # Example
/// ```ignore
/// let sens = run_sensemakr(
///     &dataset,
///     "wage",
///     "education",
///     &["experience", "age"],
///     Some(&["experience"]),
///     Some(1.0), Some(1.0), 1.0, 0.05,
/// )?;
/// println!("{}", sens);
/// ```
pub fn run_sensemakr(
    dataset: &Dataset,
    y_col: &str,
    treatment_col: &str,
    covariate_cols: &[&str],
    benchmark_covariates: Option<&[&str]>,
    kd: Option<f64>,
    ky: Option<f64>,
    q: f64,
    alpha: f64,
) -> EconResult<SensemakrResult> {
    // Build full list of X variables: treatment + covariates
    let mut x_cols: Vec<&str> = vec![treatment_col];
    x_cols.extend(covariate_cols);

    // sensemakr works on the classical lm fit
    let ols_result = run_ols(dataset, y_col, &x_cols, true, CovarianceType::Standard)?;

    // Compute the exact benchmark partial R² of the treatment on each benchmark
    // covariate, R²_{D~X_j|X_{-j}}, by regressing the treatment on the control
    // covariates (with intercept). This replaces the crude 0.5 approximation
    // that the OLS-only entry point must use for lack of the design matrix.
    let benchmark_treatment_r2 = match benchmark_covariates {
        Some(benchmarks) if !benchmarks.is_empty() => {
            let df = dataset.df();
            let treatment_vec = DesignMatrix::extract_column(df, treatment_col).map_err(|e| {
                EconError::InvalidSpecification {
                    message: format!(
                        "sensemakr benchmark: cannot read treatment '{treatment_col}': {e}"
                    ),
                }
            })?;
            let n = treatment_vec.len();

            // Design matrix for D ~ [intercept, control covariates].
            let mut x = Array2::<f64>::ones((n, covariate_cols.len() + 1));
            for (j, &cov) in covariate_cols.iter().enumerate() {
                let col = DesignMatrix::extract_column(df, cov).map_err(|e| {
                    EconError::InvalidSpecification {
                        message: format!("sensemakr benchmark: cannot read covariate '{cov}': {e}"),
                    }
                })?;
                x.column_mut(j + 1).assign(&col);
            }

            let mut map = HashMap::new();
            for &b in benchmarks {
                if let Some(pos) = covariate_cols.iter().position(|&c| c == b) {
                    // pos + 1 skips the intercept column at index 0.
                    let r2 = partial_r2_covariate_treatment(&treatment_vec, &x, pos + 1, b)?;
                    map.insert(b.to_string(), r2);
                }
            }
            Some(map)
        }
        _ => None,
    };

    // Run sensitivity analysis with the exact benchmark partial R² values.
    sensemakr_impl(
        &ols_result,
        treatment_col,
        benchmark_covariates,
        benchmark_treatment_r2.as_ref(),
        kd,
        ky,
        q,
        alpha,
    )
}

/// Generate contour data for sensitivity plots.
///
/// Creates a grid of (R²_{Y~U|X,D}, R²_{D~U|X}) values and computes
/// the adjusted estimate and t-statistic at each point.
///
/// # Arguments
/// * `ols_result` - The OLS result, with `CovarianceType::Standard`
/// * `treatment` - Name of the treatment variable
/// * `grid_size` - Number of points in each dimension (default 20)
/// * `max_r2` - Maximum partial R² value for the grid (default 0.5)
pub fn generate_contour_data(
    ols_result: &OlsResult,
    treatment: &str,
    grid_size: Option<usize>,
    max_r2: Option<f64>,
) -> EconResult<ContourData> {
    let grid_size = grid_size.unwrap_or(20);
    let max_r2 = max_r2.unwrap_or(0.5);
    require_classical_fit(ols_result)?;
    if grid_size < 2 {
        return Err(EconError::InvalidSpecification {
            message: format!("contour grid_size must be at least 2, got {grid_size}"),
        });
    }
    if !(max_r2.is_finite() && max_r2 > 0.0 && max_r2 < 1.0) {
        return Err(EconError::InvalidSpecification {
            message: format!("contour max_r2 must lie in (0, 1), got {max_r2}"),
        });
    }
    // two grid_size x grid_size grids of f64: the whole grid is held to the allocation ceiling
    crate::resource::check_alloc::<f64>(
        crate::resource::checked_len(&[grid_size, grid_size], "sensemakr contour grid")?,
        "sensemakr contour grid",
    )?;

    // Find treatment coefficient
    let treatment_idx = ols_result
        .variable_names
        .iter()
        .position(|name| name == treatment)
        .ok_or_else(|| EconError::ColumnNotFound {
            column: treatment.to_string(),
            available: ols_result.variable_names.clone(),
        })?;

    let coef = &ols_result.coefficients[treatment_idx];
    let estimate = coef.estimate;
    let se = coef.std_error;
    let df = ols_result.df_resid as f64;

    // Create grid
    let step = max_r2 / (grid_size as f64 - 1.0);
    let r2_yd_values: Vec<f64> = (0..grid_size).map(|i| i as f64 * step).collect();
    let r2_xz_values: Vec<f64> = (0..grid_size).map(|i| i as f64 * step).collect();

    let mut adjusted_estimates =
        crate::resource::try_with_capacity(grid_size, "sensemakr contour grid")?;
    let mut adjusted_t_stats =
        crate::resource::try_with_capacity(grid_size, "sensemakr contour grid")?;

    for &r2_xz in &r2_xz_values {
        crate::resource::checkpoint()?;
        let mut row_estimates: Vec<f64> =
            crate::resource::try_with_capacity(grid_size, "sensemakr contour grid")?;
        let mut row_t_stats: Vec<f64> =
            crate::resource::try_with_capacity(grid_size, "sensemakr contour grid")?;

        for &r2_yd in &r2_yd_values {
            // sensemakr's adjusted_estimate and adjusted_t (h0 = 0) at this point
            let adj_est = adjusted_estimate(estimate, se, r2_yd, r2_xz, df);
            let adj_se = adjusted_se_ovb(se, r2_yd, r2_xz, df);
            let adj_t = adj_est / adj_se;

            row_estimates.push(adj_est);
            row_t_stats.push(adj_t);
        }

        adjusted_estimates.push(row_estimates);
        adjusted_t_stats.push(row_t_stats);
    }

    Ok(ContourData {
        r2_yd_values,
        r2_xz_values,
        adjusted_estimates,
        adjusted_t_stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Dataset;
    use polars::prelude::*;

    /// Test partial R² calculation matches formula.
    #[test]
    fn test_partial_r2() {
        // t = 3.0, df = 100 -> R² = 9 / (9 + 100) = 9/109 ≈ 0.08257
        let r2 = partial_r2(3.0, 100.0);
        assert!((r2 - 0.0825688).abs() < 1e-5);

        // t = 0 -> R² = 0
        let r2_zero = partial_r2(0.0, 100.0);
        assert!((r2_zero - 0.0).abs() < 1e-10);

        // t = 10, df = 50 -> R² = 100 / 150 = 0.667
        let r2_large = partial_r2(10.0, 50.0);
        assert!((r2_large - 0.666667).abs() < 1e-5);
    }

    /// Test robustness value calculation.
    #[test]
    fn test_robustness_value() {
        // For t = 5.0, df = 100, q = 1
        // f = 5 / sqrt(100) = 0.5
        // f^2 = 0.25, f^4 = 0.0625
        // RV = 0.5 * (sqrt(0.0625 + 1) - 0.25) = 0.5 * (1.0308 - 0.25) = 0.390
        let rv = robustness_value(5.0, 100.0, 1.0);
        assert!(
            rv > 0.0 && rv < 1.0,
            "RV should be between 0 and 1, got {}",
            rv
        );

        // RV should be 0 when t = 0
        let rv_zero = robustness_value(0.0, 100.0, 1.0);
        assert!((rv_zero - 0.0).abs() < 1e-10, "RV should be 0 for t=0");

        // RV should increase with t
        let rv_small = robustness_value(2.0, 100.0, 1.0);
        let rv_large = robustness_value(5.0, 100.0, 1.0);
        assert!(rv_large > rv_small, "RV should increase with t");
    }

    /// Test bias calculation.
    #[test]
    fn test_confounding_bias() {
        // se = 0.1, r2_yd = 0.1, r2_xz = 0.1, df = 100
        // bias = 0.1 * sqrt(0.1) * sqrt(0.1 * 100 / 0.9)
        //      = 0.1 * 0.316 * sqrt(11.11)
        //      = 0.1 * 0.316 * 3.33
        //      = 0.105
        let bias = confounding_bias(0.1, 0.1, 0.1, 100.0);
        assert!(
            bias > 0.0 && bias < 0.5,
            "Bias should be reasonable, got {}",
            bias
        );

        // Bias should be 0 when r2 values are 0
        let bias_zero = confounding_bias(0.1, 0.0, 0.0, 100.0);
        assert!((bias_zero - 0.0).abs() < 1e-10);
    }

    /// Test adjusted estimate calculation.
    #[test]
    fn test_adjusted_estimate() {
        let adj = adjusted_estimate(1.0, 0.1, 0.1, 0.1, 100.0);

        // Adjusted should be less than original for positive estimate
        assert!(adj < 1.0, "Adjusted estimate should be less than original");
        assert!(
            adj > 0.0,
            "Adjusted estimate should still be positive for small confounding"
        );
    }

    /// Test sensemakr with synthetic data.
    #[test]
    fn test_sensemakr_synthetic() {
        // Create dataset with clear treatment effect
        // y = 0.5 + 2*treatment + 1*x1 + 0.5*x2 + noise
        let n = 200;
        let mut rng = rand::thread_rng();
        use rand::Rng;

        let treatment: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
        let x1: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let x2: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let noise: Vec<f64> = (0..n).map(|_| rng.gen_range(-0.5..0.5)).collect();

        let y: Vec<f64> = (0..n)
            .map(|i| 0.5 + 2.0 * treatment[i] + 1.0 * x1[i] + 0.5 * x2[i] + noise[i])
            .collect();

        let df = df! {
            "y" => y,
            "treatment" => treatment,
            "x1" => x1,
            "x2" => x2,
        }
        .unwrap();

        let dataset = Dataset::new(df);

        let result = run_sensemakr(
            &dataset,
            "y",
            "treatment",
            &["x1", "x2"],
            Some(&["x1"]),
            Some(1.0),
            Some(1.0),
            1.0,
            0.05,
        )
        .unwrap();

        // Check basic properties
        assert!(!result.treatment.is_empty());
        assert!(
            result.estimate.abs() > 1.0,
            "Treatment effect should be detectable"
        );
        assert!(result.partial_r2_yd > 0.0, "Partial R² should be positive");
        assert!(
            result.robustness_value > 0.0,
            "RV should be positive for significant effect"
        );

        // Print result for inspection
        println!("{}", result);
    }

    /// Test that RV matches expected values from Cinelli & Hazlett (2020).
    #[test]
    fn test_rv_against_paper_examples() {
        // Example from paper: t = 5.0, df = 100
        // Expected RV ≈ 0.39 (approximately)
        let rv = robustness_value(5.0, 100.0, 1.0);
        assert!(
            (rv - 0.39).abs() < 0.05,
            "RV for t=5, df=100 should be approximately 0.39, got {}",
            rv
        );

        // For very large t, RV approaches 1
        let rv_large = robustness_value(50.0, 100.0, 1.0);
        assert!(
            rv_large > 0.9,
            "RV should be close to 1 for very large t, got {}",
            rv_large
        );
    }

    const COVS: [&str; 6] = [
        "age",
        "farmer_dar",
        "herder_dar",
        "pastvoted",
        "hhsize_darfur",
        "female",
    ];

    /// Case `sensemakr_darfur`: sensemakr's formulas on the classical OLS t of darfur (no
    /// village effects): dof 1268.
    #[test]
    fn test_validate_sensemakr_darfur() {
        use crate::regression::ols::{CovarianceType, run_ols};
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("sensemakr_darfur");
        let n = case.scalar("n") as usize;
        let mut x = vec!["directlyharmed"];
        x.extend(COVS);
        let ols = run_ols(&ds, "peacefactor", &x, true, CovarianceType::Standard).unwrap();
        let s = sensemakr(&ols, "directlyharmed", None, None, None, 1.0, 0.05).unwrap();
        let (t, dof) = (case.scalar("t"), case.scalar("dof"));
        Check::new(&case)
            .scalar("estimate", s.estimate, Tol::coef(n))
            .scalar("se", s.std_error, Tol::se(n))
            .scalar("t", s.t_statistic, Tol::stat(n))
            .scalar("dof", s.df, Tol::exact())
            .scalar("r2yd_x", s.partial_r2_yd, Tol::stat(n))
            .scalar("rv_q", s.robustness_value, Tol::stat(n))
            .scalar("rv_q05", robustness_value(t, dof, 0.5), Tol::stat(n))
            .finish();
        assert!((partial_r2(t, dof) - case.scalar("r2yd_x")).abs() <= 1e-14);
        Check::new(&case)
            .scalar("rv_qa", s.robustness_value_alpha, Tol::stat(n))
            .finish();
    }

    /// Case `sensemakr_rv_alpha`: sensemakr's `robustness_value(t, dof, q, alpha)` on six
    /// inputs (`sensemakr_rv_inputs.csv`), row 1 in the extreme-RV branch. Closed forms.
    #[test]
    fn test_validate_sensemakr_rv_alpha() {
        use crate::data::columns::ColumnRequest;
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("sensemakr_rv_alpha");
        let cols = ColumnRequest::new(ds.df())
            .numerics(&["t", "dof", "q", "alpha"])
            .extract()
            .unwrap();
        let (t, dof, q, alpha) = (
            cols.numeric("t"),
            cols.numeric("dof"),
            cols.numeric("q"),
            cols.numeric("alpha"),
        );
        let rv: Vec<f64> = (0..t.len())
            .map(|i| robustness_value_alpha(t[i], dof[i], q[i], alpha[i]))
            .collect();
        // A closed form, but through a t quantile: p2a's and R's qt(0.95, 49) differ by
        // 5.5e-13 relative, and f_q - f_c amplifies that about 2 f_c / f_qa = 11 times in
        // row 5 (3.0e-12 relative), so 1e-10 rather than the closed-form 1e-12.
        Check::new(&case)
            .vector("rv", &rv, Tol::new(1e-10, 1e-14))
            .finish();
        // is_xrv: row 1 is the extreme branch (fq^2 - fc^2) / (1 + fq^2)
        assert_eq!(case.vector("is_xrv")[0], 1.0);
        let fq = q[0] * t[0].abs() / dof[0].sqrt();
        let fc = t_critical(alpha[0], dof[0] - 1.0) / (dof[0] - 1.0).sqrt();
        assert!(fq > 1.0 / fc);
        let xrv = (fq * fq - fc * fc) / (1.0 + fq * fq);
        assert!((rv[0] - xrv).abs() <= 1e-15, "{} vs {xrv}", rv[0]);
        // invalid inputs
        assert!(robustness_value_alpha(3.0, 1.0, 1.0, 0.05).is_nan());
        assert!(robustness_value_alpha(3.0, 10.0, 1.0, 0.0).is_nan());
        assert!(robustness_value_alpha(3.0, 10.0, 1.0, 1.5).is_nan());
    }

    /// sensemakr's statistics (t^2 / (t^2 + df), RV, the bias formula) hold for the classical
    /// t only: a fit with another covariance type is refused.
    #[test]
    fn test_sensemakr_refuses_robust_ols() {
        let df = df! {
            "y" => [1.0, 2.1, 2.9, 4.2, 5.0, 5.8, 7.1, 8.0, 9.2, 10.1],
            "x" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
        .unwrap();
        let ols = run_ols(&Dataset::new(df), "y", &["x"], true, CovarianceType::HC1).unwrap();
        let check = |e: EconError| match e {
            EconError::InvalidSpecification { message } => {
                assert!(message.contains("CovarianceType::Standard"), "{message}")
            }
            e => panic!("{e:?}"),
        };
        check(sensemakr(&ols, "x", None, None, None, 1.0, 0.05).unwrap_err());
        check(generate_contour_data(&ols, "x", Some(5), Some(0.5)).unwrap_err());
    }

    /// Case `sensemakr_darfur`, X-16: run_sensemakr's benchmark partial R^2 of the treatment on
    /// female is the exact R^2_{D~X_j|X_-j} (commit 5f86b49); a revert to 0.5 * R^2_{Y~X_j}
    /// fails this pin.
    #[test]
    fn test_validate_sensemakr_darfur_benchmark() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("sensemakr_darfur");
        let n = case.scalar("n") as usize;
        let r = run_sensemakr(
            &ds,
            "peacefactor",
            "directlyharmed",
            &COVS,
            Some(&["female"]),
            Some(1.0),
            None,
            1.0,
            0.05,
        )
        .unwrap();
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
        Check::new(&case)
            .scalar("estimate", r.estimate, Tol::coef(n))
            .scalar("se", r.std_error, Tol::se(n))
            .scalar("t", r.t_statistic, Tol::stat(n))
            .scalar("r2yd_x", r.partial_r2_yd, Tol::stat(n))
            .scalar("rv_q", r.robustness_value, Tol::stat(n))
            .scalar("rv_qa", r.robustness_value_alpha, Tol::stat(n))
            .scalar(
                "female_r2dxj",
                r.benchmark_bounds[0].r2_xz_benchmark,
                Tol::stat(n),
            )
            .scalar(
                "female_r2yxj",
                r.benchmark_bounds[0].r2_yd_benchmark,
                Tol::stat(n),
            )
            .scalar(
                "bound_r2dz_x",
                r.benchmark_bounds[0].r2_xz_bound,
                Tol::stat(n),
            )
            .scalar(
                "bound_r2yz_dx",
                r.benchmark_bounds[0].r2_yd_bound,
                Tol::stat(n),
            )
            .scalar(
                "bound_adjusted_estimate",
                r.benchmark_bounds[0].adjusted_estimate,
                Tol::coef(n),
            )
            .scalar(
                "bound_adjusted_se",
                r.benchmark_bounds[0].adjusted_se,
                Tol::se(n),
            )
            .scalar(
                "bound_adjusted_lower",
                r.benchmark_bounds[0].adjusted_ci_lower,
                Tol::coef(n),
            )
            .scalar(
                "bound_adjusted_upper",
                r.benchmark_bounds[0].adjusted_ci_upper,
                Tol::coef(n),
            )
            .finish();
    }

    /// Case `sensemakr_darfur_bounds`: sensemakr's bounds table (`ovb_partial_r2_bound`,
    /// `adjusted_estimate`, `adjusted_se`, `adjusted_t` with h0 = (1 - q) estimate, the CI
    /// with qt(alpha/2, df)) for benchmarks female and age at five settings, the clamp of an
    /// implied R²(Y~Z|D,X) above 1, and the contour grid.
    #[test]
    fn test_validate_sensemakr_darfur_bounds() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("sensemakr_darfur_bounds");
        let n = case.scalar("n") as usize;
        let names = ["female", "age"];
        let mut c = Check::new(&case);
        for (nm, kd, ky, q, alpha) in [
            ("k1", 1.0, 1.0, 1.0, 0.05),
            ("k2", 2.0, 2.0, 1.0, 0.05),
            ("k3", 3.0, 3.0, 1.0, 0.05),
            ("kd2ky1", 2.0, 1.0, 1.0, 0.05),
            ("q05a10", 1.0, 1.0, 0.5, 0.10),
        ] {
            let r = run_sensemakr(
                &ds,
                "peacefactor",
                "directlyharmed",
                &COVS,
                Some(&names),
                Some(kd),
                Some(ky),
                q,
                alpha,
            )
            .unwrap();
            assert!(r.warnings.is_empty(), "{nm}: {:?}", r.warnings);
            let b = &r.benchmark_bounds;
            let bn: Vec<&str> = b.iter().map(|x| x.benchmark.as_str()).collect();
            let col = |f: fn(&SensitivityBound) -> f64| b.iter().map(f).collect::<Vec<f64>>();
            c.scalar(&format!("{nm}_rv_q"), r.robustness_value, Tol::stat(n))
                .scalar(
                    &format!("{nm}_rv_qa"),
                    r.robustness_value_alpha,
                    Tol::stat(n),
                )
                .named(
                    &format!("{nm}_r2dz_x"),
                    &bn,
                    &col(|x| x.r2_xz_bound),
                    Tol::stat(n),
                )
                .named(
                    &format!("{nm}_r2yz_dx"),
                    &bn,
                    &col(|x| x.r2_yd_bound),
                    Tol::stat(n),
                )
                .named(
                    &format!("{nm}_adj_est"),
                    &bn,
                    &col(|x| x.adjusted_estimate),
                    Tol::coef(n),
                )
                .named(
                    &format!("{nm}_adj_se"),
                    &bn,
                    &col(|x| x.adjusted_se),
                    Tol::se(n),
                )
                .named(
                    &format!("{nm}_adj_t"),
                    &bn,
                    &col(|x| x.adjusted_t),
                    Tol::stat(n),
                )
                .named(
                    &format!("{nm}_lower"),
                    &bn,
                    &col(|x| x.adjusted_ci_lower),
                    Tol::coef(n),
                )
                .named(
                    &format!("{nm}_upper"),
                    &bn,
                    &col(|x| x.adjusted_ci_upper),
                    Tol::coef(n),
                );
        }
        // ky = 8: the implied R²(Y~Z|D,X) exceeds 1; set to 1 with a warning, as sensemakr
        let r8 = run_sensemakr(
            &ds,
            "peacefactor",
            "directlyharmed",
            &COVS,
            Some(&["female"]),
            Some(1.0),
            Some(8.0),
            1.0,
            0.05,
        )
        .unwrap();
        let b8 = &r8.benchmark_bounds[0];
        c.scalar("clamp_r2dz_x", b8.r2_xz_bound, Tol::stat(n))
            .scalar("clamp_r2yz_dx", b8.r2_yd_bound, Tol::exact())
            .scalar("clamp_adj_est", b8.adjusted_estimate, Tol::coef(n));
        assert_eq!(b8.adjusted_se, 0.0);
        // R: (adj_est - h0) / 0 with h0 = 0, and a CI collapsed to the estimate
        assert_eq!(b8.adjusted_t, f64::INFINITY);
        assert_eq!(
            (b8.adjusted_ci_lower, b8.adjusted_ci_upper),
            (b8.adjusted_estimate, b8.adjusted_estimate)
        );
        assert!(
            r8.warnings.iter().any(|w| w.contains("'female'")),
            "{:?}",
            r8.warnings
        );
        // contour grid: rows R²(D~Z|X), columns R²(Y~Z|D,X), h0 = 0
        let mut x = vec!["directlyharmed"];
        x.extend(COVS);
        let ols = run_ols(&ds, "peacefactor", &x, true, CovarianceType::Standard).unwrap();
        let g = generate_contour_data(&ols, "directlyharmed", Some(5), Some(0.5)).unwrap();
        let flat = |m: &[Vec<f64>]| m.iter().flatten().copied().collect::<Vec<f64>>();
        c.vector("grid", &g.r2_xz_values, Tol::exact())
            .vector("grid", &g.r2_yd_values, Tol::exact())
            .vector(
                "contour_adj_est",
                &flat(&g.adjusted_estimates),
                Tol::coef(n),
            )
            .vector("contour_adj_t", &flat(&g.adjusted_t_stats), Tol::stat(n))
            .finish();
    }

    /// Case data for tests that compare no R value.
    fn case_data(id: &str) -> (p2a_validation::Case, Dataset) {
        crate::test_support::load_case(id)
    }

    /// An impossible benchmark multiplier (implied R²(D~Z|X) >= 1) is an error, as in
    /// sensemakr; so are kd <= 0, ky < 0 and non-finite multipliers.
    #[test]
    fn test_sensemakr_impossible_kd_is_an_error() {
        let (_case, ds) = case_data("sensemakr_darfur_bounds");
        for (kd, ky, what) in [
            (400.0, 400.0, "kd"),
            (0.0, 1.0, "kd"),
            (-1.0, 1.0, "kd"),
            (f64::NAN, 1.0, "kd"),
            (1.0, -1.0, "ky"),
            (1.0, f64::INFINITY, "ky"),
        ] {
            let e = run_sensemakr(
                &ds,
                "peacefactor",
                "directlyharmed",
                &COVS,
                Some(&["female"]),
                Some(kd),
                Some(ky),
                1.0,
                0.05,
            )
            .unwrap_err();
            match e {
                EconError::InvalidSpecification { message } => {
                    assert!(message.contains(what), "{kd} {ky}: {message}")
                }
                e => panic!("{kd} {ky}: {e:?}"),
            }
        }
        for (q, alpha, what) in [
            (1.0, 1.0, "alpha"),
            (1.0, 0.0, "alpha"),
            (1.0, f64::NAN, "alpha"),
            (-0.5, 0.05, "q"),
            (f64::NAN, 0.05, "q"),
            (f64::INFINITY, 0.05, "q"),
        ] {
            let e = run_sensemakr(
                &ds,
                "peacefactor",
                "directlyharmed",
                &COVS,
                Some(&["female"]),
                Some(1.0),
                Some(1.0),
                q,
                alpha,
            )
            .unwrap_err();
            match e {
                EconError::InvalidSpecification { message } => {
                    assert!(message.contains(what), "q {q}, alpha {alpha}: {message}")
                }
                e => panic!("q {q}, alpha {alpha}: {e:?}"),
            }
        }
        assert!(ovb_partial_r2_bound(0.3, 0.1, 3.0, 1.0).is_err());
        let (r2dz, r2yz, clamped) = ovb_partial_r2_bound(0.01, 0.1, 1.0, 1.0).unwrap();
        assert!(!clamped && r2dz > 0.01 && r2yz > 0.1);
        // r2dz.x = 0 leaves sensemakr's adjusted SE at se sqrt(1 - r2yz) sqrt(df / (df - 1))
        assert_eq!(
            adjusted_se(0.2, 0.36, 101.0),
            adjusted_se_ovb(0.2, 0.36, 0.0, 101.0)
        );
        let want = 0.2 * 0.8 * (101.0f64 / 100.0).sqrt();
        assert!((adjusted_se(0.2, 0.36, 101.0) - want).abs() <= 1e-15);
    }

    /// A benchmark covariate in the span of the others is an error naming it.
    #[test]
    fn test_sensemakr_benchmark_collinear_is_an_error() {
        let x = Array2::from_shape_fn((12, 3), |(i, j)| match j {
            0 => 1.0,
            1 => (i as f64 * 0.7).sin(),
            _ => 2.0 * (i as f64 * 0.7).sin() + 1.0,
        });
        let d = Array1::from_shape_fn(12, |i| (i as f64 * 1.3).cos());
        match partial_r2_covariate_treatment(&d, &x, 2, "z").unwrap_err() {
            EconError::SingularMatrix { context, .. } => {
                assert!(
                    context.contains("'z'") && context.contains("collinear"),
                    "{context}"
                )
            }
            e => panic!("{e:?}"),
        }
    }

    /// Test contour data generation.
    #[test]
    fn test_contour_data_generation() {
        // Create simple dataset
        let df = df! {
            "y" => [1.0, 2.1, 2.9, 4.2, 5.0, 5.8, 7.1, 8.0, 9.2, 10.1],
            "x" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
        .unwrap();

        let dataset = Dataset::new(df);
        let ols = run_ols(&dataset, "y", &["x"], true, CovarianceType::Standard).unwrap();

        let contour = generate_contour_data(&ols, "x", Some(10), Some(0.3)).unwrap();

        assert_eq!(contour.r2_yd_values.len(), 10);
        assert_eq!(contour.r2_xz_values.len(), 10);
        assert_eq!(contour.adjusted_estimates.len(), 10);
        assert_eq!(contour.adjusted_t_stats.len(), 10);

        // First point should be original estimate (no confounding)
        let original = ols.coefficients.iter().find(|c| c.name == "x").unwrap();
        assert!((contour.adjusted_estimates[0][0] - original.estimate).abs() < 1e-6);
    }

    /// q is printed as given (q = 0.5 printed as "0" before), and contour data on the result
    /// is announced in the text.
    #[test]
    fn test_display_prints_q_and_contour_data() {
        let df = df! {
            "y" => [1.0, 2.1, 2.9, 4.2, 5.0, 5.8, 7.1, 8.0, 9.2, 10.1],
            "x" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
        .unwrap();
        let ols = run_ols(
            &Dataset::new(df),
            "y",
            &["x"],
            true,
            CovarianceType::Standard,
        )
        .unwrap();
        let mut r = sensemakr(&ols, "x", None, None, None, 0.5, 0.05).unwrap();
        let out = r.to_string();
        assert!(out.contains("H0: q = 0.5, reduce = 50%"), "{out}");
        assert!(
            out.contains("RV(q=0.5):") && out.contains("RV(q=0.5, α=0.05):"),
            "{out}"
        );
        assert!(!out.contains("Contour data"), "{out}");
        let one = sensemakr(&ols, "x", None, None, None, 1.0, 0.05)
            .unwrap()
            .to_string();
        assert!(
            one.contains("H0: q = 1, reduce = 100%") && one.contains("RV(q=1):"),
            "{one}"
        );
        r.contour_data = Some(generate_contour_data(&ols, "x", Some(4), Some(0.3)).unwrap());
        for (g, m) in [(1, 0.5), (0, 0.5), (4, 0.0), (4, 1.0), (4, f64::NAN)] {
            assert!(matches!(
                generate_contour_data(&ols, "x", Some(g), Some(m)),
                Err(EconError::InvalidSpecification { .. })
            ));
        }
        let out = r.to_string();
        assert!(out.contains("Contour data: 4 x 4 grid"), "{out}");
    }

    /// REG-17 (sensemakr.rs): benchmark names over 20 bytes were cut at byte 17; '²' spans bytes 16..18.
    #[test]
    fn test_display_non_ascii_benchmark_name() {
        let n = 40;
        let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        let x1: Vec<f64> = (0..n)
            .map(|i| ((i * 37) % 11) as f64 / 10.0 - 0.5)
            .collect();
        let w: Vec<f64> = (0..n).map(|i| ((i * 53) % 13) as f64 / 12.0).collect();
        let e: Vec<f64> = (0..n)
            .map(|i| ((i * 71) % 17) as f64 / 17.0 - 0.5)
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 0.5 + 2.0 * t[i] + x1[i] + 0.5 * w[i] + 0.3 * e[i])
            .collect();
        let bench = "Wohnfläche_in_m²_netto"; // 24 bytes, 22 chars
        let df = df! { "y" => y, "treatment" => t, "x1" => x1, bench => w }.unwrap();
        let r = run_sensemakr(
            &Dataset::new(df),
            "y",
            "treatment",
            &["x1", bench],
            Some(&[bench]),
            Some(1.0),
            Some(1.0),
            1.0,
            0.05,
        )
        .unwrap();
        assert_eq!(r.benchmark_bounds.len(), 1);
        let out = r.to_string();
        assert!(out.contains("Wohnfläche_in_m²_... "), "{out}"); // 20 chars, {:>20}
    }

    /// A deterministic noisy series for the T3-PERF3 equivalence tests.
    fn perf3_noise(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect()
    }

    /// The benchmark partial R² by OLS's solve decision (one normal-equations inverse for
    /// both auxiliary regressions) equals the pivoted-QR value to 1e-10 relative.
    #[test]
    fn test_partial_r2_normal_equations_match_pivoted_qr() {
        for (n, k, seed) in [(30usize, 2usize, 1u64), (500, 4, 2), (2000, 3, 3)] {
            let e = perf3_noise(n * (k + 1), seed);
            let x = Array2::from_shape_fn((n, k), |(i, j)| {
                if j == 0 {
                    1.0
                } else {
                    (i as f64 * 0.01 * j as f64).cos() + e[j * n + i]
                }
            });
            let d = Array1::from_shape_fn(n, |i| 0.4 * x.row(i).sum() + e[i]);
            for jx in 1..k {
                let got = partial_r2_covariate_treatment(&d, &x, jx, "b").unwrap();
                let cols: Vec<usize> = (0..k).filter(|&c| c != jx).collect();
                let xm = Array2::from_shape_fn((n, k - 1), |(i, c)| x[[i, cols[c]]]);
                assert!(normal_equations_inverse(xm.view()).unwrap().is_some());
                let ry = least_squares(xm.view(), d.view()).unwrap().residuals;
                let rx = least_squares(xm.view(), x.column(jx)).unwrap().residuals;
                let (my, mx) = (ry.mean().unwrap(), rx.mean().unwrap());
                let cov: f64 = ry.iter().zip(&rx).map(|(a, b)| (a - my) * (b - mx)).sum();
                let vy: f64 = ry.iter().map(|a| (a - my).powi(2)).sum();
                let vx: f64 = rx.iter().map(|b| (b - mx).powi(2)).sum();
                let want = (cov / (vy.sqrt() * vx.sqrt())).powi(2);
                assert!(
                    (got - want).abs() <= 1e-10 * want,
                    "n {n} j {jx}: {got} vs {want}"
                );
            }
        }
    }

    /// Two other covariates equal up to noise far below lm.fit's tolerance leave X_{-j}
    /// to the pivoted QR, which finds it aliased: an error naming the benchmark.
    #[test]
    fn test_sensemakr_nearly_aliased_controls_reach_the_qr() {
        let n = 40;
        let e = perf3_noise(3 * n, 4);
        let x = Array2::from_shape_fn((n, 4), |(i, j)| match j {
            0 => 1.0,
            1 => i as f64 / 9.0 + e[i],
            2 => e[n + i],
            _ => 3.0 * (i as f64 / 9.0 + e[i]) + 1e-13 * e[2 * n + i],
        });
        let d = Array1::from_shape_fn(n, |i| x[[i, 1]] + e[2 * n + i]);
        match partial_r2_covariate_treatment(&d, &x, 2, "w").unwrap_err() {
            EconError::SingularMatrix { context, .. } => {
                assert!(context.contains("'w'"), "{context}")
            }
            e => panic!("{e:?}"),
        }
    }
}
