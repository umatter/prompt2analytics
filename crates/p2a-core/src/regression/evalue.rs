//! E-value sensitivity analysis for unmeasured confounding.
//!
//! This module implements E-value calculations to assess the robustness of
//! causal effect estimates to unmeasured confounding. The E-value quantifies
//! the minimum strength of association that an unmeasured confounder would
//! need to have with both the treatment and outcome to fully explain away
//! an observed treatment-outcome association.
//!
//! # Mathematical Background
//!
//! ## E-value for Risk Ratio
//!
//! For an observed risk ratio RR >= 1, the E-value is:
//!
//! E-value = RR + sqrt(RR * (RR - 1))
//!
//! For RR < 1, first compute 1/RR and then apply the formula.
//!
//! ## E-value for Odds Ratio
//!
//! For rare outcomes (< 15% prevalence), OR approximates RR, so use the RR formula.
//!
//! For common outcomes, first apply the square root transformation to convert
//! OR to approximate RR:
//!
//! RR_approx = sqrt(OR)
//!
//! Then apply the E-value formula.
//!
//! ## E-value for Hazard Ratio
//!
//! For rare outcomes, HR approximates RR, so use the RR formula directly.
//!
//! For common outcomes, use VanderWeele's (2020) conversion:
//!
//! RR_approx = (1 - 0.5^sqrt(HR)) / (1 - 0.5^sqrt(1/HR))
//!
//! (VanderWeele, T. J. (2020). Optimal approximate conversions of odds ratios and hazard
//! ratios to risk ratios. Biometrics 76(3), 746-752; `EValue::evalues.HR(rare = FALSE)`.)
//!
//! ## E-value for Standardized Mean Difference
//!
//! Convert SMD to approximate RR using the Chinn (2000) conversion:
//!
//! RR_approx = exp(0.91 * d)
//!
//! where d is the standardized mean difference. Then apply the E-value formula.
//!
//! # Interpretation
//!
//! A large E-value implies that considerable unmeasured confounding would be
//! needed to explain away an effect estimate. A small E-value implies that
//! little unmeasured confounding would be needed.
//!
//! # References
//!
//! - VanderWeele, T. J., & Ding, P. (2017). "Sensitivity Analysis in Observational
//!   Research: Introducing the E-Value". Annals of Internal Medicine, 167(4), 268-274.
//!   <https://doi.org/10.7326/M16-2607>
//!
//! - VanderWeele, T. J. (2017). "On a Square-Root Transformation of the Odds Ratio
//!   for a Common Outcome". Epidemiology, 28(6), e58-e59.
//!   <https://doi.org/10.1097/EDE.0000000000000733>
//!
//! - Chinn, S. (2000). "A simple method for converting an odds ratio to effect size
//!   for use in meta-analysis". Statistics in Medicine, 19(22), 3127-3131.
//!   <https://doi.org/10.1002/1097-0258(20001130)19:22%3C3127::AID-SIM784%3E3.0.CO;2-M>
//!
//! - Linden, A., Mathur, M. B., & VanderWeele, T. J. (2020). "Conducting sensitivity
//!   analysis for unmeasured confounding in observational studies using E-values:
//!   The evalue package". The Stata Journal, 20(1), 162-175.
//!   <https://doi.org/10.1177/1536867X20909696>
//!
//! - R package EValue: <https://CRAN.R-project.org/package=EValue>
//!
//! R equivalent: `EValue::evalue()`

use serde::{Deserialize, Serialize};

use crate::errors::{EconError, EconResult};

/// Result of E-value calculation for a single effect measure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EValueResult {
    /// The type of effect measure used
    pub effect_type: EffectType,

    /// Original point estimate (RR, OR, HR, or SMD)
    pub point_estimate: f64,

    /// Lower bound of confidence interval (optional)
    pub ci_lower: Option<f64>,

    /// Upper bound of confidence interval (optional)
    pub ci_upper: Option<f64>,

    /// Risk ratio used for E-value calculation (after any transformation)
    pub risk_ratio: f64,

    /// E-value for the point estimate
    pub evalue_point: f64,

    /// E-value for the confidence interval limit closest to null
    /// (lower limit if RR > 1, upper limit if RR < 1)
    pub evalue_ci: Option<f64>,

    /// Whether the rare outcome approximation was used (for OR/HR)
    pub rare_outcome: Option<bool>,
}

impl std::fmt::Display for EValueResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "E-Value Sensitivity Analysis")?;
        writeln!(f, "============================")?;
        writeln!(f)?;
        writeln!(f, "Effect Type: {}", self.effect_type)?;
        writeln!(f, "Point Estimate: {:.4}", self.point_estimate)?;

        if let (Some(lo), Some(hi)) = (self.ci_lower, self.ci_upper) {
            writeln!(f, "95% CI: [{:.4}, {:.4}]", lo, hi)?;
        }

        writeln!(f)?;

        if (self.risk_ratio - self.point_estimate).abs() > 1e-10 {
            writeln!(f, "Risk Ratio (transformed): {:.4}", self.risk_ratio)?;
        }

        writeln!(f)?;
        writeln!(f, "E-value (point): {:.2}", self.evalue_point)?;

        if let Some(ev_ci) = self.evalue_ci {
            writeln!(f, "E-value (CI limit): {:.2}", ev_ci)?;
        }

        writeln!(f)?;
        writeln!(f, "Interpretation:")?;
        writeln!(
            f,
            "  An unmeasured confounder associated with both treatment and outcome"
        )?;
        writeln!(
            f,
            "  by a risk ratio of {:.2}-fold each (above and beyond measured",
            self.evalue_point
        )?;
        writeln!(
            f,
            "  confounders) could explain away the observed effect, but weaker"
        )?;
        writeln!(f, "  confounding could not.")?;

        if let Some(ev_ci) = self.evalue_ci {
            writeln!(f)?;
            writeln!(
                f,
                "  Confounding of strength {:.2} could shift the CI to include the null.",
                ev_ci
            )?;
        }

        if let Some(rare) = self.rare_outcome {
            writeln!(f)?;
            if rare {
                writeln!(
                    f,
                    "  Note: Rare outcome approximation was used ({} approximates RR).",
                    self.effect_type
                )?;
            } else {
                let conversion = match self.effect_type {
                    EffectType::HazardRatio => "(1 - 0.5^sqrt(HR)) / (1 - 0.5^sqrt(1/HR))",
                    _ => "sqrt(OR)",
                };
                writeln!(
                    f,
                    "  Note: Common outcome conversion {} was applied to {}.",
                    conversion, self.effect_type
                )?;
            }
        }

        Ok(())
    }
}

/// Type of effect measure for E-value calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EffectType {
    /// Risk Ratio (RR)
    RiskRatio,
    /// Odds Ratio (OR)
    OddsRatio,
    /// Hazard Ratio (HR)
    HazardRatio,
    /// Standardized Mean Difference (SMD)
    SMD,
    /// Risk Difference (RD)
    RiskDifference,
}

impl std::fmt::Display for EffectType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EffectType::RiskRatio => write!(f, "Risk Ratio (RR)"),
            EffectType::OddsRatio => write!(f, "Odds Ratio (OR)"),
            EffectType::HazardRatio => write!(f, "Hazard Ratio (HR)"),
            EffectType::SMD => write!(f, "Standardized Mean Difference (SMD)"),
            EffectType::RiskDifference => write!(f, "Risk Difference (RD)"),
        }
    }
}

/// Compute E-value from a risk ratio.
///
/// For RR >= 1:
///   E-value = RR + sqrt(RR * (RR - 1))
///
/// For RR < 1:
///   E-value = (1/RR) + sqrt((1/RR) * ((1/RR) - 1))
///
/// # Arguments
/// * `rr` - The risk ratio (must be positive)
///
/// # Returns
/// The E-value, which is always >= 1.
///
/// # References
/// VanderWeele & Ding (2017), Equation (1)
///
/// # Example
/// ```
/// use p2a_core::regression::evalue_rr;
///
/// let e = evalue_rr(3.9);
/// assert!((e - 7.26).abs() < 0.01);
/// ```
pub fn evalue_rr(rr: f64) -> f64 {
    if rr <= 0.0 || rr.is_nan() {
        return f64::NAN;
    }

    // For RR = 1, E-value = 1 (no confounding needed)
    if (rr - 1.0).abs() < 1e-10 {
        return 1.0;
    }

    // If RR < 1, use 1/RR
    // VanderWeele & Ding (2017): "For a risk ratio less than 1, one first
    // takes the inverse of the observed risk ratio and then applies the formula"
    let rr_use = if rr < 1.0 { 1.0 / rr } else { rr };

    // E-value = RR + sqrt(RR * (RR - 1))
    rr_use + (rr_use * (rr_use - 1.0)).sqrt()
}

/// Compute E-value for a risk ratio with confidence interval.
///
/// Returns E-values for both the point estimate and the confidence limit
/// closest to the null (RR = 1).
///
/// # Arguments
/// * `point` - Point estimate of the risk ratio
/// * `ci_lower` - Lower bound of confidence interval (optional)
/// * `ci_upper` - Upper bound of confidence interval (optional)
///
/// # Returns
/// An `EValueResult` containing E-values for point and CI.
///
/// # Example
/// ```
/// use p2a_core::regression::evalue_rr_ci;
///
/// let result = evalue_rr_ci(2.5, Some(1.8), Some(3.5)).unwrap();
/// println!("E-value (point): {:.2}", result.evalue_point);
/// println!("E-value (CI): {:.2}", result.evalue_ci.unwrap());
/// ```
pub fn evalue_rr_ci(
    point: f64,
    ci_lower: Option<f64>,
    ci_upper: Option<f64>,
) -> EconResult<EValueResult> {
    if point <= 0.0 {
        return Err(EconError::InvalidSpecification {
            message: format!("Risk ratio must be positive, got {}", point),
        });
    }

    if let Some(lo) = ci_lower {
        if lo <= 0.0 {
            return Err(EconError::InvalidSpecification {
                message: format!("CI lower bound must be positive, got {}", lo),
            });
        }
    }

    if let Some(hi) = ci_upper {
        if hi <= 0.0 {
            return Err(EconError::InvalidSpecification {
                message: format!("CI upper bound must be positive, got {}", hi),
            });
        }
    }

    let evalue_point = evalue_rr(point);

    // E-value for CI: use the limit closest to null (RR = 1)
    let evalue_ci = match (ci_lower, ci_upper) {
        (Some(lo), Some(hi)) => {
            // Determine which limit is closer to 1
            let limit_to_use = if point >= 1.0 {
                // If RR >= 1, lower limit is closest to null
                // If lower limit < 1, the CI includes null, so E-value is 1
                if lo < 1.0 { 1.0 } else { lo }
            } else {
                // If RR < 1, upper limit is closest to null
                // If upper limit > 1, the CI includes null, so E-value is 1
                if hi > 1.0 { 1.0 } else { hi }
            };
            Some(evalue_rr(limit_to_use))
        }
        (Some(lo), None) => {
            if point >= 1.0 {
                Some(evalue_rr(if lo < 1.0 { 1.0 } else { lo }))
            } else {
                None
            }
        }
        (None, Some(hi)) => {
            if point < 1.0 {
                Some(evalue_rr(if hi > 1.0 { 1.0 } else { hi }))
            } else {
                None
            }
        }
        (None, None) => None,
    };

    Ok(EValueResult {
        effect_type: EffectType::RiskRatio,
        point_estimate: point,
        ci_lower,
        ci_upper,
        risk_ratio: point,
        evalue_point,
        evalue_ci,
        rare_outcome: None,
    })
}

/// Compute E-value from an odds ratio.
///
/// For rare outcomes (< 15% prevalence), OR approximates RR.
/// For common outcomes, apply square root transformation first.
///
/// # Arguments
/// * `point` - Point estimate of the odds ratio
/// * `ci_lower` - Lower bound of confidence interval (optional)
/// * `ci_upper` - Upper bound of confidence interval (optional)
/// * `rare` - Whether the outcome is rare (< 15% prevalence).
///            If true, OR is used directly as RR approximation.
///            If false, sqrt(OR) is used as RR approximation.
///
/// # Returns
/// An `EValueResult` containing E-values for point and CI.
///
/// # References
/// - VanderWeele & Ding (2017), Section on odds ratios
/// - VanderWeele (2017), "On a Square-Root Transformation of the Odds Ratio"
///
/// # Example
/// ```
/// use p2a_core::regression::evalue_or;
///
/// // Rare outcome (OR approximates RR)
/// let result = evalue_or(2.5, None, None, true).unwrap();
///
/// // Common outcome (apply sqrt transformation)
/// let result = evalue_or(4.0, Some(2.0), Some(8.0), false).unwrap();
/// ```
pub fn evalue_or(
    point: f64,
    ci_lower: Option<f64>,
    ci_upper: Option<f64>,
    rare: bool,
) -> EconResult<EValueResult> {
    if point <= 0.0 {
        return Err(EconError::InvalidSpecification {
            message: format!("Odds ratio must be positive, got {}", point),
        });
    }

    // Transform OR to approximate RR
    // VanderWeele (2017): For common outcomes, RR_approx = sqrt(OR)
    let transform = |or: f64| -> f64 {
        if rare {
            or // Rare outcome: OR approximates RR
        } else {
            or.sqrt() // Common outcome: apply square root transformation
        }
    };

    let rr_point = transform(point);
    let rr_lower = ci_lower.map(transform);
    let rr_upper = ci_upper.map(transform);

    let evalue_point = evalue_rr(rr_point);

    // E-value for CI
    let evalue_ci = match (rr_lower, rr_upper) {
        (Some(lo), Some(hi)) => {
            let limit_to_use = if rr_point >= 1.0 {
                if lo < 1.0 { 1.0 } else { lo }
            } else if hi > 1.0 {
                1.0
            } else {
                hi
            };
            Some(evalue_rr(limit_to_use))
        }
        (Some(lo), None) => {
            if rr_point >= 1.0 {
                Some(evalue_rr(if lo < 1.0 { 1.0 } else { lo }))
            } else {
                None
            }
        }
        (None, Some(hi)) => {
            if rr_point < 1.0 {
                Some(evalue_rr(if hi > 1.0 { 1.0 } else { hi }))
            } else {
                None
            }
        }
        (None, None) => None,
    };

    Ok(EValueResult {
        effect_type: EffectType::OddsRatio,
        point_estimate: point,
        ci_lower,
        ci_upper,
        risk_ratio: rr_point,
        evalue_point,
        evalue_ci,
        rare_outcome: Some(rare),
    })
}

/// Compute E-value from a hazard ratio.
///
/// For rare outcomes (< 15% event rate), HR approximates RR.
/// For common outcomes, HR is converted with VanderWeele's (2020) formula
/// RR = (1 - 0.5^sqrt(HR)) / (1 - 0.5^sqrt(1/HR)) first, as `EValue::evalues.HR(rare = FALSE)`
/// (Biometrics 76(3)).
///
/// # Arguments
/// * `point` - Point estimate of the hazard ratio
/// * `ci_lower` - Lower bound of confidence interval (optional)
/// * `ci_upper` - Upper bound of confidence interval (optional)
/// * `rare` - Whether the outcome is rare (< 15% event rate).
///            If true, HR is used directly as RR approximation.
///            If false, (1 - 0.5^sqrt(HR)) / (1 - 0.5^sqrt(1/HR)) is used.
///
/// # Returns
/// An `EValueResult` containing E-values for point and CI.
///
/// # Example
/// ```
/// use p2a_core::regression::evalue_hr;
///
/// let result = evalue_hr(1.5, Some(1.2), Some(1.9), true).unwrap();
/// ```
pub fn evalue_hr(
    point: f64,
    ci_lower: Option<f64>,
    ci_upper: Option<f64>,
    rare: bool,
) -> EconResult<EValueResult> {
    if point <= 0.0 {
        return Err(EconError::InvalidSpecification {
            message: format!("Hazard ratio must be positive, got {}", point),
        });
    }

    let transform = |hr: f64| -> f64 { if rare { hr } else { hr_common_to_rr(hr) } };

    let rr_point = transform(point);
    let rr_lower = ci_lower.map(transform);
    let rr_upper = ci_upper.map(transform);

    let evalue_point = evalue_rr(rr_point);

    let evalue_ci = match (rr_lower, rr_upper) {
        (Some(lo), Some(hi)) => {
            let limit_to_use = if rr_point >= 1.0 {
                if lo < 1.0 { 1.0 } else { lo }
            } else if hi > 1.0 {
                1.0
            } else {
                hi
            };
            Some(evalue_rr(limit_to_use))
        }
        (Some(lo), None) => {
            if rr_point >= 1.0 {
                Some(evalue_rr(if lo < 1.0 { 1.0 } else { lo }))
            } else {
                None
            }
        }
        (None, Some(hi)) => {
            if rr_point < 1.0 {
                Some(evalue_rr(if hi > 1.0 { 1.0 } else { hi }))
            } else {
                None
            }
        }
        (None, None) => None,
    };

    Ok(EValueResult {
        effect_type: EffectType::HazardRatio,
        point_estimate: point,
        ci_lower,
        ci_upper,
        risk_ratio: rr_point,
        evalue_point,
        evalue_ci,
        rare_outcome: Some(rare),
    })
}

/// Compute E-value from a standardized mean difference (SMD).
///
/// The SMD is first converted to an approximate risk ratio using the
/// Chinn (2000) conversion:
///
/// RR_approx = exp(0.91 * d)
///
/// where d is the SMD (e.g., Cohen's d, Hedges' g).
///
/// # Arguments
/// * `smd` - Standardized mean difference (Cohen's d, Hedges' g, etc.)
/// * `se` - Standard error of the SMD (optional, for confidence interval)
///
/// # Returns
/// An `EValueResult` containing E-values for point and CI.
///
/// # References
/// - Chinn, S. (2000). "A simple method for converting an odds ratio to
///   effect size for use in meta-analysis". Statistics in Medicine.
/// - VanderWeele & Ding (2017), Section on continuous outcomes.
///
/// # Example
/// ```
/// use p2a_core::regression::evalue_smd;
///
/// // SMD of 0.5 with standard error 0.1
/// let result = evalue_smd(0.5, Some(0.1)).unwrap();
/// ```
pub fn evalue_smd(smd: f64, se: Option<f64>) -> EconResult<EValueResult> {
    // Check for valid SE if provided
    if let Some(s) = se {
        if s < 0.0 {
            return Err(EconError::InvalidSpecification {
                message: format!("Standard error must be non-negative, got {}", s),
            });
        }
    }

    // Convert SMD to approximate RR using Chinn (2000) formula
    // RR_approx = exp(0.91 * d)
    // Reference: Chinn (2000), Statistics in Medicine, Eq. (7)
    const CHINN_CONSTANT: f64 = 0.91;

    let rr_point = (CHINN_CONSTANT * smd).exp();

    // If SE provided: the 95% CI of the SMD, and the RR limits as EValue's evalues.MD
    // computes them, exp(0.91 d -/+ 1.78 se) (VanderWeele and Ding's approximation for the
    // RR scale, not 0.91 times the SMD limits)
    const EVALUE_MD_SE_MULTIPLIER: f64 = 1.78;
    let (ci_lower, ci_upper, rr_lower, rr_upper) = if let Some(s) = se {
        let z = crate::traits::estimator::z_critical(0.95)?;
        let smd_lo = smd - z * s;
        let smd_hi = smd + z * s;
        let rr_lo = (CHINN_CONSTANT * smd - EVALUE_MD_SE_MULTIPLIER * s).exp();
        let rr_hi = (CHINN_CONSTANT * smd + EVALUE_MD_SE_MULTIPLIER * s).exp();
        (Some(smd_lo), Some(smd_hi), Some(rr_lo), Some(rr_hi))
    } else {
        (None, None, None, None)
    };

    let evalue_point = evalue_rr(rr_point);

    // E-value for CI
    let evalue_ci = match (rr_lower, rr_upper) {
        (Some(lo), Some(hi)) => {
            let limit_to_use = if rr_point >= 1.0 {
                if lo < 1.0 { 1.0 } else { lo }
            } else if hi > 1.0 {
                1.0
            } else {
                hi
            };
            Some(evalue_rr(limit_to_use))
        }
        _ => None,
    };

    Ok(EValueResult {
        effect_type: EffectType::SMD,
        point_estimate: smd,
        ci_lower,
        ci_upper,
        risk_ratio: rr_point,
        evalue_point,
        evalue_ci,
        rare_outcome: None,
    })
}

/// Compute E-value from a risk difference.
///
/// This requires knowing the baseline risk (risk in the unexposed group)
/// to convert to a risk ratio.
///
/// # Arguments
/// * `rd` - Risk difference (absolute risk in exposed - absolute risk in unexposed)
/// * `baseline_risk` - Baseline risk (risk in unexposed group), between 0 and 1
/// * `se` - Standard error of the risk difference (optional)
///
/// # Returns
/// An `EValueResult` containing E-values for point and CI.
///
/// # Example
/// ```
/// use p2a_core::regression::evalue_rd;
///
/// // Risk difference of 0.1 (10% increase) with baseline risk of 0.2 (20%)
/// let result = evalue_rd(0.1, 0.2, None).unwrap();
/// ```
pub fn evalue_rd(rd: f64, baseline_risk: f64, se: Option<f64>) -> EconResult<EValueResult> {
    if baseline_risk <= 0.0 || baseline_risk >= 1.0 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Baseline risk must be between 0 and 1 (exclusive), got {}",
                baseline_risk
            ),
        });
    }

    // Risk in exposed = baseline_risk + rd
    let exposed_risk = baseline_risk + rd;

    if exposed_risk <= 0.0 || exposed_risk >= 1.0 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Implied risk in exposed group ({}) is outside valid range (0, 1). \
                 Check your risk difference ({}) and baseline risk ({}).",
                exposed_risk, rd, baseline_risk
            ),
        });
    }

    // Convert to risk ratio
    let rr = exposed_risk / baseline_risk;

    // Compute E-value
    let evalue_point = evalue_rr(rr);

    // For CI, we need to propagate uncertainty through the transformation
    // This is approximate; exact propagation requires more information
    let (ci_lower, ci_upper, evalue_ci) = if let Some(s) = se {
        // EValue's evalues.RD: qnorm(1 - alpha / 2) at alpha = 0.05
        let z = crate::traits::estimator::z_critical(0.95)?;
        let rd_lo = rd - z * s;
        let rd_hi = rd + z * s;

        // Convert RD CI to RR CI
        let exposed_lo = (baseline_risk + rd_lo).clamp(0.001, 0.999);
        let exposed_hi = (baseline_risk + rd_hi).clamp(0.001, 0.999);
        let rr_lo = exposed_lo / baseline_risk;
        let rr_hi = exposed_hi / baseline_risk;

        let limit_to_use = if rr >= 1.0 {
            if rr_lo < 1.0 { 1.0 } else { rr_lo }
        } else if rr_hi > 1.0 {
            1.0
        } else {
            rr_hi
        };

        (Some(rd_lo), Some(rd_hi), Some(evalue_rr(limit_to_use)))
    } else {
        (None, None, None)
    };

    Ok(EValueResult {
        effect_type: EffectType::RiskDifference,
        point_estimate: rd,
        ci_lower,
        ci_upper,
        risk_ratio: rr,
        evalue_point,
        evalue_ci,
        rare_outcome: None,
    })
}

/// VanderWeele's (2020) conversion of a hazard ratio with a common outcome to a risk ratio,
/// (1 - 0.5^sqrt(HR)) / (1 - 0.5^sqrt(1/HR)); 1 at HR = 1, where the formula is 0/0.
fn hr_common_to_rr(hr: f64) -> f64 {
    if hr == 1.0 {
        return 1.0;
    }
    (1.0 - 0.5f64.powf(hr.sqrt())) / (1.0 - 0.5f64.powf((1.0 / hr).sqrt()))
}

/// The bounding factor of a confounder whose two associations both equal `evalue`.
///
/// With RR_EU = RR_UD = e, the joint bounding factor of VanderWeele & Ding (2017) is
/// B(e, e) = e² / (2e - 1). At the E-value of an observed risk ratio RR this is RR itself,
/// so `bias_factor(evalue_rr(rr)) == rr` (up to rounding): the E-value is the smallest
/// common strength of both associations that explains the observed effect away.
///
/// # Arguments
/// * `evalue` - The common strength of the two associations (risk-ratio scale), e.g. an
///   E-value
///
/// # Returns
/// The bounding factor B(e, e); 1 for e <= 1 or NaN.
///
/// # References
/// VanderWeele & Ding (2017), Section on the bounding factor
pub fn bias_factor(evalue: f64) -> f64 {
    if evalue <= 1.0 || evalue.is_nan() {
        return 1.0;
    }
    evalue * evalue / (2.0 * evalue - 1.0)
}

/// Compute E-value from bounding factor.
///
/// If an unmeasured confounder U has associations RR_EU with exposure E
/// and RR_UD with disease D (outcome), the joint bias factor is:
///
/// B = RR_EU * RR_UD / (RR_EU + RR_UD - 1)
///
/// The E-value is the minimum joint bias factor assuming RR_EU = RR_UD.
///
/// # Arguments
/// * `rr_eu` - Risk ratio for U -> E (confounder -> exposure)
/// * `rr_ud` - Risk ratio for U -> D (confounder -> disease/outcome)
///
/// # Returns
/// The joint bounding factor B.
///
/// # References
/// VanderWeele & Ding (2017), Equation (2)
pub fn bounding_factor(rr_eu: f64, rr_ud: f64) -> f64 {
    if rr_eu <= 0.0 || rr_ud <= 0.0 {
        return f64::NAN;
    }

    // B = RR_EU * RR_UD / (RR_EU + RR_UD - 1)
    // This is the maximum bias from a confounder with these associations
    let denom = rr_eu + rr_ud - 1.0;
    if denom <= 0.0 {
        return f64::NAN; // Impossible confounder structure
    }

    (rr_eu * rr_ud) / denom
}

/// Check if a hypothesized confounder could explain away an observed effect.
///
/// The confounder explains the effect away when its joint bounding factor
/// B(RR_EU, RR_UD) = RR_EU RR_UD / (RR_EU + RR_UD - 1) reaches the observed risk ratio,
/// taken away from the null (RR or 1/RR, whichever is at least 1).
///
/// # Arguments
/// * `observed_rr` - The observed effect on the risk-ratio scale. For an odds ratio, hazard
///   ratio, SMD or risk difference pass `EValueResult::risk_ratio`, the converted estimate
/// * `rr_eu` - Hypothesized RR for confounder -> exposure
/// * `rr_ud` - Hypothesized RR for confounder -> outcome
///
/// # Returns
/// `true` if the hypothesized confounder could fully explain the effect; `false` for a
/// non-finite or non-positive input. A relative allowance of 1e-12 lets a confounder at
/// exactly the E-value (B(E, E) = RR in exact arithmetic) explain the effect away despite
/// rounding.
///
/// # Example
/// ```
/// use p2a_core::regression::{evalue_rr, could_explain_away};
///
/// let rr = 3.9; // E-value 7.26
///
/// // RR = 3 for both associations: B = 9/5 = 1.8 < 3.9, so no
/// assert!(!could_explain_away(rr, 3.0, 3.0));
///
/// // RR = 8 for both: B = 64/15 = 4.27 >= 3.9, so yes
/// assert!(could_explain_away(rr, 8.0, 8.0));
///
/// // Both associations at the E-value explain the effect away exactly
/// let e = evalue_rr(rr);
/// assert!(could_explain_away(rr, e, e));
/// ```
pub fn could_explain_away(observed_rr: f64, rr_eu: f64, rr_ud: f64) -> bool {
    let finite_positive = |v: f64| v.is_finite() && v > 0.0;
    if !(finite_positive(observed_rr) && finite_positive(rr_eu) && finite_positive(rr_ud)) {
        return false;
    }
    let rr = observed_rr.max(1.0 / observed_rr);
    let b = bounding_factor(rr_eu, rr_ud);
    b >= rr * (1.0 - 1e-12)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test basic E-value calculation for risk ratio.
    /// Reference: VanderWeele & Ding (2017) example
    #[test]
    fn test_evalue_rr_basic() {
        // Example from paper: RR = 3.9 -> E-value = 7.26
        let ev = evalue_rr(3.9);
        assert!(
            (ev - 7.26).abs() < 0.01,
            "E-value for RR=3.9 should be approximately 7.26, got {}",
            ev
        );

        // RR = 1 -> E-value = 1 (no confounding needed)
        let ev_null = evalue_rr(1.0);
        assert!(
            (ev_null - 1.0).abs() < 1e-10,
            "E-value for RR=1 should be 1, got {}",
            ev_null
        );

        // RR = 2 -> E-value = 2 + sqrt(2*1) = 2 + 1.41 = 3.41
        let ev_2 = evalue_rr(2.0);
        assert!(
            (ev_2 - 3.41).abs() < 0.01,
            "E-value for RR=2 should be approximately 3.41, got {}",
            ev_2
        );
    }

    /// Test E-value for RR < 1 (protective effect).
    #[test]
    fn test_evalue_rr_protective() {
        // RR = 0.5 should give same E-value as RR = 2
        let ev_05 = evalue_rr(0.5);
        let ev_20 = evalue_rr(2.0);
        assert!(
            (ev_05 - ev_20).abs() < 1e-10,
            "E-value for RR=0.5 should equal E-value for RR=2, got {} vs {}",
            ev_05,
            ev_20
        );

        // RR = 0.25 should give same as RR = 4
        let ev_025 = evalue_rr(0.25);
        let ev_40 = evalue_rr(4.0);
        assert!(
            (ev_025 - ev_40).abs() < 1e-10,
            "E-value for RR=0.25 should equal E-value for RR=4"
        );
    }

    /// Test E-value with confidence interval.
    #[test]
    fn test_evalue_rr_ci() {
        let result = evalue_rr_ci(3.9, Some(2.5), Some(6.0)).unwrap();

        assert_eq!(result.effect_type, EffectType::RiskRatio);
        assert!((result.point_estimate - 3.9).abs() < 1e-10);
        assert!((result.evalue_point - 7.26).abs() < 0.01);

        // E-value for CI should be based on lower limit (2.5) since RR > 1
        // E-value for RR=2.5 = 2.5 + sqrt(2.5*1.5) = 2.5 + 1.94 = 4.44
        let evalue_ci = result.evalue_ci.unwrap();
        assert!(
            (evalue_ci - 4.44).abs() < 0.01,
            "E-value for CI should be approximately 4.44, got {}",
            evalue_ci
        );
    }

    /// Test E-value for CI that includes null.
    #[test]
    fn test_evalue_ci_includes_null() {
        // RR = 2.0 with CI (0.8, 5.0) - includes null
        let result = evalue_rr_ci(2.0, Some(0.8), Some(5.0)).unwrap();

        // Since CI includes null, E-value for CI should be 1
        let evalue_ci = result.evalue_ci.unwrap();
        assert!(
            (evalue_ci - 1.0).abs() < 1e-10,
            "E-value for CI including null should be 1, got {}",
            evalue_ci
        );
    }

    /// Test E-value for odds ratio with rare outcome.
    #[test]
    fn test_evalue_or_rare() {
        // For rare outcomes, OR approximates RR
        let result = evalue_or(2.5, None, None, true).unwrap();

        assert_eq!(result.effect_type, EffectType::OddsRatio);
        assert!((result.risk_ratio - 2.5).abs() < 1e-10);
        assert!(result.rare_outcome == Some(true));

        // E-value should match RR = 2.5
        let expected_ev = evalue_rr(2.5);
        assert!((result.evalue_point - expected_ev).abs() < 1e-10);
    }

    /// Test E-value for odds ratio with common outcome.
    #[test]
    fn test_evalue_or_common() {
        // For common outcomes, use sqrt transformation
        // OR = 4 -> RR_approx = sqrt(4) = 2
        let result = evalue_or(4.0, None, None, false).unwrap();

        assert_eq!(result.effect_type, EffectType::OddsRatio);
        assert!((result.risk_ratio - 2.0).abs() < 1e-10);
        assert!(result.rare_outcome == Some(false));

        // E-value should match RR = 2
        let expected_ev = evalue_rr(2.0);
        assert!((result.evalue_point - expected_ev).abs() < 1e-10);
    }

    /// Test E-value for hazard ratio.
    #[test]
    fn test_evalue_hr() {
        // Rare outcome
        let result_rare = evalue_hr(1.5, Some(1.2), Some(1.9), true).unwrap();
        assert_eq!(result_rare.effect_type, EffectType::HazardRatio);
        assert!((result_rare.risk_ratio - 1.5).abs() < 1e-10);

        // Common outcome: VanderWeele (2020), (1 - 0.5^2) / (1 - 0.5^0.5)
        let result_common = evalue_hr(4.0, None, None, false).unwrap();
        assert!((result_common.risk_ratio - 2.5606601717798219).abs() < 1e-12);
        assert!(result_common.to_string().contains("0.5^sqrt(HR)"));
    }

    /// Test E-value for standardized mean difference.
    #[test]
    fn test_evalue_smd() {
        // SMD = 0.5 -> RR_approx = exp(0.91 * 0.5) = exp(0.455) = 1.576
        let result = evalue_smd(0.5, None).unwrap();

        assert_eq!(result.effect_type, EffectType::SMD);
        assert!((result.point_estimate - 0.5).abs() < 1e-10);

        let expected_rr = (0.91 * 0.5_f64).exp();
        assert!(
            (result.risk_ratio - expected_rr).abs() < 1e-10,
            "Expected RR {}, got {}",
            expected_rr,
            result.risk_ratio
        );

        let expected_ev = evalue_rr(expected_rr);
        assert!((result.evalue_point - expected_ev).abs() < 1e-10);
    }

    /// Test E-value for SMD with standard error.
    #[test]
    fn test_evalue_smd_with_se() {
        let result = evalue_smd(0.5, Some(0.1)).unwrap();

        // Should have CI computed
        assert!(result.ci_lower.is_some());
        assert!(result.ci_upper.is_some());
        assert!(result.evalue_ci.is_some());

        // CI should be approximately (0.5 - 1.96*0.1, 0.5 + 1.96*0.1) = (0.304, 0.696)
        let ci_lo = result.ci_lower.unwrap();
        let ci_hi = result.ci_upper.unwrap();
        assert!((ci_lo - 0.304).abs() < 0.01);
        assert!((ci_hi - 0.696).abs() < 0.01);
    }

    /// Test E-value for risk difference.
    #[test]
    fn test_evalue_rd() {
        // RD = 0.1 with baseline risk 0.2
        // Risk in exposed = 0.3, RR = 0.3/0.2 = 1.5
        let result = evalue_rd(0.1, 0.2, None).unwrap();

        assert_eq!(result.effect_type, EffectType::RiskDifference);
        assert!((result.risk_ratio - 1.5).abs() < 1e-10);
    }

    /// Test bounding factor calculation.
    #[test]
    fn test_bounding_factor() {
        // If RR_EU = RR_UD = 2, B = 2*2/(2+2-1) = 4/3 = 1.33
        let b = bounding_factor(2.0, 2.0);
        assert!((b - 1.333).abs() < 0.01);

        // If RR_EU = RR_UD = 4, B = 16/7 = 2.29
        let b2 = bounding_factor(4.0, 4.0);
        assert!((b2 - 2.29).abs() < 0.01);
    }

    /// could_explain_away compares the bounding factor with the observed risk ratio.
    #[test]
    fn test_could_explain_away() {
        // B(3, 3) = 9/5 = 1.8 < 3.9: cannot explain it away
        assert!(!could_explain_away(3.9, 3.0, 3.0));
        // B(8, 8) = 64/15 = 4.27 >= 3.9, although 8 is below the E-value 7.26
        assert!(could_explain_away(3.9, 8.0, 8.0));
        // invalid inputs
        assert!(!could_explain_away(f64::NAN, 8.0, 8.0));
        assert!(!could_explain_away(0.0, 8.0, 8.0));
        assert!(!could_explain_away(3.9, f64::INFINITY, 8.0));
        assert!(!could_explain_away(3.9, -1.0, 8.0));
    }

    /// VanderWeele & Ding (2017): a confounder with both associations equal to the E-value
    /// explains the observed RR away exactly, B(E, E) = RR.
    #[test]
    fn test_could_explain_away_at_the_evalue() {
        let rr = 3.9;
        let e = evalue_rr(rr);
        assert!(could_explain_away(rr, e, e));
        assert!(could_explain_away(rr, 8.0, 8.0));
        assert!(!could_explain_away(rr, 3.0, 3.0));
        assert!(could_explain_away(rr, 15.0, 15.0));
        // protective effect: the RR scale is |log RR|
        let e6 = evalue_rr(0.6);
        assert!(could_explain_away(0.6, e6, e6));
        assert!(!could_explain_away(0.6, 1.5, 1.5));
        for x in [1.2, 2.5, 3.9] {
            let b = bias_factor(evalue_rr(x));
            assert!((b - x).abs() <= 1e-12 * x, "{x}: {b}");
        }
        assert_eq!(bias_factor(1.0), 1.0);
        assert_eq!(bias_factor(0.5), 1.0);
    }

    /// Test edge cases.
    #[test]
    fn test_edge_cases() {
        // Invalid inputs
        assert!(evalue_rr(0.0).is_nan());
        assert!(evalue_rr(-1.0).is_nan());

        assert!(evalue_rr_ci(-1.0, None, None).is_err());
        assert!(evalue_or(0.0, None, None, true).is_err());
        assert!(evalue_smd(0.5, Some(-0.1)).is_err());
    }

    /// Case `evalue_formulas`: EValue 4.1.4 evalues.RR / OR / HR / MD. Closed forms, so the
    /// tolerance is 1e-12 relative plus 1e-14 absolute.
    #[test]
    fn test_validate_evalue_formulas() {
        use p2a_validation::Check;
        let (case, _ds) = crate::test_support::load_case("evalue_formulas");
        let tol = p2a_validation::Tol::new(1e-12, 1e-14);
        let rows = [
            (1, evalue_rr_ci(2.5, Some(1.8), Some(3.5)).unwrap()),
            (2, evalue_rr_ci(0.6, Some(0.45), Some(0.8)).unwrap()),
            // CI crosses 1: E-value 1
            (3, evalue_rr_ci(2.0, Some(0.9), Some(3.0)).unwrap()),
            (4, evalue_or(2.5, Some(1.5), Some(4.0), true).unwrap()),
            // sqrt(OR)
            (5, evalue_or(4.0, Some(2.0), Some(8.0), false).unwrap()),
            (6, evalue_hr(1.5, Some(1.2), Some(1.9), true).unwrap()),
            // common outcome: VanderWeele (2020)
            (7, evalue_hr(4.0, Some(2.0), Some(8.0), false).unwrap()),
        ];
        let mut c = Check::new(&case);
        for (k, r) in &rows {
            c.scalar(&format!("row{k}_rr"), r.risk_ratio, tol)
                .scalar(&format!("row{k}_e_point"), r.evalue_point, tol)
                .scalar(&format!("row{k}_e_ci"), r.evalue_ci.unwrap(), tol);
        }
        let smd = evalue_smd(0.5, Some(0.1)).unwrap();
        c.scalar("row8_rr", smd.risk_ratio, tol)
            .scalar("row8_e_point", smd.evalue_point, tol)
            .scalar("row8_e_ci", smd.evalue_ci.unwrap(), tol)
            .scalar("evalue_rr_3_9", evalue_rr(3.9), tol)
            // EValueResult carries no RR limits: pin them through the conversion
            .vector(
                "row7_rr_ci",
                &[hr_common_to_rr(2.0), hr_common_to_rr(8.0)],
                tol,
            )
            .finish();
        // REG-39: could_explain_away is a property (test_could_explain_away_at_the_evalue).
    }

    /// Case `evalue_hr_common`: EValue 4.1.4 `evalues.HR(est, lo, hi, rare = FALSE)`
    /// (inputs as `evalue_hr_inputs.csv`; hr3 has no lower limit). Closed forms.
    #[test]
    fn test_validate_evalue_hr_common() {
        use p2a_validation::Check;
        let (case, _ds) = crate::test_support::load_case("evalue_hr_common");
        let tol = p2a_validation::Tol::new(1e-12, 1e-14);
        let rows: [(usize, f64, Option<f64>, Option<f64>); 3] = [
            (1, 0.6, Some(0.4), Some(0.9)),
            (2, 1.3, Some(0.9), Some(1.8)),
            (3, 0.5, None, Some(0.8)),
        ];
        let mut c = Check::new(&case);
        for (k, est, lo, hi) in rows {
            let r = evalue_hr(est, lo, hi, false).unwrap();
            let rr_ci: Vec<f64> = [lo, hi]
                .into_iter()
                .flatten()
                .map(hr_common_to_rr)
                .collect();
            c.scalar(&format!("hr{k}_rr"), r.risk_ratio, tol)
                .scalar(&format!("hr{k}_e_point"), r.evalue_point, tol)
                .scalar(&format!("hr{k}_e_ci"), r.evalue_ci.unwrap(), tol)
                .vector(&format!("hr{k}_rr_ci"), &rr_ci, tol);
        }
        c.finish();
        assert_eq!(hr_common_to_rr(1.0), 1.0);
    }

    /// Test Display implementation.
    #[test]
    fn test_display() {
        let result = evalue_rr_ci(3.9, Some(2.5), Some(6.0)).unwrap();
        let output = format!("{}", result);

        assert!(output.contains("E-Value Sensitivity Analysis"));
        assert!(output.contains("Risk Ratio"));
        assert!(output.contains("3.9"));
        assert!(output.contains("7.2")); // E-value point
        assert!(output.contains("Interpretation"));
    }
}
