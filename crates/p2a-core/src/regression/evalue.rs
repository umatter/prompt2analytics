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
//! For common outcomes, apply the square root transformation:
//!
//! RR_approx = sqrt(HR)
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
//!   https://doi.org/10.7326/M16-2607
//!
//! - VanderWeele, T. J. (2017). "On a Square-Root Transformation of the Odds Ratio
//!   for a Common Outcome". Epidemiology, 28(6), e58-e59.
//!   https://doi.org/10.1097/EDE.0000000000000733
//!
//! - Chinn, S. (2000). "A simple method for converting an odds ratio to effect size
//!   for use in meta-analysis". Statistics in Medicine, 19(22), 3127-3131.
//!   https://doi.org/10.1002/1097-0258(20001130)19:22<3127::AID-SIM784>3.0.CO;2-M
//!
//! - Linden, A., Mathur, M. B., & VanderWeele, T. J. (2020). "Conducting sensitivity
//!   analysis for unmeasured confounding in observational studies using E-values:
//!   The evalue package". The Stata Journal, 20(1), 162-175.
//!   https://doi.org/10.1177/1536867X20909696
//!
//! - R package EValue: https://CRAN.R-project.org/package=EValue
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
                writeln!(
                    f,
                    "  Note: Common outcome transformation (sqrt) was applied to {}.",
                    self.effect_type
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
                if lo < 1.0 {
                    1.0
                } else {
                    lo
                }
            } else {
                // If RR < 1, upper limit is closest to null
                // If upper limit > 1, the CI includes null, so E-value is 1
                if hi > 1.0 {
                    1.0
                } else {
                    hi
                }
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
                if lo < 1.0 {
                    1.0
                } else {
                    lo
                }
            } else {
                if hi > 1.0 {
                    1.0
                } else {
                    hi
                }
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
/// For common outcomes, apply square root transformation first.
///
/// # Arguments
/// * `point` - Point estimate of the hazard ratio
/// * `ci_lower` - Lower bound of confidence interval (optional)
/// * `ci_upper` - Upper bound of confidence interval (optional)
/// * `rare` - Whether the outcome is rare (< 15% event rate).
///            If true, HR is used directly as RR approximation.
///            If false, sqrt(HR) is used as RR approximation.
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

    // Same transformation logic as OR
    let transform = |hr: f64| -> f64 {
        if rare {
            hr
        } else {
            hr.sqrt()
        }
    };

    let rr_point = transform(point);
    let rr_lower = ci_lower.map(transform);
    let rr_upper = ci_upper.map(transform);

    let evalue_point = evalue_rr(rr_point);

    let evalue_ci = match (rr_lower, rr_upper) {
        (Some(lo), Some(hi)) => {
            let limit_to_use = if rr_point >= 1.0 {
                if lo < 1.0 {
                    1.0
                } else {
                    lo
                }
            } else {
                if hi > 1.0 {
                    1.0
                } else {
                    hi
                }
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

    // If SE provided, compute CI for the RR
    // Using delta method: CI for SMD is (d - 1.96*se, d + 1.96*se)
    // Then transform to RR scale
    let (ci_lower, ci_upper, rr_lower, rr_upper) = if let Some(s) = se {
        let z = 1.96; // For 95% CI
        let smd_lo = smd - z * s;
        let smd_hi = smd + z * s;
        let rr_lo = (CHINN_CONSTANT * smd_lo).exp();
        let rr_hi = (CHINN_CONSTANT * smd_hi).exp();
        (Some(smd_lo), Some(smd_hi), Some(rr_lo), Some(rr_hi))
    } else {
        (None, None, None, None)
    };

    let evalue_point = evalue_rr(rr_point);

    // E-value for CI
    let evalue_ci = match (rr_lower, rr_upper) {
        (Some(lo), Some(hi)) => {
            let limit_to_use = if rr_point >= 1.0 {
                if lo < 1.0 {
                    1.0
                } else {
                    lo
                }
            } else {
                if hi > 1.0 {
                    1.0
                } else {
                    hi
                }
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
        let z = 1.96;
        let rd_lo = rd - z * s;
        let rd_hi = rd + z * s;

        // Convert RD CI to RR CI
        let exposed_lo = (baseline_risk + rd_lo).clamp(0.001, 0.999);
        let exposed_hi = (baseline_risk + rd_hi).clamp(0.001, 0.999);
        let rr_lo = exposed_lo / baseline_risk;
        let rr_hi = exposed_hi / baseline_risk;

        let limit_to_use = if rr >= 1.0 {
            if rr_lo < 1.0 {
                1.0
            } else {
                rr_lo
            }
        } else {
            if rr_hi > 1.0 {
                1.0
            } else {
                rr_hi
            }
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

/// Compute the minimum bias factor required to explain away an observed effect.
///
/// The bias factor B is defined such that B >= E-value would suffice to
/// explain away the effect. This is the square of the E-value minus 1.
///
/// B = sqrt(E-value) for confounding that affects both treatment and outcome
///
/// # Arguments
/// * `evalue` - The E-value
///
/// # Returns
/// The bias factor B, representing the minimum strength of unmeasured
/// confounding (as a risk ratio) needed with BOTH treatment and outcome.
///
/// # References
/// VanderWeele & Ding (2017), Section on the bounding factor
pub fn bias_factor(evalue: f64) -> f64 {
    if evalue <= 1.0 || evalue.is_nan() {
        return 1.0;
    }
    // The E-value is the product of two bias factors (one for U->D, one for U->Y)
    // If they are equal, each is sqrt(E-value)
    evalue.sqrt()
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
/// Given an observed effect (as E-value) and hypothesized confounder
/// associations with exposure and outcome, determine if the confounder
/// could fully explain the effect.
///
/// # Arguments
/// * `observed_evalue` - E-value for the observed effect
/// * `rr_eu` - Hypothesized RR for confounder -> exposure
/// * `rr_ud` - Hypothesized RR for confounder -> outcome
///
/// # Returns
/// `true` if the hypothesized confounder could fully explain the effect.
///
/// # Example
/// ```
/// use p2a_core::regression::{evalue_rr, could_explain_away};
///
/// let ev = evalue_rr(3.9);  // E-value = 7.26
///
/// // Could a confounder with RR=3 for both associations explain this?
/// assert!(!could_explain_away(ev, 3.0, 3.0));
///
/// // Could a confounder with RR=8 for both associations explain this?
/// assert!(could_explain_away(ev, 8.0, 8.0));
/// ```
pub fn could_explain_away(observed_evalue: f64, rr_eu: f64, rr_ud: f64) -> bool {
    let b = bounding_factor(rr_eu, rr_ud);
    b >= observed_evalue
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

        // Common outcome
        let result_common = evalue_hr(4.0, None, None, false).unwrap();
        assert!((result_common.risk_ratio - 2.0).abs() < 1e-10);
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

    /// Test could_explain_away function.
    #[test]
    fn test_could_explain_away() {
        let ev = evalue_rr(3.9); // E-value = 7.26

        // Weak confounder cannot explain
        assert!(!could_explain_away(ev, 3.0, 3.0));

        // Strong confounder can explain
        assert!(could_explain_away(ev, 10.0, 10.0));
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

    /// Validate against R EValue package.
    /// R code:
    /// ```r
    /// library(EValue)
    /// evalue(RR = 2.5)
    /// # E-value: 4.44
    ///
    /// evalue(RR = 2.5, lo = 1.8, hi = 3.5)
    /// # Point E-value: 4.44
    /// # CI E-value: 2.99
    ///
    /// evalues.OR(2.5, lo = 1.5, hi = 4.2, rare = FALSE)
    /// # Uses sqrt transformation
    ///
    /// evalues.SMD(0.5, se = 0.1)
    /// ```
    #[test]
    fn test_validate_against_r() {
        // Test 1: Simple RR
        let ev = evalue_rr(2.5);
        assert!(
            (ev - 4.44).abs() < 0.01,
            "R evalue(RR=2.5) = 4.44, got {}",
            ev
        );

        // Test 2: RR with CI
        let result = evalue_rr_ci(2.5, Some(1.8), Some(3.5)).unwrap();
        assert!(
            (result.evalue_point - 4.44).abs() < 0.01,
            "R point E-value = 4.44, got {}",
            result.evalue_point
        );
        // E-value for lower CI limit (1.8)
        // evalue_rr(1.8) = 1.8 + sqrt(1.8*0.8) = 1.8 + 1.2 = 3.0
        let ev_ci = result.evalue_ci.unwrap();
        assert!(
            (ev_ci - 3.0).abs() < 0.02,
            "R CI E-value approx 3.0, got {}",
            ev_ci
        );

        // Test 3: OR with common outcome
        // OR = 4 -> sqrt(4) = 2 -> E-value = 2 + sqrt(2*1) = 3.41
        let or_result = evalue_or(4.0, None, None, false).unwrap();
        assert!(
            (or_result.evalue_point - 3.41).abs() < 0.01,
            "R evalues.OR(4, rare=FALSE) approx 3.41, got {}",
            or_result.evalue_point
        );

        // Test 4: SMD
        // SMD = 0.5 -> exp(0.91*0.5) = 1.576 -> E-value = 1.576 + sqrt(1.576*0.576) = 2.53
        let smd_result = evalue_smd(0.5, None).unwrap();
        assert!(
            (smd_result.evalue_point - 2.53).abs() < 0.05,
            "R evalues.SMD(0.5) approx 2.53, got {}",
            smd_result.evalue_point
        );
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
