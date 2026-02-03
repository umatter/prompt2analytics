//! Identification diagnostics for matching and IPW treatment effect estimation.
//!
//! Checks for:
//! - Positivity/overlap violations (propensity scores near 0 or 1)
//! - Extreme IPW weights
//! - Residual covariate imbalance after matching

use crate::econometrics::{IpwResult, MatchResult, PropensityScoreSummary};

use super::{
    Assumption, AssumptionStatus, IdentificationReport, IdentificationWarning, WarningDiagnostics,
    WarningSeverity,
};

/// Threshold for considering propensity scores "extreme" (near 0 or 1).
const EXTREME_PS_THRESHOLD: f64 = 0.02;

/// Threshold for concerning fraction of observations with extreme weights.
const EXTREME_FRACTION_THRESHOLD: f64 = 0.05;

/// Threshold for maximum acceptable weight (relative to mean).
const MAX_WEIGHT_RATIO: f64 = 20.0;

/// Standardized mean difference threshold for balance.
const SMD_THRESHOLD: f64 = 0.1;

/// Generate identification diagnostics for a matching result.
///
/// # Diagnostics Performed
///
/// 1. **Covariate balance**: Checks if post-matching standardized mean
///    differences are below 0.1 for all covariates.
///
/// 2. **Common support**: Reports if any treated units remain unmatched
///    due to lack of suitable controls.
///
/// # Example
///
/// ```ignore
/// use p2a_core::econometrics::matching::run_matching;
/// use p2a_core::diagnostics::matching_diagnostics;
///
/// let result = run_matching(&dataset, "treated", &covariates, &method)?;
/// let report = matching_diagnostics(&result);
/// ```
pub fn matching_diagnostics(result: &MatchResult) -> IdentificationReport {
    let mut report = IdentificationReport::new("Propensity Score Matching");

    // Check covariate balance
    check_covariate_balance(result, &mut report);

    // Check common support / matching success
    check_common_support(result, &mut report);

    // Add untestable assumptions
    report.add_assumption(Assumption::untestable(
        "Unconfoundedness",
        "Treatment is independent of potential outcomes given observed covariates",
    ));

    report.add_assumption(Assumption::untestable(
        "SUTVA",
        "No spillovers between treated and control units",
    ));

    // Recommend sensitivity analysis
    report.add_warning(
        IdentificationWarning::new(
            "SENSITIVITY_RECOMMENDED",
            WarningSeverity::Info,
            "Consider Sensitivity Analysis",
            "Unconfoundedness cannot be tested from data. Consider sensitivity analysis \
             to assess how robust results are to unmeasured confounding.",
            "Unconfoundedness",
        )
        .with_remediation(vec![
            "Run sensemakr analysis for omitted variable bias bounds".to_string(),
            "Report E-values for unmeasured confounding".to_string(),
            "Discuss what unmeasured confounders could exist".to_string(),
        ]),
    );

    report
}

/// Check covariate balance after matching.
fn check_covariate_balance(result: &MatchResult, report: &mut IdentificationReport) {
    let balance = &result.balance_after;
    let n_imbalanced = balance.n_imbalanced;
    let mean_smd = balance.mean_abs_std_diff;

    // Determine balance status
    let status = if n_imbalanced == 0 {
        AssumptionStatus::NoViolation
    } else if n_imbalanced <= 2 && mean_smd < 0.15 {
        AssumptionStatus::PotentialViolation
    } else {
        AssumptionStatus::LikelyViolation
    };

    report.add_assumption(Assumption::testable(
        "Covariate balance",
        "Matched treated and control groups have similar covariate distributions",
        status,
    ));

    // Improvement from before to after
    let improvement = if result.balance_before.mean_abs_std_diff > 0.0 {
        1.0 - (balance.mean_abs_std_diff / result.balance_before.mean_abs_std_diff)
    } else {
        0.0
    };

    if n_imbalanced > 0 {
        let severity = if n_imbalanced >= 3 || mean_smd > 0.2 {
            WarningSeverity::Warning
        } else {
            WarningSeverity::Caution
        };

        // Find imbalanced variables
        let imbalanced_vars: Vec<_> = balance
            .covariates
            .iter()
            .filter(|c| c.std_diff.abs() > SMD_THRESHOLD)
            .map(|c| format!("{} (SMD={:.3})", c.name, c.std_diff))
            .take(5)
            .collect();

        let warning = IdentificationWarning::new(
            "COVARIATE_IMBALANCE",
            severity,
            "Residual Covariate Imbalance",
            format!(
                "{} covariate(s) have |SMD| > {:.1} after matching (mean |SMD| = {:.3}). \
                 Imbalanced: {}. Balance improved {:.0}% from pre-matching levels. \
                 Residual imbalance may bias treatment effect estimates.",
                n_imbalanced,
                SMD_THRESHOLD,
                mean_smd,
                imbalanced_vars.join(", "),
                improvement * 100.0
            ),
            "Covariate balance",
        )
        .with_diagnostics(
            WarningDiagnostics::new("Mean |SMD|", mean_smd).with_threshold(SMD_THRESHOLD, true),
        )
        .with_remediation(vec![
            "Re-specify the propensity score model".to_string(),
            "Try different matching methods (CEM, full matching)".to_string(),
            "Use doubly robust estimation to reduce bias".to_string(),
            "Include imbalanced covariates in outcome model".to_string(),
        ]);

        report.add_warning(warning);
    }
}

/// Check common support / matching success.
fn check_common_support(result: &MatchResult, report: &mut IdentificationReport) {
    let n_matched = result.n_matched_treated;
    let n_treated = result.n_treated;
    let match_rate = if n_treated > 0 {
        n_matched as f64 / n_treated as f64
    } else {
        1.0
    };

    // Determine support status
    let status = if match_rate >= 0.95 {
        AssumptionStatus::NoViolation
    } else if match_rate >= 0.80 {
        AssumptionStatus::PotentialViolation
    } else {
        AssumptionStatus::LikelyViolation
    };

    report.add_assumption(Assumption::testable(
        "Common support",
        "Suitable control units exist for all treated units",
        status,
    ));

    if match_rate < 0.95 {
        let severity = if match_rate < 0.80 {
            WarningSeverity::Warning
        } else {
            WarningSeverity::Caution
        };

        let warning = IdentificationWarning::new(
            "LIMITED_SUPPORT",
            severity,
            "Limited Common Support",
            format!(
                "Only {:.1}% of treated units ({}/{}) were successfully matched. \
                 Unmatched treated units have covariate values outside the control \
                 distribution. Treatment effects are only identified for matched units.",
                match_rate * 100.0,
                n_matched,
                n_treated
            ),
            "Common support",
        )
        .with_diagnostics(
            WarningDiagnostics::new("Match rate", match_rate).with_threshold(0.95, false),
        )
        .with_remediation(vec![
            "Relax caliper constraints if used".to_string(),
            "Consider alternative estimands (ATT on matched)".to_string(),
            "Examine which treated units lack support".to_string(),
            "Report results with and without off-support units".to_string(),
        ]);

        report.add_warning(warning);
    }
}

/// Generate identification diagnostics for an IPW treatment effect result.
///
/// # Diagnostics Performed
///
/// 1. **Positivity/Overlap**: Checks for propensity scores near 0 or 1,
///    which indicate positivity violations and extreme weights.
///
/// 2. **Extreme weights**: Checks if any observation has disproportionate
///    influence due to extreme propensity scores.
///
/// # Example
///
/// ```ignore
/// use p2a_core::econometrics::treatment::run_ipw;
/// use p2a_core::diagnostics::ipw_diagnostics;
///
/// let result = run_ipw(&dataset, "outcome", "treated", &covariates, &config)?;
/// let report = ipw_diagnostics(&result);
/// ```
pub fn ipw_diagnostics(result: &IpwResult) -> IdentificationReport {
    let mut report = IdentificationReport::new("Inverse Probability Weighting");

    // Check positivity using propensity score summary
    check_positivity(result, &mut report);

    // Add untestable assumptions
    report.add_assumption(Assumption::untestable(
        "Unconfoundedness",
        "Treatment is independent of potential outcomes given observed covariates",
    ));

    report.add_assumption(Assumption::untestable(
        "SUTVA",
        "No spillovers between treated and control units",
    ));

    // Recommend sensitivity analysis
    report.add_warning(
        IdentificationWarning::new(
            "SENSITIVITY_RECOMMENDED",
            WarningSeverity::Info,
            "Consider Sensitivity Analysis",
            "Unconfoundedness cannot be tested from data. Consider sensitivity analysis \
             to assess how robust results are to unmeasured confounding.",
            "Unconfoundedness",
        )
        .with_remediation(vec![
            "Run sensemakr analysis for omitted variable bias bounds".to_string(),
            "Report E-values for unmeasured confounding".to_string(),
            "Consider alternative estimators as robustness check".to_string(),
        ]),
    );

    report
}

/// Check positivity assumption using propensity score distribution.
fn check_positivity(result: &IpwResult, report: &mut IdentificationReport) {
    let ps_summary = &result.ps_summary;
    let ps_min = ps_summary.min;
    let ps_max = ps_summary.max;

    // Check for extreme propensity scores
    let has_extreme_low = ps_min < EXTREME_PS_THRESHOLD;
    let has_extreme_high = ps_max > (1.0 - EXTREME_PS_THRESHOLD);

    // Determine status
    let status = if !has_extreme_low && !has_extreme_high {
        AssumptionStatus::NoViolation
    } else if (has_extreme_low && ps_min > 0.01) || (has_extreme_high && ps_max < 0.99) {
        AssumptionStatus::PotentialViolation
    } else {
        AssumptionStatus::LikelyViolation
    };

    report.add_assumption(Assumption::testable(
        "Positivity (overlap)",
        "All covariate values have positive probability of treatment and control",
        status,
    ));

    if has_extreme_low || has_extreme_high {
        let severity = if ps_min < 0.01 || ps_max > 0.99 {
            WarningSeverity::Warning
        } else {
            WarningSeverity::Caution
        };

        let mut issue_parts = Vec::new();
        if has_extreme_low {
            issue_parts.push(format!("min = {:.4}", ps_min));
        }
        if has_extreme_high {
            issue_parts.push(format!("max = {:.4}", ps_max));
        }

        // Compute implied maximum weight (for ATE weighting)
        let max_weight = if ps_min > 0.0 && ps_max < 1.0 {
            f64::max(1.0 / ps_min, 1.0 / (1.0 - ps_max))
        } else {
            f64::INFINITY
        };

        let warning = IdentificationWarning::new(
            "POSITIVITY_VIOLATION",
            severity,
            "Extreme Propensity Scores Detected",
            format!(
                "Propensity scores approach boundary values ({}), indicating near-deterministic \
                 treatment assignment for some units. This implies extreme IPW weights \
                 (max ≈ {:.1}) where a few observations dominate the estimate. Trimmed \
                 propensity scores were used (threshold: {:.2}).",
                issue_parts.join(", "),
                max_weight,
                result.trim
            ),
            "Positivity",
        )
        .with_diagnostics(
            WarningDiagnostics::new("Min propensity score", ps_min)
                .with_threshold(EXTREME_PS_THRESHOLD, false),
        )
        .with_remediation(vec![
            "Increase propensity score trimming threshold".to_string(),
            "Use matching instead of weighting".to_string(),
            "Consider overlap weighting (Crump et al., 2009)".to_string(),
            "Report effective sample size".to_string(),
        ]);

        report.add_warning(warning);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::econometrics::{
        Estimand, MatchBalanceTable, MatchCovariateBalance, MatchMethod, PropensityScoreSummary,
    };
    use crate::traits::SignificanceLevel;

    fn create_mock_match_result(n_imbalanced: usize, match_rate: f64) -> MatchResult {
        let n_treated = 100;
        let n_matched = (n_treated as f64 * match_rate) as usize;

        let mut covariates = vec![
            MatchCovariateBalance {
                name: "age".to_string(),
                mean_treated: 45.0,
                mean_control: 44.5,
                std_diff: 0.05,
                var_ratio: 1.02,
                ks_statistic: 0.03,
            },
            MatchCovariateBalance {
                name: "income".to_string(),
                mean_treated: 50000.0,
                mean_control: 49000.0,
                std_diff: 0.08,
                var_ratio: 0.95,
                ks_statistic: 0.04,
            },
        ];

        // Add imbalanced covariates
        for i in 0..n_imbalanced {
            covariates.push(MatchCovariateBalance {
                name: format!("imbalanced_{}", i),
                mean_treated: 10.0,
                mean_control: 5.0,
                std_diff: 0.25, // Above threshold
                var_ratio: 1.5,
                ks_statistic: 0.15,
            });
        }

        let mean_smd: f64 =
            covariates.iter().map(|c| c.std_diff.abs()).sum::<f64>() / covariates.len() as f64;

        let max_smd: f64 = covariates
            .iter()
            .map(|c| c.std_diff.abs())
            .fold(0.0, f64::max);

        MatchResult {
            method: MatchMethod::NearestNeighbor {
                ratio: 1,
                caliper: None,
                replace: false,
            },
            distance: crate::econometrics::DistanceMethod::Logit,
            n_obs: 300,
            n_treated,
            n_control: 200,
            n_matched_treated: n_matched,
            n_matched_control: n_matched,
            n_discarded_treated: n_treated - n_matched,
            n_discarded_control: 0,
            balance_before: MatchBalanceTable {
                covariates: covariates
                    .iter()
                    .map(|c| MatchCovariateBalance {
                        std_diff: c.std_diff * 2.0,
                        ..c.clone()
                    })
                    .collect(),
                mean_abs_std_diff: mean_smd * 2.0,
                max_abs_std_diff: max_smd * 2.0,
                n_imbalanced: n_imbalanced + 2,
            },
            balance_after: MatchBalanceTable {
                covariates,
                mean_abs_std_diff: mean_smd,
                max_abs_std_diff: max_smd,
                n_imbalanced,
            },
            matches: vec![],
            weights: vec![1.0; 100],
            subclasses: None,
            caliper_used: None,
            propensity_scores: Some(vec![0.5; 100]),
            effective_sample_size: n_matched as f64,
        }
    }

    fn create_mock_ipw_result(ps_min: f64, ps_max: f64) -> IpwResult {
        IpwResult {
            estimand: Estimand::ATE,
            effect: 2.5,
            std_error: 0.5,
            ci_lower: 1.5,
            ci_upper: 3.5,
            t_stat: 5.0,
            p_value: 0.001,
            significance: SignificanceLevel::TenthPercent,
            n_obs: 1000,
            n_treated: 300,
            n_control: 700,
            n_trimmed: 0,
            ps_summary: PropensityScoreSummary {
                mean: 0.3,
                std_dev: 0.15,
                min: ps_min,
                max: ps_max,
                median: 0.28,
                p10: ps_min + 0.05,
                p90: ps_max - 0.05,
            },
            mean_y_treated: 10.0,
            mean_y_control: 7.5,
            normalized: true,
            trim: 0.01,
            bootstrap_reps: 100,
            warnings: vec![],
        }
    }

    #[test]
    fn test_good_balance() {
        let result = create_mock_match_result(0, 1.0);
        let report = matching_diagnostics(&result);

        // Should not have balance warning
        let balance_warning = report
            .warnings
            .iter()
            .find(|w| w.code == "COVARIATE_IMBALANCE");
        assert!(balance_warning.is_none());
    }

    #[test]
    fn test_poor_balance() {
        let result = create_mock_match_result(3, 1.0);
        let report = matching_diagnostics(&result);

        // Should have balance warning
        let balance_warning = report
            .warnings
            .iter()
            .find(|w| w.code == "COVARIATE_IMBALANCE");
        assert!(balance_warning.is_some());
    }

    #[test]
    fn test_limited_support() {
        let result = create_mock_match_result(0, 0.7);
        let report = matching_diagnostics(&result);

        // Should have support warning
        let support_warning = report.warnings.iter().find(|w| w.code == "LIMITED_SUPPORT");
        assert!(support_warning.is_some());
    }

    #[test]
    fn test_good_overlap() {
        let result = create_mock_ipw_result(0.1, 0.9);
        let report = ipw_diagnostics(&result);

        // Should not have positivity warning
        let pos_warning = report
            .warnings
            .iter()
            .find(|w| w.code == "POSITIVITY_VIOLATION");
        assert!(pos_warning.is_none());
    }

    #[test]
    fn test_positivity_violation() {
        let result = create_mock_ipw_result(0.005, 0.995);
        let report = ipw_diagnostics(&result);

        // Should have positivity warning
        let pos_warning = report
            .warnings
            .iter()
            .find(|w| w.code == "POSITIVITY_VIOLATION");
        assert!(pos_warning.is_some());
        assert!(pos_warning.unwrap().severity >= WarningSeverity::Warning);
    }

    #[test]
    fn test_sensitivity_recommendation() {
        let result = create_mock_match_result(0, 1.0);
        let report = matching_diagnostics(&result);

        // Should recommend sensitivity analysis
        let sens_warning = report
            .warnings
            .iter()
            .find(|w| w.code == "SENSITIVITY_RECOMMENDED");
        assert!(sens_warning.is_some());
    }
}
