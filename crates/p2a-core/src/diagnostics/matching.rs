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
/// Level of one tail of the propensity distribution: 0 none, 1 potential (within
/// [`EXTREME_PS_THRESHOLD`] of the boundary), 2 likely (within 0.01). `distance` is the
/// distance of the extreme score from its boundary (the minimum, or 1 - the maximum).
fn tail_level(distance: f64) -> u8 {
    if distance < 0.01 {
        2
    } else if distance < EXTREME_PS_THRESHOLD {
        1
    } else {
        0
    }
}

/// Positivity from the fitted propensity scores before trimming (`ps_summary_untrimmed`; the
/// kept units' summary for a result that does not carry it): the status is the worse of the
/// two tails.
fn check_positivity(result: &IpwResult, report: &mut IdentificationReport) {
    let ps_summary = result
        .ps_summary_untrimmed
        .as_ref()
        .unwrap_or(&result.ps_summary);
    let ps_min = ps_summary.min;
    let ps_max = ps_summary.max;

    let (low, high) = (tail_level(ps_min), tail_level(1.0 - ps_max));
    let has_extreme_low = low > 0;
    let has_extreme_high = high > 0;

    let status = match low.max(high) {
        0 => AssumptionStatus::NoViolation,
        1 => AssumptionStatus::PotentialViolation,
        _ => AssumptionStatus::LikelyViolation,
    };

    report.add_assumption(Assumption::testable(
        "Positivity (overlap)",
        "All covariate values have positive probability of treatment and control",
        status,
    ));

    if has_extreme_low || has_extreme_high {
        let severity = if low.max(high) == 2 {
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

        let n_all = result.n_obs + result.n_trimmed;
        let trimming = if result.trim > 0.0 {
            let share = 100.0 * result.n_trimmed as f64 / n_all.max(1) as f64;
            format!(
                "trimmed at {}: {} of {} units ({:.1}%) dropped",
                result.trim, result.n_trimmed, n_all, share
            )
        } else {
            "no trimming".to_string()
        };

        let warning = IdentificationWarning::new(
            "POSITIVITY_VIOLATION",
            severity,
            "Extreme Propensity Scores Detected",
            format!(
                "Propensity scores before trimming approach boundary values ({}), indicating \
                 near-deterministic treatment assignment for some units. This implies extreme \
                 IPW weights (max ≈ {:.1}) where a few observations dominate the estimate \
                 ({}).",
                issue_parts.join(", "),
                max_weight,
                trimming
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
                ecdf_mean: 0.03 / 2.0,
                ecdf_max: 0.03,
            },
            MatchCovariateBalance {
                name: "income".to_string(),
                mean_treated: 50000.0,
                mean_control: 49000.0,
                std_diff: 0.08,
                var_ratio: 0.95,
                ks_statistic: 0.04,
                ecdf_mean: 0.04 / 2.0,
                ecdf_max: 0.04,
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
                ecdf_mean: 0.15 / 2.0,
                ecdf_max: 0.15,
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
            propensity_convergence: None,
            warnings: vec![],
            m_order: Some(crate::econometrics::matching::MOrder::Largest),
            subclass: vec![],
            caliper_abs: None,
            total_distance: None,
            ess_treated: n_matched as f64,
            ess_control: n_matched as f64,
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
            ps_summary_untrimmed: None,
            mean_y_treated: 10.0,
            mean_y_control: 7.5,
            normalized: true,
            trim: 0.01,
            bootstrap_reps: 100,
            warnings: vec![],
            se_method: "M-estimation (propensity estimated)".to_string(),
            se_mest: None,
            se_naive: None,
            se_eif: None,
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

    fn positivity_status(report: &IdentificationReport) -> AssumptionStatus {
        report
            .assumptions
            .iter()
            .find(|a| a.name == "Positivity (overlap)")
            .expect("positivity assumption")
            .status
    }

    #[test]
    fn test_positivity_status_takes_the_worse_tail() {
        // Fails before: one badly extreme tail with a mildly extreme other tail was only a
        // potential violation.
        for (min, max, want) in [
            (0.005, 0.985, AssumptionStatus::LikelyViolation),
            (0.015, 0.6, AssumptionStatus::PotentialViolation),
            (0.4, 0.985, AssumptionStatus::PotentialViolation),
            (0.1, 0.9, AssumptionStatus::NoViolation),
        ] {
            let report = ipw_diagnostics(&create_mock_ipw_result(min, max));
            assert_eq!(positivity_status(&report), want, "({min}, {max})");
        }
    }

    /// Scores near 0 and 1 (logit slope 4 on a standard normal covariate), seeded.
    fn extreme_scores_dataset() -> crate::data::Dataset {
        use polars::prelude::*;
        use rand::{Rng, SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(20261111);
        let n = 400;
        let (mut y, mut d, mut x) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..n {
            let xi: f64 = rand_distr::Distribution::sample(&rand_distr::StandardNormal, &mut rng);
            let p = 1.0 / (1.0 + (-4.0 * xi).exp());
            let di = if rng.r#gen::<f64>() < p { 1.0 } else { 0.0 };
            let noise: f64 =
                rand_distr::Distribution::sample(&rand_distr::StandardNormal, &mut rng);
            x.push(xi);
            d.push(di);
            y.push(1.0 + 0.5 * di + xi + noise);
        }
        crate::data::Dataset::new(df! { "y" => y, "d" => d, "x" => x }.unwrap())
    }

    #[test]
    fn test_positivity_uses_untrimmed_scores() {
        // Fails before: the diagnostics read the trimmed scores, which the default trim
        // (0.05) keeps away from the 0.02 threshold, so the check could never fire.
        use crate::econometrics::{IpwConfig, run_ipw_treatment};
        let ds = extreme_scores_dataset();
        let r = run_ipw_treatment(&ds, "y", "d", &["x"], IpwConfig::default()).unwrap();
        assert!(
            r.ps_summary.min >= 0.05,
            "precondition: the kept scores are trimmed"
        );
        let untrimmed = r.ps_summary_untrimmed.as_ref().unwrap();
        assert!(untrimmed.min < 0.01 || untrimmed.max > 0.99);
        let report = ipw_diagnostics(&r);
        let w = report
            .warnings
            .iter()
            .find(|w| w.code == "POSITIVITY_VIOLATION")
            .expect("positivity warning");
        assert!(w.message.contains("before trimming"), "{}", w.message);
        assert_eq!(
            positivity_status(&report),
            AssumptionStatus::LikelyViolation
        );
    }

    #[test]
    fn test_positivity_reports_trimmed_share() {
        use crate::econometrics::{IpwConfig, run_ipw_treatment};
        let ds = extreme_scores_dataset();
        let r = run_ipw_treatment(&ds, "y", "d", &["x"], IpwConfig::default()).unwrap();
        let n_all = r.n_obs + r.n_trimmed;
        assert!(
            r.n_trimmed * 10 > n_all,
            "precondition: more than 10% trimmed"
        );
        let report = ipw_diagnostics(&r);
        let msg = &report
            .warnings
            .iter()
            .find(|w| w.code == "POSITIVITY_VIOLATION")
            .unwrap()
            .message;
        let share = 100.0 * r.n_trimmed as f64 / n_all as f64;
        let want = format!(
            "trimmed at 0.05: {} of {} units ({:.1}%) dropped",
            r.n_trimmed, n_all, share
        );
        assert!(msg.contains(&want), "{msg}");
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
