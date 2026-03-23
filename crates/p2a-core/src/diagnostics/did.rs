//! Identification diagnostics for difference-in-differences estimation.
//!
//! Checks for:
//! - Parallel pre-trends violations (staggered DiD)
//! - Anticipation effects (significant pre-treatment coefficients)

use crate::econometrics::{DiDResult, StaggeredDidResult};

use super::{
    Assumption, AssumptionStatus, IdentificationReport, IdentificationWarning, WarningDiagnostics,
    WarningSeverity,
};

/// Generate identification diagnostics for a canonical 2x2 DiD result.
///
/// For the simple 2x2 DiD design, we cannot directly test parallel trends
/// from the data (only 2 time periods). The function documents assumptions
/// but cannot provide data-driven warnings.
///
/// # Example
///
/// ```ignore
/// use p2a_core::econometrics::did::run_did;
/// use p2a_core::diagnostics::did_diagnostics;
///
/// let did_result = run_did(&dataset, "outcome", "treated", "post", None, None)?;
/// let report = did_diagnostics(&did_result);
/// ```
pub fn did_diagnostics(result: &DiDResult) -> IdentificationReport {
    let mut report = IdentificationReport::new("Difference-in-Differences (2x2)");

    // For 2x2 DiD, parallel trends is not directly testable
    report.add_assumption(Assumption::untestable(
        "Parallel trends",
        "Treated and control groups would follow same trajectory absent treatment",
    ));

    report.add_assumption(Assumption::untestable(
        "No anticipation",
        "Treatment effect does not occur before treatment implementation",
    ));

    report.add_assumption(Assumption::untestable(
        "SUTVA",
        "No spillovers between treated and control units",
    ));

    report.add_assumption(Assumption::untestable(
        "Stable composition",
        "Group composition does not change differentially over time",
    ));

    // Add informational note about limitations
    report.add_warning(
        IdentificationWarning::new(
            "PARALLEL_TRENDS_UNTESTABLE",
            WarningSeverity::Info,
            "Parallel Trends Cannot Be Tested",
            format!(
                "The 2x2 DiD design with {} observations has only two time periods, \
                 so parallel pre-trends cannot be tested. Consider whether the \
                 parallel trends assumption is plausible based on domain knowledge \
                 or pre-treatment data from other sources.",
                result.n_obs
            ),
            "Parallel trends",
        )
        .with_remediation(vec![
            "Examine pre-treatment trends using additional historical data".to_string(),
            "Consider synthetic control methods if pre-treatment periods exist".to_string(),
            "Discuss institutional reasons why parallel trends may hold".to_string(),
        ]),
    );

    report
}

/// Generate identification diagnostics for a staggered DiD result.
///
/// For staggered adoption designs with multiple time periods, this function
/// checks the pre-trend test to assess the parallel trends assumption.
///
/// # Diagnostics Performed
///
/// 1. **Parallel pre-trends test**: If pre-treatment periods exist, tests
///    whether pre-treatment ATT estimates are jointly zero. Rejection suggests
///    treated and control groups were on different trajectories before treatment.
///
/// 2. **Anticipation detection**: Checks for significant individual pre-treatment
///    coefficients that might indicate anticipation effects.
///
/// # Example
///
/// ```ignore
/// use p2a_core::econometrics::staggered_did::run_staggered_did;
/// use p2a_core::diagnostics::staggered_did_diagnostics;
///
/// let result = run_staggered_did(&dataset, &config)?;
/// let report = staggered_did_diagnostics(&result);
///
/// if report.has_warnings_at_level(WarningSeverity::Warning) {
///     println!("Caution: {}", report.summary());
/// }
/// ```
pub fn staggered_did_diagnostics(result: &StaggeredDidResult) -> IdentificationReport {
    let mut report = IdentificationReport::new("Staggered Difference-in-Differences");

    // Check parallel trends using pre-trend test
    check_parallel_trends(result, &mut report);

    // Check for anticipation effects
    check_anticipation(result, &mut report);

    // Add standard assumptions
    report.add_assumption(Assumption::untestable(
        "SUTVA",
        "No spillovers between treated and control units",
    ));

    report.add_assumption(Assumption::untestable(
        "Common support",
        "Comparison groups exist for all treated cohorts",
    ));

    report
}

/// Check parallel trends assumption using pre-trend test.
fn check_parallel_trends(result: &StaggeredDidResult, report: &mut IdentificationReport) {
    match &result.pretrend_test {
        Some(test) => {
            // Determine status based on p-value
            let status = if test.p_value >= 0.10 {
                AssumptionStatus::NoViolation
            } else if test.p_value >= 0.05 {
                AssumptionStatus::PotentialViolation
            } else {
                AssumptionStatus::LikelyViolation
            };

            report.add_assumption(Assumption::testable(
                "Parallel pre-trends",
                "Pre-treatment effects are jointly zero",
                status,
            ));

            // Generate warning if pre-trends test rejects
            if test.p_value < 0.05 {
                let severity = if test.p_value < 0.01 {
                    WarningSeverity::Warning
                } else {
                    WarningSeverity::Caution
                };

                let warning = IdentificationWarning::new(
                    "PARALLEL_TRENDS_VIOLATED",
                    severity,
                    "Pre-Trends Differ Significantly",
                    format!(
                        "The joint test of pre-treatment effects rejects parallel trends \
                         (χ² = {:.2}, df = {}, p = {:.4}). This suggests treated and control \
                         groups were on different trajectories before treatment, which may \
                         bias the estimated treatment effect. Examine the event study plot \
                         to assess the magnitude and pattern of pre-trend divergence.",
                        test.chi2, test.df, test.p_value
                    ),
                    "Parallel trends",
                )
                .with_diagnostics(
                    WarningDiagnostics::new("Pre-trend χ²", test.chi2)
                        .with_threshold(test.p_value, true),
                )
                .with_remediation(vec![
                    "Examine event study plot for divergence patterns".to_string(),
                    "Consider matching on pre-treatment outcomes".to_string(),
                    "Use synthetic control methods as alternative".to_string(),
                    "Report sensitivity to trend adjustments".to_string(),
                ]);

                report.add_warning(warning);
            }
        }
        None => {
            // No pre-treatment periods available
            report.add_assumption(Assumption::untestable(
                "Parallel trends",
                "No pre-treatment periods available for testing",
            ));

            report.add_warning(
                IdentificationWarning::new(
                    "NO_PRETREND_PERIODS",
                    WarningSeverity::Caution,
                    "No Pre-Treatment Periods for Testing",
                    "No pre-treatment periods are available to test parallel trends. \
                     The identifying assumption cannot be validated from the data.",
                    "Parallel trends",
                )
                .with_remediation(vec![
                    "Seek additional pre-treatment data if possible".to_string(),
                    "Justify parallel trends based on institutional knowledge".to_string(),
                ]),
            );
        }
    }
}

/// Check for anticipation effects in pre-treatment periods.
fn check_anticipation(result: &StaggeredDidResult, report: &mut IdentificationReport) {
    // Count significant pre-treatment effects (CI excludes 0)
    let significant_pre: Vec<_> = result
        .group_time_atts
        .iter()
        .filter(|att| !att.post_treatment && (att.ci_lower > 0.0 || att.ci_upper < 0.0))
        .collect();

    if significant_pre.is_empty() {
        report.add_assumption(Assumption::testable(
            "No anticipation",
            "No significant pre-treatment effects detected",
            AssumptionStatus::NoViolation,
        ));
    } else {
        report.add_assumption(Assumption::testable(
            "No anticipation",
            "Significant pre-treatment effects detected",
            AssumptionStatus::PotentialViolation,
        ));

        // Find the earliest significant pre-treatment effect
        let earliest = significant_pre
            .iter()
            .map(|att| att.time - att.group)
            .min()
            .unwrap_or(0);

        let warning = IdentificationWarning::new(
            "ANTICIPATION_DETECTED",
            WarningSeverity::Caution,
            "Possible Anticipation Effects",
            format!(
                "{} pre-treatment period(s) show significant effects (p < 0.05). \
                 Effects begin {} period(s) before treatment. This may indicate \
                 that units anticipated treatment and changed behavior, or that \
                 the treatment timing is misspecified.",
                significant_pre.len(),
                -earliest
            ),
            "No anticipation",
        )
        .with_remediation(vec![
            "Verify treatment timing is correctly coded".to_string(),
            "Consider an earlier effective treatment date".to_string(),
            "Allow for anticipation in the estimator configuration".to_string(),
            "Discuss why anticipation may or may not be present".to_string(),
        ]);

        report.add_warning(warning);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::econometrics::{AggregatedEffect, GroupTimeATT, PreTrendTest, StaggeredDidConfig};

    fn create_mock_staggered_result(
        pretrend_p_value: Option<f64>,
        has_significant_pretrends: bool,
    ) -> StaggeredDidResult {
        let mut group_time_atts = vec![
            // Post-treatment effects
            GroupTimeATT {
                group: 2020,
                time: 2020,
                att: 2.5,
                std_error: 0.5,
                ci_lower: 1.5,
                ci_upper: 3.5,
                n_treated: 100,
                n_comparison: 200,
                post_treatment: true,
                relative_time: 0,
            },
            GroupTimeATT {
                group: 2020,
                time: 2021,
                att: 3.0,
                std_error: 0.6,
                ci_lower: 1.8,
                ci_upper: 4.2,
                n_treated: 100,
                n_comparison: 200,
                post_treatment: true,
                relative_time: 1,
            },
        ];

        // Pre-treatment effects
        if has_significant_pretrends {
            // Significant: CI excludes zero (both bounds positive or both negative)
            group_time_atts.push(GroupTimeATT {
                group: 2020,
                time: 2019,
                att: 1.2,
                std_error: 0.3,
                ci_lower: 0.6, // CI excludes 0, indicating significant
                ci_upper: 1.8,
                n_treated: 100,
                n_comparison: 200,
                post_treatment: false,
                relative_time: -1,
            });
        } else {
            // Not significant: CI includes zero
            group_time_atts.push(GroupTimeATT {
                group: 2020,
                time: 2019,
                att: 0.1,
                std_error: 0.3,
                ci_lower: -0.5, // CI includes 0, not significant
                ci_upper: 0.7,
                n_treated: 100,
                n_comparison: 200,
                post_treatment: false,
                relative_time: -1,
            });
        }

        let pretrend_test = pretrend_p_value.map(|p| PreTrendTest {
            chi2: 5.0,
            df: 2,
            p_value: p,
            pre_atts: vec![0.1, 0.2],
        });

        StaggeredDidResult {
            group_time_atts,
            event_study: vec![],
            group_effects: vec![],
            overall_att: AggregatedEffect {
                key: 0,
                att: 2.75,
                std_error: 0.4,
                ci_lower: 1.95,
                ci_upper: 3.55,
                p_value: 0.001,
                n_cells: 2,
                n_obs: 1000,
            },
            pretrend_test,
            config: StaggeredDidConfig::default(),
            cohorts: vec![2020],
            periods: vec![2019, 2020, 2021],
            n_obs: 1000,
            n_treated: 300,
            n_never_treated: 700,
            warnings: vec![],
        }
    }

    #[test]
    fn test_no_pretrend_violation() {
        let result = create_mock_staggered_result(Some(0.45), false);
        let report = staggered_did_diagnostics(&result);

        // Should not have parallel trends warning
        let pt_warning = report
            .warnings
            .iter()
            .find(|w| w.code == "PARALLEL_TRENDS_VIOLATED");
        assert!(pt_warning.is_none());

        // Should have passing assumption
        let pt_assumption = report
            .assumptions
            .iter()
            .find(|a| a.name.contains("pre-trends"));
        assert!(pt_assumption.is_some());
        assert_eq!(pt_assumption.unwrap().status, AssumptionStatus::NoViolation);
    }

    #[test]
    fn test_pretrend_violation_detected() {
        let result = create_mock_staggered_result(Some(0.02), true);
        let report = staggered_did_diagnostics(&result);

        // Should have parallel trends warning
        let pt_warning = report
            .warnings
            .iter()
            .find(|w| w.code == "PARALLEL_TRENDS_VIOLATED");
        assert!(pt_warning.is_some());
        assert!(pt_warning.unwrap().severity >= WarningSeverity::Caution);
    }

    #[test]
    fn test_anticipation_detected() {
        let result = create_mock_staggered_result(Some(0.5), true); // Significant pre-period
        let report = staggered_did_diagnostics(&result);

        // Should have anticipation warning
        let antic_warning = report
            .warnings
            .iter()
            .find(|w| w.code == "ANTICIPATION_DETECTED");
        assert!(antic_warning.is_some());
    }

    #[test]
    fn test_no_pretrend_periods() {
        // Create result with no pre-treatment periods
        let result = StaggeredDidResult {
            group_time_atts: vec![GroupTimeATT {
                group: 2020,
                time: 2020,
                att: 2.5,
                std_error: 0.5,
                ci_lower: 1.5,
                ci_upper: 3.5,
                n_treated: 100,
                n_comparison: 200,
                post_treatment: true,
                relative_time: 0,
            }],
            event_study: vec![],
            group_effects: vec![],
            overall_att: AggregatedEffect {
                key: 0,
                att: 2.5,
                std_error: 0.5,
                ci_lower: 1.5,
                ci_upper: 3.5,
                p_value: 0.001,
                n_cells: 1,
                n_obs: 300,
            },
            pretrend_test: None,
            config: StaggeredDidConfig::default(),
            cohorts: vec![2020],
            periods: vec![2020],
            n_obs: 300,
            n_treated: 100,
            n_never_treated: 200,
            warnings: vec![],
        };

        let report = staggered_did_diagnostics(&result);

        // Should have caution about missing pre-treatment periods
        let warning = report
            .warnings
            .iter()
            .find(|w| w.code == "NO_PRETREND_PERIODS");
        assert!(warning.is_some());
    }

    #[test]
    fn test_canonical_did_diagnostics() {
        use crate::traits::SignificanceLevel;

        let result = DiDResult {
            dep_var: "y".to_string(),
            treatment_var: "treated".to_string(),
            post_var: "post".to_string(),
            att: 2.5,
            std_error: 0.5,
            t_stat: 5.0,
            p_value: 0.001,
            significance: SignificanceLevel::TenthPercent,
            r_squared: 0.5,
            adj_r_squared: 0.49,
            n_obs: 1000,
            df: 996,
            control_pre_mean: 4.0,
            control_post_mean: 6.5,
            treated_pre_mean: 5.0,
            treated_post_mean: 10.0,
            coefficients: vec![4.0, 1.0, 2.5, 2.5],
            std_errors: vec![0.2, 0.3, 0.3, 0.5],
            variables: vec![
                "intercept".to_string(),
                "treated".to_string(),
                "post".to_string(),
                "treated:post".to_string(),
            ],
            controls: vec![],
            cluster_var: None,
            n_clusters: None,
        };

        let report = did_diagnostics(&result);

        // Should note that parallel trends is untestable
        assert!(
            report
                .assumptions
                .iter()
                .any(|a| !a.testable && a.name.contains("Parallel"))
        );

        // Should have info warning about untestable assumption
        assert!(
            report
                .warnings
                .iter()
                .any(|w| w.code == "PARALLEL_TRENDS_UNTESTABLE")
        );
    }
}
