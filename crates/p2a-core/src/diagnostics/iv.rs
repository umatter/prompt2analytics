//! Identification diagnostics for instrumental variables (IV/2SLS) estimation.
//!
//! Checks for:
//! - Weak instruments (first-stage F < 10)
//! - Overidentification test failures (Sargan/Hansen J test)

use crate::data::Dataset;
use crate::econometrics::{IVResult, SarganTestResult, run_sargan_test};
use crate::errors::EconResult;

use super::{
    Assumption, AssumptionStatus, IdentificationReport, IdentificationWarning, WarningDiagnostics,
    WarningSeverity,
};

/// Generate identification diagnostics for an IV/2SLS result.
///
/// # Diagnostics Performed
///
/// 1. **Weak instrument test**: Checks if first-stage F-statistics exceed 10
///    (Stock & Yogo rule of thumb). F < 10 indicates weak instruments that
///    can bias 2SLS toward OLS and cause unreliable inference.
///
/// 2. **Overidentification test**: If the model is overidentified (more instruments
///    than endogenous variables), runs the Sargan test. Rejection (p < 0.05)
///    suggests at least one instrument may violate the exclusion restriction.
///
/// # Example
///
/// ```ignore
/// use p2a_core::econometrics::iv::run_2sls;
/// use p2a_core::diagnostics::iv_diagnostics;
///
/// let iv_result = run_iv2sls(&dataset, "y", &["exog"], &["endog"], &["z1", "z2"])?;
/// let report = iv_diagnostics(&dataset, &iv_result)?;
///
/// if report.has_critical() {
///     println!("Warning: {}", report.summary());
/// }
/// ```
pub fn iv_diagnostics(dataset: &Dataset, result: &IVResult) -> EconResult<IdentificationReport> {
    let mut report = IdentificationReport::new("2SLS / Instrumental Variables");

    // === Instrument Relevance (Testable) ===
    check_weak_instruments(result, &mut report);

    // === Exclusion Restriction (Partially testable if overidentified) ===
    check_overidentification(dataset, result, &mut report)?;

    // === Add untestable assumptions ===
    report.add_assumption(Assumption::untestable(
        "Exclusion restriction",
        "Instruments affect outcome only through the endogenous variable(s)",
    ));

    report.add_assumption(Assumption::untestable(
        "Independence",
        "Instruments are uncorrelated with the error term",
    ));

    Ok(report)
}

/// Check for weak instruments using first-stage F-statistics.
fn check_weak_instruments(result: &IVResult, report: &mut IdentificationReport) {
    if result.first_stage_f_stats.is_empty() {
        return;
    }

    // Check each endogenous variable's first-stage F
    let min_f = result
        .first_stage_f_stats
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);

    let avg_f: f64 =
        result.first_stage_f_stats.iter().sum::<f64>() / result.first_stage_f_stats.len() as f64;

    // Determine status based on minimum F across endogenous variables
    let status = if min_f >= 10.0 {
        AssumptionStatus::NoViolation
    } else if min_f >= 5.0 {
        AssumptionStatus::PotentialViolation
    } else {
        AssumptionStatus::LikelyViolation
    };

    report.add_assumption(Assumption::testable(
        "Instrument relevance",
        "Instruments must be correlated with endogenous variable(s)",
        status,
    ));

    // Generate warning if F < 10
    if min_f < 10.0 {
        let severity = if min_f < 5.0 {
            WarningSeverity::Critical
        } else {
            WarningSeverity::Warning
        };

        let message = if result.first_stage_f_stats.len() == 1 {
            format!(
                "First-stage F-statistic is {:.2}, below the rule-of-thumb threshold of 10. \
                 Weak instruments cause 2SLS estimates to be biased toward OLS and make \
                 standard errors unreliable. The Stock-Yogo critical values provide more \
                 precise thresholds based on acceptable bias levels.",
                min_f
            )
        } else {
            format!(
                "Minimum first-stage F-statistic across {} endogenous variables is {:.2} \
                 (average: {:.2}), below the threshold of 10. Weak instruments for any \
                 endogenous variable compromise the entire 2SLS estimation.",
                result.first_stage_f_stats.len(),
                min_f,
                avg_f
            )
        };

        let warning = IdentificationWarning::new(
            "WEAK_INSTRUMENT",
            severity,
            "Weak Instrument Detected",
            message,
            "Instrument relevance",
        )
        .with_diagnostics(
            WarningDiagnostics::new("First-stage F", min_f).with_threshold(10.0, false),
        )
        .with_remediation(vec![
            "Consider stronger instruments with higher first-stage F".to_string(),
            "Use weak-instrument-robust inference (Anderson-Rubin test, LIML)".to_string(),
            "Report reduced-form estimates as robustness check".to_string(),
            "If F is very low (<5), reconsider the identification strategy".to_string(),
        ]);

        report.add_warning(warning);
    }
}

/// Check overidentification using the Sargan test.
fn check_overidentification(
    dataset: &Dataset,
    result: &IVResult,
    report: &mut IdentificationReport,
) -> EconResult<()> {
    // Only applicable if overidentified
    let n_instruments = result.instruments.len();
    let n_endogenous = result.endogenous_vars.len();

    if n_instruments <= n_endogenous {
        // Exactly identified or underidentified - can't test exclusion restriction
        return Ok(());
    }

    // Run Sargan test
    let sargan: SarganTestResult = run_sargan_test(dataset, result)?;

    if !sargan.overidentified {
        return Ok(());
    }

    // Add testable aspect of exclusion restriction
    let status = if sargan.p_value >= 0.05 {
        AssumptionStatus::NoViolation
    } else if sargan.p_value >= 0.01 {
        AssumptionStatus::PotentialViolation
    } else {
        AssumptionStatus::LikelyViolation
    };

    report.add_assumption(Assumption::testable(
        "Overidentifying restrictions",
        "All instruments satisfy the exclusion restriction (testable when overidentified)",
        status,
    ));

    // Generate warning if Sargan test rejects
    if sargan.p_value < 0.05 {
        let severity = if sargan.p_value < 0.01 {
            WarningSeverity::Warning
        } else {
            WarningSeverity::Caution
        };

        let warning = IdentificationWarning::new(
            "OVERID_REJECTED",
            severity,
            "Overidentification Test Rejected",
            format!(
                "The Sargan test rejects the null hypothesis that all instruments are valid \
                 (J = {:.3}, df = {}, p = {:.4}). This suggests at least one instrument \
                 may be correlated with the error term, violating the exclusion restriction. \
                 However, this test has limitations: it cannot identify which instrument \
                 is invalid, and it assumes at least one instrument is valid.",
                sargan.j_statistic, sargan.df, sargan.p_value
            ),
            "Exclusion restriction",
        )
        .with_diagnostics(
            WarningDiagnostics::new("Sargan J statistic", sargan.j_statistic)
                .with_threshold(sargan.p_value, true),
        )
        .with_remediation(vec![
            "Examine theoretical justification for each instrument".to_string(),
            "Consider dropping potentially invalid instruments".to_string(),
            "Report results with different instrument subsets".to_string(),
            "Use LIML which is more robust to weak instruments".to_string(),
        ]);

        report.add_warning(warning);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn create_weak_iv_dataset() -> Dataset {
        // Create data where instrument is weakly correlated with endogenous var
        let n = 500;
        let z: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        // Very weak relationship: x = 0.05*z + lots of noise
        let x: Vec<f64> = z
            .iter()
            .enumerate()
            .map(|(i, &zi)| 0.05 * zi + (i as f64 * 0.3).cos() * 5.0)
            .collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| 2.0 * xi + (i as f64 * 0.2).sin() * 3.0)
            .collect();

        let df = DataFrame::new(vec![
            Column::new("y".into(), y),
            Column::new("x_endog".into(), x),
            Column::new("z".into(), z),
        ])
        .unwrap();

        Dataset::new(df)
    }

    fn create_strong_iv_dataset() -> Dataset {
        // Create data where instrument is strongly correlated with endogenous var
        let n = 500;
        let z: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin() * 3.0).collect();
        // Strong relationship: x = 0.8*z + small noise
        let x: Vec<f64> = z
            .iter()
            .enumerate()
            .map(|(i, &zi)| 0.8 * zi + (i as f64 * 0.5).cos() * 0.5)
            .collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| 2.0 * xi + (i as f64 * 0.2).sin())
            .collect();

        let df = DataFrame::new(vec![
            Column::new("y".into(), y),
            Column::new("x_endog".into(), x),
            Column::new("z".into(), z),
        ])
        .unwrap();

        Dataset::new(df)
    }

    #[test]
    fn test_weak_instrument_detection() {
        use crate::econometrics::run_iv2sls;

        let dataset = create_weak_iv_dataset();
        let iv_result = run_iv2sls(&dataset, "y", &[], &["x_endog"], &["z"], true).unwrap();

        let report = iv_diagnostics(&dataset, &iv_result).unwrap();

        // Should detect weak instrument
        let weak_warning = report.warnings.iter().find(|w| w.code == "WEAK_INSTRUMENT");

        // If F < 10, we should have a warning
        if iv_result.first_stage_f_stats[0] < 10.0 {
            assert!(weak_warning.is_some(), "Should detect weak instrument");
            assert!(
                weak_warning.unwrap().severity >= WarningSeverity::Warning,
                "Weak instrument should be at least Warning severity"
            );
        }
    }

    #[test]
    fn test_strong_instrument_no_warning() {
        use crate::econometrics::run_iv2sls;

        let dataset = create_strong_iv_dataset();
        let iv_result = run_iv2sls(&dataset, "y", &[], &["x_endog"], &["z"], true).unwrap();

        let report = iv_diagnostics(&dataset, &iv_result).unwrap();

        // If F > 10, should not have weak instrument warning
        if iv_result.first_stage_f_stats[0] >= 10.0 {
            let weak_warning = report.warnings.iter().find(|w| w.code == "WEAK_INSTRUMENT");
            assert!(
                weak_warning.is_none(),
                "Should not warn for strong instruments"
            );
        }
    }

    #[test]
    fn test_report_summary() {
        use crate::econometrics::run_iv2sls;

        let dataset = create_strong_iv_dataset();
        let iv_result = run_iv2sls(&dataset, "y", &[], &["x_endog"], &["z"], true).unwrap();

        let report = iv_diagnostics(&dataset, &iv_result).unwrap();

        let summary = report.summary();
        assert!(
            summary.contains("2SLS") || summary.contains("Instrumental Variables"),
            "Summary should mention method"
        );
    }

    #[test]
    fn test_assumptions_listed() {
        use crate::econometrics::run_iv2sls;

        let dataset = create_strong_iv_dataset();
        let iv_result = run_iv2sls(&dataset, "y", &[], &["x_endog"], &["z"], true).unwrap();

        let report = iv_diagnostics(&dataset, &iv_result).unwrap();

        // Should have relevance assumption
        assert!(
            report
                .assumptions
                .iter()
                .any(|a| a.name.contains("relevance")),
            "Should list instrument relevance assumption"
        );

        // Should have exclusion restriction assumption
        assert!(
            report
                .assumptions
                .iter()
                .any(|a| a.name.contains("Exclusion")),
            "Should list exclusion restriction assumption"
        );
    }
}
