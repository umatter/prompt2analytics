//! Identification diagnostics for instrumental variables (IV/2SLS) estimation.
//!
//! Checks for:
//! - Weak instruments: the first-stage partial F of the excluded instruments (HC1 Wald F
//!   when the IV was robust) below 10
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
/// 1. **Weak instrument test**: Checks if the first-stage partial F of the excluded
///    instruments (HC1 Wald F when the IV was robust) of every endogenous variable
///    reaches 10 (rule of thumb). F < 10 indicates weak instruments that can bias
///    2SLS toward OLS and cause unreliable inference.
///
/// 2. **Overidentification test**: If the model is overidentified (more instruments
///    than endogenous variables), runs the Sargan test. Rejection (p < 0.05)
///    suggests at least one instrument may violate the exclusion restriction.
///
/// # Example
///
/// ```ignore
/// use p2a_core::econometrics::run_iv2sls;
/// use p2a_core::diagnostics::iv_diagnostics;
///
/// let iv_result = run_iv2sls(&dataset, "y", &["exog"], &["endog"], &["z1", "z2"], false)?;
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

/// Check for weak instruments using the first-stage partial F of the excluded
/// instruments (HC1 Wald F when the IV was robust), `IVResult::first_stage_f_stats`.
/// A first-stage F that was not computed (NaN) counts as weak.
fn check_weak_instruments(result: &IVResult, report: &mut IdentificationReport) {
    let stats = &result.first_stage_f_stats;
    if stats.is_empty() {
        return;
    }
    let f_type = if result.first_stage_f_type.is_empty() {
        "F"
    } else {
        result.first_stage_f_type.as_str()
    };

    // The endogenous variable with the smallest first-stage F (NaN first)
    let mut weakest = 0;
    for (j, &f) in stats.iter().enumerate() {
        let current = stats[weakest];
        if !current.is_nan() && (f.is_nan() || f < current) {
            weakest = j;
        }
    }
    let min_f = stats[weakest];
    let name = result
        .endogenous_vars
        .get(weakest)
        .map_or("endogenous", String::as_str);
    let avg_f: f64 = stats.iter().sum::<f64>() / stats.len() as f64;

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

    // Generate warning if F < 10 (or not computed)
    if status != AssumptionStatus::NoViolation {
        let severity = if min_f >= 5.0 {
            WarningSeverity::Warning
        } else {
            WarningSeverity::Critical
        };

        let message = if stats.len() == 1 {
            let df = match result.first_stage_df.get(weakest) {
                Some((df1, df2)) => format!(" (df {df1}, {df2})"),
                None => String::new(),
            };
            format!(
                "First-stage {f_type} for '{name}' is {min_f:.2}{df}, below the rule-of-thumb \
                 threshold of 10. Weak instruments bias 2SLS toward OLS and make its standard \
                 errors unreliable. The Stock-Yogo critical values give sharper thresholds for \
                 a chosen maximal bias."
            )
        } else {
            format!(
                "Minimum first-stage {f_type} across {} endogenous variables is {min_f:.2} \
                 (for '{name}'; average {avg_f:.2}), below the threshold of 10. Weak \
                 instruments for any endogenous variable compromise the entire 2SLS \
                 estimation.",
                stats.len()
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
            WarningDiagnostics::new(format!("First-stage {f_type}"), min_f)
                .with_threshold(10.0, false),
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

    /// Weak excluded instruments next to a control that predicts the endogenous
    /// regressor (AER weak-instrument F 0.2267, HC1 0.2317): a Critical warning that
    /// names the statistic and its df, for classical and robust fits alike. The
    /// assertions used to sit inside `if F < 10`, which the overall F (300.8) skipped.
    #[test]
    fn test_validate_iv_diagnostics_weak_controls() {
        use crate::econometrics::run_iv2sls;
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("iv_weak_controls");
        for robust in [false, true] {
            let r = run_iv2sls(&ds, "y", &["w"], &["x"], &["z1", "z2"], robust).unwrap();
            let rep = iv_diagnostics(&ds, &r).unwrap();
            let w = rep
                .warnings
                .iter()
                .find(|w| w.code == "WEAK_INSTRUMENT")
                .expect("weak-instrument warning");
            assert_eq!(w.severity, WarningSeverity::Critical); // F < 5
            let key = if robust { "weak_f_hc1" } else { "weak_f" };
            let value = w.diagnostics.as_ref().unwrap().value;
            Check::new(&case)
                .vector(key, &[value], Tol::stat(r.n_obs))
                .finish();
            let f_type = if robust { "HC1 Wald F" } else { "partial F" };
            assert!(
                w.message.contains(f_type) && w.message.contains("(df 2, 496)"),
                "{}",
                w.message
            );
            let rel = rep
                .assumptions
                .iter()
                .find(|a| a.name == "Instrument relevance")
                .unwrap();
            assert_eq!(rel.status, AssumptionStatus::LikelyViolation);
            // Sargan (over-identified by one): p = 0.225 -> no OVERID warning
            assert!(rep.warnings.iter().all(|w| w.code != "OVERID_REJECTED"));
        }
    }

    /// Two endogenous regressors with strong instruments (partial F 30.18 and 24.40):
    /// no weak-instrument warning, and the Sargan test (p = 0.52) is not rejected.
    #[test]
    fn test_validate_iv_diagnostics_two_endog() {
        use crate::econometrics::run_iv2sls;
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("iv_two_endog");
        let r = run_iv2sls(
            &ds,
            "y",
            &["w1", "w2"],
            &["x1", "x2"],
            &["z1", "z2", "z3"],
            false,
        )
        .unwrap();
        let rep = iv_diagnostics(&ds, &r).unwrap();
        assert!(
            rep.warnings.iter().all(|w| w.code != "WEAK_INSTRUMENT"),
            "{:?}",
            rep.warnings
        );
        let status = |name: &str| {
            rep.assumptions
                .iter()
                .find(|a| a.name == name)
                .unwrap_or_else(|| panic!("no assumption {name}"))
                .status
        };
        assert_eq!(
            status("Instrument relevance"),
            AssumptionStatus::NoViolation
        );
        assert_eq!(
            status("Overidentifying restrictions"),
            AssumptionStatus::NoViolation
        );
        let sargan = run_sargan_test(&ds, &r).unwrap();
        Check::new(&case)
            .scalar("sargan_p", sargan.p_value, Tol::p_value(400))
            .finish();
    }

    /// A first-stage F that could not be computed (NaN, e.g. 0/0) is a critical
    /// weak-instrument warning, not a pass.
    #[test]
    fn test_nan_first_stage_f_is_a_critical_weak_instrument() {
        use crate::econometrics::run_iv2sls;
        let dataset = create_strong_iv_dataset();
        let mut iv = run_iv2sls(&dataset, "y", &[], &["x_endog"], &["z"], false).unwrap();
        iv.first_stage_f_stats = vec![f64::NAN];
        let rep = iv_diagnostics(&dataset, &iv).unwrap();
        let w = rep
            .warnings
            .iter()
            .find(|w| w.code == "WEAK_INSTRUMENT")
            .expect("a NaN first-stage F must warn");
        assert_eq!(w.severity, WarningSeverity::Critical);
        let rel = rep
            .assumptions
            .iter()
            .find(|a| a.name == "Instrument relevance")
            .unwrap();
        assert_eq!(rel.status, AssumptionStatus::LikelyViolation);
        assert!(!iv.first_stage_f_stats.iter().all(|&f| f >= 10.0));
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
