//! Identification diagnostics for regression discontinuity (RD) designs.
//!
//! Checks for:
//! - Small effective sample size near the cutoff
//! - Bandwidth sensitivity (comparing estimates at different bandwidths)
//!
//! Note: McCrary manipulation test is not currently implemented in p2a-core.
//! Future versions may add density discontinuity testing.

use crate::econometrics::RdResult;

use super::{
    Assumption, AssumptionStatus, IdentificationReport, IdentificationWarning, WarningDiagnostics,
    WarningSeverity,
};

/// Minimum effective sample size for reliable local polynomial estimation.
const MIN_EFFECTIVE_N: usize = 50;

/// Warning threshold for effective sample size.
const WARN_EFFECTIVE_N: usize = 100;

/// Generate identification diagnostics for an RD result.
///
/// # Diagnostics Performed
///
/// 1. **Effective sample size**: Checks if enough observations exist within
///    the bandwidth on each side of the cutoff for reliable estimation.
///
/// 2. **Bandwidth selection**: Reports the bandwidth method used and notes
///    that estimates may be sensitive to bandwidth choice.
///
/// # Example
///
/// ```ignore
/// use p2a_core::econometrics::rd::run_rd;
/// use p2a_core::diagnostics::rd_diagnostics;
///
/// let result = run_rd(&dataset, "outcome", "running", 0.0, &config)?;
/// let report = rd_diagnostics(&result);
/// ```
///
/// # Limitations
///
/// The McCrary manipulation test (testing for density discontinuity at the
/// cutoff) is not currently implemented. Users should examine running variable
/// histograms and consider domain knowledge about manipulation possibilities.
pub fn rd_diagnostics(result: &RdResult) -> IdentificationReport {
    let mut report = IdentificationReport::new("Regression Discontinuity");

    // Check effective sample size
    check_effective_sample_size(result, &mut report);

    // Note about bandwidth sensitivity
    add_bandwidth_note(result, &mut report);

    // Add standard RD assumptions
    report.add_assumption(Assumption::untestable(
        "No manipulation",
        "Units cannot precisely manipulate the running variable to select into treatment",
    ));

    report.add_assumption(Assumption::untestable(
        "Continuity",
        "Potential outcomes are continuous at the cutoff",
    ));

    // Note that manipulation test is not available
    report.add_warning(
        IdentificationWarning::new(
            "MCCRARY_NOT_AVAILABLE",
            WarningSeverity::Info,
            "Manipulation Test Not Available",
            "The McCrary density test for running variable manipulation is not \
             currently implemented. Examine a histogram of the running variable \
             near the cutoff to visually assess whether bunching or gaps suggest \
             manipulation. Consider domain knowledge about whether units can \
             control their running variable value.",
            "No manipulation",
        )
        .with_remediation(vec![
            "Plot histogram of running variable near cutoff".to_string(),
            "Check for discontinuities in predetermined covariates at cutoff".to_string(),
            "Consider whether manipulation is plausible given the context".to_string(),
        ]),
    );

    report
}

/// Check effective sample size near the cutoff.
fn check_effective_sample_size(result: &RdResult, report: &mut IdentificationReport) {
    let n_left = result.n_eff_left;
    let n_right = result.n_eff_right;
    let min_n = n_left.min(n_right);

    // Determine status
    let status = if min_n >= WARN_EFFECTIVE_N {
        AssumptionStatus::NoViolation
    } else if min_n >= MIN_EFFECTIVE_N {
        AssumptionStatus::PotentialViolation
    } else {
        AssumptionStatus::LikelyViolation
    };

    report.add_assumption(Assumption::testable(
        "Sufficient local data",
        "Enough observations within bandwidth for reliable local polynomial estimation",
        status,
    ));

    if min_n < WARN_EFFECTIVE_N {
        let severity = if min_n < MIN_EFFECTIVE_N {
            WarningSeverity::Warning
        } else {
            WarningSeverity::Caution
        };

        let side = if n_left < n_right { "left" } else { "right" };

        let warning = IdentificationWarning::new(
            "RD_SMALL_SAMPLE",
            severity,
            "Limited Observations Near Cutoff",
            format!(
                "Effective sample size is {} on the {} of the cutoff ({} left, {} right). \
                 Local polynomial estimates may be imprecise with fewer than {} observations. \
                 Bandwidth: h_left = {:.4}, h_right = {:.4}.",
                min_n, side, n_left, n_right, WARN_EFFECTIVE_N, result.h_left, result.h_right
            ),
            "Sufficient local data",
        )
        .with_diagnostics(
            WarningDiagnostics::new("Min effective n", min_n as f64)
                .with_threshold(WARN_EFFECTIVE_N as f64, false),
        )
        .with_remediation(vec![
            "Consider larger bandwidth (with bias correction)".to_string(),
            "Use lower-order polynomial to reduce variance".to_string(),
            "Report wide confidence intervals reflecting uncertainty".to_string(),
            "Consider parametric methods if local data is very sparse".to_string(),
        ]);

        report.add_warning(warning);
    }
}

/// Add note about bandwidth selection and sensitivity.
fn add_bandwidth_note(result: &RdResult, report: &mut IdentificationReport) {
    // Always add informational note about bandwidth choice
    let h_symmetric = (result.h_left - result.h_right).abs() < 0.001;

    let bandwidth_desc = if h_symmetric {
        format!("h = {:.4}", result.h_left)
    } else {
        format!(
            "h_left = {:.4}, h_right = {:.4}",
            result.h_left, result.h_right
        )
    };

    report.add_warning(
        IdentificationWarning::new(
            "BANDWIDTH_SENSITIVITY",
            WarningSeverity::Info,
            "Bandwidth Selection",
            format!(
                "RD estimates use MSE-optimal bandwidth ({}). Results may be sensitive \
                 to bandwidth choice. Consider reporting estimates for a range of \
                 bandwidths (e.g., 0.5h, 0.75h, 1.25h, 1.5h) as robustness checks.",
                bandwidth_desc
            ),
            "Bandwidth choice",
        )
        .with_remediation(vec![
            "Report estimates at multiple bandwidths".to_string(),
            "Use bias-corrected robust inference (rdrobust)".to_string(),
            "Consider different polynomial orders as robustness check".to_string(),
        ]),
    );
}

/// Run bandwidth sensitivity analysis.
///
/// Computes RD estimates at multiple bandwidths to assess sensitivity.
/// Returns a summary of how estimates vary with bandwidth choice.
///
/// # Arguments
///
/// * `estimates` - Vector of (bandwidth_multiplier, estimate, se) tuples
///
/// # Returns
///
/// Coefficient of variation across estimates, indicating sensitivity.
pub fn bandwidth_sensitivity_cv(estimates: &[(f64, f64, f64)]) -> f64 {
    if estimates.len() < 2 {
        return 0.0;
    }

    let effects: Vec<f64> = estimates.iter().map(|(_, est, _)| *est).collect();
    let mean = effects.iter().sum::<f64>() / effects.len() as f64;

    if mean.abs() < 1e-10 {
        return f64::INFINITY;
    }

    let variance =
        effects.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (effects.len() - 1) as f64;

    variance.sqrt() / mean.abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::econometrics::{BandwidthMethod, KernelType, RdConfig, VceType};
    use crate::traits::SignificanceLevel;

    fn create_mock_rd_result(n_left: usize, n_right: usize) -> RdResult {
        RdResult {
            outcome: "y".to_string(),
            running_var: "x".to_string(),
            cutoff: 0.0,
            n_left: 500,
            n_right: 500,
            n_eff_left: n_left,
            n_eff_right: n_right,
            tau_conventional: 2.5,
            tau_bc: 2.45,
            tau_robust: 2.45,
            se_conventional: 0.5,
            se_bc: 0.52,
            se_robust: 0.55,
            ci_conventional: (1.5, 3.5),
            ci_bc: (1.43, 3.47),
            ci_robust: (1.4, 3.6),
            p_conventional: 0.001,
            p_bc: 0.001,
            p_robust: 0.001,
            significance: SignificanceLevel::TenthPercent,
            h_left: 0.5,
            h_right: 0.5,
            b_left: 0.75,
            b_right: 0.75,
            bwselect: BandwidthMethod::MseTwo,
            p: 1,
            q: 2,
            kernel: KernelType::Triangular,
            vce: VceType::Hc1,
            coef_left: vec![5.0, -0.5],
            coef_right: vec![7.5, 0.3],
            bias: 0.05,
            warnings: vec![],
        }
    }

    #[test]
    fn test_sufficient_sample_size() {
        let result = create_mock_rd_result(150, 200);
        let report = rd_diagnostics(&result);

        // Should not have sample size warning
        let warning = report.warnings.iter().find(|w| w.code == "RD_SMALL_SAMPLE");
        assert!(warning.is_none());
    }

    #[test]
    fn test_small_sample_warning() {
        let result = create_mock_rd_result(40, 200);
        let report = rd_diagnostics(&result);

        // Should have sample size warning
        let warning = report.warnings.iter().find(|w| w.code == "RD_SMALL_SAMPLE");
        assert!(warning.is_some());
        assert!(warning.unwrap().severity >= WarningSeverity::Warning);
    }

    #[test]
    fn test_bandwidth_sensitivity_cv() {
        let estimates = vec![
            (0.5, 2.0, 0.6),
            (0.75, 2.2, 0.55),
            (1.0, 2.5, 0.5),
            (1.25, 2.3, 0.55),
            (1.5, 2.1, 0.6),
        ];

        let cv = bandwidth_sensitivity_cv(&estimates);
        assert!(cv > 0.0);
        assert!(cv < 1.0); // Reasonable variation
    }

    #[test]
    fn test_mccrary_not_available_note() {
        let result = create_mock_rd_result(150, 200);
        let report = rd_diagnostics(&result);

        // Should have note about McCrary not available
        let warning = report
            .warnings
            .iter()
            .find(|w| w.code == "MCCRARY_NOT_AVAILABLE");
        assert!(warning.is_some());
        assert_eq!(warning.unwrap().severity, WarningSeverity::Info);
    }

    #[test]
    fn test_bandwidth_note() {
        let result = create_mock_rd_result(150, 200);
        let report = rd_diagnostics(&result);

        // Should have bandwidth sensitivity note
        let warning = report
            .warnings
            .iter()
            .find(|w| w.code == "BANDWIDTH_SENSITIVITY");
        assert!(warning.is_some());
    }
}
