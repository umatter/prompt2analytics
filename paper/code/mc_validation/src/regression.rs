//! MC validation for regression methods.

use crate::dgp::*;
use crate::framework::*;
use p2a_core::regression::{CovarianceType, run_ols, vcov_hac, HacKernel, gls, CorrelationStructure, quantreg, QuantRegConfig};
use p2a_core::traits::LinearEstimator;

/// Run all regression MC validations.
pub fn validate_regression(config: &McConfig, n: usize) -> Vec<McResult> {
    let mut results = Vec::new();

    // OLS with standard SEs under homoskedasticity
    results.extend(validate_ols_standard(config, n));
    // OLS with HC SEs under heteroskedasticity
    results.extend(validate_ols_hc(config, n));
    // OLS standard SEs should fail under heteroskedasticity
    results.extend(validate_ols_het_failure(config, n));

    results
}

fn validate_ols_standard(config: &McConfig, n: usize) -> Vec<McResult> {
    let mut results = Vec::new();
    let x_cols = vec!["x1", "x2"];
    let z_crit = 1.96; // for 95% CI

    // Test each coefficient: intercept, β1, β2
    for (coef_idx, coef_name) in [(0, "intercept"), (1, "beta1"), (2, "beta2")] {
        let mut draws = Vec::with_capacity(config.n_sims);

        for sim in 0..config.n_sims {
            let seed = config.seed + sim as u64;
            let (dataset, dgp) = dgp_regression_homoskedastic(n, seed);
            let true_val = dgp.true_coefs[coef_idx];

            if let Ok(result) = run_ols(&dataset, "y", &x_cols, true, CovarianceType::Standard) {
                let coefs = result.coefficients();
                let ses = result.std_errors();
                // OLS output: intercept is first when intercept=true
                let est = coefs[coef_idx];
                let se = ses[coef_idx];
                draws.push(EstimatorDraw {
                    estimate: est,
                    std_error: se,
                    ci_lower: est - z_crit * se,
                    ci_upper: est + z_crit * se,
                });
            }
        }

        let true_val = [1.0, 0.5, -0.3][coef_idx];

        // CI coverage
        let mut r = evaluate_coverage(&draws, true_val, config);
        r.method = format!("OLS_Standard_{}", coef_name);
        r.property = "ci_coverage".to_string();
        r.dgp = "homoskedastic".to_string();
        r.n = n;
        results.push(r);

        // SE accuracy
        let mut r = evaluate_se_accuracy(&draws, config);
        r.method = format!("OLS_Standard_{}", coef_name);
        r.dgp = "homoskedastic".to_string();
        r.n = n;
        results.push(r);
    }

    results
}

fn validate_ols_hc(config: &McConfig, n: usize) -> Vec<McResult> {
    let mut results = Vec::new();
    let x_cols = vec!["x1", "x2"];
    let z_crit = 1.96;

    for cov_type in [CovarianceType::HC0, CovarianceType::HC1, CovarianceType::HC2, CovarianceType::HC3] {
        let cov_name = format!("{:?}", cov_type);

        // Test β1 under heteroskedasticity
        let mut draws = Vec::with_capacity(config.n_sims);
        let coef_idx = 1; // β1

        for sim in 0..config.n_sims {
            let seed = config.seed + sim as u64;
            let (dataset, dgp) = dgp_regression_heteroskedastic(n, seed);

            if let Ok(result) = run_ols(&dataset, "y", &x_cols, true, cov_type) {
                let est = result.coefficients()[coef_idx];
                let se = result.std_errors()[coef_idx];
                draws.push(EstimatorDraw {
                    estimate: est,
                    std_error: se,
                    ci_lower: est - z_crit * se,
                    ci_upper: est + z_crit * se,
                });
            }
        }

        let true_val = 0.5; // β1

        let mut r = evaluate_coverage(&draws, true_val, config);
        r.method = format!("OLS_{}_beta1", cov_name);
        r.dgp = "heteroskedastic".to_string();
        r.n = n;
        results.push(r);

        let mut r = evaluate_se_accuracy(&draws, config);
        r.method = format!("OLS_{}_beta1", cov_name);
        r.dgp = "heteroskedastic".to_string();
        r.n = n;
        results.push(r);
    }

    results
}

/// OLS standard SEs under heteroskedasticity — coverage should be WRONG.
/// This is a negative control: confirms that standard SEs are mis-calibrated.
fn validate_ols_het_failure(config: &McConfig, n: usize) -> Vec<McResult> {
    let x_cols = vec!["x1", "x2"];
    let z_crit = 1.96;
    let coef_idx = 1;

    let mut draws = Vec::with_capacity(config.n_sims);
    for sim in 0..config.n_sims {
        let seed = config.seed + sim as u64;
        let (dataset, _dgp) = dgp_regression_heteroskedastic(n, seed);

        if let Ok(result) = run_ols(&dataset, "y", &x_cols, true, CovarianceType::Standard) {
            let est = result.coefficients()[coef_idx];
            let se = result.std_errors()[coef_idx];
            draws.push(EstimatorDraw {
                estimate: est,
                std_error: se,
                ci_lower: est - z_crit * se,
                ci_upper: est + z_crit * se,
            });
        }
    }

    let true_val = 0.5;
    let mut r = evaluate_coverage(&draws, true_val, config);
    r.method = "OLS_Standard_beta1".to_string();
    r.property = "ci_coverage_negative_control".to_string();
    r.dgp = "heteroskedastic".to_string();
    r.n = n;
    // For this negative control, we EXPECT coverage to be below nominal.
    // "within_tolerance" = true means the test correctly detects the problem.
    // With strong het (x1⁴ variance), standard SEs should badly undercover.
    r.within_tolerance = r.observed < 0.90; // coverage should be well below 95%

    vec![r]
}
