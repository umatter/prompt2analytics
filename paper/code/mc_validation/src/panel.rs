//! MC validation for panel data methods.

use crate::dgp::*;
use crate::framework::*;
/// Run all panel data MC validations.
pub fn validate_panel(config: &McConfig, n: usize) -> Vec<McResult> {
    let mut results = Vec::new();

    results.extend(validate_fe(config, n));
    results.extend(validate_re(config, n));

    results
}

fn validate_fe(config: &McConfig, n: usize) -> Vec<McResult> {
    let mut results = Vec::new();
    let z_crit = 1.96;
    // Panel dimensions: sqrt(n) entities × sqrt(n) periods (approximately)
    let n_ent = (n as f64).sqrt() as usize;
    let n_per = n / n_ent;

    for (coef_idx, coef_name) in [(0, "beta1"), (1, "beta2")] {
        let mut draws = Vec::with_capacity(config.n_sims);

        for sim in 0..config.n_sims {
            let seed = config.seed + sim as u64;
            let (dataset, dgp) = dgp_panel_fe(n_ent, n_per, seed);
            let true_val = dgp.true_coefs[coef_idx];

            if let Ok(result) = p2a_core::run_fixed_effects(
                &dataset, "y", &["x1", "x2"], "entity",
            ) {
                let est = result.coefficients[coef_idx];
                let se = result.std_errors[coef_idx];
                draws.push(EstimatorDraw {
                    estimate: est,
                    std_error: se,
                    ci_lower: est - z_crit * se,
                    ci_upper: est + z_crit * se,
                });
            }
        }

        let true_val = [0.5, -0.3][coef_idx];

        let mut r = evaluate_coverage(&draws, true_val, config);
        r.method = format!("FE_{}", coef_name);
        r.dgp = "panel_fe_dgp".to_string();
        r.n = n_ent * n_per;
        results.push(r);

        let mut r = evaluate_se_accuracy(&draws, config);
        r.method = format!("FE_{}", coef_name);
        r.dgp = "panel_fe_dgp".to_string();
        r.n = n_ent * n_per;
        results.push(r);
    }

    results
}

fn validate_re(config: &McConfig, n: usize) -> Vec<McResult> {
    let mut results = Vec::new();
    let z_crit = 1.96;
    let n_ent = (n as f64).sqrt() as usize;
    let n_per = n / n_ent;

    // RE on FE DGP — should still be consistent for slopes
    for (coef_idx, coef_name) in [(0, "beta1"), (1, "beta2")] {
        let mut draws = Vec::with_capacity(config.n_sims);

        for sim in 0..config.n_sims {
            let seed = config.seed + sim as u64;
            let (dataset, dgp) = dgp_panel_fe(n_ent, n_per, seed);

            if let Ok(result) = p2a_core::run_random_effects(
                &dataset, "y", &["x1", "x2"], "entity",
            ) {
                let coefs = &result.coefficients;
                let ses = &result.std_errors;
                // RE includes intercept, so slopes are at indices 1, 2
                let est = coefs[coef_idx + 1];
                let se = ses[coef_idx + 1];
                draws.push(EstimatorDraw {
                    estimate: est,
                    std_error: se,
                    ci_lower: est - z_crit * se,
                    ci_upper: est + z_crit * se,
                });
            }
        }

        let true_val = [0.5, -0.3][coef_idx];

        let mut r = evaluate_coverage(&draws, true_val, config);
        r.method = format!("RE_{}", coef_name);
        r.dgp = "panel_fe_dgp".to_string();
        r.n = n_ent * n_per;
        results.push(r);
    }

    results
}
