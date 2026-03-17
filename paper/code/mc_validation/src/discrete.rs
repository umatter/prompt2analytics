//! MC validation for discrete choice models.

use crate::dgp::*;
use crate::framework::*;
/// Run all discrete choice MC validations.
pub fn validate_discrete(config: &McConfig, n: usize) -> Vec<McResult> {
    let mut results = Vec::new();

    results.extend(validate_logit(config, n));
    results.extend(validate_probit(config, n));

    results
}

fn validate_logit(config: &McConfig, n: usize) -> Vec<McResult> {
    let mut results = Vec::new();
    let z_crit = 1.96;

    for (coef_idx, coef_name, true_val) in [
        (0, "intercept", -0.5),
        (1, "beta1", 1.0),
        (2, "beta2", -0.5),
    ] {
        let mut draws = Vec::with_capacity(config.n_sims);

        for sim in 0..config.n_sims {
            let seed = config.seed + sim as u64;
            let (dataset, _dgp) = dgp_logit(n, seed);

            if let Ok(result) = p2a_core::run_logit(&dataset, "y", &["x1", "x2"]) {
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

        let mut r = evaluate_coverage(&draws, true_val, config);
        r.method = format!("Logit_{}", coef_name);
        r.dgp = "logit_dgp".to_string();
        r.n = n;
        results.push(r);

        let mut r = evaluate_se_accuracy(&draws, config);
        r.method = format!("Logit_{}", coef_name);
        r.dgp = "logit_dgp".to_string();
        r.n = n;
        results.push(r);
    }

    results
}

fn validate_probit(config: &McConfig, n: usize) -> Vec<McResult> {
    let mut results = Vec::new();
    let z_crit = 1.96;

    for (coef_idx, coef_name, true_val) in [
        (0, "intercept", -0.5),
        (1, "beta1", 1.0),
        (2, "beta2", -0.5),
    ] {
        let mut draws = Vec::with_capacity(config.n_sims);

        for sim in 0..config.n_sims {
            let seed = config.seed + sim as u64;
            let (dataset, _dgp) = dgp_probit(n, seed);

            if let Ok(result) = p2a_core::run_probit(&dataset, "y", &["x1", "x2"]) {
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

        let mut r = evaluate_coverage(&draws, true_val, config);
        r.method = format!("Probit_{}", coef_name);
        r.dgp = "probit_dgp".to_string();
        r.n = n;
        results.push(r);

        let mut r = evaluate_se_accuracy(&draws, config);
        r.method = format!("Probit_{}", coef_name);
        r.dgp = "probit_dgp".to_string();
        r.n = n;
        results.push(r);
    }

    results
}
