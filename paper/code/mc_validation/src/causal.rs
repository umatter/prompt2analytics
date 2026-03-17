//! MC validation for causal inference methods.

use crate::dgp::*;
use crate::framework::*;
/// Run all causal inference MC validations.
pub fn validate_causal(config: &McConfig, n: usize) -> Vec<McResult> {
    let mut results = Vec::new();

    results.extend(validate_iv(config, n));
    results.extend(validate_did(config, n));
    results.extend(validate_ipw(config, n));
    results.extend(validate_tmle(config, n));

    results
}

fn validate_iv(config: &McConfig, n: usize) -> Vec<McResult> {
    let mut results = Vec::new();
    let z_crit = 1.96;

    let mut draws = Vec::with_capacity(config.n_sims);
    for sim in 0..config.n_sims {
        let seed = config.seed + sim as u64;
        let (dataset, dgp) = dgp_iv(n, seed);

        if let Ok(result) = p2a_core::run_iv2sls(
            &dataset, "y", &["x_exog"], &["x_endog"], &["instrument"], false,
        ) {
            // IV result: coefficients are [intercept, x_exog, x_endog]
            let coefs = &result.coefficients;
            let ses = &result.std_errors;
            // x_endog coefficient is at index 2
            let est = coefs[2];
            let se = ses[2];
            draws.push(EstimatorDraw {
                estimate: est,
                std_error: se,
                ci_lower: est - z_crit * se,
                ci_upper: est + z_crit * se,
            });
        }
    }

    let true_val = 0.8; // beta_endog

    let mut r = evaluate_coverage(&draws, true_val, config);
    r.method = "IV_2SLS_beta_endog".to_string();
    r.dgp = "endogenous_with_instrument".to_string();
    r.n = n;
    results.push(r);

    let mut r = evaluate_se_accuracy(&draws, config);
    r.method = "IV_2SLS_beta_endog".to_string();
    r.dgp = "endogenous_with_instrument".to_string();
    r.n = n;
    results.push(r);

    results
}

fn validate_did(config: &McConfig, n: usize) -> Vec<McResult> {
    let mut results = Vec::new();
    let z_crit = 1.96;

    let mut draws = Vec::with_capacity(config.n_sims);
    for sim in 0..config.n_sims {
        let seed = config.seed + sim as u64;
        let (dataset, dgp) = dgp_did(n, seed);

        if let Ok(result) = p2a_core::run_did(
            &dataset, "y", "treatment", "post", Some(&["x1"]),
        ) {
            let est = result.att;
            let se = result.std_error;
            draws.push(EstimatorDraw {
                estimate: est,
                std_error: se,
                ci_lower: est - z_crit * se,
                ci_upper: est + z_crit * se,
            });
        }
    }

    let true_att = 2.0;

    let mut r = evaluate_coverage(&draws, true_att, config);
    r.method = "DiD_ATT".to_string();
    r.dgp = "canonical_2x2".to_string();
    r.n = n;
    results.push(r);

    let mut r = evaluate_se_accuracy(&draws, config);
    r.method = "DiD_ATT".to_string();
    r.dgp = "canonical_2x2".to_string();
    r.n = n;
    results.push(r);

    results
}

fn validate_ipw(config: &McConfig, n: usize) -> Vec<McResult> {
    let mut results = Vec::new();
    let z_crit = 1.96;

    let mut draws = Vec::with_capacity(config.n_sims);
    for sim in 0..config.n_sims {
        let seed = config.seed + sim as u64;
        let (dataset, dgp) = dgp_treatment(n, seed);

        if let Ok(result) = p2a_core::run_ipw_treatment(
            &dataset, "y", "treatment", &["x1", "x2"],
            p2a_core::IpwConfig::default(),
        ) {
            let est = result.effect;
            let se = result.std_error;
            draws.push(EstimatorDraw {
                estimate: est,
                std_error: se,
                ci_lower: est - z_crit * se,
                ci_upper: est + z_crit * se,
            });
        }
    }

    let true_ate = 0.5;

    let mut r = evaluate_coverage(&draws, true_ate, config);
    r.method = "IPW_ATE".to_string();
    r.dgp = "binary_treatment_confounded".to_string();
    r.n = n;
    results.push(r);

    results
}

fn validate_tmle(config: &McConfig, n: usize) -> Vec<McResult> {
    let mut results = Vec::new();
    let z_crit = 1.96;

    let mut draws = Vec::with_capacity(config.n_sims);
    for sim in 0..config.n_sims {
        let seed = config.seed + sim as u64;
        let (dataset, dgp) = dgp_treatment(n, seed);

        let tmle_config = p2a_core::TmleConfig {
            q_model: p2a_core::econometrics::QModel::Linear,
            ..Default::default()
        };
        if let Ok(result) = p2a_core::tmle(
            &dataset, "y", "treatment", &["x1", "x2"],
            tmle_config,
        ) {
            let est = result.ate;
            let se = result.ate_se;
            draws.push(EstimatorDraw {
                estimate: est,
                std_error: se,
                ci_lower: est - z_crit * se,
                ci_upper: est + z_crit * se,
            });
        }
    }

    let true_ate = 0.5;

    let mut r = evaluate_coverage(&draws, true_ate, config);
    r.method = "TMLE_ATE".to_string();
    r.dgp = "binary_treatment_confounded".to_string();
    r.n = n;
    results.push(r);

    results
}
