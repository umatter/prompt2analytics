//! Hausman specification test for panel data, as `plm::phtest` (method "chisq").
//!
//! # Mathematical Background
//!
//! Tests H₀: RE is consistent (Cov(αᵢ, Xᵢₜ) = 0) vs H₁: FE is required.
//!
//! H = |d' D⁻ d|,  d = β̂ᶠᴱ - β̂ᴿᴱ,  D = V̂ᶠᴱ - V̂ᴿᴱ,  H ~ χ²(rank D)
//!
//! over the slopes, matched by name, with the classical FE and RE covariances
//! (phtest's default). D is eigen-decomposed after scaling by the FE standard
//! errors, so the rank decision does not depend on regressor units; when D is
//! invertible, H equals phtest's statistic exactly. D⁻ is a generalized inverse
//! when D is singular, the degrees of freedom are its rank, and the absolute
//! value is taken as phtest does when D is not positive definite (both reported
//! as warnings). A zero difference has no testable content and is refused.
//!
//! # References
//!
//! - Hausman, J.A. (1978). Specification tests in econometrics. *Econometrica*,
//!   46(6), 1251-1271. https://doi.org/10.2307/1913827
//!
//! R equivalent: `plm::phtest()`

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::traits::estimator::chi_squared_p_value;

use super::linear_models::{
    run_fixed_effects_with, run_random_effects_with, symmetric_eigen_pairs,
};
use super::types::{PanelConfig, PanelResult, PanelVcov};

/// The covariance the test uses, recorded in [`HausmanResult::vcov_method`].
const VCOV_METHOD: &str = "classical FE and RE covariance (plm::phtest default)";

/// Result from a Hausman specification test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HausmanResult {
    /// Chi-squared test statistic `|d' D⁻ d|`
    pub chi2_statistic: f64,
    /// P-value for the test
    pub p_value: f64,
    /// Degrees of freedom: the rank of `Var(b_FE) - Var(b_RE)`
    pub df: usize,
    /// Recommendation based on p-value
    pub recommendation: String,
    /// Fixed Effects estimation results
    pub fe_result: PanelResult,
    /// Random Effects estimation results
    pub re_result: PanelResult,
    /// Warnings about the covariance difference (not positive definite,
    /// singular), followed by the random-effects warnings prefixed with
    /// `random effects: `.
    #[serde(default)]
    pub warnings: Vec<String>,
    /// The covariances the statistic is built from.
    #[serde(default)]
    pub vcov_method: String,
}

impl fmt::Display for HausmanResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Hausman Specification Test")?;
        writeln!(f, "===========================================")?;
        writeln!(f, "H0: Random Effects is consistent and efficient")?;
        writeln!(f, "H1: Random Effects is inconsistent (use Fixed Effects)")?;
        writeln!(f)?;
        writeln!(f, "Chi2 Statistic: {:.4}", self.chi2_statistic)?;
        writeln!(f, "Degrees of Freedom: {}", self.df)?;
        writeln!(f, "P-Value: {:.4}", self.p_value)?;
        writeln!(f, "Covariance: {}", self.vcov_method)?;
        writeln!(f)?;

        // Coefficient comparison, each RE coefficient found by name
        writeln!(f, "Coefficient Comparison:")?;
        writeln!(
            f,
            "{:<20} {:>12} {:>12} {:>12}",
            "Variable", "FE Coef", "RE Coef", "Difference"
        )?;
        writeln!(f, "{:-<60}", "")?;
        for (name, &fe_coef) in self
            .fe_result
            .variables
            .iter()
            .zip(&self.fe_result.coefficients)
        {
            let re_coef = self
                .re_result
                .variables
                .iter()
                .position(|v| v == name)
                .and_then(|j| self.re_result.coefficients.get(j).copied());
            match re_coef {
                Some(re_coef) => writeln!(
                    f,
                    "{:<20} {:>12.4} {:>12.4} {:>12.4}",
                    name,
                    fe_coef,
                    re_coef,
                    fe_coef - re_coef
                )?,
                None => writeln!(f, "{:<20} {:>12.4} {:>12} {:>12}", name, fe_coef, "-", "-")?,
            }
        }
        writeln!(f)?;

        writeln!(f, "Result: {}", self.recommendation)?;
        if !self.warnings.is_empty() {
            writeln!(f, "Warnings:")?;
            for w in &self.warnings {
                writeln!(f, "  - {w}")?;
            }
        }

        // Add interpretation note
        writeln!(f)?;
        writeln!(
            f,
            "Note: The Hausman test compares FE and RE coefficient estimates."
        )?;
        writeln!(
            f,
            "A significant result (p < 0.05) suggests entity effects are correlated"
        )?;
        writeln!(
            f,
            "with regressors, making RE inconsistent. A non-significant result"
        )?;
        writeln!(
            f,
            "suggests RE is more efficient, but may have low power in small samples."
        )?;
        Ok(())
    }
}

/// The statistic, its degrees of freedom and the warnings about `D`.
#[derive(Debug)]
struct HausmanForm {
    statistic: f64,
    df: usize,
    warnings: Vec<String>,
}

/// `|d' D⁻ d|` for `D = (v_fe - v_re)` symmetrised, as `plm::phtest`.
///
/// D is scaled by the FE standard errors, `D_s = S D S` with
/// `S = diag(v_fe_jj^-1/2)`, and eigen-decomposed; eigenvalues with
/// `|λ| <= 100 k ε max(1, max|λ|)` are dropped. With every eigenvalue kept this
/// is `d' D⁻¹ d` exactly (`d' D⁻¹ d = d_s' D_s⁻¹ d_s`), and by Sylvester's law of
/// inertia `D_s` has the signs of `D`. The scaling makes the rank decision
/// independent of regressor units, and the FE-variance floor turns pure roundoff
/// into rank 0 instead of noise.
fn hausman_quadratic_form(
    d: &Array1<f64>,
    v_fe: &Array2<f64>,
    v_re: &Array2<f64>,
) -> EconResult<HausmanForm> {
    let k = d.len();
    if k == 0 || v_fe.dim() != (k, k) || v_re.dim() != (k, k) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Hausman test: need at least one slope and {k} x {k} covariance matrices \
                 (got {:?} and {:?})",
                v_fe.dim(),
                v_re.dim()
            ),
        });
    }
    let mut s = vec![0.0; k];
    for j in 0..k {
        let v = v_fe[[j, j]];
        if !(v.is_finite() && v > 0.0) {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "Hausman test: the fixed-effects variance of slope {} is {v}; it must be \
                     finite and positive",
                    j + 1
                ),
            });
        }
        s[j] = 1.0 / v.sqrt();
    }
    let d_s_mat = Array2::from_shape_fn((k, k), |(i, j)| {
        let dij = v_fe[[i, j]] - v_re[[i, j]];
        let dji = v_fe[[j, i]] - v_re[[j, i]];
        s[i] * s[j] * (dij + dji) / 2.0
    });
    let d_s = Array1::from_shape_fn(k, |j| s[j] * d[j]);

    let (lambda, u) = symmetric_eigen_pairs(&d_s_mat)?;
    let lambda_max = lambda.iter().fold(1.0f64, |m, l| m.max(l.abs()));
    let tol = 100.0 * k as f64 * f64::EPSILON * lambda_max;
    let rank = lambda.iter().filter(|l| l.abs() > tol).count();
    if rank == 0 {
        return Err(EconError::InvalidSpecification {
            message: "Var(b_FE) - Var(b_RE) is zero to machine precision for every slope: fixed \
                      and random effects coincide (for example, no regressor varies between \
                      entities), so the Hausman test has no testable difference"
                .to_string(),
        });
    }
    let q: f64 = lambda
        .iter()
        .enumerate()
        .filter(|(_, l)| l.abs() > tol)
        .map(|(j, &l)| u.column(j).dot(&d_s).powi(2) / l)
        .sum();

    let mut warnings = Vec::new();
    if lambda.iter().any(|&l| l < -tol) {
        warnings.push(
            "Var(b_FE) - Var(b_RE) is not positive definite; the statistic is |d' D^-1 d| as in \
             plm::phtest"
                .to_string(),
        );
    }
    if rank < k {
        warnings.push(format!(
            "Var(b_FE) - Var(b_RE) is singular; generalized inverse used, df = rank {rank}"
        ));
    }
    Ok(HausmanForm {
        statistic: q.abs(),
        df: rank,
        warnings,
    })
}

/// Run the Hausman specification test of Fixed Effects against Random Effects,
/// as `plm::phtest(fe, re)` (method "chisq").
///
/// Both models are estimated with their classical covariances, whatever the
/// caller's covariance preferences, as phtest does by default. The slopes are
/// matched by name (the RE intercept is skipped), and the statistic is
/// `|d' D⁻ d|` with `d = b_FE - b_RE` and `D = Var(b_FE) - Var(b_RE)`:
/// `D⁻` is the inverse, or a generalized inverse when D is singular, with the
/// degrees of freedom equal to the rank of D. A difference that is not positive
/// definite, or singular, is reported in `warnings`; a difference that is zero
/// for every slope is refused with `InvalidSpecification`.
///
/// # References
/// - Hausman, J.A. (1978). Specification tests in econometrics. Econometrica, 46(6), 1251-1271.
/// - Cameron, A.C. & Trivedi, P.K. (2005). Microeconometrics: Methods and Applications.
///   Cambridge University Press, Section 21.4.
pub fn run_hausman_test(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    entity_col: &str,
) -> EconResult<HausmanResult> {
    let config = PanelConfig {
        vcov: PanelVcov::Classical,
    };
    let fe_result = run_fixed_effects_with(dataset, y_col, x_cols, entity_col, &config)?;
    let re_result = run_random_effects_with(dataset, y_col, x_cols, entity_col, &config)?;

    // Slopes of the FE fit and their positions in the RE fit, matched by name
    let mut fe_idx = Vec::new();
    let mut re_idx = Vec::new();
    for (i, name) in fe_result.variables.iter().enumerate() {
        if name == "(Intercept)" {
            continue;
        }
        let j = re_result
            .variables
            .iter()
            .position(|v| v == name)
            .ok_or_else(|| {
                EconError::Internal(format!(
                    "Hausman test: slope '{name}' of the fixed-effects fit is missing from the \
                     random-effects fit"
                ))
            })?;
        fe_idx.push(i);
        re_idx.push(j);
    }
    let k = fe_idx.len();
    let d = Array1::from_shape_fn(k, |a| {
        fe_result.coefficients[fe_idx[a]] - re_result.coefficients[re_idx[a]]
    });
    let v_fe = Array2::from_shape_fn((k, k), |(a, b)| fe_result.vcov[[fe_idx[a], fe_idx[b]]]);
    let v_re = Array2::from_shape_fn((k, k), |(a, b)| re_result.vcov[[re_idx[a], re_idx[b]]]);

    let form = hausman_quadratic_form(&d, &v_fe, &v_re)?;
    let chi2_statistic = form.statistic;
    let df = form.df;
    let p_value = chi_squared_p_value(chi2_statistic, df as f64);

    let mut warnings = form.warnings;
    warnings.extend(
        re_result
            .warnings
            .iter()
            .map(|w| format!("random effects: {w}")),
    );

    // Construct nuanced recommendation based on p-value and sample size
    let n_entities = fe_result.n_groups;
    let n_obs = fe_result.n_obs;

    let recommendation = if p_value < 0.01 {
        "Reject H0: Use Fixed Effects (strong evidence of systematic difference in coefficients)"
            .to_string()
    } else if p_value < 0.05 {
        "Reject H0: Use Fixed Effects (moderate evidence of systematic difference in coefficients)"
            .to_string()
    } else if p_value < 0.10 {
        format!(
            "Marginally fail to reject H0 (p={:.3}). Consider Fixed Effects if correlation between \
             entity effects and regressors is theoretically plausible.",
            p_value
        )
    } else if n_entities < 20 || n_obs < 100 {
        format!(
            "Fail to reject H0 (p={:.3}), but note: Hausman test may have low power with \
             {} entities and {} observations. Consider theoretical arguments for model choice.",
            p_value, n_entities, n_obs
        )
    } else {
        "Fail to reject H0: Random Effects is consistent and more efficient".to_string()
    };

    Ok(HausmanResult {
        chi2_statistic,
        p_value,
        df,
        recommendation,
        fe_result,
        re_result,
        warnings,
        vcov_method: VCOV_METHOD.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-14 * b.abs()
    }

    /// d' D^-1 d with D = [[3, 0.5], [0.5, 2]] and d = (1, -2): 64/23.
    #[test]
    fn test_hausman_form_full_rank() {
        let f = hausman_quadratic_form(
            &array![1.0, -2.0],
            &array![[4.0, 1.0], [1.0, 3.0]],
            &array![[1.0, 0.5], [0.5, 1.0]],
        )
        .unwrap();
        assert!(close(f.statistic, 2.7826086956521738), "{}", f.statistic);
        assert_eq!(f.df, 2);
        assert!(f.warnings.is_empty(), "{:?}", f.warnings);
    }

    /// D = diag(-1, 0.5): the signed form is -2, phtest reports |q| = 2.
    #[test]
    fn test_hausman_form_indefinite_takes_absolute_value() {
        let f = hausman_quadratic_form(
            &array![2.0, 1.0],
            &array![[1.0, 0.0], [0.0, 1.0]],
            &array![[2.0, 0.0], [0.0, 0.5]],
        )
        .unwrap();
        assert!(close(f.statistic, 2.0), "{}", f.statistic);
        assert_eq!(f.df, 2);
        assert_eq!(f.warnings.len(), 1, "{:?}", f.warnings);
        assert!(f.warnings[0].contains("not positive definite"));
    }

    /// D = [[1, 1], [1, 1]] has rank 1; d = (1, 1) lies in its range: d' D^+ d = 1.
    #[test]
    fn test_hausman_form_singular_uses_rank() {
        let f = hausman_quadratic_form(
            &array![1.0, 1.0],
            &array![[2.0, 1.0], [1.0, 2.0]],
            &array![[1.0, 0.0], [0.0, 1.0]],
        )
        .unwrap();
        assert!(close(f.statistic, 1.0), "{}", f.statistic);
        assert_eq!(f.df, 1);
        assert_eq!(f.warnings.len(), 1, "{:?}", f.warnings);
        assert!(f.warnings[0].contains("df = rank 1"), "{:?}", f.warnings);
    }

    #[test]
    fn test_hausman_form_zero_difference_is_an_error() {
        let eye = array![[1.0, 0.0], [0.0, 1.0]];
        match hausman_quadratic_form(&array![1.0, 1.0], &eye, &eye) {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(message.contains("no testable difference"), "{message}")
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    /// The full-rank case with regressor 2 rescaled by 1e7. Unscaled, D has
    /// eigenvalues 3.0 and 1.9e-14, below a relative tolerance of 1.3e-13 on the
    /// unscaled matrix; scaled by the FE standard errors the rank is 2.
    #[test]
    fn test_hausman_form_units_do_not_matter() {
        let f = hausman_quadratic_form(
            &array![1.0, -2e-7],
            &array![[4.0, 1e-7], [1e-7, 3e-14]],
            &array![[1.0, 5e-8], [5e-8, 1e-14]],
        )
        .unwrap();
        assert!(close(f.statistic, 2.7826086956521738), "{}", f.statistic);
        assert_eq!(f.df, 2);
        assert!(f.warnings.is_empty(), "{:?}", f.warnings);
    }
}
