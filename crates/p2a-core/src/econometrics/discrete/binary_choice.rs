//! Binary choice models: Logit and Probit.
//!
//! Fitted by IRLS (Fisher scoring) as R's `glm.fit`, through the shared
//! [`crate::econometrics::glm::fit_glm`]: probit SEs come from the expected information, as
//! `summary.glm`; aliased regressors get `NaN` coefficients with R's "Coefficients not
//! defined because of singularities" warning; separation is reported (glm's flag, with a
//! sentence counting the extreme fitted probabilities, plus exact univariate complete and
//! quasi-complete checks). The outcome must be coded 0/1. The control is [`MleSettings`]
//! (`glm.control(epsilon, maxit)`), by default epsilon 1e-10 and 100 iterations.
//!
//! # Mathematical Background
//!
//! For binary outcomes y in {0, 1}, the latent variable model is:
//!
//! y*_i = X_i'beta + epsilon_i,  y_i = 1[y*_i > 0]
//!
//! ## Logit Model
//!
//! Assumes epsilon follows a logistic distribution:
//!
//! P(y_i = 1 | X_i) = Lambda(X_i'beta) = exp(X_i'beta) / (1 + exp(X_i'beta))
//!
//! ## Probit Model
//!
//! Assumes epsilon follows a standard normal distribution:
//!
//! P(y_i = 1 | X_i) = Phi(X_i'beta)
//!
//! where Phi is the standard normal CDF.
//!
//! # References
//!
//! - McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior.
//! - Train, K.E. (2009). *Discrete Choice Methods with Simulation* (2nd ed.).

use ndarray::Array2;

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, NaPolicy};
use crate::econometrics::glm::{GlmSpec, fit_glm};
use crate::errors::{EconError, EconResult};
use crate::linalg::solve::aliasing_message;
use crate::traits::convergence::Convergence;
use crate::traits::estimator::{
    SignificanceLevel, logistic_cdf, normal_p_value_two_sided, normal_pdf,
};

use super::types::{DiscreteModelType, DiscreteResult, MleSettings};

/// Run logit regression.
///
/// R equivalent: `stats::glm(y ~ x1 + x2, family = binomial(link = "logit"), data = df,
/// control = glm.control(epsilon = 1e-10, maxit = 100))`
///
/// Rows with a missing outcome or regressor are dropped (as `glm`'s `na.omit`) and
/// reported in `missing`. The outcome must be numeric and coded 0/1 (Int64 and Boolean
/// columns are accepted); a text outcome is `NonNumericColumn`, any other value is
/// `InvalidSpecification`.
pub fn run_logit(dataset: &Dataset, y_col: &str, x_cols: &[&str]) -> EconResult<DiscreteResult> {
    run_discrete_model(dataset, y_col, x_cols, DiscreteModelType::Logit, None)
}

/// Run probit regression.
///
/// R equivalent: `stats::glm(y ~ x1 + x2, family = binomial(link = "probit"), data = df,
/// control = glm.control(epsilon = 1e-10, maxit = 100))`
///
/// Missing values and outcome types as in [`run_logit`].
pub fn run_probit(dataset: &Dataset, y_col: &str, x_cols: &[&str]) -> EconResult<DiscreteResult> {
    run_discrete_model(dataset, y_col, x_cols, DiscreteModelType::Probit, None)
}

/// Exact univariate separation rules: for each non-constant column, with both outcome
/// groups non-empty, complete separation when the ranges of the column for y = 0 and y = 1
/// do not overlap, quasi-complete separation when they touch at one value. Returns the
/// names of the completely and of the quasi-completely separating columns.
fn univariate_separation(
    y: &[f64],
    x: &Array2<f64>,
    names: &[String],
) -> (Vec<String>, Vec<String>) {
    let mut complete = Vec::new();
    let mut quasi = Vec::new();
    for (j, name) in names.iter().enumerate().take(x.ncols()) {
        let col = x.column(j);
        let (mut lo0, mut hi0, mut lo1, mut hi1) = (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        );
        for (&v, &yi) in col.iter().zip(y) {
            if yi == 1.0 {
                lo1 = lo1.min(v);
                hi1 = hi1.max(v);
            } else {
                lo0 = lo0.min(v);
                hi0 = hi0.max(v);
            }
        }
        // an empty group leaves its bounds infinite; a constant column has lo == hi
        if !(lo0.is_finite() && lo1.is_finite()) || lo0.min(lo1) == hi0.max(hi1) {
            continue;
        }
        if hi0 < lo1 || hi1 < lo0 {
            complete.push(name.clone());
        } else if hi0 == lo1 || hi1 == lo0 {
            quasi.push(name.clone());
        }
    }
    (complete, quasi)
}

/// Run a discrete choice model (Logit or Probit) by IRLS as R's `glm.fit`.
///
/// `settings` is `glm.control(epsilon, maxit)` ([`MleSettings`]; default epsilon 1e-10,
/// 100 iterations). Refused ([`EconError::InvalidSpecification`]): `max_iter` 0, a
/// tolerance that is not a positive finite number, and an outcome with a value other than
/// 0 or 1. A regressor whose ranges for y = 0 and y = 1 do not overlap is
/// [`EconError::PerfectSeparation`]; ranges that touch give a quasi-separation warning.
/// Aliased regressors get `NaN` coefficients, SEs, z, p and marginal effects.
pub fn run_discrete_model(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    model_type: DiscreteModelType,
    settings: Option<MleSettings>,
) -> EconResult<DiscreteResult> {
    let settings = settings.unwrap_or_default();
    if settings.max_iter == 0 {
        return Err(EconError::InvalidSpecification {
            message: "max_iter must be at least 1 (got 0)".to_string(),
        });
    }
    if !(settings.tolerance.is_finite() && settings.tolerance > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "tolerance must be a positive, finite number (got {})",
                settings.tolerance
            ),
        });
    }
    let df = dataset.df();

    // y and X from the same rows: a row with a missing outcome or regressor is dropped
    // (R glm: na.omit). A text outcome is NonNumericColumn; Int64 and Boolean are read.
    let cols = ColumnRequest::new(df)
        .numeric(y_col)
        .numerics(x_cols)
        .na(NaPolicy::Omit)
        .extract()?;
    let n = cols.n();
    let y: Vec<f64> = cols.numeric(y_col).to_vec();

    // Outcome domain (R's glm refuses y outside [0, 1]; a proportion is not binary).
    let mut off = y.iter().filter(|&&v| v != 0.0 && v != 1.0);
    if let Some(first) = off.next() {
        let k = 1 + off.count();
        return Err(EconError::InvalidSpecification {
            message: format!(
                "{model_type}: the outcome '{y_col}' must be coded 0/1 (a binary outcome), but \
                 {k} of {n} values are not (first: {first}); R's glm refuses y outside [0, 1]"
            ),
        });
    }

    let x = cols.matrix(x_cols, true)?.data;
    let k = x.ncols();
    let mut var_names = vec!["(Intercept)".to_string()];
    var_names.extend(x_cols.iter().map(|s| s.to_string()));

    let (complete, quasi) = univariate_separation(&y, &x, &var_names);
    if !complete.is_empty() {
        return Err(EconError::PerfectSeparation {
            variables: complete,
        });
    }

    let mut warnings = Vec::new();
    if let Some(note) = cols.missing().note() {
        warnings.push(note);
    }
    if !quasi.is_empty() {
        warnings.push(format!(
            "Quasi-complete separation detected for: {}. Coefficients may be unstable.",
            quasi.join(", ")
        ));
    }

    let mut spec = match model_type {
        DiscreteModelType::Logit => GlmSpec::logit(),
        DiscreteModelType::Probit => GlmSpec::probit(),
    };
    spec.epsilon = settings.tolerance;
    spec.max_iter = settings.max_iter;
    let y_arr = ndarray::Array1::from(y);
    let fit = fit_glm(x.view(), y_arr.view(), &spec)?;

    warnings.extend(fit.warnings.iter().cloned());
    if let Some(msg) = aliasing_message(&fit.aliased, &var_names) {
        warnings.push(msg);
    }
    if fit.separation {
        let eps10 = 10.0 * f64::EPSILON;
        let extreme = fit
            .fitted
            .iter()
            .filter(|&&m| m < eps10 || m > 1.0 - eps10)
            .count();
        warnings.push(format!(
            "Separation: fitted probabilities are numerically 0 or 1 for {extreme} of {n} \
             observations; the regressors (almost) perfectly predict '{y_col}', so the maximum \
             likelihood estimates do not exist and the coefficients, standard errors and \
             p-values are not reliable"
        ));
    }
    let model_name = match model_type {
        DiscreteModelType::Logit => "logit",
        DiscreteModelType::Probit => "probit",
    };
    let convergence = Convergence::new(
        fit.converged,
        fit.iterations,
        settings.max_iter,
        "relative change in deviance (glm.fit)",
        f64::NAN,
        settings.tolerance,
    );
    if let Some(note) = convergence.note(model_name) {
        warnings.push(note);
    }

    let coefficients = fit.coefficients.to_vec();
    let std_errors = fit.std_errors.to_vec();
    let z_stats: Vec<f64> = coefficients
        .iter()
        .zip(&std_errors)
        .map(|(&b, &se)| {
            if se > 0.0 && se.is_finite() {
                b / se
            } else {
                f64::NAN
            }
        })
        .collect();
    let p_values: Vec<f64> = z_stats
        .iter()
        .map(|&z| normal_p_value_two_sided(z))
        .collect();

    // Marginal effects at the mean, over the identified columns.
    let eta_mean: f64 = (0..k)
        .filter(|j| coefficients[*j].is_finite())
        .map(|j| x.column(j).mean().unwrap_or(0.0) * coefficients[j])
        .sum();
    let density = match model_type {
        DiscreteModelType::Logit => {
            let p = logistic_cdf(eta_mean);
            p * (1.0 - p)
        }
        DiscreteModelType::Probit => normal_pdf(eta_mean),
    };
    let marginal_effects: Vec<f64> = coefficients.iter().map(|&b| b * density).collect();

    let log_likelihood = fit.log_likelihood;
    // the saturated log-likelihood of a 0/1 outcome is 0
    let log_likelihood_null = -fit.null_deviance / 2.0;
    let pseudo_r_squared = 1.0 - log_likelihood / log_likelihood_null;
    let bic = -2.0 * log_likelihood + fit.rank as f64 * (n as f64).ln();
    let significance: Vec<SignificanceLevel> = p_values
        .iter()
        .map(|&p| SignificanceLevel::from_p_value(p))
        .collect();
    let n_positive = y_arr.iter().filter(|&&yi| yi == 1.0).count();
    let aliased = fit.aliased.iter().map(|&j| var_names[j].clone()).collect();

    Ok(DiscreteResult {
        model_type,
        dep_var: y_col.to_string(),
        variables: var_names,
        coefficients,
        std_errors,
        z_stats,
        p_values,
        significance,
        marginal_effects,
        log_likelihood,
        log_likelihood_null,
        pseudo_r_squared,
        aic: fit.aic,
        bic,
        n_obs: n,
        n_positive,
        iterations: fit.iterations,
        converged: fit.converged,
        warnings,
        missing: cols.missing().clone(),
        aliased,
        separation: fit.separation,
        convergence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use p2a_validation::{Check, Tol};
    use polars::prelude::*;

    /// Case data for tests that compare no R value.
    fn case_data(id: &str) -> (p2a_validation::Case, Dataset) {
        crate::test_support::load_case(id)
    }

    /// A missing outcome is dropped, as glm's na.omit does; it was read as 0.
    #[test]
    fn test_validate_logit_na_outcome_dropped() {
        let (case, ds) = crate::test_support::load_case("disc_choice_na");
        let r = run_logit(&ds, "choice_na", &["x", "w"]).unwrap();
        Check::new(&case)
            .named(
                "logit_y_coef",
                &r.variables,
                &r.coefficients,
                Tol::iterative(1e-6),
            )
            .named(
                "logit_y_se",
                &r.variables,
                &r.std_errors,
                Tol::iterative(1e-5),
            )
            .scalar("logit_y_loglik", r.log_likelihood, Tol::stat(298))
            .scalar("logit_y_nobs", r.n_obs as f64, Tol::exact())
            .finish();
        assert_eq!(r.missing.n_total, 300);
        assert_eq!(r.missing.n_dropped, 2);
        assert_eq!(r.missing.by_column, vec![("choice_na".to_string(), 2)]);
        let text = r.to_string();
        assert_eq!(
            text.lines().last().unwrap(),
            "Missing values: 2 of 300 rows dropped because of missing values (choice_na: 2)"
        );
        assert_eq!(text.matches("2 of 300 rows dropped").count(), 1, "{text}");
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("2 of 300 rows dropped")),
            "{:?}",
            r.warnings
        );
    }

    /// Missing values in the outcome and in a regressor: the incomplete rows are dropped
    /// from y and X together (glm on 297 rows).
    #[test]
    fn test_validate_logit_na_rows_dropped() {
        let (case, ds) = crate::test_support::load_case("disc_choice_na");
        let r = run_logit(&ds, "choice_na", &["x_na", "w"]).unwrap();
        Check::new(&case)
            .named(
                "logit_coef",
                &r.variables,
                &r.coefficients,
                Tol::iterative(1e-6),
            )
            .named(
                "logit_se",
                &r.variables,
                &r.std_errors,
                Tol::iterative(1e-5),
            )
            .scalar("logit_loglik", r.log_likelihood, Tol::stat(297))
            .scalar("logit_nobs", r.n_obs as f64, Tol::exact())
            .finish();
        assert_eq!(
            r.missing.by_column,
            vec![("choice_na".to_string(), 2), ("x_na".to_string(), 1)]
        );
        assert_eq!(r.missing.n_dropped, 3);
    }

    /// The probit counterpart of `test_validate_logit_na_rows_dropped`.
    #[test]
    fn test_validate_probit_na_rows_dropped() {
        let (case, ds) = crate::test_support::load_case("disc_choice_na");
        let r = run_probit(&ds, "choice_na", &["x_na", "w"]).unwrap();
        Check::new(&case)
            .named(
                "probit_coef",
                &r.variables,
                &r.coefficients,
                Tol::iterative(1e-6),
            )
            .named(
                "probit_se",
                &r.variables,
                &r.std_errors,
                Tol::iterative(1e-5),
            )
            .scalar("probit_loglik", r.log_likelihood, Tol::stat(297))
            .scalar("probit_nobs", r.n_obs as f64, Tol::exact())
            .finish();
        assert_eq!(
            r.missing.by_column,
            vec![("choice_na".to_string(), 2), ("x_na".to_string(), 1)]
        );
    }

    /// A text outcome is refused; a non-strict cast made every value null, then 0.
    #[test]
    fn test_na_logit_text_outcome_refused() {
        let (_, ds) = case_data("disc_choice_na");
        match run_logit(&ds, "alt", &["x"]) {
            Err(EconError::NonNumericColumn { column }) => assert_eq!(column, "alt"),
            other => panic!("expected NonNumericColumn, got {other:?}"),
        }
    }

    #[test]
    fn test_na_logit_absent_column() {
        let (_, ds) = case_data("disc_choice_na");
        for (y, x) in [("nope", "x"), ("choice", "nope")] {
            match run_logit(&ds, y, &[x]) {
                Err(EconError::ColumnNotFound { column, .. }) => assert_eq!(column, "nope"),
                other => panic!("expected ColumnNotFound, got {other:?}"),
            }
        }
    }

    fn create_binary_dataset() -> Dataset {
        let df = df! {
            "y" => [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            "x" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_logit_basic() {
        let dataset = create_binary_dataset();
        let result = run_logit(&dataset, "y", &["x"]).unwrap();

        assert_eq!(result.n_obs, 10);
        assert!(!result.variables.is_empty());

        let x_idx = result.variables.iter().position(|v| v == "x").unwrap();
        assert!(
            result.coefficients[x_idx] > 0.0,
            "Logit coefficient on x should be positive"
        );
    }

    #[test]
    fn test_probit_basic() {
        let dataset = create_binary_dataset();
        let result = run_probit(&dataset, "y", &["x"]).unwrap();

        assert_eq!(result.n_obs, 10);

        let x_idx = result.variables.iter().position(|v| v == "x").unwrap();
        assert!(
            result.coefficients[x_idx] > 0.0,
            "Probit coefficient on x should be positive"
        );
    }

    #[test]
    fn test_perfect_separation_detection() {
        let df = df! {
            "y" => [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "x" => [1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_logit(&dataset, "y", &["x"]);
        assert!(result.is_err(), "Should detect perfect separation");
    }

    // ════════════════════════════════════════════════════════════════════════════
    // R pins: stats::glm(binomial) on disc_binary_glm (validation/r/cases/discrete_mlogit_asc.R)
    // ════════════════════════════════════════════════════════════════════════════

    /// `names` and `v` without the entries named in `drop` (summary.glm drops NA rows).
    fn names_except(r: &DiscreteResult, drop: &[&str], v: &[f64]) -> (Vec<String>, Vec<f64>) {
        r.variables
            .iter()
            .zip(v)
            .filter(|(n, _)| !drop.contains(&n.as_str()))
            .map(|(n, x)| (n.clone(), *x))
            .unzip()
    }

    /// The logit at p2a's control (epsilon 1e-10) is the MLE to about 1e-13: glm's
    /// coefficients, SEs, z, p, logLik, AIC and the tight refit.
    #[test]
    fn test_validate_logit_binary_glm() {
        let (case, ds) = crate::test_support::load_case("disc_binary_glm");
        let r = run_logit(&ds, "y", &["x1", "x2", "d"]).unwrap();
        let n = r.n_obs;
        Check::new(&case)
            .scalar("n", n as f64, Tol::exact())
            .named("logit_coef", &r.variables, &r.coefficients, Tol::coef(n))
            .named("logit_se", &r.variables, &r.std_errors, Tol::se(n))
            .named("logit_z", &r.variables, &r.z_stats, Tol::stat(n))
            .named("logit_p", &r.variables, &r.p_values, Tol::p_value(n))
            .scalar("logit_loglik", r.log_likelihood, Tol::stat(n))
            .scalar("logit_aic", r.aic, Tol::stat(n))
            .scalar("logit_null_loglik", r.log_likelihood_null, Tol::stat(n))
            .named(
                "logit_coef_tight",
                &r.variables,
                &r.coefficients,
                Tol::iterative(1e-8),
            )
            .finish();
        assert!(r.converged && r.aliased.is_empty() && !r.separation && r.warnings.is_empty());
        assert!(r.to_string().contains("Converged: yes ("), "{r}");
    }

    /// The probit MLE (glm at epsilon 1e-14), and the default control against it: Fisher
    /// scoring converges linearly for the probit link, so glm's rule at 1e-10 stops about
    /// 1.4e-7 from the MLE (R too).
    #[test]
    fn test_validate_probit_binary_glm() {
        let (case, ds) = crate::test_support::load_case("disc_binary_glm");
        let mut s = MleSettings::default();
        s.tolerance = 1e-14;
        s.max_iter = 100;
        let t = run_discrete_model(
            &ds,
            "y",
            &["x1", "x2", "d"],
            DiscreteModelType::Probit,
            Some(s),
        )
        .unwrap();
        let n = t.n_obs;
        Check::new(&case)
            .named(
                "probit_coef_tight",
                &t.variables,
                &t.coefficients,
                Tol::coef(n),
            )
            .named("probit_se_tight", &t.variables, &t.std_errors, Tol::se(n))
            .named("probit_z_tight", &t.variables, &t.z_stats, Tol::stat(n))
            .named("probit_p_tight", &t.variables, &t.p_values, Tol::p_value(n))
            .scalar("probit_loglik_tight", t.log_likelihood, Tol::stat(n))
            .scalar("probit_aic_tight", t.aic, Tol::stat(n))
            .finish();
        assert!(t.converged, "{:?}", t.warnings);
        let r = run_probit(&ds, "y", &["x1", "x2", "d"]).unwrap();
        Check::new(&case)
            .named(
                "probit_coef_tight",
                &r.variables,
                &r.coefficients,
                Tol::iterative(1e-6),
            )
            .finish();
        assert!(r.converged && r.warnings.is_empty());
    }

    /// x1b = 2 x1 and a dummy trap: R's NA coefficients and message; the other
    /// coefficients, the SEs and the AIC (which counts the rank) are the full-rank fit's.
    #[test]
    fn test_validate_logit_aliased_is_na() {
        let (case, ds) = crate::test_support::load_case("disc_binary_glm");
        let r = run_logit(&ds, "y", &["x1", "x1b", "x2", "d"]).unwrap();
        assert_eq!(r.aliased, vec!["x1b".to_string()]);
        assert!(
            r.coefficients[2].is_nan()
                && r.std_errors[2].is_nan()
                && r.z_stats[2].is_nan()
                && r.p_values[2].is_nan()
        );
        assert!(r.marginal_effects[2].is_nan());
        assert!(r.marginal_effects[1].is_finite());
        assert!(
            r.warnings
                .iter()
                .any(|w| w == "Coefficients not defined because of singularities: x1b"),
            "{:?}",
            r.warnings
        );
        let (sn, sv) = names_except(&r, &["x1b"], &r.std_errors);
        Check::new(&case)
            .named(
                "aliased_coef",
                &r.variables,
                &r.coefficients,
                Tol::coef(400),
            )
            .named("aliased_se", &sn, &sv, Tol::se(400))
            .scalar(
                "aliased_rank",
                (r.variables.len() - r.aliased.len()) as f64,
                Tol::exact(),
            )
            .scalar("logit_aic", r.aic, Tol::stat(400))
            .finish();
        let t = run_logit(&ds, "y", &["x1", "x2", "d", "nd"]).unwrap();
        assert_eq!(t.aliased, vec!["nd".to_string()]);
        let (tn, tv) = names_except(&t, &["nd"], &t.std_errors);
        Check::new(&case)
            .named("trap_coef", &t.variables, &t.coefficients, Tol::coef(400))
            .named("trap_se", &tn, &tv, Tol::se(400))
            .finish();
        assert!(r.converged && t.converged);
    }

    /// ysep = 1[x1 + x2 > 0.5] is separated by x1 + x2 and by neither alone: no MLE. The
    /// test checks glm's flag and warnings, p2a's sentence and a floor on its count, and
    /// that a fit stopped early says so; iteration counts and coefficients are not pinned.
    #[test]
    fn test_validate_logit_multivariate_separation_warns() {
        let (case, ds) = crate::test_support::load_case("disc_binary_glm");
        assert_eq!(case.scalar("n"), 400.0);
        // (a) p2a's control: most fitted values reach glm's clamp before its criterion is
        // met (R: 397 of 400 at this control)
        let r = run_logit(&ds, "ysep", &["x1", "x2"]).unwrap();
        assert!(r.separation);
        assert!(
            r.warnings
                .iter()
                .any(|w| w == "glm.fit: fitted probabilities numerically 0 or 1 occurred")
        );
        let sentence = r
            .warnings
            .iter()
            .find(|w| w.starts_with("Separation:"))
            .unwrap_or_else(|| panic!("{:?}", r.warnings));
        let k: usize = sentence
            .split("for ")
            .nth(1)
            .and_then(|t| t.split(" of 400 observations").next())
            .and_then(|t| t.parse().ok())
            .unwrap_or_else(|| panic!("{sentence}"));
        assert!(k >= 200, "{sentence}");
        assert!(
            r.to_string()
                .contains("Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred")
        );
        // (b) stopped after 5 IRLS steps: the deviance still falls by a large fraction per
        // step, so the fit is reported as not converged
        let mut s = MleSettings::default();
        s.max_iter = 5;
        s.tolerance = 1e-10;
        let d = run_discrete_model(
            &ds,
            "ysep",
            &["x1", "x2"],
            DiscreteModelType::Logit,
            Some(s),
        )
        .unwrap();
        assert!(!d.converged);
        assert!(
            d.warnings
                .iter()
                .any(|w| w == "glm.fit: algorithm did not converge")
        );
        assert!(d.to_string().contains("Converged: no (5 iterations"), "{d}");
        // (c) R's default glm.control(epsilon = 1e-8, maxit = 25): R warns too
        let mut s = MleSettings::default();
        s.max_iter = 25;
        s.tolerance = 1e-8;
        let d = run_discrete_model(
            &ds,
            "ysep",
            &["x1", "x2"],
            DiscreteModelType::Logit,
            Some(s),
        )
        .unwrap();
        assert!(d.separation);
        assert!(
            d.warnings
                .iter()
                .any(|w| w == "glm.fit: fitted probabilities numerically 0 or 1 occurred")
        );
    }

    /// ECON-60: x1 and xnc differ by 1e-7 noise. Not aliased at glm.fit's QR tolerance;
    /// the old Newton/Armijo loop stopped "converged" at the optimum with the two
    /// coefficients forced equal (logLik -237.0064 against glm's -236.9845).
    #[test]
    fn test_validate_logit_near_collinear_reaches_optimum() {
        let (case, ds) = crate::test_support::load_case("disc_binary_glm");
        let r = run_logit(&ds, "y", &["x1", "xnc", "x2"]).unwrap();
        assert!(r.converged && r.aliased.is_empty(), "{:?}", r.warnings);
        Check::new(&case)
            .scalar("nc_loglik_tight", r.log_likelihood, Tol::stat(400))
            // condition number ~1e7: glm at epsilon 1e-10 is 1.2e-7 from the tight fit
            .named(
                "nc_coef_tight",
                &r.variables,
                &r.coefficients,
                Tol::iterative(1e-6),
            )
            .finish();
        assert!(r.log_likelihood > case.scalar("nc_restricted_loglik") + 0.02);
    }

    /// ECON-58: an outcome outside {0, 1} is refused (R's glm: "y values must be 0 <= y
    /// <= 1"); a proportion, which glm fits with a warning, is not a binary outcome here.
    #[test]
    fn test_validate_logit_rejects_y_outside_0_1() {
        let (case, ds) = crate::test_support::load_case("disc_binary_glm");
        assert_eq!(case.scalar("y012_refused"), 1.0);
        for (label, res) in [
            ("logit", run_logit(&ds, "y012", &["x1"])),
            ("probit", run_probit(&ds, "y012", &["x1"])),
        ] {
            match res {
                Err(EconError::InvalidSpecification { message }) => assert!(
                    message.contains("'y012'")
                        && message.contains("0/1")
                        && message.contains("10 of 400")
                        && message.contains("first: 2"),
                    "{label}: {message}"
                ),
                other => panic!("{label}: expected InvalidSpecification, got {other:?}"),
            }
        }
        let df = df! {
            "y" => [0.0, 0.5, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            "x" => [1.2, 0.4, 2.2, -0.3, 1.9, 0.8, -1.1, 0.1]
        }
        .unwrap();
        match run_logit(&Dataset::new(df), "y", &["x"]) {
            Err(EconError::InvalidSpecification { message }) => assert!(
                message.contains("1 of 8") && message.contains("first: 0.5"),
                "{message}"
            ),
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    /// Every d = 1 has yq = 1: no MLE for d's coefficient. The univariate rule warns; as
    /// d's coefficient grows, the identified coefficients and their SEs converge to the
    /// logit on the d = 0 rows (`quasi_limit_*`), which is what is pinned.
    #[test]
    fn test_validate_logit_quasi_separated_dummy_warns() {
        let (case, ds) = crate::test_support::load_case("disc_binary_glm");
        let r = run_logit(&ds, "yq", &["x1", "x2", "d"]).unwrap();
        assert!(
            r.warnings
                .iter()
                .any(|w| w
                    == "Quasi-complete separation detected for: d. Coefficients may be unstable."),
            "{:?}",
            r.warnings
        );
        let identified = ["(Intercept)", "x1", "x2"];
        Check::new(&case)
            .named_subset(
                "quasi_limit_coef",
                &identified,
                &r.variables,
                &r.coefficients,
                Tol::iterative(1e-6),
            )
            .named_subset(
                "quasi_limit_se",
                &identified,
                &r.variables,
                &r.std_errors,
                Tol::iterative(1e-5),
            )
            .finish();
        // R stops at 25.47 (SE 9394)
        assert!(r.coefficients[3] > 10.0 && r.std_errors[3] > 100.0);
    }

    #[test]
    fn test_mle_settings_reject_invalid_controls() {
        let ds = create_binary_dataset();
        for (max_iter, tolerance, needle) in [
            (0, 1e-10, "max_iter"),
            (100, 0.0, "tolerance"),
            (100, f64::NAN, "tolerance"),
            (100, -1e-8, "tolerance"),
        ] {
            let mut s = MleSettings::default();
            s.max_iter = max_iter;
            s.tolerance = tolerance;
            match run_discrete_model(&ds, "y", &["x"], DiscreteModelType::Logit, Some(s)) {
                Err(EconError::InvalidSpecification { message }) => {
                    assert!(message.contains(needle), "{message}")
                }
                other => panic!("expected InvalidSpecification naming {needle}, got {other:?}"),
            }
        }
    }
}
