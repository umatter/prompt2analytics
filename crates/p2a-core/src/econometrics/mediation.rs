//! Causal Mediation Analysis using Inverse Probability Weighting.
//!
//! Implements methods from Huber (2014) for decomposing treatment effects
//! into natural direct and indirect effects.
//!
//! # Overview
//!
//! Mediation analysis answers: "How much of the treatment effect operates
//! through a particular mediator variable?"
//!
//! # Key Concepts
//!
//! - **Total Effect (ATE)**: Overall effect of treatment on outcome
//! - **Natural Direct Effect (NDE)**: Effect NOT through the mediator
//! - **Natural Indirect Effect (NIE)**: Effect through the mediator
//! - **Decomposition**: ATE = NDE + NIE
//!
//! # Estimation
//!
//! Both propensity models, `p(D = 1 | X)` and `p(D = 1 | X, M)`, are logistic regressions
//! fitted by the shared GLM (`glm.fit`'s IRLS); their warnings (non-convergence, fitted
//! probabilities numerically 0 or 1, aliased covariates) are reported in
//! `MediationResult::warnings`. Units whose scores lie outside `(trim, 1 - trim)` are
//! dropped; the effects are Hajek-normalised IPW means. Standard errors come from a
//! bootstrap over the kept units (seed [`DEFAULT_BOOTSTRAP_SEED`] when none is given),
//! with type-7 percentile intervals; failed replicates are counted and reported. The
//! proportion mediated is `NIE / ATE`, never clamped, with a warning outside [0, 1].
//!
//! # Missing values and column types
//!
//! Rows with a missing value (null or NaN) in the outcome, treatment, mediator or a
//! covariate are dropped before estimation and reported in `MediationResult::missing`;
//! integer and boolean columns are read as numbers.
//!
//! # References
//!
//! - Huber, M. (2014). "Identifying causal mechanisms (primarily) based on
//!   inverse probability weighting." Journal of Applied Econometrics, 29, 920-943.
//! - Imai, K., Keele, L., & Tingley, D. (2010). "A General Approach to Causal
//!   Mediation Analysis." Psychological Methods, 15(4), 309-334.

use ndarray::{Array2, ArrayView1};
use rand::SeedableRng;
use rand::prelude::*;
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, MissingSummary, NaPolicy};
use crate::econometrics::glm::{GlmLink, GlmSpec, fit_propensity};
use crate::econometrics::medflex::{bootstrap_notes, glm_fit_notes, proportion_note};
use crate::errors::{EconError, EconResult};
use crate::resource;
use crate::stats::robust::percentile_interval;
use crate::traits::convergence::{ResultNotes, notes_text};
use crate::traits::estimator::normal_p_value_two_sided;

/// Seed of the bootstrap when `MediationConfig::seed` is `None`, so that repeated calls
/// give identical results.
pub const DEFAULT_BOOTSTRAP_SEED: u64 = 42;

/// Fewest usable bootstrap replicates from which SEs and intervals are reported.
const MIN_VALID_BOOTSTRAP: usize = 2;

/// Configuration for mediation analysis.
#[derive(Debug, Clone)]
pub struct MediationConfig {
    /// Number of bootstrap replications for standard errors (default: 999). Must be at
    /// least 2; a count whose result buffers cannot be allocated is refused with
    /// `EconError::ResourceLimit`.
    pub bootstrap: usize,
    /// Trimming threshold for propensity scores, in [0, 0.5) (default: 0.05)
    pub trim: f64,
    /// Bootstrap seed; `None` uses [`DEFAULT_BOOTSTRAP_SEED`]
    pub seed: Option<u64>,
}

impl Default for MediationConfig {
    fn default() -> Self {
        Self {
            bootstrap: 999,
            trim: 0.05,
            seed: None,
        }
    }
}

/// Result from causal mediation analysis.
#[derive(Debug, Clone)]
pub struct MediationResult {
    /// Total effect (ATE)
    pub total_effect: f64,
    /// Natural direct effect (NDE)
    pub direct_effect: f64,
    /// Natural indirect effect (NIE)
    pub indirect_effect: f64,
    /// Proportion of total effect mediated (NIE / ATE); never clamped, `NaN` at a zero
    /// total effect
    pub proportion_mediated: f64,
    /// Standard error of total effect
    pub se_total: f64,
    /// Standard error of direct effect
    pub se_direct: f64,
    /// Standard error of indirect effect
    pub se_indirect: f64,
    /// 95% CI for total effect (type-7 bootstrap percentiles)
    pub ci_total: (f64, f64),
    /// 95% CI for direct effect
    pub ci_direct: (f64, f64),
    /// 95% CI for indirect effect
    pub ci_indirect: (f64, f64),
    /// p-value for total effect
    pub p_total: f64,
    /// p-value for direct effect
    pub p_direct: f64,
    /// p-value for indirect effect
    pub p_indirect: f64,
    /// Number of observations used
    pub n_obs: usize,
    /// Number of complete rows trimmed for extreme propensity scores
    pub n_trimmed: usize,
    /// Rows dropped because of missing values before the propensity models were fitted.
    pub missing: MissingSummary,
    /// Warnings: propensity-model fit problems, failed or warned bootstrap replicates, a
    /// proportion mediated outside [0, 1].
    pub warnings: Vec<String>,
}

impl ResultNotes for MediationResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

impl fmt::Display for MediationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Causal Mediation Analysis ===")?;
        writeln!(f)?;
        writeln!(f, "Effect Decomposition:")?;
        writeln!(
            f,
            "─────────────────────────────────────────────────────────────"
        )?;
        writeln!(
            f,
            "{:<20} {:>12} {:>12} {:>12} {:>10}",
            "Effect", "Estimate", "Std.Err", "95% CI", "p-value"
        )?;
        writeln!(
            f,
            "─────────────────────────────────────────────────────────────"
        )?;

        let stars = |p: f64| -> &str {
            if p < 0.01 {
                "***"
            } else if p < 0.05 {
                "**"
            } else if p < 0.10 {
                "*"
            } else {
                ""
            }
        };

        writeln!(
            f,
            "{:<20} {:>12.4} {:>12.4} [{:>5.3},{:>5.3}] {:>8.4}{}",
            "Total (ATE)",
            self.total_effect,
            self.se_total,
            self.ci_total.0,
            self.ci_total.1,
            self.p_total,
            stars(self.p_total)
        )?;

        writeln!(
            f,
            "{:<20} {:>12.4} {:>12.4} [{:>5.3},{:>5.3}] {:>8.4}{}",
            "Direct (NDE)",
            self.direct_effect,
            self.se_direct,
            self.ci_direct.0,
            self.ci_direct.1,
            self.p_direct,
            stars(self.p_direct)
        )?;

        writeln!(
            f,
            "{:<20} {:>12.4} {:>12.4} [{:>5.3},{:>5.3}] {:>8.4}{}",
            "Indirect (NIE)",
            self.indirect_effect,
            self.se_indirect,
            self.ci_indirect.0,
            self.ci_indirect.1,
            self.p_indirect,
            stars(self.p_indirect)
        )?;

        writeln!(
            f,
            "─────────────────────────────────────────────────────────────"
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "Proportion Mediated: {:.1}%",
            self.proportion_mediated * 100.0
        )?;
        writeln!(f, "  (NIE / Total Effect)")?;
        writeln!(f)?;
        writeln!(f, "N = {} (trimmed: {})", self.n_obs, self.n_trimmed)?;
        writeln!(f)?;
        writeln!(f, "Signif. codes: *** p<0.01, ** p<0.05, * p<0.10")?;
        f.write_str(&notes_text(self))?;
        if let Some(note) = self.missing.note() {
            writeln!(f, "Missing values: {note}")?;
        }

        Ok(())
    }
}

/// Run causal mediation analysis.
///
/// Estimates the natural direct effect (NDE) and natural indirect effect (NIE)
/// using inverse probability weighting, following Huber (2014).
///
/// # Arguments
///
/// * `dataset` - The dataset containing all variables
/// * `outcome` - Name of the outcome variable column
/// * `treatment` - Name of the treatment variable column (binary 0/1)
/// * `mediator` - Name of the mediator variable column
/// * `covariates` - Names of covariate columns for adjustment
/// * `config` - Configuration options
///
/// # Returns
///
/// `MediationResult` containing effect estimates, standard errors, and p-values
///
/// # Mathematical Background
///
/// The total effect decomposes as:
/// ```text
/// ATE = E[Y(1,M(1))] - E[Y(0,M(0))] = NDE + NIE
/// ```
///
/// Where:
/// - NDE = E[Y(1,M(0))] - E[Y(0,M(0))]: effect of treatment holding mediator at control level
/// - NIE = E[Y(1,M(1))] - E[Y(1,M(0))]: effect of mediator change under treatment
///
/// # Key Assumptions
///
/// 1. **Sequential ignorability**: Treatment assignment and mediator are ignorable
///    given observed covariates
/// 2. **No treatment-mediator interaction** (for simple decomposition)
/// 3. **Positivity**: All units have positive probability of treatment and mediator values
///
/// # Missing values and column types
///
/// Rows with a missing value (null or NaN) in the outcome, treatment, mediator or a
/// covariate are dropped before estimation and reported in `missing`; integer and boolean
/// columns are read as numbers. `n_obs + n_trimmed` is the number of complete rows.
///
/// # Estimation details
///
/// The propensity models are logistic GLMs (`glm.fit` IRLS, [`fit_propensity`]); the
/// treatment must be coded 0/1. `trim` must lie in [0, 0.5). The bootstrap resamples the
/// kept units (no further trimming), uses seed [`DEFAULT_BOOTSTRAP_SEED`] when `seed` is
/// `None`, reports failed replicates in `warnings` and needs at least 2 usable ones; the
/// intervals are type-7 percentile intervals and the p-values two-sided normal ones. The
/// proportion mediated is `NIE / ATE`, unclamped.
pub fn run_mediation_analysis(
    dataset: &Dataset,
    outcome: &str,
    treatment: &str,
    mediator: &str,
    covariates: &[&str],
    config: MediationConfig,
) -> EconResult<MediationResult> {
    if !(config.trim.is_finite() && (0.0..0.5).contains(&config.trim)) {
        return Err(EconError::InvalidSpecification {
            message: format!("trim must be in [0, 0.5); got {}", config.trim),
        });
    }
    if config.bootstrap < 2 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "bootstrap must be at least 2 (the standard errors come from the bootstrap); got {}",
                config.bootstrap
            ),
        });
    }

    // One request over every model variable: R fits both propensity models on the
    // rows complete in all of them (lm/glm na.omit; mediation::mediate refuses models
    // fitted on different rows). Integer and boolean columns are read as numbers.
    let cols = ColumnRequest::new(dataset.df())
        .numeric(outcome)
        .numeric(treatment)
        .numeric(mediator)
        .numerics(covariates)
        .na(NaPolicy::Omit)
        .min_rows(1, "mediation analysis")
        .extract()?;
    let n = cols.n();
    let y = cols.numeric(outcome);
    let d = cols.numeric(treatment);
    let m = cols.numeric(mediator);
    // "(Intercept)" then the covariates, built row by row (CAUS-25)
    let x = cols.matrix(covariates, true)?.data;
    let k = x.ncols();
    let mut names: Vec<String> = Vec::with_capacity(k + 1);
    names.push("(Intercept)".to_string());
    names.extend(covariates.iter().map(|c| c.to_string()));
    names.push(mediator.to_string());

    // Compute point estimates
    let (total, direct, indirect, keep_idx, fit_notes) =
        compute_mediation_effects(y, d, m, &x, config.trim, &names)?;

    let n_obs = keep_idx.len();
    let n_trimmed = n - n_obs;

    if n_obs < 20 {
        return Err(EconError::InsufficientData {
            required: 20,
            provided: n_obs,
            context: "Too few observations after trimming for mediation analysis".to_string(),
        });
    }

    // Bootstrap for standard errors
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed.unwrap_or(DEFAULT_BOOTSTRAP_SEED));

    let what = "bootstrap replications";
    let mut boot_total = resource::try_with_capacity::<f64>(config.bootstrap, what)?;
    let mut boot_direct = resource::try_with_capacity::<f64>(config.bootstrap, what)?;
    let mut boot_indirect = resource::try_with_capacity::<f64>(config.bootstrap, what)?;
    let (mut failed, mut warned) = (0usize, 0usize);
    let (mut first_error, mut first_warning): (Option<String>, Option<String>) = (None, None);
    let mut boot_indices = resource::try_vec(n_obs, 0usize, "bootstrap resample rows")?;
    let mut y_boot = resource::try_vec(n_obs, 0.0, "bootstrap resample")?;
    let mut d_boot = resource::try_vec(n_obs, 0.0, "bootstrap resample")?;
    let mut m_boot = resource::try_vec(n_obs, 0.0, "bootstrap resample")?;
    let mut x_boot = resource::try_zeros2(n_obs, k, "bootstrap resample")?;

    for _ in 0..config.bootstrap {
        resource::checkpoint()?;

        // Resample with replacement from kept observations
        for i in boot_indices.iter_mut() {
            *i = keep_idx[rng.gen_range(0..n_obs)];
        }
        for (r, &i) in boot_indices.iter().enumerate() {
            y_boot[r] = y[i];
            d_boot[r] = d[i];
            m_boot[r] = m[i];
            x_boot.row_mut(r).assign(&x.row(i));
        }

        // Compute effects on bootstrap sample (no further trimming)
        match compute_mediation_effects(&y_boot, &d_boot, &m_boot, &x_boot, 0.0, &names) {
            Ok((t, de, ie, _, notes)) if t.is_finite() && de.is_finite() && ie.is_finite() => {
                boot_total.push(t);
                boot_direct.push(de);
                boot_indirect.push(ie);
                if let Some(first) = notes.into_iter().next() {
                    warned += 1;
                    first_warning.get_or_insert(first);
                }
            }
            Ok(_) => {
                failed += 1;
                first_error.get_or_insert_with(|| "non-finite effect".to_string());
            }
            Err(e @ (EconError::Cancelled(_) | EconError::ResourceLimit(_))) => return Err(e),
            Err(e) => {
                failed += 1;
                first_error.get_or_insert_with(|| e.to_string());
            }
        }
    }
    if boot_total.len() < MIN_VALID_BOOTSTRAP {
        return Err(EconError::Computation(format!(
            "only {} of {} bootstrap replicates could be fitted (first error: {}); the bootstrap needs at least {MIN_VALID_BOOTSTRAP}",
            boot_total.len(),
            config.bootstrap,
            first_error.as_deref().unwrap_or("none")
        )));
    }

    // Compute standard errors and CIs from bootstrap
    let (se_total, ci_total) = bootstrap_stats(&boot_total)?;
    let (se_direct, ci_direct) = bootstrap_stats(&boot_direct)?;
    let (se_indirect, ci_indirect) = bootstrap_stats(&boot_indirect)?;

    // Compute p-values (two-sided test of H0: effect = 0)
    let p_total = compute_p_value(total, se_total);
    let p_direct = compute_p_value(direct, se_direct);
    let p_indirect = compute_p_value(indirect, se_indirect);

    // Proportion mediated (never clamped)
    let proportion_mediated = if total == 0.0 {
        f64::NAN
    } else {
        indirect / total
    };

    let mut warnings = fit_notes;
    warnings.extend(bootstrap_notes(
        config.bootstrap,
        failed,
        first_error.as_deref(),
        warned,
        first_warning.as_deref(),
    ));
    warnings.extend(proportion_note(proportion_mediated));

    Ok(MediationResult {
        total_effect: total,
        direct_effect: direct,
        indirect_effect: indirect,
        proportion_mediated,
        se_total,
        se_direct,
        se_indirect,
        ci_total,
        ci_direct,
        ci_indirect,
        p_total,
        p_direct,
        p_indirect,
        n_obs,
        n_trimmed,
        missing: cols.missing().clone(),
        warnings,
    })
}

/// Compute mediation effects using IPW.
///
/// `names` names the columns of `x` and then the mediator (for the propensity-model notes).
///
/// Returns (total_effect, direct_effect, indirect_effect, kept_indices, notes)
#[allow(clippy::type_complexity)]
fn compute_mediation_effects(
    y: &[f64],
    d: &[f64],
    m: &[f64],
    x: &Array2<f64>,
    trim: f64,
    names: &[String],
) -> EconResult<(f64, f64, f64, Vec<usize>, Vec<String>)> {
    let n = y.len();

    // Step 1: Estimate propensity score p(D=1|X)
    let (ps_x, mut notes) = propensity_scores(d, x, "p(D = 1 | X)", &names[..x.ncols()])?;

    // Step 2: Estimate propensity score p(D=1|M,X)
    // Add mediator to covariates
    let mut xm_data: Vec<f64> = Vec::with_capacity(n * (x.ncols() + 1));
    for i in 0..n {
        for j in 0..x.ncols() {
            xm_data.push(x[[i, j]]);
        }
        xm_data.push(m[i]);
    }
    let xm = Array2::from_shape_vec((n, x.ncols() + 1), xm_data)
        .map_err(|e| EconError::Internal(format!("Failed to create XM matrix: {}", e)))?;

    let (ps_mx, notes_m) = propensity_scores(d, &xm, "p(D = 1 | X, M)", names)?;
    notes.extend(notes_m);

    // Step 3: Apply trimming
    let mut keep_idx: Vec<usize> = Vec::new();
    for i in 0..n {
        let ps1 = ps_x[i];
        let ps2 = ps_mx[i];
        if ps1 > trim && ps1 < (1.0 - trim) && ps2 > trim && ps2 < (1.0 - trim) {
            keep_idx.push(i);
        }
    }

    if keep_idx.is_empty() {
        return Err(EconError::InsufficientData {
            required: 10,
            provided: 0,
            context: "All observations trimmed due to extreme propensity scores".to_string(),
        });
    }

    // Step 4: Compute IPW estimates following Huber (2014)
    //
    // Total Effect (ATE):
    //   E[D*Y/p(X)] - E[(1-D)*Y/(1-p(X))]
    //
    // For the mediated effects, we use the following identification:
    //
    // NDE (Natural Direct Effect) under d=1:
    //   θ(1) = E[D*Y/p(X)] - E[(1-D)*p(M,X)*Y / ((1-p(X))*(1-p(M,X)))]
    //
    // This compares treated outcomes to a reweighted control group where
    // the mediator distribution is shifted to match the treated distribution.

    let mut sum_w1_y = 0.0;
    let mut sum_w1 = 0.0;
    let mut sum_w0_y = 0.0;
    let mut sum_w0 = 0.0;
    let mut sum_w0_nde_y = 0.0;
    let mut sum_w0_nde = 0.0;

    for &i in &keep_idx {
        let yi = y[i];
        let di = d[i];
        let ps1 = ps_x[i];
        let ps2 = ps_mx[i];

        if di >= 0.5 {
            // Treated observation
            let w = 1.0 / ps1;
            sum_w1_y += w * yi;
            sum_w1 += w;
        } else {
            // Control observation
            let w_ate = 1.0 / (1.0 - ps1);
            sum_w0_y += w_ate * yi;
            sum_w0 += w_ate;

            // NDE weight: reweight controls by ratio of propensity scores
            // This shifts the mediator distribution in the control group
            // to match what it would be if they were treated
            let w_nde = ps2 / ((1.0 - ps1) * (1.0 - ps2));
            sum_w0_nde_y += w_nde * yi;
            sum_w0_nde += w_nde;
        }
    }

    if !(sum_w1 > 0.0 && sum_w0 > 0.0 && sum_w0_nde > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "mediation needs treated and control units after trimming (treated weight {sum_w1}, control weight {sum_w0})"
            ),
        });
    }

    // Normalized (Hajek) estimates
    let mu1 = sum_w1_y / sum_w1;
    let mu0 = sum_w0_y / sum_w0;
    let mu0_nde = sum_w0_nde_y / sum_w0_nde;

    // Total effect (ATE)
    let total = mu1 - mu0;

    // Natural direct effect (NDE): effect of treatment holding mediator at M(0)
    // Comparing treated to reweighted controls with shifted mediator distribution
    let direct = mu1 - mu0_nde;

    // Natural indirect effect (NIE) = Total - Direct
    let indirect = total - direct;

    Ok((total, direct, indirect, keep_idx, notes))
}

/// Logistic propensity scores of `d` on `x` (the shared GLM, `glm.fit`'s IRLS) and the
/// fit's notes under "Propensity model {what}"; `names` names the columns of `x`.
fn propensity_scores(
    d: &[f64],
    x: &Array2<f64>,
    what: &str,
    names: &[String],
) -> EconResult<(Vec<f64>, Vec<String>)> {
    let pf = fit_propensity(x.view(), ArrayView1::from(d), GlmLink::Logit)?;
    let notes = glm_fit_notes(
        &format!("Propensity model {what}"),
        &pf.fit,
        &GlmSpec::logit(),
        names,
    );
    Ok((pf.scores.to_vec(), notes))
}

/// Bootstrap standard error (SD, n - 1) and the 95% type-7 percentile interval.
fn bootstrap_stats(samples: &[f64]) -> EconResult<(f64, (f64, f64))> {
    let n = samples.len() as f64;
    let mean: f64 = samples.iter().sum::<f64>() / n;
    let variance: f64 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    Ok((variance.sqrt(), percentile_interval(samples, 0.95)?))
}

/// Two-sided normal p-value; `NaN` when the SE is not finite or not positive.
fn compute_p_value(estimate: f64, se: f64) -> f64 {
    if se.is_finite() && se > 0.0 {
        normal_p_value_two_sided(estimate / se)
    } else {
        f64::NAN
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::econometrics::treatment::na_fixtures as nf;
    use crate::resource::{self, Budget, CancelToken};
    use p2a_validation::{Check, Tol};
    use polars::prelude::*;

    fn expect_invalid_spec(err: EconError, needle: &str) {
        match err {
            EconError::InvalidSpecification { message } => {
                assert!(
                    message.contains(needle),
                    "message `{message}` does not mention `{needle}`"
                )
            }
            other => panic!("expected InvalidSpecification mentioning `{needle}`, got {other:?}"),
        }
    }

    fn cancelled_budget() -> Budget {
        let t = CancelToken::new();
        t.cancel();
        Budget::new(None, t)
    }

    #[test]
    fn test_mediation_bootstrap_count_is_an_error_not_an_abort() {
        // the LCG data keep 296 rows at trim 0.01, so the bootstrap buffers are reached
        let err = run_mediation_analysis(
            &create_mediation_validation_dataset(),
            "y",
            "treatment",
            "mediator",
            &["x"],
            MediationConfig {
                bootstrap: usize::MAX,
                trim: 0.01,
                seed: Some(42),
            },
        )
        .unwrap_err();
        assert!(matches!(err, EconError::ResourceLimit(_)), "got {err:?}");
    }

    #[test]
    fn test_mediation_needs_two_replicates() {
        for b in [0, 1] {
            expect_invalid_spec(
                run_mediation_analysis(
                    &create_mediation_dataset(),
                    "y",
                    "d",
                    "m",
                    &["x"],
                    MediationConfig {
                        bootstrap: b,
                        trim: 0.01,
                        seed: Some(42),
                    },
                )
                .unwrap_err(),
                "bootstrap",
            );
        }
    }

    #[test]
    fn test_mediation_honours_cancellation() {
        // the LCG data reach the bootstrap loop's checkpoint (296 rows kept at trim 0.01)
        let r = resource::with_budget(cancelled_budget(), || {
            run_mediation_analysis(
                &create_mediation_validation_dataset(),
                "y",
                "treatment",
                "mediator",
                &["x"],
                MediationConfig {
                    bootstrap: 20,
                    trim: 0.01,
                    seed: Some(42),
                },
            )
        });
        assert!(matches!(r, Err(EconError::Cancelled(_))), "got {r:?}");
    }

    /// Used for column errors and the separation warning: its mediator separates the
    /// treatment (treated m in [0.9, 1.3], control m in [0.3, 0.7]), so p(D | M, X) goes to
    /// 0 or 1 and every row is trimmed at any positive `trim`.
    fn create_mediation_dataset() -> Dataset {
        // Create synthetic data with known mediation structure.
        // Key: Treatment assignment must NOT be perfectly predictable by X.
        // The X distribution must overlap substantially between treated and controls.
        //
        // DGP:
        // X ~ random covariate (same distribution for D=0 and D=1)
        // D = treatment (randomly assigned, NOT predicted by X)
        // M = 0.5*D + 0.3*X + noise (mediator depends on treatment and X)
        // Y = 0.4*D + 0.6*M + 0.2*X + noise (outcome has direct and indirect effects)
        //
        // Expected effects (approximately):
        // - Direct effect: 0.4
        // - Indirect effect: 0.5 * 0.6 = 0.3
        // - Total effect: 0.7

        // Mix treatment and control randomly with overlapping X values
        let df = df! {
            "y" => [
                // Interleaved treated (T) and control (C) with similar X
                2.1, 1.2, 2.3, 1.4, 2.0, 1.3, 2.4, 1.5, 2.2, 1.6,  // T,C,T,C,...
                1.8, 2.5, 1.5, 2.6, 1.4, 2.3, 1.7, 2.4, 1.6, 2.7,  // C,T,C,T,...
                2.1, 1.3, 2.2, 1.4, 2.0, 1.5, 2.3, 1.2, 2.1, 1.4,
                1.6, 2.4, 1.5, 2.5, 1.3, 2.2, 1.7, 2.6, 1.4, 2.3,
                2.2, 1.5, 2.1, 1.3, 2.3, 1.6, 2.0, 1.4, 2.4, 1.5,
                1.4, 2.3, 1.6, 2.4, 1.5, 2.2, 1.3, 2.5, 1.7, 2.1
            ],
            "d" => [
                // Interleaved 1,0,1,0,... pattern
                1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0
            ],
            "m" => [
                // Mediator: D*0.5 + X*0.3 + noise, so treated have higher M on average
                1.0, 0.5, 1.1, 0.6, 0.9, 0.4, 1.2, 0.7, 1.0, 0.5,
                0.4, 1.1, 0.5, 1.2, 0.3, 1.0, 0.6, 1.1, 0.5, 1.3,
                1.0, 0.4, 1.1, 0.5, 0.9, 0.6, 1.0, 0.3, 1.1, 0.5,
                0.5, 1.2, 0.4, 1.1, 0.6, 1.0, 0.5, 1.2, 0.4, 1.1,
                1.1, 0.5, 1.0, 0.4, 1.2, 0.6, 0.9, 0.5, 1.1, 0.4,
                0.6, 1.0, 0.5, 1.1, 0.4, 1.0, 0.5, 1.2, 0.6, 1.0
            ],
            "x" => [
                // Covariate: same distribution for both groups (complete overlap)
                0.5, 0.5, 0.6, 0.6, 0.4, 0.4, 0.7, 0.7, 0.5, 0.5,
                0.4, 0.6, 0.5, 0.7, 0.3, 0.5, 0.6, 0.6, 0.5, 0.8,
                0.5, 0.4, 0.6, 0.5, 0.4, 0.6, 0.5, 0.3, 0.6, 0.5,
                0.5, 0.7, 0.4, 0.6, 0.6, 0.5, 0.5, 0.7, 0.4, 0.6,
                0.6, 0.5, 0.5, 0.4, 0.7, 0.6, 0.4, 0.5, 0.6, 0.4,
                0.6, 0.5, 0.5, 0.6, 0.4, 0.5, 0.5, 0.7, 0.6, 0.5
            ]
        }
        .unwrap();

        Dataset::new(df)
    }

    #[test]
    fn test_mediation_basic() {
        let ds = create_mediation_validation_dataset();

        let config = MediationConfig {
            bootstrap: 199, // Fewer for faster test
            trim: 0.01,
            seed: Some(42),
        };

        let result =
            run_mediation_analysis(&ds, "y", "treatment", "mediator", &["x"], config).unwrap();

        // Check that we get reasonable results
        // Total effect should be positive (around 0.7-1.0)
        assert!(
            result.total_effect > 0.3 && result.total_effect < 1.5,
            "Total effect {} out of expected range",
            result.total_effect
        );

        // Direct effect should be positive
        assert!(
            result.direct_effect > -0.5 && result.direct_effect < 1.5,
            "Direct effect {} out of expected range",
            result.direct_effect
        );

        // Indirect effect can be any value
        assert!(
            result.indirect_effect.is_finite(),
            "Indirect effect {} should be finite",
            result.indirect_effect
        );

        // Decomposition should hold: total ≈ direct + indirect
        let decomp_error =
            (result.total_effect - result.direct_effect - result.indirect_effect).abs();
        assert!(
            decomp_error < 0.001,
            "Decomposition error {} too large",
            decomp_error
        );

        // Standard errors should be positive
        assert!(result.se_total > 0.0, "SE total should be positive");
        assert!(result.se_direct > 0.0, "SE direct should be positive");
        assert!(result.se_indirect > 0.0, "SE indirect should be positive");
    }

    #[test]
    fn test_mediation_display() {
        let ds = create_mediation_validation_dataset();

        let config = MediationConfig {
            bootstrap: 99,
            trim: 0.01,
            seed: Some(42),
        };

        let result =
            run_mediation_analysis(&ds, "y", "treatment", "mediator", &["x"], config).unwrap();

        let display = result.to_string();
        assert!(display.contains("Causal Mediation Analysis"));
        assert!(display.contains("Total (ATE)"));
        assert!(display.contains("Direct (NDE)"));
        assert!(display.contains("Indirect (NIE)"));
        assert!(display.contains("Proportion Mediated"));
    }

    #[test]
    fn test_missing_column_error() {
        let ds = create_mediation_dataset();

        let result = run_mediation_analysis(
            &ds,
            "nonexistent",
            "d",
            "m",
            &["x"],
            MediationConfig::default(),
        );

        assert!(result.is_err());
        if let Err(EconError::ColumnNotFound { column, .. }) = result {
            assert_eq!(column, "nonexistent");
        } else {
            panic!("Expected ColumnNotFound error");
        }
    }

    // =========================================================================
    // R Validation Tests (Phase 5)
    // =========================================================================

    /// Simple LCG for deterministic random numbers
    fn lcg_rand_med(seed: &mut u64) -> f64 {
        let a: u64 = 1103515245;
        let c: u64 = 12345;
        let m: u64 = 2_u64.pow(31);
        *seed = (a.wrapping_mul(*seed).wrapping_add(c)) % m;
        (*seed as f64) / (m as f64)
    }

    /// Create validation dataset matching R's mediation package example.
    fn create_mediation_validation_dataset() -> Dataset {
        let n = 300;
        let mut seed: u64 = 42;

        let mut x = Vec::with_capacity(n);
        let mut treatment = Vec::with_capacity(n);
        let mut mediator = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);

        for _ in 0..n {
            // Covariate
            let u1 = lcg_rand_med(&mut seed).max(1e-10);
            let u2 = lcg_rand_med(&mut seed);
            let x_i = ((-2.0_f64 * u1.ln()).sqrt()) * (2.0 * std::f64::consts::PI * u2).cos();
            x.push(x_i);

            // Treatment (random assignment with slight dependence on x)
            let ps = 1.0 / (1.0 + (-(0.5 * x_i)).exp());
            let t = if lcg_rand_med(&mut seed) < ps {
                1.0
            } else {
                0.0
            };
            treatment.push(t);

            // Mediator: M = 0.3 + 0.6*D + 0.4*X + noise
            let u3 = lcg_rand_med(&mut seed).max(1e-10);
            let u4 = lcg_rand_med(&mut seed);
            let noise_m =
                0.5 * ((-2.0_f64 * u3.ln()).sqrt()) * (2.0 * std::f64::consts::PI * u4).cos();
            let m = 0.3 + 0.6 * t + 0.4 * x_i + noise_m;
            mediator.push(m);

            // Outcome: Y = 1 + 0.4*D + 0.5*M + 0.3*X + noise
            // Direct effect = 0.4, Indirect effect = 0.6 * 0.5 = 0.3
            // Total effect = 0.4 + 0.3 = 0.7
            let u5 = lcg_rand_med(&mut seed).max(1e-10);
            let u6 = lcg_rand_med(&mut seed);
            let noise_y =
                0.5 * ((-2.0_f64 * u5.ln()).sqrt()) * (2.0 * std::f64::consts::PI * u6).cos();
            let y_i = 1.0 + 0.4 * t + 0.5 * m + 0.3 * x_i + noise_y;
            y.push(y_i);
        }

        let df = df! {
            "y" => y,
            "treatment" => treatment,
            "mediator" => mediator,
            "x" => x
        }
        .unwrap();

        Dataset::new(df)
    }

    #[test]
    fn test_mediation_recovers_dgp() {
        // No R value: the direct and indirect effects are not checked against the DGP
        // because the NDE weight is CAUS-26 (Tier 4); the emulated total effect is 0.590.

        let dataset = create_mediation_validation_dataset();
        let config = MediationConfig {
            bootstrap: 199,
            trim: 0.05,
            seed: Some(42),
        };

        let result =
            run_mediation_analysis(&dataset, "y", "treatment", "mediator", &["x"], config).unwrap();

        // True total effect 0.7 = 0.4 + 0.6 * 0.5
        assert!(
            (result.total_effect - 0.7).abs() < 0.3,
            "Total effect {:.4} should be close to 0.7",
            result.total_effect
        );
    }

    #[test]
    fn test_mediation_decomposition_identity() {
        // Validate that Total = Direct + Indirect
        let dataset = create_mediation_validation_dataset();
        let config = MediationConfig {
            bootstrap: 99,
            trim: 0.05,
            seed: Some(42),
        };

        let result =
            run_mediation_analysis(&dataset, "y", "treatment", "mediator", &["x"], config).unwrap();

        // Decomposition should hold exactly
        let decomp = result.direct_effect + result.indirect_effect;
        assert!(
            (result.total_effect - decomp).abs() < 1e-10,
            "Decomposition failed: {:.6} ≠ {:.6} + {:.6}",
            result.total_effect,
            result.direct_effect,
            result.indirect_effect
        );
    }

    #[test]
    fn test_mediation_proportion_mediated_identity() {
        // Validate proportion mediated calculation
        let dataset = create_mediation_validation_dataset();
        let config = MediationConfig {
            bootstrap: 99,
            trim: 0.05,
            seed: Some(42),
        };

        let result =
            run_mediation_analysis(&dataset, "y", "treatment", "mediator", &["x"], config).unwrap();

        // Proportion mediated = indirect / total, exactly
        assert_eq!(
            result.proportion_mediated.to_bits(),
            (result.indirect_effect / result.total_effect).to_bits()
        );
    }

    #[test]
    fn test_mediation_bootstrap_ses_positive() {
        // Validate standard error properties
        let dataset = create_mediation_validation_dataset();
        let config = MediationConfig {
            bootstrap: 199,
            trim: 0.05,
            seed: Some(42),
        };

        let result =
            run_mediation_analysis(&dataset, "y", "treatment", "mediator", &["x"], config).unwrap();

        // All SEs should be positive
        assert!(result.se_total > 0.0, "SE total should be positive");
        assert!(result.se_direct > 0.0, "SE direct should be positive");
        assert!(result.se_indirect > 0.0, "SE indirect should be positive");

        // SEs should be reasonable (not too small or large)
        assert!(
            result.se_total < 1.0,
            "SE total {:.4} seems too large",
            result.se_total
        );
        assert!(
            result.se_direct < 1.0,
            "SE direct {:.4} seems too large",
            result.se_direct
        );
        assert!(
            result.se_indirect < 1.0,
            "SE indirect {:.4} seems too large",
            result.se_indirect
        );
    }

    #[test]
    fn test_mediation_percentile_cis_ordered() {
        // Validate CI properties
        let dataset = create_mediation_validation_dataset();
        let config = MediationConfig {
            bootstrap: 199,
            trim: 0.05,
            seed: Some(42),
        };

        let result =
            run_mediation_analysis(&dataset, "y", "treatment", "mediator", &["x"], config).unwrap();

        // CIs should contain point estimates
        assert!(
            result.ci_total.0 <= result.total_effect && result.ci_total.1 >= result.total_effect
        );
        assert!(
            result.ci_direct.0 <= result.direct_effect
                && result.ci_direct.1 >= result.direct_effect
        );
        assert!(
            result.ci_indirect.0 <= result.indirect_effect
                && result.ci_indirect.1 >= result.indirect_effect
        );

        // CIs should have positive width
        assert!(result.ci_total.1 > result.ci_total.0);
        assert!(result.ci_direct.1 > result.ci_direct.0);
        assert!(result.ci_indirect.1 > result.ci_indirect.0);
    }

    #[test]
    fn test_mediation_p_values_in_unit_interval() {
        // Validate p-value properties
        let dataset = create_mediation_validation_dataset();
        let config = MediationConfig {
            bootstrap: 199,
            trim: 0.05,
            seed: Some(42),
        };

        let result =
            run_mediation_analysis(&dataset, "y", "treatment", "mediator", &["x"], config).unwrap();

        // P-values should be in [0, 1]
        assert!(result.p_total >= 0.0 && result.p_total <= 1.0);
        assert!(result.p_direct >= 0.0 && result.p_direct <= 1.0);
        assert!(result.p_indirect >= 0.0 && result.p_indirect <= 1.0);
    }

    #[test]
    fn test_mediation_seeded_bootstrap_reproducible() {
        // Validate that seed produces reproducible results
        let dataset = create_mediation_validation_dataset();

        let config1 = MediationConfig {
            bootstrap: 99,
            trim: 0.05,
            seed: Some(12345),
        };
        let result1 =
            run_mediation_analysis(&dataset, "y", "treatment", "mediator", &["x"], config1)
                .unwrap();

        let config2 = MediationConfig {
            bootstrap: 99,
            trim: 0.05,
            seed: Some(12345),
        };
        let result2 =
            run_mediation_analysis(&dataset, "y", "treatment", "mediator", &["x"], config2)
                .unwrap();

        // Point estimates should be identical (same seed)
        assert!((result1.total_effect - result2.total_effect).abs() < 1e-10);
        assert!((result1.direct_effect - result2.direct_effect).abs() < 1e-10);
        assert!((result1.indirect_effect - result2.indirect_effect).abs() < 1e-10);

        // SEs may have small differences due to bootstrap
        assert!((result1.se_total - result2.se_total).abs() < 0.01);
    }

    // =========================================================================
    // T2-WP13: missing values and column types (R case causal_mediation_na)
    // =========================================================================

    fn na_cfg(trim: f64) -> MediationConfig {
        MediationConfig {
            bootstrap: 20,
            trim,
            seed: Some(11),
        }
    }

    fn estimates(r: &MediationResult) -> Vec<u64> {
        nf::bits(&[
            r.total_effect,
            r.direct_effect,
            r.indirect_effect,
            r.se_total,
            r.se_direct,
            r.se_indirect,
        ])
    }

    #[test]
    fn test_validate_mediation_na_omit_causal_na() {
        // CAUS-24, the null rule: Float64 d isolates it from the Int64 rule.
        let (case, ds) = crate::test_support::load_case("causal_mediation_na");
        let full = nf::with_cast(&ds, "d", DataType::Float64);
        let cc = nf::complete_rows(&case, &full);
        let a = run_mediation_analysis(&full, "y", "d", "m", &["x1", "x2"], na_cfg(0.05)).unwrap();
        let b = run_mediation_analysis(&cc, "y", "d", "m", &["x1", "x2"], na_cfg(0.05)).unwrap();
        // same rows in the same order and the same seed: bit-identical, bootstrap included
        assert_eq!(estimates(&a), estimates(&b));
        Check::new(&case)
            .scalar("n_complete", (a.n_obs + a.n_trimmed) as f64, Tol::exact())
            .scalar("ipw_kept_trim05", a.n_obs as f64, Tol::exact())
            // glm.fit IRLS (fit_propensity, epsilon 1e-10) against R at 1e-14
            .scalar("ipw_total_trim05", a.total_effect, Tol::iterative(1e-8))
            .finish();
    }

    #[test]
    fn test_validate_mediation_total_effect_causal_na() {
        // CAUS-25: the covariate matrix is built row by row.
        let (case, ds) = crate::test_support::load_case("causal_mediation_na");
        let cc = nf::complete_rows(&case, &nf::with_cast(&ds, "d", DataType::Float64));
        let r05 = run_mediation_analysis(&cc, "y", "d", "m", &["x1", "x2"], na_cfg(0.05)).unwrap();
        let r0 = run_mediation_analysis(&cc, "y", "d", "m", &["x1", "x2"], na_cfg(0.0)).unwrap();
        Check::new(&case)
            .scalar("ipw_total_trim05", r05.total_effect, Tol::iterative(1e-8))
            .scalar("ipw_kept_trim05", r05.n_obs as f64, Tol::exact())
            .scalar("ipw_total_trim0", r0.total_effect, Tol::iterative(1e-8))
            .scalar("ipw_kept_trim0", r0.n_obs as f64, Tol::exact())
            .finish();
        // the point estimates do not depend on row or covariate order
        let rev = run_mediation_analysis(
            &nf::reversed(&cc),
            "y",
            "d",
            "m",
            &["x1", "x2"],
            na_cfg(0.05),
        )
        .unwrap();
        let swp = run_mediation_analysis(&cc, "y", "d", "m", &["x2", "x1"], na_cfg(0.05)).unwrap();
        let t = Tol::new(1e-12, 1e-14);
        for (label, o) in [("reversed rows", &rev), ("swapped covariates", &swp)] {
            assert!(
                t.allows(o.total_effect, r05.total_effect)
                    && t.allows(o.direct_effect, r05.direct_effect),
                "{label}: total {} direct {} vs {} {}",
                o.total_effect,
                o.direct_effect,
                r05.total_effect,
                r05.direct_effect
            );
            assert_eq!(o.n_obs, r05.n_obs, "{label}");
        }
    }

    #[test]
    fn test_validate_na_mediation_int64_treatment() {
        // CAUS-24, X-8: an Int64 or Boolean 0/1 treatment is read as numbers.
        let (case, ds) = crate::test_support::load_case("causal_mediation_na");
        let cc = nf::complete_rows(&case, &ds);
        assert_eq!(cc.df().column("d").unwrap().dtype(), &DataType::Int64);
        let run = |data: &Dataset| {
            run_mediation_analysis(data, "y", "d", "m", &["x1", "x2"], na_cfg(0.05)).unwrap()
        };
        let base = run(&cc);
        for dtype in [DataType::Float64, DataType::Boolean] {
            let other = run(&nf::with_cast(&cc, "d", dtype.clone()));
            assert_eq!(estimates(&other), estimates(&base), "{dtype}");
            assert_eq!(
                (other.n_obs, other.n_trimmed),
                (base.n_obs, base.n_trimmed),
                "{dtype}"
            );
        }
    }

    #[test]
    fn test_validate_na_mediation_reports_missing() {
        // The dropped rows are reported.
        let (case, ds) = crate::test_support::load_case("causal_mediation_na");
        let r = run_mediation_analysis(&ds, "y", "d", "m", &["x1", "x2"], na_cfg(0.05)).unwrap();
        Check::new(&case)
            .scalar("n_dropped", r.missing.n_dropped as f64, Tol::exact())
            .scalar("missing_y", nf::by_column(&r.missing, "y"), Tol::exact())
            .scalar("missing_m", nf::by_column(&r.missing, "m"), Tol::exact())
            .scalar("missing_x1", nf::by_column(&r.missing, "x1"), Tol::exact())
            .scalar("n_complete", r.missing.n_used as f64, Tol::exact())
            .finish();
        assert_eq!(r.missing.n_total, 400);
        // request order
        assert_eq!(
            r.missing
                .by_column
                .iter()
                .map(|(c, _)| c.as_str())
                .collect::<Vec<_>>(),
            ["y", "m", "x1"]
        );
        let text = r.to_string();
        assert!(text.contains("N = 280 (trimmed: 70)"), "{text}");
        assert!(
            text.trim_end().ends_with(
                "Missing values: 50 of 400 rows dropped because of missing values \
                 (y: 15, m: 25, x1: 10)"
            ),
            "{text}"
        );
        let cc = run_mediation_analysis(
            &nf::complete_rows(&case, &ds),
            "y",
            "d",
            "m",
            &["x1", "x2"],
            na_cfg(0.05),
        )
        .unwrap();
        assert!(cc.missing.is_empty() && !cc.to_string().contains("Missing values"));
    }

    #[test]
    fn test_validate_na_mediation_errors_name_the_column() {
        // A text column and an absent column are named.
        let (_case, ds) = crate::test_support::load_case("causal_mediation_na");
        let run = |data: &Dataset, d: &str, m: &str, x: &[&str]| {
            run_mediation_analysis(data, "y", d, m, x, na_cfg(0.05))
        };
        match run(&ds, "d_txt", "m", &["x1", "x2"]) {
            Err(EconError::NonNumericColumn { column }) => assert_eq!(column, "d_txt"),
            other => panic!("text treatment: {other:?}"),
        }
        match run(&ds, "d", "mm", &["x1", "x2"]) {
            Err(EconError::ColumnNotFound { column, .. }) => assert_eq!(column, "mm"),
            other => panic!("absent mediator: {other:?}"),
        }
        match run(&ds, "d", "m", &["x1", "x9"]) {
            Err(EconError::ColumnNotFound { column, .. }) => assert_eq!(column, "x9"),
            other => panic!("absent covariate: {other:?}"),
        }
        let mut df = ds.df().clone();
        df.with_column(Series::full_null("m".into(), 400, &DataType::Float64))
            .unwrap();
        match run(&Dataset::new(df), "d", "m", &["x1", "x2"]) {
            Err(EconError::InsufficientData {
                required: 1,
                provided: 0,
                context,
            }) => assert!(context.contains("mediation analysis"), "{context}"),
            other => panic!("all-missing mediator: {other:?}"),
        }
    }

    // =========================================================================
    // T3-WP26: shared GLM, percentile rule, seeds, unclamped proportion
    // =========================================================================

    #[test]
    fn test_validate_mediation_total_effect_medflex_scales() {
        // CAUS-113 part, CAUS-75 (mediation)
        let (case, ds) = crate::test_support::load_case("medflex_scales");
        let mut c = MediationConfig::default();
        c.bootstrap = 20;
        c.trim = 0.05;
        c.seed = Some(11);
        let r = run_mediation_analysis(&ds, "yo", "a", "m", &["c1", "c2"], c).unwrap();
        Check::new(&case)
            // glm.fit to epsilon 1e-10 vs R 1e-14
            .scalar("ipw_yo_total", r.total_effect, Tol::iterative(1e-8))
            .scalar("ipw_yo_kept", r.n_obs as f64, Tol::exact())
            .finish();
        // never clamped: with today's NDE weight (CAUS-26, Tier 4) NIE/TE is -7.891 (R
        // emulation); it was shown as -1
        assert_eq!(
            r.proportion_mediated.to_bits(),
            (r.indirect_effect / r.total_effect).to_bits()
        );
        let outside = r.proportion_mediated < 0.0 || r.proportion_mediated > 1.0;
        assert_eq!(
            outside,
            r.warnings
                .iter()
                .any(|w| w.starts_with("proportion mediated") && w.contains("outside [0, 1]")),
            "{:?}",
            r.warnings
        );
    }

    #[test]
    fn test_validate_mediation_percentile_rule_medflex_scales() {
        // X-18 (mediation): type-7 percentiles
        let (case, _ds) = crate::test_support::load_case("medflex_scales");
        let draws = case.vector("pct_draws");
        let (se, (lo, hi)) = bootstrap_stats(&draws).unwrap();
        Check::new(&case)
            .scalar("pct_sd", se, Tol::coef(draws.len()))
            .vector("pct_ci", &[lo, hi], Tol::coef(draws.len()))
            .finish();
    }

    #[test]
    fn test_mediation_propensity_separation_warns() {
        // CAUS-113: the shared GLM reports separation
        let mut c = MediationConfig::default();
        c.bootstrap = 20;
        c.trim = 0.0; // keep the separated rows (scores in (0, 1), never 0 or 1)
        c.seed = Some(11);
        let r =
            run_mediation_analysis(&create_mediation_dataset(), "y", "d", "m", &["x"], c).unwrap();
        assert!(
            r.warnings.iter().any(|w| w
                == "Propensity model p(D = 1 | X, M): glm.fit: fitted probabilities numerically 0 or 1 occurred"),
            "{:?}",
            r.warnings
        );
        assert!(r.total_effect.is_finite() && r.direct_effect.is_finite());
    }

    #[test]
    fn test_mediation_default_seed_is_fixed() {
        // no seed means seed 42: repeated calls are identical
        let ds = create_mediation_validation_dataset();
        let run = |seed: Option<u64>| {
            let mut c = MediationConfig::default();
            c.bootstrap = 30;
            c.trim = 0.05;
            c.seed = seed;
            run_mediation_analysis(&ds, "y", "treatment", "mediator", &["x"], c).unwrap()
        };
        let (a, b, d) = (run(None), run(None), run(Some(DEFAULT_BOOTSTRAP_SEED)));
        assert_eq!(
            (a.se_total.to_bits(), a.ci_direct),
            (b.se_total.to_bits(), b.ci_direct)
        );
        assert_eq!(a.se_indirect.to_bits(), d.se_indirect.to_bits());
    }

    #[test]
    fn test_mediation_rejects_bad_trim() {
        for trim in [-0.1, 0.5, 0.7, f64::NAN] {
            let mut c = MediationConfig::default();
            c.bootstrap = 20;
            c.trim = trim;
            expect_invalid_spec(
                run_mediation_analysis(
                    &create_mediation_validation_dataset(),
                    "y",
                    "treatment",
                    "mediator",
                    &["x"],
                    c,
                )
                .unwrap_err(),
                "trim",
            );
        }
    }
}
