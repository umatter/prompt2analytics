//! Natural effect models for causal mediation analysis (medflex's imputation estimator).
//!
//! Implements the imputation-based natural effect model estimator of Vansteelandt, Bekaert
//! and Lange (2012) as the R package `medflex` 0.6-11 implements it (`neImpute`, `neModel`,
//! `neEffdecomp`) for a binary 0/1 exposure.
//!
//! # Estimator
//!
//! For treatment A, mediator M, outcome Y and confounders C:
//!
//! 1. **Outcome working model** `Y ~ A + M + C [+ A:M]`, fitted by least squares
//!    ([`MedflexFamily::Gaussian`]), a Poisson GLM (log link) or a binomial GLM (logit
//!    link). The family defaults from the scale ([`EffectScale::default_family`]).
//! 2. **Expanded data**: two rows per unit, the direct exposure `A0` in {0, 1} and the
//!    observed exposure `A1 = A` (which determines the mediator). Every row is imputed with
//!    the working model's prediction `Yhat = E[Y | A = A0, M, C]`.
//! 3. **Natural effect model** `Yhat ~ A0 + A1 + C [+ A0:A1 + A0:C]` with the identity link
//!    (difference scale), the log link (quasi-Poisson, ratio scale) or the logit link
//!    (quasi-binomial, odds-ratio scale). With A:M in the working model the conditional
//!    natural direct effect varies with C through the mediator model, so the model carries
//!    `A0:C` (with medflex's usual `Yhat ~ A0 * A1 + C` the difference-scale NDE would use
//!    the raw control-group mean of M).
//! 4. **Effects** on the link scale at the confounder means `cbar` (medflex's `covLev`;
//!    its default is 0): NDE (pure direct) `= a0 + a0:C cbar`, NIE (total indirect)
//!    `= a1 + a0:a1`, TE `= NDE + NIE`; `exp()` gives the ratios and odds ratios. At the
//!    means, the difference-scale NDE equals VanderWeele's (2015) regression formula
//!    `b_A + b_AM (alpha_0 + alpha_C' cbar)` with the mediator model `M ~ A + C`.
//!
//! The proportion mediated is `NIE / TE` on the difference scale and VanderWeele's
//! `NDE (NIE - 1) / (NDE NIE - 1)` on the ratio scales. It is never clamped; a value
//! outside [0, 1] comes with a warning.
//!
//! # Standard errors
//!
//! - `bootstrap_ci = false`: the robust sandwich of medflex's `se = "robust"`, from the
//!   stacked estimating equations of the working model and the natural effect model (HC0).
//! - `bootstrap_ci = true` (default): units are resampled and everything is refitted
//!   (medflex's `se = "bootstrap"`), with the effects at the full-sample `cbar`; the SE is
//!   the standard deviation of the link-scale draws and the interval is medflex's default
//!   normal type (`confint(type = "norm")`, `boot::norm.ci`). Failed replicates are counted
//!   and reported; with no seed the bootstrap uses [`DEFAULT_BOOTSTRAP_SEED`].
//!
//! On the ratio scales the SEs and p-values are on the log or logit scale.
//!
//! # Identification assumptions
//!
//! 1. No unmeasured confounding of treatment-outcome relationship given C
//! 2. No unmeasured confounding of mediator-outcome relationship given (A, C)
//! 3. No unmeasured confounding of treatment-mediator relationship given C
//! 4. No effect of treatment on mediator-outcome confounders (cross-world independence)
//!
//! # References
//!
//! - Vansteelandt, S., Bekaert, M., & Lange, T. (2012). "Imputation strategies for the
//!   estimation of natural direct and indirect effects." *Epidemiologic Methods*, 1(1),
//!   131-158.
//!
//! - Lange, T., Vansteelandt, S., & Bekaert, M. (2012). "A simple unified approach for
//!   estimating natural direct and indirect effects." *American Journal of Epidemiology*,
//!   176(3), 190-195.
//!
//! - VanderWeele, T. J. (2015). *Explanation in Causal Inference: Methods for
//!   Mediation and Interaction*. Oxford University Press.
//!
//! - Steen, J., Loeys, T., Moerkerke, B., & Vansteelandt, S. (2017). "medflex:
//!   An R Package for Flexible Mediation Analysis using Natural Effect Models."
//!   *Journal of Statistical Software*, 76(11), 1-46.
//!   https://doi.org/10.18637/jss.v076.i11

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::SeedableRng;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, MissingSummary, NaPolicy};
use crate::econometrics::glm::{GlmFamily, GlmFit, GlmLink, GlmSpec, fit_glm};
use crate::errors::{EconError, EconResult};
use crate::linalg::solve::{SingularPolicy, aliasing_message, sym_inverse};
use crate::options::Choices;
use crate::regression::ols::solve_ols;
use crate::resource;
use crate::traits::convergence::{Convergence, ResultNotes, notes_text};
use crate::traits::estimator::{normal_p_value_two_sided, validate_level, z_critical};

/// Seed of the bootstrap when `MedflexConfig::seed` is `None`, so that repeated calls give
/// identical results.
pub const DEFAULT_BOOTSTRAP_SEED: u64 = 42;

/// Fewest valid bootstrap replicates from which bootstrap SEs and intervals are reported.
const MIN_VALID_BOOTSTRAP: usize = 50;

/// Scale of the natural effects: it selects the natural effect model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EffectScale {
    /// Differences: gaussian natural effect model (identity link).
    #[default]
    Difference,
    /// Ratios (risk or rate ratios): quasi-Poisson natural effect model (log link).
    Ratio,
    /// Odds ratios: quasi-binomial natural effect model (logit link).
    OddsRatio,
}

crate::option_enum!(
    EffectScale,
    Choices {
        option: "scale",
        values: &[
            (
                &[
                    "difference",
                    "diff",
                    "additive",
                    "risk_difference",
                    "mean_difference",
                    "rd",
                ],
                EffectScale::Difference,
            ),
            (
                &["ratio", "risk_ratio", "rate_ratio", "relative_risk", "rr"],
                EffectScale::Ratio,
            ),
            (&["odds_ratio", "odds", "or"], EffectScale::OddsRatio),
        ],
    },
    no_display
);

impl EffectScale {
    /// The outcome working-model family used when none is configured: Difference ->
    /// Gaussian, Ratio -> Poisson, OddsRatio -> Binomial.
    pub fn default_family(self) -> MedflexFamily {
        match self {
            Self::Difference => MedflexFamily::Gaussian,
            Self::Ratio => MedflexFamily::Poisson,
            Self::OddsRatio => MedflexFamily::Binomial,
        }
    }

    /// Link of the natural effect model.
    fn link_name(self) -> &'static str {
        match self {
            Self::Difference => "identity",
            Self::Ratio => "log",
            Self::OddsRatio => "logit",
        }
    }

    /// Family of the natural effect model, as R names it.
    fn ne_family_name(self) -> &'static str {
        match self {
            Self::Difference => "gaussian",
            Self::Ratio => "quasipoisson",
            Self::OddsRatio => "quasibinomial",
        }
    }

    /// Link-scale value to the reported scale (identity or `exp`).
    fn to_scale(self, v: f64) -> f64 {
        match self {
            Self::Difference => v,
            Self::Ratio | Self::OddsRatio => v.exp(),
        }
    }
}

impl fmt::Display for EffectScale {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Difference => write!(f, "Difference"),
            Self::Ratio => write!(f, "Ratio"),
            Self::OddsRatio => write!(f, "Odds Ratio"),
        }
    }
}

/// Outcome working-model family, each with its canonical link: gaussian (least squares,
/// identity), poisson (log), binomial (logit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MedflexFamily {
    /// Least squares (identity link).
    #[default]
    Gaussian,
    /// Poisson (log link); the outcome must be non-negative.
    Poisson,
    /// Binomial (logit link); the outcome must lie in [0, 1].
    Binomial,
}

crate::option_enum!(
    MedflexFamily,
    Choices {
        option: "family",
        values: &[
            (&["gaussian", "normal", "linear"], MedflexFamily::Gaussian),
            (
                &["poisson", "quasipoisson", "count"],
                MedflexFamily::Poisson
            ),
            (
                &["binomial", "quasibinomial", "logistic", "binary"],
                MedflexFamily::Binomial,
            ),
        ],
    }
);

impl MedflexFamily {
    fn link_name(self) -> &'static str {
        match self {
            Self::Gaussian => "identity",
            Self::Poisson => "log",
            Self::Binomial => "logit",
        }
    }

    /// R's `linkinv` of the canonical link.
    fn linkinv(self, eta: f64) -> f64 {
        match self {
            Self::Gaussian => eta,
            Self::Poisson => eta.exp().max(f64::EPSILON),
            Self::Binomial => {
                if eta < -LOGIT_THRESH {
                    f64::EPSILON
                } else if eta > LOGIT_THRESH {
                    1.0 - f64::EPSILON
                } else {
                    let e = eta.exp();
                    e / (1.0 + e)
                }
            }
        }
    }

    /// R's `mu.eta` of the canonical link.
    fn mu_eta(self, eta: f64) -> f64 {
        match self {
            Self::Gaussian => 1.0,
            Self::Poisson => eta.exp().max(f64::EPSILON),
            Self::Binomial => {
                if eta.abs() > LOGIT_THRESH {
                    f64::EPSILON
                } else {
                    let e = eta.exp();
                    e / ((1.0 + e) * (1.0 + e))
                }
            }
        }
    }

    /// Variance function at the mean.
    fn variance(self, mu: f64) -> f64 {
        match self {
            Self::Gaussian => 1.0,
            Self::Poisson => mu,
            Self::Binomial => mu * (1.0 - mu),
        }
    }
}

/// R's logit-link clamp (`family.c`, THRESH).
const LOGIT_THRESH: f64 = 30.0;

/// Configuration for Natural Effect Models (medflex).
#[derive(Debug, Clone)]
pub struct MedflexConfig {
    /// Include the treatment-mediator interaction A:M in the outcome working model, and
    /// A0:A1 and A0:C in the natural effect model (default: true)
    pub allow_interaction: bool,
    /// Bootstrap SEs and intervals (default: true); `false` gives the robust sandwich
    pub bootstrap_ci: bool,
    /// Number of bootstrap samples (default: 1000). Must be at least 50 when
    /// `bootstrap_ci` is true (unused otherwise); a count whose result buffers cannot
    /// be allocated is refused with `EconError::ResourceLimit`.
    pub n_bootstrap: usize,
    /// Confidence level for intervals, strictly between 0 and 1 (default: 0.95)
    pub confidence_level: f64,
    /// Effect scale: selects the natural effect model (default: Difference)
    pub scale: EffectScale,
    /// Bootstrap seed; `None` uses [`DEFAULT_BOOTSTRAP_SEED`]
    pub seed: Option<u64>,
    /// Outcome working-model family; `None` uses `scale.default_family()`
    pub family: Option<MedflexFamily>,
}

impl Default for MedflexConfig {
    fn default() -> Self {
        Self {
            allow_interaction: true,
            bootstrap_ci: true,
            n_bootstrap: 1000,
            confidence_level: 0.95,
            scale: EffectScale::Difference,
            seed: None,
            family: None,
        }
    }
}

/// Names used in coefficient labels, model formulas and warnings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MedflexNames {
    /// Outcome name.
    pub outcome: String,
    /// Treatment name.
    pub treatment: String,
    /// Mediator name.
    pub mediator: String,
    /// One name per confounder column.
    pub confounders: Vec<String>,
}

impl MedflexNames {
    /// "Y", "A", "M", "C1", ..., "Cp" (what [`run_medflex`] uses).
    pub fn generic(n_confounders: usize) -> Self {
        Self {
            outcome: "Y".into(),
            treatment: "A".into(),
            mediator: "M".into(),
            confounders: (1..=n_confounders).map(|j| format!("C{j}")).collect(),
        }
    }
}

/// `Vec<f64>` with `NaN` (an aliased coefficient) as JSON `null`.
mod nan_vec {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(v: &[f64], s: S) -> Result<S::Ok, S::Error> {
        v.iter()
            .map(|x| x.is_finite().then_some(*x))
            .collect::<Vec<Option<f64>>>()
            .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f64>, D::Error> {
        Ok(Vec::<Option<f64>>::deserialize(d)?
            .into_iter()
            .map(|x| x.unwrap_or(f64::NAN))
            .collect())
    }
}

/// Results from Natural Effect Models analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MedflexResult {
    // ═══════════════════════════════════════════════════════════════════
    // Core Effect Estimates (on the reported scale: differences, ratios or odds ratios)
    // ═══════════════════════════════════════════════════════════════════
    /// Total Effect (TE = NDE + NIE on the link scale)
    pub total_effect: f64,
    /// Natural Direct Effect (pure direct effect, at the confounder means)
    pub natural_direct_effect: f64,
    /// Natural Indirect Effect (total indirect effect)
    pub natural_indirect_effect: f64,
    /// Proportion mediated: NIE / TE (difference scale) or NDE (NIE - 1) / (NDE NIE - 1)
    /// (ratio scales); never clamped, `NaN` when undefined
    pub proportion_mediated: f64,

    // ═══════════════════════════════════════════════════════════════════
    // Standard Errors (link scale: difference, log or logit)
    // ═══════════════════════════════════════════════════════════════════
    /// Standard error of total effect (link scale)
    pub te_se: f64,
    /// Standard error of NDE (link scale)
    pub nde_se: f64,
    /// Standard error of NIE (link scale)
    pub nie_se: f64,

    // ═══════════════════════════════════════════════════════════════════
    // Confidence Intervals (reported scale)
    // ═══════════════════════════════════════════════════════════════════
    /// Confidence interval for total effect
    pub te_ci: (f64, f64),
    /// Confidence interval for NDE
    pub nde_ci: (f64, f64),
    /// Confidence interval for NIE
    pub nie_ci: (f64, f64),

    // ═══════════════════════════════════════════════════════════════════
    // p-values (two-sided, null of no effect, from the link-scale z)
    // ═══════════════════════════════════════════════════════════════════
    /// p-value for TE
    pub te_p_value: f64,
    /// p-value for NDE
    pub nde_p_value: f64,
    /// p-value for NIE
    pub nie_p_value: f64,

    // ═══════════════════════════════════════════════════════════════════
    // Model Coefficients
    // ═══════════════════════════════════════════════════════════════════
    /// Treatment coefficient of the mediator model `M ~ A + C` (least squares; descriptive)
    pub mediator_coef: f64,
    /// Its standard error
    pub mediator_coef_se: f64,
    /// Treatment coefficient of the outcome working model
    pub direct_coef: f64,
    /// Its standard error
    pub direct_coef_se: f64,
    /// Mediator coefficient of the outcome working model
    pub mediator_outcome_coef: f64,
    /// Its standard error
    pub mediator_outcome_coef_se: f64,
    /// A:M coefficient of the outcome working model (None without interaction)
    pub interaction_coef: Option<f64>,
    /// Its standard error
    pub interaction_coef_se: Option<f64>,

    // ═══════════════════════════════════════════════════════════════════
    // Model Fit Statistics
    // ═══════════════════════════════════════════════════════════════════
    /// R-squared for mediator model
    pub mediator_r_squared: f64,
    /// Deviance R-squared of the outcome working model (`1 - deviance / null deviance`;
    /// R-squared for the gaussian family)
    pub outcome_r_squared: f64,

    // ═══════════════════════════════════════════════════════════════════
    // Sample Information
    // ═══════════════════════════════════════════════════════════════════
    /// Number of observations
    pub n_obs: usize,
    /// Number of treated observations
    pub n_treated: usize,
    /// Number of control observations
    pub n_control: usize,
    /// Raw sample mean of the mediator among controls
    pub mediator_mean_control: f64,
    /// Raw sample mean of the mediator among the treated
    pub mediator_mean_treated: f64,

    // ═══════════════════════════════════════════════════════════════════
    // Configuration
    // ═══════════════════════════════════════════════════════════════════
    /// Whether interaction was included
    pub interaction_included: bool,
    /// Effect scale used
    pub scale: EffectScale,
    /// Confidence level
    pub confidence_level: f64,

    /// Rows dropped because of missing values (`run_medflex_dataset`); empty for
    /// `run_medflex`, whose arrays are used as given.
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,

    /// Outcome working-model family actually fitted.
    #[serde(default)]
    pub family: MedflexFamily,
    /// How the standard errors were computed.
    #[serde(default)]
    pub se_method: String,
    /// The natural effect model, e.g. `yhat ~ a0 + a1 + c1 + c2 (gaussian, identity link)`.
    #[serde(default)]
    pub ne_model: String,
    /// The outcome working model's formula, e.g. `y ~ a + m + c1 + c2 + a:m`.
    #[serde(default)]
    pub outcome_model: String,
    /// Names of the outcome working-model coefficients: `(Intercept)`, A, M, C..., `A:M`.
    #[serde(default)]
    pub outcome_coef_names: Vec<String>,
    /// Outcome working-model coefficients (`NaN`, JSON `null`, when aliased).
    #[serde(default, with = "nan_vec")]
    pub outcome_coefficients: Vec<f64>,
    /// Their standard errors (`NaN` when aliased).
    #[serde(default, with = "nan_vec")]
    pub outcome_std_errors: Vec<f64>,
    /// Names of the natural effect model coefficients: `(Intercept)`, `A0`, `A1`, C...,
    /// `A0:A1`, `A0:C`... (with the treatment's name).
    #[serde(default)]
    pub ne_coef_names: Vec<String>,
    /// Natural effect model coefficients (`NaN` when aliased).
    #[serde(default, with = "nan_vec")]
    pub ne_coefficients: Vec<f64>,
    /// Confounder levels at which the NDE and TE are evaluated (their sample means).
    #[serde(default)]
    pub confounder_means: Vec<f64>,
    /// Warnings: aliased coefficients, fit problems, bootstrap failures, the proportion
    /// mediated.
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl ResultNotes for MedflexResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

impl fmt::Display for MedflexResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "=== Natural Effect Models (medflex, imputation estimator) ==="
        )?;
        writeln!(f)?;
        writeln!(f, "Effect Decomposition ({} scale):", self.scale)?;
        writeln!(f, "{}", "─".repeat(70))?;
        writeln!(
            f,
            "{:<25} {:>12} {:>12} {:>16} {:>8}",
            "Effect",
            "Estimate",
            "Std.Err",
            format!("{:.0}% CI", self.confidence_level * 100.0),
            "p-value"
        )?;
        writeln!(f, "{}", "─".repeat(70))?;

        let stars = |p: f64| -> &str {
            if p < 0.001 {
                "***"
            } else if p < 0.01 {
                "**"
            } else if p < 0.05 {
                "*"
            } else if p < 0.10 {
                "."
            } else {
                ""
            }
        };

        for (label, est, se, ci, p) in [
            (
                "Total Effect (TE)",
                self.total_effect,
                self.te_se,
                self.te_ci,
                self.te_p_value,
            ),
            (
                "Natural Direct Effect",
                self.natural_direct_effect,
                self.nde_se,
                self.nde_ci,
                self.nde_p_value,
            ),
            (
                "Natural Indirect Effect",
                self.natural_indirect_effect,
                self.nie_se,
                self.nie_ci,
                self.nie_p_value,
            ),
        ] {
            writeln!(
                f,
                "{:<25} {:>12.4} {:>12.4} [{:>6.3}, {:>6.3}] {:>7.4}{}",
                label,
                est,
                se,
                ci.0,
                ci.1,
                p,
                stars(p)
            )?;
        }

        writeln!(f, "{}", "─".repeat(70))?;
        match self.scale {
            EffectScale::Difference => {}
            EffectScale::Ratio => writeln!(
                f,
                "Std. errors and p-values are on the log scale; estimates and intervals are ratios."
            )?,
            EffectScale::OddsRatio => writeln!(
                f,
                "Std. errors and p-values are on the logit scale; estimates and intervals are odds ratios."
            )?,
        }
        writeln!(f)?;

        match self.scale {
            EffectScale::Difference => writeln!(
                f,
                "Proportion Mediated (NIE/TE): {:.1}%",
                self.proportion_mediated * 100.0
            )?,
            _ => writeln!(
                f,
                "Proportion Mediated (NDE(NIE - 1)/(NDE NIE - 1), VanderWeele 2015): {:.1}%",
                self.proportion_mediated * 100.0
            )?,
        }
        writeln!(f, "Standard errors: {}", self.se_method)?;
        writeln!(f, "Natural effect model: {}", self.ne_model)?;
        // the confounder names are the working-model names between the mediator and A:M
        let k = self.outcome_coef_names.len();
        let conf_end = if self.interaction_included {
            k.saturating_sub(1)
        } else {
            k
        };
        let conf_names: &[String] = self.outcome_coef_names.get(3..conf_end).unwrap_or(&[]);
        if self.interaction_included && !conf_names.is_empty() {
            let levels: Vec<String> = conf_names
                .iter()
                .zip(&self.confounder_means)
                .map(|(n, v)| format!("{n} = {v:.4}"))
                .collect();
            writeln!(
                f,
                "NDE and TE evaluated at the confounder means: {}",
                levels.join(", ")
            )?;
        }
        writeln!(f)?;

        writeln!(f, "Model Components:")?;
        let name = |j: usize| -> &str {
            self.outcome_coef_names
                .get(j)
                .map(String::as_str)
                .unwrap_or("?")
        };
        let (t, m) = (name(1), name(2));
        writeln!(
            f,
            "  Outcome working model ({}, {} link): {}",
            self.family,
            self.family.link_name(),
            self.outcome_model
        )?;
        writeln!(
            f,
            "    Treatment -> Outcome: {:.4} (SE = {:.4})",
            self.direct_coef, self.direct_coef_se
        )?;
        writeln!(
            f,
            "    Mediator -> Outcome:  {:.4} (SE = {:.4})",
            self.mediator_outcome_coef, self.mediator_outcome_coef_se
        )?;
        if let (Some(int_coef), Some(int_se)) = (self.interaction_coef, self.interaction_coef_se) {
            writeln!(
                f,
                "    A*M Interaction:      {:.4} (SE = {:.4})",
                int_coef, int_se
            )?;
        }
        writeln!(f, "    Deviance R-squared: {:.4}", self.outcome_r_squared)?;
        writeln!(f)?;
        let mut med_rhs: Vec<&str> = vec![t];
        med_rhs.extend(conf_names.iter().map(String::as_str));
        writeln!(
            f,
            "  Mediator model ({m} ~ {}, least squares; descriptive):",
            med_rhs.join(" + ")
        )?;
        writeln!(
            f,
            "    Treatment -> Mediator: {:.4} (SE = {:.4})",
            self.mediator_coef, self.mediator_coef_se
        )?;
        writeln!(f, "    R-squared: {:.4}", self.mediator_r_squared)?;
        writeln!(f)?;

        writeln!(f, "Sample:")?;
        writeln!(
            f,
            "  N = {} (treated: {}, control: {})",
            self.n_obs, self.n_treated, self.n_control
        )?;
        writeln!(
            f,
            "  E[M|A=0] = {:.4}, E[M|A=1] = {:.4} (raw sample means)",
            self.mediator_mean_control, self.mediator_mean_treated
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "Signif. codes: *** p<0.001, ** p<0.01, * p<0.05, . p<0.10"
        )?;
        f.write_str(&notes_text(self))?;
        if let Some(note) = self.missing.note() {
            writeln!(f, "Missing values: {note}")?;
        }

        Ok(())
    }
}

/// Run Natural Effect Models for causal mediation analysis.
///
/// The imputation estimator of the module documentation, with generic names ("Y", "A",
/// "M", "C1", ...) in coefficient labels and warnings; [`run_medflex_with_names`] takes the
/// names, [`run_medflex_dataset`] reads columns by name.
///
/// # Arguments
///
/// * `y` - Outcome (n): any real value for the gaussian working model, non-negative for
///   poisson, in [0, 1] for binomial
/// * `treatment` - Treatment (n), coded 0/1
/// * `mediator` - Mediator (n)
/// * `confounders` - Confounder matrix (n x p), can be empty (n x 0)
/// * `config` - Configuration: scale (natural effect model), working-model family,
///   A:M interaction, robust or bootstrap SEs, confidence level, seed
///
/// # Errors
///
/// `InvalidSpecification` for a treatment not coded 0/1, non-finite input, a confidence
/// level outside (0, 1), a treatment in the span of the confounders (no overlap), an
/// aliased treatment, mediator or A:M column, imputed outcomes outside the natural effect
/// model's domain (ratio: non-negative; odds ratio: [0, 1]) and working-model data the
/// family refuses; `InsufficientData` when there are no more observations than
/// working-model coefficients or fewer than 3 units per arm; `Computation` when fewer than
/// 50 bootstrap replicates could be fitted.
///
/// # Example
///
/// ```ignore
/// use p2a_core::econometrics::medflex::{run_medflex, EffectScale, MedflexConfig};
/// use ndarray::{array, Array2};
///
/// let y = array![1.2, 2.1, 1.5, 3.0, 2.8, 1.1, 2.5, 3.2, 1.4, 2.9];
/// let treatment = array![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0];
/// let mediator = array![0.5, 1.2, 0.6, 1.5, 1.3, 0.4, 1.4, 1.6, 0.7, 0.9];
/// let confounders = Array2::zeros((10, 0)); // no confounders
///
/// let mut config = MedflexConfig::default();
/// config.allow_interaction = false;
/// config.bootstrap_ci = false; // robust sandwich SEs
/// let result = run_medflex(&y.view(), &treatment.view(), &mediator.view(),
///                          &confounders.view(), config)?;
/// println!("{}", result);
/// ```
///
/// # References
///
/// - Vansteelandt, S., Bekaert, M., & Lange, T. (2012). Imputation strategies for the
///   estimation of natural direct and indirect effects. Epidemiologic Methods.
/// - Steen, J., et al. (2017). "medflex: An R Package for Flexible Mediation
///   Analysis using Natural Effect Models." Journal of Statistical Software.
/// - VanderWeele, T. J. (2015). Explanation in Causal Inference. Oxford Univ. Press.
pub fn run_medflex(
    y: &ArrayView1<f64>,
    treatment: &ArrayView1<f64>,
    mediator: &ArrayView1<f64>,
    confounders: &ArrayView2<f64>,
    config: MedflexConfig,
) -> EconResult<MedflexResult> {
    let names = MedflexNames::generic(confounders.ncols());
    run_medflex_with_names(y, treatment, mediator, confounders, &names, config)
}

/// [`run_medflex`] with the names used in coefficient labels, model formulas and
/// warnings (`names.confounders` has one name per confounder column).
pub fn run_medflex_with_names(
    y: &ArrayView1<f64>,
    treatment: &ArrayView1<f64>,
    mediator: &ArrayView1<f64>,
    confounders: &ArrayView2<f64>,
    names: &MedflexNames,
    config: MedflexConfig,
) -> EconResult<MedflexResult> {
    let inp = Inputs {
        y: y.to_owned(),
        a: treatment.to_owned(),
        m: mediator.to_owned(),
        c: confounders.to_owned(),
    };
    let plan = validate_and_plan(&inp, names, &config)?;
    let n = inp.n();
    let est = estimate(&inp, &plan)?;
    let med = mediator_report(&inp, &plan)?;

    let n_treated = inp.a.iter().filter(|&&v| v == 1.0).count();
    let n_control = n - n_treated;
    let (mut sum_t, mut sum_c) = (0.0, 0.0);
    for i in 0..n {
        if inp.a[i] == 1.0 {
            sum_t += inp.m[i];
        } else {
            sum_c += inp.m[i];
        }
    }

    let level = config.confidence_level;
    let scale = plan.scale;
    let link = est.link;
    let (se, ci, se_method, boot_notes) = if config.bootstrap_ci {
        let mut rng =
            rand::rngs::StdRng::seed_from_u64(config.seed.unwrap_or(DEFAULT_BOOTSTRAP_SEED));
        let mut next_rows = |rows: &mut [usize]| {
            for r in rows.iter_mut() {
                *r = rng.gen_range(0..n);
            }
        };
        let draws = bootstrap_link_draws(&inp, &plan, config.n_bootstrap, &mut next_rows)?;
        let (se, ci, notes) = summarize_bootstrap(&draws, &plan, &link, level)?;
        let w = match scale {
            EffectScale::Difference => "difference",
            EffectScale::Ratio => "log",
            EffectScale::OddsRatio => "logit",
        };
        let method = format!(
            "bootstrap ({} of {} replicates; SE = SD of the {w}-scale draws, normal intervals as medflex confint(type = \"norm\"))",
            draws.link[0].len(),
            draws.total
        );
        (se, ci, method, notes)
    } else {
        let se = robust_se(&inp, &est, &plan)?;
        let z = z_critical(level)?;
        let ci: [(f64, f64); 3] = std::array::from_fn(|e| {
            (
                scale.to_scale(link[e] - z * se[e]),
                scale.to_scale(link[e] + z * se[e]),
            )
        });
        let method = "robust sandwich (stacked estimating equations of the outcome working model and the natural effect model, as medflex se = \"robust\")".to_string();
        (se, ci, method, Vec::new())
    };
    let p: [f64; 3] = std::array::from_fn(|e| {
        if se[e].is_finite() && se[e] > 0.0 {
            normal_p_value_two_sided(link[e] / se[e])
        } else {
            f64::NAN
        }
    });
    let [nde, nie, te] = link.map(|v| scale.to_scale(v));
    let proportion_mediated = match scale {
        EffectScale::Difference => {
            if te == 0.0 {
                f64::NAN
            } else {
                nie / te
            }
        }
        EffectScale::Ratio | EffectScale::OddsRatio => {
            let den = nde * nie - 1.0;
            if den == 0.0 {
                f64::NAN
            } else {
                nde * (nie - 1.0) / den
            }
        }
    };

    let mut warnings = est.notes.clone();
    warnings.extend(med.notes.iter().cloned());
    warnings.extend(boot_notes);
    warnings.extend(proportion_note(proportion_mediated));

    let w = &est.working;
    let k_w = w.coef.len();
    let (interaction_coef, interaction_coef_se) = if plan.interaction {
        (Some(w.coef[k_w - 1]), Some(w.se[k_w - 1]))
    } else {
        (None, None)
    };
    let ne_terms = plan.ne_names[1..].join(" + ");
    let ne_model = format!(
        "yhat ~ {ne_terms} ({}, {} link)",
        scale.ne_family_name(),
        scale.link_name()
    );
    let outcome_model = format!(
        "{} ~ {}",
        plan.names.outcome,
        plan.working_names[1..].join(" + ")
    );

    Ok(MedflexResult {
        total_effect: te,
        natural_direct_effect: nde,
        natural_indirect_effect: nie,
        proportion_mediated,
        te_se: se[2],
        nde_se: se[0],
        nie_se: se[1],
        te_ci: ci[2],
        nde_ci: ci[0],
        nie_ci: ci[1],
        te_p_value: p[2],
        nde_p_value: p[0],
        nie_p_value: p[1],
        mediator_coef: med.coef_a,
        mediator_coef_se: med.se_a,
        direct_coef: w.coef[1],
        direct_coef_se: w.se[1],
        mediator_outcome_coef: w.coef[2],
        mediator_outcome_coef_se: w.se[2],
        interaction_coef,
        interaction_coef_se,
        mediator_r_squared: med.r2,
        outcome_r_squared: w.r2,
        n_obs: n,
        n_treated,
        n_control,
        mediator_mean_control: sum_c / n_control as f64,
        mediator_mean_treated: sum_t / n_treated as f64,
        interaction_included: plan.interaction,
        scale,
        confidence_level: level,
        missing: MissingSummary::none(n),
        family: plan.family,
        se_method,
        ne_model,
        outcome_model,
        outcome_coef_names: plan.working_names.clone(),
        outcome_coefficients: w.coef.to_vec(),
        outcome_std_errors: w.se.to_vec(),
        ne_coef_names: plan.ne_names.clone(),
        ne_coefficients: est.ne_coef.to_vec(),
        confounder_means: plan.cbar.clone(),
        warnings,
    })
}

// ============================================================================
// Inputs, validation and the plan of one analysis
// ============================================================================

/// The analysis variables (the sample or one bootstrap resample).
struct Inputs {
    y: Array1<f64>,
    a: Array1<f64>,
    m: Array1<f64>,
    c: Array2<f64>,
}

impl Inputs {
    fn n(&self) -> usize {
        self.y.len()
    }

    /// The units `rows` (with repetition), through the allocation ceiling.
    fn resample(&self, rows: &[usize]) -> EconResult<Inputs> {
        let n = rows.len();
        let what = "bootstrap resample";
        let pick = |v: &Array1<f64>| -> EconResult<Array1<f64>> {
            let mut out = resource::try_zeros1(n, what)?;
            for (o, &r) in out.iter_mut().zip(rows) {
                *o = v[r];
            }
            Ok(out)
        };
        let mut c = resource::try_zeros2(n, self.c.ncols(), what)?;
        for (i, &r) in rows.iter().enumerate() {
            c.row_mut(i).assign(&self.c.row(r));
        }
        Ok(Inputs {
            y: pick(&self.y)?,
            a: pick(&self.a)?,
            m: pick(&self.m)?,
            c,
        })
    }
}

/// What one analysis fits, fixed for the sample and every bootstrap replicate.
struct Plan {
    family: MedflexFamily,
    scale: EffectScale,
    interaction: bool,
    /// Confounder levels of the NDE and TE contrasts (the sample means).
    cbar: Vec<f64>,
    names: MedflexNames,
    /// "(Intercept)", A, M, C..., "A:M"
    working_names: Vec<String>,
    /// "(Intercept)", "A0", "A1", C..., "A0:A1", "A0:C"...
    ne_names: Vec<String>,
    /// "(Intercept)", C..., A: the identification design `[1, C, A]`
    id_names: Vec<String>,
    /// "(Intercept)", A, C...: the reported mediator model
    med_names: Vec<String>,
}

fn spec_err(message: String) -> EconError {
    EconError::InvalidSpecification { message }
}

/// Prefix an `InvalidSpecification` message; other errors pass through.
fn prefixed(e: EconError, prefix: &str) -> EconError {
    match e {
        EconError::InvalidSpecification { message } => spec_err(format!("{prefix}{message}")),
        other => other,
    }
}

fn validate_and_plan(
    inp: &Inputs,
    names: &MedflexNames,
    config: &MedflexConfig,
) -> EconResult<Plan> {
    let n = inp.n();
    if inp.a.len() != n {
        return Err(spec_err(format!(
            "Treatment length ({}) must match outcome length ({n})",
            inp.a.len()
        )));
    }
    if inp.m.len() != n {
        return Err(spec_err(format!(
            "Mediator length ({}) must match outcome length ({n})",
            inp.m.len()
        )));
    }
    if inp.c.nrows() != n {
        return Err(spec_err(format!(
            "Confounders rows ({}) must match outcome length ({n})",
            inp.c.nrows()
        )));
    }
    let p = inp.c.ncols();
    if names.confounders.len() != p {
        return Err(spec_err(format!(
            "names.confounders has {} names but confounders has {p} columns",
            names.confounders.len()
        )));
    }
    validate_level(config.confidence_level, "confidence_level")?;
    if config.bootstrap_ci && config.n_bootstrap < MIN_VALID_BOOTSTRAP {
        return Err(spec_err(format!(
            "n_bootstrap must be at least {MIN_VALID_BOOTSTRAP} when bootstrap_ci is true (fewer give an unreliable bootstrap standard error); got {}",
            config.n_bootstrap
        )));
    }
    for (role, v) in [
        ("outcome", &inp.y),
        ("treatment", &inp.a),
        ("mediator", &inp.m),
    ] {
        if let Some(i) = v.iter().position(|x| !x.is_finite()) {
            return Err(spec_err(format!(
                "{role} has a non-finite value at row {i}"
            )));
        }
    }
    for (j, col) in inp.c.columns().into_iter().enumerate() {
        if let Some(i) = col.iter().position(|x| !x.is_finite()) {
            return Err(spec_err(format!(
                "confounder '{}' has a non-finite value at row {i}",
                names.confounders[j]
            )));
        }
    }
    if let Some((i, v)) = inp
        .a
        .iter()
        .enumerate()
        .find(|(_, v)| **v != 0.0 && **v != 1.0)
    {
        return Err(spec_err(format!(
            "natural effect models need a binary 0/1 treatment; row {i} has treatment {v}"
        )));
    }
    let min_obs = if config.allow_interaction { 10 } else { 8 };
    if n < min_obs {
        return Err(EconError::InsufficientData {
            required: min_obs,
            provided: n,
            context: "Natural effect models require sufficient observations".to_string(),
        });
    }
    let n_treated = inp.a.iter().filter(|&&v| v == 1.0).count();
    let n_control = n - n_treated;
    if n_treated < 3 || n_control < 3 {
        return Err(EconError::InsufficientData {
            required: 3,
            provided: n_treated.min(n_control),
            context: "Need at least 3 observations in each treatment group".to_string(),
        });
    }
    let k_out = 3 + p + usize::from(config.allow_interaction);
    if n <= k_out {
        return Err(EconError::InsufficientData {
            required: k_out + 1,
            provided: n,
            context: format!(
                "natural effect models: the outcome working model has {k_out} coefficients ({p} confounders{}); \
                 it needs more observations than coefficients",
                if config.allow_interaction {
                    ", A:M"
                } else {
                    ""
                }
            ),
        });
    }

    let family = config.family.unwrap_or(config.scale.default_family());
    let cbar: Vec<f64> = inp
        .c
        .columns()
        .into_iter()
        .map(|col| col.sum() / n as f64)
        .collect();
    let (t, m) = (&names.treatment, &names.mediator);
    let conf = &names.confounders;
    let intercept = "(Intercept)".to_string();
    let mut working_names = vec![intercept.clone(), t.clone(), m.clone()];
    working_names.extend(conf.iter().cloned());
    let mut ne_names = vec![intercept.clone(), format!("{t}0"), format!("{t}1")];
    ne_names.extend(conf.iter().cloned());
    if config.allow_interaction {
        working_names.push(format!("{t}:{m}"));
        ne_names.push(format!("{t}0:{t}1"));
        ne_names.extend(conf.iter().map(|c| format!("{t}0:{c}")));
    }
    let mut id_names = vec![intercept.clone()];
    id_names.extend(conf.iter().cloned());
    id_names.push(t.clone());
    let mut med_names = vec![intercept, t.clone()];
    med_names.extend(conf.iter().cloned());
    Ok(Plan {
        family,
        scale: config.scale,
        interaction: config.allow_interaction,
        cbar,
        names: names.clone(),
        working_names,
        ne_names,
        id_names,
        med_names,
    })
}

// ============================================================================
// Designs
// ============================================================================

/// Working model `[1, A, M, C, (A M)]` (n x k_w), R's column order.
fn working_design(inp: &Inputs, plan: &Plan) -> EconResult<Array2<f64>> {
    let (n, p) = inp.c.dim();
    let k = 3 + p + usize::from(plan.interaction);
    let mut x = resource::try_zeros2(n, k, "natural effect models: working-model design")?;
    for i in 0..n {
        let mut row = x.row_mut(i);
        row[0] = 1.0;
        row[1] = inp.a[i];
        row[2] = inp.m[i];
        for j in 0..p {
            row[3 + j] = inp.c[[i, j]];
        }
        if plan.interaction {
            row[k - 1] = inp.a[i] * inp.m[i];
        }
    }
    Ok(x)
}

/// Working-model rows of the expanded data (2n x k_w): row `2i + j` sets A = j.
fn counterfactual_design(inp: &Inputs, plan: &Plan) -> EconResult<Array2<f64>> {
    let (n, p) = inp.c.dim();
    let k = 3 + p + usize::from(plan.interaction);
    let rows = resource::checked_len(&[n, 2], "natural effect models: expanded data")?;
    let mut x = resource::try_zeros2(rows, k, "natural effect models: expanded data")?;
    for i in 0..n {
        for j in 0..2 {
            let a0 = j as f64;
            let mut row = x.row_mut(2 * i + j);
            row[0] = 1.0;
            row[1] = a0;
            row[2] = inp.m[i];
            for l in 0..p {
                row[3 + l] = inp.c[[i, l]];
            }
            if plan.interaction {
                row[k - 1] = a0 * inp.m[i];
            }
        }
    }
    Ok(x)
}

/// Natural effect model design (2n x k_n): row `2i + j` is
/// `[1, A0 = j, A1 = A_i, C_i, (j A_i, j C_i)]`.
fn ne_design(inp: &Inputs, plan: &Plan) -> EconResult<Array2<f64>> {
    let (n, p) = inp.c.dim();
    let k = 3 + p + if plan.interaction { 1 + p } else { 0 };
    let rows = resource::checked_len(&[n, 2], "natural effect models: expanded data")?;
    let mut z = resource::try_zeros2(rows, k, "natural effect models: natural effect design")?;
    for i in 0..n {
        for j in 0..2 {
            let a0 = j as f64;
            let mut row = z.row_mut(2 * i + j);
            row[0] = 1.0;
            row[1] = a0;
            row[2] = inp.a[i];
            for l in 0..p {
                row[3 + l] = inp.c[[i, l]];
            }
            if plan.interaction {
                row[3 + p] = a0 * inp.a[i];
                for l in 0..p {
                    row[4 + p + l] = a0 * inp.c[[i, l]];
                }
            }
        }
    }
    Ok(z)
}

/// `[1, C, A]` (A last, so that pivoting aliases A when it lies in the span of 1 and C).
fn identification_design(inp: &Inputs) -> EconResult<Array2<f64>> {
    let (n, p) = inp.c.dim();
    let mut x = resource::try_zeros2(n, p + 2, "natural effect models: mediator design")?;
    for i in 0..n {
        let mut row = x.row_mut(i);
        row[0] = 1.0;
        for j in 0..p {
            row[1 + j] = inp.c[[i, j]];
        }
        row[p + 1] = inp.a[i];
    }
    Ok(x)
}

fn zero_nan(v: &Array1<f64>) -> Array1<f64> {
    v.mapv(|x| if x.is_nan() { 0.0 } else { x })
}

// ============================================================================
// One fit: working model, imputation, natural effect model, contrasts
// ============================================================================

/// The outcome working model.
struct WorkingFit {
    /// Coefficients, `NaN` for aliased columns.
    coef: Array1<f64>,
    /// Standard errors, `NaN` for aliased columns.
    se: Array1<f64>,
    /// Fitted means.
    mu: Array1<f64>,
    aliased: Vec<usize>,
    /// Deviance R-squared.
    r2: f64,
    notes: Vec<String>,
}

fn fit_working(x: &Array2<f64>, y: &Array1<f64>, plan: &Plan) -> EconResult<WorkingFit> {
    const MODEL: &str = "Outcome working model";
    let wf = match plan.family {
        MedflexFamily::Gaussian => {
            let fit = solve_ols(x.view(), y.view())?;
            let n = y.len();
            let rss: f64 = fit.residuals.iter().map(|e| e * e).sum();
            let df = n.saturating_sub(fit.rank);
            let sigma2 = if df > 0 { rss / df as f64 } else { f64::NAN };
            let se = fit.unscaled.diag().mapv(|u| (u * sigma2).sqrt());
            let mean = y.sum() / n as f64;
            let tss: f64 = y.iter().map(|v| (v - mean).powi(2)).sum();
            let r2 = if tss > 0.0 { 1.0 - rss / tss } else { 0.0 };
            let notes = aliasing_message(&fit.aliased, &plan.working_names)
                .map(|m| format!("{MODEL}: {m}"))
                .into_iter()
                .collect();
            WorkingFit {
                mu: y - &fit.residuals,
                coef: fit.beta,
                se,
                aliased: fit.aliased,
                r2,
                notes,
            }
        }
        MedflexFamily::Poisson | MedflexFamily::Binomial => {
            let spec = if plan.family == MedflexFamily::Poisson {
                GlmSpec::poisson()
            } else {
                GlmSpec::logit()
            };
            let fit = fit_glm(x.view(), y.view(), &spec)
                .map_err(|e| prefixed(e, "outcome working model: "))?;
            let notes = glm_fit_notes(MODEL, &fit, &spec, &plan.working_names);
            let r2 = if fit.null_deviance > 0.0 {
                1.0 - fit.deviance / fit.null_deviance
            } else {
                0.0
            };
            WorkingFit {
                r2,
                notes,
                mu: fit.fitted,
                se: fit.std_errors,
                aliased: fit.aliased,
                coef: fit.coefficients,
            }
        }
    };
    // the natural effects need A, M and A:M
    let k = wf.coef.len();
    let mut needed = vec![(1, "treatment"), (2, "mediator")];
    if plan.interaction {
        needed.push((k - 1, "treatment-mediator interaction"));
    }
    for (j, role) in needed {
        if wf.aliased.contains(&j) {
            let refit = if j == k - 1 && plan.interaction {
                "; refit with allow_interaction = false"
            } else {
                ""
            };
            return Err(spec_err(format!(
                "the {role} column '{}' is aliased in the outcome working model; natural effects are not identified{refit}",
                plan.working_names[j]
            )));
        }
    }
    Ok(wf)
}

/// Both arms present (a bootstrap resample can lose one).
fn check_arms(a: &Array1<f64>) -> EconResult<()> {
    let treated = a.iter().filter(|&&v| v == 1.0).count();
    let control = a.len() - treated;
    if treated == 0 || control == 0 {
        return Err(spec_err(format!(
            "natural effects need treated and control units (treated: {treated}, control: {control})"
        )));
    }
    Ok(())
}

/// The mediator model `M ~ C + A` (A last): refuses a treatment in the span of 1 and C
/// (no overlap) and returns the coefficients with aliased ones set to 0, and the
/// aliasing note.
fn identification(inp: &Inputs, plan: &Plan) -> EconResult<(Array1<f64>, Vec<String>)> {
    let x = identification_design(inp)?;
    let fit = solve_ols(x.view(), inp.m.view())?;
    if fit.aliased.contains(&(x.ncols() - 1)) {
        return Err(spec_err(
            "treatment is a linear function of the confounders (no overlap); natural effects are not identified"
                .into(),
        ));
    }
    let notes = aliasing_message(&fit.aliased, &plan.id_names)
        .map(|m| format!("Mediator model: {m}"))
        .into_iter()
        .collect();
    Ok((zero_nan(&fit.beta), notes))
}

/// One fit of the estimator on `inp`.
struct Estimate {
    /// NDE, NIE, TE on the link scale.
    link: [f64; 3],
    /// Working-model and natural effect model notes.
    notes: Vec<String>,
    working: WorkingFit,
    /// Working design (n x k_w).
    x: Array2<f64>,
    /// Counterfactual working design (2n x k_w) and its linear predictor.
    xcf: Array2<f64>,
    eta_cf: Array1<f64>,
    /// Imputed outcomes (2n).
    yhat: Array1<f64>,
    /// Natural effect design (2n x k_n), coefficients (`NaN` aliased), fitted values.
    z: Array2<f64>,
    ne_coef: Array1<f64>,
    ne_aliased: Vec<usize>,
    nu: Array1<f64>,
}

/// The three contrasts over the natural effect model's columns: NDE, NIE, TE.
fn contrasts(plan: &Plan, k_n: usize) -> [Array1<f64>; 3] {
    let p = plan.cbar.len();
    let mut nde = Array1::zeros(k_n);
    let mut nie = Array1::zeros(k_n);
    nde[1] = 1.0;
    nie[2] = 1.0;
    if plan.interaction {
        nie[3 + p] = 1.0;
        for (j, &c) in plan.cbar.iter().enumerate() {
            nde[4 + p + j] = c;
        }
    }
    let te = &nde + &nie;
    [nde, nie, te]
}

fn estimate(inp: &Inputs, plan: &Plan) -> EconResult<Estimate> {
    check_arms(&inp.a)?;
    identification(inp, plan)?;
    let x = working_design(inp, plan)?;
    let working = fit_working(&x, &inp.y, plan)?;
    let beta_z = zero_nan(&working.coef);
    let xcf = counterfactual_design(inp, plan)?;
    let eta_cf = xcf.dot(&beta_z);
    let yhat = eta_cf.mapv(|e| plan.family.linkinv(e));
    match plan.scale {
        EffectScale::Difference => {}
        EffectScale::Ratio => {
            if let Some((r, v)) = yhat.iter().enumerate().find(|(_, v)| **v < 0.0) {
                return Err(spec_err(format!(
                    "scale 'ratio' fits a log-linear natural effect model and needs non-negative imputed outcomes, but the {} working model imputes {v} for row {}; use family 'poisson' or 'binomial'",
                    plan.family,
                    r / 2
                )));
            }
        }
        EffectScale::OddsRatio => {
            if let Some((r, v)) = yhat
                .iter()
                .enumerate()
                .find(|(_, v)| !(0.0..=1.0).contains(*v))
            {
                return Err(spec_err(format!(
                    "scale 'odds_ratio' fits a logistic natural effect model and needs imputed outcomes in [0, 1], but the {} working model imputes {v} for row {}; use family 'binomial'",
                    plan.family,
                    r / 2
                )));
            }
        }
    }

    const MODEL: &str = "Natural effect model";
    let z = ne_design(inp, plan)?;
    let mut notes = working.notes.clone();
    let (ne_coef, ne_aliased, nu) = match plan.scale {
        EffectScale::Difference => {
            let fit = solve_ols(z.view(), yhat.view())?;
            notes.extend(
                aliasing_message(&fit.aliased, &plan.ne_names).map(|m| format!("{MODEL}: {m}")),
            );
            (fit.beta, fit.aliased, &yhat - &fit.residuals)
        }
        EffectScale::Ratio | EffectScale::OddsRatio => {
            let spec = if plan.scale == EffectScale::Ratio {
                GlmSpec::new(GlmFamily::QuasiPoisson, GlmLink::Log)
            } else {
                GlmSpec::new(GlmFamily::QuasiBinomial, GlmLink::Logit)
            };
            let fit = fit_glm(z.view(), yhat.view(), &spec)
                .map_err(|e| prefixed(e, "natural effect model: "))?;
            notes.extend(glm_fit_notes(MODEL, &fit, &spec, &plan.ne_names));
            (fit.coefficients, fit.aliased, fit.fitted)
        }
    };
    let p = plan.cbar.len();
    let mut needed = vec![1, 2];
    if plan.interaction {
        needed.push(3 + p);
    }
    for j in needed {
        if ne_aliased.contains(&j) {
            return Err(spec_err(format!(
                "the natural effect model term '{}' is aliased; natural effects are not identified",
                plan.ne_names[j]
            )));
        }
    }
    let theta_z = zero_nan(&ne_coef);
    let k = contrasts(plan, theta_z.len());
    let link = [k[0].dot(&theta_z), k[1].dot(&theta_z), k[2].dot(&theta_z)];
    Ok(Estimate {
        link,
        notes,
        working,
        x,
        xcf,
        eta_cf,
        yhat,
        z,
        ne_coef,
        ne_aliased,
        nu,
    })
}

/// The difference-scale effects of a gaussian working model in closed form (the same
/// numbers as [`estimate`], without the expanded data): `NDE = b_A + b_AM (alpha_0 +
/// alpha_C' cbar)`, `NIE = alpha_A (b_M + b_AM)` with the mediator model `M ~ C + A`.
/// Used by the bootstrap replicates.
fn closed_form(inp: &Inputs, plan: &Plan) -> EconResult<([f64; 3], Vec<String>)> {
    check_arms(&inp.a)?;
    let (alpha, med_notes) = identification(inp, plan)?;
    let x = working_design(inp, plan)?;
    let working = fit_working(&x, &inp.y, plan)?;
    let b = zero_nan(&working.coef);
    let p = plan.cbar.len();
    let b_am = if plan.interaction {
        b[b.len() - 1]
    } else {
        0.0
    };
    let level = alpha[0]
        + plan
            .cbar
            .iter()
            .enumerate()
            .map(|(j, c)| alpha[1 + j] * c)
            .sum::<f64>();
    let nde = b[1] + b_am * level;
    let nie = alpha[p + 1] * (b[2] + b_am);
    let mut notes = working.notes;
    notes.extend(med_notes);
    Ok(([nde, nie, nde + nie], notes))
}

/// The columns of `m` listed in `cols`.
fn columns(m: &Array2<f64>, cols: &[usize], what: &str) -> EconResult<Array2<f64>> {
    let mut out = resource::try_zeros2(m.nrows(), cols.len(), what)?;
    for (j, &c) in cols.iter().enumerate() {
        out.column_mut(j).assign(&m.column(c));
    }
    Ok(out)
}

/// `X' diag(w) X`.
fn weighted_gram(x: &Array2<f64>, w: &Array1<f64>) -> Array2<f64> {
    let xw = x * &w.view().insert_axis(ndarray::Axis(1));
    x.t().dot(&xw)
}

/// Robust (stacked sandwich, HC0) link-scale SEs of NDE, NIE and TE, as medflex
/// `neModel(se = "robust")`: with the working model's scores `U_i = x_i (y_i - mu_i)` and
/// information `A_b`, the natural effect model's unit scores `S_i` and information `A_t`,
/// and `G = d S / d beta'`, each unit's influence is `A_t^{-1} (S_i + G A_b^{-1} U_i)`.
/// Computed on the non-aliased columns of both models.
fn robust_se(inp: &Inputs, est: &Estimate, plan: &Plan) -> EconResult<[f64; 3]> {
    let n = est.x.nrows();
    let okw: Vec<usize> = (0..est.working.coef.len())
        .filter(|j| !est.working.aliased.contains(j))
        .collect();
    let okn: Vec<usize> = (0..est.ne_coef.len())
        .filter(|j| !est.ne_aliased.contains(j))
        .collect();
    let what = "natural effect models: robust variance";
    let x = columns(&est.x, &okw, what)?;
    let xcf = columns(&est.xcf, &okw, what)?;
    let z = columns(&est.z, &okn, what)?;

    // working model: information and scores
    let v = est.working.mu.mapv(|mu| plan.family.variance(mu));
    let a_b = weighted_gram(&x, &v);
    let resid = &inp.y - &est.working.mu;
    let u = &x * &resid.view().insert_axis(ndarray::Axis(1));

    // natural effect model: information, unit scores, cross derivative
    let w = est.nu.mapv(|nu| match plan.scale {
        EffectScale::Difference => 1.0,
        EffectScale::Ratio => nu,
        EffectScale::OddsRatio => nu * (1.0 - nu),
    });
    let a_t = weighted_gram(&z, &w);
    let d = est.eta_cf.mapv(|e| plan.family.mu_eta(e));
    let xcf_d = &xcf * &d.view().insert_axis(ndarray::Axis(1));
    let g = z.t().dot(&xcf_d);
    let mut s = resource::try_zeros2(n, okn.len(), what)?;
    for i in 0..n {
        for r in [2 * i, 2 * i + 1] {
            let e = est.yhat[r] - est.nu[r];
            s.row_mut(i).scaled_add(e, &z.row(r));
        }
    }

    let a_b_inv = sym_inverse(
        a_b.view(),
        SingularPolicy::Error,
        "outcome working-model information",
    )?
    .inverse;
    let a_t_inv = sym_inverse(
        a_t.view(),
        SingularPolicy::Error,
        "natural effect model information",
    )?
    .inverse;
    let phi = (s + u.dot(&a_b_inv).dot(&g.t())).dot(&a_t_inv);
    let vcov = phi.t().dot(&phi);
    let k = contrasts(plan, est.ne_coef.len());
    Ok(std::array::from_fn(|e| {
        let ko = Array1::from_iter(okn.iter().map(|&j| k[e][j]));
        ko.dot(&vcov.dot(&ko)).sqrt()
    }))
}

/// The reported mediator model `M ~ A + C` (least squares, R's column order; descriptive).
struct MediatorReport {
    coef_a: f64,
    se_a: f64,
    r2: f64,
    notes: Vec<String>,
}

fn mediator_report(inp: &Inputs, plan: &Plan) -> EconResult<MediatorReport> {
    let (n, p) = inp.c.dim();
    let mut x = resource::try_zeros2(n, p + 2, "natural effect models: mediator design")?;
    for i in 0..n {
        let mut row = x.row_mut(i);
        row[0] = 1.0;
        row[1] = inp.a[i];
        for j in 0..p {
            row[2 + j] = inp.c[[i, j]];
        }
    }
    let fit = solve_ols(x.view(), inp.m.view())?;
    let rss: f64 = fit.residuals.iter().map(|e| e * e).sum();
    let df = n.saturating_sub(fit.rank);
    let sigma2 = if df > 0 { rss / df as f64 } else { f64::NAN };
    let mean = inp.m.sum() / n as f64;
    let tss: f64 = inp.m.iter().map(|v| (v - mean).powi(2)).sum();
    Ok(MediatorReport {
        coef_a: fit.beta[1],
        se_a: (fit.unscaled[[1, 1]] * sigma2).sqrt(),
        r2: if tss > 0.0 { 1.0 - rss / tss } else { 0.0 },
        notes: aliasing_message(&fit.aliased, &plan.med_names)
            .map(|m| format!("Mediator model: {m}"))
            .into_iter()
            .collect(),
    })
}

// ============================================================================
// Bootstrap
// ============================================================================

/// Link-scale draws of NDE, NIE and TE and the replicate accounting.
struct BootDraws {
    link: [Vec<f64>; 3],
    total: usize,
    failed: usize,
    first_error: Option<String>,
    warned: usize,
    first_warning: Option<String>,
}

/// One replicate's link-scale effects and notes: the closed form for a gaussian working
/// model on the difference scale, the full estimator otherwise.
fn replicate_link(inp: &Inputs, plan: &Plan) -> EconResult<([f64; 3], Vec<String>)> {
    if plan.family == MedflexFamily::Gaussian && plan.scale == EffectScale::Difference {
        closed_form(inp, plan)
    } else {
        let e = estimate(inp, plan)?;
        Ok((e.link, e.notes))
    }
}

/// `n_boot` replicates on the units `next_rows` draws (medflex `se = "bootstrap"`: units
/// resampled, everything refitted, the effects at the sample's `cbar`). A replicate that
/// fails, or gives a non-finite effect, is counted and dropped; cancellation and the
/// allocation ceiling stop the bootstrap.
fn bootstrap_link_draws(
    inp: &Inputs,
    plan: &Plan,
    n_boot: usize,
    next_rows: &mut dyn FnMut(&mut [usize]),
) -> EconResult<BootDraws> {
    let what = "bootstrap replications (n_bootstrap)";
    let mut link = [
        resource::try_with_capacity::<f64>(n_boot, what)?,
        resource::try_with_capacity::<f64>(n_boot, what)?,
        resource::try_with_capacity::<f64>(n_boot, what)?,
    ];
    let mut rows = resource::try_vec(inp.n(), 0usize, "bootstrap resample rows")?;
    let (mut failed, mut warned) = (0usize, 0usize);
    let (mut first_error, mut first_warning): (Option<String>, Option<String>) = (None, None);
    for _ in 0..n_boot {
        resource::checkpoint()?;
        next_rows(&mut rows);
        let sample = inp.resample(&rows)?;
        match replicate_link(&sample, plan) {
            Ok((l, notes)) if l.iter().all(|v| v.is_finite()) => {
                for (draws, v) in link.iter_mut().zip(l) {
                    draws.push(v);
                }
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
    Ok(BootDraws {
        link,
        total: n_boot,
        failed,
        first_error,
        warned,
        first_warning,
    })
}

/// Bootstrap SEs (SD of the link-scale draws) and medflex's default normal-type intervals
/// (`boot::norm.ci`: `2 theta_hat - mean(draws) -/+ z sd(draws)` on the link scale, mapped to
/// the reported scale), and the bootstrap notes. Fewer than 50 usable replicates is an
/// error.
fn summarize_bootstrap(
    d: &BootDraws,
    plan: &Plan,
    link_hat: &[f64; 3],
    level: f64,
) -> EconResult<([f64; 3], [(f64, f64); 3], Vec<String>)> {
    let valid = d.link[0].len();
    if valid < MIN_VALID_BOOTSTRAP {
        return Err(EconError::Computation(format!(
            "only {valid} of {} bootstrap replicates could be fitted (first error: {}); the bootstrap needs at least {MIN_VALID_BOOTSTRAP}",
            d.total,
            d.first_error.as_deref().unwrap_or("none")
        )));
    }
    let z = z_critical(level)?;
    let mut se = [f64::NAN; 3];
    let mut ci = [(f64::NAN, f64::NAN); 3];
    for e in 0..3 {
        let draws = &d.link[e];
        let b = draws.len() as f64;
        let mean = draws.iter().sum::<f64>() / b;
        let sd = (draws.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (b - 1.0)).sqrt();
        let centre = 2.0 * link_hat[e] - mean;
        se[e] = sd;
        ci[e] = (
            plan.scale.to_scale(centre - z * sd),
            plan.scale.to_scale(centre + z * sd),
        );
    }
    let notes = bootstrap_notes(
        d.total,
        d.failed,
        d.first_error.as_deref(),
        d.warned,
        d.first_warning.as_deref(),
    );
    Ok((se, ci, notes))
}

// ============================================================================
// Notes shared with mediation.rs
// ============================================================================

/// The notes of one GLM fit under `model`: the convergence warning, the fit's other
/// warnings and the aliasing message.
pub(crate) fn glm_fit_notes(
    model: &str,
    fit: &GlmFit,
    spec: &GlmSpec<'_>,
    names: &[String],
) -> Vec<String> {
    let mut out = Vec::new();
    if !fit.converged {
        out.extend(
            Convergence::new(
                false,
                fit.iterations,
                spec.max_iter,
                "relative change in deviance",
                f64::NAN,
                spec.epsilon,
            )
            .note(&format!("{model} (glm)")),
        );
    }
    out.extend(
        fit.warnings
            .iter()
            .filter(|w| w.as_str() != "glm.fit: algorithm did not converge")
            .map(|w| format!("{model}: {w}")),
    );
    out.extend(aliasing_message(&fit.aliased, names).map(|m| format!("{model}: {m}")));
    out
}

/// Failed and warned bootstrap replicates.
pub(crate) fn bootstrap_notes(
    total: usize,
    failed: usize,
    first_error: Option<&str>,
    warned: usize,
    first_warning: Option<&str>,
) -> Vec<String> {
    let mut out = Vec::new();
    if failed > 0 {
        out.push(format!(
            "{failed} of {total} bootstrap replicates failed and were dropped (first error: {})",
            first_error.unwrap_or("unknown")
        ));
    }
    if warned > 0 {
        out.push(format!(
            "{warned} of {total} bootstrap replicates reported fit warnings (first: {})",
            first_warning.unwrap_or("unknown")
        ));
    }
    out
}

/// A note for a proportion mediated outside [0, 1] (beyond rounding) or undefined.
pub(crate) fn proportion_note(pm: f64) -> Option<String> {
    if pm.is_nan() {
        Some("proportion mediated is undefined: the total effect is exactly at the null".into())
    } else if !(-1e-8..=1.0 + 1e-8).contains(&pm) {
        Some(format!(
            "proportion mediated {pm:.4} lies outside [0, 1]: the direct and indirect effects have opposite signs or the total effect is close to the null"
        ))
    } else {
        None
    }
}

// ============================================================================
// Dataset-based interface
// ============================================================================

/// Run Natural Effect Models on a dataset.
///
/// [`run_medflex_with_names`] on the named columns; coefficient labels and warnings use
/// the column names.
///
/// # Arguments
///
/// * `dataset` - The dataset containing all variables
/// * `outcome` - Name of the outcome variable column
/// * `treatment` - Name of the treatment variable column (coded 0/1)
/// * `mediator` - Name of the mediator variable column
/// * `confounders` - Names of confounder columns (can be empty)
/// * `config` - Configuration options
///
/// # Missing values and column types
///
/// Rows with a missing value (null or NaN) in the outcome, treatment, mediator or a
/// confounder are dropped before estimation and reported in `missing`; integer and
/// boolean columns are read as numbers.
///
/// # Example
///
/// ```ignore
/// use p2a_core::econometrics::medflex::{run_medflex_dataset, EffectScale, MedflexConfig};
///
/// let config = MedflexConfig {
///     allow_interaction: true,
///     scale: EffectScale::OddsRatio, // binomial working model, logistic natural effect model
///     n_bootstrap: 500,
///     ..Default::default()
/// };
///
/// let result = run_medflex_dataset(
///     &dataset, "outcome", "treatment", "mediator",
///     &["age", "gender"], config
/// )?;
/// println!("{}", result);
/// ```
pub fn run_medflex_dataset(
    dataset: &Dataset,
    outcome: &str,
    treatment: &str,
    mediator: &str,
    confounders: &[&str],
    config: MedflexConfig,
) -> EconResult<MedflexResult> {
    // One request over every model variable, so every model is fitted on the same
    // complete rows (R lm/glm na.omit). Integer and boolean columns are read as numbers.
    let cols = ColumnRequest::new(dataset.df())
        .numeric(outcome)
        .numeric(treatment)
        .numeric(mediator)
        .numerics(confounders)
        .na(NaPolicy::Omit)
        .min_rows(1, "natural effect models")
        .extract()?;
    let c = if confounders.is_empty() {
        Array2::zeros((cols.n(), 0))
    } else {
        cols.matrix(confounders, false)?.data
    };
    let names = MedflexNames {
        outcome: outcome.to_string(),
        treatment: treatment.to_string(),
        mediator: mediator.to_string(),
        confounders: confounders.iter().map(|s| s.to_string()).collect(),
    };
    let mut result = run_medflex_with_names(
        &ArrayView1::from(cols.numeric(outcome)),
        &ArrayView1::from(cols.numeric(treatment)),
        &ArrayView1::from(cols.numeric(mediator)),
        &c.view(),
        &names,
        config,
    )?;
    result.missing = cols.missing().clone();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::econometrics::treatment::na_fixtures as nf;
    use crate::resource::{self, Budget, CancelToken};
    use ndarray::array;
    use p2a_validation::{Case, Check, Tol};
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

    fn run_test_medflex(config: MedflexConfig) -> EconResult<MedflexResult> {
        let (y, treatment, mediator, confounders) = create_test_data();
        run_medflex(
            &y.view(),
            &treatment.view(),
            &mediator.view(),
            &confounders.view(),
            config,
        )
    }

    #[test]
    fn test_medflex_bootstrap_count_is_an_error_not_an_abort() {
        let err = run_test_medflex(MedflexConfig {
            bootstrap_ci: true,
            n_bootstrap: usize::MAX,
            seed: Some(42),
            ..Default::default()
        })
        .unwrap_err();
        assert!(matches!(err, EconError::ResourceLimit(_)), "got {err:?}");
    }

    #[test]
    fn test_medflex_needs_fifty_replicates() {
        let err = run_test_medflex(MedflexConfig {
            bootstrap_ci: true,
            n_bootstrap: 10,
            seed: Some(42),
            ..Default::default()
        })
        .unwrap_err();
        expect_invalid_spec(err, "n_bootstrap");
        // robust sandwich: the count is unused
        assert!(
            run_test_medflex(MedflexConfig {
                bootstrap_ci: false,
                n_bootstrap: 10,
                seed: Some(42),
                ..Default::default()
            })
            .is_ok()
        );
    }

    #[test]
    fn test_medflex_honours_cancellation() {
        let r = resource::with_budget(cancelled_budget(), || {
            run_test_medflex(MedflexConfig {
                bootstrap_ci: true,
                n_bootstrap: 60,
                seed: Some(42),
                ..Default::default()
            })
        });
        assert!(matches!(r, Err(EconError::Cancelled(_))), "got {r:?}");
    }

    fn create_test_data() -> (Array1<f64>, Array1<f64>, Array1<f64>, Array2<f64>) {
        // Synthetic data with known mediation structure:
        // True DGP:
        //   M = 0.5 + 0.6*A + noise  (alpha_1 = 0.6)
        //   Y = 1.0 + 0.4*A + 0.5*M + 0.2*A*M + noise
        //
        // Expected (approximately):
        //   Without interaction: NDE = 0.4, NIE = 0.6*0.5 = 0.3, TE = 0.7
        //   With interaction: more complex

        let n = 100;
        let y = array![
            // Treated (A=1) with higher Y on average
            2.1, 2.3, 2.0, 2.5, 2.2, 2.4, 2.1, 2.6, 2.3, 2.2, 2.0, 2.4, 2.5, 2.1, 2.3, 2.6, 2.2,
            2.4, 2.0, 2.5, 2.3, 2.1, 2.4, 2.2, 2.5, 2.0, 2.3, 2.6, 2.1, 2.4, 2.2, 2.5, 2.0, 2.3,
            2.1, 2.6, 2.4, 2.2, 2.5, 2.3, 2.1, 2.4, 2.2, 2.0, 2.5, 2.3, 2.6, 2.1, 2.4, 2.2,
            // Control (A=0) with lower Y on average
            1.3, 1.5, 1.2, 1.6, 1.4, 1.3, 1.5, 1.2, 1.4, 1.6, 1.3, 1.5, 1.4, 1.2, 1.6, 1.3, 1.5,
            1.4, 1.2, 1.6, 1.4, 1.3, 1.5, 1.2, 1.6, 1.4, 1.3, 1.5, 1.2, 1.6, 1.5, 1.3, 1.4, 1.2,
            1.6, 1.5, 1.3, 1.4, 1.2, 1.6, 1.4, 1.5, 1.3, 1.2, 1.6, 1.4, 1.5, 1.3, 1.2, 1.6
        ];

        let treatment = array![
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ];

        let mediator = array![
            // Treated have higher M (around 1.1 = 0.5 + 0.6)
            1.0, 1.2, 0.9, 1.3, 1.1, 1.2, 1.0, 1.4, 1.1, 1.0, 0.9, 1.2, 1.3, 1.0, 1.1, 1.4, 1.0,
            1.2, 0.9, 1.3, 1.1, 0.9, 1.2, 1.0, 1.3, 0.9, 1.1, 1.4, 0.9, 1.2, 1.0, 1.3, 0.9, 1.1,
            0.9, 1.4, 1.2, 1.0, 1.3, 1.1, 0.9, 1.2, 1.0, 0.9, 1.3, 1.1, 1.4, 0.9, 1.2, 1.0,
            // Control have lower M (around 0.5)
            0.4, 0.6, 0.3, 0.7, 0.5, 0.4, 0.6, 0.3, 0.5, 0.7, 0.4, 0.6, 0.5, 0.3, 0.7, 0.4, 0.6,
            0.5, 0.3, 0.7, 0.5, 0.4, 0.6, 0.3, 0.7, 0.5, 0.4, 0.6, 0.3, 0.7, 0.6, 0.4, 0.5, 0.3,
            0.7, 0.6, 0.4, 0.5, 0.3, 0.7, 0.5, 0.6, 0.4, 0.3, 0.7, 0.5, 0.6, 0.4, 0.3, 0.7
        ];

        // No confounders for simplicity
        let confounders = Array2::zeros((n, 0));

        (y, treatment, mediator, confounders)
    }

    #[test]
    fn test_medflex_basic() {
        let (y, treatment, mediator, confounders) = create_test_data();

        let config = MedflexConfig {
            allow_interaction: false,
            bootstrap_ci: true,
            n_bootstrap: 200, // Reduced for faster tests
            seed: Some(42),
            ..Default::default()
        };

        let result = run_medflex(
            &y.view(),
            &treatment.view(),
            &mediator.view(),
            &confounders.view(),
            config,
        )
        .unwrap();

        // Check basic properties
        assert!(result.total_effect > 0.0, "Total effect should be positive");
        assert!(result.n_obs == 100, "Should have 100 observations");
        assert!(result.n_treated == 50, "Should have 50 treated");
        assert!(result.n_control == 50, "Should have 50 control");

        // Check decomposition: TE = NDE + NIE
        let decomp_error =
            (result.total_effect - result.natural_direct_effect - result.natural_indirect_effect)
                .abs();
        assert!(
            decomp_error < 1e-10,
            "Decomposition error {} too large",
            decomp_error
        );

        // Check that SEs are positive
        assert!(result.te_se > 0.0, "TE SE should be positive");
        assert!(result.nde_se > 0.0, "NDE SE should be positive");
        assert!(result.nie_se > 0.0, "NIE SE should be positive");

        // Check confidence intervals contain point estimates
        assert!(result.te_ci.0 <= result.total_effect, "TE CI lower bound");
        assert!(result.te_ci.1 >= result.total_effect, "TE CI upper bound");
    }

    #[test]
    fn test_medflex_with_interaction() {
        let (y, treatment, mediator, confounders) = create_test_data();

        let config = MedflexConfig {
            allow_interaction: true,
            bootstrap_ci: true,
            n_bootstrap: 200,
            seed: Some(42),
            ..Default::default()
        };

        let result = run_medflex(
            &y.view(),
            &treatment.view(),
            &mediator.view(),
            &confounders.view(),
            config,
        )
        .unwrap();

        // Check that interaction coefficient is present
        assert!(
            result.interaction_coef.is_some(),
            "Interaction coefficient should be present"
        );
        assert!(
            result.interaction_included,
            "Should flag interaction as included"
        );

        // Decomposition should still hold
        let decomp_error =
            (result.total_effect - result.natural_direct_effect - result.natural_indirect_effect)
                .abs();
        assert!(
            decomp_error < 1e-10,
            "Decomposition error {} too large",
            decomp_error
        );
    }

    #[test]
    fn test_medflex_robust_se_without_bootstrap() {
        let (y, treatment, mediator, confounders) = create_test_data();

        let config = MedflexConfig {
            allow_interaction: false,
            bootstrap_ci: false, // robust sandwich
            confidence_level: 0.95,
            ..Default::default()
        };

        let result = run_medflex(
            &y.view(),
            &treatment.view(),
            &mediator.view(),
            &confounders.view(),
            config,
        )
        .unwrap();

        assert!(
            result.se_method.starts_with("robust sandwich"),
            "{}",
            result.se_method
        );
        // Standard errors should be computed
        assert!(result.te_se > 0.0, "TE SE should be positive");
        assert!(result.nde_se > 0.0, "NDE SE should be positive");
        assert!(result.nie_se > 0.0, "NIE SE should be positive");

        // CIs should be valid
        assert!(
            result.te_ci.0 < result.te_ci.1,
            "CI should have lower < upper"
        );
    }

    #[test]
    fn test_medflex_display() {
        let (y, treatment, mediator, confounders) = create_test_data();

        let config = MedflexConfig {
            n_bootstrap: 100,
            seed: Some(42),
            ..Default::default()
        };

        let result = run_medflex(
            &y.view(),
            &treatment.view(),
            &mediator.view(),
            &confounders.view(),
            config,
        )
        .unwrap();

        let display = result.to_string();
        assert!(display.contains("Natural Effect Models"));
        assert!(display.contains("Total Effect"));
        assert!(display.contains("Natural Direct Effect"));
        assert!(display.contains("Natural Indirect Effect"));
        assert!(display.contains("Proportion Mediated"));
    }

    #[test]
    fn test_medflex_dataset_interface() {
        let df = df! {
            "y" => [2.1, 2.3, 2.0, 1.3, 1.5, 1.2, 2.2, 2.4, 1.4, 1.6,
                   2.0, 2.5, 1.3, 1.5, 2.3, 2.1, 1.4, 1.2, 2.4, 2.2],
            "treatment" => [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                           1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            "mediator" => [1.0, 1.2, 0.9, 0.4, 0.6, 0.3, 1.1, 1.3, 0.5, 0.7,
                          0.9, 1.3, 0.4, 0.6, 1.1, 0.9, 0.5, 0.3, 1.2, 1.0],
            "age" => [30.0, 35.0, 28.0, 42.0, 38.0, 31.0, 33.0, 40.0, 36.0, 29.0,
                     34.0, 37.0, 41.0, 32.0, 39.0, 27.0, 43.0, 35.0, 30.0, 38.0]
        }
        .unwrap();

        let ds = Dataset::new(df);

        let config = MedflexConfig {
            bootstrap_ci: false,
            ..Default::default()
        };

        let result =
            run_medflex_dataset(&ds, "y", "treatment", "mediator", &["age"], config).unwrap();

        assert_eq!(result.n_obs, 20);
        assert!(result.total_effect.is_finite());
    }

    #[test]
    fn test_medflex_insufficient_data() {
        // Too few observations
        let y = array![1.0, 2.0, 1.5];
        let treatment = array![1.0, 0.0, 1.0];
        let mediator = array![0.5, 0.3, 0.6];
        let confounders = Array2::zeros((3, 0));

        let config = MedflexConfig::default();
        let result = run_medflex(
            &y.view(),
            &treatment.view(),
            &mediator.view(),
            &confounders.view(),
            config,
        );

        assert!(result.is_err());
        if let Err(EconError::InsufficientData { .. }) = result {
            // Expected error type
        } else {
            panic!("Expected InsufficientData error");
        }
    }

    /// The first `n` rows of the case (y = yc, a, m) and nine confounders: c1, c2 and seven LCG columns.
    /// Rows 0..14 hold exactly 3 treated units.
    fn small_design(n: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array2<f64>) {
        let ds = medflex_sim();
        let cols = ColumnRequest::new(ds.df())
            .numerics(&["yc", "a", "m", "c1", "c2"])
            .na(NaPolicy::Fail)
            .extract()
            .unwrap();
        let take = |name: &str| cols.numeric_array(name).slice(ndarray::s![..n]).to_owned();
        let (c1, c2) = (take("c1"), take("c2"));
        let mut s: u64 = 12345;
        let mut u = || {
            s = (69069 * s + 1) % (1u64 << 32);
            s as f64 / 4294967296.0 - 0.5
        };
        let mut c = Array2::zeros((n, 9));
        for i in 0..n {
            c[[i, 0]] = c1[i];
            c[[i, 1]] = c2[i];
            for j in 2..9 {
                c[[i, j]] = u();
            }
        }
        (take("yc"), take("a"), take("m"), c)
    }

    #[test]
    fn test_medflex_rejects_n_le_k() {
        let run = |n: usize, interaction: bool| {
            let (y, a, m, c) = small_design(n);
            let mut cfg = MedflexConfig::default();
            cfg.allow_interaction = interaction;
            cfg.bootstrap_ci = false;
            run_medflex(&y.view(), &a.view(), &m.view(), &c.view(), cfg)
        };
        // k = 4 + 9 with A:M, 3 + 9 without; n must exceed k
        for (n, inter, required) in [(12, true, 14), (13, true, 14), (12, false, 13)] {
            match run(n, inter) {
                Err(EconError::InsufficientData {
                    required: r,
                    provided,
                    context,
                }) => {
                    assert_eq!((r, provided), (required, n));
                    assert!(
                        context.contains("more observations than coefficients"),
                        "{context}"
                    );
                }
                other => panic!("n = {n}, A:M = {inter}: expected InsufficientData, got {other:?}"),
            }
        }
        assert!(run(14, true).is_ok());
    }

    #[test]
    fn test_effect_scale_parses_aliases_and_rejects_unknown() {
        EffectScale::choices().validate().unwrap();
        for (s, want) in [
            ("difference", EffectScale::Difference),
            ("Risk Difference", EffectScale::Difference),
            ("ratio", EffectScale::Ratio),
            ("risk-ratio", EffectScale::Ratio),
            ("RR", EffectScale::Ratio),
            ("odds_ratio", EffectScale::OddsRatio),
            ("Odds Ratio", EffectScale::OddsRatio),
            ("odds", EffectScale::OddsRatio),
            ("OR", EffectScale::OddsRatio),
        ] {
            assert_eq!(s.parse::<EffectScale>().unwrap(), want, "{s}");
        }
        assert_eq!(EffectScale::OddsRatio.to_string(), "Odds Ratio"); // Display kept
        match "logratio".parse::<EffectScale>() {
            Err(EconError::InvalidSpecification { message }) => assert!(
                message.contains("unknown scale 'logratio'")
                    && message.contains("difference, ratio, odds_ratio"),
                "{message}"
            ),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn test_medflex_family_parses_aliases_and_rejects_unknown() {
        MedflexFamily::choices().validate().unwrap();
        for (s, want) in [
            ("gaussian", MedflexFamily::Gaussian),
            ("Normal", MedflexFamily::Gaussian),
            ("poisson", MedflexFamily::Poisson),
            ("quasipoisson", MedflexFamily::Poisson),
            ("binomial", MedflexFamily::Binomial),
            ("logistic", MedflexFamily::Binomial),
        ] {
            assert_eq!(s.parse::<MedflexFamily>().unwrap(), want, "{s}");
        }
        assert_eq!(MedflexFamily::Binomial.to_string(), "binomial");
        assert_eq!(
            (
                EffectScale::Difference.default_family(),
                EffectScale::Ratio.default_family(),
                EffectScale::OddsRatio.default_family()
            ),
            (
                MedflexFamily::Gaussian,
                MedflexFamily::Poisson,
                MedflexFamily::Binomial
            )
        );
        assert!("gamma".parse::<MedflexFamily>().is_err());
    }

    #[test]
    fn test_medflex_rejects_bad_confidence_level() {
        let (y, t, m, c) = create_test_data();
        for level in [0.0, 1.0, 1.5, -0.2, f64::NAN] {
            let mut cfg = MedflexConfig::default();
            cfg.bootstrap_ci = false;
            cfg.confidence_level = level;
            expect_invalid_spec(
                run_medflex(&y.view(), &t.view(), &m.view(), &c.view(), cfg).unwrap_err(),
                "confidence_level",
            );
        }
    }

    // =========================================================================
    // T2-WP13: missing values and column types (R case causal_mediation_na)
    // =========================================================================

    /// Robust sandwich SEs, so the bootstrap count rule does not apply.
    fn na_cfg(allow_interaction: bool) -> MedflexConfig {
        MedflexConfig {
            allow_interaction,
            bootstrap_ci: false,
            ..Default::default()
        }
    }

    /// The mediator model, group counts and means: the same with and without interaction.
    fn check_mediator_side(check: &mut Check<'_>, r: &MedflexResult) {
        let n = r.n_obs;
        check
            .scalar("n_complete", n as f64, Tol::exact())
            .scalar("n_treated", r.n_treated as f64, Tol::exact())
            .scalar("n_control", r.n_control as f64, Tol::exact())
            .scalar("mean_m_control", r.mediator_mean_control, Tol::coef(n))
            .scalar("mean_m_treated", r.mediator_mean_treated, Tol::coef(n))
            .named_subset("med_coef", &["d"], &["d"], &[r.mediator_coef], Tol::coef(n))
            .named_subset("med_se", &["d"], &["d"], &[r.mediator_coef_se], Tol::se(n))
            .scalar("med_r2", r.mediator_r_squared, Tol::stat(n));
    }

    fn check_no_interaction(case: &Case, r: &MedflexResult) {
        let n = r.n_obs;
        let mut check = Check::new(case);
        check_mediator_side(&mut check, r);
        check
            .named_subset(
                "out_coef",
                &["d", "m"],
                &["d", "m"],
                &[r.direct_coef, r.mediator_outcome_coef],
                Tol::coef(n),
            )
            .named_subset(
                "out_se",
                &["d", "m"],
                &["d", "m"],
                &[r.direct_coef_se, r.mediator_outcome_coef_se],
                Tol::se(n),
            )
            .scalar("out_r2", r.outcome_r_squared, Tol::stat(n))
            .finish();
    }

    #[test]
    fn test_validate_medflex_causal_na() {
        let (case, ds) = crate::test_support::load_case("causal_mediation_na");
        let run = |data: &Dataset| {
            run_medflex_dataset(data, "y", "d", "m", &["x1", "x2"], na_cfg(false)).unwrap()
        };
        // the null rule alone (Float64 d), then the Int64 column as loaded
        check_no_interaction(&case, &run(&nf::with_cast(&ds, "d", DataType::Float64)));
        check_no_interaction(&case, &run(&ds));
    }

    #[test]
    fn test_validate_medflex_interaction_causal_na() {
        let (case, ds) = crate::test_support::load_case("causal_mediation_na");
        let r = run_medflex_dataset(&ds, "y", "d", "m", &["x1", "x2"], na_cfg(true)).unwrap();
        let n = r.n_obs;
        let names = ["d", "m", "d:m"];
        let mut check = Check::new(&case);
        check_mediator_side(&mut check, &r);
        check
            .named_subset(
                "outi_coef",
                &names,
                &names,
                &[
                    r.direct_coef,
                    r.mediator_outcome_coef,
                    r.interaction_coef.unwrap(),
                ],
                Tol::coef(n),
            )
            .named_subset(
                "outi_se",
                &names,
                &names,
                &[
                    r.direct_coef_se,
                    r.mediator_outcome_coef_se,
                    r.interaction_coef_se.unwrap(),
                ],
                Tol::se(n),
            )
            .scalar("outi_r2", r.outcome_r_squared, Tol::stat(n))
            .finish();
    }

    #[test]
    fn test_validate_na_medflex_reports_missing() {
        let (case, ds) = crate::test_support::load_case("causal_mediation_na");
        let r = run_medflex_dataset(&ds, "y", "d", "m", &["x1", "x2"], na_cfg(false)).unwrap();
        assert_eq!(
            (r.missing.n_total, r.missing.n_used, r.missing.n_dropped),
            (400, 350, 50)
        );
        assert_eq!(
            r.missing.by_column,
            [
                ("y".to_string(), 15),
                ("m".to_string(), 25),
                ("x1".to_string(), 10)
            ]
        );
        Check::new(&case)
            .scalar("n_dropped", r.missing.n_dropped as f64, Tol::exact())
            .scalar("missing_y", nf::by_column(&r.missing, "y"), Tol::exact())
            .scalar("missing_m", nf::by_column(&r.missing, "m"), Tol::exact())
            .scalar("missing_x1", nf::by_column(&r.missing, "x1"), Tol::exact())
            .finish();
        let text = r.to_string();
        assert!(
            text.contains("N = 350 (treated: 197, control: 153)"),
            "{text}"
        );
        assert!(
            text.trim_end().ends_with(
                "Missing values: 50 of 400 rows dropped because of missing values \
                 (y: 15, m: 25, x1: 10)"
            ),
            "{text}"
        );
        assert_eq!(
            serde_json::to_value(&r).unwrap()["missing"]["n_dropped"],
            50
        );
        // complete rows: nothing dropped, no key, no line
        let cc = run_medflex_dataset(
            &nf::complete_rows(&case, &ds),
            "y",
            "d",
            "m",
            &["x1", "x2"],
            na_cfg(false),
        )
        .unwrap();
        let old = serde_json::to_value(&cc).unwrap();
        assert!(old.get("missing").is_none(), "{old}");
        assert!(!cc.to_string().contains("Missing values"));
        // a result serialised before the field existed reads back as "nothing dropped"
        let back: MedflexResult = serde_json::from_value(old).unwrap();
        assert!(back.missing.is_empty());
    }

    #[test]
    fn test_validate_na_medflex_errors_name_the_column() {
        let (case, ds) = crate::test_support::load_case("causal_mediation_na");
        match run_medflex_dataset(&ds, "y", "d_txt", "m", &["x1", "x2"], na_cfg(false)) {
            Err(EconError::NonNumericColumn { column }) => assert_eq!(column, "d_txt"),
            other => panic!("text treatment: {other:?}"),
        }
        match run_medflex_dataset(&ds, "y", "d", "m", &["x1", "x9"], na_cfg(false)) {
            Err(EconError::ColumnNotFound { column, .. }) => assert_eq!(column, "x9"),
            other => panic!("absent confounder: {other:?}"),
        }
        let cc = nf::complete_rows(&case, &ds);
        let r = run_medflex_dataset(
            &nf::with_nan(&cc, "m", 0),
            "y",
            "d",
            "m",
            &["x1", "x2"],
            na_cfg(false),
        )
        .unwrap();
        assert_eq!(r.missing.by_column, [("m".to_string(), 1)]);
        assert_eq!((r.missing.n_dropped, r.n_obs), (1, 349));
        // an all-missing mediator: the error names the method
        let mut df = ds.df().clone();
        df.with_column(Series::full_null("m".into(), 400, &DataType::Float64))
            .unwrap();
        match run_medflex_dataset(
            &Dataset::new(df),
            "y",
            "d",
            "m",
            &["x1", "x2"],
            na_cfg(false),
        ) {
            Err(EconError::InsufficientData {
                required: 1,
                provided: 0,
                context,
            }) => assert!(context.contains("natural effect models"), "{context}"),
            other => panic!("all-missing mediator: {other:?}"),
        }
    }

    // =========================================================================
    // T3-WP26: the imputation estimator on medflex_scales
    // =========================================================================

    const EFF: [&str; 3] = ["nde", "nie", "te"];

    /// `validation/datasets/generated/medflex_sim.csv` (the data of case `medflex_scales`),
    /// for tests that pin no R value.
    fn medflex_sim() -> Dataset {
        let path = p2a_validation::repo_path("validation/datasets/generated/medflex_sim.csv");
        match crate::data::DataLoader::load_csv(&path) {
            Ok(df) => Dataset::new(df),
            Err(e) => panic!("cannot load {}: {e}", path.display()),
        }
    }

    fn cfg(scale: EffectScale, family: Option<MedflexFamily>, interaction: bool) -> MedflexConfig {
        let mut c = MedflexConfig::default();
        c.scale = scale;
        c.family = family;
        c.allow_interaction = interaction;
        c.bootstrap_ci = false; // robust sandwich SEs (medflex se = "robust")
        c
    }

    fn fit(ds: &Dataset, y: &str, m: &str, conf: &[&str], c: MedflexConfig) -> MedflexResult {
        run_medflex_dataset(ds, y, "a", m, conf, c).unwrap()
    }

    /// Every number the robust path reports for one configuration of medflex_scales.
    fn pin(case: &Case, key: &str, r: &MedflexResult, iterative: bool) {
        let n = r.n_obs;
        // poisson/binomial: fit_glm epsilon 1e-10 against R at 1e-14 (6.4e-13 measured);
        // the working model's summary.glm SEs come from the last IRLS step's working
        // weights and converge only linearly under the deviance-change rule (1.8e-7
        // measured, binomial), as in glm.rs; gaussian: README tolerances
        let (tc, ts, tp, tw) = if iterative {
            (
                Tol::iterative(1e-8),
                Tol::iterative(1e-8),
                Tol::iterative(1e-8),
                Tol::iterative(1e-6),
            )
        } else {
            (Tol::coef(n), Tol::se(n), Tol::p_value(n), Tol::se(n))
        };
        Check::new(case)
            .named(
                &format!("{key}_outcome_coef"),
                &r.outcome_coef_names,
                &r.outcome_coefficients,
                tc,
            )
            .named(
                &format!("{key}_outcome_se"),
                &r.outcome_coef_names,
                &r.outcome_std_errors,
                tw,
            )
            .named(
                &format!("{key}_ne_coef"),
                &r.ne_coef_names,
                &r.ne_coefficients,
                tc,
            )
            .scalar(&format!("{key}_nde"), r.natural_direct_effect, tc)
            .scalar(&format!("{key}_nie"), r.natural_indirect_effect, tc)
            .scalar(&format!("{key}_te"), r.total_effect, tc)
            .named(
                &format!("{key}_robust_se"),
                &EFF[..],
                &[r.nde_se, r.nie_se, r.te_se],
                ts,
            )
            .named(
                &format!("{key}_robust_ci_lower"),
                &EFF[..],
                &[r.nde_ci.0, r.nie_ci.0, r.te_ci.0],
                ts,
            )
            .named(
                &format!("{key}_robust_ci_upper"),
                &EFF[..],
                &[r.nde_ci.1, r.nie_ci.1, r.te_ci.1],
                ts,
            )
            .named(
                &format!("{key}_robust_p"),
                &EFF[..],
                &[r.nde_p_value, r.nie_p_value, r.te_p_value],
                tp,
            )
            .scalar(&format!("{key}_pm"), r.proportion_mediated, tc)
            .scalar(
                &format!("{key}_deviance_r2"),
                r.outcome_r_squared,
                Tol::stat(n),
            )
            .finish();
    }

    fn case_inputs(ds: &Dataset, y: &str, m: &str, conf: &[&str]) -> Inputs {
        let cols = ColumnRequest::new(ds.df())
            .numeric(y)
            .numeric("a")
            .numeric(m)
            .numerics(conf)
            .na(NaPolicy::Fail)
            .extract()
            .unwrap();
        Inputs {
            y: cols.numeric_array(y),
            a: cols.numeric_array("a"),
            m: cols.numeric_array(m),
            c: cols.matrix(conf, false).unwrap().data,
        }
    }

    fn case_plan(inp: &Inputs, y: &str, m: &str, conf: &[&str], c: &MedflexConfig) -> Plan {
        let names = MedflexNames {
            outcome: y.into(),
            treatment: "a".into(),
            mediator: m.into(),
            confounders: conf.iter().map(|s| s.to_string()).collect(),
        };
        validate_and_plan(inp, &names, c).unwrap()
    }

    /// The case's 32-bit LCG: s <- (69069 s + 1) mod 2^32, row = floor(s n / 2^32).
    fn lcg_rows(seed: u64, n: usize) -> impl FnMut(&mut [usize]) {
        let mut s = seed;
        move |rows: &mut [usize]| {
            for r in rows.iter_mut() {
                s = (69069 * s + 1) % (1u64 << 32);
                *r = ((s * n as u64) >> 32) as usize;
            }
        }
    }

    #[test]
    fn test_validate_medflex_difference() {
        let (case, ds) = crate::test_support::load_case("medflex_scales");
        let r = fit(
            &ds,
            "yc",
            "m",
            &["c1", "c2"],
            cfg(EffectScale::Difference, None, false),
        );
        pin(&case, "diff", &r, false);
        let n = r.n_obs;
        Check::new(&case)
            .scalar("n", n as f64, Tol::exact())
            .scalar("n_treated", r.n_treated as f64, Tol::exact())
            .named("cbar", &["c1", "c2"], &r.confounder_means, Tol::coef(n))
            .named_subset("med_coef", &["a"], &["a"], &[r.mediator_coef], Tol::coef(n))
            .named_subset("med_se", &["a"], &["a"], &[r.mediator_coef_se], Tol::se(n))
            .scalar("med_r2", r.mediator_r_squared, Tol::stat(n))
            .scalar("mean_m_control", r.mediator_mean_control, Tol::coef(n))
            .scalar("mean_m_treated", r.mediator_mean_treated, Tol::coef(n))
            .finish();
        assert_eq!(r.family, MedflexFamily::Gaussian);
        assert_eq!(
            r.ne_model,
            "yhat ~ a0 + a1 + c1 + c2 (gaussian, identity link)"
        );
        assert!(
            r.se_method.starts_with("robust sandwich"),
            "{}",
            r.se_method
        );
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
    }

    #[test]
    fn test_validate_medflex_ratio() {
        let (case, ds) = crate::test_support::load_case("medflex_scales");
        let r = fit(
            &ds,
            "yp",
            "m",
            &["c1", "c2"],
            cfg(EffectScale::Ratio, None, false),
        );
        pin(&case, "ratio", &r, true);
        assert_eq!(r.family, MedflexFamily::Poisson);
        assert_eq!(
            r.ne_model,
            "yhat ~ a0 + a1 + c1 + c2 (quasipoisson, log link)"
        );
        // log-linear working model without A:M: the NDE is exp(b_A) exactly
        assert!(
            (r.natural_direct_effect.ln() - case.named("ratio_outcome_coef", "a")).abs() < 1e-10
        );
        // the difference scale on the same outcome is another estimand
        let d = fit(
            &ds,
            "yp",
            "m",
            &["c1", "c2"],
            cfg(EffectScale::Difference, None, false),
        );
        assert!((r.natural_direct_effect - d.natural_direct_effect).abs() > 1.0);
    }

    #[test]
    fn test_validate_medflex_odds_ratio() {
        let (case, ds) = crate::test_support::load_case("medflex_scales");
        let r = fit(
            &ds,
            "yb",
            "m",
            &["c1", "c2"],
            cfg(EffectScale::OddsRatio, None, false),
        );
        pin(&case, "odds", &r, true);
        assert_eq!(r.family, MedflexFamily::Binomial);
        // odds ratio 1.643 vs risk ratio 1.311 on the same binary outcome
        let rr = fit(
            &ds,
            "yb",
            "m",
            &["c1", "c2"],
            cfg(EffectScale::Ratio, Some(MedflexFamily::Binomial), false),
        );
        assert!((r.natural_direct_effect - rr.natural_direct_effect).abs() > 0.2);
    }

    #[test]
    fn test_validate_medflex_family() {
        let (case, ds) = crate::test_support::load_case("medflex_scales");
        let rr = fit(
            &ds,
            "yb",
            "m",
            &["c1", "c2"],
            cfg(EffectScale::Ratio, Some(MedflexFamily::Binomial), false),
        );
        pin(&case, "rr_bin", &rr, true);
        let rd = fit(
            &ds,
            "yb",
            "m",
            &["c1", "c2"],
            cfg(
                EffectScale::Difference,
                Some(MedflexFamily::Binomial),
                false,
            ),
        );
        pin(&case, "rd_bin", &rd, true);
        // the linear-probability working model gives another risk difference
        let lpm = fit(
            &ds,
            "yb",
            "m",
            &["c1", "c2"],
            cfg(
                EffectScale::Difference,
                Some(MedflexFamily::Gaussian),
                false,
            ),
        );
        assert!((rd.natural_direct_effect - lpm.natural_direct_effect).abs() > 1e-3);
        assert_eq!(
            (rr.family, rd.family, lpm.family),
            (
                MedflexFamily::Binomial,
                MedflexFamily::Binomial,
                MedflexFamily::Gaussian
            )
        );
    }

    #[test]
    fn test_validate_medflex_interaction() {
        // CAUS-23: the NDE uses the confounder-adjusted mediator distribution
        let (case, ds) = crate::test_support::load_case("medflex_scales");
        let d = fit(
            &ds,
            "yi",
            "m",
            &["c1", "c2"],
            cfg(EffectScale::Difference, None, true),
        );
        pin(&case, "diff_int", &d, false);
        Check::new(&case)
            .named(
                "diff_int_vanderweele",
                &EFF[..2],
                &[d.natural_direct_effect, d.natural_indirect_effect],
                Tol::coef(d.n_obs),
            )
            .named(
                "cbar",
                &["c1", "c2"],
                &d.confounder_means,
                Tol::coef(d.n_obs),
            )
            .finish();
        // the pre-fix formula plugged in the raw control-group mediator mean: 0.70020 vs 0.76811
        assert!(
            (d.natural_direct_effect - case.scalar("diff_int_nde_raw_control_mean")).abs() > 0.05
        );
        assert_eq!(d.interaction_coef, Some(d.outcome_coefficients[5]));
        let r = fit(
            &ds,
            "yp",
            "m",
            &["c1", "c2"],
            cfg(EffectScale::Ratio, None, true),
        );
        pin(&case, "ratio_int", &r, true);
        let o = fit(
            &ds,
            "yb",
            "m",
            &["c1", "c2"],
            cfg(EffectScale::OddsRatio, None, true),
        );
        pin(&case, "odds_int", &o, true);
        assert_eq!(
            o.ne_model,
            "yhat ~ a0 + a1 + c1 + c2 + a0:a1 + a0:c1 + a0:c2 (quasibinomial, logit link)"
        );
    }

    #[test]
    fn test_validate_medflex_bootstrap() {
        let (case, ds) = crate::test_support::load_case("medflex_scales");
        let conf = ["c1", "c2"];
        let inp = case_inputs(&ds, "yb", "m", &conf);
        let mut c = cfg(EffectScale::OddsRatio, None, true);
        c.bootstrap_ci = true;
        c.n_bootstrap = 50;
        let plan = case_plan(&inp, "yb", "m", &conf, &c);
        // the LCG reproduces R's resample rows
        let (mut g, mut rows) = (lcg_rows(20261130, 500), vec![0usize; 500]);
        g(&mut rows);
        assert_eq!(
            rows[..5].iter().map(|&v| v as f64).collect::<Vec<_>>(),
            case.vector("boot_rows_head")
        );
        for _ in 1..50 {
            g(&mut rows);
        }
        assert_eq!(
            rows[495..].iter().map(|&v| v as f64).collect::<Vec<_>>(),
            case.vector("boot_rows_tail")
        );
        let draws = bootstrap_link_draws(&inp, &plan, 50, &mut lcg_rows(20261130, 500)).unwrap();
        assert_eq!((draws.failed, draws.warned), (0, 0));
        let link_hat = estimate(&inp, &plan).unwrap().link;
        let (se, ci, notes) = summarize_bootstrap(&draws, &plan, &link_hat, 0.95).unwrap();
        assert!(notes.is_empty(), "{notes:?}");
        let t = Tol::iterative(1e-8);
        Check::new(&case)
            .vector("odds_int_boot_draws_nde", &draws.link[0], t)
            .named("odds_int_boot_se", &EFF[..], &se, t)
            .named(
                "odds_int_boot_ci_lower",
                &EFF[..],
                &[ci[0].0, ci[1].0, ci[2].0],
                t,
            )
            .named(
                "odds_int_boot_ci_upper",
                &EFF[..],
                &[ci[0].1, ci[1].1, ci[2].1],
                t,
            )
            .finish();
    }

    #[test]
    fn test_validate_medflex_proportion_mediated_unclamped() {
        // CAUS-75 (medflex): -4.198, which the old code clamped to -2
        let (case, ds) = crate::test_support::load_case("medflex_scales");
        let r = fit(
            &ds,
            "yo",
            "m",
            &["c1", "c2"],
            cfg(EffectScale::Difference, None, false),
        );
        pin(&case, "diff_opp", &r, false);
        assert!(
            r.warnings.iter().any(|w| w
                == "proportion mediated -4.1977 lies outside [0, 1]: the direct and \
                    indirect effects have opposite signs or the total effect is close to the null"),
            "{:?}",
            r.warnings
        );
    }

    #[test]
    fn test_validate_medflex_collinear_design_warns() {
        // CAUS-73: aliased columns are NaN with R's warning, not a silent pseudo-inverse
        let (case, ds) = crate::test_support::load_case("medflex_scales");
        for (key, inter) in [("alias", false), ("alias_int", true)] {
            let r = fit(
                &ds,
                "y_lin",
                "m_lin",
                &["c1", "c2", "c1dup"],
                cfg(EffectScale::Difference, None, inter),
            );
            let n = r.n_obs;
            Check::new(&case)
                .named(
                    &format!("{key}_outcome_coef"),
                    &r.outcome_coef_names,
                    &r.outcome_coefficients,
                    Tol::coef(n),
                )
                .named(
                    &format!("{key}_ne_coef"),
                    &r.ne_coef_names,
                    &r.ne_coefficients,
                    Tol::coef(n),
                )
                .scalar(&format!("{key}_nde"), r.natural_direct_effect, Tol::coef(n))
                // ~1e-16: the absolute floor 1e-12 applies
                .scalar(
                    &format!("{key}_nie"),
                    r.natural_indirect_effect,
                    Tol::coef(n),
                )
                .scalar(&format!("{key}_te"), r.total_effect, Tol::coef(n))
                .named(
                    &format!("{key}_robust_se"),
                    &EFF[..],
                    &[r.nde_se, r.nie_se, r.te_se],
                    Tol::se(n),
                )
                .finish();
            let ne_alias = if inter { "c1dup, a0:c1dup" } else { "c1dup" };
            assert_eq!(
                r.warnings,
                vec![
                    "Outcome working model: Coefficients not defined because of singularities: c1, c1dup"
                        .to_string(),
                    format!(
                        "Natural effect model: Coefficients not defined because of singularities: {ne_alias}"
                    ),
                    "Mediator model: Coefficients not defined because of singularities: c1dup"
                        .to_string(),
                ]
            );
            assert!(r.to_string().contains(
                "Warning: Outcome working model: Coefficients not defined because of singularities: c1, c1dup"
            ));
            // the duplicate changes nothing estimable
            let s = fit(
                &ds,
                "y_lin",
                "m_lin",
                &["c1", "c2"],
                cfg(EffectScale::Difference, None, inter),
            );
            let t = Tol::new(1e-10, 1e-12);
            for (u, v) in [
                (s.natural_direct_effect, r.natural_direct_effect),
                (s.total_effect, r.total_effect),
                (s.nde_se, r.nde_se),
                (s.te_se, r.te_se),
            ] {
                assert!(t.allows(u, v), "{key}: {u} vs {v}");
            }
            if !inter {
                Check::new(&case)
                    .named_subset(
                        "alias_med_coef",
                        &["a"],
                        &["a"],
                        &[r.mediator_coef],
                        Tol::coef(n),
                    )
                    .finish();
            }
        }
    }

    #[test]
    fn test_medflex_closed_form_matches_imputation() {
        // the bootstrap's closed form (gaussian working model, difference scale) gives the
        // estimator's link-scale effects
        let ds = medflex_sim();
        let t = Tol::new(1e-10, 1e-12);
        let configs: [(&str, &str, &[&str], bool); 5] = [
            ("yc", "m", &["c1", "c2"], false),
            ("yi", "m", &["c1", "c2"], true),
            ("yo", "m", &["c1", "c2"], false),
            ("y_lin", "m_lin", &["c1", "c2", "c1dup"], false),
            ("y_lin", "m_lin", &["c1", "c2", "c1dup"], true),
        ];
        for (y, m, conf, inter) in configs {
            let inp = case_inputs(&ds, y, m, conf);
            let c = cfg(EffectScale::Difference, None, inter);
            let plan = case_plan(&inp, y, m, conf, &c);
            let generic = estimate(&inp, &plan).unwrap().link;
            let (closed, _) = closed_form(&inp, &plan).unwrap();
            for e in 0..3 {
                assert!(
                    t.allows(closed[e], generic[e]),
                    "{y} A:M={inter} {}: {} vs {}",
                    EFF[e],
                    closed[e],
                    generic[e]
                );
            }
        }
        // and on resamples
        let conf = ["c1", "c2"];
        let inp = case_inputs(&ds, "yi", "m", &conf);
        let plan = case_plan(
            &inp,
            "yi",
            "m",
            &conf,
            &cfg(EffectScale::Difference, None, true),
        );
        let (mut g, mut rows) = (lcg_rows(7, 500), vec![0usize; 500]);
        for b in 0..5 {
            g(&mut rows);
            let sample = inp.resample(&rows).unwrap();
            let generic = estimate(&sample, &plan).unwrap().link;
            let (closed, _) = closed_form(&sample, &plan).unwrap();
            for e in 0..3 {
                assert!(
                    t.allows(closed[e], generic[e]),
                    "resample {b} {}: {} vs {}",
                    EFF[e],
                    closed[e],
                    generic[e]
                );
            }
        }
    }

    #[test]
    fn test_medflex_rejects_degenerate_inputs() {
        let ds = medflex_sim();
        let conf = ["c1", "c2"];
        let run = |y: &str, c: MedflexConfig| run_medflex_dataset(&ds, y, "a", "m", &conf, c);
        use EffectScale::*;
        expect_invalid_spec(
            run("yc", cfg(OddsRatio, Some(MedflexFamily::Gaussian), false)).unwrap_err(),
            "needs imputed outcomes in [0, 1]",
        );
        expect_invalid_spec(
            run("yo", cfg(Ratio, Some(MedflexFamily::Gaussian), false)).unwrap_err(),
            "needs non-negative imputed outcomes",
        );
        expect_invalid_spec(
            run("yc", cfg(OddsRatio, None, false)).unwrap_err(),
            "outcome working model: y values must be 0 <= y <= 1",
        );
        expect_invalid_spec(
            run("yo", cfg(Ratio, None, false)).unwrap_err(),
            "outcome working model: negative values not allowed",
        );
        expect_invalid_spec(
            run_medflex_dataset(&ds, "yc", "yp", "m", &conf, cfg(Difference, None, false))
                .unwrap_err(),
            "binary 0/1 treatment",
        );
        let inp = case_inputs(&ds, "yc", "m", &conf);
        let mut c3 = Array2::zeros((inp.n(), 3));
        c3.slice_mut(ndarray::s![.., 0..2]).assign(&inp.c);
        c3.column_mut(2).assign(&inp.a); // a confounder equal to the treatment
        expect_invalid_spec(
            run_medflex(
                &inp.y.view(),
                &inp.a.view(),
                &inp.m.view(),
                &c3.view(),
                cfg(Difference, None, true),
            )
            .unwrap_err(),
            "no overlap",
        );
        expect_invalid_spec(
            run_medflex(
                &inp.y.view(),
                &inp.a.view(),
                &inp.a.view(),
                &inp.c.view(),
                cfg(Difference, None, true),
            )
            .unwrap_err(),
            "the mediator column 'M' is aliased in the outcome working model",
        );
        let names = MedflexNames::generic(3);
        expect_invalid_spec(
            run_medflex_with_names(
                &inp.y.view(),
                &inp.a.view(),
                &inp.m.view(),
                &inp.c.view(),
                &names,
                cfg(Difference, None, false),
            )
            .unwrap_err(),
            "names.confounders has 3 names but confounders has 2 columns",
        );
        let mut y_inf = inp.y.clone();
        y_inf[7] = f64::INFINITY;
        expect_invalid_spec(
            run_medflex(
                &y_inf.view(),
                &inp.a.view(),
                &inp.m.view(),
                &inp.c.view(),
                cfg(Difference, None, false),
            )
            .unwrap_err(),
            "outcome has a non-finite value at row 7",
        );
    }

    #[test]
    fn test_medflex_bootstrap_failures_are_reported() {
        // CAUS-75 (medflex bootstrap): failed replicates are counted and named
        let ds = medflex_sim();
        let conf = ["c1", "c2"];
        let inp = case_inputs(&ds, "yc", "m", &conf);
        let mut c = cfg(EffectScale::Difference, None, false);
        c.bootstrap_ci = true;
        let plan = case_plan(&inp, "yc", "m", &conf, &c);
        let controls: Vec<usize> = (0..inp.n()).filter(|&i| inp.a[i] == 0.0).collect();
        let failing = |every: usize| {
            let (mut lcg, mut b) = (lcg_rows(7, 500), 0usize);
            let controls = controls.clone();
            move |rows: &mut [usize]| {
                b += 1;
                if b % every == 0 {
                    for (k, r) in rows.iter_mut().enumerate() {
                        *r = controls[k % controls.len()];
                    }
                } else {
                    lcg(rows)
                }
            }
        };
        let first = "Invalid model specification: natural effects need treated and control units (treated: 0, control: 500)";
        let d = bootstrap_link_draws(&inp, &plan, 60, &mut failing(6)).unwrap();
        assert_eq!((d.failed, d.link[0].len()), (10, 50));
        let (_, _, notes) = summarize_bootstrap(&d, &plan, &[0.0; 3], 0.95).unwrap();
        assert_eq!(
            notes,
            vec![format!(
                "10 of 60 bootstrap replicates failed and were dropped (first error: {first})"
            )]
        );
        let d = bootstrap_link_draws(&inp, &plan, 59, &mut failing(5)).unwrap(); // 11 fail, 48 usable
        match summarize_bootstrap(&d, &plan, &[0.0; 3], 0.95) {
            Err(EconError::Computation(m)) => assert_eq!(
                m,
                format!(
                    "only 48 of 59 bootstrap replicates could be fitted (first error: {first}); the bootstrap needs at least 50"
                )
            ),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn test_medflex_default_seed_is_fixed() {
        // no seed means seed 42: repeated calls are identical
        let (y, t, m, c) = create_test_data();
        let run = |seed: Option<u64>| {
            let mut k = MedflexConfig::default();
            k.allow_interaction = false;
            k.n_bootstrap = 60;
            k.seed = seed;
            run_medflex(&y.view(), &t.view(), &m.view(), &c.view(), k).unwrap()
        };
        let (a, b, d) = (run(None), run(None), run(Some(DEFAULT_BOOTSTRAP_SEED)));
        assert_eq!(
            (a.te_se.to_bits(), a.nde_ci.0.to_bits()),
            (b.te_se.to_bits(), b.nde_ci.0.to_bits())
        );
        assert_eq!(a.nie_se.to_bits(), d.nie_se.to_bits());
        assert!(
            a.se_method.starts_with("bootstrap (60 of 60 replicates"),
            "{}",
            a.se_method
        );
    }

    #[test]
    fn test_medflex_mediator_scale_invariance() {
        // every fit is scale-free in M
        let ds = medflex_sim();
        for (y, scale) in [
            ("yc", EffectScale::Difference),
            ("yp", EffectScale::Ratio),
            ("yb", EffectScale::OddsRatio),
        ] {
            for inter in [false, true] {
                let inp = case_inputs(&ds, y, "m", &["c1", "c2"]);
                let base = run_medflex(
                    &inp.y.view(),
                    &inp.a.view(),
                    &inp.m.view(),
                    &inp.c.view(),
                    cfg(scale, None, inter),
                )
                .unwrap();
                for s in [1e-6, 1e6] {
                    let ms = inp.m.mapv(|v| v * s);
                    let r = run_medflex(
                        &inp.y.view(),
                        &inp.a.view(),
                        &ms.view(),
                        &inp.c.view(),
                        cfg(scale, None, inter),
                    )
                    .unwrap();
                    let t = Tol::new(1e-9, 1e-12);
                    for (u, v) in [
                        (r.natural_direct_effect, base.natural_direct_effect),
                        (r.natural_indirect_effect, base.natural_indirect_effect),
                        (r.total_effect, base.total_effect),
                        (r.nde_se, base.nde_se),
                        (r.nie_se, base.nie_se),
                        (r.te_se, base.te_se),
                    ] {
                        assert!(t.allows(u, v), "{y} A:M={inter} M x {s}: {u} vs {v}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_medflex_display_reports_scale_and_method() {
        // CAUS-22: the label is the estimator
        let ds = medflex_sim();
        let s = fit(
            &ds,
            "yb",
            "m",
            &["c1", "c2"],
            cfg(EffectScale::OddsRatio, None, true),
        )
        .to_string();
        for needle in [
            "Effect Decomposition (Odds Ratio scale):",
            "Std. errors and p-values are on the logit scale; estimates and intervals are odds ratios.",
            "Natural effect model: yhat ~ a0 + a1 + c1 + c2 + a0:a1 + a0:c1 + a0:c2 (quasibinomial, logit link)",
            "Standard errors: robust sandwich",
            "NDE and TE evaluated at the confounder means: c1 = -0.0076, c2 = 0.4960",
            "Outcome working model (binomial, logit link): yb ~ a + m + c1 + c2 + a:m",
            "N = 500 (treated: 201, control: 299)",
        ] {
            assert!(s.contains(needle), "missing {needle:?} in\n{s}");
        }
        let mut c = cfg(EffectScale::OddsRatio, None, true);
        c.bootstrap_ci = true;
        c.n_bootstrap = 50;
        c.seed = Some(3);
        let b = fit(&ds, "yb", "m", &["c1", "c2"], c).to_string();
        assert!(
            b.contains(
                "Standard errors: bootstrap (50 of 50 replicates; SE = SD of the logit-scale draws, \
                 normal intervals as medflex confint(type = \"norm\"))"
            ),
            "{b}"
        );
    }

    #[test]
    fn test_medflex_result_serde_round_trip() {
        let ds = medflex_sim();
        let r = fit(
            &ds,
            "y_lin",
            "m_lin",
            &["c1", "c2", "c1dup"],
            cfg(EffectScale::Difference, None, false),
        );
        let v = serde_json::to_value(&r).unwrap();
        // c1, c1dup
        assert!(v["outcome_coefficients"][3].is_null() && v["outcome_coefficients"][5].is_null());
        let back: MedflexResult = serde_json::from_value(v).unwrap();
        assert!(back.outcome_coefficients[3].is_nan() && back.ne_coefficients[5].is_nan());
        assert_eq!(
            back.outcome_coefficients[1].to_bits(),
            r.outcome_coefficients[1].to_bits()
        );
        assert_eq!(
            (back.family, back.warnings.clone(), back.ne_model.clone()),
            (r.family, r.warnings.clone(), r.ne_model.clone())
        );
    }
}
