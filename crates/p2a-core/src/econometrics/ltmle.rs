//! Longitudinal Targeted Maximum Likelihood Estimation (LTMLE) for causal inference.
//!
//! LTMLE estimates the mean outcome `E[Y^d]` under a treatment regime
//! `d = (d_1, ..., d_T)` assigned over `T` time points, with time-varying confounders
//! `L_t` affected by earlier treatment, by sequential (iterated conditional expectation)
//! regression and a targeting step at every node. The implementation follows the R
//! package `ltmle` 1.x with its defaults (Lendle et al. 2017): `stratify = FALSE`, main-term
//! `glm` models on the full history, cumulative `gbounds` and `variance.method = "ic"`.
//!
//! # Algorithm
//!
//! The nodes are ordered `L_1, A_1, L_2, A_2, ..., L_T, A_T, Y`.
//!
//! 1. **Treatment models.** `g_t = P(A_t = 1 | L_1, A_1, ..., L_t)`: a logistic regression of
//!    `A_t` on every earlier node (main terms, time order), fitted on all subjects.
//! 2. **Cumulative probability of following the regime.** `g_t` is predicted with every
//!    earlier `A_s` set to the regime (ltmle's `SetA`); `c_t = prod_{s <= t} P(A_s = d_s |
//!    ...)` is bounded to `gbounds` (ltmle bounds the cumulative product, not each factor).
//! 3. **Clever covariate.** `H_t = I(Abar_t = dbar_t) / c_t`: zero for subjects who left the
//!    regime before or at `t`.
//! 4. **Backward pass** from `t = T` to 1, starting from `Q*_{T+1} = Y`: regress `Q*_{t+1}` on
//!    the history through `A_t` (pooled over all subjects), predict with `A_1..A_t` set to the
//!    regime, and fluctuate with an intercept-only quasibinomial fit with offset `logit Q_t(d)`
//!    and weights `H_t` among the followers; its prediction is `Q*_t`.
//! 5. **Estimate and inference.** `psi = mean(Q*_1)`; the efficient influence curve is
//!    `EIC = sum_t H_t (Q*_{t+1} - Q*_t) + Q*_1 - psi` and `SE = sd(EIC) / sqrt(n)`
//!    (ltmle's `variance.method = "ic"`; its default `"tmle"` substitution variance is not
//!    reproduced). A contrast differences the two regimes' EICs.
//!
//! With the logistic Q model an outcome outside [0, 1] is rescaled by its range and the
//! estimates are scaled back (ltmle's `Yrange`). The linear Q model is p2a's Gaussian
//! variant: least-squares Q fits and a weighted-intercept fluctuation on the identity scale.
//!
//! # Regimes and contrasts
//!
//! [`LtmleConfig::intervention`] is the [`Regime`] whose mean `E[Y^d]` is estimated:
//!
//! - **Static** ([`Regime::Static`]): a fixed treatment per time point, one value for every
//!   node (`Static(vec![1])`, always treat; `Static(vec![0])`, never treat) or one value per
//!   node (`Static(vec![1, 0])`, treat at the first node only).
//! - **Dynamic** ([`Regime::Dynamic`]): one threshold rule per node,
//!   `d_t = above if L > threshold else 1 - above`, on a covariate observed at or before the
//!   node (ltmle's `rule`; [`Regime::threshold_rules`] builds `A_t = 1{L_t > c}`).
//!
//! [`LtmleConfig::contrast`] is the reference regime (default never treat): the result then
//! reports `psi(intervention) - psi(contrast)` with the differenced EIC; `None` estimates
//! the single mean.
//!
//! # Example
//!
//! ```no_run
//! use ndarray::{Array1, Array2};
//! use p2a_core::econometrics::ltmle::{DynamicRule, Regime};
//! use p2a_core::econometrics::{LtmleConfig, LtmleData, run_ltmle};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let n = 8;
//! let l0 = Array2::from_shape_vec((n, 1), vec![-1.2, 0.4, 0.9, -0.3, 1.5, -0.8, 0.2, 0.7])?;
//! let l1 = Array2::from_shape_vec((n, 1), vec![-0.5, 0.8, 1.1, 0.1, 1.9, -1.0, 0.6, 0.3])?;
//! let a1 = Array1::from(vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
//! let a2 = Array1::from(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
//! let y = Array1::from(vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
//! let data = LtmleData::new(vec![Array1::zeros(n), y], vec![a1, a2], vec![l0, l1])?;
//!
//! let mut config = LtmleConfig::default();
//! config.intervention = Regime::Dynamic(vec![
//!     DynamicRule { node: 0, column: "L1_1".into(), threshold: 0.0, above: 1 },
//!     DynamicRule { node: 1, column: "L2_1".into(), threshold: 0.0, above: 1 },
//! ]);
//! config.contrast = Some(Regime::never());
//! let result = run_ltmle(&data, config)?;
//! println!("{result}");
//! # Ok(())
//! # }
//! ```
//!
//! # References
//!
//! - van der Laan, M.J. & Gruber, S. (2012). "Targeted Minimum Loss Based
//!   Estimation of Causal Effects of Multiple Time Point Interventions."
//!   *The International Journal of Biostatistics*, 8(1), Article 9.
//!   https://doi.org/10.1515/1557-4679.1370
//!
//! - Lendle, S.D., Schwab, J., Petersen, M.L., & van der Laan, M.J. (2017).
//!   "ltmle: An R Package Implementing Targeted Minimum Loss-Based Estimation
//!   for Longitudinal Data." *Journal of Statistical Software*, 81(1), 1-21.
//!   https://doi.org/10.18637/jss.v081.i01
//!
//! - van der Laan, M.J. & Rose, S. (2011). *Targeted Learning: Causal Inference
//!   for Observational and Experimental Data*. Springer. Chapter 6.
//!   https://doi.org/10.1007/978-1-4419-9782-1
//!
//! - R package `ltmle`: https://cran.r-project.org/package=ltmle
//!   (Schwab, Lendle, Petersen, van der Laan)

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, NaPolicy};
use crate::econometrics::glm::{
    GlmControl, GlmFamily, GlmLink, GlmSpec, LeanStart, fit_glm, lean_logit_refined,
};
use crate::errors::{EconError, EconResult};
use crate::linalg::solve::aliasing_message;
use crate::regression::ols::solve_ols;
use crate::resource;
use crate::traits::convergence::{ResultNotes, notes_text};
use crate::traits::estimator::{SignificanceLevel, normal_p_value_two_sided, z_critical};

/// Variance estimator of every standard error.
const VARIANCE_METHOD: &str = "influence curve (ltmle variance.method = \"ic\")";

/// ltmle's positivity check (`ltmle:::CheckForVarianceWarning`): it warns when more than this
/// share of the subjects, pooled over the estimated regimes, have an unbounded cumulative g at
/// the last node below the lower g bound. Also the share of Q* at 0 or 1 that is flagged.
const POSITIVITY_SHARE: f64 = 0.01;
/// ltmle's rare-events check: a 0/1 outcome with a mean below this.
const RARE_OUTCOME_MEAN: f64 = 0.03;
/// A targeted prediction (on the fitted [0, 1] scale) this close to 0 or 1 counts as saturated.
const Q_STAR_AT_BOUND: f64 = 1e-8;
/// glm.fit's "fitted probabilities numerically 0 or 1 occurred" threshold.
const G_AT_BOUND: f64 = 10.0 * f64::EPSILON;

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Type of intervention for LTMLE.
///
/// Specifies the counterfactual treatment regime to estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum InterventionType {
    /// Always treat: A_t = 1 for all t
    #[default]
    AlwaysTreat,
    /// Never treat: A_t = 0 for all t
    NeverTreat,
    /// Custom static intervention: A_t = specified value for all t
    Static(bool),
}

impl fmt::Display for InterventionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InterventionType::AlwaysTreat => write!(f, "Always Treat (A=1)"),
            InterventionType::NeverTreat => write!(f, "Never Treat (A=0)"),
            InterventionType::Static(treat) => {
                write!(f, "Static (A={})", if *treat { 1 } else { 0 })
            }
        }
    }
}

/// Outcome model specification for LTMLE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LtmleQModel {
    /// Quasibinomial logistic regression (ltmle's Q model; default). An outcome outside
    /// [0, 1] is rescaled by its range.
    #[default]
    Logistic,
    /// Least-squares regression with a linear fluctuation (p2a's Gaussian variant).
    Linear,
}

impl fmt::Display for LtmleQModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LtmleQModel::Logistic => write!(f, "Logistic"),
            LtmleQModel::Linear => write!(f, "Linear"),
        }
    }
}

crate::option_enum!(
    LtmleQModel,
    crate::options::Choices {
        option: "LTMLE outcome model",
        values: &[
            (
                &["logistic", "logit", "binomial", "quasibinomial", "binary"],
                LtmleQModel::Logistic
            ),
            (
                &["linear", "gaussian", "ols", "continuous"],
                LtmleQModel::Linear
            ),
        ],
    },
    no_display
);

/// A treatment regime d = (d_1, ..., d_T): the treatment each subject is assigned at each
/// node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Regime {
    /// d_t = values[t] for every subject, each 0 or 1; one value is used at every node
    /// (`Static(vec![1])` = always treat, `Static(vec![1, 0])` = treat at the first node only).
    Static(Vec<u8>),
    /// One rule per node: d_t = `above` if the rule's covariate > threshold, else 1 - `above`.
    Dynamic(Vec<DynamicRule>),
}

impl Default for Regime {
    fn default() -> Self {
        Regime::always()
    }
}

/// d_node = above if column > threshold else 1 - above (ltmle `rule`; A_t = 1{L > c} with
/// above = 1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DynamicRule {
    /// 0-based time point whose treatment the rule assigns.
    pub node: usize,
    /// A covariate observed at or before `node` (a name in `LtmleData::covariate_names[s]`,
    /// s <= node).
    pub column: String,
    /// Finite threshold c.
    pub threshold: f64,
    /// Treatment when the covariate exceeds the threshold: 0 or 1.
    pub above: u8,
}

/// Regime keyword for MCP/CLI parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegimeKind {
    /// Always treat.
    Always,
    /// Never treat.
    Never,
    /// Threshold rules on covariates.
    Dynamic,
}

crate::option_enum!(
    RegimeKind,
    crate::options::Choices {
        option: "LTMLE regime",
        values: &[
            (
                &["always", "always_treat", "treat", "treated"],
                RegimeKind::Always
            ),
            (
                &["never", "never_treat", "control", "untreated"],
                RegimeKind::Never
            ),
            (&["dynamic", "rule", "threshold_rule"], RegimeKind::Dynamic),
        ],
    }
);

impl Regime {
    /// Always treat: `Static(vec![1])`.
    pub fn always() -> Self {
        Regime::Static(vec![1])
    }

    /// Never treat: `Static(vec![0])`.
    pub fn never() -> Self {
        Regime::Static(vec![0])
    }

    /// d_t = 1{columns[t] > threshold}, t = 0..columns.len().
    pub fn threshold_rules(columns: &[String], threshold: f64) -> Self {
        Regime::Dynamic(
            columns
                .iter()
                .enumerate()
                .map(|(node, c)| DynamicRule {
                    node,
                    column: c.clone(),
                    threshold,
                    above: 1,
                })
                .collect(),
        )
    }

    /// Regime from an option keyword: Always and Never refuse `dynamic_columns`; Dynamic
    /// needs both the columns (one per time point) and the threshold.
    pub fn from_kind(
        kind: RegimeKind,
        dynamic_columns: Option<&[String]>,
        threshold: Option<f64>,
    ) -> EconResult<Self> {
        match kind {
            RegimeKind::Always | RegimeKind::Never => {
                if dynamic_columns.is_some() {
                    return Err(spec_err(
                        "dynamic_columns is only used with regime 'dynamic'".into(),
                    ));
                }
                Ok(if kind == RegimeKind::Always {
                    Regime::always()
                } else {
                    Regime::never()
                })
            }
            RegimeKind::Dynamic => {
                let cols = dynamic_columns.ok_or_else(|| {
                    spec_err(
                        "regime 'dynamic' needs dynamic_columns (one covariate per time point)"
                            .into(),
                    )
                })?;
                let c = threshold
                    .ok_or_else(|| spec_err("regime 'dynamic' needs dynamic_threshold".into()))?;
                Ok(Regime::threshold_rules(cols, c))
            }
        }
    }

    /// "always treat" | "never treat" | "static (A1 = 1, A2 = 0)" |
    /// "dynamic (A1 = 1{L0 > 0}, A2 = 1{L1 > 0})" ("1{L0 <= 0}" when above = 0).
    pub fn label(&self, treatment_names: &[String]) -> String {
        let name = |t: usize| {
            treatment_names
                .get(t)
                .cloned()
                .unwrap_or_else(|| format!("A{}", t + 1))
        };
        match self {
            Regime::Static(v) if v.len() == 1 && v[0] == 1 => "always treat".into(),
            Regime::Static(v) if v.len() == 1 && v[0] == 0 => "never treat".into(),
            Regime::Static(v) if !v.is_empty() && v.iter().all(|x| *x == v[0]) => {
                if v[0] == 1 {
                    "always treat".into()
                } else if v[0] == 0 {
                    "never treat".into()
                } else {
                    format!("static ({v:?})")
                }
            }
            Regime::Static(v) => {
                let parts: Vec<String> = v
                    .iter()
                    .enumerate()
                    .map(|(t, x)| format!("{} = {x}", name(t)))
                    .collect();
                format!("static ({})", parts.join(", "))
            }
            Regime::Dynamic(rules) => {
                let mut rules: Vec<&DynamicRule> = rules.iter().collect();
                rules.sort_by_key(|r| r.node);
                let parts: Vec<String> = rules
                    .iter()
                    .map(|r| {
                        let op = if r.above == 0 { "<=" } else { ">" };
                        format!("{} = 1{{{} {op} {}}}", name(r.node), r.column, r.threshold)
                    })
                    .collect();
                format!("dynamic ({})", parts.join(", "))
            }
        }
    }
}

impl From<InterventionType> for Regime {
    fn from(i: InterventionType) -> Self {
        match i {
            InterventionType::AlwaysTreat => Regime::always(),
            InterventionType::NeverTreat => Regime::never(),
            InterventionType::Static(b) => Regime::Static(vec![u8::from(b)]),
        }
    }
}

/// Configuration for LTMLE estimation.
#[derive(Debug, Clone)]
pub struct LtmleConfig {
    /// Regime whose counterfactual mean E[Y^d] is estimated (default: always treat).
    pub intervention: Regime,
    /// Reference regime of the contrast psi(intervention) - psi(contrast) (default: never
    /// treat); `None` estimates E[Y^intervention] only (the contrast fields of the result
    /// are NaN).
    pub contrast: Option<Regime>,
    /// Logistic (default, ltmle; a Y outside [0, 1] is rescaled by its range) or Linear
    /// (p2a's Gaussian variant).
    pub q_model: LtmleQModel,
    /// Bounds on the cumulative probability of following the regime (ltmle `gbounds`);
    /// default (0.01, 1.0); `0 < lower < upper <= 1`.
    pub gbounds: (f64, f64),
    /// IRLS iteration limit of every g, Q and fluctuation fit (default 100, ltmle's maxit).
    pub max_iter: usize,
    /// Relative deviance-change tolerance of every iterative fit (glm.control's `epsilon`;
    /// default 1e-8); finite and > 0.
    pub tolerance: f64,
}

impl Default for LtmleConfig {
    fn default() -> Self {
        Self {
            intervention: Regime::always(),
            contrast: Some(Regime::never()),
            q_model: LtmleQModel::Logistic,
            gbounds: (0.01, 1.0),
            max_iter: 100,
            tolerance: 1e-8,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Data Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Longitudinal data structure for LTMLE.
///
/// Data is organized by time point (t = 1, ..., T):
/// - `outcomes[T-1]`: the outcome Y, observed after the last treatment; earlier entries
///   are not used (a warning is given when they are not zero)
/// - `treatments[t-1]`: Treatment A_t at time t (binary: 0 or 1)
/// - `covariates[t-1]`: Time-varying covariates L_t at time t (n x k_t matrix), observed
///   before A_t
///
/// The causal structure is:
/// L_1 -> A_1 -> L_2 -> A_2 -> ... -> L_T -> A_T -> Y
#[derive(Debug, Clone)]
pub struct LtmleData {
    /// Outcome at each time point; only the last one is the outcome node.
    pub outcomes: Vec<Array1<f64>>,
    /// Treatment indicator at each time point (0 or 1)
    pub treatments: Vec<Array1<f64>>,
    /// Time-varying covariates at each time point (n x k_t matrix)
    pub covariates: Vec<Array2<f64>>,
    /// Treatment names (default "A1", "A2", ...).
    pub treatment_names: Vec<String>,
    /// Covariate names per time point (default "L{t}_{j}", e.g. "L1_1").
    pub covariate_names: Vec<Vec<String>>,
}

impl LtmleData {
    /// Create new LTMLE data from arrays. Treatments must be 0/1; outcomes and covariates
    /// must be free of missing (NaN) and infinite values.
    pub fn new(
        outcomes: Vec<Array1<f64>>,
        treatments: Vec<Array1<f64>>,
        covariates: Vec<Array2<f64>>,
    ) -> EconResult<Self> {
        let t_count = outcomes.len();

        // Validate consistency
        if treatments.len() != t_count || covariates.len() != t_count {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "Number of time points must be consistent across outcomes ({}), \
                     treatments ({}), and covariates ({})",
                    outcomes.len(),
                    treatments.len(),
                    covariates.len()
                ),
            });
        }

        if t_count == 0 {
            return Err(EconError::EmptyDataset);
        }

        // Get sample size from first time point
        let n = outcomes[0].len();

        // Validate all arrays have same number of observations
        for (t, (y, (a, l))) in outcomes
            .iter()
            .zip(treatments.iter().zip(covariates.iter()))
            .enumerate()
        {
            if y.len() != n || a.len() != n || l.nrows() != n {
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "Time point {} has inconsistent sample sizes: \
                         outcomes={}, treatments={}, covariates={}",
                        t + 1,
                        y.len(),
                        a.len(),
                        l.nrows()
                    ),
                });
            }
        }

        // Missing and infinite treatment values are reported as such before the binary check.
        for (t, a) in treatments.iter().enumerate() {
            check_finite(a.iter(), &format!("treatment (time point {})", t + 1))?;
        }
        // Validate treatments are binary
        for (t, a) in treatments.iter().enumerate() {
            for val in a.iter() {
                if !((*val - 0.0).abs() < 1e-10 || (*val - 1.0).abs() < 1e-10) {
                    return Err(EconError::InvalidSpecification {
                        message: format!(
                            "Treatment at time {} must be binary (0 or 1), found value {}",
                            t + 1,
                            val
                        ),
                    });
                }
            }
        }
        let treatments: Vec<Array1<f64>> = treatments
            .into_iter()
            .map(|a| a.mapv(|v| if v > 0.5 { 1.0 } else { 0.0 }))
            .collect();

        let treatment_names: Vec<String> = (1..=t_count).map(|t| format!("A{t}")).collect();
        let covariate_names: Vec<Vec<String>> = covariates
            .iter()
            .enumerate()
            .map(|(t, l)| (1..=l.ncols()).map(|j| format!("L{}_{j}", t + 1)).collect())
            .collect();

        for (t, y) in outcomes.iter().enumerate() {
            check_finite(y.iter(), &format!("outcome (time point {})", t + 1))?;
        }
        for (t, l) in covariates.iter().enumerate() {
            for (j, col) in l.columns().into_iter().enumerate() {
                check_finite(
                    col.iter(),
                    &format!("{} (time point {})", covariate_names[t][j], t + 1),
                )?;
            }
        }

        Ok(Self {
            outcomes,
            treatments,
            covariates,
            treatment_names,
            covariate_names,
        })
    }

    /// Replace the names: one treatment name per time point and one covariate name per
    /// covariate column of each time point, all non-empty.
    pub fn with_names(
        mut self,
        treatment_names: Vec<String>,
        covariate_names: Vec<Vec<String>>,
    ) -> EconResult<Self> {
        let t_max = self.time_points();
        if treatment_names.len() != t_max || covariate_names.len() != t_max {
            return Err(spec_err(format!(
                "LTMLE: names for {} treatments and {} covariate sets given; the data have {t_max} \
                 time points",
                treatment_names.len(),
                covariate_names.len()
            )));
        }
        for (t, names) in covariate_names.iter().enumerate() {
            if names.len() != self.covariates[t].ncols() {
                return Err(spec_err(format!(
                    "LTMLE: {} covariate names given for time point {}, which has {} covariates",
                    names.len(),
                    t + 1,
                    self.covariates[t].ncols()
                )));
            }
        }
        if treatment_names
            .iter()
            .chain(covariate_names.iter().flatten())
            .any(|s| s.trim().is_empty())
        {
            return Err(spec_err("LTMLE: names must not be empty".into()));
        }
        self.treatment_names = treatment_names;
        self.covariate_names = covariate_names;
        Ok(self)
    }

    /// Columns of a dataset: `outcome` is the final outcome, `treatments[t]` is A_t and
    /// `covariates[t]` are observed before A_t. Missing values are refused
    /// (`NaPolicy::Fail`: there is no censoring node). Integer columns are accepted.
    pub fn from_dataset(
        dataset: &Dataset,
        outcome: &str,
        treatments: &[&str],
        covariates: &[&[&str]],
    ) -> EconResult<Self> {
        if treatments.len() != covariates.len() {
            return Err(spec_err(format!(
                "LTMLE: {} treatment columns and {} covariate sets given; one covariate set per \
                 time point",
                treatments.len(),
                covariates.len()
            )));
        }
        // The outcome is never a treatment or a covariate, and a treatment is not a covariate at
        // its own or a later time point (it would be conditioned on after it is set).
        if let Some(t) = treatments.iter().position(|a| *a == outcome) {
            return Err(spec_err(format!(
                "the outcome '{outcome}' is also the treatment of time point {}",
                t + 1
            )));
        }
        for (t, cs) in covariates.iter().enumerate() {
            for c in cs.iter() {
                if *c == outcome {
                    return Err(spec_err(format!(
                        "covariate '{c}' at time point {} is the outcome",
                        t + 1
                    )));
                }
                if let Some(s) = treatments[t..].iter().position(|a| a == c).map(|p| p + t) {
                    return Err(spec_err(format!(
                        "covariate '{c}' at time point {} is the treatment of time point {}; a \
                         treatment may only be a covariate of later time points",
                        t + 1,
                        s + 1
                    )));
                }
            }
        }
        let mut names: Vec<&str> = vec![outcome];
        names.extend_from_slice(treatments);
        for cs in covariates {
            names.extend_from_slice(cs);
        }
        names.sort_unstable();
        names.dedup();
        let cols = ColumnRequest::new(dataset.df())
            .numerics(&names)
            .na(NaPolicy::Fail)
            .extract()?;
        let n = cols.n();
        let t_max = treatments.len();
        let mut outcomes = vec![Array1::<f64>::zeros(n); t_max.saturating_sub(1)];
        outcomes.push(cols.numeric_array(outcome));
        let a = treatments.iter().map(|c| cols.numeric_array(c)).collect();
        let l = covariates
            .iter()
            .map(|cs| {
                if cs.is_empty() {
                    resource::try_zeros2(n, 0, "LTMLE covariates")
                } else {
                    cols.matrix(cs, false).map(|m| m.data)
                }
            })
            .collect::<EconResult<Vec<_>>>()?;
        LtmleData::new(outcomes, a, l)?.with_names(
            treatments.iter().map(|s| s.to_string()).collect(),
            covariates
                .iter()
                .map(|cs| cs.iter().map(|s| s.to_string()).collect())
                .collect(),
        )
    }

    /// Number of time points T.
    pub fn time_points(&self) -> usize {
        self.outcomes.len()
    }

    /// Number of observations n.
    pub fn n_obs(&self) -> usize {
        self.outcomes[0].len()
    }
}

/// Missing (NaN) values are `NullValues`, infinite ones `NonFiniteValues`.
fn check_finite<'a>(values: impl Iterator<Item = &'a f64>, column: &str) -> EconResult<()> {
    let (mut nan, mut inf) = (0usize, 0usize);
    for v in values {
        if v.is_nan() {
            nan += 1;
        } else if v.is_infinite() {
            inf += 1;
        }
    }
    if nan > 0 {
        return Err(EconError::NullValues {
            column: column.to_string(),
            count: nan,
        });
    }
    if inf > 0 {
        return Err(EconError::NonFiniteValues {
            column: column.to_string(),
            count: inf,
        });
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Result Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Estimate of one counterfactual mean.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RegimeEstimate {
    /// The regime.
    pub regime: Regime,
    /// Its label ([`Regime::label`]).
    pub label: String,
    /// E[Y^d] on the outcome scale.
    pub psi: f64,
    /// sd(EIC) / sqrt(n).
    pub se: f64,
    /// psi - z_{0.975} se.
    pub ci_lower: f64,
    /// psi + z_{0.975} se.
    pub ci_upper: f64,
    /// EIC per subject, outcome scale.
    pub influence_curve: Vec<f64>,
    /// `[t][i]` bounded cumulative probability of following the regime through node t
    /// (ltmle's `cum.g`).
    pub cumulative_g: Vec<Vec<f64>>,
    /// `[t][i]` H_t = I(follows through t) / cumulative_g.
    pub clever_covariates: Vec<Vec<f64>>,
    /// `[t][i]` Q*_t on the outcome scale.
    pub targeted_predictions: Vec<Vec<f64>>,
    /// Fluctuation coefficient eps_t per node: Q*_t = linkinv(eta + eps_t), and after a re-solve
    /// Q*_t = linkinv(clamp(eta, logit([lb, 1 - lb])) + eps_t), with `lb` the offset bound in
    /// [`LtmleSaturation::resolved_targeting`].
    pub fluctuation_coefs: Vec<f64>,
    /// Subjects with Abar_t = dbar_t.
    pub n_following: Vec<usize>,
    /// Subjects whose cumulative g was outside gbounds.
    pub n_bounded: Vec<usize>,
    /// ... among the followers (these change H_t).
    pub n_bounded_following: Vec<usize>,
    /// Q models in node order.
    pub q_models: Vec<NodeModel>,
}

/// Signs that an estimate rests on few regime followers or on saturated fits: ltmle's positivity
/// and rare-event checks, its no-variation rule for Q regressions and its score re-solve, and
/// targeted predictions or propensities at 0 or 1. Empty when none applies.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LtmleSaturation {
    /// Per estimated regime (intervention first): the share of all subjects whose unbounded
    /// cumulative probability of following it at the last node is below the lower g bound
    /// (ltmle warns when this share, pooled over the regimes, is above 0.01).
    pub share_below_gbound: Vec<f64>,
    /// Per regime: 0-based nodes whose Q response had a range below 1e-4 and was predicted as
    /// a constant (ltmle's no-variation rule).
    pub constant_q_nodes: Vec<Vec<usize>>,
    /// Per regime: 0-based nodes whose targeting step left `|sum_i H_i (Q*_{t+1,i} - Q*_{t,i})|`
    /// above 1e-3 and was re-solved, with the offset bound `lb` used (offsets bounded to
    /// `logit([lb, 1 - lb])`, as ltmle's `FixScoreEquation`).
    pub resolved_targeting: Vec<Vec<(usize, f64)>>,
    /// Per regime: the share of subjects whose Q*_1 is within 1e-8 of 0 or 1 (on the fitted
    /// [0, 1] scale of the logistic Q model; 0 with the linear Q model, whose Q* has no bounds).
    pub share_q_star_at_bound: Vec<f64>,
    /// 0-based nodes with a fitted propensity within `10 * f64::EPSILON` of 0 or 1 (glm.fit's
    /// "numerically 0 or 1" threshold).
    pub g_at_bound_nodes: Vec<usize>,
    /// A 0/1 outcome with a mean below 0.03 (ltmle's rare-events check).
    pub rare_outcome: bool,
}

impl LtmleSaturation {
    /// ltmle's positivity check: the share of subjects below the lower g bound at the last node,
    /// pooled over the estimated regimes, is above 0.01.
    pub fn positivity(&self) -> bool {
        !self.share_below_gbound.is_empty()
            && self.share_below_gbound.iter().sum::<f64>() / self.share_below_gbound.len() as f64
                > POSITIVITY_SHARE
    }

    /// Whether any check fired: positivity, a constant Q node, a re-solved targeting step, Q* at
    /// 0 or 1 for more than 1% of the subjects, a propensity at 0 or 1, a rare outcome.
    pub fn flagged(&self) -> bool {
        self.positivity()
            || self.constant_q_nodes.iter().any(|v| !v.is_empty())
            || self.resolved_targeting.iter().any(|v| !v.is_empty())
            || self
                .share_q_star_at_bound
                .iter()
                .any(|s| *s > POSITIVITY_SHARE)
            || !self.g_at_bound_nodes.is_empty()
            || self.rare_outcome
    }
}

/// Result from LTMLE estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LtmleResult {
    /// Contrast psi(intervention) - psi(contrast); NaN without a contrast.
    pub ate: f64,
    /// Standard error of the contrast: sd of the differenced EICs / sqrt(n)
    pub se: f64,
    /// 95% confidence interval lower bound
    pub ci_lower: f64,
    /// 95% confidence interval upper bound
    pub ci_upper: f64,
    /// Two-sided p-value for H0: contrast = 0
    pub p_value: f64,
    /// Significance level
    pub significance: SignificanceLevel,
    /// Z-statistic (contrast / SE)
    pub z_stat: f64,

    /// E[Y^intervention] (= `regime_estimate.psi`)
    pub psi_treated: f64,
    /// Standard error of psi_treated
    pub psi_treated_se: f64,
    /// E[Y^contrast]; NaN without a contrast
    pub psi_control: f64,
    /// Standard error of psi_control
    pub psi_control_se: f64,

    /// EIC of the contrast (of E[Y^intervention] without one), one value per observation
    pub influence_curve: Vec<f64>,

    /// Fitted P(A_t = 1 | observed history) at each time point (unbounded)
    pub propensity_scores: Vec<Vec<f64>>,
    /// The intervention's clever covariates H_t
    pub clever_covariates: Vec<Vec<f64>>,
    /// The intervention's targeted predictions Q*_t on the outcome scale
    pub targeted_predictions: Vec<Vec<f64>>,
    /// The intervention's fluctuation coefficients eps_t (after a re-solve, on the bounded
    /// offsets, as [`RegimeEstimate::fluctuation_coefs`])
    pub fluctuation_coefs: Vec<f64>,

    /// Number of observations
    pub n_obs: usize,
    /// Number of time points
    pub time_points: usize,
    /// Number of treated at each time point
    pub n_treated_by_time: Vec<usize>,
    /// Followers with bounded cumulative g, summed over the estimated regimes
    pub n_truncated_by_time: Vec<usize>,

    /// Outcome model type used
    pub q_model_type: LtmleQModel,
    /// Bounds on the cumulative probability of following the regime
    pub gbounds: (f64, f64),

    /// Whether every iterative fit converged
    pub converged: bool,
    /// Warnings generated during estimation
    pub warnings: Vec<String>,

    /// The intervention's estimate.
    #[serde(default)]
    pub regime_estimate: RegimeEstimate,
    /// The reference regime's estimate.
    #[serde(default)]
    pub contrast_estimate: Option<RegimeEstimate>,
    /// "intervention label - contrast label".
    #[serde(default)]
    pub contrast_label: Option<String>,
    /// The g models in node order.
    #[serde(default)]
    pub g_models: Vec<NodeModel>,
    /// (min, max) of the outcome when it was rescaled to [0, 1].
    #[serde(default)]
    pub outcome_range: Option<(f64, f64)>,
    /// Iterative fits (g, logistic Q, logistic fluctuations).
    #[serde(default)]
    pub n_model_fits: usize,
    /// Few followers and saturated fits ([`LtmleSaturation`]); the checks that fire also add
    /// warnings and a "Saturation:" block to the Display.
    #[serde(default)]
    pub saturation: LtmleSaturation,
}

impl ResultNotes for LtmleResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

impl fmt::Display for LtmleResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Longitudinal Targeted Maximum Likelihood Estimation (LTMLE)"
        )?;
        writeln!(
            f,
            "============================================================"
        )?;
        writeln!(f)?;
        writeln!(f, "Time points: {}", self.time_points)?;
        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(
            f,
            "Outcome model (Q): {}, on the full history",
            self.q_model_type
        )?;
        writeln!(f, "Treatment model (g): logistic, on the full history")?;
        writeln!(
            f,
            "Cumulative g bounds: [{:.4}, {:.4}]",
            self.gbounds.0, self.gbounds.1
        )?;
        writeln!(f, "Standard errors: {VARIANCE_METHOD}")?;
        writeln!(f)?;
        let estimates: Vec<&RegimeEstimate> = std::iter::once(&self.regime_estimate)
            .chain(self.contrast_estimate.iter())
            .collect();
        writeln!(f, "Counterfactual means:")?;
        for e in &estimates {
            writeln!(
                f,
                "  E[Y^{{{}}}]: {:>10.4} (SE {:.4}; 95% CI [{:.4}, {:.4}])",
                e.label, e.psi, e.se, e.ci_lower, e.ci_upper
            )?;
        }
        writeln!(f)?;
        if let Some(label) = &self.contrast_label {
            writeln!(f, "Contrast: {label}")?;
            writeln!(f, "  Estimate:   {:>12.4}", self.ate)?;
            writeln!(f, "  Std. Error: {:>12.4}", self.se)?;
            writeln!(f, "  z-stat:     {:>12.2}", self.z_stat)?;
            writeln!(
                f,
                "  p-value:    {:>12.4}{}",
                self.p_value,
                self.significance.stars()
            )?;
            writeln!(
                f,
                "  95% CI:     [{:.4}, {:.4}]",
                self.ci_lower, self.ci_upper
            )?;
            writeln!(f)?;
        }
        writeln!(
            f,
            "Regime followers by time (bounded cumulative g among them):"
        )?;
        for t in 0..self.time_points {
            let parts: Vec<String> = estimates
                .iter()
                .map(|e| {
                    format!(
                        "{} {} ({})",
                        e.label,
                        e.n_following.get(t).copied().unwrap_or(0),
                        e.n_bounded_following.get(t).copied().unwrap_or(0)
                    )
                })
                .collect();
            let a_name = self
                .regime_estimate
                .q_models
                .get(t)
                .and_then(|m| m.names.last().cloned())
                .unwrap_or_else(|| format!("A{}", t + 1));
            writeln!(
                f,
                "  Time {} ({a_name}): {}; treated {}",
                t + 1,
                parts.join(", "),
                self.n_treated_by_time.get(t).copied().unwrap_or(0)
            )?;
        }
        writeln!(f)?;
        writeln!(f, "Targeting step (epsilon by time):")?;
        for e in &estimates {
            let eps: Vec<String> = e
                .fluctuation_coefs
                .iter()
                .map(|v| format!("{v:.6}"))
                .collect();
            writeln!(f, "  {}: {}", e.label, eps.join(", "))?;
        }
        writeln!(f)?;
        let failed = self
            .g_models
            .iter()
            .chain(estimates.iter().flat_map(|e| e.q_models.iter()))
            .filter(|m| !m.converged)
            .count();
        if self.converged {
            writeln!(
                f,
                "Model fits: {} iterative fits, all converged",
                self.n_model_fits
            )?;
        } else if failed > 0 {
            writeln!(
                f,
                "Model fits: {} iterative fits, {failed} did not converge (see warnings)",
                self.n_model_fits
            )?;
        } else {
            writeln!(
                f,
                "Model fits: {} iterative fits, some did not converge (see warnings)",
                self.n_model_fits
            )?;
        }
        writeln!(f)?;
        if self.saturation.flagged() {
            self.fmt_saturation(f, &estimates)?;
            writeln!(f)?;
        }
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1")?;
        write!(f, "{}", notes_text(self))
    }
}

impl LtmleResult {
    /// The Display's "Saturation:" block, one line per regime.
    fn fmt_saturation(
        &self,
        f: &mut fmt::Formatter<'_>,
        estimates: &[&RegimeEstimate],
    ) -> fmt::Result {
        let s = &self.saturation;
        let a_name = |t: usize| -> String {
            self.regime_estimate
                .q_models
                .get(t)
                .and_then(|m| m.names.last().cloned())
                .unwrap_or_else(|| format!("A{}", t + 1))
        };
        let names = |nodes: &mut dyn Iterator<Item = usize>| -> String {
            nodes.map(a_name).collect::<Vec<_>>().join(", ")
        };
        writeln!(f, "Saturation (see the warnings):")?;
        for (k, e) in estimates.iter().enumerate() {
            let mut parts = vec![format!(
                "{} followers at the last node",
                e.n_following.last().copied().unwrap_or(0)
            )];
            if let Some(p) = s.share_below_gbound.get(k) {
                parts.push(format!("{:.1}% of subjects below the g bound", 100.0 * p));
            }
            if let Some(c) = s.constant_q_nodes.get(k).filter(|c| !c.is_empty()) {
                parts.push(format!("constant Q at {}", names(&mut c.iter().copied())));
            }
            if let Some(r) = s.resolved_targeting.get(k).filter(|r| !r.is_empty()) {
                parts.push(format!(
                    "targeting re-solved at {}",
                    names(&mut r.iter().map(|(t, _)| *t))
                ));
            }
            if let Some(q) = s.share_q_star_at_bound.get(k).filter(|q| **q > 0.0) {
                parts.push(format!("Q* at 0 or 1 for {:.1}%", 100.0 * q));
            }
            writeln!(f, "  {}: {}", e.label, parts.join("; "))?;
        }
        if !s.g_at_bound_nodes.is_empty() {
            writeln!(
                f,
                "  propensities at 0 or 1 at {}",
                names(&mut s.g_at_bound_nodes.iter().copied())
            )?;
        }
        if s.rare_outcome {
            writeln!(f, "  rare outcome (mean below {RARE_OUTCOME_MEAN})")?;
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main LTMLE Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Run Longitudinal TMLE with full configuration.
///
/// Estimates `E[Y^intervention]` and, when `config.contrast` is set, `E[Y^contrast]` and
/// their difference, with the algorithm of the module documentation (ltmle 1.x with its
/// defaults and `variance.method = "ic"`).
///
/// # Errors
///
/// - `InvalidSpecification`: `tolerance` not finite and > 0, `max_iter` 0, `gbounds` not
///   `0 < lower < upper <= 1`, fewer than 2 time points, an invalid regime (lengths, values
///   other than 0/1, not one rule per node, an unknown or later covariate, a non-finite
///   threshold), or a constant outcome with the logistic Q model.
/// - `InsufficientData`: no more observations than coefficients of the last outcome
///   regression, or no subject following a regime through some time point (positivity).
///
/// # References
///
/// - van der Laan & Gruber (2012), Int J Biostat 8(1), Article 9
/// - Lendle et al. (2017), JSS 81(1)
pub fn run_ltmle(data: &LtmleData, config: LtmleConfig) -> EconResult<LtmleResult> {
    run_ltmle_with(
        data,
        config,
        FitPaths {
            lean: true,
            warm: true,
        },
    )
}

/// How the iterative nuisance fits run. With either switch on the fits reach the same maximum-
/// likelihood estimates to the convergence rule, with iteration counts and rule-slack differences
/// as described in validation/econometrics/ltmle.md, "Nuisance fits". Tests turn them off to
/// compare with the shared GLM from glm.fit's start.
#[derive(Debug, Clone, Copy)]
struct FitPaths {
    /// Through [`lean_logit_refined`], falling back to [`fit_glm`] whenever a fit leaves its
    /// path (separation, a rank-deficient or ill-conditioned design, no convergence, a converged
    /// fit with a row beyond the bound `LEAN_MISFIT_ETA`), so that the shared GLM reports those cases.
    lean: bool,
    /// Each earlier Q regression starts from the later node's coefficients ([`warm_start`]),
    /// when `max_iter` is at least [`WARM_MIN_MAX_ITER`].
    warm: bool,
}

fn run_ltmle_with(
    data: &LtmleData,
    config: LtmleConfig,
    paths: FitPaths,
) -> EconResult<LtmleResult> {
    validate_config(&config)?;
    let n = data.n_obs();
    let t_max = data.time_points();

    if t_max < 2 {
        return Err(EconError::InvalidSpecification {
            message:
                "LTMLE requires at least 2 time points. For single time point, use standard TMLE."
                    .to_string(),
        });
    }
    let mut p_max = 1usize;
    for l in &data.covariates {
        p_max = checked_add(p_max, l.ncols())?;
    }
    p_max = checked_add(p_max, t_max)?;
    if n <= p_max {
        return Err(EconError::InsufficientData {
            required: checked_add(p_max, 1)?,
            provided: n,
            context: format!(
                "LTMLE: the outcome regression at the last time point has {p_max} coefficients"
            ),
        });
    }

    let mut notes = FitNotes {
        warnings: Vec::new(),
        n_fits: 0,
        all_converged: true,
    };
    if data.outcomes[..t_max - 1]
        .iter()
        .any(|y| y.iter().any(|v| *v != 0.0))
    {
        notes.warnings.push(format!(
            "LTMLE: outcomes at time points 1..{} are not used; p2a's LTMLE has a single \
             outcome node (time point {t_max}), no intermediate or survival outcomes",
            t_max - 1
        ));
    }

    let mut regimes: Vec<(Regime, Vec<Vec<u8>>, String)> = Vec::with_capacity(2);
    for r in std::iter::once(&config.intervention).chain(config.contrast.iter()) {
        let d = regime_matrix(data, r)?;
        regimes.push((r.clone(), d, r.label(&data.treatment_names)));
    }
    let follow: Vec<Vec<Vec<bool>>> = regimes
        .iter()
        .map(|(_, d, label)| followers(data, d, label))
        .collect::<EconResult<_>>()?;

    let y = &data.outcomes[t_max - 1];
    let scale = outcome_scale(y, config.q_model)?;
    if scale.rescaled {
        notes.warnings.push(format!(
            "LTMLE: outcome rescaled from [{:.4}, {:.4}] to [0, 1] for the logistic Q model \
             (ltmle's Yrange); estimates are on the original scale",
            scale.lower, scale.upper
        ));
    }
    let y_scaled = y.mapv(|v| (v - scale.lower) / scale.width);

    let designs = HistoryDesigns::new(data)?;
    let g = fit_treatment_models(data, &designs, &config, paths, &mut notes)?;
    // The last node's Q regression has the same response and design for every regime.
    let last_q = fit_q(
        &designs,
        t_max - 1,
        y_scaled.view(),
        &config,
        &data.treatment_names,
        None,
        paths,
        &mut notes,
    )?;

    let mut estimates = Vec::with_capacity(regimes.len());
    let mut parts = Vec::with_capacity(regimes.len());
    for ((regime, d, label), follow) in regimes.into_iter().zip(&follow) {
        let (e, part) = estimate_regime(
            data, regime, &d, label, follow, &designs, &g, &last_q, &y_scaled, &scale, &config,
            paths, &mut notes,
        )?;
        estimates.push(e);
        parts.push(part);
    }
    let saturation = saturation_checks(
        y,
        &g,
        &estimates,
        parts,
        config.gbounds.0,
        &data.treatment_names,
        &mut notes,
    );

    let n_treated_by_time = data
        .treatments
        .iter()
        .map(|a| a.iter().filter(|v| **v > 0.5).count())
        .collect();
    let n_truncated_by_time = (0..t_max)
        .map(|t| estimates.iter().map(|r| r.n_bounded_following[t]).sum())
        .collect();
    let contrast_estimate = if estimates.len() > 1 {
        estimates.pop()
    } else {
        None
    };
    let regime_estimate = estimates.pop().ok_or_else(|| {
        EconError::Computation("LTMLE: no estimate for the intervention regime".into())
    })?;

    let (contrast, contrast_label) = match &contrast_estimate {
        Some(b) => (
            contrast_inference(&regime_estimate, b, &mut notes)?,
            Some(format!("{} - {}", regime_estimate.label, b.label)),
        ),
        None => (
            Contrast {
                estimate: f64::NAN,
                se: f64::NAN,
                z: f64::NAN,
                p_value: f64::NAN,
                ci_lower: f64::NAN,
                ci_upper: f64::NAN,
                influence_curve: regime_estimate.influence_curve.clone(),
            },
            None,
        ),
    };

    Ok(LtmleResult {
        ate: contrast.estimate,
        se: contrast.se,
        ci_lower: contrast.ci_lower,
        ci_upper: contrast.ci_upper,
        p_value: contrast.p_value,
        significance: SignificanceLevel::from_p_value(contrast.p_value),
        z_stat: contrast.z,
        psi_treated: regime_estimate.psi,
        psi_treated_se: regime_estimate.se,
        psi_control: contrast_estimate.as_ref().map_or(f64::NAN, |b| b.psi),
        psi_control_se: contrast_estimate.as_ref().map_or(f64::NAN, |b| b.se),
        influence_curve: contrast.influence_curve,
        propensity_scores: g.fitted.iter().map(|p| p.to_vec()).collect(),
        clever_covariates: regime_estimate.clever_covariates.clone(),
        targeted_predictions: regime_estimate.targeted_predictions.clone(),
        fluctuation_coefs: regime_estimate.fluctuation_coefs.clone(),
        n_obs: n,
        time_points: t_max,
        n_treated_by_time,
        n_truncated_by_time,
        q_model_type: config.q_model,
        gbounds: config.gbounds,
        converged: notes.all_converged,
        warnings: notes.warnings,
        regime_estimate,
        contrast_estimate,
        contrast_label,
        g_models: g.models,
        outcome_range: scale.rescaled.then_some((scale.lower, scale.upper)),
        n_model_fits: notes.n_fits,
        saturation,
    })
}

/// One regime's part of [`LtmleSaturation`].
struct RegimeSaturation {
    share_below_gbound: f64,
    constant_q_nodes: Vec<usize>,
    resolved_targeting: Vec<(usize, f64)>,
    share_q_star_at_bound: f64,
}

/// ltmle's positivity and rare-event checks and the rest of the saturation record, with their
/// warnings.
fn saturation_checks(
    y: &Array1<f64>,
    g: &TreatmentModels,
    estimates: &[RegimeEstimate],
    parts: Vec<RegimeSaturation>,
    g_lower: f64,
    treatment_names: &[String],
    notes: &mut FitNotes,
) -> LtmleSaturation {
    let mut s = LtmleSaturation::default();
    for p in parts {
        s.share_below_gbound.push(p.share_below_gbound);
        s.constant_q_nodes.push(p.constant_q_nodes);
        s.resolved_targeting.push(p.resolved_targeting);
        s.share_q_star_at_bound.push(p.share_q_star_at_bound);
    }
    s.g_at_bound_nodes = g
        .fitted
        .iter()
        .enumerate()
        .filter(|(_, p)| p.iter().any(|v| *v < G_AT_BOUND || *v > 1.0 - G_AT_BOUND))
        .map(|(t, _)| t)
        .collect();
    let mean = y.sum() / y.len() as f64;
    s.rare_outcome = y.iter().all(|v| *v == 0.0 || *v == 1.0) && mean < RARE_OUTCOME_MEAN;
    if s.positivity() {
        for (e, p) in estimates.iter().zip(&s.share_below_gbound) {
            if *p > 0.0 {
                notes.warnings.push(format!(
                    "LTMLE: positivity: {:.1}% of subjects have a cumulative probability of \
                     following {} below {g_lower} at the last node; the influence-curve SE may be \
                     anticonservative (ltmle warns the same)",
                    100.0 * p,
                    e.label
                ));
            }
        }
    }
    if s.rare_outcome {
        notes.warnings.push(format!(
            "LTMLE: rare outcome (mean {mean:.4} < {RARE_OUTCOME_MEAN}); the influence-curve SE \
             may be anticonservative (ltmle warns the same)"
        ));
    }
    for (e, q) in estimates.iter().zip(&s.share_q_star_at_bound) {
        if *q > POSITIVITY_SHARE {
            notes.warnings.push(format!(
                "LTMLE: {}: Q* is within {Q_STAR_AT_BOUND:e} of 0 or 1 for {:.1}% of subjects; \
                 the estimate rests on saturated fits",
                e.label,
                100.0 * q
            ));
        }
    }
    if !s.g_at_bound_nodes.is_empty() {
        let nodes: Vec<String> = s
            .g_at_bound_nodes
            .iter()
            .map(|t| treatment_names[*t].clone())
            .collect();
        notes.warnings.push(format!(
            "LTMLE: fitted propensities numerically 0 or 1 at {}",
            nodes.join(", ")
        ));
    }
    s
}

/// Run LTMLE with default configuration.
///
/// Estimates E[Y^{always treat}] - E[Y^{never treat}] with the logistic Q model,
/// cumulative g bounds (0.01, 1) and influence-curve standard errors.
pub fn ltmle(data: &LtmleData) -> EconResult<LtmleResult> {
    run_ltmle(data, LtmleConfig::default())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Estimation steps
// ═══════════════════════════════════════════════════════════════════════════════

fn spec_err(message: String) -> EconError {
    EconError::InvalidSpecification { message }
}

fn checked_add(a: usize, b: usize) -> EconResult<usize> {
    a.checked_add(b)
        .ok_or_else(|| EconError::ResourceLimit("LTMLE: design size overflows usize".into()))
}

fn validate_config(config: &LtmleConfig) -> EconResult<()> {
    if !(config.tolerance.is_finite() && config.tolerance > 0.0) {
        return Err(spec_err(format!(
            "LTMLE: tolerance must be finite and > 0, got {}",
            config.tolerance
        )));
    }
    if config.max_iter == 0 {
        return Err(spec_err("LTMLE: max_iter must be at least 1, got 0".into()));
    }
    let (lo, hi) = config.gbounds;
    if !(lo.is_finite() && hi.is_finite() && lo > 0.0 && lo < hi && hi <= 1.0) {
        return Err(spec_err(format!(
            "LTMLE: gbounds must satisfy 0 < lower < upper <= 1, got ({lo}, {hi})"
        )));
    }
    Ok(())
}

/// `d[t][i]` in {0, 1}: the treatment the regime assigns to subject i at node t.
fn regime_matrix(data: &LtmleData, regime: &Regime) -> EconResult<Vec<Vec<u8>>> {
    let n = data.n_obs();
    let t_max = data.time_points();
    match regime {
        Regime::Static(v) => {
            if v.len() != 1 && v.len() != t_max {
                return Err(spec_err(format!(
                    "LTMLE: a static regime needs 1 or {t_max} values (one per time point), got {}",
                    v.len()
                )));
            }
            if let Some((t, x)) = v.iter().enumerate().find(|(_, x)| **x > 1) {
                return Err(spec_err(format!(
                    "LTMLE: static regime values must be 0 or 1, got {x} at time point {}",
                    t + 1
                )));
            }
            Ok((0..t_max)
                .map(|t| vec![if v.len() == 1 { v[0] } else { v[t] }; n])
                .collect())
        }
        Regime::Dynamic(rules) => {
            if let Some(r) = rules.iter().find(|r| r.node >= t_max) {
                return Err(spec_err(format!(
                    "LTMLE: dynamic rule for node {} is outside time points 1..={t_max}",
                    r.node
                )));
            }
            for t in 0..t_max {
                let k = rules.iter().filter(|r| r.node == t).count();
                if k != 1 {
                    return Err(spec_err(format!(
                        "LTMLE: dynamic regime needs exactly one rule per time point; time point {} \
                         has {k}",
                        t + 1
                    )));
                }
            }
            let mut d = vec![Vec::new(); t_max];
            for r in rules {
                let a_name = &data.treatment_names[r.node];
                if !r.threshold.is_finite() {
                    return Err(spec_err(format!(
                        "LTMLE: dynamic rule for {a_name}: threshold must be finite, got {}",
                        r.threshold
                    )));
                }
                if r.above > 1 {
                    return Err(spec_err(format!(
                        "LTMLE: dynamic rule for {a_name}: above must be 0 or 1, got {}",
                        r.above
                    )));
                }
                let found = (0..=r.node).rev().find_map(|s| {
                    data.covariate_names[s]
                        .iter()
                        .position(|c| *c == r.column)
                        .map(|j| (s, j))
                });
                let Some((s, j)) = found else {
                    let available: Vec<&str> = data.covariate_names[..=r.node]
                        .iter()
                        .flatten()
                        .map(String::as_str)
                        .collect();
                    return Err(spec_err(format!(
                        "LTMLE: dynamic rule for {a_name}: column '{}' is not a covariate observed \
                         at or before time point {}; available: {}",
                        r.column,
                        r.node + 1,
                        available.join(", ")
                    )));
                };
                d[r.node] = data.covariates[s]
                    .column(j)
                    .iter()
                    .map(|x| {
                        if *x > r.threshold {
                            r.above
                        } else {
                            1 - r.above
                        }
                    })
                    .collect();
            }
            Ok(d)
        }
    }
}

/// `follow[t][i]`: subject i follows the regime `d` through node t. Errors when nobody does.
fn followers(data: &LtmleData, d: &[Vec<u8>], label: &str) -> EconResult<Vec<Vec<bool>>> {
    let n = data.n_obs();
    let mut out: Vec<Vec<bool>> = Vec::with_capacity(d.len());
    for (t, dt) in d.iter().enumerate() {
        let a = &data.treatments[t];
        let f: Vec<bool> = (0..n)
            .map(|i| (t == 0 || out[t - 1][i]) && (a[i] > 0.5) == (dt[i] == 1))
            .collect();
        if !f.iter().any(|v| *v) {
            return Err(EconError::InsufficientData {
                required: 1,
                provided: 0,
                context: format!(
                    "LTMLE: no subject follows regime '{label}' through {}; its \
                     counterfactual mean is not identified (positivity)",
                    data.treatment_names[t]
                ),
            });
        }
        out.push(f);
    }
    Ok(out)
}

/// The observed-history designs, built once and shared by every regime: the Q design of node
/// t has the columns "(Intercept)", L_1.., A_1, ..., L_t.., A_t (R's `reformulate` order), so
/// each design is a column prefix of the next, and one n x p matrix (the last node's design)
/// holds them all; the g design of node t is its Q design without the last column.
/// `a_cols[t][s]` is the column of A_s (s <= t), and `l_means[t]` the means of the covariates
/// L_t.
struct HistoryDesigns {
    x: Array2<f64>,
    /// `widths[t]`: the number of columns of node t's Q design.
    widths: Vec<usize>,
    names: Vec<String>,
    a_cols: Vec<Vec<usize>>,
    l_means: Vec<Vec<f64>>,
}

impl HistoryDesigns {
    fn new(data: &LtmleData) -> EconResult<Self> {
        let n = data.n_obs();
        let t_max = data.time_points();
        let mut widths: Vec<usize> = Vec::with_capacity(t_max);
        let mut p = 1usize;
        for l in &data.covariates {
            p = checked_add(p, checked_add(l.ncols(), 1)?)?;
            widths.push(p);
        }
        let mut x = resource::try_zeros2(n, p, "LTMLE history design")?;
        x.column_mut(0).fill(1.0);
        let mut names = vec!["(Intercept)".to_string()];
        let mut a_cols: Vec<Vec<usize>> = Vec::with_capacity(t_max);
        let mut l_means = Vec::with_capacity(t_max);
        let mut col = 1usize;
        for node in 0..t_max {
            resource::checkpoint()?;
            let l = &data.covariates[node];
            for (j, c) in l.columns().into_iter().enumerate() {
                x.column_mut(col + j).assign(&c);
            }
            col += l.ncols();
            x.column_mut(col).assign(&data.treatments[node]);
            names.extend(data.covariate_names[node].iter().cloned());
            names.push(data.treatment_names[node].clone());
            let mut ac = a_cols.last().cloned().unwrap_or_default();
            ac.push(col);
            col += 1;
            l_means.push(
                l.columns()
                    .into_iter()
                    .map(|c| c.sum() / n as f64)
                    .collect::<Vec<f64>>(),
            );
            a_cols.push(ac);
        }
        Ok(Self {
            x,
            widths,
            names,
            a_cols,
            l_means,
        })
    }

    /// The Q design of node t and its names.
    fn q(&self, t: usize) -> (ArrayView2<'_, f64>, &[String]) {
        let p = self.widths[t];
        (self.x.slice(ndarray::s![.., ..p]), &self.names[..p])
    }

    /// The g design of node t (the Q design without A_t) and its names.
    fn g(&self, t: usize) -> (ArrayView2<'_, f64>, &[String]) {
        let p = self.widths[t] - 1;
        (self.x.slice(ndarray::s![.., ..p]), &self.names[..p])
    }
}

/// `X beta` over the identified columns (a NaN coefficient drops its column, as
/// `predict.lm` / `predict.glm` do), with every A_s column of `a_cols` replaced by the
/// regime's `d[s][i]` (ltmle's `SetA`).
fn identified_linear_predictor(
    x: ArrayView2<f64>,
    coefficients: ArrayView1<f64>,
    a_cols: &[usize],
    d: Option<&[Vec<u8>]>,
) -> Array1<f64> {
    // Identified coefficients off the A_s columns, row by row in the design's memory order;
    // each A_s column's coefficient applied to the regime's values afterwards.
    let mut fixed = coefficients.mapv(|b| if b.is_nan() { 0.0 } else { b });
    let mut set: Vec<(usize, f64)> = Vec::new();
    if d.is_some() {
        for (s, &c) in a_cols.iter().enumerate() {
            if c < fixed.len() {
                set.push((s, fixed[c]));
                fixed[c] = 0.0;
            }
        }
    }
    let mut eta = Array1::<f64>::zeros(x.nrows());
    for (e, row) in eta.iter_mut().zip(x.rows()) {
        *e = row.dot(&fixed);
    }
    if let Some(d) = d {
        for (s, b) in set {
            for (e, &v) in eta.iter_mut().zip(&d[s]) {
                *e += f64::from(v) * b;
            }
        }
    }
    eta
}

/// The outcome's scale: fits use `(y - lower) / width`.
struct OutcomeScale {
    lower: f64,
    upper: f64,
    width: f64,
    rescaled: bool,
}

fn outcome_scale(y: &Array1<f64>, q_model: LtmleQModel) -> EconResult<OutcomeScale> {
    let (min, max) = y
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    // Also refuses a NaN range (no ordering).
    if q_model == LtmleQModel::Logistic
        && max.partial_cmp(&min) != Some(std::cmp::Ordering::Greater)
    {
        return Err(spec_err(format!(
            "LTMLE: the outcome is constant ({min}); nothing to estimate"
        )));
    }
    if q_model == LtmleQModel::Logistic && (min < 0.0 || max > 1.0) {
        let width = max - min;
        return Ok(OutcomeScale {
            lower: min,
            upper: max,
            width,
            rescaled: true,
        });
    }
    Ok(OutcomeScale {
        lower: 0.0,
        upper: 1.0,
        width: 1.0,
        rescaled: false,
    })
}

/// Warnings, iterative-fit count and convergence of every model of one LTMLE call.
struct FitNotes {
    warnings: Vec<String>,
    n_fits: usize,
    all_converged: bool,
}

impl FitNotes {
    fn push_glm(&mut self, what: &str, fit: &crate::econometrics::glm::GlmFit, names: &[String]) {
        self.n_fits += 1;
        self.all_converged &= fit.converged;
        for w in &fit.warnings {
            self.warnings.push(format!("{what}: {w}"));
        }
        if !fit.converged && !fit.warnings.iter().any(|w| w.contains("converge")) {
            self.warnings.push(format!(
                "{what}: did not converge in {} iterations",
                fit.iterations
            ));
        }
        if let Some(m) = aliasing_message(&fit.aliased, names) {
            self.warnings.push(format!(
                "{what}: {m}; predictions use the identified columns"
            ));
        }
    }
}

/// One fitted nuisance regression.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeModel {
    /// 0-based time point.
    pub node: usize,
    /// "(Intercept)", then the history in time order.
    pub names: Vec<String>,
    /// Coefficients; NaN for aliased columns.
    pub coefficients: Vec<f64>,
    /// The fit converged (true for least-squares fits).
    pub converged: bool,
    /// IRLS iterations (0 for least-squares fits). A logistic Q node before the last normally
    /// starts from the later node's coefficients, so its count is not comparable with
    /// glm.fit's from glm.fit's own start.
    pub iterations: usize,
    /// Set when the Q response had a range below 1e-4 and was not regressed (ltmle's
    /// no-variation rule): the constant prediction on the fitted [0, 1] scale, the response's
    /// maximum. The intercept is then its logit, bounded to `[1e-4, 0.9999]` first as ltmle does,
    /// the other coefficients are NaN and `iterations` is 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constant_prediction: Option<f64>,
}

/// The g models: coefficients, records and fitted (observed-history) propensities.
struct TreatmentModels {
    coefficients: Vec<Array1<f64>>,
    models: Vec<NodeModel>,
    fitted: Vec<Array1<f64>>,
}

fn glm_spec<'a>(family: GlmFamily, config: &LtmleConfig) -> GlmSpec<'a> {
    let mut spec = GlmSpec::new(family, GlmLink::Logit);
    spec.epsilon = config.tolerance;
    spec.max_iter = config.max_iter;
    spec
}

/// A response whose mean is this close to 0 or 1 goes straight to the shared GLM.
const SATURATED_MEAN: f64 = 1e-6;

/// ltmle's no-variation rule (`ltmle:::Estimate`): a Q response whose range over the fitted
/// subjects is below this is not regressed but predicted as the constant `max(response)`, on the
/// link scale `qlogis` of it bounded to [`CONSTANT_Q_BOUNDS`].
const CONSTANT_Q_RANGE: f64 = 1e-4;
/// The bounds ltmle applies to a constant Q prediction before taking its logit.
const CONSTANT_Q_BOUNDS: (f64, f64) = (1e-4, 0.9999);

/// Below this `max_iter` the Q regressions keep glm.fit's start. The limit decides whether a fit
/// is reported as not converged, on the step count from that start, which a warm start would
/// undercut. Fits that can take the warm path (a response mean off [`SATURATED_MEAN`]) need well
/// under this many steps from glm.fit's start: Fisher scoring gains about one unit of logit per
/// step, and the most measured at the default tolerance, counting all 11,292 such fits of a
/// sweep, is 42 of these 50 (validation/econometrics/ltmle.md, "Nuisance fits").
const WARM_MIN_MAX_ITER: usize = 50;

/// One logistic nuisance fit: the coefficients (NaN for aliased columns), the fitted values
/// and the convergence record.
struct LogitNode {
    coefficients: Array1<f64>,
    fitted: Array1<f64>,
    converged: bool,
    iterations: usize,
}

/// `glm(y ~ x - 1, family)` with logit link: the lean IRLS when the fit stays on its path, from
/// `start` or glm.fit's start; otherwise the shared GLM from glm.fit's start, with its warnings
/// and rank handling.
#[allow(clippy::too_many_arguments)]
fn fit_logit_node(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    family: GlmFamily,
    start: Option<ArrayView1<f64>>,
    config: &LtmleConfig,
    what: &str,
    names: &[String],
    paths: FitPaths,
    notes: &mut FitNotes,
) -> EconResult<LogitNode> {
    // The binomial family warns about a response that is not 0/1; the lean path does not. A
    // response saturated at 0 or 1 (an extreme fluctuation at the later node) drives the linear
    // predictor to the link's clamp, where the shared GLM reports on the fit.
    let mean = y.sum() / y.len() as f64;
    let saturated = !(SATURATED_MEAN..=1.0 - SATURATED_MEAN).contains(&mean);
    let lean_ok = paths.lean
        && !saturated
        && (family != GlmFamily::Binomial || y.iter().all(|v| *v == 0.0 || *v == 1.0));
    if lean_ok {
        let ctl = GlmControl {
            epsilon: config.tolerance,
            max_iter: config.max_iter,
        };
        let from = start.map_or(LeanStart::Mustart, LeanStart::Coef);
        if let Some(fit) = lean_logit_refined(x, y, None, from, ctl)? {
            notes.n_fits += 1;
            return Ok(LogitNode {
                coefficients: fit.coefficients,
                fitted: Array1::from_vec(fit.fitted),
                converged: true,
                iterations: fit.iterations,
            });
        }
    }
    let fit = fit_glm(x, y, &glm_spec(family, config))?;
    notes.push_glm(what, &fit, names);
    Ok(LogitNode {
        coefficients: fit.coefficients,
        fitted: fit.fitted,
        converged: fit.converged,
        iterations: fit.iterations,
    })
}

fn fit_treatment_models(
    data: &LtmleData,
    designs: &HistoryDesigns,
    config: &LtmleConfig,
    paths: FitPaths,
    notes: &mut FitNotes,
) -> EconResult<TreatmentModels> {
    let t_max = data.time_points();
    let mut out = TreatmentModels {
        coefficients: Vec::with_capacity(t_max),
        models: Vec::with_capacity(t_max),
        fitted: Vec::with_capacity(t_max),
    };
    for t in 0..t_max {
        resource::checkpoint()?;
        let (x, names) = designs.g(t);
        let fit = fit_logit_node(
            x,
            data.treatments[t].view(),
            GlmFamily::Binomial,
            None,
            config,
            &format!("g model for {}", data.treatment_names[t]),
            names,
            paths,
            notes,
        )?;
        out.models.push(NodeModel {
            node: t,
            names: names.to_vec(),
            coefficients: fit.coefficients.to_vec(),
            converged: fit.converged,
            iterations: fit.iterations,
            constant_prediction: None,
        });
        out.coefficients.push(fit.coefficients);
        out.fitted.push(fit.fitted);
    }
    Ok(out)
}

/// One Q regression: its coefficients and record, and for the logistic model the response's
/// (min, max).
struct QFit {
    coefficients: Array1<f64>,
    model: NodeModel,
    range: Option<(f64, f64)>,
}

/// The Q regression of node t: `q_next` on the observed history through A_t, from `start`
/// when given (logistic model).
#[allow(clippy::too_many_arguments)]
fn fit_q(
    designs: &HistoryDesigns,
    t: usize,
    q_next: ArrayView1<f64>,
    config: &LtmleConfig,
    treatment_names: &[String],
    start: Option<ArrayView1<f64>>,
    paths: FitPaths,
    notes: &mut FitNotes,
) -> EconResult<QFit> {
    let (x, names) = designs.q(t);
    let what = format!("Q model at {}", treatment_names[t]);
    match config.q_model {
        LtmleQModel::Logistic => {
            let (min, max) = q_next
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                    (lo.min(v), hi.max(v))
                });
            if max - min < CONSTANT_Q_RANGE {
                // No regression: every subject's linear predictor is the bounded logit of the
                // maximum. The NaN coefficients drop their columns from the prediction, and the
                // next node's warm start (all coefficients finite) falls back to glm.fit's start.
                let (lo, hi) = CONSTANT_Q_BOUNDS;
                let c = max.max(lo).min(hi);
                let mut coefficients = Array1::from_elem(x.ncols(), f64::NAN);
                coefficients[0] = (c / (1.0 - c)).ln();
                let model = NodeModel {
                    node: t,
                    names: names.to_vec(),
                    coefficients: coefficients.to_vec(),
                    converged: true,
                    iterations: 0,
                    constant_prediction: Some(max),
                };
                return Ok(QFit {
                    coefficients,
                    model,
                    range: Some((min, max)),
                });
            }
            let fit = fit_logit_node(
                x,
                q_next,
                GlmFamily::QuasiBinomial,
                start,
                config,
                &what,
                names,
                paths,
                notes,
            )?;
            let model = NodeModel {
                node: t,
                names: names.to_vec(),
                coefficients: fit.coefficients.to_vec(),
                converged: fit.converged,
                iterations: fit.iterations,
                constant_prediction: None,
            };
            Ok(QFit {
                coefficients: fit.coefficients,
                model,
                range: Some((min, max)),
            })
        }
        LtmleQModel::Linear => {
            let fit = solve_ols(x, q_next)?;
            if let Some(m) = aliasing_message(&fit.aliased, names) {
                notes.warnings.push(format!(
                    "{what}: {m}; predictions use the identified columns"
                ));
            }
            let model = NodeModel {
                node: t,
                names: names.to_vec(),
                coefficients: fit.beta.to_vec(),
                converged: true,
                iterations: 0,
                constant_prediction: None,
            };
            Ok(QFit {
                coefficients: fit.beta,
                model,
                range: None,
            })
        }
    }
}

/// The start of the logistic Q regression of node `t - 1`, whose response is the targeted
/// prediction of node t: on the logit scale, node t's linear predictor with every `A_s, s <= t`,
/// at the regime's value, plus eps. The start is beta on the shared history, with the columns
/// node `t - 1` does not have (L_t and A_t) and eps moved into the intercept at their means. The
/// coefficients of the earlier `A_s` stay on the observed columns, although the response has
/// those treatments at the regime's values, so this is an approximate start, which IRLS then
/// corrects. `None` when a coefficient is aliased (glm.fit's start then).
fn warm_start(
    designs: &HistoryDesigns,
    coefficients: ArrayView1<f64>,
    t: usize,
    eps: f64,
    d_t: &[u8],
) -> Option<Array1<f64>> {
    let p0 = designs.widths[t - 1];
    let p = coefficients.len();
    let d_mean = d_t.iter().map(|v| f64::from(*v)).sum::<f64>() / d_t.len() as f64;
    let l: f64 = coefficients
        .slice(ndarray::s![p0..p - 1])
        .iter()
        .zip(&designs.l_means[t])
        .map(|(b, m)| b * m)
        .sum();
    let mut start = coefficients.slice(ndarray::s![..p0]).to_owned();
    start[0] += l + coefficients[p - 1] * d_mean + eps;
    start.iter().all(|v| v.is_finite()).then_some(start)
}

#[allow(clippy::too_many_arguments)]
fn estimate_regime(
    data: &LtmleData,
    regime: Regime,
    d: &[Vec<u8>],
    label: String,
    follow: &[Vec<bool>],
    designs: &HistoryDesigns,
    g: &TreatmentModels,
    last_q: &QFit,
    y: &Array1<f64>,
    scale: &OutcomeScale,
    config: &LtmleConfig,
    paths: FitPaths,
    notes: &mut FitNotes,
) -> EconResult<(RegimeEstimate, RegimeSaturation)> {
    let n = data.n_obs();
    let t_max = data.time_points();
    let (lo, hi) = config.gbounds;
    let mut constant_q_nodes: Vec<usize> = Vec::new();
    let mut resolved_targeting: Vec<(usize, f64)> = Vec::new();

    // Cumulative probability of following the regime, g predicted with SetA.
    let mut cum = vec![1.0f64; n];
    let mut cumulative_g = Vec::with_capacity(t_max);
    let mut h = Vec::with_capacity(t_max);
    let mut n_bounded = Vec::with_capacity(t_max);
    let mut n_bounded_following = Vec::with_capacity(t_max);
    for t in 0..t_max {
        let (x, _) = designs.g(t);
        let eta = identified_linear_predictor(
            x,
            g.coefficients[t].view(),
            &designs.a_cols[t][..t],
            Some(d),
        );
        let (mut nb, mut nbf) = (0usize, 0usize);
        let mut bounded = Vec::with_capacity(n);
        let mut ht = Vec::with_capacity(n);
        for i in 0..n {
            let p1 = linkinv(eta[i]);
            let prob = if d[t][i] == 1 { p1 } else { 1.0 - p1 };
            cum[i] *= prob;
            let out = cum[i] < lo || cum[i] > hi;
            if out {
                nb += 1;
                if follow[t][i] {
                    nbf += 1;
                }
            }
            let c = cum[i].max(lo).min(hi);
            bounded.push(c);
            ht.push(if follow[t][i] { 1.0 / c } else { 0.0 });
        }
        cumulative_g.push(bounded);
        h.push(ht);
        n_bounded.push(nb);
        n_bounded_following.push(nbf);
    }
    // ltmle's positivity check counts the unbounded cumulative g below the lower bound only.
    let share_below_gbound = cum.iter().filter(|c| **c < lo).count() as f64 / n as f64;

    // Backward pass.
    let mut q_next = y.clone();
    let mut ic = Array1::<f64>::zeros(n);
    let mut eps = vec![0.0; t_max];
    let mut q_star: Vec<Array1<f64>> = vec![Array1::zeros(0); t_max];
    let mut q_models = vec![NodeModel::default(); t_max];
    let mut start: Option<Array1<f64>> = None;
    for t in (0..t_max).rev() {
        resource::checkpoint()?;
        let fitted;
        // The last node's model is shared by every regime and copied into this one's record;
        // the other nodes' models move into it.
        let (coef, range) = if t == t_max - 1 {
            q_models[t] = last_q.model.clone();
            (&last_q.coefficients, last_q.range)
        } else {
            fitted = fit_q(
                designs,
                t,
                q_next.view(),
                config,
                &data.treatment_names,
                start.as_ref().map(|s| s.view()),
                paths,
                notes,
            )?;
            q_models[t] = fitted.model;
            (&fitted.coefficients, fitted.range)
        };
        if let Some(c) = q_models[t].constant_prediction {
            constant_q_nodes.push(t);
            let (rmin, rmax) = range.unwrap_or((f64::NAN, f64::NAN));
            notes.warnings.push(format!(
                "LTMLE: {label}: Q response at {} has range {:.1e} < {CONSTANT_Q_RANGE:e}; \
                 predicted as the constant {c:.6} (ltmle's no-variation rule)",
                data.treatment_names[t],
                rmax - rmin
            ));
        }
        let eta =
            identified_linear_predictor(designs.q(t).0, coef.view(), &designs.a_cols[t], Some(d));
        let ht = ArrayView1::from(&h[t][..]);
        let qs = match config.q_model {
            LtmleQModel::Logistic => {
                let f = fit_fluctuation(q_next.view(), eta.view(), ht, config)?;
                notes.n_fits += 1;
                notes.all_converged &= f.converged;
                if f.non_finite {
                    notes.warnings.push(format!(
                        "fluctuation at {} ({label}): the deviance stayed non-finite after \
                         {MAX_HALVINGS} step halvings in iteration {}; stopped at the last finite \
                         epsilon",
                        data.treatment_names[t], f.iterations
                    ));
                } else if !f.converged {
                    notes.warnings.push(format!(
                        "fluctuation at {} ({label}): did not converge in {} iterations",
                        data.treatment_names[t], f.iterations
                    ));
                }
                eps[t] = f.epsilon;
                // ltmle's score check (ltmle:::FixedTimeTMLE): the node's EIC term must sum to
                // about 0 over the subjects; otherwise the score equation is solved directly.
                let mut qs = eta.mapv(|e| linkinv(e + eps[t]));
                let score: f64 = (0..n).map(|i| ht[i] * (q_next[i] - qs[i])).sum();
                let mut resolved_here = false;
                if score.is_nan() || score.abs() > TARGETING_SCORE_MAX {
                    let r = resolve_fluctuation(q_next.view(), eta.view(), ht)?;
                    eps[t] = r.epsilon;
                    let (olo, ohi) = offset_bounds(r.offset_bound);
                    qs = eta.mapv(|e| linkinv(e.max(olo).min(ohi) + r.epsilon));
                    resolved_targeting.push((t, r.offset_bound));
                    resolved_here = true;
                    let lb = r.offset_bound;
                    notes.warnings.push(format!(
                        "LTMLE: {label}: the targeting step at {} left the score at {score:.1e}; \
                         re-solved with offsets bounded to [{lb:e}, 1 - {lb:e}]{}",
                        data.treatment_names[t],
                        if r.finite_root {
                            ""
                        } else {
                            " (no finite root: the score is solved only at the link's clamp)"
                        }
                    ));
                }
                if paths.warm && config.max_iter >= WARM_MIN_MAX_ITER && t > 0 {
                    // A re-solved eps was found with the offsets bounded, which warm_start does
                    // not apply, and without a finite root it sits at the link's clamp: the next
                    // node would start deep in the clamp, so it takes glm.fit's start instead.
                    start = if resolved_here {
                        None
                    } else {
                        warm_start(designs, coef.view(), t, eps[t], &d[t])
                    };
                }
                qs
            }
            LtmleQModel::Linear => {
                let (mut num, mut den) = (0.0, 0.0);
                for i in 0..n {
                    if ht[i] > 0.0 {
                        num += ht[i] * (q_next[i] - eta[i]);
                        den += ht[i];
                    }
                }
                eps[t] = num / den;
                eta.mapv(|m| m + eps[t])
            }
        };
        for i in 0..n {
            ic[i] += ht[i] * (q_next[i] - qs[i]);
        }
        q_next.assign(&qs);
        q_star[t] = qs;
    }
    let psi_s = q_next.mean().unwrap_or(f64::NAN);
    // Only the logistic Q model keeps Q* on a [0, 1] scale; a linear Q* is on the outcome's own
    // scale, where 0 and 1 are not bounds.
    let share_q_star_at_bound = if config.q_model == LtmleQModel::Logistic {
        q_next
            .iter()
            .filter(|q| **q < Q_STAR_AT_BOUND || **q > 1.0 - Q_STAR_AT_BOUND)
            .count() as f64
            / n as f64
    } else {
        0.0
    };
    for i in 0..n {
        ic[i] = (ic[i] + q_next[i] - psi_s) * scale.width;
    }
    let psi = scale.lower + scale.width * psi_s;
    let influence_curve = ic.to_vec();
    let se = ic_se(&influence_curve)?;
    let zc = z_critical(0.95)?;
    let estimate = RegimeEstimate {
        regime,
        label,
        psi,
        se,
        ci_lower: psi - zc * se,
        ci_upper: psi + zc * se,
        influence_curve,
        cumulative_g,
        clever_covariates: h,
        targeted_predictions: q_star
            .iter()
            .map(|q| q.iter().map(|v| scale.lower + scale.width * v).collect())
            .collect(),
        fluctuation_coefs: eps,
        n_following: follow
            .iter()
            .map(|f| f.iter().filter(|v| **v).count())
            .collect(),
        n_bounded,
        n_bounded_following,
        q_models,
    };
    Ok((
        estimate,
        RegimeSaturation {
            share_below_gbound,
            constant_q_nodes,
            resolved_targeting,
            share_q_star_at_bound,
        },
    ))
}

/// The logit link's inverse as R's `binomial()$linkinv` (and `predict.glm(type =
/// "response")`) computes it: `eta` beyond +/-30 gives `eps / (1 + eps)` or
/// `1 / (1 + eps)`, so a probability is never exactly 0 or 1.
fn linkinv(eta: f64) -> f64 {
    const THRESH: f64 = 30.0;
    let t = if eta < -THRESH {
        f64::EPSILON
    } else if eta > THRESH {
        1.0 / f64::EPSILON
    } else {
        eta.exp()
    };
    t / (1.0 + t)
}

/// The fluctuation of one node: epsilon of the intercept-only quasibinomial logit fit of
/// `y` with offset `offset` and prior weights `h` (`glm(y ~ 1, offset, weights = h,
/// quasibinomial)` among the followers, zero weights dropping the rest).
struct Fluctuation {
    epsilon: f64,
    converged: bool,
    iterations: usize,
    /// The step halvings ran out with the deviance still not finite.
    non_finite: bool,
}

/// For one coefficient on the canonical link, glm.fit's IRLS step is Newton's step
/// `score / information`; this solves the same score equation
/// `sum_i h_i (y_i - plogis(offset_i + eps)) = 0` in O(n) per iteration instead of a
/// general GLM fit, with glm.fit's stopping rule `|dev - dev_old| / (|dev| + 0.1) <
/// tolerance`, glm.fit's step halving when the deviance is not finite, and one polishing
/// Newton step after the rule is met. One pass over the followers per trial value gives its
/// deviance, score and information.
fn fit_fluctuation(
    y: ArrayView1<f64>,
    offset: ArrayView1<f64>,
    h: ArrayView1<f64>,
    config: &LtmleConfig,
) -> EconResult<Fluctuation> {
    // (y, offset, h) of the followers
    let mut rows: Vec<[f64; 3]> = resource::try_with_capacity(h.len(), "LTMLE fluctuation")?;
    for (&hi, (&yi, &oi)) in h.iter().zip(y.iter().zip(offset.iter())) {
        if hi > 0.0 {
            rows.push([yi, oi, hi]);
        }
    }
    // (deviance, score, information) at eps; the deviance is the binomial deviance in its
    // quasi-binomial working form, 2 sum h [y log(y/p) + (1-y) log((1-y)/(1-p))]
    let eval = |eps: f64| -> (f64, f64, f64) {
        let (mut d, mut score, mut info) = (0.0, 0.0, 0.0);
        for &[yi, oi, wi] in &rows {
            let p = linkinv(oi + eps);
            let a = if yi > 0.0 { yi * (yi / p).ln() } else { 0.0 };
            let b = if yi < 1.0 {
                (1.0 - yi) * ((1.0 - yi) / (1.0 - p)).ln()
            } else {
                0.0
            };
            d += 2.0 * wi * (a + b);
            score += wi * (yi - p);
            info += wi * p * (1.0 - p);
        }
        (d, score, info)
    };
    let mut eps = 0.0;
    let (mut dev, mut score, mut info) = eval(eps);
    if !dev.is_finite() {
        return Err(EconError::Computation(
            "LTMLE: the fluctuation's starting deviance is not finite".into(),
        ));
    }
    for iter in 1..=config.max_iter {
        resource::checkpoint()?;
        if info.is_nan() || info <= 0.0 || score == 0.0 {
            return Ok(Fluctuation {
                epsilon: eps,
                converged: score == 0.0,
                iterations: iter,
                non_finite: false,
            });
        }
        // glm.fit halves the step only when the deviance is not finite; when 30 halvings do not
        // reach a finite deviance, the last finite eps is kept and the fit stops unconverged.
        let Some((new_eps, new)) = halved_step(eps, score / info, &eval) else {
            return Ok(Fluctuation {
                epsilon: eps,
                converged: false,
                iterations: iter,
                non_finite: true,
            });
        };
        let change = (new.0 - dev).abs() / (new.0.abs() + 0.1);
        eps = new_eps;
        (dev, score, info) = new;
        if change < config.tolerance {
            // One more Newton step: the deviance rule leaves an O(sqrt(tolerance)) error in
            // eps when the weights are large, and Newton's step squares it at O(n) cost.
            if info > 0.0 {
                let polished = eps + score / info;
                if eval(polished).0.is_finite() {
                    eps = polished;
                }
            }
            return Ok(Fluctuation {
                epsilon: eps,
                converged: true,
                iterations: iter,
                non_finite: false,
            });
        }
    }
    Ok(Fluctuation {
        epsilon: eps,
        converged: false,
        iterations: config.max_iter,
        non_finite: false,
    })
}

/// Step halvings glm.fit allows when the deviance is not finite.
const MAX_HALVINGS: usize = 30;

/// `eps + step`, halved up to [`MAX_HALVINGS`] times while the deviance (the first element of
/// `eval`) is not finite: the new eps and its evaluation, or `None` when no trial is finite (at
/// most `MAX_HALVINGS + 1` evaluations).
fn halved_step(
    eps: f64,
    mut step: f64,
    eval: &impl Fn(f64) -> (f64, f64, f64),
) -> Option<(f64, (f64, f64, f64))> {
    for halvings in 0..=MAX_HALVINGS {
        if halvings > 0 {
            step /= 2.0;
        }
        let new = eval(eps + step);
        if new.0.is_finite() {
            return Some((eps + step, new));
        }
    }
    None
}

/// ltmle's trigger for re-solving a targeting step: `|sum_i H_i (Q*_{t+1,i} - Q*_{t,i})|` above
/// this (absolute, on the fitted [0, 1] scale). ltmle divides the sum by the number of regimes it
/// targets together; each regime is targeted alone here, so the divisor is 1.
const TARGETING_SCORE_MAX: f64 = 1e-3;
/// The offset bounds ltmle's `FixScoreEquation` tries in turn: offsets bounded to
/// `logit([lb, 1 - lb])`.
const OFFSET_BOUNDS: [f64; 4] = [1e-8, 1e-4, 1e-3, 0.01];
/// The re-solve's target: `|S(eps)| <= RESOLVE_REL_SCORE * sum_i h_i`.
const RESOLVE_REL_SCORE: f64 = 1e-12;
/// The re-solve's iteration limit.
const RESOLVE_MAX_ITER: usize = 200;
/// `linkinv` is constant beyond this `|eta|`.
const LINK_CLAMP: f64 = 30.0;

/// `(logit(lb), logit(1 - lb))`.
fn offset_bounds(lb: f64) -> (f64, f64) {
    let lo = (lb / (1.0 - lb)).ln();
    (lo, -lo)
}

/// A re-solved targeting step.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Resolved {
    /// The fluctuation coefficient.
    epsilon: f64,
    /// The `lb` of the offset bounds `logit([lb, 1 - lb])` it was solved with.
    offset_bound: f64,
    /// False when `S` has no root (every follower's response at 0, or every one at 1), or when
    /// the bracket reached rounding without a sign change: in both cases the score is solved
    /// only at linkinv's clamp.
    finite_root: bool,
}

/// Solves the targeting step's score equation `S(eps) = sum_i h_i (y_i - linkinv(o_i + eps)) = 0`
/// directly, as ltmle's `FixScoreEquation` does when the fluctuation glm leaves the score above
/// [`TARGETING_SCORE_MAX`], with the offsets bounded to `logit([lb, 1 - lb])`. `S` is
/// non-increasing in eps, so this is a bracketed one-dimensional root search (bisection with
/// Newton steps inside the bracket) to `|S| <= 1e-12 * sum_i h_i`, not a port of ltmle's nlminb
/// on `S^2`, which stops at `|S| < 1e-4`. Rows with `h_i = 0` do not enter.
///
/// With responses neither all at 1 nor all at 0, `S` changes sign inside the initial bracket, so
/// the search ends at the first bound, `lb = 1e-8`: at a root, or at a bracket two adjacent
/// doubles wide that `S` crosses 0 in, which counts as solved (a bracket at rounding without a
/// sign change is reported with `finite_root` false). The later bounds of [`OFFSET_BOUNDS`]
/// mirror the sequence ltmle tries when nlminb fails and are reached only if the iteration limit
/// is hit; ltmle's result is the same, since its first bound also succeeds.
fn resolve_fluctuation(
    y: ArrayView1<f64>,
    offset: ArrayView1<f64>,
    h: ArrayView1<f64>,
) -> EconResult<Resolved> {
    let mut rows: Vec<[f64; 3]> = resource::try_with_capacity(h.len(), "LTMLE re-solve")?;
    for (&hi, (&yi, &oi)) in h.iter().zip(y.iter().zip(offset.iter())) {
        if !(hi.is_finite() && yi.is_finite() && !oi.is_nan()) {
            return Err(EconError::Computation(
                "LTMLE: non-finite input to the targeting step's re-solve".into(),
            ));
        }
        if hi > 0.0 {
            rows.push([yi, oi, hi]);
        }
    }
    let sum_h: f64 = rows.iter().map(|r| r[2]).sum();
    if !(sum_h > 0.0 && sum_h.is_finite()) {
        return Err(EconError::Computation(
            "LTMLE: the targeting step's re-solve has no follower".into(),
        ));
    }
    let target = RESOLVE_REL_SCORE * sum_h;
    // S tends to sum h (y - 1) and sum h y at the two ends: with every response at 1 (or 0) it has
    // no root, and every follower goes to linkinv's clamp. The limits do not depend on the
    // offsets, so no bound would help.
    let no_root = if rows.iter().all(|r| r[0] >= 1.0) {
        Some(1.0)
    } else if rows.iter().all(|r| r[0] <= 0.0) {
        Some(-1.0)
    } else {
        None
    };
    if let Some(side) = no_root {
        let (blo, bhi) = offset_bounds(OFFSET_BOUNDS[0]);
        let (omin, omax) = rows
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), r| {
                let o = r[1].max(blo).min(bhi);
                (a.min(o), b.max(o))
            });
        return Ok(Resolved {
            epsilon: if side > 0.0 {
                LINK_CLAMP - omin
            } else {
                -LINK_CLAMP - omax
            },
            offset_bound: OFFSET_BOUNDS[0],
            finite_root: false,
        });
    }
    for lb in OFFSET_BOUNDS {
        let (blo, bhi) = offset_bounds(lb);
        let mut off: Vec<f64> = resource::try_with_capacity(rows.len(), "LTMLE re-solve")?;
        off.extend(rows.iter().map(|r| r[1].max(blo).min(bhi)));
        // (score, information) at eps
        let eval = |eps: f64| -> (f64, f64) {
            let (mut score, mut info) = (0.0, 0.0);
            for (r, o) in rows.iter().zip(&off) {
                let p = linkinv(o + eps);
                score += r[2] * (r[0] - p);
                info += r[2] * p * (1.0 - p);
            }
            (score, info)
        };
        let (omin, omax) = off
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                (a.min(v), b.max(v))
            });
        // Beyond this bracket every follower is at linkinv's clamp, where S no longer changes.
        let (mut lo, mut hi) = (-LINK_CLAMP - omax, LINK_CLAMP - omin);
        let mut eps = 0.0f64.max(lo).min(hi);
        for _ in 0..RESOLVE_MAX_ITER {
            resource::checkpoint()?;
            let (score, info) = eval(eps);
            if score.abs() <= target {
                return Ok(Resolved {
                    epsilon: eps,
                    offset_bound: lb,
                    finite_root: true,
                });
            }
            if score > 0.0 {
                lo = eps;
            } else {
                hi = eps;
            }
            let newton = eps + score / info;
            eps = if info > 0.0 && newton > lo && newton < hi {
                newton
            } else {
                0.5 * (lo + hi)
            };
            if hi - lo <= 4.0 * f64::EPSILON * (1.0 + eps.abs()) {
                // The bracket is at rounding. S steps across 0 between adjacent doubles when it is
                // positive at lo and negative at hi, as at the initial ends with mixed responses;
                // without that the search has run into linkinv's clamp, not a root.
                let crossed = eval(lo).0 > 0.0 && eval(hi).0 < 0.0;
                return Ok(Resolved {
                    epsilon: eps,
                    offset_bound: lb,
                    finite_root: crossed,
                });
            }
        }
    }
    // ltmle stops with "minimizer failed" when no bound works.
    Err(EconError::ConvergenceFailure {
        iterations: RESOLVE_MAX_ITER,
        last_change: f64::NAN,
        suggestion: "LTMLE's targeting step was not solved: raise the lower g bound (gbounds) or \
                     check how many subjects follow the regime at each node"
            .into(),
    })
}

/// `sd(ic) / sqrt(n)` with the n - 1 denominator.
fn ic_se(ic: &[f64]) -> EconResult<f64> {
    let n = ic.len();
    let df = n
        .checked_sub(1)
        .filter(|d| *d > 0)
        .ok_or_else(|| EconError::InsufficientData {
            required: 2,
            provided: n,
            context: "LTMLE: influence-curve standard error".into(),
        })?;
    let mean = ic.iter().sum::<f64>() / n as f64;
    let ss: f64 = ic.iter().map(|v| (v - mean) * (v - mean)).sum();
    Ok((ss / df as f64).sqrt() / (n as f64).sqrt())
}

/// psi(a) - psi(b) with the differenced EIC.
struct Contrast {
    estimate: f64,
    se: f64,
    z: f64,
    p_value: f64,
    ci_lower: f64,
    ci_upper: f64,
    influence_curve: Vec<f64>,
}

fn contrast_inference(
    a: &RegimeEstimate,
    b: &RegimeEstimate,
    notes: &mut FitNotes,
) -> EconResult<Contrast> {
    let estimate = a.psi - b.psi;
    let influence_curve: Vec<f64> = a
        .influence_curve
        .iter()
        .zip(&b.influence_curve)
        .map(|(x, y)| x - y)
        .collect();
    let se = ic_se(&influence_curve)?;
    let (z, p_value) = if se.is_finite() && se > 0.0 {
        let z = estimate / se;
        (z, normal_p_value_two_sided(z))
    } else {
        notes
            .warnings
            .push("LTMLE: the contrast's influence-curve variance is 0; no test statistic".into());
        (f64::NAN, f64::NAN)
    };
    let zc = z_critical(0.95)?;
    Ok(Contrast {
        estimate,
        se,
        z,
        p_value,
        ci_lower: estimate - zc * se,
        ci_upper: estimate + zc * se,
        influence_curve,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::s;
    use p2a_validation::{Check, Tol};

    /// The case dataset as LtmleData: `y` is the final outcome, `a[t]` the treatment at node t,
    /// `l[t]` the covariates observed before it (missing values refused).
    fn case_data(ds: &crate::data::Dataset, y: &str, a: &[&str], l: &[&[&str]]) -> LtmleData {
        LtmleData::from_dataset(ds, y, a, l).unwrap()
    }

    fn two_period(ds: &crate::data::Dataset) -> LtmleData {
        case_data(ds, "Y", &["A1", "A2"], &[&["L0"], &["L1"]])
    }

    fn three_period(ds: &crate::data::Dataset) -> LtmleData {
        case_data(ds, "Y3", &["A1", "A2", "A3"], &[&["L0"], &["L1"], &["L2"]])
    }

    /// The case's simulated data (`ltmle_sim.csv`) for the property and error tests, read
    /// without `load_case` so that they stay out of the R-pinned registry.
    fn sim() -> crate::data::Dataset {
        let path = p2a_validation::repo_path("validation/datasets/generated/ltmle_sim.csv");
        crate::data::Dataset::new(crate::data::DataLoader::load_csv(&path).unwrap())
    }

    /// The case's tight refit: every glm at glm.control(epsilon = 1e-14, maxit = 100).
    fn tight() -> LtmleConfig {
        let mut c = LtmleConfig::default();
        c.tolerance = 1e-14;
        c
    }

    #[test]
    fn test_validate_ltmle_ate_se() {
        let (case, ds) = crate::test_support::load_case("ltmle_two_period");
        let r = run_ltmle(&two_period(&ds), tight()).unwrap();
        let n = r.n_obs;
        Check::new(&case)
            .scalar("psi_always", r.psi_treated, Tol::coef(n))
            .scalar("psi_never", r.psi_control, Tol::coef(n))
            .scalar("se_always", r.psi_treated_se, Tol::se(n))
            .scalar("se_never", r.psi_control_se, Tol::se(n))
            .scalar("ate", r.ate, Tol::coef(n))
            .scalar("ate_se", r.se, Tol::se(n))
            .scalar("ate_z", r.z_stat, Tol::stat(n))
            .scalar("ate_p", r.p_value, Tol::p_value(n))
            .scalar("ate_ci_lower", r.ci_lower, Tol::stat(n))
            .scalar("ate_ci_upper", r.ci_upper, Tol::stat(n))
            .vector("ic_ate_head", &r.influence_curve[..5], Tol::coef(n))
            .vector("eps_always", &r.fluctuation_coefs, Tol::coef(n))
            .finish();
    }

    #[test]
    fn test_validate_ltmle_intervention_changes_estimate() {
        let (case, ds) = crate::test_support::load_case("ltmle_two_period");
        let mut c = tight();
        c.intervention = InterventionType::NeverTreat.into();
        let r = run_ltmle(&two_period(&ds), c).unwrap();
        Check::new(&case)
            .scalar("psi_never", r.psi_treated, Tol::coef(r.n_obs))
            .finish();
        assert!((r.psi_treated - case.scalar("psi_always")).abs() > 0.2);
    }

    #[test]
    fn test_validate_ltmle_default_control() {
        // The default glm control (epsilon 1e-8) against the tight (1e-14) reference values:
        // measured gaps 0 (psi_always, psi_never), 5.2e-16 relative (ATE SE).
        let (case, ds) = crate::test_support::load_case("ltmle_two_period");
        let r = run_ltmle(&two_period(&ds), LtmleConfig::default()).unwrap();
        Check::new(&case)
            .scalar("psi_always", r.psi_treated, Tol::iterative(1e-8))
            .scalar("psi_never", r.psi_control, Tol::iterative(1e-8))
            .scalar("ate_se", r.se, Tol::iterative(1e-8))
            .finish();
        // the tight eps_never at the default control: measured gap 2.0e-12 relative
        Check::new(&case)
            .vector(
                "eps_never",
                &r.contrast_estimate.as_ref().unwrap().fluctuation_coefs,
                Tol::iterative(1e-8),
            )
            .finish();
        assert!(r.converged, "{:?}", r.warnings);
        assert_eq!(r.gbounds, (0.01, 1.0));
    }

    #[test]
    fn test_validate_ltmle_linear_q() {
        let (case, ds) = crate::test_support::load_case("ltmle_two_period");
        let data = case_data(&ds, "Yc", &["A1", "A2"], &[&["L0"], &["L1"]]);
        let mut c = tight();
        c.q_model = LtmleQModel::Linear;
        let r = run_ltmle(&data, c).unwrap();
        let n = r.n_obs;
        Check::new(&case)
            .scalar("psi_always_lin", r.psi_treated, Tol::coef(n))
            .scalar("psi_never_lin", r.psi_control, Tol::coef(n))
            .scalar("ate_lin", r.ate, Tol::coef(n))
            .scalar("ate_se_lin", r.se, Tol::se(n))
            .vector("eps_always_lin", &r.fluctuation_coefs, Tol::coef(n))
            .finish();
    }

    #[test]
    fn test_validate_ltmle_randomized_known_effect() {
        let (case, ds) = crate::test_support::load_case("ltmle_two_period");
        let data = case_data(&ds, "Yr", &["R1", "R2"], &[&["L0"], &["Lr"]]);
        let mut c = tight();
        c.q_model = LtmleQModel::Linear;
        let r = run_ltmle(&data, c).unwrap();
        let n = r.n_obs;
        Check::new(&case)
            .scalar("psi_always_rand", r.psi_treated, Tol::coef(n))
            .scalar("psi_never_rand", r.psi_control, Tol::coef(n))
            .scalar("ate_rand", r.ate, Tol::coef(n))
            .scalar("ate_se_rand", r.se, Tol::se(n))
            .finish();
        let truth = case.scalar("truth_rand");
        assert!(
            (r.ate - truth).abs() < 3.0 * r.se,
            "ATE {} (SE {}) vs the known effect {truth}",
            r.ate,
            r.se
        );
    }

    #[test]
    fn test_validate_ltmle_logistic_rescaled_outcome() {
        let (case, ds) = crate::test_support::load_case("ltmle_two_period");
        let data = case_data(&ds, "Yc", &["A1", "A2"], &[&["L0"], &["L1"]]);
        let r = run_ltmle(&data, tight()).unwrap();
        let n = r.n_obs;
        Check::new(&case)
            .scalar("psi_always_cont", r.psi_treated, Tol::coef(n))
            .scalar("psi_never_cont", r.psi_control, Tol::coef(n))
            .scalar("ate_cont", r.ate, Tol::coef(n))
            .scalar("ate_se_cont", r.se, Tol::se(n))
            .vector("eps_always_cont", &r.fluctuation_coefs, Tol::coef(n))
            .finish();
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("rescaled from [-3.1578, 9.1856]")),
            "{:?}",
            r.warnings
        );
        let (lo, hi) = r.outcome_range.unwrap();
        Check::new(&case)
            .vector("yc_range", &[lo, hi], Tol::exact())
            .finish();
    }

    #[test]
    fn test_validate_ltmle_gbounds_cumulative() {
        let (case, ds) = crate::test_support::load_case("ltmle_two_period");
        let mut c = tight();
        c.gbounds = (0.1, 0.5);
        let r = run_ltmle(&two_period(&ds), c).unwrap();
        let n = r.n_obs;
        Check::new(&case)
            .scalar("psi_always_gb", r.psi_treated, Tol::coef(n))
            .scalar("psi_never_gb", r.psi_control, Tol::coef(n))
            .scalar("ate_gb", r.ate, Tol::coef(n))
            .scalar("ate_se_gb", r.se, Tol::se(n))
            .finish();
        // n_bounded_follow_always_gb + n_bounded_follow_never_gb
        assert_eq!(r.n_truncated_by_time, vec![192 + 130, 19 + 3]);
        let a = &r.regime_estimate;
        let b = r.contrast_estimate.as_ref().unwrap();
        let f = |v: &[usize]| v.iter().map(|x| *x as f64).collect::<Vec<_>>();
        Check::new(&case)
            .vector("n_bounded_always_gb", &f(&a.n_bounded), Tol::exact())
            .vector("n_bounded_never_gb", &f(&b.n_bounded), Tol::exact())
            .vector(
                "n_bounded_follow_always_gb",
                &f(&a.n_bounded_following),
                Tol::exact(),
            )
            .vector(
                "n_bounded_follow_never_gb",
                &f(&b.n_bounded_following),
                Tol::exact(),
            )
            .finish();
    }

    #[test]
    fn test_validate_ltmle_three_period() {
        let (case, ds) = crate::test_support::load_case("ltmle_two_period");
        let r = run_ltmle(&three_period(&ds), tight()).unwrap();
        let n = r.n_obs;
        Check::new(&case)
            .scalar("t3_psi_always", r.psi_treated, Tol::coef(n))
            .scalar("t3_psi_never", r.psi_control, Tol::coef(n))
            .scalar("t3_ate", r.ate, Tol::coef(n))
            .vector("t3_eps_always", &r.fluctuation_coefs, Tol::coef(n))
            .scalar("t3_se_always", r.psi_treated_se, Tol::se(n))
            .scalar("t3_se_never", r.psi_control_se, Tol::se(n))
            .scalar("t3_ate_se", r.se, Tol::se(n))
            .finish();
    }

    #[test]
    fn test_ltmle_config_validation_errors() {
        let ds = sim();
        let data = two_period(&ds);
        for gb in [
            (0.6, 0.4),
            (0.0, 1.0),
            (0.01, 1.5),
            (f64::NAN, 1.0),
            (0.5, 0.5),
        ] {
            let mut c = LtmleConfig::default();
            c.gbounds = gb;
            let e = run_ltmle(&data, c).unwrap_err().to_string();
            assert!(e.contains("gbounds"), "{gb:?}: {e}");
        }
        for tol in [0.0, f64::NAN] {
            let mut c = LtmleConfig::default();
            c.tolerance = tol;
            let e = run_ltmle(&data, c).unwrap_err().to_string();
            assert!(e.contains("tolerance"), "{tol}: {e}");
        }
        let mut c = LtmleConfig::default();
        c.max_iter = 0;
        let e = run_ltmle(&data, c).unwrap_err().to_string();
        assert!(e.contains("max_iter"), "{e}");
    }

    #[test]
    fn test_ltmle_no_follower_is_an_error() {
        let ds = sim();
        let mut data = two_period(&ds);
        data.treatments[1] = data.treatments[0].mapv(|a| 1.0 - a);
        match run_ltmle(&data, LtmleConfig::default()) {
            Err(EconError::InsufficientData {
                provided: 0,
                context,
                ..
            }) => {
                assert!(context.contains("no subject follows"), "{context}");
                assert!(context.contains("A2"), "{context}");
            }
            other => panic!("expected InsufficientData, got {other:?}"),
        }
    }

    #[test]
    fn test_ltmle_too_few_observations() {
        let ds = sim();
        let full = two_period(&ds);
        let data = LtmleData::new(
            full.outcomes
                .iter()
                .map(|y| y.slice(s![..5]).to_owned())
                .collect(),
            full.treatments
                .iter()
                .map(|a| a.slice(s![..5]).to_owned())
                .collect(),
            full.covariates
                .iter()
                .map(|l| l.slice(s![..5, ..]).to_owned())
                .collect(),
        )
        .unwrap();
        match run_ltmle(&data, LtmleConfig::default()) {
            Err(EconError::InsufficientData {
                required: 6,
                provided: 5,
                ..
            }) => {}
            other => panic!("expected InsufficientData(6, 5), got {other:?}"),
        }
    }

    #[test]
    fn test_ltmle_data_rejects_non_finite() {
        let n = 10;
        let base = || {
            (
                vec![Array1::<f64>::zeros(n), Array1::from_elem(n, 0.5)],
                vec![
                    Array1::from_iter((0..n).map(|i| (i % 2) as f64)),
                    Array1::from_iter((0..n).map(|i| ((i / 2) % 2) as f64)),
                ],
                vec![Array2::<f64>::zeros((n, 1)), Array2::<f64>::zeros((n, 1))],
            )
        };
        let (y, a, mut l) = base();
        l[1][[3, 0]] = f64::NAN;
        let e = LtmleData::new(y, a, l).unwrap_err().to_string();
        assert!(e.contains("time point 2"), "{e}");
        let (mut y, a, l) = base();
        y[1][4] = f64::NAN;
        let e = LtmleData::new(y, a, l).unwrap_err().to_string();
        assert!(e.contains("outcome"), "{e}");

        // the variant, and a treatment's missing or infinite value is reported as such rather
        // than as a non-binary value
        let (y, mut a, l) = base();
        a[0][2] = f64::NAN;
        match LtmleData::new(y, a, l) {
            Err(EconError::NullValues { column, count: 1 }) => {
                assert!(column.contains("treatment (time point 1)"), "{column}")
            }
            other => panic!("expected NullValues, got {other:?}"),
        }
        let (y, mut a, l) = base();
        a[1][2] = f64::INFINITY;
        match LtmleData::new(y, a, l) {
            Err(EconError::NonFiniteValues { column, count: 1 }) => {
                assert!(column.contains("treatment (time point 2)"), "{column}")
            }
            other => panic!("expected NonFiniteValues, got {other:?}"),
        }
        let (y, a, mut l) = base();
        l[0][[1, 0]] = f64::NEG_INFINITY;
        assert!(matches!(
            LtmleData::new(y, a, l),
            Err(EconError::NonFiniteValues { count: 1, .. })
        ));
        let (mut y, a, l) = base();
        y[1][0] = f64::INFINITY;
        match LtmleData::new(y, a, l) {
            Err(EconError::NonFiniteValues { column, count: 1 }) => {
                assert!(column.contains("outcome"), "{column}")
            }
            other => panic!("expected NonFiniteValues, got {other:?}"),
        }
        let (mut y, a, l) = base();
        y[0][0] = f64::NAN;
        assert!(matches!(
            LtmleData::new(y, a, l),
            Err(EconError::NullValues { count: 1, .. })
        ));
    }

    #[test]
    fn test_ltmle_fluctuation_halving_stops_at_a_non_finite_deviance() {
        // A deviance that is never finite: the halvings stop after MAX_HALVINGS + 1 evaluations
        // and report no step, so the fit keeps the last finite eps.
        let calls = std::cell::Cell::new(0usize);
        let eval = |_: f64| {
            calls.set(calls.get() + 1);
            (f64::INFINITY, 0.0, 0.0)
        };
        assert!(halved_step(0.25, 1.0, &eval).is_none());
        assert_eq!(calls.get(), MAX_HALVINGS + 1);
        // finite after three halvings: the step is 1/8
        calls.set(0);
        let eval = |e: f64| {
            calls.set(calls.get() + 1);
            if e > 0.3 {
                (f64::NAN, 0.0, 0.0)
            } else {
                (1.0, 2.0, 3.0)
            }
        };
        let (eps, new) = halved_step(0.1, 1.6, &eval).unwrap();
        assert_eq!(calls.get(), 5);
        assert!((eps - 0.2).abs() < 1e-15, "{eps}");
        assert_eq!(new, (1.0, 2.0, 3.0));
    }

    #[test]
    fn test_ltmle_shared_history_designs_match_per_node_designs() {
        // Each node's design is a column prefix of the shared matrix; it equals the design
        // built for that node alone, and a fit on the prefix view is bit-identical to a fit on
        // a contiguous copy.
        let data = long_horizon(300, 4, 17);
        let designs = HistoryDesigns::new(&data).unwrap();
        let n = data.n_obs();
        let mut own: Vec<Array2<f64>> = Vec::new();
        for t in 0..data.time_points() {
            let l = &data.covariates[t];
            let p0 = own.last().map_or(1, |x| x.ncols());
            let mut x = Array2::<f64>::zeros((n, p0 + l.ncols() + 1));
            for i in 0..n {
                for j in 0..p0 {
                    x[[i, j]] = own.last().map_or(1.0, |px| px[[i, j]]);
                }
                for j in 0..l.ncols() {
                    x[[i, p0 + j]] = l[[i, j]];
                }
                x[[i, p0 + l.ncols()]] = data.treatments[t][i];
            }
            own.push(x);
        }
        let config = LtmleConfig::default();
        let paths = FitPaths {
            lean: true,
            warm: true,
        };
        for (t, x) in own.iter().enumerate() {
            let (view, names) = designs.q(t);
            assert_eq!(view.dim(), x.dim());
            assert!(
                view.iter()
                    .zip(x.iter())
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            assert_eq!(names.len(), x.ncols());
            let (g, _) = designs.g(t);
            let fit = |x: ArrayView2<f64>| {
                let mut notes = FitNotes {
                    warnings: Vec::new(),
                    n_fits: 0,
                    all_converged: true,
                };
                fit_logit_node(
                    x,
                    data.treatments[t].view(),
                    GlmFamily::Binomial,
                    None,
                    &config,
                    "g",
                    names,
                    paths,
                    &mut notes,
                )
                .unwrap()
            };
            let shared = fit(g);
            let contiguous = fit(x.slice(ndarray::s![.., ..x.ncols() - 1]).to_owned().view());
            for (a, b) in shared.coefficients.iter().zip(&contiguous.coefficients) {
                assert_eq!(a.to_bits(), b.to_bits(), "node {t}");
            }
            assert_eq!(shared.iterations, contiguous.iterations);
        }
    }

    fn regime_run(data: &LtmleData, regime: Regime) -> LtmleResult {
        let mut c = tight();
        c.intervention = regime;
        c.contrast = None;
        run_ltmle(data, c).unwrap()
    }

    fn counts(v: &[usize]) -> Vec<f64> {
        v.iter().map(|x| *x as f64).collect()
    }

    #[test]
    fn test_validate_ltmle_always() {
        let (case, ds) = crate::test_support::load_case("ltmle_two_period");
        let data = two_period(&ds);
        let r = regime_run(&data, Regime::Static(vec![1, 1]));
        let e = &r.regime_estimate;
        let n = r.n_obs;
        Check::new(&case)
            .scalar("psi_always", e.psi, Tol::coef(n))
            .scalar("se_always", e.se, Tol::se(n))
            .vector("eps_always", &e.fluctuation_coefs, Tol::coef(n))
            .vector("n_follow_always", &counts(&e.n_following), Tol::exact())
            .vector(
                "h_always_2_head",
                &e.clever_covariates[1][..8],
                Tol::coef(n),
            )
            .named(
                "q2_coef",
                &e.q_models[1].names,
                &e.q_models[1].coefficients,
                Tol::coef(n),
            )
            .named(
                "q1_coef_always",
                &e.q_models[0].names,
                &e.q_models[0].coefficients,
                Tol::coef(n),
            )
            .finish();
        assert!(r.ate.is_nan() && r.contrast_estimate.is_none());
        let r1 = regime_run(&data, Regime::always());
        assert_eq!(r1.psi_treated.to_bits(), r.psi_treated.to_bits());
    }

    #[test]
    fn test_validate_ltmle_never() {
        let (case, ds) = crate::test_support::load_case("ltmle_two_period");
        let r = regime_run(&two_period(&ds), Regime::never());
        let e = &r.regime_estimate;
        let n = r.n_obs;
        Check::new(&case)
            .scalar("psi_never", e.psi, Tol::coef(n))
            .scalar("se_never", e.se, Tol::se(n))
            .vector("eps_never", &e.fluctuation_coefs, Tol::coef(n))
            .vector("n_follow_never", &counts(&e.n_following), Tol::exact())
            .named(
                "q1_coef_never",
                &e.q_models[0].names,
                &e.q_models[0].coefficients,
                Tol::coef(n),
            )
            .vector("cumg_never_2_head", &e.cumulative_g[1][..8], Tol::coef(n))
            .finish();
    }

    fn dynamic_two() -> Regime {
        Regime::Dynamic(vec![
            DynamicRule {
                node: 0,
                column: "L0".into(),
                threshold: 0.0,
                above: 1,
            },
            DynamicRule {
                node: 1,
                column: "L1".into(),
                threshold: 0.0,
                above: 1,
            },
        ])
    }

    #[test]
    fn test_validate_ltmle_dynamic() {
        let (case, ds) = crate::test_support::load_case("ltmle_two_period");
        let data = two_period(&ds);
        let mut c = tight();
        c.intervention = dynamic_two();
        c.contrast = Some(Regime::never());
        let r = run_ltmle(&data, c).unwrap();
        let e = &r.regime_estimate;
        let n = r.n_obs;
        Check::new(&case)
            .scalar("psi_dynamic", e.psi, Tol::coef(n))
            .scalar("se_dynamic", e.se, Tol::se(n))
            .vector("eps_dynamic", &e.fluctuation_coefs, Tol::coef(n))
            .vector("n_follow_dynamic", &counts(&e.n_following), Tol::exact())
            .vector(
                "h_dynamic_1_head",
                &e.clever_covariates[0][..8],
                Tol::coef(n),
            )
            .scalar("ate_dynamic_vs_never", r.ate, Tol::coef(n))
            .scalar("ate_dynamic_vs_never_se", r.se, Tol::se(n))
            .finish();
        assert_eq!(e.label, "dynamic (A1 = 1{L0 > 0}, A2 = 1{L1 > 0})");
        assert_eq!(
            r.contrast_label,
            Some("dynamic (A1 = 1{L0 > 0}, A2 = 1{L1 > 0}) - never treat".into())
        );
        let r2 = regime_run(
            &data,
            Regime::threshold_rules(&["L0".into(), "L1".into()], 0.0),
        );
        assert_eq!(r2.psi_treated.to_bits(), e.psi.to_bits());
    }

    #[test]
    fn test_validate_ltmle_static_mixed() {
        let (case, ds) = crate::test_support::load_case("ltmle_two_period");
        let r = regime_run(&two_period(&ds), Regime::Static(vec![1, 0]));
        let e = &r.regime_estimate;
        let n = r.n_obs;
        Check::new(&case)
            .scalar("psi_10", e.psi, Tol::coef(n))
            .scalar("se_10", e.se, Tol::se(n))
            .vector("eps_10", &e.fluctuation_coefs, Tol::coef(n))
            .vector("n_follow_10", &counts(&e.n_following), Tol::exact())
            .finish();
        assert_eq!(e.label, "static (A1 = 1, A2 = 0)");
        assert!((e.psi - case.scalar("psi_always")).abs() > 0.02);
    }

    #[test]
    fn test_validate_ltmle_three_period_regimes() {
        let (case, ds) = crate::test_support::load_case("ltmle_two_period");
        let data = three_period(&ds);
        let n = data.n_obs();
        let r = regime_run(&data, Regime::always());
        let e = &r.regime_estimate;
        Check::new(&case)
            .scalar("t3_psi_always", e.psi, Tol::coef(n))
            .scalar("t3_se_always", e.se, Tol::se(n))
            .vector("t3_eps_always", &e.fluctuation_coefs, Tol::coef(n))
            .vector("t3_n_follow_always", &counts(&e.n_following), Tol::exact())
            .named(
                "t3_q2_coef_always",
                &e.q_models[1].names,
                &e.q_models[1].coefficients,
                Tol::coef(n),
            )
            .named(
                "t3_g3_coef",
                &r.g_models[2].names,
                &r.g_models[2].coefficients,
                Tol::coef(n),
            )
            .finish();
        assert_eq!(
            r.g_models[2].names,
            vec!["(Intercept)", "L0", "A1", "L1", "A2", "L2"]
        );
        let r = regime_run(&data, Regime::never());
        let e = &r.regime_estimate;
        Check::new(&case)
            .scalar("t3_psi_never", e.psi, Tol::coef(n))
            .scalar("t3_se_never", e.se, Tol::se(n))
            .vector("t3_eps_never", &e.fluctuation_coefs, Tol::coef(n))
            .vector("t3_n_follow_never", &counts(&e.n_following), Tol::exact())
            .vector("t3_n_bounded_never", &counts(&e.n_bounded), Tol::exact())
            .vector(
                "t3_n_bounded_follow_never",
                &counts(&e.n_bounded_following),
                Tol::exact(),
            )
            .finish();
        let r = regime_run(
            &data,
            Regime::threshold_rules(&["L0".to_string(), "L1".to_string(), "L2".to_string()], 0.0),
        );
        let e = &r.regime_estimate;
        Check::new(&case)
            .scalar("t3_psi_dynamic", e.psi, Tol::coef(n))
            .scalar("t3_se_dynamic", e.se, Tol::se(n))
            .vector("t3_eps_dynamic", &e.fluctuation_coefs, Tol::coef(n))
            .vector("t3_n_follow_dynamic", &counts(&e.n_following), Tol::exact())
            .finish();
        let r = regime_run(&data, Regime::Static(vec![1, 0, 1]));
        let e = &r.regime_estimate;
        Check::new(&case)
            .scalar("t3_psi_101", e.psi, Tol::coef(n))
            .scalar("t3_se_101", e.se, Tol::se(n))
            .vector("t3_eps_101", &e.fluctuation_coefs, Tol::coef(n))
            .vector("t3_n_follow_101", &counts(&e.n_following), Tol::exact())
            .finish();
    }

    #[test]
    fn test_validate_ltmle_nuisance_models() {
        let (case, ds) = crate::test_support::load_case("ltmle_two_period");
        let r = run_ltmle(&two_period(&ds), tight()).unwrap();
        let n = r.n_obs;
        let never = r.contrast_estimate.as_ref().unwrap();
        Check::new(&case)
            .named(
                "g1_coef",
                &r.g_models[0].names,
                &r.g_models[0].coefficients,
                Tol::coef(n),
            )
            .named(
                "g2_coef",
                &r.g_models[1].names,
                &r.g_models[1].coefficients,
                Tol::coef(n),
            )
            .named(
                "q1_coef_never",
                &never.q_models[0].names,
                &never.q_models[0].coefficients,
                Tol::coef(n),
            )
            .finish();
        // 2 g fits, the shared last-node Q fit, and per regime one Q fit and two fluctuations
        assert_eq!(r.n_model_fits, 9);
    }

    #[test]
    fn test_ltmle_regime_validation_errors() {
        let data = two_period(&sim());
        let rule = |node: usize, column: &str, threshold: f64, above: u8| DynamicRule {
            node,
            column: column.into(),
            threshold,
            above,
        };
        let cases: Vec<(Regime, &str)> = vec![
            (Regime::Static(vec![1, 1, 1]), "1 or 2 values"),
            (Regime::Static(vec![2]), "must be 0 or 1"),
            (
                Regime::Dynamic(vec![rule(0, "L0", 0.0, 1)]),
                "time point 2 has 0",
            ),
            (
                Regime::Dynamic(vec![
                    rule(0, "L0", 0.0, 1),
                    rule(0, "L0", 1.0, 1),
                    rule(1, "L1", 0.0, 1),
                ]),
                "time point 1 has 2",
            ),
            (
                Regime::Dynamic(vec![
                    rule(0, "L0", 0.0, 1),
                    rule(1, "L1", 0.0, 1),
                    rule(5, "L1", 0.0, 1),
                ]),
                "outside time points",
            ),
            (
                Regime::Dynamic(vec![rule(0, "Lx", 0.0, 1), rule(1, "L1", 0.0, 1)]),
                "available: L0",
            ),
            (
                Regime::Dynamic(vec![rule(0, "L1", 0.0, 1), rule(1, "L1", 0.0, 1)]),
                "not a covariate observed at or before time point 1",
            ),
            (
                Regime::Dynamic(vec![rule(0, "L0", f64::NAN, 1), rule(1, "L1", 0.0, 1)]),
                "threshold must be finite",
            ),
            (
                Regime::Dynamic(vec![rule(0, "L0", 0.0, 2), rule(1, "L1", 0.0, 1)]),
                "above must be 0 or 1",
            ),
        ];
        for (regime, fragment) in cases {
            let mut c = LtmleConfig::default();
            c.intervention = regime.clone();
            let e = run_ltmle(&data, c).unwrap_err().to_string();
            assert!(e.contains(fragment), "{regime:?}: {e}");
        }
        let e = Regime::from_kind(RegimeKind::Dynamic, None, Some(0.0))
            .unwrap_err()
            .to_string();
        assert!(e.contains("needs dynamic_columns"), "{e}");
        let e = Regime::from_kind(RegimeKind::Always, Some(&["L0".to_string()][..]), None)
            .unwrap_err()
            .to_string();
        assert!(e.contains("only used with regime 'dynamic'"), "{e}");
        assert_eq!(
            Regime::from_kind(RegimeKind::Never, None, None).unwrap(),
            Regime::never()
        );
    }

    #[test]
    fn test_regime_kind_parses_aliases_and_rejects_unknown() {
        assert!(RegimeKind::choices().validate().is_ok());
        assert_eq!(
            "Always-Treat".parse::<RegimeKind>().unwrap(),
            RegimeKind::Always
        );
        assert_eq!("never".parse::<RegimeKind>().unwrap(), RegimeKind::Never);
        assert_eq!("CONTROL".parse::<RegimeKind>().unwrap(), RegimeKind::Never);
        assert_eq!(
            "dynamic".parse::<RegimeKind>().unwrap(),
            RegimeKind::Dynamic
        );
        assert_eq!(RegimeKind::Dynamic.to_string(), "dynamic");
        let e = "sometimes".parse::<RegimeKind>().unwrap_err().to_string();
        assert!(e.contains("always") && e.contains("dynamic"), "{e}");
    }

    #[test]
    fn test_ltmle_q_model_parses_aliases_and_rejects_unknown() {
        assert!(LtmleQModel::choices().validate().is_ok());
        assert_eq!(
            "Logit".parse::<LtmleQModel>().unwrap(),
            LtmleQModel::Logistic
        );
        assert_eq!(
            "quasibinomial".parse::<LtmleQModel>().unwrap(),
            LtmleQModel::Logistic
        );
        assert_eq!(
            "Gaussian".parse::<LtmleQModel>().unwrap(),
            LtmleQModel::Linear
        );
        assert_eq!(
            "linear".parse::<LtmleQModel>().unwrap(),
            LtmleQModel::Linear
        );
        assert_eq!(LtmleQModel::Logistic.to_string(), "Logistic");
        let e = "probit".parse::<LtmleQModel>().unwrap_err().to_string();
        assert!(e.contains("logistic") && e.contains("linear"), "{e}");
    }

    #[test]
    fn test_ltmle_display_reports_regimes_and_notes() {
        let data = two_period(&sim());
        let out = ltmle(&data).unwrap().to_string();
        for s in [
            "E[Y^{always treat}]",
            "E[Y^{never treat}]",
            "Contrast: always treat - never treat",
            "Cumulative g bounds: [0.0100, 1.0000]",
            "Standard errors: influence curve (ltmle variance.method = \"ic\")",
            "Model fits: 9 iterative fits, all converged",
        ] {
            assert!(out.contains(s), "missing {s:?} in\n{out}");
        }
        let mut c = LtmleConfig::default();
        c.intervention = dynamic_two();
        let out = run_ltmle(&data, c).unwrap().to_string();
        assert!(
            out.contains("E[Y^{dynamic (A1 = 1{L0 > 0}, A2 = 1{L1 > 0})}]"),
            "{out}"
        );
        let mut c = LtmleConfig::default();
        c.contrast = None;
        let out = run_ltmle(&data, c).unwrap().to_string();
        assert!(!out.contains("Contrast:"), "{out}");
        assert!(!out.contains("Saturation"), "{out}");
        let mut d2 = data.clone();
        d2.outcomes[0] = d2.outcomes[1].clone();
        let out = ltmle(&d2).unwrap().to_string();
        assert!(
            out.lines()
                .any(|l| l.starts_with("Warning: LTMLE: outcomes at time points 1..1 are not used")),
            "{out}"
        );
    }

    #[test]
    fn test_ltmle_from_dataset_refuses_overlapping_roles() {
        let n = 24;
        let noise = |i: usize, s: f64| ((i as f64 * s).sin() * 1000.0).fract();
        let ds = crate::data::Dataset::new(
            polars::df! {
                "Y" => (0..n).map(|i| 1.0 + noise(i, 1.7)).collect::<Vec<f64>>(),
                "A1" => (0..n).map(|i| (i % 2) as f64).collect::<Vec<f64>>(),
                "A2" => (0..n).map(|i| ((i / 2) % 2) as f64).collect::<Vec<f64>>(),
                "L0" => (0..n).map(|i| noise(i, 2.3)).collect::<Vec<f64>>(),
                "L1" => (0..n).map(|i| noise(i, 3.1)).collect::<Vec<f64>>(),
            }
            .unwrap(),
        );
        let msg = |r: EconResult<LtmleData>| match r {
            Err(EconError::InvalidSpecification { message }) => message,
            other => panic!("expected InvalidSpecification, got {other:?}"),
        };
        let m = msg(LtmleData::from_dataset(
            &ds,
            "Y",
            &["A1", "A2"],
            &[&["L0"], &["L1", "Y"]],
        ));
        assert_eq!(m, "covariate 'Y' at time point 2 is the outcome");
        let m = msg(LtmleData::from_dataset(
            &ds,
            "Y",
            &["A1", "A2"],
            &[&["L0"], &["A2"]],
        ));
        assert!(
            m.starts_with("covariate 'A2' at time point 2 is the treatment of time point 2"),
            "{m}"
        );
        let m = msg(LtmleData::from_dataset(
            &ds,
            "Y",
            &["A1", "A2"],
            &[&["A2"], &["L1"]],
        ));
        assert!(
            m.starts_with("covariate 'A2' at time point 1 is the treatment"),
            "{m}"
        );
        let m = msg(LtmleData::from_dataset(
            &ds,
            "A2",
            &["A1", "A2"],
            &[&["L0"], &["L1"]],
        ));
        assert_eq!(m, "the outcome 'A2' is also the treatment of time point 2");
        // a treatment is a covariate of a later time point
        assert!(
            LtmleData::from_dataset(&ds, "Y", &["A1", "A2"], &[&["L0"], &["L1", "A1"]]).is_ok()
        );
    }

    #[test]
    fn test_ltmle_saturation_is_reported_and_warned() {
        // Few followers of one regime at the last node: ltmle's positivity check fires.
        for (data, few) in [
            (shifted_outcome((5000, 10, 4), 3.0, false, 2.0), 1),
            (sparse_followers(2500, 8, 20), 1),
        ] {
            let r = ltmle(&data).unwrap();
            let s = &r.saturation;
            assert!(s.flagged() && s.positivity(), "{s:?}");
            assert_eq!(s.share_below_gbound.len(), 2);
            assert!(s.share_below_gbound[few] > 0.01, "{s:?}");
            assert!(
                r.warnings
                    .iter()
                    .any(|w| w.starts_with("LTMLE: positivity:")),
                "{:?}",
                r.warnings
            );
            let out = r.to_string();
            assert!(out.contains("Saturation"), "{out}");
            assert!(out.contains("followers at the last node"), "{out}");
        }
        // Six nodes at n = 300: ltmle 1.3-0 warns, with 10 and 137 subjects below the bound.
        let r = ltmle(&long_horizon(300, 6, 17)).unwrap();
        assert!(r.saturation.positivity(), "{:?}", r.saturation);
        let below: Vec<f64> = r
            .saturation
            .share_below_gbound
            .iter()
            .map(|s| (s * 300.0).round())
            .collect();
        assert_eq!(below, [10.0, 137.0]);
    }

    #[test]
    fn test_ltmle_linear_q_outcome_off_unit_interval_is_not_saturation() {
        // The linear Q model keeps Q* on the outcome's own scale (Yc runs from about -3 to 9),
        // where 0 and 1 are not bounds.
        let data = case_data(&sim(), "Yc", &["A1", "A2"], &[&["L0"], &["L1"]]);
        let y = &data.outcomes[1];
        assert!(y.iter().filter(|v| **v > 1.0 || **v < 0.0).count() > y.len() / 2);
        let mut c = LtmleConfig::default();
        c.q_model = LtmleQModel::Linear;
        let r = run_ltmle(&data, c).unwrap();
        assert!(!r.saturation.flagged(), "{:?}", r.saturation);
        assert!(r.saturation.share_q_star_at_bound.iter().all(|q| *q == 0.0));
        assert!(
            !r.warnings.iter().any(|w| w.contains("Q* is within")),
            "{:?}",
            r.warnings
        );
        assert!(!r.to_string().contains("Saturation"), "{r}");
    }

    #[test]
    fn test_ltmle_propensity_at_bound_warning_names_the_treatment() {
        // A treatment separated by its own covariate: its g fit reaches 0 and 1. The warning names
        // the user's column, not a positional A<t>.
        let base = long_horizon(400, 3, 7);
        let mut treatments = base.treatments.clone();
        treatments[1] = base.covariates[1]
            .column(0)
            .mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
        let names: Vec<String> = ["dose_a", "dose_b", "dose_c"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let covariate_names: Vec<Vec<String>> = base
            .covariates
            .iter()
            .enumerate()
            .map(|(t, l)| (0..l.ncols()).map(|j| format!("L{t}_{j}")).collect())
            .collect();
        let data = LtmleData::new(base.outcomes.clone(), treatments, base.covariates.clone())
            .unwrap()
            .with_names(names, covariate_names)
            .unwrap();
        let r = ltmle(&data).unwrap();
        assert!(
            r.saturation.g_at_bound_nodes.contains(&1),
            "{:?}",
            r.saturation
        );
        let w = r
            .warnings
            .iter()
            .find(|w| w.starts_with("LTMLE: fitted propensities numerically 0 or 1 at"))
            .unwrap_or_else(|| panic!("{:?}", r.warnings));
        assert!(w.contains("dose_b"), "{w}");
        assert!(!w.contains("A2"), "{w}");
    }

    #[test]
    fn test_ltmle_no_saturation_on_regular_data() {
        // ltmle 1.3-0 gives no positivity or rare-events warning on these two
        for data in [two_period(&sim()), long_horizon(600, 3, 5)] {
            let r = ltmle(&data).unwrap();
            assert!(!r.saturation.flagged(), "{:?}", r.saturation);
            assert!(!r.saturation.rare_outcome);
            assert!(
                !r.warnings
                    .iter()
                    .any(|w| w.starts_with("LTMLE: positivity")),
                "{:?}",
                r.warnings
            );
            assert!(!r.to_string().contains("Saturation"));
        }
    }

    #[test]
    fn test_ltmle_rare_outcome_is_warned() {
        use rand::{Rng, SeedableRng};
        let mut data = two_period(&sim());
        let n = data.n_obs();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(29);
        let y: Array1<f64> = (0..n)
            .map(|_| f64::from(u8::from(rng.gen_range(0.0..1.0) < 0.02)))
            .collect();
        let mean = y.sum() / n as f64;
        assert!(mean > 0.0 && mean < 0.03, "{mean}");
        data.outcomes[1] = y;
        let r = ltmle(&data).unwrap();
        assert!(r.saturation.rare_outcome && r.saturation.flagged());
        assert!(
            r.warnings
                .iter()
                .any(|w| w.starts_with("LTMLE: rare outcome (mean")),
            "{:?}",
            r.warnings
        );
        assert!(r.to_string().contains("rare outcome (mean below 0.03)"));
    }

    #[test]
    fn test_validate_ltmle_duplicated_history_column_is_aliased() {
        let (case, ds) = crate::test_support::load_case("ltmle_two_period");
        let data =
            LtmleData::from_dataset(&ds, "Y", &["A1", "A2"], &[&["L0"], &["L0", "L1"]]).unwrap();
        let r = run_ltmle(&data, tight()).unwrap();
        Check::new(&case)
            .scalar("psi_always", r.psi_treated, Tol::coef(r.n_obs))
            .finish();
        assert!(
            r.warnings.iter().any(|w| w.starts_with("g model for A2")
                && w.contains("not defined because of singularities: L0")),
            "{:?}",
            r.warnings
        );
    }

    #[test]
    fn test_ltmle_covariate_scale_invariance() {
        let data = two_period(&sim());
        let base = ltmle(&data).unwrap();
        for scale in [1e-6, 1e6] {
            let scaled = LtmleData::new(
                data.outcomes.clone(),
                data.treatments.clone(),
                data.covariates.iter().map(|l| l * scale).collect(),
            )
            .unwrap();
            let r = ltmle(&scaled).unwrap();
            let tol = Tol::new(1e-10, 0.0);
            assert!(
                tol.allows(r.psi_treated, base.psi_treated),
                "scale {scale}: {} vs {}",
                r.psi_treated,
                base.psi_treated
            );
            assert!(
                tol.allows(r.psi_control, base.psi_control),
                "scale {scale}: {} vs {}",
                r.psi_control,
                base.psi_control
            );
            assert!(r.converged, "scale {scale}: {:?}", r.warnings);
        }
    }

    #[test]
    fn test_ltmle_intermediate_outcomes_warn() {
        let data = two_period(&sim());
        let base = ltmle(&data).unwrap();
        let mut d2 = data.clone();
        d2.outcomes[0] = d2.outcomes[1].clone();
        let r = ltmle(&d2).unwrap();
        assert!(
            r.warnings.iter().any(|w| w.starts_with(
                "LTMLE: outcomes at time points 1..1 are not used; p2a's LTMLE has a single \
                 outcome node (time point 2)"
            )),
            "{:?}",
            r.warnings
        );
        assert_eq!(r.psi_treated.to_bits(), base.psi_treated.to_bits());
        assert_eq!(r.psi_control.to_bits(), base.psi_control.to_bits());
    }

    #[test]
    fn test_ltmle_data_validation() {
        let n = 10;

        // Valid data
        let outcomes = vec![Array1::zeros(n), Array1::zeros(n)];
        let treatments = vec![Array1::zeros(n), Array1::zeros(n)];
        let covariates = vec![Array2::zeros((n, 2)), Array2::zeros((n, 2))];

        let data = LtmleData::new(outcomes, treatments, covariates);
        assert!(data.is_ok());

        let data = data.unwrap();
        assert_eq!(data.time_points(), 2);
        assert_eq!(data.n_obs(), 10);
    }

    #[test]
    fn test_ltmle_data_validation_errors() {
        let n = 10;

        // Mismatched time points
        let outcomes = vec![Array1::zeros(n), Array1::zeros(n)];
        let treatments = vec![Array1::zeros(n)]; // Only 1 time point
        let covariates = vec![Array2::zeros((n, 2)), Array2::zeros((n, 2))];

        let result = LtmleData::new(outcomes, treatments, covariates);
        assert!(result.is_err());

        // Non-binary treatment
        let outcomes = vec![Array1::zeros(n)];
        let treatments = vec![Array1::from_vec(vec![0.5; n])]; // Not binary
        let covariates = vec![Array2::zeros((n, 2))];

        let result = LtmleData::new(outcomes, treatments, covariates);
        assert!(result.is_err());
    }

    #[test]
    fn test_intervention_type_display() {
        assert_eq!(
            format!("{}", InterventionType::AlwaysTreat),
            "Always Treat (A=1)"
        );
        assert_eq!(
            format!("{}", InterventionType::NeverTreat),
            "Never Treat (A=0)"
        );
        assert_eq!(
            format!("{}", InterventionType::Static(true)),
            "Static (A=1)"
        );
        assert_eq!(
            format!("{}", InterventionType::Static(false)),
            "Static (A=0)"
        );
    }

    /// `t_max` nodes with two noisy covariates each; the treatments follow the covariates and
    /// the previous treatment, the outcome is Bernoulli in the covariates and treatments.
    fn long_horizon(n: usize, t_max: usize, seed: u64) -> LtmleData {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let (mut ls, mut az) = (Vec::new(), Vec::new());
        let mut prev = Array1::<f64>::zeros(n);
        for _ in 0..t_max {
            let x = Array2::from_shape_fn((n, 2), |_| rng.gen_range(-1.0..1.0));
            let a: Array1<f64> = (0..n)
                .map(|i| {
                    let p = crate::econometrics::glm::plogis(
                        0.3 * x[[i, 0]] + 0.2 * prev[i] - 0.1 * x[[i, 1]],
                    );
                    if rng.gen_range(0.0..1.0) < p {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            prev = a.clone();
            ls.push(x);
            az.push(a);
        }
        let y: Array1<f64> = (0..n)
            .map(|i| {
                let mut v = 0.2 * ls[0][[i, 0]] - 0.3 * ls[t_max - 1][[i, 1]];
                for a in &az {
                    v += 0.4 * (a[i] - 0.5);
                }
                let p = crate::econometrics::glm::plogis(v);
                if rng.gen_range(0.0..1.0) < p {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let mut outcomes = vec![Array1::zeros(n); t_max - 1];
        outcomes.push(y);
        LtmleData::new(outcomes, az, ls).unwrap()
    }

    #[test]
    fn test_ltmle_earlier_q_fits_start_from_the_later_node() {
        // A Q regression at node t has as its response the targeted prediction of node t + 1,
        // whose coefficients on the shared history are a close start: IRLS needs far fewer
        // steps there than from glm.fit's start.
        let data = long_horizon(2000, 6, 3);
        let q_steps = |warm: bool| -> usize {
            let r = run_ltmle_with(&data, LtmleConfig::default(), FitPaths { lean: true, warm })
                .unwrap();
            assert!(r.converged, "{:?}", r.warnings);
            std::iter::once(&r.regime_estimate)
                .chain(r.contrast_estimate.iter())
                .flat_map(|e| e.q_models[..5].iter().map(|m| m.iterations))
                .sum()
        };
        let (cold, warm) = (q_steps(false), q_steps(true));
        assert!(3 * warm <= 2 * cold, "warm {warm} steps, cold {cold} steps");
    }

    /// The wave-2B harness's T-node design with its Bernoulli outcome: a few followers of each
    /// regime at the last node, so that some seeds saturate one regime's targeted predictions.
    fn sparse_followers(n: usize, t_max: usize, seed: u64) -> LtmleData {
        use rand::{Rng, SeedableRng};
        let plogis = crate::econometrics::glm::plogis;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let (mut ls, mut az) = (Vec::new(), Vec::new());
        let mut prev = Array1::<f64>::zeros(n);
        for _ in 0..t_max {
            let x = Array2::from_shape_fn((n, 2), |_| rng.gen_range(-1.0..1.0));
            let a: Array1<f64> = (0..n)
                .map(|i| {
                    let p = plogis(0.3 * x[[i, 0]] + 0.2 * prev[i]);
                    f64::from(u8::from(rng.gen_range(0.0..1.0) < p))
                })
                .collect();
            prev = a.clone();
            ls.push(x);
            az.push(a);
        }
        let latent: Array1<f64> = (0..n)
            .map(|i| {
                let mut v = 1.0 + 0.2 * ls[0][[i, 0]] + rng.gen_range(-0.5..0.5);
                for (t, a) in az.iter().enumerate() {
                    v += (0.5 - 0.2 * t as f64 / t_max as f64) * a[i];
                }
                v
            })
            .collect();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        let mean = latent.mean().unwrap();
        let y = latent.mapv(|v| f64::from(u8::from(rng.gen_range(0.0..1.0) < plogis(v - mean))));
        let mut outcomes = vec![Array1::zeros(n); t_max - 1];
        outcomes.push(y);
        LtmleData::new(outcomes, az, ls).unwrap()
    }

    /// Each number within `tol` of its reference, relative to the largest entry of the reference
    /// (a coefficient vector, one fit's EIC): `a` and `b` of length 1 are held relative to `b`.
    fn assert_relative(what: &str, a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len(), "{what}");
        let scale = b
            .iter()
            .filter(|v| v.is_finite())
            .fold(0.0f64, |m, v| m.max(v.abs()));
        for (x, y) in a.iter().zip(b) {
            assert!(
                x == y || (x.is_nan() && y.is_nan()) || (x - y).abs() <= tol * scale,
                "{what}: fast {x} vs reference {y}, {:e} relative to {scale}",
                (x - y).abs() / scale
            );
        }
    }

    /// Every number the fast paths could move, against the shared GLM from glm.fit's start:
    /// `estimand_tol` for the estimates and their standard errors, `nuisance_tol` for the
    /// coefficients, predictions, fluctuations and the EIC. The bounds passed in hold on each
    /// test's own data sets and are not general bounds; the gaps measured on a wider grid are in
    /// validation/econometrics/ltmle.md, "Nuisance fits".
    fn assert_same_as_shared_glm(
        data: &LtmleData,
        config: &LtmleConfig,
        estimand_tol: f64,
        nuisance_tol: f64,
    ) {
        let fast = run_ltmle(data, config.clone()).unwrap();
        let reference = run_ltmle_with(
            data,
            config.clone(),
            FitPaths {
                lean: false,
                warm: false,
            },
        )
        .unwrap();
        let estimand = |what: &str, a: f64, b: f64| assert_relative(what, &[a], &[b], estimand_tol);
        let nuisance = |what: &str, a: &[f64], b: &[f64]| assert_relative(what, a, b, nuisance_tol);
        estimand("ate", fast.ate, reference.ate);
        estimand("se", fast.se, reference.se);
        estimand("z", fast.z_stat, reference.z_stat);
        nuisance("p", &[fast.p_value], &[reference.p_value]);
        nuisance("eic", &fast.influence_curve, &reference.influence_curve);
        assert_eq!(fast.n_model_fits, reference.n_model_fits);
        // At a tolerance of rounding size the deviance rule itself is decided by rounding.
        if config.tolerance >= 1e-10 {
            assert_eq!(
                (fast.converged, &fast.warnings),
                (reference.converged, &reference.warnings)
            );
        }
        for (a, b) in fast.g_models.iter().zip(&reference.g_models) {
            nuisance("g coefficients", &a.coefficients, &b.coefficients);
        }
        for (a, b) in fast
            .propensity_scores
            .iter()
            .zip(&reference.propensity_scores)
        {
            nuisance("propensities", a, b);
        }
        let estimates = |r: &LtmleResult| {
            std::iter::once(r.regime_estimate.clone())
                .chain(r.contrast_estimate.clone())
                .collect::<Vec<_>>()
        };
        for (a, b) in estimates(&fast).iter().zip(&estimates(&reference)) {
            estimand("psi", a.psi, b.psi);
            estimand("psi se", a.se, b.se);
            nuisance("eps", &a.fluctuation_coefs, &b.fluctuation_coefs);
            assert_eq!(a.n_following, b.n_following);
            assert_eq!(a.n_bounded, b.n_bounded);
            assert_eq!(a.n_bounded_following, b.n_bounded_following);
            for t in 0..a.q_models.len() {
                nuisance(
                    "Q coefficients",
                    &a.q_models[t].coefficients,
                    &b.q_models[t].coefficients,
                );
                nuisance("Q*", &a.targeted_predictions[t], &b.targeted_predictions[t]);
                nuisance("cum g", &a.cumulative_g[t], &b.cumulative_g[t]);
                nuisance("H", &a.clever_covariates[t], &b.clever_covariates[t]);
            }
        }
    }

    /// The default deviance rule stops each fit a little short of the maximum-likelihood
    /// estimate, at a point that depends on its start. This bound holds on the data sets of the
    /// tests that use it; test_ltmle_fast_paths_hold_the_gaps_measured_off_the_fixtures has the
    /// data sets where it does not.
    const DEFAULT_CONTROL_NUISANCE: f64 = 1e-9;

    #[test]
    fn test_ltmle_fast_paths_match_the_shared_glm() {
        // Long histories (T = 10: the treatment columns correlate with the intercept and the
        // normal equations refine), short ones, default and tight control.
        for (t_max, n) in [(2, 400), (3, 500), (5, 800), (8, 1500), (10, 2500)] {
            let data = long_horizon(n, t_max, 11 + t_max as u64);
            assert_same_as_shared_glm(
                &data,
                &LtmleConfig::default(),
                1e-10,
                DEFAULT_CONTROL_NUISANCE,
            );
            assert_same_as_shared_glm(&data, &tight(), 1e-10, 1e-10);
        }
    }

    #[test]
    fn test_ltmle_fast_paths_match_on_other_regimes() {
        let data = long_horizon(1200, 4, 5);
        let names: Vec<String> = (1..=4).map(|t| format!("L{t}_1")).collect();
        let mut c = tight();
        c.intervention = Regime::threshold_rules(&names, 0.0);
        c.contrast = Some(Regime::Static(vec![1, 0, 1, 0]));
        assert_same_as_shared_glm(&data, &c, 1e-10, 1e-10);
        c.contrast = None;
        c.intervention = Regime::Static(vec![0, 1, 1, 0]);
        let fast = run_ltmle(&data, c.clone()).unwrap();
        let reference = run_ltmle_with(
            &data,
            c,
            FitPaths {
                lean: false,
                warm: false,
            },
        )
        .unwrap();
        assert_relative(
            "psi",
            &[fast.regime_estimate.psi],
            &[reference.regime_estimate.psi],
            1e-10,
        );
    }

    #[test]
    fn test_ltmle_fast_paths_keep_the_outcome_at_a_low_max_iter() {
        // max_iter decides what is reported as not converged, on the step count from glm.fit's
        // start, which a warm start would undercut: below WARM_MIN_MAX_ITER the fits keep it.
        let data = long_horizon(1500, 6, 8);
        let mut reported = Vec::new();
        for max_iter in [2, 3, 4, 5, 6, WARM_MIN_MAX_ITER - 1] {
            let mut c = LtmleConfig::default();
            c.max_iter = max_iter;
            assert_same_as_shared_glm(&data, &c, 1e-10, 1e-10);
            let fast = run_ltmle(&data, c).unwrap();
            let steps = |e: &RegimeEstimate| e.q_models.iter().map(|m| m.iterations).sum::<usize>();
            let most = fast
                .regime_estimate
                .q_models
                .iter()
                .map(|m| m.iterations)
                .max()
                .unwrap();
            reported.push((max_iter, fast.converged, steps(&fast.regime_estimate), most));
        }
        // the low limits do report non-convergence, and a fit then uses up to the limit
        assert!(
            reported[0..2].iter().all(|r| !r.1 && r.3 == r.0),
            "{reported:?}"
        );
        assert!(reported[reported.len() - 1].1, "{reported:?}");
        let mut c = LtmleConfig::default();
        c.max_iter = WARM_MIN_MAX_ITER;
        assert_same_as_shared_glm(&data, &c, 1e-10, DEFAULT_CONTROL_NUISANCE);
        let warm = run_ltmle(&data, c).unwrap();
        let cold = reported[reported.len() - 1].2;
        let at_limit: usize = warm
            .regime_estimate
            .q_models
            .iter()
            .map(|m| m.iterations)
            .sum();
        assert!(at_limit < cold, "warm {at_limit} steps, cold {cold} steps");
    }

    #[test]
    fn test_ltmle_fast_paths_report_a_saturated_regime_through_the_shared_glm() {
        // An extreme fluctuation at a later node leaves the earlier Q responses of a regime
        // without variation (long_horizon(2500, 10, 21)). ltmle's no-variation rule predicts those
        // as constants before any fit path is chosen, so the fast and cold paths apply it alike; a
        // response that keeps a range of at least 1e-4 with its mean within SATURATED_MEAN of 0 or
        // 1 still goes to the shared GLM. sparse_followers(4000, 10, 3), the timing harness's
        // saturating case, saturated through a last fluctuation that left its score unsolved; the
        // re-solve keeps its regime off 0 and 1 (ltmle 1.3-0: no constant node either).
        for (data, constant) in [
            (long_horizon(2500, 10, 21), true),
            (sparse_followers(4000, 10, 3), false),
        ] {
            assert_same_as_shared_glm(
                &data,
                &LtmleConfig::default(),
                1e-10,
                DEFAULT_CONTROL_NUISANCE,
            );
            assert_same_as_shared_glm(&data, &tight(), 1e-10, 1e-10);
            let r = run_ltmle(&data, LtmleConfig::default()).unwrap();
            let s = &r.saturation;
            assert!(s.flagged(), "{s:?}");
            let has_constant = s.constant_q_nodes.iter().any(|c| !c.is_empty());
            let resolved = s.resolved_targeting.iter().any(|c| !c.is_empty());
            assert_eq!((has_constant, resolved), (constant, !constant), "{s:?}");
            let marker = if constant {
                "(ltmle's no-variation rule)"
            } else {
                "re-solved with offsets bounded to [1e-8, 1 - 1e-8]"
            };
            assert!(
                r.warnings.iter().any(|w| w.ends_with(marker)),
                "{:?}",
                r.warnings
            );
        }
    }

    /// `t_max` nodes where `strong` sets the confounding and `shift` the outcome's level;
    /// `frac` keeps the outcome's probability as a fractional outcome instead of drawing from it.
    fn shifted_outcome(
        (n, t_max, seed): (usize, usize, u64),
        shift: f64,
        frac: bool,
        strong: f64,
    ) -> LtmleData {
        use rand::{Rng, SeedableRng};
        let plogis = crate::econometrics::glm::plogis;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let (mut ls, mut az) = (Vec::new(), Vec::new());
        let mut prev = Array1::<f64>::zeros(n);
        for _ in 0..t_max {
            let x = Array2::from_shape_fn((n, 2), |_| rng.gen_range(-1.0..1.0));
            let a: Array1<f64> = (0..n)
                .map(|i| {
                    let p = plogis(strong * x[[i, 0]] + 0.2 * prev[i]);
                    f64::from(u8::from(rng.gen_range(0.0..1.0) < p))
                })
                .collect();
            prev = a.clone();
            ls.push(x);
            az.push(a);
        }
        let y: Array1<f64> = (0..n)
            .map(|i| {
                let mut v = shift + 0.3 * ls[0][[i, 0]] + rng.gen_range(-0.5..0.5);
                for a in &az {
                    v += 0.4 * a[i];
                }
                let p = plogis(v);
                if frac {
                    p
                } else {
                    f64::from(u8::from(rng.gen_range(0.0..1.0) < p))
                }
            })
            .collect();
        let mut outcomes = vec![Array1::zeros(n); t_max - 1];
        outcomes.push(y);
        LtmleData::new(outcomes, az, ls).unwrap()
    }

    #[test]
    fn test_ltmle_fast_paths_hold_the_gaps_measured_off_the_fixtures() {
        // The largest gaps of the sweep in validation/econometrics/ltmle.md, "Nuisance fits", at
        // the default tolerance. Each data set has the bounds it meets, above the tests' usual
        // 1e-10 and 1e-9, and its warnings and convergence status stay equal.
        let shifted = |s, shift, strong| shifted_outcome(s, shift, false, strong);
        for (data, estimand_tol, nuisance_tol) in [
            (shifted((300, 3, 1), 1.5, 1.2), 1e-9, 2e-8),
            (shifted((800, 8, 3), 3.0, 2.0), 1e-9, 1e-8),
            (shifted((300, 3, 2), -5.0, 0.3), 1e-10, 2e-8),
            (shifted((150, 4, 3), 1.5, 1.2), 1e-9, 2e-8),
            (shifted((150, 4, 2), 3.0, 2.0), 1e-9, 1e-8),
            (shifted((300, 3, 4), -5.0, 0.3), 1e-9, 2e-7),
            (long_horizon(300, 6, 17), 5e-10, 1e-9),
        ] {
            assert_same_as_shared_glm(&data, &LtmleConfig::default(), estimand_tol, nuisance_tol);
        }

        // There the warm start causes the gap: without it the numbers keep the shared GLM's.
        let lean = FitPaths {
            lean: true,
            warm: false,
        };
        let cold = FitPaths {
            lean: false,
            warm: false,
        };
        let warm_case = shifted((300, 3, 1), 1.5, 1.2);
        let long_case = long_horizon(300, 6, 17);
        for data in [&warm_case, &long_case] {
            let a = run_ltmle_with(data, LtmleConfig::default(), lean).unwrap();
            let b = run_ltmle_with(data, LtmleConfig::default(), cold).unwrap();
            assert_relative("ate", &[a.ate], &[b.ate], 1e-13);
            assert_relative("se", &[a.se], &[b.se], 1e-13);
            let (x, y) = (&a.regime_estimate, &b.regime_estimate);
            assert_relative("eps", &x.fluctuation_coefs, &y.fluctuation_coefs, 1e-12);

            // and on these two data sets the warm numbers are the closer to the tight reference.
            // The distances are the deviance rule's slack, at least ten times apart and well above
            // rounding; across data sets either path can be the closer.
            let fast = run_ltmle(data, LtmleConfig::default()).unwrap();
            let tight_reference = run_ltmle_with(data, tight(), cold).unwrap();
            let distance = |r: &LtmleResult| (r.ate - tight_reference.ate).abs();
            assert!(distance(&fast) < distance(&b));
        }

        // An always-near-0 regime: most of its fits run out of steps on both paths and psi is about
        // 1e-13. The warm start leaves its numbers bit for bit the same. The ATE's gap stays at
        // rounding.
        let near_zero = long_horizon(1200, 8, 21);
        assert_same_as_shared_glm(&near_zero, &LtmleConfig::default(), 1e-6, 1e-6);
        let a = run_ltmle(&near_zero, LtmleConfig::default()).unwrap();
        let b = run_ltmle_with(&near_zero, LtmleConfig::default(), lean).unwrap();
        let c = run_ltmle_with(&near_zero, LtmleConfig::default(), cold).unwrap();
        let near = |r: &LtmleResult| r.contrast_estimate.clone().unwrap();
        assert!(near(&c).psi < 1e-12, "{}", near(&c).psi);
        assert_eq!(near(&a).psi.to_bits(), near(&b).psi.to_bits());
        assert_eq!(near(&a).fluctuation_coefs, near(&b).fluctuation_coefs);
        assert_relative("ate", &[a.ate], &[c.ate], 1e-13);
    }

    /// Every regime's estimate of a result, the intervention first.
    fn regime_estimates(r: &LtmleResult) -> Vec<&RegimeEstimate> {
        std::iter::once(&r.regime_estimate)
            .chain(r.contrast_estimate.iter())
            .collect()
    }

    /// The node's Q response: the outcome at the last node, else the later node's Q*.
    fn q_response(e: &RegimeEstimate, y: &Array1<f64>, t: usize) -> Vec<f64> {
        if t + 1 == e.targeted_predictions.len() {
            y.to_vec()
        } else {
            e.targeted_predictions[t + 1].clone()
        }
    }

    #[test]
    fn test_ltmle_constant_q_response_is_not_regressed() {
        // The never regime of sparse_followers(1200, 10, 12) has one follower at the last node:
        // the targeting there leaves every earlier Q response without variation, and ltmle
        // predicts each as a constant instead of regressing it.
        let data = sparse_followers(1200, 10, 12);
        let y = data.outcomes[9].clone();
        let r = ltmle(&data).unwrap();
        let never = r.contrast_estimate.as_ref().unwrap();
        let constant = &r.saturation.constant_q_nodes[1];
        assert!(!constant.is_empty(), "{:?}", r.saturation);
        assert!(
            !constant.contains(&9),
            "this fixture's 0/1 outcome is never constant at the last node (a constant one is \
             refused)"
        );
        for &t in constant {
            let m = &never.q_models[t];
            let response = q_response(never, &y, t);
            let (lo, hi) = response
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                    (a.min(v), b.max(v))
                });
            assert!(hi - lo < 1e-4, "node {t}: range {}", hi - lo);
            assert_eq!(m.constant_prediction, Some(hi), "node {t}");
            assert_eq!(m.iterations, 0);
            assert!(m.converged);
            let c = hi.clamp(1e-4, 0.9999);
            let logit = (c / (1.0 - c)).ln();
            assert_eq!(m.coefficients[0], logit);
            assert!(m.coefficients[1..].iter().all(|b| b.is_nan()));
            // Q*_t of every subject is the constant's logit moved by the node's fluctuation
            let eps = never.fluctuation_coefs[t];
            for q in &never.targeted_predictions[t] {
                assert_eq!(*q, linkinv(logit + eps), "node {t}");
            }
            assert!(
                r.warnings.iter().any(|w| w.starts_with(&format!(
                    "LTMLE: never treat: Q response at A{} has range",
                    t + 1
                )) && w.ends_with("(ltmle's no-variation rule)")),
                "{:?}",
                r.warnings
            );
        }
        // the rule sits before the choice of fit path, so the cold path applies it alike
        let cold = run_ltmle_with(
            &data,
            LtmleConfig::default(),
            FitPaths {
                lean: false,
                warm: false,
            },
        )
        .unwrap();
        assert_eq!(
            cold.saturation.constant_q_nodes,
            r.saturation.constant_q_nodes
        );
        let cold_never = cold.contrast_estimate.as_ref().unwrap();
        // Not bitwise: each constant is the maximum of a later node's Q*, and the two paths fit
        // the later nodes within the shared GLM's tolerances (assert_same_as_shared_glm), so the
        // responses, and their maxima, differ in the last digits.
        for &t in constant {
            let (a, b) = (
                never.q_models[t].constant_prediction.unwrap(),
                cold_never.q_models[t].constant_prediction.unwrap(),
            );
            assert!(
                (a - b).abs() <= 1e-9 * b.abs().max(1e-300),
                "node {t}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_ltmle_constant_outcome_is_refused() {
        let mut data = long_horizon(300, 3, 5);
        for v in [0.0, 1.0, 0.25] {
            data.outcomes[2] = Array1::from_elem(300, v);
            let err = ltmle(&data).unwrap_err();
            assert!(
                matches!(err, EconError::InvalidSpecification { ref message } if message.contains("constant")),
                "{err:?}"
            );
        }
    }

    /// ltmle_saturation.csv as LtmleData.
    fn saturation_case_data(ds: &crate::data::Dataset) -> LtmleData {
        let a: Vec<String> = (1..=8).map(|t| format!("A{t}")).collect();
        let l: Vec<[String; 2]> = (1..=8)
            .map(|t| [format!("L{t}a"), format!("L{t}b")])
            .collect();
        let a_ref: Vec<&str> = a.iter().map(String::as_str).collect();
        let l_ref: Vec<Vec<&str>> = l
            .iter()
            .map(|p| vec![p[0].as_str(), p[1].as_str()])
            .collect();
        let l_sl: Vec<&[&str]> = l_ref.iter().map(Vec::as_slice).collect();
        case_data(ds, "Y", &a_ref, &l_sl)
    }

    #[test]
    fn test_validate_ltmle_saturation_matches_ltmle() {
        // ltmle::ltmle 1.3-0 itself is the reference (case ltmle_saturation): its glms stop at
        // epsilon 1e-8, so its numbers carry that slack. Quantities off 0 and 1 are held to
        // 1e-6 relative; those within 1e-6 of 0 (the never regime's mean and SE, about 4e-11
        // and 5e-12) to 1e-8 absolute: they are the distance of a clamped prediction from 0,
        // which ltmle's nlminb-free path and p2a reach by different rounding.
        let (case, ds) = crate::test_support::load_case("ltmle_saturation");
        let r = ltmle(&saturation_case_data(&ds)).unwrap();
        let near_zero = Tol::new(0.0, 1e-8);
        let indicator = |nodes: &[usize]| -> Vec<f64> {
            (0..8)
                .map(|t| f64::from(u8::from(nodes.contains(&t))))
                .collect()
        };
        let resolved_nodes = |k: usize| -> Vec<usize> {
            r.saturation.resolved_targeting[k]
                .iter()
                .map(|(t, _)| *t)
                .collect()
        };
        Check::new(&case)
            .scalar("psi_always_sat", r.psi_treated, Tol::iterative(1e-6))
            .scalar("se_always_sat", r.psi_treated_se, Tol::iterative(1e-6))
            .scalar("psi_never_sat", r.psi_control, near_zero)
            .scalar("se_never_sat", r.psi_control_se, near_zero)
            .scalar("ate_sat", r.ate, Tol::iterative(1e-6))
            .scalar("ate_se_sat", r.se, Tol::iterative(1e-6))
            .vector(
                "constant_q_always",
                &indicator(&r.saturation.constant_q_nodes[0]),
                Tol::exact(),
            )
            .vector(
                "constant_q_never",
                &indicator(&r.saturation.constant_q_nodes[1]),
                Tol::exact(),
            )
            .scalar(
                "positivity_warning",
                f64::from(u8::from(r.saturation.positivity())),
                Tol::exact(),
            )
            .scalar(
                "rare_warning",
                f64::from(u8::from(r.saturation.rare_outcome)),
                Tol::exact(),
            )
            // ltmle's check on one regime alone: that regime's share above 0.01
            .scalar(
                "positivity_warning_always",
                f64::from(u8::from(
                    r.saturation.share_below_gbound[0] > POSITIVITY_SHARE,
                )),
                Tol::exact(),
            )
            .scalar(
                "positivity_warning_never",
                f64::from(u8::from(
                    r.saturation.share_below_gbound[1] > POSITIVITY_SHARE,
                )),
                Tol::exact(),
            )
            // ltmle re-solves no score here; neither may p2a
            .vector(
                "resolved_always",
                &indicator(&resolved_nodes(0)),
                Tol::exact(),
            )
            .vector(
                "resolved_never",
                &indicator(&resolved_nodes(1)),
                Tol::exact(),
            )
            .vector(
                "n_follow_always",
                &counts(&r.regime_estimate.n_following),
                Tol::exact(),
            )
            .vector(
                "n_follow_never",
                &counts(&r.contrast_estimate.as_ref().unwrap().n_following),
                Tol::exact(),
            )
            .finish();
        let below = |k: usize| (r.saturation.share_below_gbound[k] * r.n_obs as f64).round();
        Check::new(&case)
            .scalar("n_below_gbound_always", below(0), Tol::exact())
            .scalar("n_below_gbound_never", below(1), Tol::exact())
            .finish();
    }

    #[test]
    fn test_ltmle_saturating_fixtures_agree_with_ltmle() {
        // Quoted from ltmle 1.3-0 on the exported fixtures (glm, stratify = FALSE,
        // variance.method = "ic"). Before ltmle's no-variation rule the
        // default tolerance gave an ATE of -1 and 0.737 here.
        for (data, always, never) in [
            (
                shifted_outcome((5000, 10, 4), 3.0, false, 2.0),
                0.999_999_999_950_950_8,
                0.999_999_999_943_640_9,
            ),
            (
                sparse_followers(2500, 8, 20),
                0.746_485_824_266_510_2,
                4.028_270_621_033_234e-11,
            ),
            // the timing harness's saturating case: before the score re-solve the never mean
            // was 1 at every tolerance
            (
                sparse_followers(4000, 10, 3),
                0.890_525_501_726_346_2,
                0.494_019_949_650_668_3,
            ),
        ] {
            let mut ates = Vec::new();
            for tol in [1e-8, 1e-10, 1e-14] {
                let mut c = LtmleConfig::default();
                c.tolerance = tol;
                let r = run_ltmle(&data, c).unwrap();
                assert!(
                    (r.psi_treated - always).abs() <= 1e-8,
                    "{tol}: {}",
                    r.psi_treated
                );
                assert!(
                    (r.psi_control - never).abs() <= 1e-8,
                    "{tol}: {}",
                    r.psi_control
                );
                assert!((r.ate - (always - never)).abs() <= 1e-8, "{tol}: {}", r.ate);
                ates.push(r.ate);
            }
            assert!((ates[0] - ates[2]).abs() <= 1e-8, "{ates:?}");
        }
    }

    #[test]
    fn test_ltmle_node_after_a_resolve_matches_the_cold_path() {
        // After a re-solved targeting step the next node takes glm.fit's start rather than a warm
        // start built from the re-solved epsilon; the fast path then agrees with the cold one.
        let data = sparse_followers(4000, 10, 3);
        let fast = run_ltmle(&data, LtmleConfig::default()).unwrap();
        let cold = run_ltmle_with(
            &data,
            LtmleConfig::default(),
            FitPaths {
                lean: false,
                warm: false,
            },
        )
        .unwrap();
        // a re-solved node that is not the first: the backward pass fits node t - 1 next, from
        // glm.fit's start (`t < t_max` alone held for every node)
        assert!(
            fast.saturation
                .resolved_targeting
                .iter()
                .flatten()
                .any(|(t, _)| *t > 0),
            "{:?}",
            fast.saturation.resolved_targeting
        );
        assert_eq!(
            fast.saturation.resolved_targeting,
            cold.saturation.resolved_targeting
        );
        for (f, c) in regime_estimates(&fast)
            .into_iter()
            .zip(regime_estimates(&cold))
        {
            for (a, b) in f.fluctuation_coefs.iter().zip(&c.fluctuation_coefs) {
                assert!((a - b).abs() <= 1e-10, "{}: {a} vs {b}", f.label);
            }
            for (a, b) in f.targeted_predictions.iter().zip(&c.targeted_predictions) {
                for (x, y) in a.iter().zip(b) {
                    assert!((x - y).abs() <= 1e-10, "{}: {x} vs {y}", f.label);
                }
            }
            for (a, b) in f.q_models.iter().zip(&c.q_models) {
                assert!(
                    a.iterations <= b.iterations,
                    "{} node {}: warm {} > cold {}",
                    f.label,
                    a.node,
                    a.iterations,
                    b.iterations
                );
            }
        }
        for (a, b) in [
            (fast.psi_treated, cold.psi_treated),
            (fast.psi_control, cold.psi_control),
            (fast.ate, cold.ate),
        ] {
            assert!((a - b).abs() <= 1e-8, "{a} vs {b}");
        }
    }

    #[test]
    fn test_ltmle_targeting_solves_the_score_at_every_node() {
        // ltmle's criterion: each node's EIC term sums to at most 1e-3 over the subjects. On
        // sparse_followers(4000, 10, 3) the never regime's last fluctuation stopped with the score
        // at -200 before the re-solve.
        let shifted = |s, shift, strong| shifted_outcome(s, shift, false, strong);
        for data in [
            sparse_followers(4000, 10, 3),
            shifted((5000, 10, 4), 3.0, 2.0),
            sparse_followers(2500, 8, 20),
            sparse_followers(1200, 10, 12),
            shifted((150, 4, 2), 3.0, 2.0),
            sparse_followers(150, 8, 5),
            long_horizon(300, 6, 17),
        ] {
            let y = data.outcomes[data.time_points() - 1].clone();
            for tol in [1e-8, 1e-14] {
                let mut c = LtmleConfig::default();
                c.tolerance = tol;
                let r = run_ltmle(&data, c).unwrap();
                for (k, e) in regime_estimates(&r).into_iter().enumerate() {
                    let resolved = &r.saturation.resolved_targeting[k];
                    for t in 0..e.targeted_predictions.len() {
                        let next = q_response(e, &y, t);
                        let (h, q) = (&e.clever_covariates[t], &e.targeted_predictions[t]);
                        let score: f64 = (0..q.len()).map(|i| h[i] * (next[i] - q[i])).sum();
                        let sum_h: f64 = h.iter().sum();
                        let bound = if resolved.iter().any(|(s, _)| *s == t) {
                            1e-10 * sum_h
                        } else {
                            1e-3
                        };
                        assert!(
                            score.abs() <= bound,
                            "{tol}, {}, node {t}: score {score:e}",
                            e.label
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_resolve_fluctuation_monotone_root() {
        let score = |y: &[f64], o: &[f64], h: &[f64], lb: f64, eps: f64| -> f64 {
            let (lo, hi) = offset_bounds(lb);
            (0..y.len())
                .map(|i| h[i] * (y[i] - linkinv(o[i].max(lo).min(hi) + eps)))
                .sum()
        };
        // a root inside the bracket, with a zero-weight row that does not enter
        let y = [0.9, 0.2, 0.7, 0.95, 0.4, 0.5];
        let o = [0.3, -1.2, 2.0, 0.8, -0.4, 50.0];
        let h = [3.0, 1.5, 40.0, 2.0, 7.5, 0.0];
        let r = resolve_fluctuation(
            ArrayView1::from(&y[..]),
            ArrayView1::from(&o[..]),
            ArrayView1::from(&h[..]),
        )
        .unwrap();
        assert!(r.finite_root && r.offset_bound == 1e-8, "{r:?}");
        let sum_h: f64 = h.iter().sum();
        assert!(
            score(&y, &o, &h, 1e-8, r.epsilon).abs() <= 1e-12 * sum_h,
            "{r:?}"
        );

        // every follower's response at 1: no finite root, every follower at the link's clamp
        let y1 = [1.0, 1.0, 1.0, 0.3];
        let o1 = [2.0, -3.0, 0.5, 1.0];
        let h1 = [5.0, 1.0, 20.0, 0.0];
        let r = resolve_fluctuation(
            ArrayView1::from(&y1[..]),
            ArrayView1::from(&o1[..]),
            ArrayView1::from(&h1[..]),
        )
        .unwrap();
        assert!(!r.finite_root, "{r:?}");
        assert!(o1[..3].iter().all(|o| o + r.epsilon >= 30.0), "{r:?}");
        // and at 0
        let y0 = [0.0, 0.0];
        let r = resolve_fluctuation(
            ArrayView1::from(&y0[..]),
            ArrayView1::from(&o1[..2]),
            ArrayView1::from(&h1[..2]),
        )
        .unwrap();
        assert!(!r.finite_root && o1[..2].iter().all(|o| o + r.epsilon <= -30.0));
        // every response at linkinv's own lower clamp: S is 0 only at the clamp, and the search
        // stops at the first bound once |S| is within its target, before reaching it
        let floor = linkinv(-40.0);
        let yc = [floor, floor, floor];
        let oc = [0.5, -2.0, 1.0];
        let hc = [1.0, 2.0, 3.0];
        let r = resolve_fluctuation(
            ArrayView1::from(&yc[..]),
            ArrayView1::from(&oc[..]),
            ArrayView1::from(&hc[..]),
        )
        .unwrap();
        assert!(r.offset_bound == OFFSET_BOUNDS[0], "{r:?}");
        assert!(
            score(&yc, &oc, &hc, 1e-8, r.epsilon).abs() <= 1e-12 * 6.0,
            "{r:?}"
        );

        // offsets at +-1e3 are bounded to logit([1e-8, 1 - 1e-8]); with mixed responses S changes
        // sign in the bracket, so the first bound always solves
        let yb = [0.6, 0.3, 0.8];
        let ob = [1e3, -1e3, 0.0];
        let hb = [2.0, 2.0, 1.0];
        let r = resolve_fluctuation(
            ArrayView1::from(&yb[..]),
            ArrayView1::from(&ob[..]),
            ArrayView1::from(&hb[..]),
        )
        .unwrap();
        assert!(r.finite_root && r.offset_bound == OFFSET_BOUNDS[0], "{r:?}");
        assert!(score(&yb, &ob, &hb, r.offset_bound, r.epsilon).abs() <= 1e-12 * 5.0);

        // non-finite input is an error
        for (y, o, h) in [
            ([0.5, f64::NAN], [0.0, 0.0], [1.0, 1.0]),
            ([0.5, 0.5], [0.0, f64::NAN], [1.0, 1.0]),
            ([0.5, 0.5], [0.0, 0.0], [1.0, f64::INFINITY]),
        ] {
            assert!(
                resolve_fluctuation(
                    ArrayView1::from(&y[..]),
                    ArrayView1::from(&o[..]),
                    ArrayView1::from(&h[..]),
                )
                .is_err()
            );
        }
    }

    /// `base` with one more column in the third node's covariates: the first column of the
    /// second node's, plus uniform noise of half-width `jitter`.
    fn with_near_duplicate(base: &LtmleData, jitter: f64, seed: u64) -> LtmleData {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let n = base.n_obs();
        let mut ls = base.covariates.clone();
        let (a, b) = (ls[1].column(0).to_owned(), ls[2].ncols());
        let mut extra = Array2::<f64>::zeros((n, b + 1));
        for i in 0..n {
            for j in 0..b {
                extra[[i, j]] = ls[2][[i, j]];
            }
            extra[[i, b]] = a[i] + jitter * rng.gen_range(-1.0..1.0);
        }
        ls[2] = extra;
        LtmleData::new(base.outcomes.clone(), base.treatments.clone(), ls).unwrap()
    }

    #[test]
    fn test_ltmle_fast_paths_leave_for_collinear_and_separated_nodes() {
        let base = long_horizon(800, 4, 9);
        let cold = FitPaths {
            lean: false,
            warm: false,
        };

        // An exact duplicate reaches the shared GLM's aliasing report at each node that has it.
        let data = with_near_duplicate(&base, 0.0, 21);
        assert_same_as_shared_glm(&data, &tight(), 1e-10, 1e-10);
        let r = run_ltmle(&data, LtmleConfig::default()).unwrap();
        for node in [
            "g model for A3",
            "g model for A4",
            "Q model at A3",
            "Q model at A4",
        ] {
            assert!(
                r.warnings
                    .iter()
                    .any(|w| w.starts_with(node) && w.contains("singularities")),
                "{node}: {:?}",
                r.warnings
            );
        }

        // A near-duplicate reaches the shared GLM's QR: nodes 2 and 3 have the column, leave the
        // lean path and count the steps from glm.fit's start, node 0 stays on the warm path.
        let data = with_near_duplicate(&base, 1e-5, 21);
        assert_same_as_shared_glm(&data, &tight(), 1e-10, 1e-10);
        let data = with_near_duplicate(&base, 1e-7, 21);
        // The g fits' rounding reaches the clever covariate, and these nodes' condition number,
        // about 1e7, amplifies it in the near-duplicate's coefficient and the predictions; with
        // the warm start off the gap is the same.
        assert_same_as_shared_glm(&data, &tight(), 1e-10, 1e-8);
        let fast = run_ltmle(&data, LtmleConfig::default()).unwrap();
        let reference = run_ltmle_with(&data, LtmleConfig::default(), cold).unwrap();
        assert!(fast.warnings.is_empty(), "{:?}", fast.warnings);
        let steps = |r: &RegimeEstimate| -> Vec<usize> {
            r.q_models.iter().map(|m| m.iterations).collect()
        };
        let (a, b) = (
            steps(&fast.regime_estimate),
            steps(&reference.regime_estimate),
        );
        assert_eq!(a[2..], b[2..], "{a:?} vs {b:?}");
        assert!(a[0] < b[0], "{a:?} vs {b:?}");
        // The last node's response is the outcome itself: the same fit, bit for bit.
        assert_eq!(
            fast.regime_estimate.q_models[3].coefficients,
            reference.regime_estimate.q_models[3].coefficients
        );

        // An outcome separated by a covariate reaches its separation warnings.
        let mut outcomes = base.outcomes.clone();
        outcomes[3] = base.covariates[0]
            .column(0)
            .mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
        let data =
            LtmleData::new(outcomes, base.treatments.clone(), base.covariates.clone()).unwrap();
        assert_same_as_shared_glm(
            &data,
            &LtmleConfig::default(),
            1e-10,
            DEFAULT_CONTROL_NUISANCE,
        );
        let r = run_ltmle(&data, LtmleConfig::default()).unwrap();
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("fitted probabilities numerically 0 or 1")),
            "{:?}",
            r.warnings
        );
    }

    #[test]
    fn test_ltmle_linear_q_matches_after_the_row_predictions() {
        let data = long_horizon(1000, 5, 13);
        let mut c = LtmleConfig::default();
        c.q_model = LtmleQModel::Linear;
        assert_same_as_shared_glm(&data, &c, 1e-10, 1e-10);
    }

    #[test]
    fn test_ltmle_single_time_point_error() {
        let n = 10;
        let outcomes = vec![Array1::zeros(n)];
        let treatments = vec![Array1::zeros(n)];
        let covariates = vec![Array2::zeros((n, 2))];

        let data = LtmleData::new(outcomes, treatments, covariates).unwrap();
        let result = ltmle(&data);

        // Should error because LTMLE requires >= 2 time points
        assert!(result.is_err());
        if let Err(EconError::InvalidSpecification { message }) = result {
            assert!(message.contains("2 time points"));
        }
    }
}
