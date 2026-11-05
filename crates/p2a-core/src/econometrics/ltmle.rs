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
use crate::econometrics::glm::{GlmFamily, GlmLink, GlmSpec, fit_glm};
use crate::errors::{EconError, EconResult};
use crate::linalg::solve::aliasing_message;
use crate::regression::ols::solve_ols;
use crate::resource;
use crate::traits::convergence::{ResultNotes, notes_text};
use crate::traits::estimator::{SignificanceLevel, normal_p_value_two_sided, z_critical};

/// Variance estimator of every standard error.
const VARIANCE_METHOD: &str = "influence curve (ltmle variance.method = \"ic\")";

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
    /// Fluctuation coefficient eps_t per node.
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
    /// The intervention's fluctuation coefficients eps_t
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
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1")?;
        write!(f, "{}", notes_text(self))
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
    let g = fit_treatment_models(data, &designs, &config, &mut notes)?;
    // The last node's Q regression has the same response and design for every regime.
    let last_q = fit_q(
        &designs,
        t_max - 1,
        y_scaled.view(),
        &config,
        &data.treatment_names,
        &mut notes,
    )?;

    let mut estimates = Vec::with_capacity(regimes.len());
    for ((regime, d, label), follow) in regimes.into_iter().zip(&follow) {
        estimates.push(estimate_regime(
            data, regime, &d, label, follow, &designs, &g, &last_q, &y_scaled, &scale, &config,
            &mut notes,
        )?);
    }

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
    let regime_estimate = estimates.pop().expect("the intervention's estimate");

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
    })
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

/// The observed-history designs: `q[t]` has the columns "(Intercept)", L_1.., A_1, ...,
/// L_t.., A_t (R's `reformulate` order); the g design of node t is `q[t]` without its last
/// column. `a_cols[t][s]` is the column of A_s in `q[t]` (s <= t).
struct HistoryDesigns {
    q: Vec<Array2<f64>>,
    names: Vec<Vec<String>>,
    a_cols: Vec<Vec<usize>>,
}

impl HistoryDesigns {
    fn new(data: &LtmleData) -> EconResult<Self> {
        let n = data.n_obs();
        let t_max = data.time_points();
        let (mut q, mut names, mut a_cols) = (
            Vec::with_capacity(t_max),
            Vec::with_capacity(t_max),
            Vec::with_capacity(t_max),
        );
        for node in 0..t_max {
            resource::checkpoint()?;
            let mut p = 1usize;
            for s in 0..=node {
                p = checked_add(p, checked_add(data.covariates[s].ncols(), 1)?)?;
            }
            let mut x = resource::try_zeros2(n, p, "LTMLE history design")?;
            let mut nm = Vec::with_capacity(p);
            let mut ac = Vec::with_capacity(node + 1);
            x.column_mut(0).fill(1.0);
            nm.push("(Intercept)".to_string());
            let mut j = 1;
            for s in 0..=node {
                for (c, col) in data.covariates[s].columns().into_iter().enumerate() {
                    x.column_mut(j).assign(&col);
                    nm.push(data.covariate_names[s][c].clone());
                    j += 1;
                }
                x.column_mut(j).assign(&data.treatments[s]);
                nm.push(data.treatment_names[s].clone());
                ac.push(j);
                j += 1;
            }
            q.push(x);
            names.push(nm);
            a_cols.push(ac);
        }
        Ok(Self { q, names, a_cols })
    }

    /// The g design of node t (the Q design without A_t) and its names.
    fn g(&self, t: usize) -> (ArrayView2<'_, f64>, &[String]) {
        let p = self.q[t].ncols() - 1;
        (self.q[t].slice(ndarray::s![.., ..p]), &self.names[t][..p])
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
    let p = x.ncols();
    // column -> node whose regime value replaces it
    let mut set: Vec<Option<usize>> = vec![None; p];
    if d.is_some() {
        for (s, &c) in a_cols.iter().enumerate() {
            if c < p {
                set[c] = Some(s);
            }
        }
    }
    let mut eta = Array1::<f64>::zeros(x.nrows());
    for (j, (&b, col)) in coefficients.iter().zip(x.columns()).enumerate() {
        if b.is_nan() {
            continue;
        }
        match (set[j], d) {
            (Some(s), Some(d)) => {
                for (e, &v) in eta.iter_mut().zip(&d[s]) {
                    *e += f64::from(v) * b;
                }
            }
            _ => eta.scaled_add(b, &col),
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
    if q_model == LtmleQModel::Logistic && (min < 0.0 || max > 1.0) {
        let width = max - min;
        if !(width > 0.0) {
            return Err(spec_err(format!(
                "LTMLE: the outcome is constant ({min}); nothing to estimate"
            )));
        }
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
    /// IRLS iterations (0 for least-squares fits).
    pub iterations: usize,
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

fn fit_treatment_models(
    data: &LtmleData,
    designs: &HistoryDesigns,
    config: &LtmleConfig,
    notes: &mut FitNotes,
) -> EconResult<TreatmentModels> {
    let t_max = data.time_points();
    let mut out = TreatmentModels {
        coefficients: Vec::with_capacity(t_max),
        models: Vec::with_capacity(t_max),
        fitted: Vec::with_capacity(t_max),
    };
    let spec = glm_spec(GlmFamily::Binomial, config);
    for t in 0..t_max {
        resource::checkpoint()?;
        let (x, names) = designs.g(t);
        let fit = fit_glm(x, data.treatments[t].view(), &spec)?;
        notes.push_glm(
            &format!("g model for {}", data.treatment_names[t]),
            &fit,
            names,
        );
        out.models.push(NodeModel {
            node: t,
            names: names.to_vec(),
            coefficients: fit.coefficients.to_vec(),
            converged: fit.converged,
            iterations: fit.iterations,
        });
        out.coefficients.push(fit.coefficients);
        out.fitted.push(fit.fitted);
    }
    Ok(out)
}

/// The Q regression of node t: `q_next` on the observed history through A_t.
fn fit_q(
    designs: &HistoryDesigns,
    t: usize,
    q_next: ArrayView1<f64>,
    config: &LtmleConfig,
    treatment_names: &[String],
    notes: &mut FitNotes,
) -> EconResult<(Array1<f64>, NodeModel)> {
    let x = designs.q[t].view();
    let names = &designs.names[t];
    let what = format!("Q model at {}", treatment_names[t]);
    match config.q_model {
        LtmleQModel::Logistic => {
            let fit = fit_glm(x, q_next, &glm_spec(GlmFamily::QuasiBinomial, config))?;
            notes.push_glm(&what, &fit, names);
            let model = NodeModel {
                node: t,
                names: names.clone(),
                coefficients: fit.coefficients.to_vec(),
                converged: fit.converged,
                iterations: fit.iterations,
            };
            Ok((fit.coefficients, model))
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
                names: names.clone(),
                coefficients: fit.beta.to_vec(),
                converged: true,
                iterations: 0,
            };
            Ok((fit.beta, model))
        }
    }
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
    last_q: &(Array1<f64>, NodeModel),
    y: &Array1<f64>,
    scale: &OutcomeScale,
    config: &LtmleConfig,
    notes: &mut FitNotes,
) -> EconResult<RegimeEstimate> {
    let n = data.n_obs();
    let t_max = data.time_points();
    let (lo, hi) = config.gbounds;

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

    // Backward pass.
    let mut q_next = y.clone();
    let mut ic = Array1::<f64>::zeros(n);
    let mut eps = vec![0.0; t_max];
    let mut q_star: Vec<Array1<f64>> = vec![Array1::zeros(0); t_max];
    let mut q_models = vec![NodeModel::default(); t_max];
    for t in (0..t_max).rev() {
        resource::checkpoint()?;
        let fitted;
        let (coef, model) = if t == t_max - 1 {
            (&last_q.0, &last_q.1)
        } else {
            fitted = fit_q(
                designs,
                t,
                q_next.view(),
                config,
                &data.treatment_names,
                notes,
            )?;
            (&fitted.0, &fitted.1)
        };
        q_models[t] = model.clone();
        let eta = identified_linear_predictor(
            designs.q[t].view(),
            coef.view(),
            &designs.a_cols[t],
            Some(d),
        );
        let ht = Array1::from(h[t].clone());
        let qs = match config.q_model {
            LtmleQModel::Logistic => {
                let f = fit_fluctuation(q_next.view(), eta.view(), ht.view(), config)?;
                notes.n_fits += 1;
                notes.all_converged &= f.converged;
                if !f.converged {
                    notes.warnings.push(format!(
                        "fluctuation at {} ({label}): did not converge in {} iterations",
                        data.treatment_names[t], f.iterations
                    ));
                }
                eps[t] = f.epsilon;
                eta.mapv(|e| linkinv(e + eps[t]))
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
        q_next = qs.clone();
        q_star[t] = qs;
    }
    let psi_s = q_next.mean().unwrap_or(f64::NAN);
    for i in 0..n {
        ic[i] = (ic[i] + q_next[i] - psi_s) * scale.width;
    }
    let psi = scale.lower + scale.width * psi_s;
    let influence_curve = ic.to_vec();
    let se = ic_se(&influence_curve)?;
    let zc = z_critical(0.95)?;
    Ok(RegimeEstimate {
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
    })
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
}

/// For one coefficient on the canonical link, glm.fit's IRLS step is Newton's step
/// `score / information`; this solves the same score equation
/// `sum_i h_i (y_i - plogis(offset_i + eps)) = 0` in O(n) per iteration instead of a
/// general GLM fit, with glm.fit's stopping rule `|dev - dev_old| / (|dev| + 0.1) <
/// tolerance`, glm.fit's step halving when the deviance is not finite, and one polishing
/// Newton step after the rule is met.
fn fit_fluctuation(
    y: ArrayView1<f64>,
    offset: ArrayView1<f64>,
    h: ArrayView1<f64>,
    config: &LtmleConfig,
) -> EconResult<Fluctuation> {
    let rows: Vec<usize> = (0..h.len()).filter(|&i| h[i] > 0.0).collect();
    // binomial deviance, quasi-binomial working form: 2 sum h [y log(y/p) + (1-y) log((1-y)/(1-p))]
    let deviance = |eps: f64| -> f64 {
        let mut d = 0.0;
        for &i in &rows {
            let p = linkinv(offset[i] + eps);
            let (yi, wi) = (y[i], h[i]);
            let a = if yi > 0.0 { yi * (yi / p).ln() } else { 0.0 };
            let b = if yi < 1.0 {
                (1.0 - yi) * ((1.0 - yi) / (1.0 - p)).ln()
            } else {
                0.0
            };
            d += 2.0 * wi * (a + b);
        }
        d
    };
    let mut eps = 0.0;
    let mut dev = deviance(eps);
    if !dev.is_finite() {
        return Err(EconError::Computation(
            "LTMLE: the fluctuation's starting deviance is not finite".into(),
        ));
    }
    for iter in 1..=config.max_iter {
        resource::checkpoint()?;
        let (mut score, mut info) = (0.0, 0.0);
        for &i in &rows {
            let p = linkinv(offset[i] + eps);
            score += h[i] * (y[i] - p);
            info += h[i] * p * (1.0 - p);
        }
        if !(info > 0.0) || score == 0.0 {
            return Ok(Fluctuation {
                epsilon: eps,
                converged: score == 0.0,
                iterations: iter,
            });
        }
        let mut step = score / info;
        let mut new_eps = eps + step;
        let mut new_dev = deviance(new_eps);
        let mut halvings = 0;
        // glm.fit halves the step only when the deviance is not finite.
        while !new_dev.is_finite() && halvings < 30 {
            step /= 2.0;
            new_eps = eps + step;
            new_dev = deviance(new_eps);
            halvings += 1;
        }
        let change = (new_dev - dev).abs() / (new_dev.abs() + 0.1);
        eps = new_eps;
        dev = new_dev;
        if change < config.tolerance {
            // One more Newton step: the deviance rule leaves an O(sqrt(tolerance)) error in
            // eps when the weights are large, and Newton's step squares it at O(n) cost.
            let (mut score, mut info) = (0.0, 0.0);
            for &i in &rows {
                let p = linkinv(offset[i] + eps);
                score += h[i] * (y[i] - p);
                info += h[i] * p * (1.0 - p);
            }
            if info > 0.0 {
                let polished = eps + score / info;
                if deviance(polished).is_finite() {
                    eps = polished;
                }
            }
            return Ok(Fluctuation {
                epsilon: eps,
                converged: true,
                iterations: iter,
            });
        }
    }
    Ok(Fluctuation {
        epsilon: eps,
        converged: false,
        iterations: config.max_iter,
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
        // measured gaps 2.8e-13 (psi_always), 1.1e-11 (psi_never), 6.0e-15 (ATE SE).
        let (case, ds) = crate::test_support::load_case("ltmle_two_period");
        let r = run_ltmle(&two_period(&ds), LtmleConfig::default()).unwrap();
        Check::new(&case)
            .scalar("psi_always", r.psi_treated, Tol::iterative(1e-8))
            .scalar("psi_never", r.psi_control, Tol::iterative(1e-8))
            .scalar("ate_se", r.se, Tol::iterative(1e-8))
            .finish();
        // the tight eps_never at the default control: measured gap 4.1e-10 relative
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
