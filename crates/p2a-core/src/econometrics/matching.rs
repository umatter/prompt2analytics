//! Propensity Score Matching (MatchIt) for causal inference.
//!
//! This module provides various matching methods to create balanced comparison groups
//! for estimating causal effects from observational data. Matching reduces confounding
//! by creating treated and control groups with similar covariate distributions.
//!
//! # Methods
//!
//! - **Nearest Neighbor Matching**: For each treated unit, find the closest control
//!   unit(s) by propensity score distance (logit or probit), or by Mahalanobis (pooled
//!   within-group covariance) or Euclidean covariate distance. Supports with/without
//!   replacement, calipers and MatchIt's `m.order` (the order treated units are matched in).
//!
//! - **Coarsened Exact Matching (CEM)**: Coarsen covariates into bins and match exactly
//!   within strata. Prunes observations in unmatched strata.
//!
//! - **Full/Optimal Matching**: Creates optimal strata containing both treated and
//!   control units, minimizing total distance within strata.
//!
//! - **Subclassification**: Stratifies observations into subclasses based on propensity
//!   scores for within-stratum comparisons.
//!
//! # Balance Diagnostics
//!
//! All matching methods return balance diagnostics including:
//! - Standardised mean differences (`summary.matchit`, treated SD of the full sample)
//! - Variance ratios (NA for binary covariates)
//! - eCDF mean and max (`ks_statistic` = eCDF max)
//!
//! # References
//!
//! - Ho, D.E., Imai, K., King, G., & Stuart, E.A. (2007). Matching as Nonparametric
//!   Preprocessing for Reducing Model Dependence in Parametric Causal Inference.
//!   *Political Analysis*, 15(3), 199-236. https://doi.org/10.1093/pan/mpl013
//!
//! - Rosenbaum, P.R. & Rubin, D.B. (1983). The Central Role of the Propensity Score
//!   in Observational Studies for Causal Effects. *Biometrika*, 70(1), 41-55.
//!   https://doi.org/10.1093/biomet/70.1.41
//!
//! - Iacus, S.M., King, G., & Porro, G. (2012). Causal Inference without Balance
//!   Checking: Coarsened Exact Matching. *Political Analysis*, 20(1), 1-24.
//!   https://doi.org/10.1093/pan/mpr013
//!
//! - Hansen, B.B. (2004). Full Matching in an Observational Study of Coaching for
//!   the SAT. *Journal of the American Statistical Association*, 99(467), 609-618.
//!   https://doi.org/10.1198/016214504000000647
//!
//! - Pinned to MatchIt 4.8.0 and optmatch 0.10.8 (validation/r/cases/causal_matching.R; see
//!   validation/econometrics/matching.md for the deviations).
//!   https://cran.r-project.org/package=MatchIt

use ndarray::{Array1, Array2, ArrayView2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::econ_error_from_design;
use crate::econometrics::glm::{GlmLink, fit_propensity};
use crate::econometrics::min_cost_flow::{FlowStatus, Network, network_simplex};
use crate::errors::{EconError, EconResult};
use crate::linalg::design::DesignMatrix;
use crate::linalg::matrix_ops::eig_symmetric;
use crate::linalg::solve::{SingularPolicy, sym_inverse};
use crate::stats::robust::{QuantileType, quantile_sorted};
use crate::traits::convergence::{Convergence, ResultNotes, notes_text};

/// Threshold for using parallel algorithms (to avoid overhead on small datasets)
const PARALLEL_THRESHOLD: usize = 500;

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Matching method configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MatchMethod {
    /// Nearest neighbor matching on propensity score.
    NearestNeighbor {
        /// Number of control matches per treated unit (1:k matching).
        ratio: usize,
        /// Maximum distance for a valid match. If None, no caliper is used.
        caliper: Option<f64>,
        /// Whether to sample controls with replacement.
        replace: bool,
    },
    /// Coarsened exact matching within covariate strata (MatchIt `method = "cem"`).
    CoarsenedExact {
        /// Cut points per covariate (MatchIt `cutpoints = list(..)`), left-closed intervals as
        /// `findInterval`. Takes precedence over `n_bins`.
        cutpoints: Option<Vec<Vec<f64>>>,
        /// `Some(k)`: `k` quantile bins per covariate (MatchIt `cutpoints = "q<k>"`, type-7
        /// quantiles); `None`: MatchIt's default, `nclass.Sturges` equal-width bins
        /// (`cutpoints = "sturges"`).
        n_bins: Option<usize>,
    },
    /// Full/optimal matching creating optimal strata.
    Full {
        /// Minimum ratio of controls to treated within any stratum.
        min_ratio: f64,
        /// Maximum ratio of controls to treated within any stratum.
        max_ratio: f64,
    },
    /// Propensity score subclassification.
    Subclass {
        /// Number of subclasses to create.
        n_subclasses: usize,
    },
}

impl Default for MatchMethod {
    fn default() -> Self {
        MatchMethod::NearestNeighbor {
            ratio: 1,
            caliper: None,
            replace: false,
        }
    }
}

impl fmt::Display for MatchMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatchMethod::NearestNeighbor {
                ratio,
                caliper,
                replace,
            } => {
                write!(f, "Nearest Neighbor (1:{})", ratio)?;
                if let Some(c) = caliper {
                    write!(f, ", caliper={:.4}", c)?;
                }
                if *replace {
                    write!(f, ", with replacement")?;
                }
                Ok(())
            }
            MatchMethod::CoarsenedExact { cutpoints, n_bins } => match (cutpoints, n_bins) {
                (Some(_), _) => write!(f, "Coarsened Exact Matching (user cut points)"),
                (None, Some(k)) => write!(f, "Coarsened Exact Matching ({k} quantile bins)"),
                (None, None) => write!(f, "Coarsened Exact Matching (Sturges bins)"),
            },
            MatchMethod::Full {
                min_ratio,
                max_ratio,
            } => {
                let b = |v: f64| {
                    if v == f64::INFINITY {
                        "Inf".to_string()
                    } else {
                        v.to_string()
                    }
                };
                write!(
                    f,
                    "Optimal Full Matching (controls per treated unit in [{}, {}])",
                    b(*min_ratio),
                    b(*max_ratio)
                )
            }
            MatchMethod::Subclass { n_subclasses } => {
                write!(f, "Subclassification ({} subclasses)", n_subclasses)
            }
        }
    }
}

/// Distance metric for matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DistanceMethod {
    /// Logistic regression propensity score (default).
    #[default]
    Logit,
    /// Probit propensity score.
    Probit,
    /// Mahalanobis distance on covariates.
    Mahalanobis,
    /// Euclidean distance on covariates.
    Euclidean,
}

impl fmt::Display for DistanceMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistanceMethod::Logit => write!(f, "Logit (Propensity Score)"),
            DistanceMethod::Probit => write!(f, "Probit (Propensity Score)"),
            DistanceMethod::Mahalanobis => write!(f, "Mahalanobis Distance"),
            DistanceMethod::Euclidean => write!(f, "Euclidean Distance"),
        }
    }
}

crate::option_enum!(
    DistanceMethod,
    crate::options::Choices {
        option: "distance",
        values: &[
            (&["logit", "glm", "logistic", "ps"], DistanceMethod::Logit),
            (&["probit"], DistanceMethod::Probit),
            (&["mahalanobis"], DistanceMethod::Mahalanobis),
            (&["euclidean"], DistanceMethod::Euclidean),
        ],
    },
    no_display
);

/// MatchIt `m.order`: the order in which treated units choose their nearest controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MOrder {
    /// Largest propensity score first (MatchIt's default for a propensity score).
    Largest,
    /// Smallest propensity score first.
    Smallest,
    /// Data order (MatchIt's default for Mahalanobis and Euclidean distances, and always
    /// used with replacement).
    Data,
    /// Seeded random order ([`MatchOptions::seed`]).
    Random,
}

crate::option_enum!(
    MOrder,
    crate::options::Choices {
        option: "m_order",
        values: &[
            (&["largest", "descending"], MOrder::Largest),
            (&["smallest", "ascending"], MOrder::Smallest),
            (&["data", "data_order"], MOrder::Data),
            (&["random"], MOrder::Random),
        ],
    }
);

/// The method names the MCP and CLI accept (the variant data stay in [`MatchMethod`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchMethodKind {
    /// Nearest-neighbour matching.
    Nearest,
    /// Coarsened exact matching.
    Cem,
    /// Optimal full matching.
    Full,
    /// Propensity-score subclassification.
    Subclass,
}

crate::option_enum!(
    MatchMethodKind,
    crate::options::Choices {
        option: "method",
        values: &[
            (
                &["nearest", "nn", "nearest_neighbor", "nearest_neighbour"],
                MatchMethodKind::Nearest
            ),
            (
                &["cem", "coarsened", "coarsened_exact", "coarsenedexact"],
                MatchMethodKind::Cem
            ),
            (
                &["full", "optimal", "full_matching", "optimal_full"],
                MatchMethodKind::Full
            ),
            (
                &[
                    "subclass",
                    "subclassification",
                    "stratify",
                    "stratification"
                ],
                MatchMethodKind::Subclass
            ),
        ],
    }
);

/// Seed of [`MOrder::Random`] when [`MatchOptions::seed`] is None.
pub const DEFAULT_MATCH_SEED: u64 = 42;

/// MatchIt arguments beyond the method and the distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MatchOptions {
    /// None = MatchIt's default: Largest for logit/probit distances, Data for
    /// Mahalanobis/Euclidean. Ignored by CEM, full and subclass matching, and by
    /// nearest-neighbour matching with replacement (always data order, as MatchIt; an
    /// explicit order other than Data then adds a warning).
    pub m_order: Option<MOrder>,
    /// Seed of [`MOrder::Random`]; None = [`DEFAULT_MATCH_SEED`].
    pub seed: Option<u64>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Result Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Balance statistics for a single covariate in matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchCovariateBalance {
    /// Covariate name.
    pub name: String,
    /// Mean in treated group.
    pub mean_treated: f64,
    /// Mean in control group.
    pub mean_control: f64,
    /// summary.matchit's "Std. Mean Diff." for the ATT: the weighted mean difference over the
    /// standard deviation of the treated group in the full (unmatched, unweighted) sample.
    /// Rule of thumb: |SMD| < 0.1 indicates good balance.
    pub std_diff: f64,
    /// summary.matchit's "Var. Ratio": weighted variance of the treated over that of the
    /// controls; NaN for binary covariates and groups with fewer than two weighted units.
    pub var_ratio: f64,
    /// Equal to `ecdf_max` (the weighted Kolmogorov-Smirnov statistic; with unit weights the
    /// two-sample KS statistic).
    pub ks_statistic: f64,
    /// summary.matchit's "eCDF Mean": mean absolute difference of the weighted eCDFs.
    #[serde(default)]
    pub ecdf_mean: f64,
    /// summary.matchit's "eCDF Max": largest absolute difference of the weighted eCDFs.
    #[serde(default)]
    pub ecdf_max: f64,
}

/// A table cell: NaN prints as "-".
fn cell(v: f64) -> String {
    if v.is_nan() {
        "-".to_string()
    } else {
        format!("{v:.4}")
    }
}

/// Balance table comparing covariate distributions for matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchBalanceTable {
    /// Balance statistics for each covariate.
    pub covariates: Vec<MatchCovariateBalance>,
    /// Mean absolute standardized difference across all covariates.
    pub mean_abs_std_diff: f64,
    /// Maximum absolute standardized difference.
    pub max_abs_std_diff: f64,
    /// Number of covariates with |SMD| > 0.1 (imbalanced).
    pub n_imbalanced: usize,
}

impl fmt::Display for MatchBalanceTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{:<20} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "Covariate", "Mean(T)", "Mean(C)", "Std.Diff", "Var.Ratio", "eCDF Mean", "eCDF Max"
        )?;
        writeln!(f, "{}", "-".repeat(92))?;

        for cov in &self.covariates {
            let balance_flag = if cov.std_diff.abs() > 0.1 { "!" } else { "" };
            writeln!(
                f,
                "{:<20} {:>10} {:>10} {:>10}{} {:>10} {:>10} {:>10}",
                cov.name,
                cell(cov.mean_treated),
                cell(cov.mean_control),
                cell(cov.std_diff),
                balance_flag,
                cell(cov.var_ratio),
                cell(cov.ecdf_mean),
                cell(cov.ecdf_max)
            )?;
        }

        writeln!(f, "{}", "-".repeat(92))?;
        writeln!(f, "Mean Abs. Std. Diff: {:.4}", self.mean_abs_std_diff)?;
        writeln!(f, "Max Abs. Std. Diff:  {:.4}", self.max_abs_std_diff)?;
        writeln!(f, "Imbalanced (|SMD|>0.1): {}", self.n_imbalanced)?;

        Ok(())
    }
}

/// Match information for a single treated unit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchInfo {
    /// Index of the treated unit in the original data.
    pub treated_idx: usize,
    /// Indices of matched control units.
    pub control_indices: Vec<usize>,
    /// Distances to matched controls.
    pub distances: Vec<f64>,
    /// Matching weight for this treated unit.
    pub weight: f64,
}

/// Subclass information for stratification methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubclassInfo {
    /// Subclass ID (1-indexed).
    pub subclass_id: usize,
    /// Indices of treated units in this subclass.
    pub treated_indices: Vec<usize>,
    /// Indices of control units in this subclass.
    pub control_indices: Vec<usize>,
    /// Propensity score range [min, max] for this subclass.
    pub ps_range: (f64, f64),
}

/// Result from propensity score matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchResult {
    /// Matching method used.
    pub method: MatchMethod,
    /// Distance method used.
    pub distance: DistanceMethod,
    /// Balance table before matching.
    pub balance_before: MatchBalanceTable,
    /// Balance table after matching.
    pub balance_after: MatchBalanceTable,
    /// Match information for each treated unit.
    pub matches: Vec<MatchInfo>,
    /// Subclass information (for Full/Subclass methods).
    pub subclasses: Option<Vec<SubclassInfo>>,
    /// MatchIt's matching weights for all observations (0 for unmatched), normalised so the
    /// positive weights of each group sum to their count.
    pub weights: Vec<f64>,
    /// Propensity scores (None when no propensity score is used).
    pub propensity_scores: Option<Vec<f64>>,
    /// Total number of observations.
    pub n_obs: usize,
    /// Number of treated units.
    pub n_treated: usize,
    /// Number of control units.
    pub n_control: usize,
    /// Number of matched treated units.
    pub n_matched_treated: usize,
    /// Number of matched control units.
    pub n_matched_control: usize,
    /// Number of treated units discarded (unmatched).
    pub n_discarded_treated: usize,
    /// Number of control units discarded (unmatched).
    pub n_discarded_control: usize,
    /// Caliper used (if applicable).
    pub caliper_used: Option<f64>,
    /// `ess_treated + ess_control` (MatchIt's "Matched (ESS)" of both groups).
    pub effective_sample_size: f64,
    /// Convergence of the propensity-score glm (None when no propensity score is used).
    #[serde(default)]
    pub propensity_convergence: Option<Convergence>,
    /// Warnings: the propensity-score glm's, MatchIt's sample warnings and notes on
    /// generalised inverses, unmatched units and discarded subclasses.
    #[serde(default)]
    pub warnings: Vec<String>,
    /// Order used by nearest-neighbour matching (Data with replacement, as MatchIt); None for
    /// the other methods.
    #[serde(default)]
    pub m_order: Option<MOrder>,
    /// MatchIt's `subclass` per row, 1-based: the pair (nearest neighbour without
    /// replacement, numbered in treated data order), stratum (CEM), matched set (full) or
    /// subclass; None = unmatched, or nearest neighbour with replacement (MatchIt: NULL).
    #[serde(default)]
    pub subclass: Vec<Option<usize>>,
    /// Caliper on the distance scale (propensity score: `caliper * sd(ps)`; covariate
    /// distances: as given).
    #[serde(default)]
    pub caliper_abs: Option<f64>,
    /// Full matching: the total distance within the matched sets.
    #[serde(default)]
    pub total_distance: Option<f64>,
    /// Kish effective sample size `(sum w)^2 / sum w^2` of the matched treated units.
    #[serde(default)]
    pub ess_treated: f64,
    /// Kish effective sample size of the matched controls.
    #[serde(default)]
    pub ess_control: f64,
}

impl ResultNotes for MatchResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }

    fn notes_convergence(&self) -> Option<Convergence> {
        self.propensity_convergence.clone()
    }
}

impl fmt::Display for MatchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Propensity Score Matching Results")?;
        writeln!(f, "==================================")?;
        writeln!(f)?;
        writeln!(f, "Method:   {}", self.method)?;
        if let Some(o) = self.m_order {
            writeln!(f, "Matching order: {o} (MatchIt m.order)")?;
        }
        if matches!(self.method, MatchMethod::CoarsenedExact { .. }) {
            writeln!(f, "Distance: not used (coarsened exact matching)")?;
        } else {
            writeln!(f, "Distance: {}", self.distance)?;
        }
        writeln!(f)?;

        writeln!(f, "Sample Sizes:")?;
        writeln!(f, "  Total:           {}", self.n_obs)?;
        writeln!(
            f,
            "  Treated:         {} (matched: {}, discarded: {})",
            self.n_treated, self.n_matched_treated, self.n_discarded_treated
        )?;
        writeln!(
            f,
            "  Control:         {} (matched: {}, discarded: {})",
            self.n_control, self.n_matched_control, self.n_discarded_control
        )?;
        writeln!(
            f,
            "  Matched ESS:     {:.1} treated, {:.1} control",
            self.ess_treated, self.ess_control
        )?;
        writeln!(f, "  Effective N:     {:.1}", self.effective_sample_size)?;
        writeln!(f)?;

        if let Some(caliper) = self.caliper_used {
            match self.distance {
                DistanceMethod::Logit | DistanceMethod::Probit => writeln!(
                    f,
                    "Caliper: {:.4} SD of the propensity score ({:.6} on the score scale)",
                    caliper,
                    self.caliper_abs.unwrap_or(f64::NAN)
                )?,
                d => writeln!(
                    f,
                    "Caliper: {} (in {} distance units)",
                    caliper,
                    DistanceMethod::choices().canonical(d)
                )?,
            }
            writeln!(f)?;
        }
        if let Some(d) = self.total_distance {
            writeln!(f, "Total distance within matched sets: {d:.6} (optimal)")?;
            writeln!(f)?;
        }

        writeln!(
            f,
            "Balance Before Matching (summary.matchit; SMD standardised by the treated SD):"
        )?;
        write!(f, "{}", self.balance_before)?;
        writeln!(f)?;
        writeln!(f, "Balance After Matching:")?;
        write!(f, "{}", self.balance_after)?;
        writeln!(f)?;

        let improvement = 1.0
            - (self.balance_after.mean_abs_std_diff
                / self.balance_before.mean_abs_std_diff.max(1e-10));
        writeln!(f, "Balance Improvement: {:.1}%", improvement * 100.0)?;
        write!(f, "{}", notes_text(self))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main Matching Function
// ═══════════════════════════════════════════════════════════════════════════════

/// Run propensity score matching (MatchIt's `matchit` for the ATT).
///
/// `match_it(..)` is [`match_it_with`] with [`MatchOptions::default()`]: MatchIt's default
/// `m.order` (largest first for a propensity score, data order for Mahalanobis and
/// Euclidean distances).
///
/// # Example
/// ```ignore
/// let method = MatchMethod::NearestNeighbor {
///     ratio: 1,
///     caliper: Some(0.2),
///     replace: false,
/// };
/// let result = match_it(&dataset, "treatment", &["age", "income", "education"],
///                       method, Some(DistanceMethod::Logit))?;
/// println!("{}", result);
/// ```
///
/// # References
///
/// Ho, D.E., Imai, K., King, G., & Stuart, E.A. (2007). "Matching as Nonparametric
/// Preprocessing for Reducing Model Dependence in Parametric Causal Inference."
/// *Political Analysis*, 15(3), 199-236.
pub fn match_it(
    dataset: &Dataset,
    treatment_col: &str,
    covariate_cols: &[&str],
    method: MatchMethod,
    distance: Option<DistanceMethod>,
) -> EconResult<MatchResult> {
    match_it_with(
        dataset,
        treatment_col,
        covariate_cols,
        method,
        distance,
        &MatchOptions::default(),
    )
}

/// What one matching method hands back to [`match_it_with`].
struct MethodOutput {
    /// MatchIt's weights per row, already normalised (`.make_sum_to_n`) by the method.
    weights: Vec<f64>,
    matches: Vec<MatchInfo>,
    subclass: Vec<Option<usize>>,
    subclasses: Option<Vec<SubclassInfo>>,
    m_order: Option<MOrder>,
    caliper_abs: Option<f64>,
    total_distance: Option<f64>,
    warnings: Vec<String>,
}

/// [`match_it`] with MatchIt's further arguments ([`MatchOptions`]: `m.order` and the seed of
/// the random order).
///
/// Steps (S1): the treatment must be coded 0/1; the covariates are extracted; the method's
/// parameters are checked (`ratio`, `n_subclasses`, `n_bins` and `cutpoints` against the data,
/// a positive finite caliper, `m_order` largest/smallest only with a propensity score,
/// subclassification only on a propensity score); balance before matching; the method; the
/// weights normalised as MatchIt's (`.make_sum_to_n`); balance after matching and the ESS.
pub fn match_it_with(
    dataset: &Dataset,
    treatment_col: &str,
    covariate_cols: &[&str],
    method: MatchMethod,
    distance: Option<DistanceMethod>,
    options: &MatchOptions,
) -> EconResult<MatchResult> {
    let distance_method = distance.unwrap_or_default();

    // Missing values are refused (MatchIt, WeightIt and tmle refuse them).
    let treatment = DesignMatrix::extract_column(dataset.df(), treatment_col)
        .map_err(|e| econ_error_from_design(e, dataset.df()))?;
    let n = treatment.len();
    if let Some((i, v)) = treatment
        .iter()
        .enumerate()
        .find(|(_, v)| **v != 0.0 && **v != 1.0)
    {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "treatment column '{treatment_col}' must be coded 0/1 (row {} is {v})",
                i + 1
            ),
        });
    }
    let is_treated: Vec<bool> = treatment.iter().map(|&v| v == 1.0).collect();
    let n_treated = is_treated.iter().filter(|&&t| t).count();
    let n_control = n - n_treated;
    if n_treated == 0 || n_control == 0 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Treatment column '{}' must have both treated (1) and control (0) units. Found {} treated, {} control.",
                treatment_col, n_treated, n_control
            ),
        });
    }

    validate_match_counts(&method, n, n_treated, n_control, covariate_cols.len())?;

    let design = DesignMatrix::from_dataframe(dataset.df(), covariate_cols, false)
        .map_err(|e| econ_error_from_design(e, dataset.df()))?;
    let x = design.data;
    let covariate_names: Vec<String> = covariate_cols.iter().map(|s| s.to_string()).collect();

    let ps_link = match distance_method {
        DistanceMethod::Logit => Some(GlmLink::Logit),
        DistanceMethod::Probit => Some(GlmLink::Probit),
        DistanceMethod::Mahalanobis | DistanceMethod::Euclidean => None,
    };
    validate_match_options(
        &method,
        distance_method,
        ps_link.is_some(),
        options,
        covariate_cols,
    )?;

    let balance_before = balance_table(&x, &covariate_names, &is_treated, &vec![1.0; n]);

    let uses_ps = ps_link.is_some() && !matches!(method, MatchMethod::CoarsenedExact { .. });
    let ps_model = match ps_link {
        Some(link) if uses_ps => Some(estimate_propensity(&x, &treatment, link)?),
        _ => None,
    };
    let scores: Option<Vec<f64>> = ps_model.as_ref().map(|m| m.scores.to_vec());

    let out = match &method {
        MatchMethod::NearestNeighbor {
            ratio,
            caliper,
            replace,
        } => nearest_neighbour(
            &x,
            &is_treated,
            scores.as_deref(),
            distance_method,
            *ratio,
            *caliper,
            *replace,
            options,
        )?,
        MatchMethod::CoarsenedExact { cutpoints, n_bins } => {
            cem_method(&x, &is_treated, cutpoints.as_deref(), *n_bins)?
        }
        MatchMethod::Full {
            min_ratio,
            max_ratio,
        } => full_method(
            &x,
            &is_treated,
            scores.as_deref(),
            distance_method,
            *min_ratio,
            *max_ratio,
        )?,
        MatchMethod::Subclass { n_subclasses } => {
            subclass_method(&is_treated, scores.as_deref().unwrap_or(&[]), *n_subclasses)?
        }
    };

    let weights = out.weights;
    let n_matched_treated = (0..n)
        .filter(|&i| is_treated[i] && weights[i] > 0.0)
        .count();
    let n_matched_control = (0..n)
        .filter(|&i| !is_treated[i] && weights[i] > 0.0)
        .count();
    if n_matched_treated == 0 || n_matched_control == 0 {
        let message = match &method {
            MatchMethod::NearestNeighbor { caliper, .. } => format!(
                "no treated unit has a control within the caliper ({} = {} on the distance \
                 scale); widen the caliper",
                caliper.unwrap_or(f64::NAN),
                out.caliper_abs.unwrap_or(f64::NAN)
            ),
            MatchMethod::CoarsenedExact { .. } => "no units were matched: no stratum holds both \
                 treated and control units; coarsen the variables further or match on fewer \
                 covariates"
                .to_string(),
            MatchMethod::Subclass { .. } => {
                "no subclass holds both treated and control units".to_string()
            }
            MatchMethod::Full { .. } => "no units were matched".to_string(),
        };
        return Err(EconError::InvalidSpecification { message });
    }
    let mut matches = out.matches;
    for m in &mut matches {
        m.weight = weights[m.treated_idx];
    }

    let balance_after = balance_table(&x, &covariate_names, &is_treated, &weights);
    let wt: Vec<f64> = (0..n)
        .filter(|&i| is_treated[i])
        .map(|i| weights[i])
        .collect();
    let wc: Vec<f64> = (0..n)
        .filter(|&i| !is_treated[i])
        .map(|i| weights[i])
        .collect();
    let (ess_treated, ess_control) = (kish_ess(&wt), kish_ess(&wc));

    let caliper_used = match &method {
        MatchMethod::NearestNeighbor { caliper, .. } => *caliper,
        _ => None,
    };
    let mut warnings = Vec::new();
    let mut propensity_convergence = None;
    if let Some(m) = ps_model {
        warnings.extend(m.warnings);
        propensity_convergence = Some(m.convergence);
    }
    warnings.extend(out.warnings);

    Ok(MatchResult {
        method,
        distance: distance_method,
        balance_before,
        balance_after,
        matches,
        subclasses: out.subclasses,
        weights,
        propensity_scores: scores,
        n_obs: n,
        n_treated,
        n_control,
        n_matched_treated,
        n_matched_control,
        n_discarded_treated: n_treated - n_matched_treated,
        n_discarded_control: n_control - n_matched_control,
        caliper_used,
        effective_sample_size: ess_treated + ess_control,
        propensity_convergence,
        warnings,
        m_order: out.m_order,
        subclass: out.subclass,
        caliper_abs: out.caliper_abs,
        total_distance: out.total_distance,
        ess_treated,
        ess_control,
    })
}

/// S1 step 3: the caliper, the full-matching bounds, the cut points, `m_order` and the
/// subclassification distance.
fn validate_match_options(
    method: &MatchMethod,
    distance: DistanceMethod,
    has_ps: bool,
    options: &MatchOptions,
    covariate_cols: &[&str],
) -> EconResult<()> {
    let spec = |message: String| Err(EconError::InvalidSpecification { message });
    let dname = DistanceMethod::choices().canonical(distance);
    match method {
        MatchMethod::NearestNeighbor { caliper, .. } => {
            if let Some(c) = *caliper {
                if !(c.is_finite() && c > 0.0) {
                    return spec(format!(
                        "caliper must be a positive, finite number (in standard deviations of \
                         the propensity score for logit/probit, in distance units for \
                         mahalanobis/euclidean), got {c}"
                    ));
                }
            }
            if let Some(o @ (MOrder::Largest | MOrder::Smallest)) = options.m_order {
                if !has_ps {
                    return spec(format!(
                        "m_order = '{o}' orders treated units by their propensity score; with \
                         distance = '{dname}' use 'data' or 'random' (as MatchIt)"
                    ));
                }
            }
        }
        MatchMethod::Full {
            min_ratio,
            max_ratio,
        } => check_full_bounds(*min_ratio, *max_ratio)?,
        MatchMethod::Subclass { .. } if !has_ps => {
            return spec(format!(
                "subclassification needs a propensity-score distance (logit or probit), got \
                 {dname}"
            ));
        }
        MatchMethod::CoarsenedExact {
            cutpoints: Some(cuts),
            ..
        } => {
            for (name, c) in covariate_cols.iter().zip(cuts) {
                if c.iter().any(|v| !v.is_finite()) {
                    return spec(format!(
                        "cutpoints for covariate '{name}' must be finite numbers"
                    ));
                }
            }
        }
        _ => {}
    }
    Ok(())
}

/// S3-S5: nearest-neighbour matching on the propensity score or a covariate distance.
#[allow(clippy::too_many_arguments)]
fn nearest_neighbour(
    x: &Array2<f64>,
    is_treated: &[bool],
    scores: Option<&[f64]>,
    distance: DistanceMethod,
    ratio: usize,
    caliper: Option<f64>,
    replace: bool,
    options: &MatchOptions,
) -> EconResult<MethodOutput> {
    let n = is_treated.len();
    let (n_t, n_c) = (
        is_treated.iter().filter(|&&t| t).count(),
        is_treated.iter().filter(|&&t| !t).count(),
    );
    let order_kind = if replace {
        MOrder::Data
    } else {
        options.m_order.unwrap_or(if scores.is_some() {
            MOrder::Largest
        } else {
            MOrder::Data
        })
    };
    let order = treated_order(
        is_treated,
        scores,
        order_kind,
        options.seed.unwrap_or(DEFAULT_MATCH_SEED),
    );
    let mut warnings = Vec::new();
    if let Some(o) = options.m_order.filter(|&o| replace && o != MOrder::Data) {
        warnings.push(format!(
            "m_order = '{o}' has no effect with replacement; treated units are matched in data \
             order (as MatchIt)"
        ));
    }
    if !replace {
        if n_c < n_t {
            warnings.push(
                "Fewer control units than treated units; not all treated units will get a match."
                    .to_string(),
            );
        } else if n_c < ratio.saturating_mul(n_t) {
            warnings.push(format!("Not all treated units will get {ratio} matches."));
        }
    }
    let (matched, caliper_abs, dist): (
        Vec<Vec<usize>>,
        Option<f64>,
        Box<dyn Fn(usize, usize) -> f64 + '_>,
    ) = match scores {
        Some(s) => {
            let caliper_abs = caliper.map(|c| c * sample_sd(s));
            let m = nn_on_scores(s, is_treated, &order, ratio, caliper_abs, replace)?;
            (m, caliper_abs, Box::new(move |t, c| (s[t] - s[c]).abs()))
        }
        None => {
            let metric = CovariateMetric::new(
                x,
                is_treated,
                distance == DistanceMethod::Mahalanobis,
                &mut warnings,
            )?;
            let m = if metric.z.ncols() == 1 {
                // matchit2nearest: one covariate is matched as a scalar distance
                let z = metric.z.column(0).to_vec();
                nn_on_scores(&z, is_treated, &order, ratio, caliper, replace)?
            } else {
                nn_on_covariates(&metric, is_treated, &order, ratio, caliper, replace)?
            };
            (m, caliper, Box::new(move |t, c| metric.distance(t, c)))
        }
    };
    let mut subclass = vec![None; n];
    let mut label = 0;
    let mut matches = Vec::with_capacity(n_t);
    for t in (0..n).filter(|&i| is_treated[i]) {
        let m = &matched[t];
        if !replace && !m.is_empty() {
            label += 1;
            subclass[t] = Some(label);
            for &c in m {
                subclass[c] = Some(label);
            }
        }
        matches.push(MatchInfo {
            treated_idx: t,
            control_indices: m.clone(),
            distances: m.iter().map(|&c| dist(t, c)).collect(),
            weight: 0.0,
        });
    }
    Ok(MethodOutput {
        weights: weights_from_matches(&matched, is_treated),
        matches,
        subclass,
        subclasses: None,
        m_order: Some(order_kind),
        caliper_abs,
        total_distance: None,
        warnings,
    })
}

/// R's `sd`: the sample standard deviation (divisor `n - 1`).
fn sample_sd(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    (x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (n - 1.0)).sqrt()
}

/// Check the method's count parameters against the data before any work.
///
/// `ratio` must lie in `1..=n_control` (MatchIt only warns above the number of
/// controls; here it would size every treated unit's control list),
/// `n_subclasses` in `1..=n_treated`, CEM `n_bins` in `1..=n`, and custom CEM
/// cutpoints need one vector per covariate. Nearest-neighbour matching may
/// return up to `ratio` controls per treated unit, so that total is checked
/// against the allocation ceiling as well.
fn validate_match_counts(
    method: &MatchMethod,
    n: usize,
    n_treated: usize,
    n_control: usize,
    n_covariates: usize,
) -> EconResult<()> {
    let spec = |message: String| Err(EconError::InvalidSpecification { message });
    match method {
        MatchMethod::NearestNeighbor { ratio, .. } => {
            let ratio = *ratio;
            if !(1..=n_control).contains(&ratio) {
                return spec(format!(
                    "ratio = {ratio} is out of range: each treated unit can be matched to between \
                     1 and {n_control} control units (the number of controls in the data)."
                ));
            }
            let pairs = crate::resource::checked_len(
                &[n_treated, ratio],
                "matched control lists (ratio controls per treated unit)",
            )?;
            crate::resource::check_alloc::<(usize, f64)>(
                pairs,
                "matched control lists (ratio controls per treated unit)",
            )?;
        }
        MatchMethod::Subclass { n_subclasses } => {
            let k = *n_subclasses;
            if !(1..=n_treated).contains(&k) {
                return spec(format!(
                    "n_subclasses = {k} is out of range: use between 1 and {n_treated} subclasses \
                     (the number of treated units)."
                ));
            }
        }
        MatchMethod::CoarsenedExact { cutpoints, n_bins } => {
            if let Some(k) = *n_bins {
                if !(1..=n).contains(&k) {
                    return spec(format!(
                        "n_bins = {k} is out of range: use between 1 and {n} bins"
                    ));
                }
            }
            if let Some(c) = cutpoints {
                if c.len() != n_covariates {
                    return spec(format!(
                        "cutpoints must contain one vector per covariate ({n_covariates} \
                         covariates, got {})",
                        c.len()
                    ));
                }
            }
        }
        // min_ratio/max_ratio cannot cause unbounded work (CAUS-70 owns them).
        MatchMethod::Full { .. } => {}
    }
    Ok(())
}

/// Check the total size of the per-treated copies of stratum control lists
/// before they are built: every treated unit of a stratum gets its own copy
/// of the stratum's controls, one `T` per control. `strata` yields
/// `(n_treated, n_control)` of each kept stratum. `MatchInfo` is public and
/// keeps this shape; only its size is bounded here (G7).
fn check_stratum_control_lists<T>(
    strata: impl IntoIterator<Item = (usize, usize)>,
) -> EconResult<()> {
    let overflow = || {
        EconError::ResourceLimit(
            "matched control lists: (treated, control) pair count overflows".to_string(),
        )
    };
    let mut total: usize = 0;
    for (n_t, n_c) in strata {
        let pairs = n_t.checked_mul(n_c).ok_or_else(overflow)?;
        total = total.checked_add(pairs).ok_or_else(overflow)?;
    }
    crate::resource::check_alloc::<T>(
        total,
        "matched control lists (one copy of the stratum's controls per treated unit)",
    )
}

// ═══════════════════════════════════════════════════════════════════════════════
// Convenience Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Perform nearest neighbor 1:1 propensity score matching.
///
/// Convenience function for the most common matching scenario.
///
/// # Arguments
/// * `dataset` - Dataset containing treatment and covariates
/// * `treatment_col` - Name of binary treatment column
/// * `covariate_cols` - Names of matching covariates
/// * `caliper` - Optional caliper (in SD of propensity score)
/// * `replace` - Whether to sample with replacement
///
/// # Example
/// ```ignore
/// let result = nearest_neighbor_match(&dataset, "treatment", &["x1", "x2"],
///                                     Some(0.2), false)?;
/// ```
pub fn nearest_neighbor_match(
    dataset: &Dataset,
    treatment_col: &str,
    covariate_cols: &[&str],
    caliper: Option<f64>,
    replace: bool,
) -> EconResult<MatchResult> {
    let method = MatchMethod::NearestNeighbor {
        ratio: 1,
        caliper,
        replace,
    };
    match_it(
        dataset,
        treatment_col,
        covariate_cols,
        method,
        Some(DistanceMethod::Logit),
    )
}

/// Perform coarsened exact matching (CEM).
///
/// CEM creates coarsened bins for each covariate and matches exactly within
/// strata defined by the cross-product of bins.
///
/// # Arguments
/// * `dataset` - Dataset containing treatment and covariates
/// * `treatment_col` - Name of binary treatment column
/// * `covariate_cols` - Names of matching covariates
///
/// # Example
/// ```ignore
/// let result = cem_match(&dataset, "treatment", &["age", "income"])?;
/// ```
pub fn cem_match(
    dataset: &Dataset,
    treatment_col: &str,
    covariate_cols: &[&str],
) -> EconResult<MatchResult> {
    let method = MatchMethod::CoarsenedExact {
        cutpoints: None,
        n_bins: None,
    };
    match_it(dataset, treatment_col, covariate_cols, method, None)
}

/// Optimal full matching on a logit propensity score with 0.5 to 2 controls per treated unit
/// (`matchit(method = "full", min.controls = 0.5, max.controls = 2)`).
///
/// Every matched set is one treated unit with controls or one control with treated units,
/// and the total within-set distance `|ps_t - ps_c|` is minimal ([`optimal_full_match`]).
/// Problems above [`FULL_MATCH_MAX_PAIRS`] treated-control pairs are refused.
///
/// # Arguments
/// * `dataset` - Dataset containing treatment and covariates
/// * `treatment_col` - Name of binary treatment column
/// * `covariate_cols` - Names of matching covariates
///
/// # Example
/// ```ignore
/// let result = full_match(&dataset, "treatment", &["x1", "x2"])?;
/// ```
pub fn full_match(
    dataset: &Dataset,
    treatment_col: &str,
    covariate_cols: &[&str],
) -> EconResult<MatchResult> {
    let method = MatchMethod::Full {
        min_ratio: 0.5,
        max_ratio: 2.0,
    };
    match_it(
        dataset,
        treatment_col,
        covariate_cols,
        method,
        Some(DistanceMethod::Logit),
    )
}

/// Perform propensity score subclassification.
///
/// Creates propensity score subclasses (strata) for within-stratum comparisons.
///
/// # Arguments
/// * `dataset` - Dataset containing treatment and covariates
/// * `treatment_col` - Name of binary treatment column
/// * `covariate_cols` - Names of matching covariates
/// * `n_subclasses` - Number of subclasses to create
///
/// # Example
/// ```ignore
/// let result = subclass_match(&dataset, "treatment", &["x1", "x2"], 5)?;
/// ```
pub fn subclass_match(
    dataset: &Dataset,
    treatment_col: &str,
    covariate_cols: &[&str],
    n_subclasses: usize,
) -> EconResult<MatchResult> {
    let method = MatchMethod::Subclass { n_subclasses };
    match_it(
        dataset,
        treatment_col,
        covariate_cols,
        method,
        Some(DistanceMethod::Logit),
    )
}

// ═══════════════════════════════════════════════════════════════════════════════
// Propensity Score Estimation
// ═══════════════════════════════════════════════════════════════════════════════

/// A propensity-score model: the binomial glm of the treatment on `[1, x]`.
struct PropensityModel {
    scores: Array1<f64>,
    coefficients: Array1<f64>,
    convergence: Convergence,
    warnings: Vec<String>,
}

/// Propensity scores from `econometrics::glm::fit_propensity` (the logit or probit MLE of
/// `glm.fit`, as MatchIt's `distance = "glm"`); no clipping (MatchIt does not clip).
fn estimate_propensity(
    x: &Array2<f64>,
    treat: &Array1<f64>,
    link: GlmLink,
) -> EconResult<PropensityModel> {
    let (n, k) = x.dim();
    let mut design = crate::resource::try_zeros2(n, k + 1, "propensity-score design matrix")?;
    design.column_mut(0).fill(1.0);
    design.slice_mut(ndarray::s![.., 1..]).assign(x);
    let fit = fit_propensity(design.view(), treat.view(), link)?;
    // the record of the fit that ran (its max_iter and epsilon); the non-canonical links
    // also refine until the coefficients settle (fit_propensity)
    let mut convergence = fit.fit.notes_convergence().unwrap_or_else(|| {
        Convergence::new(
            fit.fit.converged,
            fit.fit.iterations,
            fit.fit.max_iter,
            "relative change in deviance",
            f64::NAN,
            fit.fit.epsilon,
        )
    });
    if link != GlmLink::Logit {
        convergence.criterion = format!(
            "relative change in deviance, then relative coefficient change < {}",
            crate::econometrics::glm::PROPENSITY_COEF_TOL
        );
    }
    let mut warnings: Vec<String> = convergence
        .note("propensity-score glm")
        .into_iter()
        .collect();
    warnings.extend(
        fit.fit
            .warnings
            .iter()
            .filter(|w| w.as_str() != "glm.fit: algorithm did not converge")
            .map(|w| format!("propensity-score glm: {w}")),
    );
    Ok(PropensityModel {
        scores: fit.scores,
        coefficients: fit.fit.coefficients,
        convergence,
        warnings,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Nearest-neighbour matching (MatchIt matchit2nearest / nn_matchC)
// ═══════════════════════════════════════════════════════════════════════════════

/// Treated rows in the order they choose their controls (S3). `Largest`/`Smallest` sort
/// stably by score (ties keep data order, as R's `order`); `Random` shuffles the data order
/// with a seeded ChaCha8 generator.
fn treated_order(
    is_treated: &[bool],
    scores: Option<&[f64]>,
    order: MOrder,
    seed: u64,
) -> Vec<usize> {
    let mut rows: Vec<usize> = (0..is_treated.len()).filter(|&i| is_treated[i]).collect();
    match (order, scores) {
        (MOrder::Largest, Some(s)) => rows.sort_by(|&a, &b| s[b].total_cmp(&s[a])),
        (MOrder::Smallest, Some(s)) => rows.sort_by(|&a, &b| s[a].total_cmp(&s[b])),
        (MOrder::Random, _) => {
            use rand::SeedableRng;
            use rand::seq::SliceRandom;
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
            rows.shuffle(&mut rng);
        }
        _ => {}
    }
    rows
}

/// Disjoint-set "skip" pointers over sorted control positions: the nearest available
/// position at or below / at or above a position, with path compression.
struct Skip {
    /// `left[p + 1]` leads to the nearest available position `<= p`, encoded `+ 1` (0 = none).
    left: Vec<usize>,
    /// `right[p]` leads to the nearest available position `>= p` (`len` = none).
    right: Vec<usize>,
}

impl Skip {
    fn new(len: usize) -> EconResult<Skip> {
        let mut left = crate::resource::try_vec(len + 1, 0usize, "nearest-neighbour skip lists")?;
        let mut right = crate::resource::try_vec(len + 1, 0usize, "nearest-neighbour skip lists")?;
        for i in 0..=len {
            left[i] = i;
            right[i] = i;
        }
        Ok(Skip { left, right })
    }

    fn find(par: &mut [usize], mut x: usize) -> usize {
        let mut root = x;
        while par[root] != root {
            root = par[root];
        }
        while par[x] != root {
            let next = par[x];
            par[x] = root;
            x = next;
        }
        root
    }

    /// Nearest available position `< pos`.
    fn below(&mut self, pos: usize) -> Option<usize> {
        let r = Self::find(&mut self.left, pos);
        (r > 0).then(|| r - 1)
    }

    /// Nearest available position `>= pos`.
    fn above(&mut self, pos: usize) -> Option<usize> {
        let r = Self::find(&mut self.right, pos);
        (r < self.right.len() - 1).then_some(r)
    }

    fn mark_used(&mut self, p: usize) {
        self.left[p + 1] = p;
        self.right[p] = p + 1;
    }
}

/// Nearest-neighbour matching on a scalar distance (S4, MatchIt's `nn_matchC_vec` rules):
/// per row, the matched controls in match order (empty for controls and unmatched treated
/// units). Without replacement treated units match in `order` and in rounds (every unit
/// gets its r-th match before any gets its (r+1)-th); with replacement each treated unit
/// takes its `ratio` nearest controls. An exact tie between the nearest score below and the
/// nearest at or above goes below.
fn nn_on_scores(
    scores: &[f64],
    is_treated: &[bool],
    order: &[usize],
    ratio: usize,
    caliper_abs: Option<f64>,
    replace: bool,
) -> EconResult<Vec<Vec<usize>>> {
    let n = scores.len();
    let mut cr: Vec<usize> = (0..n).filter(|&i| !is_treated[i]).collect();
    cr.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
    let cs: Vec<f64> = cr.iter().map(|&i| scores[i]).collect();
    let nc = cs.len();
    let within = |d: f64| caliper_abs.is_none_or(|c| d <= c);
    let mut matched: Vec<Vec<usize>> = vec![Vec::new(); n];
    if replace {
        for &t in order {
            crate::resource::checkpoint()?;
            let st = scores[t];
            let pos = cs.partition_point(|&s| s < st);
            let (mut lo, mut hi) = (pos, pos);
            let mut got = Vec::with_capacity(ratio.min(nc));
            while got.len() < ratio && (lo > 0 || hi < nc) {
                let db = (lo > 0).then(|| st - cs[lo - 1]);
                let da = (hi < nc).then(|| cs[hi] - st);
                let take_below = match (db, da) {
                    (Some(b), Some(a)) => b <= a,
                    (Some(_), None) => true,
                    _ => false,
                };
                let d = if take_below { db } else { da }.unwrap_or(f64::INFINITY);
                if !within(d) {
                    break;
                }
                if take_below {
                    lo -= 1;
                    got.push(cr[lo]);
                } else {
                    got.push(cr[hi]);
                    hi += 1;
                }
            }
            matched[t] = got;
        }
        return Ok(matched);
    }
    let mut skip = Skip::new(nc)?;
    for r in 1..=ratio {
        for &t in order {
            if matched[t].len() != r - 1 {
                continue;
            }
            crate::resource::checkpoint()?;
            let st = scores[t];
            let pos = cs.partition_point(|&s| s < st);
            let below = skip.below(pos);
            let above = skip.above(pos);
            let pick = match (below, above) {
                (Some(b), Some(a)) => {
                    if st - cs[b] <= cs[a] - st {
                        b
                    } else {
                        a
                    }
                }
                (Some(b), None) => b,
                (None, Some(a)) => a,
                (None, None) => continue,
            };
            if !within((st - cs[pick]).abs()) {
                continue;
            }
            matched[t].push(cr[pick]);
            skip.mark_used(pick);
        }
    }
    Ok(matched)
}

/// Covariates prepared for a Mahalanobis or Euclidean distance: `z` such that the distance of
/// rows i and j is `||z_i - z_j||` (S5; whitened covariates for Mahalanobis, MatchIt's
/// `mahalanobize`).
struct CovariateMetric {
    z: Array2<f64>,
}

impl CovariateMetric {
    fn distance(&self, i: usize, j: usize) -> f64 {
        let (a, b) = (self.z.row(i), self.z.row(j));
        a.iter()
            .zip(b.iter())
            .map(|(u, v)| (u - v) * (u - v))
            .sum::<f64>()
            .sqrt()
    }

    /// The metric of `distance` on `x` (n x k), with MatchIt's `transform_covariates`:
    /// columns whose range is below `sqrt(eps)` are dropped, and the Mahalanobis metric uses
    /// the pooled within-group covariance. Warnings are pushed to `warnings`.
    fn new(
        x: &Array2<f64>,
        is_treated: &[bool],
        mahalanobis: bool,
        warnings: &mut Vec<String>,
    ) -> EconResult<CovariateMetric> {
        let (n, k) = x.dim();
        let tol = f64::EPSILON.sqrt();
        let keep: Vec<usize> = (0..k)
            .filter(|&j| {
                let c = x.column(j);
                let (lo, hi) = c
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                        (lo.min(v), hi.max(v))
                    });
                hi - lo >= tol
            })
            .collect();
        if keep.is_empty() {
            warnings.push("every matching covariate is constant; all distances are 0".into());
            return Ok(CovariateMetric {
                z: crate::resource::try_zeros2(n, 0, "matching covariates")?,
            });
        }
        let mut xk = crate::resource::try_zeros2(n, keep.len(), "matching covariates")?;
        for (c, &j) in keep.iter().enumerate() {
            xk.column_mut(c).assign(&x.column(j));
        }
        if !mahalanobis {
            return Ok(CovariateMetric { z: xk });
        }
        let s = pooled_within_covariance(&xk, is_treated)?;
        let what = "pooled within-group covariance of the matching covariates";
        let inv = sym_inverse(s.view(), SingularPolicy::Generalized, what)?;
        if let Some(w) = inv.warning(what) {
            warnings.push(w);
        }
        let (values, vectors) = eig_symmetric(&inv.inverse.view()).map_err(|e| {
            EconError::Computation(format!("eigen-decomposition of the {what}: {e:?}"))
        })?;
        let kk = keep.len();
        let mut root = Array2::<f64>::zeros((kk, kk));
        for c in 0..kk {
            let s = values[c].max(0.0).sqrt();
            for r in 0..kk {
                root[[r, c]] = vectors[[r, c]] * s;
            }
        }
        let mut z = crate::resource::try_zeros2(n, kk, "whitened matching covariates")?;
        ndarray::linalg::general_mat_mul(1.0, &xk, &root, 0.0, &mut z);
        Ok(CovariateMetric { z })
    }
}

/// MatchIt's `pooled_cov`: `sum_g sum_{i in g} (x_i - mean_g)(x_i - mean_g)' / (n - 2)`.
fn pooled_within_covariance(x: &Array2<f64>, is_treated: &[bool]) -> EconResult<Array2<f64>> {
    let (n, k) = x.dim();
    let denom = n
        .checked_sub(2)
        .filter(|&d| d > 0)
        .ok_or_else(|| EconError::InsufficientData {
            required: 3,
            provided: n,
            context: "pooled within-group covariance for the Mahalanobis distance".into(),
        })?;
    let mut means = [Array1::<f64>::zeros(k), Array1::<f64>::zeros(k)];
    let mut counts = [0usize; 2];
    for i in 0..n {
        let g = usize::from(is_treated[i]);
        counts[g] += 1;
        means[g] += &x.row(i);
    }
    for g in 0..2 {
        if counts[g] > 0 {
            means[g] /= counts[g] as f64;
        }
    }
    let mut s = Array2::<f64>::zeros((k, k));
    for i in 0..n {
        let g = usize::from(is_treated[i]);
        let d = &x.row(i) - &means[g];
        for a in 0..k {
            for b in 0..=a {
                s[[a, b]] += d[a] * d[b];
            }
        }
    }
    for a in 0..k {
        for b in 0..=a {
            let v = s[[a, b]] / denom as f64;
            s[[a, b]] = v;
            s[[b, a]] = v;
        }
    }
    Ok(s)
}

/// Nearest-neighbour matching on a covariate distance (S5): without replacement in rounds,
/// each treated unit takes the available control of smallest `(distance, row)`; with
/// replacement its `ratio` smallest. Same output as [`nn_on_scores`].
fn nn_on_covariates(
    metric: &CovariateMetric,
    is_treated: &[bool],
    order: &[usize],
    ratio: usize,
    caliper_abs: Option<f64>,
    replace: bool,
) -> EconResult<Vec<Vec<usize>>> {
    let n = is_treated.len();
    let controls: Vec<usize> = (0..n).filter(|&i| !is_treated[i]).collect();
    let within = |d: f64| caliper_abs.is_none_or(|c| d <= c);
    let key = |a: &(f64, usize), b: &(f64, usize)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
    let mut matched: Vec<Vec<usize>> = vec![Vec::new(); n];
    if replace {
        let one = |t: usize| -> Vec<usize> {
            let mut d: Vec<(f64, usize)> = controls
                .iter()
                .map(|&c| (metric.distance(t, c), c))
                .collect();
            let k = ratio.min(d.len());
            if k < d.len() {
                d.select_nth_unstable_by(k, key);
                d.truncate(k);
            }
            d.sort_by(key);
            d.into_iter()
                .take_while(|p| within(p.0))
                .map(|p| p.1)
                .collect()
        };
        if order.len() >= PARALLEL_THRESHOLD {
            let budget = crate::resource::current_budget();
            let out: EconResult<Vec<Vec<usize>>> = order
                .par_iter()
                .map(|&t| {
                    if let Some(b) = &budget {
                        b.check()?;
                    }
                    Ok(one(t))
                })
                .collect();
            for (&t, m) in order.iter().zip(out?) {
                matched[t] = m;
            }
        } else {
            for &t in order {
                crate::resource::checkpoint()?;
                matched[t] = one(t);
            }
        }
        return Ok(matched);
    }
    let mut available = controls;
    for r in 1..=ratio {
        for &t in order {
            if matched[t].len() != r - 1 || available.is_empty() {
                continue;
            }
            crate::resource::checkpoint()?;
            let best = if available.len() >= PARALLEL_THRESHOLD {
                let budget = crate::resource::current_budget();
                if let Some(b) = &budget {
                    b.check()?;
                }
                available
                    .par_iter()
                    .enumerate()
                    .map(|(pos, &c)| (metric.distance(t, c), c, pos))
                    .min_by(|a, b| key(&(a.0, a.1), &(b.0, b.1)))
            } else {
                available
                    .iter()
                    .enumerate()
                    .map(|(pos, &c)| (metric.distance(t, c), c, pos))
                    .min_by(|a, b| key(&(a.0, a.1), &(b.0, b.1)))
            };
            let Some((d, c, pos)) = best else { continue };
            if !within(d) {
                continue;
            }
            matched[t].push(c);
            available.swap_remove(pos);
        }
    }
    Ok(matched)
}

/// MatchIt's match-matrix weights (S6, `weights_matrixC`): 1 for a treated unit with at least
/// one match, and for a control the sum of `1 / m_t` over the treated units `t` it is matched
/// to (`m_t` their number of matches); normalised per group.
fn weights_from_matches(matched: &[Vec<usize>], is_treated: &[bool]) -> Vec<f64> {
    let mut w = vec![0.0; is_treated.len()];
    for (t, m) in matched.iter().enumerate() {
        if !is_treated[t] || m.is_empty() {
            continue;
        }
        w[t] = 1.0;
        let share = 1.0 / m.len() as f64;
        for &c in m {
            w[c] += share;
        }
    }
    normalize_weights(&mut w, is_treated);
    w
}

/// MatchIt's `.make_sum_to_n`: within each group, the positive weights are scaled to sum to
/// their count.
fn normalize_weights(w: &mut [f64], is_treated: &[bool]) {
    for g in [true, false] {
        let (mut sum, mut count) = (0.0, 0usize);
        for (wi, &t) in w.iter().zip(is_treated) {
            if t == g && *wi > 0.0 {
                sum += *wi;
                count += 1;
            }
        }
        if count > 0 && sum > 0.0 {
            let f = count as f64 / sum;
            for (wi, &t) in w.iter_mut().zip(is_treated) {
                if t == g && *wi > 0.0 {
                    *wi *= f;
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stratified methods
// ═══════════════════════════════════════════════════════════════════════════════

fn group_indices(is_treated: &[bool]) -> (Vec<usize>, Vec<usize>) {
    (0..is_treated.len()).partition(|&i| is_treated[i])
}

/// Row labels from a list of subclasses.
fn sets_from_subclasses(n: usize, sc: &Option<Vec<SubclassInfo>>) -> Vec<Option<usize>> {
    let mut sets = vec![None; n];
    for s in sc.iter().flatten() {
        for &i in s.treated_indices.iter().chain(&s.control_indices) {
            sets[i] = Some(s.subclass_id);
        }
    }
    sets
}

/// S7: coarsened exact matching (MatchIt's `cem_matchit`): strata of the coded covariates,
/// numbered by first appearance among those holding both groups.
fn cem_method(
    x: &Array2<f64>,
    is_treated: &[bool],
    cutpoints: Option<&[Vec<f64>]>,
    n_bins: Option<usize>,
) -> EconResult<MethodOutput> {
    let (n, k) = x.dim();
    let breaks: Vec<Vec<f64>> = (0..k)
        .map(|j| {
            let col = x.column(j).to_vec();
            cem_breaks(&col, cutpoints.map(|c| c[j].as_slice()), n_bins)
        })
        .collect();
    let mut ids: HashMap<Vec<usize>, usize> = HashMap::new();
    let mut stratum_of = vec![0usize; n];
    let mut counts: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        let key: Vec<usize> = (0..k)
            .map(|j| find_interval(&breaks[j], x[[i, j]]))
            .collect();
        let next = ids.len();
        let id = *ids.entry(key).or_insert(next);
        if id == counts.len() {
            counts.push((0, 0));
        }
        if is_treated[i] {
            counts[id].0 += 1;
        } else {
            counts[id].1 += 1;
        }
        stratum_of[i] = id;
    }
    let mut label = vec![None; counts.len()];
    let mut kept = 0;
    for (id, &(t, c)) in counts.iter().enumerate() {
        if t > 0 && c > 0 {
            kept += 1;
            label[id] = Some(kept);
        }
    }
    let subclass: Vec<Option<usize>> = stratum_of.iter().map(|&id| label[id]).collect();
    stratified_output(is_treated, subclass, None, None, Vec::new())
}

/// The output of a partition (CEM, subclassification, full matching): MatchIt's ATT weights,
/// the subclass list (`ps_range` from `ranges`, or NaN), and one `MatchInfo` per treated unit
/// in data order listing the controls of its set (with `distance(t, c)` when given).
fn stratified_output(
    is_treated: &[bool],
    subclass: Vec<Option<usize>>,
    ranges: Option<&dyn Fn(usize, &[usize], &[usize]) -> (f64, f64)>,
    distance: Option<&dyn Fn(usize, usize) -> f64>,
    warnings: Vec<String>,
) -> EconResult<MethodOutput> {
    let n_sets = subclass.iter().flatten().copied().max().unwrap_or(0);
    let mut treated: Vec<Vec<usize>> = vec![Vec::new(); n_sets];
    let mut controls: Vec<Vec<usize>> = vec![Vec::new(); n_sets];
    for (i, s) in subclass.iter().enumerate() {
        if let Some(s) = s {
            if is_treated[i] {
                treated[s - 1].push(i);
            } else {
                controls[s - 1].push(i);
            }
        }
    }
    check_stratum_control_lists::<usize>(
        (0..n_sets).map(|s| (treated[s].len(), controls[s].len())),
    )?;
    let weights = weights_from_sets(&subclass, is_treated);
    let mut matches = Vec::new();
    for t in (0..is_treated.len()).filter(|&i| is_treated[i]) {
        let control_indices = subclass[t].map_or_else(Vec::new, |s| controls[s - 1].clone());
        let distances = match distance {
            Some(d) => control_indices.iter().map(|&c| d(t, c)).collect(),
            None => Vec::new(),
        };
        matches.push(MatchInfo {
            treated_idx: t,
            control_indices,
            distances,
            weight: 0.0,
        });
    }
    let subclasses: Vec<SubclassInfo> = treated
        .into_iter()
        .zip(controls)
        .enumerate()
        .filter(|(_, (t, c))| !t.is_empty() || !c.is_empty())
        .map(|(s, (t, c))| SubclassInfo {
            subclass_id: s + 1,
            ps_range: ranges.map_or((f64::NAN, f64::NAN), |r| r(s + 1, &t, &c)),
            treated_indices: t,
            control_indices: c,
        })
        .collect();
    Ok(MethodOutput {
        weights,
        matches,
        subclass,
        subclasses: Some(subclasses),
        m_order: None,
        caliper_abs: None,
        total_distance: None,
        warnings,
    })
}

/// S9: optimal full matching ([`optimal_full_match`]) on `|ps_t - ps_c|` (the score scale, as
/// MatchIt) or the S5 covariate metric; sets numbered by their smallest data row.
fn full_method(
    x: &Array2<f64>,
    is_treated: &[bool],
    scores: Option<&[f64]>,
    distance: DistanceMethod,
    min_ratio: f64,
    max_ratio: f64,
) -> EconResult<MethodOutput> {
    let n = is_treated.len();
    let (t, c) = group_indices(is_treated);
    check_full_pairs(t.len(), c.len())?;
    let mut warnings = Vec::new();
    let dist: Box<dyn Fn(usize, usize) -> f64> = match scores {
        Some(s) => Box::new(move |i, j| (s[i] - s[j]).abs()),
        None => {
            let m = CovariateMetric::new(
                x,
                is_treated,
                distance == DistanceMethod::Mahalanobis,
                &mut warnings,
            )?;
            Box::new(move |i, j| m.distance(i, j))
        }
    };
    let mut dm = crate::resource::try_zeros2(t.len(), c.len(), "full-matching distance matrix")?;
    for (r, &ti) in t.iter().enumerate() {
        crate::resource::checkpoint()?;
        for (k, &ci) in c.iter().enumerate() {
            dm[[r, k]] = dist(ti, ci);
        }
    }
    let sol = optimal_full_match(dm.view(), min_ratio, max_ratio)?;
    drop(dm);
    let mut set_of = vec![None; n];
    for (r, s) in sol.row_sets.iter().enumerate() {
        set_of[t[r]] = *s;
    }
    for (k, s) in sol.col_sets.iter().enumerate() {
        set_of[c[k]] = *s;
    }
    let mut label = vec![None; sol.n_sets];
    let mut next = 0;
    let subclass: Vec<Option<usize>> = set_of
        .iter()
        .map(|s| {
            s.map(|s| {
                *label[s].get_or_insert_with(|| {
                    next += 1;
                    next
                })
            })
        })
        .collect();
    if let Some(f) = sol.omit_fraction {
        let u = sol.col_sets.iter().filter(|s| s.is_none()).count();
        warnings.push(format!(
            "max_ratio = {max_ratio} cannot absorb all {} control units; as optmatch::fullmatch, \
             {u} control units are left unmatched (omit fraction {f:.6})",
            c.len()
        ));
    }
    let ps_range = |_: usize, ts: &[usize], cs: &[usize]| match scores {
        Some(sc) => ts
            .iter()
            .chain(cs)
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &i| {
                (lo.min(sc[i]), hi.max(sc[i]))
            }),
        None => (f64::NAN, f64::NAN),
    };
    let mut out = stratified_output(
        is_treated,
        subclass,
        Some(&ps_range),
        Some(&*dist),
        warnings,
    )?;
    out.total_distance = Some(sol.total_distance);
    Ok(out)
}

/// S8: propensity-score subclassification (`matchit2subclass`, `min.n = 0`); `ps_range` of a
/// subclass is its pair of cut points.
fn subclass_method(is_treated: &[bool], scores: &[f64], k: usize) -> EconResult<MethodOutput> {
    let (labels, q, warnings) = subclass_labels(scores, is_treated, k);
    // the original subclass (before discarded ones were skipped) of every kept label
    let mut orig = vec![0usize; k + 1];
    for (l, &sc) in labels.iter().zip(scores) {
        if let Some(l) = l {
            orig[*l] = find_interval(&q, sc).clamp(1, k);
        }
    }
    let ranges = |s: usize, _: &[usize], _: &[usize]| (q[orig[s] - 1], q[orig[s]]);
    stratified_output(is_treated, labels, Some(&ranges), None, warnings)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Coarsened exact matching and subclassification (MatchIt cem_matchit, matchit2subclass)
// ═══════════════════════════════════════════════════════════════════════════════

/// R's `findInterval(x, breaks)`: the number of breaks `<= x` (left-closed intervals).
fn find_interval(breaks: &[f64], x: f64) -> usize {
    breaks.partition_point(|&b| b <= x)
}

/// `[-inf] + sorted unique(cuts) + [+inf]`.
fn padded_breaks(mut cuts: Vec<f64>) -> Vec<f64> {
    cuts.sort_by(f64::total_cmp);
    cuts.dedup();
    let mut b = Vec::with_capacity(cuts.len() + 2);
    b.push(f64::NEG_INFINITY);
    b.extend(cuts);
    b.push(f64::INFINITY);
    b
}

/// The CEM breaks of one covariate (S7): explicit cut points, MatchIt's `"q<k>"` (type-7
/// quantiles at `1/k, ..., (k-1)/k`) or, by default, `nclass.Sturges` equal-width bins.
fn cem_breaks(col: &[f64], cutpoints: Option<&[f64]>, n_bins: Option<usize>) -> Vec<f64> {
    if let Some(c) = cutpoints {
        return padded_breaks(c.to_vec());
    }
    let mut sorted = col.to_vec();
    sorted.sort_by(f64::total_cmp);
    match n_bins {
        Some(k) => {
            let step = 1.0 / k as f64;
            let cuts = (0..k.saturating_sub(1))
                .map(|i| quantile_sorted(&sorted, step + i as f64 * step, QuantileType::T7))
                .collect();
            padded_breaks(cuts)
        }
        None => {
            let n = col.len();
            let k = ((n as f64).log2() + 1.0).ceil().max(1.0) as usize;
            let (lo, hi) = (sorted[0], sorted[n - 1]);
            let by = (hi - lo) / k as f64;
            let mut b = Vec::with_capacity(k + 1);
            b.push(f64::NEG_INFINITY);
            b.extend((1..k).map(|i| lo + i as f64 * by));
            b.push(f64::INFINITY);
            b
        }
    }
}

/// MatchIt's ATT weights from a partition (S6, `get_weights_from_subclass`): a treated unit in
/// a set gets 1, a control `n_t(s) / n_c(s)`, an unmatched unit 0; normalised per group.
fn weights_from_sets(sets: &[Option<usize>], is_treated: &[bool]) -> Vec<f64> {
    let mut counts: HashMap<usize, (usize, usize)> = HashMap::new();
    for (s, &t) in sets.iter().zip(is_treated) {
        if let Some(s) = s {
            let e = counts.entry(*s).or_insert((0, 0));
            if t {
                e.0 += 1;
            } else {
                e.1 += 1;
            }
        }
    }
    let mut w: Vec<f64> = sets
        .iter()
        .zip(is_treated)
        .map(|(s, &t)| match s {
            None => 0.0,
            Some(s) => {
                let (nt, nc) = counts[s];
                if nt == 0 || nc == 0 {
                    0.0
                } else if t {
                    1.0
                } else {
                    nt as f64 / nc as f64
                }
            }
        })
        .collect();
    normalize_weights(&mut w, is_treated);
    w
}

/// Subclass labels (1-based) of `matchit2subclass` with `min.n = 0` (S8): cut points at the
/// type-7 quantiles of the treated scores at `seq(0, 1, length.out = k + 1)`, assignment by
/// `findInterval(.., all.inside = TRUE)`; subclasses without treated or control units are
/// left unmatched and the rest renumbered. Returns the labels, the cut points and warnings.
fn subclass_labels(
    scores: &[f64],
    is_treated: &[bool],
    k: usize,
) -> (Vec<Option<usize>>, Vec<f64>, Vec<String>) {
    let mut st: Vec<f64> = scores
        .iter()
        .zip(is_treated)
        .filter(|(_, t)| **t)
        .map(|(s, _)| *s)
        .collect();
    st.sort_by(f64::total_cmp);
    let step = 1.0 / k as f64;
    let q: Vec<f64> = (0..=k)
        .map(|i| {
            let p = if i == k { 1.0 } else { i as f64 * step };
            quantile_sorted(&st, p, QuantileType::T7)
        })
        .collect();
    let raw: Vec<usize> = scores
        .iter()
        .map(|&s| find_interval(&q, s).clamp(1, k))
        .collect();
    let mut warnings = Vec::new();
    let mut nt = vec![0usize; k + 1];
    let mut nc = vec![0usize; k + 1];
    for (&l, &t) in raw.iter().zip(is_treated) {
        if t {
            nt[l] += 1;
        } else {
            nc[l] += 1;
        }
    }
    let distinct = (1..=k).filter(|&l| nt[l] + nc[l] > 0).count();
    if distinct < k {
        warnings.push(format!(
            "due to discreteness in the propensity score, fewer subclasses were generated than \
             requested ({distinct} of {k})"
        ));
    }
    let mut relabel = vec![None; k + 1];
    let mut next = 0;
    for l in 1..=k {
        if nt[l] + nc[l] == 0 {
            continue;
        }
        if nt[l] == 0 || nc[l] == 0 {
            let (group, m) = if nt[l] == 0 {
                ("treated", nc[l])
            } else {
                ("control", nt[l])
            };
            warnings.push(format!(
                "subclass {l} holds no {group} units; its {m} units are left unmatched \
                 (MatchIt's default min.n = 1 would move units from an adjacent subclass instead)"
            ));
            continue;
        }
        next += 1;
        relabel[l] = Some(next);
    }
    let labels = raw.iter().map(|&l| relabel[l]).collect();
    (labels, q, warnings)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Optimal full matching (optmatch::fullmatch)
// ═══════════════════════════════════════════════════════════════════════════════

/// Largest treated x control problem optimal full matching accepts (optmatch's default
/// maximum problem size), in every mode. The dense problem needs about
/// [`FULL_MATCH_BYTES_PER_PAIR`] bytes per pair (the network's arcs, the simplex's arc state
/// and the distance matrix), so the peak at this limit is about 1 GB.
pub const FULL_MATCH_MAX_PAIRS: usize = 10_000_000;

/// Approximate peak bytes per treated-control pair of optimal full matching: 32 in the
/// `Network` arc (tail, head, capacity, cost), about 65 in `network_simplex` (source and
/// target, capacity, cost and flow, state) and 8 in the distance matrix.
pub const FULL_MATCH_BYTES_PER_PAIR: usize = 105;

/// An optimal full matching of the rows (treated) and columns (controls) of a distance matrix.
#[derive(Debug, Clone)]
pub struct FullMatchSolution {
    /// (row, column) pairs joined by a unit of flow, rows ascending, then columns ascending.
    pub pairs: Vec<(usize, usize)>,
    /// 0-based matched set of every row (sets numbered by first appearance, scanning rows then
    /// columns); None = unmatched (only after the max_controls recovery).
    pub row_sets: Vec<Option<usize>>,
    /// 0-based matched set of every column; None = unmatched.
    pub col_sets: Vec<Option<usize>>,
    /// Number of matched sets.
    pub n_sets: usize,
    /// Sum of the distances over `pairs` (the optmatch objective; every set is a star).
    pub total_distance: f64,
    /// optmatch's omit fraction when the recovery left controls unmatched.
    pub omit_fraction: Option<f64>,
    /// The problem was solved transposed (max_controls <= 0.5).
    pub flipped: bool,
    /// Network simplex pivots of the final solve.
    pub pivots: usize,
}

fn full_bounds_error(min_c: f64, max_c: f64) -> EconError {
    EconError::InvalidSpecification {
        message: format!(
            "min_ratio = {min_c} and max_ratio = {max_c} must satisfy 0 <= min_ratio <= max_ratio \
             and max_ratio > 0 (controls per treated unit, optmatch's min.controls / max.controls)"
        ),
    }
}

/// S1: finite `min_ratio >= 0`, `max_ratio > 0` (`+inf` allowed), `min_ratio <= max_ratio`.
fn check_full_bounds(min_c: f64, max_c: f64) -> EconResult<()> {
    if min_c.is_finite() && min_c >= 0.0 && !max_c.is_nan() && max_c > 0.0 && min_c <= max_c {
        Ok(())
    } else {
        Err(full_bounds_error(min_c, max_c))
    }
}

fn check_full_pairs(n_t: usize, n_c: usize) -> EconResult<()> {
    match n_t.checked_mul(n_c) {
        Some(p) if p <= FULL_MATCH_MAX_PAIRS => Ok(()),
        p => {
            let pairs = p.map_or_else(|| "more than usize::MAX".to_string(), |p| p.to_string());
            let mb = p.map_or(f64::INFINITY, |p| {
                (FULL_MATCH_BYTES_PER_PAIR as f64 * p as f64 / 1e6).round()
            });
            Err(EconError::InvalidSpecification {
                message: format!(
                    "optimal full matching needs a dense {n_t} x {n_c} problem ({pairs} \
                     treated-control pairs, about {mb} MB); at most {FULL_MATCH_MAX_PAIRS} pairs \
                     are supported (optmatch's default maximum problem size); use nearest-\
                     neighbour or subclass matching, or match within subgroups"
                ),
            })
        }
    }
}

/// One `fmatch` problem (S10 steps 1-8): the pairs (in the original orientation) of the
/// optimal full matching with `min_c`/`max_c` controls per treated unit and omit fraction
/// `omf`, or None when it is infeasible.
fn solve_full(
    dist: ArrayView2<'_, f64>,
    min_c: f64,
    max_c: f64,
    omf: Option<f64>,
) -> EconResult<Option<(Vec<(usize, usize)>, bool, usize)>> {
    let flipped = match omf {
        None => max_c.is_nan() || max_c <= 0.5,
        Some(o) => o.is_nan() || o < 0.0,
    };
    let (nt, nc) = dist.dim();
    // rows x cols of the problem actually solved
    let (nr, ncl) = if flipped { (nc, nt) } else { (nt, nc) };
    let (maxc, minc, omf_c) = if flipped {
        (
            (1.0 / min_c).min(ncl as f64),
            (1.0 / max_c).max(1.0 / nr as f64),
            omf.map(|o| -o),
        )
    } else {
        (max_c.min(ncl as f64), min_c.max(1.0 / nr as f64), omf)
    };
    let f = 1.0 - omf_c.unwrap_or(0.0);
    if minc.floor() > maxc.ceil() || (1.0 / minc).ceil() < (1.0 / maxc).floor() {
        return Ok(None);
    }
    let mut mxr = (1.0 / minc).ceil();
    let mxc = maxc.ceil();
    let mnc = minc.floor().max(1.0);
    if mnc > 1.0 {
        mxr = 1.0;
    }
    let n_mc = (ncl as f64 * f).round_ties_even();
    let nr_f = nr as f64;
    if (mxr > 1.0 && nr_f / mxr > n_mc) || (mxr == 1.0 && nr_f * mnc > n_mc) || (nr_f * mxc < n_mc)
    {
        return Ok(None);
    }
    let at = |r: usize, c: usize| if flipped { dist[[c, r]] } else { dist[[r, c]] };
    let max_d = dist.iter().copied().fold(0.0, f64::max);
    let unit = max_d < f64::EPSILON.sqrt();
    let eps = max_d / (1u64 << 40) as f64;
    let cost = |d: f64| -> i64 {
        if unit {
            1
        } else {
            (0.5 + d / eps).ceil() as i64
        }
    };

    let (mxr, mxc, mnc, n_mc) = (mxr as i64, mxc as i64, mnc as i64, n_mc as i64);
    let pairs_n = nr * ncl;
    let n_arcs = pairs_n + nr + 2 * ncl;
    crate::resource::check_alloc::<(usize, usize, i64, i64)>(n_arcs, "full-matching network")?;
    let end = nr + ncl;
    let sink = end + 1;
    let mut g = Network::with_arc_capacity(nr + ncl + 2, n_arcs)?;
    for r in 0..nr {
        crate::resource::checkpoint()?;
        for c in 0..ncl {
            g.add_arc(r, nr + c, 1, cost(at(r, c)))?;
        }
    }
    for r in 0..nr {
        if mxc - mnc > 0 {
            g.add_arc(r, end, mxc - mnc, 0)?;
        }
        g.set_supply(r, mxc)?;
    }
    for c in 0..ncl {
        if mxr - 1 > 0 {
            g.add_arc(nr + c, end, mxr - 1, 0)?;
        }
        g.add_arc(nr + c, sink, 1, 0)?;
    }
    g.set_supply(end, -(mxc * nr as i64 - n_mc))?;
    g.set_supply(sink, -n_mc)?;
    let sol = network_simplex(&g)?;
    if sol.status == FlowStatus::Infeasible {
        return Ok(None);
    }
    let mut pairs = Vec::new();
    for r in 0..nr {
        for c in 0..ncl {
            if sol.flows[r * ncl + c] > 0 {
                pairs.push(if flipped { (c, r) } else { (r, c) });
            }
        }
    }
    pairs.sort_unstable();
    Ok(Some((pairs, flipped, sol.pivots)))
}

/// Optimal full matching of the rows (treated) and columns (controls) of a non-negative
/// distance matrix: `optmatch::fullmatch(dist, min.controls, max.controls)` solved exactly.
///
/// Every matched set is one treated unit with one or more controls or one control with one
/// or more treated units, with `min_controls <= controls / treated <= max_controls`, and the
/// sum of the distances within the sets is minimal (Hansen and Klopfer 2006): the network of
/// optmatch's `fmatch`, solved by [`network_simplex`] on integer costs at resolution
/// `max(dist) / 2^40`. As optmatch, `max_controls <= 0.5` is solved transposed, and when
/// `max_controls` cannot absorb every control the omit-fraction recovery leaves the controls
/// unmatched that the unconstrained solution cannot use. Refused: an empty or non-finite or
/// negative matrix, invalid bounds, more than [`FULL_MATCH_MAX_PAIRS`] entries, and
/// infeasible bounds.
pub fn optimal_full_match(
    dist: ArrayView2<'_, f64>,
    min_controls: f64,
    max_controls: f64,
) -> EconResult<FullMatchSolution> {
    let (nt, nc) = dist.dim();
    if nt == 0 || nc == 0 {
        return Err(EconError::InvalidSpecification {
            message: format!("the distance matrix must have rows and columns, got {nt} x {nc}"),
        });
    }
    check_full_pairs(nt, nc)?;
    if dist.iter().any(|d| !d.is_finite() || *d < 0.0) {
        return Err(EconError::InvalidSpecification {
            message: "the distance matrix must hold finite, non-negative distances".into(),
        });
    }
    check_full_bounds(min_controls, max_controls)?;
    let (min_c, max_c) = (min_controls, max_controls);
    let mut omit_fraction = None;
    let mut solved = None;
    let mut tried = false;
    if max_c * nt as f64 >= nc as f64 {
        tried = true;
        solved = solve_full(dist, min_c, max_c, None)?;
    }
    if solved.is_none() {
        if max_c.is_finite() && max_c >= 1.0 {
            if let Some((p0, _, _)) = solve_full(dist, min_c, f64::INFINITY, None)? {
                let (row_sets, col_sets, n_sets) = star_sets(&p0, nt, nc);
                let mut controls = vec![0usize; n_sets];
                let mut has_treated = vec![false; n_sets];
                for s in col_sets.iter().flatten() {
                    controls[*s] += 1;
                }
                for s in row_sets.iter().flatten() {
                    has_treated[*s] = true;
                }
                let num: f64 = (0..n_sets)
                    .filter(|&s| has_treated[s])
                    .map(|s| (controls[s] as f64).min(max_c))
                    .sum();
                if num == 0.0 {
                    // the unconstrained solve uses no control: retry the original problem
                    // unless it has already been solved and found infeasible
                    if !tried {
                        solved = solve_full(dist, min_c, max_c, None)?;
                    }
                } else {
                    let omf = 1.0 - num / nc as f64;
                    solved = solve_full(dist, min_c, max_c, Some(omf))?;
                    if solved.is_some() {
                        omit_fraction = Some(omf);
                    }
                }
            }
        } else if !tried {
            solved = solve_full(dist, min_c, max_c, None)?;
        }
    }
    let Some((pairs, flipped, pivots)) = solved else {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "full matching is infeasible with min_ratio = {min_c}, max_ratio = {max_c} \
                 ({nt} treated, {nc} control units)"
            ),
        });
    };
    let (row_sets, col_sets, n_sets) = star_sets(&pairs, nt, nc);
    let total_distance = pairs.iter().map(|&(r, c)| dist[[r, c]]).sum();
    Ok(FullMatchSolution {
        pairs,
        row_sets,
        col_sets,
        n_sets,
        total_distance,
        omit_fraction,
        flipped,
        pivots,
    })
}

/// Connected components of the pair graph, numbered by first appearance scanning rows, then
/// columns.
fn star_sets(
    pairs: &[(usize, usize)],
    nt: usize,
    nc: usize,
) -> (Vec<Option<usize>>, Vec<Option<usize>>, usize) {
    let mut par: Vec<usize> = (0..nt + nc).collect();
    fn find(par: &mut [usize], mut x: usize) -> usize {
        while par[x] != x {
            par[x] = par[par[x]];
            x = par[x];
        }
        x
    }
    let mut paired = vec![false; nt + nc];
    for &(r, c) in pairs {
        let (a, b) = (find(&mut par, r), find(&mut par, nt + c));
        if a != b {
            par[a] = b;
        }
        paired[r] = true;
        paired[nt + c] = true;
    }
    let mut label = vec![usize::MAX; nt + nc];
    let mut n_sets = 0;
    let mut sets = vec![None; nt + nc];
    for v in 0..nt + nc {
        if !paired[v] {
            continue;
        }
        let root = find(&mut par, v);
        if label[root] == usize::MAX {
            label[root] = n_sets;
            n_sets += 1;
        }
        sets[v] = Some(label[root]);
    }
    let col_sets = sets.split_off(nt);
    (sets, col_sets, n_sets)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Balance (summary.matchit, standardize = TRUE, ATT)
// ═══════════════════════════════════════════════════════════════════════════════

/// MatchIt's weighted variance (`col_w_var`): weights normalised to sum 1, `m (1 - m)` for a
/// binary variable, else `sum w (x - m)^2 / (1 - sum w^2)`.
fn wvar(x: &[f64], w: &[f64], bin: bool) -> f64 {
    let sw: f64 = w.iter().sum();
    let mut m = 0.0;
    let mut sw2 = 0.0;
    for (&xi, &wi) in x.iter().zip(w) {
        let v = wi / sw;
        m += v * xi;
        sw2 += v * v;
    }
    if bin {
        return m * (1.0 - m);
    }
    let ss: f64 = x
        .iter()
        .zip(w)
        .map(|(&xi, &wi)| (wi / sw) * (xi - m) * (xi - m))
        .sum();
    ss / (1.0 - sw2)
}

/// MatchIt's `pooled_sd` (standardisation fallback): binary `sqrt(sum_g s_g (1 - s_g / n_g) / n)`
/// with `s_g` the group sum, else `sqrt(sum_g sum_{i in g} (x_i - mean_g)^2 / (n - 2))`.
fn pooled_sd(x: &[f64], is_treated: &[bool], bin: bool) -> f64 {
    let n = x.len() as f64;
    let mut sum = [0.0; 2];
    let mut cnt = [0.0; 2];
    for (&xi, &t) in x.iter().zip(is_treated) {
        sum[usize::from(t)] += xi;
        cnt[usize::from(t)] += 1.0;
    }
    if bin {
        let v: f64 = (0..2)
            .filter(|&g| cnt[g] > 0.0)
            .map(|g| sum[g] * (1.0 - sum[g] / cnt[g]))
            .sum();
        return (v / n).sqrt();
    }
    let mean = [sum[0] / cnt[0], sum[1] / cnt[1]];
    let ss: f64 = x
        .iter()
        .zip(is_treated)
        .map(|(&xi, &t)| (xi - mean[usize::from(t)]).powi(2))
        .sum();
    (ss / (n - 2.0)).sqrt()
}

/// Weighted eCDF mean and max differences between the groups (summary.matchit's "eCDF Mean"
/// and "eCDF Max"): weights normalised to sum 1 per group, units ordered by `x` (stably),
/// the running difference taken at the last unit of each run of equal values. With unit
/// weights the max is the two-sample Kolmogorov-Smirnov statistic.
fn ecdf_stats(x: &[f64], is_treated: &[bool], w: &[f64]) -> (f64, f64) {
    let (mut st, mut sc) = (0.0, 0.0);
    for (&wi, &t) in w.iter().zip(is_treated) {
        if t {
            st += wi;
        } else {
            sc += wi;
        }
    }
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let (mut run, mut total, mut max, mut points) = (0.0_f64, 0.0_f64, 0.0_f64, 0usize);
    for (k, &i) in idx.iter().enumerate() {
        run += if is_treated[i] { w[i] / st } else { -w[i] / sc };
        let last = k + 1 == idx.len() || x[idx[k + 1]] != x[i];
        if last {
            total += run.abs();
            max = max.max(run.abs());
            points += 1;
        }
    }
    if points == 0 {
        return (f64::NAN, f64::NAN);
    }
    (total / points as f64, max)
}

/// summary.matchit's balance table for the ATT (S12): weighted means, the standardised mean
/// difference over the treated SD of the full sample, the weighted variance ratio (NaN for
/// binary covariates) and the weighted eCDF mean and max.
fn balance_table(
    x: &Array2<f64>,
    names: &[String],
    is_treated: &[bool],
    w: &[f64],
) -> MatchBalanceTable {
    let tol = f64::EPSILON.sqrt();
    let covariates: Vec<MatchCovariateBalance> = (0..x.ncols())
        .map(|j| {
            let col = x.column(j).to_vec();
            let bin = col.iter().all(|&v| v == 0.0 || v == 1.0);
            let (mut xt, mut wt, mut xc, mut wc) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for i in 0..col.len() {
                if is_treated[i] {
                    xt.push(col[i]);
                    wt.push(w[i]);
                } else {
                    xc.push(col[i]);
                    wc.push(w[i]);
                }
            }
            let too_small = wt.iter().filter(|&&v| v != 0.0).count() < 2
                && wc.iter().filter(|&&v| v != 0.0).count() < 2;
            let wmean = |x: &[f64], w: &[f64]| {
                x.iter().zip(w).map(|(a, b)| a * b).sum::<f64>() / w.iter().sum::<f64>()
            };
            let (mean_t, mean_c) = (wmean(&xt, &wt), wmean(&xc, &wc));
            let md = mean_t - mean_c;
            let std_diff = if md.abs() > tol {
                if too_small {
                    f64::NAN
                } else {
                    let mut sd = wvar(&xt, &vec![1.0; xt.len()], bin).sqrt();
                    if !sd.is_finite() || sd < tol {
                        sd = pooled_sd(&col, is_treated, bin);
                    }
                    md / sd
                }
            } else {
                md
            };
            let var_ratio = if bin || too_small {
                f64::NAN
            } else {
                wvar(&xt, &wt, false) / wvar(&xc, &wc, false)
            };
            let (ecdf_mean, ecdf_max) = if bin {
                (md.abs(), md.abs())
            } else if too_small {
                (f64::NAN, f64::NAN)
            } else {
                ecdf_stats(&col, is_treated, w)
            };
            MatchCovariateBalance {
                name: names[j].clone(),
                mean_treated: mean_t,
                mean_control: mean_c,
                std_diff,
                var_ratio,
                ks_statistic: ecdf_max,
                ecdf_mean,
                ecdf_max,
            }
        })
        .collect();
    let finite: Vec<f64> = covariates
        .iter()
        .map(|c| c.std_diff.abs())
        .filter(|v| v.is_finite())
        .collect();
    let mean_abs_std_diff = if finite.is_empty() {
        0.0
    } else {
        finite.iter().sum::<f64>() / finite.len() as f64
    };
    MatchBalanceTable {
        mean_abs_std_diff,
        max_abs_std_diff: finite.iter().copied().fold(0.0, f64::max),
        n_imbalanced: finite.iter().filter(|&&v| v > 0.1).count(),
        covariates,
    }
}

/// Kish's effective sample size `(sum w)^2 / sum w^2` over the positive weights; 0 for none.
fn kish_ess(w: &[f64]) -> f64 {
    let (s, s2) = w
        .iter()
        .filter(|&&v| v > 0.0)
        .fold((0.0, 0.0), |(s, s2), &v| (s + v, s2 + v * v));
    if s2 > 0.0 { s * s / s2 } else { 0.0 }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use p2a_validation::{Check, Tol};
    use polars::prelude::*;

    const COVS: [&str; 3] = ["x1", "x2", "x3"];

    fn nn(ratio: usize, caliper: Option<f64>, replace: bool) -> MatchMethod {
        MatchMethod::NearestNeighbor {
            ratio,
            caliper,
            replace,
        }
    }

    fn full(min_ratio: f64, max_ratio: f64) -> MatchMethod {
        MatchMethod::Full {
            min_ratio,
            max_ratio,
        }
    }

    const BAL: [&str; 6] = [
        "mean_t",
        "mean_c",
        "smd",
        "var_ratio",
        "ecdf_mean",
        "ecdf_max",
    ];

    fn opts(o: MOrder) -> MatchOptions {
        let mut m = MatchOptions::default();
        m.m_order = Some(o);
        m
    }

    /// R's .mt_pairs(): treated row ids (1-based, data order), then each one's first matched
    /// control (NaN if none).
    fn pairs_vec(r: &MatchResult) -> Vec<f64> {
        let t: Vec<f64> = r
            .matches
            .iter()
            .map(|m| (m.treated_idx + 1) as f64)
            .collect();
        let c = r.matches.iter().map(|m| {
            m.control_indices
                .first()
                .map_or(f64::NAN, |&c| (c + 1) as f64)
        });
        t.into_iter().chain(c).collect()
    }

    /// R's cbind(treated ids, match.matrix columns) for ratio k.
    fn match_matrix_vec(r: &MatchResult, k: usize) -> Vec<f64> {
        let mut v: Vec<f64> = r
            .matches
            .iter()
            .map(|m| (m.treated_idx + 1) as f64)
            .collect();
        for col in 0..k {
            v.extend(r.matches.iter().map(|m| {
                m.control_indices
                    .get(col)
                    .map_or(f64::NAN, |&c| (c + 1) as f64)
            }));
        }
        v
    }

    /// R's .mt_canon(): each row's label is 1 + the smallest row index of its set; NaN if
    /// unmatched.
    fn canonical_sets(sub: &[Option<usize>]) -> Vec<f64> {
        let mut first = std::collections::HashMap::new();
        for (i, s) in sub.iter().enumerate() {
            if let Some(s) = s {
                first.entry(*s).or_insert(i);
            }
        }
        sub.iter()
            .map(|s| s.map_or(f64::NAN, |s| (first[&s] + 1) as f64))
            .collect()
    }

    fn mask(ds: &Dataset) -> Vec<bool> {
        DesignMatrix::extract_column(ds.df(), "treat")
            .unwrap()
            .iter()
            .map(|&v| v == 1.0)
            .collect()
    }

    fn balance_rows(t: &MatchBalanceTable) -> [Vec<f64>; 6] {
        let f =
            |g: fn(&MatchCovariateBalance) -> f64| t.covariates.iter().map(g).collect::<Vec<f64>>();
        [
            f(|c| c.mean_treated),
            f(|c| c.mean_control),
            f(|c| c.std_diff),
            f(|c| c.var_ratio),
            f(|c| c.ecdf_mean),
            f(|c| c.ecdf_max),
        ]
    }

    /// matchit(.., method = "nearest", distance = "glm"): MatchIt's default m.order for a PS is "largest".
    #[test]
    fn test_validate_matchit_nearest_logit_largest() {
        let (case, ds) = crate::test_support::load_case("matchit_nearest");
        let r = match_it(
            &ds,
            "treat",
            &COVS,
            nn(1, None, false),
            Some(DistanceMethod::Logit),
        )
        .unwrap();
        let ps = r.propensity_scores.clone().unwrap();
        Check::new(&case)
            .vector("logit_largest_pairs", &pairs_vec(&r), Tol::exact())
            .vector("logit_ps", &ps[..12], Tol::iterative(1e-8))
            .scalar("n_treated", r.n_treated as f64, Tol::exact())
            .scalar("n_control", r.n_control as f64, Tol::exact())
            .finish();
        assert_eq!(
            (r.m_order, r.n_matched_treated, r.n_matched_control),
            (Some(MOrder::Largest), 89, 89)
        );
        assert!(r.weights.iter().all(|&w| w == 0.0 || w == 1.0));
        assert_eq!(r.subclass.iter().flatten().count(), 178);
    }

    /// m.order = "data" and "smallest" reproduce MatchIt; R distinguishes the orders and so must p2a.
    #[test]
    fn test_validate_matchit_nearest_logit_data() {
        let (case, ds) = crate::test_support::load_case("matchit_nearest");
        let run = |o| {
            match_it_with(
                &ds,
                "treat",
                &COVS,
                nn(1, None, false),
                Some(DistanceMethod::Logit),
                &opts(o),
            )
            .unwrap()
        };
        let (data, small, large) = (
            run(MOrder::Data),
            run(MOrder::Smallest),
            run(MOrder::Largest),
        );
        Check::new(&case)
            .vector("logit_data_pairs", &pairs_vec(&data), Tol::exact())
            .vector("logit_smallest_pairs", &pairs_vec(&small), Tol::exact())
            .finish();
        assert_ne!(pairs_vec(&data), pairs_vec(&large));
        assert_ne!(pairs_vec(&small), pairs_vec(&large));
    }

    #[test]
    fn test_validate_matchit_nearest_probit() {
        let (case, ds) = crate::test_support::load_case("matchit_nearest");
        let r = match_it(
            &ds,
            "treat",
            &COVS,
            nn(1, None, false),
            Some(DistanceMethod::Probit),
        )
        .unwrap();
        let ps = r.propensity_scores.clone().unwrap();
        Check::new(&case)
            .vector("probit_largest_pairs", &pairs_vec(&r), Tol::exact())
            .vector("probit_ps", &ps[..12], Tol::iterative(1e-5)) // MatchIt's glm (epsilon 1e-8): up to 4.6e-7 apart
            .finish();
    }

    /// 1:2 without replacement: every treated unit gets its first match before any gets a second.
    #[test]
    fn test_validate_matchit_nearest_ratio2_rounds() {
        let (case, ds) = crate::test_support::load_case("matchit_nearest");
        let r = match_it(
            &ds,
            "treat",
            &COVS,
            nn(2, None, false),
            Some(DistanceMethod::Logit),
        )
        .unwrap();
        Check::new(&case)
            .vector("ratio2_mm", &match_matrix_vec(&r, 2), Tol::exact())
            .vector("ratio2_weights", &r.weights, Tol::coef(r.n_obs))
            .finish();
        assert_eq!(
            r.matches
                .iter()
                .filter(|m| m.control_indices.len() == 2)
                .count(),
            62
        );
        assert!(
            r.warnings
                .iter()
                .any(|w| w == "Not all treated units will get 2 matches."),
            "{:?}",
            r.warnings
        );
    }

    #[test]
    fn test_validate_matchit_nearest_caliper() {
        let (case, ds) = crate::test_support::load_case("matchit_nearest");
        let r = match_it(
            &ds,
            "treat",
            &COVS,
            nn(1, Some(0.1), false),
            Some(DistanceMethod::Logit),
        )
        .unwrap();
        Check::new(&case)
            .scalar("caliper_abs", r.caliper_abs.unwrap(), Tol::iterative(1e-8))
            .vector("caliper_pairs", &pairs_vec(&r), Tol::exact())
            .vector("caliper_weights", &r.weights, Tol::coef(r.n_obs))
            .finish();
        assert_eq!((r.n_matched_treated, r.caliper_used), (74, Some(0.1)));
    }

    #[test]
    fn test_validate_matchit_nearest_replace() {
        let (case, ds) = crate::test_support::load_case("matchit_nearest");
        let r = match_it_with(
            &ds,
            "treat",
            &COVS,
            nn(1, None, true),
            Some(DistanceMethod::Logit),
            &opts(MOrder::Largest),
        )
        .unwrap();
        Check::new(&case)
            .vector("replace_pairs", &pairs_vec(&r), Tol::exact())
            .vector("replace_weights", &r.weights, Tol::coef(r.n_obs))
            .finish();
        assert_eq!(r.m_order, Some(MOrder::Data)); // with replacement MatchIt matches in data order
        assert!(r.subclass.iter().all(Option::is_none)); // MatchIt: subclass NULL
    }

    #[test]
    fn test_validate_matchit_nearest_mahalanobis() {
        let (case, ds) = crate::test_support::load_case("matchit_nearest");
        let r = match_it(
            &ds,
            "treat",
            &COVS,
            nn(1, None, false),
            Some(DistanceMethod::Mahalanobis),
        )
        .unwrap();
        let x = DesignMatrix::from_dataframe(ds.df(), &COVS, false)
            .unwrap()
            .data;
        let s = pooled_within_covariance(&x, &mask(&ds)).unwrap();
        Check::new(&case)
            .vector("maha_pairs", &pairs_vec(&r), Tol::exact())
            .vector(
                "maha_pooled_cov",
                &s.iter().copied().collect::<Vec<f64>>(),
                Tol::coef(r.n_obs),
            )
            .finish();
        assert_eq!(
            (r.m_order, r.propensity_scores.is_none()),
            (Some(MOrder::Data), true)
        );
    }

    /// Regression guard: Euclidean nearest-neighbour matching already matched MatchIt (data order, raw covariates).
    #[test]
    fn test_validate_matchit_nearest_euclidean() {
        let (case, ds) = crate::test_support::load_case("matchit_nearest");
        let r = match_it(
            &ds,
            "treat",
            &COVS,
            nn(1, None, false),
            Some(DistanceMethod::Euclidean),
        )
        .unwrap();
        Check::new(&case)
            .vector("euclid_pairs", &pairs_vec(&r), Tol::exact())
            .finish();
    }

    /// MatchIt 4.8.0's tie rules, probed with matchit(t ~ x, distance = <score vector>) and distance =
    /// "mahalanobis" / "euclidean" (Appendix B, ties.R, ties2.R, ties3.R):
    /// - an exact tie between the nearest score below and the nearest at or above goes below;
    /// - among equal scores below, the later row (the binary-search neighbour) wins, at or above the earlier row;
    /// - with one covariate, Mahalanobis and Euclidean switch to these scalar rules (x = -1 wins over x = 1
    ///   whatever its row).
    /// With two or more covariates p2a takes the lowest row; that is p2a's rule, not MatchIt's (deviation 14).
    #[test]
    fn test_nearest_neighbour_ties_as_matchit() {
        let one = |s: &[f64], ratio: usize| {
            let t: Vec<bool> = (0..s.len()).map(|i| i == 0).collect();
            nn_on_scores(s, &t, &[0], ratio, None, false).unwrap()[0].clone()
        };
        assert_eq!(one(&[0.5, 0.25, 0.75, 0.125], 1), vec![1]);
        assert_eq!(one(&[0.5, 0.75, 0.25, 0.125], 1), vec![2]);
        assert_eq!(one(&[0.5, 0.4, 0.4, 0.9], 1), vec![2]);
        assert_eq!(one(&[0.5, 0.6, 0.6, 0.1], 1), vec![1]);
        assert_eq!(one(&[0.5, 0.5, 0.5, 0.125], 1), vec![1]);
        assert_eq!(one(&[0.5, 0.25, 0.75, 0.75, 0.125], 2), vec![1, 2]);
        for dist in [DistanceMethod::Mahalanobis, DistanceMethod::Euclidean] {
            for (xs, want) in [([0.0, -1.0, 1.0], 1usize), ([0.0, 1.0, -1.0], 2)] {
                let df = df! { "d" => [1.0, 0.0, 0.0], "x" => xs.to_vec() }.unwrap();
                let r = match_it(
                    &Dataset::new(df),
                    "d",
                    &["x"],
                    nn(1, None, false),
                    Some(dist),
                )
                .unwrap();
                assert_eq!(r.matches[0].control_indices, vec![want], "{dist:?} {xs:?}");
            }
        }
        let x = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, -1.0, -0.5, 1.0, 0.5]).unwrap();
        let m = CovariateMetric { z: x };
        assert_eq!(
            nn_on_covariates(&m, &[true, false, false], &[0], 1, None, false).unwrap()[0],
            vec![1]
        );
    }

    #[test]
    fn test_validate_matchit_m_order_random_is_seeded() {
        let (_, ds) = crate::test_support::load_case("matchit_nearest");
        let run = |seed| {
            let mut o = opts(MOrder::Random);
            o.seed = seed;
            // bit patterns, so the unmatched units' NaN compare equal
            pairs_vec(
                &match_it_with(&ds, "treat", &COVS, nn(1, Some(0.1), false), None, &o).unwrap(),
            )
            .into_iter()
            .map(f64::to_bits)
            .collect::<Vec<u64>>()
        };
        assert_eq!(run(Some(7)), run(Some(7)));
        assert_eq!(run(None), run(Some(DEFAULT_MATCH_SEED)));
        assert_ne!(run(Some(7)), run(Some(8)));
        // with replacement the order does not matter: an explicit non-data order is reported
        let r = match_it_with(
            &ds,
            "treat",
            &COVS,
            nn(1, None, true),
            None,
            &opts(MOrder::Random),
        )
        .unwrap();
        assert_eq!(r.m_order, Some(MOrder::Data));
        assert!(
            r.warnings.iter().any(|w| w
                == "m_order = 'random' has no effect with replacement; treated units are matched \
                    in data order (as MatchIt)"),
            "{:?}",
            r.warnings
        );
        let r = match_it_with(
            &ds,
            "treat",
            &COVS,
            nn(1, None, true),
            None,
            &opts(MOrder::Data),
        )
        .unwrap();
        assert!(!r.warnings.iter().any(|w| w.contains("m_order")));
    }

    #[test]
    fn test_m_order_parses_aliases_and_rejects_unknown() {
        MOrder::choices().validate().unwrap();
        for (s, v) in [
            ("largest", MOrder::Largest),
            ("Descending", MOrder::Largest),
            ("smallest", MOrder::Smallest),
            ("ASCENDING", MOrder::Smallest),
            ("data", MOrder::Data),
            ("data-order", MOrder::Data),
            ("random", MOrder::Random),
        ] {
            assert_eq!(s.parse::<MOrder>().unwrap(), v, "{s}");
        }
        let e = "closest".parse::<MOrder>().unwrap_err().to_string();
        assert!(
            e.contains("m_order") && e.contains("largest, smallest, data, random"),
            "{e}"
        );
        assert_eq!(MOrder::Largest.to_string(), "largest");
    }

    #[test]
    fn test_distance_method_parses_aliases_and_rejects_unknown() {
        DistanceMethod::choices().validate().unwrap();
        for (s, v) in [
            ("logit", DistanceMethod::Logit),
            ("GLM", DistanceMethod::Logit),
            ("logistic", DistanceMethod::Logit),
            ("probit", DistanceMethod::Probit),
            ("Mahalanobis", DistanceMethod::Mahalanobis),
            ("euclidean", DistanceMethod::Euclidean),
        ] {
            assert_eq!(s.parse::<DistanceMethod>().unwrap(), v, "{s}");
        }
        let e = "manhattan"
            .parse::<DistanceMethod>()
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("distance") && e.contains("logit, probit, mahalanobis, euclidean"),
            "{e}"
        );
        assert_eq!(
            DistanceMethod::Logit.to_string(),
            "Logit (Propensity Score)"
        ); // Display unchanged
    }

    #[test]
    fn test_match_method_kind_parses_aliases_and_rejects_unknown() {
        MatchMethodKind::choices().validate().unwrap();
        for (s, v) in [
            ("nearest", MatchMethodKind::Nearest),
            ("nn", MatchMethodKind::Nearest),
            ("Nearest Neighbor", MatchMethodKind::Nearest),
            ("cem", MatchMethodKind::Cem),
            ("coarsened", MatchMethodKind::Cem),
            ("coarsenedexact", MatchMethodKind::Cem),
            ("full", MatchMethodKind::Full),
            ("optimal", MatchMethodKind::Full),
            ("subclass", MatchMethodKind::Subclass),
            ("stratify", MatchMethodKind::Subclass),
            ("stratification", MatchMethodKind::Subclass),
        ] {
            assert_eq!(s.parse::<MatchMethodKind>().unwrap(), v, "{s}");
        }
        let e = "genetic"
            .parse::<MatchMethodKind>()
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("method") && e.contains("nearest, cem, full, subclass"),
            "{e}"
        );
    }

    #[test]
    fn test_validate_matching_rejects_invalid_options() {
        let (_, ds) = crate::test_support::load_case("matchit_nearest");
        let msg = |r: EconResult<MatchResult>| match r {
            Err(EconError::InvalidSpecification { message }) => message,
            other => panic!("expected InvalidSpecification, got ok = {}", other.is_ok()),
        };
        for c in [0.0, -0.1, f64::NAN, f64::INFINITY] {
            assert!(
                msg(match_it(&ds, "treat", &COVS, nn(1, Some(c), false), None)).contains("caliper")
            );
        }
        let m = msg(match_it_with(
            &ds,
            "treat",
            &COVS,
            nn(1, None, false),
            Some(DistanceMethod::Mahalanobis),
            &opts(MOrder::Largest),
        ));
        assert!(m.contains("m_order") && m.contains("largest"), "{m}");
        assert!(
            msg(match_it(
                &ds,
                "treat",
                &COVS,
                MatchMethod::Subclass { n_subclasses: 5 },
                Some(DistanceMethod::Euclidean)
            ))
            .contains("propensity")
        );
        assert!(
            msg(match_it(
                &ds,
                "treat",
                &COVS,
                nn(1, Some(1e-9), false),
                None
            ))
            .contains("no treated unit")
        ); // MatchIt: "No matches were found"
        let df =
            df! { "d" => [1.0, 0.0, 2.0, 0.0, 1.0, 0.0], "x" => [0.1, 0.4, 0.2, 0.9, 0.5, 0.7] }
                .unwrap();
        let m = msg(match_it(
            &Dataset::new(df),
            "d",
            &["x"],
            nn(1, None, false),
            None,
        ));
        assert!(m.contains("0/1") && m.contains("row 3"), "{m}");
    }

    /// summary(matchit(..), standardize = TRUE)$sum.all / $sum.matched for x1, x2, x3 (ATT: SMD standardised
    /// by the treated SD of the full sample; Var. Ratio NA for the binary x3).
    /// matchit(method = "cem"): numeric covariates are cut into nclass.Sturges(x) = 9 equal-width bins.
    #[test]
    fn test_validate_cem_sturges() {
        let (case, ds) = crate::test_support::load_case("matchit_full_cem");
        let r = cem_match(&ds, "treat", &COVS).unwrap();
        Check::new(&case)
            .vector("cem_weights", &r.weights, Tol::coef(r.n_obs))
            .scalar("cem_n_matched_t", r.n_matched_treated as f64, Tol::exact())
            .scalar("cem_n_matched_c", r.n_matched_control as f64, Tol::exact())
            .scalar(
                "cem_n_strata",
                r.subclasses.as_ref().unwrap().len() as f64,
                Tol::exact(),
            )
            .finish();
        assert!(r.propensity_scores.is_none());
        assert!(
            r.to_string()
                .contains("Coarsened Exact Matching (Sturges bins)")
        );
    }

    /// n_bins: Some(4) is MatchIt's cutpoints = "q4"; explicit cut points are left-closed (findInterval).
    /// The two binnings R distinguishes give different strata here too.
    #[test]
    fn test_validate_cem_quantile_and_explicit_cut_points() {
        let (case, ds) = crate::test_support::load_case("matchit_full_cem");
        let q4 = match_it(
            &ds,
            "treat",
            &COVS,
            MatchMethod::CoarsenedExact {
                cutpoints: None,
                n_bins: Some(4),
            },
            None,
        )
        .unwrap();
        let cuts = vec![vec![-1.0, 0.0, 1.0], vec![-0.5, 0.5], vec![0.2, 0.8]];
        let cx = match_it(
            &ds,
            "treat",
            &COVS,
            MatchMethod::CoarsenedExact {
                cutpoints: Some(cuts),
                n_bins: None,
            },
            None,
        )
        .unwrap();
        Check::new(&case)
            .vector("cem_q4_weights", &q4.weights, Tol::coef(q4.n_obs))
            .scalar(
                "cem_q4_n_strata",
                q4.subclasses.as_ref().unwrap().len() as f64,
                Tol::exact(),
            )
            .vector("cem_cut_weights", &cx.weights, Tol::coef(cx.n_obs))
            .scalar(
                "cem_cut_n_strata",
                cx.subclasses.as_ref().unwrap().len() as f64,
                Tol::exact(),
            )
            .finish();
        assert_ne!(q4.weights, cem_match(&ds, "treat", &COVS).unwrap().weights);
    }

    #[test]
    fn test_find_interval_is_left_closed() {
        let b = [f64::NEG_INFINITY, 0.0, 1.0, f64::INFINITY];
        assert_eq!(
            [-0.5, 0.0, 0.5, 1.0, 2.0].map(|x| find_interval(&b, x)),
            [1, 2, 2, 3, 3]
        );
    }

    #[test]
    fn test_cem_without_common_strata_is_an_error() {
        let df = df! { "d" => [1.0, 1.0, 0.0, 0.0], "x" => [0.0, 0.1, 10.0, 10.1] }.unwrap();
        let r = match_it(
            &Dataset::new(df),
            "d",
            &["x"],
            MatchMethod::CoarsenedExact {
                cutpoints: Some(vec![vec![5.0]]),
                n_bins: None,
            },
            None,
        );
        assert!(
            matches!(r, Err(EconError::InvalidSpecification { ref message }) if message.contains("no units were matched"))
        );
    }

    #[test]
    fn test_cem_non_finite_cutpoints_name_the_covariate() {
        let df = df! {
            "d" => [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            "x" => [0.0, 0.1, 10.0, 10.1, 3.2, 4.7],
            "age" => [21.0, 34.0, 45.0, 29.0, 52.0, 38.0]
        }
        .unwrap();
        let r = match_it(
            &Dataset::new(df),
            "d",
            &["x", "age"],
            MatchMethod::CoarsenedExact {
                cutpoints: Some(vec![vec![5.0], vec![30.0, f64::NAN]]),
                n_bins: None,
            },
            None,
        );
        assert!(
            matches!(r, Err(EconError::InvalidSpecification { ref message })
                if message == "cutpoints for covariate 'age' must be finite numbers"),
            "{r:?}"
        );
    }

    #[test]
    fn test_validate_matchit_balance_table() {
        let (case, ds) = crate::test_support::load_case("matchit_nearest");
        let r1 = match_it(
            &ds,
            "treat",
            &COVS,
            nn(1, None, false),
            Some(DistanceMethod::Logit),
        )
        .unwrap();
        let r2 = match_it(
            &ds,
            "treat",
            &COVS,
            nn(2, None, false),
            Some(DistanceMethod::Logit),
        )
        .unwrap();
        let n = r1.n_obs;
        let mut c = Check::new(&case);
        for (prefix, r) in [("logit_largest", &r1), ("ratio2", &r2)] {
            let (b, a) = (
                balance_rows(&r.balance_before),
                balance_rows(&r.balance_after),
            );
            for (i, stat) in BAL.iter().enumerate() {
                c.vector(&format!("{prefix}_before_{stat}"), &b[i], Tol::stat(n));
                c.vector(&format!("{prefix}_after_{stat}"), &a[i], Tol::stat(n));
            }
        }
        let (b1, a1) = (
            balance_rows(&r1.balance_before),
            balance_rows(&r1.balance_after),
        );
        c.vector("logit_largest_smd_before", &b1[2], Tol::stat(n))
            .vector("logit_largest_smd_after", &a1[2], Tol::stat(n))
            .vector(
                "ratio2_ess",
                &[r2.ess_control, r2.ess_treated],
                Tol::stat(n),
            )
            .finish();
        assert!(
            r1.balance_after
                .covariates
                .iter()
                .all(|cv| cv.ks_statistic == cv.ecdf_max || cv.ks_statistic.is_nan())
        );
        assert_eq!(r1.effective_sample_size, r1.ess_treated + r1.ess_control);
    }

    /// T0-WP11's R pins of the two-sample KS statistic, through the weighted eCDF with unit weights:
    /// ks.test(c(1.2,3.4,3.4,0.5,2.2,5.0,2.2,4.1), c(0.7,3.4,2.9,2.2,6.3,1.1,4.4))$statistic and
    /// ks.test(((0:59 * 37) %% 23) * 0.25, ((0:44 * 11) %% 19) * 0.3 + 0.1)$statistic.
    #[test]
    fn test_ecdf_max_is_the_ks_statistic_with_unit_weights() {
        let ks = |a: &[f64], b: &[f64]| {
            let x: Vec<f64> = a.iter().chain(b).copied().collect();
            let t: Vec<bool> = (0..x.len()).map(|i| i < a.len()).collect();
            ecdf_stats(&x, &t, &vec![1.0; x.len()]).1
        };
        assert!(
            (ks(
                &[1.2, 3.4, 3.4, 0.5, 2.2, 5.0, 2.2, 4.1],
                &[0.7, 3.4, 2.9, 2.2, 6.3, 1.1, 4.4]
            ) - 0.16071428571428575)
                .abs()
                < 1e-12
        );
        let s1: Vec<f64> = (0..60u64).map(|i| ((i * 37) % 23) as f64 * 0.25).collect();
        let s2: Vec<f64> = (0..45u64)
            .map(|i| ((i * 11) % 19) as f64 * 0.3 + 0.1)
            .collect();
        assert!((ks(&s1, &s2) - 0.06666666666666664).abs() < 1e-12);
        assert!(
            ks(&[1.0, 2.0, 3.0], &[10.0, 11.0, 12.0]) > 0.99 && ks(&[1.0, 2.0], &[1.0, 2.0]) == 0.0
        );
    }

    #[test]
    fn test_validate_matchit_display_shows_the_balance_tables_and_notes() {
        let (_, ds) = crate::test_support::load_case("matchit_nearest");
        let r = match_it(&ds, "treat", &COVS, nn(1, Some(0.1), false), None).unwrap();
        let s = r.to_string();
        for want in [
            "Propensity Score Matching",
            "Matching order: largest",
            "Matched ESS:",
            "on the score scale",
            "Balance Before Matching",
            "Balance After Matching",
            "eCDF Max",
            "Converged: yes",
        ] {
            assert!(s.contains(want), "missing {want:?} in\n{s}");
        }
        let cem = cem_match(&ds, "treat", &COVS).unwrap().to_string();
        assert!(cem.contains("Distance: not used"), "{cem}"); // the binning label is Task 6's
    }

    /// glm(treat ~ x1 + x2 + x3, binomial(logit | probit)) at epsilon 1e-14 (the *_tight keys)
    /// and MatchIt's own glm (epsilon 1e-8, the package values);
    /// validation/r/cases/causal_matching.R, case matchit_nearest.
    #[test]
    fn test_validate_matchit_propensity_models() {
        let (case, ds) = crate::test_support::load_case("matchit_nearest");
        let x = DesignMatrix::from_dataframe(ds.df(), &COVS, false)
            .unwrap()
            .data;
        let d = DesignMatrix::extract_column(ds.df(), "treat").unwrap();
        let names = ["(Intercept)", "x1", "x2", "x3"];
        let lg = estimate_propensity(&x, &d, GlmLink::Logit).unwrap();
        let pr = estimate_propensity(&x, &d, GlmLink::Probit).unwrap();
        Check::new(&case)
            .vector("logit_ps_tight", &lg.scores.to_vec(), Tol::iterative(1e-8))
            .named(
                "logit_coef",
                &names,
                &lg.coefficients.to_vec(),
                Tol::iterative(1e-8),
            )
            .vector("probit_ps_tight", &pr.scores.to_vec(), Tol::iterative(1e-8))
            .named(
                "probit_coef_tight",
                &names,
                &pr.coefficients.to_vec(),
                Tol::iterative(1e-8),
            )
            // MatchIt's glm (epsilon 1e-8) stops about 1e-6 away from the MLE
            .named(
                "probit_coef",
                &names,
                &pr.coefficients.to_vec(),
                Tol::iterative(1e-5),
            )
            .finish();
        assert!(lg.convergence.converged && pr.convergence.converged && lg.warnings.is_empty());
    }

    /// CAUS-20: nearest-neighbour, full and subclass matching all use the probit MLE when
    /// asked for probit (Full and Subclass used to fit a logit; nearest-neighbour solved the
    /// logit score equation).
    #[test]
    fn test_validate_matchit_probit_in_every_ps_method() {
        let (case, ds) = crate::test_support::load_case("matchit_nearest");
        for m in [
            nn(1, None, false),
            full(0.0, f64::INFINITY),
            MatchMethod::Subclass { n_subclasses: 5 },
        ] {
            let r = match_it(&ds, "treat", &COVS, m, Some(DistanceMethod::Probit)).unwrap();
            Check::new(&case)
                .vector(
                    "probit_ps_tight",
                    &r.propensity_scores.clone().unwrap(),
                    Tol::iterative(1e-8),
                )
                .finish();
            assert!(r.propensity_convergence.as_ref().unwrap().converged);
        }
    }

    /// Perfect separation (R glm: "glm.fit: fitted probabilities numerically 0 or 1
    /// occurred", converged in 28 iterations, x2 aliased; MatchIt matches every treated
    /// unit). Replaces the old test that accepted any outcome (CAUS-92).
    #[test]
    fn test_matching_reports_propensity_glm_warnings() {
        let df = df! { "treatment" => [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        "x1" => [10.0, 11.0, 12.0, 1.0, 2.0, 3.0], "x2" => [10.0, 11.0, 12.0, 1.0, 2.0, 3.0] }
        .unwrap();
        let r = nearest_neighbor_match(&Dataset::new(df), "treatment", &["x1", "x2"], None, false)
            .unwrap();
        assert_eq!(r.n_matched_treated, 3);
        let sep = "propensity-score glm: glm.fit: fitted probabilities numerically 0 or 1 occurred";
        assert!(r.warnings.iter().any(|w| w == sep), "{:?}", r.warnings);
        assert!(r.to_string().contains(&format!("Warning: {sep}")));
    }

    /// Create test dataset with known treatment assignment mechanism.
    ///
    /// DGP: Treatment is more likely for higher x1 and x2 values.
    /// This creates imbalance in covariates before matching.
    fn create_test_dataset() -> Dataset {
        let df = df! {
            "treatment" => [
                // Treated group: higher x1, x2 on average
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0,
                // Control group: lower x1, x2 on average
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ],
            "x1" => [
                // Treated: mostly higher values
                0.7, 0.8, 0.6, 0.9, 0.75, 0.85, 0.65, 0.95, 0.72, 0.88,
                0.5, 0.6, 0.55, 0.7, 0.62,
                // Control: mostly lower values (but with overlap)
                0.2, 0.3, 0.1, 0.4, 0.25, 0.35, 0.15, 0.45, 0.22, 0.38,
                0.5, 0.55, 0.4, 0.6, 0.48, 0.3, 0.25, 0.35, 0.42, 0.28
            ],
            "x2" => [
                // Treated: mostly higher values
                0.65, 0.75, 0.55, 0.85, 0.7, 0.8, 0.6, 0.9, 0.68, 0.82,
                0.45, 0.55, 0.5, 0.65, 0.58,
                // Control: mostly lower values (but with overlap)
                0.15, 0.25, 0.05, 0.35, 0.2, 0.3, 0.1, 0.4, 0.18, 0.32,
                0.45, 0.5, 0.35, 0.55, 0.42, 0.25, 0.2, 0.3, 0.38, 0.22
            ]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_nearest_neighbor_matching_basic() {
        let dataset = create_test_dataset();
        let result =
            nearest_neighbor_match(&dataset, "treatment", &["x1", "x2"], None, false).unwrap();

        // Check basic structure
        assert_eq!(result.n_treated, 15);
        assert_eq!(result.n_control, 20);
        assert!(result.n_matched_treated > 0);
        assert!(result.n_matched_control > 0);

        // Balance should improve after matching
        assert!(
            result.balance_after.mean_abs_std_diff <= result.balance_before.mean_abs_std_diff + 0.1
        );

        // Check propensity scores were computed
        assert!(result.propensity_scores.is_some());
    }

    #[test]
    fn test_nearest_neighbor_with_replacement() {
        let dataset = create_test_dataset();
        let result =
            nearest_neighbor_match(&dataset, "treatment", &["x1", "x2"], None, true).unwrap();

        // With replacement, all treated units should be matched
        assert_eq!(result.n_matched_treated, result.n_treated);
    }

    #[test]
    fn test_full_matching_basic() {
        let dataset = create_test_dataset();
        let result = full_match(&dataset, "treatment", &["x1", "x2"]).unwrap();

        // Full matching should create subclasses
        assert!(result.subclasses.is_some());
        assert_eq!(
            result.n_matched_treated + result.n_matched_control,
            result.n_obs
        );

        // All units should be matched in full matching
        assert!(result.n_matched_treated > 0);
        assert!(result.n_matched_control > 0);
    }

    #[test]
    fn test_subclassification_basic() {
        let dataset = create_test_dataset();
        let result = subclass_match(&dataset, "treatment", &["x1", "x2"], 5).unwrap();

        // Should create specified number of subclasses
        assert!(result.subclasses.is_some());
        let subclasses = result.subclasses.as_ref().unwrap();
        assert!(subclasses.len() <= 5);
    }

    #[test]
    fn test_balance_improvement() {
        let dataset = create_test_dataset();
        let result =
            nearest_neighbor_match(&dataset, "treatment", &["x1", "x2"], None, false).unwrap();

        // Check balance statistics are computed
        assert!(!result.balance_before.covariates.is_empty());
        assert!(!result.balance_after.covariates.is_empty());

        // Before matching, there should be imbalance
        assert!(result.balance_before.mean_abs_std_diff > 0.0);
    }

    #[test]
    fn test_match_with_different_distances() {
        let dataset = create_test_dataset();

        // Test with Mahalanobis distance
        let result = match_it(
            &dataset,
            "treatment",
            &["x1", "x2"],
            MatchMethod::default(),
            Some(DistanceMethod::Mahalanobis),
        )
        .unwrap();
        assert_eq!(result.distance, DistanceMethod::Mahalanobis);

        // Test with Euclidean distance
        let result = match_it(
            &dataset,
            "treatment",
            &["x1", "x2"],
            MatchMethod::default(),
            Some(DistanceMethod::Euclidean),
        )
        .unwrap();
        assert_eq!(result.distance, DistanceMethod::Euclidean);
    }

    #[test]
    fn test_ratio_matching() {
        let dataset = create_test_dataset();
        let method = MatchMethod::NearestNeighbor {
            ratio: 2,
            caliper: None,
            replace: true,
        };
        let result = match_it(
            &dataset,
            "treatment",
            &["x1", "x2"],
            method,
            Some(DistanceMethod::Logit),
        )
        .unwrap();

        // Each treated unit should have up to 2 controls
        for m in &result.matches {
            if m.weight > 0.0 {
                assert!(m.control_indices.len() <= 2);
            }
        }
    }

    #[test]
    fn test_effective_sample_size() {
        // Kish: (sum w)^2 / sum w^2 = 4^2 / 3.5
        assert!((kish_ess(&[1.0, 1.0, 1.0, 0.5, 0.5]) - 4.571428571428571).abs() < 1e-15);
        assert_eq!(kish_ess(&[0.0, 0.0]), 0.0);
    }

    #[test]
    fn test_display_traits() {
        let dataset = create_test_dataset();
        let result =
            nearest_neighbor_match(&dataset, "treatment", &["x1", "x2"], None, false).unwrap();

        // Test Display for MatchResult
        let output = format!("{}", result);
        assert!(output.contains("Propensity Score Matching"));
        assert!(output.contains("Matching order: largest (MatchIt m.order)"));
        assert!(output.contains("Matched ESS:"));
        assert!(output.contains("Balance Before Matching (summary.matchit"));
        assert!(output.contains("Balance After Matching:"));

        // Test Display for MatchBalanceTable
        let balance_output = format!("{}", result.balance_before);
        assert!(balance_output.contains("Covariate"));
    }

    #[test]
    fn test_missing_column_error() {
        let dataset = create_test_dataset();
        let result = nearest_neighbor_match(&dataset, "nonexistent", &["x1", "x2"], None, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_optimized_matching_correctness() {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;

        // Create larger dataset with known structure
        let n = 500;
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let x1: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..1.0)).collect();
        let x2: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..1.0)).collect();

        // Treatment probability depends on x1 and x2
        let treatment: Vec<f64> = x1
            .iter()
            .zip(x2.iter())
            .map(|(a, b)| {
                if a + b > 1.0 && rng.gen_range(0.0..1.0) > 0.3 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();

        let df = df! {
            "treatment" => treatment.clone(),
            "x1" => x1,
            "x2" => x2,
        }
        .unwrap();
        let dataset = Dataset::new(df);

        // Test with replacement
        let result_with = match_it(
            &dataset,
            "treatment",
            &["x1", "x2"],
            MatchMethod::NearestNeighbor {
                ratio: 1,
                caliper: None,
                replace: true,
            },
            Some(DistanceMethod::Logit),
        )
        .unwrap();

        // All treated units should be matched with replacement
        let n_treated = treatment.iter().filter(|&&t| t > 0.5).count();
        assert_eq!(result_with.n_matched_treated, n_treated);

        // Test without replacement
        let result_without = match_it(
            &dataset,
            "treatment",
            &["x1", "x2"],
            MatchMethod::NearestNeighbor {
                ratio: 1,
                caliper: None,
                replace: false,
            },
            Some(DistanceMethod::Logit),
        )
        .unwrap();

        // Each control should be matched at most once (without replacement)
        let control_counts: std::collections::HashMap<usize, usize> = result_without
            .matches
            .iter()
            .flat_map(|m| m.control_indices.iter().copied())
            .fold(std::collections::HashMap::new(), |mut acc, ci| {
                *acc.entry(ci).or_insert(0) += 1;
                acc
            });
        for &count in control_counts.values() {
            assert_eq!(
                count, 1,
                "Control unit matched multiple times without replacement"
            );
        }

        // Test Mahalanobis distance
        let result_maha = match_it(
            &dataset,
            "treatment",
            &["x1", "x2"],
            MatchMethod::NearestNeighbor {
                ratio: 1,
                caliper: None,
                replace: true,
            },
            Some(DistanceMethod::Mahalanobis),
        )
        .unwrap();

        assert_eq!(result_maha.n_matched_treated, n_treated);
        assert!(
            result_maha.balance_after.mean_abs_std_diff
                <= result_maha.balance_before.mean_abs_std_diff + 0.2
        );
    }

    // =========================================================================
    // R Validation Tests (Phase 4)
    // =========================================================================

    /// Simple LCG for deterministic random numbers
    fn lcg_rand(seed: &mut u64) -> f64 {
        let a: u64 = 1103515245;
        let c: u64 = 12345;
        let m: u64 = 2_u64.pow(31);
        *seed = (a.wrapping_mul(*seed).wrapping_add(c)) % m;
        (*seed as f64) / (m as f64)
    }

    /// Create dataset matching R's MatchIt validation example.
    fn create_matchit_validation_dataset() -> Dataset {
        let n = 500;
        let mut seed: u64 = 42;

        let mut x1 = Vec::with_capacity(n);
        let mut x2 = Vec::with_capacity(n);
        let mut treatment = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);

        for _ in 0..n {
            // Box-Muller transform for Normal(0,1)
            let u1 = lcg_rand(&mut seed).max(1e-10);
            let u2 = lcg_rand(&mut seed);
            let z1 = ((-2.0_f64 * u1.ln()).sqrt()) * (2.0 * std::f64::consts::PI * u2).cos();
            let z2 = ((-2.0_f64 * u1.ln()).sqrt()) * (2.0 * std::f64::consts::PI * u2).sin();
            x1.push(z1);
            x2.push(z2);

            // Generate treatment based on propensity score
            let ps = 1.0 / (1.0 + (-(-0.5 + 0.8 * z1 + 0.4 * z2)).exp());
            let t = if lcg_rand(&mut seed) < ps { 1.0 } else { 0.0 };
            treatment.push(t);

            // Generate outcome: y = 1 + 0.5*x1 - 0.3*x2 + 0.75*treatment + noise
            let u3 = lcg_rand(&mut seed).max(1e-10);
            let u4 = lcg_rand(&mut seed);
            let noise =
                0.5 * ((-2.0_f64 * u3.ln()).sqrt()) * (2.0 * std::f64::consts::PI * u4).cos();
            let y_i = 1.0 + 0.5 * z1 - 0.3 * z2 + 0.75 * t + noise;
            y.push(y_i);
        }

        let df = df! {
            "y" => y,
            "treatment" => treatment,
            "x1" => x1,
            "x2" => x2
        }
        .unwrap();

        Dataset::new(df)
    }

    /// Shared checks of a full matching on the PS: the optimal total is pinned to optmatch at tol = 1e-8 and lies
    /// below MatchIt's default solution (tol = 0.001); every unit is kept; every set is a star within the bounds;
    /// the weights are MatchIt's formula on p2a's own partition. The PS optimum is not unique on a line, so the
    /// partition itself is not pinned (deviation 3).
    fn check_full_ps(
        case: &p2a_validation::Case,
        r: &MatchResult,
        t: &[bool],
        key: &str,
        default_key: &str,
        bounds: (f64, f64),
    ) {
        let total = r.total_distance.unwrap();
        Check::new(case)
            .scalar(key, total, Tol::iterative(1e-8))
            .finish();
        assert!(
            total < case.scalar(default_key),
            "{total} vs MatchIt's default {}",
            case.scalar(default_key)
        );
        assert_eq!((r.n_matched_treated, r.n_matched_control), (89, 151));
        for sc in r.subclasses.as_ref().unwrap() {
            let (nt, nc) = (
                sc.treated_indices.len() as f64,
                sc.control_indices.len() as f64,
            );
            assert!(
                nt == 1.0 || nc == 1.0,
                "set {}: {nt} treated, {nc} controls",
                sc.subclass_id
            );
            assert!(
                nc / nt >= bounds.0 && nc / nt <= bounds.1,
                "set {}: {nc}/{nt} outside {bounds:?}",
                sc.subclass_id
            );
        }
        assert_eq!(r.weights, weights_from_sets(&r.subclass, t));
    }

    /// matchit(method = "full", distance = "glm"): MatchIt's default bounds (0, Inf).
    #[test]
    fn test_validate_full_matching_total_distance() {
        let (case, ds) = crate::test_support::load_case("matchit_full_cem");
        let r = match_it(
            &ds,
            "treat",
            &COVS,
            full(0.0, f64::INFINITY),
            Some(DistanceMethod::Logit),
        )
        .unwrap();
        check_full_ps(
            &case,
            &r,
            &mask(&ds),
            "full_tight_total_distance",
            "full_total_distance",
            (0.0, f64::INFINITY),
        );
    }

    /// min.controls = 0.5, max.controls = 2, on the PS and on the Mahalanobis distance (where the optimum is
    /// unique, so the partition and the weights are pinned). The old sweep let strata reach 52 controls per
    /// treated unit.
    #[test]
    fn test_validate_full_matching_bounded() {
        let (case, ds) = crate::test_support::load_case("matchit_full_cem");
        let r = match_it(
            &ds,
            "treat",
            &COVS,
            full(0.5, 2.0),
            Some(DistanceMethod::Logit),
        )
        .unwrap();
        check_full_ps(
            &case,
            &r,
            &mask(&ds),
            "full_bounded_tight_total_distance",
            "full_bounded_total_distance",
            (0.5, 2.0),
        );
        let bd = match_it(
            &ds,
            "treat",
            &COVS,
            full(0.5, 2.0),
            Some(DistanceMethod::Mahalanobis),
        )
        .unwrap();
        Check::new(&case)
            .vector(
                "full_maha_bounded_sets",
                &canonical_sets(&bd.subclass),
                Tol::exact(),
            )
            .vector(
                "full_maha_bounded_weights",
                &bd.weights,
                Tol::coef(bd.n_obs),
            )
            .scalar(
                "full_maha_bounded_total_distance",
                bd.total_distance.unwrap(),
                Tol::coef(bd.n_obs),
            )
            .finish();
    }

    /// MatchIt's full-matching weights from a partition: R's own (default) partitions in, R's weights out.
    #[test]
    fn test_validate_full_matching_weights() {
        let (case, ds) = crate::test_support::load_case("matchit_full_cem");
        let t = mask(&ds);
        for (sets, weights) in [
            ("full_subclass", "full_weights"),
            ("full_bounded_subclass", "full_bounded_weights"),
        ] {
            let s: Vec<Option<usize>> = case
                .vector(sets)
                .iter()
                .map(|&v| Some(v as usize))
                .collect();
            Check::new(&case)
                .vector(weights, &weights_from_sets(&s, &t), Tol::coef(t.len()))
                .finish();
        }
    }

    /// Mahalanobis distance, unconstrained: the optimum is unique, so partition, weights and ESS are pinned.
    #[test]
    fn test_validate_full_matching_mahalanobis() {
        let (case, ds) = crate::test_support::load_case("matchit_full_cem");
        let un = match_it(
            &ds,
            "treat",
            &COVS,
            full(0.0, f64::INFINITY),
            Some(DistanceMethod::Mahalanobis),
        )
        .unwrap();
        let n = un.n_obs;
        Check::new(&case)
            .vector(
                "full_maha_tight_sets",
                &canonical_sets(&un.subclass),
                Tol::exact(),
            )
            .vector("full_maha_tight_weights", &un.weights, Tol::coef(n))
            .scalar(
                "full_maha_tight_total_distance",
                un.total_distance.unwrap(),
                Tol::coef(n),
            )
            .vector(
                "full_maha_tight_ess",
                &[un.ess_control, un.ess_treated],
                Tol::stat(n),
            )
            .finish();
        assert!(un.total_distance.unwrap() < case.scalar("full_maha_default_total_distance"));
    }

    /// max_ratio = 1 cannot absorb 151 controls with 89 treated units: optmatch's recovery matches the 79
    /// controls the unconstrained solution can use (omit fraction 1 - 79/151).
    #[test]
    fn test_validate_full_matching_recovery() {
        let (case, ds) = crate::test_support::load_case("matchit_full_cem");
        let r = match_it(
            &ds,
            "treat",
            &COVS,
            full(0.0, 1.0),
            Some(DistanceMethod::Mahalanobis),
        )
        .unwrap();
        Check::new(&case)
            .vector(
                "full_maha_max1_sets",
                &canonical_sets(&r.subclass),
                Tol::exact(),
            )
            .vector("full_maha_max1_weights", &r.weights, Tol::coef(r.n_obs))
            .scalar(
                "full_maha_max1_total_distance",
                r.total_distance.unwrap(),
                Tol::coef(r.n_obs),
            )
            .finish();
        assert_eq!(r.n_matched_control, 79);
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("72 control units are left unmatched")),
            "{:?}",
            r.warnings
        );
    }

    /// max_ratio = 0.5 (at least two treated units per control) is solved transposed, as optmatch; it needs
    /// nt >= 2 nc, so the case uses all treated units and the controls with id <= flip_max_id (129 rows).
    #[test]
    fn test_validate_full_matching_flipped() {
        let (case, ds) = crate::test_support::load_case("matchit_full_cem");
        let col = |c: &str| DesignMatrix::extract_column(ds.df(), c).unwrap();
        let (id, t) = (col("id"), col("treat"));
        let keep: Vec<bool> = (0..id.len())
            .map(|i| t[i] == 1.0 || id[i] <= case.scalar("flip_max_id"))
            .collect();
        let sub = Dataset::new(
            ds.df()
                .filter(&BooleanChunked::from_slice("keep".into(), &keep))
                .unwrap(),
        );
        let r = match_it(
            &sub,
            "treat",
            &COVS,
            full(0.0, 0.5),
            Some(DistanceMethod::Mahalanobis),
        )
        .unwrap();
        assert_eq!(r.n_obs, 129);
        Check::new(&case)
            .vector(
                "full_maha_flip_sets",
                &canonical_sets(&r.subclass),
                Tol::exact(),
            )
            .vector("full_maha_flip_weights", &r.weights, Tol::coef(r.n_obs))
            .scalar(
                "full_maha_flip_total_distance",
                r.total_distance.unwrap(),
                Tol::coef(r.n_obs),
            )
            .finish();
    }

    #[test]
    fn test_validate_full_matching_probit() {
        let (case, ds) = crate::test_support::load_case("matchit_full_cem");
        let r = match_it(
            &ds,
            "treat",
            &COVS,
            full(0.0, f64::INFINITY),
            Some(DistanceMethod::Probit),
        )
        .unwrap();
        Check::new(&case)
            .scalar(
                "full_probit_tight_total_distance",
                r.total_distance.unwrap(),
                Tol::iterative(1e-8),
            )
            .finish();
    }

    /// optmatch fails for min.controls = 2 here (178 controls needed, 151 available; MatchIt: "No matches were
    /// found"); bad bounds and distances are refused with the parameter named.
    #[test]
    fn test_validate_full_matching_rejects_infeasible_and_invalid_bounds() {
        let (_, ds) = crate::test_support::load_case("matchit_full_cem");
        let msg = |m: MatchMethod| match match_it(&ds, "treat", &COVS, m, None) {
            Err(EconError::InvalidSpecification { message }) => message,
            other => panic!("expected InvalidSpecification, got ok = {}", other.is_ok()),
        };
        assert!(msg(full(2.0, f64::INFINITY)).contains("infeasible"));
        for (a, b) in [
            (-1.0, 2.0),
            (f64::NAN, 2.0),
            (3.0, 2.0),
            (0.0, 0.0),
            (0.0, f64::NAN),
        ] {
            let m = msg(full(a, b));
            assert!(m.contains("min_ratio") && m.contains("max_ratio"), "{m}");
        }
        let d = Array2::from_shape_vec((1, 2), vec![0.5, f64::NAN]).unwrap();
        assert!(optimal_full_match(d.view(), 0.0, f64::INFINITY).is_err());
        let d = Array2::from_shape_vec((1, 2), vec![0.5, -0.1]).unwrap();
        assert!(optimal_full_match(d.view(), 0.0, f64::INFINITY).is_err());
    }

    /// WP25-8 (a): above FULL_MATCH_MAX_PAIRS the dense problem is refused before the n_t x n_c
    /// distance matrix is allocated.
    #[test]
    fn test_full_matching_refuses_problems_above_the_pair_cap() {
        let n = 6400;
        let d: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        let x1: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let x2: Vec<f64> = (0..n).map(|i| (1.7 * i as f64).cos()).collect();
        let ds = Dataset::new(df! { "d" => d, "x1" => x1, "x2" => x2 }.unwrap());
        let r = crate::resource::with_alloc_ceiling(Some(64 << 20), || {
            match_it(&ds, "d", &["x1", "x2"], full(0.0, f64::INFINITY), None)
        });
        assert!(
            matches!(r, Err(EconError::InvalidSpecification { ref message })
                if message.contains("10000000 pairs") && message.contains("3200 x 3200")),
            "{:?}",
            r.err()
        );
    }

    #[test]
    fn test_validate_subclass_logit_probit() {
        let (case, ds) = crate::test_support::load_case("matchit_subclass");
        let mut check = Check::new(&case);
        for (lk, dist) in [
            ("logit", DistanceMethod::Logit),
            ("probit", DistanceMethod::Probit),
        ] {
            let r = match_it(
                &ds,
                "treat",
                &COVS,
                MatchMethod::Subclass { n_subclasses: 5 },
                Some(dist),
            )
            .unwrap();
            let sc = r.subclasses.as_ref().unwrap();
            let mut q: Vec<f64> = sc.iter().map(|s| s.ps_range.0).collect();
            q.push(sc.last().unwrap().ps_range.1);
            let labels: Vec<f64> = r
                .subclass
                .iter()
                .map(|s| s.map_or(f64::NAN, |v| v as f64))
                .collect();
            check
                .vector(
                    &format!("subclass5_{lk}_q_cut_tight"),
                    &q,
                    Tol::iterative(1e-8),
                )
                .vector(&format!("subclass5_{lk}_subclass"), &labels, Tol::exact())
                .vector(
                    &format!("subclass5_{lk}_weights"),
                    &r.weights,
                    Tol::coef(r.n_obs),
                );
            assert!(r.warnings.is_empty(), "{:?}", r.warnings);
        }
        check.finish();
    }

    /// Hand case: treated scores 0.1, 0.2, 0.3, 0.4 give q = (0.1, 0.25, 0.4) for k = 2 (type 7:
    /// 1 + 3 * 0.5 = 2.5, 0.2 + 0.5 * 0.1). Controls 0.15 and 0.12 fall in subclass 1, so subclass 2 has no
    /// control and is discarded with a warning (MatchIt's min.n = 0).
    #[test]
    fn test_subclass_without_controls_is_discarded_with_a_warning() {
        let scores = [0.1, 0.2, 0.3, 0.4, 0.15, 0.12];
        let t = [true, true, true, true, false, false];
        let (labels, q, warnings) = subclass_labels(&scores, &t, 2);
        assert_eq!(q, vec![0.1, 0.25, 0.4]);
        assert_eq!(labels, vec![Some(1), Some(1), None, None, Some(1), Some(1)]);
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("subclass 2 holds no control units")),
            "{warnings:?}"
        );
        let (_, _, w2) = subclass_labels(&[0.3; 4], &[true, true, false, false], 2);
        assert!(
            w2.iter()
                .any(|w| w.contains("fewer subclasses were generated than requested (1 of 2)")),
            "{w2:?}"
        );
    }

    // =========================================================================
    // Linear KS, count bounds, bounded control lists, cancellation (T0-WP11)
    // =========================================================================

    fn cancelled<R>(f: impl FnOnce() -> R) -> R {
        use crate::resource::{Budget, CancelToken, with_budget};
        let token = CancelToken::new();
        token.cancel();
        with_budget(Budget::new(None, token), f)
    }

    /// R 4.3.3, MatchIt 4.8.0, on the CSV of create_matchit_validation_dataset() (regenerated
    /// with the same LCG and Box-Muller steps in Python, 17-digit repr):
    ///   m <- matchit(treatment ~ x1 + x2, data = d, method = NULL, distance = "glm")
    ///   summary(m)$sum.all[, "eCDF Max"]   # x1 0.34643081041698437, x2 0.16215117165658516
    ///   (191 treated; equal to ks.test(x[treat == 1], x[treat == 0])$statistic)
    #[test]
    fn test_validate_ks_before_matching_vs_matchit() {
        let r = match_it(
            &create_matchit_validation_dataset(),
            "treatment",
            &["x1", "x2"],
            MatchMethod::default(),
            Some(DistanceMethod::Logit),
        )
        .unwrap();
        assert_eq!(r.n_treated, 191);
        let ks = |name: &str| {
            r.balance_before
                .covariates
                .iter()
                .find(|c| c.name == name)
                .unwrap()
                .ks_statistic
        };
        assert!((ks("x1") - 0.34643081041698437).abs() < 1e-12);
        assert!((ks("x2") - 0.16215117165658516).abs() < 1e-12);
    }

    #[test]
    fn test_matching_rejects_out_of_range_counts() {
        let ds = create_matchit_validation_dataset(); // 191 treated, 309 control
        let run = |m: MatchMethod| {
            match_it(
                &ds,
                "treatment",
                &["x1", "x2"],
                m,
                Some(DistanceMethod::Logit),
            )
        };
        let msg = |r: EconResult<MatchResult>| match r {
            Err(EconError::InvalidSpecification { message }) => message,
            other => panic!("expected InvalidSpecification, got ok={}", other.is_ok()),
        };
        // usize::MAX panicked with "capacity overflow" before the fix; 0 and 310 returned Ok
        for ratio in [0usize, 310, usize::MAX] {
            let m = msg(run(MatchMethod::NearestNeighbor {
                ratio,
                caliper: None,
                replace: true,
            }));
            assert!(m.contains("ratio") && m.contains("309"), "{m}");
        }
        // 0 underflowed (panic) before the fix; 192 returned Ok
        for k in [0usize, 192, usize::MAX] {
            let m = msg(run(MatchMethod::Subclass { n_subclasses: k }));
            assert!(m.contains("n_subclasses") && m.contains("191"), "{m}");
        }
        let m = msg(run(MatchMethod::CoarsenedExact {
            cutpoints: None,
            n_bins: Some(0),
        }));
        assert!(m.contains("n_bins"), "{m}");
        let m = msg(run(MatchMethod::CoarsenedExact {
            cutpoints: Some(vec![vec![0.0]]),
            n_bins: None,
        }));
        // one vector for two covariates: index panic before
        assert!(m.contains("cutpoints"), "{m}");
        assert!(
            run(MatchMethod::NearestNeighbor {
                ratio: 309,
                caliper: None,
                replace: true
            })
            .is_ok()
        );
    }

    #[test]
    fn test_stratified_matching_respects_allocation_ceiling() {
        let ds = create_matchit_validation_dataset();
        // about 191*309/5 ≈ 11.8k (treated, control) pairs * 8 B ≈ 94 KB > 48 KiB. The
        // ceiling is 48 KiB, not 1 KiB: the propensity-score design matrix (500 x 3
        // doubles, 12 KB) and the glm's work arrays are allocated fallibly too and would
        // trip a 1 KiB ceiling before the matched-control lists this test targets.
        for method in [
            MatchMethod::Subclass { n_subclasses: 5 },
            MatchMethod::Full {
                min_ratio: 0.5,
                max_ratio: 2.0,
            },
            MatchMethod::CoarsenedExact {
                cutpoints: None,
                n_bins: Some(2),
            },
        ] {
            let label = format!("{method}");
            let r = crate::resource::with_alloc_ceiling(Some(48 * 1024), || {
                match_it(&ds, "treatment", &["x1", "x2"], method, None)
            });
            // Full matching fails earlier, on the 191 x 309 distance matrix (472 KB)
            let what = if label.starts_with("Optimal Full") {
                "full-matching distance matrix"
            } else {
                "matched control"
            };
            assert!(
                matches!(r, Err(EconError::ResourceLimit(ref m)) if m.contains(what)),
                "{label}: ok={}",
                r.is_ok()
            );
        }
    }

    #[test]
    fn test_full_matching_unchanged_by_two_pass_strata() {
        // S9: matches in treated data order; each treated unit lists its matched
        // set's controls.
        let r = match_it(
            &create_matchit_validation_dataset(),
            "treatment",
            &["x1", "x2"],
            MatchMethod::Full {
                min_ratio: 0.5,
                max_ratio: 2.0,
            },
            Some(DistanceMethod::Logit),
        )
        .unwrap();
        let sc = r.subclasses.as_ref().unwrap();
        let mut expected = Vec::new();
        for s in sc {
            for &ti in &s.treated_indices {
                expected.push((ti, s.control_indices.clone()));
            }
        }
        expected.sort_by_key(|e| e.0);
        let got: Vec<(usize, Vec<usize>)> = r
            .matches
            .iter()
            .filter(|m| m.weight > 0.0)
            .map(|m| (m.treated_idx, m.control_indices.clone()))
            .collect();
        assert_eq!(got, expected);
        assert_eq!(r.matches.len(), r.n_treated);
    }

    #[test]
    fn test_matching_stops_when_cancelled() {
        let ds = create_matchit_validation_dataset();
        let r = cancelled(|| {
            match_it(
                &ds,
                "treatment",
                &["x1", "x2"],
                MatchMethod::default(),
                Some(DistanceMethod::Logit),
            )
        });
        assert!(matches!(r, Err(EconError::Cancelled(_))));
    }

    #[test]
    fn test_distance_matching_stops_when_cancelled() {
        // Mahalanobis and Euclidean matching never estimate a propensity score,
        // so their own loops must check the budget.
        let ds = create_matchit_validation_dataset();
        for replace in [false, true] {
            for distance in [DistanceMethod::Mahalanobis, DistanceMethod::Euclidean] {
                let r = cancelled(|| {
                    match_it(
                        &ds,
                        "treatment",
                        &["x1", "x2"],
                        MatchMethod::NearestNeighbor {
                            ratio: 1,
                            caliper: None,
                            replace,
                        },
                        Some(distance),
                    )
                });
                assert!(
                    matches!(r, Err(EconError::Cancelled(_))),
                    "{distance} replace={replace}"
                );
            }
        }
    }

    // ── T2-WP13: missing values and column types (R case causal_mediation_na) ──

    #[test]
    fn test_validate_na_match_it_columns() {
        use crate::econometrics::treatment::na_fixtures as nf;
        let (case, ds) = crate::test_support::load_case("causal_mediation_na");
        let run = |ds: &Dataset, _y: &str, d: &str, x: &[&str]| {
            match_it(
                ds,
                d,
                x,
                MatchMethod::default(),
                Some(DistanceMethod::Logit),
            )
        };
        nf::assert_refusals(&case, &ds, false, run);
        nf::assert_treatment_dtypes_agree(&case, &ds, run, |r| r.weights.clone());
    }
}
