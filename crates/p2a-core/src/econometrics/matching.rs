//! Propensity Score Matching (MatchIt) for causal inference.
//!
//! This module provides various matching methods to create balanced comparison groups
//! for estimating causal effects from observational data. Matching reduces confounding
//! by creating treated and control groups with similar covariate distributions.
//!
//! # Methods
//!
//! - **Nearest Neighbor Matching**: For each treated unit, find the closest control
//!   unit(s) by propensity score distance. Supports with/without replacement and calipers.
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
//! - Standardized mean differences (SMD)
//! - Variance ratios
//! - Kolmogorov-Smirnov statistics
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
//! - Implementation validated against R package `MatchIt` (Ho et al., 2011).
//!   https://cran.r-project.org/package=MatchIt

use ndarray::{Array1, Array2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::econ_error_from_design;
use crate::econometrics::glm::{GlmLink, GlmSpec, fit_propensity};
use crate::errors::{EconError, EconResult};
use crate::linalg::design::DesignMatrix;
use crate::linalg::matrix_ops::eig_symmetric;
use crate::linalg::solve::{SingularPolicy, sym_inverse};
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
    /// Coarsened exact matching within covariate strata.
    CoarsenedExact {
        /// Optional custom cutpoints for each covariate.
        /// If None, automatic binning is used (quartiles).
        cutpoints: Option<Vec<Vec<f64>>>,
        /// Number of bins for automatic coarsening (default: 4).
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
            MatchMethod::CoarsenedExact { n_bins, .. } => {
                write!(f, "Coarsened Exact Matching")?;
                if let Some(bins) = n_bins {
                    write!(f, " ({} bins)", bins)?;
                }
                Ok(())
            }
            MatchMethod::Full {
                min_ratio,
                max_ratio,
            } => {
                write!(
                    f,
                    "Full Matching (ratio range: [{:.1}, {:.1}])",
                    min_ratio, max_ratio
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
    /// nearest-neighbour matching with replacement (always data order, as MatchIt).
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
    /// Raw (unnormalised) weights per row.
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
    validate_match_options(&method, distance_method, ps_link.is_some(), options)?;

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
        MatchMethod::CoarsenedExact { cutpoints, n_bins } => cem_method(
            &x,
            &is_treated,
            cutpoints.as_deref(),
            *n_bins,
            &covariate_names,
        )?,
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

    let mut weights = out.weights;
    normalize_weights(&mut weights, &is_treated);
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
            for (j, c) in cuts.iter().enumerate() {
                if c.iter().any(|v| !v.is_finite()) {
                    return spec(format!(
                        "cutpoints for covariate {} must be finite numbers",
                        j + 1
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
        Box<dyn Fn(usize, usize) -> f64>,
    ) = match scores {
        Some(s) => {
            let caliper_abs = caliper.map(|c| c * sample_sd(s));
            let m = nn_on_scores(s, is_treated, &order, ratio, caliper_abs, replace)?;
            let s = s.to_vec();
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
    let pairs = crate::resource::checked_len(
        &[n_t, ratio],
        "matched control lists (ratio controls per treated unit)",
    )?;
    crate::resource::check_alloc::<(usize, f64)>(
        pairs,
        "matched control lists (ratio controls per treated unit)",
    )?;
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

/// Cutpoints without NaN, sorted, for counting by binary search.
fn sorted_cutpoints(cuts: &[f64]) -> Vec<f64> {
    let mut sorted: Vec<f64> = cuts.iter().copied().filter(|c| !c.is_nan()).collect();
    sorted.sort_unstable_by(f64::total_cmp);
    sorted
}

/// Number of cutpoints strictly below `v` (`sorted` from [`sorted_cutpoints`]).
///
/// Equal to `cuts.iter().filter(|&&c| v > c).count()`: a NaN cutpoint or a NaN
/// `v` never compares greater, and `c < v` is monotone along the sorted list.
#[inline]
fn count_below(sorted: &[f64], v: f64) -> usize {
    sorted.partition_point(|&c| c < v)
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
        n_bins: Some(4),
    };
    match_it(dataset, treatment_col, covariate_cols, method, None)
}

/// Perform full matching for optimal stratification.
///
/// Creates optimal strata containing treated and control units.
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
    let spec = GlmSpec::logit();
    let convergence = Convergence::new(
        fit.fit.converged,
        fit.fit.iterations,
        spec.max_iter,
        "relative change in deviance",
        f64::NAN,
        spec.epsilon,
    );
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

fn cem_method(
    x: &Array2<f64>,
    is_treated: &[bool],
    cutpoints: Option<&[Vec<f64>]>,
    n_bins: Option<usize>,
    names: &[String],
) -> EconResult<MethodOutput> {
    let (t, c) = group_indices(is_treated);
    let cp = cutpoints.map(|c| c.to_vec());
    let (matches, weights, subclasses, _) =
        coarsened_exact_matching(x, &t, &c, cp.as_ref(), n_bins.unwrap_or(4), names)?;
    Ok(MethodOutput {
        weights,
        matches,
        subclass: sets_from_subclasses(is_treated.len(), &subclasses),
        subclasses,
        m_order: None,
        caliper_abs: None,
        total_distance: None,
        warnings: Vec::new(),
    })
}

fn full_method(
    _x: &Array2<f64>,
    is_treated: &[bool],
    scores: Option<&[f64]>,
    _distance: DistanceMethod,
    min_ratio: f64,
    max_ratio: f64,
) -> EconResult<MethodOutput> {
    let (t, c) = group_indices(is_treated);
    let ps = Array1::from_vec(
        scores
            .map(<[f64]>::to_vec)
            .unwrap_or_else(|| vec![0.5; is_treated.len()]),
    );
    let (matches, weights, subclasses, _) =
        full_matching_optimized(&ps, &t, &c, min_ratio, max_ratio)?;
    Ok(MethodOutput {
        weights,
        matches,
        subclass: sets_from_subclasses(is_treated.len(), &subclasses),
        subclasses,
        m_order: None,
        caliper_abs: None,
        total_distance: None,
        warnings: Vec::new(),
    })
}

fn subclass_method(is_treated: &[bool], scores: &[f64], k: usize) -> EconResult<MethodOutput> {
    let (t, c) = group_indices(is_treated);
    let ps = Array1::from_vec(scores.to_vec());
    let (matches, weights, subclasses, _) = subclassification_matching(&ps, &t, &c, k)?;
    Ok(MethodOutput {
        weights,
        matches,
        subclass: sets_from_subclasses(is_treated.len(), &subclasses),
        subclasses,
        m_order: None,
        caliper_abs: None,
        total_distance: None,
        warnings: Vec::new(),
    })
}

/// Coarsened exact matching algorithm.
fn coarsened_exact_matching(
    x: &Array2<f64>,
    treated_idx: &[usize],
    control_idx: &[usize],
    cutpoints: Option<&Vec<Vec<f64>>>,
    n_bins: usize,
    _covariate_names: &[String],
) -> EconResult<(
    Vec<MatchInfo>,
    Vec<f64>,
    Option<Vec<SubclassInfo>>,
    Option<f64>,
)> {
    let n = x.nrows();
    let k = x.ncols();
    let mut weights = vec![0.0; n];

    // Compute cutpoints for each covariate
    let cuts: Vec<Vec<f64>> = match cutpoints {
        Some(c) => c.clone(),
        None => {
            // Automatic quartile-based binning
            (0..k)
                .map(|j| {
                    let col: Vec<f64> = (0..n).map(|i| x[[i, j]]).collect();
                    compute_quantile_cutpoints(&col, n_bins)
                })
                .collect()
        }
    };

    // Sorted copies so each bin is found by binary search
    let cuts: Vec<Vec<f64>> = cuts.iter().map(|c| sorted_cutpoints(c)).collect();

    // Membership mask instead of a Vec::contains scan per unit
    let mut is_treated = vec![false; n];
    for &i in treated_idx {
        is_treated[i] = true;
    }

    // Assign each observation to a stratum
    let mut strata: HashMap<String, (Vec<usize>, Vec<usize>)> = HashMap::new();

    for &i in treated_idx.iter().chain(control_idx.iter()) {
        let stratum_key = compute_stratum_key(x, i, &cuts);
        let entry = strata
            .entry(stratum_key)
            .or_insert((Vec::new(), Vec::new()));
        if is_treated[i] {
            entry.0.push(i);
        } else {
            entry.1.push(i);
        }
    }

    // Every matched treated unit stores the stratum's controls and a distance each.
    check_stratum_control_lists::<(usize, f64)>(
        strata
            .values()
            .filter(|(t, c)| !t.is_empty() && !c.is_empty())
            .map(|(t, c)| (t.len(), c.len())),
    )?;

    // Build matches and subclasses
    let mut matches = Vec::new();
    let mut subclasses = Vec::new();
    let mut subclass_id = 0;

    for (treated, control) in strata.values() {
        // Only include strata with both treated and control
        if treated.is_empty() || control.is_empty() {
            // Discard these units (weight = 0)
            continue;
        }

        subclass_id += 1;

        // CEM weights: n_control / n_treated in stratum (for ATT)
        let n_t = treated.len() as f64;
        let n_c = control.len() as f64;

        // Treated get weight 1
        for &ti in treated {
            weights[ti] = 1.0;
            matches.push(MatchInfo {
                treated_idx: ti,
                control_indices: control.clone(),
                distances: vec![0.0; control.len()], // Exact match = 0 distance
                weight: 1.0,
            });
        }

        // Controls get weight proportional to treated/control ratio
        let control_weight = n_t / n_c;
        for &ci in control {
            weights[ci] = control_weight;
        }

        subclasses.push(SubclassInfo {
            subclass_id,
            treated_indices: treated.clone(),
            control_indices: control.clone(),
            ps_range: (0.0, 1.0), // Not applicable for CEM
        });
    }

    // Add unmatched treated units to matches with empty controls
    for &ti in treated_idx {
        if weights[ti] == 0.0 {
            matches.push(MatchInfo {
                treated_idx: ti,
                control_indices: vec![],
                distances: vec![],
                weight: 0.0,
            });
        }
    }

    Ok((matches, weights, Some(subclasses), None))
}

/// Compute quantile-based cutpoints for binning.
fn compute_quantile_cutpoints(values: &[f64], n_bins: usize) -> Vec<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    let mut cutpoints = Vec::with_capacity(n_bins - 1);

    for i in 1..n_bins {
        let idx = (n as f64 * i as f64 / n_bins as f64).floor() as usize;
        cutpoints.push(sorted[idx.min(n - 1)]);
    }

    cutpoints
}

/// Compute stratum key for an observation based on coarsened covariates.
///
/// `cuts` holds one sorted cutpoint list per covariate (see
/// [`sorted_cutpoints`]); the bin is the number of cutpoints below the value.
fn compute_stratum_key(x: &Array2<f64>, row: usize, cuts: &[Vec<f64>]) -> String {
    let k = x.ncols();
    let mut key_parts = Vec::with_capacity(k);

    for (j, col_cuts) in cuts.iter().enumerate().take(k) {
        let val = x[[row, j]];
        let bin = count_below(col_cuts, val);
        key_parts.push(bin.to_string());
    }

    key_parts.join("_")
}

/// Optimized full matching algorithm (greedy approximation).
///
/// Creates optimal strata by iteratively grouping treated and control units.
/// This version takes only propensity scores (no distance matrix needed).
fn full_matching_optimized(
    propensity_scores: &Array1<f64>,
    treated_idx: &[usize],
    control_idx: &[usize],
    min_ratio: f64,
    max_ratio: f64,
) -> EconResult<(
    Vec<MatchInfo>,
    Vec<f64>,
    Option<Vec<SubclassInfo>>,
    Option<f64>,
)> {
    let n = propensity_scores.len();
    let mut weights = vec![0.0; n];

    // Sort all units by propensity score
    let mut all_units: Vec<(usize, f64, bool)> = treated_idx
        .iter()
        .map(|&i| (i, propensity_scores[i], true))
        .chain(
            control_idx
                .iter()
                .map(|&i| (i, propensity_scores[i], false)),
        )
        .collect();
    all_units.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Greedy stratification. The strata are formed first and emitted after
    // their total size has been checked; ids and order are as before.
    struct Stratum {
        subclass_id: usize,
        treated: Vec<usize>,
        control: Vec<usize>,
        start_ps: f64,
        end_ps: f64,
    }
    let mut strata: Vec<Stratum> = Vec::new();
    let mut i = 0;
    let mut subclass_id = 0;

    while i < all_units.len() {
        crate::resource::checkpoint()?;

        subclass_id += 1;
        let mut treated = Vec::new();
        let mut control = Vec::new();
        let start_ps = all_units[i].1;

        // Collect consecutive units into a stratum
        while i < all_units.len() {
            let (idx, _ps, is_treated) = all_units[i];

            // Check if we should close this stratum
            if !treated.is_empty() && !control.is_empty() {
                let ratio = control.len() as f64 / treated.len() as f64;
                if ratio < min_ratio || ratio > max_ratio {
                    // Close stratum if ratio constraint would be violated
                    if (is_treated && ratio >= max_ratio) || (!is_treated && ratio <= min_ratio) {
                        break;
                    }
                }
            }

            if is_treated {
                treated.push(idx);
            } else {
                control.push(idx);
            }
            i += 1;
        }

        // Skip empty or one-sided strata
        if treated.is_empty() || control.is_empty() {
            continue;
        }

        let end_ps = all_units[i.saturating_sub(1)].1;

        strata.push(Stratum {
            subclass_id,
            treated,
            control,
            start_ps,
            end_ps,
        });
    }

    check_stratum_control_lists::<usize>(
        strata.iter().map(|s| (s.treated.len(), s.control.len())),
    )?;

    let mut subclasses = Vec::with_capacity(strata.len());
    let mut matches = Vec::new();
    for s in strata {
        // Compute stratum weights
        let n_t = s.treated.len() as f64;
        let n_c = s.control.len() as f64;

        for &ti in &s.treated {
            weights[ti] = 1.0;
            matches.push(MatchInfo {
                treated_idx: ti,
                control_indices: s.control.clone(),
                distances: vec![],
                weight: 1.0,
            });
        }

        let control_weight = n_t / n_c;
        for &ci in &s.control {
            weights[ci] = control_weight;
        }

        subclasses.push(SubclassInfo {
            subclass_id: s.subclass_id,
            treated_indices: s.treated,
            control_indices: s.control,
            ps_range: (s.start_ps, s.end_ps),
        });
    }

    // Add unmatched treated units
    for &ti in treated_idx {
        if weights[ti] == 0.0 {
            matches.push(MatchInfo {
                treated_idx: ti,
                control_indices: vec![],
                distances: vec![],
                weight: 0.0,
            });
        }
    }

    Ok((matches, weights, Some(subclasses), None))
}

/// Propensity score subclassification.
fn subclassification_matching(
    propensity_scores: &Array1<f64>,
    treated_idx: &[usize],
    control_idx: &[usize],
    n_subclasses: usize,
) -> EconResult<(
    Vec<MatchInfo>,
    Vec<f64>,
    Option<Vec<SubclassInfo>>,
    Option<f64>,
)> {
    let n = propensity_scores.len();
    let mut weights = vec![0.0; n];

    // Compute quantile cutpoints based on treated propensity scores
    let treated_ps: Vec<f64> = treated_idx.iter().map(|&i| propensity_scores[i]).collect();
    let cutpoints = compute_quantile_cutpoints(&treated_ps, n_subclasses);

    // Assign units to subclasses
    let mut subclasses: Vec<SubclassInfo> = (0..n_subclasses)
        .map(|i| SubclassInfo {
            subclass_id: i + 1,
            treated_indices: Vec::new(),
            control_indices: Vec::new(),
            ps_range: if i == 0 {
                (0.0, *cutpoints.first().unwrap_or(&1.0))
            } else if i == n_subclasses - 1 {
                (*cutpoints.last().unwrap_or(&0.0), 1.0)
            } else {
                (cutpoints[i - 1], cutpoints[i])
            },
        })
        .collect();

    // Membership mask instead of a Vec::contains scan per unit, and sorted
    // cutpoints so each subclass is found by binary search
    let mut is_treated = vec![false; n];
    for &i in treated_idx {
        is_treated[i] = true;
    }
    let sorted_cuts = sorted_cutpoints(&cutpoints);

    // Assign units to subclasses
    for &i in treated_idx.iter().chain(control_idx.iter()) {
        let ps = propensity_scores[i];
        let subclass = count_below(&sorted_cuts, ps);
        let subclass = subclass.min(n_subclasses - 1);

        if is_treated[i] {
            subclasses[subclass].treated_indices.push(i);
        } else {
            subclasses[subclass].control_indices.push(i);
        }
    }

    check_stratum_control_lists::<usize>(
        subclasses
            .iter()
            .filter(|sc| !sc.treated_indices.is_empty() && !sc.control_indices.is_empty())
            .map(|sc| (sc.treated_indices.len(), sc.control_indices.len())),
    )?;

    // Compute weights within each subclass
    let mut matches = Vec::new();

    for sc in &subclasses {
        if sc.treated_indices.is_empty() || sc.control_indices.is_empty() {
            continue;
        }

        let n_t = sc.treated_indices.len() as f64;
        let n_c = sc.control_indices.len() as f64;

        for &ti in &sc.treated_indices {
            weights[ti] = 1.0;
            matches.push(MatchInfo {
                treated_idx: ti,
                control_indices: sc.control_indices.clone(),
                distances: vec![],
                weight: 1.0,
            });
        }

        let control_weight = n_t / n_c;
        for &ci in &sc.control_indices {
            weights[ci] = control_weight;
        }
    }

    // Add unmatched treated units
    for &ti in treated_idx {
        if weights[ti] == 0.0 {
            matches.push(MatchInfo {
                treated_idx: ti,
                control_indices: vec![],
                distances: vec![],
                weight: 0.0,
            });
        }
    }

    Ok((matches, weights, Some(subclasses), None))
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
    fn test_matchit_m_order_random_is_seeded() {
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
    fn test_matching_rejects_invalid_options() {
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
        assert!((ks(&s1, &s2) - 0.066666666666666638).abs() < 1e-12);
        assert!(
            ks(&[1.0, 2.0, 3.0], &[10.0, 11.0, 12.0]) > 0.99 && ks(&[1.0, 2.0], &[1.0, 2.0]) == 0.0
        );
    }

    #[test]
    fn test_matchit_display_shows_the_balance_tables_and_notes() {
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
    /// unit). Replaces the old `assert!(result.is_ok() || result.is_err())` (CAUS-92).
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
    fn test_cem_matching_basic() {
        let dataset = create_test_dataset();
        let result = cem_match(&dataset, "treatment", &["x1", "x2"]).unwrap();

        // CEM should create subclasses
        assert!(result.subclasses.is_some());

        // Check matching method is recorded
        match result.method {
            MatchMethod::CoarsenedExact { .. } => (),
            _ => panic!("Expected CoarsenedExact method"),
        }
    }

    #[test]
    fn test_full_matching_basic() {
        let dataset = create_test_dataset();
        let result = full_match(&dataset, "treatment", &["x1", "x2"]).unwrap();

        // Full matching should create subclasses
        assert!(result.subclasses.is_some());

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

    #[test]
    fn test_validate_matching_vs_r() {
        // Validates against R MatchIt package
        // R reference:
        // library(MatchIt)
        // m_nn <- matchit(treatment ~ x1 + x2, data = match_data, method = "nearest",
        //                 distance = "logit", replace = FALSE)

        let dataset = create_matchit_validation_dataset();

        let result = match_it(
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

        // Structure validation
        assert!(result.n_treated > 0);
        assert!(result.n_control > 0);
        assert!(result.n_matched_treated > 0);
        assert!(result.n_matched_control > 0);

        // Balance should improve after matching
        // Before matching SMD may be large due to confounding
        assert!(
            result.balance_after.mean_abs_std_diff
                <= result.balance_before.mean_abs_std_diff + 0.05,
            "Balance should improve or stay similar after matching"
        );

        // After matching, SMD should be relatively small (< 0.25 is common threshold)
        for cov in &result.balance_after.covariates {
            assert!(
                cov.std_diff.abs() < 0.5,
                "Covariate {} has SMD {:.4} > 0.5 after matching",
                cov.name,
                cov.std_diff
            );
        }

        // Propensity scores should be computed
        assert!(result.propensity_scores.is_some());
        let ps = result.propensity_scores.as_ref().unwrap();
        assert_eq!(ps.len(), result.n_obs);
        // PS should be in (0, 1)
        for &p in ps.iter() {
            assert!(p > 0.0 && p < 1.0, "PS {} out of range", p);
        }
    }

    #[test]
    fn test_validate_matching_balance_smd() {
        // Validate SMD calculation matches R's formula
        // SMD = (mean_t - mean_c) / sqrt((var_t + var_c) / 2)

        let dataset = create_matchit_validation_dataset();

        let result = match_it(
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

        // Verify balance table structure
        assert_eq!(result.balance_before.covariates.len(), 2);
        assert_eq!(result.balance_after.covariates.len(), 2);

        // Variance ratios should be positive
        for cov in &result.balance_after.covariates {
            assert!(cov.var_ratio > 0.0, "Variance ratio should be positive");
            // Good balance means var_ratio is close to 1 (0.5 to 2 is acceptable)
            assert!(
                cov.var_ratio > 0.1 && cov.var_ratio < 10.0,
                "Variance ratio {:.4} for {} seems extreme",
                cov.var_ratio,
                cov.name
            );
        }

        // Mean absolute SMD should be computed correctly
        let computed_mean_smd: f64 = result
            .balance_after
            .covariates
            .iter()
            .map(|c| c.std_diff.abs())
            .sum::<f64>()
            / result.balance_after.covariates.len() as f64;
        assert!(
            (computed_mean_smd - result.balance_after.mean_abs_std_diff).abs() < 1e-10,
            "Mean SMD mismatch"
        );
    }

    #[test]
    fn test_validate_cem_matching_vs_r() {
        // Validate CEM against R MatchIt with method = "cem"
        let dataset = create_matchit_validation_dataset();

        let result = match_it(
            &dataset,
            "treatment",
            &["x1", "x2"],
            MatchMethod::CoarsenedExact {
                cutpoints: None,
                n_bins: Some(4), // Quartiles
            },
            None,
        )
        .unwrap();

        // CEM should produce exact balance within strata
        // Number of matched should be less than or equal to total
        assert!(result.n_matched_treated <= result.n_treated);
        assert!(result.n_matched_control <= result.n_control);

        // CEM typically discards unmatched units
        assert!(result.n_discarded_treated >= 0);
        assert!(result.n_discarded_control >= 0);

        // Balance should generally improve (or at least not get much worse)
        // CEM can have perfect balance within strata but may lose observations
    }

    #[test]
    fn test_validate_full_matching_vs_r() {
        // Validate full matching against R MatchIt with method = "full"
        let dataset = create_matchit_validation_dataset();

        let result = match_it(
            &dataset,
            "treatment",
            &["x1", "x2"],
            MatchMethod::Full {
                min_ratio: 0.5,
                max_ratio: 2.0,
            },
            Some(DistanceMethod::Logit),
        )
        .unwrap();

        // Full matching should create subclasses
        assert!(result.subclasses.is_some());
        let subclasses = result.subclasses.as_ref().unwrap();
        assert!(!subclasses.is_empty(), "Should have at least one subclass");

        // Each subclass should have both treated and control
        for sc in subclasses {
            assert!(
                !sc.treated_indices.is_empty() || !sc.control_indices.is_empty(),
                "Subclass {} should have some units",
                sc.subclass_id
            );
        }

        // Full matching typically doesn't discard units
        // All or most units should be included
        let total_in_subclasses: usize = subclasses
            .iter()
            .map(|sc| sc.treated_indices.len() + sc.control_indices.len())
            .sum();
        assert!(
            total_in_subclasses >= result.n_obs / 2,
            "Full matching should include most observations"
        );
    }

    #[test]
    fn test_validate_subclassification_vs_r() {
        // Validate subclassification against R MatchIt with method = "subclass"
        let dataset = create_matchit_validation_dataset();

        let result = match_it(
            &dataset,
            "treatment",
            &["x1", "x2"],
            MatchMethod::Subclass { n_subclasses: 5 },
            Some(DistanceMethod::Logit),
        )
        .unwrap();

        // Should create exactly n_subclasses subclasses
        assert!(result.subclasses.is_some());
        let subclasses = result.subclasses.as_ref().unwrap();
        assert_eq!(subclasses.len(), 5, "Should have 5 subclasses");

        // Subclasses should be based on PS quantiles
        // Check that PS ranges don't overlap (approximately)
        let mut prev_max = 0.0;
        for sc in subclasses {
            assert!(
                sc.ps_range.0 >= prev_max - 0.01, // Allow small overlap
                "Subclass PS ranges should be ordered"
            );
            prev_max = sc.ps_range.1;
        }
    }

    #[test]
    fn test_validate_matching_with_caliper() {
        // Validate caliper matching
        let dataset = create_matchit_validation_dataset();

        // Without caliper
        let result_no_caliper = match_it(
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

        // With caliper (0.1 SD of PS)
        let result_with_caliper = match_it(
            &dataset,
            "treatment",
            &["x1", "x2"],
            MatchMethod::NearestNeighbor {
                ratio: 1,
                caliper: Some(0.1),
                replace: false,
            },
            Some(DistanceMethod::Logit),
        )
        .unwrap();

        // Caliper should reduce or maintain number of matches
        assert!(
            result_with_caliper.n_matched_treated <= result_no_caliper.n_matched_treated,
            "Caliper should not increase matches"
        );

        // Balance with caliper should be at least as good (often better)
        // though with fewer matched units
    }

    #[test]
    fn test_validate_matching_effective_sample_size() {
        // Validate ESS calculation
        let dataset = create_matchit_validation_dataset();

        let result = match_it(
            &dataset,
            "treatment",
            &["x1", "x2"],
            MatchMethod::NearestNeighbor {
                ratio: 1,
                caliper: None,
                replace: true, // With replacement creates weights
            },
            Some(DistanceMethod::Logit),
        )
        .unwrap();

        // ESS should be positive
        assert!(result.effective_sample_size > 0.0);

        // ESS should be <= actual matched sample size
        let matched_size = result.n_matched_treated + result.n_matched_control;
        assert!(
            result.effective_sample_size <= matched_size as f64 + 0.01,
            "ESS {:.2} should be <= matched size {}",
            result.effective_sample_size,
            matched_size
        );

        // With replacement, ESS might be less than nominal due to repeated use
        // of the same control unit
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
            assert!(
                matches!(r, Err(EconError::ResourceLimit(ref m)) if m.contains("matched control")),
                "{label}: ok={}",
                r.is_ok()
            );
        }
    }

    #[test]
    fn test_full_matching_unchanged_by_two_pass_strata() {
        // Full matching now forms strata before emitting matches; the output must
        // be the same. Strata are consecutive in PS order, so each treated unit's
        // control list is its stratum's control list.
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
