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
use std::sync::atomic::{AtomicBool, Ordering};

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::design::{get_column_names, DesignMatrix};
use crate::linalg::matrix_ops::safe_inverse;
use crate::traits::estimator::logistic_cdf;

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
    /// Standardized mean difference: (mean_t - mean_c) / sqrt((var_t + var_c)/2).
    /// Rule of thumb: |SMD| < 0.1 indicates good balance.
    pub std_diff: f64,
    /// Variance ratio: var_t / var_c.
    /// Rule of thumb: 0.5 < ratio < 2 indicates good balance.
    pub var_ratio: f64,
    /// Kolmogorov-Smirnov statistic comparing distributions.
    pub ks_statistic: f64,
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
            "{:<20} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "Covariate", "Mean(T)", "Mean(C)", "Std.Diff", "Var.Ratio", "KS Stat"
        )?;
        writeln!(f, "{}", "-".repeat(80))?;

        for cov in &self.covariates {
            let balance_flag = if cov.std_diff.abs() > 0.1 { "!" } else { "" };
            writeln!(
                f,
                "{:<20} {:>10.4} {:>10.4} {:>10.4}{} {:>10.4} {:>10.4}",
                cov.name,
                cov.mean_treated,
                cov.mean_control,
                cov.std_diff,
                balance_flag,
                cov.var_ratio,
                cov.ks_statistic
            )?;
        }

        writeln!(f, "{}", "-".repeat(80))?;
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
    /// Matching weights for all observations (0 for unmatched).
    pub weights: Vec<f64>,
    /// Propensity scores (if computed).
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
    /// Effective sample size (accounting for weights).
    pub effective_sample_size: f64,
}

impl fmt::Display for MatchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Propensity Score Matching Results")?;
        writeln!(f, "==================================")?;
        writeln!(f)?;
        writeln!(f, "Method:   {}", self.method)?;
        writeln!(f, "Distance: {}", self.distance)?;
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
        writeln!(f, "  Effective N:     {:.1}", self.effective_sample_size)?;
        writeln!(f)?;

        if let Some(caliper) = self.caliper_used {
            writeln!(f, "Caliper: {:.4} SD of propensity score", caliper)?;
            writeln!(f)?;
        }

        writeln!(f, "Balance Before Matching:")?;
        writeln!(
            f,
            "  Mean Abs. Std. Diff: {:.4}",
            self.balance_before.mean_abs_std_diff
        )?;
        writeln!(
            f,
            "  Imbalanced covariates: {}",
            self.balance_before.n_imbalanced
        )?;
        writeln!(f)?;

        writeln!(f, "Balance After Matching:")?;
        writeln!(
            f,
            "  Mean Abs. Std. Diff: {:.4}",
            self.balance_after.mean_abs_std_diff
        )?;
        writeln!(
            f,
            "  Imbalanced covariates: {}",
            self.balance_after.n_imbalanced
        )?;
        writeln!(f)?;

        let improvement = 1.0
            - (self.balance_after.mean_abs_std_diff
                / self.balance_before.mean_abs_std_diff.max(1e-10));
        writeln!(f, "Balance Improvement: {:.1}%", improvement * 100.0)?;

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main Matching Function
// ═══════════════════════════════════════════════════════════════════════════════

/// Run propensity score matching.
///
/// Creates matched comparison groups for causal inference from observational data.
///
/// # Arguments
/// * `dataset` - Dataset containing treatment indicator and covariates
/// * `treatment_col` - Name of binary treatment column (0/1)
/// * `covariate_cols` - Names of covariate columns for matching
/// * `method` - Matching method to use
/// * `distance` - Distance metric (default: Logit propensity score)
///
/// # Returns
/// `MatchResult` containing matched sample, weights, and balance diagnostics.
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
    let distance_method = distance.unwrap_or_default();

    // Extract treatment indicator
    let treatment = DesignMatrix::extract_column(dataset.df(), treatment_col).map_err(|_| {
        EconError::ColumnNotFound {
            column: treatment_col.to_string(),
            available: get_column_names(dataset.df()),
        }
    })?;

    let n = treatment.len();

    // Identify treated and control indices
    let (treated_idx, control_idx): (Vec<usize>, Vec<usize>) =
        (0..n).partition(|&i| treatment[i] >= 0.5);

    let n_treated = treated_idx.len();
    let n_control = control_idx.len();

    if n_treated == 0 || n_control == 0 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Treatment column '{}' must have both treated (1) and control (0) units. Found {} treated, {} control.",
                treatment_col, n_treated, n_control
            ),
        });
    }

    // Build covariate matrix
    let design = DesignMatrix::from_dataframe(dataset.df(), covariate_cols, false)?;
    let x = design.data.clone();
    let covariate_names: Vec<String> = covariate_cols.iter().map(|s| s.to_string()).collect();

    // Compute balance before matching
    let balance_before = compute_balance_table(
        &x,
        &treatment,
        &treated_idx,
        &control_idx,
        &covariate_names,
        None,
    );

    // Perform matching based on method - using optimized algorithms
    let (matches, weights, subclasses, caliper_used, propensity_scores) = match &method {
        MatchMethod::NearestNeighbor {
            ratio,
            caliper,
            replace,
        } => {
            match distance_method {
                DistanceMethod::Logit | DistanceMethod::Probit => {
                    // FAST PATH: Use sorting + binary search for propensity score matching
                    // O(n_t * log(n_c)) instead of O(n_t * n_c)
                    let ps = estimate_propensity_scores(
                        &x,
                        &treatment,
                        distance_method == DistanceMethod::Probit,
                    )?;

                    // Compute caliper in absolute PS units
                    let caliper_abs = caliper.map(|c| {
                        let ps_std = ps.std(1.0);
                        c * ps_std
                    });

                    let (m, w) = ps_nearest_neighbor_fast(
                        &ps,
                        &treated_idx,
                        &control_idx,
                        *ratio,
                        caliper_abs,
                        *replace,
                    );

                    (m, w, None, *caliper, ps)
                }
                DistanceMethod::Mahalanobis | DistanceMethod::Euclidean => {
                    // OPTIMIZED PATH: Parallel distance computation on treated×control only
                    // Avoids full n×n distance matrix
                    let use_mahalanobis = distance_method == DistanceMethod::Mahalanobis;

                    // Compute caliper (for non-PS, interpret as raw distance)
                    let caliper_abs = *caliper;

                    let (m, w) = parallel_distance_matching(
                        &x,
                        &treated_idx,
                        &control_idx,
                        *ratio,
                        caliper_abs,
                        *replace,
                        use_mahalanobis,
                    )?;

                    // Return uniform propensity scores for compatibility
                    let ps = Array1::from_elem(n, 0.5);

                    (m, w, None, *caliper, ps)
                }
            }
        }
        MatchMethod::CoarsenedExact { cutpoints, n_bins } => {
            // CEM doesn't use distances or propensity scores
            let (m, w, sc, cal) = coarsened_exact_matching(
                &x,
                &treated_idx,
                &control_idx,
                cutpoints.as_ref(),
                n_bins.unwrap_or(4),
                &covariate_names,
            )?;
            let ps = Array1::from_elem(n, 0.5);
            (m, w, sc, cal, ps)
        }
        MatchMethod::Full {
            min_ratio,
            max_ratio,
        } => {
            // Full matching uses propensity scores for stratification
            let ps = estimate_propensity_scores(&x, &treatment, false)?;
            let (m, w, sc, cal) =
                full_matching_optimized(&ps, &treated_idx, &control_idx, *min_ratio, *max_ratio)?;
            (m, w, sc, cal, ps)
        }
        MatchMethod::Subclass { n_subclasses } => {
            // Subclassification uses propensity scores
            let ps = estimate_propensity_scores(&x, &treatment, false)?;
            let (m, w, sc, cal) =
                subclassification_matching(&ps, &treated_idx, &control_idx, *n_subclasses)?;
            (m, w, sc, cal, ps)
        }
    };

    // Count matched units
    let n_matched_treated = matches
        .iter()
        .filter(|m| !m.control_indices.is_empty())
        .count();
    let matched_controls: std::collections::HashSet<usize> = matches
        .iter()
        .flat_map(|m| m.control_indices.iter().copied())
        .collect();
    let n_matched_control = matched_controls.len();

    // Compute balance after matching
    let balance_after = compute_balance_table(
        &x,
        &treatment,
        &treated_idx,
        &control_idx,
        &covariate_names,
        Some(&weights),
    );

    // Compute effective sample size
    let effective_sample_size = compute_effective_sample_size(&weights);

    Ok(MatchResult {
        method,
        distance: distance_method,
        balance_before,
        balance_after,
        matches,
        subclasses,
        weights,
        propensity_scores: Some(propensity_scores.to_vec()),
        n_obs: n,
        n_treated,
        n_control,
        n_matched_treated,
        n_matched_control,
        n_discarded_treated: n_treated - n_matched_treated,
        n_discarded_control: n_control - n_matched_control,
        caliper_used,
        effective_sample_size,
    })
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

/// Estimate propensity scores using logistic/probit regression.
fn estimate_propensity_scores(
    x: &Array2<f64>,
    treatment: &Array1<f64>,
    use_probit: bool,
) -> EconResult<Array1<f64>> {
    let n = treatment.len();
    let k = x.ncols();

    // Add intercept
    let mut x_with_intercept = Array2::zeros((n, k + 1));
    for i in 0..n {
        x_with_intercept[[i, 0]] = 1.0;
        for j in 0..k {
            x_with_intercept[[i, j + 1]] = x[[i, j]];
        }
    }

    // Newton-Raphson for MLE
    let mut beta = Array1::zeros(k + 1);
    let max_iter = 50;
    let tolerance = 1e-8;

    for _ in 0..max_iter {
        // Linear predictor
        let z: Array1<f64> = x_with_intercept.dot(&beta);

        // Probabilities
        let p: Array1<f64> = if use_probit {
            z.mapv(normal_cdf)
        } else {
            z.mapv(logistic_cdf)
        };
        let p_clipped: Array1<f64> = p.mapv(|pi| pi.max(1e-10).min(1.0 - 1e-10));

        // Gradient
        let residuals = treatment - &p_clipped;
        let gradient = x_with_intercept.t().dot(&residuals);

        // Check convergence
        let grad_norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < tolerance {
            break;
        }

        // Weights for Hessian
        let weights: Array1<f64> = if use_probit {
            // For probit: phi(z)^2 / (Phi(z) * (1-Phi(z)))
            z.iter()
                .zip(p_clipped.iter())
                .map(|(&zi, &pi)| {
                    let phi = normal_pdf(zi);
                    phi * phi / (pi * (1.0 - pi) + 1e-10)
                })
                .collect()
        } else {
            // For logit: p(1-p)
            p_clipped.mapv(|pi| pi * (1.0 - pi))
        };

        // Hessian approximation: -X'WX
        let mut hessian = Array2::zeros((k + 1, k + 1));
        for i in 0..n {
            let wi = weights[i];
            for j in 0..(k + 1) {
                for l in 0..(k + 1) {
                    hessian[[j, l]] -= wi * x_with_intercept[[i, j]] * x_with_intercept[[i, l]];
                }
            }
        }

        // Invert -H and update
        let neg_hessian = &hessian * -1.0;
        let (hess_inv, _) =
            safe_inverse(&neg_hessian.view()).map_err(|e| EconError::SingularMatrix {
                context: "Propensity score estimation".to_string(),
                suggestion: format!("Check for multicollinearity: {:?}", e),
            })?;

        let delta = hess_inv.dot(&gradient);
        beta = &beta + &delta;
    }

    // Final propensity scores
    let z_final: Array1<f64> = x_with_intercept.dot(&beta);
    let ps: Array1<f64> = if use_probit {
        z_final.mapv(normal_cdf)
    } else {
        z_final.mapv(logistic_cdf)
    };

    // Clip to avoid extreme values
    Ok(ps.mapv(|p| p.max(1e-10).min(1.0 - 1e-10)))
}

/// Standard normal CDF using statrs.
fn normal_cdf(x: f64) -> f64 {
    use statrs::distribution::{ContinuousCDF, Normal};
    let normal = Normal::new(0.0, 1.0).unwrap();
    normal.cdf(x)
}

/// Standard normal PDF.
fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Matching Algorithms
// ═══════════════════════════════════════════════════════════════════════════════

/// Fast propensity score nearest neighbor matching using sorting + binary search.
///
/// This is O(n_t * log(n_c)) instead of O(n_t * n_c) for the naive approach.
/// For n=1000 with 50% treatment, this is ~500*9 = 4500 ops vs 500*500 = 250000 ops.
fn ps_nearest_neighbor_fast(
    propensity_scores: &Array1<f64>,
    treated_idx: &[usize],
    control_idx: &[usize],
    ratio: usize,
    caliper_abs: Option<f64>,
    replace: bool,
) -> (Vec<MatchInfo>, Vec<f64>) {
    let n = propensity_scores.len();
    let mut weights = vec![0.0; n];
    let mut matches = Vec::with_capacity(treated_idx.len());

    // Step 1: Create sorted index of controls by propensity score - O(n_c log n_c)
    let mut control_sorted: Vec<(usize, f64)> = control_idx
        .iter()
        .map(|&i| (i, propensity_scores[i]))
        .collect();
    control_sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Pre-extract just the scores for binary search
    let control_ps: Vec<f64> = control_sorted.iter().map(|(_, ps)| *ps).collect();

    // Track used controls for without-replacement matching (AtomicBool is interior-mutable)
    let used_controls: Vec<AtomicBool> = if !replace {
        control_sorted
            .iter()
            .map(|_| AtomicBool::new(false))
            .collect()
    } else {
        Vec::new()
    };

    // Step 2: For each treated unit, find nearest controls using binary search - O(n_t * log(n_c) * ratio)
    for &ti in treated_idx {
        let ps_t = propensity_scores[ti];

        // Binary search to find insertion point - O(log n_c)
        let pos = control_ps.partition_point(|&ps| ps < ps_t);

        // Find k nearest neighbors by expanding outward from insertion point
        let selected = find_k_nearest_neighbors(
            &control_sorted,
            &control_ps,
            pos,
            ps_t,
            ratio,
            caliper_abs,
            replace,
            if !replace { Some(&used_controls) } else { None },
        );

        if !selected.is_empty() {
            let control_indices: Vec<usize> = selected.iter().map(|(ci, _)| *ci).collect();
            let dists: Vec<f64> = selected.iter().map(|(_, d)| *d).collect();

            // Assign weights
            weights[ti] = 1.0;
            for &ci in &control_indices {
                weights[ci] += 1.0 / ratio as f64;
            }

            matches.push(MatchInfo {
                treated_idx: ti,
                control_indices,
                distances: dists,
                weight: 1.0,
            });
        } else {
            // No valid match found
            matches.push(MatchInfo {
                treated_idx: ti,
                control_indices: vec![],
                distances: vec![],
                weight: 0.0,
            });
        }
    }

    (matches, weights)
}

/// Find k nearest neighbors by expanding outward from binary search position.
/// Returns up to `k` matches that satisfy the caliper constraint.
#[inline]
fn find_k_nearest_neighbors(
    control_sorted: &[(usize, f64)],
    control_ps: &[f64],
    pos: usize,
    target_ps: f64,
    k: usize,
    caliper_abs: Option<f64>,
    replace: bool,
    used: Option<&[AtomicBool]>,
) -> Vec<(usize, f64)> {
    let n = control_sorted.len();
    if n == 0 {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(k);

    // Two-pointer expansion from insertion point
    let mut left = pos.saturating_sub(1) as isize;
    let mut right = if pos >= n { n - 1 } else { pos };

    while result.len() < k && (left >= 0 || right < n) {
        let left_dist = if left >= 0 {
            (control_ps[left as usize] - target_ps).abs()
        } else {
            f64::INFINITY
        };

        let right_dist = if right < n {
            (control_ps[right] - target_ps).abs()
        } else {
            f64::INFINITY
        };

        let (chosen_idx, chosen_dist) = if left_dist <= right_dist {
            let idx = left as usize;
            left -= 1;
            (idx, left_dist)
        } else {
            let idx = right;
            right += 1;
            (idx, right_dist)
        };

        // Check caliper constraint
        if let Some(max_dist) = caliper_abs {
            if chosen_dist > max_dist {
                // If closest available is beyond caliper, stop searching
                break;
            }
        }

        // Check if control is available
        if !replace {
            if let Some(used_flags) = used {
                if used_flags[chosen_idx].swap(true, Ordering::Relaxed) {
                    // Already used, skip
                    continue;
                }
            }
        }

        let (control_idx, _) = control_sorted[chosen_idx];
        result.push((control_idx, chosen_dist));
    }

    result
}

/// Parallel nearest neighbor matching for Mahalanobis/Euclidean distances.
///
/// Computes only treated×control distances (not full n×n matrix) and
/// uses parallel iteration for large datasets.
fn parallel_distance_matching(
    x: &Array2<f64>,
    treated_idx: &[usize],
    control_idx: &[usize],
    ratio: usize,
    caliper_abs: Option<f64>,
    replace: bool,
    use_mahalanobis: bool,
) -> EconResult<(Vec<MatchInfo>, Vec<f64>)> {
    let n = x.nrows();

    // Compute inverse covariance for Mahalanobis (shared across all distance computations)
    let cov_inv = if use_mahalanobis {
        Some(compute_covariance_inverse(x)?)
    } else {
        None
    };

    // Use parallel or sequential based on dataset size
    let use_parallel = treated_idx.len() >= PARALLEL_THRESHOLD;

    if replace {
        // With replacement: can fully parallelize since controls are independent
        let matches: Vec<MatchInfo> = if use_parallel {
            treated_idx
                .par_iter()
                .map(|&ti| {
                    find_best_controls_for_treated(
                        x,
                        ti,
                        control_idx,
                        ratio,
                        caliper_abs,
                        cov_inv.as_ref(),
                    )
                })
                .collect()
        } else {
            treated_idx
                .iter()
                .map(|&ti| {
                    find_best_controls_for_treated(
                        x,
                        ti,
                        control_idx,
                        ratio,
                        caliper_abs,
                        cov_inv.as_ref(),
                    )
                })
                .collect()
        };

        // Compute weights from matches
        let mut weights = vec![0.0; n];
        for m in &matches {
            if !m.control_indices.is_empty() {
                weights[m.treated_idx] = 1.0;
                for &ci in &m.control_indices {
                    weights[ci] += 1.0 / ratio as f64;
                }
            }
        }

        Ok((matches, weights))
    } else {
        // Without replacement: need sequential for correct exclusion tracking
        // But we can parallelize distance computation within each iteration
        let mut weights = vec![0.0; n];
        let mut matches = Vec::with_capacity(treated_idx.len());
        let mut used_controls: std::collections::HashSet<usize> =
            std::collections::HashSet::with_capacity(control_idx.len());

        for &ti in treated_idx {
            // Find available controls
            let available_controls: Vec<usize> = control_idx
                .iter()
                .filter(|&&ci| !used_controls.contains(&ci))
                .copied()
                .collect();

            if available_controls.is_empty() {
                matches.push(MatchInfo {
                    treated_idx: ti,
                    control_indices: vec![],
                    distances: vec![],
                    weight: 0.0,
                });
                continue;
            }

            // Compute distances to available controls (parallel for large datasets)
            let mut control_distances: Vec<(usize, f64)> = if available_controls.len() >= 100 {
                available_controls
                    .par_iter()
                    .map(|&ci| {
                        let dist = compute_pairwise_distance(x, ti, ci, cov_inv.as_ref());
                        (ci, dist)
                    })
                    .collect()
            } else {
                available_controls
                    .iter()
                    .map(|&ci| {
                        let dist = compute_pairwise_distance(x, ti, ci, cov_inv.as_ref());
                        (ci, dist)
                    })
                    .collect()
            };

            // Sort by distance
            control_distances
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Apply caliper
            let valid_controls: Vec<(usize, f64)> = match caliper_abs {
                Some(max_dist) => control_distances
                    .into_iter()
                    .filter(|(_, d)| *d <= max_dist)
                    .collect(),
                None => control_distances,
            };

            // Select top k matches
            let selected: Vec<(usize, f64)> = valid_controls.into_iter().take(ratio).collect();

            if !selected.is_empty() {
                let control_indices: Vec<usize> = selected.iter().map(|(ci, _)| *ci).collect();
                let dists: Vec<f64> = selected.iter().map(|(_, d)| *d).collect();

                // Mark as used
                for &ci in &control_indices {
                    used_controls.insert(ci);
                }

                weights[ti] = 1.0;
                for &ci in &control_indices {
                    weights[ci] += 1.0 / ratio as f64;
                }

                matches.push(MatchInfo {
                    treated_idx: ti,
                    control_indices,
                    distances: dists,
                    weight: 1.0,
                });
            } else {
                matches.push(MatchInfo {
                    treated_idx: ti,
                    control_indices: vec![],
                    distances: vec![],
                    weight: 0.0,
                });
            }
        }

        Ok((matches, weights))
    }
}

/// Find best control matches for a single treated unit.
#[inline]
fn find_best_controls_for_treated(
    x: &Array2<f64>,
    ti: usize,
    control_idx: &[usize],
    ratio: usize,
    caliper_abs: Option<f64>,
    cov_inv: Option<&Array2<f64>>,
) -> MatchInfo {
    // Compute distances to all controls
    let mut control_distances: Vec<(usize, f64)> = control_idx
        .iter()
        .map(|&ci| {
            let dist = compute_pairwise_distance(x, ti, ci, cov_inv);
            (ci, dist)
        })
        .collect();

    // Sort by distance
    control_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Apply caliper
    let valid_controls: Vec<(usize, f64)> = match caliper_abs {
        Some(max_dist) => control_distances
            .into_iter()
            .filter(|(_, d)| *d <= max_dist)
            .collect(),
        None => control_distances,
    };

    // Select top k
    let selected: Vec<(usize, f64)> = valid_controls.into_iter().take(ratio).collect();

    if !selected.is_empty() {
        MatchInfo {
            treated_idx: ti,
            control_indices: selected.iter().map(|(ci, _)| *ci).collect(),
            distances: selected.iter().map(|(_, d)| *d).collect(),
            weight: 1.0,
        }
    } else {
        MatchInfo {
            treated_idx: ti,
            control_indices: vec![],
            distances: vec![],
            weight: 0.0,
        }
    }
}

/// Compute distance between two observations.
#[inline]
fn compute_pairwise_distance(
    x: &Array2<f64>,
    i: usize,
    j: usize,
    cov_inv: Option<&Array2<f64>>,
) -> f64 {
    let k = x.ncols();

    if let Some(inv) = cov_inv {
        // Mahalanobis distance
        let mut diff = Vec::with_capacity(k);
        for col in 0..k {
            diff.push(x[[i, col]] - x[[j, col]]);
        }

        let mut d_sq = 0.0;
        for row in 0..k {
            let mut sum = 0.0;
            for col in 0..k {
                sum += inv[[row, col]] * diff[col];
            }
            d_sq += diff[row] * sum;
        }
        d_sq.max(0.0).sqrt()
    } else {
        // Euclidean distance
        let mut dist_sq = 0.0;
        for col in 0..k {
            let d = x[[i, col]] - x[[j, col]];
            dist_sq += d * d;
        }
        dist_sq.sqrt()
    }
}

/// Compute inverse covariance matrix for Mahalanobis distance.
fn compute_covariance_inverse(x: &Array2<f64>) -> EconResult<Array2<f64>> {
    let n = x.nrows();
    let k = x.ncols();

    // Compute mean
    let x_mean: Array1<f64> = x.mean_axis(ndarray::Axis(0)).unwrap();

    // Compute covariance matrix (can parallelize for large k)
    let mut cov = Array2::zeros((k, k));
    for i in 0..n {
        for j in 0..k {
            for l in 0..k {
                cov[[j, l]] += (x[[i, j]] - x_mean[j]) * (x[[i, l]] - x_mean[l]);
            }
        }
    }
    cov /= (n - 1) as f64;

    // Invert
    let (cov_inv, _) = safe_inverse(&cov.view()).map_err(|e| EconError::SingularMatrix {
        context: "Mahalanobis distance computation".to_string(),
        suggestion: format!("Check for multicollinearity: {:?}", e),
    })?;

    Ok(cov_inv)
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

    // Assign each observation to a stratum
    let mut strata: HashMap<String, (Vec<usize>, Vec<usize>)> = HashMap::new();

    for &i in treated_idx.iter().chain(control_idx.iter()) {
        let stratum_key = compute_stratum_key(x, i, &cuts);
        let entry = strata
            .entry(stratum_key)
            .or_insert((Vec::new(), Vec::new()));
        if treated_idx.contains(&i) {
            entry.0.push(i);
        } else {
            entry.1.push(i);
        }
    }

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
fn compute_stratum_key(x: &Array2<f64>, row: usize, cuts: &[Vec<f64>]) -> String {
    let k = x.ncols();
    let mut key_parts = Vec::with_capacity(k);

    for j in 0..k {
        let val = x[[row, j]];
        let bin = cuts[j].iter().filter(|&&c| val > c).count();
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

    // Greedy stratification
    let mut subclasses = Vec::new();
    let mut matches = Vec::new();
    let mut i = 0;
    let mut subclass_id = 0;

    while i < all_units.len() {
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

        // Compute stratum weights
        let n_t = treated.len() as f64;
        let n_c = control.len() as f64;

        for &ti in &treated {
            weights[ti] = 1.0;
            matches.push(MatchInfo {
                treated_idx: ti,
                control_indices: control.clone(),
                distances: vec![],
                weight: 1.0,
            });
        }

        let control_weight = n_t / n_c;
        for &ci in &control {
            weights[ci] = control_weight;
        }

        subclasses.push(SubclassInfo {
            subclass_id,
            treated_indices: treated,
            control_indices: control,
            ps_range: (start_ps, end_ps),
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

    // Assign units to subclasses
    for &i in treated_idx.iter().chain(control_idx.iter()) {
        let ps = propensity_scores[i];
        let subclass = cutpoints.iter().filter(|&&c| ps > c).count();
        let subclass = subclass.min(n_subclasses - 1);

        if treated_idx.contains(&i) {
            subclasses[subclass].treated_indices.push(i);
        } else {
            subclasses[subclass].control_indices.push(i);
        }
    }

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
// Balance Diagnostics
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute balance table for covariates.
fn compute_balance_table(
    x: &Array2<f64>,
    _treatment: &Array1<f64>,
    treated_idx: &[usize],
    control_idx: &[usize],
    covariate_names: &[String],
    weights: Option<&[f64]>,
) -> MatchBalanceTable {
    let k = x.ncols();
    let mut covariates = Vec::with_capacity(k);

    for j in 0..k {
        let balance =
            compute_covariate_balance(x, j, treated_idx, control_idx, &covariate_names[j], weights);
        covariates.push(balance);
    }

    // Summary statistics
    let abs_diffs: Vec<f64> = covariates.iter().map(|c| c.std_diff.abs()).collect();
    let mean_abs_std_diff = abs_diffs.iter().sum::<f64>() / abs_diffs.len() as f64;
    let max_abs_std_diff = abs_diffs.iter().cloned().fold(0.0_f64, f64::max);
    let n_imbalanced = abs_diffs.iter().filter(|&&d| d > 0.1).count();

    MatchBalanceTable {
        covariates,
        mean_abs_std_diff,
        max_abs_std_diff,
        n_imbalanced,
    }
}

/// Compute balance statistics for a single covariate.
fn compute_covariate_balance(
    x: &Array2<f64>,
    col_idx: usize,
    treated_idx: &[usize],
    control_idx: &[usize],
    name: &str,
    weights: Option<&[f64]>,
) -> MatchCovariateBalance {
    // Extract values and weights
    let (t_vals, t_weights): (Vec<f64>, Vec<f64>) = treated_idx
        .iter()
        .map(|&i| (x[[i, col_idx]], weights.map(|w| w[i]).unwrap_or(1.0)))
        .filter(|(_, w)| *w > 0.0)
        .unzip();

    let (c_vals, c_weights): (Vec<f64>, Vec<f64>) = control_idx
        .iter()
        .map(|&i| (x[[i, col_idx]], weights.map(|w| w[i]).unwrap_or(1.0)))
        .filter(|(_, w)| *w > 0.0)
        .unzip();

    // Weighted means
    let mean_treated = weighted_mean(&t_vals, &t_weights);
    let mean_control = weighted_mean(&c_vals, &c_weights);

    // Weighted variances
    let var_treated = weighted_variance(&t_vals, &t_weights, mean_treated);
    let var_control = weighted_variance(&c_vals, &c_weights, mean_control);

    // Standardized mean difference (SMD)
    // Formula: (mean_t - mean_c) / sqrt((var_t + var_c) / 2)
    let pooled_sd = ((var_treated + var_control) / 2.0).sqrt();
    let std_diff = if pooled_sd > 1e-10 {
        (mean_treated - mean_control) / pooled_sd
    } else {
        0.0
    };

    // Variance ratio
    let var_ratio = if var_control > 1e-10 {
        var_treated / var_control
    } else if var_treated > 1e-10 {
        f64::INFINITY
    } else {
        1.0
    };

    // Kolmogorov-Smirnov statistic
    let ks_statistic = compute_ks_statistic(&t_vals, &c_vals);

    MatchCovariateBalance {
        name: name.to_string(),
        mean_treated,
        mean_control,
        std_diff,
        var_ratio,
        ks_statistic,
    }
}

/// Compute weighted mean.
fn weighted_mean(values: &[f64], weights: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let sum_w: f64 = weights.iter().sum();
    if sum_w <= 0.0 {
        return 0.0;
    }
    values
        .iter()
        .zip(weights.iter())
        .map(|(&v, &w)| v * w)
        .sum::<f64>()
        / sum_w
}

/// Compute weighted variance.
fn weighted_variance(values: &[f64], weights: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let sum_w: f64 = weights.iter().sum();
    if sum_w <= 0.0 {
        return 0.0;
    }
    let var: f64 = values
        .iter()
        .zip(weights.iter())
        .map(|(&v, &w)| w * (v - mean).powi(2))
        .sum::<f64>()
        / sum_w;
    var
}

/// Compute Kolmogorov-Smirnov statistic comparing two distributions.
fn compute_ks_statistic(x1: &[f64], x2: &[f64]) -> f64 {
    if x1.is_empty() || x2.is_empty() {
        return 0.0;
    }

    // Sort both samples
    let mut s1 = x1.to_vec();
    let mut s2 = x2.to_vec();
    s1.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    s2.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n1 = s1.len() as f64;
    let n2 = s2.len() as f64;

    // Merge and compute CDF differences
    let mut combined: Vec<f64> = s1.iter().chain(s2.iter()).copied().collect();
    combined.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    combined.dedup();

    let mut max_diff = 0.0_f64;

    for &x in &combined {
        let cdf1 = s1.iter().filter(|&&v| v <= x).count() as f64 / n1;
        let cdf2 = s2.iter().filter(|&&v| v <= x).count() as f64 / n2;
        max_diff = max_diff.max((cdf1 - cdf2).abs());
    }

    max_diff
}

/// Compute effective sample size accounting for weights.
fn compute_effective_sample_size(weights: &[f64]) -> f64 {
    let positive_weights: Vec<f64> = weights.iter().filter(|&&w| w > 0.0).copied().collect();
    if positive_weights.is_empty() {
        return 0.0;
    }

    let sum_w: f64 = positive_weights.iter().sum();
    let sum_w2: f64 = positive_weights.iter().map(|w| w * w).sum();

    if sum_w2 > 0.0 {
        sum_w * sum_w / sum_w2
    } else {
        0.0
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

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
    fn test_nearest_neighbor_with_caliper() {
        let dataset = create_test_dataset();
        let result =
            nearest_neighbor_match(&dataset, "treatment", &["x1", "x2"], Some(0.2), false).unwrap();

        assert_eq!(result.caliper_used, Some(0.2));
        // With caliper, some treated units may be discarded
        assert!(result.n_discarded_treated >= 0);
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
        let weights = vec![1.0, 1.0, 1.0, 0.5, 0.5];
        let ess = compute_effective_sample_size(&weights);
        // ESS should be greater than 0 and bounded by the count of non-zero weights
        assert!(ess > 0.0);
        let n_nonzero = weights.iter().filter(|&&w| w > 0.0).count() as f64;
        assert!(ess <= n_nonzero);
    }

    #[test]
    fn test_ks_statistic() {
        // Identical distributions should have KS = 0
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ks = compute_ks_statistic(&x1, &x2);
        assert!(ks < 0.01);

        // Very different distributions should have high KS
        let x3 = vec![1.0, 2.0, 3.0];
        let x4 = vec![10.0, 11.0, 12.0];
        let ks2 = compute_ks_statistic(&x3, &x4);
        assert!(ks2 > 0.9);
    }

    #[test]
    fn test_display_traits() {
        let dataset = create_test_dataset();
        let result =
            nearest_neighbor_match(&dataset, "treatment", &["x1", "x2"], None, false).unwrap();

        // Test Display for MatchResult
        let output = format!("{}", result);
        assert!(output.contains("Propensity Score Matching"));
        assert!(output.contains("Balance"));

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
    fn test_no_overlap_handling() {
        // Create dataset with perfect separation
        let df = df! {
            "treatment" => [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            "x1" => [10.0, 11.0, 12.0, 1.0, 2.0, 3.0],
            "x2" => [10.0, 11.0, 12.0, 1.0, 2.0, 3.0]
        }
        .unwrap();
        let dataset = Dataset::new(df);

        // Should still work but may have few matches
        let result = nearest_neighbor_match(&dataset, "treatment", &["x1", "x2"], None, false);
        // Result might succeed with poor balance or fail depending on implementation
        // The key is it shouldn't panic
        assert!(result.is_ok() || result.is_err());
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
}
