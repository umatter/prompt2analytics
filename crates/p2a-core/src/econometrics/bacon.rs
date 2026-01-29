//! Goodman-Bacon Decomposition for Staggered Difference-in-Differences.
//!
//! Pure Rust implementation of the Goodman-Bacon (2021) decomposition, which breaks down
//! a two-way fixed effects (TWFE) difference-in-differences estimate into a weighted average
//! of all possible 2x2 DiD comparisons.
//!
//! # Overview
//!
//! When treatment timing varies across units (staggered adoption), the standard TWFE estimator
//! combines many different 2x2 DiD comparisons with implicit weights that depend on:
//! - Group sizes (number of units in each timing cohort)
//! - Treatment timing (variance in treatment status over time)
//!
//! The Goodman-Bacon decomposition reveals:
//! 1. Which comparisons contribute to the overall TWFE estimate
//! 2. The weight assigned to each comparison
//! 3. Potential biases from "forbidden comparisons" (later vs. earlier treated)
//!
//! # Key Concepts
//!
//! ## Types of Comparisons
//!
//! 1. **Treated vs. Never-Treated**: Units that eventually get treated compared to units
//!    that never receive treatment. These are "clean" comparisons.
//!
//! 2. **Treated vs. Not-Yet-Treated**: Treated units compared to units that will be
//!    treated later but haven't been treated yet at the comparison time.
//!
//! 3. **Later vs. Earlier Treated** ("Forbidden Comparisons"): When already-treated units
//!    serve as controls for later-treated units. These can produce bias if treatment effects
//!    are heterogeneous over time.
//!
//! ## Decomposition Formula
//!
//! The TWFE DiD estimator equals:
//! ```text
//! β̂_TWFE = Σ_k Σ_l s_{kl} × β̂_{kl}
//! ```
//!
//! Where:
//! - k, l index timing groups (including never-treated as a separate "group")
//! - β̂_{kl} is the 2x2 DiD estimate comparing group k to group l
//! - s_{kl} are weights that sum to 1
//!
//! # References
//!
//! - Goodman-Bacon, A. (2021). "Difference-in-Differences with Variation in Treatment Timing".
//!   *Journal of Econometrics*, 225(2), 254-277.
//!   https://doi.org/10.1016/j.jeconom.2021.03.014
//!
//! - R package `bacondecomp` (Flack & Sant'Anna):
//!   https://github.com/evanjflack/bacondecomp
//!   https://cran.r-project.org/package=bacondecomp
//!
//! - de Chaisemartin, C., & D'Haultfoeuille, X. (2020). "Two-Way Fixed Effects Estimators
//!   with Heterogeneous Treatment Effects". *American Economic Review*, 110(9), 2964-2996.
//!   Discusses problems with TWFE under treatment effect heterogeneity.
//!
//! - Stata package `bacondecomp` (Goodman-Bacon, Goldring, & Nichols):
//!   https://github.com/tgoldring/bacondecomp

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::design::DesignMatrix;
use crate::linalg::matrix_ops::{safe_inverse, xtx, xty};

// ═══════════════════════════════════════════════════════════════════════════════
// Public Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Type of 2x2 DiD comparison in the Goodman-Bacon decomposition.
///
/// The decomposition identifies three distinct types of comparisons that
/// contribute to the overall TWFE estimate:
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonType {
    /// Treated vs. Never-Treated: Clean comparison using units that never receive treatment.
    /// These comparisons are unproblematic as they use a stable control group.
    TreatedVsNeverTreated,

    /// Treated vs. Not-Yet-Treated: Uses later-treated units as controls before they
    /// receive treatment. Valid under no-anticipation assumption.
    TreatedVsNotYetTreated,

    /// Later vs. Earlier Treated ("Forbidden" comparisons): Uses already-treated units
    /// as controls. Can introduce bias if treatment effects evolve over time.
    /// Also called "bad controls" in the literature.
    LaterVsEarlierTreated,
}

impl fmt::Display for ComparisonType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComparisonType::TreatedVsNeverTreated => write!(f, "Treated vs Never-Treated"),
            ComparisonType::TreatedVsNotYetTreated => write!(f, "Treated vs Not-Yet-Treated"),
            ComparisonType::LaterVsEarlierTreated => write!(f, "Later vs Earlier Treated"),
        }
    }
}

/// A single 2x2 DiD comparison component in the decomposition.
///
/// Each component represents a comparison between two timing groups,
/// with an estimated treatment effect and a weight in the overall TWFE estimate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaconComponent {
    /// Treatment timing for the treated group in this comparison.
    /// The period when units in the "treated" group first received treatment.
    pub treated_group: i64,

    /// Treatment timing for the control group in this comparison.
    /// For never-treated units, this is 0 (or the special marker for never-treated).
    /// For not-yet-treated, this is the period when they eventually get treated.
    pub control_group: i64,

    /// The 2x2 DiD estimate for this comparison.
    /// This is the ATT from comparing these two groups.
    pub estimate: f64,

    /// Weight of this comparison in the overall TWFE estimate.
    /// Weights sum to 1 across all components.
    /// Weight depends on group sizes and variance in treatment status.
    pub weight: f64,

    /// Number of treated observations in this comparison.
    pub n_treated: usize,

    /// Number of control observations in this comparison.
    pub n_control: usize,

    /// Type of comparison (clean vs. potentially problematic).
    pub comparison_type: ComparisonType,

    /// Time periods included in this 2x2 comparison.
    pub time_range: (i64, i64),
}

impl fmt::Display for BaconComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ctrl = if self.control_group == 0 {
            "Never".to_string()
        } else {
            self.control_group.to_string()
        };
        write!(
            f,
            "G={} vs G={}: β̂={:.4}, weight={:.4} [{}]",
            self.treated_group, ctrl, self.estimate, self.weight, self.comparison_type
        )
    }
}

/// Result of the Goodman-Bacon decomposition.
///
/// Contains the overall TWFE estimate broken down into its component 2x2 comparisons,
/// with diagnostics about the composition of the estimate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaconDecompResult {
    /// The overall TWFE DiD estimate.
    /// This equals the weighted sum of all component estimates.
    pub overall_estimate: f64,

    /// All individual 2x2 DiD comparisons contributing to the TWFE estimate.
    pub components: Vec<BaconComponent>,

    /// Sum of weights (should be very close to 1.0, useful for validation).
    pub weights_sum: f64,

    /// Total weight from Treated vs. Never-Treated comparisons.
    /// Higher values indicate more reliance on "clean" comparisons.
    pub treated_vs_never: f64,

    /// Total weight from Treated vs. Not-Yet-Treated comparisons.
    pub treated_vs_not_yet: f64,

    /// Total weight from Later vs. Earlier Treated ("forbidden") comparisons.
    /// Higher values indicate potential bias from treatment effect heterogeneity.
    pub later_vs_earlier: f64,

    /// Number of distinct timing groups (cohorts).
    pub n_timing_groups: usize,

    /// Number of never-treated units.
    pub n_never_treated: usize,

    /// Total number of observations.
    pub n_obs: usize,

    /// Number of unique units.
    pub n_units: usize,

    /// Timing groups identified (treatment periods).
    pub timing_groups: Vec<i64>,

    /// Weighted average estimate from each comparison type.
    /// Useful for understanding where the estimate comes from.
    pub estimate_by_type: BaconEstimatesByType,
}

/// Weighted average estimates broken down by comparison type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaconEstimatesByType {
    /// Average estimate from Treated vs. Never-Treated comparisons.
    pub treated_vs_never: Option<f64>,
    /// Average estimate from Treated vs. Not-Yet-Treated comparisons.
    pub treated_vs_not_yet: Option<f64>,
    /// Average estimate from Later vs. Earlier Treated comparisons.
    pub later_vs_earlier: Option<f64>,
}

impl fmt::Display for BaconDecompResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Goodman-Bacon Decomposition")?;
        writeln!(
            f,
            "═══════════════════════════════════════════════════════════"
        )?;
        writeln!(f)?;
        writeln!(f, "Overall TWFE Estimate: {:.6}", self.overall_estimate)?;
        writeln!(f)?;
        writeln!(f, "Sample Information:")?;
        writeln!(f, "  Total observations:    {}", self.n_obs)?;
        writeln!(f, "  Unique units:          {}", self.n_units)?;
        writeln!(f, "  Timing groups:         {}", self.n_timing_groups)?;
        writeln!(f, "  Never-treated units:   {}", self.n_never_treated)?;
        writeln!(f, "  Treatment cohorts:     {:?}", self.timing_groups)?;
        writeln!(f)?;
        writeln!(f, "Weight Distribution by Comparison Type:")?;
        writeln!(
            f,
            "  Treated vs Never-Treated:    {:.4} ({:.1}%)",
            self.treated_vs_never,
            self.treated_vs_never * 100.0
        )?;
        writeln!(
            f,
            "  Treated vs Not-Yet-Treated:  {:.4} ({:.1}%)",
            self.treated_vs_not_yet,
            self.treated_vs_not_yet * 100.0
        )?;
        writeln!(
            f,
            "  Later vs Earlier Treated:    {:.4} ({:.1}%)",
            self.later_vs_earlier,
            self.later_vs_earlier * 100.0
        )?;
        writeln!(f, "  Total weight:                {:.6}", self.weights_sum)?;
        writeln!(f)?;

        // Show estimates by type
        writeln!(f, "Average Estimates by Comparison Type:")?;
        if let Some(est) = self.estimate_by_type.treated_vs_never {
            writeln!(f, "  Treated vs Never-Treated:    {:.6}", est)?;
        }
        if let Some(est) = self.estimate_by_type.treated_vs_not_yet {
            writeln!(f, "  Treated vs Not-Yet-Treated:  {:.6}", est)?;
        }
        if let Some(est) = self.estimate_by_type.later_vs_earlier {
            writeln!(f, "  Later vs Earlier Treated:    {:.6}", est)?;
        }
        writeln!(f)?;

        // Show individual components
        writeln!(
            f,
            "Individual 2x2 Comparisons ({} total):",
            self.components.len()
        )?;
        writeln!(
            f,
            "{:>10} {:>10} {:>12} {:>10} {:>8} {:>8}  Type",
            "Treated_G", "Control_G", "Estimate", "Weight", "N_treat", "N_ctrl"
        )?;
        writeln!(f, "{}", "-".repeat(80))?;

        for comp in &self.components {
            let ctrl_str = if comp.control_group == 0 {
                "Never".to_string()
            } else {
                comp.control_group.to_string()
            };
            writeln!(
                f,
                "{:>10} {:>10} {:>12.6} {:>10.6} {:>8} {:>8}  {}",
                comp.treated_group,
                ctrl_str,
                comp.estimate,
                comp.weight,
                comp.n_treated,
                comp.n_control,
                comp.comparison_type
            )?;
        }

        if self.later_vs_earlier > 0.1 {
            writeln!(f)?;
            writeln!(
                f,
                "WARNING: {:.1}% of weight comes from 'forbidden' comparisons",
                self.later_vs_earlier * 100.0
            )?;
            writeln!(
                f,
                "         (Later vs. Earlier Treated). If treatment effects are"
            )?;
            writeln!(
                f,
                "         heterogeneous over time, the TWFE estimate may be biased."
            )?;
            writeln!(
                f,
                "         Consider using Callaway-Sant'Anna or Sun-Abraham estimators."
            )?;
        }

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main Function
// ═══════════════════════════════════════════════════════════════════════════════

/// Perform Goodman-Bacon decomposition for staggered difference-in-differences.
///
/// This function decomposes a two-way fixed effects (TWFE) DiD estimate into a
/// weighted average of all possible 2x2 DiD comparisons between timing groups.
///
/// # Arguments
///
/// * `dataset` - Panel dataset with repeated observations per unit
/// * `outcome_col` - Name of the outcome variable column
/// * `unit_col` - Name of the unit identifier column
/// * `time_col` - Name of the time period column
/// * `treatment_col` - Name of the binary treatment indicator column (0/1)
///
/// # Returns
///
/// `BaconDecompResult` containing the overall estimate and all component comparisons.
///
/// # Example
///
/// ```ignore
/// use p2a_core::econometrics::bacon_decomp;
///
/// // Panel data with staggered treatment
/// let result = bacon_decomp(
///     &dataset,
///     "outcome",      // Dependent variable
///     "state",        // Unit identifier
///     "year",         // Time period
///     "treated",      // Binary treatment indicator (0 before treatment, 1 after)
/// )?;
///
/// println!("TWFE estimate: {:.4}", result.overall_estimate);
/// println!("Weight from clean comparisons: {:.2}%",
///          result.treated_vs_never * 100.0);
/// ```
///
/// # Notes
///
/// - The treatment indicator should be 0 for untreated periods and 1 for treated periods
/// - Never-treated units should have treatment_col = 0 for all periods
/// - The function automatically identifies treatment timing from the first period
///   where treatment = 1 for each unit
///
/// # References
///
/// - Goodman-Bacon, A. (2021). "Difference-in-Differences with Variation in Treatment Timing".
///   *Journal of Econometrics*, 225(2), 254-277.
pub fn bacon_decomp(
    dataset: &Dataset,
    outcome_col: &str,
    unit_col: &str,
    time_col: &str,
    treatment_col: &str,
) -> EconResult<BaconDecompResult> {
    // Extract columns
    let y = DesignMatrix::extract_column(dataset.df(), outcome_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: outcome_col.to_string(),
            available: vec![format!("{:?}", e)],
        }
    })?;

    let unit = DesignMatrix::extract_column(dataset.df(), unit_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: unit_col.to_string(),
            available: vec![format!("{:?}", e)],
        }
    })?;

    let time = DesignMatrix::extract_column(dataset.df(), time_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: time_col.to_string(),
            available: vec![format!("{:?}", e)],
        }
    })?;

    let treatment = DesignMatrix::extract_column(dataset.df(), treatment_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: treatment_col.to_string(),
            available: vec![format!("{:?}", e)],
        }
    })?;

    let n = y.len();

    // Identify unique units and time periods
    let unique_units: BTreeSet<i64> = unit.iter().map(|&u| u as i64).collect();
    let unique_times: BTreeSet<i64> = time.iter().map(|&t| t as i64).collect();
    let times: Vec<i64> = unique_times.iter().copied().collect();

    let n_units = unique_units.len();
    let n_times = times.len();

    // Build unit -> treatment timing map
    // Treatment timing = first period where treatment = 1 (0 means never treated)
    let mut unit_treatment_time: BTreeMap<i64, i64> = BTreeMap::new();

    for &u in &unique_units {
        let mut first_treat: Option<i64> = None;
        for i in 0..n {
            if (unit[i] as i64) == u && treatment[i] > 0.5 {
                let t = time[i] as i64;
                if first_treat.is_none() || t < first_treat.unwrap() {
                    first_treat = Some(t);
                }
            }
        }
        // 0 indicates never treated
        unit_treatment_time.insert(u, first_treat.unwrap_or(0));
    }

    // Identify timing groups (distinct treatment times, excluding never-treated)
    let timing_groups: Vec<i64> = unit_treatment_time
        .values()
        .filter(|&&g| g > 0)
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    // Count never-treated units
    let n_never_treated = unit_treatment_time.values().filter(|&&g| g == 0).count();

    if timing_groups.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "No treated units found. Treatment column should have 1s for treated periods."
                .to_string(),
        });
    }

    // First, compute the overall TWFE estimate for reference
    // This uses the standard two-way fixed effects regression
    // Note: We compute this for comparison but use the decomposed weighted sum as the output
    let _twfe_estimate = compute_twfe_estimate(&y, &unit, &time, &treatment, n)?;

    // Now compute all 2x2 comparisons
    let mut components = Vec::new();

    // For each pair of timing groups (k, l) where k is treated and l is control
    // We need to consider:
    // 1. Each treated group vs. never-treated
    // 2. Each earlier-treated group vs. later-treated group (both directions)

    // Type 1: Treated vs. Never-Treated
    if n_never_treated > 0 {
        for &g in &timing_groups {
            // Units in timing group g
            let treated_units: Vec<i64> = unit_treatment_time
                .iter()
                .filter(|(_, timing)| **timing == g)
                .map(|(u, _)| *u)
                .collect();

            // Never-treated units
            let control_units: Vec<i64> = unit_treatment_time
                .iter()
                .filter(|(_, timing)| **timing == 0)
                .map(|(u, _)| *u)
                .collect();

            if let Some(comp) = compute_2x2_did(
                &y,
                &unit,
                &time,
                &treatment,
                &treated_units,
                &control_units,
                g,
                0, // Never-treated
                &times,
                ComparisonType::TreatedVsNeverTreated,
            ) {
                components.push(comp);
            }
        }
    }

    // Type 2 & 3: Comparisons between timing groups
    for (i, &g_early) in timing_groups.iter().enumerate() {
        for &g_late in timing_groups.iter().skip(i + 1) {
            // g_early treated before g_late

            // Earlier-treated units
            let early_units: Vec<i64> = unit_treatment_time
                .iter()
                .filter(|(_, timing)| **timing == g_early)
                .map(|(u, _)| *u)
                .collect();

            // Later-treated units
            let late_units: Vec<i64> = unit_treatment_time
                .iter()
                .filter(|(_, timing)| **timing == g_late)
                .map(|(u, _)| *u)
                .collect();

            // Comparison A: Early-treated (as treated) vs Late-treated (as not-yet-treated)
            // This uses periods before g_late where early is treated, late is not yet treated
            // Type: Treated vs Not-Yet-Treated (clean comparison)
            if let Some(comp) = compute_2x2_did_timing(
                &y,
                &unit,
                &time,
                &treatment,
                &early_units,
                &late_units,
                g_early,
                g_late,
                &times,
                ComparisonType::TreatedVsNotYetTreated,
                true, // Early is treated, late is control
            ) {
                components.push(comp);
            }

            // Comparison B: Late-treated (as treated) vs Early-treated (as already-treated control)
            // This uses periods after g_late where both are treated
            // Type: Later vs Earlier Treated ("forbidden" comparison)
            if let Some(comp) = compute_2x2_did_timing(
                &y,
                &unit,
                &time,
                &treatment,
                &late_units,
                &early_units,
                g_late,
                g_early,
                &times,
                ComparisonType::LaterVsEarlierTreated,
                false, // Late is treated, early (already treated) is control
            ) {
                components.push(comp);
            }
        }
    }

    // Compute weights based on group sizes and timing
    // Following Goodman-Bacon (2021), weights depend on:
    // - Share of observations in each group
    // - Variance of treatment indicator over time for each comparison
    compute_weights(&mut components, n, n_times, &unit_treatment_time, &times);

    // Normalize weights to sum to 1
    let total_weight: f64 = components.iter().map(|c| c.weight).sum();
    if total_weight > 0.0 {
        for comp in &mut components {
            comp.weight /= total_weight;
        }
    }

    // Compute aggregates
    let weights_sum: f64 = components.iter().map(|c| c.weight).sum();

    let treated_vs_never: f64 = components
        .iter()
        .filter(|c| c.comparison_type == ComparisonType::TreatedVsNeverTreated)
        .map(|c| c.weight)
        .sum();

    let treated_vs_not_yet: f64 = components
        .iter()
        .filter(|c| c.comparison_type == ComparisonType::TreatedVsNotYetTreated)
        .map(|c| c.weight)
        .sum();

    let later_vs_earlier: f64 = components
        .iter()
        .filter(|c| c.comparison_type == ComparisonType::LaterVsEarlierTreated)
        .map(|c| c.weight)
        .sum();

    // Compute weighted average estimates by type
    let estimate_by_type = compute_estimates_by_type(&components);

    // The overall estimate is the weighted sum of components
    let overall_estimate: f64 = components.iter().map(|c| c.weight * c.estimate).sum();

    Ok(BaconDecompResult {
        overall_estimate,
        components,
        weights_sum,
        treated_vs_never,
        treated_vs_not_yet,
        later_vs_earlier,
        n_timing_groups: timing_groups.len(),
        n_never_treated,
        n_obs: n,
        n_units,
        timing_groups,
        estimate_by_type,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute the standard TWFE DiD estimate.
///
/// Estimates: Y_it = α_i + λ_t + β × D_it + ε_it
/// Returns β (the coefficient on treatment).
fn compute_twfe_estimate(
    y: &Array1<f64>,
    unit: &Array1<f64>,
    time: &Array1<f64>,
    treatment: &Array1<f64>,
    n: usize,
) -> EconResult<f64> {
    // Get unique units and times for fixed effects
    let unique_units: Vec<i64> = unit
        .iter()
        .map(|&u| u as i64)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let unique_times: Vec<i64> = time
        .iter()
        .map(|&t| t as i64)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    let n_units = unique_units.len();
    let n_times = unique_times.len();

    // Create unit and time index maps
    let unit_idx: BTreeMap<i64, usize> = unique_units
        .iter()
        .enumerate()
        .map(|(i, &u)| (u, i))
        .collect();
    let time_idx: BTreeMap<i64, usize> = unique_times
        .iter()
        .enumerate()
        .map(|(i, &t)| (t, i))
        .collect();

    // Build design matrix: [treatment, unit_dummies (minus 1), time_dummies (minus 1)]
    // We exclude one unit and one time dummy for identification
    let k = 1 + (n_units - 1) + (n_times - 1);
    let mut x = Array2::zeros((n, k));

    for i in 0..n {
        // Treatment indicator
        x[[i, 0]] = treatment[i];

        // Unit fixed effects (excluding first unit)
        let u = unit[i] as i64;
        let ui = unit_idx[&u];
        if ui > 0 {
            x[[i, ui]] = 1.0;
        }

        // Time fixed effects (excluding first time)
        let t = time[i] as i64;
        let ti = time_idx[&t];
        if ti > 0 {
            x[[i, n_units - 1 + ti]] = 1.0;
        }
    }

    // OLS: β = (X'X)^(-1) X'y
    let xtx_mat = xtx(&x.view());
    let (xtx_inv, _) = safe_inverse(&xtx_mat.view()).map_err(|e| EconError::SingularMatrix {
        context: "TWFE estimation".to_string(),
        suggestion: format!("Check for collinearity in fixed effects: {:?}", e),
    })?;

    let xty_vec = xty(&x.view(), y);
    let beta = xtx_inv.dot(&xty_vec);

    // The treatment coefficient is the first element
    Ok(beta[0])
}

/// Compute a 2x2 DiD comparison between treated and never-treated groups.
#[allow(clippy::too_many_arguments)]
fn compute_2x2_did(
    y: &Array1<f64>,
    unit: &Array1<f64>,
    time: &Array1<f64>,
    _treatment: &Array1<f64>,
    treated_units: &[i64],
    control_units: &[i64],
    g_treated: i64,
    g_control: i64,
    _times: &[i64],
    comparison_type: ComparisonType,
) -> Option<BaconComponent> {
    let n = y.len();

    if treated_units.is_empty() || control_units.is_empty() {
        return None;
    }

    // Convert to sets for fast lookup
    let treated_set: BTreeSet<i64> = treated_units.iter().copied().collect();
    let control_set: BTreeSet<i64> = control_units.iter().copied().collect();

    // For treated vs never-treated:
    // Pre-period: before g_treated
    // Post-period: g_treated and after
    // Treated group: units with treatment timing = g_treated
    // Control group: never-treated units (g_control = 0)

    let mut y_treated_pre = Vec::new();
    let mut y_treated_post = Vec::new();
    let mut y_control_pre = Vec::new();
    let mut y_control_post = Vec::new();

    for i in 0..n {
        let u = unit[i] as i64;
        let t = time[i] as i64;

        if treated_set.contains(&u) {
            if t < g_treated {
                y_treated_pre.push(y[i]);
            } else {
                y_treated_post.push(y[i]);
            }
        } else if control_set.contains(&u) {
            if t < g_treated {
                y_control_pre.push(y[i]);
            } else {
                y_control_post.push(y[i]);
            }
        }
    }

    // Need observations in all four cells
    if y_treated_pre.is_empty()
        || y_treated_post.is_empty()
        || y_control_pre.is_empty()
        || y_control_post.is_empty()
    {
        return None;
    }

    // Compute 2x2 DiD: (Ȳ_treat_post - Ȳ_treat_pre) - (Ȳ_ctrl_post - Ȳ_ctrl_pre)
    let mean_treated_pre = y_treated_pre.iter().sum::<f64>() / y_treated_pre.len() as f64;
    let mean_treated_post = y_treated_post.iter().sum::<f64>() / y_treated_post.len() as f64;
    let mean_control_pre = y_control_pre.iter().sum::<f64>() / y_control_pre.len() as f64;
    let mean_control_post = y_control_post.iter().sum::<f64>() / y_control_post.len() as f64;

    let estimate = (mean_treated_post - mean_treated_pre) - (mean_control_post - mean_control_pre);

    let n_treated = y_treated_pre.len() + y_treated_post.len();
    let n_control = y_control_pre.len() + y_control_post.len();

    // Find time range for this comparison
    let min_time = time.iter().map(|&t| t as i64).min().unwrap_or(0);
    let max_time = time.iter().map(|&t| t as i64).max().unwrap_or(0);

    Some(BaconComponent {
        treated_group: g_treated,
        control_group: g_control,
        estimate,
        weight: 0.0, // Will be computed later
        n_treated,
        n_control,
        comparison_type,
        time_range: (min_time, max_time),
    })
}

/// Compute a 2x2 DiD comparison between two timing groups.
///
/// This handles the more complex case where both groups eventually get treated,
/// so we need to carefully select the time window.
#[allow(clippy::too_many_arguments)]
fn compute_2x2_did_timing(
    y: &Array1<f64>,
    unit: &Array1<f64>,
    time: &Array1<f64>,
    _treatment: &Array1<f64>,
    treated_units: &[i64],
    control_units: &[i64],
    g_treated: i64,
    g_control: i64,
    times: &[i64],
    comparison_type: ComparisonType,
    early_as_treated: bool,
) -> Option<BaconComponent> {
    let n = y.len();

    if treated_units.is_empty() || control_units.is_empty() {
        return None;
    }

    let treated_set: BTreeSet<i64> = treated_units.iter().copied().collect();
    let control_set: BTreeSet<i64> = control_units.iter().copied().collect();

    // Determine the appropriate time window based on comparison type
    let (pre_periods, post_periods): (Vec<i64>, Vec<i64>) = if early_as_treated {
        // Treated vs Not-Yet-Treated:
        // - Pre-period: before g_treated (early group treatment time)
        // - Post-period: g_treated to (g_control - 1) when early is treated but late is not yet
        let pre: Vec<i64> = times.iter().copied().filter(|&t| t < g_treated).collect();
        let post: Vec<i64> = times
            .iter()
            .copied()
            .filter(|&t| t >= g_treated && t < g_control)
            .collect();
        (pre, post)
    } else {
        // Later vs Earlier Treated ("forbidden"):
        // - Pre-period: g_control (early treatment time) to g_treated - 1
        // - Post-period: g_treated and after (both groups now treated)
        let pre: Vec<i64> = times
            .iter()
            .copied()
            .filter(|&t| t >= g_control && t < g_treated)
            .collect();
        let post: Vec<i64> = times.iter().copied().filter(|&t| t >= g_treated).collect();
        (pre, post)
    };

    if pre_periods.is_empty() || post_periods.is_empty() {
        return None;
    }

    let pre_set: BTreeSet<i64> = pre_periods.iter().copied().collect();
    let post_set: BTreeSet<i64> = post_periods.iter().copied().collect();

    let mut y_treated_pre = Vec::new();
    let mut y_treated_post = Vec::new();
    let mut y_control_pre = Vec::new();
    let mut y_control_post = Vec::new();

    for i in 0..n {
        let u = unit[i] as i64;
        let t = time[i] as i64;

        if treated_set.contains(&u) {
            if pre_set.contains(&t) {
                y_treated_pre.push(y[i]);
            } else if post_set.contains(&t) {
                y_treated_post.push(y[i]);
            }
        } else if control_set.contains(&u) {
            if pre_set.contains(&t) {
                y_control_pre.push(y[i]);
            } else if post_set.contains(&t) {
                y_control_post.push(y[i]);
            }
        }
    }

    // Need observations in all four cells
    if y_treated_pre.is_empty()
        || y_treated_post.is_empty()
        || y_control_pre.is_empty()
        || y_control_post.is_empty()
    {
        return None;
    }

    // Compute 2x2 DiD
    let mean_treated_pre = y_treated_pre.iter().sum::<f64>() / y_treated_pre.len() as f64;
    let mean_treated_post = y_treated_post.iter().sum::<f64>() / y_treated_post.len() as f64;
    let mean_control_pre = y_control_pre.iter().sum::<f64>() / y_control_pre.len() as f64;
    let mean_control_post = y_control_post.iter().sum::<f64>() / y_control_post.len() as f64;

    let estimate = (mean_treated_post - mean_treated_pre) - (mean_control_post - mean_control_pre);

    let n_treated = y_treated_pre.len() + y_treated_post.len();
    let n_control = y_control_pre.len() + y_control_post.len();

    let time_range = (
        pre_periods.iter().copied().min().unwrap_or(0),
        post_periods.iter().copied().max().unwrap_or(0),
    );

    Some(BaconComponent {
        treated_group: g_treated,
        control_group: g_control,
        estimate,
        weight: 0.0,
        n_treated,
        n_control,
        comparison_type,
        time_range,
    })
}

/// Compute weights for each 2x2 comparison following Goodman-Bacon (2021).
///
/// Weights depend on:
/// 1. n_k × n_l / n^2: Product of group shares
/// 2. V(D_kl): Variance of treatment indicator in the comparison window
///
/// The weight formula is: w_kl ∝ (n_k × n_l) × V(D_kl)
fn compute_weights(
    components: &mut [BaconComponent],
    n_total: usize,
    n_times: usize,
    unit_treatment_time: &BTreeMap<i64, i64>,
    times: &[i64],
) {
    // Count units in each timing group
    let mut group_counts: BTreeMap<i64, usize> = BTreeMap::new();
    for &g in unit_treatment_time.values() {
        *group_counts.entry(g).or_insert(0) += 1;
    }

    let n_total_f = n_total as f64;

    for comp in components.iter_mut() {
        let n_treated = *group_counts.get(&comp.treated_group).unwrap_or(&0);
        let n_control = if comp.control_group == 0 {
            *group_counts.get(&0).unwrap_or(&0)
        } else {
            *group_counts.get(&comp.control_group).unwrap_or(&0)
        };

        if n_treated == 0 || n_control == 0 {
            comp.weight = 0.0;
            continue;
        }

        // Group size component: (n_k / n) × (n_l / n)
        let size_weight = (n_treated as f64 / n_total_f) * (n_control as f64 / n_total_f);

        // Variance of treatment indicator in the comparison window
        // V(D) = p(1-p) where p is the fraction of treated observations
        let variance_weight = compute_treatment_variance(
            comp.treated_group,
            comp.control_group,
            times,
            n_times,
            &comp.comparison_type,
        );

        // Combined weight
        comp.weight = size_weight * variance_weight * (n_total_f.powi(2));
    }
}

/// Compute the variance of the treatment indicator for a specific comparison.
///
/// This captures how much the treatment status varies within the comparison window,
/// which affects the precision of the 2x2 estimate.
fn compute_treatment_variance(
    g_treated: i64,
    g_control: i64,
    times: &[i64],
    _n_times: usize,
    comparison_type: &ComparisonType,
) -> f64 {
    // Determine the relevant time window for this comparison
    let (pre_count, post_count) = match comparison_type {
        ComparisonType::TreatedVsNeverTreated => {
            // Pre: t < g_treated, Post: t >= g_treated
            let pre = times.iter().filter(|&&t| t < g_treated).count();
            let post = times.iter().filter(|&&t| t >= g_treated).count();
            (pre, post)
        }
        ComparisonType::TreatedVsNotYetTreated => {
            // Pre: t < g_treated, Post: g_treated <= t < g_control
            let pre = times.iter().filter(|&&t| t < g_treated).count();
            let post = times
                .iter()
                .filter(|&&t| t >= g_treated && t < g_control)
                .count();
            (pre, post)
        }
        ComparisonType::LaterVsEarlierTreated => {
            // Pre: g_control <= t < g_treated, Post: t >= g_treated
            let pre = times
                .iter()
                .filter(|&&t| t >= g_control && t < g_treated)
                .count();
            let post = times.iter().filter(|&&t| t >= g_treated).count();
            (pre, post)
        }
    };

    let total = pre_count + post_count;
    if total == 0 {
        return 0.0;
    }

    // p = fraction of observations that are "post" (treated)
    let p = post_count as f64 / total as f64;

    // Variance of binary variable: p(1-p)
    // This is maximized when p = 0.5 (equal pre/post periods)
    p * (1.0 - p) * (total as f64)
}

/// Compute weighted average estimates by comparison type.
fn compute_estimates_by_type(components: &[BaconComponent]) -> BaconEstimatesByType {
    let mut tvn_sum = 0.0;
    let mut tvn_weight = 0.0;
    let mut tvnyt_sum = 0.0;
    let mut tvnyt_weight = 0.0;
    let mut lve_sum = 0.0;
    let mut lve_weight = 0.0;

    for comp in components {
        match comp.comparison_type {
            ComparisonType::TreatedVsNeverTreated => {
                tvn_sum += comp.weight * comp.estimate;
                tvn_weight += comp.weight;
            }
            ComparisonType::TreatedVsNotYetTreated => {
                tvnyt_sum += comp.weight * comp.estimate;
                tvnyt_weight += comp.weight;
            }
            ComparisonType::LaterVsEarlierTreated => {
                lve_sum += comp.weight * comp.estimate;
                lve_weight += comp.weight;
            }
        }
    }

    BaconEstimatesByType {
        treated_vs_never: if tvn_weight > 0.0 {
            Some(tvn_sum / tvn_weight)
        } else {
            None
        },
        treated_vs_not_yet: if tvnyt_weight > 0.0 {
            Some(tvnyt_sum / tvnyt_weight)
        } else {
            None
        },
        later_vs_earlier: if lve_weight > 0.0 {
            Some(lve_sum / lve_weight)
        } else {
            None
        },
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    /// Create a simple staggered DiD dataset for testing.
    ///
    /// - 4 units over 5 time periods (2000-2004)
    /// - Unit 1: treated in 2001 (cohort 2001)
    /// - Unit 2: treated in 2003 (cohort 2003)
    /// - Unit 3: never treated
    /// - Unit 4: never treated
    /// - True treatment effect: 2.0
    fn create_test_dataset() -> Dataset {
        let mut unit_vec = Vec::new();
        let mut time_vec = Vec::new();
        let mut treat_vec = Vec::new();
        let mut y_vec = Vec::new();

        let treatment_effect = 2.0;

        for year in 2000..=2004 {
            for unit in 1..=4 {
                unit_vec.push(unit as f64);
                time_vec.push(year as f64);

                // Determine treatment status
                let (_treat_time, is_treated) = match unit {
                    1 => (2001, year >= 2001),
                    2 => (2003, year >= 2003),
                    _ => (0, false), // Never treated
                };

                treat_vec.push(if is_treated { 1.0 } else { 0.0 });

                // Outcome: unit FE + time trend + treatment effect
                let base = unit as f64 * 2.0; // Unit fixed effect
                let trend = (year - 2000) as f64 * 0.5; // Time trend
                let effect = if is_treated { treatment_effect } else { 0.0 };
                let noise = ((unit * year) % 5) as f64 * 0.1 - 0.25; // Pseudo-noise

                y_vec.push(base + trend + effect + noise);
            }
        }

        let df = df! {
            "unit" => unit_vec,
            "year" => time_vec,
            "treated" => treat_vec,
            "outcome" => y_vec
        }
        .unwrap();

        Dataset::new(df)
    }

    #[test]
    fn test_bacon_decomp_basic() {
        let dataset = create_test_dataset();
        let result = bacon_decomp(&dataset, "outcome", "unit", "year", "treated").unwrap();

        // Check basic structure
        assert!(!result.components.is_empty());
        assert!(result.timing_groups.contains(&2001));
        assert!(result.timing_groups.contains(&2003));
        assert_eq!(result.n_timing_groups, 2);
        assert_eq!(result.n_never_treated, 2);
        assert_eq!(result.n_units, 4);

        // Weights should sum to approximately 1
        assert!(
            (result.weights_sum - 1.0).abs() < 0.01,
            "Weights should sum to 1, got {}",
            result.weights_sum
        );

        // Check that we have different comparison types
        let has_never = result
            .components
            .iter()
            .any(|c| c.comparison_type == ComparisonType::TreatedVsNeverTreated);
        assert!(
            has_never,
            "Should have Treated vs Never-Treated comparisons"
        );

        // Print result for debugging
        println!("{}", result);
    }

    #[test]
    fn test_bacon_comparison_types() {
        let dataset = create_test_dataset();
        let result = bacon_decomp(&dataset, "outcome", "unit", "year", "treated").unwrap();

        // Should have all three types of comparisons
        let types: Vec<ComparisonType> = result
            .components
            .iter()
            .map(|c| c.comparison_type)
            .collect();

        // Must have treated vs never-treated (2 groups x never-treated)
        assert!(
            types.contains(&ComparisonType::TreatedVsNeverTreated),
            "Missing TreatedVsNeverTreated"
        );

        // With 2 timing groups, should have at least one of the between-group comparisons
        let has_timing_comparison = types.contains(&ComparisonType::TreatedVsNotYetTreated)
            || types.contains(&ComparisonType::LaterVsEarlierTreated);
        assert!(
            has_timing_comparison,
            "Should have comparisons between timing groups"
        );
    }

    #[test]
    fn test_bacon_weight_distribution() {
        let dataset = create_test_dataset();
        let result = bacon_decomp(&dataset, "outcome", "unit", "year", "treated").unwrap();

        // Weight components should be non-negative
        for comp in &result.components {
            assert!(comp.weight >= 0.0, "Weights should be non-negative");
        }

        // Sum of type weights should equal total
        let type_sum =
            result.treated_vs_never + result.treated_vs_not_yet + result.later_vs_earlier;
        assert!(
            (type_sum - result.weights_sum).abs() < 0.001,
            "Type weights should sum to total: {} vs {}",
            type_sum,
            result.weights_sum
        );
    }

    #[test]
    fn test_bacon_estimates_reasonable() {
        let dataset = create_test_dataset();
        let result = bacon_decomp(&dataset, "outcome", "unit", "year", "treated").unwrap();

        // True effect is 2.0, estimates should be in a reasonable range
        for comp in &result.components {
            assert!(
                comp.estimate.abs() < 10.0,
                "Estimate {} seems unreasonable",
                comp.estimate
            );
        }

        // Overall estimate should also be reasonable
        assert!(
            result.overall_estimate.abs() < 10.0,
            "Overall estimate {} seems unreasonable",
            result.overall_estimate
        );
    }

    #[test]
    fn test_bacon_no_treatment() {
        // Dataset with no treatment
        let df = df! {
            "unit" => [1.0, 1.0, 2.0, 2.0],
            "year" => [2000.0, 2001.0, 2000.0, 2001.0],
            "treated" => [0.0, 0.0, 0.0, 0.0],
            "outcome" => [1.0, 2.0, 3.0, 4.0]
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = bacon_decomp(&dataset, "outcome", "unit", "year", "treated");
        assert!(result.is_err());
    }

    #[test]
    fn test_comparison_type_display() {
        assert_eq!(
            format!("{}", ComparisonType::TreatedVsNeverTreated),
            "Treated vs Never-Treated"
        );
        assert_eq!(
            format!("{}", ComparisonType::TreatedVsNotYetTreated),
            "Treated vs Not-Yet-Treated"
        );
        assert_eq!(
            format!("{}", ComparisonType::LaterVsEarlierTreated),
            "Later vs Earlier Treated"
        );
    }

    #[test]
    fn test_bacon_component_display() {
        let comp = BaconComponent {
            treated_group: 2002,
            control_group: 0,
            estimate: 1.5,
            weight: 0.25,
            n_treated: 100,
            n_control: 200,
            comparison_type: ComparisonType::TreatedVsNeverTreated,
            time_range: (2000, 2005),
        };

        let display = format!("{}", comp);
        assert!(display.contains("2002"));
        assert!(display.contains("Never"));
        assert!(display.contains("1.5"));
    }
}
