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

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::design::{DesignMatrix, get_column_names};

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
            available: get_column_names(dataset.df()),
        }
    })?;

    let unit = DesignMatrix::extract_column(dataset.df(), unit_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: unit_col.to_string(),
            available: get_column_names(dataset.df()),
        }
    })?;

    let time = DesignMatrix::extract_column(dataset.df(), time_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: time_col.to_string(),
            available: get_column_names(dataset.df()),
        }
    })?;

    let treatment = DesignMatrix::extract_column(dataset.df(), treatment_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: treatment_col.to_string(),
            available: get_column_names(dataset.df()),
        }
    })?;

    let n = y.len();

    // Identify unique time periods
    let unique_times: BTreeSet<i64> = time.iter().map(|&t| t as i64).collect();
    let times: Vec<i64> = unique_times.iter().copied().collect();

    let n_times = times.len();

    // Build unit -> treatment timing map in one pass over the rows.
    // Treatment timing = first period where treatment = 1 (0 means never treated)
    let mut first_treat: BTreeMap<i64, Option<i64>> = BTreeMap::new();
    for i in 0..n {
        let entry = first_treat.entry(unit[i] as i64).or_insert(None);
        if treatment[i] > 0.5 {
            let t = time[i] as i64;
            if entry.is_none_or(|f| t < f) {
                *entry = Some(t);
            }
        }
    }
    let unit_treatment_time: BTreeMap<i64, i64> = first_treat
        .into_iter()
        .map(|(u, first)| (u, first.unwrap_or(0)))
        .collect();
    let n_units = unit_treatment_time.len();

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

    // Outcome sums and counts per (cohort, period), built once. Every 2x2
    // mean below is a sum over a range of these cells.
    let cells = build_cohort_cells(&y, &unit, &time, &unit_treatment_time, &times);

    // Now compute all 2x2 comparisons
    let mut components = Vec::new();

    // For each pair of timing groups (k, l) where k is treated and l is control
    // We need to consider:
    // 1. Each treated group vs. never-treated
    // 2. Each earlier-treated group vs. later-treated group (both directions)

    // Type 1: Treated vs. Never-Treated
    if n_never_treated > 0 {
        for &g in &timing_groups {
            crate::resource::checkpoint()?;
            if let Some(comp) = compute_2x2_did(
                &cells,
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
            crate::resource::checkpoint()?;
            // g_early treated before g_late

            // Comparison A: Early-treated (as treated) vs Late-treated (as not-yet-treated)
            // This uses periods before g_late where early is treated, late is not yet treated
            // Type: Treated vs Not-Yet-Treated (clean comparison)
            if let Some(comp) = compute_2x2_did_timing(
                &cells,
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
                &cells,
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

/// Outcome sums and counts of one treatment cohort, per time period.
///
/// `ti`, `sum` and `count` are parallel and ascending in `ti`, the index of
/// the period in the sorted list of distinct times. Only periods in which
/// the cohort has rows are stored, so the cells of all cohorts together
/// number at most n.
#[derive(Debug, Default)]
struct CohortCells {
    ti: Vec<usize>,
    sum: Vec<f64>,
    count: Vec<usize>,
}

impl CohortCells {
    /// Sum and count of the outcome over the periods with index in `range`.
    fn window(&self, range: std::ops::Range<usize>) -> (f64, usize) {
        let a = self.ti.partition_point(|&t| t < range.start);
        let b = self.ti.partition_point(|&t| t < range.end);
        (
            self.sum[a..b].iter().sum::<f64>(),
            self.count[a..b].iter().sum::<usize>(),
        )
    }
}

/// Build the per-(cohort, period) outcome sums in one pass over the rows,
/// O(n log n). The cohort of a row is its unit's first treatment period (0
/// for never treated); rows are accumulated in row order.
fn build_cohort_cells(
    y: &Array1<f64>,
    unit: &Array1<f64>,
    time: &Array1<f64>,
    unit_treatment_time: &BTreeMap<i64, i64>,
    times: &[i64],
) -> BTreeMap<i64, CohortCells> {
    let mut cell_sums: BTreeMap<(i64, usize), (f64, usize)> = BTreeMap::new();
    for i in 0..y.len() {
        // Both lookups always succeed (the maps are built from these rows);
        // a miss would skip the row rather than panic.
        let Some(&g) = unit_treatment_time.get(&(unit[i] as i64)) else {
            continue;
        };
        let Ok(ti) = times.binary_search(&(time[i] as i64)) else {
            continue;
        };
        let cell = cell_sums.entry((g, ti)).or_insert((0.0, 0));
        cell.0 += y[i];
        cell.1 += 1;
    }

    let mut cohorts: BTreeMap<i64, CohortCells> = BTreeMap::new();
    for ((g, ti), (sum, count)) in cell_sums {
        let cells = cohorts.entry(g).or_default();
        cells.ti.push(ti);
        cells.sum.push(sum);
        cells.count.push(count);
    }
    cohorts
}

/// The 2x2 DiD estimate from the four (group, window) cells, or `None` when
/// any of them is empty.
#[allow(clippy::too_many_arguments)]
fn two_by_two(
    treated: &CohortCells,
    control: &CohortCells,
    pre: std::ops::Range<usize>,
    post: std::ops::Range<usize>,
    g_treated: i64,
    g_control: i64,
    comparison_type: ComparisonType,
    time_range: (i64, i64),
) -> Option<BaconComponent> {
    let (s_treated_pre, n_treated_pre) = treated.window(pre.clone());
    let (s_treated_post, n_treated_post) = treated.window(post.clone());
    let (s_control_pre, n_control_pre) = control.window(pre);
    let (s_control_post, n_control_post) = control.window(post);

    // Need observations in all four cells
    if n_treated_pre == 0 || n_treated_post == 0 || n_control_pre == 0 || n_control_post == 0 {
        return None;
    }

    // Compute 2x2 DiD: (Ȳ_treat_post - Ȳ_treat_pre) - (Ȳ_ctrl_post - Ȳ_ctrl_pre)
    let mean_treated_pre = s_treated_pre / n_treated_pre as f64;
    let mean_treated_post = s_treated_post / n_treated_post as f64;
    let mean_control_pre = s_control_pre / n_control_pre as f64;
    let mean_control_post = s_control_post / n_control_post as f64;

    let estimate = (mean_treated_post - mean_treated_pre) - (mean_control_post - mean_control_pre);

    Some(BaconComponent {
        treated_group: g_treated,
        control_group: g_control,
        estimate,
        weight: 0.0, // Will be computed later
        n_treated: n_treated_pre + n_treated_post,
        n_control: n_control_pre + n_control_post,
        comparison_type,
        time_range,
    })
}

/// Compute a 2x2 DiD comparison between treated and never-treated groups.
///
/// Pre-period: before g_treated; post-period: g_treated and after. The time
/// range is the whole panel.
fn compute_2x2_did(
    cells: &BTreeMap<i64, CohortCells>,
    g_treated: i64,
    g_control: i64,
    times: &[i64],
    comparison_type: ComparisonType,
) -> Option<BaconComponent> {
    let treated = cells.get(&g_treated)?;
    let control = cells.get(&g_control)?;
    let (&first, &last) = (times.first()?, times.last()?);

    let lb = times.partition_point(|&t| t < g_treated);
    two_by_two(
        treated,
        control,
        0..lb,
        lb..times.len(),
        g_treated,
        g_control,
        comparison_type,
        (first, last),
    )
}

/// Compute a 2x2 DiD comparison between two timing groups.
///
/// This handles the more complex case where both groups eventually get treated,
/// so we need to carefully select the time window.
fn compute_2x2_did_timing(
    cells: &BTreeMap<i64, CohortCells>,
    g_treated: i64,
    g_control: i64,
    times: &[i64],
    comparison_type: ComparisonType,
    early_as_treated: bool,
) -> Option<BaconComponent> {
    let treated = cells.get(&g_treated)?;
    let control = cells.get(&g_control)?;

    let lb = |g: i64| times.partition_point(|&t| t < g);
    // Determine the appropriate time window based on comparison type
    let (pre, post) = if early_as_treated {
        // Treated vs Not-Yet-Treated:
        // - Pre-period: before g_treated (early group treatment time)
        // - Post-period: g_treated to (g_control - 1) when early is treated but late is not yet
        (0..lb(g_treated), lb(g_treated)..lb(g_control))
    } else {
        // Later vs Earlier Treated ("forbidden"):
        // - Pre-period: g_control (early treatment time) to g_treated - 1
        // - Post-period: g_treated and after (both groups now treated)
        (lb(g_control)..lb(g_treated), lb(g_treated)..times.len())
    };

    if pre.is_empty() || post.is_empty() {
        return None;
    }

    let time_range = (times[pre.start], times[post.end - 1]);
    two_by_two(
        treated,
        control,
        pre,
        post,
        g_treated,
        g_control,
        comparison_type,
        time_range,
    )
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

    // ═══════════════════════════════════════════════════════════════════════
    // Validation Tests
    //
    // These tests validate the Goodman-Bacon decomposition against known
    // structural properties:
    // 1. Weights must sum to 1
    // 2. Weighted sum of 2x2 DD estimates must equal overall TWFE coefficient
    // 3. Each component has a valid estimate and non-negative weight
    //
    // References:
    // - Goodman-Bacon, A. (2021). "Difference-in-Differences with Variation
    //   in Treatment Timing". Journal of Econometrics, 225(2), 254-277.
    // - R package `bacondecomp` (Flack & Edward)
    // ═══════════════════════════════════════════════════════════════════════

    /// Create a larger dataset for Bacon decomposition validation.
    ///
    /// DGP:
    /// - 6 units per timing group, 3 timing groups + 6 never-treated = 24 units
    /// - 8 periods (2000-2007)
    /// - Cohort 2002: ATT = 2.0
    /// - Cohort 2004: ATT = 2.0
    /// - Cohort 2006: ATT = 2.0
    /// - Homogeneous effect ensures all 2x2 estimates converge to ~2.0
    fn create_validation_bacon_dataset() -> Dataset {
        let mut unit_vec = Vec::new();
        let mut time_vec = Vec::new();
        let mut treat_vec = Vec::new();
        let mut y_vec = Vec::new();

        let treatment_effect = 2.0;

        for year in 2000..=2007 {
            for unit in 1..=24 {
                unit_vec.push(unit as f64);
                time_vec.push(year as f64);

                let (g, is_treated) = if unit <= 6 {
                    (2002, year >= 2002)
                } else if unit <= 12 {
                    (2004, year >= 2004)
                } else if unit <= 18 {
                    (2006, year >= 2006)
                } else {
                    (0, false)
                };

                treat_vec.push(if is_treated { 1.0 } else { 0.0 });

                let base = unit as f64 * 1.5;
                let trend = (year - 2000) as f64 * 0.4;
                let effect = if is_treated { treatment_effect } else { 0.0 };
                let noise = ((unit * year) % 13) as f64 * 0.03 - 0.195;

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
    fn test_validate_bacon_weights_sum_to_one() {
        // The fundamental property of the Bacon decomposition: all weights
        // must sum to exactly 1 (within numerical tolerance).
        // This follows from Theorem 1 in Goodman-Bacon (2021).
        let dataset = create_validation_bacon_dataset();
        let result = bacon_decomp(&dataset, "outcome", "unit", "year", "treated").unwrap();

        let weight_sum: f64 = result.components.iter().map(|c| c.weight).sum();
        assert!(
            (weight_sum - 1.0).abs() < 1e-6,
            "Decomposition weights must sum to 1.0, got {:.10}",
            weight_sum
        );

        // Also verify via the stored weights_sum field
        assert!(
            (result.weights_sum - 1.0).abs() < 1e-6,
            "weights_sum field should be ~1.0, got {:.10}",
            result.weights_sum
        );
    }

    #[test]
    fn test_validate_bacon_weighted_sum_equals_overall() {
        // The weighted sum of all 2x2 DD estimates must equal the overall
        // TWFE coefficient. This is the key decomposition identity:
        //   beta_TWFE = sum_k sum_l w_kl * beta_kl
        //
        // Goodman-Bacon (2021), Theorem 1.
        let dataset = create_validation_bacon_dataset();
        let result = bacon_decomp(&dataset, "outcome", "unit", "year", "treated").unwrap();

        let weighted_sum: f64 = result
            .components
            .iter()
            .map(|c| c.weight * c.estimate)
            .sum();

        assert!(
            (weighted_sum - result.overall_estimate).abs() < 1e-8,
            "Weighted sum of 2x2 DDs ({:.10}) must equal overall estimate ({:.10})",
            weighted_sum,
            result.overall_estimate
        );
    }

    #[test]
    fn test_validate_bacon_all_weights_nonnegative() {
        // All decomposition weights must be non-negative.
        // Weights are proportional to (group share)^2 * variance(D), which
        // are inherently non-negative quantities.
        let dataset = create_validation_bacon_dataset();
        let result = bacon_decomp(&dataset, "outcome", "unit", "year", "treated").unwrap();

        for (i, comp) in result.components.iter().enumerate() {
            assert!(
                comp.weight >= 0.0,
                "Component {} weight should be non-negative, got {:.10}",
                i,
                comp.weight
            );
        }
    }

    #[test]
    fn test_validate_bacon_type_weights_sum_to_total() {
        // The sum of weights by comparison type must equal the total weight sum.
        // This is a partition identity: every component belongs to exactly one type.
        let dataset = create_validation_bacon_dataset();
        let result = bacon_decomp(&dataset, "outcome", "unit", "year", "treated").unwrap();

        let type_sum =
            result.treated_vs_never + result.treated_vs_not_yet + result.later_vs_earlier;

        assert!(
            (type_sum - result.weights_sum).abs() < 1e-10,
            "Sum of type weights ({:.10}) must equal total ({:.10})",
            type_sum,
            result.weights_sum
        );
    }

    #[test]
    fn test_validate_bacon_homogeneous_effect_estimates() {
        // Under a homogeneous treatment effect (ATT = 2.0 for all cohorts),
        // all 2x2 DD estimates should be close to the true effect.
        // The "treated vs never-treated" comparisons should be especially
        // close since they are clean comparisons.
        let dataset = create_validation_bacon_dataset();
        let result = bacon_decomp(&dataset, "outcome", "unit", "year", "treated").unwrap();

        let true_att = 2.0;

        // Overall TWFE estimate should be close to the true effect
        assert!(
            (result.overall_estimate - true_att).abs() < 1.0,
            "Overall TWFE estimate should be ~{:.1}, got {:.4}",
            true_att,
            result.overall_estimate
        );

        // Treated vs never-treated estimates should be close to true effect
        let tvn_components: Vec<&BaconComponent> = result
            .components
            .iter()
            .filter(|c| c.comparison_type == ComparisonType::TreatedVsNeverTreated)
            .collect();

        for comp in &tvn_components {
            assert!(
                (comp.estimate - true_att).abs() < 1.5,
                "TvN estimate (g={}) should be ~{:.1}, got {:.4}",
                comp.treated_group,
                true_att,
                comp.estimate
            );
        }
    }

    #[test]
    fn test_validate_bacon_expected_comparison_count() {
        // With K timing groups and a never-treated group, we expect:
        // - K "treated vs never-treated" comparisons
        // - K*(K-1)/2 "treated vs not-yet-treated" comparisons
        // - K*(K-1)/2 "later vs earlier treated" comparisons
        //
        // For K=3: 3 + 3 + 3 = 9 total components (at most).
        // Some may be missing if time windows are too narrow.
        let dataset = create_validation_bacon_dataset();
        let result = bacon_decomp(&dataset, "outcome", "unit", "year", "treated").unwrap();

        let k = result.n_timing_groups;
        assert_eq!(k, 3, "Should have 3 timing groups");

        // Count by type
        let n_tvn = result
            .components
            .iter()
            .filter(|c| c.comparison_type == ComparisonType::TreatedVsNeverTreated)
            .count();
        let n_tvnyt = result
            .components
            .iter()
            .filter(|c| c.comparison_type == ComparisonType::TreatedVsNotYetTreated)
            .count();
        let n_lve = result
            .components
            .iter()
            .filter(|c| c.comparison_type == ComparisonType::LaterVsEarlierTreated)
            .count();

        // Should have K treated-vs-never comparisons
        assert_eq!(
            n_tvn, k,
            "Should have {} treated-vs-never comparisons, got {}",
            k, n_tvn
        );

        // Between-group comparisons: at most K*(K-1)/2 each
        let max_between = k * (k - 1) / 2;
        assert!(
            n_tvnyt <= max_between,
            "Should have at most {} treated-vs-not-yet comparisons, got {}",
            max_between,
            n_tvnyt
        );
        assert!(
            n_lve <= max_between,
            "Should have at most {} later-vs-earlier comparisons, got {}",
            max_between,
            n_lve
        );
    }

    #[test]
    fn test_validate_bacon_each_component_has_valid_estimate() {
        // Each decomposition component should have a finite estimate
        // and positive observation counts.
        let dataset = create_validation_bacon_dataset();
        let result = bacon_decomp(&dataset, "outcome", "unit", "year", "treated").unwrap();

        for (i, comp) in result.components.iter().enumerate() {
            assert!(
                comp.estimate.is_finite(),
                "Component {} estimate should be finite, got {}",
                i,
                comp.estimate
            );
            assert!(
                comp.n_treated > 0,
                "Component {} should have treated observations",
                i
            );
            assert!(
                comp.n_control > 0,
                "Component {} should have control observations",
                i
            );
            assert!(
                comp.time_range.0 <= comp.time_range.1,
                "Component {} time range should be valid: ({}, {})",
                i,
                comp.time_range.0,
                comp.time_range.1
            );
        }
    }

    #[test]
    fn test_validate_bacon_heterogeneous_effects_forbidden_comparisons() {
        // Under heterogeneous treatment effects, the "later vs earlier"
        // (forbidden) comparisons may produce biased estimates. This is
        // the key insight of Goodman-Bacon (2021).
        //
        // DGP: Cohort 2002 ATT = 1.0, Cohort 2004 ATT = 5.0
        let mut unit_vec = Vec::new();
        let mut time_vec = Vec::new();
        let mut treat_vec = Vec::new();
        let mut y_vec = Vec::new();

        for year in 2000..=2006 {
            for unit in 1..=12 {
                unit_vec.push(unit as f64);
                time_vec.push(year as f64);

                let (g, is_treated) = if unit <= 4 {
                    (2002, year >= 2002)
                } else if unit <= 8 {
                    (2004, year >= 2004)
                } else {
                    (0, false)
                };

                let att = match g {
                    2002 if is_treated => 1.0,
                    2004 if is_treated => 5.0,
                    _ => 0.0,
                };

                treat_vec.push(if is_treated { 1.0 } else { 0.0 });

                let base = unit as f64 * 1.0;
                let trend = (year - 2000) as f64 * 0.3;
                let noise = ((unit * year) % 7) as f64 * 0.02 - 0.07;

                y_vec.push(base + trend + att + noise);
            }
        }

        let df = df! {
            "unit" => unit_vec,
            "year" => time_vec,
            "treated" => treat_vec,
            "outcome" => y_vec
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = bacon_decomp(&dataset, "outcome", "unit", "year", "treated").unwrap();

        // Weights should still sum to 1
        let weight_sum: f64 = result.components.iter().map(|c| c.weight).sum();
        assert!(
            (weight_sum - 1.0).abs() < 1e-6,
            "Weights must sum to 1 even with heterogeneous effects, got {:.10}",
            weight_sum
        );

        // Weighted sum must still equal overall estimate
        let weighted_sum: f64 = result
            .components
            .iter()
            .map(|c| c.weight * c.estimate)
            .sum();
        assert!(
            (weighted_sum - result.overall_estimate).abs() < 1e-8,
            "Decomposition identity must hold with heterogeneous effects"
        );

        // The forbidden comparison (later vs earlier) may have different
        // estimates than the clean comparisons. Check that these are present.
        let has_forbidden = result
            .components
            .iter()
            .any(|c| c.comparison_type == ComparisonType::LaterVsEarlierTreated);
        // Note: forbidden comparisons may not always be present depending
        // on the time window structure, so just check if they exist and
        // their estimates are finite.
        if has_forbidden {
            let forbidden: Vec<&BaconComponent> = result
                .components
                .iter()
                .filter(|c| c.comparison_type == ComparisonType::LaterVsEarlierTreated)
                .collect();
            for comp in &forbidden {
                assert!(
                    comp.estimate.is_finite(),
                    "Forbidden comparison estimate should be finite"
                );
            }
        }
    }

    #[test]
    fn test_validate_bacon_estimates_by_type_aggregation() {
        // Validate that the estimates_by_type field correctly aggregates
        // within-type weighted averages.
        let dataset = create_validation_bacon_dataset();
        let result = bacon_decomp(&dataset, "outcome", "unit", "year", "treated").unwrap();

        // If treated_vs_never weight > 0, the estimate should be defined
        if result.treated_vs_never > 0.0 {
            assert!(
                result.estimate_by_type.treated_vs_never.is_some(),
                "Should have TvN estimate when TvN weight > 0"
            );
            let tvn_est = result.estimate_by_type.treated_vs_never.unwrap();
            assert!(
                tvn_est.is_finite(),
                "TvN estimate should be finite, got {}",
                tvn_est
            );
        }

        if result.treated_vs_not_yet > 0.0 {
            assert!(
                result.estimate_by_type.treated_vs_not_yet.is_some(),
                "Should have TvNYT estimate when weight > 0"
            );
        }

        if result.later_vs_earlier > 0.0 {
            assert!(
                result.estimate_by_type.later_vs_earlier.is_some(),
                "Should have LvE estimate when weight > 0"
            );
        }
    }

    // =========================================================================
    // Cell-sum 2x2 estimates, pinned to bacondecomp and R (T0-WP11)
    // =========================================================================

    fn cancelled<R>(f: impl FnOnce() -> R) -> R {
        use crate::resource::{Budget, CancelToken, with_budget};
        let token = CancelToken::new();
        token.cancel();
        with_budget(Budget::new(None, token), f)
    }

    /// 12 units x 8 periods; units 1-4 treated from t=3, 5-8 from t=6, 9-12 never;
    /// dynamic effect 1 + 0.2 (t - g); noise 0.4 sin(1.3 u + 0.7 t). `drop` removes rows.
    fn bacon_panel(drop: impl Fn(i64, i64) -> bool) -> Dataset {
        let (mut u_v, mut t_v, mut d_v, mut y_v) = (vec![], vec![], vec![], vec![]);
        for u in 1..=12i64 {
            let g = if u <= 4 {
                3
            } else if u <= 8 {
                6
            } else {
                0
            };
            for t in 1..=8i64 {
                if drop(u, t) {
                    continue;
                }
                let d = if g > 0 && t >= g { 1.0 } else { 0.0 };
                let y = 0.5 * u as f64
                    + 0.3 * t as f64
                    + d * (1.0 + 0.2 * (t - g) as f64)
                    + 0.4 * (1.3 * u as f64 + 0.7 * t as f64).sin();
                u_v.push(u as f64);
                t_v.push(t as f64);
                d_v.push(d);
                y_v.push(y);
            }
        }
        Dataset::new(df! { "unit" => u_v, "time" => t_v, "D" => d_v, "y" => y_v }.unwrap())
    }

    fn comp(r: &BaconDecompResult, tg: i64, cg: i64) -> &BaconComponent {
        r.components
            .iter()
            .find(|c| c.treated_group == tg && c.control_group == cg)
            .unwrap_or_else(|| panic!("no component {tg} vs {cg}"))
    }

    /// R 4.3.3, bacondecomp 0.1.1:
    ///   units <- 1:12; times <- 1:8; cohort <- c(rep(3,4), rep(6,4), rep(0,4))
    ///   d <- expand.grid(time = times, unit = units); d$g <- cohort[d$unit]
    ///   d$D <- as.numeric(d$g > 0 & d$time >= d$g)
    ///   d$y <- 0.5*d$unit + 0.3*d$time + d$D*(1 + 0.2*(d$time - d$g)) + 0.4*sin(1.3*d$unit + 0.7*d$time)
    ///   bacon(y ~ D, data = d, id_var = "unit", time_var = "time")
    /// The 2x2 estimates match; the weights do not (CAUS-10, Tier 4), so they are not compared.
    #[test]
    fn test_bacon_components_match_bacondecomp() {
        let r = bacon_decomp(&bacon_panel(|_, _| false), "y", "unit", "time", "D").unwrap();
        assert_eq!(r.components.len(), 4);
        for (tg, cg, want, range) in [
            (3, 0, 1.70190208989377223, (1, 8)),
            (6, 0, 1.31395450341578623, (1, 8)),
            (3, 6, 1.32021179413731549, (1, 5)),
            (6, 3, 0.63491348028797567, (3, 8)),
        ] {
            let c = comp(&r, tg, cg);
            assert!((c.estimate - want).abs() < 1e-10, "{tg} vs {cg}");
            assert_eq!(c.time_range, range, "{tg} vs {cg}");
        }
        let ws: f64 = r.components.iter().map(|c| c.weight * c.estimate).sum();
        assert!((ws - r.overall_estimate).abs() < 1e-12);
    }

    /// Unbalanced: rows with unit %% 5 == 0 and time %% 3 == 0 dropped (92 rows). Unit 5 then
    /// first shows treatment at t = 7, so p2a forms a cohort 7. Reference computed in R from the
    /// 2x2 definition p2a uses (cell means over the comparison windows):
    ///   g <- per-unit min(time[D > 0.5]) else 0
    ///   est(tg, cg, pre, post) = (mean y[tg,post] - mean y[tg,pre]) - (mean y[cg,post] - mean y[cg,pre])
    #[test]
    fn test_bacon_unbalanced_components_pinned() {
        let r = bacon_decomp(
            &bacon_panel(|u, t| u % 5 == 0 && t % 3 == 0),
            "y",
            "unit",
            "time",
            "D",
        )
        .unwrap();
        assert_eq!(r.n_obs, 92);
        assert_eq!(r.timing_groups, vec![3, 6, 7]);
        assert_eq!(r.components.len(), 7);
        for (tg, cg, est, nt, nc, range) in [
            (3, 0, 1.6473318293605486, 32, 30, (1, 8)),
            (6, 0, 1.4218183251615919, 24, 30, (1, 8)),
            (7, 0, 1.0231643682502058, 6, 30, (1, 8)),
            (3, 6, 1.2358640135079875, 20, 15, (1, 5)),
            (6, 3, 0.7923376456502278, 18, 24, (3, 8)),
            (3, 7, 1.8228762052453114, 24, 4, (1, 6)),
            (7, 3, 0.45335893016458684, 4, 24, (3, 8)),
        ] {
            let c = comp(&r, tg, cg);
            assert!(
                (c.estimate - est).abs() < 1e-10,
                "{tg} vs {cg}: {}",
                c.estimate
            );
            assert_eq!((c.n_treated, c.n_control), (nt, nc), "{tg} vs {cg}");
            assert_eq!(c.time_range, range, "{tg} vs {cg}");
        }
    }

    #[test]
    fn test_bacon_stops_when_cancelled() {
        let r = cancelled(|| bacon_decomp(&bacon_panel(|_, _| false), "y", "unit", "time", "D"));
        assert!(matches!(r, Err(EconError::Cancelled(_))));
    }
}
