//! Callaway-Sant'Anna Difference-in-Differences for Staggered Treatment.
//!
//! Pure Rust implementation of the Callaway and Sant'Anna (2021) estimator for
//! difference-in-differences with multiple time periods and staggered treatment adoption.
//!
//! # Overview
//!
//! This module extends traditional DiD to handle:
//! - Multiple time periods (T > 2)
//! - Staggered treatment timing (units treated at different points)
//! - Treatment effect heterogeneity across cohorts and time
//! - Conditional parallel trends (controlling for covariates)
//!
//! # Key Concepts
//!
//! ## Group-Time ATT
//!
//! The fundamental building block is ATT(g,t): the average treatment effect for
//! units first treated in period g, measured in period t.
//!
//! Under never-treated parallel trends:
//! ```text
//! ATT(g,t) = E[Y_t - Y_{g-1} | G=g] - E[Y_t - Y_{g-1} | C=1]
//! ```
//!
//! Where:
//! - G=g: units first treated in period g (cohort g)
//! - C=1: never-treated units (comparison group)
//!
//! ## Aggregation Schemes
//!
//! Group-time ATTs can be aggregated into:
//! - **Event Study**: Effects by time since treatment (dynamic effects)
//! - **Group Average**: Average effect for each cohort
//! - **Overall ATT**: Single summary measure across all cohorts
//!
//! # References
//!
//! - Callaway, B., & Sant'Anna, P.H.C. (2021). Difference-in-Differences with
//!   Multiple Time Periods. *Journal of Econometrics*, 225(2), 200-230.
//!   https://doi.org/10.1016/j.jeconom.2020.12.001
//!
//! - R package: `did` (https://bcallaway11.github.io/did/)
//!
//! - Stata package: `csdid` (https://github.com/friosavila/csdid)

use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, MissingSummary, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::safe_inverse;
use crate::resource;
use crate::traits::estimator::{SignificanceLevel, normal_cdf};

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Comparison group strategy for Callaway-Sant'Anna estimator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ComparisonGroup {
    /// Use never-treated units only (G_i = ∞)
    /// More restrictive but cleaner identification
    #[default]
    NeverTreated,
    /// Use not-yet-treated units (G_i > t)
    /// Larger comparison group but requires no anticipation
    NotYetTreated,
}

impl fmt::Display for ComparisonGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComparisonGroup::NeverTreated => write!(f, "Never-Treated"),
            ComparisonGroup::NotYetTreated => write!(f, "Not-Yet-Treated"),
        }
    }
}

/// Estimation method for group-time ATTs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AttEstimationMethod {
    /// Outcome regression (OLS within comparison periods)
    #[default]
    OutcomeRegression,
    /// Inverse probability weighting
    IPW,
    /// Doubly robust (AIPW) - recommended
    DoublyRobust,
}

impl fmt::Display for AttEstimationMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AttEstimationMethod::OutcomeRegression => write!(f, "Outcome Regression"),
            AttEstimationMethod::IPW => write!(f, "IPW"),
            AttEstimationMethod::DoublyRobust => write!(f, "Doubly Robust (AIPW)"),
        }
    }
}

/// Configuration for Callaway-Sant'Anna staggered DiD.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaggeredDidConfig {
    /// Comparison group strategy
    pub comparison_group: ComparisonGroup,
    /// Estimation method for ATT(g,t)
    pub estimation_method: AttEstimationMethod,
    /// Base period for pre-treatment (relative to g)
    /// Default: -1 (one period before treatment)
    pub base_period: i32,
    /// Whether to compute pre-treatment effects (for parallel trends testing)
    pub anticipation: usize,
    /// Bootstrap replications for standard errors
    pub bootstrap: usize,
    /// Confidence level for intervals
    pub confidence_level: f64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Minimum observations per cell (g,t)
    pub min_obs_per_cell: usize,
}

impl Default for StaggeredDidConfig {
    fn default() -> Self {
        Self {
            comparison_group: ComparisonGroup::NeverTreated,
            estimation_method: AttEstimationMethod::OutcomeRegression,
            base_period: -1,
            anticipation: 0,
            bootstrap: 999,
            confidence_level: 0.95,
            seed: None,
            min_obs_per_cell: 10,
        }
    }
}

/// Aggregation type for group-time ATTs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Aggregation {
    /// Event study: aggregate by relative time e = t - g
    EventStudy,
    /// Group average: aggregate by cohort g
    Group,
    /// Overall: single weighted average
    Overall,
    /// Calendar time: aggregate by time period t
    Calendar,
}

impl fmt::Display for Aggregation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Aggregation::EventStudy => write!(f, "Event Study"),
            Aggregation::Group => write!(f, "Group Average"),
            Aggregation::Overall => write!(f, "Overall"),
            Aggregation::Calendar => write!(f, "Calendar Time"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Result Types
// ═══════════════════════════════════════════════════════════════════════════════

/// A single group-time ATT estimate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupTimeATT {
    /// Treatment cohort (period when treatment started)
    pub group: i64,
    /// Time period
    pub time: i64,
    /// ATT estimate
    pub att: f64,
    /// Standard error
    pub std_error: f64,
    /// Confidence interval lower bound
    pub ci_lower: f64,
    /// Confidence interval upper bound
    pub ci_upper: f64,
    /// Number of treated observations in this cell
    pub n_treated: usize,
    /// Number of comparison observations
    pub n_comparison: usize,
    /// Whether this is a post-treatment effect (t >= g)
    pub post_treatment: bool,
    /// Relative time (t - g)
    pub relative_time: i64,
}

/// Aggregated treatment effect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedEffect {
    /// Aggregation key (relative time for event study, group for group avg, etc.)
    pub key: i64,
    /// Aggregated ATT
    pub att: f64,
    /// Standard error
    pub std_error: f64,
    /// Confidence interval lower bound
    pub ci_lower: f64,
    /// Confidence interval upper bound
    pub ci_upper: f64,
    /// p-value
    pub p_value: f64,
    /// Number of contributing cells
    pub n_cells: usize,
    /// Total observations
    pub n_obs: usize,
}

/// Pre-trend test result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreTrendTest {
    /// Chi-squared statistic
    pub chi2: f64,
    /// Degrees of freedom
    pub df: usize,
    /// p-value
    pub p_value: f64,
    /// Pre-treatment ATT estimates (should be ~0 under parallel trends)
    pub pre_atts: Vec<f64>,
}

/// Result from Callaway-Sant'Anna staggered DiD estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaggeredDidResult {
    /// All group-time ATT estimates
    pub group_time_atts: Vec<GroupTimeATT>,
    /// Event study aggregation (by relative time)
    pub event_study: Vec<AggregatedEffect>,
    /// Group average aggregation (by cohort)
    pub group_effects: Vec<AggregatedEffect>,
    /// Overall ATT
    pub overall_att: AggregatedEffect,
    /// Pre-trend test (joint test that pre-treatment ATTs are zero)
    pub pretrend_test: Option<PreTrendTest>,
    /// Configuration used
    pub config: StaggeredDidConfig,
    /// Treatment cohorts identified
    pub cohorts: Vec<i64>,
    /// Time periods
    pub periods: Vec<i64>,
    /// Total observations
    pub n_obs: usize,
    /// Number of treated units
    pub n_treated: usize,
    /// Number of never-treated units
    pub n_never_treated: usize,
    /// Warnings generated
    pub warnings: Vec<String>,
    /// Rows dropped because of missing values (empty when none were).
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
}

impl fmt::Display for StaggeredDidResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Callaway-Sant'Anna Staggered DiD Estimation")?;
        writeln!(f, "============================================")?;
        writeln!(f)?;
        writeln!(f, "Settings:")?;
        writeln!(f, "  Comparison Group:   {}", self.config.comparison_group)?;
        writeln!(f, "  Estimation Method:  {}", self.config.estimation_method)?;
        writeln!(f, "  Base Period:        {}", self.config.base_period)?;
        writeln!(f, "  Bootstrap Reps:     {}", self.config.bootstrap)?;
        writeln!(f)?;
        writeln!(f, "Sample:")?;
        writeln!(f, "  Total observations: {}", self.n_obs)?;
        writeln!(f, "  Treated units:      {}", self.n_treated)?;
        writeln!(f, "  Never-treated:      {}", self.n_never_treated)?;
        writeln!(f, "  Treatment cohorts:  {:?}", self.cohorts)?;
        writeln!(f)?;

        // Overall ATT
        writeln!(f, "OVERALL ATT:")?;
        writeln!(
            f,
            "  ATT = {:.4} (SE = {:.4}, 95% CI [{:.4}, {:.4}])",
            self.overall_att.att,
            self.overall_att.std_error,
            self.overall_att.ci_lower,
            self.overall_att.ci_upper
        )?;
        let sig = SignificanceLevel::from_p_value(self.overall_att.p_value);
        writeln!(
            f,
            "  p-value = {:.4}{}",
            self.overall_att.p_value,
            sig.stars()
        )?;
        writeln!(f)?;

        // Event Study
        writeln!(f, "EVENT STUDY (by relative time):")?;
        writeln!(
            f,
            "{:>8} {:>12} {:>12} {:>20}",
            "e", "ATT", "Std.Err", "95% CI"
        )?;
        writeln!(f, "{}", "-".repeat(60))?;
        for eff in &self.event_study {
            let sig = SignificanceLevel::from_p_value(eff.p_value);
            writeln!(
                f,
                "{:>8} {:>12.4} {:>12.4} [{:>8.4}, {:>8.4}]{}",
                eff.key,
                eff.att,
                eff.std_error,
                eff.ci_lower,
                eff.ci_upper,
                sig.stars()
            )?;
        }
        writeln!(f)?;

        // Pre-trend test
        if let Some(ref pretrend) = self.pretrend_test {
            writeln!(
                f,
                "PRE-TREND TEST (joint test that pre-treatment ATTs = 0):"
            )?;
            writeln!(
                f,
                "  Chi-squared({}) = {:.2}, p-value = {:.4}",
                pretrend.df, pretrend.chi2, pretrend.p_value
            )?;
            if pretrend.p_value < 0.05 {
                writeln!(f, "  WARNING: Pre-trends may be violated")?;
            } else {
                writeln!(f, "  Parallel trends assumption not rejected")?;
            }
        }

        // The missing-value note is printed once, as the last line.
        let note = self.missing.note();
        let warnings: Vec<&String> = self
            .warnings
            .iter()
            .filter(|w| Some(w.as_str()) != note.as_deref())
            .collect();
        if !warnings.is_empty() {
            writeln!(f)?;
            writeln!(f, "Warnings:")?;
            for w in warnings {
                writeln!(f, "  - {}", w)?;
            }
        }
        if let Some(note) = note {
            writeln!(f)?;
            writeln!(f, "Missing values: {note}")?;
        }

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main Estimation Function
// ═══════════════════════════════════════════════════════════════════════════════

/// Run Callaway-Sant'Anna staggered difference-in-differences estimation.
///
/// # Arguments
///
/// * `dataset` - Panel dataset with repeated observations
/// * `outcome` - Name of the outcome variable (numeric)
/// * `treatment_time` - Name of the column indicating when each unit was first treated
///   (whole numbers or dates; 0 or negative for never-treated units). It must be
///   constant within each unit, as `did::att_gt` requires.
/// * `time_col` - Name of the time period column (whole numbers or dates)
/// * `unit_col` - Name of the unit/individual identifier column (any type: integers,
///   fractional numbers, strings, categoricals, dates)
/// * `covariates` - Optional covariate columns for conditional parallel trends (numeric)
/// * `config` - Configuration options
///
/// # Missing values
///
/// The columns are read through [`crate::data::columns::ColumnRequest`]: a row with a
/// missing value (null or NaN) in the outcome, the first treatment period, the time, the
/// unit or a covariate is dropped from every column, and `StaggeredDidResult::missing`,
/// the warnings and the last Display line report what was dropped. Infinite values,
/// fractional periods and text outcomes are refused with an error naming the column.
///
/// # Model
///
/// For each group g (units first treated at time g) and time t:
///
/// ```text
/// ATT(g,t) = E[Y_t - Y_{g-1} | G=g] - E[Y_t - Y_{g-1} | Comparison]
/// ```
///
/// The comparison group depends on the configuration:
/// - NeverTreated: Units with G = ∞ (never treated)
/// - NotYetTreated: Units with G > t (not yet treated by time t)
///
/// # Bootstrap
///
/// With `bootstrap > 0`, standard errors come from a nonparametric bootstrap
/// that resamples units. Each cell's outcome changes are tabulated once (at
/// most `n_cohorts x n` entries), so the bootstrap costs
/// O(bootstrap x (units + entries)) and its replicate-by-cell draws are one
/// fallibly allocated matrix: an impossible replication count is a
/// `ResourceLimit` error. Every (g, t) cell and every replicate honours the
/// cooperative cancellation budget (`p2a_core::resource`). A replicate that
/// draws a unit without a base-period row has no estimate for that cell
/// (unbalanced panels, CAUS-85), and replicates ignore covariates (CAUS-36).
///
/// # Example
///
/// ```ignore
/// use p2a_core::econometrics::{run_staggered_did, StaggeredDidConfig};
///
/// let config = StaggeredDidConfig::default();
/// let result = run_staggered_did(
///     &dataset,
///     "outcome",
///     "first_treat",  // Period when treatment started (0 = never treated)
///     "year",
///     "state",
///     None,  // No covariates
///     config,
/// )?;
///
/// println!("Overall ATT: {:.4}", result.overall_att.att);
/// for e in &result.event_study {
///     println!("e={}: ATT={:.4}", e.key, e.att);
/// }
/// ```
pub fn run_staggered_did(
    dataset: &Dataset,
    outcome: &str,
    treatment_time: &str,
    time_col: &str,
    unit_col: &str,
    covariates: Option<&[&str]>,
    config: StaggeredDidConfig,
) -> EconResult<StaggeredDidResult> {
    let mut warnings = Vec::new();

    // One request over every column the estimator uses: rows with a missing
    // value in any of them are dropped jointly (reported in `missing`).
    let covariates = covariates.filter(|c| !c.is_empty());
    let mut request = ColumnRequest::new(dataset.df())
        .numeric(outcome)
        .time_index(treatment_time)
        .time_index(time_col)
        .factor(unit_col);
    if let Some(c) = covariates {
        request = request.numerics(c);
    }
    let cols = request.na(NaPolicy::Omit).extract()?;
    let n = cols.n();
    let unit = cols.factor(unit_col);
    let first_treat = cols.time_index(treatment_time);
    let period = cols.time_index(time_col);

    // The arrays the cell and bootstrap helpers take. Time indices are exact as
    // f64 below 2^53; unit codes are 0..n_units in factor-level order, which is
    // the numeric order for numeric ids.
    let y = Array1::from(cols.numeric(outcome).to_vec());
    let g_col: Array1<f64> = first_treat.iter().map(|&g| g as f64).collect();
    let t_col: Array1<f64> = period.iter().map(|&t| t as f64).collect();
    let unit_ids: Array1<f64> = unit.codes.iter().map(|&u| f64::from(u)).collect();
    // The intercept column stays, as in the design matrix used before.
    let x_cov: Option<Array2<f64>> = covariates.map(|c| cols.matrix(c, true).data);

    // Each unit's first treatment period, in one pass: did::att_gt requires it
    // to be constant within a unit.
    let n_units = unit.n_levels();
    let mut unit_g: Vec<Option<i64>> =
        resource::try_vec(n_units, None, "staggered DiD unit cohorts")?;
    for i in 0..n {
        let u = unit.codes[i] as usize;
        match unit_g[u] {
            None => unit_g[u] = Some(first_treat[i]),
            Some(g) if g != first_treat[i] => {
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "Column '{treatment_time}' must be constant within each unit \
                         (did::att_gt requires it); unit '{}' has {g} and {}.",
                        unit.label(i),
                        first_treat[i]
                    ),
                });
            }
            Some(_) => {}
        }
    }

    // Cohorts, periods and unit counts (never treated: first treatment <= 0)
    let cohorts: Vec<i64> = unit_g
        .iter()
        .flatten()
        .filter(|&&g| g > 0)
        .copied()
        .collect::<BTreeSet<i64>>()
        .into_iter()
        .collect();
    let periods: Vec<i64> = period
        .iter()
        .copied()
        .collect::<BTreeSet<i64>>()
        .into_iter()
        .collect();
    let n_treated_units = unit_g.iter().flatten().filter(|&&g| g > 0).count();
    let never_treated_count = unit_g.iter().flatten().filter(|&&g| g <= 0).count();

    if cohorts.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "No treated units found. Treatment time column must have positive values for treated units.".to_string(),
        });
    }

    if never_treated_count == 0 && config.comparison_group == ComparisonGroup::NeverTreated {
        return Err(EconError::InvalidSpecification {
            message: "No never-treated units found but NeverTreated comparison group specified. Use NotYetTreated or add never-treated units.".to_string(),
        });
    }

    // Build panel structure: map (unit, time) -> observation index
    let panel_map = build_panel_map(&t_col, &unit_ids);

    // Compute group-time ATTs
    let mut group_time_atts = Vec::new();

    for &g in &cohorts {
        // Base period is g - 1 (or as configured)
        let base_t = g + config.base_period as i64;

        for &t in &periods {
            resource::checkpoint()?;

            // Skip if base period not available
            if !periods.contains(&base_t) {
                continue;
            }

            // Compute ATT(g, t)
            let att_result = compute_group_time_att(
                &y, &g_col, &t_col, &unit_ids, &x_cov, &panel_map, g, t, base_t, &config,
            );

            match att_result {
                Ok(att) => {
                    if att.n_treated >= config.min_obs_per_cell
                        && att.n_comparison >= config.min_obs_per_cell
                    {
                        group_time_atts.push(att);
                    } else {
                        warnings.push(format!(
                            "Skipped ATT({},{}) due to small cell size: n_treated={}, n_comparison={}",
                            g, t, att.n_treated, att.n_comparison
                        ));
                    }
                }
                Err(e) => {
                    warnings.push(format!("Could not compute ATT({},{}): {:?}", g, t, e));
                }
            }
        }
    }

    if group_time_atts.is_empty() {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "No valid group-time ATT estimates could be computed".to_string(),
        });
    }

    // Bootstrap for standard errors
    let rng_seed = config.seed.unwrap_or_else(|| {
        use rand::RngCore;
        rand::thread_rng().next_u64()
    });
    let mut rng = StdRng::seed_from_u64(rng_seed);

    let z = quantile_normal(1.0 - (1.0 - config.confidence_level) / 2.0);

    // Unit-resampling bootstrap: each cell's outcome changes are tabulated once,
    // and every replicate weights them by how often it drew each unit.
    if config.bootstrap > 0 {
        let cells: Vec<(i64, i64, i64)> = group_time_atts
            .iter()
            .map(|gta| (gta.group, gta.time, gta.group + config.base_period as i64))
            .collect();
        let table = build_boot_table(
            &y,
            &g_col,
            &t_col,
            &unit_ids,
            &panel_map,
            &cells,
            config.comparison_group,
        )?;
        let draws = bootstrap_draws(&table, config.bootstrap, &mut rng)?;

        // Update standard errors from the finite draws of each cell, in replicate order
        for (idx, gta) in group_time_atts.iter_mut().enumerate() {
            let boot_atts: Vec<f64> = draws
                .column(idx)
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .collect();
            if boot_atts.len() >= 50 {
                let mean: f64 = boot_atts.iter().sum::<f64>() / boot_atts.len() as f64;
                let var: f64 = boot_atts.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                    / (boot_atts.len() - 1) as f64;
                gta.std_error = var.sqrt();
                gta.ci_lower = gta.att - z * gta.std_error;
                gta.ci_upper = gta.att + z * gta.std_error;
            }
        }
    }

    // Aggregate: Event Study
    let event_study = aggregate_event_study(&group_time_atts, z);

    // Aggregate: Group averages
    let group_effects = aggregate_by_group(&group_time_atts, z);

    // Aggregate: Overall ATT
    let overall_att = aggregate_overall(&group_time_atts, z);

    // Pre-trend test
    let pretrend_test = compute_pretrend_test(&group_time_atts);

    let missing = cols.missing().clone();
    if let Some(note) = missing.note() {
        warnings.push(note);
    }
    Ok(StaggeredDidResult {
        group_time_atts,
        event_study,
        group_effects,
        overall_att,
        pretrend_test,
        config,
        cohorts,
        periods,
        n_obs: n,
        n_treated: n_treated_units,
        n_never_treated: never_treated_count,
        warnings,
        missing,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// ATT Computation
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute a single group-time ATT.
#[allow(clippy::too_many_arguments)]
fn compute_group_time_att(
    y: &Array1<f64>,
    g_col: &Array1<f64>,
    t_col: &Array1<f64>,
    unit_ids: &Array1<f64>,
    x_cov: &Option<Array2<f64>>,
    panel_map: &BTreeMap<(i64, i64), usize>,
    g: i64,
    t: i64,
    base_t: i64,
    config: &StaggeredDidConfig,
) -> EconResult<GroupTimeATT> {
    let n = y.len();

    // Find treated units (G_i = g) with observations at both t and base_t
    let mut treated_indices_t: Vec<usize> = Vec::new();
    let mut treated_indices_base: Vec<usize> = Vec::new();

    for i in 0..n {
        let gi = g_col[i] as i64;
        let ti = t_col[i] as i64;
        let ui = unit_ids[i] as i64;

        if gi == g && ti == t {
            // Check if this unit has observation at base period
            if let Some(&base_idx) = panel_map.get(&(ui, base_t)) {
                treated_indices_t.push(i);
                treated_indices_base.push(base_idx);
            }
        }
    }

    // Find comparison units based on strategy
    let mut comp_indices_t: Vec<usize> = Vec::new();
    let mut comp_indices_base: Vec<usize> = Vec::new();

    for i in 0..n {
        let gi = g_col[i] as i64;
        let ti = t_col[i] as i64;
        let ui = unit_ids[i] as i64;

        let is_comparison = match config.comparison_group {
            ComparisonGroup::NeverTreated => gi <= 0,
            ComparisonGroup::NotYetTreated => gi > t || gi <= 0,
        };

        if is_comparison && ti == t {
            if let Some(&base_idx) = panel_map.get(&(ui, base_t)) {
                comp_indices_t.push(i);
                comp_indices_base.push(base_idx);
            }
        }
    }

    let n_treated = treated_indices_t.len();
    let n_comparison = comp_indices_t.len();

    if n_treated == 0 || n_comparison == 0 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: n_treated.min(n_comparison),
            context: format!(
                "ATT({},{}) has {} treated and {} comparison obs",
                g, t, n_treated, n_comparison
            ),
        });
    }

    // Compute outcome changes: ΔY = Y_t - Y_{base}
    let dy_treated: Vec<f64> = treated_indices_t
        .iter()
        .zip(treated_indices_base.iter())
        .map(|(&i_t, &i_base)| y[i_t] - y[i_base])
        .collect();

    let dy_comp: Vec<f64> = comp_indices_t
        .iter()
        .zip(comp_indices_base.iter())
        .map(|(&i_t, &i_base)| y[i_t] - y[i_base])
        .collect();

    // Estimate ATT based on method
    let att = match config.estimation_method {
        AttEstimationMethod::OutcomeRegression => {
            // Simple DiD: mean change in treated - mean change in comparison
            let mean_dy_treated: f64 = dy_treated.iter().sum::<f64>() / n_treated as f64;
            let mean_dy_comp: f64 = dy_comp.iter().sum::<f64>() / n_comparison as f64;
            mean_dy_treated - mean_dy_comp
        }
        AttEstimationMethod::IPW | AttEstimationMethod::DoublyRobust => {
            // IPW or DR estimation with covariates
            if let Some(x) = x_cov {
                estimate_att_with_covariates(
                    &dy_treated,
                    &dy_comp,
                    x,
                    &treated_indices_t,
                    &comp_indices_t,
                    config.estimation_method,
                )?
            } else {
                // Fall back to simple DiD if no covariates
                let mean_dy_treated: f64 = dy_treated.iter().sum::<f64>() / n_treated as f64;
                let mean_dy_comp: f64 = dy_comp.iter().sum::<f64>() / n_comparison as f64;
                mean_dy_treated - mean_dy_comp
            }
        }
    };

    // Simple standard error estimate (will be updated by bootstrap)
    let var_treated: f64 = if n_treated > 1 {
        let mean = dy_treated.iter().sum::<f64>() / n_treated as f64;
        dy_treated.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n_treated - 1) as f64
    } else {
        0.0
    };

    let var_comp: f64 = if n_comparison > 1 {
        let mean = dy_comp.iter().sum::<f64>() / n_comparison as f64;
        dy_comp.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n_comparison - 1) as f64
    } else {
        0.0
    };

    let se = (var_treated / n_treated as f64 + var_comp / n_comparison as f64).sqrt();

    let relative_time = t - g;
    let post_treatment = t >= g;

    Ok(GroupTimeATT {
        group: g,
        time: t,
        att,
        std_error: se,
        ci_lower: att - 1.96 * se,
        ci_upper: att + 1.96 * se,
        n_treated,
        n_comparison,
        post_treatment,
        relative_time,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Bootstrap
// ═══════════════════════════════════════════════════════════════════════════════

/// Map each `(unit, time)` pair to its row index. Rows are inserted in order, so
/// when a pair occurs more than once the last row wins.
fn build_panel_map(t_col: &Array1<f64>, unit_ids: &Array1<f64>) -> BTreeMap<(i64, i64), usize> {
    let mut panel_map: BTreeMap<(i64, i64), usize> = BTreeMap::new();
    for i in 0..t_col.len() {
        let u = unit_ids[i] as i64;
        let t = t_col[i] as i64;
        panel_map.insert((u, t), i);
    }
    panel_map
}

/// One row of a cell's bootstrap table: the outcome change `Y_t - Y_base` of a
/// treated or comparison row, and the index of its unit in the sorted unit list.
struct BootEntry {
    unit: usize,
    dy: f64,
    treated: bool,
}

/// The outcome changes every bootstrap replicate needs, computed once.
///
/// Cell `c` owns `entries[starts[c]..starts[c + 1]]` in ascending row order and
/// `blockers[blocker_starts[c]..blocker_starts[c + 1]]`: the units with a row at
/// time `t` but no row at the base period. A replicate that draws a blocker has
/// no estimate for that cell, as before (unbalanced panels are CAUS-85).
///
/// The table holds at most `n_cohorts x n` entries: a row appears once per cell
/// at its time period, and there is one cell per cohort and period.
struct BootTable {
    n_units: usize,
    entries: Vec<BootEntry>,
    starts: Vec<usize>,
    blockers: Vec<usize>,
    blocker_starts: Vec<usize>,
}

impl BootTable {
    fn n_cells(&self) -> usize {
        self.starts.len().saturating_sub(1)
    }
}

/// Tabulate, for each `(g, t, base_t)` cell, the outcome changes of its treated
/// and comparison rows, with the same row semantics as the point estimate:
/// treated rows have `G = g`, comparison rows follow `comparison_group`, and
/// rows of other cohorts are left out.
///
/// Covariates are not used: the replicates are unadjusted differences in mean
/// changes (CAUS-36), as before.
#[allow(clippy::too_many_arguments)]
fn build_boot_table(
    y: &Array1<f64>,
    g_col: &Array1<f64>,
    t_col: &Array1<f64>,
    unit_ids: &Array1<f64>,
    panel_map: &BTreeMap<(i64, i64), usize>,
    cells: &[(i64, i64, i64)],
    comparison_group: ComparisonGroup,
) -> EconResult<BootTable> {
    let n = y.len();

    let mut unit_list: Vec<i64> = (0..n).map(|i| unit_ids[i] as i64).collect();
    unit_list.sort_unstable();
    unit_list.dedup();
    let unit_of_row: Vec<usize> = (0..n)
        .map(|i| match unit_list.binary_search(&(unit_ids[i] as i64)) {
            Ok(pos) | Err(pos) => pos,
        })
        .collect();

    let mut rows_by_time: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for i in 0..n {
        rows_by_time.entry(t_col[i] as i64).or_default().push(i);
    }

    let mut capacity: usize = 0;
    for &(_, t, _) in cells {
        let rows = rows_by_time.get(&t).map_or(0, Vec::len);
        capacity = capacity.checked_add(rows).ok_or_else(|| {
            EconError::ResourceLimit(
                "staggered DiD bootstrap table is too large: the requested size overflows"
                    .to_string(),
            )
        })?;
    }
    let mut entries =
        resource::try_with_capacity::<BootEntry>(capacity, "staggered DiD bootstrap table")?;
    let mut starts = Vec::with_capacity(cells.len() + 1);
    let mut blockers = Vec::new();
    let mut blocker_starts = Vec::with_capacity(cells.len() + 1);
    starts.push(0);
    blocker_starts.push(0);

    for &(g, t, base_t) in cells {
        resource::checkpoint()?;
        if let Some(rows) = rows_by_time.get(&t) {
            for &i in rows {
                let ui = unit_ids[i] as i64;
                let Some(&base_idx) = panel_map.get(&(ui, base_t)) else {
                    blockers.push(unit_of_row[i]);
                    continue;
                };
                let gi = g_col[i] as i64;
                let dy = y[i] - y[base_idx];
                let treated = gi == g;
                let is_comparison = match comparison_group {
                    ComparisonGroup::NeverTreated => gi <= 0,
                    ComparisonGroup::NotYetTreated => gi > t || gi <= 0,
                };
                if treated || is_comparison {
                    entries.push(BootEntry {
                        unit: unit_of_row[i],
                        dy,
                        treated,
                    });
                }
            }
        }
        starts.push(entries.len());
        blocker_starts.push(blockers.len());
    }

    Ok(BootTable {
        n_units: unit_list.len(),
        entries,
        starts,
        blockers,
        blocker_starts,
    })
}

/// Draw `bootstrap` unit-resampling replicates of every cell's ATT.
///
/// Each replicate draws `n_units` units with replacement from the one `rng`
/// (the same stream as before) and turns the draws into multiplicity weights,
/// so it costs O(units + table entries). Rows of the result are replicates and
/// columns are cells, which keeps the joint draws available for aggregate
/// standard errors (CAUS-2). A cell without an estimate in a replicate (a drawn
/// unit lacks the base period, or one side has no weight) is `NaN`.
fn bootstrap_draws(
    table: &BootTable,
    bootstrap: usize,
    rng: &mut StdRng,
) -> EconResult<Array2<f64>> {
    let n_cells = table.n_cells();
    let n_units = table.n_units;
    let mut draws = resource::try_zeros2(bootstrap, n_cells, "bootstrap replicate draws")?;
    let mut counts = vec![0usize; n_units];

    for b in 0..bootstrap {
        resource::checkpoint()?;
        if n_units == 0 {
            draws.row_mut(b).fill(f64::NAN);
            continue;
        }

        counts.fill(0);
        for _ in 0..n_units {
            counts[rng.gen_range(0..n_units)] += 1;
        }

        for c in 0..n_cells {
            let blocked = table.blockers[table.blocker_starts[c]..table.blocker_starts[c + 1]]
                .iter()
                .any(|&u| counts[u] > 0);
            let draw = if blocked {
                f64::NAN
            } else {
                let mut sum_dy_treated = 0.0;
                let mut n_treated_weighted = 0.0;
                let mut sum_dy_comp = 0.0;
                let mut n_comp_weighted = 0.0;
                for entry in &table.entries[table.starts[c]..table.starts[c + 1]] {
                    let weight = counts[entry.unit] as f64;
                    // Skipping zero weights keeps the sums bit-identical to a
                    // pass over only the resampled units.
                    if weight == 0.0 {
                        continue;
                    }
                    if entry.treated {
                        sum_dy_treated += weight * entry.dy;
                        n_treated_weighted += weight;
                    } else {
                        sum_dy_comp += weight * entry.dy;
                        n_comp_weighted += weight;
                    }
                }
                if n_treated_weighted > 0.0 && n_comp_weighted > 0.0 {
                    sum_dy_treated / n_treated_weighted - sum_dy_comp / n_comp_weighted
                } else {
                    f64::NAN
                }
            };
            draws[[b, c]] = if draw.is_finite() { draw } else { f64::NAN };
        }
    }

    Ok(draws)
}

/// Estimate ATT with covariates using IPW or doubly robust (AIPW).
///
/// # References
///
/// - Sant'Anna, P.H.C., & Zhao, J. (2020). "Doubly Robust Difference-in-Differences Estimators".
///   *Journal of Econometrics*, 219(1), 101-122.
///   https://doi.org/10.1016/j.jeconom.2020.06.003
///
/// - For the doubly robust (AIPW) method, implements the improved ATT estimator:
///   ATT_DR = E[D/E[D]] * (ΔY - m₀(X)) - E[(1-D)·ψ(X)/E[D]] * (ΔY - m₀(X))
///   where m₀(X) is the outcome regression for controls and ψ(X) = p(X)/(1-p(X))
fn estimate_att_with_covariates(
    dy_treated: &[f64],
    dy_comp: &[f64],
    x: &Array2<f64>,
    treated_indices: &[usize],
    comp_indices: &[usize],
    method: AttEstimationMethod,
) -> EconResult<f64> {
    let n_treated = treated_indices.len();
    let n_comp = comp_indices.len();
    let n_total = n_treated + n_comp;
    let k = x.ncols();

    // Build combined arrays
    let mut dy_all = Vec::with_capacity(n_total);
    let mut d_all = Vec::with_capacity(n_total);
    let mut x_all = Array2::zeros((n_total, k));

    for (idx, &i) in treated_indices.iter().enumerate() {
        dy_all.push(dy_treated[idx]);
        d_all.push(1.0);
        x_all.row_mut(idx).assign(&x.row(i));
    }

    for (idx, &i) in comp_indices.iter().enumerate() {
        dy_all.push(dy_comp[idx]);
        d_all.push(0.0);
        x_all.row_mut(n_treated + idx).assign(&x.row(i));
    }

    let dy = Array1::from(dy_all);
    let d = Array1::from(d_all);

    match method {
        AttEstimationMethod::IPW => {
            // Estimate propensity scores
            let ps = estimate_propensity_logit(&x_all, &d)?;

            // IPW for ATT (Hajek estimator)
            let mut sum_treated = 0.0;
            let mut n_t = 0.0;
            let mut sum_weighted_comp = 0.0;
            let mut sum_weights = 0.0;

            for i in 0..n_total {
                let ps_i = ps[i].max(0.001).min(0.999);
                if d[i] > 0.5 {
                    sum_treated += dy[i];
                    n_t += 1.0;
                } else {
                    // Weight = p(X)/(1-p(X))
                    let w = ps_i / (1.0 - ps_i);
                    sum_weighted_comp += w * dy[i];
                    sum_weights += w;
                }
            }

            if n_t > 0.0 && sum_weights > 0.0 {
                Ok(sum_treated / n_t - sum_weighted_comp / sum_weights)
            } else {
                // Fall back to simple difference
                let mean_t: f64 = dy_treated.iter().sum::<f64>() / n_treated as f64;
                let mean_c: f64 = dy_comp.iter().sum::<f64>() / n_comp as f64;
                Ok(mean_t - mean_c)
            }
        }
        AttEstimationMethod::DoublyRobust => {
            // Doubly robust (AIPW) estimator from Sant'Anna & Zhao (2020)
            // Combines IPW with outcome regression for robustness to misspecification

            // Step 1: Estimate propensity scores P(D=1|X)
            let ps = estimate_propensity_logit(&x_all, &d)?;

            // Step 2: Estimate outcome regression for comparison group
            // Regress ΔY on X using only comparison units
            let m0 = estimate_outcome_regression(&x_all, &dy, &d)?;

            // Step 3: Compute AIPW ATT estimator
            // ATT_DR = (1/n₁) Σ_i [D_i·(ΔY_i - m₀(X_i))] -
            //          (1/n₁) Σ_i [(1-D_i)·ψ(X_i)·(ΔY_i - m₀(X_i))]
            // where ψ(X) = p(X)/(1-p(X))

            let n1 = d.iter().filter(|&&di| di > 0.5).count() as f64;

            if n1 == 0.0 {
                // Fall back to simple difference
                let mean_t: f64 = dy_treated.iter().sum::<f64>() / n_treated as f64;
                let mean_c: f64 = dy_comp.iter().sum::<f64>() / n_comp as f64;
                return Ok(mean_t - mean_c);
            }

            let mut att_dr = 0.0;

            for i in 0..n_total {
                let ps_i = ps[i].max(0.001).min(0.999);
                let residual = dy[i] - m0[i];

                if d[i] > 0.5 {
                    // Treated unit contribution
                    att_dr += residual;
                } else {
                    // Comparison unit contribution (IPW weighted)
                    let w = ps_i / (1.0 - ps_i);
                    att_dr -= w * residual;
                }
            }

            Ok(att_dr / n1)
        }
        AttEstimationMethod::OutcomeRegression => {
            // This case is handled separately, but include for completeness
            let mean_t: f64 = dy_treated.iter().sum::<f64>() / n_treated as f64;
            let mean_c: f64 = dy_comp.iter().sum::<f64>() / n_comp as f64;
            Ok(mean_t - mean_c)
        }
    }
}

/// Estimate outcome regression E[ΔY|X, D=0] for comparison group.
///
/// Uses OLS regression of ΔY on X using only comparison units,
/// then predicts for all units.
fn estimate_outcome_regression(
    x: &Array2<f64>,
    dy: &Array1<f64>,
    d: &Array1<f64>,
) -> EconResult<Array1<f64>> {
    let n = dy.len();
    let k = x.ncols();

    // Collect comparison units
    let comp_mask: Vec<bool> = d.iter().map(|&di| di < 0.5).collect();
    let n_comp: usize = comp_mask.iter().filter(|&&m| m).count();

    if n_comp <= k {
        // Not enough data for regression, return zeros
        return Ok(Array1::zeros(n));
    }

    // Build comparison-only design matrix with intercept
    let mut x_comp = Array2::zeros((n_comp, k + 1));
    let mut dy_comp = Array1::zeros(n_comp);

    let mut idx = 0;
    for i in 0..n {
        if comp_mask[i] {
            x_comp[[idx, 0]] = 1.0; // Intercept
            for j in 0..k {
                x_comp[[idx, j + 1]] = x[[i, j]];
            }
            dy_comp[idx] = dy[i];
            idx += 1;
        }
    }

    // OLS: β = (X'X)^(-1) X'y
    use crate::linalg::matrix_ops::{xtx, xty};

    let xtx_mat = xtx(&x_comp.view());
    let xty_vec = xty(&x_comp.view(), &dy_comp);

    let beta = match safe_inverse(&xtx_mat.view()) {
        Ok((inv, _)) => inv.dot(&xty_vec),
        Err(_) => {
            // Regression failed, return outcome means
            let mean_comp = dy_comp.mean().unwrap_or(0.0);
            return Ok(Array1::from_elem(n, mean_comp));
        }
    };

    // Predict for all units
    let mut m0 = Array1::zeros(n);
    for i in 0..n {
        let mut pred = beta[0]; // Intercept
        for j in 0..k {
            pred += beta[j + 1] * x[[i, j]];
        }
        m0[i] = pred;
    }

    Ok(m0)
}

/// Estimate propensity scores using logistic regression.
fn estimate_propensity_logit(x: &Array2<f64>, d: &Array1<f64>) -> EconResult<Array1<f64>> {
    let n = d.len();
    let k = x.ncols();

    // Newton-Raphson
    let mut beta = Array1::zeros(k);
    let max_iter = 25;
    let tol = 1e-6;

    for _ in 0..max_iter {
        let z: Array1<f64> = x.dot(&beta);
        let p: Array1<f64> = z.mapv(|zi| 1.0 / (1.0 + (-zi).exp()));
        let p_clip: Array1<f64> = p.mapv(|pi| pi.max(1e-10).min(1.0 - 1e-10));

        // Gradient
        let resid = d - &p_clip;
        let mut grad = Array1::zeros(k);
        for i in 0..n {
            for j in 0..k {
                grad[j] += resid[i] * x[[i, j]];
            }
        }

        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < tol {
            break;
        }

        // Hessian
        let w: Array1<f64> = p_clip.mapv(|pi| pi * (1.0 - pi));
        let mut hess = Array2::zeros((k, k));
        for i in 0..n {
            for j in 0..k {
                for l in 0..k {
                    hess[[j, l]] -= w[i] * x[[i, j]] * x[[i, l]];
                }
            }
        }

        // Update
        let neg_hess = &hess * -1.0;
        if let Ok((hess_inv, _)) = safe_inverse(&neg_hess.view()) {
            let delta = hess_inv.dot(&grad);
            beta = &beta + &delta;
        } else {
            break;
        }
    }

    // Final predictions
    let z_final: Array1<f64> = x.dot(&beta);
    Ok(z_final.mapv(|zi| 1.0 / (1.0 + (-zi).exp())))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Aggregation Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Aggregate group-time ATTs to event study (by relative time).
fn aggregate_event_study(atts: &[GroupTimeATT], z: f64) -> Vec<AggregatedEffect> {
    // Group by relative time
    let mut by_e: BTreeMap<i64, Vec<&GroupTimeATT>> = BTreeMap::new();

    for att in atts {
        by_e.entry(att.relative_time).or_default().push(att);
    }

    let mut effects = Vec::new();

    for (&e, group) in &by_e {
        let n_cells = group.len();
        if n_cells == 0 {
            continue;
        }

        // Weight by number of observations
        let total_obs: usize = group.iter().map(|a| a.n_treated).sum();

        let mut weighted_att = 0.0;
        for att in group {
            let w = att.n_treated as f64 / total_obs as f64;
            weighted_att += w * att.att;
        }

        // Pooled SE (simplified)
        let mut var_sum = 0.0;
        for att in group {
            let w = att.n_treated as f64 / total_obs as f64;
            var_sum += w * w * att.std_error * att.std_error;
        }
        let se = var_sum.sqrt();

        let p_value = if se > 0.0 {
            2.0 * (1.0 - normal_cdf((weighted_att / se).abs()))
        } else {
            1.0
        };

        effects.push(AggregatedEffect {
            key: e,
            att: weighted_att,
            std_error: se,
            ci_lower: weighted_att - z * se,
            ci_upper: weighted_att + z * se,
            p_value,
            n_cells,
            n_obs: total_obs,
        });
    }

    effects.sort_by_key(|e| e.key);
    effects
}

/// Aggregate by treatment cohort.
fn aggregate_by_group(atts: &[GroupTimeATT], z: f64) -> Vec<AggregatedEffect> {
    // Group by cohort, only post-treatment
    let mut by_g: BTreeMap<i64, Vec<&GroupTimeATT>> = BTreeMap::new();

    for att in atts {
        if att.post_treatment {
            by_g.entry(att.group).or_default().push(att);
        }
    }

    let mut effects = Vec::new();

    for (&g, group) in &by_g {
        let n_cells = group.len();
        if n_cells == 0 {
            continue;
        }

        // Simple average
        let avg_att: f64 = group.iter().map(|a| a.att).sum::<f64>() / n_cells as f64;

        let mut var_sum = 0.0;
        for att in group {
            var_sum += att.std_error * att.std_error;
        }
        let se = (var_sum / n_cells as f64).sqrt() / (n_cells as f64).sqrt();

        let total_obs: usize = group.iter().map(|a| a.n_treated).sum();

        let p_value = if se > 0.0 {
            2.0 * (1.0 - normal_cdf((avg_att / se).abs()))
        } else {
            1.0
        };

        effects.push(AggregatedEffect {
            key: g,
            att: avg_att,
            std_error: se,
            ci_lower: avg_att - z * se,
            ci_upper: avg_att + z * se,
            p_value,
            n_cells,
            n_obs: total_obs,
        });
    }

    effects.sort_by_key(|e| e.key);
    effects
}

/// Aggregate to overall ATT.
fn aggregate_overall(atts: &[GroupTimeATT], z: f64) -> AggregatedEffect {
    // Filter to post-treatment only
    let post_atts: Vec<&GroupTimeATT> = atts.iter().filter(|a| a.post_treatment).collect();

    if post_atts.is_empty() {
        return AggregatedEffect {
            key: 0,
            att: 0.0,
            std_error: 0.0,
            ci_lower: 0.0,
            ci_upper: 0.0,
            p_value: 1.0,
            n_cells: 0,
            n_obs: 0,
        };
    }

    // Weight by number of treated observations
    let total_obs: usize = post_atts.iter().map(|a| a.n_treated).sum();

    let mut weighted_att = 0.0;
    for att in &post_atts {
        let w = att.n_treated as f64 / total_obs as f64;
        weighted_att += w * att.att;
    }

    // Pooled SE
    let mut var_sum = 0.0;
    for att in &post_atts {
        let w = att.n_treated as f64 / total_obs as f64;
        var_sum += w * w * att.std_error * att.std_error;
    }
    let se = var_sum.sqrt();

    let p_value = if se > 0.0 {
        2.0 * (1.0 - normal_cdf((weighted_att / se).abs()))
    } else {
        1.0
    };

    AggregatedEffect {
        key: 0,
        att: weighted_att,
        std_error: se,
        ci_lower: weighted_att - z * se,
        ci_upper: weighted_att + z * se,
        p_value,
        n_cells: post_atts.len(),
        n_obs: total_obs,
    }
}

/// Compute pre-trend test (joint test that all pre-treatment ATTs = 0).
fn compute_pretrend_test(atts: &[GroupTimeATT]) -> Option<PreTrendTest> {
    // Collect pre-treatment ATTs
    let pre_atts: Vec<&GroupTimeATT> = atts.iter().filter(|a| !a.post_treatment).collect();

    if pre_atts.is_empty() {
        return None;
    }

    let _n_pre = pre_atts.len();
    let pre_effects: Vec<f64> = pre_atts.iter().map(|a| a.att).collect();

    // Wald test: sum of (ATT/SE)^2 ~ chi-squared(n_pre)
    let mut chi2 = 0.0;
    let mut valid_count = 0;

    for att in &pre_atts {
        if att.std_error > 0.0 && att.std_error.is_finite() {
            let t = att.att / att.std_error;
            chi2 += t * t;
            valid_count += 1;
        }
    }

    if valid_count == 0 {
        return None;
    }

    // Chi-squared p-value
    let df = valid_count;
    let p_value = chi_squared_p_value(chi2, df);

    Some(PreTrendTest {
        chi2,
        df,
        p_value,
        pre_atts: pre_effects,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Standard normal quantile.
fn quantile_normal(p: f64) -> f64 {
    // Approximation using Abramowitz & Stegun
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p < 0.5 { -z } else { z }
}

/// Chi-squared p-value (upper tail) using statrs if available.
fn chi_squared_p_value(x: f64, df: usize) -> f64 {
    if df == 0 || x < 0.0 {
        return 1.0;
    }
    if x == 0.0 {
        return 1.0;
    }

    // Use the regularized incomplete gamma function: P(chi2 > x) = 1 - P(a, x/2)
    // where a = df/2 and P(a, x) is the lower regularized incomplete gamma function
    let a = df as f64 / 2.0;
    let x_half = x / 2.0;

    // Upper tail: Q(a, x) = 1 - P(a, x)
    regularized_gamma_q(a, x_half)
}

/// Upper regularized incomplete gamma function Q(a, x) = 1 - P(a, x).
/// Uses series expansion for small x, continued fraction for large x.
fn regularized_gamma_q(a: f64, x: f64) -> f64 {
    if x < 0.0 {
        return 1.0;
    }
    if x == 0.0 {
        return 1.0;
    }
    if a <= 0.0 {
        return 0.0;
    }

    if x < a + 1.0 {
        // Use series expansion for P(a, x), then Q = 1 - P
        1.0 - regularized_gamma_p_series(a, x)
    } else {
        // Use continued fraction for Q(a, x) directly
        regularized_gamma_q_cf(a, x)
    }
}

/// Lower incomplete gamma P(a, x) via series expansion.
fn regularized_gamma_p_series(a: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 1e-14;

    let ln_gamma_a = ln_gamma(a);

    let mut sum = 1.0 / a;
    let mut term = 1.0 / a;

    for n in 1..max_iter {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < eps * sum.abs() {
            break;
        }
    }

    sum * (-x + a * x.ln() - ln_gamma_a).exp()
}

/// Upper incomplete gamma Q(a, x) via continued fraction (Lentz's algorithm).
fn regularized_gamma_q_cf(a: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 1e-14;
    let tiny = 1e-30;

    let ln_gamma_a = ln_gamma(a);

    // Lentz's algorithm for continued fraction
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b;
    let mut h = d;

    for n in 1..max_iter {
        let an = -(n as f64) * (n as f64 - a);
        b += 2.0;

        d = an * d + b;
        if d.abs() < tiny {
            d = tiny;
        }

        c = b + an / c;
        if c.abs() < tiny {
            c = tiny;
        }

        d = 1.0 / d;
        let delta = d * c;
        h *= delta;

        if (delta - 1.0).abs() < eps {
            break;
        }
    }

    h * (-x + a * x.ln() - ln_gamma_a).exp()
}

/// Natural log of gamma function using Lanczos approximation.
fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }

    // Lanczos coefficients for g=7, n=9
    let c = [
        0.999_999_999_999_809_9,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];

    let g = 7.0;

    if x < 0.5 {
        // Reflection formula: Gamma(1-x) * Gamma(x) = pi / sin(pi * x)
        let pi = std::f64::consts::PI;
        pi.ln() - (pi * x).sin().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut sum = c[0];
        for (i, &ci) in c.iter().enumerate().skip(1) {
            sum += ci / (x + i as f64);
        }

        let t = x + g + 0.5;
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::design::DesignMatrix;
    use crate::resource::{Budget, CancelToken, with_budget};
    use polars::prelude::*;

    fn cancelled() -> Budget {
        let token = CancelToken::new();
        token.cancel();
        Budget::new(None, token)
    }

    fn expect_cancelled<T>(res: EconResult<T>) {
        match res {
            Err(EconError::Cancelled(_)) => {}
            Err(e) => panic!("expected Cancelled, got: {e}"),
            Ok(_) => panic!("expected Cancelled, got Ok"),
        }
    }

    fn expect_resource_limit<T>(res: EconResult<T>) {
        match res {
            Err(EconError::ResourceLimit(_)) => {}
            Err(e) => panic!("expected ResourceLimit, got: {e}"),
            Ok(_) => panic!("expected ResourceLimit, got Ok"),
        }
    }

    /// Create a simple staggered treatment panel dataset.
    ///
    /// 3 cohorts treated in years 2002, 2004, 2006
    /// 1 never-treated cohort
    /// Years 2000-2008
    fn create_staggered_dataset() -> Dataset {
        let mut year_vec = Vec::new();
        let mut unit_vec = Vec::new();
        let mut g_vec = Vec::new();
        let mut y_vec = Vec::new();

        // Unit 1-5: treated in 2002 (cohort 2002)
        // Unit 6-10: treated in 2004 (cohort 2004)
        // Unit 11-15: treated in 2006 (cohort 2006)
        // Unit 16-20: never treated (cohort 0)

        let treatment_effect = 2.0; // True ATT

        for year in 2000..=2008 {
            for unit in 1..=20 {
                year_vec.push(year as f64);
                unit_vec.push(unit as f64);

                // Determine treatment timing
                let g = if unit <= 5 {
                    2002
                } else if unit <= 10 {
                    2004
                } else if unit <= 15 {
                    2006
                } else {
                    0 // never treated
                };
                g_vec.push(g as f64);

                // Outcome: base + trend + treatment effect if post-treatment
                let base = unit as f64 * 0.5; // unit fixed effect
                let trend = (year - 2000) as f64 * 0.3; // common trend
                let treated = if g > 0 && year >= g {
                    treatment_effect
                } else {
                    0.0
                };
                let noise = ((unit * year) % 10) as f64 * 0.1 - 0.5; // pseudo-random noise

                y_vec.push(base + trend + treated + noise);
            }
        }

        let df = df! {
            "year" => year_vec,
            "unit" => unit_vec,
            "first_treat" => g_vec,
            "outcome" => y_vec
        }
        .unwrap();

        Dataset::new(df)
    }

    /// `create_staggered_dataset` made unbalanced: `(unit 3, 2001)` and `(unit 17, 2003)`
    /// are missing, so some cells lack a base period, and `(unit 18, 2005)` appears twice
    /// (the second row, `+ 0.37`, is the one `panel_map` keeps).
    fn create_unbalanced_staggered_dataset() -> Dataset {
        let mut year_vec = Vec::new();
        let mut unit_vec = Vec::new();
        let mut g_vec = Vec::new();
        let mut y_vec = Vec::new();

        let outcome = |unit: i32, year: i32, g: i32| {
            let base = unit as f64 * 0.5;
            let trend = (year - 2000) as f64 * 0.3;
            let treated = if g > 0 && year >= g { 2.0 } else { 0.0 };
            let noise = ((unit * year) % 10) as f64 * 0.1 - 0.5;
            base + trend + treated + noise
        };
        let cohort = |unit: i32| {
            if unit <= 5 {
                2002
            } else if unit <= 10 {
                2004
            } else if unit <= 15 {
                2006
            } else {
                0
            }
        };

        for year in 2000..=2008 {
            for unit in 1..=20 {
                if (unit, year) == (3, 2001) || (unit, year) == (17, 2003) {
                    continue;
                }
                let g = cohort(unit);
                year_vec.push(year as f64);
                unit_vec.push(unit as f64);
                g_vec.push(g as f64);
                y_vec.push(outcome(unit, year, g));
            }
        }
        // duplicate (unit, time) row
        year_vec.push(2005.0);
        unit_vec.push(18.0);
        g_vec.push(0.0);
        y_vec.push(outcome(18, 2005, 0) + 0.37);

        let df = df! {
            "year" => year_vec,
            "unit" => unit_vec,
            "first_treat" => g_vec,
            "outcome" => y_vec
        }
        .unwrap();

        Dataset::new(df)
    }

    fn pinned_bootstrap_ses(dataset: &Dataset) -> (Vec<f64>, f64) {
        let config = StaggeredDidConfig {
            comparison_group: ComparisonGroup::NeverTreated,
            estimation_method: AttEstimationMethod::OutcomeRegression,
            bootstrap: 200,
            seed: Some(42),
            min_obs_per_cell: 3,
            ..Default::default()
        };
        let result = run_staggered_did(
            dataset,
            "outcome",
            "first_treat",
            "year",
            "unit",
            None,
            config,
        )
        .unwrap();
        (
            result.group_time_atts.iter().map(|g| g.std_error).collect(),
            result.overall_att.std_error,
        )
    }

    /// Seeded bootstrap SEs captured from the per-replicate implementation.
    #[test]
    fn test_staggered_did_bootstrap_ses_pinned() {
        let balanced: [f64; 27] = [
            0.18490145985527776,
            0.0,
            0.1808923855953215,
            0.2120487328975394,
            0.23072744644304863,
            0.20823152025499436,
            0.2234537656899955,
            0.20283205173975313,
            0.2643189612585981,
            0.23320011561681045,
            0.19070322569314244,
            0.21563141541771244,
            0.0,
            0.2663685066745486,
            0.2820728319220123,
            0.16984345547733584,
            0.3512417480039077,
            0.2329391410405387,
            0.17593078822888253,
            0.18756652059065274,
            0.23298033218340553,
            0.26624152070207663,
            0.24366619071737705,
            0.0,
            0.2664967811861835,
            0.24702693059618694,
            0.2644702388339487,
        ];
        let unbalanced: [f64; 27] = [
            0.21337412146962192,
            0.0,
            0.232739771272861,
            0.21241129431089897,
            0.2817675615414031,
            0.21708208206051982,
            0.23785277691140902,
            0.19642515916927691,
            0.3138385482197044,
            0.26805398079298853,
            0.1934452831682178,
            0.24658069499501026,
            0.0,
            0.1959086120946041,
            0.26577482181776185,
            0.20143567711747046,
            0.3307623266903197,
            0.19900134217799495,
            0.1678336974288942,
            0.16490510588398063,
            0.21398659180199545,
            0.24759731634793528,
            0.25467304901955323,
            0.047623165826116136,
            0.23329045958420777,
            0.2280247643824537,
            0.2594670200444654,
        ];
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b.abs().max(1.0);
        for (ds, want, want_overall) in [
            (create_staggered_dataset(), &balanced, 0.06304150304892771),
            (
                create_unbalanced_staggered_dataset(),
                &unbalanced,
                0.06328624510379224,
            ),
        ] {
            let (ses, overall) = pinned_bootstrap_ses(&ds);
            assert_eq!(ses.len(), want.len());
            for (i, (&a, &b)) in ses.iter().zip(want.iter()).enumerate() {
                assert!(close(a, b), "cell {i}: {a} vs pinned {b}");
            }
            assert!(
                close(overall, want_overall),
                "overall {overall} vs pinned {want_overall}"
            );
        }
    }

    #[test]
    fn test_staggered_did_basic() {
        let dataset = create_staggered_dataset();
        let config = StaggeredDidConfig {
            comparison_group: ComparisonGroup::NeverTreated,
            estimation_method: AttEstimationMethod::OutcomeRegression,
            bootstrap: 100,
            seed: Some(42),
            min_obs_per_cell: 3,
            ..Default::default()
        };

        let result = run_staggered_did(
            &dataset,
            "outcome",
            "first_treat",
            "year",
            "unit",
            None,
            config,
        )
        .unwrap();

        // Check structure
        assert!(!result.group_time_atts.is_empty());
        assert!(!result.event_study.is_empty());
        assert!(!result.group_effects.is_empty());
        assert_eq!(result.cohorts.len(), 3); // 2002, 2004, 2006

        // Overall ATT should be close to 2.0 (true effect)
        assert!(
            (result.overall_att.att - 2.0).abs() < 1.0,
            "Overall ATT should be close to 2.0, got {}",
            result.overall_att.att
        );
    }

    #[test]
    fn test_staggered_did_not_yet_treated() {
        let dataset = create_staggered_dataset();
        let config = StaggeredDidConfig {
            comparison_group: ComparisonGroup::NotYetTreated,
            bootstrap: 50,
            seed: Some(42),
            min_obs_per_cell: 3,
            ..Default::default()
        };

        let result = run_staggered_did(
            &dataset,
            "outcome",
            "first_treat",
            "year",
            "unit",
            None,
            config,
        )
        .unwrap();

        // Should work with not-yet-treated comparison
        assert!(!result.group_time_atts.is_empty());
    }

    #[test]
    fn test_event_study_aggregation() {
        let dataset = create_staggered_dataset();
        let config = StaggeredDidConfig {
            bootstrap: 50,
            seed: Some(42),
            min_obs_per_cell: 3,
            ..Default::default()
        };

        let result = run_staggered_did(
            &dataset,
            "outcome",
            "first_treat",
            "year",
            "unit",
            None,
            config,
        )
        .unwrap();

        // Event study should have both pre and post effects
        let pre_count = result.event_study.iter().filter(|e| e.key < 0).count();
        let post_count = result.event_study.iter().filter(|e| e.key >= 0).count();

        assert!(pre_count > 0, "Should have pre-treatment effects");
        assert!(post_count > 0, "Should have post-treatment effects");

        // Pre-treatment effects should be close to 0 (parallel trends)
        for e in result.event_study.iter().filter(|e| e.key < 0) {
            assert!(
                e.att.abs() < 1.5,
                "Pre-treatment effect at e={} should be small, got {}",
                e.key,
                e.att
            );
        }
    }

    #[test]
    fn test_pretrend_test() {
        let dataset = create_staggered_dataset();
        let config = StaggeredDidConfig {
            bootstrap: 50,
            seed: Some(42),
            min_obs_per_cell: 3,
            ..Default::default()
        };

        let result = run_staggered_did(
            &dataset,
            "outcome",
            "first_treat",
            "year",
            "unit",
            None,
            config,
        )
        .unwrap();

        // Should have pre-trend test
        assert!(result.pretrend_test.is_some());

        let pretrend = result.pretrend_test.unwrap();
        // Under correct parallel trends, p-value should be > 0.05
        assert!(
            pretrend.p_value > 0.01,
            "Pre-trend test p-value should not reject parallel trends, got {}",
            pretrend.p_value
        );
    }

    #[test]
    fn test_display() {
        let dataset = create_staggered_dataset();
        let config = StaggeredDidConfig {
            bootstrap: 20,
            seed: Some(42),
            min_obs_per_cell: 3,
            ..Default::default()
        };

        let result = run_staggered_did(
            &dataset,
            "outcome",
            "first_treat",
            "year",
            "unit",
            None,
            config,
        )
        .unwrap();

        let output = format!("{}", result);
        assert!(output.contains("Callaway-Sant'Anna"));
        assert!(output.contains("OVERALL ATT"));
        assert!(output.contains("EVENT STUDY"));
    }

    #[test]
    fn test_quantile_normal() {
        // Test standard normal quantiles
        assert!((quantile_normal(0.5) - 0.0).abs() < 0.01);
        assert!((quantile_normal(0.975) - 1.96).abs() < 0.05);
        assert!((quantile_normal(0.025) - (-1.96)).abs() < 0.05);
    }

    #[test]
    fn test_chi_squared_p_value() {
        // Chi-squared(1) = 3.84 should give p ≈ 0.05
        let p = chi_squared_p_value(3.84, 1);
        assert!(
            (p - 0.05).abs() < 0.02,
            "Chi-squared(1) = 3.84 should give p ≈ 0.05, got {}",
            p
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Validation Tests
    //
    // These tests validate the Callaway-Sant'Anna estimator against known
    // DGP properties. The data-generating process uses deterministic
    // pseudo-noise so results are reproducible without external RNG.
    //
    // References:
    // - Callaway, B. & Sant'Anna, P.H.C. (2021). "Difference-in-Differences
    //   with Multiple Time Periods". Journal of Econometrics, 225(2), 200-230.
    // - R package `did` (https://bcallaway11.github.io/did/)
    // ═══════════════════════════════════════════════════════════════════════

    /// Create a larger staggered dataset with heterogeneous treatment effects
    /// and known group-time ATTs.
    ///
    /// DGP:
    /// - 50 units over 9 periods (2000-2008)
    /// - Cohort 2002 (units 1-10): ATT = 2.0 for all post-treatment periods
    /// - Cohort 2004 (units 11-20): ATT = 4.0 for all post-treatment periods
    /// - Cohort 2006 (units 21-30): ATT = 6.0 for all post-treatment periods
    /// - Never-treated (units 31-50): no treatment effect
    /// - Common time trend: 0.3 per period
    /// - Unit FE: unit_id * 0.5
    /// - Deterministic pseudo-noise: ((unit * year) % 17) * 0.05 - 0.425
    fn create_validation_dataset_heterogeneous() -> Dataset {
        let mut year_vec = Vec::new();
        let mut unit_vec = Vec::new();
        let mut g_vec = Vec::new();
        let mut y_vec = Vec::new();

        for year in 2000..=2008 {
            for unit in 1..=50 {
                year_vec.push(year as f64);
                unit_vec.push(unit as f64);

                let g = if unit <= 10 {
                    2002
                } else if unit <= 20 {
                    2004
                } else if unit <= 30 {
                    2006
                } else {
                    0
                };
                g_vec.push(g as f64);

                // True cohort-specific ATT
                let att = match g {
                    2002 => 2.0,
                    2004 => 4.0,
                    2006 => 6.0,
                    _ => 0.0,
                };

                let base = unit as f64 * 0.5;
                let trend = (year - 2000) as f64 * 0.3;
                let treated = if g > 0 && year >= g { att } else { 0.0 };
                let noise = ((unit * year) % 17) as f64 * 0.05 - 0.425;

                y_vec.push(base + trend + treated + noise);
            }
        }

        let df = df! {
            "year" => year_vec,
            "unit" => unit_vec,
            "first_treat" => g_vec,
            "outcome" => y_vec
        }
        .unwrap();

        Dataset::new(df)
    }

    #[test]
    fn test_validate_staggered_did_group_time_atts_recover_true_effects() {
        // Validate that group-time ATT estimates recover the true ATTs
        // from the DGP within tolerance.
        //
        // True ATTs: cohort 2002 -> 2.0, cohort 2004 -> 4.0, cohort 2006 -> 6.0
        let dataset = create_validation_dataset_heterogeneous();
        let config = StaggeredDidConfig {
            comparison_group: ComparisonGroup::NeverTreated,
            estimation_method: AttEstimationMethod::OutcomeRegression,
            bootstrap: 50,
            seed: Some(42),
            min_obs_per_cell: 3,
            ..Default::default()
        };

        let result = run_staggered_did(
            &dataset,
            "outcome",
            "first_treat",
            "year",
            "unit",
            None,
            config,
        )
        .unwrap();

        // Check that we have the expected cohorts
        assert_eq!(result.cohorts.len(), 3);
        assert!(result.cohorts.contains(&2002));
        assert!(result.cohorts.contains(&2004));
        assert!(result.cohorts.contains(&2006));

        // Check post-treatment group-time ATTs
        let true_atts: std::collections::HashMap<i64, f64> =
            [(2002, 2.0), (2004, 4.0), (2006, 6.0)]
                .into_iter()
                .collect();

        for att in &result.group_time_atts {
            if att.post_treatment {
                let true_att = true_atts[&att.group];
                assert!(
                    (att.att - true_att).abs() < 1.0,
                    "ATT(g={}, t={}) = {:.4}, expected ~{:.1} (tol=1.0)",
                    att.group,
                    att.time,
                    att.att,
                    true_att
                );
            }
        }
    }

    #[test]
    fn test_validate_staggered_did_event_study_pre_treatment_near_zero() {
        // Validate that pre-treatment event study coefficients are near zero,
        // confirming parallel trends hold in the DGP.
        let dataset = create_validation_dataset_heterogeneous();
        let config = StaggeredDidConfig {
            comparison_group: ComparisonGroup::NeverTreated,
            estimation_method: AttEstimationMethod::OutcomeRegression,
            bootstrap: 50,
            seed: Some(42),
            min_obs_per_cell: 3,
            ..Default::default()
        };

        let result = run_staggered_did(
            &dataset,
            "outcome",
            "first_treat",
            "year",
            "unit",
            None,
            config,
        )
        .unwrap();

        // All pre-treatment event study effects (relative time < 0) should be near 0
        let pre_effects: Vec<&AggregatedEffect> =
            result.event_study.iter().filter(|e| e.key < 0).collect();

        assert!(
            !pre_effects.is_empty(),
            "Should have at least one pre-treatment event study coefficient"
        );

        for e in &pre_effects {
            assert!(
                e.att.abs() < 1.5,
                "Pre-treatment event study at e={} should be near 0, got {:.4}",
                e.key,
                e.att
            );
        }
    }

    #[test]
    fn test_validate_staggered_did_event_study_post_treatment_positive() {
        // Validate that post-treatment event study coefficients are positive
        // and reflect the (weighted) true ATTs.
        let dataset = create_validation_dataset_heterogeneous();
        let config = StaggeredDidConfig {
            comparison_group: ComparisonGroup::NeverTreated,
            estimation_method: AttEstimationMethod::OutcomeRegression,
            bootstrap: 50,
            seed: Some(42),
            min_obs_per_cell: 3,
            ..Default::default()
        };

        let result = run_staggered_did(
            &dataset,
            "outcome",
            "first_treat",
            "year",
            "unit",
            None,
            config,
        )
        .unwrap();

        let post_effects: Vec<&AggregatedEffect> =
            result.event_study.iter().filter(|e| e.key >= 0).collect();

        assert!(
            !post_effects.is_empty(),
            "Should have post-treatment event study coefficients"
        );

        // Post-treatment effects should be positive (true ATTs range from 2 to 6)
        for e in &post_effects {
            assert!(
                e.att > 0.0,
                "Post-treatment event study at e={} should be positive, got {:.4}",
                e.key,
                e.att
            );
        }
    }

    #[test]
    fn test_validate_staggered_did_overall_att_weighted_average() {
        // Validate that the overall ATT is a reasonable weighted average
        // of the group-specific effects.
        //
        // With equal group sizes (10 units each) and true ATTs of 2, 4, 6,
        // the simple average of group ATTs would be ~4.0 (but weighted by
        // number of post-treatment observations, so may differ).
        let dataset = create_validation_dataset_heterogeneous();
        let config = StaggeredDidConfig {
            comparison_group: ComparisonGroup::NeverTreated,
            estimation_method: AttEstimationMethod::OutcomeRegression,
            bootstrap: 50,
            seed: Some(42),
            min_obs_per_cell: 3,
            ..Default::default()
        };

        let result = run_staggered_did(
            &dataset,
            "outcome",
            "first_treat",
            "year",
            "unit",
            None,
            config,
        )
        .unwrap();

        // Overall ATT should be between 2 and 6 (the range of true ATTs)
        assert!(
            result.overall_att.att > 1.0 && result.overall_att.att < 7.0,
            "Overall ATT should be between 1 and 7, got {:.4}",
            result.overall_att.att
        );

        // Overall ATT should be statistically significant (large true effects)
        assert!(
            result.overall_att.p_value < 0.10,
            "Overall ATT should be significant at 10%, p = {:.4}",
            result.overall_att.p_value
        );
    }

    #[test]
    fn test_validate_staggered_did_group_effects_heterogeneity() {
        // Validate that group-level aggregated effects preserve the
        // treatment effect heterogeneity across cohorts.
        //
        // The ordering should be: ATT(cohort 2006) > ATT(cohort 2004) > ATT(cohort 2002)
        let dataset = create_validation_dataset_heterogeneous();
        let config = StaggeredDidConfig {
            comparison_group: ComparisonGroup::NeverTreated,
            estimation_method: AttEstimationMethod::OutcomeRegression,
            bootstrap: 50,
            seed: Some(42),
            min_obs_per_cell: 3,
            ..Default::default()
        };

        let result = run_staggered_did(
            &dataset,
            "outcome",
            "first_treat",
            "year",
            "unit",
            None,
            config,
        )
        .unwrap();

        // Find group-level effects
        let group_2002 = result.group_effects.iter().find(|e| e.key == 2002);
        let group_2004 = result.group_effects.iter().find(|e| e.key == 2004);
        let group_2006 = result.group_effects.iter().find(|e| e.key == 2006);

        assert!(group_2002.is_some(), "Should have cohort 2002 effect");
        assert!(group_2004.is_some(), "Should have cohort 2004 effect");
        assert!(group_2006.is_some(), "Should have cohort 2006 effect");

        let att_2002 = group_2002.unwrap().att;
        let att_2004 = group_2004.unwrap().att;
        let att_2006 = group_2006.unwrap().att;

        // Each group ATT should be close to the true value
        assert!(
            (att_2002 - 2.0).abs() < 1.0,
            "Cohort 2002 ATT should be ~2.0, got {:.4}",
            att_2002
        );
        assert!(
            (att_2004 - 4.0).abs() < 1.0,
            "Cohort 2004 ATT should be ~4.0, got {:.4}",
            att_2004
        );
        assert!(
            (att_2006 - 6.0).abs() < 1.5,
            "Cohort 2006 ATT should be ~6.0, got {:.4}",
            att_2006
        );

        // Ordering should preserve heterogeneity
        assert!(
            att_2006 > att_2004,
            "Cohort 2006 ({:.4}) should exceed cohort 2004 ({:.4})",
            att_2006,
            att_2004
        );
        assert!(
            att_2004 > att_2002,
            "Cohort 2004 ({:.4}) should exceed cohort 2002 ({:.4})",
            att_2004,
            att_2002
        );
    }

    #[test]
    fn test_validate_staggered_did_never_treated_comparison() {
        // Validate with never-treated comparison group explicitly, checking
        // structural properties of the result.
        let dataset = create_validation_dataset_heterogeneous();
        let config = StaggeredDidConfig {
            comparison_group: ComparisonGroup::NeverTreated,
            estimation_method: AttEstimationMethod::OutcomeRegression,
            bootstrap: 50,
            seed: Some(42),
            min_obs_per_cell: 3,
            ..Default::default()
        };

        let result = run_staggered_did(
            &dataset,
            "outcome",
            "first_treat",
            "year",
            "unit",
            None,
            config,
        )
        .unwrap();

        // Dataset has 50 units x 9 periods = 450 observations
        assert_eq!(result.n_obs, 450);

        // 30 treated units, 20 never-treated
        assert_eq!(result.n_treated, 30);
        assert_eq!(result.n_never_treated, 20);

        // All group-time ATTs should have non-zero comparison observations
        for att in &result.group_time_atts {
            assert!(
                att.n_comparison > 0,
                "ATT(g={}, t={}) should have comparison observations",
                att.group,
                att.time
            );
        }

        // Pre-trend test should not reject (DGP has parallel trends)
        if let Some(ref pretrend) = result.pretrend_test {
            assert!(
                pretrend.p_value > 0.01,
                "Pre-trend test should not reject at 1% level, p = {:.4}",
                pretrend.p_value
            );
        }
    }

    #[test]
    fn test_validate_staggered_did_homogeneous_effect() {
        // Validate with homogeneous treatment effect (all cohorts get ATT = 3.0).
        // The overall ATT should be very close to 3.0.
        let mut year_vec = Vec::new();
        let mut unit_vec = Vec::new();
        let mut g_vec = Vec::new();
        let mut y_vec = Vec::new();

        let true_att = 3.0;

        for year in 2000..=2008 {
            for unit in 1..=40 {
                year_vec.push(year as f64);
                unit_vec.push(unit as f64);

                let g = if unit <= 10 {
                    2002
                } else if unit <= 20 {
                    2004
                } else {
                    0
                };
                g_vec.push(g as f64);

                let base = unit as f64 * 0.5;
                let trend = (year - 2000) as f64 * 0.3;
                let treated = if g > 0 && year >= g { true_att } else { 0.0 };
                let noise = ((unit * year) % 13) as f64 * 0.04 - 0.26;

                y_vec.push(base + trend + treated + noise);
            }
        }

        let df = df! {
            "year" => year_vec,
            "unit" => unit_vec,
            "first_treat" => g_vec,
            "outcome" => y_vec
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let config = StaggeredDidConfig {
            comparison_group: ComparisonGroup::NeverTreated,
            estimation_method: AttEstimationMethod::OutcomeRegression,
            bootstrap: 50,
            seed: Some(42),
            min_obs_per_cell: 3,
            ..Default::default()
        };

        let result = run_staggered_did(
            &dataset,
            "outcome",
            "first_treat",
            "year",
            "unit",
            None,
            config,
        )
        .unwrap();

        // With homogeneous effects, overall ATT should be close to true_att
        assert!(
            (result.overall_att.att - true_att).abs() < 1.0,
            "Overall ATT should be ~{}, got {:.4}",
            true_att,
            result.overall_att.att
        );

        // Group effects should also be close to the common ATT
        for ge in &result.group_effects {
            assert!(
                (ge.att - true_att).abs() < 1.0,
                "Group {} ATT should be ~{}, got {:.4}",
                ge.key,
                true_att,
                ge.att
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Bootstrap: the previous per-replicate implementation, kept as the reference
    // ─────────────────────────────────────────────────────────────────────────

    /// The bootstrap ATT of one cell as computed before the table-based rewrite
    /// (verbatim apart from the name).
    #[allow(clippy::too_many_arguments)]
    fn reference_bootstrap_att(
        y: &Array1<f64>,
        g_col: &Array1<f64>,
        t_col: &Array1<f64>,
        unit_ids: &Array1<f64>,
        _x_cov: &Option<Array2<f64>>,
        panel_map: &BTreeMap<(i64, i64), usize>,
        boot_units: &[i64],
        g: i64,
        t: i64,
        base_t: i64,
        config: &StaggeredDidConfig,
    ) -> Option<f64> {
        let n = y.len();

        // Count unit occurrences in bootstrap sample
        let mut unit_weights: BTreeMap<i64, usize> = BTreeMap::new();
        for &u in boot_units {
            *unit_weights.entry(u).or_insert(0) += 1;
        }

        // Compute weighted outcome changes
        let mut sum_dy_treated = 0.0;
        let mut n_treated_weighted = 0.0;
        let mut sum_dy_comp = 0.0;
        let mut n_comp_weighted = 0.0;

        for i in 0..n {
            let gi = g_col[i] as i64;
            let ti = t_col[i] as i64;
            let ui = unit_ids[i] as i64;

            // Get bootstrap weight
            let weight = *unit_weights.get(&ui).unwrap_or(&0) as f64;
            if weight == 0.0 {
                continue;
            }

            if ti != t {
                continue;
            }

            // Get base period observation
            let base_idx = panel_map.get(&(ui, base_t))?;
            let dy = y[i] - y[*base_idx];

            if gi == g {
                // Treated
                sum_dy_treated += weight * dy;
                n_treated_weighted += weight;
            } else {
                // Check if comparison
                let is_comparison = match config.comparison_group {
                    ComparisonGroup::NeverTreated => gi <= 0,
                    ComparisonGroup::NotYetTreated => gi > t || gi <= 0,
                };

                if is_comparison {
                    sum_dy_comp += weight * dy;
                    n_comp_weighted += weight;
                }
            }
        }

        if n_treated_weighted > 0.0 && n_comp_weighted > 0.0 {
            Some(sum_dy_treated / n_treated_weighted - sum_dy_comp / n_comp_weighted)
        } else {
            None
        }
    }

    /// The previous bootstrap loop: per replicate, resample units and recompute
    /// every cell from the rows.
    fn reference_draws(
        ds: &Dataset,
        config: &StaggeredDidConfig,
        cells: &[(i64, i64, i64)],
    ) -> Vec<Vec<f64>> {
        let col = |c: &str| DesignMatrix::extract_column(ds.df(), c).unwrap();
        let (y, g_col, t_col, unit_ids) =
            (col("outcome"), col("first_treat"), col("year"), col("unit"));
        let n = y.len();
        let panel_map = build_panel_map(&t_col, &unit_ids);
        let unit_list: Vec<i64> = (0..n)
            .map(|i| unit_ids[i] as i64)
            .collect::<BTreeSet<i64>>()
            .into_iter()
            .collect();
        let n_units = unit_list.len();
        let mut rng = StdRng::seed_from_u64(config.seed.unwrap());

        let mut out = vec![Vec::new(); cells.len()];
        for _ in 0..config.bootstrap {
            let boot_units: Vec<i64> = (0..n_units)
                .map(|_| unit_list[rng.gen_range(0..n_units)])
                .collect();
            for (c, &(g, t, base_t)) in cells.iter().enumerate() {
                if let Some(a) = reference_bootstrap_att(
                    &y,
                    &g_col,
                    &t_col,
                    &unit_ids,
                    &None,
                    &panel_map,
                    &boot_units,
                    g,
                    t,
                    base_t,
                    config,
                ) {
                    if a.is_finite() {
                        out[c].push(a);
                    }
                }
            }
        }
        out
    }

    #[test]
    fn test_bootstrap_draws_match_reference() {
        for (ds, comparison_group) in [
            (create_staggered_dataset(), ComparisonGroup::NeverTreated),
            (create_staggered_dataset(), ComparisonGroup::NotYetTreated),
            (
                create_unbalanced_staggered_dataset(),
                ComparisonGroup::NeverTreated,
            ),
        ] {
            let config = StaggeredDidConfig {
                comparison_group,
                bootstrap: 200,
                seed: Some(42),
                min_obs_per_cell: 3,
                ..Default::default()
            };
            let point = run_staggered_did(
                &ds,
                "outcome",
                "first_treat",
                "year",
                "unit",
                None,
                StaggeredDidConfig {
                    bootstrap: 0,
                    ..config.clone()
                },
            )
            .unwrap();
            let cells: Vec<(i64, i64, i64)> = point
                .group_time_atts
                .iter()
                .map(|g| (g.group, g.time, g.group + config.base_period as i64))
                .collect();
            let expected = reference_draws(&ds, &config, &cells);

            let col = |c: &str| DesignMatrix::extract_column(ds.df(), c).unwrap();
            let (y, g, t, u) = (col("outcome"), col("first_treat"), col("year"), col("unit"));
            let panel_map = build_panel_map(&t, &u);
            let table =
                build_boot_table(&y, &g, &t, &u, &panel_map, &cells, comparison_group).unwrap();
            // the table builder checks the budget once per (g, t) cell
            expect_cancelled(with_budget(cancelled(), || {
                build_boot_table(&y, &g, &t, &u, &panel_map, &cells, comparison_group)
            }));
            let draws = bootstrap_draws(&table, 200, &mut StdRng::seed_from_u64(42)).unwrap();
            assert_eq!(draws.dim(), (200, cells.len()));
            for (c, want) in expected.iter().enumerate() {
                let got: Vec<f64> = draws
                    .column(c)
                    .iter()
                    .copied()
                    .filter(|v| v.is_finite())
                    .collect();
                assert_eq!(got.len(), want.len(), "cell {:?}", cells[c]);
                for (a, b) in got.iter().zip(want) {
                    assert!(
                        (a - b).abs() <= 1e-12 * b.abs().max(1.0),
                        "cell {:?}: {a} vs {b}",
                        cells[c]
                    );
                }
            }
            // the bootstrap honours a cancelled budget
            expect_cancelled(with_budget(cancelled(), || {
                bootstrap_draws(&table, 10, &mut StdRng::seed_from_u64(1))
            }));
        }
    }

    #[test]
    fn test_staggered_did_rejects_unallocatable_bootstrap() {
        let ds = create_staggered_dataset();
        let config = StaggeredDidConfig {
            bootstrap: usize::MAX / 4,
            seed: Some(1),
            min_obs_per_cell: 3,
            ..Default::default()
        };
        expect_resource_limit(run_staggered_did(
            &ds,
            "outcome",
            "first_treat",
            "year",
            "unit",
            None,
            config,
        ));
    }

    #[test]
    fn test_staggered_did_stops_when_cancelled() {
        let ds = create_staggered_dataset();
        let config = StaggeredDidConfig {
            bootstrap: 50,
            seed: Some(42),
            min_obs_per_cell: 3,
            ..Default::default()
        };
        expect_cancelled(with_budget(cancelled(), || {
            run_staggered_did(&ds, "outcome", "first_treat", "year", "unit", None, config)
        }));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // T2-WP12: the R case did_cs_analytic (validation/r/cases/causal_did.R)
    // ─────────────────────────────────────────────────────────────────────────

    use p2a_validation::{Check, Tol};

    /// The case panel for tests that are not R-pinned (not registered).
    fn cs_data() -> Dataset {
        crate::test_support::load_case("did_cs_analytic").1
    }

    /// `ds` with column `col` replaced by `values` (same height).
    fn replace_col(ds: &Dataset, col: &str, values: Column) -> Dataset {
        let mut df = ds.df().clone();
        assert_eq!(values.len(), df.height(), "column {col}");
        df.with_column(values.with_name(col.into())).unwrap();
        Dataset::new(df)
    }

    /// `ds` without the rows for which `drop` is true.
    fn drop_rows(ds: &Dataset, drop: impl Fn(usize) -> bool) -> Dataset {
        let keep =
            BooleanChunked::from_iter_values("keep".into(), (0..ds.nrows()).map(|i| !drop(i)));
        Dataset::new(ds.df().filter(&keep).unwrap())
    }

    /// A null-free Int64 column (test-only read).
    fn i64s(ds: &Dataset, col: &str) -> Vec<i64> {
        ds.df()
            .column(col)
            .unwrap()
            .i64()
            .unwrap()
            .iter()
            .map(|v| v.expect("null-free column"))
            .collect()
    }

    /// `ds` with nulls in column `col` at `rows` (Int64 or Float64 columns).
    fn with_nulls(ds: &Dataset, col: &str, rows: &[usize]) -> Dataset {
        let c = ds.df().column(col).unwrap();
        let values = match c.dtype() {
            DataType::Int64 => {
                let v: Vec<Option<i64>> = c
                    .i64()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .map(|(i, x)| if rows.contains(&i) { None } else { x })
                    .collect();
                Column::new(col.into(), v)
            }
            _ => {
                let v: Vec<Option<f64>> = c
                    .cast(&DataType::Float64)
                    .unwrap()
                    .f64()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .map(|(i, x)| if rows.contains(&i) { None } else { x })
                    .collect();
                Column::new(col.into(), v)
            }
        };
        replace_col(ds, col, values)
    }

    /// The row of `(unit, time)` in the case panel.
    fn row_of(ds: &Dataset, unit: i64, time: i64) -> usize {
        let (u, t) = (i64s(ds, "unit"), i64s(ds, "time"));
        (0..u.len())
            .find(|&i| u[i] == unit && t[i] == time)
            .unwrap_or_else(|| panic!("no row ({unit}, {time})"))
    }

    fn run(ds: &Dataset, unit: &str, cfg: StaggeredDidConfig) -> EconResult<StaggeredDidResult> {
        run_staggered_did(ds, "y", "first_treat", "time", unit, None, cfg)
    }

    fn analytic(cg: ComparisonGroup) -> StaggeredDidConfig {
        StaggeredDidConfig {
            comparison_group: cg,
            bootstrap: 0,
            ..Default::default()
        }
    }

    /// Cell ATTs in result order.
    fn att(r: &StaggeredDidResult) -> Vec<f64> {
        r.group_time_atts.iter().map(|c| c.att).collect()
    }

    fn same(a: f64, b: f64) -> bool {
        a == b || (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12 * a.abs().max(b.abs())
    }

    /// Every number of two results agrees to 1e-12 relative (cells, aggregates, pre-trend).
    fn assert_same(a: &StaggeredDidResult, b: &StaggeredDidResult) {
        assert_eq!(a.n_obs, b.n_obs, "n_obs");
        assert_eq!(a.n_treated, b.n_treated, "n_treated");
        assert_eq!(a.n_never_treated, b.n_never_treated, "n_never_treated");
        assert_eq!(a.cohorts, b.cohorts, "cohorts");
        assert_eq!(a.periods, b.periods, "periods");
        assert_eq!(a.group_time_atts.len(), b.group_time_atts.len(), "cells");
        for (x, y) in a.group_time_atts.iter().zip(&b.group_time_atts) {
            let cell = (x.group, x.time);
            assert_eq!(cell, (y.group, y.time));
            assert_eq!((x.n_treated, x.n_comparison), (y.n_treated, y.n_comparison));
            for (p, q, what) in [
                (x.att, y.att, "att"),
                (x.std_error, y.std_error, "se"),
                (x.ci_lower, y.ci_lower, "ci_lower"),
                (x.ci_upper, y.ci_upper, "ci_upper"),
            ] {
                assert!(same(p, q), "cell {cell:?} {what}: {p} vs {q}");
            }
        }
        let effects = |r: &StaggeredDidResult| {
            let mut v: Vec<AggregatedEffect> = r.event_study.clone();
            v.extend(r.group_effects.iter().cloned());
            v.push(r.overall_att.clone());
            v
        };
        let (ea, eb) = (effects(a), effects(b));
        assert_eq!(ea.len(), eb.len(), "aggregates");
        for (x, y) in ea.iter().zip(&eb) {
            assert_eq!((x.key, x.n_cells, x.n_obs), (y.key, y.n_cells, y.n_obs));
            for (p, q, what) in [
                (x.att, y.att, "att"),
                (x.std_error, y.std_error, "se"),
                (x.ci_lower, y.ci_lower, "ci_lower"),
                (x.ci_upper, y.ci_upper, "ci_upper"),
                (x.p_value, y.p_value, "p"),
            ] {
                assert!(same(p, q), "aggregate {} {what}: {p} vs {q}", x.key);
            }
        }
        match (&a.pretrend_test, &b.pretrend_test) {
            (None, None) => {}
            (Some(x), Some(y)) => {
                assert_eq!(x.df, y.df, "pre-trend df");
                assert!(same(x.chi2, y.chi2), "chi2 {} vs {}", x.chi2, y.chi2);
                assert!(
                    same(x.p_value, y.p_value),
                    "p {} vs {}",
                    x.p_value,
                    y.p_value
                );
                assert_eq!(x.pre_atts.len(), y.pre_atts.len());
                for (p, q) in x.pre_atts.iter().zip(&y.pre_atts) {
                    assert!(same(*p, *q), "pre att {p} vs {q}");
                }
            }
            _ => panic!("one result has a pre-trend test, the other not"),
        }
    }

    #[test]
    fn test_validate_staggered_did_string_unit_ids() {
        use ComparisonGroup::NeverTreated;
        let (case, ds) = crate::test_support::load_case("did_cs_analytic");
        // C order of the labels: state_1 < state_10 < ... differs from the numeric order
        let labels: Vec<String> = i64s(&ds, "unit")
            .iter()
            .map(|u| format!("state_{u}"))
            .collect();
        let s = replace_col(&ds, "unit", Column::new("unit".into(), labels));
        let mut f = ds.clone();
        for c in ["unit", "time", "first_treat"] {
            let cast = f.df().column(c).unwrap().cast(&DataType::Float64).unwrap();
            f = replace_col(&f, c, cast);
        }
        let ri = run(&ds, "unit", analytic(NeverTreated)).unwrap();
        let rs = run(&s, "unit", analytic(NeverTreated)).unwrap();
        let rf = run(&f, "unit", analytic(NeverTreated)).unwrap();
        assert_same(&ri, &rs);
        assert_same(&ri, &rf);
        // R: fit <- .cs_attgt(d, "never"); ag <- .cs_aggregate(fit)
        let n = rs.n_obs;
        Check::new(&case)
            .vector("never_cell_att", &att(&rs), Tol::coef(n))
            .scalar("never_simple_att", rs.overall_att.att, Tol::coef(n))
            .vector(
                "never_group_att",
                &rs.group_effects.iter().map(|e| e.att).collect::<Vec<_>>(),
                Tol::coef(n),
            )
            .vector(
                "never_event_att",
                &rs.event_study.iter().map(|e| e.att).collect::<Vec<_>>(),
                Tol::coef(n),
            )
            .finish();
        assert_eq!(rs.n_obs, 720);
        assert_eq!(rs.n_treated, 68);
        assert_eq!(rs.n_never_treated, 22);
    }

    #[test]
    fn test_na_staggered_did_omits_missing_rows() {
        let ds = cs_data();
        let rows = [
            row_of(&ds, 4, 2),
            row_of(&ds, 9, 6),
            row_of(&ds, 30, 7),
            row_of(&ds, 50, 8),
        ];
        let mut na = with_nulls(&ds, "y", &rows[..1]);
        na = with_nulls(&na, "first_treat", &rows[1..2]);
        na = with_nulls(&na, "time", &rows[2..3]);
        na = with_nulls(&na, "unit", &rows[3..4]);
        let kept = drop_rows(&ds, |i| rows.contains(&i));
        let a = run(&na, "unit", analytic(ComparisonGroup::NeverTreated)).unwrap();
        let b = run(&kept, "unit", analytic(ComparisonGroup::NeverTreated)).unwrap();
        assert_same(&a, &b);
        assert_eq!(a.n_obs, 716);
        assert_eq!(
            a.missing,
            MissingSummary {
                n_total: 720,
                n_used: 716,
                n_dropped: 4,
                by_column: vec![
                    ("y".into(), 1),
                    ("first_treat".into(), 1),
                    ("time".into(), 1),
                    ("unit".into(), 1)
                ],
            }
        );
        assert!(b.missing.is_empty());
        let note = "4 of 720 rows dropped because of missing values (y: 1, first_treat: 1, time: 1, unit: 1)";
        assert!(a.warnings.iter().any(|w| w == note), "{:?}", a.warnings);
        let shown = format!("{a}");
        assert!(
            shown
                .trim_end()
                .ends_with(&format!("Missing values: {note}")),
            "{shown}"
        );
        assert_eq!(shown.matches(note).count(), 1, "{shown}");
    }

    #[test]
    fn test_na_staggered_did_errors_name_the_column() {
        let ds = cs_data();
        let cfg = || analytic(ComparisonGroup::NeverTreated);
        // an outcome stored as text
        let text: Vec<String> = ds
            .df()
            .column("y")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .map(|v| format!("{}", v.unwrap()))
            .collect();
        let s = replace_col(&ds, "y", Column::new("y".into(), text));
        match run(&s, "unit", cfg()) {
            Err(EconError::NonNumericColumn { column }) => assert_eq!(column, "y"),
            other => panic!("expected NonNumericColumn for y, got {other:?}"),
        }
        // an absent unit column
        match run(&ds, "no_such", cfg()) {
            Err(EconError::ColumnNotFound { column, .. }) => assert_eq!(column, "no_such"),
            other => panic!("expected ColumnNotFound, got {other:?}"),
        }
        // a fractional period
        let mut time: Vec<f64> = i64s(&ds, "time").iter().map(|&t| t as f64).collect();
        time[1] = 2.5;
        let frac = replace_col(&ds, "time", Column::new("time".into(), time));
        match run(&frac, "unit", cfg()) {
            Err(EconError::InvalidSpecification { message }) => assert!(
                message.contains("Time column 'time' must hold whole numbers"),
                "{message}"
            ),
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
        // a first treatment period that varies within a unit (unit 1 is cohort 3)
        let mut ft = i64s(&ds, "first_treat");
        ft[row_of(&ds, 1, 3)] = 5;
        let varying = replace_col(&ds, "first_treat", Column::new("first_treat".into(), ft));
        match run(&varying, "unit", cfg()) {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(
                    message.contains("must be constant within each unit"),
                    "{message}"
                );
                assert!(message.contains("unit '1'"), "{message}");
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }
}
