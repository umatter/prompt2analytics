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
use crate::errors::{EconError, EconResult};
use crate::linalg::design::{DesignMatrix, get_column_names};
use crate::linalg::matrix_ops::safe_inverse;
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

        if !self.warnings.is_empty() {
            writeln!(f)?;
            writeln!(f, "Warnings:")?;
            for w in &self.warnings {
                writeln!(f, "  - {}", w)?;
            }
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
/// * `outcome` - Name of the outcome variable
/// * `treatment_time` - Name of the column indicating when each unit was first treated
///   (use 0 or negative for never-treated units)
/// * `time_col` - Name of the time period column
/// * `unit_col` - Name of the unit/individual identifier column
/// * `covariates` - Optional covariate columns for conditional parallel trends
/// * `config` - Configuration options
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

    // Extract data
    let y = DesignMatrix::extract_column(dataset.df(), outcome).map_err(|e| {
        EconError::ColumnNotFound {
            column: outcome.to_string(),
            available: get_column_names(dataset.df()),
        }
    })?;

    let g_col = DesignMatrix::extract_column(dataset.df(), treatment_time).map_err(|e| {
        EconError::ColumnNotFound {
            column: treatment_time.to_string(),
            available: get_column_names(dataset.df()),
        }
    })?;

    let t_col = DesignMatrix::extract_column(dataset.df(), time_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: time_col.to_string(),
            available: get_column_names(dataset.df()),
        }
    })?;

    let unit_ids = DesignMatrix::extract_column(dataset.df(), unit_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: unit_col.to_string(),
            available: get_column_names(dataset.df()),
        }
    })?;

    let n = y.len();

    // Extract covariates if provided
    let x_cov: Option<Array2<f64>> = if let Some(cov_cols) = covariates {
        let design = DesignMatrix::from_dataframe(dataset.df(), cov_cols, true)?;
        Some(design.data)
    } else {
        None
    };

    // Identify cohorts and periods
    let mut cohort_set: BTreeSet<i64> = BTreeSet::new();
    let mut period_set: BTreeSet<i64> = BTreeSet::new();
    let mut never_treated_count = 0;
    let mut treated_units: BTreeSet<i64> = BTreeSet::new();

    for i in 0..n {
        let g = g_col[i] as i64;
        let t = t_col[i] as i64;
        let u = unit_ids[i] as i64;

        period_set.insert(t);

        if g <= 0 {
            // Never treated (G = 0 or negative indicates never-treated)
            never_treated_count += 1;
        } else {
            cohort_set.insert(g);
            treated_units.insert(u);
        }
    }

    let cohorts: Vec<i64> = cohort_set.into_iter().collect();
    let periods: Vec<i64> = period_set.into_iter().collect();
    let n_treated_units = treated_units.len();

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
    let mut panel_map: BTreeMap<(i64, i64), usize> = BTreeMap::new();
    for i in 0..n {
        let u = unit_ids[i] as i64;
        let t = t_col[i] as i64;
        panel_map.insert((u, t), i);
    }

    // Compute group-time ATTs
    let mut group_time_atts = Vec::new();

    for &g in &cohorts {
        // Base period is g - 1 (or as configured)
        let base_t = g + config.base_period as i64;

        for &t in &periods {
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

    // Collect bootstrap samples for group-time ATTs
    let n_units: usize = {
        let unique_units: BTreeSet<i64> = (0..n).map(|i| unit_ids[i] as i64).collect();
        unique_units.len()
    };

    let unit_list: Vec<i64> = {
        let unique_units: BTreeSet<i64> = (0..n).map(|i| unit_ids[i] as i64).collect();
        unique_units.into_iter().collect()
    };

    let mut boot_atts: Vec<Vec<f64>> = vec![Vec::new(); group_time_atts.len()];

    for _ in 0..config.bootstrap {
        // Block bootstrap by unit
        let boot_units: Vec<i64> = (0..n_units)
            .map(|_| unit_list[rng.gen_range(0..n_units)])
            .collect();

        // Compute bootstrap ATTs
        for (idx, gta) in group_time_atts.iter().enumerate() {
            let boot_att = compute_bootstrap_att(
                &y,
                &g_col,
                &t_col,
                &unit_ids,
                &x_cov,
                &panel_map,
                &boot_units,
                gta.group,
                gta.time,
                gta.group + config.base_period as i64,
                &config,
            );

            if let Some(att) = boot_att {
                if att.is_finite() {
                    boot_atts[idx].push(att);
                }
            }
        }
    }

    // Update standard errors from bootstrap
    let z = quantile_normal(1.0 - (1.0 - config.confidence_level) / 2.0);

    for (idx, gta) in group_time_atts.iter_mut().enumerate() {
        if boot_atts[idx].len() >= 50 {
            let mean: f64 = boot_atts[idx].iter().sum::<f64>() / boot_atts[idx].len() as f64;
            let var: f64 = boot_atts[idx]
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / (boot_atts[idx].len() - 1) as f64;
            gta.std_error = var.sqrt();
            gta.ci_lower = gta.att - z * gta.std_error;
            gta.ci_upper = gta.att + z * gta.std_error;
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

    let n_periods = periods.len();
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
        n_never_treated: never_treated_count / n_periods.max(1),
        warnings,
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

/// Compute bootstrap ATT (simplified for efficiency).
#[allow(clippy::too_many_arguments)]
fn compute_bootstrap_att(
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
    use polars::prelude::*;

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
            [(2002, 2.0), (2004, 4.0), (2006, 6.0)].into_iter().collect();

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
}
