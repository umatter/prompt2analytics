//! Extended Two-Way Fixed Effects (ETWFE) for Staggered Treatments.
//!
//! This module implements Wooldridge's (2021, 2023) Extended TWFE approach for
//! difference-in-differences with staggered treatment adoption.
//!
//! # Overview
//!
//! Standard TWFE can produce biased estimates when treatment timing varies across
//! units (staggered adoption). ETWFE addresses this by saturating the model with
//! all cohort × time interactions, allowing heterogeneous treatment effects.
//!
//! # Model Specification
//!
//! ## Standard TWFE (potentially biased)
//!
//! ```text
//! y_it = α_i + λ_t + β · D_it + ε_it
//! ```
//!
//! ## Extended TWFE (Wooldridge)
//!
//! ```text
//! y_it = α_i + λ_t + Σ_g Σ_s β_{g,s} · 1(G_i=g) · 1(t=s) · D_it + X_it'γ + ε_it
//! ```
//!
//! Where:
//! - α_i: Unit fixed effects
//! - λ_t: Time fixed effects
//! - G_i: Treatment cohort (first treatment period for unit i)
//! - D_it: Treatment indicator (1 if treated at time t)
//! - β_{g,s}: Cohort-time specific treatment effect
//!
//! # Aggregation
//!
//! The cohort-time effects can be aggregated into:
//! - **Event Study**: Effects by relative time since treatment
//! - **Cohort Average**: Average effect for each treatment cohort
//! - **Overall ATT**: Single weighted average across all effects
//!
//! # References
//!
//! - Wooldridge, J.M. (2021). Two-Way Fixed Effects, the Two-Way Mundlak
//!   Regression, and Difference-in-Differences Estimators. Working paper.
//!
//! - Wooldridge, J.M. (2023). Simple Approaches to Nonlinear Difference-in-
//!   Differences with Panel Data. *The Econometrics Journal*, 26(3), C31-C66.
//!
//! R equivalent: `etwfe::etwfe()` (Grant McDermott)

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::design::{DesignMatrix, get_column_names};
use crate::linalg::matrix_ops::{safe_inverse, xtx, xty};
use crate::traits::estimator::t_test_p_value;

/// Configuration for ETWFE estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EtwfeConfig {
    /// Reference time period (default: first period)
    pub tref: Option<i64>,
    /// Reference cohort (default: never-treated or last cohort)
    pub gref: Option<i64>,
    /// Control group: "never" (never-treated only) or "notyet" (not-yet-treated)
    pub cgroup: ControlGroup,
    /// Include anticipation effects (periods before treatment)
    pub anticipation: usize,
}

impl Default for EtwfeConfig {
    fn default() -> Self {
        Self {
            tref: None,
            gref: None,
            cgroup: ControlGroup::NotYet,
            anticipation: 0,
        }
    }
}

/// Control group selection for ETWFE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ControlGroup {
    /// Use only never-treated units as controls
    Never,
    /// Use not-yet-treated units as controls (default)
    #[default]
    NotYet,
}

impl fmt::Display for ControlGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ControlGroup::Never => write!(f, "Never-Treated"),
            ControlGroup::NotYet => write!(f, "Not-Yet-Treated"),
        }
    }
}

/// Cohort-time specific treatment effect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohortTimeEffect {
    /// Treatment cohort (first treatment period)
    pub cohort: i64,
    /// Time period
    pub time: i64,
    /// Relative time (time - cohort)
    pub rel_time: i64,
    /// Estimated effect
    pub estimate: f64,
    /// Standard error
    pub std_error: f64,
    /// t-statistic
    pub t_stat: f64,
    /// p-value
    pub p_value: f64,
    /// Number of treated observations
    pub n_treated: usize,
}

/// Aggregated treatment effect (event study or cohort average).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedEffect {
    /// Aggregation key (relative time for event study, cohort for group average)
    pub key: i64,
    /// Aggregation type
    pub agg_type: String,
    /// Estimated effect (weighted average)
    pub estimate: f64,
    /// Standard error
    pub std_error: f64,
    /// 95% CI lower bound
    pub ci_lower: f64,
    /// 95% CI upper bound
    pub ci_upper: f64,
    /// Number of underlying cohort-time effects
    pub n_effects: usize,
}

/// Result from ETWFE estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EtwfeResult {
    /// Model description
    pub model: String,
    /// Dependent variable
    pub dep_var: String,
    /// Control variables (if any)
    pub controls: Vec<String>,
    /// Control group type
    pub control_group: ControlGroup,
    /// Cohort-time specific effects
    pub cohort_time_effects: Vec<CohortTimeEffect>,
    /// Event study aggregation (effects by relative time)
    pub event_study: Vec<AggregatedEffect>,
    /// Cohort average effects
    pub cohort_avg: Vec<AggregatedEffect>,
    /// Overall ATT (simple average)
    pub att_simple: f64,
    /// Overall ATT standard error
    pub att_se: f64,
    /// Overall ATT (weighted by cohort size)
    pub att_weighted: f64,
    /// Overall ATT weighted standard error
    pub att_weighted_se: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of treated observations
    pub n_treated: usize,
    /// Number of control observations
    pub n_control: usize,
    /// Number of cohorts
    pub n_cohorts: usize,
    /// Number of time periods
    pub n_periods: usize,
    /// R-squared from underlying regression
    pub r_squared: f64,
    /// Reference time period
    pub ref_time: i64,
    /// Reference cohort (if applicable)
    pub ref_cohort: Option<i64>,
}

impl fmt::Display for EtwfeResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{}", "=".repeat(70))?;
        writeln!(f, "{:^70}", "Extended Two-Way Fixed Effects (ETWFE)")?;
        writeln!(f, "{}", "=".repeat(70))?;

        writeln!(f, "\nModel: {}", self.model)?;
        writeln!(f, "Outcome: {}", self.dep_var)?;
        writeln!(f, "Control Group: {}", self.control_group)?;
        if !self.controls.is_empty() {
            writeln!(f, "Controls: {}", self.controls.join(", "))?;
        }

        writeln!(f, "\n{:-<70}", "")?;
        writeln!(
            f,
            "Sample: {} obs ({} treated, {} control)",
            self.n_obs, self.n_treated, self.n_control
        )?;
        writeln!(
            f,
            "Cohorts: {}, Periods: {}",
            self.n_cohorts, self.n_periods
        )?;

        // Overall ATT
        writeln!(f, "\n{:-<70}", "")?;
        writeln!(f, "Overall Average Treatment Effect on the Treated (ATT)")?;
        writeln!(f, "{:-<70}", "")?;
        writeln!(
            f,
            "  Simple Average:   {:>10.4} (SE: {:.4})",
            self.att_simple, self.att_se
        )?;
        writeln!(
            f,
            "  Weighted Average: {:>10.4} (SE: {:.4})",
            self.att_weighted, self.att_weighted_se
        )?;

        // Event Study
        if !self.event_study.is_empty() {
            writeln!(f, "\n{:-<70}", "")?;
            writeln!(f, "Event Study (Effects by Time Relative to Treatment)")?;
            writeln!(f, "{:-<70}", "")?;
            writeln!(
                f,
                "{:>8} {:>12} {:>10} {:>12} {:>12}",
                "Rel.Time", "Estimate", "SE", "95% CI Lo", "95% CI Hi"
            )?;
            for eff in &self.event_study {
                let marker = if eff.key == 0 { " *" } else { "" };
                writeln!(
                    f,
                    "{:>8}{} {:>12.4} {:>10.4} {:>12.4} {:>12.4}",
                    eff.key, marker, eff.estimate, eff.std_error, eff.ci_lower, eff.ci_upper
                )?;
            }
            writeln!(f, "\n  * Treatment onset (relative time = 0)")?;
        }

        // Cohort effects
        if !self.cohort_avg.is_empty() && self.cohort_avg.len() <= 10 {
            writeln!(f, "\n{:-<70}", "")?;
            writeln!(f, "Cohort Average Effects")?;
            writeln!(f, "{:-<70}", "")?;
            writeln!(
                f,
                "{:>10} {:>12} {:>10} {:>12} {:>12}",
                "Cohort", "Estimate", "SE", "95% CI Lo", "95% CI Hi"
            )?;
            for eff in &self.cohort_avg {
                writeln!(
                    f,
                    "{:>10} {:>12.4} {:>10.4} {:>12.4} {:>12.4}",
                    eff.key, eff.estimate, eff.std_error, eff.ci_lower, eff.ci_upper
                )?;
            }
        }

        writeln!(f, "\nR-squared: {:.4}", self.r_squared)?;

        Ok(())
    }
}

/// Run Extended Two-Way Fixed Effects estimation.
///
/// # Arguments
///
/// * `dataset` - Panel dataset
/// * `y_col` - Outcome variable column
/// * `unit_col` - Unit/entity identifier column
/// * `time_col` - Time period column
/// * `treat_col` - Treatment indicator column (1 if treated, 0 otherwise)
/// * `first_treat_col` - First treatment period column (cohort identifier)
/// * `x_cols` - Optional control variables
/// * `config` - ETWFE configuration
///
/// # Returns
///
/// `EtwfeResult` with cohort-time effects, event study, and aggregated ATTs.
pub fn run_etwfe(
    dataset: &Dataset,
    y_col: &str,
    unit_col: &str,
    time_col: &str,
    treat_col: &str,
    first_treat_col: &str,
    x_cols: Option<&[&str]>,
    config: Option<EtwfeConfig>,
) -> EconResult<EtwfeResult> {
    let config = config.unwrap_or_default();
    let df = dataset.df();
    let n = df.height();

    // Extract columns
    let y = DesignMatrix::extract_column(df, y_col).map_err(|_| EconError::ColumnNotFound {
        column: y_col.to_string(),
        available: get_column_names(df),
    })?;

    let units: Vec<String> = extract_string_column(df, unit_col)?;
    let times: Vec<i64> = extract_int_column(df, time_col)?;
    let treat: Vec<f64> = extract_float_column(df, treat_col)?;
    let first_treat: Vec<i64> = extract_int_column(df, first_treat_col)?;

    // Identify unique cohorts and time periods
    let unique_times: BTreeSet<i64> = times.iter().copied().collect();
    let unique_cohorts: BTreeSet<i64> = first_treat
        .iter()
        .filter(|&&g| g > 0) // Exclude never-treated (coded as 0 or very large)
        .copied()
        .collect();

    let all_periods: Vec<i64> = unique_times.iter().copied().collect();
    let n_periods = all_periods.len();

    // Determine reference time (first period by default)
    let ref_time = config
        .tref
        .unwrap_or_else(|| *all_periods.first().unwrap_or(&0));

    // Identify never-treated units
    let _never_treated: BTreeSet<String> = units
        .iter()
        .zip(first_treat.iter())
        .filter(|(_, g)| **g == 0 || **g == i64::MAX || !unique_cohorts.contains(*g))
        .map(|(u, _)| u.clone())
        .collect();

    // Build cohort-time interaction terms
    // For each (cohort g, time t) combination where t >= g, create an indicator
    let mut cohort_time_pairs: Vec<(i64, i64)> = Vec::new();
    for &g in &unique_cohorts {
        for &t in &all_periods {
            // Include post-treatment periods
            let rel_time = t - g;
            if rel_time >= -(config.anticipation as i64) {
                cohort_time_pairs.push((g, t));
            }
        }
    }

    // Count observations per cohort-time
    let mut cohort_time_counts: HashMap<(i64, i64), usize> = HashMap::new();
    for i in 0..n {
        let g = first_treat[i];
        let t = times[i];
        if unique_cohorts.contains(&g) && treat[i] > 0.5 {
            *cohort_time_counts.entry((g, t)).or_insert(0) += 1;
        }
    }

    // Build design matrix with cohort-time interactions
    let n_gt = cohort_time_pairs.len();
    let n_controls = x_cols.map_or(0, |x| x.len());
    let n_params = n_gt + n_controls;

    if n_params == 0 {
        return Err(EconError::InvalidSpecification {
            message: "No cohort-time effects to estimate (no treated units?)".to_string(),
        });
    }

    // Create X matrix: cohort-time dummies + controls
    let mut x_data = Array2::<f64>::zeros((n, n_params));

    for i in 0..n {
        let g = first_treat[i];
        let t = times[i];
        let d = treat[i];

        // Cohort-time interactions: D_it × 1(G=g) × 1(t=t)
        for (j, &(cg, ct)) in cohort_time_pairs.iter().enumerate() {
            if g == cg && t == ct && d > 0.5 {
                x_data[[i, j]] = 1.0;
            }
        }
    }

    // Add control variables
    if let Some(controls) = x_cols {
        for (c_idx, &ctrl) in controls.iter().enumerate() {
            let ctrl_vals = extract_float_column(df, ctrl)?;
            for i in 0..n {
                x_data[[i, n_gt + c_idx]] = ctrl_vals[i];
            }
        }
    }

    // Demean for unit and time fixed effects (within transformation)
    let (y_demeaned, x_demeaned) = demean_panel(&y, &x_data, &units, &times)?;

    // Run OLS on demeaned data
    let xtx_mat = xtx(&x_demeaned.view());
    let (xtx_inv, _) = safe_inverse(&xtx_mat.view()).map_err(|_| EconError::SingularMatrix {
        context: "ETWFE design matrix".to_string(),
        suggestion: "Check for multicollinearity or empty cohort-time cells".to_string(),
    })?;

    let xty_vec = xty(&x_demeaned.view(), &y_demeaned);
    let beta: Array1<f64> = xtx_inv.dot(&xty_vec);

    // Compute residuals and standard errors
    let fitted = x_demeaned.dot(&beta);
    let residuals = &y_demeaned - &fitted;
    let ssr: f64 = residuals.iter().map(|r| r * r).sum();
    let df_resid = n.saturating_sub(n_params);
    let sigma2 = if df_resid > 0 {
        ssr / df_resid as f64
    } else {
        ssr
    };

    // Variance-covariance matrix
    let vcov = &xtx_inv * sigma2;

    // R-squared
    let y_mean = y_demeaned.mean().unwrap_or(0.0);
    let tss: f64 = y_demeaned.iter().map(|yi| (yi - y_mean).powi(2)).sum();
    let r_squared = if tss > 0.0 { 1.0 - ssr / tss } else { 0.0 };

    // Extract cohort-time effects
    let mut cohort_time_effects = Vec::new();
    for (j, &(g, t)) in cohort_time_pairs.iter().enumerate() {
        let estimate = beta[j];
        let se = vcov[[j, j]].max(0.0).sqrt();
        let t_stat = if se > 1e-10 { estimate / se } else { 0.0 };
        let p_value = t_test_p_value(t_stat, df_resid as f64);
        let n_treated = cohort_time_counts.get(&(g, t)).copied().unwrap_or(0);

        cohort_time_effects.push(CohortTimeEffect {
            cohort: g,
            time: t,
            rel_time: t - g,
            estimate,
            std_error: se,
            t_stat,
            p_value,
            n_treated,
        });
    }

    // Aggregate to event study (by relative time)
    let event_study = aggregate_by_relative_time(&cohort_time_effects);

    // Aggregate to cohort averages
    let cohort_avg = aggregate_by_cohort(&cohort_time_effects);

    // Overall ATT (simple and weighted)
    let (att_simple, att_se, att_weighted, att_weighted_se) =
        compute_overall_att(&cohort_time_effects);

    // Count treated/control observations
    let n_treated: usize = treat.iter().filter(|&&d| d > 0.5).count();
    let n_control = n - n_treated;

    let controls_vec = x_cols.map_or(vec![], |c| c.iter().map(|s| s.to_string()).collect());

    Ok(EtwfeResult {
        model: format!(
            "ETWFE ({} cohorts × {} periods)",
            unique_cohorts.len(),
            n_periods
        ),
        dep_var: y_col.to_string(),
        controls: controls_vec,
        control_group: config.cgroup,
        cohort_time_effects,
        event_study,
        cohort_avg,
        att_simple,
        att_se,
        att_weighted,
        att_weighted_se,
        n_obs: n,
        n_treated,
        n_control,
        n_cohorts: unique_cohorts.len(),
        n_periods,
        r_squared,
        ref_time,
        ref_cohort: None,
    })
}

/// Demean data by unit and time for two-way fixed effects.
fn demean_panel(
    y: &Array1<f64>,
    x: &Array2<f64>,
    units: &[String],
    times: &[i64],
) -> EconResult<(Array1<f64>, Array2<f64>)> {
    let n = y.len();
    let k = x.ncols();

    // Compute unit means
    let mut unit_counts: HashMap<String, usize> = HashMap::new();
    let mut unit_y_sums: HashMap<String, f64> = HashMap::new();
    let mut unit_x_sums: HashMap<String, Vec<f64>> = HashMap::new();

    for i in 0..n {
        let u = &units[i];
        *unit_counts.entry(u.clone()).or_insert(0) += 1;
        *unit_y_sums.entry(u.clone()).or_insert(0.0) += y[i];

        let x_sum = unit_x_sums.entry(u.clone()).or_insert_with(|| vec![0.0; k]);
        for j in 0..k {
            x_sum[j] += x[[i, j]];
        }
    }

    // Compute time means
    let mut time_counts: HashMap<i64, usize> = HashMap::new();
    let mut time_y_sums: HashMap<i64, f64> = HashMap::new();
    let mut time_x_sums: HashMap<i64, Vec<f64>> = HashMap::new();

    for i in 0..n {
        let t = times[i];
        *time_counts.entry(t).or_insert(0) += 1;
        *time_y_sums.entry(t).or_insert(0.0) += y[i];

        let x_sum = time_x_sums.entry(t).or_insert_with(|| vec![0.0; k]);
        for j in 0..k {
            x_sum[j] += x[[i, j]];
        }
    }

    // Grand means
    let grand_y_mean = y.mean().unwrap_or(0.0);
    let mut grand_x_mean = vec![0.0; k];
    for j in 0..k {
        grand_x_mean[j] = x.column(j).mean().unwrap_or(0.0);
    }

    // Apply two-way demeaning: x_it - x_i. - x_.t + x_..
    let mut y_demeaned = Array1::zeros(n);
    let mut x_demeaned = Array2::zeros((n, k));

    for i in 0..n {
        let u = &units[i];
        let t = times[i];

        let unit_n = unit_counts[u] as f64;
        let time_n = time_counts[&t] as f64;

        let unit_y_mean = unit_y_sums[u] / unit_n;
        let time_y_mean = time_y_sums[&t] / time_n;

        y_demeaned[i] = y[i] - unit_y_mean - time_y_mean + grand_y_mean;

        for j in 0..k {
            let unit_x_mean = unit_x_sums[u][j] / unit_n;
            let time_x_mean = time_x_sums[&t][j] / time_n;
            x_demeaned[[i, j]] = x[[i, j]] - unit_x_mean - time_x_mean + grand_x_mean[j];
        }
    }

    Ok((y_demeaned, x_demeaned))
}

/// Aggregate cohort-time effects by relative time (event study).
fn aggregate_by_relative_time(effects: &[CohortTimeEffect]) -> Vec<AggregatedEffect> {
    let mut by_rel_time: BTreeMap<i64, Vec<&CohortTimeEffect>> = BTreeMap::new();

    for eff in effects {
        by_rel_time.entry(eff.rel_time).or_default().push(eff);
    }

    by_rel_time
        .into_iter()
        .map(|(rel_time, effs)| {
            let n = effs.len();
            let estimate: f64 = effs.iter().map(|e| e.estimate).sum::<f64>() / n as f64;

            // Pooled standard error (simplified)
            let var_sum: f64 = effs.iter().map(|e| e.std_error.powi(2)).sum();
            let std_error = (var_sum / (n * n) as f64).sqrt();

            let ci_lower = estimate - 1.96 * std_error;
            let ci_upper = estimate + 1.96 * std_error;

            AggregatedEffect {
                key: rel_time,
                agg_type: "Event Study".to_string(),
                estimate,
                std_error,
                ci_lower,
                ci_upper,
                n_effects: n,
            }
        })
        .collect()
}

/// Aggregate cohort-time effects by cohort.
fn aggregate_by_cohort(effects: &[CohortTimeEffect]) -> Vec<AggregatedEffect> {
    let mut by_cohort: BTreeMap<i64, Vec<&CohortTimeEffect>> = BTreeMap::new();

    for eff in effects {
        // Only include post-treatment effects (rel_time >= 0)
        if eff.rel_time >= 0 {
            by_cohort.entry(eff.cohort).or_default().push(eff);
        }
    }

    by_cohort
        .into_iter()
        .map(|(cohort, effs)| {
            let n = effs.len();
            let estimate: f64 = effs.iter().map(|e| e.estimate).sum::<f64>() / n as f64;

            let var_sum: f64 = effs.iter().map(|e| e.std_error.powi(2)).sum();
            let std_error = (var_sum / (n * n) as f64).sqrt();

            let ci_lower = estimate - 1.96 * std_error;
            let ci_upper = estimate + 1.96 * std_error;

            AggregatedEffect {
                key: cohort,
                agg_type: "Cohort Average".to_string(),
                estimate,
                std_error,
                ci_lower,
                ci_upper,
                n_effects: n,
            }
        })
        .collect()
}

/// Compute overall ATT (simple and weighted averages).
fn compute_overall_att(effects: &[CohortTimeEffect]) -> (f64, f64, f64, f64) {
    // Only post-treatment effects
    let post_effects: Vec<&CohortTimeEffect> = effects.iter().filter(|e| e.rel_time >= 0).collect();

    if post_effects.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    // Simple average
    let n = post_effects.len();
    let att_simple: f64 = post_effects.iter().map(|e| e.estimate).sum::<f64>() / n as f64;
    let var_sum: f64 = post_effects.iter().map(|e| e.std_error.powi(2)).sum();
    let att_se = (var_sum / (n * n) as f64).sqrt();

    // Weighted by number of treated observations
    let total_weight: f64 = post_effects.iter().map(|e| e.n_treated as f64).sum();
    if total_weight > 0.0 {
        let att_weighted: f64 = post_effects
            .iter()
            .map(|e| e.estimate * e.n_treated as f64)
            .sum::<f64>()
            / total_weight;

        let weighted_var: f64 = post_effects
            .iter()
            .map(|e| (e.n_treated as f64).powi(2) * e.std_error.powi(2))
            .sum::<f64>()
            / (total_weight * total_weight);
        let att_weighted_se = weighted_var.sqrt();

        (att_simple, att_se, att_weighted, att_weighted_se)
    } else {
        (att_simple, att_se, att_simple, att_se)
    }
}

// Helper functions for column extraction

fn extract_string_column(df: &polars::frame::DataFrame, col: &str) -> EconResult<Vec<String>> {
    let column = df.column(col).map_err(|_| EconError::ColumnNotFound {
        column: col.to_string(),
        available: df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect(),
    })?;

    // Try string first
    if let Ok(str_col) = column.str() {
        return Ok(str_col
            .into_iter()
            .map(|opt| opt.unwrap_or("").to_string())
            .collect());
    }

    // Try integer and convert to string
    if let Ok(int_col) = column.i64() {
        return Ok(int_col
            .into_iter()
            .map(|opt| opt.unwrap_or(0).to_string())
            .collect());
    }

    Err(EconError::InvalidSpecification {
        message: format!("Column '{}' must be string or integer type", col),
    })
}

fn extract_int_column(df: &polars::frame::DataFrame, col: &str) -> EconResult<Vec<i64>> {
    let column = df.column(col).map_err(|_| EconError::ColumnNotFound {
        column: col.to_string(),
        available: df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect(),
    })?;

    // Try i64 first
    if let Ok(int_col) = column.i64() {
        return Ok(int_col.into_iter().map(|opt| opt.unwrap_or(0)).collect());
    }

    // Try i32
    if let Ok(int_col) = column.i32() {
        return Ok(int_col
            .into_iter()
            .map(|opt| opt.unwrap_or(0) as i64)
            .collect());
    }

    // Try f64 and convert
    if let Ok(float_col) = column.f64() {
        return Ok(float_col
            .into_iter()
            .map(|opt| opt.unwrap_or(0.0) as i64)
            .collect());
    }

    Err(EconError::InvalidSpecification {
        message: format!("Column '{}' must be numeric type", col),
    })
}

fn extract_float_column(df: &polars::frame::DataFrame, col: &str) -> EconResult<Vec<f64>> {
    let column = df.column(col).map_err(|_| EconError::ColumnNotFound {
        column: col.to_string(),
        available: df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect(),
    })?;

    if let Ok(float_col) = column.f64() {
        return Ok(float_col
            .into_iter()
            .map(|opt| opt.unwrap_or(0.0))
            .collect());
    }

    if let Ok(int_col) = column.i64() {
        return Ok(int_col
            .into_iter()
            .map(|opt| opt.unwrap_or(0) as f64)
            .collect());
    }

    if let Ok(int_col) = column.i32() {
        return Ok(int_col
            .into_iter()
            .map(|opt| opt.unwrap_or(0) as f64)
            .collect());
    }

    Err(EconError::InvalidSpecification {
        message: format!("Column '{}' must be numeric type", col),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn create_staggered_panel() -> Dataset {
        // Create a panel with staggered treatment:
        // - Units A, B: treated starting period 3
        // - Units C, D: treated starting period 4
        // - Units E, F: never treated

        // True effect: ATT = 2.0 for early cohort, 3.0 for late cohort
        let df = df! {
            "unit" => ["A", "A", "A", "A", "A",
                       "B", "B", "B", "B", "B",
                       "C", "C", "C", "C", "C",
                       "D", "D", "D", "D", "D",
                       "E", "E", "E", "E", "E",
                       "F", "F", "F", "F", "F"],
            "time" => [1i64, 2, 3, 4, 5,
                       1, 2, 3, 4, 5,
                       1, 2, 3, 4, 5,
                       1, 2, 3, 4, 5,
                       1, 2, 3, 4, 5,
                       1, 2, 3, 4, 5],
            "treat" => [0.0, 0.0, 1.0, 1.0, 1.0,  // A: treated from t=3
                        0.0, 0.0, 1.0, 1.0, 1.0,  // B: treated from t=3
                        0.0, 0.0, 0.0, 1.0, 1.0,  // C: treated from t=4
                        0.0, 0.0, 0.0, 1.0, 1.0,  // D: treated from t=4
                        0.0, 0.0, 0.0, 0.0, 0.0,  // E: never treated
                        0.0, 0.0, 0.0, 0.0, 0.0], // F: never treated
            "first_treat" => [3i64, 3, 3, 3, 3,    // cohort 3
                              3, 3, 3, 3, 3,       // cohort 3
                              4, 4, 4, 4, 4,       // cohort 4
                              4, 4, 4, 4, 4,       // cohort 4
                              0, 0, 0, 0, 0,       // never (0)
                              0, 0, 0, 0, 0],      // never (0)
            "y" => [1.0, 1.5, 4.0, 4.5, 5.0,       // A: y jumps by ~2 at t=3
                    1.2, 1.4, 3.8, 4.3, 4.9,       // B: similar
                    2.0, 2.5, 2.8, 6.0, 6.5,       // C: y jumps by ~3 at t=4
                    2.2, 2.6, 2.9, 5.8, 6.3,       // D: similar
                    1.0, 1.5, 2.0, 2.5, 3.0,       // E: trend only
                    1.1, 1.6, 2.1, 2.6, 3.1]       // F: trend only
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_etwfe_basic() {
        let dataset = create_staggered_panel();
        let result = run_etwfe(
            &dataset,
            "y",
            "unit",
            "time",
            "treat",
            "first_treat",
            None,
            None,
        );

        assert!(
            result.is_ok(),
            "ETWFE should succeed, got {:?}",
            result.err()
        );

        let result = result.unwrap();

        assert_eq!(result.n_obs, 30);
        assert_eq!(result.n_cohorts, 2);
        assert!(result.n_treated > 0);
        assert!(result.n_control > 0);

        // Should have cohort-time effects
        assert!(!result.cohort_time_effects.is_empty());

        // Should have event study results
        assert!(!result.event_study.is_empty());

        // ATT should be positive (treatment has positive effect)
        // True effect is around 2-3
        assert!(
            result.att_simple > 0.0,
            "ATT should be positive, got {}",
            result.att_simple
        );
    }

    #[test]
    fn test_etwfe_event_study() {
        let dataset = create_staggered_panel();
        let result = run_etwfe(
            &dataset,
            "y",
            "unit",
            "time",
            "treat",
            "first_treat",
            None,
            None,
        )
        .unwrap();

        // Event study should have effects for different relative times
        assert!(!result.event_study.is_empty());

        // Pre-treatment effects (rel_time < 0) should be close to 0 if parallel trends hold
        for eff in &result.event_study {
            if eff.key < 0 {
                // Pre-treatment: effect should be relatively small
                assert!(
                    eff.estimate.abs() < 5.0,
                    "Pre-treatment effect should be small, got {} at rel_time={}",
                    eff.estimate,
                    eff.key
                );
            }
        }
    }

    #[test]
    fn test_etwfe_display() {
        let dataset = create_staggered_panel();
        let result = run_etwfe(
            &dataset,
            "y",
            "unit",
            "time",
            "treat",
            "first_treat",
            None,
            None,
        )
        .unwrap();

        let display = format!("{}", result);
        assert!(display.contains("ETWFE"));
        assert!(display.contains("ATT"));
        assert!(display.contains("Event Study"));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Validation Tests
    //
    // These tests validate the Extended TWFE (Wooldridge) estimator against
    // known DGP properties, particularly verifying that it correctly handles
    // treatment effect heterogeneity that would bias naive TWFE.
    //
    // References:
    // - Wooldridge, J.M. (2021). "Two-Way Fixed Effects, the Two-Way Mundlak
    //   Regression, and Difference-in-Differences Estimators." Working paper.
    // - Wooldridge, J.M. (2023). "Simple Approaches to Nonlinear
    //   Difference-in-Differences with Panel Data." The Econometrics Journal.
    // - R package `etwfe` (Grant McDermott)
    // ═══════════════════════════════════════════════════════════════════════

    /// Create a large staggered panel with heterogeneous treatment effects
    /// across cohorts for ETWFE validation.
    ///
    /// DGP:
    /// - 10 units per cohort, 3 cohorts + 10 never-treated = 40 units
    /// - 7 periods (1-7)
    /// - Cohort 3 (treated from t=3): true ATT = 2.0
    /// - Cohort 5 (treated from t=5): true ATT = 5.0
    /// - Never-treated: no effect
    /// - Unit FE: unit_id * 0.3
    /// - Time FE: period * 0.5
    /// - Noise: deterministic pseudo-random
    fn create_validation_etwfe_panel() -> (Dataset, std::collections::HashMap<i64, f64>) {
        let true_atts: std::collections::HashMap<i64, f64> =
            [(3, 2.0), (5, 5.0)].into_iter().collect();

        let mut unit_vec: Vec<i64> = Vec::new();
        let mut time_vec: Vec<i64> = Vec::new();
        let mut treat_vec: Vec<f64> = Vec::new();
        let mut first_treat_vec: Vec<i64> = Vec::new();
        let mut y_vec: Vec<f64> = Vec::new();

        for t in 1..=7i64 {
            for uid in 1..=40i64 {
                // Assign cohort
                let g: i64 = if uid <= 10 {
                    3 // Cohort 3
                } else if uid <= 20 {
                    5 // Cohort 5
                } else {
                    0 // Never treated
                };

                let is_treated = g > 0 && t >= g;
                let att = if is_treated {
                    *true_atts.get(&g).unwrap_or(&0.0)
                } else {
                    0.0
                };

                let unit_fe = uid as f64 * 0.3;
                let time_fe = t as f64 * 0.5;
                let noise = ((uid * t) % 11) as f64 * 0.03 - 0.165;

                unit_vec.push(uid);
                time_vec.push(t);
                treat_vec.push(if is_treated { 1.0 } else { 0.0 });
                first_treat_vec.push(g);
                y_vec.push(unit_fe + time_fe + att + noise);
            }
        }

        let df = df! {
            "unit" => unit_vec,
            "time" => time_vec,
            "treat" => treat_vec,
            "first_treat" => first_treat_vec,
            "y" => y_vec
        }
        .unwrap();

        (Dataset::new(df), true_atts)
    }

    #[test]
    fn test_validate_etwfe_recovers_cohort_specific_effects() {
        // Validate that ETWFE correctly recovers cohort-specific treatment effects.
        // This is the key advantage of ETWFE over naive TWFE: it allows
        // heterogeneous effects across cohorts without bias.
        let (dataset, true_atts) = create_validation_etwfe_panel();

        let result = run_etwfe(
            &dataset,
            "y",
            "unit",
            "time",
            "treat",
            "first_treat",
            None,
            None,
        )
        .unwrap();

        assert_eq!(result.n_cohorts, 2, "Should have 2 treatment cohorts");
        assert_eq!(result.n_periods, 7, "Should have 7 periods");

        // Check cohort average effects
        for avg in &result.cohort_avg {
            if let Some(&true_att) = true_atts.get(&avg.key) {
                assert!(
                    (avg.estimate - true_att).abs() < 1.0,
                    "Cohort {} avg effect should be ~{:.1}, got {:.4}",
                    avg.key,
                    true_att,
                    avg.estimate
                );
            }
        }
    }

    #[test]
    fn test_validate_etwfe_att_positive_with_positive_treatment() {
        // Validate that the overall ATT is positive when the DGP has
        // strictly positive treatment effects.
        let (dataset, _true_atts) = create_validation_etwfe_panel();

        let result = run_etwfe(
            &dataset,
            "y",
            "unit",
            "time",
            "treat",
            "first_treat",
            None,
            None,
        )
        .unwrap();

        assert!(
            result.att_simple > 0.0,
            "Simple ATT should be positive, got {:.4}",
            result.att_simple
        );
        assert!(
            result.att_weighted > 0.0,
            "Weighted ATT should be positive, got {:.4}",
            result.att_weighted
        );
    }

    #[test]
    fn test_validate_etwfe_event_study_pattern() {
        // Validate that the event study shows the expected pattern:
        // - Pre-treatment (rel_time < 0): effects near 0
        // - Post-treatment (rel_time >= 0): positive effects
        let (dataset, _true_atts) = create_validation_etwfe_panel();

        let result = run_etwfe(
            &dataset,
            "y",
            "unit",
            "time",
            "treat",
            "first_treat",
            None,
            None,
        )
        .unwrap();

        assert!(
            !result.event_study.is_empty(),
            "Event study should have entries"
        );

        // Post-treatment effects should be meaningfully positive
        let post_effects: Vec<&AggregatedEffect> =
            result.event_study.iter().filter(|e| e.key >= 0).collect();

        assert!(
            !post_effects.is_empty(),
            "Should have post-treatment event study effects"
        );

        for e in &post_effects {
            assert!(
                e.estimate > -1.0,
                "Post-treatment event study at e={} should not be strongly negative, got {:.4}",
                e.key,
                e.estimate
            );
        }
    }

    #[test]
    fn test_validate_etwfe_heterogeneous_vs_homogeneous() {
        // Validate that ETWFE produces unbiased estimates under treatment
        // effect heterogeneity. Compare results from a heterogeneous DGP
        // (cohort 3 -> ATT=2, cohort 5 -> ATT=5) against a homogeneous
        // DGP (all cohorts -> ATT=3.5).
        //
        // Both should produce overall ATTs in a reasonable range.

        // Heterogeneous case
        let (dataset_het, _) = create_validation_etwfe_panel();
        let result_het = run_etwfe(
            &dataset_het,
            "y",
            "unit",
            "time",
            "treat",
            "first_treat",
            None,
            None,
        )
        .unwrap();

        // Homogeneous case: same structure but ATT = 3.5 for all
        let mut unit_vec: Vec<i64> = Vec::new();
        let mut time_vec: Vec<i64> = Vec::new();
        let mut treat_vec: Vec<f64> = Vec::new();
        let mut first_treat_vec: Vec<i64> = Vec::new();
        let mut y_vec: Vec<f64> = Vec::new();

        let homogeneous_att = 3.5;

        for t in 1..=7i64 {
            for uid in 1..=40i64 {
                let g: i64 = if uid <= 10 {
                    3
                } else if uid <= 20 {
                    5
                } else {
                    0
                };
                let is_treated = g > 0 && t >= g;
                let att = if is_treated { homogeneous_att } else { 0.0 };
                let unit_fe = uid as f64 * 0.3;
                let time_fe = t as f64 * 0.5;
                let noise = ((uid * t) % 11) as f64 * 0.03 - 0.165;

                unit_vec.push(uid);
                time_vec.push(t);
                treat_vec.push(if is_treated { 1.0 } else { 0.0 });
                first_treat_vec.push(g);
                y_vec.push(unit_fe + time_fe + att + noise);
            }
        }

        let df_hom = df! {
            "unit" => unit_vec,
            "time" => time_vec,
            "treat" => treat_vec,
            "first_treat" => first_treat_vec,
            "y" => y_vec
        }
        .unwrap();
        let dataset_hom = Dataset::new(df_hom);

        let result_hom = run_etwfe(
            &dataset_hom,
            "y",
            "unit",
            "time",
            "treat",
            "first_treat",
            None,
            None,
        )
        .unwrap();

        // Homogeneous ATT should be close to 3.5
        assert!(
            (result_hom.att_simple - homogeneous_att).abs() < 1.5,
            "Homogeneous ATT should be ~{}, got {:.4}",
            homogeneous_att,
            result_hom.att_simple
        );

        // Heterogeneous ATT should be between the min and max true ATTs (2 and 5)
        assert!(
            result_het.att_simple > 0.5 && result_het.att_simple < 7.0,
            "Heterogeneous ATT should be in (0.5, 7.0), got {:.4}",
            result_het.att_simple
        );
    }

    #[test]
    fn test_validate_etwfe_structural_counts() {
        // Validate structural properties of the ETWFE result.
        let (dataset, _) = create_validation_etwfe_panel();

        let result = run_etwfe(
            &dataset,
            "y",
            "unit",
            "time",
            "treat",
            "first_treat",
            None,
            None,
        )
        .unwrap();

        // 40 units x 7 periods = 280 observations
        assert_eq!(result.n_obs, 280);
        assert!(result.n_treated > 0);
        assert!(result.n_control > 0);
        assert_eq!(result.n_treated + result.n_control, 280);

        // R-squared should be between 0 and 1
        assert!(
            result.r_squared >= 0.0 && result.r_squared <= 1.0,
            "R-squared should be in [0,1], got {:.4}",
            result.r_squared
        );

        // Standard errors should be positive and finite
        assert!(
            result.att_se > 0.0 && result.att_se.is_finite(),
            "ATT SE should be positive and finite, got {:.4}",
            result.att_se
        );

        // Cohort-time effects should have correct relative time computation
        for ct in &result.cohort_time_effects {
            assert_eq!(
                ct.rel_time,
                ct.time - ct.cohort,
                "rel_time should equal time - cohort"
            );
        }
    }
}
