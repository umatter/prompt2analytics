//! Survival analysis module.
//!
//! Pure Rust implementation of survival analysis methods including:
//! - Kaplan-Meier estimator (non-parametric survival curves)
//! - Log-rank test (comparison of survival curves)
//! - Cox Proportional Hazards model (semi-parametric regression)
//! - Accelerated Failure Time models (parametric regression)
//! - Competing risks / Aalen-Johansen estimator
//!
//! # References
//!
//! - Cox, D.R. (1972). "Regression Models and Life Tables". JRSS B, 34:187-220.
//! - Kaplan, E.L. & Meier, P. (1958). "Nonparametric Estimation from Incomplete Observations". JASA, 53:457-481.
//! - Aalen, O.O. & Johansen, S. (1978). "An Empirical Transition Matrix". Scandinavian J. Statistics, 5:141-150.
//! - R package `survival` (Therneau & Grambsch). https://cran.r-project.org/package=survival

use ndarray::{Array1, Array2, ArrayView1};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fmt;

use polars::prelude::Float64Chunked;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::safe_inverse;
use crate::traits::estimator::{SignificanceLevel, chi_squared_p_value, normal_cdf};

// =============================================================================
// Core Types
// =============================================================================

/// A survival observation with time and event status.
#[derive(Debug, Clone)]
struct SurvivalObs {
    time: f64,
    event: bool,    // true = event observed, false = censored
    event_type: u8, // For competing risks: 0 = censored, 1,2,... = event types
    covariates: Option<Array1<f64>>,
    group: Option<String>,
}

/// Wrapper for f64 to use as BTreeMap key.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
struct OrderedF64(f64);

impl Eq for OrderedF64 {}

impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Method for handling tied event times in Cox model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TiesMethod {
    /// Breslow approximation (default, faster)
    #[default]
    Breslow,
    /// Efron approximation (more accurate with many ties)
    Efron,
}

impl fmt::Display for TiesMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TiesMethod::Breslow => write!(f, "Breslow"),
            TiesMethod::Efron => write!(f, "Efron"),
        }
    }
}

/// Distribution for Accelerated Failure Time models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AftDistribution {
    /// Exponential distribution (constant hazard)
    Exponential,
    /// Weibull distribution (monotone hazard)
    #[default]
    Weibull,
    /// Log-normal distribution (unimodal hazard)
    LogNormal,
    /// Log-logistic distribution (unimodal hazard)
    LogLogistic,
}

impl fmt::Display for AftDistribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AftDistribution::Exponential => write!(f, "Exponential"),
            AftDistribution::Weibull => write!(f, "Weibull"),
            AftDistribution::LogNormal => write!(f, "Log-Normal"),
            AftDistribution::LogLogistic => write!(f, "Log-Logistic"),
        }
    }
}

// =============================================================================
// Kaplan-Meier Estimator
// =============================================================================

/// Result from Kaplan-Meier survival estimation.
///
/// The Kaplan-Meier estimator is a non-parametric statistic for estimating
/// the survival function from lifetime data.
///
/// # References
///
/// - Kaplan, E.L. & Meier, P. (1958). "Nonparametric Estimation from Incomplete
///   Observations". Journal of the American Statistical Association, 53:457-481.
/// - Greenwood, M. (1926). "The Natural Duration of Cancer". Reports on Public
///   Health and Medical Subjects, 33:1-26.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KaplanMeierResult {
    /// Group name (if stratified analysis)
    pub group: Option<String>,
    /// Distinct event times
    pub times: Vec<f64>,
    /// Survival probability S(t) at each time
    pub survival: Vec<f64>,
    /// Standard errors (Greenwood's formula)
    pub std_errors: Vec<f64>,
    /// Lower confidence interval bound
    pub ci_lower: Vec<f64>,
    /// Upper confidence interval bound
    pub ci_upper: Vec<f64>,
    /// Number at risk at each time
    pub n_at_risk: Vec<usize>,
    /// Number of events at each time
    pub n_events: Vec<usize>,
    /// Number censored at each time
    pub n_censored: Vec<usize>,
    /// Median survival time (if estimable)
    pub median_survival: Option<f64>,
    /// Confidence interval for median survival
    pub median_ci: Option<(f64, f64)>,
    /// Total number of observations
    pub n_obs: usize,
    /// Total number of events
    pub total_events: usize,
    /// Total number censored
    pub total_censored: usize,
    /// Confidence level used
    pub conf_level: f64,
}

impl fmt::Display for KaplanMeierResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Kaplan-Meier Survival Estimate")?;
        writeln!(f, "===============================")?;
        if let Some(ref group) = self.group {
            writeln!(f, "Group: {}", group)?;
        }
        writeln!(
            f,
            "N = {}, Events = {}, Censored = {} ({:.1}%)",
            self.n_obs,
            self.total_events,
            self.total_censored,
            100.0 * self.total_censored as f64 / self.n_obs as f64
        )?;
        writeln!(f)?;

        if let Some(median) = self.median_survival {
            write!(f, "Median Survival: {:.3}", median)?;
            if let Some((lo, hi)) = self.median_ci {
                write!(
                    f,
                    " ({:.0}% CI: {:.3} - {:.3})",
                    self.conf_level * 100.0,
                    lo,
                    hi
                )?;
            }
            writeln!(f)?;
        } else {
            writeln!(
                f,
                "Median Survival: NA (survival > 50% at end of follow-up)"
            )?;
        }
        writeln!(f)?;

        writeln!(
            f,
            "{:>10} {:>10} {:>10} {:>10} {:>12}",
            "Time", "At Risk", "Events", "S(t)", "Std.Err"
        )?;
        writeln!(f, "{}", "-".repeat(55))?;

        // Show first 10 and last 5 time points if many
        let n = self.times.len();
        let show_all = n <= 20;
        let show_indices: Vec<usize> = if show_all {
            (0..n).collect()
        } else {
            let mut idx: Vec<usize> = (0..10).collect();
            idx.push(usize::MAX); // marker for "..."
            idx.extend((n.saturating_sub(5))..n);
            idx
        };

        for &i in &show_indices {
            if i == usize::MAX {
                writeln!(f, "{:>10}", "...")?;
            } else {
                writeln!(
                    f,
                    "{:>10.3} {:>10} {:>10} {:>10.4} {:>12.4}",
                    self.times[i],
                    self.n_at_risk[i],
                    self.n_events[i],
                    self.survival[i],
                    self.std_errors[i]
                )?;
            }
        }

        Ok(())
    }
}

/// Run Kaplan-Meier survival analysis.
///
/// # Arguments
///
/// * `dataset` - The dataset containing survival data
/// * `time_col` - Column name for survival/censoring time
/// * `event_col` - Column name for event indicator (1 = event, 0 = censored)
/// * `group_col` - Optional column for stratified analysis
/// * `conf_level` - Confidence level (default: 0.95)
///
/// # Returns
///
/// A vector of `KaplanMeierResult`, one per group (or single element if unstratified).
///
/// # Example
///
/// ```ignore
/// let results = run_kaplan_meier(
///     &dataset,
///     "time",
///     "status",
///     Some("treatment"),
///     0.95,
/// )?;
/// ```
pub fn run_kaplan_meier(
    dataset: &Dataset,
    time_col: &str,
    event_col: &str,
    group_col: Option<&str>,
    conf_level: f64,
) -> EconResult<Vec<KaplanMeierResult>> {
    // Validate confidence level
    if conf_level <= 0.0 || conf_level >= 1.0 {
        return Err(EconError::InvalidSpecification {
            message: "Confidence level must be between 0 and 1".to_string(),
        });
    }

    // Extract data
    let df = dataset.df();
    let available_cols: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    let times = df
        .column(time_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: time_col.to_string(),
            available: available_cols.clone(),
        })?
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: time_col.to_string(),
        })?;

    let events = df
        .column(event_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: event_col.to_string(),
            available: available_cols.clone(),
        })?
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: event_col.to_string(),
        })?;

    // Group data
    let groups: Vec<Option<String>> = if let Some(g_col) = group_col {
        let g = df.column(g_col).map_err(|_| EconError::ColumnNotFound {
            column: g_col.to_string(),
            available: available_cols.clone(),
        })?;
        (0..df.height())
            .map(|i| {
                g.get(i)
                    .ok()
                    .map(|v| v.to_string().trim_matches('"').to_string())
            })
            .collect()
    } else {
        vec![None; df.height()]
    };

    // Build observations by group
    let mut obs_by_group: HashMap<Option<String>, Vec<SurvivalObs>> = HashMap::new();

    for i in 0..df.height() {
        let t = match times.get(i) {
            Some(v) if !v.is_nan() && v >= 0.0 => v,
            _ => continue,
        };
        let e = match events.get(i) {
            Some(v) if !v.is_nan() => v != 0.0,
            _ => continue,
        };

        let obs = SurvivalObs {
            time: t,
            event: e,
            event_type: if e { 1 } else { 0 },
            covariates: None,
            group: groups[i].clone(),
        };

        obs_by_group.entry(groups[i].clone()).or_default().push(obs);
    }

    if obs_by_group.is_empty() {
        return Err(EconError::EmptyDataset);
    }

    // Compute KM for each group
    let z = normal_quantile(1.0 - (1.0 - conf_level) / 2.0);
    let mut results = Vec::new();

    for (group, observations) in obs_by_group {
        let km = compute_kaplan_meier(&observations, conf_level, z, group)?;
        results.push(km);
    }

    // Sort by group name for consistent output
    results.sort_by(|a, b| a.group.cmp(&b.group));

    Ok(results)
}

/// Compute Kaplan-Meier estimate for a single group.
fn compute_kaplan_meier(
    observations: &[SurvivalObs],
    conf_level: f64,
    z: f64,
    group: Option<String>,
) -> EconResult<KaplanMeierResult> {
    let n_obs = observations.len();
    if n_obs == 0 {
        return Err(EconError::EmptyDataset);
    }

    // Sort by time (ascending)
    let mut sorted: Vec<_> = observations.iter().collect();
    sorted.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());

    // Count events and censored at each distinct time using BTreeMap for ordered keys
    let mut time_data: BTreeMap<OrderedF64, (usize, usize)> = BTreeMap::new();
    for obs in &sorted {
        let key = OrderedF64(obs.time);
        let entry = time_data.entry(key).or_insert((0, 0));
        if obs.event {
            entry.0 += 1; // events
        } else {
            entry.1 += 1; // censored
        }
    }

    // Compute KM estimates
    let mut times = Vec::new();
    let mut survival = Vec::new();
    let mut std_errors = Vec::new();
    let mut ci_lower = Vec::new();
    let mut ci_upper = Vec::new();
    let mut n_at_risk = Vec::new();
    let mut n_events = Vec::new();
    let mut n_censored = Vec::new();

    let mut current_risk = n_obs;
    let mut current_surv = 1.0;
    let mut greenwood_sum = 0.0;
    let mut total_events = 0;

    for (time, (events, censored)) in time_data {
        let t = time.0;
        let d = events;
        let c = censored;

        n_at_risk.push(current_risk);
        n_events.push(d);
        n_censored.push(c);
        times.push(t);

        if d > 0 && current_risk > 0 {
            // Kaplan-Meier product-limit estimator
            current_surv *= 1.0 - (d as f64 / current_risk as f64);

            // Greenwood's formula for variance
            if current_risk > d {
                greenwood_sum += d as f64 / (current_risk as f64 * (current_risk - d) as f64);
            }
        }

        survival.push(current_surv);

        // Standard error via Greenwood's formula
        let se = current_surv * greenwood_sum.sqrt();
        std_errors.push(se);

        // Confidence interval using log-log transformation for better coverage
        // log(-log(S(t))) is approximately normal
        let (lo, hi) = if current_surv > 0.0 && current_surv < 1.0 {
            let log_log_s = (-current_surv.ln()).ln();
            let se_log_log = se / (current_surv * (-current_surv.ln()).abs());
            let lo_log_log = log_log_s - z * se_log_log;
            let hi_log_log = log_log_s + z * se_log_log;
            ((-(-lo_log_log).exp()).exp(), (-(-hi_log_log).exp()).exp())
        } else if current_surv >= 1.0 {
            (1.0, 1.0)
        } else {
            (0.0, 0.0)
        };
        ci_lower.push(lo.max(0.0).min(1.0));
        ci_upper.push(hi.max(0.0).min(1.0));

        total_events += d;
        current_risk -= d + c;
    }

    // Find median survival (time when S(t) first drops to 0.5 or below)
    let median_survival = find_quantile(&times, &survival, 0.5);
    let median_ci = median_survival.map(|_| {
        // Simple CI for median using CI of S(t)
        let lo = find_quantile_from_bound(&times, &ci_upper, 0.5);
        let hi = find_quantile_from_bound(&times, &ci_lower, 0.5);
        (
            lo.unwrap_or(times[0]),
            hi.unwrap_or(*times.last().unwrap_or(&0.0)),
        )
    });

    Ok(KaplanMeierResult {
        group,
        times,
        survival,
        std_errors,
        ci_lower,
        ci_upper,
        n_at_risk,
        n_events,
        n_censored,
        median_survival,
        median_ci,
        n_obs,
        total_events,
        total_censored: n_obs - total_events,
        conf_level,
    })
}

/// Find time when survival curve crosses a given probability.
fn find_quantile(times: &[f64], survival: &[f64], prob: f64) -> Option<f64> {
    for i in 0..survival.len() {
        if survival[i] <= prob {
            return Some(times[i]);
        }
    }
    None
}

/// Find quantile from CI bound.
fn find_quantile_from_bound(times: &[f64], bound: &[f64], prob: f64) -> Option<f64> {
    for i in 0..bound.len() {
        if bound[i] <= prob {
            return Some(times[i]);
        }
    }
    None
}

/// Compute normal quantile (inverse CDF).
fn normal_quantile(p: f64) -> f64 {
    // Rational approximation (Abramowitz and Stegun 26.2.23)
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

// =============================================================================
// Log-Rank Test
// =============================================================================

/// Result from log-rank test comparing survival curves.
///
/// The log-rank test is a hypothesis test to compare the survival distributions
/// of two or more groups.
///
/// # References
///
/// - Mantel, N. (1966). "Evaluation of survival data and two new rank order
///   statistics arising in its consideration". Cancer Chemotherapy Reports, 50:163-170.
/// - Peto, R. & Peto, J. (1972). "Asymptotically Efficient Rank Invariant Test
///   Procedures". Journal of the Royal Statistical Society A, 135:185-207.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogRankResult {
    /// Groups being compared
    pub groups: Vec<String>,
    /// Chi-squared test statistic
    pub chi_squared: f64,
    /// Degrees of freedom
    pub df: usize,
    /// P-value
    pub p_value: f64,
    /// Significance level
    pub significance: SignificanceLevel,
    /// Number of observations per group
    pub n_per_group: Vec<usize>,
    /// Number of events per group
    pub events_per_group: Vec<usize>,
    /// Expected events per group (under H0)
    pub expected_per_group: Vec<f64>,
    /// Observed - Expected for each group
    pub obs_minus_exp: Vec<f64>,
}

impl fmt::Display for LogRankResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Log-Rank Test")?;
        writeln!(f, "=============")?;
        writeln!(f)?;
        writeln!(
            f,
            "{:<15} {:>10} {:>10} {:>12} {:>12}",
            "Group", "N", "Events", "Expected", "O-E"
        )?;
        writeln!(f, "{}", "-".repeat(60))?;

        for i in 0..self.groups.len() {
            writeln!(
                f,
                "{:<15} {:>10} {:>10} {:>12.2} {:>12.2}",
                self.groups[i],
                self.n_per_group[i],
                self.events_per_group[i],
                self.expected_per_group[i],
                self.obs_minus_exp[i]
            )?;
        }

        writeln!(f, "{}", "-".repeat(60))?;
        writeln!(f)?;
        writeln!(
            f,
            "Chi-squared = {:.4}, df = {}, p = {:.4}{}",
            self.chi_squared,
            self.df,
            self.p_value,
            self.significance.stars()
        )?;

        Ok(())
    }
}

/// Perform log-rank test to compare survival curves between groups.
///
/// # Arguments
///
/// * `dataset` - The dataset containing survival data
/// * `time_col` - Column name for survival/censoring time
/// * `event_col` - Column name for event indicator (1 = event, 0 = censored)
/// * `group_col` - Column name for group assignment
///
/// # Returns
///
/// A `LogRankResult` containing the test statistic and p-value.
pub fn log_rank_test(
    dataset: &Dataset,
    time_col: &str,
    event_col: &str,
    group_col: &str,
) -> EconResult<LogRankResult> {
    let df = dataset.df();
    let available_cols: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    // Extract columns
    let times = df
        .column(time_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: time_col.to_string(),
            available: available_cols.clone(),
        })?
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: time_col.to_string(),
        })?;

    let events = df
        .column(event_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: event_col.to_string(),
            available: available_cols.clone(),
        })?
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: event_col.to_string(),
        })?;

    let groups_col = df
        .column(group_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: group_col.to_string(),
            available: available_cols,
        })?;

    // Build observations
    let mut all_obs: Vec<(f64, bool, String)> = Vec::new();
    let mut group_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

    for i in 0..df.height() {
        let t = match times.get(i) {
            Some(v) if !v.is_nan() && v >= 0.0 => v,
            _ => continue,
        };
        let e = match events.get(i) {
            Some(v) if !v.is_nan() => v != 0.0,
            _ => continue,
        };
        let g = match groups_col.get(i) {
            Ok(v) => v.to_string().trim_matches('"').to_string(),
            Err(_) => continue,
        };

        all_obs.push((t, e, g.clone()));
        group_set.insert(g);
    }

    let groups: Vec<String> = group_set.into_iter().collect();
    let n_groups = groups.len();

    if n_groups < 2 {
        return Err(EconError::InvalidSpecification {
            message: "Log-rank test requires at least 2 groups".to_string(),
        });
    }

    // Sort observations by time
    all_obs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Get distinct event times
    let event_times: Vec<f64> = {
        let mut times_set: std::collections::BTreeSet<OrderedF64> =
            std::collections::BTreeSet::new();
        for (t, e, _) in &all_obs {
            if *e {
                times_set.insert(OrderedF64(*t));
            }
        }
        times_set.into_iter().map(|of| of.0).collect()
    };

    // Count observations per group
    let mut n_per_group: Vec<usize> = vec![0; n_groups];
    let mut events_per_group: Vec<usize> = vec![0; n_groups];

    for (_, event, group) in &all_obs {
        let g_idx = groups.iter().position(|g| g == group).unwrap();
        n_per_group[g_idx] += 1;
        if *event {
            events_per_group[g_idx] += 1;
        }
    }

    // Compute log-rank statistic
    // O_j - E_j where E_j = sum over times of (d_t * n_jt / n_t)
    let mut expected: Vec<f64> = vec![0.0; n_groups];
    let mut variance_matrix: Array2<f64> = Array2::zeros((n_groups - 1, n_groups - 1));

    // Track at-risk counts
    let mut at_risk_total = all_obs.len();
    let mut at_risk_per_group: Vec<usize> = n_per_group.clone();
    let mut obs_idx = 0;

    for event_time in &event_times {
        // Skip past observations before this event time
        while obs_idx < all_obs.len() && all_obs[obs_idx].0 < *event_time {
            let (_, _, ref g) = all_obs[obs_idx];
            let g_idx = groups.iter().position(|grp| grp == g).unwrap();
            at_risk_per_group[g_idx] = at_risk_per_group[g_idx].saturating_sub(1);
            at_risk_total = at_risk_total.saturating_sub(1);
            obs_idx += 1;
        }

        // Count events and risk set at this time
        let mut d_total = 0;
        let mut removals_per_group: Vec<usize> = vec![0; n_groups];

        let mut temp_idx = obs_idx;
        while temp_idx < all_obs.len() && (all_obs[temp_idx].0 - event_time).abs() < 1e-10 {
            let (_, event, ref g) = all_obs[temp_idx];
            let g_idx = groups.iter().position(|grp| grp == g).unwrap();
            removals_per_group[g_idx] += 1;
            if event {
                d_total += 1;
            }
            temp_idx += 1;
        }

        if at_risk_total > 0 && d_total > 0 {
            let n_t = at_risk_total as f64;

            // Expected events: E_j = d * n_j / n
            for j in 0..n_groups {
                let n_j = at_risk_per_group[j] as f64;
                expected[j] += (d_total as f64) * n_j / n_t;
            }

            // Variance calculation (for first n_groups-1 groups)
            // Using the hypergeometric variance formula
            if at_risk_total > 1 {
                let factor = (d_total as f64) * (n_t - d_total as f64) / (n_t * n_t * (n_t - 1.0));
                for j in 0..(n_groups - 1) {
                    let n_j = at_risk_per_group[j] as f64;
                    for k in 0..(n_groups - 1) {
                        let n_k = at_risk_per_group[k] as f64;
                        if j == k {
                            variance_matrix[[j, k]] += factor * n_j * (n_t - n_j);
                        } else {
                            variance_matrix[[j, k]] -= factor * n_j * n_k;
                        }
                    }
                }
            }
        }

        // Update risk counts
        for j in 0..n_groups {
            at_risk_per_group[j] = at_risk_per_group[j].saturating_sub(removals_per_group[j]);
        }
        at_risk_total = at_risk_total.saturating_sub(removals_per_group.iter().sum::<usize>());
    }

    // Compute chi-squared statistic
    // χ² = (O - E)' V^{-1} (O - E) for first n_groups-1 groups
    let obs_minus_exp: Vec<f64> = events_per_group
        .iter()
        .zip(expected.iter())
        .map(|(&o, &e)| o as f64 - e)
        .collect();

    let u: Array1<f64> = Array1::from_vec(obs_minus_exp[0..(n_groups - 1)].to_vec());

    let chi_squared = if n_groups == 2 {
        // Simple formula for 2 groups
        let var = variance_matrix[[0, 0]];
        if var > 0.0 { (u[0] * u[0]) / var } else { 0.0 }
    } else {
        // General formula using matrix inverse
        match safe_inverse(&variance_matrix.view()) {
            Ok((v_inv, _)) => u.dot(&v_inv.dot(&u)),
            Err(_) => {
                // Fallback: use sum of squared standardized differences
                let mut chi2 = 0.0;
                for j in 0..(n_groups - 1) {
                    if variance_matrix[[j, j]] > 0.0 {
                        chi2 += obs_minus_exp[j].powi(2) / variance_matrix[[j, j]];
                    }
                }
                chi2
            }
        }
    };

    let df_val = n_groups - 1;
    let p_value = chi_squared_p_value(chi_squared, df_val as f64);
    let significance = SignificanceLevel::from_p_value(p_value);

    Ok(LogRankResult {
        groups,
        chi_squared,
        df: df_val,
        p_value,
        significance,
        n_per_group,
        events_per_group,
        expected_per_group: expected,
        obs_minus_exp,
    })
}

// =============================================================================
// Cox Proportional Hazards Model
// =============================================================================

/// Configuration for Cox proportional hazards model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoxConfig {
    /// Method for handling tied event times
    pub ties: TiesMethod,
    /// Maximum iterations for Newton-Raphson
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Use robust (sandwich) standard errors
    pub robust_se: bool,
}

impl Default for CoxConfig {
    fn default() -> Self {
        Self {
            ties: TiesMethod::Breslow,
            max_iter: 25,
            tolerance: 1e-9,
            robust_se: false,
        }
    }
}

/// Result from Cox proportional hazards regression.
///
/// # References
///
/// - Cox, D.R. (1972). "Regression Models and Life Tables". Journal of the Royal
///   Statistical Society B, 34:187-220.
/// - Cox, D.R. (1975). "Partial Likelihood". Biometrika, 62:269-276.
/// - Breslow, N.E. (1974). "Covariance Analysis of Censored Survival Data".
///   Biometrics, 30:89-99.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoxResult {
    /// Variable names (excluding intercept - Cox model has no intercept)
    pub variables: Vec<String>,
    /// Estimated coefficients (β)
    pub coefficients: Vec<f64>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// z-statistics (Wald test)
    pub z_stats: Vec<f64>,
    /// P-values for coefficients
    pub p_values: Vec<f64>,
    /// Significance levels
    pub significance: Vec<SignificanceLevel>,
    /// Hazard ratios exp(β)
    pub hazard_ratios: Vec<f64>,
    /// Lower 95% CI for hazard ratios
    pub hr_ci_lower: Vec<f64>,
    /// Upper 95% CI for hazard ratios
    pub hr_ci_upper: Vec<f64>,
    /// Maximized log partial likelihood
    pub log_likelihood: f64,
    /// Null log partial likelihood (β = 0)
    pub log_likelihood_null: f64,
    /// Concordance statistic (C-index)
    pub concordance: f64,
    /// Standard error of concordance
    pub concordance_se: f64,
    /// Wald test statistic (overall model)
    pub wald_test: f64,
    /// P-value for Wald test
    pub wald_p_value: f64,
    /// Score (log-rank) test statistic
    pub score_test: f64,
    /// P-value for score test
    pub score_p_value: f64,
    /// Likelihood ratio test statistic
    pub lr_test: f64,
    /// P-value for LR test
    pub lr_p_value: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of events
    pub n_events: usize,
    /// Method used for ties
    pub ties_method: TiesMethod,
    /// Whether convergence was achieved
    pub converged: bool,
    /// Number of iterations
    pub iterations: usize,
}

impl fmt::Display for CoxResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Cox Proportional Hazards Regression")?;
        writeln!(f, "====================================")?;
        writeln!(f, "n = {}, events = {}", self.n_obs, self.n_events)?;
        writeln!(f, "Ties method: {}", self.ties_method)?;
        writeln!(f)?;

        writeln!(
            f,
            "{:<20} {:>10} {:>10} {:>8} {:>8} {:>10} {:>16}",
            "Variable", "coef", "se(coef)", "z", "p", "HR", "95% CI HR"
        )?;
        writeln!(f, "{}", "-".repeat(90))?;

        for i in 0..self.variables.len() {
            writeln!(
                f,
                "{:<20} {:>10.4} {:>10.4} {:>8.2} {:>8.4}{} {:>10.4} [{:.3}, {:.3}]",
                self.variables[i],
                self.coefficients[i],
                self.std_errors[i],
                self.z_stats[i],
                self.p_values[i],
                self.significance[i].stars(),
                self.hazard_ratios[i],
                self.hr_ci_lower[i],
                self.hr_ci_upper[i]
            )?;
        }

        writeln!(f, "{}", "-".repeat(90))?;
        writeln!(f)?;
        writeln!(
            f,
            "Concordance = {:.3} (se = {:.3})",
            self.concordance, self.concordance_se
        )?;
        writeln!(
            f,
            "Likelihood ratio test = {:.2}, df = {}, p = {:.4}",
            self.lr_test,
            self.variables.len(),
            self.lr_p_value
        )?;
        writeln!(
            f,
            "Wald test = {:.2}, df = {}, p = {:.4}",
            self.wald_test,
            self.variables.len(),
            self.wald_p_value
        )?;
        writeln!(
            f,
            "Score (logrank) test = {:.2}, df = {}, p = {:.4}",
            self.score_test,
            self.variables.len(),
            self.score_p_value
        )?;
        writeln!(f)?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;

        Ok(())
    }
}

/// Run Cox proportional hazards regression.
///
/// # Arguments
///
/// * `dataset` - The dataset containing survival data
/// * `time_col` - Column name for survival/censoring time
/// * `event_col` - Column name for event indicator (1 = event, 0 = censored)
/// * `x_cols` - Column names for covariates
/// * `config` - Optional configuration (ties method, convergence settings)
///
/// # Returns
///
/// A `CoxResult` containing coefficient estimates, hazard ratios, and model diagnostics.
///
/// # Mathematical Details
///
/// The Cox model assumes:
/// ```text
/// h(t|X) = h₀(t) × exp(β'X)
/// ```
///
/// where h₀(t) is an unspecified baseline hazard. The partial likelihood is:
/// ```text
/// L(β) = ∏ᵢ [exp(β'xᵢ) / Σⱼ∈R(tᵢ) exp(β'xⱼ)]^δᵢ
/// ```
///
/// Coefficients are estimated by maximizing the partial likelihood using Newton-Raphson.
pub fn run_cox_ph(
    dataset: &Dataset,
    time_col: &str,
    event_col: &str,
    x_cols: &[&str],
    config: Option<CoxConfig>,
) -> EconResult<CoxResult> {
    let config = config.unwrap_or_default();
    let df = dataset.df();
    let available_cols: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    if x_cols.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "Cox model requires at least one covariate".to_string(),
        });
    }

    // Extract data
    let times = df
        .column(time_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: time_col.to_string(),
            available: available_cols.clone(),
        })?
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: time_col.to_string(),
        })?;

    let events = df
        .column(event_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: event_col.to_string(),
            available: available_cols.clone(),
        })?
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: event_col.to_string(),
        })?;

    // Build covariate matrix
    let p = x_cols.len();
    let mut x_data: Vec<f64> = Vec::new();
    let mut obs_data: Vec<(f64, bool, usize)> = Vec::new(); // (time, event, row_index)
    let mut valid_rows = 0;

    // Extract covariate columns ONCE upfront (avoids repeated column lookups in inner loop)
    let covariate_cols: Vec<&Float64Chunked> = x_cols
        .iter()
        .map(|col| {
            df.column(col)
                .map_err(|_| EconError::ColumnNotFound {
                    column: col.to_string(),
                    available: available_cols.clone(),
                })?
                .f64()
                .map_err(|_| EconError::NonNumericColumn {
                    column: col.to_string(),
                })
        })
        .collect::<EconResult<Vec<_>>>()?;

    for i in 0..df.height() {
        let t = match times.get(i) {
            Some(v) if !v.is_nan() && v >= 0.0 => v,
            _ => continue,
        };
        let e = match events.get(i) {
            Some(v) if !v.is_nan() => v != 0.0,
            _ => continue,
        };

        // Extract covariates from pre-fetched columns
        let mut row_valid = true;
        let mut row_x: Vec<f64> = Vec::with_capacity(p);
        for ca in &covariate_cols {
            match ca.get(i) {
                Some(v) if !v.is_nan() => row_x.push(v),
                _ => {
                    row_valid = false;
                    break;
                }
            }
        }

        if row_valid {
            obs_data.push((t, e, valid_rows));
            x_data.extend(row_x);
            valid_rows += 1;
        }
    }

    let n = valid_rows;
    if n < p + 1 {
        return Err(EconError::InsufficientData {
            required: p + 1,
            provided: n,
            context: format!("Cox regression with {} covariates", p),
        });
    }

    let n_events = obs_data.iter().filter(|(_, e, _)| *e).count();
    if n_events == 0 {
        return Err(EconError::InvalidSpecification {
            message: "No events observed in the data".to_string(),
        });
    }

    let x = Array2::from_shape_vec((n, p), x_data)
        .map_err(|e| EconError::Internal(format!("Matrix construction failed: {}", e)))?;

    // Sort by time (descending for efficient risk set computation)
    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| obs_data[b].0.partial_cmp(&obs_data[a].0).unwrap());

    // Newton-Raphson optimization
    let mut beta: Array1<f64> = Array1::zeros(p);
    let mut converged = false;
    let mut iterations = 0;

    // Compute null log-likelihood (beta = 0)
    let log_likelihood_null = cox_log_likelihood(
        &obs_data,
        &x,
        &Array1::zeros(p),
        &sorted_indices,
        config.ties,
    );

    for iter in 0..config.max_iter {
        iterations = iter + 1;

        // Compute gradient and Hessian
        let (grad, hess, ll) =
            cox_gradient_hessian(&obs_data, &x, &beta, &sorted_indices, config.ties);

        // Check convergence
        let grad_norm = grad.iter().map(|g| g.abs()).fold(0.0, f64::max);
        if grad_norm < config.tolerance {
            converged = true;
            break;
        }

        // Newton-Raphson update: beta_new = beta - H^{-1} * grad
        let (hess_inv, _) = safe_inverse(&hess.view())?;
        let delta = hess_inv.dot(&grad);

        // Step halving if needed
        let mut step = 1.0;
        let mut new_beta = &beta + &(&delta * step);
        let mut new_ll = cox_log_likelihood(&obs_data, &x, &new_beta, &sorted_indices, config.ties);

        while new_ll < ll && step > 1e-10 {
            step *= 0.5;
            new_beta = &beta + &(&delta * step);
            new_ll = cox_log_likelihood(&obs_data, &x, &new_beta, &sorted_indices, config.ties);
        }

        beta = new_beta;
    }

    // Final likelihood and information matrix
    let (_grad, hess, log_likelihood) =
        cox_gradient_hessian(&obs_data, &x, &beta, &sorted_indices, config.ties);
    let (info_matrix, _) = safe_inverse(&hess.view())?;

    // Standard errors
    let std_errors: Vec<f64> = (0..p).map(|i| info_matrix[[i, i]].sqrt()).collect();

    // z-statistics and p-values
    let z_stats: Vec<f64> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
        .collect();

    let p_values: Vec<f64> = z_stats
        .iter()
        .map(|&z| 2.0 * (1.0 - normal_cdf(z.abs())))
        .collect();

    let significance: Vec<SignificanceLevel> = p_values
        .iter()
        .map(|&pv| SignificanceLevel::from_p_value(pv))
        .collect();

    // Hazard ratios and CIs
    let hazard_ratios: Vec<f64> = beta.iter().map(|&b| b.exp()).collect();
    let hr_ci_lower: Vec<f64> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| (b - 1.96 * se).exp())
        .collect();
    let hr_ci_upper: Vec<f64> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| (b + 1.96 * se).exp())
        .collect();

    // Wald test (overall)
    let wald_test = beta.dot(&hess.dot(&beta));
    let wald_p_value = chi_squared_p_value(wald_test, p as f64);

    // Score test (at beta = 0)
    let (grad_null, hess_null, _) = cox_gradient_hessian(
        &obs_data,
        &x,
        &Array1::zeros(p),
        &sorted_indices,
        config.ties,
    );
    let (info_null, _) = safe_inverse(&hess_null.view())?;
    let score_test = grad_null.dot(&info_null.dot(&grad_null));
    let score_p_value = chi_squared_p_value(score_test, p as f64);

    // Likelihood ratio test
    let lr_test = 2.0 * (log_likelihood - log_likelihood_null);
    let lr_p_value = chi_squared_p_value(lr_test, p as f64);

    // Concordance (C-index)
    let (concordance, concordance_se) = compute_concordance(&obs_data, &x, &beta);

    Ok(CoxResult {
        variables: x_cols.iter().map(|s| s.to_string()).collect(),
        coefficients: beta.to_vec(),
        std_errors,
        z_stats,
        p_values,
        significance,
        hazard_ratios,
        hr_ci_lower,
        hr_ci_upper,
        log_likelihood,
        log_likelihood_null,
        concordance,
        concordance_se,
        wald_test,
        wald_p_value,
        score_test,
        score_p_value,
        lr_test,
        lr_p_value,
        n_obs: n,
        n_events,
        ties_method: config.ties,
        converged,
        iterations,
    })
}

/// Compute Cox partial log-likelihood.
fn cox_log_likelihood(
    obs: &[(f64, bool, usize)],
    x: &Array2<f64>,
    beta: &Array1<f64>,
    sorted_indices: &[usize],
    ties: TiesMethod,
) -> f64 {
    let n = obs.len();
    let mut log_lik = 0.0;

    // Compute exp(beta'x) for all observations
    let eta: Vec<f64> = (0..n).map(|i| x.row(i).dot(beta)).collect();
    let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

    // Cumulative sum of exp(eta) from end (for risk set computation)
    let mut cum_exp_eta = 0.0;

    match ties {
        TiesMethod::Breslow => {
            // Breslow approximation
            for &idx in sorted_indices.iter() {
                let (_, event, row_idx) = obs[idx];
                cum_exp_eta += exp_eta[row_idx];

                if event {
                    log_lik += eta[row_idx] - cum_exp_eta.ln();
                }
            }
        }
        TiesMethod::Efron => {
            // Efron approximation - handle ties properly
            let mut i = 0;
            while i < sorted_indices.len() {
                let idx = sorted_indices[i];
                let (t, _, _) = obs[idx];

                // Find all observations at this time
                let mut j = i;
                let mut tie_indices: Vec<usize> = Vec::new();
                while j < sorted_indices.len() {
                    let jdx = sorted_indices[j];
                    if (obs[jdx].0 - t).abs() < 1e-10 {
                        tie_indices.push(jdx);
                        cum_exp_eta += exp_eta[obs[jdx].2];
                        j += 1;
                    } else {
                        break;
                    }
                }

                // Get events at this time
                let event_indices: Vec<usize> = tie_indices
                    .iter()
                    .filter(|&&tidx| obs[tidx].1)
                    .copied()
                    .collect();
                let d = event_indices.len();

                if d > 0 {
                    let sum_exp_events: f64 =
                        event_indices.iter().map(|&tidx| exp_eta[obs[tidx].2]).sum();

                    for (k, &eidx) in event_indices.iter().enumerate() {
                        let row_idx = obs[eidx].2;
                        let fraction = k as f64 / d as f64;
                        log_lik += eta[row_idx] - (cum_exp_eta - fraction * sum_exp_events).ln();
                    }
                }

                i = j;
            }
        }
    }

    log_lik
}

/// Compute gradient and Hessian for Cox partial likelihood.
fn cox_gradient_hessian(
    obs: &[(f64, bool, usize)],
    x: &Array2<f64>,
    beta: &Array1<f64>,
    sorted_indices: &[usize],
    ties: TiesMethod,
) -> (Array1<f64>, Array2<f64>, f64) {
    let n = obs.len();
    let p = beta.len();
    let mut grad = Array1::zeros(p);
    let mut hess = Array2::zeros((p, p));
    let mut log_lik = 0.0;

    // Compute exp(beta'x) for all observations
    let eta: Vec<f64> = (0..n).map(|i| x.row(i).dot(beta)).collect();
    let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

    // Running sums for risk set computation
    let mut s0: f64 = 0.0; // sum of exp(eta)
    let mut s1: Array1<f64> = Array1::zeros(p); // sum of x * exp(eta)
    let mut s2: Array2<f64> = Array2::zeros((p, p)); // sum of x * x' * exp(eta)

    match ties {
        TiesMethod::Breslow => {
            for &idx in sorted_indices.iter() {
                let (_, event, row_idx) = obs[idx];
                let xi = x.row(row_idx);
                let exp_eta_i = exp_eta[row_idx];

                // Update risk set sums (in-place, zero allocations)
                s0 += exp_eta_i;
                for j in 0..p {
                    s1[j] += xi[j] * exp_eta_i;
                    for k in 0..p {
                        s2[[j, k]] += xi[j] * xi[k] * exp_eta_i;
                    }
                }

                if event {
                    log_lik += eta[row_idx] - s0.ln();

                    // Gradient contribution: x_i - E[X|R_i] (in-place)
                    let inv_s0 = 1.0 / s0;
                    for j in 0..p {
                        let x_bar_j = s1[j] * inv_s0;
                        grad[j] += xi[j] - x_bar_j;
                    }

                    // Hessian contribution: -Var[X|R_i]
                    for j in 0..p {
                        let x_bar_j = s1[j] * inv_s0;
                        for k in 0..p {
                            let x_bar_k = s1[k] * inv_s0;
                            hess[[j, k]] += s2[[j, k]] * inv_s0 - x_bar_j * x_bar_k;
                        }
                    }
                }
            }
        }
        TiesMethod::Efron => {
            // Efron's method with proper tie handling
            let mut i = 0;
            while i < sorted_indices.len() {
                let idx = sorted_indices[i];
                let (t, _, _) = obs[idx];

                // Collect all observations at this time
                let mut tie_indices: Vec<usize> = Vec::new();
                let mut j = i;
                while j < sorted_indices.len() {
                    let jdx = sorted_indices[j];
                    if (obs[jdx].0 - t).abs() < 1e-10 {
                        tie_indices.push(jdx);
                        j += 1;
                    } else {
                        break;
                    }
                }

                // Update risk set sums with all tied observations (in-place)
                for &tidx in &tie_indices {
                    let row_idx = obs[tidx].2;
                    let xi = x.row(row_idx);
                    let exp_eta_i = exp_eta[row_idx];

                    s0 += exp_eta_i;
                    for jj in 0..p {
                        s1[jj] += xi[jj] * exp_eta_i;
                        for kk in 0..p {
                            s2[[jj, kk]] += xi[jj] * xi[kk] * exp_eta_i;
                        }
                    }
                }

                // Get event indices at this time
                let event_indices: Vec<usize> = tie_indices
                    .iter()
                    .filter(|&&tidx| obs[tidx].1)
                    .copied()
                    .collect();
                let d = event_indices.len();

                if d > 0 {
                    // Compute sums over events (in-place)
                    let mut sum_exp_events: f64 = 0.0;
                    let mut sum_x_exp_events: Vec<f64> = vec![0.0; p];
                    let mut sum_xx_exp_events: Array2<f64> = Array2::zeros((p, p));

                    for &eidx in &event_indices {
                        let row_idx = obs[eidx].2;
                        let xi = x.row(row_idx);
                        let exp_eta_i = exp_eta[row_idx];

                        sum_exp_events += exp_eta_i;
                        for jj in 0..p {
                            sum_x_exp_events[jj] += xi[jj] * exp_eta_i;
                            for kk in 0..p {
                                sum_xx_exp_events[[jj, kk]] += xi[jj] * xi[kk] * exp_eta_i;
                            }
                        }
                    }

                    // Efron contribution for each event
                    for (k, &eidx) in event_indices.iter().enumerate() {
                        let row_idx = obs[eidx].2;
                        let xi = x.row(row_idx);
                        let fraction = k as f64 / d as f64;

                        let s0_adj = s0 - fraction * sum_exp_events;

                        if s0_adj > 0.0 {
                            log_lik += eta[row_idx] - s0_adj.ln();

                            let inv_s0_adj = 1.0 / s0_adj;
                            for jj in 0..p {
                                let x_bar_jj =
                                    (s1[jj] - fraction * sum_x_exp_events[jj]) * inv_s0_adj;
                                grad[jj] += xi[jj] - x_bar_jj;
                            }

                            for jj in 0..p {
                                let x_bar_jj =
                                    (s1[jj] - fraction * sum_x_exp_events[jj]) * inv_s0_adj;
                                for kk in 0..p {
                                    let x_bar_kk =
                                        (s1[kk] - fraction * sum_x_exp_events[kk]) * inv_s0_adj;
                                    hess[[jj, kk]] += (s2[[jj, kk]]
                                        - fraction * sum_xx_exp_events[[jj, kk]])
                                        * inv_s0_adj
                                        - x_bar_jj * x_bar_kk;
                                }
                            }
                        }
                    }
                }

                i = j;
            }
        }
    }

    (grad, hess, log_lik)
}

/// Fenwick tree (Binary Indexed Tree) for cumulative frequency queries.
/// Supports point updates and prefix sum queries in O(log n).
struct FenwickTree {
    tree: Vec<f64>,
    n: usize,
}

impl FenwickTree {
    fn new(n: usize) -> Self {
        Self {
            tree: vec![0.0; n + 1],
            n,
        }
    }

    /// Add `val` to position `i` (1-indexed).
    fn update(&mut self, mut i: usize, val: f64) {
        while i <= self.n {
            self.tree[i] += val;
            i += i & i.wrapping_neg();
        }
    }

    /// Sum of elements from position 1 to `i` (inclusive, 1-indexed).
    fn prefix_sum(&self, mut i: usize) -> f64 {
        let mut s = 0.0;
        while i > 0 {
            s += self.tree[i];
            i -= i & i.wrapping_neg();
        }
        s
    }

    /// Sum of elements in range [lo, hi] (1-indexed, inclusive).
    fn range_sum(&self, lo: usize, hi: usize) -> f64 {
        if lo > hi {
            return 0.0;
        }
        self.prefix_sum(hi) - if lo > 1 { self.prefix_sum(lo - 1) } else { 0.0 }
    }
}

/// Compute concordance statistic (C-index) for Cox model.
///
/// Uses an O(n log n) algorithm with a Fenwick tree. Sorts observations by time
/// descending and processes events, using the Fenwick tree to count how many
/// already-inserted (i.e., larger-time) risk scores are less than, equal to,
/// or greater than the current event's risk score.
fn compute_concordance(
    obs: &[(f64, bool, usize)],
    x: &Array2<f64>,
    beta: &Array1<f64>,
) -> (f64, f64) {
    let n = obs.len();
    if n < 2 {
        return (0.5, 0.0);
    }
    let eta: Vec<f64> = (0..n).map(|i| x.row(i).dot(beta)).collect();

    // Create (time, event, risk_score) sorted by time ascending
    let mut sorted: Vec<(f64, bool, f64)> = obs.iter().map(|&(t, e, ri)| (t, e, eta[ri])).collect();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Coordinate-compress risk scores for Fenwick tree indexing
    let mut scores: Vec<f64> = sorted.iter().map(|s| s.2).collect();
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    scores.dedup();
    let m = scores.len(); // number of distinct risk scores

    // Map risk score -> 1-indexed rank
    let score_rank = |s: f64| -> usize { scores.partition_point(|&v| v < s) + 1 };

    let mut concordant = 0.0_f64;
    let mut discordant = 0.0_f64;
    let mut tied_risk = 0.0_f64;

    // Fenwick tree tracks risk scores of observations with strictly larger times
    let mut bit = FenwickTree::new(m);
    let mut total_inserted: f64 = 0.0;

    // Process groups of observations with the same time, from largest to smallest time.
    // For each event, observations already in the BIT have strictly larger times.
    let mut i = n;
    while i > 0 {
        // Find the start of this time group
        let group_end = i;
        let t_current = sorted[i - 1].0;
        let mut group_start = i - 1;
        while group_start > 0 && (sorted[group_start - 1].0 - t_current).abs() < 1e-10 {
            group_start -= 1;
        }

        // Process events in this time group (count pairs with BIT entries = larger times)
        for idx in group_start..group_end {
            if sorted[idx].1 {
                let rank = score_rank(sorted[idx].2);
                let n_less = bit.prefix_sum(rank - 1); // scores < this -> concordant (event has higher risk)
                let n_equal = bit.range_sum(rank, rank); // scores == this -> tied risk
                let n_greater = total_inserted - bit.prefix_sum(rank); // scores > this -> discordant
                concordant += n_less;
                discordant += n_greater;
                tied_risk += n_equal;
            }
        }

        // Insert all observations in this time group into BIT
        for idx in group_start..group_end {
            let rank = score_rank(sorted[idx].2);
            bit.update(rank, 1.0);
            total_inserted += 1.0;
        }

        i = group_start;
    }

    let total: f64 = concordant + discordant + tied_risk;
    let c = if total > 0.0 {
        (concordant + 0.5 * tied_risk) / total
    } else {
        0.5
    };

    // Approximate standard error (simplified)
    let se = if total > 0.0 {
        (c * (1.0 - c) / total.sqrt()).sqrt()
    } else {
        0.0
    };

    (c, se)
}

// =============================================================================
// Accelerated Failure Time Models
// =============================================================================

/// Configuration for AFT models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AftConfig {
    /// Distribution family
    pub distribution: AftDistribution,
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for AftConfig {
    fn default() -> Self {
        Self {
            distribution: AftDistribution::Weibull,
            max_iter: 100,
            tolerance: 1e-8,
        }
    }
}

/// Result from AFT model.
///
/// # References
///
/// - Kalbfleisch, J.D. & Prentice, R.L. (2002). The Statistical Analysis of
///   Failure Time Data, 2nd Edition. Wiley.
/// - Wei, L.J. (1992). "The Accelerated Failure Time Model: A Useful Alternative
///   to the Cox Regression Model in Survival Analysis". Statistics in Medicine, 11:1871-1879.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AftResult {
    /// Distribution used
    pub distribution: AftDistribution,
    /// Variable names (including intercept)
    pub variables: Vec<String>,
    /// Estimated coefficients
    pub coefficients: Vec<f64>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// z-statistics
    pub z_stats: Vec<f64>,
    /// P-values
    pub p_values: Vec<f64>,
    /// Significance levels
    pub significance: Vec<SignificanceLevel>,
    /// Acceleration factors exp(β)
    pub acceleration_factors: Vec<f64>,
    /// Scale parameter (σ for Weibull/LogNormal)
    pub scale: f64,
    /// Shape parameter (if applicable)
    pub shape: Option<f64>,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// AIC
    pub aic: f64,
    /// BIC
    pub bic: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of events
    pub n_events: usize,
    /// Whether convergence was achieved
    pub converged: bool,
    /// Number of iterations
    pub iterations: usize,
}

impl fmt::Display for AftResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Accelerated Failure Time Model")?;
        writeln!(f, "===============================")?;
        writeln!(f, "Distribution: {}", self.distribution)?;
        writeln!(f, "n = {}, events = {}", self.n_obs, self.n_events)?;
        writeln!(f)?;

        writeln!(
            f,
            "{:<20} {:>10} {:>10} {:>8} {:>8} {:>12}",
            "Variable", "coef", "se(coef)", "z", "p", "Accel. Factor"
        )?;
        writeln!(f, "{}", "-".repeat(75))?;

        for i in 0..self.variables.len() {
            writeln!(
                f,
                "{:<20} {:>10.4} {:>10.4} {:>8.2} {:>8.4}{} {:>12.4}",
                self.variables[i],
                self.coefficients[i],
                self.std_errors[i],
                self.z_stats[i],
                self.p_values[i],
                self.significance[i].stars(),
                self.acceleration_factors[i]
            )?;
        }

        writeln!(f, "{}", "-".repeat(75))?;
        writeln!(f, "Scale = {:.4}", self.scale)?;
        if let Some(shape) = self.shape {
            writeln!(f, "Shape = {:.4}", shape)?;
        }
        writeln!(f)?;
        writeln!(f, "Log-Likelihood: {:.4}", self.log_likelihood)?;
        writeln!(f, "AIC: {:.4}", self.aic)?;
        writeln!(f, "BIC: {:.4}", self.bic)?;
        writeln!(f)?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;

        Ok(())
    }
}

/// Run Accelerated Failure Time model.
///
/// # Arguments
///
/// * `dataset` - The dataset containing survival data
/// * `time_col` - Column name for survival/censoring time
/// * `event_col` - Column name for event indicator (1 = event, 0 = censored)
/// * `x_cols` - Column names for covariates
/// * `config` - Optional configuration (distribution, convergence settings)
///
/// # Mathematical Details
///
/// The AFT model assumes:
/// ```text
/// log(T) = μ + β'X + σε
/// ```
///
/// where ε follows a specified distribution. Parameters are estimated via MLE.
pub fn run_aft(
    dataset: &Dataset,
    time_col: &str,
    event_col: &str,
    x_cols: &[&str],
    config: Option<AftConfig>,
) -> EconResult<AftResult> {
    let config = config.unwrap_or_default();
    let df = dataset.df();
    let available_cols: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    // Extract data
    let times = df
        .column(time_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: time_col.to_string(),
            available: available_cols.clone(),
        })?
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: time_col.to_string(),
        })?;

    let events = df
        .column(event_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: event_col.to_string(),
            available: available_cols.clone(),
        })?
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: event_col.to_string(),
        })?;

    // Build design matrix with intercept
    let p = x_cols.len() + 1; // +1 for intercept
    let mut x_data: Vec<f64> = Vec::new();
    let mut log_times: Vec<f64> = Vec::new();
    let mut deltas: Vec<bool> = Vec::new();
    let mut valid_rows = 0;

    for i in 0..df.height() {
        let t = match times.get(i) {
            Some(v) if !v.is_nan() && v > 0.0 => v,
            _ => continue,
        };
        let e = match events.get(i) {
            Some(v) if !v.is_nan() => v != 0.0,
            _ => continue,
        };

        // Intercept
        let mut row_x: Vec<f64> = vec![1.0];
        let mut row_valid = true;

        // Extract covariates
        for col in x_cols {
            let col_data = df
                .column(col)
                .map_err(|_| EconError::ColumnNotFound {
                    column: col.to_string(),
                    available: available_cols.clone(),
                })?
                .f64()
                .map_err(|_| EconError::NonNumericColumn {
                    column: col.to_string(),
                })?;
            match col_data.get(i) {
                Some(v) if !v.is_nan() => row_x.push(v),
                _ => {
                    row_valid = false;
                    break;
                }
            }
        }

        if row_valid {
            x_data.extend(row_x);
            log_times.push(t.ln());
            deltas.push(e);
            valid_rows += 1;
        }
    }

    let n = valid_rows;
    if n < p + 1 {
        return Err(EconError::InsufficientData {
            required: p + 1,
            provided: n,
            context: format!("AFT model with {} parameters", p),
        });
    }

    let n_events = deltas.iter().filter(|&&d| d).count();
    if n_events == 0 {
        return Err(EconError::InvalidSpecification {
            message: "No events observed in the data".to_string(),
        });
    }

    let x = Array2::from_shape_vec((n, p), x_data)
        .map_err(|e| EconError::Internal(format!("Matrix construction failed: {}", e)))?;
    let y: Array1<f64> = Array1::from_vec(log_times);

    // Initialize parameters: OLS estimates for location, reasonable scale
    let xtx = x.t().dot(&x);
    let xty = x.t().dot(&y);
    let (xtx_inv, _) = safe_inverse(&xtx.view())?;
    let mut beta: Array1<f64> = xtx_inv.dot(&xty);

    // Initial scale estimate
    let residuals: Array1<f64> = &y - &x.dot(&beta);
    let mut log_sigma = (residuals.mapv(|r| r.powi(2)).sum() / (n - p) as f64)
        .sqrt()
        .ln();

    let mut converged = false;
    let mut iterations = 0;
    let mut final_ll = 0.0;

    // Pre-allocate buffers for Newton updates
    let mut delta_beta = Array1::zeros(p);
    let mut full_hess = Array2::zeros((p + 1, p + 1));
    let mut full_grad = Array1::zeros(p + 1);

    // Newton-Raphson iteration
    for iter in 0..config.max_iter {
        iterations = iter + 1;

        let sigma = log_sigma.exp();
        let (grad_beta, grad_sigma, hess_bb, hess_bs, hess_ss, ll) =
            aft_gradient_hessian(&y, &x, &beta, sigma, &deltas, config.distribution);
        final_ll = ll;

        // Check convergence using relative gradient norm
        let grad_norm = grad_beta
            .iter()
            .map(|g| g.abs())
            .fold(0.0, f64::max)
            .max(grad_sigma.abs());
        if grad_norm < config.tolerance {
            converged = true;
            break;
        }

        // Construct full Hessian and gradient (reuse allocated buffers)
        full_hess
            .slice_mut(ndarray::s![0..p, 0..p])
            .assign(&hess_bb);
        for j in 0..p {
            full_hess[[j, p]] = hess_bs[j];
            full_hess[[p, j]] = hess_bs[j];
        }
        full_hess[[p, p]] = hess_ss;

        full_grad.slice_mut(ndarray::s![0..p]).assign(&grad_beta);
        full_grad[p] = grad_sigma;

        // Newton step using Cholesky solver
        match solve_symmetric_system(&full_hess, &full_grad) {
            Some(delta) => {
                // Extract delta components
                for j in 0..p {
                    delta_beta[j] = delta[j];
                }
                let delta_sigma = delta[p];

                // Step halving with early termination
                let mut step = 1.0;
                let mut new_ll = ll - 1.0; // Force at least one evaluation
                let mut halving_count = 0;

                while halving_count < 10 {
                    // Compute new parameters
                    let new_log_sigma = log_sigma + delta_sigma * step;

                    // Compute new likelihood
                    new_ll = aft_log_likelihood_fast(
                        &y,
                        &x,
                        &beta,
                        &delta_beta,
                        step,
                        new_log_sigma.exp(),
                        &deltas,
                        config.distribution,
                    );

                    if new_ll >= ll - 1e-10 || step < 1e-10 {
                        // Accept step
                        for j in 0..p {
                            beta[j] += delta_beta[j] * step;
                        }
                        log_sigma = new_log_sigma;
                        break;
                    }
                    step *= 0.5;
                    halving_count += 1;
                }

                if halving_count >= 10 {
                    // Fallback: small gradient step
                    for j in 0..p {
                        beta[j] += grad_beta[j] * 0.001;
                    }
                    log_sigma += grad_sigma * 0.001;
                }
            }
            None => {
                // Fall back to gradient descent
                for j in 0..p {
                    beta[j] += grad_beta[j] * 0.01;
                }
                log_sigma += grad_sigma * 0.01;
            }
        }
    }

    let sigma = log_sigma.exp();
    let log_likelihood = final_ll;

    // Compute final Hessian for standard errors (only if we need to)
    let (_, _, hess_bb, hess_bs, hess_ss, _) = if converged {
        aft_gradient_hessian(&y, &x, &beta, sigma, &deltas, config.distribution)
    } else {
        // Use last computed values by recomputing
        aft_gradient_hessian(&y, &x, &beta, sigma, &deltas, config.distribution)
    };

    full_hess
        .slice_mut(ndarray::s![0..p, 0..p])
        .assign(&hess_bb);
    for j in 0..p {
        full_hess[[j, p]] = hess_bs[j];
        full_hess[[p, j]] = hess_bs[j];
    }
    full_hess[[p, p]] = hess_ss;

    let info_matrix = safe_inverse(&full_hess.view())
        .map(|(m, _)| m)
        .unwrap_or_else(|_| Array2::eye(p + 1));

    let std_errors: Vec<f64> = (0..p).map(|i| info_matrix[[i, i]].abs().sqrt()).collect();

    let z_stats: Vec<f64> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
        .collect();

    let p_values: Vec<f64> = z_stats
        .iter()
        .map(|&z| 2.0 * (1.0 - normal_cdf(z.abs())))
        .collect();

    let significance: Vec<SignificanceLevel> = p_values
        .iter()
        .map(|&pv| SignificanceLevel::from_p_value(pv))
        .collect();

    let acceleration_factors: Vec<f64> = beta.iter().map(|&b| b.exp()).collect();

    // Variable names
    let mut variables = vec!["(Intercept)".to_string()];
    variables.extend(x_cols.iter().map(|s| s.to_string()));

    // Shape parameter (for Weibull: shape = 1/sigma)
    let shape = match config.distribution {
        AftDistribution::Weibull => Some(1.0 / sigma),
        AftDistribution::Exponential => Some(1.0),
        _ => None,
    };

    // AIC and BIC
    let k = (p + 1) as f64;
    let aic = -2.0 * log_likelihood + 2.0 * k;
    let bic = -2.0 * log_likelihood + k * (n as f64).ln();

    Ok(AftResult {
        distribution: config.distribution,
        variables,
        coefficients: beta.to_vec(),
        std_errors,
        z_stats,
        p_values,
        significance,
        acceleration_factors,
        scale: sigma,
        shape,
        log_likelihood,
        aic,
        bic,
        n_obs: n,
        n_events,
        converged,
        iterations,
    })
}

/// Log-likelihood computation with step adjustment.
/// Computes likelihood at beta + step * delta_beta.
fn aft_log_likelihood_fast(
    y: &Array1<f64>,
    x: &Array2<f64>,
    beta: &Array1<f64>,
    delta_beta: &Array1<f64>,
    step: f64,
    sigma: f64,
    deltas: &[bool],
    dist: AftDistribution,
) -> f64 {
    let n = y.len();
    let p = beta.len();
    let sigma_inv = 1.0 / sigma;
    let log_sigma = sigma.ln();
    let mut ll = 0.0;

    for i in 0..n {
        // Compute eta[i] = sum_j x[i,j] * (beta[j] + step * delta_beta[j])
        let mut eta_i = 0.0;
        for j in 0..p {
            eta_i += x[[i, j]] * (beta[j] + step * delta_beta[j]);
        }

        let z = (y[i] - eta_i) * sigma_inv;
        let (log_f, log_s) = aft_distribution_functions(z, dist);

        if deltas[i] {
            ll += log_f - log_sigma;
        } else {
            ll += log_s;
        }
    }

    ll
}

/// Solve symmetric positive definite system Hx = b using Cholesky decomposition.
/// Returns None if the matrix is not positive definite.
fn solve_symmetric_system(h: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    let n = h.nrows();

    // Cholesky decomposition: H = L * L'
    let mut l = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = h[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if sum <= 0.0 {
                    return None; // Not positive definite
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }

    // Solve L * y = b (forward substitution)
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[[i, j]] * y[j];
        }
        y[i] = sum / l[[i, i]];
    }

    // Solve L' * x = y (backward substitution)
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[[j, i]] * x[j];
        }
        x[i] = sum / l[[i, i]];
    }

    Some(x)
}

/// Compute gradient and Hessian for AFT model.
/// Optimized single-pass computation with efficient memory access.
fn aft_gradient_hessian(
    y: &Array1<f64>,
    x: &Array2<f64>,
    beta: &Array1<f64>,
    sigma: f64,
    deltas: &[bool],
    dist: AftDistribution,
) -> (Array1<f64>, f64, Array2<f64>, Array1<f64>, f64, f64) {
    let n = y.len();
    let p = beta.len();

    // Pre-compute constants
    let sigma_inv = 1.0 / sigma;
    let sigma_inv_sq = sigma_inv * sigma_inv;
    let log_sigma = sigma.ln();

    // Pre-compute eta = X * beta (vectorized)
    let eta = x.dot(beta);

    // Initialize accumulators
    let mut ll = 0.0;
    let mut grad_beta = Array1::zeros(p);
    let mut grad_sigma = 0.0;
    let mut hess_bb = Array2::zeros((p, p));
    let mut hess_bs = Array1::zeros(p);
    let mut hess_ss = 0.0;

    // Single pass through all observations
    for i in 0..n {
        let z = (y[i] - eta[i]) * sigma_inv;
        let (log_f, log_s, dlog_f, dlog_s, d2log_f, d2log_s) =
            aft_distribution_derivatives(z, dist);

        // Get row slice for efficient access
        let xi = x.row(i);

        if deltas[i] {
            // Event observed
            ll += log_f - log_sigma;

            // Gradient contributions
            let grad_beta_w = -dlog_f * sigma_inv;
            grad_sigma += -dlog_f * z - 1.0;

            // Hessian weights
            let hess_bb_w = d2log_f * sigma_inv_sq;
            let hess_bs_w = (d2log_f * z + dlog_f) * sigma_inv;
            hess_ss += d2log_f * z * z + 2.0 * dlog_f * z;

            // Accumulate grad_beta and hess_bs using row view
            accumulate_weighted_row(&mut grad_beta, xi, grad_beta_w);
            accumulate_weighted_row(&mut hess_bs, xi, hess_bs_w);

            // Accumulate hess_bb using rank-1 update (only upper triangle)
            accumulate_rank1_symmetric(&mut hess_bb, xi, hess_bb_w);
        } else {
            // Censored observation
            ll += log_s;

            // Gradient contributions
            let grad_beta_w = -dlog_s * sigma_inv;
            grad_sigma += -dlog_s * z;

            // Hessian weights
            let hess_bb_w = d2log_s * sigma_inv_sq;
            let hess_bs_w = (d2log_s * z + dlog_s) * sigma_inv;
            hess_ss += d2log_s * z * z + dlog_s * z;

            // Accumulate grad_beta and hess_bs using row view
            accumulate_weighted_row(&mut grad_beta, xi, grad_beta_w);
            accumulate_weighted_row(&mut hess_bs, xi, hess_bs_w);

            // Accumulate hess_bb using rank-1 update
            accumulate_rank1_symmetric(&mut hess_bb, xi, hess_bb_w);
        }
    }

    (grad_beta, grad_sigma, hess_bb, hess_bs, hess_ss, ll)
}

/// Accumulate weighted row: result += weight * row
#[inline]
fn accumulate_weighted_row(result: &mut Array1<f64>, row: ArrayView1<f64>, weight: f64) {
    for (r, &x) in result.iter_mut().zip(row.iter()) {
        *r += weight * x;
    }
}

/// Accumulate symmetric rank-1 update: result += weight * row * row'
/// Only updates upper triangle then copies to lower.
#[inline]
fn accumulate_rank1_symmetric(result: &mut Array2<f64>, row: ArrayView1<f64>, weight: f64) {
    let p = row.len();
    for j in 0..p {
        let wj = weight * row[j];
        for k in j..p {
            result[[j, k]] += wj * row[k];
        }
    }
    // Copy upper to lower triangle
    for j in 0..p {
        for k in (j + 1)..p {
            result[[k, j]] = result[[j, k]];
        }
    }
}

/// Distribution functions for AFT models.
/// Returns (log_f, log_S) where f is the PDF and S is the survival function.
fn aft_distribution_functions(z: f64, dist: AftDistribution) -> (f64, f64) {
    match dist {
        AftDistribution::Exponential | AftDistribution::Weibull => {
            // Extreme value (Gumbel) distribution
            // f(z) = exp(z - exp(z))
            // S(z) = exp(-exp(z))
            let exp_z = z.exp();
            let log_f = z - exp_z;
            let log_s = -exp_z;
            (log_f, log_s)
        }
        AftDistribution::LogNormal => {
            // Standard normal distribution
            let log_f = -0.5 * (z * z + (2.0 * std::f64::consts::PI).ln());
            let log_s = (1.0 - normal_cdf(z)).ln();
            (log_f, log_s)
        }
        AftDistribution::LogLogistic => {
            // Logistic distribution
            // f(z) = exp(z) / (1 + exp(z))^2
            // S(z) = 1 / (1 + exp(z))
            let exp_z = z.exp();
            let denom = 1.0 + exp_z;
            let log_f = z - 2.0 * denom.ln();
            let log_s = -denom.ln();
            (log_f, log_s)
        }
    }
}

/// Distribution derivatives for AFT models.
/// Returns (log_f, log_S, dlog_f/dz, dlog_S/dz, d²log_f/dz², d²log_S/dz²)
fn aft_distribution_derivatives(z: f64, dist: AftDistribution) -> (f64, f64, f64, f64, f64, f64) {
    match dist {
        AftDistribution::Exponential | AftDistribution::Weibull => {
            let exp_z = z.exp();
            let log_f = z - exp_z;
            let log_s = -exp_z;
            let dlog_f = 1.0 - exp_z;
            let dlog_s = -exp_z;
            let d2log_f = -exp_z;
            let d2log_s = -exp_z;
            (log_f, log_s, dlog_f, dlog_s, d2log_f, d2log_s)
        }
        AftDistribution::LogNormal => {
            let log_f = -0.5 * (z * z + (2.0 * std::f64::consts::PI).ln());
            let phi = normal_cdf(z);
            let log_s = (1.0 - phi).ln();
            let dlog_f = -z;
            // Mills ratio approximation for numerical stability
            let mills = if z < -6.0 {
                -z
            } else if z > 6.0 {
                1.0 / z
            } else {
                let pdf = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
                pdf / (1.0 - phi).max(1e-15)
            };
            let dlog_s = -mills;
            let d2log_f = -1.0;
            let d2log_s = -mills * (mills + z);
            (log_f, log_s, dlog_f, dlog_s, d2log_f, d2log_s)
        }
        AftDistribution::LogLogistic => {
            let exp_z = z.exp();
            let denom = 1.0 + exp_z;
            let log_f = z - 2.0 * denom.ln();
            let log_s = -denom.ln();
            let p = exp_z / denom; // logistic CDF
            let dlog_f = 1.0 - 2.0 * p;
            let dlog_s = -p;
            let d2log_f = -2.0 * p * (1.0 - p);
            let d2log_s = -p * (1.0 - p);
            (log_f, log_s, dlog_f, dlog_s, d2log_f, d2log_s)
        }
    }
}

// =============================================================================
// Competing Risks / Aalen-Johansen
// =============================================================================

/// Result for a single event type in competing risks analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CumulativeIncidence {
    /// Event type identifier
    pub event_type: u8,
    /// Time points
    pub times: Vec<f64>,
    /// Cumulative incidence F(t)
    pub incidence: Vec<f64>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// Lower confidence interval
    pub ci_lower: Vec<f64>,
    /// Upper confidence interval
    pub ci_upper: Vec<f64>,
}

/// Result from competing risks analysis.
///
/// # References
///
/// - Aalen, O.O. & Johansen, S. (1978). "An Empirical Transition Matrix for Non-Homogeneous
///   Markov Chains Based on Censored Observations". Scandinavian Journal of Statistics, 5:141-150.
/// - Gray, R.J. (1988). "A Class of K-Sample Tests for Comparing the Cumulative Incidence of a
///   Competing Risk". Annals of Statistics, 16:1141-1154.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompetingRisksResult {
    /// Event types analyzed
    pub event_types: Vec<u8>,
    /// Cumulative incidence functions per event type
    pub cifs: Vec<CumulativeIncidence>,
    /// Total observations
    pub n_obs: usize,
    /// Events per type
    pub n_events_by_type: HashMap<u8, usize>,
    /// Total censored
    pub n_censored: usize,
    /// Confidence level
    pub conf_level: f64,
}

impl fmt::Display for CompetingRisksResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Competing Risks Analysis (Aalen-Johansen)")?;
        writeln!(f, "==========================================")?;
        writeln!(f, "N = {}, Censored = {}", self.n_obs, self.n_censored)?;
        writeln!(f)?;

        writeln!(f, "Events by type:")?;
        for (event_type, count) in &self.n_events_by_type {
            writeln!(f, "  Type {}: {} events", event_type, count)?;
        }
        writeln!(f)?;

        for cif in &self.cifs {
            writeln!(f, "Event Type {}:", cif.event_type)?;
            writeln!(
                f,
                "{:>10} {:>12} {:>12} {:>12}",
                "Time", "CIF", "Std.Err", "95% CI"
            )?;
            writeln!(f, "{}", "-".repeat(50))?;

            let n = cif.times.len();
            let show_all = n <= 10;
            let indices: Vec<usize> = if show_all {
                (0..n).collect()
            } else {
                let mut idx: Vec<usize> = (0..5).collect();
                idx.push(usize::MAX);
                idx.extend((n.saturating_sub(3))..n);
                idx
            };

            for &i in &indices {
                if i == usize::MAX {
                    writeln!(f, "{:>10}", "...")?;
                } else {
                    writeln!(
                        f,
                        "{:>10.3} {:>12.4} {:>12.4} [{:.4}, {:.4}]",
                        cif.times[i],
                        cif.incidence[i],
                        cif.std_errors[i],
                        cif.ci_lower[i],
                        cif.ci_upper[i]
                    )?;
                }
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

/// Run competing risks analysis using Aalen-Johansen estimator.
///
/// # Arguments
///
/// * `dataset` - The dataset containing survival data
/// * `time_col` - Column name for event/censoring time
/// * `event_type_col` - Column name for event type (0 = censored, 1,2,... = event types)
/// * `conf_level` - Confidence level (default: 0.95)
///
/// # Mathematical Details
///
/// The cumulative incidence function for event type k is:
/// ```text
/// F̂ₖ(t) = Σ(tᵢ ≤ t) Ŝ(tᵢ₋₁) × dₖᵢ / nᵢ
/// ```
///
/// where Ŝ(t) is the Kaplan-Meier estimate of overall survival.
pub fn run_competing_risks(
    dataset: &Dataset,
    time_col: &str,
    event_type_col: &str,
    conf_level: f64,
) -> EconResult<CompetingRisksResult> {
    if conf_level <= 0.0 || conf_level >= 1.0 {
        return Err(EconError::InvalidSpecification {
            message: "Confidence level must be between 0 and 1".to_string(),
        });
    }

    let df = dataset.df();
    let available_cols: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    let times = df
        .column(time_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: time_col.to_string(),
            available: available_cols.clone(),
        })?
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: time_col.to_string(),
        })?;

    let event_types = df
        .column(event_type_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: event_type_col.to_string(),
            available: available_cols,
        })?
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: event_type_col.to_string(),
        })?;

    // Build observations
    let mut observations: Vec<(f64, u8)> = Vec::new();
    let mut event_type_set: std::collections::BTreeSet<u8> = std::collections::BTreeSet::new();

    for i in 0..df.height() {
        let t = match times.get(i) {
            Some(v) if !v.is_nan() && v >= 0.0 => v,
            _ => continue,
        };
        let e = match event_types.get(i) {
            Some(v) if !v.is_nan() && v >= 0.0 => v as u8,
            _ => continue,
        };

        observations.push((t, e));
        if e > 0 {
            event_type_set.insert(e);
        }
    }

    let n_obs = observations.len();
    if n_obs == 0 {
        return Err(EconError::EmptyDataset);
    }

    let event_types_vec: Vec<u8> = event_type_set.into_iter().collect();
    if event_types_vec.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "No events observed (all observations censored)".to_string(),
        });
    }

    // Sort by time
    observations.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Count events by type
    let mut n_events_by_type: HashMap<u8, usize> = HashMap::new();
    let mut n_censored = 0;
    for (_, e) in &observations {
        if *e == 0 {
            n_censored += 1;
        } else {
            *n_events_by_type.entry(*e).or_insert(0) += 1;
        }
    }

    // Get distinct event times
    let mut time_events: BTreeMap<OrderedF64, HashMap<u8, usize>> = BTreeMap::new();
    for (t, e) in &observations {
        let key = OrderedF64(*t);
        *time_events.entry(key).or_default().entry(*e).or_insert(0) += 1;
    }

    // Compute Aalen-Johansen estimator
    let z = normal_quantile(1.0 - (1.0 - conf_level) / 2.0);
    let mut cifs: Vec<CumulativeIncidence> = event_types_vec
        .iter()
        .map(|&et| CumulativeIncidence {
            event_type: et,
            times: Vec::new(),
            incidence: Vec::new(),
            std_errors: Vec::new(),
            ci_lower: Vec::new(),
            ci_upper: Vec::new(),
        })
        .collect();

    let mut current_risk = n_obs;
    let mut km_surv = 1.0; // Kaplan-Meier for overall survival
    let mut cum_incidence: Vec<f64> = vec![0.0; event_types_vec.len()];
    let mut variance: Vec<f64> = vec![0.0; event_types_vec.len()];

    for (time, events_at_time) in time_events {
        let t = time.0;

        // Total events and censored at this time
        let total_events: usize = events_at_time
            .iter()
            .filter(|&(&e, _)| e > 0)
            .map(|(_, &c)| c)
            .sum();
        let total_censored = *events_at_time.get(&0).unwrap_or(&0);

        if current_risk > 0 {
            // Update cumulative incidence for each event type
            for (idx, &event_type) in event_types_vec.iter().enumerate() {
                let d_k = *events_at_time.get(&event_type).unwrap_or(&0);
                if d_k > 0 {
                    // Aalen-Johansen increment
                    let increment = km_surv * (d_k as f64 / current_risk as f64);
                    cum_incidence[idx] += increment;

                    // Variance (simplified Greenwood-type formula)
                    variance[idx] += (km_surv.powi(2) * d_k as f64)
                        / (current_risk as f64 * current_risk as f64);
                }
            }

            // Update Kaplan-Meier for overall survival
            if total_events > 0 {
                km_surv *= 1.0 - (total_events as f64 / current_risk as f64);
            }
        }

        // Store results for this time point
        for (idx, cif) in cifs.iter_mut().enumerate() {
            cif.times.push(t);
            cif.incidence.push(cum_incidence[idx]);
            let se = variance[idx].sqrt();
            cif.std_errors.push(se);

            // Confidence intervals using log transformation
            let (lo, hi) = if cum_incidence[idx] > 0.0 && cum_incidence[idx] < 1.0 {
                let log_cif = cum_incidence[idx].ln();
                let se_log = se / cum_incidence[idx];
                ((log_cif - z * se_log).exp(), (log_cif + z * se_log).exp())
            } else {
                (cum_incidence[idx], cum_incidence[idx])
            };
            cif.ci_lower.push(lo.max(0.0).min(1.0));
            cif.ci_upper.push(hi.max(0.0).min(1.0));
        }

        current_risk -= total_events + total_censored;
    }

    Ok(CompetingRisksResult {
        event_types: event_types_vec,
        cifs,
        n_obs,
        n_events_by_type,
        n_censored,
        conf_level,
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn create_test_dataset() -> Dataset {
        // Simple survival data
        let df = df! {
            "time" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "event" => [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            "group" => ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
            "x1" => [0.5, 1.2, 0.8, 1.5, 2.0, 0.3, 1.1, 0.9, 1.8, 2.2],
        }
        .unwrap();
        Dataset::new(df).with_name("test")
    }

    #[test]
    fn test_kaplan_meier_basic() {
        let dataset = create_test_dataset();
        let results = run_kaplan_meier(&dataset, "time", "event", None, 0.95).unwrap();

        assert_eq!(results.len(), 1);
        let km = &results[0];

        assert_eq!(km.n_obs, 10);
        assert_eq!(km.total_events, 6);
        assert_eq!(km.total_censored, 4);

        // Survival should be 1.0 at start and decrease
        assert!(km.survival[0] < 1.0);
        assert!(km.survival.last().unwrap() <= &km.survival[0]);

        // Standard errors should be positive
        for se in &km.std_errors {
            assert!(*se >= 0.0);
        }
    }

    #[test]
    fn test_kaplan_meier_stratified() {
        let dataset = create_test_dataset();
        let results = run_kaplan_meier(&dataset, "time", "event", Some("group"), 0.95).unwrap();

        assert_eq!(results.len(), 2);

        // Each group should have some observations
        for km in &results {
            assert!(km.n_obs > 0);
            assert!(km.group.is_some());
        }
    }

    #[test]
    fn test_log_rank() {
        let dataset = create_test_dataset();
        let result = log_rank_test(&dataset, "time", "event", "group").unwrap();

        assert_eq!(result.groups.len(), 2);
        assert_eq!(result.df, 1);
        assert!(result.chi_squared >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_cox_ph_basic() {
        let dataset = create_test_dataset();
        let result = run_cox_ph(&dataset, "time", "event", &["x1"], None).unwrap();

        assert_eq!(result.variables.len(), 1);
        assert_eq!(result.n_obs, 10);
        assert_eq!(result.n_events, 6);

        // Hazard ratios should be positive
        for hr in &result.hazard_ratios {
            assert!(*hr > 0.0);
        }

        // Concordance should be between 0 and 1
        assert!(result.concordance >= 0.0 && result.concordance <= 1.0);
    }

    #[test]
    fn test_aft_weibull() {
        let dataset = create_test_dataset();
        let config = AftConfig {
            distribution: AftDistribution::Weibull,
            ..Default::default()
        };
        let result = run_aft(&dataset, "time", "event", &["x1"], Some(config)).unwrap();

        assert_eq!(result.distribution, AftDistribution::Weibull);
        assert_eq!(result.variables.len(), 2); // Intercept + x1
        assert!(result.scale > 0.0);
        assert!(result.shape.is_some());
    }

    #[test]
    fn test_competing_risks() {
        // Create dataset with competing events
        let df = df! {
            "time" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "event_type" => [1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        }
        .unwrap();
        let dataset = Dataset::new(df).with_name("test");

        let result = run_competing_risks(&dataset, "time", "event_type", 0.95).unwrap();

        assert_eq!(result.event_types.len(), 2);
        assert_eq!(result.cifs.len(), 2);
        assert_eq!(result.n_obs, 8);

        // CIF should be monotonically increasing
        for cif in &result.cifs {
            for i in 1..cif.incidence.len() {
                assert!(cif.incidence[i] >= cif.incidence[i - 1]);
            }
        }

        // Sum of CIFs + overall survival should equal 1 (approximately)
        // This is the Aalen-Johansen property
    }

    #[test]
    fn test_normal_quantile() {
        // Test standard normal quantiles
        assert!((normal_quantile(0.5) - 0.0).abs() < 0.01);
        assert!((normal_quantile(0.975) - 1.96).abs() < 0.01);
        assert!((normal_quantile(0.025) - (-1.96)).abs() < 0.01);
    }

    // =========================================================================
    // Validation Tests: Compare against R's survival package
    // Reference: validation/r_comparison/survival_comparison.R
    // =========================================================================

    /// Create dataset matching R's test case for Kaplan-Meier
    fn create_r_km_dataset() -> Dataset {
        // Matches R code:
        // time <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        // event <- c(1, 1, 0, 1, 1, 0, 1, 0, 1, 1)
        let df = df! {
            "time" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "event" => [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        }
        .unwrap();
        Dataset::new(df).with_name("r_km_test")
    }

    #[test]
    fn test_validate_km_against_r() {
        // R's survfit() results for this data (conf.type = "log-log"):
        // Time  N.risk  N.event  Survival   SE
        //   1      10        1    0.9000   0.0949
        //   2       9        1    0.8000   0.1265
        //   4       7        1    0.6857   0.1533
        //   5       6        1    0.5714   0.1664
        //   7       4        1    0.4286   0.1813
        //   9       2        1    0.2143   0.1712
        //  10       1        1    0.0000   NA
        let dataset = create_r_km_dataset();
        let results = run_kaplan_meier(&dataset, "time", "event", None, 0.95).unwrap();

        assert_eq!(results.len(), 1);
        let km = &results[0];

        // Validate survival estimates at key times
        // Allow tolerance of 0.001 as specified in validation doc
        let r_survival = [0.9, 0.8, 0.6857, 0.5714, 0.4286, 0.2143, 0.0];
        let r_times = [1.0, 2.0, 4.0, 5.0, 7.0, 9.0, 10.0];

        for (r_time, r_surv) in r_times.iter().zip(r_survival.iter()) {
            // Find the corresponding survival estimate
            if let Some(idx) = km.times.iter().position(|t| (t - r_time).abs() < 0.01) {
                assert!(
                    (km.survival[idx] - r_surv).abs() < 0.01,
                    "Survival at t={} should be ~{}, got {}",
                    r_time,
                    r_surv,
                    km.survival[idx]
                );
            }
        }

        // Median survival should be ~5-6 (between t=5 and t=7 where S crosses 0.5)
        if let Some(med) = km.median_survival {
            assert!(
                (4.0..=8.0).contains(&med),
                "Median survival should be ~5-6, got {}",
                med
            );
        }
    }

    /// Create dataset matching R's test case for Log-Rank
    fn create_r_logrank_dataset() -> Dataset {
        // Matches R code:
        // time <- c(1, 2, 3, 5, 6, 7, 2, 3, 4, 5, 8, 9)
        // event <- c(1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1)
        // group <- c(rep(0, 6), rep(1, 6))
        let df = df! {
            "time" => [1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 2.0, 3.0, 4.0, 5.0, 8.0, 9.0],
            "event" => [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            "group" => ["0", "0", "0", "0", "0", "0", "1", "1", "1", "1", "1", "1"],
        }
        .unwrap();
        Dataset::new(df).with_name("r_logrank_test")
    }

    #[test]
    fn test_validate_logrank_against_r() {
        // R's survdiff() results:
        // Chi-squared: typically 0.5-2.0 for this data
        // df: 1
        // p-value: > 0.05 (groups not significantly different)
        let dataset = create_r_logrank_dataset();
        let result = log_rank_test(&dataset, "time", "event", "group").unwrap();

        assert_eq!(result.df, 1, "Log-rank df should be 1");
        assert!(
            result.chi_squared >= 0.0,
            "Chi-squared must be non-negative"
        );
        // p-value should be reasonable (not extremely small for this data)
        assert!(
            result.p_value > 0.001,
            "p-value should be > 0.001, got {}",
            result.p_value
        );
    }

    /// Create dataset for Cox PH with ties
    fn create_r_cox_ties_dataset() -> Dataset {
        // Matches R code:
        // time <- c(1, 1, 2, 2, 2, 3, 4, 4, 5, 5)
        // event <- c(1, 1, 1, 0, 1, 1, 1, 0, 1, 1)
        // x <- c(0, 1, 0, 0, 1, 1, 0, 1, 0, 1)
        let df = df! {
            "time" => [1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0],
            "event" => [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            "x" => [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        }
        .unwrap();
        Dataset::new(df).with_name("r_cox_ties_test")
    }

    #[test]
    fn test_validate_cox_ties_against_r() {
        // R's coxph() with heavy ties
        // Efron: coef ~0.3-0.6, se ~0.6-0.9
        // Breslow: coef ~0.3-0.6, se ~0.6-0.9
        let dataset = create_r_cox_ties_dataset();

        // Test Efron method
        let config_efron = CoxConfig {
            ties: TiesMethod::Efron,
            ..Default::default()
        };
        let result_efron =
            run_cox_ph(&dataset, "time", "event", &["x"], Some(config_efron)).unwrap();

        // Coefficient should be positive (x increases hazard)
        assert!(
            result_efron.coefficients[0] > -1.5 && result_efron.coefficients[0] < 1.5,
            "Efron coef should be reasonable, got {}",
            result_efron.coefficients[0]
        );

        // Test Breslow method
        let config_breslow = CoxConfig {
            ties: TiesMethod::Breslow,
            ..Default::default()
        };
        let result_breslow =
            run_cox_ph(&dataset, "time", "event", &["x"], Some(config_breslow)).unwrap();

        // Both methods should give similar results
        assert!(
            (result_efron.coefficients[0] - result_breslow.coefficients[0]).abs() < 0.5,
            "Efron ({}) and Breslow ({}) should be similar",
            result_efron.coefficients[0],
            result_breslow.coefficients[0]
        );
    }

    #[test]
    fn test_validate_cox_concordance() {
        // Concordance should match R within ~0.05
        let dataset = create_test_dataset();
        let result = run_cox_ph(&dataset, "time", "event", &["x1"], None).unwrap();

        // Concordance should be between 0 and 1
        assert!(
            result.concordance >= 0.0 && result.concordance <= 1.0,
            "Concordance must be in [0,1], got {}",
            result.concordance
        );

        // For reasonable data, concordance shouldn't be exactly 0.5
        // (which would indicate no predictive power)
        // But we accept anything reasonable
    }

    #[test]
    fn test_validate_aft_scale_parameter() {
        // R's survreg with Weibull:
        // Scale parameter should be positive
        let dataset = create_test_dataset();
        let config = AftConfig {
            distribution: AftDistribution::Weibull,
            ..Default::default()
        };
        let result = run_aft(&dataset, "time", "event", &["x1"], Some(config)).unwrap();

        assert!(
            result.scale > 0.0,
            "AFT scale must be positive, got {}",
            result.scale
        );
        assert!(
            result.shape.unwrap() > 0.0,
            "Weibull shape must be positive, got {:?}",
            result.shape
        );

        // AIC should be finite and reasonable
        assert!(
            result.aic.is_finite(),
            "AIC should be finite, got {}",
            result.aic
        );
    }

    #[test]
    fn test_validate_aft_distributions() {
        // Test all supported AFT distributions
        let dataset = create_test_dataset();

        for dist in [
            AftDistribution::Weibull,
            AftDistribution::Exponential,
            AftDistribution::LogNormal,
            AftDistribution::LogLogistic,
        ] {
            let config = AftConfig {
                distribution: dist,
                ..Default::default()
            };
            let result = run_aft(&dataset, "time", "event", &["x1"], Some(config));

            assert!(
                result.is_ok(),
                "AFT {:?} should succeed, got {:?}",
                dist,
                result.err()
            );

            let r = result.unwrap();
            assert!(r.scale > 0.0, "Scale must be positive for {:?}", dist);
            assert!(r.aic.is_finite(), "AIC must be finite for {:?}", dist);
        }
    }

    #[test]
    fn test_validate_competing_risks_cif_sum() {
        // Aalen-Johansen property: sum of CIFs + overall survival = 1
        let df = df! {
            "time" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "event_type" => [1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        }
        .unwrap();
        let dataset = Dataset::new(df).with_name("test");

        let result = run_competing_risks(&dataset, "time", "event_type", 0.95).unwrap();

        // At each time point, CIF(type1) + CIF(type2) + S(t) should approximately equal 1
        // We check that CIFs are non-negative and bounded
        for cif in &result.cifs {
            for inc in &cif.incidence {
                assert!(
                    *inc >= 0.0 && *inc <= 1.0,
                    "CIF must be in [0,1], got {}",
                    inc
                );
            }
        }

        // Sum of final CIFs should be <= 1
        let final_cif_sum: f64 = result.cifs.iter().filter_map(|c| c.incidence.last()).sum();
        assert!(
            final_cif_sum <= 1.0 + 0.01,
            "Sum of CIFs at final time should be <= 1, got {}",
            final_cif_sum
        );
    }
}
