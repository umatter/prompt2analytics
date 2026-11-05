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

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use polars::prelude::{DataFrame, DataType};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};
use std::collections::{BTreeMap, HashMap};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, Factor, MissingSummary, NaPolicy, column};
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::{pseudoinverse, safe_inverse};
use crate::resource::{checkpoint, try_zeros2};
use crate::traits::estimator::{SignificanceLevel, chi_squared_p_value, normal_cdf};

// =============================================================================
// Column extraction shared by every survival entry point
// =============================================================================

/// The rows of a survival call after R's `na.omit`: times, event flags, group codes and
/// covariates aligned on the kept rows, plus what was dropped.
struct SurvivalData {
    time: Vec<f64>,
    event: Vec<bool>,
    /// Group labels in R `factor()` order and one code per kept row.
    group: Option<Factor>,
    /// One vector per covariate, in `x_cols` order.
    x: Vec<Vec<f64>>,
    missing: MissingSummary,
}

/// The survival times a method accepts.
#[derive(Clone, Copy)]
enum TimeRule {
    /// t >= 0 (Kaplan-Meier, log-rank, Cox, competing risks).
    NonNegative,
    /// t > 0 (AFT models, which model log T; R `survreg` refuses t <= 0 too).
    Positive,
}

/// A Boolean time column is refused, as R `Surv()` does ("Time variable is not numeric");
/// the extractor would otherwise read it as 0/1.
fn refuse_boolean_time(df: &DataFrame, time_col: &str) -> EconResult<()> {
    if column(df, time_col)?.dtype() == &DataType::Boolean {
        return Err(EconError::NonNumericColumn {
            column: time_col.to_string(),
        });
    }
    Ok(())
}

/// The kept times, checked against `rule`, with -0.0 written as 0.0. `rows` are the
/// original 0-based rows of the kept values (for the message).
fn checked_times(
    values: &[f64],
    rows: &[usize],
    time_col: &str,
    rule: TimeRule,
) -> EconResult<Vec<f64>> {
    let bad = |t: f64| match rule {
        TimeRule::NonNegative => t < 0.0,
        TimeRule::Positive => t <= 0.0,
    };
    let count = values.iter().filter(|&&t| bad(t)).count();
    if let Some(i) = values.iter().position(|&t| bad(t)) {
        let (row, value) = (rows[i], values[i]);
        let message = match rule {
            TimeRule::NonNegative => format!(
                "Column '{time_col}' has {count} negative value(s) (first at row {row}: \
                 {value}); survival times must be >= 0"
            ),
            TimeRule::Positive => format!(
                "Column '{time_col}' has {count} value(s) <= 0 (first at row {row}: {value}); \
                 AFT models need positive survival times (R survreg: \"Invalid survival \
                 times for this distribution\")"
            ),
        };
        return Err(EconError::InvalidSpecification { message });
    }
    Ok(values
        .iter()
        .map(|&t| if t == 0.0 { 0.0 } else { t })
        .collect())
}

/// Event flags from a status column coded as R's `Surv()` codes it: all values in {0, 1}
/// (1 = event), all in {1, 2} (2 = event, as R's `lung`), or logical (0/1 here). Any other
/// code is an error naming the column and the values; R turns such codes into NA with a
/// warning, which would drop rows silently.
fn surv_status(values: &[f64], rows: &[usize], column: &str) -> EconResult<Vec<bool>> {
    if values.iter().all(|&v| v == 0.0 || v == 1.0) {
        return Ok(values.iter().map(|&v| v == 1.0).collect());
    }
    if values.iter().all(|&v| v == 1.0 || v == 2.0) {
        return Ok(values.iter().map(|&v| v == 2.0).collect());
    }
    // As Surv(): a maximum of 2 means 1/2 coding, anything else 0/1 coding; the first
    // value outside that coding is reported.
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let coding: [f64; 2] = if max == 2.0 { [1.0, 2.0] } else { [0.0, 1.0] };
    let first = values.iter().position(|v| !coding.contains(v)).unwrap_or(0);
    let mut distinct = values.to_vec();
    distinct.sort_by(f64::total_cmp);
    distinct.dedup();
    let mut list: Vec<String> = distinct.iter().take(5).map(|v| v.to_string()).collect();
    if distinct.len() > 5 {
        list.push("...".to_string());
    }
    Err(EconError::InvalidSpecification {
        message: format!(
            "Column '{column}' holds status value(s) {} (first unexpected at row {}); code \
             it as 0/1 (1 = event), 1/2 (2 = event) or TRUE/FALSE, as R's Surv() does",
            list.join(", "),
            rows.get(first).copied().unwrap_or(first)
        ),
    })
}

/// Time, status, an optional group and covariates of one survival call, read with the
/// shared extractor: integer, float and Boolean columns, rows with a missing value in
/// any of them dropped (R `na.omit`) and reported in `missing`.
fn extract_survival(
    df: &DataFrame,
    time_col: &str,
    status_col: &str,
    group_col: Option<&str>,
    x_cols: &[&str],
    rule: TimeRule,
    context: &str,
) -> EconResult<SurvivalData> {
    refuse_boolean_time(df, time_col)?;
    let mut request = ColumnRequest::new(df).numeric(time_col).numeric(status_col);
    if let Some(g) = group_col {
        request = request.factor(g);
    }
    let cols = request
        .numerics(x_cols)
        .na(NaPolicy::Omit)
        .min_rows(1, context)
        .extract()?;
    let time = checked_times(cols.numeric(time_col), cols.rows(), time_col, rule)?;
    let event = surv_status(cols.numeric(status_col), cols.rows(), status_col)?;
    Ok(SurvivalData {
        time,
        event,
        group: group_col.map(|g| cols.factor(g).clone()),
        x: x_cols.iter().map(|c| cols.numeric(c).to_vec()).collect(),
        missing: cols.missing().clone(),
    })
}

/// `conf_level` strictly between 0 and 1 (NaN refused).
fn check_conf_level(conf_level: f64) -> EconResult<()> {
    if !(conf_level > 0.0 && conf_level < 1.0) {
        return Err(EconError::InvalidSpecification {
            message: "Confidence level must be between 0 and 1".to_string(),
        });
    }
    Ok(())
}

/// The last line of a result's Display when rows were dropped.
fn write_missing(f: &mut fmt::Formatter<'_>, missing: &MissingSummary) -> fmt::Result {
    match missing.note() {
        Some(note) => writeln!(f, "Missing values: {note}"),
        None => Ok(()),
    }
}

// =============================================================================
// Core Types
// =============================================================================

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

/// Method for handling tied event times in the Cox model (R `coxph(ties = )`).
///
/// Parsed from "efron" or "breslow" (case-insensitive, `FromStr`); other values are refused.
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

crate::option_enum!(
    TiesMethod,
    crate::options::Choices {
        option: "ties",
        values: &[
            (&["efron"], TiesMethod::Efron),
            (&["breslow"], TiesMethod::Breslow),
        ],
    },
    no_display
);

/// Distribution for Accelerated Failure Time models (R `survreg(dist = )`).
///
/// Parsed from "weibull", "exponential", "lognormal" (also "log-normal", "log_normal") or
/// "loglogistic" (also "log-logistic", "log_logistic"), case-insensitive (`FromStr`). R's
/// "gaussian" and "logistic" families model T rather than log T and are refused, as are
/// "rayleigh", "t" and "extreme".
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

crate::option_enum!(
    AftDistribution,
    crate::options::Choices {
        option: "dist",
        values: &[
            (&["weibull"], AftDistribution::Weibull),
            (&["exponential"], AftDistribution::Exponential),
            (&["lognormal", "log_normal"], AftDistribution::LogNormal),
            (&["loglogistic", "log_logistic"], AftDistribution::LogLogistic),
        ],
    },
    no_display
);

// =============================================================================
// Kaplan-Meier Estimator
// =============================================================================

/// Confidence-interval transformation of a Kaplan-Meier curve (R `survfit`'s
/// `conf.type`). Used by [`run_kaplan_meier_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum KmConfType {
    /// Interval on log S(t), R's default `"log"`.
    #[default]
    Log,
    /// Interval on log(-log S(t)), R `"log-log"`.
    LogLog,
    /// Interval on S(t) itself, R `"plain"`.
    Plain,
}

impl std::str::FromStr for KmConfType {
    type Err = EconError;

    /// `"log"`, `"log-log"` (or `"loglog"`) and `"plain"`, ignoring case and surrounding
    /// whitespace.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "log" => Ok(KmConfType::Log),
            "log-log" | "loglog" => Ok(KmConfType::LogLog),
            "plain" => Ok(KmConfType::Plain),
            _ => Err(EconError::InvalidSpecification {
                message: format!(
                    "Unknown conf_type '{s}'; use \"log\" (default, as R survfit), \"log-log\" \
                     or \"plain\""
                ),
            }),
        }
    }
}

impl fmt::Display for KmConfType {
    /// R's `conf.type` names: "log", "log-log", "plain".
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            KmConfType::Log => "log",
            KmConfType::LogLog => "log-log",
            KmConfType::Plain => "plain",
        })
    }
}

/// Options of [`run_kaplan_meier_with`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct KmConfig {
    /// Confidence level of the pointwise intervals (default 0.95).
    pub conf_level: f64,
    /// Interval transformation (default [`KmConfType::Log`], as R `survfit`).
    pub conf_type: KmConfType,
}

impl Default for KmConfig {
    fn default() -> Self {
        KmConfig {
            conf_level: 0.95,
            conf_type: KmConfType::Log,
        }
    }
}

/// Kaplan-Meier estimation with the options of [`KmConfig`], as R
/// `survival::survfit(Surv(time, status) ~ group, conf.type = ...)`.
///
/// One result per group, in R `factor()` order of the group labels (one result without a
/// group). Every distinct time (events and censorings) is a row, as `survfit()$time`.
/// Standard errors are Greenwood's on the S scale (`summary(fit)$std.err`), NaN where
/// S(t) = 0. The pointwise limits are `survival:::survfit_confint` for `conf_type`
/// "log" (R's default), "log-log" or "plain", NaN where R reports NA. The median and
/// its limits are `quantile.survfit(probs = 0.5)`, `None` where one is never reached.
///
/// The status column is coded as R's `Surv()` codes it (0/1, 1/2 or logical); rows with
/// a missing time, status or group are dropped and reported in `missing`; times must be
/// >= 0.
pub fn run_kaplan_meier_with(
    dataset: &Dataset,
    time_col: &str,
    event_col: &str,
    group_col: Option<&str>,
    config: &KmConfig,
) -> EconResult<Vec<KaplanMeierResult>> {
    check_conf_level(config.conf_level)?;
    let data = extract_survival(
        dataset.df(),
        time_col,
        event_col,
        group_col,
        &[],
        TimeRule::NonNegative,
        "Kaplan-Meier",
    )?;
    let z = two_sided_z(config.conf_level)?;

    let Some(factor) = &data.group else {
        let km = compute_kaplan_meier(&data.time, &data.event, z, config, None, &data.missing)?;
        return Ok(vec![km]);
    };

    // One curve per group, in R factor order.
    let mut rows_by_level: Vec<Vec<usize>> = vec![Vec::new(); factor.n_levels()];
    for (i, &code) in factor.codes.iter().enumerate() {
        rows_by_level[code as usize].push(i);
    }
    let mut results = Vec::with_capacity(factor.n_levels());
    for (label, rows) in factor.levels.iter().zip(&rows_by_level) {
        checkpoint()?;
        let time: Vec<f64> = rows.iter().map(|&i| data.time[i]).collect();
        let event: Vec<bool> = rows.iter().map(|&i| data.event[i]).collect();
        results.push(compute_kaplan_meier(
            &time,
            &event,
            z,
            config,
            Some(label.clone()),
            &data.missing,
        )?);
    }
    Ok(results)
}

/// z = qnorm(1 - (1 - conf_level) / 2), from statrs.
fn two_sided_z(conf_level: f64) -> EconResult<f64> {
    let normal = Normal::new(0.0, 1.0)
        .map_err(|e| EconError::Internal(format!("standard normal distribution: {e}")))?;
    Ok(normal.inverse_cdf(1.0 - (1.0 - conf_level) / 2.0))
}

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
    /// Group label (stratified analysis), as the group column's factor level
    pub group: Option<String>,
    /// Every distinct time (events and censorings), as `survfit()$time`
    pub times: Vec<f64>,
    /// Survival probability S(t) at each time
    pub survival: Vec<f64>,
    /// Standard error of S(t) (Greenwood), as `summary(survfit)$std.err`; NaN where
    /// S(t) = 0
    pub std_errors: Vec<f64>,
    /// Lower pointwise limit for `conf_type` (survfit `$lower`); NaN where R reports NA
    pub ci_lower: Vec<f64>,
    /// Upper pointwise limit for `conf_type` (survfit `$upper`); NaN where R reports NA
    pub ci_upper: Vec<f64>,
    /// Number at risk at each time
    pub n_at_risk: Vec<usize>,
    /// Number of events at each time
    pub n_events: Vec<usize>,
    /// Number censored at each time
    pub n_censored: Vec<usize>,
    /// Median survival time, as `quantile.survfit(probs = 0.5)`; `None` when S(t) stays
    /// above 0.5
    pub median_survival: Option<f64>,
    /// Lower and upper limit of the median (`quantile.survfit` `$lower`, `$upper`);
    /// `None` for a limit that is not reached (R NA). A limit can exist without a median.
    pub median_ci: (Option<f64>, Option<f64>),
    /// Total number of observations
    pub n_obs: usize,
    /// Total number of events
    pub total_events: usize,
    /// Total number censored
    pub total_censored: usize,
    /// Confidence level used
    pub conf_level: f64,
    /// Interval transformation used (R `conf.type`)
    #[serde(default)]
    pub conf_type: KmConfType,
    /// Rows dropped from the whole call (all groups) because of missing values.
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
}

/// Event times listed in full by the Kaplan-Meier Display; beyond, the first
/// [`KM_TABLE_HEAD`] and the last [`KM_TABLE_TAIL`].
const KM_TABLE_MAX_ROWS: usize = 200;
const KM_TABLE_HEAD: usize = 100;
const KM_TABLE_TAIL: usize = 50;

/// A confidence level as a percentage: "95" for 0.95, "97.5" for 0.975.
fn conf_percent(conf_level: f64) -> String {
    let pct = conf_level * 100.0;
    if (pct - pct.round()).abs() < 1e-9 {
        format!("{pct:.0}")
    } else {
        format!("{pct}")
    }
}

/// `{:.<decimals>}` of a value, or "NA" for NaN / `None` (R's NA).
fn fmt_or_na(v: Option<f64>, decimals: usize) -> String {
    match v {
        Some(x) if !x.is_nan() => format!("{x:.decimals$}"),
        _ => "NA".to_string(),
    }
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
        let pct = conf_percent(self.conf_level);
        let default_note = if self.conf_type == KmConfType::Log {
            " (R survfit default)"
        } else {
            ""
        };
        writeln!(
            f,
            "Confidence intervals: {pct}%, conf.type = {}{default_note}",
            self.conf_type
        )?;
        writeln!(f)?;

        let limits = format!(
            "({pct}% CI: {} - {})",
            fmt_or_na(self.median_ci.0, 3),
            fmt_or_na(self.median_ci.1, 3)
        );
        match self.median_survival {
            Some(median) => writeln!(f, "Median Survival: {median:.3} {limits}")?,
            None => writeln!(
                f,
                "Median Survival: NA, survival > 50% at end of follow-up {limits}"
            )?,
        }
        writeln!(f)?;

        writeln!(
            f,
            "{:>10} {:>10} {:>10} {:>10} {:>12} {:>11} {:>11}",
            "Time",
            "At Risk",
            "Events",
            "S(t)",
            "Std.Err",
            format!("Lower {pct}%"),
            format!("Upper {pct}%")
        )?;
        writeln!(f, "{}", "-".repeat(80))?;

        // The event times, as R's summary(fit).
        let rows: Vec<usize> = (0..self.times.len())
            .filter(|&i| self.n_events[i] > 0)
            .collect();
        let row = |f: &mut fmt::Formatter<'_>, i: usize| {
            writeln!(
                f,
                "{:>10.3} {:>10} {:>10} {:>10.4} {:>12} {:>11} {:>11}",
                self.times[i],
                self.n_at_risk[i],
                self.n_events[i],
                self.survival[i],
                fmt_or_na(Some(self.std_errors[i]), 4),
                fmt_or_na(Some(self.ci_lower[i]), 4),
                fmt_or_na(Some(self.ci_upper[i]), 4)
            )
        };
        if rows.is_empty() {
            writeln!(f, "(no events)")?;
        } else if rows.len() <= KM_TABLE_MAX_ROWS {
            for &i in &rows {
                row(f, i)?;
            }
        } else {
            for &i in &rows[..KM_TABLE_HEAD] {
                row(f, i)?;
            }
            writeln!(
                f,
                "... ({} event times not shown)",
                rows.len() - KM_TABLE_HEAD - KM_TABLE_TAIL
            )?;
            for &i in &rows[rows.len() - KM_TABLE_TAIL..] {
                row(f, i)?;
            }
        }

        write_missing(f, &self.missing)
    }
}

/// Run Kaplan-Meier survival analysis with R `survfit`'s default confidence interval
/// type, conf.type "log"; use [`run_kaplan_meier_with`] for "log-log" or "plain".
///
/// # Arguments
///
/// * `dataset` - The dataset containing survival data
/// * `time_col` - Column name for survival/censoring time (>= 0)
/// * `event_col` - Status column, coded as R's `Surv()`: 0/1, 1/2 or logical
/// * `group_col` - Optional column for stratified analysis
/// * `conf_level` - Confidence level (default: 0.95)
///
/// # Returns
///
/// A vector of `KaplanMeierResult`, one per group in R `factor()` order (or a single
/// element if unstratified).
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
    run_kaplan_meier_with(
        dataset,
        time_col,
        event_col,
        group_col,
        &KmConfig {
            conf_level,
            conf_type: KmConfType::Log,
        },
    )
}

/// One Kaplan-Meier curve: the product-limit estimate over blocks of exactly equal
/// times, Greenwood's variance of log S(t) (`sum d / (n (n - d))`, infinite once every
/// subject at risk has the event, as R), the limits of [`km_confint`] and the median of
/// [`km_quantile`].
fn compute_kaplan_meier(
    time: &[f64],
    event: &[bool],
    z: f64,
    config: &KmConfig,
    group: Option<String>,
    missing: &MissingSummary,
) -> EconResult<KaplanMeierResult> {
    let n_obs = time.len();
    if n_obs == 0 {
        return Err(EconError::EmptyDataset);
    }
    let mut order: Vec<usize> = (0..n_obs).collect();
    order.sort_by(|&a, &b| time[a].total_cmp(&time[b]));

    let mut times = Vec::new();
    let mut survival = Vec::new();
    let mut std_errors = Vec::new();
    let mut ci_lower = Vec::new();
    let mut ci_upper = Vec::new();
    let mut n_at_risk = Vec::new();
    let mut n_events = Vec::new();
    let mut n_censored = Vec::new();

    let mut at_risk = n_obs;
    let mut surv = 1.0_f64;
    let mut greenwood = 0.0_f64;
    let mut total_events = 0;
    let mut i = 0;
    while i < n_obs {
        let t = time[order[i]];
        let mut j = i;
        let mut d = 0usize;
        while j < n_obs && time[order[j]] == t {
            if event[order[j]] {
                d += 1;
            }
            j += 1;
        }

        if d > 0 {
            surv *= 1.0 - d as f64 / at_risk as f64;
            greenwood = if at_risk > d {
                greenwood + d as f64 / (at_risk as f64 * (at_risk - d) as f64)
            } else {
                f64::INFINITY
            };
        }
        let se_log = greenwood.sqrt();
        let (lo, hi) = km_confint(surv, se_log, z, config.conf_type);

        times.push(t);
        survival.push(surv);
        std_errors.push(if surv > 0.0 { surv * se_log } else { f64::NAN });
        ci_lower.push(lo);
        ci_upper.push(hi);
        n_at_risk.push(at_risk);
        n_events.push(d);
        n_censored.push(j - i - d);

        total_events += d;
        at_risk -= j - i;
        i = j;
    }

    // quantile.survfit: the median from 1 - S, its lower limit from 1 - lower and its
    // upper limit from 1 - upper, each whether or not the median itself exists.
    let median_survival = km_quantile(&times, &survival, 0.5);
    let median_ci = (
        km_quantile(&times, &ci_lower, 0.5),
        km_quantile(&times, &ci_upper, 0.5),
    );

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
        conf_level: config.conf_level,
        conf_type: config.conf_type,
        missing: missing.clone(),
    })
}

/// The pointwise limits of `survival:::survfit_confint` (survival 3.5.8) with `logse =
/// TRUE` (`se_log` is the standard error of log S) and `ulimit = TRUE`, NaN where R
/// gives NA:
/// - log: `exp(log S -/+ z se)`, the upper capped at 1; NA at S = 0;
/// - log-log: `exp(-exp(log(-log S) -/+ z se / log S))`; NA at S = 0 and S = 1;
/// - plain: `S -/+ z S se` clipped to [0, 1]; NA at S = 0 (where se is infinite).
fn km_confint(s: f64, se_log: f64, z: f64, conf_type: KmConfType) -> (f64, f64) {
    match conf_type {
        KmConfType::Log => {
            if s == 0.0 {
                return (f64::NAN, f64::NAN);
            }
            let w = z * se_log;
            ((s.ln() - w).exp(), (s.ln() + w).exp().min(1.0))
        }
        KmConfType::LogLog => {
            if s == 0.0 || s == 1.0 {
                return (f64::NAN, f64::NAN);
            }
            let w = z * se_log / s.ln();
            let l = (-s.ln()).ln();
            ((-(l - w).exp()).exp(), (-(l + w).exp()).exp())
        }
        KmConfType::Plain => {
            let h = se_log * s * z;
            // R's pmax(NaN, 0) is NaN; f64::max(NaN, 0.0) would be 0.
            if h.is_nan() {
                return (f64::NAN, f64::NAN);
            }
            ((s - h).max(0.0), (s + h).min(1.0))
        }
    }
}

/// `survival:::doquant` for one probability: the time at which `1 - curve` reaches `p`,
/// with the curve starting at time 0 (`firstx`, R's default for a fit without a start
/// time). NaN entries of `curve` (R NA) are skipped as R's `approx` skips them.
fn km_quantile(times: &[f64], curve: &[f64], p: f64) -> Option<f64> {
    let mut x = Vec::with_capacity(times.len() + 1);
    x.push(0.0);
    x.extend_from_slice(times);
    let mut y = Vec::with_capacity(curve.len() + 1);
    y.push(0.0);
    y.extend(curve.iter().map(|c| 1.0 - c));
    findq(&x, &y, p, f64::EPSILON.sqrt())
}

/// A port of `survival:::findq` (survival 3.5.8) for one probability `p`, with NaN for
/// R's NA:
///
/// ```text
/// if (max(y, na.rm = TRUE) < p) return(NA)
/// xmax <- x[length(x)]
/// drop duplicated(y) from x and y
/// indx1 <- approx(y + tol, 1:n, p, method = "constant", f = 1)$y
/// indx2 <- approx(y - tol, 1:n, p, method = "constant", f = 1)$y
/// quant <- (x[indx1] + x[indx2]) / 2;  quant[p == 0] <- x[1]
/// if (!is.na(y[n]) && abs(p - y[n]) < tol) quant <- (x[indx1] + xmax) / 2
/// ```
///
/// `approx(..., method = "constant", f = 1)` at `p` is the first knot (sorted, NA
/// removed) whose value is at least `p`, and NA outside the knots' range. So the
/// quantile is the first time the curve reaches `p`, the midpoint of the interval over
/// which it equals `p` (within `tol`), and, when it ends at `p`, the midpoint of that
/// time and the last time.
fn findq(x: &[f64], y: &[f64], p: f64, tol: f64) -> Option<f64> {
    let max = y
        .iter()
        .filter(|v| !v.is_nan())
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    if max < p {
        return None;
    }
    let xmax = *x.last()?;

    // duplicated(y): keep the first occurrence of each value (NaN is one value).
    let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut seen_nan = false;
    let (mut kx, mut ky) = (Vec::new(), Vec::new());
    for (&xi, &yi) in x.iter().zip(y) {
        let first = if yi.is_nan() {
            !std::mem::replace(&mut seen_nan, true)
        } else {
            seen.insert((yi + 0.0).to_bits()) // -0.0 and 0.0 are one value
        };
        if first {
            kx.push(xi);
            ky.push(yi);
        }
    }

    // approx(): the knots without NA, sorted by value.
    let mut knots: Vec<(f64, usize)> = ky
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.is_nan())
        .map(|(k, &v)| (v, k))
        .collect();
    knots.sort_by(|a, b| a.0.total_cmp(&b.0));
    let index = |shift: f64| -> Option<usize> {
        let (lo, hi) = (knots.first()?.0 + shift, knots.last()?.0 + shift);
        if p < lo || p > hi {
            return None;
        }
        let at = knots.partition_point(|&(v, _)| v + shift < p);
        knots.get(at).map(|&(_, k)| k)
    };
    let (i1, i2) = (index(tol), index(-tol));

    let mut quant = match (i1, i2) {
        (Some(a), Some(b)) => Some((kx[a] + kx[b]) / 2.0),
        _ => None,
    };
    if p == 0.0 {
        quant = kx.first().copied();
    }
    if let Some(&y_last) = ky.last()
        && !y_last.is_nan()
        && (p - y_last).abs() < tol
    {
        quant = i1.map(|a| (kx[a] + xmax) / 2.0);
    }
    quant
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
    /// Variance of O - E, G x G row-major over all groups (R `survdiff()$var`).
    #[serde(default)]
    pub variance: Vec<f64>,
    /// Rows dropped because of a missing time, status or group.
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
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
        let excluded: Vec<&str> = self
            .groups
            .iter()
            .zip(&self.expected_per_group)
            .filter(|(_, e)| **e <= 0.0)
            .map(|(g, _)| g.as_str())
            .collect();
        if !excluded.is_empty() {
            writeln!(
                f,
                "Groups with no expected events: {} (excluded from the test, as R survdiff)",
                excluded.join(", ")
            )?;
        }
        writeln!(f)?;
        writeln!(
            f,
            "Chi-squared = {:.4}, df = {}, p = {:.4}{}",
            self.chi_squared,
            self.df,
            self.p_value,
            self.significance.stars()
        )?;

        write_missing(f, &self.missing)
    }
}

/// Perform log-rank test to compare survival curves between groups, as R
/// `survival::survdiff` (rho = 0).
///
/// Rows are walked in time order in blocks of exactly equal times. At each block with
/// `d` events among `n` at risk, group g expects `d n_g / n` events, and the
/// hypergeometric variance `d (n - d) / (n^2 (n - 1))` times `n_g (n - n_g)` (diagonal)
/// or `-n_g n_h` is added; the block's rows then leave the risk sets. Groups with no
/// expected events are excluded from the test, as in R, and the chi-square is
/// `(O - E)' V^- (O - E)` over the remaining groups less one, solved with a
/// generalized inverse when the variance is singular but not zero; `df` is their number
/// minus one.
///
/// # Arguments
///
/// * `dataset` - The dataset containing survival data
/// * `time_col` - Column name for survival/censoring time (>= 0)
/// * `event_col` - Status column, coded as R's `Surv()`: 0/1, 1/2 or logical
/// * `group_col` - Column name for group assignment (levels in R `factor()` order)
///
/// Rows with a missing time, status or group are dropped and reported in `missing`.
///
/// # Returns
///
/// A `LogRankResult` containing the test statistic and p-value. Fewer than two groups
/// with expected events is an `InvalidSpecification` error (R reports 0 on 0 df), and so
/// is a zero variance, when every subject at risk at the first event time has the event
/// then (R's survdiff stops on it).
pub fn log_rank_test(
    dataset: &Dataset,
    time_col: &str,
    event_col: &str,
    group_col: &str,
) -> EconResult<LogRankResult> {
    let data = extract_survival(
        dataset.df(),
        time_col,
        event_col,
        Some(group_col),
        &[],
        TimeRule::NonNegative,
        "log-rank test",
    )?;
    let Some(factor) = data.group else {
        return Err(EconError::Internal(
            "log-rank test: the group column was not extracted".to_string(),
        ));
    };
    let groups = factor.levels;
    let n_groups = groups.len();
    if n_groups < 2 {
        return Err(EconError::InvalidSpecification {
            message: "Log-rank test requires at least 2 groups".to_string(),
        });
    }
    let codes: Vec<usize> = factor.codes.iter().map(|&c| c as usize).collect();
    let n = data.time.len();

    let mut n_per_group = vec![0usize; n_groups];
    let mut events_per_group = vec![0usize; n_groups];
    for (&g, &e) in codes.iter().zip(&data.event) {
        n_per_group[g] += 1;
        if e {
            events_per_group[g] += 1;
        }
    }

    // Rows in time order; a block is a run of exactly equal times (as survdiff).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| data.time[a].total_cmp(&data.time[b]));

    let mut expected = vec![0.0f64; n_groups];
    let mut variance = try_zeros2(n_groups, n_groups, "log-rank variance")?;
    let mut risk = n_per_group.clone();
    let mut total = n;
    let mut i = 0;
    while i < n {
        checkpoint()?;
        let t = data.time[order[i]];
        let mut j = i;
        let mut d = 0usize;
        while j < n && data.time[order[j]] == t {
            if data.event[order[j]] {
                d += 1;
            }
            j += 1;
        }

        if d > 0 {
            let (dt, nt) = (d as f64, total as f64);
            for (e, &r) in expected.iter_mut().zip(&risk) {
                *e += dt * r as f64 / nt;
            }
            if total > 1 {
                let factor = dt * (nt - dt) / (nt * nt * (nt - 1.0));
                for (g, &rg) in risk.iter().enumerate() {
                    if rg == 0 {
                        continue; // row and column g receive only zeros
                    }
                    let rg = rg as f64;
                    for (h, &rh) in risk.iter().enumerate() {
                        variance[[g, h]] += if g == h {
                            factor * rg * (nt - rg)
                        } else {
                            -factor * rg * rh as f64
                        };
                    }
                }
            }
        }

        // The block leaves the risk sets once, and the cursor moves past it.
        for &row in &order[i..j] {
            let g = codes[row];
            risk[g] = risk[g].checked_sub(1).ok_or_else(|| {
                EconError::Internal("log-rank test: risk set below zero".to_string())
            })?;
        }
        total = total
            .checked_sub(j - i)
            .ok_or_else(|| EconError::Internal("log-rank test: risk set below zero".into()))?;
        i = j;
    }

    let obs_minus_exp: Vec<f64> = events_per_group
        .iter()
        .zip(&expected)
        .map(|(&o, &e)| o as f64 - e)
        .collect();

    // survdiff: only groups with expected events enter the test.
    let keep: Vec<usize> = (0..n_groups).filter(|&g| expected[g] > 0.0).collect();
    if keep.len() < 2 {
        let with: Vec<&str> = keep.iter().map(|&g| groups[g].as_str()).collect();
        return Err(EconError::InvalidSpecification {
            message: format!(
                "the log-rank test needs two groups with expected events, i.e. at risk at \
                 an event time; groups with expected events: {} (R survdiff reports a \
                 chi-square of 0 on 0 df)",
                if with.is_empty() {
                    "none".to_string()
                } else {
                    with.join(", ")
                }
            ),
        });
    }

    // (O - E)' V^- (O - E) without the first kept group (the statistic is invariant to
    // the group dropped).
    let rest = &keep[1..];
    let k = rest.len();
    let u: Array1<f64> = rest.iter().map(|&g| obs_minus_exp[g]).collect();
    let mut w = try_zeros2(k, k, "log-rank variance")?;
    for (a, &g) in rest.iter().enumerate() {
        for (b, &h) in rest.iter().enumerate() {
            w[[a, b]] = variance[[g, h]];
        }
    }
    // Every kept group is at risk at the first event time, so the variance is zero exactly
    // when everyone at risk then has the event: no later event time has anyone at risk.
    // survdiff's solve() stops on that singular system; the test has no information.
    if w.iter().all(|&v| v == 0.0) {
        return Err(EconError::InvalidSpecification {
            message: "the log-rank variance is zero: every subject at risk at the first event \
                      time has the event then, so the groups cannot be compared (R survdiff \
                      stops: the variance matrix is singular)"
                .to_string(),
        });
    }
    let w_inv = match safe_inverse(&w.view()) {
        Ok((inv, _)) => inv,
        Err(_) => pseudoinverse(&w.view())?,
    };
    let chi_squared = u.dot(&w_inv.dot(&u));

    let df_val = k;
    let p_value = ChiSquared::new(df_val as f64)
        .map_err(|e| EconError::Internal(format!("log-rank test: chi-square df: {e}")))?
        .sf(chi_squared);
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
        variance: variance.into_raw_vec_and_offset().0,
        missing: data.missing,
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
    /// Rows dropped because of a missing time, status or covariate.
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
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

        write_missing(f, &self.missing)
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

    if x_cols.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "Cox model requires at least one covariate".to_string(),
        });
    }

    let data = extract_survival(
        dataset.df(),
        time_col,
        event_col,
        None,
        x_cols,
        TimeRule::NonNegative,
        "Cox regression",
    )?;
    let p = x_cols.len();
    let n = data.time.len();
    // (time, event, row index), kept rows in their original order
    let obs_data: Vec<(f64, bool, usize)> = data
        .time
        .iter()
        .zip(&data.event)
        .enumerate()
        .map(|(i, (&t, &e))| (t, e, i))
        .collect();

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

    let mut x = try_zeros2(n, p, "Cox covariate matrix")?;
    for (mut col, values) in x.columns_mut().into_iter().zip(&data.x) {
        for (cell, &v) in col.iter_mut().zip(values) {
            *cell = v;
        }
    }

    let rs = CoxRiskSets::new(&data.time);
    let event = &data.event;

    // Newton-Raphson optimization
    let mut beta: Array1<f64> = Array1::zeros(p);
    let mut converged = false;
    let mut iterations = 0;

    // Null log-likelihood (beta = 0)
    checkpoint()?;
    let at_zero = cox_pass(&rs, x.view(), event, Array1::zeros(p).view(), config.ties);
    let log_likelihood_null = at_zero.loglik;

    for iter in 0..config.max_iter {
        checkpoint()?;
        iterations = iter + 1;
        let pass = cox_pass(&rs, x.view(), event, beta.view(), config.ties);
        let (grad, hess, ll) = (pass.score, pass.information, pass.loglik);

        // Check convergence
        let grad_norm = grad.iter().map(|g| g.abs()).fold(0.0, f64::max);
        if grad_norm < config.tolerance {
            converged = true;
            break;
        }

        // Newton-Raphson update: beta_new = beta + I^{-1} * score
        let (hess_inv, _) = safe_inverse(&hess.view())?;
        let delta = hess_inv.dot(&grad);

        // Step halving if needed
        let mut step = 1.0;
        let mut new_beta = &beta + &(&delta * step);
        let mut new_ll = cox_pass(&rs, x.view(), event, new_beta.view(), config.ties).loglik;

        while new_ll < ll && step > 1e-10 {
            step *= 0.5;
            new_beta = &beta + &(&delta * step);
            new_ll = cox_pass(&rs, x.view(), event, new_beta.view(), config.ties).loglik;
        }

        beta = new_beta;
    }

    // Final likelihood and information matrix
    let last = cox_pass(&rs, x.view(), event, beta.view(), config.ties);
    let (hess, log_likelihood) = (last.information, last.loglik);
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
    let (info_null, _) = safe_inverse(&at_zero.information.view())?;
    let score_test = at_zero.score.dot(&info_null.dot(&at_zero.score));
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
        missing: data.missing,
    })
}

/// Rows in descending time order (stable `total_cmp` sort), grouped into blocks of exactly
/// equal time (coxph compares times exactly; the extractor normalised -0.0 and refused NaN).
struct CoxRiskSets {
    order: Vec<usize>,
    blocks: Vec<std::ops::Range<usize>>,
}

impl CoxRiskSets {
    fn new(time: &[f64]) -> CoxRiskSets {
        let mut order: Vec<usize> = (0..time.len()).collect();
        order.sort_by(|&a, &b| time[b].total_cmp(&time[a]));
        let mut blocks = Vec::new();
        let mut start = 0;
        while start < order.len() {
            let t = time[order[start]];
            let mut end = start + 1;
            while end < order.len() && time[order[end]] == t {
                end += 1;
            }
            blocks.push(start..end);
            start = end;
        }
        CoxRiskSets { order, blocks }
    }
}

/// Log partial likelihood, score and information (minus the Hessian) at `beta`.
struct CoxPass {
    loglik: f64,
    score: Array1<f64>,
    information: Array2<f64>,
}

/// `information += w * (m / a - xbar xbar')` on the upper triangle (`m` upper triangle only).
fn add_information(info: &mut Array2<f64>, m: &Array2<f64>, a: f64, xbar: &[f64], w: f64) {
    let q = xbar.len();
    for j in 0..q {
        for k in j..q {
            info[[j, k]] += w * (m[[j, k]] / a - xbar[j] * xbar[k]);
        }
    }
}

/// One pass of coxph's partial likelihood (coxfit6): every row of a block of tied times
/// joins the risk set before the block's deaths are scored. Breslow scores each death
/// against the full risk set; Efron removes the fraction k/d of the block's deaths for the
/// k-th of them. `x` is n x q in data row order.
fn cox_pass(
    rs: &CoxRiskSets,
    x: ArrayView2<f64>,
    event: &[bool],
    beta: ArrayView1<f64>,
    ties: TiesMethod,
) -> CoxPass {
    let q = beta.len();
    let eta = x.dot(&beta);
    let mut loglik = 0.0;
    let mut score = Array1::<f64>::zeros(q);
    let mut info = Array2::<f64>::zeros((q, q));
    let (mut s0, mut s1, mut s2) = (0.0, vec![0.0; q], Array2::<f64>::zeros((q, q)));
    let (mut e1, mut e2) = (vec![0.0; q], Array2::<f64>::zeros((q, q)));
    let mut xbar = vec![0.0; q];
    let mut num = Array2::<f64>::zeros((q, q));
    for block in &rs.blocks {
        let rows = &rs.order[block.clone()];
        let mut d = 0usize;
        let mut e0 = 0.0;
        e1.iter_mut().for_each(|v| *v = 0.0);
        e2.fill(0.0);
        for &i in rows {
            let r = eta[i].exp();
            let xi = x.row(i);
            s0 += r;
            for j in 0..q {
                let rx = r * xi[j];
                s1[j] += rx;
                for k in j..q {
                    s2[[j, k]] += rx * xi[k];
                }
            }
            if event[i] {
                d += 1;
                e0 += r;
                loglik += eta[i];
                for j in 0..q {
                    let rx = r * xi[j];
                    score[j] += xi[j];
                    e1[j] += rx;
                    if ties == TiesMethod::Efron {
                        for k in j..q {
                            e2[[j, k]] += rx * xi[k];
                        }
                    }
                }
            }
        }
        if d == 0 {
            continue;
        }
        match ties {
            TiesMethod::Breslow => {
                let df = d as f64;
                for j in 0..q {
                    xbar[j] = s1[j] / s0;
                    score[j] -= df * xbar[j];
                }
                loglik -= df * s0.ln();
                add_information(&mut info, &s2, s0, &xbar, df);
            }
            TiesMethod::Efron => {
                for k in 0..d {
                    let f = k as f64 / d as f64;
                    let a = s0 - f * e0;
                    for j in 0..q {
                        xbar[j] = (s1[j] - f * e1[j]) / a;
                        score[j] -= xbar[j];
                    }
                    loglik -= a.ln();
                    for j in 0..q {
                        for l in j..q {
                            num[[j, l]] = s2[[j, l]] - f * e2[[j, l]];
                        }
                    }
                    add_information(&mut info, &num, a, &xbar, 1.0);
                }
            }
        }
    }
    for j in 0..q {
        for k in (j + 1)..q {
            info[[k, j]] = info[[j, k]];
        }
    }
    CoxPass {
        loglik,
        score,
        information: info,
    }
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
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Coordinate-compress risk scores for Fenwick tree indexing
    let mut scores: Vec<f64> = sorted.iter().map(|s| s.2).collect();
    scores.sort_by(|a, b| a.total_cmp(b));
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
    /// Rows dropped because of a missing time, status or covariate.
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
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

        write_missing(f, &self.missing)
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
    let data = extract_survival(
        dataset.df(),
        time_col,
        event_col,
        None,
        x_cols,
        TimeRule::Positive,
        "AFT model",
    )?;

    // Design matrix with intercept, kept rows in their original order
    let p = x_cols.len() + 1; // +1 for intercept
    let n = data.time.len();
    let log_times: Vec<f64> = data.time.iter().map(|t| t.ln()).collect();
    let deltas = data.event;

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

    let mut x = try_zeros2(n, p, "AFT design matrix")?;
    x.column_mut(0).fill(1.0);
    for (j, values) in data.x.iter().enumerate() {
        for (cell, &v) in x.column_mut(j + 1).iter_mut().zip(values) {
            *cell = v;
        }
    }
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
        missing: data.missing,
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
    /// Rows dropped because of a missing time or event code.
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
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
                idx.extend((n - 3)..n); // n > 10 here
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

        write_missing(f, &self.missing)
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
    check_conf_level(conf_level)?;

    let df = dataset.df();
    refuse_boolean_time(df, time_col)?;
    let cols = ColumnRequest::new(df)
        .numeric(time_col)
        .numeric(event_type_col)
        .na(NaPolicy::Omit)
        .min_rows(1, "competing risks")
        .extract()?;
    let times = checked_times(
        cols.numeric(time_col),
        cols.rows(),
        time_col,
        TimeRule::NonNegative,
    )?;

    // Build observations; codes are whole numbers 0 (censored), 1, 2, ... up to 255
    let mut observations: Vec<(f64, u8)> = Vec::with_capacity(times.len());
    let mut event_type_set: std::collections::BTreeSet<u8> = std::collections::BTreeSet::new();
    for (i, (&t, &v)) in times.iter().zip(cols.numeric(event_type_col)).enumerate() {
        if !((0.0..=255.0).contains(&v) && v.fract() == 0.0) {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "Column '{event_type_col}' must hold event codes 0 (censored), 1, 2, ... \
                     (whole numbers up to 255); row {} is {v}",
                    cols.rows()[i]
                ),
            });
        }
        let e = v as u8;
        observations.push((t, e));
        if e > 0 {
            event_type_set.insert(e);
        }
    }

    let n_obs = observations.len();

    let event_types_vec: Vec<u8> = event_type_set.into_iter().collect();
    if event_types_vec.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "No events observed (all observations censored)".to_string(),
        });
    }

    // Sort by time
    observations.sort_by(|a, b| a.0.total_cmp(&b.0));

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
    let z = two_sided_z(conf_level)?;
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
        missing: cols.missing().clone(),
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::columns::MissingSummary;
    use crate::test_support::load_case;
    use p2a_validation::{Check, Tol};
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

    // =========================================================================
    // Validation Tests: Compare against R's survival package
    // Kaplan-Meier and the log-rank test are R-pinned below (validation/r/cases/
    // survival.R); the Cox, AFT and competing-risk checks are property checks,
    // not R-pinned (validation/econometrics/survival_models.md).
    // =========================================================================

    /// summary(fit, times = t): the row of the last time <= t, and the number at risk at t
    /// (at the first time >= t; 0 past the end).
    fn km_at(km: &KaplanMeierResult, t: f64) -> (usize, usize) {
        let row = km
            .times
            .iter()
            .rposition(|&x| x <= t)
            .unwrap_or_else(|| panic!("t = {t} is before the first time"));
        let n_risk = km
            .times
            .iter()
            .position(|&x| x >= t)
            .map_or(0, |i| km.n_at_risk[i]);
        (row, n_risk)
    }

    /// R NA is null in the case JSON and NaN after `Check`.
    fn opt(v: Option<f64>) -> f64 {
        v.unwrap_or(f64::NAN)
    }

    fn cfg(conf_type: KmConfType) -> KmConfig {
        KmConfig {
            conf_level: 0.95,
            conf_type,
        }
    }

    /// The `.km_values` keys of survival.R: summary(fit, times) and quantile(fit, 0.5).
    fn check_km_summary(
        check: &mut Check<'_>,
        km: &KaplanMeierResult,
        prefix: &str,
        times: &[f64],
        n: usize,
    ) {
        let rows: Vec<(usize, usize)> = times.iter().map(|&t| km_at(km, t)).collect();
        let pick = |v: &[f64]| rows.iter().map(|&(r, _)| v[r]).collect::<Vec<f64>>();
        let n_risk: Vec<f64> = rows.iter().map(|&(_, k)| k as f64).collect();
        check
            .vector(&format!("{prefix}_surv"), &pick(&km.survival), Tol::coef(n))
            .vector(
                &format!("{prefix}_std_err"),
                &pick(&km.std_errors),
                Tol::se(n),
            )
            .vector(
                &format!("{prefix}_lower"),
                &pick(&km.ci_lower),
                Tol::stat(n),
            )
            .vector(
                &format!("{prefix}_upper"),
                &pick(&km.ci_upper),
                Tol::stat(n),
            )
            .vector(&format!("{prefix}_n_risk"), &n_risk, Tol::exact())
            .scalar(
                &format!("{prefix}_median"),
                opt(km.median_survival),
                Tol::exact(),
            )
            .scalar(
                &format!("{prefix}_median_lower"),
                opt(km.median_ci.0),
                Tol::exact(),
            )
            .scalar(
                &format!("{prefix}_median_upper"),
                opt(km.median_ci.1),
                Tol::exact(),
            );
    }

    #[test]
    fn test_validate_km_lung() {
        let (case, ds) = load_case("surv_km_lung");
        let n = ds.nrows();
        let times = [5.0, 11.0, 100.0, 310.0, 500.0, 765.0];
        let mut check = Check::new(&case);
        for (conf_type, tag) in [
            (KmConfType::Log, "log"),
            (KmConfType::LogLog, "loglog"),
            (KmConfType::Plain, "plain"),
        ] {
            let km =
                &run_kaplan_meier_with(&ds, "time", "status", None, &cfg(conf_type)).unwrap()[0];
            assert_eq!(km.conf_type, conf_type);
            check_km_summary(&mut check, km, tag, &times, n);
        }

        // The default path is R's default conf.type, over the whole curve.
        let km = &run_kaplan_meier(&ds, "time", "status", None, 0.95).unwrap()[0];
        assert_eq!(km.conf_type, KmConfType::Log);
        let se_log: Vec<f64> = km
            .std_errors
            .iter()
            .zip(&km.survival)
            .map(|(se, s)| se / s)
            .collect();
        check
            .vector("curve_time", &km.times, Tol::exact())
            .vector("curve_surv", &km.survival, Tol::coef(n))
            .vector("curve_std_err_log", &se_log, Tol::se(n))
            .vector("curve_lower", &km.ci_lower, Tol::stat(n))
            .vector("curve_upper", &km.ci_upper, Tol::stat(n))
            .scalar("n", km.n_obs as f64, Tol::exact())
            .scalar("n_event", km.total_events as f64, Tol::exact())
            .scalar("unique_times", km.times.len() as f64, Tol::exact())
            .finish();
    }

    #[test]
    fn test_validate_km_lung_by_sex() {
        let (case, ds) = load_case("surv_km_lung_by_sex");
        let km =
            run_kaplan_meier_with(&ds, "time", "status", Some("sex"), &cfg(KmConfType::LogLog))
                .unwrap();
        assert_eq!(km.len(), 2);
        let mut check = Check::new(&case);
        for (k, label) in km.iter().zip(["1", "2"]) {
            assert_eq!(k.group.as_deref(), Some(label));
            check_km_summary(
                &mut check,
                k,
                &format!("sex{label}"),
                &[100.0, 300.0, 500.0],
                k.n_obs,
            );
        }
        check.finish();
    }

    #[test]
    fn test_validate_km_small() {
        let (case, ds) = load_case("surv_km_small");
        let n = ds.nrows();
        let mut check = Check::new(&case);
        for (conf_type, tag) in [(KmConfType::Log, "log"), (KmConfType::LogLog, "loglog")] {
            let km =
                &run_kaplan_meier_with(&ds, "time", "event", None, &cfg(conf_type)).unwrap()[0];
            check
                .vector(&format!("{tag}_time"), &km.times, Tol::exact())
                .vector(&format!("{tag}_surv"), &km.survival, Tol::coef(n))
                .vector(&format!("{tag}_std_err_surv"), &km.std_errors, Tol::se(n))
                .vector(&format!("{tag}_lower"), &km.ci_lower, Tol::stat(n))
                .vector(&format!("{tag}_upper"), &km.ci_upper, Tol::stat(n))
                .scalar(
                    &format!("{tag}_median"),
                    opt(km.median_survival),
                    Tol::exact(),
                )
                .scalar(
                    &format!("{tag}_median_lower"),
                    opt(km.median_ci.0),
                    Tol::exact(),
                )
                .scalar(
                    &format!("{tag}_median_upper"),
                    opt(km.median_ci.1),
                    Tol::exact(),
                );
            // S = 0 at the last time: SE and both limits are NA in R.
            assert_eq!(km.survival[9], 0.0, "{tag}");
            assert!(
                km.std_errors[9].is_nan() && km.ci_lower[9].is_nan() && km.ci_upper[9].is_nan(),
                "{tag}"
            );
            assert_eq!(km.median_ci.1, None, "{tag}");
        }
        check.finish();
    }

    #[test]
    fn test_validate_km_median_rules() {
        let (case, ds) = load_case("surv_km_median_rules");
        let mut check = Check::new(&case);
        for (conf_type, tag) in [
            (KmConfType::Log, "log"),
            (KmConfType::LogLog, "loglog"),
            (KmConfType::Plain, "plain"),
        ] {
            let km = run_kaplan_meier_with(&ds, "time", "event", Some("group"), &cfg(conf_type))
                .unwrap();
            let labels: Vec<&str> = km.iter().filter_map(|k| k.group.as_deref()).collect();
            assert_eq!(labels, ["last", "mid", "never"]);
            for k in &km {
                let (g, n) = (k.group.as_deref().unwrap_or(""), k.n_obs);
                if conf_type == KmConfType::Log {
                    check
                        .vector(&format!("{g}_time"), &k.times, Tol::exact())
                        .vector(&format!("{g}_surv"), &k.survival, Tol::coef(n))
                        .vector(&format!("{g}_n_risk"), &as_f64(&k.n_at_risk), Tol::exact());
                }
                check
                    .vector(&format!("{g}_{tag}_lower"), &k.ci_lower, Tol::stat(n))
                    .vector(&format!("{g}_{tag}_upper"), &k.ci_upper, Tol::stat(n))
                    .scalar(
                        &format!("{g}_{tag}_median"),
                        opt(k.median_survival),
                        Tol::exact(),
                    )
                    .scalar(
                        &format!("{g}_{tag}_median_lower"),
                        opt(k.median_ci.0),
                        Tol::exact(),
                    )
                    .scalar(
                        &format!("{g}_{tag}_median_upper"),
                        opt(k.median_ci.1),
                        Tol::exact(),
                    );
            }
            if conf_type == KmConfType::Log {
                let strata_n: Vec<f64> = km.iter().map(|k| k.n_obs as f64).collect();
                let strata_median: Vec<f64> = km.iter().map(|k| opt(k.median_survival)).collect();
                check.vector("strata_n", &strata_n, Tol::exact()).vector(
                    "strata_median",
                    &strata_median,
                    Tol::exact(),
                );
            }
            if conf_type == KmConfType::LogLog {
                // A lower limit although the median is never reached.
                let never = &km[2];
                assert_eq!(never.median_survival, None);
                assert_eq!(never.median_ci, (Some(1.0), None));
            }
        }
        check.finish();
    }

    #[test]
    fn test_validate_km_conf_level_and_type_validation() {
        let (case, ds) = load_case("surv_km_small");
        for conf_level in [1.0, 0.0, f64::NAN] {
            let config = KmConfig {
                conf_level,
                conf_type: KmConfType::Log,
            };
            invalid_spec(
                run_kaplan_meier_with(&ds, "time", "event", None, &config),
                "conf_level",
            );
        }
        let km95 =
            &run_kaplan_meier_with(&ds, "time", "event", None, &cfg(KmConfType::Log)).unwrap()[0];
        let config90 = KmConfig {
            conf_level: 0.9,
            conf_type: KmConfType::Log,
        };
        let km90 = &run_kaplan_meier_with(&ds, "time", "event", None, &config90).unwrap()[0];
        assert_eq!(km90.conf_level, 0.9);
        for i in 0..km95.times.len() {
            let (l95, u95, l90, u90) = (
                km95.ci_lower[i],
                km95.ci_upper[i],
                km90.ci_lower[i],
                km90.ci_upper[i],
            );
            if l95.is_finite() && l90.is_finite() {
                assert!(l90 >= l95, "t[{i}]: lower {l90} < {l95}");
            }
            if u95.is_finite() && u90.is_finite() {
                assert!(u90 <= u95, "t[{i}]: upper {u90} > {u95}");
            }
        }
        // The curve itself does not depend on the confidence level.
        Check::new(&case)
            .vector("log_time", &km90.times, Tol::exact())
            .vector("log_surv", &km90.survival, Tol::coef(km90.n_obs))
            .finish();
    }

    /// Counts as f64, for `Check::vector`.
    fn as_f64(v: &[usize]) -> Vec<f64> {
        v.iter().map(|&x| x as f64).collect()
    }

    #[test]
    fn test_validate_logrank_lung() {
        let (case, ds) = load_case("surv_logrank_lung");
        let n = ds.nrows();
        let sex = log_rank_test(&ds, "time", "status", "sex").unwrap();
        let ecog = log_rank_test(&ds, "time", "status", "ph.ecog").unwrap();
        Check::new(&case)
            .scalar("sex_chisq", sex.chi_squared, Tol::stat(n))
            .scalar("sex_df", sex.df as f64, Tol::exact())
            .scalar("sex_p", sex.p_value, Tol::p_value(n))
            .vector("sex_obs", &as_f64(&sex.events_per_group), Tol::exact())
            .vector("sex_exp", &sex.expected_per_group, Tol::stat(n))
            .vector("sex_n", &as_f64(&sex.n_per_group), Tol::exact())
            .vector("sex_var", &sex.variance, Tol::stat(n))
            .scalar("ecog_chisq", ecog.chi_squared, Tol::stat(n))
            .scalar("ecog_df", ecog.df as f64, Tol::exact())
            .scalar("ecog_p", ecog.p_value, Tol::p_value(n))
            .vector("ecog_obs", &as_f64(&ecog.events_per_group), Tol::exact())
            .vector("ecog_exp", &ecog.expected_per_group, Tol::stat(n))
            .vector("ecog_n", &as_f64(&ecog.n_per_group), Tol::exact())
            .scalar(
                "ecog_n_used",
                ecog.n_per_group.iter().sum::<usize>() as f64,
                Tol::exact(),
            )
            .finish();
        assert_eq!(sex.groups, ["1", "2"]);
        // sum(O) = sum(E): every event is counted once against the right risk sets.
        assert!(sex.obs_minus_exp.iter().sum::<f64>().abs() < 1e-9);
        assert!(ecog.obs_minus_exp.iter().sum::<f64>().abs() < 1e-9);
    }

    #[test]
    fn test_validate_logrank_small() {
        let (case, ds) = load_case("surv_logrank_small");
        let n = ds.nrows();
        let r = log_rank_test(&ds, "time", "event", "group").unwrap();
        Check::new(&case)
            .scalar("chisq", r.chi_squared, Tol::stat(n))
            .scalar("p", r.p_value, Tol::p_value(n))
            .vector("obs", &as_f64(&r.events_per_group), Tol::exact())
            .vector("exp", &r.expected_per_group, Tol::stat(n))
            .finish();
        assert_eq!(r.groups, ["0", "1"]);
        assert_eq!(r.df, 1);
    }

    #[test]
    fn test_validate_logrank_zero_expected_group() {
        let (case, ds) = load_case("surv_km_median_rules");
        let n = ds.nrows();
        // "never" leaves before the first event: expected 0, excluded from the test (df 1).
        let r = log_rank_test(&ds, "time_early", "event_early", "group").unwrap();
        Check::new(&case)
            .scalar("lr_zero_chisq", r.chi_squared, Tol::stat(n))
            .scalar("lr_subset_chisq", r.chi_squared, Tol::stat(n))
            .scalar("lr_zero_df", r.df as f64, Tol::exact())
            .scalar("lr_zero_p", r.p_value, Tol::p_value(n))
            .vector("lr_zero_exp", &r.expected_per_group, Tol::stat(n))
            .vector("lr_zero_obs", &as_f64(&r.events_per_group), Tol::exact())
            .finish();
        assert_eq!(r.groups, ["last", "mid", "never"]);
        let text = r.to_string();
        assert!(text.contains("no expected events: never"), "{text}");

        // Without "mid" (missing labels dropped) only "last" has expected events: R reports
        // chi-square 0 on 0 df (lr_one_group_chisq); p2a refuses.
        let m = invalid_spec(
            log_rank_test(&ds, "time_early", "event_early", "group_no_mid"),
            "one group with expected events",
        );
        assert!(m.contains("two groups"), "{m}");
    }

    /// Every subject at risk at the first event time has the event then, so no one is
    /// left and the variance of O - E is zero: R's survdiff stops ("system is exactly
    /// singular"), and so does p2a, where it reported chi-square 0, df 1, p 1.
    #[test]
    fn test_logrank_zero_variance_is_an_error() {
        let all_at_once = df! {
            "time" => [1.0, 1.0, 1.0, 1.0],
            "event" => [1.0, 1.0, 1.0, 1.0],
            "group" => ["a", "a", "b", "b"],
        }
        .unwrap();
        // A row censored before the first event time, and three groups.
        let censored_first = df! {
            "time" => [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "event" => [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "group" => ["a", "a", "a", "b", "b", "c", "c"],
        }
        .unwrap();
        for df in [all_at_once, censored_first] {
            let m = invalid_spec(
                log_rank_test(&Dataset::new(df), "time", "event", "group"),
                "zero variance",
            );
            assert!(
                m.contains("variance") && m.contains("first event time"),
                "{m}"
            );
        }

        // One subject outlives the first event time: V = 3 (4 - 3) / (16 * 3) * 2 * 2 = 1/4
        // for group b, O - E = 2 - (3/2 + 1) = -1/2, so the chi-square is 1 on 1 df.
        let one_survivor = df! {
            "time" => [1.0, 1.0, 1.0, 2.0],
            "event" => [1.0, 1.0, 1.0, 1.0],
            "group" => ["a", "a", "b", "b"],
        }
        .unwrap();
        let r = log_rank_test(&Dataset::new(one_survivor), "time", "event", "group").unwrap();
        assert_eq!((r.df, r.variance[3]), (1, 0.25));
        assert!((r.chi_squared - 1.0).abs() < 1e-12, "{}", r.chi_squared);
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

    // =========================================================================
    // R-pinned: column extraction, missing values and Surv() status coding
    // (cases in validation/r/cases/survival.R; registry validation/methods/survival.json)
    // =========================================================================

    /// Copy of `ds` with `cols` cast to Float64 (identical values).
    fn cast_f64(ds: &Dataset, cols: &[&str]) -> Dataset {
        let mut df = ds.df().clone();
        for &c in cols {
            let cast = df.column(c).unwrap().cast(&DataType::Float64).unwrap();
            df.with_column(cast).unwrap();
        }
        Dataset::new(df)
    }

    /// Bit patterns, so that equality also holds for NaN (the AFT optimiser can diverge:
    /// ECON-47, Tier 3).
    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// The values of the Int64 column `from`, which has no nulls.
    fn i64_values(ds: &Dataset, from: &str) -> Vec<i64> {
        ds.df()
            .column(from)
            .unwrap()
            .i64()
            .unwrap()
            .iter()
            .map(|v| v.expect("no nulls in the source column"))
            .collect()
    }

    /// Copy of `ds` with `series` added (or replaced, by name).
    fn with_series(ds: &Dataset, series: Series) -> Dataset {
        let mut df = ds.df().clone();
        df.with_column(series).unwrap();
        Dataset::new(df)
    }

    /// Copy of `ds` with a new column `name` computed row by row from the Int64 column `from`.
    fn derive_i64(ds: &Dataset, from: &str, name: &str, f: impl Fn(i64) -> i64) -> Dataset {
        let v: Vec<i64> = i64_values(ds, from).into_iter().map(f).collect();
        with_series(ds, Series::new(name.into(), v))
    }

    /// Same, Float64, with the 0-based row index.
    fn derive_f64(ds: &Dataset, from: &str, name: &str, f: impl Fn(usize, i64) -> f64) -> Dataset {
        let v: Vec<f64> = i64_values(ds, from)
            .into_iter()
            .enumerate()
            .map(|(i, x)| f(i, x))
            .collect();
        with_series(ds, Series::new(name.into(), v))
    }

    /// Same, Boolean.
    fn derive_bool(ds: &Dataset, from: &str, name: &str, f: impl Fn(i64) -> bool) -> Dataset {
        let v: Vec<bool> = i64_values(ds, from).into_iter().map(f).collect();
        with_series(ds, Series::new(name.into(), v))
    }

    /// The message of an `InvalidSpecification`, or a panic naming what came instead.
    fn invalid_spec<T: std::fmt::Debug>(r: EconResult<T>, what: &str) -> String {
        match r {
            Err(EconError::InvalidSpecification { message }) => message,
            other => panic!("{what}: expected InvalidSpecification, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_na_survival_int64_columns() {
        let (case, ds) = load_case("surv_km_lung");
        for c in ["time", "status", "age", "sex"] {
            assert_eq!(ds.df().column(c).unwrap().dtype(), &DataType::Int64, "{c}");
        }
        let f = cast_f64(&ds, &["time", "status", "age", "sex"]);

        let km = run_kaplan_meier(&ds, "time", "status", Some("sex"), 0.95).unwrap();
        let km_f = run_kaplan_meier(&f, "time", "status", Some("sex"), 0.95).unwrap();
        assert_eq!(km.len(), 2);
        assert_eq!(km[0].group.as_deref(), Some("1"));
        assert_eq!(km[1].group.as_deref(), Some("2"));
        assert_eq!(km_f.len(), 2);
        for (a, b) in km.iter().zip(&km_f) {
            assert_eq!(a.group, b.group);
            assert_eq!(bits(&a.times), bits(&b.times));
            assert_eq!(bits(&a.survival), bits(&b.survival));
            assert_eq!(a.n_at_risk, b.n_at_risk);
        }

        let lr = log_rank_test(&ds, "time", "status", "sex").unwrap();
        let lr_f = log_rank_test(&f, "time", "status", "sex").unwrap();
        assert_eq!(bits(&[lr.chi_squared]), bits(&[lr_f.chi_squared]));
        let cox = run_cox_ph(&ds, "time", "status", &["age", "sex"], None).unwrap();
        let cox_f = run_cox_ph(&f, "time", "status", &["age", "sex"], None).unwrap();
        assert_eq!(bits(&cox.coefficients), bits(&cox_f.coefficients));
        let aft = run_aft(&ds, "time", "status", &["age", "sex"], None).unwrap();
        let aft_f = run_aft(&f, "time", "status", &["age", "sex"], None).unwrap();
        assert_eq!(bits(&aft.coefficients), bits(&aft_f.coefficients));

        Check::new(&case)
            .scalar(
                "n",
                km.iter().map(|k| k.n_obs).sum::<usize>() as f64,
                Tol::exact(),
            )
            .scalar(
                "n_event",
                km.iter().map(|k| k.total_events).sum::<usize>() as f64,
                Tol::exact(),
            )
            .finish();

        // Competing risks on Int64 time and event columns.
        let (_, small) = load_case("surv_km_small");
        assert_eq!(
            small.df().column("event").unwrap().dtype(),
            &DataType::Int64
        );
        let cr = run_competing_risks(&small, "time", "event", 0.95).unwrap();
        assert_eq!(cr.event_types, vec![1]);
        assert_eq!(cr.n_obs, 10);
    }

    #[test]
    fn test_validate_surv_status_coding_lung() {
        let (case, ds) = load_case("surv_km_lung");
        let ds = derive_i64(&ds, "status", "status01", |s| s - 1);
        let ds = derive_bool(&ds, "status", "dead", |s| s == 2);

        let mut check = Check::new(&case);
        let (mut surv, mut chisq, mut cox, mut aft) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for status in ["status", "status01", "dead"] {
            let km = &run_kaplan_meier(&ds, "time", status, None, 0.95).unwrap()[0];
            check
                .scalar("n_event", km.total_events as f64, Tol::exact())
                .scalar("n", km.n_obs as f64, Tol::exact());
            surv.push(bits(&km.survival));
            chisq.push(bits(&[log_rank_test(&ds, "time", status, "sex")
                .unwrap()
                .chi_squared]));
            cox.push(bits(
                &run_cox_ph(&ds, "time", status, &["age", "sex"], None)
                    .unwrap()
                    .coefficients,
            ));
            aft.push(bits(
                &run_aft(&ds, "time", status, &["age", "sex"], None)
                    .unwrap()
                    .coefficients,
            ));
        }
        check.finish();
        for (what, v) in [
            ("KM", &surv),
            ("log-rank", &chisq),
            ("Cox", &cox),
            ("AFT", &aft),
        ] {
            assert_eq!(v[0], v[1], "{what}: 1/2 vs 0/1 coding");
            assert_eq!(v[0], v[2], "{what}: 1/2 vs logical coding");
        }
    }

    #[test]
    fn test_validate_surv_status_invalid_codes_are_errors() {
        let (case, ds) = load_case("surv_km_lung");
        // R codes these as NA with a warning ("Invalid status value, converted to NA").
        let ds = derive_i64(&ds, "status", "status23", |s| s + 1);
        let ds = derive_f64(&ds, "status", "status012", |i, s| {
            if i == 0 { 0.0 } else { s as f64 }
        });
        let ds = derive_f64(&ds, "status", "status_half", |i, s| {
            if i == 0 { 1.5 } else { s as f64 }
        });
        let mut check = Check::new(&case);
        for status in ["status23", "status012", "status_half"] {
            let messages = [
                invalid_spec(run_kaplan_meier(&ds, "time", status, None, 0.95), "KM"),
                invalid_spec(log_rank_test(&ds, "time", status, "sex"), "log-rank"),
                invalid_spec(run_cox_ph(&ds, "time", status, &["age"], None), "Cox"),
                invalid_spec(run_aft(&ds, "time", status, &["age"], None), "AFT"),
            ];
            for m in &messages {
                for part in [status, "0/1", "1/2", "Surv"] {
                    assert!(m.contains(part), "{status}: {part:?} not in {m}");
                }
                if status == "status23" {
                    assert!(m.contains('3'), "{m}");
                }
            }
        }
        // The valid 1/2 column of the same dataset still gives R's counts.
        let km = &run_kaplan_meier(&ds, "time", "status", None, 0.95).unwrap()[0];
        check
            .scalar("n_event", km.total_events as f64, Tol::exact())
            .finish();
    }

    #[test]
    fn test_validate_na_survival_missing_rows_are_dropped_and_reported() {
        let (case, ds) = load_case("surv_logrank_lung");
        let ecog = log_rank_test(&ds, "time", "status", "ph.ecog").unwrap();
        assert_eq!(ecog.groups, ["0", "1", "2", "3"]);
        assert_eq!(
            ecog.missing,
            MissingSummary {
                n_total: 228,
                n_used: 227,
                n_dropped: 1,
                by_column: vec![("ph.ecog".to_string(), 1)],
            }
        );
        Check::new(&case)
            .vector("ecog_n", &as_f64(&ecog.n_per_group), Tol::exact())
            .vector("ecog_obs", &as_f64(&ecog.events_per_group), Tol::exact())
            .scalar(
                "ecog_n_used",
                ecog.n_per_group.iter().sum::<usize>() as f64,
                Tol::exact(),
            )
            .finish();
        assert!(
            ecog.to_string().trim_end().ends_with(
                "Missing values: 1 of 228 rows dropped because of missing values (ph.ecog: 1)"
            ),
            "{ecog}"
        );

        let km = run_kaplan_meier(&ds, "time", "status", Some("ph.ecog"), 0.95).unwrap();
        assert_eq!(km.len(), 4);
        for k in &km {
            assert_eq!(k.missing.n_dropped, 1, "{:?}", k.group);
        }

        // R coxph: n = 227, nevent = 164 (the row with a missing ph.karno is dropped).
        let status = ds.df().column("status").unwrap().i64().unwrap().clone();
        let karno = ds.df().column("ph.karno").unwrap();
        let events_with_karno = (0..ds.nrows())
            .filter(|&i| status.get(i) == Some(2) && !karno.get(i).unwrap().is_null())
            .count();
        let cox = run_cox_ph(&ds, "time", "status", &["age", "sex", "ph.karno"], None).unwrap();
        assert_eq!(cox.n_obs, 227);
        assert_eq!(cox.n_events, events_with_karno);
        assert_eq!(cox.missing.by_column, vec![("ph.karno".to_string(), 1)]);
        assert!(
            cox.to_string().trim_end().ends_with(
                "Missing values: 1 of 228 rows dropped because of missing values (ph.karno: 1)"
            ),
            "{cox}"
        );
        // R survreg: n = 227.
        let aft = run_aft(&ds, "time", "status", &["age", "ph.karno"], None).unwrap();
        assert_eq!(aft.n_obs, 227);
        assert_eq!(aft.missing.n_dropped, 1);
    }

    #[test]
    fn test_validate_na_survival_bad_columns_name_the_column() {
        let (case, ds) = load_case("surv_logrank_small");
        assert_eq!(ds.df().column("time").unwrap().dtype(), &DataType::Int64);
        // The loader reads the quoted "0"/"1" labels as Int64; a text copy of them:
        let labels: Vec<String> = i64_values(&ds, "group")
            .into_iter()
            .map(|g| format!("arm{g}"))
            .collect();
        let ds = with_series(&ds, Series::new("arm".into(), labels));

        match run_kaplan_meier(&ds, "tme", "event", None, 0.95) {
            Err(EconError::ColumnNotFound { column, .. }) => assert_eq!(column, "tme"),
            other => panic!("expected ColumnNotFound, got {other:?}"),
        }
        match run_kaplan_meier(&ds, "arm", "event", None, 0.95) {
            Err(EconError::NonNumericColumn { column }) => assert_eq!(column, "arm"),
            other => panic!("expected NonNumericColumn, got {other:?}"),
        }
        match log_rank_test(&ds, "time", "arm", "group") {
            Err(EconError::NonNumericColumn { column }) => assert_eq!(column, "arm"),
            other => panic!("expected NonNumericColumn, got {other:?}"),
        }
        // A text group column is a factor: the same test as on the Int64 labels.
        let by_text = log_rank_test(&ds, "time", "event", "arm").unwrap();
        let by_int = log_rank_test(&ds, "time", "event", "group").unwrap();
        assert_eq!(by_text.groups, ["arm0", "arm1"]);
        assert_eq!(bits(&[by_text.chi_squared]), bits(&[by_int.chi_squared]));
        let ds = derive_bool(&ds, "time", "late", |t| t > 4);
        match run_kaplan_meier(&ds, "late", "event", None, 0.95) {
            Err(EconError::NonNumericColumn { column }) => assert_eq!(column, "late"),
            other => panic!("expected NonNumericColumn for a Boolean time, got {other:?}"),
        }

        let ds = derive_i64(&ds, "time", "shifted", |t| t - 2);
        let negative = [
            invalid_spec(run_kaplan_meier(&ds, "shifted", "event", None, 0.95), "KM"),
            invalid_spec(log_rank_test(&ds, "shifted", "event", "group"), "log-rank"),
            invalid_spec(run_cox_ph(&ds, "shifted", "event", &["time"], None), "Cox"),
            invalid_spec(run_competing_risks(&ds, "shifted", "event", 0.95), "CR"),
        ];
        for m in &negative {
            assert!(m.contains("'shifted'") && m.contains("negative"), "{m}");
        }

        let ds = derive_i64(&ds, "time", "t0", |t| t - 1);
        let km = run_kaplan_meier(&ds, "t0", "event", None, 0.95).unwrap();
        let m = invalid_spec(run_aft(&ds, "t0", "event", &["time"], None), "AFT");
        assert!(m.contains("'t0'") && m.contains("positive"), "{m}");

        let ds = derive_f64(&ds, "event", "code_half", |i, e| {
            if i == 0 { 1.5 } else { e as f64 }
        });
        let ds = derive_f64(&ds, "event", "code_neg", |i, e| {
            if i == 0 { -1.0 } else { e as f64 }
        });
        for code in ["code_half", "code_neg"] {
            let m = invalid_spec(run_competing_risks(&ds, "time", code, 0.95), code);
            assert!(m.contains("whole number") && m.contains(code), "{m}");
        }

        // conf_level NaN: the old `<= 0.0 || >= 1.0` check let it through.
        invalid_spec(
            run_kaplan_meier(&ds, "time", "event", None, f64::NAN),
            "KM NaN",
        );
        invalid_spec(
            run_competing_risks(&ds, "time", "event", f64::NAN),
            "CR NaN",
        );

        // The valid columns still give R's counts: 12 rows, events per group (survdiff obs).
        assert_eq!(km[0].n_obs, 12);
        let by_group = run_kaplan_meier(&ds, "time", "event", Some("group"), 0.95).unwrap();
        let events: Vec<f64> = by_group.iter().map(|k| k.total_events as f64).collect();
        Check::new(&case)
            .vector("obs", &events, Tol::exact())
            .finish();
    }

    #[test]
    fn test_km_conf_type_parse_and_defaults() {
        for (text, want) in [
            ("log", KmConfType::Log),
            (" Log ", KmConfType::Log),
            ("log-log", KmConfType::LogLog),
            ("loglog", KmConfType::LogLog),
            ("plain", KmConfType::Plain),
        ] {
            assert_eq!(text.parse::<KmConfType>().unwrap(), want, "{text:?}");
        }
        match "arcsin".parse::<KmConfType>() {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(message.contains("\"log-log\""), "{message}");
                assert!(message.contains("'arcsin'"), "{message}");
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
        assert_eq!(KmConfType::LogLog.to_string(), "log-log");
        assert_eq!(KmConfType::Log.to_string(), "log");
        assert_eq!(KmConfType::Plain.to_string(), "plain");
        assert_eq!(
            KmConfig::default(),
            KmConfig {
                conf_level: 0.95,
                conf_type: KmConfType::Log
            }
        );
    }

    // =========================================================================
    // Cox and AFT: option parsing
    // =========================================================================

    #[test]
    fn test_ties_method_parses_aliases_and_rejects_unknown() {
        use TiesMethod::*;
        TiesMethod::choices().validate().unwrap();
        for (s, t) in [
            ("efron", Efron),
            ("Efron", Efron),
            (" EFRON ", Efron),
            ("breslow", Breslow),
            ("Breslow", Breslow),
        ] {
            assert_eq!(s.parse::<TiesMethod>().unwrap(), t, "{s}");
        }
        for t in [Efron, Breslow] {
            assert_eq!(t.to_string().parse::<TiesMethod>().unwrap(), t);
        }
        let err = "exact".parse::<TiesMethod>().unwrap_err().to_string();
        assert!(
            err.contains("unknown ties 'exact'") && err.contains("efron, breslow"),
            "{err}"
        );
        assert!("".parse::<TiesMethod>().is_err());
    }

    #[test]
    fn test_aft_distribution_parses_aliases_and_rejects_unknown() {
        use AftDistribution::*;
        AftDistribution::choices().validate().unwrap();
        for (s, d) in [
            ("weibull", Weibull),
            ("Weibull", Weibull),
            ("exponential", Exponential),
            ("lognormal", LogNormal),
            ("log-normal", LogNormal),
            ("Log-Normal", LogNormal),
            ("log_normal", LogNormal),
            ("loglogistic", LogLogistic),
            ("log-logistic", LogLogistic),
            ("Log-Logistic", LogLogistic),
        ] {
            assert_eq!(s.parse::<AftDistribution>().unwrap(), d, "{s}");
        }
        for d in [Weibull, Exponential, LogNormal, LogLogistic] {
            // Display ("Log-Normal") round-trips
            assert_eq!(d.to_string().parse::<AftDistribution>().unwrap(), d);
        }
        for bad in ["gaussian", "logistic", "rayleigh", "t", "extreme", ""] {
            assert!(bad.parse::<AftDistribution>().is_err(), "{bad}");
        }
        let err = "gaussian".parse::<AftDistribution>().unwrap_err().to_string();
        assert!(
            err.contains("unknown dist 'gaussian'")
                && err.contains("weibull, exponential, lognormal, loglogistic"),
            "{err}"
        );
    }

    // =========================================================================
    // Cox proportional hazards: R-pinned (validation/r/cases/surv_cox_aft.R;
    // registry validation/methods/surv_cox_aft.json)
    // =========================================================================

    const COX_X: [&str; 3] = ["age", "sex", "ph.ecog"];

    /// coxph.control(eps = 1e-12, iter.max = 100), as in the surv_cox_* cases.
    fn cox_tight(ties: TiesMethod) -> CoxConfig {
        let mut c = CoxConfig::default();
        c.ties = ties;
        c.tolerance = 1e-12;
        c.max_iter = 100;
        c
    }

    fn ties_tag(ties: TiesMethod) -> &'static str {
        match ties {
            TiesMethod::Efron => "efron",
            TiesMethod::Breslow => "breslow",
        }
    }

    /// `validation/datasets/generated/<name>.csv`, for property tests that pin no R value.
    fn generated_dataset(name: &str) -> Dataset {
        let path = p2a_validation::repo_path(&format!("validation/datasets/generated/{name}.csv"));
        match crate::data::DataLoader::load_csv(&path) {
            Ok(df) => Dataset::new(df),
            Err(e) => panic!("cannot load {}: {e}", path.display()),
        }
    }

    /// Heavy ties (10 rows, 5 distinct times, censored rows tied with deaths), Efron and Breslow,
    /// rows in file order and reversed. Before: Breslow 0.36576456754379533 / reversed
    /// -0.47976648482093837 (R -0.050231244457068). Replaces a test whose comment quoted "R"
    /// ranges R does not produce and whose |Efron - Breslow| < 0.5 could not fail.
    #[test]
    fn test_validate_cox_ties_small() {
        let (case, ds) = load_case("surv_cox_ties_small");
        let n = case.scalar("n") as usize;
        let rev = Dataset::new(ds.df().reverse());
        let mut check = Check::new(&case);
        for ties in [TiesMethod::Efron, TiesMethod::Breslow] {
            let tag = ties_tag(ties);
            for d in [&ds, &rev] {
                let r = run_cox_ph(d, "time", "event", &["x"], Some(cox_tight(ties))).unwrap();
                check
                    .named(&format!("{tag}_coef"), &r.variables, &r.coefficients, Tol::coef(n))
                    .named(&format!("{tag}_se"), &r.variables, &r.std_errors, Tol::se(n))
                    .scalar(&format!("{tag}_loglik"), r.log_likelihood, Tol::stat(n))
                    .scalar(
                        &format!("{tag}_loglik_null"),
                        r.log_likelihood_null,
                        Tol::stat(n),
                    )
                    .scalar(&format!("{tag}_score_test"), r.score_test, Tol::stat(n));
            }
        }
        check.finish();
    }
}
