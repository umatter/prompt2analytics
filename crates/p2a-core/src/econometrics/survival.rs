//! Survival analysis module.
//!
//! Pure Rust implementation of survival analysis methods including:
//! - Kaplan-Meier estimator (non-parametric survival curves)
//! - Log-rank test (comparison of survival curves)
//! - Cox Proportional Hazards model (semi-parametric regression)
//! - Accelerated Failure Time models (parametric regression)
//! - Competing risks / Aalen-Johansen estimator, with survfit's delta-method standard errors
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
use statrs::distribution::{ContinuousCDF, Normal};
use std::collections::{BTreeMap, HashMap};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, Factor, MissingSummary, NaPolicy, column};
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::{pseudoinverse, safe_inverse};
use crate::linalg::solve::{LM_QR_TOL, Qr, SingularPolicy, aliasing_message, sym_inverse};
use crate::regression::ols::{normal_equations_inverse, solve_ols};
use crate::resource::{checkpoint, try_zeros2};
use crate::traits::convergence::{Convergence, ResultNotes, notes_text, relative_change};
use crate::traits::estimator::{
    SignificanceLevel, chi_squared_p_value, normal_p_value_two_sided, z_critical,
};

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

/// Wrapper for f64 to use as BTreeMap key, ordered by `f64::total_cmp` (equality, order and
/// `Ord` agree on -0.0 and NaN).
#[derive(Debug, Clone, Copy)]
struct OrderedF64(f64);

impl PartialEq for OrderedF64 {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl Eq for OrderedF64 {}

impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Method for handling tied event times in the Cox model (R `coxph(ties = )`).
///
/// Parsed from "efron" or "breslow" (case-insensitive, `FromStr`); other values are refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TiesMethod {
    /// Breslow approximation.
    Breslow,
    /// Efron approximation (default, as coxph).
    #[default]
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
            (
                &["loglogistic", "log_logistic"],
                AftDistribution::LogLogistic
            ),
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
    // u lies in the column space of W, so an ill-conditioned W changes neither the statistic
    // nor survdiff's answer.
    // inverse-ok: u' W^- u is the same for every generalized inverse
    let w_inv = match safe_inverse(&w.view()) {
        Ok((inv, _)) => inv,
        Err(_) => pseudoinverse(&w.view())?,
    };
    let chi_squared = u.dot(&w_inv.dot(&u));

    let df_val = k;
    let p_value = crate::traits::estimator::chi_squared_p_value(chi_squared, df_val as f64);
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
    /// Lin-Wei (1989) robust (sandwich) standard errors from the score residuals, as
    /// `coxph(robust = TRUE)`. Implied by `cluster`.
    pub robust_se: bool,
    /// Column whose levels define clusters for the clustered sandwich (coxph `cluster`): the
    /// score residuals are summed within clusters first. Implies `robust_se`. Rows with a
    /// missing cluster label are dropped and reported in `missing`.
    #[serde(default)]
    pub cluster: Option<String>,
}

/// How the reported Cox standard errors were computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CoxSeType {
    /// sqrt(diag(I^-1)), the inverse information at the estimate (coxph's default).
    #[default]
    Model,
    /// Lin-Wei (1989) sandwich I^-1 (sum_i U_i U_i') I^-1 from the score residuals
    /// (`coxph(robust = TRUE)`).
    Robust,
    /// The same with the score residuals summed within clusters first
    /// (`coxph(..., cluster = <col>)`).
    Cluster,
}

impl fmt::Display for CoxSeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            CoxSeType::Model => "model-based",
            CoxSeType::Robust => "robust (Lin-Wei sandwich)",
            CoxSeType::Cluster => "robust, clustered (Lin-Wei sandwich)",
        })
    }
}

impl Default for CoxConfig {
    fn default() -> Self {
        Self {
            ties: TiesMethod::Efron,
            max_iter: 25,
            tolerance: 1e-9,
            robust_se: false,
            cluster: None,
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
    /// Covariates not estimated because they are constant or collinear with earlier ones
    /// (coefficient NA in coxph); their entries in every per-variable vector are NaN.
    #[serde(default)]
    pub aliased: Vec<String>,
    /// Warnings (aliasing, non-convergence, "coefficient may be infinite").
    #[serde(default)]
    pub warnings: Vec<String>,
    /// The Newton iteration's convergence record.
    #[serde(default)]
    pub convergence: Convergence,
    /// What `std_errors` are; z, p, the hazard-ratio interval and the Wald test use them.
    #[serde(default)]
    pub se_type: CoxSeType,
    /// The cluster column of a clustered sandwich.
    #[serde(default)]
    pub cluster: Option<String>,
    /// The number of clusters of a clustered sandwich.
    #[serde(default)]
    pub n_clusters: Option<usize>,
    /// Model-based SEs sqrt(diag(I^-1)) (R: sqrt(diag(fit$naive.var)) when robust); equal to
    /// `std_errors` when `se_type` is `Model`.
    #[serde(default)]
    pub naive_std_errors: Vec<f64>,
    /// Robust score test (R `fit$rscore`) from the score residuals at beta = 0; `None` unless
    /// `se_type` is robust, `Some(NaN)` when it cannot be computed (see `warnings`).
    #[serde(default)]
    pub robust_score_test: Option<f64>,
    /// p-value of `robust_score_test`.
    #[serde(default)]
    pub robust_score_p_value: Option<f64>,
}

impl CoxResult {
    /// Degrees of freedom of the overall tests: the identified coefficients.
    pub fn df(&self) -> usize {
        self.variables.len() - self.aliased.len()
    }
}

impl ResultNotes for CoxResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
    fn notes_convergence(&self) -> Option<Convergence> {
        Some(self.convergence.clone())
    }
}

/// One overall-test line, or "not reported" when the statistic is NaN.
fn write_test_line(
    f: &mut fmt::Formatter<'_>,
    name: &str,
    stat: f64,
    df: usize,
    p: f64,
) -> fmt::Result {
    if stat.is_nan() {
        writeln!(f, "{name}: not reported")
    } else {
        writeln!(f, "{name} = {stat:.2}, df = {df}, p = {p:.4}")
    }
}

impl fmt::Display for CoxResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Cox Proportional Hazards Regression")?;
        writeln!(f, "====================================")?;
        writeln!(f, "n = {}, events = {}", self.n_obs, self.n_events)?;
        writeln!(f, "Ties method: {}", self.ties_method)?;
        match (self.se_type, &self.cluster, self.n_clusters) {
            (CoxSeType::Cluster, Some(c), Some(g)) => writeln!(
                f,
                "Standard errors: robust, clustered by {c} ({g} clusters; Lin-Wei sandwich)"
            )?,
            (t, _, _) => writeln!(f, "Standard errors: {t}")?,
        }
        writeln!(f)?;

        let robust = self.se_type != CoxSeType::Model;
        let width = if robust { 101 } else { 90 };
        if robust {
            writeln!(
                f,
                "{:<20} {:>10} {:>10} {:>10} {:>8} {:>8} {:>10} {:>16}",
                "Variable", "coef", "se(coef)", "robust se", "z", "p", "HR", "95% CI HR"
            )?;
        } else {
            writeln!(
                f,
                "{:<20} {:>10} {:>10} {:>8} {:>8} {:>10} {:>16}",
                "Variable", "coef", "se(coef)", "z", "p", "HR", "95% CI HR"
            )?;
        }
        writeln!(f, "{}", "-".repeat(width))?;

        for i in 0..self.variables.len() {
            write!(
                f,
                "{:<20} {:>10.4} {:>10.4}",
                self.variables[i],
                self.coefficients[i],
                self.naive_std_errors
                    .get(i)
                    .copied()
                    .unwrap_or(self.std_errors[i]),
            )?;
            if robust {
                write!(f, " {:>10.4}", self.std_errors[i])?;
            }
            writeln!(
                f,
                " {:>8.2} {:>8.4}{} {:>10.4} [{:.3}, {:.3}]",
                self.z_stats[i],
                self.p_values[i],
                self.significance[i].stars(),
                self.hazard_ratios[i],
                self.hr_ci_lower[i],
                self.hr_ci_upper[i]
            )?;
        }

        writeln!(f, "{}", "-".repeat(width))?;
        writeln!(f)?;
        writeln!(
            f,
            "Concordance = {:.3} (se = {:.3})",
            self.concordance, self.concordance_se
        )?;
        let df = self.df();
        write_test_line(
            f,
            "Likelihood ratio test",
            self.lr_test,
            df,
            self.lr_p_value,
        )?;
        write_test_line(f, "Wald test", self.wald_test, df, self.wald_p_value)?;
        write_test_line(
            f,
            "Score (logrank) test",
            self.score_test,
            df,
            self.score_p_value,
        )?;
        if let Some(rs) = self.robust_score_test {
            write_test_line(
                f,
                "Robust score test",
                rs,
                df,
                self.robust_score_p_value.unwrap_or(f64::NAN),
            )?;
            writeln!(
                f,
                "Note: the likelihood ratio and score tests assume independent observations \
                 within a cluster; the Wald and robust score tests do not."
            )?;
        }
        writeln!(f)?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;
        write!(f, "{}", notes_text(self))?;
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
    if config.max_iter == 0 {
        return Err(EconError::InvalidSpecification {
            message: "Cox regression: max_iter must be at least 1, got 0".to_string(),
        });
    }
    if !(config.tolerance.is_finite() && config.tolerance > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Cox regression: tolerance must be finite and > 0, got {}",
                config.tolerance
            ),
        });
    }
    if x_cols.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "Cox model requires at least one covariate".to_string(),
        });
    }

    let data = extract_survival(
        dataset.df(),
        time_col,
        event_col,
        config.cluster.as_deref(),
        x_cols,
        TimeRule::NonNegative,
        "Cox regression",
    )?;
    let clusters = data
        .group
        .as_ref()
        .map(|g| (g.codes.as_slice(), g.levels.len()));
    if let (Some(c), Some((_, g))) = (config.cluster.as_deref(), clusters)
        && g < 2
    {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Cox regression: cluster column '{c}' has a single level; clustered standard \
                 errors need at least 2 clusters"
            ),
        });
    }
    let p = x_cols.len();
    let n = data.time.len();
    if n < p + 1 {
        return Err(EconError::InsufficientData {
            required: p + 1,
            provided: n,
            context: format!("Cox regression with {} covariates", p),
        });
    }
    let n_events = data.event.iter().filter(|&&e| e).count();
    if n_events == 0 {
        return Err(EconError::InvalidSpecification {
            message: "No events observed in the data".to_string(),
        });
    }
    let names: Vec<String> = x_cols.iter().map(|s| s.to_string()).collect();

    // Aliasing: a column of [1 | x] that the pivoted QR drops. The intercept stands in for the
    // baseline hazard, so a constant covariate is aliased with it; of two collinear covariates
    // the later one is aliased, as coxph decides.
    let aliased_idx = cox_aliased(&data.x, n)?;
    let id: Vec<usize> = (0..p).filter(|j| !aliased_idx.contains(j)).collect();
    let aliased: Vec<String> = aliased_idx.iter().map(|&j| names[j].clone()).collect();
    if id.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Cox regression: every covariate is constant or collinear with earlier ones \
                 (aliased: {})",
                aliased.join(", ")
            ),
        });
    }
    let q = id.len();
    let mut warnings: Vec<String> = aliasing_message(&aliased_idx, &names).into_iter().collect();

    // Identified columns, centred and scaled as coxph does: the partial likelihood does not
    // change under centring, and scaling makes the iteration independent of the units.
    let mut xs = try_zeros2(n, q, "Cox design")?;
    let mut scale = Array1::<f64>::zeros(q);
    for (c, &j) in id.iter().enumerate() {
        let col = &data.x[j];
        let centre = col.iter().sum::<f64>() / n as f64;
        let mad = col.iter().map(|v| (v - centre).abs()).sum::<f64>() / n as f64;
        let s = 1.0 / mad;
        scale[c] = s;
        for (cell, &v) in xs.column_mut(c).iter_mut().zip(col) {
            *cell = (v - centre) * s;
        }
    }

    let rs = CoxRiskSets::new(&data.time);
    let fit = cox_newton(
        &rs,
        xs.view(),
        &data.event,
        config.ties,
        config.max_iter,
        config.tolerance,
    )?;
    let tol = config.tolerance;

    let v_s = sym_inverse(
        fit.last.information.view(),
        SingularPolicy::Error,
        "Cox information matrix",
    )?
    .inverse;
    let beta_id: Array1<f64> = &fit.beta * &scale;
    let mut v_model = v_s.clone();
    for j in 0..q {
        for k in 0..q {
            v_model[[j, k]] *= scale[j] * scale[k];
        }
    }

    // coxph.fit's warnings
    if fit.convergence.converged {
        let step = v_s.dot(&fit.last.score) * &scale;
        let flagged: Vec<String> = (0..q)
            .filter(|&c| {
                !fit.last.score[c].is_finite()
                    || (step[c].abs() > tol && step[c].abs() > tol.sqrt() * beta_id[c].abs())
            })
            .map(|c| format!("{} ({})", id[c] + 1, names[id[c]]))
            .collect();
        if !flagged.is_empty() {
            warnings.push(format!(
                "Loglik converged before variable {}; coefficient may be infinite",
                flagged.join(", ")
            ));
        }
    } else {
        if let Some(note) = fit.convergence.note("Cox regression") {
            warnings.push(note);
        }
        let eta = xs.dot(&fit.beta);
        if eta.iter().any(|e| !e.is_finite() || e.abs() > 500.0) || !fit.last.loglik.is_finite() {
            warnings.push("one or more coefficients may be infinite".to_string());
        }
    }

    let se_model: Vec<f64> = (0..q).map(|j| v_model[[j, j]].sqrt()).collect();
    let robust = config.robust_se || config.cluster.is_some();
    let (se_id, wald_test, se_type, n_clusters, robust_score_test) = if robust {
        // Lin-Wei: dfbeta = U V (scaled space), summed within clusters; V_r = dfbeta' dfbeta.
        let u = cox_score_residuals(&rs, xs.view(), &data.event, fit.beta.view(), config.ties)?;
        let dfb = collapse_clusters(u.dot(&v_s), clusters)?;
        checkpoint()?;
        let v_r_s = dfb.t().dot(&dfb);
        let se: Vec<f64> = (0..q).map(|j| v_r_s[[j, j]].sqrt() * scale[j]).collect();
        let g_text = clusters.map_or(String::new(), |(_, g)| format!(" ({g} clusters)"));
        // Both tests are scale-free, so they are computed in the scaled space.
        let wald = match quadratic_form_full_rank(&v_r_s, &fit.beta, "Cox robust variance")? {
            Ok(w) => w,
            Err(rank) => {
                warnings.push(format!(
                    "robust Wald test not reported: the robust variance matrix has rank {rank} \
                     of {q}{g_text}"
                ));
                f64::NAN
            }
        };
        let u0 = cox_score_residuals(
            &rs,
            xs.view(),
            &data.event,
            Array1::zeros(q).view(),
            config.ties,
        )?;
        let u0 = collapse_clusters(u0, clusters)?;
        let score0 = u0.sum_axis(ndarray::Axis(0));
        let rscore = match quadratic_form_full_rank(
            &u0.t().dot(&u0),
            &score0,
            "Cox score residual cross-product",
        )? {
            Ok(v) => v,
            Err(rank) => {
                warnings.push(format!(
                    "robust score test not reported: the score residual cross-product has rank \
                     {rank} of {q}{g_text}"
                ));
                f64::NAN
            }
        };
        let kind = if clusters.is_some() {
            CoxSeType::Cluster
        } else {
            CoxSeType::Robust
        };
        (se, wald, kind, clusters.map(|(_, g)| g), Some(rscore))
    } else {
        let wald = fit.beta.dot(&fit.last.information.dot(&fit.beta));
        (se_model.clone(), wald, CoxSeType::Model, None, None)
    };

    // Per-variable vectors over every covariate, NaN for the aliased ones.
    let spread = |v: &[f64]| -> Vec<f64> {
        let mut out = vec![f64::NAN; p];
        for (c, &j) in id.iter().enumerate() {
            out[j] = v[c];
        }
        out
    };
    let coefficients = spread(&beta_id.to_vec());
    let std_errors = spread(&se_id);
    let zc = z_critical(0.95)?;
    let z_stats: Vec<f64> = coefficients
        .iter()
        .zip(&std_errors)
        .map(|(b, s)| b / s)
        .collect();
    let p_values: Vec<f64> = z_stats
        .iter()
        .map(|&z| normal_p_value_two_sided(z))
        .collect();
    let significance = p_values
        .iter()
        .map(|&pv| SignificanceLevel::from_p_value(pv))
        .collect();
    let hazard_ratios = coefficients.iter().map(|b| b.exp()).collect();
    let hr_ci_lower = coefficients
        .iter()
        .zip(&std_errors)
        .map(|(b, s)| (b - zc * s).exp())
        .collect();
    let hr_ci_upper = coefficients
        .iter()
        .zip(&std_errors)
        .map(|(b, s)| (b + zc * s).exp())
        .collect();

    let df = q as f64;
    let wald_p_value = chi_squared_p_value(wald_test, df);
    let info0 = sym_inverse(
        fit.at_zero.information.view(),
        SingularPolicy::Error,
        "Cox information matrix at beta = 0",
    )?
    .inverse;
    let score_test = fit.at_zero.score.dot(&info0.dot(&fit.at_zero.score));
    let score_p_value = chi_squared_p_value(score_test, df);
    let log_likelihood = fit.last.loglik;
    let log_likelihood_null = fit.at_zero.loglik;
    let lr_test = 2.0 * (log_likelihood - log_likelihood_null);
    let lr_p_value = chi_squared_p_value(lr_test, df);

    // Concordance on the identified original-scale columns.
    let mut x_id = try_zeros2(n, q, "Cox covariate matrix")?;
    for (c, &j) in id.iter().enumerate() {
        for (cell, &v) in x_id.column_mut(c).iter_mut().zip(&data.x[j]) {
            *cell = v;
        }
    }
    let obs_data: Vec<(f64, bool, usize)> = data
        .time
        .iter()
        .zip(&data.event)
        .enumerate()
        .map(|(i, (&t, &e))| (t, e, i))
        .collect();
    let (concordance, concordance_se) = compute_concordance(&obs_data, &x_id, &beta_id);

    Ok(CoxResult {
        variables: names,
        coefficients,
        naive_std_errors: spread(&se_model),
        se_type,
        cluster: config.cluster.clone(),
        n_clusters,
        robust_score_p_value: robust_score_test.map(|t| chi_squared_p_value(t, df)),
        robust_score_test,
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
        converged: fit.convergence.converged,
        iterations: fit.convergence.iterations,
        missing: data.missing,
        aliased,
        warnings,
        convergence: fit.convergence,
    })
}

/// The covariates (0-based) that the pivoted QR of `[1 | x]` drops (R's `dqrdc2`, tolerance
/// `LM_QR_TOL`). A well-conditioned full-rank design is certified by its cross-product first.
fn cox_aliased(cols: &[Vec<f64>], n: usize) -> EconResult<Vec<usize>> {
    let p = cols.len();
    let mut xa = try_zeros2(n, p + 1, "Cox design")?;
    xa.column_mut(0).fill(1.0);
    for (j, col) in cols.iter().enumerate() {
        for (cell, &v) in xa.column_mut(j + 1).iter_mut().zip(col) {
            *cell = v;
        }
    }
    if normal_equations_inverse(xa.view())?.is_some() {
        return Ok(Vec::new());
    }
    Ok(Qr::decompose(xa.view(), LM_QR_TOL)?
        .aliased()
        .into_iter()
        .filter(|&j| j >= 1)
        .map(|j| j - 1)
        .collect())
}

/// The Newton fit of [`cox_newton`]: the estimate (scaled space), the pass at it, the pass at
/// beta = 0 and the convergence record.
struct CoxFit {
    beta: Array1<f64>,
    last: CoxPass,
    at_zero: CoxPass,
    convergence: Convergence,
}

/// coxph's iteration (coxfit6): Newton steps from beta = 0, step halving when the log partial
/// likelihood falls (or is not finite), convergence on its relative change.
fn cox_newton(
    rs: &CoxRiskSets,
    x: ArrayView2<f64>,
    event: &[bool],
    ties: TiesMethod,
    max_iter: usize,
    tol: f64,
) -> EconResult<CoxFit> {
    let q = x.ncols();
    let solve = |pass: &CoxPass| -> EconResult<Array1<f64>> {
        Ok(sym_inverse(
            pass.information.view(),
            SingularPolicy::Error,
            "Cox information matrix",
        )?
        .inverse
        .dot(&pass.score))
    };
    checkpoint()?;
    let at_zero = cox_pass(rs, x, event, Array1::zeros(q).view(), ties);
    let mut beta = Array1::<f64>::zeros(q);
    let mut newbeta = solve(&at_zero)?;
    let mut ll_old = at_zero.loglik;
    let mut halving = false;
    let mut converged = false;
    let mut change = f64::NAN;
    let mut it = 0;
    let mut last = None;
    while it < max_iter {
        it += 1;
        checkpoint()?;
        let pass = cox_pass(rs, x, event, newbeta.view(), ties);
        change = relative_change(pass.loglik, ll_old);
        if pass.loglik.is_finite() && change <= tol && !halving {
            converged = true;
            last = Some(pass);
            break;
        }
        if it == max_iter {
            last = Some(pass);
            break;
        }
        if !pass.loglik.is_finite() || pass.loglik < ll_old {
            halving = true;
            newbeta = (&newbeta + &beta) / 2.0;
        } else {
            halving = false;
            ll_old = pass.loglik;
            let step = solve(&pass)?;
            beta = newbeta.clone();
            newbeta = &newbeta + &step;
        }
    }
    let last = match last {
        Some(pass) => pass,
        None => cox_pass(rs, x, event, newbeta.view(), ties),
    };
    Ok(CoxFit {
        beta: newbeta,
        last,
        at_zero,
        convergence: Convergence::new(
            converged,
            it,
            max_iter,
            "relative change in the log partial likelihood",
            change,
            tol,
        ),
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

/// Score residuals (R `residuals(fit, type = "score")`, Lin & Wei 1989): one row per
/// observation in data row order (n x q), summing to the score at `beta`. With Efron ties a
/// death's share of the k-th tied term is (1 - k/d), as coxph weights it.
fn cox_score_residuals(
    rs: &CoxRiskSets,
    x: ArrayView2<f64>,
    event: &[bool],
    beta: ArrayView1<f64>,
    ties: TiesMethod,
) -> EconResult<Array2<f64>> {
    let (n, q) = x.dim();
    let nb = rs.blocks.len();
    let len = crate::resource::checked_len(&[nb, q], "Cox score residuals")?;
    let risk: Vec<f64> = x.dot(&beta).iter().map(|e| e.exp()).collect();
    // Per block: h, hd (scalars) and hx, hxd, xbm (q each).
    let mut h = crate::resource::try_vec(nb, 0.0, "Cox score residuals")?;
    let mut hd = crate::resource::try_vec(nb, 0.0, "Cox score residuals")?;
    let mut hx = crate::resource::try_vec(len, 0.0, "Cox score residuals")?;
    let mut hxd = crate::resource::try_vec(len, 0.0, "Cox score residuals")?;
    let mut xbm = crate::resource::try_vec(len, 0.0, "Cox score residuals")?;
    let mut ndeath = crate::resource::try_vec(nb, 0usize, "Cox score residuals")?;
    let (mut s0, mut s1) = (0.0, vec![0.0; q]);
    let mut e1 = vec![0.0; q];
    let mut xbar = vec![0.0; q];
    for (b, block) in rs.blocks.iter().enumerate() {
        let rows = &rs.order[block.clone()];
        let (mut d, mut e0) = (0usize, 0.0);
        e1.iter_mut().for_each(|v| *v = 0.0);
        for &i in rows {
            let r = risk[i];
            s0 += r;
            for j in 0..q {
                s1[j] += r * x[[i, j]];
            }
            if event[i] {
                d += 1;
                e0 += r;
                for j in 0..q {
                    e1[j] += r * x[[i, j]];
                }
            }
        }
        ndeath[b] = d;
        if d == 0 {
            continue;
        }
        let df = d as f64;
        let off = b * q;
        for k in 0..d {
            let f = match ties {
                TiesMethod::Efron => k as f64 / df,
                TiesMethod::Breslow => 0.0,
            };
            let a = s0 - f * e0;
            for j in 0..q {
                xbar[j] = (s1[j] - f * e1[j]) / a;
            }
            let c = 1.0 - f;
            h[b] += 1.0 / a;
            hd[b] += c / a;
            for j in 0..q {
                hx[off + j] += xbar[j] / a;
                hxd[off + j] += c * xbar[j] / a;
                xbm[off + j] += xbar[j] / df;
            }
        }
    }
    checkpoint()?;
    let mut u = try_zeros2(n, q, "Cox score residuals")?;
    let (mut cum_h, mut cum_a) = (0.0, vec![0.0; q]);
    for (b, block) in rs.blocks.iter().enumerate().rev() {
        let off = b * q;
        for &i in &rs.order[block.clone()] {
            let r = risk[i];
            for j in 0..q {
                let xij = x[[i, j]];
                let mut v = -r * (xij * cum_h - cum_a[j]);
                if ndeath[b] > 0 {
                    if event[i] {
                        v += (xij - xbm[off + j]) - r * (xij * hd[b] - hxd[off + j]);
                    } else {
                        v -= r * (xij * h[b] - hx[off + j]);
                    }
                }
                u[[i, j]] = v;
            }
        }
        cum_h += h[b];
        for j in 0..q {
            cum_a[j] += hx[off + j];
        }
    }
    Ok(u)
}

/// `rows` summed within the clusters `codes` (G x q), or `rows` itself without clusters.
fn collapse_clusters(rows: Array2<f64>, codes: Option<(&[u32], usize)>) -> EconResult<Array2<f64>> {
    let Some((codes, g)) = codes else {
        return Ok(rows);
    };
    let q = rows.ncols();
    let mut out = try_zeros2(g, q, "Cox cluster sums")?;
    for (row, &c) in rows.rows().into_iter().zip(codes) {
        let mut o = out.row_mut(c as usize);
        o += &row;
    }
    Ok(out)
}

/// `b' M^-1 b` through a generalized inverse of the symmetric `m`; `Err(rank)` when `m` is
/// rank-deficient (the test is then not reported).
fn quadratic_form_full_rank(
    m: &Array2<f64>,
    b: &Array1<f64>,
    what: &str,
) -> EconResult<Result<f64, usize>> {
    let inv = sym_inverse(m.view(), SingularPolicy::Generalized, what)?;
    if !inv.is_full_rank() {
        return Ok(Err(inv.rank));
    }
    Ok(Ok(b.dot(&inv.inverse.dot(b))))
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
    /// Maximum Newton iterations (at least 1).
    pub max_iter: usize,
    /// Convergence: relative change in the log-likelihood (survreg.control `rel.tolerance`);
    /// finite and > 0.
    pub tolerance: f64,
}

impl Default for AftConfig {
    fn default() -> Self {
        Self {
            distribution: AftDistribution::Weibull,
            max_iter: 100,
            tolerance: 1e-9,
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
    /// SE of log(scale) (survreg's "Log(scale)" row); `None` when the scale is fixed.
    #[serde(default)]
    pub log_scale_se: Option<f64>,
    /// True when the scale is fixed at 1 (exponential), not estimated.
    #[serde(default)]
    pub scale_fixed: bool,
    /// Covariates not estimated because they are constant or collinear with earlier ones
    /// (coefficient NA in survreg); their entries in every per-variable vector are NaN.
    #[serde(default)]
    pub aliased: Vec<String>,
    /// Warnings (aliasing, non-convergence).
    #[serde(default)]
    pub warnings: Vec<String>,
    /// The Newton iteration's convergence record.
    #[serde(default)]
    pub convergence: Convergence,
}

impl AftResult {
    /// (z, p) of survreg's Log(scale) row: ln(scale) / log_scale_se and its two-sided normal
    /// p-value; `None` when the scale is fixed.
    pub fn log_scale_test(&self) -> Option<(f64, f64)> {
        let se = self.log_scale_se?;
        let z = self.scale.ln() / se;
        Some((z, normal_p_value_two_sided(z)))
    }
}

impl ResultNotes for AftResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
    fn notes_convergence(&self) -> Option<Convergence> {
        Some(self.convergence.clone())
    }
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
        if let (Some(se), Some((z, p))) = (self.log_scale_se, self.log_scale_test()) {
            writeln!(
                f,
                "{:<20} {:>10.4} {:>10.4} {:>8.2} {:>8.4}{}",
                "Log(scale)",
                self.scale.ln(),
                se,
                z,
                p,
                SignificanceLevel::from_p_value(p).stars()
            )?;
        }

        writeln!(f, "{}", "-".repeat(75))?;
        if self.scale_fixed {
            writeln!(f, "Scale fixed at 1 (exponential)")?;
        } else {
            writeln!(f, "Scale = {:.4}", self.scale)?;
        }
        if let Some(shape) = self.shape {
            writeln!(f, "Shape = {:.4}", shape)?;
        }
        writeln!(f)?;
        writeln!(
            f,
            "Log-Likelihood: {:.4} (time scale, as survreg)",
            self.log_likelihood
        )?;
        writeln!(f, "AIC: {:.4}", self.aic)?;
        writeln!(f, "BIC: {:.4}", self.bic)?;
        writeln!(f)?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;
        write!(f, "{}", notes_text(self))?;
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
    if config.max_iter == 0 {
        return Err(EconError::InvalidSpecification {
            message: "AFT model: max_iter must be at least 1, got 0".to_string(),
        });
    }
    if !(config.tolerance.is_finite() && config.tolerance > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "AFT model: tolerance must be finite and > 0, got {}",
                config.tolerance
            ),
        });
    }
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
    if n < p + 1 {
        return Err(EconError::InsufficientData {
            required: p + 1,
            provided: n,
            context: format!("AFT model with {} parameters", p),
        });
    }
    let event = data.event;
    let n_events = event.iter().filter(|&&d| d).count();
    if n_events == 0 {
        return Err(EconError::InvalidSpecification {
            message: "No events observed in the data".to_string(),
        });
    }
    let mut variables = vec!["(Intercept)".to_string()];
    variables.extend(x_cols.iter().map(|s| s.to_string()));

    let mut x = try_zeros2(n, p, "AFT design")?;
    x.column_mut(0).fill(1.0);
    for (j, values) in data.x.iter().enumerate() {
        for (cell, &v) in x.column_mut(j + 1).iter_mut().zip(values) {
            *cell = v;
        }
    }
    let y: Array1<f64> = data.time.iter().map(|t| t.ln()).collect();

    // Start: least squares on log t; its pivoted QR decides the aliased covariates (survreg's
    // NA coefficients).
    let ls = solve_ols(x.view(), y.view())?;
    let mut warnings: Vec<String> = aliasing_message(&ls.aliased, &variables)
        .into_iter()
        .collect();
    let id: Vec<usize> = (0..p).filter(|j| !ls.aliased.contains(j)).collect();
    let q = id.len();
    let xi = x.select(ndarray::Axis(1), &id);
    let fixed = config.distribution == AftDistribution::Exponential;
    let rss: f64 = ls.residuals.iter().map(|r| r * r).sum();
    let df = n.saturating_sub(ls.rank).max(1);
    let tau0 = if fixed || rss.is_nan() || rss <= 0.0 {
        0.0
    } else {
        0.5 * (rss / df as f64).ln()
    };
    let m = q + usize::from(!fixed);
    let mut theta = Array1::<f64>::zeros(m);
    for (c, &j) in id.iter().enumerate() {
        theta[c] = ls.beta[j];
    }
    if !fixed {
        theta[q] = tau0;
    }
    let mut param_names: Vec<String> = id.iter().map(|&j| variables[j].clone()).collect();
    if !fixed {
        param_names.push("Log(scale)".to_string());
    }

    let dist = config.distribution;
    let tol = config.tolerance;
    let eval = |th: &Array1<f64>| aft_eval(y.view(), xi.view(), &event, th.view(), dist, fixed);
    checkpoint()?;
    let e0 = eval(&theta);
    let mut ll_old = e0.loglik;
    let mut theta_new = &theta + &damped_newton_step(&e0.neg_hessian, &e0.grad, &param_names)?;
    let mut halving = false;
    let mut converged = false;
    let mut change = f64::NAN;
    let mut it = 0;
    let mut at_new = None;
    while it < config.max_iter {
        it += 1;
        checkpoint()?;
        let e = eval(&theta_new);
        change = relative_change(e.loglik, ll_old);
        if e.loglik.is_finite() && change <= tol && !halving {
            converged = true;
            theta = theta_new.clone();
            at_new = Some(e);
            break;
        }
        if it == config.max_iter {
            break;
        }
        if !e.loglik.is_finite() || e.loglik < ll_old {
            halving = true;
            theta_new = (&theta_new + &theta) / 2.0;
        } else {
            halving = false;
            ll_old = e.loglik;
            theta = theta_new.clone();
            theta_new = &theta + &damped_newton_step(&e.neg_hessian, &e.grad, &param_names)?;
        }
    }
    let convergence = Convergence::new(
        converged,
        it,
        config.max_iter,
        "relative change in the log-likelihood",
        change,
        tol,
    );
    let e = match at_new {
        Some(e) => e,
        None => eval(&theta),
    };
    let var_diag_ok = |v: &Array2<f64>| (0..m).map(|j| v[[j, j]]).collect::<Vec<f64>>();
    let v = match sym_inverse(
        e.neg_hessian.view(),
        SingularPolicy::Error,
        "AFT information matrix",
    ) {
        Ok(inv) => inv.inverse,
        Err(_) => {
            return Err(aft_failure(it, change));
        }
    };
    let var_diag = var_diag_ok(&v);
    aft_check_finite(
        e.loglik,
        theta.as_slice().unwrap_or(&[]),
        &var_diag,
        it,
        change,
    )?;
    if let Some(note) = convergence.note("AFT model") {
        warnings.push(note);
    }

    let mut coefficients = vec![f64::NAN; p];
    let mut std_errors = vec![f64::NAN; p];
    for (c, &j) in id.iter().enumerate() {
        coefficients[j] = theta[c];
        std_errors[j] = var_diag[c].sqrt();
    }
    let z_stats: Vec<f64> = coefficients
        .iter()
        .zip(&std_errors)
        .map(|(b, s)| b / s)
        .collect();
    let p_values: Vec<f64> = z_stats
        .iter()
        .map(|&z| normal_p_value_two_sided(z))
        .collect();
    let significance = p_values
        .iter()
        .map(|&pv| SignificanceLevel::from_p_value(pv))
        .collect();
    let acceleration_factors = coefficients.iter().map(|b| b.exp()).collect();
    let (scale, log_scale_se) = if fixed {
        (1.0, None)
    } else {
        (theta[q].exp(), Some(var_diag[q].sqrt()))
    };
    let shape = match dist {
        AftDistribution::Weibull => Some(1.0 / scale),
        AftDistribution::Exponential => Some(1.0),
        _ => None,
    };
    // survreg's log-likelihood is on the time scale: the log-T density carries the Jacobian 1/t.
    let jacobian: f64 = y
        .iter()
        .zip(&event)
        .filter(|(_, e)| **e)
        .map(|(v, _)| *v)
        .sum();
    let log_likelihood = e.loglik - jacobian;
    let k = m as f64;
    let aic = -2.0 * log_likelihood + 2.0 * k;
    let bic = -2.0 * log_likelihood + k * (n as f64).ln();
    let aliased = ls.aliased.iter().map(|&j| variables[j].clone()).collect();

    Ok(AftResult {
        distribution: dist,
        variables,
        coefficients,
        std_errors,
        z_stats,
        p_values,
        significance,
        acceleration_factors,
        scale,
        shape,
        log_likelihood,
        aic,
        bic,
        n_obs: n,
        n_events,
        converged,
        iterations: it,
        missing: data.missing,
        log_scale_se,
        scale_fixed: fixed,
        aliased,
        warnings,
        convergence,
    })
}

fn aft_failure(iterations: usize, last_change: f64) -> EconError {
    EconError::ConvergenceFailure {
        iterations,
        last_change,
        suggestion: "the AFT log-likelihood or its information is not finite or not positive \
                     definite at the last iterate; if the fit stopped at max_iter, raise \
                     max_iter; otherwise check for a covariate that separates events from \
                     censorings, or rescale extreme times"
            .to_string(),
    }
}

/// R2: a fit whose log-likelihood, parameters or variances are not finite (or a variance is
/// negative) is an error, not a returned result.
fn aft_check_finite(
    loglik: f64,
    theta: &[f64],
    var_diag: &[f64],
    iterations: usize,
    last_change: f64,
) -> EconResult<()> {
    let ok = loglik.is_finite()
        && theta.iter().all(|v| v.is_finite())
        && var_diag.iter().all(|v| v.is_finite() && *v >= 0.0);
    if ok {
        Ok(())
    } else {
        Err(aft_failure(iterations, last_change))
    }
}

/// Log-likelihood on the log-time scale, its gradient and the negative Hessian over the free
/// parameters (beta, then tau = ln sigma unless the scale is fixed at 1).
struct AftEval {
    loglik: f64,
    grad: Array1<f64>,
    neg_hessian: Array2<f64>,
}

fn aft_eval(
    y: ArrayView1<f64>,
    x: ArrayView2<f64>,
    event: &[bool],
    theta: ArrayView1<f64>,
    dist: AftDistribution,
    fixed_scale: bool,
) -> AftEval {
    let q = x.ncols();
    let m = q + usize::from(!fixed_scale);
    let tau = if fixed_scale { 0.0 } else { theta[q] };
    let sigma = tau.exp();
    let inv_s = 1.0 / sigma;
    let eta = x.dot(&theta.slice(ndarray::s![..q]));
    let mut loglik = 0.0;
    let mut grad = Array1::<f64>::zeros(m);
    // Hessian (not negated) accumulated on the upper triangle
    let mut h = Array2::<f64>::zeros((m, m));
    for i in 0..y.len() {
        let z = (y[i] - eta[i]) * inv_s;
        let (l0, l1, l2) = if event[i] {
            let (g, g1, g2) = aft_log_density(z, dist);
            (g - tau, g1, g2)
        } else {
            aft_log_survival(z, dist)
        };
        loglik += l0;
        let xi = x.row(i);
        let wb = -l1 * inv_s;
        let wbb = l2 * inv_s * inv_s;
        for j in 0..q {
            grad[j] += wb * xi[j];
            let a = wbb * xi[j];
            for k in j..q {
                h[[j, k]] += a * xi[k];
            }
        }
        if !fixed_scale {
            grad[q] += -l1 * z - if event[i] { 1.0 } else { 0.0 };
            let wbt = (l2 * z + l1) * inv_s;
            for j in 0..q {
                h[[j, q]] += wbt * xi[j];
            }
            h[[q, q]] += l2 * z * z + l1 * z;
        }
    }
    let mut neg_hessian = Array2::<f64>::zeros((m, m));
    for j in 0..m {
        for k in j..m {
            neg_hessian[[j, k]] = -h[[j, k]];
            neg_hessian[[k, j]] = -h[[j, k]];
        }
    }
    AftEval {
        loglik,
        grad,
        neg_hessian,
    }
}

const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_8;

/// `(g, g', g'')` with `g = ln f` of the error distribution at `z`.
fn aft_log_density(z: f64, dist: AftDistribution) -> (f64, f64, f64) {
    match dist {
        AftDistribution::Exponential | AftDistribution::Weibull => {
            let ez = z.exp();
            (z - ez, 1.0 - ez, -ez)
        }
        AftDistribution::LogNormal => (-0.5 * z * z - LN_SQRT_2PI, -z, -1.0),
        AftDistribution::LogLogistic => {
            let pz = logistic(z);
            (
                z - 2.0 * softplus(z),
                1.0 - 2.0 * pz,
                -2.0 * pz * (1.0 - pz),
            )
        }
    }
}

/// `(G, G', G'')` with `G = ln S` of the error distribution at `z`.
fn aft_log_survival(z: f64, dist: AftDistribution) -> (f64, f64, f64) {
    match dist {
        AftDistribution::Exponential | AftDistribution::Weibull => {
            let ez = z.exp();
            (-ez, -ez, -ez)
        }
        AftDistribution::LogNormal => {
            let (ls, m) = normal_ln_sf_hazard(z);
            (ls, -m, -m * (m - z))
        }
        AftDistribution::LogLogistic => {
            let pz = logistic(z);
            (-softplus(z), -pz, -pz * (1.0 - pz))
        }
    }
}

/// `ln(1 - Phi(z))` without underflow (R `pnorm(z, lower.tail = FALSE, log.p = TRUE)`).
#[cfg(test)]
fn ln_normal_sf(z: f64) -> f64 {
    normal_ln_sf_hazard(z).0
}

/// The normal hazard `phi(z) / (1 - Phi(z))`, exact in both tails.
#[cfg(test)]
fn normal_hazard(z: f64) -> f64 {
    normal_ln_sf_hazard(z).1
}

/// `(ln(1 - Phi(z)), phi(z) / (1 - Phi(z)))` in one evaluation.
///
/// Cody's rational approximations as in R's `pnorm_both` (about 1e-16 relative in the tail
/// it evaluates). Above 0.674 the tail is `exp(-z^2 / 2) r(z)` with `r` rational, so the
/// hazard is `1 / (sqrt(2 pi) r(z))` without evaluating the exponential; `exp(-z^2 / 2)` is
/// formed as R does, splitting `z^2` so the log keeps full precision.
#[allow(clippy::excessive_precision)] // Cody's coefficients as R's pnorm.c gives them
fn normal_ln_sf_hazard(z: f64) -> (f64, f64) {
    const A: [f64; 5] = [
        2.235_252_035_460_683_928_7,
        161.028_231_068_555_878_81,
        1_067.689_485_460_370_958_2,
        18_154.981_253_343_561_249,
        0.065_682_337_918_207_449_113,
    ];
    const B: [f64; 4] = [
        47.202_581_904_688_241_87,
        976.098_551_737_776_693_22,
        10_260.932_208_618_978_205,
        45_507.789_335_026_729_956,
    ];
    const C: [f64; 9] = [
        0.398_941_512_088_134_667_64,
        8.883_149_794_388_375_941_2,
        93.506_656_132_177_855_979,
        597.270_276_394_800_262_26,
        2_494.537_585_290_372_671_1,
        6_848.190_450_536_282_332_6,
        11_602.651_437_647_350_124,
        9_842.714_838_383_978_021_8,
        1.076_557_677_372_019_231_7e-8,
    ];
    const D: [f64; 8] = [
        22.266_688_044_328_115_691,
        235.387_901_782_624_998_61,
        1_519.377_599_407_554_805,
        6_485.558_298_266_760_755,
        18_615.571_640_885_098_091,
        34_900.952_721_145_977_266,
        38_912.003_286_093_271_411,
        19_685.429_676_859_990_727,
    ];
    const P: [f64; 6] = [
        0.215_898_534_057_956_99,
        0.127_401_161_160_247_363_9,
        0.022_235_277_870_649_807,
        0.001_421_619_193_227_893_466,
        2.911_287_495_116_879_2e-5,
        0.023_073_441_764_940_173_03,
    ];
    const Q: [f64; 5] = [
        1.284_260_096_144_911_21,
        0.468_238_212_480_865_118,
        0.065_988_137_868_928_551_5,
        0.003_782_396_332_027_582_44,
        7.297_515_550_839_662_05e-5,
    ];
    const FRAC_1_SQRT_2PI: f64 = 0.398_942_280_401_432_677_94;
    if z.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    let y = z.abs();
    if y <= 0.674_489_75 {
        let (xnum, xden) = if y > 0.5 * f64::EPSILON {
            let xsq = z * z;
            let (mut xnum, mut xden) = (A[4] * xsq, xsq);
            for (a, b) in A.iter().zip(&B).take(3) {
                xnum = (xnum + a) * xsq;
                xden = (xden + b) * xsq;
            }
            (xnum, xden)
        } else {
            (0.0, 0.0)
        };
        let ls = (0.5 - z * (xnum + A[3]) / (xden + B[3])).ln();
        return (ls, (-0.5 * z * z - LN_SQRT_2PI - ls).exp());
    }
    if y >= 1e170 {
        // z^2 overflows: ln S is -inf above and 0 below, the hazard z^+ to working precision.
        return if z > 0.0 {
            (f64::NEG_INFINITY, z)
        } else {
            (0.0, 0.0)
        };
    }
    // r(y) with Phi(-y) = exp(-y^2 / 2) r(y)
    let r = if y <= 32f64.sqrt() {
        let (mut xnum, mut xden) = (C[8] * y, y);
        for (c, d) in C.iter().zip(&D).take(7) {
            xnum = (xnum + c) * y;
            xden = (xden + d) * y;
        }
        (xnum + C[7]) / (xden + D[7])
    } else {
        let xsq = 1.0 / (y * y);
        let (mut xnum, mut xden) = (P[5] * xsq, xsq);
        for (p, q) in P.iter().zip(&Q).take(4) {
            xnum = (xnum + p) * xsq;
            xden = (xden + q) * xsq;
        }
        (FRAC_1_SQRT_2PI - xsq * (xnum + P[4]) / (xden + Q[4])) / y
    };
    let ysq = (y * 16.0).trunc() / 16.0;
    let del = (y - ysq) * (y + ysq);
    let ln_small = -0.5 * ysq * ysq - 0.5 * del + r.ln();
    if z > 0.0 {
        (ln_small, FRAC_1_SQRT_2PI / r)
    } else {
        let ls = (-ln_small.exp()).ln_1p();
        (ls, (-0.5 * z * z - LN_SQRT_2PI - ls).exp())
    }
}

/// `ln(1 + e^z)` without overflow.
fn softplus(z: f64) -> f64 {
    z.max(0.0) + (-z.abs()).exp().ln_1p()
}

/// `1 / (1 + e^-z)`, exact in both tails.
fn logistic(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// Newton step `d` solving `a d = g` (`a` = the negative Hessian) on the diagonally
/// equilibrated system; when that is not positive definite, Levenberg damping `mu I` with
/// `mu` = 1e-8, 1e-7, ..., 1e8.
fn damped_newton_step(
    a: &Array2<f64>,
    g: &Array1<f64>,
    names: &[String],
) -> EconResult<Array1<f64>> {
    let m = g.len();
    let mut d = Array1::<f64>::zeros(m);
    for j in 0..m {
        let v = a[[j, j]].abs();
        if !(v.is_finite() && v > 0.0) {
            return Err(EconError::SingularMatrix {
                context: format!(
                    "AFT information matrix: no information on {}",
                    names.get(j).map_or("a parameter", |s| s.as_str())
                ),
                suggestion: "check for a constant covariate or non-finite values".to_string(),
            });
        }
        d[j] = 1.0 / v.sqrt();
    }
    let mut b = Array2::<f64>::zeros((m, m));
    for j in 0..m {
        for k in 0..m {
            b[[j, k]] = d[j] * a[[j, k]] * d[k];
        }
    }
    let rhs = &d * g;
    let mut mu = 0.0;
    loop {
        let mut bm = b.clone();
        for j in 0..m {
            bm[[j, j]] += mu;
        }
        if let Some(sol) = cholesky_solve_spd(&bm, &rhs) {
            return Ok(&d * &sol);
        }
        mu = if mu == 0.0 { 1e-8 } else { mu * 10.0 };
        if mu > 1e8 * 1.000_001 {
            return Err(EconError::SingularMatrix {
                context: "AFT information matrix is not positive definite even with damping 1e8"
                    .to_string(),
                suggestion: "check for a covariate that separates events from censorings, or \
                             rescale extreme times"
                    .to_string(),
            });
        }
    }
}

/// Solve the symmetric positive definite system Hx = b by Cholesky decomposition; `None`
/// when the matrix is not positive definite.
fn cholesky_solve_spd(h: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
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
    /// Delta-method SE of F(t), as survfit's multi-state `std.err`.
    pub std_errors: Vec<f64>,
    /// Lower limit, log scale as survfit's `conf.type = "log"`; NaN while F(t) = 0.
    pub ci_lower: Vec<f64>,
    /// Upper limit, log scale, capped at 1; NaN while F(t) = 0.
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
    let mut km_surv = 1.0; // Kaplan-Meier for overall survival, S(t-) before each time
    let n_types = event_types_vec.len();
    let mut cum_incidence = crate::resource::try_vec(n_types, 0.0, "competing risks")?;
    // Running sums of the delta-method variance per cause (CifVarianceSums).
    let mut sums =
        crate::resource::try_vec(n_types, CifVarianceSums::default(), "competing risks")?;

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
            let n_j = current_risk as f64;
            let d_j = total_events as f64;
            for (idx, &event_type) in event_types_vec.iter().enumerate() {
                let d_k = *events_at_time.get(&event_type).unwrap_or(&0) as f64;
                // Aalen-Johansen increment
                cum_incidence[idx] += km_surv * d_k / n_j;
                sums[idx].add(n_j, d_j, d_k, km_surv, cum_incidence[idx]);
            }
            // Update Kaplan-Meier for overall survival
            if total_events > 0 {
                km_surv *= 1.0 - d_j / n_j;
            }
        }

        // Store results for this time point
        for (idx, cif) in cifs.iter_mut().enumerate() {
            let f_k = cum_incidence[idx];
            let se = sums[idx].variance(f_k).max(0.0).sqrt();
            cif.times.push(t);
            cif.incidence.push(f_k);
            cif.std_errors.push(se);

            // Confidence limits on the log scale, as survfit's conf.type = "log"; undefined
            // (NaN, R's NA) while the incidence is 0.
            let (lo, hi) = if f_k > 0.0 {
                let log_cif = f_k.ln();
                let se_log = se / f_k;
                (
                    (log_cif - z * se_log).exp().clamp(0.0, 1.0),
                    (log_cif + z * se_log).exp().clamp(0.0, 1.0),
                )
            } else {
                (f64::NAN, f64::NAN)
            };
            cif.ci_lower.push(lo);
            cif.ci_upper.push(hi);
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

/// Running sums of the delta-method variance of one cause's cumulative incidence F_k, as
/// survfit reports it for the multi-state Aalen-Johansen estimate. At the distinct times t_j
/// (n_j at risk, d_j events of any cause, d_kj of cause k, S_{j-1} the overall Kaplan-Meier
/// before t_j and F_j = F_k(t_j)):
///
/// ```text
/// Var F_k(t) = sum_{t_j <= t} [ a_j (F - F_j)^2 + b_j - 2 c_j (F - F_j) ],   F = F_k(t)
///   a_j = d_j / (n_j (n_j - d_j))   (0 when n_j = d_j)
///   b_j = S_{j-1}^2 d_kj (n_j - d_kj) / n_j^3
///   c_j = S_{j-1} d_kj / n_j^2
/// ```
///
/// expanded in F so that each time costs O(1).
#[derive(Debug, Clone, Copy, Default)]
struct CifVarianceSums {
    a0: f64,
    a1: f64,
    a2: f64,
    b: f64,
    c0: f64,
    c1: f64,
}

impl CifVarianceSums {
    fn add(&mut self, n: f64, d: f64, dk: f64, s_prev: f64, f_j: f64) {
        let a = if n > d { d / (n * (n - d)) } else { 0.0 };
        let c = s_prev * dk / (n * n);
        self.a0 += a;
        self.a1 += a * f_j;
        self.a2 += a * f_j * f_j;
        self.b += s_prev * s_prev * dk * (n - dk) / (n * n * n);
        self.c0 += c;
        self.c1 += c * f_j;
    }

    fn variance(&self, f: f64) -> f64 {
        f * f * self.a0 - 2.0 * f * self.a1 + self.a2 + self.b - 2.0 * (f * self.c0 - self.c1)
    }
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
    // survival.R); Cox, AFT and competing risks further down (surv_cox_aft.R,
    // validation/econometrics/survival.md, Test Cases 5-8).
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
    fn test_cox_concordance_properties() {
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
    fn test_competing_risks_cif_sum_properties() {
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
        let err = "gaussian"
            .parse::<AftDistribution>()
            .unwrap_err()
            .to_string();
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

    /// ECON-46: Breslow scores every death at t against the full risk set {t_j >= t} whatever the
    /// row order (coxph(ties = "breslow")). Before: rows entered the risk set one at a time; lung
    /// gave age 0.011070584319508605 in file order and 0.011048660708497025 reversed (R
    /// 0.011041136385707521).
    #[test]
    fn test_validate_cox_breslow_row_order_invariant() {
        let (case, ds) = load_case("surv_cox_robust");
        let n = case.scalar("n") as usize;
        let rev = Dataset::new(ds.df().reverse());
        let cfg = cox_tight(TiesMethod::Breslow);
        let a = run_cox_ph(&ds, "time", "status", &COX_X, Some(cfg.clone())).unwrap();
        let b = run_cox_ph(&rev, "time", "status", &COX_X, Some(cfg)).unwrap();
        for (x, y) in a.coefficients.iter().zip(&b.coefficients) {
            assert!(
                (x - y).abs() <= 1e-12 * y.abs(),
                "row order changed the Breslow fit: {x} vs {y}"
            );
        }
        let mut check = Check::new(&case);
        for r in [&a, &b] {
            check
                .scalar("n", r.n_obs as f64, Tol::exact())
                .named("breslow_coef", &r.variables, &r.coefficients, Tol::coef(n))
                .named("breslow_se", &r.variables, &r.std_errors, Tol::se(n))
                .scalar("breslow_loglik", r.log_likelihood, Tol::stat(n))
                .scalar("breslow_loglik_null", r.log_likelihood_null, Tol::stat(n))
                .scalar("breslow_score_test", r.score_test, Tol::stat(n));
        }
        check.finish();
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
                    .named(
                        &format!("{tag}_coef"),
                        &r.variables,
                        &r.coefficients,
                        Tol::coef(n),
                    )
                    .named(
                        &format!("{tag}_se"),
                        &r.variables,
                        &r.std_errors,
                        Tol::se(n),
                    )
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

    /// Every number summary.coxph reports for the model-based fit (deviation ECON-89: the
    /// concordance and its SE are not pinned). Before: up to 100 iterations under the absolute
    /// score rule, HR limits with 1.96 (sex and ph.ecog upper limits off by 6.0e-6 and 4.1e-6
    /// relative), no convergence fields.
    fn check_cox_model(case_id: &str, ties: TiesMethod) {
        let (case, ds) = load_case(case_id);
        let n = case.scalar("n") as usize;
        let tag = ties_tag(ties);
        let r = run_cox_ph(&ds, "time", "status", &COX_X, Some(cox_tight(ties))).unwrap();
        assert!(
            r.converged && r.convergence.converged,
            "{:?}",
            r.convergence
        );
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
        assert_eq!(r.df(), 3);
        assert!(r.to_string().contains("Converged: yes ("), "{r}");
        assert_eq!(r.se_type, CoxSeType::Model);
        assert_eq!(r.naive_std_errors, r.std_errors);
        Check::new(&case)
            .scalar("n", r.n_obs as f64, Tol::exact())
            .scalar("n_event", r.n_events as f64, Tol::exact())
            .named(
                &format!("{tag}_coef"),
                &r.variables,
                &r.coefficients,
                Tol::coef(n),
            )
            .named(
                &format!("{tag}_se"),
                &r.variables,
                &r.std_errors,
                Tol::se(n),
            )
            .named(&format!("{tag}_z"), &r.variables, &r.z_stats, Tol::stat(n))
            .named(
                &format!("{tag}_p"),
                &r.variables,
                &r.p_values,
                Tol::p_value(n),
            )
            .named(
                &format!("{tag}_hr_lower"),
                &r.variables,
                &r.hr_ci_lower,
                Tol::stat(n),
            )
            .named(
                &format!("{tag}_hr_upper"),
                &r.variables,
                &r.hr_ci_upper,
                Tol::stat(n),
            )
            .scalar(&format!("{tag}_loglik"), r.log_likelihood, Tol::stat(n))
            .scalar(
                &format!("{tag}_loglik_null"),
                r.log_likelihood_null,
                Tol::stat(n),
            )
            .scalar(&format!("{tag}_lr_test"), r.lr_test, Tol::stat(n))
            .scalar(&format!("{tag}_lr_p"), r.lr_p_value, Tol::p_value(n))
            .scalar(&format!("{tag}_score_test"), r.score_test, Tol::stat(n))
            .scalar(&format!("{tag}_score_p"), r.score_p_value, Tol::p_value(n))
            .scalar(&format!("{tag}_wald_test"), r.wald_test, Tol::stat(n))
            .scalar(&format!("{tag}_wald_p"), r.wald_p_value, Tol::p_value(n))
            .finish();
    }

    #[test]
    fn test_validate_cox_efron_model() {
        check_cox_model("surv_cox_robust", TiesMethod::Efron)
    }

    #[test]
    fn test_validate_cox_breslow_model() {
        check_cox_model("surv_cox_robust", TiesMethod::Breslow)
    }

    /// X-15: the stopping rule is the relative change in the log partial likelihood, so the
    /// iteration does not depend on a covariate's units. Before: the absolute max |score| < 1e-9
    /// rule ran to max_iter (25), converged = false, at age x 365250 and x 1e6. Coefficients
    /// (mapped back to age units) are pinned to coxph's tight fit; coxph's default-eps fits are
    /// a cross-check. Iteration counts are p2a's own property (within one of each other).
    #[test]
    fn test_validate_cox_converges_at_large_scale() {
        let (case, ds) = load_case("surv_cox_robust");
        let mut check = Check::new(&case);
        let mut iter_e0 = None;
        for (tag, s) in [("e0", 1.0), ("em6", 1e-6), ("days", 365250.0), ("e6", 1e6)] {
            let d = derive_f64(&ds, "age", "age_s", |_, a| a as f64 * s);
            let mut cfg = CoxConfig::default(); // tolerance 1e-9 = coxph's eps, max_iter 25
            cfg.ties = TiesMethod::Efron;
            let r = run_cox_ph(&d, "time", "status", &["age_s", "sex"], Some(cfg)).unwrap();
            assert!(r.converged, "age x {s}: {:?}", r.convergence);
            assert!(r.iterations <= 10, "age x {s}: {} iterations", r.iterations);
            let base = *iter_e0.get_or_insert(r.iterations);
            assert!(
                r.iterations.abs_diff(base) <= 1,
                "age x {s}: {} vs {base}",
                r.iterations
            );
            let back = [r.coefficients[0] * s, r.coefficients[1]];
            check
                .named(
                    "scale_tight_coef",
                    &["age", "sex"],
                    &back,
                    Tol::iterative(1e-7),
                )
                .named(
                    &format!("scale_{tag}_coef"),
                    &["age", "sex"],
                    &[r.coefficients[0], r.coefficients[1]],
                    Tol::iterative(1e-6),
                );
        }
        check.finish();
    }

    /// R10: stopping at max_iter is reported, not hidden (coxph: "Ran out of iterations and did
    /// not converge").
    #[test]
    fn test_cox_iteration_limit_reports_non_convergence() {
        let ds = generated_dataset("lung");
        let mut cfg = cox_tight(TiesMethod::Efron);
        cfg.max_iter = 2;
        let r = run_cox_ph(&ds, "time", "status", &COX_X, Some(cfg)).unwrap();
        assert!(!r.converged && !r.convergence.converged);
        assert_eq!(r.iterations, 2);
        assert!(
            r.warnings
                .iter()
                .any(|w| w.starts_with("Cox regression did not converge after 2 iterations")),
            "{:?}",
            r.warnings
        );
        assert!(r.to_string().contains("Converged: no (2 iterations"), "{r}");
    }

    /// Monotone likelihood (every death has x = 1): coxph 3.5.8 converges at iteration 20 with
    /// coef 20.794537153866514 and warns "Loglik converged before variable 1; coefficient may
    /// be infinite". Hand-built fixture, not a reference dataset.
    #[test]
    fn test_cox_monotone_likelihood_warns() {
        let df = df! {
            "time" => [2.3, 1.1, 4.7, 3.2, 5.9, 0.8, 7.4, 6.1, 8.8, 9.5, 10.2, 3.9],
            "status" => [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "x" => [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "z" => [0.3, -1.2, 0.8, 0.1, -0.5, 1.4, -0.9, 0.6, 2.1, -0.3, 0.4, -1.7],
        }
        .unwrap();
        let mut cfg = CoxConfig::default();
        cfg.ties = TiesMethod::Efron;
        cfg.max_iter = 100;
        let r = run_cox_ph(&Dataset::new(df), "time", "status", &["x", "z"], Some(cfg)).unwrap();
        assert!(r.converged, "{:?}", r.convergence);
        assert!(r.coefficients[0] > 15.0, "{:?}", r.coefficients);
        assert!(
            r.warnings
                .iter()
                .any(|w| w == "Loglik converged before variable 1 (x); coefficient may be infinite"),
            "{:?}",
            r.warnings
        );
    }

    /// ECON-61 (survival part): age2 = 2 age + sex is aliased; coxph reports NA for it and fits
    /// age and sex. Before: a pseudo-inverse gave three finite coefficients without a warning.
    #[test]
    fn test_validate_cox_aliased_covariate() {
        let (case, ds) = load_case("surv_cox_robust");
        let n = case.scalar("scale_n") as usize;
        let sex = i64_values(&ds, "sex");
        let d = derive_f64(&ds, "age", "age2", |i, a| 2.0 * a as f64 + sex[i] as f64);
        let r = run_cox_ph(
            &d,
            "time",
            "status",
            &["age", "sex", "age2"],
            Some(cox_tight(TiesMethod::Efron)),
        )
        .unwrap();
        assert_eq!(r.aliased, vec!["age2".to_string()]);
        assert_eq!(r.df(), 2);
        assert!(
            r.warnings
                .iter()
                .any(|w| w == "Coefficients not defined because of singularities: age2"),
            "{:?}",
            r.warnings
        );
        Check::new(&case)
            .named("aliased_coef", &r.variables, &r.coefficients, Tol::coef(n)) // age2: NaN == NA
            .named("aliased_se", &r.variables, &r.std_errors, Tol::se(n))
            .scalar("aliased_loglik", r.log_likelihood, Tol::stat(n))
            .scalar("aliased_lr_test", r.lr_test, Tol::stat(n))
            .finish();
    }

    /// R3: iteration settings are validated in core.
    #[test]
    fn test_cox_rejects_bad_iteration_settings() {
        let ds = generated_dataset("cox_ties_small");
        for (max_iter, tol, what) in [
            (0, 1e-9, "max_iter"),
            (25, 0.0, "tolerance"),
            (25, -1.0, "tolerance"),
            (25, f64::NAN, "tolerance"),
            (25, f64::INFINITY, "tolerance"),
        ] {
            let mut cfg = CoxConfig::default();
            cfg.max_iter = max_iter;
            cfg.tolerance = tol;
            match run_cox_ph(&ds, "time", "event", &["x"], Some(cfg)) {
                Err(EconError::InvalidSpecification { message }) => {
                    assert!(message.contains(what), "{message}")
                }
                other => panic!("max_iter {max_iter}, tolerance {tol}: {other:?}"),
            }
        }
    }

    /// ECON-45: coxph(robust = TRUE). Before: robust_se was never read, so std_errors were the
    /// model SEs (efron age 0.0092674110143657 vs robust 0.00988139278005566) and nothing said so.
    fn check_cox_robust(case_id: &str, ties: TiesMethod) {
        let (case, ds) = load_case(case_id);
        let n = case.scalar("n") as usize;
        let tag = ties_tag(ties);
        let mut cfg = cox_tight(ties);
        cfg.robust_se = true;
        let r = run_cox_ph(&ds, "time", "status", &COX_X, Some(cfg)).unwrap();
        assert_eq!(r.se_type, CoxSeType::Robust);
        assert_eq!((r.cluster.as_deref(), r.n_clusters), (None, None));
        let text = r.to_string();
        assert!(
            text.contains("robust se")
                && text.contains("Standard errors: robust (Lin-Wei sandwich)")
                && text.contains("Robust score test"),
            "{text}"
        );
        Check::new(&case)
            .named(
                &format!("{tag}_coef"),
                &r.variables,
                &r.coefficients,
                Tol::coef(n),
            )
            .named(
                &format!("{tag}_se"),
                &r.variables,
                &r.naive_std_errors,
                Tol::se(n),
            )
            .named(
                &format!("{tag}_se_robust"),
                &r.variables,
                &r.std_errors,
                Tol::se(n),
            )
            .named(
                &format!("{tag}_z_robust"),
                &r.variables,
                &r.z_stats,
                Tol::stat(n),
            )
            .named(
                &format!("{tag}_p_robust"),
                &r.variables,
                &r.p_values,
                Tol::p_value(n),
            )
            .named(
                &format!("{tag}_hr_lower_robust"),
                &r.variables,
                &r.hr_ci_lower,
                Tol::stat(n),
            )
            .named(
                &format!("{tag}_hr_upper_robust"),
                &r.variables,
                &r.hr_ci_upper,
                Tol::stat(n),
            )
            .scalar(
                &format!("{tag}_wald_test_robust"),
                r.wald_test,
                Tol::stat(n),
            )
            .scalar(
                &format!("{tag}_wald_p_robust"),
                r.wald_p_value,
                Tol::p_value(n),
            )
            .scalar(
                &format!("{tag}_rscore_robust"),
                r.robust_score_test.unwrap_or(f64::NAN),
                Tol::stat(n),
            )
            .scalar(
                &format!("{tag}_rscore_p_robust"),
                r.robust_score_p_value.unwrap_or(f64::NAN),
                Tol::p_value(n),
            )
            .scalar(&format!("{tag}_score_test"), r.score_test, Tol::stat(n)) // model-based
            .scalar(&format!("{tag}_lr_test"), r.lr_test, Tol::stat(n))
            .finish();
    }

    #[test]
    fn test_validate_cox_efron_robust() {
        check_cox_robust("surv_cox_robust", TiesMethod::Efron)
    }

    #[test]
    fn test_validate_cox_breslow_robust() {
        check_cox_robust("surv_cox_robust", TiesMethod::Breslow)
    }

    /// coxph(... + cluster(inst)): dfbeta residuals summed within the 18 institutions; the row
    /// with a missing inst is dropped (226 rows) and reported; robust_se stays false (cluster
    /// implies it, as in coxph).
    #[test]
    fn test_validate_cox_cluster() {
        let (case, ds) = load_case("surv_cox_robust");
        let n = case.scalar("n_cluster_obs") as usize;
        let mut check = Check::new(&case);
        for ties in [TiesMethod::Efron, TiesMethod::Breslow] {
            let tag = ties_tag(ties);
            let mut cfg = cox_tight(ties);
            cfg.cluster = Some("inst".to_string());
            let r = run_cox_ph(&ds, "time", "status", &COX_X, Some(cfg)).unwrap();
            assert_eq!(r.se_type, CoxSeType::Cluster);
            assert_eq!(r.cluster.as_deref(), Some("inst"));
            assert_eq!(r.missing.n_dropped, 2, "{:?}", r.missing); // ph.ecog 1, inst 1
            assert!(
                r.to_string().contains("clustered by inst (18 clusters"),
                "{r}"
            );
            check
                .scalar("n_cluster_obs", r.n_obs as f64, Tol::exact())
                .scalar("n_cluster_event", r.n_events as f64, Tol::exact())
                .scalar(
                    "n_clusters",
                    r.n_clusters.map_or(f64::NAN, |g| g as f64),
                    Tol::exact(),
                )
                .named(
                    &format!("{tag}_coef_cluster"),
                    &r.variables,
                    &r.coefficients,
                    Tol::coef(n),
                )
                .named(
                    &format!("{tag}_se_model_cluster"),
                    &r.variables,
                    &r.naive_std_errors,
                    Tol::se(n),
                )
                .named(
                    &format!("{tag}_se_cluster"),
                    &r.variables,
                    &r.std_errors,
                    Tol::se(n),
                )
                .named(
                    &format!("{tag}_z_cluster"),
                    &r.variables,
                    &r.z_stats,
                    Tol::stat(n),
                )
                .named(
                    &format!("{tag}_p_cluster"),
                    &r.variables,
                    &r.p_values,
                    Tol::p_value(n),
                )
                .scalar(
                    &format!("{tag}_wald_test_cluster"),
                    r.wald_test,
                    Tol::stat(n),
                )
                .scalar(
                    &format!("{tag}_wald_p_cluster"),
                    r.wald_p_value,
                    Tol::p_value(n),
                )
                .scalar(
                    &format!("{tag}_rscore_cluster"),
                    r.robust_score_test.unwrap_or(f64::NAN),
                    Tol::stat(n),
                )
                .scalar(
                    &format!("{tag}_rscore_p_cluster"),
                    r.robust_score_p_value.unwrap_or(f64::NAN),
                    Tol::p_value(n),
                );
        }
        check.finish();
    }

    /// A clustered variance needs at least two clusters (coxph returns a variance of about 0
    /// with one).
    #[test]
    fn test_cox_cluster_needs_two_levels() {
        let ds = generated_dataset("lung");
        let d = derive_i64(&ds, "sex", "one", |_| 1);
        let mut cfg = cox_tight(TiesMethod::Efron);
        cfg.cluster = Some("one".into());
        match run_cox_ph(&d, "time", "status", &COX_X, Some(cfg)) {
            Err(EconError::InvalidSpecification { message }) => assert!(
                message.contains("'one'") && message.contains("at least 2 clusters"),
                "{message}"
            ),
            other => panic!("{other:?}"),
        }
    }

    /// With no more clusters than coefficients the clustered matrices are rank-deficient: SEs
    /// are reported, the robust Wald and score tests are NaN with a warning (coxph prints
    /// generalized-inverse statistics).
    #[test]
    fn test_cox_robust_tests_need_more_clusters_than_coefficients() {
        let ds = generated_dataset("lung");
        let d = derive_f64(&ds, "age", "half", |i, _| if i < 114 { 1.0 } else { 2.0 });
        let mut cfg = cox_tight(TiesMethod::Efron);
        cfg.cluster = Some("half".into());
        let r = run_cox_ph(&d, "time", "status", &COX_X, Some(cfg)).unwrap();
        assert_eq!(r.n_clusters, Some(2));
        assert!(r.std_errors.iter().all(|s| s.is_finite()));
        assert!(r.wald_test.is_nan() && r.robust_score_test.is_some_and(f64::is_nan));
        assert!(
            r.warnings
                .iter()
                .any(|w| w.starts_with("robust Wald test not reported")),
            "{:?}",
            r.warnings
        );
        assert!(
            r.warnings
                .iter()
                .any(|w| w.starts_with("robust score test not reported")),
            "{:?}",
            r.warnings
        );
        assert!(r.to_string().contains("not reported"), "{r}");
    }

    // =========================================================================
    // Accelerated failure time: R-pinned (case surv_aft)
    // =========================================================================

    const AFT_X: [&str; 3] = ["age", "sex", "ph.ecog"];

    /// survreg.control(rel.tolerance = 1e-12, maxiter = 100), as in the surv_aft case.
    fn aft_tight(dist: AftDistribution) -> AftConfig {
        let mut c = AftConfig::default();
        c.distribution = dist;
        c.tolerance = 1e-12;
        c.max_iter = 100;
        c
    }

    /// Everything summary.survreg reports except the Log(scale) row (check_aft_log_scale).
    fn check_aft(case_id: &str, dist: AftDistribution, tag: &str) -> AftResult {
        let (case, ds) = load_case(case_id);
        let n = case.scalar("n") as usize;
        let r = run_aft(&ds, "time", "status", &AFT_X, Some(aft_tight(dist))).unwrap();
        assert!(
            r.converged && r.warnings.is_empty(),
            "{dist}: {:?} {:?}",
            r.convergence,
            r.warnings
        );
        Check::new(&case)
            .scalar("n", r.n_obs as f64, Tol::exact())
            .scalar("n_event", r.n_events as f64, Tol::exact())
            .named(
                &format!("{tag}_coef"),
                &r.variables,
                &r.coefficients,
                Tol::coef(n),
            )
            .named(
                &format!("{tag}_se"),
                &r.variables,
                &r.std_errors,
                Tol::se(n),
            )
            .named(&format!("{tag}_z"), &r.variables, &r.z_stats, Tol::stat(n))
            .named(
                &format!("{tag}_p"),
                &r.variables,
                &r.p_values,
                Tol::p_value(n),
            )
            .scalar(&format!("{tag}_scale"), r.scale, Tol::stat(n))
            .scalar(&format!("{tag}_loglik"), r.log_likelihood, Tol::stat(n))
            .scalar(&format!("{tag}_aic"), r.aic, Tol::stat(n))
            .scalar(&format!("{tag}_bic"), r.bic, Tol::stat(n))
            .finish();
        r
    }

    fn check_aft_log_scale(case_id: &str, r: &AftResult, tag: &str) {
        let (case, _) = load_case(case_id);
        let n = case.scalar("n") as usize;
        let (z, p) = r.log_scale_test().unwrap();
        assert!(!r.scale_fixed);
        Check::new(&case)
            .scalar(
                &format!("{tag}_se_log_scale"),
                r.log_scale_se.unwrap(),
                Tol::se(n),
            )
            .scalar(&format!("{tag}_z_log_scale"), z, Tol::stat(n))
            .scalar(&format!("{tag}_p_log_scale"), p, Tol::p_value(n))
            .finish();
    }

    /// ECON-47. Before: no Newton step was ever taken (Cholesky of the negative-definite
    /// Hessian), so 100 steps of beta += 0.01 grad diverged: age 54.48, scale inf, logLik -inf.
    #[test]
    fn test_validate_aft_weibull() {
        let r = check_aft("surv_aft", AftDistribution::Weibull, "weibull");
        check_aft_log_scale("surv_aft", &r, "weibull");
        assert_eq!(r.shape, Some(1.0 / r.scale));
    }

    /// ECON-48: exponential = extreme-value errors with the scale fixed at 1; AIC counts 4
    /// parameters (R 2295.1263026263628 = 2 x 1143.5631513131814 + 8). Before: the Weibull fit
    /// relabelled.
    #[test]
    fn test_validate_aft_exponential() {
        let r = check_aft("surv_aft", AftDistribution::Exponential, "exponential");
        assert_eq!(r.scale, 1.0);
        assert!(r.scale_fixed && r.log_scale_se.is_none() && r.log_scale_test().is_none());
        assert_eq!(r.shape, Some(1.0));
        assert!(r.to_string().contains("Scale fixed at 1"), "{r}");
    }

    /// ECON-47 also: the normal censored second derivative -m (m - z) (was -m (m + z)), the
    /// normal hazard in the tails, ln S(z) without underflow.
    #[test]
    fn test_validate_aft_lognormal() {
        let r = check_aft("surv_aft", AftDistribution::LogNormal, "lognormal");
        check_aft_log_scale("surv_aft", &r, "lognormal");
        assert_eq!(r.shape, None);
    }

    #[test]
    fn test_validate_aft_loglogistic() {
        let r = check_aft("surv_aft", AftDistribution::LogLogistic, "loglogistic");
        check_aft_log_scale("surv_aft", &r, "loglogistic");
    }

    /// Phantom-option pair: survreg distinguishes exponential from weibull.
    #[test]
    fn test_aft_exponential_differs_from_weibull() {
        let ds = generated_dataset("lung");
        let w = run_aft(
            &ds,
            "time",
            "status",
            &AFT_X,
            Some(aft_tight(AftDistribution::Weibull)),
        )
        .unwrap();
        let e = run_aft(
            &ds,
            "time",
            "status",
            &AFT_X,
            Some(aft_tight(AftDistribution::Exponential)),
        )
        .unwrap();
        assert!(
            (w.coefficients[1] / e.coefficients[1] - 1.0).abs() > 0.1,
            "{:?} {:?}",
            w.coefficients,
            e.coefficients
        );
        assert!((w.scale - e.scale).abs() > 0.1 && (w.aic - e.aic).abs() > 1.0);
    }

    /// Newton on the equilibrated information does not depend on a covariate's units: age x
    /// 1e-6 and x 1e6 give survreg's rescaled coefficients in (about) the same number of
    /// iterations.
    #[test]
    fn test_validate_aft_converges_at_any_covariate_scale() {
        let (case, ds) = load_case("surv_aft");
        let n = case.scalar("n") as usize;
        let base = run_aft(
            &ds,
            "time",
            "status",
            &AFT_X,
            Some(aft_tight(AftDistribution::Weibull)),
        )
        .unwrap();
        let mut check = Check::new(&case);
        for (tag, s) in [("em6", 1e-6), ("e6", 1e6)] {
            let d = derive_f64(&ds, "age", "age_s", |_, a| a as f64 * s);
            let r = run_aft(
                &d,
                "time",
                "status",
                &["age_s", "sex", "ph.ecog"],
                Some(aft_tight(AftDistribution::Weibull)),
            )
            .unwrap();
            assert!(r.converged, "age x {s}: {:?}", r.convergence);
            assert!(
                r.iterations.abs_diff(base.iterations) <= 1,
                "age x {s}: {} vs {}",
                r.iterations,
                base.iterations
            );
            check.named(
                &format!("weibull_scale_{tag}_coef"),
                &["(Intercept)", "age", "sex", "ph.ecog"],
                &r.coefficients,
                Tol::coef(n),
            );
        }
        check.finish();
    }

    /// ECON-61 (survival part): survreg reports NA for age2 = 2 age + sex. Before: an
    /// identity-matrix SE fallback and a pseudo-inverse start.
    #[test]
    fn test_validate_aft_aliased_covariate() {
        let (case, ds) = load_case("surv_aft");
        let n = case.scalar("aliased_n") as usize;
        let sex = i64_values(&ds, "sex");
        let d = derive_f64(&ds, "age", "age2", |i, a| 2.0 * a as f64 + sex[i] as f64);
        let r = run_aft(
            &d,
            "time",
            "status",
            &["age", "sex", "age2"],
            Some(aft_tight(AftDistribution::Weibull)),
        )
        .unwrap();
        assert_eq!(r.aliased, vec!["age2".to_string()]);
        assert!(r.std_errors[3].is_nan());
        assert!(
            r.warnings
                .iter()
                .any(|w| w == "Coefficients not defined because of singularities: age2"),
            "{:?}",
            r.warnings
        );
        Check::new(&case)
            .named("aliased_coef", &r.variables, &r.coefficients, Tol::coef(n))
            .scalar("aliased_scale", r.scale, Tol::stat(n))
            .scalar("aliased_loglik", r.log_likelihood, Tol::stat(n))
            .finish();
    }

    /// R10/R2: the iteration limit is a warning with finite results (survreg: "Ran out of
    /// iterations").
    #[test]
    fn test_aft_iteration_limit_is_a_warning() {
        let ds = generated_dataset("lung");
        let mut cfg = aft_tight(AftDistribution::Weibull);
        cfg.max_iter = 1;
        let r = run_aft(&ds, "time", "status", &AFT_X, Some(cfg)).unwrap();
        assert!(!r.converged && r.coefficients.iter().all(|b| b.is_finite()));
        assert!(
            r.warnings
                .iter()
                .any(|w| w.starts_with("AFT model did not converge after 1 iteration")),
            "{:?}",
            r.warnings
        );
        assert!(r.to_string().contains("Converged: no"), "{r}");
    }

    #[test]
    fn test_aft_rejects_bad_iteration_settings() {
        let ds = generated_dataset("lung");
        for (max_iter, tol, what) in [
            (0, 1e-9, "max_iter"),
            (100, 0.0, "tolerance"),
            (100, f64::NAN, "tolerance"),
        ] {
            let mut cfg = AftConfig::default();
            cfg.max_iter = max_iter;
            cfg.tolerance = tol;
            match run_aft(&ds, "time", "status", &AFT_X, Some(cfg)) {
                Err(EconError::InvalidSpecification { message }) => {
                    assert!(message.contains(what), "{message}")
                }
                other => panic!("max_iter {max_iter}, tolerance {tol}: {other:?}"),
            }
        }
    }

    /// The tail helpers against R 4.3.3: pnorm(z, lower.tail = FALSE, log.p = TRUE),
    /// exp(dnorm(z, log = TRUE) - that), and pmax(z, 0) + log1p(exp(-abs(z))).
    #[test]
    fn test_aft_tail_helpers_match_r() {
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b.abs() + 1e-300;
        for (z, ls, h) in [
            (-40.0, -0.0, 0.0),
            (-5.0, -2.8665161296376358e-07, 1.4867199409049061e-06),
            (-1.0, -0.17275377902344988, 0.28759997093917838),
            (0.0, -0.69314718055994529, 0.79788456080286529),
            (1.5, -2.7059444008238898, 1.9386771666225433),
            (8.0, -35.013437159914552, 8.121368112236123),
            (29.9, -451.32291245852861, 29.933370411296476),
            (30.0, -454.32124395634321, 30.033259667434663),
            (40.0, -804.6084420137538, 40.024968847206338),
            // both sides of each branch boundary of the rational approximations
            (-37.5, -4.6053530095819552e-308, 1.7282337322840418e-306),
            (-8.3, -5.2055697448902552e-17, 4.3816394355093207e-16),
            (-5.66, -7.5686497806399427e-09, 4.410297812317309e-08),
            (-5.65, -8.0223918828428826e-09, 4.6668868350338607e-08),
            (-2.2, -0.01400100575938458, 0.035974766118774183),
            (-0.6745, -0.28767772961281002, 0.42369732762989082),
            (-0.674, -0.28788963644545978, 0.42393001482066067),
            (-0.3, -0.48141016158848116, 0.61722085361273438),
            (-1e-17, -0.69314718055994529, 0.79788456080286529),
            (1e-17, -0.69314718055994529, 0.79788456080286529),
            (0.2, -0.86573952268159537, 0.92941584808564293),
            (0.674, -1.3856719275105964, 1.2707348997179717),
            (0.6745, -1.386307389749966, 1.2711140638164968),
            (2.2, -4.2756184470441436, 2.5514961531512093),
            (5.65, -18.641029223787847, 5.8173259103620936),
            (5.66, -18.699251153702779, 5.8270602068397075),
            (12.25, -78.462270827375932, 12.330579239953863),
            (37.6, -711.42664867077633, 37.626558252922429),
            (100.0, -5005.5242086942053, 100.00999800101664),
        ] {
            assert!(
                close(ln_normal_sf(z), ls),
                "ln S({z}) = {} vs R {ls}",
                ln_normal_sf(z)
            );
            assert!(
                close(normal_hazard(z), h),
                "hazard({z}) = {} vs R {h}",
                normal_hazard(z)
            );
        }
        // Far tail: R's ln S; its hazard loses digits to cancellation there, so the hazard is
        // checked against Mills' ratio z + 1/z - 2/z^3.
        let z = 1e5;
        assert!(close(ln_normal_sf(z), -5000000012.4318638));
        assert!(close(normal_hazard(z), z + 1.0 / z - 2.0 / (z * z * z)));
        assert!(ln_normal_sf(f64::NAN).is_nan() && normal_hazard(f64::NAN).is_nan());
        for (z, sp) in [
            (-800.0, 0.0),
            (-30.0, 9.3576229688397368e-14),
            (0.0, 0.69314718055994529),
            (2.0, 2.1269280110429727),
            (30.0, 30.000000000000092),
            (800.0, 800.0),
        ] {
            assert!(
                close(softplus(z), sp),
                "softplus({z}) = {} vs R {sp}",
                softplus(z)
            );
        }
    }

    /// R2: non-finite results are an error, not a returned fit.
    #[test]
    fn test_aft_non_finite_result_is_an_error() {
        assert!(aft_check_finite(-1.0, &[0.1, 0.2], &[0.01, 0.02], 5, 1e-13).is_ok());
        for (ll, theta, var) in [
            (f64::NEG_INFINITY, [0.1, 0.2], [0.01, 0.02]),
            (-1.0, [f64::NAN, 0.2], [0.01, 0.02]),
            (-1.0, [0.1, 0.2], [0.01, f64::INFINITY]),
            (-1.0, [0.1, 0.2], [-1e-3, 0.02]),
        ] {
            assert!(matches!(
                aft_check_finite(ll, &theta, &var, 7, 1e-3),
                Err(EconError::ConvergenceFailure { iterations: 7, .. })
            ));
        }
    }

    /// coxph's default ties are Efron (and MCP cox_ph already defaults to Efron).
    #[test]
    fn test_validate_cox_default_ties_is_efron() {
        assert_eq!(CoxConfig::default().ties, TiesMethod::Efron);
        assert_eq!(TiesMethod::default(), TiesMethod::Efron);
        let (case, ds) = load_case("surv_cox_robust");
        let r = run_cox_ph(&ds, "time", "status", &COX_X, None).unwrap();
        assert_eq!(r.ties_method, TiesMethod::Efron);
        // default eps 1e-9 stops at iteration 3, 3.3e-9 relative from the tight fit (measured in R)
        Check::new(&case)
            .named(
                "efron_coef",
                &r.variables,
                &r.coefficients,
                Tol::iterative(1e-7),
            )
            .finish();
    }

    // =========================================================================
    // Competing risks: R-pinned (case surv_competing_risks)
    // =========================================================================

    /// ECON-78: CIF SEs and log-scale limits as survfit (multi-state Aalen-Johansen), continuous
    /// and tied times, levels 0.95 and 0.5. Before: SEs from sum S(t-)^2 d_k / n^2, larger than
    /// survfit's.
    #[test]
    fn test_validate_competing_risks_se_survfit() {
        let (case, cont) = load_case("surv_competing_risks"); // analysis dataset: surv_cr_cont
        let ties = generated_dataset("surv_cr_ties");
        let mut check = Check::new(&case);
        for (tag, ds) in [("cont", &cont), ("ties", &ties)] {
            let n = case.scalar(&format!("{tag}_n")) as usize;
            for (lev, pct) in [(0.95, 95), (0.5, 50)] {
                let r = run_competing_risks(ds, "time", "event_type", lev).unwrap();
                assert_eq!(r.event_types, vec![1, 2]);
                for (k, cif) in r.cifs.iter().enumerate() {
                    let k1 = k + 1;
                    if pct == 95 {
                        check
                            .vector(&format!("{tag}_times"), &cif.times, Tol::exact())
                            .vector(&format!("{tag}_cif_{k1}"), &cif.incidence, Tol::coef(n))
                            .vector(&format!("{tag}_se_{k1}"), &cif.std_errors, Tol::se(n));
                    }
                    check
                        .vector(
                            &format!("{tag}_ci{pct}_lower_{k1}"),
                            &cif.ci_lower,
                            Tol::stat(n),
                        )
                        .vector(
                            &format!("{tag}_ci{pct}_upper_{k1}"),
                            &cif.ci_upper,
                            Tol::stat(n),
                        );
                }
            }
        }
        check.finish();
    }

    /// The O(m) running-sum variance equals the direct double sum over the event times (tied
    /// times; no R).
    #[test]
    fn test_competing_risks_variance_running_sums_match_direct_sum() {
        let ds = generated_dataset("surv_cr_ties");
        let time: Vec<f64> = i64_values(&ds, "time").iter().map(|&t| t as f64).collect();
        let ev = i64_values(&ds, "event_type");
        let r = run_competing_risks(&ds, "time", "event_type", 0.95).unwrap();
        let mut uniq = time.clone();
        uniq.sort_by(f64::total_cmp);
        uniq.dedup();
        for (k, cif) in r.cifs.iter().enumerate() {
            let cause = k as i64 + 1;
            let (mut s, mut f) = (1.0, 0.0);
            // (a_j, b_j, c_j, F_j) per time
            let mut terms: Vec<(f64, f64, f64, f64)> = Vec::new();
            for (j, &t) in uniq.iter().enumerate() {
                let n = time.iter().filter(|&&x| x >= t).count() as f64;
                let at = |pred: &dyn Fn(i64) -> bool| {
                    time.iter()
                        .zip(&ev)
                        .filter(|&(&x, &e)| x == t && pred(e))
                        .count() as f64
                };
                let d = at(&|e| e > 0);
                let dk = at(&|e| e == cause);
                let s_prev = s;
                f += s_prev * dk / n;
                s *= 1.0 - d / n;
                let a = if n > d { d / (n * (n - d)) } else { 0.0 };
                terms.push((
                    a,
                    s_prev * s_prev * dk * (n - dk) / (n * n * n),
                    s_prev * dk / (n * n),
                    f,
                ));
                let var: f64 = terms
                    .iter()
                    .map(|&(a, b, c, fj)| a * (f - fj).powi(2) + b - 2.0 * c * (f - fj))
                    .sum();
                let se = var.max(0.0).sqrt();
                assert_eq!(cif.times[j], t);
                assert!((cif.incidence[j] - f).abs() <= 1e-14, "cause {cause} t {t}");
                assert!(
                    (cif.std_errors[j] - se).abs() <= 1e-12 * se + 1e-15,
                    "cause {cause} t {t}: {} vs direct {se}",
                    cif.std_errors[j]
                );
            }
        }
    }
}
