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
//! The cohort-time effects are aggregated as `etwfe::emfx` does: the mean
//! per-observation effect over emfx's rows, with delta-method standard errors
//! from the full covariance matrix:
//! - **simple**: the overall ATT (cohorts weighted by treated observations)
//! - **event**: by time since first treatment
//! - **group**: by cohort
//! - **calendar**: by period
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

use ndarray::linalg::general_mat_vec_mul;
use ndarray::{Array1, Array2, ArrayView1};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, Columns, MissingSummary, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::solve::{SingularPolicy, least_squares, sym_inverse};
use crate::regression::ols::normal_inverse_from_gram;
use crate::resource;
use crate::traits::convergence::{ResultNotes, notes_text};
use crate::traits::estimator::{
    normal_p_value_two_sided, t_test_p_value, validate_level, z_critical,
};
use faer::Mat;

/// Configuration for ETWFE estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EtwfeConfig {
    /// Reference time period (default: first period)
    pub tref: Option<i64>,
    /// Reference cohort (default: never-treated or last cohort)
    pub gref: Option<i64>,
    /// Control group: "never" (never-treated only) or "notyet" (not-yet-treated)
    pub cgroup: ControlGroup,
    /// Anticipation periods: cohort `g` is treated from period `g - anticipation` on (its
    /// cells and onset shift that many periods earlier). Must be below the first treated
    /// cohort minus the first period.
    pub anticipation: usize,
    /// Variance of the coefficients.
    #[serde(default)]
    pub vcov: EtwfeVcov,
    /// Level of the aggregate confidence intervals, in (0, 1).
    #[serde(default = "default_confidence_level")]
    pub confidence_level: f64,
}

fn default_confidence_level() -> f64 {
    0.95
}

impl Default for EtwfeConfig {
    fn default() -> Self {
        Self {
            tref: None,
            gref: None,
            cgroup: ControlGroup::NotYet,
            anticipation: 0,
            vcov: EtwfeVcov::Cluster,
            confidence_level: 0.95,
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

crate::option_enum!(
    ControlGroup,
    crate::options::Choices {
        option: "cgroup",
        values: &[
            (
                &["notyet", "not_yet", "not_yet_treated", "notyettreated"],
                ControlGroup::NotYet
            ),
            (
                &["never", "never_treated", "nevertreated"],
                ControlGroup::Never
            ),
        ],
    },
    no_display
);

/// Variance of the ETWFE coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EtwfeVcov {
    /// Clustered by unit with fixest's CR1 small-sample factor
    /// `G/(G-1) * (n-1)/(n-K)`, `K` = coefficients + periods; t tests with `G - 1` df.
    #[default]
    Cluster,
    /// Homoskedastic: `SSR / (n - k - rank of the unit and period effects)`; t tests with
    /// that df.
    Iid,
}

crate::option_enum!(
    EtwfeVcov,
    crate::options::Choices {
        option: "vcov",
        values: &[
            (&["cluster", "clustered", "cr1"], EtwfeVcov::Cluster),
            (
                &["iid", "classical", "standard", "homoskedastic"],
                EtwfeVcov::Iid
            ),
        ],
    }
);

/// emfx aggregation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EtwfeAggregation {
    /// One overall ATT.
    Simple,
    /// By time since first treatment.
    Event,
    /// By cohort.
    Group,
    /// By period.
    Calendar,
}

crate::option_enum!(
    EtwfeAggregation,
    crate::options::Choices {
        option: "aggregation",
        values: &[
            (&["simple", "overall", "att"], EtwfeAggregation::Simple),
            (
                &["event", "dynamic", "event_study"],
                EtwfeAggregation::Event
            ),
            (&["group", "cohort"], EtwfeAggregation::Group),
            (&["calendar", "period", "time"], EtwfeAggregation::Calendar),
        ],
    }
);

/// One estimated coefficient, named as fixest names it
/// (`.Dtreat:first_treat::4:year::5`, `.Dtreat:first_treat::4:year::5:xc_dm`, `year::2:xc`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EtwfeTerm {
    /// fixest's coefficient name.
    pub name: String,
    /// Estimate.
    pub estimate: f64,
    /// Standard error (as `EtwfeResult::vcov`).
    pub std_error: f64,
    /// t statistic.
    pub t_stat: f64,
    /// Two-sided p-value of the t test with `EtwfeResult::df_t` df.
    pub p_value: f64,
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
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AggregatedEffect {
    /// Aggregation key (relative time for event study, cohort for group average)
    pub key: i64,
    /// Aggregation type
    pub agg_type: String,
    /// Estimated effect (weighted average)
    pub estimate: f64,
    /// Standard error
    pub std_error: f64,
    /// Lower confidence bound at [`EtwfeResult::confidence_level`]
    pub ci_lower: f64,
    /// Upper confidence bound at [`EtwfeResult::confidence_level`]
    pub ci_upper: f64,
    /// Number of underlying cohort-time effects
    pub n_effects: usize,
    /// z statistic.
    #[serde(default)]
    pub z_stat: f64,
    /// Two-sided normal p-value.
    #[serde(default)]
    pub p_value: f64,
    /// Rows averaged.
    #[serde(default)]
    pub n_obs: usize,
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
    /// Overall ATT: emfx's simple ATT, the average effect over the treated cohort-period
    /// rows (weighted by treated observations)
    pub att_simple: f64,
    /// Delta-method standard error of `att_simple`
    pub att_se: f64,
    /// Same as `att_simple` (kept for compatibility)
    pub att_weighted: f64,
    /// Same as `att_se` (kept for compatibility)
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
    /// Rows dropped because of missing values (empty when none were).
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
    /// Every estimated coefficient (cells, then control terms), named as fixest names it.
    #[serde(default)]
    pub terms: Vec<EtwfeTerm>,
    /// Terms the unit and period effects absorb (zero after the within transformation).
    #[serde(default)]
    pub dropped_terms: Vec<String>,
    /// Control terms the pivoted QR aliased; removed and the model refitted, as fixest does.
    #[serde(default)]
    pub collinear_terms: Vec<String>,
    /// Variance of the coefficients.
    #[serde(default)]
    pub vcov: EtwfeVcov,
    /// Units in the estimation sample (the clusters under `EtwfeVcov::Cluster`).
    #[serde(default)]
    pub n_units: usize,
    /// Never-treated units in the estimation sample.
    #[serde(default)]
    pub n_never_units: usize,
    /// Degrees of freedom of the coefficient t tests.
    #[serde(default)]
    pub df_t: usize,
    /// Anticipation periods: cohort g is treated from g - anticipation.
    #[serde(default)]
    pub anticipation: usize,
    /// Warnings about the fit.
    #[serde(default)]
    pub warnings: Vec<String>,
    /// emfx simple: the overall ATT.
    #[serde(default)]
    pub overall: AggregatedEffect,
    /// emfx calendar: effects by period.
    #[serde(default)]
    pub calendar: Vec<AggregatedEffect>,
    /// Level of the aggregate confidence intervals.
    #[serde(default = "default_confidence_level")]
    pub confidence_level: f64,
    /// First period dropped because the reference cohort is treated from it on
    /// (`cgroup = notyet` with a reference cohort).
    #[serde(default)]
    pub dropped_from: Option<i64>,
}

/// A confidence level as a percentage without trailing zeros ("95", "90", "97.5").
fn level_percent(level: f64) -> String {
    let s = format!("{:.4}", level * 100.0);
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

impl EtwfeResult {
    /// The emfx table of one type (`Simple` is the single overall row).
    pub fn aggregation(&self, which: EtwfeAggregation) -> &[AggregatedEffect] {
        match which {
            EtwfeAggregation::Simple => std::slice::from_ref(&self.overall),
            EtwfeAggregation::Event => &self.event_study,
            EtwfeAggregation::Group => &self.cohort_avg,
            EtwfeAggregation::Calendar => &self.calendar,
        }
    }

    /// That table as Display prints it.
    pub fn aggregation_text(&self, which: EtwfeAggregation) -> String {
        let level = level_percent(self.confidence_level);
        if which == EtwfeAggregation::Simple {
            let o = &self.overall;
            return format!(
                "ATT (emfx simple; mean over {} treated observations): {:.4} (SE {:.4}), z = {:.2}, \
                 p = {:.4}, {level}% CI [{:.4}, {:.4}]\n",
                o.n_obs, o.estimate, o.std_error, o.z_stat, o.p_value, o.ci_lower, o.ci_upper
            );
        }
        let (title, key) = match which {
            EtwfeAggregation::Event => (
                "Event Study (emfx event; by time since first_treat)",
                "Rel.Time",
            ),
            EtwfeAggregation::Group => ("Cohort Averages (emfx group)", "Cohort"),
            _ => ("Period Averages (emfx calendar)", "Period"),
        };
        let mut out = String::new();
        out.push_str(&format!("{:-<70}\n{title}\n{:-<70}\n", "", ""));
        out.push_str(&format!(
            "{key:>8} {:>10} {:>9} {:>7} {:>8} {:>9} {:>9} {:>6}\n",
            "Estimate", "SE", "z", "P>|z|", "CI lo", "CI hi", "N"
        ));
        for e in self.aggregation(which) {
            if e.n_effects == 0 {
                out.push_str(&format!(
                    "{:>8} {:>10.4} (reference) {:>40}\n",
                    e.key, e.estimate, e.n_obs
                ));
            } else {
                out.push_str(&format!(
                    "{:>8} {:>10.4} {:>9.4} {:>7.2} {:>8.4} {:>9.4} {:>9.4} {:>6}\n",
                    e.key,
                    e.estimate,
                    e.std_error,
                    e.z_stat,
                    e.p_value,
                    e.ci_lower,
                    e.ci_upper,
                    e.n_obs
                ));
            }
        }
        out
    }
}

impl ResultNotes for EtwfeResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

/// Cells printed one per line up to this many; beyond it the table is summarised.
const MAX_CELL_ROWS: usize = 60;

impl fmt::Display for EtwfeResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{}", "=".repeat(70))?;
        writeln!(f, "{:^70}", "Extended Two-Way Fixed Effects (ETWFE)")?;
        writeln!(f, "{}", "=".repeat(70))?;

        writeln!(f, "\nModel: {}", self.model)?;
        writeln!(f, "Outcome: {}", self.dep_var)?;
        let mut reference = Vec::new();
        if self.n_never_units > 0 {
            reference.push(format!("never treated ({} units)", self.n_never_units));
        }
        if let Some(c) = self.ref_cohort {
            let mut part = format!("cohort {c}");
            if let Some(s) = self.dropped_from {
                part.push_str(&format!(" (periods from {s} dropped)"));
            }
            reference.push(part);
        }
        let cgroup = match self.control_group {
            ControlGroup::NotYet => "notyet",
            ControlGroup::Never => "never",
        };
        writeln!(
            f,
            "Control group: {} (cgroup = {cgroup}); reference group: {}",
            self.control_group,
            reference.join(" and ")
        )?;
        writeln!(f, "Reference period (tref): {}", self.ref_time)?;
        if !self.controls.is_empty() {
            writeln!(f, "Controls: {}", self.controls.join(", "))?;
        }
        if !self.dropped_terms.is_empty() {
            writeln!(
                f,
                "Absorbed by the unit and period effects: {}",
                self.dropped_terms.join(", ")
            )?;
        }
        if !self.collinear_terms.is_empty() {
            writeln!(
                f,
                "Removed as collinear: {}",
                self.collinear_terms.join(", ")
            )?;
        }
        if self.anticipation > 0 {
            writeln!(
                f,
                "Anticipation: {a} period(s); cohort g is treated from g - {a}",
                a = self.anticipation
            )?;
        }

        writeln!(f, "\n{:-<70}", "")?;
        writeln!(
            f,
            "Sample: {} obs ({} treated, {} not treated), {} units, {} periods",
            self.n_obs, self.n_treated, self.n_control, self.n_units, self.n_periods
        )?;
        match self.vcov {
            EtwfeVcov::Cluster => writeln!(
                f,
                "Standard errors: clustered by unit ({} clusters; CR1 as fixest), t with {} df",
                self.n_units, self.df_t
            )?,
            EtwfeVcov::Iid => writeln!(f, "Standard errors: iid, t with {} df", self.df_t)?,
        }
        writeln!(f, "Within R-squared: {:.4}", self.r_squared)?;

        writeln!(f, "\n{:-<70}", "")?;
        writeln!(f, "Cohort x Period Effects (etwfe coefficients)")?;
        writeln!(f, "{:-<70}", "")?;
        if self.cohort_time_effects.len() <= MAX_CELL_ROWS {
            writeln!(
                f,
                "{:>7} {:>7} {:>9} {:>10} {:>9} {:>7} {:>8} {:>6}",
                "Cohort", "Period", "Rel.Time", "Estimate", "SE", "t", "P>|t|", "N"
            )?;
            for c in &self.cohort_time_effects {
                writeln!(
                    f,
                    "{:>7} {:>7} {:>9} {:>10.4} {:>9.4} {:>7.2} {:>8.4} {:>6}",
                    c.cohort,
                    c.time,
                    c.rel_time,
                    c.estimate,
                    c.std_error,
                    c.t_stat,
                    c.p_value,
                    c.n_treated
                )?;
            }
        } else {
            writeln!(
                f,
                "{} cohort-period effects (see cohort_time_effects in the JSON output)",
                self.cohort_time_effects.len()
            )?;
        }

        writeln!(f, "\n{:-<70}", "")?;
        write!(f, "{}", self.aggregation_text(EtwfeAggregation::Simple))?;
        for which in [
            EtwfeAggregation::Event,
            EtwfeAggregation::Group,
            EtwfeAggregation::Calendar,
        ] {
            if !self.aggregation(which).is_empty() {
                writeln!(f)?;
                write!(f, "{}", self.aggregation_text(which))?;
            }
        }

        let notes = notes_text(self);
        if !notes.is_empty() {
            writeln!(f)?;
            write!(f, "{notes}")?;
        }
        if let Some(note) = self.missing.note() {
            writeln!(f)?;
            writeln!(f, "Missing values: {note}")?;
        }
        Ok(())
    }
}

/// Run Extended Two-Way Fixed Effects estimation, as `etwfe::etwfe(ivar = unit)`.
///
/// The model is `y_it = a_i + l_t + sum_(g,s) b_gs 1(G_i = g) 1(t = s) + e_it` over the
/// cohort-period cells etwfe estimates: with `cgroup = notyet` one cell per treated
/// cohort-period (`t >= onset`, none at `tref`), with `cgroup = never` every period of a
/// treated cohort except its reference period `onset - 1`. The onset of cohort g is
/// `g - anticipation`. Without never-treated units the last cohort is the reference group
/// and the periods from its onset on are dropped (`cgroup = notyet`, as etwfe). Unit and
/// period effects are absorbed exactly (balanced or not); a cell they leave unidentified is
/// an error.
///
/// # Arguments
///
/// * `dataset` - Panel dataset
/// * `y_col` - Outcome variable column (numeric)
/// * `unit_col` - Unit/entity identifier column (any type: integers, fractional
///   numbers, strings, categoricals, dates)
/// * `time_col` - Time period column (whole numbers or dates)
/// * `treat_col` - Treatment indicator column: 1 exactly when `0 < first_treat <= period`,
///   else 0 (checked)
/// * `first_treat_col` - First treatment period column (cohort identifier; whole
///   numbers or dates; 0 or negative, or a period after the last observed one, for
///   never-treated units)
/// * `x_cols` - Optional control variables (numeric), entered as etwfe's formula
///   controls: each demeaned within cohort and interacted with every cell, plus the control
///   itself and its cohort and period interactions. Terms the fixed effects absorb (a
///   time-invariant control's main effect and cohort interactions; the period interactions
///   of a control that varies only over time) are reported in `dropped_terms`; collinear
///   control terms are removed (`collinear_terms`, with a warning) and the model refitted
/// * `config` - ETWFE configuration
///
/// # Missing values
///
/// A row with a missing value (null or NaN) in the outcome, unit, time, treatment,
/// first treatment period or a control is dropped from every column, as fixest and
/// etwfe drop incomplete rows; `EtwfeResult::missing` and the last Display line
/// report what was dropped. Nothing is imputed: a missing `first_treat` does not make
/// a unit never treated. A fractional or non-finite `first_treat` or time value is
/// an error (an infinite `first_treat` one that says never-treated units are coded
/// 0), and a text outcome is `NonNumericColumn`.
///
/// # Returns
///
/// `EtwfeResult` with cohort-time effects, event study, and aggregated ATTs.
#[allow(clippy::too_many_arguments)]
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
    let control_cols: &[&str] = x_cols.unwrap_or(&[]);

    // An infinite first_treat is refused with the never-treated code in the message.
    crate::econometrics::staggered_did::refuse_infinite_first_treat(dataset, first_treat_col, "0")?;
    validate_level(config.confidence_level, "confidence_level")?;
    let z_crit = z_critical(config.confidence_level)?;
    // One request over every column the model uses: rows with a missing value in
    // any of them are dropped jointly (reported in `missing`), never imputed.
    let cols = ColumnRequest::new(dataset.df())
        .numeric(y_col)
        .factor(unit_col)
        .time_index(time_col)
        .numeric(treat_col)
        .time_index(first_treat_col)
        .numerics(control_cols)
        .na(NaPolicy::Omit)
        .extract()?;
    let names = SpecColumns {
        y: y_col,
        unit: unit_col,
        time: time_col,
        treat: treat_col,
        first_treat: first_treat_col,
        controls: control_cols,
    };
    let spec = Spec::build(&cols, &names, &config)?;
    let design = Design::build(&spec, &names)?;
    let within = TwoWayWithin::new(&spec.unit, &spec.period, spec.n_units, spec.periods.len())?;
    let fit = fit_design(&spec, &design, &within, config.vcov)?;

    let mut warnings = spec.warnings.clone();
    warnings.extend(within.warning.clone());
    warnings.extend(fit.warning(&design));

    let terms: Vec<EtwfeTerm> = fit
        .kept
        .iter()
        .enumerate()
        .map(|(j, &c)| fit.term(j, design.names[c].clone()))
        .collect();
    let cohort_time_effects: Vec<CohortTimeEffect> = spec
        .cells
        .iter()
        .enumerate()
        .map(|(c, &(g, t))| {
            // every cell is a kept column: an absorbed or aliased cell is an error
            let term = &terms[fit.term_of_col[c].unwrap_or_default()];
            CohortTimeEffect {
                cohort: g,
                time: t,
                rel_time: t - g,
                estimate: term.estimate,
                std_error: term.std_error,
                t_stat: term.t_stat,
                p_value: term.p_value,
                n_treated: spec.cell_rows[c],
            }
        })
        .collect();

    // emfx: per-row effect gradient (the row's cell coefficient)
    let gradient = |i: usize, out: &mut Vec<(usize, f64)>| {
        let Some(c) = spec.cell_of_row[i] else {
            return;
        };
        if let Some(j) = fit.term_of_col[c] {
            out.push((j, 1.0));
        }
        for (x, xdm) in spec.xdm.iter().enumerate() {
            let col = design.n_cells + x * design.per_ctrl + c;
            if let Some(j) = fit.term_of_col[col] {
                out.push((j, xdm[i]));
            }
        }
    };
    let [simple, event_study, cohort_avg, calendar] = emfx_tables(&spec, &fit, z_crit, &gradient)?;
    let overall = simple
        .into_iter()
        .next()
        .filter(|o| o.n_effects > 0)
        .ok_or_else(|| EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "ETWFE: no post-treatment cohort-period effect to aggregate (emfx simple)"
                .to_string(),
        })?;
    let (att_simple, att_se) = (overall.estimate, overall.std_error);

    let n = spec.n;
    let n_treated: usize = spec.treat.iter().filter(|&&d| d == 1.0).count();
    let n_cohorts = spec
        .cells
        .iter()
        .map(|&(g, _)| g)
        .collect::<BTreeSet<i64>>()
        .len();
    let n_periods = spec.periods.len();
    let r_squared = if fit.tss_within > 0.0 {
        1.0 - fit.ssr / fit.tss_within
    } else {
        f64::NAN
    };

    Ok(EtwfeResult {
        model: format!(
            "ETWFE ({n_cohorts} cohorts × {n_periods} periods), unit and period fixed effects"
        ),
        dep_var: y_col.to_string(),
        controls: control_cols.iter().map(|s| s.to_string()).collect(),
        control_group: config.cgroup,
        cohort_time_effects,
        event_study,
        cohort_avg,
        att_simple,
        att_se,
        att_weighted: att_simple,
        att_weighted_se: att_se,
        n_obs: n,
        n_treated,
        n_control: n - n_treated,
        n_cohorts,
        n_periods,
        r_squared,
        ref_time: spec.tref,
        ref_cohort: spec.gref_cohort,
        missing: cols.missing().clone(),
        terms,
        dropped_terms: fit
            .absorbed
            .iter()
            .map(|&c| design.names[c].clone())
            .collect(),
        collinear_terms: fit
            .collinear
            .iter()
            .map(|&c| design.names[c].clone())
            .collect(),
        vcov: config.vcov,
        n_units: spec.n_units,
        n_never_units: spec.n_never_units,
        df_t: fit.df_t,
        anticipation: config.anticipation,
        warnings,
        overall,
        calendar,
        confidence_level: config.confidence_level,
        dropped_from: spec.dropped_from,
    })
}

/// A column counts as absorbed by the unit and period effects when its within sum of
/// squares is at most this fraction of its centred raw sum of squares (or the raw column is
/// constant). Relative, so the rule does not depend on the scale of a control.
const ABSORBED_TOL: f64 = 1e-12;

/// The column names `run_etwfe` was called with.
struct SpecColumns<'a> {
    y: &'a str,
    unit: &'a str,
    time: &'a str,
    treat: &'a str,
    first_treat: &'a str,
    controls: &'a [&'a str],
}

/// The estimation sample and etwfe's specification on it.
struct Spec {
    /// Rows of the estimation sample.
    n: usize,
    y: Vec<f64>,
    /// Dense unit code (0..n_units) of each row.
    unit: Vec<usize>,
    /// Index into `periods` of each row.
    period: Vec<usize>,
    t: Vec<i64>,
    g: Vec<i64>,
    /// First treated period, `g - anticipation` (`g` for never-treated units).
    onset: Vec<i64>,
    /// The row's unit is in the reference group (never treated, or the `gref` cohort).
    reference: Vec<bool>,
    /// etwfe's `.Dtreat`: the row belongs to a treated cohort-period of etwfe's design.
    treated_cell: Vec<bool>,
    /// Index into `cells` of the rows with a cell coefficient.
    cell_of_row: Vec<Option<usize>>,
    treat: Vec<f64>,
    /// Controls on the estimation sample, in `x_cols` order.
    x: Vec<Vec<f64>>,
    /// Controls demeaned within cohort (`first_treat` value), the means taken over every
    /// complete row before the reference cohort's periods are dropped (as etwfe demeans its
    /// data before feols drops the rows without `.Dtreat`).
    xdm: Vec<Vec<f64>>,
    /// Cohort-period cells `(first_treat, period)`, sorted.
    cells: Vec<(i64, i64)>,
    /// Rows in each cell.
    cell_rows: Vec<usize>,
    /// Periods of the estimation sample, sorted.
    periods: Vec<i64>,
    n_units: usize,
    n_never_units: usize,
    tref: i64,
    /// The cohort that is the reference group (`gref`, or the last cohort without
    /// never-treated units).
    gref_cohort: Option<i64>,
    /// First period dropped because the reference cohort is treated from it on.
    dropped_from: Option<i64>,
    cgroup: ControlGroup,
    warnings: Vec<String>,
}

impl Spec {
    fn build(cols: &Columns, names: &SpecColumns<'_>, config: &EtwfeConfig) -> EconResult<Spec> {
        let a = config.anticipation;
        let y_all = cols.numeric(names.y);
        let factor = cols.factor(names.unit);
        let times = cols.time_index(names.time);
        let treat_all = cols.numeric(names.treat);
        let ft = cols.time_index(names.first_treat);
        let n_all = cols.n();

        let all_periods: Vec<i64> = times
            .iter()
            .copied()
            .collect::<BTreeSet<i64>>()
            .into_iter()
            .collect();
        if all_periods.len() < 2 {
            return Err(EconError::InsufficientData {
                required: 2,
                provided: all_periods.len(),
                context: "ETWFE needs at least two periods".to_string(),
            });
        }
        let (lo, hi) = (all_periods[0], all_periods[all_periods.len() - 1]);
        let is_never = |g: i64| g <= 0 || g > hi;

        for i in 0..n_all {
            let v = treat_all[i];
            if v != 0.0 && v != 1.0 {
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "treat column '{}' must be 0 or 1; unit '{}', period {} has {v}",
                        names.treat,
                        factor.label(i),
                        times[i]
                    ),
                });
            }
        }
        let mut unit_g: Vec<Option<i64>> =
            resource::try_vec(factor.n_levels(), None, "ETWFE unit cohorts")?;
        for i in 0..n_all {
            let slot = &mut unit_g[factor.codes[i] as usize];
            match *slot {
                None => *slot = Some(ft[i]),
                Some(g0) if g0 != ft[i] => {
                    return Err(EconError::InvalidSpecification {
                        message: format!(
                            "{} must be constant within each unit; unit '{}' has {g0} and {}",
                            names.first_treat,
                            factor.label(i),
                            ft[i]
                        ),
                    });
                }
                Some(_) => {}
            }
        }
        for i in 0..n_all {
            let treated = !is_never(ft[i]) && times[i] >= ft[i];
            if (treat_all[i] == 1.0) != treated {
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "treat column '{}' is {} for unit '{}' in period {}, but {} is {}: \
                         ETWFE needs an absorbing treatment that starts at {} (treat = 1 \
                         exactly when 0 < {} <= period)",
                        names.treat,
                        treat_all[i],
                        factor.label(i),
                        times[i],
                        names.first_treat,
                        ft[i],
                        names.first_treat,
                        names.first_treat
                    ),
                });
            }
        }
        let tref = config.tref.unwrap_or(lo);
        if all_periods.binary_search(&tref).is_err() {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "tref {tref} is not a period of '{}' ({lo}..{hi})",
                    names.time
                ),
            });
        }
        let observed_g: BTreeSet<i64> = ft.iter().copied().collect();
        if let Some(v) = config.gref {
            if !observed_g.contains(&v) {
                let values: Vec<String> = observed_g.iter().map(|g| g.to_string()).collect();
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "gref {v} is not a value of '{}' (values: {})",
                        names.first_treat,
                        values.join(", ")
                    ),
                });
            }
        }
        let any_never = ft.iter().any(|&g| is_never(g));
        let mut gref_cohort = config.gref.filter(|&v| !is_never(v));
        if !any_never && gref_cohort.is_none() {
            match config.cgroup {
                ControlGroup::NotYet => gref_cohort = observed_g.iter().next_back().copied(),
                ControlGroup::Never => {
                    return Err(EconError::InvalidSpecification {
                        message: format!(
                            "cgroup = never needs never-treated units ({} <= 0 or after the \
                             last period {hi}); none found. Use cgroup = notyet, or name the \
                             reference cohort with gref",
                            names.first_treat
                        ),
                    });
                }
            }
        }
        let is_reference = |g: i64| is_never(g) || Some(g) == gref_cohort;
        let Some(g_first) = ft.iter().copied().filter(|&g| !is_reference(g)).min() else {
            return Err(EconError::InvalidSpecification {
                message: "no treated cohort: every unit is in the reference group".to_string(),
            });
        };
        // Every treated cohort needs a period before its onset g - anticipation.
        let bound = g_first.saturating_sub(lo);
        let d = i64::try_from(a)
            .ok()
            .filter(|&d| a == 0 || d < bound)
            .ok_or_else(|| EconError::InvalidSpecification {
                message: format!(
                    "anticipation must be below {} (first treated cohort {g_first} minus first \
                     period {lo}), got {a}",
                    bound.max(0)
                ),
            })?;
        let mut warnings = Vec::new();
        if let (ControlGroup::Never, Some(c)) = (config.cgroup, gref_cohort) {
            warnings.push(format!(
                "gref = {c} is a cohort first treated in period {c}; with cgroup = never it is \
                 the reference group in every period, as in etwfe"
            ));
        }
        // Rows from the reference cohort's onset on are dropped (etwfe's `.Dtreat = NA`).
        let dropped_from = match (config.cgroup, gref_cohort) {
            (ControlGroup::NotYet, Some(c)) => {
                let s = c.saturating_sub(d);
                (s >= lo).then_some(s)
            }
            _ => None,
        };

        let keep: Vec<usize> = (0..n_all)
            .filter(|&i| dropped_from.is_none_or(|s| times[i] < s))
            .collect();
        let n = keep.len();
        let periods: Vec<i64> = keep
            .iter()
            .map(|&i| times[i])
            .collect::<BTreeSet<i64>>()
            .into_iter()
            .collect();
        if periods.len() < 2 {
            return Err(EconError::InsufficientData {
                required: 2,
                provided: periods.len(),
                context: "ETWFE needs at least two periods before the reference cohort's onset"
                    .to_string(),
            });
        }
        let mut dense: Vec<Option<usize>> =
            resource::try_vec(factor.n_levels(), None, "ETWFE unit codes")?;
        let mut n_units = 0usize;
        let mut n_never_units = 0usize;
        let mut unit = resource::try_with_capacity(n, "ETWFE units")?;
        for &i in &keep {
            let slot = &mut dense[factor.codes[i] as usize];
            let u = match *slot {
                Some(u) => u,
                None => {
                    *slot = Some(n_units);
                    n_units += 1;
                    if is_never(ft[i]) {
                        n_never_units += 1;
                    }
                    n_units - 1
                }
            };
            unit.push(u);
        }
        if n_units < 2 {
            return Err(EconError::InsufficientData {
                required: 2,
                provided: n_units,
                context: "ETWFE units".to_string(),
            });
        }

        let pick_f = |v: &[f64]| keep.iter().map(|&i| v[i]).collect::<Vec<f64>>();
        let pick_i = |v: &[i64]| keep.iter().map(|&i| v[i]).collect::<Vec<i64>>();
        let t = pick_i(times);
        let g = pick_i(ft);
        let mut onset = Vec::with_capacity(n);
        for &gi in &g {
            onset.push(if is_never(gi) {
                gi
            } else {
                gi.saturating_sub(d)
            });
        }
        let reference: Vec<bool> = g.iter().map(|&gi| is_reference(gi)).collect();
        let treated_cell: Vec<bool> = (0..n)
            .map(|i| {
                !reference[i]
                    && match config.cgroup {
                        ControlGroup::NotYet => t[i] >= onset[i],
                        ControlGroup::Never => onset[i].checked_sub(1) != Some(t[i]),
                    }
            })
            .collect();
        let has_cell =
            |i: usize| treated_cell[i] && !(config.cgroup == ControlGroup::NotYet && t[i] == tref);
        let cells: Vec<(i64, i64)> = (0..n)
            .filter(|&i| has_cell(i))
            .map(|i| (g[i], t[i]))
            .collect::<BTreeSet<(i64, i64)>>()
            .into_iter()
            .collect();
        if cells.is_empty() {
            return Err(EconError::InvalidSpecification {
                message: "no cohort-period cell to estimate: every treated cohort-period is \
                          in the reference period"
                    .to_string(),
            });
        }
        let mut cell_rows = vec![0usize; cells.len()];
        let cell_of_row: Vec<Option<usize>> = (0..n)
            .map(|i| {
                has_cell(i).then(|| {
                    // cells holds every (g, t) of a cell row
                    let c = cells.binary_search(&(g[i], t[i])).unwrap_or_default();
                    cell_rows[c] += 1;
                    c
                })
            })
            .collect();
        let period: Vec<usize> = t
            .iter()
            .map(|ti| periods.binary_search(ti).unwrap_or_default())
            .collect();
        let x: Vec<Vec<f64>> = names
            .controls
            .iter()
            .map(|c| pick_f(cols.numeric(c)))
            .collect();
        let mut xdm = Vec::with_capacity(names.controls.len());
        for c in names.controls {
            let v = cols.numeric(c);
            let mut by_g: BTreeMap<i64, (f64, f64)> = BTreeMap::new();
            for i in 0..n_all {
                let e = by_g.entry(ft[i]).or_insert((0.0, 0.0));
                e.0 += v[i];
                e.1 += 1.0;
            }
            xdm.push(
                keep.iter()
                    .map(|&i| {
                        let (sum, cnt) = by_g[&ft[i]];
                        v[i] - sum / cnt
                    })
                    .collect::<Vec<f64>>(),
            );
        }

        Ok(Spec {
            n,
            y: pick_f(y_all),
            unit,
            period,
            t,
            g,
            onset,
            reference,
            treated_cell,
            cell_of_row,
            treat: pick_f(treat_all),
            x,
            xdm,
            cells,
            cell_rows,
            periods,
            n_units,
            n_never_units,
            tref,
            gref_cohort,
            dropped_from,
            cgroup: config.cgroup,
            warnings,
        })
    }
}

/// A column of the ETWFE design.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Col {
    /// Cohort-period cell dummy.
    Cell(usize),
    /// Cell dummy times control `j` demeaned within cohort (`cell:x_dm`).
    CellX(usize, usize),
    /// Control `j` (`x`).
    X(usize),
    /// Cohort dummy times control `j` (`first_treat::v:x`).
    CohortX(i64, usize),
    /// Period dummy times control `j` (`time::s:x`).
    PeriodX(i64, usize),
}

/// The sparse ETWFE design: one column per cell, then the control columns.
struct Design {
    names: Vec<String>,
    /// Columns per control (`cell:x_dm` for every cell, `x`, cohort and period
    /// interactions); control `j`'s columns start at `n_cells + j * per_ctrl`.
    per_ctrl: usize,
    n_cells: usize,
    kinds: Vec<Col>,
    /// Row-major nonzeros: row `i` holds `row_col[row_ptr[i]..row_ptr[i + 1]]`.
    row_ptr: Vec<usize>,
    row_col: Vec<usize>,
    row_val: Vec<f64>,
    /// Column-major nonzeros.
    col_ptr: Vec<usize>,
    col_row: Vec<usize>,
    col_val: Vec<f64>,
}

impl Design {
    /// etwfe's design: the cells, then per control `x` (in `x_cols` order) its cell
    /// interactions `cell:x_dm` (x demeaned within cohort), `x`, its cohort interactions
    /// `first_treat::v:x` (non-reference cohorts) and its period interactions
    /// `time::s:x` (periods other than `tref`), as `etwfe(y ~ x)` builds them.
    fn build(spec: &Spec, names: &SpecColumns<'_>) -> EconResult<Design> {
        let n_cells = spec.cells.len();
        let cohorts: Vec<i64> = (0..spec.n)
            .filter(|&i| !spec.reference[i])
            .map(|i| spec.g[i])
            .collect::<BTreeSet<i64>>()
            .into_iter()
            .collect();
        let periods: Vec<i64> = spec
            .periods
            .iter()
            .copied()
            .filter(|&s| s != spec.tref)
            .collect();
        let per_ctrl = n_cells + 1 + cohorts.len() + periods.len();
        let k_raw = per_ctrl
            .checked_mul(spec.x.len())
            .and_then(|v| v.checked_add(n_cells))
            .ok_or_else(|| resource::overflow("ETWFE design"))?;
        let mut col_names: Vec<String> = resource::try_with_capacity(k_raw, "ETWFE design")?;
        let mut kinds: Vec<Col> = resource::try_with_capacity(k_raw, "ETWFE design")?;
        let cell_name = |&(g, t): &(i64, i64)| {
            format!(".Dtreat:{}::{g}:{}::{t}", names.first_treat, names.time)
        };
        for (c, cell) in spec.cells.iter().enumerate() {
            col_names.push(cell_name(cell));
            kinds.push(Col::Cell(c));
        }
        for (j, x) in names.controls.iter().enumerate() {
            for (c, cell) in spec.cells.iter().enumerate() {
                col_names.push(format!("{}:{x}_dm", cell_name(cell)));
                kinds.push(Col::CellX(c, j));
            }
            col_names.push(x.to_string());
            kinds.push(Col::X(j));
            for &v in &cohorts {
                col_names.push(format!("{}::{v}:{x}", names.first_treat));
                kinds.push(Col::CohortX(v, j));
            }
            for &s in &periods {
                col_names.push(format!("{}::{s}:{x}", names.time));
                kinds.push(Col::PeriodX(s, j));
            }
        }
        let per_row = 1 + 4 * spec.x.len();
        let cap = resource::checked_len(&[spec.n, per_row], "ETWFE design")?;
        let mut row_ptr = resource::try_with_capacity(spec.n + 1, "ETWFE design")?;
        let mut row_col = resource::try_with_capacity(cap, "ETWFE design")?;
        let mut row_val = resource::try_with_capacity(cap, "ETWFE design")?;
        row_ptr.push(0);
        for i in 0..spec.n {
            let cell = spec.cell_of_row[i];
            if let Some(c) = cell {
                row_col.push(c);
                row_val.push(1.0);
            }
            for (j, x) in spec.x.iter().enumerate() {
                let base = n_cells + j * per_ctrl;
                let mut push = |col: usize, v: f64| {
                    if v != 0.0 {
                        row_col.push(col);
                        row_val.push(v);
                    }
                };
                if let Some(c) = cell {
                    push(base + c, spec.xdm[j][i]);
                }
                push(base + n_cells, x[i]);
                if !spec.reference[i] {
                    if let Ok(v) = cohorts.binary_search(&spec.g[i]) {
                        push(base + n_cells + 1 + v, x[i]);
                    }
                }
                if let Ok(s) = periods.binary_search(&spec.t[i]) {
                    push(base + n_cells + 1 + cohorts.len() + s, x[i]);
                }
            }
            row_ptr.push(row_col.len());
        }
        Design::from_rows(col_names, kinds, row_ptr, row_col, row_val, per_ctrl)
    }

    fn from_rows(
        names: Vec<String>,
        kinds: Vec<Col>,
        row_ptr: Vec<usize>,
        row_col: Vec<usize>,
        row_val: Vec<f64>,
        per_ctrl: usize,
    ) -> EconResult<Design> {
        let n_cells = kinds.iter().filter(|k| matches!(k, Col::Cell(_))).count();
        let k = names.len();
        let nnz = row_col.len();
        let mut col_ptr = resource::try_vec(k + 1, 0usize, "ETWFE design")?;
        for &c in &row_col {
            col_ptr[c + 1] += 1;
        }
        for c in 0..k {
            col_ptr[c + 1] += col_ptr[c];
        }
        let mut next = col_ptr.clone();
        let mut col_row = resource::try_vec(nnz, 0usize, "ETWFE design")?;
        let mut col_val = resource::try_vec(nnz, 0.0f64, "ETWFE design")?;
        for i in 0..row_ptr.len() - 1 {
            for e in row_ptr[i]..row_ptr[i + 1] {
                let c = row_col[e];
                col_row[next[c]] = i;
                col_val[next[c]] = row_val[e];
                next[c] += 1;
            }
        }
        Ok(Design {
            names,
            per_ctrl,
            n_cells,
            kinds,
            row_ptr,
            row_col,
            row_val,
            col_ptr,
            col_row,
            col_val,
        })
    }

    fn ncols(&self) -> usize {
        self.names.len()
    }

    /// The nonzeros `(row, value)` of column `c`.
    fn column(&self, c: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let r = self.col_ptr[c]..self.col_ptr[c + 1];
        self.col_row[r.clone()]
            .iter()
            .copied()
            .zip(self.col_val[r].iter().copied())
    }

    /// Column `c` written densely into `out` (length n).
    fn scatter(&self, c: usize, out: &mut [f64]) {
        out.fill(0.0);
        for (i, v) in self.column(c) {
            out[i] = v;
        }
    }

    /// Centred sum of squares of raw column `c` over `n` rows.
    fn centred_ss(&self, c: usize, n: usize) -> f64 {
        let nnz = self.col_ptr[c + 1] - self.col_ptr[c];
        let mean = self.column(c).map(|(_, v)| v).sum::<f64>() / n as f64;
        self.column(c)
            .map(|(_, v)| (v - mean) * (v - mean))
            .sum::<f64>()
            + (n - nnz) as f64 * mean * mean
    }
}

/// Exact absorption of unit and period effects on a balanced or unbalanced panel:
/// sweep the unit means, then project off the unit-demeaned period dummies (periods
/// `1..T`) through their `(T-1) x (T-1)` normal equations `M` (Frisch-Waugh). With
/// `D~_is = 1[p_i = s] - n_{u_i s} / n_{u_i}`, `D~'D~ = M` and the residual of the
/// unit-demeaned `x` on `D~` is the residual of `x` on unit and period dummies.
struct TwoWayWithin {
    unit: Vec<usize>,
    period: Vec<usize>,
    n_units: usize,
    n_periods: usize,
    unit_n: Vec<f64>,
    /// Per unit, its periods `>= 1` with their row shares `n_up / n_u`.
    share_ptr: Vec<usize>,
    share_period: Vec<usize>,
    share_val: Vec<f64>,
    /// (Generalized) inverse of `M`.
    m_inv: Array2<f64>,
    /// Rank of the unit and period effects together.
    fe_rank: usize,
    /// Set when `M` is singular (the unit-period graph is disconnected).
    warning: Option<String>,
    /// Scratch reused by every [`Self::residualize_in_place`] call: per-unit means
    /// (`n_units`), the period right-hand side and its solution (`T - 1` each).
    scratch: RefCell<(Vec<f64>, Array1<f64>, Array1<f64>)>,
}

impl TwoWayWithin {
    fn new(unit: &[usize], period: &[usize], n_units: usize, n_periods: usize) -> EconResult<Self> {
        let n = unit.len();
        let mut pairs: Vec<(usize, usize)> = resource::try_with_capacity(n, "ETWFE panel cells")?;
        pairs.extend(unit.iter().copied().zip(period.iter().copied()));
        pairs.sort_unstable();
        let mut unit_n = resource::try_vec(n_units, 0.0f64, "ETWFE unit counts")?;
        for &u in unit {
            unit_n[u] += 1.0;
        }
        // Run lengths: (unit, period, rows) in unit order.
        let mut runs: Vec<(usize, usize, f64)> = Vec::new();
        for &(u, p) in &pairs {
            match runs.last_mut() {
                Some(last) if last.0 == u && last.1 == p => last.2 += 1.0,
                _ => runs.push((u, p, 1.0)),
            }
        }
        let dim = n_periods.saturating_sub(1);
        let mut m = resource::try_zeros2(dim, dim, "ETWFE period normal equations")?;
        let mut share_ptr = resource::try_vec(n_units + 1, 0usize, "ETWFE period shares")?;
        let mut share_period = Vec::new();
        let mut share_val = Vec::new();
        let mut start = 0;
        while start < runs.len() {
            resource::checkpoint()?;
            let u = runs[start].0;
            let mut end = start;
            while end < runs.len() && runs[end].0 == u {
                end += 1;
            }
            let nu = unit_n[u];
            let own: Vec<(usize, f64)> = runs[start..end]
                .iter()
                .filter(|r| r.1 >= 1)
                .map(|r| (r.1, r.2))
                .collect();
            for &(a, na) in &own {
                m[[a - 1, a - 1]] += na;
                for &(b, nb) in &own {
                    m[[a - 1, b - 1]] -= na * nb / nu;
                }
                share_period.push(a);
                share_val.push(na / nu);
            }
            share_ptr[u + 1] = share_period.len();
            start = end;
        }
        for u in 0..n_units {
            share_ptr[u + 1] = share_ptr[u + 1].max(share_ptr[u]);
        }
        let (m_inv, m_rank, warning) = if dim == 0 {
            (Array2::zeros((0, 0)), 0, None)
        } else {
            let inv = sym_inverse(
                m.view(),
                SingularPolicy::Generalized,
                "ETWFE period-effect normal equations",
            )?;
            let warning = inv.warning(
                "the period-effect normal equations (the unit-period graph is disconnected)",
            );
            (inv.inverse, inv.rank, warning)
        };
        Ok(Self {
            unit: unit.to_vec(),
            period: period.to_vec(),
            n_units,
            n_periods,
            unit_n,
            share_ptr,
            share_period,
            share_val,
            m_inv,
            fe_rank: n_units + m_rank,
            warning,
            scratch: RefCell::new((
                resource::try_vec(n_units, 0.0f64, "ETWFE unit means")?,
                resource::try_zeros1(dim, "ETWFE period right-hand side")?,
                resource::try_zeros1(dim, "ETWFE period effects")?,
            )),
        })
    }

    /// Replace `x` by its residual on unit and period dummies. O(n + T^2 + sum_u T_u).
    fn residualize_in_place(&self, x: &mut [f64]) {
        let mut scratch = self.scratch.borrow_mut();
        let (means, rhs, lambda) = &mut *scratch;
        means.fill(0.0);
        for (i, &v) in x.iter().enumerate() {
            means[self.unit[i]] += v;
        }
        for (m, &nu) in means.iter_mut().zip(&self.unit_n) {
            *m /= nu;
        }
        for (i, v) in x.iter_mut().enumerate() {
            *v -= means[self.unit[i]];
        }
        if self.n_periods < 2 {
            return;
        }
        rhs.fill(0.0);
        for (i, &v) in x.iter().enumerate() {
            let p = self.period[i];
            if p > 0 {
                rhs[p - 1] += v;
            }
        }
        general_mat_vec_mul(1.0, &self.m_inv, &*rhs, 0.0, lambda);
        // sum_a share_{u,a} lambda_a, once per unit
        for (u, m) in means.iter_mut().enumerate() {
            *m = (self.share_ptr[u]..self.share_ptr[u + 1])
                .map(|e| self.share_val[e] * lambda[self.share_period[e] - 1])
                .sum();
        }
        for (i, v) in x.iter_mut().enumerate() {
            let p = self.period[i];
            let lp = if p > 0 { lambda[p - 1] } else { 0.0 };
            *v -= lp - means[self.unit[i]];
        }
    }

    /// The residual of `x` on unit and period dummies.
    fn residualize(&self, x: &Array1<f64>) -> EconResult<Array1<f64>> {
        let mut out = resource::try_zeros1(x.len(), "ETWFE within transformation")?;
        out.assign(x);
        if let Some(s) = out.as_slice_mut() {
            self.residualize_in_place(s);
        }
        Ok(out)
    }
}

/// The fitted ETWFE regression on the within-transformed design.
struct Fit {
    /// Raw design column of each term, ascending.
    kept: Vec<usize>,
    /// Term index of each raw design column.
    term_of_col: Vec<Option<usize>>,
    /// Raw columns absorbed by the unit and period effects.
    absorbed: Vec<usize>,
    /// Raw control columns removed as collinear with the columns before them (by the
    /// Cholesky of the Gram, or by the pivoted QR when that fits), then refitted.
    collinear: Vec<usize>,
    beta: Array1<f64>,
    /// `(X_w'X_w)^{-1}` of the kept columns.
    unscaled: Array2<f64>,
    /// Cluster: `sqrt(c) B S'` (k x G), so that `w'Vw = ||Z'w||^2`.
    z: Option<Array2<f64>>,
    /// Iid residual variance.
    s2: f64,
    df_t: usize,
    ssr: f64,
    tss_within: f64,
}

impl Fit {
    /// `w'Vw` for a gradient given on its support `(term, weight)`.
    fn variance(&self, w: &[(usize, f64)]) -> f64 {
        match &self.z {
            Some(z) => (0..z.ncols())
                .map(|g| {
                    let s: f64 = w.iter().map(|&(j, wj)| z[[j, g]] * wj).sum();
                    s * s
                })
                .sum(),
            None => {
                let mut q = 0.0;
                for &(j, wj) in w {
                    for &(l, wl) in w {
                        q += wj * wl * self.unscaled[[j, l]];
                    }
                }
                self.s2 * q
            }
        }
    }

    fn term(&self, j: usize, name: String) -> EtwfeTerm {
        let estimate = self.beta[j];
        let std_error = self.variance(&[(j, 1.0)]).sqrt();
        let t_stat = estimate / std_error;
        EtwfeTerm {
            name,
            estimate,
            std_error,
            t_stat,
            p_value: t_test_p_value(t_stat, self.df_t as f64),
        }
    }

    /// The collinearity warning, when control columns were removed.
    fn warning(&self, design: &Design) -> Option<String> {
        (!self.collinear.is_empty()).then(|| {
            let names: Vec<&str> = self
                .collinear
                .iter()
                .map(|&c| design.names[c].as_str())
                .collect();
            format!(
                "Removed because collinear with other terms: {}",
                names.join(", ")
            )
        })
    }
}

#[cfg(test)]
thread_local! {
    /// Test switch: fit through the dense pivoted QR even when the normal equations qualify.
    static FORCE_DENSE_FIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Test probe: the path the last `fit_design` on this thread took (`Some(true)`: the
    /// structured normal equations; `Some(false)`: the dense pivoted QR).
    static LAST_FIT_STRUCTURED: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// A control column counts as collinear with the columns before it when the squared norm
/// of its within residual on them is below this fraction of its own within sum of squares
/// (a pivot of the column-order Cholesky of the equilibrated Gram). fixest's `collin.tol`
/// (1e-10, on its Cholesky of X'X), applied relative to the column so that the rule does not
/// depend on the scale of a control.
const GRAM_COLLIN_TOL: f64 = 1e-10;

/// The Gram positions that a column-order Cholesky of the equilibrated Gram `E = S G S`
/// (`S = diag(1 / sqrt G_jj)`) skips because their pivot is below [`GRAM_COLLIN_TOL`]: the
/// columns `lm.fit`'s limited pivoting and fixest's Cholesky would remove as collinear with
/// the columns before them. The first `n_lead` positions (the cells) are factored as one
/// block; `None` when one of them is skipped (a cell that is not identified is an error,
/// which the pivoted QR then names). Every Gram column has a positive diagonal (absorbed
/// columns are already out). The cell block and its Schur complement by faer, the controls
/// one by one: O(k1^3 / 3 + k1^2 k2 + k2^3 / 6) for `k1` cells and `k2` control columns.
fn gram_aliased(gram: &Array2<f64>, n_lead: usize) -> EconResult<Option<Vec<usize>>> {
    let k = gram.nrows();
    let (k1, k2) = (n_lead, k - n_lead);
    resource::check_alloc::<f64>(
        resource::checked_len(&[k, k, 3], "ETWFE collinearity check")?,
        "ETWFE collinearity check",
    )?;
    let sc: Vec<f64> = gram.diag().iter().map(|&d| 1.0 / d.sqrt()).collect();
    let e = |i: usize, j: usize| gram[[i, j]] * sc[i] * sc[j];
    let schur = if k1 == 0 {
        Mat::from_fn(k2, k2, &e)
    } else {
        let e11 = Mat::from_fn(k1, k1, &e);
        let Ok(llt) = e11.llt(faer::Side::Lower) else {
            return Ok(None);
        };
        let l = llt.L();
        if (0..k1).any(|j| l[(j, j)] * l[(j, j)] < GRAM_COLLIN_TOL) {
            return Ok(None);
        }
        resource::checkpoint()?;
        // W = L11^{-1} E12; the Schur complement E22 - W'W
        let mut w = Mat::from_fn(k1, k2, |i, j| e(i, k1 + j));
        faer::linalg::triangular_solve::solve_lower_triangular_in_place(
            l,
            w.as_mut(),
            faer::get_global_parallelism(),
        );
        resource::checkpoint()?;
        // upper triangle only: the factorisation below reads nothing else
        let mut schur = Mat::from_fn(k2, k2, |i, j| if i <= j { e(k1 + i, k1 + j) } else { 0.0 });
        faer::linalg::matmul::triangular::matmul(
            schur.as_mut(),
            faer::linalg::matmul::triangular::BlockStructure::TriangularUpper,
            faer::Accum::Add,
            w.transpose(),
            faer::linalg::matmul::triangular::BlockStructure::Rectangular,
            &w,
            faer::linalg::matmul::triangular::BlockStructure::Rectangular,
            -1.0,
            faer::get_global_parallelism(),
        );
        schur
    };
    // Up-looking Cholesky of the Schur complement (its upper triangle): row j of R from the
    // rows above it, a skipped column leaving its row zero. Column l of `r` holds R[0..l, l],
    // contiguous.
    let mut r = Mat::<f64>::zeros(k2, k2);
    let mut aliased = Vec::new();
    for j in 0..k2 {
        resource::checkpoint()?;
        let d = schur[(j, j)] - dot(&r.col_as_slice(j)[..j], &r.col_as_slice(j)[..j]);
        if d < GRAM_COLLIN_TOL {
            aliased.push(k1 + j);
            continue;
        }
        let rjj = d.sqrt();
        r[(j, j)] = rjj;
        for l in j + 1..k2 {
            let v = (schur[(j, l)] - dot(&r.col_as_slice(j)[..j], &r.col_as_slice(l)[..j])) / rjj;
            r[(j, l)] = v;
        }
    }
    Ok(Some(aliased))
}

/// `sum a_i b_i` in four lanes (no single dependency chain).
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = [0.0f64; 4];
    let (ac, ar) = a.as_chunks::<4>();
    let (bc, br) = b.as_chunks::<4>();
    for (a4, b4) in ac.iter().zip(bc) {
        for l in 0..4 {
            acc[l] += a4[l] * b4[l];
        }
    }
    let rest: f64 = ar.iter().zip(br).map(|(x, y)| x * y).sum();
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + rest
}

/// The "not identified" error for the cells among the raw columns `cols`.
fn unidentified_cells(spec: &Spec, design: &Design, cols: &[usize]) -> Option<EconError> {
    let cells: Vec<String> = cols
        .iter()
        .filter_map(|&c| match design.kinds[c] {
            Col::Cell(k) => Some(format!("({},{})", spec.cells[k].0, spec.cells[k].1)),
            _ => None,
        })
        .collect();
    (!cells.is_empty()).then(|| EconError::SingularMatrix {
        context: format!(
            "ETWFE cohort-period effects {} are not identified",
            cells.join(", ")
        ),
        suggestion: match spec.cgroup {
            ControlGroup::Never => {
                "they are collinear with the unit and period effects: no comparison unit is \
                 observed in those periods; use cgroup = notyet, drop those periods, or set gref"
            }
            ControlGroup::NotYet => {
                "they are collinear with the unit and period effects: no comparison unit is \
                 observed in those periods; drop those periods, or set gref"
            }
        }
        .to_string(),
    })
}

/// Fit the ETWFE design: structured normal equations of the within-transformed columns
/// (built column by column from the sparse design, solved through OLS's normal-equations
/// decision). When those are singular or ill-conditioned, the control columns that a
/// column-order Cholesky of the Gram finds collinear ([`gram_aliased`]) are removed and the
/// reduced normal equations decide again; the pivoted QR on the dense within-transformed
/// design is the last resort (a cell that is not identified, or a Gram that stays
/// ill-conditioned without collinear controls).
fn fit_design(
    spec: &Spec,
    design: &Design,
    within: &TwoWayWithin,
    vcov: EtwfeVcov,
) -> EconResult<Fit> {
    let n = spec.n;
    let k_raw = design.ncols();
    let mut y_w = resource::try_zeros1(n, "ETWFE within outcome")?;
    for (yw, &y) in y_w.iter_mut().zip(&spec.y) {
        *yw = y;
    }
    let mut col = resource::try_vec(n, 0.0f64, "ETWFE within column")?;
    if let Some(s) = y_w.as_slice_mut() {
        within.residualize_in_place(s);
    }

    // Pass over the columns: absorbed check, upper Gram columns, X'y_w.
    let mut term_of_col: Vec<Option<usize>> = vec![None; k_raw];
    let mut kept: Vec<usize> = Vec::new();
    let mut absorbed: Vec<usize> = Vec::new();
    let mut gram_cols: Vec<Vec<f64>> = Vec::new();
    let mut xty: Vec<f64> = Vec::new();
    for c in 0..k_raw {
        resource::checkpoint()?;
        let css = design.centred_ss(c, n);
        design.scatter(c, &mut col);
        within.residualize_in_place(&mut col);
        let ss: f64 = col.iter().map(|v| v * v).sum();
        if css == 0.0 || ss <= ABSORBED_TOL * css {
            absorbed.push(c);
            continue;
        }
        let kk = kept.len();
        term_of_col[c] = Some(kk);
        kept.push(c);
        let mut gc = resource::try_vec(kk + 1, 0.0f64, "ETWFE cross-products")?;
        for (i, &r) in col.iter().enumerate() {
            if r == 0.0 {
                continue;
            }
            for e in design.row_ptr[i]..design.row_ptr[i + 1] {
                let k = design.row_col[e];
                if k <= c {
                    if let Some(kj) = term_of_col[k] {
                        gc[kj] += design.row_val[e] * r;
                    }
                }
            }
        }
        gram_cols.push(gc);
        xty.push(design.column(c).map(|(i, v)| v * y_w[i]).sum());
    }
    if let Some(e) = unidentified_cells(spec, design, &absorbed) {
        return Err(e);
    }
    let k = kept.len();
    let mut gram = resource::try_zeros2(k, k, "ETWFE cross-products")?;
    for (c, gc) in gram_cols.iter().enumerate() {
        for (r, &v) in gc.iter().enumerate() {
            gram[[r, c]] = v;
            gram[[c, r]] = v;
        }
    }
    drop(gram_cols);
    #[cfg(test)]
    let force_dense = FORCE_DENSE_FIT.with(|f| f.get());
    #[cfg(not(test))]
    let force_dense = false;
    let mut collinear: Vec<usize> = Vec::new();
    let mut normal = None;
    if k > 0 && k < n && !force_dense {
        // Collinear controls (a control's cell interaction in a one-unit cohort is
        // proportional to the cell) leave at the Gram level before the solve decision, so
        // the structured path survives them.
        let n_lead = kept.iter().filter(|&&c| c < design.n_cells).count();
        if n_lead < k {
            if let Some(drop) =
                gram_aliased(&gram, n_lead)?.filter(|d| !d.is_empty() && d.len() < k)
            {
                let keep_j: Vec<usize> =
                    (0..k).filter(|j| drop.binary_search(j).is_err()).collect();
                let kr = keep_j.len();
                let mut reduced = resource::try_zeros2(kr, kr, "ETWFE cross-products")?;
                for (a, &ja) in keep_j.iter().enumerate() {
                    for (b, &jb) in keep_j.iter().enumerate() {
                        reduced[[a, b]] = gram[[ja, jb]];
                    }
                }
                gram = reduced;
                collinear = drop.iter().map(|&j| kept[j]).collect();
                xty = keep_j.iter().map(|&j| xty[j]).collect();
                kept = keep_j.iter().map(|&j| kept[j]).collect();
                term_of_col = vec![None; k_raw];
                for (j, &c) in kept.iter().enumerate() {
                    term_of_col[c] = Some(j);
                }
            }
        }
        normal = normal_inverse_from_gram(gram.view())?;
    }
    drop(gram);
    #[cfg(test)]
    LAST_FIT_STRUCTURED.with(|p| p.set(Some(normal.is_some())));

    // y - X beta over the kept columns, within-transformed
    let within_residual = |beta: &Array1<f64>, kept: &[usize]| -> EconResult<Array1<f64>> {
        let mut u = resource::try_zeros1(n, "ETWFE residuals")?;
        for (ui, &y) in u.iter_mut().zip(&spec.y) {
            *ui = y;
        }
        for (j, &c) in kept.iter().enumerate() {
            for (i, v) in design.column(c) {
                u[i] -= v * beta[j];
            }
        }
        if let Some(s) = u.as_slice_mut() {
            within.residualize_in_place(s);
        }
        Ok(u)
    };

    let (beta, unscaled, resid, dense) = match normal {
        Some(inv) => {
            let xty = Array1::from(xty);
            let mut beta = inv.inverse.dot(&xty);
            if inv.refine {
                let u = within_residual(&beta, &kept)?;
                let g: Array1<f64> = kept
                    .iter()
                    .map(|&c| design.column(c).map(|(i, v)| v * u[i]).sum())
                    .collect();
                beta += &inv.inverse.dot(&g);
            }
            let u = within_residual(&beta, &kept)?;
            (beta, inv.inverse, u, None)
        }
        None => {
            let mut x_w = resource::try_zeros2(n, kept.len(), "ETWFE within design")?;
            for (j, &c) in kept.iter().enumerate() {
                resource::checkpoint()?;
                design.scatter(c, &mut col);
                within.residualize_in_place(&mut col);
                x_w.column_mut(j).assign(&ArrayView1::from(&col[..]));
            }
            // Remove the aliased columns and refit until the QR aliases none (a refit
            // normally aliases nothing; the loop keeps a second round from leaving NaN
            // coefficients behind).
            let mut ls = least_squares(x_w.view(), y_w.view())?;
            while !ls.aliased.is_empty() {
                resource::checkpoint()?;
                let aliased: Vec<usize> = ls.aliased.iter().map(|&j| kept[j]).collect();
                if let Some(e) = unidentified_cells(spec, design, &aliased) {
                    return Err(e);
                }
                let keep_j: Vec<usize> = (0..kept.len())
                    .filter(|j| !ls.aliased.contains(j))
                    .collect();
                let mut reduced = resource::try_zeros2(n, keep_j.len(), "ETWFE within design")?;
                for (jj, &j) in keep_j.iter().enumerate() {
                    reduced.column_mut(jj).assign(&x_w.column(j));
                }
                x_w = reduced;
                ls = least_squares(x_w.view(), y_w.view())?;
                collinear.extend(aliased);
                collinear.sort_unstable();
                kept = keep_j.iter().map(|&j| kept[j]).collect();
                term_of_col = vec![None; k_raw];
                for (j, &c) in kept.iter().enumerate() {
                    term_of_col[c] = Some(j);
                }
            }
            (ls.coefficients, ls.unscaled_cov, ls.residuals, Some(x_w))
        }
    };
    let k = kept.len();
    let ssr: f64 = resid.iter().map(|u| u * u).sum();
    let tss_within: f64 = y_w.iter().map(|v| v * v).sum();

    let (z, s2, df_t) = match vcov {
        EtwfeVcov::Cluster => {
            let g = spec.n_units;
            let big_k = k + spec.periods.len();
            let dof = n.checked_sub(big_k).filter(|&v| v > 0).ok_or_else(|| {
                EconError::InsufficientData {
                    required: big_k + 1,
                    provided: n,
                    context: "ETWFE clustered variance".to_string(),
                }
            })?;
            // S' (k x G): per-unit scores of the within design, one row per term
            let mut scores_t = resource::try_zeros2(k, g, "ETWFE cluster scores")?;
            match &dense {
                Some(x_w) => {
                    for i in 0..n {
                        let u = spec.unit[i];
                        for j in 0..k {
                            scores_t[[j, u]] += x_w[[i, j]] * resid[i];
                        }
                    }
                }
                None => {
                    let mut acc = resource::try_vec(g, 0.0f64, "ETWFE cluster scores")?;
                    let units = &spec.unit[..n];
                    for (j, &c) in kept.iter().enumerate() {
                        resource::checkpoint()?;
                        design.scatter(c, &mut col);
                        within.residualize_in_place(&mut col);
                        acc.fill(0.0);
                        for ((&r, &e), &u) in col.iter().zip(resid.iter()).zip(units) {
                            acc[u] += r * e;
                        }
                        scores_t.row_mut(j).assign(&ArrayView1::from(&acc[..]));
                    }
                }
            }
            let gf = g as f64;
            let factor = gf / (gf - 1.0) * (n as f64 - 1.0) / dof as f64;
            let z = unscaled.dot(&scores_t) * factor.sqrt();
            (Some(z), f64::NAN, g - 1)
        }
        EtwfeVcov::Iid => {
            let df = n
                .checked_sub(k + within.fe_rank)
                .filter(|&v| v > 0)
                .ok_or_else(|| EconError::InsufficientData {
                    required: k + within.fe_rank + 1,
                    provided: n,
                    context: "ETWFE residual degrees of freedom".to_string(),
                })?;
            (None, ssr / df as f64, df)
        }
    };
    Ok(Fit {
        kept,
        term_of_col,
        absorbed,
        collinear,
        beta,
        unscaled,
        z,
        s2,
        df_t,
        ssr,
        tss_within,
    })
}

/// Whether row `i` is among the rows etwfe's `emfx(type)` averages over (etwfe 0.6.2,
/// `post_only = TRUE`, `compress = FALSE`): with `cgroup = notyet` the treated cohort-period
/// rows; with `cgroup = never` every row of a treated cohort for the event study (placebo
/// leads and the reference period included) and its rows from the onset on otherwise. The
/// event-study rows include the others.
fn emfx_row(spec: &Spec, i: usize, event: bool) -> bool {
    match spec.cgroup {
        ControlGroup::NotYet => spec.treated_cell[i],
        ControlGroup::Never => {
            !spec.reference[i] && (event || (spec.treated_cell[i] && spec.t[i] >= spec.onset[i]))
        }
    }
}

/// The four emfx tables (simple, event, group, calendar) in one pass over the rows, each
/// row's effect gradient computed once. Per table and key (ascending): the mean over its rows
/// of the per-row effect gradient `w`, the estimate `w'beta`, its delta-method SE
/// `sqrt(w'Vw)` (NaN when no row carries a coefficient), z p-value and CI at `z_crit`.
fn emfx_tables(
    spec: &Spec,
    fit: &Fit,
    z_crit: f64,
    gradient: &dyn Fn(usize, &mut Vec<(usize, f64)>),
) -> EconResult<[Vec<AggregatedEffect>; 4]> {
    #[derive(Default)]
    struct Acc {
        rows: usize,
        w: BTreeMap<usize, f64>,
        cells: BTreeSet<usize>,
    }
    const WHICH: [EtwfeAggregation; 4] = [
        EtwfeAggregation::Simple,
        EtwfeAggregation::Event,
        EtwfeAggregation::Group,
        EtwfeAggregation::Calendar,
    ];
    let mut tables: [BTreeMap<i64, Acc>; 4] = Default::default();
    let mut g = Vec::new();
    for i in 0..spec.n {
        if i % 4096 == 0 {
            resource::checkpoint()?;
        }
        if !emfx_row(spec, i, true) {
            continue;
        }
        let post = emfx_row(spec, i, false);
        g.clear();
        gradient(i, &mut g);
        let cell = spec.cell_of_row[i].filter(|_| !g.is_empty());
        for (which, table) in WHICH.iter().zip(tables.iter_mut()) {
            let key = match which {
                EtwfeAggregation::Event => spec.t[i] - spec.g[i],
                _ if !post => continue,
                EtwfeAggregation::Simple => 0,
                EtwfeAggregation::Group => spec.g[i],
                EtwfeAggregation::Calendar => spec.t[i],
            };
            let acc = table.entry(key).or_default();
            acc.rows += 1;
            if let Some(c) = cell {
                acc.cells.insert(c);
            }
            for &(j, v) in &g {
                *acc.w.entry(j).or_insert(0.0) += v;
            }
        }
    }
    let finish = |which: EtwfeAggregation, table: BTreeMap<i64, Acc>| -> Vec<AggregatedEffect> {
        table
            .into_iter()
            .map(|(key, acc)| {
                let m = acc.rows as f64;
                let w: Vec<(usize, f64)> = acc.w.into_iter().map(|(j, v)| (j, v / m)).collect();
                let estimate: f64 = w.iter().map(|&(j, v)| v * fit.beta[j]).sum();
                let std_error = if w.is_empty() {
                    f64::NAN
                } else {
                    fit.variance(&w).sqrt()
                };
                let z_stat = estimate / std_error;
                AggregatedEffect {
                    key,
                    agg_type: which.to_string(),
                    estimate,
                    std_error,
                    ci_lower: estimate - z_crit * std_error,
                    ci_upper: estimate + z_crit * std_error,
                    n_effects: acc.cells.len(),
                    z_stat,
                    p_value: normal_p_value_two_sided(z_stat),
                    n_obs: acc.rows,
                }
            })
            .collect()
    };
    let [s0, s1, s2, s3] = tables;
    Ok([
        finish(WHICH[0], s0),
        finish(WHICH[1], s1),
        finish(WHICH[2], s2),
        finish(WHICH[3], s3),
    ])
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
    fn test_etwfe_cohort_average_recovers_dgp() {
        // etwfe(ivar = uid, vcov = ~uid) on this fixture: 1.99372 and 5.00763 (truth 2, 5)
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
        assert_eq!(result.cohort_avg.len(), 2);
        for avg in &result.cohort_avg {
            let truth = true_atts[&avg.key];
            assert!(
                (avg.estimate - truth).abs() < 0.02,
                "cohort {}: {} vs {truth}",
                avg.key,
                avg.estimate
            );
        }
    }

    #[test]
    fn test_etwfe_event_study_recovers_dgp() {
        // event e = 0..2: cohorts 3 and 5 (2 and 5, equal sizes); e = 3, 4: cohort 3 only
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
        let got: Vec<(i64, f64)> = result
            .event_study
            .iter()
            .map(|e| (e.key, e.estimate))
            .collect();
        let truth = [(0, 3.5), (1, 3.5), (2, 3.5), (3, 2.0), (4, 2.0)];
        assert_eq!(got.len(), truth.len());
        for ((k, e), (tk, t)) in got.iter().zip(truth) {
            assert_eq!(*k, tk);
            assert!((e - t).abs() < 0.02, "e = {k}: {e} vs {t}");
        }
    }

    #[test]
    fn test_etwfe_heterogeneous_vs_homogeneous_recovers_dgp() {
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

        // etwfe on the fixtures: 3.49894 (truth 3.5) and 3.12394 (truth 3.125)
        assert!(
            (result_hom.att_simple - homogeneous_att).abs() < 0.02,
            "Homogeneous ATT should be ~{}, got {:.4}",
            homogeneous_att,
            result_hom.att_simple
        );

        // Heterogeneous ATT should be between the min and max true ATTs (2 and 5)
        assert!(
            (result_het.att_simple - 3.125).abs() < 0.02,
            "Heterogeneous ATT should be ~3.125, got {:.4}",
            result_het.att_simple
        );
    }

    #[test]
    fn test_etwfe_structural_counts() {
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

    // ═══════════════════════════════════════════════════════════════════════
    // T2-WP12: columns through the shared extractor, no imputation (CAUS-63)
    // ═══════════════════════════════════════════════════════════════════════

    fn etwfe(ds: &Dataset) -> EconResult<EtwfeResult> {
        run_etwfe(ds, "y", "unit", "time", "treat", "first_treat", None, None)
    }

    fn i64_col(ds: &Dataset, col: &str) -> Vec<i64> {
        ds.df()
            .column(col)
            .unwrap()
            .i64()
            .unwrap()
            .iter()
            .map(|v| v.unwrap())
            .collect()
    }

    /// `ds` with nulls in the Int64 column `col` on the rows where `null` is true.
    fn null_i64(ds: &Dataset, col: &str, null: impl Fn(usize) -> bool) -> Dataset {
        let v: Vec<Option<i64>> = i64_col(ds, col)
            .into_iter()
            .enumerate()
            .map(|(i, x)| if null(i) { None } else { Some(x) })
            .collect();
        let mut df = ds.df().clone();
        df.with_column(Column::new(col.into(), v)).unwrap();
        Dataset::new(df)
    }

    fn without_rows(ds: &Dataset, drop: impl Fn(usize) -> bool) -> Dataset {
        let keep =
            BooleanChunked::from_iter_values("keep".into(), (0..ds.nrows()).map(|i| !drop(i)));
        Dataset::new(ds.df().filter(&keep).unwrap())
    }

    fn assert_same_etwfe(a: &EtwfeResult, b: &EtwfeResult) {
        let close = |x: f64, y: f64| x == y || (x - y).abs() <= 1e-12 * x.abs().max(y.abs());
        assert_eq!(a.n_obs, b.n_obs);
        assert_eq!(a.n_periods, b.n_periods);
        assert_eq!(a.cohort_time_effects.len(), b.cohort_time_effects.len());
        for (x, y) in a.cohort_time_effects.iter().zip(&b.cohort_time_effects) {
            assert_eq!(
                (x.cohort, x.time, x.n_treated),
                (y.cohort, y.time, y.n_treated)
            );
            assert!(
                close(x.estimate, y.estimate),
                "{} vs {}",
                x.estimate,
                y.estimate
            );
            assert!(
                close(x.std_error, y.std_error),
                "{} vs {}",
                x.std_error,
                y.std_error
            );
        }
        for (x, y) in [
            (a.att_simple, b.att_simple),
            (a.att_se, b.att_se),
            (a.att_weighted, b.att_weighted),
            (a.att_weighted_se, b.att_weighted_se),
        ] {
            assert!(close(x, y), "{x} vs {y}");
        }
    }

    #[test]
    fn test_na_etwfe_null_first_treat_drops_rows() {
        let (ds, _) = create_validation_etwfe_panel();
        let units = i64_col(&ds, "unit");
        let gone = |i: usize| units[i] == 1 || units[i] == 11;
        let na = null_i64(&ds, "first_treat", gone);
        let a = etwfe(&na).unwrap();
        let b = etwfe(&without_rows(&ds, gone)).unwrap();
        assert_same_etwfe(&a, &b);
        assert_eq!(a.n_obs, 266);
        assert_eq!(a.missing.by_column, vec![("first_treat".to_string(), 14)]);
        assert!(b.missing.is_empty());
        let shown = format!("{a}");
        assert!(
            shown.trim_end().ends_with(
                "Missing values: 14 of 280 rows dropped because of missing values \
                 (first_treat: 14)"
            ),
            "{shown}"
        );
    }

    #[test]
    fn test_na_etwfe_null_ids_and_time_drop_rows() {
        let (ds, _) = create_validation_etwfe_panel();
        let (units, times) = (i64_col(&ds, "unit"), i64_col(&ds, "time"));
        let row = |u: i64, t: i64| (0..units.len()).find(|&i| units[i] == u && times[i] == t);
        let (r_time, r_unit) = (row(25, 4).unwrap(), row(30, 6).unwrap());
        let na = null_i64(&null_i64(&ds, "time", |i| i == r_time), "unit", |i| {
            i == r_unit
        });
        let a = etwfe(&na).unwrap();
        let b = etwfe(&without_rows(&ds, |i| i == r_time || i == r_unit)).unwrap();
        assert_same_etwfe(&a, &b);
        assert_eq!(a.n_periods, 7);
        assert_eq!(a.n_obs, 278);
        assert_eq!(
            a.missing.by_column,
            vec![("unit".to_string(), 1), ("time".to_string(), 1)]
        );
    }

    #[test]
    fn test_na_etwfe_errors_name_the_column() {
        let (ds, _) = create_validation_etwfe_panel();
        let text: Vec<String> = ds
            .df()
            .column("y")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .map(|v| format!("{}", v.unwrap()))
            .collect();
        let mut df = ds.df().clone();
        df.with_column(Column::new("y".into(), text)).unwrap();
        match etwfe(&Dataset::new(df)) {
            Err(EconError::NonNumericColumn { column }) => assert_eq!(column, "y"),
            other => panic!("expected NonNumericColumn, got {other:?}"),
        }
        let mut time: Vec<f64> = i64_col(&ds, "time").iter().map(|&t| t as f64).collect();
        time[5] = 2.5;
        let mut df = ds.df().clone();
        df.with_column(Column::new("time".into(), time)).unwrap();
        match etwfe(&Dataset::new(df)) {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(message.contains("whole numbers"), "{message}")
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
        let df = ds.df().drop("treat").unwrap();
        match etwfe(&Dataset::new(df)) {
            Err(EconError::ColumnNotFound { column, .. }) => assert_eq!(column, "treat"),
            other => panic!("expected ColumnNotFound, got {other:?}"),
        }
        // never-treated units coded as -inf or inf: the error names the column and
        // the code to use instead
        let g = i64_col(&ds, "first_treat");
        let first_never = g.iter().position(|&g| g == 0).unwrap();
        for (code, shown) in [(f64::NEG_INFINITY, "-inf"), (f64::INFINITY, "inf")] {
            let ft: Vec<f64> = g
                .iter()
                .map(|&g| if g == 0 { code } else { g as f64 })
                .collect();
            let mut df = ds.df().clone();
            df.with_column(Column::new("first_treat".into(), ft))
                .unwrap();
            match etwfe(&Dataset::new(df)) {
                Err(EconError::InvalidSpecification { message }) => assert_eq!(
                    message,
                    format!(
                        "Column 'first_treat' (the first treatment period) is {shown} in row \
                         {first_never} (0-based); never-treated units must be coded 0, not \
                         infinity."
                    )
                ),
                other => panic!("expected InvalidSpecification, got {other:?}"),
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // T3-WP22: etwfe(ivar = unit) + emfx (validation/r/cases/causal_etwfe.R)
    // ═══════════════════════════════════════════════════════════════════════

    use crate::traits::convergence::notes_text;
    use crate::traits::estimator::z_critical;
    use p2a_validation::{Check, Tol};

    fn fit(ds: &Dataset, x: Option<&[&str]>, cfg: EtwfeConfig) -> EtwfeResult {
        run_etwfe(ds, "y", "id", "year", "treat", "first_treat", x, Some(cfg)).unwrap()
    }

    /// The path the last fit on this thread took: `true` for the structured normal
    /// equations, `false` for the dense pivoted QR.
    fn last_fit_structured() -> bool {
        LAST_FIT_STRUCTURED
            .with(|p| p.take())
            .expect("fit_design ran on this thread")
    }

    fn spec(cgroup: ControlGroup) -> EtwfeConfig {
        EtwfeConfig {
            cgroup,
            ..EtwfeConfig::default()
        }
    }

    /// A generated CSV of the ETWFE cases, read without loading a case (property tests).
    fn panel_csv(name: &str) -> Dataset {
        let p = p2a_validation::repo_path(&format!("validation/datasets/generated/{name}.csv"));
        Dataset::new(crate::data::DataLoader::load_csv(&p).unwrap())
    }

    fn all_treated_panel() -> Dataset {
        panel_csv("etwfe_panel_all_treated")
    }

    fn with_i64(ds: &Dataset, col: &str, f: impl Fn(usize, i64) -> i64) -> Dataset {
        let v: Vec<i64> = i64_col(ds, col)
            .into_iter()
            .enumerate()
            .map(|(i, x)| f(i, x))
            .collect();
        let mut df = ds.df().clone();
        df.with_column(Column::new(col.into(), v)).unwrap();
        Dataset::new(df)
    }

    /// `ds` without the rows where `pred(first_treat, year)` holds.
    fn drop_rows(ds: &Dataset, pred: impl Fn(i64, i64) -> bool) -> Dataset {
        let (g, t) = (i64_col(ds, "first_treat"), i64_col(ds, "year"));
        without_rows(ds, |i| pred(g[i], t[i]))
    }

    /// `<prefix>_coef|_se|_pvalue` (named as fixest) and `<prefix>_nobs`.
    fn check_terms(ck: &mut Check<'_>, prefix: &str, r: &EtwfeResult) {
        let names: Vec<&str> = r.terms.iter().map(|t| t.name.as_str()).collect();
        let col = |f: fn(&EtwfeTerm) -> f64| r.terms.iter().map(f).collect::<Vec<f64>>();
        let n = r.n_obs;
        ck.named(
            &format!("{prefix}_coef"),
            &names,
            &col(|t| t.estimate),
            Tol::coef(n),
        )
        .named(
            &format!("{prefix}_se"),
            &names,
            &col(|t| t.std_error),
            Tol::se(n),
        )
        .named(
            &format!("{prefix}_pvalue"),
            &names,
            &col(|t| t.p_value),
            Tol::p_value(n),
        )
        .scalar(&format!("{prefix}_nobs"), n as f64, Tol::exact());
    }

    #[test]
    fn test_validate_etwfe_notyet() {
        let (case, ds) = crate::test_support::load_case("etwfe_staggered");
        let r = fit(&ds, None, EtwfeConfig::default());
        assert!(
            last_fit_structured(),
            "R-pinned case fitted through the dense QR"
        );
        let mut ck = Check::new(&case);
        check_terms(&mut ck, "notyet", &r);
        ck.scalar("n_clusters", r.n_units as f64, Tol::exact());
        ck.finish();
        assert_eq!(
            (r.vcov, r.df_t, r.ref_time, r.ref_cohort, r.n_never_units),
            (EtwfeVcov::Cluster, 119, 1, None, 35)
        );
        assert_eq!(r.cohort_time_effects.len(), 9);
        assert!(r.dropped_terms.is_empty() && r.warnings.is_empty());
    }

    #[test]
    fn test_validate_etwfe_iid() {
        let (case, ds) = crate::test_support::load_case("etwfe_staggered");
        let mut c = EtwfeConfig::default();
        c.vcov = EtwfeVcov::Iid;
        let r = fit(&ds, None, c);
        let mut ck = Check::new(&case);
        check_terms(&mut ck, "notyet_iid", &r);
        ck.scalar("notyet_iid_df", r.df_t as f64, Tol::exact());
        ck.finish();
    }

    #[test]
    fn test_validate_etwfe_never() {
        let (case, ds) = crate::test_support::load_case("etwfe_staggered");
        let r = fit(&ds, None, spec(ControlGroup::Never));
        let mut ck = Check::new(&case);
        check_terms(&mut ck, "never", &r);
        ck.finish();
        assert_eq!(r.cohort_time_effects.len(), 21);
        assert!(r.cohort_time_effects.iter().all(|c| c.rel_time != -1));
    }

    #[test]
    fn test_validate_etwfe_tref() {
        let (case, ds) = crate::test_support::load_case("etwfe_staggered");
        let mut c = EtwfeConfig::default();
        c.tref = Some(4);
        let r = fit(&ds, None, c);
        let mut ck = Check::new(&case);
        check_terms(&mut ck, "tref4", &r);
        ck.finish();
        assert_eq!(r.ref_time, 4);
        assert!(!r.cohort_time_effects.iter().any(|c| c.time == 4));
    }

    #[test]
    fn test_validate_etwfe_anticipation() {
        let (case, ds) = crate::test_support::load_case("etwfe_staggered");
        let mut c = EtwfeConfig::default();
        c.anticipation = 1;
        let r = fit(&ds, None, c);
        let cells = &r.cohort_time_effects;
        // R fits etwfe on the onset first_treat - 1 (column first_treat_ant)
        let names: Vec<String> = cells
            .iter()
            .map(|e| format!(".Dtreat:first_treat_ant::{}:year::{}", e.cohort - 1, e.time))
            .collect();
        let v = |f: fn(&CohortTimeEffect) -> f64| cells.iter().map(f).collect::<Vec<f64>>();
        let n = r.n_obs;
        Check::new(&case)
            .named("ant1_coef", &names, &v(|e| e.estimate), Tol::coef(n))
            .named("ant1_se", &names, &v(|e| e.std_error), Tol::se(n))
            .named("ant1_pvalue", &names, &v(|e| e.p_value), Tol::p_value(n))
            .scalar("ant1_nobs", n as f64, Tol::exact())
            .finish();
        let leads: Vec<(i64, i64)> = cells
            .iter()
            .filter(|e| e.rel_time == -1)
            .map(|e| (e.cohort, e.time))
            .collect();
        assert_eq!(leads, vec![(4, 3), (6, 5), (8, 7)]);
        assert_eq!(r.anticipation, 1);
    }

    #[test]
    fn test_validate_etwfe_unbalanced() {
        let (case, ds) = crate::test_support::load_case("etwfe_unbalanced");
        let mut ck = Check::new(&case);
        check_terms(&mut ck, "unbal", &fit(&ds, None, EtwfeConfig::default()));
        check_terms(
            &mut ck,
            "unbal_never",
            &fit(&ds, None, spec(ControlGroup::Never)),
        );
        ck.finish();
    }

    #[test]
    fn test_validate_etwfe_all_treated() {
        let (case, _) = crate::test_support::load_case("etwfe_unbalanced");
        let ds = all_treated_panel();
        let r = fit(&ds, None, EtwfeConfig::default());
        let mut gref6 = EtwfeConfig::default();
        gref6.gref = Some(6);
        let mut never8 = spec(ControlGroup::Never);
        never8.gref = Some(8);
        let mut ck = Check::new(&case);
        check_terms(&mut ck, "alltreated", &r);
        check_terms(&mut ck, "alltreated_gref6", &fit(&ds, None, gref6));
        check_terms(&mut ck, "alltreated_never_gref8", &fit(&ds, None, never8));
        ck.finish();
        assert_eq!(
            (r.ref_cohort, r.n_periods, r.n_never_units, r.n_obs),
            (Some(8), 7, 0, 840)
        );
        let never = run_etwfe(
            &ds,
            "y",
            "id",
            "year",
            "treat",
            "first_treat",
            None,
            Some(spec(ControlGroup::Never)),
        );
        match never {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(message.contains("cgroup = never"), "{message}")
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    #[test]
    fn test_etwfe_unidentified_cells_are_an_error() {
        // evidence: on the all-treated panel with cgroup = never and gref = 8, fixest
        // drops .Dtreat:first_treat::6:year::2 as collinear; p2a refuses it
        let ds = all_treated_panel();
        let mut c = spec(ControlGroup::Never);
        c.gref = Some(8);
        // QR-aliased: year 2 holds cohorts 4 and 6 only
        let aliased = drop_rows(&ds, |g, t| g == 8 && t == 2);
        // absorbed: year 2 holds cohort 6 only
        let absorbed = drop_rows(&ds, |g, t| (g == 4 || g == 8) && t == 2);
        for (d, rows) in [(aliased, 927), (absorbed, 885)] {
            assert_eq!(d.nrows(), rows);
            let r = run_etwfe(
                &d,
                "y",
                "id",
                "year",
                "treat",
                "first_treat",
                None,
                Some(c.clone()),
            );
            match r {
                Err(EconError::SingularMatrix { context, .. }) => {
                    assert!(
                        context.contains("(6,2)") && context.contains("not identified"),
                        "{context}"
                    )
                }
                other => panic!("expected SingularMatrix, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_etwfe_rejects_degenerate_inputs() {
        let ds = panel_csv("etwfe_panel");
        let err = |d: &Dataset, c: EtwfeConfig| {
            run_etwfe(d, "y", "id", "year", "treat", "first_treat", None, Some(c))
                .expect_err("expected an error")
        };
        let mut c = EtwfeConfig::default();
        c.tref = Some(99);
        let e = err(&ds, c).to_string();
        assert!(e.contains("tref 99"), "{e}");
        let mut c = EtwfeConfig::default();
        c.gref = Some(5);
        let e = err(&ds, c).to_string();
        assert!(e.contains("gref 5"), "{e}");
        // row 0 is id 1, year 1, first_treat 4, treat 0
        let d = with_i64(&ds, "treat", |i, v| if i == 0 { 1 } else { v });
        let e = err(&d, EtwfeConfig::default()).to_string();
        assert!(e.contains("treat") && e.contains("first_treat is 4"), "{e}");
        let d = with_i64(&ds, "first_treat", |i, v| if i == 0 { 6 } else { v });
        let e = err(&d, EtwfeConfig::default()).to_string();
        assert!(e.contains("constant within each unit"), "{e}");
        let d = with_i64(&with_i64(&ds, "first_treat", |_, _| 0), "treat", |_, _| 0);
        let e = err(&d, EtwfeConfig::default()).to_string();
        assert!(e.contains("no treated cohort"), "{e}");
        let year1 = ds.df().column("year").unwrap().i64().unwrap().equal(1);
        let d = Dataset::new(ds.df().filter(&year1).unwrap());
        match err(&d, EtwfeConfig::default()) {
            EconError::InsufficientData {
                required: 2,
                provided: 1,
                ..
            } => {}
            other => panic!("expected InsufficientData, got {other:?}"),
        }
        let mut c = EtwfeConfig::default();
        c.anticipation = usize::MAX;
        let e = err(&ds, c).to_string();
        assert!(e.contains("anticipation"), "{e}");
        // nothing post-treatment to estimate: cohort 8's only treated period is tref
        let d = drop_rows(&ds, |g, _| g == 4 || g == 6);
        let mut c = EtwfeConfig::default();
        c.tref = Some(8);
        let e = err(&d, c).to_string();
        assert!(e.contains("no cohort-period cell to estimate"), "{e}");
    }

    #[test]
    fn test_etwfe_two_way_within_matches_dummy_regression() {
        let ds = panel_csv("etwfe_panel_unbalanced");
        let u: Vec<usize> = i64_col(&ds, "id").iter().map(|&v| v as usize - 1).collect();
        let p: Vec<usize> = i64_col(&ds, "year")
            .iter()
            .map(|&v| v as usize - 1)
            .collect();
        let y: Array1<f64> = ds
            .df()
            .column("y")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .map(|v| v.unwrap())
            .collect();
        let n = y.len();
        let mut d = Array2::<f64>::zeros((n, 1 + 119 + 7));
        for i in 0..n {
            d[[i, 0]] = 1.0;
            if u[i] > 0 {
                d[[i, u[i]]] = 1.0;
            }
            if p[i] > 0 {
                d[[i, 119 + p[i]]] = 1.0;
            }
        }
        let want = crate::linalg::solve::least_squares(d.view(), y.view())
            .unwrap()
            .residuals;
        let tw = TwoWayWithin::new(&u, &p, 120, 8).unwrap();
        let got = tw.residualize(&y).unwrap();
        for i in 0..n {
            assert!(
                (got[i] - want[i]).abs() < 1e-10,
                "row {i}: {} vs {}",
                got[i],
                want[i]
            );
        }
        assert_eq!(tw.fe_rank, 127);
        assert!(tw.warning.is_none());
        // two disconnected balanced 2 x 2 blocks
        let u = [0, 0, 1, 1, 2, 2, 3, 3];
        let p = [0, 1, 0, 1, 2, 3, 2, 3];
        let x = Array1::from(vec![1.3, 2.9, 0.4, 2.2, 5.1, 3.3, 4.6, 6.0]);
        let tw = TwoWayWithin::new(&u, &p, 4, 4).unwrap();
        let got = tw.residualize(&x).unwrap();
        let want = [0.05, -0.05, -0.05, 0.05, 0.8, -0.8, -0.8, 0.8];
        for i in 0..8 {
            assert!(
                (got[i] - want[i]).abs() < 1e-12,
                "row {i}: {} vs {}",
                got[i],
                want[i]
            );
        }
        assert_eq!(tw.fe_rank, 6);
        assert!(tw.warning.is_some());
    }

    #[test]
    fn test_control_group_parses_aliases_and_rejects_unknown() {
        assert_eq!(ControlGroup::choices().validate(), Ok(()));
        for s in ["Not-Yet-Treated", "notyet", " not yet "] {
            assert_eq!(
                s.parse::<ControlGroup>().unwrap(),
                ControlGroup::NotYet,
                "{s}"
            );
        }
        for s in ["NEVER", "never_treated"] {
            assert_eq!(
                s.parse::<ControlGroup>().unwrap(),
                ControlGroup::Never,
                "{s}"
            );
        }
        match "nevr".parse::<ControlGroup>() {
            Err(EconError::InvalidSpecification { message }) => assert_eq!(
                message,
                "unknown cgroup 'nevr'; expected one of: notyet, never"
            ),
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
        assert_eq!(ControlGroup::NotYet.to_string(), "Not-Yet-Treated");
    }

    #[test]
    fn test_etwfe_vcov_parses_aliases_and_rejects_unknown() {
        assert_eq!(EtwfeVcov::choices().validate(), Ok(()));
        for s in ["CR1", "Clustered"] {
            assert_eq!(s.parse::<EtwfeVcov>().unwrap(), EtwfeVcov::Cluster, "{s}");
        }
        for s in ["IID", "classical"] {
            assert_eq!(s.parse::<EtwfeVcov>().unwrap(), EtwfeVcov::Iid, "{s}");
        }
        assert_eq!(EtwfeVcov::Cluster.to_string(), "cluster");
        assert_eq!(EtwfeVcov::Iid.to_string(), "iid");
        match "hc1".parse::<EtwfeVcov>() {
            Err(EconError::InvalidSpecification { message }) => {
                assert_eq!(message, "unknown vcov 'hc1'; expected one of: cluster, iid")
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    /// One emfx table against `<prefix>_emfx_<type>_*` / `<prefix>_delta_<type>_*`.
    /// `key_shift`: p2a event keys are t - first_treat; R's anticipation case uses
    /// t - (first_treat - d).
    fn check_agg(
        ck: &mut Check<'_>,
        prefix: &str,
        r: &EtwfeResult,
        ty: EtwfeAggregation,
        key_shift: i64,
    ) {
        let a = r.aggregation(ty);
        let v = |f: fn(&AggregatedEffect) -> f64| a.iter().map(f).collect::<Vec<f64>>();
        let (n, t) = (r.n_obs, ty.to_string());
        ck.vector(
            &format!("{prefix}_emfx_{t}_estimate"),
            &v(|e| e.estimate),
            Tol::coef(n),
        )
        .vector(
            &format!("{prefix}_delta_{t}_se"),
            &v(|e| e.std_error),
            Tol::se(n),
        )
        // marginaleffects' numerical Jacobian: <= 4.7e-7 from the exact delta method here
        .vector(
            &format!("{prefix}_emfx_{t}_se"),
            &v(|e| e.std_error),
            Tol::iterative(1e-6),
        )
        .vector(
            &format!("{prefix}_delta_{t}_p"),
            &v(|e| e.p_value),
            Tol::p_value(n),
        )
        .vector(
            &format!("{prefix}_delta_{t}_n"),
            &v(|e| e.n_obs as f64),
            Tol::exact(),
        );
        if ty == EtwfeAggregation::Simple {
            ck.vector(
                &format!("{prefix}_delta_simple_ci95"),
                &[a[0].ci_lower, a[0].ci_upper],
                Tol::stat(n),
            );
        } else {
            let keys: Vec<f64> = a.iter().map(|e| (e.key + key_shift) as f64).collect();
            ck.vector(&format!("{prefix}_emfx_{t}_key"), &keys, Tol::exact());
        }
    }

    /// The staggered-case fits by prefix.
    fn staggered_fits(ds: &Dataset) -> Vec<(&'static str, EtwfeResult, i64)> {
        let mut iid = EtwfeConfig::default();
        iid.vcov = EtwfeVcov::Iid;
        let mut tref4 = EtwfeConfig::default();
        tref4.tref = Some(4);
        let mut ant1 = EtwfeConfig::default();
        ant1.anticipation = 1;
        vec![
            ("notyet", fit(ds, None, EtwfeConfig::default()), 0),
            ("notyet_iid", fit(ds, None, iid), 0),
            ("never", fit(ds, None, spec(ControlGroup::Never)), 0),
            ("tref4", fit(ds, None, tref4), 0),
            ("ant1", fit(ds, None, ant1), 1),
        ]
    }

    /// The unbalanced and all-treated fits by prefix.
    fn unbalanced_fits(ds: &Dataset) -> Vec<(&'static str, EtwfeResult, i64)> {
        let at = all_treated_panel();
        let mut gref6 = EtwfeConfig::default();
        gref6.gref = Some(6);
        let mut never8 = spec(ControlGroup::Never);
        never8.gref = Some(8);
        vec![
            ("unbal", fit(ds, None, EtwfeConfig::default()), 0),
            ("unbal_never", fit(ds, None, spec(ControlGroup::Never)), 0),
            ("alltreated", fit(&at, None, EtwfeConfig::default()), 0),
            ("alltreated_gref6", fit(&at, None, gref6), 0),
            ("alltreated_never_gref8", fit(&at, None, never8), 0),
        ]
    }

    fn check_agg_fits(
        case: &p2a_validation::Case,
        fits: &[(&'static str, EtwfeResult, i64)],
        prefixes: &[&str],
        ty: EtwfeAggregation,
    ) {
        let mut ck = Check::new(case);
        for (p, r, shift) in fits {
            if prefixes.contains(p) {
                check_agg(&mut ck, p, r, ty, *shift);
            }
        }
        ck.finish();
    }

    #[test]
    fn test_validate_etwfe_emfx_simple() {
        let (case, ds) = crate::test_support::load_case("etwfe_staggered");
        let fits = staggered_fits(&ds);
        let all = ["notyet", "notyet_iid", "never", "tref4", "ant1"];
        check_agg_fits(&case, &fits, &all, EtwfeAggregation::Simple);
        let (case_u, ds_u) = crate::test_support::load_case("etwfe_unbalanced");
        let fits_u = unbalanced_fits(&ds_u);
        let all_u = [
            "unbal",
            "unbal_never",
            "alltreated",
            "alltreated_gref6",
            "alltreated_never_gref8",
        ];
        check_agg_fits(&case_u, &fits_u, &all_u, EtwfeAggregation::Simple);
        for (_, r, _) in fits.iter().chain(&fits_u) {
            assert_eq!(r.att_simple, r.overall.estimate);
            assert_eq!(r.att_se, r.overall.std_error);
            assert_eq!(r.att_weighted, r.att_simple);
        }
    }

    #[test]
    fn test_validate_etwfe_emfx_event() {
        let (case, ds) = crate::test_support::load_case("etwfe_staggered");
        let fits = staggered_fits(&ds);
        let ty = EtwfeAggregation::Event;
        check_agg_fits(&case, &fits, &["notyet", "never", "tref4", "ant1"], ty);
        let (case_u, ds_u) = crate::test_support::load_case("etwfe_unbalanced");
        let fits_u = unbalanced_fits(&ds_u);
        let pu = [
            "unbal",
            "unbal_never",
            "alltreated",
            "alltreated_never_gref8",
        ];
        check_agg_fits(&case_u, &fits_u, &pu, ty);
        let never = &fits[2].1;
        let e = never.event_study.iter().find(|e| e.key == -1).unwrap();
        assert_eq!((e.estimate, e.n_effects, e.n_obs), (0.0, 0, 85));
        assert!(e.std_error.is_nan() && e.p_value.is_nan());
    }

    #[test]
    fn test_validate_etwfe_emfx_group() {
        let (case, ds) = crate::test_support::load_case("etwfe_staggered");
        let ty = EtwfeAggregation::Group;
        check_agg_fits(&case, &staggered_fits(&ds), &["notyet", "never"], ty);
        let (case_u, ds_u) = crate::test_support::load_case("etwfe_unbalanced");
        check_agg_fits(
            &case_u,
            &unbalanced_fits(&ds_u),
            &["unbal", "alltreated"],
            ty,
        );
    }

    #[test]
    fn test_validate_etwfe_emfx_calendar() {
        let (case, ds) = crate::test_support::load_case("etwfe_staggered");
        let ty = EtwfeAggregation::Calendar;
        check_agg_fits(&case, &staggered_fits(&ds), &["notyet", "never"], ty);
        let (case_u, ds_u) = crate::test_support::load_case("etwfe_unbalanced");
        check_agg_fits(
            &case_u,
            &unbalanced_fits(&ds_u),
            &["unbal", "alltreated"],
            ty,
        );
    }

    #[test]
    fn test_validate_etwfe_cgroup_changes_estimate() {
        let (case, ds) = crate::test_support::load_case("etwfe_staggered");
        let notyet = fit(&ds, None, EtwfeConfig::default());
        let never = fit(&ds, None, spec(ControlGroup::Never));
        Check::new(&case)
            .vector(
                "notyet_emfx_simple_estimate",
                &[notyet.att_simple],
                Tol::coef(960),
            )
            .vector(
                "never_emfx_simple_estimate",
                &[never.att_simple],
                Tol::coef(960),
            )
            .finish();
        assert!((never.att_simple - notyet.att_simple).abs() > 0.04);
    }

    #[test]
    fn test_validate_etwfe_confidence_level() {
        let (case, ds) = crate::test_support::load_case("etwfe_staggered");
        let mut c = EtwfeConfig::default();
        c.confidence_level = 0.90;
        let r = fit(&ds, None, c);
        Check::new(&case)
            .vector(
                "notyet_delta_simple_ci90",
                &[r.overall.ci_lower, r.overall.ci_upper],
                Tol::stat(960),
            )
            .finish();
        let z = z_critical(0.90).unwrap();
        for e in &r.event_study {
            assert!((e.ci_upper - (e.estimate + z * e.std_error)).abs() < 1e-12);
        }
        for level in [1.5, 0.0] {
            let mut c = EtwfeConfig::default();
            c.confidence_level = level;
            let e = run_etwfe(
                &ds,
                "y",
                "id",
                "year",
                "treat",
                "first_treat",
                None,
                Some(c),
            )
            .expect_err("level outside (0, 1)")
            .to_string();
            assert!(e.contains("confidence_level"), "{e}");
        }
    }

    #[test]
    fn test_etwfe_aggregation_parses_aliases_and_rejects_unknown() {
        assert_eq!(EtwfeAggregation::choices().validate(), Ok(()));
        let p = |s: &str| s.parse::<EtwfeAggregation>().unwrap();
        assert_eq!(p("Event-Study"), EtwfeAggregation::Event);
        assert_eq!(p("dynamic"), EtwfeAggregation::Event);
        assert_eq!(p("cohort"), EtwfeAggregation::Group);
        assert_eq!(p("overall"), EtwfeAggregation::Simple);
        assert_eq!(p("period"), EtwfeAggregation::Calendar);
        assert_eq!(EtwfeAggregation::Calendar.to_string(), "calendar");
        match "weekly".parse::<EtwfeAggregation>() {
            Err(EconError::InvalidSpecification { message }) => assert_eq!(
                message,
                "unknown aggregation 'weekly'; expected one of: simple, event, group, calendar"
            ),
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    fn with_f64(ds: &Dataset, col: &str, f: impl Fn(f64) -> f64) -> Dataset {
        let v: Vec<f64> = ds
            .df()
            .column(col)
            .unwrap()
            .cast(&DataType::Float64)
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .map(|x| f(x.unwrap()))
            .collect();
        let mut df = ds.df().clone();
        df.with_column(Column::new(col.into(), v)).unwrap();
        Dataset::new(df)
    }

    fn with_new_f64(ds: &Dataset, from: &str, to: &str, f: impl Fn(f64) -> f64) -> Dataset {
        let v: Vec<f64> = ds
            .df()
            .column(from)
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .map(|x| f(x.unwrap()))
            .collect();
        let mut df = ds.df().clone();
        df.with_column(Column::new(to.into(), v)).unwrap();
        Dataset::new(df)
    }

    const XC_ABSORBED: [&str; 4] = [
        "xc",
        "first_treat::4:xc",
        "first_treat::6:xc",
        "first_treat::8:xc",
    ];

    #[test]
    fn test_validate_etwfe_control() {
        let all = [
            EtwfeAggregation::Simple,
            EtwfeAggregation::Event,
            EtwfeAggregation::Group,
            EtwfeAggregation::Calendar,
        ];
        for (id, prefix) in [
            ("etwfe_staggered", "notyet_xc"),
            ("etwfe_unbalanced", "unbal_xc"),
        ] {
            let (case, ds) = crate::test_support::load_case(id);
            let r = fit(&ds, Some(&["xc"]), EtwfeConfig::default());
            assert!(last_fit_structured(), "{id}: fitted through the dense QR");
            let mut ck = Check::new(&case);
            check_terms(&mut ck, prefix, &r);
            for ty in all {
                check_agg(&mut ck, prefix, &r, ty, 0);
            }
            ck.finish();
            assert_eq!(r.dropped_terms, XC_ABSORBED);
            assert_eq!(r.controls, ["xc"]);
        }
    }

    #[test]
    fn test_etwfe_scale_invariant() {
        let ds = panel_csv("etwfe_panel");
        let x = Some(&["xc"][..]);
        let base = fit(&ds, x, EtwfeConfig::default());
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * b.abs().max(1e-300);
        let y_big = fit(&with_f64(&ds, "y", |v| v * 1e6), x, EtwfeConfig::default());
        assert_eq!(y_big.dropped_terms, base.dropped_terms);
        for (a, b) in y_big.terms.iter().zip(&base.terms) {
            assert_eq!(a.name, b.name);
            assert!(
                close(a.estimate, b.estimate * 1e6),
                "{}: {} vs {}",
                a.name,
                a.estimate,
                b.estimate
            );
            assert!(close(a.std_error, b.std_error * 1e6), "{}", a.name);
        }
        let x_small = fit(
            &with_f64(&ds, "xc", |v| v * 1e-6),
            x,
            EtwfeConfig::default(),
        );
        assert_eq!(x_small.dropped_terms, base.dropped_terms);
        for (a, b) in x_small.terms.iter().zip(&base.terms) {
            assert_eq!(a.name, b.name);
            let scale = if a.name.starts_with(".Dtreat") && !a.name.ends_with("_dm") {
                1.0
            } else {
                1e6
            };
            assert!(
                close(a.estimate, b.estimate * scale),
                "{}: {} vs {}",
                a.name,
                a.estimate,
                b.estimate
            );
        }
    }

    #[test]
    fn test_etwfe_collinear_controls_are_reported() {
        // evidence: fixest removes the xc2 = 2 xc terms as collinear and refits
        let ds = with_new_f64(&panel_csv("etwfe_panel"), "xc", "xc2", |v| 2.0 * v);
        let r1 = fit(&ds, Some(&["xc"]), EtwfeConfig::default());
        let r2 = fit(&ds, Some(&["xc", "xc2"]), EtwfeConfig::default());
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-10 * b.abs().max(1e-300);
        for t in &r1.terms {
            let u = r2.terms.iter().find(|u| u.name == t.name).unwrap();
            assert!(
                close(u.estimate, t.estimate),
                "{}: {} vs {}",
                t.name,
                u.estimate,
                t.estimate
            );
        }
        assert!(close(r2.att_simple, r1.att_simple));
        assert_eq!(
            r2.dropped_terms,
            [
                "xc",
                "first_treat::4:xc",
                "first_treat::6:xc",
                "first_treat::8:xc",
                "xc2",
                "first_treat::4:xc2",
                "first_treat::6:xc2",
                "first_treat::8:xc2"
            ]
        );
        let mut want: Vec<String> = [
            (4, 4),
            (4, 5),
            (4, 6),
            (4, 7),
            (4, 8),
            (6, 6),
            (6, 7),
            (6, 8),
            (8, 8),
        ]
        .iter()
        .map(|(g, t)| format!(".Dtreat:first_treat::{g}:year::{t}:xc2_dm"))
        .collect();
        want.extend((2..=8).map(|s| format!("year::{s}:xc2")));
        assert_eq!(r2.collinear_terms, want);
        assert_eq!(
            r2.warnings,
            vec![format!(
                "Removed because collinear with other terms: {}",
                r2.collinear_terms.join(", ")
            )]
        );
        assert!(notes_text(&r2).contains("Warning: Removed because collinear"));
    }

    #[test]
    fn test_etwfe_structured_fit_matches_dense_qr() {
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-10 * b.abs().max(1e-300);
        let collinear = with_new_f64(&panel_csv("etwfe_panel"), "xc", "xc2", |v| 2.0 * v);
        let runs: [(Dataset, Option<&[&str]>, EtwfeConfig); 3] = [
            (
                panel_csv("etwfe_panel_unbalanced"),
                Some(&["xc"]),
                EtwfeConfig::default(),
            ),
            (panel_csv("etwfe_panel"), None, spec(ControlGroup::Never)),
            // singular normal equations: the xc2 columns leave at the Gram level
            (collinear, Some(&["xc", "xc2"]), EtwfeConfig::default()),
        ];
        for (k, (ds, x, cfg)) in runs.into_iter().enumerate() {
            let name = format!("run {k}");
            for vcov in [EtwfeVcov::Cluster, EtwfeVcov::Iid] {
                let mut c = cfg.clone();
                c.vcov = vcov;
                let structured = fit(&ds, x, c.clone());
                assert!(last_fit_structured(), "{name}: not the structured path");
                FORCE_DENSE_FIT.with(|f| f.set(true));
                let dense = fit(&ds, x, c);
                FORCE_DENSE_FIT.with(|f| f.set(false));
                assert!(!last_fit_structured(), "{name}: not the dense path");
                assert_eq!(structured.collinear_terms, dense.collinear_terms, "{name}");
                assert_eq!(structured.dropped_terms, dense.dropped_terms, "{name}");
                assert_eq!(structured.terms.len(), dense.terms.len());
                for (a, b) in structured.terms.iter().zip(&dense.terms) {
                    assert_eq!(a.name, b.name);
                    assert!(
                        close(a.estimate, b.estimate),
                        "{name} {}: {} vs {}",
                        a.name,
                        a.estimate,
                        b.estimate
                    );
                    assert!(
                        close(a.std_error, b.std_error),
                        "{name} {}: {} vs {}",
                        a.name,
                        a.std_error,
                        b.std_error
                    );
                }
                assert!(close(structured.overall.std_error, dense.overall.std_error));
            }
        }
    }

    #[test]
    fn test_etwfe_display_reports_what_was_estimated() {
        let ds = panel_csv("etwfe_panel");
        let text = fit(&ds, Some(&["xc"]), spec(ControlGroup::Never)).to_string();
        for line in [
            "Control group: Never-Treated (cgroup = never); reference group: never treated (35 units)",
            "Reference period (tref): 1",
            "Standard errors: clustered by unit (120 clusters; CR1 as fixest), t with 119 df",
            "Absorbed by the unit and period effects: xc, first_treat::4:xc, first_treat::6:xc, first_treat::8:xc",
            "ATT (emfx simple",
            "Event Study (emfx event",
            "(reference)",
            "Cohort Averages (emfx group)",
            "Period Averages (emfx calendar)",
        ] {
            assert!(text.contains(line), "missing {line:?} in\n{text}");
        }
        assert!(!text.contains("Simple Average") && !text.contains("Weighted Average"));
        let mut iid = EtwfeConfig::default();
        iid.vcov = EtwfeVcov::Iid;
        let text = fit(&ds, None, iid).to_string();
        assert!(
            text.contains("Standard errors: iid, t with 824 df"),
            "{text}"
        );
        let text = fit(&all_treated_panel(), None, EtwfeConfig::default()).to_string();
        assert!(
            text.contains("reference group: cohort 8 (periods from 8 dropped)"),
            "{text}"
        );
        let mut ant = EtwfeConfig::default();
        ant.anticipation = 1;
        let text = fit(&ds, None, ant).to_string();
        assert!(
            text.contains("Anticipation: 1 period(s); cohort g is treated from g - 1"),
            "{text}"
        );
    }

    /// The comprehensive benchmark's staggered panel (cohorts of one or two units) with one
    /// noisy control `x`.
    fn small_cohort_panel(n_units: usize, n_periods: usize) -> Dataset {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        let (mut id, mut year, mut y, mut ft, mut treat, mut x) =
            (vec![], vec![], vec![], vec![], vec![], vec![]);
        let half = n_periods as i64 / 2;
        for u in 0..n_units {
            let g = if u < n_units / 3 {
                0
            } else {
                n_periods as i64 / 3 + u as i64 % half + 1
            };
            for t in 0..n_periods as i64 {
                let d = g > 0 && t >= g;
                let xv: f64 = rng.gen_range(-1.0..1.0);
                id.push(u as i64);
                year.push(t);
                ft.push(g);
                treat.push(i64::from(d));
                x.push(xv);
                let e: f64 = rng.gen_range(-0.5..0.5);
                y.push(0.1 * u as f64 + 0.05 * t as f64 + if d { 2.0 } else { 0.0 } + 0.3 * xv + e);
            }
        }
        Dataset::new(
            df! {"id" => id, "year" => year, "y" => y, "first_treat" => ft, "treat" => treat,
            "x" => x}
            .unwrap(),
        )
    }

    #[test]
    fn test_etwfe_one_unit_cohort_controls_leave_at_the_gram_level() {
        // A one-unit cohort's cell:x_dm column is proportional to its cell dummy: the
        // normal equations are singular, the Gram-level check removes those columns and the
        // structured path fits; the dense QR removes the same columns.
        let ds = small_cohort_panel(24, 24);
        let x = Some(&["x"][..]);
        let r = fit(&ds, x, EtwfeConfig::default());
        assert!(last_fit_structured(), "fitted through the dense QR");
        assert!(!r.collinear_terms.is_empty());
        assert!(
            r.collinear_terms.iter().all(|t| t.ends_with(":x_dm")),
            "{:?}",
            r.collinear_terms
        );
        FORCE_DENSE_FIT.with(|f| f.set(true));
        let dense = fit(&ds, x, EtwfeConfig::default());
        FORCE_DENSE_FIT.with(|f| f.set(false));
        assert!(!last_fit_structured());
        assert_eq!(r.collinear_terms, dense.collinear_terms);
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * b.abs().max(1e-12);
        for (a, b) in r.terms.iter().zip(&dense.terms) {
            assert_eq!(a.name, b.name);
            assert!(
                close(a.estimate, b.estimate),
                "{}: {} vs {}",
                a.name,
                a.estimate,
                b.estimate
            );
            assert!(
                close(a.std_error, b.std_error),
                "{}: {} vs {}",
                a.name,
                a.std_error,
                b.std_error
            );
        }
        assert!(close(r.att_simple, dense.att_simple) && close(r.att_se, dense.att_se));
    }

    #[test]
    fn test_etwfe_anticipation_bound_is_named() {
        // etwfe_panel: periods 1..8, first treated cohort 4
        let ds = panel_csv("etwfe_panel");
        let err = |a: usize| {
            let mut c = EtwfeConfig::default();
            c.anticipation = a;
            run_etwfe(
                &ds,
                "y",
                "id",
                "year",
                "treat",
                "first_treat",
                None,
                Some(c),
            )
        };
        assert!(err(2).is_ok());
        for a in [3, 7, usize::MAX] {
            match err(a) {
                Err(EconError::InvalidSpecification { message }) => assert_eq!(
                    message,
                    format!(
                        "anticipation must be below 3 (first treated cohort 4 minus first \
                         period 1), got {a}"
                    )
                ),
                other => panic!("expected InvalidSpecification, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_etwfe_unidentified_suggestion_follows_cgroup() {
        let ds = all_treated_panel();
        // cgroup = never, gref = 8, year 2 holds cohort 6 only
        let mut never = spec(ControlGroup::Never);
        never.gref = Some(8);
        let d = drop_rows(&ds, |g, t| (g == 4 || g == 8) && t == 2);
        // cgroup = notyet, gref = 6: year 5 holds cohort 4 only
        let mut notyet = EtwfeConfig::default();
        notyet.gref = Some(6);
        let d2 = drop_rows(&ds, |g, t| (g == 6 || g == 8) && t == 5);
        for (data, c, suggests_notyet) in [(d, never, true), (d2, notyet, false)] {
            match run_etwfe(
                &data,
                "y",
                "id",
                "year",
                "treat",
                "first_treat",
                None,
                Some(c),
            ) {
                Err(EconError::SingularMatrix {
                    context,
                    suggestion,
                }) => {
                    assert!(context.contains("not identified"), "{context}");
                    assert_eq!(
                        suggestion.contains("cgroup = notyet"),
                        suggests_notyet,
                        "{suggestion}"
                    );
                    assert!(suggestion.contains("set gref"), "{suggestion}");
                }
                other => panic!("expected SingularMatrix, got {other:?}"),
            }
        }
    }
}
