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
//! Under not-yet-treated parallel trends the comparison group of ATT(g,t) is the
//! never-treated units plus the units first treated after both t and the base
//! period g - 1, excluding cohort g (the rule of `did::att_gt`).
//!
//! ## Covariates
//!
//! With covariates each cell is a two-period panel of its treated and comparison
//! units with the outcome change dY = Y_t - Y_base and the covariates of the
//! earlier period min(t, base) (did's rule), and ATT(g,t) is one of the DRDID
//! panel estimators (Sant'Anna & Zhao 2020), as `did::att_gt` calls them:
//! - outcome regression (`reg_did_panel`): OLS of dY on [1, X] over the comparison
//!   units, ATT = mean(dY - X b | treated);
//! - IPW (`std_ipw_did_panel`): a logit propensity score p(X), capped at 1 - 1e-6,
//!   and normalised weights p/(1 - p) for the comparison units;
//! - doubly robust (`drdid_panel`): the outcome-regression residuals with the IPW
//!   weights.
//!
//! IPW and DR give comparison units with p >= 0.995 weight 0 (DRDID's
//! `trim.level`). A cell whose largest propensity score is >= 0.999 (did's overlap
//! guard) or whose covariates are collinear is not computed (a warning; did reports
//! NA). The reference cell t = base is 0 with a zero influence function. Each cell
//! is fitted once; its ATT and influence function serve the point estimate, the
//! analytic standard errors and the bootstrap. Without covariates the three methods
//! coincide (the unconditional difference in mean changes).
//!
//! ## Anticipation
//!
//! With `anticipation = d` (did's option) the base period is g - 1 - d and
//! not-yet-treated comparison units are first treated after max(t, base) + d.
//! Post-treatment cells are t >= g, as in did, so the d anticipation cells before
//! treatment are pre-treatment cells and enter the pre-trend test. A cohort whose
//! base period is not observed is not computed (a warning; did drops it). When no
//! post-treatment cell can be computed the call is an error; a cohort with only
//! pre-treatment cells is left out of the group aggregation, as `did::aggte` does.
//!
//! ## Aggregation Schemes
//!
//! Group-time ATTs are aggregated as `did::aggte` does, with the weights
//! P(G = g), the share of units in cohort g:
//! - **Event Study** (`type = "dynamic"`): for each e = t - g, the average of the
//!   cells at e weighted by P(G = g)
//! - **Group Average** (`type = "group"`): for each cohort, the mean of its
//!   post-treatment cells
//! - **Overall ATT** (`type = "simple"`): the average of all post-treatment cells
//!   weighted by P(G = g)
//!
//! ## Standard errors
//!
//! - `bootstrap = 0`: analytic standard errors from unit-level influence functions,
//!   as `did::att_gt` with `bstrap = FALSE`, with or without covariates. The
//!   event-study and overall influence functions include the
//!   term for the estimated weights (did's `wif`), and the pre-trend test is the
//!   Wald statistic `a' V^+ a` with the influence-function covariance of the
//!   pre-treatment cells on chi-squared(rank V) (generalized inverse, df = rank, if
//!   V is singular). The chi-squared reference is asymptotic and over-rejects on
//!   small panels: in a null simulation with nine pre-treatment cells it rejected
//!   about 19% of panels at the 5% level with 90 units (cohorts as small as 15
//!   units) and about 6% with 900. For small panels use the bootstrap test below,
//!   whose F reference accounts for the estimated covariance.
//! - `bootstrap >= 50`: a unit bootstrap; values 1 to 49 are refused. Every
//!   replicate resamples units, computes all cells, the aggregates (with the
//!   replicate's own cohort shares) and the pre-trend vector; the reported SEs are
//!   the SDs of those replicate statistics. The pre-trend statistic `W = a' V^+ a`
//!   uses the covariance V of the m replicates that computed every pre-treatment
//!   cell, so it is Hotelling's T^2: with r = rank V, `F = (m - r) W / (r (m - 1))`
//!   is referred to F(r, m - r), which accounts for the Monte Carlo error in V (a
//!   chi-squared(r) reference over-rejects at small B). With m <= r it is not
//!   computed. A replicate in which a needed cell cannot be computed (no treated
//!   or comparison unit drawn, or a drawn unit without its base period) is dropped
//!   for that statistic, and one warning says how many replicates were affected
//!   and the fewest any SE used. Without covariates a replicate cell is the
//!   difference in mean changes of the drawn units; with covariates it is the
//!   multiplier bootstrap of the cell's influence function with the same unit
//!   draw counts, `att + sum (count - 1) IF / n_units` (did's default bootstrap is
//!   a multiplier bootstrap too): it estimates the analytic SE up to Monte Carlo
//!   error without re-fitting the cells. The cost is O(B (units + table entries +
//!   cells)) plus the aggregates.
//!
//! On both paths the joint pre-trend test over k pre-treatment cells costs
//! O(n k^2 + k^3), with n the units (analytic) or the replicates used (bootstrap):
//! the covariance is accumulated over blocks of rows with a budget check per block.
//! It is computed for at most 2000 pre-treatment cells; above that it is reported
//! as not computed.
//!
//! # References
//!
//! - Callaway, B., & Sant'Anna, P.H.C. (2021). Difference-in-Differences with
//!   Multiple Time Periods. *Journal of Econometrics*, 225(2), 200-230.
//!   https://doi.org/10.1016/j.jeconom.2020.12.001
//!
//! - Sant'Anna, P.H.C., & Zhao, J. (2020). Doubly robust difference-in-differences
//!   estimators. *Journal of Econometrics*, 219(1), 101-122.
//!   https://doi.org/10.1016/j.jeconom.2020.06.003
//!
//! - R packages: `did` (https://bcallaway11.github.io/did/), `DRDID`
//!
//! - Stata package: `csdid` (https://github.com/friosavila/csdid)

use ndarray::{Array1, Array2, s};
use rand::SeedableRng;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, MissingSummary, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::eig_symmetric;
use crate::resource;
use crate::traits::convergence::{ResultNotes, notes_text};
use crate::traits::estimator::{SignificanceLevel, chi_squared_p_value};
use statrs::distribution::{ContinuousCDF, FisherSnedecor, Normal};

/// Fewest bootstrap replicates accepted (`bootstrap = 0` is the analytic path).
const MIN_BOOTSTRAP: usize = 50;

/// A standard error at or below this is reported as NaN (R `NA`), as `did::att_gt`
/// and `did::aggte` do (`se <= sqrt(.Machine$double.eps) * 10`): the reference
/// cell ATT(g, base) and its event time have a zero influence function.
const SE_NA_LIMIT: f64 = 1.4901161193847656e-7;

/// `se`, or NaN when it is at or below [`SE_NA_LIMIT`] (did's rule).
fn did_se(se: f64) -> f64 {
    if se <= SE_NA_LIMIT { f64::NAN } else { se }
}

/// The base period `g + offset` as text: `g - 3`, `g + 1`, or `g` for 0.
pub(crate) fn base_period_label(offset: i64) -> String {
    match offset {
        0 => "g".to_string(),
        o if o < 0 => format!("g - {}", o.unsigned_abs()),
        o => format!("g + {o}"),
    }
}

/// The base period `g + base_period - anticipation` the estimator uses.
pub(crate) fn base_period_offset(base_period: i32, anticipation: usize) -> i64 {
    i64::from(base_period).saturating_sub(i64::try_from(anticipation).unwrap_or(i64::MAX))
}

/// A number at 4 decimals, or `NA` for NaN (R's print of a missing SE).
fn num4(v: f64) -> String {
    if v.is_nan() {
        "NA".to_string()
    } else {
        format!("{v:.4}")
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Comparison group strategy for Callaway-Sant'Anna estimator.
///
/// Parsed from strings by `FromStr` (case- and separator-insensitive; unknown
/// values are an error listing the accepted ones).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ComparisonGroup {
    /// Use never-treated units only (G_i = ∞)
    /// More restrictive but cleaner identification
    #[default]
    NeverTreated,
    /// Use never-treated units and units first treated after both t and the base
    /// period, other than cohort g (`did`'s `control_group = "notyettreated"`).
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

crate::option_enum!(
    ComparisonGroup,
    crate::options::Choices {
        option: "comparison_group",
        values: &[
            (
                &["never_treated", "nevertreated", "never"],
                ComparisonGroup::NeverTreated
            ),
            (
                &["not_yet_treated", "notyettreated", "notyet", "nyt"],
                ComparisonGroup::NotYetTreated
            ),
        ],
    },
    no_display
);

/// Estimation method for group-time ATTs (DRDID's panel estimators, as
/// `did::att_gt`'s `est_method`; with no covariates all three are the difference in
/// mean outcome changes).
///
/// Parsed from strings by `FromStr` (case- and separator-insensitive; unknown
/// values are an error listing the accepted ones).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AttEstimationMethod {
    /// Outcome regression (DRDID `reg_did_panel`, did's `est_method = "reg"`): the
    /// outcome change regressed on the covariates over the comparison units
    #[default]
    OutcomeRegression,
    /// Normalised inverse probability weighting (DRDID `std_ipw_did_panel`, `"ipw"`)
    IPW,
    /// Doubly robust (DRDID `drdid_panel`, `"dr"`)
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

crate::option_enum!(
    AttEstimationMethod,
    crate::options::Choices {
        option: "estimation_method",
        values: &[
            (
                &["outcome_regression", "or", "reg", "regression"],
                AttEstimationMethod::OutcomeRegression
            ),
            (
                &["ipw", "std_ipw", "inverse_probability_weighting"],
                AttEstimationMethod::IPW
            ),
            (
                &["doubly_robust", "dr", "aipw", "doublyrobust", "drdid"],
                AttEstimationMethod::DoublyRobust
            ),
        ],
    },
    no_display
);

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
    /// Periods before treatment in which units may anticipate it (did's
    /// `anticipation`, default 0). The base period is g + base_period - anticipation
    /// (g - 1 - anticipation by default); not-yet-treated comparison units are first
    /// treated after max(t, base) + anticipation; post-treatment cells are t >= g, as
    /// in did, so the anticipation cells are pre-treatment cells. Must be smaller than
    /// the number of periods.
    pub anticipation: usize,
    /// Standard errors: 0 = analytic influence-function standard errors (did's
    /// `bstrap = FALSE`), with or without covariates; otherwise the number of
    /// bootstrap replicates, at least 50 (1..=49 is an `InvalidSpecification`
    /// error): a unit-resampling bootstrap without covariates, a multiplier bootstrap
    /// of the influence functions with covariates (module docs).
    pub bootstrap: usize,
    /// Confidence level of every interval (cells and aggregates); must lie in
    /// (0, 1). The normal quantile comes from statrs.
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
    /// Standard error; NaN (JSON `null`) for the reference cell (t = g - 1 -
    /// anticipation), whose ATT is 0 by construction, as did reports NA (did's
    /// rule: an SE at or below `sqrt(.Machine$double.eps) * 10` is NA)
    pub std_error: f64,
    /// Confidence interval lower bound (NaN when the SE is)
    pub ci_lower: f64,
    /// Confidence interval upper bound (NaN when the SE is)
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
    /// Standard error; NaN at the reference event time (did reports NA for an SE at
    /// or below `sqrt(.Machine$double.eps) * 10`)
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
///
/// With analytic standard errors the statistic `W = a' V^+ a` uses the
/// influence-function covariance and the p-value is chi-squared(`df`). With a
/// bootstrap covariance estimated from m replicates, `W` is Hotelling's T^2 and the
/// p-value is from `F = (m - df) W / (df (m - 1))` on F(`df`, `df2` = m - df), which
/// accounts for the Monte Carlo error in V.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreTrendTest {
    /// Wald statistic `a' V^+ a` (Hotelling's T^2 with a bootstrap covariance)
    pub chi2: f64,
    /// Degrees of freedom: the rank of V (the numerator df of the F reference)
    pub df: usize,
    /// p-value: chi-squared(`df`), or F(`df`, `df2`) with a bootstrap covariance
    pub p_value: f64,
    /// The F statistic of a bootstrap covariance; `None` with chi-squared(`df`)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub f_stat: Option<f64>,
    /// Denominator df (replicates used minus `df`) of a bootstrap covariance
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub df2: Option<usize>,
    /// Pre-treatment ATT estimates (should be ~0 under parallel trends)
    pub pre_atts: Vec<f64>,
    /// How the covariance of the pre-treatment ATT(g,t) was obtained, e.g.
    /// "Wald, influence-function covariance of 9 pre-treatment ATT(g,t)".
    #[serde(default)]
    pub method: String,
}

impl PreTrendTest {
    /// The most pre-treatment cells the joint test is computed for; with more it is
    /// reported as not computed (NaN p-value, `pre_atts` still filled).
    pub const MAX_CELLS: usize = MAX_PRETREND_CELLS;

    /// The statistic and its reference distribution as printed:
    /// "Chi-squared(9) = 7.26", or "T^2 = 12.31, F(9, 191) = 1.2967" with a
    /// bootstrap covariance.
    pub fn statistic_text(&self) -> String {
        match (self.f_stat, self.df2) {
            (Some(f), Some(df2)) => {
                format!("T^2 = {:.2}, F({}, {df2}) = {f:.4}", self.chi2, self.df)
            }
            _ => format!("Chi-squared({}) = {:.2}", self.df, self.chi2),
        }
    }
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
    /// Covariate names (values taken from the earlier of the two periods of each
    /// cell, as did); empty for the unconditional estimator.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub covariates: Vec<String>,
}

impl ResultNotes for StaggeredDidResult {
    /// The warnings, without the missing-value note (it has its own Display line
    /// and the `missing` field).
    fn notes_warnings(&self) -> Vec<String> {
        let note = self.missing.note();
        self.warnings
            .iter()
            .filter(|w| Some(w.as_str()) != note.as_deref())
            .cloned()
            .collect()
    }
}

impl fmt::Display for StaggeredDidResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Callaway-Sant'Anna Staggered DiD Estimation")?;
        writeln!(f, "============================================")?;
        writeln!(f)?;
        writeln!(f, "Settings:")?;
        writeln!(f, "  Comparison Group:   {}", self.config.comparison_group)?;
        writeln!(f, "  Estimation Method:  {}", self.config.estimation_method)?;
        if self.covariates.is_empty() {
            writeln!(
                f,
                "  Covariates:         none (OR, IPW and DR coincide: unconditional difference in mean changes)"
            )?;
        } else {
            writeln!(
                f,
                "  Covariates:         {} (values from the earlier of the two periods, as did)",
                self.covariates.join(", ")
            )?;
        }
        writeln!(f, "  Base Period:        {}", self.config.base_period)?;
        let d = self.config.anticipation;
        if d > 0 {
            writeln!(
                f,
                "  Anticipation:       {d} period(s) (base period {}; not-yet-treated comparisons first treated after max(t, base) + {d})",
                base_period_label(base_period_offset(self.config.base_period, d))
            )?;
        } else {
            writeln!(f, "  Anticipation:       none")?;
        }
        if self.config.bootstrap == 0 {
            writeln!(
                f,
                "  Standard Errors:    analytic (influence functions, as did with bstrap = FALSE)"
            )?;
        } else if self.covariates.is_empty() {
            writeln!(
                f,
                "  Standard Errors:    unit bootstrap, {} replicates",
                self.config.bootstrap
            )?;
        } else {
            writeln!(
                f,
                "  Standard Errors:    multiplier bootstrap of the influence functions, {} replicates (with covariates, as did::att_gt's default)",
                self.config.bootstrap
            )?;
        }
        let level = level_label(self.config.confidence_level);
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
            "  ATT = {:.4} (SE = {}, {level}% CI [{}, {}])",
            self.overall_att.att,
            num4(self.overall_att.std_error),
            num4(self.overall_att.ci_lower),
            num4(self.overall_att.ci_upper)
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
            "e",
            "ATT",
            "Std.Err",
            format!("{level}% CI")
        )?;
        writeln!(f, "{}", "-".repeat(60))?;
        for eff in &self.event_study {
            let sig = SignificanceLevel::from_p_value(eff.p_value);
            writeln!(
                f,
                "{:>8} {:>12.4} {:>12} [{:>8}, {:>8}]{}",
                eff.key,
                eff.att,
                num4(eff.std_error),
                num4(eff.ci_lower),
                num4(eff.ci_upper),
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
            if pretrend.p_value.is_finite() {
                writeln!(f, "  Method: {}", pretrend.method)?;
                writeln!(
                    f,
                    "  {}, p-value = {:.4}",
                    pretrend.statistic_text(),
                    pretrend.p_value
                )?;
                if pretrend.p_value < 0.05 {
                    writeln!(f, "  WARNING: Pre-trends may be violated")?;
                } else {
                    writeln!(f, "  Parallel trends assumption not rejected")?;
                }
            } else {
                writeln!(f, "  Not computed: {}", pretrend.method)?;
            }
        }

        // The notes, then the missing-value note once, as the last line.
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

/// A confidence level as a percentage without trailing zeros: 95, 90, 97.5.
fn level_label(level: f64) -> String {
    format!("{}", (level * 100.0 * 1e6).round() / 1e6)
}

/// Refuse an infinite first-treatment period before the columns are extracted, with
/// an error that names the column and the never-treated code `never` ("0", "0 or
/// negative"). Coding never-treated units as infinity is common, and the
/// time-index error alone would not say what to use instead. A missing column or a
/// column that is not floating point is left to the extractor.
pub(crate) fn refuse_infinite_first_treat(
    dataset: &Dataset,
    name: &str,
    never: &str,
) -> EconResult<()> {
    let Ok(col) = dataset.df().column(name) else {
        return Ok(());
    };
    if !col.dtype().is_float() {
        return Ok(());
    }
    let polars_err = |e: polars::prelude::PolarsError| {
        EconError::Computation(format!("reading column '{name}': {e}"))
    };
    let cast = col
        .cast(&polars::prelude::DataType::Float64)
        .map_err(polars_err)?;
    let infinite = cast
        .f64()
        .map_err(polars_err)?
        .iter()
        .enumerate()
        .find_map(|(i, v)| v.filter(|x| x.is_infinite()).map(|x| (i, x)));
    match infinite {
        Some((row, x)) => Err(EconError::InvalidSpecification {
            message: format!(
                "Column '{name}' (the first treatment period) is {x} in row {row} (0-based); \
                 never-treated units must be coded {never}, not infinity."
            ),
        }),
        None => Ok(()),
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
/// fractional periods and text outcomes are refused with an error naming the column;
/// an infinite first treatment period is refused with one that says never-treated
/// units are coded 0 or negative.
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
/// - NotYetTreated: never-treated units and units with G > max(t, base) + anticipation
///   and G != g (not yet treated by time t nor by the base period; `did::att_gt`'s rule)
///
/// The base period is g + base_period - anticipation (g - 1 by default). With
/// covariates every cell is DRDID's estimator for `estimation_method` with the
/// covariates of the earlier of its two periods (module docs).
///
/// # Standard errors
///
/// With `bootstrap = 0`, standard errors are analytic: did's influence functions
/// of every cell, aggregated with the weight-estimation term, and the pre-trend
/// test uses their joint covariance (module docs), with or without covariates.
///
/// With `bootstrap >= 50`, standard errors come from a nonparametric bootstrap
/// that resamples units. Each cell's outcome changes are tabulated once (at
/// most `n_cohorts x n` entries), so the bootstrap costs
/// O(bootstrap x (units + entries)) and its replicate-by-cell draws are one
/// fallibly allocated matrix: an impossible replication count is a
/// `ResourceLimit` error. Every (g, t) cell and every replicate honours the
/// cooperative cancellation budget (`p2a_core::resource`). The SE of every cell
/// and aggregate is the SD of its replicate statistics, each aggregate recomputed
/// with the replicate's cohort counts, and the pre-trend test is Hotelling's T^2
/// with the replicate covariance, on F(r, m - r) (module docs). A replicate that
/// draws a unit without a base-period row has no
/// estimate for that cell (unbalanced panels, CAUS-85) and is dropped for the
/// statistics that need the cell; a warning counts them and, when a statistic
/// keeps fewer than 50 replicates, recommends `bootstrap = 0`. With covariates the
/// replicates are a multiplier bootstrap of the cells' influence functions with
/// the same unit draws (module docs).
///
/// # Errors
///
/// `InvalidSpecification` for `anticipation` not smaller than the number of
/// periods; `InsufficientData` when no group-time ATT, or no post-treatment one,
/// can be computed (the context gives the first reason).
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
    let covariates = covariates.filter(|c| !c.is_empty());

    if (1..MIN_BOOTSTRAP).contains(&config.bootstrap) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "bootstrap must be 0 for analytic influence-function standard errors or at \
                 least {MIN_BOOTSTRAP} replicates; got {}",
                config.bootstrap
            ),
        });
    }
    if !(config.confidence_level > 0.0 && config.confidence_level < 1.0) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "confidence_level must lie strictly between 0 and 1; got {}",
                config.confidence_level
            ),
        });
    }
    refuse_infinite_first_treat(dataset, treatment_time, "0 or negative")?;
    // One request over every column the estimator uses: rows with a missing
    // value in any of them are dropped jointly (reported in `missing`).
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
    // The covariates without an intercept column: the cell estimators add one.
    let x_cov: Option<Array2<f64>> = covariates
        .map(|c| cols.matrix(c, false).map(|m| m.data))
        .transpose()?;
    let covariate_names: Vec<String> = covariates
        .map(|c| c.iter().map(|s| s.to_string()).collect())
        .unwrap_or_default();

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

    // did's `anticipation`: fewer than the number of periods
    let delta = i64::try_from(config.anticipation)
        .ok()
        .filter(|&d| (d as u128) < periods.len() as u128)
        .ok_or_else(|| EconError::InvalidSpecification {
            message: format!(
                "anticipation must be smaller than the number of periods ({}); got {}",
                periods.len(),
                config.anticipation
            ),
        })?;
    let base_offset = i64::from(config.base_period) - delta;

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
    let z = std_normal().inverse_cdf(1.0 - (1.0 - config.confidence_level) / 2.0);

    // Compute group-time ATTs
    let mut group_time_atts = Vec::new();
    // The first reason a cell was not computed, for the error when none is left.
    let mut first_skip: Option<String> = None;

    for &g in &cohorts {
        // Base period g + base_period - anticipation (g - 1 - anticipation by default)
        let base_t = match g.checked_add(base_offset) {
            Some(b) if periods.binary_search(&b).is_ok() => b,
            other => {
                let shown = other.map_or_else(|| "out of range".to_string(), |b| b.to_string());
                warnings.push(format!(
                    "cohort {g}: base period {shown} (g + base_period - anticipation) is not \
                     observed; its ATT(g,t) are not computed (did drops such cohorts)"
                ));
                continue;
            }
        };

        for &t in &periods {
            resource::checkpoint()?;

            let att_result = compute_group_time_att(
                &y,
                &g_col,
                &t_col,
                &unit_ids,
                &panel_map,
                (g, t, base_t),
                config.comparison_group,
                delta,
            );

            match att_result {
                Ok(att) => {
                    if att.n_treated >= config.min_obs_per_cell
                        && att.n_comparison >= config.min_obs_per_cell
                    {
                        group_time_atts.push(att);
                    } else {
                        push_skip(
                            &mut first_skip,
                            &mut warnings,
                            format!(
                                "Skipped ATT({},{}) due to small cell size: n_treated={}, n_comparison={}",
                                g, t, att.n_treated, att.n_comparison
                            ),
                        );
                    }
                }
                Err(e) => {
                    push_skip(
                        &mut first_skip,
                        &mut warnings,
                        format!("Could not compute ATT({g},{t}): {e}"),
                    );
                }
            }
        }
    }

    let none_left = |first_skip: &Option<String>| EconError::InsufficientData {
        required: 1,
        provided: 0,
        context: match first_skip {
            Some(r) => {
                format!("No valid group-time ATT estimates could be computed; first reason: {r}")
            }
            None => "No valid group-time ATT estimates could be computed".to_string(),
        },
    };
    if group_time_atts.is_empty() {
        return Err(none_left(&first_skip));
    }

    // Each cell's outcome changes are tabulated once: the influence functions
    // and every bootstrap replicate are computed from this table.
    let cell_list = |atts: &[GroupTimeATT]| -> Vec<(i64, i64, i64)> {
        atts.iter()
            .map(|c| (c.group, c.time, c.group + base_offset))
            .collect()
    };
    let mut cells = cell_list(&group_time_atts);
    let mut table = build_boot_table(
        &y,
        &g_col,
        &t_col,
        &unit_ids,
        &panel_map,
        &cells,
        config.comparison_group,
        delta,
    )?;

    // With covariates every cell is DRDID's estimator, fitted once: its ATT and
    // influence function serve the point estimate, the analytic SEs and the
    // multiplier bootstrap.
    if let Some(x) = &x_cov {
        let fits = covariate_fits(&table, &cells, x, config.estimation_method)?;
        let mut kept_atts = Vec::with_capacity(fits.len());
        let mut kept_fits = Vec::with_capacity(fits.len());
        for (mut gta, fit) in group_time_atts.into_iter().zip(fits) {
            match fit {
                Ok(fit) => {
                    for note in &fit.notes {
                        warnings.push(format!("ATT({},{}): {note}", gta.group, gta.time));
                    }
                    gta.att = fit.att;
                    kept_atts.push(gta);
                    kept_fits.push(fit);
                }
                Err(e) => push_skip(
                    &mut first_skip,
                    &mut warnings,
                    format!("Could not compute ATT({},{}): {e}", gta.group, gta.time),
                ),
            }
        }
        group_time_atts = kept_atts;
        if group_time_atts.is_empty() {
            return Err(none_left(&first_skip));
        }
        if group_time_atts.len() < cells.len() {
            // the table of the surviving cells (each cell's entries are unchanged)
            cells = cell_list(&group_time_atts);
            table = build_boot_table(
                &y,
                &g_col,
                &t_col,
                &unit_ids,
                &panel_map,
                &cells,
                config.comparison_group,
                delta,
            )?;
        }
        table.linear = Some(Linearised::new(&table, &kept_fits)?);
    }

    // No post-treatment cell: nothing to aggregate (CAUS-109)
    if !group_time_atts.iter().any(|c| c.post_treatment) {
        let last = periods.last().copied().unwrap_or_default();
        let late: Vec<i64> = cohorts.iter().copied().filter(|&g| g > last).collect();
        let mut context = "no post-treatment ATT(g,t) could be computed, so there is no treatment \
                           effect to aggregate"
            .to_string();
        if !late.is_empty() {
            context.push_str(&format!(
                "; cohorts first treated after the last observed period {last}: {late:?} \
                 (did::att_gt treats such units as never treated)"
            ));
        }
        if let Some(r) = &first_skip {
            context.push_str(&format!("; first reason a cell was not computed: {r}"));
        }
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context,
        });
    }
    // Cohorts whose cells are all pre-treatment are left out of the group
    // aggregation, as did::aggte drops them.
    let cell_cohorts: BTreeSet<i64> = group_time_atts.iter().map(|c| c.group).collect();
    for g in cell_cohorts {
        if !group_time_atts
            .iter()
            .any(|c| c.group == g && c.post_treatment)
        {
            warnings.push(format!(
                "cohort {g} has no post-treatment ATT(g,t) and is left out of the group \
                 aggregation (as did::aggte)"
            ));
        }
    }

    let n_cohort = table.cohort_sizes();
    let (aggregates, mut pretrend_test) = if config.bootstrap == 0 {
        // Analytic standard errors: did's influence functions (bstrap = FALSE).
        let inf = cell_influence(&table)?;
        let joint = Joint::Influence {
            inf: &inf,
            unit_cohort: &table.unit_cohort,
        };
        let aggregator = Aggregator {
            cohorts: &table.cohorts,
            n_cohort: &n_cohort,
            joint: &joint,
            z,
            base_offset,
        };
        inference(
            &mut group_time_atts,
            &aggregator,
            base_offset,
            &mut warnings,
        )?
    } else {
        // Bootstrap: every replicate draws units with replacement; without
        // covariates it weights the table's outcome changes by how often it drew
        // each unit, with covariates it perturbs the cells' influence functions by
        // those counts (a multiplier bootstrap). The cells, aggregates and
        // pre-trend vector are recomputed in every replicate.
        let rng_seed = config.seed.unwrap_or_else(|| {
            use rand::RngCore;
            rand::thread_rng().next_u64()
        });
        let mut rng = StdRng::seed_from_u64(rng_seed);
        let draws = bootstrap_draws(&table, config.bootstrap, &mut rng)?;
        let joint = Joint::Replicates { draws: &draws };
        let aggregator = Aggregator {
            cohorts: &table.cohorts,
            n_cohort: &n_cohort,
            joint: &joint,
            z,
            base_offset,
        };
        let (aggregates, pretrend) = inference(
            &mut group_time_atts,
            &aggregator,
            base_offset,
            &mut warnings,
        )?;
        let incomplete = draws
            .cells
            .rows()
            .into_iter()
            .filter(|row| row.iter().any(|v| !v.is_finite()))
            .count();
        if incomplete > 0
            && let Some((used, label)) = &aggregates.fewest.0
        {
            let mut w = format!(
                "bootstrap: {incomplete} of {} replicates could not compute every ATT(g,t) \
                 (a resample without treated or comparison units, or a unit without its base \
                 period); each standard error uses the replicates in which its cells were \
                 computed (fewest: {used}, for {label})",
                config.bootstrap
            );
            if *used < MIN_BOOTSTRAP {
                w.push_str(&format!(
                    "; with fewer than {MIN_BOOTSTRAP} replicates that standard error is \
                     imprecise; bootstrap = 0 gives analytic standard errors"
                ));
            }
            warnings.push(w);
        }
        (aggregates, pretrend)
    };
    if delta > 0
        && let Some(pt) = pretrend_test.as_mut()
    {
        pt.method.push_str(&format!(
            " (including the {delta} anticipation period(s) before each cohort's treatment, \
             as did::att_gt)"
        ));
    }
    let Aggregates {
        event_study,
        group_effects,
        overall: overall_att,
        ..
    } = aggregates;

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
        covariates: covariate_names,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// ATT Computation
// ═══════════════════════════════════════════════════════════════════════════════

/// Whether a unit first treated at `gi` (`<= 0`: never treated) is a comparison
/// unit for the cell `(g, t)` with base period `base_t`.
///
/// The rule of `did::compute.att_gt`: never treated units under `NeverTreated`;
/// under `NotYetTreated` also the units first treated after both `t` and the base
/// period plus `anticipation` (they must not anticipate their treatment by then),
/// excluding cohort `g` itself (a cohort treated by the base period has treatment
/// effects in its outcome change).
fn is_comparison(
    gi: i64,
    g: i64,
    t: i64,
    base_t: i64,
    group: ComparisonGroup,
    anticipation: i64,
) -> bool {
    match group {
        ComparisonGroup::NeverTreated => gi <= 0,
        ComparisonGroup::NotYetTreated => {
            gi <= 0 || (gi > t.max(base_t).saturating_add(anticipation) && gi != g)
        }
    }
}

/// Record why a cell was not computed: a warning, and the first reason for the
/// error when no cell is left.
fn push_skip(first_skip: &mut Option<String>, warnings: &mut Vec<String>, w: String) {
    if first_skip.is_none() {
        *first_skip = Some(w.clone());
    }
    warnings.push(w);
}

/// Compute a single group-time ATT `(g, t, base_t)` and its cell counts: the
/// unconditional difference in mean outcome changes (with covariates the ATT is
/// replaced by the cell's DRDID estimate). The standard error and the CI are set
/// later, from the influence functions or the bootstrap.
#[allow(clippy::too_many_arguments)]
fn compute_group_time_att(
    y: &Array1<f64>,
    g_col: &Array1<f64>,
    t_col: &Array1<f64>,
    unit_ids: &Array1<f64>,
    panel_map: &BTreeMap<(i64, i64), usize>,
    (g, t, base_t): (i64, i64, i64),
    comparison_group: ComparisonGroup,
    anticipation: i64,
) -> EconResult<GroupTimeATT> {
    let n = y.len();
    let (mut sum1, mut n_treated, mut sum0, mut n_comparison) = (0.0, 0usize, 0.0, 0usize);
    for i in 0..n {
        if t_col[i] as i64 != t {
            continue;
        }
        let gi = g_col[i] as i64;
        let treated = gi == g;
        if !treated && !is_comparison(gi, g, t, base_t, comparison_group, anticipation) {
            continue;
        }
        // only units observed at the base period enter the cell
        let Some(&base_idx) = panel_map.get(&(unit_ids[i] as i64, base_t)) else {
            continue;
        };
        let dy = y[i] - y[base_idx];
        if treated {
            sum1 += dy;
            n_treated += 1;
        } else {
            sum0 += dy;
            n_comparison += 1;
        }
    }

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

    Ok(GroupTimeATT {
        group: g,
        time: t,
        att: sum1 / n_treated as f64 - sum0 / n_comparison as f64,
        // set from the influence functions or the bootstrap replicates
        std_error: f64::NAN,
        ci_lower: f64::NAN,
        ci_upper: f64::NAN,
        n_treated,
        n_comparison,
        // post = 1(g <= t), as did (anticipation cells are pre-treatment cells)
        post_treatment: t >= g,
        relative_time: t - g,
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
/// treated or comparison row, the index of its unit in the sorted unit list, and
/// the data row of the earlier of the two periods `min(t, base)`, whose covariates
/// the cell uses (did's rule).
struct BootEntry {
    unit: usize,
    dy: f64,
    treated: bool,
    xrow: usize,
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
    /// The sorted distinct cohorts `g` of the cells.
    cohorts: Vec<i64>,
    /// Per unit, the index into `cohorts` of its first treatment period (from its
    /// first row); `None` for never-treated units and cohorts without a cell.
    unit_cohort: Vec<Option<usize>>,
    /// With covariates, every cell's DRDID estimate and the scaled influence value
    /// of every entry; `None` for the unconditional estimator.
    linear: Option<Linearised>,
}

/// The fitted covariate model of every cell: the ATT of cell `c` and, per entry
/// `e` of the table, `(n_units / n_cell) * IF_e`, did's rescaled influence function
/// (`entries` order).
struct Linearised {
    att: Vec<f64>,
    inf: Vec<f64>,
}

impl Linearised {
    /// From the fits of the table's cells, in cell order.
    fn new(table: &BootTable, fits: &[drdid::CellFit]) -> EconResult<Self> {
        let n_units = table.n_units as f64;
        let mut inf =
            resource::try_vec(table.entries.len(), 0.0, "staggered DiD influence values")?;
        for (c, fit) in fits.iter().enumerate() {
            let (a, b) = (table.starts[c], table.starts[c + 1]);
            if fit.influence.len() != b - a {
                return Err(EconError::Computation(format!(
                    "influence function of cell {c} has {} values for {} entries",
                    fit.influence.len(),
                    b - a
                )));
            }
            let scale = n_units / (b - a) as f64;
            for (slot, v) in inf[a..b].iter_mut().zip(fit.influence.iter()) {
                *slot = scale * v;
            }
        }
        Ok(Linearised {
            att: fits.iter().map(|f| f.att).collect(),
            inf,
        })
    }
}

/// Fit DRDID's estimator `method` in every cell of `table` (cells `(g, t, base)`
/// in table order), with the covariates `x` (no intercept column) of each entry's
/// earlier period. The reference cell `t == base` has ATT 0 and a zero influence
/// function without a fit, as did. A cell whose fit fails on its data is an `Err`
/// in its slot (the caller drops the cell with a warning); a resource limit or a
/// cancellation is the call's error, not a cell's.
fn covariate_fits(
    table: &BootTable,
    cells: &[(i64, i64, i64)],
    x: &Array2<f64>,
    method: AttEstimationMethod,
) -> EconResult<Vec<EconResult<drdid::CellFit>>> {
    covariate_fits_with(table, cells, x, |cell| drdid::estimate(method, cell))
}

/// [`covariate_fits`] with the cell estimator `fit` (a test seam).
fn covariate_fits_with(
    table: &BootTable,
    cells: &[(i64, i64, i64)],
    x: &Array2<f64>,
    fit: impl Fn(&drdid::CellInput<'_>) -> EconResult<drdid::CellFit>,
) -> EconResult<Vec<EconResult<drdid::CellFit>>> {
    let p = x.ncols() + 1;
    let mut out = Vec::with_capacity(cells.len());
    for (c, &(_, t, base_t)) in cells.iter().enumerate() {
        resource::checkpoint()?;
        let entries = &table.entries[table.starts[c]..table.starts[c + 1]];
        let m = entries.len();
        if t == base_t {
            out.push(Ok(drdid::CellFit {
                att: 0.0,
                influence: resource::try_zeros1(m, "staggered DiD cell influence")?,
                notes: Vec::new(),
            }));
            continue;
        }
        let dy: Array1<f64> = entries.iter().map(|e| e.dy).collect();
        let d: Array1<f64> = entries
            .iter()
            .map(|e| if e.treated { 1.0 } else { 0.0 })
            .collect();
        let mut xm = resource::try_zeros2(m, p, "staggered DiD cell design")?;
        for (mut row, e) in xm.rows_mut().into_iter().zip(entries) {
            row[0] = 1.0;
            row.slice_mut(s![1..]).assign(&x.row(e.xrow));
        }
        let cell = drdid::CellInput {
            dy: dy.view(),
            d: d.view(),
            x: xm.view(),
        };
        match fit(&cell) {
            Err(e @ (EconError::ResourceLimit(_) | EconError::Cancelled(_))) => return Err(e),
            r => out.push(r.and_then(|fit| {
                if fit.att.is_finite() && fit.influence.iter().all(|v| v.is_finite()) {
                    Ok(fit)
                } else {
                    Err(EconError::Computation(
                        "the cell estimate is not finite".to_string(),
                    ))
                }
            })),
        }
    }
    Ok(out)
}

impl BootTable {
    fn n_cells(&self) -> usize {
        self.starts.len().saturating_sub(1)
    }

    /// Number of units in each cohort of `cohorts` (the weights P(G = g) of
    /// `did::aggte`, up to the common factor n).
    fn cohort_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![0usize; self.cohorts.len()];
        for j in self.unit_cohort.iter().flatten() {
            sizes[*j] += 1;
        }
        sizes
    }
}

/// Tabulate, for each `(g, t, base_t)` cell, the outcome changes of its treated
/// and comparison rows, with the same row semantics as the point estimate:
/// treated rows have `G = g`, comparison rows follow `comparison_group` (with
/// `anticipation`), and rows of other cohorts are left out. Each entry records the
/// row of its earlier period for the covariates; the covariate fits are attached
/// later (`BootTable::linear`).
#[allow(clippy::too_many_arguments)]
fn build_boot_table(
    y: &Array1<f64>,
    g_col: &Array1<f64>,
    t_col: &Array1<f64>,
    unit_ids: &Array1<f64>,
    panel_map: &BTreeMap<(i64, i64), usize>,
    cells: &[(i64, i64, i64)],
    comparison_group: ComparisonGroup,
    anticipation: i64,
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
                if treated || is_comparison(gi, g, t, base_t, comparison_group, anticipation) {
                    entries.push(BootEntry {
                        unit: unit_of_row[i],
                        dy,
                        treated,
                        xrow: if t <= base_t { i } else { base_idx },
                    });
                }
            }
        }
        starts.push(entries.len());
        blocker_starts.push(blockers.len());
    }

    let mut cohorts: Vec<i64> = cells.iter().map(|&(g, _, _)| g).collect();
    cohorts.sort_unstable();
    cohorts.dedup();
    let mut unit_cohort: Vec<Option<usize>> =
        resource::try_vec(unit_list.len(), None, "staggered DiD unit cohorts")?;
    let mut seen = resource::try_vec(unit_list.len(), false, "staggered DiD unit cohorts")?;
    for i in 0..n {
        let u = unit_of_row[i];
        if !seen[u] {
            seen[u] = true;
            unit_cohort[u] = cohorts.binary_search(&(g_col[i] as i64)).ok();
        }
    }

    Ok(BootTable {
        n_units: unit_list.len(),
        entries,
        starts,
        blockers,
        blocker_starts,
        cohorts,
        unit_cohort,
        linear: None,
    })
}

/// The joint bootstrap draws: replicate x cell ATTs (`NaN` where a replicate
/// could not compute a cell) and replicate x cohort counts of drawn units (the
/// replicate's own cohort sizes; columns follow `BootTable::cohorts`).
struct BootDraws {
    cells: Array2<f64>,
    cohort_counts: Array2<f64>,
}

/// Draw `bootstrap` replicates of every cell's ATT.
///
/// Each replicate draws `n_units` units with replacement from the one `rng`
/// (the same stream as before) and turns the draws into multiplicity weights
/// `counts`, so it costs O(units + table entries). Without covariates a replicate
/// is the cell's difference in mean changes with those weights (a unit
/// bootstrap). With covariates (`BootTable::linear`) it is the multiplier
/// bootstrap of the cell's influence function, `att_c + sum_e (counts_e - 1)
/// IF_e / n_units` with did's rescaled IF_e, which estimates the analytic SE up
/// to Monte Carlo error (as did::att_gt's default bootstrap) instead of
/// re-fitting every cell in every replicate. Rows of `cells` are replicates and
/// columns are cells, which keeps the joint draws available for the aggregates
/// and the pre-trend covariance (CAUS-2, CAUS-86). A cell without an estimate in
/// a replicate (a drawn unit lacks the base period, or one side has no weight)
/// is `NaN`. `cohort_counts` holds how many drawn units each cohort has.
fn bootstrap_draws(table: &BootTable, bootstrap: usize, rng: &mut StdRng) -> EconResult<BootDraws> {
    let n_cells = table.n_cells();
    let n_units = table.n_units;
    let n_cohorts = table.cohorts.len();
    let mut draws = resource::try_zeros2(bootstrap, n_cells, "bootstrap replicate draws")?;
    resource::checked_len(&[bootstrap, n_cohorts], "bootstrap cohort counts")?;
    let mut cohort_counts = resource::try_zeros2(bootstrap, n_cohorts, "bootstrap cohort counts")?;
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
        for (u, &count) in counts.iter().enumerate() {
            if let Some(j) = table.unit_cohort[u] {
                cohort_counts[[b, j]] += count as f64;
            }
        }

        if let Some(lin) = &table.linear {
            let n = n_units as f64;
            for c in 0..n_cells {
                let (lo, hi) = (table.starts[c], table.starts[c + 1]);
                let shift: f64 = table.entries[lo..hi]
                    .iter()
                    .zip(&lin.inf[lo..hi])
                    .map(|(e, v)| (counts[e.unit] as f64 - 1.0) * v)
                    .sum();
                draws[[b, c]] = lin.att[c] + shift / n;
            }
            continue;
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

    Ok(BootDraws {
        cells: draws,
        cohort_counts,
    })
}

/// DRDID's panel ATT estimators for one (g, t) cell (Sant'Anna and Zhao 2020; the
/// DRDID 1.3.0 functions `reg_did_panel`, `std_ipw_did_panel` and `drdid_panel` as
/// `did::att_gt` calls them), each with its influence function.
///
/// Notation: `n` entries (treated and comparison units of the cell), `dy` their
/// outcome changes, `d` the treatment indicator, `x` the design `[1, covariates]` of
/// the earlier of the two periods. All unit weights are 1 (sampling weights are not
/// supported).
mod drdid {
    use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

    use super::AttEstimationMethod;
    use crate::econometrics::glm::{GlmSpec, fit_glm};
    use crate::errors::{EconError, EconResult};
    use crate::linalg::solve::{SingularPolicy, sym_inverse};
    use crate::regression::ols::solve_ols;

    /// DRDID caps the propensity score at `1 - 1e-6` (`pmin(ps, 1 - 1e-6)`).
    pub(super) const PS_CAP: f64 = 1.0 - 1e-6;
    /// DRDID's `trim.level`: comparison units with a propensity score at or above it
    /// get weight 0.
    pub(super) const TRIM_LEVEL: f64 = 0.995;
    /// did's per-cell overlap guard: a largest propensity score at or above it makes
    /// the cell NA.
    pub(super) const OVERLAP_LIMIT: f64 = 0.999;

    /// One cell's data.
    pub(super) struct CellInput<'a> {
        pub dy: ArrayView1<'a, f64>,
        pub d: ArrayView1<'a, f64>,
        /// `[1, covariates]`, one row per entry.
        pub x: ArrayView2<'a, f64>,
    }

    /// One cell's estimate, its influence function (one value per entry, unscaled:
    /// did multiplies it by n_units / n_cell) and notes for the result warnings.
    pub(super) struct CellFit {
        pub att: f64,
        pub influence: Array1<f64>,
        pub notes: Vec<String>,
    }

    /// The outcome regression of `dy` on `x` over the comparison entries: the fitted
    /// values `x b` of every entry and `A^-1 = (X0'X0 / n)^-1`.
    struct OutcomeModel {
        od: Array1<f64>,
        a_inv: Array2<f64>,
    }

    fn outcome_model(cell: &CellInput<'_>) -> EconResult<OutcomeModel> {
        let n = cell.d.len();
        let p = cell.x.ncols();
        let comp: Vec<usize> = (0..n).filter(|&i| cell.d[i] == 0.0).collect();
        let x0 = cell.x.select(Axis(0), &comp);
        let y0 = cell.dy.select(Axis(0), &comp);
        let singular = || EconError::SingularMatrix {
            context: "outcome regression of ATT(g,t) on the comparison units".to_string(),
            suggestion: "the covariates are collinear among the comparison units (did: \
                         'Covariate matrix for control units is singular')"
                .to_string(),
        };
        if comp.len() < p {
            return Err(singular());
        }
        let fit = solve_ols(x0.view(), y0.view())?;
        if fit.rank < p {
            return Err(singular());
        }
        let od = cell.x.dot(&fit.beta);
        // A = X0'X0 / n, so A^-1 = n (X0'X0)^-1
        let a_inv = fit.unscaled * n as f64;
        Ok(OutcomeModel { od, a_inv })
    }

    /// `X' v / n`.
    fn col_means(x: ArrayView2<'_, f64>, v: &Array1<f64>) -> Array1<f64> {
        x.t().dot(v) / v.len() as f64
    }

    /// Estimate the cell's ATT and its influence function with `method`.
    pub(super) fn estimate(
        method: AttEstimationMethod,
        cell: &CellInput<'_>,
    ) -> EconResult<CellFit> {
        let n = cell.d.len();
        let nf = n as f64;
        let (x, dy, d) = (cell.x, cell.dy, cell.d);
        let mut notes = Vec::new();
        let mean = |v: &Array1<f64>| v.sum() / nf;

        if method == AttEstimationMethod::OutcomeRegression {
            let om = outcome_model(cell)?;
            let wt: Array1<f64> = d.to_owned();
            let mt = mean(&wt);
            let eta_t = mean(&(&wt * &dy)) / mt;
            let eta_c = mean(&(&wt * &om.od)) / mt;
            let z = om.a_inv.dot(&col_means(x, &wt));
            let xz = x.dot(&z);
            let influence = Array1::from_iter((0..n).map(|i| {
                let ols = (1.0 - d[i]) * (dy[i] - om.od[i]) * xz[i];
                (wt[i] * dy[i] - wt[i] * eta_t) / mt - (wt[i] * om.od[i] - wt[i] * eta_c + ols) / mt
            }));
            return Ok(CellFit {
                att: eta_t - eta_c,
                influence,
                notes,
            });
        }

        // propensity score: logit MLE by IRLS
        let fit = fit_glm(x, d, &GlmSpec::logit())?;
        if !fit.aliased.is_empty() {
            return Err(EconError::SingularMatrix {
                context: "propensity score of ATT(g,t)".to_string(),
                suggestion: "propensity score model coefficients are not identified; the \
                             covariates are collinear (DRDID: 'Multicollinearity (or lack of \
                             variation) of covariates is a likely reason')"
                    .to_string(),
            });
        }
        if !fit.converged {
            notes.push(format!(
                "propensity score did not converge ({} IRLS iterations); the ATT uses the last \
                 iterate",
                fit.iterations
            ));
        }
        let max_ps = fit.fitted.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
        if max_ps.is_nan() || max_ps >= OVERLAP_LIMIT {
            return Err(EconError::Computation(format!(
                "overlap condition violated (largest propensity score {max_ps:.6} >= 0.999); \
                 did::att_gt reports NA for this ATT(g,t)"
            )));
        }
        let ps: Array1<f64> = fit.fitted.mapv(|v| v.min(PS_CAP));
        let trim: Array1<f64> = Array1::from_iter((0..n).map(|i| {
            if d[i] == 1.0 || ps[i] < TRIM_LEVEL {
                1.0
            } else {
                0.0
            }
        }));
        let n_trim = trim.iter().filter(|&&t| t == 0.0).count();
        if n_trim > 0 {
            notes.push(format!(
                "{n_trim} comparison units with propensity score >= 0.995 get weight 0 (DRDID \
                 trim.level)"
            ));
        }
        let wtt: Array1<f64> = &trim * &d;
        let wc: Array1<f64> =
            Array1::from_iter((0..n).map(|i| trim[i] * ps[i] * (1.0 - d[i]) / (1.0 - ps[i])));
        let (mt, mc) = (mean(&wtt), mean(&wc));
        if mt <= 0.0 || mc <= 0.0 || mt.is_nan() || mc.is_nan() {
            return Err(EconError::Computation(
                "the propensity-score weights of ATT(g,t) sum to zero".to_string(),
            ));
        }
        // H = n (X' diag(ps (1 - ps)) X)^-1
        let xw = &x * &ps.mapv(|p| p * (1.0 - p)).insert_axis(Axis(1));
        let info = x.t().dot(&xw);
        let h = sym_inverse(
            info.view(),
            SingularPolicy::Error,
            "propensity-score information of ATT(g,t)",
        )?
        .inverse
            * nf;
        // (d_i - ps_i) x_i' H v for a vector v
        let ps_term = |m: &Array1<f64>| -> Array1<f64> {
            let xv = x.dot(&h.dot(m));
            Array1::from_iter((0..n).map(|i| (d[i] - ps[i]) * xv[i]))
        };

        let (att, influence) = if method == AttEstimationMethod::IPW {
            let eta_t = mean(&(&wtt * &dy)) / mt;
            let eta_c = mean(&(&wc * &dy)) / mc;
            let m2 = col_means(x, &(&wc * &dy.mapv(|v| v - eta_c)));
            let pst = ps_term(&m2);
            let inf = Array1::from_iter((0..n).map(|i| {
                (wtt[i] * dy[i] - wtt[i] * eta_t) / mt
                    - ((wc[i] * dy[i] - wc[i] * eta_c) + pst[i]) / mc
            }));
            (eta_t - eta_c, inf)
        } else {
            let om = outcome_model(cell)?;
            let r: Array1<f64> = &dy - &om.od;
            let eta_t = mean(&(&wtt * &r)) / mt;
            let eta_c = mean(&(&wc * &r)) / mc;
            // (1 - d_i) r_i x_i' A^-1 v for a vector v
            let ols_term = |m: &Array1<f64>| -> Array1<f64> {
                let xv = x.dot(&om.a_inv.dot(m));
                Array1::from_iter((0..n).map(|i| (1.0 - d[i]) * r[i] * xv[i]))
            };
            let o1 = ols_term(&col_means(x, &wtt));
            let o3 = ols_term(&col_means(x, &wc));
            let pst = ps_term(&col_means(x, &(&wc * &r.mapv(|v| v - eta_c))));
            let inf = Array1::from_iter((0..n).map(|i| {
                let treat = ((wtt[i] * r[i] - wtt[i] * eta_t) - o1[i]) / mt;
                let cont = ((wc[i] * r[i] - wc[i] * eta_c) + pst[i] - o3[i]) / mc;
                treat - cont
            }));
            (eta_t - eta_c, inf)
        };
        Ok(CellFit {
            att,
            influence,
            notes,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Influence functions, aggregation and the pre-trend test
// ═══════════════════════════════════════════════════════════════════════════════

/// Unit-level influence functions of every cell (rows: units of the table,
/// columns: cells), as `did::att_gt`: with covariates the cells' DRDID influence
/// functions (`BootTable::linear`); without, `est_method = "reg"` and no covariates
/// (`DRDID::reg_did_panel`, rescaled by n / n_cell), as follows.
///
/// For a cell with `n1` treated entries of mean change `mu1` and `n0` comparison
/// entries of mean change `mu0`, a treated entry adds `n (dy - mu1) / n1` to its
/// unit's influence function and a comparison entry adds `-n (dy - mu0) / n0`,
/// with `n` the number of units. The cell's SE is then `sqrt(sum_u IF_u^2) / n`,
/// which is did's `sqrt(mean(IF^2) / n)`. A base-period cell `(g, g - 1)` has
/// `dy = 0` and an influence function of exactly zero; its SE is reported as NaN,
/// as did reports NA.
fn cell_influence(table: &BootTable) -> EconResult<Array2<f64>> {
    let n_units = table.n_units;
    let n_cells = table.n_cells();
    let mut inf = resource::try_zeros2(n_units, n_cells, "staggered DiD influence functions")?;
    let n = n_units as f64;
    if let Some(lin) = &table.linear {
        // covariates: each cell's DRDID influence function, already rescaled
        for c in 0..n_cells {
            resource::checkpoint()?;
            let (lo, hi) = (table.starts[c], table.starts[c + 1]);
            for (e, v) in table.entries[lo..hi].iter().zip(&lin.inf[lo..hi]) {
                inf[[e.unit, c]] += v;
            }
        }
        return Ok(inf);
    }
    for c in 0..n_cells {
        resource::checkpoint()?;
        let entries = &table.entries[table.starts[c]..table.starts[c + 1]];
        let (mut sum1, mut n1, mut sum0, mut n0) = (0.0, 0usize, 0.0, 0usize);
        for e in entries {
            if e.treated {
                sum1 += e.dy;
                n1 += 1;
            } else {
                sum0 += e.dy;
                n0 += 1;
            }
        }
        if n1 == 0 || n0 == 0 {
            inf.column_mut(c).fill(f64::NAN);
            continue;
        }
        let (mu1, mu0) = (sum1 / n1 as f64, sum0 / n0 as f64);
        for e in entries {
            inf[[e.unit, c]] += if e.treated {
                n * (e.dy - mu1) / n1 as f64
            } else {
                -n * (e.dy - mu0) / n0 as f64
            };
        }
    }
    Ok(inf)
}

/// How the cells of one aggregate are weighted (`did::aggte`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Weighting {
    /// Weights P(G = g) of each cell's cohort: the "simple" (overall) and
    /// "dynamic" (event-study) aggregations, with the weight-estimation term.
    CohortShare,
    /// Equal weights: a cohort's average over its post-treatment cells ("group").
    Equal,
}

/// The joint sampling information of the kept cells; columns of `inf` and
/// `draws.cells` follow `group_time_atts`.
enum Joint<'a> {
    /// Unit-level influence functions (n_units x n_cells) and each unit's cohort
    /// index (`BootTable::unit_cohort`).
    Influence {
        inf: &'a Array2<f64>,
        unit_cohort: &'a [Option<usize>],
    },
    /// Unit-bootstrap replicates of every cell and the replicates' cohort counts.
    Replicates { draws: &'a BootDraws },
}

/// An aggregate estimate, its SE, and how many units (influence functions) or
/// bootstrap replicates the SE used.
#[derive(Debug, Clone, Copy)]
struct CellStat {
    att: f64,
    se: f64,
    used: usize,
}

/// The estimate and SE of a weighted average of the cells `keep`.
///
/// `cohorts` are the sorted cohorts of the cells and `n_cohort` their unit
/// counts. With `CohortShare` the estimate is `sum_K pg_k att_k / S` with
/// `S = sum_K pg_k`, and the influence function of unit u is
/// `sum_K (pg_k / S) IF_uk + wif_u`, where the weight term
/// `wif_u = (1/S) sum_{k in K, g_k = G_u} (att_k - theta)` is did's
/// `wif(..) %*% att` simplified (its `pg` terms cancel because the weights sum to
/// S). With `Equal` it is the mean of the cells' influence functions.
///
/// With replicates, each replicate b recomputes the statistic with its own
/// cohort counts `c_bk` (`sum_K c_bk att_bk / sum_K c_bk`, or the mean with
/// `Equal`); a replicate counts when every `att_bk` is finite (and, with
/// `CohortShare`, `sum_K c_bk > 0`). The SE is the SD of the used replicate
/// statistics (denominator `used - 1`; NaN with fewer than 2).
fn aggregate_cells(
    atts: &[GroupTimeATT],
    keep: &[usize],
    w: Weighting,
    cohorts: &[i64],
    n_cohort: &[usize],
    joint: &Joint,
) -> EconResult<CellStat> {
    resource::checkpoint()?;
    let cohort_of = |k: usize| cohorts.binary_search(&atts[k].group).ok();
    let weights: Vec<f64> = keep
        .iter()
        .map(|&k| match w {
            Weighting::Equal => 1.0,
            Weighting::CohortShare => cohort_of(k).map_or(0.0, |j| n_cohort[j] as f64),
        })
        .collect();
    let sum_w: f64 = weights.iter().sum();
    let theta = keep
        .iter()
        .zip(&weights)
        .map(|(&k, &wk)| wk * atts[k].att)
        .sum::<f64>()
        / sum_w;
    match joint {
        Joint::Influence { inf, unit_cohort } => {
            let n_units = inf.nrows();
            let mut agg = resource::try_zeros1(n_units, "staggered DiD aggregate influence")?;
            for (&k, &wk) in keep.iter().zip(&weights) {
                agg.scaled_add(wk / sum_w, &inf.column(k));
            }
            if w == Weighting::CohortShare {
                let mut wif = resource::try_vec(cohorts.len(), 0.0, "staggered DiD weight term")?;
                for &k in keep {
                    if let Some(j) = cohort_of(k) {
                        wif[j] += atts[k].att - theta;
                    }
                }
                // 1/S with S = sum_K pg_k = sum_w / n
                let scale = n_units as f64 / sum_w;
                for (a, cohort) in agg.iter_mut().zip(unit_cohort.iter()) {
                    if let Some(j) = cohort {
                        *a += scale * wif[*j];
                    }
                }
            }
            let se = agg.iter().map(|v| v * v).sum::<f64>().sqrt() / n_units as f64;
            Ok(CellStat {
                att: theta,
                se,
                used: n_units,
            })
        }
        Joint::Replicates { draws } => {
            let n_reps = draws.cells.nrows();
            let mut stats: Vec<f64> =
                resource::try_with_capacity(n_reps, "bootstrap replicate statistics")?;
            for b in 0..n_reps {
                let (mut num, mut den, mut complete) = (0.0, 0.0, true);
                for &k in keep {
                    let a = draws.cells[[b, k]];
                    if !a.is_finite() {
                        complete = false;
                        break;
                    }
                    let c = match w {
                        Weighting::Equal => 1.0,
                        Weighting::CohortShare => {
                            cohort_of(k).map_or(0.0, |j| draws.cohort_counts[[b, j]])
                        }
                    };
                    num += c * a;
                    den += c;
                }
                if complete && den > 0.0 {
                    stats.push(num / den);
                }
            }
            Ok(CellStat {
                att: theta,
                se: sample_sd(&stats),
                used: stats.len(),
            })
        }
    }
}

/// Sample SD (denominator `len - 1`); NaN with fewer than 2 values.
fn sample_sd(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return f64::NAN;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    var.sqrt()
}

/// The cohorts and the joint sampling information that the aggregates of one
/// result share.
struct Aggregator<'a> {
    cohorts: &'a [i64],
    n_cohort: &'a [usize],
    joint: &'a Joint<'a>,
    z: f64,
    /// The reference cells' event time (`base_period - anticipation`).
    base_offset: i64,
}

/// The aggregates of one result, and the statistic whose SE used the fewest
/// replicates (or units).
struct Aggregates {
    event_study: Vec<AggregatedEffect>,
    group_effects: Vec<AggregatedEffect>,
    overall: AggregatedEffect,
    fewest: Fewest,
    /// Aggregates over at least one non-reference cell whose SE was at or below
    /// [`SE_NA_LIMIT`] and is reported as NaN.
    na_se: usize,
}

/// The fewest replicates (or units) any SE used, and that statistic's label;
/// the first statistic wins a tie.
#[derive(Debug, Default)]
struct Fewest(Option<(usize, String)>);

impl Fewest {
    fn note(&mut self, used: usize, label: impl FnOnce() -> String) {
        if self.0.as_ref().is_none_or(|(n, _)| used < *n) {
            self.0 = Some((used, label()));
        }
    }
}

impl Aggregator<'_> {
    fn stat(&self, atts: &[GroupTimeATT], keep: &[usize], w: Weighting) -> EconResult<CellStat> {
        aggregate_cells(atts, keep, w, self.cohorts, self.n_cohort, self.joint)
    }

    /// Each cell's estimate and SE.
    fn cells(&self, atts: &[GroupTimeATT]) -> EconResult<Vec<CellStat>> {
        (0..atts.len())
            .map(|c| self.stat(atts, &[c], Weighting::Equal))
            .collect()
    }

    fn effect(
        &self,
        atts: &[GroupTimeATT],
        key: i64,
        keep: &[usize],
        w: Weighting,
        fewest: &mut Fewest,
        na_se: &mut usize,
        label: impl FnOnce() -> String,
    ) -> EconResult<AggregatedEffect> {
        let mut stat = self.stat(atts, keep, w)?;
        fewest.note(stat.used, label);
        if stat.se <= SE_NA_LIMIT
            && keep
                .iter()
                .any(|&k| atts[k].relative_time != self.base_offset)
        {
            *na_se += 1;
        }
        stat.se = did_se(stat.se);
        Ok(AggregatedEffect {
            key,
            att: stat.att,
            std_error: stat.se,
            ci_lower: stat.att - self.z * stat.se,
            ci_upper: stat.att + self.z * stat.se,
            p_value: normal_p_value(stat.att, stat.se),
            n_cells: keep.len(),
            n_obs: keep.iter().map(|&k| atts[k].n_treated).sum(),
        })
    }

    /// The event-study ("dynamic"), group and overall ("simple") aggregates of
    /// `did::aggte`.
    fn aggregates(&self, atts: &[GroupTimeATT]) -> EconResult<Aggregates> {
        let mut fewest = Fewest::default();
        let mut na_se = 0;
        let mut by_e: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
        let mut by_g: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
        let mut post = Vec::new();
        for (k, c) in atts.iter().enumerate() {
            by_e.entry(c.relative_time).or_default().push(k);
            if c.post_treatment {
                by_g.entry(c.group).or_default().push(k);
                post.push(k);
            }
        }
        let mut event_study = Vec::with_capacity(by_e.len());
        for (&e, keep) in &by_e {
            let label = || format!("event time {e}");
            event_study.push(self.effect(
                atts,
                e,
                keep,
                Weighting::CohortShare,
                &mut fewest,
                &mut na_se,
                label,
            )?);
        }
        let mut group_effects = Vec::with_capacity(by_g.len());
        for (&g, keep) in &by_g {
            let label = || format!("group {g}");
            group_effects.push(self.effect(
                atts,
                g,
                keep,
                Weighting::Equal,
                &mut fewest,
                &mut na_se,
                label,
            )?);
        }
        if post.is_empty() {
            // run_staggered_did refuses this before aggregating (CAUS-109)
            return Err(EconError::InsufficientData {
                required: 1,
                provided: 0,
                context: "no post-treatment ATT(g,t) to aggregate".to_string(),
            });
        }
        let overall = self.effect(
            atts,
            0,
            &post,
            Weighting::CohortShare,
            &mut fewest,
            &mut na_se,
            || "the overall ATT".to_string(),
        )?;
        Ok(Aggregates {
            event_study,
            group_effects,
            overall,
            fewest,
            na_se,
        })
    }
}

/// Cell SEs and CIs, the aggregates and the pre-trend test from one joint
/// sampling distribution. Returns the aggregates (with the statistic whose SE
/// used the fewest replicates or units) and the pre-trend test.
fn inference(
    atts: &mut [GroupTimeATT],
    aggregator: &Aggregator,
    base_offset: i64,
    warnings: &mut Vec<String>,
) -> EconResult<(Aggregates, Option<PreTrendTest>)> {
    let z = aggregator.z;
    let cell_stats = aggregator.cells(atts)?;
    let mut fewest = Fewest::default();
    let mut na_se = 0usize;
    for (gta, stat) in atts.iter_mut().zip(&cell_stats) {
        // the reference cell's SE is 0 (zero influence function, constant
        // replicates) and is reported as NaN, as did reports NA; any other cell's
        // SE at or below the limit is NaN too (did's rule) and is counted
        if stat.se <= SE_NA_LIMIT && gta.time != gta.group + base_offset {
            na_se += 1;
        }
        let se = did_se(stat.se);
        gta.std_error = se;
        gta.ci_lower = gta.att - z * se;
        gta.ci_upper = gta.att + z * se;
        fewest.note(stat.used, || format!("ATT({},{})", gta.group, gta.time));
    }
    let mut aggregates = aggregator.aggregates(atts)?;
    if let Some((used, label)) = aggregates.fewest.0.take() {
        fewest.note(used, || label);
    }
    aggregates.fewest = fewest;
    na_se += aggregates.na_se;
    if na_se > 0 {
        warnings.push(format!(
            "{na_se} standard error(s) of cells or aggregates other than the base-period \
             reference cells are at or below {SE_NA_LIMIT:.2e} and are reported as NA \
             (did's rule); rescale the outcome if these are real standard errors"
        ));
    }
    let pre = pretrend_cells(atts, base_offset);
    let pretrend = pretrend_wald(atts, &pre, aggregator.joint, warnings)?;
    Ok((aggregates, pretrend))
}

/// The cells of the pre-trend test: pre-treatment cells (t < g, including the
/// anticipation cells, as did) other than the base period `g + base_offset` with a
/// positive, finite SE (did drops zero-variance cells).
fn pretrend_cells(atts: &[GroupTimeATT], base_offset: i64) -> Vec<usize> {
    atts.iter()
        .enumerate()
        .filter(|(_, c)| {
            !c.post_treatment
                && c.time != c.group + base_offset
                && c.std_error > 0.0
                && c.std_error.is_finite()
        })
        .map(|(k, _)| k)
        .collect()
}

/// The joint pre-trend test is computed for at most this many pre-treatment cells
/// k. Forming their k x k covariance from n units (or n bootstrap replicates) costs
/// O(n k^2) and its eigendecomposition O(k^3), so the test costs O(n k^2 + k^3).
const MAX_PRETREND_CELLS: usize = 2000;

/// Rows (units or replicates) per block when the pre-trend covariance is
/// accumulated; the budget is checked once per block.
const PRETREND_BLOCK_ROWS: usize = 4096;

/// `scale * X'X` for the `n_rows x k` matrix X, accumulated over blocks of at most
/// [`PRETREND_BLOCK_ROWS`] rows: `fill(r0, block)` writes rows `r0..r0 + block.nrows()`
/// of X into `block`. X is never held whole, and the budget is checked before every
/// block, so the O(n_rows k^2) product can be cancelled.
fn blocked_gram(
    n_rows: usize,
    k: usize,
    scale: f64,
    mut fill: impl FnMut(usize, ndarray::ArrayViewMut2<f64>),
) -> EconResult<Array2<f64>> {
    let mut v = resource::try_zeros2(k, k, "pre-trend covariance")?;
    let rows = PRETREND_BLOCK_ROWS.min(n_rows);
    let mut block = resource::try_zeros2(rows, k, "pre-trend covariance block")?;
    let mut r0 = 0;
    while r0 < n_rows {
        resource::checkpoint()?;
        let len = rows.min(n_rows - r0);
        fill(r0, block.slice_mut(s![..len, ..]));
        let b = block.slice(s![..len, ..]);
        ndarray::linalg::general_mat_mul(scale, &b.t(), &b, 1.0, &mut v);
        r0 += len;
    }
    Ok(v)
}

/// Hotelling's F for a statistic `w = a' V^+ a` whose covariance V (rank `rank`) is
/// the sample covariance of `used` bootstrap replicates: `F = (m - r) w / (r (m - 1))`
/// on F(r, m - r), with m = `used` and r = `rank`. Returns the F statistic, the
/// denominator df and the p-value; `None` when `rank` is 0 or `used <= rank`.
fn hotelling_f(w: f64, rank: usize, used: usize) -> Option<(f64, usize, f64)> {
    if rank == 0 || used <= rank {
        return None;
    }
    let df2 = used - rank;
    let f = df2 as f64 * w / (rank as f64 * (used - 1) as f64);
    let p = crate::traits::estimator::f_test_p_value(f, rank as f64, df2 as f64);
    Some((f, df2, p))
}

/// The joint test that the pre-treatment ATT(g,t) `pre` are zero, `W = a' V^+ a`
/// with the joint covariance V of the cells (`None` without pre cells): on
/// chi-squared(rank V) with the influence-function covariance, and as Hotelling's
/// T^2 on F(r, m - r) ([`hotelling_f`]) with the covariance of m bootstrap
/// replicates.
fn pretrend_wald(
    atts: &[GroupTimeATT],
    pre: &[usize],
    joint: &Joint,
    warnings: &mut Vec<String>,
) -> EconResult<Option<PreTrendTest>> {
    let k = pre.len();
    if k == 0 {
        return Ok(None);
    }
    let a: Vec<f64> = pre.iter().map(|&c| atts[c].att).collect();
    let not_computed = |method: String| PreTrendTest {
        chi2: f64::NAN,
        df: 0,
        p_value: f64::NAN,
        f_stat: None,
        df2: None,
        pre_atts: a.clone(),
        method,
    };
    if k > MAX_PRETREND_CELLS {
        return Ok(Some(not_computed(format!(
            "{k} pre-treatment ATT(g,t) exceed the {MAX_PRETREND_CELLS} the joint test is \
             computed for"
        ))));
    }
    // V, and (used, drawn) replicates for a bootstrap covariance
    let (v, replicates) = match joint {
        Joint::Influence { inf, .. } => {
            // V = IF'IF / n^2 over the pre columns, block by block of units
            let n_units = inf.nrows();
            let n = n_units as f64;
            let v = blocked_gram(n_units, k, 1.0 / (n * n), |r0, mut block| {
                let len = block.nrows();
                for (j, &c) in pre.iter().enumerate() {
                    block.column_mut(j).assign(&inf.slice(s![r0..r0 + len, c]));
                }
            })?;
            (v, None)
        }
        Joint::Replicates { draws } => {
            // the replicates in which every pre-treatment cell was computed, and
            // the mean of each pre cell over them
            let n_reps = draws.cells.nrows();
            let mut complete: Vec<usize> =
                resource::try_with_capacity(n_reps, "pre-trend bootstrap replicates")?;
            let mut mean = resource::try_vec(k, 0.0, "pre-trend bootstrap means")?;
            for b in 0..n_reps {
                if b % PRETREND_BLOCK_ROWS == 0 {
                    resource::checkpoint()?;
                }
                let row = draws.cells.row(b);
                if pre.iter().all(|&c| row[c].is_finite()) {
                    complete.push(b);
                    for (m, &c) in mean.iter_mut().zip(pre) {
                        *m += row[c];
                    }
                }
            }
            let used = complete.len();
            if used < 2 {
                return Ok(Some(not_computed(
                    "fewer than 2 bootstrap replicates computed every pre-treatment ATT(g,t)"
                        .to_string(),
                )));
            }
            for m in &mut mean {
                *m /= used as f64;
            }
            // V = X'X / (m - 1) with X the centred complete replicates
            let v = blocked_gram(used, k, 1.0 / (used - 1) as f64, |r0, mut block| {
                for (i, mut row) in block.rows_mut().into_iter().enumerate() {
                    let b = complete[r0 + i];
                    for (j, &c) in pre.iter().enumerate() {
                        row[j] = draws.cells[[b, c]] - mean[j];
                    }
                }
            })?;
            (v, Some((used, n_reps)))
        }
    };
    resource::checkpoint()?;
    let (w, rank) = wald_ginv(&a, &v)?;
    let method = match replicates {
        None => format!("Wald, influence-function covariance of {k} pre-treatment ATT(g,t)"),
        Some((used, n_reps)) => format!(
            "Hotelling T^2, bootstrap covariance of {k} pre-treatment ATT(g,t) ({used} of \
             {n_reps} replicates)"
        ),
    };
    if rank == 0 {
        return Ok(Some(not_computed(format!(
            "{method} (the covariance has rank 0)"
        ))));
    }
    let (p_value, f_stat, df2, method) = match replicates {
        None => (chi_squared_p_value(w, rank as f64), None, None, method),
        Some((used, _)) => match hotelling_f(w, rank, used) {
            Some((f, df2, p)) => (p, Some(f), Some(df2), format!("{method}, F({rank}, {df2})")),
            None => {
                return Ok(Some(not_computed(format!(
                    "{method}: {used} replicates do not exceed the rank {rank} of the \
                     covariance, so F({rank}, {used} - {rank}) has no denominator degrees of \
                     freedom"
                ))));
            }
        },
    };
    if rank < k {
        warnings.push(format!(
            "the covariance of the {k} pre-treatment ATT(g,t) is singular (rank {rank}); a \
             generalized inverse was used and df = {rank}"
        ));
    }
    Ok(Some(PreTrendTest {
        chi2: w,
        df: rank,
        p_value,
        f_stat,
        df2,
        pre_atts: a,
        method,
    }))
}

/// `a' V^+ a` and the rank of `V`, through the eigendecomposition of the
/// correlation-scaled `S V S` (`s_j = V_jj^-1/2`): eigenvalues at most
/// `max |lambda| * k * eps * 100` count as zero. A component with `V_jj = 0` gets
/// `s_j = 0` and drops out of the rank.
fn wald_ginv(a: &[f64], v: &Array2<f64>) -> EconResult<(f64, usize)> {
    let k = a.len();
    if v.iter().chain(a).any(|x| !x.is_finite()) {
        return Ok((f64::NAN, 0));
    }
    let s: Vec<f64> = (0..k)
        .map(|j| {
            let d = v[[j, j]];
            if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 }
        })
        .collect();
    let mut vs = resource::try_zeros2(k, k, "pre-trend covariance")?;
    for i in 0..k {
        for j in 0..k {
            vs[[i, j]] = 0.5 * (v[[i, j]] + v[[j, i]]) * s[i] * s[j];
        }
    }
    let (_, u) = eig_symmetric(&vs.view()).map_err(|e| {
        EconError::Computation(format!("pre-trend covariance eigendecomposition: {e}"))
    })?;
    // eigenvalues as Rayleigh quotients of the eigenvectors (pairing-safe)
    let vu = vs.dot(&u);
    let lambda: Vec<f64> = (0..k).map(|j| u.column(j).dot(&vu.column(j))).collect();
    let sa: Array1<f64> = a.iter().zip(&s).map(|(x, sj)| x * sj).collect();
    let lmax = lambda.iter().fold(0.0f64, |m, l| m.max(l.abs()));
    let tol = lmax * k as f64 * f64::EPSILON * 100.0;
    let (mut w, mut rank) = (0.0, 0usize);
    for (j, &l) in lambda.iter().enumerate() {
        if l > tol {
            let proj = u.column(j).dot(&sa);
            w += proj * proj / l;
            rank += 1;
        }
    }
    Ok((w, rank))
}

/// The standard normal distribution.
fn std_normal() -> Normal {
    Normal::standard()
}

/// Two-sided normal p-value of `est / se`: 1 when `se == 0`, NaN when either is NaN.
fn normal_p_value(est: f64, se: f64) -> f64 {
    if est.is_nan() || se.is_nan() {
        f64::NAN
    } else if se == 0.0 {
        1.0
    } else {
        2.0 * std_normal().sf((est / se).abs())
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

    fn expect_invalid<T>(res: EconResult<T>, needle: &str) -> String {
        match res {
            Err(EconError::InvalidSpecification { message }) if message.contains(needle) => message,
            Err(e) => panic!("expected InvalidSpecification naming {needle:?}, got: {e}"),
            Ok(_) => panic!("expected InvalidSpecification naming {needle:?}, got Ok"),
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

    fn pinned_bootstrap_ses(dataset: &Dataset) -> Vec<f64> {
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
        result.group_time_atts.iter().map(|g| g.std_error).collect()
    }

    /// Seeded bootstrap SEs captured from the per-replicate implementation.
    #[test]
    fn test_staggered_did_bootstrap_ses_pinned() {
        let balanced: [f64; 27] = [
            0.18490145985527776,
            f64::NAN, // the reference cell: SE NA as did
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
            f64::NAN, // the reference cell: SE NA as did
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
            f64::NAN, // the reference cell: SE NA as did
            0.2664967811861835,
            0.24702693059618694,
            0.2644702388339487,
        ];
        let unbalanced: [f64; 27] = [
            0.21337412146962192,
            f64::NAN, // the reference cell: SE NA as did
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
            f64::NAN, // the reference cell: SE NA as did
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
        let close = |a: f64, b: f64| {
            (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12 * b.abs().max(1.0)
        };
        // The overall SE is the SD of the replicate aggregates now; the test
        // test_staggered_did_bootstrap_aggregates_are_replicate_sds recomputes it.
        for (ds, want) in [
            (create_staggered_dataset(), &balanced),
            (create_unbalanced_staggered_dataset(), &unbalanced),
        ] {
            let ses = pinned_bootstrap_ses(&ds);
            assert_eq!(ses.len(), want.len());
            for (i, (&a, &b)) in ses.iter().zip(want.iter()).enumerate() {
                assert!(close(a, b), "cell {i}: {a} vs pinned {b}");
            }
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

    /// Parallel trends hold in the DGP, so the joint test does not reject at 1%,
    /// analytically (W = 16.55370047, df 9, p = 0.05618128981 with did's formulas)
    /// and with the 9 x 9 covariance of 50 bootstrap replicates, which is referred
    /// to Hotelling's F(9, m - 9) rather than chi-squared(9).
    #[test]
    fn test_pretrend_test() {
        let dataset = create_staggered_dataset();
        for bootstrap in [0, 50] {
            let config = StaggeredDidConfig {
                bootstrap,
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
            let pretrend = result.pretrend_test.as_ref().expect("pre-trend test");
            assert!(
                pretrend.p_value > 0.01,
                "bootstrap {bootstrap}: the pre-trend test should not reject parallel trends, \
                 got {pretrend:?}"
            );
            assert_eq!(pretrend.df, 9, "bootstrap {bootstrap}");
            let shown = format!("{result}");
            if bootstrap == 0 {
                assert_eq!((pretrend.f_stat, pretrend.df2), (None, None));
                assert!(shown.contains("  Chi-squared(9) = "), "{shown}");
            } else {
                let df2 = pretrend.df2.expect("F denominator df");
                let f = pretrend.f_stat.expect("F statistic");
                assert!(df2 <= 50 - 9, "{df2}");
                assert!(
                    pretrend.method.ends_with(&format!(", F(9, {df2})")),
                    "{}",
                    pretrend.method
                );
                let line = format!(
                    "  T^2 = {:.2}, F(9, {df2}) = {f:.4}, p-value = {:.4}",
                    pretrend.chi2, pretrend.p_value
                );
                assert!(shown.contains(&line), "{line:?} not in\n{shown}");
                assert!(!shown.contains("Chi-squared("), "{shown}");
            }
        }
    }

    /// Hotelling's F for a bootstrap covariance against R:
    /// `pf((m - r) / (r * (m - 1)) * W, r, m - r, lower.tail = FALSE)`.
    #[test]
    fn test_pretrend_hotelling_f_matches_pf() {
        // (W, r, m, F, p) with F and p from R 4.3.3
        let cases = [
            (16.5, 9, 50, 1.534013605442177, 0.1685901927291372),
            (3.2, 2, 200, 1.591959798994975, 0.20612021667265973),
            (250.0, 10, 11, 2.5, 0.4587204829251879),
        ];
        for (w, r, m, f_r, p_r) in cases {
            let (f, df2, p) = hotelling_f(w, r, m).expect("F test");
            assert_eq!(df2, m - r);
            assert!(rel_close(f, f_r, 1e-14), "F {f} vs {f_r}");
            assert!(rel_close(p, p_r, 1e-10), "p {p} vs {p_r}");
        }
        // no denominator degrees of freedom, or nothing to test
        assert!(hotelling_f(4.0, 9, 9).is_none());
        assert!(hotelling_f(4.0, 9, 5).is_none());
        assert!(hotelling_f(4.0, 0, 50).is_none());
    }

    /// The blocked covariance equals the naive `scale * X'X` over several blocks of
    /// noisy rows, and the budget is checked before every block: cancelling during
    /// the first block stops the product before the second.
    #[test]
    fn test_pretrend_blocked_gram_matches_naive_and_checks_every_block() {
        let (n, k) = (2 * PRETREND_BLOCK_ROWS + 17, 5);
        let x = Array2::from_shape_fn((n, k), |(i, j)| {
            ((i * 7 + j * 13) % 29) as f64 * 0.1 - 1.4 + ((i * j) % 5) as f64 * 0.03
        });
        let scale = 1.0 / (n - 1) as f64;
        let mut fills = Vec::new();
        let v = blocked_gram(n, k, scale, |r0, mut block| {
            fills.push((r0, block.nrows()));
            block.assign(&x.slice(s![r0..r0 + block.nrows(), ..]));
        })
        .unwrap();
        assert_eq!(
            fills,
            [
                (0, PRETREND_BLOCK_ROWS),
                (PRETREND_BLOCK_ROWS, PRETREND_BLOCK_ROWS),
                (2 * PRETREND_BLOCK_ROWS, 17)
            ]
        );
        for i in 0..k {
            for j in 0..k {
                let naive: f64 = (0..n).map(|r| x[[r, i]] * x[[r, j]]).sum::<f64>() * scale;
                assert!(
                    rel_close(v[[i, j]], naive, 1e-12),
                    "({i}, {j}): {} vs {naive}",
                    v[[i, j]]
                );
            }
        }

        let token = CancelToken::new();
        let mut blocks = 0;
        let res = with_budget(Budget::new(None, token.clone()), || {
            blocked_gram(n, k, scale, |r0, mut block| {
                blocks += 1;
                token.cancel();
                block.assign(&x.slice(s![r0..r0 + block.nrows(), ..]));
            })
        });
        expect_cancelled(res);
        assert_eq!(blocks, 1);
    }

    #[test]
    fn test_display() {
        let dataset = create_staggered_dataset();
        let config = StaggeredDidConfig {
            bootstrap: 0,
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
        assert!(output.contains("Standard Errors:"));
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
    fn test_staggered_did_group_time_atts_recovers_dgp() {
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
    fn test_staggered_did_event_study_pre_treatment_near_zero_recovers_dgp() {
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
    fn test_staggered_did_event_study_post_treatment_positive_properties() {
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
    fn test_staggered_did_overall_att_weighted_average_properties() {
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
    fn test_staggered_did_group_effects_heterogeneity_properties() {
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
    fn test_staggered_did_never_treated_comparison_properties() {
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
    fn test_staggered_did_homogeneous_effect_recovers_dgp() {
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
                // Check if comparison (did's rule, inlined so that the reference
                // stays independent of the production helper)
                let is_comparison = match config.comparison_group {
                    ComparisonGroup::NeverTreated => gi <= 0,
                    ComparisonGroup::NotYetTreated => gi <= 0 || (gi > t.max(base_t) && gi != g),
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
                build_boot_table(&y, &g, &t, &u, &panel_map, &cells, comparison_group, 0).unwrap();
            // the table builder checks the budget once per (g, t) cell
            expect_cancelled(with_budget(cancelled(), || {
                build_boot_table(&y, &g, &t, &u, &panel_map, &cells, comparison_group, 0)
            }));
            let draws = bootstrap_draws(&table, 200, &mut StdRng::seed_from_u64(42)).unwrap();
            assert_eq!(draws.cells.dim(), (200, cells.len()));
            for (c, want) in expected.iter().enumerate() {
                let got: Vec<f64> = draws
                    .cells
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

    /// The replicates of `reference_draws` (same extraction, unit list, seed and
    /// per-replicate `boot_units`), kept aligned across cells: per replicate, every
    /// cell's ATT (`None` when it cannot be computed) and the drawn cohort counts
    /// (`count[G_u] += 1` per drawn unit with `G_u > 0`, `G_u` from the unit's first row).
    fn reference_replicates(
        ds: &Dataset,
        config: &StaggeredDidConfig,
        cells: &[(i64, i64, i64)],
    ) -> Vec<(Vec<Option<f64>>, BTreeMap<i64, usize>)> {
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
        let mut unit_g: BTreeMap<i64, i64> = BTreeMap::new();
        for i in 0..n {
            unit_g.entry(unit_ids[i] as i64).or_insert(g_col[i] as i64);
        }
        let n_units = unit_list.len();
        let mut rng = StdRng::seed_from_u64(config.seed.unwrap());
        let mut out = Vec::new();
        for _ in 0..config.bootstrap {
            let boot_units: Vec<i64> = (0..n_units)
                .map(|_| unit_list[rng.gen_range(0..n_units)])
                .collect();
            let atts = cells
                .iter()
                .map(|&(g, t, base_t)| {
                    reference_bootstrap_att(
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
                    )
                    .filter(|a| a.is_finite())
                })
                .collect();
            let mut counts: BTreeMap<i64, usize> = BTreeMap::new();
            for u in &boot_units {
                let g = unit_g[u];
                if g > 0 {
                    *counts.entry(g).or_insert(0) += 1;
                }
            }
            out.push((atts, counts));
        }
        out
    }

    /// The SD (denominator `len - 1`) of the replicate statistic over the replicates
    /// in which every cell of `keep` is computed; `weight` gives a cell's weight in a
    /// replicate. Also returns the number of replicates used.
    fn reference_sd(
        reps: &[(Vec<Option<f64>>, BTreeMap<i64, usize>)],
        keep: &[usize],
        weight: impl Fn(usize, &BTreeMap<i64, usize>) -> f64,
    ) -> (f64, usize) {
        let mut stats = Vec::new();
        for (atts, counts) in reps {
            let Some(vals) = keep.iter().map(|&k| atts[k]).collect::<Option<Vec<f64>>>() else {
                continue;
            };
            let w: Vec<f64> = keep.iter().map(|&k| weight(k, counts)).collect();
            let den: f64 = w.iter().sum();
            if den > 0.0 {
                stats.push(vals.iter().zip(&w).map(|(a, w)| a * w).sum::<f64>() / den);
            }
        }
        let m = stats.len();
        let mean = stats.iter().sum::<f64>() / m as f64;
        let var = stats.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (m - 1) as f64;
        (var.sqrt(), m)
    }

    fn rel_close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1e-300)
    }

    /// Every bootstrap SE is the SD of the replicate statistic recomputed from the
    /// reference replicates (same seed): cells, overall (cohort-count weights of the
    /// replicate), groups (mean of the cohort's post cells), event times, and the
    /// pre-trend Wald statistic with the replicate covariance.
    #[test]
    fn test_staggered_did_bootstrap_aggregates_are_replicate_sds() {
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
            let r = run_staggered_did(
                &ds,
                "outcome",
                "first_treat",
                "year",
                "unit",
                None,
                config.clone(),
            )
            .unwrap();
            let atts = &r.group_time_atts;
            let cells: Vec<(i64, i64, i64)> = atts
                .iter()
                .map(|c| (c.group, c.time, c.group + config.base_period as i64))
                .collect();
            let reps = reference_replicates(&ds, &config, &cells);
            let cohort_w = |k: usize, counts: &BTreeMap<i64, usize>| {
                counts.get(&atts[k].group).copied().unwrap_or(0) as f64
            };
            let equal_w = |_: usize, _: &BTreeMap<i64, usize>| 1.0;
            let what = format!("{comparison_group:?}, {} rows", ds.nrows());

            for (k, c) in atts.iter().enumerate() {
                let (sd, _) = reference_sd(&reps, &[k], equal_w);
                if sd <= SE_NA_LIMIT {
                    // the reference cell: constant 0 replicates, SE NaN as did's NA
                    assert!(c.std_error.is_nan(), "{what}: {c:?}");
                    continue;
                }
                assert!(
                    rel_close(c.std_error, sd, 1e-10),
                    "{what}: cell ({}, {}) {} vs {sd}",
                    c.group,
                    c.time,
                    c.std_error
                );
            }
            let post: Vec<usize> = (0..atts.len())
                .filter(|&k| atts[k].post_treatment)
                .collect();
            let (sd, _) = reference_sd(&reps, &post, cohort_w);
            assert!(
                rel_close(r.overall_att.std_error, sd, 1e-10),
                "{what}: overall {} vs {sd}",
                r.overall_att.std_error
            );
            for e in &r.group_effects {
                let keep: Vec<usize> = post
                    .iter()
                    .copied()
                    .filter(|&k| atts[k].group == e.key)
                    .collect();
                let (sd, _) = reference_sd(&reps, &keep, equal_w);
                assert!(
                    rel_close(e.std_error, sd, 1e-10),
                    "{what}: group {} {} vs {sd}",
                    e.key,
                    e.std_error
                );
            }
            for e in &r.event_study {
                let keep: Vec<usize> = (0..atts.len())
                    .filter(|&k| atts[k].relative_time == e.key)
                    .collect();
                let (sd, _) = reference_sd(&reps, &keep, cohort_w);
                if sd <= SE_NA_LIMIT {
                    // the reference event time: SE NaN as did's NA
                    assert!(e.std_error.is_nan(), "{what}: event {}", e.key);
                    continue;
                }
                assert!(
                    rel_close(e.std_error, sd, 1e-10),
                    "{what}: event {} {} vs {sd}",
                    e.key,
                    e.std_error
                );
            }

            // pre-trend: sample covariance of the complete pre-cell replicates
            let pre: Vec<usize> = (0..atts.len())
                .filter(|&k| {
                    let c = &atts[k];
                    !c.post_treatment && c.time != c.group - 1 && c.std_error > 0.0
                })
                .collect();
            let rows: Vec<Vec<f64>> = reps
                .iter()
                .filter_map(|(a, _)| pre.iter().map(|&k| a[k]).collect::<Option<Vec<f64>>>())
                .collect();
            let (m, kp) = (rows.len(), pre.len());
            let mean: Vec<f64> = (0..kp)
                .map(|j| rows.iter().map(|r| r[j]).sum::<f64>() / m as f64)
                .collect();
            let mut v = Array2::<f64>::zeros((kp, kp));
            for r in &rows {
                for i in 0..kp {
                    for j in 0..kp {
                        v[[i, j]] += (r[i] - mean[i]) * (r[j] - mean[j]) / (m - 1) as f64;
                    }
                }
            }
            let vinv = crate::linalg::matrix_ops::matrix_inverse(&v.view()).unwrap();
            let a = Array1::from(pre.iter().map(|&k| atts[k].att).collect::<Vec<_>>());
            let w = a.dot(&vinv.dot(&a));
            let pt = r.pretrend_test.as_ref().unwrap();
            assert!(rel_close(pt.chi2, w, 1e-8), "{what}: W {} vs {w}", pt.chi2);
            assert_eq!(pt.df, 9, "{what}");
            // Hotelling's T^2: F = (m - 9) W / (9 (m - 1)) on F(9, m - 9)
            let f = (m - kp) as f64 * w / (kp as f64 * (m - 1) as f64);
            let p = FisherSnedecor::new(kp as f64, (m - kp) as f64)
                .unwrap()
                .sf(f);
            assert_eq!(pt.df2, Some(m - kp), "{what}");
            assert!(
                rel_close(pt.f_stat.unwrap(), f, 1e-8),
                "{what}: F {:?} vs {f}",
                pt.f_stat
            );
            assert!(
                rel_close(pt.p_value, p, 1e-8),
                "{what}: p {} vs {p}",
                pt.p_value
            );
            // the chi-squared(9) p-value ignores the error in V and is smaller
            assert!(chi_squared_p_value(w, kp as f64) < p, "{what}");
            assert_eq!(
                pt.method,
                format!(
                    "Hotelling T^2, bootstrap covariance of 9 pre-treatment ATT(g,t) ({m} of \
                     200 replicates), F(9, {})",
                    m - kp
                ),
                "{what}"
            );
        }
    }

    /// Units 3 and 17 of the unbalanced panel lack the base periods of cohorts 2002 and
    /// 2004 (CAUS-85), so only replicates that draw neither unit compute every post cell
    /// (about (18/20)^20 x 200 = 24): one warning says so and points to bootstrap = 0.
    #[test]
    fn test_staggered_did_bootstrap_reports_dropped_replicates() {
        let ds = create_unbalanced_staggered_dataset();
        let config = StaggeredDidConfig {
            bootstrap: 200,
            seed: Some(42),
            min_obs_per_cell: 3,
            ..Default::default()
        };
        let r = run_staggered_did(
            &ds,
            "outcome",
            "first_treat",
            "year",
            "unit",
            None,
            config.clone(),
        )
        .unwrap();
        let cells: Vec<(i64, i64, i64)> = r
            .group_time_atts
            .iter()
            .map(|c| (c.group, c.time, c.group - 1))
            .collect();
        let reps = reference_replicates(&ds, &config, &cells);
        let post: Vec<usize> = (0..cells.len())
            .filter(|&k| r.group_time_atts[k].post_treatment)
            .collect();
        let (_, fewest) = reference_sd(&reps, &post, |_, _| 1.0);
        assert!(fewest < 50, "{fewest}");
        let hits: Vec<&String> = r
            .warnings
            .iter()
            .filter(|w| w.contains("replicates could not compute every ATT(g,t)"))
            .collect();
        assert_eq!(hits.len(), 1, "{:?}", r.warnings);
        let w = hits[0];
        assert!(w.contains(&format!("fewest: {fewest},")), "{w}");
        assert!(w.contains("imprecise"), "{w}");
        assert!(w.contains("bootstrap = 0"), "{w}");
        assert!(r.overall_att.std_error.is_finite());
        let g2006 = r.group_effects.iter().find(|e| e.key == 2006).unwrap();
        assert!(g2006.std_error.is_finite() && g2006.std_error > 0.0);
    }

    /// With B = 999 the replicate SEs of the overall and group aggregates are within
    /// 15% of did's analytic SEs (seed 42: ratios 1.019 overall; 1.063, 0.959 and
    /// 1.022 for the groups).
    #[test]
    fn test_validate_staggered_did_bootstrap_close_to_analytic() {
        let (case, ds) = crate::test_support::load_case("did_cs_analytic");
        let cfg = StaggeredDidConfig {
            bootstrap: 999,
            seed: Some(42),
            ..analytic(ComparisonGroup::NeverTreated)
        };
        let r = run(&ds, "unit", cfg).unwrap();
        let ratio = r.overall_att.std_error / case.scalar("never_simple_se");
        assert!((ratio - 1.0).abs() < 0.15, "overall ratio {ratio}");
        let want = case.vector("never_group_se");
        assert_eq!(r.group_effects.len(), want.len());
        for (e, w) in r.group_effects.iter().zip(&want) {
            let ratio = e.std_error / w;
            assert!((ratio - 1.0).abs() < 0.15, "group {} ratio {ratio}", e.key);
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
        // never-treated units coded as infinity: the error says how to code them
        let g = i64s(&ds, "first_treat");
        let first_never = g.iter().position(|&g| g == 0).unwrap();
        let ft: Vec<f64> = g
            .iter()
            .map(|&g| if g == 0 { f64::INFINITY } else { g as f64 })
            .collect();
        let inf = replace_col(&ds, "first_treat", Column::new("first_treat".into(), ft));
        let message = expect_invalid(run(&inf, "unit", cfg()), "Column 'first_treat'");
        assert_eq!(
            message,
            format!(
                "Column 'first_treat' (the first treatment period) is inf in row {first_never} \
                 (0-based); never-treated units must be coded 0 or negative, not infinity."
            )
        );
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

    /// did::compute.att_gt: under "notyettreated" a unit is a comparison when never
    /// treated, or first treated after max(t, g - 1) and not in cohort g.
    /// R: ctrl <- G == 0 | (G > max(t, g - 1) & G != g); sum(ctrl) per cell.
    #[test]
    fn test_validate_staggered_did_notyet_comparison_sets() {
        let (case, ds) = crate::test_support::load_case("did_cs_analytic");
        let r = run(&ds, "unit", analytic(ComparisonGroup::NotYetTreated)).unwrap();
        let n = r.n_obs;
        let cmp: Vec<usize> = r.group_time_atts.iter().map(|c| c.n_comparison).collect();
        assert_eq!(
            cmp,
            [
                67, 67, 67, 67, 44, 44, 22, 22, 44, 44, 44, 44, 44, 44, 22, 22, 22, 22, 22, 22, 22,
                22, 22, 22
            ]
        );
        assert_eq!(
            r.group_time_atts
                .iter()
                .map(|c| c.n_treated)
                .collect::<Vec<_>>(),
            [vec![23; 16], vec![22; 8]].concat()
        );
        Check::new(&case)
            .vector("notyet_cell_att", &att(&r), Tol::coef(n))
            .finish();
    }

    /// R: qnorm(0.975)
    const Z95: f64 = 1.9599639845400536;

    /// CI bounds and p-values computed in R from the case (z95 <- qnorm(0.975)):
    ///   c(ag$simple_att - z95 * ag$simple_se, ag$simple_att + z95 * ag$simple_se)
    ///   2 * pnorm(-abs(ag$simple_att / ag$simple_se)); 2 * pnorm(-abs(ag$group_att / ag$group_se))
    ///   2 * pnorm(-abs(ag$event_att / ag$event_se))   (NA at the reference event time)
    struct RInference {
        overall: [f64; 3],
        group_lo: [f64; 3],
        group_hi: [f64; 3],
        group_p: [f64; 3],
        event_lo: [f64; 12],
        event_hi: [f64; 12],
        event_p: [f64; 12],
    }

    fn assert_tol(actual: f64, expected: f64, tol: Tol, what: &str) {
        assert!(
            tol.allows(actual, expected),
            "{what}: rust={actual:?} R={expected:?}"
        );
    }

    /// Pin the analytic result `r` to the case values with prefix `p` ("never" or
    /// "notyet") and to the R inference literals.
    fn check_analytic(
        case: &p2a_validation::Case,
        p: &str,
        r: &StaggeredDidResult,
        lit: &RInference,
    ) {
        let n = r.n_obs;
        let f = |g: fn(&GroupTimeATT) -> f64| r.group_time_atts.iter().map(g).collect::<Vec<_>>();
        let ge =
            |g: fn(&AggregatedEffect) -> f64| r.group_effects.iter().map(g).collect::<Vec<_>>();
        let es = |g: fn(&AggregatedEffect) -> f64| r.event_study.iter().map(g).collect::<Vec<_>>();
        let pt = r.pretrend_test.as_ref().expect("pre-trend test");
        let k = |s: &str| format!("{p}_{s}");
        Check::new(case)
            .vector(&k("cell_group"), &f(|c| c.group as f64), Tol::exact())
            .vector(&k("cell_time"), &f(|c| c.time as f64), Tol::exact())
            .vector(&k("cell_att"), &f(|c| c.att), Tol::coef(n))
            .vector(&k("cell_se"), &f(|c| c.std_error), Tol::se(n))
            .scalar(&k("simple_att"), r.overall_att.att, Tol::coef(n))
            .scalar(&k("simple_se"), r.overall_att.std_error, Tol::se(n))
            .vector(&k("group_att"), &ge(|e| e.att), Tol::coef(n))
            .vector(&k("group_se"), &ge(|e| e.std_error), Tol::se(n))
            .vector(&k("event_e"), &es(|e| e.key as f64), Tol::exact())
            .vector(&k("event_att"), &es(|e| e.att), Tol::coef(n))
            .vector(&k("event_se"), &es(|e| e.std_error), Tol::se(n))
            .scalar(&k("pretrend_w"), pt.chi2, Tol::stat(n))
            .scalar(&k("pretrend_df"), pt.df as f64, Tol::exact())
            .scalar(&k("pretrend_p"), pt.p_value, Tol::p_value(n))
            .finish();

        // cell CIs: the case's att -/+ qnorm(0.975) se
        let (catt, cse) = (case.vector(&k("cell_att")), case.vector(&k("cell_se")));
        for (i, c) in r.group_time_atts.iter().enumerate() {
            assert_tol(
                c.ci_lower,
                catt[i] - Z95 * cse[i],
                Tol::stat(n),
                "cell ci_lower",
            );
            assert_tol(
                c.ci_upper,
                catt[i] + Z95 * cse[i],
                Tol::stat(n),
                "cell ci_upper",
            );
        }
        let o = &r.overall_att;
        assert_tol(o.ci_lower, lit.overall[0], Tol::stat(n), "overall ci_lower");
        assert_tol(o.ci_upper, lit.overall[1], Tol::stat(n), "overall ci_upper");
        assert_tol(o.p_value, lit.overall[2], Tol::p_value(n), "overall p");
        for (i, e) in r.group_effects.iter().enumerate() {
            assert_tol(e.ci_lower, lit.group_lo[i], Tol::stat(n), "group ci_lower");
            assert_tol(e.ci_upper, lit.group_hi[i], Tol::stat(n), "group ci_upper");
            assert_tol(e.p_value, lit.group_p[i], Tol::p_value(n), "group p");
        }
        assert_eq!(r.event_study.len(), 12);
        for (i, e) in r.event_study.iter().enumerate() {
            assert_tol(e.ci_lower, lit.event_lo[i], Tol::stat(n), "event ci_lower");
            assert_tol(e.ci_upper, lit.event_hi[i], Tol::stat(n), "event ci_upper");
            assert_tol(e.p_value, lit.event_p[i], Tol::p_value(n), "event p");
        }

        // the pre-trend vector: the 9 pre cells (t < g, t != g - 1)
        let (cg, ct) = (case.vector(&k("cell_group")), case.vector(&k("cell_time")));
        let want: Vec<f64> = (0..cg.len())
            .filter(|&i| ct[i] < cg[i] && ct[i] != cg[i] - 1.0)
            .map(|i| catt[i])
            .collect();
        assert_eq!(pt.pre_atts.len(), 9);
        for (a, b) in pt.pre_atts.iter().zip(&want) {
            assert_tol(*a, *b, Tol::coef(n), "pre att");
        }
        assert_eq!(
            pt.method,
            "Wald, influence-function covariance of 9 pre-treatment ATT(g,t)"
        );
        assert_eq!(
            r.group_time_atts
                .iter()
                .map(|c| c.n_treated)
                .collect::<Vec<_>>(),
            [vec![23; 16], vec![22; 8]].concat()
        );
    }

    #[test]
    fn test_validate_staggered_did_analytic_never() {
        let (case, ds) = crate::test_support::load_case("did_cs_analytic");
        let r = run(&ds, "unit", analytic(ComparisonGroup::NeverTreated)).unwrap();
        check_analytic(
            &case,
            "never",
            &r,
            &RInference {
                overall: [2.1349147382930185, 2.932089075962024, 1.266087571375279e-35],
                group_lo: [2.307971699916159, 1.390528178312571, 1.8192665650244195],
                group_hi: [3.4293224273606526, 2.7778848084665793, 3.0243348158392167],
                group_p: [
                    1.148345076036387e-23,
                    3.888850310316624e-9,
                    3.3320079464744728e-15,
                ],
                event_lo: [
                    -0.42008613329171285,
                    -0.05530863020039245,
                    -0.574033871318034,
                    -0.6220950272841064,
                    -0.3723490114988929,
                    f64::NAN,
                    1.433280378525943,
                    2.1643359105209194,
                    1.9318514813537235,
                    2.1910176646242636,
                    2.469185460443568,
                    2.8522233955292493,
                ],
                event_hi: [
                    1.1785525250008035,
                    1.485343499054938,
                    0.6181163690505591,
                    0.41965228093764173,
                    0.33033273091841153,
                    f64::NAN,
                    2.082068412290233,
                    2.909839780934294,
                    3.0977159129852097,
                    3.3762731154947083,
                    4.03822832952402,
                    4.414967348692885,
                ],
                event_p: [
                    0.3524252201154725,
                    0.06887473761654846,
                    0.9422245549355032,
                    0.7032924524534246,
                    0.9067059689605106,
                    f64::NAN,
                    2.4124486336216627e-26,
                    1.3502157488096779e-40,
                    2.782941440768191e-17,
                    3.378993069389494e-20,
                    4.338779192564776e-16,
                    7.912628431054361e-20,
                ],
            },
        );
        assert!(r.group_time_atts.iter().all(|c| c.n_comparison == 22));
    }

    #[test]
    fn test_validate_staggered_did_analytic_notyet() {
        let (case, ds) = crate::test_support::load_case("did_cs_analytic");
        let r = run(&ds, "unit", analytic(ComparisonGroup::NotYetTreated)).unwrap();
        check_analytic(
            &case,
            "notyet",
            &r,
            &RInference {
                overall: [2.2060687985688765, 2.980298506458476, 2.237186220112448e-39],
                group_lo: [2.430040665610496, 1.4865028369326752, 1.8192665650244195],
                group_hi: [3.469549343544538, 2.7939619408510206, 3.0243348158392167],
                group_p: [
                    9.64043830025874e-29,
                    1.3927117449031205e-10,
                    3.3320079464744728e-15,
                ],
                event_lo: [
                    -0.42008613329171285,
                    -0.05530863020039245,
                    -0.56155407646327,
                    -0.7322408807428932,
                    -0.38378642476540636,
                    f64::NAN,
                    1.441287662531535,
                    2.1617069452922903,
                    2.0834596279422626,
                    2.3914960590560606,
                    2.469185460443568,
                    2.8522233955292493,
                ],
                event_hi: [
                    1.1785525250008035,
                    1.485343499054938,
                    0.59284280522428,
                    0.3453813101544084,
                    0.35047894447195616,
                    f64::NAN,
                    2.0840620928181175,
                    2.9877518627678716,
                    3.1735181568830337,
                    3.5333034382765476,
                    4.03822832952402,
                    4.414967348692885,
                ],
                event_p: [
                    0.3524252201154725,
                    0.06887473761654846,
                    0.9576340744181483,
                    0.48167274802730536,
                    0.9291556790255424,
                    f64::NAN,
                    5.952866395633425e-27,
                    2.486309632503913e-34,
                    3.3167833283499864e-21,
                    2.6939324338815217e-24,
                    4.338779192564776e-16,
                    7.912628431054361e-20,
                ],
            },
        );
        let cmp: Vec<usize> = r.group_time_atts.iter().map(|c| c.n_comparison).collect();
        assert_eq!(
            cmp,
            [
                67, 67, 67, 67, 44, 44, 22, 22, 44, 44, 44, 44, 44, 44, 22, 22, 22, 22, 22, 22, 22,
                22, 22, 22
            ]
        );
    }

    #[test]
    fn test_validate_staggered_did_confidence_level_90() {
        let (case, ds) = crate::test_support::load_case("did_cs_analytic");
        let cfg = StaggeredDidConfig {
            confidence_level: 0.90,
            ..analytic(ComparisonGroup::NeverTreated)
        };
        let r = run(&ds, "unit", cfg).unwrap();
        let n = r.n_obs;
        // R: qnorm(0.95)
        const Z90: f64 = 1.6448536269514715;
        assert_tol(
            r.overall_att.ci_lower,
            2.1989970103614693,
            Tol::stat(n),
            "ci_lower",
        );
        assert_tol(
            r.overall_att.ci_upper,
            2.8680068038935733,
            Tol::stat(n),
            "ci_upper",
        );
        // cell (3, 3), the third cell
        let (att, se) = (
            case.vector("never_cell_att")[2],
            case.vector("never_cell_se")[2],
        );
        let c = &r.group_time_atts[2];
        assert_eq!((c.group, c.time), (3, 3));
        assert_tol(c.ci_lower, att - Z90 * se, Tol::stat(n), "cell ci_lower");
        assert_tol(c.ci_upper, att + Z90 * se, Tol::stat(n), "cell ci_upper");
        Check::new(&case)
            .scalar("never_simple_se", r.overall_att.std_error, Tol::se(n))
            .finish();
        let shown = format!("{r}");
        assert!(
            shown.contains("(SE = 0.2034, 90% CI [2.1990, 2.8680])"),
            "{shown}"
        );
        assert!(shown.contains("90% CI"), "{shown}");
    }

    #[test]
    fn test_staggered_did_bootstrap_below_50_is_error() {
        let ds = create_staggered_dataset();
        for b in [1, 20, 49] {
            let cfg = StaggeredDidConfig {
                bootstrap: b,
                seed: Some(1),
                min_obs_per_cell: 3,
                ..Default::default()
            };
            let m = expect_invalid(
                run_staggered_did(&ds, "outcome", "first_treat", "year", "unit", None, cfg),
                "bootstrap",
            );
            assert!(m.contains("at least 50"), "{m}");
            assert!(m.contains("0 for analytic"), "{m}");
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // T3-WP21: covariates (DRDID), anticipation, no post cells; the R case
    // did_cs_covariates (validation/r/cases/causal_did_cov.R)
    // ─────────────────────────────────────────────────────────────────────────

    fn cfg(cg: ComparisonGroup, m: AttEstimationMethod, anticipation: usize) -> StaggeredDidConfig {
        StaggeredDidConfig {
            comparison_group: cg,
            estimation_method: m,
            anticipation,
            bootstrap: 0,
            ..Default::default()
        }
    }

    fn run_cov(ds: &Dataset, covs: Option<&[&str]>, c: StaggeredDidConfig) -> StaggeredDidResult {
        run_staggered_did(ds, "y", "first_treat", "time", "unit", covs, c).unwrap()
    }

    fn cell_vec(r: &StaggeredDidResult, f: impl Fn(&GroupTimeATT) -> f64) -> Vec<f64> {
        r.group_time_atts.iter().map(f).collect()
    }

    /// Every cell, aggregate and pre-test key of one `<tag>_*_<m>` block of
    /// did_cs_covariates.
    fn pin_block(ck: &mut Check<'_>, tag: &str, m: &str, r: &StaggeredDidResult, iterative: bool) {
        let n = r.n_obs;
        let (a, s, st) = if iterative {
            (
                Tol::iterative(1e-8),
                Tol::iterative(1e-8),
                Tol::iterative(1e-8),
            )
        } else {
            (Tol::coef(n), Tol::se(n), Tol::stat(n))
        };
        let pt = r.pretrend_test.as_ref().expect("pre-trend test");
        let ev = |f: fn(&AggregatedEffect) -> f64| r.event_study.iter().map(f).collect::<Vec<_>>();
        let gr =
            |f: fn(&AggregatedEffect) -> f64| r.group_effects.iter().map(f).collect::<Vec<_>>();
        ck.vector(
            &format!("{tag}_cell_group"),
            &cell_vec(r, |c| c.group as f64),
            Tol::exact(),
        )
        .vector(
            &format!("{tag}_cell_time"),
            &cell_vec(r, |c| c.time as f64),
            Tol::exact(),
        )
        .vector(
            &format!("{tag}_cell_n_treated"),
            &cell_vec(r, |c| c.n_treated as f64),
            Tol::exact(),
        )
        .vector(
            &format!("{tag}_cell_n_comparison"),
            &cell_vec(r, |c| c.n_comparison as f64),
            Tol::exact(),
        )
        .vector(&format!("{tag}_cell_att_{m}"), &cell_vec(r, |c| c.att), a)
        .vector(
            &format!("{tag}_cell_se_{m}"),
            &cell_vec(r, |c| c.std_error),
            s,
        )
        .scalar(&format!("{tag}_simple_att_{m}"), r.overall_att.att, a)
        .scalar(&format!("{tag}_simple_se_{m}"), r.overall_att.std_error, s)
        .vector(&format!("{tag}_group_att_{m}"), &gr(|e| e.att), a)
        .vector(&format!("{tag}_group_se_{m}"), &gr(|e| e.std_error), s)
        .vector(
            &format!("{tag}_event_e"),
            &ev(|e| e.key as f64),
            Tol::exact(),
        )
        .vector(&format!("{tag}_event_att_{m}"), &ev(|e| e.att), a)
        .vector(&format!("{tag}_event_se_{m}"), &ev(|e| e.std_error), s)
        .scalar(&format!("{tag}_pretrend_w_{m}"), pt.chi2, st)
        .scalar(
            &format!("{tag}_pretrend_df_{m}"),
            pt.df as f64,
            Tol::exact(),
        )
        .scalar(
            &format!("{tag}_pretrend_p_{m}"),
            pt.p_value,
            Tol::p_value(n),
        );
    }

    const X12: &[&str] = &["x1", "x2"];
    const X13: &[&str] = &["x1", "x3"];

    /// The case panel `cs_cov_panel` for the tests that are not R-pinned (not
    /// registered).
    fn cov_panel() -> Dataset {
        crate::regression::ols::na_fixtures::repo_dataset(
            "validation/datasets/generated/cs_cov_panel.csv",
        )
    }

    /// `ds` with column `name` set to `f(row)`.
    fn with_column(ds: &Dataset, name: &str, f: impl Fn(usize) -> f64) -> Dataset {
        let mut df = ds.df().clone();
        let v: Vec<f64> = (0..df.height()).map(f).collect();
        df.with_column(Column::new(name.into(), v)).unwrap();
        Dataset::new(df)
    }

    /// A float column of `ds` (test-only read of a null-free column).
    fn f64s(ds: &Dataset, col: &str) -> Vec<f64> {
        let c = ds
            .df()
            .column(col)
            .unwrap()
            .cast(&DataType::Float64)
            .unwrap();
        c.f64().unwrap().iter().map(|v| v.unwrap()).collect()
    }

    /// CAUS-34: outcome regression with covariates is DRDID's reg_did_panel. R:
    /// `.dc_attgt(d, "never", 0)` with `m = "or"`. Before: bootstrap = 0 was
    /// NotImplemented and the bootstrap path ignored the covariates ((3,3) 0.4023
    /// against 0.3862).
    #[test]
    fn test_validate_cs_covariates_or_never() {
        let (case, ds) = crate::test_support::load_case("did_cs_covariates");
        use AttEstimationMethod::OutcomeRegression;
        let r = run_cov(
            &ds,
            Some(X12),
            cfg(ComparisonGroup::NeverTreated, OutcomeRegression, 0),
        );
        assert_eq!(r.n_obs, 1200);
        assert_eq!(r.covariates, vec!["x1".to_string(), "x2".to_string()]);
        let mut ck = Check::new(&case);
        pin_block(&mut ck, "never_ant0", "or", &r, false);
        ck.finish();
        let shown = format!("{r}");
        assert!(
            shown
                .contains("Covariates:         x1, x2 (values from the earlier of the two periods"),
            "{shown}"
        );
        assert!(shown.contains("Anticipation:       none"), "{shown}");
    }

    #[test]
    fn test_validate_cs_covariates_or_notyet() {
        let (case, ds) = crate::test_support::load_case("did_cs_covariates");
        use AttEstimationMethod::OutcomeRegression;
        let r = run_cov(
            &ds,
            Some(X12),
            cfg(ComparisonGroup::NotYetTreated, OutcomeRegression, 0),
        );
        let mut ck = Check::new(&case);
        pin_block(&mut ck, "notyet_ant0", "or", &r, false);
        ck.finish();
    }

    /// CAUS-35: each cell takes the covariates of the earlier of its two periods.
    /// The period-t rule (the code before the fix) is recorded in the case.
    #[test]
    fn test_validate_cs_covariates_or_time_varying() {
        let (case, ds) = crate::test_support::load_case("did_cs_covariates");
        use AttEstimationMethod::OutcomeRegression;
        let r = run_cov(
            &ds,
            Some(X13),
            cfg(ComparisonGroup::NeverTreated, OutcomeRegression, 0),
        );
        let n = r.n_obs;
        Check::new(&case)
            .vector(
                "tv_never_ant0_cell_att_or",
                &cell_vec(&r, |c| c.att),
                Tol::coef(n),
            )
            .vector(
                "tv_never_ant0_cell_se_or",
                &cell_vec(&r, |c| c.std_error),
                Tol::se(n),
            )
            .finish();
        let c33 = r
            .group_time_atts
            .iter()
            .position(|c| (c.group, c.time) == (3, 3))
            .unwrap();
        let period_t = case.vector("tv_period_t_cell_att_or")[c33];
        assert!((r.group_time_atts[c33].att - period_t).abs() > 0.05);
    }

    /// Without covariates the three methods are the unconditional estimator (did
    /// with xformla = ~1).
    #[test]
    fn test_validate_cs_methods_coincide_without_covariates() {
        let (case, ds) = crate::test_support::load_case("did_cs_covariates");
        for m in [
            AttEstimationMethod::OutcomeRegression,
            AttEstimationMethod::IPW,
            AttEstimationMethod::DoublyRobust,
        ] {
            let r = run_cov(&ds, None, cfg(ComparisonGroup::NeverTreated, m, 0));
            assert!(r.covariates.is_empty());
            let n = r.n_obs;
            Check::new(&case)
                .vector(
                    "nocov_never_ant0_cell_att",
                    &cell_vec(&r, |c| c.att),
                    Tol::coef(n),
                )
                .vector(
                    "nocov_never_ant0_cell_se",
                    &cell_vec(&r, |c| c.std_error),
                    Tol::se(n),
                )
                .finish();
            assert!(format!("{r}").contains("Covariates:         none"));
        }
    }

    /// CAUS-37, CAUS-35: IPW and DR are DRDID's std_ipw_did_panel and drdid_panel
    /// (normalised weights, one intercept, earlier-period covariates). R fits the
    /// propensity score at epsilon 1e-14, p2a's IRLS stops at 1e-10: measured gap
    /// 8.8e-13 relative, so Tol::iterative(1e-8).
    #[test]
    fn test_validate_cs_covariates_ipw_dr_cells() {
        let (case, ds) = crate::test_support::load_case("did_cs_covariates");
        let mut ck = Check::new(&case);
        for (cg, tag) in [
            (ComparisonGroup::NeverTreated, "never_ant0"),
            (ComparisonGroup::NotYetTreated, "notyet_ant0"),
        ] {
            for (m, key) in [
                (AttEstimationMethod::IPW, "ipw"),
                (AttEstimationMethod::DoublyRobust, "dr"),
            ] {
                let r = run_cov(&ds, Some(X12), cfg(cg, m, 0));
                pin_block(&mut ck, tag, key, &r, true);
                assert!(
                    !r.warnings.iter().any(|w| w.contains("did not converge")),
                    "{:?}",
                    r.warnings
                );
            }
        }
        ck.finish();
    }

    #[test]
    fn test_validate_cs_covariates_ipw_dr_time_varying() {
        let (case, ds) = crate::test_support::load_case("did_cs_covariates");
        let mut ck = Check::new(&case);
        for (m, key) in [
            (AttEstimationMethod::IPW, "ipw"),
            (AttEstimationMethod::DoublyRobust, "dr"),
        ] {
            let r = run_cov(&ds, Some(X13), cfg(ComparisonGroup::NeverTreated, m, 0));
            ck.vector(
                &format!("tv_never_ant0_cell_att_{key}"),
                &cell_vec(&r, |c| c.att),
                Tol::iterative(1e-8),
            )
            .vector(
                &format!("tv_never_ant0_cell_se_{key}"),
                &cell_vec(&r, |c| c.std_error),
                Tol::iterative(1e-8),
            );
        }
        ck.finish();
    }

    /// The same blocks against did 2.5.1 itself (att_gt and aggte with DRDID 1.4.0;
    /// the `did_*` keys). did fits the propensity score with its own stopping rule,
    /// so the comparison is at 1e-6 relative; the transcription in the case is the
    /// tight reference (the tests above).
    #[test]
    fn test_validate_cs_covariates_match_did() {
        use AttEstimationMethod::{DoublyRobust, IPW, OutcomeRegression};
        use ComparisonGroup::{NeverTreated, NotYetTreated};
        let (case, ds) = crate::test_support::load_case("did_cs_covariates");
        let tol = Tol::iterative(1e-6);
        let mut ck = Check::new(&case);
        for (cg, delta, tag) in [
            (NeverTreated, 0, "never_ant0"),
            (NotYetTreated, 0, "notyet_ant0"),
            (NeverTreated, 1, "never_ant1"),
            (NotYetTreated, 1, "notyet_ant1"),
        ] {
            for (m, key) in [
                (OutcomeRegression, "or"),
                (IPW, "ipw"),
                (DoublyRobust, "dr"),
            ] {
                let r = run_cov(&ds, Some(X12), cfg(cg, m, delta));
                let ev = |f: fn(&AggregatedEffect) -> f64| {
                    r.event_study.iter().map(f).collect::<Vec<_>>()
                };
                let gr = |f: fn(&AggregatedEffect) -> f64| {
                    r.group_effects.iter().map(f).collect::<Vec<_>>()
                };
                let p = format!("did_{tag}");
                ck.vector(
                    &format!("{p}_cell_att_{key}"),
                    &cell_vec(&r, |c| c.att),
                    tol,
                )
                .vector(
                    &format!("{p}_cell_se_{key}"),
                    &cell_vec(&r, |c| c.std_error),
                    tol,
                )
                .vector(&format!("{p}_simple_att_{key}"), &[r.overall_att.att], tol)
                .vector(
                    &format!("{p}_simple_se_{key}"),
                    &[r.overall_att.std_error],
                    tol,
                )
                .vector(&format!("{p}_group_att_{key}"), &gr(|e| e.att), tol)
                .vector(&format!("{p}_group_se_{key}"), &gr(|e| e.std_error), tol)
                .vector(&format!("{p}_event_att_{key}"), &ev(|e| e.att), tol)
                .vector(&format!("{p}_event_se_{key}"), &ev(|e| e.std_error), tol);
            }
        }
        for (m, key) in [
            (OutcomeRegression, "or"),
            (IPW, "ipw"),
            (DoublyRobust, "dr"),
        ] {
            let r = run_cov(&ds, Some(X13), cfg(NeverTreated, m, 0));
            ck.vector(
                &format!("did_tv_never_ant0_cell_att_{key}"),
                &cell_vec(&r, |c| c.att),
                tol,
            )
            .vector(
                &format!("did_tv_never_ant0_cell_se_{key}"),
                &cell_vec(&r, |c| c.std_error),
                tol,
            );
        }
        ck.finish();
    }

    /// CAUS-37's failure scenario: adding a common trend to the outcome changes no
    /// DR cell (the pre-fix code added a second intercept and moved with it).
    #[test]
    fn test_cs_dr_invariant_to_common_trend() {
        let ds = cov_panel();
        let (y, t) = (f64s(&ds, "y"), f64s(&ds, "time"));
        let trended = with_column(&ds, "y", |i| y[i] + 10.0 * t[i]);
        let c = || {
            cfg(
                ComparisonGroup::NeverTreated,
                AttEstimationMethod::DoublyRobust,
                0,
            )
        };
        let a = run_cov(&ds, Some(X12), c());
        let b = run_cov(&trended, Some(X12), c());
        for (x, y) in a.group_time_atts.iter().zip(&b.group_time_atts) {
            for (p, q) in [(x.att, y.att), (x.std_error, y.std_error)] {
                assert!(
                    same(p, q) || (p - q).abs() <= 1e-10 * p.abs().max(1.0),
                    "{x:?} vs {y:?}"
                );
            }
        }
    }

    /// Collinear covariates are refused (did: NA for every cell), not fitted by a
    /// generalized inverse.
    #[test]
    fn test_cs_ipw_dr_refuse_collinear_covariates() {
        let ds = cov_panel();
        let x1 = f64s(&ds, "x1");
        let ds = with_column(&ds, "x1b", |i| 2.0 * x1[i]);
        for m in [
            AttEstimationMethod::OutcomeRegression,
            AttEstimationMethod::IPW,
            AttEstimationMethod::DoublyRobust,
        ] {
            let res = run_staggered_did(
                &ds,
                "y",
                "first_treat",
                "time",
                "unit",
                Some(&["x1", "x1b"]),
                cfg(ComparisonGroup::NeverTreated, m, 0),
            );
            match res {
                Err(EconError::InsufficientData { context, .. }) => {
                    assert!(context.contains("singular"), "{m}: {context}")
                }
                other => panic!("{m}: expected InsufficientData, got {other:?}"),
            }
        }
    }

    /// did's overlap guard: a covariate that separates the cohort from the
    /// comparison units makes every cell NA. R's glm.fit converges (iteration 31)
    /// with a largest fitted value 0.99999999999999978.
    #[test]
    fn test_cs_overlap_guard_skips_cell() {
        let (mut unit, mut time, mut ft, mut z, mut y) = (vec![], vec![], vec![], vec![], vec![]);
        for u in 0..60i64 {
            let (g, k) = if u < 30 { (4, u) } else { (0, u - 30) };
            for t in 1..=6i64 {
                unit.push(u);
                time.push(t);
                ft.push(g);
                let zz = if g > 0 { 3.0 } else { -3.0 } + 0.1 * k as f64;
                z.push(zz);
                y.push(0.5 * u as f64 + 0.3 * t as f64 + 0.2 * ((7 * u + 3 * t) % 5) as f64);
            }
        }
        let df = DataFrame::new(vec![
            Column::new("unit".into(), unit),
            Column::new("time".into(), time),
            Column::new("first_treat".into(), ft),
            Column::new("z".into(), z),
            Column::new("y".into(), y),
        ])
        .unwrap();
        let config = StaggeredDidConfig {
            min_obs_per_cell: 3,
            ..cfg(ComparisonGroup::NeverTreated, AttEstimationMethod::IPW, 0)
        };
        match run_staggered_did(
            &Dataset::new(df),
            "y",
            "first_treat",
            "time",
            "unit",
            Some(&["z"]),
            config,
        ) {
            Err(EconError::InsufficientData { context, .. }) => {
                assert!(context.contains("overlap condition violated"), "{context}")
            }
            other => panic!("expected InsufficientData, got {other:?}"),
        }
    }

    /// did's absolute NA rule (SE <= 1.49e-7) applies to every cell and aggregate;
    /// outside the reference cells it is announced. Before, an outcome in very small
    /// units got NaN SEs with no note.
    #[test]
    fn test_cs_tiny_standard_errors_are_announced() {
        let ds = cov_panel();
        let y = f64s(&ds, "y");
        let small = with_column(&ds, "y", |i| y[i] * 1e-9);
        let m = AttEstimationMethod::OutcomeRegression;
        let note = |r: &StaggeredDidResult| {
            r.warnings
                .iter()
                .find(|w| w.contains("reported as NA (did's rule)"))
                .cloned()
        };
        let base = run_cov(&ds, None, cfg(ComparisonGroup::NeverTreated, m, 0));
        assert!(note(&base).is_none(), "{:?}", base.warnings);
        let r = run_cov(&small, None, cfg(ComparisonGroup::NeverTreated, m, 0));
        assert!(r.overall_att.std_error.is_nan());
        let w = note(&r).unwrap_or_else(|| panic!("{:?}", r.warnings));
        assert!(w.contains("1.49e-7"), "{w}");
        // every non-reference cell and every aggregate except the reference event time
        let cells = r
            .group_time_atts
            .iter()
            .filter(|c| c.relative_time != -1)
            .count();
        let aggs = r.event_study.iter().filter(|e| e.key != -1).count() + r.group_effects.len() + 1;
        assert!(
            w.starts_with(&format!("{} standard error(s)", cells + aggs)),
            "{w}"
        );
    }

    /// A resource limit or a cancellation inside a cell fit is the call's error;
    /// before, it was stored in the cell's slot and the cell was dropped with a
    /// "Could not compute ATT(g,t)" warning, as if the cell's data were the problem.
    /// A data failure stays a per-cell `Err`.
    #[test]
    fn test_cs_covariate_fits_propagate_resource_errors() {
        let ds = cov_panel();
        let col = |c: &str| Array1::from(f64s(&ds, c));
        let (y, g, t, u) = (col("y"), col("first_treat"), col("time"), col("unit"));
        let x = ndarray::stack(ndarray::Axis(1), &[col("x1").view(), col("x2").view()]).unwrap();
        let panel_map = build_panel_map(&t, &u);
        let cells: Vec<(i64, i64, i64)> = vec![(3, 3, 2), (3, 4, 2), (4, 4, 3)];
        let table = build_boot_table(
            &y,
            &g,
            &t,
            &u,
            &panel_map,
            &cells,
            ComparisonGroup::NeverTreated,
            0,
        )
        .unwrap();
        let data_err = |_: &drdid::CellInput<'_>| -> EconResult<drdid::CellFit> {
            Err(EconError::InsufficientData {
                required: 2,
                provided: 1,
                context: "test cell".to_string(),
            })
        };
        let slots = covariate_fits_with(&table, &cells, &x, data_err).unwrap();
        assert_eq!(slots.len(), 3);
        assert!(
            slots
                .iter()
                .all(|r| matches!(r, Err(EconError::InsufficientData { .. })))
        );
        let limit = |_: &drdid::CellInput<'_>| -> EconResult<drdid::CellFit> {
            Err(EconError::ResourceLimit("glm IRLS workspace".to_string()))
        };
        assert!(matches!(
            covariate_fits_with(&table, &cells, &x, limit),
            Err(EconError::ResourceLimit(_))
        ));
        let cancelled = |_: &drdid::CellInput<'_>| -> EconResult<drdid::CellFit> {
            Err(EconError::Cancelled("deadline".to_string()))
        };
        assert!(matches!(
            covariate_fits_with(&table, &cells, &x, cancelled),
            Err(EconError::Cancelled(_))
        ));
        // End to end under a tiny allocation ceiling: the resource error, not a
        // partial result or an InsufficientData "no cell left".
        let r = crate::resource::with_alloc_ceiling(Some(64), || {
            run_staggered_did(
                &ds,
                "y",
                "first_treat",
                "time",
                "unit",
                Some(X12),
                cfg(ComparisonGroup::NeverTreated, AttEstimationMethod::IPW, 0),
            )
        });
        assert!(matches!(r, Err(EconError::ResourceLimit(_))), "{r:?}");
    }

    /// CAUS-36: with covariates every bootstrap replicate is the multiplier draw
    /// of the cell's own influence function (T0's random stream), not the
    /// unconditional estimator's.
    #[test]
    fn test_staggered_did_covariate_bootstrap_draws_are_linearised() {
        let ds = cov_panel();
        let col = |c: &str| Array1::from(f64s(&ds, c));
        let (y, g, t, u) = (col("y"), col("first_treat"), col("time"), col("unit"));
        let x = ndarray::stack(ndarray::Axis(1), &[col("x1").view(), col("x2").view()]).unwrap();
        let panel_map = build_panel_map(&t, &u);
        let point = run_cov(
            &ds,
            Some(X12),
            StaggeredDidConfig {
                min_obs_per_cell: 1,
                ..cfg(
                    ComparisonGroup::NeverTreated,
                    AttEstimationMethod::OutcomeRegression,
                    0,
                )
            },
        );
        let cells: Vec<(i64, i64, i64)> = point
            .group_time_atts
            .iter()
            .map(|c| (c.group, c.time, c.group - 1))
            .collect();
        for m in [
            AttEstimationMethod::OutcomeRegression,
            AttEstimationMethod::IPW,
            AttEstimationMethod::DoublyRobust,
        ] {
            let mut table = build_boot_table(
                &y,
                &g,
                &t,
                &u,
                &panel_map,
                &cells,
                ComparisonGroup::NeverTreated,
                0,
            )
            .unwrap();
            let fits: Vec<drdid::CellFit> = covariate_fits(&table, &cells, &x, m)
                .unwrap()
                .into_iter()
                .map(|f| f.unwrap())
                .collect();
            table.linear = Some(Linearised::new(&table, &fits).unwrap());
            let draws = bootstrap_draws(&table, 3, &mut StdRng::seed_from_u64(7)).unwrap();
            // replay T0's stream and recompute every draw from independent fits
            let n_units = table.n_units;
            let mut rng = StdRng::seed_from_u64(7);
            for b in 0..3 {
                let mut counts = vec![0usize; n_units];
                for _ in 0..n_units {
                    counts[rng.gen_range(0..n_units)] += 1;
                }
                for (c, &(_, ct, base)) in cells.iter().enumerate() {
                    let entries = &table.entries[table.starts[c]..table.starts[c + 1]];
                    let (att, inf) = if ct == base {
                        (0.0, vec![0.0; entries.len()])
                    } else {
                        let dy: Array1<f64> = entries.iter().map(|e| e.dy).collect();
                        let d: Array1<f64> = entries
                            .iter()
                            .map(|e| f64::from(u8::from(e.treated)))
                            .collect();
                        let mut xm = Array2::<f64>::ones((entries.len(), 3));
                        for (i, e) in entries.iter().enumerate() {
                            xm[[i, 1]] = x[[e.xrow, 0]];
                            xm[[i, 2]] = x[[e.xrow, 1]];
                        }
                        let input = drdid::CellInput {
                            dy: dy.view(),
                            d: d.view(),
                            x: xm.view(),
                        };
                        let f = drdid::estimate(m, &input).unwrap();
                        (f.att, f.influence.to_vec())
                    };
                    let scale = n_units as f64 / entries.len() as f64;
                    let want = att
                        + entries
                            .iter()
                            .zip(&inf)
                            .map(|(e, v)| (counts[e.unit] as f64 - 1.0) * scale * v)
                            .sum::<f64>()
                            / n_units as f64;
                    let got = draws.cells[[b, c]];
                    assert!(
                        (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                        "{m} cell {c}: {got} vs {want}"
                    );
                }
            }
            // the first non-reference covariate cell is not the unconditional draw
            assert_eq!(
                (
                    point.group_time_atts[2].group,
                    point.group_time_atts[2].time
                ),
                (3, 3)
            );
        }
        // the bootstrap keeps the point estimates bitwise
        let c0 = cfg(
            ComparisonGroup::NeverTreated,
            AttEstimationMethod::DoublyRobust,
            0,
        );
        let a = run_cov(&ds, Some(X12), c0.clone());
        let b = run_cov(
            &ds,
            Some(X12),
            StaggeredDidConfig {
                bootstrap: 50,
                seed: Some(1),
                ..c0
            },
        );
        assert_eq!(att(&a), att(&b));
        assert!(format!("{b}").contains("multiplier bootstrap of the influence functions"));
    }

    /// The multiplier bootstrap estimates the analytic SE up to Monte Carlo error
    /// (about 1/sqrt(2B) = 2.2% at B = 999). Before: the replicates were the
    /// unconditional estimator, simple SE about 0.256 (ratio 1.507). A Monte Carlo
    /// band against the analytic SEs (R-pinned by the tests above), not an R pin:
    /// did's own bootstrap draws (Mammen multipliers) are not reproduced.
    #[test]
    fn test_cs_covariates_bootstrap_close_to_analytic() {
        let ds = cov_panel();
        let c0 = cfg(
            ComparisonGroup::NeverTreated,
            AttEstimationMethod::DoublyRobust,
            0,
        );
        let analytic = run_cov(&ds, Some(X12), c0.clone());
        let c = StaggeredDidConfig {
            bootstrap: 999,
            seed: Some(42),
            ..c0
        };
        let r = run_cov(&ds, Some(X12), c);
        let ratio = r.overall_att.std_error / analytic.overall_att.std_error;
        assert!((ratio - 1.0).abs() < 0.10, "overall ratio {ratio}");
        assert_eq!(r.group_effects.len(), analytic.group_effects.len());
        for (e, w) in r.group_effects.iter().zip(&analytic.group_effects) {
            let ratio = e.std_error / w.std_error;
            assert!((ratio - 1.0).abs() < 0.10, "group {} ratio {ratio}", e.key);
        }
    }

    /// CAUS-83: anticipation = 1 moves the base period to g - 2 and tightens the
    /// not-yet-treated comparison set, as did::att_gt. Before: ignored ((3,3) 0.386
    /// against 1.508).
    #[test]
    fn test_validate_cs_anticipation() {
        use AttEstimationMethod::{DoublyRobust, OutcomeRegression};
        let (case, ds) = crate::test_support::load_case("did_cs_covariates");
        let mut ck = Check::new(&case);
        let r = run_cov(
            &ds,
            Some(X12),
            cfg(ComparisonGroup::NeverTreated, OutcomeRegression, 1),
        );
        pin_block(&mut ck, "never_ant1", "or", &r, false);
        let pt = r.pretrend_test.as_ref().unwrap();
        assert!(pt.method.contains("anticipation"), "{}", pt.method);
        // did's pre-test set: on integer periods a re-parametrisation of anticipation 0
        ck.scalar("never_ant0_pretrend_w_or", pt.chi2, Tol::stat(r.n_obs));
        let shown = format!("{r}");
        assert!(shown.contains("Anticipation:"), "{shown}");
        assert!(shown.contains("1 period(s) (base period g - 2"), "{shown}");
        let r = run_cov(
            &ds,
            Some(X12),
            cfg(ComparisonGroup::NotYetTreated, OutcomeRegression, 1),
        );
        pin_block(&mut ck, "notyet_ant1", "or", &r, false);
        let r = run_cov(
            &ds,
            Some(X12),
            cfg(ComparisonGroup::NeverTreated, DoublyRobust, 1),
        );
        pin_block(&mut ck, "never_ant1", "dr", &r, true);
        ck.finish();
    }

    /// The Anticipation line prints the base period the estimator uses,
    /// g + base_period - anticipation, not g - 1 - anticipation.
    #[test]
    fn test_cs_anticipation_display_follows_base_period() {
        let mut c = cfg(
            ComparisonGroup::NeverTreated,
            AttEstimationMethod::OutcomeRegression,
            1,
        );
        c.base_period = -2;
        let r = run_cov(&cov_panel(), Some(X12), c);
        let shown = format!("{r}");
        assert!(shown.contains("Base Period:        -2"), "{shown}");
        assert!(shown.contains("1 period(s) (base period g - 3;"), "{shown}");
        assert_eq!(base_period_label(0), "g");
        assert_eq!(base_period_label(2), "g + 2");
    }

    #[test]
    fn test_cs_anticipation_zero_and_one_differ() {
        let ds = cov_panel();
        let c33 = |r: &StaggeredDidResult| {
            r.group_time_atts
                .iter()
                .find(|c| (c.group, c.time) == (3, 3))
                .unwrap()
                .att
        };
        let m = AttEstimationMethod::OutcomeRegression;
        let a = run_cov(&ds, Some(X12), cfg(ComparisonGroup::NeverTreated, m, 0));
        let b = run_cov(&ds, Some(X12), cfg(ComparisonGroup::NeverTreated, m, 1));
        assert!((c33(&a) - c33(&b)).abs() > 1.0, "{} {}", c33(&a), c33(&b));
    }

    #[test]
    fn test_cs_anticipation_out_of_range_is_error() {
        let ds = cov_panel();
        for a in [usize::MAX, 8] {
            let m = expect_invalid(
                run_staggered_did(
                    &ds,
                    "y",
                    "first_treat",
                    "time",
                    "unit",
                    None,
                    cfg(
                        ComparisonGroup::NeverTreated,
                        AttEstimationMethod::OutcomeRegression,
                        a,
                    ),
                ),
                "anticipation",
            );
            assert!(m.contains("smaller than the number of periods (8)"), "{m}");
        }
    }

    #[test]
    fn test_cs_anticipation_skips_cohort_without_base() {
        let ds = cov_panel();
        let r = run_cov(
            &ds,
            None,
            cfg(
                ComparisonGroup::NeverTreated,
                AttEstimationMethod::OutcomeRegression,
                2,
            ),
        );
        assert!(r.group_time_atts.iter().all(|c| c.group != 3));
        let w = "cohort 3: base period 0 (g + base_period - anticipation) is not observed; its \
                 ATT(g,t) are not computed (did drops such cohorts)";
        assert!(r.warnings.iter().any(|x| x == w), "{:?}", r.warnings);
    }

    /// 12 never-treated units and 12 first treated at 6, observed in t = 1..=5;
    /// with `with_g3`, 12 more first treated at 3.
    fn panel_g6_in_1_to_5(with_g3: bool) -> Dataset {
        let (mut unit, mut time, mut ft, mut y) = (vec![], vec![], vec![], vec![]);
        let groups: &[i64] = if with_g3 { &[0, 6, 3] } else { &[0, 6] };
        let mut u = 0i64;
        for &g in groups {
            for _ in 0..12 {
                u += 1;
                for t in 1..=5i64 {
                    unit.push(u);
                    time.push(t);
                    ft.push(g);
                    let effect = if g > 0 && t >= g { 1.5 } else { 0.0 };
                    y.push(
                        0.5 * u as f64
                            + 0.3 * t as f64
                            + 0.2 * ((7 * u + 3 * t) % 5) as f64
                            + effect,
                    );
                }
            }
        }
        Dataset::new(
            DataFrame::new(vec![
                Column::new("unit".into(), unit),
                Column::new("time".into(), time),
                Column::new("first_treat".into(), ft),
                Column::new("y".into(), y),
            ])
            .unwrap(),
        )
    }

    /// CAUS-109: no post-treatment cell is an error, not "ATT = 0 (SE = 0), p = 1".
    #[test]
    fn test_cs_no_post_cells_is_error() {
        let c = StaggeredDidConfig {
            min_obs_per_cell: 3,
            ..cfg(
                ComparisonGroup::NeverTreated,
                AttEstimationMethod::OutcomeRegression,
                0,
            )
        };
        match run_staggered_did(
            &panel_g6_in_1_to_5(false),
            "y",
            "first_treat",
            "time",
            "unit",
            None,
            c,
        ) {
            Err(EconError::InsufficientData { context, .. }) => {
                assert!(context.contains("no post-treatment ATT(g,t)"), "{context}");
                assert!(
                    context.contains("first treated after the last observed period 5: [6]"),
                    "{context}"
                );
            }
            other => panic!("expected InsufficientData, got {other:?}"),
        }
    }

    #[test]
    fn test_cs_cohort_without_post_cells_left_out_of_groups() {
        let c = StaggeredDidConfig {
            min_obs_per_cell: 3,
            ..cfg(
                ComparisonGroup::NeverTreated,
                AttEstimationMethod::OutcomeRegression,
                0,
            )
        };
        let r = run_staggered_did(
            &panel_g6_in_1_to_5(true),
            "y",
            "first_treat",
            "time",
            "unit",
            None,
            c,
        )
        .unwrap();
        assert_eq!(
            r.group_effects.iter().map(|e| e.key).collect::<Vec<_>>(),
            [3]
        );
        let w = "cohort 6 has no post-treatment ATT(g,t) and is left out of the group aggregation \
                 (as did::aggte)";
        assert!(r.warnings.iter().any(|x| x == w), "{:?}", r.warnings);
        assert_eq!(r.overall_att.n_cells, 3);
    }

    #[test]
    fn test_comparison_group_parses_aliases_and_rejects_unknown() {
        ComparisonGroup::choices().validate().unwrap();
        for s in ["never_treated", "nevertreated", "Never", "NEVER-TREATED"] {
            assert_eq!(
                s.parse::<ComparisonGroup>().unwrap(),
                ComparisonGroup::NeverTreated
            );
        }
        for s in [
            "not_yet_treated",
            "notyettreated",
            "notyet",
            "nyt",
            "Not Yet Treated",
        ] {
            assert_eq!(
                s.parse::<ComparisonGroup>().unwrap(),
                ComparisonGroup::NotYetTreated
            );
        }
        let e = "always".parse::<ComparisonGroup>().unwrap_err().to_string();
        assert!(
            e.contains("comparison_group")
                && e.contains("never_treated")
                && e.contains("not_yet_treated"),
            "{e}"
        );
        // Display is unchanged
        assert_eq!(ComparisonGroup::NeverTreated.to_string(), "Never-Treated");
    }

    #[test]
    fn test_att_estimation_method_parses_aliases_and_rejects_unknown() {
        use AttEstimationMethod::{DoublyRobust, IPW, OutcomeRegression};
        AttEstimationMethod::choices().validate().unwrap();
        for (s, m) in [
            ("or", OutcomeRegression),
            ("reg", OutcomeRegression),
            ("outcome_regression", OutcomeRegression),
            ("IPW", IPW),
            ("std_ipw", IPW),
            ("dr", DoublyRobust),
            ("doubly-robust", DoublyRobust),
            ("aipw", DoublyRobust),
            ("doublyrobust", DoublyRobust),
        ] {
            assert_eq!(s.parse::<AttEstimationMethod>().unwrap(), m, "{s}");
        }
        let e = "tmle"
            .parse::<AttEstimationMethod>()
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("estimation_method") && e.contains("doubly_robust"),
            "{e}"
        );
    }

    /// CAUS-117: did reports NA for the SE of a reference cell (t = g - 1 - anticipation)
    /// and of the reference event time; the cell's ATT is 0 by construction. Before: SE 0
    /// and CI [0, 0] (zero influence function, zero bootstrap spread). The case's
    /// `never_cell_se` and `never_event_se` hold R's NA (did's rule) at those positions.
    #[test]
    fn test_validate_staggered_did_reference_cells_have_no_se() {
        let (case, ds) = crate::test_support::load_case("did_cs_analytic");
        for bootstrap in [0, 199] {
            let config = StaggeredDidConfig {
                bootstrap,
                seed: Some(7),
                ..Default::default()
            };
            let r = run(&ds, "unit", config).unwrap();
            let mut seen = 0;
            for c in &r.group_time_atts {
                if c.time == c.group - 1 {
                    seen += 1;
                    assert_eq!(c.att, 0.0);
                    assert!(
                        c.std_error.is_nan() && c.ci_lower.is_nan() && c.ci_upper.is_nan(),
                        "{c:?}"
                    );
                } else {
                    assert!(c.std_error.is_finite() && c.std_error > 0.0, "{c:?}");
                }
            }
            assert_eq!(seen, 3);
            let e = r.event_study.iter().find(|e| e.key == -1).unwrap();
            assert_eq!(e.att, 0.0);
            assert!(e.std_error.is_nan() && e.ci_lower.is_nan() && e.p_value.is_nan());
            // the pre-test still runs on the other pre-treatment cells
            assert_eq!(r.pretrend_test.as_ref().unwrap().df, 9);
            let shown = format!("{r}");
            assert!(
                shown.contains("          NA [      NA,       NA]"),
                "{shown}"
            );
            if bootstrap == 0 {
                let n = r.n_obs;
                let se = |c: &GroupTimeATT| c.std_error;
                Check::new(&case)
                    .vector(
                        "never_cell_se",
                        &r.group_time_atts.iter().map(se).collect::<Vec<_>>(),
                        Tol::se(n),
                    )
                    .vector(
                        "never_event_se",
                        &r.event_study
                            .iter()
                            .map(|e| e.std_error)
                            .collect::<Vec<_>>(),
                        Tol::se(n),
                    )
                    .finish();
            }
        }
    }

    #[test]
    fn test_staggered_did_rejects_bad_confidence_level() {
        let ds = create_staggered_dataset();
        for level in [1.0, 0.0, f64::NAN] {
            let cfg = StaggeredDidConfig {
                confidence_level: level,
                bootstrap: 0,
                min_obs_per_cell: 3,
                ..Default::default()
            };
            expect_invalid(
                run_staggered_did(&ds, "outcome", "first_treat", "year", "unit", None, cfg),
                "confidence_level",
            );
        }
    }
}
