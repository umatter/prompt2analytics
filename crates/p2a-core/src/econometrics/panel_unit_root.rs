//! Panel Unit Root Tests.
//!
//! Implements various panel unit root tests for testing stationarity
//! in panel data. Panel unit root tests exploit the cross-sectional
//! dimension to increase statistical power over univariate tests.
//!
//! # Available Tests
//!
//! - **Levin-Lin-Chu (LLC)**: Common unit root process across panels
//! - **Im-Pesaran-Shin (IPS)**: Heterogeneous unit root processes
//! - **Fisher-type (Maddala-Wu)**: Combines individual ADF test p-values
//! - **Hadri**: Tests null of stationarity
//!
//! # References
//!
//! - Levin, A., Lin, C.-F., & Chu, C.-S. J. (2002). "Unit Root Tests in Panel Data:
//!   Asymptotic and Finite-Sample Properties". *Journal of Econometrics*, 108(1), 1-24.
//!   https://doi.org/10.1016/S0304-4076(01)00098-7
//!
//! - Im, K. S., Pesaran, M. H., & Shin, Y. (2003). "Testing for Unit Roots in
//!   Heterogeneous Panels". *Journal of Econometrics*, 115(1), 53-74.
//!   https://doi.org/10.1016/S0304-4076(03)00109-2
//!
//! - Maddala, G. S., & Wu, S. (1999). "A Comparative Study of Unit Root Tests with
//!   Panel Data and a New Simple Test". *Oxford Bulletin of Economics and Statistics*,
//!   61(S1), 631-652.
//!
//! - Hadri, K. (2000). "Testing for Stationarity in Heterogeneous Panel Data".
//!   *Econometrics Journal*, 3(2), 148-161.
//!
//! - R packages: `plm` (Croissant & Millo), `tseries`
//!
//! # Data requirements
//!
//! - **Units** are read as a factor: numeric or text ids, in R `factor()` order (numbers
//!   numerically, text in C order), which is the order of `unit_statistics`.
//! - **Time** must hold whole numbers or dates; a fractional value is an error. Each unit's
//!   observations are taken in time order.
//! - A repeated (unit, time) row is an error.
//! - **Gaps** in a unit's time are joined: its observations are used in time order as if
//!   consecutive, as `plm::purtest` does; the result warns how many units have gaps.
//! - A Date time column counts days, so dates more than a day apart are reported as gaps;
//!   recode monthly, quarterly or yearly dates as period numbers first.
//! - Units with fewer than 3 observations are dropped, with a warning naming them.
//! - Rows with a missing value in the variable, the unit or the time are dropped and
//!   reported in `missing` and `warnings`; an infinite value is an error.
//!
//! # Lag order
//!
//! The ADF-type tests (LLC, IPS, Fisher) regress each unit's first difference on its
//! lagged level, the deterministic terms and `p` lagged differences.
//!
//! - **Default** (neither `lags` nor `lag_selection`): `floor(T_max^(1/3))` lags for every
//!   unit, with `T_max` the longest unit, capped by `max_lags`. The result labels it so.
//! - **Fixed** (`lags = Some(p)` or `lag_selection = Fixed(p)`): `p` lags for every unit.
//! - **Per unit** (`lag_selection = Aic | Sic | Hall`), as `plm::purtest(lags = "AIC" |
//!   "SIC" | "Hall", pmax, fixedT = TRUE)`: every candidate order `0..=pmax` is fitted on
//!   the common sample `t = pmax + 2..T` (`fixedT`); AIC and SIC pick the smallest
//!   `ln(rss/n) + c K / n` (`c = 2` or `ln n`; ties go to the larger order), and Hall's rule
//!   the largest order whose last lagged difference has `|t| >= 1.96` (0 if none). `pmax`
//!   defaults to `max_lags`, else [`DEFAULT_PMAX`] (plm's default).
//! - **`dfcor`** selects the residual variance of the ADF regression: `rss/(n - K)` (true,
//!   the default here and `urca::ur.df`'s convention) or `rss/n` (false, plm's default).
//!   Hall's rule and the per-unit t-statistics depend on it; AIC and SIC do not.
//!
//! A unit needs `T >= 3 + n_det + 2p` observations for `p` lags (`n_det` = 0, 1 or 2
//! deterministic terms), so that the ADF regression keeps a residual degree of freedom; a
//! shorter unit, or one whose regressors are collinear (an exactly linear series), is
//! skipped and named in the warnings. With no unit left the test is an error.
//!
//! The per-unit ADF regressions and the per-unit lag orders are validated against R
//! (`urca::ur.df`, `plm::purtest`); the LLC, IPS, Fisher and Hadri statistics are not.

use ndarray::{Array1, Array2, ArrayView1};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, MissingSummary, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::solve::least_squares;
use crate::options::Choices;
use crate::regression::ols::normal_equations_inverse;
use crate::resource;
use crate::traits::convergence::ResultNotes;
use crate::traits::estimator::{chi_squared_p_value, normal_cdf};

// ═══════════════════════════════════════════════════════════════════════════════
// Types and Configuration
// ═══════════════════════════════════════════════════════════════════════════════

/// Type of panel unit root test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PanelUnitRootTest {
    /// Levin-Lin-Chu test (common unit root)
    #[default]
    LLC,
    /// Im-Pesaran-Shin test (heterogeneous unit root)
    IPS,
    /// Fisher-type test combining individual ADF p-values (Maddala-Wu)
    Fisher,
    /// Hadri test (null: stationarity)
    Hadri,
}

crate::option_enum!(
    PanelUnitRootTest,
    crate::options::Choices {
        option: "test",
        values: &[
            (
                &["llc", "levinlin", "levin_lin_chu"],
                PanelUnitRootTest::LLC
            ),
            (&["ips", "im_pesaran_shin"], PanelUnitRootTest::IPS),
            (
                &["madwu", "maddala_wu", "fisher"],
                PanelUnitRootTest::Fisher
            ),
            (&["hadri"], PanelUnitRootTest::Hadri),
        ],
    },
    no_display
);

impl fmt::Display for PanelUnitRootTest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PanelUnitRootTest::LLC => write!(f, "Levin-Lin-Chu"),
            PanelUnitRootTest::IPS => write!(f, "Im-Pesaran-Shin"),
            PanelUnitRootTest::Fisher => write!(f, "Fisher (Maddala-Wu)"),
            PanelUnitRootTest::Hadri => write!(f, "Hadri"),
        }
    }
}

/// Model specification for panel unit root tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PanelModel {
    /// No deterministic components
    None,
    /// Individual intercepts only
    #[default]
    Intercept,
    /// Individual intercepts and trends
    Trend,
}

crate::option_enum!(
    PanelModel,
    crate::options::Choices {
        option: "model",
        values: &[
            (&["none", "nc"], PanelModel::None),
            (&["intercept", "drift", "c", "const"], PanelModel::Intercept),
            (&["trend", "ct"], PanelModel::Trend),
        ],
    },
    no_display
);

impl fmt::Display for PanelModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PanelModel::None => write!(f, "None"),
            PanelModel::Intercept => write!(f, "Individual Intercepts"),
            PanelModel::Trend => write!(f, "Individual Intercepts and Trends"),
        }
    }
}

/// plm::purtest's default maximum lag order for lag selection.
pub const DEFAULT_PMAX: usize = 10;

/// How the ADF-type tests choose each unit's lag order (module docs, "Lag order").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LagSelection {
    /// The same order for every unit.
    Fixed(usize),
    /// Akaike's criterion per unit (plm `lags = "AIC"`).
    Aic,
    /// Schwarz's criterion per unit (plm `lags = "SIC"`).
    Sic,
    /// Hall's general-to-specific t rule per unit (plm `lags = "Hall"`).
    Hall,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LagRule {
    Fixed,
    Aic,
    Sic,
    Hall,
}

static LAG_RULES: Choices<LagRule> = Choices {
    option: "lag_selection",
    values: &[
        (&["fixed"], LagRule::Fixed),
        (&["aic"], LagRule::Aic),
        (&["sic", "bic", "schwarz"], LagRule::Sic),
        (&["hall", "t_sig", "tsig"], LagRule::Hall),
    ],
};

impl LagSelection {
    /// `name`: fixed | aic | sic (bic, schwarz) | hall (t_sig, tsig); case and separator
    /// insensitive. `fixed` needs `lags`; the criteria refuse `lags` (their bound is
    /// `pmax`). An unknown name is `InvalidSpecification` listing the names.
    pub fn from_option(name: &str, lags: Option<usize>) -> EconResult<LagSelection> {
        let rule = LAG_RULES.parse(name)?;
        match (rule, lags) {
            (LagRule::Fixed, Some(p)) => Ok(LagSelection::Fixed(p)),
            (LagRule::Fixed, None) => Err(EconError::InvalidSpecification {
                message: "lag_selection = fixed needs lags (the lag order of every unit)"
                    .to_string(),
            }),
            (_, Some(_)) => Err(EconError::InvalidSpecification {
                message: format!(
                    "lags fixes the lag order and cannot be combined with lag_selection = {} \
                     (use pmax)",
                    LAG_RULES.canonical(rule)
                ),
            }),
            (LagRule::Aic, None) => Ok(LagSelection::Aic),
            (LagRule::Sic, None) => Ok(LagSelection::Sic),
            (LagRule::Hall, None) => Ok(LagSelection::Hall),
        }
    }

    /// "fixed | aic | sic | hall", for CLI help and tool descriptions.
    pub fn names_help() -> String {
        LAG_RULES.help()
    }

    /// The canonical option spelling of the rule ("aic").
    fn option_name(&self) -> &'static str {
        match self {
            LagSelection::Fixed(_) => "fixed",
            LagSelection::Aic => "aic",
            LagSelection::Sic => "sic",
            LagSelection::Hall => "hall",
        }
    }
}

impl fmt::Display for LagSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LagSelection::Fixed(p) => write!(f, "fixed ({p})"),
            LagSelection::Aic => write!(f, "AIC"),
            LagSelection::Sic => write!(f, "SIC"),
            LagSelection::Hall => write!(f, "Hall"),
        }
    }
}

fn default_true() -> bool {
    true
}

/// Configuration for panel unit root tests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelUnitRootConfig {
    /// Type of test to perform
    pub test_type: PanelUnitRootTest,
    /// Model specification
    pub model: PanelModel,
    /// The lag order of every unit's ADF-type regression. `None` with no
    /// `lag_selection`: `floor(T_max^(1/3))` for every unit (module docs, "Lag order").
    pub lags: Option<usize>,
    /// Cap of the default lag order; the default `pmax` of lag selection.
    pub max_lags: Option<usize>,
    /// Per-unit lag selection (AIC, SIC, Hall) or a fixed order; `None` = the default rule.
    #[serde(default)]
    pub lag_selection: Option<LagSelection>,
    /// Largest order lag selection considers (default `max_lags`, else
    /// [`DEFAULT_PMAX`]). Only with `lag_selection = Aic | Sic | Hall`.
    #[serde(default)]
    pub pmax: Option<usize>,
    /// ADF residual variance `rss/(n - K)` (true, `urca::ur.df`) or `rss/n` (false, plm's
    /// default). Default: true.
    #[serde(default = "default_true")]
    pub dfcor: bool,
}

impl Default for PanelUnitRootConfig {
    fn default() -> Self {
        Self {
            test_type: PanelUnitRootTest::LLC,
            model: PanelModel::Intercept,
            lags: None,
            max_lags: None,
            lag_selection: None,
            pmax: None,
            dfcor: true,
        }
    }
}

/// Result from a panel unit root test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelUnitRootResult {
    /// Test type used
    pub test_type: PanelUnitRootTest,
    /// Model specification
    pub model: PanelModel,
    /// Test statistic
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Null hypothesis description
    pub null_hypothesis: String,
    /// Alternative hypothesis description
    pub alternative_hypothesis: String,
    /// Number of panels (N)
    pub n_panels: usize,
    /// Average time periods (T)
    pub avg_time_periods: f64,
    /// Lags used: the largest unit lag order (the common order when it is fixed)
    pub lags_used: usize,
    /// Individual unit test statistics (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit_statistics: Option<Vec<f64>>,
    /// Individual unit p-values (for Fisher test)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit_p_values: Option<Vec<f64>>,
    /// Warnings: units with gaps in time, dropped short units, skipped units, dropped rows.
    #[serde(default)]
    pub warnings: Vec<String>,
    /// Rows dropped because of missing values.
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
    /// Ids of the units used, in the order of `unit_statistics`
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unit_ids: Vec<String>,
    /// Lag order of each unit used (empty for Hadri)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unit_lags: Vec<usize>,
    /// How the lag orders were chosen, as printed after "Lags:"
    #[serde(default)]
    pub lag_selection: String,
    /// ADF residual variance `rss/(n - K)` (true) or `rss/n` (false)
    #[serde(default = "default_true")]
    pub dfcor: bool,
}

impl ResultNotes for PanelUnitRootResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

impl fmt::Display for PanelUnitRootResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "═══════════════════════════════════════════════════════════════"
        )?;
        writeln!(f, "Panel Unit Root Test: {}", self.test_type)?;
        writeln!(
            f,
            "═══════════════════════════════════════════════════════════════"
        )?;
        writeln!(f)?;
        writeln!(f, "Model:      {}", self.model)?;
        writeln!(f, "N panels:   {}", self.n_panels)?;
        writeln!(f, "Avg T:      {:.1}", self.avg_time_periods)?;
        if self.lag_selection.is_empty() {
            writeln!(f, "Lags:       {}", self.lags_used)?;
        } else {
            writeln!(f, "Lags:       {}", self.lag_selection)?;
        }
        if let (Some(&lo), Some(&hi)) = (self.unit_lags.iter().min(), self.unit_lags.iter().max())
            && lo != hi
        {
            let mean = self.unit_lags.iter().sum::<usize>() as f64 / self.unit_lags.len() as f64;
            writeln!(
                f,
                "            per unit: min {lo}, max {hi}, mean {mean:.2}"
            )?;
        }
        if self.test_type != PanelUnitRootTest::Hadri {
            if self.dfcor {
                writeln!(f, "ADF residual variance: rss/(n - k) (dfcor = true)")?;
            } else {
                writeln!(
                    f,
                    "ADF residual variance: rss/n (dfcor = false, plm's default)"
                )?;
            }
        }
        if self.test_type == PanelUnitRootTest::Fisher {
            writeln!(
                f,
                "Unit p-values: MacKinnon (1994) response surface, asymptotic \
                 (plm::purtest p.approx = \"MacKinnon1994\")"
            )?;
        }
        writeln!(f)?;
        writeln!(f, "H0: {}", self.null_hypothesis)?;
        writeln!(f, "H1: {}", self.alternative_hypothesis)?;
        writeln!(f)?;
        writeln!(f, "Test statistic: {:.4}", self.statistic)?;
        writeln!(f, "P-value:        {:.4}", self.p_value)?;
        writeln!(f)?;

        let sig = if self.p_value < 0.01 {
            "*** (reject H0 at 1%)"
        } else if self.p_value < 0.05 {
            "** (reject H0 at 5%)"
        } else if self.p_value < 0.10 {
            "* (reject H0 at 10%)"
        } else {
            "(fail to reject H0)"
        };
        writeln!(f, "Significance: {}", sig)?;

        // The missing-value note is the last line, not a warning line.
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
                writeln!(f, "  - {w}")?;
            }
        }
        if let Some(note) = note {
            writeln!(f, "Missing values: {note}")?;
        }

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main Entry Point
// ═══════════════════════════════════════════════════════════════════════════════

/// Run a panel unit root test.
///
/// # Arguments
///
/// * `dataset` - Panel dataset
/// * `var_col` - Variable to test for unit root
/// * `unit_col` - Panel unit identifier column
/// * `time_col` - Time period column
/// * `config` - Test configuration (lag order: module docs, "Lag order")
///
/// # Returns
///
/// Test result with statistic and p-value.
///
/// # Example
///
/// ```ignore
/// use p2a_core::econometrics::{run_panel_unit_root, PanelUnitRootConfig, PanelUnitRootTest};
/// use p2a_core::econometrics::panel_unit_root::LagSelection;
///
/// let config = PanelUnitRootConfig {
///     test_type: PanelUnitRootTest::IPS,
///     lag_selection: Some(LagSelection::Aic),
///     pmax: Some(4),
///     ..Default::default()
/// };
/// let result = run_panel_unit_root(&dataset, "gdp", "country", "year", config)?;
/// ```
pub fn run_panel_unit_root(
    dataset: &Dataset,
    var_col: &str,
    unit_col: &str,
    time_col: &str,
    config: PanelUnitRootConfig,
) -> EconResult<PanelUnitRootResult> {
    check_lag_options(&config)?;
    let Panels {
        series: panels,
        mut warnings,
        missing,
    } = extract_panels(dataset, var_col, unit_col, time_col)?;

    let mut result = if config.test_type == PanelUnitRootTest::Hadri {
        run_hadri_test(&panels, config.model)?
    } else {
        let (plan, label) = lag_plan(&config, &panels);
        let mut r = match config.test_type {
            PanelUnitRootTest::LLC => run_llc_test(&panels, config.model, &plan, config.dfcor),
            PanelUnitRootTest::IPS => run_ips_test(&panels, config.model, &plan, config.dfcor),
            _ => run_fisher_test(&panels, config.model, &plan, config.dfcor),
        }?;
        r.lag_selection = label;
        r
    };
    result.dfcor = config.dfcor;
    warnings.append(&mut result.warnings);
    if let Some(note) = missing.note() {
        warnings.push(note);
    }
    result.warnings = warnings;
    result.missing = missing;
    Ok(result)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Lag order
// ═══════════════════════════════════════════════════════════════════════════════

fn invalid(message: String) -> EconError {
    EconError::InvalidSpecification { message }
}

/// The option combinations that contradict each other (module docs, "Lag order").
fn check_lag_options(config: &PanelUnitRootConfig) -> EconResult<()> {
    if config.test_type == PanelUnitRootTest::Hadri {
        if config.lag_selection.is_some() || config.pmax.is_some() {
            return Err(invalid(
                "the Hadri test uses no lagged differences; lag_selection and pmax do not apply"
                    .to_string(),
            ));
        }
        return Ok(());
    }
    match (config.lag_selection, config.lags) {
        (Some(LagSelection::Fixed(p)), Some(q)) if p != q => {
            return Err(invalid(format!(
                "lags = {q} conflicts with lag_selection = fixed ({p})"
            )));
        }
        (Some(LagSelection::Fixed(_)), _) | (None, _) => {
            if config.pmax.is_some() {
                return Err(invalid(
                    "pmax applies only to lag_selection = aic, sic or hall".to_string(),
                ));
            }
        }
        (Some(rule), Some(_)) => {
            return Err(invalid(format!(
                "lags fixes the lag order and cannot be combined with lag_selection = {} \
                 (use pmax)",
                rule.option_name()
            )));
        }
        (Some(_), None) => {
            if config.pmax == Some(0) {
                return Err(invalid(
                    "pmax must be at least 1 for lag selection (got 0)".to_string(),
                ));
            }
            if let (Some(a), Some(b)) = (config.pmax, config.max_lags)
                && a != b
            {
                return Err(invalid(format!(
                    "pmax and max_lags give different maximum lags ({a}, {b})"
                )));
            }
        }
    }
    Ok(())
}

/// How each unit's lag order is found.
enum LagPlan {
    Fixed(usize),
    Select { rule: LagSelection, pmax: usize },
}

/// The plan and its label (the text after "Lags:"); the options are checked.
fn lag_plan(config: &PanelUnitRootConfig, panels: &[(String, Vec<f64>)]) -> (LagPlan, String) {
    match (config.lag_selection, config.lags) {
        (Some(LagSelection::Fixed(p)), _) | (None, Some(p)) => {
            (LagPlan::Fixed(p), format!("{p} (fixed by the user)"))
        }
        (None, None) => {
            let max_t = panels.iter().map(|(_, s)| s.len()).max().unwrap_or(0);
            let default = (max_t as f64).cbrt().floor() as usize;
            let cap = config.max_lags.unwrap_or(default).max(1);
            let p = default.min(cap);
            let label = if cap < default {
                format!("{p} (fixed: min(floor(T_max^(1/3)), max_lags))")
            } else {
                format!("{p} (fixed: floor(T_max^(1/3)))")
            };
            (LagPlan::Fixed(p), label)
        }
        (Some(rule), _) => {
            let pmax = config.pmax.or(config.max_lags).unwrap_or(DEFAULT_PMAX);
            (
                LagPlan::Select { rule, pmax },
                format!("{rule} per unit, pmax = {pmax}"),
            )
        }
    }
}

/// A unit's lag order under `plan`.
fn unit_lag(
    series: &[f64],
    plan: &LagPlan,
    model: PanelModel,
    dfcor: bool,
) -> Result<usize, UnitError> {
    match *plan {
        LagPlan::Fixed(p) => Ok(p),
        LagPlan::Select { rule, pmax } => select_lag(series, rule, pmax, model, dfcor),
    }
}

/// The number of deterministic terms of `model`.
fn n_det(model: PanelModel) -> usize {
    match model {
        PanelModel::None => 0,
        PanelModel::Intercept => 1,
        PanelModel::Trend => 2,
    }
}

fn det_text(model: PanelModel) -> &'static str {
    match model {
        PanelModel::None => "no deterministic terms",
        PanelModel::Intercept => "an intercept",
        PanelModel::Trend => "an intercept and a trend",
    }
}

/// `3 + n_det + 2 * lags`, the observations a unit needs for `lags` lags (saturating).
fn adf_required(model: PanelModel, lags: usize) -> usize {
    lags.saturating_mul(2).saturating_add(3 + n_det(model))
}

/// Why one unit's regression failed: a reason to skip the unit (too short, rank
/// deficient, an exact fit) or an error that stops the test (cancellation, memory).
enum UnitError {
    Skip { reason: String, error: EconError },
    Fatal(EconError),
}

impl From<EconError> for UnitError {
    fn from(e: EconError) -> Self {
        UnitError::Fatal(e)
    }
}

/// A unit has `T >= 3 + n_det + 2 * lags` observations, so that its ADF regression on
/// rows `lags + 1..T` keeps a residual degree of freedom.
fn adf_length_check(t: usize, model: PanelModel, lags: usize) -> Result<(), UnitError> {
    let required = adf_required(model, lags);
    if t >= required {
        return Ok(());
    }
    let det = det_text(model);
    Err(UnitError::Skip {
        reason: format!("{t} observations; {lags} lag(s) with {det} need at least {required}"),
        error: EconError::InsufficientData {
            required,
            provided: t,
            context: format!(
                "ADF regression with {lags} lag(s) and {det} needs at least {required} \
                 observations"
            ),
        },
    })
}

fn rank_deficient(what: String, rank: usize, cols: usize) -> UnitError {
    UnitError::Skip {
        reason: format!("rank deficient: rank {rank} of {cols}"),
        error: EconError::SingularMatrix {
            context: format!("{what} is rank deficient (rank {rank} of {cols})"),
            suggestion: "the series is exactly linear or constant over the sample".to_string(),
        },
    }
}

/// The ADF design of order `lags` on rows `first..T` (0-based `i`, `first > lags`):
/// response `s[i] - s[i-1]`; columns `s[i-1]`, 1 (n_det >= 1), `i + 1` (n_det = 2), then
/// `s[i-j] - s[i-j-1]` for j = 1..=lags.
fn adf_design(
    s: &[f64],
    model: PanelModel,
    lags: usize,
    first: usize,
) -> EconResult<(Array2<f64>, Array1<f64>)> {
    let what = "ADF regression design";
    let nd = n_det(model);
    let n = s.len() - first;
    let k = 1 + nd + lags;
    let mut x = resource::try_zeros2(n, k, what)?;
    let mut y = resource::try_zeros1(n, what)?;
    for (r, i) in (first..s.len()).enumerate() {
        y[r] = s[i] - s[i - 1];
        x[[r, 0]] = s[i - 1];
        if nd >= 1 {
            x[[r, 1]] = 1.0;
        }
        if nd == 2 {
            x[[r, 2]] = (i + 1) as f64;
        }
        for j in 1..=lags {
            x[[r, nd + j]] = s[i - j] - s[i - j - 1];
        }
    }
    Ok((x, y))
}

fn rss(residuals: &Array1<f64>) -> f64 {
    residuals.iter().map(|e| e * e).sum()
}

/// One unit's lag order by AIC, SIC or Hall's rule on the common sample `t = pmax + 2..T`
/// (plm's `lagsel`, `fixedT = TRUE`; module docs, "Lag order").
fn select_lag(
    s: &[f64],
    rule: LagSelection,
    pmax: usize,
    model: PanelModel,
    dfcor: bool,
) -> Result<usize, UnitError> {
    let t = s.len();
    let required = adf_required(model, pmax);
    let det = det_text(model);
    if t < required {
        return Err(UnitError::Skip {
            reason: format!(
                "{t} observations; lag selection with pmax = {pmax} and {det} needs at least \
                 {required}"
            ),
            error: EconError::InsufficientData {
                required,
                provided: t,
                context: format!(
                    "lag selection with pmax = {pmax} and {det} needs at least {required} \
                     observations per unit"
                ),
            },
        });
    }
    let nd = n_det(model);
    let (x, y) = adf_design(s, model, pmax, pmax + 1)?;
    let n = y.len();
    let nf = n as f64;
    let fit = |p: usize| -> Result<crate::linalg::solve::LeastSquares, UnitError> {
        resource::checkpoint()?;
        let k = 1 + nd + p;
        let ls = least_squares(x.slice(ndarray::s![.., ..k]), y.view())?;
        if ls.rank < k {
            return Err(rank_deficient(
                format!("the lag-selection regression with {p} lag(s)"),
                ls.rank,
                k,
            ));
        }
        Ok(ls)
    };
    match rule {
        LagSelection::Aic | LagSelection::Sic => {
            let penalty = if rule == LagSelection::Aic {
                2.0
            } else {
                nf.ln()
            };
            let mut best: Option<(usize, f64)> = None;
            // Descending, strictly smaller wins: ties go to the larger order (which.min).
            for p in (0..=pmax).rev() {
                let ls = fit(p)?;
                let k = (1 + nd + p) as f64;
                let crit = (rss(&ls.residuals) / nf).ln() + penalty * k / nf;
                if best.is_none_or(|(_, c)| crit < c) {
                    best = Some((p, crit));
                }
            }
            Ok(best.map_or(0, |(p, _)| p))
        }
        LagSelection::Hall => {
            for p in (1..=pmax).rev() {
                let ls = fit(p)?;
                let k = 1 + nd + p;
                let s2 = rss(&ls.residuals) / if dfcor { (n - k) as f64 } else { nf };
                let j = nd + p;
                let t_last = ls.coefficients[j] / (s2 * ls.unscaled_cov[[j, j]]).sqrt();
                // numeric-ok: plm's lagsel (Hall's rule) hard-codes 1.96
                if t_last.abs() >= 1.96 {
                    return Ok(p);
                }
            }
            Ok(0)
        }
        LagSelection::Fixed(p) => Ok(p),
    }
}

/// One unit's ADF regression.
struct AdfFit {
    /// t-statistic of the lagged level.
    t_stat: f64,
}

/// The ADF regression of one unit with `lags` lagged differences on rows `lags + 1..T`;
/// residual variance `rss/(n - K)` (`dfcor`) or `rss/n`.
fn adf_regression(
    s: &[f64],
    model: PanelModel,
    lags: usize,
    dfcor: bool,
) -> Result<AdfFit, UnitError> {
    adf_length_check(s.len(), model, lags)?;
    let (x, y) = adf_design(s, model, lags, lags + 1)?;
    let k = x.ncols();
    let ls = least_squares(x.view(), y.view())?;
    if ls.rank < k {
        return Err(rank_deficient(
            format!("the ADF regression with {lags} lag(s)"),
            ls.rank,
            k,
        ));
    }
    let n = y.len();
    let df_resid = n.checked_sub(k).filter(|&d| d > 0).ok_or_else(|| {
        EconError::Computation(format!("ADF regression: {n} rows for {k} columns"))
    })?;
    let s2 = rss(&ls.residuals) / if dfcor { df_resid as f64 } else { n as f64 };
    let t_stat = ls.coefficients[0] / (s2 * ls.unscaled_cov[[0, 0]]).sqrt();
    if !t_stat.is_finite() {
        return Err(UnitError::Skip {
            reason: "exact fit: no residual variation".to_string(),
            error: EconError::SingularMatrix {
                context: "ADF regression fits exactly (no residual variation)".to_string(),
                suggestion: "the series is exactly linear or constant over the sample".to_string(),
            },
        });
    }
    Ok(AdfFit { t_stat })
}

/// Runs `f` on every unit; a unit `f` cannot estimate is skipped and named in one warning,
/// any other error stops the test. With no unit left, the first unit's error, its context
/// prefixed with `test`.
fn skip_or_propagate<T>(
    test: PanelUnitRootTest,
    panels: &[(String, Vec<f64>)],
    mut f: impl FnMut(&[f64]) -> Result<T, UnitError>,
) -> EconResult<(Vec<(usize, T)>, Vec<String>)> {
    let mut kept = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    let mut first_error: Option<EconError> = None;
    for (i, (label, series)) in panels.iter().enumerate() {
        resource::checkpoint()?;
        match f(series) {
            Ok(v) => kept.push((i, v)),
            Err(UnitError::Fatal(e)) => return Err(e),
            Err(UnitError::Skip { reason, error }) => {
                skipped.push(format!("'{label}' ({reason})"));
                first_error.get_or_insert(error);
            }
        }
    }
    if kept.is_empty() {
        let prefix = format!("{test}: no unit can be used: ");
        return Err(match first_error {
            Some(EconError::InsufficientData {
                required,
                provided,
                context,
            }) => EconError::InsufficientData {
                required,
                provided,
                context: format!("{prefix}{context}"),
            },
            Some(EconError::SingularMatrix {
                context,
                suggestion,
            }) => EconError::SingularMatrix {
                context: format!("{prefix}{context}"),
                suggestion,
            },
            Some(e) => e,
            None => EconError::InsufficientData {
                required: 1,
                provided: 0,
                context: format!("{test}: no unit"),
            },
        });
    }
    let mut warnings = Vec::new();
    if !skipped.is_empty() {
        let mut list = skipped
            .iter()
            .take(5)
            .cloned()
            .collect::<Vec<_>>()
            .join("; ");
        if skipped.len() > 5 {
            list.push_str(&format!(" (and {} more)", skipped.len() - 5));
        }
        warnings.push(format!(
            "Skipped {} unit(s) whose ADF regression cannot be estimated: {list}",
            skipped.len()
        ));
    }
    Ok((kept, warnings))
}

/// A result with the fields every test fills; the caller sets the rest.
#[allow(clippy::too_many_arguments)]
fn unit_root_result(
    test_type: PanelUnitRootTest,
    model: PanelModel,
    statistic: f64,
    p_value: f64,
    hypotheses: (&str, &str),
    panels: &[(String, Vec<f64>)],
    used: &[usize],
    unit_lags: Vec<usize>,
    warnings: Vec<String>,
) -> PanelUnitRootResult {
    let n = used.len();
    let avg_t = used.iter().map(|&i| panels[i].1.len() as f64).sum::<f64>() / n as f64;
    PanelUnitRootResult {
        test_type,
        model,
        statistic,
        p_value,
        null_hypothesis: hypotheses.0.to_string(),
        alternative_hypothesis: hypotheses.1.to_string(),
        n_panels: n,
        avg_time_periods: avg_t,
        lags_used: unit_lags.iter().copied().max().unwrap_or(0),
        unit_statistics: None,
        unit_p_values: None,
        warnings,
        missing: MissingSummary::default(),
        unit_ids: used.iter().map(|&i| panels[i].0.clone()).collect(),
        unit_lags,
        lag_selection: String::new(),
        dfcor: true,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Panel Data Extraction
// ═══════════════════════════════════════════════════════════════════════════════

/// The panels of a unit-root test: each unit's label and observations in time order, in
/// level order, with what the extraction dropped or joined.
struct Panels {
    series: Vec<(String, Vec<f64>)>,
    warnings: Vec<String>,
    missing: MissingSummary,
}

/// "'a', 'b', 'c' (and 4 more)": at most five labels.
fn label_list(labels: &[&str]) -> String {
    let shown: Vec<String> = labels.iter().take(5).map(|l| format!("'{l}'")).collect();
    let mut text = shown.join(", ");
    if labels.len() > 5 {
        text.push_str(&format!(" (and {} more)", labels.len() - 5));
    }
    text
}

/// Extract the panels: units as a factor (text or numeric ids), times as a whole-number
/// index, rows with a missing value dropped. A repeated (unit, time) is an error; gaps
/// are joined in time order, as `plm::purtest` does, with a warning; units with fewer
/// than 3 observations are dropped with a warning naming them.
fn extract_panels(
    dataset: &Dataset,
    var_col: &str,
    unit_col: &str,
    time_col: &str,
) -> EconResult<Panels> {
    let cols = ColumnRequest::new(dataset.df())
        .factor(unit_col)
        .time_index(time_col)
        .numeric(var_col)
        .na(NaPolicy::Omit)
        .extract()?;
    let factor = cols.factor(unit_col);
    let times = cols.time_index(time_col);
    let values = cols.numeric(var_col);
    let n_units = factor.n_levels();

    // Rows grouped by unit in level order (a stable counting sort), then by time.
    let what = "panel unit-root rows";
    let mut start = resource::try_vec(n_units.saturating_add(1), 0usize, what)?;
    for &code in &factor.codes {
        start[code as usize + 1] += 1;
    }
    for u in 0..n_units {
        start[u + 1] += start[u];
    }
    let mut next = resource::try_vec(n_units, 0usize, what)?;
    next.copy_from_slice(&start[..n_units]);
    let mut order = resource::try_vec(factor.codes.len(), 0usize, what)?;
    for (row, &code) in factor.codes.iter().enumerate() {
        let slot = &mut next[code as usize];
        order[*slot] = row;
        *slot += 1;
    }

    let mut series = Vec::new();
    let mut gapped = 0usize;
    let mut first_gap: Option<(usize, i64, i64)> = None;
    let mut short: Vec<&str> = Vec::new();
    for u in 0..n_units {
        resource::checkpoint()?;
        let rows = &mut order[start[u]..start[u + 1]];
        rows.sort_by_key(|&r| times[r]);
        let mut gap_here = false;
        for (p, pair) in rows.windows(2).enumerate() {
            let (t1, t2) = (times[pair[0]], times[pair[1]]);
            if t1 == t2 {
                let k = rows[p..].partition_point(|&r| times[r] == t1);
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "Unit '{}' has {k} rows for time {t1}; panel unit-root tests need \
                         one row per (unit, time).",
                        factor.levels[u]
                    ),
                });
            }
            if !gap_here && t2.saturating_sub(t1) > 1 {
                gap_here = true;
                first_gap.get_or_insert((u, t1, t2));
            }
        }
        gapped += usize::from(gap_here);
        if rows.len() < 3 {
            short.push(&factor.levels[u]);
            continue;
        }
        let mut unit_values = resource::try_with_capacity(rows.len(), what)?;
        unit_values.extend(rows.iter().map(|&r| values[r]));
        series.push((factor.levels[u].clone(), unit_values));
    }

    let mut warnings = Vec::new();
    if let Some((u, t1, t2)) = first_gap {
        warnings.push(format!(
            "{gapped} of {n_units} unit(s) have gaps in time (first: unit '{}' between {t1} \
             and {t2}); their observations are joined in time order as if consecutive, as \
             plm::purtest does.",
            factor.levels[u]
        ));
    }
    if !short.is_empty() {
        warnings.push(format!(
            "Dropped {} unit(s) with fewer than 3 observations: {}",
            short.len(),
            label_list(&short)
        ));
    }
    if series.is_empty() {
        return Err(EconError::InsufficientData {
            required: 3,
            provided: 0,
            context: format!(
                "panel unit-root test: no unit of '{unit_col}' has 3 or more observations \
                 of '{var_col}'"
            ),
        });
    }
    Ok(Panels {
        series,
        warnings,
        missing: cols.missing().clone(),
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Levin-Lin-Chu Test
// ═══════════════════════════════════════════════════════════════════════════════

/// Levin-Lin-Chu (2002) panel unit root test.
///
/// Tests H0: All panels have a unit root (common ρ = 0)
/// vs H1: All panels are stationary (common ρ < 0)
///
/// The LLC test assumes a common autoregressive parameter across panels
/// and is based on a pooled ADF regression with panel-specific intercepts/trends.
fn run_llc_test(
    panels: &[(String, Vec<f64>)],
    model: PanelModel,
    plan: &LagPlan,
    dfcor: bool,
) -> EconResult<PanelUnitRootResult> {
    // Step 1: each unit's auxiliary regressions; units that cannot be estimated are skipped.
    // dfcor enters only Hall's lag choice (plm's lagsel), not the auxiliary residuals.
    let (units, warnings) = skip_or_propagate(PanelUnitRootTest::LLC, panels, |series| {
        let p = unit_lag(series, plan, model, dfcor)?;
        let (e, v) = compute_llc_residuals(series, model, p)?;
        Ok((p, e, v))
    })?;

    let mut delta_y_tilde_all: Vec<f64> = Vec::new();
    let mut y_lag_tilde_all: Vec<f64> = Vec::new();
    let mut s2_all: Vec<f64> = Vec::with_capacity(units.len());
    for (_, (_, e, v)) in &units {
        // Panel-specific variance
        let s2: f64 = e.iter().map(|ei| ei * ei).sum::<f64>() / e.len() as f64;
        s2_all.push(s2.max(1e-10));

        // Standardize residuals by sqrt(s2)
        let s = s2.sqrt().max(1e-10);
        delta_y_tilde_all.extend(e.iter().map(|ei| ei / s));
        y_lag_tilde_all.extend(v.iter().map(|vi| vi / s));
    }
    let used: Vec<usize> = units.iter().map(|(i, _)| *i).collect();
    let unit_lags: Vec<usize> = units.iter().map(|(_, (p, _, _))| *p).collect();
    let n = used.len();
    let avg_t: f64 = used.iter().map(|&i| panels[i].1.len() as f64).sum::<f64>() / n as f64;

    // Step 2: Pooled regression of standardized residuals
    // Δ̃y = ρ * ṽ_{t-1} + error
    let n_obs = delta_y_tilde_all.len();
    let delta_y_tilde = Array1::from(delta_y_tilde_all);
    let y_lag_tilde = Array1::from(y_lag_tilde_all);

    // Simple regression: ρ̂ = Σ(ṽ·Δ̃y) / Σ(ṽ²)
    let sum_vy: f64 = y_lag_tilde
        .iter()
        .zip(delta_y_tilde.iter())
        .map(|(v, d)| v * d)
        .sum();
    let sum_vv: f64 = y_lag_tilde.iter().map(|v| v * v).sum();

    if sum_vv.abs() < 1e-10 {
        return Err(EconError::SingularMatrix {
            context: "LLC: ṽ'ṽ near zero".to_string(),
            suggestion: "Check for constant or near-constant series".to_string(),
        });
    }

    let rho_hat = sum_vy / sum_vv;

    // Compute standard error
    let resid: Array1<f64> = &delta_y_tilde - &y_lag_tilde.mapv(|v| v * rho_hat);
    let s2_resid: f64 = resid.iter().map(|r| r * r).sum::<f64>() / n_obs.saturating_sub(1) as f64;
    let se_rho = (s2_resid / sum_vv).sqrt();

    // Compute t-statistic
    let t_rho = rho_hat / se_rho.max(1e-10);

    // Step 3: Bias adjustment (simplified)
    // Use approximate adjustment factors from LLC (2002) tables
    let (mu_star, sigma_star) = get_llc_adjustment_factors(avg_t, model);

    // Adjusted t-statistic
    let n_bar_t = n as f64 * avg_t;
    let s_n_t = (s2_all.iter().sum::<f64>() / n as f64).sqrt();

    // t*_ρ = (t_ρ - N̄T * S_N,T * μ*_m,T / σ̂²) / σ*_m,T
    let t_star = (t_rho * se_rho - n_bar_t.sqrt() * s_n_t * mu_star) / sigma_star;

    // P-value from standard normal
    let p_value = normal_cdf(t_star);

    Ok(unit_root_result(
        PanelUnitRootTest::LLC,
        model,
        t_star,
        p_value,
        (
            "Panels contain unit roots (common ρ = 0)",
            "Panels are stationary (common ρ < 0)",
        ),
        panels,
        &used,
        unit_lags,
        warnings,
    ))
}

/// LLC auxiliary regression residuals of one unit: Δy and y_{t-1} on the deterministic
/// terms and `lags` lagged differences, on rows `lags + 1..T`.
fn compute_llc_residuals(
    series: &[f64],
    model: PanelModel,
    lags: usize,
) -> Result<(Vec<f64>, Vec<f64>), UnitError> {
    adf_length_check(series.len(), model, lags)?;
    let delta_y: Vec<f64> = series.windows(2).map(|w| w[1] - w[0]).collect();
    let y_lag = &series[..series.len() - 1];
    let t = delta_y.len();
    let t_eff = t - lags;

    // Design matrix: [intercept, trend?, lagged Δy_1, ..., lagged Δy_p]
    let n_cols = n_det(model) + lags;
    let mut x = resource::try_zeros2(t_eff, n_cols, "LLC auxiliary regression")?;
    let mut dy_vec = Vec::with_capacity(t_eff);
    let mut ylag_vec = Vec::with_capacity(t_eff);

    for i in lags..t {
        let row = i - lags;
        dy_vec.push(delta_y[i]);
        ylag_vec.push(y_lag[i]);

        let mut col = 0;
        if model == PanelModel::Intercept || model == PanelModel::Trend {
            x[[row, col]] = 1.0;
            col += 1;
        }
        if model == PanelModel::Trend {
            x[[row, col]] = (i + 1) as f64;
            col += 1;
        }
        for j in 0..lags {
            x[[row, col + j]] = delta_y[i - j - 1];
        }
    }

    if n_cols == 0 {
        // No regressors (no deterministic terms, no lags): the data are the residuals.
        return Ok((dy_vec, ylag_vec));
    }

    // OLS's solve decision: on a full-rank, well-conditioned design the normal equations
    // give both residual vectors from one X'X; otherwise lm.fit's pivoted QR decides rank.
    if let Some(normal) = normal_equations_inverse(x.view())? {
        let residuals = |v: Vec<f64>| -> Vec<f64> {
            let v = Array1::from(v);
            let b = normal.coefficients(x.view(), &v);
            (&v - &x.dot(&b)).to_vec()
        };
        return Ok((residuals(dy_vec), residuals(ylag_vec)));
    }
    let dy = least_squares(x.view(), ArrayView1::from(&dy_vec))?;
    if dy.rank < n_cols {
        return Err(rank_deficient(
            format!("the LLC auxiliary regression with {lags} lag(s)"),
            dy.rank,
            n_cols,
        ));
    }
    let v = least_squares(x.view(), ArrayView1::from(&ylag_vec))?;
    Ok((dy.residuals.to_vec(), v.residuals.to_vec()))
}

/// Get LLC adjustment factors (μ*, σ*) based on T and model.
/// These are approximations from LLC (2002) simulation tables.
fn get_llc_adjustment_factors(avg_t: f64, model: PanelModel) -> (f64, f64) {
    // Simplified approximations based on LLC (2002) Table 2
    // For more accuracy, use interpolation tables

    match model {
        PanelModel::None => {
            // No deterministics
            let mu = -0.5 / avg_t.sqrt();
            let sigma = 1.0 + 0.5 / avg_t;
            (mu, sigma)
        }
        PanelModel::Intercept => {
            // Individual means
            // μ* ≈ -0.71 - 1.61/T - 2.39/T²
            let mu = -0.71 - 1.61 / avg_t - 2.39 / (avg_t * avg_t);
            // σ* ≈ 1.0
            let sigma = 1.0;
            (mu, sigma)
        }
        PanelModel::Trend => {
            // Individual means and trends
            // μ* ≈ -1.04 - 4.69/T - 10.6/T²
            let mu = -1.04 - 4.69 / avg_t - 10.6 / (avg_t * avg_t);
            let sigma = 1.0;
            (mu, sigma)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Im-Pesaran-Shin Test
// ═══════════════════════════════════════════════════════════════════════════════

/// Every unit's lag order and ADF regression; units that cannot be estimated are skipped.
fn unit_adf_fits(
    test: PanelUnitRootTest,
    panels: &[(String, Vec<f64>)],
    model: PanelModel,
    plan: &LagPlan,
    dfcor: bool,
) -> EconResult<(Vec<usize>, Vec<usize>, Vec<AdfFit>, Vec<String>)> {
    let (units, warnings) = skip_or_propagate(test, panels, |series| {
        let p = unit_lag(series, plan, model, dfcor)?;
        Ok((p, adf_regression(series, model, p, dfcor)?))
    })?;
    let mut used = Vec::with_capacity(units.len());
    let mut lags = Vec::with_capacity(units.len());
    let mut fits = Vec::with_capacity(units.len());
    for (i, (p, fit)) in units {
        used.push(i);
        lags.push(p);
        fits.push(fit);
    }
    Ok((used, lags, fits, warnings))
}

/// Im-Pesaran-Shin (2003) panel unit root test.
///
/// Tests H0: All panels have a unit root (heterogeneous ρ_i = 0)
/// vs H1: Some panels are stationary (some ρ_i < 0)
///
/// Based on averaging individual ADF t-statistics and using
/// tabulated means and variances for standardization.
fn run_ips_test(
    panels: &[(String, Vec<f64>)],
    model: PanelModel,
    plan: &LagPlan,
    dfcor: bool,
) -> EconResult<PanelUnitRootResult> {
    let (used, unit_lags, fits, warnings) =
        unit_adf_fits(PanelUnitRootTest::IPS, panels, model, plan, dfcor)?;
    let t_stats: Vec<f64> = fits.iter().map(|f| f.t_stat).collect();
    let n_valid = t_stats.len();
    let avg_t: f64 = used.iter().map(|&i| panels[i].1.len() as f64).sum::<f64>() / n_valid as f64;

    // Compute average t-statistic
    let t_bar: f64 = t_stats.iter().sum::<f64>() / n_valid as f64;

    // Get moments E[t] and Var[t] from IPS (2003) tables
    let (e_t, var_t) = get_ips_moments(avg_t, model);

    // W-statistic: standardized t-bar
    let w_stat = (n_valid as f64).sqrt() * (t_bar - e_t) / var_t.sqrt();

    // P-value from standard normal
    let p_value = normal_cdf(w_stat);

    let mut r = unit_root_result(
        PanelUnitRootTest::IPS,
        model,
        w_stat,
        p_value,
        (
            "All panels have unit roots (ρ_i = 0 for all i)",
            "Some panels are stationary (ρ_i < 0 for some i)",
        ),
        panels,
        &used,
        unit_lags,
        warnings,
    );
    r.unit_statistics = Some(t_stats);
    Ok(r)
}

/// Get IPS (2003) expected mean and variance of t-statistic.
/// These are approximations based on Table 2 of IPS (2003).
fn get_ips_moments(avg_t: f64, model: PanelModel) -> (f64, f64) {
    match model {
        PanelModel::None => {
            // No constant: E[t] ≈ 0, Var[t] ≈ 1
            (0.0, 1.0)
        }
        PanelModel::Intercept => {
            // Individual intercept only
            // Approximation: E[t] ≈ -1.52 - 0.5/T, Var[t] ≈ 0.96 + 0.56/T
            let e_t = -1.52 - 0.5 / avg_t;
            let var_t = 0.96 + 0.56 / avg_t;
            (e_t, var_t)
        }
        PanelModel::Trend => {
            // Individual intercept and trend
            // E[t] ≈ -2.15 - 2.0/T, Var[t] ≈ 0.89 + 0.91/T
            let e_t = -2.15 - 2.0 / avg_t;
            let var_t = 0.89 + 0.91 / avg_t;
            (e_t, var_t)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Fisher Test
// ═══════════════════════════════════════════════════════════════════════════════

/// Fisher-type panel unit root test (Maddala-Wu, 1999).
///
/// Combines p-values from individual ADF tests using Fisher's method:
/// P = -2 Σ ln(p_i) ~ χ²(2N) under H0
///
/// This test allows for unbalanced panels and heterogeneous lag structures.
fn run_fisher_test(
    panels: &[(String, Vec<f64>)],
    model: PanelModel,
    plan: &LagPlan,
    dfcor: bool,
) -> EconResult<PanelUnitRootResult> {
    let (used, unit_lags, fits, warnings) =
        unit_adf_fits(PanelUnitRootTest::Fisher, panels, model, plan, dfcor)?;
    let t_stats: Vec<f64> = fits.iter().map(|f| f.t_stat).collect();
    // MacKinnon's p of a t far in the lower tail stays positive; the floor keeps ln p finite.
    let p_values: Vec<f64> = t_stats
        .iter()
        .map(|&t| mackinnon_p(t, model).max(f64::MIN_POSITIVE))
        .collect();
    let n_valid = p_values.len();

    // Fisher statistic: P = -2 Σ ln(p_i)
    let fisher_stat: f64 = -2.0 * p_values.iter().map(|p| p.ln()).sum::<f64>();

    // P-value from χ²(2N)
    let p_value = chi_squared_p_value(fisher_stat, 2.0 * n_valid as f64);

    let mut r = unit_root_result(
        PanelUnitRootTest::Fisher,
        model,
        fisher_stat,
        p_value,
        (
            "All panels have unit roots",
            "At least one panel is stationary",
        ),
        panels,
        &used,
        unit_lags,
        warnings,
    );
    r.unit_statistics = Some(t_stats);
    r.unit_p_values = Some(p_values);
    Ok(r)
}

/// MacKinnon's (1994) response surface for the asymptotic distribution of the Dickey-Fuller
/// t of one series, as `plm:::padf(p.approx = "MacKinnon1994")` stores it:
/// `p = Phi(small(t))` up to `limit`, `Phi(large(t))` above, with polynomial coefficients in
/// increasing powers.
struct MacKinnon94 {
    small: [f64; 3],
    large: [f64; 4],
    limit: f64,
}

impl MacKinnon94 {
    /// MacKinnon (1994), Table 3 (N = 1), rescaled as plm does (`small[, 3] / 100`,
    /// `large[, 2:3] / 10`, `large[, 4] / 100`).
    fn of(model: PanelModel) -> MacKinnon94 {
        match model {
            PanelModel::None => MacKinnon94 {
                small: [0.6344, 1.2378, 3.2496 / 100.0],
                large: [0.4797, 9.3557 / 10.0, -0.6999 / 10.0, 3.3066 / 100.0],
                limit: -1.04,
            },
            PanelModel::Intercept => MacKinnon94 {
                small: [2.1659, 1.4412, 3.8269 / 100.0],
                large: [1.7339, 9.3202 / 10.0, -1.2745 / 10.0, -1.0368 / 100.0],
                limit: -1.61,
            },
            PanelModel::Trend => MacKinnon94 {
                small: [3.2512, 1.6047, 4.9588 / 100.0],
                large: [2.5261, 6.1654 / 10.0, -3.7956 / 10.0, -6.0285 / 100.0],
                limit: -2.89,
            },
        }
    }

    /// The range over which the surface rises: from the vertex of the quadratic to the local
    /// maximum of the cubic (none without deterministic terms). These are MacKinnon's tau_min
    /// and tau_max; outside them the polynomials turn back.
    fn range(&self) -> (f64, f64) {
        let [_, b1, b2] = self.small;
        let lo = -b1 / (2.0 * b2);
        let [_, c1, c2, c3] = self.large;
        // c1 + 2 c2 t + 3 c3 t^2 = 0; with c3 < 0 the larger root is the maximum.
        let disc = c2 * c2 - 3.0 * c3 * c1;
        let hi = if c3 < 0.0 && disc >= 0.0 {
            (-c2 - disc.sqrt()) / (3.0 * c3)
        } else {
            f64::INFINITY
        };
        (lo, hi)
    }
}

/// MacKinnon's (1994) asymptotic p-value of an ADF t, as `plm::purtest(p.approx =
/// "MacKinnon1994")`, with `t` held inside the range where the surface rises (so p is
/// non-decreasing; plm evaluates the polynomials beyond it). It takes no degrees of freedom:
/// the surface is the asymptotic one.
fn mackinnon_p(t: f64, model: PanelModel) -> f64 {
    let m = MacKinnon94::of(model);
    let (lo, hi) = m.range();
    let t = t.clamp(lo, hi);
    let z = if t <= m.limit {
        let [b0, b1, b2] = m.small;
        b0 + b1 * t + b2 * (t * t)
    } else {
        let [c0, c1, c2, c3] = m.large;
        c0 + c1 * t + c2 * (t * t) + c3 * (t * t * t)
    };
    normal_cdf(z)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Hadri Test
// ═══════════════════════════════════════════════════════════════════════════════

/// Hadri (2000) panel stationarity test.
///
/// Tests H0: All panels are stationary
/// vs H1: Some panels have unit roots
///
/// Based on KPSS-type LM statistics averaged across panels.
fn run_hadri_test(
    panels: &[(String, Vec<f64>)],
    model: PanelModel,
) -> EconResult<PanelUnitRootResult> {
    let mut lm_stats: Vec<f64> = Vec::with_capacity(panels.len());
    for (_, series) in panels {
        resource::checkpoint()?;
        lm_stats.push(compute_kpss_statistic(series, model)?);
    }

    let n_valid = lm_stats.len();
    let used: Vec<usize> = (0..n_valid).collect();

    // Average LM statistic
    let lm_bar: f64 = lm_stats.iter().sum::<f64>() / n_valid as f64;

    // Get moments for KPSS statistic
    let (mu, sigma2) = get_hadri_moments(model);

    // Z-statistic
    let z_stat = (n_valid as f64).sqrt() * (lm_bar - mu) / sigma2.sqrt();

    // P-value (one-sided, upper tail)
    let p_value = Normal::standard().sf(z_stat);

    let mut r = unit_root_result(
        PanelUnitRootTest::Hadri,
        model,
        z_stat,
        p_value,
        ("All panels are stationary", "Some panels have unit roots"),
        panels,
        &used,
        Vec::new(),
        Vec::new(),
    );
    r.unit_statistics = Some(lm_stats);
    Ok(r)
}

/// Compute KPSS-type LM statistic for a single series.
fn compute_kpss_statistic(series: &[f64], model: PanelModel) -> EconResult<f64> {
    let t = series.len();
    if t < 3 {
        return Err(EconError::InsufficientData {
            required: 3,
            provided: t,
            context: "KPSS statistic".to_string(),
        });
    }

    // Detrend the series
    let residuals: Vec<f64> = match model {
        PanelModel::None => series.to_vec(),
        PanelModel::Intercept => {
            let mean: f64 = series.iter().sum::<f64>() / t as f64;
            series.iter().map(|&x| x - mean).collect()
        }
        PanelModel::Trend => {
            // Regress on constant and trend (full rank for t >= 2)
            let mut x = resource::try_zeros2(t, 2, "KPSS trend regression")?;
            for i in 0..t {
                x[[i, 0]] = 1.0;
                x[[i, 1]] = (i + 1) as f64;
            }
            least_squares(x.view(), ArrayView1::from(series))?
                .residuals
                .to_vec()
        }
    };

    // Compute partial sums S_t = Σ_{s=1}^{t} e_s
    let mut s: Vec<f64> = Vec::with_capacity(t);
    let mut cumsum = 0.0;
    for e in &residuals {
        cumsum += e;
        s.push(cumsum);
    }

    // Long-run variance estimate (simple Newey-West with automatic bandwidth)
    let bandwidth = ((t as f64).powf(1.0 / 3.0)).ceil() as usize;
    let mut sigma2_lr = residuals.iter().map(|e| e * e).sum::<f64>() / t as f64;

    for j in 1..=bandwidth {
        let weight = 1.0 - j as f64 / (bandwidth + 1) as f64;
        let mut gamma_j = 0.0;
        for i in j..t {
            gamma_j += residuals[i] * residuals[i - j];
        }
        gamma_j /= t as f64;
        sigma2_lr += 2.0 * weight * gamma_j;
    }

    sigma2_lr = sigma2_lr.max(1e-10);

    // LM statistic: η = (1/T²) Σ S_t² / σ²_LR
    let sum_s2: f64 = s.iter().map(|si| si * si).sum();
    let lm = sum_s2 / ((t * t) as f64 * sigma2_lr);

    Ok(lm)
}

/// Get Hadri (2000) moments for LM statistic under H0.
fn get_hadri_moments(model: PanelModel) -> (f64, f64) {
    match model {
        PanelModel::None | PanelModel::Intercept => {
            // Individual intercept: μ = 1/6, σ² = 1/45
            (1.0 / 6.0, 1.0 / 45.0)
        }
        PanelModel::Trend => {
            // Individual intercept and trend: μ = 1/15, σ² = 11/6300
            (1.0 / 15.0, 11.0 / 6300.0)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn create_panel_dataset() -> Dataset {
        // Create a balanced panel with 10 units and 20 time periods
        let n_units = 10;
        let n_time = 20;

        let mut unit_ids = Vec::new();
        let mut time_ids = Vec::new();
        let mut y_stationary = Vec::new();
        let mut y_unit_root = Vec::new();

        let mut rng_seed: u64 = 12345;
        let next_rand = |seed: &mut u64| -> f64 {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((*seed >> 33) as f64 / 2147483648.0) - 1.0
        };

        for i in 0..n_units {
            let mut y_i_stat = 0.0;
            let mut y_i_ur = 0.0;

            for t in 0..n_time {
                unit_ids.push(i as i64);
                time_ids.push(t as i64);

                let eps = next_rand(&mut rng_seed) * 0.5;

                // Stationary: AR(1) with |ρ| < 1
                y_i_stat = 0.5 * y_i_stat + eps + (i as f64) * 0.1; // unit-specific intercept
                y_stationary.push(y_i_stat);

                // Unit root: random walk
                y_i_ur += eps;
                y_unit_root.push(y_i_ur + (i as f64) * 0.1);
            }
        }

        let df = DataFrame::new(vec![
            Column::new("unit".into(), &unit_ids),
            Column::new("time".into(), &time_ids),
            Column::new("y_stat".into(), &y_stationary),
            Column::new("y_ur".into(), &y_unit_root),
        ])
        .unwrap();

        Dataset::new(df)
    }

    #[test]
    fn test_llc_stationary() {
        let dataset = create_panel_dataset();
        let config = PanelUnitRootConfig {
            test_type: PanelUnitRootTest::LLC,
            model: PanelModel::Intercept,
            lags: Some(1),
            ..Default::default()
        };

        let result = run_panel_unit_root(&dataset, "y_stat", "unit", "time", config).unwrap();

        assert_eq!(result.test_type, PanelUnitRootTest::LLC);
        assert_eq!(result.n_panels, 10);
        assert!(result.statistic.is_finite());
        assert!((0.0..=1.0).contains(&result.p_value));
    }

    #[test]
    fn test_llc_unit_root() {
        let dataset = create_panel_dataset();
        let config = PanelUnitRootConfig {
            test_type: PanelUnitRootTest::LLC,
            model: PanelModel::Intercept,
            lags: Some(1),
            ..Default::default()
        };

        let result = run_panel_unit_root(&dataset, "y_ur", "unit", "time", config).unwrap();

        // The value is not checked: the LLC statistic is ECON-41 (Tier 4).
        assert_eq!(result.n_panels, 10);
        assert_eq!(result.lags_used, 1);
        assert!(result.statistic.is_finite());
        assert!((0.0..=1.0).contains(&result.p_value));
    }

    #[test]
    fn test_ips_basic() {
        let dataset = create_panel_dataset();
        let config = PanelUnitRootConfig {
            test_type: PanelUnitRootTest::IPS,
            model: PanelModel::Intercept,
            lags: Some(1),
            ..Default::default()
        };

        let result = run_panel_unit_root(&dataset, "y_stat", "unit", "time", config).unwrap();

        assert_eq!(result.test_type, PanelUnitRootTest::IPS);
        assert_eq!(result.unit_statistics.as_ref().map(Vec::len), Some(10));
        assert!(result.statistic.is_finite());
    }

    #[test]
    fn test_fisher_basic() {
        let dataset = create_panel_dataset();
        let config = PanelUnitRootConfig {
            test_type: PanelUnitRootTest::Fisher,
            model: PanelModel::Intercept,
            lags: Some(1),
            ..Default::default()
        };

        let result = run_panel_unit_root(&dataset, "y_stat", "unit", "time", config).unwrap();

        assert_eq!(result.test_type, PanelUnitRootTest::Fisher);
        assert_eq!(result.unit_p_values.as_ref().map(Vec::len), Some(10));
        assert!((0.0..=1.0).contains(&result.p_value));
    }

    #[test]
    fn test_hadri_basic() {
        let dataset = create_panel_dataset();
        let config = PanelUnitRootConfig {
            test_type: PanelUnitRootTest::Hadri,
            model: PanelModel::Intercept,
            ..Default::default()
        };

        let result = run_panel_unit_root(&dataset, "y_stat", "unit", "time", config).unwrap();

        assert_eq!(result.test_type, PanelUnitRootTest::Hadri);
        assert_eq!(result.n_panels, 10);
        assert!(result.unit_lags.is_empty());
        assert!(result.statistic.is_finite());
        assert!((0.0..=1.0).contains(&result.p_value));
    }

    #[test]
    fn test_display() {
        let result = PanelUnitRootResult {
            test_type: PanelUnitRootTest::LLC,
            model: PanelModel::Intercept,
            statistic: -2.5,
            p_value: 0.006,
            null_hypothesis: "Unit root".to_string(),
            alternative_hypothesis: "Stationary".to_string(),
            n_panels: 10,
            avg_time_periods: 20.0,
            lags_used: 1,
            unit_statistics: None,
            unit_p_values: None,
            warnings: vec!["w".into()],
            missing: Default::default(),
            unit_ids: Vec::new(),
            unit_lags: vec![1, 2, 1],
            lag_selection: "AIC per unit, pmax = 2".into(),
            dfcor: false,
        };

        let display = format!("{}", result);
        assert!(display.contains("Levin-Lin-Chu"));
        assert!(display.contains("10"));
        assert!(display.contains("Warnings:"));
        assert!(
            display.contains("Lags:       AIC per unit, pmax = 2\n"),
            "{display}"
        );
        assert!(
            display.contains("per unit: min 1, max 2, mean 1.33"),
            "{display}"
        );
        assert!(
            display.contains("rss/n (dfcor = false, plm's default)"),
            "{display}"
        );
    }
    // ── Data extraction (validation/r/cases/econ_dynamic_panel.R, panel_pur_extraction) ──

    use p2a_validation::{Check, Tol};

    /// A committed CSV loaded as `p2a data load` loads it; for the property tests, which
    /// compare no R value and so load no case.
    fn repo_dataset(rel: &str) -> Dataset {
        Dataset::new(crate::data::DataLoader::load_csv(p2a_validation::repo_path(rel)).unwrap())
    }

    const AB_BALANCED: &str = "validation/datasets/generated/ab_balanced.csv";
    const AB_COMMON_GAP: &str = "validation/datasets/generated/ab_common_gap.csv";

    fn ips_one_lag() -> PanelUnitRootConfig {
        let mut c = PanelUnitRootConfig::default();
        c.test_type = PanelUnitRootTest::IPS;
        c.model = PanelModel::Intercept;
        c.lags = Some(1);
        c
    }

    /// `ds` with column `column` added or replaced.
    fn with_column(ds: &Dataset, column: Column) -> Dataset {
        let mut df = ds.df().clone();
        df.with_column(column).unwrap();
        Dataset::new(df)
    }

    /// The `year` column as f64.
    fn years_f64(ds: &Dataset) -> Vec<f64> {
        ds.df()
            .column("year")
            .unwrap()
            .i64()
            .unwrap()
            .iter()
            .map(|y| y.unwrap() as f64)
            .collect()
    }

    /// Text unit ids, a year no unit observes (2003) and 10 missing y cells: the per-unit
    /// ADF taus of urca::ur.df on each unit's observations in time order. Before: text ids
    /// gave `EmptyDataset`.
    #[test]
    fn test_validate_panel_pur_extraction_common_gap() {
        let (case, ds) = crate::test_support::load_case("panel_pur_extraction");
        let r = run_panel_unit_root(&ds, "y", "unit", "year", ips_one_lag()).unwrap();
        let taus = r.unit_statistics.clone().unwrap();
        let mut c = Check::new(&case);
        c.vector("unit_tau", &taus, Tol::stat(690))
            .scalar("n_panels", r.n_panels as f64, Tol::exact())
            .scalar("avg_t", r.avg_time_periods, Tol::exact())
            .scalar("n_rows_dropped", r.missing.n_dropped as f64, Tol::exact());
        c.finish();
        assert_eq!(r.missing.by_column, [("y".to_string(), 10)]);
        let note = r.missing.note().unwrap();
        assert!(r.warnings.contains(&note), "{:?}", r.warnings);
        assert!(
            r.warnings
                .iter()
                .any(|w| w.starts_with("100 of 100 unit(s) have gaps in time")),
            "{:?}",
            r.warnings
        );
        // The note is the last line of the text, and only there.
        let text = r.to_string();
        assert!(text.contains("Warnings:"), "{text}");
        assert!(
            text.trim_end()
                .ends_with(&format!("Missing values: {note}")),
            "{text}"
        );
        assert_eq!(text.matches(note.as_str()).count(), 1, "{text}");
        // Numeric ids give the same units in the same order.
        let by_id = run_panel_unit_root(&ds, "y", "id", "year", ips_one_lag()).unwrap();
        assert_eq!(by_id.unit_statistics.unwrap(), taus);
    }

    /// Before: `EmptyDataset` (the truncated times left 2 values per unit).
    #[test]
    fn test_pur_fractional_time_is_an_error() {
        let ds = repo_dataset(AB_BALANCED);
        let quarters: Vec<f64> = years_f64(&ds)
            .iter()
            .map(|y| 2000.0 + (y - 2000.0) / 4.0)
            .collect();
        let ds = with_column(&ds, Column::new("qtime".into(), quarters));
        let e = run_panel_unit_root(&ds, "y", "id", "qtime", ips_one_lag()).unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message } if message.contains("whole numbers")),
            "{e:?}"
        );
    }

    /// Before: `Ok`, the second row overwrote the first.
    #[test]
    fn test_pur_duplicate_unit_time_is_an_error() {
        let ds = repo_dataset(AB_BALANCED);
        let mut df = ds.df().clone();
        df.vstack_mut(&ds.df().slice(2, 1)).unwrap();
        let e =
            run_panel_unit_root(&Dataset::new(df), "y", "id", "year", ips_one_lag()).unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message }
                if message.contains("Unit '1' has 2 rows for time 2002")),
            "{e:?}"
        );
    }

    /// Before: the unit vanished silently.
    #[test]
    fn test_pur_short_units_dropped_with_warning() {
        let ds = repo_dataset(AB_BALANCED);
        let df = ds.df();
        let ids = df.column("id").unwrap().i64().unwrap();
        let years = df.column("year").unwrap().i64().unwrap();
        let keep: polars::prelude::BooleanChunked = ids
            .iter()
            .zip(years.iter())
            .map(|(i, y)| !(i == Some(7) && y.is_some_and(|y| y >= 2002)))
            .collect();
        let short = Dataset::new(df.filter(&keep).unwrap());
        let r = run_panel_unit_root(&short, "y", "id", "year", ips_one_lag()).unwrap();
        assert_eq!(r.n_panels, 99);
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("Dropped 1 unit(s) with fewer than 3 observations: '7'")),
            "{:?}",
            r.warnings
        );
    }

    #[test]
    fn test_na_pur_int64_and_float64_time() {
        let ds = repo_dataset(AB_BALANCED);
        let float_years = with_column(&ds, Column::new("year".into(), years_f64(&ds)));
        let a = run_panel_unit_root(&ds, "y", "id", "year", ips_one_lag()).unwrap();
        let b = run_panel_unit_root(&float_years, "y", "id", "year", ips_one_lag()).unwrap();
        assert_eq!(a.statistic.to_bits(), b.statistic.to_bits());
        assert_eq!(a.unit_statistics, b.unit_statistics);
    }

    #[test]
    fn test_na_pur_text_variable_is_non_numeric() {
        let ds = repo_dataset(AB_COMMON_GAP);
        let e = run_panel_unit_root(&ds, "unit", "id", "year", ips_one_lag()).unwrap_err();
        assert!(
            matches!(&e, EconError::NonNumericColumn { column } if column == "unit"),
            "{e:?}"
        );
    }

    // ── Lag selection (validation/r/cases/econ_purtest.R, purtest_lag_selection) ──

    fn purtest(
        ds: &Dataset,
        test: PanelUnitRootTest,
        model: PanelModel,
        sel: LagSelection,
        dfcor: bool,
    ) -> PanelUnitRootResult {
        let mut c = PanelUnitRootConfig::default();
        c.test_type = test;
        c.model = model;
        c.lag_selection = Some(sel);
        c.pmax = Some(4);
        c.dfcor = dfcor;
        run_panel_unit_root(ds, "y", "id", "t", c).unwrap()
    }

    const RULES: [(LagSelection, &str); 3] = [
        (LagSelection::Aic, "aic"),
        (LagSelection::Sic, "sic"),
        (LagSelection::Hall, "hall"),
    ];

    fn as_f64(v: &[usize]) -> Vec<f64> {
        v.iter().map(|&p| p as f64).collect()
    }

    fn rule_label(sel: LagSelection) -> String {
        format!("{sel} per unit, pmax = 4")
    }

    fn twelve_ids() -> Vec<String> {
        (1..=12).map(|i| i.to_string()).collect()
    }

    #[test]
    fn test_validate_purtest_lag_selection_ips() {
        let (case, ds) = crate::test_support::load_case("purtest_lag_selection");
        // deviation ECON-43: the W-tbar statistic uses ad-hoc moments; ips_*_stat/_p are not compared
        let mut c = Check::new(&case);
        for (sel, rule) in RULES {
            for dfcor in [false, true] {
                let sfx = if dfcor { "_dfcor" } else { "" };
                let r = purtest(
                    &ds,
                    PanelUnitRootTest::IPS,
                    PanelModel::Intercept,
                    sel,
                    dfcor,
                );
                c.vector(
                    &format!("ips_{rule}{sfx}_lags"),
                    &as_f64(&r.unit_lags),
                    Tol::exact(),
                )
                .vector(
                    &format!("trho_{rule}{sfx}"),
                    r.unit_statistics.as_deref().unwrap(),
                    Tol::stat(480),
                );
                assert_eq!(r.n_panels, 12);
                assert_eq!(r.lags_used, *r.unit_lags.iter().max().unwrap());
                assert_eq!(r.unit_ids, twelve_ids());
                assert_eq!(r.lag_selection, rule_label(sel));
                assert_eq!(r.dfcor, dfcor);
            }
        }
        let r = purtest(
            &ds,
            PanelUnitRootTest::IPS,
            PanelModel::Trend,
            LagSelection::Aic,
            true,
        );
        c.vector("trend_lags_aic_dfcor", &as_f64(&r.unit_lags), Tol::exact())
            .vector(
                "trend_trho_aic_dfcor",
                r.unit_statistics.as_deref().unwrap(),
                Tol::stat(480),
            );
        c.finish();
    }

    #[test]
    fn test_validate_purtest_lag_selection_llc() {
        let (case, ds) = crate::test_support::load_case("purtest_lag_selection");
        // deviation ECON-41: levinlin_*_stat/_p not compared
        let mut c = Check::new(&case);
        for (sel, rule) in RULES {
            for dfcor in [false, true] {
                let sfx = if dfcor { "_dfcor" } else { "" };
                let r = purtest(
                    &ds,
                    PanelUnitRootTest::LLC,
                    PanelModel::Intercept,
                    sel,
                    dfcor,
                );
                c.vector(
                    &format!("levinlin_{rule}{sfx}_lags"),
                    &as_f64(&r.unit_lags),
                    Tol::exact(),
                );
                assert_eq!(r.lags_used, *r.unit_lags.iter().max().unwrap());
                assert_eq!(r.lag_selection, rule_label(sel));
                assert_eq!(r.unit_ids, twelve_ids());
            }
        }
        c.finish();
    }

    #[test]
    fn test_validate_purtest_lag_selection_madwu() {
        let (case, ds) = crate::test_support::load_case("purtest_lag_selection");
        // deviation ECON-44: madwu_*_stat/_p and ptrho_* are plm's default p.approx =
        // MacKinnon1996 (urca's tables); p2a uses MacKinnon (1994): asserted close, not pinned.
        let mut c = Check::new(&case);
        for (sel, rule) in RULES {
            for dfcor in [false, true] {
                let sfx = if dfcor { "_dfcor" } else { "" };
                let r = purtest(
                    &ds,
                    PanelUnitRootTest::Fisher,
                    PanelModel::Intercept,
                    sel,
                    dfcor,
                );
                c.vector(
                    &format!("madwu_{rule}{sfx}_lags"),
                    &as_f64(&r.unit_lags),
                    Tol::exact(),
                )
                .vector(
                    &format!("trho_{rule}{sfx}"),
                    r.unit_statistics.as_deref().unwrap(),
                    Tol::stat(480),
                );
                assert_eq!(r.n_panels, 12);
                let m96 = case.scalar(&format!("madwu_{rule}{sfx}_stat"));
                assert!(
                    (r.statistic - m96).abs() <= 0.02 * m96,
                    "{rule}{sfx}: {} vs MacKinnon (1996) {m96}",
                    r.statistic
                );
            }
        }
        let r = purtest(
            &ds,
            PanelUnitRootTest::Fisher,
            PanelModel::None,
            LagSelection::Sic,
            true,
        );
        c.vector("none_lags_sic_dfcor", &as_f64(&r.unit_lags), Tol::exact())
            .vector(
                "none_trho_sic_dfcor",
                r.unit_statistics.as_deref().unwrap(),
                Tol::stat(480),
            );
        c.finish();
    }

    /// Property test (no R value): AIC and SIC, and Hall under the two variances, choose
    /// different orders on `purtest_panel`.
    #[test]
    fn test_lag_selection_rules_give_different_lags() {
        let ds = repo_dataset("validation/datasets/generated/purtest_panel.csv");
        let ips = |sel, dfcor| {
            purtest(
                &ds,
                PanelUnitRootTest::IPS,
                PanelModel::Intercept,
                sel,
                dfcor,
            )
        };
        let aic = ips(LagSelection::Aic, false);
        let sic = ips(LagSelection::Sic, false);
        assert_eq!(aic.unit_lags, [4, 1, 2, 3, 1, 2, 3, 4, 1, 0, 2, 1]);
        assert_eq!(sic.unit_lags, [2, 1, 1, 1, 1, 2, 2, 0, 1, 0, 1, 1]);
        assert_ne!(aic.unit_statistics, sic.unit_statistics);
        let hall = ips(LagSelection::Hall, false);
        let hall_dfcor = ips(LagSelection::Hall, true);
        assert_eq!((hall.unit_lags[0], hall_dfcor.unit_lags[0]), (4, 2));
        assert_eq!(hall.unit_lags[1..], hall_dfcor.unit_lags[1..]);
    }

    #[test]
    fn test_panel_unit_root_test_parses_aliases_and_rejects_unknown() {
        PanelUnitRootTest::choices().validate().unwrap();
        for (s, t) in [
            ("llc", PanelUnitRootTest::LLC),
            ("LLC", PanelUnitRootTest::LLC),
            ("levinlin", PanelUnitRootTest::LLC),
            ("Levin-Lin-Chu", PanelUnitRootTest::LLC),
            ("ips", PanelUnitRootTest::IPS),
            ("im_pesaran_shin", PanelUnitRootTest::IPS),
            ("madwu", PanelUnitRootTest::Fisher),
            ("Maddala-Wu", PanelUnitRootTest::Fisher),
            ("fisher", PanelUnitRootTest::Fisher),
            ("hadri", PanelUnitRootTest::Hadri),
        ] {
            assert_eq!(s.parse::<PanelUnitRootTest>().unwrap(), t, "{s}");
        }
        let e = "llc2".parse::<PanelUnitRootTest>().unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message }
                if message.contains("'llc2'") && message.contains("llc, ips, madwu, hadri")),
            "{e:?}"
        );
        // The long Display labels stay.
        assert_eq!(PanelUnitRootTest::Fisher.to_string(), "Fisher (Maddala-Wu)");
    }

    #[test]
    fn test_panel_model_parses_aliases_and_rejects_unknown() {
        PanelModel::choices().validate().unwrap();
        for (s, m) in [
            ("none", PanelModel::None),
            ("nc", PanelModel::None),
            ("intercept", PanelModel::Intercept),
            ("drift", PanelModel::Intercept),
            ("c", PanelModel::Intercept),
            ("const", PanelModel::Intercept),
            ("trend", PanelModel::Trend),
            ("ct", PanelModel::Trend),
        ] {
            assert_eq!(s.parse::<PanelModel>().unwrap(), m, "{s}");
        }
        let e = "quadratic".parse::<PanelModel>().unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message }
                if message.contains("none, intercept, trend")),
            "{e:?}"
        );
    }

    #[test]
    fn test_lag_selection_parses_aliases_and_rejects_unknown() {
        LAG_RULES.validate().unwrap();
        assert_eq!(
            LagSelection::from_option("AIC", None).unwrap(),
            LagSelection::Aic
        );
        assert_eq!(
            LagSelection::from_option("bic", None).unwrap(),
            LagSelection::Sic
        );
        assert_eq!(
            LagSelection::from_option("Schwarz", None).unwrap(),
            LagSelection::Sic
        );
        assert_eq!(
            LagSelection::from_option("t-sig", None).unwrap(),
            LagSelection::Hall
        );
        assert_eq!(
            LagSelection::from_option("fixed", Some(2)).unwrap(),
            LagSelection::Fixed(2)
        );
        for (name, lags) in [("fixed", None), ("aic", Some(2))] {
            let e = LagSelection::from_option(name, lags).unwrap_err();
            assert!(
                matches!(&e, EconError::InvalidSpecification { message } if message.contains("lags")),
                "{e:?}"
            );
        }
        let e = LagSelection::from_option("gts", None).unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message }
                if message.contains("fixed, aic, sic, hall")),
            "{e:?}"
        );
        assert_eq!(LagSelection::names_help(), "fixed | aic | sic | hall");
        assert_eq!(LagSelection::Fixed(2).to_string(), "fixed (2)");
    }

    #[test]
    fn test_lag_selection_config_conflicts_are_errors() {
        let ds = repo_dataset("validation/datasets/generated/purtest_panel.csv");
        let run = |f: &dyn Fn(&mut PanelUnitRootConfig)| {
            let mut c = PanelUnitRootConfig::default();
            c.test_type = PanelUnitRootTest::IPS;
            f(&mut c);
            run_panel_unit_root(&ds, "y", "id", "t", c)
        };
        let message = |r: EconResult<PanelUnitRootResult>| match r {
            Err(EconError::InvalidSpecification { message }) => message,
            other => panic!("{other:?}"),
        };
        assert_eq!(
            message(run(&|c| {
                c.lag_selection = Some(LagSelection::Sic);
                c.lags = Some(2);
            })),
            "lags fixes the lag order and cannot be combined with lag_selection = sic (use pmax)"
        );
        assert_eq!(
            message(run(&|c| {
                c.lag_selection = Some(LagSelection::Fixed(2));
                c.lags = Some(3);
            })),
            "lags = 3 conflicts with lag_selection = fixed (2)"
        );
        assert_eq!(
            message(run(&|c| {
                c.lag_selection = Some(LagSelection::Aic);
                c.pmax = Some(0);
            })),
            "pmax must be at least 1 for lag selection (got 0)"
        );
        assert_eq!(
            message(run(&|c| {
                c.lag_selection = Some(LagSelection::Hall);
                c.pmax = Some(3);
                c.max_lags = Some(4);
            })),
            "pmax and max_lags give different maximum lags (3, 4)"
        );
        assert_eq!(
            message(run(&|c| {
                c.test_type = PanelUnitRootTest::Hadri;
                c.lag_selection = Some(LagSelection::Aic);
            })),
            "the Hadri test uses no lagged differences; lag_selection and pmax do not apply"
        );
        assert_eq!(
            message(run(&|c| c.pmax = Some(2))),
            "pmax applies only to lag_selection = aic, sic or hall"
        );
        // pmax = 19 needs 3 + 1 + 38 = 42 observations; every unit has 40.
        match run(&|c| {
            c.lag_selection = Some(LagSelection::Aic);
            c.pmax = Some(19);
        }) {
            Err(EconError::InsufficientData {
                required: 42,
                provided: 40,
                context,
            }) => assert!(context.contains("pmax = 19"), "{context}"),
            other => panic!("{other:?}"),
        }
        // Equal pmax and max_lags agree.
        let r = run(&|c| {
            c.lag_selection = Some(LagSelection::Aic);
            c.pmax = Some(4);
            c.max_lags = Some(4);
        })
        .unwrap();
        assert_eq!(r.lag_selection, "AIC per unit, pmax = 4");
    }

    #[test]
    fn test_default_lag_rule_is_unchanged() {
        let ds = repo_dataset("validation/datasets/generated/purtest_panel.csv");
        let mut c = PanelUnitRootConfig::default();
        c.test_type = PanelUnitRootTest::IPS;
        let r = run_panel_unit_root(&ds, "y", "id", "t", c.clone()).unwrap();
        assert_eq!(r.lags_used, 3);
        assert!(r.unit_lags.iter().all(|&p| p == 3), "{:?}", r.unit_lags);
        assert_eq!(r.lag_selection, "3 (fixed: floor(T_max^(1/3)))");
        c.lag_selection = Some(LagSelection::Fixed(3));
        let fixed = run_panel_unit_root(&ds, "y", "id", "t", c.clone()).unwrap();
        assert_eq!(fixed.lag_selection, "3 (fixed by the user)");
        let bits = |r: &PanelUnitRootResult| {
            r.unit_statistics
                .as_ref()
                .unwrap()
                .iter()
                .map(|t| t.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&r), bits(&fixed));
        // A binding max_lags is labelled so.
        c.lag_selection = None;
        c.max_lags = Some(2);
        let capped = run_panel_unit_root(&ds, "y", "id", "t", c).unwrap();
        assert_eq!(
            capped.lag_selection,
            "2 (fixed: min(floor(T_max^(1/3)), max_lags))"
        );
    }

    // ── Short and degenerate units (ECON-42) ──

    /// Units 1..=k with `lengths[k-1]` periods of an AR(1) (rho 0.5) with uniform noise.
    fn noisy_panel(lengths: &[usize], seed: u64) -> Dataset {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(12345);
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / 2147483648.0) - 1.0
        };
        let (mut unit, mut time, mut y) = (Vec::new(), Vec::new(), Vec::new());
        for (k, &len) in lengths.iter().enumerate() {
            let mut v = 0.0;
            for t in 0..len {
                v = 0.5 * v + next();
                unit.push(k as i64 + 1);
                time.push(t as i64 + 1);
                y.push(v);
            }
        }
        Dataset::new(
            DataFrame::new(vec![
                Column::new("unit".into(), unit),
                Column::new("time".into(), time),
                Column::new("y".into(), y),
            ])
            .unwrap(),
        )
    }

    /// `noisy_panel(&[t_len; k], seed)` plus unit `k + 1` with y = 2, 4, ..., 2 t_len.
    fn noisy_panel_with_linear_unit(k: usize, t_len: usize, seed: u64) -> Dataset {
        let mut df = noisy_panel(&vec![t_len; k], seed).df().clone();
        let line = DataFrame::new(vec![
            Column::new("unit".into(), vec![k as i64 + 1; t_len]),
            Column::new("time".into(), (1..=t_len as i64).collect::<Vec<_>>()),
            Column::new(
                "y".into(),
                (1..=t_len).map(|t| 2.0 * t as f64).collect::<Vec<_>>(),
            ),
        ])
        .unwrap();
        df.vstack_mut(&line).unwrap();
        Dataset::new(df)
    }

    fn cfg(test: PanelUnitRootTest, model: PanelModel, lags: Option<usize>) -> PanelUnitRootConfig {
        let mut c = PanelUnitRootConfig::default();
        c.test_type = test;
        c.model = model;
        c.lags = lags;
        c
    }

    #[test]
    fn test_adf_rejects_short_panels() {
        let short = noisy_panel(&[6; 5], 1);
        for t in [
            PanelUnitRootTest::IPS,
            PanelUnitRootTest::Fisher,
            PanelUnitRootTest::LLC,
        ] {
            // intercept, 2 lags: needs 3 + 1 + 4 = 8 observations
            match run_panel_unit_root(
                &short,
                "y",
                "unit",
                "time",
                cfg(t, PanelModel::Intercept, Some(2)),
            ) {
                Err(EconError::InsufficientData {
                    required: 8,
                    provided: 6,
                    context,
                }) => assert!(context.contains("2 lag"), "{context}"),
                other => panic!("{t:?}: {other:?}"),
            }
        }
        // default rule floor(5^(1/3)) = 1 with a trend: needs 7 > 5
        assert!(matches!(
            run_panel_unit_root(
                &noisy_panel(&[5; 5], 2),
                "y",
                "unit",
                "time",
                cfg(PanelUnitRootTest::IPS, PanelModel::Trend, None)
            ),
            Err(EconError::InsufficientData {
                required: 7,
                provided: 5,
                ..
            })
        ));
        // df exactly 0 (trend, T = 8, 2 lags)
        assert!(matches!(
            run_panel_unit_root(
                &noisy_panel(&[8; 5], 3),
                "y",
                "unit",
                "time",
                cfg(PanelUnitRootTest::IPS, PanelModel::Trend, Some(2))
            ),
            Err(EconError::InsufficientData {
                required: 9,
                provided: 8,
                ..
            })
        ));
        // unbalanced: nine units of 40 and one of 6 -> the short unit is skipped and named
        let mut lengths = vec![40; 9];
        lengths.push(6);
        let r = run_panel_unit_root(
            &noisy_panel(&lengths, 4),
            "y",
            "unit",
            "time",
            cfg(PanelUnitRootTest::IPS, PanelModel::Intercept, Some(2)),
        )
        .unwrap();
        assert_eq!(
            (
                r.n_panels,
                r.unit_statistics.as_ref().unwrap().len(),
                r.unit_lags.len()
            ),
            (9, 9, 9)
        );
        assert!(
            r.warnings.iter().any(|w| w
                == "Skipped 1 unit(s) whose ADF regression cannot be estimated: '10' (6 \
                    observations; 2 lag(s) with an intercept need at least 8)"),
            "{:?}",
            r.warnings
        );
        assert!(r.to_string().contains("Skipped 1 unit(s)"));
    }

    #[test]
    fn test_adf_and_llc_skip_rank_deficient_units() {
        // unit 7 is the exact line y = 2 t: its lagged difference duplicates the intercept
        let ds = noisy_panel_with_linear_unit(6, 20, 5);
        for t in [
            PanelUnitRootTest::IPS,
            PanelUnitRootTest::Fisher,
            PanelUnitRootTest::LLC,
        ] {
            let r = run_panel_unit_root(
                &ds,
                "y",
                "unit",
                "time",
                cfg(t, PanelModel::Intercept, Some(1)),
            )
            .unwrap();
            assert_eq!(r.n_panels, 6, "{t:?}");
            assert_eq!(r.unit_ids, ["1", "2", "3", "4", "5", "6"], "{t:?}");
            assert!(
                r.warnings.iter().any(|w| w.starts_with(
                    "Skipped 1 unit(s) whose ADF regression cannot be estimated: '7' (rank \
                     deficient"
                )),
                "{t:?} {:?}",
                r.warnings
            );
        }
    }

    // ── T3-PERF1: the LLC auxiliary regressions take OLS's solve decision

    /// The LLC auxiliary design of `series` (as `compute_llc_residuals` builds it).
    fn llc_design(
        series: &[f64],
        model: PanelModel,
        lags: usize,
    ) -> (Array2<f64>, Vec<f64>, Vec<f64>) {
        let d: Vec<f64> = series.windows(2).map(|w| w[1] - w[0]).collect();
        let t = d.len();
        let nd = n_det(model);
        let x = Array2::from_shape_fn((t - lags, nd + lags), |(r, c)| {
            let i = r + lags;
            if c < nd {
                if c == 0 { 1.0 } else { (i + 1) as f64 }
            } else {
                d[i - (c - nd) - 1]
            }
        });
        (x, d[lags..].to_vec(), series[lags..t].to_vec())
    }

    /// On noisy AR(1) units the normal-equations residuals of both auxiliary regressions
    /// agree with lm.fit's QR residuals to 1e-10 relative (to the residual norm), with an
    /// intercept and with a trend (whose uncentred time column is refined once); an exactly
    /// linear unit still reaches the QR and is skipped as rank deficient.
    #[test]
    fn test_llc_residuals_normal_equations_match_qr() {
        // AR(1) units, rho 0.5, uniform noise (as noisy_panel)
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut y = Vec::new();
        for len in [60usize, 100, 35] {
            let mut v = 0.0;
            let mut s = Vec::new();
            for _ in 0..len {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                v = 0.5 * v + ((state >> 33) as f64 / 2147483648.0) - 1.0;
                s.push(v);
            }
            y.push(s);
        }
        let units = [&y[0][..], &y[1][..], &y[2][..]];
        for model in [PanelModel::Intercept, PanelModel::Trend] {
            for lags in [1usize, 4] {
                for s in units {
                    let (x, dy, yl) = llc_design(s, model, lags);
                    assert!(normal_equations_inverse(x.view()).unwrap().is_some());
                    let (e, v) = match compute_llc_residuals(s, model, lags) {
                        Ok(r) => r,
                        Err(_) => panic!("unit not estimated"),
                    };
                    for (got, resp) in [(&e, &dy), (&v, &yl)] {
                        let r = least_squares(x.view(), ArrayView1::from(resp))
                            .unwrap()
                            .residuals;
                        let scale = r.iter().map(|a| a * a).sum::<f64>().sqrt();
                        let diff = got
                            .iter()
                            .zip(r.iter())
                            .map(|(a, b)| (a - b).abs())
                            .fold(0.0, f64::max);
                        assert!(
                            diff <= 1e-10 * scale,
                            "{model:?} lags {lags}: {diff} vs {scale}"
                        );
                    }
                }
            }
        }
        let line: Vec<f64> = (1..=40).map(|t| 2.0 * t as f64).collect();
        let (x, _, _) = llc_design(&line, PanelModel::Intercept, 2);
        assert!(normal_equations_inverse(x.view()).unwrap().is_none());
        assert!(matches!(
            compute_llc_residuals(&line, PanelModel::Intercept, 2),
            Err(UnitError::Skip { .. })
        ));
    }

    // ── Statistics (validation/r/cases/econ_purtest_t4.R) ──

    const EXOS: [(PanelModel, &str); 3] = [
        (PanelModel::Intercept, "intercept"),
        (PanelModel::Trend, "trend"),
        (PanelModel::None, "none"),
    ];

    /// `test` on column `col` of a (id, t) panel with fixed `lags`.
    fn stat_run(
        ds: &Dataset,
        col: &str,
        test: PanelUnitRootTest,
        model: PanelModel,
        lags: Option<usize>,
        dfcor: bool,
    ) -> PanelUnitRootResult {
        let mut c = cfg(test, model, lags);
        c.dfcor = dfcor;
        run_panel_unit_root(ds, col, "id", "t", c).unwrap()
    }

    /// The case key of a column, model and dfcor: "rw_trend_dfcor".
    fn stat_tag(col: &str, exo: &str, dfcor: bool) -> String {
        let rw = if col == "y" { "" } else { "rw_" };
        let sfx = if dfcor { "_dfcor" } else { "" };
        format!("{rw}{exo}{sfx}")
    }

    /// plm:::padf(q, exo, p.approx = "MacKinnon1994") on a grid from -6.5 to 3, inside the
    /// range where the surface rises; outside it p is held at the turning point's value.
    #[test]
    fn test_validate_mackinnon_p_grid() {
        let (case, _) = crate::test_support::load_case("mackinnon_punitroot_grid");
        let q = case.vector("q");
        let mut c = Check::new(&case);
        c.vector("q", &q, Tol::exact());
        let tol = Tol::p_value(191);
        let mut failures = Vec::new();
        let mut held = 0;
        for (model, exo) in EXOS {
            let expected = case.vector(&format!("p94_{exo}"));
            let (lo, hi) = MacKinnon94::of(model).range();
            let p: Vec<f64> = q.iter().map(|&t| mackinnon_p(t, model)).collect();
            for (i, &t) in q.iter().enumerate() {
                if (lo..=hi).contains(&t) {
                    if !tol.allows(p[i], expected[i]) {
                        failures.push(format!("{exo} q = {t}: {} vs {}", p[i], expected[i]));
                    }
                } else {
                    // deviation ECON-44: plm evaluates the polynomial past its turning point
                    held += 1;
                    let edge = mackinnon_p(t.clamp(lo, hi), model);
                    assert_eq!(p[i], edge, "{exo} q = {t}");
                }
            }
            assert!(p.windows(2).all(|w| w[0] <= w[1]), "{exo}: {p:?}");
        }
        assert!(failures.is_empty(), "{failures:#?}");
        // Above 2.74 with an intercept (6 points) and above 0.70 with a trend (47).
        assert_eq!(held, 53);
        c.finish();
    }

    /// Non-decreasing in t over a wide grid for every model, and the turning points are
    /// MacKinnon's (1994) tau_min and tau_max (N = 1): -19.04, -18.83, -16.18; inf, 2.74, 0.70.
    #[test]
    fn test_mackinnon_p_monotone_with_mackinnons_range() {
        for (model, (tau_min, tau_max)) in [
            (PanelModel::None, (-19.04, f64::INFINITY)),
            (PanelModel::Intercept, (-18.83, 2.74)),
            (PanelModel::Trend, (-16.18, 0.70)),
        ] {
            let (lo, hi) = MacKinnon94::of(model).range();
            assert!((lo - tau_min).abs() < 0.01, "{model:?}: {lo}");
            assert!(
                hi == tau_max || (hi - tau_max).abs() < 0.01,
                "{model:?}: {hi}"
            );
            let mut last = 0.0;
            for k in 0..=30_000 {
                let t = -25.0 + 0.001 * f64::from(k);
                let p = mackinnon_p(t, model);
                assert!(
                    p >= last && p > 0.0 && p < 1.0,
                    "{model:?} t = {t}: {p} < {last}"
                );
                last = p;
            }
        }
    }

    /// Maddala-Wu with MacKinnon's (1994) p-values: plm::purtest(test = "madwu", lags = 1,
    /// p.approx = "MacKinnon1994"); the per-unit ADF t for every model and dfcor.
    #[test]
    fn test_validate_purtest_stat_madwu() {
        let (case, ds) = crate::test_support::load_case("purtest_stat_ar05");
        let mut c = Check::new(&case);
        for (model, exo) in EXOS {
            let r = stat_run(&ds, "y", PanelUnitRootTest::Fisher, model, Some(1), false);
            c.scalar(
                &format!("madwu_{exo}_m94_stat"),
                r.statistic,
                Tol::stat(500),
            )
            .scalar(&format!("madwu_{exo}_m94_p"), r.p_value, Tol::p_value(500))
            .vector(
                &format!("madwu_{exo}_m94_ptrho"),
                r.unit_p_values.as_deref().unwrap(),
                Tol::p_value(500),
            );
            for col in ["y", "y_rw"] {
                for dfcor in [false, true] {
                    let tag = stat_tag(col, exo, dfcor);
                    let r = stat_run(&ds, col, PanelUnitRootTest::Fisher, model, Some(1), dfcor);
                    c.vector(
                        &format!("madwu_{tag}_trho"),
                        r.unit_statistics.as_deref().unwrap(),
                        Tol::stat(500),
                    );
                    // deviation ECON-44: madwu_*_stat/_p/_ptrho without m94 are plm's default
                    // p.approx = MacKinnon1996 (urca's tables); asserted close, not pinned.
                    let m96 = case.scalar(&format!("madwu_{tag}_stat"));
                    assert!(
                        (r.statistic - m96).abs() <= 0.02 * m96,
                        "{tag}: {} vs MacKinnon (1996) {m96}",
                        r.statistic
                    );
                }
            }
        }
        c.finish();
    }
}
