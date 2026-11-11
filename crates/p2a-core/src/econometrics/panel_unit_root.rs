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

/// The residual variance of Hadri's statistic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum HadriVariance {
    /// Per-unit short-run variance, plm `Hcons = TRUE` (default).
    #[default]
    Heteroskedastic,
    /// Pooled short-run variance, plm `Hcons = FALSE`.
    Homoskedastic,
    /// Per-unit Bartlett long-run variance with `lags` truncation lags (None:
    /// `trunc(4 (T/100)^(1/4))`, `tseries::kpss.test`'s short rule).
    LongRun { lags: Option<usize> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HadriVarianceName {
    Heteroskedastic,
    Homoskedastic,
    LongRun,
}

static HADRI_VARIANCES: Choices<HadriVarianceName> = Choices {
    option: "hadri_variance",
    values: &[
        (
            &["heteroskedastic", "hcons", "het"],
            HadriVarianceName::Heteroskedastic,
        ),
        (
            &["homoskedastic", "homo", "iid"],
            HadriVarianceName::Homoskedastic,
        ),
        (&["long_run", "lrv", "bartlett"], HadriVarianceName::LongRun),
    ],
};

impl HadriVariance {
    /// `name`: heteroskedastic (hcons, het) | homoskedastic (homo, iid) | long_run (lrv,
    /// bartlett); case and separator insensitive. `lags` only with long_run. An unknown name
    /// is `InvalidSpecification` listing the names.
    pub fn from_option(name: &str, lags: Option<usize>) -> EconResult<HadriVariance> {
        match (HADRI_VARIANCES.parse(name)?, lags) {
            (HadriVarianceName::LongRun, lags) => Ok(HadriVariance::LongRun { lags }),
            (v, Some(_)) => Err(invalid(format!(
                "lags of the Hadri variance apply only to hadri_variance = long_run, not {}",
                HADRI_VARIANCES.canonical(v)
            ))),
            (HadriVarianceName::Heteroskedastic, None) => Ok(HadriVariance::Heteroskedastic),
            (HadriVarianceName::Homoskedastic, None) => Ok(HadriVariance::Homoskedastic),
        }
    }

    /// "heteroskedastic | homoskedastic | long_run", for CLI help and tool descriptions.
    pub fn names_help() -> String {
        HADRI_VARIANCES.help()
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
    /// Hadri's residual variance; only with `test_type = Hadri` (a non-default value with
    /// another test is an error). Module docs, "Hadri".
    #[serde(default)]
    pub hadri_variance: HadriVariance,
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
            hadri_variance: HadriVariance::Heteroskedastic,
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
    /// Hadri: how the residual variance was computed, as printed after "Variance:" (empty
    /// otherwise).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub variance: String,
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
        if self.test_type == PanelUnitRootTest::Hadri {
            writeln!(f, "Variance:   {}", self.variance)?;
            if !self.variance.starts_with("long-run") {
                if self.dfcor {
                    writeln!(f, "Residual variance: rss/(T - k) (dfcor = true)")?;
                } else {
                    writeln!(f, "Residual variance: rss/T (dfcor = false, plm's default)")?;
                }
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
        run_hadri_test(&panels, config.model, config.hadri_variance, config.dfcor)?
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
        if config.lags.is_some() || config.lag_selection.is_some() || config.pmax.is_some() {
            return Err(invalid(
                "the Hadri test uses no lagged differences; lags, lag_selection and pmax do \
                 not apply (the long-run variance's truncation lag is set with \
                 hadri_variance = long_run)"
                    .to_string(),
            ));
        }
        return Ok(());
    }
    if config.hadri_variance != HadriVariance::default() {
        return Err(invalid(format!(
            "hadri_variance applies only to the Hadri test, not {}",
            config.test_type
        )));
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
        variance: String::new(),
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

/// Levin-Lin-Chu (2002) panel unit root test, as `plm::purtest(test = "levinlin")`.
///
/// Tests H0: All panels have a unit root (common ρ = 0)
/// vs H1: All panels are stationary (common ρ < 0)
///
/// Each unit's ADF residuals are normalised by the unit's ADF residual sd, pooled into one
/// regression whose t is corrected by the ratio of long-run to short-run sd and the mean and
/// variance adjustments of LLC (2002) Table 2. Needs a balanced panel, as plm.
fn run_llc_test(
    panels: &[(String, Vec<f64>)],
    model: PanelModel,
    plan: &LagPlan,
    dfcor: bool,
) -> EconResult<PanelUnitRootResult> {
    // Step 1: each unit's auxiliary regressions and ADF residual sd; units that cannot be
    // estimated are skipped.
    let (units, mut warnings) = skip_or_propagate(PanelUnitRootTest::LLC, panels, |series| {
        let p = unit_lag(series, plan, model, dfcor)?;
        let (e, v) = compute_llc_residuals(series, model, p)?;
        let sigma = llc_unit_sigma(series, &e, &v, model, p, dfcor)?;
        Ok((p, e, v, sigma))
    })?;
    let used: Vec<usize> = units.iter().map(|(i, _)| *i).collect();
    let unit_lags: Vec<usize> = units.iter().map(|(_, (p, ..))| *p).collect();
    let lens = used.iter().map(|&i| panels[i].1.len());
    let (t_min, t_max) = (lens.clone().min().unwrap_or(0), lens.max().unwrap_or(0));
    if t_min != t_max {
        return Err(invalid(format!(
            "the Levin-Lin-Chu test needs a balanced panel (every unit with the same number of \
             observations, as plm::purtest); the units used have {t_min} to {t_max} \
             observations; the Im-Pesaran-Shin and Maddala-Wu tests allow unbalanced panels"
        )));
    }
    let t_len = t_max;
    let n = used.len();
    let nf = n as f64;

    // Step 2: the ratio of each unit's long-run sd of Δy (Bartlett kernel, q lags) to its
    // ADF residual sd, averaged.
    let q_rule = (3.21 * (t_len as f64).powf(1.0 / 3.0)).round_ties_even() as usize;
    let q = q_rule.min(t_len - 2);
    if q < q_rule {
        warnings.push(format!(
            "Levin-Lin-Chu: the long-run variance's truncation lag is capped at T - 2 = {q} \
             (round(3.21 T^(1/3)) = {q_rule} exceeds the autocovariances of a series of {t_len})"
        ));
    }
    let mut s_bar = 0.0;
    for (&i, (_, (_, _, _, sigma))) in used.iter().zip(&units) {
        resource::checkpoint()?;
        s_bar += llc_long_run_variance(&panels[i].1, model, q)?.sqrt() / sigma;
    }
    s_bar /= nf;

    // Step 3: the pooled regression of the normalised residuals, d = e / σ on w = v / σ.
    let (mut sum_wd, mut sum_ww, mut n_obs) = (0.0, 0.0, 0usize);
    for (_, (_, e, v, sigma)) in &units {
        for (ei, vi) in e.iter().zip(v) {
            let (d, w) = (ei / sigma, vi / sigma);
            sum_wd += w * d;
            sum_ww += w * w;
        }
        n_obs += e.len();
    }
    let rho = sum_wd / sum_ww;
    let mut rss = 0.0;
    for (_, (_, e, v, sigma)) in &units {
        for (ei, vi) in e.iter().zip(v) {
            let r = ei / sigma - rho * (vi / sigma);
            rss += r * r;
        }
    }
    let s2 = if dfcor {
        rss / (n_obs - 1) as f64
    } else {
        rss / n_obs as f64
    };
    let sd_rho = (s2 / sum_ww).sqrt();
    let t_rho = rho / sd_rho;

    // Step 4: the adjusted t, z = (t_rho - N T~ S_N / σ~² sd(ρ) μ*) / σ*.
    let mean_lag = unit_lags.iter().sum::<usize>() as f64 / nf;
    let t_tilde = t_len as f64 - mean_lag - 1.0;
    let sigma_eps2 = rss / (nf * t_tilde);
    let (mu_star, sigma_star, edge) = llc_adjustment(t_len, model);
    match edge {
        Some(TableEdge::Below) => warnings.push(format!(
            "Levin-Lin-Chu: T = {t_len} is below the shortest row (T = 25) of the LLC (2002) \
             Table 2 adjustment; the T = 25 values are used, as plm::purtest does"
        )),
        Some(TableEdge::Above) => warnings.push(format!(
            "Levin-Lin-Chu: T = {t_len} is above the longest row (T = 500) of the LLC (2002) \
             Table 2 adjustment; the T = 500 values are used, as plm::purtest does"
        )),
        None => {}
    }
    let t_star = (t_rho - nf * t_tilde * s_bar / sigma_eps2 * sd_rho * mu_star) / sigma_star;
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

/// One unit's ADF residual sd from its auxiliary residuals (Frisch-Waugh-Lovell: the ADF
/// regression's rss is `e'e - (v'e)^2 / v'v`), with `rss/(n - K)` (`dfcor`) or `rss/n`, as
/// plm's `tsadf`. A lagged level that the other regressors explain (`|v| < 1e-7 |y_{t-1}|`,
/// the pivoted QR's rule) makes the ADF regression rank deficient and skips the unit.
fn llc_unit_sigma(
    series: &[f64],
    e: &[f64],
    v: &[f64],
    model: PanelModel,
    lags: usize,
    dfcor: bool,
) -> Result<f64, UnitError> {
    let k = 1 + n_det(model) + lags;
    let n = e.len();
    let level = &series[lags..series.len() - 1];
    let level_norm = level.iter().map(|x| x * x).sum::<f64>().sqrt();
    let vv: f64 = v.iter().map(|x| x * x).sum();
    let full_rank = vv > 0.0 && vv.sqrt() >= 1e-7 * level_norm;
    if !full_rank {
        return Err(rank_deficient(
            format!("the ADF regression with {lags} lag(s)"),
            k - 1,
            k,
        ));
    }
    let ve: f64 = v.iter().zip(e).map(|(a, b)| a * b).sum();
    let ee: f64 = e.iter().map(|x| x * x).sum();
    let rss = ee - ve * ve / vv;
    let df = if dfcor { n - k } else { n };
    let sigma = (rss / df as f64).sqrt();
    if sigma.is_nan() || sigma <= 0.0 || sigma.is_infinite() {
        return Err(UnitError::Skip {
            reason: "exact fit: no residual variation".to_string(),
            error: EconError::SingularMatrix {
                context: "ADF regression fits exactly (no residual variation)".to_string(),
                suggestion: "the series is exactly linear or constant over the sample".to_string(),
            },
        });
    }
    Ok(sigma)
}

/// The long-run variance of one unit's first differences, as plm's `longrunvar`: Δy demeaned
/// (intercept), detrended on (1, t) (trend) or raw (none), Bartlett kernel with `q` lags,
/// autocovariances divided by T - 1.
fn llc_long_run_variance(series: &[f64], model: PanelModel, q: usize) -> EconResult<f64> {
    let m = series.len() - 1;
    let mut dx = resource::try_with_capacity(m, "LLC long-run variance")?;
    dx.extend(series.windows(2).map(|w| w[1] - w[0]));
    let mf = m as f64;
    match model {
        PanelModel::None => {}
        PanelModel::Intercept => {
            let mean = dx.iter().sum::<f64>() / mf;
            dx.iter_mut().for_each(|x| *x -= mean);
        }
        PanelModel::Trend => {
            // Residuals of the regression on (1, 1..m), in closed form.
            let t_mean = (mf + 1.0) / 2.0;
            let x_mean = dx.iter().sum::<f64>() / mf;
            let (mut sxy, mut sxx) = (0.0, 0.0);
            for (j, x) in dx.iter().enumerate() {
                let tc = (j + 1) as f64 - t_mean;
                sxy += tc * (x - x_mean);
                sxx += tc * tc;
            }
            let slope = sxy / sxx;
            for (j, x) in dx.iter_mut().enumerate() {
                *x -= x_mean + slope * ((j + 1) as f64 - t_mean);
            }
        }
    }
    let mut lrv = dx.iter().map(|x| x * x).sum::<f64>() / mf;
    for lag in 1..=q {
        let gamma: f64 = dx[lag..]
            .iter()
            .zip(&dx[..m - lag])
            .map(|(a, b)| a * b)
            .sum();
        lrv += 2.0 * (1.0 - lag as f64 / (q + 1) as f64) * gamma / mf;
    }
    Ok(lrv)
}

/// The T rows of LLC (2002) Table 2 as `plm:::adj.levinlin` stores them.
const LLC_TS: [f64; 13] = [
    25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 250.0, 500.0,
];

/// LLC (2002) Table 2 (`plm:::adj.levinlin`): the mean adjustment mu* by T.
#[rustfmt::skip]
const LLC_MU: [[f64; 13]; 3] = [
    // none
    [0.004, 0.003, 0.002, 0.002, 0.001, 0.001, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    // intercept
    [-0.554, -0.546, -0.541, -0.537, -0.533, -0.531, -0.527, -0.524, -0.521, -0.520, -0.518, -0.509, -0.500],
    // trend
    [-0.703, -0.674, -0.653, -0.637, -0.624, -0.614, -0.598, -0.587, -0.578, -0.571, -0.566, -0.533, -0.500],
];

/// LLC (2002) Table 2 (`plm:::adj.levinlin`): the standard deviation adjustment sigma* by T.
#[rustfmt::skip]
const LLC_SIGMA: [[f64; 13]; 3] = [
    [1.049, 1.035, 1.027, 1.021, 1.017, 1.014, 1.011, 1.008, 1.007, 1.006, 1.005, 1.001, 1.000],
    [0.919, 0.889, 0.867, 0.850, 0.837, 0.826, 0.810, 0.798, 0.789, 0.782, 0.776, 0.742, 0.707],
    [1.003, 0.949, 0.906, 0.871, 0.842, 0.818, 0.780, 0.751, 0.728, 0.710, 0.695, 0.603, 0.500],
];

/// (mu*, sigma*) at T, interpolated between rows and clamped to T = 25 and 500 as
/// `plm:::adj.levinlin.value` does, with the end clamped to, if any.
fn llc_adjustment(t_len: usize, model: PanelModel) -> (f64, f64, Option<TableEdge>) {
    let row = n_det(model);
    let pos = select_t(t_len as f64, &LLC_TS);
    (
        pos.value(&LLC_MU[row], &LLC_TS),
        pos.value(&LLC_SIGMA[row], &LLC_TS),
        pos.edge,
    )
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
    if model == PanelModel::None {
        return Err(invalid(
            "the Im-Pesaran-Shin test needs model intercept or trend (its W-tbar moments are \
             tabulated only for those; plm::purtest refuses exo = 'none')"
                .to_string(),
        ));
    }
    let (used, unit_lags, fits, warnings) =
        unit_adf_fits(PanelUnitRootTest::IPS, panels, model, plan, dfcor)?;
    let t_stats: Vec<f64> = fits.iter().map(|f| f.t_stat).collect();
    let n_valid = t_stats.len();

    // Each unit's E and Var of t from IPS (2003) Table 3 at (T_i - p_i - 1, p_i), averaged.
    let (mut e_sum, mut v_sum) = (0.0, 0.0);
    let (mut below, mut above, mut many_lags) = (0usize, 0usize, 0usize);
    for (&i, &p) in used.iter().zip(&unit_lags) {
        let l = panels[i].1.len() - p - 1;
        let (e, v, edge) = ips_moments(l, p, model)?;
        e_sum += e;
        v_sum += v;
        match edge {
            Some(TableEdge::Below) => below += 1,
            Some(TableEdge::Above) => above += 1,
            None => {}
        }
        many_lags += usize::from(p > IPS_MAX_LAG);
    }
    let e_t = e_sum / n_valid as f64;
    let var_t = v_sum / n_valid as f64;
    let mut warnings = warnings;
    if many_lags > 0 {
        warnings.push(format!(
            "Im-Pesaran-Shin: {many_lags} unit(s) use more than {IPS_MAX_LAG} lags; the \
             {IPS_MAX_LAG}-lag moments of IPS (2003) Table 3 are used, as plm::purtest does"
        ));
    }
    if below > 0 {
        warnings.push(format!(
            "Im-Pesaran-Shin: {below} unit(s) have T - lags - 1 below 10 (the shortest row of \
             IPS (2003) Table 3); the T = 10 moments are used, as plm::purtest does"
        ));
    }
    if above > 0 {
        warnings.push(format!(
            "Im-Pesaran-Shin: {above} unit(s) have T - lags - 1 above 100 (the longest row of \
             IPS (2003) Table 3); the T = 100 moments are used, as plm::purtest does"
        ));
    }

    let t_bar: f64 = t_stats.iter().sum::<f64>() / n_valid as f64;
    // W-tbar: the standardised t-bar, lower tail.
    let w_stat = (n_valid as f64).sqrt() * (t_bar - e_t) / var_t.sqrt();
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

/// Which end of a table a value was clamped to (plm's `selectT` warns "the time series is
/// short" or "long").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TableEdge {
    Below,
    Above,
}

/// plm's `selectT` over the tabulated `ts` (increasing): the row of `x`, the two rows around
/// it, or the first or last row when `x` lies outside.
struct TablePos {
    lo: usize,
    hi: usize,
    x: f64,
    edge: Option<TableEdge>,
}

fn select_t(x: f64, ts: &[f64]) -> TablePos {
    let last = ts.len() - 1;
    if let Some(i) = ts.iter().position(|&t| t == x) {
        return TablePos {
            lo: i,
            hi: i,
            x,
            edge: None,
        };
    }
    if x < ts[0] {
        return TablePos {
            lo: 0,
            hi: 0,
            x: ts[0],
            edge: Some(TableEdge::Below),
        };
    }
    if x > ts[last] {
        return TablePos {
            lo: last,
            hi: last,
            x: ts[last],
            edge: Some(TableEdge::Above),
        };
    }
    let hi = ts.partition_point(|&t| t <= x);
    TablePos {
        lo: hi - 1,
        hi,
        x,
        edge: None,
    }
}

impl TablePos {
    /// The tabulated value of `row` at `x`, linearly interpolated between rows as plm does
    /// (NaN when either end is not tabulated).
    fn value(&self, row: &[f64], ts: &[f64]) -> f64 {
        if self.lo == self.hi {
            return row[self.lo];
        }
        let (t0, t1) = (ts[self.lo], ts[self.hi]);
        row[self.lo] + (self.x - t0) / (t1 - t0) * (row[self.hi] - row[self.lo])
    }
}

/// The largest lag order of IPS (2003) Table 3; larger orders use its row.
const IPS_MAX_LAG: usize = 8;

/// The T columns of IPS (2003) Table 3 (T - lags - 1 of the ADF regression).
const IPS_TS: [f64; 10] = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 100.0];

const NA: f64 = f64::NAN;

/// IPS (2003) Table 3 as `plm:::adj.ips.wtbar` stores it: mean of the ADF t by lag order
/// (rows 0..=8) and T (`IPS_TS`), individual intercepts; NaN = not tabulated.
#[rustfmt::skip]
const IPS_INTERCEPT_MEAN: [[f64; 10]; 9] = [
    [-1.504, -1.514, -1.522, -1.520, -1.526, -1.523, -1.527, -1.519, -1.524, -1.532],
    [-1.488, -1.503, -1.516, -1.514, -1.519, -1.520, -1.524, -1.519, -1.522, -1.530],
    [-1.319, -1.387, -1.428, -1.443, -1.460, -1.476, -1.493, -1.490, -1.498, -1.514],
    [-1.306, -1.366, -1.413, -1.433, -1.453, -1.471, -1.489, -1.486, -1.495, -1.512],
    [-1.171, -1.260, -1.329, -1.363, -1.394, -1.428, -1.454, -1.458, -1.470, -1.495],
    [NA, NA, -1.313, -1.351, -1.384, -1.421, -1.451, -1.454, -1.467, -1.494],
    [NA, NA, NA, -1.289, -1.331, -1.380, -1.418, -1.427, -1.444, -1.476],
    [NA, NA, NA, -1.273, -1.319, -1.371, -1.411, -1.423, -1.441, -1.474],
    [NA, NA, NA, -1.212, -1.266, -1.329, -1.377, -1.393, -1.415, -1.456],
];

/// As [`IPS_INTERCEPT_MEAN`], the variance of t.
#[rustfmt::skip]
const IPS_INTERCEPT_VAR: [[f64; 10]; 9] = [
    [1.069, 0.923, 0.851, 0.809, 0.789, 0.770, 0.760, 0.749, 0.736, 0.735],
    [1.255, 1.011, 0.915, 0.861, 0.831, 0.803, 0.781, 0.770, 0.753, 0.745],
    [1.421, 1.078, 0.969, 0.905, 0.865, 0.830, 0.798, 0.789, 0.766, 0.754],
    [1.759, 1.181, 1.037, 0.952, 0.907, 0.858, 0.819, 0.802, 0.782, 0.761],
    [2.080, 1.279, 1.097, 1.005, 0.946, 0.886, 0.842, 0.819, 0.801, 0.771],
    [NA, NA, 1.171, 1.055, 0.980, 0.912, 0.863, 0.839, 0.814, 0.781],
    [NA, NA, NA, 1.114, 1.023, 0.942, 0.886, 0.858, 0.834, 0.795],
    [NA, NA, NA, 1.164, 1.062, 0.968, 0.910, 0.875, 0.851, 0.806],
    [NA, NA, NA, 1.217, 1.105, 0.996, 0.929, 0.896, 0.871, 0.818],
];

/// As [`IPS_INTERCEPT_MEAN`], individual intercepts and trends.
#[rustfmt::skip]
const IPS_TREND_MEAN: [[f64; 10]; 9] = [
    [-2.166, -2.167, -2.168, -2.167, -2.172, -2.173, -2.176, -2.174, -2.174, -2.177],
    [-2.173, -2.169, -2.172, -2.172, -2.173, -2.177, -2.180, -2.178, -2.176, -2.179],
    [-1.914, -1.999, -2.047, -2.074, -2.095, -2.120, -2.137, -2.143, -2.146, -2.158],
    [-1.922, -1.977, -2.032, -2.065, -2.091, -2.117, -2.137, -2.142, -2.146, -2.158],
    [-1.750, -1.823, -1.911, -1.968, -2.009, -2.057, -2.091, -2.103, -2.114, -2.135],
    [NA, NA, -1.888, -1.955, -1.998, -2.051, -2.087, -2.101, -2.111, -2.135],
    [NA, NA, NA, -1.868, -1.923, -1.995, -2.042, -2.065, -2.081, -2.113],
    [NA, NA, NA, -1.851, -1.912, -1.986, -2.036, -2.063, -2.079, -2.112],
    [NA, NA, NA, -1.761, -1.835, -1.925, -1.987, -2.024, -2.046, -2.088],
];

/// As [`IPS_TREND_MEAN`], the variance of t.
#[rustfmt::skip]
const IPS_TREND_VAR: [[f64; 10]; 9] = [
    [1.132, 0.869, 0.763, 0.713, 0.690, 0.655, 0.633, 0.621, 0.610, 0.597],
    [1.453, 0.975, 0.845, 0.769, 0.734, 0.687, 0.654, 0.641, 0.627, 0.605],
    [1.627, 1.036, 0.882, 0.796, 0.756, 0.702, 0.661, 0.653, 0.634, 0.613],
    [2.482, 1.214, 0.983, 0.861, 0.808, 0.735, 0.688, 0.674, 0.650, 0.625],
    [3.947, 1.332, 1.052, 0.913, 0.845, 0.759, 0.705, 0.685, 0.662, 0.629],
    [NA, NA, 1.165, 0.991, 0.899, 0.792, 0.730, 0.705, 0.673, 0.638],
    [NA, NA, NA, 1.055, 0.945, 0.828, 0.753, 0.725, 0.689, 0.650],
    [NA, NA, NA, 1.145, 1.009, 0.872, 0.786, 0.747, 0.713, 0.661],
    [NA, NA, NA, 1.208, 1.063, 0.902, 0.808, 0.766, 0.728, 0.670],
];

/// The mean and variance of one unit's ADF t under the null, from IPS (2003) Table 3 at
/// `l = T - lags - 1` and the lag order (8 for more), as `plm:::adj.ips.wtbar.value`; with
/// the table end it was clamped to, if any. Untabulated (lags, T) is an error (plm: NA).
fn ips_moments(
    l: usize,
    lags: usize,
    model: PanelModel,
) -> EconResult<(f64, f64, Option<TableEdge>)> {
    let (mean, var) = match model {
        PanelModel::Intercept => (&IPS_INTERCEPT_MEAN, &IPS_INTERCEPT_VAR),
        PanelModel::Trend => (&IPS_TREND_MEAN, &IPS_TREND_VAR),
        PanelModel::None => {
            return Err(invalid(
                "the Im-Pesaran-Shin moments are tabulated only for model intercept or trend"
                    .to_string(),
            ));
        }
    };
    let row = lags.min(IPS_MAX_LAG);
    let pos = select_t(l as f64, &IPS_TS);
    let e = pos.value(&mean[row], &IPS_TS);
    let v = pos.value(&var[row], &IPS_TS);
    if e.is_nan() || v.is_nan() {
        return Err(invalid(format!(
            "Im-Pesaran-Shin: the W-tbar moments are not tabulated for {row} lags at T - lags \
             - 1 = {l} (IPS 2003 Table 3 has 5 lags from 20 and 6 to 8 lags from 25 periods); \
             use fewer lags or a longer panel"
        )));
    }
    Ok((e, v, pos.edge))
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

/// Hadri (2000) panel stationarity test, as `plm::purtest(test = "hadri")`.
///
/// Tests H0: All panels are stationary
/// vs H1: Some panels have unit roots
///
/// Each unit's KPSS statistic `sum (cumsum e)^2 / (T^2 sigma^2)` of its demeaned or detrended
/// series, averaged and standardised with the moments of its limit. Needs a deterministic
/// term and a balanced panel, as plm.
fn run_hadri_test(
    panels: &[(String, Vec<f64>)],
    model: PanelModel,
    variance: HadriVariance,
    dfcor: bool,
) -> EconResult<PanelUnitRootResult> {
    if model == PanelModel::None {
        return Err(invalid(
            "Hadri's test needs model intercept or trend (plm::purtest refuses exo = 'none'; \
             without deterministic terms the KPSS limit has mean 1/2, not 1/6)"
                .to_string(),
        ));
    }
    let lens = panels.iter().map(|(_, s)| s.len());
    let (t_min, t_max) = (lens.clone().min().unwrap_or(0), lens.max().unwrap_or(0));
    if t_min != t_max {
        return Err(invalid(format!(
            "Hadri's test needs a balanced panel (every unit with the same number of \
             observations, as plm::purtest); the units have {t_min} to {t_max} observations"
        )));
    }
    let t_len = t_max;
    let tf = t_len as f64;
    let nd = n_det(model);
    let short_df = if dfcor { (t_len - nd) as f64 } else { tf };
    let long_lags = match variance {
        HadriVariance::LongRun { lags } => {
            let l = lags.unwrap_or((4.0 * (tf / 100.0).powf(0.25)).trunc() as usize);
            if l >= t_len {
                return Err(invalid(format!(
                    "the long-run variance's truncation lag ({l}) must be below the number of \
                     observations per unit ({t_len})"
                )));
            }
            Some(l)
        }
        _ => None,
    };

    // Per unit: S_i = sum (cumsum e)^2 / T^2, e'e, and the long-run variance if asked.
    let n = panels.len();
    let mut partial = Vec::with_capacity(n);
    let mut rss = Vec::with_capacity(n);
    let mut lrv = Vec::with_capacity(if long_lags.is_some() { n } else { 0 });
    for (_, series) in panels {
        resource::checkpoint()?;
        let e = kpss_residuals(series, model)?;
        let mut cum = 0.0;
        let s2: f64 = e
            .iter()
            .map(|x| {
                cum += x;
                cum * cum
            })
            .sum();
        partial.push(s2 / (tf * tf));
        rss.push(e.iter().map(|x| x * x).sum::<f64>());
        if let Some(l) = long_lags {
            lrv.push(bartlett_long_run_variance(&e, l));
        }
    }
    let nf = n as f64;
    let (unit_lm, lm_bar, label) = match variance {
        HadriVariance::Heteroskedastic => {
            let lm: Vec<f64> = partial
                .iter()
                .zip(&rss)
                .map(|(s, r)| s / (r / short_df))
                .collect();
            let bar = lm.iter().sum::<f64>() / nf;
            (
                lm,
                bar,
                "heteroskedasticity-consistent, per unit (plm Hcons = TRUE)".to_string(),
            )
        }
        HadriVariance::Homoskedastic => {
            let sigma2 = rss.iter().sum::<f64>() / (nf * short_df);
            let lm: Vec<f64> = partial
                .iter()
                .zip(&rss)
                .map(|(s, r)| s / (r / short_df))
                .collect();
            let bar = partial.iter().sum::<f64>() / nf / sigma2;
            (
                lm,
                bar,
                "homoskedastic, pooled (plm Hcons = FALSE)".to_string(),
            )
        }
        HadriVariance::LongRun { .. } => {
            let l = long_lags.unwrap_or(0);
            let lm: Vec<f64> = partial.iter().zip(&lrv).map(|(s, v)| s / v).collect();
            let bar = lm.iter().sum::<f64>() / nf;
            (
                lm,
                bar,
                format!("long-run, Bartlett kernel, {l} lag(s) (tseries::kpss.test)"),
            )
        }
    };

    let (mu, sigma2) = get_hadri_moments(model);
    let z_stat = nf.sqrt() * (lm_bar - mu) / sigma2.sqrt();
    // One-sided, upper tail.
    let p_value = Normal::standard().sf(z_stat);

    let used: Vec<usize> = (0..n).collect();
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
    r.unit_statistics = Some(unit_lm);
    r.variance = label;
    Ok(r)
}

/// A unit's series demeaned (intercept) or detrended on (1, 1..T) (trend), closed form.
fn kpss_residuals(series: &[f64], model: PanelModel) -> EconResult<Vec<f64>> {
    let tf = series.len() as f64;
    let mean = series.iter().sum::<f64>() / tf;
    let mut e = resource::try_with_capacity(series.len(), "Hadri residuals")?;
    if model != PanelModel::Trend {
        e.extend(series.iter().map(|x| x - mean));
        return Ok(e);
    }
    let t_mean = (tf + 1.0) / 2.0;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (j, x) in series.iter().enumerate() {
        let tc = (j + 1) as f64 - t_mean;
        sxy += tc * (x - mean);
        sxx += tc * tc;
    }
    let slope = sxy / sxx;
    e.extend(
        series
            .iter()
            .enumerate()
            .map(|(j, x)| x - mean - slope * ((j + 1) as f64 - t_mean)),
    );
    Ok(e)
}

/// The Bartlett long-run variance of residuals `e` with `l` lags, as `tseries::kpss.test`:
/// `(1/T) e'e + (2/T) sum_j (1 - j/(l+1)) sum_t e_t e_{t-j}`.
fn bartlett_long_run_variance(e: &[f64], l: usize) -> f64 {
    let t = e.len();
    let tf = t as f64;
    let mut v = e.iter().map(|x| x * x).sum::<f64>() / tf;
    for j in 1..=l {
        let gamma: f64 = e[j..].iter().zip(&e[..t - j]).map(|(a, b)| a * b).sum();
        v += 2.0 / tf * (1.0 - j as f64 / (l + 1) as f64) * gamma;
    }
    v
}

/// Hadri (2000) moments of the KPSS limit under H0 (model none is refused before).
fn get_hadri_moments(model: PanelModel) -> (f64, f64) {
    match model {
        PanelModel::Trend => (1.0 / 15.0, 11.0 / 6300.0),
        _ => (1.0 / 6.0, 1.0 / 45.0),
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

        // The LLC statistic is pinned to plm in test_validate_purtest_stat_llc.
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
        assert_eq!(
            result.variance,
            "heteroskedasticity-consistent, per unit (plm Hcons = TRUE)"
        );
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
            variance: String::new(),
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
                )
                .scalar(
                    &format!("ips_{rule}{sfx}_stat"),
                    r.statistic,
                    Tol::stat(480),
                )
                .scalar(
                    &format!("ips_{rule}{sfx}_p"),
                    r.p_value,
                    Tol::p_value(480),
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
                )
                .scalar(
                    &format!("levinlin_{rule}{sfx}_stat"),
                    r.statistic,
                    Tol::stat(480),
                )
                .scalar(
                    &format!("levinlin_{rule}{sfx}_p"),
                    r.p_value,
                    Tol::p_value(480),
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
            "the Hadri test uses no lagged differences; lags, lag_selection and pmax do not \
             apply (the long-run variance's truncation lag is set with hadri_variance = long_run)"
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

    /// IPS W-tbar with IPS (2003) Table 3 moments: plm::purtest(test = "ips", lags = 1 | 3)
    /// on T = 50 and lags = 1 | 2 | 4 on T = 15 (rows 10 and 15, interpolation between).
    #[test]
    fn test_validate_purtest_stat_ips() {
        // The averaged table moments plm reports (rowMeans(adjval)).
        let moments = |ds: &Dataset, r: &PanelUnitRootResult, model| {
            let lens: Vec<usize> = (1..=r.n_panels)
                .map(|i| {
                    let ids = ds.df().column("id").unwrap().i64().unwrap();
                    ids.iter().filter(|&v| v == Some(i as i64)).count()
                })
                .collect();
            let (mut e, mut v) = (0.0, 0.0);
            for (len, &p) in lens.iter().zip(&r.unit_lags) {
                let (a, b, _) = ips_moments(len - p - 1, p, model).unwrap();
                e += a;
                v += b;
            }
            let n = r.n_panels as f64;
            vec![e / n, v / n]
        };
        let tight = Tol::new(1e-14, 1e-14);
        let (case, ds) = crate::test_support::load_case("purtest_stat_ar05");
        let mut c = Check::new(&case);
        for (model, exo) in &EXOS[..2] {
            for col in ["y", "y_rw"] {
                for dfcor in [false, true] {
                    for lags in [1, 3] {
                        let key = format!("ips_{}_lags{lags}", stat_tag(col, exo, dfcor));
                        let r =
                            stat_run(&ds, col, PanelUnitRootTest::IPS, *model, Some(lags), dfcor);
                        c.scalar(&format!("{key}_stat"), r.statistic, Tol::stat(500))
                            .scalar(&format!("{key}_p"), r.p_value, Tol::p_value(500))
                            .vector(&format!("{key}_moments"), &moments(&ds, &r, *model), tight);
                    }
                }
            }
        }
        c.finish();
        let (case, ds) = crate::test_support::load_case("purtest_stat_short");
        let mut c = Check::new(&case);
        for (model, exo) in &EXOS[..2] {
            for lags in [1, 2, 4] {
                let key = format!("ips_{exo}_lags{lags}");
                let r = stat_run(&ds, "y", PanelUnitRootTest::IPS, *model, Some(lags), true);
                c.scalar(&format!("{key}_stat"), r.statistic, Tol::stat(120))
                    .scalar(&format!("{key}_p"), r.p_value, Tol::p_value(120))
                    .vector(&format!("{key}_moments"), &moments(&ds, &r, *model), tight);
            }
        }
        c.finish();
    }

    /// plm::purtest refuses exo = "none" for IPS.
    #[test]
    fn test_ips_refuses_model_none() {
        let ds = noisy_panel(&[30; 6], 7);
        let e = run_panel_unit_root(
            &ds,
            "y",
            "unit",
            "time",
            cfg(PanelUnitRootTest::IPS, PanelModel::None, Some(1)),
        )
        .unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message }
                if message.contains("Im-Pesaran-Shin test needs model intercept or trend")),
            "{e:?}"
        );
    }

    /// T = 18 with 5 lags: T - lags - 1 = 12 lies between rows 10 and 15, where 5 lags are not
    /// tabulated (plm returns NA).
    #[test]
    fn test_ips_untabulated_moments_is_an_error() {
        let ds = noisy_panel(&[18; 6], 8);
        let e = run_panel_unit_root(
            &ds,
            "y",
            "unit",
            "time",
            cfg(PanelUnitRootTest::IPS, PanelModel::Intercept, Some(5)),
        )
        .unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message }
                if message.contains("not tabulated for 5 lags at T - lags - 1 = 12")),
            "{e:?}"
        );
    }

    /// Clamped table rows and lag orders above 8 are used as plm does, with a warning.
    #[test]
    fn test_ips_table_clamp_warnings() {
        let short = run_panel_unit_root(
            &noisy_panel(&[9; 6], 9),
            "y",
            "unit",
            "time",
            cfg(PanelUnitRootTest::IPS, PanelModel::Intercept, Some(1)),
        )
        .unwrap();
        assert!(
            short
                .warnings
                .iter()
                .any(|w| w.starts_with("Im-Pesaran-Shin: 6 unit(s) have T - lags - 1 below 10")),
            "{:?}",
            short.warnings
        );
        let (e, v, edge) = ips_moments(7, 1, PanelModel::Intercept).unwrap();
        assert_eq!((e, v, edge), (-1.488, 1.255, Some(TableEdge::Below)));
        let many = run_panel_unit_root(
            &noisy_panel(&[60; 6], 10),
            "y",
            "unit",
            "time",
            cfg(PanelUnitRootTest::IPS, PanelModel::Intercept, Some(9)),
        )
        .unwrap();
        assert!(
            many.warnings
                .iter()
                .any(|w| w.starts_with("Im-Pesaran-Shin: 6 unit(s) use more than 8 lags")),
            "{:?}",
            many.warnings
        );
        let (_, _, edge) = ips_moments(150, 1, PanelModel::Trend).unwrap();
        assert_eq!(edge, Some(TableEdge::Above));
    }

    /// The LLC z of plm::purtest(test = "levinlin", lags = 1): every model and dfcor, on the
    /// stationary and the random-walk panels; the z of 1000 y (scale invariance); T = 15,
    /// below the table's first row.
    #[test]
    fn test_validate_purtest_stat_llc() {
        let (case, ds) = crate::test_support::load_case("purtest_stat_ar05");
        let mut c = Check::new(&case);
        for (model, exo) in EXOS {
            for col in ["y", "y_rw"] {
                for dfcor in [false, true] {
                    let key = format!("llc_{}", stat_tag(col, exo, dfcor));
                    let r = stat_run(&ds, col, PanelUnitRootTest::LLC, model, Some(1), dfcor);
                    c.scalar(&format!("{key}_stat"), r.statistic, Tol::stat(500))
                        .scalar(&format!("{key}_p"), r.p_value, Tol::p_value(500));
                    assert!(
                        !r.warnings.iter().any(|w| w.starts_with("Levin-Lin-Chu")),
                        "{:?}",
                        r.warnings
                    );
                }
            }
        }
        let y: Vec<f64> = ds
            .df()
            .column("y")
            .unwrap()
            .f64()
            .unwrap()
            .into_iter()
            .map(|v| 1000.0 * v.unwrap())
            .collect();
        let scaled = with_column(&ds, Column::new("y1000".into(), y));
        let r = stat_run(
            &scaled,
            "y1000",
            PanelUnitRootTest::LLC,
            PanelModel::Intercept,
            Some(1),
            false,
        );
        c.scalar("llc_intercept_x1000_stat", r.statistic, Tol::stat(500));
        c.finish();

        let (case, ds) = crate::test_support::load_case("purtest_stat_short");
        let mut c = Check::new(&case);
        for (model, exo) in &EXOS[..2] {
            let r = stat_run(&ds, "y", PanelUnitRootTest::LLC, *model, Some(1), true);
            c.scalar(&format!("llc_{exo}_stat"), r.statistic, Tol::stat(120))
                .scalar(&format!("llc_{exo}_p"), r.p_value, Tol::p_value(120));
            assert!(
                r.warnings
                    .iter()
                    .any(|w| w
                        .starts_with("Levin-Lin-Chu: T = 15 is below the shortest row (T = 25)")),
                "{:?}",
                r.warnings
            );
        }
        c.finish();
    }

    /// plm::purtest refuses LLC on an unbalanced panel.
    #[test]
    fn test_llc_refuses_unbalanced_panels() {
        let mut lengths = vec![40; 9];
        lengths.push(38);
        let e = run_panel_unit_root(
            &noisy_panel(&lengths, 11),
            "y",
            "unit",
            "time",
            cfg(PanelUnitRootTest::LLC, PanelModel::Intercept, Some(1)),
        )
        .unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message }
                if message.contains("needs a balanced panel")
                    && message.contains("38 to 40 observations")
                    && message.contains("Im-Pesaran-Shin and Maddala-Wu")),
            "{e:?}"
        );
    }

    /// T = 7: round(3.21 * 7^(1/3)) = 6 exceeds T - 2 = 5 autocovariance lags.
    #[test]
    fn test_llc_long_run_lag_capped_on_very_short_series() {
        let r = run_panel_unit_root(
            &noisy_panel(&[7; 8], 12),
            "y",
            "unit",
            "time",
            cfg(PanelUnitRootTest::LLC, PanelModel::Intercept, Some(0)),
        )
        .unwrap();
        assert!(r.statistic.is_finite());
        assert!(
            r.warnings.iter().any(|w| w.starts_with(
                "Levin-Lin-Chu: the long-run variance's truncation lag is capped at T - 2 = 5 \
                 (round(3.21 T^(1/3)) = 6"
            )),
            "{:?}",
            r.warnings
        );
    }

    /// A lagged level that the other regressors explain is a rank-deficient ADF regression.
    #[test]
    fn test_llc_unit_sigma_skips_explained_level() {
        let series: Vec<f64> = (0..20).map(|t| 1.0 + 0.1 * f64::from(t % 3)).collect();
        let e: Vec<f64> = (0..19).map(|t| 0.01 * f64::from(t % 5) - 0.02).collect();
        let v = vec![0.0; 19];
        assert!(matches!(
            llc_unit_sigma(&series, &e, &v, PanelModel::Intercept, 0, true),
            Err(UnitError::Skip { reason, .. }) if reason.starts_with("rank deficient")
        ));
        let v: Vec<f64> = (0..19).map(|t| 0.3 * f64::from(t % 4) - 0.45).collect();
        assert!(llc_unit_sigma(&series, &e, &v, PanelModel::Intercept, 0, true).is_ok());
    }

    /// Hadri's z of plm::purtest(test = "hadri", Hcons, dfcor) on the stationary and the
    /// random-walk panels, with the per-unit LM.
    #[test]
    fn test_validate_purtest_stat_hadri() {
        let (case, ds) = crate::test_support::load_case("purtest_stat_ar05");
        let mut c = Check::new(&case);
        for (model, exo) in &EXOS[..2] {
            for col in ["y", "y_rw"] {
                for dfcor in [false, true] {
                    for (variance, v) in [
                        (HadriVariance::Heteroskedastic, "hcons"),
                        (HadriVariance::Homoskedastic, "homo"),
                    ] {
                        let key = format!("hadri_{}_{v}", stat_tag(col, exo, dfcor));
                        let mut cf = cfg(PanelUnitRootTest::Hadri, *model, None);
                        cf.dfcor = dfcor;
                        cf.hadri_variance = variance;
                        let r = run_panel_unit_root(&ds, col, "id", "t", cf).unwrap();
                        c.scalar(&format!("{key}_stat"), r.statistic, Tol::stat(500))
                            .scalar(&format!("{key}_p"), r.p_value, Tol::p_value(500));
                        if variance == HadriVariance::Heteroskedastic {
                            c.vector(
                                &format!("{key}_lm"),
                                r.unit_statistics.as_deref().unwrap(),
                                Tol::stat(500),
                            );
                        }
                    }
                }
            }
        }
        c.finish();
    }

    /// Hadri's long-run option: the per-unit statistics are tseries::kpss.test's, with its
    /// short truncation lag (3 at T = 50) or a given one (10, its long rule at T = 50).
    #[test]
    fn test_validate_purtest_stat_hadri_long_run() {
        let (case, ds) = crate::test_support::load_case("purtest_stat_ar05");
        let mut c = Check::new(&case);
        for (model, null) in [
            (PanelModel::Intercept, "level"),
            (PanelModel::Trend, "trend"),
        ] {
            for col in ["y", "y_rw"] {
                let rw = if col == "y" { "" } else { "rw_" };
                for (lags, len) in [(None, "short"), (Some(10), "long")] {
                    let key = format!("kpss_{rw}{null}_{len}");
                    let mut cf = cfg(PanelUnitRootTest::Hadri, model, None);
                    cf.hadri_variance = HadriVariance::LongRun { lags };
                    let r = run_panel_unit_root(&ds, col, "id", "t", cf).unwrap();
                    c.vector(&key, r.unit_statistics.as_deref().unwrap(), Tol::stat(500));
                    let lag = case.scalar(&format!("{key}_lag"));
                    assert_eq!(
                        r.variance,
                        format!("long-run, Bartlett kernel, {lag} lag(s) (tseries::kpss.test)")
                    );
                }
            }
        }
        c.finish();
    }

    #[test]
    fn test_hadri_refuses_model_none_unbalanced_and_lags() {
        let message = |ds: &Dataset, f: &dyn Fn(&mut PanelUnitRootConfig)| {
            let mut c = cfg(PanelUnitRootTest::Hadri, PanelModel::Intercept, None);
            f(&mut c);
            match run_panel_unit_root(ds, "y", "unit", "time", c) {
                Err(EconError::InvalidSpecification { message }) => message,
                other => panic!("{other:?}"),
            }
        };
        let ds = noisy_panel(&[30; 6], 13);
        assert!(
            message(&ds, &|c| c.model = PanelModel::None)
                .starts_with("Hadri's test needs model intercept or trend")
        );
        assert!(
            message(&ds, &|c| c.lags = Some(2))
                .starts_with("the Hadri test uses no lagged differences; lags,")
        );
        assert!(
            message(&ds, &|c| {
                c.hadri_variance = HadriVariance::LongRun { lags: Some(30) }
            })
            .contains("truncation lag (30) must be below")
        );
        let unbalanced = noisy_panel(&[30, 30, 30, 28], 14);
        assert!(
            message(&unbalanced, &|_| {}).contains(
                "needs a balanced panel (every unit with the same number of observations"
            )
        );
    }

    #[test]
    fn test_hadri_variance_option_parses_and_conflicts() {
        HADRI_VARIANCES.validate().unwrap();
        for (name, v) in [
            ("heteroskedastic", HadriVariance::Heteroskedastic),
            ("Hcons", HadriVariance::Heteroskedastic),
            ("het", HadriVariance::Heteroskedastic),
            ("homo", HadriVariance::Homoskedastic),
            ("iid", HadriVariance::Homoskedastic),
            ("long-run", HadriVariance::LongRun { lags: None }),
            ("bartlett", HadriVariance::LongRun { lags: None }),
        ] {
            assert_eq!(HadriVariance::from_option(name, None).unwrap(), v, "{name}");
        }
        assert_eq!(
            HadriVariance::from_option("lrv", Some(4)).unwrap(),
            HadriVariance::LongRun { lags: Some(4) }
        );
        let e = HadriVariance::from_option("newey", None).unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message }
                if message.contains("heteroskedastic, homoskedastic, long_run")),
            "{e:?}"
        );
        assert!(HadriVariance::from_option("homo", Some(2)).is_err());
        assert_eq!(
            HadriVariance::names_help(),
            "heteroskedastic | homoskedastic | long_run"
        );
        let mut c = cfg(PanelUnitRootTest::LLC, PanelModel::Intercept, Some(1));
        c.hadri_variance = HadriVariance::Homoskedastic;
        let e =
            run_panel_unit_root(&noisy_panel(&[30; 6], 15), "y", "unit", "time", c).unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message }
                if message == "hadri_variance applies only to the Hadri test, not Levin-Lin-Chu"),
            "{e:?}"
        );
    }

    #[test]
    fn test_hadri_display_names_variance() {
        let ds = noisy_panel(&[30; 6], 16);
        let mut c = cfg(PanelUnitRootTest::Hadri, PanelModel::Trend, None);
        let text = run_panel_unit_root(&ds, "y", "unit", "time", c.clone())
            .unwrap()
            .to_string();
        assert!(
            text.contains(
                "Variance:   heteroskedasticity-consistent, per unit (plm Hcons = TRUE)\n"
            ),
            "{text}"
        );
        assert!(
            text.contains("Residual variance: rss/(T - k) (dfcor = true)"),
            "{text}"
        );
        assert!(!text.contains("ADF residual variance"), "{text}");
        c.hadri_variance = HadriVariance::LongRun { lags: None };
        let text = run_panel_unit_root(&ds, "y", "unit", "time", c)
            .unwrap()
            .to_string();
        assert!(
            text.contains("Variance:   long-run, Bartlett kernel, 2 lag(s) (tseries::kpss.test)"),
            "{text}"
        );
        assert!(!text.contains("Residual variance"), "{text}");
    }
}
