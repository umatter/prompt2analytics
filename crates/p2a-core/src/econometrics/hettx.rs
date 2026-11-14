//! Treatment effect heterogeneity testing (hettx).
//!
//! Randomization tests for treatment effect variation in a completely randomized experiment,
//! after Ding, Feller & Miratrix (2016, 2019). The procedures are written out from the papers;
//! hettx itself is not installed on the reference machine, so the R pins are a base-R
//! transcription of the same procedures (`validation/econometrics/hettx.md`).
//!
//! # Test of a constant effect (DFM 2016)
//!
//! H0: tau_i = tau for every unit, with tau unknown. For each tau on an evenly spaced grid over
//! the 1 - gamma Neyman confidence interval of the average effect, the missing potential
//! outcomes are imputed under H0 (Y(0) = Y - tau Z, Y(1) = Y(0) + tau). Every drawn assignment
//! Zb (complete randomization: a permutation of Z) gives Yb = Y(0) + tau Zb and a statistic
//! t(Yb, Zb), and p(tau) = (1 + #{t(Yb, Zb) >= t(Y, Z)}) / (B + 1), with the same B
//! assignments for every tau. The test's p-value is min(1, max_tau p(tau) + gamma)
//! (Berger and Boos); with `plugin` it is p(ATE), without gamma.
//!
//! Statistics:
//! - **SKS** (default, as hettx): the shifted Kolmogorov-Smirnov statistic, the largest
//!   distance between the empirical distributions of the two arms' outcomes, each centred at
//!   its arm mean.
//! - **Spread of imputed unit effects** (variance, range, IQR, mean absolute deviation): the
//!   unit effects are imputed by matching or by regression on the covariates. Both imputations
//!   are linear in the outcome for a fixed assignment, so each draw imputes once:
//!   tau_hat(tau) = imp(Y) + tau imp(Zb - Z).
//!
//! # Decomposition (DFM 2019, DFM 2016 section 6)
//!
//! - **Systematic variation**: `y ~ 1 + z + X + z:X` by least squares; the treatment-covariate
//!   interactions are tested with a Wald chi-squared test on their HC2 covariance, and the
//!   systematic variance is beta' S_xx beta.
//! - **Idiosyncratic variation**: H0: tau_i = tau(X_i) = b_z + X_i' beta, with the fitted
//!   (plug-in) coefficients. The statistic is the Kolmogorov-Smirnov distance between the two
//!   arms' residuals from their own regressions on X (SKS with covariate adjustment), and its
//!   randomization p-value uses the same assignments as the test of a constant effect.
//!
//! # References
//!
//! - Ding, P., Feller, A., & Miratrix, L. (2016). Randomization inference for
//!   treatment effect variation. *Journal of the Royal Statistical Society:
//!   Series B*, 78(3), 655-671. https://doi.org/10.1111/rssb.12124
//!
//! - Ding, P., Feller, A., & Miratrix, L. (2019). Decomposing treatment effect
//!   variation. *Journal of the American Statistical Association*, 114(525),
//!   304-317. https://doi.org/10.1080/01621459.2017.1407322
//!
//! - Berger, R. L., & Boos, D. D. (1994). P values maximized over a confidence set for the
//!   nuisance parameter. *JASA*, 89(427), 1012-1016.
//!
//! R equivalent: `hettx::detect_idiosyncratic()`, `hettx::detect_systematic()`

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::SeedableRng;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::xtx;
use crate::linalg::solve::{SingularPolicy, least_squares, sym_inverse};
use crate::options::Choices;
use crate::regression::ols::normal_equations_inverse;
use crate::resource;
use crate::stats::robust::{QuantileType, quantile_sorted};
use crate::traits::estimator::{SignificanceLevel, chi_squared_p_value, normal_quantile};

// ===============================================================================
// Configuration Types
// ===============================================================================

/// Test statistic of the randomization test of a constant effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum HetTestStat {
    /// Shifted Kolmogorov-Smirnov statistic (default, as hettx): the largest distance between
    /// the empirical distributions of the arm-centred outcomes. Needs no imputation.
    #[default]
    SKS,
    /// Variance of the imputed unit effects.
    Variance,
    /// Range of the imputed unit effects. Sensitive to extreme values.
    Range,
    /// Interquartile range (type 7) of the imputed unit effects. Robust to outliers.
    IQR,
    /// Mean absolute deviation of the imputed unit effects from their mean.
    MeanAbsDeviation,
}

// The accepted spellings; the first of each entry is canonical. `sks` is listed last.
crate::option_enum!(
    HetTestStat,
    Choices {
        option: "test_statistic",
        values: &[
            (&["variance", "var"], HetTestStat::Variance),
            (&["range"], HetTestStat::Range),
            (&["iqr", "interquartile_range"], HetTestStat::IQR),
            (
                &["mad", "mean_abs_deviation", "mean_absolute_deviation"],
                HetTestStat::MeanAbsDeviation,
            ),
            (
                &["sks", "shifted_ks", "shifted_kolmogorov_smirnov"],
                HetTestStat::SKS
            ),
        ],
    },
    no_display
);

impl fmt::Display for HetTestStat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HetTestStat::SKS => write!(f, "Shifted Kolmogorov-Smirnov (SKS)"),
            HetTestStat::Variance => write!(f, "Variance of imputed effects"),
            HetTestStat::Range => write!(f, "Range of imputed effects"),
            HetTestStat::IQR => write!(f, "Interquartile Range (IQR) of imputed effects"),
            HetTestStat::MeanAbsDeviation => {
                write!(f, "Mean Absolute Deviation of imputed effects")
            }
        }
    }
}

/// Method for imputing the unit effects (the spread statistics and the effect summary).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EffectEstimationMethod {
    /// Matching-based imputation (default).
    /// For each unit, imputes the missing potential outcome using
    /// the average outcome of matched units in the opposite treatment group.
    #[default]
    Matching,
    /// Regression-based imputation.
    /// Fits outcome models for treated and control groups separately,
    /// then predicts counterfactual outcomes.
    Regression,
    /// The difference in means for every unit (no strata are formed), so every imputed
    /// effect is the same. Refused with a spread statistic, whose value would be 0 for
    /// every assignment; it runs with [`HetTestStat::SKS`], which imputes nothing.
    Stratified,
}

// The accepted spellings (hettx's EffectEstimationMethod); the first of each entry is canonical.
crate::option_enum!(
    EffectEstimationMethod,
    Choices {
        option: "effect_method",
        values: &[
            (&["matching", "match"], EffectEstimationMethod::Matching),
            (&["regression", "reg"], EffectEstimationMethod::Regression),
            (&["stratified", "strat"], EffectEstimationMethod::Stratified),
        ],
    },
    no_display
);

impl fmt::Display for EffectEstimationMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EffectEstimationMethod::Matching => write!(f, "Nearest Neighbor Matching"),
            EffectEstimationMethod::Regression => write!(f, "Regression Imputation"),
            EffectEstimationMethod::Stratified => {
                write!(f, "Difference in Means (no strata)")
            }
        }
    }
}

/// Configuration for treatment effect heterogeneity test.
#[derive(Debug, Clone)]
pub struct HetTxConfig {
    /// Number of drawn assignments B (default: 1000). Must be at least 1.
    pub n_permutations: usize,
    /// Test statistic (default: [`HetTestStat::SKS`], as hettx).
    pub test_statistic: HetTestStat,
    /// Whether to decompose heterogeneity into systematic and idiosyncratic parts (default:
    /// true; needs covariates).
    pub decompose: bool,
    /// Method for imputing unit effects (default: Matching).
    pub effect_method: EffectEstimationMethod,
    /// Number of nearest neighbors for matching (default: 3).
    /// Must be at least 1 with [`EffectEstimationMethod::Matching`].
    pub n_neighbors: usize,
    /// Random seed of the drawn assignments (`None`: [`HETTX_DEFAULT_SEED`]). Draw `b` uses
    /// its own stream of this seed.
    pub seed: Option<u64>,
    /// Whether to compute covariate importance in the decomposition (default: true).
    pub compute_importance: bool,
    /// gamma of the confidence interval the test maximizes over (default: 1e-4, as hettx);
    /// 0 < gamma < 1. The p-value adds gamma.
    pub gamma: f64,
    /// Number of evenly spaced tau values over the 1 - gamma interval (default: 151, as
    /// hettx); at least 2 unless `plugin`.
    pub grid_size: usize,
    /// Test at tau = ATE only, without gamma (default: false). Not size-controlled in small
    /// samples (DFM 2016).
    pub plugin: bool,
}

/// The seed of the permutations when [`HetTxConfig::seed`] is `None`, so every run is
/// reproducible (the MCP handler's default seed is the same value).
pub const HETTX_DEFAULT_SEED: u64 = 42;

impl Default for HetTxConfig {
    fn default() -> Self {
        Self {
            n_permutations: 1000,
            test_statistic: HetTestStat::SKS,
            decompose: true,
            effect_method: EffectEstimationMethod::Matching,
            n_neighbors: 3,
            seed: None,
            compute_importance: true,
            gamma: 1e-4,
            grid_size: 151,
            plugin: false,
        }
    }
}

// ===============================================================================
// Result Types
// ===============================================================================

/// Decomposition of treatment effect heterogeneity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HetDecomposition {
    /// Sample variance of the imputed unit effects (descriptive).
    pub total_variance: f64,
    /// Systematic variance beta' S_xx beta: beta the treatment-covariate interaction
    /// coefficients, S_xx the sample covariance of the covariates (divisor n - 1).
    pub systematic_variance: f64,
    /// `max(total - systematic, 0)` (descriptive).
    pub idiosyncratic_variance: f64,
    /// systematic / total, in [0, 1] (descriptive).
    pub r_squared: f64,
    /// Wald chi-squared statistic of the interaction coefficients (HC2 covariance).
    pub systematic_test_stat: f64,
    /// Upper-tail chi-squared p-value of the Wald statistic on [`Self::systematic_df`] df.
    pub systematic_p_value: f64,
    /// Kolmogorov-Smirnov distance between the arms' covariate-adjusted residuals (NaN when
    /// an arm has too few units for its regression).
    pub idiosyncratic_test_stat: f64,
    /// Randomization p-value of no idiosyncratic variation given the covariates (plug-in
    /// coefficients; NaN when not computed).
    pub idiosyncratic_p_value: f64,
    /// Covariate importance: |beta_j| sd(x_j), normalised to sum 1, largest first.
    /// Format: (covariate_index, importance_score)
    pub covariate_importance: Vec<(usize, f64)>,
    /// Covariate names (if available)
    pub covariate_names: Vec<String>,
    /// Degrees of freedom of the Wald test (non-aliased interactions).
    #[serde(default)]
    pub systematic_df: usize,
    /// Interaction coefficients, one per covariate (NaN when aliased).
    #[serde(default)]
    pub systematic_coefficients: Vec<f64>,
    /// HC2 standard errors of the interaction coefficients (NaN when aliased).
    #[serde(default)]
    pub systematic_se: Vec<f64>,
}

impl fmt::Display for HetDecomposition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Treatment Effect Heterogeneity Decomposition")?;
        writeln!(f, "=============================================")?;
        writeln!(f)?;
        writeln!(f, "Variance Components (descriptive):")?;
        writeln!(f, "  Total variance:       {:>10.4}", self.total_variance)?;
        writeln!(
            f,
            "  Systematic variance:  {:>10.4} ({:.1}%)",
            self.systematic_variance,
            self.r_squared * 100.0
        )?;
        writeln!(
            f,
            "  Idiosyncratic variance: {:>8.4} ({:.1}%)",
            self.idiosyncratic_variance,
            (1.0 - self.r_squared) * 100.0
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "Systematic variation (H0: no interaction of treatment and covariates):"
        )?;
        writeln!(
            f,
            "  HC2 Wald chi-squared = {:.4} on {} df, p-value: {:.4}",
            self.systematic_test_stat, self.systematic_df, self.systematic_p_value
        )?;
        for (j, (b, se)) in self
            .systematic_coefficients
            .iter()
            .zip(&self.systematic_se)
            .enumerate()
        {
            let name = self
                .covariate_names
                .get(j)
                .cloned()
                .unwrap_or_else(|| format!("x{}", j + 1));
            writeln!(f, "    treat:{:<16} {:>10.4} (SE {:.4})", name, b, se)?;
        }
        writeln!(
            f,
            "Idiosyncratic variation (H0: tau_i = tau(X_i); covariate-adjusted KS, plug-in):"
        )?;
        writeln!(
            f,
            "  KS = {:.4}, p-value: {:.4}",
            self.idiosyncratic_test_stat, self.idiosyncratic_p_value
        )?;

        if !self.covariate_importance.is_empty() {
            writeln!(f)?;
            writeln!(f, "Covariate Importance (explaining heterogeneity):")?;
            for (idx, (cov_idx, importance)) in self.covariate_importance.iter().enumerate() {
                let name = if *cov_idx < self.covariate_names.len() {
                    &self.covariate_names[*cov_idx]
                } else {
                    "Unknown"
                };
                if idx >= 10 {
                    writeln!(
                        f,
                        "  ... {} more covariates",
                        self.covariate_importance.len() - 10
                    )?;
                    break;
                }
                writeln!(f, "  {:<20} {:.4}", name, importance)?;
            }
        }

        Ok(())
    }
}

/// Result from treatment effect heterogeneity test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HetTxResult {
    /// Observed test statistic (SKS: the largest ecdf distance, in [0, 1]).
    pub test_statistic: f64,
    /// Type of test statistic used
    pub test_statistic_type: HetTestStat,
    /// p-value of H0: tau_i = tau: min(1, max over the grid of p(tau) + gamma), or p(ATE)
    /// with `plugin`.
    pub p_value: f64,
    /// Significance level
    pub significance: SignificanceLevel,
    /// The statistics of the drawn assignments at the tau that attains the largest p(tau).
    pub null_distribution: Vec<f64>,
    /// Imputed unit effects at the observed assignment.
    pub estimated_effects: Vec<f64>,
    /// Average treatment effect: the difference in means.
    pub ate: f64,
    /// Neyman standard error of the ATE, sqrt(s1^2 / n1 + s0^2 / n0).
    pub ate_se: f64,
    /// Method used for effect estimation
    pub effect_method: EffectEstimationMethod,
    /// Heterogeneity decomposition (if computed)
    pub decomposition: Option<HetDecomposition>,
    /// Number of observations
    pub n_obs: usize,
    /// Number of treated units
    pub n_treated: usize,
    /// Number of control units
    pub n_control: usize,
    /// Number of drawn assignments
    pub n_permutations: usize,
    /// Summary statistics for individual effects
    pub effect_summary: EffectSummary,
    /// Warnings generated during estimation
    pub warnings: Vec<String>,
    /// The 1 - gamma Neyman interval of the ATE the grid spans (`None` with `plugin`).
    #[serde(default)]
    pub ci: Option<(f64, f64)>,
    /// The tau values tested.
    #[serde(default)]
    pub tau_grid: Vec<f64>,
    /// p(tau) at each grid value.
    #[serde(default)]
    pub p_by_tau: Vec<f64>,
    /// gamma of the interval (added to the maximum p(tau)).
    #[serde(default)]
    pub gamma: f64,
    /// Whether the test is the plug-in test at tau = ATE.
    #[serde(default)]
    pub plugin: bool,
}

/// Summary statistics for individual treatment effects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectSummary {
    /// Minimum effect
    pub min: f64,
    /// 10th percentile
    pub p10: f64,
    /// 25th percentile (Q1)
    pub p25: f64,
    /// Median (Q2)
    pub median: f64,
    /// 75th percentile (Q3)
    pub p75: f64,
    /// 90th percentile
    pub p90: f64,
    /// Maximum effect
    pub max: f64,
    /// Standard deviation
    pub std_dev: f64,
}

impl fmt::Display for HetTxResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Treatment Effect Heterogeneity Test (hettx)")?;
        writeln!(f, "===========================================")?;
        writeln!(f)?;

        writeln!(f, "Test of a constant effect (H0: tau_i = tau for all i)")?;
        writeln!(f, "H1: Treatment effects vary across units")?;
        writeln!(f)?;

        writeln!(f, "Test Results:")?;
        writeln!(
            f,
            "  Statistic:     {} = {:.4}",
            self.test_statistic_type, self.test_statistic
        )?;
        writeln!(
            f,
            "  P-value:       {:.4}{}",
            self.p_value,
            self.significance.stars()
        )?;
        match self.ci {
            Some((lo, hi)) => {
                writeln!(
                    f,
                    "  p = max over {} grid points of tau in the {} interval [{:.4}, {:.4}] + gamma ({})",
                    self.tau_grid.len(),
                    format_level(1.0 - self.gamma),
                    lo,
                    hi,
                    self.gamma
                )?;
            }
            None => writeln!(f, "  p at tau = ATE (plug-in, no gamma)")?,
        }
        writeln!(f, "  Assignments:   {}", self.n_permutations)?;
        writeln!(f)?;

        writeln!(f, "Average Treatment Effect:")?;
        writeln!(f, "  ATE = {:.4} (Neyman SE: {:.4})", self.ate, self.ate_se)?;
        writeln!(f)?;

        writeln!(f, "Imputed Individual Effects ({}):", self.effect_method)?;
        writeln!(f, "  Min:    {:.4}", self.effect_summary.min)?;
        writeln!(f, "  10%:    {:.4}", self.effect_summary.p10)?;
        writeln!(f, "  25%:    {:.4}", self.effect_summary.p25)?;
        writeln!(f, "  Median: {:.4}", self.effect_summary.median)?;
        writeln!(f, "  75%:    {:.4}", self.effect_summary.p75)?;
        writeln!(f, "  90%:    {:.4}", self.effect_summary.p90)?;
        writeln!(f, "  Max:    {:.4}", self.effect_summary.max)?;
        writeln!(f, "  SD:     {:.4}", self.effect_summary.std_dev)?;
        writeln!(f)?;

        writeln!(f, "Sample:")?;
        writeln!(f, "  Total:     {}", self.n_obs)?;
        writeln!(f, "  Treated:   {}", self.n_treated)?;
        writeln!(f, "  Control:   {}", self.n_control)?;
        writeln!(f)?;

        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05")?;

        if let Some(ref decomp) = self.decomposition {
            writeln!(f)?;
            write!(f, "{}", decomp)?;
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

/// A level as a percentage with trailing zeros trimmed (0.9999 -> "99.99%").
fn format_level(level: f64) -> String {
    let s = format!("{:.6}", 100.0 * level);
    let s = s.trim_end_matches('0').trim_end_matches('.');
    format!("{s}%")
}

// ===============================================================================
// Main Functions
// ===============================================================================

/// Run treatment effect heterogeneity test using Dataset interface.
///
/// Columns are read with [`NaPolicy::Fail`]: a missing value is an error naming the column,
/// as is a text column ([`EconError::NonNumericColumn`]).
///
/// # Arguments
/// * `dataset` - Dataset containing outcome, treatment, and covariates
/// * `outcome_col` - Name of the outcome variable column
/// * `treatment_col` - Name of the binary treatment indicator column (0/1)
/// * `covariate_cols` - Names of covariate columns (used for matching/imputation and decomposition)
/// * `config` - Configuration options
///
/// # Example
/// ```ignore
/// let mut config = HetTxConfig::default();
/// config.n_permutations = 1000;
/// let result = run_hettx_dataset(&dataset, "outcome", "treatment", &["x1", "x2"], config)?;
/// println!("{}", result);
/// ```
pub fn run_hettx_dataset(
    dataset: &Dataset,
    outcome_col: &str,
    treatment_col: &str,
    covariate_cols: &[&str],
    config: HetTxConfig,
) -> EconResult<HetTxResult> {
    let cols = ColumnRequest::new(dataset.df())
        .numeric(outcome_col)
        .numeric(treatment_col)
        .numerics(covariate_cols)
        .na(NaPolicy::Fail)
        .extract()?;
    let y = cols.numeric_array(outcome_col);
    let treatment = cols.numeric_array(treatment_col);
    let covariates = cols.matrix(covariate_cols, false)?.data;
    let covariate_names: Vec<String> = covariate_cols.iter().map(|s| s.to_string()).collect();

    run_hettx_with_names(
        &y.view(),
        &treatment.view(),
        Some(&covariates.view()),
        config,
        covariate_names,
    )
}

/// Run treatment effect heterogeneity test using ArrayView interface.
///
/// # Arguments
/// * `y` - Outcome variable (n x 1)
/// * `treatment` - Binary treatment indicator (n x 1); values of at least 0.5 are treated
/// * `covariates` - Optional covariate matrix (n x k) for imputation and decomposition
/// * `config` - Configuration options
///
/// # Mathematical Details
///
/// See the module documentation: the test of a constant effect over the 1 - gamma interval,
/// the imputation of unit effects (matching: the mean outcome of the k nearest units of the
/// other arm, ties of distance broken by row order; regression: separate least-squares fits
/// per arm), the HC2 interaction test and the covariate-adjusted KS test.
///
/// # References
///
/// Ding, P., Feller, A., & Miratrix, L. (2016). "Randomization inference for
/// treatment effect variation." *JRSSB*, 78(3), 655-671.
pub fn run_hettx(
    y: &ArrayView1<f64>,
    treatment: &ArrayView1<f64>,
    covariates: Option<&ArrayView2<f64>>,
    config: HetTxConfig,
) -> EconResult<HetTxResult> {
    run_hettx_with_names(y, treatment, covariates, config, Vec::new())
}

/// Internal function with covariate names support.
fn run_hettx_with_names(
    y: &ArrayView1<f64>,
    treatment: &ArrayView1<f64>,
    covariates: Option<&ArrayView2<f64>>,
    config: HetTxConfig,
    covariate_names: Vec<String>,
) -> EconResult<HetTxResult> {
    let n = y.len();
    let mut warnings = Vec::new();

    // Validate inputs
    if treatment.len() != n {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Treatment vector length ({}) does not match outcome length ({})",
                treatment.len(),
                n
            ),
        });
    }

    if let Some(x) = covariates
        && x.nrows() != n
    {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Covariate matrix rows ({}) do not match outcome length ({})",
                x.nrows(),
                n
            ),
        });
    }

    validate_config(y, treatment, covariates, &config)?;

    let z: Vec<u8> = treatment.iter().map(|&t| u8::from(t >= 0.5)).collect();
    let n_treated = z.iter().filter(|&&v| v == 1).count();
    let n_control = n - n_treated;

    if n_treated < 5 || n_control < 5 {
        return Err(EconError::InsufficientData {
            required: 5,
            provided: n_treated.min(n_control),
            context: "Need at least 5 units in each treatment group".to_string(),
        });
    }

    // Neyman ATE and SE
    let (ate, ate_se) = neyman(y, &z);

    // The tau grid over the 1 - gamma interval, or the ATE alone.
    let (ci, tau_grid) = if config.plugin {
        (None, vec![ate])
    } else {
        let half = normal_quantile(1.0 - config.gamma / 2.0)? * ate_se;
        let (lo, hi) = (ate - half, ate + half);
        (Some((lo, hi)), even_grid(lo, hi, config.grid_size))
    };

    // Imputed unit effects at the observed assignment
    let x_int = covariates.map(add_intercept);
    let names = covariates
        .map(|x| design_names(&covariate_names, x.ncols()))
        .unwrap_or_default();
    let imputer = Imputer {
        x: covariates.map(|x| x.view()),
        x_int: x_int.as_ref(),
        method: config.effect_method,
        n_neighbors: config.n_neighbors,
        names: &names,
    };
    let (tau_hat, _) = imputer.impute(y, None, &z, &mut warnings)?;
    let effect_summary = compute_effect_summary(&tau_hat);

    let seed = config.seed.unwrap_or(HETTX_DEFAULT_SEED);
    let mut assign = |b: usize, zb: &mut [u8]| draw_assignment(&z, seed, b, zb);
    let frt = frt_constant_effect(
        y,
        &z,
        &imputer,
        config.test_statistic,
        Some(&tau_hat),
        &tau_grid,
        config.n_permutations,
        &mut assign,
    )?;
    if frt.n_valid < config.n_permutations {
        warnings.push(format!(
            "{} of {} drawn assignments yielded valid statistics",
            frt.n_valid, config.n_permutations
        ));
    }

    let p_max = frt
        .p_by_tau
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let p_value = if config.plugin {
        p_max
    } else {
        (p_max + config.gamma).min(1.0)
    };
    let significance = SignificanceLevel::from_p_value(p_value);
    let argmax = frt
        .p_by_tau
        .iter()
        .position(|&p| p == p_max)
        .unwrap_or_default();
    let null_distribution: Vec<f64> = frt.null.column(argmax).to_vec();

    // Decomposition (needs covariates)
    let decomposition = match covariates {
        Some(x) if config.decompose && x.ncols() > 0 => Some(compute_heterogeneity_decomposition(
            y,
            &z,
            x,
            &tau_hat,
            &covariate_names,
            config.n_permutations,
            &mut assign,
            config.compute_importance,
            &mut warnings,
        )?),
        _ => {
            if config.decompose {
                warnings.push("Decomposition requested but no covariates provided".to_string());
            }
            None
        }
    };

    Ok(HetTxResult {
        test_statistic: frt.t_obs,
        test_statistic_type: config.test_statistic,
        p_value,
        significance,
        null_distribution,
        estimated_effects: tau_hat.to_vec(),
        ate,
        ate_se,
        effect_method: config.effect_method,
        decomposition,
        n_obs: n,
        n_treated,
        n_control,
        n_permutations: config.n_permutations,
        effect_summary,
        warnings,
        ci,
        tau_grid,
        p_by_tau: frt.p_by_tau,
        gamma: config.gamma,
        plugin: config.plugin,
    })
}

// ===============================================================================
// Randomization test of a constant effect
// ===============================================================================

/// Draw `b` of complete randomization: a permutation of `z` from stream `b` of `seed`.
fn draw_assignment(z: &[u8], seed: u64, b: usize, zb: &mut [u8]) {
    zb.copy_from_slice(z);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(b as u64);
    zb.shuffle(&mut rng);
}

/// The difference in means and its Neyman SE sqrt(s1^2 / n1 + s0^2 / n0).
fn neyman(y: &ArrayView1<f64>, z: &[u8]) -> (f64, f64) {
    let arm = |a: u8| {
        let v: Vec<f64> = y
            .iter()
            .zip(z)
            .filter(|(_, zi)| **zi == a)
            .map(|(&v, _)| v)
            .collect();
        let m = v.iter().sum::<f64>() / v.len() as f64;
        let s2 = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
        (m, s2 / v.len() as f64)
    };
    let (m1, v1) = arm(1);
    let (m0, v0) = arm(0);
    (m1 - m0, (v1 + v0).sqrt())
}

/// `size` evenly spaced values from `lo` to `hi`, as R's `seq(lo, hi, length.out = size)`
/// (the last value is `hi` itself).
fn even_grid(lo: f64, hi: f64, size: usize) -> Vec<f64> {
    if size == 1 {
        return vec![lo];
    }
    let by = (hi - lo) / (size - 1) as f64;
    (0..size)
        .map(|i| {
            if i + 1 == size {
                hi
            } else {
                lo + i as f64 * by
            }
        })
        .collect()
}

/// The outcome of [`frt_constant_effect`].
#[derive(Debug, Clone)]
pub(crate) struct ConstantEffectFrt {
    /// Observed statistic (independent of tau).
    pub(crate) t_obs: f64,
    /// SKS only: the observed statistic's integer numerator max |a n0 - b n1|.
    pub(crate) sks_numerator: Option<u64>,
    /// p(tau) for each grid value.
    pub(crate) p_by_tau: Vec<f64>,
    /// The statistics of the valid draws (rows) at each grid value (columns).
    pub(crate) null: Array2<f64>,
    /// Number of draws whose statistic could be computed.
    pub(crate) n_valid: usize,
}

/// The randomization test of H0: tau_i = tau at every tau of `grid`, with `n_draws`
/// assignments from `assign(b, zb)` (the same assignments for every tau).
///
/// `observed_effects` are the imputed effects at the observed assignment (needed for a spread
/// statistic). Each draw sorts nothing for SKS beyond one pass over the outcome order computed
/// once, and imputes once for a spread statistic (see the module documentation).
#[allow(clippy::too_many_arguments)]
pub(crate) fn frt_constant_effect(
    y: &ArrayView1<f64>,
    z: &[u8],
    imputer: &Imputer<'_>,
    stat: HetTestStat,
    observed_effects: Option<&Array1<f64>>,
    grid: &[f64],
    n_draws: usize,
    assign: &mut dyn FnMut(usize, &mut [u8]),
) -> EconResult<ConstantEffectFrt> {
    let n = y.len();
    let n1 = z.iter().filter(|&&v| v == 1).count();
    let n0 = n - n1;
    let g = grid.len();
    let mut null = resource::try_zeros2(n_draws, g, "randomization null distribution")?;
    let mut zb = resource::try_vec(n, 0u8, "drawn assignment")?;
    let mut n_valid = 0usize;

    let (t_obs, sks_numerator) = if stat == HetTestStat::SKS {
        let order = outcome_order(y);
        let mut pieces = SksPieces::default();
        pieces.fill(y, &order, z, z);
        let num = pieces.numerator(0.0, n1, n0);
        let scale = (n1 * n0) as f64;
        for b in 0..n_draws {
            resource::checkpoint()?;
            assign(b, &mut zb);
            pieces.fill(y, &order, z, &zb);
            for (cell, &tau) in null.row_mut(b).iter_mut().zip(grid) {
                *cell = pieces.numerator(tau, n1, n0) as f64 / scale;
            }
            n_valid += 1;
        }
        (num as f64 / (n1 * n0) as f64, Some(num))
    } else {
        if imputer.method == EffectEstimationMethod::Stratified {
            return Err(stratified_spread_error(stat));
        }
        let observed = observed_effects.ok_or_else(|| EconError::InvalidSpecification {
            message: "a spread statistic needs the imputed effects".to_string(),
        })?;
        let t_obs = spread_statistic(stat, &observed.to_vec(), None, 0.0, &mut Vec::new());
        let mut d = vec![0.0; n];
        let mut buf = Vec::with_capacity(n);
        let mut ignored = Vec::new();
        for b in 0..n_draws {
            resource::checkpoint()?;
            assign(b, &mut zb);
            for i in 0..n {
                d[i] = f64::from(zb[i]) - f64::from(z[i]);
            }
            let (a, bv) = match imputer.impute(y, Some(&d), &zb, &mut ignored) {
                Ok((a, Some(bv))) => (a, bv),
                Ok((_, None)) => unreachable!("imputing a direction returns its effects"),
                Err(e) if is_resource_stop(&e) => return Err(e),
                Err(_) => continue,
            };
            let a = a.to_vec();
            let bv = bv.to_vec();
            let moments = SpreadMoments::new(&a, &bv);
            let mut row = null.row_mut(n_valid);
            let mut ok = true;
            for (cell, &tau) in row.iter_mut().zip(grid) {
                let t = if stat == HetTestStat::Variance {
                    moments.variance(tau)
                } else {
                    spread_statistic(stat, &a, Some(&bv), tau, &mut buf)
                };
                ok &= t.is_finite();
                *cell = t;
            }
            if ok {
                n_valid += 1;
            }
        }
        (t_obs, None)
    };

    let valid = null.slice(ndarray::s![..n_valid, ..]);
    let p_by_tau = (0..g)
        .map(|j| {
            let exceed = valid.column(j).iter().filter(|&&t| t >= t_obs).count();
            (exceed as f64 + 1.0) / (n_valid as f64 + 1.0)
        })
        .collect();
    Ok(ConstantEffectFrt {
        t_obs,
        sks_numerator,
        p_by_tau,
        null: valid.to_owned(),
        n_valid,
    })
}

/// Row indices in increasing order of `y` (ties by row).
fn outcome_order(y: &ArrayView1<f64>) -> Vec<usize> {
    let mut order: Vec<usize> = (0..y.len()).collect();
    order.sort_by(|&a, &b| y[a].total_cmp(&y[b]).then(a.cmp(&b)));
    order
}

/// The outcomes split by (observed arm Z, drawn arm Zb), each piece in increasing order of
/// the outcome. Under H0 at tau, the drawn treated arm holds `t1` (Z = 1, unshifted) and
/// `c1` (Z = 0, shifted by +tau); the drawn control arm holds `c0` (Z = 0) and `t0`
/// (Z = 1, shifted by -tau).
#[derive(Debug, Default)]
struct SksPieces {
    t1: Vec<f64>,
    c1: Vec<f64>,
    c0: Vec<f64>,
    t0: Vec<f64>,
}

impl SksPieces {
    fn fill(&mut self, y: &ArrayView1<f64>, order: &[usize], z: &[u8], zb: &[u8]) {
        for v in [&mut self.t1, &mut self.c1, &mut self.c0, &mut self.t0] {
            v.clear();
        }
        for &i in order {
            let piece = match (z[i], zb[i]) {
                (1, 1) => &mut self.t1,
                (0, 1) => &mut self.c1,
                (0, _) => &mut self.c0,
                _ => &mut self.t0,
            };
            piece.push(y[i]);
        }
    }

    /// max over pooled points u of |a(u) n0 - b(u) n1|, a and b the counts of the centred
    /// treated and control values at or below u (the SKS statistic times n1 n0).
    fn numerator(&self, tau: f64, n1: usize, n0: usize) -> u64 {
        let sum = |v: &[f64]| v.iter().sum::<f64>();
        let m1 = (sum(&self.t1) + sum(&self.c1) + tau * self.c1.len() as f64) / n1 as f64;
        let m0 = (sum(&self.c0) + sum(&self.t0) - tau * self.t0.len() as f64) / n0 as f64;
        // (values, shift, arm): arm 1 counts toward a, arm 0 toward b
        let pieces: [(&[f64], f64, bool); 4] = [
            (&self.t1, 0.0, true),
            (&self.c1, tau, true),
            (&self.c0, 0.0, false),
            (&self.t0, -tau, false),
        ];
        let value = |p: usize, k: usize| {
            let (v, shift, arm) = pieces[p];
            (v[k] + shift) - if arm { m1 } else { m0 }
        };
        let mut pos = [0usize; 4];
        let (mut a, mut b) = (0i64, 0i64);
        let (n1, n0) = (n1 as i64, n0 as i64);
        let mut best = 0i64;
        loop {
            let mut u = f64::INFINITY;
            let mut any = false;
            for p in 0..4 {
                if pos[p] < pieces[p].0.len() {
                    let v = value(p, pos[p]);
                    if !any || v < u {
                        u = v;
                        any = true;
                    }
                }
            }
            if !any {
                break;
            }
            for p in 0..4 {
                while pos[p] < pieces[p].0.len() && value(p, pos[p]) <= u {
                    pos[p] += 1;
                    if pieces[p].2 {
                        a += 1;
                    } else {
                        b += 1;
                    }
                }
            }
            best = best.max((a * n0 - b * n1).abs());
        }
        best as u64
    }
}

/// Centred second moments of imputed effects `a + tau b`, for the variance in closed form.
struct SpreadMoments {
    saa: f64,
    sab: f64,
    sbb: f64,
    denom: f64,
}

impl SpreadMoments {
    fn new(a: &[f64], b: &[f64]) -> Self {
        let n = a.len();
        let ma = a.iter().sum::<f64>() / n as f64;
        let mb = b.iter().sum::<f64>() / n as f64;
        let (mut saa, mut sab, mut sbb) = (0.0, 0.0, 0.0);
        for (x, y) in a.iter().zip(b) {
            let (da, db) = (x - ma, y - mb);
            saa += da * da;
            sab += da * db;
            sbb += db * db;
        }
        Self {
            saa,
            sab,
            sbb,
            denom: (n - 1).max(1) as f64,
        }
    }

    /// The sample variance of `a + tau b`.
    fn variance(&self, tau: f64) -> f64 {
        ((self.saa + 2.0 * tau * self.sab + tau * tau * self.sbb) / self.denom).max(0.0)
    }
}

/// A spread statistic of `a + tau b` (`a` alone without `b`); `buf` is workspace.
fn spread_statistic(
    stat: HetTestStat,
    a: &[f64],
    b: Option<&[f64]>,
    tau: f64,
    buf: &mut Vec<f64>,
) -> f64 {
    buf.clear();
    match b {
        Some(b) => buf.extend(a.iter().zip(b).map(|(x, y)| x + tau * y)),
        None => buf.extend_from_slice(a),
    }
    compute_test_statistic(buf, stat)
}

// ===============================================================================
// Helper Functions
// ===============================================================================

/// Whether an error must stop the computation instead of being skipped as a
/// failed draw: a cancelled budget or a refused allocation.
fn is_resource_stop(e: &EconError) -> bool {
    matches!(e, EconError::Cancelled(_) | EconError::ResourceLimit(_))
}

/// Check the controls and refuse non-finite data, which the nearest-neighbour search and
/// the statistics cannot order.
fn validate_config(
    y: &ArrayView1<f64>,
    treatment: &ArrayView1<f64>,
    covariates: Option<&ArrayView2<f64>>,
    config: &HetTxConfig,
) -> EconResult<()> {
    if config.n_permutations == 0 {
        return Err(EconError::InvalidSpecification {
            message: "n_permutations must be at least 1 (got 0)".to_string(),
        });
    }
    if config.effect_method == EffectEstimationMethod::Matching && config.n_neighbors == 0 {
        return Err(EconError::InvalidSpecification {
            message: "n_neighbors must be at least 1 for matching (got 0)".to_string(),
        });
    }
    if !(config.gamma.is_finite() && config.gamma > 0.0 && config.gamma < 1.0) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "gamma must be strictly between 0 and 1, got {}",
                config.gamma
            ),
        });
    }
    if config.grid_size == 0 || (!config.plugin && config.grid_size < 2) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "grid_size must be at least 2 (1 only with plugin), got {}",
                config.grid_size
            ),
        });
    }
    if config.effect_method == EffectEstimationMethod::Stratified
        && config.test_statistic != HetTestStat::SKS
    {
        return Err(stratified_spread_error(config.test_statistic));
    }
    let non_finite = |what: &str| {
        Err(EconError::InvalidSpecification {
            message: format!("{what} contains non-finite values (NaN or infinite)"),
        })
    };
    if y.iter().any(|v| !v.is_finite()) {
        return non_finite("The outcome");
    }
    if treatment.iter().any(|v| !v.is_finite()) {
        return non_finite("The treatment indicator");
    }
    if let Some(x) = covariates
        && x.iter().any(|v| !v.is_finite())
    {
        return non_finite("The covariate matrix");
    }
    Ok(())
}

/// The refusal of a spread statistic with [`EffectEstimationMethod::Stratified`].
fn stratified_spread_error(stat: HetTestStat) -> EconError {
    EconError::InvalidSpecification {
        message: format!(
            "effect_method stratified gives every unit the difference in means, so the \
             statistic ({stat}) is 0 for every assignment; use test_statistic sks, or \
             effect_method matching or regression"
        ),
    }
}

/// Imputation of unit effects for one assignment. Both methods are linear in the outcome
/// for a fixed assignment (the neighbour sets and the projections depend only on the
/// covariates and the assignment), so a second outcome `d` is imputed with the same
/// neighbours or fits.
pub(crate) struct Imputer<'a> {
    x: Option<ArrayView2<'a, f64>>,
    x_int: Option<&'a Array2<f64>>,
    method: EffectEstimationMethod,
    n_neighbors: usize,
    names: &'a [String],
}

impl Imputer<'_> {
    /// The imputed effects of `y` and, when given, of `d`, under assignment `z`.
    fn impute(
        &self,
        y: &ArrayView1<f64>,
        d: Option<&[f64]>,
        z: &[u8],
        warnings: &mut Vec<String>,
    ) -> EconResult<(Array1<f64>, Option<Array1<f64>>)> {
        let n = y.len();
        let treated: Vec<usize> = (0..n).filter(|&i| z[i] == 1).collect();
        let control: Vec<usize> = (0..n).filter(|&i| z[i] == 0).collect();
        let mut a = Array1::zeros(n);
        let mut b = d.map(|_| Array1::zeros(n));
        match self.method {
            EffectEstimationMethod::Matching => {
                let x = self.x.ok_or_else(|| EconError::InvalidSpecification {
                    message: "Matching method requires covariates".to_string(),
                })?;
                let mut dist = Vec::with_capacity(treated.len().max(control.len()));
                for i in 0..n {
                    resource::checkpoint()?;
                    let donors = if z[i] == 1 { &control } else { &treated };
                    let nbrs = nearest(i, &x, donors, self.n_neighbors, &mut dist);
                    let k = nbrs.len() as f64;
                    let sign = if z[i] == 1 { 1.0 } else { -1.0 };
                    let my = nbrs.iter().map(|&(_, j)| y[j]).sum::<f64>() / k;
                    a[i] = sign * (y[i] - my);
                    if let (Some(d), Some(b)) = (d, b.as_mut()) {
                        let md = nbrs.iter().map(|&(_, j)| d[j]).sum::<f64>() / k;
                        b[i] = sign * (d[i] - md);
                    }
                }
            }
            EffectEstimationMethod::Regression => {
                let x_int = self.x_int.ok_or_else(|| EconError::InvalidSpecification {
                    message: "Regression method requires covariates".to_string(),
                })?;
                let (fit1, w1) = fit_arm(x_int, y, d, &treated, self.names)?;
                let (fit0, w0) = fit_arm(x_int, y, d, &control, self.names)?;
                for (arm, w) in [("treated", w1), ("control", w0)] {
                    if let Some(w) = w {
                        push_once(warnings, format!("Outcome model ({arm} arm): {w}"));
                    }
                }
                for i in 0..n {
                    let row = x_int.row(i);
                    a[i] = predict(&fit1[0], &row) - predict(&fit0[0], &row);
                    if let Some(b) = b.as_mut() {
                        b[i] = predict(&fit1[1], &row) - predict(&fit0[1], &row);
                    }
                }
            }
            EffectEstimationMethod::Stratified => {
                let diff = |v: &dyn Fn(usize) -> f64| {
                    treated.iter().map(|&i| v(i)).sum::<f64>() / treated.len() as f64
                        - control.iter().map(|&i| v(i)).sum::<f64>() / control.len() as f64
                };
                a.fill(diff(&|i| y[i]));
                if let (Some(d), Some(b)) = (d, b.as_mut()) {
                    b.fill(diff(&|i| d[i]));
                }
            }
        }
        Ok((a, b))
    }
}

/// The `k` nearest donors of unit `target` (Euclidean distance, ties by row order, as a
/// stable sort by distance would give), nearest first. `dist` is workspace.
fn nearest<'d>(
    target: usize,
    x: &ArrayView2<f64>,
    donors: &[usize],
    k: usize,
    dist: &'d mut Vec<(f64, usize)>,
) -> &'d [(f64, usize)] {
    let t = x.row(target);
    dist.clear();
    dist.extend(donors.iter().map(|&j| {
        let d2: f64 = t
            .iter()
            .zip(x.row(j).iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        (d2.sqrt(), j)
    }));
    let key = |a: &(f64, usize), b: &(f64, usize)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
    let k = k.min(dist.len());
    if k > 0 && k < dist.len() {
        dist.select_nth_unstable_by(k - 1, key);
    }
    dist[..k].sort_by(key);
    &dist[..k]
}

/// Least-squares fits of `y` (and `d`) on the rows `rows` of `x` (normal equations when the
/// design is of full rank and well conditioned, otherwise lm's pivoted QR with aliased
/// coefficients NaN), and the aliasing warning.
fn fit_arm(
    x: &Array2<f64>,
    y: &ArrayView1<f64>,
    d: Option<&[f64]>,
    rows: &[usize],
    names: &[String],
) -> EconResult<(Vec<Array1<f64>>, Option<String>)> {
    let (m, k) = (rows.len(), x.ncols());
    if m <= k {
        return Err(EconError::InsufficientData {
            required: k + 1,
            provided: m,
            context: "Need more observations than parameters for regression".to_string(),
        });
    }
    let q = if d.is_some() { 2 } else { 1 };
    let mut xs = Array2::zeros((m, k));
    let mut ys = Array2::zeros((m, q));
    for (r, &i) in rows.iter().enumerate() {
        xs.row_mut(r).assign(&x.row(i));
        ys[[r, 0]] = y[i];
        if let Some(d) = d {
            ys[[r, 1]] = d[i];
        }
    }
    match normal_equations_inverse(xs.view())? {
        Some(ne) => {
            let beta = ne.coefficients_multi(xs.view(), ys.view());
            Ok(((0..q).map(|c| beta.column(c).to_owned()).collect(), None))
        }
        None => {
            let mut fits = Vec::with_capacity(q);
            let mut warning = None;
            for c in 0..q {
                let ls = least_squares(xs.view(), ys.column(c))?;
                warning = ls.aliasing_warning(names);
                fits.push(ls.coefficients);
            }
            Ok((fits, warning))
        }
    }
}

/// A prediction as `predict.lm`: aliased (NaN) coefficients count as 0.
fn predict(beta: &Array1<f64>, row: &ArrayView1<f64>) -> f64 {
    row.iter()
        .zip(beta.iter())
        .filter(|(_, b)| b.is_finite())
        .map(|(x, b)| x * b)
        .sum()
}

/// The names of the design columns: "(Intercept)", then the covariates (`x1`, `x2`, ... for
/// unnamed ones, as R's default names).
fn design_names(covariate_names: &[String], k: usize) -> Vec<String> {
    std::iter::once("(Intercept)".to_string())
        .chain((0..k).map(|j| covariate_name(covariate_names, j)))
        .collect()
}

fn covariate_name(covariate_names: &[String], j: usize) -> String {
    covariate_names
        .get(j)
        .cloned()
        .unwrap_or_else(|| format!("x{}", j + 1))
}

/// Push `w` unless `warnings` already holds it.
fn push_once(warnings: &mut Vec<String>, w: String) {
    if !warnings.contains(&w) {
        warnings.push(w);
    }
}

/// Add intercept column to design matrix.
fn add_intercept(x: &ArrayView2<f64>) -> Array2<f64> {
    let n = x.nrows();
    let k = x.ncols();
    let mut x_with_int = Array2::zeros((n, k + 1));

    for i in 0..n {
        x_with_int[[i, 0]] = 1.0;
        for j in 0..k {
            x_with_int[[i, j + 1]] = x[[i, j]];
        }
    }

    x_with_int
}

/// An outcome model fitted on a subset of rows, as R's `lm`: aliased coefficients are NaN
/// (R's NA) and the fit records R's "Coefficients not defined because of singularities".
#[derive(Debug, Clone)]
struct OutcomeFit {
    coefficients: Array1<f64>,
    r2: f64,
    warning: Option<String>,
}

impl OutcomeFit {
    /// The prediction for one design row, as `predict.lm`: aliased coefficients count as 0.
    fn predict(&self, row: &ArrayView1<f64>) -> f64 {
        predict(&self.coefficients, row)
    }
}

/// Fit OLS on a subset of observations: the normal equations when the design is of full rank
/// and well conditioned, otherwise lm's pivoted QR (aliased columns NaN, with R's warning).
/// `names` names the columns of `x` for the warning.
fn fit_outcome_model_on_subset(
    x: &Array2<f64>,
    y: &ArrayView1<f64>,
    subset_idx: &[usize],
    names: &[String],
) -> EconResult<OutcomeFit> {
    let (fits, warning) = fit_arm(x, y, None, subset_idx, names)?;
    let beta = fits.into_iter().next().unwrap_or_default();
    let ys: Vec<f64> = subset_idx.iter().map(|&i| y[i]).collect();
    let y_mean = ys.iter().sum::<f64>() / ys.len() as f64;
    let sst: f64 = ys.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let ssr: f64 = subset_idx
        .iter()
        .zip(&ys)
        .map(|(&i, &yi)| (yi - predict(&beta, &x.row(i))).powi(2))
        .sum();
    let r2 = if sst > 0.0 { 1.0 - ssr / sst } else { 0.0 };
    Ok(OutcomeFit {
        coefficients: beta,
        r2,
        warning,
    })
}

/// A spread statistic of the unit effects `tau`.
fn compute_test_statistic(tau: &[f64], stat_type: HetTestStat) -> f64 {
    let n = tau.len();
    if n == 0 {
        return 0.0;
    }
    let mean = || tau.iter().sum::<f64>() / n as f64;

    match stat_type {
        HetTestStat::Variance => {
            let m = mean();
            tau.iter().map(|&t| (t - m).powi(2)).sum::<f64>() / (n - 1).max(1) as f64
        }
        HetTestStat::Range => {
            let min = tau.iter().copied().fold(f64::INFINITY, f64::min);
            let max = tau.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            max - min
        }
        HetTestStat::IQR => {
            // R's IQR(): type-7 quantiles (tau is computed from finite, checked inputs)
            let mut sorted: Vec<f64> = tau.to_vec();
            sorted.sort_unstable_by(f64::total_cmp);
            let q = |p| quantile_sorted(&sorted, p, QuantileType::T7);
            q(0.75) - q(0.25)
        }
        HetTestStat::MeanAbsDeviation => {
            let m = mean();
            tau.iter().map(|&t| (t - m).abs()).sum::<f64>() / n as f64
        }
        // The SKS statistic is computed from the outcomes, not from imputed effects.
        HetTestStat::SKS => f64::NAN,
    }
}

/// Compute summary statistics for individual effects.
fn compute_effect_summary(tau: &Array1<f64>) -> EffectSummary {
    let n = tau.len();
    if n == 0 {
        return EffectSummary {
            min: 0.0,
            p10: 0.0,
            p25: 0.0,
            median: 0.0,
            p75: 0.0,
            p90: 0.0,
            max: 0.0,
            std_dev: 0.0,
        };
    }

    // R's quantile() (type 7); tau is computed from finite, checked inputs
    let mut sorted: Vec<f64> = tau.to_vec();
    sorted.sort_by(f64::total_cmp);
    let q = |p| quantile_sorted(&sorted, p, QuantileType::T7);

    let mean = tau.mean().unwrap_or(0.0);
    let var = tau.iter().map(|&t| (t - mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;

    EffectSummary {
        min: sorted[0],
        p10: q(0.10),
        p25: q(0.25),
        median: q(0.5),
        p75: q(0.75),
        p90: q(0.90),
        max: sorted[n - 1],
        std_dev: var.sqrt(),
    }
}

// ===============================================================================
// Decomposition
// ===============================================================================

/// The HC2 interaction fit `y ~ 1 + z + X + z:X`.
struct InteractionFit {
    /// Coefficient of z (the effect at X = 0); NaN when aliased.
    b_z: f64,
    /// Interaction coefficients, one per covariate; NaN when aliased.
    beta: Vec<f64>,
    /// Their HC2 standard errors; NaN when aliased (or at a leverage of 1).
    se: Vec<f64>,
    /// Wald chi-squared of the non-aliased interactions and its df.
    wald: f64,
    df: usize,
}

/// Fit `y ~ 1 + z + X + z:X` by least squares and test the interactions with their HC2
/// covariance (sandwich's `vcovHC(type = "HC2")`: leverages of the fitted design, meat
/// sum e_i^2 / (1 - h_i) x_i x_i').
fn interaction_hc2(
    y: &ArrayView1<f64>,
    z: &[u8],
    x: &ArrayView2<f64>,
    covariate_names: &[String],
    warnings: &mut Vec<String>,
) -> EconResult<InteractionFit> {
    let (n, k) = x.dim();
    let p = 2 + 2 * k;
    let mut xd = resource::try_zeros2(n, p, "interaction design")?;
    for i in 0..n {
        let zi = f64::from(z[i]);
        xd[[i, 0]] = 1.0;
        xd[[i, 1]] = zi;
        for j in 0..k {
            xd[[i, 2 + j]] = x[[i, j]];
            xd[[i, 2 + k + j]] = zi * x[[i, j]];
        }
    }
    let mut names = vec!["(Intercept)".to_string(), "treat".to_string()];
    names.extend((0..k).map(|j| covariate_name(covariate_names, j)));
    names.extend((0..k).map(|j| format!("treat:{}", covariate_name(covariate_names, j))));
    let ls = least_squares(xd.view(), y.view())?;
    if let Some(w) = ls.aliasing_warning(&names) {
        warnings.push(format!("Interaction model: {w}"));
    }
    let kept: Vec<usize> = (0..p).filter(|&j| ls.coefficients[j].is_finite()).collect();
    let r = kept.len();
    let mut bread = Array2::zeros((r, r));
    for (a, &ja) in kept.iter().enumerate() {
        for (b, &jb) in kept.iter().enumerate() {
            bread[[a, b]] = ls.unscaled_cov[[ja, jb]];
        }
    }
    // Rows of the kept design scaled by sqrt(e_i^2 / (1 - h_i)).
    let mut scaled = resource::try_zeros2(n, r, "HC2 scores")?;
    let mut leverage_one = false;
    for i in 0..n {
        let row: Vec<f64> = kept.iter().map(|&j| xd[[i, j]]).collect();
        let mut h = 0.0;
        for a in 0..r {
            let mut s = 0.0;
            for b in 0..r {
                s += bread[[a, b]] * row[b];
            }
            h += row[a] * s;
        }
        let e = ls.residuals[i];
        let w = if 1.0 - h > 1e-10 {
            e.abs() / (1.0 - h).sqrt()
        } else {
            leverage_one = true;
            f64::NAN
        };
        for a in 0..r {
            scaled[[i, a]] = row[a] * w;
        }
    }
    if leverage_one {
        warnings.push(
            "An observation has leverage 1 in the interaction model, so its HC2 weight and \
             the systematic test are undefined"
                .to_string(),
        );
    }
    let meat = xtx(&scaled.view());
    let v = bread.dot(&meat).dot(&bread);

    let mut beta = vec![f64::NAN; k];
    let mut se = vec![f64::NAN; k];
    let mut idx = Vec::new();
    for (a, &j) in kept.iter().enumerate() {
        if j >= 2 + k {
            beta[j - 2 - k] = ls.coefficients[j];
            se[j - 2 - k] = v[[a, a]].sqrt();
            idx.push(a);
        }
    }
    let df = idx.len();
    let wald = if df == 0 {
        f64::NAN
    } else {
        let b: Array1<f64> = idx.iter().map(|&a| ls.coefficients[kept[a]]).collect();
        let vi = Array2::from_shape_fn((df, df), |(r1, r2)| v[[idx[r1], idx[r2]]]);
        match sym_inverse(
            vi.view(),
            SingularPolicy::Error,
            "HC2 covariance of the interaction coefficients",
        ) {
            Ok(inv) => b.dot(&inv.inverse.dot(&b)),
            Err(e) if is_resource_stop(&e) => return Err(e),
            Err(e) => {
                warnings.push(format!("Systematic test not computed: {e}"));
                f64::NAN
            }
        }
    };
    Ok(InteractionFit {
        b_z: ls.coefficients[1],
        beta,
        se,
        wald,
        df,
    })
}

/// The sample covariance of the columns of `x` (divisor n - 1).
fn sample_cov(x: &ArrayView2<f64>) -> Array2<f64> {
    let (n, k) = x.dim();
    let means: Vec<f64> = (0..k).map(|j| x.column(j).sum() / n as f64).collect();
    let mut c = Array2::zeros((n, k));
    for i in 0..n {
        for j in 0..k {
            c[[i, j]] = x[[i, j]] - means[j];
        }
    }
    xtx(&c.view()) / (n - 1).max(1) as f64
}

/// The KS distance between the two arms' residuals from their own regressions on
/// `x_int` (the SKS statistic with covariate adjustment), as the integer numerator
/// max |a n0 - b n1|; `None` when an arm's regression cannot be fitted.
fn residual_ks_numerator(
    y: &[f64],
    z: &[u8],
    x_int: &Array2<f64>,
    n1: usize,
    n0: usize,
) -> EconResult<Option<u64>> {
    let mut res = [Vec::with_capacity(n0), Vec::with_capacity(n1)];
    for arm in [0u8, 1] {
        let rows: Vec<usize> = (0..y.len()).filter(|&i| z[i] == arm).collect();
        let yv = ArrayView1::from(y);
        let beta = match fit_arm(x_int, &yv, None, &rows, &[]) {
            Ok((fits, _)) => fits.into_iter().next().unwrap_or_default(),
            Err(e) if is_resource_stop(&e) => return Err(e),
            Err(_) => return Ok(None),
        };
        let r = &mut res[usize::from(arm)];
        for &i in &rows {
            r.push(y[i] - predict(&beta, &x_int.row(i)));
        }
        r.sort_unstable_by(f64::total_cmp);
    }
    let [r0, r1] = &res;
    let (mut i, mut j) = (0usize, 0usize);
    let (n1i, n0i) = (n1 as i64, n0 as i64);
    let mut best = 0i64;
    while i < r1.len() || j < r0.len() {
        let u = match (r1.get(i), r0.get(j)) {
            (Some(&a), Some(&b)) => a.min(b),
            (Some(&a), None) => a,
            (None, Some(&b)) => b,
            (None, None) => break,
        };
        while i < r1.len() && r1[i] <= u {
            i += 1;
        }
        while j < r0.len() && r0[j] <= u {
            j += 1;
        }
        best = best.max((i as i64 * n0i - j as i64 * n1i).abs());
    }
    Ok(Some(best as u64))
}

/// The conditional randomization test of no idiosyncratic variation (plug-in): H0:
/// tau_i = `tau_x[i]`; Y(0) = Y - Z tau_x, Yb = Y(0) + Zb tau_x; the statistic is
/// [`residual_ks_numerator`]. Returns the observed numerator, the p-value and the number of
/// valid draws; `None` when the observed statistic cannot be computed.
fn frt_idiosyncratic(
    y: &ArrayView1<f64>,
    z: &[u8],
    x_int: &Array2<f64>,
    tau_x: &[f64],
    n_draws: usize,
    assign: &mut dyn FnMut(usize, &mut [u8]),
) -> EconResult<Option<(u64, f64, usize)>> {
    let n = y.len();
    let n1 = z.iter().filter(|&&v| v == 1).count();
    let n0 = n - n1;
    let yv: Vec<f64> = y.to_vec();
    let Some(t_obs) = residual_ks_numerator(&yv, z, x_int, n1, n0)? else {
        return Ok(None);
    };
    let y0: Vec<f64> = (0..n).map(|i| y[i] - f64::from(z[i]) * tau_x[i]).collect();
    let mut zb = resource::try_vec(n, 0u8, "drawn assignment")?;
    let mut yb = vec![0.0; n];
    let (mut valid, mut exceed) = (0usize, 0usize);
    for b in 0..n_draws {
        resource::checkpoint()?;
        assign(b, &mut zb);
        for i in 0..n {
            yb[i] = y0[i] + f64::from(zb[i]) * tau_x[i];
        }
        if let Some(t) = residual_ks_numerator(&yb, &zb, x_int, n1, n0)? {
            valid += 1;
            if t >= t_obs {
                exceed += 1;
            }
        }
    }
    let p = (exceed as f64 + 1.0) / (valid as f64 + 1.0);
    Ok(Some((t_obs, p, valid)))
}

/// The heterogeneity decomposition: the HC2 interaction test of systematic variation, the
/// covariate-adjusted KS randomization test of idiosyncratic variation (plug-in), and the
/// descriptive variance split of the imputed effects `tau_hat`.
///
/// # References
/// Ding, P., Feller, A., & Miratrix, L. (2019). "Decomposing treatment effect
/// variation." *JASA*, 114(525), 304-317.
#[allow(clippy::too_many_arguments)]
fn compute_heterogeneity_decomposition(
    y: &ArrayView1<f64>,
    z: &[u8],
    x: &ArrayView2<f64>,
    tau_hat: &Array1<f64>,
    covariate_names: &[String],
    n_draws: usize,
    assign: &mut dyn FnMut(usize, &mut [u8]),
    compute_importance: bool,
    warnings: &mut Vec<String>,
) -> EconResult<HetDecomposition> {
    let (n, k) = x.dim();
    let fit = interaction_hc2(y, z, x, covariate_names, warnings)?;
    let systematic_p_value = if fit.wald.is_finite() {
        chi_squared_p_value(fit.wald, fit.df as f64)
    } else {
        f64::NAN
    };

    let s_xx = sample_cov(x);
    let b0: Vec<f64> = fit
        .beta
        .iter()
        .map(|b| if b.is_finite() { *b } else { 0.0 })
        .collect();
    let b0 = Array1::from_vec(b0);
    let systematic_variance = b0.dot(&s_xx.dot(&b0));

    let tau_mean = tau_hat.mean().unwrap_or(0.0);
    let total_variance =
        tau_hat.iter().map(|&t| (t - tau_mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;
    let idiosyncratic_variance = (total_variance - systematic_variance).max(0.0);
    let r_squared = if total_variance > 0.0 {
        (systematic_variance / total_variance).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Idiosyncratic test under tau_i = b_z + X_i' beta (aliased coefficients as 0)
    let b_z = if fit.b_z.is_finite() { fit.b_z } else { 0.0 };
    let tau_x: Vec<f64> = (0..n).map(|i| b_z + x.row(i).dot(&b0)).collect();
    let x_int = add_intercept(x);
    let n1 = z.iter().filter(|&&v| v == 1).count();
    let (idiosyncratic_test_stat, idiosyncratic_p_value) =
        match frt_idiosyncratic(y, z, &x_int, &tau_x, n_draws, assign)? {
            Some((num, p, _)) => (num as f64 / (n1 * (n - n1)) as f64, p),
            None => {
                warnings.push(format!(
                    "Idiosyncratic test not computed: each arm needs more than {} units for its \
                     regression on the covariates",
                    k + 1
                ));
                (f64::NAN, f64::NAN)
            }
        };

    let covariate_importance = if compute_importance {
        let mut imp: Vec<(usize, f64)> = (0..k)
            .map(|j| (j, b0[j].abs() * s_xx[[j, j]].sqrt()))
            .collect();
        let total: f64 = imp.iter().map(|(_, v)| v).sum();
        if total > 0.0 {
            for (_, v) in imp.iter_mut() {
                *v /= total;
            }
        }
        imp.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        imp
    } else {
        Vec::new()
    };

    Ok(HetDecomposition {
        total_variance,
        systematic_variance,
        idiosyncratic_variance,
        r_squared,
        systematic_test_stat: fit.wald,
        systematic_p_value,
        idiosyncratic_test_stat,
        idiosyncratic_p_value,
        covariate_importance,
        covariate_names: covariate_names.to_vec(),
        systematic_df: fit.df,
        systematic_coefficients: fit.beta,
        systematic_se: fit.se,
    })
}

// ===============================================================================
// Tests
// ===============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{Budget, CancelToken, with_budget};
    use ndarray::Array1;
    use polars::prelude::*;

    #[test]
    fn test_effect_estimation_method_parses_aliases_and_rejects_unknown() {
        EffectEstimationMethod::choices().validate().unwrap();
        assert_eq!(
            "Reg".parse::<EffectEstimationMethod>().unwrap(),
            EffectEstimationMethod::Regression
        );
        assert_eq!(
            "strat".parse::<EffectEstimationMethod>().unwrap(),
            EffectEstimationMethod::Stratified
        );
        assert_eq!(
            "match".parse::<EffectEstimationMethod>().unwrap(),
            EffectEstimationMethod::Matching
        );
        match "forest".parse::<EffectEstimationMethod>() {
            Err(crate::errors::EconError::InvalidSpecification { message }) => {
                assert_eq!(
                    message,
                    "unknown effect_method 'forest'; expected one of: matching, regression, stratified"
                )
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    #[test]
    fn test_het_test_stat_parses_aliases_and_rejects_unknown() {
        HetTestStat::choices().validate().unwrap();
        assert_eq!("IQR".parse::<HetTestStat>().unwrap(), HetTestStat::IQR);
        assert_eq!(
            "mean absolute deviation".parse::<HetTestStat>().unwrap(),
            HetTestStat::MeanAbsDeviation
        );
        assert_eq!("var".parse::<HetTestStat>().unwrap(), HetTestStat::Variance);
        assert_eq!("SKS".parse::<HetTestStat>().unwrap(), HetTestStat::SKS);
        match "sd".parse::<HetTestStat>() {
            Err(crate::errors::EconError::InvalidSpecification { message }) => {
                assert_eq!(
                    message,
                    "unknown test_statistic 'sd'; expected one of: variance, range, iqr, mad, sks"
                )
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_hettx_outcome_model_aliased_subset() {
        // On the treated arm x2 = 2 x1 + 1: lm reports x2 NA and predict.lm counts it as 0;
        // a generalized inverse splits its role between the columns instead.
        use crate::data::columns::{NaPolicy, numeric_column};
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("inv_hettx_outcome_model");
        let col = |c: &str| numeric_column(ds.df(), c, NaPolicy::Fail).unwrap().0;
        let (y, treat) = (Array1::from(col("y")), col("treat"));
        let x = add_intercept(
            &ndarray::stack![
                ndarray::Axis(1),
                Array1::from(col("x1")),
                Array1::from(col("x2"))
            ]
            .view(),
        );
        let treated: Vec<usize> = (0..y.len()).filter(|&i| treat[i] == 1.0).collect();
        let names = design_names(&["x1".to_string(), "x2".to_string()], 2);
        let fit = fit_outcome_model_on_subset(&x, &y.view(), &treated, &names).unwrap();
        let pred: Vec<f64> = (0..y.len()).map(|i| fit.predict(&x.row(i))).collect();
        Check::new(&case)
            .vector("coef", fit.coefficients.as_slice().unwrap(), Tol::coef(20))
            .vector("predictions", &pred, Tol::coef(40))
            .scalar("r2", fit.r2, Tol::stat(20))
            .finish();
        assert_eq!(
            fit.warning.as_deref(),
            Some("Coefficients not defined because of singularities: x2")
        );
    }

    #[test]
    fn test_hettx_unseeded_is_reproducible() {
        // Without a seed the permutations used entropy; they now use HETTX_DEFAULT_SEED.
        let (y, treatment, x) = create_hettx_test_data();
        let config = || HetTxConfig {
            n_permutations: 40,
            seed: None,
            ..Default::default()
        };
        let a = run_hettx(&y.view(), &treatment.view(), Some(&x.view()), config()).unwrap();
        let b = run_hettx(&y.view(), &treatment.view(), Some(&x.view()), config()).unwrap();
        assert_eq!(a.null_distribution, b.null_distribution);
        assert_eq!(a.p_value, b.p_value);
    }

    #[test]
    fn test_validate_hettx_effect_summary_type7() {
        use crate::data::columns::{NaPolicy, numeric_column};
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("qtl_hettx_summary");
        let tau = Array1::from(numeric_column(ds.df(), "tau", NaPolicy::Fail).unwrap().0);
        let s = compute_effect_summary(&tau);
        Check::new(&case)
            .vector(
                "quantiles",
                &[s.p10, s.p25, s.median, s.p75, s.p90],
                Tol::coef(38),
            )
            .scalar("min", s.min, Tol::exact())
            .scalar("max", s.max, Tol::exact())
            .scalar("sd", s.std_dev, Tol::stat(38))
            .scalar(
                "iqr",
                compute_test_statistic(tau.as_slice().unwrap(), HetTestStat::IQR),
                Tol::coef(38),
            )
            .finish();
    }

    fn cancelled() -> Budget {
        let token = CancelToken::new();
        token.cancel();
        Budget::new(None, token)
    }

    fn expect_invalid<T>(res: EconResult<T>, needle: &str) {
        match res {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(
                    message.contains(needle),
                    "message {message:?} should name `{needle}`"
                )
            }
            Err(e) => panic!("expected InvalidSpecification naming `{needle}`, got: {e}"),
            Ok(_) => panic!("expected InvalidSpecification naming `{needle}`, got Ok"),
        }
    }

    fn expect_resource_limit<T>(res: EconResult<T>) {
        match res {
            Err(EconError::ResourceLimit(_)) => {}
            Err(e) => panic!("expected ResourceLimit, got: {e}"),
            Ok(_) => panic!("expected ResourceLimit, got Ok"),
        }
    }

    fn expect_cancelled<T>(res: EconResult<T>) {
        match res {
            Err(EconError::Cancelled(_)) => {}
            Err(e) => panic!("expected Cancelled, got: {e}"),
            Ok(_) => panic!("expected Cancelled, got Ok"),
        }
    }

    /// Create test data with known heterogeneity structure.
    ///
    /// DGP: Y_i(1) - Y_i(0) = 0.5 + 0.3 * X1_i + noise
    /// So true effects are heterogeneous, varying with X1.
    fn create_hettx_test_data() -> (Array1<f64>, Array1<f64>, Array2<f64>) {
        let n = 100;

        // Fixed seed for reproducibility
        let mut rng = ChaCha8Rng::seed_from_u64(12345);

        // Covariates
        let x1: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let x2: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();

        // Treatment assignment (random 50%)
        let treatment: Vec<f64> = (0..n)
            .map(|_| if rng.gen_bool(0.5) { 1.0 } else { 0.0 })
            .collect();

        // Potential outcomes:
        // Y(0) = 1 + 0.2*X1 + 0.1*X2 + noise
        // Y(1) = 1.5 + 0.5*X1 + 0.1*X2 + noise
        // So tau_i = 0.5 + 0.3*X1 (heterogeneous effect!)
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let noise = rng.gen_range(-0.1..0.1);
                if treatment[i] >= 0.5 {
                    // Y(1)
                    1.5 + 0.5 * x1[i] + 0.1 * x2[i] + noise
                } else {
                    // Y(0)
                    1.0 + 0.2 * x1[i] + 0.1 * x2[i] + noise
                }
            })
            .collect();

        let y_arr = Array1::from(y);
        let treatment_arr = Array1::from(treatment);

        let mut x_arr = Array2::zeros((n, 2));
        for i in 0..n {
            x_arr[[i, 0]] = x1[i];
            x_arr[[i, 1]] = x2[i];
        }

        (y_arr, treatment_arr, x_arr)
    }

    /// Create test data with NO heterogeneity (constant effect).
    fn create_homogeneous_test_data() -> (Array1<f64>, Array1<f64>, Array2<f64>) {
        let n = 100;
        let mut rng = ChaCha8Rng::seed_from_u64(54321);

        let x1: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let x2: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let treatment: Vec<f64> = (0..n)
            .map(|_| if rng.gen_bool(0.5) { 1.0 } else { 0.0 })
            .collect();

        // Constant treatment effect of 0.5
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let noise = rng.gen_range(-0.1..0.1);
                let base = 1.0 + 0.2 * x1[i] + 0.1 * x2[i] + noise;
                if treatment[i] >= 0.5 {
                    base + 0.5 // constant effect
                } else {
                    base
                }
            })
            .collect();

        let y_arr = Array1::from(y);
        let treatment_arr = Array1::from(treatment);

        let mut x_arr = Array2::zeros((n, 2));
        for i in 0..n {
            x_arr[[i, 0]] = x1[i];
            x_arr[[i, 1]] = x2[i];
        }

        (y_arr, treatment_arr, x_arr)
    }

    #[test]
    fn test_hettx_homogeneous_effects() {
        let (y, treatment, x) = create_homogeneous_test_data();

        let config = HetTxConfig {
            n_permutations: 500,
            test_statistic: HetTestStat::Variance,
            decompose: true,
            seed: Some(42),
            ..Default::default()
        };

        let result = run_hettx(&y.view(), &treatment.view(), Some(&x.view()), config).unwrap();

        // With constant effects, p-value should be higher (fail to reject null)
        // Note: matching-based estimation adds noise, so we don't expect perfect
        assert!(
            result.p_value > 0.01,
            "P-value should be higher for homogeneous effects"
        );

        // ATE should be around 0.5
        assert!(
            (result.ate - 0.5).abs() < 0.3,
            "ATE {} should be near 0.5",
            result.ate
        );
    }

    #[test]
    fn test_hettx_different_statistics() {
        let (y, treatment, x) = create_hettx_test_data();

        for stat_type in [
            HetTestStat::Variance,
            HetTestStat::Range,
            HetTestStat::IQR,
            HetTestStat::MeanAbsDeviation,
        ] {
            let config = HetTxConfig {
                n_permutations: 100,
                test_statistic: stat_type,
                decompose: false,
                seed: Some(42),
                ..Default::default()
            };

            let result = run_hettx(&y.view(), &treatment.view(), Some(&x.view()), config).unwrap();

            assert!(
                result.test_statistic >= 0.0,
                "{:?} statistic should be non-negative",
                stat_type
            );
            assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
            assert_eq!(result.test_statistic_type, stat_type);
        }
    }

    #[test]
    fn test_hettx_regression_method() {
        let (y, treatment, x) = create_hettx_test_data();

        let config = HetTxConfig {
            n_permutations: 100,
            effect_method: EffectEstimationMethod::Regression,
            decompose: false,
            seed: Some(42),
            ..Default::default()
        };

        let result = run_hettx(&y.view(), &treatment.view(), Some(&x.view()), config).unwrap();

        assert_eq!(result.effect_method, EffectEstimationMethod::Regression);
        assert!(result.test_statistic > 0.0);
    }

    #[test]
    fn test_effect_summary_statistics() {
        let (y, treatment, x) = create_hettx_test_data();

        let config = HetTxConfig {
            n_permutations: 100,
            seed: Some(42),
            ..Default::default()
        };

        let result = run_hettx(&y.view(), &treatment.view(), Some(&x.view()), config).unwrap();

        let summary = &result.effect_summary;

        // Check ordering of percentiles
        assert!(summary.min <= summary.p10);
        assert!(summary.p10 <= summary.p25);
        assert!(summary.p25 <= summary.median);
        assert!(summary.median <= summary.p75);
        assert!(summary.p75 <= summary.p90);
        assert!(summary.p90 <= summary.max);

        // Standard deviation should be positive
        assert!(summary.std_dev >= 0.0);
    }

    #[test]
    fn test_covariate_importance() {
        let (y, treatment, x) = create_hettx_test_data();

        let config = HetTxConfig {
            n_permutations: 100,
            decompose: true,
            compute_importance: true,
            seed: Some(42),
            ..Default::default()
        };

        let result = run_hettx(&y.view(), &treatment.view(), Some(&x.view()), config).unwrap();

        let decomp = result.decomposition.unwrap();

        // Should have importance for 2 covariates
        assert_eq!(decomp.covariate_importance.len(), 2);

        // X1 should be more important than X2 (since tau varies with X1)
        let (x1_idx, x1_imp) = decomp.covariate_importance[0];
        let (_, x2_imp) = decomp.covariate_importance[1];

        // X1 is index 0, should be first (most important)
        assert_eq!(x1_idx, 0, "X1 should be the most important covariate");
        assert!(x1_imp > x2_imp, "X1 importance should exceed X2");
    }

    #[test]
    fn test_hettx_display() {
        let (y, treatment, x) = create_hettx_test_data();

        let config = HetTxConfig {
            n_permutations: 50,
            decompose: true,
            seed: Some(42),
            ..Default::default()
        };

        let result = run_hettx(&y.view(), &treatment.view(), Some(&x.view()), config).unwrap();

        // Test Display trait
        let output = format!("{}", result);
        assert!(output.contains("Treatment Effect Heterogeneity Test"));
        assert!(output.contains("H0:"));
        assert!(output.contains("P-value:"));
        assert!(output.contains("ATE"));
        assert!(output.contains("Decomposition"));
    }

    #[test]
    fn test_hettx_dataset_interface() {
        // Create a Polars DataFrame
        let n = 50;
        let mut rng = ChaCha8Rng::seed_from_u64(99999);

        let x1: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let x2: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let treatment: Vec<f64> = (0..n)
            .map(|_| if rng.gen_bool(0.5) { 1.0 } else { 0.0 })
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let noise = rng.gen_range(-0.1..0.1);
                if treatment[i] >= 0.5 {
                    1.5 + 0.5 * x1[i] + noise
                } else {
                    1.0 + 0.2 * x1[i] + noise
                }
            })
            .collect();

        let df = df! {
            "outcome" => y,
            "treatment" => treatment,
            "x1" => x1,
            "x2" => x2,
        }
        .unwrap();

        let dataset = Dataset::new(df);

        let config = HetTxConfig {
            n_permutations: 50,
            decompose: true,
            seed: Some(42),
            ..Default::default()
        };

        let result =
            run_hettx_dataset(&dataset, "outcome", "treatment", &["x1", "x2"], config).unwrap();

        assert_eq!(result.n_obs, n);
        assert!(result.decomposition.is_some());

        // Check that covariate names are preserved
        let decomp = result.decomposition.unwrap();
        assert_eq!(decomp.covariate_names.len(), 2);
    }

    #[test]
    fn test_hettx_invalid_inputs() {
        let y = Array1::from(vec![1.0, 2.0, 3.0]);
        let treatment = Array1::from(vec![1.0, 0.0]); // Wrong length

        let config = HetTxConfig::default();
        let result = run_hettx(&y.view(), &treatment.view(), None, config);

        assert!(result.is_err());
    }

    #[test]
    fn test_hettx_insufficient_data() {
        let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0]); // Only 4 observations
        let treatment = Array1::from(vec![1.0, 1.0, 0.0, 0.0]); // 2 each group

        let config = HetTxConfig::default();
        let result = run_hettx(&y.view(), &treatment.view(), None, config);

        assert!(result.is_err()); // Need at least 5 in each group
    }

    #[test]
    fn test_compute_test_statistic() {
        let tau = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        // Variance: Var([1,2,3,4,5]) = 2.5
        let var_stat = compute_test_statistic(tau.as_slice().unwrap(), HetTestStat::Variance);
        assert!((var_stat - 2.5).abs() < 0.01);

        // Range: 5 - 1 = 4
        let range_stat = compute_test_statistic(tau.as_slice().unwrap(), HetTestStat::Range);
        assert!((range_stat - 4.0).abs() < 0.01);

        // IQR: Q3 - Q1 = 4 - 2 = 2
        let iqr_stat = compute_test_statistic(tau.as_slice().unwrap(), HetTestStat::IQR);
        assert!((iqr_stat - 2.0).abs() < 0.01);

        // MAD: mean(|x - 3|) for x in [1,2,3,4,5] = (2+1+0+1+2)/5 = 1.2
        let mad_stat =
            compute_test_statistic(tau.as_slice().unwrap(), HetTestStat::MeanAbsDeviation);
        assert!((mad_stat - 1.2).abs() < 0.01);
    }

    #[test]
    fn test_hettx_rejects_bad_counts_and_data() {
        let (y, tr, x) = create_hettx_test_data();
        let run = |config: HetTxConfig| run_hettx(&y.view(), &tr.view(), Some(&x.view()), config);
        expect_invalid(
            run(HetTxConfig {
                n_permutations: 0,
                seed: Some(1),
                ..Default::default()
            }),
            "n_permutations",
        );
        expect_invalid(
            run(HetTxConfig {
                n_neighbors: 0,
                n_permutations: 10,
                seed: Some(1),
                ..Default::default()
            }),
            "n_neighbors",
        );
        let mut y_bad = y.clone();
        y_bad[0] = f64::NAN;
        expect_invalid(
            run_hettx(
                &y_bad.view(),
                &tr.view(),
                Some(&x.view()),
                HetTxConfig {
                    n_permutations: 10,
                    seed: Some(1),
                    ..Default::default()
                },
            ),
            "non-finite",
        );
        let mut x_bad = x.clone();
        x_bad[[3, 1]] = f64::INFINITY;
        expect_invalid(
            run_hettx(
                &y.view(),
                &tr.view(),
                Some(&x_bad.view()),
                HetTxConfig {
                    n_permutations: 10,
                    seed: Some(1),
                    ..Default::default()
                },
            ),
            "non-finite",
        );
    }

    #[test]
    fn test_hettx_rejects_unallocatable_permutations() {
        let (y, tr, x) = create_hettx_test_data();
        expect_resource_limit(run_hettx(
            &y.view(),
            &tr.view(),
            Some(&x.view()),
            HetTxConfig {
                n_permutations: usize::MAX / 4,
                seed: Some(1),
                ..Default::default()
            },
        ));
    }

    #[test]
    fn test_hettx_stops_when_cancelled() {
        let (y, tr, x) = create_hettx_test_data();
        expect_cancelled(with_budget(cancelled(), || {
            run_hettx(
                &y.view(),
                &tr.view(),
                Some(&x.view()),
                HetTxConfig {
                    n_permutations: 20,
                    seed: Some(1),
                    ..Default::default()
                },
            )
        }));
    }

    /// Regression imputation has no checkpoint of its own, so a cancelled budget
    /// is first seen at the top of the permutation loop. Without the
    /// decomposition, whose loop has its own checkpoint, nothing else would stop
    /// the run.
    #[test]
    fn test_hettx_permutation_loop_stops_when_cancelled() {
        let (y, tr, x) = create_hettx_test_data();
        let config = HetTxConfig {
            effect_method: EffectEstimationMethod::Regression,
            n_permutations: 20,
            decompose: false,
            seed: Some(1),
            ..Default::default()
        };
        expect_cancelled(with_budget(cancelled(), || {
            run_hettx(&y.view(), &tr.view(), Some(&x.view()), config)
        }));
    }

    // ---- T4: constant-effect test, Neyman SE, decomposition ----

    /// The hettx_dfm_t4 data: (y_het, y_homo, z, x) and the stored assignments.
    fn dfm_case(
        id: &str,
    ) -> (
        p2a_validation::Case,
        Array1<f64>,
        Array1<f64>,
        Vec<u8>,
        Array2<f64>,
        Vec<u8>,
    ) {
        use crate::data::columns::{NaPolicy, numeric_column};
        let (case, ds) = crate::test_support::load_case(id);
        let col = |c: &str| numeric_column(ds.df(), c, NaPolicy::Fail).unwrap().0;
        let z: Vec<u8> = col("treat").iter().map(|&t| t as u8).collect();
        let x = ndarray::stack![
            ndarray::Axis(1),
            Array1::from(col("x1")),
            Array1::from(col("x2"))
        ];
        let assignments: Vec<u8> = case
            .vector("assignments")
            .iter()
            .map(|&v| v as u8)
            .collect();
        (
            case,
            Array1::from(col("y_het")),
            Array1::from(col("y_homo")),
            z,
            x,
            assignments,
        )
    }

    fn sks_imputer() -> Imputer<'static> {
        Imputer {
            x: None,
            x_int: None,
            method: EffectEstimationMethod::Matching,
            n_neighbors: 3,
            names: &[],
        }
    }

    #[test]
    fn test_validate_hettx_frt_sks_stored_assignments() {
        use p2a_validation::{Check, Tol};
        let (case, y_het, y_homo, z, _, assignments) = dfm_case("hettx_dfm_t4");
        let n = z.len();
        let b_draws = case.scalar("B") as usize;
        let gamma = case.scalar("gamma");
        let g = case.scalar("grid_size") as usize;
        let mut check = Check::new(&case);
        for (out, y) in [("het", &y_het), ("homo", &y_homo)] {
            let (ate, se) = neyman(&y.view(), &z);
            let half = normal_quantile(1.0 - gamma / 2.0).unwrap() * se;
            let grid = even_grid(ate - half, ate + half, g);
            let mut stored =
                |b: usize, zb: &mut [u8]| zb.copy_from_slice(&assignments[b * n..(b + 1) * n]);
            let frt = frt_constant_effect(
                &y.view(),
                &z,
                &sks_imputer(),
                HetTestStat::SKS,
                None,
                &grid,
                b_draws,
                &mut stored,
            )
            .unwrap();
            let plug = frt_constant_effect(
                &y.view(),
                &z,
                &sks_imputer(),
                HetTestStat::SKS,
                None,
                &[ate],
                b_draws,
                &mut stored,
            )
            .unwrap();
            check
                .scalar(&format!("{out}_ate"), ate, Tol::coef(n))
                .scalar(&format!("{out}_ate_se"), se, Tol::se(n))
                .vector(
                    &format!("{out}_ci"),
                    &[ate - half, ate + half],
                    Tol::coef(n),
                )
                .vector(&format!("{out}_grid"), &grid, Tol::coef(n))
                .scalar(&format!("{out}_sks"), frt.t_obs, Tol::exact());
            assert_eq!(
                frt.sks_numerator.map(|v| v as f64),
                Some(case.scalar(&format!("{out}_sks_num"))),
                "{out}: SKS numerator"
            );
            // p-values are multiples of 1/200 (plus gamma): exact
            assert_eq!(
                frt.p_by_tau,
                case.vector(&format!("{out}_p_by_tau")),
                "{out}"
            );
            let p_max = frt.p_by_tau.iter().copied().fold(0.0, f64::max);
            assert_eq!((p_max + gamma).min(1.0), case.scalar(&format!("{out}_p")));
            assert_eq!(plug.p_by_tau, vec![case.scalar(&format!("{out}_p_plugin"))]);
            assert_eq!(frt.n_valid, b_draws);
        }
        check.finish();
    }

    #[test]
    fn test_validate_hettx_systematic_interaction_hc2() {
        use p2a_validation::{Check, Tol};
        let (case, y_het, y_homo, z, x, assignments) = dfm_case("hettx_dfm_t4");
        let n = z.len();
        let b_draws = case.scalar("B") as usize;
        let names = vec!["x1".to_string(), "x2".to_string()];
        let mut check = Check::new(&case);
        for (out, y) in [("het", &y_het), ("homo", &y_homo)] {
            let mut stored =
                |b: usize, zb: &mut [u8]| zb.copy_from_slice(&assignments[b * n..(b + 1) * n]);
            let mut warnings = Vec::new();
            let d = compute_heterogeneity_decomposition(
                &y.view(),
                &z,
                &x.view(),
                &Array1::zeros(n),
                &names,
                b_draws,
                &mut stored,
                true,
                &mut warnings,
            )
            .unwrap();
            assert!(warnings.is_empty(), "{warnings:?}");
            assert_eq!(d.systematic_df, 2);
            check
                .named(
                    &format!("{out}_sys_beta"),
                    &names,
                    &d.systematic_coefficients,
                    Tol::coef(n),
                )
                .named(
                    &format!("{out}_sys_se"),
                    &names,
                    &d.systematic_se,
                    Tol::se(n),
                )
                .scalar(
                    &format!("{out}_sys_wald"),
                    d.systematic_test_stat,
                    Tol::stat(n),
                )
                .scalar(
                    &format!("{out}_sys_p"),
                    d.systematic_p_value,
                    Tol::p_value(n),
                )
                .scalar(
                    &format!("{out}_sys_var"),
                    d.systematic_variance,
                    Tol::stat(n),
                );
            // the plug-in null of the idiosyncratic test, its statistic and p-value
            let fit = interaction_hc2(&y.view(), &z, &x.view(), &names, &mut Vec::new()).unwrap();
            let tau_x: Vec<f64> = (0..n)
                .map(|i| fit.b_z + x[[i, 0]] * fit.beta[0] + x[[i, 1]] * fit.beta[1])
                .collect();
            check.vector(&format!("{out}_idio_tau"), &tau_x, Tol::coef(n));
            assert_eq!(
                d.idiosyncratic_test_stat * (20.0 * 20.0),
                case.scalar(&format!("{out}_idio_ks_num")),
                "{out}: KS numerator"
            );
            assert_eq!(
                d.idiosyncratic_p_value,
                case.scalar(&format!("{out}_idio_p")),
                "{out}"
            );
        }
        check.finish();
    }

    /// Data with tau_i = 1 + 1.5 x1 and noise; `shift` is added to every treated outcome.
    fn het_data(n: usize, seed: u64, shift: f64) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut x = Array2::zeros((n, 2));
        let mut z: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i % 2 == 0))).collect();
        z.shuffle(&mut rng);
        let mut y = Array1::zeros(n);
        for i in 0..n {
            let x1: f64 = rng.gen_range(-1.5..1.5);
            let x2: f64 = rng.gen_range(-1.5..1.5);
            x[[i, 0]] = x1;
            x[[i, 1]] = x2;
            let noise: f64 = rng.gen_range(-1.0..1.0);
            y[i] = 0.5 * x2 + noise + z[i] * (1.0 + 1.5 * x1 + shift);
        }
        (y, Array1::from(z), x)
    }

    #[test]
    fn test_hettx_p_value_is_shift_invariant() {
        // H0 is a constant effect of unknown size: adding c to every treated outcome moves the
        // effect by c and must leave the test unchanged (the old label permutation tested a
        // zero effect, so its p-value moved with c).
        let (y, z, x) = het_data(200, 7, 0.0);
        let c = 20.0;
        let y_c = &y + &(&z * c);
        for stat in [HetTestStat::Variance, HetTestStat::SKS] {
            let mut config = HetTxConfig::default();
            config.n_permutations = 199;
            config.test_statistic = stat;
            config.decompose = false;
            config.grid_size = 21;
            config.seed = Some(7);
            let a = run_hettx(&y.view(), &z.view(), Some(&x.view()), config.clone()).unwrap();
            let b = run_hettx(&y_c.view(), &z.view(), Some(&x.view()), config).unwrap();
            assert!(
                (a.p_value - b.p_value).abs() <= 1.0 / 200.0 + 1e-12,
                "{stat}: p {} vs {}",
                a.p_value,
                b.p_value
            );
            assert!((b.ate - a.ate - c).abs() < 1e-9);
            let (ca, cb) = (a.ci.unwrap(), b.ci.unwrap());
            assert!((cb.0 - ca.0 - c).abs() < 1e-9 && (cb.1 - ca.1 - c).abs() < 1e-9);
            assert!(b.p_value > 0.0 && b.p_value <= 1.0);
        }
    }

    #[test]
    fn test_hettx_ate_se_is_neyman() {
        let y = Array1::from(vec![
            2.3, 1.1, 3.7, 0.4, 2.9, 1.8, 0.7, 1.5, 2.2, 0.9, 3.1, 1.4,
        ]);
        let z: Vec<u8> = vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let (ate, se) = neyman(&y.view(), &z);
        let arm = |a: u8| -> Vec<f64> { (0..12).filter(|&i| z[i] == a).map(|i| y[i]).collect() };
        let (t, c) = (arm(1), arm(0));
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let var = |v: &[f64]| {
            let m = mean(v);
            v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64
        };
        assert_eq!(ate, mean(&t) - mean(&c));
        assert_eq!(se, (var(&t) / 6.0 + var(&c) / 6.0).sqrt());
        let tr = Array1::from(z.iter().map(|&v| f64::from(v)).collect::<Vec<_>>());
        let x = Array2::from_shape_fn((12, 1), |(i, _)| (i as f64 * 0.37).sin());
        let mut config = HetTxConfig::default();
        config.n_permutations = 9;
        config.n_neighbors = 1;
        config.decompose = false;
        let r = run_hettx(&y.view(), &tr.view(), Some(&x.view()), config).unwrap();
        assert_eq!((r.ate, r.ate_se), (ate, se));
    }

    #[test]
    fn test_hettx_stratified_with_spread_statistic_is_refused() {
        let (y, tr, x) = create_hettx_test_data();
        for stat in [HetTestStat::Variance, HetTestStat::IQR] {
            let mut config = HetTxConfig::default();
            config.effect_method = EffectEstimationMethod::Stratified;
            config.test_statistic = stat;
            config.n_permutations = 10;
            let res = run_hettx(&y.view(), &tr.view(), Some(&x.view()), config);
            expect_invalid(res, "stratified");
        }
    }

    #[test]
    fn test_hettx_stratified_with_sks_runs() {
        let (y, tr, x) = create_hettx_test_data();
        let mut config = HetTxConfig::default();
        config.effect_method = EffectEstimationMethod::Stratified;
        config.n_permutations = 50;
        config.grid_size = 11;
        let r = run_hettx(&y.view(), &tr.view(), Some(&x.view()), config).unwrap();
        assert_eq!(r.effect_method, EffectEstimationMethod::Stratified);
        assert!(r.test_statistic > 0.0 && r.test_statistic <= 1.0);
        assert!(r.p_value > 0.0 && r.p_value <= 1.0);
        // every imputed effect is the difference in means
        assert!(r.estimated_effects.iter().all(|&t| t == r.ate));
    }

    #[test]
    fn test_hettx_dataset_text_column_is_non_numeric() {
        let mut config = HetTxConfig::default();
        config.n_permutations = 5;
        let n = 12;
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).sin()).collect();
        let t: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i % 2 == 0))).collect();
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 1.3).cos()).collect();
        let words: Vec<String> = (0..n).map(|i| format!("w{i}")).collect();
        let df = df! { "y" => &y, "t" => &t, "x" => &x, "s" => &words }.unwrap();
        let ds = Dataset::new(df);
        match run_hettx_dataset(&ds, "s", "t", &["x"], config.clone()) {
            Err(EconError::NonNumericColumn { column }) => assert_eq!(column, "s"),
            other => panic!("expected NonNumericColumn, got {other:?}"),
        }
        let y_na: Vec<Option<f64>> = (0..n).map(|i| (i != 3).then_some(y[i])).collect();
        let df = df! { "y" => &y_na, "t" => &t, "x" => &x }.unwrap();
        match run_hettx_dataset(&Dataset::new(df), "y", "t", &["x"], config.clone()) {
            Err(EconError::NullValues { column, count }) => {
                assert_eq!((column.as_str(), count), ("y", 1))
            }
            other => panic!("expected NullValues, got {other:?}"),
        }
        match run_hettx_dataset(&ds, "nope", "t", &["x"], config) {
            Err(EconError::ColumnNotFound { column, .. }) => assert_eq!(column, "nope"),
            other => panic!("expected ColumnNotFound, got {other:?}"),
        }
    }

    /// The SKS numerator by evaluating both ecdfs at every pooled point (R's definition).
    fn naive_sks(y: &[f64], z: &[u8]) -> u64 {
        let arm = |a: u8| -> Vec<f64> {
            let v: Vec<f64> = (0..y.len()).filter(|&i| z[i] == a).map(|i| y[i]).collect();
            let m = v.iter().sum::<f64>() / v.len() as f64;
            v.iter().map(|x| x - m).collect()
        };
        let (a, b) = (arm(1), arm(0));
        let (n1, n0) = (a.len() as i64, b.len() as i64);
        a.iter()
            .chain(&b)
            .map(|&u| {
                let ca = a.iter().filter(|&&v| v <= u).count() as i64;
                let cb = b.iter().filter(|&&v| v <= u).count() as i64;
                (ca * n0 - cb * n1).unsigned_abs()
            })
            .max()
            .unwrap()
    }

    #[test]
    fn test_hettx_sks_counts_match_naive() {
        // The merge over the four pre-sorted pieces against the ecdf definition, with ties
        // in the outcomes (values on a grid of quarters) and at several tau.
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        for draw in 0..50 {
            let n = 30 + draw % 7;
            let y: Vec<f64> = (0..n)
                .map(|_| f64::from(rng.gen_range(-8i32..8)) / 4.0)
                .collect();
            let mut z: Vec<u8> = (0..n).map(|i| u8::from(i < n / 2)).collect();
            z.shuffle(&mut rng);
            let mut zb = z.clone();
            zb.shuffle(&mut rng);
            let yv = Array1::from(y.clone());
            let order = outcome_order(&yv.view());
            let mut pieces = SksPieces::default();
            pieces.fill(&yv.view(), &order, &z, &zb);
            let n1 = z.iter().filter(|&&v| v == 1).count();
            for tau in [0.0, 0.5, -1.25, 0.3] {
                let yb: Vec<f64> = (0..n)
                    .map(|i| y[i] - tau * f64::from(z[i]) + tau * f64::from(zb[i]))
                    .collect();
                assert_eq!(
                    pieces.numerator(tau, n1, n - n1),
                    naive_sks(&yb, &zb),
                    "draw {draw}, tau {tau}"
                );
            }
        }
    }

    /// The k nearest donors by a full stable sort on distance (the old selection).
    fn nearest_by_sort(t: usize, x: &Array2<f64>, donors: &[usize], k: usize) -> Vec<usize> {
        let mut d: Vec<(usize, f64)> = donors
            .iter()
            .map(|&j| {
                let s: f64 = x
                    .row(t)
                    .iter()
                    .zip(x.row(j).iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                (j, s.sqrt())
            })
            .collect();
        d.sort_by(|a, b| a.1.total_cmp(&b.1));
        d.iter().take(k).map(|p| p.0).collect()
    }

    #[test]
    fn test_hettx_matching_k_nearest_matches_full_sort() {
        // Integer covariates, so many distances tie; the selection must keep the same
        // neighbours (ties by row order) as a stable sort.
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let n = 60;
        let x = Array2::from_shape_fn((n, 2), |_| f64::from(rng.gen_range(0i32..4)));
        let donors: Vec<usize> = (0..n).filter(|i| i % 3 != 0).collect();
        let mut dist = Vec::new();
        for k in [1, 3, 7] {
            for t in (0..n).filter(|i| i % 3 == 0) {
                let got: Vec<usize> = nearest(t, &x.view(), &donors, k, &mut dist)
                    .iter()
                    .map(|p| p.1)
                    .collect();
                assert_eq!(got, nearest_by_sort(t, &x, &donors, k), "t {t}, k {k}");
            }
        }
    }

    #[test]
    fn test_hettx_linear_in_tau_imputation_matches_rerun() {
        // tau_hat(tau) = imp(Y) + tau imp(Zb - Z) equals imputing Yb = Y + tau (Zb - Z) afresh.
        let (y, tr, _) = het_data(80, 3, 0.0);
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let x = Array2::from_shape_fn((80, 2), |_| f64::from(rng.gen_range(0i32..5)));
        let z: Vec<u8> = tr.iter().map(|&t| t as u8).collect();
        let mut zb = z.clone();
        zb.shuffle(&mut rng);
        let d: Vec<f64> = (0..80)
            .map(|i| f64::from(zb[i]) - f64::from(z[i]))
            .collect();
        let x_int = add_intercept(&x.view());
        let names = design_names(&[], 2);
        for method in [
            EffectEstimationMethod::Matching,
            EffectEstimationMethod::Regression,
        ] {
            let imp = Imputer {
                x: Some(x.view()),
                x_int: Some(&x_int),
                method,
                n_neighbors: 3,
                names: &names,
            };
            let (a, b) = imp
                .impute(&y.view(), Some(&d), &zb, &mut Vec::new())
                .unwrap();
            let b = b.unwrap();
            for tau in [-2.0, -0.5, 0.0, 1.3, 4.0] {
                let yb = &y + &(Array1::from(d.clone()) * tau);
                let (direct, _) = imp.impute(&yb.view(), None, &zb, &mut Vec::new()).unwrap();
                for i in 0..80 {
                    let lin = a[i] + tau * b[i];
                    assert!(
                        (lin - direct[i]).abs() <= 1e-12 * (1.0 + direct[i].abs()),
                        "{method}: tau {tau}, unit {i}: {lin} vs {}",
                        direct[i]
                    );
                }
                let mom = SpreadMoments::new(a.as_slice().unwrap(), b.as_slice().unwrap());
                let v = compute_test_statistic(direct.as_slice().unwrap(), HetTestStat::Variance);
                assert!((mom.variance(tau) - v).abs() <= 1e-10 * (1.0 + v));
            }
        }
    }

    #[test]
    fn test_hettx_detects_heterogeneity() {
        let (y, z, x) = het_data(400, 21, 0.0);
        let mut config = HetTxConfig::default();
        config.n_permutations = 199;
        config.grid_size = 31;
        config.seed = Some(42);
        let result = run_hettx(&y.view(), &z.view(), Some(&x.view()), config).unwrap();
        assert_eq!(result.n_obs, 400);
        assert_eq!(result.test_statistic_type, HetTestStat::SKS);
        assert!(result.p_value < 0.05, "SKS p = {}", result.p_value);
        let d = result.decomposition.unwrap();
        assert!(
            d.systematic_p_value < 1e-6,
            "Wald p = {}",
            d.systematic_p_value
        );
        assert_eq!(d.systematic_df, 2);
        assert!((d.systematic_coefficients[0] - 1.5).abs() < 0.4);
        // the effect is linear in x1: no idiosyncratic variation
        assert!(
            d.idiosyncratic_p_value > 0.01,
            "{}",
            d.idiosyncratic_p_value
        );
        assert!(d.r_squared >= 0.0 && d.r_squared <= 1.0);
    }

    #[test]
    fn test_hettx_idiosyncratic_test_detects_unexplained_variation() {
        // tau_i = 1 + u_i with u_i independent of X and a large spread: the covariates
        // explain none of it, so the covariate-adjusted KS test rejects.
        let n = 400;
        let mut rng = ChaCha8Rng::seed_from_u64(31);
        let mut z: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i % 2 == 0))).collect();
        z.shuffle(&mut rng);
        let x = Array2::from_shape_fn((n, 2), |_| rng.gen_range(-1.0..1.0));
        let y: Array1<f64> = (0..n)
            .map(|i| {
                let u: f64 = if rng.gen_bool(0.5) { 3.0 } else { -3.0 };
                0.3 * x[[i, 0]] + rng.gen_range(-0.5..0.5) + z[i] * (1.0 + u)
            })
            .collect();
        let mut config = HetTxConfig::default();
        config.n_permutations = 199;
        config.grid_size = 11;
        config.seed = Some(1);
        let r = run_hettx(&y.view(), &Array1::from(z).view(), Some(&x.view()), config).unwrap();
        let d = r.decomposition.clone().unwrap();
        assert!(
            d.idiosyncratic_p_value < 0.01,
            "{}",
            d.idiosyncratic_p_value
        );
        assert!(d.systematic_p_value > 0.001, "{}", d.systematic_p_value);
        let text = format!("{}", r);
        assert!(text.contains("covariate-adjusted KS"), "{text}");
        assert!(text.contains("HC2 Wald"), "{text}");
    }

    #[test]
    fn test_hettx_grid_and_plugin_options() {
        let (y, z, x) = het_data(100, 4, 0.0);
        let mut config = HetTxConfig::default();
        config.n_permutations = 49;
        config.decompose = false;
        config.grid_size = 1;
        expect_invalid(
            run_hettx(&y.view(), &z.view(), Some(&x.view()), config.clone()),
            "grid_size",
        );
        for bad in [0.0, 1.0, f64::NAN] {
            let mut c = config.clone();
            c.grid_size = 5;
            c.gamma = bad;
            expect_invalid(run_hettx(&y.view(), &z.view(), Some(&x.view()), c), "gamma");
        }
        config.plugin = true;
        let r = run_hettx(&y.view(), &z.view(), Some(&x.view()), config).unwrap();
        assert!(r.ci.is_none() && r.plugin);
        assert_eq!(r.tau_grid, vec![r.ate]);
        assert_eq!(r.p_value, r.p_by_tau[0]);
        assert_eq!(r.null_distribution.len(), 49);
    }

    /// The decomposition's randomization loop checks the budget once per draw.
    #[test]
    fn test_hettx_decomposition_stops_when_cancelled() {
        let (y, tr, x) = create_hettx_test_data();
        let z: Vec<u8> = tr.iter().map(|&t| t as u8).collect();
        let names = vec!["x1".to_string(), "x2".to_string()];
        let tau = Array1::zeros(y.len());
        let decompose = || {
            let mut assign = |b: usize, zb: &mut [u8]| draw_assignment(&z, 1, b, zb);
            compute_heterogeneity_decomposition(
                &y.view(),
                &z,
                &x.view(),
                &tau,
                &names,
                20,
                &mut assign,
                false,
                &mut Vec::new(),
            )
        };
        assert!(decompose().is_ok());
        expect_cancelled(with_budget(cancelled(), decompose));
    }
}
