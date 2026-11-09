//! Covariate Balancing Propensity Score (CBPS), as `CBPS::CBPS()` (CBPS 0.24).
//!
//! CBPS (Imai & Ratkovic 2014) estimates a logistic propensity score whose inverse-probability
//! weights balance the covariates. With `x` the design (intercept first), `t` the 0/1 treatment
//! and `p = plogis(x beta)`:
//!
//! - [`CbpsMethod::Over`] (R `method = "over"`, the default): the k score conditions
//!   `x'(t - p) / n = 0` and the k balance conditions, an over-identified two-step GMM problem
//!   `min g' W g` with CBPS's weight matrix `W = V^+` fixed at the start value
//!   (`twostep = TRUE`).
//! - [`CbpsMethod::Exact`] (R `method = "exact"`): the k balance conditions alone (just
//!   identified), so the weighted covariate means of the two groups coincide.
//! - [`CbpsMethod::Logit`]: plain logistic regression (`glm(binomial)`), no balance conditions.
//!
//! The balance target is [`CbpsEstimand::Ate`] (R `ATT = 0`: `t/p - (1-t)/(1-p)`, both groups
//! weighted to the full sample) or [`CbpsEstimand::Att`] (R `ATT = 1`: controls weighted to the
//! treated by `p/(1-p)`). The library default is ATE; `CBPS::CBPS` defaults to `ATT = 1`.
//!
//! As CBPS does, the non-intercept covariates are standardised and the design whitened by its
//! thin SVD (`U`, left singular vectors); the estimator runs on `U` and the coefficients are
//! mapped back. Start values are the logit fit scaled by `alpha`, the minimiser of
//! `g' V^+ g` over `alpha` in `[0.8, 1.1]` (Brent's method, as `optimize`). Exact balance is
//! solved by damped Newton, the over-identified problem by Gauss-Newton from the scaled logit
//! start and from the exact-balance solution (the smaller `g'Wg` wins); both stop on the
//! relative Newton step, Gauss-Newton also when `g'Wg` is at its minimum to rounding (a slowly
//! converging over-identified fit can reach that floor first). Probabilities are clipped to
//! `[1e-6, 1 - 1e-6]` (CBPS's `probs.min`).
//!
//! Reported: CBPS's `J = g'Wg` at the estimate ([`CbpsResult::cbps_j`], no factor n) and, for
//! `Over`, Hansen's test of the over-identifying restrictions on `n J` with k degrees of freedom
//! ([`CbpsResult::j_statistic`], [`CbpsResult::j_p_value`]); CBPS's standardised weights (each
//! group sums to 1); the deviance; coefficient SEs from the GMM sandwich.
//!
//! Deviations from CBPS 0.24 (details and numbers in `validation/econometrics/cbps.md`):
//! - over-identified ATT: CBPS's analytic gradient of the ATT balance block is n/n_t too large,
//!   so its BFGS stops where its objective's gradient is not zero; p2a minimises the objective.
//! - SEs: the delta method maps the whitened-scale sandwich back (CBPS.fit's transform is not
//!   scale-equivariant).
//! - alpha is Brent's minimiser at tolerance 1e-12 (CBPS: `optimize`'s default tolerance).
//!
//! Collinear covariates and treatments not coded 0/1 are refused.
//!
//! # References
//!
//! - Imai, K. & Ratkovic, M. (2014). "Covariate Balancing Propensity Score."
//!   *Journal of the Royal Statistical Society: Series B*, 76(1), 243-263.
//!   DOI: 10.1111/rssb.12027
//!
//! - R package `CBPS`: Fong, C., Ratkovic, M., & Imai, K. (2022).
//!   CBPS: Covariate Balancing Propensity Score.
//!   <https://cran.r-project.org/package=CBPS>
//!
//! R equivalent: `CBPS::CBPS(method = "over" | "exact", ATT = 0 | 1, standardize = TRUE,
//! twostep = TRUE)`

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::econ_error_from_design;
use crate::econometrics::glm::{GlmLink, GlmSpec, fit_glm, fit_propensity, plogis};
use crate::errors::{EconError, EconResult};
use crate::linalg::design::DesignMatrix;
use crate::linalg::matrix_ops::{faer_ref_to_ndarray, ndarray_to_faer};
use crate::linalg::solve::{LM_QR_TOL, Qr, SingularPolicy, sym_inverse};
use crate::traits::convergence::{Convergence, ResultNotes, notes_text};
use crate::traits::estimator::{SignificanceLevel, chi_squared_p_value, normal_p_value_two_sided};

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration Types
// ═══════════════════════════════════════════════════════════════════════════════

/// CBPS estimation method, with the R CBPS package's meanings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CbpsMethod {
    /// R `method = "over"` (R's default): the k score and k balance conditions, over-identified
    /// two-step GMM with `W = V^+` fixed at the start value (`twostep = TRUE`).
    #[default]
    #[serde(alias = "ExactBalance", alias = "OverBalance")]
    Over,
    /// R `method = "exact"`: the k balance conditions alone (just identified).
    Exact,
    /// Logistic regression (`glm(binomial)`), no balance conditions.
    #[serde(alias = "JustIdentified")]
    Logit,
}

#[allow(non_upper_case_globals)]
impl CbpsMethod {
    /// Legacy name: it always ran the score + balance GMM, R's `"over"`.
    pub const ExactBalance: CbpsMethod = CbpsMethod::Over;
    /// Legacy name of [`CbpsMethod::Over`].
    pub const OverBalance: CbpsMethod = CbpsMethod::Over;
    /// Legacy name of [`CbpsMethod::Logit`].
    pub const JustIdentified: CbpsMethod = CbpsMethod::Logit;
}

impl CbpsMethod {
    /// What the method estimates, for display.
    pub fn description(self) -> &'static str {
        match self {
            CbpsMethod::Over => "score and balance conditions, two-step GMM",
            CbpsMethod::Exact => "balance conditions only",
            CbpsMethod::Logit => "logistic regression, no balance conditions",
        }
    }
}

crate::option_enum!(
    CbpsMethod,
    crate::options::Choices {
        option: "CBPS method",
        values: &[
            (
                &[
                    "over",
                    "overidentified",
                    "over_identified",
                    "overbalance",
                    "over_balance",
                ],
                CbpsMethod::Over,
            ),
            (&["exact", "balance_only"], CbpsMethod::Exact),
            (
                &[
                    "logit",
                    "glm",
                    "just",
                    "justified",
                    "just_identified",
                    "justidentified",
                ],
                CbpsMethod::Logit,
            ),
        ],
    }
);

/// Balance target (R `ATT = 0 | 1`). The library default is ATE; `CBPS::CBPS` defaults to
/// `ATT = 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CbpsEstimand {
    /// R `ATT = 0`: both groups weighted to the full sample.
    #[default]
    Ate,
    /// R `ATT = 1`: controls weighted to the treated.
    Att,
}

crate::option_enum!(
    CbpsEstimand,
    crate::options::Choices {
        option: "CBPS estimand",
        values: &[
            (&["ATE", "average_treatment_effect"], CbpsEstimand::Ate),
            (
                &["ATT", "average_treatment_effect_on_the_treated", "treated"],
                CbpsEstimand::Att,
            ),
        ],
    }
);

/// Configuration for CBPS estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CbpsConfig {
    /// CBPS method (default [`CbpsMethod::Over`], R's default).
    pub method: CbpsMethod,
    /// Balance target (default [`CbpsEstimand::Ate`]; R's default is ATT).
    #[serde(default)]
    pub estimand: CbpsEstimand,
    /// Convergence tolerance: relative Newton / Gauss-Newton step on the whitened scale
    /// (default 1e-8).
    pub tolerance: f64,
    /// Maximum iterations of each Newton / Gauss-Newton run (default 100).
    pub max_iter: usize,
    /// Standardized difference threshold for balance (default: 0.1)
    pub balance_threshold: f64,
}

impl Default for CbpsConfig {
    fn default() -> Self {
        Self {
            method: CbpsMethod::Over,
            estimand: CbpsEstimand::Ate,
            tolerance: 1e-8,
            max_iter: 100,
            balance_threshold: 0.1,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Result Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Covariate balance statistics for a single covariate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CovariateBalance {
    /// Covariate name
    pub name: String,
    /// Mean in treated group
    pub mean_treated: f64,
    /// Mean in control group
    pub mean_control: f64,
    /// Standardized difference: (mean_treated - mean_control) / pooled_sd
    pub std_diff: f64,
    /// Variance ratio: var_treated / var_control
    pub var_ratio: f64,
    /// Whether balance is achieved (|std_diff| < threshold)
    pub balanced: bool,
}

/// Balance table showing covariate balance before and after weighting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceTable {
    /// Covariate names
    pub covariates: Vec<String>,
    /// Mean in treated group
    pub mean_treated: Vec<f64>,
    /// Mean in control group
    pub mean_control: Vec<f64>,
    /// Standardized differences
    pub std_diff: Vec<f64>,
    /// Variance ratios
    pub var_ratio: Vec<f64>,
    /// Maximum absolute standardized difference
    pub max_std_diff: f64,
    /// Number of covariates with |std_diff| < threshold
    pub n_balanced: usize,
}

impl BalanceTable {
    /// Create an empty balance table
    pub fn new() -> Self {
        Self {
            covariates: Vec::new(),
            mean_treated: Vec::new(),
            mean_control: Vec::new(),
            std_diff: Vec::new(),
            var_ratio: Vec::new(),
            max_std_diff: 0.0,
            n_balanced: 0,
        }
    }

    /// Check if all covariates are balanced (|std_diff| < threshold)
    pub fn all_balanced(&self, threshold: f64) -> bool {
        self.std_diff.iter().all(|&d| d.abs() < threshold)
    }
}

impl Default for BalanceTable {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for BalanceTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Covariate Balance Table")?;
        writeln!(
            f,
            "{:>20} {:>12} {:>12} {:>12} {:>12}",
            "Covariate", "Mean(T)", "Mean(C)", "Std.Diff", "Var.Ratio"
        )?;
        writeln!(f, "{}", "-".repeat(72))?;

        for i in 0..self.covariates.len() {
            let balanced_marker = if self.std_diff[i].abs() < 0.1 {
                " "
            } else {
                "*"
            };
            writeln!(
                f,
                "{:>20} {:>12.4} {:>12.4} {:>12.4}{} {:>12.4}",
                self.covariates[i],
                self.mean_treated[i],
                self.mean_control[i],
                self.std_diff[i],
                balanced_marker,
                self.var_ratio[i]
            )?;
        }

        writeln!(f, "{}", "-".repeat(72))?;
        writeln!(
            f,
            "Max |Std.Diff|: {:.4}  Balanced covariates: {}/{}",
            self.max_std_diff,
            self.n_balanced,
            self.covariates.len()
        )?;

        Ok(())
    }
}

/// Result from CBPS estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CbpsResult {
    /// Logistic regression coefficients (beta), on the scale of the original covariates
    pub coefficients: Vec<f64>,
    /// Coefficient names (intercept + covariates)
    pub names: Vec<String>,
    /// Standard errors: the GMM sandwich mapped from the whitened scale by the delta method
    /// (`Over`, `Exact`), or glm's model-based SEs (`Logit`)
    pub std_errors: Vec<f64>,
    /// Z-statistics
    pub z_stats: Vec<f64>,
    /// Two-sided normal p-values
    pub p_values: Vec<f64>,
    /// Estimated propensity scores P(T=1|X), clipped to `[1e-6, 1 - 1e-6]` for `Over` and
    /// `Exact` (R's `fitted.values`), glm's fitted values for `Logit`
    pub propensity_scores: Vec<f64>,
    /// CBPS's standardised weights: within each group they sum to 1 (ATE: `1/p` treated,
    /// `1/(1-p)` controls; ATT: `1/n_t` treated, `p/(1-p)` controls, normalised)
    pub weights: Vec<f64>,
    /// Covariate balance before weighting
    pub balance_before: BalanceTable,
    /// Covariate balance after CBPS weighting
    pub balance_after: BalanceTable,
    /// Whether the optimisation converged
    pub converged: bool,
    /// Iterations of the reported Newton / Gauss-Newton run (IRLS for `Logit`)
    pub iterations: usize,
    /// Final objective: `g'Wg` (`Over`), the balance loss `g_b'g_b` (`Exact`), 0 (`Logit`)
    pub gmm_criterion: f64,
    /// Hansen's statistic of the over-identifying restrictions, `n J` (`Over` only)
    pub j_statistic: Option<f64>,
    /// P-value of Hansen's test, chi-squared with `j_df` degrees of freedom (`Over` only)
    pub j_p_value: Option<f64>,
    /// Degrees of freedom of Hansen's test (k, `Over` only)
    pub j_df: Option<usize>,
    /// CBPS method used
    pub method: CbpsMethod,
    /// Number of observations
    pub n_obs: usize,
    /// Number of treated observations
    pub n_treated: usize,
    /// Number of control observations
    pub n_control: usize,
    /// Number of parameters
    pub n_params: usize,
    /// Number of moment conditions
    pub n_moments: usize,
    /// Balance target
    #[serde(default)]
    pub estimand: CbpsEstimand,
    /// CBPS's `J = g'Wg` at the estimate with W fixed at the start value, no factor n
    /// (`Over` and `Exact`; R's `J`)
    #[serde(default)]
    pub cbps_j: Option<f64>,
    /// `-2 sum[t ln p + (1 - t) ln(1 - p)]` (R's `deviance`)
    #[serde(default)]
    pub deviance: f64,
    /// CBPS's start scale alpha (`Over` and `Exact`)
    #[serde(default)]
    pub start_scale: Option<f64>,
    /// Convergence record of the reported run
    #[serde(default)]
    pub convergence: Convergence,
    /// Warnings (start values, singular matrices, non-convergence, Hansen rejection)
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl ResultNotes for CbpsResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
    fn notes_convergence(&self) -> Option<Convergence> {
        Some(self.convergence.clone())
    }
}

impl fmt::Display for CbpsResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Covariate Balancing Propensity Score (CBPS)")?;
        writeln!(f, "============================================")?;
        writeln!(f)?;
        writeln!(f, "Method: {} ({})", self.method, self.method.description())?;
        match self.estimand {
            CbpsEstimand::Ate => writeln!(f, "Estimand: ATE")?,
            CbpsEstimand::Att => writeln!(f, "Estimand: ATT (controls weighted to the treated)")?,
        }
        writeln!(
            f,
            "Observations: {} (Treated: {}, Control: {})",
            self.n_obs, self.n_treated, self.n_control
        )?;
        writeln!(
            f,
            "Parameters: {}  Moment conditions: {}",
            self.n_params, self.n_moments
        )?;
        writeln!(f)?;

        writeln!(f, "Coefficients:")?;
        writeln!(
            f,
            "{:>15} {:>12} {:>12} {:>10} {:>10}",
            "Parameter", "Estimate", "Std.Err", "z-stat", "P>|z|"
        )?;
        writeln!(f, "{}", "-".repeat(65))?;

        for i in 0..self.n_params {
            let sig = SignificanceLevel::from_p_value(self.p_values[i]);
            writeln!(
                f,
                "{:>15} {:>12.6} {:>12.6} {:>10.3} {:>10.4}{}",
                self.names[i],
                self.coefficients[i],
                self.std_errors[i],
                self.z_stats[i],
                self.p_values[i],
                sig.stars()
            )?;
        }
        writeln!(f, "{}", "-".repeat(65))?;
        writeln!(f)?;

        if let Some(j) = self.cbps_j {
            match self.method {
                CbpsMethod::Over => {
                    writeln!(f, "J-statistic (CBPS, g'Wg): {j:.8}")?;
                    if let (Some(nj), Some(p), Some(df)) =
                        (self.j_statistic, self.j_p_value, self.j_df)
                    {
                        writeln!(
                            f,
                            "Hansen test of the over-identifying restrictions: n*J = {nj:.4} (df = {df}), p-value = {p:.4}"
                        )?;
                    }
                }
                _ => writeln!(
                    f,
                    "J-statistic (CBPS, g'Wg at the exact-balance estimate; no test): {j:.8}"
                )?,
            }
        }
        writeln!(
            f,
            "Residual deviance: {:.4}  Log-likelihood: {:.4}",
            self.deviance,
            -self.deviance / 2.0
        )?;

        writeln!(f)?;
        writeln!(f, "Propensity Score Summary:")?;
        let ps_min = self
            .propensity_scores
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let ps_max = self
            .propensity_scores
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let ps_mean: f64 = self.propensity_scores.iter().sum::<f64>() / self.n_obs as f64;
        writeln!(
            f,
            "  Mean: {:.4}  Min: {:.4}  Max: {:.4}",
            ps_mean, ps_min, ps_max
        )?;

        writeln!(f)?;
        writeln!(f, "Balance Improvement:")?;
        writeln!(
            f,
            "  Before CBPS: Max |Std.Diff| = {:.4}",
            self.balance_before.max_std_diff
        )?;
        writeln!(
            f,
            "  After CBPS:  Max |Std.Diff| = {:.4}",
            self.balance_after.max_std_diff
        )?;
        writeln!(
            f,
            "  Balanced covariates: {}/{} -> {}/{}",
            self.balance_before.n_balanced,
            self.balance_before.covariates.len(),
            self.balance_after.n_balanced,
            self.balance_after.covariates.len()
        )?;

        writeln!(f)?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1")?;
        write!(f, "{}", notes_text(self))?;

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main Estimation Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// CBPS's `probs.min`: fitted probabilities are clipped to `[PROBS_MIN, 1 - PROBS_MIN]`.
const PROBS_MIN: f64 = 1e-6;
/// CBPS.fit's threshold on the singular values of the standardised design.
const SINGULAR_VALUE_MIN: f64 = 1e-5;
/// Tolerance of the Brent search for the start scale alpha.
const ALPHA_TOL: f64 = 1e-12;
/// Armijo sufficient-decrease constant of the line searches.
const ARMIJO_C: f64 = 1e-4;
/// Maximum step halvings of a line search.
const MAX_HALVINGS: usize = 30;

/// Run CBPS estimation on a dataset.
///
/// `CBPS::CBPS(treat ~ covariates, method, ATT, standardize = TRUE, twostep = TRUE)` with
/// `method` and `ATT` from [`CbpsConfig::method`] and [`CbpsConfig::estimand`] (see the module
/// documentation for the estimator and its deviations from CBPS 0.24).
///
/// # Arguments
///
/// * `dataset` - The dataset containing treatment and covariates
/// * `treatment_col` - Name of the binary treatment variable (coded 0/1)
/// * `covariate_cols` - Names of covariate columns (an intercept is added)
/// * `config` - Configuration options (optional, uses defaults if None)
///
/// # Example
///
/// ```ignore
/// use p2a_core::econometrics::cbps::{run_cbps, CbpsConfig, CbpsEstimand, CbpsMethod};
///
/// let mut config = CbpsConfig::default();
/// config.method = CbpsMethod::Exact;
/// config.estimand = CbpsEstimand::Att;
///
/// let result = run_cbps(&dataset, "treatment", &["x1", "x2", "x3"], Some(config))?;
/// println!("{}", result);
/// ```
///
/// # References
///
/// - Imai, K. & Ratkovic, M. (2014). "Covariate Balancing Propensity Score."
///   *Journal of the Royal Statistical Society: Series B*, 76(1), 243-263.
pub fn run_cbps(
    dataset: &Dataset,
    treatment_col: &str,
    covariate_cols: &[&str],
    config: Option<CbpsConfig>,
) -> EconResult<CbpsResult> {
    let config = config.unwrap_or_default();
    validate_config(&config)?;

    // Missing values are refused (MatchIt, WeightIt and tmle refuse them).
    let t = DesignMatrix::extract_column(dataset.df(), treatment_col)
        .map_err(|e| econ_error_from_design(e, dataset.df()))?;
    let n = t.len();
    if let Some((i, v)) = t.iter().enumerate().find(|(_, v)| **v != 0.0 && **v != 1.0) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "CBPS: treatment column '{treatment_col}' must be coded 0/1; row {} has {v}",
                i + 1
            ),
        });
    }

    let n_treated: usize = t.iter().filter(|&&v| v == 1.0).count();
    let n_control = n - n_treated;
    if n_treated == 0 || n_control == 0 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Treatment variable '{}' must have both treated (1) and control (0) observations. Found {} treated, {} control.",
                treatment_col, n_treated, n_control
            ),
        });
    }

    let design = DesignMatrix::from_dataframe(dataset.df(), covariate_cols, true)
        .map_err(|e| econ_error_from_design(e, dataset.df()))?;
    let x = design.data;
    let names = design.column_names;
    let k = x.ncols();
    if n <= k {
        return Err(EconError::InsufficientData {
            required: k + 1,
            provided: n,
            context: "CBPS: more observations than coefficients are needed".into(),
        });
    }
    let qr = Qr::decompose(x.view(), LM_QR_TOL)?;
    if qr.rank() < k {
        let aliased: Vec<&str> = qr.aliased().iter().map(|&j| names[j].as_str()).collect();
        return Err(EconError::InvalidSpecification {
            message: format!(
                "CBPS: the covariates are collinear (rank {} of {k}); remove: {}",
                qr.rank(),
                aliased.join(", ")
            ),
        });
    }

    let balance_before = compute_balance_table(&x, &t, &names, None, config.balance_threshold);

    let fit = match config.method {
        CbpsMethod::Logit => fit_logit(&x, &t, config.estimand)?,
        CbpsMethod::Over | CbpsMethod::Exact => fit_gmm(&x, &t, &config)?,
    };

    let balance_after =
        compute_balance_table(&x, &t, &names, Some(&fit.weights), config.balance_threshold);

    let z_stats: Vec<f64> = fit
        .coefficients
        .iter()
        .zip(&fit.std_errors)
        .map(|(&b, &se)| b / se)
        .collect();
    let p_values: Vec<f64> = z_stats
        .iter()
        .map(|&z| normal_p_value_two_sided(z))
        .collect();

    let mut warnings = fit.warnings;
    let (j_statistic, j_p_value, j_df) = match (config.method, fit.cbps_j) {
        (CbpsMethod::Over, Some(j)) => {
            let nj = n as f64 * j;
            let p = chi_squared_p_value(nj, k as f64);
            if p < 0.05 {
                warnings.push(format!(
                    "the over-identifying restrictions are rejected at 5% (Hansen p = {p:.4})"
                ));
            }
            (Some(nj), Some(p), Some(k))
        }
        _ => (None, None, None),
    };
    let n_moments = match config.method {
        CbpsMethod::Over => 2 * k,
        CbpsMethod::Exact | CbpsMethod::Logit => k,
    };

    Ok(CbpsResult {
        coefficients: fit.coefficients,
        names,
        std_errors: fit.std_errors,
        z_stats,
        p_values,
        propensity_scores: fit.propensity_scores,
        weights: fit.weights,
        balance_before,
        balance_after,
        converged: fit.convergence.converged,
        iterations: fit.convergence.iterations,
        gmm_criterion: fit.criterion,
        j_statistic,
        j_p_value,
        j_df,
        method: config.method,
        n_obs: n,
        n_treated,
        n_control,
        n_params: k,
        n_moments,
        estimand: config.estimand,
        cbps_j: fit.cbps_j,
        deviance: fit.deviance,
        start_scale: fit.start_scale,
        convergence: fit.convergence,
        warnings,
    })
}

/// Refuse settings no fit can honour.
fn validate_config(config: &CbpsConfig) -> EconResult<()> {
    let spec = |message: String| Err(EconError::InvalidSpecification { message });
    if !(config.tolerance.is_finite() && config.tolerance > 0.0) {
        return spec(format!(
            "CBPS: tolerance must be finite and > 0, got {}",
            config.tolerance
        ));
    }
    if config.max_iter < 1 {
        return spec("CBPS: max_iter must be at least 1, got 0".to_string());
    }
    if !(config.balance_threshold.is_finite() && config.balance_threshold > 0.0) {
        return spec(format!(
            "CBPS: balance_threshold must be finite and > 0, got {}",
            config.balance_threshold
        ));
    }
    Ok(())
}

/// [`run_cbps`] with the default configuration and `method` replaced.
pub fn cbps(
    dataset: &Dataset,
    treatment_col: &str,
    covariate_cols: &[&str],
    method: CbpsMethod,
) -> EconResult<CbpsResult> {
    let mut config = CbpsConfig::default();
    config.method = method;
    run_cbps(dataset, treatment_col, covariate_cols, Some(config))
}

/// What one method's fit hands to [`run_cbps`].
struct Fit {
    coefficients: Vec<f64>,
    std_errors: Vec<f64>,
    propensity_scores: Vec<f64>,
    weights: Vec<f64>,
    criterion: f64,
    cbps_j: Option<f64>,
    deviance: f64,
    start_scale: Option<f64>,
    convergence: Convergence,
    warnings: Vec<String>,
}

/// `CbpsMethod::Logit`: `glm(t ~ x, binomial)`.
fn fit_logit(x: &Array2<f64>, t: &Array1<f64>, e: CbpsEstimand) -> EconResult<Fit> {
    let pf = fit_propensity(x.view(), t.view(), GlmLink::Logit)?;
    let spec = GlmSpec::logit();
    let clipped: Vec<f64> = pf
        .scores
        .iter()
        .map(|&p| p.clamp(PROBS_MIN, 1.0 - PROBS_MIN))
        .collect();
    let estimand_weights = cbps_weights(&clipped, t, e);
    let convergence = Convergence::new(
        pf.fit.converged,
        pf.fit.iterations,
        spec.max_iter,
        "relative change in deviance",
        f64::NAN,
        spec.epsilon,
    );
    Ok(Fit {
        coefficients: pf.fit.coefficients.to_vec(),
        std_errors: pf.fit.std_errors.to_vec(),
        propensity_scores: pf.scores.to_vec(),
        weights: estimand_weights,
        criterion: 0.0,
        cbps_j: None,
        deviance: pf.fit.deviance,
        start_scale: None,
        convergence,
        warnings: pf.fit.warnings,
    })
}

/// `CbpsMethod::Over` and `CbpsMethod::Exact` (CBPS.fit + CBPS.2Treat).
fn fit_gmm(x: &Array2<f64>, t: &Array1<f64>, config: &CbpsConfig) -> EconResult<Fit> {
    let e = config.estimand;
    let (tol, max_iter) = (config.tolerance, config.max_iter);
    let wh = whiten(x)?;
    let u = &wh.u;

    let mut warnings: Vec<String> = Vec::new();
    let (b0, start_warnings) = logit_start(u, t)?;
    warnings.extend(start_warnings);
    let alpha = start_scale(u, t, &b0, e)?;
    let gmm_init = &b0 * alpha;

    let v = moment_covariance(u, t, &gmm_init, e);
    let w_inv = sym_inverse(
        v.view(),
        SingularPolicy::Generalized,
        "CBPS moment covariance V",
    )?;
    if let Some(msg) = w_inv.warning("CBPS moment covariance V") {
        warnings.push(msg);
    }
    let w = w_inv.inverse;

    let exact = solve_exact(u, t, e, &gmm_init, tol, max_iter)?;
    let (run, criterion) = match config.method {
        CbpsMethod::Exact => {
            let l = exact.objective;
            (exact, l)
        }
        _ => {
            let from_glm = solve_over(u, t, e, &w, &gmm_init, tol, max_iter)?;
            let from_bal = solve_over(u, t, e, &w, &exact.beta, tol, max_iter)?;
            let run = if from_glm.objective < from_bal.objective {
                from_glm
            } else {
                from_bal
            };
            let q = run.objective;
            (run, q)
        }
    };
    warnings.extend(run.warnings.iter().cloned());

    let m = moments(u, t, &run.beta, e);
    let cbps_j = quad_form(&m.gbar, &w);
    let vu = whitened_vcov(u, t, &run.beta, e, config.method, &w)?;
    let tm = &wh.to_original;
    let beta = tm.dot(&run.beta);
    let var = tm.dot(&vu).dot(&tm.t()) / t.len() as f64;
    let std_errors: Vec<f64> = (0..beta.len()).map(|j| var[[j, j]].sqrt()).collect();

    let p = m.p.to_vec();
    let mut deduped: Vec<String> = Vec::new();
    for msg in warnings {
        if !deduped.contains(&msg) {
            deduped.push(msg);
        }
    }
    Ok(Fit {
        coefficients: beta.to_vec(),
        std_errors,
        weights: cbps_weights(&p, t, e),
        deviance: cbps_deviance(&p, t),
        propensity_scores: p,
        criterion,
        cbps_j: Some(cbps_j),
        start_scale: Some(alpha),
        convergence: run.convergence,
        warnings: deduped,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Whitening, moments and start values
// ═══════════════════════════════════════════════════════════════════════════════

/// The whitened design of CBPS.fit and the map back to the original coefficients.
struct Whitened {
    /// Left singular vectors of the standardised design (n x k).
    u: Array2<f64>,
    /// `T = M V diag(1/d)`: original coefficients `beta = T beta_u`.
    to_original: Array2<f64>,
}

/// CBPS.fit's whitening: standardise the non-intercept columns of `x` (intercept first), take
/// the thin SVD `Xs = U diag(d) V'`.
fn whiten(x: &Array2<f64>) -> EconResult<Whitened> {
    let (n, k) = x.dim();
    let mut xs = crate::resource::try_zeros2(n, k, "CBPS standardised design")?;
    let mut m = Array2::<f64>::zeros((k, k));
    m[[0, 0]] = 1.0;
    for i in 0..n {
        xs[[i, 0]] = 1.0;
    }
    for j in 1..k {
        let col = x.column(j);
        let mean = col.sum() / n as f64;
        let ss: f64 = col.iter().map(|v| (v - mean) * (v - mean)).sum();
        let sd = (ss / (n as f64 - 1.0)).sqrt();
        for i in 0..n {
            xs[[i, j]] = (x[[i, j]] - mean) / sd;
        }
        m[[0, j]] = -mean / sd;
        m[[j, j]] = 1.0 / sd;
    }
    let svd = ndarray_to_faer(&xs.view()).thin_svd().map_err(|e| {
        EconError::Computation(format!(
            "CBPS: SVD of the standardised design failed: {e:?}"
        ))
    })?;
    let u = faer_ref_to_ndarray(svd.U());
    let v = faer_ref_to_ndarray(svd.V());
    let s = svd.S().column_vector();
    let d: Array1<f64> = (0..k).map(|j| s[j]).collect();
    let d_min = d.iter().copied().fold(f64::INFINITY, f64::min);
    if d_min.is_nan() || d_min <= SINGULAR_VALUE_MIN {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "CBPS: the standardised covariates are numerically collinear (smallest singular value {d_min:.3e} <= 1e-5, CBPS.fit's threshold)"
            ),
        });
    }
    let mut vd = v;
    for j in 0..k {
        let inv = 1.0 / d[j];
        vd.column_mut(j).mapv_inplace(|a| a * inv);
    }
    Ok(Whitened {
        u,
        to_original: m.dot(&vd),
    })
}

/// The CBPS moments at `beta` (whitened scale) and their Jacobian.
struct Moments {
    /// Clipped probabilities.
    p: Array1<f64>,
    /// `[g_score; g_balance]`, length 2k.
    gbar: Array1<f64>,
    /// d gbar / d beta, 2k x k.
    jacobian: Array2<f64>,
}

/// Probabilities, their derivatives and the balance weights at `beta`.
struct Pointwise {
    p: Array1<f64>,
    /// d p / d theta of the clipped probability (0 where clipped).
    dp: Array1<f64>,
    /// Balance weights `w_i`.
    w: Array1<f64>,
}

fn pointwise(u: &Array2<f64>, t: &Array1<f64>, beta: &Array1<f64>, e: CbpsEstimand) -> Pointwise {
    let n = t.len();
    let theta = u.dot(beta);
    let mut p = Array1::<f64>::zeros(n);
    let mut dp = Array1::<f64>::zeros(n);
    let mut w = Array1::<f64>::zeros(n);
    let s = att_scale(t);
    for i in 0..n {
        let q = plogis(theta[i]);
        let inside = q > PROBS_MIN && q < 1.0 - PROBS_MIN;
        let pi = q.clamp(PROBS_MIN, 1.0 - PROBS_MIN);
        p[i] = pi;
        dp[i] = if inside { pi * (1.0 - pi) } else { 0.0 };
        w[i] = match e {
            CbpsEstimand::Ate => 1.0 / (pi - 1.0 + t[i]),
            CbpsEstimand::Att => s * (t[i] - pi) / (1.0 - pi),
        };
    }
    Pointwise { p, dp, w }
}

/// `n / n_t`, the ATT balance scale.
fn att_scale(t: &Array1<f64>) -> f64 {
    let n_t = t.iter().filter(|&&v| v == 1.0).count();
    t.len() as f64 / n_t as f64
}

/// `U' diag(h) U / n` (k x k, symmetric).
fn weighted_gram(u: &Array2<f64>, h: &Array1<f64>) -> Array2<f64> {
    let (n, k) = u.dim();
    let mut g = Array2::<f64>::zeros((k, k));
    for i in 0..n {
        let hi = h[i];
        if hi == 0.0 {
            continue;
        }
        let row = u.row(i);
        for a in 0..k {
            let ua = hi * row[a];
            for b in a..k {
                g[[a, b]] += ua * row[b];
            }
        }
    }
    for a in 0..k {
        for b in a..k {
            let v = g[[a, b]] / n as f64;
            g[[a, b]] = v;
            g[[b, a]] = v;
        }
    }
    g
}

/// `[U'(t - p); U'w] / n`.
fn moment_vector(u: &Array2<f64>, t: &Array1<f64>, pw: &Pointwise) -> Array1<f64> {
    let (n, k) = u.dim();
    let resid = t - &pw.p;
    let gs = u.t().dot(&resid) / n as f64;
    let gb = u.t().dot(&pw.w) / n as f64;
    let mut g = Array1::<f64>::zeros(2 * k);
    g.slice_mut(ndarray::s![..k]).assign(&gs);
    g.slice_mut(ndarray::s![k..]).assign(&gb);
    g
}

/// The 2k CBPS moments and their exact Jacobian (CAUS-11).
fn moments(u: &Array2<f64>, t: &Array1<f64>, beta: &Array1<f64>, e: CbpsEstimand) -> Moments {
    let k = u.ncols();
    let pw = pointwise(u, t, beta, e);
    let gbar = moment_vector(u, t, &pw);
    let s = att_scale(t);
    let dw: Array1<f64> = (0..t.len())
        .map(|i| {
            let (p, dp) = (pw.p[i], pw.dp[i]);
            match e {
                CbpsEstimand::Ate => -dp / ((p - 1.0 + t[i]) * (p - 1.0 + t[i])),
                CbpsEstimand::Att => {
                    if t[i] == 1.0 {
                        0.0
                    } else {
                        -s * dp / ((1.0 - p) * (1.0 - p))
                    }
                }
            }
        })
        .collect();
    let js = weighted_gram(u, &pw.dp);
    let jb = weighted_gram(u, &dw);
    let mut jacobian = Array2::<f64>::zeros((2 * k, k));
    jacobian
        .slice_mut(ndarray::s![..k, ..])
        .assign(&js.mapv(|v| -v));
    jacobian.slice_mut(ndarray::s![k.., ..]).assign(&jb);
    Moments {
        p: pw.p,
        gbar,
        jacobian,
    }
}

/// CBPS's model-based covariance of the moments (`gmm.func` with `invV = NULL`), 2k x 2k.
fn moment_covariance(
    u: &Array2<f64>,
    t: &Array1<f64>,
    beta: &Array1<f64>,
    e: CbpsEstimand,
) -> Array2<f64> {
    let k = u.ncols();
    let pw = pointwise(u, t, beta, e);
    let p = &pw.p;
    let a = p.mapv(|pi| pi * (1.0 - pi));
    let (c, ee, s) = match e {
        CbpsEstimand::Ate => (
            Array1::<f64>::ones(p.len()),
            p.mapv(|pi| 1.0 / (pi * (1.0 - pi))),
            1.0,
        ),
        CbpsEstimand::Att => (p.clone(), p.mapv(|pi| pi / (1.0 - pi)), att_scale(t)),
    };
    let ba = weighted_gram(u, &a);
    let bc = weighted_gram(u, &c) * s;
    let be = weighted_gram(u, &ee) * (s * s);
    let mut v = Array2::<f64>::zeros((2 * k, 2 * k));
    v.slice_mut(ndarray::s![..k, ..k]).assign(&ba);
    v.slice_mut(ndarray::s![..k, k..]).assign(&bc);
    v.slice_mut(ndarray::s![k.., ..k]).assign(&bc);
    v.slice_mut(ndarray::s![k.., k..]).assign(&be);
    v
}

/// `g' W g`.
fn quad_form(g: &Array1<f64>, w: &Array2<f64>) -> f64 {
    g.dot(&w.dot(g))
}

/// CBPS's start-scale objective `f(alpha) = g(b)' V(b)^+ g(b)`, `b = alpha b0`.
fn start_objective(
    u: &Array2<f64>,
    t: &Array1<f64>,
    b0: &Array1<f64>,
    alpha: f64,
    e: CbpsEstimand,
) -> EconResult<f64> {
    let b = b0 * alpha;
    let pw = pointwise(u, t, &b, e);
    let g = moment_vector(u, t, &pw);
    let v = moment_covariance(u, t, &b, e);
    let vi = sym_inverse(
        v.view(),
        SingularPolicy::Generalized,
        "CBPS moment covariance V",
    )?;
    Ok(quad_form(&g, &vi.inverse))
}

/// Brent's minimiser on `[ax, bx]` (R's `Brent_fmin`, the algorithm of `stats::optimize`).
fn brent_fmin<F: FnMut(f64) -> EconResult<f64>>(
    ax: f64,
    bx: f64,
    tol: f64,
    mut f: F,
) -> EconResult<f64> {
    let c = (3.0 - 5.0_f64.sqrt()) * 0.5;
    let eps = f64::EPSILON.sqrt();
    let (mut a, mut b) = (ax, bx);
    let mut v = a + c * (b - a);
    let (mut w, mut x) = (v, v);
    let (mut d, mut e) = (0.0_f64, 0.0_f64);
    let mut fx = f(x)?;
    let (mut fv, mut fw) = (fx, fx);
    let tol3 = tol / 3.0;
    // R has no cap; Brent ends long before.
    for _ in 0..1000 {
        crate::resource::checkpoint()?;
        let xm = (a + b) * 0.5;
        let tol1 = eps * x.abs() + tol3;
        let t2 = tol1 * 2.0;
        if (x - xm).abs() <= t2 - (b - a) * 0.5 {
            break;
        }
        let (mut p, mut q, mut r) = (0.0_f64, 0.0_f64, 0.0_f64);
        if e.abs() > tol1 {
            // fit parabola
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = (q - r) * 2.0;
            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }
            r = e;
            e = d;
        }
        if p.abs() >= (q * 0.5 * r).abs() || p <= q * (a - x) || p >= q * (b - x) {
            // golden-section step
            e = if x < xm { b - x } else { a - x };
            d = c * e;
        } else {
            // parabolic-interpolation step
            d = p / q;
            let u = x + d;
            if u - a < t2 || b - u < t2 {
                d = tol1;
                if x >= xm {
                    d = -d;
                }
            }
        }
        let u = if d.abs() >= tol1 {
            x + d
        } else if d > 0.0 {
            x + tol1
        } else {
            x - tol1
        };
        let fu = f(u)?;
        if fu <= fx {
            if u < x {
                b = x;
            } else {
                a = x;
            }
            v = w;
            w = x;
            x = u;
            fv = fw;
            fw = fx;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }
    Ok(x)
}

/// CBPS's start scale: the minimiser of [`start_objective`] over `alpha` in `[0.8, 1.1]`.
fn start_scale(
    u: &Array2<f64>,
    t: &Array1<f64>,
    b0: &Array1<f64>,
    e: CbpsEstimand,
) -> EconResult<f64> {
    brent_fmin(0.8, 1.1, ALPHA_TOL, |a| start_objective(u, t, b0, a, e))
}

/// CBPS's glm start `glm(treat ~ U - 1, binomial)` at `glm.control`'s defaults; aliased
/// (NaN) coefficients become 0, as in CBPS.
fn logit_start(u: &Array2<f64>, t: &Array1<f64>) -> EconResult<(Array1<f64>, Vec<String>)> {
    let mut spec = GlmSpec::logit();
    spec.epsilon = 1e-8;
    spec.max_iter = 25;
    let fit = fit_glm(u.view(), t.view(), &spec)?;
    let b0 = fit.coefficients.mapv(|b| if b.is_nan() { 0.0 } else { b });
    let warnings = fit
        .warnings
        .iter()
        .map(|w| format!("CBPS start values: {w}"))
        .collect();
    Ok((b0, warnings))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Solvers
// ═══════════════════════════════════════════════════════════════════════════════

/// One Newton / Gauss-Newton run.
struct Run {
    beta: Array1<f64>,
    objective: f64,
    convergence: Convergence,
    warnings: Vec<String>,
}

/// Gauss-Newton stops once its predicted decrease of `g'Wg` is below this fraction of `g'Wg`
/// (the objective is then at its minimum to rounding).
const DECREMENT_RTOL: f64 = 1e-14;
/// When no step lowers `g'Wg` (its rounding noise exceeds the predicted decrease), the
/// Gauss-Newton run counts as converged if the predicted decrease is below this fraction of
/// `g'Wg` (R's `optim` stops at a relative objective change of 1e-8).
const FLOOR_RTOL: f64 = 1e-10;

const STEP_CRITERION: &str = "relative change in coefficients (whitened scale)";

/// `max|s| / max|beta|`.
fn step_change(s: &Array1<f64>, beta: &Array1<f64>) -> f64 {
    let num = s.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let den = beta.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    num / den.max(1e-300)
}

/// Balance loss `g_b' g_b` (CBPS's `bal.loss` on the whitened design, `(U'U)^+ = I`).
fn balance_loss(u: &Array2<f64>, t: &Array1<f64>, beta: &Array1<f64>, e: CbpsEstimand) -> f64 {
    let n = t.len() as f64;
    let pw = pointwise(u, t, beta, e);
    let gb = u.t().dot(&pw.w) / n;
    gb.dot(&gb)
}

/// Exact balance (CBPS `opt.bal`): damped Newton on the k balance conditions.
fn solve_exact(
    u: &Array2<f64>,
    t: &Array1<f64>,
    e: CbpsEstimand,
    start: &Array1<f64>,
    tol: f64,
    max_iter: usize,
) -> EconResult<Run> {
    let k = u.ncols();
    let mut beta = start.clone();
    let mut warnings: Vec<String> = Vec::new();
    let mut converged = false;
    let mut change = f64::NAN;
    let mut it = 0;
    while it < max_iter {
        it += 1;
        crate::resource::checkpoint()?;
        let m = moments(u, t, &beta, e);
        let gb = m.gbar.slice(ndarray::s![k..]).to_owned();
        let neg_jb = m.jacobian.slice(ndarray::s![k.., ..]).mapv(|v| -v);
        let inv = sym_inverse(
            neg_jb.view(),
            SingularPolicy::Generalized,
            "CBPS balance Jacobian",
        )?;
        if let Some(msg) = inv.warning("CBPS balance Jacobian")
            && !warnings.contains(&msg)
        {
            warnings.push(msg);
        }
        let s = inv.inverse.dot(&gb).mapv(|v| -v);
        change = step_change(&s, &beta);
        if change <= tol {
            beta = &beta - &s;
            converged = true;
            break;
        }
        let l0 = gb.dot(&gb);
        let mut step = 1.0;
        let mut accepted = false;
        for _ in 0..=MAX_HALVINGS {
            let cand = &beta - &(&s * step);
            let l = balance_loss(u, t, &cand, e);
            if l.is_finite() && l < l0 && l <= (1.0 - 2.0 * ARMIJO_C * step) * l0 {
                beta = cand;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            warnings.push(format!(
                "CBPS (exact): the line search found no decrease of the balance loss at iteration {it}"
            ));
            break;
        }
    }
    let convergence = Convergence::new(converged, it, max_iter, STEP_CRITERION, change, tol);
    if let Some(msg) = convergence.note("CBPS (exact)") {
        warnings.push(msg);
    }
    Ok(Run {
        objective: balance_loss(u, t, &beta, e),
        beta,
        convergence,
        warnings,
    })
}

/// Over-identified CBPS (`gmm.loss` with W fixed): Gauss-Newton on `g' W g`.
fn solve_over(
    u: &Array2<f64>,
    t: &Array1<f64>,
    e: CbpsEstimand,
    w: &Array2<f64>,
    start: &Array1<f64>,
    tol: f64,
    max_iter: usize,
) -> EconResult<Run> {
    let mut beta = start.clone();
    let mut warnings: Vec<String> = Vec::new();
    let mut converged = false;
    let mut change = f64::NAN;
    let mut it = 0;
    let objective = |b: &Array1<f64>| {
        let pw = pointwise(u, t, b, e);
        quad_form(&moment_vector(u, t, &pw), w)
    };
    while it < max_iter {
        it += 1;
        crate::resource::checkpoint()?;
        let m = moments(u, t, &beta, e);
        let wg = w.dot(&m.gbar);
        let r = m.jacobian.t().dot(&wg);
        let h = m.jacobian.t().dot(&w.dot(&m.jacobian));
        let h = symmetrize(h);
        let inv = sym_inverse(
            h.view(),
            SingularPolicy::Generalized,
            "CBPS Gauss-Newton matrix J'WJ",
        )?;
        if let Some(msg) = inv.warning("CBPS Gauss-Newton matrix J'WJ")
            && !warnings.contains(&msg)
        {
            warnings.push(msg);
        }
        let s = inv.inverse.dot(&r);
        change = step_change(&s, &beta);
        let q0 = m.gbar.dot(&wg);
        let slope = r.dot(&s);
        if change <= tol || slope <= DECREMENT_RTOL * q0 {
            beta = &beta - &s;
            converged = true;
            break;
        }
        let mut step = 1.0;
        let mut accepted = false;
        for _ in 0..=MAX_HALVINGS {
            let cand = &beta - &(&s * step);
            let q = objective(&cand);
            if q.is_finite() && q < q0 && q <= q0 - 2.0 * ARMIJO_C * step * slope {
                beta = cand;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            if slope <= FLOOR_RTOL * q0 {
                // g'Wg cannot be lowered in floating point and its predicted decrease is
                // negligible: the minimum to the precision g'Wg can be evaluated.
                converged = true;
                break;
            }
            warnings.push(format!(
                "CBPS (over): the line search found no decrease of g'Wg at iteration {it}"
            ));
            break;
        }
    }
    let convergence = Convergence::new(converged, it, max_iter, STEP_CRITERION, change, tol);
    if let Some(msg) = convergence.note("CBPS (over)") {
        warnings.push(msg);
    }
    Ok(Run {
        objective: objective(&beta),
        beta,
        convergence,
        warnings,
    })
}

/// `(m + m') / 2`.
fn symmetrize(m: Array2<f64>) -> Array2<f64> {
    let mt = m.t().to_owned();
    (m + mt) * 0.5
}

/// CBPS.2Treat's sandwich on the whitened scale (n times the covariance of `beta_u`).
fn whitened_vcov(
    u: &Array2<f64>,
    t: &Array1<f64>,
    beta: &Array1<f64>,
    e: CbpsEstimand,
    method: CbpsMethod,
    w: &Array2<f64>,
) -> EconResult<Array2<f64>> {
    let k = u.ncols();
    let m = moments(u, t, beta, e);
    let pw = pointwise(u, t, beta, e);
    match method {
        CbpsMethod::Exact => {
            let neg_jb = m.jacobian.slice(ndarray::s![k.., ..]).mapv(|v| -v);
            let inv = sym_inverse(
                neg_jb.view(),
                SingularPolicy::Error,
                "CBPS balance Jacobian",
            )?;
            let a = inv.inverse.mapv(|v| -v);
            let omega = weighted_gram(u, &pw.w.mapv(|v| v * v));
            Ok(symmetrize(a.dot(&omega).dot(&a.t())))
        }
        _ => {
            let h = symmetrize(m.jacobian.t().dot(&w.dot(&m.jacobian)));
            let inv = sym_inverse(h.view(), SingularPolicy::Error, "CBPS sandwich J'WJ")?;
            let a = inv.inverse.dot(&m.jacobian.t()).dot(w);
            // Omega = sum_i m_i m_i' / n with m_i = [u_i (t_i - p_i); u_i w_i].
            let r = t - &pw.p;
            let oss = weighted_gram(u, &r.mapv(|v| v * v));
            let osb = weighted_gram(u, &(&r * &pw.w));
            let obb = weighted_gram(u, &pw.w.mapv(|v| v * v));
            let mut omega = Array2::<f64>::zeros((2 * k, 2 * k));
            omega.slice_mut(ndarray::s![..k, ..k]).assign(&oss);
            omega.slice_mut(ndarray::s![..k, k..]).assign(&osb);
            omega.slice_mut(ndarray::s![k.., ..k]).assign(&osb);
            omega.slice_mut(ndarray::s![k.., k..]).assign(&obb);
            Ok(symmetrize(a.dot(&omega).dot(&a.t())))
        }
    }
}

/// CBPS's standardised weights (`standardize = TRUE`) from fitted probabilities; within each
/// group they sum to 1.
fn cbps_weights(p: &[f64], t: &Array1<f64>, e: CbpsEstimand) -> Vec<f64> {
    let raw: Vec<f64> = p
        .iter()
        .zip(t.iter())
        .map(|(&pi, &ti)| match e {
            CbpsEstimand::Ate => {
                if ti == 1.0 {
                    1.0 / pi
                } else {
                    1.0 / (1.0 - pi)
                }
            }
            CbpsEstimand::Att => {
                if ti == 1.0 {
                    1.0
                } else {
                    pi / (1.0 - pi)
                }
            }
        })
        .collect();
    let (mut st, mut sc) = (0.0, 0.0);
    for (r, &ti) in raw.iter().zip(t.iter()) {
        if ti == 1.0 {
            st += r;
        } else {
            sc += r;
        }
    }
    raw.iter()
        .zip(t.iter())
        .map(|(r, &ti)| if ti == 1.0 { r / st } else { r / sc })
        .collect()
}

/// `-2 sum[t ln p + (1 - t) ln(1 - p)]`.
fn cbps_deviance(p: &[f64], t: &Array1<f64>) -> f64 {
    -2.0 * p
        .iter()
        .zip(t.iter())
        .map(|(&pi, &ti)| ti * pi.ln() + (1.0 - ti) * (1.0 - pi).ln())
        .sum::<f64>()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Balance and Weight Computation
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute covariate balance table.
///
/// Computes standardized differences and variance ratios for each covariate.
fn compute_balance_table(
    x: &Array2<f64>,
    t: &Array1<f64>,
    names: &[String],
    weights: Option<&[f64]>,
    threshold: f64,
) -> BalanceTable {
    let n = t.len();
    let k = x.ncols();

    let mut table = BalanceTable::new();

    // Skip intercept column (usually first)
    let start_col = if names.first().map(|s| s.as_str()) == Some("(Intercept)") {
        1
    } else {
        0
    };

    for j in start_col..k {
        let name = names[j].clone();

        // Extract values for treated and control groups
        let mut treated_vals: Vec<f64> = Vec::new();
        let mut treated_weights: Vec<f64> = Vec::new();
        let mut control_vals: Vec<f64> = Vec::new();
        let mut control_weights: Vec<f64> = Vec::new();

        for i in 0..n {
            let w = weights.map(|ws| ws[i]).unwrap_or(1.0);
            if t[i] >= 0.5 {
                treated_vals.push(x[[i, j]]);
                treated_weights.push(w);
            } else {
                control_vals.push(x[[i, j]]);
                control_weights.push(w);
            }
        }

        // Compute weighted means
        let mean_treated = weighted_mean(&treated_vals, &treated_weights);
        let mean_control = weighted_mean(&control_vals, &control_weights);

        // Compute weighted variances
        let var_treated = weighted_variance(&treated_vals, &treated_weights, mean_treated);
        let var_control = weighted_variance(&control_vals, &control_weights, mean_control);

        // Pooled standard deviation (using unweighted for standardization)
        let pooled_var = (var_treated + var_control) / 2.0;
        let pooled_sd = pooled_var.sqrt().max(1e-10);

        // Standardized difference
        let std_diff = (mean_treated - mean_control) / pooled_sd;

        // Variance ratio
        let var_ratio = if var_control > 1e-10 {
            var_treated / var_control
        } else {
            f64::NAN
        };

        let balanced = std_diff.abs() < threshold;

        table.covariates.push(name);
        table.mean_treated.push(mean_treated);
        table.mean_control.push(mean_control);
        table.std_diff.push(std_diff);
        table.var_ratio.push(var_ratio);

        if balanced {
            table.n_balanced += 1;
        }
    }

    // Update max standardized difference
    table.max_std_diff = table
        .std_diff
        .iter()
        .map(|&d| d.abs())
        .fold(0.0_f64, f64::max);

    table
}

/// Compute weighted mean.
fn weighted_mean(values: &[f64], weights: &[f64]) -> f64 {
    let total_weight: f64 = weights.iter().sum();
    if total_weight == 0.0 {
        return 0.0;
    }
    let weighted_sum: f64 = values
        .iter()
        .zip(weights.iter())
        .map(|(&v, &w)| v * w)
        .sum();
    weighted_sum / total_weight
}

/// Compute weighted variance.
fn weighted_variance(values: &[f64], weights: &[f64], mean: f64) -> f64 {
    let total_weight: f64 = weights.iter().sum();
    if total_weight == 0.0 {
        return 0.0;
    }
    let weighted_sq_diff: f64 = values
        .iter()
        .zip(weights.iter())
        .map(|(&v, &w)| w * (v - mean).powi(2))
        .sum();
    weighted_sq_diff / total_weight
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DataLoader;
    use CbpsEstimand::{Ate, Att};
    use CbpsMethod::{Exact, Logit, Over};
    use p2a_validation::{Case, Check, Tol};
    use polars::prelude::*;

    const COVS: [&str; 3] = ["x1", "x2", "x3"];

    /// `cbps_sim` read directly (for the property tests, which pin no R value and so do not
    /// load the case).
    fn sim() -> Dataset {
        let path = p2a_validation::repo_path("validation/datasets/generated/cbps_sim.csv");
        Dataset::new(DataLoader::load_csv(&path).unwrap())
    }

    fn fit_with(ds: &Dataset, method: CbpsMethod, estimand: CbpsEstimand) -> CbpsResult {
        let mut config = CbpsConfig::default();
        config.method = method;
        config.estimand = estimand;
        run_cbps(ds, "treat", &COVS, Some(config)).unwrap()
    }
    fn column(ds: &Dataset, name: &str) -> Vec<f64> {
        DesignMatrix::extract_column(ds.df(), name)
            .unwrap()
            .to_vec()
    }
    fn treated_weight_sum(ds: &Dataset, r: &CbpsResult) -> f64 {
        column(ds, "treat")
            .iter()
            .zip(&r.weights)
            .filter(|(t, _)| **t == 1.0)
            .map(|(_, w)| *w)
            .sum()
    }
    struct Tols {
        coef: Tol,
        se: Tol,
        stat: Tol,
    }
    /// `<prefix>_tight_*` of cbps_sim_methods (patched CBPS.2Treat, validation/r/cases/causal_cbps.R).
    fn check_tight(
        case: &Case,
        ds: &Dataset,
        r: &CbpsResult,
        prefix: &str,
        alpha_key: &str,
        tol: Tols,
    ) {
        let key = |q: &str| format!("{prefix}_tight_{q}");
        let mut check = Check::new(case);
        check
            .named(&key("coef"), &r.names, &r.coefficients, tol.coef)
            .named(&key("se"), &r.names, &r.std_errors, tol.se)
            // Tol: J depends on alpha through W; CBPS's start objective carries about 1e-12
            // relative rounding noise, so Brent's alpha is within about 1e-7 of the reference's
            // refined alpha, and J moves by about 3 times that.
            .scalar(&key("J"), r.cbps_j.unwrap(), Tol::iterative(5e-6))
            .vector(&key("ps_head"), &r.propensity_scores[..10], tol.coef)
            .vector(&key("weights_head"), &r.weights[..10], tol.coef)
            .scalar(
                &key("weights_sum_treated"),
                treated_weight_sum(ds, r),
                Tol::coef(r.n_obs),
            )
            .scalar(&key("deviance"), r.deviance, tol.stat)
            // Tol: the reference refines alpha by bisection on f'(alpha); p2a takes Brent's
            // minimiser at tol 1e-12, within about 1e-7 (rounding noise of the objective).
            .scalar(alpha_key, r.start_scale.unwrap(), Tol::iterative(1e-6));
        if r.method == Over {
            check.scalar(&key("hansen_p"), r.j_p_value.unwrap(), tol.stat);
            assert_eq!(r.j_statistic, Some(r.n_obs as f64 * r.cbps_j.unwrap()));
        }
        check.finish();
    }
    /// GMM: p2a's Gauss-Newton (relative step 1e-8) and R's BFGS (reltol 1e-15) agree to 2e-8
    /// (coef) and 9e-8 (SE).
    fn over_tols() -> Tols {
        Tols {
            coef: Tol::iterative(1e-6),
            se: Tol::iterative(1e-6),
            stat: Tol::iterative(1e-6),
        }
    }
    fn exact_tols(n: usize) -> Tols {
        Tols {
            coef: Tol::coef(n),
            se: Tol::se(n),
            stat: Tol::stat(n),
        }
    }
    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    /// CAUS-11: before, the score + balance GMM diverged (intercept 9.7e11 after step 1 on
    /// this sample).
    #[test]
    fn test_validate_cbps_over_ate() {
        let (case, ds) = crate::test_support::load_case("cbps_sim_methods");
        let r = fit_with(&ds, Over, Ate);
        assert!(r.converged, "{r}");
        assert_eq!((r.n_moments, r.j_df), (8, Some(4)));
        check_tight(&case, &ds, &r, "over_ate", "ate_tight_alpha", over_tols());
    }

    #[test]
    fn test_validate_cbps_over_att() {
        let (case, ds) = crate::test_support::load_case("cbps_sim_methods");
        let r = fit_with(&ds, Over, Att);
        assert!(r.converged, "{r}");
        check_tight(&case, &ds, &r, "over_att", "att_tight_alpha", over_tols());
        // deviation CAUS-11: CBPS 0.24 stops at a non-stationary point (J 0.0024947)
        assert!(r.cbps_j.unwrap() < 0.75 * case.scalar("over_att_J"));
    }

    /// CAUS-55: exact balance could not be requested before.
    #[test]
    fn test_validate_cbps_exact_ate() {
        let (case, ds) = crate::test_support::load_case("cbps_sim_methods");
        let r = fit_with(&ds, Exact, Ate);
        let n = r.n_obs;
        assert!(r.converged, "{r}");
        assert_eq!(
            (r.n_moments, r.j_df, r.j_p_value, r.j_statistic),
            (4, None, None, None)
        );
        check_tight(
            &case,
            &ds,
            &r,
            "exact_ate",
            "ate_tight_alpha",
            exact_tols(n),
        );
    }

    #[test]
    fn test_validate_cbps_exact_att() {
        let (case, ds) = crate::test_support::load_case("cbps_sim_methods");
        let r = fit_with(&ds, Exact, Att);
        let n = r.n_obs;
        assert!(r.converged, "{r}");
        assert_eq!(
            (r.n_moments, r.j_df, r.j_p_value, r.j_statistic),
            (4, None, None, None)
        );
        check_tight(
            &case,
            &ds,
            &r,
            "exact_att",
            "att_tight_alpha",
            exact_tols(n),
        );
    }

    /// Logit: glm(binomial) coefficients, model-based SEs (the old code reported a score
    /// sandwich), deviance and the weights of both estimands.
    #[test]
    fn test_validate_cbps_logit() {
        let (case, ds) = crate::test_support::load_case("cbps_sim_methods");
        let r = fit_with(&ds, Logit, Ate);
        let ra = fit_with(&ds, Logit, Att);
        let n = r.n_obs;
        Check::new(&case)
            .named("logit_coef", &r.names, &r.coefficients, Tol::coef(n))
            .named("logit_se", &r.names, &r.std_errors, Tol::se(n))
            .scalar("logit_deviance", r.deviance, Tol::stat(n))
            .vector("logit_ate_weights_head", &r.weights[..10], Tol::coef(n))
            .vector("logit_att_weights_head", &ra.weights[..10], Tol::coef(n))
            .finish();
        assert!(
            r.j_statistic.is_none()
                && r.cbps_j.is_none()
                && r.start_scale.is_none()
                && r.n_moments == 4
        );
    }

    /// CBPS() as shipped (optimize tol eps^0.25, BFGS reltol 1e-8) is at most 1.14e-4 (exact
    /// coef/ps), 5.95e-4 (exact J), 7.5e-6 / 1.6e-5 (over ATE coef / J) and 1.9e-7 (deviance)
    /// from the tight refit.
    #[test]
    fn test_validate_cbps_package_cross_check() {
        let (case, ds) = crate::test_support::load_case("cbps_sim_methods");
        // deviation CAUS-11: over_att_* not compared (CBPS 0.24 stops at a non-stationary point)
        for (method, estimand, p, rel, rel_j) in [
            (Exact, Ate, "exact_ate", 1e-3, 2e-3),
            (Exact, Att, "exact_att", 1e-3, 2e-3),
            (Over, Ate, "over_ate", 1e-4, 1e-4),
        ] {
            let r = fit_with(&ds, method, estimand);
            Check::new(&case)
                .named(
                    &format!("{p}_coef"),
                    &r.names,
                    &r.coefficients,
                    Tol::iterative(rel),
                )
                .scalar(&format!("{p}_J"), r.cbps_j.unwrap(), Tol::iterative(rel_j))
                .vector(
                    &format!("{p}_ps_head"),
                    &r.propensity_scores[..10],
                    Tol::iterative(rel),
                )
                .vector(
                    &format!("{p}_weights_head"),
                    &r.weights[..10],
                    Tol::iterative(rel),
                )
                .scalar(&format!("{p}_deviance"), r.deviance, Tol::iterative(1e-6))
                .finish();
        }
    }

    /// CAUS-55: before, ExactBalance and OverBalance ran the same GMM bit for bit, and no
    /// ATT existed.
    #[test]
    fn test_cbps_methods_and_estimands_differ() {
        let ds = sim();
        let over = fit_with(&ds, Over, Ate);
        let exact = fit_with(&ds, Exact, Ate);
        let logit = fit_with(&ds, Logit, Ate);
        let att = fit_with(&ds, Over, Att);
        // R: x2 -0.43756 (over) vs -0.42588 (exact)
        assert!(max_abs_diff(&over.coefficients, &exact.coefficients) > 1e-3);
        // R: x1 0.54499 (over) vs 0.55140 (glm)
        assert!(max_abs_diff(&over.coefficients, &logit.coefficients) > 1e-3);
        // R: x2 -0.43756 (ATE) vs -0.39838 (ATT)
        assert!(max_abs_diff(&over.coefficients, &att.coefficients) > 1e-2);
        for e in [Ate, Att] {
            let r = fit_with(&ds, Exact, e);
            assert!(
                r.balance_after.std_diff.iter().all(|d| d.abs() < 1e-8),
                "{e}: {:?}",
                r.balance_after.std_diff
            );
        }
        let t = column(&ds, "treat");
        for (ti, w) in t.iter().zip(&att.weights) {
            if *ti == 1.0 {
                assert!((w - 1.0 / 178.0).abs() < 1e-15);
            }
        }
    }

    /// CAUS-11: the old balance rows carried an extra p(1-p) and the wrong control sign.
    #[test]
    fn test_cbps_moment_jacobian_matches_finite_differences() {
        let ds = sim();
        let design = DesignMatrix::from_dataframe(ds.df(), &COVS, true).unwrap();
        let wh = whiten(&design.data).unwrap();
        let t = Array1::from(column(&ds, "treat"));
        let b0 = logit_start(&wh.u, &t).unwrap().0;
        let h = 1e-5;
        for e in [Ate, Att] {
            let beta = &b0 * 0.9;
            let jac = moments(&wh.u, &t, &beta, e).jacobian;
            let scale = jac.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            for j in 0..beta.len() {
                let mut up = beta.clone();
                let mut dn = beta.clone();
                up[j] += h;
                dn[j] -= h;
                let fd =
                    (moments(&wh.u, &t, &up, e).gbar - moments(&wh.u, &t, &dn, e).gbar) / (2.0 * h);
                for i in 0..fd.len() {
                    assert!(
                        (fd[i] - jac[[i, j]]).abs() <= 1e-6 * scale,
                        "{e} J[{i},{j}]: fd {} vs {}",
                        fd[i],
                        jac[[i, j]]
                    );
                }
            }
        }
    }

    /// Brent's termination rule leaves the minimiser of a unimodal function inside a bracket
    /// of width at most 4 tol1 (R's optimize satisfies the bound on these functions).
    #[test]
    fn test_brent_fmin_finds_the_minimum() {
        let fs: [(fn(f64) -> f64, f64); 4] = [
            (|x| (x - 0.93) * (x - 0.93) + 0.1, 0.93),
            (
                |x| {
                    let d = x - 1.0001994863243313;
                    (d * d) * (d * d)
                },
                1.0001994863243313,
            ),
            (|x| (x - 0.93) * (x - 0.93) * (x + 0.5), 0.93),
            (|x| -x, 1.1),
        ];
        for tol in [0.0001220703125, 1e-12] {
            for (i, (f, xstar)) in fs.iter().enumerate() {
                let x = brent_fmin(0.8, 1.1, tol, |x| Ok(f(x))).unwrap();
                let bound = 4.0 * (f64::EPSILON.sqrt() * xstar.abs() + tol / 3.0);
                assert!((0.8..=1.1).contains(&x), "f{i}: {x}");
                assert!(
                    (x - xstar).abs() <= bound,
                    "f{i} tol {tol}: {x} vs {xstar} (bound {bound})"
                );
            }
        }
    }

    /// The whitening makes the fit equivariant to the units of a covariate (R10).
    #[test]
    fn test_cbps_scale_equivariance() {
        let ds = sim();
        let scale = |c: &str, s: f64| column(&ds, c).iter().map(|v| v * s).collect::<Vec<f64>>();
        let scaled = Dataset::new(
            df! {
                "treat" => column(&ds, "treat"),
                "x1" => scale("x1", 1e-6),
                "x2" => scale("x2", 1e6),
                "x3" => column(&ds, "x3"),
            }
            .unwrap(),
        );
        let factor = [1.0, 1e6, 1e-6, 1.0];
        for (method, rel) in [(Exact, 1e-9), (Over, 1e-6)] {
            let a = fit_with(&ds, method, Ate);
            let b = fit_with(&scaled, method, Ate);
            assert!(a.converged && b.converged);
            for j in 0..4 {
                let (ca, cb) = (a.coefficients[j], b.coefficients[j] / factor[j]);
                assert!(
                    (cb - ca).abs() <= rel * ca.abs(),
                    "{method} coef {j}: {ca} vs {cb}"
                );
                let (sa, sb) = (a.std_errors[j], b.std_errors[j] / factor[j]);
                assert!(
                    (sb - sa).abs() <= rel * sa.abs(),
                    "{method} se {j}: {sa} vs {sb}"
                );
            }
            assert!(a.iterations.abs_diff(b.iterations) <= 1);
        }
    }

    /// A slowly converging over-identified fit (linear Gauss-Newton rate about 0.6) reaches
    /// the point where g'Wg can no longer be lowered in floating point before the relative
    /// step falls below 1e-8; it must still report convergence. Data: the comprehensive
    /// benchmark's design (x1, x2 ~ U(-2, 2), logit 0.3 x1 + 0.2 x2), n = 1000.
    #[test]
    fn test_cbps_over_converges_at_the_rounding_floor() {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15 ^ (1000 * 31 + 2);
        let mut unif = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let (mut t, mut x1, mut x2) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..1000 {
            let (a, b) = (unif() * 4.0 - 2.0, unif() * 4.0 - 2.0);
            let p = 1.0 / (1.0 + (-(0.3 * a + 0.2 * b)).exp());
            t.push(if unif() < p { 1.0 } else { 0.0 });
            x1.push(a);
            x2.push(b);
        }
        let ds = Dataset::new(df! { "treatment" => t, "x1" => x1, "x2" => x2 }.unwrap());
        for e in [Ate, Att] {
            let mut config = CbpsConfig::default();
            config.estimand = e;
            let r = run_cbps(&ds, "treatment", &["x1", "x2"], Some(config)).unwrap();
            assert!(r.converged, "{e}: {:?}", r.warnings);
            assert!(
                r.warnings.iter().all(|w| !w.contains("line search")),
                "{e}: {:?}",
                r.warnings
            );
        }
    }

    #[test]
    fn test_cbps_reports_non_convergence() {
        let ds = sim();
        let mut config = CbpsConfig::default();
        config.max_iter = 1;
        let r = run_cbps(&ds, "treat", &COVS, Some(config)).unwrap();
        assert!(!r.converged);
        assert_eq!(r.iterations, 1);
        assert!(
            r.warnings
                .iter()
                .any(|w| w.starts_with("CBPS (over) did not converge after 1 iteration")),
            "{:?}",
            r.warnings
        );
        assert!(r.to_string().contains("Converged: no (1 iteration"));
    }

    /// CAUS-113: the old local Newton logit reported nothing on separated data.
    #[test]
    fn test_cbps_logit_reports_glm_warnings() {
        let ds = Dataset::new(
            df! {
                "treat" => [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                "x1" => [-3.0, -2.1, -1.3, -0.4, 0.6, 1.2, 2.2, 3.1],
            }
            .unwrap(),
        );
        let r = cbps(&ds, "treat", &["x1"], Logit).unwrap();
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("fitted probabilities numerically 0 or 1 occurred")),
            "{:?}",
            r.warnings
        );
        assert!(r.to_string().contains("Warning: "));
    }

    #[test]
    fn test_cbps_method_parses_aliases_and_rejects_unknown() {
        CbpsMethod::choices().validate().unwrap();
        for s in ["over", "Over-Identified", "overbalance"] {
            assert_eq!(s.parse::<CbpsMethod>().unwrap(), Over, "{s}");
        }
        for s in ["exact", "EXACT", "balance only"] {
            assert_eq!(s.parse::<CbpsMethod>().unwrap(), Exact, "{s}");
        }
        for s in ["logit", "glm", "just", "Just_Identified"] {
            assert_eq!(s.parse::<CbpsMethod>().unwrap(), Logit, "{s}");
        }
        let msg = "exact_balance"
            .parse::<CbpsMethod>()
            .unwrap_err()
            .to_string();
        assert!(msg.contains("unknown CBPS method 'exact_balance'"), "{msg}");
        assert!(msg.contains("over, exact, logit"), "{msg}");
        assert_eq!(Exact.to_string(), "exact");
        assert_eq!(CbpsMethod::default(), Over);
        assert_eq!(CbpsMethod::ExactBalance, Over);
        assert_eq!(CbpsMethod::OverBalance, Over);
        assert_eq!(CbpsMethod::JustIdentified, Logit);
        assert_eq!(
            serde_json::from_str::<CbpsMethod>("\"ExactBalance\"").unwrap(),
            Over
        );
        assert_eq!(
            serde_json::from_str::<CbpsMethod>("\"JustIdentified\"").unwrap(),
            Logit
        );
        assert_eq!(serde_json::to_string(&Exact).unwrap(), "\"Exact\"");
    }

    #[test]
    fn test_cbps_estimand_parses_aliases_and_rejects_unknown() {
        CbpsEstimand::choices().validate().unwrap();
        for s in ["ATE", "ate", "average treatment effect"] {
            assert_eq!(s.parse::<CbpsEstimand>().unwrap(), Ate, "{s}");
        }
        for s in ["ATT", "att", "treated"] {
            assert_eq!(s.parse::<CbpsEstimand>().unwrap(), Att, "{s}");
        }
        let msg = "ATC".parse::<CbpsEstimand>().unwrap_err().to_string();
        assert!(msg.contains("unknown CBPS estimand 'ATC'"), "{msg}");
        assert!(msg.contains("ATE, ATT"), "{msg}");
        assert_eq!(Att.to_string(), "ATT");
        assert_eq!(CbpsEstimand::default(), Ate);
    }

    #[test]
    fn test_cbps_weights_sum_to_one_per_group() {
        let p = [0.3, 0.4, 0.5, 0.6, 0.7, 0.3, 0.4, 0.5, 0.6, 0.7];
        let t = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        for e in [Ate, Att] {
            let w = cbps_weights(&p, &t, e);
            let st: f64 = w[..5].iter().sum();
            let sc: f64 = w[5..].iter().sum();
            assert!((st - 1.0).abs() < 1e-12 && (sc - 1.0).abs() < 1e-12, "{e}");
            if e == Att {
                assert!(w[..5].iter().all(|wi| (wi - 0.2).abs() < 1e-15));
            }
        }
    }

    #[test]
    fn test_cbps_display() {
        let ds = sim();
        let s = fit_with(&ds, Over, Ate).to_string();
        for needle in [
            "Method: over (score and balance conditions, two-step GMM)",
            "Estimand: ATE",
            "J-statistic (CBPS, g'Wg): 0.00047848",
            "n*J = 0.1914 (df = 4)",
            "Balance Improvement",
            "Converged: yes",
        ] {
            assert!(s.contains(needle), "missing {needle:?} in\n{s}");
        }
        let s = fit_with(&ds, Exact, Att).to_string();
        assert!(s.contains("Method: exact (balance conditions only)"), "{s}");
        assert!(s.contains("Estimand: ATT"), "{s}");
        assert!(!s.contains("Hansen"), "{s}");
    }

    #[test]
    fn test_cbps_missing_column() {
        let ds = sim();
        assert!(run_cbps(&ds, "nonexistent", &["x1"], None).is_err());
    }

    #[test]
    fn test_cbps_constant_treatment() {
        // All treated - should fail
        let df = df! {
            "treatment" => [1.0, 1.0, 1.0, 1.0, 1.0],
            "x1" => [1.0, 2.0, 3.0, 4.0, 5.0]
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_cbps(&dataset, "treatment", &["x1"], None);
        assert!(result.is_err());
    }

    fn is_spec(e: &EconError) -> bool {
        matches!(e, EconError::InvalidSpecification { .. })
    }

    #[test]
    fn test_cbps_rejects_invalid_parameters() {
        let ds = sim();
        let err = |f: &dyn Fn(&mut CbpsConfig)| {
            let mut c = CbpsConfig::default();
            f(&mut c);
            let e = run_cbps(&ds, "treat", &COVS, Some(c)).unwrap_err();
            assert!(is_spec(&e), "{e:?}");
            e.to_string()
        };
        for v in [0.0, f64::NAN, -1.0, f64::INFINITY] {
            assert!(err(&|c| c.tolerance = v).contains("tolerance"));
        }
        assert!(err(&|c| c.max_iter = 0).contains("max_iter"));
        for v in [0.0, f64::NAN] {
            assert!(err(&|c| c.balance_threshold = v).contains("balance_threshold"));
        }
    }

    #[test]
    fn test_cbps_rejects_non_binary_treatment() {
        let ds = sim();
        let mut treat = column(&ds, "treat");
        treat[0] = 2.0;
        let bad = Dataset::new(
            df! {
                "treat" => treat,
                "x1" => column(&ds, "x1"),
                "x2" => column(&ds, "x2"),
                "x3" => column(&ds, "x3"),
            }
            .unwrap(),
        );
        let e = run_cbps(&bad, "treat", &COVS, None).unwrap_err();
        let msg = e.to_string();
        assert!(is_spec(&e), "{e:?}");
        assert!(
            msg.contains("treat") && msg.contains("0/1") && msg.contains("row 1 has 2"),
            "{msg}"
        );
    }

    #[test]
    fn test_cbps_rejects_collinear_covariates() {
        let ds = sim();
        let x1 = column(&ds, "x1");
        let x2 = column(&ds, "x2");
        let x4: Vec<f64> = x1.iter().zip(&x2).map(|(a, b)| a + b).collect();
        let n = x1.len();
        let wide = Dataset::new(
            df! {
                "treat" => column(&ds, "treat"),
                "x1" => x1,
                "x2" => x2,
                "x3" => column(&ds, "x3"),
                "x4" => x4,
                "k" => vec![1.0; n],
            }
            .unwrap(),
        );
        let e = run_cbps(&wide, "treat", &["x1", "x2", "x3", "x4"], None).unwrap_err();
        let msg = e.to_string();
        assert!(is_spec(&e), "{e:?}");
        assert!(msg.contains("collinear") && msg.contains("x4"), "{msg}");
        let e = run_cbps(&wide, "treat", &["x1", "k"], None).unwrap_err();
        let msg = e.to_string();
        assert!(is_spec(&e), "{e:?}");
        assert!(
            msg.contains("collinear") && msg.contains("remove: k"),
            "{msg}"
        );
    }

    #[test]
    fn test_cbps_rejects_too_few_observations() {
        let ds = sim();
        let head = Dataset::new(ds.df().head(Some(4)));
        assert_eq!(column(&head, "treat"), vec![1.0, 1.0, 0.0, 0.0]);
        match run_cbps(&head, "treat", &COVS, None) {
            Err(EconError::InsufficientData {
                required: 5,
                provided: 4,
                ..
            }) => {}
            other => panic!("expected InsufficientData {{ 5, 4 }}, got {other:?}"),
        }
    }

    #[test]
    fn test_balance_table_display() {
        let mut table = BalanceTable::new();
        table.covariates = vec!["x1".to_string(), "x2".to_string()];
        table.mean_treated = vec![0.9, 0.8];
        table.mean_control = vec![0.4, 0.3];
        table.std_diff = vec![1.2, 1.5];
        table.var_ratio = vec![1.1, 0.9];
        table.max_std_diff = 1.5;
        table.n_balanced = 0;

        let output = format!("{}", table);
        assert!(output.contains("Balance Table"));
        assert!(output.contains("x1"));
        assert!(output.contains("x2"));
    }

    // ── T2-WP13: missing values and column types (R case causal_mediation_na) ──

    #[test]
    fn test_validate_na_cbps_columns() {
        use crate::econometrics::treatment::na_fixtures as nf;
        let (case, ds) = crate::test_support::load_case("causal_mediation_na");
        // the dtype check runs the Logit path
        let run = |ds: &Dataset, _y: &str, d: &str, x: &[&str]| {
            run_cbps(
                ds,
                d,
                x,
                Some(CbpsConfig {
                    method: CbpsMethod::JustIdentified,
                    ..Default::default()
                }),
            )
        };
        nf::assert_refusals(&case, &ds, false, run);
        nf::assert_treatment_dtypes_agree(&case, &ds, run, |r| r.coefficients.clone());
    }
}
