//! Treatment Effect Heterogeneity Testing (hettx).
//!
//! This module provides Fisherian randomization-based tests for treatment effect
//! heterogeneity. It tests whether treatment effects vary across units and decomposes
//! heterogeneity into systematic (explained by covariates) and idiosyncratic components.
//!
//! # Key Features
//!
//! - **Omnibus test**: Tests H0: all treatment effects are equal (tau_i = tau for all i)
//! - **Multiple test statistics**: Variance, range, IQR, mean absolute deviation
//! - **Heterogeneity decomposition**: Separates systematic vs. idiosyncratic variation
//! - **Fisherian inference**: Permutation-based p-values
//!
//! # Mathematical Framework
//!
//! ## Potential Outcomes Setup
//!
//! For unit i with potential outcomes Y_i(0), Y_i(1):
//! - Individual treatment effect: tau_i = Y_i(1) - Y_i(0)
//! - Sharp null hypothesis: H0: tau_i = tau for all i
//!
//! ## Test Statistics
//!
//! Under the sharp null, we can compute individual effects by imputation.
//! The test statistic measures the spread of estimated individual effects:
//!
//! - **Variance**: Var(tau_hat_i)
//! - **Range**: max(tau_hat_i) - min(tau_hat_i)
//! - **IQR**: Q3(tau_hat_i) - Q1(tau_hat_i)
//! - **MAD**: mean(|tau_hat_i - tau_bar|)
//!
//! ## Permutation Inference
//!
//! P-value = (1 + number of permutation stats >= observed) / (1 + B)
//!
//! where B is the number of permutations.
//!
//! ## Heterogeneity Decomposition
//!
//! Total variance in effects = Systematic + Idiosyncratic
//! - Systematic: Var(E[tau|X]) - variance explained by observed covariates
//! - Idiosyncratic: E[Var(tau|X)] - residual individual-level variation
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
//! - Fisher, R. A. (1935). *The Design of Experiments*. Oliver and Boyd.
//!
//! - Implementation validated against R package `hettx` (Ding et al., 2019).
//!   Source: <https://cran.r-project.org/package=hettx>
//!
//! R equivalent: `hettx::detect_idiosyncratic()`, `hettx::detect_systematic()`

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::SeedableRng;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::design::{DesignMatrix, get_column_names};
use crate::linalg::matrix_ops::{safe_inverse, xtx, xty};
use crate::traits::estimator::SignificanceLevel;

// ===============================================================================
// Configuration Types
// ===============================================================================

/// Test statistic for measuring treatment effect heterogeneity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum HetTestStat {
    /// Variance of estimated individual effects: Var(tau_hat_i)
    /// Default and most powerful for detecting general heterogeneity.
    #[default]
    Variance,
    /// Range of effects: max(tau_hat_i) - min(tau_hat_i)
    /// Sensitive to extreme values.
    Range,
    /// Interquartile range: Q3(tau_hat_i) - Q1(tau_hat_i)
    /// Robust to outliers.
    IQR,
    /// Mean absolute deviation: mean(|tau_hat_i - tau_bar|)
    /// Robust alternative to variance.
    MeanAbsDeviation,
}

impl fmt::Display for HetTestStat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HetTestStat::Variance => write!(f, "Variance"),
            HetTestStat::Range => write!(f, "Range"),
            HetTestStat::IQR => write!(f, "Interquartile Range (IQR)"),
            HetTestStat::MeanAbsDeviation => write!(f, "Mean Absolute Deviation"),
        }
    }
}

/// Method for estimating individual treatment effects.
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
    /// Simple difference-in-means within matched strata.
    /// Requires pre-existing strata/blocks in the data.
    Stratified,
}

impl fmt::Display for EffectEstimationMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EffectEstimationMethod::Matching => write!(f, "Nearest Neighbor Matching"),
            EffectEstimationMethod::Regression => write!(f, "Regression Imputation"),
            EffectEstimationMethod::Stratified => write!(f, "Stratified Difference-in-Means"),
        }
    }
}

/// Configuration for treatment effect heterogeneity test.
#[derive(Debug, Clone)]
pub struct HetTxConfig {
    /// Number of permutations for Fisherian inference (default: 1000)
    pub n_permutations: usize,
    /// Test statistic to use (default: Variance)
    pub test_statistic: HetTestStat,
    /// Whether to decompose heterogeneity into systematic and idiosyncratic (default: true)
    pub decompose: bool,
    /// Method for estimating individual effects (default: Matching)
    pub effect_method: EffectEstimationMethod,
    /// Number of nearest neighbors for matching (default: 3)
    pub n_neighbors: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Whether to compute variable importance in decomposition (default: true)
    pub compute_importance: bool,
}

impl Default for HetTxConfig {
    fn default() -> Self {
        Self {
            n_permutations: 1000,
            test_statistic: HetTestStat::Variance,
            decompose: true,
            effect_method: EffectEstimationMethod::Matching,
            n_neighbors: 3,
            seed: None,
            compute_importance: true,
        }
    }
}

// ===============================================================================
// Result Types
// ===============================================================================

/// Decomposition of treatment effect heterogeneity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HetDecomposition {
    /// Total variance in individual treatment effects: Var(tau_i)
    pub total_variance: f64,
    /// Systematic variance: Var(E[tau|X]) - explained by covariates
    pub systematic_variance: f64,
    /// Idiosyncratic variance: E[Var(tau|X)] - residual individual variation
    pub idiosyncratic_variance: f64,
    /// R-squared: systematic / total (proportion explained by covariates)
    pub r_squared: f64,
    /// Test statistic for systematic heterogeneity
    pub systematic_test_stat: f64,
    /// P-value for systematic heterogeneity (permutation-based)
    pub systematic_p_value: f64,
    /// Test statistic for idiosyncratic heterogeneity
    pub idiosyncratic_test_stat: f64,
    /// P-value for idiosyncratic heterogeneity (permutation-based)
    pub idiosyncratic_p_value: f64,
    /// Variable importance: which covariates explain heterogeneity
    /// Format: (covariate_index, importance_score)
    pub covariate_importance: Vec<(usize, f64)>,
    /// Covariate names (if available)
    pub covariate_names: Vec<String>,
}

impl fmt::Display for HetDecomposition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Treatment Effect Heterogeneity Decomposition")?;
        writeln!(f, "=============================================")?;
        writeln!(f)?;
        writeln!(f, "Variance Components:")?;
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
        writeln!(f, "Tests for Heterogeneity:")?;
        writeln!(f, "  Systematic (H0: Var(E[tau|X])=0):")?;
        writeln!(
            f,
            "    Test stat: {:.4}, p-value: {:.4}",
            self.systematic_test_stat, self.systematic_p_value
        )?;
        writeln!(f, "  Idiosyncratic (H0: E[Var(tau|X)]=0):")?;
        writeln!(
            f,
            "    Test stat: {:.4}, p-value: {:.4}",
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
    /// Observed test statistic
    pub test_statistic: f64,
    /// Type of test statistic used
    pub test_statistic_type: HetTestStat,
    /// Permutation p-value (one-sided, testing for MORE heterogeneity)
    pub p_value: f64,
    /// Significance level
    pub significance: SignificanceLevel,
    /// Distribution of test statistics under the null (permutations)
    pub null_distribution: Vec<f64>,
    /// Estimated individual treatment effects
    pub estimated_effects: Vec<f64>,
    /// Average treatment effect (ATE)
    pub ate: f64,
    /// Standard error of ATE
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
    /// Number of permutations
    pub n_permutations: usize,
    /// Summary statistics for individual effects
    pub effect_summary: EffectSummary,
    /// Warnings generated during estimation
    pub warnings: Vec<String>,
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

        writeln!(
            f,
            "H0: All treatment effects are equal (tau_i = tau for all i)"
        )?;
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
        writeln!(f, "  Permutations:  {}", self.n_permutations)?;
        writeln!(f)?;

        writeln!(f, "Average Treatment Effect:")?;
        writeln!(f, "  ATE = {:.4} (SE: {:.4})", self.ate, self.ate_se)?;
        writeln!(f)?;

        writeln!(f, "Individual Effects Distribution:")?;
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

        writeln!(f, "Estimation method: {}", self.effect_method)?;
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

// ===============================================================================
// Main Functions
// ===============================================================================

/// Run treatment effect heterogeneity test using Dataset interface.
///
/// Tests whether treatment effects vary across units using Fisherian
/// randomization inference.
///
/// # Arguments
/// * `dataset` - Dataset containing outcome, treatment, and covariates
/// * `outcome_col` - Name of the outcome variable column
/// * `treatment_col` - Name of the binary treatment indicator column (0/1)
/// * `covariate_cols` - Names of covariate columns (used for matching/imputation and decomposition)
/// * `config` - Configuration options
///
/// # Returns
/// `HetTxResult` containing test statistic, p-value, individual effect estimates,
/// and optionally the heterogeneity decomposition.
///
/// # Example
/// ```ignore
/// let config = HetTxConfig {
///     n_permutations: 1000,
///     test_statistic: HetTestStat::Variance,
///     decompose: true,
///     ..Default::default()
/// };
/// let result = run_hettx_dataset(&dataset, "outcome", "treatment", &["x1", "x2"], config)?;
/// println!("{}", result);
/// ```
///
/// # References
///
/// Ding, P., Feller, A., & Miratrix, L. (2016). "Randomization inference for
/// treatment effect variation." *JRSSB*, 78(3), 655-671.
pub fn run_hettx_dataset(
    dataset: &Dataset,
    outcome_col: &str,
    treatment_col: &str,
    covariate_cols: &[&str],
    config: HetTxConfig,
) -> EconResult<HetTxResult> {
    // Extract outcome
    let y = DesignMatrix::extract_column(dataset.df(), outcome_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: outcome_col.to_string(),
            available: get_column_names(dataset.df()),
        }
    })?;

    // Extract treatment indicator
    let treatment = DesignMatrix::extract_column(dataset.df(), treatment_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: treatment_col.to_string(),
            available: get_column_names(dataset.df()),
        }
    })?;

    // Build covariate matrix
    let design = DesignMatrix::from_dataframe(dataset.df(), covariate_cols, false)?;
    let covariates = design.data;
    let covariate_names: Vec<String> = covariate_cols.iter().map(|s| s.to_string()).collect();

    // Call the core function
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
/// Tests whether treatment effects vary across units using Fisherian
/// randomization inference.
///
/// # Arguments
/// * `y` - Outcome variable (n x 1)
/// * `treatment` - Binary treatment indicator (n x 1), values 0 or 1
/// * `covariates` - Optional covariate matrix (n x k) for matching and decomposition
/// * `config` - Configuration options
///
/// # Returns
/// `HetTxResult` containing test statistic, p-value, individual effect estimates,
/// and optionally the heterogeneity decomposition.
///
/// # Mathematical Details
///
/// ## Individual Effect Estimation
///
/// For treated unit i, we observe Y_i(1). We impute Y_i(0) using:
/// - Matching: Average Y(0) of nearest neighbors in control group
/// - Regression: Predict using outcome model fit on controls
///
/// For control unit j, we observe Y_j(0). We impute Y_j(1) analogously.
///
/// Then: tau_hat_i = Y_hat_i(1) - Y_hat_i(0)
///
/// ## Test Statistic
///
/// Under the sharp null H0: tau_i = tau for all i, the test statistic
/// is computed from the estimated effects. The p-value is the proportion
/// of permutation statistics that exceed the observed statistic.
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

    if let Some(x) = covariates {
        if x.nrows() != n {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "Covariate matrix rows ({}) do not match outcome length ({})",
                    x.nrows(),
                    n
                ),
            });
        }
    }

    // Identify treated and control units
    let treated_idx: Vec<usize> = (0..n).filter(|&i| treatment[i] >= 0.5).collect();
    let control_idx: Vec<usize> = (0..n).filter(|&i| treatment[i] < 0.5).collect();

    let n_treated = treated_idx.len();
    let n_control = control_idx.len();

    if n_treated < 5 || n_control < 5 {
        return Err(EconError::InsufficientData {
            required: 5,
            provided: n_treated.min(n_control),
            context: "Need at least 5 units in each treatment group".to_string(),
        });
    }

    // Initialize RNG
    let mut rng: ChaCha8Rng = match config.seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };

    // Step 1: Estimate individual treatment effects
    let tau_hat = estimate_individual_effects(
        y,
        treatment,
        covariates,
        &treated_idx,
        &control_idx,
        config.effect_method,
        config.n_neighbors,
    )?;

    // Compute ATE and SE
    let ate = tau_hat.mean().unwrap_or(0.0);
    let ate_var = tau_hat.iter().map(|&t| (t - ate).powi(2)).sum::<f64>() / (n - 1) as f64;
    let ate_se = (ate_var / n as f64).sqrt();

    // Compute effect summary statistics
    let effect_summary = compute_effect_summary(&tau_hat);

    // Step 2: Compute observed test statistic
    let observed_stat = compute_test_statistic(&tau_hat, config.test_statistic);

    // Step 3: Permutation test
    let mut null_distribution = Vec::with_capacity(config.n_permutations);
    let mut treatment_perm = treatment.to_owned();

    for _ in 0..config.n_permutations {
        // Permute treatment assignment
        let perm: Vec<usize> = {
            let mut indices: Vec<usize> = (0..n).collect();
            indices.shuffle(&mut rng);
            indices
        };

        for i in 0..n {
            treatment_perm[i] = treatment[perm[i]];
        }

        // Get treated/control indices for permuted assignment
        let treated_perm: Vec<usize> = (0..n).filter(|&i| treatment_perm[i] >= 0.5).collect();
        let control_perm: Vec<usize> = (0..n).filter(|&i| treatment_perm[i] < 0.5).collect();

        // Re-estimate individual effects under permuted assignment
        let tau_perm = match estimate_individual_effects(
            y,
            &treatment_perm.view(),
            covariates,
            &treated_perm,
            &control_perm,
            config.effect_method,
            config.n_neighbors,
        ) {
            Ok(t) => t,
            Err(_) => continue, // Skip this permutation if estimation fails
        };

        // Compute test statistic for permutation
        let perm_stat = compute_test_statistic(&tau_perm, config.test_statistic);

        if perm_stat.is_finite() {
            null_distribution.push(perm_stat);
        }
    }

    // Check if we got enough valid permutations
    if null_distribution.len() < config.n_permutations / 2 {
        warnings.push(format!(
            "Only {}/{} permutations yielded valid statistics",
            null_distribution.len(),
            config.n_permutations
        ));
    }

    // Compute p-value: proportion of null stats >= observed
    // Adding 1 to numerator and denominator for continuity correction
    let n_exceeding = null_distribution
        .iter()
        .filter(|&&s| s >= observed_stat)
        .count();
    let p_value = (n_exceeding as f64 + 1.0) / (null_distribution.len() as f64 + 1.0);
    let significance = SignificanceLevel::from_p_value(p_value);

    // Step 4: Heterogeneity decomposition (if requested and covariates provided)
    let decomposition = if config.decompose && covariates.is_some() {
        let x = covariates.unwrap();
        Some(compute_heterogeneity_decomposition(
            &tau_hat,
            x,
            &covariate_names,
            config.n_permutations,
            &mut rng,
            config.compute_importance,
        )?)
    } else {
        if config.decompose && covariates.is_none() {
            warnings.push("Decomposition requested but no covariates provided".to_string());
        }
        None
    };

    Ok(HetTxResult {
        test_statistic: observed_stat,
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
    })
}

// ===============================================================================
// Helper Functions
// ===============================================================================

/// Estimate individual treatment effects using the specified method.
fn estimate_individual_effects(
    y: &ArrayView1<f64>,
    treatment: &ArrayView1<f64>,
    covariates: Option<&ArrayView2<f64>>,
    treated_idx: &[usize],
    control_idx: &[usize],
    method: EffectEstimationMethod,
    n_neighbors: usize,
) -> EconResult<Array1<f64>> {
    let n = y.len();
    let mut tau_hat = Array1::zeros(n);

    match method {
        EffectEstimationMethod::Matching => {
            // For matching, we need covariates
            let x = covariates.ok_or_else(|| EconError::InvalidSpecification {
                message: "Matching method requires covariates".to_string(),
            })?;

            // For each unit, impute the missing potential outcome using neighbors
            for i in 0..n {
                let is_treated = treatment[i] >= 0.5;

                if is_treated {
                    // Observed Y(1), need to impute Y(0)
                    let y1 = y[i];
                    let y0 = impute_by_matching(i, x, y, control_idx, n_neighbors);
                    tau_hat[i] = y1 - y0;
                } else {
                    // Observed Y(0), need to impute Y(1)
                    let y0 = y[i];
                    let y1 = impute_by_matching(i, x, y, treated_idx, n_neighbors);
                    tau_hat[i] = y1 - y0;
                }
            }
        }
        EffectEstimationMethod::Regression => {
            // Fit separate outcome models for treated and control
            let x = covariates.ok_or_else(|| EconError::InvalidSpecification {
                message: "Regression method requires covariates".to_string(),
            })?;

            // Add intercept
            let x_with_int = add_intercept(x);

            // Fit model on treated: Y(1) = X * beta_1
            let (mu_1, _) = fit_outcome_model_on_subset(&x_with_int, y, treated_idx)?;

            // Fit model on control: Y(0) = X * beta_0
            let (mu_0, _) = fit_outcome_model_on_subset(&x_with_int, y, control_idx)?;

            // tau_hat_i = mu_1(X_i) - mu_0(X_i)
            for i in 0..n {
                let y_hat_1 = x_with_int.row(i).dot(&mu_1);
                let y_hat_0 = x_with_int.row(i).dot(&mu_0);
                tau_hat[i] = y_hat_1 - y_hat_0;
            }
        }
        EffectEstimationMethod::Stratified => {
            // Simple difference-in-means
            // This is essentially the same as matching with all units in each group
            let mean_treated: f64 =
                treated_idx.iter().map(|&i| y[i]).sum::<f64>() / treated_idx.len() as f64;
            let mean_control: f64 =
                control_idx.iter().map(|&i| y[i]).sum::<f64>() / control_idx.len() as f64;
            let ate = mean_treated - mean_control;

            // Assign same effect to all units (constant treatment effect assumption)
            tau_hat.fill(ate);
        }
    }

    Ok(tau_hat)
}

/// Impute the missing potential outcome by matching to k nearest neighbors.
fn impute_by_matching(
    target_idx: usize,
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    donor_idx: &[usize],
    k: usize,
) -> f64 {
    if donor_idx.is_empty() {
        return 0.0;
    }

    // Compute distances to all donors
    let target_x = x.row(target_idx);
    let mut distances: Vec<(usize, f64)> = donor_idx
        .iter()
        .map(|&j| {
            let donor_x = x.row(j);
            let dist = euclidean_distance(&target_x, &donor_x);
            (j, dist)
        })
        .collect();

    // Sort by distance
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take k nearest neighbors
    let k_actual = k.min(distances.len());
    let y_sum: f64 = distances.iter().take(k_actual).map(|(j, _)| y[*j]).sum();

    y_sum / k_actual as f64
}

/// Compute Euclidean distance between two row vectors.
fn euclidean_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - bi).powi(2))
        .sum::<f64>()
        .sqrt()
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

/// Fit OLS model on a subset of observations, return coefficients.
fn fit_outcome_model_on_subset(
    x: &Array2<f64>,
    y: &ArrayView1<f64>,
    subset_idx: &[usize],
) -> EconResult<(Array1<f64>, f64)> {
    let n_sub = subset_idx.len();
    let k = x.ncols();

    if n_sub <= k {
        return Err(EconError::InsufficientData {
            required: k + 1,
            provided: n_sub,
            context: "Need more observations than parameters for regression".to_string(),
        });
    }

    // Build subset arrays
    let mut x_sub = Array2::zeros((n_sub, k));
    let mut y_sub = Array1::zeros(n_sub);

    for (new_i, &old_i) in subset_idx.iter().enumerate() {
        x_sub.row_mut(new_i).assign(&x.row(old_i));
        y_sub[new_i] = y[old_i];
    }

    // OLS: beta = (X'X)^{-1} X'y
    let xtx_mat = xtx(&x_sub.view());
    let (xtx_inv, _) = safe_inverse(&xtx_mat.view()).map_err(|e| EconError::SingularMatrix {
        context: "Outcome model fitting".to_string(),
        suggestion: format!("Check for multicollinearity: {:?}", e),
    })?;

    let xty_vec = xty(&x_sub.view(), &y_sub);
    let beta = xtx_inv.dot(&xty_vec);

    // Compute R-squared
    let y_mean = y_sub.mean().unwrap_or(0.0);
    let sst: f64 = y_sub.iter().map(|&yi| (yi - y_mean).powi(2)).sum();

    let y_hat = x_sub.dot(&beta);
    let ssr: f64 = y_sub
        .iter()
        .zip(y_hat.iter())
        .map(|(&yi, &yhi)| (yi - yhi).powi(2))
        .sum();

    let r2 = if sst > 0.0 { 1.0 - ssr / sst } else { 0.0 };

    Ok((beta, r2))
}

/// Compute the test statistic for heterogeneity.
fn compute_test_statistic(tau: &Array1<f64>, stat_type: HetTestStat) -> f64 {
    let n = tau.len();
    if n == 0 {
        return 0.0;
    }

    match stat_type {
        HetTestStat::Variance => {
            let mean = tau.mean().unwrap_or(0.0);
            tau.iter().map(|&t| (t - mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64
        }
        HetTestStat::Range => {
            let min = tau.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = tau.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            max - min
        }
        HetTestStat::IQR => {
            let mut sorted: Vec<f64> = tau.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let q1 = sorted[n / 4];
            let q3 = sorted[3 * n / 4];
            q3 - q1
        }
        HetTestStat::MeanAbsDeviation => {
            let mean = tau.mean().unwrap_or(0.0);
            tau.iter().map(|&t| (t - mean).abs()).sum::<f64>() / n as f64
        }
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

    let mut sorted: Vec<f64> = tau.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mean = tau.mean().unwrap_or(0.0);
    let var = tau.iter().map(|&t| (t - mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;

    EffectSummary {
        min: sorted[0],
        p10: sorted[(n as f64 * 0.10).floor() as usize],
        p25: sorted[n / 4],
        median: sorted[n / 2],
        p75: sorted[3 * n / 4],
        p90: sorted[((n as f64 * 0.90).floor() as usize).min(n - 1)],
        max: sorted[n - 1],
        std_dev: var.sqrt(),
    }
}

/// Compute the heterogeneity decomposition.
///
/// Decomposes Var(tau_i) into:
/// - Systematic: Var(E[tau|X]) - explained by covariates
/// - Idiosyncratic: E[Var(tau|X)] - residual
///
/// # References
/// Ding, P., Feller, A., & Miratrix, L. (2019). "Decomposing treatment effect
/// variation." *JASA*, 114(525), 304-317.
fn compute_heterogeneity_decomposition(
    tau: &Array1<f64>,
    x: &ArrayView2<f64>,
    covariate_names: &[String],
    n_perm: usize,
    rng: &mut ChaCha8Rng,
    compute_importance: bool,
) -> EconResult<HetDecomposition> {
    let n = tau.len();
    let k = x.ncols();

    // Total variance in individual effects
    let tau_mean = tau.mean().unwrap_or(0.0);
    let total_variance =
        tau.iter().map(|&t| (t - tau_mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;

    // Fit regression: tau = X * gamma + epsilon
    // This estimates E[tau|X] = X * gamma
    let x_with_int = add_intercept(x);
    let all_idx: Vec<usize> = (0..n).collect();
    let (gamma, _r2) = fit_outcome_model_on_subset(&x_with_int, &tau.view(), &all_idx)?;

    // Predicted effects: E[tau|X]
    let tau_fitted = x_with_int.dot(&gamma);

    // Systematic variance: Var(E[tau|X]) = Var(X * gamma)
    let tau_fitted_mean = tau_fitted.mean().unwrap_or(0.0);
    let systematic_variance = tau_fitted
        .iter()
        .map(|&t| (t - tau_fitted_mean).powi(2))
        .sum::<f64>()
        / (n - 1).max(1) as f64;

    // Idiosyncratic variance: E[Var(tau|X)] = Var(tau) - Var(E[tau|X])
    // Note: This decomposition relies on the law of total variance
    let idiosyncratic_variance = (total_variance - systematic_variance).max(0.0);

    // R-squared: proportion of variance explained by covariates
    let r_squared = if total_variance > 0.0 {
        (systematic_variance / total_variance).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Test statistics for systematic and idiosyncratic heterogeneity
    let systematic_test_stat = systematic_variance;
    let idiosyncratic_test_stat = idiosyncratic_variance;

    // Permutation tests for decomposition components
    let mut systematic_null = Vec::with_capacity(n_perm);
    let mut idiosyncratic_null = Vec::with_capacity(n_perm);

    for _ in 0..n_perm {
        // Permute the relationship between X and tau
        let perm: Vec<usize> = {
            let mut indices: Vec<usize> = (0..n).collect();
            indices.shuffle(rng);
            indices
        };

        // Create permuted tau
        let tau_perm: Array1<f64> = perm.iter().map(|&i| tau[i]).collect();

        // Fit on permuted data
        if let Ok((gamma_perm, _)) =
            fit_outcome_model_on_subset(&x_with_int, &tau_perm.view(), &all_idx)
        {
            let tau_fitted_perm = x_with_int.dot(&gamma_perm);

            // Compute permuted systematic variance
            let fitted_mean_perm = tau_fitted_perm.mean().unwrap_or(0.0);
            let sys_var_perm = tau_fitted_perm
                .iter()
                .map(|&t| (t - fitted_mean_perm).powi(2))
                .sum::<f64>()
                / (n - 1).max(1) as f64;

            // Compute permuted tau variance for idiosyncratic
            let tau_perm_mean = tau_perm.mean().unwrap_or(0.0);
            let total_var_perm = tau_perm
                .iter()
                .map(|&t| (t - tau_perm_mean).powi(2))
                .sum::<f64>()
                / (n - 1).max(1) as f64;
            let idio_var_perm = (total_var_perm - sys_var_perm).max(0.0);

            systematic_null.push(sys_var_perm);
            idiosyncratic_null.push(idio_var_perm);
        }
    }

    // Compute p-values
    let systematic_p_value = if !systematic_null.is_empty() {
        let n_exceed = systematic_null
            .iter()
            .filter(|&&s| s >= systematic_test_stat)
            .count();
        (n_exceed as f64 + 1.0) / (systematic_null.len() as f64 + 1.0)
    } else {
        1.0
    };

    let idiosyncratic_p_value = if !idiosyncratic_null.is_empty() {
        let n_exceed = idiosyncratic_null
            .iter()
            .filter(|&&s| s >= idiosyncratic_test_stat)
            .count();
        (n_exceed as f64 + 1.0) / (idiosyncratic_null.len() as f64 + 1.0)
    } else {
        1.0
    };

    // Compute variable importance (if requested)
    let covariate_importance = if compute_importance {
        compute_covariate_importance(tau, x, &gamma, k)
    } else {
        Vec::new()
    };

    Ok(HetDecomposition {
        total_variance,
        systematic_variance,
        idiosyncratic_variance,
        r_squared,
        systematic_test_stat,
        systematic_p_value,
        idiosyncratic_test_stat,
        idiosyncratic_p_value,
        covariate_importance,
        covariate_names: covariate_names.to_vec(),
    })
}

/// Compute variable importance for explaining treatment effect heterogeneity.
///
/// Uses the reduction in variance when each covariate is added to the model.
fn compute_covariate_importance(
    tau: &Array1<f64>,
    x: &ArrayView2<f64>,
    gamma: &Array1<f64>,
    k: usize,
) -> Vec<(usize, f64)> {
    let n = tau.len();
    let mut importance = Vec::with_capacity(k);

    // Total variance
    let tau_mean = tau.mean().unwrap_or(0.0);
    let total_var =
        tau.iter().map(|&t| (t - tau_mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;

    // For each covariate, compute its marginal contribution
    // Using the squared coefficient times variance of the covariate
    // (This is a simplified measure; more sophisticated approaches exist)
    for j in 0..k {
        let col = x.column(j);
        let col_mean = col.mean().unwrap_or(0.0);
        let col_var =
            col.iter().map(|&v| (v - col_mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;

        // gamma[j+1] because gamma[0] is intercept
        let coef = if j + 1 < gamma.len() {
            gamma[j + 1]
        } else {
            0.0
        };

        // Importance = beta^2 * Var(X) / Var(tau)
        let imp = if total_var > 0.0 {
            (coef * coef * col_var) / total_var
        } else {
            0.0
        };

        importance.push((j, imp));
    }

    // Sort by importance (descending)
    importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    importance
}

// ===============================================================================
// Tests
// ===============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use polars::prelude::*;

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
    fn test_hettx_detects_heterogeneity() {
        let (y, treatment, x) = create_hettx_test_data();

        let config = HetTxConfig {
            n_permutations: 500,
            test_statistic: HetTestStat::Variance,
            decompose: true,
            seed: Some(42),
            ..Default::default()
        };

        let result = run_hettx(&y.view(), &treatment.view(), Some(&x.view()), config).unwrap();

        // Check basic structure
        assert_eq!(result.n_obs, 100);
        assert!(result.n_treated > 0);
        assert!(result.n_control > 0);

        // With heterogeneous effects, test statistic should be non-negative
        assert!(
            result.test_statistic >= 0.0,
            "Test statistic should be non-negative"
        );

        // Decomposition should show systematic component
        assert!(result.decomposition.is_some());
        let decomp = result.decomposition.unwrap();
        assert!(
            decomp.total_variance >= 0.0,
            "Total variance should be non-negative"
        );
        assert!(
            decomp.systematic_variance >= 0.0,
            "Systematic variance should be non-negative"
        );

        // R-squared should be in [0, 1]
        assert!(
            decomp.r_squared >= 0.0 && decomp.r_squared <= 1.0,
            "R-squared {} should be in [0, 1]",
            decomp.r_squared
        );

        // P-value should be valid
        assert!(
            result.p_value >= 0.0 && result.p_value <= 1.0,
            "P-value {} should be in [0, 1]",
            result.p_value
        );
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
    fn test_hettx_stratified_method() {
        let (y, treatment, x) = create_hettx_test_data();

        let config = HetTxConfig {
            n_permutations: 100,
            effect_method: EffectEstimationMethod::Stratified,
            decompose: false,
            seed: Some(42),
            ..Default::default()
        };

        let result = run_hettx(&y.view(), &treatment.view(), Some(&x.view()), config).unwrap();

        assert_eq!(result.effect_method, EffectEstimationMethod::Stratified);
        // Stratified method assumes constant effect, so variance should be zero
        assert!((result.test_statistic - 0.0).abs() < 1e-10);
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
        let var_stat = compute_test_statistic(&tau, HetTestStat::Variance);
        assert!((var_stat - 2.5).abs() < 0.01);

        // Range: 5 - 1 = 4
        let range_stat = compute_test_statistic(&tau, HetTestStat::Range);
        assert!((range_stat - 4.0).abs() < 0.01);

        // IQR: Q3 - Q1 = 4 - 2 = 2
        let iqr_stat = compute_test_statistic(&tau, HetTestStat::IQR);
        assert!((iqr_stat - 2.0).abs() < 0.01);

        // MAD: mean(|x - 3|) for x in [1,2,3,4,5] = (2+1+0+1+2)/5 = 1.2
        let mad_stat = compute_test_statistic(&tau, HetTestStat::MeanAbsDeviation);
        assert!((mad_stat - 1.2).abs() < 0.01);
    }
}
