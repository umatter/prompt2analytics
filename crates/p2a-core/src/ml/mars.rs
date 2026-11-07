//! Multivariate Adaptive Regression Splines (MARS).
//!
//! MARS is a non-parametric regression technique that automatically discovers
//! nonlinear relationships and interactions using piecewise linear basis functions.
//!
//! # Algorithm Overview
//!
//! MARS builds models of the form:
//!
//! y = beta_0 + sum_{m=1}^{M} beta_m * B_m(x)
//!
//! where each B_m is a product of hinge functions:
//!
//! B_m(x) = prod_{k=1}^{K_m} h_{km}(x_{v(k,m)})
//!
//! ## Hinge Functions
//!
//! The hinge (hockey stick) functions are:
//! - h(x - t) = max(0, x - t)  (right hinge)
//! - h(t - x) = max(0, t - x)  (left hinge)
//!
//! where t is a knot (threshold) value.
//!
//! ## Two-Stage Fitting
//!
//! 1. **Forward Pass**: Greedy addition of basis function pairs that maximize
//!    reduction in residual sum of squares (RSS). Adds both h(x-t) and h(t-x)
//!    to maintain model symmetry.
//!
//! 2. **Backward Pass**: Prunes basis functions using Generalized Cross-Validation
//!    (GCV) criterion to avoid overfitting.
//!
//! ## GCV Criterion
//!
//! GCV = RSS / (n * (1 - d/n)^2), and infinite once d >= n (earth's `GetGcv`; Friedman 1991, eq. 30)
//!
//! where:
//! - RSS = Residual Sum of Squares
//! - n = number of observations
//! - d = effective degrees of freedom (including penalty for knots)
//!
//! ## Fitting
//!
//! Each forward candidate is scored by orthogonalising it against the current model (modified
//! Gram-Schmidt with one re-orthogonalisation pass), so one candidate costs O(n m). A hinge whose
//! remainder is below the relative column tolerance of [`crate::linalg::solve::least_squares`] is
//! aliased: a collinear pair adds only its identified hinge, as earth does. The backward pass does
//! one pivoted QR per pruning step and reads every deletion's RSS from it.
//!
//! The method is not pinned to R (earth is not installed): see `validation/ml/mars.md`.
//!
//! # References
//!
//! - Friedman, J. H. (1991). "Multivariate Adaptive Regression Splines".
//!   *The Annals of Statistics*, 19(1), 1-67.
//!   https://doi.org/10.1214/aos/1176347963
//!   The foundational MARS paper describing the algorithm.
//!
//! - Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
//!   Statistical Learning* (2nd ed.), Section 9.4. Springer.
//!   https://hastie.su.domains/ElemStatLearn/
//!   Accessible treatment of MARS with intuition.
//!
//! - Milborrow, S. (2024). earth: Multivariate Adaptive Regression Splines.
//!   R package version 5.3.3. https://CRAN.R-project.org/package=earth
//!   Reference implementation used for validation.
//!
//! R equivalent: `earth::earth()`

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::linalg::solve::{LM_QR_TOL, LeastSquares, least_squares};
use crate::resource;

/// Configuration for MARS model fitting.
#[derive(Debug, Clone)]
pub struct MarsConfig {
    /// Maximum degree of interaction (1 = additive, 2 = two-way interactions, etc.)
    /// Default: 1 (additive model)
    pub degree: usize,

    /// A cap: the pruned model has at most `nprune` terms (intercept included), chosen by GCV among
    /// the sizes up to `nprune`, as earth. If None, the best GCV of any size.
    /// Default: None
    pub nprune: Option<usize>,

    /// Maximum number of model terms before pruning.
    /// Controls the size of the model during the forward pass.
    /// Default: min(200, max(20, 2 * n_features + n / 10))
    pub nk: Option<usize>,

    /// Threshold for the forward pass: minimum GCV improvement to add a term.
    /// Default: 0.001
    pub thresh: f64,

    /// Minimum number of observations between knots (per predictor).
    /// Default: 0 (auto-calculate based on n)
    pub minspan: Option<usize>,

    /// Minimum number of observations before first and after last knot.
    /// Default: 0 (auto-calculate based on n)
    pub endspan: Option<usize>,

    /// Penalty per knot when calculating GCV.
    /// d = 1 recovers AIC; d = 2 recovers BIC-like penalty.
    /// Default: 3.0 (Friedman's recommended value)
    pub penalty: f64,

    /// Use fast MARS algorithm (subsampling for knot search).
    /// Default: true for n > 200
    pub fast: Option<bool>,

    /// Number of candidate knots to consider per variable in fast mode.
    /// Default: 20
    pub fast_k: usize,

    /// Include an intercept term. `false`: no constant column, and R-squared is uncentred, as in
    /// `lm(y ~ 0 + ...)`.
    /// Default: true
    pub intercept: bool,
}

impl Default for MarsConfig {
    fn default() -> Self {
        MarsConfig {
            degree: 1,
            nprune: None,
            nk: None,
            thresh: 0.001,
            minspan: None,
            endspan: None,
            penalty: 3.0,
            fast: None,
            fast_k: 20,
            intercept: true,
        }
    }
}

/// Type of hinge function in a basis function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HingeType {
    /// h(x - t) = max(0, x - t) - positive/right hinge
    Positive,
    /// h(t - x) = max(0, t - x) - negative/left hinge
    Negative,
}

impl std::fmt::Display for HingeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HingeType::Positive => write!(f, "+"),
            HingeType::Negative => write!(f, "-"),
        }
    }
}

/// A single hinge function component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HingeFunction {
    /// Variable index (0-based)
    pub variable: usize,
    /// Knot location
    pub knot: f64,
    /// Type of hinge
    pub hinge_type: HingeType,
    /// Optional variable name
    pub variable_name: Option<String>,
}

impl HingeFunction {
    /// Evaluate the hinge function at a given x value.
    pub fn evaluate(&self, x: f64) -> f64 {
        match self.hinge_type {
            HingeType::Positive => (x - self.knot).max(0.0),
            HingeType::Negative => (self.knot - x).max(0.0),
        }
    }
}

impl std::fmt::Display for HingeFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let var_name = self
            .variable_name
            .clone()
            .unwrap_or_else(|| format!("x{}", self.variable));
        match self.hinge_type {
            HingeType::Positive => write!(f, "h({} - {:.4})", var_name, self.knot),
            HingeType::Negative => write!(f, "h({:.4} - {})", self.knot, var_name),
        }
    }
}

/// A basis function (potentially a product of hinge functions).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasisFunction {
    /// Component hinge functions (empty for intercept)
    pub hinges: Vec<HingeFunction>,
    /// Index of parent basis function (if this is an interaction term)
    pub parent: Option<usize>,
}

impl BasisFunction {
    /// Create intercept basis function.
    pub fn intercept() -> Self {
        BasisFunction {
            hinges: Vec::new(),
            parent: None,
        }
    }

    /// Create a single-hinge basis function.
    pub fn single(hinge: HingeFunction) -> Self {
        BasisFunction {
            hinges: vec![hinge],
            parent: None,
        }
    }

    /// Create an interaction term from parent and new hinge.
    pub fn interaction(
        parent_idx: usize,
        parent_hinges: &[HingeFunction],
        new_hinge: HingeFunction,
    ) -> Self {
        let mut hinges = parent_hinges.to_vec();
        hinges.push(new_hinge);
        BasisFunction {
            hinges,
            parent: Some(parent_idx),
        }
    }

    /// Evaluate the basis function for a single observation.
    pub fn evaluate(&self, x: &ArrayView1<f64>) -> f64 {
        if self.hinges.is_empty() {
            return 1.0; // Intercept
        }
        self.hinges
            .iter()
            .map(|h| h.evaluate(x[h.variable]))
            .product()
    }

    /// Evaluate the basis function for multiple observations.
    pub fn evaluate_all(&self, x: &ArrayView2<f64>) -> Array1<f64> {
        let n = x.nrows();
        if self.hinges.is_empty() {
            return Array1::ones(n); // Intercept
        }
        let mut result = Array1::ones(n);
        for hinge in &self.hinges {
            for i in 0..n {
                result[i] *= hinge.evaluate(x[[i, hinge.variable]]);
            }
        }
        result
    }

    /// Get the degree (number of hinges) of this basis function.
    pub fn degree(&self) -> usize {
        self.hinges.len()
    }

    /// Check if this basis function uses a particular variable.
    pub fn uses_variable(&self, var: usize) -> bool {
        self.hinges.iter().any(|h| h.variable == var)
    }
}

impl std::fmt::Display for BasisFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.hinges.is_empty() {
            write!(f, "(Intercept)")
        } else {
            let parts: Vec<String> = self.hinges.iter().map(|h| h.to_string()).collect();
            write!(f, "{}", parts.join(" * "))
        }
    }
}

/// Result of MARS model fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarsResult {
    /// Basis functions in the final model (after pruning)
    pub basis_functions: Vec<BasisFunction>,

    /// Regression coefficients (one per basis function)
    pub coefficients: Vec<f64>,

    /// GCV value of the final model
    pub gcv: f64,

    /// RSS (residual sum of squares) of the final model
    pub rss: f64,

    /// R-squared
    pub r_squared: f64,

    /// Adjusted R-squared
    pub r_squared_adj: f64,

    /// Variable importance scores (sum of |coefficient| x standard deviation of the basis function)
    pub variable_importance: Vec<f64>,

    /// Feature names (if provided)
    pub feature_names: Option<Vec<String>>,

    /// Knot locations per variable (variable index -> sorted knots)
    #[serde(skip)]
    pub cuts: HashMap<usize, Vec<f64>>,

    /// Fitted values
    #[serde(skip)]
    pub fitted_values: Vec<f64>,

    /// Residuals
    #[serde(skip)]
    pub residuals: Vec<f64>,

    /// Number of observations
    pub n_obs: usize,

    /// Number of features
    pub n_features: usize,

    /// Configuration used
    #[serde(skip)]
    pub config: MarsConfig,

    /// GCV history during backward pass (for diagnostics)
    pub gcv_history: Vec<(usize, f64)>,
}

impl MarsResult {
    /// Predict for new data.
    pub fn predict(&self, x: &ArrayView2<f64>) -> Array1<f64> {
        let n = x.nrows();
        let mut predictions = Array1::zeros(n);

        for (basis, &coef) in self.basis_functions.iter().zip(self.coefficients.iter()) {
            let basis_vals = basis.evaluate_all(x);
            predictions = predictions + coef * &basis_vals;
        }

        predictions
    }

    /// Get the effective degrees of freedom.
    pub fn effective_df(&self) -> f64 {
        let n_terms = self.basis_functions.len();
        let n_knots: usize = self.basis_functions.iter().map(|b| b.hinges.len()).sum();
        // Effective df = number of terms + penalty * number of knots / 2
        // (divided by 2 because hinges come in pairs)
        n_terms as f64 + self.config.penalty * (n_knots as f64) / 2.0
    }
}

impl std::fmt::Display for MarsResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "MARS Model Results")?;
        writeln!(f, "==================")?;
        writeln!(f)?;
        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(f, "Features: {}", self.n_features)?;
        writeln!(f, "Terms: {}", self.basis_functions.len())?;
        writeln!(f)?;
        writeln!(f, "Fit Statistics:")?;
        writeln!(f, "  RSS:            {:.4}", self.rss)?;
        writeln!(f, "  GCV:            {:.4}", self.gcv)?;
        writeln!(f, "  R-squared:      {:.4}", self.r_squared)?;
        writeln!(f, "  Adj R-squared:  {:.4}", self.r_squared_adj)?;
        writeln!(f)?;
        writeln!(f, "Basis Functions:")?;
        writeln!(f, "{:-<60}", "")?;
        writeln!(f, "{:>10}  {}", "Coef", "Term")?;
        writeln!(f, "{:-<60}", "")?;

        for (i, (basis, &coef)) in self
            .basis_functions
            .iter()
            .zip(self.coefficients.iter())
            .enumerate()
        {
            if i < 20 || i == self.basis_functions.len() - 1 {
                writeln!(f, "{:10.4}  {}", coef, basis)?;
            } else if i == 20 {
                writeln!(f, "  ... ({} more terms)", self.basis_functions.len() - 21)?;
            }
        }

        writeln!(f)?;
        writeln!(f, "Variable Importance:")?;
        writeln!(f, "{:-<40}", "")?;

        // Sort by importance
        let mut importance_with_names: Vec<(String, f64)> = self
            .variable_importance
            .iter()
            .enumerate()
            .map(|(i, &imp)| {
                let name = self
                    .feature_names
                    .as_ref()
                    .and_then(|names| names.get(i).cloned())
                    .unwrap_or_else(|| format!("x{}", i));
                (name, imp)
            })
            .collect();

        importance_with_names.sort_by(|a, b| b.1.total_cmp(&a.1));

        // Normalize to 100
        let max_imp = importance_with_names
            .iter()
            .map(|(_, imp)| *imp)
            .fold(f64::NEG_INFINITY, f64::max);

        for (name, imp) in importance_with_names.iter().take(15) {
            let normalized = if max_imp > 0.0 {
                *imp / max_imp * 100.0
            } else {
                0.0
            };
            writeln!(f, "  {:20} {:6.1}", name, normalized)?;
        }

        if importance_with_names.len() > 15 {
            writeln!(
                f,
                "  ... ({} more variables)",
                importance_with_names.len() - 15
            )?;
        }

        Ok(())
    }
}

/// Fit a MARS model.
///
/// # Arguments
/// * `x` - Feature matrix (n_obs x n_features)
/// * `y` - Response vector (length n_obs)
/// * `config` - MARS configuration
/// * `feature_names` - Optional feature names
///
/// # Returns
/// A `MarsResult` containing the fitted model.
///
/// # Example
/// ```
/// use ndarray::{Array1, Array2};
/// use p2a_core::ml::{MarsConfig, mars};
///
/// // Piecewise-linear response with a kink at x = 5:
/// //   y = x        for x < 5
/// //   y = 2x - 5   for x >= 5
/// // plus a small deterministic wobble so the fit is not exact.
/// let n = 100;
/// let mut x_data = Vec::with_capacity(n);
/// let mut y_data = Vec::with_capacity(n);
/// for i in 0..n {
///     let xi = i as f64 / 10.0;
///     let yi = if xi < 5.0 { xi } else { 2.0 * xi - 5.0 };
///     x_data.push(xi);
///     y_data.push(yi + 0.05 * (i as f64 % 5.0 - 2.0));
/// }
/// let x = Array2::from_shape_vec((n, 1), x_data).unwrap();
/// let y = Array1::from_vec(y_data);
///
/// let result = mars(
///     x.view(),
///     y.view(),
///     MarsConfig::default(),
///     Some(vec!["x".to_string()]),
/// )
/// .unwrap();
///
/// // The hinge basis reproduces the kink, so the fit is near-perfect.
/// assert!(result.r_squared > 0.95, "R-squared = {}", result.r_squared);
/// assert_eq!(result.n_obs, 100);
/// assert_eq!(result.n_features, 1);
///
/// // The forward pass places a knot at the break point.
/// let knots = result.cuts.get(&0).expect("a knot on the only predictor");
/// assert!(
///     knots.iter().any(|&k| (k - 5.0).abs() < 1.0),
///     "knots = {:?}",
///     knots
/// );
///
/// // Predict at a new point: the true value at x = 5.5 is 2 * 5.5 - 5 = 6.0.
/// let x_new = Array2::from_shape_vec((1, 1), vec![5.5]).unwrap();
/// let pred = result.predict(&x_new.view());
/// assert!((pred[0] - 6.0).abs() < 1.0, "prediction = {}", pred[0]);
/// ```
///
/// # References
///
/// - Friedman, J. H. (1991). "Multivariate Adaptive Regression Splines".
///   *The Annals of Statistics*, 19(1), 1-67.
pub fn mars(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    config: MarsConfig,
    feature_names: Option<Vec<String>>,
) -> Result<MarsResult, String> {
    validate_config(&x, &y, &config, feature_names.as_deref())?;
    let n = x.nrows();
    let p = x.ncols();

    // Compute y statistics
    let (null_rss, constant) = if config.intercept {
        let y_mean = y.mean().unwrap_or(0.0);
        (
            y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>(),
            y.iter().all(|&yi| yi == y[0]),
        )
    } else {
        (
            y.iter().map(|&yi| yi * yi).sum::<f64>(),
            y.iter().all(|&yi| yi == 0.0),
        )
    };
    if constant {
        return Err("Response variable has zero variance".to_string());
    }

    // Determine algorithm parameters
    let nk = config.nk.unwrap_or_else(|| {
        let default_nk = (2 * p + n / 10).max(20).min(200);
        default_nk
    });

    let use_fast = config.fast.unwrap_or(n > 200);
    let minspan = config.minspan.unwrap_or_else(|| compute_minspan(n));
    let endspan = config
        .endspan
        .unwrap_or_else(|| compute_endspan(n).min((n - 1) / 2));

    // Find candidate knots for each variable
    let candidate_knots = find_candidate_knots(&x, minspan, endspan, use_fast, config.fast_k);

    // Forward pass: build up model by adding basis functions
    let (forward_basis, forward_columns) = forward_pass(
        &x,
        &y,
        &candidate_knots,
        nk,
        config.degree,
        config.thresh,
        &config,
    )?;
    if forward_basis.is_empty() {
        return Err(
            "MARS: no basis function improves on the empty model and intercept = false".to_string(),
        );
    }

    // Backward pass: prune using GCV
    let (selected, gcv_history) = backward_pass(
        &y,
        forward_basis,
        forward_columns,
        config.nprune,
        config.penalty,
    )?;
    let BasisFit {
        basis: final_basis,
        coefficients: final_coefs,
        fitted,
        rss,
        gcv,
    } = selected;

    let residuals: Vec<f64> = y
        .iter()
        .zip(fitted.iter())
        .map(|(&yi, &fi)| yi - fi)
        .collect();

    let r_squared = 1.0 - rss / null_rss;
    let n_terms = final_basis.len();
    let df_residual = n.checked_sub(n_terms).filter(|&d| d > 0).ok_or_else(|| {
        format!("MARS: no residual degrees of freedom (n = {n}, terms = {n_terms})")
    })?;
    let df_null = if config.intercept { n - 1 } else { n };
    let r_squared_adj = 1.0 - (1.0 - r_squared) * (df_null as f64) / (df_residual as f64);

    // Variable importance
    let variable_importance = compute_variable_importance(&x, &final_basis, &final_coefs, p);

    // Collect knot locations
    let mut cuts: HashMap<usize, Vec<f64>> = HashMap::new();
    for basis in &final_basis {
        for hinge in &basis.hinges {
            cuts.entry(hinge.variable).or_default().push(hinge.knot);
        }
    }
    for knots in cuts.values_mut() {
        knots.sort_by(f64::total_cmp);
        knots.dedup();
    }

    Ok(MarsResult {
        basis_functions: final_basis,
        coefficients: final_coefs,
        gcv,
        rss,
        r_squared,
        r_squared_adj,
        variable_importance,
        feature_names,
        cuts,
        fitted_values: fitted,
        residuals,
        n_obs: n,
        n_features: p,
        config,
        gcv_history,
    })
}

/// Every parameter against what the data can support, before any work.
fn validate_config(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    config: &MarsConfig,
    feature_names: Option<&[String]>,
) -> Result<(), String> {
    let (n, p) = x.dim();
    if y.len() != n {
        return Err(format!("y has length {} but x has {} rows", y.len(), n));
    }
    if n < 3 {
        return Err("Need at least 3 observations for MARS".to_string());
    }
    if p == 0 {
        return Err("MARS: need at least one predictor".to_string());
    }
    if x.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
        return Err("MARS: x and y must not contain non-finite values".to_string());
    }
    if config.degree == 0 {
        return Err("MARS: degree must be at least 1, got 0".to_string());
    }
    if config.nprune == Some(0) {
        return Err("MARS: nprune must be at least 1, got 0".to_string());
    }
    if config.nk == Some(0) {
        return Err("MARS: nk must be at least 1, got 0".to_string());
    }
    if let Some(k) = config.minspan {
        if k == 0 || k > n {
            return Err(format!(
                "MARS: minspan must be between 1 and n = {n}, got {k}"
            ));
        }
    }
    if let Some(k) = config.endspan {
        if k == 0 || k.saturating_mul(2) >= n {
            return Err(format!(
                "MARS: endspan must be at least 1 and less than n/2 (n = {n}), got {k}"
            ));
        }
    }
    if !(config.penalty.is_finite() && config.penalty >= 0.0) {
        return Err(format!(
            "MARS: penalty must be finite and non-negative, got {}",
            config.penalty
        ));
    }
    if !(config.thresh.is_finite() && config.thresh >= 0.0) {
        return Err(format!(
            "MARS: thresh must be finite and non-negative, got {}",
            config.thresh
        ));
    }
    if config.fast_k == 0 {
        return Err("MARS: fast_k must be at least 1, got 0".to_string());
    }
    if let Some(names) = feature_names {
        if names.len() != p {
            return Err(format!(
                "MARS: feature_names has {} names for {p} predictors",
                names.len()
            ));
        }
    }
    Ok(())
}

/// Compute minimum span between knots.
/// Formula from Friedman (1991): floor(-log2(-1/n * log(1-0.05)) / 2.5)
fn compute_minspan(n: usize) -> usize {
    if n <= 10 {
        return 1;
    }
    let term = -1.0 / (n as f64) * (1.0 - 0.05_f64).ln();
    let span = (-term.ln() / 2.5_f64.ln()).floor() as usize;
    span.max(1)
}

/// Compute end span (same formula as minspan in most implementations).
fn compute_endspan(n: usize) -> usize {
    compute_minspan(n)
}

/// Find candidate knot locations for each variable.
fn find_candidate_knots(
    x: &ArrayView2<f64>,
    minspan: usize,
    endspan: usize,
    use_fast: bool,
    fast_k: usize,
) -> Vec<Vec<f64>> {
    let n = x.nrows();
    let p = x.ncols();
    let mut knots = Vec::with_capacity(p);

    for j in 0..p {
        // Get sorted unique values for this variable
        let mut col_vals: Vec<f64> = x.column(j).iter().copied().collect();
        col_vals.sort_by(f64::total_cmp);

        // Apply endspan: skip first and last endspan observations
        let start = endspan;
        let end = n - endspan;

        let valid_range: Vec<f64> = col_vals[start..end].to_vec();

        if valid_range.is_empty() {
            knots.push(Vec::new());
            continue;
        }

        // Apply minspan: space out knots
        let mut selected: Vec<f64> = Vec::new();
        let mut last_idx: Option<usize> = None;

        for (i, &val) in valid_range.iter().enumerate() {
            match last_idx {
                None => {
                    selected.push(val);
                    last_idx = Some(i);
                }
                Some(li) if i >= li + minspan => {
                    // Check if value is different from last selected
                    if (val - *selected.last().unwrap()).abs() > 1e-10 {
                        selected.push(val);
                        last_idx = Some(i);
                    }
                }
                _ => {}
            }
        }

        // Subsample if using fast mode
        if use_fast && selected.len() > fast_k {
            let step = selected.len() / fast_k;
            selected = selected
                .into_iter()
                .step_by(step.max(1))
                .take(fast_k)
                .collect();
        }

        knots.push(selected);
    }

    knots
}

/// One candidate pair against the current model.
#[derive(Debug, Clone, Copy, PartialEq)]
struct PairScore {
    /// RSS with the identified hinges of the pair added
    rss: f64,
    keep_pos: bool,
    keep_neg: bool,
}

/// An orthonormal basis `q` of the identified columns of the current model and the residual `r` of
/// `y` on it.
struct ForwardState {
    q: Vec<Vec<f64>>,
    r: Vec<f64>,
    rss: f64,
}

/// Remove the component of `v` in the span of `q` (modified Gram-Schmidt).
fn project_out(q: &[Vec<f64>], v: &mut [f64]) {
    for col in q {
        let d = dot(col, v);
        for (vi, ci) in v.iter_mut().zip(col) {
            *vi -= d * ci;
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

impl ForwardState {
    fn new(y: &ArrayView1<f64>) -> Result<Self, String> {
        let mut r = resource::try_vec(y.len(), 0.0, "MARS residuals").map_err(|e| e.to_string())?;
        for (ri, yi) in r.iter_mut().zip(y.iter()) {
            *ri = *yi;
        }
        let rss = dot(&r, &r);
        Ok(ForwardState {
            q: Vec::new(),
            r,
            rss,
        })
    }

    /// `u` becomes the unit remainder of `b` off the model and off `prior`; false when `b` is
    /// aliased: the remainder is at most `LM_QR_TOL` of its norm, the column rule of the pivoted QR.
    fn identify(&self, b: &[f64], prior: Option<&[f64]>, u: &mut [f64]) -> bool {
        let norm_b = dot(b, b).sqrt();
        u.copy_from_slice(b);
        for _ in 0..2 {
            project_out(&self.q, u);
            if let Some(q1) = prior {
                let d = dot(q1, u);
                for (ui, ci) in u.iter_mut().zip(q1) {
                    *ui -= d * ci;
                }
            }
        }
        let norm_u = dot(u, u).sqrt();
        if !(norm_u > LM_QR_TOL * norm_b) {
            return false;
        }
        u.iter_mut().for_each(|v| *v /= norm_u);
        true
    }

    /// The RSS with `pos` and `neg` added, in O(n m). `u1` and `u2` are work space of length n.
    fn score(&self, pos: &[f64], neg: &[f64], u1: &mut [f64], u2: &mut [f64]) -> PairScore {
        let mut rss = self.rss;
        let keep_pos = self.identify(pos, None, u1);
        if keep_pos {
            rss -= dot(u1, &self.r).powi(2);
        }
        let keep_neg = self.identify(neg, keep_pos.then_some(&*u1), u2);
        if keep_neg {
            rss -= dot(u2, &self.r).powi(2);
        }
        PairScore {
            rss: rss.max(0.0),
            keep_pos,
            keep_neg,
        }
    }

    /// Append an identified column; false (and no change) when it is aliased.
    fn accept(&mut self, b: &[f64]) -> Result<bool, String> {
        let mut u = resource::try_vec(b.len(), 0.0, "MARS basis").map_err(|e| e.to_string())?;
        if !self.identify(b, None, &mut u) {
            return Ok(false);
        }
        let d = dot(&u, &self.r);
        for (ri, ui) in self.r.iter_mut().zip(&u) {
            *ri -= d * ui;
        }
        self.rss = dot(&self.r, &self.r);
        self.q.push(u);
        Ok(true)
    }
}

/// Forward pass: greedily add basis functions. Returns them with their evaluated columns.
fn forward_pass(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    candidate_knots: &[Vec<f64>],
    max_terms: usize,
    max_degree: usize,
    thresh: f64,
    config: &MarsConfig,
) -> Result<(Vec<BasisFunction>, Vec<Vec<f64>>), String> {
    let n = x.nrows();
    let p = x.ncols();
    let alloc = |what: &str, fill: f64| resource::try_vec(n, fill, what).map_err(|e| e.to_string());

    let mut state = ForwardState::new(y)?;
    let mut basis_functions: Vec<BasisFunction> = Vec::new();
    let mut columns: Vec<Vec<f64>> = Vec::new();
    let mut n_knots = 0usize;
    // Without an intercept the constant is still a parent of the first hinges, outside the design.
    let ones = alloc("MARS basis", 1.0)?;
    if config.intercept {
        basis_functions.push(BasisFunction::intercept());
        state.accept(&ones)?;
        columns.push(ones.clone());
    }
    let mut current_gcv = compute_gcv(
        state.rss,
        n,
        effective_df(basis_functions.len(), n_knots, config.penalty),
    );

    let (mut pos, mut neg) = (alloc("MARS basis", 0.0)?, alloc("MARS basis", 0.0)?);
    let (mut u1, mut u2) = (alloc("MARS basis", 0.0)?, alloc("MARS basis", 0.0)?);
    let mut xj = alloc("MARS basis", 0.0)?;

    // Iteratively add basis function pairs
    while basis_functions.len() < max_terms {
        let mut best_gcv_reduction = 0.0;
        // (parent's index in the basis, variable, knot, what the pair keeps)
        let mut best: Option<(Option<usize>, usize, f64, PairScore)> = None;

        let first_parent = usize::from(!config.intercept);
        for parent_pos in 0..basis_functions.len() + first_parent {
            let parent = parent_pos.checked_sub(first_parent);
            let (parent_col, parent_hinges): (&[f64], &[HingeFunction]) = match parent {
                Some(i) => (&columns[i], &basis_functions[i].hinges),
                None => (&ones, &[]),
            };
            let parent_degree = parent_hinges.len();

            // Check if we can add another hinge to this parent
            if parent_degree >= max_degree {
                continue;
            }

            // Try each variable not already in parent
            for j in 0..p {
                if parent_hinges.iter().any(|h| h.variable == j) {
                    continue;
                }
                resource::checkpoint().map_err(|e| e.to_string())?;
                for (slot, v) in xj.iter_mut().zip(x.column(j).iter()) {
                    *slot = *v;
                }

                // Try each knot
                for &knot in &candidate_knots[j] {
                    for i in 0..n {
                        pos[i] = parent_col[i] * (xj[i] - knot).max(0.0);
                        neg[i] = parent_col[i] * (knot - xj[i]).max(0.0);
                    }
                    let score = state.score(&pos, &neg, &mut u1, &mut u2);
                    let added = usize::from(score.keep_pos) + usize::from(score.keep_neg);
                    if added == 0 {
                        continue;
                    }
                    let test_df = effective_df(
                        basis_functions.len() + added,
                        n_knots + added * (parent_degree + 1),
                        config.penalty,
                    );
                    let gcv_reduction = current_gcv - compute_gcv(score.rss, n, test_df);
                    if gcv_reduction > best_gcv_reduction {
                        best_gcv_reduction = gcv_reduction;
                        best = Some((parent, j, knot, score));
                    }
                }
            }
        }

        // Check if improvement meets threshold
        let relative_improvement = best_gcv_reduction / (current_gcv + 1e-10);
        let Some((parent, j, knot, score)) = best else {
            break;
        };
        if relative_improvement < thresh {
            break;
        }

        // Add the identified hinges of the best pair
        for (keep, hinge_type) in [
            (score.keep_pos, HingeType::Positive),
            (score.keep_neg, HingeType::Negative),
        ] {
            if !keep {
                continue;
            }
            let hinge = HingeFunction {
                variable: j,
                knot,
                hinge_type,
                variable_name: None,
            };
            let parent_col: &[f64] = parent.map_or(&ones, |i| &columns[i]);
            let mut column = alloc("MARS basis", 0.0)?;
            for i in 0..n {
                column[i] = parent_col[i] * hinge.evaluate(x[[i, j]]);
            }
            if !state.accept(&column)? {
                continue; // rounding moved it across the rule; the model stays identified
            }
            let term = match parent {
                Some(i) if !basis_functions[i].hinges.is_empty() => {
                    let hinges = basis_functions[i].hinges.clone();
                    BasisFunction::interaction(i, &hinges, hinge)
                }
                _ => BasisFunction::single(hinge),
            };
            n_knots += term.hinges.len();
            basis_functions.push(term);
            columns.push(column);
        }
        current_gcv = compute_gcv(
            state.rss,
            n,
            effective_df(basis_functions.len(), n_knots, config.penalty),
        );
    }

    Ok((basis_functions, columns))
}

/// The model the backward pass chose, with the statistics of its one fit: `gcv_history`, the result
/// and `predict` agree exactly.
struct BasisFit {
    basis: Vec<BasisFunction>,
    coefficients: Vec<f64>,
    fitted: Vec<f64>,
    rss: f64,
    gcv: f64,
}

/// The RSS increase from deleting each identified column of a fit: `beta_j^2 / (X'X)^-1_jj`. An
/// aliased column costs 0.
fn deletion_costs(fit: &LeastSquares) -> Vec<f64> {
    (0..fit.coefficients.len())
        .map(|j| {
            if fit.aliased.contains(&j) {
                0.0
            } else {
                fit.coefficients[j].powi(2) / fit.unscaled_cov[[j, j]]
            }
        })
        .collect()
}

fn fit_columns(y: &ArrayView1<f64>, columns: &[Vec<f64>]) -> Result<LeastSquares, String> {
    let mut design =
        resource::try_zeros2(y.len(), columns.len(), "MARS design").map_err(|e| e.to_string())?;
    for (mut slot, column) in design.columns_mut().into_iter().zip(columns) {
        for (d, v) in slot.iter_mut().zip(column) {
            *d = *v;
        }
    }
    least_squares(design.view(), *y).map_err(|e| format!("MARS: {e}"))
}

/// Backward pass: prune to one term, recording the GCV of every size, and return the best model of
/// at most `nprune` terms (ties to the smaller). One QR per step.
fn backward_pass(
    y: &ArrayView1<f64>,
    mut basis: Vec<BasisFunction>,
    mut columns: Vec<Vec<f64>>,
    nprune: Option<usize>,
    penalty: f64,
) -> Result<(BasisFit, Vec<(usize, f64)>), String> {
    let n = y.len();
    let cap = nprune.unwrap_or(usize::MAX);
    let mut gcv_history = Vec::with_capacity(basis.len());
    let mut best: Option<BasisFit> = None;

    loop {
        resource::checkpoint().map_err(|e| e.to_string())?;
        let fit = fit_columns(y, &columns)?;
        let identified = fit.aliased.is_empty();
        // On a full-rank fit the RSS comes from the coefficients, as predict computes it.
        let mut fitted = vec![0.0; if identified { n } else { 0 }];
        if identified {
            for (column, &c) in columns.iter().zip(fit.coefficients.iter()) {
                for (f, v) in fitted.iter_mut().zip(column) {
                    *f += c * v;
                }
            }
        }
        let rss: f64 = if identified {
            y.iter().zip(&fitted).map(|(a, b)| (a - b).powi(2)).sum()
        } else {
            fit.residuals.iter().map(|e| e * e).sum()
        };
        let gcv = compute_gcv(rss, n, compute_effective_df(&basis, penalty));
        gcv_history.push((basis.len(), gcv));

        // An aliased size is never chosen: the same RSS with the column gone has a lower GCV.
        if identified && basis.len() <= cap && best.as_ref().is_none_or(|b| gcv <= b.gcv) {
            best = Some(BasisFit {
                basis: basis.clone(),
                coefficients: fit.coefficients.to_vec(),
                fitted,
                rss,
                gcv,
            });
        }
        if basis.len() == 1 {
            break;
        }

        // The intercept is never removed; aliased columns go first.
        let first = usize::from(basis[0].hinges.is_empty());
        let costs = deletion_costs(&fit);
        let removable_aliased = fit.aliased.iter().any(|&j| j >= first);
        let mut remove: Option<(usize, f64)> = None;
        for j in first..basis.len() {
            if removable_aliased && !fit.aliased.contains(&j) {
                continue;
            }
            let term = &basis[j];
            let knots: usize = basis.iter().map(|b| b.hinges.len()).sum();
            let d = effective_df(basis.len() - 1, knots - term.hinges.len(), penalty);
            let test = compute_gcv(rss + costs[j], n, d);
            if remove.is_none_or(|(_, g)| test < g) {
                remove = Some((j, test));
            }
        }
        let Some((j, _)) = remove else {
            break;
        };
        basis.remove(j);
        columns.remove(j);
    }

    let selected = best.ok_or_else(|| {
        "MARS: no model of at most nprune terms has a finite fit; rescale the data".to_string()
    })?;
    Ok((selected, gcv_history))
}

/// Effective degrees of freedom: d = number_of_terms + penalty * number_of_knots / 2
/// (hinges come in pairs, so divide by 2).
fn effective_df(n_terms: usize, n_knots: usize, penalty: f64) -> f64 {
    n_terms as f64 + penalty * n_knots as f64 / 2.0
}

/// Effective degrees of freedom of a basis.
fn compute_effective_df(basis: &[BasisFunction], penalty: f64) -> f64 {
    let n_knots = basis.iter().map(|b| b.hinges.len()).sum();
    effective_df(basis.len(), n_knots, penalty)
}

/// GCV = RSS / (n * (1 - d/n)^2), infinite once d >= n: earth's `GetGcv`; Friedman (1991), eq. 30.
fn compute_gcv(rss: f64, n: usize, effective_df: f64) -> f64 {
    let n_f = n as f64;
    if !(effective_df < n_f) {
        return f64::INFINITY;
    }
    rss / (n_f * (1.0 - effective_df / n_f).powi(2))
}

/// Variable importance: sum of |coefficient| x standard deviation of the basis function.
fn compute_variable_importance(
    x: &ArrayView2<f64>,
    basis: &[BasisFunction],
    coefs: &[f64],
    n_vars: usize,
) -> Vec<f64> {
    let n = x.nrows();
    let mut importance = vec![0.0; n_vars];

    for (b, &c) in basis.iter().zip(coefs.iter()) {
        if b.hinges.is_empty() {
            continue; // Skip intercept
        }

        // Compute variance of this basis function
        let vals = b.evaluate_all(x);
        let mean = vals.sum() / n as f64;
        let var: f64 = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n as f64;

        // Contribution = |coef| * sqrt(variance)
        let contrib = c.abs() * var.sqrt();

        // Distribute to all variables involved
        for hinge in &b.hinges {
            importance[hinge.variable] += contrib;
        }
    }

    importance
}

/// Run MARS (convenience wrapper).
///
/// # Arguments
/// * `x` - Feature matrix
/// * `y` - Response vector
/// * `degree` - Maximum interaction degree (default: 1)
/// * `nprune` - A cap: the pruned model has at most nprune terms (intercept included), chosen by GCV
///   among sizes <= nprune, as earth (default: the best GCV of any size)
/// * `thresh` - GCV improvement threshold (default: 0.001)
pub fn run_mars(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    degree: Option<usize>,
    nprune: Option<usize>,
    thresh: Option<f64>,
    feature_names: Option<Vec<String>>,
) -> Result<MarsResult, String> {
    let config = MarsConfig {
        degree: degree.unwrap_or(1),
        nprune,
        thresh: thresh.unwrap_or(0.001),
        ..Default::default()
    };
    mars(x, y, config, feature_names)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_hinge_function() {
        let hinge_pos = HingeFunction {
            variable: 0,
            knot: 5.0,
            hinge_type: HingeType::Positive,
            variable_name: None,
        };

        assert_relative_eq!(hinge_pos.evaluate(3.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(hinge_pos.evaluate(5.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(hinge_pos.evaluate(7.0), 2.0, epsilon = 1e-10);

        let hinge_neg = HingeFunction {
            variable: 0,
            knot: 5.0,
            hinge_type: HingeType::Negative,
            variable_name: None,
        };

        assert_relative_eq!(hinge_neg.evaluate(3.0), 2.0, epsilon = 1e-10);
        assert_relative_eq!(hinge_neg.evaluate(5.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(hinge_neg.evaluate(7.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_basis_function_intercept() {
        let intercept = BasisFunction::intercept();
        assert_eq!(intercept.degree(), 0);

        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let vals = intercept.evaluate_all(&x.view());
        assert_relative_eq!(vals[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(vals[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mars_linear() {
        // Simple linear relationship: y = 2*x1 + 3*x2 + noise
        let n = 50;
        let mut x_data = Vec::with_capacity(n * 2);
        let mut y_data = Vec::with_capacity(n);

        for i in 0..n {
            let x1 = i as f64 / 10.0;
            let x2 = (i as f64 / 10.0).sin();
            x_data.push(x1);
            x_data.push(x2);
            y_data.push(2.0 * x1 + 3.0 * x2 + 0.1 * (i as f64 % 3.0 - 1.0));
        }

        let x = Array2::from_shape_vec((n, 2), x_data).unwrap();
        let y = Array1::from_vec(y_data);

        let config = MarsConfig {
            degree: 1,
            thresh: 0.001,
            ..Default::default()
        };

        let result = mars(x.view(), y.view(), config, None).unwrap();

        // Should have decent fit
        assert!(result.r_squared > 0.9, "R-squared = {}", result.r_squared);
        assert!(result.gcv > 0.0);
        assert!(!result.basis_functions.is_empty());
    }

    #[test]
    fn test_mars_nonlinear() {
        // Nonlinear relationship with a kink
        let n = 100;
        let mut x_data = Vec::with_capacity(n);
        let mut y_data = Vec::with_capacity(n);

        for i in 0..n {
            let x = i as f64 / 10.0;
            x_data.push(x);
            // Piecewise linear: y = x for x < 5, y = 2*x - 5 for x >= 5
            let y = if x < 5.0 { x } else { 2.0 * x - 5.0 };
            y_data.push(y + 0.05 * (i as f64 % 5.0 - 2.0)); // Small noise
        }

        let x = Array2::from_shape_vec((n, 1), x_data).unwrap();
        let y = Array1::from_vec(y_data);

        let result = mars(x.view(), y.view(), MarsConfig::default(), None).unwrap();

        // Should fit well (MARS is designed for this)
        assert!(result.r_squared > 0.95, "R-squared = {}", result.r_squared);

        // Should find knot around x = 5
        let has_knot_near_5 = result
            .cuts
            .get(&0)
            .map_or(false, |knots| knots.iter().any(|&k| (k - 5.0).abs() < 1.0));
        assert!(has_knot_near_5, "Should find knot near x=5");
    }

    #[test]
    fn test_mars_prediction() {
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0]
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let result = mars(x.view(), y.view(), MarsConfig::default(), None).unwrap();

        // Predict on training data
        let preds = result.predict(&x.view());
        assert_eq!(preds.len(), 10);

        // Predictions should be close to actual
        for (pred, &actual) in preds.iter().zip(y.iter()) {
            assert!(
                (pred - actual).abs() < 1.0,
                "pred={}, actual={}",
                pred,
                actual
            );
        }

        // Predict on new data
        let x_new = array![[5.5], [6.5]];
        let new_preds = result.predict(&x_new.view());
        assert_eq!(new_preds.len(), 2);
    }

    #[test]
    fn test_mars_with_interactions() {
        // y = x1 + x2 + x1*x2
        let n = 50;
        let mut x_data = Vec::with_capacity(n * 2);
        let mut y_data = Vec::with_capacity(n);

        for i in 0..n {
            let x1 = (i % 10) as f64;
            let x2 = (i / 10) as f64;
            x_data.push(x1);
            x_data.push(x2);
            y_data.push(x1 + x2 + 0.5 * x1 * x2 + 0.1 * (i as f64 % 3.0));
        }

        let x = Array2::from_shape_vec((n, 2), x_data).unwrap();
        let y = Array1::from_vec(y_data);

        let config = MarsConfig {
            degree: 2, // Allow interactions
            ..Default::default()
        };

        let result = mars(x.view(), y.view(), config, None).unwrap();

        assert!(result.r_squared > 0.9, "R-squared = {}", result.r_squared);
    }

    #[test]
    fn test_gcv_computation() {
        let rss = 100.0;
        let n = 50;
        let df = 5.0;

        let gcv = compute_gcv(rss, n, df);

        // GCV = 100 / (50 * (1 - 5/50)^2) = 100 / (50 * 0.81) = 2.469...
        assert_relative_eq!(gcv, 100.0 / (50.0 * 0.81), epsilon = 0.01);
    }

    #[test]
    fn test_mars_variable_importance() {
        // x1 is informative, x2 is noise
        let n = 50;
        let mut x_data = Vec::with_capacity(n * 2);
        let mut y_data = Vec::with_capacity(n);

        for i in 0..n {
            let x1 = i as f64;
            let x2 = (i as f64 * 17.0) % 13.0; // Pseudo-random noise
            x_data.push(x1);
            x_data.push(x2);
            y_data.push(x1 + 0.01 * (i as f64 % 3.0));
        }

        let x = Array2::from_shape_vec((n, 2), x_data).unwrap();
        let y = Array1::from_vec(y_data);

        let result = mars(
            x.view(),
            y.view(),
            MarsConfig::default(),
            Some(vec!["informative".to_string(), "noise".to_string()]),
        )
        .unwrap();

        // x1 should be more important than x2
        assert!(
            result.variable_importance[0] > result.variable_importance[1],
            "x1 importance ({}) should be > x2 importance ({})",
            result.variable_importance[0],
            result.variable_importance[1]
        );
    }

    #[test]
    fn test_run_mars_convenience() {
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0]
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let result = run_mars(x.view(), y.view(), None, None, None, None).unwrap();

        assert!(result.r_squared > 0.9);
        assert_eq!(result.n_obs, 10);
        assert_eq!(result.n_features, 1);
    }

    fn two_kink(n: usize, extra: usize) -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_fn((n, 1 + extra), |(i, j)| {
            if j == 0 {
                i as f64 / (n - 1) as f64
            } else {
                ((i * (7 + 4 * j)) % 17) as f64 / 16.0
            }
        });
        let y = (0..n)
            .map(|i| {
                let t = x[[i, 0]];
                2.0 * (t - 0.3).max(0.0) - 5.0 * (t - 0.7).max(0.0)
                    + 0.02 * (((i * 13) % 7) as f64 - 3.0)
            })
            .collect();
        (x, Array1::from_vec(y))
    }

    /// WT-12: nprune is a cap on the pruned model (earth). Before: the best-GCV model of any size.
    #[test]
    fn test_mars_respects_nprune() {
        for (n, fast) in [(120, Some(false)), (240, None)] {
            let (x, y) = two_kink(n, 2);
            let full = {
                let c = MarsConfig {
                    fast,
                    ..Default::default()
                };
                mars(x.view(), y.view(), c, None).unwrap()
            };
            assert!(
                full.basis_functions.len() > 2,
                "n = {n}: full model {}",
                full.basis_functions.len()
            );
            for cap in [1usize, 2, 3] {
                let c = MarsConfig {
                    fast,
                    nprune: Some(cap),
                    ..Default::default()
                };
                let r = mars(x.view(), y.view(), c, None).unwrap();
                assert!(
                    r.basis_functions.len() <= cap,
                    "n = {n}, nprune = {cap}: {} terms",
                    r.basis_functions.len()
                );
                let best = r
                    .gcv_history
                    .iter()
                    .filter(|(k, _)| *k <= cap)
                    .map(|(_, g)| *g)
                    .fold(f64::INFINITY, f64::min);
                assert_eq!(
                    r.gcv, best,
                    "returned GCV must be the best among sizes <= {cap}"
                );
            }
        }
    }

    /// ML-96: earth's GCV is infinite once the effective number of parameters reaches n.
    #[test]
    fn test_mars_gcv_infinite_at_df_ge_n() {
        assert!((compute_gcv(2.5, 10, 4.0) - 0.6944444444444444).abs() < 1e-15);
        assert_eq!(compute_gcv(2.5, 10, 10.0), f64::INFINITY);
        assert_eq!(compute_gcv(2.5, 10, 16.0), f64::INFINITY);
        let x = Array2::from_shape_fn((8, 3), |(i, j)| match j {
            0 => i as f64 / 7.0,
            1 => ((i * 3) % 8) as f64 / 7.0,
            _ => ((i * 5) % 8) as f64 / 7.0,
        });
        let y = Array1::from_shape_fn(8, |i| {
            4.0 * (x[[i, 0]] - 0.5).abs() + 0.01 * ((i * 7) % 5) as f64
        });
        let r = mars(x.view(), y.view(), MarsConfig::default(), None).unwrap();
        assert!(
            r.effective_df() < 8.0 && r.gcv.is_finite() && r.r_squared_adj.is_finite(),
            "{r}"
        );
        assert!(r.basis_functions.len() < 8);
    }

    /// GCV of the returned additive model = Friedman (1991) eq. 30 / earth GetGcv.
    #[test]
    fn test_mars_gcv_matches_friedman_formula() {
        let (x, y) = two_kink(120, 2);
        let c = MarsConfig {
            fast: Some(false),
            ..Default::default()
        };
        let r = mars(x.view(), y.view(), c, None).unwrap();
        let m = r.basis_functions.len() as f64;
        assert!(
            r.basis_functions
                .iter()
                .skip(1)
                .all(|b| b.hinges.len() == 1)
        );
        let d = m + 3.0 * (m - 1.0) / 2.0;
        assert_eq!(d, r.effective_df());
        let want = r.rss / (120.0 * (1.0 - d / 120.0).powi(2));
        assert!((r.gcv - want).abs() <= 1e-12 * want, "{} vs {want}", r.gcv);
    }

    /// WT-12: intercept = false fits without the constant; the fit is least squares on the
    /// returned basis.
    #[test]
    fn test_mars_without_intercept() {
        let (x, y) = two_kink(120, 1);
        let c = MarsConfig {
            intercept: false,
            fast: Some(false),
            ..Default::default()
        };
        let r = mars(x.view(), y.view(), c, None).unwrap();
        assert!(
            r.basis_functions.iter().all(|b| !b.hinges.is_empty()),
            "intercept present"
        );
        let design = Array2::from_shape_fn((120, r.basis_functions.len()), |(i, j)| {
            r.basis_functions[j].evaluate(&x.row(i))
        });
        let ls = crate::linalg::solve::least_squares(design.view(), y.view()).unwrap();
        for (a, b) in r.coefficients.iter().zip(ls.coefficients.iter()) {
            assert!((a - b).abs() <= 1e-10 * b.abs().max(1.0));
        }
        let syy: f64 = y.iter().map(|v| v * v).sum();
        assert!((r.r_squared - (1.0 - r.rss / syy)).abs() < 1e-12);
        let m = r.basis_functions.len() as f64;
        assert!(
            (r.r_squared_adj - (1.0 - (1.0 - r.r_squared) * 120.0 / (120.0 - m))).abs() < 1e-12
        );
        assert_eq!(r.predict(&x.view()).to_vec(), r.fitted_values);
    }

    /// Every returned basis is identified; a collinear second hinge pair on one variable enters as
    /// a single hinge (earth). Before: both hinges, design rank = terms - 1, 1e-8 ridge.
    #[test]
    fn test_mars_design_has_full_rank() {
        let (x, y) = two_kink(120, 0);
        let c = MarsConfig {
            fast: Some(false),
            ..Default::default()
        };
        let r = mars(x.view(), y.view(), c, None).unwrap();
        let k = r.basis_functions.len();
        let design =
            Array2::from_shape_fn((120, k), |(i, j)| r.basis_functions[j].evaluate(&x.row(i)));
        let ls = crate::linalg::solve::least_squares(design.view(), y.view()).unwrap();
        assert_eq!(ls.rank, k, "rank {} of {k}", ls.rank);
        assert!(r.coefficients.iter().all(|b| b.is_finite()));
    }

    /// Parameters are validated; nothing panics. Before: an endspan above n panicked.
    #[test]
    fn test_mars_validates_parameters() {
        let (x, y) = two_kink(20, 1);
        let err = |f: &dyn Fn(&mut MarsConfig)| {
            let mut c = MarsConfig::default();
            f(&mut c);
            mars(x.view(), y.view(), c, None).unwrap_err()
        };
        assert!(err(&|c| c.endspan = Some(25)).contains("endspan"));
        assert!(err(&|c| c.endspan = Some(10)).contains("endspan")); // 2 * endspan must be < n
        assert!(err(&|c| c.minspan = Some(0)).contains("minspan"));
        assert!(err(&|c| c.nprune = Some(0)).contains("nprune"));
        assert!(err(&|c| c.nk = Some(0)).contains("nk"));
        assert!(err(&|c| c.degree = 0).contains("degree"));
        assert!(err(&|c| c.penalty = -1.0).contains("penalty"));
        assert!(err(&|c| c.thresh = f64::NAN).contains("thresh"));
        assert!(err(&|c| c.fast_k = 0).contains("fast_k"));
        let mut xn = x.clone();
        xn[[3, 0]] = f64::NAN;
        assert!(
            mars(xn.view(), y.view(), MarsConfig::default(), None)
                .unwrap_err()
                .contains("non-finite")
        );
        assert!(
            mars(
                x.view(),
                y.view(),
                MarsConfig::default(),
                Some(vec!["a".into()])
            )
            .unwrap_err()
            .contains("feature_names")
        );
    }

    /// Every forward candidate's RSS and every backward deletion's RSS equal a full least-squares
    /// refit, with the same aliased set, on noisy data with collinear second hinges and degree 2.
    #[test]
    fn test_mars_incremental_scoring_matches_refits() {
        let n = 240;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 {
                (i as f64 / (n - 1) as f64) * 2.0
            } else {
                ((i * 37) % 101) as f64 / 100.0
            }
        });
        let y = Array1::from_shape_fn(n, |i| {
            let (a, b) = (x[[i, 0]], x[[i, 1]]);
            (a - 0.6).max(0.0) * 3.0 + a * b + 0.05 * (((i * 13) % 11) as f64 - 5.0)
        });
        let refit = |cols: &[Vec<f64>]| {
            let d = Array2::from_shape_fn((n, cols.len()), |(i, j)| cols[j][i]);
            least_squares(d.view(), y.view()).unwrap()
        };
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * b.abs().max(1e-300);

        let mut state = ForwardState::new(&y.view()).unwrap();
        let ones = vec![1.0; n];
        assert!(state.accept(&ones).unwrap());
        let mut cols = vec![ones.clone()];
        let (mut u1, mut u2) = (vec![0.0; n], vec![0.0; n]);
        let mut aliased_seen = 0;
        // Accept a pair per step; the later knots are collinear with the earlier pairs, and the
        // first one has an all-zero hinge.
        for (step, (j, parent, knot)) in [
            (0, 0, 0.0),
            (0, 0, 0.6),
            (1, 2, 0.5), // on the hinge of x0 at 0.6: degree 2
            (0, 0, 1.4),
            (0, 0, 0.6),
            (1, 0, 0.3),
        ]
        .into_iter()
        .enumerate()
        {
            let parent_col = cols[parent.min(cols.len() - 1)].clone();
            let pos: Vec<f64> = (0..n)
                .map(|i| parent_col[i] * (x[[i, j]] - knot).max(0.0))
                .collect();
            let neg: Vec<f64> = (0..n)
                .map(|i| parent_col[i] * (knot - x[[i, j]]).max(0.0))
                .collect();
            let score = state.score(&pos, &neg, &mut u1, &mut u2);
            let mut design = cols.clone();
            design.push(pos.clone());
            design.push(neg.clone());
            let ls = refit(&design);
            let rss: f64 = ls.residuals.iter().map(|e| e * e).sum();
            assert!(close(score.rss, rss), "step {step}: {} {rss}", score.rss);
            let k = cols.len();
            assert_eq!(!score.keep_pos, ls.aliased.contains(&k), "step {step}");
            assert_eq!(
                !score.keep_neg,
                ls.aliased.contains(&(k + 1)),
                "step {step}"
            );
            aliased_seen += usize::from(!score.keep_pos) + usize::from(!score.keep_neg);
            for (keep, column) in [(score.keep_pos, pos), (score.keep_neg, neg)] {
                if keep {
                    assert!(state.accept(&column).unwrap());
                    cols.push(column);
                }
            }
        }
        assert!(
            aliased_seen >= 2,
            "the test needs collinear hinges: {aliased_seen}"
        );

        // Backward: each deletion's RSS from one fit equals the refit without that column.
        let ls = refit(&cols);
        assert!(ls.aliased.is_empty());
        let rss: f64 = ls.residuals.iter().map(|e| e * e).sum();
        let costs = deletion_costs(&ls);
        for j in 1..cols.len() {
            let mut fewer = cols.clone();
            fewer.remove(j);
            let r = refit(&fewer);
            let want: f64 = r.residuals.iter().map(|e| e * e).sum();
            assert!(
                close(rss + costs[j], want),
                "{j}: {} {want}",
                rss + costs[j]
            );
        }
        // An aliased column costs nothing.
        let mut twice = cols.clone();
        twice.push(cols[1].clone());
        let ls = refit(&twice);
        assert_eq!(ls.aliased, vec![cols.len()]);
        assert_eq!(deletion_costs(&ls)[cols.len()], 0.0);
    }
}
