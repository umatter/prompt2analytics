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
//! GCV = RSS / (n * (1 - d/n)^2)
//!
//! where:
//! - RSS = Residual Sum of Squares
//! - n = number of observations
//! - d = effective degrees of freedom (including penalty for knots)
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

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for MARS model fitting.
#[derive(Debug, Clone)]
pub struct MarsConfig {
    /// Maximum degree of interaction (1 = additive, 2 = two-way interactions, etc.)
    /// Default: 1 (additive model)
    pub degree: usize,

    /// Maximum number of terms (basis functions) in the final model after pruning.
    /// If None, uses automatic selection based on GCV.
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

    /// Include intercept term.
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

    /// Variable importance scores (sum of |coefficient * basis function variance|)
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

        importance_with_names
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

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
/// use p2a_core::ml::mars::{mars, MarsConfig};
/// use ndarray::{Array1, Array2};
///
/// let x = Array2::from_shape_vec((10, 2), vec![
///     1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0,
///     6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 9.0, 10.0, 10.0, 11.0,
/// ]).unwrap();
/// let y = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0]);
///
/// let config = MarsConfig::default();
/// let result = mars(x.view(), y.view(), config, None).unwrap();
/// println!("R-squared: {:.3}", result.r_squared);
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
    let n = x.nrows();
    let p = x.ncols();

    if y.len() != n {
        return Err(format!("y has length {} but x has {} rows", y.len(), n));
    }

    if n < 3 {
        return Err("Need at least 3 observations for MARS".to_string());
    }

    // Compute y statistics
    let y_mean = y.mean().unwrap_or(0.0);
    let tss: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();

    if tss < 1e-10 {
        return Err("Response variable has zero variance".to_string());
    }

    // Determine algorithm parameters
    let nk = config.nk.unwrap_or_else(|| {
        let default_nk = (2 * p + n / 10).max(20).min(200);
        default_nk
    });

    let use_fast = config.fast.unwrap_or(n > 200);
    let minspan = config.minspan.unwrap_or_else(|| compute_minspan(n));
    let endspan = config.endspan.unwrap_or_else(|| compute_endspan(n));

    // Find candidate knots for each variable
    let candidate_knots = find_candidate_knots(&x, minspan, endspan, use_fast, config.fast_k);

    // Forward pass: build up model by adding basis functions
    let (forward_basis, forward_coefs) = forward_pass(
        &x,
        &y,
        &candidate_knots,
        nk,
        config.degree,
        config.thresh,
        &config,
    )?;

    // Backward pass: prune using GCV
    let (final_basis, final_coefs, gcv_history) = backward_pass(
        &x,
        &y,
        forward_basis,
        forward_coefs,
        config.nprune,
        config.penalty,
    )?;

    // Compute final statistics
    let fitted = predict_internal(&x, &final_basis, &final_coefs);
    let residuals: Vec<f64> = y
        .iter()
        .zip(fitted.iter())
        .map(|(&yi, &fi)| yi - fi)
        .collect();
    let rss: f64 = residuals.iter().map(|r| r * r).sum();

    let r_squared = 1.0 - rss / tss;
    let n_terms = final_basis.len();
    let r_squared_adj =
        1.0 - (1.0 - r_squared) * ((n - 1) as f64) / ((n - n_terms) as f64).max(1.0);

    // GCV for final model
    let effective_df = compute_effective_df(&final_basis, config.penalty);
    let gcv = compute_gcv(rss, n, effective_df);

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
        knots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
        col_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Apply endspan: skip first and last endspan observations
        let start = endspan.min(n / 2);
        let end = (n - endspan).max(start + 1);

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

/// Forward pass: greedily add basis functions.
fn forward_pass(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    candidate_knots: &[Vec<f64>],
    max_terms: usize,
    max_degree: usize,
    thresh: f64,
    config: &MarsConfig,
) -> Result<(Vec<BasisFunction>, Vec<f64>), String> {
    let n = x.nrows();
    let p = x.ncols();

    // Start with intercept only
    let mut basis_functions: Vec<BasisFunction> = vec![BasisFunction::intercept()];

    // Fit initial model (intercept only)
    let y_mean = y.mean().unwrap_or(0.0);
    let initial_rss: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();

    let mut current_rss = initial_rss;
    let mut residuals = y.to_owned();
    for r in residuals.iter_mut() {
        *r -= y_mean;
    }

    let initial_gcv = compute_gcv(
        initial_rss,
        n,
        compute_effective_df(&basis_functions, config.penalty),
    );
    let mut current_gcv = initial_gcv;

    // Iteratively add basis function pairs
    while basis_functions.len() < max_terms {
        let mut best_gcv_reduction = 0.0;
        let mut best_pair: Option<(BasisFunction, BasisFunction)> = None;
        let mut best_parent_idx: Option<usize> = None;

        // Try adding a new hinge pair to each existing basis function
        // (or as a new main effect if degree allows)
        let n_current = basis_functions.len();

        for parent_idx in 0..n_current {
            let parent = &basis_functions[parent_idx];
            let parent_degree = parent.degree();

            // Check if we can add another hinge to this parent
            if parent_degree >= max_degree {
                continue;
            }

            // Get variables already used by this parent
            let used_vars: Vec<usize> = parent.hinges.iter().map(|h| h.variable).collect();

            // Try each variable not already in parent
            for j in 0..p {
                if used_vars.contains(&j) {
                    continue;
                }

                // Try each knot
                for &knot in &candidate_knots[j] {
                    // Create positive and negative hinge pair
                    let hinge_pos = HingeFunction {
                        variable: j,
                        knot,
                        hinge_type: HingeType::Positive,
                        variable_name: None,
                    };
                    let hinge_neg = HingeFunction {
                        variable: j,
                        knot,
                        hinge_type: HingeType::Negative,
                        variable_name: None,
                    };

                    let basis_pos = if parent_degree == 0 {
                        BasisFunction::single(hinge_pos.clone())
                    } else {
                        BasisFunction::interaction(parent_idx, &parent.hinges, hinge_pos.clone())
                    };

                    let basis_neg = if parent_degree == 0 {
                        BasisFunction::single(hinge_neg.clone())
                    } else {
                        BasisFunction::interaction(parent_idx, &parent.hinges, hinge_neg.clone())
                    };

                    // Evaluate new basis functions
                    let b_pos = basis_pos.evaluate_all(x);
                    let b_neg = basis_neg.evaluate_all(x);

                    // Check if basis functions have any variation
                    let var_pos: f64 = b_pos.iter().map(|&v| v * v).sum::<f64>() / n as f64
                        - (b_pos.sum() / n as f64).powi(2);
                    let var_neg: f64 = b_neg.iter().map(|&v| v * v).sum::<f64>() / n as f64
                        - (b_neg.sum() / n as f64).powi(2);

                    if var_pos < 1e-10 && var_neg < 1e-10 {
                        continue;
                    }

                    // Build design matrix with existing basis + new pair
                    let mut test_basis = basis_functions.clone();
                    test_basis.push(basis_pos.clone());
                    test_basis.push(basis_neg.clone());

                    // Fit model and compute GCV
                    if let Ok((test_coefs, test_rss)) = fit_ols(x, y, &test_basis) {
                        let test_df = compute_effective_df(&test_basis, config.penalty);
                        let test_gcv = compute_gcv(test_rss, n, test_df);

                        let gcv_reduction = current_gcv - test_gcv;
                        if gcv_reduction > best_gcv_reduction {
                            best_gcv_reduction = gcv_reduction;
                            best_pair = Some((basis_pos, basis_neg));
                            best_parent_idx = Some(parent_idx);
                        }
                    }
                }
            }
        }

        // Check if improvement meets threshold
        let relative_improvement = best_gcv_reduction / (current_gcv + 1e-10);
        if relative_improvement < thresh || best_pair.is_none() {
            break;
        }

        // Add best pair
        let (basis_pos, basis_neg) = best_pair.unwrap();
        basis_functions.push(basis_pos);
        basis_functions.push(basis_neg);

        // Update current model
        if let Ok((_, new_rss)) = fit_ols(x, y, &basis_functions) {
            current_rss = new_rss;
            let new_df = compute_effective_df(&basis_functions, config.penalty);
            current_gcv = compute_gcv(current_rss, n, new_df);
        }
    }

    // Final fit
    let (coefs, _) = fit_ols(x, y, &basis_functions)?;

    Ok((basis_functions, coefs))
}

/// Backward pass: prune basis functions using GCV.
fn backward_pass(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    mut basis: Vec<BasisFunction>,
    mut coefs: Vec<f64>,
    nprune: Option<usize>,
    penalty: f64,
) -> Result<(Vec<BasisFunction>, Vec<f64>, Vec<(usize, f64)>), String> {
    let n = x.nrows();
    let mut gcv_history = Vec::new();

    // Initial GCV
    let (_, rss) = fit_ols(x, y, &basis)?;
    let initial_gcv = compute_gcv(rss, n, compute_effective_df(&basis, penalty));
    gcv_history.push((basis.len(), initial_gcv));

    let mut best_gcv = initial_gcv;
    let mut best_basis = basis.clone();
    let mut best_coefs = coefs.clone();

    // Try removing each term (except intercept) and check GCV
    while basis.len() > 1 {
        let target_terms = nprune.unwrap_or(1);
        if basis.len() <= target_terms {
            break;
        }

        let mut best_removal_gcv = f64::INFINITY;
        let mut best_removal_idx = None;

        // Try removing each non-intercept term
        for i in 1..basis.len() {
            let mut test_basis = basis.clone();
            test_basis.remove(i);

            if let Ok((_, test_rss)) = fit_ols(x, y, &test_basis) {
                let test_df = compute_effective_df(&test_basis, penalty);
                let test_gcv = compute_gcv(test_rss, n, test_df);

                if test_gcv < best_removal_gcv {
                    best_removal_gcv = test_gcv;
                    best_removal_idx = Some(i);
                }
            }
        }

        if let Some(idx) = best_removal_idx {
            basis.remove(idx);
            let (new_coefs, _) = fit_ols(x, y, &basis)?;
            coefs = new_coefs;

            gcv_history.push((basis.len(), best_removal_gcv));

            // Track best model
            if best_removal_gcv < best_gcv {
                best_gcv = best_removal_gcv;
                best_basis = basis.clone();
                best_coefs = coefs.clone();
            }
        } else {
            break;
        }
    }

    // Return best model found during pruning
    Ok((best_basis, best_coefs, gcv_history))
}

/// Fit OLS given basis functions.
fn fit_ols(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    basis: &[BasisFunction],
) -> Result<(Vec<f64>, f64), String> {
    let n = x.nrows();
    let k = basis.len();

    if k == 0 {
        return Err("No basis functions".to_string());
    }

    // Build design matrix
    let mut design = Array2::zeros((n, k));
    for (j, b) in basis.iter().enumerate() {
        let col = b.evaluate_all(x);
        design.column_mut(j).assign(&col);
    }

    // Solve (X'X)^-1 X'y via QR decomposition approach
    // Use simple normal equations for stability
    let xtx = design.t().dot(&design);
    let xty = design.t().dot(y);

    // Add small regularization for numerical stability
    let mut xtx_reg = xtx.clone();
    for i in 0..k {
        xtx_reg[[i, i]] += 1e-8;
    }

    // Solve using Cholesky-like approach (via inversion)
    let coefs = solve_linear_system(&xtx_reg, &xty)?;

    // Compute RSS
    let fitted = design.dot(&coefs);
    let rss: f64 = y
        .iter()
        .zip(fitted.iter())
        .map(|(&yi, &fi)| (yi - fi).powi(2))
        .sum();

    Ok((coefs.to_vec(), rss))
}

/// Solve linear system Ax = b.
fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, String> {
    let n = a.nrows();

    // Simple Gaussian elimination with partial pivoting
    let mut aug = Array2::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        let mut max_val = aug[[i, i]].abs();
        for k in (i + 1)..n {
            if aug[[k, i]].abs() > max_val {
                max_val = aug[[k, i]].abs();
                max_row = k;
            }
        }

        if max_val < 1e-14 {
            return Err("Singular matrix in linear solve".to_string());
        }

        // Swap rows
        if max_row != i {
            for j in 0..=n {
                let temp = aug[[i, j]];
                aug[[i, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Eliminate
        for k in (i + 1)..n {
            let factor = aug[[k, i]] / aug[[i, i]];
            for j in i..=n {
                aug[[k, j]] -= factor * aug[[i, j]];
            }
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * x[j];
        }
        x[i] = sum / aug[[i, i]];
    }

    Ok(x)
}

/// Predict using basis functions and coefficients.
fn predict_internal(x: &ArrayView2<f64>, basis: &[BasisFunction], coefs: &[f64]) -> Vec<f64> {
    let n = x.nrows();
    let mut predictions = vec![0.0; n];

    for (b, &c) in basis.iter().zip(coefs.iter()) {
        let vals = b.evaluate_all(x);
        for i in 0..n {
            predictions[i] += c * vals[i];
        }
    }

    predictions
}

/// Compute effective degrees of freedom.
/// d = number_of_terms + penalty * number_of_knots / 2
fn compute_effective_df(basis: &[BasisFunction], penalty: f64) -> f64 {
    let n_terms = basis.len() as f64;
    let n_knots: f64 = basis.iter().map(|b| b.hinges.len() as f64).sum();
    // Hinges come in pairs, so divide by 2
    n_terms + penalty * n_knots / 2.0
}

/// Compute GCV criterion.
/// GCV = RSS / (n * (1 - d/n)^2)
fn compute_gcv(rss: f64, n: usize, effective_df: f64) -> f64 {
    let n_f = n as f64;
    let denom = (1.0 - effective_df / n_f).powi(2) * n_f;
    if denom > 1e-10 {
        rss / denom
    } else {
        f64::INFINITY
    }
}

/// Compute variable importance.
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
/// * `nprune` - Maximum terms in final model (default: auto)
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
}
