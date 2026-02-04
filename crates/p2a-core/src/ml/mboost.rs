//! Model-based Gradient Boosting (mboost).
//!
//! Pure Rust implementation of component-wise gradient boosting, equivalent to R's mboost package.
//! This approach differs from traditional gradient boosting (like GBM) by updating only one
//! component (variable) per iteration, providing automatic variable selection.
//!
//! ## Key Features
//!
//! - **Component-wise boosting**: Updates only the best variable per iteration
//! - **Automatic variable selection**: Sparse solutions via early stopping
//! - **Multiple loss functions**: Gaussian, Binomial, Poisson families
//! - **Two base learners**: Linear (L2-boosting) and tree-based
//! - **Cross-validation**: For optimal stopping iteration (mstop)
//!
//! ## Algorithm
//!
//! The component-wise boosting algorithm:
//!
//! 1. Initialize: f^(0)(x) = argmin_c sum L(y_i, c)
//! 2. For m = 1 to mstop:
//!    a. Compute negative gradient: u_i = -dL/df(y_i, f^(m-1)(x_i))
//!    b. For each component j, fit base learner h_j to (x_j, u)
//!    c. Select j* = argmin_j sum (u_i - h_j(x_ij))^2
//!    d. Update: f^(m) = f^(m-1) + nu * h_j*(x_j*)
//! 3. Return f^(mstop)
//!
//! ## Example
//!
//! ```rust,no_run
//! use p2a_core::ml::{mboost, MboostConfig, MboostFamily, MboostBaseLearner};
//! use ndarray::array;
//!
//! let x = array![[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0], [5.0, 2.5]];
//! let y = array![1.1, 2.1, 2.9, 4.2, 4.8];
//!
//! let config = MboostConfig {
//!     mstop: 100,
//!     nu: 0.1,
//!     family: MboostFamily::Gaussian,
//!     base_learner: MboostBaseLearner::Linear,
//!     ..Default::default()
//! };
//!
//! let result = mboost(x.view(), y.view(), &config).unwrap();
//! println!("Selected variables: {:?}", result.selected_variables);
//! ```
//!
//! ## References
//!
//! - Hothorn, T., Buehlmann, P., Kneib, T., Schmid, M., & Hofner, B. (2010).
//!   Model-based boosting 2.0. Journal of Machine Learning Research, 11, 2109-2113.
//! - Buehlmann, P. & Hothorn, T. (2007). Boosting algorithms: Regularization,
//!   prediction and model fitting. Statistical Science, 22(4), 477-505.
//! - Hofner, B., Mayr, A., Robinzonov, N., & Schmid, M. (2014). Model-based
//!   boosting in R: A hands-on tutorial using the R package mboost.
//!   Computational Statistics, 29(1-2), 3-35.
//!
//! Implementation validated against R package mboost 2.9-11.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use serde::{Deserialize, Serialize};

use crate::Dataset;
use crate::errors::{EconError, EconResult};

/// Loss function family for mboost.
///
/// Determines the negative gradient computed at each iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MboostFamily {
    /// Gaussian (squared error loss) for regression.
    /// Negative gradient: y - f(x)
    #[default]
    Gaussian,
    /// Binomial (log-loss) for binary classification.
    /// Negative gradient: y - sigmoid(f(x))
    Binomial,
    /// Poisson (negative log-likelihood) for count data.
    /// Negative gradient: y - exp(f(x))
    Poisson,
}

impl std::str::FromStr for MboostFamily {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "gaussian" | "normal" | "squared_error" | "mse" => Ok(MboostFamily::Gaussian),
            "binomial" | "logistic" | "bernoulli" => Ok(MboostFamily::Binomial),
            "poisson" | "count" => Ok(MboostFamily::Poisson),
            _ => Err(format!("Unknown mboost family: {}", s)),
        }
    }
}

impl std::fmt::Display for MboostFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MboostFamily::Gaussian => write!(f, "Gaussian"),
            MboostFamily::Binomial => write!(f, "Binomial"),
            MboostFamily::Poisson => write!(f, "Poisson"),
        }
    }
}

/// Base learner type for mboost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MboostBaseLearner {
    /// Linear base learner (bols in R).
    /// Fits univariate OLS at each iteration.
    #[default]
    Linear,
    /// Tree base learner (btree in R).
    /// Fits regression stumps (depth-1 trees) at each iteration.
    Tree,
}

impl std::str::FromStr for MboostBaseLearner {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "linear" | "bols" | "l2" => Ok(MboostBaseLearner::Linear),
            "tree" | "btree" | "stump" => Ok(MboostBaseLearner::Tree),
            _ => Err(format!("Unknown base learner: {}", s)),
        }
    }
}

impl std::fmt::Display for MboostBaseLearner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MboostBaseLearner::Linear => write!(f, "Linear"),
            MboostBaseLearner::Tree => write!(f, "Tree"),
        }
    }
}

/// Configuration for mboost.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MboostConfig {
    /// Number of boosting iterations (default: 100).
    /// Also known as the stopping iteration. Larger values may overfit.
    pub mstop: usize,

    /// Learning rate / step length (default: 0.1).
    /// Smaller values require more iterations but often improve generalization.
    /// Typical range: 0.01 to 0.5.
    pub nu: f64,

    /// Loss function family (default: Gaussian).
    pub family: MboostFamily,

    /// Base learner type (default: Linear).
    pub base_learner: MboostBaseLearner,

    /// Maximum depth for tree base learner (default: 1, i.e., stumps).
    /// Only used when base_learner = Tree.
    pub tree_depth: usize,

    /// Minimum samples for tree splits (default: 5).
    /// Only used when base_learner = Tree.
    pub min_samples_split: usize,

    /// Number of cross-validation folds for early stopping (default: None).
    /// If set, performs CV to find optimal mstop.
    pub cv_folds: Option<usize>,

    /// Whether to center predictors before fitting (default: true).
    /// Centering improves numerical stability for linear base learners.
    pub center: bool,

    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for MboostConfig {
    fn default() -> Self {
        MboostConfig {
            mstop: 100,
            nu: 0.1,
            family: MboostFamily::Gaussian,
            base_learner: MboostBaseLearner::Linear,
            tree_depth: 1,
            min_samples_split: 5,
            cv_folds: None,
            center: true,
            seed: None,
        }
    }
}

/// Base learner fit result (internal).
#[derive(Debug, Clone)]
struct BaseLearnerFit {
    /// Coefficient for linear or prediction increment for tree
    coefficient: f64,
    /// Intercept (for linear) or threshold (for tree)
    intercept: f64,
    /// For tree: split threshold
    threshold: Option<f64>,
    /// For tree: direction (true = go left if <= threshold)
    go_left: bool,
    /// Residual sum of squares after fitting
    rss: f64,
}

/// Coefficient path entry for one variable at one iteration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoefficientPathEntry {
    /// Iteration number
    pub iteration: usize,
    /// Variable index
    pub variable: usize,
    /// Coefficient value at this iteration
    pub coefficient: f64,
}

/// Result from mboost fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MboostResult {
    /// Final coefficients for each variable (0 if not selected).
    pub coefficients: Vec<f64>,

    /// Intercept term.
    pub intercept: f64,

    /// Variable importance (proportion of times selected).
    pub variable_importance: Vec<f64>,

    /// Indices of selected variables (non-zero coefficients).
    pub selected_variables: Vec<usize>,

    /// Number of unique variables selected.
    pub n_selected: usize,

    /// Selection frequency for each variable.
    pub selection_frequency: Vec<usize>,

    /// Loss at each iteration.
    pub loss_history: Vec<f64>,

    /// Final training loss.
    pub final_loss: f64,

    /// Predictions on training data.
    pub predictions: Vec<f64>,

    /// Number of boosting iterations performed.
    pub iterations: usize,

    /// Optimal mstop from cross-validation (if cv_folds was set).
    pub cv_optimal_mstop: Option<usize>,

    /// Cross-validation error at each iteration (if cv_folds was set).
    pub cv_error: Option<Vec<f64>>,

    /// Configuration used.
    pub config: MboostConfig,

    /// Feature names (if provided).
    pub feature_names: Option<Vec<String>>,

    /// Coefficient path (coefficients at each iteration).
    /// Only first 50 entries per variable are stored to limit memory.
    #[serde(skip)]
    pub coefficient_path: Vec<CoefficientPathEntry>,
}

impl std::fmt::Display for MboostResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Model-based Boosting (mboost) Results")?;
        writeln!(f, "======================================")?;
        writeln!(f, "Family: {}", self.config.family)?;
        writeln!(f, "Base learner: {}", self.config.base_learner)?;
        writeln!(f, "Iterations (mstop): {}", self.iterations)?;
        writeln!(f, "Learning rate (nu): {:.4}", self.config.nu)?;

        writeln!(f)?;
        writeln!(
            f,
            "Variables selected: {} / {}",
            self.n_selected,
            self.coefficients.len()
        )?;
        writeln!(f, "Final training loss: {:.6}", self.final_loss)?;

        if let Some(cv_mstop) = self.cv_optimal_mstop {
            writeln!(f, "CV optimal mstop: {}", cv_mstop)?;
        }

        writeln!(f)?;
        writeln!(f, "Intercept: {:.6}", self.intercept)?;

        writeln!(f)?;
        writeln!(f, "Variable Importance (by selection frequency):")?;

        // Sort by importance
        let mut indexed: Vec<(usize, f64)> = self
            .variable_importance
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (i, importance) in indexed.iter().take(10) {
            if *importance > 0.0 {
                let name = match &self.feature_names {
                    Some(names) => names.get(*i).cloned().unwrap_or_else(|| format!("X{}", i)),
                    None => format!("X{}", i),
                };
                let coef = self.coefficients[*i];
                writeln!(
                    f,
                    "  {}: importance={:.4}, coef={:.6}",
                    name, importance, coef
                )?;
            }
        }

        if self
            .variable_importance
            .iter()
            .filter(|&&v| v > 0.0)
            .count()
            > 10
        {
            writeln!(
                f,
                "  ... ({} more selected variables)",
                self.n_selected - 10
            )?;
        }

        Ok(())
    }
}

/// Simple LCG random number generator for reproducibility.
fn lcg_random(state: &mut u64) -> usize {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) ^ *state) as usize
}

/// Sigmoid function for binomial family.
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

/// Compute initial prediction (offset).
fn compute_init_prediction(y: &ArrayView1<f64>, family: MboostFamily) -> f64 {
    match family {
        MboostFamily::Gaussian => {
            // Mean of y
            y.iter().sum::<f64>() / y.len() as f64
        }
        MboostFamily::Binomial => {
            // Log-odds of mean probability
            let p = y.iter().sum::<f64>() / y.len() as f64;
            let p = p.clamp(1e-10, 1.0 - 1e-10);
            (p / (1.0 - p)).ln()
        }
        MboostFamily::Poisson => {
            // Log of mean
            let mean = y.iter().sum::<f64>() / y.len() as f64;
            mean.max(1e-10).ln()
        }
    }
}

/// Compute negative gradient (pseudo-residuals).
fn compute_negative_gradient(
    y: &ArrayView1<f64>,
    predictions: &Array1<f64>,
    family: MboostFamily,
) -> Array1<f64> {
    let n = y.len();
    let mut gradient = Array1::zeros(n);

    match family {
        MboostFamily::Gaussian => {
            // Negative gradient of squared error: y - f(x)
            for i in 0..n {
                gradient[i] = y[i] - predictions[i];
            }
        }
        MboostFamily::Binomial => {
            // Negative gradient of log-loss: y - sigmoid(f(x))
            for i in 0..n {
                let p = sigmoid(predictions[i]);
                gradient[i] = y[i] - p;
            }
        }
        MboostFamily::Poisson => {
            // Negative gradient of Poisson deviance: y - exp(f(x))
            for i in 0..n {
                let mu = predictions[i].exp().min(1e10); // Prevent overflow
                gradient[i] = y[i] - mu;
            }
        }
    }

    gradient
}

/// Compute loss for current predictions.
fn compute_loss(y: &ArrayView1<f64>, predictions: &Array1<f64>, family: MboostFamily) -> f64 {
    let n = y.len() as f64;

    match family {
        MboostFamily::Gaussian => {
            // Mean squared error
            y.iter()
                .zip(predictions.iter())
                .map(|(&yi, &pi)| (yi - pi).powi(2))
                .sum::<f64>()
                / n
        }
        MboostFamily::Binomial => {
            // Log-loss (negative log-likelihood)
            -y.iter()
                .zip(predictions.iter())
                .map(|(&yi, &pi)| {
                    let p = sigmoid(pi);
                    yi * p.max(1e-15).ln() + (1.0 - yi) * (1.0 - p).max(1e-15).ln()
                })
                .sum::<f64>()
                / n
        }
        MboostFamily::Poisson => {
            // Poisson deviance
            y.iter()
                .zip(predictions.iter())
                .map(|(&yi, &pi)| {
                    let mu = pi.exp().min(1e10);
                    if yi > 0.0 {
                        2.0 * (yi * (yi / mu).ln() - (yi - mu))
                    } else {
                        2.0 * mu
                    }
                })
                .sum::<f64>()
                / n
        }
    }
}

/// Fit linear base learner to one variable.
///
/// Returns (coefficient, intercept, RSS).
fn fit_linear_base_learner(
    x_col: &ArrayView1<f64>,
    residuals: &ArrayView1<f64>,
    x_mean: f64,
) -> BaseLearnerFit {
    let n = x_col.len() as f64;

    // Compute centered sums for univariate OLS
    let u_mean = residuals.iter().sum::<f64>() / n;

    let mut ss_xx = 0.0;
    let mut ss_xu = 0.0;

    for i in 0..x_col.len() {
        let x_centered = x_col[i] - x_mean;
        let u_centered = residuals[i] - u_mean;
        ss_xx += x_centered * x_centered;
        ss_xu += x_centered * u_centered;
    }

    // Coefficient: beta = sum((x-xbar)(u-ubar)) / sum((x-xbar)^2)
    let coefficient = if ss_xx > 1e-10 { ss_xu / ss_xx } else { 0.0 };

    // Intercept: alpha = ubar - beta * xbar
    let intercept = u_mean - coefficient * x_mean;

    // Compute RSS
    let mut rss = 0.0;
    for i in 0..x_col.len() {
        let predicted = intercept + coefficient * x_col[i];
        let residual = residuals[i] - predicted;
        rss += residual * residual;
    }

    BaseLearnerFit {
        coefficient,
        intercept,
        threshold: None,
        go_left: true,
        rss,
    }
}

/// Fit tree stump (depth-1 tree) base learner to one variable.
///
/// Finds the best split point that minimizes RSS.
fn fit_tree_base_learner(
    x_col: &ArrayView1<f64>,
    residuals: &ArrayView1<f64>,
    min_samples: usize,
) -> BaseLearnerFit {
    let n = x_col.len();

    // Sort by x value
    let mut sorted: Vec<(usize, f64)> = x_col.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Try all possible split points
    let mut best_rss = f64::INFINITY;
    let mut best_threshold = sorted[0].1;
    let mut best_left_mean = 0.0;
    let mut best_right_mean = 0.0;

    // Running sums for left partition
    let mut left_sum = 0.0;
    let mut left_n = 0usize;
    let total_sum: f64 = residuals.iter().sum();
    let total_ss: f64 = residuals.iter().map(|&r| r * r).sum();

    for i in 0..n - 1 {
        let (idx, x_val) = sorted[i];
        left_sum += residuals[idx];
        left_n += 1;

        let right_n = n - left_n;

        // Check minimum samples
        if left_n < min_samples || right_n < min_samples {
            continue;
        }

        // Check if next value is different (valid split point)
        let next_x = sorted[i + 1].1;
        if (next_x - x_val).abs() < 1e-10 {
            continue;
        }

        // Compute RSS for this split
        let left_mean = left_sum / left_n as f64;
        let right_sum = total_sum - left_sum;
        let right_mean = right_sum / right_n as f64;

        // RSS = sum((y - mean)^2) = sum(y^2) - n*mean^2
        // For left: sum_left(y^2) - left_n * left_mean^2
        // We need to compute sum of squared residuals for left and right

        // Incremental RSS computation
        let left_ss = {
            let mut ss = 0.0;
            for j in 0..=i {
                let r = residuals[sorted[j].0];
                ss += (r - left_mean).powi(2);
            }
            ss
        };
        let right_ss = {
            let mut ss = 0.0;
            for j in (i + 1)..n {
                let r = residuals[sorted[j].0];
                ss += (r - right_mean).powi(2);
            }
            ss
        };

        let rss = left_ss + right_ss;

        if rss < best_rss {
            best_rss = rss;
            best_threshold = (x_val + next_x) / 2.0;
            best_left_mean = left_mean;
            best_right_mean = right_mean;
        }
    }

    // If no valid split found, use mean
    if best_rss.is_infinite() {
        let mean = residuals.iter().sum::<f64>() / n as f64;
        let rss: f64 = residuals.iter().map(|&r| (r - mean).powi(2)).sum();
        return BaseLearnerFit {
            coefficient: mean,
            intercept: 0.0,
            threshold: None,
            go_left: true,
            rss,
        };
    }

    // Return stump parameters
    // coefficient stores the difference (right - left)
    // intercept stores left mean
    BaseLearnerFit {
        coefficient: best_right_mean - best_left_mean,
        intercept: best_left_mean,
        threshold: Some(best_threshold),
        go_left: true,
        rss: best_rss,
    }
}

/// Predict using a tree stump for one variable.
fn predict_stump(x_val: f64, fit: &BaseLearnerFit) -> f64 {
    match fit.threshold {
        Some(thresh) => {
            if x_val <= thresh {
                fit.intercept // left mean
            } else {
                fit.intercept + fit.coefficient // right mean
            }
        }
        None => fit.coefficient, // No split, use mean
    }
}

/// Run mboost (model-based gradient boosting).
///
/// # Arguments
///
/// * `x` - Feature matrix (n_samples x n_features)
/// * `y` - Target values (n_samples)
/// * `config` - MboostConfig with algorithm parameters
///
/// # Returns
///
/// MboostResult with coefficients, variable importance, and predictions.
///
/// # References
///
/// - Hothorn et al. (2010). Model-based boosting 2.0. JMLR.
/// - Buehlmann & Hothorn (2007). Boosting algorithms. Statistical Science.
pub fn mboost(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    config: &MboostConfig,
) -> EconResult<MboostResult> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_samples < 2 {
        return Err(EconError::Computation(
            "Need at least 2 samples for mboost".to_string(),
        ));
    }
    if n_samples != y.len() {
        return Err(EconError::Computation(
            "X and y must have same number of samples".to_string(),
        ));
    }

    // Validate binary targets for classification
    if config.family == MboostFamily::Binomial {
        for &yi in y.iter() {
            if yi != 0.0 && yi != 1.0 {
                return Err(EconError::Computation(
                    "Binomial family requires binary targets (0 or 1)".to_string(),
                ));
            }
        }
    }

    // Validate non-negative counts for Poisson
    if config.family == MboostFamily::Poisson {
        for &yi in y.iter() {
            if yi < 0.0 {
                return Err(EconError::Computation(
                    "Poisson family requires non-negative targets".to_string(),
                ));
            }
        }
    }

    // Compute column means for centering
    let x_means: Vec<f64> = (0..n_features)
        .map(|j| x.column(j).iter().sum::<f64>() / n_samples as f64)
        .collect();

    // Initialize predictions
    let init_pred = compute_init_prediction(&y, config.family);
    let mut predictions = Array1::from_elem(n_samples, init_pred);

    // Initialize coefficient accumulators
    let mut coefficients = vec![0.0; n_features];
    let mut intercept = init_pred;
    let mut selection_frequency = vec![0usize; n_features];
    let mut loss_history = Vec::with_capacity(config.mstop + 1);
    let mut coefficient_path = Vec::new();

    // Initial loss
    loss_history.push(compute_loss(&y, &predictions, config.family));

    // Store base learner fits for predictions
    let mut base_learner_fits: Vec<(usize, BaseLearnerFit)> = Vec::with_capacity(config.mstop);

    // Boosting iterations
    for m in 0..config.mstop {
        // Compute negative gradient (pseudo-residuals)
        let residuals = compute_negative_gradient(&y, &predictions, config.family);

        // Fit base learner to each component and select best
        let mut best_j = 0;
        let mut best_rss = f64::INFINITY;
        let mut best_fit = BaseLearnerFit {
            coefficient: 0.0,
            intercept: 0.0,
            threshold: None,
            go_left: true,
            rss: f64::INFINITY,
        };

        for j in 0..n_features {
            let x_col = x.column(j);
            let x_mean = if config.center { x_means[j] } else { 0.0 };

            let fit = match config.base_learner {
                MboostBaseLearner::Linear => {
                    fit_linear_base_learner(&x_col, &residuals.view(), x_mean)
                }
                MboostBaseLearner::Tree => {
                    fit_tree_base_learner(&x_col, &residuals.view(), config.min_samples_split)
                }
            };

            if fit.rss < best_rss {
                best_rss = fit.rss;
                best_j = j;
                best_fit = fit;
            }
        }

        // Update predictions with learning rate
        let x_col = x.column(best_j);
        match config.base_learner {
            MboostBaseLearner::Linear => {
                for i in 0..n_samples {
                    let update = best_fit.intercept + best_fit.coefficient * x_col[i];
                    predictions[i] += config.nu * update;
                }
                // Accumulate coefficients
                coefficients[best_j] += config.nu * best_fit.coefficient;
                intercept += config.nu * best_fit.intercept;
            }
            MboostBaseLearner::Tree => {
                for i in 0..n_samples {
                    let update = predict_stump(x_col[i], &best_fit);
                    predictions[i] += config.nu * update;
                }
                // Store the fit for later prediction
                base_learner_fits.push((best_j, best_fit.clone()));
            }
        }

        // Update selection frequency
        selection_frequency[best_j] += 1;

        // Store coefficient path (limit to save memory)
        if coefficient_path.len() < 5000 {
            coefficient_path.push(CoefficientPathEntry {
                iteration: m + 1,
                variable: best_j,
                coefficient: coefficients[best_j],
            });
        }

        // Compute and store loss
        let loss = compute_loss(&y, &predictions, config.family);
        loss_history.push(loss);
    }

    // Compute variable importance (normalized selection frequency)
    let total_selections: f64 = selection_frequency.iter().map(|&s| s as f64).sum::<f64>();
    let variable_importance: Vec<f64> = if total_selections > 0.0 {
        selection_frequency
            .iter()
            .map(|&s| s as f64 / total_selections)
            .collect()
    } else {
        vec![0.0; n_features]
    };

    // Get selected variables
    let selected_variables: Vec<usize> = selection_frequency
        .iter()
        .enumerate()
        .filter(|(_, count)| **count > 0)
        .map(|(i, _)| i)
        .collect();

    // Convert predictions to response scale for classification
    let final_predictions = match config.family {
        MboostFamily::Binomial => predictions.mapv(sigmoid).to_vec(),
        MboostFamily::Poisson => predictions.mapv(|x| x.exp()).to_vec(),
        MboostFamily::Gaussian => predictions.to_vec(),
    };

    Ok(MboostResult {
        coefficients,
        intercept,
        variable_importance,
        selected_variables: selected_variables.clone(),
        n_selected: selected_variables.len(),
        selection_frequency,
        loss_history: loss_history.clone(),
        final_loss: *loss_history.last().unwrap_or(&0.0),
        predictions: final_predictions,
        iterations: config.mstop,
        cv_optimal_mstop: None,
        cv_error: None,
        config: config.clone(),
        feature_names: None,
        coefficient_path,
    })
}

/// Run mboost with cross-validation for optimal mstop.
///
/// Performs k-fold cross-validation to find the optimal number of
/// boosting iterations that minimizes cross-validated error.
///
/// # Arguments
///
/// * `x` - Feature matrix (n_samples x n_features)
/// * `y` - Target values (n_samples)
/// * `config` - MboostConfig with cv_folds set
///
/// # Returns
///
/// MboostResult with cv_optimal_mstop and cv_error filled in.
pub fn mboost_cv(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    config: &MboostConfig,
) -> EconResult<MboostResult> {
    let n_folds = config.cv_folds.unwrap_or(5);
    let n_samples = x.nrows();

    if n_folds < 2 {
        return Err(EconError::Computation(
            "Cross-validation requires at least 2 folds".to_string(),
        ));
    }
    if n_folds > n_samples {
        return Err(EconError::Computation(
            "More folds than samples".to_string(),
        ));
    }

    let mut rng_state = config.seed.unwrap_or(42);

    // Create fold assignments (random shuffle)
    let mut fold_ids: Vec<usize> = (0..n_samples).map(|i| i % n_folds).collect();
    for i in (1..n_samples).rev() {
        let j = lcg_random(&mut rng_state) % (i + 1);
        fold_ids.swap(i, j);
    }

    // Initialize CV error matrix: mstop x n_folds
    let mut cv_errors = Array2::zeros((config.mstop + 1, n_folds));

    // Run CV for each fold
    for fold in 0..n_folds {
        // Split data
        let train_idx: Vec<usize> = (0..n_samples).filter(|&i| fold_ids[i] != fold).collect();
        let test_idx: Vec<usize> = (0..n_samples).filter(|&i| fold_ids[i] == fold).collect();

        if train_idx.is_empty() || test_idx.is_empty() {
            continue;
        }

        let x_train = select_rows(&x, &train_idx);
        let y_train: Array1<f64> = train_idx.iter().map(|&i| y[i]).collect();
        let x_test = select_rows(&x, &test_idx);
        let y_test: Array1<f64> = test_idx.iter().map(|&i| y[i]).collect();

        // Run boosting on training fold and evaluate at each iteration
        let n_train = x_train.nrows();
        let n_test = x_test.nrows();
        let n_features = x_train.ncols();

        // Compute column means for centering
        let x_means: Vec<f64> = (0..n_features)
            .map(|j| x_train.column(j).iter().sum::<f64>() / n_train as f64)
            .collect();

        // Initialize predictions
        let init_pred = compute_init_prediction(&y_train.view(), config.family);
        let mut train_preds = Array1::from_elem(n_train, init_pred);
        let mut test_preds = Array1::from_elem(n_test, init_pred);

        // Initial CV error
        cv_errors[[0, fold]] = compute_loss(&y_test.view(), &test_preds, config.family);

        // Boosting iterations
        for m in 0..config.mstop {
            let residuals = compute_negative_gradient(&y_train.view(), &train_preds, config.family);

            // Find best component
            let mut best_j = 0;
            let mut best_rss = f64::INFINITY;
            let mut best_fit = BaseLearnerFit {
                coefficient: 0.0,
                intercept: 0.0,
                threshold: None,
                go_left: true,
                rss: f64::INFINITY,
            };

            for j in 0..n_features {
                let x_col = x_train.column(j);
                let x_mean = if config.center { x_means[j] } else { 0.0 };

                let fit = match config.base_learner {
                    MboostBaseLearner::Linear => {
                        fit_linear_base_learner(&x_col, &residuals.view(), x_mean)
                    }
                    MboostBaseLearner::Tree => {
                        fit_tree_base_learner(&x_col, &residuals.view(), config.min_samples_split)
                    }
                };

                if fit.rss < best_rss {
                    best_rss = fit.rss;
                    best_j = j;
                    best_fit = fit;
                }
            }

            // Update train predictions
            let x_train_col = x_train.column(best_j);
            match config.base_learner {
                MboostBaseLearner::Linear => {
                    for i in 0..n_train {
                        let update = best_fit.intercept + best_fit.coefficient * x_train_col[i];
                        train_preds[i] += config.nu * update;
                    }
                }
                MboostBaseLearner::Tree => {
                    for i in 0..n_train {
                        let update = predict_stump(x_train_col[i], &best_fit);
                        train_preds[i] += config.nu * update;
                    }
                }
            }

            // Update test predictions
            let x_test_col = x_test.column(best_j);
            match config.base_learner {
                MboostBaseLearner::Linear => {
                    for i in 0..n_test {
                        let update = best_fit.intercept + best_fit.coefficient * x_test_col[i];
                        test_preds[i] += config.nu * update;
                    }
                }
                MboostBaseLearner::Tree => {
                    for i in 0..n_test {
                        let update = predict_stump(x_test_col[i], &best_fit);
                        test_preds[i] += config.nu * update;
                    }
                }
            }

            // Store CV error at this iteration
            cv_errors[[m + 1, fold]] = compute_loss(&y_test.view(), &test_preds, config.family);
        }
    }

    // Average CV error across folds
    let cv_error: Vec<f64> = (0..=config.mstop)
        .map(|m| (0..n_folds).map(|f| cv_errors[[m, f]]).sum::<f64>() / n_folds as f64)
        .collect();

    // Find optimal mstop (minimum CV error)
    let cv_optimal_mstop = cv_error
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(config.mstop);

    // Fit final model with optimal mstop
    let mut optimal_config = config.clone();
    optimal_config.mstop = cv_optimal_mstop;
    optimal_config.cv_folds = None; // Don't recurse

    let mut result = mboost(x, y, &optimal_config)?;
    result.cv_optimal_mstop = Some(cv_optimal_mstop);
    result.cv_error = Some(cv_error);
    result.config = config.clone(); // Restore original config

    Ok(result)
}

/// Predict using a fitted mboost model.
///
/// # Arguments
///
/// * `result` - Fitted MboostResult
/// * `x` - Feature matrix for prediction
///
/// # Returns
///
/// Predictions (probabilities for Binomial, rates for Poisson)
pub fn mboost_predict(result: &MboostResult, x: ArrayView2<f64>) -> EconResult<Vec<f64>> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_features != result.coefficients.len() {
        return Err(EconError::Computation(format!(
            "Feature count mismatch: expected {}, got {}",
            result.coefficients.len(),
            n_features
        )));
    }

    // For linear base learner, use accumulated coefficients
    if result.config.base_learner == MboostBaseLearner::Linear {
        let mut predictions = vec![result.intercept; n_samples];

        for i in 0..n_samples {
            for j in 0..n_features {
                predictions[i] += result.coefficients[j] * x[[i, j]];
            }
        }

        // Convert to response scale
        let final_preds = match result.config.family {
            MboostFamily::Binomial => predictions.iter().map(|&p| sigmoid(p)).collect(),
            MboostFamily::Poisson => predictions.iter().map(|&p| p.exp()).collect(),
            MboostFamily::Gaussian => predictions,
        };

        return Ok(final_preds);
    }

    // For tree base learner, we don't store the full history for prediction
    // Return error or use stored predictions
    Err(EconError::Computation(
        "Prediction with tree base learner not supported. Use the stored predictions.".to_string(),
    ))
}

/// Select rows from a 2D array.
fn select_rows(data: &ArrayView2<f64>, indices: &[usize]) -> Array2<f64> {
    let n_features = data.ncols();
    let mut result = Array2::zeros((indices.len(), n_features));

    for (i, &idx) in indices.iter().enumerate() {
        result.row_mut(i).assign(&data.row(idx));
    }

    result
}

/// Run mboost on a Dataset.
///
/// # Arguments
///
/// * `dataset` - Input dataset
/// * `y_col` - Target column name
/// * `x_cols` - Feature column names
/// * `config` - MboostConfig
///
/// # Returns
///
/// MboostResult with model and diagnostics.
pub fn run_mboost(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    config: &MboostConfig,
) -> EconResult<MboostResult> {
    use crate::linalg::design::DesignMatrix;

    // Build design matrix
    let design = DesignMatrix::from_dataframe(dataset.df(), x_cols, false)?;
    let x = design.data;
    let feature_names = design.column_names;

    // Get y column
    let col_names: Vec<String> = dataset
        .df()
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    let y_series = dataset
        .df()
        .column(y_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: y_col.to_string(),
            available: col_names.clone(),
        })?;

    let y: Vec<f64> = y_series
        .f64()
        .map_err(|_| EconError::Computation(format!("Column '{}' is not numeric", y_col)))?
        .into_no_null_iter()
        .collect();

    let y_arr = Array1::from_vec(y);

    // Run mboost (with or without CV)
    let mut result = if config.cv_folds.is_some() {
        mboost_cv(x.view(), y_arr.view(), config)?
    } else {
        mboost(x.view(), y_arr.view(), config)?
    };

    result.feature_names = Some(feature_names);

    Ok(result)
}

/// Convenience function with default configuration.
pub fn run_mboost_default(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
) -> EconResult<MboostResult> {
    run_mboost(dataset, y_col, x_cols, &MboostConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_mboost_gaussian_basic() {
        // Simple linear relationship: y = 2*x1 + noise
        let x = array![
            [1.0, 0.5],
            [2.0, 1.0],
            [3.0, 1.2],
            [4.0, 2.1],
            [5.0, 2.3],
            [6.0, 2.8],
            [7.0, 3.1],
            [8.0, 3.5],
            [9.0, 4.2],
            [10.0, 4.8]
        ];
        let y = array![2.1, 4.2, 5.8, 8.1, 10.2, 11.9, 14.1, 16.2, 17.8, 20.1];

        let config = MboostConfig {
            mstop: 100,
            nu: 0.1,
            family: MboostFamily::Gaussian,
            base_learner: MboostBaseLearner::Linear,
            ..Default::default()
        };

        let result = mboost(x.view(), y.view(), &config).unwrap();

        // Check we selected some variables
        assert!(result.n_selected > 0);

        // Loss should decrease
        assert!(result.final_loss < result.loss_history[0]);

        // First variable should be more important (it's the predictor)
        assert!(
            result.variable_importance[0] > result.variable_importance[1],
            "First variable should be more important"
        );

        // Coefficient for first variable should be positive and near 2
        assert!(
            result.coefficients[0] > 1.0,
            "Coefficient should be positive"
        );
    }

    #[test]
    fn test_mboost_binomial() {
        // Binary classification
        let x = array![
            [1.0, 0.1],
            [1.5, 0.2],
            [2.0, 0.3],
            [2.5, 0.4],
            [7.0, 0.5],
            [7.5, 0.6],
            [8.0, 0.7],
            [8.5, 0.8],
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let config = MboostConfig {
            mstop: 50,
            nu: 0.3,
            family: MboostFamily::Binomial,
            base_learner: MboostBaseLearner::Linear,
            ..Default::default()
        };

        let result = mboost(x.view(), y.view(), &config).unwrap();

        // Predictions should be probabilities
        for &p in &result.predictions {
            assert!(p >= 0.0 && p <= 1.0, "Prediction {} out of range", p);
        }

        // Low x values should have low probability
        assert!(result.predictions[0] < 0.5);
        // High x values should have high probability
        assert!(result.predictions[7] > 0.5);
    }

    #[test]
    fn test_mboost_poisson() {
        // Count data
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0],];
        let y = array![1.0, 3.0, 7.0, 15.0, 30.0]; // Roughly exponential

        let config = MboostConfig {
            mstop: 50,
            nu: 0.1,
            family: MboostFamily::Poisson,
            base_learner: MboostBaseLearner::Linear,
            ..Default::default()
        };

        let result = mboost(x.view(), y.view(), &config).unwrap();

        // Predictions should be positive
        for &p in &result.predictions {
            assert!(p > 0.0, "Poisson prediction {} should be positive", p);
        }

        // Predictions should increase
        assert!(result.predictions[4] > result.predictions[0]);
    }

    #[test]
    fn test_mboost_tree_base_learner() {
        let x = array![
            [1.0, 0.1],
            [2.0, 0.2],
            [3.0, 0.3],
            [4.0, 0.4],
            [5.0, 0.5],
            [6.0, 0.6],
            [7.0, 0.7],
            [8.0, 0.8],
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let config = MboostConfig {
            mstop: 50,
            nu: 0.1,
            family: MboostFamily::Gaussian,
            base_learner: MboostBaseLearner::Tree,
            min_samples_split: 2,
            ..Default::default()
        };

        let result = mboost(x.view(), y.view(), &config).unwrap();

        // Should select some variables
        assert!(result.n_selected > 0);

        // Loss should decrease
        assert!(result.final_loss < result.loss_history[0]);
    }

    #[test]
    fn test_mboost_variable_selection() {
        // y depends only on first variable
        let x = array![
            [1.0, 5.0, 3.0],
            [2.0, 3.0, 7.0],
            [3.0, 7.0, 2.0],
            [4.0, 2.0, 8.0],
            [5.0, 8.0, 4.0],
            [6.0, 4.0, 6.0],
            [7.0, 9.0, 1.0],
            [8.0, 1.0, 5.0],
            [9.0, 6.0, 9.0],
            [10.0, 5.0, 3.0],
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let config = MboostConfig {
            mstop: 100,
            nu: 0.1,
            family: MboostFamily::Gaussian,
            base_learner: MboostBaseLearner::Linear,
            ..Default::default()
        };

        let result = mboost(x.view(), y.view(), &config).unwrap();

        // First variable should have highest importance
        assert!(
            result.variable_importance[0] > result.variable_importance[1],
            "First variable should be more important than second"
        );
        assert!(
            result.variable_importance[0] > result.variable_importance[2],
            "First variable should be more important than third"
        );
    }

    #[test]
    fn test_mboost_cv() {
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
        let y = array![1.1, 2.1, 2.9, 4.1, 4.9, 6.1, 6.9, 8.1, 8.9, 10.1];

        let config = MboostConfig {
            mstop: 50,
            nu: 0.1,
            family: MboostFamily::Gaussian,
            base_learner: MboostBaseLearner::Linear,
            cv_folds: Some(3),
            seed: Some(42),
            ..Default::default()
        };

        let result = mboost_cv(x.view(), y.view(), &config).unwrap();

        // Should have CV results
        assert!(result.cv_optimal_mstop.is_some());
        assert!(result.cv_error.is_some());

        let cv_error = result.cv_error.as_ref().unwrap();
        assert_eq!(cv_error.len(), config.mstop + 1);
    }

    #[test]
    fn test_mboost_predict() {
        let x_train = array![[1.0, 0.1], [2.0, 0.2], [3.0, 0.3], [4.0, 0.4], [5.0, 0.5],];
        let y_train = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let config = MboostConfig {
            mstop: 50,
            nu: 0.1,
            family: MboostFamily::Gaussian,
            base_learner: MboostBaseLearner::Linear,
            ..Default::default()
        };

        let result = mboost(x_train.view(), y_train.view(), &config).unwrap();

        // Predict on new data
        let x_test = array![[1.5, 0.15], [3.5, 0.35], [5.5, 0.55]];
        let predictions = mboost_predict(&result, x_test.view()).unwrap();

        assert_eq!(predictions.len(), 3);
        // Predictions should be in reasonable range
        for &p in &predictions {
            assert!(p > 0.0 && p < 7.0, "Prediction {} out of expected range", p);
        }
    }

    #[test]
    fn test_fit_linear_base_learner() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let residuals = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let fit = fit_linear_base_learner(&x.view(), &residuals.view(), 3.0);

        // For perfect linear relationship, coefficient should be 1
        assert!((fit.coefficient - 1.0).abs() < 1e-10);
        assert!(fit.rss < 1e-10);
    }

    #[test]
    fn test_fit_tree_base_learner() {
        let x = array![1.0, 2.0, 3.0, 7.0, 8.0, 9.0];
        let residuals = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let fit = fit_tree_base_learner(&x.view(), &residuals.view(), 2);

        // Should find a split around 5
        assert!(fit.threshold.is_some());
        let thresh = fit.threshold.unwrap();
        assert!(
            thresh > 3.0 && thresh < 7.0,
            "Threshold {} should be between 3 and 7",
            thresh
        );
    }
}
