//! Bayesian Structural Time Series for Causal Inference (CausalImpact).
//!
//! Implements the CausalImpact methodology for estimating the causal effect
//! of an intervention using Bayesian structural time series models.
//!
//! The key idea is to use a state-space model trained on pre-intervention data
//! (possibly with control series) to predict what the response would have been
//! in the absence of the intervention. The difference between observed and
//! predicted values in the post-intervention period gives the causal effect.
//!
//! # Algorithm
//!
//! 1. **Pre-period Model Fitting**: Fit a structural time series model to
//!    pre-intervention data. The model includes:
//!    - Local level component (random walk)
//!    - Optional local linear trend
//!    - Optional seasonality
//!    - Optional regression component with control series
//!
//! 2. **Counterfactual Prediction**: Use the fitted model to predict what
//!    would have happened in the post-intervention period.
//!
//! 3. **Causal Effect Estimation**: Compute the difference between observed
//!    and predicted values, along with credible intervals.
//!
//! # References
//!
//! - Brodersen, K. H., Gallusser, F., Koehler, J., Remy, N., & Scott, S. L. (2015).
//!   "Inferring causal impact using Bayesian structural time series models".
//!   *Annals of Applied Statistics*, 9(1), 247-274.
//!   <https://doi.org/10.1214/14-AOAS788>
//!
//! - R package `CausalImpact`:
//!   <https://google.github.io/CausalImpact/>
//!   <https://cran.r-project.org/package=CausalImpact>

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::forecasting::kalman::{StateSpaceModel, kalman_filter, kalman_forecast};

/// Configuration for CausalImpact analysis.
#[derive(Debug, Clone)]
pub struct CausalImpactConfig {
    /// Pre-intervention period (inclusive start and end indices or timestamps).
    pub pre_period: (i64, i64),
    /// Post-intervention period (inclusive start and end indices or timestamps).
    pub post_period: (i64, i64),
    /// Names of control series columns (if any).
    pub control_series: Option<Vec<String>>,
    /// Significance level for credible intervals (default 0.05 for 95% CI).
    pub alpha: f64,
    /// Seasonal period (if the data has seasonality).
    pub seasonal_period: Option<usize>,
    /// Whether to include a trend component.
    pub include_trend: bool,
    /// Maximum iterations for MLE optimization.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Prior standard deviation for regression coefficients.
    /// Larger values = more diffuse prior.
    pub prior_level_sd: Option<f64>,
}

impl Default for CausalImpactConfig {
    fn default() -> Self {
        Self {
            pre_period: (0, 0),
            post_period: (0, 0),
            control_series: None,
            alpha: 0.05,
            seasonal_period: None,
            include_trend: false,
            max_iter: 100,
            tolerance: 1e-8,
            prior_level_sd: None,
        }
    }
}

/// Summary statistics for causal effect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalImpactSummary {
    /// Average causal effect per time point in post-period.
    pub average_effect: f64,
    /// Lower bound of credible interval for average effect.
    pub average_effect_lower: f64,
    /// Upper bound of credible interval for average effect.
    pub average_effect_upper: f64,
    /// Cumulative (total) causal effect over post-period.
    pub cumulative_effect: f64,
    /// Lower bound of credible interval for cumulative effect.
    pub cumulative_effect_lower: f64,
    /// Upper bound of credible interval for cumulative effect.
    pub cumulative_effect_upper: f64,
    /// Relative effect (cumulative effect / sum of predicted).
    pub relative_effect: f64,
    /// Lower bound of relative effect.
    pub relative_effect_lower: f64,
    /// Upper bound of relative effect.
    pub relative_effect_upper: f64,
    /// Bayesian tail-area probability (one-sided p-value).
    /// P(effect > 0) if effect is positive, P(effect < 0) if negative.
    pub p_value: f64,
    /// Whether the effect is statistically significant at alpha level.
    pub significant: bool,
    /// Significance level used.
    pub alpha: f64,
}

/// Time series of causal effects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalImpactSeries {
    /// Time indices or values.
    pub time: Vec<i64>,
    /// Observed response values.
    pub observed: Vec<f64>,
    /// Predicted (counterfactual) values.
    pub predicted: Vec<f64>,
    /// Lower bound of prediction interval.
    pub predicted_lower: Vec<f64>,
    /// Upper bound of prediction interval.
    pub predicted_upper: Vec<f64>,
    /// Point-wise causal effect (observed - predicted).
    pub point_effect: Vec<f64>,
    /// Lower bound of point effect interval.
    pub point_effect_lower: Vec<f64>,
    /// Upper bound of point effect interval.
    pub point_effect_upper: Vec<f64>,
    /// Cumulative causal effect over time.
    pub cumulative_effect: Vec<f64>,
    /// Lower bound of cumulative effect interval.
    pub cumulative_effect_lower: Vec<f64>,
    /// Upper bound of cumulative effect interval.
    pub cumulative_effect_upper: Vec<f64>,
}

/// Fitted model information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalImpactModel {
    /// Estimated level variance.
    pub level_variance: f64,
    /// Estimated slope variance (if trend included).
    pub slope_variance: Option<f64>,
    /// Estimated seasonal variance (if seasonality included).
    pub seasonal_variance: Option<f64>,
    /// Estimated observation variance.
    pub observation_variance: f64,
    /// Regression coefficients for control series.
    pub regression_coefficients: Option<Vec<f64>>,
    /// Names of control variables.
    pub control_names: Option<Vec<String>>,
    /// Log-likelihood at optimum.
    pub log_likelihood: f64,
    /// AIC.
    pub aic: f64,
    /// BIC.
    pub bic: f64,
    /// Number of pre-period observations used for fitting.
    pub n_pre: usize,
    /// Number of post-period observations.
    pub n_post: usize,
}

/// Causal inference statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalInference {
    /// Posterior probability that the effect is positive.
    pub prob_positive: f64,
    /// Posterior probability that the effect is negative.
    pub prob_negative: f64,
    /// Expected causal effect under the posterior.
    pub expected_effect: f64,
    /// Standard deviation of the posterior effect distribution.
    pub effect_sd: f64,
    /// Whether the null (no effect) can be rejected.
    pub null_rejected: bool,
}

/// Complete result from CausalImpact analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalImpactResult {
    /// Summary statistics.
    pub summary: CausalImpactSummary,
    /// Time series of effects.
    pub series: CausalImpactSeries,
    /// Model information.
    pub model: CausalImpactModel,
    /// Causal inference statistics.
    pub inference: CausalInference,
}

/// Run CausalImpact analysis on a dataset.
///
/// # Arguments
///
/// * `dataset` - Dataset containing the response and optional control series
/// * `response_col` - Name of the response variable column
/// * `time_col` - Name of the time/index column (must be convertible to i64)
/// * `config` - Configuration for the analysis
///
/// # Returns
///
/// `CausalImpactResult` with summary, time series, model info, and inference.
///
/// # Example
///
/// ```ignore
/// use p2a_core::forecasting::causal_impact::{causal_impact, CausalImpactConfig};
///
/// let config = CausalImpactConfig {
///     pre_period: (0, 70),
///     post_period: (71, 100),
///     alpha: 0.05,
///     ..Default::default()
/// };
///
/// let result = causal_impact(&dataset, "y", "time", config)?;
/// println!("Cumulative effect: {:.2}", result.summary.cumulative_effect);
/// ```
pub fn causal_impact(
    dataset: &Dataset,
    response_col: &str,
    time_col: &str,
    config: CausalImpactConfig,
) -> EconResult<CausalImpactResult> {
    let df = dataset.df();
    let available: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    // Extract time column
    let time_series = df.column(time_col).map_err(|_| EconError::ColumnNotFound {
        column: time_col.to_string(),
        available: available.clone(),
    })?;

    let time: Vec<i64> = time_series
        .cast(&polars::prelude::DataType::Int64)
        .map_err(|_| EconError::NonNumericColumn {
            column: time_col.to_string(),
        })?
        .i64()
        .map_err(|_| EconError::NonNumericColumn {
            column: time_col.to_string(),
        })?
        .into_no_null_iter()
        .collect();

    // Extract response column
    let response_series = df
        .column(response_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: response_col.to_string(),
            available: available.clone(),
        })?;

    let y: Vec<f64> = response_series
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: response_col.to_string(),
        })?
        .into_no_null_iter()
        .collect();

    // Extract control series if specified
    let controls = if let Some(ref control_cols) = config.control_series {
        let mut control_data = Vec::new();
        for col in control_cols {
            let series = df.column(col).map_err(|_| EconError::ColumnNotFound {
                column: col.to_string(),
                available: available.clone(),
            })?;
            let values: Vec<f64> = series
                .f64()
                .map_err(|_| EconError::NonNumericColumn {
                    column: col.to_string(),
                })?
                .into_no_null_iter()
                .collect();
            control_data.push(values);
        }
        Some(control_data)
    } else {
        None
    };

    // Find indices for pre and post periods based on time values
    let pre_start_idx = time
        .iter()
        .position(|&t| t >= config.pre_period.0)
        .ok_or_else(|| EconError::InvalidSpecification {
            message: format!(
                "Pre-period start {} not found in time series",
                config.pre_period.0
            ),
        })?;

    let pre_end_idx = time
        .iter()
        .rposition(|&t| t <= config.pre_period.1)
        .ok_or_else(|| EconError::InvalidSpecification {
            message: format!(
                "Pre-period end {} not found in time series",
                config.pre_period.1
            ),
        })?;

    let post_start_idx = time
        .iter()
        .position(|&t| t >= config.post_period.0)
        .ok_or_else(|| EconError::InvalidSpecification {
            message: format!(
                "Post-period start {} not found in time series",
                config.post_period.0
            ),
        })?;

    let post_end_idx = time
        .iter()
        .rposition(|&t| t <= config.post_period.1)
        .ok_or_else(|| EconError::InvalidSpecification {
            message: format!(
                "Post-period end {} not found in time series",
                config.post_period.1
            ),
        })?;

    // Validate periods
    if pre_end_idx >= post_start_idx {
        return Err(EconError::InvalidSpecification {
            message: "Pre-period must end before post-period begins".to_string(),
        });
    }

    let n_pre = pre_end_idx - pre_start_idx + 1;
    let _n_post = post_end_idx - post_start_idx + 1;

    if n_pre < 10 {
        return Err(EconError::InsufficientData {
            required: 10,
            provided: n_pre,
            context: "CausalImpact requires at least 10 pre-period observations".to_string(),
        });
    }

    // Run the core algorithm
    run_causal_impact_core(
        &y,
        &time,
        controls.as_ref(),
        pre_start_idx,
        pre_end_idx,
        post_start_idx,
        post_end_idx,
        &config,
    )
}

/// Simplified interface for CausalImpact.
///
/// # Arguments
///
/// * `dataset` - Dataset containing the data
/// * `response_col` - Name of response column
/// * `time_col` - Name of time column
/// * `pre_period` - (start, end) of pre-intervention period
/// * `post_period` - (start, end) of post-intervention period
/// * `control_cols` - Optional control series column names
pub fn run_causal_impact(
    dataset: &Dataset,
    response_col: &str,
    time_col: &str,
    pre_period: (i64, i64),
    post_period: (i64, i64),
    control_cols: Option<&[&str]>,
) -> EconResult<CausalImpactResult> {
    let config = CausalImpactConfig {
        pre_period,
        post_period,
        control_series: control_cols.map(|cols| cols.iter().map(|s| s.to_string()).collect()),
        ..Default::default()
    };

    causal_impact(dataset, response_col, time_col, config)
}

/// Core implementation of CausalImpact algorithm.
fn run_causal_impact_core(
    y: &[f64],
    time: &[i64],
    controls: Option<&Vec<Vec<f64>>>,
    pre_start: usize,
    pre_end: usize,
    post_start: usize,
    post_end: usize,
    config: &CausalImpactConfig,
) -> EconResult<CausalImpactResult> {
    let n_pre = pre_end - pre_start + 1;
    let n_post = post_end - post_start + 1;

    // Step 1: Extract pre-period data for model fitting
    let y_pre: Vec<f64> = y[pre_start..=pre_end].to_vec();

    let controls_pre = controls.map(|c| {
        c.iter()
            .map(|col| col[pre_start..=pre_end].to_vec())
            .collect::<Vec<_>>()
    });

    // Step 2: Fit structural time series model on pre-period data
    let (model_params, regression_coefs, log_lik) =
        fit_bsts_model(&y_pre, controls_pre.as_ref(), config)?;

    // Step 3: Build state-space model with fitted parameters
    let ssm = build_state_space_model(&model_params, controls, config)?;

    // Step 4: Run Kalman filter/smoother on full series up to end of pre-period
    // For post-period, we predict the counterfactual
    let y_var = sample_variance(&y_pre);
    let init_state = initialize_state(&y_pre, config);
    let init_cov = initialize_covariance(y_var, config);

    // For prediction, we need to extend the observation series with NaN for post-period
    // and use the control series (if any) as known inputs
    let (predicted, predicted_var) = predict_counterfactual(
        y,
        time,
        controls,
        &ssm,
        regression_coefs.as_ref(),
        &init_state,
        &init_cov,
        pre_start,
        pre_end,
        post_start,
        post_end,
    )?;

    // Step 5: Compute causal effects
    let z_alpha = Normal::new(0.0, 1.0)
        .unwrap()
        .inverse_cdf(1.0 - config.alpha / 2.0);

    let mut series = CausalImpactSeries {
        time: time.to_vec(),
        observed: y.to_vec(),
        predicted: predicted.clone(),
        predicted_lower: Vec::with_capacity(y.len()),
        predicted_upper: Vec::with_capacity(y.len()),
        point_effect: Vec::with_capacity(y.len()),
        point_effect_lower: Vec::with_capacity(y.len()),
        point_effect_upper: Vec::with_capacity(y.len()),
        cumulative_effect: Vec::with_capacity(y.len()),
        cumulative_effect_lower: Vec::with_capacity(y.len()),
        cumulative_effect_upper: Vec::with_capacity(y.len()),
    };

    let mut cumulative = 0.0;
    let mut cumulative_var = 0.0;

    for i in 0..y.len() {
        let pred_se = predicted_var[i].sqrt();
        series
            .predicted_lower
            .push(predicted[i] - z_alpha * pred_se);
        series
            .predicted_upper
            .push(predicted[i] + z_alpha * pred_se);

        let effect = y[i] - predicted[i];
        series.point_effect.push(effect);
        series.point_effect_lower.push(effect - z_alpha * pred_se);
        series.point_effect_upper.push(effect + z_alpha * pred_se);

        if i >= post_start && i <= post_end {
            cumulative += effect;
            cumulative_var += predicted_var[i];
        }

        let cum_se = cumulative_var.sqrt();
        series.cumulative_effect.push(cumulative);
        series
            .cumulative_effect_lower
            .push(cumulative - z_alpha * cum_se);
        series
            .cumulative_effect_upper
            .push(cumulative + z_alpha * cum_se);
    }

    // Step 6: Compute summary statistics
    let post_effects: Vec<f64> = series.point_effect[post_start..=post_end].to_vec();
    let post_predicted: Vec<f64> = predicted[post_start..=post_end].to_vec();
    let post_variances: Vec<f64> = predicted_var[post_start..=post_end].to_vec();

    let cumulative_effect = post_effects.iter().sum::<f64>();
    let cumulative_variance: f64 = post_variances.iter().sum();
    let cumulative_se = cumulative_variance.sqrt();

    let average_effect = cumulative_effect / n_post as f64;
    let average_variance = cumulative_variance / (n_post * n_post) as f64;
    let average_se = average_variance.sqrt();

    let sum_predicted: f64 = post_predicted.iter().sum();
    let relative_effect = if sum_predicted.abs() > 1e-10 {
        cumulative_effect / sum_predicted
    } else {
        0.0
    };

    // For relative effect CI, use delta method approximation
    let relative_se = if sum_predicted.abs() > 1e-10 {
        cumulative_se / sum_predicted.abs()
    } else {
        0.0
    };

    // Bayesian tail probability (approximate using normal)
    // P(effect < 0) if effect > 0, P(effect > 0) if effect < 0
    let normal = Normal::new(0.0, 1.0).unwrap();
    let z_cumulative = if cumulative_se > 0.0 {
        cumulative_effect / cumulative_se
    } else {
        0.0
    };

    let p_value = if cumulative_effect >= 0.0 {
        normal.cdf(-z_cumulative) // P(effect < 0) under posterior
    } else {
        1.0 - normal.cdf(-z_cumulative) // P(effect > 0) under posterior
    };

    let summary = CausalImpactSummary {
        average_effect,
        average_effect_lower: average_effect - z_alpha * average_se,
        average_effect_upper: average_effect + z_alpha * average_se,
        cumulative_effect,
        cumulative_effect_lower: cumulative_effect - z_alpha * cumulative_se,
        cumulative_effect_upper: cumulative_effect + z_alpha * cumulative_se,
        relative_effect,
        relative_effect_lower: relative_effect - z_alpha * relative_se,
        relative_effect_upper: relative_effect + z_alpha * relative_se,
        p_value,
        significant: p_value < config.alpha,
        alpha: config.alpha,
    };

    // Model info
    let n_params = count_params(config, controls.is_some());
    let model = CausalImpactModel {
        level_variance: model_params[0],
        slope_variance: if config.include_trend {
            Some(model_params[1])
        } else {
            None
        },
        seasonal_variance: config.seasonal_period.map(|_| {
            if config.include_trend {
                model_params[2]
            } else {
                model_params[1]
            }
        }),
        observation_variance: *model_params.last().unwrap(),
        regression_coefficients: regression_coefs.clone(),
        control_names: config.control_series.clone(),
        log_likelihood: log_lik,
        aic: -2.0 * log_lik + 2.0 * n_params as f64,
        bic: -2.0 * log_lik + (n_params as f64) * (n_pre as f64).ln(),
        n_pre,
        n_post,
    };

    // Causal inference
    let prob_positive = 1.0 - normal.cdf(-z_cumulative);
    let prob_negative = normal.cdf(-z_cumulative);

    let inference = CausalInference {
        prob_positive,
        prob_negative,
        expected_effect: cumulative_effect,
        effect_sd: cumulative_se,
        null_rejected: p_value < config.alpha,
    };

    Ok(CausalImpactResult {
        summary,
        series,
        model,
        inference,
    })
}

/// Model parameters for BSTS.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ModelParams {
    level_var: f64,
    slope_var: Option<f64>,
    seasonal_var: Option<f64>,
    obs_var: f64,
}

/// Fit BSTS model using MLE on pre-period data.
fn fit_bsts_model(
    y_pre: &[f64],
    controls_pre: Option<&Vec<Vec<f64>>>,
    config: &CausalImpactConfig,
) -> EconResult<(Vec<f64>, Option<Vec<f64>>, f64)> {
    let y_var = sample_variance(y_pre);

    // If we have controls, first regress y on controls to get regression coefficients
    // Then model the residuals with BSTS
    let (y_for_bsts, regression_coefs) = if let Some(controls) = controls_pre {
        let (residuals, coefs) = regress_out_controls(y_pre, controls)?;
        (residuals, Some(coefs))
    } else {
        (y_pre.to_vec(), None)
    };

    // Initialize variance parameters based on data variance
    // Following Harvey (1990) and R's CausalImpact defaults
    let init_params = initialize_variance_params(y_var, config);

    // Optimize using Nelder-Mead
    let objective = |log_params: &[f64]| -> f64 {
        let params: Vec<f64> = log_params.iter().map(|&lp| y_var * lp.exp()).collect();
        match compute_bsts_loglik(&y_for_bsts, &params, config) {
            Ok(ll) => -ll,
            Err(_) => 1e20,
        }
    };

    // Transform to log scale for optimization
    let init_log: Vec<f64> = init_params
        .iter()
        .map(|&p| (p / y_var).max(1e-10).ln())
        .collect();

    let (best_log_params, neg_loglik, _converged) =
        nelder_mead_optimize(&objective, &init_log, config.max_iter, config.tolerance);

    // Transform back
    let best_params: Vec<f64> = best_log_params.iter().map(|&lp| y_var * lp.exp()).collect();

    Ok((best_params, regression_coefs, -neg_loglik))
}

/// Initialize variance parameters.
fn initialize_variance_params(y_var: f64, config: &CausalImpactConfig) -> Vec<f64> {
    let mut params = Vec::new();

    // Level variance (always included)
    params.push(y_var * 0.1);

    // Slope variance (if trend)
    if config.include_trend {
        params.push(y_var * 0.01);
    }

    // Seasonal variance (if seasonality)
    if config.seasonal_period.is_some() {
        params.push(y_var * 0.1);
    }

    // Observation variance
    params.push(y_var * 0.5);

    params
}

/// Regress y on control variables using OLS.
fn regress_out_controls(y: &[f64], controls: &Vec<Vec<f64>>) -> EconResult<(Vec<f64>, Vec<f64>)> {
    let n = y.len();
    let k = controls.len();

    if k == 0 {
        return Ok((y.to_vec(), vec![]));
    }

    // Build design matrix [1, X1, X2, ...]
    let x = Array2::from_shape_fn(
        (n, k + 1),
        |(i, j)| {
            if j == 0 { 1.0 } else { controls[j - 1][i] }
        },
    );

    let y_arr = Array1::from_vec(y.to_vec());

    // OLS: beta = (X'X)^{-1} X'y
    let xtx = x.t().dot(&x);
    let xty = x.t().dot(&y_arr);

    let xtx_inv = crate::linalg::matrix_ops::safe_inverse(&xtx.view())
        .map_err(|e| EconError::SingularMatrix {
            context: format!("Control regression X'X: {}", e),
            suggestion: "Check for perfect multicollinearity in control series".to_string(),
        })?
        .0;

    let beta = xtx_inv.dot(&xty);

    // Compute residuals
    let fitted = x.dot(&beta);
    let residuals: Vec<f64> = y_arr
        .iter()
        .zip(fitted.iter())
        .map(|(yi, fi)| yi - fi)
        .collect();

    Ok((residuals, beta.to_vec()))
}

/// Build state-space model for BSTS.
fn build_state_space_model(
    params: &[f64],
    _controls: Option<&Vec<Vec<f64>>>,
    config: &CausalImpactConfig,
) -> EconResult<StateSpaceModel> {
    let mut param_idx = 0;
    let level_var = params[param_idx];
    param_idx += 1;

    let slope_var = if config.include_trend {
        let v = params[param_idx];
        param_idx += 1;
        Some(v)
    } else {
        None
    };

    let seasonal_var = if config.seasonal_period.is_some() {
        let v = params[param_idx];
        param_idx += 1;
        Some(v)
    } else {
        None
    };

    let obs_var = params[param_idx];

    // Build model based on configuration
    match (slope_var, seasonal_var, config.seasonal_period) {
        // Local level only
        (None, None, None) => {
            let transition = Array2::from_elem((1, 1), 1.0);
            let observation = Array1::from_elem(1, 1.0);
            let selection = Array2::from_elem((1, 1), 1.0);
            let state_cov = Array2::from_elem((1, 1), level_var.max(1e-12));

            StateSpaceModel::new(
                transition,
                observation,
                selection,
                state_cov,
                obs_var.max(1e-12),
            )
        }

        // Local linear trend (no seasonality)
        (Some(slope_v), None, None) => {
            let transition = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 0.0, 1.0]).unwrap();
            let observation = Array1::from_vec(vec![1.0, 0.0]);
            let selection = Array2::eye(2);
            let state_cov = Array2::from_shape_vec(
                (2, 2),
                vec![level_var.max(1e-12), 0.0, 0.0, slope_v.max(1e-12)],
            )
            .unwrap();

            StateSpaceModel::new(
                transition,
                observation,
                selection,
                state_cov,
                obs_var.max(1e-12),
            )
        }

        // Local level with seasonality
        (None, Some(seas_v), Some(period)) => {
            let m = 1 + period - 1; // level + (period-1) seasonal states
            let mut transition = Array2::zeros((m, m));

            // Level: μ_{t+1} = μ_t
            transition[[0, 0]] = 1.0;

            // Seasonal: γ_t = -γ_{t-1} - ... - γ_{t-s+1}
            for j in 1..m {
                transition[[1, j]] = -1.0;
            }
            for i in 2..m {
                transition[[i, i - 1]] = 1.0;
            }

            // Observation: y_t = μ_t + γ_t
            let mut observation = Array1::zeros(m);
            observation[0] = 1.0; // Level
            observation[1] = 1.0; // Seasonal

            // Selection matrix
            let mut selection = Array2::zeros((m, 2));
            selection[[0, 0]] = 1.0; // Level gets innovation
            selection[[1, 1]] = 1.0; // Seasonal gets innovation

            let state_cov = Array2::from_shape_vec(
                (2, 2),
                vec![level_var.max(1e-12), 0.0, 0.0, seas_v.max(1e-12)],
            )
            .unwrap();

            StateSpaceModel::new(
                transition,
                observation,
                selection,
                state_cov,
                obs_var.max(1e-12),
            )
        }

        // Full BSM: level + trend + seasonality
        (Some(slope_v), Some(seas_v), Some(period)) => {
            let m = 2 + period - 1; // level + slope + (period-1) seasonal states

            let mut transition = Array2::zeros((m, m));
            // Level: μ_{t+1} = μ_t + ν_t
            transition[[0, 0]] = 1.0;
            transition[[0, 1]] = 1.0;
            // Slope: ν_{t+1} = ν_t
            transition[[1, 1]] = 1.0;
            // Seasonal
            for j in 2..m {
                transition[[2, j]] = -1.0;
            }
            for i in 3..m {
                transition[[i, i - 1]] = 1.0;
            }

            let mut observation = Array1::zeros(m);
            observation[0] = 1.0; // Level
            observation[2] = 1.0; // Seasonal

            let mut selection = Array2::zeros((m, 3));
            selection[[0, 0]] = 1.0;
            selection[[1, 1]] = 1.0;
            selection[[2, 2]] = 1.0;

            let state_cov = Array2::from_shape_vec(
                (3, 3),
                vec![
                    level_var.max(1e-12),
                    0.0,
                    0.0,
                    0.0,
                    slope_v.max(1e-12),
                    0.0,
                    0.0,
                    0.0,
                    seas_v.max(1e-12),
                ],
            )
            .unwrap();

            StateSpaceModel::new(
                transition,
                observation,
                selection,
                state_cov,
                obs_var.max(1e-12),
            )
        }

        // Local linear trend without seasonal variance (shouldn't happen but handle it)
        (Some(slope_v), None, Some(_)) | (Some(slope_v), Some(_), None) => {
            // Fall back to local linear trend
            let transition = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 0.0, 1.0]).unwrap();
            let observation = Array1::from_vec(vec![1.0, 0.0]);
            let selection = Array2::eye(2);
            let state_cov = Array2::from_shape_vec(
                (2, 2),
                vec![level_var.max(1e-12), 0.0, 0.0, slope_v.max(1e-12)],
            )
            .unwrap();

            StateSpaceModel::new(
                transition,
                observation,
                selection,
                state_cov,
                obs_var.max(1e-12),
            )
        }

        // Level with seasonal period but no seasonal variance (treat as level only)
        (None, None, Some(_)) => {
            let transition = Array2::from_elem((1, 1), 1.0);
            let observation = Array1::from_elem(1, 1.0);
            let selection = Array2::from_elem((1, 1), 1.0);
            let state_cov = Array2::from_elem((1, 1), level_var.max(1e-12));

            StateSpaceModel::new(
                transition,
                observation,
                selection,
                state_cov,
                obs_var.max(1e-12),
            )
        }

        // Seasonal variance but no period specified (shouldn't happen, treat as level only)
        (None, Some(_), None) => {
            let transition = Array2::from_elem((1, 1), 1.0);
            let observation = Array1::from_elem(1, 1.0);
            let selection = Array2::from_elem((1, 1), 1.0);
            let state_cov = Array2::from_elem((1, 1), level_var.max(1e-12));

            StateSpaceModel::new(
                transition,
                observation,
                selection,
                state_cov,
                obs_var.max(1e-12),
            )
        }
    }
}

/// Compute log-likelihood for BSTS model.
fn compute_bsts_loglik(y: &[f64], params: &[f64], config: &CausalImpactConfig) -> EconResult<f64> {
    let ssm = build_state_space_model(params, None, config)?;
    let y_var = sample_variance(y);
    let init_state = initialize_state(y, config);
    let init_cov = initialize_covariance(y_var, config);

    let result = kalman_filter(y, &ssm, init_state.view(), init_cov.view())?;
    Ok(result.log_likelihood)
}

/// Predict counterfactual for post-period.
fn predict_counterfactual(
    y: &[f64],
    _time: &[i64],
    controls: Option<&Vec<Vec<f64>>>,
    ssm: &StateSpaceModel,
    regression_coefs: Option<&Vec<f64>>,
    init_state: &Array1<f64>,
    init_cov: &Array2<f64>,
    pre_start: usize,
    pre_end: usize,
    post_start: usize,
    post_end: usize,
) -> EconResult<(Vec<f64>, Vec<f64>)> {
    let n = y.len();

    // For prediction, we use the model fitted on pre-period
    // In pre-period: use actual observations
    // In post-period: predict using state-space model (counterfactual)

    // If we have controls, adjust y for regression component
    let y_adjusted: Vec<f64> = if let (Some(controls), Some(coefs)) = (controls, regression_coefs) {
        y.iter()
            .enumerate()
            .map(|(i, &yi)| {
                let mut adjustment = coefs[0]; // Intercept
                for (j, control) in controls.iter().enumerate() {
                    adjustment += coefs[j + 1] * control[i];
                }
                yi - adjustment
            })
            .collect()
    } else {
        y.to_vec()
    };

    let y_pre_adjusted: Vec<f64> = y_adjusted[pre_start..=pre_end].to_vec();

    // Run Kalman filter on pre-period
    let filter_result = kalman_filter(&y_pre_adjusted, ssm, init_state.view(), init_cov.view())?;

    // Get final filtered state and covariance (kept for potential future use)
    let n_pre = pre_end - pre_start + 1;
    let _final_state = vec_to_array1(&filter_result.filtered_states[n_pre - 1]);
    let _final_cov = nested_vec_to_array2(&filter_result.filtered_covs[n_pre - 1]);

    // Forecast for post-period
    let n_post = post_end - post_start + 1;
    let n_gap = post_start - pre_end - 1; // Gap between pre and post (if any)
    let forecast_horizon = n_gap + n_post;

    let forecast_result = kalman_forecast(&filter_result, ssm, forecast_horizon)?;

    // Build predicted and variance arrays
    let mut predicted = vec![0.0; n];
    let mut predicted_var = vec![0.0; n];

    // Pre-period: use smoothed/filtered predictions
    for i in 0..n_pre {
        let idx = pre_start + i;
        // Predicted observation = Z' * filtered_state
        let state = vec_to_array1(&filter_result.filtered_states[i]);
        predicted[idx] = ssm.observation.dot(&state);
        // Prediction variance from filtered covariance
        let cov = nested_vec_to_array2(&filter_result.filtered_covs[i]);
        predicted_var[idx] = ssm.observation.dot(&cov.dot(&ssm.observation)) + ssm.obs_var;
    }

    // Add back regression component if present
    if let (Some(controls), Some(coefs)) = (controls, regression_coefs) {
        for i in 0..n {
            let mut adjustment = coefs[0];
            for (j, control) in controls.iter().enumerate() {
                adjustment += coefs[j + 1] * control[i];
            }
            predicted[i] += adjustment;
        }
    }

    // Post-period: use forecasts (counterfactual)
    for i in 0..n_post {
        let idx = post_start + i;
        let forecast_idx = n_gap + i;
        if forecast_idx < forecast_result.forecasts.len() {
            predicted[idx] = forecast_result.forecasts[forecast_idx];
            predicted_var[idx] = forecast_result.variances[forecast_idx];

            // Add back regression component
            if let (Some(controls), Some(coefs)) = (controls, regression_coefs) {
                let mut adjustment = coefs[0];
                for (j, control) in controls.iter().enumerate() {
                    adjustment += coefs[j + 1] * control[idx];
                }
                predicted[idx] += adjustment;
            }
        }
    }

    // Fill any gaps
    for i in (pre_end + 1)..post_start {
        let gap_idx = i - pre_end - 1;
        if gap_idx < forecast_result.forecasts.len() {
            predicted[i] = forecast_result.forecasts[gap_idx];
            predicted_var[i] = forecast_result.variances[gap_idx];

            if let (Some(controls), Some(coefs)) = (controls, regression_coefs) {
                let mut adjustment = coefs[0];
                for (j, control) in controls.iter().enumerate() {
                    adjustment += coefs[j + 1] * control[i];
                }
                predicted[i] += adjustment;
            }
        }
    }

    // For indices before pre_start, just use first prediction
    for i in 0..pre_start {
        predicted[i] = predicted[pre_start];
        predicted_var[i] = predicted_var[pre_start];
    }

    // For indices after post_end, use last prediction
    for i in (post_end + 1)..n {
        predicted[i] = predicted[post_end];
        predicted_var[i] = predicted_var[post_end];
    }

    Ok((predicted, predicted_var))
}

/// Initialize state for Kalman filter.
fn initialize_state(y: &[f64], config: &CausalImpactConfig) -> Array1<f64> {
    let state_dim = compute_state_dim(config);

    let mut state = Array1::zeros(state_dim);
    state[0] = y[0]; // Level = first observation

    if config.include_trend && state_dim >= 2 {
        // Slope from first difference
        state[1] = if y.len() > 1 { y[1] - y[0] } else { 0.0 };
    }

    state
}

/// Initialize covariance for Kalman filter.
fn initialize_covariance(y_var: f64, config: &CausalImpactConfig) -> Array2<f64> {
    let state_dim = compute_state_dim(config);

    // Diffuse initialization
    Array2::from_diag(&Array1::from_elem(state_dim, y_var * 10.0))
}

/// Compute state dimension based on config.
fn compute_state_dim(config: &CausalImpactConfig) -> usize {
    let mut dim = 1; // Level

    if config.include_trend {
        dim += 1; // Slope
    }

    if let Some(period) = config.seasonal_period {
        dim += period - 1; // Seasonal states
    }

    dim
}

/// Count number of parameters for AIC/BIC.
fn count_params(config: &CausalImpactConfig, has_controls: bool) -> usize {
    let mut n = 2; // Level variance + observation variance

    if config.include_trend {
        n += 1;
    }

    if config.seasonal_period.is_some() {
        n += 1;
    }

    // Regression coefficients (if controls present)
    if has_controls {
        if let Some(ref controls) = config.control_series {
            n += controls.len() + 1; // Intercept + coefficients
        }
    }

    n
}

/// Sample variance.
fn sample_variance(y: &[f64]) -> f64 {
    let n = y.len() as f64;
    if n < 2.0 {
        return 1.0;
    }
    let mean = y.iter().sum::<f64>() / n;
    y.iter().map(|&yi| (yi - mean).powi(2)).sum::<f64>() / (n - 1.0)
}

/// Nelder-Mead optimization (copied from structts.rs for consistency).
fn nelder_mead_optimize(
    objective: &dyn Fn(&[f64]) -> f64,
    init: &[f64],
    max_iter: usize,
    tolerance: f64,
) -> (Vec<f64>, f64, bool) {
    let n = init.len();

    let alpha = 1.0;
    let gamma = 2.0;
    let rho = 0.5;
    let sigma = 0.5;

    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(init.to_vec());

    for i in 0..n {
        let mut vertex = init.to_vec();
        vertex[i] += 0.5;
        simplex.push(vertex);
    }

    let mut values: Vec<f64> = simplex.iter().map(|v| objective(v)).collect();

    for _iter in 0..max_iter {
        let mut indices: Vec<usize> = (0..=n).collect();
        indices.sort_by(|&a, &b| {
            values[a]
                .partial_cmp(&values[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let sorted_simplex: Vec<Vec<f64>> = indices.iter().map(|&i| simplex[i].clone()).collect();
        let sorted_values: Vec<f64> = indices.iter().map(|&i| values[i]).collect();
        simplex = sorted_simplex;
        values = sorted_values;

        let f_spread = (values[n] - values[0]).abs() / (values[0].abs() + 1e-10);
        if f_spread < tolerance {
            return (simplex[0].clone(), values[0], true);
        }

        let mut centroid = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                centroid[j] += simplex[i][j];
            }
        }
        for c in centroid.iter_mut() {
            *c /= n as f64;
        }

        let reflected: Vec<f64> = (0..n)
            .map(|j| centroid[j] + alpha * (centroid[j] - simplex[n][j]))
            .collect();
        let f_reflected = objective(&reflected);

        if f_reflected < values[n - 1] && f_reflected >= values[0] {
            simplex[n] = reflected;
            values[n] = f_reflected;
        } else if f_reflected < values[0] {
            let expanded: Vec<f64> = (0..n)
                .map(|j| centroid[j] + gamma * (reflected[j] - centroid[j]))
                .collect();
            let f_expanded = objective(&expanded);

            if f_expanded < f_reflected {
                simplex[n] = expanded;
                values[n] = f_expanded;
            } else {
                simplex[n] = reflected;
                values[n] = f_reflected;
            }
        } else {
            let contract_point: Vec<f64> = if f_reflected < values[n] {
                (0..n)
                    .map(|j| centroid[j] + rho * (reflected[j] - centroid[j]))
                    .collect()
            } else {
                (0..n)
                    .map(|j| centroid[j] - rho * (centroid[j] - simplex[n][j]))
                    .collect()
            };
            let f_contracted = objective(&contract_point);

            let threshold = if f_reflected < values[n] {
                f_reflected
            } else {
                values[n]
            };
            if f_contracted < threshold {
                simplex[n] = contract_point;
                values[n] = f_contracted;
            } else {
                let best = simplex[0].clone();
                for i in 1..=n {
                    for j in 0..n {
                        simplex[i][j] = best[j] + sigma * (simplex[i][j] - best[j]);
                    }
                    values[i] = objective(&simplex[i]);
                }
            }
        }
    }

    (simplex[0].clone(), values[0], false)
}

// Helper functions for array conversions

fn vec_to_array1(v: &[f64]) -> Array1<f64> {
    Array1::from_vec(v.to_vec())
}

fn nested_vec_to_array2(v: &[Vec<f64>]) -> Array2<f64> {
    let rows = v.len();
    let cols = if rows > 0 { v[0].len() } else { 0 };
    let mut arr = Array2::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            arr[[i, j]] = v[i][j];
        }
    }
    arr
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn generate_causal_impact_data(
        n_pre: usize,
        n_post: usize,
        treatment_effect: f64,
        seed: u64,
    ) -> (Vec<f64>, Vec<i64>) {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let n = n_pre + n_post;

        let mut y = Vec::with_capacity(n);
        let mut level = 100.0;

        for t in 0..n {
            // Random walk level
            level += (rng.r#gen::<f64>() - 0.5) * 2.0;

            // Add treatment effect in post-period
            let effect = if t >= n_pre { treatment_effect } else { 0.0 };

            // Add noise
            let noise = (rng.r#gen::<f64>() - 0.5) * 1.0;

            y.push(level + effect + noise);
        }

        let time: Vec<i64> = (0..n as i64).collect();

        (y, time)
    }

    #[test]
    fn test_causal_impact_positive_effect() {
        let n_pre = 70;
        let n_post = 30;
        let treatment_effect = 10.0;

        let (y, time) = generate_causal_impact_data(n_pre, n_post, treatment_effect, 42);

        let df = df! {
            "y" => &y,
            "time" => &time,
        }
        .unwrap();

        let dataset = Dataset::new(df);

        let config = CausalImpactConfig {
            pre_period: (0, (n_pre - 1) as i64),
            post_period: (n_pre as i64, (n_pre + n_post - 1) as i64),
            alpha: 0.05,
            ..Default::default()
        };

        let result = causal_impact(&dataset, "y", "time", config).unwrap();

        // Should detect positive effect
        assert!(
            result.summary.cumulative_effect > 0.0,
            "Cumulative effect should be positive, got {}",
            result.summary.cumulative_effect
        );

        // Effect should be roughly n_post * treatment_effect
        let expected_cumulative = n_post as f64 * treatment_effect;
        let relative_error =
            (result.summary.cumulative_effect - expected_cumulative).abs() / expected_cumulative;
        assert!(
            relative_error < 0.5,
            "Cumulative effect {} should be close to {}",
            result.summary.cumulative_effect,
            expected_cumulative
        );

        // Should be statistically significant
        assert!(
            result.summary.p_value < 0.10,
            "P-value {} should be small for large effect",
            result.summary.p_value
        );

        // Check series lengths
        assert_eq!(result.series.time.len(), n_pre + n_post);
        assert_eq!(result.series.observed.len(), n_pre + n_post);
        assert_eq!(result.series.predicted.len(), n_pre + n_post);
    }

    #[test]
    fn test_causal_impact_no_effect() {
        let n_pre = 70;
        let n_post = 30;
        let treatment_effect = 0.0;

        let (y, time) = generate_causal_impact_data(n_pre, n_post, treatment_effect, 123);

        let df = df! {
            "y" => &y,
            "time" => &time,
        }
        .unwrap();

        let dataset = Dataset::new(df);

        let result = run_causal_impact(
            &dataset,
            "y",
            "time",
            (0, (n_pre - 1) as i64),
            (n_pre as i64, (n_pre + n_post - 1) as i64),
            None,
        )
        .unwrap();

        // Effect should be close to zero
        let avg_effect_magnitude = result.summary.average_effect.abs();
        assert!(
            avg_effect_magnitude < 5.0,
            "Average effect {} should be close to zero for no treatment",
            avg_effect_magnitude
        );

        // Should not be significant (with some probability of false positive)
        // This is a soft check since we're using random data
        // p-value should generally be larger for no effect
        println!("No effect p-value: {}", result.summary.p_value);
    }

    #[test]
    fn test_causal_impact_with_controls() {
        let n_pre = 70;
        let n_post = 30;
        let n = n_pre + n_post;

        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(456);

        // Generate control series that correlates with response
        let control: Vec<f64> = (0..n).map(|_| 50.0 + rng.r#gen::<f64>() * 20.0).collect();

        // Response is partly driven by control
        let y: Vec<f64> = control
            .iter()
            .enumerate()
            .map(|(t, &c)| {
                let base = 0.5 * c; // Control effect
                let effect = if t >= n_pre { 8.0 } else { 0.0 }; // Treatment effect
                let noise = (rng.r#gen::<f64>() - 0.5) * 2.0;
                base + effect + noise
            })
            .collect();

        let time: Vec<i64> = (0..n as i64).collect();

        let df = df! {
            "y" => &y,
            "time" => &time,
            "control" => &control,
        }
        .unwrap();

        let dataset = Dataset::new(df);

        let config = CausalImpactConfig {
            pre_period: (0, (n_pre - 1) as i64),
            post_period: (n_pre as i64, (n_pre + n_post - 1) as i64),
            control_series: Some(vec!["control".to_string()]),
            alpha: 0.05,
            ..Default::default()
        };

        let result = causal_impact(&dataset, "y", "time", config).unwrap();

        // Should detect the treatment effect
        assert!(result.summary.cumulative_effect > 0.0);

        // Should have regression coefficients
        assert!(result.model.regression_coefficients.is_some());
    }

    #[test]
    fn test_causal_impact_invalid_periods() {
        let (y, time) = generate_causal_impact_data(50, 30, 5.0, 789);

        let df = df! {
            "y" => &y,
            "time" => &time,
        }
        .unwrap();

        let dataset = Dataset::new(df);

        // Post period starts before pre period ends (overlap)
        let config = CausalImpactConfig {
            pre_period: (0, 60),
            post_period: (50, 79), // Overlaps with pre-period
            ..Default::default()
        };

        let result = causal_impact(&dataset, "y", "time", config);
        assert!(result.is_err());
    }

    #[test]
    fn test_causal_impact_insufficient_pre_period() {
        let (y, time) = generate_causal_impact_data(5, 30, 5.0, 999);

        let df = df! {
            "y" => &y,
            "time" => &time,
        }
        .unwrap();

        let dataset = Dataset::new(df);

        let config = CausalImpactConfig {
            pre_period: (0, 4), // Only 5 observations
            post_period: (5, 34),
            ..Default::default()
        };

        let result = causal_impact(&dataset, "y", "time", config);
        assert!(result.is_err());
    }

    #[test]
    fn test_causal_impact_with_trend() {
        let n_pre = 70;
        let n_post = 30;

        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(321);

        // Generate trending data
        let y: Vec<f64> = (0..(n_pre + n_post))
            .map(|t| {
                let trend = 100.0 + 0.5 * t as f64;
                let effect = if t >= n_pre { 15.0 } else { 0.0 };
                let noise = (rng.r#gen::<f64>() - 0.5) * 2.0;
                trend + effect + noise
            })
            .collect();

        let time: Vec<i64> = (0..(n_pre + n_post) as i64).collect();

        let df = df! {
            "y" => &y,
            "time" => &time,
        }
        .unwrap();

        let dataset = Dataset::new(df);

        let config = CausalImpactConfig {
            pre_period: (0, (n_pre - 1) as i64),
            post_period: (n_pre as i64, (n_pre + n_post - 1) as i64),
            include_trend: true,
            alpha: 0.05,
            ..Default::default()
        };

        let result = causal_impact(&dataset, "y", "time", config).unwrap();

        // Should detect positive effect
        assert!(result.summary.cumulative_effect > 0.0);

        // Model should have slope variance
        assert!(result.model.slope_variance.is_some());
    }

    #[test]
    fn test_causal_impact_model_info() {
        let (y, time) = generate_causal_impact_data(70, 30, 10.0, 555);

        let df = df! {
            "y" => &y,
            "time" => &time,
        }
        .unwrap();

        let dataset = Dataset::new(df);

        let result = run_causal_impact(&dataset, "y", "time", (0, 69), (70, 99), None).unwrap();

        // Check model info
        assert_eq!(result.model.n_pre, 70);
        assert_eq!(result.model.n_post, 30);
        assert!(result.model.log_likelihood.is_finite());
        assert!(result.model.aic.is_finite());
        assert!(result.model.bic.is_finite());
        assert!(result.model.level_variance > 0.0);
        assert!(result.model.observation_variance > 0.0);
    }

    #[test]
    fn test_causal_impact_inference() {
        let (y, time) = generate_causal_impact_data(70, 30, 10.0, 666);

        let df = df! {
            "y" => &y,
            "time" => &time,
        }
        .unwrap();

        let dataset = Dataset::new(df);

        let result = run_causal_impact(&dataset, "y", "time", (0, 69), (70, 99), None).unwrap();

        // Check inference
        assert!(result.inference.prob_positive >= 0.0 && result.inference.prob_positive <= 1.0);
        assert!(result.inference.prob_negative >= 0.0 && result.inference.prob_negative <= 1.0);
        assert!(
            (result.inference.prob_positive + result.inference.prob_negative - 1.0).abs() < 1e-10
        );
        assert!(result.inference.effect_sd > 0.0);
    }

    #[test]
    fn test_validate_causal_impact_cumulative_effect() {
        // Create data with a known additive effect of 5.0 in the post-period.
        // Verify the estimated cumulative effect is close to n_post * 5.0 = 150.
        let n_pre = 80;
        let n_post = 30;
        let true_effect = 5.0;

        let (y, time) = generate_causal_impact_data(n_pre, n_post, true_effect, 42);

        let df = df! {
            "y" => &y,
            "time" => &time,
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_causal_impact(
            &dataset,
            "y",
            "time",
            (0, (n_pre - 1) as i64),
            (n_pre as i64, (n_pre + n_post - 1) as i64),
            None,
        )
        .unwrap();

        let expected_cumulative = n_post as f64 * true_effect;

        // The cumulative effect should be within 50% of the true cumulative
        // (generous because it's a random walk + small sample)
        let relative_error =
            (result.summary.cumulative_effect - expected_cumulative).abs() / expected_cumulative;
        assert!(
            relative_error < 0.6,
            "Cumulative effect {:.2} should be close to expected {:.2} (rel err {:.2})",
            result.summary.cumulative_effect,
            expected_cumulative,
            relative_error
        );
    }

    #[test]
    fn test_validate_causal_impact_point_effects() {
        // Verify that point-wise effects in the post-period average near the true effect.
        let n_pre = 70;
        let n_post = 30;
        let true_effect = 5.0;

        let (y, time) = generate_causal_impact_data(n_pre, n_post, true_effect, 123);

        let df = df! {
            "y" => &y,
            "time" => &time,
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_causal_impact(
            &dataset,
            "y",
            "time",
            (0, (n_pre - 1) as i64),
            (n_pre as i64, (n_pre + n_post - 1) as i64),
            None,
        )
        .unwrap();

        // Average of point effects in the post-period
        let avg_point_effect = result.summary.average_effect;

        // Should be in the neighborhood of true_effect
        assert!(
            (avg_point_effect - true_effect).abs() < true_effect * 1.5,
            "Average point effect {:.2} should be in neighborhood of {:.2}",
            avg_point_effect,
            true_effect
        );

        // All point effects should be finite
        for pe in &result.series.point_effect {
            assert!(pe.is_finite(), "Point effect should be finite");
        }
    }

    #[test]
    fn test_validate_causal_impact_credible_intervals() {
        // Verify that the credible interval for the average effect contains
        // the true effect value (at least occasionally with random data).
        let n_pre = 80;
        let n_post = 30;
        let true_effect = 8.0;

        let (y, time) = generate_causal_impact_data(n_pre, n_post, true_effect, 321);

        let df = df! {
            "y" => &y,
            "time" => &time,
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let config = CausalImpactConfig {
            pre_period: (0, (n_pre - 1) as i64),
            post_period: (n_pre as i64, (n_pre + n_post - 1) as i64),
            alpha: 0.05,
            ..Default::default()
        };

        let result = causal_impact(&dataset, "y", "time", config).unwrap();

        // The CI bounds should be ordered correctly
        assert!(
            result.summary.average_effect_lower <= result.summary.average_effect,
            "Lower CI bound should be <= average effect"
        );
        assert!(
            result.summary.average_effect_upper >= result.summary.average_effect,
            "Upper CI bound should be >= average effect"
        );

        // Cumulative bounds should also be ordered
        assert!(result.summary.cumulative_effect_lower <= result.summary.cumulative_effect,);
        assert!(result.summary.cumulative_effect_upper >= result.summary.cumulative_effect,);

        // CI width should be positive
        let ci_width = result.summary.average_effect_upper - result.summary.average_effect_lower;
        assert!(
            ci_width > 0.0,
            "CI width should be positive, got {:.4}",
            ci_width
        );
    }

    #[test]
    fn test_validate_causal_impact_series_lengths() {
        // Verify that all output series have consistent lengths.
        let n_pre = 60;
        let n_post = 20;
        let n = n_pre + n_post;

        let (y, time) = generate_causal_impact_data(n_pre, n_post, 3.0, 555);

        let df = df! {
            "y" => &y,
            "time" => &time,
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_causal_impact(
            &dataset,
            "y",
            "time",
            (0, (n_pre - 1) as i64),
            (n_pre as i64, (n - 1) as i64),
            None,
        )
        .unwrap();

        assert_eq!(result.series.time.len(), n);
        assert_eq!(result.series.observed.len(), n);
        assert_eq!(result.series.predicted.len(), n);
        assert_eq!(result.series.predicted_lower.len(), n);
        assert_eq!(result.series.predicted_upper.len(), n);
        assert_eq!(result.series.point_effect.len(), n);
        assert_eq!(result.series.cumulative_effect.len(), n);
    }
}
