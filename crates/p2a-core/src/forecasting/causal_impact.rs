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
//! Model fitting is the StructTS engine (diffuse prior, R's objective, projected
//! Newton); the intervals are the plug-in Gaussian forecast of that model, not a
//! posterior. The `CausalImpact` package (bsts) is the method's source, not its
//! numerical reference: `validation/forecasting/causal_impact.md`.
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

use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::forecasting::kalman::{
    admit_filter_history_and_forecast, kalman_filter, kalman_forecast,
};
use crate::forecasting::structts::{SpecNaming, StructuralFit, StructuralSpec, fit_structural};
use crate::linalg::solve::aliasing_message;
use crate::regression::ols::solve_ols;
use crate::resource::{try_zeros2, with_remedy};
use crate::traits::convergence::{Convergence, ResultNotes};
use crate::traits::estimator::{validate_level, z_critical};

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
    /// Seasonal period (if the data has seasonality): at least 2, and at most
    /// the number of pre-period observations, so that the pre-period holds a
    /// full season.
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
    /// The model fit's convergence record.
    #[serde(default)]
    pub convergence: Convergence,
    /// Warnings: clipped periods, an aliased control, the fit's convergence.
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl ResultNotes for CausalImpactResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
    fn notes_convergence(&self) -> Option<Convergence> {
        Some(self.convergence.clone())
    }
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
/// The fit's likelihood evaluations keep no Kalman history. The
/// counterfactual runs [`kalman_filter`] over the pre-period, which keeps
/// about 16·n_pre·m² bytes, and at least 320 bytes per observation (m = 1 +
/// an optional slope + `seasonal_period` − 1 seasonal states), and holds it
/// while [`kalman_forecast`] runs over the gap and the post-period, 8m + 64
/// bytes a step (80 at m = 1). The two are admitted as one figure before the
/// fit: [`EconError::ResourceLimit`] when it exceeds the per-allocation
/// ceiling, naming `seasonal_period` when there is one. The seasonal model's
/// transition matrix and initial state covariance (m × m) are allocated
/// fallibly: a [`EconError::ResourceLimit`] naming `seasonal_period` when
/// they cannot be.
/// [`EconError::InvalidSpecification`] for a `seasonal_period` below 2 or
/// longer than the pre-period, checked before anything is sized by it; for an
/// `alpha` outside (0, 1), a `max_iter` below 1 or a `tolerance` that is not
/// finite and positive; for a period whose start is after its end or that holds
/// no time value; for a pre-period that does not end before the post-period; and
/// for a time column that is not strictly increasing.
/// A period reaching past the data is clipped to it, with a warning.
/// [`EconError::Cancelled`] once the budget installed with
/// [`crate::resource::with_budget`] is cancelled or past its deadline.
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
///
/// Integer and Boolean columns are accepted; a missing value (empty cell, NA or NaN) is an
/// error naming the column. The time column must hold whole numbers or dates.
pub fn causal_impact(
    dataset: &Dataset,
    response_col: &str,
    time_col: &str,
    config: CausalImpactConfig,
) -> EconResult<CausalImpactResult> {
    // The configuration before any column is read.
    validate_level(config.alpha, "alpha")?;
    if config.max_iter < 1 {
        return Err(EconError::InvalidSpecification {
            message: format!("max_iter must be at least 1, got {}", config.max_iter),
        });
    }
    if !config.tolerance.is_finite() || config.tolerance <= 0.0 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "tolerance must be finite and positive, got {}",
                config.tolerance
            ),
        });
    }
    for (name, (start, end)) in [
        ("pre_period", config.pre_period),
        ("post_period", config.post_period),
    ] {
        if start > end {
            return Err(EconError::InvalidSpecification {
                message: format!("{name} start {start} is after its end {end}"),
            });
        }
    }

    // One request: a whole-number time index (a fractional time is an error, not
    // truncated), the response and the controls, all without missing values.
    let control_names: Vec<&str> = config
        .control_series
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(String::as_str)
        .collect();
    let cols = ColumnRequest::new(dataset.df())
        .time_index(time_col)
        .numeric(response_col)
        .numerics(&control_names)
        .na(NaPolicy::Fail)
        .extract()?;
    let time: Vec<i64> = cols.time_index(time_col).to_vec();
    let y: Vec<f64> = cols.numeric(response_col).to_vec();
    // An empty control list is no control regression.
    let controls = config
        .control_series
        .as_ref()
        .filter(|names| !names.is_empty())
        .map(|names| {
            names
                .iter()
                .map(|name| cols.numeric(name).to_vec())
                .collect::<Vec<Vec<f64>>>()
        });

    if let Some(i) = (1..time.len()).find(|&i| time[i] <= time[i - 1]) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "time column '{time_col}' must be strictly increasing; row {} has {} after {}",
                i + 1,
                time[i],
                time[i - 1]
            ),
        });
    }

    let (periods, warnings) = PeriodIndices::locate(&time, config.pre_period, config.post_period)?;

    if periods.n_pre < 10 {
        return Err(EconError::InsufficientData {
            required: 10,
            provided: periods.n_pre,
            context: "CausalImpact requires at least 10 pre-period observations".to_string(),
        });
    }
    if let Some(period) = config.seasonal_period {
        check_seasonal_period(period, periods.n_pre)?;
    }

    // Run the core algorithm
    run_causal_impact_core(&y, &time, controls.as_ref(), &periods, &config, warnings)
}

/// Where the two periods sit in the series. Every length is checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PeriodIndices {
    pre_start: usize,
    pre_end: usize,
    post_start: usize,
    post_end: usize,
    n_pre: usize,
    n_post: usize,
    /// Observations between the two periods.
    n_gap: usize,
}

impl PeriodIndices {
    /// The first index at or after a period's start and the last at or before its
    /// end, for strictly increasing `time`, and the warnings for what was clipped.
    fn locate(time: &[i64], pre: (i64, i64), post: (i64, i64)) -> EconResult<(Self, Vec<String>)> {
        let (Some(&first), Some(&last)) = (time.first(), time.last()) else {
            return Err(EconError::InsufficientData {
                required: 10,
                provided: 0,
                context: "CausalImpact requires at least 10 pre-period observations".to_string(),
            });
        };
        let find = |name: &str, (start, end): (i64, i64)| -> EconResult<(usize, usize)> {
            let from = time.iter().position(|&t| t >= start);
            let to = time.iter().rposition(|&t| t <= end);
            match (from, to) {
                (Some(from), Some(to)) if from <= to => Ok((from, to)),
                _ => Err(EconError::InvalidSpecification {
                    message: format!(
                        "{name} [{start}, {end}] contains no time values (the time column runs \
                         from {first} to {last})"
                    ),
                }),
            }
        };
        let (pre_start, pre_end) = find("pre_period", pre)?;
        let (post_start, post_end) = find("post_period", post)?;
        if pre_end >= post_start {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "Pre-period must end before post-period begins (pre_period ends at time {}, \
                     post_period starts at time {})",
                    time[pre_end], time[post_start]
                ),
            });
        }
        let length = |from: usize, to: usize, what: &str| -> EconResult<usize> {
            to.checked_sub(from)
                .and_then(|d| d.checked_add(1))
                .ok_or_else(|| EconError::InvalidSpecification {
                    message: format!("{what} has no observations"),
                })
        };
        let periods = Self {
            pre_start,
            pre_end,
            post_start,
            post_end,
            n_pre: length(pre_start, pre_end, "pre_period")?,
            n_post: length(post_start, post_end, "post_period")?,
            // pre_end < post_start was checked above.
            n_gap: post_start - pre_end - 1,
        };
        let mut warnings = Vec::new();
        if pre.0 < first {
            warnings.push(format!(
                "pre_period start {} is before the first time value {first}; the pre-period \
                 starts at {first}",
                pre.0
            ));
        }
        if post.1 > last {
            warnings.push(format!(
                "post_period end {} is after the last time value {last}; the post-period ends \
                 at {last}",
                post.1
            ));
        }
        Ok((periods, warnings))
    }

    /// Steps from the end of the pre-period to the end of the post-period.
    fn horizon(&self) -> EconResult<usize> {
        self.n_gap
            .checked_add(self.n_post)
            .ok_or_else(|| crate::resource::overflow("the forecast horizon"))
    }
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
///
/// Integer and Boolean columns are accepted; a missing value (empty cell, NA or NaN) is an
/// error naming the column.
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
    periods: &PeriodIndices,
    config: &CausalImpactConfig,
    mut warnings: Vec<String>,
) -> EconResult<CausalImpactResult> {
    let PeriodIndices {
        pre_start,
        pre_end,
        post_start,
        post_end,
        n_pre,
        n_post,
        ..
    } = *periods;

    // The counterfactual's filter pass keeps its history and holds it while
    // the forecast over the gap and the post-period (predict_counterfactual's
    // horizon) is built. Admit both as one figure before the fit, whose
    // likelihood passes keep neither, so that a refusal comes before any
    // fitting work.
    let spec = causal_spec(config);
    let m = spec.state_dim()?;
    let horizon = periods.horizon()?;
    admit_filter_history_and_forecast(n_pre, m, horizon)
        .map_err(|e| with_remedy(e, &history_remedy(&spec, m)))?;

    // Step 1: The control regression on the pre-period; what it leaves is what
    // the model is fitted to.
    let y_pre = &y[pre_start..=pre_end];
    let regression = match controls {
        Some(controls) => Some(regress_out_controls(
            y,
            controls,
            periods,
            config.control_series.as_deref().unwrap_or_default(),
            &mut warnings,
        )?),
        None => None,
    };
    let y_for_model = regression
        .as_ref()
        .map_or(y_pre, |r| r.residuals.as_slice());

    // Step 2: The structural model on the pre-period.
    let fit = fit_structural(y_for_model, spec, None, config.max_iter, config.tolerance)?;
    warnings.extend(fit.warnings.iter().cloned());

    // Step 3: The filter over the pre-period, the forecast beyond it.
    let (predicted, predicted_var) =
        predict_counterfactual(y_for_model, y.len(), regression.as_ref(), &fit, periods)?;

    // Step 4: Compute causal effects
    let z_alpha = z_critical(1.0 - config.alpha)?;

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

    // Step 5: Compute summary statistics
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

    // Tail probabilities of the plug-in normal: the upper tail through sf, since
    // 1 - cdf is exactly 0 from about 8.3 standard deviations.
    let normal = Normal::standard();
    let z_cumulative = if cumulative_se > 0.0 {
        cumulative_effect / cumulative_se
    } else {
        0.0
    };
    let prob_positive = normal.sf(-z_cumulative);
    let prob_negative = normal.cdf(-z_cumulative);
    let p_value = if cumulative_effect >= 0.0 {
        prob_negative
    } else {
        prob_positive
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
    let coef = spec.coefficients(&fit.variances);
    let log_lik = fit.log_likelihood;
    let model = CausalImpactModel {
        level_variance: coef.level,
        slope_variance: coef.slope,
        seasonal_variance: coef.seasonal,
        observation_variance: coef.epsilon,
        regression_coefficients: regression.map(|r| r.coefficients),
        control_names: config.control_series.clone(),
        log_likelihood: log_lik,
        aic: -2.0 * log_lik + 2.0 * n_params as f64,
        bic: -2.0 * log_lik + (n_params as f64) * (n_pre as f64).ln(),
        n_pre,
        n_post,
    };

    // Causal inference
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
        convergence: fit.convergence,
        warnings,
    })
}

/// What the control regression gives the fit and the counterfactual.
struct ControlRegression {
    /// Intercept, then one per control; NaN for an aliased control.
    coefficients: Vec<f64>,
    /// The pre-period residuals, which the model is fitted to.
    residuals: Vec<f64>,
    /// The regression's value at every row of the series, from the identified
    /// coefficients only, as R's `predict.lm` on a rank-deficient fit.
    adjustment: Vec<f64>,
}

/// Regress the pre-period of `y` on an intercept and the controls by OLS, aliased
/// controls reported in `warnings`.
fn regress_out_controls(
    y: &[f64],
    controls: &[Vec<f64>],
    periods: &PeriodIndices,
    control_names: &[String],
    warnings: &mut Vec<String>,
) -> EconResult<ControlRegression> {
    let (from, to) = (periods.pre_start, periods.pre_end);
    let k = controls.len() + 1; // a Vec's length is below usize::MAX
    let mut x = try_zeros2(periods.n_pre, k, "the control regression's design")?;
    x.column_mut(0).fill(1.0);
    for (j, control) in controls.iter().enumerate() {
        x.column_mut(j + 1)
            .assign(&ndarray::ArrayView1::from(&control[from..=to]));
    }
    let y_pre = ndarray::ArrayView1::from(&y[from..=to]);
    let solved = solve_ols(x.view(), y_pre)?;

    let mut names = Vec::with_capacity(k);
    names.push("(Intercept)".to_string());
    names.extend(control_names.iter().cloned());
    if let Some(message) = aliasing_message(&solved.aliased, &names) {
        warnings.push(message);
        warnings.push(
            "the control regression is rank deficient: the counterfactual uses the identified \
             coefficients only"
                .to_string(),
        );
    }

    let coefficients = solved.beta.to_vec();
    let identified = |j: usize| {
        let b = coefficients[j];
        if b.is_nan() { 0.0 } else { b }
    };
    let adjustment = (0..y.len())
        .map(|i| {
            identified(0)
                + controls
                    .iter()
                    .enumerate()
                    .map(|(j, control)| identified(j + 1) * control[i])
                    .sum::<f64>()
        })
        .collect();
    Ok(ControlRegression {
        residuals: solved.residuals.to_vec(),
        coefficients,
        adjustment,
    })
}

/// The counterfactual and its variance at all `n` rows: in the pre-period the
/// filtered fit of `y_for_model`, in the gap and the post-period the forecast of the
/// fitted model, outside the two periods the nearest value.
fn predict_counterfactual(
    y_for_model: &[f64],
    n: usize,
    regression: Option<&ControlRegression>,
    fit: &StructuralFit,
    periods: &PeriodIndices,
) -> EconResult<(Vec<f64>, Vec<f64>)> {
    let PeriodIndices {
        pre_start,
        pre_end,
        post_end,
        ..
    } = *periods;
    let ssm = &fit.model;
    let filter_result =
        kalman_filter(y_for_model, ssm, fit.init_state.view(), fit.init_cov.view())?;
    let forecast_result = kalman_forecast(&filter_result, ssm, periods.horizon()?)?;
    let adjustment = |i: usize| regression.map_or(0.0, |r| r.adjustment[i]);

    let mut predicted = vec![0.0; n];
    let mut predicted_var = vec![0.0; n];

    // Pre-period: Z' a_t|t and Z' P_t|t Z + H.
    let z = ssm.observation.as_slice_memory_order();
    let z = z.ok_or_else(|| EconError::Computation("the observation vector".to_string()))?;
    for (i, (state, cov)) in filter_result
        .filtered_states
        .iter()
        .zip(&filter_result.filtered_covs)
        .enumerate()
    {
        let idx = pre_start + i;
        let level: f64 = z.iter().zip(state).map(|(a, b)| a * b).sum();
        let quadratic: f64 = z
            .iter()
            .zip(cov)
            .filter(|(zj, _)| **zj != 0.0)
            .map(|(zj, row)| zj * z.iter().zip(row).map(|(a, b)| a * b).sum::<f64>())
            .sum();
        predicted[idx] = level + adjustment(idx);
        predicted_var[idx] = quadratic + ssm.obs_var;
    }

    // The gap and the post-period: the forecast (the counterfactual).
    for (h, (forecast, variance)) in forecast_result
        .forecasts
        .iter()
        .zip(&forecast_result.variances)
        .enumerate()
    {
        let idx = pre_end + 1 + h;
        predicted[idx] = forecast + adjustment(idx);
        predicted_var[idx] = *variance;
    }

    // Outside the two periods, the nearest value.
    for i in 0..pre_start {
        predicted[i] = predicted[pre_start];
        predicted_var[i] = predicted_var[pre_start];
    }
    for i in (post_end + 1)..n {
        predicted[i] = predicted[post_end];
        predicted_var[i] = predicted_var[post_end];
    }

    Ok((predicted, predicted_var))
}

/// CausalImpact's level, optional trend and optional seasonal as an engine model.
fn causal_spec(config: &CausalImpactConfig) -> StructuralSpec {
    StructuralSpec {
        slope: config.include_trend,
        seasonal: config.seasonal_period,
        naming: CAUSAL_IMPACT_NAMING,
    }
}

const CAUSAL_IMPACT_NAMING: SpecNaming = SpecNaming {
    method: "CausalImpact",
    transition: "the seasonal model's transition matrix",
    prior: "the seasonal model's initial state covariance",
    remedy: seasonal_square_remedy,
    check_period: check_seasonal_period,
};

/// What to change when the Kalman history and forecast of an `m`-dimensional
/// state are refused: the history grows with the pre-period, the forecast
/// with the post-period, and with a seasonal component the state dimension
/// follows from `seasonal_period`.
fn history_remedy(spec: &StructuralSpec, m: usize) -> String {
    match seasonal_dim_note(spec, m) {
        Some(note) => {
            format!("{note}; use a smaller seasonal_period or a shorter pre_period or post_period")
        }
        None => "use a shorter pre_period or post_period".to_string(),
    }
}

/// How `spec`'s `seasonal` sets the state dimension `m`, for a remedy, or `None`
/// without a seasonal component. The period is read from `spec`, so the note
/// cannot describe a different one from the model that was sized.
fn seasonal_dim_note(spec: &StructuralSpec, m: usize) -> Option<String> {
    spec.seasonal.map(|period| {
        if spec.slope {
            format!(
                "with seasonal_period {period} and include_trend the state dimension is \
                 seasonal_period + 1 = {m}"
            )
        } else {
            format!("with seasonal_period {period} the state dimension is seasonal_period = {m}")
        }
    })
}

/// `Ok` when a seasonal component with `period` seasons can be fitted to a
/// pre-period of `n_pre` observations: a cycle of at least two seasons, and
/// at least one full cycle in the pre-period. This bounds the state
/// dimension by the data before any m × m matrix is sized by it.
fn check_seasonal_period(period: usize, n_pre: usize) -> EconResult<()> {
    if period < 2 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "seasonal_period must be at least 2, got {period}: a seasonal cycle has at \
                 least two seasons; omit seasonal_period for a series without seasonality"
            ),
        });
    }
    if period > n_pre {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "seasonal_period {period} is longer than the pre-period ({n_pre} observations): \
                 the pre-period must hold at least one full season, so seasonal_period must \
                 be at most {n_pre}; use a smaller seasonal_period or a longer pre_period"
            ),
        });
    }
    Ok(())
}

/// The remedy of a refused m × m matrix: it names `seasonal_period` and `m`.
fn seasonal_square_remedy(spec: &StructuralSpec, m: usize) -> String {
    seasonal_dim_note(spec, m).map_or_else(
        || "use a smaller seasonal_period".to_string(),
        |note| format!("{note}; use a smaller seasonal_period"),
    )
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

#[cfg(test)]
mod tests {
    use super::*;
    use p2a_validation::{Check, Tol};
    use polars::prelude::*;

    /// X-2: an Int64 response fits as its Float64 copy; missing values in the response,
    /// the time index or a control, text and absent columns are refused by name, and a
    /// fractional time index is an error (a non-strict cast used to truncate it).
    #[test]
    fn test_na_causal_impact_int64_and_missing() {
        use crate::forecasting::test_fixtures::*;
        let ds = with_float_copy(&arima_series(), "ai");
        let run = |col: &str| run_causal_impact(&ds, col, "t", (1, 200), (201, 300), None);
        let int = run("ai").unwrap();
        let flt = run("ai_f64").unwrap();
        assert_same(
            &[int.summary.average_effect, int.summary.cumulative_effect],
            &[flt.summary.average_effect, flt.summary.cumulative_effect],
        );
        assert_null_values(&run("a_na").unwrap_err(), "a_na", 2);
        assert_null_values(&run("ai_na").unwrap_err(), "ai_na", 2);
        assert_non_numeric(&run("lab").unwrap_err(), "lab");
        assert_not_found(&run("nope").unwrap_err(), "nope");
        let time = |t: &str| run_causal_impact(&ds, "a", t, (1, 200), (201, 300), None);
        assert_null_values(&time("ai_na").unwrap_err(), "ai_na", 2);
        let fractional = time("a").unwrap_err().to_string();
        assert!(fractional.contains("whole numbers"), "{fractional}");
        let control = run_causal_impact(&ds, "a", "t", (1, 200), (201, 300), Some(&["a_na"]));
        assert_null_values(&control.unwrap_err(), "a_na", 2);
    }

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

    fn ds(y: &[f64], time: &[i64]) -> Dataset {
        Dataset::new(df! {"y" => y, "time" => time}.unwrap())
    }

    fn refused(dataset: &Dataset, config: CausalImpactConfig, needles: &[&str]) {
        match causal_impact(dataset, "y", "time", config) {
            Err(EconError::InvalidSpecification { message }) => {
                for needle in needles {
                    assert!(message.contains(needle), "{needle:?} not in {message}");
                }
            }
            other => panic!(
                "expected InvalidSpecification for {needles:?}, got {:?}",
                other.map(|_| ())
            ),
        }
    }

    fn periods(pre_period: (i64, i64), post_period: (i64, i64)) -> CausalImpactConfig {
        CausalImpactConfig {
            pre_period,
            post_period,
            ..Default::default()
        }
    }

    #[test]
    fn test_causal_impact_rejects_inverted_periods() {
        let (y, time) = generate_causal_impact_data(50, 30, 5.0, 789);
        let d = ds(&y, &time);
        refused(
            &d,
            periods((0, 39), (50, 45)),
            &["post_period start 50 is after its end 45"],
        );
        refused(
            &d,
            periods((39, 0), (40, 79)),
            &["pre_period start 39 is after its end 0"],
        );
        refused(
            &d,
            periods((0, 39), (200, 210)),
            &["post_period [200, 210] contains no time values", "0 to 79"],
        );
        // A period between two time values holds no observation.
        let even: Vec<i64> = time.iter().map(|t| 2 * t).collect();
        refused(
            &ds(&y, &even),
            periods((0, 78), (81, 81)),
            &["post_period [81, 81] contains no time values"],
        );
        // Rows 4 and 5 (1-based) swapped.
        let mut swapped = time.clone();
        swapped.swap(3, 4);
        refused(
            &ds(&y, &swapped),
            periods((0, 39), (40, 79)),
            &["must be strictly increasing", "row 5"],
        );
    }

    #[test]
    fn test_causal_impact_rejects_degenerate_config() {
        let (y, time) = generate_causal_impact_data(50, 30, 5.0, 789);
        let d = ds(&y, &time);
        for alpha in [0.0, 1.0, 1.5, f64::NAN] {
            refused(
                &d,
                CausalImpactConfig {
                    alpha,
                    ..periods((0, 49), (50, 79))
                },
                &["alpha must be strictly between 0 and 1"],
            );
        }
        refused(
            &d,
            CausalImpactConfig {
                max_iter: 0,
                ..periods((0, 49), (50, 79))
            },
            &["max_iter"],
        );
        refused(
            &d,
            CausalImpactConfig {
                tolerance: 0.0,
                ..periods((0, 49), (50, 79))
            },
            &["tolerance"],
        );
    }

    #[test]
    fn test_validate_causal_impact_counterfactual_nile() {
        let (case, cds) = crate::test_support::load_case("structts_nile_ukgas");
        // Rows t = 1..100: nile is missing in rows 101-108.
        let d = Dataset::new(cds.df().slice(0, 100));
        let r = run_causal_impact(&d, "nile", "t", (1, 90), (91, 100), None).unwrap();
        let (m, s) = (&r.model, &r.series);
        let tol = Tol::iterative(1e-6);
        Check::new(&case)
            .named(
                "nile_pre90_coef_tight",
                &["level", "epsilon"],
                &[m.level_variance, m.observation_variance],
                tol,
            )
            .scalar("nile_pre90_loglik_tight", m.log_likelihood, Tol::stat(90))
            .vector("nile_pre90_forecast", &s.predicted[90..100], tol)
            .vector("nile_pre90_upper", &s.predicted_upper[90..100], tol)
            .vector("nile_pre90_point_effect", &s.point_effect[90..100], tol)
            .scalar("nile_pre90_cum_effect", r.summary.cumulative_effect, tol)
            .finish();
        assert_eq!((m.n_pre, m.n_post), (90, 10));
        assert!(r.convergence.converged);
    }

    #[test]
    fn test_causal_impact_tail_probability_is_not_zero() {
        // Effect -8 on noise of half-width 0.5 and level steps of half-width 1.
        let (y, time) = generate_causal_impact_data(70, 30, -8.0, 42);
        let r = run_causal_impact(&ds(&y, &time), "y", "time", (0, 69), (70, 99), None).unwrap();
        assert!(r.summary.cumulative_effect < 0.0);
        let z = r.inference.expected_effect / r.inference.effect_sd;
        // Below -9, 1 - Phi(|z|) is exactly 0 in double precision; above -37,
        // Phi(z) is still a normal double.
        assert!(-37.0 < z && z < -9.0, "{z}");
        assert!(r.summary.p_value > 0.0);
        assert!(r.inference.prob_positive > 0.0);
        let want = Normal::standard().sf(-z);
        assert!(
            ((r.summary.p_value - want) / want).abs() < 1e-12,
            "{} vs {want}",
            r.summary.p_value
        );
    }

    #[test]
    fn test_causal_impact_clipped_periods_warn() {
        let (y, time) = generate_causal_impact_data(70, 30, 5.0, 37);
        let r = causal_impact(&ds(&y, &time), "y", "time", periods((-5, 69), (70, 500))).unwrap();
        for needle in [
            "pre_period start -5 is before the first time value 0",
            "post_period end 500 is after the last time value 99",
        ] {
            assert!(
                r.warnings.iter().any(|w| w.contains(needle)),
                "{needle}: {:?}",
                r.warnings
            );
        }
        assert_eq!((r.model.n_pre, r.model.n_post), (70, 30));
        assert_eq!(r.notes_warnings(), r.warnings);
    }

    #[test]
    fn test_causal_impact_reports_convergence() {
        let (y, time) = generate_causal_impact_data(70, 30, 10.0, 555);
        let d = ds(&y, &time);
        let limited = causal_impact(
            &d,
            "y",
            "time",
            CausalImpactConfig {
                max_iter: 1,
                ..periods((0, 69), (70, 99))
            },
        )
        .unwrap();
        assert!(!limited.convergence.converged);
        assert!(
            limited.warnings[0].starts_with("CausalImpact did not converge after 1 iteration"),
            "{:?}",
            limited.warnings
        );
        assert_eq!(limited.notes_convergence().unwrap().iterations, 1);
        let default = causal_impact(&d, "y", "time", periods((0, 69), (70, 99))).unwrap();
        assert!(default.convergence.converged);
        assert!(default.warnings.is_empty(), "{:?}", default.warnings);
    }

    #[test]
    fn test_causal_impact_aliased_control_warns() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(29);
        let n = 80;
        let c1: Vec<f64> = (0..n).map(|_| rng.r#gen::<f64>()).collect();
        let c2: Vec<f64> = c1.iter().map(|c| 2.0 * c).collect();
        let y: Vec<f64> = c1
            .iter()
            .enumerate()
            .map(|(t, c)| {
                let effect = if t >= 60 { 3.0 } else { 0.0 };
                0.5 * c + (rng.r#gen::<f64>() - 0.5) + effect
            })
            .collect();
        let time: Vec<i64> = (0..n as i64).collect();
        let d = Dataset::new(df! {"y" => &y, "time" => &time, "c1" => &c1, "c2" => &c2}.unwrap());
        let r = run_causal_impact(&d, "y", "time", (0, 59), (60, 79), Some(&["c1", "c2"])).unwrap();
        assert!(r.model.regression_coefficients.as_ref().unwrap()[2].is_nan());
        for needle in [
            "Coefficients not defined because of singularities: c2",
            "the control regression is rank deficient",
        ] {
            assert!(
                r.warnings.iter().any(|w| w.contains(needle)),
                "{needle}: {:?}",
                r.warnings
            );
        }
        assert!(r.summary.cumulative_effect.is_finite());
    }

    #[test]
    fn test_causal_impact_empty_control_list_is_no_controls() {
        let (y, time) = generate_causal_impact_data(50, 30, 5.0, 19);
        let d = ds(&y, &time);
        let none = run_causal_impact(&d, "y", "time", (0, 49), (50, 79), None).unwrap();
        let empty = run_causal_impact(&d, "y", "time", (0, 49), (50, 79), Some(&[])).unwrap();
        assert_eq!(
            none.summary.cumulative_effect,
            empty.summary.cumulative_effect
        );
        assert!(empty.model.regression_coefficients.is_none());
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
    fn test_causal_impact_placebo_calibration() {
        // No R reference: the CausalImpact package is not installed, and p2a's interval is
        // the plug-in Gaussian forecast of a StructTS model, not bsts's posterior. Properties
        // on 200 seeded placebo series (local level, sigma_eta = 0.5, sigma_eps = 1,
        // n_pre = 80, n_post = 20, no intervention). The same design with stats::StructTS +
        // KalmanForecast in R (six seeds) gave one-step 95% coverage 0.905-0.965 and
        // all-horizon pointwise coverage 0.918-0.939 (MC SE about 0.015 and 0.009). The
        // cumulative interval sums marginal variances: its coverage was 0.35-0.46 there; it
        // is a registry deviation (TS-5), not asserted here.
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use rand_distr::{Distribution, StandardNormal};
        let (reps, n_pre, n_post) = (200usize, 80usize, 20usize);
        let mut rng = ChaCha8Rng::seed_from_u64(20261022);
        let (mut first, mut all, mut effects) = (0usize, 0.0f64, Vec::with_capacity(reps));
        for _ in 0..reps {
            let mut level = 100.0;
            let y: Vec<f64> = (0..n_pre + n_post)
                .map(|_| {
                    let eta: f64 = StandardNormal.sample(&mut rng);
                    let eps: f64 = StandardNormal.sample(&mut rng);
                    level += 0.5 * eta;
                    level + eps
                })
                .collect();
            let time: Vec<i64> = (0..(n_pre + n_post) as i64).collect();
            let r = run_causal_impact(
                &ds(&y, &time),
                "y",
                "time",
                (0, n_pre as i64 - 1),
                (n_pre as i64, (n_pre + n_post) as i64 - 1),
                None,
            )
            .unwrap();
            assert!(r.convergence.converged);
            let s = &r.series;
            let covers =
                |i: usize| s.point_effect_lower[i] <= 0.0 && 0.0 <= s.point_effect_upper[i];
            first += covers(n_pre) as usize;
            all += (n_pre..n_pre + n_post).filter(|&i| covers(i)).count() as f64 / n_post as f64;
            effects.push(r.summary.average_effect);
        }
        let first_rate = first as f64 / reps as f64;
        let all_rate = all / reps as f64;
        assert!(
            (0.88..=0.995).contains(&first_rate),
            "one-step coverage {first_rate}"
        );
        assert!(
            (0.88..=0.975).contains(&all_rate),
            "pointwise coverage {all_rate}"
        );
        let mean = effects.iter().sum::<f64>() / reps as f64;
        let sd =
            (effects.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (reps as f64 - 1.0)).sqrt();
        assert!(
            mean.abs() <= 4.0 * sd / (reps as f64).sqrt(),
            "average placebo effect {mean} (sd {sd})"
        );
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
        // A variance can be 0 at the bound, as in R.
        assert!(result.model.level_variance >= 0.0 && result.model.level_variance.is_finite());
        assert!(
            result.model.observation_variance >= 0.0
                && result.model.observation_variance.is_finite()
        );
        assert!(result.convergence.converged);
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
    fn test_causal_impact_cumulative_effect_recovers_dgp() {
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
    fn test_causal_impact_point_effects_recover_dgp() {
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
    fn test_causal_impact_intervals_are_ordered() {
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
    fn test_causal_impact_series_lengths() {
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

    fn cancelled<R>(f: impl FnOnce() -> R) -> R {
        let token = crate::resource::CancelToken::new();
        token.cancel();
        crate::resource::with_budget(crate::resource::Budget::new(None, token), f)
    }

    /// A seasonal (period 12) configuration over `n_pre` pre-period rows.
    fn seasonal_config(n_pre: usize, n_post: usize) -> CausalImpactConfig {
        CausalImpactConfig {
            pre_period: (0, (n_pre - 1) as i64),
            post_period: (n_pre as i64, (n_pre + n_post - 1) as i64),
            seasonal_period: Some(12),
            ..Default::default()
        }
    }

    #[test]
    fn causal_impact_refuses_a_kalman_history_above_the_allocation_ceiling() {
        // The counterfactual's kalman_filter keeps a covariance history that
        // grows with n_pre·m² (m = seasonal_period, + 1 with a trend), so with
        // a large state dimension (a seasonal model with a long period)
        // memory grew without bound. causal_impact now admits it, with the
        // forecast, before the fit, and the refusal names seasonal_period,
        // which sets m. Before the fix this returned Ok under the ceiling.
        let (n_pre, n_post) = (40, 10);
        let (y, time) = generate_causal_impact_data(n_pre, n_post, 2.0, 31);
        let dataset = Dataset::new(df! { "y" => &y, "time" => &time }.unwrap());
        let ceiling = Some(64 << 10);
        // m = 12: 40 · (16·12² + 96·12 + 176) + 10 · (8·12 + 64) = 146,880
        // bytes > 64 KiB; m = 13 with include_trend: 40 · 4,128 + 10 · 168 =
        // 166,800 bytes.
        for (include_trend, expected) in [
            (
                false,
                "with seasonal_period 12 the state dimension is seasonal_period = 12; \
                 use a smaller seasonal_period or a shorter pre_period or post_period",
            ),
            (
                true,
                "with seasonal_period 12 and include_trend the state dimension is \
                 seasonal_period + 1 = 13; use a smaller seasonal_period or a shorter \
                 pre_period or post_period",
            ),
        ] {
            let config = CausalImpactConfig {
                include_trend,
                ..seasonal_config(n_pre, n_post)
            };
            let refused = crate::resource::with_alloc_ceiling(ceiling, || {
                causal_impact(&dataset, "y", "time", config.clone())
            });
            match &refused {
                Err(EconError::ResourceLimit(msg)) => assert!(
                    msg.contains("Kalman filter")
                        && msg.contains("40 observations")
                        && msg.contains("10-step forecast")
                        && msg.ends_with(expected)
                        && msg.matches("; use ").count() == 1,
                    "{msg}"
                ),
                other => panic!(
                    "expected ResourceLimit, got {:?}",
                    other.as_ref().map(|r| &r.summary)
                ),
            }
            // The refusal comes before the fit: under a stopped budget it is
            // still the ResourceLimit, so no likelihood pass ran first.
            let refused_first = cancelled(|| {
                crate::resource::with_alloc_ceiling(ceiling, || {
                    causal_impact(&dataset, "y", "time", config.clone())
                })
            });
            assert!(
                matches!(refused_first, Err(EconError::ResourceLimit(_))),
                "{:?}",
                refused_first.map(|r| r.summary)
            );
        }
        // The hosted ceiling (256 MiB) leaves an ordinary seasonal model alone.
        let fitted = crate::resource::with_alloc_ceiling(Some(256 << 20), || {
            causal_impact(&dataset, "y", "time", seasonal_config(n_pre, n_post))
        });
        assert_eq!(fitted.unwrap().model.n_pre, n_pre);
    }

    #[test]
    fn causal_impact_admits_a_small_state_history_by_its_bytes() {
        // Without a seasonal component the state is one-dimensional: 48
        // bytes of values a filter step, but its nested Vecs hold several
        // times that, so counting values admitted a long pre-period's history
        // far past the ceiling. Before the fix this returned Ok.
        let (n_pre, n_post) = (1000, 10);
        let (y, time) = generate_causal_impact_data(n_pre, n_post, 2.0, 33);
        let dataset = Dataset::new(df! { "y" => &y, "time" => &time }.unwrap());
        let config = CausalImpactConfig {
            pre_period: (0, (n_pre - 1) as i64),
            post_period: (n_pre as i64, (n_pre + n_post - 1) as i64),
            ..Default::default()
        };
        // 1,000 · 320 + 10 · 80 = 320,800 bytes > 64 KiB (the history's
        // values: 48,000 bytes).
        let refused = crate::resource::with_alloc_ceiling(Some(64 << 10), || {
            causal_impact(&dataset, "y", "time", config.clone())
        });
        match &refused {
            Err(EconError::ResourceLimit(msg)) => assert!(
                msg.contains("1000 observations and a 1-dimensional state")
                    && msg.ends_with("; use a shorter pre_period or post_period"),
                "{msg}"
            ),
            other => panic!(
                "expected ResourceLimit, got {:?}",
                other.as_ref().map(|r| &r.summary)
            ),
        }
        // Under the hosted ceiling it fits.
        let fitted = crate::resource::with_alloc_ceiling(Some(256 << 20), || {
            causal_impact(&dataset, "y", "time", config)
        });
        assert_eq!(fitted.unwrap().model.n_pre, n_pre);
    }

    #[test]
    fn causal_impact_admits_the_counterfactual_as_one_figure() {
        // The counterfactual holds the pre-period's filter history while the
        // forecast over the post-period is built. With a one-dimensional
        // state, 100 pre-period rows (32,000 bytes) and 500 post-period rows
        // (40,000 bytes) each fit 64 KiB, but not together (72,000 bytes).
        // Before the fix this returned Ok: the history was never admitted
        // with the forecast, and the forecast was counted by its values
        // (4,000 bytes).
        let ceiling = Some(64 << 10);
        let fit = |n_post: usize| {
            let n_pre = 100;
            let (y, time) = generate_causal_impact_data(n_pre, n_post, 2.0, 35);
            let dataset = Dataset::new(df! { "y" => &y, "time" => &time }.unwrap());
            let config = CausalImpactConfig {
                pre_period: (0, (n_pre - 1) as i64),
                post_period: (n_pre as i64, (n_pre + n_post - 1) as i64),
                ..Default::default()
            };
            crate::resource::with_alloc_ceiling(ceiling, || {
                causal_impact(&dataset, "y", "time", config)
            })
        };
        match fit(500) {
            Err(EconError::ResourceLimit(msg)) => assert!(
                msg.contains("100 observations and a 1-dimensional state")
                    && msg.contains("500-step forecast")
                    && msg.ends_with("; use a shorter pre_period or post_period"),
                "{msg}"
            ),
            other => panic!(
                "expected ResourceLimit, got {:?}",
                other.as_ref().map(|r| &r.summary)
            ),
        }
        // 32,000 + 300 · 80 = 56,000 bytes fit.
        assert_eq!(fit(300).unwrap().model.n_post, 300);
    }

    /// Fits CausalImpact to `n_pre` + `n_post` rows of noisy data with
    /// `seasonal_period` and `include_trend`.
    fn seasonal_fit(
        n_pre: usize,
        n_post: usize,
        seasonal_period: usize,
        include_trend: bool,
        seed: u64,
    ) -> EconResult<CausalImpactResult> {
        let (y, time) = generate_causal_impact_data(n_pre, n_post, 2.0, seed);
        let dataset = Dataset::new(df! { "y" => &y, "time" => &time }.unwrap());
        let config = CausalImpactConfig {
            seasonal_period: Some(seasonal_period),
            include_trend,
            ..seasonal_config(n_pre, n_post)
        };
        causal_impact(&dataset, "y", "time", config)
    }

    #[test]
    fn causal_impact_refuses_a_seasonal_period_below_two() {
        // A seasonal cycle has at least two seasons. With seasonal_period 0
        // or 1 build_state_space_model indexed past its matrices: [0, 0] of a
        // 0 × 0 transition and observation[1] of a 1-vector, or with a trend
        // [0, 1] of a 1 × 1 and observation[2] of a 2-vector. Before the fix
        // all four panicked.
        for include_trend in [false, true] {
            for period in [0, 1] {
                match seasonal_fit(40, 10, period, include_trend, 37) {
                    Err(EconError::InvalidSpecification { message }) => assert!(
                        message.starts_with(&format!(
                            "seasonal_period must be at least 2, got {period}"
                        )),
                        "{message}"
                    ),
                    other => panic!(
                        "expected InvalidSpecification for seasonal_period {period}, got {:?}",
                        other.map(|r| r.summary)
                    ),
                }
            }
            // seasonal_period 2 is a model.
            assert_eq!(
                seasonal_fit(40, 10, 2, include_trend, 37)
                    .unwrap()
                    .model
                    .n_pre,
                40
            );
        }
    }

    #[test]
    fn causal_impact_refuses_a_seasonal_period_longer_than_the_pre_period() {
        // seasonal_period was never checked against the pre-period: one
        // longer than it fit a model in which no full season is observed,
        // and with no allocation ceiling (stdio, desktop, library) a huge one
        // passed the history admission and aborted the process in the m × m
        // Array2::zeros of build_state_space_model. The pre-period must now
        // hold at least one full season. Before the fix the refusal of 21
        // returned Ok.
        //
        // usize::MAX pins the order: the period is checked before the
        // history admission and before any m × m is sized by it. Before the
        // fix compute_state_dim saturated and the admission refused it with
        // a ResourceLimit; it must be the named InvalidSpecification.
        let (n_pre, n_post) = (20, 5);
        for include_trend in [false, true] {
            for period in [21, usize::MAX] {
                match seasonal_fit(n_pre, n_post, period, include_trend, 39) {
                    Err(EconError::InvalidSpecification { message }) => assert!(
                        message.starts_with(&format!(
                            "seasonal_period {period} is longer than the pre-period \
                             (20 observations)"
                        )) && message.contains("seasonal_period must be at most 20"),
                        "{message}"
                    ),
                    other => panic!(
                        "expected InvalidSpecification for seasonal_period {period}, got {:?}",
                        other.map(|r| r.summary)
                    ),
                }
            }
            // One full season in the pre-period is enough.
            assert_eq!(
                seasonal_fit(n_pre, n_post, n_pre, include_trend, 39)
                    .unwrap()
                    .model
                    .n_pre,
                n_pre
            );
        }
    }

    #[test]
    fn bsts_model_matrices_are_allocated_fallibly() {
        // The m × m transition and prior are allocated fallibly: a matrix the
        // allocator cannot provide is a ResourceLimit that names seasonal_period,
        // and one whose size overflows is refused before any allocation.
        let (y, _) = generate_causal_impact_data(40, 0, 0.0, 41);
        // 8 · 12² = 1,152 bytes and 8 · 13² = 1,352 bytes > 1 KiB.
        let ceiling = Some(1 << 10);
        for (include_trend, m, remedy) in [
            (
                false,
                12,
                "; with seasonal_period 12 the state dimension is seasonal_period = 12; \
                 use a smaller seasonal_period",
            ),
            (
                true,
                13,
                "; with seasonal_period 12 and include_trend the state dimension is \
                 seasonal_period + 1 = 13; use a smaller seasonal_period",
            ),
        ] {
            let config = CausalImpactConfig {
                include_trend,
                ..seasonal_config(40, 10)
            };
            let spec = causal_spec(&config);
            let variances = vec![1.0; spec.parameter_names().len()];
            // The remedy below is checked against the dimension the code computes.
            assert_eq!(spec.state_dim().unwrap(), m, "{config:?}");
            match crate::resource::with_alloc_ceiling(ceiling, || spec.model(&variances)) {
                Err(EconError::ResourceLimit(msg)) => assert!(
                    msg.starts_with("the seasonal model's transition matrix would need")
                        && msg.ends_with(remedy)
                        && msg.matches("; use ").count() == 1,
                    "{msg}"
                ),
                other => panic!("expected ResourceLimit, got {other:?}"),
            }
            match crate::resource::with_alloc_ceiling(ceiling, || spec.prior(&y)) {
                Err(EconError::ResourceLimit(msg)) => assert!(
                    msg.starts_with("the seasonal model's initial state covariance would need")
                        && msg.ends_with(remedy),
                    "{msg}"
                ),
                other => panic!("expected ResourceLimit, got {other:?}"),
            }
            // An m × m whose size overflows is refused before any allocation.
            let huge = StructuralSpec {
                seasonal: Some(usize::MAX / 4),
                ..spec
            };
            assert!(matches!(
                huge.model(&variances),
                Err(EconError::ResourceLimit(msg)) if msg.contains("overflows")
            ));
            assert!(matches!(
                huge.prior(&y),
                Err(EconError::ResourceLimit(msg)) if msg.contains("overflows")
            ));
        }
    }

    #[test]
    fn remedies_without_a_seasonal_component_name_no_period() {
        // Without a seasonal component the note is absent and the remedy names no
        // period, not "seasonal_period 0".
        for include_trend in [false, true] {
            let spec = causal_spec(&CausalImpactConfig {
                include_trend,
                seasonal_period: None,
                ..seasonal_config(40, 10)
            });
            let m = spec.state_dim().unwrap();
            assert_eq!(seasonal_dim_note(&spec, m), None, "{spec:?}");
            assert_eq!(
                history_remedy(&spec, m),
                "use a shorter pre_period or post_period"
            );
            // 8 · 1² and 8 · 2² bytes are both above a 4-byte ceiling.
            match crate::resource::with_alloc_ceiling(Some(4), || {
                spec.square(m, spec.naming.transition)
            }) {
                Err(EconError::ResourceLimit(msg)) => assert!(
                    msg.ends_with("; use a smaller seasonal_period")
                        && !msg.contains("seasonal_period 0")
                        && !msg.contains("state dimension"),
                    "{msg}"
                ),
                other => panic!("expected ResourceLimit, got {other:?}"),
            }
        }
        // With a period the note is the one both remedies carry.
        let seasonal = causal_spec(&seasonal_config(40, 10));
        let m = seasonal.state_dim().unwrap();
        let note = seasonal_dim_note(&seasonal, m).unwrap();
        assert_eq!(
            note,
            "with seasonal_period 12 the state dimension is seasonal_period = 12"
        );
        assert_eq!(
            history_remedy(&seasonal, m),
            format!("{note}; use a smaller seasonal_period or a shorter pre_period or post_period")
        );
    }

    #[test]
    fn state_dim_is_the_dimension_of_the_model_that_runs() {
        // The admission sizes the history by state_dim; the filter runs the model
        // the spec builds, from its prior. All must agree.
        let (y, _) = generate_causal_impact_data(40, 0, 0.0, 41);
        for (include_trend, seasonal_period, m) in [
            (false, None, 1),
            (true, None, 2),
            (false, Some(2), 2),
            (false, Some(12), 12),
            (true, Some(2), 3),
            (true, Some(12), 13),
        ] {
            let spec = causal_spec(&CausalImpactConfig {
                include_trend,
                seasonal_period,
                ..seasonal_config(40, 10)
            });
            let variances = vec![1.0; spec.parameter_names().len()];
            assert_eq!(spec.state_dim().unwrap(), m, "{spec:?}");
            assert_eq!(spec.model(&variances).unwrap().state_dim(), m, "{spec:?}");
            let (a0, p0, _) = spec.prior(&y).unwrap();
            assert_eq!(a0.len(), m);
            assert_eq!(p0.dim(), (m, m));
        }
    }

    #[test]
    fn fit_structural_keeps_no_history_and_stops_on_a_stopped_budget() {
        // The likelihood passes keep no Kalman history: had they run kalman_filter,
        // which admits its history, every evaluation under this ceiling would be
        // ResourceLimit. A stopped budget stops the search instead of scoring.
        let (y, _) = generate_causal_impact_data(40, 0, 0.0, 32);
        let spec = causal_spec(&seasonal_config(40, 1));
        let run = || fit_structural(&y, spec, None, 100, 1e-8);
        // The history would be 145,280 bytes, above 64 KiB; the fit keeps none of
        // it and finds exactly what it finds without a ceiling.
        let bounded = crate::resource::with_alloc_ceiling(Some(64 << 10), run).unwrap();
        assert_eq!(bounded.variances, run().unwrap().variances);
        assert!(matches!(cancelled(run), Err(EconError::Cancelled(_))));
        // Without a budget or a ceiling nothing changes.
        assert!(run().is_ok());
    }
}
