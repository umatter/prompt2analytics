//! Structural time-series models for causal inference (the CausalImpact method).
//!
//! Estimates the effect of an intervention as the difference between the observed series
//! and a counterfactual: the forecast of a structural time-series model fitted to the
//! pre-intervention period, optionally with control series as regression terms.
//!
//! What is computed. The model (local level, optional local linear trend, optional
//! seasonal) is fitted by maximum likelihood with the StructTS engine (diffuse prior,
//! R's objective, projected Newton). The counterfactual is the model's Gaussian forecast
//! with the fitted variances plugged in. Pointwise intervals come from the forecast
//! variances; the intervals of the cumulative and average effects from the variance of the
//! **sum** of the post-period forecast errors, which correlate through the state (the joint
//! forecast-error covariance, Durbin and Koopman 2012, sec. 4.13). Control series enter as
//! diffuse regression states: their coefficients are the GLS regression of the response's
//! filter innovations on theirs, and their uncertainty is carried into every interval. The
//! p-value is two-sided under the normal approximation. There is no prior and no
//! posterior: the `CausalImpact` package (bsts, a Bayesian sampler) is the method's source,
//! not its algorithm, and its numbers are not reproduced
//! (`validation/forecasting/causal_impact.md`). The intervals do not include the
//! uncertainty of the estimated variances.
//!
//! # Algorithm
//!
//! 1. **Pre-period model fitting**: fit the structural model to the pre-intervention data
//!    (with controls: to the residuals of an OLS regression on them).
//! 2. **Counterfactual**: filter the pre-period and forecast the gap and the
//!    post-intervention period.
//! 3. **Effects**: observed minus counterfactual, pointwise, cumulative, average and
//!    relative, with confidence intervals and a p-value.
//!
//! # References
//!
//! - Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*
//!   (2nd ed.), sec. 4.13 (forecasting) and sec. 6.2 (regression effects as diffuse states).
//!   Oxford University Press.
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
use crate::linalg::solve::{aliasing_message, least_squares};
use crate::regression::ols::solve_ols;
use crate::resource::{checked_len, checkpoint, try_vec, try_zeros2, with_remedy};
use crate::traits::convergence::{Convergence, ResultNotes};
use crate::traits::estimator::{normal_p_value_two_sided, validate_level, z_critical};

/// Configuration for CausalImpact analysis.
#[derive(Debug, Clone)]
pub struct CausalImpactConfig {
    /// Pre-intervention period (inclusive start and end indices or timestamps).
    pub pre_period: (i64, i64),
    /// Post-intervention period (inclusive start and end indices or timestamps).
    pub post_period: (i64, i64),
    /// Names of control series columns (if any).
    pub control_series: Option<Vec<String>>,
    /// Significance level: 1 - alpha confidence intervals (default 0.05 for 95%).
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
        }
    }
}

/// Summary statistics for causal effect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalImpactSummary {
    /// Average causal effect per time point in post-period.
    pub average_effect: f64,
    /// Lower bound of the confidence interval for the average effect.
    pub average_effect_lower: f64,
    /// Upper bound of the confidence interval for the average effect.
    pub average_effect_upper: f64,
    /// Cumulative (total) causal effect over post-period.
    pub cumulative_effect: f64,
    /// Lower bound of the confidence interval for the cumulative effect: from the variance
    /// of the sum of the post-period forecast errors.
    pub cumulative_effect_lower: f64,
    /// Upper bound of the confidence interval for the cumulative effect.
    pub cumulative_effect_upper: f64,
    /// Relative effect (cumulative effect / sum of predicted); NaN, with a warning, when the
    /// counterfactual sums to 0.
    pub relative_effect: f64,
    /// Lower bound of the relative effect (delta method, observed sum fixed).
    pub relative_effect_lower: f64,
    /// Upper bound of relative effect.
    pub relative_effect_upper: f64,
    /// Two-sided p-value of the cumulative effect, 2 Phi(-|z|) with z its estimate over its
    /// standard error (normal approximation, fitted parameters plugged in).
    pub p_value: f64,
    /// Whether `p_value < alpha`, that is, whether the 1 - alpha interval of the cumulative
    /// effect excludes 0.
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
    /// Lower bound of the pointwise prediction interval.
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
    /// Lower bound of the cumulative effect's interval (variance of the running sum).
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
    /// Regression coefficients of the control series, one per control in `control_names`
    /// (no intercept: the level absorbs it): GLS on the filter innovations, NaN for an
    /// aliased control.
    pub regression_coefficients: Option<Vec<f64>>,
    /// Standard errors of `regression_coefficients`, NaN for an aliased control.
    #[serde(default)]
    pub regression_std_errors: Option<Vec<f64>>,
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

/// Normal-approximation probabilities of the cumulative effect under the fitted model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalInference {
    /// Phi(z), z the cumulative effect over its standard error.
    pub prob_positive: f64,
    /// Phi(-z).
    pub prob_negative: f64,
    /// The cumulative effect.
    pub expected_effect: f64,
    /// Standard error of the cumulative effect.
    pub effect_sd: f64,
    /// Whether the two-sided p-value is below alpha.
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

    // Step 1: with controls, the variances are fitted to what an OLS regression on the
    // controls leaves of the pre-period; the coefficients themselves come from the
    // state-space model (step 3).
    let y_pre = &y[pre_start..=pre_end];
    let residuals = match controls {
        Some(controls) => Some(ols_residuals(y, controls, periods)?),
        None => None,
    };
    let y_for_model = residuals.as_deref().unwrap_or(y_pre);

    // Step 2: The structural model on the pre-period.
    let fit = fit_structural(y_for_model, spec, None, config.max_iter, config.tolerance)?;
    warnings.extend(fit.warnings.iter().cloned());

    // Step 3: The filter over the pre-period, the forecast beyond it, the variance of the
    // running sum of the post-period forecast errors.
    let control_names = config.control_series.as_deref().unwrap_or_default();
    let cf = predict_counterfactual(
        y,
        controls.map(Vec::as_slice),
        control_names,
        &fit,
        periods,
        &mut warnings,
    )?;
    let predicted = cf.predicted;
    let predicted_var = cf.predicted_var;

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
            // The variance of the sum so far, not the sum of the variances.
            cumulative_var = cf.cumulative_var[i - post_start];
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
    let observed_sum: f64 = y[post_start..=post_end].iter().sum();
    let predicted_sum: f64 = predicted[post_start..=post_end].iter().sum();
    let cumulative_effect: f64 = series.point_effect[post_start..=post_end].iter().sum();
    let cumulative_se = cf.cumulative_var[n_post - 1].sqrt();

    let average_effect = cumulative_effect / n_post as f64;
    let average_se = cumulative_se / n_post as f64;

    let (relative_effect, relative_effect_lower, relative_effect_upper, relative_warning) =
        relative_effect(observed_sum, predicted_sum, cumulative_se, z_alpha);
    warnings.extend(relative_warning);

    // The normal approximation under the fitted model: Phi(z) and Phi(-z) for the sign of
    // the effect, the two-sided p-value 2 Phi(-|z|), which is below alpha exactly when the
    // 1 - alpha interval excludes 0.
    let z_cumulative = if cumulative_se > 0.0 {
        cumulative_effect / cumulative_se
    } else {
        0.0
    };
    let normal = Normal::standard();
    let prob_positive = normal.cdf(z_cumulative);
    let prob_negative = normal.cdf(-z_cumulative);
    let p_value = normal_p_value_two_sided(z_cumulative);

    let summary = CausalImpactSummary {
        average_effect,
        average_effect_lower: average_effect - z_alpha * average_se,
        average_effect_upper: average_effect + z_alpha * average_se,
        cumulative_effect,
        cumulative_effect_lower: cumulative_effect - z_alpha * cumulative_se,
        cumulative_effect_upper: cumulative_effect + z_alpha * cumulative_se,
        relative_effect,
        relative_effect_lower,
        relative_effect_upper,
        p_value,
        significant: p_value < config.alpha,
        alpha: config.alpha,
    };

    // Model info
    let n_params = count_params(config, controls.is_some());
    let coef = spec.coefficients(&fit.variances);
    let log_lik = fit.log_likelihood;
    let (regression_coefficients, regression_std_errors) = match cf.regression {
        Some((b, se)) => (Some(b), Some(se)),
        None => (None, None),
    };
    let model = CausalImpactModel {
        level_variance: coef.level,
        slope_variance: coef.slope,
        seasonal_variance: coef.seasonal,
        observation_variance: coef.epsilon,
        regression_coefficients,
        regression_std_errors,
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

/// The relative effect (Y - P) / P of an observed post-period sum Y against the
/// counterfactual sum P, and its interval by the delta method with Y fixed: the standard
/// error of P is `cumulative_se`, so that of Y / P - 1 is |Y| / P^2 times it. Not defined
/// (NaN, with a warning) when P is 0 or not finite. Written as (|Y| / |P|) (SE / |P|), so
/// that no square of P can underflow.
fn relative_effect(
    observed_sum: f64,
    predicted_sum: f64,
    cumulative_se: f64,
    z: f64,
) -> (f64, f64, f64, Option<String>) {
    if predicted_sum == 0.0 || !predicted_sum.is_finite() {
        return (
            f64::NAN,
            f64::NAN,
            f64::NAN,
            Some(format!(
                "the counterfactual sums to {predicted_sum} over the post-period: the relative \
                 effect is not defined"
            )),
        );
    }
    let rel = (observed_sum - predicted_sum) / predicted_sum;
    let se = (observed_sum.abs() / predicted_sum.abs()) * (cumulative_se / predicted_sum.abs());
    (rel, rel - z * se, rel + z * se, None)
}

/// The pre-period residuals of `y` on an intercept and the controls (OLS): the series the
/// model's variances are fitted to when there are controls. An exact fit is refused: the
/// model would have no variance to estimate.
fn ols_residuals(
    y: &[f64],
    controls: &[Vec<f64>],
    periods: &PeriodIndices,
) -> EconResult<Vec<f64>> {
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
    // Residuals at the rounding level of y are an exact fit.
    let rss: f64 = solved.residuals.iter().map(|r| r * r).sum();
    let yss: f64 = y_pre.iter().map(|v| v * v).sum();
    let eps_n = periods.n_pre as f64 * f64::EPSILON;
    if rss <= eps_n * eps_n * yss {
        return Err(EconError::InvalidSpecification {
            message: "the control regression leaves no residual variance in the pre-period; \
                      drop controls that reproduce the response"
                .to_string(),
        });
    }
    Ok(solved.residuals.to_vec())
}

/// What the filter and the forecast give the effects.
struct Counterfactual {
    /// The counterfactual at every row.
    predicted: Vec<f64>,
    /// Its pointwise variance.
    predicted_var: Vec<f64>,
    /// The variance of the running sum of the post-period forecast errors, one entry per
    /// post-period step.
    cumulative_var: Vec<f64>,
    /// The controls' coefficients and standard errors, NaN for an aliased control.
    regression: Option<(Vec<f64>, Vec<f64>)>,
}

/// The nonzero entries of a matrix, for the O(nnz m) products of the structural models.
fn nonzeros(a: &ndarray::Array2<f64>) -> Vec<(usize, usize, f64)> {
    a.indexed_iter()
        .filter(|(_, v)| **v != 0.0)
        .map(|((i, j), &v)| (i, j, v))
        .collect()
}

/// out = T x for a vector `x`.
fn t_times(t: &[(usize, usize, f64)], x: &[f64], out: &mut [f64]) {
    out.fill(0.0);
    for &(i, j, v) in t {
        out[i] += v * x[j];
    }
}

/// The variance of the running sum of the forecast errors over the post-period, one entry
/// per post-period step, under the model whose filtered state covariance at the end of the
/// pre-period is `p_final` (row-major m x m), for a forecast over `horizon` steps of which
/// the first `post_from` are the gap between the periods.
///
/// Cov(e_i, e_j) = Z' T^(j-i) P_i Z for j > i, plus H on the diagonal, so the variance of the
/// sum is 1' Sigma 1, not the sum of the marginal variances. With P_h = T P_{h-1} T' + R Q R',
/// g_h = T g_{h-1} + 1{h in post} P_h Z and V_h = V_{h-1} + 1{h in post} (2 Z'g_h - Z'P_h Z + H)
/// one pass computes it: O(m^2) memory, the cost of the forecast's own covariance step.
pub(crate) fn cumulative_forecast_variance(
    model: &crate::forecasting::kalman::StateSpaceModel,
    p_final: &[f64],
    horizon: usize,
    post_from: usize,
) -> EconResult<Vec<f64>> {
    const WHAT: &str = "the cumulative forecast variance";
    let m = model.state_dim();
    let mm = checked_len(&[m, m], WHAT)?;
    if p_final.len() != mm || post_from > horizon {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "{WHAT}: a {m} x {m} covariance and a gap within the horizon are required"
            ),
        });
    }
    let t = nonzeros(&model.transition);
    let rqr = nonzeros(&model.state_disturbance_cov());
    let z: Vec<f64> = model.observation.iter().copied().collect();
    let h = model.obs_var;

    let mut p = try_vec(mm, 0.0, WHAT)?;
    p.copy_from_slice(p_final);
    let mut tp = try_vec(mm, 0.0, WHAT)?;
    let mut g = try_vec(m, 0.0, WHAT)?;
    let mut g_next = try_vec(m, 0.0, WHAT)?;
    let mut pz = try_vec(m, 0.0, WHAT)?;
    let mut out = try_vec(horizon - post_from, 0.0, WHAT)?;
    let mut v = 0.0;
    for step in 1..=horizon {
        checkpoint()?;
        // tp = T P
        tp.fill(0.0);
        for &(i, j, a) in &t {
            for c in 0..m {
                tp[i * m + c] += a * p[j * m + c];
            }
        }
        // P = (T P) T' + R Q R'
        p.fill(0.0);
        for &(i, j, a) in &t {
            for r in 0..m {
                p[r * m + i] += tp[r * m + j] * a;
            }
        }
        for &(i, j, a) in &rqr {
            p[i * m + j] += a;
        }
        if step > post_from {
            for r in 0..m {
                pz[r] = (0..m).map(|c| p[r * m + c] * z[c]).sum();
            }
            t_times(&t, &g, &mut g_next);
            for (gn, pzr) in g_next.iter_mut().zip(&pz) {
                *gn += pzr;
            }
            std::mem::swap(&mut g, &mut g_next);
            let zg: f64 = z.iter().zip(&g).map(|(a, b)| a * b).sum();
            let zpz: f64 = z.iter().zip(&pz).map(|(a, b)| a * b).sum();
            v += 2.0 * zg - zpz + h;
            out[step - post_from - 1] = v;
        }
    }
    Ok(out)
}

/// The counterfactual and its variance at all `n` rows: in the pre-period the filtered
/// fit, in the gap and the post-period the forecast of the fitted model, outside the two
/// periods the nearest value; and the variance of the running post-period sum.
///
/// Without controls the filter runs on the pre-period of `y`. With controls, the
/// regression is carried as diffuse static states (de Jong's augmented filter): `y` is
/// filtered from the fit's prior with its first value as the initial level, each control
/// runs a mean-only pass with `y`'s gains (the gains do not depend on the data), and the
/// coefficients are the GLS regression of `y`'s innovations on the controls' with weights
/// 1 / F_t, no intercept (the level absorbs it). Their covariance (sum v_t v_t' / F_t)^-1
/// enters every forecast step as d' Var(beta) d, d the controls' forecast errors, and the
/// running sum as D' Var(beta) D. Pre-period bands treat the coefficients as known.
fn predict_counterfactual(
    y: &[f64],
    controls: Option<&[Vec<f64>]>,
    control_names: &[String],
    fit: &StructuralFit,
    periods: &PeriodIndices,
    warnings: &mut Vec<String>,
) -> EconResult<Counterfactual> {
    let PeriodIndices {
        pre_start,
        pre_end,
        post_start,
        post_end,
        n_pre,
        n_gap,
        ..
    } = *periods;
    let n = y.len();
    let horizon = periods.horizon()?;
    let ssm = &fit.model;
    let m = ssm.state_dim();
    let y_pre = &y[pre_start..=pre_end];
    let mut init_state = fit.init_state.clone();
    init_state[0] = y_pre[0];
    let filter_result = kalman_filter(y_pre, ssm, init_state.view(), fit.init_cov.view())?;
    let forecast_result = kalman_forecast(&filter_result, ssm, horizon)?;
    let p_final: Vec<f64> = filter_result.filtered_covs[n_pre - 1]
        .iter()
        .flatten()
        .copied()
        .collect();
    let mut cumulative_var = cumulative_forecast_variance(ssm, &p_final, horizon, n_gap)?;

    const WHAT: &str = "the counterfactual";
    let mut predicted = try_vec(n, 0.0, WHAT)?;
    let mut predicted_var = try_vec(n, 0.0, WHAT)?;
    let z: Vec<f64> = ssm.observation.iter().copied().collect();
    let z_dot = |a: &[f64]| -> f64 { z.iter().zip(a).map(|(p, q)| p * q).sum() };

    // Pre-period: Z' a_t|t and Z' P_t|t Z + H.
    for (i, (state, cov)) in filter_result
        .filtered_states
        .iter()
        .zip(&filter_result.filtered_covs)
        .enumerate()
    {
        let idx = pre_start + i;
        let quadratic: f64 = z
            .iter()
            .zip(cov)
            .filter(|(zj, _)| **zj != 0.0)
            .map(|(zj, row)| zj * z_dot(row))
            .sum();
        predicted[idx] = z_dot(state);
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
        predicted[idx] = *forecast;
        predicted_var[idx] = *variance;
    }

    let regression = match controls {
        Some(controls) => {
            let reg = control_states(
                controls,
                control_names,
                &filter_result,
                ssm,
                &init_state,
                periods,
                warnings,
            )?;
            let k = controls.len();
            let identified = |j: usize| {
                let b = reg.beta[j];
                if b.is_nan() { 0.0 } else { b }
            };
            let vb = |i: usize, j: usize| {
                let v = reg.cov[[i, j]];
                if v.is_nan() { 0.0 } else { v }
            };
            // Pre-period: Z'a(y) - beta' Z'a(X) + beta' x_t.
            for t in 0..n_pre {
                let idx = pre_start + t;
                predicted[idx] += (0..k)
                    .map(|j| identified(j) * (controls[j][idx] - reg.filtered_means[[t, j]]))
                    .sum::<f64>();
            }
            // Forecast rows: the controls' state forecasts T^h a_T(x).
            let t_nz = nonzeros(&ssm.transition);
            let mut states = reg.final_states;
            let mut next = try_vec(m, 0.0, WHAT)?;
            let mut d = try_vec(k, 0.0, WHAT)?;
            let mut big_d = try_vec(k, 0.0, WHAT)?;
            for h in 0..horizon {
                checkpoint()?;
                let idx = pre_end + 1 + h;
                for (j, state) in states.iter_mut().enumerate() {
                    t_times(&t_nz, state, &mut next);
                    state.copy_from_slice(&next);
                    d[j] = controls[j][idx] - z_dot(state);
                }
                let quad = |u: &[f64]| -> f64 {
                    (0..k)
                        .map(|i| (0..k).map(|j| u[i] * vb(i, j) * u[j]).sum::<f64>())
                        .sum()
                };
                predicted[idx] += (0..k).map(|j| identified(j) * d[j]).sum::<f64>();
                predicted_var[idx] += quad(&d);
                if idx >= post_start {
                    for (bd, dj) in big_d.iter_mut().zip(&d) {
                        *bd += dj;
                    }
                    cumulative_var[idx - post_start] += quad(&big_d);
                }
            }
            let se: Vec<f64> = (0..k).map(|j| reg.cov[[j, j]].sqrt()).collect();
            Some((reg.beta, se))
        }
        None => None,
    };

    // Outside the two periods, the nearest value.
    for i in 0..pre_start {
        predicted[i] = predicted[pre_start];
        predicted_var[i] = predicted_var[pre_start];
    }
    for i in (post_end + 1)..n {
        predicted[i] = predicted[post_end];
        predicted_var[i] = predicted_var[post_end];
    }

    Ok(Counterfactual {
        predicted,
        predicted_var,
        cumulative_var,
        regression,
    })
}

/// The controls as diffuse regression states: what [`predict_counterfactual`] needs.
struct ControlStates {
    /// GLS coefficients, NaN for an aliased control.
    beta: Vec<f64>,
    /// Their covariance, NaN rows and columns for an aliased control.
    cov: ndarray::Array2<f64>,
    /// The innovations of each control's mean-only pass over the pre-period (n_pre x k).
    innovations: ndarray::Array2<f64>,
    /// Z' a_t|t of each control's mean-only pass over the pre-period (n_pre x k).
    filtered_means: ndarray::Array2<f64>,
    /// Each control's filtered state at the end of the pre-period.
    final_states: Vec<Vec<f64>>,
}

/// One mean-only filter pass per control with `y`'s gains M_t = P_t|t-1 Z and innovation
/// variances F_t, then the GLS regression of `y`'s innovations on the controls'.
fn control_states(
    controls: &[Vec<f64>],
    control_names: &[String],
    filter_result: &crate::forecasting::kalman::KalmanFilterResult,
    ssm: &crate::forecasting::kalman::StateSpaceModel,
    init_state: &ndarray::Array1<f64>,
    periods: &PeriodIndices,
    warnings: &mut Vec<String>,
) -> EconResult<ControlStates> {
    const WHAT: &str = "the control series' filter passes";
    let PeriodIndices {
        pre_start, n_pre, ..
    } = *periods;
    let k = controls.len();
    let m = ssm.state_dim();
    let t_nz = nonzeros(&ssm.transition);
    let z: Vec<f64> = ssm.observation.iter().copied().collect();
    let z_dot = |a: &[f64]| -> f64 { z.iter().zip(a).map(|(p, q)| p * q).sum() };

    let mut innovations = try_zeros2(n_pre, k, WHAT)?;
    let mut filtered_means = try_zeros2(n_pre, k, WHAT)?;
    let mut states: Vec<Vec<f64>> = Vec::new();
    for control in controls {
        let mut a = try_vec(m, 0.0, WHAT)?;
        for (s, v) in a.iter_mut().zip(init_state.iter()) {
            *s = *v;
        }
        a[0] = control[pre_start];
        states.push(a);
    }
    let mut predicted = try_vec(m, 0.0, WHAT)?;
    let mut gain = try_vec(m, 0.0, WHAT)?;
    for t in 0..n_pre {
        checkpoint()?;
        let f = filter_result.innovation_vars[t];
        let p_pred = &filter_result.predicted_covs[t];
        for (r, row) in p_pred.iter().enumerate() {
            gain[r] = z_dot(row);
        }
        for (j, state) in states.iter_mut().enumerate() {
            t_times(&t_nz, state, &mut predicted);
            let v = controls[j][pre_start + t] - z_dot(&predicted);
            for (s, (pr, g)) in state.iter_mut().zip(predicted.iter().zip(&gain)) {
                *s = pr + g * v / f;
            }
            innovations[[t, j]] = v;
            filtered_means[[t, j]] = z_dot(state);
        }
    }

    // GLS on the innovations: rows scaled by 1 / sqrt(F_t), no intercept.
    let mut xw = try_zeros2(n_pre, k, WHAT)?;
    let mut yw = crate::resource::try_zeros1(n_pre, WHAT)?;
    for t in 0..n_pre {
        let w = filter_result.innovation_vars[t].sqrt().recip();
        yw[t] = filter_result.innovations[t] * w;
        for j in 0..k {
            xw[[t, j]] = innovations[[t, j]] * w;
        }
    }
    let ls = least_squares(xw.view(), yw.view())?;
    if let Some(message) = aliasing_message(&ls.aliased, control_names) {
        warnings.push(message);
        warnings.push(
            "the control regression (GLS on the filter innovations) is rank deficient: the \
             counterfactual uses the identified coefficients only"
                .to_string(),
        );
    }
    Ok(ControlStates {
        beta: ls.coefficients.to_vec(),
        cov: ls.unscaled_cov,
        innovations,
        filtered_means,
        final_states: states,
    })
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
            n += controls.len(); // one coefficient per control; the level absorbs an intercept
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

    /// structts_series.csv (Nile 1871-1970 with 8 NA rows, log10 UKgas 1960-1986).
    fn structts_series() -> Dataset {
        crate::test_support::load_case("structts_nile_ukgas").1
    }

    /// Pins the intervals of one fit of case causal_impact_intervals_t4 (prefix `p`): the
    /// cumulative SE, bounds, average bounds, relative effect and bounds, Phi(z), the
    /// two-sided p-value and the running cumulative band on the post rows.
    fn check_intervals(
        chk: &mut Check,
        p: &str,
        r: &CausalImpactResult,
        post: std::ops::Range<usize>,
    ) {
        let tol = Tol::iterative(1e-6);
        let (s, i) = (&r.summary, &r.inference);
        let k = |name: &str| format!("{p}_{name}");
        chk.vector(&k("cum_effect"), &[s.cumulative_effect], tol)
            .vector(&k("cum_se"), &[i.effect_sd], tol)
            .vector(&k("cum_lower"), &[s.cumulative_effect_lower], tol)
            .vector(&k("cum_upper"), &[s.cumulative_effect_upper], tol)
            .vector(&k("avg_lower"), &[s.average_effect_lower], tol)
            .vector(&k("avg_upper"), &[s.average_effect_upper], tol)
            .vector(&k("rel"), &[s.relative_effect], tol)
            .vector(&k("rel_lower"), &[s.relative_effect_lower], tol)
            .vector(&k("rel_upper"), &[s.relative_effect_upper], tol)
            .vector(&k("prob_positive"), &[i.prob_positive], tol)
            .vector(&k("p_two_sided"), &[s.p_value], tol)
            .vector(
                &k("cum_series_lower"),
                &r.series.cumulative_effect_lower[post.clone()],
                tol,
            )
            .vector(
                &k("cum_series_upper"),
                &r.series.cumulative_effect_upper[post],
                tol,
            );
    }

    /// The variance of the cumulative effect is that of the sum of the post-period forecast
    /// errors (case causal_impact_intervals_t4: StructTS level on Nile 1-90, forecast 91-100;
    /// R's marginal sum 263,532 against the joint 1,027,927).
    #[test]
    fn test_validate_causal_impact_intervals_nile_level() {
        let case = p2a_validation::Case::load("causal_impact_intervals_t4");
        let d = Dataset::new(structts_series().df().slice(0, 100));
        let r = run_causal_impact(&d, "nile", "t", (1, 90), (91, 100), None).unwrap();
        let mut chk = Check::new(&case);
        chk.named(
            "nile_level_coef_tight",
            &["level", "epsilon"],
            &[r.model.level_variance, r.model.observation_variance],
            Tol::iterative(1e-6),
        );
        check_intervals(&mut chk, "nile_level", &r, 90..100);
        chk.finish();
        // The p-value is two-sided and agrees with the interval.
        assert!(!r.summary.significant);
        assert!(r.summary.cumulative_effect_lower < 0.0 && r.summary.cumulative_effect_upper > 0.0);
    }

    #[test]
    fn test_validate_causal_impact_intervals_nile_trend() {
        let case = p2a_validation::Case::load("causal_impact_intervals_t4");
        let d = Dataset::new(structts_series().df().slice(0, 100));
        let config = CausalImpactConfig {
            include_trend: true,
            ..periods((1, 90), (91, 100))
        };
        let r = causal_impact(&d, "nile", "t", config).unwrap();
        let mut chk = Check::new(&case);
        chk.named(
            "nile_trend_coef_tight",
            &["level", "slope", "epsilon"],
            &[
                r.model.level_variance,
                r.model.slope_variance.unwrap(),
                r.model.observation_variance,
            ],
            Tol::iterative(1e-6),
        );
        check_intervals(&mut chk, "nile_trend", &r, 90..100);
        chk.finish();
    }

    #[test]
    fn test_validate_causal_impact_intervals_ukgas_bsm() {
        let case = p2a_validation::Case::load("causal_impact_intervals_t4");
        let d = structts_series();
        let config = CausalImpactConfig {
            include_trend: true,
            seasonal_period: Some(4),
            ..periods((1, 92), (93, 108))
        };
        let r = causal_impact(&d, "ukgas", "t", config).unwrap();
        let mut chk = Check::new(&case);
        chk.named(
            "ukgas_bsm_coef_tight",
            &["level", "slope", "seas", "epsilon"],
            &[
                r.model.level_variance,
                r.model.slope_variance.unwrap(),
                r.model.seasonal_variance.unwrap(),
                r.model.observation_variance,
            ],
            Tol::iterative(1e-6),
        );
        check_intervals(&mut chk, "ukgas_bsm", &r, 92..108);
        chk.finish();
    }

    /// A control series as diffuse regression states (case causal_impact_control_t4: y = 2x +
    /// level + noise, effect 3 from t = 81): the GLS coefficient and its SE, the
    /// counterfactual, its pointwise variance and the cumulative interval with the
    /// coefficient's uncertainty.
    #[test]
    fn test_validate_causal_impact_intervals_control() {
        let case = p2a_validation::Case::load("causal_impact_intervals_t4");
        let path =
            p2a_validation::repo_path("validation/datasets/generated/causal_impact_control_t4.csv");
        let d = Dataset::new(crate::data::DataLoader::load_csv(&path).unwrap());
        let r = run_causal_impact(&d, "y", "t", (1, 80), (81, 100), Some(&["x"])).unwrap();
        let tol = Tol::iterative(1e-6);
        let z = z_critical(0.95).unwrap();
        let point_var: Vec<f64> = (80..100)
            .map(|i| ((r.series.predicted_upper[i] - r.series.predicted[i]) / z).powi(2))
            .collect();
        let mut chk = Check::new(&case);
        chk.named(
            "ctrl_coef_tight",
            &["level", "epsilon"],
            &[r.model.level_variance, r.model.observation_variance],
            tol,
        )
        .scalar(
            "ctrl_beta",
            r.model.regression_coefficients.as_ref().unwrap()[0],
            tol,
        )
        .scalar(
            "ctrl_beta_se",
            r.model.regression_std_errors.as_ref().unwrap()[0],
            tol,
        )
        .vector("ctrl_pred", &r.series.predicted[80..100], tol)
        .vector("ctrl_point_var", &point_var, tol);
        check_intervals(&mut chk, "ctrl", &r, 80..100);
        chk.finish();
        assert!(r.summary.significant, "{}", r.summary.p_value);
        assert_eq!(r.model.regression_coefficients.as_ref().unwrap().len(), 1);
    }

    /// The running variance of the sum equals the sum of the post-period block of the full
    /// forecast-error covariance Z' T^(j-i) P_i Z + H 1{i = j}, built here entry by entry, and
    /// its diagonal is kalman_forecast's variance.
    #[test]
    fn cumulative_variance_matches_the_full_covariance() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(20261116);
        for (slope, seasonal) in [
            (false, None),
            (true, None),
            (false, Some(5)),
            (true, Some(12)),
        ] {
            for n_gap in [0usize, 3] {
                let spec = StructuralSpec {
                    slope,
                    seasonal,
                    naming: CAUSAL_IMPACT_NAMING,
                };
                let m = spec.state_dim().unwrap();
                let variances: Vec<f64> = (0..spec.parameter_names().len())
                    .map(|_| 0.2 + rng.r#gen::<f64>())
                    .collect();
                let model = spec.model(&variances).unwrap();
                let y: Vec<f64> = (0..40)
                    .map(|t| 10.0 + 0.1 * t as f64 + (t % 4) as f64 + rng.r#gen::<f64>())
                    .collect();
                let (a0, p0, _) = spec.prior(&y).unwrap();
                let fr = kalman_filter(&y, &model, a0.view(), p0.view()).unwrap();
                let n_post = 9;
                let horizon = n_gap + n_post;
                let fc = kalman_forecast(&fr, &model, horizon).unwrap();
                let p_final: Vec<f64> = fr.filtered_covs[39].iter().flatten().copied().collect();
                let got = cumulative_forecast_variance(&model, &p_final, horizon, n_gap).unwrap();
                // The forecast covariances P_h written out densely.
                let t = model.transition.clone();
                let rqr = model.state_disturbance_cov();
                let zv = model.observation.clone();
                let mut p = ndarray::Array2::from_shape_vec((m, m), p_final.clone()).unwrap();
                let mut ps = Vec::new();
                for _ in 0..horizon {
                    p = t.dot(&p).dot(&t.t()) + &rqr;
                    ps.push(p.clone());
                }
                let mut sigma = vec![vec![0.0; horizon]; horizon];
                for i in 0..horizon {
                    let mut tp = ndarray::Array2::<f64>::eye(m);
                    for j in i..horizon {
                        let c = zv.dot(&tp.dot(&ps[i]).dot(&zv));
                        let c = c + if i == j { model.obs_var } else { 0.0 };
                        sigma[i][j] = c;
                        sigma[j][i] = c;
                        tp = t.dot(&tp);
                    }
                }
                for i in 0..horizon {
                    let rel = (sigma[i][i] - fc.variances[i]).abs() / fc.variances[i];
                    assert!(rel < 1e-12, "m {m} diag {i}: {rel}");
                }
                for (h, v) in got.iter().enumerate() {
                    let want: f64 = (n_gap..=n_gap + h)
                        .map(|i| (n_gap..=n_gap + h).map(|j| sigma[i][j]).sum::<f64>())
                        .sum();
                    assert!(
                        (v - want).abs() <= 1e-12 * want,
                        "m {m} gap {n_gap} h {h}: {v} vs {want}"
                    );
                }
                // More than the sum of the marginal variances (positively correlated errors).
                let marginal: f64 = fc.variances[n_gap..].iter().sum();
                assert!(got[n_post - 1] > marginal, "m {m}");
            }
        }
    }

    /// (Y - P) / P is undefined for P = 0: NaN and a warning, not 0; at any scale the
    /// interval is that of the scale-1 call.
    #[test]
    fn causal_impact_relative_effect_is_nan_when_the_counterfactual_sums_to_zero() {
        let z = z_critical(0.95).unwrap();
        for p in [0.0, -0.0, f64::NAN, f64::INFINITY] {
            let (rel, lo, hi, w) = relative_effect(1.2, p, 0.3, z);
            assert!(rel.is_nan() && lo.is_nan() && hi.is_nan(), "{p}");
            assert!(w.unwrap().contains("relative effect is not defined"));
        }
        let unit = relative_effect(1.2, 1.0, 0.3, z);
        let tiny = relative_effect(1.2e-300, 1e-300, 0.3e-300, z);
        assert!(tiny.3.is_none());
        for (a, b) in [(unit.0, tiny.0), (unit.1, tiny.1), (unit.2, tiny.2)] {
            assert!(
                a.is_finite() && (a - b).abs() <= 1e-14 * a.abs(),
                "{a} vs {b}"
            );
        }
        // The delta-method SE: |Y| / P^2 SE.
        assert!((unit.2 - unit.0 - z * 1.2 * 0.3).abs() < 1e-14);
    }

    /// y = 3 + 2c exactly in the pre-period leaves the model no variance: the message names
    /// the controls, not "the series is constant".
    #[test]
    fn causal_impact_names_a_control_fit_without_residual_variance() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(77);
        let n = 60;
        let c: Vec<f64> = (0..n).map(|_| rng.r#gen::<f64>() * 10.0).collect();
        let y: Vec<f64> = c
            .iter()
            .enumerate()
            .map(|(t, c)| {
                3.0 + 2.0 * c
                    + if t >= 40 {
                        5.0 + rng.r#gen::<f64>()
                    } else {
                        0.0
                    }
            })
            .collect();
        let time: Vec<i64> = (0..n as i64).collect();
        let d = Dataset::new(df! {"y" => &y, "time" => &time, "c" => &c}.unwrap());
        match run_causal_impact(&d, "y", "time", (0, 39), (40, 59), Some(&["c"])) {
            Err(EconError::InvalidSpecification { message }) => assert!(
                message.contains("the control regression leaves no residual variance")
                    && message.contains("drop controls that reproduce the response"),
                "{message}"
            ),
            other => panic!(
                "expected InvalidSpecification, got {:?}",
                other.map(|r| r.summary)
            ),
        }
    }

    /// A constant control has no innovation after the first step: it is aliased with the
    /// level, warned about, and its coefficient and SE are NaN.
    #[test]
    fn causal_impact_constant_control_is_aliased_with_the_level() {
        let (y, time) = generate_causal_impact_data(50, 20, 4.0, 91);
        let c = vec![7.5; y.len()];
        let d = Dataset::new(df! {"y" => &y, "time" => &time, "c" => &c}.unwrap());
        let r = run_causal_impact(&d, "y", "time", (0, 49), (50, 69), Some(&["c"])).unwrap();
        assert!(r.model.regression_coefficients.as_ref().unwrap()[0].is_nan());
        assert!(r.model.regression_std_errors.as_ref().unwrap()[0].is_nan());
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("Coefficients not defined because of singularities: c")),
            "{:?}",
            r.warnings
        );
        // The counterfactual is that of no control.
        let none = run_causal_impact(&d, "y", "time", (0, 49), (50, 69), None).unwrap();
        assert!(
            r.summary.cumulative_effect.is_finite() && none.summary.cumulative_effect.is_finite()
        );
    }

    /// The mean-only passes are the filter, and beta solves the GLS normal equations, for a
    /// level + trend + seasonal model with two controls (the R case covers the local level
    /// with one control only).
    #[test]
    fn control_gls_solves_the_innovation_normal_equations() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(4242);
        let n_pre = 40;
        let spec = StructuralSpec {
            slope: true,
            seasonal: Some(4),
            naming: CAUSAL_IMPACT_NAMING,
        };
        let model = spec.model(&[0.3, 0.01, 0.05, 1.0]).unwrap();
        let mut walk = 0.0;
        let x1: Vec<f64> = (0..n_pre)
            .map(|_| {
                walk += rng.r#gen::<f64>() - 0.5;
                5.0 + walk
            })
            .collect();
        let x2: Vec<f64> = (0..n_pre)
            .map(|t| (t as f64 * 0.7).sin() + rng.r#gen::<f64>())
            .collect();
        let y: Vec<f64> = (0..n_pre)
            .map(|t| {
                1.5 * x1[t] - 0.8 * x2[t] + (t % 4) as f64 + 0.1 * t as f64 + rng.r#gen::<f64>()
            })
            .collect();
        let (mut a0, p0, _) = spec.prior(&y).unwrap();
        a0[0] = y[0];
        let fr = kalman_filter(&y, &model, a0.view(), p0.view()).unwrap();
        let periods = PeriodIndices {
            pre_start: 0,
            pre_end: n_pre - 1,
            post_start: n_pre,
            post_end: n_pre,
            n_pre,
            n_post: 1,
            n_gap: 0,
        };
        let controls = vec![x1.clone(), x2.clone()];
        let names = vec!["x1".to_string(), "x2".to_string()];
        let mut warnings = Vec::new();
        let cs =
            control_states(&controls, &names, &fr, &model, &a0, &periods, &mut warnings).unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
        let b = &cs.beta;
        // (i) each mean-only pass is the filter of its control (same model and prior).
        for (j, x) in controls.iter().enumerate() {
            let mut ax = a0.clone();
            ax[0] = x[0];
            let fx = kalman_filter(x, &model, ax.view(), p0.view()).unwrap();
            for t in 0..n_pre {
                let scale = x[t].abs().max(1.0);
                assert!((fx.innovations[t] - cs.innovations[[t, j]]).abs() <= 1e-10 * scale);
                let zx: f64 = model
                    .observation
                    .dot(&ndarray::ArrayView1::from(&fx.filtered_states[t]));
                assert!((zx - cs.filtered_means[[t, j]]).abs() <= 1e-10 * scale);
            }
        }
        // ... so the filter of y - beta'X has the innovations v(y) - beta'v(X) ...
        let resid: Vec<f64> = (0..n_pre)
            .map(|t| y[t] - b[0] * x1[t] - b[1] * x2[t])
            .collect();
        let mut ar = a0.clone();
        ar[0] = resid[0];
        let fres = kalman_filter(&resid, &model, ar.view(), p0.view()).unwrap();
        for t in 0..n_pre {
            let want =
                fr.innovations[t] - b[0] * cs.innovations[[t, 0]] - b[1] * cs.innovations[[t, 1]];
            let scale = y[t].abs().max(1.0);
            assert!((fres.innovations[t] - want).abs() <= 1e-10 * scale, "t {t}");
        }
        // (ii) and beta solves sum_t v_t(x_j) (v_t(y) - beta'v_t(X)) / F_t = 0.
        for j in 0..2 {
            let (mut dot, mut size) = (0.0, 0.0);
            for t in 0..n_pre {
                let term = cs.innovations[[t, j]] * fres.innovations[t] / fr.innovation_vars[t];
                dot += term;
                size += term.abs();
            }
            assert!(dot.abs() <= 1e-9 * size, "{j}: {dot} vs {size}");
        }
        assert!(
            (b[0] - 1.5).abs() < 0.5 && (b[1] + 0.8).abs() < 0.5,
            "{b:?}"
        );
    }

    #[test]
    fn test_causal_impact_tail_probability_is_not_zero() {
        // Effect -25 on noise of half-width 0.5 and level steps of half-width 1.
        let (y, time) = generate_causal_impact_data(70, 30, -25.0, 42);
        let r = run_causal_impact(&ds(&y, &time), "y", "time", (0, 69), (70, 99), None).unwrap();
        assert!(r.summary.cumulative_effect < 0.0);
        let z = r.inference.expected_effect / r.inference.effect_sd;
        // Below -9, 1 - Phi(|z|) is exactly 0 in double precision; above -37,
        // Phi(z) is still a normal double.
        assert!(-37.0 < z && z < -9.0, "{z}");
        assert!(r.summary.p_value > 0.0);
        assert!(r.inference.prob_positive > 0.0);
        // Two-sided: 2 Phi(-|z|) (statrs' cdf, the reference here, is good to about 1e-10
        // relative this far out; the p-value comes from the direct tail of
        // traits::estimator).
        let want = 2.0 * Normal::standard().cdf(z);
        assert!(
            ((r.summary.p_value - want) / want).abs() < 1e-9,
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
        // One coefficient per control (no intercept): c2 is aliased with c1.
        assert!(r.model.regression_coefficients.as_ref().unwrap()[1].is_nan());
        assert!(r.model.regression_coefficients.as_ref().unwrap()[0].is_finite());
        for needle in [
            "Coefficients not defined because of singularities: c2",
            "the control regression (GLS on the filter innovations) is rank deficient",
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
        // cumulative interval from the joint forecast-error covariance covered 0.88-0.92 (the
        // sum of the marginal variances, used before: 0.35-0.46), and the two-sided p < 0.05
        // fired in 0.08-0.12 of the series (MC SE about 0.021 at 200 reps); below 0.95 because
        // the fitted variances are plugged in.
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use rand_distr::{Distribution, StandardNormal};
        let (reps, n_pre, n_post) = (200usize, 80usize, 20usize);
        let mut rng = ChaCha8Rng::seed_from_u64(20261022);
        let (mut first, mut all, mut effects) = (0usize, 0.0f64, Vec::with_capacity(reps));
        let (mut cum_covers, mut rejected) = (0usize, 0usize);
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
            cum_covers += (r.summary.cumulative_effect_lower <= 0.0
                && 0.0 <= r.summary.cumulative_effect_upper) as usize;
            rejected += r.summary.significant as usize;
            // The test and the interval agree.
            assert_eq!(
                r.summary.significant,
                !(r.summary.cumulative_effect_lower <= 0.0
                    && 0.0 <= r.summary.cumulative_effect_upper)
            );
        }
        let cum_rate = cum_covers as f64 / reps as f64;
        assert!(
            (0.84..=0.96).contains(&cum_rate),
            "cumulative coverage {cum_rate}"
        );
        let fp_rate = rejected as f64 / reps as f64;
        assert!(
            (0.02..=0.18).contains(&fp_rate),
            "false positives {fp_rate}"
        );
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
