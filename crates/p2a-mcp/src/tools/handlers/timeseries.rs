//! Time series tool handlers.
//!
//! This module defines time series tool handlers using the `#[tool_router(router = timeseries_router)]` pattern.

use rmcp::{
    ErrorData as McpError,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content},
    tool, tool_router,
};

use crate::server::AnalyticsServer;
use crate::tools::requests::timeseries::*;

use p2a_core::{
    ArConfig,
    ArMethod,
    CausalImpactConfig,
    CostFunction,
    DecomposeConfig,
    DecomposeType,
    EndRule,
    FilterMethod,
    FilterSides,
    GarchConfig,
    SeasonalType,
    StructTsConfig,
    StructTsType,
    acf_to_ar,
    ar,
    arima_sim,
    arma_acf,
    arma_to_ma,
    causal_impact,
    // Time series utilities
    cpgram,
    decompose,
    diffinv,
    embed,
    filter as ts_filter,
    forecast_arima,
    garch,
    granger_test,
    holt_winters_forecast,
    lag as ts_lag,
    run_arima,
    run_binary_segmentation,
    run_changepoint,
    run_granger_test,
    run_holt_winters,
    run_mstl,
    // Time series modeling
    run_var,
    run_var_irf,
    run_varma,
    run_vecm,
    runmed,
    stats::{
        AcfType, BoxTestType, CcfType, SpectrumConfig, run_acf, run_box_test, run_ccf, run_pacf,
        run_pp_test, run_spectrum, run_spectrum_ar,
    },
    struct_ts,
    toeplitz,
    toeplitz_asymmetric,
    toeplitz_to_vec,
    window as ts_window,
};

#[tool_router(router = timeseries_router, vis = "pub")]
impl AnalyticsServer {
    /// - R `stats::acf`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/acf.html
    #[tool(
        description = "Compute autocorrelation function (ACF), autocovariance, or partial autocorrelation function (PACF) for a time series. ACF measures correlation between observations at different lags. PACF measures correlation after removing effects of intermediate lags - useful for identifying AR order. Returns values for lags 0 to lag_max with 95% confidence bounds."
    )]
    pub async fn timeseries_acf(
        &self,
        Parameters(request): Parameters<AcfRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        // Parse ACF type
        let acf_type = match request.acf_type.as_deref() {
            Some("covariance") | Some("cov") => AcfType::Covariance,
            Some("partial") | Some("pacf") => AcfType::Partial,
            _ => AcfType::Correlation,
        };

        // For PACF, use the dedicated function
        if matches!(acf_type, AcfType::Partial) {
            match run_pacf(dataset, &request.column, request.lag_max) {
                Ok(result) => {
                    // Convert to JSON for structured output
                    let json_output = serde_json::json!({
                        "type": "Partial Autocorrelation (PACF)",
                        "series": request.column,
                        "n_obs": result.n_obs,
                        "lags": result.lags,
                        "values": result.values,
                        "confidence_bound_95": result.confidence_bound,
                        "interpretation": format!(
                            "Values outside ±{:.4} are significant at 5% level. PACF cuts off after lag p for AR(p) process.",
                            result.confidence_bound
                        )
                    });
                    Ok(CallToolResult::success(vec![Content::text(format!(
                        "{}\n\nJSON:\n{}",
                        result,
                        serde_json::to_string_pretty(&json_output).unwrap_or_default()
                    ))]))
                }
                Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                    "PACF computation failed: {}",
                    e
                ))])),
            }
        } else {
            match run_acf(dataset, &request.column, request.lag_max, acf_type) {
                Ok(result) => {
                    let type_str = match acf_type {
                        AcfType::Correlation => "Autocorrelation (ACF)",
                        AcfType::Covariance => "Autocovariance (ACVF)",
                        AcfType::Partial => "Partial Autocorrelation (PACF)",
                    };
                    let json_output = serde_json::json!({
                        "type": type_str,
                        "series": request.column,
                        "n_obs": result.n_obs,
                        "lags": result.lags,
                        "values": result.values,
                        "confidence_bound_95": result.confidence_bound,
                        "interpretation": if matches!(acf_type, AcfType::Correlation) {
                            format!(
                                "Values outside ±{:.4} are significant at 5% level. ACF tails off for AR, cuts off for MA.",
                                result.confidence_bound.unwrap_or(0.0)
                            )
                        } else {
                            "Autocovariance values (not normalized). Divide by ACVF(0) to get ACF.".to_string()
                        }
                    });
                    Ok(CallToolResult::success(vec![Content::text(format!(
                        "{}\n\nJSON:\n{}",
                        result,
                        serde_json::to_string_pretty(&json_output).unwrap_or_default()
                    ))]))
                }
                Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                    "ACF computation failed: {}",
                    e
                ))])),
            }
        }
    }

    /// - R `stats::ccf`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/acf.html
    #[tool(
        description = "Compute cross-correlation function (CCF) between two time series. CCF at lag k estimates correlation between x_{t+k} and y_t. Positive lag k means x leads y; negative lag means y leads x. Useful for identifying lead-lag relationships between variables."
    )]
    pub async fn timeseries_ccf(
        &self,
        Parameters(request): Parameters<CcfRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        match run_ccf(
            dataset,
            &request.x,
            &request.y,
            request.lag_max,
            CcfType::Correlation,
        ) {
            Ok(result) => {
                // Find the lag with maximum absolute correlation
                let max_idx = result
                    .values
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let max_lag = result.lags[max_idx];
                let max_ccf = result.values[max_idx];

                let lead_lag_interpretation = if max_lag > 0 {
                    format!("{} leads {} by {} periods", request.x, request.y, max_lag)
                } else if max_lag < 0 {
                    format!("{} leads {} by {} periods", request.y, request.x, -max_lag)
                } else {
                    format!(
                        "{} and {} are contemporaneously correlated",
                        request.x, request.y
                    )
                };

                let json_output = serde_json::json!({
                    "type": "Cross-Correlation (CCF)",
                    "x_series": request.x,
                    "y_series": request.y,
                    "n_obs": result.n_obs,
                    "lags": result.lags,
                    "values": result.values,
                    "confidence_bound_95": result.confidence_bound,
                    "max_correlation": {
                        "lag": max_lag,
                        "value": max_ccf,
                        "interpretation": lead_lag_interpretation
                    }
                });
                Ok(CallToolResult::success(vec![Content::text(format!(
                    "{}\n\nJSON:\n{}",
                    result,
                    serde_json::to_string_pretty(&json_output).unwrap_or_default()
                ))]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "CCF computation failed: {}",
                e
            ))])),
        }
    }

    /// - R `stats::spectrum`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/spectrum.html
    #[tool(
        description = "Estimate spectral density (power spectrum) of a time series. Returns spectral density at Fourier frequencies showing how variance is distributed across frequency components. Methods: 'pgram' (periodogram with optional smoothing) or 'ar' (AR model-based). Use spans parameter for smoothing raw periodogram (e.g., spans=[3,3] for moderate smoothing). Peak frequency reveals dominant cyclical patterns."
    )]
    pub async fn timeseries_spectrum(
        &self,
        Parameters(request): Parameters<SpectrumRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        // Determine method
        let method = request.method.as_deref().unwrap_or("pgram");

        let result = match method.to_lowercase().as_str() {
            "ar" | "autoregressive" => {
                match run_spectrum_ar(dataset, &request.column, request.ar_order, None) {
                    Ok(r) => r,
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "AR spectral estimation failed: {}",
                            e
                        ))]));
                    }
                }
            }
            _ => {
                // Periodogram method
                let config = SpectrumConfig {
                    spans: request.spans.clone(),
                    taper: request.taper.unwrap_or(0.1),
                    detrend: request.detrend.unwrap_or(true),
                    demean: false,
                    pad_ratio: 0.0,
                };

                match run_spectrum(dataset, &request.column, config) {
                    Ok(r) => r,
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Spectral estimation failed: {}",
                            e
                        ))]));
                    }
                }
            }
        };

        // Get peak frequency info
        let (peak_freq, peak_spec) = result.peak_frequency().unwrap_or((0.0, 0.0));
        let peak_period = if peak_freq > 0.0 {
            1.0 / peak_freq
        } else {
            f64::INFINITY
        };

        // Compute confidence interval multipliers
        let (ci_lower_mult, ci_upper_mult) = result.confidence_multipliers(0.95);

        let json_output = serde_json::json!({
            "type": format!("Spectral Density ({})", result.method),
            "series": request.column,
            "n_obs": result.n_obs,
            "n_used": result.n_used,
            "bandwidth": result.bandwidth,
            "degrees_of_freedom": result.df,
            "taper": result.taper,
            "detrend": result.detrend,
            "spans": result.kernel_spans,
            "n_frequencies": result.freq.len(),
            "frequency_range": [
                result.freq.first().copied().unwrap_or(0.0),
                result.freq.last().copied().unwrap_or(0.5)
            ],
            "peak": {
                "frequency": peak_freq,
                "period": peak_period,
                "spectral_density": peak_spec,
                "interpretation": format!(
                    "Dominant cycle at frequency {:.4} (period = {:.1} time units)",
                    peak_freq, peak_period
                )
            },
            "confidence_interval_95": {
                "multiplier_lower": ci_lower_mult,
                "multiplier_upper": ci_upper_mult,
                "interpretation": "Multiply spectral estimate by these values for 95% CI"
            },
            // Include first/last few frequency-spectrum pairs
            "spectrum_sample": {
                "first_5": result.freq.iter().take(5).zip(result.spec.iter().take(5))
                    .map(|(&f, &s)| serde_json::json!({"freq": f, "spec": s}))
                    .collect::<Vec<_>>(),
                "last_5": result.freq.iter().rev().take(5).rev().zip(result.spec.iter().rev().take(5).rev())
                    .map(|(&f, &s)| serde_json::json!({"freq": f, "spec": s}))
                    .collect::<Vec<_>>()
            }
        });

        Ok(CallToolResult::success(vec![Content::text(format!(
            "{}\n\nJSON:\n{}",
            result,
            serde_json::to_string_pretty(&json_output).unwrap_or_default()
        ))]))
    }

    /// - Ljung, G. M. & Box, G. E. P. (1978). "On a measure of lack of fit in time series models." Biometrika, 65, 297-303.
    #[tool(
        description = "Test for autocorrelation in a time series using Box-Pierce or Ljung-Box test. Tests H₀: no autocorrelation up to specified lag. Commonly used to check whether ARIMA residuals are white noise. Ljung-Box (default) has better finite-sample properties. For ARMA(p,q) residuals, set fitdf=p+q to adjust degrees of freedom. Returns: X-squared statistic, df, p-value, and sample autocorrelations."
    )]
    pub async fn timeseries_box_test(
        &self,
        Parameters(request): Parameters<BoxTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        // Parse test type
        let test_type = match request.test_type.as_deref() {
            Some("box-pierce") | Some("Box-Pierce") | Some("boxpierce") => BoxTestType::BoxPierce,
            _ => BoxTestType::LjungBox, // Default
        };

        let fitdf = request.fitdf.unwrap_or(0);

        let result = match run_box_test(dataset, &request.column, request.lag, test_type, fitdf) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Box test failed: {}",
                    e
                ))]));
            }
        };

        let json_output = serde_json::json!({
            "test_type": result.test_type.to_string(),
            "series": request.column,
            "statistic": result.statistic,
            "df": result.df,
            "p_value": result.p_value,
            "significance": result.significance.to_string(),
            "n_obs": result.n_obs,
            "lag": result.lag,
            "fitdf": result.fitdf,
            "autocorrelations": result.autocorrelations,
            "interpretation": if result.p_value < 0.05 {
                format!(
                    "Reject H₀ (p={:.4}): Significant autocorrelation detected at the 5% level. The series is not white noise.",
                    result.p_value
                )
            } else {
                format!(
                    "Fail to reject H₀ (p={:.4}): No significant autocorrelation detected. The series is consistent with white noise.",
                    result.p_value
                )
            }
        });

        Ok(CallToolResult::success(vec![Content::text(format!(
            "{}\n\nJSON:\n{}",
            result,
            serde_json::to_string_pretty(&json_output).unwrap_or_default()
        ))]))
    }

    /// - Banerjee, A., Dolado, J. J., Galbraith, J. W., & Hendry, D. (1993). Co-integration, Error Correction, and the Econometric Analysis of Non-Stationary Data. Oxford University Press.
    #[tool(
        description = "Test for unit root in a time series using the Phillips-Perron test. Tests H₀: series has unit root (non-stationary) vs H₁: series is stationary. Uses Newey-West long-run variance estimator with Bartlett weights for serial correlation correction. Similar to ADF test but makes non-parametric correction. Returns: Z(τ) statistic, truncation lag, p-value, and diagnostics."
    )]
    pub async fn timeseries_pp_test(
        &self,
        Parameters(request): Parameters<PPTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let lshort = request.lshort.unwrap_or(true);

        let result = match run_pp_test(dataset, &request.column, lshort) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Phillips-Perron test failed: {}",
                    e
                ))]));
            }
        };

        let json_output = serde_json::json!({
            "test": "Phillips-Perron",
            "series": request.column,
            "statistic": result.statistic,
            "truncation_lag": result.truncation_lag,
            "p_value": result.p_value,
            "significance": result.significance.to_string(),
            "n_obs": result.n_obs,
            "lshort": result.lshort,
            "gamma_hat": result.gamma_hat,
            "t_statistic": result.t_statistic,
            "sigma_squared": result.sigma_squared,
            "lambda_squared": result.lambda_squared,
            "interpretation": if result.p_value < 0.05 {
                format!(
                    "Reject H₀ (p={:.4}): Evidence of stationarity at the 5% level. The series likely does not have a unit root.",
                    result.p_value
                )
            } else {
                format!(
                    "Fail to reject H₀ (p={:.4}): No significant evidence against unit root. The series may be non-stationary.",
                    result.p_value
                )
            },
            "recommendation": if result.p_value < 0.05 {
                "Series appears stationary. Can proceed with standard time series methods."
            } else {
                "Series may have unit root. Consider differencing (Δy = y_t - y_{t-1}) before analysis."
            }
        });

        Ok(CallToolResult::success(vec![Content::text(format!(
            "{}\n\nJSON:\n{}",
            result,
            serde_json::to_string_pretty(&json_output).unwrap_or_default()
        ))]))
    }

    /// Run VAR (Vector Autoregression) model.
    #[tool(
        description = "Run Vector Autoregression (VAR) model for multivariate time series. Returns coefficients, residual covariance, AIC, and BIC."
    )]
    pub async fn ts_var(
        &self,
        Parameters(request): Parameters<VarRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let columns: Vec<&str> = request.columns.iter().map(|s| s.as_str()).collect();

        let result = match run_var(dataset, &columns, request.lags) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "VAR estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Granger causality test.
    #[tool(
        description = "Test Granger causality between two time series variables. Tests whether lagged values of 'cause' help predict 'dependent' after controlling for lagged 'dependent'. Uses F-test comparing restricted (y lags only) vs unrestricted (y and x lags) models."
    )]
    pub async fn ts_granger(
        &self,
        Parameters(request): Parameters<GrangerRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let result = match request.lags {
            Some(lags) => granger_test(dataset, &request.dependent, &request.cause, lags),
            None => run_granger_test(dataset, &request.dependent, &request.cause),
        };

        let result = match result {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Granger causality test failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run VARMA (Vector ARMA) model.
    #[tool(
        description = "Run VARMA(p,q) model using Hannan-Rissanen estimation. Combines autoregressive and moving average components for multivariate time series."
    )]
    pub async fn ts_varma(
        &self,
        Parameters(request): Parameters<VarmaRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let columns: Vec<&str> = request.columns.iter().map(|s| s.as_str()).collect();

        let result = match run_varma(dataset, &columns, request.p, request.q) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "VARMA estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run VECM (Vector Error Correction Model).
    #[tool(
        description = "Run VECM using Johansen Maximum Likelihood. For cointegrated I(1) time series. Returns cointegration vectors (beta), adjustment speeds (alpha), and eigenvalues."
    )]
    pub async fn ts_vecm(
        &self,
        Parameters(request): Parameters<VecmRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let columns: Vec<&str> = request.columns.iter().map(|s| s.as_str()).collect();

        let result = match run_vecm(dataset, &columns, request.lags, request.rank) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "VECM estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Compute VAR Impulse Response Functions.
    #[tool(
        description = "Compute Impulse Response Functions (IRF) from a VAR model. Shows how variables respond to shocks over time using Cholesky orthogonalization."
    )]
    pub async fn ts_var_irf(
        &self,
        Parameters(request): Parameters<VarIrfRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let columns: Vec<&str> = request.columns.iter().map(|s| s.as_str()).collect();

        let result = match run_var_irf(dataset, &columns, request.lags, request.steps) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "VAR IRF computation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Fit an ARIMA model.
    #[tool(
        description = "Fit an ARIMA(p,d,q) model to a univariate time series. Returns AR/MA coefficients, residuals, AIC, and model diagnostics."
    )]
    pub async fn ts_arima_fit(
        &self,
        Parameters(request): Parameters<ArimaRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let result = match run_arima(dataset, &request.column, request.p, request.d, request.q) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "ARIMA fitting failed: {}",
                    e
                ))]));
            }
        };

        // Format result
        let output = format!(
            "ARIMA({},{},{}) Model Results\n\
             ==============================\n\
             Column: {}\n\
             Observations: {}\n\n\
             AR Coefficients (phi): {:?}\n\
             MA Coefficients (theta): {:?}\n\
             Intercept: {:.6}\n\n\
             Sum of Squared Residuals: {:.4}\n\
             AIC: {:.4}",
            result.p,
            result.d,
            result.q,
            result.column,
            result.n_obs,
            result.ar_coeffs,
            result.ma_coeffs,
            result.intercept,
            result.ssr,
            result.aic
        );

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    /// Forecast using an ARIMA model.
    #[tool(
        description = "Forecast future values using an ARIMA(p,d,q) model. Fits the model and generates h-step ahead forecasts."
    )]
    pub async fn ts_arima_forecast(
        &self,
        Parameters(request): Parameters<ArimaForecastRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let result = match forecast_arima(
            dataset,
            &request.column,
            request.p,
            request.d,
            request.q,
            request.horizon,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "ARIMA forecasting failed: {}",
                    e
                ))]));
            }
        };

        // Format result
        let mut output = format!(
            "ARIMA Forecast Results\n\
             ======================\n\
             Column: {}\n\
             Horizon: {} periods\n\n\
             Forecasted Values:\n",
            result.column, result.horizon
        );

        for (i, val) in result.forecast.iter().enumerate() {
            output.push_str(&format!("  t+{}: {:.4}\n", i + 1, val));
        }

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    /// Fit a GARCH model for volatility modeling.
    #[tool(
        description = "Fit a GARCH(p,q) model for time-varying volatility. Estimates the conditional variance equation: σ²_t = ω + Σα_i ε²_{t-i} + Σβ_j σ²_{t-j}. Reports persistence, unconditional variance, and half-life of volatility shocks."
    )]
    pub async fn ts_garch_fit(
        &self,
        Parameters(request): Parameters<GarchRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        // Extract column data
        let col = dataset
            .df()
            .column(&request.column)
            .map_err(|e| McpError::invalid_request(format!("Column error: {}", e), None))?;

        let data: Vec<f64> = col
            .f64()
            .map_err(|e| McpError::invalid_request(format!("Column must be numeric: {}", e), None))?
            .into_no_null_iter()
            .collect();

        let config = GarchConfig {
            p: request.p.unwrap_or(1),
            q: request.q.unwrap_or(1),
            include_mean: request.include_mean.unwrap_or(true),
            ..Default::default()
        };

        let result = match garch(&data, Some(config)) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "GARCH fitting failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run MSTL decomposition.
    #[tool(
        description = "Perform MSTL (Multiple Seasonal-Trend decomposition using LOESS) on a time series. Extracts trend, seasonal components, and residuals."
    )]
    pub async fn ts_mstl(
        &self,
        Parameters(request): Parameters<MstlRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let result = match run_mstl(dataset, &request.column, &request.periods) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "MSTL decomposition failed: {}",
                    e
                ))]));
            }
        };

        // Format result with summary statistics
        let trend_mean: f64 = result.trend.iter().sum::<f64>() / result.trend.len() as f64;
        let resid_var: f64 =
            result.residuals.iter().map(|r| r * r).sum::<f64>() / result.residuals.len() as f64;

        let mut output = format!(
            "MSTL Decomposition Results\n\
             ==========================\n\
             Column: {}\n\
             Observations: {}\n\
             Seasonal Periods: {:?}\n\n\
             Component Statistics:\n\
             - Trend mean: {:.4}\n\
             - Residual variance: {:.4}\n",
            result.column, result.n_obs, result.periods, trend_mean, resid_var
        );

        // Show first few values of each component
        let show_n = 5.min(result.n_obs);
        output.push_str(&format!("\nFirst {} values:\n", show_n));
        output.push_str("  Trend: [");
        for (i, val) in result.trend.iter().take(show_n).enumerate() {
            if i > 0 {
                output.push_str(", ");
            }
            output.push_str(&format!("{:.2}", val));
        }
        output.push_str("]\n");

        for (idx, seasonal) in result.seasonal.iter().enumerate() {
            output.push_str(&format!("  Seasonal (period {}): [", result.periods[idx]));
            for (i, val) in seasonal.iter().take(show_n).enumerate() {
                if i > 0 {
                    output.push_str(", ");
                }
                output.push_str(&format!("{:.2}", val));
            }
            output.push_str("]\n");
        }

        output.push_str("  Residuals: [");
        for (i, val) in result.residuals.iter().take(show_n).enumerate() {
            if i > 0 {
                output.push_str(", ");
            }
            output.push_str(&format!("{:.2}", val));
        }
        output.push_str("]\n");

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    /// Detect changepoints (structural breaks) in a time series.
    #[tool(
        description = "Detect changepoints (structural breaks) in a time series using PELT or Binary Segmentation. Identifies points where the statistical properties (mean, variance) change significantly. Useful for regime detection, anomaly detection, and segmenting time series into homogeneous periods."
    )]
    pub async fn ts_changepoint(
        &self,
        Parameters(request): Parameters<ChangepointRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        // Determine cost function
        let cost_fn = match request.change_type.as_deref() {
            Some("variance") => CostFunction::VarianceChange,
            Some("both") => CostFunction::MeanAndVariance,
            _ => CostFunction::MeanChange,
        };

        // Run detection based on method
        let result = match request.method.as_deref() {
            Some("binary") => {
                match run_binary_segmentation(
                    dataset,
                    &request.column,
                    Some(10), // max changepoints
                    request.min_segment_length,
                    request.penalty,
                ) {
                    Ok(r) => r,
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Changepoint detection failed: {}",
                            e
                        ))]));
                    }
                }
            }
            _ => {
                match run_changepoint(
                    dataset,
                    &request.column,
                    request.penalty,
                    request.min_segment_length,
                    cost_fn,
                ) {
                    Ok(r) => r,
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Changepoint detection failed: {}",
                            e
                        ))]));
                    }
                }
            }
        };

        // Get observation count from result
        let n_obs: usize = result.segments.iter().map(|s| s.n_points).sum();

        // Format output
        let mut output = format!(
            "Changepoint Detection Results\n\
             ==============================\n\
             Column: {}\n\
             Observations: {}\n\
             Method: {}\n\
             Penalty: {:.4}\n\n\
             Changepoints Detected: {}\n",
            request.column, n_obs, result.method, result.penalty, result.n_changepoints,
        );

        if result.n_changepoints > 0 {
            output.push_str(&format!(
                "Changepoint Positions: {:?}\n\n",
                result.changepoints
            ));
        } else {
            output.push_str("\nNo changepoints detected (series appears stationary).\n\n");
        }

        output.push_str("Segment Statistics:\n");
        for (i, seg) in result.segments.iter().enumerate() {
            output.push_str(&format!(
                "  Segment {}: indices [{}, {}) | n={} | mean={:.4} | variance={:.4}\n",
                i + 1,
                seg.start,
                seg.end,
                seg.n_points,
                seg.mean,
                seg.variance
            ));
        }

        output.push_str(&format!("\nTotal Cost: {:.4}\n", result.total_cost));

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    ///   OTexts. https://otexts.com/fpp3/
    #[tool(
        description = "Fit Holt-Winters exponential smoothing model to a time series with trend and seasonality. Supports both additive (constant seasonal variation) and multiplicative (proportional variation) seasonality. Automatically optimizes smoothing parameters (alpha, beta, gamma) if not provided. Can generate forecasts for future periods. Returns fitted values, residuals, optimized parameters, and seasonal coefficients."
    )]
    pub async fn ts_holt_winters(
        &self,
        Parameters(request): Parameters<HoltWintersRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        // Parse seasonal type
        let seasonal = match request.seasonal.as_deref() {
            Some("multiplicative") | Some("Multiplicative") | Some("mult") => {
                SeasonalType::Multiplicative
            }
            _ => SeasonalType::Additive,
        };

        // Run Holt-Winters
        let result = match run_holt_winters(
            dataset,
            &request.column,
            request.period,
            seasonal,
            request.alpha,
            request.beta,
            request.gamma,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Holt-Winters fitting failed: {}",
                    e
                ))]));
            }
        };

        // Format output
        let seasonal_type_str = match result.seasonal_type {
            SeasonalType::Additive => "Additive",
            SeasonalType::Multiplicative => "Multiplicative",
        };

        let mut output = format!(
            "Holt-Winters Exponential Smoothing Results\n\
             ==========================================\n\
             Column: {}\n\
             Observations: {}\n\
             Seasonal Period: {}\n\
             Seasonal Type: {}\n\n\
             Optimized Smoothing Parameters:\n\
             - Alpha (level): {:.6}\n\
             - Beta (trend): {}\n\
             - Gamma (seasonal): {}\n\n\
             Sum of Squared Errors: {:.4}\n\n\
             Final Coefficients:\n\
             - Level: {:.6}\n\
             - Trend: {}\n",
            result.column.as_deref().unwrap_or(&request.column),
            result.n_obs,
            result.period,
            seasonal_type_str,
            result.alpha,
            result
                .beta
                .map_or("N/A".to_string(), |v| format!("{:.6}", v)),
            result
                .gamma
                .map_or("N/A".to_string(), |v| format!("{:.6}", v)),
            result.sse,
            result.coefficients.level,
            result
                .coefficients
                .trend
                .map_or("N/A".to_string(), |v| format!("{:.6}", v)),
        );

        // Add seasonal coefficients
        if let Some(ref seasonal_coeffs) = result.coefficients.seasonal {
            output.push_str(&format!("- Seasonal coefficients: {:?}\n", seasonal_coeffs));
        }

        // If forecast horizon was requested, generate forecasts
        if let Some(horizon) = request.horizon {
            if horizon > 0 {
                match holt_winters_forecast(&result, horizon) {
                    Ok(forecasts) => {
                        output.push_str(&format!("\nForecasts ({} periods ahead):\n", horizon));
                        for (i, val) in forecasts.iter().enumerate() {
                            output.push_str(&format!("  t+{}: {:.4}\n", i + 1, val));
                        }
                    }
                    Err(e) => {
                        output.push_str(&format!("\nForecast generation failed: {}\n", e));
                    }
                }
            }
        }

        // Show first few fitted values and residuals
        let show_n = 5.min(result.n_obs);
        output.push_str(&format!("\nFirst {} values:\n", show_n));
        output.push_str("  Fitted: [");
        for (i, val) in result.fitted.iter().take(show_n).enumerate() {
            if i > 0 {
                output.push_str(", ");
            }
            output.push_str(&format!("{:.4}", val));
        }
        output.push_str("...]\n");

        output.push_str("  Residuals: [");
        for (i, val) in result.residuals.iter().take(show_n).enumerate() {
            if i > 0 {
                output.push_str(", ");
            }
            output.push_str(&format!("{:.4}", val));
        }
        output.push_str("...]\n");

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    /// Fit an autoregressive (AR) model to time series data.
    #[tool(
        description = "Fit an autoregressive model to time series data with automatic order selection via AIC. Supports Yule-Walker (default), Burg, and OLS methods. Returns AR coefficients, prediction variance, AIC values, partial autocorrelations, and residuals."
    )]
    pub async fn timeseries_ar(
        &self,
        Parameters(request): Parameters<ArModelRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        // Extract time series column
        let column = dataset.df().column(&request.column).map_err(|e| {
            McpError::invalid_request(
                format!("Column '{}' not found: {}", request.column, e),
                None,
            )
        })?;

        let x: Vec<f64> = column
            .f64()
            .map_err(|e| McpError::invalid_request(format!("Column must be numeric: {}", e), None))?
            .into_no_null_iter()
            .collect();

        // Parse method
        let method = match request.method.as_deref() {
            Some("burg") | Some("Burg") => ArMethod::Burg,
            Some("ols") | Some("OLS") => ArMethod::Ols,
            _ => ArMethod::YuleWalker,
        };

        // Build config
        let config = ArConfig {
            aic: request.aic.unwrap_or(true),
            order_max: request.order_max,
            order: request.order,
            method,
            demean: request.demean.unwrap_or(true),
        };

        // Fit AR model
        let result = match ar(&x, config) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "AR model fitting failed: {}",
                    e
                ))]));
            }
        };

        // Build output
        let method_str = match result.method {
            ArMethod::YuleWalker => "Yule-Walker",
            ArMethod::Burg => "Burg",
            ArMethod::Ols => "OLS",
        };

        let output = serde_json::json!({
            "method": format!("AR({}) via {}", result.order, method_str),
            "order": result.order,
            "ar_coefficients": result.ar,
            "prediction_variance": result.var_pred,
            "x_mean": result.x_mean,
            "n_obs": result.n_obs,
            "partial_acf": result.partial_acf,
            "aic_relative": result.aic,
            "interpretation": {
                "order_selected": if request.aic.unwrap_or(true) {
                    format!("Order {} selected by AIC minimization", result.order)
                } else {
                    format!("Order {} specified by user", result.order)
                },
                "coefficients": format!(
                    "AR model: x_t = {:.4} + {} + ε_t",
                    result.x_mean,
                    result.ar.iter().enumerate()
                        .map(|(i, c)| format!("{:.4}*x_{{t-{}}}", c, i + 1))
                        .collect::<Vec<_>>().join(" + ")
                )
            },
            "references": "Brockwell & Davis (1991), Time Series: Theory and Methods"
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| format!("{:?}", result)),
        )]))
    }

    /// Perform classical seasonal decomposition by moving averages.
    #[tool(
        description = "Decompose a time series into trend, seasonal, and random components using moving averages. Implements R's decompose() function. Supports additive (Y = T + S + R) and multiplicative (Y = T × S × R) decomposition."
    )]
    pub async fn timeseries_decompose(
        &self,
        Parameters(request): Parameters<DecomposeRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        // Extract time series column
        let column = dataset.df().column(&request.column).map_err(|e| {
            McpError::invalid_request(
                format!("Column '{}' not found: {}", request.column, e),
                None,
            )
        })?;

        let x: Vec<f64> = column
            .f64()
            .map_err(|e| McpError::invalid_request(format!("Column must be numeric: {}", e), None))?
            .into_no_null_iter()
            .collect();

        // Parse decomposition type
        let decompose_type = match request.decompose_type.as_deref() {
            Some("multiplicative") | Some("mult") => DecomposeType::Multiplicative,
            _ => DecomposeType::Additive,
        };

        // Run decomposition
        let result = match decompose(
            &x,
            request.period,
            DecomposeConfig {
                decompose_type,
                filter: None,
            },
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Decomposition failed: {}",
                    e
                ))]));
            }
        };

        // Calculate statistics for output
        let valid_trend: Vec<f64> = result
            .trend
            .iter()
            .filter(|t| !t.is_nan())
            .copied()
            .collect();
        let valid_random: Vec<f64> = result
            .random
            .iter()
            .filter(|r| !r.is_nan())
            .copied()
            .collect();

        let trend_mean = if valid_trend.is_empty() {
            f64::NAN
        } else {
            valid_trend.iter().sum::<f64>() / valid_trend.len() as f64
        };

        let random_var = if valid_random.is_empty() {
            f64::NAN
        } else {
            let mean = valid_random.iter().sum::<f64>() / valid_random.len() as f64;
            valid_random.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / valid_random.len() as f64
        };

        // Build JSON output
        let output = serde_json::json!({
            "method": "Classical Decomposition by Moving Averages",
            "decompose_type": match decompose_type {
                DecomposeType::Additive => "additive",
                DecomposeType::Multiplicative => "multiplicative",
            },
            "period": result.period,
            "n_obs": result.n_obs,
            "summary": {
                "trend_mean": format!("{:.4}", trend_mean),
                "seasonal_range": format!("[{:.4}, {:.4}]",
                    result.figure.iter().cloned().fold(f64::INFINITY, f64::min),
                    result.figure.iter().cloned().fold(f64::NEG_INFINITY, f64::max)),
                "random_variance": format!("{:.4}", random_var),
            },
            "seasonal_figure": result.figure.iter()
                .enumerate()
                .map(|(i, &v)| serde_json::json!({
                    "period_index": i + 1,
                    "value": format!("{:.4}", v)
                }))
                .collect::<Vec<_>>(),
            "components_sample": {
                "first_5_trend": result.trend.iter().take(5)
                    .map(|&v| if v.is_nan() { "NA".to_string() } else { format!("{:.4}", v) })
                    .collect::<Vec<_>>(),
                "first_5_seasonal": result.seasonal.iter().take(5)
                    .map(|&v| format!("{:.4}", v))
                    .collect::<Vec<_>>(),
                "first_5_random": result.random.iter().take(5)
                    .map(|&v| if v.is_nan() { "NA".to_string() } else { format!("{:.4}", v) })
                    .collect::<Vec<_>>(),
            },
            "interpretation": match decompose_type {
                DecomposeType::Additive => "Additive model: Original = Trend + Seasonal + Random. Seasonal effects are constant over time.",
                DecomposeType::Multiplicative => "Multiplicative model: Original = Trend × Seasonal × Random. Seasonal effects are proportional to the level.",
            },
            "notes": "Trend component has NA values at the boundaries (first and last period/2 observations).",
            "references": "Kendall, M. (1976). Time Series. Charles Griffin."
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| format!("{:?}", result)),
        )]))
    }

    /// Fit a structural time series model.
    #[tool(
        description = "Fit a structural time series model by maximum likelihood using the Kalman filter. Supports: 'level' (local level/random walk + noise, equivalent to ARIMA(0,1,1)), 'trend' (local linear trend, equivalent to ARIMA(0,2,2)), and 'bsm' (basic structural model with level, trend, and seasonality). Returns variance parameters, smoothed components, fitted values, residuals, log-likelihood, AIC, and BIC."
    )]
    pub async fn timeseries_structts(
        &self,
        Parameters(request): Parameters<StructTsRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        // Extract time series column
        let column = dataset.df().column(&request.column).map_err(|e| {
            McpError::invalid_request(
                format!("Column '{}' not found: {}", request.column, e),
                None,
            )
        })?;

        let y: Vec<f64> = column
            .f64()
            .map_err(|e| McpError::invalid_request(format!("Column must be numeric: {}", e), None))?
            .into_no_null_iter()
            .collect();

        // Parse model type
        let model_type = match request.model_type.as_deref() {
            Some("trend") | Some("Trend") => StructTsType::Trend,
            Some("bsm") | Some("BSM") => StructTsType::BSM,
            _ => StructTsType::Level,
        };

        // Validate period for BSM
        if model_type == StructTsType::BSM && request.period.is_none() {
            return Ok(CallToolResult::error(vec![Content::text(
                "BSM model requires a period parameter (e.g., 12 for monthly data)",
            )]));
        }

        // Run StructTS
        let result = match struct_ts(
            &y,
            StructTsConfig {
                model_type,
                period: request.period,
                ..Default::default()
            },
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "StructTS fitting failed: {}",
                    e
                ))]));
            }
        };

        // Build JSON output
        let mut coef_json = serde_json::json!({
            "level_variance": format!("{:.6}", result.coef.level),
            "observation_variance": format!("{:.6}", result.coef.epsilon),
        });

        if let Some(slope_var) = result.coef.slope {
            coef_json["slope_variance"] = serde_json::json!(format!("{:.6}", slope_var));
        }
        if let Some(seasonal_var) = result.coef.seasonal {
            coef_json["seasonal_variance"] = serde_json::json!(format!("{:.6}", seasonal_var));
        }

        let model_type_str = match result.model_type {
            StructTsType::Level => "level",
            StructTsType::Trend => "trend",
            StructTsType::BSM => "bsm",
        };

        let output = serde_json::json!({
            "method": "Structural Time Series Model",
            "model_type": model_type_str,
            "n_obs": result.n_obs,
            "n_params": result.n_params,
            "converged": result.converged,
            "variance_coefficients": coef_json,
            "fit_statistics": {
                "log_likelihood": format!("{:.4}", result.log_likelihood),
                "aic": format!("{:.4}", result.aic),
                "bic": format!("{:.4}", result.bic),
            },
            "components_sample": {
                "first_5_level": result.level.iter().take(5)
                    .map(|&v| format!("{:.4}", v))
                    .collect::<Vec<_>>(),
                "first_5_slope": result.slope.as_ref().map(|s|
                    s.iter().take(5).map(|&v| format!("{:.4}", v)).collect::<Vec<_>>()
                ),
                "first_5_seasonal": result.seasonal.as_ref().map(|s|
                    s.iter().take(5).map(|&v| format!("{:.4}", v)).collect::<Vec<_>>()
                ),
            },
            "residual_summary": {
                "mean": format!("{:.4}", result.residuals.iter().sum::<f64>() / result.n_obs as f64),
                "std": format!("{:.4}", {
                    let mean = result.residuals.iter().sum::<f64>() / result.n_obs as f64;
                    (result.residuals.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (result.n_obs - 1) as f64).sqrt()
                }),
            },
            "interpretation": match result.model_type {
                StructTsType::Level => "Local level model: Y_t = μ_t + ε_t where μ_t follows a random walk. Equivalent to ARIMA(0,1,1).",
                StructTsType::Trend => "Local linear trend model: Y_t = μ_t + ε_t where μ_t has time-varying level and slope. Equivalent to ARIMA(0,2,2).",
                StructTsType::BSM => "Basic Structural Model: Y_t = μ_t + γ_t + ε_t with level, slope, and seasonal components.",
            },
            "references": "Harvey, A. C. (1990). Forecasting, Structural Time Series Models and the Kalman Filter."
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| format!("{:?}", result)),
        )]))
    }

    /// CausalImpact analysis using Bayesian Structural Time Series.
    #[tool(
        description = "Estimate the causal effect of an intervention using Bayesian Structural Time Series (BSTS). Uses pre-intervention data to build a counterfactual prediction, then compares with observed post-intervention data to estimate the causal effect. Optionally uses control time series that are correlated with the response but unaffected by the intervention. Returns cumulative/average effects with credible intervals and Bayesian p-value."
    )]
    pub async fn causal_impact_analysis(
        &self,
        Parameters(request): Parameters<CausalImpactRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        // Build config
        let config = CausalImpactConfig {
            pre_period: (request.pre_period_start, request.pre_period_end),
            post_period: (request.post_period_start, request.post_period_end),
            control_series: request.control_cols,
            alpha: request.alpha.unwrap_or(0.05),
            seasonal_period: request.seasonal_period,
            include_trend: request.include_trend.unwrap_or(false),
            ..Default::default()
        };

        // Run CausalImpact
        let result = match causal_impact(dataset, &request.response_col, &request.time_col, config)
        {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "CausalImpact analysis failed: {}",
                    e
                ))]));
            }
        };

        // Build output
        let output = serde_json::json!({
            "method": "CausalImpact (Bayesian Structural Time Series)",
            "periods": {
                "pre_period": {
                    "start": request.pre_period_start,
                    "end": request.pre_period_end,
                    "n_obs": result.model.n_pre,
                },
                "post_period": {
                    "start": request.post_period_start,
                    "end": request.post_period_end,
                    "n_obs": result.model.n_post,
                },
            },
            "summary": {
                "average_effect": {
                    "estimate": format!("{:.4}", result.summary.average_effect),
                    "ci_lower": format!("{:.4}", result.summary.average_effect_lower),
                    "ci_upper": format!("{:.4}", result.summary.average_effect_upper),
                },
                "cumulative_effect": {
                    "estimate": format!("{:.4}", result.summary.cumulative_effect),
                    "ci_lower": format!("{:.4}", result.summary.cumulative_effect_lower),
                    "ci_upper": format!("{:.4}", result.summary.cumulative_effect_upper),
                },
                "relative_effect": {
                    "estimate": format!("{:.2}%", result.summary.relative_effect * 100.0),
                    "ci_lower": format!("{:.2}%", result.summary.relative_effect_lower * 100.0),
                    "ci_upper": format!("{:.2}%", result.summary.relative_effect_upper * 100.0),
                },
                "p_value": format!("{:.4}", result.summary.p_value),
                "significant": result.summary.significant,
                "alpha": result.summary.alpha,
            },
            "inference": {
                "prob_positive_effect": format!("{:.4}", result.inference.prob_positive),
                "prob_negative_effect": format!("{:.4}", result.inference.prob_negative),
                "expected_effect": format!("{:.4}", result.inference.expected_effect),
                "effect_sd": format!("{:.4}", result.inference.effect_sd),
                "null_rejected": result.inference.null_rejected,
            },
            "model": {
                "level_variance": format!("{:.6}", result.model.level_variance),
                "slope_variance": result.model.slope_variance.map(|v| format!("{:.6}", v)),
                "seasonal_variance": result.model.seasonal_variance.map(|v| format!("{:.6}", v)),
                "observation_variance": format!("{:.6}", result.model.observation_variance),
                "regression_coefficients": result.model.regression_coefficients,
                "control_names": result.model.control_names,
                "log_likelihood": format!("{:.4}", result.model.log_likelihood),
                "aic": format!("{:.4}", result.model.aic),
                "bic": format!("{:.4}", result.model.bic),
            },
            "interpretation": if result.summary.significant {
                format!(
                    "The intervention had a statistically significant effect (p = {:.4}). \
                    The cumulative effect over the post-period was {:.2} [{:.2}, {:.2}], \
                    representing a {:.1}% change from what would have occurred without intervention.",
                    result.summary.p_value,
                    result.summary.cumulative_effect,
                    result.summary.cumulative_effect_lower,
                    result.summary.cumulative_effect_upper,
                    result.summary.relative_effect * 100.0
                )
            } else {
                format!(
                    "The intervention did not have a statistically significant effect at alpha = {} (p = {:.4}). \
                    While the estimated cumulative effect was {:.2}, this is within the range of natural variation \
                    expected without intervention.",
                    result.summary.alpha,
                    result.summary.p_value,
                    result.summary.cumulative_effect
                )
            },
            "references": "Brodersen et al. (2015). Inferring causal impact using Bayesian structural time series models. Annals of Applied Statistics, 9(1), 247-274."
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| format!("{:?}", result)),
        )]))
    }

    /// Compute cumulative periodogram for white noise test.
    #[tool(
        description = "Cumulative periodogram (cpgram) - diagnostic tool for testing if a time series is white noise. Plots cumulative sum of periodogram ordinates. For white noise, should follow a uniform distribution. Returns cumulative periodogram, confidence bands, and Kolmogorov-Smirnov test for white noise."
    )]
    pub async fn timeseries_cpgram(
        &self,
        Parameters(request): Parameters<CpgramRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let df = dataset.df();

        // Extract time series column
        let col = match df.column(&request.column) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Column '{}' not found: {}",
                    request.column, e
                ))]));
            }
        };

        let values: Vec<f64> = match col.cast(&DataType::Float64) {
            Ok(c) => match c.f64() {
                Ok(f) => f.into_iter().flatten().collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cannot convert column to numeric: {}",
                    e
                ))]));
            }
        };

        let result = match cpgram(&values, request.taper) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cumulative periodogram failed: {}",
                    e
                ))]));
            }
        };

        // Format output
        let output = serde_json::json!({
            "n_observations": result.n,
            "taper": result.taper,
            "white_noise_test": {
                "is_white_noise": result.is_white_noise,
                "ks_statistic": result.ks_statistic,
                "ks_p_value": result.ks_p_value,
                "max_deviation": result.max_deviation
            },
            "freq_sample": &result.freq[..result.freq.len().min(20)],
            "cpgram_sample": &result.cpgram[..result.cpgram.len().min(20)],
            "note": if result.freq.len() > 20 { Some(format!("Showing first 20 of {} frequencies", result.freq.len())) } else { None }
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| format!("{:?}", result)),
        )]))
    }

    /// Construct a Toeplitz matrix.
    #[tool(
        description = "Construct a Toeplitz matrix - a matrix with constant values along each diagonal. Useful for autocorrelation matrices, circulant matrices, and time series analysis. Can create symmetric or asymmetric Toeplitz matrices."
    )]
    pub async fn linalg_toeplitz(
        &self,
        Parameters(request): Parameters<ToeplitzRequest>,
    ) -> Result<CallToolResult, McpError> {
        let result = if let Some(ref row) = request.row {
            // Asymmetric Toeplitz matrix
            match toeplitz_asymmetric(&request.column, row) {
                Ok(mat) => toeplitz_to_vec(&mat),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Toeplitz matrix construction failed: {}",
                        e
                    ))]));
                }
            }
        } else {
            // Symmetric Toeplitz matrix
            match toeplitz(&request.column) {
                Ok(mat) => toeplitz_to_vec(&mat),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Toeplitz matrix construction failed: {}",
                        e
                    ))]));
                }
            }
        };

        let output = serde_json::json!({
            "dimensions": [result.len(), if result.is_empty() { 0 } else { result[0].len() }],
            "symmetric": request.row.is_none(),
            "matrix": result
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| format!("{:?}", result)),
        )]))
    }

    /// Lag a time series.
    #[tool(
        description = "Shift a time series by k positions. Positive k shifts values backward (lag), negative k shifts forward (lead). Returns the lagged series with NA padding."
    )]
    pub async fn timeseries_lag(
        &self,
        Parameters(request): Parameters<LagRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let df = dataset.df();
        let values: Vec<f64> = match df.column(&request.column) {
            Ok(c) => match c.cast(&DataType::Float64) {
                Ok(c) => c
                    .f64()
                    .unwrap()
                    .into_iter()
                    .map(|v| v.unwrap_or(f64::NAN))
                    .collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Column not found: {}",
                    e
                ))]));
            }
        };

        let k = request.k.unwrap_or(1);
        let result = match ts_lag(&values, k) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Lag failed: {}",
                    e
                ))]));
            }
        };

        let output = serde_json::json!({
            "lag": k,
            "n": result.values.len(),
            "values_sample": &result.values[..result.values.len().min(20)],
            "note": if result.values.len() > 20 { Some(format!("Showing first 20 of {} values", result.values.len())) } else { None }
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap(),
        )]))
    }

    /// Embed a time series into a matrix.
    #[tool(
        description = "Create a lag embedding matrix from a time series. Each row contains consecutive values, useful for building AR models or phase space reconstruction."
    )]
    pub async fn timeseries_embed(
        &self,
        Parameters(request): Parameters<EmbedRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let df = dataset.df();
        let values: Vec<f64> = match df.column(&request.column) {
            Ok(c) => match c.cast(&DataType::Float64) {
                Ok(c) => c
                    .f64()
                    .unwrap()
                    .into_iter()
                    .map(|v| v.unwrap_or(f64::NAN))
                    .collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Column not found: {}",
                    e
                ))]));
            }
        };

        let result = match embed(&values, request.dimension) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Embedding failed: {}",
                    e
                ))]));
            }
        };

        let output = serde_json::json!({
            "dimension": result.dimension,
            "n_rows": result.n_rows,
            "matrix_sample": &result.matrix[..result.n_rows.min(10)]
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap(),
        )]))
    }

    /// Inverse of differencing.
    #[tool(
        description = "Compute the inverse of differencing (cumulative sum). Reconstructs the original series from differences given initial values."
    )]
    pub async fn timeseries_diffinv(
        &self,
        Parameters(request): Parameters<DiffinvRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let df = dataset.df();
        let values: Vec<f64> = match df.column(&request.column) {
            Ok(c) => match c.cast(&DataType::Float64) {
                Ok(c) => c
                    .f64()
                    .unwrap()
                    .into_iter()
                    .map(|v| v.unwrap_or(f64::NAN))
                    .collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Column not found: {}",
                    e
                ))]));
            }
        };

        let xi = request.xi.as_deref();
        let lag_val = request.lag.unwrap_or(1);
        let differences = request.differences.unwrap_or(1);

        let result = match diffinv(&values, lag_val, differences, xi) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "diffinv failed: {}",
                    e
                ))]));
            }
        };

        let output = serde_json::json!({
            "n": result.values.len(),
            "lag": lag_val,
            "values_sample": &result.values[..result.values.len().min(20)],
            "note": if result.values.len() > 20 { Some(format!("Showing first 20 of {} values", result.values.len())) } else { None }
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap(),
        )]))
    }

    /// Linear filtering of time series.
    #[tool(
        description = "Apply a linear filter to a time series using convolution or recursive filtering. Useful for smoothing, differencing, or implementing ARMA models."
    )]
    pub async fn timeseries_filter(
        &self,
        Parameters(request): Parameters<FilterRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let df = dataset.df();
        let values: Vec<f64> = match df.column(&request.column) {
            Ok(c) => match c.cast(&DataType::Float64) {
                Ok(c) => c
                    .f64()
                    .unwrap()
                    .into_iter()
                    .map(|v| v.unwrap_or(f64::NAN))
                    .collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Column not found: {}",
                    e
                ))]));
            }
        };

        let method = match request.method.as_deref() {
            Some("recursive") => FilterMethod::Recursive,
            Some("convolution") | None => FilterMethod::Convolution,
            Some(other) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown method '{}'. Use 'convolution' or 'recursive'.",
                    other
                ))]));
            }
        };

        let sides = match request.sides {
            Some(1) => FilterSides::One,
            Some(2) | None => FilterSides::Two,
            Some(other) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Invalid sides '{}'. Use 1 (one-sided) or 2 (two-sided).",
                    other
                ))]));
            }
        };

        let result = match ts_filter(&values, &request.filter, method, sides, None) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Filter failed: {}",
                    e
                ))]));
            }
        };

        let output = serde_json::json!({
            "n": result.values.len(),
            "method": format!("{:?}", method),
            "filter_length": request.filter.len(),
            "values_sample": &result.values[..result.values.len().min(20)],
            "note": if result.values.len() > 20 { Some(format!("Showing first 20 of {} values", result.values.len())) } else { None }
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap(),
        )]))
    }

    /// Extract a window from time series.
    #[tool(
        description = "Extract a contiguous window (subset) from a time series by specifying start and end indices."
    )]
    pub async fn timeseries_window(
        &self,
        Parameters(request): Parameters<WindowRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let df = dataset.df();
        let values: Vec<f64> = match df.column(&request.column) {
            Ok(c) => match c.cast(&DataType::Float64) {
                Ok(c) => c
                    .f64()
                    .unwrap()
                    .into_iter()
                    .map(|v| v.unwrap_or(f64::NAN))
                    .collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Column not found: {}",
                    e
                ))]));
            }
        };

        let result = match ts_window(&values, request.start, request.end) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Window extraction failed: {}",
                    e
                ))]));
            }
        };

        let output = serde_json::json!({
            "start": result.start,
            "end": result.end,
            "n": result.values.len(),
            "values": result.values
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap(),
        )]))
    }

    /// Compute theoretical ACF for ARMA model.
    #[tool(
        description = "Compute the theoretical autocorrelation function (ACF) or partial ACF for an ARMA(p,q) model given AR and MA coefficients."
    )]
    pub async fn timeseries_arma_acf(
        &self,
        Parameters(request): Parameters<ArmaAcfRequest>,
    ) -> Result<CallToolResult, McpError> {
        let ar = request.ar.unwrap_or_default();
        let ma = request.ma.unwrap_or_default();
        let lag_max = request.lag_max.unwrap_or(10);
        let pacf = request.pacf.unwrap_or(false);

        let result = match arma_acf(&ar, &ma, lag_max, pacf) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "ARMAacf failed: {}",
                    e
                ))]));
            }
        };

        let output = serde_json::json!({
            "type": if pacf { "PACF" } else { "ACF" },
            "ar_order": ar.len(),
            "ma_order": ma.len(),
            "lag_max": lag_max,
            "values": result.values,
            "lags": result.lags
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap(),
        )]))
    }

    /// Convert ARMA to MA (psi weights).
    #[tool(
        description = "Convert ARMA model to its infinite MA representation (psi weights). The psi weights show the impulse response function of the model."
    )]
    pub async fn timeseries_arma_to_ma(
        &self,
        Parameters(request): Parameters<ArmaToMaRequest>,
    ) -> Result<CallToolResult, McpError> {
        let ar = request.ar.unwrap_or_default();
        let ma = request.ma.unwrap_or_default();
        let lag_max = request.lag_max.unwrap_or(10);

        let result = match arma_to_ma(&ar, &ma, lag_max) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "ARMAtoMA failed: {}",
                    e
                ))]));
            }
        };

        let output = serde_json::json!({
            "ar_order": ar.len(),
            "ma_order": ma.len(),
            "n_weights": result.psi.len(),
            "psi_weights": result.psi
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap(),
        )]))
    }

    /// Convert ACF to AR coefficients.
    #[tool(
        description = "Compute AR coefficients from autocorrelation function using the Yule-Walker equations. Also returns partial autocorrelations."
    )]
    pub async fn timeseries_acf_to_ar(
        &self,
        Parameters(request): Parameters<Acf2ArRequest>,
    ) -> Result<CallToolResult, McpError> {
        let result = match acf_to_ar(&request.acf) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "acf2AR failed: {}",
                    e
                ))]));
            }
        };

        let output = serde_json::json!({
            "max_order": result.max_order,
            "ar_matrix": result.ar_matrix,
            "partial_acf": result.pacf,
            "acf": result.acf
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap(),
        )]))
    }

    /// Simulate from ARIMA model.
    #[tool(
        description = "Simulate a time series from an ARIMA(p,d,q) model. Generates random innovations and applies the ARMA recursion with optional differencing."
    )]
    pub async fn timeseries_arima_sim(
        &self,
        Parameters(request): Parameters<ArimaSimRequest>,
    ) -> Result<CallToolResult, McpError> {
        let ar = request.ar.unwrap_or_default();
        let ma = request.ma.unwrap_or_default();
        let d = request.d.unwrap_or(0);

        let result = match arima_sim(&ar, &ma, d, request.n, None, None, request.seed) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "arima.sim failed: {}",
                    e
                ))]));
            }
        };

        let output = serde_json::json!({
            "n": result.values.len(),
            "model": {
                "ar": result.ar,
                "d": result.d,
                "ma": result.ma
            },
            "n_start": result.n_start,
            "values_sample": &result.values[..result.values.len().min(50)],
            "note": if result.values.len() > 50 { Some(format!("Showing first 50 of {} values", result.values.len())) } else { None }
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap(),
        )]))
    }

    /// Running median smoothing.
    #[tool(
        description = "Apply running median smoother to a time series. More robust to outliers than running mean. Uses Tukey's median polish for the smoothing."
    )]
    pub async fn timeseries_runmed(
        &self,
        Parameters(request): Parameters<RunmedRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let df = dataset.df();
        let values: Vec<f64> = match df.column(&request.column) {
            Ok(c) => match c.cast(&DataType::Float64) {
                Ok(c) => c
                    .f64()
                    .unwrap()
                    .into_iter()
                    .map(|v| v.unwrap_or(f64::NAN))
                    .collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Column not found: {}",
                    e
                ))]));
            }
        };

        let endrule = match request.endrule.as_deref() {
            Some("constant") => EndRule::Constant,
            Some("median") => EndRule::Median,
            Some("keep") | None => EndRule::Keep,
            Some(other) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown endrule '{}'. Use 'keep', 'constant', or 'median'.",
                    other
                ))]));
            }
        };

        let result = match runmed(&values, request.k, endrule) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "runmed failed: {}",
                    e
                ))]));
            }
        };

        let output = serde_json::json!({
            "n": result.n_obs,
            "k": result.k,
            "endrule": format!("{:?}", result.endrule),
            "values_sample": &result.values[..result.values.len().min(20)],
            "note": if result.values.len() > 20 { Some(format!("Showing first 20 of {} values", result.values.len())) } else { None }
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap(),
        )]))
    }
}
