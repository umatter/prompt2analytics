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
use crate::tools::common::{NaPolicy, limit_refusal, missing_json, numeric_column, parse_option};
use crate::tools::requests::timeseries::*;

use p2a_core::econometrics::timeseries::varma::{VarmaConfig, run_varma_with};
use p2a_core::econometrics::timeseries::vecm::{VecmConfig, run_vecm_with};
use p2a_core::forecasting::garch::GarchDistribution;
use p2a_core::forecasting::{ArimaSpec, forecast_arima_with, run_arima_with};
use p2a_core::{
    ArConfig,
    ArMethod,
    CausalImpactConfig,
    CostFunction,
    DecomposeType,
    EndRule,
    FilterMethod,
    FilterSides,
    GarchConfig,
    SeasonalType,
    StructTsType,
    acf_to_ar,
    ar,
    arima_sim,
    arma_acf,
    arma_to_ma,
    causal_impact,
    // Time series utilities
    cpgram,
    diffinv,
    embed,
    filter as ts_filter,
    garch,
    garch_forecast,
    granger_test,
    holt_winters_forecast,
    lag as ts_lag,
    run_binary_segmentation,
    run_changepoint,
    run_decompose,
    run_granger_test,
    run_holt_winters,
    run_mstl,
    run_struct_ts,
    // Time series modeling
    run_var,
    run_var_irf,
    runmed,
    stats::{
        AcfType, BoxTestType, CcfType, SpectrumConfig, run_acf, run_box_test, run_ccf, run_pacf,
        run_pp_test, run_spectrum, run_spectrum_ar,
    },
    toeplitz,
    toeplitz_asymmetric,
    toeplitz_to_vec,
    window as ts_window,
};

use crate::tools::handlers::stats::parse_choice;
use p2a_core::options::Choices;

/// Spectrum estimator of `timeseries_spectrum` (R's spectrum `method`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpectrumMethod {
    Pgram,
    Ar,
}

static SPECTRUM_METHOD: Choices<SpectrumMethod> = Choices {
    option: "method",
    values: &[
        (
            &["pgram", "periodogram", "spec.pgram"],
            SpectrumMethod::Pgram,
        ),
        (&["ar", "autoregressive", "spec.ar"], SpectrumMethod::Ar),
    ],
};

/// `timeseries_box_test` type (R's Box.test `type`).
static BOX_TEST_TYPE: Choices<BoxTestType> = Choices {
    option: "test_type",
    values: &[
        (
            &["Ljung-Box", "ljung", "lb", "ljungbox"],
            BoxTestType::LjungBox,
        ),
        (
            &["Box-Pierce", "box", "bp", "boxpierce"],
            BoxTestType::BoxPierce,
        ),
    ],
};

/// `ts_changepoint` change type (changepoint's cpt.mean, cpt.var, cpt.meanvar).
static CHANGE_TYPE: Choices<CostFunction> = Choices {
    option: "change_type",
    values: &[
        (&["mean", "cpt.mean"], CostFunction::MeanChange),
        (
            &["variance", "var", "cpt.var"],
            CostFunction::VarianceChange,
        ),
        (
            &["meanvar", "both", "mean_and_variance", "cpt.meanvar"],
            CostFunction::MeanAndVariance,
        ),
    ],
};

/// Search method of `ts_changepoint` (changepoint's `method`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CpMethod {
    Pelt,
    BinSeg,
}

static CP_METHOD: Choices<CpMethod> = Choices {
    option: "method",
    values: &[
        (&["pelt"], CpMethod::Pelt),
        (
            &["binary", "binseg", "binary_segmentation"],
            CpMethod::BinSeg,
        ),
    ],
};

/// Seasonal component of `ts_holt_winters` (R's HoltWinters `seasonal`).
static SEASONAL_TYPE: Choices<SeasonalType> = Choices {
    option: "seasonal",
    values: &[
        (&["additive", "add"], SeasonalType::Additive),
        (&["multiplicative", "mult"], SeasonalType::Multiplicative),
    ],
};

/// Estimation method of `ts_ar` (R's ar `method`).
static AR_METHOD: Choices<ArMethod> = Choices {
    option: "method",
    values: &[
        (&["yule-walker", "yw", "yulewalker"], ArMethod::YuleWalker),
        (&["burg"], ArMethod::Burg),
        (&["ols"], ArMethod::Ols),
    ],
};

/// `ts_decompose` type (R's decompose `type`).
static DECOMPOSE_TYPE: Choices<DecomposeType> = Choices {
    option: "decompose_type",
    values: &[
        (&["additive", "add"], DecomposeType::Additive),
        (&["multiplicative", "mult"], DecomposeType::Multiplicative),
    ],
};

/// `ts_structts` model (R's StructTS `type`).
static STRUCTTS_TYPE: Choices<StructTsType> = Choices {
    option: "model_type",
    values: &[
        (&["level", "local_level"], StructTsType::Level),
        (&["trend", "local_linear_trend"], StructTsType::Trend),
        (&["bsm", "basic_structural_model"], StructTsType::BSM),
    ],
};

/// `ts_filter` method (R's filter `method`).
static FILTER_METHOD: Choices<FilterMethod> = Choices {
    option: "method",
    values: &[
        (&["convolution"], FilterMethod::Convolution),
        (&["recursive"], FilterMethod::Recursive),
    ],
};

/// `ts_runmed` end rule (R's runmed `endrule`).
static END_RULE: Choices<EndRule> = Choices {
    option: "endrule",
    values: &[
        (&["keep"], EndRule::Keep),
        (&["constant"], EndRule::Constant),
        (&["median"], EndRule::Median),
    ],
};

#[tool_router(router = timeseries_router, vis = "pub")]
impl AnalyticsServer {
    /// - R `stats::acf`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/acf.html
    #[tool(
        description = "Compute autocorrelation function (ACF), autocovariance, or partial autocorrelation function (PACF) for a time series. ACF measures correlation between observations at different lags. PACF measures correlation after removing effects of intermediate lags. Returns values for lags 0 to lag_max with 95% confidence bounds. For model fitting, use ts_arima_fit directly instead of computing ACF first."
    )]
    pub async fn timeseries_acf(
        &self,
        Parameters(request): Parameters<AcfRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        // Parse ACF type
        let acf_type = match parse_option(request.acf_type.as_deref(), AcfType::Correlation) {
            Ok(v) => v,
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
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
        description = "Compute the cross-correlation function of two series, as R's ccf(x, y): the value at lag k estimates cor(x[t+k], y[t]), so a peak at a negative lag k means x leads y by |k| periods and a peak at a positive lag means y leads x. Missing or infinite values are refused (R: na.fail). Returns the values for lags -lag_max to lag_max, the white-noise bound qnorm(0.975)/sqrt(n) and the lag of the largest |correlation|."
    )]
    pub async fn timeseries_ccf(
        &self,
        Parameters(request): Parameters<CcfRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        match run_ccf(
            dataset,
            &request.x,
            &request.y,
            request.lag_max,
            CcfType::Correlation,
        ) {
            Ok(result) => {
                let mut json_output = serde_json::json!({
                    "type": "Cross-Correlation (CCF)",
                    "x_series": request.x,
                    "y_series": request.y,
                    "n_obs": result.n_obs,
                    "lag_convention": "value at lag k = cor(x[t+k], y[t]) (R's ccf(x, y))",
                    "lags": result.lags,
                    "values": result.values,
                    "confidence_bound_95": result.confidence_bound,
                });
                match peak_lag(&result.lags, &result.values) {
                    Some((k, v)) => {
                        json_output["max_correlation"] = serde_json::json!({
                            "lag": k,
                            "value": v,
                            "interpretation": lead_lag(&request.x, &request.y, k)
                        });
                    }
                    None => {
                        json_output["max_correlation"] = serde_json::Value::Null;
                        json_output["note"] = serde_json::json!("no finite cross-correlation");
                    }
                }
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
        description = "Estimate spectral density (power spectrum) of a time series. Returns spectral density at Fourier frequencies showing how variance is distributed across frequency components. Methods: 'pgram' (periodogram with optional smoothing) or 'ar' (AR model-based). Use spans parameter for smoothing raw periodogram (e.g., spans=[3,3] for moderate smoothing). Peak frequency reveals dominant cyclical patterns. The periodogram reports R's spec.ci 95% multipliers at its degrees of freedom; AR spectra have no confidence interval."
    )]
    pub async fn timeseries_spectrum(
        &self,
        Parameters(request): Parameters<SpectrumRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        // Determine method
        let method = match parse_choice(
            &SPECTRUM_METHOD,
            request.method.as_deref(),
            SpectrumMethod::Pgram,
        ) {
            Ok(v) => v,
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
        };

        let result = match method {
            SpectrumMethod::Ar => {
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
            SpectrumMethod::Pgram => {
                // Periodogram method
                let mut config = SpectrumConfig::default();
                config.spans = request.spans.clone();
                config.taper = request.taper.unwrap_or(0.1);
                config.detrend = request.detrend.unwrap_or(true);
                config.demean = false;
                config.pad_ratio = 0.0;

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

        // R's spec.ci multipliers at the reported df; none for an AR spectrum (no df)
        let ci = match result.confidence_interval_multipliers(0.95) {
            Ok(ci) => ci,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Spectral estimation failed: {e}"
                ))]));
            }
        };

        let mut json_output = serde_json::json!({
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
            "confidence_interval_95": ci.map(|(lo, hi)| serde_json::json!({
                "multiplier_lower": lo,
                "multiplier_upper": hi,
                "degrees_of_freedom": result.df,
                "interpretation": "multiply a spectral estimate by these for a 95% interval (R's spec.ci at this df)"
            })),
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
        if ci.is_none() {
            json_output["confidence_interval_note"] = serde_json::json!(
                "no chi-square interval for AR spectra (R's spec.ar has no degrees of freedom)"
            );
        }

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

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        // Parse test type
        let test_type = match parse_choice(
            &BOX_TEST_TYPE,
            request.test_type.as_deref(),
            BoxTestType::LjungBox,
        ) {
            Ok(v) => v,
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
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

        let dataset = crate::get_dataset!(datasets, &request.dataset);

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
        description = "Vector autoregression VAR(p) with a constant, as vars::VAR(type = 'const'): per-equation coefficients, standard errors (df T - Kp - 1), t and p values, the residual covariance, logLik, AIC and BIC. Returns the summary as text followed by the full result as JSON."
    )]
    pub async fn ts_var(
        &self,
        Parameters(request): Parameters<VarRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let columns: Vec<&str> = request.columns.iter().map(|s| s.as_str()).collect();

        let result = match run_var(dataset, &columns, request.lags) {
            Ok(r) => r,
            Err(e) => {
                return Ok(ts_error("VAR estimation failed", &e));
            }
        };

        Ok(text_and_json(&result))
    }

    /// Granger causality test.
    #[tool(
        description = "Test Granger causality between two time series variables, as lmtest::grangertest: whether lagged values of 'cause' help predict 'dependent' after controlling for lagged 'dependent' (F test of the restricted against the unrestricted model). Without lags, Schwert's order is used, capped so that the F test keeps at least one residual degree of freedom, with a warning. Returns text followed by the result as JSON (F, df1, df2, p, the order used and any warning)."
    )]
    pub async fn ts_granger(
        &self,
        Parameters(request): Parameters<GrangerRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let result = match request.lags {
            Some(lags) => granger_test(dataset, &request.dependent, &request.cause, lags),
            None => run_granger_test(dataset, &request.dependent, &request.cause),
        };

        let result = match result {
            Ok(r) => r,
            Err(e) => {
                return Ok(ts_error("Granger causality test failed", &e));
            }
        };

        Ok(text_and_json(&result))
    }

    /// Run VARMA (Vector ARMA) model.
    #[tool(
        description = "VARMA(p, q) with a constant by Hannan & Rissanen (1982): a long VAR(m) by OLS, then OLS of y_t on its lags and the lagged step-1 residuals; long_var_order sets m. Returns the coefficient table (standard errors, t and p only for q = 0, where the second step is an exact OLS VAR), logLik, AIC and BIC, as text followed by JSON."
    )]
    pub async fn ts_varma(
        &self,
        Parameters(request): Parameters<VarmaRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let columns: Vec<&str> = request.columns.iter().map(|s| s.as_str()).collect();

        let mut config = VarmaConfig::default();
        config.p = request.p;
        config.q = request.q;
        config.long_var_order = request.long_var_order;
        let result = match run_varma_with(dataset, &columns, &config) {
            Ok(r) => r,
            Err(e) => {
                return Ok(ts_error("VARMA estimation failed", &e));
            }
        };

        let mut json = match serde_json::to_value(&result) {
            Ok(j) => j,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "the result could not be serialised: {e}"
                ))]));
            }
        };
        json["t_values"] = serde_json::json!(result.t_values());
        json["p_values"] = serde_json::json!(result.p_values());
        Ok(CallToolResult::success(vec![Content::text(format!(
            "{result}\n\nJSON:\n{json}"
        ))]))
    }

    /// Run VECM (Vector Error Correction Model).
    #[tool(
        description = "Johansen procedure and restricted VECM as urca::ca.jo (spec = 'transitory'; trace and maximal-eigenvalue tests with urca's Osterwald-Lenum critical values) and urca::cajorls (beta, alpha, Gamma), for cointegrated I(1) series. ecdet as urca's; the rank is chosen by the trace test at 5% unless given. Refused: lags below 2, a rank outside 1..k-1, more than 11 variables. Returns text followed by the result as JSON."
    )]
    pub async fn ts_vecm(
        &self,
        Parameters(request): Parameters<VecmRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let columns: Vec<&str> = request.columns.iter().map(|s| s.as_str()).collect();

        let mut config = VecmConfig::default();
        config.lag_order = request.lags;
        config.ecdet = match parse_option(request.ecdet.as_deref(), config.ecdet) {
            Ok(v) => v,
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
        };
        config.rank = request.rank;
        let result = match run_vecm_with(dataset, &columns, &config) {
            Ok(r) => r,
            Err(e) => {
                return Ok(ts_error("VECM estimation failed", &e));
            }
        };

        Ok(text_and_json(&result))
    }

    /// Compute VAR Impulse Response Functions.
    #[tool(
        description = "Compute impulse response functions from a VAR(p) with a constant, orthogonalised (lower Cholesky factor of Sigma_u) as vars::irf(ortho = TRUE). Returns text followed by every horizon as JSON: irf[h][response][impulse] for h = 0 .. steps - 1."
    )]
    pub async fn ts_var_irf(
        &self,
        Parameters(request): Parameters<VarIrfRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let columns: Vec<&str> = request.columns.iter().map(|s| s.as_str()).collect();

        let result = match run_var_irf(dataset, &columns, request.lags, request.steps) {
            Ok(r) => r,
            Err(e) => {
                return Ok(ts_error("VAR IRF computation failed", &e));
            }
        };

        Ok(text_and_json(&result))
    }

    /// Fit an ARIMA model by conditional sum of squares.
    #[tool(
        description = "Fit an ARIMA(p,d,q) model to a univariate time series by conditional sum of squares (as R arima(method = 'CSS') and forecast::Arima(method = 'CSS')). Returns the AR and MA coefficients, the mean (R's 'intercept', estimated by default when d = 0) or a drift (include_drift, d = 0 or 1), sigma^2, the CSS log-likelihood and AIC, and whether the optimiser converged. The series must have no missing values. If the user does not specify p, d, q, use reasonable defaults like ARIMA(1,0,1) or ARIMA(1,1,1). Do NOT call timeseries_acf first — call this tool directly."
    )]
    pub async fn ts_arima_fit(
        &self,
        Parameters(request): Parameters<ArimaRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);
        let orders = (request.p, request.d, request.q);
        if let Some(refusal) = arima_refusal("ts_arima_fit", dataset, &request.column, orders) {
            return Ok(refusal);
        }
        let spec = arima_spec(
            request.p,
            request.d,
            request.q,
            request.include_mean,
            request.include_drift,
        );
        match run_arima_with(dataset, &request.column, &spec) {
            Ok(fit) => Ok(CallToolResult::success(vec![Content::text(
                fit.to_string(),
            )])),
            Err(e) => Ok(ts_error("ARIMA fitting failed", &e)),
        }
    }

    /// Forecast with an ARIMA model fitted by conditional sum of squares.
    #[tool(
        description = "Forecast h steps ahead with an ARIMA(p,d,q) model fitted by conditional sum of squares. Returns the fitted model, point forecasts, standard errors and 80% / 95% prediction intervals computed as R's forecast(Arima(x, order, method = 'CSS')). Accepts include_mean and include_drift like ts_arima_fit. The series must have no missing values."
    )]
    pub async fn ts_arima_forecast(
        &self,
        Parameters(request): Parameters<ArimaForecastRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);
        let orders = (request.p, request.d, request.q);
        if let Some(refusal) = arima_refusal("ts_arima_forecast", dataset, &request.column, orders)
        {
            return Ok(refusal);
        }
        let spec = arima_spec(
            request.p,
            request.d,
            request.q,
            request.include_mean,
            request.include_drift,
        );
        match forecast_arima_with(dataset, &request.column, &spec, request.horizon) {
            Ok(fc) => {
                let fit = fc.fit.as_ref().map(|f| f.to_string()).unwrap_or_default();
                Ok(CallToolResult::success(vec![Content::text(format!(
                    "{fit}\n\n{fc}"
                ))]))
            }
            Err(e) => Ok(ts_error("ARIMA forecasting failed", &e)),
        }
    }

    /// Fit a GARCH model for volatility modeling.
    #[tool(
        description = "GARCH(p, q) by conditional maximum likelihood with normal ('norm') or standardised Student-t ('std') errors, as fGarch::garchFit (mci start h_t = mean(e²)); SEs from the inverse observed information, z and normal p-values; omega is in the data's units; parameters on a bound are reported with a warning; optional variance forecasts."
    )]
    pub async fn ts_garch_fit(
        &self,
        Parameters(request): Parameters<GarchRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let data = match numeric_column(dataset, &request.column, NaPolicy::Fail) {
            Ok((values, _)) => values,
            Err(e) => return Ok(ts_error("GARCH fitting failed", &e)),
        };

        let mut config = GarchConfig {
            p: request.p.unwrap_or(1),
            q: request.q.unwrap_or(1),
            include_mean: request.include_mean.unwrap_or(true),
            ..Default::default()
        };
        config.distribution =
            match parse_option(request.distribution.as_deref(), GarchDistribution::Normal) {
                Ok(v) => v,
                Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
            };

        let result = match garch(&data, Some(config)) {
            Ok(r) => r,
            Err(e) => {
                return Ok(ts_error("GARCH fitting failed", &e));
            }
        };

        let mut output = serde_json::json!({
            "model": result.model,
            "distribution": result.distribution.to_string(),
            "param_names": result.param_names,
            "estimates": result.estimates(),
            "std_errors": result.std_errors,
            "z_values": result.t_stats,
            "p_values": result.p_values,
            "shape": result.shape,
            "vcov": result.vcov,
            "log_likelihood": result.log_likelihood,
            "aic": result.aic,
            "bic": result.bic,
            "persistence": result.persistence,
            "unconditional_variance": result.unconditional_variance,
            "half_life": result.half_life,
            "n_obs": result.n_obs,
            "n_params": result.n_params,
            "at_bound": result.at_bound,
        });
        if let Some(h) = request.horizon {
            let (variance, volatility, _) = match garch_forecast(&result, h) {
                Ok(v) => v,
                Err(e) => return Ok(ts_error("GARCH forecast failed", &e)),
            };
            output["forecast"] = serde_json::json!({
                "horizon": h,
                "variance": variance,
                "volatility": volatility,
            });
        }
        crate::tools::handlers::stats::insert_notes(
            &mut output,
            Some((result.converged, result.iterations)),
            &result.warnings,
        );

        Ok(CallToolResult::success(vec![Content::text(format!(
            "{result}\n\nJSON:\n{output}"
        ))]))
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

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let result = match run_mstl(dataset, &request.column, &request.periods) {
            Ok(r) => r,
            Err(e) => {
                return Ok(ts_error("MSTL decomposition failed", &e));
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
        description = "Detect changepoints (structural breaks) in a time series using PELT (changepoint's cpt.mean / cpt.var / cpt.meanvar with the MBIC penalty by default; the mean cost assumes unit variance) or Binary Segmentation (mean changes only). Identifies points where the mean or variance changes. Useful for regime detection, anomaly detection, and segmenting time series into homogeneous periods."
    )]
    pub async fn ts_changepoint(
        &self,
        Parameters(request): Parameters<ChangepointRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        // Determine cost function
        let cost_fn = match parse_choice(
            &CHANGE_TYPE,
            request.change_type.as_deref(),
            CostFunction::MeanChange,
        ) {
            Ok(v) => v,
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
        };
        let cp_method = match parse_choice(&CP_METHOD, request.method.as_deref(), CpMethod::Pelt) {
            Ok(v) => v,
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
        };
        if cp_method == CpMethod::BinSeg && cost_fn != CostFunction::MeanChange {
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "method 'binary' detects mean changes only (binary_segmentation has no variance cost); use method 'pelt' for change_type '{}'",
                CHANGE_TYPE.canonical(cost_fn)
            ))]));
        }

        // Run detection based on method
        let result = match cp_method {
            CpMethod::BinSeg => {
                match run_binary_segmentation(
                    dataset,
                    &request.column,
                    Some(10), // max changepoints
                    request.min_segment_length,
                    request.penalty,
                ) {
                    Ok(r) => r,
                    Err(e) => {
                        return Ok(ts_error("Changepoint detection failed", &e));
                    }
                }
            }
            CpMethod::Pelt => {
                match run_changepoint(
                    dataset,
                    &request.column,
                    request.penalty,
                    request.min_segment_length,
                    cost_fn,
                ) {
                    Ok(r) => r,
                    Err(e) => {
                        return Ok(ts_error("Changepoint detection failed", &e));
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
        description = "Fit Holt-Winters exponential smoothing model to a time series with trend and seasonality. Supports both additive (constant seasonal variation) and multiplicative (proportional variation) seasonality. Automatically optimizes smoothing parameters (alpha, beta, gamma) if not provided. Can generate forecasts for future periods, with predict.HoltWinters' 95% prediction intervals for additive and non-seasonal fits (point forecasts only for multiplicative fits). Returns fitted values, residuals, optimized parameters, and seasonal coefficients."
    )]
    pub async fn ts_holt_winters(
        &self,
        Parameters(request): Parameters<HoltWintersRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        // Parse seasonal type
        let seasonal = match parse_choice(
            &SEASONAL_TYPE,
            request.seasonal.as_deref(),
            SeasonalType::Additive,
        ) {
            Ok(v) => v,
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
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
                return Ok(ts_error("Holt-Winters fitting failed", &e));
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
                // predict.HoltWinters' interval formula is wrong for multiplicative
                // seasonality, so those fits get point forecasts only.
                let with_intervals =
                    result.seasonal_type == SeasonalType::Additive || result.gamma.is_none();
                let forecasts = if with_intervals {
                    result.predict(horizon, 0.95).map(|fc| {
                        output.push_str(&format!(
                            "\nForecasts ({horizon} periods ahead; 95% prediction intervals as predict.HoltWinters):\n"
                        ));
                        for i in 1..=fc.forecast.len() {
                            output.push_str(&format!(
                                "  t+{i}: {:.4}  [{:.4}, {:.4}]\n",
                                fc.forecast[i - 1],
                                fc.lower[i - 1],
                                fc.upper[i - 1]
                            ));
                        }
                    })
                } else {
                    holt_winters_forecast(&result, horizon).map(|forecasts| {
                        output.push_str(&format!("\nForecasts ({} periods ahead):\n", horizon));
                        for (i, val) in forecasts.iter().enumerate() {
                            output.push_str(&format!("  t+{}: {:.4}\n", i + 1, val));
                        }
                        output.push_str(
                            "Note: prediction intervals are implemented for additive and non-seasonal fits only (predict.HoltWinters' multiplicative formula is wrong, TS-22)\n",
                        );
                    })
                };
                match forecasts {
                    Ok(()) => {}
                    // A resource or deadline failure is a failure of the call, not a
                    // note on a successful fit: the governor maps an expired budget to 504.
                    Err(
                        e @ (p2a_core::EconError::ResourceLimit(_)
                        | p2a_core::EconError::Cancelled(_)),
                    ) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Holt-Winters forecast failed: {e}"
                        ))]));
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

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x = match numeric_column(dataset, &request.column, NaPolicy::Fail) {
            Ok((values, _)) => values,
            Err(e) => return Ok(ts_error("AR model fitting failed", &e)),
        };

        // Parse method
        let method = match parse_choice(&AR_METHOD, request.method.as_deref(), ArMethod::YuleWalker)
        {
            Ok(v) => v,
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
        };

        // Build config
        let mut config = ArConfig::default();
        config.aic = request.aic.unwrap_or(true);
        config.order_max = request.order_max;
        config.order = request.order;
        config.method = method;
        config.demean = request.demean.unwrap_or(true);

        // Fit AR model
        let result = match ar(&x, config) {
            Ok(r) => r,
            Err(e) => {
                return Ok(ts_error("AR model fitting failed", &e));
            }
        };

        // Build output
        let method_str = match result.method {
            ArMethod::YuleWalker => "Yule-Walker",
            ArMethod::Burg => "Burg",
            ArMethod::Ols => "OLS",
        };

        let mut output = serde_json::json!({
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
        crate::tools::handlers::stats::insert_notes(&mut output, None, &result.warnings);

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

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        // Parse decomposition type
        let decompose_type = match parse_choice(
            &DECOMPOSE_TYPE,
            request.decompose_type.as_deref(),
            DecomposeType::Additive,
        ) {
            Ok(v) => v,
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
        };

        // Run decomposition (the core wrapper reads the column: Int64 accepted, a missing
        // value refused by name)
        let result = match run_decompose(dataset, &request.column, request.period, decompose_type) {
            Ok(r) => r,
            Err(e) => {
                return Ok(ts_error("Decomposition failed", &e));
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

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        // Parse model type
        let model_type = match parse_choice(
            &STRUCTTS_TYPE,
            request.model_type.as_deref(),
            StructTsType::Level,
        ) {
            Ok(v) => v,
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
        };

        // Validate period for BSM
        if model_type == StructTsType::BSM && request.period.is_none() {
            return Ok(CallToolResult::error(vec![Content::text(
                "BSM model requires a period parameter (e.g., 12 for monthly data)",
            )]));
        }

        // Run StructTS (the core wrapper reads the column: Int64 accepted, a missing value
        // refused by name)
        let result = match run_struct_ts(dataset, &request.column, model_type, request.period) {
            Ok(r) => r,
            Err(e) => {
                return Ok(ts_error("StructTS fitting failed", &e));
            }
        };

        // Build JSON output
        let mut coef_json = serde_json::json!({
            "level_variance": result.coef.level,
            "observation_variance": result.coef.epsilon,
        });

        if let Some(slope_var) = result.coef.slope {
            coef_json["slope_variance"] = serde_json::json!(slope_var);
        }
        if let Some(seasonal_var) = result.coef.seasonal {
            coef_json["seasonal_variance"] = serde_json::json!(seasonal_var);
        }

        let model_type_str = match result.model_type {
            StructTsType::Level => "level",
            StructTsType::Trend => "trend",
            StructTsType::BSM => "bsm",
        };

        let mut output = serde_json::json!({
            "method": "Structural Time Series Model",
            "model_type": model_type_str,
            "n_obs": result.n_obs,
            "n_params": result.n_params,
            "converged": result.converged,
            "variance_coefficients": coef_json,
            "fit_statistics": {
                "log_likelihood": result.log_likelihood,
                "aic": result.aic,
                "bic": result.bic,
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
        crate::tools::handlers::stats::insert_notes(
            &mut output,
            Some((result.convergence.converged, result.convergence.iterations)),
            &result.warnings,
        );

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

        let dataset = crate::get_dataset!(datasets, &request.dataset);

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
                return Ok(ts_error("CausalImpact analysis failed", &e));
            }
        };

        // Build output
        let mut output = serde_json::json!({
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
        crate::tools::handlers::stats::insert_notes(
            &mut output,
            Some((result.convergence.converged, result.convergence.iterations)),
            &result.warnings,
        );

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| format!("{:?}", result)),
        )]))
    }

    /// Compute cumulative periodogram for white noise test.
    #[tool(
        description = "Cumulative periodogram as R's cpgram: centred, split-cosine tapered (taper 0.1), cumulative periodogram over the Fourier frequencies; band 1.358 / (sqrt(m) + 0.12 + 0.11/sqrt(m)); Stephens' statistic and the Kolmogorov p-value; missing values are dropped, as R. Tests whether a series is white noise."
    )]
    pub async fn timeseries_cpgram(
        &self,
        Parameters(request): Parameters<CpgramRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        // R's cpgram drops missing values (x <- x[!is.na(x)]); the drop is reported.
        let (values, missing) = match numeric_column(dataset, &request.column, NaPolicy::Omit) {
            Ok(extracted) => extracted,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cumulative periodogram failed: {e}"
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
        let mut output = serde_json::json!({
            "n_observations": result.n,
            "taper": result.taper,
            "white_noise_test": {
                "is_white_noise": result.is_white_noise,
                "critical_value": result.critical_value,
                "ks_statistic": result.ks_statistic,
                "ks_p_value": result.ks_p_value,
                "max_deviation": result.max_deviation
            },
            "freq_sample": &result.freq[..result.freq.len().min(20)],
            "cpgram_sample": &result.cpgram[..result.cpgram.len().min(20)],
            "note": if result.freq.len() > 20 { Some(format!("Showing first 20 of {} frequencies", result.freq.len())) } else { None }
        });
        if let Some(m) = missing_json(&missing) {
            output["missing"] = m;
        }

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

        let values = match series_with_gaps(dataset, &request.column) {
            Ok(values) => values,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
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

        let values = match series_with_gaps(dataset, &request.column) {
            Ok(values) => values,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        // The response shows the first 10 rows only, so embed only the prefix those rows
        // need (dimension + 9 values). With dimension 0 or above the series length, embed
        // returns the same errors as on the whole series.
        let n = values.len();
        let prefix = n.min(request.dimension.saturating_add(9));
        let result = match embed(&values[..prefix], request.dimension) {
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
            // 1 <= dimension <= n here, since embed succeeded
            "n_rows": n - request.dimension + 1,
            "matrix_sample": &result.matrix
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

        let values = match series_with_gaps(dataset, &request.column) {
            Ok(values) => values,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
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

        let values = match series_with_gaps(dataset, &request.column) {
            Ok(values) => values,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        let method = match parse_choice(
            &FILTER_METHOD,
            request.method.as_deref(),
            FilterMethod::Convolution,
        ) {
            Ok(v) => v,
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
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
            "method": FILTER_METHOD.canonical(method),
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

        let values = match series_with_gaps(dataset, &request.column) {
            Ok(values) => values,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
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

        let values = match numeric_column(dataset, &request.column, NaPolicy::Fail) {
            Ok((values, _)) => values,
            Err(e) => return Ok(ts_error("runmed failed", &e)),
        };

        let endrule = match parse_choice(&END_RULE, request.endrule.as_deref(), EndRule::Keep) {
            Ok(v) => v,
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
        };

        let result = match runmed(&values, request.k, endrule) {
            Ok(r) => r,
            Err(e) => {
                return Ok(ts_error("runmed failed", &e));
            }
        };

        let output = serde_json::json!({
            "n": result.n_obs,
            "k": result.k,
            "endrule": END_RULE.canonical(result.endrule),
            "values_sample": &result.values[..result.values.len().min(20)],
            "note": if result.values.len() > 20 { Some(format!("Showing first 20 of {} values", result.values.len())) } else { None }
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap(),
        )]))
    }
}

/// The refusal of an ARIMA fit of `column` at orders `p`, `d` and `q` that the installed
/// `arima_fit_work` budget does not allow, checked before anything is fitted. The in-house CSS
/// fit checks the call's budget on every iteration, so the deadline does stop it; the rule is
/// kept as an up-front cost bound (defense in depth), so that a call the hosted deadline could
/// not finish is refused before it starts. While the budget is set a series that would slow
/// the fit's arithmetic is refused as well. That check reads the values the fit reads, the
/// column's valid values (`numeric_options`: the strict numeric rule, so integer columns are
/// bounded too; a missing cell is `None`, and the buffer value behind a null slot is never
/// read). A missing, text or infinite column is not refused here, so that the core reports it.
fn arima_refusal(
    tool: &str,
    dataset: &p2a_core::Dataset,
    column: &str,
    (p, d, q): (usize, usize, usize),
) -> Option<CallToolResult> {
    let values = p2a_core::data::columns::numeric_options(dataset.df(), column).ok()?;
    let limits = crate::tools::limits::current();
    limits
        .check_arima_fit(tool, values.len(), p, d, q)
        .err()
        .or_else(|| {
            limits
                .check_arima_series(tool, column, values.iter().flatten().copied())
                .err()
        })
        .map(limit_refusal)
}

/// The ARIMA specification of a request: R's defaults unless the options are given.
fn arima_spec(
    p: usize,
    d: usize,
    q: usize,
    include_mean: Option<bool>,
    include_drift: Option<bool>,
) -> ArimaSpec {
    ArimaSpec {
        p,
        d,
        q,
        include_mean,
        include_drift: include_drift.unwrap_or(false),
    }
}

/// A model result as its text summary followed by "JSON:" and the result as compact JSON
/// (NaN and infinite values become null).
fn text_and_json<T: serde::Serialize + std::fmt::Display>(result: &T) -> CallToolResult {
    match serde_json::to_string(result) {
        Ok(json) => {
            CallToolResult::success(vec![Content::text(format!("{result}\n\nJSON:\n{json}"))])
        }
        Err(e) => CallToolResult::error(vec![Content::text(format!(
            "the result could not be serialised: {e}"
        ))]),
    }
}

/// Lag of the largest |value| among the finite values (ties: the first in `lags` order); None
/// when none is finite.
fn peak_lag(lags: &[i32], values: &[f64]) -> Option<(i32, f64)> {
    lags.iter().zip(values).filter(|(_, v)| v.is_finite()).fold(
        None,
        |best: Option<(i32, f64)>, (&k, &v)| match best {
            Some((_, b)) if v.abs().total_cmp(&b.abs()) != std::cmp::Ordering::Greater => best,
            _ => Some((k, v)),
        },
    )
}

/// R's reading of a ccf(x, y) peak at lag k: k < 0 means x leads y.
fn lead_lag(x: &str, y: &str, k: i32) -> String {
    let n = k.unsigned_abs();
    let unit = if n == 1 { "period" } else { "periods" };
    match k.cmp(&0) {
        std::cmp::Ordering::Less => format!("{x} leads {y} by {n} {unit}"),
        std::cmp::Ordering::Greater => format!("{y} leads {x} by {n} {unit}"),
        std::cmp::Ordering::Equal => format!("{x} and {y} are contemporaneously correlated"),
    }
}

/// A tool error `prefix: err`; for a missing value, with how to fill the gap. The hint
/// narrows the generic "Drop or impute them first." of the missing-value error to imputing.
fn ts_error(prefix: &str, err: &dyn std::fmt::Display) -> CallToolResult {
    let message = err.to_string();
    let mut text = format!("{prefix}: {message}");
    if message.contains("missing value") {
        text.push_str(
            "\nHint: for a time series, impute rather than drop: p2a's time-series methods \
             need a complete series, and dropping a value would splice it. Fill the gaps first \
             (for example munge_fill_na with strategy \"forward\" for empty cells; NaN values \
             from arithmetic must be recomputed or replaced).",
        );
    }
    CallToolResult::error(vec![Content::text(text)])
}

/// The column with missing cells as NaN, for the series utilities: R propagates NA through
/// lag, embed, diffinv, filter and window, so missing cells stay in place as NaN; text is
/// refused.
fn series_with_gaps(dataset: &p2a_core::Dataset, column: &str) -> Result<Vec<f64>, String> {
    let values = p2a_core::data::columns::numeric_options(dataset.df(), column)
        .map_err(|e| e.to_string())?;
    Ok(values.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use p2a_core::data::Dataset;
    use p2a_core::polars::prelude::*;
    use rmcp::model::RawContent;

    #[test]
    fn test_mcp_ts_error_hint_covers_nan_gaps() {
        let r = ts_error(
            "ACF failed",
            &"Column 'x' contains 1 missing value(s) (null or NaN). Drop or impute them first.",
        );
        let t = match &r.content[0].raw {
            RawContent::Text(t) => t.text.clone(),
            _ => panic!("not text"),
        };
        assert!(t.contains("NaN values from arithmetic"), "{t}");
        assert!(t.contains("impute rather than drop"), "{t}");
    }

    /// TOOL-11: the peak is taken over finite values by a total order; NaN never panics.
    #[test]
    fn test_ccf_peak_lag_total_order() {
        assert_eq!(
            peak_lag(&[-1, 0, 1], &[0.2, f64::NAN, -0.5]),
            Some((1, -0.5))
        );
        // tie on |v|: first in lag order
        assert_eq!(peak_lag(&[-1, 0, 1], &[0.4, -0.4, 0.1]), Some((-1, 0.4)));
        assert_eq!(peak_lag(&[-1, 0], &[f64::NAN, f64::NAN]), None);
        assert_eq!(peak_lag(&[], &[]), None);
        assert_eq!(peak_lag(&[0], &[f64::INFINITY]), None);
    }

    #[test]
    fn test_timeseries_option_tables() {
        let checks = [
            SPECTRUM_METHOD.validate(),
            BOX_TEST_TYPE.validate(),
            CHANGE_TYPE.validate(),
            CP_METHOD.validate(),
            SEASONAL_TYPE.validate(),
            AR_METHOD.validate(),
            DECOMPOSE_TYPE.validate(),
            STRUCTTS_TYPE.validate(),
            FILTER_METHOD.validate(),
            END_RULE.validate(),
        ];
        assert!(checks.iter().all(|r| r.is_ok()), "{checks:?}");
        assert_eq!(
            parse_choice(&BOX_TEST_TYPE, Some("box_pierce"), BoxTestType::LjungBox),
            Ok(BoxTestType::BoxPierce)
        );
        assert_eq!(
            parse_choice(&AR_METHOD, Some("YW"), ArMethod::Burg),
            Ok(ArMethod::YuleWalker)
        );
        assert_eq!(
            parse_choice(&AR_METHOD, Some("mle"), ArMethod::YuleWalker).unwrap_err(),
            "unknown method 'mle'; expected one of: yule-walker, burg, ols"
        );
        assert_eq!(
            parse_choice(&CHANGE_TYPE, Some("cpt.var"), CostFunction::MeanChange),
            Ok(CostFunction::VarianceChange)
        );
        assert_eq!(
            parse_choice(&STRUCTTS_TYPE, Some("BSM"), StructTsType::Level),
            Ok(StructTsType::BSM)
        );
        assert_eq!(
            parse_choice(&END_RULE, Some("Keep"), EndRule::Median),
            Ok(EndRule::Keep)
        );
        assert_eq!(
            parse_choice(&SPECTRUM_METHOD, Some("welch"), SpectrumMethod::Pgram).unwrap_err(),
            "unknown method 'welch'; expected one of: pgram, ar"
        );
    }

    fn text_of(r: &CallToolResult) -> String {
        r.content
            .iter()
            .filter_map(|c| match &c.raw {
                RawContent::Text(t) => Some(t.text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    async fn server_with_series(name: &str, y: &[f64]) -> AnalyticsServer {
        let server = AnalyticsServer::new();
        let df = df!("y" => y).unwrap();
        server
            .datasets
            .write()
            .await
            .insert(name.to_string(), Dataset::new(df));
        server
    }

    fn seasonal(n: usize) -> Vec<f64> {
        (0..n)
            .map(|t| {
                let tf = t as f64;
                50.0 + 0.4 * tf
                    + 5.0 * (2.0 * std::f64::consts::PI * tf / 12.0).sin()
                    + 0.3 * (((t * 7) % 5) as f64 - 2.0)
            })
            .collect()
    }

    fn hw(horizon: usize) -> HoltWintersRequest {
        HoltWintersRequest {
            dataset: "hw".into(),
            column: "y".into(),
            period: 12,
            seasonal: None,
            alpha: None,
            beta: None,
            gamma: None,
            horizon: Some(horizon),
        }
    }

    #[tokio::test]
    async fn ts_holt_winters_oversized_horizon_is_a_tool_error() {
        let server = server_with_series("hw", &seasonal(48)).await;
        // Before the fix this aborted the whole process.
        let r = server
            .ts_holt_winters(Parameters(hw(1usize << 50)))
            .await
            .unwrap();
        assert_eq!(r.is_error, Some(true), "{}", text_of(&r));
        assert!(text_of(&r).contains("Resource limit"), "{}", text_of(&r));
        let ok = server.ts_holt_winters(Parameters(hw(12))).await.unwrap();
        assert_ne!(ok.is_error, Some(true), "{}", text_of(&ok));
        assert!(text_of(&ok).contains("t+12:"));
    }

    #[tokio::test]
    async fn timeseries_embed_output_is_unchanged() {
        let x = seasonal(300);
        let server = server_with_series("e", &x).await;
        let r = server
            .timeseries_embed(Parameters(EmbedRequest {
                dataset: "e".into(),
                column: "y".into(),
                dimension: 40,
            }))
            .await
            .unwrap();
        let full = p2a_core::embed(&x, 40).unwrap();
        let expected = serde_json::to_string_pretty(&serde_json::json!({
            "dimension": 40, "n_rows": 261, "matrix_sample": &full.matrix[..10]
        }))
        .unwrap();
        assert_eq!(text_of(&r), expected);
    }

    /// The handler returns 10 rows, so it must not build the full 4,001 x 1,000 embedding
    /// (which the 1 MiB ceiling refuses).
    #[test]
    fn timeseries_embed_builds_only_the_rows_it_returns() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let x = seasonal(5_000);
        let r = p2a_core::resource::with_alloc_ceiling(Some(1 << 20), || {
            rt.block_on(async {
                let server = server_with_series("e", &x).await;
                server
                    .timeseries_embed(Parameters(EmbedRequest {
                        dataset: "e".into(),
                        column: "y".into(),
                        dimension: 1_000,
                    }))
                    .await
                    .unwrap()
            })
        });
        assert_ne!(r.is_error, Some(true), "{}", text_of(&r));
        assert!(text_of(&r).contains("\"n_rows\": 4001"), "{}", text_of(&r));
    }

    /// A noisy AR(1) series.
    fn ar1(n: usize) -> Vec<f64> {
        let mut prev = 0.0;
        (0..n)
            .map(|t| {
                prev = 0.6 * prev + (((t * 37 + 11) % 23) as f64 / 23.0 - 0.5);
                prev
            })
            .collect()
    }

    /// Runs `ts_arima_fit` and `ts_arima_forecast` (horizon 5) at ARIMA(1, `d`, 1) on `column`
    /// of dataset "a" under `limits`, returning both results.
    fn arima_calls(
        rt: &tokio::runtime::Runtime,
        server: &AnalyticsServer,
        limits: &crate::tools::limits::Limits,
        column: &str,
        d: usize,
    ) -> [(&'static str, CallToolResult); 2] {
        let fit = ArimaRequest {
            dataset: "a".into(),
            column: column.into(),
            p: 1,
            d,
            q: 1,
            include_mean: None,
            include_drift: None,
        };
        let forecast = ArimaForecastRequest {
            dataset: "a".into(),
            column: column.into(),
            p: 1,
            d,
            q: 1,
            horizon: 5,
            include_mean: None,
            include_drift: None,
        };
        crate::tools::limits::with_limits(limits.clone(), || {
            [
                (
                    "ts_arima_fit",
                    rt.block_on(server.ts_arima_fit(Parameters(fit))).unwrap(),
                ),
                (
                    "ts_arima_forecast",
                    rt.block_on(server.ts_arima_forecast(Parameters(forecast)))
                        .unwrap(),
                ),
            ]
        })
    }

    /// Both ARIMA tools check the fit's work against `arima_fit_work` before they fit (an
    /// up-front cost bound, integer columns included), and fit as before within the budget.
    #[test]
    fn ts_arima_tools_refuse_a_fit_over_the_work_budget_before_fitting() {
        use crate::tools::limits::Limits;
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let server = rt.block_on(server_with_series("a", &ar1(400)));
        // 400 x ((1 + 1 + 3)^2 + 1) = 10,400 > 5,000; 5,000 / 26 = 192 observations
        let tight = Limits::hosted()
            .with_overrides(&["arima_fit_work=5000".to_string()])
            .unwrap();
        for (tool, r) in arima_calls(&rt, &server, &tight, "y", 1) {
            let t = text_of(&r);
            assert_eq!(r.is_error, Some(true), "{t}");
            assert!(
                t.starts_with(&format!(
                    "`{tool}` with p = 1, d = 1 and q = 1 on 400 observations needs 10400 \
                     (arima_fit_work"
                )) && t.contains("limit of 5000")
                    && t.contains("at most 192 observations"),
                "{t}"
            );
        }
        // An integer copy of the series is bounded too.
        let integer: Vec<i64> = ar1(400).iter().map(|v| (v * 1e6).round() as i64).collect();
        let int_server = AnalyticsServer::new();
        rt.block_on(int_server.datasets.write())
            .insert("a".to_string(), Dataset::new(df!("y" => integer).unwrap()));
        for (tool, r) in arima_calls(&rt, &int_server, &tight, "y", 1) {
            let t = text_of(&r);
            assert_eq!(r.is_error, Some(true), "{tool}: {t}");
            assert!(
                t.starts_with(&format!(
                    "`{tool}` with p = 1, d = 1 and q = 1 on 400 observations needs 10400 \
                     (arima_fit_work"
                )),
                "{t}"
            );
        }
        // A missing column keeps the core's error.
        for (_, r) in arima_calls(&rt, &server, &tight, "nope", 1) {
            assert_eq!(r.is_error, Some(true));
            assert!(
                text_of(&r).contains("not found") && !text_of(&r).contains("arima_fit_work"),
                "{}",
                text_of(&r)
            );
        }
        // The hosted budget and the off profile fit the same request.
        for limits in [Limits::hosted(), Limits::off()] {
            let [(_, fit), (_, forecast)] = arima_calls(&rt, &server, &limits, "y", 1);
            assert_ne!(fit.is_error, Some(true), "{}", text_of(&fit));
            assert!(
                text_of(&fit).contains("ARIMA(1,1,1) fitted by conditional sum of squares"),
                "{}",
                text_of(&fit)
            );
            assert_ne!(forecast.is_error, Some(true), "{}", text_of(&forecast));
            assert!(
                text_of(&forecast).contains("t+5:"),
                "{}",
                text_of(&forecast)
            );
        }
    }

    /// While the work budget is set, both ARIMA tools refuse a column holding a nonzero value
    /// below `ARIMA_MIN_MAGNITUDE` before they fit (the budget's time per unit does not hold for
    /// such values); the off profile fits it as before.
    #[test]
    fn ts_arima_tools_refuse_a_column_below_the_magnitude_floor_before_fitting() {
        use crate::tools::limits::Limits;
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let mut y = ar1(400);
        y[123] = 1e-200;
        let server = rt.block_on(server_with_series("a", &y));
        for (tool, r) in arima_calls(&rt, &server, &Limits::hosted(), "y", 0) {
            let t = text_of(&r);
            assert_eq!(r.is_error, Some(true), "{t}");
            assert!(
                t.starts_with(&format!(
                    "`{tool}` cannot fit column `y`: it holds a nonzero value of magnitude \
                     1e-200, below this server's floor of 1e-100"
                )),
                "{t}"
            );
        }
        let [(_, fit), (_, forecast)] = arima_calls(&rt, &server, &Limits::off(), "y", 0);
        assert!(
            text_of(&fit)
                .contains("ARIMA(1,0,1) with non-zero mean fitted by conditional sum of squares"),
            "{}",
            text_of(&fit)
        );
        assert!(
            text_of(&forecast).contains("t+5:"),
            "{}",
            text_of(&forecast)
        );
    }

    /// The magnitude floor reads only the values the fit uses. Polars keeps a buffer value
    /// behind every null, and arithmetic on a column runs over it too, so a null can hide a
    /// value below the floor; the fit never reads it (the core refuses a column with nulls), so
    /// the floor does not refuse for it, in either profile. A valid value below the floor in
    /// the same column is still refused.
    #[test]
    fn ts_arima_floor_ignores_the_buffer_behind_a_null() {
        use crate::tools::limits::Limits;
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let y = ar1(400);
        let column = |tiny_at: Option<usize>| {
            let values: Vec<Option<f64>> = y
                .iter()
                .enumerate()
                .map(|(i, &v)| match (i, tiny_at) {
                    (123, _) => None,
                    (i, Some(t)) if i == t => Some(-1e-150),
                    _ => Some(v),
                })
                .collect();
            // 0.0 + 1e-200 behind the null; the valid values do not change.
            let ca = &Float64Chunked::from_slice_options("y".into(), &values) + 1e-200;
            assert_eq!(ca.into_no_null_iter().nth(123), Some(1e-200));
            assert_eq!(ca.null_count(), 1);
            assert!(ca.iter().zip(&values).all(|(a, b)| a == *b));
            ca
        };
        let server = AnalyticsServer::new();
        for (name, ca) in [("a", column(None)), ("b", column(Some(200)))] {
            let df = DataFrame::new(vec![ca.into_series().into()]).unwrap();
            rt.block_on(server.datasets.write())
                .insert(name.to_string(), Dataset::new(df));
        }
        for limits in [Limits::hosted(), Limits::off()] {
            for (tool, r) in arima_calls(&rt, &server, &limits, "y", 0) {
                let t = text_of(&r);
                assert_eq!(r.is_error, Some(true), "{tool}: {t}");
                assert!(
                    t.contains("Column 'y' contains 1 missing value(s)") && !t.contains("floor"),
                    "{tool}: {t}"
                );
            }
        }
        // The same column with a valid value below the floor ("b"): refused for that value.
        let fit = ArimaRequest {
            dataset: "b".into(),
            column: "y".into(),
            p: 1,
            d: 0,
            q: 1,
            include_mean: None,
            include_drift: None,
        };
        let r = crate::tools::limits::with_limits(Limits::hosted(), || {
            rt.block_on(server.ts_arima_fit(Parameters(fit))).unwrap()
        });
        assert!(
            text_of(&r).starts_with(
                "`ts_arima_fit` cannot fit column `y`: it holds a nonzero value of magnitude \
                 1e-150, below this server's floor of 1e-100"
            ),
            "{}",
            text_of(&r)
        );
    }

    /// Dimension 0 and dimension above the series length keep their messages.
    #[tokio::test]
    async fn timeseries_embed_invalid_dimension_messages_are_unchanged() {
        let server = server_with_series("e", &seasonal(30)).await;
        for (dimension, expected) in [
            (0, "Embedding dimension must be at least 1"),
            (31, "Embedding dimension 31 exceeds series length 30"),
        ] {
            let r = server
                .timeseries_embed(Parameters(EmbedRequest {
                    dataset: "e".into(),
                    column: "y".into(),
                    dimension,
                }))
                .await
                .unwrap();
            assert_eq!(r.is_error, Some(true), "{}", text_of(&r));
            assert!(text_of(&r).contains(expected), "{}", text_of(&r));
        }
    }

    /// The case's dataset as "arima_series", and the same with a Float64 copy of `ai`
    /// ("ai_f64") as "arima_series_f64". Tests pass the case id as a literal.
    async fn server_with_case(id: &str) -> (p2a_validation::Case, AnalyticsServer) {
        let case = p2a_validation::Case::load(id);
        let df = p2a_core::DataLoader::load_csv(case.dataset_path()).unwrap();
        let copy = df
            .column("ai")
            .unwrap()
            .cast(&DataType::Float64)
            .unwrap()
            .with_name("ai_f64".into());
        let mut with_copy = df.clone();
        with_copy.with_column(copy).unwrap();
        let server = AnalyticsServer::new();
        {
            let mut datasets = server.datasets.write().await;
            datasets.insert("arima_series".to_string(), Dataset::new(df));
            datasets.insert("arima_series_f64".to_string(), Dataset::new(with_copy));
        }
        (case, server)
    }

    /// The six numbers after ':' on every line that starts with "t+".
    fn forecast_rows(text: &str) -> Vec<Vec<f64>> {
        text.lines()
            .map(str::trim)
            .filter(|l| l.starts_with("t+"))
            .map(|l| {
                let (_, numbers) = l.split_once(':').unwrap();
                numbers
                    .split_whitespace()
                    .map(|v| v.parse().unwrap())
                    .collect::<Vec<f64>>()
            })
            .filter(|row| row.len() == 6)
            .collect()
    }

    /// The second token of the first trimmed line whose first token is `name`.
    fn coef_value(text: &str, name: &str) -> f64 {
        text.lines()
            .find_map(|l| {
                let mut tokens = l.split_whitespace();
                (tokens.next() == Some(name)).then(|| tokens.next()?.parse().ok())?
            })
            .unwrap_or_else(|| panic!("no coefficient line `{name}` in\n{text}"))
    }

    /// R: forecast::forecast(Arima(a, c(1, 0, 1), method = "CSS"), h = 5) at the exact CSS
    /// optimum, through the MCP tool's text.
    #[tokio::test]
    async fn test_validate_mcp_ts_arima_forecast_intervals() {
        use p2a_validation::{Check, Tol};
        let (case, server) = server_with_case("arima_css_forecast").await;
        let req = serde_json::from_value(serde_json::json!({
            "dataset": "arima_series", "column": "a", "p": 1, "d": 0, "q": 1, "horizon": 5
        }))
        .unwrap();
        let r = server.ts_arima_forecast(Parameters(req)).await.unwrap();
        let t = text_of(&r);
        assert_ne!(r.is_error, Some(true), "{t}");
        let rows = forecast_rows(&t);
        assert_eq!(rows.len(), 5, "{t}");
        let col = |j: usize| rows.iter().map(|row| row[j]).collect::<Vec<_>>();
        Check::new(&case)
            .vector("a_101_mean_tight", &col(0), Tol::printed(4))
            .vector("a_101_se_tight", &col(1), Tol::printed(4))
            .vector("a_101_lower80_tight", &col(2), Tol::printed(4))
            .vector("a_101_upper80_tight", &col(3), Tol::printed(4))
            .vector("a_101_lower95_tight", &col(4), Tol::printed(4))
            .vector("a_101_upper95_tight", &col(5), Tol::printed(4))
            .named(
                "a_101_coef_tight",
                &["ar1", "ma1", "intercept"],
                &[
                    coef_value(&t, "ar1"),
                    coef_value(&t, "ma1"),
                    coef_value(&t, "intercept"),
                ],
                Tol::printed(6),
            )
            .finish();
    }

    /// R: forecast::Arima(dr, c(0, 1, 1), include.drift = TRUE, method = "CSS") at the exact
    /// CSS optimum, through the MCP tool's text.
    #[tokio::test]
    async fn test_validate_mcp_ts_arima_fit_drift() {
        use p2a_validation::text::labelled;
        use p2a_validation::{Check, Tol};
        let (case, server) = server_with_case("arima_css_forecast").await;
        let req = serde_json::from_value(serde_json::json!({
            "dataset": "arima_series", "column": "dr", "p": 0, "d": 1, "q": 1,
            "include_drift": true
        }))
        .unwrap();
        let r = server.ts_arima_fit(Parameters(req)).await.unwrap();
        let t = text_of(&r);
        assert_ne!(r.is_error, Some(true), "{t}");
        Check::new(&case)
            .named(
                "dr_011_drift_coef_tight",
                &["ma1", "drift"],
                &[coef_value(&t, "ma1"), coef_value(&t, "drift")],
                Tol::printed(6),
            )
            .scalar(
                "dr_011_drift_sigma2_tight",
                labelled(&t, "sigma^2:").unwrap(),
                Tol::printed(6),
            )
            .scalar(
                "dr_011_drift_loglik_tight",
                labelled(&t, "Log-likelihood (CSS):").unwrap(),
                Tol::printed(4),
            )
            .scalar(
                "dr_011_drift_aic_css_tight",
                labelled(&t, "AIC (CSS likelihood; R reports NA for CSS fits):").unwrap(),
                Tol::printed(4),
            )
            .finish();
        assert!(t.contains("Converged: yes"), "{t}");
    }

    /// The new request options are checked, missing values are a tool error with the fill
    /// hint, text is refused, and an Int64 column fits as its Float64 copy.
    #[tokio::test]
    async fn test_mcp_ts_arima_options_and_missing() {
        let (_, server) = server_with_case("arima_css_forecast").await;
        let fit = |v: serde_json::Value| {
            let req: ArimaRequest = serde_json::from_value(v).unwrap();
            server.ts_arima_fit(Parameters(req))
        };
        let r = fit(serde_json::json!({"dataset": "arima_series", "column": "c",
            "p": 0, "d": 1, "q": 1, "include_mean": true}))
        .await
        .unwrap();
        assert_eq!(r.is_error, Some(true), "{}", text_of(&r));
        assert!(text_of(&r).contains("include_mean"), "{}", text_of(&r));
        let r = fit(
            serde_json::json!({"dataset": "arima_series", "column": "dr",
            "p": 1, "d": 2, "q": 1, "include_drift": true}),
        )
        .await
        .unwrap();
        assert_eq!(r.is_error, Some(true), "{}", text_of(&r));
        assert!(text_of(&r).contains("include_drift"), "{}", text_of(&r));
        let req: ArimaForecastRequest = serde_json::from_value(serde_json::json!({
            "dataset": "arima_series", "column": "a_na", "p": 1, "d": 0, "q": 1, "horizon": 5
        }))
        .unwrap();
        let r = server.ts_arima_forecast(Parameters(req)).await.unwrap();
        let t = text_of(&r);
        assert_eq!(r.is_error, Some(true), "{t}");
        assert!(
            t.contains("Column 'a_na' contains 2 missing value(s)")
                && t.contains("munge_fill_na")
                && t.contains("impute rather than drop"),
            "{t}"
        );
        let r = fit(
            serde_json::json!({"dataset": "arima_series", "column": "lab",
            "p": 1, "d": 0, "q": 1}),
        )
        .await
        .unwrap();
        assert_eq!(r.is_error, Some(true), "{}", text_of(&r));
        assert!(text_of(&r).contains("non-numeric"), "{}", text_of(&r));
        let int = fit(
            serde_json::json!({"dataset": "arima_series", "column": "ai",
            "p": 1, "d": 0, "q": 1}),
        )
        .await
        .unwrap();
        assert_ne!(int.is_error, Some(true), "{}", text_of(&int));
        let flt = fit(
            serde_json::json!({"dataset": "arima_series_f64", "column": "ai_f64",
            "p": 1, "d": 0, "q": 1}),
        )
        .await
        .unwrap();
        assert_eq!(text_of(&int), text_of(&flt).replace("ai_f64", "ai"));
    }

    /// GARCH, AR, decompose and StructTS read Int64 columns as their Float64 copies and refuse
    /// missing values with the fill hint; cpgram drops them as R and reports the drop; the
    /// series utilities refuse text and keep NA in place; runmed refuses missing values.
    #[tokio::test]
    async fn test_mcp_timeseries_handlers_extraction() {
        let (_, server) = server_with_case("arima_css_forecast").await;
        /// Calls `tool` with `args` plus the dataset and column.
        async fn call(
            server: &AnalyticsServer,
            tool: &str,
            dataset: &str,
            mut args: serde_json::Value,
            column: &str,
        ) -> CallToolResult {
            args["dataset"] = dataset.into();
            args["column"] = column.into();
            let a = args;
            match tool {
                "ts_garch_fit" => server
                    .ts_garch_fit(Parameters(serde_json::from_value(a).unwrap()))
                    .await
                    .unwrap(),
                "timeseries_ar" => server
                    .timeseries_ar(Parameters(serde_json::from_value(a).unwrap()))
                    .await
                    .unwrap(),
                "timeseries_decompose" => server
                    .timeseries_decompose(Parameters(serde_json::from_value(a).unwrap()))
                    .await
                    .unwrap(),
                "timeseries_structts" => server
                    .timeseries_structts(Parameters(serde_json::from_value(a).unwrap()))
                    .await
                    .unwrap(),
                "timeseries_cpgram" => server
                    .timeseries_cpgram(Parameters(serde_json::from_value(a).unwrap()))
                    .await
                    .unwrap(),
                "timeseries_lag" => server
                    .timeseries_lag(Parameters(serde_json::from_value(a).unwrap()))
                    .await
                    .unwrap(),
                "timeseries_runmed" => server
                    .timeseries_runmed(Parameters(serde_json::from_value(a).unwrap()))
                    .await
                    .unwrap(),
                other => panic!("unknown tool {other}"),
            }
        }
        for (tool, args) in [
            ("ts_garch_fit", serde_json::json!({})),
            (
                "timeseries_ar",
                serde_json::json!({"method": "ols", "order": 2, "aic": false}),
            ),
            ("timeseries_decompose", serde_json::json!({"period": 12})),
            ("timeseries_structts", serde_json::json!({})),
        ] {
            let int = call(&server, tool, "arima_series", args.clone(), "ai").await;
            let flt = call(&server, tool, "arima_series_f64", args.clone(), "ai_f64").await;
            assert_eq!(int.is_error, flt.is_error, "{tool}: {}", text_of(&int));
            assert_eq!(
                text_of(&int),
                text_of(&flt).replace("ai_f64", "ai"),
                "{tool}"
            );
            let gap = call(&server, tool, "arima_series", args, "a_na").await;
            let t = text_of(&gap);
            assert_eq!(gap.is_error, Some(true), "{tool}: {t}");
            assert!(
                t.contains("Column 'a_na' contains 2 missing value(s)")
                    && t.contains("munge_fill_na")
                    && t.contains("impute rather than drop"),
                "{tool}: {t}"
            );
        }
        let r = call(
            &server,
            "timeseries_cpgram",
            "arima_series",
            serde_json::json!({}),
            "a_na",
        )
        .await;
        assert_ne!(r.is_error, Some(true), "{}", text_of(&r));
        let json: serde_json::Value = serde_json::from_str(&text_of(&r)).unwrap();
        assert_eq!(json["missing"]["n_dropped"], 2, "{json}");
        assert_eq!(json["missing"]["n_used"], 298, "{json}");
        let r = call(
            &server,
            "timeseries_lag",
            "arima_series",
            serde_json::json!({"k": 1}),
            "lab",
        )
        .await;
        assert_eq!(r.is_error, Some(true), "{}", text_of(&r));
        assert!(text_of(&r).contains("non-numeric"), "{}", text_of(&r));
        let r = call(
            &server,
            "timeseries_lag",
            "arima_series",
            serde_json::json!({"k": 1}),
            "ai_na",
        )
        .await;
        assert_ne!(r.is_error, Some(true), "{}", text_of(&r));
        let r = call(
            &server,
            "timeseries_runmed",
            "arima_series",
            serde_json::json!({"k": 3}),
            "a_na",
        )
        .await;
        assert_eq!(r.is_error, Some(true), "{}", text_of(&r));
        assert!(text_of(&r).contains("a_na"), "{}", text_of(&r));
    }
}
