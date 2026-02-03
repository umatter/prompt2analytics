//! Regression tools handlers.
//!
//! This module provides MCP tool handlers for regression analysis:
//! - OLS regression
//! - Regression diagnostics
//! - Breusch-Godfrey test for serial correlation
//! - RESET test for functional form
//! - Wald test for nested models
//! - Harvey-Collier test for linearity
//! - HAC (Newey-West) standard errors
//! - Bootstrap covariance estimation
//! - Driscoll-Kraay panel-robust SEs
//! - Quantile regression
//! - Clustered standard errors
//! - Nonlinear least squares (NLS)
//! - LOESS local regression
//! - Super smoother (supsmu)
//! - Tukey's resistant line
//! - Stepwise regression
//! - GLS (Generalized Least Squares)
//! - Smoothing splines

use rmcp::{
    ErrorData as McpError, handler::server::wrapper::Parameters, model::*, tool, tool_router,
};

use crate::server::AnalyticsServer;
use crate::tools::common::extract_column_f64;
use crate::tools::requests::regression::{
    BgTestRequest, BootstrapCovRequest, CvGlmnetRequest, DiagnosticsRequest, DriscollKraayRequest,
    GlmnetRequest, GlsRequest, HacRequest, HarveyCollierRequest, LassoRequest, LineRequest,
    LoessRequest, NlsRequest, OlsClusteredRequest, OlsRequest, QuantRegRequest, ResetTestRequest,
    RidgeRequest, SmoothSplineRequest, StepRequest, SupsmuRequest, WaldTestRequest,
};

use p2a_core::regression::{
    bg_test, gls, harvey_collier_test, quantreg_multi, reset_test, run_cv_glmnet, run_diagnostics,
    run_glmnet, run_lasso, run_line, run_loess, run_ols, run_ols_clustered, run_quantreg,
    run_ridge, run_step, run_vcov_bootstrap, run_vcov_driscoll_kraay, run_vcov_hac, smooth_spline,
    smooth_spline_predict, supsmu, wald_test, BgTestType, CorrelationStructure, CovarianceType,
    GlmnetConfig, GlmnetFamily, ResetType, SmoothSplineConfig,
};

#[tool_router(router = regression_router, vis = "pub")]
impl AnalyticsServer {
    /// Run OLS regression.
    #[tool(
        description = "Run Ordinary Least Squares (OLS) regression. Returns coefficients, standard errors, t-values, p-values, R-squared, and F-statistic."
    )]
    pub async fn regression_ols(
        &self,
        Parameters(request): Parameters<OlsRequest>,
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

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let result = match run_ols(dataset, &request.y, &x_refs, true, CovarianceType::HC1) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Regression failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run regression diagnostics.
    #[tool(
        description = "Run comprehensive regression diagnostics. Tests include: Jarque-Bera (normality), Breusch-Pagan (heteroskedasticity), Durbin-Watson (autocorrelation), VIF (multicollinearity), and condition number."
    )]
    pub async fn regression_diagnostics(
        &self,
        Parameters(request): Parameters<DiagnosticsRequest>,
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

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let result = match run_diagnostics(dataset, &request.y, &x_refs) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Diagnostics failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run Breusch-Godfrey test for serial correlation.
    #[tool(
        description = "Breusch-Godfrey test for higher-order serial correlation in regression residuals. More general than Durbin-Watson as it: (1) tests for AR(p) or MA(p) correlation, (2) allows lagged dependent variables as regressors, (3) is valid regardless of regressor properties. Returns LM statistic and p-value."
    )]
    pub async fn regression_bgtest(
        &self,
        Parameters(request): Parameters<BgTestRequest>,
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

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();
        let order = request.order.unwrap_or(1);
        let fill = request.fill.unwrap_or(0.0);

        let test_type = match request.test_type.as_deref() {
            Some(t) => BgTestType::from_str(t).unwrap_or(BgTestType::Chisq),
            None => BgTestType::Chisq,
        };

        let result = match bg_test(dataset, &request.y, &x_refs, order, test_type, fill) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Breusch-Godfrey test failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run Ramsey's RESET test for functional form.
    #[tool(
        description = "Ramsey's RESET test for functional form misspecification. Tests whether nonlinear terms (powers of fitted values) should be added to the model. Significant result suggests the linear model is misspecified and nonlinear terms may be needed."
    )]
    pub async fn regression_resettest(
        &self,
        Parameters(request): Parameters<ResetTestRequest>,
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

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();
        let powers = request.powers.unwrap_or_else(|| vec![2, 3]);

        let reset_type_enum = match request.reset_type.as_deref() {
            Some(t) => ResetType::from_str(t).unwrap_or(ResetType::Fitted),
            None => ResetType::Fitted,
        };

        let result = match reset_test(dataset, &request.y, &x_refs, &powers, reset_type_enum) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "RESET test failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Wald test for comparing nested linear models.
    #[tool(
        description = "Wald test (F-test) for comparing nested linear models. Tests whether variables excluded from the restricted model are jointly significant. Common uses: testing joint significance of multiple coefficients, testing nested model hypotheses, comparing model specifications."
    )]
    pub async fn regression_waldtest(
        &self,
        Parameters(request): Parameters<WaldTestRequest>,
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

        let x_unrestricted_refs: Vec<&str> =
            request.x_unrestricted.iter().map(|s| s.as_str()).collect();
        let x_restricted_refs: Vec<&str> =
            request.x_restricted.iter().map(|s| s.as_str()).collect();
        let use_f_test = request.use_f_test.unwrap_or(true);

        let result = match wald_test(
            dataset,
            &request.y,
            &x_unrestricted_refs,
            &x_restricted_refs,
            use_f_test,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Wald test failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Harvey-Collier test for linearity using recursive residuals.
    #[tool(
        description = "Harvey-Collier test for detecting departure from linearity. Uses recursive residuals to test whether the mean of one-step-ahead forecast errors differs from zero. A significant result suggests convex or concave functional misspecification - the linear model may need polynomial terms. Equivalent to R's lmtest::harvtest()."
    )]
    pub async fn regression_harvtest(
        &self,
        Parameters(request): Parameters<HarveyCollierRequest>,
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

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let result = match harvey_collier_test(dataset, &request.y, &x_refs) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Harvey-Collier test failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// HAC (Newey-West) standard errors for time series regression.
    #[tool(
        description = "Compute HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors using the Newey-West estimator. Essential for time series regression where errors may be both heteroskedastic and autocorrelated. Supports multiple kernel functions and automatic bandwidth selection."
    )]
    pub async fn regression_hac(
        &self,
        Parameters(request): Parameters<HacRequest>,
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

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();
        let prewhiten = request.prewhiten.unwrap_or(false);

        let result = match run_vcov_hac(
            dataset,
            &request.y,
            &x_refs,
            request.bandwidth,
            request.kernel.as_deref(),
            prewhiten,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "HAC estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Bootstrap covariance estimation (vcovBS).
    #[tool(
        description = "Compute bootstrap covariance matrix and standard errors for OLS regression. Supports pairs bootstrap (most robust, resamples observations), residual bootstrap (assumes homoskedasticity), and wild bootstrap (robust to heteroskedasticity). Useful when asymptotic standard errors may be unreliable."
    )]
    pub async fn regression_bootstrap_cov(
        &self,
        Parameters(request): Parameters<BootstrapCovRequest>,
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

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let result = match run_vcov_bootstrap(
            dataset,
            &request.y,
            &x_refs,
            request.n_boot,
            request.bootstrap_type.as_deref(),
            request.seed,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Bootstrap covariance estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Driscoll-Kraay panel-robust standard errors.
    #[tool(
        description = "Compute Driscoll-Kraay (1998) panel-robust standard errors. Robust to arbitrary cross-sectional correlation (spatial dependence) and serial correlation in panel data. Aggregates score vectors by time period and applies Newey-West HAC correction. Best for panels with large T (many time periods). Returns coefficient estimates with panel-robust SEs, t-stats, and p-values."
    )]
    pub async fn regression_driscoll_kraay(
        &self,
        Parameters(request): Parameters<DriscollKraayRequest>,
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

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let result = match run_vcov_driscoll_kraay(
            dataset,
            &request.y,
            &x_refs,
            &request.time_col,
            request.bandwidth,
            request.kernel.as_deref(),
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Driscoll-Kraay estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Quantile regression.
    #[tool(
        description = "Run quantile regression to estimate conditional quantiles instead of conditional means. Useful when the relationship varies across the distribution, or when the error distribution is non-Gaussian. Can estimate single quantile (tau) or multiple quantiles simultaneously."
    )]
    pub async fn regression_quantreg(
        &self,
        Parameters(request): Parameters<QuantRegRequest>,
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

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        // Handle multiple quantiles or single quantile
        if let Some(taus) = &request.taus {
            // Multiple quantiles
            let results = match quantreg_multi(dataset, &request.y, &x_refs, taus) {
                Ok(r) => r,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Quantile regression failed: {}",
                        e
                    ))]));
                }
            };

            // Format all results
            let output: String = results
                .iter()
                .map(|r| format!("{}", r))
                .collect::<Vec<_>>()
                .join("\n---\n");

            Ok(CallToolResult::success(vec![Content::text(output)]))
        } else {
            // Single quantile
            let tau = request.tau.unwrap_or(0.5);
            let result = match run_quantreg(dataset, &request.y, &x_refs, tau) {
                Ok(r) => r,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Quantile regression failed: {}",
                        e
                    ))]));
                }
            };

            Ok(CallToolResult::success(vec![Content::text(
                result.to_string(),
            )]))
        }
    }

    /// Run OLS with clustered standard errors.
    #[tool(
        description = "Run OLS regression with clustered standard errors. Supports one-way (firm, state) or two-way (firm + time) clustering. Essential for panel data with correlated errors."
    )]
    pub async fn regression_clustered(
        &self,
        Parameters(request): Parameters<OlsClusteredRequest>,
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

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let result = match run_ols_clustered(
            dataset,
            &request.y,
            &x_refs,
            &request.cluster1,
            request.cluster2.as_deref(),
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Clustered regression failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run nonlinear least squares regression.
    #[tool(
        description = "Fit a nonlinear regression model using Levenberg-Marquardt algorithm. Supports common models: exponential decay/growth, Michaelis-Menten kinetics, logistic growth, power law, asymptotic. Returns parameter estimates, standard errors, t-values, and convergence info."
    )]
    pub async fn regression_nls(
        &self,
        Parameters(request): Parameters<NlsRequest>,
    ) -> Result<CallToolResult, McpError> {
        use ndarray::Array1;
        use p2a_core::linalg::design::DesignMatrix;
        use p2a_core::regression::{
            model_asymptotic, model_exponential_decay, model_exponential_growth,
            model_logistic_growth, model_michaelis_menten, model_power, nls, NlsAlgorithm,
            NlsConfig,
        };

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

        // Extract x and y columns
        let x = match DesignMatrix::extract_column(dataset.df(), &request.x) {
            Ok(col) => col,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to extract X column '{}': {:?}",
                    request.x, e
                ))]));
            }
        };

        let y = match DesignMatrix::extract_column(dataset.df(), &request.y) {
            Ok(col) => col,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to extract Y column '{}': {:?}",
                    request.y, e
                ))]));
            }
        };

        // Parse model type and get function + param names
        let (model_fn, param_names): (fn(&Array1<f64>, &Array1<f64>) -> f64, Vec<&str>) =
            match request.model.to_lowercase().as_str() {
                "exponential_decay" | "exp_decay" => {
                    if request.start.len() != 3 {
                        return Ok(CallToolResult::error(vec![Content::text(
                            "exponential_decay requires 3 starting values: [a, b, c] for y = a*exp(-b*x) + c",
                        )]));
                    }
                    (model_exponential_decay, vec!["a", "b", "c"])
                }
                "exponential_growth" | "exp_growth" => {
                    if request.start.len() != 2 {
                        return Ok(CallToolResult::error(vec![Content::text(
                            "exponential_growth requires 2 starting values: [a, b] for y = a*exp(b*x)",
                        )]));
                    }
                    (model_exponential_growth, vec!["a", "b"])
                }
                "michaelis_menten" | "mm" => {
                    if request.start.len() != 2 {
                        return Ok(CallToolResult::error(vec![Content::text(
                            "michaelis_menten requires 2 starting values: [Vmax, Km] for y = Vmax*x/(Km+x)",
                        )]));
                    }
                    (model_michaelis_menten, vec!["Vmax", "Km"])
                }
                "logistic" | "logistic_growth" => {
                    if request.start.len() != 3 {
                        return Ok(CallToolResult::error(vec![Content::text(
                            "logistic requires 3 starting values: [K, r, x0] for y = K/(1+exp(-r*(x-x0)))",
                        )]));
                    }
                    (model_logistic_growth, vec!["K", "r", "x0"])
                }
                "power" => {
                    if request.start.len() != 2 {
                        return Ok(CallToolResult::error(vec![Content::text(
                            "power requires 2 starting values: [a, b] for y = a*x^b",
                        )]));
                    }
                    (model_power, vec!["a", "b"])
                }
                "asymptotic" => {
                    if request.start.len() != 3 {
                        return Ok(CallToolResult::error(vec![Content::text(
                            "asymptotic requires 3 starting values: [a, b, c] for y = a - b*exp(-c*x)",
                        )]));
                    }
                    (model_asymptotic, vec!["a", "b", "c"])
                }
                _ => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Unknown model '{}'. Supported: exponential_decay, exponential_growth, michaelis_menten, logistic, power, asymptotic",
                        request.model
                    ))]));
                }
            };

        // Parse algorithm
        let algorithm = match request.algorithm.as_deref() {
            Some("gauss_newton") | Some("gn") => NlsAlgorithm::GaussNewton,
            Some("levenberg_marquardt") | Some("lm") | None => NlsAlgorithm::LevenbergMarquardt,
            Some(other) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown algorithm '{}'. Supported: levenberg_marquardt (default), gauss_newton",
                    other
                ))]));
            }
        };

        let config = NlsConfig {
            algorithm,
            ..Default::default()
        };

        let start = Array1::from_vec(request.start);

        let result = match nls(&x, &y, model_fn, &start, &param_names, config) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "NLS fitting failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run LOESS (local polynomial regression) smoothing.
    #[tool(
        description = "Fit a LOESS (LOcally Estimated Scatterplot Smoothing) model. LOESS fits local polynomial regressions at each point, weighted by distance from the target point using tricubic weights. Useful for non-parametric trend estimation, data smoothing, and exploring nonlinear relationships. Returns fitted values, residuals, R-squared, and equivalent number of parameters."
    )]
    pub async fn regression_loess(
        &self,
        Parameters(request): Parameters<LoessRequest>,
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

        let span = request.span.unwrap_or(0.75);
        let degree = request.degree.unwrap_or(2);
        let robust = request.robust.unwrap_or(false);

        // Validate parameters
        if span <= 0.0 {
            return Ok(CallToolResult::error(vec![Content::text(
                "span must be positive".to_string(),
            )]));
        }
        if degree > 2 {
            return Ok(CallToolResult::error(vec![Content::text(
                "degree must be 0, 1, or 2".to_string(),
            )]));
        }

        let result = match run_loess(dataset, &request.y, &request.x, span, degree, robust) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "LOESS fitting failed: {}",
                    e
                ))]));
            }
        };

        // Build JSON output for LLM
        let output = serde_json::json!({
            "method": "LOESS",
            "span": result.span,
            "degree": result.degree,
            "family": if result.robust { "symmetric" } else { "gaussian" },
            "n_obs": result.n_obs,
            "equivalent_number_parameters": result.enp,
            "residual_sum_squares": result.rss,
            "residual_standard_error": result.residual_se,
            "r_squared": result.r_squared,
            "robust_iterations": result.robust_iterations,
            "fitted_values": result.fitted,
            "residuals": result.residuals,
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| result.to_string()),
        )]))
    }

    /// Fit Friedman's SuperSmoother.
    #[tool(
        description = "Friedman's SuperSmoother - a variable-bandwidth smoother that locally adapts to the data. Uses cross-validation to select optimal span at each point. Returns smoothed y values and can handle periodic data."
    )]
    pub async fn regression_supsmu(
        &self,
        Parameters(request): Parameters<SupsmuRequest>,
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

        // Extract x column
        let x_col = match df.column(&request.x) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "X column '{}' not found: {}",
                    request.x, e
                ))]));
            }
        };

        let x_values: Vec<f64> = match x_col.cast(&DataType::Float64) {
            Ok(c) => match c.f64() {
                Ok(f) => f.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "X column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cannot convert X to numeric: {}",
                    e
                ))]));
            }
        };

        // Extract y column
        let y_col = match df.column(&request.y) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Y column '{}' not found: {}",
                    request.y, e
                ))]));
            }
        };

        let y_values: Vec<f64> = match y_col.cast(&DataType::Float64) {
            Ok(c) => match c.f64() {
                Ok(f) => f.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Y column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cannot convert Y to numeric: {}",
                    e
                ))]));
            }
        };

        // Extract optional weights
        let weights: Option<Vec<f64>> = if let Some(ref wt_col) = request.weights {
            let w_col = match df.column(wt_col) {
                Ok(c) => c,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Weight column '{}' not found: {}",
                        wt_col, e
                    ))]));
                }
            };
            match w_col.cast(&DataType::Float64) {
                Ok(c) => match c.f64() {
                    Ok(f) => Some(f.into_iter().map(|v| v.unwrap_or(1.0)).collect()),
                    Err(_) => None,
                },
                Err(_) => None,
            }
        } else {
            None
        };

        let result = match supsmu(
            &x_values,
            &y_values,
            weights.as_deref(),
            request.span,
            request.periodic.unwrap_or(false),
            request.bass.unwrap_or(0.0),
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "SuperSmoother failed: {}",
                    e
                ))]));
            }
        };

        // Format output
        let output = serde_json::json!({
            "n_unique_points": result.n,
            "bass": result.bass,
            "periodic": result.periodic,
            "x_sample": &result.x[..result.n.min(20)],
            "y_smoothed_sample": &result.y[..result.n.min(20)],
            "note": if result.n > 20 { Some(format!("Showing first 20 of {} points", result.n)) } else { None::<String> }
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| format!("{:?}", result)),
        )]))
    }

    /// Fit Tukey's resistant line.
    #[tool(
        description = "Tukey's resistant line - robust regression using medians instead of means, making it resistant to outliers. Divides data into three groups and uses group medians to determine slope and intercept. Returns coefficients, fitted values, and residuals."
    )]
    pub async fn regression_line(
        &self,
        Parameters(request): Parameters<LineRequest>,
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

        // Extract x column
        let x_col = match df.column(&request.x) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "X column '{}' not found: {}",
                    request.x, e
                ))]));
            }
        };

        let x_values: Vec<f64> = match x_col.cast(&DataType::Float64) {
            Ok(c) => match c.f64() {
                Ok(f) => f.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "X column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cannot convert X to numeric: {}",
                    e
                ))]));
            }
        };

        // Extract y column
        let y_col = match df.column(&request.y) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Y column '{}' not found: {}",
                    request.y, e
                ))]));
            }
        };

        let y_values: Vec<f64> = match y_col.cast(&DataType::Float64) {
            Ok(c) => match c.f64() {
                Ok(f) => f.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Y column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cannot convert Y to numeric: {}",
                    e
                ))]));
            }
        };

        let result = match run_line(&x_values, &y_values, request.iter) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Tukey's line failed: {}",
                    e
                ))]));
            }
        };

        // Format output
        let output = serde_json::json!({
            "n_observations": result.n,
            "iterations": result.iter,
            "coefficients": {
                "intercept": result.intercept,
                "slope": result.slope
            },
            "fitted_sample": &result.fitted[..result.n.min(20)],
            "residuals_sample": &result.residuals[..result.n.min(20)],
            "note": if result.n > 20 { Some(format!("Showing first 20 of {} values", result.n)) } else { None::<String> }
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| format!("{:?}", result)),
        )]))
    }

    /// Stepwise regression model selection.
    #[tool(
        description = "Stepwise regression model selection using AIC or BIC. Can perform forward selection, backward elimination, or both. Returns the best model and selection history."
    )]
    pub async fn regression_step(
        &self,
        Parameters(request): Parameters<StepRequest>,
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

        let direction_str = request.direction.as_deref().unwrap_or("both");
        let use_bic = request.criterion.as_deref() == Some("bic");

        // For stepwise, scope_lower is empty (no forced variables), scope_upper is all predictors
        let scope_lower: Vec<&str> = vec![];
        let scope_upper: Vec<&str> = request.predictors.iter().map(|s| s.as_str()).collect();

        let result = match run_step(
            dataset,
            &request.response,
            &scope_lower,
            &scope_upper,
            direction_str,
            use_bic,
            true,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Stepwise selection failed: {}",
                    e
                ))]));
            }
        };

        let output = serde_json::json!({
            "final_model": {
                "variables": result.final_variables,
                "criterion": result.criterion_name,
                "k": result.k
            },
            "initial_variables": result.initial_variables,
            "direction": format!("{}", result.direction),
            "n_steps": result.n_steps,
            "steps": result.steps.iter().map(|s| serde_json::json!({
                "step": s.step,
                "action": s.action,
                "variables": s.variables,
                "df": s.df,
                "rss": s.rss,
                "criterion": s.criterion
            })).collect::<Vec<_>>()
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap(),
        )]))
    }

    /// Fit a Generalized Least Squares model.
    #[tool(
        description = "Fit a Generalized Least Squares (GLS) regression model. GLS extends OLS to handle correlated or heteroscedastic errors. Correlation structures: 'ar1' (autoregressive), 'compound_symmetry' (equal correlation), 'identity' (OLS). For AR(1), rho can be specified or auto-estimated from OLS residuals. Returns coefficients, standard errors, t-values, p-values, and model fit statistics."
    )]
    pub async fn regression_gls(
        &self,
        Parameters(request): Parameters<GlsRequest>,
    ) -> Result<CallToolResult, McpError> {
        use ndarray::Array1;
        use p2a_core::linalg::design::DesignMatrix;

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

        // Extract y column
        let y = match extract_column_f64(dataset, &request.y) {
            Ok(v) => Array1::from_vec(v),
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        // Build design matrix
        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();
        let intercept = request.intercept.unwrap_or(true);

        let dm = match DesignMatrix::from_dataframe(dataset.df(), &x_refs, intercept) {
            Ok(dm) => dm,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to build design matrix: {}",
                    e
                ))]));
            }
        };

        let x = dm.view();

        // Determine correlation structure
        let corr_type = request.correlation.as_deref().unwrap_or("ar1");

        let correlation = match corr_type {
            "identity" | "ols" => CorrelationStructure::Identity,
            "compound_symmetry" | "cs" => {
                let rho = request.rho.unwrap_or(0.5);
                CorrelationStructure::CompoundSymmetry { rho }
            }
            "ar1" | "ar" => {
                // If rho not provided, auto-estimate from OLS residuals
                let rho = if let Some(r) = request.rho {
                    r
                } else {
                    // Run OLS to get residuals for AR(1) estimation
                    match gls(&y.view(), &x, CorrelationStructure::Identity) {
                        Ok(ols_result) => {
                            // Estimate rho from residuals
                            let residuals = &ols_result.residuals;
                            let n = residuals.len();
                            if n > 1 {
                                let mut sum_prod = 0.0;
                                let mut sum_sq = 0.0;
                                for i in 1..n {
                                    sum_prod += residuals[i] * residuals[i - 1];
                                    sum_sq += residuals[i - 1].powi(2);
                                }
                                (sum_prod / sum_sq).clamp(-0.99, 0.99)
                            } else {
                                0.0
                            }
                        }
                        Err(_) => 0.5, // fallback
                    }
                };
                CorrelationStructure::AR1 { rho }
            }
            other => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown correlation structure '{}'. Supported: ar1, compound_symmetry, identity.",
                    other
                ))]));
            }
        };

        match gls(&y.view(), &x, correlation) {
            Ok(result) => {
                // Build coefficient names
                let mut coef_names: Vec<String> = if intercept {
                    vec!["(Intercept)".to_string()]
                } else {
                    vec![]
                };
                coef_names.extend(request.x.iter().cloned());

                let mut coef_table: Vec<serde_json::Value> = Vec::new();
                for (i, name) in coef_names.iter().enumerate() {
                    coef_table.push(serde_json::json!({
                        "term": name,
                        "estimate": result.coefficients[i],
                        "std_error": result.std_errors[i],
                        "t_value": result.t_values[i],
                        "p_value": result.p_values[i]
                    }));
                }

                let json_output = serde_json::json!({
                    "model": "Generalized Least Squares",
                    "correlation": result.correlation,
                    "correlation_param": result.correlation_param,
                    "coefficients": coef_table,
                    "sigma": result.sigma,
                    "r_squared": result.r_squared,
                    "adj_r_squared": result.adj_r_squared,
                    "log_likelihood": result.log_likelihood,
                    "aic": result.aic,
                    "bic": result.bic,
                    "n_obs": result.n_obs,
                    "df_residual": result.df_residual
                });

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&json_output).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "GLS regression failed: {}",
                e
            ))])),
        }
    }

    /// Fit a smoothing spline.
    #[tool(
        description = "Fit a smoothing spline to data. Smoothing splines balance goodness-of-fit against smoothness using a penalty on curvature. The smoothing parameter (spar) or degrees of freedom (df) controls the tradeoff. If neither is specified, uses generalized cross-validation (GCV) to automatically select optimal smoothing. Returns fitted values, effective degrees of freedom, and GCV score."
    )]
    pub async fn regression_smooth_spline(
        &self,
        Parameters(request): Parameters<SmoothSplineRequest>,
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

        let x = match extract_column_f64(dataset, &request.x) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        let y = match extract_column_f64(dataset, &request.y) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        let config = SmoothSplineConfig {
            spar: request.spar,
            df: request.df,
            ..Default::default()
        };

        match smooth_spline(&x, &y, config) {
            Ok(result) => {
                // Get predictions at xout if specified, otherwise at data points
                let (pred_x, pred_y) = if let Some(ref xout) = request.xout {
                    match smooth_spline_predict(&result, xout) {
                        Ok(yout) => (xout.clone(), yout),
                        Err(e) => {
                            return Ok(CallToolResult::error(vec![Content::text(format!(
                                "Prediction failed: {}",
                                e
                            ))]));
                        }
                    }
                } else {
                    (result.x.clone(), result.y.clone())
                };

                let json_output = serde_json::json!({
                    "model": "Smoothing Spline",
                    "x": pred_x,
                    "fitted": pred_y,
                    "spar": result.spar,
                    "df": result.df,
                    "lambda": result.lambda,
                    "cv_crit": result.cv_crit,
                    "n_obs": result.n_obs,
                    "n_knots": result.n_knots
                });

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&json_output).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Smooth spline failed: {}",
                e
            ))])),
        }
    }

    /// Run elastic net/lasso/ridge regression (glmnet).
    #[tool(
        description = "Run regularized regression with elastic net penalty (glmnet). Combines L1 (lasso) and L2 (ridge) penalties. Set alpha=1 for lasso, alpha=0 for ridge, or 0<alpha<1 for elastic net. Returns coefficient path along lambda sequence."
    )]
    pub async fn regression_glmnet(
        &self,
        Parameters(request): Parameters<GlmnetRequest>,
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

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let family = match request.family.as_deref() {
            Some("binomial") => GlmnetFamily::Binomial,
            _ => GlmnetFamily::Gaussian,
        };

        let config = GlmnetConfig {
            alpha: request.alpha.unwrap_or(1.0),
            lambda: request.lambda,
            nlambda: request.nlambda.unwrap_or(100),
            standardize: request.standardize.unwrap_or(true),
            family,
            ..Default::default()
        };

        match run_glmnet(dataset, &request.y, &x_refs, &config) {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(
                result.to_string(),
            )])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Glmnet failed: {}",
                e
            ))])),
        }
    }

    /// Run cross-validated glmnet for optimal lambda selection.
    #[tool(
        description = "Run cross-validated elastic net/lasso/ridge regression to select optimal lambda. Returns lambda.min (minimum CV error) and lambda.1se (most regularized within 1 SE of minimum)."
    )]
    pub async fn regression_cv_glmnet(
        &self,
        Parameters(request): Parameters<CvGlmnetRequest>,
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

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let family = match request.family.as_deref() {
            Some("binomial") => GlmnetFamily::Binomial,
            _ => GlmnetFamily::Gaussian,
        };

        let config = GlmnetConfig {
            alpha: request.alpha.unwrap_or(1.0),
            nlambda: request.nlambda.unwrap_or(100),
            standardize: request.standardize.unwrap_or(true),
            family,
            ..Default::default()
        };

        let nfolds = request.nfolds.unwrap_or(10);

        match run_cv_glmnet(dataset, &request.y, &x_refs, &config, nfolds, request.seed) {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(
                result.to_string(),
            )])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "CV Glmnet failed: {}",
                e
            ))])),
        }
    }

    /// Run ridge regression (L2 penalty).
    #[tool(
        description = "Run ridge regression (L2-penalized linear regression). Shortcut for glmnet with alpha=0. Useful when predictors are highly correlated."
    )]
    pub async fn regression_ridge(
        &self,
        Parameters(request): Parameters<RidgeRequest>,
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

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        match run_ridge(dataset, &request.y, &x_refs, request.lambda) {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(
                result.to_string(),
            )])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Ridge regression failed: {}",
                e
            ))])),
        }
    }

    /// Run lasso regression (L1 penalty).
    #[tool(
        description = "Run lasso regression (L1-penalized linear regression). Shortcut for glmnet with alpha=1. Performs automatic feature selection by shrinking some coefficients to zero."
    )]
    pub async fn regression_lasso(
        &self,
        Parameters(request): Parameters<LassoRequest>,
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

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        match run_lasso(dataset, &request.y, &x_refs, request.lambda) {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(
                result.to_string(),
            )])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Lasso regression failed: {}",
                e
            ))])),
        }
    }
}
