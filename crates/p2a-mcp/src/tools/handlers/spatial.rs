//! Spatial econometrics tool handlers.
//!
//! This module defines spatial econometrics tool handlers using the `#[tool_router(router = spatial_router)]` pattern.

use rmcp::{
    ErrorData as McpError,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content},
    tool, tool_router,
};

use crate::server::AnalyticsServer;
use crate::tools::requests::spatial::*;

use p2a_core::{
    // Traits
    LinearEstimator,
    MoranAlternative,
    // Spatial weights and neighbors
    Neighbors,
    SarConfig,
    SemConfig,
    SpatialErrorType,
    SpatialPanelEffect,
    SpatialPanelModel,
    SpatialProbitConfig,
    SpatialWeights,
    SpgmConfig,
    SpgmMethod,
    SphetConfig,
    SphetModel,
    SphetSE,
    SpmlConfig,
    WeightStyle,
    // Moran's I test
    moran_test,
    // Regression
    regression::{CovarianceType, run_ols},
    // Spatial regression models
    run_sar,
    // Spatial probit
    run_sar_probit,
    run_sem,
    run_sem_probit,
    run_spgm,
    run_sphet,
    // Spatial panel
    run_spml,
    // Spatial LM tests
    spatial_lm_tests,
};

#[tool_router(router = spatial_router, vis = "pub")]
impl AnalyticsServer {
    /// Create spatial neighbors and weights from coordinates.
    #[tool(
        description = "Create spatial neighbors and weights matrix from coordinate data. Supports k-nearest neighbors (knn), distance-based neighbors, and great-circle distance for lon/lat coordinates. The resulting weights are stored for use with spatial tests and models (Moran's I, SAR, SEM). Equivalent to R's spdep::knearneigh() + nb2listw() or dnearneigh() + nb2listw()."
    )]
    pub async fn spatial_neighbors(
        &self,
        Parameters(request): Parameters<SpatialNeighborsRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let df = dataset.df();

        // Extract coordinates
        let x_series = match df.column(&request.x_coord) {
            Ok(s) => s,
            Err(_) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Coordinate column '{}' not found in dataset.",
                    request.x_coord
                ))]));
            }
        };
        let y_series = match df.column(&request.y_coord) {
            Ok(s) => s,
            Err(_) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Coordinate column '{}' not found in dataset.",
                    request.y_coord
                ))]));
            }
        };

        let x_vals: Vec<f64> = match x_series.f64() {
            Ok(ca) => ca.into_no_null_iter().collect(),
            Err(_) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Column '{}' must be numeric.",
                    request.x_coord
                ))]));
            }
        };
        let y_vals: Vec<f64> = match y_series.f64() {
            Ok(ca) => ca.into_no_null_iter().collect(),
            Err(_) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Column '{}' must be numeric.",
                    request.y_coord
                ))]));
            }
        };

        let coords: Vec<(f64, f64)> = x_vals.into_iter().zip(y_vals.into_iter()).collect();
        let n = coords.len();

        // Create neighbors based on method
        let method = request.method.as_deref().unwrap_or("knn");
        let neighbors = match method {
            "knn" => {
                let k = request.k.unwrap_or(5);
                Neighbors::from_knn(&coords, k)
            }
            "distance" => {
                let d_max = request.d_max.unwrap_or(1.0);
                let d_min = request.d_min.unwrap_or(0.0);
                Neighbors::from_distance(&coords, d_min, d_max)
            }
            "distance_longlat" | "longlat" => {
                let d_max = request.d_max.unwrap_or(100.0); // km
                let d_min = request.d_min.unwrap_or(0.0);
                Neighbors::from_distance_longlat(&coords, d_min, d_max)
            }
            other => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown neighbor method '{}'. Use 'knn', 'distance', or 'distance_longlat'.",
                    other
                ))]));
            }
        };

        // Determine weight style
        let style = match request.style.as_deref() {
            Some("B") | Some("binary") => WeightStyle::Binary,
            Some("C") | Some("global") => WeightStyle::GlobalStd,
            Some("U") | Some("unstandardized") => WeightStyle::Binary, // Closest approximation
            _ => WeightStyle::RowStd,                                  // Default: row-standardized
        };

        // Create spatial weights
        let weights = SpatialWeights::from_neighbors(&neighbors, style);

        // Get stats from neighbors before we lose access to it
        let avg_neighbors = neighbors.avg_neighbors();
        let has_isolates = neighbors.has_isolates();
        let n_isolates = neighbors.isolates().len();
        let is_symmetric = neighbors.is_symmetric();

        // Store the weights
        let weights_name = request
            .weights_name
            .unwrap_or_else(|| format!("{}_weights", request.dataset));

        drop(datasets);
        let mut spatial_weights = self.spatial_weights.write().await;
        spatial_weights.insert(weights_name.clone(), weights);

        let mut output = format!(
            "Spatial Weights Created: '{}'\n\
             ========================\n\
             Observations: {}\n\
             Method: {}\n\
             Style: {:?}\n\
             Average neighbors: {:.2}\n\
             Symmetric: {}\n",
            weights_name, n, method, style, avg_neighbors, is_symmetric
        );

        if has_isolates {
            output.push_str(&format!(
                "\nWarning: {} observation(s) have no neighbors (isolates).\n",
                n_isolates
            ));
        }

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    /// Run Moran's I test for spatial autocorrelation.
    #[tool(
        description = "Run Moran's I test for spatial autocorrelation on a variable. Tests whether nearby observations tend to have similar values (positive autocorrelation) or dissimilar values (negative autocorrelation). Requires spatial weights created with 'spatial_neighbors'. Equivalent to R's spdep::moran.test()."
    )]
    pub async fn moran_test(
        &self,
        Parameters(request): Parameters<MoranTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let spatial_weights = self.spatial_weights.read().await;
        let listw = match spatial_weights.get(&request.weights) {
            Some(w) => w,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Spatial weights '{}' not found. Use 'spatial_neighbors' first to create weights.",
                    request.weights
                ))]));
            }
        };

        // Extract variable
        let df = dataset.df();
        let var_series = match df.column(&request.variable) {
            Ok(s) => s,
            Err(_) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Variable '{}' not found in dataset.",
                    request.variable
                ))]));
            }
        };
        let x_vec: Vec<f64> = match var_series.f64() {
            Ok(ca) => ca.into_no_null_iter().collect(),
            Err(_) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Variable '{}' must be numeric.",
                    request.variable
                ))]));
            }
        };
        let x = ndarray::Array1::from_vec(x_vec);

        // Parse alternative
        let alternative = match request.alternative.as_deref() {
            Some("less") | Some("negative") => MoranAlternative::Less,
            Some("two.sided") | Some("two-sided") => MoranAlternative::TwoSided,
            _ => MoranAlternative::Greater, // Default: positive autocorrelation
        };

        match moran_test(&x, listw, alternative) {
            Ok(result) => {
                let output = format!(
                    "Moran's I Test for Spatial Autocorrelation\n\
                     ==========================================\n\
                     Variable: {}\n\
                     Moran I statistic: {:.6}\n\
                     Expected I: {:.6}\n\
                     Variance: {:.6}\n\
                     Z-score: {:.4}\n\
                     P-value: {:.6}\n\
                     Alternative: {:?}\n\n\
                     Interpretation:\n\
                     - I > E[I]: positive spatial autocorrelation (clustering)\n\
                     - I < E[I]: negative spatial autocorrelation (dispersion)\n\
                     - I ≈ E[I]: random spatial pattern",
                    request.variable,
                    result.statistic,
                    result.expectation,
                    result.variance,
                    result.z_score,
                    result.p_value,
                    alternative
                );
                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Moran's I test failed: {}",
                e
            ))])),
        }
    }

    /// Run spatial LM tests to choose between SAR and SEM models.
    #[tool(
        description = "Run Lagrange Multiplier tests for spatial dependence. Tests for spatial lag (SAR) vs spatial error (SEM) dependence in OLS residuals. Includes robust versions that account for the presence of the other form of spatial dependence. Use to decide between SAR and SEM models. Equivalent to R's spdep::lm.LMtests()."
    )]
    pub async fn spatial_lm_tests_tool(
        &self,
        Parameters(request): Parameters<SpatialLmTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let mut spatial_weights_map = self.spatial_weights.write().await;
        let listw = match spatial_weights_map.get_mut(&request.weights) {
            Some(w) => w,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Spatial weights '{}' not found. Use 'spatial_neighbors' first to create weights.",
                    request.weights
                ))]));
            }
        };

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        // Run OLS to get residuals
        let ols_result = match run_ols(dataset, &request.y, &x_refs, true, CovarianceType::Standard)
        {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "OLS estimation failed: {}",
                    e
                ))]));
            }
        };

        // Build design matrix with intercept
        let df = dataset.df();
        let n = df.height();
        let k = request.x.len() + 1; // +1 for intercept

        let mut x_mat = ndarray::Array2::<f64>::zeros((n, k));
        for i in 0..n {
            x_mat[[i, 0]] = 1.0; // Intercept
        }
        for (j, col_name) in request.x.iter().enumerate() {
            let col = match df.column(col_name) {
                Ok(c) => c,
                Err(_) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Column '{}' not found.",
                        col_name
                    ))]));
                }
            };
            let col_f64 = match col.f64() {
                Ok(c) => c,
                Err(_) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Column '{}' must be numeric.",
                        col_name
                    ))]));
                }
            };
            for (i, val) in col_f64.into_no_null_iter().enumerate() {
                x_mat[[i, j + 1]] = val;
            }
        }

        match spatial_lm_tests(ols_result.residuals(), &x_mat, listw) {
            Ok(result) => {
                let output = format!(
                    "Spatial Lagrange Multiplier Tests\n\
                     ==================================\n\
                     Dependent variable: {}\n\n\
                     LM tests for spatial dependence:\n\
                     --------------------------------\n\
                     LM-Lag (ρ):     statistic = {:.4}, df = {}, p-value = {:.6}\n\
                     LM-Error (λ):   statistic = {:.4}, df = {}, p-value = {:.6}\n\n\
                     Robust LM tests (controlling for the other):\n\
                     --------------------------------------------\n\
                     RLM-Lag:        statistic = {:.4}, df = {}, p-value = {:.6}\n\
                     RLM-Error:      statistic = {:.4}, df = {}, p-value = {:.6}\n\n\
                     Joint SARMA test:\n\
                     -----------------\n\
                     LM-SARMA:       statistic = {:.4}, df = {}, p-value = {:.6}\n\n\
                     Decision Rule:\n\
                     - Both LM tests significant: use robust versions\n\
                     - Only RLM-Lag significant: use SAR model (lagsarlm)\n\
                     - Only RLM-Error significant: use SEM model (errorsarlm)\n\
                     - Both robust tests significant: use SARAR model",
                    request.y,
                    result.lm_lag.statistic,
                    result.lm_lag.df,
                    result.lm_lag.p_value,
                    result.lm_error.statistic,
                    result.lm_error.df,
                    result.lm_error.p_value,
                    result.rlm_lag.statistic,
                    result.rlm_lag.df,
                    result.rlm_lag.p_value,
                    result.rlm_error.statistic,
                    result.rlm_error.df,
                    result.rlm_error.p_value,
                    result.lm_sarma.statistic,
                    result.lm_sarma.df,
                    result.lm_sarma.p_value
                );
                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Spatial LM tests failed: {}",
                e
            ))])),
        }
    }

    /// Run Spatial Autoregressive Lag (SAR) model.
    #[tool(
        description = "Run Spatial Autoregressive Lag (SAR) model: y = ρWy + Xβ + ε. Estimates spatial lag parameter ρ via maximum likelihood. Use when there is substantive spatial interaction (e.g., neighbors' outcomes affect own outcome). Optionally estimates Spatial Durbin Model (SDM) with spatially lagged covariates. Equivalent to R's spatialreg::lagsarlm()."
    )]
    pub async fn sar_model(
        &self,
        Parameters(request): Parameters<SarModelRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let mut spatial_weights = self.spatial_weights.write().await;
        let listw = match spatial_weights.get_mut(&request.weights) {
            Some(w) => w,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Spatial weights '{}' not found. Use 'spatial_neighbors' first to create weights.",
                    request.weights
                ))]));
            }
        };

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let config = SarConfig {
            durbin: request.durbin.unwrap_or(false),
            compute_impacts: request.compute_impacts.unwrap_or(true),
            ..Default::default()
        };

        match run_sar(dataset, &request.y, &x_refs, listw, config) {
            Ok(result) => {
                let model_type = if result.is_durbin {
                    "Spatial Durbin Model (SDM)"
                } else {
                    "Spatial Lag Model (SAR)"
                };

                let mut output = format!(
                    "{}\n{}\n\
                     y = ρWy + Xβ + ε\n\n\
                     Spatial autoregressive coefficient (ρ):\n\
                     ---------------------------------------\n\
                     Estimate: {:.6}\n\
                     Std.Error: {:.6}\n\
                     Z-value: {:.4}\n\
                     P-value: {:.6}\n\n\
                     Regression Coefficients:\n\
                     ------------------------\n",
                    model_type,
                    "=".repeat(model_type.len()),
                    result.rho,
                    result.rho_se,
                    result.rho_z,
                    result.rho_p
                );

                for (i, name) in result.coef_names.iter().enumerate() {
                    output.push_str(&format!(
                        "{:15} {:>12.6} {:>10.6} {:>10.4} {:>10.6}\n",
                        name,
                        result.coefficients[i],
                        result.std_errors[i],
                        result.z_values[i],
                        result.p_values[i]
                    ));
                }

                output.push_str(&format!(
                    "\nModel Statistics:\n\
                     -----------------\n\
                     Log-Likelihood: {:.4}\n\
                     AIC: {:.4}\n\
                     BIC: {:.4}\n\
                     Sigma²: {:.6}\n\
                     N: {}\n",
                    result.log_likelihood, result.aic, result.bic, result.sigma2, result.n_obs
                ));

                // Add impacts if computed
                if let Some(impacts) = &result.impacts {
                    output.push_str("\nSpatial Impacts:\n");
                    output.push_str("----------------\n");
                    output.push_str(&format!(
                        "{:15} {:>12} {:>12} {:>12}\n",
                        "Variable", "Direct", "Indirect", "Total"
                    ));
                    for (i, name) in impacts.var_names.iter().enumerate() {
                        output.push_str(&format!(
                            "{:15} {:>12.6} {:>12.6} {:>12.6}\n",
                            name, impacts.direct[i], impacts.indirect[i], impacts.total[i]
                        ));
                    }
                    output.push_str(
                        "\nNote: Direct = own effect, Indirect = spillover effects from neighbors",
                    );
                }

                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "SAR model estimation failed: {}",
                e
            ))])),
        }
    }

    /// Run Spatial Error Model (SEM).
    #[tool(
        description = "Run Spatial Error Model (SEM): y = Xβ + u, where u = λWu + ε. Estimates spatial error parameter λ via maximum likelihood. Use when spatial dependence is in the error term (nuisance dependence, e.g., omitted spatially correlated variables). Equivalent to R's spatialreg::errorsarlm()."
    )]
    pub async fn sem_model(
        &self,
        Parameters(request): Parameters<SemModelRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let mut spatial_weights = self.spatial_weights.write().await;
        let listw = match spatial_weights.get_mut(&request.weights) {
            Some(w) => w,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Spatial weights '{}' not found. Use 'spatial_neighbors' first to create weights.",
                    request.weights
                ))]));
            }
        };

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let config = SemConfig::default();

        match run_sem(dataset, &request.y, &x_refs, listw, config) {
            Ok(result) => {
                let mut output = format!(
                    "Spatial Error Model (SEM)\n\
                     =========================\n\
                     y = Xβ + u, where u = λWu + ε\n\n\
                     Spatial error coefficient (λ):\n\
                     ------------------------------\n\
                     Estimate: {:.6}\n\
                     Std.Error: {:.6}\n\
                     Z-value: {:.4}\n\
                     P-value: {:.6}\n\n\
                     Regression Coefficients:\n\
                     ------------------------\n\
                     {:15} {:>12} {:>10} {:>10} {:>10}\n",
                    result.lambda,
                    result.lambda_se,
                    result.lambda_z,
                    result.lambda_p,
                    "Variable",
                    "Estimate",
                    "Std.Err",
                    "Z-value",
                    "P-value"
                );

                for (i, name) in result.coef_names.iter().enumerate() {
                    output.push_str(&format!(
                        "{:15} {:>12.6} {:>10.6} {:>10.4} {:>10.6}\n",
                        name,
                        result.coefficients[i],
                        result.std_errors[i],
                        result.z_values[i],
                        result.p_values[i]
                    ));
                }

                output.push_str(&format!(
                    "\nModel Statistics:\n\
                     -----------------\n\
                     Log-Likelihood: {:.4}\n\
                     AIC: {:.4}\n\
                     BIC: {:.4}\n\
                     Sigma²: {:.6}\n\
                     N: {}\n\n\
                     Note: SEM coefficients have their usual interpretation\n\
                     (no spatial multiplier effects unlike SAR)",
                    result.log_likelihood, result.aic, result.bic, result.sigma2, result.n_obs
                ));

                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "SEM model estimation failed: {}",
                e
            ))])),
        }
    }

    /// Run Spatial GMM with heteroscedasticity robustness (sphet).
    #[tool(
        description = "Run spatial regression via GMM with heteroscedasticity-robust estimation. Implements the Kelejian-Prucha (1998, 1999, 2010) GMM estimator that is robust to heteroscedasticity of unknown form. Supports SAR (lag), SEM (error), and SARAR (combined) models. Unlike ML estimation, GMM does not require normally distributed errors. Returns heteroscedasticity-robust or HAC standard errors. Equivalent to R's sphet::spreg() and sphet::gstslshet()."
    )]
    pub async fn sphet_model(
        &self,
        Parameters(request): Parameters<SphetRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let spatial_weights = self.spatial_weights.read().await;
        let listw = match spatial_weights.get(&request.weights) {
            Some(w) => w,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Spatial weights '{}' not found. Use 'spatial_neighbors' first to create weights.",
                    request.weights
                ))]));
            }
        };

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        // Parse model type
        let model = match request.model.as_deref() {
            Some("error") | Some("sem") => SphetModel::SpatialError,
            Some("sarar") | Some("both") | Some("combined") => SphetModel::SARAR,
            _ => SphetModel::SpatialLag, // Default: lag/SAR
        };

        // Parse SE type
        let se_type = match request.se_type.as_deref() {
            Some("hac") | Some("HAC") => SphetSE::HAC,
            Some("standard") | Some("homoscedastic") => SphetSE::Standard,
            _ => SphetSE::Robust, // Default: robust
        };

        // Parse HAC kernel
        use p2a_core::regression::HacKernel;
        let kernel = match request.kernel.as_deref() {
            Some("parzen") => HacKernel::Parzen,
            Some("quadratic_spectral") | Some("qs") => HacKernel::QuadraticSpectral,
            Some("tukey_hanning") | Some("tukey") => HacKernel::TukeyHanning,
            Some("truncated") => HacKernel::Truncated,
            _ => HacKernel::Bartlett, // Default
        };

        let config = SphetConfig {
            model,
            het: true,
            se_type,
            kernel,
            bandwidth: request.bandwidth,
            instrument_order: request.instrument_order.unwrap_or(2),
            ..Default::default()
        };

        match run_sphet(dataset, &request.y, &x_refs, listw, config) {
            Ok(result) => {
                let model_name = match result.model_type {
                    SphetModel::SpatialLag => "Spatial Lag (SAR) - GMM",
                    SphetModel::SpatialError => "Spatial Error (SEM) - GMM",
                    SphetModel::SARAR => "SARAR (Lag + Error) - GMM",
                };

                let mut output = format!(
                    "{}\n{}\n\
                     Heteroscedasticity-robust GMM estimation (Kelejian-Prucha)\n\
                     Standard errors: {}\n\n",
                    model_name,
                    "=".repeat(model_name.len()),
                    result.se_type
                );

                // Spatial parameters
                if let (Some(lambda), Some(se), Some(z), Some(p)) = (
                    result.lambda,
                    result.lambda_se,
                    result.lambda_z,
                    result.lambda_p,
                ) {
                    output.push_str(&format!(
                        "Spatial lag coefficient (lambda):\n\
                         ---------------------------------\n\
                         Estimate: {:.6}\n\
                         Std.Error: {:.6}\n\
                         Z-value: {:.4}\n\
                         P-value: {:.6}\n\n",
                        lambda, se, z, p
                    ));
                }

                if let (Some(rho), Some(se), Some(z), Some(p)) =
                    (result.rho, result.rho_se, result.rho_z, result.rho_p)
                {
                    output.push_str(&format!(
                        "Spatial error coefficient (rho):\n\
                         --------------------------------\n\
                         Estimate: {:.6}\n\
                         Std.Error: {:.6}\n\
                         Z-value: {:.4}\n\
                         P-value: {:.6}\n\n",
                        rho, se, z, p
                    ));
                }

                output.push_str("Regression Coefficients:\n");
                output.push_str("------------------------\n");
                output.push_str(&format!(
                    "{:15} {:>12} {:>10} {:>10} {:>10}\n",
                    "Variable", "Estimate", "Std.Err", "Z-value", "P-value"
                ));

                for (i, name) in result.coef_names.iter().enumerate() {
                    output.push_str(&format!(
                        "{:15} {:>12.6} {:>10.6} {:>10.4} {:>10.6}\n",
                        name,
                        result.coefficients[i],
                        result.std_errors[i],
                        result.z_values[i],
                        result.p_values[i]
                    ));
                }

                output.push_str(&format!(
                    "\nModel Statistics:\n\
                     -----------------\n\
                     Sigma²: {:.6}\n\
                     N: {}\n\
                     df: {}\n\
                     Iterations: {}\n\
                     Converged: {}\n\n\
                     Note: GMM estimation is robust to heteroscedasticity of unknown form.\n\
                     Unlike ML, it does not require normally distributed errors.",
                    result.sigma2, result.n_obs, result.df, result.iterations, result.converged
                ));

                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Spatial GMM estimation failed: {}",
                e
            ))])),
        }
    }

    /// Run SAR Probit model (spatial lag probit for binary outcomes).
    #[tool(
        description = "Run Spatial Autoregressive (SAR) Probit model for binary dependent variables: y* = rho*W*y* + X*beta + epsilon, y = 1(y* > 0). Estimates spatial lag parameter rho via Bayesian MCMC with data augmentation. Use when binary outcomes have substantive spatial interaction (e.g., neighbor's choices affect own choice). Returns posterior means, credible intervals, and spatial impacts (direct, indirect, total effects). Equivalent to R's spatialprobit::sarprobit()."
    )]
    pub async fn sar_probit_model(
        &self,
        Parameters(request): Parameters<SarProbitRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let mut spatial_weights = self.spatial_weights.write().await;
        let listw = match spatial_weights.get_mut(&request.weights) {
            Some(w) => w,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Spatial weights '{}' not found. Use 'spatial_neighbors' first to create weights.",
                    request.weights
                ))]));
            }
        };

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let config = SpatialProbitConfig {
            n_draws: request.n_draws.unwrap_or(1000),
            burn_in: request.burn_in.unwrap_or(200),
            compute_impacts: request.compute_impacts.unwrap_or(true),
            seed: request.seed,
            ..Default::default()
        };

        match run_sar_probit(dataset, &request.y, &x_refs, listw, config) {
            Ok(result) => {
                let mut output = format!(
                    "SAR Probit Model (Bayesian MCMC)\n\
                     ================================\n\
                     y* = rho*W*y* + X*beta + epsilon, y = 1(y* > 0)\n\n\
                     Spatial lag coefficient (rho):\n\
                     ------------------------------\n\
                     Posterior Mean: {:.6}\n\
                     Posterior SD: {:.6}\n\
                     Z-value: {:.4}\n\
                     P-value: {:.6}\n\
                     95% Credible Interval: ({:.4}, {:.4})\n\n\
                     Regression Coefficients (Posterior):\n\
                     ------------------------------------\n\
                     {:15} {:>12} {:>10} {:>10} {:>10}\n",
                    result.rho,
                    result.rho_se,
                    result.rho_z,
                    result.rho_p,
                    result.credible_interval_rho(0.95).0,
                    result.credible_interval_rho(0.95).1,
                    "Variable",
                    "Mean",
                    "Std.Dev",
                    "Z-value",
                    "P-value"
                );

                for (i, name) in result.coef_names.iter().enumerate() {
                    let _ci = result.credible_interval_beta(i, 0.95);
                    output.push_str(&format!(
                        "{:15} {:>12.6} {:>10.6} {:>10.4} {:>10.6}\n",
                        name,
                        result.coefficients[i],
                        result.std_errors[i],
                        result.z_values[i],
                        result.p_values[i]
                    ));
                }

                output.push_str(&format!(
                    "\nModel Statistics:\n\
                     -----------------\n\
                     Log-Likelihood: {:.4}\n\
                     Log Marginal Likelihood (approx): {:.4}\n\
                     DIC: {:.4}\n\
                     Pseudo R-squared: {:.4}\n\
                     Percent Correctly Predicted: {:.2}%\n\
                     N: {} (Y=1: {})\n\
                     MCMC Draws: {}, Acceptance Rate: {:.2}%\n",
                    result.log_likelihood,
                    result.log_marginal_likelihood,
                    result.dic,
                    result.pseudo_r_squared,
                    result.pcp,
                    result.n_obs,
                    result.n_positive,
                    result.n_draws,
                    result.acceptance_rate * 100.0
                ));

                if let Some(impacts) = &result.impacts {
                    output.push_str("\nSpatial Marginal Effects (Probability Scale):\n");
                    output.push_str("--------------------------------------------\n");
                    output.push_str(&format!(
                        "{:15} {:>12} {:>12} {:>12}\n",
                        "Variable", "Direct", "Indirect", "Total"
                    ));
                    for (i, name) in impacts.var_names.iter().enumerate() {
                        output.push_str(&format!(
                            "{:15} {:>12.6} {:>12.6} {:>12.6}\n",
                            name, impacts.direct[i], impacts.indirect[i], impacts.total[i]
                        ));
                    }
                    output.push_str(&format!(
                        "{:15} ({:>10.6}) ({:>10.6}) ({:>10.6})\n",
                        "Std.Errors",
                        impacts.direct_se[0],
                        impacts.indirect_se[0],
                        impacts.total_se[0]
                    ));
                    output.push_str("\nNote: Direct = own probability effect, Indirect = spillover from neighbors");
                }

                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "SAR Probit estimation failed: {}",
                e
            ))])),
        }
    }

    /// Run SEM Probit model (spatial error probit for binary outcomes).
    #[tool(
        description = "Run Spatial Error (SEM) Probit model for binary dependent variables: y* = X*beta + u, u = lambda*W*u + epsilon, y = 1(y* > 0). Estimates spatial error parameter lambda via Bayesian MCMC. Use when binary outcomes have spatially correlated unobserved factors (nuisance spatial dependence). Equivalent to R's spatialprobit::semprobit()."
    )]
    pub async fn sem_probit_model(
        &self,
        Parameters(request): Parameters<SemProbitRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let mut spatial_weights = self.spatial_weights.write().await;
        let listw = match spatial_weights.get_mut(&request.weights) {
            Some(w) => w,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Spatial weights '{}' not found. Use 'spatial_neighbors' first to create weights.",
                    request.weights
                ))]));
            }
        };

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let config = SpatialProbitConfig {
            n_draws: request.n_draws.unwrap_or(1000),
            burn_in: request.burn_in.unwrap_or(200),
            compute_impacts: false, // SEM doesn't have spatial multiplier impacts
            seed: request.seed,
            ..Default::default()
        };

        match run_sem_probit(dataset, &request.y, &x_refs, listw, config) {
            Ok(result) => {
                let mut output = format!(
                    "SEM Probit Model (Bayesian MCMC)\n\
                     ================================\n\
                     y* = X*beta + u, u = lambda*W*u + epsilon, y = 1(y* > 0)\n\n\
                     Spatial error coefficient (lambda):\n\
                     -----------------------------------\n\
                     Posterior Mean: {:.6}\n\
                     Posterior SD: {:.6}\n\
                     Z-value: {:.4}\n\
                     P-value: {:.6}\n\
                     95% Credible Interval: ({:.4}, {:.4})\n\n\
                     Regression Coefficients (Posterior):\n\
                     ------------------------------------\n\
                     {:15} {:>12} {:>10} {:>10} {:>10}\n",
                    result.rho, // lambda stored in rho field
                    result.rho_se,
                    result.rho_z,
                    result.rho_p,
                    result.credible_interval_rho(0.95).0,
                    result.credible_interval_rho(0.95).1,
                    "Variable",
                    "Mean",
                    "Std.Dev",
                    "Z-value",
                    "P-value"
                );

                for (i, name) in result.coef_names.iter().enumerate() {
                    output.push_str(&format!(
                        "{:15} {:>12.6} {:>10.6} {:>10.4} {:>10.6}\n",
                        name,
                        result.coefficients[i],
                        result.std_errors[i],
                        result.z_values[i],
                        result.p_values[i]
                    ));
                }

                output.push_str(&format!(
                    "\nModel Statistics:\n\
                     -----------------\n\
                     Log-Likelihood: {:.4}\n\
                     Log Marginal Likelihood (approx): {:.4}\n\
                     DIC: {:.4}\n\
                     Pseudo R-squared: {:.4}\n\
                     Percent Correctly Predicted: {:.2}%\n\
                     N: {} (Y=1: {})\n\
                     MCMC Draws: {}, Acceptance Rate: {:.2}%\n\n\
                     Note: SEM Probit coefficients have their usual interpretation\n\
                     (marginal effect = beta * phi(X'beta) at mean, no spatial multiplier)",
                    result.log_likelihood,
                    result.log_marginal_likelihood,
                    result.dic,
                    result.pseudo_r_squared,
                    result.pcp,
                    result.n_obs,
                    result.n_positive,
                    result.n_draws,
                    result.acceptance_rate * 100.0
                ));

                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "SEM Probit estimation failed: {}",
                e
            ))])),
        }
    }

    /// Run Spatial Panel ML model (spml).
    #[tool(
        description = "Run spatial panel data model via Maximum Likelihood. Combines panel data methods (fixed/random effects) with spatial dependence (lag and/or error). Supports: (1) Spatial lag panel models: y = rho*W*y + X*beta + alpha + e, (2) Spatial error panel models: y = X*beta + alpha + u, u = lambda*W*u + e, (3) Combined spatial lag and error. Equivalent to R's splm::spml(). Reference: Millo & Piras (2012), JSS."
    )]
    pub async fn spatial_panel_ml(
        &self,
        Parameters(request): Parameters<SpmlRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let mut spatial_weights = self.spatial_weights.write().await;
        let listw = match spatial_weights.get_mut(&request.weights) {
            Some(w) => w,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Spatial weights '{}' not found. Use 'spatial_neighbors' first to create weights.",
                    request.weights
                ))]));
            }
        };

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        // Parse model type
        let model = match request.model.as_deref() {
            Some("random") | Some("re") => SpatialPanelModel::Random,
            Some("pooling") | Some("pool") | Some("pooled") => SpatialPanelModel::Pooling,
            _ => SpatialPanelModel::Within, // Default: fixed effects
        };

        // Parse effect type
        let effect = match request.effect.as_deref() {
            Some("time") => SpatialPanelEffect::Time,
            Some("twoways") | Some("twoway") | Some("both") => SpatialPanelEffect::TwoWays,
            _ => SpatialPanelEffect::Individual, // Default
        };

        // Parse spatial error type
        let spatial_error = match request.spatial_error.as_deref() {
            Some("baltagi") | Some("b") | Some("sem") => SpatialErrorType::Baltagi,
            Some("kkp") | Some("sem2") => SpatialErrorType::KKP,
            _ => SpatialErrorType::None,
        };

        let config = SpmlConfig {
            model,
            effect,
            lag: request.lag.unwrap_or(false),
            spatial_error,
            ..Default::default()
        };

        match run_spml(
            dataset,
            &request.y,
            &x_refs,
            &request.entity_col,
            &request.time_col,
            listw,
            config,
        ) {
            Ok(result) => {
                let mut output = format!(
                    "Spatial Panel Model (ML)\n\
                     ========================\n\
                     Model: {}  Effect: {}\n\
                     Spatial Lag: {}  Spatial Error: {}\n\
                     Observations: {}  Entities: {}  Time periods: {}\n\
                     Log-Likelihood: {:.4}  AIC: {:.4}  BIC: {:.4}\n\n",
                    result.model,
                    result.effect,
                    result.has_lag,
                    result.spatial_error,
                    result.n_obs,
                    result.n_entities,
                    result.n_time,
                    result.log_likelihood,
                    result.aic,
                    result.bic
                );

                // Spatial parameters
                if let Some(rho) = result.rho {
                    output.push_str(&format!(
                        "Spatial lag coefficient (rho):\n\
                         ------------------------------\n\
                         Estimate: {:.6}  SE: {:.6}  Z: {:.4}  P: {:.4}\n\n",
                        rho,
                        result.rho_se.unwrap_or(0.0),
                        result.rho_z.unwrap_or(0.0),
                        result.rho_p.unwrap_or(1.0)
                    ));
                }

                if let Some(lambda) = result.lambda {
                    output.push_str(&format!(
                        "Spatial error coefficient (lambda):\n\
                         -----------------------------------\n\
                         Estimate: {:.6}  SE: {:.6}  Z: {:.4}  P: {:.4}\n\n",
                        lambda,
                        result.lambda_se.unwrap_or(0.0),
                        result.lambda_z.unwrap_or(0.0),
                        result.lambda_p.unwrap_or(1.0)
                    ));
                }

                output.push_str(&format!(
                    "Regression Coefficients:\n\
                     ------------------------\n\
                     {:20} {:>12} {:>10} {:>10} {:>10}\n",
                    "Variable", "Estimate", "Std.Err", "Z-value", "P-value"
                ));

                for (i, name) in result.coef_names.iter().enumerate() {
                    output.push_str(&format!(
                        "{:20} {:>12.6} {:>10.6} {:>10.4} {:>10.4}{}\n",
                        name,
                        result.coefficients[i],
                        result.std_errors[i],
                        result.z_values[i],
                        result.p_values[i],
                        result.significance[i].stars()
                    ));
                }

                // Variance components
                output.push_str("\nVariance Components:\n");
                if let Some(sigma_mu) = result.variance.sigma_mu {
                    output.push_str(&format!("  sigma_mu (individual): {:.6}\n", sigma_mu));
                }
                output.push_str(&format!(
                    "  sigma_epsilon (error): {:.6}\n",
                    result.variance.sigma_epsilon
                ));

                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Spatial panel ML estimation failed: {}",
                e
            ))])),
        }
    }

    /// Run Spatial Panel GMM model (spgm).
    #[tool(
        description = "Run spatial panel data model via Generalized Method of Moments. Uses IV/GMM estimation for spatial lag models and moment conditions for spatial error. More robust to non-normality than ML. Methods: w2sls (fixed effects), g2sls (random effects GLS), b2sls (between), ec2sls (Baltagi's EC2SLS). Equivalent to R's splm::spgm(). Reference: Kapoor, Kelejian & Prucha (2007)."
    )]
    pub async fn spatial_panel_gmm(
        &self,
        Parameters(request): Parameters<SpgmRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let mut spatial_weights = self.spatial_weights.write().await;
        let listw = match spatial_weights.get_mut(&request.weights) {
            Some(w) => w,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Spatial weights '{}' not found. Use 'spatial_neighbors' first to create weights.",
                    request.weights
                ))]));
            }
        };

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        // Parse method
        let method = match request.method.as_deref() {
            Some("g2sls") | Some("gls") | Some("random") => SpgmMethod::G2sls,
            Some("b2sls") | Some("between") => SpgmMethod::B2sls,
            Some("ec2sls") | Some("baltagi") => SpgmMethod::Ec2sls,
            _ => SpgmMethod::W2sls, // Default: within/fixed effects
        };

        let config = SpgmConfig {
            method,
            lag: request.lag.unwrap_or(false),
            spatial_error: request.spatial_error.unwrap_or(true),
            ..Default::default()
        };

        match run_spgm(
            dataset,
            &request.y,
            &x_refs,
            &request.entity_col,
            &request.time_col,
            listw,
            config,
        ) {
            Ok(result) => {
                let mut output = format!(
                    "Spatial Panel Model (GMM)\n\
                     =========================\n\
                     Method: {}\n\
                     Spatial Lag: {}  Spatial Error: {}\n\
                     Observations: {}  Entities: {}  Time periods: {}\n\
                     Instruments: {}\n\n",
                    result.method,
                    result.has_lag,
                    result.has_spatial_error,
                    result.n_obs,
                    result.n_entities,
                    result.n_time,
                    result.n_instruments
                );

                // Spatial parameters
                if let Some(rho) = result.rho {
                    output.push_str(&format!("Spatial lag coefficient (rho): {:.6}\n\n", rho));
                }
                if let Some(lambda) = result.lambda {
                    output.push_str(&format!(
                        "Spatial error coefficient (lambda): {:.6}\n\n",
                        lambda
                    ));
                }

                output.push_str(&format!(
                    "Regression Coefficients:\n\
                     ------------------------\n\
                     {:20} {:>12} {:>10} {:>10} {:>10}\n",
                    "Variable", "Estimate", "Std.Err", "Z-value", "P-value"
                ));

                for (i, name) in result.coef_names.iter().enumerate() {
                    output.push_str(&format!(
                        "{:20} {:>12.6} {:>10.6} {:>10.4} {:>10.4}{}\n",
                        name,
                        result.coefficients[i],
                        result.std_errors[i],
                        result.z_values[i],
                        result.p_values[i],
                        result.significance[i].stars()
                    ));
                }

                // Sargan test
                if let (Some(stat), Some(p), Some(df)) =
                    (result.sargan_stat, result.sargan_p, result.sargan_df)
                {
                    output.push_str(&format!(
                        "\nSargan test for overidentifying restrictions:\n\
                         chi2({}) = {:.4}, p-value = {:.4}\n",
                        df, stat, p
                    ));
                }

                // Variance components
                output.push_str(&format!(
                    "\nResidual variance (sigma2): {:.6}\n",
                    result.sigma2
                ));
                if let Some(sigma2_mu) = result.sigma2_mu {
                    output.push_str(&format!(
                        "Individual variance (sigma2_mu): {:.6}\n",
                        sigma2_mu
                    ));
                }

                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Spatial panel GMM estimation failed: {}",
                e
            ))])),
        }
    }
}
