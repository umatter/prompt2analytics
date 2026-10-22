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
    EconError,
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
        // Snapshot the dataset (a cheap clone) so no lock is held while computing.
        let dataset = {
            let datasets = self.datasets.read().await;
            crate::get_dataset!(datasets, &request.dataset).clone()
        };

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

        // The link budget (none without a resource profile).
        let limits = crate::tools::limits::current();
        let cap = limits.spatial_links;
        let cap_links = cap.map(|c| usize::try_from(c).unwrap_or(usize::MAX));

        // Create neighbors based on method
        let method = request.method.as_deref().unwrap_or("knn");
        let (built, band) = match method {
            "knn" => {
                let k = request.k.unwrap_or(5);
                // kNN stores exactly n * min(k, n - 1) links: refuse before building anything.
                let links = (n as u64).saturating_mul(k.min(n.saturating_sub(1)) as u64);
                if let Err(v) =
                    limits.check_budget("spatial_neighbors", "spatial_links", links, cap)
                {
                    tracing::info!(
                        tool = "spatial_neighbors",
                        override_key = v.override_key,
                        requested = links,
                        limit = cap,
                        "request refused by the spatial_links budget"
                    );
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "{} Use a smaller k.",
                        v.message
                    ))]));
                }
                (Neighbors::try_from_knn(&coords, k), None)
            }
            "distance" => {
                let d_max = request.d_max.unwrap_or(1.0);
                let d_min = request.d_min.unwrap_or(0.0);
                (
                    Neighbors::try_from_distance(&coords, d_min, d_max, cap_links),
                    Some((d_min, d_max)),
                )
            }
            "distance_longlat" | "longlat" => {
                let d_max = request.d_max.unwrap_or(100.0); // km
                let d_min = request.d_min.unwrap_or(0.0);
                (
                    Neighbors::try_from_distance_longlat(&coords, d_min, d_max, cap_links),
                    Some((d_min, d_max)),
                )
            }
            other => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown neighbor method '{}'. Use 'knn', 'distance', or 'distance_longlat'.",
                    other
                ))]));
            }
        };
        let neighbors = match (built, cap, band) {
            (Ok(nb), _, _) => nb,
            // The core stopped at the cap without counting further.
            (Err(EconError::ResourceLimit(_)), Some(c), Some((d_min, d_max))) => {
                tracing::info!(
                    tool = "spatial_neighbors",
                    override_key = "spatial_links",
                    limit = c,
                    "request refused by the spatial_links budget"
                );
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "This request to `spatial_neighbors` needs more than {c} neighbour links \
                     (spatial_links), which exceeds this server's limit of {c}. The distance \
                     band from d_min = {d_min} to d_max = {d_max} is too wide for these \
                     coordinates; use a smaller d_max, or method 'knn' with a small k."
                ))]));
            }
            (Err(e), _, _) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Could not create spatial neighbours: {e}"
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

        // Get stats from neighbors before we lose access to it
        let avg_neighbors = neighbors.avg_neighbors();
        let has_isolates = neighbors.has_isolates();
        let n_isolates = neighbors.isolates().len();
        let is_symmetric = neighbors.is_symmetric();

        // Create spatial weights (they share the neighbour lists)
        let weights = SpatialWeights::from_neighbors(&neighbors, style);
        drop(neighbors);
        drop(dataset);

        // Store the weights; the lock is held only for the insert
        let weights_name = request
            .weights_name
            .unwrap_or_else(|| format!("{}_weights", request.dataset));

        self.spatial_weights
            .write()
            .await
            .insert(weights_name.clone(), weights);

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
        let dataset = {
            let datasets = self.datasets.read().await;
            crate::get_dataset!(datasets, &request.dataset).clone()
        };

        let listw = match self.weights_snapshot(&request.weights).await {
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

        match moran_test(&x, &listw, alternative) {
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
        let dataset = {
            let datasets = self.datasets.read().await;
            crate::get_dataset!(datasets, &request.dataset).clone()
        };

        let mut listw = match self.weights_snapshot(&request.weights).await {
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
        let ols_result = match run_ols(
            &dataset,
            &request.y,
            &x_refs,
            true,
            CovarianceType::Standard,
        ) {
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

        let tests = spatial_lm_tests(ols_result.residuals(), &x_mat, &mut listw);
        self.keep_weights_cache(&request.weights, listw).await;
        match tests {
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
        let dataset = {
            let datasets = self.datasets.read().await;
            crate::get_dataset!(datasets, &request.dataset).clone()
        };

        let mut listw = match self.weights_snapshot(&request.weights).await {
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

        let fit = run_sar(&dataset, &request.y, &x_refs, &mut listw, config);
        self.keep_weights_cache(&request.weights, listw).await;
        match fit {
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
                    for warning in &impacts.warnings {
                        output.push_str(&format!("\n⚠ WARNING: {}", warning));
                    }
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
        let dataset = {
            let datasets = self.datasets.read().await;
            crate::get_dataset!(datasets, &request.dataset).clone()
        };

        let mut listw = match self.weights_snapshot(&request.weights).await {
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

        let fit = run_sem(&dataset, &request.y, &x_refs, &mut listw, config);
        self.keep_weights_cache(&request.weights, listw).await;
        match fit {
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
        let dataset = {
            let datasets = self.datasets.read().await;
            crate::get_dataset!(datasets, &request.dataset).clone()
        };

        let listw = match self.weights_snapshot(&request.weights).await {
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

        match run_sphet(&dataset, &request.y, &x_refs, &listw, config) {
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
        let dataset = {
            let datasets = self.datasets.read().await;
            crate::get_dataset!(datasets, &request.dataset).clone()
        };

        let mut listw = match self.weights_snapshot(&request.weights).await {
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

        let fit = run_sar_probit(&dataset, &request.y, &x_refs, &mut listw, config);
        self.keep_weights_cache(&request.weights, listw).await;
        match fit {
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
        let dataset = {
            let datasets = self.datasets.read().await;
            crate::get_dataset!(datasets, &request.dataset).clone()
        };

        let mut listw = match self.weights_snapshot(&request.weights).await {
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

        let fit = run_sem_probit(&dataset, &request.y, &x_refs, &mut listw, config);
        self.keep_weights_cache(&request.weights, listw).await;
        match fit {
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
        let dataset = {
            let datasets = self.datasets.read().await;
            crate::get_dataset!(datasets, &request.dataset).clone()
        };

        let mut listw = match self.weights_snapshot(&request.weights).await {
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

        let fit = run_spml(
            &dataset,
            &request.y,
            &x_refs,
            &request.entity_col,
            &request.time_col,
            &mut listw,
            config,
        );
        self.keep_weights_cache(&request.weights, listw).await;
        match fit {
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
        let dataset = {
            let datasets = self.datasets.read().await;
            crate::get_dataset!(datasets, &request.dataset).clone()
        };

        let mut listw = match self.weights_snapshot(&request.weights).await {
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

        let fit = run_spgm(
            &dataset,
            &request.y,
            &x_refs,
            &request.entity_col,
            &request.time_col,
            &mut listw,
            config,
        );
        self.keep_weights_cache(&request.weights, listw).await;
        match fit {
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

/// Lock discipline for the weights map: handlers compute on O(1) snapshots and hold the
/// map's lock only to take a snapshot or to store a cache computed on one.
impl AnalyticsServer {
    /// O(1) clone of a stored weights object; the read lock is released before returning.
    async fn weights_snapshot(&self, name: &str) -> Option<SpatialWeights> {
        self.spatial_weights.read().await.get(name).cloned()
    }

    /// Store eigenvalues computed on a snapshot, if the entry is still the weights the
    /// snapshot was taken from and has none.
    async fn keep_weights_cache(&self, name: &str, used: SpatialWeights) {
        if used.get_eigenvalues().is_none() {
            return;
        }
        let mut map = self.spatial_weights.write().await;
        if let Some(entry) = map.get_mut(name)
            && entry.same_weights(&used)
            && entry.get_eigenvalues().is_none()
        {
            *entry = used;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::limits::{self, Limits};
    use polars::prelude::*;

    /// A current-thread runtime: `limits::with_limits` is thread-local.
    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    fn server_with_points(n: usize) -> AnalyticsServer {
        // jittered 10-column grid: no distance ties needed here
        let x: Vec<f64> = (0..n)
            .map(|i| (i % 10) as f64 + 0.01 * ((i * 7) % 13) as f64)
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| (i / 10) as f64 + 0.01 * ((i * 5) % 11) as f64)
            .collect();
        let server = AnalyticsServer::new();
        let ds = p2a_core::Dataset::new(df! { "x" => x, "y" => y }.unwrap());
        runtime()
            .block_on(server.datasets.write())
            .insert("pts".into(), ds);
        server
    }

    fn neighbors_req(
        method: &str,
        k: Option<usize>,
        d_min: Option<f64>,
        d_max: Option<f64>,
    ) -> SpatialNeighborsRequest {
        SpatialNeighborsRequest {
            dataset: "pts".into(),
            x_coord: "x".into(),
            y_coord: "y".into(),
            method: Some(method.into()),
            k,
            d_max,
            d_min,
            weights_name: Some("w".into()),
            style: None,
        }
    }

    fn call(
        server: &AnalyticsServer,
        limits: Limits,
        req: SpatialNeighborsRequest,
    ) -> CallToolResult {
        let rt = runtime();
        limits::with_limits(limits, || {
            rt.block_on(server.spatial_neighbors(Parameters(req)))
        })
        .expect("refusals are in-band tool errors")
    }

    /// The first content block's text.
    fn text(r: &CallToolResult) -> String {
        r.content
            .first()
            .and_then(|c| c.raw.as_text())
            .map(|t| t.text.clone())
            .unwrap_or_default()
    }

    fn tight() -> Limits {
        Limits::hosted()
            .with_overrides(&["spatial_links=100".to_string()])
            .unwrap()
    }

    #[test]
    fn knn_over_the_link_budget_is_refused() {
        let s = server_with_points(50);
        // 150 links
        let r = call(&s, tight(), neighbors_req("knn", Some(3), None, None));
        assert_eq!(r.is_error, Some(true));
        let t = text(&r);
        assert!(
            t.contains("spatial_links") && t.contains("150") && t.contains("100"),
            "{t}"
        );
        assert!(runtime().block_on(s.spatial_weights.read()).is_empty());
    }

    #[test]
    fn distance_band_over_the_link_budget_is_refused() {
        let s = server_with_points(50);
        // 2,450 links
        let r = call(
            &s,
            tight(),
            neighbors_req("distance", None, None, Some(1e9)),
        );
        assert_eq!(r.is_error, Some(true));
        let t = text(&r);
        assert!(
            t.contains("spatial_links") && t.contains("more than 100") && t.contains("d_max"),
            "{t}"
        );
        let r = call(
            &s,
            tight(),
            neighbors_req("distance_longlat", None, None, Some(30_000.0)),
        );
        assert_eq!(r.is_error, Some(true));
        assert!(runtime().block_on(s.spatial_weights.read()).is_empty());
    }

    #[test]
    fn requests_within_budget_and_unprofiled_requests_succeed() {
        let s = server_with_points(50);
        let r = call(
            &s,
            Limits::hosted(),
            neighbors_req("knn", Some(5), None, None),
        );
        assert_ne!(r.is_error, Some(true), "{}", text(&r));
        assert!(text(&r).contains("Average neighbors: 5.00"));
        let r = call(
            &s,
            Limits::off(),
            neighbors_req("distance", None, None, Some(1e9)),
        );
        assert_ne!(r.is_error, Some(true), "{}", text(&r));
        assert!(text(&r).contains("Average neighbors: 49.00"));
    }

    #[test]
    fn invalid_neighbour_parameters_are_refused() {
        let s = server_with_points(50);
        for req in [
            neighbors_req("knn", Some(0), None, None),
            neighbors_req("knn", Some(50), None, None),
            neighbors_req("distance", None, Some(-1.0), Some(1.0)),
            neighbors_req("distance", None, Some(2.0), Some(1.0)),
        ] {
            let r = call(&s, Limits::off(), req);
            assert_eq!(r.is_error, Some(true), "{}", text(&r));
        }
    }

    // The 30-point example of p2a-core's spatial test fixture (cfg(test) there, so copied).
    const CX: [f64; 30] = [
        1.551, 9.684, 4.683, 7.768, 4.079, 5.388, 2.069, 1.871, 7.800, 1.939, 4.342, 0.023, 8.347,
        8.301, 9.570, 9.485, 6.007, 2.618, 6.430, 5.262, 8.812, 6.075, 7.400, 8.031, 9.024, 0.927,
        5.745, 5.755, 9.442, 2.190,
    ];
    const CY: [f64; 30] = [
        4.780, 0.447, 7.991, 3.683, 5.633, 2.470, 7.343, 7.037, 7.368, 8.019, 7.521, 6.338, 2.476,
        5.513, 2.348, 2.586, 9.529, 8.569, 2.757, 5.545, 8.769, 1.188, 5.467, 2.328, 6.023, 7.125,
        6.489, 1.876, 2.384, 0.704,
    ];
    const X1: [f64; 30] = [
        -0.376, 4.290, 0.149, 2.401, 2.935, 1.013, 0.149, -0.074, 2.054, 0.720, 2.530, -0.795,
        1.424, 2.333, 1.799, 2.707, 1.205, -1.399, 2.170, 1.319, 3.544, 2.764, 3.688, 3.116, 3.526,
        -0.015, 3.142, 3.225, 2.176, -0.196,
    ];
    const Y: [f64; 30] = [
        2.954, 10.913, 7.509, 8.861, 11.166, 5.213, 3.960, 2.370, 7.030, 6.307, 11.371, 1.777,
        5.011, 10.318, 5.315, 8.269, 7.287, 2.026, 5.994, 6.341, 13.936, 7.310, 12.132, 7.804,
        10.951, 3.608, 11.282, 8.275, 6.480, 1.193,
    ];

    fn fx_neighbors(k: usize) -> SpatialNeighborsRequest {
        SpatialNeighborsRequest {
            dataset: "fx".into(),
            x_coord: "cx".into(),
            y_coord: "cy".into(),
            method: Some("knn".into()),
            k: Some(k),
            d_max: None,
            d_min: None,
            weights_name: Some("w".into()),
            style: None,
        }
    }

    /// Dataset "fx" (the fixture, plus `y_bin`), panel "fxp" (the same 30 units over 3
    /// periods) and kNN-3 weights "w" built through `spatial_neighbors`.
    fn fixture_server() -> AnalyticsServer {
        let server = AnalyticsServer::new();
        let y_bin: Vec<f64> = Y.iter().map(|&v| if v > 7.0 { 1.0 } else { 0.0 }).collect();
        let fx = p2a_core::Dataset::new(
            df! {
                "y" => Y.to_vec(), "x1" => X1.to_vec(), "y_bin" => y_bin,
                "cx" => CX.to_vec(), "cy" => CY.to_vec(),
            }
            .unwrap(),
        );
        let (mut entity, mut time, mut py, mut px1, mut px2) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for t in 0..3_i64 {
            for i in 0..30 {
                let (fi, ft) = (i as f64, t as f64);
                let v1 = X1[i] + 0.1 * ft + 0.05 * (fi * ft).sin();
                let v2 = CY[i] / 10.0 + 0.2 * (1.3 * fi + ft).cos();
                entity.push(i as i64);
                time.push(t);
                px1.push(v1);
                px2.push(v2);
                py.push(Y[i] / 4.0 + 0.6 * v1 - 0.4 * v2 + 0.15 * (2.1 * fi + 0.7 * ft).sin());
            }
        }
        let fxp = p2a_core::Dataset::new(
            df! { "entity" => entity, "time" => time, "y" => py, "x1" => px1, "x2" => px2 }
                .unwrap(),
        );
        {
            let rt = runtime();
            let mut map = rt.block_on(server.datasets.write());
            map.insert("fx".into(), fx);
            map.insert("fxp".into(), fxp);
        }
        let r = call(&server, Limits::off(), fx_neighbors(3));
        assert_ne!(r.is_error, Some(true), "{}", text(&r));
        server
    }

    fn sar_req() -> SarModelRequest {
        SarModelRequest {
            dataset: "fx".into(),
            y: "y".into(),
            x: vec!["x1".into()],
            weights: "w".into(),
            durbin: None,
            compute_impacts: None,
        }
    }

    #[test]
    fn weights_snapshot_releases_the_lock() {
        let s = fixture_server();
        let rt = runtime();
        let snap = rt.block_on(s.weights_snapshot("w")).unwrap();
        assert!(s.spatial_weights.try_write().is_ok());
        assert!(snap.same_weights(&rt.block_on(s.spatial_weights.read())["w"]));
        assert!(rt.block_on(s.weights_snapshot("missing")).is_none());
    }

    #[test]
    fn model_handlers_keep_the_eigenvalue_cache() {
        let s = fixture_server();
        let rt = runtime();
        let r = rt.block_on(s.sar_model(Parameters(sar_req()))).unwrap();
        assert_ne!(r.is_error, Some(true), "{}", text(&r));
        let map = s
            .spatial_weights
            .try_read()
            .expect("no lock held after the call");
        assert!(map["w"].get_eigenvalues().is_some());
    }

    #[test]
    fn cache_is_not_written_into_replaced_weights() {
        let s = fixture_server();
        let rt = runtime();
        let mut snap = rt.block_on(s.weights_snapshot("w")).unwrap();
        let _ = snap.eigenvalues();
        // the user rebuilds "w" while a model ran on the snapshot
        let r = call(&s, Limits::off(), fx_neighbors(4));
        assert_ne!(r.is_error, Some(true), "{}", text(&r));
        rt.block_on(s.keep_weights_cache("w", snap));
        let map = rt.block_on(s.spatial_weights.read());
        assert!(map["w"].get_eigenvalues().is_none());
        assert!((map["w"].neighbors().avg_neighbors() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn every_weights_handler_runs_on_a_snapshot() {
        let s = fixture_server();
        let rt = runtime();
        let x = || vec!["x1".to_string()];
        let px = || vec!["x1".to_string(), "x2".to_string()];
        let results = [
            rt.block_on(s.moran_test(Parameters(MoranTestRequest {
                dataset: "fx".into(),
                variable: "y".into(),
                weights: "w".into(),
                alternative: None,
            }))),
            rt.block_on(s.spatial_lm_tests_tool(Parameters(SpatialLmTestRequest {
                dataset: "fx".into(),
                y: "y".into(),
                x: x(),
                weights: "w".into(),
            }))),
            rt.block_on(s.sar_model(Parameters(sar_req()))),
            rt.block_on(s.sem_model(Parameters(SemModelRequest {
                dataset: "fx".into(),
                y: "y".into(),
                x: x(),
                weights: "w".into(),
            }))),
            rt.block_on(s.sphet_model(Parameters(SphetRequest {
                dataset: "fx".into(),
                y: "y".into(),
                x: x(),
                weights: "w".into(),
                model: Some("sarar".into()),
                se_type: None,
                kernel: None,
                bandwidth: None,
                instrument_order: None,
            }))),
            rt.block_on(s.sar_probit_model(Parameters(SarProbitRequest {
                dataset: "fx".into(),
                y: "y_bin".into(),
                x: x(),
                weights: "w".into(),
                n_draws: Some(40),
                burn_in: Some(10),
                compute_impacts: Some(true),
                seed: Some(1),
            }))),
            rt.block_on(s.sem_probit_model(Parameters(SemProbitRequest {
                dataset: "fx".into(),
                y: "y_bin".into(),
                x: x(),
                weights: "w".into(),
                n_draws: Some(40),
                burn_in: Some(10),
                seed: Some(1),
            }))),
            rt.block_on(s.spatial_panel_ml(Parameters(SpmlRequest {
                dataset: "fxp".into(),
                y: "y".into(),
                x: px(),
                entity_col: "entity".into(),
                time_col: "time".into(),
                weights: "w".into(),
                model: None,
                lag: Some(true),
                spatial_error: None,
                effect: None,
            }))),
            rt.block_on(s.spatial_panel_gmm(Parameters(SpgmRequest {
                dataset: "fxp".into(),
                y: "y".into(),
                x: px(),
                entity_col: "entity".into(),
                time_col: "time".into(),
                weights: "w".into(),
                method: None,
                lag: Some(true),
                spatial_error: None,
            }))),
        ];
        for (i, r) in results.into_iter().enumerate() {
            let r = r.unwrap();
            assert_ne!(r.is_error, Some(true), "handler {i}: {}", text(&r));
            assert!(s.spatial_weights.try_write().is_ok(), "handler {i}");
            assert!(s.datasets.try_write().is_ok(), "handler {i}");
        }
        let missing = rt
            .block_on(s.sem_model(Parameters(SemModelRequest {
                dataset: "fx".into(),
                y: "y".into(),
                x: x(),
                weights: "nope".into(),
            })))
            .unwrap();
        assert!(text(&missing).contains("Spatial weights 'nope' not found"));
    }
}
