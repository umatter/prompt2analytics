//! Machine learning tool handlers.
//!
//! This module defines ML tool handlers using the `#[tool_router(router = ml_router)]` pattern.

use rmcp::{
    ErrorData as McpError,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content},
    tool, tool_router,
};

use crate::server::AnalyticsServer;
use crate::tools::common::extract_numeric_matrix;
use crate::tools::requests::causal::{BartCausalRequest, CausalForestRequest, HetTxRequest};
use crate::tools::requests::ml::*;

use p2a_core::{
    AdaBoostConfig,
    AdaBoostType,
    CartConfig,
    CartMethod,
    EffectEstimationMethod,
    GbmConfig,
    GbmFamily,
    HetTestStat,
    HetTxConfig,
    KernelSvmConfig,
    Linkage,
    PprConfig,
    SmoothingMethod,
    SvmKernel,
    adaboost,
    cart,
    cmdscale,
    cmdscale_from_data,
    confusion_matrix,
    cutree,
    dbscan,
    // GBM, AdaBoost, CART
    gbm,
    hierarchical,
    // Kernel SVM
    kernel_svm,
    kmeans,
    linear_svm,
    pca,
    ppr,
    random_forest,
    // Model Evaluation
    roc_auc,
    run_bart_causal,
    run_causal_forest,
    run_hettx_dataset,
    tsne,
};

#[tool_router(router = ml_router, vis = "pub")]
impl AnalyticsServer {
    // ========================================================================
    // Clustering Tools
    // ========================================================================

    /// Run K-means clustering.
    #[tool(
        description = "Run K-means clustering to partition data into k clusters. Uses k-means++ initialization for better convergence. Returns cluster assignments, centroids, and inertia (within-cluster sum of squares)."
    )]
    pub async fn ml_kmeans(
        &self,
        Parameters(request): Parameters<KMeansRequest>,
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

        let data = match extract_numeric_matrix(dataset, &request.columns) {
            Ok(d) => d,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        // Use per-tool seed if provided, otherwise fall back to global seed
        let global_seed = self.global_seed.read().await;
        let seed = request.seed.or(*global_seed);

        let result = match kmeans(
            data.view(),
            request.k,
            request.max_iterations,
            None, // tolerance
            request.n_init,
            seed,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "K-means clustering failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run DBSCAN clustering.
    #[tool(
        description = "Run DBSCAN (Density-Based Spatial Clustering of Applications with Noise) clustering. Finds clusters of arbitrary shape and identifies outliers as noise points. Does not require specifying number of clusters."
    )]
    pub async fn ml_dbscan(
        &self,
        Parameters(request): Parameters<DBSCANRequest>,
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

        let data = match extract_numeric_matrix(dataset, &request.columns) {
            Ok(d) => d,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        let result = match dbscan(data.view(), request.eps, request.min_samples) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "DBSCAN clustering failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run Hierarchical (Agglomerative) clustering.
    #[tool(
        description = "Run Hierarchical clustering using agglomerative approach. Supports Ward, single, complete, and average linkage methods. Returns cluster assignments and dendrogram information."
    )]
    pub async fn ml_hierarchical(
        &self,
        Parameters(request): Parameters<HierarchicalRequest>,
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

        let data = match extract_numeric_matrix(dataset, &request.columns) {
            Ok(d) => d,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        let linkage_method = match request.linkage.as_deref() {
            Some("single") => Linkage::Single,
            Some("complete") => Linkage::Complete,
            Some("average") => Linkage::Average,
            Some("ward") | None => Linkage::Ward,
            Some(other) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown linkage method '{}'. Use 'single', 'complete', 'average', or 'ward'.",
                    other
                ))]));
            }
        };

        let result = match hierarchical(
            data.view(),
            request.n_clusters,
            linkage_method,
            request.distance_threshold,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Hierarchical clustering failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Cut a hierarchical clustering tree into groups.
    #[tool(
        description = "Cut a hierarchical clustering dendrogram into groups (cutree). First performs hierarchical clustering, then cuts the resulting tree at a specified number of clusters (k) or height. Returns cluster assignments for each observation. Useful for extracting cluster memberships from a dendrogram."
    )]
    pub async fn ml_cutree(
        &self,
        Parameters(request): Parameters<CutreeRequest>,
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

        let data = match extract_numeric_matrix(dataset, &request.columns) {
            Ok(d) => d,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        let linkage_method = match request.linkage.as_deref() {
            Some("single") => Linkage::Single,
            Some("complete") => Linkage::Complete,
            Some("average") => Linkage::Average,
            Some("ward") | None => Linkage::Ward,
            Some(other) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown linkage method '{}'. Use 'single', 'complete', 'average', or 'ward'.",
                    other
                ))]));
            }
        };

        // First, perform hierarchical clustering to get full dendrogram
        let hclust = match hierarchical(data.view(), Some(1), linkage_method, None) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Hierarchical clustering failed: {}",
                    e
                ))]));
            }
        };

        // Then cut the tree
        let result = match cutree(&hclust, request.k, request.cut_height) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "cutree failed: {}",
                    e
                ))]));
            }
        };

        // Format output as JSON
        let mut cluster_counts: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        for &label in &result.labels {
            *cluster_counts.entry(label).or_insert(0) += 1;
        }

        let output = serde_json::json!({
            "n_observations": result.n,
            "n_clusters": result.k,
            "cut_height": result.cut_height,
            "cluster_assignments": &result.labels[..result.n.min(100)],
            "cluster_sizes": cluster_counts,
            "note": if result.n > 100 { Some(format!("Showing first 100 of {} assignments", result.n)) } else { None }
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| result.to_string()),
        )]))
    }

    // ========================================================================
    // Dimensionality Reduction Tools
    // ========================================================================

    /// Run PCA (Principal Component Analysis).
    #[tool(
        description = "Run Principal Component Analysis (PCA) for dimensionality reduction. Returns principal components, explained variance ratios, and loadings. Useful for understanding data structure and reducing feature dimensionality."
    )]
    pub async fn ml_pca(
        &self,
        Parameters(request): Parameters<PCARequest>,
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

        let data = match extract_numeric_matrix(dataset, &request.columns) {
            Ok(d) => d,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        let result = match pca(data.view(), request.n_components, false) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "PCA failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run t-SNE dimensionality reduction.
    #[tool(
        description = "Run t-SNE (t-distributed Stochastic Neighbor Embedding) for visualizing high-dimensional data in 2D or 3D. Preserves local structure while revealing clusters. Good for exploratory visualization."
    )]
    pub async fn ml_tsne(
        &self,
        Parameters(request): Parameters<TsneRequest>,
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

        let data = match extract_numeric_matrix(dataset, &request.columns) {
            Ok(d) => d,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        // Use per-tool seed if provided, otherwise fall back to global seed
        let global_seed = self.global_seed.read().await;
        let seed = request.seed.or(*global_seed);

        let result = match tsne(
            data.view(),
            request.n_components,
            request.perplexity,
            request.max_iterations,
            request.learning_rate,
            seed,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "t-SNE failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run Classical Multidimensional Scaling (cmdscale).
    #[tool(
        description = "Classical Multidimensional Scaling (cmdscale) for embedding distances into Euclidean space. Takes a data matrix and computes a low-dimensional configuration that preserves pairwise Euclidean distances. Returns point coordinates and goodness-of-fit measures. Useful for visualizing similarity/dissimilarity data."
    )]
    pub async fn ml_cmdscale(
        &self,
        Parameters(request): Parameters<CmdscaleRequest>,
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

        let data = match extract_numeric_matrix(dataset, &request.columns) {
            Ok(d) => d,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        let k = request.k.unwrap_or(2);
        let is_dist = request.is_distance_matrix.unwrap_or(false);

        let result = if is_dist {
            // Input is already a distance matrix
            let n = data.nrows();
            if data.ncols() != n {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Distance matrix must be square. Got {} rows x {} cols.",
                    n,
                    data.ncols()
                ))]));
            }
            // Convert to nested Vec
            let dist: Vec<Vec<f64>> = (0..n).map(|i| data.row(i).to_vec()).collect();
            cmdscale(&dist, Some(k), Some(true), None)
        } else {
            // Compute Euclidean distances from data
            cmdscale_from_data(data.view(), Some(k))
        };

        match result {
            Ok(r) => {
                let output = serde_json::json!({
                    "n_points": r.n,
                    "k_dimensions": r.k,
                    "gof": {
                        "gof1_positive_eigenvalues": r.gof[0],
                        "gof2_k_eigenvalues": r.gof[1]
                    },
                    "eigenvalues": &r.eig[..r.k.min(10)],
                    "configuration": &r.points[..r.n.min(20)],
                    "note": if r.n > 20 { Some(format!("Showing first 20 of {} points", r.n)) } else { None }
                });
                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&output).unwrap_or_else(|_| r.to_string()),
                )]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Classical MDS (cmdscale) failed: {}",
                e
            ))])),
        }
    }

    // ========================================================================
    // Supervised Learning Tools
    // ========================================================================

    /// Run Random Forest regression.
    #[tool(
        description = "Run Random Forest regression. Ensemble of decision trees for robust predictions. Returns feature importances, out-of-bag score, and predictions."
    )]
    pub async fn ml_random_forest(
        &self,
        Parameters(request): Parameters<RandomForestRequest>,
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

        let data = match extract_numeric_matrix(dataset, &request.features) {
            Ok(d) => d,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        // Extract target column
        let df = dataset.df();
        let target_col = match df.column(&request.target) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Target column '{}' not found: {}",
                    request.target, e
                ))]));
            }
        };

        let target_values: Vec<f64> = match target_col.cast(&DataType::Float64) {
            Ok(c) => match c.f64() {
                Ok(f) => f.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Target column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cannot convert target to numeric: {}",
                    e
                ))]));
            }
        };

        let target = ndarray::Array1::from_vec(target_values);

        // Use per-tool seed if provided, otherwise fall back to global seed
        let global_seed = self.global_seed.read().await;
        let seed = request.seed.or(*global_seed);

        let result = match random_forest(
            data.view(),
            target.view(),
            request.n_trees,
            request.max_depth,
            request.min_samples_split,
            request.max_features.as_deref(),
            seed,
            Some(request.features.clone()),
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Random Forest failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run Linear SVM classification.
    #[tool(
        description = "Run Linear Support Vector Machine (SVM) for binary classification. Uses SMO algorithm. Returns weights, bias, support vector count, and predictions."
    )]
    pub async fn ml_svm(
        &self,
        Parameters(request): Parameters<SvmRequest>,
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

        let data = match extract_numeric_matrix(dataset, &request.features) {
            Ok(d) => d,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        // Extract target column
        let df = dataset.df();
        let target_col = match df.column(&request.target) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Target column '{}' not found: {}",
                    request.target, e
                ))]));
            }
        };

        let target_values: Vec<f64> = match target_col.cast(&DataType::Float64) {
            Ok(c) => match c.f64() {
                Ok(f) => f.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Target column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cannot convert target to numeric: {}",
                    e
                ))]));
            }
        };

        let target = ndarray::Array1::from_vec(target_values);

        let result = match linear_svm(
            data.view(),
            target.view(),
            request.c,
            request.max_iterations,
            request.tolerance,
            Some(request.features.clone()),
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "SVM failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run Projection Pursuit Regression (PPR).
    #[tool(
        description = "Projection Pursuit Regression (PPR) - a dimension reduction regression that fits models of the form y = sum(f_k(alpha_k' * x)). Finds optimal projection directions and ridge functions. Returns projection directions, coefficients, fitted values, and goodness-of-fit metrics. Useful for non-linear regression when relationships are complex."
    )]
    pub async fn ml_ppr(
        &self,
        Parameters(request): Parameters<PprRequest>,
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

        let data = match extract_numeric_matrix(dataset, &request.features) {
            Ok(d) => d,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        // Extract target column
        let df = dataset.df();
        let target_col = match df.column(&request.target) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Target column '{}' not found: {}",
                    request.target, e
                ))]));
            }
        };

        let target_values: Vec<f64> = match target_col.cast(&DataType::Float64) {
            Ok(c) => match c.f64() {
                Ok(f) => f.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Target column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cannot convert target to numeric: {}",
                    e
                ))]));
            }
        };

        // Parse smoothing method
        let sm_method = match request.sm_method.as_deref() {
            Some("spline") => SmoothingMethod::Spline,
            Some("gcvspline") => SmoothingMethod::GcvSpline,
            Some("supsmu") | None => SmoothingMethod::Supsmu,
            Some(other) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown smoothing method '{}'. Use 'supsmu', 'spline', or 'gcvspline'.",
                    other
                ))]));
            }
        };

        let config = PprConfig {
            nterms: request.nterms.unwrap_or(1),
            max_terms: request.max_terms.unwrap_or(request.nterms.unwrap_or(1)),
            sm_method,
            bass: request.bass.unwrap_or(0.0),
            ..Default::default()
        };

        let result = match ppr(data.view(), &target_values, None, config) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "PPR failed: {}",
                    e
                ))]));
            }
        };

        // Format output
        let output = serde_json::json!({
            "nterms": result.nterms,
            "n_observations": result.n,
            "n_predictors": result.p,
            "projection_directions": result.alpha,
            "ridge_coefficients": result.beta,
            "gof_rss": result.gofn,
            "fitted_values_sample": &result.fitted[..result.n.min(20)],
            "residuals_sample": &result.residuals[..result.n.min(20)],
            "note": if result.n > 20 { Some(format!("Showing first 20 of {} fitted values", result.n)) } else { None }
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| format!("{:?}", result)),
        )]))
    }

    // ========================================================================
    // Causal ML Tools
    // ========================================================================

    /// Run Causal Forest for heterogeneous treatment effects (Wager & Athey 2018).
    #[tool(
        description = "Causal Forest estimates heterogeneous treatment effects (CATE) using random forests adapted for causal inference. Key features: honest splitting (separate data for tree structure vs estimation), local centering, bootstrap variance estimation. Returns: CATE estimates for each unit, ATE with confidence interval, variable importance showing which covariates drive treatment effect heterogeneity. Based on R package 'grf'."
    )]
    pub async fn ml_causal_forest(
        &self,
        Parameters(request): Parameters<CausalForestRequest>,
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

        // Use per-tool seed if provided, otherwise fall back to global seed
        let global_seed = self.global_seed.read().await;
        let seed = request.seed.or(*global_seed);

        let result = match run_causal_forest(
            dataset,
            &request.outcome,
            &request.treatment,
            request.covariates.clone(),
            request.n_trees,
            request.min_node_size,
            request.honesty,
            request.max_depth,
            seed,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Causal Forest failed: {}",
                    e
                ))]));
            }
        };

        // Create a summary JSON output
        let output = serde_json::json!({
            "ate": result.ate,
            "ate_se": result.ate_se,
            "ate_t_stat": result.ate_t_stat,
            "ate_p_value": result.ate_p_value,
            "ate_ci_lower": result.ate_ci_lower,
            "ate_ci_upper": result.ate_ci_upper,
            "ate_significance": result.ate_significance.stars(),
            "n_obs": result.n_obs,
            "n_trees": result.n_trees,
            "oob_error": result.oob_error,
            "variable_importance": result.variable_importance,
            "cate_summary": {
                "min": result.predictions.iter().cloned().fold(f64::INFINITY, f64::min),
                "max": result.predictions.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                "mean": result.predictions.iter().sum::<f64>() / result.predictions.len() as f64,
            },
            "config": {
                "honesty": result.config.honesty,
                "min_node_size": result.config.min_node_size,
                "max_depth": result.config.max_depth,
            }
        });

        Ok(CallToolResult::success(vec![Content::text(format!(
            "{}\n\nJSON Summary:\n{}",
            result,
            serde_json::to_string_pretty(&output).unwrap_or_default()
        ))]))
    }

    /// Run BART-based causal inference for heterogeneous treatment effects (bartCause style).
    #[tool(
        description = "BART Causal estimates heterogeneous treatment effects using Bayesian Additive Regression Trees methodology. Fits separate response surfaces for treated and control groups, then computes CATE = E[Y|T=1,X] - E[Y|T=0,X]. Uses bootstrap for uncertainty quantification. Returns: ATE with confidence interval, CATE estimates for each unit, variable importance for treatment effect heterogeneity. Simplified frequentist approximation to R's bartCause package."
    )]
    pub async fn ml_bart_causal(
        &self,
        Parameters(request): Parameters<BartCausalRequest>,
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

        // Use per-tool seed if provided, otherwise fall back to global seed
        let global_seed = self.global_seed.read().await;
        let seed = request.seed.or(*global_seed);

        let result = match run_bart_causal(
            dataset,
            &request.outcome,
            &request.treatment,
            request.covariates.clone(),
            request.n_trees,
            request.max_depth,
            request.n_bootstrap,
            request.include_propensity,
            seed,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "BART Causal failed: {}",
                    e
                ))]));
            }
        };

        // Create a summary JSON output
        let output = serde_json::json!({
            "ate": result.ate,
            "ate_se": result.ate_se,
            "ate_t_stat": result.ate_t_stat,
            "ate_p_value": result.ate_p_value,
            "ate_ci_lower": result.ate_ci_lower,
            "ate_ci_upper": result.ate_ci_upper,
            "ate_significance": result.ate_significance.stars(),
            "n_obs": result.n_obs,
            "n_treated": result.n_treated,
            "n_control": result.n_control,
            "n_trees": result.n_trees,
            "n_bootstrap": result.n_bootstrap,
            "variable_importance": result.variable_importance,
            "cate_summary": {
                "min": result.cate.iter().cloned().fold(f64::INFINITY, f64::min),
                "max": result.cate.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                "mean": result.cate.iter().sum::<f64>() / result.cate.len() as f64,
            },
            "config": {
                "include_propensity": result.config.include_propensity,
                "max_depth": result.config.max_depth,
                "min_node_size": result.config.min_node_size,
            }
        });

        Ok(CallToolResult::success(vec![Content::text(format!(
            "{}\n\nJSON Summary:\n{}",
            result,
            serde_json::to_string_pretty(&output).unwrap_or_default()
        ))]))
    }

    /// Run Treatment Effect Heterogeneity Test (hettx).
    #[tool(
        description = "Test for treatment effect heterogeneity using Fisherian randomization inference. Tests H0: all individual treatment effects are equal (tau_i = tau). Returns permutation p-value, estimated individual effects, ATE, and optionally decomposes heterogeneity into systematic (explained by covariates) and idiosyncratic components. Based on R package 'hettx' by Ding, Feller & Miratrix."
    )]
    pub async fn heterogeneity_test(
        &self,
        Parameters(request): Parameters<HetTxRequest>,
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

        // Use per-tool seed if provided, otherwise fall back to global seed
        let global_seed = self.global_seed.read().await;
        let seed = request.seed.or(*global_seed);

        // Parse test statistic
        let test_statistic = match request.test_statistic.as_deref() {
            Some("range") => HetTestStat::Range,
            Some("iqr") => HetTestStat::IQR,
            Some("mad") => HetTestStat::MeanAbsDeviation,
            _ => HetTestStat::Variance,
        };

        // Parse effect estimation method
        let effect_method = match request.effect_method.as_deref() {
            Some("regression") | Some("reg") => EffectEstimationMethod::Regression,
            Some("stratified") | Some("strat") => EffectEstimationMethod::Stratified,
            _ => EffectEstimationMethod::Matching,
        };

        let config = HetTxConfig {
            n_permutations: request.n_permutations.unwrap_or(1000),
            test_statistic,
            decompose: request.decompose.unwrap_or(true),
            effect_method,
            n_neighbors: request.n_neighbors.unwrap_or(3),
            seed,
            compute_importance: true,
        };

        let cov_refs: Vec<&str> = request
            .covariates
            .iter()
            .map(|s: &String| s.as_str())
            .collect();

        let result = match run_hettx_dataset(
            dataset,
            &request.outcome,
            &request.treatment,
            &cov_refs,
            config,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Heterogeneity test failed: {}",
                    e
                ))]));
            }
        };

        // Create JSON summary
        let mut output = serde_json::json!({
            "test_statistic": result.test_statistic,
            "test_statistic_type": format!("{}", result.test_statistic_type),
            "p_value": result.p_value,
            "significance": result.significance.stars(),
            "ate": result.ate,
            "ate_se": result.ate_se,
            "n_obs": result.n_obs,
            "n_treated": result.n_treated,
            "n_control": result.n_control,
            "n_permutations": result.n_permutations,
            "effect_summary": {
                "min": result.effect_summary.min,
                "p10": result.effect_summary.p10,
                "p25": result.effect_summary.p25,
                "median": result.effect_summary.median,
                "p75": result.effect_summary.p75,
                "p90": result.effect_summary.p90,
                "max": result.effect_summary.max,
                "std_dev": result.effect_summary.std_dev,
            }
        });

        // Add decomposition if available
        if let Some(ref decomp) = result.decomposition {
            output["decomposition"] = serde_json::json!({
                "total_variance": decomp.total_variance,
                "systematic_variance": decomp.systematic_variance,
                "idiosyncratic_variance": decomp.idiosyncratic_variance,
                "r_squared": decomp.r_squared,
                "systematic_p_value": decomp.systematic_p_value,
                "idiosyncratic_p_value": decomp.idiosyncratic_p_value,
                "covariate_importance": decomp.covariate_importance,
            });
        }

        Ok(CallToolResult::success(vec![Content::text(format!(
            "{}\n\nJSON Summary:\n{}",
            result,
            serde_json::to_string_pretty(&output).unwrap_or_default()
        ))]))
    }

    // ========================================================================
    // Gradient Boosting, AdaBoost, CART Tools
    // ========================================================================

    /// Run Gradient Boosting Machine (GBM).
    #[tool(
        description = "Gradient Boosting Machine (GBM) for regression and classification. Builds an ensemble of decision trees sequentially, each correcting the errors of the previous. Supports Gaussian (MSE), Huber (robust), and Binomial (classification) loss functions. Returns fitted model with variable importance, training loss, and predictions."
    )]
    pub async fn ml_gbm(
        &self,
        Parameters(request): Parameters<GbmRequest>,
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

        let data = match extract_numeric_matrix(dataset, &request.features) {
            Ok(d) => d,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        // Extract target column
        let df = dataset.df();
        let target_col = match df.column(&request.target) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Target column '{}' not found: {}",
                    request.target, e
                ))]));
            }
        };

        let target_values: Vec<f64> = match target_col.cast(&DataType::Float64) {
            Ok(c) => match c.f64() {
                Ok(f) => f.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Target column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cannot convert target to numeric: {}",
                    e
                ))]));
            }
        };

        let target = ndarray::Array1::from_vec(target_values);

        // Parse family
        let family = match request.family.as_deref() {
            Some("huber") => GbmFamily::Huber,
            Some("binomial") => GbmFamily::Binomial,
            Some("gaussian") | None => GbmFamily::Gaussian,
            Some(other) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown family '{}'. Use 'gaussian', 'huber', or 'binomial'.",
                    other
                ))]));
            }
        };

        // Use per-tool seed if provided, otherwise fall back to global seed
        let global_seed = self.global_seed.read().await;
        let seed = request.seed.or(*global_seed);

        let config = GbmConfig {
            n_trees: request.n_trees.unwrap_or(100),
            learning_rate: request.learning_rate.unwrap_or(0.1),
            max_depth: request.max_depth.unwrap_or(3),
            min_samples_split: request.min_samples_split.unwrap_or(10),
            subsample: request.subsample.unwrap_or(1.0),
            family,
            seed,
            ..Default::default()
        };

        let result = match gbm(data.view(), target.view(), &config) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "GBM failed: {}",
                    e
                ))]));
            }
        };

        // Format output
        let output = serde_json::json!({
            "n_trees": result.n_trees,
            "final_train_loss": result.final_train_loss,
            "family": format!("{:?}", config.family),
            "learning_rate": config.learning_rate,
            "max_depth": config.max_depth,
            "feature_importances": result.feature_importances,
            "feature_names": request.features,
            "n_observations": data.nrows(),
        });

        Ok(CallToolResult::success(vec![Content::text(format!(
            "{}\n\nJSON Summary:\n{}",
            result,
            serde_json::to_string_pretty(&output).unwrap_or_default()
        ))]))
    }

    /// Run AdaBoost.
    #[tool(
        description = "AdaBoost (Adaptive Boosting) for classification and regression. AdaBoost.M1 for binary classification, AdaBoost.R2 for regression, SAMME for multi-class. Uses decision stumps (depth-1 trees) as weak learners by default. Returns model with estimator weights, training accuracy/loss, and predictions."
    )]
    pub async fn ml_adaboost(
        &self,
        Parameters(request): Parameters<AdaBoostRequest>,
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

        let data = match extract_numeric_matrix(dataset, &request.features) {
            Ok(d) => d,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        // Extract target column
        let df = dataset.df();
        let target_col = match df.column(&request.target) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Target column '{}' not found: {}",
                    request.target, e
                ))]));
            }
        };

        let target_values: Vec<f64> = match target_col.cast(&DataType::Float64) {
            Ok(c) => match c.f64() {
                Ok(f) => f.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Target column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cannot convert target to numeric: {}",
                    e
                ))]));
            }
        };

        let target = ndarray::Array1::from_vec(target_values);

        // Parse boost type
        let boost_type = match request.boost_type.as_deref() {
            Some("r2") | Some("R2") => AdaBoostType::R2,
            Some("samme") | Some("SAMME") => AdaBoostType::Samme,
            Some("m1") | Some("M1") | None => AdaBoostType::M1,
            Some(other) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown boost_type '{}'. Use 'm1', 'r2', or 'samme'.",
                    other
                ))]));
            }
        };

        // Use per-tool seed if provided, otherwise fall back to global seed
        let global_seed = self.global_seed.read().await;
        let seed = request.seed.or(*global_seed);

        let config = AdaBoostConfig {
            n_estimators: request.n_estimators.unwrap_or(50),
            boost_type,
            max_depth: request.max_depth.unwrap_or(1),
            learning_rate: request.learning_rate.unwrap_or(1.0),
            seed,
            ..Default::default()
        };

        let result = match adaboost(data.view(), target.view(), &config) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "AdaBoost failed: {}",
                    e
                ))]));
            }
        };

        // Format output
        let output = serde_json::json!({
            "n_estimators": result.n_estimators,
            "boost_type": format!("{:?}", config.boost_type),
            "train_accuracy": result.train_accuracy,
            "train_error": result.train_error,
            "estimator_weights": result.estimator_weights,
            "feature_importances": result.feature_importances,
            "feature_names": request.features,
            "n_observations": data.nrows(),
        });

        Ok(CallToolResult::success(vec![Content::text(format!(
            "{}\n\nJSON Summary:\n{}",
            result,
            serde_json::to_string_pretty(&output).unwrap_or_default()
        ))]))
    }

    /// Run CART decision tree.
    #[tool(
        description = "CART (Classification and Regression Trees) decision tree. Supports regression (ANOVA/MSE), classification (Gini impurity or entropy). Includes cost-complexity pruning with cross-validation. Returns tree structure, variable importance, cp table for pruning, and predictions. Similar to R's rpart package."
    )]
    pub async fn ml_cart(
        &self,
        Parameters(request): Parameters<CartRequest>,
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

        let data = match extract_numeric_matrix(dataset, &request.features) {
            Ok(d) => d,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        // Extract target column
        let df = dataset.df();
        let target_col = match df.column(&request.target) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Target column '{}' not found: {}",
                    request.target, e
                ))]));
            }
        };

        let target_values: Vec<f64> = match target_col.cast(&DataType::Float64) {
            Ok(c) => match c.f64() {
                Ok(f) => f.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Target column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cannot convert target to numeric: {}",
                    e
                ))]));
            }
        };

        let target = ndarray::Array1::from_vec(target_values);

        // Parse method
        let method = match request.method.as_deref() {
            Some("gini") => CartMethod::Gini,
            Some("entropy") | Some("information") => CartMethod::Entropy,
            Some("anova") | Some("mse") | None => CartMethod::Anova,
            Some(other) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown method '{}'. Use 'anova', 'gini', or 'entropy'.",
                    other
                ))]));
            }
        };

        let config = CartConfig {
            method,
            max_depth: request.max_depth.unwrap_or(30),
            min_split: request.min_split.unwrap_or(20),
            min_bucket: request.min_bucket.unwrap_or(7),
            cp: request.cp.unwrap_or(0.01),
            xval: request.xval.unwrap_or(10),
            ..Default::default()
        };

        let result = match cart(data.view(), target.view(), &config) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "CART failed: {}",
                    e
                ))]));
            }
        };

        // Format output
        let output = serde_json::json!({
            "n_nodes": result.n_nodes,
            "n_terminal": result.n_terminal,
            "depth": result.depth,
            "method": format!("{:?}", config.method),
            "variable_importance": result.variable_importance,
            "feature_names": request.features,
            "cp_table": result.cp_table.iter().map(|row| {
                serde_json::json!({
                    "cp": row.cp,
                    "nsplit": row.nsplit,
                    "rel_error": row.rel_error,
                    "xerror": row.xerror,
                    "xstd": row.xstd,
                })
            }).collect::<Vec<_>>(),
            "n_observations": data.nrows(),
        });

        Ok(CallToolResult::success(vec![Content::text(format!(
            "{}\n\nJSON Summary:\n{}",
            result,
            serde_json::to_string_pretty(&output).unwrap_or_default()
        ))]))
    }

    /// Run Kernel SVM classification.
    #[tool(
        description = "Kernel Support Vector Machine for binary classification. Supports RBF (Gaussian), polynomial, sigmoid, and linear kernels. RBF kernel can handle non-linearly separable data. Returns support vectors, predictions, and training accuracy. Based on SMO algorithm."
    )]
    pub async fn ml_kernel_svm(
        &self,
        Parameters(request): Parameters<KernelSvmRequest>,
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

        let data = match extract_numeric_matrix(dataset, &request.features) {
            Ok(d) => d,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        // Extract target column
        let df = dataset.df();
        let target_col = match df.column(&request.target) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Target column '{}' not found: {}",
                    request.target, e
                ))]));
            }
        };

        let target_values: Vec<f64> = match target_col.cast(&DataType::Float64) {
            Ok(c) => match c.f64() {
                Ok(f) => f.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Target column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cannot convert target to numeric: {}",
                    e
                ))]));
            }
        };

        let target = ndarray::Array1::from_vec(target_values);

        // Parse kernel type
        let kernel = match request.kernel.as_deref() {
            Some("linear") => SvmKernel::Linear,
            Some("polynomial") | Some("poly") => SvmKernel::Polynomial,
            Some("sigmoid") | Some("tanh") => SvmKernel::Sigmoid,
            Some("rbf") | Some("gaussian") | None => SvmKernel::Rbf,
            Some(other) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown kernel '{}'. Use 'linear', 'rbf', 'polynomial', or 'sigmoid'.",
                    other
                ))]));
            }
        };

        let config = KernelSvmConfig {
            kernel,
            c: request.c.unwrap_or(1.0),
            gamma: request.gamma,
            degree: request.degree.unwrap_or(3),
            coef0: request.coef0.unwrap_or(0.0),
            max_iter: request.max_iterations.unwrap_or(1000),
            tolerance: request.tolerance.unwrap_or(1e-3),
        };

        let result = match kernel_svm(
            data.view(),
            target.view(),
            &config,
            Some(request.features.clone()),
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Kernel SVM failed: {}",
                    e
                ))]));
            }
        };

        // Format output
        let output = serde_json::json!({
            "kernel": format!("{:?}", config.kernel),
            "c": config.c,
            "gamma": config.gamma,
            "degree": config.degree,
            "n_support_vectors": result.n_support_vectors,
            "support_vector_indices": result.support_vector_indices,
            "train_accuracy": result.train_accuracy,
            "converged": result.converged,
            "n_iterations": result.n_iterations,
            "bias": result.bias,
            "class_labels": result.class_labels,
            "feature_names": request.features,
            "n_observations": data.nrows(),
        });

        Ok(CallToolResult::success(vec![Content::text(format!(
            "{}\n\nJSON Summary:\n{}",
            result,
            serde_json::to_string_pretty(&output).unwrap_or_default()
        ))]))
    }

    /// Calculate ROC curve and AUC.
    #[tool(
        description = "Calculate ROC (Receiver Operating Characteristic) curve and AUC (Area Under Curve) for binary classification evaluation. Requires predicted probabilities and actual binary labels. Returns TPR/FPR pairs for plotting, AUC value, and optimal threshold using Youden's J statistic."
    )]
    pub async fn ml_roc_auc(
        &self,
        Parameters(request): Parameters<RocAucRequest>,
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

        // Extract predictions column
        let pred_col = match df.column(&request.predictions) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Predictions column '{}' not found: {}",
                    request.predictions, e
                ))]));
            }
        };

        let predictions: Vec<f64> = match pred_col.cast(&DataType::Float64) {
            Ok(c) => match c.f64() {
                Ok(f) => f.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Predictions column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cannot convert predictions to numeric: {}",
                    e
                ))]));
            }
        };

        // Extract actual column
        let actual_col = match df.column(&request.actual) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Actual labels column '{}' not found: {}",
                    request.actual, e
                ))]));
            }
        };

        let actual: Vec<f64> = match actual_col.cast(&DataType::Float64) {
            Ok(c) => match c.f64() {
                Ok(f) => f.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Actual labels column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cannot convert actual labels to numeric: {}",
                    e
                ))]));
            }
        };

        let result = match roc_auc(&predictions, &actual, request.n_thresholds) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "ROC/AUC calculation failed: {}",
                    e
                ))]));
            }
        };

        // Format output
        let output = serde_json::json!({
            "auc": result.auc,
            "optimal_threshold": result.optimal_threshold,
            "n_positive": result.n_positive,
            "n_negative": result.n_negative,
            "optimal_metrics": {
                "sensitivity": result.optimal_metrics.sensitivity,
                "specificity": result.optimal_metrics.specificity,
                "precision": result.optimal_metrics.precision,
                "f1_score": result.optimal_metrics.f1_score,
                "accuracy": result.optimal_metrics.accuracy,
                "youden_j": result.optimal_metrics.youden_j,
            },
            "roc_curve_points": result.thresholds.len(),
        });

        Ok(CallToolResult::success(vec![Content::text(format!(
            "{}\n\nJSON Summary:\n{}",
            result,
            serde_json::to_string_pretty(&output).unwrap_or_default()
        ))]))
    }

    /// Calculate variable importance using tree-based models.
    #[tool(
        description = "Calculate variable importance for machine learning models. Uses Mean Decrease in Impurity (MDI) from tree-based models (Random Forest, GBM, CART). Returns importance scores, ranks, and feature names. Higher scores indicate more important features."
    )]
    pub async fn ml_variable_importance(
        &self,
        Parameters(request): Parameters<VariableImportanceRequest>,
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

        let data = match extract_numeric_matrix(dataset, &request.features) {
            Ok(d) => d,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        let df = dataset.df();

        // Extract target column
        use p2a_core::polars::prelude::*;
        let target_col = match df.column(&request.target) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Target column '{}' not found: {}",
                    request.target, e
                ))]));
            }
        };

        let target: Vec<f64> = match target_col.cast(&DataType::Float64) {
            Ok(c) => match c.f64() {
                Ok(f) => f.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Target column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cannot convert target to numeric: {}",
                    e
                ))]));
            }
        };

        let target_arr = ndarray::Array1::from_vec(target);
        let feature_names: Vec<String> = request.features.clone();

        // Use per-tool seed if provided, otherwise fall back to global seed
        let global_seed = self.global_seed.read().await;
        let seed = request.seed.or(*global_seed);

        let model_type = request.model.as_deref().unwrap_or("rf");

        let result = match model_type {
            "rf" | "random_forest" => {
                // Train Random Forest and extract importance
                match random_forest(
                    data.view(),
                    target_arr.view(),
                    Some(100), // n_trees
                    Some(10),  // max_depth
                    Some(2),   // min_samples_split
                    Some("sqrt"),
                    seed,
                    Some(feature_names.clone()),
                ) {
                    Ok(rf_result) => {
                        p2a_core::rf_variable_importance(&rf_result, Some(&feature_names))
                    }
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Random Forest training failed: {}",
                            e
                        ))]));
                    }
                }
            }
            "gbm" => {
                // Train GBM and extract importance
                let config = GbmConfig {
                    n_trees: 100,
                    learning_rate: 0.1,
                    max_depth: 3,
                    min_samples_split: 10,
                    subsample: 1.0,
                    colsample_bytree: 1.0,
                    family: GbmFamily::Gaussian,
                    huber_delta: 1.35,
                    seed,
                };
                match gbm(data.view(), target_arr.view(), &config) {
                    Ok(gbm_result) => {
                        p2a_core::gbm_variable_importance(&gbm_result, Some(&feature_names))
                    }
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "GBM training failed: {}",
                            e
                        ))]));
                    }
                }
            }
            "cart" => {
                // Train CART and extract importance
                let config = CartConfig {
                    method: CartMethod::Anova,
                    max_depth: 30,
                    min_split: 20,
                    min_bucket: 7,
                    cp: 0.01,
                    xval: 10,
                    max_surrogate: 5,
                    use_surrogate: true,
                    seed,
                };
                match cart(data.view(), target_arr.view(), &config) {
                    Ok(cart_result) => {
                        p2a_core::cart_variable_importance(&cart_result, Some(&feature_names))
                    }
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "CART training failed: {}",
                            e
                        ))]));
                    }
                }
            }
            _ => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown model type '{}'. Supported: 'rf', 'gbm', 'cart'",
                    model_type
                ))]));
            }
        };

        // Format output
        let output = serde_json::json!({
            "model_type": result.model_type,
            "baseline_score": result.baseline_score,
            "features": result.feature_names.iter()
                .zip(result.importance.iter())
                .zip(result.ranks.iter())
                .map(|((name, &imp), &rank)| {
                    serde_json::json!({
                        "name": name,
                        "importance": imp,
                        "rank": rank
                    })
                })
                .collect::<Vec<_>>()
        });

        Ok(CallToolResult::success(vec![Content::text(format!(
            "{}\n\nJSON Summary:\n{}",
            result,
            serde_json::to_string_pretty(&output).unwrap_or_default()
        ))]))
    }

    /// Calculate partial dependence plot.
    #[tool(
        description = "Calculate partial dependence plot for machine learning models. Shows the marginal effect of one feature on the predicted outcome, averaging over the values of all other features. Supported models: 'gbm', 'cart'. Returns grid values and corresponding partial dependence values for visualization."
    )]
    pub async fn ml_partial_dependence(
        &self,
        Parameters(request): Parameters<PartialDependenceRequest>,
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

        let data = match extract_numeric_matrix(dataset, &request.features) {
            Ok(d) => d,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        let df = dataset.df();

        // Extract target column
        use p2a_core::polars::prelude::*;
        let target_col = match df.column(&request.target) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Target column '{}' not found: {}",
                    request.target, e
                ))]));
            }
        };

        let target: Vec<f64> = match target_col.cast(&DataType::Float64) {
            Ok(c) => match c.f64() {
                Ok(f) => f.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Target column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cannot convert target to numeric: {}",
                    e
                ))]));
            }
        };

        let target_arr = ndarray::Array1::from_vec(target);
        let feature_names: Vec<String> = request.features.clone();

        // Get the PD feature - only support 1D for now
        if request.pd_features.is_empty() {
            return Ok(CallToolResult::error(vec![Content::text(
                "Must specify at least one pd_feature".to_string(),
            )]));
        }

        let pd_feature = &request.pd_features[0];
        let pd_feature_idx = match feature_names.iter().position(|f| f == pd_feature) {
            Some(idx) => idx,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "PD feature '{}' not found in features list",
                    pd_feature
                ))]));
            }
        };

        let grid_resolution = request.grid_resolution.unwrap_or(20);

        // Use per-tool seed if provided, otherwise fall back to global seed
        let global_seed = self.global_seed.read().await;
        let seed = request.seed.or(*global_seed);

        let model_type = request.model.as_deref().unwrap_or("gbm");

        let result = match model_type {
            "gbm" => {
                // Train GBM and compute PDP
                let config = GbmConfig {
                    n_trees: 100,
                    learning_rate: 0.1,
                    max_depth: 3,
                    min_samples_split: 10,
                    subsample: 1.0,
                    colsample_bytree: 1.0,
                    family: GbmFamily::Gaussian,
                    huber_delta: 1.35,
                    seed,
                };
                match gbm(data.view(), target_arr.view(), &config) {
                    Ok(gbm_result) => p2a_core::gbm_partial_dependence(
                        data.view(),
                        &gbm_result,
                        pd_feature_idx,
                        pd_feature,
                        grid_resolution,
                    ),
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "GBM training failed: {}",
                            e
                        ))]));
                    }
                }
            }
            "cart" => {
                // Train CART and compute PDP
                let config = CartConfig {
                    method: CartMethod::Anova,
                    max_depth: 30,
                    min_split: 20,
                    min_bucket: 7,
                    cp: 0.01,
                    xval: 10,
                    max_surrogate: 5,
                    use_surrogate: true,
                    seed,
                };
                match cart(data.view(), target_arr.view(), &config) {
                    Ok(cart_result) => p2a_core::cart_partial_dependence(
                        data.view(),
                        &cart_result,
                        pd_feature_idx,
                        pd_feature,
                        grid_resolution,
                    ),
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "CART training failed: {}",
                            e
                        ))]));
                    }
                }
            }
            _ => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown model type '{}'. Supported: 'gbm', 'cart'",
                    model_type
                ))]));
            }
        };

        let result = match result {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Partial dependence calculation failed: {}",
                    e
                ))]));
            }
        };

        // Format output
        let output = serde_json::json!({
            "model_type": result.model_type,
            "feature": pd_feature,
            "grid_resolution": grid_resolution,
            "data": result.grid_values[0].iter()
                .zip(result.pd_values.iter())
                .map(|(&x, &y)| {
                    serde_json::json!({
                        "x": x,
                        "y": y
                    })
                })
                .collect::<Vec<_>>()
        });

        Ok(CallToolResult::success(vec![Content::text(format!(
            "{}\n\nJSON Summary:\n{}",
            result,
            serde_json::to_string_pretty(&output).unwrap_or_default()
        ))]))
    }
}
