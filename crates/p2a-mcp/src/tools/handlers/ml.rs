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
    C50Config,
    CartConfig,
    CartMethod,
    CtreeConfig,
    CubistConfig,
    EffectEstimationMethod,
    GbmConfig,
    GbmFamily,
    HetTestStat,
    HetTxConfig,
    Linkage,
    MboostBaseLearner,
    MboostConfig,
    MboostFamily,
    PprConfig,
    ShapConfig,
    SmoothingMethod,
    cmdscale,
    cmdscale_from_data,
    cutree,
    dbscan,
    hierarchical,
    kmeans,
    linear_svm,
    pca,
    ppr,
    random_forest,
    // SHAP values
    random_forest_with_trees,
    run_bart_causal,
    // C5.0 Decision Trees
    run_c50,
    run_cart,
    run_causal_forest,
    // Conditional Inference Trees
    run_ctree,
    // Cubist rule-based regression
    run_cubist,
    // GBM and CART
    run_gbm,
    run_hettx_dataset,
    // Model-based Boosting (mboost)
    run_mboost,
    shap_summary,
    shap_values_model,
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

    /// Run C5.0 Decision Tree classification.
    #[tool(
        description = "C5.0 Decision Tree for classification (Quinlan's successor to C4.5). Uses information gain ratio for splitting, pessimistic error pruning, and optionally boosting via AdaBoost.M1. Features: automatic feature selection (winnowing), rule extraction, multiclass support. Returns tree structure, rules (if requested), variable importance, and class predictions. Based on R's C50 package."
    )]
    pub async fn ml_c50(
        &self,
        Parameters(request): Parameters<C50Request>,
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

        // Build C5.0 config
        let config = C50Config {
            trials: request.trials.unwrap_or(1),
            rules: request.rules.unwrap_or(false),
            winnow: request.winnow.unwrap_or(false),
            min_cases: request.min_cases.unwrap_or(2),
            cf: request.cf.unwrap_or(0.25),
            seed,
            ..Default::default()
        };

        // Convert feature names to references
        let feature_refs: Vec<&str> = request.features.iter().map(|s| s.as_str()).collect();

        let result = match run_c50(dataset, &request.target, &feature_refs, &config) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "C5.0 failed: {}",
                    e
                ))]));
            }
        };

        // Build variable importance list
        let var_importance: Vec<serde_json::Value> = result
            .variable_importance
            .iter()
            .enumerate()
            .map(|(i, &imp)| {
                let name = result
                    .feature_names
                    .as_ref()
                    .and_then(|names| names.get(i))
                    .map(|s| s.as_str())
                    .unwrap_or("?");
                serde_json::json!({
                    "feature": name,
                    "importance": imp
                })
            })
            .collect();

        // Build rules list if available
        let rules_json: Option<Vec<serde_json::Value>> = result.rules.as_ref().map(|rules| {
            rules
                .iter()
                .take(20) // Limit to first 20 rules
                .map(|rule| {
                    serde_json::json!({
                        "rule_id": rule.id,
                        "conditions": rule.conditions.iter().map(|c| {
                            format!("{} {} {:.4}", c.feature, c.operator, c.threshold)
                        }).collect::<Vec<_>>(),
                        "predicted_class": rule.predicted_class,
                        "confidence": rule.confidence,
                        "support": rule.support,
                        "lift": rule.lift
                    })
                })
                .collect()
        });

        let n_predictions = result.predictions.len();

        // Create output JSON
        let output = serde_json::json!({
            "n_observations": n_predictions,
            "n_classes": result.n_classes,
            "class_labels": result.class_labels,
            "actual_trials": result.actual_trials,
            "accuracy": result.accuracy,
            "error_rate": 1.0 - result.accuracy,
            "variable_importance": var_importance,
            "rules": rules_json,
            "n_rules": result.rules.as_ref().map(|r| r.len()).unwrap_or(0),
            "config": {
                "winnow": result.selected_features.is_some(),
                "boosting": result.actual_trials > 1,
                "min_cases": request.min_cases.unwrap_or(2),
                "cf": request.cf.unwrap_or(0.25)
            },
            "predictions_sample": &result.predictions[..n_predictions.min(20)],
            "note": if n_predictions > 20 {
                Some(format!("Showing first 20 of {} predictions", n_predictions))
            } else {
                None
            }
        });

        Ok(CallToolResult::success(vec![Content::text(format!(
            "{}\n\nJSON Summary:\n{}",
            result,
            serde_json::to_string_pretty(&output).unwrap_or_default()
        ))]))
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
    // Cubist Rule-Based Regression
    // ========================================================================

    /// Run Cubist rule-based regression with linear models in terminal nodes.
    #[tool(
        description = "Cubist is a rule-based regression model with linear models in terminal nodes. Based on Quinlan's M5 model tree algorithm. Key features: (1) Rule extraction with linear regression in leaves, (2) Committee models (boosted ensembles) for improved accuracy, (3) Instance-based (k-NN) prediction adjustment. Returns rules, variable importance, and predictions. Based on Quinlan (1992) 'Learning with Continuous Classes' and R's Cubist package."
    )]
    pub async fn ml_cubist(
        &self,
        Parameters(request): Parameters<CubistRequest>,
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

        // Build configuration
        let config = CubistConfig {
            committees: request.committees.unwrap_or(1),
            neighbors: request.neighbors.unwrap_or(0),
            max_depth: request.max_depth.unwrap_or(10),
            min_split: request.min_split.unwrap_or(10),
            min_bucket: request.min_bucket.unwrap_or(5),
            seed: request.seed,
            ..Default::default()
        };

        let feature_refs: Vec<&str> = request.features.iter().map(|s| s.as_str()).collect();

        let result = match run_cubist(dataset, &request.target, &feature_refs, &config) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cubist regression failed: {}",
                    e
                ))]));
            }
        };

        // Build rules summary for JSON output
        let rules_summary: Vec<serde_json::Value> = result
            .rules
            .iter()
            .take(10)
            .map(|rule| {
                let conditions: Vec<serde_json::Value> = rule
                    .conditions
                    .iter()
                    .map(|c| {
                        serde_json::json!({
                            "feature": c.feature,
                            "feature_name": c.feature_name,
                            "operator": c.operator,
                            "threshold": c.threshold
                        })
                    })
                    .collect();

                // coefficients is Vec<(usize, String, f64)> - (feature_idx, feature_name, coefficient)
                let coefficients: Vec<serde_json::Value> = rule
                    .coefficients
                    .iter()
                    .filter(|(_, _, coef)| coef.abs() > 1e-10)
                    .map(|(idx, name, coef)| {
                        serde_json::json!({
                            "feature_index": idx,
                            "feature": name,
                            "coefficient": coef
                        })
                    })
                    .collect();

                serde_json::json!({
                    "id": rule.id,
                    "coverage": rule.coverage,
                    "conditions": conditions,
                    "intercept": rule.intercept,
                    "coefficients": coefficients,
                    "mean_response": rule.mean_response
                })
            })
            .collect();

        // Variable importance (top 10)
        let var_importance: Vec<(String, f64)> = if let Some(ref names) = result.feature_names {
            names
                .iter()
                .zip(result.variable_importance.iter())
                .map(|(name, &imp)| (name.clone(), imp))
                .collect()
        } else {
            result
                .variable_importance
                .iter()
                .enumerate()
                .map(|(i, &imp)| (format!("X{}", i), imp))
                .collect()
        };
        let mut var_importance_sorted = var_importance;
        var_importance_sorted
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_importance: Vec<serde_json::Value> = var_importance_sorted
            .iter()
            .take(10)
            .map(|(name, imp)| serde_json::json!({"feature": name, "importance": imp}))
            .collect();

        let output = serde_json::json!({
            "n_observations": result.n_obs,
            "n_features": result.n_features,
            "committees": result.committees,
            "neighbors": result.neighbors,
            "n_rules": result.rules.len(),
            "variable_importance": top_importance,
            "rules_sample": rules_summary,
            "r_squared": result.train_r_squared,
            "rmse": result.train_rmse,
            "config": {
                "committees": config.committees,
                "neighbors": config.neighbors,
                "max_depth": config.max_depth,
                "min_split": config.min_split,
                "min_bucket": config.min_bucket
            }
        });

        Ok(CallToolResult::success(vec![Content::text(format!(
            "{}\n\nJSON Summary:\n{}",
            result,
            serde_json::to_string_pretty(&output).unwrap_or_default()
        ))]))
    }

    // ========================================================================
    // Model Interpretation Tools (XAI)
    // ========================================================================

    /// Compute SHAP values for model interpretation.
    #[tool(
        description = "SHAP (SHapley Additive exPlanations) computes feature importance for individual predictions using game-theoretic Shapley values. For tree-based models, uses TreeSHAP (Lundberg et al. 2018) for exact O(TLD^2) computation. Returns: SHAP values matrix (n_obs x n_features), global feature importance (mean |SHAP|), and summary statistics. Each SHAP value represents the contribution of a feature to the difference between the actual prediction and the average prediction. Implements R's 'fastshap' and Python's 'shap' packages."
    )]
    pub async fn ml_shap_values(
        &self,
        Parameters(request): Parameters<ShapValuesRequest>,
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

        // Limit observations if requested
        let max_obs = request.max_obs.unwrap_or(data.nrows());
        let n_obs = data.nrows().min(max_obs);
        let data_subset = data.slice(ndarray::s![..n_obs, ..]).to_owned();

        // Use per-tool seed if provided, otherwise fall back to global seed
        let global_seed = self.global_seed.read().await;
        let seed = request.seed.or(*global_seed);

        // Train Random Forest and compute SHAP values
        let n_trees = request.n_trees.unwrap_or(50);
        let max_depth = request.max_depth.unwrap_or(6);

        // Train RF with tree storage for SHAP
        let model = match random_forest_with_trees(
            data.view(),
            target.view(),
            Some(n_trees),
            Some(max_depth),
            Some(2),
            Some("sqrt"),
            seed,
            Some(request.features.clone()),
        ) {
            Ok(m) => m,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Random Forest training failed: {}",
                    e
                ))]));
            }
        };

        // Compute SHAP values
        let shap_config = ShapConfig {
            n_samples: request.n_samples,
            seed,
            check_additivity: true,
            ..Default::default()
        };

        let shap_result = match shap_values_model(&model, data_subset.view(), &shap_config) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "SHAP value computation failed: {}",
                    e
                ))]));
            }
        };

        // Compute summary if requested
        let summary = if request.compute_summary.unwrap_or(true) {
            Some(shap_summary(&shap_result))
        } else {
            None
        };

        // Build SHAP values sample before json! macro
        let n_show = shap_result.n_obs.min(5);
        let shap_values_sample: Vec<serde_json::Value> = (0..n_show)
            .map(|i| {
                let row = shap_result.shap_values.row(i);
                let values: Vec<serde_json::Value> = shap_result
                    .feature_names
                    .as_ref()
                    .map(|names| {
                        names
                            .iter()
                            .zip(row.iter())
                            .map(|(name, &val)| {
                                serde_json::json!({
                                    "feature": name,
                                    "shap_value": val
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_else(|| {
                        row.iter()
                            .enumerate()
                            .map(|(j, &val)| {
                                serde_json::json!({
                                    "feature": format!("Feature_{}", j),
                                    "shap_value": val
                                })
                            })
                            .collect()
                    });
                serde_json::json!(values)
            })
            .collect();

        // Build feature importance list
        let feature_importance_list: Vec<serde_json::Value> = shap_result
            .feature_names
            .as_ref()
            .map(|names| {
                names
                    .iter()
                    .zip(shap_result.feature_importance.iter())
                    .map(|(name, &imp)| {
                        serde_json::json!({
                            "feature": name,
                            "mean_abs_shap": imp
                        })
                    })
                    .collect()
            })
            .unwrap_or_else(|| {
                shap_result
                    .feature_importance
                    .iter()
                    .enumerate()
                    .map(|(i, &imp)| {
                        serde_json::json!({
                            "feature": format!("Feature_{}", i),
                            "mean_abs_shap": imp
                        })
                    })
                    .collect()
            });

        // Build summary if available
        let summary_json = summary.map(|s| {
            let top_features: Vec<serde_json::Value> = s
                .feature_names
                .iter()
                .zip(s.mean_abs_shap.iter())
                .zip(s.std_shap.iter())
                .zip(s.importance_rank.iter())
                .map(|(((name, &mean_abs), &std), &rank)| {
                    serde_json::json!({
                        "feature": name,
                        "mean_abs_shap": mean_abs,
                        "std_shap": std,
                        "rank": rank
                    })
                })
                .collect();
            serde_json::json!({ "top_features": top_features })
        });

        // Create output
        let output = serde_json::json!({
            "n_observations": shap_result.n_obs,
            "n_features": shap_result.n_features,
            "base_value": shap_result.base_value,
            "additivity_check": shap_result.additivity_check_passed,
            "max_additivity_error": shap_result.max_additivity_error,
            "model_info": {
                "n_trees": n_trees,
                "max_depth": max_depth,
                "oob_score": model.oob_score
            },
            "feature_importance": feature_importance_list,
            "summary": summary_json,
            "shap_values_sample": shap_values_sample,
            "note": if n_obs < data.nrows() {
                Some(format!("SHAP values computed for {} of {} observations", n_obs, data.nrows()))
            } else {
                None::<String>
            }
        });

        Ok(CallToolResult::success(vec![Content::text(format!(
            "{}\n\nJSON Summary:\n{}",
            shap_result,
            serde_json::to_string_pretty(&output).unwrap_or_default()
        ))]))
    }

    // NOTE: ICE (Individual Conditional Expectation) curves are not yet implemented.
    // TODO: Implement compute_ice_curves() and IceConfig in p2a-core.

    // ========================================================================
    // Statistical Decision Trees
    // ========================================================================

    /// Run Conditional Inference Trees (ctree) for regression or classification.
    #[tool(
        description = "Conditional Inference Trees (ctree) - decision trees with unbiased variable selection via permutation tests. Based on Hothorn, Hornik & Zeileis (2006). Unlike CART, ctree uses statistical significance tests for variable selection, avoiding bias towards variables with many possible split points. The tree grows until no significant relationship exists between predictors and response (no pruning needed). Equivalent to R's partykit::ctree."
    )]
    pub async fn ml_ctree(
        &self,
        Parameters(request): Parameters<CtreeRequest>,
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

        // Build configuration
        let config = CtreeConfig {
            mincriterion: request.mincriterion.unwrap_or(0.95),
            minsplit: request.minsplit.unwrap_or(20),
            minbucket: request.minbucket.unwrap_or(7),
            maxdepth: request.maxdepth.unwrap_or(0),
            teststat: request
                .teststat
                .clone()
                .unwrap_or_else(|| "quadratic".to_string()),
            testtype: request
                .testtype
                .clone()
                .unwrap_or_else(|| "bonferroni".to_string()),
            seed: request.seed,
        };

        let feature_refs: Vec<&str> = request.features.iter().map(|s| s.as_str()).collect();

        let result = match run_ctree(dataset, &request.target, &feature_refs, &config) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Conditional Inference Trees failed: {}",
                    e
                ))]));
            }
        };

        // Build tree structure for JSON output
        fn node_to_json(
            node: &p2a_core::CtreeNode,
            feature_names: &Option<Vec<String>>,
        ) -> serde_json::Value {
            let mut obj = serde_json::json!({
                "id": node.id,
                "n_samples": node.n,
                "prediction": node.prediction
            });

            if let Some(ref split) = node.split {
                let var_name = feature_names
                    .as_ref()
                    .and_then(|names| names.get(split.feature))
                    .cloned()
                    .unwrap_or_else(|| format!("X{}", split.feature));
                obj["split"] = serde_json::json!({
                    "feature_index": split.feature,
                    "feature_name": var_name,
                    "threshold": split.threshold,
                    "p_value": split.p_value,
                    "statistic": split.statistic
                });
            }

            if let (Some(left), Some(right)) = (&node.left, &node.right) {
                obj["left"] = node_to_json(left, feature_names);
                obj["right"] = node_to_json(right, feature_names);
            }

            if let Some(ref probs) = node.class_probs {
                obj["class_probabilities"] = serde_json::json!(probs);
            }

            obj
        }

        let n_features = result.variable_importance.len();
        let output = serde_json::json!({
            "n_nodes": result.n_nodes,
            "n_features": n_features,
            "is_classification": result.is_classification,
            "n_terminal_nodes": result.n_terminal,
            "depth": result.depth,
            "variable_importance": result.variable_importance,
            "feature_names": result.feature_names,
            "tree_structure": node_to_json(&result.root, &result.feature_names),
            "config": {
                "mincriterion": config.mincriterion,
                "minsplit": config.minsplit,
                "minbucket": config.minbucket,
                "maxdepth": config.maxdepth,
                "teststat": config.teststat,
                "testtype": config.testtype
            }
        });

        Ok(CallToolResult::success(vec![Content::text(format!(
            "{}\n\nJSON Summary:\n{}",
            result,
            serde_json::to_string_pretty(&output).unwrap_or_default()
        ))]))
    }

    // ========================================================================
    // Model-based Boosting (mboost)
    // ========================================================================

    /// Run Model-based Boosting (mboost).
    #[tool(
        description = "Run Model-based Boosting (mboost), a component-wise gradient boosting method that selects one variable per iteration. Supports Gaussian (regression), Binomial (classification), and Poisson (count data) families with linear or tree base learners. Equivalent to R's mboost package. Returns variable importance via selection frequency and automatic variable selection."
    )]
    pub async fn ml_mboost(
        &self,
        Parameters(request): Parameters<MboostRequest>,
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

        // Parse family
        let family = match request.family.as_deref() {
            Some("binomial") | Some("logistic") => MboostFamily::Binomial,
            Some("poisson") | Some("count") => MboostFamily::Poisson,
            Some("gaussian") | Some("normal") | None => MboostFamily::Gaussian,
            Some(f) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown family '{}'. Valid options: 'gaussian', 'binomial', 'poisson'.",
                    f
                ))]));
            }
        };

        // Parse base learner
        let base_learner = match request.base_learner.as_deref() {
            Some("tree") | Some("btree") | Some("stump") => MboostBaseLearner::Tree,
            Some("linear") | Some("bols") | None => MboostBaseLearner::Linear,
            Some(bl) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown base learner '{}'. Valid options: 'linear', 'tree'.",
                    bl
                ))]));
            }
        };

        // Build config
        let config = MboostConfig {
            mstop: request.mstop.unwrap_or(100),
            nu: request.nu.unwrap_or(0.1),
            family,
            base_learner,
            tree_depth: request.tree_depth.unwrap_or(1),
            min_samples_split: request.min_samples_split.unwrap_or(5),
            cv_folds: request.cv_folds,
            center: request.center.unwrap_or(true),
            seed: request.seed,
        };

        // Convert column names to &str
        let x_cols: Vec<&str> = request.x_cols.iter().map(|s| s.as_str()).collect();

        let result = match run_mboost(dataset, &request.y_col, &x_cols, &config) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "mboost failed: {}",
                    e
                ))]));
            }
        };

        // Build JSON summary
        let output = serde_json::json!({
            "family": result.config.family.to_string(),
            "base_learner": result.config.base_learner.to_string(),
            "iterations": result.iterations,
            "learning_rate": result.config.nu,
            "n_selected": result.n_selected,
            "n_features": result.coefficients.len(),
            "final_loss": result.final_loss,
            "intercept": result.intercept,
            "coefficients": result.coefficients,
            "variable_importance": result.variable_importance,
            "selected_variables": result.selected_variables,
            "selection_frequency": result.selection_frequency,
            "cv_optimal_mstop": result.cv_optimal_mstop,
            "feature_names": result.feature_names,
            "loss_history_summary": {
                "initial": result.loss_history.first(),
                "final": result.loss_history.last(),
                "min": result.loss_history.iter().cloned().fold(f64::INFINITY, f64::min),
            }
        });

        Ok(CallToolResult::success(vec![Content::text(format!(
            "{}\n\nJSON Summary:\n{}",
            result,
            serde_json::to_string_pretty(&output).unwrap_or_default()
        ))]))
    }
}
