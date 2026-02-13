//! Machine learning tool request types.

use schemars::JsonSchema;
use serde::Deserialize;

/// Request for K-means clustering.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct KMeansRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to use as features for clustering
    #[schemars(description = "Names of the numeric columns to use as features for clustering.")]
    pub columns: Vec<String>,

    /// Number of clusters (k)
    #[schemars(description = "Number of clusters to create.")]
    pub k: usize,

    /// Maximum iterations (optional, default: 300)
    #[schemars(description = "Maximum number of iterations. Default is 300.")]
    pub max_iterations: Option<usize>,

    /// Number of initializations (optional, default: 10)
    #[schemars(description = "Number of random initializations to try. Default is 10.")]
    pub n_init: Option<usize>,

    /// Random seed for reproducibility (optional)
    #[schemars(description = "Optional random seed for reproducible results.")]
    pub seed: Option<u64>,
}

/// Request for DBSCAN clustering.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DBSCANRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to use as features for clustering
    #[schemars(description = "Names of the numeric columns to use as features for clustering.")]
    pub columns: Vec<String>,

    /// Epsilon (neighborhood radius)
    #[schemars(
        description = "Maximum distance between two samples for them to be considered in the same neighborhood."
    )]
    pub eps: f64,

    /// Minimum samples for core point
    #[schemars(
        description = "Minimum number of samples in a neighborhood for a point to be considered a core point."
    )]
    pub min_samples: usize,
}

/// Request for PCA (Principal Component Analysis).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PCARequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to use as features
    #[schemars(description = "Names of the numeric columns to include in PCA.")]
    pub columns: Vec<String>,

    /// Number of principal components to keep (optional)
    #[schemars(
        description = "Number of principal components to keep. If not specified, keeps all components."
    )]
    pub n_components: Option<usize>,
}

/// Request for Hierarchical clustering.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct HierarchicalRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to use as features for clustering
    #[schemars(description = "Names of the numeric columns to use as features for clustering.")]
    pub columns: Vec<String>,

    /// Number of clusters to cut the dendrogram into (optional)
    #[schemars(
        description = "Number of clusters to create. If not specified, uses distance_threshold."
    )]
    pub n_clusters: Option<usize>,

    /// Distance threshold for cutting the dendrogram (optional)
    #[schemars(
        description = "Distance threshold for cutting. Used if n_clusters is not specified."
    )]
    pub distance_threshold: Option<f64>,

    /// Linkage method
    #[schemars(
        description = "Linkage method: 'single', 'complete', 'average', or 'ward'. Default is 'ward'."
    )]
    pub linkage: Option<String>,
}

/// Request for t-SNE dimensionality reduction.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct TsneRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to use as features
    #[schemars(description = "Names of the numeric columns to include in t-SNE.")]
    pub columns: Vec<String>,

    /// Number of output dimensions (default: 2)
    #[schemars(description = "Number of output dimensions. Default is 2.")]
    pub n_components: Option<usize>,

    /// Perplexity parameter (default: 30.0)
    #[schemars(
        description = "Perplexity parameter, related to number of nearest neighbors. Default is 30."
    )]
    pub perplexity: Option<f64>,

    /// Maximum iterations (default: 1000)
    #[schemars(description = "Maximum number of iterations. Default is 1000.")]
    pub max_iterations: Option<usize>,

    /// Learning rate (default: 200.0)
    #[schemars(description = "Learning rate for optimization. Default is 200.")]
    pub learning_rate: Option<f64>,

    /// Random seed for reproducibility
    #[schemars(description = "Optional random seed for reproducible results.")]
    pub seed: Option<u64>,
}

/// Request for Classical Multidimensional Scaling (cmdscale).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CmdscaleRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to use for computing distances
    #[schemars(
        description = "Names of the numeric columns to include. Euclidean distances will be computed from these columns."
    )]
    pub columns: Vec<String>,

    /// Number of output dimensions (default: 2)
    #[schemars(description = "Number of dimensions in the output configuration. Default is 2.")]
    pub k: Option<usize>,

    /// Whether input is already a distance matrix
    #[schemars(
        description = "Set to true if the input columns represent a distance matrix. Default is false (data is converted to distances)."
    )]
    pub is_distance_matrix: Option<bool>,
}

/// Request for cutting a hierarchical clustering tree (cutree).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CutreeRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to use for hierarchical clustering
    #[schemars(description = "Names of the numeric columns to cluster.")]
    pub columns: Vec<String>,

    /// Number of clusters to form
    #[schemars(
        description = "Number of clusters to cut the tree into. Takes priority over cut_height."
    )]
    pub k: Option<usize>,

    /// Height at which to cut the tree
    #[schemars(
        description = "Height (distance threshold) at which to cut the dendrogram. Ignored if k is specified."
    )]
    pub cut_height: Option<f64>,

    /// Linkage method for hierarchical clustering
    #[schemars(
        description = "Linkage method: 'single', 'complete', 'average', or 'ward'. Default is 'ward'."
    )]
    pub linkage: Option<String>,
}

/// Request for Random Forest regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RandomForestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Feature column names
    #[schemars(description = "Names of the feature columns (X variables).")]
    pub features: Vec<String>,

    /// Target column name
    #[schemars(description = "Name of the target column (Y variable).")]
    pub target: String,

    /// Number of trees (default: 100)
    #[schemars(description = "Number of trees in the forest. Default is 100.")]
    pub n_trees: Option<usize>,

    /// Maximum tree depth (default: 10)
    #[schemars(description = "Maximum depth of each tree. Default is 10.")]
    pub max_depth: Option<usize>,

    /// Minimum samples to split (default: 2)
    #[schemars(description = "Minimum samples required to split a node. Default is 2.")]
    pub min_samples_split: Option<usize>,

    /// Max features per split
    #[schemars(
        description = "Max features to consider per split: 'sqrt', 'log2', 'all', or a number. Default is 'sqrt'."
    )]
    pub max_features: Option<String>,

    /// Random seed for reproducibility
    #[schemars(description = "Optional random seed for reproducible results.")]
    pub seed: Option<u64>,
}

/// Request for Linear SVM classification.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SvmRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Feature column names
    #[schemars(description = "Names of the feature columns (X variables).")]
    pub features: Vec<String>,

    /// Target column name
    #[schemars(
        description = "Name of the binary target column (Y variable). Must have exactly 2 unique values."
    )]
    pub target: String,

    /// Regularization parameter C (default: 1.0)
    #[schemars(
        description = "Regularization parameter C. Larger values = less regularization. Default is 1.0."
    )]
    pub c: Option<f64>,

    /// Maximum iterations (default: 1000)
    #[schemars(description = "Maximum number of iterations. Default is 1000.")]
    pub max_iterations: Option<usize>,

    /// Convergence tolerance (default: 1e-3)
    #[schemars(description = "Convergence tolerance. Default is 0.001.")]
    pub tolerance: Option<f64>,
}

/// Request for Projection Pursuit Regression (PPR).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PprRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Feature column names
    #[schemars(description = "Names of the feature columns (X variables).")]
    pub features: Vec<String>,

    /// Target column name
    #[schemars(description = "Name of the target column (Y variable).")]
    pub target: String,

    /// Number of terms (default: 1)
    #[schemars(description = "Number of terms in the PPR model. Default is 1.")]
    pub nterms: Option<usize>,

    /// Maximum terms to consider (default: nterms)
    #[schemars(
        description = "Maximum terms to consider during forward selection. Default is nterms."
    )]
    pub max_terms: Option<usize>,

    /// Smoothing method
    #[schemars(
        description = "Smoothing method for ridge functions: 'supsmu' (default), 'spline', or 'gcvspline'."
    )]
    pub sm_method: Option<String>,

    /// Bass parameter for supsmu (0-10)
    #[schemars(
        description = "Bass parameter for supsmu smoothing (0-10). Higher = smoother. Default is 0."
    )]
    pub bass: Option<f64>,
}

/// Request for Conditional Inference Trees (ctree).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CtreeRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Feature column names
    #[schemars(description = "Names of the feature columns (X variables).")]
    pub features: Vec<String>,

    /// Target column name
    #[schemars(
        description = "Name of the target column (Y variable). Can be numeric (regression) or categorical (classification)."
    )]
    pub target: String,

    /// Criterion for splitting (1 - p-value threshold), default: 0.95 meaning p < 0.05
    #[schemars(
        description = "Value of 1 - p-value that must be exceeded to implement a split. Default is 0.95 (p < 0.05)."
    )]
    pub mincriterion: Option<f64>,

    /// Minimum observations in a node to attempt split, default: 20
    #[schemars(
        description = "Minimum number of observations in a node required to attempt a split. Default is 20."
    )]
    pub minsplit: Option<usize>,

    /// Minimum observations in terminal nodes, default: 7
    #[schemars(description = "Minimum number of observations in a terminal node. Default is 7.")]
    pub minbucket: Option<usize>,

    /// Maximum tree depth (0 = unlimited), default: 0
    #[schemars(description = "Maximum depth of the tree. 0 means unlimited. Default is 0.")]
    pub maxdepth: Option<usize>,

    /// Test statistic type: "quadratic" or "max"
    #[schemars(
        description = "Type of test statistic: 'quadratic' (chi-squared) or 'max' (maximum). Default is 'quadratic'."
    )]
    pub teststat: Option<String>,

    /// P-value adjustment: "bonferroni", "univariate", or "none"
    #[schemars(
        description = "P-value adjustment method: 'bonferroni', 'univariate', or 'none'. Default is 'bonferroni'."
    )]
    pub testtype: Option<String>,

    /// Random seed for reproducibility
    #[schemars(description = "Optional random seed for reproducible results.")]
    pub seed: Option<u64>,
}

/// Request for C5.0 Decision Tree classification.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct C50Request {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Target column name (categorical outcome)
    #[schemars(description = "Name of the target column (categorical Y variable).")]
    pub target: String,

    /// Feature column names
    #[schemars(description = "Names of the feature columns (X variables).")]
    pub features: Vec<String>,

    /// Number of boosting trials (default: 1 = no boosting)
    #[schemars(
        description = "Number of boosting iterations. Default is 1 (no boosting). Set > 1 for AdaBoost.M1 ensemble."
    )]
    pub trials: Option<usize>,

    /// Output rules instead of tree (default: false)
    #[schemars(
        description = "If true, extract rule-based model from tree. Default is false (tree output)."
    )]
    pub rules: Option<bool>,

    /// Enable feature winnowing (default: false)
    #[schemars(
        description = "If true, perform feature selection to remove uninformative predictors. Default is false."
    )]
    pub winnow: Option<bool>,

    /// Minimum cases per leaf (default: 2)
    #[schemars(description = "Minimum number of cases required in a leaf node. Default is 2.")]
    pub min_cases: Option<usize>,

    /// Confidence factor for pruning (default: 0.25)
    #[schemars(
        description = "Confidence factor for pessimistic error pruning (0-1). Lower = more pruning. Default is 0.25."
    )]
    pub cf: Option<f64>,

    /// Random seed for reproducibility
    #[schemars(description = "Optional random seed for reproducible results.")]
    pub seed: Option<u64>,
}

/// Request for computing SHAP values.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ShapValuesRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Feature column names
    #[schemars(description = "Names of the feature columns (X variables).")]
    pub features: Vec<String>,

    /// Target column name
    #[schemars(description = "Name of the target column (Y variable).")]
    pub target: String,

    /// Number of trees for the internal Random Forest model (default: 50)
    #[schemars(description = "Number of trees in the Random Forest. Default is 50.")]
    pub n_trees: Option<usize>,

    /// Maximum tree depth (default: 6)
    #[schemars(description = "Maximum depth of each tree. Default is 6.")]
    pub max_depth: Option<usize>,

    /// Maximum observations to compute SHAP for (default: all)
    #[schemars(
        description = "Maximum observations to compute SHAP values for. Default is all observations."
    )]
    pub max_obs: Option<usize>,

    /// Number of background samples for approximation (default: use all training data)
    #[schemars(
        description = "Number of background samples for SHAP approximation. If not specified, uses TreeSHAP exact computation."
    )]
    pub n_samples: Option<usize>,

    /// Whether to compute summary statistics (default: true)
    #[schemars(
        description = "Whether to compute global feature importance summary. Default is true."
    )]
    pub compute_summary: Option<bool>,

    /// Random seed for reproducibility
    #[schemars(description = "Optional random seed for reproducible results.")]
    pub seed: Option<u64>,
}

/// Request for Cubist rule-based regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CubistRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Feature column names
    #[schemars(description = "Names of the feature columns (X variables).")]
    pub features: Vec<String>,

    /// Target column name
    #[schemars(description = "Name of the target column (Y variable).")]
    pub target: String,

    /// Number of committees (boosted ensembles, default: 1)
    #[schemars(
        description = "Number of committee models (boosted ensemble). Default is 1 (no boosting). More committees can improve accuracy."
    )]
    pub committees: Option<usize>,

    /// Number of neighbors for instance-based correction (default: 0)
    #[schemars(
        description = "Number of nearest neighbors for prediction adjustment. Default is 0 (no adjustment). Higher values add k-NN smoothing."
    )]
    pub neighbors: Option<usize>,

    /// Maximum tree depth (default: 10)
    #[schemars(description = "Maximum depth of rule trees. Default is 10.")]
    pub max_depth: Option<usize>,

    /// Minimum samples to attempt a split (default: 10)
    #[schemars(description = "Minimum samples required to attempt a split. Default is 10.")]
    pub min_split: Option<usize>,

    /// Minimum samples in a leaf node (default: 5)
    #[schemars(description = "Minimum samples required in a leaf node. Default is 5.")]
    pub min_bucket: Option<usize>,

    /// Random seed for reproducibility
    #[schemars(description = "Optional random seed for reproducible results.")]
    pub seed: Option<u64>,
}

/// Request for Model-based Boosting (mboost).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MboostRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Target column (response variable)
    #[schemars(description = "Name of the target/response column.")]
    pub y_col: String,

    /// Feature columns
    #[schemars(description = "Names of the feature columns to use as predictors.")]
    pub x_cols: Vec<String>,

    /// Number of boosting iterations (default: 100)
    #[schemars(
        description = "Number of boosting iterations (mstop). Default is 100. Larger values may overfit."
    )]
    pub mstop: Option<usize>,

    /// Learning rate (default: 0.1)
    #[schemars(
        description = "Learning rate (nu). Default is 0.1. Smaller values require more iterations but often improve generalization."
    )]
    pub nu: Option<f64>,

    /// Loss function family
    #[schemars(
        description = "Loss function family: 'gaussian' (default, for regression), 'binomial' (for binary classification), or 'poisson' (for count data)."
    )]
    pub family: Option<String>,

    /// Base learner type
    #[schemars(
        description = "Base learner type: 'linear' (default, L2-boosting) or 'tree' (regression stumps)."
    )]
    pub base_learner: Option<String>,

    /// Maximum tree depth for tree base learner (default: 1)
    #[schemars(
        description = "Maximum depth for tree base learner. Default is 1 (stumps). Only used when base_learner='tree'."
    )]
    pub tree_depth: Option<usize>,

    /// Minimum samples for tree splits (default: 5)
    #[schemars(
        description = "Minimum samples required for a tree split. Default is 5. Only used when base_learner='tree'."
    )]
    pub min_samples_split: Option<usize>,

    /// Number of cross-validation folds for early stopping
    #[schemars(
        description = "Number of CV folds for finding optimal mstop. If not set, no CV is performed."
    )]
    pub cv_folds: Option<usize>,

    /// Whether to center predictors (default: true)
    #[schemars(description = "Whether to center predictors before fitting. Default is true.")]
    pub center: Option<bool>,

    /// Random seed for reproducibility
    #[schemars(description = "Optional random seed for reproducible results.")]
    pub seed: Option<u64>,
}
