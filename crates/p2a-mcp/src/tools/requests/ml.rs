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
