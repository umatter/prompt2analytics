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

/// Request for Gradient Boosting Machine (GBM).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GbmRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Feature column names
    #[schemars(description = "Names of the feature columns (X variables).")]
    pub features: Vec<String>,

    /// Target column name
    #[schemars(description = "Name of the target column (Y variable).")]
    pub target: String,

    /// Number of boosting iterations/trees (default: 100)
    #[schemars(description = "Number of boosting iterations (trees). Default is 100.")]
    pub n_trees: Option<usize>,

    /// Learning rate (default: 0.1)
    #[schemars(
        description = "Learning rate (shrinkage). Smaller values require more trees but often give better results. Default is 0.1."
    )]
    pub learning_rate: Option<f64>,

    /// Maximum tree depth (default: 3)
    #[schemars(description = "Maximum depth of individual trees. Default is 3.")]
    pub max_depth: Option<usize>,

    /// Minimum samples to split (default: 10)
    #[schemars(description = "Minimum samples required to split a node. Default is 10.")]
    pub min_samples_split: Option<usize>,

    /// Subsample fraction (default: 1.0)
    #[schemars(
        description = "Fraction of samples to use for each tree (stochastic gradient boosting). Default is 1.0."
    )]
    pub subsample: Option<f64>,

    /// Distribution/family
    #[schemars(
        description = "Loss function: 'gaussian' for regression (MSE), 'huber' for robust regression, 'binomial' for classification. Default is 'gaussian'."
    )]
    pub family: Option<String>,

    /// Random seed for reproducibility
    #[schemars(description = "Optional random seed for reproducible results.")]
    pub seed: Option<u64>,
}

/// Request for AdaBoost.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct AdaBoostRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Feature column names
    #[schemars(description = "Names of the feature columns (X variables).")]
    pub features: Vec<String>,

    /// Target column name
    #[schemars(
        description = "Name of the target column. For classification (M1/SAMME), should be binary (-1/1 or 0/1). For regression (R2), any numeric."
    )]
    pub target: String,

    /// Number of estimators/iterations (default: 50)
    #[schemars(description = "Number of boosting iterations (weak learners). Default is 50.")]
    pub n_estimators: Option<usize>,

    /// AdaBoost type
    #[schemars(
        description = "AdaBoost variant: 'm1' for binary classification (Freund & Schapire), 'r2' for regression, 'samme' for multi-class. Default is 'm1'."
    )]
    pub boost_type: Option<String>,

    /// Maximum depth of weak learners (default: 1)
    #[schemars(description = "Maximum depth of base decision trees. Default is 1 (stumps).")]
    pub max_depth: Option<usize>,

    /// Learning rate (default: 1.0)
    #[schemars(
        description = "Weight applied to each classifier. Smaller values require more estimators. Default is 1.0."
    )]
    pub learning_rate: Option<f64>,

    /// Random seed for reproducibility
    #[schemars(description = "Optional random seed for reproducible results.")]
    pub seed: Option<u64>,
}

/// Request for CART decision tree.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CartRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Feature column names
    #[schemars(description = "Names of the feature columns (X variables).")]
    pub features: Vec<String>,

    /// Target column name
    #[schemars(description = "Name of the target column (Y variable).")]
    pub target: String,

    /// Method/objective
    #[schemars(
        description = "Splitting criterion: 'anova' for regression (MSE), 'gini' for classification (Gini impurity), 'entropy' for classification (information gain). Default is 'anova'."
    )]
    pub method: Option<String>,

    /// Maximum tree depth (default: 30)
    #[schemars(description = "Maximum depth of the tree. Default is 30.")]
    pub max_depth: Option<usize>,

    /// Minimum samples to split (default: 20)
    #[schemars(description = "Minimum samples required to split a node. Default is 20.")]
    pub min_split: Option<usize>,

    /// Minimum samples per leaf (default: 7)
    #[schemars(description = "Minimum samples in a terminal node. Default is 7.")]
    pub min_bucket: Option<usize>,

    /// Complexity parameter (default: 0.01)
    #[schemars(
        description = "Complexity parameter for cost-complexity pruning. Larger values = simpler trees. Default is 0.01."
    )]
    pub cp: Option<f64>,

    /// Number of cross-validation folds (default: 10)
    #[schemars(
        description = "Number of cross-validation folds for pruning. Set to 0 to disable. Default is 10."
    )]
    pub xval: Option<usize>,
}

/// Request for Kernel SVM.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct KernelSvmRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Feature column names
    #[schemars(description = "Names of the feature columns (X variables).")]
    pub features: Vec<String>,

    /// Target column name
    #[schemars(
        description = "Name of the binary target column. Must have exactly 2 unique values."
    )]
    pub target: String,

    /// Kernel type
    #[schemars(
        description = "Kernel function: 'linear', 'rbf' (Gaussian), 'polynomial', or 'sigmoid'. Default is 'rbf'."
    )]
    pub kernel: Option<String>,

    /// Regularization parameter C (default: 1.0)
    #[schemars(
        description = "Regularization parameter C. Larger values = less regularization. Default is 1.0."
    )]
    pub c: Option<f64>,

    /// Kernel coefficient gamma
    #[schemars(
        description = "Kernel coefficient for 'rbf', 'polynomial', 'sigmoid'. Default is 1/n_features."
    )]
    pub gamma: Option<f64>,

    /// Polynomial degree
    #[schemars(description = "Degree for 'polynomial' kernel. Default is 3.")]
    pub degree: Option<usize>,

    /// Polynomial/sigmoid coefficient
    #[schemars(description = "Independent term in polynomial/sigmoid kernel. Default is 0.")]
    pub coef0: Option<f64>,

    /// Maximum iterations (default: 1000)
    #[schemars(description = "Maximum number of iterations. Default is 1000.")]
    pub max_iterations: Option<usize>,

    /// Convergence tolerance (default: 1e-3)
    #[schemars(description = "Convergence tolerance. Default is 0.001.")]
    pub tolerance: Option<f64>,
}

/// Request for ROC curve and AUC calculation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RocAucRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Predicted probabilities column
    #[schemars(description = "Name of the column containing predicted probabilities (0-1 scale).")]
    pub predictions: String,

    /// Actual binary labels column
    #[schemars(description = "Name of the column containing actual binary labels (0/1 or -1/1).")]
    pub actual: String,

    /// Number of thresholds for ROC curve (default: 100)
    #[schemars(description = "Number of threshold points for ROC curve. Default is 100.")]
    pub n_thresholds: Option<usize>,
}

/// Request for variable importance (permutation-based).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct VariableImportanceRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Feature column names
    #[schemars(description = "Names of the feature columns.")]
    pub features: Vec<String>,

    /// Target column name
    #[schemars(description = "Name of the target column.")]
    pub target: String,

    /// Model type
    #[schemars(
        description = "Model to use: 'rf' (random forest), 'gbm', or 'cart'. Default is 'rf'."
    )]
    pub model: Option<String>,

    /// Number of permutations (default: 10)
    #[schemars(
        description = "Number of permutations per feature for importance estimation. Default is 10."
    )]
    pub n_permutations: Option<usize>,

    /// Random seed for reproducibility
    #[schemars(description = "Optional random seed for reproducible results.")]
    pub seed: Option<u64>,
}

/// Request for Partial Dependence Plot.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PartialDependenceRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Feature column names for model training
    #[schemars(description = "Names of all feature columns used in the model.")]
    pub features: Vec<String>,

    /// Target column name
    #[schemars(description = "Name of the target column.")]
    pub target: String,

    /// Feature(s) for partial dependence
    #[schemars(
        description = "Name(s) of feature(s) for partial dependence. Use one for 1D plot, two for 2D."
    )]
    pub pd_features: Vec<String>,

    /// Model type
    #[schemars(
        description = "Model to use: 'rf' (random forest), 'gbm', or 'cart'. Default is 'rf'."
    )]
    pub model: Option<String>,

    /// Number of grid points (default: 20)
    #[schemars(
        description = "Number of grid points for evaluating partial dependence. Default is 20."
    )]
    pub grid_resolution: Option<usize>,

    /// Random seed for reproducibility
    #[schemars(description = "Optional random seed for reproducible results.")]
    pub seed: Option<u64>,
}
