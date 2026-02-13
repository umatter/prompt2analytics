//! Machine learning module.
//!
//! This module provides 40+ machine learning methods for clustering, dimensionality
//! reduction, supervised learning, and causal ML. All algorithms are implemented in
//! pure Rust with automatic performance optimization for large datasets.
//!
//! ## Clustering (25+ methods)
//!
//! | Method | Function | Description |
//! |--------|----------|-------------|
//! | **K-Means** | [`kmeans`] | Lloyd's algorithm with k-means++ init |
//! | **Mini-Batch K-Means** | [`mini_batch_kmeans`] | Scalable k-means for large data |
//! | **Trimmed K-Means** | [`trimmed_kmeans`] | Robust k-means with outlier trimming |
//! | **K-Medoids (PAM)** | [`kmedoids`] | Partitioning Around Medoids |
//! | **CLARA** | [`clara`] | K-medoids for large datasets |
//! | **DBSCAN** | [`dbscan`] | Density-based clustering |
//! | **HDBSCAN** | [`hdbscan`] | Hierarchical DBSCAN |
//! | **OPTICS** | [`optics`] | Ordering points for cluster structure |
//! | **Hierarchical** | [`hierarchical`] | Agglomerative clustering |
//! | **AGNES** | [`agnes`] | Agglomerative nesting |
//! | **DIANA** | [`diana`] | Divisive analysis |
//! | **FastCluster** | [`fastcluster`] | Fast hierarchical (O(n²) memory) |
//! | **Dynamic Tree Cut** | [`dynamic_tree_cut`] | Adaptive dendrogram cutting |
//! | **Spectral** | [`spectral_clustering`] | Graph Laplacian clustering |
//! | **Affinity Propagation** | [`affinity_propagation`] | Message-passing clustering |
//! | **Gaussian Mixture** | [`gaussian_mixture`] | EM-based GMM |
//! | **FlexMix** | [`flexmix`] | Flexible mixture models |
//! | **Fuzzy C-Means** | [`fuzzy_cmeans`] | Soft clustering |
//! | **FANNY** | [`fanny`] | Fuzzy analysis |
//! | **Spherical K-Means** | [`skmeans`] | For directional data |
//! | **K-Prototypes** | [`kprototypes`] | Mixed numeric/categorical |
//! | **PV-Clust** | [`pvclust`] | Bootstrap cluster validation |
//! | **Normal Mixture EM** | [`normal_mix_em`] | Univariate mixture models |
//! | **MV Normal Mixture** | [`mvnorm_mix_em`] | Multivariate mixture models |
//!
//! ## Cluster Validation
//!
//! | Method | Function | Description |
//! |--------|----------|-------------|
//! | **Silhouette** | [`silhouette`] | Cluster cohesion/separation |
//! | **Calinski-Harabasz** | [`calinski_harabasz`] | Variance ratio criterion |
//! | **Davies-Bouldin** | [`davies_bouldin`] | Inter/intra cluster ratio |
//! | **Dunn Index** | [`dunn_index`] | Min inter / max intra |
//! | **Rand Index** | [`rand_index`] | Agreement with ground truth |
//! | **NMI** | [`nmi`] | Normalized mutual information |
//! | **Gap Statistic** | [`gap_statistic`] | Optimal cluster count |
//! | **Cluster Stats** | [`cluster_stats`] | Comprehensive cluster statistics |
//!
//! ## Dimensionality Reduction
//!
//! | Method | Function | Description |
//! |--------|----------|-------------|
//! | **PCA** | [`pca`] | Principal component analysis |
//! | **t-SNE** | [`tsne`] | t-distributed stochastic neighbor embedding |
//! | **MDS** | [`cmdscale`] | Classical multidimensional scaling |
//!
//! ## Supervised Learning
//!
//! | Method | Function | Description |
//! |--------|----------|-------------|
//! | **BART** | [`bart`] | Bayesian Additive Regression Trees (prediction) |
//! | **CART** | [`cart`] | Decision trees (classification/regression) |
//! | **Random Forest** | [`random_forest`] | CART-based ensemble |
//! | **Gradient Boosting** | [`gbm`] | Gradient boosting machine (GBM) |
//! | **AdaBoost** | [`adaboost`] | Adaptive boosting (M1, R2, SAMME) |
//! | **Linear SVM** | [`linear_svm`] | Support vector machine (SMO) |
//! | **PPR** | [`ppr`] | Projection pursuit regression |
//!
//! ## Causal Machine Learning
//!
//! | Method | Function | Description |
//! |--------|----------|-------------|
//! | **Causal Forest** | [`causal_forest`] | Heterogeneous treatment effects (grf) |
//! | **BART Causal** | [`bart_causal`] | Bayesian trees for causal inference |
//!
//! ## Large-N Performance
//!
//! HDBSCAN and OPTICS automatically select optimal algorithms for large datasets:
//! - n < 500: O(n²) brute-force
//! - n < 10,000 && d <= 15: O(n log n) KD-Tree + Prim's
//! - n >= 10,000 && d <= 15: O(n log n) Dual-Tree Boruvka
//! - d > 15: O(n²) parallel brute-force
//!
//! ## Example
//!
//! ```
//! use p2a_core::ml::{kmeans, silhouette, pca};
//! use ndarray::Array2;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let data: Array2<f64> = Array2::from_shape_vec(
//!     (100, 3),
//!     (0..300).map(|x| x as f64).collect()
//! )?;
//!
//! // Cluster with k-means
//! let clusters = kmeans(&data.view(), 3, Some(100), Some(42))?;
//! println!("Cluster sizes: {:?}", clusters.cluster_sizes);
//!
//! // Validate clusters
//! let sil = silhouette(&data.view(), &clusters.labels)?;
//! println!("Silhouette score: {:.3}", sil.average_silhouette);
//!
//! // Reduce dimensions with PCA (data, n_components, transform)
//! let pca_result = pca(data.view(), Some(2), false)?;
//! println!("Variance explained: {:.1}%", pca_result.explained_variance_ratio.sum() * 100.0);
//! # Ok(())
//! # }
//! ```
//!
//! ## R Package Equivalents
//!
//! | R Package | p2a-core Functions |
//! |-----------|-------------------|
//! | `stats` (kmeans) | [`kmeans`], [`hierarchical`], [`cmdscale`] |
//! | `cluster` | [`kmedoids`], [`agnes`], [`diana`], [`clara`], [`fanny`], [`silhouette`] |
//! | `dbscan` | [`dbscan`], [`hdbscan`], [`optics`] |
//! | `mclust` | [`gaussian_mixture`], [`normal_mix_em`] |
//! | `fastcluster` | [`fastcluster`] |
//! | `dynamicTreeCut` | [`dynamic_tree_cut`] |
//! | `flexmix` | [`flexmix`] |
//! | `pvclust` | [`pvclust`] |
//! | `grf` | [`causal_forest`] |
//! | `BART` | [`bart`], [`bart_causal`] |

mod adaboost;
mod advanced_clustering_mod;
mod bart;
mod bart_causal;
mod c50;
mod cart;
mod causal_forest;
mod cluster_optimized;
mod cluster_validation;
mod clustering;
mod ctree;
mod cubist;
pub mod dual_tree;
mod gbm;
mod ice;
pub mod kdtree;
mod mboost;
mod pdp;
pub mod ppr;
mod quantile_rf;
mod reduction;
mod shap;
mod svm;
pub mod trees;

/// Simple Linear Congruential Generator for reproducible randomness.
///
/// Used across ML modules for deterministic random sampling (bootstrap,
/// feature subsampling, bagging, etc.) without external RNG dependencies.
pub(crate) fn lcg_random(state: &mut u64) -> usize {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) ^ *state) as usize
}

pub use adaboost::{
    AdaBoostConfig, AdaBoostLoss, AdaBoostResult, AdaBoostType, adaboost, adaboost_predict,
    adaboost_predict_class, run_adaboost, run_adaboost_default,
};
pub use advanced_clustering_mod::{
    AffinityPropagationResult,
    AffinityType,
    AgnesLinkage,
    AgnesResult,
    ClaraResult,
    ClusterStatsResult,
    ClusteringAlgorithm,
    CovarianceType,
    DianaResult,
    DynamicCutMethod,
    DynamicTreeCutResult,
    FannyResult,
    FastClusterResult,
    FastLinkage,
    FcmDistance,
    FlexMixResult,
    FuzzyCMeansResult,
    GaussianMixtureResult,
    HdbscanResult,
    KMedoidsResult,
    KPrototypesResult,
    MiniBatchKMeansResult,
    MultivariateNormalMixEMResult,
    NormalMixEMResult,
    OpticsResult,
    PvClustResult,
    SKMeansResult,
    SpectralClusteringResult,
    TrimmedKMeansResult,
    affinity_propagation,
    agnes,
    clara,
    cluster_stats,
    diana,
    dynamic_tree_cut,
    fanny,
    fastcluster,
    // Batch 2: Additional clustering methods
    flexmix,
    fuzzy_cmeans,
    gaussian_mixture,
    hdbscan,
    kmedoids,
    kprototypes,
    mini_batch_kmeans,
    mvnorm_mix_em,
    normal_mix_em,
    optics,
    pvclust,
    run_affinity_propagation,
    run_agnes,
    run_clara,
    run_cluster_stats,
    run_diana,
    run_dynamic_tree_cut,
    run_fanny,
    run_fastcluster,
    run_flexmix,
    run_fuzzy_cmeans,
    run_gaussian_mixture,
    run_hdbscan,
    run_kmedoids,
    run_kprototypes,
    run_mini_batch_kmeans,
    run_mvnorm_mix_em,
    run_normal_mix_em,
    run_optics,
    run_pvclust,
    run_skmeans,
    run_spectral_clustering,
    run_trimmed_kmeans,
    // Batch 3: skmeans, fastcluster, dynamicTreeCut, mixtools, kprototypes
    skmeans,
    spectral_clustering,
    trimmed_kmeans,
};
pub use bart::{BartConfig, BartResult, bart, bart_arrays, run_bart};
pub use bart_causal::{
    BartCausalConfig, BartCausalResult, bart_causal, bart_causal_arrays, bart_causal_predict,
    bart_causal_predict_arrays, run_bart_causal,
};
pub use c50::{
    C50Config, C50Node, C50Result, C50Rule, C50RuleCondition, C50Split, ComparisonOp, c50,
    c50_predict, c50_predict_proba, run_c50, run_c50_default,
};
pub use cart::{
    CartConfig, CartMethod, CartNode, CartResult, CartSplit, CpTableRow, cart, cart_predict,
    cart_prune, run_cart, run_cart_default,
};
pub use causal_forest::{
    CausalForestConfig, CausalForestResult, average_treatment_effect, causal_forest,
    causal_forest_arrays, causal_forest_predict, causal_forest_predict_arrays, run_causal_forest,
};
pub use cluster_optimized::{
    affinity_propagation_optimized, compute_pairwise_distances_parallel,
    compute_pairwise_sq_distances_parallel, hdbscan_optimized, kmedoids_optimized,
    optics_optimized, silhouette_optimized,
};
pub use cluster_validation::{
    CalinskiHarabaszResult, DaviesBouldinResult, DunnIndexResult, GapStatisticResult, NmiResult,
    RandIndexResult, SilhouetteInfo, SilhouetteResult, calinski_harabasz, davies_bouldin,
    dunn_index, gap_statistic, nmi, rand_index, run_calinski_harabasz, run_davies_bouldin,
    run_dunn_index, run_gap_statistic, run_nmi, run_rand_index, run_silhouette, silhouette,
    silhouette_from_dist,
};
pub use clustering::{
    CutreeResult, DBSCANResult, HierarchicalResult, KMeansResult, Linkage, cutree,
    cutree_multiple_k, dbscan, hierarchical, kmeans, run_cutree,
};
pub use ctree::{
    CtreeConfig, CtreeNode, CtreeResult, CtreeSplit, ctree, ctree_predict, ctree_predict_proba,
    run_ctree, run_ctree_default,
};
pub use cubist::{
    CubistConfig, CubistResult, CubistRule, RuleCondition, cubist, cubist_predict, run_cubist,
    run_cubist_default,
};
pub use dual_tree::{DualTreeMstResult, dual_tree_boruvka_mst, kdtree_prim_mst};
pub use gbm::{GbmConfig, GbmFamily, GbmResult, gbm, gbm_predict, run_gbm, run_gbm_default};
pub use kdtree::{KdNode, KdTree, UnionFind, build_connected_mst, euclidean_distance, kruskal_mst};
pub use mboost::{
    CoefficientPathEntry, MboostBaseLearner, MboostConfig, MboostFamily, MboostResult, mboost,
    mboost_cv, mboost_predict, run_mboost, run_mboost_default,
};
pub use ppr::{PprConfig, PprResult, SmoothingMethod, ppr, run_ppr};
pub use quantile_rf::{
    QuantileRfConfig, QuantileRfResult, predict_quantiles, predict_quantiles_at,
    predict_quantiles_with_y, predict_quantiles_with_y_at, prediction_intervals, quantile_rf,
    quantile_rf_with_names, run_quantile_rf, run_quantile_rf_default,
};
pub use reduction::{
    CmdscaleResult, PCAResult, TsneResult, cmdscale, cmdscale_from_data, pca,
    pca_inverse_transform, pca_transform, run_cmdscale, tsne,
};
pub use shap::{
    FeaturePerturbation, ShapConfig, ShapResult, ShapSummary, kernel_shap, run_kernel_shap,
    run_shap_values_model, shap_kernel, shap_summary, shap_tree_ensemble, shap_values_model,
    shap_values_random_forest,
};
pub use svm::{SvmResult, linear_svm, svm_predict};
pub use trees::{
    DecisionTree, RandomForestModel, RandomForestResult, TreeNode, random_forest,
    random_forest_with_trees,
};
// ICE curves (Individual Conditional Expectation) - enhanced version with heterogeneity analysis
pub use ice::{
    IceConfig, IceResult, IceSpread, compute_ice_curves, ice_curves_cart, ice_curves_gbm,
    ice_curves_rf,
};
// PDP (Partial Dependence Plot) - original version
pub use pdp::{ice_curves, partial_dependence};
