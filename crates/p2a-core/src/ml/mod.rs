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
//! ## Model Interpretability
//!
//! | Method | Function | Description |
//! |--------|----------|-------------|
//! | **ICE Curves** | [`ice_curves`] | Individual Conditional Expectation |
//! | **LIME** | [`lime`] | Local Interpretable Model-agnostic Explanations |
//! | **SHAP** | [`shap_values`] | SHapley Additive exPlanations |
//!
//! ## Advanced Tree Methods
//!
//! | Method | Function | Description |
//! |--------|----------|-------------|
//! | **Quantile RF** | [`quantile_rf`] | Quantile regression forests |
//! | **CTree** | [`ctree`] | Conditional inference trees |
//! | **Boruta** | [`boruta`] | Feature selection with shadow features |
//! | **C5.0** | [`c50`] | C5.0 decision trees |
//! | **Cubist** | [`cubist`] | Rule-based regression |
//! | **MARS** | [`mars`] | Multivariate adaptive regression splines |
//!
//! ## Association Rules
//!
//! | Method | Function | Description |
//! |--------|----------|-------------|
//! | **Apriori** | [`apriori`] | Association rule mining |
//! | **Eclat** | [`eclat`] | Vertical data format mining |
//!
//! ## Gradient Boosting Methods
//!
//! | Method | Function | Description |
//! |--------|----------|-------------|
//! | **XGBoost** | [`xgboost`] | Extreme gradient boosting with L1/L2 regularization |
//! | **LightGBM** | [`lightgbm`] | Histogram-based gradient boosting |
//! | **MBoost** | [`mboost`] | Model-based gradient boosting framework |
//! | **BART** | [`bart`] | Bayesian additive regression trees |
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
//! // 90 observations in 3 dimensions: three tight blobs of 30 points each,
//! // centred on (0,0,0), (10,10,10) and (20,20,20).
//! let mut values: Vec<f64> = Vec::with_capacity(270);
//! for i in 0..90 {
//!     let centre = (i / 30) as f64 * 10.0;
//!     let jitter = (i % 30) as f64 * 0.01;
//!     values.extend_from_slice(&[centre + jitter, centre - jitter, centre + 2.0 * jitter]);
//! }
//! let data: Array2<f64> = Array2::from_shape_vec((90, 3), values)?;
//!
//! // Cluster with k-means (data, k, max_iterations, tolerance, n_init, seed)
//! let clusters = kmeans(data.view(), 3, Some(100), Some(1e-4), Some(10), Some(42))?;
//! println!("Cluster sizes: {:?}", clusters.cluster_sizes);
//! let mut sizes = clusters.cluster_sizes.clone();
//! sizes.sort();
//! assert_eq!(sizes, vec![30, 30, 30]); // each blob recovered exactly
//!
//! // Validate clusters: blobs are ~17 apart but only ~0.3 wide, so s(i) is close to 1
//! let sil = silhouette(data.view(), &clusters.labels)?;
//! println!("Silhouette score: {:.3}", sil.average_silhouette);
//! assert!(sil.average_silhouette > 0.9);
//!
//! // Reduce dimensions with PCA (data, n_components, transform)
//! let pca_result = pca(data.view(), Some(2), false)?;
//! println!("Variance explained: {:.1}%", pca_result.explained_variance_ratio.sum() * 100.0);
//! // The points lie exactly in the plane spanned by (1,1,1) and (1,-1,2), so two
//! // components capture all the variance; the between-blob direction dominates
//! // (between-blob variance 200 vs. within-blob 0.045).
//! assert!(pca_result.explained_variance_ratio[0] > 0.99);
//! assert!((pca_result.explained_variance_ratio.sum() - 1.0).abs() < 1e-6);
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
//! | `iml`, `DALEX` | [`ice_curves`], [`lime`], [`shap_values`] |
//! | `quantregForest` | [`quantile_rf`] |
//! | `party` | [`ctree`] |
//! | `Boruta` | [`boruta`] |
//! | `C50` | [`c50`] |
//! | `Cubist` | [`cubist`] |
//! | `earth` | [`mars`] |
//! | `arules` | [`apriori`], [`eclat`] |

mod adaboost;
mod advanced_clustering_mod;
pub mod apriori;
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
mod evaluation;
mod evaluation_fast;
mod gbm;
mod ice;
pub mod kdtree;
mod lightgbm;
mod mars;
mod mboost;
mod mboost_fast;
mod pdp;
pub mod ppr;
mod quantile_rf;
mod reduction;
mod shap;
mod svm;
mod svm_fast;
pub mod trees;
mod xgboost;
mod xgboost_fast;

/// Bytes a heap allocation of `size` bytes occupies under glibc malloc (the
/// allocator of the Linux builds): the size plus an 8-byte header, rounded up
/// to a multiple of 16, and at least 32. The memory bounds of the tree
/// ensembles count each boxed tree node at this size.
pub(crate) const fn heap_chunk_bytes(size: usize) -> u128 {
    let chunk = (size as u128 + 8).div_ceil(16) * 16;
    if chunk < 32 { 32 } else { chunk }
}

/// `a * b`, saturating at `u128::MAX`, for the terms of the tree ensembles' memory bounds.
pub(crate) fn mul(a: u128, b: u128) -> u128 {
    a.saturating_mul(b)
}

/// `2^d`, saturating at `u128::MAX`.
pub(crate) fn pow2(d: usize) -> u128 {
    1u128
        .checked_shl(d.min(u32::MAX as usize) as u32)
        .unwrap_or(u128::MAX)
}

/// The covariates a split of the causal forest or of BART draws by default from `p`:
/// `ceil(sqrt(p))`, and at least one.
pub(crate) fn default_mtry(p: usize) -> usize {
    ((p as f64).sqrt().ceil() as usize).max(1)
}

/// `indices` split into those `pred` holds for and the others, each in its original order, as
/// [`Iterator::partition`] splits them, but with each vector allocated at its exact length.
///
/// `partition` grows its vectors by pushing, so their capacity can be close to twice their
/// length; the memory bounds of the tree ensembles count their index vectors by their lengths.
/// `pred` is called twice for each index, so it must be a pure function of it.
pub(crate) fn partition_exact<I>(
    indices: I,
    pred: impl Fn(usize) -> bool,
) -> (Vec<usize>, Vec<usize>)
where
    I: ExactSizeIterator<Item = usize> + Clone,
{
    let n = indices.len();
    let n_true = indices.clone().filter(|&i| pred(i)).count();
    let mut yes = Vec::with_capacity(n_true);
    let mut no = Vec::with_capacity(n - n_true);
    for i in indices {
        if pred(i) {
            yes.push(i);
        } else {
            no.push(i);
        }
    }
    (yes, no)
}

/// Simple Linear Congruential Generator for reproducible randomness.
///
/// Used across ML modules for deterministic random sampling (bootstrap,
/// feature subsampling, bagging, etc.) without external RNG dependencies.
pub(crate) fn lcg_random(state: &mut u64) -> usize {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) ^ *state) as usize
}

/// Whether this test host matches the platform the ML golden (exact-bits)
/// constants were captured on: x86_64 glibc Linux with AVX2+FMA.
///
/// Every code path behind those constants needs both features: glibc picks
/// its FMA `exp`/`ln` variants only when FMA and AVX2 are usable,
/// matrixmultiply's `KernelFmaAvx2` needs AVX2 (it falls back to `KernelFma`
/// without it), nano-gemm's f64 kernels need AVX2 and pulp's x86-v3 path
/// needs both. Elsewhere the golden tests compare to a tolerance instead.
///
/// AVX-512 hosts pass this gate because they run the same x86-v3 kernels:
/// pulp's `nightly-x86-v4` and gemm's and nano-gemm's `nightly` features are
/// not enabled. Enabling any of them requires recapturing the constants or
/// adding `avx512f` to this gate.
#[cfg(test)]
pub(crate) fn golden_bits_platform() -> bool {
    #[cfg(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu"))]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu")))]
    {
        false
    }
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
pub use bart::{BartConfig, BartResult, bart, bart_arrays, bart_predict, run_bart};
pub use bart_causal::{
    BartCausalConfig, BartCausalResult, bart_causal, bart_causal_arrays, bart_causal_predict,
    bart_causal_predict_arrays, run_bart_causal, run_bart_causal_config,
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
    run_causal_forest_config,
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
pub use evaluation::{
    ClassificationMetrics, ConfusionMatrixResult, PartialDependenceResult, RocAucResult,
    VariableImportanceResult, cart_partial_dependence, cart_variable_importance, confusion_matrix,
    gbm_partial_dependence, gbm_variable_importance, rf_variable_importance, roc_auc,
};
pub use evaluation_fast::{FastRocAucResult, fast_roc_auc, fast_roc_auc_parallel};
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
pub use svm::{
    KernelSvmConfig, KernelSvmResult, SvmKernel, SvmResult, kernel_svm, kernel_svm_predict,
    linear_svm, svm_predict,
};
pub use svm_fast::{FastKernel, FastSvmConfig, FastSvmResult, fast_svm};
pub use trees::{
    DecisionTree, RandomForestModel, RandomForestResult, TreeNode, random_forest,
    random_forest_with_trees,
};
// ICE curves (Individual Conditional Expectation) - enhanced version with heterogeneity analysis
pub use ice::{
    IceConfig, IceResult, IceSpread, compute_ice_curves, ice_curves_cart, ice_curves_gbm,
    ice_curves_rf,
};
// Association rules
pub use apriori::{
    AprioriConfig, AprioriResult, AssociationRule, FrequentItemset, apriori, eclat,
    matrix_to_transactions,
};
// MARS
pub use mars::{MarsConfig, MarsResult, mars, run_mars};
// XGBoost
pub use xgboost::{
    XGBoostConfig, XGBoostNode, XGBoostObjective, XGBoostResult, XGBoostTree, run_xgboost,
    run_xgboost_default, xgboost, xgboost_predict, xgboost_predict_class,
};
// LightGBM
pub use lightgbm::{
    ImportanceType, LightGbmConfig, LightGbmObjective, LightGbmResult, lightgbm,
    lightgbm_feature_importance, lightgbm_predict, run_lightgbm, run_lightgbm_default,
};
// Fast XGBoost (histogram-based with parallel processing)
pub use xgboost_fast::{
    FastXgbConfig, FastXgbObjective, FastXgbResult, fast_xgboost, fast_xgboost_predict,
};
// Fast MBoost (optimized componentwise linear with parallel processing)
pub use mboost_fast::{
    FastMboostConfig, FastMboostFamily, FastMboostLearner, FastMboostResult, fast_mboost,
    fast_mboost_predict,
};
// PDP (Partial Dependence Plot) - original version.
// The config and result types are part of the callable surface: `partial_dependence`
// takes `&PdpConfig`, so without it no caller outside the crate can set
// `grid_resolution`, `use_quantiles`, `quantiles`, or `center`.
pub use pdp::{
    Pdp2dResult, PdpConfig, PdpResult, partial_dependence, partial_dependence_2d,
    partial_dependence_2d_named, partial_dependence_named,
};

#[cfg(test)]
mod tests {
    use super::partition_exact;

    /// `partition_exact` splits as `Iterator::partition` does, with every vector at exactly its
    /// length, also where one side keeps almost every index.
    #[test]
    fn partition_exact_matches_partition_with_capacity_equal_to_length() {
        let indices: Vec<usize> = (0..1_000).map(|i| (i * 37) % 1_009).collect();
        for cut in [0, 1, 4, 500, 996, 999, 1_009] {
            let pred = |i: usize| i < cut;
            let (yes, no) = partition_exact(indices.iter().copied(), pred);
            let (want_yes, want_no): (Vec<usize>, Vec<usize>) =
                indices.iter().partition(|&&i| pred(i));
            assert_eq!((&yes, &no), (&want_yes, &want_no), "cut {cut}");
            assert_eq!(yes.capacity(), yes.len(), "cut {cut}");
            assert_eq!(no.capacity(), no.len(), "cut {cut}");
        }
        let (even, odd) = partition_exact(0..777, |i| i % 2 == 0);
        assert_eq!((even.len(), odd.len()), (389, 388));
        assert_eq!((even.capacity(), odd.capacity()), (389, 388));
    }
}
