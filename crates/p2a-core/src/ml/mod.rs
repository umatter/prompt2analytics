//! Machine learning module.
//!
//! Provides clustering, dimensionality reduction, supervised learning,
//! and causal machine learning algorithms.
//!
//! # Large-N Performance
//!
//! HDBSCAN and OPTICS automatically select optimal algorithms for large datasets:
//! - n < 500: O(n²) brute-force
//! - n < 10,000 && d <= 15: O(n log n) KD-Tree + Prim's
//! - n >= 10,000 && d <= 15: O(n log n) Dual-Tree Boruvka
//! - d > 15: O(n²) parallel brute-force

mod clustering;
mod cluster_validation;
mod advanced_clustering;
mod cluster_optimized;
pub mod kdtree;
pub mod dual_tree;
mod reduction;
mod trees;
mod svm;
pub mod ppr;
mod causal_forest;
mod bart_causal;

pub use clustering::{
    kmeans, dbscan, hierarchical,
    KMeansResult, DBSCANResult, HierarchicalResult, Linkage,
    cutree, cutree_multiple_k, run_cutree, CutreeResult,
};
pub use cluster_validation::{
    silhouette, silhouette_from_dist, run_silhouette, SilhouetteResult, SilhouetteInfo,
    calinski_harabasz, run_calinski_harabasz, CalinskiHarabaszResult,
    davies_bouldin, run_davies_bouldin, DaviesBouldinResult,
    dunn_index, run_dunn_index, DunnIndexResult,
    rand_index, run_rand_index, RandIndexResult,
    nmi, run_nmi, NmiResult,
    gap_statistic, run_gap_statistic, GapStatisticResult,
};
pub use advanced_clustering::{
    kmedoids, run_kmedoids, KMedoidsResult,
    spectral_clustering, run_spectral_clustering, SpectralClusteringResult, AffinityType,
    affinity_propagation, run_affinity_propagation, AffinityPropagationResult,
    optics, run_optics, OpticsResult,
    hdbscan, run_hdbscan, HdbscanResult,
    gaussian_mixture, run_gaussian_mixture, GaussianMixtureResult, CovarianceType,
    ClusteringAlgorithm,
    fuzzy_cmeans, run_fuzzy_cmeans, FuzzyCMeansResult, FcmDistance,
    mini_batch_kmeans, run_mini_batch_kmeans, MiniBatchKMeansResult,
    trimmed_kmeans, run_trimmed_kmeans, TrimmedKMeansResult,
    diana, run_diana, DianaResult,
    agnes, run_agnes, AgnesResult, AgnesLinkage,
    // Batch 2: Additional clustering methods
    flexmix, run_flexmix, FlexMixResult,
    pvclust, run_pvclust, PvClustResult,
    clara, run_clara, ClaraResult,
    cluster_stats, run_cluster_stats, ClusterStatsResult,
    fanny, run_fanny, FannyResult,
    // Batch 3: skmeans, fastcluster, dynamicTreeCut, mixtools, kprototypes
    skmeans, run_skmeans, SKMeansResult,
    fastcluster, run_fastcluster, FastClusterResult, FastLinkage,
    dynamic_tree_cut, run_dynamic_tree_cut, DynamicTreeCutResult, DynamicCutMethod,
    normal_mix_em, run_normal_mix_em, NormalMixEMResult,
    mvnorm_mix_em, run_mvnorm_mix_em, MultivariateNormalMixEMResult,
    kprototypes, run_kprototypes, KPrototypesResult,
};
pub use kdtree::{KdTree, KdNode, UnionFind, euclidean_distance, kruskal_mst, build_connected_mst};
pub use dual_tree::{dual_tree_boruvka_mst, kdtree_prim_mst, DualTreeMstResult};
pub use reduction::{
    pca, pca_transform, pca_inverse_transform, tsne, PCAResult, TsneResult,
    cmdscale, cmdscale_from_data, run_cmdscale, CmdscaleResult,
};
pub use trees::{random_forest, RandomForestResult};
pub use svm::{linear_svm, svm_predict, SvmResult};
pub use ppr::{ppr, run_ppr, PprResult, PprConfig, SmoothingMethod};
pub use causal_forest::{
    causal_forest, causal_forest_arrays, causal_forest_predict, causal_forest_predict_arrays,
    average_treatment_effect, run_causal_forest,
    CausalForestConfig, CausalForestResult,
};
pub use bart_causal::{
    bart_causal, bart_causal_arrays, bart_causal_predict, bart_causal_predict_arrays,
    run_bart_causal, BartCausalConfig, BartCausalResult,
};
pub use cluster_optimized::{
    compute_pairwise_distances_parallel,
    compute_pairwise_sq_distances_parallel,
    silhouette_optimized,
    kmedoids_optimized,
    hdbscan_optimized,
    optics_optimized,
    affinity_propagation_optimized,
};
