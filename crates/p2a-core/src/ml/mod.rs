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
//! | **Random Forest** | [`random_forest`] | CART-based ensemble |
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
//! ```rust,no_run
//! use p2a_core::ml::{kmeans, silhouette, pca};
//! use ndarray::Array2;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let data = Array2::from_shape_vec((100, 3), (0..300).map(|x| x as f64).collect())?;
//!
//! // Cluster with k-means
//! let clusters = kmeans(&data.view(), 3, Some(100), Some(42))?;
//! println!("Cluster sizes: {:?}", clusters.cluster_sizes);
//!
//! // Validate clusters
//! let sil = silhouette(&data.view(), &clusters.labels)?;
//! println!("Silhouette score: {:.3}", sil.average_silhouette);
//!
//! // Reduce dimensions with PCA
//! let pca_result = pca(&data.view(), Some(2))?;
//! println!("Variance explained: {:.1}%", pca_result.variance_explained.iter().sum::<f64>() * 100.0);
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
//! | `BART` | [`bart_causal`] |

mod advanced_clustering_mod;
mod bart_causal;
mod causal_forest;
mod cluster_optimized;
mod cluster_validation;
mod clustering;
pub mod dual_tree;
pub mod kdtree;
pub mod ppr;
mod reduction;
mod svm;
mod trees;

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
pub use bart_causal::{
    BartCausalConfig, BartCausalResult, bart_causal, bart_causal_arrays, bart_causal_predict,
    bart_causal_predict_arrays, run_bart_causal,
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
pub use dual_tree::{DualTreeMstResult, dual_tree_boruvka_mst, kdtree_prim_mst};
pub use kdtree::{KdNode, KdTree, UnionFind, build_connected_mst, euclidean_distance, kruskal_mst};
pub use ppr::{PprConfig, PprResult, SmoothingMethod, ppr, run_ppr};
pub use reduction::{
    CmdscaleResult, PCAResult, TsneResult, cmdscale, cmdscale_from_data, pca,
    pca_inverse_transform, pca_transform, run_cmdscale, tsne,
};
pub use svm::{SvmResult, linear_svm, svm_predict};
pub use trees::{RandomForestResult, random_forest};
