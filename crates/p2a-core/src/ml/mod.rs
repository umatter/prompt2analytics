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

mod clustering;
mod cluster_validation;
mod advanced_clustering_mod;
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
pub use advanced_clustering_mod::{
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
