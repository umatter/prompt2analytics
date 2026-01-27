//! Advanced clustering algorithms.
//!
//! Provides sophisticated clustering methods beyond standard k-means and hierarchical.
//!
//! # Algorithms
//! - K-Medoids (PAM): Partitioning using actual data points as centers
//! - Spectral Clustering: Graph-based clustering using Laplacian eigenvectors
//! - Affinity Propagation: Message-passing clustering with automatic k selection
//! - OPTICS: Ordering Points To Identify Clustering Structure
//! - HDBSCAN: Hierarchical Density-Based Spatial Clustering
//! - Gaussian Mixture Models: Probabilistic clustering with EM algorithm
//!
//! # Large-N Performance
//!
//! HDBSCAN and OPTICS automatically select the optimal algorithm based on dataset size:
//! - n < 500: Brute-force (no overhead)
//! - n < 10,000 && d <= 15: KD-Tree + Prim's MST (good balance)
//! - n >= 10,000 && d <= 15: Dual-Tree Boruvka (best scaling)
//! - d > 15: Parallel brute-force (KD-tree degrades in high dimensions)

// The advanced module contains all the clustering algorithms
// In the future, these can be split into separate files
mod advanced;

// Re-export all public items from advanced module
pub use advanced::{
    // Algorithm selection
    ClusteringAlgorithm,

    // K-Medoids (PAM)
    kmedoids, run_kmedoids, KMedoidsResult,

    // Spectral Clustering
    spectral_clustering, run_spectral_clustering, SpectralClusteringResult, AffinityType,

    // Affinity Propagation
    affinity_propagation, run_affinity_propagation, AffinityPropagationResult,

    // OPTICS
    optics, run_optics, OpticsResult,

    // HDBSCAN
    hdbscan, run_hdbscan, HdbscanResult,

    // Gaussian Mixture Models
    gaussian_mixture, run_gaussian_mixture, GaussianMixtureResult, CovarianceType,

    // Fuzzy C-Means
    fuzzy_cmeans, run_fuzzy_cmeans, FuzzyCMeansResult, FcmDistance,

    // Mini-Batch K-Means
    mini_batch_kmeans, run_mini_batch_kmeans, MiniBatchKMeansResult,

    // Trimmed K-Means
    trimmed_kmeans, run_trimmed_kmeans, TrimmedKMeansResult,

    // DIANA (Divisive Analysis)
    diana, run_diana, DianaResult,

    // AGNES (Agglomerative Nesting)
    agnes, run_agnes, AgnesResult, AgnesLinkage,

    // FlexMix
    flexmix, run_flexmix, FlexMixResult,

    // pvclust (Bootstrap Cluster Assessment)
    pvclust, run_pvclust, PvClustResult,

    // CLARA (Clustering Large Applications)
    clara, run_clara, ClaraResult,

    // Cluster Statistics
    cluster_stats, run_cluster_stats, ClusterStatsResult,

    // FANNY (Fuzzy Analysis)
    fanny, run_fanny, FannyResult,

    // Spherical K-Means
    skmeans, run_skmeans, SKMeansResult,

    // Fast Hierarchical Clustering
    fastcluster, run_fastcluster, FastClusterResult, FastLinkage,

    // Dynamic Tree Cut
    dynamic_tree_cut, run_dynamic_tree_cut, DynamicTreeCutResult, DynamicCutMethod,

    // Normal Mixture EM
    normal_mix_em, run_normal_mix_em, NormalMixEMResult,
    mvnorm_mix_em, run_mvnorm_mix_em, MultivariateNormalMixEMResult,

    // K-Prototypes
    kprototypes, run_kprototypes, KPrototypesResult,
};
