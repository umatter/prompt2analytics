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
    AffinityPropagationResult,

    AffinityType,

    AgnesLinkage,

    AgnesResult,
    ClaraResult,

    ClusterStatsResult,

    // Algorithm selection
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

    // Affinity Propagation
    affinity_propagation,
    // AGNES (Agglomerative Nesting)
    agnes,
    // CLARA (Clustering Large Applications)
    clara,
    // Cluster Statistics
    cluster_stats,
    // DIANA (Divisive Analysis)
    diana,
    // Dynamic Tree Cut
    dynamic_tree_cut,
    // FANNY (Fuzzy Analysis)
    fanny,
    // Fast Hierarchical Clustering
    fastcluster,
    // FlexMix
    flexmix,
    // Fuzzy C-Means
    fuzzy_cmeans,
    // Gaussian Mixture Models
    gaussian_mixture,
    // HDBSCAN
    hdbscan,
    // K-Medoids (PAM)
    kmedoids,
    // K-Prototypes
    kprototypes,
    // Mini-Batch K-Means
    mini_batch_kmeans,
    mvnorm_mix_em,
    // Normal Mixture EM
    normal_mix_em,
    // OPTICS
    optics,
    // pvclust (Bootstrap Cluster Assessment)
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
    // Spherical K-Means
    skmeans,
    // Spectral Clustering
    spectral_clustering,
    // Trimmed K-Means
    trimmed_kmeans,
};
