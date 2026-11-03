//! Advanced clustering algorithms, one submodule per algorithm family.
//!
//! The functions and result types are re-exported here and from [`crate::ml`]; for example
//! `p2a_core::ml::hdbscan` and `p2a_core::ml::advanced_clustering_mod::hdbscan::hdbscan` are the same function.
//!
//! | Module | Methods |
//! |--------|---------|
//! | [`kmedoids`](mod@kmedoids) | PAM ([`kmedoids()`]), CLARA ([`clara`]) |
//! | [`spectral`] | Spectral clustering ([`spectral_clustering`]) |
//! | [`affinity`] | Affinity propagation ([`affinity_propagation`]) |
//! | [`optics`](mod@optics) | OPTICS ([`optics()`]) |
//! | [`hdbscan`](mod@hdbscan) | HDBSCAN ([`hdbscan()`]) |
//! | [`gmm`] | Gaussian mixture models ([`gaussian_mixture`]) |
//! | [`fuzzy`] | Fuzzy c-means ([`fuzzy_cmeans`]), FANNY ([`fanny`]) |
//! | [`kmeans_variants`] | Mini-batch ([`mini_batch_kmeans`]), trimmed ([`trimmed_kmeans`]) and spherical ([`skmeans`]) k-means |
//! | [`hierarchical`] | DIANA ([`diana`]), AGNES ([`agnes`]), pvclust ([`pvclust`]), fastcluster ([`fastcluster`]) |
//! | [`mixtures`] | Normal mixtures by EM ([`normal_mix_em`], [`mvnorm_mix_em`]), mixtures of regressions ([`flexmix`]) |
//! | [`dynamic_tree_cut`](mod@dynamic_tree_cut) | Dynamic tree cut ([`dynamic_tree_cut()`]) |
//! | [`other`] | K-prototypes for mixed data ([`kprototypes`]) |
//! | [`common`] | Algorithm selection ([`ClusteringAlgorithm`]), distance and silhouette helpers, cluster statistics ([`cluster_stats`]) |
//!
//! # Large-N Performance
//!
//! HDBSCAN and OPTICS automatically select the optimal algorithm based on dataset size:
//! - n < 500: Brute-force (no overhead)
//! - n < 10,000 && d <= 15: KD-Tree + Prim's MST (good balance)
//! - n >= 10,000 && d <= 15: Dual-Tree Boruvka (best scaling)
//! - d > 15: Parallel brute-force (KD-tree degrades in high dimensions)

pub mod affinity;
pub mod common;
pub mod dynamic_tree_cut;
pub mod fuzzy;
pub mod gmm;
pub mod hdbscan;
pub mod hierarchical;
pub mod kmeans_variants;
pub mod kmedoids;
pub mod mixtures;
pub mod optics;
pub mod other;
pub mod spectral;

pub use affinity::{AffinityPropagationResult, affinity_propagation, run_affinity_propagation};
pub use common::{ClusterStatsResult, ClusteringAlgorithm, cluster_stats, run_cluster_stats};
pub use dynamic_tree_cut::{
    DynamicCutMethod, DynamicTreeCutResult, dynamic_tree_cut, run_dynamic_tree_cut,
};
pub use fuzzy::{
    FannyResult, FcmDistance, FuzzyCMeansResult, fanny, fuzzy_cmeans, run_fanny, run_fuzzy_cmeans,
};
pub use gmm::{CovarianceType, GaussianMixtureResult, gaussian_mixture, run_gaussian_mixture};
pub use hdbscan::{HdbscanResult, hdbscan, run_hdbscan};
pub use hierarchical::{
    AgnesLinkage, AgnesResult, DianaResult, FastClusterResult, FastLinkage, PvClustResult, agnes,
    diana, fastcluster, pvclust, run_agnes, run_diana, run_fastcluster, run_pvclust,
};
pub use kmeans_variants::{
    MiniBatchKMeansResult, SKMeansResult, TrimmedKMeansResult, mini_batch_kmeans,
    run_mini_batch_kmeans, run_skmeans, run_trimmed_kmeans, skmeans, trimmed_kmeans,
};
pub use kmedoids::{ClaraResult, KMedoidsResult, clara, kmedoids, run_clara, run_kmedoids};
pub use mixtures::{
    FlexMixResult, MultivariateNormalMixEMResult, NormalMixEMResult, flexmix, mvnorm_mix_em,
    normal_mix_em, run_flexmix, run_mvnorm_mix_em, run_normal_mix_em,
};
pub use optics::{OpticsResult, optics, run_optics};
pub use other::{KPrototypesResult, kprototypes, run_kprototypes};
pub use spectral::{
    AffinityType, SpectralClusteringResult, run_spectral_clustering, spectral_clustering,
};
