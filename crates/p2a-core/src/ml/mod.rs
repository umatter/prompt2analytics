//! Machine learning module.
//!
//! Provides clustering, dimensionality reduction, and supervised learning algorithms.

mod clustering;
mod reduction;
mod trees;
mod svm;

pub use clustering::{
    kmeans, dbscan, hierarchical,
    KMeansResult, DBSCANResult, HierarchicalResult, Linkage,
};
pub use reduction::{pca, pca_transform, pca_inverse_transform, tsne, PCAResult, TsneResult};
pub use trees::{random_forest, RandomForestResult};
pub use svm::{linear_svm, svm_predict, SvmResult};
