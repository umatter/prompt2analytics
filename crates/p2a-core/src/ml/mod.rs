//! Machine learning module.
//!
//! Provides clustering and dimensionality reduction algorithms.

mod clustering;
mod reduction;

pub use clustering::{kmeans, dbscan, KMeansResult, DBSCANResult};
pub use reduction::{pca, pca_transform, pca_inverse_transform, PCAResult};
