//! Spatial econometrics infrastructure module.
//!
//! Provides spatial weights matrices, neighbor definitions, and spatial diagnostics
//! that form the foundation for spatial regression models.
//!
//! # Overview
//!
//! Spatial econometrics requires defining spatial relationships between observations.
//! This module provides:
//!
//! - **Neighbors**: Define which observations are "neighbors" of each other
//! - **SpatialWeights**: Weight matrices for spatial relationships
//! - **Diagnostics**: Tests for spatial autocorrelation (Moran's I, Geary's C)
//!
//! # Example
//!
//! ```rust,ignore
//! use p2a_core::spatial::{Neighbors, SpatialWeights, WeightStyle, moran_test};
//!
//! // Create k-nearest neighbors from coordinates
//! let coords = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
//! let nb = Neighbors::from_knn(&coords, 2);
//!
//! // Create row-standardized weights matrix
//! let listw = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);
//!
//! // Test for spatial autocorrelation
//! let y = array![1.0, 2.0, 1.5, 2.5];
//! let result = moran_test(&y, &listw, MoranAlternative::TwoSided)?;
//! ```

mod diagnostics;
mod neighbors;
mod weights;

pub use diagnostics::{
    GearyResult,
    // Local Moran's I (LISA)
    LisaCluster,
    LmTestResult,
    LocalMoranObs,
    LocalMoranResult,
    MoranAlternative,
    MoranResult,
    SpatialLmTests,
    geary_test,
    localmoran,
    moran_test,
    moran_test_residuals,
    spatial_lm_tests,
};
pub use neighbors::{NeighborMethod, Neighbors};
pub use weights::{SparseWeights, SpatialWeights, WeightStyle};
