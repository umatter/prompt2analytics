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
//! # Complexity
//!
//! k-nearest-neighbour and distance-band neighbours are built with a k-d tree in
//! O(n log n + links) time for well-spread points, with the same result as an all-pairs
//! search; coordinates shared by many observations do not make it quadratic, except, for
//! great-circle bands, when the distance between two such locations lies within the
//! rounding margin (about 1e-9 relative) of d1 or d2. The
//! `Neighbors::try_from_knn`, `try_from_distance` and `try_from_distance_longlat`
//! constructors also validate their input and stop when an installed
//! [`Budget`](crate::resource::Budget) expires, and the distance-band ones take a cap on
//! the number of links; the older `from_*` constructors keep their behaviour.
//!
//! `SpatialWeights` keeps W sparse (one index and weight list per row) and forms a dense
//! n x n matrix for the eigenvalues that maximum-likelihood estimation needs. The SAR
//! total impact forms a dense I - ρW as well when it needs the solve (I - ρW) v = 1 (the
//! rows of W do not all have one sum) and the Neumann series is ruled out (|ρ| times the
//! spectral radius of W is at least 0.99, which binary weights often reach) or does not
//! settle. If the eigen decomposition or that solve fails, or a sum is not finite, the
//! impacts fall back to the dense inverse of I - ρW.
//!
//! # Example
//!
//! ```
//! use p2a_core::spatial::{Neighbors, SpatialWeights, WeightStyle, moran_test, MoranAlternative};
//! use ndarray::array;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
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
//! # Ok(())
//! # }
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

#[cfg(test)]
pub(crate) mod test_fixtures {
    //! 30 points used by the spatial tests. R 4.3.3:
    //!   set.seed(17); n <- 30; cx <- round(runif(n, 0, 10), 3); cy <- round(runif(n, 0, 10), 3)
    //!   set.seed(5); x1 <- round(0.3 * cx + rnorm(n), 3); y <- round(1 + 2 * x1 + 0.5 * cy + rnorm(n), 3)
    //!
    //! Each `%.3f` value parses to the double R holds (`identical()` was checked in R).

    use super::{Neighbors, SpatialWeights, WeightStyle};

    pub(crate) const CX: [f64; 30] = [
        1.551, 9.684, 4.683, 7.768, 4.079, 5.388, 2.069, 1.871, 7.800, 1.939, 4.342, 0.023, 8.347,
        8.301, 9.570, 9.485, 6.007, 2.618, 6.430, 5.262, 8.812, 6.075, 7.400, 8.031, 9.024, 0.927,
        5.745, 5.755, 9.442, 2.190,
    ];
    pub(crate) const CY: [f64; 30] = [
        4.780, 0.447, 7.991, 3.683, 5.633, 2.470, 7.343, 7.037, 7.368, 8.019, 7.521, 6.338, 2.476,
        5.513, 2.348, 2.586, 9.529, 8.569, 2.757, 5.545, 8.769, 1.188, 5.467, 2.328, 6.023, 7.125,
        6.489, 1.876, 2.384, 0.704,
    ];
    pub(crate) const X1: [f64; 30] = [
        -0.376, 4.290, 0.149, 2.401, 2.935, 1.013, 0.149, -0.074, 2.054, 0.720, 2.530, -0.795,
        1.424, 2.333, 1.799, 2.707, 1.205, -1.399, 2.170, 1.319, 3.544, 2.764, 3.688, 3.116, 3.526,
        -0.015, 3.142, 3.225, 2.176, -0.196,
    ];
    pub(crate) const Y: [f64; 30] = [
        2.954, 10.913, 7.509, 8.861, 11.166, 5.213, 3.960, 2.370, 7.030, 6.307, 11.371, 1.777,
        5.011, 10.318, 5.315, 8.269, 7.287, 2.026, 5.994, 6.341, 13.936, 7.310, 12.132, 7.804,
        10.951, 3.608, 11.282, 8.275, 6.480, 1.193,
    ];

    /// The 30 coordinate pairs (`CX` zipped with `CY`).
    pub(crate) fn coords() -> Vec<(f64, f64)> {
        CX.iter().copied().zip(CY.iter().copied()).collect()
    }

    /// k = 3 nearest-neighbour weights.
    pub(crate) fn knn3(style: WeightStyle) -> SpatialWeights {
        SpatialWeights::from_neighbors(&Neighbors::from_knn(&coords(), 3), style)
    }

    /// Distance band (0, 2]; observations 0, 16 and 29 are isolates.
    pub(crate) fn band2(style: WeightStyle) -> SpatialWeights {
        SpatialWeights::from_neighbors(&Neighbors::from_distance(&coords(), 0.0, 2.0), style)
    }

    /// Columns `y` (`Y`), `x1` (`X1`) and `y_bin` (1.0 where `Y > 7`, 16 ones).
    pub(crate) fn dataset() -> crate::data::Dataset {
        use polars::prelude::*;
        let y_bin: Vec<f64> = Y.iter().map(|&v| if v > 7.0 { 1.0 } else { 0.0 }).collect();
        let df = df! {
            "y" => Y.to_vec(),
            "x1" => X1.to_vec(),
            "y_bin" => y_bin,
        }
        .expect("fixture columns have equal length");
        crate::data::Dataset::new(df)
    }
}
