//! Spatial weights matrices.
//!
//! Provides spatial weights structures that combine neighbor definitions with
//! weighting schemes, equivalent to R's `listw` class from spdep.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use super::neighbors::Neighbors;

/// Weighting style for spatial weights matrix.
///
/// Controls how the raw neighbor relationships are converted to weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum WeightStyle {
    /// Binary weights (0/1) - "B" in R
    Binary,
    /// Row-standardized (weights sum to 1 for each row) - "W" in R
    #[default]
    RowStd,
    /// Globally standardized (weights sum to n) - "C" in R
    GlobalStd,
    /// Variance-stabilizing (Tiefelsdorf et al. 1999) - "S" in R
    VarStab,
    /// Minmax normalization (divide by largest row sum) - "minmax" in R
    MinMax,
}

/// Sparse representation of spatial weights for a single observation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseWeights {
    /// Indices of neighbors
    pub indices: Vec<usize>,
    /// Weights corresponding to each neighbor
    pub weights: Vec<f64>,
}

/// Spatial weights list - equivalent to R's `listw` class.
///
/// Combines a neighbors structure with a weighting scheme to create a
/// spatial weights matrix that can be used in spatial regression models.
///
/// # Example
///
/// ```
/// use p2a_core::spatial::{Neighbors, SpatialWeights, WeightStyle};
/// use ndarray::array;
///
/// // Create neighbors
/// let coords = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
/// let nb = Neighbors::from_knn(&coords, 2);
///
/// // Create row-standardized weights
/// let listw = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);
///
/// // Compute spatial lag: Wy
/// let y = array![1.0, 2.0, 1.5, 2.5];
/// let wy = listw.lag(&y);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialWeights {
    /// Underlying neighbors structure
    neighbors: Neighbors,
    /// Sparse weights for each observation
    weights: Vec<SparseWeights>,
    /// Style of weights
    style: WeightStyle,
    /// Number of observations
    n: usize,
    /// Cached eigenvalues of W (computed on demand)
    #[serde(skip)]
    eigenvalues: Option<Array1<f64>>,
}

impl SpatialWeights {
    /// Create spatial weights from a neighbors object with specified style.
    ///
    /// # Arguments
    ///
    /// * `nb` - Neighbors structure
    /// * `style` - Weighting style (row-standardized, binary, etc.)
    ///
    /// # Example
    ///
    /// ```
    /// use p2a_core::spatial::{Neighbors, SpatialWeights, WeightStyle};
    ///
    /// let coords = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
    /// let nb = Neighbors::from_knn(&coords, 3);
    /// let listw = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);
    /// ```
    pub fn from_neighbors(nb: &Neighbors, style: WeightStyle) -> Self {
        let n = nb.n();
        let mut weights = Vec::with_capacity(n);

        // First pass: calculate raw weights based on style
        let raw_weights: Vec<SparseWeights> = (0..n)
            .map(|i| {
                let indices = nb.get_neighbors(i).to_vec();
                let n_neighbors = indices.len();
                let w = if n_neighbors == 0 {
                    vec![]
                } else {
                    match style {
                        WeightStyle::Binary => vec![1.0; n_neighbors],
                        WeightStyle::RowStd => vec![1.0 / n_neighbors as f64; n_neighbors],
                        // For other styles, start with binary and transform later
                        _ => vec![1.0; n_neighbors],
                    }
                };
                SparseWeights {
                    indices,
                    weights: w,
                }
            })
            .collect();

        // Apply global transformations for certain styles
        match style {
            WeightStyle::GlobalStd => {
                // Sum of all weights should equal n
                let total: f64 = raw_weights
                    .iter()
                    .map(|sw| sw.weights.iter().sum::<f64>())
                    .sum();
                let scale = if total > 0.0 { n as f64 / total } else { 1.0 };
                weights = raw_weights
                    .into_iter()
                    .map(|sw| SparseWeights {
                        indices: sw.indices,
                        weights: sw.weights.iter().map(|&w| w * scale).collect(),
                    })
                    .collect();
            }
            WeightStyle::VarStab => {
                // Variance-stabilizing: w_ij = 1/sqrt(n_i * n_j)
                let cardinalities = nb.cardinalities();
                weights = raw_weights
                    .into_iter()
                    .enumerate()
                    .map(|(i, sw)| {
                        let ni = cardinalities[i] as f64;
                        let w: Vec<f64> = sw
                            .indices
                            .iter()
                            .map(|&j| {
                                let nj = cardinalities[j] as f64;
                                if ni > 0.0 && nj > 0.0 {
                                    1.0 / (ni * nj).sqrt()
                                } else {
                                    0.0
                                }
                            })
                            .collect();
                        SparseWeights {
                            indices: sw.indices,
                            weights: w,
                        }
                    })
                    .collect();
            }
            WeightStyle::MinMax => {
                // Divide all weights by the maximum row sum
                let row_sums: Vec<f64> = raw_weights
                    .iter()
                    .map(|sw| sw.weights.iter().sum())
                    .collect();
                let max_sum = row_sums.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let scale = if max_sum > 0.0 { 1.0 / max_sum } else { 1.0 };
                weights = raw_weights
                    .into_iter()
                    .map(|sw| SparseWeights {
                        indices: sw.indices,
                        weights: sw.weights.iter().map(|&w| w * scale).collect(),
                    })
                    .collect();
            }
            _ => {
                weights = raw_weights;
            }
        }

        SpatialWeights {
            neighbors: nb.clone(),
            weights,
            style,
            n,
            eigenvalues: None,
        }
    }

    /// Create spatial weights with distance-decay weighting.
    ///
    /// Weights are computed as w_ij = d_ij^(-alpha) for neighbors,
    /// then optionally row-standardized.
    ///
    /// # Arguments
    ///
    /// * `nb` - Neighbors structure
    /// * `coords` - Coordinates for computing distances
    /// * `alpha` - Distance decay parameter (typically 1 or 2)
    /// * `row_standardize` - Whether to row-standardize the weights
    pub fn from_distance_decay(
        nb: &Neighbors,
        coords: &[(f64, f64)],
        alpha: f64,
        row_standardize: bool,
    ) -> Self {
        let n = nb.n();
        let mut weights_list = Vec::with_capacity(n);

        for i in 0..n {
            let indices = nb.get_neighbors(i).to_vec();
            let mut w: Vec<f64> = indices
                .iter()
                .map(|&j| {
                    let dx = coords[i].0 - coords[j].0;
                    let dy = coords[i].1 - coords[j].1;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist > 0.0 { dist.powf(-alpha) } else { 0.0 }
                })
                .collect();

            if row_standardize {
                let sum: f64 = w.iter().sum();
                if sum > 0.0 {
                    for weight in &mut w {
                        *weight /= sum;
                    }
                }
            }

            weights_list.push(SparseWeights {
                indices,
                weights: w,
            });
        }

        let style = if row_standardize {
            WeightStyle::RowStd
        } else {
            WeightStyle::Binary // Not quite accurate, but closest
        };

        SpatialWeights {
            neighbors: nb.clone(),
            weights: weights_list,
            style,
            n,
            eigenvalues: None,
        }
    }

    /// Compute the spatial lag: Wy.
    ///
    /// For each observation i, computes the weighted average of y values
    /// of its neighbors.
    ///
    /// # Arguments
    ///
    /// * `y` - Values to lag
    ///
    /// # Returns
    ///
    /// Spatially lagged values (same length as input)
    pub fn lag(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut wy = Array1::zeros(self.n);

        for i in 0..self.n {
            let sw = &self.weights[i];
            let mut sum = 0.0;
            for (idx, &j) in sw.indices.iter().enumerate() {
                sum += sw.weights[idx] * y[j];
            }
            wy[i] = sum;
        }

        wy
    }

    /// Compute the transpose spatial lag: W'y.
    ///
    /// This is useful for some spatial estimators.
    pub fn lag_transpose(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut wty = Array1::zeros(self.n);

        for i in 0..self.n {
            let sw = &self.weights[i];
            for (idx, &j) in sw.indices.iter().enumerate() {
                wty[j] += sw.weights[idx] * y[i];
            }
        }

        wty
    }

    /// Convert to dense matrix representation.
    ///
    /// Creates an n x n matrix where W[i,j] is the weight from i to j.
    /// Use with caution for large n as this uses O(n²) memory.
    pub fn to_dense(&self) -> Array2<f64> {
        let mut w = Array2::zeros((self.n, self.n));

        for i in 0..self.n {
            let sw = &self.weights[i];
            for (idx, &j) in sw.indices.iter().enumerate() {
                w[[i, j]] = sw.weights[idx];
            }
        }

        w
    }

    /// Compute eigenvalues of W.
    ///
    /// Eigenvalues are needed for ML estimation to compute log|I - ρW|
    /// and to determine the valid range for ρ.
    ///
    /// This uses dense matrix operations so is only suitable for moderate n.
    pub fn eigenvalues(&mut self) -> &Array1<f64> {
        if self.eigenvalues.is_none() {
            let w = self.to_dense();
            // Use faer for eigenvalue computation
            let eigs = compute_eigenvalues(&w);
            self.eigenvalues = Some(eigs);
        }
        self.eigenvalues.as_ref().unwrap()
    }

    /// Get cached eigenvalues if available.
    pub fn get_eigenvalues(&self) -> Option<&Array1<f64>> {
        self.eigenvalues.as_ref()
    }

    /// Compute log|I - ρW| using eigenvalues.
    ///
    /// This is the log-determinant term needed for spatial ML estimation.
    /// Uses the formula: log|I - ρW| = Σ log(1 - ρ*λ_i)
    ///
    /// # Arguments
    ///
    /// * `rho` - Spatial autoregressive parameter
    ///
    /// # Returns
    ///
    /// The log-determinant value
    pub fn log_det(&mut self, rho: f64) -> f64 {
        let eigs = self.eigenvalues();
        eigs.iter().map(|&lambda| (1.0 - rho * lambda).ln()).sum()
    }

    /// Get the valid range for ρ based on eigenvalues.
    ///
    /// For the model to be valid, we need |I - ρW| > 0, which requires
    /// ρ to be in (1/λ_min, 1/λ_max) where λ_min and λ_max are the
    /// extreme eigenvalues of W.
    pub fn rho_range(&mut self) -> (f64, f64) {
        let eigs = self.eigenvalues();
        let lambda_min = eigs.iter().copied().fold(f64::INFINITY, f64::min);
        let lambda_max = eigs.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let lower = if lambda_min < 0.0 {
            1.0 / lambda_min
        } else {
            f64::NEG_INFINITY
        };
        let upper = if lambda_max > 0.0 {
            1.0 / lambda_max
        } else {
            f64::INFINITY
        };

        (lower, upper)
    }

    /// Compute (I - ρW)y.
    ///
    /// This transformation is used in spatial error model estimation.
    pub fn transform_y(&self, y: &Array1<f64>, rho: f64) -> Array1<f64> {
        let wy = self.lag(y);
        y - rho * &wy
    }

    /// Compute (I - ρW)X for a matrix X.
    ///
    /// This transformation is used in spatial error model estimation.
    pub fn transform_x(&self, x: &Array2<f64>, rho: f64) -> Array2<f64> {
        let n = x.nrows();
        let k = x.ncols();
        let mut result = Array2::zeros((n, k));

        for col in 0..k {
            let x_col = x.column(col).to_owned();
            let wx_col = self.lag(&x_col);
            for i in 0..n {
                result[[i, col]] = x_col[i] - rho * wx_col[i];
            }
        }

        result
    }

    /// Get the number of observations.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the weight style.
    pub fn style(&self) -> WeightStyle {
        self.style
    }

    /// Get the underlying neighbors structure.
    pub fn neighbors(&self) -> &Neighbors {
        &self.neighbors
    }

    /// Get sparse weights for observation i.
    pub fn get_weights(&self, i: usize) -> &SparseWeights {
        &self.weights[i]
    }

    /// Check if the weights matrix is symmetric.
    pub fn is_symmetric(&self) -> bool {
        // Check if W[i,j] == W[j,i] for all neighbor pairs
        for i in 0..self.n {
            let sw_i = &self.weights[i];
            for (idx, &j) in sw_i.indices.iter().enumerate() {
                let w_ij = sw_i.weights[idx];

                // Find w_ji
                let sw_j = &self.weights[j];
                let pos = sw_j.indices.iter().position(|&x| x == i);
                match pos {
                    Some(jdx) => {
                        if (sw_j.weights[jdx] - w_ij).abs() > 1e-10 {
                            return false;
                        }
                    }
                    None => return false,
                }
            }
        }
        true
    }

    /// Get the sum of all weights.
    pub fn sum_weights(&self) -> f64 {
        self.weights
            .iter()
            .map(|sw| sw.weights.iter().sum::<f64>())
            .sum()
    }

    /// Compute trace(W²) efficiently using sparse representation.
    ///
    /// trace(W²) = Σᵢ (W²)ᵢᵢ = Σᵢ Σⱼ Wᵢⱼ * Wⱼᵢ
    ///
    /// This is O(m) where m is the number of non-zero entries, not O(n³).
    pub fn trace_w2(&self) -> f64 {
        let mut trace = 0.0;
        for i in 0..self.n {
            let sw_i = &self.weights[i];
            for (idx, &j) in sw_i.indices.iter().enumerate() {
                let w_ij = sw_i.weights[idx];
                // Find w_ji in sparse representation
                let sw_j = &self.weights[j];
                if let Some(jdx) = sw_j.indices.iter().position(|&x| x == i) {
                    let w_ji = sw_j.weights[jdx];
                    trace += w_ij * w_ji;
                }
            }
        }
        trace
    }

    /// Compute trace(W'W) efficiently using sparse representation.
    ///
    /// trace(W'W) = Σᵢⱼ Wᵢⱼ² (sum of squared weights)
    ///
    /// This is O(m) where m is the number of non-zero entries.
    pub fn trace_wtw(&self) -> f64 {
        self.weights
            .iter()
            .map(|sw| sw.weights.iter().map(|&w| w * w).sum::<f64>())
            .sum()
    }

    /// Get summary statistics about the weights.
    pub fn summary(&self) -> WeightsSummary {
        let cardinalities = self.neighbors.cardinalities();
        let min_links = *cardinalities.iter().min().unwrap_or(&0);
        let max_links = *cardinalities.iter().max().unwrap_or(&0);
        let avg_links = self.neighbors.avg_neighbors();

        // Weight statistics
        let all_weights: Vec<f64> = self
            .weights
            .iter()
            .flat_map(|sw| sw.weights.iter().copied())
            .collect();
        let min_weight = all_weights.iter().copied().fold(f64::INFINITY, f64::min);
        let max_weight = all_weights
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        WeightsSummary {
            n: self.n,
            n_links: self.neighbors.n_links(),
            min_links,
            max_links,
            avg_links,
            min_weight,
            max_weight,
            symmetric: self.is_symmetric(),
            style: self.style,
        }
    }
}

/// Summary statistics for spatial weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightsSummary {
    pub n: usize,
    pub n_links: usize,
    pub min_links: usize,
    pub max_links: usize,
    pub avg_links: f64,
    pub min_weight: f64,
    pub max_weight: f64,
    pub symmetric: bool,
    pub style: WeightStyle,
}

/// Compute eigenvalues of a matrix using faer.
fn compute_eigenvalues(m: &Array2<f64>) -> Array1<f64> {
    use crate::linalg::matrix_ops::ndarray_to_faer;

    let n = m.nrows();
    let mat = ndarray_to_faer(&m.view());

    // For general (non-symmetric) matrices, use complex eigenvalue decomposition
    // and take real parts (spatial weights matrices typically have real eigenvalues)
    let eigenvalues = mat.eigenvalues();

    // Extract real parts from the Result
    let mut eigs_vec: Vec<f64> = match eigenvalues {
        Ok(evs) => evs.iter().map(|ev| ev.re).collect(),
        Err(_) => {
            // Fallback: return zeros if eigenvalue computation fails
            vec![0.0; n]
        }
    };

    // Sort eigenvalues
    eigs_vec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    Array1::from_vec(eigs_vec)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_neighbors() -> Neighbors {
        // 4-point grid: 0-1
        //               | |
        //               2-3
        Neighbors::from_indices(vec![
            vec![1, 2], // 0 neighbors 1, 2
            vec![0, 3], // 1 neighbors 0, 3
            vec![0, 3], // 2 neighbors 0, 3
            vec![1, 2], // 3 neighbors 1, 2
        ])
    }

    #[test]
    fn test_row_standardized_weights() {
        let nb = simple_neighbors();
        let listw = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);

        // Each observation has 2 neighbors, so weights should be 0.5 each
        let sw = listw.get_weights(0);
        assert_eq!(sw.weights.len(), 2);
        assert!((sw.weights[0] - 0.5).abs() < 1e-10);
        assert!((sw.weights[1] - 0.5).abs() < 1e-10);

        // Row sum should be 1
        let row_sum: f64 = sw.weights.iter().sum();
        assert!((row_sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_binary_weights() {
        let nb = simple_neighbors();
        let listw = SpatialWeights::from_neighbors(&nb, WeightStyle::Binary);

        let sw = listw.get_weights(0);
        assert_eq!(sw.weights.len(), 2);
        assert!((sw.weights[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_spatial_lag() {
        let nb = simple_neighbors();
        let listw = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);

        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let wy = listw.lag(&y);

        // For observation 0: neighbors are 1, 2 with weights 0.5 each
        // Wy[0] = 0.5 * 2.0 + 0.5 * 3.0 = 2.5
        assert!((wy[0] - 2.5).abs() < 1e-10);

        // For observation 3: neighbors are 1, 2 with weights 0.5 each
        // Wy[3] = 0.5 * 2.0 + 0.5 * 3.0 = 2.5
        assert!((wy[3] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_to_dense() {
        let nb = simple_neighbors();
        let listw = SpatialWeights::from_neighbors(&nb, WeightStyle::Binary);
        let w = listw.to_dense();

        assert_eq!(w.shape(), &[4, 4]);
        assert!((w[[0, 1]] - 1.0).abs() < 1e-10);
        assert!((w[[0, 2]] - 1.0).abs() < 1e-10);
        assert!((w[[0, 0]] - 0.0).abs() < 1e-10); // No self-neighbors
        assert!((w[[0, 3]] - 0.0).abs() < 1e-10); // 0 and 3 not neighbors
    }

    #[test]
    fn test_symmetry() {
        let nb = simple_neighbors();
        let listw = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);

        // This neighbor structure is symmetric
        assert!(listw.is_symmetric());
    }

    #[test]
    fn test_transform_y() {
        let nb = simple_neighbors();
        let listw = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);

        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let rho = 0.5;
        let transformed = listw.transform_y(&y, rho);

        // (I - ρW)y = y - ρ*Wy
        let wy = listw.lag(&y);
        let expected = &y - rho * &wy;

        for i in 0..4 {
            assert!((transformed[i] - expected[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_eigenvalues() {
        let nb = simple_neighbors();
        let mut listw = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);

        let eigs = listw.eigenvalues();
        assert_eq!(eigs.len(), 4);

        // For row-standardized weights, maximum eigenvalue is 1
        let max_eig = eigs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!((max_eig - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_rho_range() {
        let nb = simple_neighbors();
        let mut listw = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);

        let (lower, upper) = listw.rho_range();

        // For row-standardized weights, range is typically around (-1, 1)
        assert!(lower < 0.0);
        assert!(upper > 0.0);
        assert!((upper - 1.0).abs() < 0.01);
    }
}
