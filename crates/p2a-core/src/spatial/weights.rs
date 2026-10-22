//! Spatial weights matrices.
//!
//! Provides spatial weights structures that combine neighbor definitions with
//! weighting schemes, equivalent to R's `listw` class from spdep.

use std::sync::Arc;

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use super::neighbors::Neighbors;
use crate::errors::EconResult;
use crate::resource::{checkpoint, try_zeros2};

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
///
/// For every weights object this crate builds, `indices` is strictly increasing (the
/// neighbour lists are sorted and free of duplicates); lookups rely on that.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseWeights {
    /// Indices of neighbors (strictly increasing)
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
///
/// The neighbour lists, the weight lists and the eigenvalue cache are shared, so `clone()`
/// is O(1); [`SpatialWeights::same_weights`] tells whether two objects are clones.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialWeights {
    /// Underlying neighbors structure
    neighbors: Neighbors,
    /// Sparse weights for each observation
    #[serde(with = "super::neighbors::arc_serde")]
    weights: Arc<Vec<SparseWeights>>,
    /// Style of weights
    style: WeightStyle,
    /// Number of observations
    n: usize,
    /// Cached eigen decomposition of W (computed on demand)
    #[serde(skip)]
    spectrum: Option<Arc<Spectrum>>,
}

/// The eigenvalues of W, as the dense eigen decomposition returned them.
#[derive(Debug)]
struct Spectrum {
    /// Real parts, sorted ascending (what [`SpatialWeights::eigenvalues`] returns).
    real_sorted: Array1<f64>,
    /// All eigenvalues as (real, imaginary) pairs; empty when the decomposition failed.
    values: Vec<(f64, f64)>,
    /// False when the decomposition failed and `real_sorted` is the legacy zero fallback.
    ok: bool,
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
            // O(1): the neighbour lists are shared, not copied
            neighbors: nb.clone(),
            weights: Arc::new(weights),
            style,
            n,
            spectrum: None,
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
            // O(1): the neighbour lists are shared, not copied
            neighbors: nb.clone(),
            weights: Arc::new(weights_list),
            style,
            n,
            spectrum: None,
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
        let spectrum = match self.spectrum.take() {
            Some(s) => s,
            // Use faer for eigenvalue computation
            None => Arc::new(compute_spectrum(&self.to_dense())),
        };
        &self.spectrum.insert(spectrum).real_sorted
    }

    /// Dense n x n copy of W, allocated fallibly.
    ///
    /// # Errors
    ///
    /// [`EconError::ResourceLimit`](crate::errors::EconError::ResourceLimit) when the n x n
    /// matrix cannot be allocated (or exceeds an
    /// installed per-allocation ceiling).
    pub fn try_to_dense(&self) -> EconResult<Array2<f64>> {
        let mut w = try_zeros2(self.n, self.n, "dense spatial weights matrix")?;
        for (i, sw) in self.weights.iter().enumerate() {
            for (&j, &v) in sw.indices.iter().zip(sw.weights.iter()) {
                w[[i, j]] = v;
            }
        }
        Ok(w)
    }

    /// Eigenvalues of W like [`SpatialWeights::eigenvalues`], but the dense matrix behind
    /// them is allocated fallibly and an installed budget is checked first.
    ///
    /// # Errors
    ///
    /// [`EconError::ResourceLimit`](crate::errors::EconError::ResourceLimit) when the dense
    /// matrix cannot be allocated; [`EconError::Cancelled`](crate::errors::EconError::Cancelled)
    /// when an installed budget has expired.
    pub fn try_eigenvalues(&mut self) -> EconResult<&Array1<f64>> {
        Ok(&self.try_spectrum()?.real_sorted)
    }

    /// The valid range for ρ like [`SpatialWeights::rho_range`], with the errors of
    /// [`SpatialWeights::try_eigenvalues`].
    pub fn try_rho_range(&mut self) -> EconResult<(f64, f64)> {
        Ok(rho_range_from(self.try_eigenvalues()?))
    }

    /// log|I - ρW| like [`SpatialWeights::log_det`], with the errors of
    /// [`SpatialWeights::try_eigenvalues`].
    pub fn try_log_det(&mut self, rho: f64) -> EconResult<f64> {
        Ok(log_det_from(self.try_eigenvalues()?, rho))
    }

    /// The cached spectrum, computed fallibly when missing.
    fn try_spectrum(&mut self) -> EconResult<&Arc<Spectrum>> {
        let spectrum = match self.spectrum.take() {
            Some(s) => s,
            None => {
                checkpoint()?;
                Arc::new(compute_spectrum(&self.try_to_dense()?))
            }
        };
        Ok(self.spectrum.insert(spectrum))
    }

    /// tr((I - ρW)^-1) = Σ Re(1 / (1 - ρλ)) over the complex eigenvalues λ of W.
    ///
    /// `None` when the eigen decomposition failed or the sum is not finite (1 - ρλ = 0).
    pub(crate) fn trace_inverse_i_minus_rho_w(&mut self, rho: f64) -> EconResult<Option<f64>> {
        let spectrum = self.try_spectrum()?;
        if !spectrum.ok {
            return Ok(None);
        }
        let sum: f64 = spectrum
            .values
            .iter()
            .map(|&(a, b)| {
                // 1 - ρ(a + bi) = (1 - ρa) - ρb i
                let re = 1.0 - rho * a;
                let im = rho * b;
                re / (re * re + im * im)
            })
            .sum();
        Ok(sum.is_finite().then_some(sum))
    }

    /// The common row sum of W, if every row sums to the same value (within 1e-12
    /// relative), as row-standardised weights without isolates do.
    pub(crate) fn constant_row_sum(&self) -> Option<f64> {
        let mut sums = self.weights.iter().map(|sw| sw.weights.iter().sum::<f64>());
        let first = sums.next()?;
        let tol = 1e-12 * first.abs().max(1.0);
        sums.all(|s| (s - first).abs() <= tol).then_some(first)
    }

    /// Solves (I - ρW) v = b.
    ///
    /// With a spectral radius |ρ| max|λ| below 0.99 this is the Neumann iteration
    /// v ← b + ρWv (O(links) per step); otherwise, or if the iteration does not settle, a
    /// dense LU solve on a fallibly allocated n x n matrix.
    pub(crate) fn solve_i_minus_rho_w(
        &mut self,
        rho: f64,
        b: &Array1<f64>,
    ) -> EconResult<Array1<f64>> {
        let spectrum = Arc::clone(self.try_spectrum()?);
        let q = if spectrum.ok {
            rho.abs()
                * spectrum
                    .values
                    .iter()
                    .map(|&(a, im)| a.hypot(im))
                    .fold(0.0, f64::max)
        } else {
            f64::INFINITY
        };
        if q < 0.99 {
            let max_iter = if q > 0.0 {
                let needed = (1e-17_f64.ln() / q.ln()).ceil();
                (needed as usize).saturating_add(50).min(100_000)
            } else {
                50
            };
            let mut v = b.clone();
            for it in 0..max_iter {
                if it % 64 == 0 {
                    checkpoint()?;
                }
                let next = b + &(rho * &self.lag(&v));
                let delta = next
                    .iter()
                    .zip(v.iter())
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0, f64::max);
                let scale = next.iter().map(|x| x.abs()).fold(0.0, f64::max);
                v = next;
                if delta <= 1e-14 * scale.max(1.0) {
                    return Ok(v);
                }
            }
        }
        let mut a = try_zeros2(self.n, self.n, "dense matrix I - rho*W")?;
        for (i, sw) in self.weights.iter().enumerate() {
            a[[i, i]] = 1.0;
            for (&j, &w) in sw.indices.iter().zip(sw.weights.iter()) {
                a[[i, j]] -= rho * w;
            }
        }
        Ok(crate::linalg::matrix_ops::solve(&a.view(), b)?)
    }

    /// Get cached eigenvalues if available.
    pub fn get_eigenvalues(&self) -> Option<&Array1<f64>> {
        self.spectrum.as_ref().map(|s| &s.real_sorted)
    }

    /// Whether `other` is a clone of `self` (they share their weight lists). Weights built
    /// separately are never the same, even when they are equal.
    pub fn same_weights(&self, other: &SpatialWeights) -> bool {
        Arc::ptr_eq(&self.weights, &other.weights)
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
        log_det_from(self.eigenvalues(), rho)
    }

    /// Get the valid range for ρ based on eigenvalues.
    ///
    /// For the model to be valid, we need |I - ρW| > 0, which requires
    /// ρ to be in (1/λ_min, 1/λ_max) where λ_min and λ_max are the
    /// extreme eigenvalues of W.
    pub fn rho_range(&mut self) -> (f64, f64) {
        rho_range_from(self.eigenvalues())
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

                // Find w_ji (indices are sorted)
                let sw_j = &self.weights[j];
                match sw_j.indices.binary_search(&i) {
                    Ok(jdx) => {
                        if (sw_j.weights[jdx] - w_ij).abs() > 1e-10 {
                            return false;
                        }
                    }
                    Err(_) => return false,
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
    /// This is O(m log d) for m non-zero entries and at most d per row, not O(n³).
    pub fn trace_w2(&self) -> f64 {
        let mut trace = 0.0;
        for i in 0..self.n {
            let sw_i = &self.weights[i];
            for (idx, &j) in sw_i.indices.iter().enumerate() {
                let w_ij = sw_i.weights[idx];
                // Find w_ji in sparse representation (indices are sorted)
                let sw_j = &self.weights[j];
                if let Ok(jdx) = sw_j.indices.binary_search(&i) {
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

/// log|I - ρW| = Σ log(1 - ρλ_i) from the (real parts of the) eigenvalues.
fn log_det_from(eigs: &Array1<f64>, rho: f64) -> f64 {
    eigs.iter().map(|&lambda| (1.0 - rho * lambda).ln()).sum()
}

/// (1/λ_min, 1/λ_max), open-ended when no eigenvalue has that sign.
fn rho_range_from(eigs: &Array1<f64>) -> (f64, f64) {
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

/// Compute the eigenvalues of a matrix using faer.
fn compute_spectrum(m: &Array2<f64>) -> Spectrum {
    use crate::linalg::matrix_ops::ndarray_to_faer;

    let n = m.nrows();
    let mat = ndarray_to_faer(&m.view());

    // For general (non-symmetric) matrices, use complex eigenvalue decomposition
    // and take real parts (spatial weights matrices typically have real eigenvalues)
    let eigenvalues = mat.eigenvalues();

    // Extract real parts from the Result
    let (mut eigs_vec, values, ok): (Vec<f64>, Vec<(f64, f64)>, bool) = match eigenvalues {
        Ok(evs) => (
            evs.iter().map(|ev| ev.re).collect(),
            evs.iter().map(|ev| (ev.re, ev.im)).collect(),
            true,
        ),
        Err(_) => {
            // Fallback: return zeros if eigenvalue computation fails
            (vec![0.0; n], Vec::new(), false)
        }
    };

    // Sort eigenvalues
    eigs_vec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    Spectrum {
        real_sorted: Array1::from_vec(eigs_vec),
        values,
        ok,
    }
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
    fn fallible_eigen_paths_respect_the_ceiling_and_cancellation() {
        use crate::errors::EconError;
        use crate::resource::{Budget, CancelToken, with_alloc_ceiling, with_budget};
        let nb = Neighbors::fully_connected(64);
        let mut w = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);
        // a 64 x 64 matrix needs 32 KiB
        let r = with_alloc_ceiling(Some(1024), || w.try_eigenvalues().map(|e| e.len()));
        assert!(matches!(r, Err(EconError::ResourceLimit(_))), "{r:?}");
        let t = CancelToken::new();
        t.cancel();
        let r = with_budget(Budget::new(None, t), || w.try_rho_range());
        assert!(matches!(r, Err(EconError::Cancelled(_))), "{r:?}");
        assert!(w.get_eigenvalues().is_none());
        // the fallible forms return what the legacy ones return
        let mut legacy = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);
        assert_eq!(w.try_rho_range().unwrap(), legacy.rho_range());
        assert_eq!(w.try_log_det(0.3).unwrap(), legacy.log_det(0.3));
        assert_eq!(w.try_to_dense().unwrap(), legacy.to_dense());
    }

    #[test]
    fn from_neighbors_shares_the_neighbour_lists() {
        let nb = simple_neighbors();
        let w = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);
        assert!(std::ptr::eq(w.neighbors().as_slice(), nb.as_slice()));
    }

    #[test]
    fn clone_shares_weight_lists_and_eigen_cache() {
        let mut w = SpatialWeights::from_neighbors(&simple_neighbors(), WeightStyle::RowStd);
        let _ = w.eigenvalues();
        let c = w.clone();
        assert!(c.same_weights(&w));
        assert!(std::ptr::eq(c.get_weights(0), w.get_weights(0)));
        assert!(c.get_eigenvalues().is_some());
        assert!(
            !SpatialWeights::from_neighbors(&simple_neighbors(), WeightStyle::RowStd)
                .same_weights(&w)
        );
    }

    #[test]
    fn serialized_shape_is_unchanged() {
        let w = SpatialWeights::from_neighbors(&simple_neighbors(), WeightStyle::Binary);
        let v = serde_json::to_value(&w).unwrap();
        assert_eq!(v["neighbors"]["neighbors"][0], serde_json::json!([1, 2]));
        assert_eq!(v["weights"][0]["indices"], serde_json::json!([1, 2]));
        assert_eq!(v["n"], serde_json::json!(4));
        let back: SpatialWeights = serde_json::from_value(v).unwrap();
        assert_eq!(back.n(), 4);
        assert!(back.get_eigenvalues().is_none());
    }

    #[test]
    fn sparse_traces_and_symmetry_match_dense_on_wide_rows() {
        // fully connected, 199 links per row: the old position() scans were O(m * degree)
        for style in [WeightStyle::Binary, WeightStyle::RowStd] {
            let w = SpatialWeights::from_neighbors(&Neighbors::fully_connected(200), style);
            let d = w.to_dense();
            let tr_w2: f64 = d.dot(&d).diag().sum();
            let tr_wtw: f64 = d.iter().map(|v| v * v).sum();
            assert!((w.trace_w2() - tr_w2).abs() <= 1e-12 * tr_w2.abs().max(1.0));
            assert!((w.trace_wtw() - tr_wtw).abs() <= 1e-12 * tr_wtw.abs().max(1.0));
            assert!(w.is_symmetric());
        }
        assert!(!crate::spatial::test_fixtures::knn3(WeightStyle::Binary).is_symmetric());
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
