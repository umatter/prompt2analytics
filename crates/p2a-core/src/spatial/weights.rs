//! Spatial weights matrices.
//!
//! Provides spatial weights structures that combine neighbor definitions with
//! weighting schemes, equivalent to R's `listw` class from spdep.

use std::sync::Arc;

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use super::neighbors::Neighbors;
use crate::errors::{EconError, EconResult};
use crate::resource::{checkpoint, try_zeros2};

/// Weighting style for spatial weights matrix, as `spdep::nb2listw` with `zero.policy = TRUE`.
///
/// Controls how the binary neighbour list B is converted to weights. A unit without neighbours
/// (an island) keeps an empty row and does not count in `eff.n = n - #islands`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum WeightStyle {
    /// spdep "B": w_ij = 1.
    Binary,
    /// spdep "W": w_ij = 1 / n_i; each row with neighbours sums to 1.
    #[default]
    RowStd,
    /// spdep "C": w_ij = eff.n / sum(B); the weights sum to eff.n.
    GlobalStd,
    /// spdep "S" (Tiefelsdorf, Griffith & Boots 1999): w_ij = (eff.n / Q) / sqrt(n_i) with
    /// Q = sum_i sqrt(n_i); the weights sum to eff.n.
    VarStab,
    /// spdep "minmax" (Kelejian & Prucha 2010): w_ij = 1 / min(max_i row sum, max_j column sum).
    MinMax,
    /// spdep "U": w_ij = 1 / sum(B); the weights sum to 1.
    UnitSum,
    /// General weights not derived from a style, such as the unstandardised inverse-distance
    /// weights of [`SpatialWeights::from_distance_decay`]. A label only: `FromStr` refuses it,
    /// and [`SpatialWeights::from_neighbors`] has no general weights to apply, so it builds
    /// binary weights and labels them `Binary`.
    Custom,
}

crate::option_enum!(
    WeightStyle,
    crate::options::Choices {
        option: "weight style",
        values: &[
            (
                &["W", "row", "rowstd", "row_standardised", "row_standardized"],
                WeightStyle::RowStd
            ),
            (&["B", "binary"], WeightStyle::Binary),
            (
                &[
                    "C",
                    "global",
                    "globalstd",
                    "globally_standardised",
                    "globally_standardized"
                ],
                WeightStyle::GlobalStd
            ),
            (&["U", "unit_sum", "unitsum"], WeightStyle::UnitSum),
            (&["minmax", "mm", "min_max"], WeightStyle::MinMax),
            (
                &[
                    "S",
                    "varstab",
                    "variance_stabilising",
                    "variance_stabilizing"
                ],
                WeightStyle::VarStab
            ),
        ],
    },
    no_display
);

impl std::fmt::Display for WeightStyle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WeightStyle::Custom => f.write_str("custom"),
            s => f.write_str(Self::choices().canonical(*s)),
        }
    }
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
    /// All eigenvalues as (real, imaginary) pairs.
    values: Vec<(f64, f64)>,
    /// False for the NaN placeholder the infallible legacy methods cache after a failure.
    ok: bool,
}

/// The eigenvalues of W as the spatial likelihoods need them (spatialreg method "eigen").
///
/// Built by [`SpatialWeights::try_spectrum`], which caches it; cheap to clone.
#[derive(Debug, Clone)]
pub struct WeightsSpectrum(Arc<Spectrum>);

impl WeightsSpectrum {
    /// log|I - ρW| = Re Σ log(1 - ρλ_i) = Σ ln|1 - ρλ_i| over the complex eigenvalues. A conjugate
    /// pair a ± bi contributes ln((1 - ρa)² + (ρb)²). One logarithm per eigenvalue, the method
    /// is evaluated at every step of a likelihood search.
    pub fn log_det(&self, rho: f64) -> f64 {
        self.0
            .values
            .iter()
            .map(|&(a, b)| {
                let u = 1.0 - rho * a;
                if b == 0.0 {
                    u.abs().ln()
                } else {
                    let v = rho * b;
                    0.5 * (u * u + v * v).ln()
                }
            })
            .sum()
    }

    /// tr((I - ρW)^-1) = Re Σ 1 / (1 - ρλ_i).
    pub fn trace_inverse(&self, rho: f64) -> f64 {
        self.0
            .values
            .iter()
            .map(|&(a, b)| {
                let u = 1.0 - rho * a;
                let v = -rho * b;
                u / (u * u + v * v)
            })
            .sum()
    }

    /// tr(W (I - ρW)^-1) = Re Σ λ_i / (1 - ρλ_i).
    pub fn trace_w_inverse(&self, rho: f64) -> f64 {
        self.0
            .values
            .iter()
            .map(|&(a, b)| {
                let u = 1.0 - rho * a;
                let v = -rho * b;
                (a * u + b * v) / (u * u + v * v)
            })
            .sum()
    }

    /// min_i |1 - ρλ_i|; 0 means that I - ρW is singular.
    pub fn min_modulus(&self, rho: f64) -> f64 {
        self.0
            .values
            .iter()
            .map(|&(a, b)| {
                let u = 1.0 - rho * a;
                let v = rho * b;
                u * u + v * v
            })
            .fold(f64::INFINITY, f64::min)
            .sqrt()
    }

    /// The number of eigenvalues with |Im λ| above `im_tol`.
    pub fn n_complex(&self, im_tol: f64) -> usize {
        self.0
            .values
            .iter()
            .filter(|(_, b)| b.abs() > im_tol)
            .count()
    }

    /// The eigenvalues as (real, imaginary) pairs, in the order of the eigen decomposition.
    pub fn eigenvalues(&self) -> &[(f64, f64)] {
        &self.0.values
    }

    /// The real parts, sorted ascending (what [`SpatialWeights::eigenvalues`] returns).
    pub fn real_parts_sorted(&self) -> &Array1<f64> {
        &self.0.real_sorted
    }
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
        let style = if style == WeightStyle::Custom {
            WeightStyle::Binary
        } else {
            style
        };
        let card = nb.cardinalities();
        let eff_n = card.iter().filter(|&&c| c > 0).count() as f64;
        let links = nb.n_links() as f64;

        // The one weight of each row (spdep nb2listw on the binary list)
        let per_row: Vec<f64> = match style {
            WeightStyle::Binary | WeightStyle::Custom => vec![1.0; n],
            WeightStyle::RowStd => card
                .iter()
                .map(|&c| if c > 0 { 1.0 / c as f64 } else { 0.0 })
                .collect(),
            WeightStyle::GlobalStd => {
                vec![if links > 0.0 { eff_n / links } else { 0.0 }; n]
            }
            WeightStyle::UnitSum => vec![if links > 0.0 { 1.0 / links } else { 0.0 }; n],
            WeightStyle::MinMax => {
                let mut col_sums = vec![0usize; n];
                for i in 0..n {
                    for &j in nb.get_neighbors(i) {
                        if let Some(c) = col_sums.get_mut(j) {
                            *c += 1;
                        }
                    }
                }
                let max_row = card.iter().copied().max().unwrap_or(0);
                let max_col = col_sums.iter().copied().max().unwrap_or(0);
                let divisor = max_row.min(max_col) as f64;
                vec![if divisor > 0.0 { 1.0 / divisor } else { 0.0 }; n]
            }
            WeightStyle::VarStab => {
                // Rows scaled by 1 / sqrt(sum_j b_ij^2) = 1 / sqrt(n_i), then all rescaled to eff.n
                let q: f64 = card.iter().map(|&c| (c as f64).sqrt()).sum();
                card.iter()
                    .map(|&c| {
                        if c > 0 && q > 0.0 {
                            eff_n / q / (c as f64).sqrt()
                        } else {
                            0.0
                        }
                    })
                    .collect()
            }
        };
        let weights: Vec<SparseWeights> = (0..n)
            .map(|i| SparseWeights {
                indices: nb.get_neighbors(i).to_vec(),
                weights: vec![per_row[i]; card[i]],
            })
            .collect();

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
            WeightStyle::Custom
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

    /// Compute the real parts of the eigenvalues of W, sorted ascending.
    ///
    /// Eigenvalues are needed for ML estimation to compute log|I - ρW|
    /// and to determine the valid range for ρ.
    ///
    /// This uses dense matrix operations so is only suitable for moderate n. If the
    /// decomposition fails, every eigenvalue is reported as NaN;
    /// [`SpatialWeights::try_eigenvalues`] reports the failure as an error.
    pub fn eigenvalues(&mut self) -> &Array1<f64> {
        &self.cached_or_failed_spectrum().real_sorted
    }

    /// The cached spectrum, computed when missing; a failure is cached as the NaN placeholder.
    fn cached_or_failed_spectrum(&mut self) -> &Arc<Spectrum> {
        let spectrum = match self.spectrum.take() {
            Some(s) => s,
            None => Arc::new(
                compute_spectrum(&self.to_dense()).unwrap_or_else(|_| failed_spectrum(self.n)),
            ),
        };
        self.spectrum.insert(spectrum)
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
    /// them is allocated fallibly, an installed budget is checked first and a failed eigen
    /// decomposition is an error.
    ///
    /// # Errors
    ///
    /// As [`SpatialWeights::try_spectrum`].
    pub fn try_eigenvalues(&mut self) -> EconResult<&Array1<f64>> {
        self.try_spectrum()?;
        match &self.spectrum {
            Some(s) => Ok(&s.real_sorted),
            None => Err(spectrum_unavailable()),
        }
    }

    /// The valid range for ρ like [`SpatialWeights::rho_range`], with the errors of
    /// [`SpatialWeights::try_spectrum`].
    pub fn try_rho_range(&mut self) -> EconResult<(f64, f64)> {
        Ok(rho_range_from(self.try_eigenvalues()?))
    }

    /// log|I - ρW| from the complex eigenvalues ([`WeightsSpectrum::log_det`]), with the errors
    /// of [`SpatialWeights::try_spectrum`].
    pub fn try_log_det(&mut self, rho: f64) -> EconResult<f64> {
        Ok(self.try_spectrum()?.log_det(rho))
    }

    /// The eigenvalues of W, computed once (cached, and shared by clones).
    ///
    /// # Errors
    ///
    /// [`EconError::Computation`] when a weight is not finite, or the eigen decomposition does
    /// not converge or returns a non-finite value (not cached);
    /// [`EconError::ResourceLimit`] when the dense matrix cannot be allocated;
    /// [`EconError::Cancelled`] when an installed budget has expired.
    pub fn try_spectrum(&mut self) -> EconResult<WeightsSpectrum> {
        if let Some(s) = &self.spectrum {
            return if s.ok {
                Ok(WeightsSpectrum(Arc::clone(s)))
            } else {
                Err(spectrum_unavailable())
            };
        }
        checkpoint()?;
        let spectrum = Arc::new(compute_spectrum(&self.try_to_dense()?)?);
        self.spectrum = Some(Arc::clone(&spectrum));
        Ok(WeightsSpectrum(spectrum))
    }

    /// tr((I - ρW)^-1) = Σ Re(1 / (1 - ρλ)) over the complex eigenvalues λ of W.
    ///
    /// `None` when the sum is not finite (1 - ρλ = 0).
    pub(crate) fn trace_inverse_i_minus_rho_w(&mut self, rho: f64) -> EconResult<Option<f64>> {
        let sum = self.try_spectrum()?.trace_inverse(rho);
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
        let spectrum = self.try_spectrum()?;
        let q = rho.abs()
            * spectrum
                .eigenvalues()
                .iter()
                .map(|&(a, im)| a.hypot(im))
                .fold(0.0, f64::max);
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

    /// Compute log|I - ρW| from the complex eigenvalues (spatialreg method "eigen").
    ///
    /// log|I - ρW| = Re Σ log(1 - ρλ_i) = Σ ln|1 - ρλ_i|: a conjugate pair a ± bi contributes
    /// ln((1 - ρa)² + (ρb)²), and summing ln(1 - ρa) over the real parts alone is wrong for an
    /// asymmetric W such as k nearest neighbours.
    ///
    /// # Arguments
    ///
    /// * `rho` - Spatial autoregressive parameter
    ///
    /// # Returns
    ///
    /// The log-determinant value, NaN when the eigen decomposition failed;
    /// [`SpatialWeights::try_log_det`] reports the failure as an error.
    pub fn log_det(&mut self, rho: f64) -> f64 {
        WeightsSpectrum(Arc::clone(self.cached_or_failed_spectrum())).log_det(rho)
    }

    /// Get the valid range for ρ based on eigenvalues.
    ///
    /// The interval is (1/λ_min, 1/λ_max) over the **real parts** of the eigenvalues, as
    /// spatialreg's `eig.range`. It is (NaN, NaN) when the eigen decomposition failed;
    /// [`SpatialWeights::try_rho_range`] reports the failure as an error.
    pub fn rho_range(&mut self) -> (f64, f64) {
        let spectrum = self.cached_or_failed_spectrum();
        if spectrum.ok {
            rho_range_from(&spectrum.real_sorted)
        } else {
            (f64::NAN, f64::NAN)
        }
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

/// The eigenvalues of a dense W, from faer's general (complex) decomposition.
fn compute_spectrum(w: &Array2<f64>) -> EconResult<Spectrum> {
    use crate::linalg::matrix_ops::ndarray_to_faer;

    let n = w.nrows();
    let bad = w.iter().filter(|v| !v.is_finite()).count();
    if bad > 0 {
        return Err(EconError::Computation(format!(
            "spatial weights contain {bad} non-finite value(s); log|I - rho W| is undefined"
        )));
    }
    let evs = ndarray_to_faer(&w.view()).eigenvalues().map_err(|_| {
        EconError::Computation(format!(
            "eigendecomposition of the {n} x {n} spatial weights matrix did not converge; \
             log|I - rho W| cannot be computed"
        ))
    })?;
    let values: Vec<(f64, f64)> = evs.iter().map(|ev| (ev.re, ev.im)).collect();
    if values.iter().any(|(a, b)| !a.is_finite() || !b.is_finite()) {
        return Err(EconError::Computation(format!(
            "eigendecomposition of the {n} x {n} spatial weights matrix returned a non-finite \
             eigenvalue; log|I - rho W| cannot be computed"
        )));
    }
    let mut real: Vec<f64> = values.iter().map(|&(a, _)| a).collect();
    real.sort_by(f64::total_cmp);
    Ok(Spectrum {
        real_sorted: Array1::from_vec(real),
        values,
        ok: true,
    })
}

/// What the infallible legacy methods cache after a failed decomposition: every value NaN.
fn failed_spectrum(n: usize) -> Spectrum {
    Spectrum {
        real_sorted: Array1::from_elem(n, f64::NAN),
        values: vec![(f64::NAN, f64::NAN); n],
        ok: false,
    }
}

fn spectrum_unavailable() -> EconError {
    EconError::Computation(
        "the eigenvalues of the spatial weights matrix could not be computed".to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::columns::{NaPolicy, numeric_column};
    use crate::errors::EconError;
    use crate::spatial::{MoranAlternative, moran_test};

    fn col(ds: &crate::data::Dataset, name: &str) -> Vec<f64> {
        numeric_column(ds.df(), name, NaPolicy::Fail).unwrap().0
    }

    fn coords_of(ds: &crate::data::Dataset) -> Vec<(f64, f64)> {
        col(ds, "cx").into_iter().zip(col(ds, "cy")).collect()
    }

    fn expect_spec(r: EconResult<impl std::fmt::Debug>, needles: &[&str]) {
        match r {
            Err(EconError::InvalidSpecification { message }) => {
                for n in needles {
                    assert!(message.contains(n), "`{message}` does not mention `{n}`");
                }
            }
            other => panic!("expected InvalidSpecification mentioning {needles:?}, got {other:?}"),
        }
    }

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

    #[test]
    fn test_weight_style_parses_aliases_and_rejects_unknown() {
        WeightStyle::choices().validate().unwrap();
        use WeightStyle::*;
        for (s, v) in [
            ("W", RowStd),
            ("w", RowStd),
            ("row", RowStd),
            ("RowStd", RowStd),
            ("row-standardised", RowStd),
            ("B", Binary),
            ("binary", Binary),
            ("C", GlobalStd),
            ("global", GlobalStd),
            ("U", UnitSum),
            ("unit sum", UnitSum),
            ("minmax", MinMax),
            ("MM", MinMax),
            ("min-max", MinMax),
            ("S", VarStab),
            ("varstab", VarStab),
            ("Variance_Stabilizing", VarStab),
        ] {
            assert_eq!(s.parse::<WeightStyle>().unwrap(), v, "{s}");
        }
        for bad in ["custom", "unstandardized", "Q", ""] {
            expect_spec(
                bad.parse::<WeightStyle>(),
                &["weight style", "W, B, C, U, minmax, S"],
            );
        }
        let shown: Vec<String> = [RowStd, Binary, GlobalStd, UnitSum, MinMax, VarStab, Custom]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(shown, ["W", "B", "C", "U", "minmax", "S", "custom"]);
    }

    /// R: spdep::nb2listw styles written out (validation/r/cases/spatial_ml_weights.R, case
    /// spatial_weights_moran).
    #[test]
    fn test_validate_weight_styles() {
        use WeightStyle::*;
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("spatial_weights_moran");
        let coords = coords_of(&ds);
        let n = coords.len();
        let x1 = Array1::from(col(&ds, "x1"));
        let band = Neighbors::try_from_distance(&coords, 0.0, 0.2, None).unwrap();
        let cards: Vec<f64> = band.cardinalities().iter().map(|&c| c as f64).collect();
        let mut chk = Check::new(&case);
        chk.scalar("band_links", band.n_links() as f64, Tol::exact())
            .vector("band_row_sums", &cards, Tol::exact());
        for (key, style) in [
            ("W", RowStd),
            ("B", Binary),
            ("C", GlobalStd),
            ("U", UnitSum),
            ("minmax", MinMax),
            ("S", VarStab),
        ] {
            let w = SpatialWeights::from_neighbors(&band, style);
            assert_eq!(w.style(), style);
            let row_sums: Vec<f64> = (0..n)
                .map(|i| w.get_weights(i).weights.iter().sum())
                .collect();
            let moran = moran_test(&x1, &w, MoranAlternative::Greater).unwrap();
            chk.vector(
                &format!("style_{key}_row1"),
                &w.get_weights(0).weights,
                Tol::coef(n),
            )
            .scalar(&format!("style_{key}_sum"), w.sum_weights(), Tol::coef(n))
            .vector(&format!("style_{key}_row_sums"), &row_sums, Tol::coef(n))
            .scalar(
                &format!("style_{key}_moran_I"),
                moran.statistic,
                Tol::stat(n),
            );
        }
        // minmax divides by min(max row sum 3, max column sum 2) (spdep; Kelejian & Prucha 2010)
        let asym = SpatialWeights::from_neighbors(
            &Neighbors::from_indices(vec![vec![1, 2, 3], vec![2], vec![3], vec![1]]),
            MinMax,
        );
        chk.scalar(
            "asym_minmax_weight",
            asym.get_weights(0).weights[0],
            Tol::exact(),
        )
        .scalar("asym_minmax_sum", asym.sum_weights(), Tol::coef(4));
        // islands: C and S scale to eff.n = n - #islands (zero.policy = TRUE), U to 1
        let isl = Neighbors::try_from_distance(&coords, 0.0, 0.12, None).unwrap();
        let ids: Vec<f64> = isl.isolates().iter().map(|&i| i as f64).collect();
        chk.scalar("n_islands", ids.len() as f64, Tol::exact())
            .vector("islands", &ids, Tol::exact())
            .scalar(
                "islands_style_C_sum",
                SpatialWeights::from_neighbors(&isl, GlobalStd).sum_weights(),
                Tol::coef(n),
            )
            .scalar(
                "islands_style_S_sum",
                SpatialWeights::from_neighbors(&isl, VarStab).sum_weights(),
                Tol::coef(n),
            )
            .scalar(
                "islands_style_U_sum",
                SpatialWeights::from_neighbors(&isl, UnitSum).sum_weights(),
                Tol::coef(n),
            )
            .finish();
    }

    #[test]
    fn test_distance_decay_weights_are_labelled_custom() {
        let coords = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 2.0), (1.5, 1.5)];
        let nb = Neighbors::from_knn(&coords, 2);
        assert_eq!(
            SpatialWeights::from_distance_decay(&nb, &coords, 1.0, false).style(),
            WeightStyle::Custom
        );
        assert_eq!(
            SpatialWeights::from_distance_decay(&nb, &coords, 1.0, true).style(),
            WeightStyle::RowStd
        );
        // from_neighbors has no general weights to apply: Custom builds binary weights and says so
        let b = SpatialWeights::from_neighbors(&nb, WeightStyle::Custom);
        assert_eq!(b.style(), WeightStyle::Binary);
        assert!(b.get_weights(0).weights.iter().all(|&w| w == 1.0));
    }

    /// The case's W: 4 nearest neighbours on (cx, cy), row-standardised (spdep style "W").
    fn knn4(ds: &crate::data::Dataset) -> SpatialWeights {
        SpatialWeights::from_neighbors(
            &Neighbors::try_from_knn(&coords_of(ds), 4).unwrap(),
            WeightStyle::RowStd,
        )
    }

    /// R: Re(sum(log(1 - rho * eigen(W)$values))) (spatialreg method "eigen"); cross-checked against
    /// determinant(I - rho W) in the case. spatial_ml_weights.R, case spatial_sar_sdm_sem.
    #[test]
    fn test_validate_logdet_complex_eigenvalues() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("spatial_sar_sdm_sem");
        let mut w = knn4(&ds);
        let n = w.n();
        let rhos = case.vector("logdet_rho");
        let spec = w.try_spectrum().unwrap();
        let ld: Vec<f64> = rhos.iter().map(|&r| w.try_log_det(r).unwrap()).collect();
        let legacy: Vec<f64> = rhos.iter().map(|&r| w.log_det(r)).collect();
        assert_eq!(ld, legacy);
        let (lo, hi) = w.try_rho_range().unwrap();
        Check::new(&case)
            .vector("logdet", &ld, Tol::stat(n))
            .vector("logdet_direct", &ld, Tol::stat(n))
            // |Im| > 1e-4: the genuine conjugate pairs (near-defective eigenvalues carry
            // |Im| <= 1.2e-6 of rounding)
            .scalar("n_complex_eigen", spec.n_complex(1e-4) as f64, Tol::exact())
            .vector("rho_range", &[lo, hi], Tol::coef(n))
            .scalar("tr_inv_rho05", spec.trace_inverse(0.5), Tol::stat(n))
            .scalar("tr_aw_rho05", spec.trace_w_inverse(0.5), Tol::stat(n))
            .finish();
    }

    #[test]
    fn test_eigen_failure_is_an_error_not_zeros() {
        let mut w = SpatialWeights::from_neighbors(&simple_neighbors(), WeightStyle::RowStd);
        Arc::make_mut(&mut w.weights)[0].weights[0] = f64::NAN;
        match w.try_log_det(0.5) {
            Err(EconError::Computation(m)) => assert!(m.contains("non-finite"), "{m}"),
            other => panic!("expected Computation, got {other:?}"),
        }
        assert!(w.try_spectrum().is_err() && w.try_eigenvalues().is_err());
        assert!(w.try_rho_range().is_err());
        assert!(w.log_det(0.5).is_nan());
        let (lo, hi) = w.rho_range();
        assert!(lo.is_nan() && hi.is_nan());
        assert!(w.eigenvalues().iter().all(|v| v.is_nan()));
        // the placeholder stays an error for the fallible forms
        assert!(w.try_spectrum().is_err());
    }

    /// The spatial timing cells compute the eigenvalues once and fit many times.
    #[test]
    fn test_spectrum_is_cached_after_eigenvalues() {
        let mut w = SpatialWeights::from_neighbors(&simple_neighbors(), WeightStyle::RowStd);
        let _ = w.eigenvalues();
        let a = w.try_spectrum().unwrap();
        let b = w.try_spectrum().unwrap();
        assert!(Arc::ptr_eq(&a.0, &b.0));
    }

    /// `log_det` on a W with complex eigenvalues is the log-determinant of the dense matrix: a
    /// directed 3-cycle has eigenvalues 1 and exp(±2πi/3), det(I - ρW) = 1 - ρ³.
    #[test]
    fn test_log_det_of_a_directed_cycle() {
        let nb = Neighbors::from_indices(vec![vec![1], vec![2], vec![0]]);
        let mut w = SpatialWeights::from_neighbors(&nb, WeightStyle::Binary);
        for rho in [-0.8_f64, 0.3, 0.9] {
            let exact = (1.0 - rho.powi(3)).ln();
            assert!((w.try_log_det(rho).unwrap() - exact).abs() < 1e-12, "{rho}");
        }
    }
}
