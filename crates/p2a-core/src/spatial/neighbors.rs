//! Spatial neighbors definitions.
//!
//! Provides structures and methods for defining which observations are neighbors
//! of each other, based on distance, k-nearest neighbors, or explicit specification.
//!
//! k-nearest-neighbour and distance-band neighbours are found with a bucketed k-d tree
//! (planar coordinates, or unit vectors on the sphere for longitude/latitude), so building
//! them costs O(n log n + links) for well-spread points instead of visiting all n² pairs.
//! Coordinates shared by many observations do not make it quadratic: a k-nearest search
//! also rules a box out by the lowest index it holds, and a band search skips a box whose
//! points all sit at the query. Candidates are always judged with the exact distance
//! formula, and ties are broken by the lower index, so the result is the one an all-pairs
//! search gives.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::errors::{EconError, EconResult};
use crate::resource::{Budget, checkpoint, with_budget};

/// Method used to construct neighbors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NeighborMethod {
    /// K-nearest neighbors
    Knn { k: usize },
    /// Distance threshold (within d1 to d2)
    Distance { d1: f64, d2: f64 },
    /// Explicit specification
    Explicit,
    /// Contiguity (queen or rook)
    Contiguity { queen: bool },
}

/// Neighbors list - indices of neighbors for each observation.
///
/// Equivalent to R's `nb` class from the spdep package.
/// For each observation i, stores the indices of observations considered neighbors.
///
/// # Example
///
/// ```
/// use p2a_core::spatial::Neighbors;
///
/// // Create from coordinates using k-nearest neighbors
/// let coords = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
/// let nb = Neighbors::from_knn(&coords, 2);
///
/// // Check neighbors
/// assert!(nb.are_neighbors(0, 1)); // 0 and 1 are neighbors
/// ```
///
/// The neighbour lists are shared, so `clone()` is O(1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neighbors {
    /// For each observation i, sorted vector of neighbor indices (0-indexed)
    /// Empty vector means no neighbors
    #[serde(with = "arc_serde")]
    neighbors: Arc<Vec<Vec<usize>>>,
    /// Number of observations
    n: usize,
    /// Region identifiers (optional)
    region_ids: Option<Vec<String>>,
    /// Whether the neighborhood is symmetric (i neighbor of j implies j neighbor of i)
    symmetric: bool,
    /// Method used to construct neighbors
    method: NeighborMethod,
}

impl Neighbors {
    /// Create neighbors from explicit neighbor list.
    ///
    /// # Arguments
    ///
    /// * `neighbors` - For each observation, a vector of neighbor indices (0-indexed)
    ///
    /// # Example
    ///
    /// ```
    /// use p2a_core::spatial::Neighbors;
    ///
    /// let nb = Neighbors::from_indices(vec![
    ///     vec![1, 2],    // Observation 0 has neighbors 1 and 2
    ///     vec![0, 2, 3], // Observation 1 has neighbors 0, 2, and 3
    ///     vec![0, 1, 3], // etc.
    ///     vec![1, 2],
    /// ]);
    /// ```
    pub fn from_indices(neighbors: Vec<Vec<usize>>) -> Self {
        let n = neighbors.len();
        let sorted_neighbors: Vec<Vec<usize>> = neighbors
            .into_iter()
            .map(|mut nb| {
                nb.sort_unstable();
                nb.dedup();
                nb
            })
            .collect();

        // Check symmetry
        let symmetric = Self::check_symmetry(&sorted_neighbors);

        Neighbors {
            neighbors: Arc::new(sorted_neighbors),
            n,
            region_ids: None,
            symmetric,
            method: NeighborMethod::Explicit,
        }
    }

    /// Create k-nearest neighbors from coordinates.
    ///
    /// For each observation, finds the k closest observations based on Euclidean distance.
    /// Note: k-NN is inherently asymmetric (if i is among j's k nearest, j may not be among i's).
    ///
    /// # Arguments
    ///
    /// * `coords` - Vector of (x, y) coordinate pairs
    /// * `k` - Number of nearest neighbors
    ///
    /// # Example
    ///
    /// ```
    /// use p2a_core::spatial::Neighbors;
    ///
    /// let coords = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
    /// let nb = Neighbors::from_knn(&coords, 2);
    /// ```
    ///
    /// `k` is clamped to `n - 1`. Coordinates that are not finite fall back to an all-pairs
    /// search, as before. Use [`Neighbors::try_from_knn`] to have invalid input rejected.
    pub fn from_knn(coords: &[(f64, f64)], k: usize) -> Self {
        let n = coords.len();
        let k = k.min(n.saturating_sub(1)); // Can't have more neighbors than n-1
        if k == 0 {
            return Self::from_lists(vec![Vec::new(); n], NeighborMethod::Knn { k }, None);
        }
        // An installed budget must not make the legacy constructor fail.
        with_budget(Budget::unlimited(), || Self::try_from_knn(coords, k)).unwrap_or_else(|_| {
            Self::from_lists(knn_brute_force(coords, k), NeighborMethod::Knn { k }, None)
        })
    }

    /// k nearest neighbours (Euclidean), with input validation and cooperative cancellation.
    ///
    /// For each observation, its `k` closest other observations; distance ties are broken by
    /// the lower index, exactly as an all-pairs search with a stable sort would. Runs in
    /// O(n log n) time for well-spread points (a bucketed k-d tree), also when many
    /// observations share one location, and O(n k) memory.
    ///
    /// # Errors
    ///
    /// - [`EconError::InvalidSpecification`] if there are fewer than 2 observations, a
    ///   coordinate is not finite, or `k` is not between 1 and `n - 1`.
    /// - [`EconError::Cancelled`] if a budget installed with
    ///   [`with_budget`](crate::resource::with_budget) expires.
    ///
    /// # Example
    ///
    /// ```
    /// use p2a_core::spatial::Neighbors;
    ///
    /// let coords = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (3.0, 3.0)];
    /// let nb = Neighbors::try_from_knn(&coords, 1).unwrap();
    /// assert_eq!(nb.get_neighbors(0), &[1]); // (1, 0) and (0, 1) tie; the lower index wins
    /// assert!(Neighbors::try_from_knn(&coords, 4).is_err()); // k must be at most n - 1
    /// ```
    pub fn try_from_knn(coords: &[(f64, f64)], k: usize) -> EconResult<Neighbors> {
        let n = coords.len();
        validate_coords(coords, Metric::Euclidean)?;
        if k == 0 || k >= n {
            return Err(invalid(format!(
                "k must be between 1 and n - 1 = {}, got {k}",
                n - 1
            )));
        }
        let (lists, _) = knn_lists(coords, k)?;
        let symmetric = Self::check_symmetry(&lists);
        Ok(Self::from_lists(
            lists,
            NeighborMethod::Knn { k },
            Some(symmetric),
        ))
    }

    /// Create neighbors based on distance threshold.
    ///
    /// Two observations are neighbors if their distance is between d1 and d2 (exclusive of d1).
    /// Setting d1 = 0 includes all observations within distance d2.
    ///
    /// # Arguments
    ///
    /// * `coords` - Vector of (x, y) coordinate pairs
    /// * `d1` - Minimum distance (exclusive, usually 0)
    /// * `d2` - Maximum distance (inclusive)
    ///
    /// # Example
    ///
    /// ```
    /// use p2a_core::spatial::Neighbors;
    ///
    /// let coords = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
    /// let nb = Neighbors::from_distance(&coords, 0.0, 1.5); // Neighbors within distance 1.5
    /// ```
    ///
    /// Input that [`Neighbors::try_from_distance`] rejects (non-finite coordinates or bounds,
    /// `d2 < d1`, fewer than 2 observations) falls back to an all-pairs search, as before.
    pub fn from_distance(coords: &[(f64, f64)], d1: f64, d2: f64) -> Self {
        Self::legacy_band(coords, d1, d2, Metric::Euclidean)
    }

    /// Create neighbors based on distance threshold with longlat coordinates.
    ///
    /// Uses great-circle distance (Haversine formula) for longitude/latitude coordinates.
    ///
    /// # Arguments
    ///
    /// * `coords` - Vector of (longitude, latitude) pairs in degrees
    /// * `d1` - Minimum distance in kilometers
    /// * `d2` - Maximum distance in kilometers
    ///
    /// Input that [`Neighbors::try_from_distance_longlat`] rejects falls back to an all-pairs
    /// search, as before.
    pub fn from_distance_longlat(coords: &[(f64, f64)], d1: f64, d2: f64) -> Self {
        Self::legacy_band(coords, d1, d2, Metric::GreatCircle)
    }

    /// Distance-band neighbours (Euclidean): pairs with `d1 < distance <= d2`, with input
    /// validation, a cap on the number of links and cooperative cancellation.
    ///
    /// Runs in O(n log n + links) time for well-spread points (a bucketed k-d tree), also
    /// when many observations share one location. The result is symmetric and identical to
    /// an all-pairs search with the same formula.
    ///
    /// # Errors
    ///
    /// - [`EconError::InvalidSpecification`] if there are fewer than 2 observations, a
    ///   coordinate is not finite, `d1` is negative or not finite, or `d2` is not finite or
    ///   smaller than `d1`.
    /// - [`EconError::ResourceLimit`] as soon as the band would hold more than `max_links`
    ///   directed links (each unordered pair counts twice). Nothing larger than the cap is
    ///   allocated.
    /// - [`EconError::Cancelled`] if a budget installed with
    ///   [`with_budget`](crate::resource::with_budget) expires.
    pub fn try_from_distance(
        coords: &[(f64, f64)],
        d1: f64,
        d2: f64,
        max_links: Option<usize>,
    ) -> EconResult<Neighbors> {
        Self::try_band(coords, d1, d2, max_links, Metric::Euclidean)
    }

    /// Distance-band neighbours for (longitude, latitude) coordinates in degrees, with
    /// great-circle (haversine, R = 6371 km) distances in kilometres.
    ///
    /// Same contract and complexity as [`Neighbors::try_from_distance`]; additionally a
    /// latitude outside [-90, 90] is an [`EconError::InvalidSpecification`].
    pub fn try_from_distance_longlat(
        coords: &[(f64, f64)],
        d1: f64,
        d2: f64,
        max_links: Option<usize>,
    ) -> EconResult<Neighbors> {
        Self::try_band(coords, d1, d2, max_links, Metric::GreatCircle)
    }

    fn try_band(
        coords: &[(f64, f64)],
        d1: f64,
        d2: f64,
        max_links: Option<usize>,
        metric: Metric,
    ) -> EconResult<Neighbors> {
        validate_coords(coords, metric)?;
        if !(d1.is_finite() && d1 >= 0.0) {
            return Err(invalid(format!(
                "minimum distance must be finite and >= 0, got {d1}"
            )));
        }
        if !(d2.is_finite() && d2 >= d1) {
            return Err(invalid(format!(
                "maximum distance must be finite and at least the minimum distance {d1}, got {d2}"
            )));
        }
        let (lists, _) = band_lists(coords, metric, d1, d2, max_links)?;
        // Distance-based neighbors are symmetric by construction
        Ok(Self::from_lists(
            lists,
            NeighborMethod::Distance { d1, d2 },
            Some(true),
        ))
    }

    fn legacy_band(coords: &[(f64, f64)], d1: f64, d2: f64, metric: Metric) -> Self {
        // An installed budget must not make the legacy constructor fail.
        with_budget(Budget::unlimited(), || {
            Self::try_band(coords, d1, d2, None, metric)
        })
        .unwrap_or_else(|_| {
            Self::from_lists(
                distance_brute_force(coords, d1, d2, metric),
                NeighborMethod::Distance { d1, d2 },
                Some(true),
            )
        })
    }

    /// Wraps finished, sorted neighbour lists; `symmetric: None` checks symmetry.
    fn from_lists(lists: Vec<Vec<usize>>, method: NeighborMethod, symmetric: Option<bool>) -> Self {
        let symmetric = symmetric.unwrap_or_else(|| Self::check_symmetry(&lists));
        Neighbors {
            n: lists.len(),
            neighbors: Arc::new(lists),
            region_ids: None,
            symmetric,
            method,
        }
    }

    /// Create inverse-distance neighbors (all pairs with distance weighting).
    ///
    /// All observations are neighbors, but weights will be based on inverse distance.
    /// This creates a fully-connected neighbor structure.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of observations
    pub fn fully_connected(n: usize) -> Self {
        let neighbors: Vec<Vec<usize>> = (0..n)
            .map(|i| (0..n).filter(|&j| j != i).collect())
            .collect();

        Neighbors {
            neighbors: Arc::new(neighbors),
            n,
            region_ids: None,
            symmetric: true,
            method: NeighborMethod::Explicit,
        }
    }

    /// Check if i and j are neighbors.
    pub fn are_neighbors(&self, i: usize, j: usize) -> bool {
        if i >= self.n || j >= self.n {
            return false;
        }
        self.neighbors[i].binary_search(&j).is_ok()
    }

    /// Get neighbors of observation i.
    pub fn get_neighbors(&self, i: usize) -> &[usize] {
        &self.neighbors[i]
    }

    /// Get the number of neighbors for observation i (cardinality).
    pub fn cardinality(&self, i: usize) -> usize {
        self.neighbors[i].len()
    }

    /// Get cardinalities for all observations.
    pub fn cardinalities(&self) -> Vec<usize> {
        self.neighbors.iter().map(|nb| nb.len()).collect()
    }

    /// Get the number of observations.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Check if the neighbor structure is symmetric.
    pub fn is_symmetric(&self) -> bool {
        self.symmetric
    }

    /// Get the total number of neighbor links.
    pub fn n_links(&self) -> usize {
        self.neighbors.iter().map(|nb| nb.len()).sum()
    }

    /// Get the average number of neighbors.
    pub fn avg_neighbors(&self) -> f64 {
        self.n_links() as f64 / self.n as f64
    }

    /// Make the neighbor structure symmetric.
    ///
    /// If i is a neighbor of j, makes j a neighbor of i. Costs O(m log d) for m links and
    /// at most d neighbours per observation.
    pub fn make_symmetric(&mut self) {
        if self.symmetric {
            return;
        }

        let mut new_neighbors: Vec<Vec<usize>> = (*self.neighbors).clone();

        // Add i as neighbor of j; duplicates are removed below
        for (i, row) in self.neighbors.iter().enumerate() {
            for &j in row {
                new_neighbors[j].push(i);
            }
        }

        // Sort all neighbor lists
        for nb in &mut new_neighbors {
            nb.sort_unstable();
            nb.dedup();
        }

        self.neighbors = Arc::new(new_neighbors);
        self.symmetric = true;
    }

    /// Set region identifiers.
    pub fn set_region_ids(&mut self, ids: Vec<String>) {
        if ids.len() == self.n {
            self.region_ids = Some(ids);
        }
    }

    /// Get region identifiers.
    pub fn region_ids(&self) -> Option<&Vec<String>> {
        self.region_ids.as_ref()
    }

    /// Get the neighbor construction method.
    pub fn method(&self) -> &NeighborMethod {
        &self.method
    }

    /// Get observations with no neighbors (isolates).
    pub fn isolates(&self) -> Vec<usize> {
        self.neighbors
            .iter()
            .enumerate()
            .filter(|(_, nb)| nb.is_empty())
            .map(|(i, _)| i)
            .collect()
    }

    /// Check if any observation has no neighbors.
    pub fn has_isolates(&self) -> bool {
        self.neighbors.iter().any(|nb| nb.is_empty())
    }

    /// Get the raw neighbor list.
    pub fn as_slice(&self) -> &[Vec<usize>] {
        self.neighbors.as_slice()
    }

    /// Check symmetry of a neighbor list.
    fn check_symmetry(neighbors: &[Vec<usize>]) -> bool {
        for (i, nb) in neighbors.iter().enumerate() {
            for &j in nb {
                if j >= neighbors.len() || neighbors[j].binary_search(&i).is_err() {
                    return false;
                }
            }
        }
        true
    }
}

/// Calculate great-circle distance using Haversine formula.
///
/// # Arguments
///
/// * `p1` - (longitude, latitude) in degrees
/// * `p2` - (longitude, latitude) in degrees
///
/// # Returns
///
/// Distance in kilometers
fn haversine_distance(p1: (f64, f64), p2: (f64, f64)) -> f64 {
    let (lon1, lat1) = (p1.0.to_radians(), p1.1.to_radians());
    let (lon2, lat2) = (p2.0.to_radians(), p2.1.to_radians());

    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;

    let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();

    EARTH_RADIUS_KM * c
}

/// Serde adapter for `Arc<T>` fields: serializes the shared value as `T` itself, so the
/// JSON shape is the one an unshared field had (serde's `rc` feature is not needed).
pub(super) mod arc_serde {
    use std::sync::Arc;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<T: Serialize, S: Serializer>(v: &Arc<T>, s: S) -> Result<S::Ok, S::Error> {
        T::serialize(&**v, s)
    }

    pub fn deserialize<'de, T: Deserialize<'de>, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Arc<T>, D::Error> {
        T::deserialize(d).map(Arc::new)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Neighbour search
// ═══════════════════════════════════════════════════════════════════════════

/// Earth radius used by [`haversine_distance`], in kilometres.
const EARTH_RADIUS_KM: f64 = 6371.0;
/// Maximum number of points in a k-d tree leaf.
const LEAF_SIZE: usize = 16;
/// Exact distance evaluations between two cancellation checkpoints.
const CHECKPOINT_EVALS: u64 = 65_536;

/// The distance a neighbour search uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Metric {
    /// Planar distance between (x, y) pairs.
    Euclidean,
    /// Great-circle distance in km between (longitude, latitude) pairs in degrees.
    GreatCircle,
}

impl Metric {
    /// The exact distance from `a` (the query) to `b`, with the formulas the all-pairs
    /// constructors always used.
    #[inline]
    fn distance(self, a: (f64, f64), b: (f64, f64)) -> f64 {
        match self {
            Metric::Euclidean => {
                let dx = a.0 - b.0;
                let dy = a.1 - b.1;
                (dx * dx + dy * dy).sqrt()
            }
            Metric::GreatCircle => haversine_distance(a, b),
        }
    }

    /// The k-d tree point for a coordinate pair: (x, y, 0) for planar coordinates, the unit
    /// vector for (longitude, latitude).
    fn point(self, c: (f64, f64)) -> [f64; 3] {
        match self {
            Metric::Euclidean => [c.0, c.1, 0.0],
            Metric::GreatCircle => {
                let (lon, lat) = (c.0.to_radians(), c.1.to_radians());
                [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()]
            }
        }
    }

    /// Number of tree coordinates the bounds use.
    fn dims(self) -> usize {
        match self {
            Metric::Euclidean => 2,
            Metric::GreatCircle => 3,
        }
    }
}

/// Work done by a neighbour query, for tests of its complexity.
#[derive(Debug, Default, Clone, Copy)]
struct QueryStats {
    /// Number of exact distance evaluations.
    dist_evals: u64,
}

fn invalid(message: String) -> EconError {
    EconError::InvalidSpecification { message }
}

/// The checks every `try_*` constructor makes first: at least 2 observations, finite
/// coordinates and, for longitude/latitude, latitudes in [-90, 90].
fn validate_coords(coords: &[(f64, f64)], metric: Metric) -> EconResult<()> {
    let n = coords.len();
    if n < 2 {
        return Err(invalid(format!(
            "need at least 2 observations to define neighbours, got {n}"
        )));
    }
    if let Some((i, &(x, y))) = coords
        .iter()
        .enumerate()
        .find(|(_, c)| !(c.0.is_finite() && c.1.is_finite()))
    {
        return Err(invalid(format!(
            "coordinates of observation {} are not finite ({x}, {y})",
            i + 1
        )));
    }
    if metric == Metric::GreatCircle
        && let Some((i, &(_, lat))) = coords
            .iter()
            .enumerate()
            .find(|(_, c)| !(-90.0..=90.0).contains(&c.1))
    {
        return Err(invalid(format!(
            "latitude of observation {} is outside [-90, 90]: {lat}",
            i + 1
        )));
    }
    Ok(())
}

/// The all-pairs k-nearest-neighbour search the constructor used to run. Kept as the
/// fallback for input the k-d tree path rejects and as the test reference.
fn knn_brute_force(coords: &[(f64, f64)], k: usize) -> Vec<Vec<usize>> {
    let n = coords.len();
    if n == 0 {
        return vec![];
    }
    let k = k.min(n.saturating_sub(1)); // Can't have more neighbors than n-1

    let mut neighbors = Vec::with_capacity(n);

    for i in 0..n {
        // Calculate distances to all other points
        let mut distances: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let dx = coords[i].0 - coords[j].0;
                let dy = coords[i].1 - coords[j].1;
                (j, (dx * dx + dy * dy).sqrt())
            })
            .collect();

        // Sort by distance and take k nearest
        distances.sort_by(|a, b| a.1.total_cmp(&b.1));
        let mut knn: Vec<usize> = distances.iter().take(k).map(|(j, _)| *j).collect();
        knn.sort_unstable();

        neighbors.push(knn);
    }

    neighbors
}

/// The all-pairs distance-band search the constructors used to run. Kept as the fallback
/// for input the k-d tree path rejects and as the test reference.
fn distance_brute_force(
    coords: &[(f64, f64)],
    d1: f64,
    d2: f64,
    metric: Metric,
) -> Vec<Vec<usize>> {
    let n = coords.len();
    let mut neighbors = Vec::with_capacity(n);

    for i in 0..n {
        let mut nb = Vec::new();
        for j in 0..n {
            if i != j {
                let dist = metric.distance(coords[i], coords[j]);
                if dist > d1 && dist <= d2 {
                    nb.push(j);
                }
            }
        }
        nb.sort_unstable();
        neighbors.push(nb);
    }

    neighbors
}

/// A node of [`KdTree`]: the bounding box of the points `order[start..end]`.
struct KdNode {
    lo: [f64; 3],
    hi: [f64; 3],
    start: usize,
    end: usize,
    /// The smallest observation index in `order[start..end]`.
    min_idx: usize,
    /// Whether every point has the input coordinates of `order[start]` (compared as given,
    /// not as tree coordinates).
    coincident: bool,
    /// Child node ids; `None` for a leaf.
    children: Option<(usize, usize)>,
}

/// A static, bucketed k-d tree. Every node stores its exact bounding box (minima and
/// maxima of its points' coordinates), so box bounds never cut off a point. Splits order
/// equal coordinates by observation index, so observations that share a location fill
/// index-contiguous subtrees.
struct KdTree {
    /// Tree coordinates, indexed by observation.
    points: Vec<[f64; 3]>,
    /// Observation indices; every node covers a contiguous range.
    order: Vec<usize>,
    nodes: Vec<KdNode>,
}

impl KdTree {
    /// Builds the tree over `metric`'s tree points for `coords` in O(n log n): each node
    /// splits on its widest axis at the median.
    fn build(coords: &[(f64, f64)], metric: Metric) -> Self {
        let n = coords.len();
        let mut tree = KdTree {
            points: coords.iter().map(|&c| metric.point(c)).collect(),
            order: (0..n).collect(),
            nodes: Vec::with_capacity(2 * (n / LEAF_SIZE + 1)),
        };
        if n > 0 {
            tree.build_node(coords, 0, n);
        }
        tree
    }

    fn build_node(&mut self, coords: &[(f64, f64)], start: usize, end: usize) -> usize {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        let mut min_idx = usize::MAX;
        let first = coords[self.order[start]];
        let mut coincident = true;
        for &j in &self.order[start..end] {
            let p = &self.points[j];
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
            min_idx = min_idx.min(j);
            coincident &= coords[j] == first;
        }
        let id = self.nodes.len();
        self.nodes.push(KdNode {
            lo,
            hi,
            start,
            end,
            min_idx,
            coincident,
            children: None,
        });
        let len = end - start;
        if len > LEAF_SIZE {
            let axis = (0..3)
                .max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b])))
                .unwrap_or(0);
            let mid = len / 2;
            let points = &self.points;
            self.order[start..end].select_nth_unstable_by(mid, |&a, &b| {
                points[a][axis].total_cmp(&points[b][axis]).then(a.cmp(&b))
            });
            let left = self.build_node(coords, start, start + mid);
            let right = self.build_node(coords, start + mid, end);
            self.nodes[id].children = Some((left, right));
        }
        id
    }

    /// Lower bound of the distance from `q` to any point of `node`, over the first `dims`
    /// coordinates. For planar points this uses the same floating-point operations as
    /// [`Metric::distance`] on a nearer coordinate difference; rounding is monotone, so the
    /// bound never exceeds a computed distance.
    #[inline]
    fn lower_bound(q: &[f64; 3], node: &KdNode, dims: usize) -> f64 {
        let mut s = 0.0;
        for a in 0..dims {
            let gap = if q[a] < node.lo[a] {
                node.lo[a] - q[a]
            } else if q[a] > node.hi[a] {
                q[a] - node.hi[a]
            } else {
                0.0
            };
            s += gap * gap;
        }
        s.sqrt()
    }

    /// Upper bound of the distance from `q` to any point of `node` (the farthest corner),
    /// computed like [`KdTree::lower_bound`].
    #[inline]
    fn upper_bound(q: &[f64; 3], node: &KdNode, dims: usize) -> f64 {
        let mut s = 0.0;
        for a in 0..dims {
            let far = (q[a] - node.lo[a]).max(node.hi[a] - q[a]);
            s += far * far;
        }
        s.sqrt()
    }
}

/// Counts exact evaluations and hits a cancellation checkpoint every [`CHECKPOINT_EVALS`].
struct EvalCounter {
    stats: QueryStats,
    last_check: u64,
}

impl EvalCounter {
    fn new() -> EconResult<Self> {
        checkpoint()?;
        Ok(EvalCounter {
            stats: QueryStats::default(),
            last_check: 0,
        })
    }

    #[inline]
    fn add(&mut self, evals: u64) -> EconResult<()> {
        self.stats.dist_evals += evals;
        if self.stats.dist_evals - self.last_check >= CHECKPOINT_EVALS {
            self.last_check = self.stats.dist_evals;
            checkpoint()?;
        }
        Ok(())
    }
}

/// A k-nearest-neighbour candidate, ordered by distance and then by index.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    dist: f64,
    idx: usize,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then(self.idx.cmp(&other.idx))
    }
}

/// k nearest neighbours (Euclidean) of every observation, each list sorted by index.
/// Expects validated input: finite coordinates and `1 <= k < n`.
fn knn_lists(coords: &[(f64, f64)], k: usize) -> EconResult<(Vec<Vec<usize>>, QueryStats)> {
    let metric = Metric::Euclidean;
    let dims = metric.dims();
    let n = coords.len();
    let tree = KdTree::build(coords, metric);
    let mut counter = EvalCounter::new()?;
    let mut lists = Vec::with_capacity(n);
    let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(k + 1);
    let mut stack: Vec<(usize, f64)> = Vec::new();

    for i in 0..n {
        let q = tree.points[i];
        heap.clear();
        stack.clear();
        if let Some(root) = tree.nodes.first() {
            stack.push((0, KdTree::lower_bound(&q, root, dims)));
        }
        while let Some((id, lower)) = stack.pop() {
            let node = &tree.nodes[id];
            // Every candidate in the box is at least (lower, min_idx) in the (distance,
            // index) order, so the box is skipped when that cannot beat the current worst.
            // A box at exactly the worst distance is still visited if it may hold a
            // lower-index tie, and only then: on duplicated coordinates every bound is 0.
            if heap.len() == k
                && heap.peek().is_some_and(|worst| {
                    Candidate {
                        dist: lower,
                        idx: node.min_idx,
                    } >= *worst
                })
            {
                continue;
            }
            match node.children {
                Some((left, right)) => {
                    let least = |child: usize| Candidate {
                        dist: KdTree::lower_bound(&q, &tree.nodes[child], dims),
                        idx: tree.nodes[child].min_idx,
                    };
                    let (bl, br) = (least(left), least(right));
                    // the child that may hold the smaller candidate is pushed last, so it
                    // is visited first
                    if bl <= br {
                        stack.push((right, br.dist));
                        stack.push((left, bl.dist));
                    } else {
                        stack.push((left, bl.dist));
                        stack.push((right, br.dist));
                    }
                }
                None => {
                    let mut evals = 0u64;
                    for &j in &tree.order[node.start..node.end] {
                        if j == i {
                            continue;
                        }
                        evals += 1;
                        let cand = Candidate {
                            dist: metric.distance(coords[i], coords[j]),
                            idx: j,
                        };
                        if heap.len() < k {
                            heap.push(cand);
                        } else if let Some(mut worst) = heap.peek_mut()
                            && cand < *worst
                        {
                            *worst = cand;
                        }
                    }
                    counter.add(evals)?;
                }
            }
        }
        let mut row: Vec<usize> = heap.drain().map(|c| c.idx).collect();
        row.sort_unstable();
        lists.push(row);
    }
    Ok((lists, counter.stats))
}

/// Distance-band neighbours (`d1 < dist <= d2`) of every observation, each list sorted by
/// index. Expects validated input. Stops with [`EconError::ResourceLimit`] as soon as the
/// total number of links would exceed `max_links`.
fn band_lists(
    coords: &[(f64, f64)],
    metric: Metric,
    d1: f64,
    d2: f64,
    max_links: Option<usize>,
) -> EconResult<(Vec<Vec<usize>>, QueryStats)> {
    let dims = metric.dims();
    let n = coords.len();
    // Box pruning thresholds, in the units of the tree coordinates: prune a box whose lower
    // bound exceeds `prune_above`; skip a box whose upper bound is at most `skip_below`.
    let (prune_above, skip_below) = match metric {
        // Planar bounds are exact lower/upper bounds of every computed distance.
        Metric::Euclidean => (d2, d1),
        Metric::GreatCircle => {
            // Chord lengths between unit vectors, widened so that the rounding differences
            // between the chord geometry and the haversine formula can never drop a pair.
            // Longitude differences lose absolute precision with the longitude's magnitude.
            let max_lon_rad = coords
                .iter()
                .map(|c| c.0.to_radians().abs())
                .fold(0.0, f64::max);
            let margin = 1e-12 + 4.0 * f64::EPSILON * max_lon_rad;
            let half_angle =
                |d: f64| (d / (2.0 * EARTH_RADIUS_KM)).min(std::f64::consts::FRAC_PI_2);
            let hi = if d2 / EARTH_RADIUS_KM >= std::f64::consts::PI {
                f64::INFINITY
            } else {
                2.0 * half_angle(d2).sin() * (1.0 + 1e-9) + margin
            };
            let lo = 2.0 * half_angle(d1).sin() * (1.0 - 1e-9) - margin;
            (hi, if lo > 0.0 { lo } else { f64::NEG_INFINITY })
        }
    };

    let tree = KdTree::build(coords, metric);
    let mut counter = EvalCounter::new()?;
    let mut lists = Vec::with_capacity(n);
    let mut total_links = 0usize;
    let mut stack: Vec<usize> = Vec::new();

    for i in 0..n {
        let q = tree.points[i];
        let mut row = Vec::new();
        stack.clear();
        if !tree.nodes.is_empty() {
            stack.push(0);
        }
        while let Some(id) = stack.pop() {
            let node = &tree.nodes[id];
            // A box whose points all have the query's coordinates holds only distances of
            // exactly 0, never above d1 >= 0. On the sphere `skip_below` cannot rule it out
            // when d1 is 0 or tiny, so without this check repeated coordinates cost n²
            // evaluations.
            if KdTree::lower_bound(&q, node, dims) > prune_above
                || KdTree::upper_bound(&q, node, dims) <= skip_below
                || (node.coincident && coords[tree.order[node.start]] == coords[i])
            {
                continue;
            }
            match node.children {
                Some((left, right)) => {
                    stack.push(left);
                    stack.push(right);
                }
                None => {
                    let mut evals = 0u64;
                    for &j in &tree.order[node.start..node.end] {
                        if j == i {
                            continue;
                        }
                        evals += 1;
                        let dist = metric.distance(coords[i], coords[j]);
                        if dist > d1 && dist <= d2 {
                            if let Some(cap) = max_links
                                && total_links >= cap
                            {
                                return Err(EconError::ResourceLimit(format!(
                                    "the distance band links more than {cap} pairs of observations"
                                )));
                            }
                            total_links += 1;
                            row.push(j);
                        }
                    }
                    counter.add(evals)?;
                }
            }
        }
        row.sort_unstable();
        lists.push(row);
    }
    Ok((lists, counter.stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::{EconError, EconResult};
    use crate::spatial::test_fixtures::coords;

    // R 4.3.3, dbscan 1.2.6. spdep::knearneigh(use_kd_tree = TRUE) and dnearneigh() delegate to these:
    //   coords <- cbind(cx, cy)                                   # fixture
    //   t(apply(dbscan::kNN(coords, k = 3)$id, 1, sort)) - 1      # == brute force via order(D[i, ])
    //   lapply(dbscan::frNN(coords, eps = 2.0)$id, sort)          # == which(D[i, ] > 0 & D[i, ] <= 2)
    //   D <- as.matrix(dist(coords)); which(D[i, ] > 1 & D[i, ] <= 2.5)   # band (1, 2.5]
    // Smallest gap between pairwise distances 1.1e-5; nearest distance to 1.0 / 2.0 / 2.5 is 0.016 / 0.018 / 0.004.
    #[test]
    fn knn_matches_r_dbscan_reference() {
        let nb = Neighbors::try_from_knn(&coords(), 3).unwrap();
        let expected: Vec<Vec<usize>> = vec![
            vec![7, 11, 25],
            vec![14, 15, 28],
            vec![10, 16, 26],
            vec![12, 18, 23],
            vec![10, 19, 26],
            vec![18, 21, 27],
            vec![7, 9, 25],
            vec![6, 9, 25],
            vec![13, 20, 24],
            vec![6, 7, 17],
            vec![2, 4, 26],
            vec![0, 7, 25],
            vec![15, 23, 28],
            vec![3, 22, 24],
            vec![12, 15, 28],
            vec![12, 14, 28],
            vec![2, 8, 10],
            vec![6, 7, 9],
            vec![5, 21, 27],
            vec![4, 22, 26],
            vec![8, 16, 24],
            vec![5, 18, 27],
            vec![3, 13, 24],
            vec![3, 12, 28],
            vec![8, 13, 22],
            vec![6, 7, 11],
            vec![2, 10, 19],
            vec![5, 18, 21],
            vec![12, 14, 15],
            vec![5, 21, 27],
        ];
        assert_eq!(nb.as_slice(), expected.as_slice());
        assert!(!nb.is_symmetric());
    }

    #[test]
    fn distance_band_matches_r_frnn_reference() {
        let nb = Neighbors::try_from_distance(&coords(), 0.0, 2.0, None).unwrap();
        assert_eq!(nb.n_links(), 102);
        assert_eq!(nb.isolates(), vec![0, 16, 29]);
        assert_eq!(nb.get_neighbors(3), &[12, 13, 18, 22, 23]);
        assert_eq!(nb.get_neighbors(12), &[3, 14, 15, 18, 23, 28]);
        let band = Neighbors::try_from_distance(&coords(), 1.0, 2.5, None).unwrap();
        assert_eq!(band.n_links(), 116);
        assert_eq!(band.isolates(), vec![29]);
        assert_eq!(band.get_neighbors(10), &[4, 6, 9, 17, 19, 26]);
        assert_eq!(band.get_neighbors(23), &[3, 14, 15, 18, 21, 27, 28]);
    }

    #[test]
    fn tree_constructors_match_brute_force() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let mut c: Vec<(f64, f64)> = (0..1500)
            .map(|_| (rng.gen_range(0.0..100.0), rng.gen_range(0.0..100.0)))
            .collect();
        // exact ties
        c.extend((0..15).flat_map(|i| (0..15).map(move |j| (i as f64, j as f64))));
        // zero distances
        let dups = c[..100].to_vec();
        c.extend(dups);
        for k in [1, 4, 10] {
            assert_eq!(
                Neighbors::try_from_knn(&c, k).unwrap().as_slice(),
                knn_brute_force(&c, k).as_slice(),
                "k={k}"
            );
        }
        for (d1, d2) in [(0.0, 0.5), (0.0, 3.0), (1.0, 3.0), (0.0, 20.0)] {
            assert_eq!(
                Neighbors::try_from_distance(&c, d1, d2, None)
                    .unwrap()
                    .as_slice(),
                distance_brute_force(&c, d1, d2, Metric::Euclidean).as_slice(),
                "({d1}, {d2}]"
            );
        }
    }

    #[test]
    fn longlat_band_matches_brute_force_across_dateline_and_poles() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        let mut c: Vec<(f64, f64)> = (0..800)
            .map(|_| (rng.gen_range(-180.0..180.0), rng.gen_range(-89.9..89.9)))
            .collect();
        c.extend([(179.9, 10.0), (-179.9, 10.0), (0.0, 89.99), (180.0, 89.99)]);
        for (d1, d2) in [(0.0, 50.0), (0.0, 500.0), (100.0, 2_000.0), (0.0, 25_000.0)] {
            assert_eq!(
                Neighbors::try_from_distance_longlat(&c, d1, d2, None)
                    .unwrap()
                    .as_slice(),
                distance_brute_force(&c, d1, d2, Metric::GreatCircle).as_slice(),
                "({d1}, {d2}]"
            );
        }
        let nb = Neighbors::try_from_distance_longlat(&c, 0.0, 50.0, None).unwrap();
        assert!(nb.are_neighbors(800, 801) && nb.are_neighbors(802, 803));
    }

    #[test]
    fn distance_band_stops_at_the_link_cap() {
        let c = coords();
        let err = Neighbors::try_from_distance(&c, 0.0, 1e9, Some(500)).unwrap_err();
        assert!(
            matches!(&err, EconError::ResourceLimit(m) if m.contains("500")),
            "{err}"
        );
        // 30 * 29 links
        assert_eq!(
            Neighbors::try_from_distance(&c, 0.0, 1e9, Some(870))
                .unwrap()
                .n_links(),
            870
        );
        assert!(Neighbors::try_from_distance(&c, 0.0, 1e9, Some(869)).is_err());
        let ll: Vec<(f64, f64)> = c.iter().map(|&(x, y)| (x * 10.0, y * 5.0)).collect();
        assert!(matches!(
            Neighbors::try_from_distance_longlat(&ll, 0.0, 30_000.0, Some(100)),
            Err(EconError::ResourceLimit(_))
        ));
    }

    #[test]
    fn invalid_parameters_are_rejected() {
        let c = coords();
        let invalid =
            |r: EconResult<Neighbors>| matches!(r, Err(EconError::InvalidSpecification { .. }));
        for k in [0, 30, usize::MAX] {
            assert!(invalid(Neighbors::try_from_knn(&c, k)), "k={k}");
        }
        assert!(invalid(Neighbors::try_from_knn(&c[..1], 1)));
        let mut bad = c.clone();
        bad[4].0 = f64::NAN;
        assert!(invalid(Neighbors::try_from_knn(&bad, 3)));
        assert!(invalid(Neighbors::try_from_distance(&bad, 0.0, 1.0, None)));
        bad[4] = (1.0, f64::INFINITY);
        assert!(invalid(Neighbors::try_from_knn(&bad, 3)));
        for (d1, d2) in [
            (-1.0, 1.0),
            (0.0, f64::NAN),
            (0.0, f64::INFINITY),
            (2.0, 1.0),
            (f64::NAN, 1.0),
        ] {
            assert!(
                invalid(Neighbors::try_from_distance(&c, d1, d2, None)),
                "({d1}, {d2})"
            );
        }
        assert!(invalid(Neighbors::try_from_distance_longlat(
            &[(0.0, 95.0), (1.0, 1.0)],
            0.0,
            100.0,
            None
        )));
    }

    #[test]
    fn legacy_constructors_keep_their_semantics() {
        let c = coords();
        assert_eq!(
            Neighbors::from_knn(&c, 3).as_slice(),
            Neighbors::try_from_knn(&c, 3).unwrap().as_slice()
        );
        // k clamped to n - 1
        assert_eq!(Neighbors::from_knn(&c, 100).n_links(), 870);
        assert_eq!(Neighbors::from_knn(&c, 0).n_links(), 0);
        // used to underflow n - 1
        assert_eq!(Neighbors::from_knn(&[], 3).n(), 0);
        let mut bad = c.clone();
        bad[4] = (f64::NAN, 1.0);
        assert_eq!(
            Neighbors::from_knn(&bad, 3).as_slice(),
            knn_brute_force(&bad, 3).as_slice()
        );
        assert_eq!(
            Neighbors::from_distance(&c, 0.0, f64::INFINITY).n_links(),
            870
        );
        assert_eq!(
            Neighbors::from_distance(&c, 0.0, 2.0).as_slice(),
            Neighbors::try_from_distance(&c, 0.0, 2.0, None)
                .unwrap()
                .as_slice()
        );
    }

    #[test]
    fn construction_honours_cancellation_but_legacy_ignores_it() {
        use crate::resource::{Budget, CancelToken, with_budget};
        let t = CancelToken::new();
        t.cancel();
        let budget = || Budget::new(None, t.clone());
        assert!(matches!(
            with_budget(budget(), || Neighbors::try_from_knn(&coords(), 3)),
            Err(EconError::Cancelled(_))
        ));
        assert!(matches!(
            with_budget(budget(), || Neighbors::try_from_distance(
                &coords(),
                0.0,
                2.0,
                None
            )),
            Err(EconError::Cancelled(_))
        ));
        assert_eq!(
            with_budget(budget(), || Neighbors::from_knn(&coords(), 3)).n_links(),
            90
        );
    }

    #[test]
    fn queries_do_sub_quadratic_work() {
        // Deterministic work count instead of a timing assertion: brute force would make
        // n * (n - 1) evaluations.
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);
        let n = 20_000;
        let c: Vec<(f64, f64)> = (0..n)
            .map(|_| (rng.gen_range(0.0..100.0), rng.gen_range(0.0..100.0)))
            .collect();
        let (_, stats) = knn_lists(&c, 5).unwrap();
        assert!(stats.dist_evals <= 200 * n as u64, "{}", stats.dist_evals);
        let (lists, stats) = band_lists(&c, Metric::Euclidean, 0.0, 0.5, None).unwrap();
        let links: usize = lists.iter().map(Vec::len).sum();
        assert!(
            stats.dist_evals <= 200 * n as u64 + 2 * links as u64,
            "{}",
            stats.dist_evals
        );
    }

    #[test]
    fn knn_on_duplicated_coordinates_matches_brute_force() {
        use rand::seq::SliceRandom;
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(11);
        let mut c: Vec<(f64, f64)> = (0..300)
            .map(|_| (rng.gen_range(0.0..100.0), rng.gen_range(0.0..100.0)))
            .collect();
        // exact ties
        c.extend((0..15).flat_map(|i| (0..15).map(move |j| (i as f64, j as f64))));
        // zero distances
        let dups = c[..100].to_vec();
        c.extend(dups);
        // 2,000 observations at one grid point, at scattered indices
        c.extend(std::iter::repeat_n((7.0, 7.0), 2_000));
        c.shuffle(&mut rng);
        let n = c.len();
        for k in [1, 5] {
            assert_eq!(
                Neighbors::try_from_knn(&c, k).unwrap().as_slice(),
                knn_brute_force(&c, k).as_slice(),
                "k={k}"
            );
            // all-pairs work would be n * (n - 1)
            let (_, stats) = knn_lists(&c, k).unwrap();
            assert!(
                stats.dist_evals <= 200 * n as u64,
                "k={k}: {}",
                stats.dist_evals
            );
        }
    }

    #[test]
    fn knn_on_one_repeated_location_does_linear_work() {
        // Deterministic work count: every box has lower bound 0 here, so only the index
        // order can rule a box out.
        let n = 20_000;
        let c = vec![(3.5, -1.25); n];
        let (lists, stats) = knn_lists(&c, 5).unwrap();
        assert!(stats.dist_evals <= 200 * n as u64, "{}", stats.dist_evals);
        // every distance ties, so each row holds the five lowest other indices
        for (i, row) in lists.iter().enumerate() {
            let expected: Vec<usize> = (0..6).filter(|&j| j != i).take(5).collect();
            assert_eq!(row, &expected, "row {i}");
        }
        let small = vec![(3.5, -1.25); 300];
        assert_eq!(knn_lists(&small, 5).unwrap().0, knn_brute_force(&small, 5));
    }

    #[test]
    fn band_on_duplicated_coordinates_matches_brute_force() {
        use rand::seq::SliceRandom;
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(13);
        let mut c: Vec<(f64, f64)> = (0..800)
            .map(|_| (rng.gen_range(-180.0..180.0), rng.gen_range(-89.9..89.9)))
            .collect();
        // 2,000 observations at one location, at scattered indices
        c.extend(std::iter::repeat_n((8.55, 47.37), 2_000));
        c.shuffle(&mut rng);
        let n = c.len() as u64;
        for (d1, d2) in [(0.0, 500.0), (0.0, 3_000.0), (100.0, 2_000.0)] {
            assert_eq!(
                Neighbors::try_from_distance_longlat(&c, d1, d2, None)
                    .unwrap()
                    .as_slice(),
                distance_brute_force(&c, d1, d2, Metric::GreatCircle).as_slice(),
                "({d1}, {d2}]"
            );
            // all-pairs work would be n * (n - 1)
            let (lists, stats) = band_lists(&c, Metric::GreatCircle, d1, d2, None).unwrap();
            let links: u64 = lists.iter().map(|row| row.len() as u64).sum();
            assert!(
                stats.dist_evals <= 200 * n + 2 * links,
                "({d1}, {d2}]: {} evaluations, {links} links",
                stats.dist_evals
            );
        }
    }

    #[test]
    fn band_on_one_repeated_location_does_linear_work() {
        // Deterministic work count. With d1 = 0 the sphere has no lower chord threshold, so
        // only the coordinates themselves rule out a box of observations at the query.
        let n = 20_000;
        let c = vec![(8.55, 47.37); n];
        for metric in [Metric::GreatCircle, Metric::Euclidean] {
            let (lists, stats) = band_lists(&c, metric, 0.0, 50.0, None).unwrap();
            // distance 0 is not above d1 = 0
            assert!(lists.iter().all(Vec::is_empty), "{metric:?}");
            assert!(
                stats.dist_evals <= 200 * n as u64,
                "{metric:?}: {}",
                stats.dist_evals
            );
        }
    }

    #[test]
    fn test_knn_neighbors() {
        // Simple 2x2 grid
        let coords = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
        let nb = Neighbors::from_knn(&coords, 2);

        assert_eq!(nb.n(), 4);
        assert_eq!(nb.cardinality(0), 2);

        // For point (0,0), nearest neighbors should be (1,0) and (0,1)
        let n0 = nb.get_neighbors(0);
        assert!(n0.contains(&1) && n0.contains(&2));
    }

    #[test]
    fn test_distance_neighbors() {
        let coords = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (2.0, 0.0)];
        let nb = Neighbors::from_distance(&coords, 0.0, 1.1);

        // Point 0 should be neighbors with 1 and 2 (distance 1.0), but not 3 (distance 2.0)
        assert!(nb.are_neighbors(0, 1));
        assert!(nb.are_neighbors(0, 2));
        assert!(!nb.are_neighbors(0, 3));

        // Distance-based should be symmetric
        assert!(nb.is_symmetric());
    }

    #[test]
    fn test_explicit_neighbors() {
        let nb = Neighbors::from_indices(vec![vec![1, 2], vec![0, 2], vec![0, 1]]);

        assert_eq!(nb.n(), 3);
        assert!(nb.is_symmetric());
        assert!(nb.are_neighbors(0, 1));
        assert!(nb.are_neighbors(1, 0));
    }

    #[test]
    fn test_make_symmetric() {
        let mut nb = Neighbors::from_indices(vec![
            vec![1],    // 0 -> 1
            vec![],     // 1 has no explicit neighbors
            vec![0, 1], // 2 -> 0, 1
        ]);

        assert!(!nb.is_symmetric());

        nb.make_symmetric();

        assert!(nb.is_symmetric());
        assert!(nb.are_neighbors(1, 0)); // Now 1 should have 0 as neighbor
        assert!(nb.are_neighbors(0, 2)); // And 0 should have 2
        assert!(nb.are_neighbors(1, 2)); // And 1 should have 2
    }

    #[test]
    fn test_isolates() {
        let nb = Neighbors::from_indices(vec![
            vec![1],
            vec![0],
            vec![], // Isolate
        ]);

        assert!(nb.has_isolates());
        assert_eq!(nb.isolates(), vec![2]);
    }

    #[test]
    fn test_haversine() {
        // New York to Los Angeles (approximately)
        let ny = (-74.0060, 40.7128);
        let la = (-118.2437, 34.0522);
        let dist = haversine_distance(ny, la);

        // Should be approximately 3940 km
        assert!((dist - 3940.0).abs() < 50.0);
    }
}
