//! Spatial neighbors definitions.
//!
//! Provides structures and methods for defining which observations are neighbors
//! of each other, based on distance, k-nearest neighbors, or explicit specification.

use serde::{Deserialize, Serialize};

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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neighbors {
    /// For each observation i, sorted vector of neighbor indices (0-indexed)
    /// Empty vector means no neighbors
    neighbors: Vec<Vec<usize>>,
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
            neighbors: sorted_neighbors,
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
    pub fn from_knn(coords: &[(f64, f64)], k: usize) -> Self {
        let n = coords.len();
        let k = k.min(n - 1); // Can't have more neighbors than n-1

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
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let mut knn: Vec<usize> = distances.iter().take(k).map(|(j, _)| *j).collect();
            knn.sort_unstable();

            neighbors.push(knn);
        }

        let symmetric = Self::check_symmetry(&neighbors);

        Neighbors {
            neighbors,
            n,
            region_ids: None,
            symmetric,
            method: NeighborMethod::Knn { k },
        }
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
    pub fn from_distance(coords: &[(f64, f64)], d1: f64, d2: f64) -> Self {
        let n = coords.len();
        let mut neighbors = Vec::with_capacity(n);

        for i in 0..n {
            let mut nb = Vec::new();
            for j in 0..n {
                if i != j {
                    let dx = coords[i].0 - coords[j].0;
                    let dy = coords[i].1 - coords[j].1;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist > d1 && dist <= d2 {
                        nb.push(j);
                    }
                }
            }
            nb.sort_unstable();
            neighbors.push(nb);
        }

        // Distance-based neighbors are symmetric by construction
        Neighbors {
            neighbors,
            n,
            region_ids: None,
            symmetric: true,
            method: NeighborMethod::Distance { d1, d2 },
        }
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
    pub fn from_distance_longlat(coords: &[(f64, f64)], d1: f64, d2: f64) -> Self {
        let n = coords.len();
        let mut neighbors = Vec::with_capacity(n);

        for i in 0..n {
            let mut nb = Vec::new();
            for j in 0..n {
                if i != j {
                    let dist = haversine_distance(coords[i], coords[j]);
                    if dist > d1 && dist <= d2 {
                        nb.push(j);
                    }
                }
            }
            nb.sort_unstable();
            neighbors.push(nb);
        }

        Neighbors {
            neighbors,
            n,
            region_ids: None,
            symmetric: true,
            method: NeighborMethod::Distance { d1, d2 },
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
            neighbors,
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
    /// If i is a neighbor of j, makes j a neighbor of i.
    pub fn make_symmetric(&mut self) {
        if self.symmetric {
            return;
        }

        let mut new_neighbors = self.neighbors.clone();

        for i in 0..self.n {
            for &j in &self.neighbors[i] {
                // Add i as neighbor of j if not already
                if !new_neighbors[j].contains(&i) {
                    new_neighbors[j].push(i);
                }
            }
        }

        // Sort all neighbor lists
        for nb in &mut new_neighbors {
            nb.sort_unstable();
        }

        self.neighbors = new_neighbors;
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
        &self.neighbors
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
    const EARTH_RADIUS_KM: f64 = 6371.0;

    let (lon1, lat1) = (p1.0.to_radians(), p1.1.to_radians());
    let (lon2, lat2) = (p2.0.to_radians(), p2.1.to_radians());

    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;

    let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();

    EARTH_RADIUS_KM * c
}

#[cfg(test)]
mod tests {
    use super::*;

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
