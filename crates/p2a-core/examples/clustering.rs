//! Clustering Example
//!
//! Demonstrates machine learning clustering with:
//! - K-means clustering
//! - DBSCAN density-based clustering
//! - Cluster validation

use p2a_core::ml::{kmeans, dbscan, KMeansResult, DBSCANResult};
use ndarray::Array2;
use rand::prelude::*;
use rand_distr::Normal;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Clustering Example ===\n");

    // Generate synthetic data with 3 clusters
    let data = generate_cluster_data(150, 2, 3);
    println!("Generated {} data points in 2D with 3 true clusters\n", data.nrows());

    // K-means clustering
    println!("--- K-means Clustering (k=3) ---");
    let kmeans_result = kmeans(data.view(), 3, Some(100), Some(1e-6), Some(10), Some(42))?;
    print_kmeans_results(&kmeans_result);

    // Try different k values
    println!("\n--- K-means with Different k Values ---");
    for k in 2..=5 {
        let result = kmeans(data.view(), k, Some(100), Some(1e-6), Some(10), Some(42))?;
        println!("k={}: inertia={:.2}, clusters={}", k, result.inertia, result.cluster_sizes.len());
    }

    // DBSCAN clustering
    println!("\n--- DBSCAN Clustering ---");
    let eps = 1.5;
    let min_samples = 5;
    println!("Parameters: eps={}, min_samples={}", eps, min_samples);

    let dbscan_result = dbscan(data.view(), eps, min_samples)?;
    print_dbscan_results(&dbscan_result);

    // Compare clustering results
    println!("\n--- Comparison ---");
    println!("{:<20} {:>10} {:>12}", "Method", "Clusters", "Inertia");
    println!("{:-<20} {:-<10} {:-<12}", "", "", "");
    println!("{:<20} {:>10} {:>12.2}", "K-means", kmeans_result.cluster_sizes.len(), kmeans_result.inertia);
    println!("{:<20} {:>10} {:>12}", "DBSCAN", dbscan_result.n_clusters, "-");

    // Cluster sizes
    println!("\n--- Cluster Sizes ---");
    println!("K-means clusters:");
    for (i, &size) in kmeans_result.cluster_sizes.iter().enumerate() {
        println!("  Cluster {}: {} points", i, size);
    }

    println!("\nDBSCAN clusters:");
    if dbscan_result.n_noise > 0 {
        println!("  Noise points: {}", dbscan_result.n_noise);
    }
    let dbscan_sizes = cluster_sizes_i32(&dbscan_result.labels, dbscan_result.n_clusters);
    for (i, size) in dbscan_sizes.iter().enumerate() {
        println!("  Cluster {}: {} points", i, size);
    }

    Ok(())
}

/// Generate synthetic cluster data
fn generate_cluster_data(n: usize, dims: usize, n_clusters: usize) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Cluster centers spread apart
    let centers: Vec<Vec<f64>> = (0..n_clusters)
        .map(|i| {
            (0..dims)
                .map(|d| (i as f64) * 5.0 + if d == 0 { 0.0 } else { (i as f64) * 3.0 })
                .collect()
        })
        .collect();

    let mut data = Array2::zeros((n, dims));
    for i in 0..n {
        let cluster = i % n_clusters;
        for j in 0..dims {
            data[[i, j]] = centers[cluster][j] + rng.sample(normal);
        }
    }

    // Shuffle the data
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);

    let mut shuffled = Array2::zeros((n, dims));
    for (new_i, &old_i) in indices.iter().enumerate() {
        shuffled.row_mut(new_i).assign(&data.row(old_i));
    }

    shuffled
}

fn print_kmeans_results(result: &KMeansResult) {
    println!("Clusters found: {}", result.cluster_sizes.len());
    println!("Inertia (within-cluster sum of squares): {:.2}", result.inertia);
    println!("Iterations: {}", result.n_iterations);

    println!("\nCluster centers:");
    for (i, center) in result.centroids.outer_iter().enumerate() {
        let coords: Vec<String> = center.iter().map(|x| format!("{:.2}", x)).collect();
        println!("  Cluster {}: [{}]", i, coords.join(", "));
    }
}

fn print_dbscan_results(result: &DBSCANResult) {
    println!("Clusters found: {}", result.n_clusters);
    println!("Noise points: {}", result.n_noise);
    println!("Core samples: {}", result.core_sample_indices.len());
}

fn cluster_sizes_i32(labels: &[i32], n_clusters: usize) -> Vec<usize> {
    let mut sizes = vec![0; n_clusters];
    for &label in labels {
        if label >= 0 && (label as usize) < n_clusters {
            sizes[label as usize] += 1;
        }
    }
    sizes
}
