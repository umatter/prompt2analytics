use std::time::Instant;
use ndarray::Array2;
use rand::prelude::*;
use rand_distr::Normal;

fn generate_cluster_data(n: usize, k: usize, n_clusters: usize) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, 0.5).unwrap();
    let mut data = Array2::zeros((n, k));
    for i in 0..n {
        let cluster = i % n_clusters;
        let center = cluster as f64 * 5.0;
        for j in 0..k {
            data[[i, j]] = center + rng.sample(normal);
        }
    }
    data
}

fn main() {
    use p2a_core::ml::*;
    
    println!("BATCH 2 PERFORMANCE BENCHMARKS (Rust)");
    println!("======================================\n");
    
    for n in [100, 500, 1000] {
        let data = generate_cluster_data(n, 5, 3);
        let labels: Vec<usize> = (0..n).map(|i| i % 3).collect();
        
        // cluster_stats
        let start = Instant::now();
        for _ in 0..10 {
            let _ = cluster_stats(data.view(), &labels);
        }
        let elapsed = start.elapsed().as_secs_f64() * 100.0; // ms per iter
        println!("cluster_stats n={}: {:.3} ms", n, elapsed);
        
        // clara
        let start = Instant::now();
        for _ in 0..10 {
            let _ = clara(data.view(), 3, Some(5), None, Some(100), Some(42));
        }
        let elapsed = start.elapsed().as_secs_f64() * 100.0;
        println!("clara n={}: {:.3} ms", n, elapsed);
        
        // fanny
        let start = Instant::now();
        for _ in 0..10 {
            let _ = fanny(data.view(), 3, Some(2.0), Some(100), None, Some(42));
        }
        let elapsed = start.elapsed().as_secs_f64() * 100.0;
        println!("fanny n={}: {:.3} ms", n, elapsed);
        
        println!();
    }
    
    // flexmix and pvclust with smaller datasets
    println!("Complex methods (smaller datasets):");
    for n in [100, 200] {
        let data = generate_cluster_data(n, 5, 3);
        
        // Create X (with intercept) and y for flexmix
        let x = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 { 1.0 } else { data[[i, 0]] }
        });
        let y = Array2::from_shape_fn((n, 1), |(i, _)| {
            data[[i, 0]] * 2.0 + 1.0 + (i % 3) as f64 * 0.1
        });
        
        // flexmix
        let start = Instant::now();
        for _ in 0..5 {
            let _ = flexmix(y.view(), x.view(), 2, Some(50), Some(1e-4), Some(42));
        }
        let elapsed = start.elapsed().as_secs_f64() * 200.0;
        println!("flexmix n={}: {:.3} ms", n, elapsed);
        
        // pvclust (expensive due to bootstrap)
        let start = Instant::now();
        for _ in 0..3 {
            let _ = pvclust(data.view(), Some("average"), Some(50), None, Some(0.95), Some(42));
        }
        let elapsed = start.elapsed().as_secs_f64() * 333.3;
        println!("pvclust n={}: {:.3} ms", n, elapsed);
    }
}
