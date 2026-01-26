use std::time::Instant;
use ndarray::Array2;
use rand::prelude::*;
use rand_distr::Normal;

fn generate_cluster_data(n: usize, d: usize, n_clusters: usize) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, 0.5).unwrap();  // Match R's sd=0.5
    
    let mut data = Array2::zeros((n, d));
    for i in 0..n {
        let cluster = i % n_clusters;
        let center = cluster as f64 * 5.0;  // Match R's separation=5.0
        for j in 0..d {
            data[[i, j]] = center + rng.sample(normal);
        }
    }
    data
}

fn main() {
    use p2a_core::ml::hdbscan;
    
    println!("HDBSCAN: Rust vs R Comparison");
    println!("==============================");
    println!("Using minPts=5 to match R benchmark");
    println!();
    println!("{:>6} {:>12} {:>12} {:>10}", "n", "Rust (ms)", "R (ms)", "Speedup");
    println!("{:-<6} {:-<12} {:-<12} {:-<10}", "", "", "", "");
    
    // R benchmark results (median)
    let r_times = [(100, 0.41), (500, 4.45), (1000, 15.29), (2000, 60.0), (3000, 135.0), (5000, 375.0)];
    
    for (n, r_time) in r_times {
        let data = generate_cluster_data(n, 5, 3);
        
        // Warmup
        let _ = hdbscan(data.view(), Some(5), Some(5));
        
        // Benchmark
        let start = Instant::now();
        let iterations = if n <= 1000 { 20 } else { 5 };
        for _ in 0..iterations {
            let _ = hdbscan(data.view(), Some(5), Some(5));
        }
        let rust_time = start.elapsed().as_secs_f64() / iterations as f64 * 1000.0;
        
        let speedup = r_time / rust_time;
        println!("{:>6} {:>12.2} {:>12.2} {:>9.1}x", n, rust_time, r_time, speedup);
    }
}
