use ndarray::Array2;
use rand::prelude::*;
use rand_distr::Normal;
use std::time::Instant;

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

fn generate_univariate_mixture(n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(42);
    let normal1 = Normal::new(0.0, 1.0).unwrap();
    let normal2 = Normal::new(10.0, 1.0).unwrap();

    (0..n)
        .map(|i| {
            if i % 2 == 0 {
                rng.sample(normal1)
            } else {
                rng.sample(normal2)
            }
        })
        .collect()
}

fn main() {
    use p2a_core::ml::*;

    println!("BATCH 3 PERFORMANCE BENCHMARKS (Rust)");
    println!("======================================\n");

    // Test sizes
    let sizes = [100, 500, 1000];

    // skmeans benchmarks
    println!("=== skmeans (Spherical K-Means) ===");
    for n in sizes {
        let data = generate_cluster_data(n, 10, 3);
        let start = Instant::now();
        let iters = if n <= 500 { 10 } else { 5 };
        for _ in 0..iters {
            let _ = skmeans(data.view(), 3, Some(100), None, Some(5), Some(42));
        }
        let elapsed = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!("skmeans n={}: {:.3} ms", n, elapsed);
    }
    println!();

    // fastcluster benchmarks
    println!("=== fastcluster (Fast Hierarchical) ===");
    for n in sizes {
        let data = generate_cluster_data(n, 5, 3);
        let start = Instant::now();
        let iters = if n <= 500 { 10 } else { 3 };
        for _ in 0..iters {
            let _ = fastcluster(data.view(), Some(FastLinkage::Ward), None);
        }
        let elapsed = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!("fastcluster n={}: {:.3} ms", n, elapsed);
    }
    println!();

    // dynamicTreeCut benchmarks
    println!("=== dynamicTreeCut ===");
    for n in sizes {
        let data = generate_cluster_data(n, 5, 3);
        let start = Instant::now();
        let iters = if n <= 500 { 10 } else { 3 };
        for _ in 0..iters {
            let _ =
                run_dynamic_tree_cut(data.view(), Some(FastLinkage::Ward), None, Some(2), Some(2));
        }
        let elapsed = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!("dynamicTreeCut n={}: {:.3} ms", n, elapsed);
    }
    println!();

    // normal_mix_em benchmarks (univariate)
    println!("=== normal_mix_em (Univariate Mixture) ===");
    for n in sizes {
        let data = generate_univariate_mixture(n);
        let start = Instant::now();
        let iters = 10;
        for _ in 0..iters {
            let _ = normal_mix_em(&data, 2, Some(100), Some(1e-6), Some(42));
        }
        let elapsed = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!("normal_mix_em n={}: {:.3} ms", n, elapsed);
    }
    println!();

    // mvnorm_mix_em benchmarks (multivariate)
    println!("=== mvnorm_mix_em (Multivariate Mixture) ===");
    for n in sizes {
        let data = generate_cluster_data(n, 5, 2);
        let start = Instant::now();
        let iters = 10;
        for _ in 0..iters {
            let _ = mvnorm_mix_em(data.view(), 2, Some(100), Some(1e-6), Some(42));
        }
        let elapsed = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!("mvnorm_mix_em n={}: {:.3} ms", n, elapsed);
    }
    println!();

    // kprototypes benchmarks
    println!("=== kprototypes (Mixed Data) ===");
    for n in sizes {
        let numeric_data = generate_cluster_data(n, 5, 3);
        let categorical_data: Vec<Vec<usize>> = (0..n).map(|i| vec![i % 3, i % 5]).collect();

        let start = Instant::now();
        let iters = if n <= 500 { 10 } else { 5 };
        for _ in 0..iters {
            let _ = kprototypes(
                numeric_data.view(),
                &categorical_data,
                3,
                None,
                Some(50),
                Some(5),
                Some(42),
            );
        }
        let elapsed = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!("kprototypes n={}: {:.3} ms", n, elapsed);
    }

    println!("\n=== Summary ===");
    println!("All benchmarks completed. Times are in milliseconds per iteration.");
}
