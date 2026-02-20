//! GPU vs CPU benchmarks for core linear algebra operations.
//!
//! Measures speedup from GPU acceleration across different matrix sizes
//! on the NVIDIA DGX Spark (Grace Blackwell, unified memory).
//!
//! Run with:
//! ```bash
//! cargo bench -p p2a-core --bench gpu_benchmarks --features cuda
//! ```

#[path = "bench_utils.rs"]
mod bench_utils;

use bench_utils::{BenchConfig, BenchmarkResult, print_header, print_result, save_results};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

fn generate_matrix(n: usize, k: usize, rng: &mut ChaCha8Rng) -> Array2<f64> {
    let normal = Normal::new(0.0, 1.0).unwrap();
    Array2::from_shape_fn((n, k), |_| normal.sample(rng))
}

fn generate_vector(n: usize, rng: &mut ChaCha8Rng) -> Array1<f64> {
    let normal = Normal::new(0.0, 1.0).unwrap();
    Array1::from_shape_fn(n, |_| normal.sample(rng))
}

/// Config for large benchmarks: fewer iterations to keep total time reasonable.
fn large_config() -> BenchConfig {
    BenchConfig {
        warmup_iterations: 3,
        measurement_iterations: 20,
        capture_raw_times: true,
    }
}

/// Config for standard benchmarks.
fn std_config() -> BenchConfig {
    BenchConfig {
        warmup_iterations: 5,
        measurement_iterations: 50,
        capture_raw_times: true,
    }
}

/// Run the full benchmark suite.
fn main() {
    let mut results: Vec<BenchmarkResult> = Vec::new();

    println!("=== GPU vs CPU Benchmarks (DGX Spark) ===");
    println!();

    // Report GPU status and thresholds
    #[cfg(feature = "cuda")]
    {
        if let Some(ctx) = p2a_core::linalg::gpu::GpuContext::get() {
            println!("GPU: ENABLED (cuBLAS + cuSOLVER active)");
            println!("Thresholds:");
            println!("  xtx    n*k²  >= {}, k >= {}", ctx.thresholds.xtx_min_nkk, ctx.thresholds.xtx_min_k);
            println!("  xty    n     >= {} ({})", ctx.thresholds.xty_min_n,
                if ctx.thresholds.xty_min_n == usize::MAX { "disabled" } else { "active" });
            println!("  inv    k     >= {}", ctx.thresholds.inverse_min_k);
            println!("  matmul mnk   >= {}, shape >= {:.2}", ctx.thresholds.matmul_min_mnk, ctx.thresholds.matmul_min_shape_ratio);
            println!("  kmeans n     >= {}, d >= {}", ctx.thresholds.kmeans_min_n, ctx.thresholds.kmeans_min_d);
        } else {
            println!("GPU: NOT AVAILABLE (CUDA init failed, CPU-only run)");
        }
    }
    #[cfg(not(feature = "cuda"))]
    println!("GPU: DISABLED (cuda feature not compiled)");

    println!();

    let config = std_config();
    let lg_config = large_config();

    // ===================================================================
    // xtx (X'X) — test both small k (CPU wins) and large k (GPU wins)
    // ===================================================================
    println!("--- xtx (X'X) ---");
    print_header();

    // Small k (k=10): GPU dispatch threshold n*k²=n*100, needs n >= 200K for 20M
    // These should stay on CPU with new thresholds.
    let xtx_sizes: Vec<(usize, usize, bool)> = vec![
        // (n, k, use_large_config)
        (1_000, 10, false),
        (10_000, 10, false),
        (100_000, 10, false),
        (500_000, 10, false),
        (1_000_000, 10, true),
        // Medium k (k=50): n*k²=n*2500, needs n >= 8K for 20M → GPU from 10K
        (10_000, 50, false),
        (50_000, 50, false),
        (100_000, 50, false),
        (500_000, 50, true),
        (1_000_000, 50, true),
        // Large k (k=100): n*k²=n*10000, needs n >= 2K for 20M → GPU from 2K
        (5_000, 100, false),
        (10_000, 100, false),
        (50_000, 100, false),
        (100_000, 100, true),
        (500_000, 100, true),
        // Very large k (k=200)
        (10_000, 200, false),
        (50_000, 200, true),
        (100_000, 200, true),
    ];
    for (n, k, large) in &xtx_sizes {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let x = generate_matrix(*n, *k, &mut rng);
        let cfg = if *large { &lg_config } else { &config };

        let nkk = *n * *k * *k;
        let tag = if *k >= 30 && nkk >= 20_000_000 { "GPU" } else { "CPU" };
        let r = bench_utils::run_benchmark(
            &format!("xtx_{}", tag),
            &format!("{}x{}", n, k),
            *n,
            cfg,
            || p2a_core::linalg::xtx(&x.view()),
        );
        print_result(&r);
        results.push(r);
    }
    println!();

    // ===================================================================
    // xty (X'y) — bandwidth-bound DGEMV, GPU helps at large n
    // ===================================================================
    println!("--- xty (X'y) ---");
    print_header();
    let xty_sizes: Vec<(usize, usize, bool)> = vec![
        (1_000, 10, false),
        (10_000, 10, false),
        (50_000, 10, false),
        (100_000, 10, false),
        (500_000, 10, false),
        (1_000_000, 10, true),
        (10_000, 50, false),
        (100_000, 50, false),
        (500_000, 50, true),
        (1_000_000, 50, true),
        (100_000, 100, false),
        (500_000, 100, true),
    ];
    for (n, k, large) in &xty_sizes {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let x = generate_matrix(*n, *k, &mut rng);
        let y = generate_vector(*n, &mut rng);
        let cfg = if *large { &lg_config } else { &config };

        // xty GPU dispatch is disabled by default (DGEMV slower than CPU)
        let r = bench_utils::run_benchmark(
            "xty_CPU",
            &format!("{}x{}", n, k),
            *n,
            cfg,
            || p2a_core::linalg::xty(&x.view(), &y),
        );
        print_result(&r);
        results.push(r);
    }
    println!();

    // ===================================================================
    // matmul (A * B) — cuBLAS DGEMM
    // ===================================================================
    println!("--- matmul (A * B) ---");
    print_header();
    let matmul_sizes: Vec<(usize, usize, usize, bool)> = vec![
        (100, 100, 100, false),
        (500, 500, 500, false),
        (1_000, 1_000, 1_000, false),
        (2_000, 2_000, 2_000, false),
        (3_000, 3_000, 3_000, true),
        (5_000, 5_000, 5_000, true),
        // Tall-skinny
        (10_000, 50, 50, false),
        (100_000, 50, 50, false),
        (500_000, 50, 50, true),
        // Fat
        (50, 1_000, 1_000, false),
    ];
    for (m, p, n_out, large) in &matmul_sizes {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let a = generate_matrix(*m, *p, &mut rng);
        let b = generate_matrix(*p, *n_out, &mut rng);
        let cfg = if *large { &lg_config } else { &config };

        let mnk = *m * *p * *n_out;
        let dims = [*m, *p, *n_out];
        let min_d = *dims.iter().min().unwrap() as f64;
        let max_d = *dims.iter().max().unwrap() as f64;
        let shape_ratio = if max_d > 0.0 { min_d / max_d } else { 0.0 };
        let tag = if mnk >= 1_000_000 && shape_ratio >= 0.1 { "GPU" } else { "CPU" };
        let r = bench_utils::run_benchmark(
            &format!("matmul_{}", tag),
            &format!("{}x{}x{}", m, p, n_out),
            *m,
            cfg,
            || p2a_core::linalg::matmul(&a.view(), &b.view()).unwrap(),
        );
        print_result(&r);
        results.push(r);
    }
    println!();

    // ===================================================================
    // cholesky_inverse — GPU path for k >= 100
    // ===================================================================
    println!("--- cholesky_inverse ---");
    print_header();
    let inv_sizes: Vec<(usize, bool)> =
        vec![(10, false), (50, false), (100, false), (200, false), (500, false), (1_000, true)];
    for (k, large) in &inv_sizes {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        // Positive definite matrix: X'X + I
        let x = generate_matrix(k * 2, *k, &mut rng);
        let xtx = p2a_core::linalg::xtx(&x.view());
        let eye: Array2<f64> = Array2::eye(*k);
        let m: Array2<f64> = &xtx + &eye;
        let cfg = if *large { &lg_config } else { &config };

        let tag = if *k >= 100 { "GPU" } else { "CPU" };
        let r = bench_utils::run_benchmark(
            &format!("cholesky_inv_{}", tag),
            &format!("{}x{}", k, k),
            *k,
            cfg,
            || p2a_core::linalg::cholesky_inverse(&m.view()).unwrap(),
        );
        print_result(&r);
        results.push(r);
    }
    println!();

    // ===================================================================
    // Full OLS pipeline: xtx + inverse + xty + beta
    // ===================================================================
    println!("--- OLS pipeline (xtx + inverse + xty + dot) ---");
    print_header();
    let ols_sizes: Vec<(usize, usize, bool)> = vec![
        (1_000, 5, false),
        (10_000, 10, false),
        (100_000, 10, false),
        (500_000, 10, true),
        (1_000_000, 10, true),
        // Larger k where GPU xtx helps
        (10_000, 50, false),
        (100_000, 50, false),
        (500_000, 50, true),
        (1_000_000, 50, true),
        (100_000, 100, false),
        (500_000, 100, true),
    ];
    for (n, k, large) in &ols_sizes {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let x = generate_matrix(*n, *k, &mut rng);
        let y = generate_vector(*n, &mut rng);
        let cfg = if *large { &lg_config } else { &config };

        let r = bench_utils::run_benchmark(
            "ols_pipeline",
            &format!("{}x{}", n, k),
            *n,
            cfg,
            || {
                let xtx = p2a_core::linalg::xtx(&x.view());
                let (xtx_inv, _) = p2a_core::linalg::safe_inverse(&xtx.view()).unwrap();
                let xty_val = p2a_core::linalg::xty(&x.view(), &y);
                xtx_inv.dot(&xty_val)
            },
        );
        print_result(&r);
        results.push(r);
    }
    println!();

    // ===================================================================
    // Sandwich estimator (HC robust SEs) via full run_ols
    // ===================================================================
    println!("--- sandwich meat (X' diag(w) X) via run_ols HC1 ---");
    print_header();
    let sandwich_sizes: Vec<(usize, usize, bool)> = vec![
        (1_000, 5, false),
        (10_000, 10, false),
        (50_000, 10, false),
        (100_000, 10, false),
        (500_000, 10, true),
        // Larger k
        (10_000, 50, false),
        (50_000, 50, false),
        (100_000, 50, true),
    ];
    for (n, k, large) in &sandwich_sizes {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Build a dataset with noise for HC1
        let x_data: Vec<f64> = (0..*n).map(|_| normal.sample(&mut rng)).collect();
        let y_data: Vec<f64> = x_data
            .iter()
            .map(|xi| 2.0 + 3.0 * xi + normal.sample(&mut rng))
            .collect();

        // Build multi-regressor dataset
        let mut columns: Vec<polars::prelude::Column> = Vec::new();
        columns.push(polars::prelude::Column::new("y".into(), &y_data));
        let x_names: Vec<String> = (0..*k).map(|i| format!("x{}", i)).collect();
        for (i, name) in x_names.iter().enumerate() {
            let col_data: Vec<f64> = if i == 0 {
                x_data.clone()
            } else {
                (0..*n).map(|_| normal.sample(&mut rng)).collect()
            };
            columns.push(polars::prelude::Column::new(
                name.as_str().into(),
                &col_data,
            ));
        }

        let df = polars::prelude::DataFrame::new(columns).unwrap();
        let dataset = p2a_core::Dataset::new(df);
        let x_refs: Vec<&str> = x_names.iter().map(|s| s.as_str()).collect();
        let cfg = if *large { &lg_config } else { &config };

        let r = bench_utils::run_benchmark(
            "run_ols_hc1",
            &format!("{}x{}", n, k),
            *n,
            cfg,
            || {
                p2a_core::regression::run_ols(
                    &dataset,
                    "y",
                    &x_refs,
                    true,
                    p2a_core::regression::CovarianceType::HC1,
                )
                .unwrap()
            },
        );
        print_result(&r);
        results.push(r);
    }
    println!();

    // ===================================================================
    // K-means clustering
    // ===================================================================
    println!("--- K-means clustering ---");
    print_header();
    let kmeans_sizes: Vec<(usize, usize, usize, bool)> = vec![
        (1_000, 10, 5, false),
        (5_000, 10, 5, false),
        (10_000, 10, 5, false),
        (50_000, 10, 5, false),
        (100_000, 10, 5, true),
        (200_000, 10, 5, true),
        // Higher dimensions
        (10_000, 50, 10, false),
        (50_000, 50, 10, true),
        (100_000, 50, 10, true),
        // More clusters
        (50_000, 20, 20, true),
    ];
    for (n, d, k_clusters, large) in &kmeans_sizes {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = generate_matrix(*n, *d, &mut rng);
        let cfg = if *large { &lg_config } else { &config };

        let tag = if *n >= 10_000 && *d >= 20 { "GPU" } else { "CPU" };
        let r = bench_utils::run_benchmark(
            &format!("kmeans_{}", tag),
            &format!("{}x{}_k{}", n, d, k_clusters),
            *n,
            cfg,
            || {
                p2a_core::kmeans(
                    data.view(),
                    *k_clusters,
                    Some(50),
                    None,
                    Some(1),
                    Some(42),
                )
                .unwrap()
            },
        );
        print_result(&r);
        results.push(r);
    }
    println!();

    // ===================================================================
    // PCA — biggest GPU win (covariance path avoids full SVD)
    // ===================================================================
    println!("--- PCA ---");
    print_header();
    let pca_sizes: Vec<(usize, usize, usize, bool)> = vec![
        (1_000, 20, 5, false),
        (5_000, 50, 10, false),
        (10_000, 50, 10, false),
        (50_000, 50, 10, false),
        (100_000, 50, 10, true),
        (200_000, 50, 10, true),
        (500_000, 50, 10, true),
        // Higher dimensions
        (10_000, 100, 20, false),
        (50_000, 100, 20, true),
        (100_000, 100, 20, true),
        (200_000, 100, 20, true),
        // Very wide
        (10_000, 200, 30, true),
        (50_000, 200, 30, true),
    ];
    for (n, p, nc, large) in &pca_sizes {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = generate_matrix(*n, *p, &mut rng);
        let cfg = if *large { &lg_config } else { &config };

        let npp = *n * *p * *p;
        let tag = if *p >= 30 && npp >= 20_000_000 && *n > 4 * *p {
            "GPU"
        } else {
            "CPU"
        };
        let r = bench_utils::run_benchmark(
            &format!("pca_{}", tag),
            &format!("{}x{}_nc{}", n, p, nc),
            *n,
            cfg,
            || p2a_core::pca(data.view(), Some(*nc), true).unwrap(),
        );
        print_result(&r);
        results.push(r);
    }
    println!();

    // ===================================================================
    // HDFE (High-Dimensional Fixed Effects) — lfe/felm style
    // GPU helps via xtx/xty when k is large; demeaning dominates for small k
    // ===================================================================
    println!("--- HDFE (2-way FE, lfe/felm style) ---");
    print_header();
    let hdfe_sizes: Vec<(usize, usize, usize, bool)> = vec![
        // (n_entities, n_periods, n_regressors, large_config)
        // Small: typical panel
        (100, 50, 2, false),      // n=5K, k=2
        (500, 100, 5, false),     // n=50K, k=5
        (1_000, 100, 10, false),  // n=100K, k=10
        (2_000, 100, 10, true),   // n=200K, k=10
        (5_000, 100, 10, true),   // n=500K, k=10
        // High-dimensional controls
        (500, 100, 50, true),     // n=50K, k=50 — GPU xtx helps here
        (1_000, 100, 50, true),   // n=100K, k=50
    ];
    for (n_ent, n_per, n_reg, large) in &hdfe_sizes {
        let n = *n_ent * *n_per;
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Generate balanced panel data
        let entity_ids: Vec<i64> = (0..n).map(|i| (i / *n_per) as i64).collect();
        let time_ids: Vec<i64> = (0..n).map(|i| (i % *n_per) as i64).collect();

        // Entity and time effects
        let entity_effects: Vec<f64> = (0..*n_ent).map(|_| normal.sample(&mut rng) * 2.0).collect();
        let time_effects: Vec<f64> = (0..*n_per).map(|_| normal.sample(&mut rng)).collect();

        // Build x columns and y
        let mut columns: Vec<polars::prelude::Column> = Vec::new();
        let x_names: Vec<String> = (0..*n_reg).map(|i| format!("x{}", i)).collect();

        let mut x_data: Vec<Vec<f64>> = Vec::new();
        for _ in 0..*n_reg {
            x_data.push((0..n).map(|_| normal.sample(&mut rng)).collect());
        }

        let y_data: Vec<f64> = (0..n)
            .map(|i| {
                let ent = i / *n_per;
                let t = i % *n_per;
                let mut y = entity_effects[ent] + time_effects[t];
                for (j, xd) in x_data.iter().enumerate() {
                    y += (j as f64 + 1.0) * xd[i];
                }
                y + normal.sample(&mut rng)
            })
            .collect();

        columns.push(polars::prelude::Column::new("y".into(), &y_data));
        columns.push(polars::prelude::Column::new("entity".into(), &entity_ids));
        columns.push(polars::prelude::Column::new("time".into(), &time_ids));
        for (j, name) in x_names.iter().enumerate() {
            columns.push(polars::prelude::Column::new(name.as_str().into(), &x_data[j]));
        }

        let df = polars::prelude::DataFrame::new(columns).unwrap();
        let dataset = p2a_core::Dataset::new(df);
        let x_refs: Vec<&str> = x_names.iter().map(|s| s.as_str()).collect();
        let cfg = if *large { &lg_config } else { &config };

        let r = bench_utils::run_benchmark(
            "hdfe_2way",
            &format!("{}ent_{}per_{}x", n_ent, n_per, n_reg),
            n,
            cfg,
            || {
                p2a_core::run_hdfe(
                    &dataset,
                    "y",
                    &x_refs,
                    &["entity", "time"],
                    None,
                    p2a_core::regression::CovarianceType::HC1,
                )
                .unwrap()
            },
        );
        print_result(&r);
        results.push(r);
    }
    println!();

    // ---------- Summary ----------
    println!("=== Summary: {} benchmarks completed ===", results.len());

    // Save results — cargo bench may run from crate or workspace root,
    // so resolve relative to the workspace Cargo.toml location.
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let rel_path = format!(
        "performance/comparisons/r_comparison/results/rust_gpu_{}.json",
        timestamp
    );
    // Try workspace root first, then try crate root (../../ from crates/p2a-core/)
    let candidates = [
        std::path::PathBuf::from(&rel_path),
        std::path::PathBuf::from(format!("../../{}", rel_path)),
    ];
    let mut saved = false;
    for path in &candidates {
        if let Some(parent) = path.parent() {
            if parent.exists() {
                match save_results(&results, path.to_str().unwrap()) {
                    Ok(()) => {
                        println!("Results saved to {}", path.display());
                        saved = true;
                        break;
                    }
                    Err(e) => eprintln!("Could not write to {}: {}", path.display(), e),
                }
            }
        }
    }
    if !saved {
        eprintln!("Warning: could not save JSON results to any candidate path");
    }
}
