//! Benchmarks for cache and memory operations
//!
//! Run with: `cargo bench -p p2a-core --bench cache_benchmarks`
//!
//! Tests performance of:
//! - ResultCache insert/get operations
//! - LRU eviction under load
//! - MemoryProfiler tracking
//! - Dataset memory estimation

mod bench_utils;

use bench_utils::{BenchConfig, print_header, print_result, run_benchmark, save_results};
use p2a_core::Dataset;
use p2a_core::cache::{CacheKey, ResultCache};
use p2a_core::memory::{MemoryProfiler, estimate_dataset_memory};
use polars::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// A simple cacheable result for benchmarking
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchResult {
    coefficients: Vec<f64>,
    r_squared: f64,
    n_obs: usize,
}

impl BenchResult {
    fn random(rng: &mut impl Rng, n_coefs: usize) -> Self {
        Self {
            coefficients: (0..n_coefs).map(|_| rng.gen_range(-10.0..10.0)).collect(),
            r_squared: rng.gen_range(0.0..1.0),
            n_obs: rng.gen_range(100..10000),
        }
    }
}

/// Generate test dataset of specified size
fn generate_dataset(n_rows: usize, n_cols: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut columns: Vec<Column> = Vec::with_capacity(n_cols);
    for i in 0..n_cols {
        let values: Vec<f64> = (0..n_rows).map(|_| rng.gen_range(-100.0..100.0)).collect();
        columns.push(Column::new(format!("col_{}", i).into(), values));
    }

    let df = DataFrame::new(columns).expect("Failed to create DataFrame");
    Dataset::new(df)
}

fn main() {
    println!("Cache and Memory Benchmarks");
    println!("============================\n");

    let config = BenchConfig {
        warmup_iterations: 5,
        measurement_iterations: 50,
        capture_raw_times: true,
    };

    let mut results = Vec::new();

    // ========================================================================
    // Cache Insert Benchmarks
    // ========================================================================
    println!("\n## Cache Insert Operations\n");
    print_header();

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Benchmark cache insert with varying result sizes
    for n_coefs in [10, 100, 1000] {
        let result = BenchResult::random(&mut rng, n_coefs);
        let mut cache = ResultCache::new(1000);

        let bench_result = run_benchmark(
            "cache_insert",
            &format!("{}coefs", n_coefs),
            n_coefs,
            &config,
            || {
                let key = CacheKey::new("test")
                    .with_num("iter", rng.r#gen::<u32>())
                    .with_param("type", "benchmark");
                cache.insert(&key, &result);
            },
        );
        print_result(&bench_result);
        results.push(bench_result);
    }

    // ========================================================================
    // Cache Get Benchmarks (Hit vs Miss)
    // ========================================================================
    println!("\n## Cache Get Operations\n");
    print_header();

    // Populate cache first
    let mut cache = ResultCache::new(1000);
    for i in 0..500 {
        let key = CacheKey::new("benchmark").with_num("id", i);
        let result = BenchResult::random(&mut rng, 50);
        cache.insert(&key, &result);
    }

    // Benchmark cache hit
    let hit_key = CacheKey::new("benchmark").with_num("id", 250);
    let bench_result = run_benchmark("cache_get", "hit", 500, &config, || {
        let _: Option<BenchResult> = cache.get(&hit_key);
    });
    print_result(&bench_result);
    results.push(bench_result);

    // Benchmark cache miss
    let miss_key = CacheKey::new("benchmark").with_num("id", 9999);
    let bench_result = run_benchmark("cache_get", "miss", 500, &config, || {
        let _: Option<BenchResult> = cache.get(&miss_key);
    });
    print_result(&bench_result);
    results.push(bench_result);

    // ========================================================================
    // Cache LRU Eviction Benchmarks
    // ========================================================================
    println!("\n## Cache LRU Eviction\n");
    print_header();

    for capacity in [100, 500, 1000] {
        let mut cache = ResultCache::new(capacity);

        // Fill cache to capacity
        for i in 0..capacity {
            let key = CacheKey::new("evict").with_num("id", i);
            let result = BenchResult::random(&mut rng, 20);
            cache.insert(&key, &result);
        }

        // Benchmark insertion with eviction
        let bench_result = run_benchmark(
            "cache_evict",
            &format!("cap{}", capacity),
            capacity,
            &config,
            || {
                let key = CacheKey::new("evict").with_num("id", rng.r#gen::<u32>());
                let result = BenchResult::random(&mut rng, 20);
                cache.insert(&key, &result);
            },
        );
        print_result(&bench_result);
        results.push(bench_result);
    }

    // ========================================================================
    // Memory Estimation Benchmarks
    // ========================================================================
    println!("\n## Memory Estimation\n");
    print_header();

    for (n_rows, n_cols) in [(1000, 10), (10000, 20), (100000, 50)] {
        let dataset = generate_dataset(n_rows, n_cols, 42);

        let bench_result = run_benchmark(
            "estimate_memory",
            &format!("{}x{}", n_rows, n_cols),
            n_rows * n_cols,
            &config,
            || estimate_dataset_memory(&dataset),
        );
        print_result(&bench_result);
        results.push(bench_result);
    }

    // ========================================================================
    // Memory Profiler Benchmarks
    // ========================================================================
    println!("\n## Memory Profiler Operations\n");
    print_header();

    let dataset = generate_dataset(10000, 20, 42);

    // Benchmark track_dataset
    let mut profiler = MemoryProfiler::new();
    let bench_result = run_benchmark("profiler_track", "10kx20", 10000, &config, || {
        profiler.track_dataset(&format!("ds_{}", rng.r#gen::<u32>()), &dataset);
    });
    print_result(&bench_result);
    results.push(bench_result);

    // Benchmark stats generation
    let mut profiler = MemoryProfiler::new();
    for i in 0..100 {
        profiler.track_dataset(&format!("dataset_{}", i), &dataset);
    }

    let bench_result = run_benchmark("profiler_stats", "100ds", 100, &config, || profiler.stats());
    print_result(&bench_result);
    results.push(bench_result);

    // ========================================================================
    // CacheKey Construction Benchmarks
    // ========================================================================
    println!("\n## CacheKey Construction\n");
    print_header();

    // Simple key
    let bench_result = run_benchmark("cache_key", "simple", 1, &config, || {
        CacheKey::new("ols").with_param("dataset", "mydata")
    });
    print_result(&bench_result);
    results.push(bench_result);

    // Complex key with many params
    let bench_result = run_benchmark("cache_key", "complex", 10, &config, || {
        CacheKey::new("regression")
            .with_param("dataset", "large_dataset")
            .with_param("y", "outcome")
            .with_params("x", &["var1", "var2", "var3", "var4", "var5"])
            .with_bool("intercept", true)
            .with_num("clusters", 5)
            .with_param("method", "ols")
    });
    print_result(&bench_result);
    results.push(bench_result);

    // ========================================================================
    // Cache with Expiration Benchmarks
    // ========================================================================
    println!("\n## Cache with Expiration\n");
    print_header();

    use std::time::Duration;

    let mut cache = ResultCache::new(500).with_max_age(Duration::from_secs(60));

    // Populate with entries
    for i in 0..250 {
        let key = CacheKey::new("expiring").with_num("id", i);
        let result = BenchResult::random(&mut rng, 30);
        cache.insert(&key, &result);
    }

    // Benchmark contains check (includes expiration check)
    let key = CacheKey::new("expiring").with_num("id", 125);
    let bench_result = run_benchmark("cache_contains", "with_expiry", 250, &config, || {
        cache.contains(&key)
    });
    print_result(&bench_result);
    results.push(bench_result);

    // Benchmark cleanup_expired
    let bench_result = run_benchmark("cache_cleanup", "250entries", 250, &config, || {
        cache.cleanup_expired()
    });
    print_result(&bench_result);
    results.push(bench_result);

    // ========================================================================
    // Summary
    // ========================================================================
    println!("\n## Summary\n");

    let cache_stats = cache.stats();
    println!(
        "Final cache state: {} entries, {:.1}% hit rate",
        cache_stats.size,
        cache_stats.hit_rate * 100.0
    );

    // Save results
    if let Err(e) = save_results(&results, "benchmark_results_cache.json") {
        eprintln!("Warning: Could not save results: {}", e);
    } else {
        println!("\nResults saved to benchmark_results_cache.json");
    }
}
