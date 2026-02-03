//! Benchmarks for data loading operations
//!
//! Run with: `cargo bench -p p2a-core --bench data_loading_benchmarks`
//!
//! Tests performance of:
//! - CSV loading at various sizes
//! - Parquet loading at various sizes
//! - Chunked CSV reading
//! - Dataset operations (lazy, sample, filter)

mod bench_utils;

use bench_utils::{BenchConfig, print_header, print_result, run_benchmark, save_results};
use p2a_core::data::DataLoader;
use p2a_core::Dataset;
use polars::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::io::Write;
use tempfile::NamedTempFile;

/// Generate a test CSV file with specified dimensions
fn create_test_csv(n_rows: usize, n_cols: usize, seed: u64) -> NamedTempFile {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut file = NamedTempFile::new().expect("Failed to create temp file");

    // Write header
    let headers: Vec<String> = (0..n_cols).map(|i| format!("col_{}", i)).collect();
    writeln!(file, "{}", headers.join(",")).expect("Failed to write header");

    // Write data rows
    for _ in 0..n_rows {
        let values: Vec<String> = (0..n_cols)
            .map(|_| format!("{:.4}", rng.gen_range(-100.0f64..100.0)))
            .collect();
        writeln!(file, "{}", values.join(",")).expect("Failed to write row");
    }

    file.flush().expect("Failed to flush file");
    file
}

/// Generate a test Parquet file with specified dimensions
fn create_test_parquet(n_rows: usize, n_cols: usize, seed: u64) -> NamedTempFile {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut columns: Vec<Column> = Vec::with_capacity(n_cols);
    for i in 0..n_cols {
        let values: Vec<f64> = (0..n_rows).map(|_| rng.gen_range(-100.0..100.0)).collect();
        columns.push(Column::new(format!("col_{}", i).into(), values));
    }

    let df = DataFrame::new(columns).expect("Failed to create DataFrame");

    let file = NamedTempFile::with_suffix(".parquet").expect("Failed to create temp file");
    let path = file.path();

    let mut parquet_file = std::fs::File::create(path).expect("Failed to create parquet file");
    ParquetWriter::new(&mut parquet_file)
        .finish(&mut df.clone())
        .expect("Failed to write parquet");

    file
}

/// Generate test dataset in memory
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
    println!("Data Loading Benchmarks");
    println!("=======================\n");

    let config = BenchConfig {
        warmup_iterations: 3,
        measurement_iterations: 20,
        capture_raw_times: true,
    };

    let mut results = Vec::new();

    // ========================================================================
    // CSV Loading Benchmarks
    // ========================================================================
    println!("\n## CSV Loading\n");
    print_header();

    for (n_rows, n_cols) in [(1000, 10), (10000, 20), (50000, 30)] {
        let csv_file = create_test_csv(n_rows, n_cols, 42);
        let path = csv_file.path().to_path_buf();

        let bench_result = run_benchmark(
            "load_csv",
            &format!("{}x{}", n_rows, n_cols),
            n_rows * n_cols,
            &config,
            || DataLoader::load_csv(&path).expect("Failed to load CSV"),
        );
        print_result(&bench_result);
        results.push(bench_result);
    }

    // ========================================================================
    // Parquet Loading Benchmarks
    // ========================================================================
    println!("\n## Parquet Loading\n");
    print_header();

    for (n_rows, n_cols) in [(1000, 10), (10000, 20), (50000, 30)] {
        let parquet_file = create_test_parquet(n_rows, n_cols, 42);
        let path = parquet_file.path().to_path_buf();

        let bench_result = run_benchmark(
            "load_parquet",
            &format!("{}x{}", n_rows, n_cols),
            n_rows * n_cols,
            &config,
            || DataLoader::load_parquet(&path).expect("Failed to load Parquet"),
        );
        print_result(&bench_result);
        results.push(bench_result);
    }

    // ========================================================================
    // CSV vs Parquet Comparison (same data size)
    // ========================================================================
    println!("\n## CSV vs Parquet (10k x 20 columns)\n");
    print_header();

    let csv_file = create_test_csv(10000, 20, 123);
    let csv_path = csv_file.path().to_path_buf();

    let parquet_file = create_test_parquet(10000, 20, 123);
    let parquet_path = parquet_file.path().to_path_buf();

    let bench_csv = run_benchmark("format_compare", "csv", 200000, &config, || {
        DataLoader::load_csv(&csv_path).expect("Failed to load CSV")
    });
    print_result(&bench_csv);
    results.push(bench_csv);

    let bench_parquet = run_benchmark("format_compare", "parquet", 200000, &config, || {
        DataLoader::load_parquet(&parquet_path).expect("Failed to load Parquet")
    });
    print_result(&bench_parquet);
    results.push(bench_parquet);

    // ========================================================================
    // Row-Limited Loading
    // ========================================================================
    println!("\n## Row-Limited Loading (50k row file, load first N rows)\n");
    print_header();

    let large_csv = create_test_csv(50000, 20, 42);
    let large_path = large_csv.path().to_path_buf();

    for n_rows in [1000, 5000, 10000] {
        let bench_result = run_benchmark(
            "load_csv_limit",
            &format!("first{}", n_rows),
            n_rows,
            &config,
            || DataLoader::load_csv_with_limit(&large_path, Some(n_rows)).expect("Failed to load CSV"),
        );
        print_result(&bench_result);
        results.push(bench_result);
    }

    // ========================================================================
    // Chunked CSV Reading
    // ========================================================================
    println!("\n## Chunked CSV Reading (50k rows)\n");
    print_header();

    for chunk_size in [1000, 5000, 10000] {
        let mut total_rows = 0usize;
        let bench_result = run_benchmark(
            "csv_chunks",
            &format!("chunk{}", chunk_size),
            50000,
            &config,
            || {
                total_rows = 0;
                for chunk in DataLoader::iter_csv_chunks(&large_path, chunk_size) {
                    if let Ok(df) = chunk {
                        total_rows += df.height();
                    }
                }
                total_rows
            },
        );
        print_result(&bench_result);
        results.push(bench_result);
    }

    // ========================================================================
    // Dataset Operations
    // ========================================================================
    println!("\n## Dataset Operations (10k x 20)\n");
    print_header();

    let dataset = generate_dataset(10000, 20, 42);

    // Benchmark lazy()
    let bench_result = run_benchmark("dataset_op", "lazy", 10000, &config, || {
        dataset.lazy()
    });
    print_result(&bench_result);
    results.push(bench_result);

    // Benchmark sample()
    let bench_result = run_benchmark("dataset_op", "sample_1k", 1000, &config, || {
        dataset.sample(1000, Some(42)).expect("Failed to sample")
    });
    print_result(&bench_result);
    results.push(bench_result);

    // Benchmark filter()
    let bench_result = run_benchmark("dataset_op", "filter", 10000, &config, || {
        dataset
            .filter(col("col_0").gt(lit(0.0)))
            .expect("Failed to filter")
    });
    print_result(&bench_result);
    results.push(bench_result);

    // Benchmark select_columns()
    let bench_result = run_benchmark("dataset_op", "select_5col", 5, &config, || {
        dataset
            .select_columns(&["col_0", "col_1", "col_2", "col_3", "col_4"])
            .expect("Failed to select")
    });
    print_result(&bench_result);
    results.push(bench_result);

    // ========================================================================
    // File Info (Metadata Only)
    // ========================================================================
    println!("\n## File Info (No Data Loading)\n");
    print_header();

    let bench_result = run_benchmark("file_info", "csv_50k", 1, &config, || {
        DataLoader::file_info(&large_path).expect("Failed to get file info")
    });
    print_result(&bench_result);
    results.push(bench_result);

    // ========================================================================
    // Summary Statistics
    // ========================================================================
    println!("\n## Summary\n");

    // Calculate speedup of parquet vs csv
    if let (Some(csv_bench), Some(parquet_bench)) = (
        results.iter().find(|r| r.method == "format_compare" && r.variant == "csv"),
        results.iter().find(|r| r.method == "format_compare" && r.variant == "parquet"),
    ) {
        let speedup = csv_bench.time_median_us / parquet_bench.time_median_us;
        println!(
            "Parquet loading is {:.1}x faster than CSV for same data size",
            speedup
        );
    }

    // File sizes
    println!(
        "\nCSV file size: {} bytes",
        std::fs::metadata(&csv_path).map(|m| m.len()).unwrap_or(0)
    );
    println!(
        "Parquet file size: {} bytes",
        std::fs::metadata(&parquet_path).map(|m| m.len()).unwrap_or(0)
    );

    // Save results
    if let Err(e) = save_results(&results, "benchmark_results_data_loading.json") {
        eprintln!("Warning: Could not save results: {}", e);
    } else {
        println!("\nResults saved to benchmark_results_data_loading.json");
    }
}
