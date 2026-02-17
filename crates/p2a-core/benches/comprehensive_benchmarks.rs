//! Comprehensive benchmarks with distribution statistics and memory tracking
//!
//! Run with: `cargo bench -p p2a-core --bench comprehensive_benchmarks`
//!
//! This generates output similar to R's `bench::mark()` with:
//! - Full distribution statistics (min, p25, median, p75, max, mean, std)
//! - Memory allocation tracking
//! - Iterations per second
//! - Raw timing data for detailed analysis

mod bench_utils;

use bench_utils::{
    BenchConfig, BenchmarkResult, print_header, print_result, run_benchmark, save_results,
};
use p2a_core::regression::CovarianceType;
use p2a_core::spatial::{Neighbors, SpatialWeights, WeightStyle};
use p2a_core::stats::{RotationMethod, ScoresMethod, factanal, fisher_exact_test, isoreg};
use p2a_core::regression::jarque_bera_test;
use p2a_core::{
    CostFunction, DRMethod, Dataset, DoublyRobustConfig, Estimand, FisherAlternative, Linkage,
    PredictorSpec, SarConfig, SemConfig, SynthConfig, dbscan, hierarchical, kmeans, pca,
    random_forest, run_arima, run_changepoint, run_doubly_robust, run_fixed_effects, run_hdfe,
    run_loess, run_logit, run_mstl, run_ols, run_probit, run_random_effects, run_sar_dataset,
    run_sem_dataset, run_synthetic_control,
};
use polars::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Generate synthetic regression data
fn generate_regression_data(n: usize, k: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut columns: Vec<Column> = Vec::new();

    for i in 1..=k {
        let x: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        columns.push(Column::new(format!("x{}", i).into(), x));
    }

    let y: Vec<f64> = (0..n)
        .map(|row| {
            let mut sum = 0.0;
            for col in &columns {
                if let Ok(val) = col.get(row) {
                    if let Ok(v) = val.try_extract::<f64>() {
                        sum += v;
                    }
                }
            }
            sum + rng.gen_range(0.0..0.5)
        })
        .collect();

    columns.insert(0, Column::new("y".into(), y));

    let df = DataFrame::new(columns).expect("Failed to create DataFrame");
    Dataset::new(df)
}

/// Generate synthetic panel data
fn generate_panel_data(n_entities: usize, n_periods: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n = n_entities * n_periods;

    let entity: Vec<i64> = (0..n).map(|i| (i / n_periods) as i64).collect();
    let time: Vec<i64> = (0..n).map(|i| (i % n_periods) as i64).collect();
    let x1: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let y: Vec<f64> = (0..n)
        .map(|i| {
            let entity_effect = (entity[i] as f64) * 0.1;
            entity_effect + x1[i] * 0.5 + x2[i] * 0.3 + rng.gen_range(0.0..0.5)
        })
        .collect();

    let df = df! {
        "entity" => entity,
        "time" => time,
        "y" => y,
        "x1" => x1,
        "x2" => x2,
    }
    .expect("Failed to create DataFrame");

    Dataset::new(df)
}

/// Generate binary outcome data for logit/probit
fn generate_binary_data(n: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let x1: Vec<f64> = (0..n).map(|_| rng.gen_range(-2.0..2.0)).collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.gen_range(-2.0..2.0)).collect();

    let y: Vec<f64> = (0..n)
        .map(|i| {
            let linear = -1.0 + 0.5 * x1[i] + 0.3 * x2[i];
            let prob = 1.0 / (1.0 + (-linear).exp());
            if rng.gen_range(0.0..1.0) < prob {
                1.0
            } else {
                0.0
            }
        })
        .collect();

    let df = df! {
        "y" => y,
        "x1" => x1,
        "x2" => x2,
    }
    .expect("Failed to create DataFrame");

    Dataset::new(df)
}

/// Generate time series data
fn generate_time_series(n: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let y: Vec<f64> = (0..n)
        .map(|t| {
            let trend = 0.01 * t as f64;
            let seasonal = (t as f64 * std::f64::consts::PI / 6.0).sin() * 2.0;
            let noise = rng.gen_range(0.0..0.5);
            trend + seasonal + noise
        })
        .collect();

    let df = df! {
        "y" => y,
    }
    .expect("Failed to create DataFrame");

    Dataset::new(df)
}

/// Generate clustering data
fn generate_cluster_data(n: usize, k: usize, seed: u64) -> ndarray::Array2<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut data = ndarray::Array2::zeros((n, k));

    for i in 0..n {
        let cluster = i % 3;
        let center = match cluster {
            0 => 0.0,
            1 => 3.0,
            _ => 6.0,
        };
        for j in 0..k {
            data[[i, j]] = center + rng.gen_range(-0.5..0.5);
        }
    }

    data
}

fn main() {
    let config = BenchConfig {
        warmup_iterations: 10,
        measurement_iterations: 100,
        capture_raw_times: true,
    };

    let mut results: Vec<BenchmarkResult> = Vec::new();

    println!("\n=== p2a Comprehensive Benchmarks ===\n");
    println!(
        "Configuration: {} warmup, {} measurement iterations\n",
        config.warmup_iterations, config.measurement_iterations
    );

    // ============================================
    // Regression Benchmarks
    // ============================================
    println!("\n--- Regression ---");
    print_header();

    for n in [100, 1000, 10000] {
        let dataset = generate_regression_data(n, 5, 42);
        let x_cols = vec!["x1", "x2", "x3", "x4", "x5"];

        // OLS Standard
        let result = run_benchmark("OLS", "standard", n, &config, || {
            run_ols(&dataset, "y", &x_cols, true, CovarianceType::Standard)
        });
        print_result(&result);
        results.push(result);

        // OLS HC1
        let result = run_benchmark("OLS", "HC1", n, &config, || {
            run_ols(&dataset, "y", &x_cols, true, CovarianceType::HC1)
        });
        print_result(&result);
        results.push(result);
    }

    // ============================================
    // Panel Data Benchmarks
    // ============================================
    println!("\n--- Panel Data ---");
    print_header();

    for (n_ent, n_per) in [(10, 10), (50, 20), (100, 50)] {
        let n = n_ent * n_per;
        let dataset = generate_panel_data(n_ent, n_per, 42);

        // Fixed Effects
        let result = run_benchmark("FixedEffects", "within", n, &config, || {
            run_fixed_effects(&dataset, "y", &["x1", "x2"], "entity")
        });
        print_result(&result);
        results.push(result);

        // Random Effects
        let result = run_benchmark("RandomEffects", "GLS", n, &config, || {
            run_random_effects(&dataset, "y", &["x1", "x2"], "entity")
        });
        print_result(&result);
        results.push(result);

        // HDFE
        let result = run_benchmark("HDFE", "2-way", n, &config, || {
            run_hdfe(
                &dataset,
                "y",
                &["x1", "x2"],
                &["entity", "time"],
                None,
                CovarianceType::Standard,
            )
        });
        print_result(&result);
        results.push(result);
    }

    // ============================================
    // Discrete Choice Benchmarks
    // ============================================
    println!("\n--- Discrete Choice ---");
    print_header();

    for n in [100, 500, 1000] {
        let dataset = generate_binary_data(n, 42);

        // Logit
        let result = run_benchmark("Logit", "MLE", n, &config, || {
            run_logit(&dataset, "y", &["x1", "x2"])
        });
        print_result(&result);
        results.push(result);

        // Probit
        let result = run_benchmark("Probit", "MLE", n, &config, || {
            run_probit(&dataset, "y", &["x1", "x2"])
        });
        print_result(&result);
        results.push(result);
    }

    // ============================================
    // Time Series Benchmarks
    // ============================================
    println!("\n--- Time Series ---");
    print_header();

    for n in [100, 200, 500] {
        let dataset = generate_time_series(n, 42);

        // ARIMA
        let result = run_benchmark("ARIMA", "(1,1,1)", n, &config, || {
            run_arima(&dataset, "y", 1, 1, 1)
        });
        print_result(&result);
        results.push(result);

        // MSTL
        let result = run_benchmark("MSTL", "period=12", n, &config, || {
            run_mstl(&dataset, "y", &[12])
        });
        print_result(&result);
        results.push(result);
    }

    // ============================================
    // ML Benchmarks
    // ============================================
    println!("\n--- Machine Learning ---");
    print_header();

    for n in [100, 1000, 5000] {
        let data = generate_cluster_data(n, 5, 42);

        // K-Means
        let result = run_benchmark("K-Means", "k=3", n, &config, || {
            kmeans(data.view(), 3, Some(100), Some(1e-4), Some(5), Some(42))
        });
        print_result(&result);
        results.push(result);

        // PCA
        let result = run_benchmark("PCA", "k=3", n, &config, || {
            pca(data.view(), Some(3), false)
        });
        print_result(&result);
        results.push(result);
    }

    // ============================================
    // Round 2 Optimized Methods (LOESS, Synth, StructTS, Changepoint)
    // ============================================
    println!("\n--- Round 2 Optimized Methods ---");
    print_header();

    // LOESS
    for n in [100, 500, 1000] {
        let dataset = generate_regression_data(n, 1, 42);
        let result = run_benchmark("LOESS", "span=0.75", n, &config, || {
            run_loess(&dataset, "y", "x1", 0.75, 1, false)
        });
        print_result(&result);
        results.push(result);
    }

    // Hierarchical Clustering
    for n in [100, 500, 1000] {
        let data = generate_cluster_data(n, 5, 42);
        let result = run_benchmark("Hierarchical", "Ward", n, &config, || {
            hierarchical(data.view(), Some(3), Linkage::Ward, None)
        });
        print_result(&result);
        results.push(result);
    }

    // Random Forest
    for n in [100, 500, 1000] {
        let data = generate_cluster_data(n, 5, 42);
        let target: ndarray::Array1<f64> = data.column(0).to_owned();
        let features = data.slice(ndarray::s![.., 1..]);
        let result = run_benchmark("RandomForest", "100trees", n, &config, || {
            random_forest(
                features.view(),
                target.view(),
                Some(100),
                Some(10),
                Some(5),
                None,
                Some(42),
                None,
            )
        });
        print_result(&result);
        results.push(result);
    }

    // Doubly Robust (AIPW)
    for n in [200, 500, 1000] {
        let dataset = generate_binary_data(n, 42);
        let dr_config = DoublyRobustConfig {
            method: DRMethod::AIPW,
            estimand: Estimand::ATE,
            bootstrap: 999,
            seed: Some(42),
            ..Default::default()
        };
        let result = run_benchmark("Doubly_Robust", "AIPW", n, &config, || {
            run_doubly_robust(&dataset, "y", "x1", &["x2"], dr_config.clone())
        });
        print_result(&result);
        results.push(result);
    }

    // Changepoint
    for n in [100, 500, 1000] {
        let dataset = generate_time_series(n, 42);
        let result = run_benchmark("Changepoint", "PELT", n, &config, || {
            run_changepoint(&dataset, "y", None, None, CostFunction::MeanChange)
        });
        print_result(&result);
        results.push(result);
    }

    // Synthetic Control
    for n_units in [10, 30] {
        let n_periods = 10;
        let n = n_units * n_periods;
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let mut unit_ids: Vec<String> = Vec::new();
        let mut time_ids: Vec<i64> = Vec::new();
        let mut outcome: Vec<f64> = Vec::new();
        let mut pred1: Vec<f64> = Vec::new();
        let mut pred2: Vec<f64> = Vec::new();

        for u in 0..n_units {
            for t in 0..n_periods {
                unit_ids.push(format!("unit_{}", u));
                time_ids.push(t as i64);
                let base = (u as f64) * 0.5 + (t as f64) * 0.1;
                let treatment_effect = if u == 0 && t >= 7 { 2.0 } else { 0.0 };
                outcome.push(base + treatment_effect + rng.gen_range(-0.3..0.3));
                pred1.push(rng.gen_range(0.0..1.0));
                pred2.push(rng.gen_range(0.0..1.0));
            }
        }

        let df = df! {
            "unit" => &unit_ids,
            "time" => &time_ids,
            "outcome" => &outcome,
            "pred1" => &pred1,
            "pred2" => &pred2,
        }
        .expect("synth data");
        let dataset = Dataset::new(df);

        let predictors = vec![PredictorSpec::new("pred1"), PredictorSpec::new("pred2")];
        let config_synth = SynthConfig {
            treatment_time: 7,
            treated_unit: "unit_0".to_string(),
            run_placebos: false,
            ..Default::default()
        };

        let result = run_benchmark("SynthControl", "Nelder-Mead", n, &config, || {
            run_synthetic_control(
                &dataset,
                "outcome",
                "unit",
                "time",
                &predictors,
                config_synth.clone(),
            )
        });
        print_result(&result);
        results.push(result);
    }

    // ============================================
    // Round 3: Optimized Methods (SAR, SEM, DBSCAN, Factor Analysis, Fisher, Isotonic, JB)
    // ============================================
    println!("\n--- Round 3 Optimized Methods ---");
    print_header();

    // --- Spatial SAR/SEM (Ord 1975 optimization) ---
    for n_side in [10, 20, 32] {
        let n = n_side * n_side;
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let mut coords = Vec::with_capacity(n);
        for y_coord in 0..n_side {
            for x_coord in 0..n_side {
                coords.push((x_coord as f64, y_coord as f64));
            }
        }

        let nb = Neighbors::from_knn(&coords, 4);
        let mut listw = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);
        let _ = listw.eigenvalues(); // Pre-compute

        let x_vals: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let y_vals: Vec<f64> = (0..n)
            .map(|i| {
                let (cx, cy) = coords[i];
                2.0 + 0.7 * x_vals[i] + 0.3 * (cx + cy) / (n_side as f64) + rng.gen_range(-0.25..0.25)
            })
            .collect();

        let df = df! { "y" => &y_vals, "x" => &x_vals }.expect("spatial data");
        let dataset = Dataset::new(df);
        let sar_config = SarConfig { compute_impacts: false, ..Default::default() };
        let sem_config = SemConfig::default();

        let listw_clone = listw.clone();
        let result = run_benchmark("SAR", "lagsarlm", n, &config, || {
            let mut lw = listw_clone.clone();
            run_sar_dataset(&dataset, "y", &["x"], &mut lw, sar_config.clone())
        });
        print_result(&result);
        results.push(result);

        let listw_clone = listw.clone();
        let result = run_benchmark("SEM", "errorsarlm", n, &config, || {
            let mut lw = listw_clone.clone();
            run_sem_dataset(&dataset, "y", &["x"], &mut lw, sem_config.clone())
        });
        print_result(&result);
        results.push(result);
    }

    // --- DBSCAN (small-n condensed distance matrix optimization) ---
    for n in [100, 1000, 5000] {
        let data = generate_cluster_data(n, 5, 42);
        let result = run_benchmark("DBSCAN", "eps=1.5", n, &config, || {
            dbscan(data.view(), 1.5, 5)
        });
        print_result(&result);
        results.push(result);
    }

    // --- Factor Analysis (top-k eigenpairs, Cholesky log-det) ---
    for n in [100, 500, 1000] {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let p = 10; // 10 variables, 3 factors
        let k = 3;
        let mut data = ndarray::Array2::zeros((n, p));
        for i in 0..n {
            let f1: f64 = rng.gen_range(-2.0..2.0);
            let f2: f64 = rng.gen_range(-2.0..2.0);
            let f3: f64 = rng.gen_range(-2.0..2.0);
            // Variables 1-3 load on factor 1
            data[[i, 0]] = 0.8 * f1 + rng.gen_range(-0.3..0.3);
            data[[i, 1]] = 0.7 * f1 + rng.gen_range(-0.4..0.4);
            data[[i, 2]] = 0.75 * f1 + rng.gen_range(-0.35..0.35);
            // Variables 4-6 load on factor 2
            data[[i, 3]] = 0.8 * f2 + rng.gen_range(-0.3..0.3);
            data[[i, 4]] = 0.7 * f2 + rng.gen_range(-0.4..0.4);
            data[[i, 5]] = 0.75 * f2 + rng.gen_range(-0.35..0.35);
            // Variables 7-9 load on factor 3
            data[[i, 6]] = 0.8 * f3 + rng.gen_range(-0.3..0.3);
            data[[i, 7]] = 0.7 * f3 + rng.gen_range(-0.4..0.4);
            data[[i, 8]] = 0.75 * f3 + rng.gen_range(-0.35..0.35);
            // Variable 10: noise
            data[[i, 9]] = rng.gen_range(-1.0..1.0);
        }

        let result = run_benchmark("factanal", "none", n, &config, || {
            factanal(&data.view(), k, RotationMethod::None, ScoresMethod::None)
        });
        print_result(&result);
        results.push(result);

        let result = run_benchmark("factanal", "varimax", n, &config, || {
            factanal(&data.view(), k, RotationMethod::Varimax, ScoresMethod::None)
        });
        print_result(&result);
        results.push(result);
    }

    // --- Fisher Exact Test (pre-computed PMF, early termination) ---
    for n in [20, 100, 500, 1000] {
        // Create a 2x2 contingency table with total ~n
        let a = (n as f64 * 0.3) as f64;
        let b = (n as f64 * 0.2) as f64;
        let c = (n as f64 * 0.15) as f64;
        let d = n as f64 - a - b - c;
        let table = [[a, b], [c, d]];

        let result = run_benchmark("Fisher", "twosided", n, &config, || {
            fisher_exact_test(&table, FisherAlternative::TwoSided, None)
        });
        print_result(&result);
        results.push(result);
    }

    // --- Fisher Exact Test with CI ---
    for n in [20, 100, 500, 1000] {
        let a = (n as f64 * 0.3) as f64;
        let b = (n as f64 * 0.2) as f64;
        let c = (n as f64 * 0.15) as f64;
        let d = n as f64 - a - b - c;
        let table = [[a, b], [c, d]];

        let result = run_benchmark("Fisher", "with_ci", n, &config, || {
            fisher_exact_test(&table, FisherAlternative::TwoSided, Some(0.95))
        });
        print_result(&result);
        results.push(result);
    }

    // --- Isotonic Regression (O(n) stack-based PAVA) ---
    for n in [100, 1000, 10000] {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * 2.0 + rng.gen_range(-0.5..0.5)).collect();

        let result = run_benchmark("Isotonic_Regression", "PAVA", n, &config, || {
            isoreg(&x, &y)
        });
        print_result(&result);
        results.push(result);
    }

    // --- Jarque-Bera (standalone, matching R's jarque.bera.test) ---
    for n in [100, 1000, 10000] {
        let dataset = generate_regression_data(n, 5, 42);
        let x_cols = vec!["x1", "x2", "x3", "x4", "x5"];
        // Pre-compute OLS residuals (not timed — R's jarque.bera.test also takes a vector)
        let ols_result = run_ols(&dataset, "y", &x_cols, true, CovarianceType::Standard).unwrap();
        use p2a_core::traits::LinearEstimator;
        let residuals: Vec<f64> = ols_result.residuals().to_vec();

        let result = run_benchmark("Jarque_Bera", "standalone", n, &config, || {
            jarque_bera_test(&residuals)
        });
        print_result(&result);
        results.push(result);
    }

    // ============================================
    // Save Results
    // ============================================
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let results_path = format!("performance/results/rust_comprehensive_{}.json", timestamp);

    // Save to both performance/results/ and r_comparison/results/ for merge pipeline
    let r_comparison_path = format!(
        "performance/comparisons/r_comparison/results/rust_comprehensive_{}.json",
        timestamp
    );

    // Try to save, but don't fail if directory doesn't exist
    if let Err(e) = save_results(&results, &results_path) {
        eprintln!("\nNote: Could not save results to {}: {}", results_path, e);
        // Try current directory as fallback
        let fallback_path = format!("rust_comprehensive_{}.json", timestamp);
        if let Ok(()) = save_results(&results, &fallback_path) {
            println!("\nResults saved to: {}", fallback_path);
        }
    } else {
        println!("\nResults saved to: {}", results_path);
    }

    // Also save to r_comparison results directory for merge pipeline
    if let Err(e) = save_results(&results, &r_comparison_path) {
        eprintln!("Note: Could not save to r_comparison: {}", e);
    } else {
        println!("Results also saved to: {}", r_comparison_path);
    }

    // Print summary
    println!("\n=== Summary ===");
    println!("Total benchmarks: {}", results.len());
    println!(
        "Iterations per benchmark: {}",
        config.measurement_iterations
    );

    // Print distribution example
    if let Some(r) = results.first() {
        println!("\nExample distribution ({} {}):", r.method, r.variant);
        println!("  Min:    {:>10.1} µs", r.time_min_us);
        println!("  P25:    {:>10.1} µs", r.time_p25_us);
        println!("  Median: {:>10.1} µs", r.time_median_us);
        println!("  P75:    {:>10.1} µs", r.time_p75_us);
        println!("  Max:    {:>10.1} µs", r.time_max_us);
        println!("  Mean:   {:>10.1} µs", r.time_mean_us);
        println!("  Std:    {:>10.1} µs", r.time_std_us);
    }
}
