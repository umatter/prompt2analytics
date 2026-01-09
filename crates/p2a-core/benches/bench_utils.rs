//! Benchmark utilities for comprehensive performance measurement
//!
//! Provides distribution statistics and memory tracking similar to R's `bench` package.

use std::time::Instant;
use memory_stats::memory_stats;
use serde::{Serialize, Deserialize};

/// Result from a single benchmark run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub method: String,
    pub variant: String,
    pub n: usize,
    pub iterations: usize,

    // Time statistics (in microseconds)
    pub time_min_us: f64,
    pub time_p25_us: f64,
    pub time_median_us: f64,
    pub time_p75_us: f64,
    pub time_max_us: f64,
    pub time_mean_us: f64,
    pub time_std_us: f64,

    // Iterations per second
    pub itr_per_sec: f64,

    // Memory statistics (in bytes)
    pub mem_before_bytes: usize,
    pub mem_after_bytes: usize,
    pub mem_peak_bytes: usize,
    pub mem_alloc_bytes: i64,  // Can be negative if memory was freed

    // All individual timings for distribution analysis
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub raw_times_us: Vec<f64>,
}

/// Configuration for benchmark runs
#[derive(Debug, Clone)]
pub struct BenchConfig {
    pub warmup_iterations: usize,
    pub measurement_iterations: usize,
    pub capture_raw_times: bool,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            measurement_iterations: 100,
            capture_raw_times: true,
        }
    }
}

/// Run a benchmark with full statistics and memory tracking
pub fn run_benchmark<F, T>(
    method: &str,
    variant: &str,
    n: usize,
    config: &BenchConfig,
    mut f: F,
) -> BenchmarkResult
where
    F: FnMut() -> T,
{
    // Warmup phase
    for _ in 0..config.warmup_iterations {
        std::hint::black_box(f());
    }

    // Get memory before measurement
    let mem_before = memory_stats().map(|m| m.physical_mem).unwrap_or(0);
    let mut mem_peak = mem_before;

    // Measurement phase
    let mut times_us: Vec<f64> = Vec::with_capacity(config.measurement_iterations);

    for _ in 0..config.measurement_iterations {
        let start = Instant::now();
        std::hint::black_box(f());
        let elapsed = start.elapsed();
        times_us.push(elapsed.as_secs_f64() * 1_000_000.0);

        // Track peak memory
        if let Some(stats) = memory_stats() {
            if stats.physical_mem > mem_peak {
                mem_peak = stats.physical_mem;
            }
        }
    }

    // Get memory after measurement
    let mem_after = memory_stats().map(|m| m.physical_mem).unwrap_or(0);

    // Sort for percentile calculations
    times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n_times = times_us.len();
    let time_min = times_us[0];
    let time_max = times_us[n_times - 1];
    let time_median = percentile(&times_us, 50.0);
    let time_p25 = percentile(&times_us, 25.0);
    let time_p75 = percentile(&times_us, 75.0);
    let time_mean = times_us.iter().sum::<f64>() / n_times as f64;
    let time_std = (times_us.iter().map(|t| (t - time_mean).powi(2)).sum::<f64>()
                   / (n_times - 1) as f64).sqrt();

    let itr_per_sec = 1_000_000.0 / time_median;

    BenchmarkResult {
        method: method.to_string(),
        variant: variant.to_string(),
        n,
        iterations: config.measurement_iterations,
        time_min_us: time_min,
        time_p25_us: time_p25,
        time_median_us: time_median,
        time_p75_us: time_p75,
        time_max_us: time_max,
        time_mean_us: time_mean,
        time_std_us: time_std,
        itr_per_sec,
        mem_before_bytes: mem_before,
        mem_after_bytes: mem_after,
        mem_peak_bytes: mem_peak,
        mem_alloc_bytes: mem_after as i64 - mem_before as i64,
        raw_times_us: if config.capture_raw_times { times_us } else { vec![] },
    }
}

/// Calculate percentile from sorted data
fn percentile(sorted_data: &[f64], p: f64) -> f64 {
    let n = sorted_data.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted_data[0];
    }

    let idx = (p / 100.0) * (n - 1) as f64;
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    let frac = idx - lower as f64;

    if upper >= n {
        sorted_data[n - 1]
    } else {
        sorted_data[lower] * (1.0 - frac) + sorted_data[upper] * frac
    }
}

/// Save benchmark results to JSON file
pub fn save_results(results: &[BenchmarkResult], path: &str) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(results)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Print benchmark result in a format similar to R's bench::mark()
pub fn print_result(result: &BenchmarkResult) {
    let mem_alloc_kb = result.mem_alloc_bytes as f64 / 1024.0;
    let mem_alloc_str = if mem_alloc_kb.abs() < 1.0 {
        format!("{} B", result.mem_alloc_bytes)
    } else if mem_alloc_kb.abs() < 1024.0 {
        format!("{:.1} KB", mem_alloc_kb)
    } else {
        format!("{:.2} MB", mem_alloc_kb / 1024.0)
    };

    println!(
        "{:20} {:10} n={:>6}  median: {:>10.1} µs  IQR: [{:>8.1}, {:>8.1}]  itr/s: {:>8.1}  mem: {:>10}",
        result.method,
        result.variant,
        result.n,
        result.time_median_us,
        result.time_p25_us,
        result.time_p75_us,
        result.itr_per_sec,
        mem_alloc_str,
    );
}

/// Print summary header
pub fn print_header() {
    println!("{:20} {:10} {:>8}  {:>17}  {:>22}  {:>12}  {:>12}",
        "method", "variant", "n", "median", "IQR", "itr/s", "mem_alloc");
    println!("{}", "-".repeat(100));
}

#[cfg(test)]
mod tests {
    use super::percentile;

    #[test]
    fn test_percentile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&data, 50.0) - 3.0).abs() < 0.01);
        assert!((percentile(&data, 25.0) - 2.0).abs() < 0.01);
        assert!((percentile(&data, 75.0) - 4.0).abs() < 0.01);
    }
}
