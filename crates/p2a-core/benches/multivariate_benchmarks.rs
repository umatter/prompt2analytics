//! Multivariate analysis benchmarks for p2a-core
//!
//! Run with: `cargo bench -p p2a-core -- multivariate`

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ndarray::Array2;
use p2a_core::stats::{RotationMethod, ScoresMethod, cancor, factanal, mahalanobis};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Generate synthetic factor analysis data with known factor structure
fn generate_factor_data(n: usize, p: usize, k: usize, seed: u64) -> Array2<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Generate k latent factors
    let mut factors = Array2::zeros((n, k));
    for i in 0..n {
        for j in 0..k {
            factors[[i, j]] = rng.gen_range(-2.0..2.0);
        }
    }

    // Generate loading matrix (each var loads on one factor primarily)
    let mut loadings = Array2::zeros((p, k));
    let vars_per_factor = p / k;
    for j in 0..k {
        let start = j * vars_per_factor;
        let end = if j == k - 1 {
            p
        } else {
            (j + 1) * vars_per_factor
        };
        for i in start..end {
            loadings[[i, j]] = rng.gen_range(0.6..0.9);
            // Add small cross-loadings
            for other_j in 0..k {
                if other_j != j {
                    loadings[[i, other_j]] = rng.gen_range(-0.1..0.1);
                }
            }
        }
    }

    // Generate data: X = F * L' + e
    let mut data = Array2::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            let mut val = 0.0;
            for f in 0..k {
                val += factors[[i, f]] * loadings[[j, f]];
            }
            // Add noise
            val += rng.gen_range(-0.3..0.3);
            data[[i, j]] = val;
        }
    }

    data
}

fn factanal_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("FactorAnalysis");

    // Different configurations: (n_obs, n_vars, n_factors)
    let configs = [(100, 6, 2), (500, 10, 3), (1000, 15, 4), (5000, 20, 5)];

    for (n, p, k) in configs {
        let data = generate_factor_data(n, p, k, 42);
        let label = format!("n{}_p{}_k{}", n, p, k);

        // No rotation benchmark
        group.bench_with_input(BenchmarkId::new("no_rotation", &label), &data, |b, data| {
            b.iter(|| {
                factanal(&data.view(), k, RotationMethod::None, ScoresMethod::None)
                    .expect("Factor analysis should succeed")
            });
        });

        // Varimax rotation benchmark
        group.bench_with_input(BenchmarkId::new("varimax", &label), &data, |b, data| {
            b.iter(|| {
                factanal(&data.view(), k, RotationMethod::Varimax, ScoresMethod::None)
                    .expect("Factor analysis should succeed")
            });
        });

        // Promax rotation benchmark
        group.bench_with_input(BenchmarkId::new("promax", &label), &data, |b, data| {
            b.iter(|| {
                factanal(&data.view(), k, RotationMethod::Promax, ScoresMethod::None)
                    .expect("Factor analysis should succeed")
            });
        });

        // With factor scores (regression method)
        group.bench_with_input(BenchmarkId::new("with_scores", &label), &data, |b, data| {
            b.iter(|| {
                factanal(
                    &data.view(),
                    k,
                    RotationMethod::Varimax,
                    ScoresMethod::Regression,
                )
                .expect("Factor analysis should succeed")
            });
        });
    }

    group.finish();
}

/// Generate correlated data for canonical correlation benchmarks
fn generate_cancor_data(n: usize, p: usize, q: usize, seed: u64) -> (Array2<f64>, Array2<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Generate common latent factor
    let mut common = Vec::with_capacity(n);
    for _ in 0..n {
        common.push(rng.gen_range(-2.0..2.0));
    }

    // Generate X with correlation to common factor
    let mut x = Array2::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            let loading = 0.8 - 0.1 * (j as f64); // Decreasing correlation
            x[[i, j]] = loading * common[i] + rng.gen_range(-0.5..0.5);
        }
    }

    // Generate Y with correlation to common factor
    let mut y = Array2::zeros((n, q));
    for i in 0..n {
        for j in 0..q {
            let loading = 0.9 - 0.15 * (j as f64); // Decreasing correlation
            y[[i, j]] = loading * common[i] + rng.gen_range(-0.5..0.5);
        }
    }

    (x, y)
}

fn cancor_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("CanonicalCorrelation");

    // Different sample sizes with fixed variable counts (5 X vars, 3 Y vars)
    for n in [100, 1000, 10000, 100000] {
        let (x, y) = generate_cancor_data(n, 5, 3, 42);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n{}", n)),
            &(x, y),
            |b, (x, y)| {
                b.iter(|| cancor(&x.view(), &y.view(), true, true).expect("Cancor should succeed"));
            },
        );
    }

    group.finish();
}

fn cancor_variable_scaling_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("CanonicalCorrelation_VarScaling");

    // Different variable configurations with n=1000
    let n = 1000;
    let configs = [(2, 2), (5, 3), (10, 5), (20, 10)];

    for (p, q) in configs {
        let (x, y) = generate_cancor_data(n, p, q, 42);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("p{}_q{}", p, q)),
            &(x, y),
            |b, (x, y)| {
                b.iter(|| cancor(&x.view(), &y.view(), true, true).expect("Cancor should succeed"));
            },
        );
    }

    group.finish();
}

/// Generate multivariate data for Mahalanobis distance benchmarks
fn generate_mahalanobis_data(n: usize, p: usize, seed: u64) -> Array2<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut data = Array2::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            // Add some correlation between adjacent variables
            if j == 0 {
                data[[i, j]] = rng.gen_range(-3.0..3.0);
            } else {
                data[[i, j]] = 0.5 * data[[i, j - 1]] + rng.gen_range(-2.0..2.0);
            }
        }
    }

    data
}

fn mahalanobis_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mahalanobis");

    // Different sample sizes with 5 variables
    for n in [100, 1000, 10000, 100000] {
        let data = generate_mahalanobis_data(n, 5, 42);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n{}_p5", n)),
            &data,
            |b, data| {
                b.iter(|| {
                    mahalanobis(data.view(), None, None, false).expect("Mahalanobis should succeed")
                });
            },
        );
    }

    group.finish();
}

fn mahalanobis_variable_scaling_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mahalanobis_VarScaling");

    // Different variable counts with n=1000
    let n = 1000;
    for p in [2, 5, 10, 20, 50] {
        let data = generate_mahalanobis_data(n, p, 42);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n{}_p{}", n, p)),
            &data,
            |b, data| {
                b.iter(|| {
                    mahalanobis(data.view(), None, None, false).expect("Mahalanobis should succeed")
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    factanal_benchmark,
    cancor_benchmark,
    cancor_variable_scaling_benchmark,
    mahalanobis_benchmark,
    mahalanobis_variable_scaling_benchmark
);
criterion_main!(benches);
