//! String and regex operation benchmarks for p2a-core
//!
//! Benchmarks string cleaning and regex operations against R's stringi/stringr.
//! Run with: `cargo bench -p p2a-core -- string`

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p2a_core::Dataset;
use p2a_core::data::munging::{
    regex_count, regex_extract, regex_replace, replace, str_concat, str_length, str_split,
    str_substring, to_lowercase, to_uppercase, trim,
};
use polars::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Generate synthetic string data for benchmarks
/// Creates: id, text (various formats), email, phone, code
fn generate_string_data(n: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let id: Vec<i64> = (1..=n as i64).collect();

    // Text with leading/trailing whitespace (for trim tests)
    let text: Vec<String> = (0..n)
        .map(|_| {
            let spaces_before = " ".repeat(rng.gen_range(0..5));
            let spaces_after = " ".repeat(rng.gen_range(0..5));
            let word_len = rng.gen_range(5..20);
            let word: String = (0..word_len)
                .map(|_| rng.gen_range(b'a'..=b'z') as char)
                .collect();
            format!("{}{}{}", spaces_before, word, spaces_after)
        })
        .collect();

    // Mixed case text (for case conversion tests)
    let mixed_case: Vec<String> = (0..n)
        .map(|_| {
            let len = rng.gen_range(10..50);
            (0..len)
                .map(|_| {
                    if rng.gen_bool(0.5) {
                        rng.gen_range(b'A'..=b'Z') as char
                    } else {
                        rng.gen_range(b'a'..=b'z') as char
                    }
                })
                .collect()
        })
        .collect();

    // Email-like strings (for regex tests)
    let email: Vec<String> = (0..n)
        .map(|i| {
            let domains = [
                "gmail.com",
                "yahoo.com",
                "outlook.com",
                "example.org",
                "test.net",
            ];
            let domain = domains[rng.gen_range(0..domains.len())];
            format!("user{}@{}", i, domain)
        })
        .collect();

    // Phone-like strings (for regex extract tests)
    let phone: Vec<String> = (0..n)
        .map(|_| {
            let area = rng.gen_range(100..999);
            let prefix = rng.gen_range(100..999);
            let line = rng.gen_range(1000..9999);
            if rng.gen_bool(0.5) {
                format!("({}) {}-{}", area, prefix, line)
            } else {
                format!("{}-{}-{}", area, prefix, line)
            }
        })
        .collect();

    // Code-like strings with patterns (for regex count tests)
    let code: Vec<String> = (0..n)
        .map(|_| {
            let n_vars = rng.gen_range(1..10);
            (0..n_vars)
                .map(|_| {
                    let var_len = rng.gen_range(3..8);
                    let var: String = (0..var_len)
                        .map(|_| rng.gen_range(b'a'..=b'z') as char)
                        .collect();
                    format!("let {} = {};", var, rng.gen_range(0..100))
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect();

    // Delimited strings (for split tests)
    let delimited: Vec<String> = (0..n)
        .map(|_| {
            let n_parts = rng.gen_range(2..8);
            (0..n_parts)
                .map(|_| {
                    let len = rng.gen_range(3..10);
                    (0..len)
                        .map(|_| rng.gen_range(b'a'..=b'z') as char)
                        .collect::<String>()
                })
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect();

    // First and last name columns (for concat tests)
    let first_name: Vec<String> = (0..n)
        .map(|_| {
            let names = [
                "John", "Jane", "Bob", "Alice", "Charlie", "Diana", "Eve", "Frank",
            ];
            names[rng.gen_range(0..names.len())].to_string()
        })
        .collect();

    let last_name: Vec<String> = (0..n)
        .map(|_| {
            let names = [
                "Smith", "Johnson", "Williams", "Brown", "Jones", "Davis", "Miller",
            ];
            names[rng.gen_range(0..names.len())].to_string()
        })
        .collect();

    let df = DataFrame::new(vec![
        Column::new("id".into(), id),
        Column::new("text".into(), text),
        Column::new("mixed_case".into(), mixed_case),
        Column::new("email".into(), email),
        Column::new("phone".into(), phone),
        Column::new("code".into(), code),
        Column::new("delimited".into(), delimited),
        Column::new("first_name".into(), first_name),
        Column::new("last_name".into(), last_name),
    ])
    .expect("Failed to create DataFrame");

    Dataset::new(df)
}

// =============================================================================
// TRIM BENCHMARKS
// =============================================================================

fn trim_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("str_trim");

    for n in [10_000, 100_000, 1_000_000] {
        let dataset = generate_string_data(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, ds| {
            b.iter(|| trim(ds, Some(&["text"])));
        });
    }

    group.finish();
}

// =============================================================================
// CASE CONVERSION BENCHMARKS
// =============================================================================

fn to_lowercase_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("str_to_lowercase");

    for n in [10_000, 100_000, 1_000_000] {
        let dataset = generate_string_data(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, ds| {
            b.iter(|| to_lowercase(ds, "mixed_case"));
        });
    }

    group.finish();
}

fn to_uppercase_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("str_to_uppercase");

    for n in [10_000, 100_000, 1_000_000] {
        let dataset = generate_string_data(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, ds| {
            b.iter(|| to_uppercase(ds, "mixed_case"));
        });
    }

    group.finish();
}

// =============================================================================
// REPLACE BENCHMARKS
// =============================================================================

fn replace_literal_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("str_replace_literal");

    for n in [10_000, 100_000, 1_000_000] {
        let dataset = generate_string_data(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, ds| {
            b.iter(|| replace(ds, "email", "gmail.com", "google.com"));
        });
    }

    group.finish();
}

fn regex_replace_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("str_regex_replace");

    for n in [10_000, 100_000, 1_000_000] {
        let dataset = generate_string_data(n, 42);

        // Replace domain in email addresses
        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, ds| {
            b.iter(|| regex_replace(ds, "email", r"@\w+\.com", "@replaced.com"));
        });
    }

    group.finish();
}

// =============================================================================
// REGEX EXTRACT BENCHMARKS
// =============================================================================

fn regex_extract_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("str_regex_extract");

    for n in [10_000, 100_000, 1_000_000] {
        let dataset = generate_string_data(n, 42);

        // Extract area code from phone numbers
        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, ds| {
            b.iter(|| regex_extract(ds, "phone", r"\(?(\d{3})\)?", "area_code", 1));
        });
    }

    group.finish();
}

// =============================================================================
// REGEX COUNT BENCHMARKS
// =============================================================================

fn regex_count_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("str_regex_count");

    for n in [10_000, 100_000, 1_000_000] {
        let dataset = generate_string_data(n, 42);

        // Count "let" keywords in code
        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, ds| {
            b.iter(|| regex_count(ds, "code", r"let\s+\w+", "let_count"));
        });
    }

    group.finish();
}

// =============================================================================
// STRING SPLIT BENCHMARKS
// =============================================================================

fn str_split_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("str_split");

    for n in [10_000, 100_000, 1_000_000] {
        let dataset = generate_string_data(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, ds| {
            b.iter(|| str_split(ds, "delimited", ",", Some(3), "part"));
        });
    }

    group.finish();
}

// =============================================================================
// STRING CONCAT BENCHMARKS
// =============================================================================

fn str_concat_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("str_concat");

    for n in [10_000, 100_000, 1_000_000] {
        let dataset = generate_string_data(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, ds| {
            b.iter(|| str_concat(ds, &["first_name", "last_name"], "full_name", Some(" ")));
        });
    }

    group.finish();
}

// =============================================================================
// STRING LENGTH BENCHMARKS
// =============================================================================

fn str_length_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("str_length");

    for n in [10_000, 100_000, 1_000_000] {
        let dataset = generate_string_data(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, ds| {
            b.iter(|| str_length(ds, "text", "text_len"));
        });
    }

    group.finish();
}

// =============================================================================
// STRING SUBSTRING BENCHMARKS
// =============================================================================

fn str_substring_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("str_substring");

    for n in [10_000, 100_000, 1_000_000] {
        let dataset = generate_string_data(n, 42);

        // Extract first 10 characters
        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, ds| {
            b.iter(|| str_substring(ds, "mixed_case", 0, Some(10)));
        });
    }

    group.finish();
}

// =============================================================================
// CRITERION SETUP
// =============================================================================

criterion_group!(trim_benches, trim_benchmark);

criterion_group!(case_benches, to_lowercase_benchmark, to_uppercase_benchmark);

criterion_group!(
    replace_benches,
    replace_literal_benchmark,
    regex_replace_benchmark
);

criterion_group!(
    extract_benches,
    regex_extract_benchmark,
    regex_count_benchmark
);

criterion_group!(
    manipulation_benches,
    str_split_benchmark,
    str_concat_benchmark,
    str_length_benchmark,
    str_substring_benchmark
);

criterion_main!(
    trim_benches,
    case_benches,
    replace_benches,
    extract_benches,
    manipulation_benches
);
