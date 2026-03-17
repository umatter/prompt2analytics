//! Monte Carlo Validation Pipeline
//!
//! Standalone evaluation tool that validates statistical properties of p2a-core
//! methods using Monte Carlo simulation. This is a paper-level evaluation
//! artifact, not part of the core library or CI.
//!
//! Usage:
//!   cargo run --release                          # Full run (1000 sims)
//!   cargo run --release -- --sims 100            # Quick check
//!   cargo run --release -- --category regression # Single category
//!   cargo run --release -- --n 500               # Custom sample size

mod causal;
mod dgp;
mod discrete;
mod framework;
mod hypothesis;
mod panel;
mod regression;

use framework::{McConfig, McResult};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut config = McConfig::default();
    let mut category: Option<String> = None;
    let mut sample_sizes = vec![200, 1000];

    // Parse CLI arguments
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--sims" => {
                i += 1;
                config.n_sims = args[i].parse().expect("Invalid --sims value");
            }
            "--seed" => {
                i += 1;
                config.seed = args[i].parse().expect("Invalid --seed value");
            }
            "--category" => {
                i += 1;
                category = Some(args[i].clone());
            }
            "--n" => {
                i += 1;
                let n: usize = args[i].parse().expect("Invalid --n value");
                sample_sizes = vec![n];
            }
            "--help" | "-h" => {
                eprintln!("Usage: mc-validation [OPTIONS]");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --sims N        Number of MC simulations (default: 1000)");
                eprintln!("  --seed N        Master seed (default: 42)");
                eprintln!("  --category CAT  Run only one category:");
                eprintln!("                  regression, hypothesis, panel, causal, discrete");
                eprintln!("  --n N           Custom sample size (default: 200, 1000)");
                eprintln!("  --help          Show this help");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    println!("=== Monte Carlo Validation Pipeline ===\n");
    println!("Configuration:");
    println!("  Simulations:   {}", config.n_sims);
    println!("  Master seed:   {}", config.seed);
    println!("  Alpha:         {}", config.alpha);
    println!("  MC confidence: {}", config.mc_confidence);
    println!("  Sample sizes:  {:?}", sample_sizes);
    if let Some(ref cat) = category {
        println!("  Category:      {}", cat);
    }
    println!();

    let start = Instant::now();
    let mut all_results: Vec<McResult> = Vec::new();

    for &n in &sample_sizes {
        println!("--- Sample size n = {} ---\n", n);

        let categories: Vec<(&str, Box<dyn Fn(&McConfig, usize) -> Vec<McResult>>)> = vec![
            ("regression", Box::new(|c, n| regression::validate_regression(c, n))),
            ("hypothesis", Box::new(|c, n| hypothesis::validate_hypothesis(c, n))),
            ("panel", Box::new(|c, n| panel::validate_panel(c, n))),
            ("causal", Box::new(|c, n| causal::validate_causal(c, n))),
            ("discrete", Box::new(|c, n| discrete::validate_discrete(c, n))),
        ];

        for (name, runner) in &categories {
            if let Some(ref cat) = category {
                if cat != name {
                    continue;
                }
            }

            let cat_start = Instant::now();
            print!("  {:<15}", name);

            let results = runner(&config, n);
            let n_pass = results.iter().filter(|r| r.within_tolerance).count();
            let n_total = results.len();
            let elapsed = cat_start.elapsed();

            println!(
                " {:>3}/{:>3} pass  ({:.1}s)",
                n_pass,
                n_total,
                elapsed.as_secs_f64()
            );

            all_results.extend(results);
        }
        println!();
    }

    let elapsed = start.elapsed();

    // Summary
    println!("=== Summary ===\n");
    let n_pass = all_results.iter().filter(|r| r.within_tolerance).count();
    let n_total = all_results.len();
    println!("Total: {}/{} pass ({:.1}%)", n_pass, n_total, 100.0 * n_pass as f64 / n_total as f64);
    println!("Time:  {:.1}s\n", elapsed.as_secs_f64());

    // Print failures
    let failures: Vec<&McResult> = all_results.iter().filter(|r| !r.within_tolerance).collect();
    if !failures.is_empty() {
        println!("Failures:");
        for r in &failures {
            println!(
                "  {:<35} {:<25} n={:<6} observed={:.4} expected={:.4} [{:.4}, {:.4}]",
                r.method, r.property, r.n, r.observed, r.expected,
                r.tolerance_lower, r.tolerance_upper
            );
        }
        println!();
    }

    // Print detailed results by property type
    print_summary_by_property(&all_results);

    // Save results
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let output_path = format!("results/mc_validation_{}.json", timestamp);
    std::fs::create_dir_all("results").ok();
    let json = serde_json::to_string_pretty(&all_results).expect("JSON serialization failed");
    std::fs::write(&output_path, &json).expect("Failed to write results");
    println!("Results saved to: {}", output_path);
}

fn print_summary_by_property(results: &[McResult]) {
    let properties = ["ci_coverage", "se_accuracy", "type_i_error", "power", "ci_coverage_negative_control"];

    for prop in &properties {
        let subset: Vec<&McResult> = results.iter().filter(|r| r.property == *prop).collect();
        if subset.is_empty() { continue; }

        let n_pass = subset.iter().filter(|r| r.within_tolerance).count();
        println!("{:<30} {}/{} pass", prop, n_pass, subset.len());

        if *prop == "ci_coverage" {
            let coverages: Vec<f64> = subset.iter().map(|r| r.observed).collect();
            let mean_cov = coverages.iter().sum::<f64>() / coverages.len() as f64;
            println!("  Mean coverage: {:.3} (nominal: 0.950)", mean_cov);
        }
        if *prop == "se_accuracy" {
            let ratios: Vec<f64> = subset.iter().map(|r| r.observed).collect();
            let mean_ratio = ratios.iter().sum::<f64>() / ratios.len() as f64;
            println!("  Mean SE ratio: {:.3} (ideal: 1.000)", mean_ratio);
        }
        if *prop == "type_i_error" {
            let rates: Vec<f64> = subset.iter().map(|r| r.observed).collect();
            let mean_rate = rates.iter().sum::<f64>() / rates.len() as f64;
            println!("  Mean rejection rate: {:.4} (nominal: 0.0500)", mean_rate);
        }
    }
    println!();
}
