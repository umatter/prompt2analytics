//! OLS Regression Example
//!
//! Demonstrates ordinary least squares regression with:
//! - Robust standard errors (HC0-HC3)
//! - Regression diagnostics
//! - LaTeX export

use p2a_core::{
    data::Dataset,
    export::LatexTableBuilder,
    regression::{CovarianceType, run_diagnostics, run_ols},
    traits::LinearEstimator,
};
use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OLS Regression Example ===\n");

    // Create sample data (Mincer-style wage equation)
    let df = df! {
        "log_wage" => [2.5, 2.8, 3.1, 3.3, 3.5, 3.2, 3.6, 3.8, 3.4, 3.9,
                       3.1, 3.4, 3.7, 3.9, 4.0, 3.5, 3.8, 4.1, 3.7, 4.2],
        "education" => [12.0, 14.0, 16.0, 14.0, 18.0, 12.0, 16.0, 18.0, 14.0, 20.0,
                        12.0, 14.0, 16.0, 16.0, 18.0, 14.0, 16.0, 18.0, 16.0, 20.0],
        "experience" => [5.0, 8.0, 3.0, 10.0, 2.0, 15.0, 6.0, 4.0, 12.0, 1.0,
                         20.0, 14.0, 8.0, 6.0, 5.0, 18.0, 10.0, 7.0, 12.0, 3.0],
        "experience_sq" => [25.0, 64.0, 9.0, 100.0, 4.0, 225.0, 36.0, 16.0, 144.0, 1.0,
                            400.0, 196.0, 64.0, 36.0, 25.0, 324.0, 100.0, 49.0, 144.0, 9.0],
    }?;

    let dataset = Dataset::new(df);
    println!("Dataset: {} observations\n", dataset.nrows());

    // Run OLS with homoskedastic standard errors
    println!("--- OLS with Standard Errors ---");
    let result_std = run_ols(
        &dataset,
        "log_wage",
        &["education", "experience", "experience_sq"],
        true, // include intercept
        CovarianceType::Standard,
    )?;
    println!("{}\n", result_std);

    // Run OLS with heteroskedasticity-robust standard errors (HC1)
    println!("--- OLS with Robust Standard Errors (HC1) ---");
    let result_robust = run_ols(
        &dataset,
        "log_wage",
        &["education", "experience", "experience_sq"],
        true,
        CovarianceType::HC1,
    )?;
    println!("{}\n", result_robust);

    // Compare standard errors
    println!("--- Standard Error Comparison ---");
    let var_names = ["Intercept", "education", "experience", "experience_sq"];
    let se_std = result_std.std_errors();
    let se_robust = result_robust.std_errors();
    println!(
        "{:<15} {:>12} {:>12}",
        "Variable", "Standard SE", "Robust SE"
    );
    println!("{:-<15} {:-<12} {:-<12}", "", "", "");
    for (i, name) in var_names.iter().enumerate() {
        println!("{:<15} {:>12.6} {:>12.6}", name, se_std[i], se_robust[i]);
    }
    println!();

    // Run diagnostics
    println!("--- Regression Diagnostics ---");
    let diagnostics = run_diagnostics(
        &dataset,
        "log_wage",
        &["education", "experience", "experience_sq"],
    )?;
    println!("{}\n", diagnostics);

    // Export to LaTeX
    println!("--- LaTeX Export ---");
    let latex = LatexTableBuilder::new()
        .add_model("(1)", result_robust.clone())
        .caption("Mincer Wage Equation")
        .label("tab:wages")
        .build();
    println!("{}", latex);

    Ok(())
}
