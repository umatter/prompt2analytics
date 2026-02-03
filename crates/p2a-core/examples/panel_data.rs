//! Panel Data Analysis Example
//!
//! Demonstrates panel data econometrics with:
//! - Fixed effects estimation
//! - Random effects estimation
//! - Hausman specification test

use p2a_core::{
    data::Dataset,
    econometrics::{run_fixed_effects, run_hausman_test, run_random_effects},
};
use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Panel Data Analysis Example ===\n");

    // Create balanced panel data (5 firms, 4 time periods)
    // Simulating investment equation: investment = f(value, capital)
    let df = df! {
        "firm" => ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C",
                   "D", "D", "D", "D", "E", "E", "E", "E"],
        "year" => [2018, 2019, 2020, 2021, 2018, 2019, 2020, 2021, 2018, 2019, 2020, 2021,
                   2018, 2019, 2020, 2021, 2018, 2019, 2020, 2021],
        "investment" => [120.5, 135.2, 142.1, 155.8, 85.3, 92.7, 88.4, 95.1,
                        210.4, 225.6, 248.3, 262.1, 65.2, 72.4, 78.9, 82.3,
                        180.7, 195.4, 188.2, 205.9],
        "value" => [1500.0, 1650.0, 1580.0, 1720.0, 980.0, 1050.0, 1020.0, 1100.0,
                   2800.0, 3050.0, 3200.0, 3450.0, 750.0, 820.0, 880.0, 920.0,
                   2200.0, 2400.0, 2350.0, 2550.0],
        "capital" => [800.0, 850.0, 920.0, 980.0, 450.0, 480.0, 520.0, 560.0,
                     1500.0, 1600.0, 1720.0, 1850.0, 380.0, 410.0, 450.0, 490.0,
                     1200.0, 1280.0, 1350.0, 1420.0],
    }?;

    let dataset = Dataset::new(df);
    println!(
        "Panel structure: 5 firms × 4 years = {} observations\n",
        dataset.df().height()
    );

    // Fixed Effects Estimation
    println!("--- Fixed Effects (Within Estimator) ---");
    let fe_result = run_fixed_effects(&dataset, "investment", &["value", "capital"], "firm")?;
    println!("{}\n", fe_result);

    // Random Effects Estimation
    println!("--- Random Effects (GLS Estimator) ---");
    let re_result = run_random_effects(&dataset, "investment", &["value", "capital"], "firm")?;
    println!("{}\n", re_result);

    // Compare coefficient estimates
    println!("--- Coefficient Comparison ---");
    let fe_coefs = &fe_result.coefficients;
    let re_coefs = &re_result.coefficients;
    println!("{:<12} {:>12} {:>12}", "Variable", "FE", "RE");
    println!("{:-<12} {:-<12} {:-<12}", "", "", "");
    println!(
        "{:<12} {:>12.6} {:>12.6}",
        "value", fe_coefs[0], re_coefs[1]
    ); // RE has intercept
    println!(
        "{:<12} {:>12.6} {:>12.6}",
        "capital", fe_coefs[1], re_coefs[2]
    );
    println!();

    // Hausman Test
    println!("--- Hausman Specification Test ---");
    println!("H0: Random effects is consistent and efficient (RE preferred)");
    println!("H1: Random effects is inconsistent (FE preferred)\n");

    let hausman = run_hausman_test(&dataset, "investment", &["value", "capital"], "firm")?;
    println!("{}\n", hausman);

    // Interpretation
    if hausman.p_value < 0.05 {
        println!("Conclusion: Reject H0 at 5% level.");
        println!("Fixed effects estimator is preferred due to correlation");
        println!("between unobserved heterogeneity and regressors.");
    } else {
        println!("Conclusion: Cannot reject H0 at 5% level.");
        println!("Random effects estimator is preferred as it is more efficient");
        println!("under the assumption of no correlation with regressors.");
    }

    Ok(())
}
