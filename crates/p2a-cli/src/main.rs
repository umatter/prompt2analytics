//! p2a - Command-line interface for prompt2analytics
//!
//! Provides direct access to analytics functions and enables reproducible
//! script generation from interactive sessions.

mod commands;
mod output;
mod session;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

use commands::{
    causal, data, discrete, ml, munge, panel, regression, script, spatial, stats, survival,
    timeseries, viz,
};
use output::OutputFormat;
use session::SessionManager;

/// p2a - Analytics from the command line
#[derive(Parser)]
#[command(name = "p2a")]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
#[command(after_help = "\
EXAMPLES:
    # Load a dataset and run OLS regression
    p2a --session analysis.json data load data.csv --name mydata
    p2a --session analysis.json reg ols mydata -y price -x sqft bedrooms

    # Panel fixed effects with clustered SEs
    p2a panel fe mydata -y revenue -x employees --entity firm_id

    # Difference-in-differences
    p2a causal did mydata -y outcome --treat treatment --post post_period

    # K-means clustering
    p2a ml kmeans mydata --cols x1 x2 x3 -k 3

    # Create a scatter plot
    p2a viz scatter mydata -x income -y spending -f plot.png
")]
pub struct Cli {
    /// Output format: text, json, or table
    #[arg(short = 'F', long = "format", global = true, default_value = "text")]
    pub format: output::OutputFormat,

    /// Session file for recording commands (enables reproducibility)
    #[arg(long, global = true)]
    pub session: Option<PathBuf>,

    /// Suppress non-essential output
    #[arg(short, long, global = true)]
    pub quiet: bool,

    /// Enable verbose output for debugging (-v for info, -vv for debug, -vvv for trace)
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    pub verbose: u8,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Data loading and inspection
    #[command(subcommand)]
    Data(data::DataCommands),

    /// Data munging (filter, join, reshape, clean, aggregate)
    #[command(subcommand)]
    Munge(munge::MungeCommands),

    /// Regression analysis
    #[command(subcommand, visible_alias = "reg")]
    Regression(regression::RegressionCommands),

    /// Panel data estimation
    #[command(subcommand)]
    Panel(panel::PanelCommands),

    /// Causal inference methods
    #[command(subcommand)]
    Causal(causal::CausalCommands),

    /// Discrete choice models
    #[command(subcommand)]
    Discrete(discrete::DiscreteCommands),

    /// Statistical tests and descriptive statistics
    #[command(subcommand)]
    Stats(stats::StatsCommands),

    /// Time series analysis
    #[command(subcommand, visible_alias = "ts")]
    Timeseries(timeseries::TimeseriesCommands),

    /// Survival analysis
    #[command(subcommand)]
    Survival(survival::SurvivalCommands),

    /// Spatial econometrics
    #[command(subcommand)]
    Spatial(spatial::SpatialCommands),

    /// Machine learning
    #[command(subcommand, visible_alias = "ml")]
    MachineLearning(ml::MlCommands),

    /// Visualization (output to file)
    #[command(subcommand, visible_alias = "viz")]
    Visualize(viz::VizCommands),

    /// Script generation and session management
    #[command(subcommand)]
    Script(script::ScriptCommands),

    /// Run a smoke test to verify the CLI works correctly
    SmokeTest,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize logger based on verbosity level
    let log_level = match cli.verbose {
        0 => log::LevelFilter::Warn,  // Default: only warnings and errors
        1 => log::LevelFilter::Info,  // -v: info messages
        2 => log::LevelFilter::Debug, // -vv: debug messages
        _ => log::LevelFilter::Trace, // -vvv+: trace messages
    };

    env_logger::Builder::new()
        .filter_level(log_level)
        .format_timestamp_secs()
        .init();

    log::debug!("Verbose mode enabled (level: {:?})", log_level);

    // Initialize session manager if --session is provided
    let mut session_manager = if let Some(session_path) = &cli.session {
        Some(SessionManager::new(session_path.clone())?)
    } else {
        None
    };

    // Execute the command
    let result = match &cli.command {
        Commands::Data(cmd) => data::execute(cmd, &cli.format, session_manager.as_mut()),
        Commands::Munge(cmd) => munge::execute(cmd, &cli.format, session_manager.as_mut()),
        Commands::Regression(cmd) => {
            regression::execute(cmd, &cli.format, session_manager.as_mut())
        }
        Commands::Panel(cmd) => panel::execute(cmd, &cli.format, session_manager.as_mut()),
        Commands::Causal(cmd) => causal::execute(cmd, &cli.format, session_manager.as_mut()),
        Commands::Discrete(cmd) => discrete::execute(cmd, &cli.format, session_manager.as_mut()),
        Commands::Stats(cmd) => stats::execute(cmd, &cli.format, session_manager.as_mut()),
        Commands::Timeseries(cmd) => {
            timeseries::execute(cmd, &cli.format, session_manager.as_mut())
        }
        Commands::Survival(cmd) => survival::execute(cmd, &cli.format, session_manager.as_mut()),
        Commands::Spatial(cmd) => spatial::execute(cmd, &cli.format, session_manager.as_mut()),
        Commands::MachineLearning(cmd) => ml::execute(cmd, &cli.format, session_manager.as_mut()),
        Commands::Visualize(cmd) => viz::execute(cmd, &cli.format, session_manager.as_mut()),
        Commands::Script(cmd) => script::execute(cmd, &cli.format),
        Commands::SmokeTest => run_smoke_test(&cli.format),
    };

    // Save session if recording
    if let Some(ref mut manager) = session_manager {
        manager.save()?;
    }

    result
}

/// Run a smoke test to verify the CLI works correctly.
///
/// This function creates a small synthetic dataset, runs OLS regression,
/// and verifies the output. Used for quick installation verification.
fn run_smoke_test(format: &OutputFormat) -> anyhow::Result<()> {
    use p2a_core::regression::CovarianceType;
    use p2a_core::{Dataset, run_ols};
    use polars::prelude::*;

    println!("Running p2a smoke test...");
    println!();

    // 1. Create a small test dataset
    println!("1. Creating test dataset...");
    let df = df! {
        "y" => [1.1, 2.2, 2.9, 4.1, 5.0, 5.9, 7.2, 7.8, 9.1, 10.0],
        "x1" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "x2" => [0.5, 1.0, 0.7, 1.2, 0.8, 1.5, 1.1, 1.3, 0.9, 1.4],
    }?;
    let dataset = Dataset::new(df);
    println!(
        "   Created dataset: {} rows x {} columns",
        dataset.nrows(),
        dataset.ncols()
    );

    // 2. Run OLS regression
    println!("2. Running OLS regression: y ~ x1 + x2...");
    let result = run_ols(&dataset, "y", &["x1", "x2"], true, CovarianceType::HC1)?;
    println!("   N = {}, R² = {:.4}", result.n_obs, result.r_squared);

    // 3. Verify results
    println!("3. Verifying results...");
    if result.n_obs != 10 {
        anyhow::bail!(
            "Smoke test FAILED: Expected 10 observations, got {}",
            result.n_obs
        );
    }
    if result.r_squared < 0.9 {
        anyhow::bail!(
            "Smoke test FAILED: Expected R² > 0.9, got {:.4}",
            result.r_squared
        );
    }
    if result.coefficients.is_empty() {
        anyhow::bail!("Smoke test FAILED: No coefficients returned");
    }

    // Check that x1 has a positive coefficient close to 1
    let x1_coef = result
        .coefficients
        .iter()
        .find(|c| c.name == "x1")
        .map(|c| c.estimate)
        .unwrap_or(0.0);
    if !(0.8..=1.2).contains(&x1_coef) {
        anyhow::bail!(
            "Smoke test FAILED: Expected x1 coefficient ~1.0, got {:.4}",
            x1_coef
        );
    }

    println!();
    match format {
        OutputFormat::Json => {
            let json = serde_json::json!({
                "status": "success",
                "tests_passed": 4,
                "n_obs": result.n_obs,
                "r_squared": result.r_squared,
            });
            println!("{}", serde_json::to_string_pretty(&json)?);
        }
        _ => {
            println!("========================================");
            println!("Smoke test PASSED!");
            println!("========================================");
            println!("- Dataset creation: OK");
            println!("- OLS regression: OK");
            println!("- Result validation: OK");
            println!("- Coefficient check: OK");
            println!();
            println!("p2a CLI is working correctly.");
        }
    }

    Ok(())
}
