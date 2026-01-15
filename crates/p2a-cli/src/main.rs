//! p2a - Command-line interface for prompt2analytics
//!
//! Provides direct access to analytics functions and enables reproducible
//! script generation from interactive sessions.

mod commands;
mod output;
mod session;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

use commands::{data, munge, regression, panel, causal, discrete, timeseries, survival, ml, viz, script};
use session::SessionManager;

/// p2a - Analytics from the command line
#[derive(Parser)]
#[command(name = "p2a")]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
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

    /// Time series analysis
    #[command(subcommand, visible_alias = "ts")]
    Timeseries(timeseries::TimeseriesCommands),

    /// Survival analysis
    #[command(subcommand)]
    Survival(survival::SurvivalCommands),

    /// Machine learning
    #[command(subcommand, visible_alias = "ml")]
    MachineLearning(ml::MlCommands),

    /// Visualization (output to file)
    #[command(subcommand, visible_alias = "viz")]
    Visualize(viz::VizCommands),

    /// Script generation and session management
    #[command(subcommand)]
    Script(script::ScriptCommands),
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

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
        Commands::Regression(cmd) => regression::execute(cmd, &cli.format, session_manager.as_mut()),
        Commands::Panel(cmd) => panel::execute(cmd, &cli.format, session_manager.as_mut()),
        Commands::Causal(cmd) => causal::execute(cmd, &cli.format, session_manager.as_mut()),
        Commands::Discrete(cmd) => discrete::execute(cmd, &cli.format, session_manager.as_mut()),
        Commands::Timeseries(cmd) => timeseries::execute(cmd, &cli.format, session_manager.as_mut()),
        Commands::Survival(cmd) => survival::execute(cmd, &cli.format, session_manager.as_mut()),
        Commands::MachineLearning(cmd) => ml::execute(cmd, &cli.format, session_manager.as_mut()),
        Commands::Visualize(cmd) => viz::execute(cmd, &cli.format, session_manager.as_mut()),
        Commands::Script(cmd) => script::execute(cmd, &cli.format),
    };

    // Save session if recording
    if let Some(ref mut manager) = session_manager {
        manager.save()?;
    }

    result
}
