//! p2a - Command-line interface for prompt2analytics
//!
//! Provides direct access to analytics functions and enables reproducible
//! script generation from interactive sessions.

mod commands;
mod memory;
mod output;
mod progress;
mod session;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

use commands::{
    causal, chat, data, discrete, ml, munge, panel, regression, script, spatial, stats, survival,
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
    #[arg(long, global = true)]
    pub quiet: bool,

    /// Enable verbose logging on stderr (--verbose for info, twice for debug, three times for trace)
    #[arg(long, global = true, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Maximum memory usage in MB (warns when exceeded, 0 = unlimited)
    #[arg(long = "max-memory", global = true, default_value = "0")]
    pub max_memory_mb: usize,

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

    /// Interactive chat with a p2a-mcp server (natural language analytics)
    Chat(chat::ChatArgs),

    /// Run a smoke test to verify the CLI works correctly
    SmokeTest,
}

/// Global flags of `Cli`: (long name, short, takes a value). Kept in sync by
/// `global_flag_table_matches_cli`.
const GLOBAL_FLAGS: &[(&str, Option<char>, bool)] = &[
    ("format", Some('F'), true),
    ("session", None, true),
    ("quiet", None, false),
    ("verbose", None, false),
    ("max-memory", None, true),
];

/// The number of tokens a global flag of [`GLOBAL_FLAGS`] occupies when `tok`
/// is one, or `None` when `tok` is not a global flag.
///
/// `--<long>` takes 2 tokens if the flag takes a value, else 1; `--<long>=<v>`
/// takes 1 (value-taking flags only); `-F` takes 2; `-F<v>` and `-F=<v>` take 1.
fn global_flag_width(tok: &str) -> Option<usize> {
    if let Some(rest) = tok.strip_prefix("--") {
        let (long, inline_value) = match rest.split_once('=') {
            Some((long, _)) => (long, true),
            None => (rest, false),
        };
        let &(_, _, takes_value) = GLOBAL_FLAGS.iter().find(|(l, _, _)| *l == long)?;
        return match (takes_value, inline_value) {
            (true, false) => Some(2),
            (true, true) | (false, false) => Some(1),
            (false, true) => None,
        };
    }
    let rest = tok.strip_prefix('-')?;
    let mut chars = rest.chars();
    let short = chars.next()?;
    let &(_, _, takes_value) = GLOBAL_FLAGS.iter().find(|(_, s, _)| *s == Some(short))?;
    match (takes_value, chars.as_str().is_empty()) {
        (true, true) => Some(2),
        (true, false) | (false, true) => Some(1),
        (false, false) => None,
    }
}

/// Extract the command-portion argv (category, subcommand, and args) from the
/// raw process arguments, as the session records it for script export.
///
/// Drops the program name and every global flag before the subcommand token
/// (all forms of [`GLOBAL_FLAGS`]: `--format x`, `--format=x`, `-F x`, `-Fx`,
/// `-F=x`, `--session`, `--quiet`, `--verbose`, `--max-memory`). After it,
/// every token is kept as typed, so a global typed after the subcommand
/// (`-F json`, `--quiet`) is replayed, except `--session <v>` and
/// `--session=<v>`, which are dropped anywhere before a `--` token (everything
/// after `--` is copied verbatim).
///
/// - A trailing `--session` goes because clap keeps the deeper value when the
///   parent and the subcommand both set a global (clap_builder
///   `parser/arg_matcher.rs:66-71`): a recorded `--session OLD` would override
///   the replay script's own `--session "$SESSION_FILE"` and redirect the
///   replay's recording into the original session file.
/// - No subcommand flag is spelled like a global: `cli_debug_asserts` (clap's
///   short/long uniqueness asserts) and `no_subcommand_arg_shadows_a_global`
///   (ids, which the asserts cannot see) guarantee it, so a kept `-q` or `-v`
///   is always the subcommand's own option.
/// - No option accepts hyphen values (`allow_hyphen_values`), so a token that
///   looks like a global flag is never another option's value.
fn command_argv(raw: &[String]) -> Vec<String> {
    let mut i = 1; // skip the program name
    while i < raw.len() {
        match global_flag_width(&raw[i]) {
            Some(width) => i += width,
            None => break,
        }
    }

    let mut out = Vec::new();
    let mut rest = raw.iter().skip(i);
    while let Some(tok) = rest.next() {
        if tok == "--" {
            out.push(tok.clone());
            out.extend(rest.cloned());
            break;
        }
        if tok == "--session" {
            rest.next(); // and its value
        } else if !tok.starts_with("--session=") {
            out.push(tok.clone());
        }
    }
    out
}

fn main() -> anyhow::Result<()> {
    let raw_args: Vec<String> = std::env::args().collect();
    let cli = Cli::parse();

    // Initialize logger based on verbosity level
    let log_level = match cli.verbose {
        0 => log::LevelFilter::Warn,  // Default: only warnings and errors
        1 => log::LevelFilter::Info,  // --verbose: info messages
        2 => log::LevelFilter::Debug, // twice: debug messages
        _ => log::LevelFilter::Trace, // three times or more: trace messages
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
    if let Some(ref mut manager) = session_manager {
        manager.start_command();
    }

    // Execute the command
    let result = match &cli.command {
        Commands::Data(cmd) => data::execute(cmd, &cli.format, cli.quiet, session_manager.as_mut()),
        Commands::Munge(cmd) => {
            munge::execute(cmd, &cli.format, cli.quiet, session_manager.as_mut())
        }
        Commands::Regression(cmd) => {
            regression::execute(cmd, &cli.format, cli.quiet, session_manager.as_mut())
        }
        Commands::Panel(cmd) => {
            panel::execute(cmd, &cli.format, cli.quiet, session_manager.as_mut())
        }
        Commands::Causal(cmd) => {
            causal::execute(cmd, &cli.format, cli.quiet, session_manager.as_mut())
        }
        Commands::Discrete(cmd) => {
            discrete::execute(cmd, &cli.format, cli.quiet, session_manager.as_mut())
        }
        Commands::Stats(cmd) => {
            stats::execute(cmd, &cli.format, cli.quiet, session_manager.as_mut())
        }
        Commands::Timeseries(cmd) => {
            timeseries::execute(cmd, &cli.format, cli.quiet, session_manager.as_mut())
        }
        Commands::Survival(cmd) => {
            survival::execute(cmd, &cli.format, cli.quiet, session_manager.as_mut())
        }
        Commands::Spatial(cmd) => {
            spatial::execute(cmd, &cli.format, cli.quiet, session_manager.as_mut())
        }
        Commands::MachineLearning(cmd) => {
            ml::execute(cmd, &cli.format, cli.quiet, session_manager.as_mut())
        }
        Commands::Visualize(cmd) => {
            viz::execute(cmd, &cli.format, cli.quiet, session_manager.as_mut())
        }
        Commands::Script(cmd) => script::execute(cmd, &cli.format),
        Commands::Chat(args) => {
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(chat::run(args));
        }
        Commands::SmokeTest => run_smoke_test(&cli.format),
    };

    // Record this invocation centrally (every analytics command; meta commands
    // like `script`/`chat`/`smoke-test` are not recorded) and save the session.
    if let Some(ref mut manager) = session_manager {
        let is_meta = matches!(
            cli.command,
            Commands::Script(_) | Commands::Chat(_) | Commands::SmokeTest
        );
        if !is_meta {
            let argv = command_argv(&raw_args);
            let success = result.is_ok() && !output::error_was_reported();
            let error = result.as_ref().err().map(|e| e.to_string());
            manager.record_invocation(argv, success, error);
        }
        manager.save()?;
    }

    // Exit non-zero on any error — including "soft" errors that a handler
    // printed via print_error before returning Ok(()) — so scripts and CI can
    // detect failures. Route hard errors through the format-aware printer too.
    match result {
        Ok(()) => {
            if output::error_was_reported() {
                std::process::exit(1);
            }
            Ok(())
        }
        Err(e) => {
            output::print_error(&e.to_string(), &cli.format);
            std::process::exit(1);
        }
    }
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

#[cfg(test)]
mod cli_definition_tests {
    use super::{Cli, Commands};
    use crate::commands::{
        munge::MungeCommands, panel::PanelCommands, stats::StatsCommands,
        timeseries::TimeseriesCommands,
    };
    use clap::{CommandFactory, Parser};

    fn parse(args: &[&str]) -> Cli {
        Cli::try_parse_from(std::iter::once("p2a").chain(args.iter().copied()))
            .unwrap_or_else(|e| panic!("{e}"))
    }

    #[test]
    fn cli_debug_asserts() {
        Cli::command().debug_assert();
    }

    #[test]
    fn subcommand_short_q_and_v_bind_to_the_subcommand() {
        let cli = parse(&["ts", "arima", "d", "--col", "y", "-q", "2"]);
        assert!(!cli.quiet);
        assert!(matches!(
            cli.command,
            Commands::Timeseries(TimeseriesCommands::Arima { ma: 2, .. })
        ));
        assert!(matches!(
            parse(&["ts", "varma", "d", "--cols", "a", "b", "-q", "2"]).command,
            Commands::Timeseries(TimeseriesCommands::Varma { ma: 2, .. })
        ));
        assert!(matches!(
            parse(&["ts", "garch", "d", "--col", "r", "-p", "1", "-q", "2"]).command,
            Commands::Timeseries(TimeseriesCommands::Garch { garch: 2, .. })
        ));
        let cli = parse(&[
            "panel",
            "unit-root",
            "d",
            "-v",
            "gdp",
            "--unit",
            "c",
            "--time",
            "t",
        ]);
        assert_eq!(cli.verbose, 0);
        assert!(
            matches!(cli.command, Commands::Panel(PanelCommands::UnitRoot { ref var, .. }) if var == "gdp")
        );
        assert!(matches!(
            parse(&["stats", "friedman", "d", "-v", "score", "-g", "trt", "-b", "subj"]).command,
            Commands::Stats(StatsCommands::Friedman { ref value, .. }) if value == "score"
        ));
        assert!(matches!(
            parse(&["munge", "filter", "d", "-c", "age", "-o", "gt", "-v", "30"]).command,
            Commands::Munge(MungeCommands::Filter { ref value, .. }) if value == "30"
        ));
        assert!(matches!(
            parse(&["munge", "pivot", "d", "-i", "id", "-o", "var", "-v", "val"]).command,
            Commands::Munge(MungeCommands::Pivot { ref values, .. }) if values == "val"
        ));
        assert!(matches!(
            parse(&["munge", "melt", "d", "-i", "id", "-v", "jan", "feb"]).command,
            Commands::Munge(MungeCommands::Melt { ref value_vars, .. }) if value_vars == &["jan", "feb"]
        ));
    }

    #[test]
    fn munge_mutate_name_has_its_own_short() {
        assert!(matches!(
            parse(&["munge", "mutate", "d", "-n", "total", "-e", "add:a:b", "-N", "out"]).command,
            Commands::Munge(MungeCommands::Mutate { ref new_col, name: Some(ref n), .. })
                if new_col == "total" && n == "out"
        ));
    }

    #[test]
    fn global_quiet_and_verbose_are_long_only() {
        let cli = parse(&[
            "ts",
            "arima",
            "d",
            "--col",
            "y",
            "--quiet",
            "--verbose",
            "--verbose",
        ]);
        assert!(cli.quiet);
        assert_eq!(cli.verbose, 2);
        assert!(parse(&["--quiet", "data", "list"]).quiet);
        assert!(Cli::try_parse_from(["p2a", "-q", "data", "list"]).is_err());
        assert!(Cli::try_parse_from(["p2a", "-v", "data", "list"]).is_err());
    }

    /// clap skips propagating a global whose id the subcommand already defines, so debug_assert
    /// cannot see a shadowing id (CLI-2). Walk the unbuilt tree, which holds local args only.
    #[test]
    fn no_subcommand_arg_shadows_a_global() {
        fn walk(
            cmd: &clap::Command,
            path: &str,
            globals: &[clap::Arg],
            bad: &mut Vec<String>,
            seen: &mut usize,
        ) {
            for sc in cmd.get_subcommands() {
                let p = format!("{path} {}", sc.get_name());
                for a in sc.get_arguments() {
                    *seen += 1;
                    for g in globals {
                        let same_long = a.get_long().is_some() && a.get_long() == g.get_long();
                        let same_short = a.get_short().is_some() && a.get_short() == g.get_short();
                        if a.get_id() == g.get_id() || same_long || same_short {
                            bad.push(format!(
                                "{p}: `{}` shadows the global `{}`",
                                a.get_id(),
                                g.get_id()
                            ));
                        }
                    }
                }
                walk(sc, &p, globals, bad, seen);
            }
        }
        let root = Cli::command();
        let globals: Vec<clap::Arg> = root
            .get_arguments()
            .filter(|a| a.is_global_set())
            .cloned()
            .collect();
        let (mut bad, mut seen) = (Vec::new(), 0usize);
        walk(&root, "p2a", &globals, &mut bad, &mut seen);
        // about 680 today
        assert!(
            seen > 300,
            "walk saw only {seen} subcommand args; are subcommands deferred?"
        );
        assert!(bad.is_empty(), "{bad:#?}");
    }

    #[test]
    fn data_save_file_format_and_global_format() {
        use crate::commands::data::DataCommands;
        use crate::output::OutputFormat;
        let cli = parse(&[
            "data",
            "save",
            "d",
            "-o",
            "out.dat",
            "--file-format",
            "csv",
            "-F",
            "json",
        ]);
        assert!(matches!(cli.format, OutputFormat::Json));
        assert!(
            matches!(cli.command, Commands::Data(DataCommands::Save { ref file_format, .. }) if file_format.as_deref() == Some("csv"))
        );
        let cli = parse(&["data", "save", "d", "-o", "x.csv", "-f", "parquet"]);
        assert!(matches!(cli.format, OutputFormat::Text));
        assert!(
            matches!(cli.command, Commands::Data(DataCommands::Save { ref file_format, .. }) if file_format.as_deref() == Some("parquet"))
        );
        assert!(matches!(
            parse(&["data", "save", "d", "-o", "x.csv"]).command,
            Commands::Data(DataCommands::Save {
                file_format: None,
                ..
            })
        ));
    }

    #[test]
    fn data_load_na_strings_parses() {
        use crate::commands::data::DataCommands;
        let na = |args: &[&str]| match parse(args).command {
            Commands::Data(DataCommands::Load { na_strings, .. }) => na_strings,
            _ => unreachable!(),
        };
        assert_eq!(na(&["data", "load", "f.csv"]), None);
        assert_eq!(
            na(&["data", "load", "f.csv", "--na-strings", "NA,,."]),
            Some(vec!["NA".into(), "".into(), ".".into()])
        );
        assert_eq!(
            na(&[
                "data",
                "load",
                "f.csv",
                "--na-strings",
                ".",
                "--na-strings",
                "NA"
            ]),
            Some(vec![".".into(), "NA".into()])
        );
        assert_eq!(
            na(&["data", "load", "f.csv", "--na-strings=-99,."]),
            Some(vec!["-99".into(), ".".into()])
        );
    }
}

#[cfg(test)]
mod command_argv_tests {
    use super::command_argv;

    fn argv(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn strips_leading_globals() {
        let raw = argv(&[
            "p2a",
            "--session",
            "s.json",
            "reg",
            "ols",
            "sales",
            "-y",
            "price",
        ]);
        assert_eq!(
            command_argv(&raw),
            argv(&["reg", "ols", "sales", "-y", "price"])
        );
    }

    #[test]
    fn preserves_command_values_that_look_like_flags_values() {
        // A column literally named after a flag value is kept.
        let raw = argv(&["p2a", "reg", "ols", "d", "-x", "price"]);
        assert_eq!(
            command_argv(&raw),
            argv(&["reg", "ols", "d", "-x", "price"])
        );
    }

    #[test]
    fn keeps_globals_after_the_subcommand_except_session() {
        let raw = argv(&[
            "p2a",
            "data",
            "load",
            "f.csv",
            "--session",
            "s.json",
            "-F",
            "json",
            "--quiet",
            "--name",
            "d",
        ]);
        assert_eq!(
            command_argv(&raw),
            argv(&[
                "data", "load", "f.csv", "-F", "json", "--quiet", "--name", "d"
            ])
        );
    }

    #[test]
    fn strips_every_leading_global_form() {
        let raw = argv(&[
            "p2a",
            "--session=s.json",
            "--verbose",
            "--verbose",
            "--max-memory",
            "512",
            "-Fjson",
            "--format=table",
            "--quiet",
            "stats",
            "describe",
            "d",
        ]);
        assert_eq!(command_argv(&raw), argv(&["stats", "describe", "d"]));
        let raw = argv(&["p2a", "--max-memory=64", "-F=json", "reg", "ols", "d"]);
        assert_eq!(command_argv(&raw), argv(&["reg", "ols", "d"]));
    }

    #[test]
    fn strips_session_anywhere_before_double_dash() {
        let raw = argv(&[
            "p2a",
            "reg",
            "--session",
            "s.json",
            "ols",
            "d",
            "--session=t.json",
            "-y",
            "y",
        ]);
        assert_eq!(command_argv(&raw), argv(&["reg", "ols", "d", "-y", "y"]));
        let raw = argv(&["p2a", "data", "describe", "--", "--session"]);
        assert_eq!(
            command_argv(&raw),
            argv(&["data", "describe", "--", "--session"])
        );
    }

    #[test]
    fn keeps_subcommand_short_q_and_v() {
        let raw = argv(&[
            "p2a",
            "--session",
            "s.json",
            "ts",
            "arima",
            "d",
            "--col",
            "y",
            "-p",
            "1",
            "-d",
            "1",
            "-q",
            "2",
        ]);
        assert_eq!(
            command_argv(&raw),
            argv(&[
                "ts", "arima", "d", "--col", "y", "-p", "1", "-d", "1", "-q", "2"
            ])
        );
        let raw = argv(&[
            "p2a",
            "panel",
            "unit-root",
            "d",
            "-v",
            "gdp",
            "--unit",
            "c",
            "--time",
            "t",
        ]);
        assert_eq!(command_argv(&raw), raw[1..].to_vec());
    }

    #[test]
    fn keeps_data_save_file_format() {
        let raw = argv(&[
            "p2a",
            "--session",
            "s",
            "data",
            "save",
            "d",
            "-o",
            "out.dat",
            "--file-format",
            "csv",
        ]);
        assert_eq!(
            command_argv(&raw),
            argv(&["data", "save", "d", "-o", "out.dat", "--file-format", "csv"])
        );
    }

    #[test]
    fn global_flag_table_matches_cli() {
        use clap::CommandFactory;
        let mut from_cli: Vec<(String, Option<char>, bool)> = super::Cli::command()
            .get_arguments()
            .filter(|a| a.is_global_set())
            .map(|a| {
                (
                    a.get_long().expect("globals have a long name").to_string(),
                    a.get_short(),
                    a.get_action().takes_values(),
                )
            })
            .collect();
        let mut table: Vec<(String, Option<char>, bool)> = super::GLOBAL_FLAGS
            .iter()
            .map(|&(l, s, v)| (l.to_string(), s, v))
            .collect();
        from_cli.sort();
        table.sort();
        assert_eq!(table, from_cli);
    }
}
