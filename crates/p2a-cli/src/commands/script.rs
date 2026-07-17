//! Script generation and session management commands

use clap::Subcommand;
use std::path::PathBuf;

use crate::output::{OutputFormat, print_error, print_message};
use crate::session::Session;

#[derive(Subcommand)]
pub enum ScriptCommands {
    /// Export session to a reproducible bash script
    Export {
        /// Session file to export
        session_file: PathBuf,

        /// Output script file
        #[arg(short, long)]
        output: PathBuf,

        /// Include comments in generated script
        #[arg(long, default_value = "true")]
        comments: bool,
    },

    /// Show session history
    History {
        /// Session file to display
        session_file: PathBuf,
    },

    /// Run a p2a script
    Run {
        /// Script file to execute
        script_file: PathBuf,
    },
}

pub fn execute(cmd: &ScriptCommands, format: &OutputFormat) -> anyhow::Result<()> {
    match cmd {
        ScriptCommands::Export {
            session_file,
            output,
            comments,
        } => execute_export(session_file, output, *comments, format),
        ScriptCommands::History { session_file } => execute_history(session_file, format),
        ScriptCommands::Run { script_file } => execute_run(script_file, format),
    }
}

fn execute_export(
    session_file: &PathBuf,
    output: &PathBuf,
    comments: bool,
    format: &OutputFormat,
) -> anyhow::Result<()> {
    // Load the session
    let session = match Session::load(session_file) {
        Ok(s) => s,
        Err(e) => {
            print_error(&format!("Failed to load session: {}", e), format);
            return Ok(());
        }
    };

    // Generate the bash script
    let mut script = String::new();

    // Shebang and header
    script.push_str("#!/bin/bash\n");
    script.push_str("# p2a analytics script\n");

    if let Some(title) = &session.title {
        script.push_str(&format!("# {}\n", title));
    }

    script.push_str(&format!("# Generated: {}\n", session.updated_at));
    script.push_str(&format!("# p2a version: {}\n", session.version));
    script.push('\n');
    script.push_str("set -euo pipefail\n");
    script.push('\n');

    // Create a temporary session file for the replay
    let temp_session = format!(".p2a_session_{}.json", uuid::Uuid::new_v4());
    script.push_str(&format!("SESSION_FILE=\"{}\"\n", temp_session));
    script.push('\n');

    // Generate commands from the session
    for record in &session.commands {
        if comments {
            script.push_str(&format!("# {}\n", record.command_line));
        }

        // Prefer the recorded argv (shell-escaped) for a faithful and
        // injection-safe reproduction; fall back to reconstructing from the
        // structured arguments for legacy records that predate `argv`.
        let cmd = if !record.argv.is_empty() {
            crate::output::shell_join(&record.argv)
        } else {
            reconstruct_command(record)
        };
        script.push_str(&format!("p2a --session \"$SESSION_FILE\" {}\n", cmd));
        script.push('\n');
    }

    // Cleanup
    script.push_str("# Cleanup temporary session file\n");
    script.push_str("rm -f \"$SESSION_FILE\"\n");
    script.push('\n');
    script.push_str("echo \"Script completed successfully\"\n");

    // Write the script
    std::fs::write(output, &script)?;

    // Make executable on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(output)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(output, perms)?;
    }

    print_message(
        &format!(
            "Exported {} commands to: {}",
            session.commands.len(),
            output.display()
        ),
        format,
    );

    Ok(())
}

fn execute_history(session_file: &PathBuf, format: &OutputFormat) -> anyhow::Result<()> {
    let session = match Session::load(session_file) {
        Ok(s) => s,
        Err(e) => {
            print_error(&format!("Failed to load session: {}", e), format);
            return Ok(());
        }
    };

    match format {
        OutputFormat::Json => {
            let json = serde_json::json!({
                "session_id": session.id,
                "title": session.title,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "datasets": session.datasets.keys().collect::<Vec<_>>(),
                "commands": session.commands.iter().map(|c| {
                    serde_json::json!({
                        "timestamp": c.timestamp,
                        "command": c.command_line,
                        "success": c.success,
                        "duration_ms": c.duration_ms,
                    })
                }).collect::<Vec<_>>(),
            });
            println!("{}", serde_json::to_string_pretty(&json)?);
        }
        _ => {
            println!("Session: {}", session.id);
            if let Some(title) = &session.title {
                println!("Title: {}", title);
            }
            println!("Created: {}", session.created_at);
            println!("Updated: {}", session.updated_at);
            println!("Version: {}", session.version);
            println!();

            println!("Datasets ({}):", session.datasets.len());
            for (name, meta) in &session.datasets {
                println!(
                    "  - {} ({} rows, {} cols)",
                    name,
                    meta.nrows,
                    meta.columns.len()
                );
            }
            println!();

            println!("Commands ({}):", session.commands.len());
            for (i, cmd) in session.commands.iter().enumerate() {
                let status = if cmd.success { "OK" } else { "FAIL" };
                println!(
                    "  {}. [{}] {} ({}ms)",
                    i + 1,
                    status,
                    cmd.command_line,
                    cmd.duration_ms
                );
            }
        }
    }

    Ok(())
}

fn execute_run(script_file: &PathBuf, format: &OutputFormat) -> anyhow::Result<()> {
    use std::process::Command;

    if !script_file.exists() {
        print_error(
            &format!("Script file not found: {}", script_file.display()),
            format,
        );
        return Ok(());
    }

    print_message(
        &format!("Running script: {}", script_file.display()),
        format,
    );

    let output = Command::new("bash").arg(script_file).output()?;

    // Print stdout
    if !output.stdout.is_empty() {
        print!("{}", String::from_utf8_lossy(&output.stdout));
    }

    // Print stderr
    if !output.stderr.is_empty() {
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
    }

    if output.status.success() {
        print_message("Script completed successfully", format);
    } else {
        print_error(
            &format!("Script failed with exit code: {:?}", output.status.code()),
            format,
        );
    }

    Ok(())
}

/// Reconstruct a CLI command from a CommandRecord
fn reconstruct_command(record: &crate::session::CommandRecord) -> String {
    use crate::output::shell_quote;

    // Quote a space-separated list of values, escaping each element.
    let quote_list = |vals: &[&str]| -> String {
        vals.iter()
            .map(|v| shell_quote(v))
            .collect::<Vec<_>>()
            .join(" ")
    };

    // Start with category and subcommand (method identifiers, not user data).
    let mut cmd = format!("{} {}", record.category, record.subcommand);

    // Add arguments based on the stored JSON
    if let Some(args) = record.arguments.as_object() {
        // Dataset reference (first positional arg for most commands)
        if let Some(dataset) = args.get("dataset").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" {}", shell_quote(dataset)));
        }

        // Path (for data load)
        if let Some(path) = args.get("path").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" {}", shell_quote(path)));
        }

        // Name
        if let Some(name) = args.get("name").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" --name {}", shell_quote(name)));
        }

        // Dependent variable
        if let Some(dep_var) = args.get("dep_var").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" -y {}", shell_quote(dep_var)));
        }

        // Independent variables
        if let Some(indep_vars) = args.get("indep_vars").and_then(|v| v.as_array()) {
            let vars: Vec<&str> = indep_vars.iter().filter_map(|v| v.as_str()).collect();
            if !vars.is_empty() {
                cmd.push_str(&format!(" -x {}", quote_list(&vars)));
            }
        }

        // Entity (for panel data)
        if let Some(entity) = args.get("entity").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" --entity {}", shell_quote(entity)));
        }

        // Time (for two-way FE)
        if let Some(time) = args.get("time").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" --time {}", shell_quote(time)));
        }

        // Cluster
        if let Some(cluster) = args.get("cluster").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" --cluster {}", shell_quote(cluster)));
        }

        // Robust SE type
        if let Some(robust) = args.get("robust").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" --robust {}", shell_quote(robust)));
        }

        // Output file
        if let Some(output) = args.get("output").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" -o {}", shell_quote(output)));
        }

        // Column
        if let Some(col) = args.get("col").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" --col {}", shell_quote(col)));
        }

        // Columns (multiple)
        if let Some(cols) = args.get("cols").and_then(|v| v.as_array()) {
            let cols_str: Vec<&str> = cols.iter().filter_map(|v| v.as_str()).collect();
            if !cols_str.is_empty() {
                cmd.push_str(&format!(" --cols {}", quote_list(&cols_str)));
            }
        }

        // K (for kmeans)
        if let Some(k) = args.get("k").and_then(|v| v.as_u64()) {
            cmd.push_str(&format!(" -k {}", k));
        }

        // N (for head)
        if let Some(n) = args.get("n").and_then(|v| v.as_u64()) {
            cmd.push_str(&format!(" -n {}", n));
        }

        // Lags (for time series)
        if let Some(lags) = args.get("lags").and_then(|v| v.as_u64()) {
            cmd.push_str(&format!(" --lags {}", lags));
        }

        // Horizon (for forecasting)
        if let Some(horizon) = args.get("horizon").and_then(|v| v.as_u64()) {
            cmd.push_str(&format!(" --horizon {}", horizon));
        }

        // Fixed effects
        if let Some(fe) = args.get("fe").and_then(|v| v.as_array()) {
            let fe_str: Vec<&str> = fe.iter().filter_map(|v| v.as_str()).collect();
            if !fe_str.is_empty() {
                cmd.push_str(&format!(" --fe {}", quote_list(&fe_str)));
            }
        }

        // Instruments
        if let Some(inst) = args.get("instruments").and_then(|v| v.as_array()) {
            let inst_str: Vec<&str> = inst.iter().filter_map(|v| v.as_str()).collect();
            if !inst_str.is_empty() {
                cmd.push_str(&format!(" --instruments {}", quote_list(&inst_str)));
            }
        }

        // Exogenous
        if let Some(exog) = args.get("exog").and_then(|v| v.as_array()) {
            let exog_str: Vec<&str> = exog.iter().filter_map(|v| v.as_str()).collect();
            if !exog_str.is_empty() {
                cmd.push_str(&format!(" --exog {}", quote_list(&exog_str)));
            }
        }

        // Endogenous
        if let Some(endog) = args.get("endog").and_then(|v| v.as_array()) {
            let endog_str: Vec<&str> = endog.iter().filter_map(|v| v.as_str()).collect();
            if !endog_str.is_empty() {
                cmd.push_str(&format!(" --endog {}", quote_list(&endog_str)));
            }
        }

        // Treatment and post (for DiD)
        if let Some(treat) = args.get("treat").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" --treat {}", shell_quote(treat)));
        }
        if let Some(post) = args.get("post").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" --post {}", shell_quote(post)));
        }
    }

    cmd
}
