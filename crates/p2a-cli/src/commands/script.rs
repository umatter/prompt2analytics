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

        // Reconstruct the command from the record
        let cmd = reconstruct_command(record);
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
    // Start with category and subcommand
    let mut cmd = format!("{} {}", record.category, record.subcommand);

    // Add arguments based on the stored JSON
    if let Some(args) = record.arguments.as_object() {
        // Dataset reference (first positional arg for most commands)
        if let Some(dataset) = args.get("dataset").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" {}", dataset));
        }

        // Path (for data load)
        if let Some(path) = args.get("path").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" \"{}\"", path));
        }

        // Name
        if let Some(name) = args.get("name").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" --name {}", name));
        }

        // Dependent variable
        if let Some(dep_var) = args.get("dep_var").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" -y {}", dep_var));
        }

        // Independent variables
        if let Some(indep_vars) = args.get("indep_vars").and_then(|v| v.as_array()) {
            let vars: Vec<&str> = indep_vars.iter().filter_map(|v| v.as_str()).collect();
            if !vars.is_empty() {
                cmd.push_str(&format!(" -x {}", vars.join(" ")));
            }
        }

        // Entity (for panel data)
        if let Some(entity) = args.get("entity").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" --entity {}", entity));
        }

        // Time (for two-way FE)
        if let Some(time) = args.get("time").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" --time {}", time));
        }

        // Cluster
        if let Some(cluster) = args.get("cluster").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" --cluster {}", cluster));
        }

        // Robust SE type
        if let Some(robust) = args.get("robust").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" --robust {}", robust));
        }

        // Output file
        if let Some(output) = args.get("output").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" -o \"{}\"", output));
        }

        // Column
        if let Some(col) = args.get("col").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" --col {}", col));
        }

        // Columns (multiple)
        if let Some(cols) = args.get("cols").and_then(|v| v.as_array()) {
            let cols_str: Vec<&str> = cols.iter().filter_map(|v| v.as_str()).collect();
            if !cols_str.is_empty() {
                cmd.push_str(&format!(" --cols {}", cols_str.join(" ")));
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
                cmd.push_str(&format!(" --fe {}", fe_str.join(" ")));
            }
        }

        // Instruments
        if let Some(inst) = args.get("instruments").and_then(|v| v.as_array()) {
            let inst_str: Vec<&str> = inst.iter().filter_map(|v| v.as_str()).collect();
            if !inst_str.is_empty() {
                cmd.push_str(&format!(" --instruments {}", inst_str.join(" ")));
            }
        }

        // Exogenous
        if let Some(exog) = args.get("exog").and_then(|v| v.as_array()) {
            let exog_str: Vec<&str> = exog.iter().filter_map(|v| v.as_str()).collect();
            if !exog_str.is_empty() {
                cmd.push_str(&format!(" --exog {}", exog_str.join(" ")));
            }
        }

        // Endogenous
        if let Some(endog) = args.get("endog").and_then(|v| v.as_array()) {
            let endog_str: Vec<&str> = endog.iter().filter_map(|v| v.as_str()).collect();
            if !endog_str.is_empty() {
                cmd.push_str(&format!(" --endog {}", endog_str.join(" ")));
            }
        }

        // Treatment and post (for DiD)
        if let Some(treat) = args.get("treat").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" --treat {}", treat));
        }
        if let Some(post) = args.get("post").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" --post {}", post));
        }
    }

    cmd
}
