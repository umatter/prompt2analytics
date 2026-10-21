//! Script generation and session management commands

use clap::Subcommand;
use std::path::PathBuf;

use crate::output::{OutputFormat, print_error, print_message, shell_quote};
use crate::session::Session;

#[derive(Subcommand)]
pub enum ScriptCommands {
    /// Export session to a reproducible bash script
    ///
    /// Every recorded string is written either as one escaped comment line or
    /// as one quoted argument. Replaying a session file runs the p2a commands
    /// it contains, so review a session from someone else like any script.
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

/// Export a session file as a bash script.
///
/// Every recorded string (title, version, each command line and argument) is
/// written either as one `# ` comment line with control characters escaped,
/// or as one quoted argument on its command's line. Replaying a session file
/// runs the p2a commands it contains, so a session from someone else should
/// be reviewed like any script.
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

    // Generate the bash script, replaying into a temporary session file
    let temp_session = format!(".p2a_session_{}.json", uuid::Uuid::new_v4());
    let script = render_script(&session, comments, &temp_session);

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

/// Render the bash script for `session`, replaying into `temp_session`.
fn render_script(session: &Session, comments: bool, temp_session: &str) -> String {
    let mut script = String::new();

    // Shebang and header
    script.push_str("#!/bin/bash\n");
    script.push_str("# p2a analytics script\n");

    if let Some(title) = &session.title {
        push_comment(&mut script, title);
    }

    script.push_str(&format!("# Generated: {}\n", session.updated_at));
    script.push_str(&format!(
        "# p2a version: {}\n",
        comment_text(&session.version)
    ));
    script.push('\n');
    script.push_str("set -euo pipefail\n");
    script.push('\n');

    // Create a temporary session file for the replay
    script.push_str(&format!("SESSION_FILE=\"{}\"\n", temp_session));
    script.push('\n');

    // Generate commands from the session
    for record in &session.commands {
        if comments {
            push_comment(&mut script, &record.command_line);
        }

        // Prefer the recorded argv (shell-escaped) for a faithful and
        // injection-safe reproduction; fall back to reconstructing from the
        // structured arguments for legacy records that predate `argv`.
        let cmd = if !record.argv.is_empty() {
            script_join(&record.argv)
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

    script
}

/// Text for a `# ` comment line, with control characters shown as escapes
/// (`\n`, `\r`, `\t`, `\u{..}`), so the comment cannot end early.
fn comment_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c.is_control() {
            out.extend(c.escape_default());
        } else {
            out.push(c);
        }
    }
    out
}

/// Append `text` to the script as exactly one `# ` comment line.
fn push_comment(script: &mut String, text: &str) {
    script.push_str("# ");
    script.push_str(&comment_text(text));
    script.push('\n');
}

/// Quote one argument so that it stays on its command's line.
///
/// Arguments without control characters use [`shell_quote`] unchanged. An
/// argument with a control character is written in bash ANSI-C quoting
/// (`$'...'`): backslash and single quote are escaped, LF, CR and TAB become
/// `\n`, `\r` and `\t`, and every other control character becomes `\xHH` per
/// UTF-8 byte (`\x` rather than `\u`, which bash 3.2 does not support).
fn script_quote(s: &str) -> String {
    use std::fmt::Write as _;

    if !s.chars().any(char::is_control) {
        return shell_quote(s);
    }
    let mut out = String::with_capacity(s.len() + 3);
    out.push_str("$'");
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                let mut buf = [0u8; 4];
                for b in c.encode_utf8(&mut buf).bytes() {
                    // Writing to a String cannot fail.
                    let _ = write!(out, "\\x{b:02x}");
                }
            }
            c => out.push(c),
        }
    }
    out.push('\'');
    out
}

/// [`script_quote`] each argument and join them with spaces.
fn script_join(args: &[String]) -> String {
    args.iter()
        .map(|a| script_quote(a))
        .collect::<Vec<_>>()
        .join(" ")
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
    // Quote a space-separated list of values, escaping each element.
    let quote_list = |vals: &[&str]| -> String {
        vals.iter()
            .map(|v| script_quote(v))
            .collect::<Vec<_>>()
            .join(" ")
    };

    // Start with category and subcommand. They come from the session file,
    // so they are quoted like any other argument.
    let mut cmd = format!(
        "{} {}",
        script_quote(&record.category),
        script_quote(&record.subcommand)
    );

    // Add arguments based on the stored JSON
    if let Some(args) = record.arguments.as_object() {
        // Dataset reference (first positional arg for most commands)
        if let Some(dataset) = args.get("dataset").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" {}", script_quote(dataset)));
        }

        // Path (for data load)
        if let Some(path) = args.get("path").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" {}", script_quote(path)));
        }

        // Name
        if let Some(name) = args.get("name").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" --name {}", script_quote(name)));
        }

        // Dependent variable
        if let Some(dep_var) = args.get("dep_var").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" -y {}", script_quote(dep_var)));
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
            cmd.push_str(&format!(" --entity {}", script_quote(entity)));
        }

        // Time (for two-way FE)
        if let Some(time) = args.get("time").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" --time {}", script_quote(time)));
        }

        // Cluster
        if let Some(cluster) = args.get("cluster").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" --cluster {}", script_quote(cluster)));
        }

        // Robust SE type
        if let Some(robust) = args.get("robust").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" --robust {}", script_quote(robust)));
        }

        // Output file
        if let Some(output) = args.get("output").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" -o {}", script_quote(output)));
        }

        // Column
        if let Some(col) = args.get("col").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" --col {}", script_quote(col)));
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
            cmd.push_str(&format!(" --treat {}", script_quote(treat)));
        }
        if let Some(post) = args.get("post").and_then(|v| v.as_str()) {
            cmd.push_str(&format!(" --post {}", script_quote(post)));
        }
    }

    cmd
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::CommandRecord;
    use chrono::Utc;

    fn record(
        argv: &[&str],
        command_line: &str,
        category: &str,
        subcommand: &str,
        arguments: serde_json::Value,
    ) -> CommandRecord {
        CommandRecord {
            id: "test".into(),
            timestamp: Utc::now(),
            command_line: command_line.into(),
            category: category.into(),
            subcommand: subcommand.into(),
            arguments,
            argv: argv.iter().map(|s| s.to_string()).collect(),
            dataset_refs: vec![],
            output_files: vec![],
            success: true,
            error: None,
            duration_ms: 0,
        }
    }

    #[test]
    fn render_script_ordinary_session_is_unchanged() {
        let mut s = Session::new();
        s.title = Some("Demo".into());
        s.version = "0.1.3".into();
        s.updated_at = "2026-01-01T00:00:00Z".parse().unwrap();
        s.commands.push(record(
            &["data", "load", "d.csv", "--name", "d"],
            "data load d.csv --name d",
            "data",
            "load",
            serde_json::Value::Null,
        ));
        s.commands.push(record(
            &[],
            "reg ols d -y y -x x1 x2",
            "reg",
            "ols",
            serde_json::json!({"dataset": "d", "dep_var": "y", "indep_vars": ["x1", "x2"], "robust": "hc1"}),
        ));
        assert_eq!(
            render_script(&s, true, ".p2a_session_test.json"),
            "#!/bin/bash\n# p2a analytics script\n# Demo\n# Generated: 2026-01-01 00:00:00 UTC\n# p2a version: 0.1.3\n\nset -euo pipefail\n\nSESSION_FILE=\".p2a_session_test.json\"\n\n# data load d.csv --name d\np2a --session \"$SESSION_FILE\" data load d.csv --name d\n\n# reg ols d -y y -x x1 x2\np2a --session \"$SESSION_FILE\" reg ols d -y y -x x1 x2 --robust hc1\n\n# Cleanup temporary session file\nrm -f \"$SESSION_FILE\"\n\necho \"Script completed successfully\"\n"
        );
    }

    #[test]
    fn comment_text_escapes_control_characters() {
        assert_eq!(
            comment_text("a\nb\rc\td\u{1b}[0m"),
            "a\\nb\\rc\\td\\u{1b}[0m"
        );
        assert_eq!(comment_text("C:\\data\\x.csv"), "C:\\data\\x.csv");
        assert_eq!(comment_text("plain 'quoted' text"), "plain 'quoted' text");
    }

    #[test]
    fn script_quote_keeps_arguments_on_one_line() {
        assert_eq!(script_quote("plain"), "plain");
        assert_eq!(script_quote("two words"), "'two words'");
        assert_eq!(script_quote("x\ny"), "$'x\\ny'");
        assert_eq!(script_quote("it's\tok"), "$'it\\'s\\tok'");
        assert_eq!(script_quote("a\\b\r"), "$'a\\\\b\\r'");
        assert_eq!(script_quote("\u{1b}[1m"), "$'\\x1b[1m'");
        assert_eq!(script_quote("\u{85}"), "$'\\xc2\\x85'");
    }

    #[test]
    fn reconstruct_command_quotes_legacy_fields() {
        let r = record(
            &[],
            "x",
            "data;touch x;",
            "describe",
            serde_json::json!({"dataset": "d\ny"}),
        );
        assert_eq!(reconstruct_command(&r), "'data;touch x;' describe $'d\\ny'");
    }
}
