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

        /// Write each recorded command line as a comment above it (true/false; default true).
        /// The header and the lines naming failed commands are always written
        #[arg(
            long,
            value_name = "BOOL",
            num_args = 0..=1,
            default_value_t = true,
            default_missing_value = "true",
            action = clap::ArgAction::Set,
            value_parser = clap::builder::BoolishValueParser::new()
        )]
        comments: bool,

        /// Replay the commands that failed when they were recorded too, each with `|| true`
        /// so that a failure does not stop the script (by default they are only named in a
        /// comment)
        #[arg(long)]
        include_failed: bool,
    },

    /// Show session history
    History {
        /// Session file to display
        session_file: PathBuf,
    },

    /// Run a p2a script with bash; its output and exit status are the script's
    ///
    /// The script's `$P2A` is this p2a binary.
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
            include_failed,
        } => execute_export(session_file, output, *comments, *include_failed, format),
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
    include_failed: bool,
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

    let script = render_script(&session, comments, include_failed);
    crate::session::write_atomically(output, script.as_bytes())?;

    // Make executable on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(output)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(output, perms)?;
    }

    let failed = session.commands.iter().filter(|r| !r.success).count();
    let replayed = if include_failed {
        session.commands.len()
    } else {
        session.commands.len() - failed
    };
    let skipped = if include_failed || failed == 0 {
        String::new()
    } else {
        format!(" ({failed} failed commands not replayed)")
    };
    print_message(
        &format!(
            "Exported {replayed} commands to: {}{skipped}",
            output.display()
        ),
        format,
    );

    Ok(())
}

/// Render the bash script for `session`.
///
/// The script replays into a session file in a fresh `mktemp -d` directory that an `EXIT`
/// trap removes (with the derived datasets stored beside that session file), whether the
/// replay succeeds or not. It changes into the directory the session was recorded in
/// (`P2A_WORKDIR` overrides it) and into each command's own directory where that differs,
/// and runs `$P2A` (default `p2a`). Commands that failed when recorded are named in a
/// comment and not replayed, unless `include_failed`, which replays them with `|| true`.
fn render_script(session: &Session, comments: bool, include_failed: bool) -> String {
    let mut script = String::new();

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
    script.push_str(
        "# Replays the recorded commands in the directory they ran in. Set P2A_WORKDIR to replay\n\
         # elsewhere and P2A to use another p2a binary.\n",
    );
    script.push('\n');
    script.push_str("set -euo pipefail\n");
    script.push('\n');

    let recorded = if session.working_dir.as_os_str().is_empty() {
        ".".to_string()
    } else {
        script_quote(&session.working_dir.to_string_lossy())
    };
    script.push_str("P2A=\"${P2A:-p2a}\"\n");
    script.push_str(&format!("P2A_RECORDED_DIR={recorded}\n"));
    script.push_str("P2A_WORKDIR=\"${P2A_WORKDIR:-$P2A_RECORDED_DIR}\"\n");
    script.push_str("P2A_REPLAY_DIR=\"$(mktemp -d \"${TMPDIR:-/tmp}/p2a_replay.XXXXXX\")\"\n");
    script.push_str("trap 'rm -rf -- \"$P2A_REPLAY_DIR\"' EXIT\n");
    script.push_str("SESSION_FILE=\"$P2A_REPLAY_DIR/session.json\"\n");
    script.push_str("cd -- \"$P2A_WORKDIR\"\n");
    script.push('\n');

    // the directory the replay is in, as a recorded path
    let mut current = session.working_dir.clone();
    for record in &session.commands {
        if !record.success && !include_failed {
            script.push_str("# not replayed (failed when recorded): ");
            script.push_str(&comment_text(&record.command_line));
            script.push('\n');
            continue;
        }
        if let Some(cwd) = record.cwd.as_ref().filter(|c| **c != current) {
            script.push_str(&cd_line(cwd, &session.working_dir));
            current = cwd.clone();
        }
        if !record.success {
            script.push_str("# failed when recorded; a failure here does not stop the script\n");
        }
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
        let tail = if record.success { "" } else { " || true" };
        script.push_str(&format!(
            "\"$P2A\" --session \"$SESSION_FILE\" {cmd}{tail}\n"
        ));
        script.push('\n');
    }

    script.push_str("echo \"Script completed successfully\"\n");
    script
}

/// The `cd` line into the recorded directory `cwd`: relative to `$P2A_WORKDIR` when it lies
/// under the session's `working_dir`, else the absolute path.
fn cd_line(cwd: &std::path::Path, working_dir: &std::path::Path) -> String {
    if cwd == working_dir {
        return "cd -- \"$P2A_WORKDIR\"\n".to_string();
    }
    match cwd.strip_prefix(working_dir) {
        Ok(rel) if !working_dir.as_os_str().is_empty() => format!(
            "cd -- \"$P2A_WORKDIR\"/{}\n",
            script_quote(&rel.to_string_lossy())
        ),
        _ => format!("cd -- {}\n", script_quote(&cwd.to_string_lossy())),
    }
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

/// Run `script_file` with bash: the script reads this process's stdin and writes to its
/// stdout and stderr as it runs, and `$P2A` is this p2a binary. Text mode frames the
/// script's output with `Running script:` and `Script completed successfully`; JSON mode
/// adds nothing to stdout. A failed script ends this process with the script's exit code
/// (128 + the signal number when a signal killed it), so `main` does not save a
/// `--session` file afterwards, which is harmless: `script` commands are not recorded.
fn execute_run(script_file: &PathBuf, format: &OutputFormat) -> anyhow::Result<()> {
    use std::io::Write as _;
    use std::process::{Command, Stdio};

    if !script_file.exists() {
        print_error(
            &format!("Script file not found: {}", script_file.display()),
            format,
        );
        return Ok(());
    }
    let text = !matches!(format, OutputFormat::Json);
    if text {
        print_message(
            &format!("Running script: {}", script_file.display()),
            format,
        );
    }
    let _ = std::io::stdout().flush();

    let status = Command::new("bash")
        .arg("--")
        .arg(script_file)
        .env("P2A", std::env::current_exe()?)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()?;

    if status.success() {
        if text {
            print_message("Script completed successfully", format);
        }
        return Ok(());
    }
    let code = exit_code(&status);
    print_error(&format!("Script failed with exit code {code}"), format);
    std::process::exit(code);
}

/// The exit code of a finished child: its own, or 128 + the signal number on Unix when a
/// signal killed it (as a shell reports it); 1 when neither is known.
fn exit_code(status: &std::process::ExitStatus) -> i32 {
    if let Some(code) = status.code() {
        return code;
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = status.signal() {
            return 128 + sig;
        }
    }
    1
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
            cwd: None,
            success: true,
            error: None,
            duration_ms: 0,
        }
    }

    /// The fixture: title "Demo", working directory /work/proj, a load and a legacy record.
    fn demo() -> Session {
        let mut s = Session::new();
        s.title = Some("Demo".into());
        s.version = "0.1.3".into();
        s.updated_at = "2026-01-01T00:00:00Z".parse().unwrap();
        s.working_dir = "/work/proj".into();
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
        s
    }

    const HEADER: &str = "#!/bin/bash\n# p2a analytics script\n# Demo\n# Generated: 2026-01-01 00:00:00 UTC\n# p2a version: 0.1.3\n# Replays the recorded commands in the directory they ran in. Set P2A_WORKDIR to replay\n# elsewhere and P2A to use another p2a binary.\n\nset -euo pipefail\n\nP2A=\"${P2A:-p2a}\"\nP2A_RECORDED_DIR=/work/proj\nP2A_WORKDIR=\"${P2A_WORKDIR:-$P2A_RECORDED_DIR}\"\nP2A_REPLAY_DIR=\"$(mktemp -d \"${TMPDIR:-/tmp}/p2a_replay.XXXXXX\")\"\ntrap 'rm -rf -- \"$P2A_REPLAY_DIR\"' EXIT\nSESSION_FILE=\"$P2A_REPLAY_DIR/session.json\"\ncd -- \"$P2A_WORKDIR\"\n\n";

    #[test]
    fn render_script_ordinary_session_is_unchanged() {
        assert_eq!(
            render_script(&demo(), true, false),
            format!(
                "{HEADER}# data load d.csv --name d\n\"$P2A\" --session \"$SESSION_FILE\" data load d.csv --name d\n\n# reg ols d -y y -x x1 x2\n\"$P2A\" --session \"$SESSION_FILE\" reg ols d -y y -x x1 x2 --robust hc1\n\necho \"Script completed successfully\"\n"
            )
        );
    }

    /// `--comments false`: the same script without the per-command comment lines; the
    /// header stays.
    #[test]
    fn render_script_without_comments_omits_command_lines() {
        assert_eq!(
            render_script(&demo(), false, false),
            format!(
                "{HEADER}\"$P2A\" --session \"$SESSION_FILE\" data load d.csv --name d\n\n\"$P2A\" --session \"$SESSION_FILE\" reg ols d -y y -x x1 x2 --robust hc1\n\necho \"Script completed successfully\"\n"
            )
        );
    }

    /// The replay session lives in a `mktemp -d` directory that an EXIT trap removes, its
    /// derived datasets included; no cleanup lines at the end.
    #[test]
    fn render_script_removes_the_replay_directory_on_exit() {
        let mut s = demo();
        s.commands.push(record(
            &["munge", "select", "d", "--columns", "x", "--name", "e"],
            "munge select d --columns x --name e",
            "munge",
            "select",
            serde_json::Value::Null,
        ));
        let text = render_script(&s, false, false);
        assert!(
            text.contains(
                "P2A_REPLAY_DIR=\"$(mktemp -d \"${TMPDIR:-/tmp}/p2a_replay.XXXXXX\")\"\ntrap 'rm -rf -- \"$P2A_REPLAY_DIR\"' EXIT\nSESSION_FILE=\"$P2A_REPLAY_DIR/session.json\"\n"
            ),
            "{text}"
        );
        assert!(
            !text.contains("rm -f") && !text.contains(".datasets"),
            "{text}"
        );
        assert!(text.ends_with(
            "\"$P2A\" --session \"$SESSION_FILE\" munge select d --columns x --name e\n\necho \"Script completed successfully\"\n"
        ));
    }

    #[test]
    fn render_script_skips_failed_records() {
        let mut s = demo();
        let mut bad = record(
            &["reg", "ols", "d", "-y", "y", "-x", "valuee"],
            "reg ols d -y y -x valuee",
            "reg",
            "ols",
            serde_json::Value::Null,
        );
        bad.success = false;
        s.commands.insert(1, bad);
        let load = "# data load d.csv --name d\n\"$P2A\" --session \"$SESSION_FILE\" data load d.csv --name d\n\n";
        let ols = "# reg ols d -y y -x x1 x2\n\"$P2A\" --session \"$SESSION_FILE\" reg ols d -y y -x x1 x2 --robust hc1\n\necho \"Script completed successfully\"\n";
        assert_eq!(
            render_script(&s, true, false),
            format!(
                "{HEADER}{load}# not replayed (failed when recorded): reg ols d -y y -x valuee\n{ols}"
            )
        );
        // the line naming the failed command stays without comments
        assert!(
            render_script(&s, false, false)
                .contains("\n# not replayed (failed when recorded): reg ols d -y y -x valuee\n")
        );
        assert_eq!(
            render_script(&s, true, true),
            format!(
                "{HEADER}{load}# failed when recorded; a failure here does not stop the script\n# reg ols d -y y -x valuee\n\"$P2A\" --session \"$SESSION_FILE\" reg ols d -y y -x valuee || true\n\n{ols}"
            )
        );
    }

    #[test]
    fn render_script_changes_directory_per_record() {
        let mut s = demo();
        s.commands.truncate(1);
        s.commands[0].cwd = Some("/work/proj".into());
        for (dir, name) in [
            ("/work/proj/sub", "a"),
            ("/work/proj/sub", "b"),
            ("/elsewhere", "c"),
            ("/work/proj", "d"),
            ("/work/proj/my dir", "e"),
        ] {
            let mut r = record(
                &["data", "head", name],
                &format!("data head {name}"),
                "data",
                "head",
                serde_json::Value::Null,
            );
            r.cwd = Some(dir.into());
            s.commands.push(r);
        }
        let text = render_script(&s, false, false);
        let body = text.strip_prefix(HEADER).unwrap();
        assert_eq!(
            body,
            "\"$P2A\" --session \"$SESSION_FILE\" data load d.csv --name d\n\n\
             cd -- \"$P2A_WORKDIR\"/sub\n\"$P2A\" --session \"$SESSION_FILE\" data head a\n\n\
             \"$P2A\" --session \"$SESSION_FILE\" data head b\n\n\
             cd -- /elsewhere\n\"$P2A\" --session \"$SESSION_FILE\" data head c\n\n\
             cd -- \"$P2A_WORKDIR\"\n\"$P2A\" --session \"$SESSION_FILE\" data head d\n\n\
             cd -- \"$P2A_WORKDIR\"/'my dir'\n\"$P2A\" --session \"$SESSION_FILE\" data head e\n\n\
             echo \"Script completed successfully\"\n"
        );
    }

    #[test]
    fn render_script_quotes_a_crafted_working_dir() {
        let mut s = demo();
        s.working_dir = "x\ntouch injected".into();
        let text = render_script(&s, true, false);
        assert!(
            text.contains("\nP2A_RECORDED_DIR=$'x\\ntouch injected'\n"),
            "{text}"
        );
        s.working_dir = "a b;rm".into();
        assert!(render_script(&s, true, false).contains("\nP2A_RECORDED_DIR='a b;rm'\n"));
        s.working_dir = PathBuf::new();
        assert!(render_script(&s, true, false).contains("\nP2A_RECORDED_DIR=.\n"));
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
