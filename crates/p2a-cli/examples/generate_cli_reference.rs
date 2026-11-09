//! Generated parts of the CLI documentation, rendered from the `p2a` binary's own help text (clap).
//!
//! Two sections are generated, each between a BEGIN and an END marker line:
//! - `docs/guides/CLI_REFERENCE.md`: the command reference (global options, and for every command its
//!   usage line, arguments and options);
//! - `README.md`: the command list (each top-level command with its subcommands).
//!
//! `crates/p2a-cli/tests/docs_examples.rs` fails when a section differs from what this renders
//! (INF-3). Rewrite both with either of
//!
//! ```text
//! cargo build -p p2a-cli && cargo run -p p2a-cli --example generate_cli_reference [path/to/p2a]
//! P2A_UPDATE_DOCS=1 cargo test -p p2a-cli --test docs_examples
//! ```
//!
//! The binary is the first argument, else `$P2A_BIN`, else `p2a` next to this example's executable
//! (`target/<profile>/p2a`; `cargo run --example` does not build it). p2a-cli has no library target,
//! so the help text is the interface. Help is read with a cleared environment plus `NO_COLOR=1` and
//! `COLUMNS=100`, so colours and `[env: ...]` values never reach the files.

use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

/// A generated section: the file it lives in and the name its marker lines carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Section {
    pub file: &'static str,
    pub name: &'static str,
}

/// The command reference in `docs/guides/CLI_REFERENCE.md`.
pub const REFERENCE: Section = Section {
    file: "docs/guides/CLI_REFERENCE.md",
    name: "p2a command reference",
};

/// The command list in `README.md`.
pub const COMMAND_LIST: Section = Section {
    file: "README.md",
    name: "p2a command list",
};

impl Section {
    pub fn begin(&self) -> String {
        format!(
            "<!-- BEGIN GENERATED: {} (cargo run -p p2a-cli --example generate_cli_reference; do not edit by hand) -->",
            self.name
        )
    }

    pub fn end(&self) -> String {
        format!("<!-- END GENERATED: {} -->", self.name)
    }
}

/// One entry of a help section: `-y, --dep-var <DEP_VAR>` and its description.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub spec: String,
    pub help: String,
}

/// The parsed `-h` output of one command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandHelp {
    /// Subcommand names after `p2a` (empty for `p2a` itself).
    pub path: Vec<String>,
    pub about: String,
    pub usage: String,
    pub commands: Vec<Entry>,
    pub arguments: Vec<Entry>,
    pub options: Vec<Entry>,
    /// The `p2a ...` lines of the after-help text (EXAMPLES), continuation lines joined.
    pub examples: Vec<String>,
}

impl CommandHelp {
    /// `(name, description, visible aliases)` of each subcommand, without clap's `help`.
    pub fn subcommands(&self) -> Vec<(String, String, Vec<String>)> {
        self.commands
            .iter()
            .filter(|e| e.spec != "help")
            .map(|e| {
                let (help, aliases) = split_aliases(&e.help);
                (e.spec.clone(), help, aliases)
            })
            .collect()
    }
}

/// `"Regression analysis [aliases: reg]"` -> `("Regression analysis", ["reg"])`.
pub fn split_aliases(help: &str) -> (String, Vec<String>) {
    for key in ["[aliases: ", "[alias: "] {
        if let Some(start) = help.find(key)
            && let Some(len) = help[start..].find(']')
        {
            let inner = &help[start + key.len()..start + len];
            let rest = format!(
                "{} {}",
                help[..start].trim_end(),
                help[start + len + 1..].trim_start()
            );
            let aliases = inner
                .split(',')
                .map(|a| a.trim().to_string())
                .filter(|a| !a.is_empty())
                .collect();
            return (rest.trim().to_string(), aliases);
        }
    }
    (help.trim().to_string(), Vec::new())
}

/// Runs `p2a <path> -h` with a cleared environment and stdin closed (EOF), so no help path can block on it.
pub fn help_text(bin: &Path, path: &[String]) -> Result<String, String> {
    let shown = format!("p2a {} -h", path.join(" "));
    let out = Command::new(bin)
        .args(path)
        .arg("-h")
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("NO_COLOR", "1")
        .env("COLUMNS", "100")
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("cannot run {}: {e}", bin.display()))?;
    if !out.status.success() {
        return Err(format!(
            "`{shown}` exited with {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    String::from_utf8(out.stdout).map_err(|e| format!("`{shown}` printed non-UTF-8 text: {e}"))
}

/// An entry starts at column 2 (`-x`, `--x`, `<ARG>`, `[ARG]`, a subcommand name) or, for a
/// long-only option, at column 6; deeper lines continue its description.
fn entry_starts(line: &str, section: &str) -> bool {
    if line.starts_with("      --") {
        return true;
    }
    let Some(rest) = line.strip_prefix("  ") else {
        return false;
    };
    match rest.chars().next() {
        Some('-' | '<' | '[') => true,
        Some(c) => section == "Commands" && c.is_ascii_lowercase(),
        None => false,
    }
}

/// The spec ends at the first run of two spaces; the rest of the line is the description.
fn split_spec(line: &str) -> Entry {
    let t = line.trim();
    match t.find("  ") {
        Some(i) => Entry {
            spec: t[..i].to_string(),
            help: t[i..].trim().to_string(),
        },
        None => Entry {
            spec: t.to_string(),
            help: String::new(),
        },
    }
}

/// Parses clap 4's `-h` layout: about lines, `Usage:`, the `Commands:`, `Arguments:` and `Options:`
/// sections, then after-help text (column 0, or indented text after a section's blank line), whose
/// `p2a` lines are the examples.
pub fn parse_help(path: &[String], text: &str) -> Result<CommandHelp, String> {
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    let mut about = Vec::new();
    while i < lines.len() && !lines[i].starts_with("Usage: ") {
        if !lines[i].trim().is_empty() {
            about.push(lines[i].trim());
        }
        i += 1;
    }
    let Some(first) = lines.get(i).and_then(|l| l.strip_prefix("Usage: ")) else {
        return Err(format!("`p2a {} -h` printed no Usage line", path.join(" ")));
    };
    let mut usage = vec![first.trim()];
    i += 1;
    while i < lines.len() && lines[i].starts_with(' ') && !lines[i].trim().is_empty() {
        usage.push(lines[i].trim());
        i += 1;
    }
    let mut help = CommandHelp {
        path: path.to_vec(),
        about: about.join(" "),
        usage: usage.join("\n"),
        commands: Vec::new(),
        arguments: Vec::new(),
        options: Vec::new(),
        examples: Vec::new(),
    };
    let mut section = "";
    let mut after = Vec::new();
    for &line in &lines[i..] {
        if matches!(line, "Commands:" | "Arguments:" | "Options:") {
            section = line.trim_end_matches(':');
            continue;
        }
        if section == "after" || (!line.is_empty() && !line.starts_with(' ')) {
            section = "after";
            after.push(line);
            continue;
        }
        if line.trim().is_empty() {
            if !section.is_empty() {
                section = "gap";
            }
            continue;
        }
        let list = match section {
            "Commands" => &mut help.commands,
            "Arguments" => &mut help.arguments,
            "Options" => &mut help.options,
            _ => {
                section = "after";
                after.push(line);
                continue;
            }
        };
        if entry_starts(line, section) {
            list.push(split_spec(line));
        } else if let Some(last) = list.last_mut() {
            if !last.help.is_empty() {
                last.help.push(' ');
            }
            last.help.push_str(line.trim());
        }
    }
    help.examples = example_lines(&after);
    Ok(help)
}

/// True when `s` ends inside a '...' or "..." quote (a backslash escapes outside single quotes).
pub fn open_quote(s: &str) -> bool {
    let (mut single, mut double, mut escaped) = (false, false, false);
    for c in s.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        match c {
            '\\' if !single => escaped = true,
            '\'' if !double => single = !single,
            '"' if !single => double = !double,
            _ => {}
        }
    }
    single || double
}

/// The `p2a ...` lines of free text. A line ending in `\`, or one that leaves a quote open,
/// continues on the next (joined with one space).
pub fn example_lines(lines: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    let mut pending: Option<String> = None;
    for line in lines {
        let t = line.trim();
        let cur = match pending.take() {
            Some(mut p) => {
                p.push(' ');
                p.push_str(t);
                p
            }
            None if t.starts_with("p2a ") => t.to_string(),
            None => continue,
        };
        if let Some(head) = cur.strip_suffix('\\') {
            pending = Some(head.trim_end().to_string());
        } else if open_quote(&cur) {
            pending = Some(cur);
        } else {
            out.push(cur);
        }
    }
    out.extend(pending);
    out
}

/// Splits a documented command line as a POSIX shell would for these examples: whitespace separates
/// words; '...' is literal; "..." keeps its text (\" \\ \$ \` are escapes); a backslash outside quotes
/// escapes the next character; an unquoted `#` at the start of a word starts a comment; an unquoted
/// `|`, `;`, `&`, `<` or `>` ends the command. An unterminated quote is an error.
pub fn shell_words(line: &str) -> Result<Vec<String>, String> {
    let mut words = Vec::new();
    let mut cur = String::new();
    let mut in_word = false;
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        match c {
            ' ' | '\t' | '\n' => {
                if in_word {
                    words.push(std::mem::take(&mut cur));
                    in_word = false;
                }
            }
            '#' if !in_word => break,
            '|' | ';' | '&' | '<' | '>' => break,
            '\'' => {
                in_word = true;
                loop {
                    match chars.next() {
                        Some('\'') => break,
                        Some(ch) => cur.push(ch),
                        None => return Err(format!("unterminated ' in `{line}`")),
                    }
                }
            }
            '"' => {
                in_word = true;
                loop {
                    match chars.next() {
                        Some('"') => break,
                        Some('\\') => match chars.next() {
                            Some(ch @ ('"' | '\\' | '$' | '`')) => cur.push(ch),
                            Some(ch) => {
                                cur.push('\\');
                                cur.push(ch);
                            }
                            None => return Err(format!("unterminated \" in `{line}`")),
                        },
                        Some(ch) => cur.push(ch),
                        None => return Err(format!("unterminated \" in `{line}`")),
                    }
                }
            }
            '\\' => {
                in_word = true;
                if let Some(ch) = chars.next() {
                    cur.push(ch);
                }
            }
            _ => {
                in_word = true;
                cur.push(c);
            }
        }
    }
    if in_word {
        words.push(cur);
    }
    Ok(words)
}

/// Every command's help, depth first in help order, starting with `p2a` itself.
pub fn walk(bin: &Path) -> Result<Vec<CommandHelp>, String> {
    fn visit(bin: &Path, path: Vec<String>, out: &mut Vec<CommandHelp>) -> Result<(), String> {
        let help = parse_help(&path, &help_text(bin, &path)?)?;
        let subs: Vec<String> = help
            .subcommands()
            .into_iter()
            .map(|(name, _, _)| name)
            .collect();
        out.push(help);
        for sub in subs {
            let mut next = path.clone();
            next.push(sub);
            visit(bin, next, out)?;
        }
        Ok(())
    }
    let mut out = Vec::new();
    visit(bin, Vec::new(), &mut out)?;
    Ok(out)
}

/// Markdown table-cell text: `\`, `|`, `*`, `<` and `>` escaped (help text may contain any of them).
fn cell(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '|' => out.push_str("\\|"),
            '*' => out.push_str("\\*"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(c),
        }
    }
    out
}

/// An inline code span that is safe inside a table.
fn code(s: &str) -> String {
    format!("`{}`", s.replace('|', "\\|"))
}

fn code_list(items: &[String]) -> String {
    items.iter().map(|s| code(s)).collect::<Vec<_>>().join(", ")
}

fn commands_table(md: &mut String, help: &CommandHelp) {
    md.push_str("| Command | Aliases | Description |\n|---|---|---|\n");
    for (name, about, aliases) in help.subcommands() {
        md.push_str(&format!(
            "| {} | {} | {} |\n",
            code(&name),
            code_list(&aliases),
            cell(&about)
        ));
    }
    md.push('\n');
}

fn root(cmds: &[CommandHelp]) -> Result<&CommandHelp, String> {
    cmds.first()
        .filter(|c| c.path.is_empty())
        .ok_or_else(|| "the help walk does not start at `p2a`".to_string())
}

/// The command reference: global options, the top-level commands, then one subsection per command.
pub fn render_reference(cmds: &[CommandHelp]) -> Result<String, String> {
    let root = root(cmds)?;
    let globals: Vec<&str> = root.options.iter().map(|e| e.spec.as_str()).collect();
    let mut md = String::from("## Command reference\n\n");
    md.push_str(
        "Rendered from this build's `p2a <command> -h`; `p2a <command> --help` prints the long descriptions.\n\n",
    );
    md.push_str(
        "### Global options\n\nEvery command accepts these, before or after the subcommand.\n\n",
    );
    md.push_str("| Option | Description |\n|---|---|\n");
    for e in &root.options {
        md.push_str(&format!("| {} | {} |\n", code(&e.spec), cell(&e.help)));
    }
    md.push_str("\n### Commands\n\n");
    commands_table(&mut md, root);
    for c in &cmds[1..] {
        let Some((name, parent_path)) = c.path.split_last() else {
            return Err("a command without a name in the help walk".to_string());
        };
        let parent = cmds
            .iter()
            .find(|p| p.path.as_slice() == parent_path)
            .ok_or_else(|| format!("no parent help for `p2a {}`", c.path.join(" ")))?;
        let aliases = parent
            .subcommands()
            .into_iter()
            .find(|(n, _, _)| n == name)
            .map(|(_, _, a)| a)
            .unwrap_or_default();
        let level = if c.path.len() == 1 { "###" } else { "####" };
        md.push_str(&format!("{level} `p2a {}`", c.path.join(" ")));
        if !aliases.is_empty() {
            md.push_str(&format!(" (alias {})", code_list(&aliases)));
        }
        md.push_str("\n\n");
        if !c.about.is_empty() {
            md.push_str(&cell(&c.about));
            md.push_str("\n\n");
        }
        md.push_str(&format!("```text\n{}\n```\n\n", c.usage));
        if c.subcommands().is_empty() {
            let rows: Vec<&Entry> = c
                .arguments
                .iter()
                .chain(
                    c.options
                        .iter()
                        .filter(|e| !globals.contains(&e.spec.as_str())),
                )
                .collect();
            if !rows.is_empty() {
                md.push_str("| Argument or option | Description |\n|---|---|\n");
                for e in rows {
                    md.push_str(&format!("| {} | {} |\n", code(&e.spec), cell(&e.help)));
                }
                md.push('\n');
            }
        } else {
            commands_table(&mut md, c);
        }
    }
    Ok(format!("{}\n", md.trim_end()))
}

/// README's list: each top-level command with its aliases, description and subcommands.
pub fn render_command_list(cmds: &[CommandHelp]) -> Result<String, String> {
    let root = root(cmds)?;
    let mut md = String::new();
    for (name, about, aliases) in root.subcommands() {
        let sub = cmds
            .iter()
            .find(|c| c.path.len() == 1 && c.path[0] == name)
            .ok_or_else(|| format!("`p2a {name}` is missing from the help walk"))?;
        md.push_str(&format!("- {}", code(&name)));
        if !aliases.is_empty() {
            md.push_str(&format!(" (alias {})", code_list(&aliases)));
        }
        md.push_str(&format!(": {}", cell(&about)));
        let subs: Vec<String> = sub.subcommands().into_iter().map(|(n, _, _)| n).collect();
        if !subs.is_empty() {
            md.push_str(&format!(". Subcommands: {}", code_list(&subs)));
        }
        md.push('\n');
    }
    Ok(md)
}

/// Each generated section with its current body.
pub fn render_all(cmds: &[CommandHelp]) -> Result<Vec<(Section, String)>, String> {
    Ok(vec![
        (REFERENCE, render_reference(cmds)?),
        (COMMAND_LIST, render_command_list(cmds)?),
    ])
}

/// Byte range of the text strictly between the section's marker lines.
fn section_range(doc: &str, s: Section) -> Result<Range<usize>, String> {
    let (begin, end) = (s.begin(), s.end());
    let start = doc
        .find(&begin)
        .ok_or_else(|| format!("{}: no `{begin}` line", s.file))?;
    let body = start + begin.len();
    if doc[body..].contains(&begin) {
        return Err(format!(
            "{}: the `{}` section appears twice",
            s.file, s.name
        ));
    }
    let len = doc[body..]
        .find(&end)
        .ok_or_else(|| format!("{}: no `{end}` line after the BEGIN marker", s.file))?;
    Ok(body..body + len)
}

/// The text between a section's markers.
pub fn section_body(doc: &str, s: Section) -> Result<&str, String> {
    Ok(&doc[section_range(doc, s)?])
}

/// What a section holds when `body` is current: a newline, the body, a newline.
pub fn wrap_body(body: &str) -> String {
    format!("\n{}\n", body.trim_end())
}

/// `doc` with the section's text replaced by `body`.
pub fn splice(doc: &str, s: Section, body: &str) -> Result<String, String> {
    let r = section_range(doc, s)?;
    Ok(format!(
        "{}{}{}",
        &doc[..r.start],
        wrap_body(body),
        &doc[r.end..]
    ))
}

/// The workspace root (two levels above this crate).
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// Rewrites every stale section (a temporary file, then a rename); returns the files written.
pub fn update(cmds: &[CommandHelp]) -> Result<Vec<&'static str>, String> {
    let mut written = Vec::new();
    for (s, body) in render_all(cmds)? {
        let path = repo_root().join(s.file);
        let doc = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", s.file))?;
        let new = splice(&doc, s, &body)?;
        if new != doc {
            // A name of its own: the MCP generator also rewrites README.md.
            let tmp = path.with_extension(format!("md.{}.cli.tmp", std::process::id()));
            std::fs::write(&tmp, &new).map_err(|e| {
                let _ = std::fs::remove_file(&tmp);
                format!("{}: {e}", tmp.display())
            })?;
            std::fs::rename(&tmp, &path).map_err(|e| {
                let _ = std::fs::remove_file(&tmp);
                format!("{}: {e}", path.display())
            })?;
            written.push(s.file);
        }
    }
    Ok(written)
}

/// `$P2A_BIN`, else `p2a` next to this executable's directory (`target/<profile>/examples/..`).
pub fn default_bin() -> PathBuf {
    if let Some(p) = std::env::var_os("P2A_BIN") {
        return PathBuf::from(p);
    }
    let name = format!("p2a{}", std::env::consts::EXE_SUFFIX);
    std::env::current_exe()
        .ok()
        .and_then(|exe| {
            exe.parent()
                .and_then(Path::parent)
                .map(|dir| dir.join(&name))
        })
        .unwrap_or_else(|| PathBuf::from(&name))
}

fn main() -> ExitCode {
    let bin = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(default_bin);
    if !bin.is_file() {
        eprintln!(
            "no p2a binary at {}: run `cargo build -p p2a-cli` first, or pass its path (or set P2A_BIN)",
            bin.display()
        );
        return ExitCode::FAILURE;
    }
    match walk(&bin).and_then(|cmds| update(&cmds)) {
        Ok(written) if written.is_empty() => {
            println!("the generated sections are current");
            ExitCode::SUCCESS
        }
        Ok(written) => {
            for f in written {
                println!("rewrote {f}");
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
