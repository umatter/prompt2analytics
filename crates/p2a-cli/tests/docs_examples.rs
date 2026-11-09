//! The CLI documentation is checked against the `p2a` binary and the source tree (T3-WP63).
//!
//! - Every `p2a ...` command in a shell fence (```bash, ```sh, ```console or unlabelled) or inline code
//!   of README.md, docs/cookbook.md, docs/guides/CLI_REFERENCE.md and CLAUDE.md runs through the binary
//!   in an empty directory and must not be a usage error (exit 2) or a panic; table cells name real
//!   subcommands (INF-3).
//! - The after-help EXAMPLES of every command parse (INF-3).
//! - The generated command reference and command list are what the binary's help renders (INF-3).
//! - CLAUDE.md names only source files that exist (WT-17, INF-10).
//!
//! The commands run with `--session` removed, so one that needs a dataset stops with "No session
//! active" (exit 1): only the command line is tested. `chat` gets `--help` appended so nothing
//! connects to a server. Each run has a 60 s hang guard; that is not a timing assertion.
#![cfg(feature = "spectral-analysis")]

#[path = "../examples/generate_cli_reference.rs"]
mod generator;

use generator::{CommandHelp, open_quote, shell_words};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

const DOCS: [&str; 4] = [
    "README.md",
    "docs/cookbook.md",
    "docs/guides/CLI_REFERENCE.md",
    "CLAUDE.md",
];
/// Fence info strings whose `p2a` lines are examples; ```text and the rest are not checked.
const SHELL_FENCES: [&str; 6] = ["", "bash", "sh", "shell", "console", "zsh"];
/// Commands that need no session.
const SESSIONLESS: [&str; 3] = ["script", "chat", "smoke-test"];

fn bin() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_p2a"))
}

fn read(rel: &str) -> String {
    std::fs::read_to_string(generator::repo_root().join(rel))
        .unwrap_or_else(|e| panic!("{rel}: {e}"))
}

/// The help of every command, read once per test binary.
fn commands() -> &'static [CommandHelp] {
    static WALK: OnceLock<Result<Vec<CommandHelp>, String>> = OnceLock::new();
    match WALK.get_or_init(|| generator::walk(bin())) {
        Ok(c) => c,
        Err(e) => panic!("reading the help of every command failed: {e}"),
    }
}

fn update_requested() -> bool {
    std::env::var_os("P2A_UPDATE_DOCS").is_some_and(|v| v == "1")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Fenced,
    Inline,
    Table,
}

#[derive(Debug, Clone)]
struct Doc {
    file: &'static str,
    line: usize,
    text: String,
    kind: Kind,
}

impl Doc {
    fn label(&self) -> String {
        format!("{}:{}: {}", self.file, self.line, self.text)
    }
}

/// The document with the lines of every generated section blanked (line numbers kept).
fn without_generated(md: &str) -> String {
    let mut out = Vec::new();
    let mut inside = false;
    for line in md.lines() {
        if line.starts_with("<!-- BEGIN GENERATED:") {
            inside = true;
            out.push("");
        } else if line.starts_with("<!-- END GENERATED:") {
            inside = false;
            out.push("");
        } else {
            out.push(if inside { "" } else { line });
        }
    }
    out.join("\n")
}

/// `(marker char, marker length, info string)` of a fence line.
fn fence(t: &str) -> Option<(char, usize, &str)> {
    let c = t.chars().next().filter(|c| *c == '`' || *c == '~')?;
    let n = t.chars().take_while(|x| *x == c).count();
    (n >= 3).then(|| (c, n, t[n..].trim()))
}

/// The text from `p2a` on, after an optional `$ ` prompt and `NAME=value` environment words.
fn command_start(text: &str) -> Option<&str> {
    let t = text.trim_start();
    let mut rest = t.strip_prefix("$ ").unwrap_or(t).trim_start();
    loop {
        let word = rest.split_whitespace().next()?;
        if word == "p2a" {
            return Some(rest);
        }
        let env = word.split_once('=').is_some_and(|(k, _)| {
            !k.is_empty()
                && k.chars()
                    .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
        });
        if !env {
            return None;
        }
        rest = rest[word.len()..].trim_start();
    }
}

fn needs_more(s: &str) -> bool {
    s.trim_end().ends_with('\\') || open_quote(s)
}

fn continue_with(acc: &mut String, line: &str) {
    let trimmed = acc.trim_end();
    let kept = trimmed
        .strip_suffix('\\')
        .unwrap_or(trimmed)
        .trim_end()
        .to_string();
    *acc = kept;
    acc.push(' ');
    acc.push_str(line.trim());
}

/// Why a command line is a synopsis rather than an example, if it is one.
fn placeholder(text: &str) -> Option<&'static str> {
    let chars: Vec<char> = text.chars().collect();
    for (i, c) in chars.iter().enumerate() {
        let next = chars.get(i + 1).copied().unwrap_or(' ');
        if *c == '<' && (next.is_ascii_alphabetic() || next == '_') && chars[i + 1..].contains(&'>')
        {
            return Some("a <PLACEHOLDER>");
        }
        if *c == '[' && (next == '-' || next == '<') {
            return Some("an [optional] group");
        }
        if *c == '$' {
            return Some("a shell variable");
        }
    }
    text.split_whitespace()
        .any(|w| w == "...")
        .then_some("an ellipsis")
}

/// Commands of a document: shell-fence lines, inline `p2a ...` spans, and table cells that start with
/// a top-level command name or alias.
fn documented(file: &'static str, md: &str, top: &[String]) -> Vec<Doc> {
    let md = without_generated(md);
    let lines: Vec<&str> = md.lines().collect();
    let mut out = Vec::new();
    let mut open: Option<(char, usize, bool)> = None;
    let mut i = 0;
    while i < lines.len() {
        let raw = lines[i];
        let t = raw.trim_start();
        if let Some((c, n, checked)) = open {
            if let Some((c2, n2, info)) = fence(t)
                && c2 == c
                && n2 >= n
                && info.is_empty()
            {
                open = None;
            } else if checked && let Some(cmd) = command_start(raw) {
                let start = i;
                let mut text = cmd.trim_end().to_string();
                while needs_more(&text)
                    && i + 1 < lines.len()
                    && fence(lines[i + 1].trim_start()).is_none()
                {
                    i += 1;
                    continue_with(&mut text, lines[i]);
                }
                out.push(Doc {
                    file,
                    line: start + 1,
                    text,
                    kind: Kind::Fenced,
                });
            }
            i += 1;
            continue;
        }
        if let Some((c, n, info)) = fence(t) {
            let lang = info
                .split_whitespace()
                .next()
                .unwrap_or("")
                .to_ascii_lowercase();
            open = Some((c, n, SHELL_FENCES.contains(&lang.as_str())));
            i += 1;
            continue;
        }
        if !t.starts_with('#') {
            let table = t.starts_with('|');
            for span in raw.split('`').skip(1).step_by(2) {
                if let Some(cmd) = command_start(span)
                    && cmd.split_whitespace().count() >= 2
                {
                    out.push(Doc {
                        file,
                        line: i + 1,
                        text: cmd.to_string(),
                        kind: Kind::Inline,
                    });
                } else if table
                    && span
                        .split_whitespace()
                        .next()
                        .is_some_and(|w| top.iter().any(|n| n == w))
                {
                    out.push(Doc {
                        file,
                        line: i + 1,
                        text: span.to_string(),
                        kind: Kind::Table,
                    });
                }
            }
        }
        i += 1;
    }
    out
}

/// The names and aliases of `p2a`'s top-level commands.
fn top_level_names() -> Vec<String> {
    let mut names = Vec::new();
    for (name, _, aliases) in commands()[0].subcommands() {
        names.push(name);
        names.extend(aliases);
    }
    names
}

/// The subcommand path a table cell names (`reg ols`), through names and aliases; it stops at a leaf
/// or a flag.
fn resolve(words: &[&str]) -> Result<(), String> {
    let cmds = commands();
    let mut path: Vec<String> = Vec::new();
    for w in words {
        if w.starts_with('-') {
            break;
        }
        let Some(cur) = cmds.iter().find(|c| c.path == path) else {
            return Err(format!("`p2a {}` is not a command", path.join(" ")));
        };
        let subs = cur.subcommands();
        if subs.is_empty() {
            break;
        }
        let Some((name, _, _)) = subs
            .iter()
            .find(|(n, _, a)| n == w || a.iter().any(|x| x == w))
        else {
            let known: Vec<&str> = subs.iter().map(|s| s.0.as_str()).collect();
            return Err(format!(
                "`{w}` is not a subcommand of `p2a {}` (it has {})",
                path.join(" "),
                known.join(", ")
            ));
        };
        path.push(name.clone());
    }
    Ok(())
}

/// The arguments of a documented command: the words after `p2a`, without `--session <file>`, and
/// `--help` appended to `chat`.
fn args_of(text: &str) -> Result<Vec<String>, String> {
    let start = command_start(text).ok_or_else(|| format!("not a p2a command: `{text}`"))?;
    let mut words = shell_words(start)?.into_iter().skip(1);
    let mut args = Vec::new();
    while let Some(w) = words.next() {
        if w == "--session" {
            words.next();
            continue;
        }
        if w.starts_with("--session=") {
            continue;
        }
        args.push(w);
    }
    if subcommand_of(&args) == Some("chat") {
        args.push("--help".to_string());
    }
    Ok(args)
}

/// The first word that is neither a flag nor the value of `-F`/`--format`/`--max-memory`.
fn subcommand_of(args: &[String]) -> Option<&str> {
    let mut skip = false;
    for a in args {
        if skip {
            skip = false;
            continue;
        }
        if matches!(a.as_str(), "-F" | "--format" | "--max-memory") {
            skip = true;
            continue;
        }
        if !a.starts_with('-') {
            return Some(a);
        }
    }
    None
}

fn names_session(text: &str) -> bool {
    shell_words(text).is_ok_and(|w| {
        w.iter()
            .any(|x| x == "--session" || x.starts_with("--session="))
    })
}

struct Run {
    code: Option<i32>,
    stderr: String,
    timed_out: bool,
}

/// Runs `p2a <args>` in a fresh directory with a cleared environment.
fn run(args: &[String]) -> Run {
    let dir = tempfile::tempdir().expect("temporary directory");
    let err_path = dir.path().join("stderr.txt");
    let err = std::fs::File::create(&err_path).expect("stderr file");
    let mut child = Command::new(bin())
        .args(args)
        .current_dir(dir.path())
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("HOME", dir.path())
        .env("NO_COLOR", "1")
        .env("COLUMNS", "100")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(err))
        .spawn()
        .expect("start p2a");
    // Hang guard, not a timing assertion: parsing takes milliseconds.
    let deadline = Instant::now() + Duration::from_secs(60);
    let status = loop {
        match child.try_wait().expect("wait for p2a") {
            Some(status) => break Some(status),
            None if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            None => std::thread::sleep(Duration::from_millis(5)),
        }
    };
    Run {
        code: status.and_then(|s| s.code()),
        stderr: std::fs::read_to_string(&err_path).unwrap_or_default(),
        timed_out: status.is_none(),
    }
}

fn first_line<'a>(text: &'a str, needle: &str) -> &'a str {
    text.lines()
        .find(|l| l.contains(needle))
        .or_else(|| text.lines().next())
        .unwrap_or("")
        .trim()
}

/// What is wrong with a run, if anything: a usage error, a panic, a signal or a hang.
fn failure(r: &Run) -> Option<String> {
    if r.timed_out {
        return Some("did not finish within 60 s".to_string());
    }
    if r.stderr.contains("panicked at") {
        return Some(format!(
            "panicked: {}",
            first_line(&r.stderr, "panicked at")
        ));
    }
    match r.code {
        Some(2) => Some(format!("usage error: {}", first_line(&r.stderr, "error:"))),
        Some(101) => Some("exit 101".to_string()),
        None => Some("killed by a signal".to_string()),
        Some(_) => None,
    }
}

/// Runs the jobs on up to four threads; returns "label: problem" per failure, sorted.
fn run_all(jobs: &[(String, Vec<String>)]) -> Vec<String> {
    let next = AtomicUsize::new(0);
    let failures = Mutex::new(Vec::new());
    let workers = std::thread::available_parallelism()
        .map_or(2, |n| n.get())
        .min(4);
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some((label, args)) = jobs.get(i) else {
                        break;
                    };
                    if let Some(why) = failure(&run(args)) {
                        failures
                            .lock()
                            .expect("lock")
                            .push(format!("{label}: {why}"));
                    }
                }
            });
        }
    });
    let mut out = failures.into_inner().expect("lock");
    out.sort();
    out
}

// ---------------------------------------------------------------- Task 1

/// INF-3: the generated command reference (CLI_REFERENCE.md) and command list (README.md) are what
/// this binary's help renders; `P2A_UPDATE_DOCS=1` rewrites them. Fails before: neither marker exists.
#[test]
fn test_generated_cli_sections_are_current() {
    let cmds = commands();
    if update_requested() {
        let written = generator::update(cmds).unwrap_or_else(|e| panic!("{e}"));
        eprintln!("rewrote {written:?}");
    }
    let mut stale = Vec::new();
    for (s, body) in generator::render_all(cmds).unwrap_or_else(|e| panic!("{e}")) {
        let doc = read(s.file);
        match generator::section_body(&doc, s) {
            Ok(have) if have == generator::wrap_body(&body) => {}
            Ok(_) => stale.push(format!("{}: the `{}` section is stale", s.file, s.name)),
            Err(e) => stale.push(e),
        }
    }
    assert!(
        stale.is_empty(),
        "{}\nrun `P2A_UPDATE_DOCS=1 cargo test -p p2a-cli --test docs_examples` \
         (or `cargo build -p p2a-cli && cargo run -p p2a-cli --example generate_cli_reference`)",
        stale.join("\n")
    );
}

/// The help parser reads clap's same-line and next-line layouts, long-only options, aliases and an
/// after-help example continued by `\` and then by an open quote (no binary needed).
#[test]
fn test_help_parser_reads_both_clap_layouts() {
    let text = "Regression and more\n\nUsage: p2a reg [OPTIONS] --dep-var <DEP_VAR> <DATASET>\n\n\
        Commands:\n  ols   Ordinary Least Squares [aliases: o]\n  help  Print this message\n\n\
        Arguments:\n  <DATASET>  Dataset name\n\n\
        Options:\n  -y, --dep-var <DEP_VAR>           Dependent variable column\n      \
        --session <SESSION>           Session file\n      --treatment-time <TREATMENT_TIME>\n          \
        Treatment timing column [default: 0]\n\n\
        EXAMPLES:\n    # Generate\n    p2a --session s.json data generate -n 10 -d sim \\\n        \
        --columns '[{\"name\":\"x\"},\n                    {\"name\":\"e\"}]'\n    p2a reg ols d -y y\n";
    let h = generator::parse_help(&["reg".to_string()], text).expect("parse");
    assert_eq!(h.about, "Regression and more");
    assert_eq!(h.usage, "p2a reg [OPTIONS] --dep-var <DEP_VAR> <DATASET>");
    assert_eq!(
        h.subcommands(),
        vec![(
            "ols".to_string(),
            "Ordinary Least Squares".to_string(),
            vec!["o".to_string()]
        )]
    );
    assert_eq!(h.arguments[0].spec, "<DATASET>");
    let specs: Vec<&str> = h.options.iter().map(|e| e.spec.as_str()).collect();
    assert_eq!(
        specs,
        [
            "-y, --dep-var <DEP_VAR>",
            "--session <SESSION>",
            "--treatment-time <TREATMENT_TIME>"
        ]
    );
    assert_eq!(h.options[2].help, "Treatment timing column [default: 0]");
    assert_eq!(h.examples.len(), 2);
    assert!(
        h.examples[0].ends_with("{\"name\":\"e\"}]'"),
        "{}",
        h.examples[0]
    );
    assert_eq!(
        shell_words(&h.examples[0])
            .expect("words")
            .last()
            .map(String::as_str),
        Some("[{\"name\":\"x\"}, {\"name\":\"e\"}]")
    );
}

/// The shell splitter keeps quoted words, stops at pipes and comments, and refuses an open quote.
#[test]
fn test_shell_words_split_like_a_shell() {
    let w = shell_words("p2a --session \"my s.json\" reg ols d -x 'a b' c | jq . # note")
        .expect("words");
    assert_eq!(
        w,
        [
            "p2a",
            "--session",
            "my s.json",
            "reg",
            "ols",
            "d",
            "-x",
            "a b",
            "c"
        ]
    );
    assert!(shell_words("p2a data load 'x.csv").is_err());
    assert_eq!(
        args_of("p2a --session s.json -F json reg ols d").expect("args"),
        ["-F", "json", "reg", "ols", "d"]
    );
    assert_eq!(
        args_of("p2a chat --url http://x").expect("args"),
        ["chat", "--url", "http://x", "--help"]
    );
}

/// Synopses are recognised; concrete JSON arguments are not mistaken for them.
#[test]
fn test_placeholders_are_recognised() {
    assert!(placeholder("p2a reg ols <DATASET> -y <DEP_VAR>").is_some());
    assert!(placeholder("p2a data load file.csv [--name NAME]").is_some());
    assert!(placeholder("p2a --format json ...").is_some());
    assert!(placeholder("p2a data generate --columns '[{\"name\":\"x\"}]'").is_none());
}
