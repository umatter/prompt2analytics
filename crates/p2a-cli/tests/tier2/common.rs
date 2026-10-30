//! Helpers shared by every Tier 2 CLI test module: the debug `p2a` binary,
//! a scratch session, and output assertions.

pub use p2a_validation::{Case, Check, Tol, repo_path};

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// The `p2a` binary Cargo built for this run (`env!("CARGO_BIN_EXE_p2a")`): the debug profile under
/// `cargo test`, so clap's debug assertions are active.
pub fn p2a() -> Command {
    Command::new(env!("CARGO_BIN_EXE_p2a"))
}

/// A scratch directory and a session-file path inside it (`<dir>/session.json`, created by the
/// first command).
pub struct TempSession {
    dir: tempfile::TempDir,
    path: PathBuf,
}

/// A fresh scratch session.
pub fn temp_session() -> TempSession {
    let dir = tempfile::tempdir().expect("temporary directory");
    let path = dir.path().join("session.json");
    TempSession { dir, path }
}

/// `args` as one line for failure messages.
fn shown(args: &[&str]) -> String {
    args.join(" ")
}

/// Stdout and stderr of `out` for failure messages.
fn streams(out: &Output) -> String {
    format!(
        "exit: {:?}\n--- stdout\n{}\n--- stderr\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

impl TempSession {
    /// The scratch directory (the working directory of every command).
    pub fn dir(&self) -> &Path {
        self.dir.path()
    }

    /// The session file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// `p2a --session <path> <args…>` with the scratch directory as working directory.
    pub fn cmd(&self, args: &[&str]) -> Command {
        let mut cmd = p2a();
        cmd.arg("--session")
            .arg(&self.path)
            .args(args)
            .current_dir(self.dir());
        cmd
    }

    /// Runs `p2a --session <path> <args…>` and returns its output.
    pub fn run(&self, args: &[&str]) -> Output {
        self.cmd(args)
            .output()
            .unwrap_or_else(|e| panic!("could not run p2a {}: {e}", shown(args)))
    }

    /// Runs; asserts exit 0 and no panic (message shows args, stdout, stderr); returns stdout.
    pub fn ok(&self, args: &[&str]) -> String {
        let out = self.run(args);
        assert_no_panic(&out, &shown(args));
        assert!(
            out.status.success(),
            "p2a {} failed\n{}",
            shown(args),
            streams(&out)
        );
        String::from_utf8_lossy(&out.stdout).into_owned()
    }

    /// Runs; asserts exit code 1 (an error reported by p2a, not a panic (101) or a usage
    /// error (2)); returns stderr.
    pub fn err(&self, args: &[&str]) -> String {
        let out = self.run(args);
        assert_no_panic(&out, &shown(args));
        assert_eq!(
            out.status.code(),
            Some(1),
            "p2a {} should exit 1\n{}",
            shown(args),
            streams(&out)
        );
        String::from_utf8_lossy(&out.stderr).into_owned()
    }

    /// `-F json` followed by `args`.
    fn json_args<'a>(args: &[&'a str]) -> Vec<&'a str> {
        let mut all = vec!["-F", "json"];
        all.extend_from_slice(args);
        all
    }

    /// `p2a --session <path> -F json <args…>`: asserts success, parses stdout as one JSON value.
    pub fn json(&self, args: &[&str]) -> Value {
        let all = Self::json_args(args);
        let stdout = self.ok(&all);
        serde_json::from_str(&stdout).unwrap_or_else(|e| {
            panic!(
                "p2a {}: stdout is not one JSON value ({e})\n{stdout}",
                shown(&all)
            )
        })
    }

    /// Same, for commands that print several JSON documents in a row (`data describe`).
    pub fn json_docs(&self, args: &[&str]) -> Vec<Value> {
        let all = Self::json_args(args);
        let stdout = self.ok(&all);
        serde_json::Deserializer::from_str(&stdout)
            .into_iter::<Value>()
            .collect::<Result<Vec<_>, _>>()
            .unwrap_or_else(|e| {
                panic!(
                    "p2a {}: stdout is not a sequence of JSON values ({e})\n{stdout}",
                    shown(&all)
                )
            })
    }

    /// `data load <case.dataset_path()> --name <file stem>`; returns the name (e.g. "lung").
    pub fn load(&self, case: &Case) -> String {
        let path = case.dataset_path();
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_else(|| panic!("case {}: dataset path has no stem", case.id))
            .to_string();
        self.load_as(&path, &name, &[]);
        name
    }

    /// `data load <path> --name <name> <extra…>` (asserts success).
    pub fn load_as(&self, path: &Path, name: &str, extra: &[&str]) {
        let path = path.to_str().expect("UTF-8 dataset path");
        let mut args = vec!["data", "load", path, "--name", name];
        args.extend_from_slice(extra);
        self.ok(&args);
    }

    /// `commands[*].argv` of the session file, in order.
    pub fn recorded_argv(&self) -> Vec<Vec<String>> {
        let text = std::fs::read_to_string(&self.path)
            .unwrap_or_else(|e| panic!("session file {}: {e}", self.path.display()));
        let session: Value = serde_json::from_str(&text).expect("session file is JSON");
        session["commands"]
            .as_array()
            .expect("session has a commands array")
            .iter()
            .map(|c| {
                c["argv"]
                    .as_array()
                    .expect("command has an argv array")
                    .iter()
                    .map(|t| t.as_str().expect("argv entries are strings").to_string())
                    .collect()
            })
            .collect()
    }

    /// `bash <script>` in the scratch directory with this build's `p2a` first on PATH.
    #[cfg(unix)]
    pub fn replay(&self, script: &Path) -> Output {
        let bin = Path::new(env!("CARGO_BIN_EXE_p2a"));
        let bin_dir = bin.parent().expect("p2a binary has a directory");
        let mut dirs = vec![bin_dir.to_path_buf()];
        if let Some(path) = std::env::var_os("PATH") {
            dirs.extend(std::env::split_paths(&path));
        }
        let path = std::env::join_paths(dirs).expect("PATH entries without ':'");
        Command::new("bash")
            .arg(script)
            .current_dir(self.dir())
            .env("PATH", path)
            .output()
            .unwrap_or_else(|e| panic!("could not run bash {}: {e}", script.display()))
    }
}

/// Exit code is not 101 and stderr has no "panicked at".
pub fn assert_no_panic(out: &Output, what: &str) {
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.code() != Some(101) && !stderr.contains("panicked at"),
        "{what}: p2a panicked\n{}",
        streams(out)
    );
}

/// One CSV cell with surrounding double quotes trimmed.
fn cell(c: &str) -> &str {
    let c = c.trim();
    c.strip_prefix('"')
        .and_then(|c| c.strip_suffix('"'))
        .unwrap_or(c)
}

/// Same number of lines and the same cells (surrounding quotes trimmed); cells that parse as
/// f64 in both files are compared as f64. The files must not contain embedded commas.
pub fn assert_same_csv(a: &Path, b: &Path) {
    let read =
        |p: &Path| std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let (ta, tb) = (read(a), read(b));
    let (la, lb): (Vec<&str>, Vec<&str>) = (ta.lines().collect(), tb.lines().collect());
    assert_eq!(
        la.len(),
        lb.len(),
        "{} and {} have different line counts",
        a.display(),
        b.display()
    );
    for (i, (ra, rb)) in la.iter().zip(&lb).enumerate() {
        let (ca, cb): (Vec<&str>, Vec<&str>) = (
            ra.split(',').map(cell).collect(),
            rb.split(',').map(cell).collect(),
        );
        assert_eq!(
            ca.len(),
            cb.len(),
            "line {}: different cell counts\n{ra}\n{rb}",
            i + 1
        );
        for (x, y) in ca.iter().zip(&cb) {
            let same = match (x.parse::<f64>(), y.parse::<f64>()) {
                (Ok(fx), Ok(fy)) => fx == fy,
                _ => x == y,
            };
            assert!(
                same,
                "line {} of {} and {}: {x:?} != {y:?}\n{ra}\n{rb}",
                i + 1,
                a.display(),
                b.display()
            );
        }
    }
}
