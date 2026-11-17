//! T4-WP33: session files under parallel use, sessions and exported scripts from other
//! directories, `script run`.

use super::common::*;
use std::path::Path;
use std::process::Command;

/// A CSV of `n` rows `x,y` with noise around y = 1 + 2x.
fn noisy_csv(path: &Path, n: usize) {
    let mut s = String::from("x,y\n");
    for i in 0..n {
        let x = i as f64 * 0.5 + 0.25;
        let noise = [0.31, -0.27, 0.12, -0.45, 0.38, -0.08, 0.22][i % 7];
        s.push_str(&format!("{x},{}\n", 1.0 + 2.0 * x + noise));
    }
    std::fs::write(path, s).unwrap();
}

/// `script history -F json` of `session`: (command line, success) per command.
fn history(session: &Path) -> Vec<(String, bool)> {
    let out = p2a()
        .args(["-F", "json", "script", "history"])
        .arg(session)
        .output()
        .unwrap();
    assert!(out.status.success(), "{out:?}");
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    v["commands"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            (
                c["command"].as_str().unwrap().to_string(),
                c["success"].as_bool().unwrap(),
            )
        })
        .collect()
}

#[cfg(unix)]
#[test]
fn test_cli_parallel_invocations_keep_every_command() {
    let s = temp_session();
    noisy_csv(&s.dir().join("d.csv"), 20);
    s.ok(&["data", "load", "d.csv", "--name", "d"]);
    let children: Vec<_> = (0..12)
        .map(|_| {
            s.cmd(&["reg", "ols", "d", "-y", "y", "-x", "x"])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .unwrap()
        })
        .collect();
    for child in children {
        let out = child.wait_with_output().unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let h = history(s.path());
    assert_eq!(h.len(), 13, "{h:?}");
    assert!(h.iter().all(|(_, ok)| *ok), "{h:?}");
}

#[test]
fn test_cli_mktemp_session_file_works() {
    let s = temp_session();
    noisy_csv(&s.dir().join("d.csv"), 10);
    std::fs::write(s.path(), "").unwrap();
    s.ok(&["data", "load", "d.csv", "--name", "d"]);
    assert_eq!(history(s.path()).len(), 1);
}

#[test]
fn test_cli_session_works_from_another_directory() {
    let a = tempfile::tempdir().unwrap();
    let b = tempfile::tempdir().unwrap();
    noisy_csv(&a.path().join("data.csv"), 12);
    let session = a.path().join("s.json");
    let out = p2a()
        .arg("--session")
        .arg(&session)
        .args(["data", "load", "data.csv", "--name", "d"])
        .current_dir(a.path())
        .output()
        .unwrap();
    assert!(out.status.success(), "{out:?}");
    let out = p2a()
        .arg("--session")
        .arg(&session)
        .args(["reg", "ols", "d", "-y", "y", "-x", "x"])
        .current_dir(b.path())
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert!(!stderr.contains("Warning"), "{stderr}");
}

/// `p2a script export <session> -o <out> <extra…>` (asserts success).
fn export(session: &Path, out: &Path, extra: &[&str]) {
    let o = p2a()
        .args(["script", "export"])
        .arg(session)
        .arg("-o")
        .arg(out)
        .args(extra)
        .output()
        .unwrap();
    assert!(o.status.success(), "{o:?}");
}

/// A PATH with only the system directories: no `p2a` on it.
const SYSTEM_PATH: &str = "/usr/bin:/bin";

/// `bash <script>` in `cwd` with `P2A` set to this build's binary, a system PATH and
/// `TMPDIR` = `tmp`; `extra` adds environment variables.
#[cfg(unix)]
fn replay(script: &Path, cwd: &Path, tmp: &Path, extra: &[(&str, &Path)]) -> std::process::Output {
    let mut c = Command::new("bash");
    c.arg(script)
        .current_dir(cwd)
        .env_remove("P2A_WORKDIR")
        .env("P2A", env!("CARGO_BIN_EXE_p2a"))
        .env("PATH", SYSTEM_PATH)
        .env("TMPDIR", tmp);
    for (k, v) in extra {
        c.env(k, v);
    }
    c.output().unwrap()
}

fn entries(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

#[cfg(unix)]
#[test]
fn test_cli_replay_skips_failed_commands() {
    let s = temp_session();
    noisy_csv(&s.dir().join("d.csv"), 14);
    s.ok(&["data", "load", "d.csv", "--name", "d"]);
    s.err(&["reg", "ols", "d", "-y", "y", "-x", "xx"]);
    s.ok(&["reg", "ols", "d", "-y", "y", "-x", "x"]);
    let script = s.dir().join("r.sh");
    export(s.path(), &script, &[]);
    let tmp = tempfile::tempdir().unwrap();
    let out = replay(&script, s.dir(), tmp.path(), &[]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(text_table(&stdout).names, ["(Intercept)", "x"], "{stdout}");
    assert!(
        stdout.ends_with("Script completed successfully\n"),
        "{stdout}"
    );
}

#[cfg(unix)]
#[test]
fn test_cli_replay_runs_in_the_recorded_directory() {
    let a = tempfile::tempdir().unwrap();
    let b = tempfile::tempdir().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    noisy_csv(&a.path().join("d.csv"), 12);
    let session = a.path().join("s.json");
    for args in [
        &["data", "load", "d.csv", "--name", "d"][..],
        &["data", "head", "d", "-n", "2"][..],
    ] {
        let o = p2a()
            .arg("--session")
            .arg(&session)
            .args(args)
            .current_dir(a.path())
            .output()
            .unwrap();
        assert!(o.status.success(), "{o:?}");
    }
    let script = b.path().join("r.sh");
    export(&session, &script, &[]);
    let out = replay(&script, b.path(), tmp.path(), &[]);
    assert!(out.status.success(), "{out:?}");
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("Rows: 12"),
        "{out:?}"
    );

    // P2A_WORKDIR: a copy of the directory, with its own data, is read instead
    let copy = tempfile::tempdir().unwrap();
    std::fs::write(
        copy.path().join("d.csv"),
        "x,y\n1.5,3.25\n2.5,5.75\n3.5,7.5\n",
    )
    .unwrap();
    std::fs::remove_file(a.path().join("d.csv")).unwrap();
    let out = replay(
        &script,
        b.path(),
        tmp.path(),
        &[("P2A_WORKDIR", copy.path())],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("Rows: 3"), "{stdout}");
    assert_eq!(entries(tmp.path()), Vec::<String>::new());
}

#[cfg(unix)]
#[test]
fn test_cli_replay_uses_the_p2a_variable() {
    let s = temp_session();
    noisy_csv(&s.dir().join("d.csv"), 8);
    s.ok(&["data", "load", "d.csv", "--name", "d"]);
    let script = s.dir().join("r.sh");
    export(s.path(), &script, &[]);
    let tmp = tempfile::tempdir().unwrap();
    let out = replay(&script, s.dir(), tmp.path(), &[]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[cfg(unix)]
#[test]
fn test_cli_failed_replay_leaves_nothing() {
    let s = temp_session();
    noisy_csv(&s.dir().join("a.csv"), 10);
    noisy_csv(&s.dir().join("b.csv"), 10);
    s.ok(&["data", "load", "a.csv", "--name", "a"]);
    s.ok(&[
        "munge", "filter", "a", "-c", "x", "-o", "gt", "-v", "1", "-n", "sub",
    ]);
    s.ok(&["data", "load", "b.csv", "--name", "b"]);
    let script = s.dir().join("r.sh");
    export(s.path(), &script, &[]);
    std::fs::remove_file(s.dir().join("b.csv")).unwrap();
    let before = entries(s.dir());
    let tmp = tempfile::tempdir().unwrap();
    let out = replay(&script, s.dir(), tmp.path(), &[]);
    assert!(!out.status.success(), "{out:?}");
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("Created dataset 'sub'"),
        "{out:?}"
    );
    assert_eq!(entries(tmp.path()), Vec::<String>::new());
    assert_eq!(entries(s.dir()), before);
}

/// A script file `name` in `dir` with `body`.
fn script_file(dir: &Path, name: &str, body: &str) -> std::path::PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, body).unwrap();
    p
}

#[cfg(unix)]
#[test]
fn test_cli_script_run_returns_the_script_status() {
    let dir = tempfile::tempdir().unwrap();
    let script = script_file(dir.path(), "s.sh", "echo one\necho two >&2\nexit 3\n");
    let out = p2a()
        .args(["script", "run"])
        .arg(&script)
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(3), "{stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        format!("Running script: {}\none\n", script.display())
    );
    assert!(stderr.contains("two"), "{stderr}");
    assert!(
        stderr.contains("Script failed with exit code 3"),
        "{stderr}"
    );
}

#[cfg(unix)]
#[test]
fn test_cli_script_run_passes_stdin_and_the_binary() {
    use std::io::Write as _;
    let dir = tempfile::tempdir().unwrap();
    let script = script_file(
        dir.path(),
        "s.sh",
        "read x\necho \"got $x\"\n\"$P2A\" --version\n",
    );
    let mut child = p2a()
        .args(["script", "run"])
        .arg(&script)
        .current_dir(dir.path())
        .env_remove("P2A")
        .env("PATH", SYSTEM_PATH)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"hi\n").unwrap();
    let out = child.wait_with_output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("got hi\n"), "{stdout}");
    assert!(
        stdout.contains(&format!("p2a {}", env!("CARGO_PKG_VERSION"))),
        "{stdout}"
    );
}

#[cfg(unix)]
#[test]
fn test_cli_script_run_json_mode_prints_only_the_child_output() {
    let dir = tempfile::tempdir().unwrap();
    let script = script_file(dir.path(), "s.sh", "echo plain\n");
    let out = p2a()
        .args(["-F", "json", "script", "run"])
        .arg(&script)
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(out.status.success(), "{out:?}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "plain\n");
}
