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
