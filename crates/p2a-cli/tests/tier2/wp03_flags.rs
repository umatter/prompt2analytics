//! T2-WP03: global flags, subcommand short flags, session recording, data save/load through the p2a CLI.
use crate::common::*;

#[test]
fn test_tier2_cli_helpers() {
    let out = p2a().arg("--version").output().unwrap();
    assert!(out.status.success());
    let s = temp_session();
    assert_eq!(s.json(&["data", "list"])["message"], "No datasets loaded");
    assert!(
        s.path().exists(),
        "the session file is written by the first command"
    );
}
