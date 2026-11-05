//! T3-WP21: the 2x2 DiD command through the p2a CLI.
use crate::common::*;
use serde_json::Value;

/// An aliased control is dropped with lm's NA and R's singularity warning; the
/// CLI's text and JSON output say so (before, both printed neither).
#[test]
fn test_cli_did_reports_aliased_control() {
    let s = temp_session();
    let path = repo_path("validation/datasets/employee_training.csv");
    s.load_as(&path, "training", &[]);
    let args = |extra: &[&'static str]| -> Vec<&'static str> {
        let mut a = vec![
            "causal",
            "did",
            "training",
            "-y",
            "performance",
            "--treat",
            "training_group",
            "--post",
            "post_training",
        ];
        a.extend_from_slice(extra);
        a
    };
    let aliased = args(&["-x", "tenure", "years_since_hire"]);
    let j = s.json(&aliased);
    assert_eq!(j["aliased"], Value::from(vec!["years_since_hire"]), "{j}");
    let w = j["warnings"].as_array().expect("warnings array");
    assert!(
        w.iter()
            .any(|w| w.as_str().unwrap_or("").contains("singularities")),
        "{j}"
    );
    let txt = s.ok(&aliased);
    assert!(
        txt.contains("Aliased (dropped, coefficient NA): years_since_hire"),
        "{txt}"
    );
    assert!(
        txt.contains("Warning: ") && txt.contains("singularities"),
        "{txt}"
    );

    let j = s.json(&args(&["-x", "tenure"]));
    assert_eq!(j["aliased"], Value::Array(vec![]), "{j}");
    assert_eq!(j["warnings"], Value::Array(vec![]), "{j}");
    let txt = s.ok(&args(&["-x", "tenure"]));
    assert!(
        !txt.contains("Aliased") && !txt.contains("Warning: "),
        "{txt}"
    );
}
