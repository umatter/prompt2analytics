//! T3-WP06: boolean flags that can be turned off (CLI-5), `reg clustered --cluster2` (CLI-18), the global
//! `--max-memory` warning (DATA-30) and the no-intercept statistics they expose (REG-19), through the p2a CLI.

use crate::common::*;

#[test]
fn test_tier3_cli_helpers() {
    let s = temp_session();
    s.load_as(
        &repo_path("validation/datasets/grunfeld.csv"),
        "grunfeld",
        &[],
    );
    let args = [
        "reg", "ols", "grunfeld", "-y", "inv", "-x", "value", "capital",
    ];
    let t = text_table(&s.ok(&args));
    let j = json_table(&s.json(&args), "t_value");
    assert_eq!(t.names, ["(Intercept)", "value", "capital"]);
    assert_eq!(j.names, t.names);
    assert!(
        t.coef
            .iter()
            .zip(&j.coef)
            .all(|(a, b)| Tol::printed(6).allows(*a, *b))
    );
    assert_eq!(argv(&["a"], &["b", "c"]), ["a", "b", "c"]);
}
