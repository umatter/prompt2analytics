//! T2-WP18: discrete choice (ordered, mixed logit) through the p2a CLI.

use p2a_validation::text::{coefficient_rows, labelled};
use serde_json::{Value, json};

use crate::common::*;

/// The whitespace tokens of the first line whose first token is `name`.
fn row<'a>(text: &'a str, name: &str) -> Vec<&'a str> {
    text.lines()
        .map(|l| l.split_whitespace().collect::<Vec<_>>())
        .find(|tokens| tokens.first() == Some(&name))
        .unwrap_or_else(|| panic!("no row {name:?} in:\n{text}"))
}

/// The numbers of the first line that starts with `start` (leading spaces ignored), once
/// `(),:;` are trimmed from each token.
fn numbers(text: &str, start: &str) -> Vec<f64> {
    let line = text
        .lines()
        .map(str::trim_start)
        .find(|l| l.starts_with(start))
        .unwrap_or_else(|| panic!("no line starting with {start:?} in:\n{text}"));
    line[start.len()..]
        .split_whitespace()
        .filter_map(|t| {
            t.trim_matches(|c| matches!(c, '(' | ')' | ',' | ':' | ';'))
                .parse::<f64>()
                .ok()
        })
        .collect()
}

fn number(token: &str) -> f64 {
    token
        .parse()
        .unwrap_or_else(|_| panic!("{token:?} is not a number"))
}

/// The numbers of a JSON array (`null` is NaN).
fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap_or_else(|| panic!("not an array: {v}"))
        .iter()
        .map(|x| x.as_f64().unwrap_or(f64::NAN))
        .collect()
}

/// The case, a scratch session and the dataset loaded under its file stem.
fn loaded(id: &str) -> (Case, TempSession, String) {
    let case = Case::load(id);
    let s = temp_session();
    let name = s.load(&case);
    (case, s, name)
}

/// The printed o-logit/o-probit fit of `-y y_label --levels ...` against the keys with
/// prefix `p`, and its JSON at full tolerance.
fn check_ordered_levels(case: &Case, s: &TempSession, args: &[&str], p: &str) {
    let k = |key: &str| format!("{p}{key}");
    let t = s.ok(args);
    let xs = ["x1", "x2"];
    let (x1, x2) = (row(&t, "x1"), row(&t, "x2"));
    let cut1 = numbers(&t, "low|mid:");
    let cut2 = numbers(&t, "mid|high:");
    Check::new(case)
        .named(
            &k("coef"),
            &xs,
            &[number(x1[1]), number(x2[1])],
            Tol::printed(6),
        )
        .named(
            &k("se_coef"),
            &xs,
            &[number(x1[2]), number(x2[2])],
            Tol::printed(6),
        )
        .named(
            &k("z"),
            &xs,
            &[number(x1[3]), number(x2[3])],
            Tol::printed(4),
        )
        .vector(&k("zeta"), &[cut1[0], cut2[0]], Tol::printed(6))
        .vector(&k("se_zeta"), &[cut1[1], cut2[1]], Tol::printed(6))
        .finish();
    assert!(t.contains("Converged: Yes ("), "{t}");
    let j = s.json(args);
    Check::new(case)
        .named(
            &k("coef"),
            &xs,
            &floats(&j["coefficients"]),
            Tol::iterative(1e-6),
        )
        .vector(&k("zeta"), &floats(&j["thresholds"]), Tol::iterative(1e-6))
        .named(
            &k("se_coef"),
            &xs,
            &floats(&j["std_errors"]),
            Tol::iterative(1e-5),
        )
        .vector(
            &k("se_zeta"),
            &floats(&j["threshold_std_errors"]),
            Tol::iterative(1e-5),
        )
        .finish();
    assert_eq!(j["categories"], json!(["low", "mid", "high"]), "{j}");
    assert_eq!(j["converged"], json!(true), "{j}");
    assert!(j["iterations"].as_u64().unwrap() >= 1, "{j}");
    assert!(j.get("missing").is_none(), "{j}");
}

/// `--levels low,mid,high` fits the text outcome in that order (it was an unknown
/// argument).
#[test]
fn test_validate_cli_o_logit_levels() {
    let (case, s, name) = loaded("disc_ordered");
    let args = [
        "discrete",
        "o-logit",
        &name,
        "-y",
        "y_label",
        "-x",
        "x1",
        "x2",
        "--levels",
        "low,mid,high",
    ];
    check_ordered_levels(&case, &s, &args, "logit_");
}

/// The same for o-probit, with the levels space-separated.
#[test]
fn test_validate_cli_o_probit_levels() {
    let (case, s, name) = loaded("disc_ordered");
    let args = [
        "discrete", "o-probit", &name, "-y", "y_label", "-x", "x1", "x2", "--levels", "low", "mid",
        "high",
    ];
    check_ordered_levels(&case, &s, &args, "probit_");
}

/// Without --levels a text outcome is an error that names the flag.
#[test]
fn test_cli_o_logit_text_outcome_needs_levels() {
    let (_, s, name) = loaded("disc_ordered");
    let e = s.err(&[
        "discrete", "o-logit", &name, "-y", "y_label", "-x", "x1", "x2",
    ]);
    for part in ["'y_label'", "high", "low", "mid", "--levels"] {
        assert!(e.contains(part), "{part:?} not in {e}");
    }
}

/// Rows with a missing outcome or regressor are dropped and reported.
#[test]
fn test_validate_cli_o_logit_missing_note() {
    let (case, s, name) = loaded("disc_ordered");
    let args = [
        "discrete", "o-logit", &name, "-y", "y_na", "-x", "x1_na", "x2",
    ];
    let t = s.ok(&args);
    let xs = ["x1_na", "x2"];
    let (x1, x2) = (row(&t, "x1_na"), row(&t, "x2"));
    Check::new(&case)
        .named(
            "na_coef",
            &xs,
            &[number(x1[1]), number(x2[1])],
            Tol::printed(6),
        )
        .named(
            "na_se_coef",
            &xs,
            &[number(x1[2]), number(x2[2])],
            Tol::printed(6),
        )
        .finish();
    assert!(
        t.contains("Note: 7 of 400 rows dropped because of missing values (y_na: 4, x1_na: 3)"),
        "{t}"
    );
    let j = s.json(&args);
    assert_eq!(j["missing"]["n_dropped"], json!(7), "{j}");
    assert_eq!(
        j["missing"]["by_column"],
        json!({"y_na": 4, "x1_na": 3}),
        "{j}"
    );
}

/// The printed and JSON mixed-logit fit (x normal, w fixed) against `case`.
fn check_mixed_logit(case: &Case, t: &str, j: &Value, n: usize) {
    // tokens: name, mean, SE, SD, SD SE, dist, p
    let (x, w) = (row(t, "x"), row(t, "w"));
    let names = ["mean_x", "mean_w", "sd_x"];
    Check::new(case)
        .named(
            "coef",
            &names,
            &[number(x[1]), number(w[1]), number(x[3])],
            Tol::printed(4),
        )
        .named(
            "se",
            &names,
            &[number(x[2]), number(w[2]), number(x[4])],
            Tol::printed(4),
        )
        .finish();
    assert_eq!((w[3], w[4]), ("-", "-"), "{t}");
    assert!(t.contains("Converged: Yes ("), "{t}");
    let (means, sds) = (floats(&j["means"]), floats(&j["std_devs"]));
    let ses = floats(&j["mean_std_errors"]);
    let sd_ses = floats(&j["std_dev_std_errors"]);
    Check::new(case)
        .named(
            "coef",
            &names,
            &[means[0], means[1], sds[0]],
            Tol::iterative(1e-6),
        )
        .named(
            "se",
            &names,
            &[ses[0], ses[1], sd_ses[0]],
            Tol::iterative(1e-5),
        )
        .scalar(
            "loglik",
            j["log_likelihood"].as_f64().unwrap(),
            Tol::stat(n),
        )
        .finish();
    assert!(j["std_dev_std_errors"][1].is_null(), "{j}");
    assert_eq!(j["converged"], json!(true), "{j}");
}

/// mlogit's estimate with Hessian SEs (the SEs were 0.1000).
#[test]
fn test_validate_cli_mixed_logit() {
    let (case, s, name) = loaded("disc_mixed_logit");
    let args = [
        "discrete",
        "mixed-logit",
        &name,
        "--choice-id",
        "id",
        "--alt-id",
        "alt",
        "-y",
        "choice",
        "-x",
        "x",
        "w",
        "--random-vars",
        "x",
        "--n-draws",
        "50",
    ];
    let t = s.ok(&args);
    let j = s.json(&args);
    check_mixed_logit(&case, &t, &j, 300);
    assert!(
        t.contains(
            "Draws: Halton, prime 2, first 100 points dropped, 50 draws per choice situation"
        ),
        "{t}"
    );
}

/// `--individual` takes the draws per person (mlogit panel = TRUE); it was an unknown
/// argument.
#[test]
fn test_validate_cli_mixed_logit_panel() {
    let (case, s, name) = loaded("disc_mixed_logit_panel");
    let args = [
        "discrete",
        "mixed-logit",
        &name,
        "--choice-id",
        "id",
        "--alt-id",
        "alt",
        "-y",
        "choice",
        "-x",
        "x",
        "w",
        "--random-vars",
        "x",
        "--n-draws",
        "50",
        "--individual",
        "person",
    ];
    let t = s.ok(&args);
    let j = s.json(&args);
    check_mixed_logit(&case, &t, &j, 400);
    assert!(t.contains("Individuals: 100"), "{t}");
    assert_eq!(j["n_individuals"], json!(100), "{j}");
    assert_eq!(j["n_units"], json!(100), "{j}");
}

/// negbin prints theta with its SE and the convergence line.
#[test]
fn test_validate_cli_negbin_theta_se() {
    let (case, s, name) = loaded("disc_negbin");
    let args = ["discrete", "negbin", &name, "-y", "y", "-x", "x1", "x2"];
    let t = s.ok(&args);
    let theta = numbers(&t, "Theta (dispersion):");
    Check::new(&case)
        .scalar("theta", theta[0], Tol::printed(6))
        .scalar("se_theta", theta[1], Tol::printed(6))
        .finish();
    assert!(t.contains("Converged: Yes ("), "{t}");
    let j = s.json(&args);
    Check::new(&case)
        .scalar(
            "se_theta",
            j["theta_std_error"].as_f64().unwrap(),
            Tol::iterative(1e-5),
        )
        .finish();
    assert_eq!(j["converged"], json!(true), "{j}");
}

/// zip and zinb print the convergence line; zinb prints theta with both SEs.
#[test]
fn test_validate_cli_zip_zinb_convergence_and_theta_se() {
    let (case, s, name) = loaded("disc_zeroinfl");
    let zip = [
        "discrete", "zip", &name, "-y", "y", "-x", "x1", "x2", "-z", "x1", "x2",
    ];
    let zinb = [
        "discrete", "zinb", &name, "-y", "y_nb", "-x", "x1", "x2", "-z", "x1", "x2",
    ];
    let t = s.ok(&zip);
    assert!(t.contains("Converged: Yes ("), "{t}");
    let t = s.ok(&zinb);
    assert!(t.contains("Converged: Yes ("), "{t}");
    let theta = numbers(&t, "Theta (dispersion):");
    Check::new(&case)
        .scalar("zinb_theta", theta[0], Tol::printed(6))
        .scalar("zinb_se_theta", theta[1], Tol::printed(6))
        .scalar("zinb_se_log_theta", theta[2], Tol::printed(6))
        .finish();
    let j = s.json(&zip);
    assert!(j["theta_std_error"].is_null(), "{j}");
    assert_eq!(j["converged"], json!(true), "{j}");
    let j = s.json(&zinb);
    Check::new(&case)
        .scalar(
            "zinb_se_theta",
            j["theta_std_error"].as_f64().unwrap(),
            Tol::iterative(1e-5),
        )
        .finish();
    assert_eq!(j["converged"], json!(true), "{j}");
}

/// logit drops rows with a missing outcome and says so.
#[test]
fn test_validate_cli_logit_missing_note() {
    let (case, s, name) = loaded("disc_choice_na");
    let args = [
        "discrete",
        "logit",
        &name,
        "-y",
        "choice_na",
        "-x",
        "x",
        "w",
    ];
    let t = s.ok(&args);
    let rows = coefficient_rows(&t);
    let names: Vec<&str> = rows.iter().map(|(n, _)| n.as_str()).collect();
    let coef: Vec<f64> = rows.iter().map(|(_, v)| v[0]).collect();
    let se: Vec<f64> = rows.iter().map(|(_, v)| v[1]).collect();
    Check::new(&case)
        .named("logit_y_coef", &names, &coef, Tol::printed(6))
        .named("logit_y_se", &names, &se, Tol::printed(6))
        .scalar(
            "logit_y_nobs",
            labelled(&t, "Observations:").unwrap(),
            Tol::exact(),
        )
        .finish();
    assert!(
        t.contains("Note: 2 of 300 rows dropped because of missing values (choice_na: 2)"),
        "{t}"
    );
    let j = s.json(&args);
    assert_eq!(j["missing"]["n_dropped"], json!(2), "{j}");
}
