//! T2-WP11: causal rd and fuzzy-rd through the p2a CLI (--h, --b, --vce), pinned to rdrobust.

use crate::common::*;
use p2a_validation::text::labelled;
use serde_json::Value;

/// A number of the JSON output.
fn num(v: &Value, key: &str) -> f64 {
    v[key]
        .as_f64()
        .unwrap_or_else(|| panic!("no number `{key}` in {v}"))
}

/// An array of numbers of the JSON output.
fn nums(v: &Value, key: &str) -> Vec<f64> {
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("no array `{key}` in {v}"))
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

// T3-WP61: --vce, --kernel and --bwselect are parsed by clap (a usage error, exit 2).
/// Runs; asserts exit code 2 and no panic; returns stderr.
fn usage_err(s: &TempSession, args: &[&str]) -> String {
    let out = s.run(args);
    assert_no_panic(&out, &args.join(" "));
    assert_eq!(
        out.status.code(),
        Some(2),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// The number after `label` in `text` (panics with the text when absent).
fn label(text: &str, l: &str) -> f64 {
    labelled(text, l).unwrap_or_else(|| panic!("no `{l}` line in\n{text}"))
}

#[test]
fn test_validate_cli_causal_rd_vce() {
    let case = Case::load("rd_sharp_fixed_bw");
    let s = temp_session();
    let name = s.load(&case);
    let n = 1000;
    for (vce, prefix, shown) in [
        (None, "tri_nn_hb", "Nearest-Neighbor"),
        (Some("hc1"), "tri_hc1_hb", "HC1"),
    ] {
        let mut args = vec![
            "causal", "rd", &name, "-y", "y", "-r", "x", "-c", "0", "--h", "0.3", "--b", "0.3",
        ];
        if let Some(v) = vce {
            args.extend(["--vce", v]);
        }
        let k = |s: &str| format!("{prefix}_{s}");
        let j = s.json(&args);
        assert_eq!(j["vce"], shown, "{j}");
        let text = s.ok(&args);
        Check::new(&case)
            .scalar(&k("coef_conv"), num(&j, "tau_conventional"), Tol::coef(n))
            .scalar(&k("coef_bc"), num(&j, "tau_robust"), Tol::coef(n))
            .scalar(&k("coef_bc"), num(&j, "tau_bc"), Tol::coef(n))
            .scalar(&k("se_conv"), num(&j, "se_conventional"), Tol::se(n))
            .scalar(&k("se_robust"), num(&j, "se_robust"), Tol::se(n))
            .scalar(&k("p_robust"), num(&j, "p_robust"), Tol::p_value(n))
            .vector(&k("ci_robust"), &nums(&j, "ci_robust"), Tol::stat(n))
            .vector(
                &k("n_h"),
                &[num(&j, "n_eff_left"), num(&j, "n_eff_right")],
                Tol::exact(),
            )
            .vector(
                &k("n_b"),
                &[num(&j, "n_b_left"), num(&j, "n_b_right")],
                Tol::exact(),
            )
            .scalar(
                &k("coef_bc"),
                label(&text, "Robust RD Estimate:"),
                Tol::printed(6),
            )
            .scalar(&k("se_robust"), label(&text, "Std Error:"), Tol::printed(6))
            .scalar(
                &k("coef_conv"),
                label(&text, "Conventional Estimate:"),
                Tol::printed(6),
            )
            .scalar(
                &k("se_conv"),
                label(&text, "Conventional Std Error:"),
                Tol::printed(6),
            )
            .scalar(&k("p_robust"), label(&text, "p-value:"), Tol::printed(4))
            .finish();
    }
}

#[test]
fn test_cli_causal_rd_rejects_unknown_vce() {
    // --vce is parsed before the dataset lookup: no dataset is loaded
    let s = temp_session();
    let err = usage_err(
        &s,
        &[
            "causal", "rd", "none", "-y", "y", "-r", "x", "-c", "0", "--vce", "robust",
        ],
    );
    // the core message alone, without a second prefix
    let once = |err: &str, value: &str| {
        assert!(
            err.contains(&format!("Unknown vce '{value}'; use")) && err.contains("hc1"),
            "{err}"
        );
        assert!(
            !err.contains("Invalid --vce") && !err.contains("Invalid model specification"),
            "{err}"
        );
    };
    once(&err, "robust");
    let err = usage_err(
        &s,
        &[
            "causal", "fuzzy-rd", "none", "-y", "y", "-r", "x", "-t", "d", "-c", "0", "--vce",
            "hc4",
        ],
    );
    once(&err, "hc4");
}

#[test]
fn test_cli_causal_rd_rejects_unknown_kernel_and_bwselect() {
    // --kernel and --bwselect are parsed before the dataset lookup: no dataset is loaded
    let s = temp_session();
    let kernel = |err: &str| {
        assert!(
            err.contains("Unknown kernel 'gaussian'; use \"triangular\"")
                && err.contains("epanechnikov")
                && err.contains("uniform"),
            "{err}"
        );
    };
    kernel(&usage_err(
        &s,
        &[
            "causal", "rd", "none", "-y", "y", "-r", "x", "-c", "0", "--kernel", "gaussian",
        ],
    ));
    kernel(&usage_err(
        &s,
        &[
            "causal", "fuzzy-rd", "none", "-y", "y", "-r", "x", "-t", "d", "-c", "0", "--kernel",
            "gaussian",
        ],
    ));
    let err = usage_err(
        &s,
        &[
            "causal",
            "rd",
            "none",
            "-y",
            "y",
            "-r",
            "x",
            "-c",
            "0",
            "--bwselect",
            "ik",
        ],
    );
    assert!(
        err.contains("Unknown bwselect 'ik'; use \"mserd\"") && err.contains("certwo"),
        "{err}"
    );
}

#[test]
fn test_cli_causal_rd_short_kernel_names_and_certwo() {
    // the short names select the named kernel, and certwo the separate CER bandwidths,
    // never the triangular / mserd fallback
    let s = temp_session();
    s.load_as(
        &repo_path("validation/datasets/generated/rd_sim.csv"),
        "rd",
        &[],
    );
    let sharp = |extra: &[&str]| {
        let mut args = vec![
            "causal", "rd", "rd", "-y", "y", "-r", "x", "-c", "0", "--h", "0.3",
        ];
        args.extend_from_slice(extra);
        s.json(&args)
    };
    let epa = sharp(&["--kernel", "epa"]);
    assert_eq!(epa["kernel"], "Epanechnikov", "{epa}");
    let full = sharp(&["--kernel", "epanechnikov"]);
    assert_eq!(num(&epa, "tau_robust"), num(&full, "tau_robust"));
    assert_eq!(sharp(&["--kernel", "uni"])["kernel"], "Uniform");
    assert_eq!(sharp(&["--kernel", "TRI"])["kernel"], "Triangular");

    let fuzzy = |kernel: &str| {
        num(
            &s.json(&[
                "causal", "fuzzy-rd", "rd", "-y", "yf", "-r", "x", "-t", "d", "-c", "0", "--h",
                "0.3", "--kernel", kernel,
            ]),
            "tau_fuzzy",
        )
    };
    let uni = fuzzy("uni");
    assert_eq!(uni, fuzzy("uniform"));
    assert_ne!(uni, fuzzy("triangular"));

    let selected = |bwselect: &str| {
        s.json(&[
            "causal",
            "rd",
            "rd",
            "-y",
            "y",
            "-r",
            "x",
            "-c",
            "0",
            "--bwselect",
            bwselect,
        ])
    };
    let common = selected("cerrd");
    assert_eq!(num(&common, "h_left"), num(&common, "h_right"), "{common}");
    let two = selected("certwo");
    assert_ne!(num(&two, "h_left"), num(&two, "h_right"), "{two}");
}

#[test]
fn test_validate_cli_causal_fuzzy_rd() {
    let case = Case::load("rd_fuzzy_fixed_bw");
    let s = temp_session();
    let name = s.load(&case);
    let n = 1000;
    let args = [
        "causal", "fuzzy-rd", &name, "-y", "yf", "-r", "x", "-t", "d", "-c", "0", "--h", "0.3",
        "--b", "0.5",
    ];
    let k = |s: &str| format!("fuzzy_nn_h03_b05_{s}");
    let j = s.json(&args);
    let text = s.ok(&args);
    Check::new(&case)
        .scalar(&k("coef_conv"), num(&j, "tau_fuzzy"), Tol::coef(n))
        .scalar(&k("se_conv"), num(&j, "se_fuzzy"), Tol::se(n))
        .scalar(&k("coef_bc"), num(&j, "tau_fuzzy_bc"), Tol::coef(n))
        .scalar(&k("se_robust"), num(&j, "se_fuzzy_robust"), Tol::se(n))
        .scalar(&k("p_robust"), num(&j, "p_fuzzy_robust"), Tol::p_value(n))
        .vector(&k("ci_robust"), &nums(&j, "ci_fuzzy_robust"), Tol::stat(n))
        .scalar(
            &k("first_stage_conv"),
            num(&j, "first_stage_tau"),
            Tol::coef(n),
        )
        .scalar(
            &k("first_stage_se_conv"),
            num(&j, "first_stage_se"),
            Tol::se(n),
        )
        .scalar(
            &k("coef_bc"),
            label(&text, "Robust LATE (bias-corrected):"),
            Tol::printed(6),
        )
        .finish();
    let f_case =
        (case.scalar(&k("first_stage_conv")) / case.scalar(&k("first_stage_se_conv"))).powi(2);
    assert!(
        Tol::stat(n).allows(num(&j, "first_stage_f"), f_case),
        "first-stage F {} vs {f_case}",
        num(&j, "first_stage_f")
    );
}

#[test]
fn test_cli_causal_rd_missing_value_note() {
    let s = temp_session();
    let src = repo_path("validation/datasets/generated/rd_sim.csv");
    let csv = std::fs::read_to_string(&src).expect("rd_sim.csv");
    // blank the y field (second column) of data row 11 (line 12)
    let lines: Vec<String> = csv
        .lines()
        .enumerate()
        .map(|(i, line)| {
            if i == 11 {
                let mut cells: Vec<&str> = line.split(',').collect();
                cells[1] = "";
                cells.join(",")
            } else {
                line.to_string()
            }
        })
        .collect();
    let path = s.dir().join("rd_na.csv");
    std::fs::write(&path, lines.join("\n") + "\n").expect("write rd_na.csv");
    s.load_as(&path, "rd_na", &[]);
    let args = [
        "causal", "rd", "rd_na", "-y", "y", "-r", "x", "-c", "0", "--h", "0.3",
    ];
    let text = s.ok(&args);
    assert!(
        text.contains("Note: 1 of 1000 rows dropped because of missing values (y: 1)"),
        "{text}"
    );
    let j = s.json(&args);
    assert_eq!(j["missing"]["n_dropped"], 1, "{j}");
}
