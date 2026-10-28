//! Self-tests of the harness: every committed case parses, the self-test case
//! matches values recomputed here from the CSV (R -> JSON -> Rust), and the
//! tolerances, the collecting comparator and the CLI text parsers behave as
//! validation/README.md documents.

use std::panic::{AssertUnwindSafe, catch_unwind};

use p2a_validation::text::{coefficient_rows, labelled};
use p2a_validation::{Case, CaseError, Check, Tol, repo_path, validation_dir};

fn selftest() -> Case {
    Case::load("harness_selftest")
}

/// Grunfeld rows `[firm, year, inv, value, capital]`, read by hand.
fn grunfeld() -> Vec<[f64; 5]> {
    let text = std::fs::read_to_string(repo_path("validation/datasets/grunfeld.csv"))
        .expect("read validation/datasets/grunfeld.csv");
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some("firm,year,inv,value,capital"));
    lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let cells: Vec<f64> = l
                .split(',')
                .map(|c| c.trim().parse::<f64>().expect("numeric cell"))
                .collect();
            assert_eq!(cells.len(), 5, "row `{l}`");
            [cells[0], cells[1], cells[2], cells[3], cells[4]]
        })
        .collect()
}

fn panic_text(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else {
        panic!("panic payload is neither String nor &str")
    }
}

#[test]
fn committed_cases_parse_with_schema_1_and_existing_datasets() {
    let dir = validation_dir().join("expected");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("read validation/expected")
        .map(|e| e.expect("directory entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    files.sort();
    let mut checked = 0;
    for path in files {
        let stem = path.file_stem().unwrap().to_str().unwrap().to_string();
        let case = Case::try_load(&stem).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(case.schema, 1, "{stem}");
        assert_eq!(case.id, stem);
        assert!(
            case.datasets.contains(&case.dataset),
            "{stem}: dataset {} is not in datasets {:?}",
            case.dataset,
            case.datasets
        );
        for ds in &case.datasets {
            assert!(
                repo_path(ds).exists(),
                "{stem}: dataset {ds} does not exist"
            );
        }
        assert!(case.keys().next().is_some(), "{stem}: no values");
        assert!(
            case.r_version.starts_with("R version"),
            "{stem}: r_version {}",
            case.r_version
        );
        checked += 1;
    }
    assert!(checked >= 1, "no case file in {}", dir.display());
}

#[test]
fn harness_selftest_matches_values_recomputed_from_csv() {
    let case = selftest();
    let rows = grunfeld();
    let n = rows.len();
    let inv: Vec<f64> = rows.iter().map(|r| r[2]).collect();
    let sum_inv = inv.iter().fold(0.0, |s, x| s + x);
    let mean = sum_inv / n as f64;
    let var = inv.iter().fold(0.0, |s, x| s + (x - mean) * (x - mean)) / (n - 1) as f64;
    let mean_value = rows.iter().fold(0.0, |s, r| s + r[3]) / n as f64;
    let mean_capital = rows.iter().fold(0.0, |s, r| s + r[4]) / n as f64;
    let max_inv = inv.iter().fold(f64::NEG_INFINITY, |m, &x| m.max(x));

    // R accumulates means and sums in long double; a naive f64 sum differs by a few ulps
    // (measured: 5.8e-16 relative for the mean, 3.1e-16 for the variance, 6.2e-16 for the sum).
    let ulps = Tol::new(1e-14, 0.0);
    Check::new(&case)
        .scalar("n", n as f64, Tol::exact())
        .scalar("inv_mean", mean, ulps) // R 145.95824999999999
        .scalar("inv_var", var, ulps) // R 47034.894115012568
        .named(
            "means",
            &["capital", "value"],
            &[mean_capital, mean_value],
            ulps,
        ) // R 276.01715000000002, 1081.6811
        .named("inv_max", &["inv"], &[max_inv], Tol::exact()) // R 1486.7
        .vector("inv_first3", &inv[..3], Tol::exact()) // R 317.6, 391.8, 410.6 (same doubles as the CSV text)
        .vector("with_na", &[sum_inv, f64::NAN], ulps) // R 29191.650000000001, null
        .finish();
    assert!(case.vector("with_na")[1].is_nan());
}

#[test]
fn tol_rows_match_readme_table() {
    assert_eq!(Tol::coef(99).rel, 1e-6);
    assert_eq!(Tol::coef(100).rel, 1e-8);
    assert_eq!(Tol::coef(1000).rel, 1e-8);
    assert_eq!(Tol::coef(1001).rel, 1e-10);
    for n in [1, 99, 100, 1000, 1001, 1_000_000] {
        assert_eq!(Tol::coef(n).abs, 1e-12);
        assert_eq!(Tol::se(n).abs, 1e-14);
    }
    assert_eq!(Tol::se(99).rel, 1e-5);
    assert_eq!(Tol::se(100).rel, 1e-6);
    assert_eq!(Tol::se(1000).rel, 1e-6);
    assert_eq!(Tol::se(1001).rel, 1e-8);
    for n in [99, 100, 1000, 1001] {
        assert_eq!(Tol::stat(n), Tol::se(n));
    }
    assert_eq!(Tol::p_value(500), Tol::new(1e-6, 1e-12));
    assert_eq!(Tol::iterative(1e-4), Tol::new(1e-4, 1e-12));
    assert_eq!(Tol::printed(6).rel, 0.0);
    assert!((Tol::printed(6).abs - (0.5e-6 + 1e-12)).abs() < 1e-20);
    assert_eq!(Tol::exact(), Tol::new(0.0, 0.0));
}

#[test]
fn tol_allows_abs_plus_rel_nan_and_infinities() {
    let rel = Tol::new(1e-6, 0.0);
    assert!(rel.allows(1.0 + 0.9e-6, 1.0));
    assert!(!rel.allows(1.0 + 1.1e-6, 1.0));
    assert!(Tol::new(0.0, 1e-12).allows(1e-13, 0.0));
    assert!(rel.allows(f64::NAN, f64::NAN));
    assert!(!rel.allows(f64::NAN, 1.0));
    assert!(!rel.allows(1.0, f64::NAN));
    assert!(rel.allows(f64::INFINITY, f64::INFINITY));
    assert!(!rel.allows(f64::INFINITY, f64::NEG_INFINITY));
    assert!(!rel.allows(f64::INFINITY, 1e308));
    assert!(Tol::printed(4).allows(0.3119, 0.3118654460548856));
    assert!(!Tol::printed(4).allows(0.3120, 0.3118654460548856));
}

#[test]
#[should_panic(expected = "case harness_selftest: 2 of 3 comparisons failed")]
fn check_failure_count_in_one_panic() {
    let case = selftest();
    Check::new(&case)
        .scalar("inv_mean", 0.0, Tol::exact())
        .scalar("inv_var", 0.0, Tol::exact())
        .scalar("n", 200.0, Tol::exact())
        .finish();
}

#[test]
fn check_lists_every_failure() {
    let case = selftest();
    let payload = catch_unwind(AssertUnwindSafe(|| {
        Check::new(&case)
            .scalar("inv_mean", 0.0, Tol::exact())
            .scalar("inv_var", 0.0, Tol::exact())
            .scalar("n", 200.0, Tol::exact())
            .finish();
    }))
    .expect_err("the check must fail");
    let text = panic_text(payload);
    assert!(text.contains("inv_mean rust=0.0 R=145.95825"), "{text}");
    assert!(text.contains("inv_var rust=0.0"), "{text}");
    // The passing `n` row is not listed ("inv_mean rust=" also ends in "n rust=").
    assert!(!text.contains("harness_selftest: n rust="), "{text}");
}

#[test]
#[should_panic(expected = "Check for case harness_selftest dropped without finish()")]
fn check_panics_when_dropped_unfinished() {
    let case = selftest();
    let mut c = Check::new(&case);
    c.scalar("n", 200.0, Tol::exact());
}

#[test]
#[should_panic(expected = "Check finished but nothing was compared")]
fn check_panics_when_nothing_compared() {
    let case = selftest();
    Check::new(&case).finish();
}

#[test]
fn check_named_matches_by_name_in_any_order() {
    let case = selftest();
    Check::new(&case)
        .named(
            "means",
            &["capital", "value"],
            &[276.01715, 1081.6811],
            Tol::exact(),
        )
        .finish();
}

#[test]
#[should_panic(expected = "R name `capital` not in the Rust names")]
fn check_named_rejects_missing_name() {
    let case = selftest();
    Check::new(&case)
        .named("means", &["value"], &[1081.6811], Tol::exact())
        .finish();
}

#[test]
fn check_named_missing_name_also_reports_the_length() {
    let case = selftest();
    let payload = catch_unwind(AssertUnwindSafe(|| {
        Check::new(&case)
            .named("means", &["value"], &[1081.6811], Tol::exact())
            .finish();
    }))
    .expect_err("the check must fail");
    let text = panic_text(payload);
    assert!(text.contains("length 1 (Rust) vs 2 (R)"), "{text}");
    assert!(
        text.contains("R name `capital` not in the Rust names"),
        "{text}"
    );
}

#[test]
#[should_panic(expected = "`intercept` is not an R name")]
fn check_named_rejects_extra_name() {
    let case = selftest();
    Check::new(&case)
        .named(
            "means",
            &["value", "capital", "intercept"],
            &[1081.6811, 276.01715, 1.0],
            Tol::exact(),
        )
        .finish();
}

#[test]
#[should_panic(expected = "name `value` occurs 2 times in the Rust names")]
fn check_named_rejects_duplicate_name() {
    let case = selftest();
    Check::new(&case)
        .named(
            "means",
            &["value", "value"],
            &[1081.6811, 1081.6811],
            Tol::exact(),
        )
        .finish();
}

#[test]
fn check_named_subset_compares_only_listed_names() {
    let case = selftest();
    Check::new(&case)
        .named_subset(
            "means",
            &["value"],
            &["value", "other"],
            &[1081.6811, 99.0],
            Tol::exact(),
        )
        .finish();
}

#[test]
fn check_structural_problems_are_failures() {
    let case = selftest();
    let payload = catch_unwind(AssertUnwindSafe(|| {
        Check::new(&case)
            .scalar("nope", 1.0, Tol::exact())
            .scalar("means", 1.0, Tol::exact())
            .vector("inv_first3", &[1.0], Tol::exact())
            .finish();
    }))
    .expect_err("the check must fail");
    let text = panic_text(payload);
    assert!(text.contains("no value `nope` in the case"), "{text}");
    assert!(text.contains("value `means` is not a scalar"), "{text}");
    assert!(text.contains("length 1 (Rust) vs 3 (R)"), "{text}");
    assert!(text.contains("3 of 3 comparisons failed"), "{text}");
}

#[test]
fn case_accessors_return_values_in_r_order() {
    let case = selftest();
    assert_eq!(case.scalar("n"), 200.0);
    assert_eq!(case.vector("means"), vec![1081.6811, 276.01715]);
    assert_eq!(case.names("means"), ["value", "capital"]);
    assert_eq!(case.named("means", "capital"), 276.01715);
    assert_eq!(
        case.by_names("means", &["capital", "value"]),
        vec![276.01715, 1081.6811]
    );
    // A length-1 named vector stays an array.
    assert_eq!(case.vector("inv_max"), vec![1486.7]);
    assert_eq!(case.vector("inv_mean"), vec![145.95825]);
    let ds = case.dataset_path();
    assert!(
        ds.ends_with("validation/datasets/grunfeld.csv"),
        "{}",
        ds.display()
    );
    assert!(ds.exists(), "{}", ds.display());
    let keys: Vec<&str> = case.keys().collect();
    assert_eq!(
        keys,
        [
            "inv_first3",
            "inv_max",
            "inv_mean",
            "inv_var",
            "means",
            "n",
            "with_na"
        ]
    );
}

#[test]
#[should_panic(expected = "value `means` is not a scalar")]
fn case_scalar_on_named_value_panics() {
    let _ = selftest().scalar("means");
}

#[test]
fn missing_case_names_the_write_command() {
    match Case::try_load("no_such_case_xyz") {
        Err(e @ CaseError::Missing(_)) => {
            let CaseError::Missing(p) = &e else {
                unreachable!()
            };
            assert!(
                p.ends_with("validation/expected/no_such_case_xyz.json"),
                "{}",
                p.display()
            );
            assert!(
                e.to_string()
                    .contains("run `Rscript validation/r/run.R --write no_such_case_xyz`"),
                "{e}"
            );
        }
        other => panic!("expected CaseError::Missing, got {other:?}"),
    }
    let payload =
        catch_unwind(|| Case::load("no_such_case_xyz")).expect_err("Case::load must panic");
    let text = panic_text(payload);
    assert!(
        text.contains("run `Rscript validation/r/run.R --write no_such_case_xyz`"),
        "{text}"
    );
}

/// `p2a reg ols grunfeld -y inv -x value capital --robust hc1`, text output (the paper's
/// cli_ols capture, which has the same rows; produced by the text arm of
/// `format_regression_results` in crates/p2a-cli/src/output.rs).
const CAPTURED_REG_OLS: &str = "
OLS Regression Results
============================================================

Variable         Coefficient   Std. Error    t-value    p-value
------------------------------------------------------------
(Intercept)       -42.714369    11.574701     -3.690     0.0003***
value               0.115562     0.006811     16.967     0.0000***
capital             0.230678     0.048866      4.721     0.0000***
------------------------------------------------------------
R-squared: 0.812408
Adj. R-squared: 0.809537
Observations: 200

Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1
";

#[test]
fn coefficient_rows_parse_captured_reg_ols_table() {
    let rows = coefficient_rows(CAPTURED_REG_OLS);
    let names: Vec<&str> = rows.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["(Intercept)", "value", "capital"]);
    assert_eq!(rows[0].1, vec![-42.714369, 11.574701, -3.690, 0.0003]);
}

#[test]
fn coefficient_rows_strip_stars_dot_and_read_nan() {
    let text = "\
Variable         Coefficient   Std. Error    z-value    p-value
------------------------------------------------------------
(Intercept)       -57.834415    28.898935     -2.001     0.0454*
lag value           0.050000     0.026000      1.923     0.0545.
aliased                  NaN          NaN        NaN        NaN
------------------------------------------------------------
";
    let rows = coefficient_rows(text);
    let names: Vec<&str> = rows.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["(Intercept)", "lag value", "aliased"]);
    assert_eq!(rows[0].1[3], 0.0454);
    assert_eq!(rows[1].1[3], 0.0545);
    assert_eq!(rows[1].1[0], 0.05);
    assert!(rows[2].1.iter().all(|v| v.is_nan()), "{:?}", rows[2]);
}

#[test]
#[should_panic(expected = "no coefficient table found")]
fn coefficient_rows_panics_without_table() {
    let _ = coefficient_rows("R-squared: 0.5\n");
}

#[test]
fn labelled_reads_first_matching_line() {
    let t = CAPTURED_REG_OLS;
    assert_eq!(labelled(t, "R-squared:"), Some(0.812408));
    assert_eq!(labelled(t, "Adj. R-squared:"), Some(0.809537));
    assert_eq!(labelled(t, "Observations:"), Some(200.0));
    assert_eq!(labelled(t, "Missing:"), None);
    let other = "Adj. R-squared: 0.7\nR-squared: 0.8\nF-statistic: 309.0142 (p-value: 0.0000)\nsigma_u: 84.2010, sigma_e: 52.7680\n";
    assert_eq!(labelled(other, "R-squared:"), Some(0.8));
    assert_eq!(labelled(other, "F-statistic:"), Some(309.0142));
    assert_eq!(labelled(other, "sigma_u:"), Some(84.201));
}

#[test]
fn printed_cli_numbers_agree_with_r_at_print_precision() {
    // R 4.3.3, sandwich 3.1.3, lmtest 0.9.40:
    //   d <- read.csv("validation/datasets/grunfeld.csv")
    //   m <- lm(inv ~ value + capital, data = d)
    //   print(unclass(coeftest(m, vcov. = vcovHC(m, type = "HC1"))), digits = 17)
    //   summary(m)$r.squared
    // The constants are the shortest round-trip forms of R's 17-digit output (same doubles).
    const R_ROWS: [(&str, [f64; 4]); 3] = [
        (
            "(Intercept)",
            [
                -42.71436943655936,
                11.574701117099652,
                -3.69032159054683,
                0.00028980093362191437,
            ],
        ),
        (
            "value",
            [
                0.11556215636055213,
                0.006810954456871954,
                16.967101614364047,
                2.165754515305109e-40,
            ],
        ),
        (
            "capital",
            [
                0.2306784887319696,
                0.04886553953434222,
                4.720678231125454,
                4.453595026390839e-06,
            ],
        ),
    ];
    const R_R2: f64 = 0.8124080125447284;
    // The captured `Adj. R-squared: 0.809537` is REG-3 (R: 0.8105035253624413); it is
    // deliberately not part of this parser test.
    let tols = [
        Tol::printed(6),
        Tol::printed(6),
        Tol::printed(3),
        Tol::printed(4),
    ];
    let rows = coefficient_rows(CAPTURED_REG_OLS);
    assert_eq!(rows.len(), R_ROWS.len());
    for ((name, printed), (r_name, r)) in rows.iter().zip(R_ROWS.iter()) {
        assert_eq!(name, r_name);
        for j in 0..4 {
            assert!(
                tols[j].allows(printed[j], r[j]),
                "{name} column {j}: printed {} vs R {}",
                printed[j],
                r[j]
            );
        }
    }
    let r2 = labelled(CAPTURED_REG_OLS, "R-squared:").expect("R-squared line");
    assert!(
        Tol::printed(6).allows(r2, R_R2),
        "R-squared {r2} vs R {R_R2}"
    );
}
