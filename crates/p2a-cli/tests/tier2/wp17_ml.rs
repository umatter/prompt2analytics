//! T2-WP17: ML commands through the p2a CLI (integer columns, missing values).
use crate::common::*;
use serde_json::json;

/// The 30-row ML fixture, built with the same IEEE operations as the R cross-check (so R
/// sees the same values bit for bit): y_na and cls_na are missing at row 4, x1_na at row 7.
/// Rows listed in `skip` are left out (complete-row comparisons). An empty field is
/// missing (T2-WP01 loader).
fn fixture_csv(skip: &[usize]) -> String {
    let mut out =
        String::from("id,x1,x2,xi,xi_f,y,yi,yi_f,cls,cls_f,cls_b,lab,y_na,x1_na,cls_na\n");
    for i in (0..30usize).filter(|i| !skip.contains(i)) {
        let x1 = ((i * 7) % 11) as f64 + 0.5 * ((i * 3) % 5) as f64;
        let x2 = ((i * 5) % 13) as f64 / 2.0;
        let xi = (i * 5) % 13;
        let noise = ((i * 7919) % 101) as f64 / 100.0 - 0.5;
        let y = 2.0 * x1 - x2 + noise;
        let yi = y.floor() as i64;
        let cls = u8::from(x1 - 0.5 * x2 + noise > 3.0);
        let lab = if cls == 1 { "yes" } else { "no" };
        let blank_if = |row: usize, v: String| if i == row { String::new() } else { v };
        out.push_str(&format!(
            "{i},{x1},{x2},{xi},{xi}.0,{y},{yi},{yi}.0,{cls},{cls}.0,{},{lab},{},{},{}\n",
            cls == 1,
            blank_if(4, y.to_string()),
            blank_if(7, x1.to_string()),
            blank_if(4, cls.to_string()),
        ));
    }
    out
}

/// Datasets "ml" (the fixture) and "ml_clean" (rows 4 and 7 removed) in a scratch session.
fn session_with_fixture() -> TempSession {
    let s = temp_session();
    for (name, skip) in [("ml", &[][..]), ("ml_clean", &[4, 7][..])] {
        let path = s.dir().join(format!("{name}.csv"));
        std::fs::write(&path, fixture_csv(skip)).unwrap();
        s.load_as(&path, name, &[]);
    }
    s
}

#[test]
fn test_cli_ml_integer_columns_accepted() {
    // CLI-8. Before: "Column 'xi' must be numeric: ... expected `Float64`, got `i64`" and
    // "Target column must be numeric".
    let s = session_with_fixture();
    let km = |c: &str| {
        s.json(&[
            "ml", "kmeans", "ml", "--cols", "x1", c, "-k", "2", "--seed", "1",
        ])
    };
    let (a, b) = (km("xi"), km("xi_f"));
    for key in [
        "labels",
        "inertia",
        "centroids",
        "cluster_sizes",
        "n_iterations",
    ] {
        assert_eq!(a[key], b[key], "{key}");
    }
    let rf = |y: &str| {
        s.json(&[
            "ml",
            "random-forest",
            "ml",
            "--cols",
            "x1",
            "x2",
            "-y",
            y,
            "--n-trees",
            "10",
            "--seed",
            "1",
        ])
    };
    let (a, b) = (rf("yi"), rf("yi_f"));
    for key in ["predictions", "feature_importances", "oob_score"] {
        assert_eq!(a[key], b[key], "{key}");
    }
}

#[test]
fn test_na_cli_ml_missing_values_refused() {
    // CLI-9, ML-41. Before: the null target was read as 0.0 and the forest ran (exit 0,
    // "OOB Score: ..."); the feature null gave "Missing value at row 7 in column 'x1_na'";
    // the text column "Column 'lab' must be numeric: invalid series dtype ...".
    let s = session_with_fixture();
    let e = s.err(&[
        "ml",
        "random-forest",
        "ml",
        "--cols",
        "x1",
        "x2",
        "-y",
        "y_na",
        "--n-trees",
        "10",
        "--seed",
        "1",
    ]);
    assert!(
        e.contains("Column 'y_na' contains 1 missing value(s) (null or NaN)"),
        "{e}"
    );
    let e = s.err(&[
        "ml",
        "hierarchical",
        "ml",
        "--cols",
        "x1_na",
        "x2",
        "-n",
        "2",
    ]);
    assert!(
        e.contains("Column 'x1_na' contains 1 missing value(s) (null or NaN)"),
        "{e}"
    );
    let e = s.err(&["ml", "pca", "ml", "--cols", "x1", "lab"]);
    assert!(
        e.contains("Column 'lab' contains non-numeric values"),
        "{e}"
    );
}

#[test]
fn test_na_cli_ml_svm_omits_incomplete_rows() {
    // CLI-9, ML-33. e1071::svm(factor(cls_na) ~ x1_na + x2) uses 28 rows (na.omit).
    // Before: "Missing value at row 7 in column 'x1_na'" (and a null target read as class
    // 0).
    let s = session_with_fixture();
    let args = |ds: &'static str| ["ml", "svm", ds, "--cols", "x1_na", "x2", "-y", "cls_na"];
    let (a, b) = (s.json(&args("ml")), s.json(&args("ml_clean")));
    assert_eq!(
        a["missing"],
        json!({"n_total": 30, "n_used": 28, "n_dropped": 2,
               "by_column": {"cls_na": 1, "x1_na": 1}})
    );
    assert!(b.get("missing").is_none());
    for key in [
        "weights",
        "bias",
        "n_iterations",
        "n_support_vectors",
        "predictions",
        "class_labels",
        "converged",
    ] {
        assert_eq!(a[key], b[key], "{key}");
    }
    assert_eq!(a["predictions"].as_array().unwrap().len(), 28);
    let out = s.ok(&args("ml"));
    assert!(
        out.contains("Note: 2 of 30 rows dropped because of missing values (cls_na: 1, x1_na: 1)"),
        "{out}"
    );
}
