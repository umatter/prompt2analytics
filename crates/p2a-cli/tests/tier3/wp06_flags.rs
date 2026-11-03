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

/// Two-way CR1 (sandwich::vcovCL(~firm + year, HC1)): SEs from T1's ols_grunfeld, t and p from ols_twoway_cluster_grunfeld.
#[test]
fn test_validate_cli_clustered_two_way() {
    let r = Case::load("ols_grunfeld");
    let case = Case::load("ols_twoway_cluster_grunfeld");
    let n = r.scalar("n_obs") as usize;
    let s = temp_session();
    let g = s.load(&r);
    let one = [
        "reg",
        "clustered",
        &g,
        "-y",
        "inv",
        "-x",
        "value",
        "capital",
        "--cluster",
        "firm",
    ];
    let two = argv(&one, &["--cluster2", "year"]);
    let out = s.ok(&two);
    let t = text_table(&out);
    Check::new(&r)
        .named("coef", &t.names, &t.coef, Tol::printed(6))
        .named("se_cluster_firm_year_hc1", &t.names, &t.se, Tol::printed(6))
        .finish();
    Check::new(&case)
        .named(
            "t_cluster_firm_year_hc1",
            &t.names,
            &t.stat,
            Tol::printed(3),
        )
        .named("p_cluster_firm_year_hc1", &t.names, &t.p, Tol::printed(4))
        .finish();
    assert!(
        out.contains(
            "\nStandard errors: clustered by firm and year (two-way CR1, G = 10 and 20)\n"
        ),
        "{out}"
    );
    let v = s.json(&two);
    let j = json_table(&v, "t_value");
    Check::new(&r)
        .named("coef", &j.names, &j.coef, Tol::coef(n))
        .named("se_cluster_firm_year_hc1", &j.names, &j.se, Tol::se(n))
        .finish();
    Check::new(&case)
        .named("t_cluster_firm_year_hc1", &j.names, &j.stat, Tol::stat(n))
        .named("p_cluster_firm_year_hc1", &j.names, &j.p, Tol::p_value(n))
        .scalar("n_clusters_firm", num(&v["n_clusters"]), Tol::exact())
        .scalar("n_clusters_year", num(&v["n_clusters_2"]), Tol::exact())
        .finish();
    assert_eq!(v["se_type"], "cluster:firm+year");
    assert_eq!(v["method"], "Clustered OLS (clusters: firm, year)");
    // without --cluster2: T1's one-way result, unchanged
    let o = json_table(&s.json(&one), "t_value");
    Check::new(&r)
        .named("se_cluster_firm_hc1", &o.names, &o.se, Tol::se(n))
        .finish();
}
