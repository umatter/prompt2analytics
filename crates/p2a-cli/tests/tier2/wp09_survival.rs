//! T2-WP09: Kaplan-Meier and the log-rank test, --conf-type through the p2a CLI.
use crate::common::*;
use serde_json::{Value, json};

/// A JSON array as f64 (`null`, R's NA, as NaN).
fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap_or_else(|| panic!("not an array: {v}"))
        .iter()
        .map(num)
        .collect()
}

/// A JSON number as f64 (`null` as NaN).
fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

#[test]
fn test_validate_cli_survival_km_lung() {
    // validation/r/cases/survival.R, case surv_km_lung: survfit(Surv(time, status) ~ 1,
    // lung) with R's default conf.type "log" over the whole curve; the log-log summary at
    // t = 5 and quantile(fit, 0.5).
    let case = Case::load("surv_km_lung");
    let s = temp_session();
    let name = s.load(&case);
    let n = case.scalar("n") as usize;

    let j = s.json(&["survival", "km", &name, "-t", "time", "-e", "status"]);
    Check::new(&case)
        .vector("curve_time", &nums(&j["times"]), Tol::exact())
        .vector("curve_surv", &nums(&j["survival"]), Tol::coef(n))
        .vector("curve_lower", &nums(&j["ci_lower"]), Tol::stat(n))
        .vector("curve_upper", &nums(&j["ci_upper"]), Tol::stat(n))
        .scalar("log_median", num(&j["median_survival"]), Tol::exact())
        .scalar("log_median_lower", num(&j["median_ci"][0]), Tol::exact())
        .scalar("log_median_upper", num(&j["median_ci"][1]), Tol::exact())
        .scalar("n", num(&j["n_obs"]), Tol::exact())
        .scalar("n_event", num(&j["total_events"]), Tol::exact())
        .finish();
    assert_eq!(j["conf_type"], "log");
    assert_eq!(j["conf_level"], 0.95);
    assert!(j.get("missing").is_none(), "{j}");

    let out = s.ok(&[
        "survival",
        "km",
        &name,
        "-t",
        "time",
        "-e",
        "status",
        "--conf-type",
        "log-log",
    ]);
    let at = |k: &str| case.vector(k)[0]; // t = 5, the first time
    for expected in [
        "Confidence intervals: 95%, conf.type = log-log".to_string(),
        format!(
            "Median survival time: {:.4} (95% CI: {:.4} - {:.4})",
            case.scalar("loglog_median"),
            case.scalar("loglog_median_lower"),
            case.scalar("loglog_median_upper")
        ),
        format!(
            "{:<12.4} {:<12.4} {:<12.4} {:<12.4} {:<12.4}",
            5.0,
            at("loglog_surv"),
            at("loglog_std_err"),
            at("loglog_lower"),
            at("loglog_upper")
        )
        .trim_end()
        .to_string(),
    ] {
        assert!(out.contains(&expected), "{expected:?} not in\n{out}");
    }
}

#[test]
fn test_validate_cli_survival_km_groups_missing_note() {
    // Case surv_logrank_lung: one ph.ecog label is NA, so 227 of 228 rows are used and the
    // groups are R's factor levels 0, 1, 2, 3 with survdiff's n per group.
    let case = Case::load("surv_logrank_lung");
    let s = temp_session();
    let name = s.load(&case);
    let args = [
        "survival", "km", &name, "-t", "time", "-e", "status", "-g", "ph.ecog",
    ];
    let out = s.ok(&args);
    let groups: Vec<&str> = out.lines().filter(|l| l.starts_with("Group: ")).collect();
    assert_eq!(
        groups,
        ["Group: 0", "Group: 1", "Group: 2", "Group: 3"],
        "{out}"
    );
    let note = "Note: 1 of 228 rows dropped because of missing values (ph.ecog: 1)";
    assert_eq!(out.matches("Note:").count(), 1, "{out}");
    let last_group = out.find("Group: 3").unwrap();
    assert!(
        out.find(note).is_some_and(|at| at > last_group),
        "{note:?} not after the last group\n{out}"
    );

    let docs = s.json_docs(&args);
    assert_eq!(docs.len(), 4);
    for d in &docs {
        assert_eq!(d["missing"]["n_dropped"], 1, "{d}");
    }
    let n_obs: Vec<f64> = docs.iter().map(|d| num(&d["n_obs"])).collect();
    Check::new(&case)
        .vector("ecog_n", &n_obs, Tol::exact())
        .finish();
}

#[test]
fn test_validate_cli_survival_log_rank_lung() {
    // Case surv_logrank_lung: survdiff(Surv(time, status) ~ sex) and ~ ph.ecog.
    let case = Case::load("surv_logrank_lung");
    let s = temp_session();
    let name = s.load(&case);
    let n = case.vector("sex_n").iter().sum::<f64>() as usize; // 228 rows
    let sex_args = [
        "survival", "log-rank", &name, "-t", "time", "-e", "status", "-g", "sex",
    ];
    let ecog_args = [
        "survival", "log-rank", &name, "-t", "time", "-e", "status", "-g", "ph.ecog",
    ];

    let sex = s.json(&sex_args);
    let ecog = s.json(&ecog_args);
    Check::new(&case)
        .scalar("sex_chisq", num(&sex["chi_squared"]), Tol::stat(n))
        .scalar("sex_p", num(&sex["p_value"]), Tol::p_value(n))
        .scalar("sex_df", num(&sex["degrees_of_freedom"]), Tol::exact())
        .vector("sex_exp", &nums(&sex["expected_per_group"]), Tol::stat(n))
        .vector("sex_obs", &nums(&sex["events_per_group"]), Tol::exact())
        .vector("sex_n", &nums(&sex["n_per_group"]), Tol::exact())
        .vector("sex_var", &nums(&sex["variance"]), Tol::stat(n))
        .scalar("ecog_chisq", num(&ecog["chi_squared"]), Tol::stat(n))
        .scalar("ecog_df", num(&ecog["degrees_of_freedom"]), Tol::exact())
        .vector("ecog_exp", &nums(&ecog["expected_per_group"]), Tol::stat(n))
        .finish();
    assert_eq!(
        ecog["missing"],
        json!({"n_total": 228, "n_used": 227, "n_dropped": 1, "by_column": {"ph.ecog": 1}})
    );
    assert!(sex.get("missing").is_none(), "{sex}");

    let out = s.ok(&sex_args);
    for expected in [
        format!("Chi-squared statistic: {:.4}", case.scalar("sex_chisq")),
        format!("p-value: {:.6}", case.scalar("sex_p")),
    ] {
        assert!(out.contains(&expected), "{expected:?} not in\n{out}");
    }
}

#[test]
fn test_validate_cli_survival_km_rejects_unknown_conf_type() {
    let case = Case::load("surv_km_lung");
    let s = temp_session();
    let name = s.load(&case);
    let err = s.err(&[
        "survival",
        "km",
        &name,
        "-t",
        "time",
        "-e",
        "status",
        "--conf-type",
        "logit",
    ]);
    assert!(err.contains("logit") && err.contains("log-log"), "{err}");
}

#[test]
fn test_validate_cli_survival_cox_aft_integer_columns() {
    // Case surv_km_lung: Int64 time, status (1/2) and covariates. R coxph(Surv(time,
    // status) ~ age + sex + ph.karno, lung) uses 227 rows (one ph.karno is NA);
    // survreg(Surv(time, status) ~ age + sex) all 228 with 165 events.
    let case = Case::load("surv_km_lung");
    let s = temp_session();
    let name = s.load(&case);
    let cox = s.json(&[
        "survival", "cox", &name, "-t", "time", "-e", "status", "-x", "age", "sex", "ph.karno",
    ]);
    assert_eq!(cox["n_obs"], 227, "{cox}");
    assert_eq!(cox["missing"]["by_column"]["ph.karno"], 1, "{cox}");
    let aft = s.json(&[
        "survival", "aft", &name, "-t", "time", "-e", "status", "-x", "age", "sex",
    ]);
    Check::new(&case)
        .scalar("n", num(&aft["n_obs"]), Tol::exact())
        .scalar("n_event", num(&aft["n_events"]), Tol::exact())
        .finish();
}
