//! T4-WP27 MCP-path tests: grouped and time-indexed shifts, grouped fills, the sign of `periods`,
//! one-hot names and row binds through the munging tools. Owned by T4-WP27.
use crate::common::*;
use p2a_core::data::Dataset;
use p2a_core::polars::prelude::*;
use p2a_validation::{Case, Check, Tol};
use serde_json::json;

/// The session's dataset `name`.
async fn dataset(session: &p2a_mcp::session::Session, name: &str) -> Dataset {
    let map = session.datasets.read().await;
    map.get(name).cloned().unwrap_or_else(|| {
        panic!(
            "no dataset '{name}'; have {:?}",
            map.keys().collect::<Vec<_>>()
        )
    })
}

/// A numeric column as f64, missing as NaN (as `Case::vector` reads JSON null).
fn nums(ds: &Dataset, c: &str) -> Vec<f64> {
    ds.df()
        .column(c)
        .unwrap()
        .cast(&DataType::Float64)
        .unwrap()
        .f64()
        .unwrap()
        .into_iter()
        .map(|v| v.unwrap_or(f64::NAN))
        .collect()
}

/// `nums`, with the null count checked against the case's NA count (NaN passes a `Check`).
fn checked(ds: &Dataset, c: &str, case: &Case, key: &str) -> Vec<f64> {
    let nulls = ds.df().column(c).unwrap().null_count();
    let want = case.vector(key).iter().filter(|v| v.is_nan()).count();
    assert_eq!(nulls, want, "{c} vs {key}: missing cells");
    nums(ds, c)
}

/// DATA-29: munge_diff took no groups, so a panel's first difference ran across firms.
/// R: dplyr::mutate(group_by(d, firm), y - lag(y), (x - lag(x)) / lag(x)) and tidyr::fill by firm
/// (validation/r/cases/data_munging_t4.R, case data_munging_panel_shift).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_diff_group_by() {
    let case = Case::load("data_munging_panel_shift");
    let (session, server) = session_with("data_munging_panel_shift"); // dataset "munging_panel"
    let r = call(&server, &session, "munge_diff",
        json!({"dataset": "munging_panel", "column": "y", "group_by": ["firm"], "result_name": "d"})).await;
    assert!(r.success, "{}", text(&r));
    let r = call(
        &server,
        &session,
        "munge_diff",
        json!({"dataset": "munging_panel", "column": "x", "diff_type": "pct_change",
               "group_by": ["firm"], "result_name": "p"}),
    )
    .await;
    assert!(r.success, "{}", text(&r));
    let r = call(
        &server,
        &session,
        "munge_fill_na",
        json!({"dataset": "munging_panel", "columns": ["z"], "strategy": "forward",
               "group_by": ["firm"], "result_name": "down"}),
    )
    .await;
    assert!(r.success, "{}", text(&r));
    let r = call(
        &server,
        &session,
        "munge_fill_na",
        json!({"dataset": "munging_panel", "columns": ["z"], "strategy": "backward",
               "group_by": ["firm"], "result_name": "up"}),
    )
    .await;
    assert!(r.success, "{}", text(&r));
    let (d, p) = (dataset(&session, "d").await, dataset(&session, "p").await);
    let (down, up) = (
        dataset(&session, "down").await,
        dataset(&session, "up").await,
    );
    let float = Tol::new(1e-14, 0.0);
    Check::new(&case)
        .vector(
            "grouped_diff1_y",
            &checked(&d, "y_diff1", &case, "grouped_diff1_y"),
            float,
        )
        .vector(
            "grouped_pct1_x",
            &checked(&p, "x_pct_change1", &case, "grouped_pct1_x"),
            float,
        )
        .vector(
            "fill_down_z_by_firm",
            &checked(&down, "z", &case, "fill_down_z_by_firm"),
            Tol::exact(),
        )
        .vector(
            "fill_up_z_by_firm",
            &checked(&up, "z", &case, "fill_up_z_by_firm"),
            Tol::exact(),
        )
        .finish();
    // a group fill takes only forward and backward
    let r = call(&server, &session, "munge_fill_na",
        json!({"dataset": "munging_panel", "columns": ["z"], "strategy": "mean", "group_by": ["firm"]})).await;
    assert!(!r.success, "{}", text(&r));
}

/// DATA-39: lags were positional, so a missing year shifted the wrong year in. R: plm::lag, lead
/// and diff on pdata.frame(index = c("firm", "year")), mapped back to data order.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_lag_time_index() {
    let case = Case::load("data_munging_panel_shift");
    let (session, server) = session_with("data_munging_panel_shift");
    for (args, name) in [
        (json!({"column": "y", "periods": 1}), "l1"),
        (json!({"column": "y", "periods": 2}), "l2"),
        (json!({"column": "x", "periods": 1}), "lx"),
        (json!({"column": "y", "periods": -1}), "ld"),
    ] {
        let mut args = args;
        args["dataset"] = json!("munging_panel");
        args["group_by"] = json!(["firm"]);
        args["time"] = json!("year");
        args["result_name"] = json!(name);
        let r = call(&server, &session, "munge_lag_lead", args).await;
        assert!(r.success, "{name}: {}", text(&r));
    }
    let r = call(
        &server,
        &session,
        "munge_diff",
        json!({"dataset": "munging_panel", "column": "y", "group_by": ["firm"], "time": "year",
               "result_name": "d1"}),
    )
    .await;
    assert!(r.success, "{}", text(&r));
    let r = call(
        &server,
        &session,
        "munge_diff",
        json!({"dataset": "munging_panel", "column": "x", "diff_type": "pct_change",
               "group_by": ["firm"], "time": "year", "result_name": "p1"}),
    )
    .await;
    assert!(r.success, "{}", text(&r));
    let l1 = dataset(&session, "l1").await;
    let l2 = dataset(&session, "l2").await;
    let lx = dataset(&session, "lx").await;
    let ld = dataset(&session, "ld").await;
    let d1 = dataset(&session, "d1").await;
    let p1 = dataset(&session, "p1").await;
    assert_eq!(lx.df().column("x_lag1").unwrap().dtype(), &DataType::Int64);
    let float = Tol::new(1e-14, 0.0);
    Check::new(&case)
        .vector(
            "time_lag1_y",
            &checked(&l1, "y_lag1", &case, "time_lag1_y"),
            Tol::exact(),
        )
        .vector(
            "time_lag2_y",
            &checked(&l2, "y_lag2", &case, "time_lag2_y"),
            Tol::exact(),
        )
        .vector(
            "time_lag1_x",
            &checked(&lx, "x_lag1", &case, "time_lag1_x"),
            Tol::exact(),
        )
        .vector(
            "time_lead1_y",
            &checked(&ld, "y_lead1", &case, "time_lead1_y"),
            Tol::exact(),
        )
        .vector(
            "time_diff1_y",
            &checked(&d1, "y_diff1", &case, "time_diff1_y"),
            float,
        )
        .vector(
            "time_pct1_x",
            &checked(&p1, "x_pct_change1", &case, "time_pct1_x"),
            float,
        )
        .finish();
}

/// DATA-66: a negative `periods` with direction 'lag' silently led, and munge_diff took 0 or a
/// negative `periods`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mcp_shifts_follow_the_sign_of_periods() {
    let ds = Dataset::new(df!("y" => [1.5, 2.25, 4.0, 7.5]).unwrap());
    let (session, server) = session_with_datasets(vec![("d", ds)]);
    // no direction: the sign decides
    let r = call(
        &server,
        &session,
        "munge_lag_lead",
        json!({"dataset": "d", "column": "y", "periods": -1, "result_name": "o"}),
    )
    .await;
    assert!(r.success, "{}", text(&r));
    let o = dataset(&session, "o").await;
    let lead: Vec<f64> = nums(&o, "y_lead1");
    assert_eq!(&lead[..3], &[2.25, 4.0, 7.5]);
    assert!(lead[3].is_nan());
    // 'lag' with a negative periods is an error
    let r = call(
        &server,
        &session,
        "munge_lag_lead",
        json!({"dataset": "d", "column": "y", "periods": -1, "direction": "lag"}),
    )
    .await;
    assert!(!r.success && text(&r).contains("negative"), "{}", text(&r));
    // i64::MIN has no positive counterpart: an error naming periods, not an overflow
    for direction in [None, Some("lead")] {
        let mut args = json!({"dataset": "d", "column": "y", "periods": i64::MIN});
        if let Some(direction) = direction {
            args["direction"] = json!(direction);
        }
        let r = call(&server, &session, "munge_lag_lead", args).await;
        assert!(
            !r.success && text(&r).contains("periods"),
            "{direction:?}: {}",
            text(&r)
        );
    }
    // differences need periods of at least 1
    for p in [0, -2] {
        let r = call(
            &server,
            &session,
            "munge_diff",
            json!({"dataset": "d", "column": "y", "periods": p}),
        )
        .await;
        assert!(
            !r.success && text(&r).contains("periods must be at least 1"),
            "{p}: {}",
            text(&r)
        );
    }
    // a time index must be an integer column
    let r = call(
        &server,
        &session,
        "munge_lag_lead",
        json!({"dataset": "d", "column": "y", "periods": 1, "time": "y"}),
    )
    .await;
    assert!(!r.success, "{}", text(&r));
}

/// DATA-40, DATA-41: one-hot names took polars' level order and the dropped level changed between
/// calls. R: levels(factor(x)) and model.matrix's treatment contrasts (case data_munging_factor_t4).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_one_hot_names() {
    let case = Case::load("data_munging_factor_t4");
    let (session, server) = session_with("data_munging_factor_t4"); // dataset "munging_regions"
    let r = call(
        &server,
        &session,
        "munge_one_hot_encode",
        json!({"dataset": "munging_regions", "column": "region", "result_name": "o"}),
    )
    .await;
    let t = text(&r);
    assert!(r.success, "{t}");
    let levels = case.names("region_levels");
    for l in levels {
        assert!(t.contains(&format!("region_{l}")), "{t}");
    }
    assert!(t.contains("missing in every dummy"), "{t}");
    let o = dataset(&session, "o").await;
    let mut check = Check::new(&case);
    for l in levels {
        let c = format!("region_{l}");
        check.vector(&c, &nums(&o, &c), Tol::exact());
        assert_eq!(o.df().column(&c).unwrap().null_count(), 1, "{c}");
    }
    check.finish();
    let dropped = &case.names("treatment_dropped")[0];
    for _ in 0..10 {
        let r = call(
            &server,
            &session,
            "munge_one_hot_encode",
            json!({"dataset": "munging_regions", "column": "region", "drop_first": true,
                   "result_name": "t"}),
        )
        .await;
        let t = text(&r);
        assert!(t.contains(&format!("(dropped level: {dropped})")), "{t}");
        let ds = dataset(&session, "t").await;
        assert!(ds.df().column(&format!("region_{dropped}")).is_err());
    }
}

/// DATA-44: an integer column bound with a float one failed, as "Concat failed: Join failed: ...".
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mcp_concat_binds_integers_with_floats_and_names_a_type_clash() {
    let a = Dataset::new(df!("x" => [1i64, 2], "s" => ["p", "q"]).unwrap());
    let b = Dataset::new(df!("x" => [2.5, 3.5]).unwrap());
    let c = Dataset::new(df!("x" => ["u", "v"]).unwrap());
    let (session, server) = session_with_datasets(vec![("a", a), ("b", b), ("c", c)]);
    let r = call(
        &server,
        &session,
        "munge_concat",
        json!({"datasets": ["a", "b"], "result_name": "ab"}),
    )
    .await;
    assert!(r.success, "{}", text(&r));
    let ab = dataset(&session, "ab").await;
    assert_eq!(ab.df().column("x").unwrap().dtype(), &DataType::Float64);
    assert_eq!(nums(&ab, "x"), [1.0, 2.0, 2.5, 3.5]);
    let r = call(
        &server,
        &session,
        "munge_concat",
        json!({"datasets": ["a", "c"]}),
    )
    .await;
    let t = text(&r);
    assert!(!r.success, "{t}");
    assert_eq!(t.matches("failed").count(), 1, "{t}");
    assert!(
        t.contains("Concat failed: Column 'x' has incompatible type"),
        "{t}"
    );
    assert!(!t.contains("Join failed"), "{t}");
    // the default name is replaced, and the reply says so
    let r = call(
        &server,
        &session,
        "munge_concat",
        json!({"datasets": ["a", "b"]}),
    )
    .await;
    assert!(r.success, "{}", text(&r));
    let r = call(
        &server,
        &session,
        "munge_concat",
        json!({"datasets": ["a", "b"]}),
    )
    .await;
    assert!(
        text(&r).contains("(replaced the existing dataset 'concatenated')"),
        "{}",
        text(&r)
    );
    // an in-place tool replaces its source without a note
    let r = call(&server, &session, "munge_drop_na", json!({"dataset": "ab"})).await;
    assert!(r.success && !text(&r).contains("replaced"), "{}", text(&r));
}
