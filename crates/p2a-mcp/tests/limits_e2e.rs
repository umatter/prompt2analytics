//! End-to-end tests of the request limits, the response-size cap and the
//! audit trail on every entry path (T0-WP06): the HTTP route, the LLM tool
//! loop and websocket messages (all through the tool governor, feature
//! `http`) and the stdio transport (always built).
//!
//! No test calls `limits::install`, which is process-global: each test picks
//! its limits with `limits::with_limits` on its own `current_thread` runtime,
//! and the governor carries them onto its blocking thread.

use p2a_mcp::tools::limits::{self, Limits};

/// Runs `f` on a current_thread runtime with `l` as this thread's limits. The
/// governor carries them to its blocking thread. Its `Handle::block_on` works
/// there because this thread keeps driving IO and timers while it awaits the
/// join.
fn run<F: std::future::Future>(l: Limits, f: F) -> F::Output {
    limits::with_limits(l, || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime")
            .block_on(f)
    })
}

#[cfg(feature = "http")]
mod http_paths {
    use super::*;

    use axum::Router;
    use axum::body::Body;
    use axum::http::{Request, StatusCode, header};
    use p2a_mcp::config::{ServerConfig, TransportType};
    use serde_json::{Value, json};
    use tower::ServiceExt;

    const ABSURD: u64 = 10_000_000_000;

    /// Open public mode (no token, `--allow-unauthenticated`): the hosted
    /// profile applies.
    pub(super) fn open_public_config() -> ServerConfig {
        let mut c = ServerConfig {
            transport: TransportType::Http,
            ..Default::default()
        };
        c.http.addr = "0.0.0.0:8080".parse().unwrap();
        c.http.allow_unauthenticated = true;
        c
    }

    /// Local mode (loopback, no token): limits off by default.
    pub(super) fn local_config() -> ServerConfig {
        ServerConfig {
            transport: TransportType::Http,
            ..Default::default()
        }
    }

    async fn send(app: &Router, req: Request<Body>) -> (StatusCode, Value) {
        let resp = app.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    }

    fn post(uri: &str, body: Value) -> Request<Body> {
        Request::post(uri)
            .header(header::HOST, "localhost")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    pub(super) async fn new_session(app: &Router) -> String {
        let (status, body) = send(app, post("/api/sessions", json!({}))).await;
        assert_eq!(status, StatusCode::CREATED, "{body}");
        body["data"]["session_id"].as_str().unwrap().to_string()
    }

    pub(super) async fn post_tool(
        app: &Router,
        sid: &str,
        tool: &str,
        args: Value,
    ) -> (StatusCode, Value) {
        send(
            app,
            post(
                &format!("/api/tools/{tool}"),
                json!({"session_id": sid, "arguments": args}),
            ),
        )
        .await
    }

    async fn get_health(app: &Router) -> StatusCode {
        let req = Request::get("/health")
            .header(header::HOST, "localhost")
            .body(Body::empty())
            .unwrap();
        send(app, req).await.0
    }

    #[test]
    fn x1_canonical_case_is_refused() {
        let cfg = open_public_config();
        let hosted = limits::resolve(&cfg).unwrap();
        run(hosted, async {
            let (app, _state) = p2a_mcp::transport::http::build_app(&cfg).await.unwrap();
            let sid = new_session(&app).await;
            let (status, body) = post_tool(
                &app,
                &sid,
                "timeseries_arma_to_ma",
                json!({"lag_max": ABSURD, "ar": [0.5], "ma": [0.3]}),
            )
            .await;
            assert_eq!(status, 422, "{body}");
            assert_eq!(body["success"], false);
            assert_eq!(body["code"], "limit_exceeded");
            assert_eq!(
                body["details"],
                json!({"parameter": "lag_max", "requested": "10000000000", "limit": "1000"})
            );
            assert_eq!(
                body["error"],
                "Parameter `lag_max` = 10000000000 exceeds this server's limit of 1000 (lags) \
                 for `timeseries_arma_to_ma`. Use a smaller value."
            );
            assert_eq!(get_health(&app).await, 200);
        });
    }

    /// Every tool from X-1's evidence plus the merged duplicates: (tool,
    /// field, value, hosted limit fixed by design 4.2).
    const X1_CASES: &[(&str, &str, u64, Option<&str>)] = &[
        ("timeseries_arma_to_ma", "lag_max", ABSURD, Some("1000")),
        ("timeseries_arma_acf", "lag_max", ABSURD, Some("1000")),
        ("stats_density", "n", ABSURD, Some("10000")),
        ("viz_histogram", "bins", ABSURD, Some("1000")),
        ("regression_glmnet", "nlambda", ABSURD, Some("10000")),
        ("regression_cv_glmnet", "nlambda", ABSURD, Some("10000")),
        ("rd_estimate", "p", ABSURD, Some("20")),
        ("rd_fuzzy", "p", ABSURD, Some("20")),
        ("rd_bw", "p", ABSURD, Some("20")),
        ("rd_multi", "p", ABSURD, Some("20")),
        ("iv_mte", "mte_degree", ABSURD, Some("20")),
        ("iv_mte", "n_grid", ABSURD, Some("10000")),
        ("mixed_logit", "n_draws", ABSURD, Some("20000")),
        ("treatment_ipw", "bootstrap", ABSURD, Some("10000")),
        ("treatment_twang", "n_trees", ABSURD, Some("5000")),
        ("ml_random_forest", "n_trees", ABSURD, Some("5000")),
        ("ml_shap_values", "n_trees", ABSURD, Some("5000")),
        ("ml_causal_forest", "n_trees", ABSURD, Some("5000")),
        ("ml_bart_causal", "n_trees", ABSURD, Some("5000")),
        ("ml_cubist", "committees", ABSURD, Some("5000")),
        ("ml_c50", "trials", ABSURD, Some("10000")),
        ("ml_ppr", "nterms", ABSURD, Some("10000")),
        ("propensity_matching", "ratio", ABSURD, None),
        ("regression_bootstrap_cov", "n_boot", ABSURD, Some("10000")),
        ("regression_standardization", "n_bootstrap", ABSURD, Some("10000")),
        ("collaborative_tmle", "n_folds", ABSURD, Some("20")),
        ("ts_var_irf", "steps", ABSURD, Some("5000")),
        ("ts_holt_winters", "horizon", ABSURD, Some("5000")),
        ("timeseries_arima_sim", "n", ABSURD, Some("1000000")),
        ("timeseries_diffinv", "lag", ABSURD, Some("1000")),
        ("timeseries_diffinv", "differences", ABSURD, Some("1000")),
        ("heterogeneity_test", "n_permutations", ABSURD, Some("10000")),
        // merged duplicates of X-1
        ("ml_kmeans", "n_init", ABSURD, Some("100")),
        ("ml_tsne", "max_iterations", ABSURD, Some("100000")),
        ("ml_mboost", "mstop", ABSURD, Some("10000")),
        ("sar_probit_model", "n_draws", ABSURD, Some("20000")),
        ("viz_dendrogram", "width", 100_000, Some("4000")), // u32 field
    ];

    #[test]
    fn x1_cost_parameters_are_refused_in_open_public_mode() {
        let cfg = open_public_config();
        let hosted = limits::resolve(&cfg).unwrap();
        // A case the catalogue does not refuse would run the real allocation,
        // which aborts the test binary; list those as catalogue gaps instead
        // of sending them.
        let gaps: Vec<String> = X1_CASES
            .iter()
            .filter(|(tool, field, value, _)| {
                hosted
                    .check_tool_arguments(tool, &json!({ *field: value }))
                    .is_ok()
            })
            .map(|(tool, field, _, _)| format!("{tool}.{field}"))
            .collect();
        assert!(gaps.is_empty(), "catalogue gaps: {gaps:?}");

        run(hosted, async {
            let (app, _state) = p2a_mcp::transport::http::build_app(&cfg).await.unwrap();
            let sid = new_session(&app).await;
            let mut failures = Vec::new();
            for &(tool, field, value, limit) in X1_CASES {
                // Only the cost field: the check precedes deserialisation.
                let (status, body) = post_tool(&app, &sid, tool, json!({ field: value })).await;
                let details = &body["details"];
                let limit_ok = match limit {
                    Some(expected) => details["limit"] == expected,
                    None => details["limit"]
                        .as_str()
                        .and_then(|l| l.parse::<u64>().ok())
                        .is_some_and(|l| l < value),
                };
                let error = body["error"].as_str().unwrap_or_default();
                let ok = status == StatusCode::UNPROCESSABLE_ENTITY
                    && body["code"] == "limit_exceeded"
                    && details["parameter"] == field
                    && details["requested"] == value.to_string()
                    && limit_ok
                    && error.contains("exceeds this server's limit")
                    && error.contains(&format!("`{field}`"));
                if !ok {
                    failures.push(format!("{tool}.{field}: {status} {body}"));
                }
            }
            assert!(failures.is_empty(), "not refused as expected:\n{failures:#?}");
            assert_eq!(get_health(&app).await, 200);
        });
    }

    #[test]
    fn x1_sane_values_still_run_in_open_public_mode() {
        let cfg = open_public_config();
        run(limits::resolve(&cfg).unwrap(), async {
            let (app, _state) = p2a_mcp::transport::http::build_app(&cfg).await.unwrap();
            let sid = new_session(&app).await;
            let (status, body) = post_tool(
                &app,
                &sid,
                "timeseries_arma_to_ma",
                json!({"lag_max": 10, "ar": [0.5], "ma": [0.3]}),
            )
            .await;
            assert_eq!(status, 200, "{body}");
            assert_eq!(body["data"]["success"], true, "{body}");
        });
    }

    #[test]
    fn x1_local_mode_is_unlimited() {
        // The local default (limits off) runs a value above the hosted cap.
        let cfg = local_config();
        run(limits::resolve(&cfg).unwrap(), async {
            let (app, _state) = p2a_mcp::transport::http::build_app(&cfg).await.unwrap();
            let sid = new_session(&app).await;
            let (status, body) = post_tool(
                &app,
                &sid,
                "timeseries_arma_to_ma",
                json!({"lag_max": 2000, "ar": [0.5], "ma": [0.3]}),
            )
            .await;
            assert_eq!(status, 200, "{body}");
            assert_eq!(body["data"]["success"], true, "{body}");
        });
    }
}
