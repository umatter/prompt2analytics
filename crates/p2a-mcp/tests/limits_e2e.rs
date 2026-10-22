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

/// An enabled audit configuration writing to `dir/audit.log`.
fn audit_cfg(dir: &tempfile::TempDir) -> p2a_mcp::config::AuditConfig {
    p2a_mcp::config::AuditConfig {
        enabled: true,
        path: dir.path().join("audit.log").to_string_lossy().into_owned(),
    }
}

/// Every line of the audit log at `cfg.path`, read after flushing the
/// process's one writer for that file (`AuditLogger::new` returns it while
/// another handle keeps it alive).
async fn audit_lines(cfg: &p2a_mcp::config::AuditConfig) -> Vec<serde_json::Value> {
    p2a_mcp::audit::AuditLogger::new(cfg).unwrap().flush().await;
    std::fs::read_to_string(&cfg.path)
        .unwrap_or_default()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[cfg(feature = "http")]
mod http_paths {
    use super::*;

    use axum::Router;
    use axum::body::Body;
    use axum::http::{Request, StatusCode, header};
    use p2a_mcp::config::{ServerConfig, TransportType};
    use p2a_mcp::transport::http::AppState;
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
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
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
        (
            "regression_standardization",
            "n_bootstrap",
            ABSURD,
            Some("10000"),
        ),
        ("collaborative_tmle", "n_folds", ABSURD, Some("20")),
        ("ts_var_irf", "steps", ABSURD, Some("5000")),
        ("ts_holt_winters", "horizon", ABSURD, Some("5000")),
        ("timeseries_arima_sim", "n", ABSURD, Some("1000000")),
        ("timeseries_diffinv", "lag", ABSURD, Some("1000")),
        ("timeseries_diffinv", "differences", ABSURD, Some("1000")),
        (
            "heterogeneity_test",
            "n_permutations",
            ABSURD,
            Some("10000"),
        ),
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
                    && details["requested"].as_str() == Some(value.to_string().as_str())
                    && limit_ok
                    && error.contains("exceeds this server's limit")
                    && error.contains(&format!("`{field}`"));
                if !ok {
                    failures.push(format!("{tool}.{field}: {status} {body}"));
                }
            }
            assert!(
                failures.is_empty(),
                "not refused as expected:\n{failures:#?}"
            );
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

    /// Writes a noisy four-column dataset of `rows` rows straight into the
    /// session (bypassing the dataset quotas: only the limits are under test).
    pub(super) async fn insert_dataset(state: &AppState, sid: &str, name: &str, rows: usize) {
        let x1: Vec<f64> = (0..rows)
            .map(|i| ((i * 7919) % 1013) as f64 / 1013.0)
            .collect();
        let x2: Vec<f64> = (0..rows)
            .map(|i| ((i * 104_729) % 997) as f64 / 997.0)
            .collect();
        let y: Vec<f64> = (0..rows)
            .map(|i| 1.0 + 2.0 * x1[i] - x2[i] + ((i * 31) % 17) as f64 / 170.0)
            .collect();
        let t: Vec<f64> = (0..rows).map(|i| (i % 2) as f64).collect();
        let df = polars::prelude::df! { "x1" => x1, "x2" => x2, "y" => y, "t" => t }.unwrap();
        state
            .session_manager
            .get_session(sid)
            .await
            .unwrap()
            .datasets
            .write()
            .await
            .insert(name.to_string(), p2a_core::data::Dataset::new(df));
    }

    #[test]
    fn x4_row_class_override_refuses_small_dataset() {
        let cfg = open_public_config();
        let l = Limits::hosted()
            .with_overrides(&["dense_rows=50".to_string()])
            .unwrap();
        run(l, async {
            let (app, state) = p2a_mcp::transport::http::build_app(&cfg).await.unwrap();
            let sid = new_session(&app).await;
            insert_dataset(&state, &sid, "small", 60).await;
            let (status, body) = post_tool(
                &app,
                &sid,
                "ml_hierarchical",
                json!({"dataset": "small", "columns": ["x1", "x2"], "n_clusters": 2}),
            )
            .await;
            assert_eq!(status, 422, "{body}");
            assert_eq!(body["code"], "limit_exceeded", "{body}");
            assert_eq!(body["details"]["parameter"], "dataset", "{body}");
            assert_eq!(body["details"]["requested"], "60", "{body}");
            assert_eq!(body["details"]["limit"], "50", "{body}");
            assert!(body["error"].as_str().unwrap().contains("rows"), "{body}");
        });
    }

    #[test]
    fn x4_row_classes_do_not_apply_when_off() {
        let cfg = local_config();
        run(Limits::off(), async {
            let (app, state) = p2a_mcp::transport::http::build_app(&cfg).await.unwrap();
            let sid = new_session(&app).await;
            insert_dataset(&state, &sid, "small", 60).await;
            let (status, body) = post_tool(
                &app,
                &sid,
                "ml_hierarchical",
                json!({"dataset": "small", "columns": ["x1", "x2"], "n_clusters": 2}),
            )
            .await;
            assert_eq!(status, 200, "{body}");
            assert_eq!(body["data"]["success"], true, "{body}");
        });
    }

    #[test]
    fn x4_every_row_rule_refuses_oversized_datasets() {
        use p2a_mcp::tools::limits::{ROW_RULES, RowClass};
        assert!(!ROW_RULES.is_empty());
        let cfg = open_public_config();
        let hosted = limits::resolve(&cfg).unwrap();
        // Design 4.3 caps, one dataset just above each.
        let class = |c: RowClass| match c {
            RowClass::Dense => ("dense", 5_000usize),
            RowClass::Cubic => ("cubic", 2_000),
            RowClass::Quadratic => ("quadratic", 50_000),
        };
        for c in [RowClass::Dense, RowClass::Cubic, RowClass::Quadratic] {
            assert_eq!(hosted.row_cap(c), Some(class(c).1), "{c:?}");
        }
        run(hosted, async {
            let (app, state) = p2a_mcp::transport::http::build_app(&cfg).await.unwrap();
            let sid = new_session(&app).await;
            for c in [RowClass::Dense, RowClass::Cubic, RowClass::Quadratic] {
                let (name, cap) = class(c);
                insert_dataset(&state, &sid, name, cap + 1).await;
            }
            let mut failures = Vec::new();
            for rule in ROW_RULES {
                let (name, cap) = class(rule.class);
                let (status, body) =
                    post_tool(&app, &sid, rule.tool, json!({ rule.dataset_field: name })).await;
                let ok = status == StatusCode::UNPROCESSABLE_ENTITY
                    && body["code"] == "limit_exceeded"
                    && body["details"]["requested"].as_str()
                        == Some((cap + 1).to_string().as_str())
                    && body["details"]["limit"].as_str() == Some(cap.to_string().as_str());
                if !ok {
                    failures.push(format!("{}: {status} {body}", rule.tool));
                }
            }
            assert!(
                failures.is_empty(),
                "not refused as expected:\n{failures:#?}"
            );
        });
    }

    #[test]
    fn x4_row_rules_name_a_string_field_of_their_tool() {
        let tools = p2a_mcp::AnalyticsServer::new().list_tools();
        for rule in limits::ROW_RULES {
            let tool = tools
                .iter()
                .find(|t| t.name == rule.tool)
                .unwrap_or_else(|| panic!("ROW_RULES names unknown tool {}", rule.tool));
            assert_eq!(
                tool.input_schema["properties"][rule.dataset_field]["type"], "string",
                "{}.{} is not a string field",
                rule.tool, rule.dataset_field
            );
        }
    }

    #[test]
    fn response_cap_refuses_oversized_output() {
        let cfg = open_public_config();
        let head = |app: Router, state: AppState| async move {
            let sid = new_session(&app).await;
            insert_dataset(&state, &sid, "small", 60).await;
            post_tool(
                &app,
                &sid,
                "head_dataset",
                json!({"dataset": "small", "n": 50}),
            )
            .await
        };
        let l = Limits::hosted()
            .with_overrides(&["response_bytes=200".to_string()])
            .unwrap();
        run(l, async {
            let (app, state) = p2a_mcp::transport::http::build_app(&cfg).await.unwrap();
            let (status, body) = head(app, state).await;
            assert_eq!(status, 422, "{body}");
            assert_eq!(body["code"], "limit_exceeded", "{body}");
            assert_eq!(body["details"]["parameter"], "response_bytes", "{body}");
            assert_eq!(body["details"]["limit"], "200", "{body}");
            let requested: u64 = body["details"]["requested"]
                .as_str()
                .and_then(|r| r.parse().ok())
                .unwrap();
            assert!(requested > 200, "{body}");
            assert!(
                body["error"]
                    .as_str()
                    .unwrap()
                    .contains("above this server's limit of 200 bytes"),
                "{body}"
            );
        });
        // The hosted default (16 MiB) lets the same call through.
        run(Limits::hosted(), async {
            let (app, state) = p2a_mcp::transport::http::build_app(&cfg).await.unwrap();
            let (status, body) = head(app, state).await;
            assert_eq!(status, 200, "{body}");
            assert_eq!(body["data"]["success"], true, "{body}");
        });
    }

    #[test]
    fn response_cap_does_not_bypass_dataset_quotas() {
        // The handler runs before its output is measured. A refused output
        // must not let the call's dataset growth escape the quota check.
        let cfg = local_config();
        let l = Limits::off()
            .with_overrides(&["response_bytes=10".to_string()])
            .unwrap();
        run(l, async {
            let exec = p2a_mcp::exec::ExecConfig {
                session_data_bytes: Some(1),
                ..Default::default()
            };
            let (app, state) = p2a_mcp::transport::http::build_app_with(&cfg, exec)
                .await
                .unwrap();
            let sid = new_session(&app).await;
            let (status, body) = post_tool(
                &app,
                &sid,
                "create_dataset",
                json!({"name": "d", "csv_content": "x,y\n1.0,1.1\n2.0,1.9\n3.0,3.2\n"}),
            )
            .await;
            assert_eq!(status, 422, "{body}");
            assert_eq!(body["code"], "quota_exceeded", "{body}");
            let session = state.session_manager.get_session(&sid).await.unwrap();
            assert!(
                session.datasets.read().await.is_empty(),
                "change not undone"
            );
        });
    }

    /// `cfg` with auditing enabled into `dir/audit.log`.
    fn with_audit(mut cfg: ServerConfig, dir: &tempfile::TempDir) -> ServerConfig {
        cfg.audit = super::audit_cfg(dir);
        cfg
    }

    #[test]
    fn audit_records_every_origin_once() {
        use p2a_mcp::exec::CallOrigin;
        let dir = tempfile::tempdir().unwrap();
        let cfg = with_audit(local_config(), &dir);
        let (lines, sid) = run(Limits::off(), async {
            let (app, state) = p2a_mcp::transport::http::build_app(&cfg).await.unwrap();
            let sid = new_session(&app).await;
            let (status, body) = post_tool(&app, &sid, "list_datasets", json!({})).await;
            assert_eq!(status, 200, "{body}");
            let session = state.session_manager.get_session(&sid).await.unwrap();
            for origin in [CallOrigin::LlmTool, CallOrigin::WebSocket] {
                state
                    .governor
                    .call(
                        state.server.clone(),
                        session.clone(),
                        "list_datasets",
                        json!({}),
                        origin,
                    )
                    .await
                    .unwrap();
            }
            (super::audit_lines(&cfg.audit).await, sid)
        });
        assert_eq!(lines.len(), 3, "{lines:#?}");
        let origins: Vec<&str> = lines
            .iter()
            .map(|l| l["origin"].as_str().unwrap())
            .collect();
        assert_eq!(origins, ["http", "llm_tool", "websocket"]);
        for line in &lines {
            assert_eq!(line["tool_name"], "list_datasets", "{line}");
            assert_eq!(line["outcome"], "ok", "{line}");
            assert_eq!(line["success"], true, "{line}");
            assert_eq!(line["session_id"], sid.as_str(), "{line}");
        }
    }

    #[test]
    fn audit_contains_no_data_values() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = with_audit(local_config(), &dir);
        run(Limits::off(), async {
            let (app, _state) = p2a_mcp::transport::http::build_app(&cfg).await.unwrap();
            let sid = new_session(&app).await;
            let csv = "email,income\nalice@example.org,51000\nbob@example.org,73000\n";
            let (status, body) = post_tool(
                &app,
                &sid,
                "create_dataset",
                json!({"name": "pii", "csv_content": csv}),
            )
            .await;
            assert_eq!(status, 200, "{body}");
            let (status, body) =
                post_tool(&app, &sid, "head_dataset", json!({"dataset": "pii"})).await;
            assert_eq!(status, 200, "{body}");
            assert!(body.to_string().contains("alice@example.org"), "{body}");
            super::audit_lines(&cfg.audit).await;
        });
        let text = std::fs::read_to_string(&cfg.audit.path).unwrap();
        assert!(!text.contains("alice@example.org"), "{text}");
        assert!(!text.contains("51000"), "{text}");
        let lines: Vec<Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines.len(), 2, "{text}");
        assert_eq!(lines[0]["tool_name"], "create_dataset");
        assert_eq!(lines[0]["arguments"]["csv_content"], "[redacted: 59 bytes]");
        assert_eq!(lines[1]["tool_name"], "head_dataset");
        assert!(
            lines[1]["result"]["text_bytes"].as_u64().unwrap() > 0,
            "{text}"
        );
    }

    #[test]
    fn audit_records_limit_refusals() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = with_audit(open_public_config(), &dir);
        let lines = run(limits::resolve(&cfg).unwrap(), async {
            let (app, _state) = p2a_mcp::transport::http::build_app(&cfg).await.unwrap();
            let sid = new_session(&app).await;
            let (status, body) = post_tool(
                &app,
                &sid,
                "timeseries_arma_to_ma",
                json!({"lag_max": ABSURD}),
            )
            .await;
            assert_eq!(status, 422, "{body}");
            super::audit_lines(&cfg.audit).await
        });
        assert_eq!(lines.len(), 1, "{lines:#?}");
        let line = &lines[0];
        assert_eq!(line["origin"], "http");
        assert_eq!(line["outcome"], "limit_exceeded");
        assert_eq!(line["success"], false);
        assert_eq!(line["arguments"]["lag_max"], ABSURD);
    }

    /// A refusal before the work starts (memory admission; Linux reports the
    /// process memory) is written once, by the governor's async side.
    #[cfg(target_os = "linux")]
    #[test]
    fn audit_records_admission_refusals_once() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = with_audit(local_config(), &dir);
        let lines = run(Limits::off(), async {
            let mut exec = p2a_mcp::exec::ExecConfig::from_config(&cfg);
            exec.rss_soft_limit_bytes = Some(1);
            let (app, _state) = p2a_mcp::transport::http::build_app_with(&cfg, exec)
                .await
                .unwrap();
            let sid = new_session(&app).await;
            let (status, body) = post_tool(&app, &sid, "list_datasets", json!({})).await;
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
            assert_eq!(body["code"], "memory_pressure", "{body}");
            super::audit_lines(&cfg.audit).await
        });
        assert_eq!(lines.len(), 1, "{lines:#?}");
        assert_eq!(lines[0]["outcome"], "memory_pressure");
        assert_eq!(lines[0]["success"], false);
        assert_eq!(lines[0]["duration_ms"], 0);
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

mod stdio_path {
    use super::*;

    use p2a_mcp::AnalyticsServer;
    use rmcp::ServiceExt;
    use serde_json::{Value, json};
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    async fn send_line<W: AsyncWriteExt + Unpin>(w: &mut W, msg: Value) {
        w.write_all(format!("{msg}\n").as_bytes()).await.unwrap();
        w.flush().await.unwrap();
    }

    async fn read_until_id<R: AsyncBufReadExt + Unpin>(
        lines: &mut tokio::io::Lines<R>,
        id: u64,
    ) -> Value {
        loop {
            let line = lines
                .next_line()
                .await
                .unwrap()
                .expect("server closed the stream");
            let msg: Value = serde_json::from_str(&line).unwrap();
            if msg["id"] == id {
                return msg;
            }
        }
    }

    /// Serves `server` over an in-memory duplex stream, as the stdio
    /// transport does over stdin/stdout, initialises the session and returns
    /// the JSON-RPC response to one request (`method`, `params`).
    async fn stdio_request(server: AnalyticsServer, method: &str, params: Value) -> Value {
        let (client, server_end) = tokio::io::duplex(1 << 22);
        let (r, w) = tokio::io::split(server_end);
        let service = tokio::spawn(async move {
            let running = server.serve((r, w)).await.expect("serve");
            let _ = running.waiting().await;
        });
        let (cr, mut cw) = tokio::io::split(client);
        let mut lines = BufReader::new(cr).lines();
        send_line(
            &mut cw,
            json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "limits-e2e", "version": "0"}}}),
        )
        .await;
        read_until_id(&mut lines, 1).await;
        send_line(
            &mut cw,
            json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        )
        .await;
        send_line(
            &mut cw,
            json!({"jsonrpc": "2.0", "id": 2, "method": method, "params": params}),
        )
        .await;
        let msg = read_until_id(&mut lines, 2).await;
        drop(cw);
        service.abort();
        msg
    }

    /// The JSON-RPC response to one `tools/call` over the stdio transport.
    pub(super) async fn stdio_call(server: AnalyticsServer, tool: &str, args: Value) -> Value {
        stdio_request(
            server,
            "tools/call",
            json!({"name": tool, "arguments": args}),
        )
        .await
    }

    #[test]
    fn stdio_refuses_cost_parameters_when_hosted() {
        let msg = run(
            Limits::hosted(),
            stdio_call(
                AnalyticsServer::new(),
                "timeseries_arma_to_ma",
                json!({"lag_max": 2000, "ar": [0.5], "ma": [0.3]}),
            ),
        );
        assert_eq!(msg["error"]["code"], -32602, "{msg}");
        assert!(
            msg["error"]["message"]
                .as_str()
                .unwrap()
                .contains("exceeds this server's limit of 1000"),
            "{msg}"
        );
        assert_eq!(msg["error"]["data"]["code"], "limit_exceeded", "{msg}");
        assert_eq!(msg["error"]["data"]["parameter"], "lag_max", "{msg}");
        assert_eq!(msg["error"]["data"]["requested"], "2000", "{msg}");
        assert_eq!(msg["error"]["data"]["limit"], "1000", "{msg}");
    }

    #[test]
    fn stdio_runs_the_same_call_when_off() {
        let msg = run(
            Limits::off(),
            stdio_call(
                AnalyticsServer::new(),
                "timeseries_arma_to_ma",
                json!({"lag_max": 2000, "ar": [0.5], "ma": [0.3]}),
            ),
        );
        assert!(msg.get("error").is_none(), "{msg}");
        assert_ne!(msg["result"]["isError"], true, "{msg}");
    }

    #[test]
    fn stdio_caps_the_response_size() {
        let l = Limits::off()
            .with_overrides(&["response_bytes=100".to_string()])
            .unwrap();
        let msg = run(
            l,
            stdio_call(
                AnalyticsServer::new(),
                "timeseries_arma_to_ma",
                json!({"lag_max": 200, "ar": [0.5], "ma": [0.3]}),
            ),
        );
        assert_eq!(msg["error"]["code"], -32602, "{msg}");
        assert_eq!(msg["error"]["data"]["code"], "limit_exceeded", "{msg}");
        assert_eq!(msg["error"]["data"]["parameter"], "response_bytes", "{msg}");
        assert_eq!(msg["error"]["data"]["limit"], "100", "{msg}");
    }

    #[test]
    fn stdio_calls_are_audited() {
        use p2a_mcp::audit::AuditLogger;
        use std::sync::Arc;
        let dir = tempfile::tempdir().unwrap();
        let cfg = audit_cfg(&dir);
        // This handle keeps the writer alive until the log is read.
        let logger = Arc::new(AuditLogger::new(&cfg).unwrap());
        let server = AnalyticsServer::new().with_audit_logger(logger.clone());
        let msg = run(
            Limits::off(),
            stdio_call(
                server,
                "timeseries_arma_to_ma",
                json!({"lag_max": 10, "ar": [0.5]}),
            ),
        );
        assert!(msg.get("error").is_none(), "{msg}");
        let lines = run(Limits::off(), audit_lines(&cfg));
        assert_eq!(lines.len(), 1, "{lines:#?}");
        let line = &lines[0];
        assert_eq!(line["origin"], "stdio");
        assert_eq!(line["session_id"], "stdio");
        assert_eq!(line["tool_name"], "timeseries_arma_to_ma");
        assert_eq!(line["outcome"], "ok");
        assert!(line["result"]["text_bytes"].as_u64().unwrap() > 0, "{line}");
        drop(logger);
    }

    #[test]
    fn stdio_audits_refusals_and_unknown_tools() {
        use p2a_mcp::audit::AuditLogger;
        use std::sync::Arc;
        let dir = tempfile::tempdir().unwrap();
        let cfg = audit_cfg(&dir);
        let logger = Arc::new(AuditLogger::new(&cfg).unwrap());
        let calls = [
            (
                "timeseries_arma_to_ma",
                json!({"lag_max": 2000, "ar": [0.5]}),
            ),
            ("no_such_tool", json!({})),
            ("timeseries_arma_to_ma", json!({"lag_max": "ten"})),
        ];
        for (tool, args) in calls {
            let server = AnalyticsServer::new().with_audit_logger(logger.clone());
            let msg = run(Limits::hosted(), stdio_call(server, tool, args));
            assert!(msg.get("error").is_some(), "{msg}");
        }
        let lines = run(Limits::off(), audit_lines(&cfg));
        let outcomes: Vec<&str> = lines
            .iter()
            .map(|l| l["outcome"].as_str().unwrap())
            .collect();
        assert_eq!(
            outcomes,
            ["limit_exceeded", "unknown_tool", "invalid_arguments"]
        );
        assert!(lines.iter().all(|l| l["success"] == false));
        drop(logger);
    }

    #[test]
    fn stdio_lists_every_tool() {
        // The manual ServerHandler still lists the router's tools.
        let msg = run(
            Limits::off(),
            stdio_request(AnalyticsServer::new(), "tools/list", json!({})),
        );
        let tools = msg["result"]["tools"].as_array().expect("tools");
        assert!(tools.len() > 200, "{} tools", tools.len());
        assert!(tools.iter().any(|t| t["name"] == "timeseries_arma_to_ma"));
    }
}
