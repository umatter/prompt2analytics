//! T3-WP05 MCP mechanics: unknown fields, dispatch, registry over the real entry points. Owned by T3-WP05.
use crate::common::*;
use serde_json::json;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_tier3_helpers_drive_the_real_entry_points() {
    // direct dispatch
    let (session, server) = session_with_datasets(vec![]);
    let r = call(
        &server,
        &session,
        "create_dataset",
        json!({"name": "d", "csv_content": "x,y\n1,1.1\n2,1.9\n3,3.2\n4,3.8\n5,5.1\n"}),
    )
    .await;
    assert!(r.success, "{}", text(&r));
    let listed = text(&call(&server, &session, "list_datasets", json!({})).await);
    assert!(listed.contains("**d**: 5 rows x 2 columns"), "{listed}");
    // HTTP
    let (app, state) = http_app().await;
    let sid = new_session(&app).await;
    let (status, body) = post_tool(
        &app,
        &sid,
        "create_dataset",
        json!({"name": "h", "csv_content": "a\n1\n2\n"}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        session_dataset(&state, &sid, "h")
            .await
            .expect("h")
            .df()
            .height(),
        2
    );
    // stdio
    let out = stdio_calls(
        p2a_mcp::server::AnalyticsServer::new(),
        &[("list_datasets", json!({}))],
    )
    .await;
    assert!(
        out[0]["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("No datasets currently loaded"),
        "{}",
        out[0]
    );
}

/// A declared file_type is checked against the extension (the web UI's File Type selector).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_load_dataset_checks_declared_file_type() {
    // inside the path-jail root (the test cwd); removed on drop
    let dir = tempfile::Builder::new()
        .prefix("wp05_ft_")
        .tempdir_in(env!("CARGO_MANIFEST_DIR"))
        .unwrap();
    let p = dir.path().join("ft.csv");
    std::fs::write(&p, "a,b\n1,2.5\n2,3.5\n3,4.1\n").unwrap();
    let path = p.to_str().unwrap();
    let (session, server) = session_with_datasets(vec![]);
    let ok = call(
        &server,
        &session,
        "load_dataset",
        json!({"path": path, "name": "ft", "file_type": "csv"}),
    )
    .await;
    assert!(ok.success, "{}", text(&ok));
    let bad = call(
        &server,
        &session,
        "load_dataset",
        json!({"path": path, "name": "ft2", "file_type": "parquet"}),
    )
    .await;
    assert!(
        !bad.success && text(&bad).contains("does not match the file 'ft.csv'"),
        "{}",
        text(&bad)
    );
    assert!(!session.datasets.read().await.contains_key("ft2"));
    let unk = call(
        &server,
        &session,
        "load_dataset",
        json!({"path": path, "file_type": "tsv"}),
    )
    .await;
    assert!(
        !unk.success && text(&unk).contains("unknown file_type 'tsv'"),
        "{}",
        text(&unk)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_upload_dataset_checks_declared_file_type() {
    use base64::Engine as _;
    let content = base64::engine::general_purpose::STANDARD.encode("a,b\n1,2.5\n2,3.5\n");
    let (session, server) = session_with_datasets(vec![]);
    let ok = call(
        &server,
        &session,
        "upload_dataset",
        json!({"content": content, "filename": "u.csv", "name": "u", "file_type": "csv"}),
    )
    .await;
    assert!(ok.success, "{}", text(&ok));
    let bad = call(
        &server,
        &session,
        "upload_dataset",
        json!({"content": content, "filename": "u.csv", "name": "u2", "file_type": "stata"}),
    )
    .await;
    assert!(
        !bad.success && text(&bad).contains("stata files end in .dta"),
        "{}",
        text(&bad)
    );
}

const CSV: &str = "x,y\n1,2.1\n,3.9\n3,6.2\n4,7.8\n"; // x Int64 with one missing value; y noisy

/// TOOL-13 over POST /api/tools (the finding's live repro). Before: 200 "saved as 'before'" and
/// x became f64.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_unknown_field_is_refused_http() {
    use p2a_core::polars::prelude::*;
    let (app, state) = http_app().await;
    let sid = new_session(&app).await;
    let created = post_tool(
        &app,
        &sid,
        "create_dataset",
        json!({"name": "before", "csv_content": CSV}),
    )
    .await;
    assert_eq!(created.0, 200, "{}", created.1);
    let (status, body) = post_tool(
        &app,
        &sid,
        "munge_fill_na",
        json!({"dataset": "before", "columns": ["x"], "strategy": "mean", "output_name": "after"}),
    )
    .await;
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["code"], "invalid_arguments");
    let msg = body["error"].as_str().unwrap();
    assert!(
        msg.contains("unknown field `output_name`") && msg.contains("`result_name`"),
        "{msg}"
    );
    let (status, body) = post_tool(
        &app,
        &sid,
        "regression_ols",
        json!({"dataset": "before", "y": "y", "x": ["x"], "robust_type": "HC3"}),
    )
    .await;
    assert_eq!(status, 400, "{body}");
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .contains("unknown field `robust_type`"),
        "{body}"
    );
    // the input is untouched and nothing else was written
    let before = session_dataset(&state, &sid, "before").await.unwrap();
    let x = before.df().column("x").unwrap();
    assert_eq!((x.dtype(), x.null_count()), (&DataType::Int64, 1));
    assert!(session_dataset(&state, &sid, "after").await.is_none());
    // the correct spelling still works. R: mean(c(1L, NA, 3L, 4L), na.rm = TRUE) = 2.6666666666666665
    let (status, body) = post_tool(
        &app,
        &sid,
        "munge_fill_na",
        json!({"dataset": "before", "columns": ["x"], "strategy": "mean", "result_name": "after"}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let after = session_dataset(&state, &sid, "after").await.unwrap();
    let v: Vec<f64> = after
        .df()
        .column("x")
        .unwrap()
        .as_materialized_series()
        .f64()
        .unwrap()
        .into_iter()
        .map(|o| o.unwrap())
        .collect();
    assert_eq!(v[0], 1.0);
    assert_eq!(v[2], 3.0);
    assert_eq!(v[3], 4.0);
    assert!((v[1] - 2.6666666666666665).abs() <= 1e-15, "{}", v[1]);
    let before = session_dataset(&state, &sid, "before").await.unwrap();
    assert_eq!(before.df().column("x").unwrap().null_count(), 1);
}

/// TOOL-13 in the LLM tool loop: an Ollama stub emits the misspelled call; interpret=false
/// returns the tool results. Before: is_error false and the input overwritten.
#[cfg(feature = "llm")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_unknown_field_is_refused_llm_loop() {
    use p2a_core::polars::prelude::*;
    let (app, state) = http_app().await;
    let sid = new_session(&app).await;
    let created = post_tool(
        &app,
        &sid,
        "create_dataset",
        json!({"name": "before", "csv_content": CSV}),
    )
    .await;
    assert_eq!(created.0, 200, "{}", created.1);
    let reply = json!({"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {
        "name": "munge_fill_na",
        "arguments": {"dataset": "before", "columns": ["x"], "strategy": "mean", "output_name": "after"}}}]},
        "done": true});
    let stub = axum::Router::new().route(
        "/api/chat",
        axum::routing::post(move || {
            let r = reply.clone();
            async move { axum::Json(r) }
        }),
    );
    let addr = spawn_stub(stub).await;
    let (status, body) = post_json(
        &app,
        "/api/llm/chat",
        json!({"session_id": sid, "message": "fill x", "interpret": false,
               "provider": {"provider_type": "ollama", "base_url": format!("http://{addr}"),
                            "model": "stub"}}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let r = &body["data"]["message"]["tool_results"][0];
    assert_eq!(r["is_error"], true, "{body}");
    assert!(
        r["content"]
            .as_str()
            .unwrap()
            .contains("unknown field `output_name`"),
        "{body}"
    );
    let x = session_dataset(&state, &sid, "before")
        .await
        .unwrap()
        .df()
        .column("x")
        .unwrap()
        .clone();
    assert_eq!((x.dtype(), x.null_count()), (&DataType::Int64, 1));
}

/// TOOL-13 on stdio (rmcp Parameters). Before: a success result "saved as 'before'".
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_unknown_field_is_refused_stdio() {
    let out = stdio_calls(
        p2a_mcp::server::AnalyticsServer::new(),
        &[
            (
                "create_dataset",
                json!({"name": "before", "csv_content": CSV}),
            ),
            (
                "munge_fill_na",
                json!({"dataset": "before", "columns": ["x"], "strategy": "mean", "output_name": "after"}),
            ),
            ("list_datasets", json!({"verbose": true})),
        ],
    )
    .await;
    assert!(out[0].get("result").is_some(), "{}", out[0]);
    for (i, field) in [(1, "output_name"), (2, "verbose")] {
        assert_eq!(out[i]["error"]["code"], -32602, "{}", out[i]);
        assert!(
            out[i]["error"]["message"]
                .as_str()
                .unwrap()
                .contains(&format!("unknown field `{field}`")),
            "{}",
            out[i]
        );
    }
}

/// D2 for every routed tool: a made-up key is refused before any handler runs.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_every_routed_tool_refuses_unknown_fields() {
    let (session, server) = session_with_datasets(vec![]);
    let mut wrong = Vec::new();
    for tool in server.router_tools() {
        let name = tool.name.to_string();
        match try_call(
            &server,
            &session,
            &name,
            json!({"p2a_unknown_field_probe": true}),
        )
        .await
        {
            Err(p2a_mcp::exec::ToolCallError::InvalidArguments(m))
                if m.contains("unknown field `p2a_unknown_field_probe`") => {}
            other => wrong.push(format!("{name}: {other:?}")),
        }
    }
    assert!(
        wrong.is_empty(),
        "{} tools accept an unknown field:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// The parameterless tool: null and {} run, anything else is refused (HTTP).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_no_params_tool_refuses_arguments() {
    let (app, _state) = http_app().await;
    let sid = new_session(&app).await;
    assert_eq!(
        post_json(&app, "/api/tools/list_datasets", json!({"session_id": sid}))
            .await
            .0,
        200
    );
    assert_eq!(
        post_tool(&app, &sid, "list_datasets", json!({})).await.0,
        200
    );
    let (status, body) = post_tool(&app, &sid, "list_datasets", json!({"name": "x"})).await;
    assert_eq!(
        (status, body["code"].as_str()),
        (
            axum::http::StatusCode::BAD_REQUEST,
            Some("invalid_arguments")
        ),
        "{body}"
    );
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .contains("unknown field `name`, there are no fields"),
        "{body}"
    );
}

/// Aliases stay accepted under deny_unknown_fields (kruskal: response/factor = value/group).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_aliases_are_still_accepted() {
    use p2a_core::polars::prelude::*;
    // Hollander & Wolfe (?kruskal.test): R kruskal.test(list(x, y, z))
    let ds = p2a_core::data::Dataset::new(
        df!(
            "v" => [2.9, 3.0, 2.5, 2.6, 3.2, 3.8, 2.7, 4.0, 2.4, 2.8, 3.4, 3.7, 2.2, 2.0],
            "g" => ["a", "a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c", "c"]
        )
        .unwrap(),
    );
    let (session, server) = session_with_datasets(vec![("k", ds)]);
    let canonical = call(
        &server,
        &session,
        "hypothesis_kruskal_wallis",
        json!({"dataset": "k", "value": "v", "group": "g"}),
    )
    .await;
    let aliased = call(
        &server,
        &session,
        "hypothesis_kruskal_wallis",
        json!({"dataset": "k", "response": "v", "factor": "g"}),
    )
    .await;
    assert!(
        canonical.success && aliased.success,
        "{} / {}",
        text(&canonical),
        text(&aliased)
    );
    assert_eq!(text(&canonical), text(&aliased));
}

/// GET /api/tools lists the tools sorted by name, the same in every call (before: HashMap
/// order, different in every process).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_tool_list_is_sorted_and_stable() {
    let (app, _state) = http_app().await;
    let get = |app: axum::Router| async move {
        use tower::ServiceExt;
        let req = axum::http::Request::get("/api/tools")
            .header("host", "localhost")
            .body(axum::body::Body::empty())
            .unwrap();
        let b = axum::body::to_bytes(app.oneshot(req).await.unwrap().into_body(), usize::MAX)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&b).unwrap();
        v["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap().to_string())
            .collect::<Vec<_>>()
    };
    let a = get(app.clone()).await;
    let mut sorted = a.clone();
    sorted.sort();
    assert_eq!(a, sorted);
    assert_eq!(a, get(app).await);
}
