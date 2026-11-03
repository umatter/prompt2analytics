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
