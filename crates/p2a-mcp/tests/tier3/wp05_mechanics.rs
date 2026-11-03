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
