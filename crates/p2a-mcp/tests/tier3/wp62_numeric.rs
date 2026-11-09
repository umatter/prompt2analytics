//! T3-WP62 MCP-path tests (reply sizes of the numeric sweep). Owned by T3-WP62.
use crate::common::*;
use serde_json::{Value, json};

/// A noisy regression dataset with `m` features x0..x{m-1}, as CSV text.
fn noisy_csv(n: usize, m: usize) -> String {
    let mut csv: Vec<String> = vec![
        (0..m)
            .map(|j| format!("x{j}"))
            .chain(["y".to_string()])
            .collect::<Vec<_>>()
            .join(","),
    ];
    for i in 0..n {
        let x: Vec<f64> = (0..m)
            .map(|j| (((i * 31 + j * 17) % 101) as f64 / 10.0 + (i as f64 * 0.7 + j as f64).sin()))
            .collect();
        let y = x
            .iter()
            .enumerate()
            .map(|(j, v)| v * (1.0 + j as f64 * 0.3))
            .sum::<f64>()
            + x[0] * x[1] * 0.2
            + (i as f64 * 1.3).cos();
        csv.push(
            x.iter()
                .chain([y].iter())
                .map(|v| format!("{v:.6}"))
                .collect::<Vec<_>>()
                .join(","),
        );
    }
    csv.join("\n")
}

/// The SHAP interactions reply at M = 10 with the default arguments fits the chat loop's
/// tool-result budget: per explained row the main effects and the 10 largest of the 45
/// pairs, with the count of the pairs not shown. Before: the three full 10 x 10 matrices
/// took the reply past the budget, so the chat loop cut it.
#[cfg(feature = "llm")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_shap_interactions_reply_fits_the_chat_budget() {
    let (s, server) = session_with_datasets(vec![]);
    let r = call(
        &server,
        &s,
        "create_dataset",
        json!({"name": "d", "csv_content": noisy_csv(300, 10)}),
    )
    .await;
    assert!(r.success, "{}", text(&r));
    let features: Vec<String> = (0..10).map(|j| format!("x{j}")).collect();
    let r = call(
        &server,
        &s,
        "ml_shap_values",
        json!({"dataset": "d", "features": features, "target": "y", "seed": 1,
               "compute_interactions": true}),
    )
    .await;
    assert!(r.success, "{}", text(&r));
    let t = text(&r);
    assert!(
        t.len() < p2a_mcp::llm::MAX_TOOL_RESULT_BYTES_IN_LOOP,
        "reply is {} bytes",
        t.len()
    );
    let j: Value = serde_json::from_str(t.split("JSON Summary:\n").nth(1).unwrap()).unwrap();
    let rows = j["interaction_values_sample"].as_array().unwrap();
    assert_eq!(rows.len(), 3);
    for (k, row) in rows.iter().enumerate() {
        assert_eq!(row["row"], json!(k));
        assert_eq!(row["main_effects"].as_array().unwrap().len(), 10);
        let pairs = row["top_pairs"].as_array().unwrap();
        assert_eq!(pairs.len(), 10);
        assert_eq!(row["pairs_not_shown"], json!(35));
        let mags: Vec<f64> = pairs
            .iter()
            .map(|p| p["interaction"].as_f64().unwrap().abs())
            .collect();
        assert!(mags.windows(2).all(|w| w[0] >= w[1]), "{mags:?}");
        let f = &pairs[0]["features"];
        assert!(
            f[0].as_str().unwrap().starts_with('x') && f[0] != f[1],
            "{f}"
        );
    }
}

/// viz_boxplot lists at most 20 outliers per column and counts the rest, so a heavy-tailed
/// column does not crowd the other columns out of the reply. Before: every outlier was listed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_boxplot_caps_the_listed_outliers() {
    let mut csv = vec!["x,y".to_string()];
    for i in 0..130 {
        let x = if i < 100 {
            50.0 + ((i * 37) % 23) as f64 * 0.1
        } else {
            1000.0 + (i - 100) as f64 * 3.7
        };
        csv.push(format!("{x},{}", 1.0 + ((i * 7) % 11) as f64 * 0.2));
    }
    let (s, server) = session_with_datasets(vec![]);
    let r = call(
        &server,
        &s,
        "create_dataset",
        json!({"name": "d", "csv_content": csv.join("\n")}),
    )
    .await;
    assert!(r.success, "{}", text(&r));
    let r = call(
        &server,
        &s,
        "viz_boxplot",
        json!({"dataset": "d", "columns": ["x", "y"]}),
    )
    .await;
    let t = text(&r);
    assert!(r.success, "{t}");
    let line = t
        .lines()
        .skip_while(|l| !l.starts_with("x (n = 130)"))
        .find(|l| l.trim_start().starts_with("Outliers:"))
        .unwrap_or_else(|| panic!("no outlier line for x:\n{t}"));
    assert!(line.ends_with(", and 10 more"), "{line}");
    let listed = line.trim_start().trim_start_matches("Outliers: ");
    let listed = listed.trim_end_matches(", and 10 more");
    assert_eq!(listed.split(", ").count(), 20, "{line}");
    assert!(t.contains("y (n = 130):"), "{t}");
}
