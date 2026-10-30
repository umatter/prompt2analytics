//! Shared helpers of the Tier 2 MCP-path tests (design 2.5, 4.1.5), and the scaffold's own
//! tests.
//!
//! A test builds a [`Session`] holding its datasets, then runs tools through
//! `AnalyticsServer::call_tool_with_session`, the dispatcher the HTTP API uses (request
//! limits, the response cap and the session's datasets included).

use std::sync::atomic::{AtomicUsize, Ordering};

use p2a_core::data::{DataLoader, Dataset};
use p2a_mcp::AnalyticsServer;
use p2a_mcp::session::Session;
use p2a_mcp::transport::http::{ContentItem, ToolResult};

/// Session ids of this test binary: unique without a clock or an RNG.
static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

/// A session holding the case's dataset under its file stem (e.g. "two_sample_na"), and a
/// dispatcher.
pub fn session_with(case_id: &str) -> (Session, AnalyticsServer) {
    let case = p2a_validation::Case::load(case_id);
    let path = case.dataset_path();
    let df = DataLoader::load_csv(&path)
        .unwrap_or_else(|e| panic!("case {case_id}: cannot load {}: {e}", path.display()));
    let stem = std::path::Path::new(&case.dataset)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_else(|| panic!("case {case_id}: dataset {} has no file stem", case.dataset))
        .to_string();
    session_with_datasets(vec![(stem.as_str(), Dataset::new(df))])
}

/// A session holding the given datasets (for Rust-built fixtures, design 2.2), and a
/// dispatcher.
///
/// The session's cached byte size is not refreshed: the default limits charge no session
/// quota. A test that installs `Limits::hosted()` calls `session.refresh_data_bytes().await`
/// first.
pub fn session_with_datasets(datasets: Vec<(&str, Dataset)>) -> (Session, AnalyticsServer) {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let session = Session::new(format!("tier2-{id}"), None);
    {
        let mut map = session.datasets.try_write().expect("fresh session");
        for (name, ds) in datasets {
            map.insert(name.to_string(), ds);
        }
    }
    let server = AnalyticsServer::with_session(&session);
    (session, server)
}

/// `server.call_tool_with_session(tool, args, session)`; panics with the tool name on a
/// dispatch error.
pub async fn call(
    server: &AnalyticsServer,
    session: &Session,
    tool: &str,
    args: serde_json::Value,
) -> ToolResult {
    match server.call_tool_with_session(tool, args, session).await {
        Ok(r) => r,
        Err(e) => panic!("{tool}: dispatch failed: {e:?}"),
    }
}

/// Concatenated text items of a result.
pub fn text(result: &ToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|c| match c {
            ContentItem::Text { text } => Some(text.as_str()),
            ContentItem::Image { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The `<k>` of a line "<column> (<dtype>): <k> nulls" in a dataset description.
fn nulls_of(text: &str, column: &str, dtype: &str) -> f64 {
    let needle = format!("{column} ({dtype}): ");
    let line = text
        .lines()
        .find(|l| l.trim_start().trim_start_matches("- ").starts_with(&needle))
        .unwrap_or_else(|| panic!("no line for {column} ({dtype}) in:\n{text}"));
    let rest = &line[line.find(&needle).unwrap() + needle.len()..];
    let count = rest
        .split_whitespace()
        .next()
        .unwrap_or_else(|| panic!("no count in {line:?}"));
    count
        .parse()
        .unwrap_or_else(|e| panic!("count {count:?} in {line:?}: {e}"))
}

#[tokio::test]
async fn test_validate_mcp_describe_dataset_columns_conventions() {
    use p2a_validation::{Check, Tol};
    let case = p2a_validation::Case::load("columns_conventions");
    let (session, server) = session_with("columns_conventions");
    let r = call(
        &server,
        &session,
        "describe_dataset",
        serde_json::json!({"dataset": "columns_conventions"}),
    )
    .await;
    let t = text(&r);
    assert!(r.success, "{t}");
    assert!(t.contains("10 rows x 5 columns"), "{t}");
    Check::new(&case)
        .scalar("missing_y", nulls_of(&t, "y", "f64"), Tol::exact())
        .scalar("missing_n", nulls_of(&t, "n", "i64"), Tol::exact())
        .scalar("missing_s", nulls_of(&t, "s", "str"), Tol::exact())
        .finish();
    assert!(t.contains("flag (bool): 1 nulls"), "{t}");
}

#[test]
fn test_tier2_api_stubs_are_reachable() {
    use p2a_core::EconResult;
    use p2a_core::data::columns::MissingSummary;
    use p2a_core::econometrics::{
        KaplanMeierResult, KmConfType, KmConfig, OrderedConfig, OrderedModelType, OrderedResult,
    };
    use p2a_core::polars::prelude::DataFrame;
    use p2a_core::stats::{
        ChiSquaredResult, ContingencyTable, GofExpected, SsType, TwoWayAnovaConfig,
        TwoWayAnovaResult,
    };

    let _: fn(&DataFrame, &str, &str) -> EconResult<ContingencyTable> =
        p2a_core::stats::contingency_table;
    let _: fn(&DataFrame, &str, &[String]) -> EconResult<(Vec<String>, Vec<f64>, MissingSummary)> =
        p2a_core::stats::level_counts;
    let _: fn(
        &DataFrame,
        &str,
        &str,
        &str,
    ) -> EconResult<(Vec<String>, Vec<ContingencyTable>, MissingSummary)> =
        p2a_core::stats::stratified_tables;
    let _: fn(&Dataset, &str, &GofExpected, bool) -> EconResult<ChiSquaredResult> =
        p2a_core::stats::run_chisq_gof_with;
    let _: fn(&Dataset, &str, &str, &str, &TwoWayAnovaConfig) -> EconResult<TwoWayAnovaResult> =
        p2a_core::stats::run_two_way_anova_with;
    let _: fn(&Dataset, &str, &str, Option<&str>, &KmConfig) -> EconResult<Vec<KaplanMeierResult>> =
        p2a_core::econometrics::run_kaplan_meier_with;
    let _: fn(&Dataset, &str, &[&str], &OrderedConfig) -> EconResult<OrderedResult> =
        p2a_core::econometrics::run_ordered_model_with;

    assert_eq!(
        TwoWayAnovaConfig::default(),
        TwoWayAnovaConfig {
            interaction: true,
            ss_type: SsType::I
        }
    );
    assert_eq!(GofExpected::default(), GofExpected::Uniform);
    assert_eq!(KmConfig::default().conf_type, KmConfType::Log);
    assert_eq!(KmConfig::default().conf_level, 0.95);
    assert_eq!("log-log".parse::<KmConfType>().unwrap(), KmConfType::LogLog);
    assert_eq!(
        OrderedConfig::default(),
        OrderedConfig {
            link: OrderedModelType::Logit,
            levels: None,
            max_iter: 100,
            tol: 1e-10
        }
    );
}
