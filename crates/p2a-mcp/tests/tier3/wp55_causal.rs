//! T3-WP55 MCP-path tests (causal surface). Owned by T3-WP55.
//! Each test drives `call_tool_with_session`. An R pin computes the library result for the configuration the
//! request must map to, checks it against the case (Tier 1 tolerances), and requires the tool text to be that
//! result's text.
#![allow(dead_code, unused_imports)]
use crate::common::*;
use ndarray::{Array1, Array2};
use p2a_core::data::columns::{NaPolicy, numeric_column};
use p2a_core::data::{DataLoader, Dataset};
use p2a_mcp::AnalyticsServer;
use p2a_mcp::session::Session;
use p2a_validation::{Case, Check, Tol};
use serde_json::{Value, json};

fn case_ds(case: &Case) -> Dataset {
    Dataset::new(DataLoader::load_csv(case.dataset_path()).expect("case dataset"))
}
fn col(ds: &Dataset, name: &str) -> Array1<f64> {
    Array1::from(numeric_column(ds.df(), name, NaPolicy::Fail).unwrap().0)
}
async fn ok(server: &AnalyticsServer, session: &Session, tool: &str, args: Value) -> String {
    let r = call(server, session, tool, args).await;
    let t = text(&r);
    assert!(r.success, "{tool} failed: {t}");
    t
}
async fn refused(server: &AnalyticsServer, session: &Session, tool: &str, args: Value) -> String {
    let r = call(server, session, tool, args).await;
    let t = text(&r);
    assert!(!r.success, "{tool} should be refused, got: {t}");
    t
}
fn with(base: &Value, pairs: &[(&str, Value)]) -> Value {
    let mut a = base.clone();
    for (k, v) in pairs {
        a[*k] = v.clone();
    }
    a
}
