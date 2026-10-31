//! T2-WP22: the X-2 / X-8 reviewer repro through the MCP dispatcher (`create_dataset` from the
//! case CSV, as the reviewer did): empty cells are dropped as R drops NA, with a note, and integer
//! columns are numeric.
use crate::common::{call, session_with_datasets, text};
use p2a_mcp::AnalyticsServer;
use p2a_mcp::session::Session;
use p2a_validation::{Case, Check, Tol};
use serde_json::{Value, json};

/// The case `id` (the reviewer's), a session and a dispatcher; the case CSV is loaded as
/// dataset `na` through `create_dataset`.
async fn reviewer(id: &str) -> (Case, Session, AnalyticsServer) {
    let case = Case::load(id);
    let csv = std::fs::read_to_string(case.dataset_path()).unwrap();
    let (session, server) = session_with_datasets(vec![]);
    let r = call(
        &server,
        &session,
        "create_dataset",
        json!({"name": "na", "csv_content": csv}),
    )
    .await;
    assert!(r.success, "create_dataset: {:?} {}", r.error, text(&r));
    (case, session, server)
}

/// The text of a successful call of `tool` (panics with the tool's error otherwise).
async fn ok(server: &AnalyticsServer, session: &Session, tool: &str, args: Value) -> String {
    let r = call(server, session, tool, args).await;
    assert!(r.success, "{tool}: {:?} {}", r.error, text(&r));
    text(&r)
}

/// The number that follows the first `label` in `text` (spaces skipped).
fn after(text: &str, label: &str) -> f64 {
    let at = text
        .find(label)
        .unwrap_or_else(|| panic!("no `{label}` in:\n{text}"));
    let rest = text[at + label.len()..].trim_start();
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E')))
        .unwrap_or(rest.len());
    rest[..end]
        .parse()
        .unwrap_or_else(|e| panic!("`{label}` is not followed by a number ({e}):\n{text}"))
}

/// The numbers on the line whose first token is `label` (other tokens skipped).
fn row(text: &str, label: &str) -> Vec<f64> {
    let line = text
        .lines()
        .find(|l| l.split_whitespace().next() == Some(label))
        .unwrap_or_else(|| panic!("no `{label}` row in:\n{text}"));
    line.split_whitespace()
        .skip(1)
        .filter_map(|t| t.parse().ok())
        .collect()
}

/// The JSON object of a tool's text: all of it, or the part from the first line that opens an
/// object (`hypothesis_wilcoxon` prints its Display before the JSON).
fn json_of(text: &str) -> Value {
    let start = if text.starts_with('{') {
        0
    } else {
        text.find("\n{").map_or(0, |i| i + 1)
    };
    serde_json::from_str(&text[start..]).unwrap_or_else(|e| panic!("no JSON object ({e}):\n{text}"))
}

/// The rows a result dropped, from its `Missing values: <k> of <n> rows dropped ...` line; 0
/// without one.
fn dropped(text: &str) -> f64 {
    if text.contains("Missing values: ") {
        after(text, "Missing values: ")
    } else {
        0.0
    }
}

/// X-2: the reviewer's x and y each have one empty cell. The t-tests, cor.test and the one-way
/// ANOVA drop it as R does (they read it as 0 before Tier 2).
#[tokio::test(flavor = "multi_thread")]
async fn test_validate_mcp_x2_reviewer_missing_cells() {
    let (case, session, server) = reviewer("na_x2_x8_reviewer").await;
    let (s, v) = (&server, &session);
    let one = ok(
        s,
        v,
        "hypothesis_t_test",
        json!({"dataset": "na", "x": "x"}),
    )
    .await;
    let paired = ok(
        s,
        v,
        "hypothesis_t_test",
        json!({"dataset": "na", "x": "x", "y": "y", "paired": true}),
    )
    .await;
    let welch = ok(
        s,
        v,
        "hypothesis_t_test",
        json!({"dataset": "na", "x": "x", "y": "y"}),
    )
    .await;
    let cor: Value = serde_json::from_str(
        &ok(
            s,
            v,
            "hypothesis_cor_test",
            json!({"dataset": "na", "x": "x", "y": "y"}),
        )
        .await,
    )
    .unwrap();
    let aov = ok(
        s,
        v,
        "anova_one_way",
        json!({"dataset": "na", "response": "y", "factor": "g"}),
    )
    .await;
    let (b, w) = (row(&aov, "Between"), row(&aov, "Within")); // [SS, DF, MS, F, p], [SS, DF, MS]
    let rows = 8.0;
    let n = 8;
    Check::new(&case)
        .scalar("one_t", after(&one, "t ="), Tol::printed(6))
        .scalar("one_df", after(&one, "df ="), Tol::printed(2))
        .scalar("one_n", rows - dropped(&one), Tol::exact())
        .scalar("paired_t", after(&paired, "t ="), Tol::printed(6))
        .scalar("paired_df", after(&paired, "df ="), Tol::printed(2))
        .scalar("paired_n", rows - dropped(&paired), Tol::exact())
        .scalar("welch_t", after(&welch, "t ="), Tol::printed(6))
        .scalar("welch_df", after(&welch, "df ="), Tol::printed(2))
        .scalar("cor_r", cor["estimate"].as_f64().unwrap(), Tol::stat(n))
        .scalar("cor_t", cor["statistic"].as_f64().unwrap(), Tol::stat(n))
        .scalar("cor_df", cor["df"].as_f64().unwrap(), Tol::exact())
        .scalar("cor_p", cor["p_value"].as_f64().unwrap(), Tol::p_value(n))
        .scalar("paired_n", cor["n"].as_f64().unwrap(), Tol::exact())
        .scalar("anova_y_f", b[3], Tol::printed(4))
        .scalar("anova_y_p", b[4], Tol::printed(4))
        .vector("anova_y_df", &[b[1], w[1]], Tol::exact())
        .vector("anova_y_ss", &[b[0], w[0]], Tol::printed(4))
        .scalar("anova_y_n", after(&aov, "N ="), Tol::exact())
        .finish();
    assert!(
        one.contains("Missing values: 1 of 8 rows dropped because of missing values (x: 1)"),
        "{one}"
    );
    assert!(
        paired
            .contains("Missing values: 2 of 8 rows dropped because of missing values (x: 1, y: 1)"),
        "{paired}"
    );
    assert_eq!(cor["missing"]["n_dropped"].as_u64(), Some(2), "{cor}");
    assert!(
        aov.contains("Missing values: 1 of 8 rows dropped because of missing values (y: 1)"),
        "{aov}"
    );
}

/// X-8: the reviewer's i and the survival columns are whole numbers, which polars reads as
/// Int64. The t-test, the ANOVA, the Wilcoxon test, cor.test and Kaplan-Meier accept them (they
/// refused them before Tier 2).
#[tokio::test(flavor = "multi_thread")]
async fn test_validate_mcp_x8_reviewer_integer_columns() {
    let (case, session, server) = reviewer("na_x2_x8_reviewer").await;
    let (s, v) = (&server, &session);
    let ti = ok(
        s,
        v,
        "hypothesis_t_test",
        json!({"dataset": "na", "x": "i"}),
    )
    .await;
    let ai = ok(
        s,
        v,
        "anova_one_way",
        json!({"dataset": "na", "response": "i", "factor": "g"}),
    )
    .await;
    let wi = json_of(
        &ok(
            s,
            v,
            "hypothesis_wilcoxon",
            json!({"dataset": "na", "x": "i"}),
        )
        .await,
    );
    let ci: Value = serde_json::from_str(
        &ok(
            s,
            v,
            "hypothesis_cor_test",
            json!({"dataset": "na", "x": "i", "y": "y"}),
        )
        .await,
    )
    .unwrap();
    let km = ok(
        s,
        v,
        "kaplan_meier",
        json!({"dataset": "na", "time": "time", "event": "event"}),
    )
    .await;
    let (b, w) = (row(&ai, "Between"), row(&ai, "Within"));
    let n = 8;
    Check::new(&case)
        .scalar("int_t", after(&ti, "t ="), Tol::printed(6))
        .scalar("int_df", after(&ti, "df ="), Tol::printed(2))
        .scalar("anova_i_f", b[3], Tol::printed(4))
        .scalar("anova_i_p", b[4], Tol::printed(4))
        .vector("anova_i_df", &[b[1], w[1]], Tol::exact())
        .scalar(
            "wilcox_i_v",
            wi["statistic"].as_f64().unwrap(),
            Tol::exact(),
        )
        .scalar(
            "wilcox_i_p",
            wi["p_value"].as_f64().unwrap(),
            Tol::p_value(n),
        )
        .scalar("cor_iy_r", ci["estimate"].as_f64().unwrap(), Tol::stat(n))
        .scalar("cor_iy_df", ci["df"].as_f64().unwrap(), Tol::exact())
        .scalar("cor_iy_p", ci["p_value"].as_f64().unwrap(), Tol::p_value(n))
        .scalar("km_median", after(&km, "Median Survival:"), Tol::printed(3))
        .finish();
    assert!(
        !ti.contains("Missing values") && ai.contains("N = 8"),
        "{ti}\n{ai}"
    );
    assert_eq!(
        (wi["exact"].as_bool(), wi["n1"].as_u64()),
        (Some(true), Some(8)),
        "{wi}"
    );
    assert_eq!(
        (ci["n"].as_u64(), ci["missing"]["n_dropped"].as_u64()),
        (Some(7), Some(1)),
        "{ci}"
    );
}

/// The staggered_did schema says what `bootstrap = 0` does (analytic influence-function SEs)
/// and that 1-49 replicates are refused, and the IV tools name their first-stage statistic.
#[test]
fn test_mcp_staggered_did_bootstrap_description() {
    let (_session, server) = session_with_datasets(vec![]);
    let tools = server.list_tools();
    let tool = |name: &str| {
        tools
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("no tool {name}"))
    };
    let did = tool("staggered_did");
    let bootstrap = did.input_schema["properties"]["bootstrap"]["description"]
        .as_str()
        .unwrap_or_default();
    assert!(
        bootstrap.contains("0 = analytic influence-function standard errors")
            && bootstrap.contains("at least 50"),
        "{bootstrap}"
    );
    assert!(
        tool("iv_2sls")
            .description
            .contains("partial F of the excluded instruments (the HC1 Wald F with robust = true"),
        "{}",
        tool("iv_2sls").description
    );
    assert!(
        tool("iv_first_stage")
            .description
            .contains("partial F of the excluded instruments"),
        "{}",
        tool("iv_first_stage").description
    );
}
