//! T3-WP57 MCP-path tests (machine-learning and spatial surface). Owned by T3-WP57.
//! Every test drives AnalyticsServer::call_tool_with_session through the shared helpers and pins
//! the result to an R case (validation/methods/mcp_ml_spatial.json) or to a stated property.
use crate::common::*;
use p2a_core::polars::prelude::*;
use p2a_mcp::AnalyticsServer;
use p2a_mcp::session::Session;
use p2a_mcp::transport::http::ToolResult;
use p2a_validation::{Case, Tol};
use serde_json::{Value, json};

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

fn case_df(id: &str) -> (Case, DataFrame) {
    let case = Case::load(id);
    let df = p2a_core::DataLoader::load_csv(&case.dataset_path()).expect("case dataset loads");
    (case, df)
}

fn dataset(df: DataFrame) -> p2a_core::Dataset {
    p2a_core::Dataset::new(df)
}

/// Text of a successful call (panics with the message otherwise).
fn ok(r: &ToolResult) -> String {
    assert!(r.success, "tool error: {} {:?}", text(r), r.error);
    text(r)
}

/// Message of a tool error (panics on success).
fn err(r: &ToolResult) -> String {
    assert!(!r.success, "expected a tool error, got:\n{}", text(r));
    let t = text(r);
    if t.is_empty() {
        r.error.clone().unwrap_or_default()
    } else {
        t
    }
}

fn json_block(t: &str) -> Value {
    let j = t
        .split("JSON Summary:\n")
        .nth(1)
        .unwrap_or_else(|| panic!("no JSON Summary in:\n{t}"));
    serde_json::from_str(j).unwrap_or_else(|e| panic!("JSON Summary does not parse ({e}):\n{j}"))
}

/// The first line containing `needle`.
fn line<'a>(t: &'a str, needle: &str) -> &'a str {
    t.lines()
        .find(|l| l.contains(needle))
        .unwrap_or_else(|| panic!("no line with {needle:?} in:\n{t}"))
}

/// The line after the first line containing `header`.
fn line_after<'a>(t: &'a str, header: &str) -> &'a str {
    let mut it = t.lines().skip_while(|l| !l.contains(header));
    it.next()
        .unwrap_or_else(|| panic!("no {header:?} in:\n{t}"));
    it.next()
        .unwrap_or_else(|| panic!("nothing after {header:?} in:\n{t}"))
}

/// The number right after `key` in `l`.
fn num_after(l: &str, key: &str) -> f64 {
    let at = l
        .find(key)
        .unwrap_or_else(|| panic!("{key:?} not in {l:?}"))
        + key.len();
    let tok: String = l[at..]
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit() || matches!(c, '-' | '+' | '.' | 'e' | 'E'))
        .collect();
    tok.parse()
        .unwrap_or_else(|_| panic!("no number after {key:?} in {l:?}"))
}

/// The numbers after the name on the first line whose first token is `name` (significance
/// stars, parentheses and commas stripped).
fn row(t: &str, name: &str) -> Vec<f64> {
    let l = t
        .lines()
        .find(|l| l.split_whitespace().next() == Some(name))
        .unwrap_or_else(|| panic!("no row {name:?} in:\n{t}"));
    l.split_whitespace()
        .skip(1)
        .filter_map(|w| {
            w.trim_end_matches('*')
                .trim_matches(|c| c == '(' || c == ')' || c == ',')
                .parse()
                .ok()
        })
        .collect()
}

fn printed(actual: f64, r: f64, decimals: u32, what: &str) {
    assert!(
        Tol::printed(decimals).allows(actual, r),
        "{what}: printed {actual}, R {r}"
    );
}

fn close(actual: f64, r: f64, tol: Tol, what: &str) {
    assert!(tol.allows(actual, r), "{what}: {actual} vs R {r}");
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap_or_else(|| panic!("not an array: {v}"))
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn with(base: &Value, extra: Value) -> Value {
    let mut a = base.clone();
    for (k, v) in extra.as_object().unwrap() {
        a[k] = v.clone();
    }
    a
}

/// spatial_neighbors on `ds` with coordinates cx, cy and `extra` request fields; returns the text.
async fn neighbors(
    server: &AnalyticsServer,
    s: &Session,
    ds: &str,
    name: &str,
    extra: Value,
) -> String {
    let a = with(
        &json!({"dataset": ds, "x_coord": "cx", "y_coord": "cy", "weights_name": name}),
        extra,
    );
    ok(&call(server, s, "spatial_neighbors", a).await)
}

/// a and b describe the same partition (a bijection of labels).
fn same_partition(a: &[i64], b: &[f64]) -> bool {
    use std::collections::HashMap;
    let (mut ab, mut ba) = (HashMap::new(), HashMap::new());
    a.len() == b.len()
        && a.iter().zip(b).all(|(&x, &y)| {
            let y = y as i64;
            *ab.entry(x).or_insert(y) == y && *ba.entry(y).or_insert(x) == x
        })
}

fn int_column(s: &Session, ds: &str, col: &str) -> Vec<Option<i64>> {
    let map = s.datasets.try_read().expect("no lock held after the call");
    map[ds]
        .df()
        .column(col)
        .unwrap_or_else(|_| panic!("{ds} has no column {col}"))
        .i64()
        .expect("Int64")
        .into_iter()
        .collect()
}

fn float_column(s: &Session, ds: &str, col: &str) -> Vec<Option<f64>> {
    let map = s.datasets.try_read().expect("no lock held after the call");
    map[ds]
        .df()
        .column(col)
        .unwrap_or_else(|_| panic!("{ds} has no column {col}"))
        .f64()
        .expect("Float64")
        .into_iter()
        .collect()
}

/// The description of `field` in the input schema of `tool`.
fn field_description(tool: &str, field: &str) -> String {
    let tools = AnalyticsServer::new().router_tools();
    let t = tools
        .iter()
        .find(|t| t.name == tool)
        .unwrap_or_else(|| panic!("{tool}"));
    Value::Object((*t.input_schema).clone())["properties"][field]["description"]
        .as_str()
        .unwrap_or_else(|| panic!("{tool}.{field}"))
        .to_string()
}

fn tool_description(tool: &str) -> String {
    let tools = AnalyticsServer::new().router_tools();
    tools
        .iter()
        .find(|t| t.name == tool)
        .unwrap_or_else(|| panic!("{tool}"))
        .description
        .clone()
        .unwrap_or_default()
        .to_string()
}

/// Every name in `names` appears quoted in the description of `tool.field`.
fn lists_every_value(tool: &str, field: &str, names: Vec<&str>) {
    let d = field_description(tool, field);
    for n in names {
        assert!(
            d.contains(&format!("'{n}'")),
            "{tool}.{field} lacks '{n}': {d}"
        );
    }
}

/// Column `c` of `df` as f64 (complete case data).
fn col_f64(df: &DataFrame, c: &str) -> Vec<f64> {
    df.column(c)
        .unwrap()
        .cast(&DataType::Float64)
        .unwrap()
        .f64()
        .unwrap()
        .into_iter()
        .map(|v| v.expect("complete case data"))
        .collect()
}

// ----------------------------------------------------------------------------
// Spatial option strings (spdep's names)
// ----------------------------------------------------------------------------

/// spdep's S0 and Moran's I of x1 per nb2listw style on the band (0, 0.2] of spatial_sim.
/// Before: "S" and "minmax" ran row standardisation and "U" binary weights; the output said
/// "Style: RowStd".
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_spatial_styles() {
    let (s, server) = session_with("spatial_weights_moran");
    let case = Case::load("spatial_weights_moran");
    for (style, canon) in [
        ("W", "W"),
        ("B", "B"),
        ("C", "C"),
        ("U", "U"),
        ("minmax", "minmax"),
        ("S", "S"),
        ("row", "W"),
        ("binary", "B"),
        ("MinMax", "minmax"),
        ("s", "S"),
    ] {
        let t = neighbors(
            &server,
            &s,
            "spatial_sim",
            "wb",
            json!({"method": "distance", "d_min": 0.0, "d_max": 0.2, "style": style}),
        )
        .await;
        assert!(
            line(&t, "Style:")
                .trim_start()
                .starts_with(&format!("Style: {canon} (")),
            "{style}: {t}"
        );
        printed(
            num_after(line(&t, "Sum of weights (S0):"), "Sum of weights (S0):"),
            case.scalar(&format!("style_{canon}_sum")),
            6,
            &format!("{style} S0"),
        );
        let m = ok(&call(
            &server,
            &s,
            "moran_test",
            json!({"dataset": "spatial_sim", "variable": "x1", "weights": "wb"}),
        )
        .await);
        printed(
            num_after(line(&m, "Moran I statistic:"), "Moran I statistic:"),
            case.scalar(&format!("style_{canon}_moran_I")),
            6,
            &format!("{style} I"),
        );
    }
}

/// spdep style "U" is C / n: S0(U) = S0(C) / 80. Before: "U" was mapped to binary weights.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_style_u_is_c_over_n() {
    let (s, server) = session_with("spatial_weights_moran");
    let case = Case::load("spatial_weights_moran");
    let band = |st: &str| json!({"method": "distance", "d_min": 0.0, "d_max": 0.2, "style": st});
    let u = neighbors(&server, &s, "spatial_sim", "wu", band("U")).await;
    let c = neighbors(&server, &s, "spatial_sim", "wc", band("C")).await;
    let s0 = |t: &str| num_after(line(t, "Sum of weights (S0):"), "Sum of weights (S0):");
    printed(
        s0(&u),
        case.scalar("style_C_sum") / 80.0,
        6,
        "S0(U) = S0(C) / n",
    );
    printed(s0(&u), case.scalar("style_U_sum"), 6, "S0(U)");
    printed(s0(&c), case.scalar("style_C_sum"), 6, "S0(C)");
    assert!(!line(&u, "Style:").contains("B ("), "{u}");
}

/// R (4-NN W): p greater 0.5291689446820709, two-sided 0.9416621106358581, less = pnorm(z) =
/// 0.47083105531792907. Before: "twosided" and "Two-Sided" ran the greater test.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_moran_alternative() {
    let (s, server) = session_with("spatial_weights_moran");
    let case = Case::load("spatial_weights_moran");
    neighbors(
        &server,
        &s,
        "spatial_sim",
        "w4",
        json!({"method": "knn", "k": 4}),
    )
    .await;
    let p_less = 0.470_831_055_317_929_07;
    let two = case.named("moran", "p_two_sided");
    for (alt, canon, p) in [
        (None, "greater", case.named("moran", "p_rand")),
        (Some("two.sided"), "two.sided", two),
        (Some("twosided"), "two.sided", two),
        (Some("Two-Sided"), "two.sided", two),
        (Some("less"), "less", p_less),
        (Some("Negative"), "less", p_less),
    ] {
        let mut a = json!({"dataset": "spatial_sim", "variable": "x1", "weights": "w4"});
        if let Some(alt) = alt {
            a["alternative"] = json!(alt);
        }
        let t = ok(&call(&server, &s, "moran_test", a).await);
        assert_eq!(
            line(&t, "Alternative:").trim(),
            format!("Alternative: {canon}"),
            "{alt:?}"
        );
        printed(
            num_after(line(&t, "P-value:"), "P-value:"),
            p,
            6,
            &format!("{alt:?} p"),
        );
        for (label, key, d) in [
            ("Moran I statistic:", "I", 6),
            ("Expected I:", "E", 6),
            ("Variance:", "var_rand", 6),
            ("Z-score:", "z_rand", 4),
        ] {
            printed(
                num_after(line(&t, label), label),
                case.named("moran", key),
                d,
                key,
            );
        }
    }
}

/// R: the band (0, 0.12] leaves 4 islands; spdep zero.policy = TRUE, adjust.n = TRUE uses n = 76.
/// Before: no islands line.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_moran_islands() {
    let (s, server) = session_with("spatial_weights_moran");
    let case = Case::load("spatial_weights_moran");
    neighbors(
        &server,
        &s,
        "spatial_sim",
        "w12",
        json!({"method": "distance", "d_min": 0.0, "d_max": 0.12}),
    )
    .await;
    let m = ok(&call(
        &server,
        &s,
        "moran_test",
        json!({"dataset": "spatial_sim", "variable": "x1", "weights": "w12"}),
    )
    .await);
    let isl = line(&m, "Islands");
    assert_eq!(
        num_after(isl, "Islands (units without neighbours):"),
        case.scalar("n_islands"),
        "{isl}"
    );
    assert!(isl.contains("76 linked units"), "{isl}");
    for (label, key, d) in [
        ("Moran I statistic:", "I", 6),
        ("Expected I:", "E", 6),
        ("Variance:", "var_rand", 6),
        ("Z-score:", "z_rand", 4),
        ("P-value:", "p_rand", 6),
    ] {
        printed(
            num_after(line(&m, label), label),
            case.named("moran_islands", key),
            d,
            key,
        );
    }
}

/// Unknown option strings are tool errors that list the accepted values; R, CLI and old MCP
/// spellings still work.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_spatial_unknown_option_is_error() {
    let (s, server) = session_with("spatial_weights_moran");
    // before: "Unknown neighbor method 'KNN'"
    neighbors(
        &server,
        &s,
        "spatial_sim",
        "w4",
        json!({"method": "KNN", "k": 4}),
    )
    .await;
    let base = json!({"dataset": "spatial_sim", "x_coord": "cx", "y_coord": "cy"});
    let e = err(&call(
        &server,
        &s,
        "spatial_neighbors",
        with(&base, json!({"style": "Q"})),
    )
    .await);
    assert!(
        e.contains("unknown weight style 'Q'") && e.contains("W, B, C, U, minmax, S"),
        "{e}"
    );
    let e = err(&call(
        &server,
        &s,
        "spatial_neighbors",
        with(&base, json!({"method": "radius"})),
    )
    .await);
    assert!(
        e.contains("unknown method 'radius'") && e.contains("knn, distance, distance_longlat"),
        "{e}"
    );
    let e = err(&call(
        &server,
        &s,
        "moran_test",
        json!({"dataset": "spatial_sim", "variable": "x1", "weights": "w4", "alternative": "bogus"}),
    )
    .await);
    assert!(
        e.contains("unknown alternative 'bogus'") && e.contains("two.sided, greater, less"),
        "{e}"
    );
    for (tool, field, value, expect) in [
        ("sphet_model", "model", "durbin", "lag"),
        ("sphet_model", "se_type", "newey", "robust"),
        ("spatial_panel_ml", "spatial_error", "error", "kkp"),
        ("spatial_panel_ml", "effect", "tw", "twoways"),
        ("spatial_panel_ml", "model", "fd", "pooling"),
        ("spatial_panel_gmm", "method", "gmm", "ec2sls"),
    ] {
        let mut a = json!({"dataset": "spatial_sim", "y": "y", "x": ["x1"], "weights": "w4"});
        if tool != "sphet_model" {
            a["entity_col"] = json!("id");
            a["time_col"] = json!("id");
        }
        a[field] = json!(value);
        let e = err(&call(&server, &s, tool, a).await);
        assert!(
            e.contains(&format!("'{value}'")) && e.contains(expect),
            "{tool}.{field}: {e}"
        );
    }
}

/// Every option field's schema description lists every canonical value the parser accepts.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_spatial_option_descriptions_list_every_value() {
    lists_every_value(
        "spatial_neighbors",
        "style",
        p2a_core::WeightStyle::choices().names(),
    );
    lists_every_value(
        "spatial_neighbors",
        "method",
        vec!["knn", "distance", "distance_longlat"],
    );
    lists_every_value(
        "moran_test",
        "alternative",
        p2a_core::MoranAlternative::choices().names(),
    );
    lists_every_value(
        "sphet_model",
        "model",
        p2a_core::SphetModel::choices().names(),
    );
    lists_every_value(
        "sphet_model",
        "se_type",
        p2a_core::SphetSE::choices().names(),
    );
    lists_every_value(
        "sphet_model",
        "kernel",
        p2a_core::econometrics::sphet::SpatialHacKernel::choices().names(),
    );
    lists_every_value(
        "spatial_panel_ml",
        "model",
        p2a_core::SpatialPanelModel::choices().names(),
    );
    lists_every_value(
        "spatial_panel_ml",
        "effect",
        p2a_core::SpatialPanelEffect::choices().names(),
    );
    lists_every_value(
        "spatial_panel_ml",
        "spatial_error",
        p2a_core::SpatialErrorType::choices().names(),
    );
    lists_every_value(
        "spatial_panel_gmm",
        "method",
        p2a_core::SpgmMethod::choices().names(),
    );
}
