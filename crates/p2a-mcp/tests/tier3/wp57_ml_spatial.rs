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

// ----------------------------------------------------------------------------
// ML option strings
// ----------------------------------------------------------------------------

/// R cutree(hclust(dist(X), method), 3) on hclust_60. Before: every linkage other than the
/// four lowercase names was refused.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_cutree_linkages() {
    let (s, server) = session_with("hclust_methods");
    let case = Case::load("hclust_methods");
    for (linkage, key) in [
        ("ward", "wardD2"),
        ("ward.D2", "wardD2"),
        ("WARD.D", "wardD"),
        ("mcquitty", "mcquitty"),
        ("weighted", "mcquitty"),
        ("centroid", "centroid"),
        ("median", "median"),
        ("UPGMA", "average"),
    ] {
        let t = ok(&call(
            &server,
            &s,
            "ml_cutree",
            json!({"dataset": "hclust_60", "columns": ["a", "b", "c"], "k": 3, "linkage": linkage}),
        )
        .await);
        let j: Value = serde_json::from_str(&t).expect("ml_cutree returns JSON");
        assert_eq!(
            f64s(&j["cluster_assignments"]),
            case.vector(&format!("{key}_cutree3")),
            "{linkage}"
        );
    }
}

/// stats::ppr through ml_ppr (nterms 2, max_terms 3). The smoother names parse as R's
/// sm.method. Before: "GCV spline" was refused. Tolerances are the core pin's: the
/// super-smoother runs agree to 1e-8, the spline runs only to the measured gap of the exact
/// penalty (deviation ML-34).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_ppr() {
    let (s, server) = session_with("ppr_methods");
    let case = Case::load("ppr_methods");
    let supsmu = Tol::new(1e-8, 1e-10);
    let spline = Tol::new(0.03, 0.02);
    for (extra, key, tol) in [
        (json!({}), "supsmu_b0", supsmu),
        (json!({"bass": 5.0}), "supsmu_b5", supsmu),
        (json!({"sm_method": "GCV spline"}), "gcvspline", spline),
        (json!({"sm_method": "gcvspline"}), "gcvspline", spline),
    ] {
        let a = with(
            &json!({"dataset": "ppr_sim", "features": ["x1", "x2", "x3"], "target": "y", "nterms": 2, "max_terms": 3}),
            extra,
        );
        let j: Value = serde_json::from_str(&ok(&call(&server, &s, "ml_ppr", a).await)).unwrap();
        let alpha: Vec<f64> = j["projection_directions"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(f64s)
            .collect();
        for direction in j["projection_directions"].as_array().unwrap() {
            let norm: f64 = f64s(direction).iter().map(|a| a * a).sum::<f64>().sqrt();
            close(norm, 1.0, Tol::new(0.0, 1e-12), "unit-norm alpha");
        }
        for (got, want, what) in [
            (alpha, case.vector(&format!("{key}_alpha")), "alpha"),
            (
                f64s(&j["ridge_coefficients"]),
                case.vector(&format!("{key}_beta")),
                "beta",
            ),
            (
                f64s(&j["gof_rss"]),
                case.vector(&format!("{key}_gofn")),
                "gofn",
            ),
            (
                f64s(&j["fitted_values_sample"]),
                case.vector(&format!("{key}_fitted"))[..20].to_vec(),
                "fitted",
            ),
        ] {
            assert_eq!(got.len(), want.len(), "{key} {what}");
            for (g, w) in got.iter().zip(&want) {
                close(*g, *w, tol, &format!("{key} {what}"));
            }
        }
    }
}

/// mboost bols paths (mstop 100, nu 0.1, centred) for the four families through ml_mboost.
/// Before: "binomial_glm" was refused ("Unknown family").
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_mboost_families() {
    let (s, server) = session_with("ml_mboost_bols");
    let case = Case::load("ml_mboost_bols");
    for (family, key, y) in [
        ("gaussian", "gaussian", "yg"),
        ("binomial", "binomial", "yb"),
        ("Binomial-GLM", "binomial_glm", "yb"),
        ("poisson", "poisson", "yc"),
    ] {
        let t = ok(&call(
            &server,
            &s,
            "ml_mboost",
            json!({"dataset": "mboost_sim", "y_col": y, "x_cols": ["x1", "x2", "x3"], "family": family, "mstop": 100, "nu": 0.1}),
        )
        .await);
        let j = json_block(&t);
        let coef = case.vector(&format!("{key}_m100_coef"));
        assert_eq!(f64s(&j["coefficients"]).len(), coef.len(), "{key}");
        for (g, w) in f64s(&j["coefficients"]).iter().zip(coef) {
            close(*g, w, Tol::coef(200), &format!("{key} coef"));
        }
        close(
            j["intercept"].as_f64().unwrap(),
            case.scalar(&format!("{key}_m100_intercept")),
            Tol::coef(200),
            &format!("{key} intercept"),
        );
        assert_eq!(
            f64s(&j["selection_frequency"]),
            case.vector(&format!("{key}_m100_sel")),
            "{key} selections"
        );
    }
}

/// ctree options are parsed by the core enums and echoed canonically; a request giving only
/// minbucket gets a valid minsplit. Before: the raw string was echoed and minbucket 12 alone was
/// refused by the minsplit >= 2 minbucket rule.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_ctree_options_canonical_and_minsplit_default() {
    let (s, server) = session_with("shap_tree_rpart");
    let base =
        json!({"dataset": "shap_tree_sim", "features": ["x0", "x1", "x2", "x3"], "target": "y"});
    let lower = json_block(&ok(&call(
        &server,
        &s,
        "ml_ctree",
        with(
            &base,
            json!({"testtype": "bonferroni", "teststat": "quadratic"}),
        ),
    )
    .await));
    let upper = json_block(&ok(&call(
        &server,
        &s,
        "ml_ctree",
        with(&base, json!({"testtype": "Bonferroni", "teststat": "Quad"})),
    )
    .await));
    assert_eq!(upper["config"]["testtype"], json!("bonferroni"));
    assert_eq!(upper["config"]["teststat"], json!("quadratic"));
    assert_eq!(upper["tree_structure"], lower["tree_structure"]);
    let mb = json_block(&ok(&call(
        &server,
        &s,
        "ml_ctree",
        with(&base, json!({"minbucket": 12})),
    )
    .await));
    assert_eq!(
        (
            mb["config"]["minsplit"].clone(),
            mb["config"]["minbucket"].clone()
        ),
        (json!(24), json!(12))
    );
    let e = err(&call(
        &server,
        &s,
        "ml_ctree",
        with(&base, json!({"testtype": "MonteCarlo"})),
    )
    .await);
    assert!(e.to_lowercase().contains("montecarlo"), "{e}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_ml_unknown_option_is_error() {
    let (s, server) = session_with("hclust_methods");
    let e = err(&call(
        &server,
        &s,
        "ml_hierarchical",
        json!({"dataset": "hclust_60", "columns": ["a", "b"], "n_clusters": 2, "linkage": "wpgm"}),
    )
    .await);
    assert!(
        e.contains("unknown linkage 'wpgm'") && e.contains("mcquitty"),
        "{e}"
    );
    let e = err(&call(
        &server,
        &s,
        "ml_ppr",
        json!({"dataset": "hclust_60", "features": ["a", "b"], "target": "c", "sm_method": "loess"}),
    )
    .await);
    assert!(e.contains("'loess'") && e.contains("gcvspline"), "{e}");
    let e = err(&call(
        &server,
        &s,
        "ml_mboost",
        json!({"dataset": "hclust_60", "y_col": "c", "x_cols": ["a"], "base_learner": "spline"}),
    )
    .await);
    assert!(e.contains("'spline'") && e.contains("tree"), "{e}");
    let e = err(&call(
        &server,
        &s,
        "ml_ctree",
        json!({"dataset": "hclust_60", "features": ["a"], "target": "c", "teststat": "bogus"}),
    )
    .await);
    assert!(e.contains("'bogus'"), "{e}");
    let mut ds = s.datasets.try_read().unwrap()["hclust_60"].clone();
    let tr: Vec<i64> = (0..60).map(|i| (i % 2) as i64).collect();
    ds.df_mut()
        .with_column(Series::new("tr".into(), tr))
        .unwrap();
    s.datasets.try_write().unwrap().insert("h2".into(), ds);
    for (field, expect) in [
        ("test_statistic", "variance, range, iqr, mad"),
        ("effect_method", "matching, regression, stratified"),
    ] {
        let mut a = json!({"dataset": "h2", "outcome": "c", "treatment": "tr", "covariates": ["a"], "n_permutations": 10});
        a[field] = json!("bogus");
        // before: the default ran
        let e = err(&call(&server, &s, "heterogeneity_test", a).await);
        assert!(e.contains("'bogus'") && e.contains(expect), "{field}: {e}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_ml_option_descriptions_list_every_value() {
    lists_every_value(
        "ml_hierarchical",
        "linkage",
        p2a_core::Linkage::choices().names(),
    );
    lists_every_value("ml_cutree", "linkage", p2a_core::Linkage::choices().names());
    lists_every_value(
        "ml_ppr",
        "sm_method",
        p2a_core::SmoothingMethod::choices().names(),
    );
    lists_every_value(
        "ml_mboost",
        "family",
        p2a_core::MboostFamily::choices().names(),
    );
    lists_every_value(
        "ml_mboost",
        "base_learner",
        p2a_core::MboostBaseLearner::choices().names(),
    );
    lists_every_value(
        "ml_ctree",
        "teststat",
        p2a_core::CtreeTestStat::choices().names(),
    );
    lists_every_value(
        "ml_ctree",
        "testtype",
        p2a_core::CtreeTestType::choices().names(),
    );
    lists_every_value(
        "heterogeneity_test",
        "test_statistic",
        vec!["variance", "range", "iqr", "mad"],
    );
    lists_every_value(
        "heterogeneity_test",
        "effect_method",
        vec!["matching", "regression", "stratified"],
    );
}

// ----------------------------------------------------------------------------
// Spatial panels
// ----------------------------------------------------------------------------

/// splm_sim with its unit-coordinate dataset (period 1, sorted by id: W row i = the i-th id)
/// and a 4-NN W.
async fn splm_session() -> (Case, Session, AnalyticsServer) {
    let (case, df) = case_df("splm_spml_spgm");
    let units = df
        .clone()
        .lazy()
        .filter(col("time").eq(lit(1)))
        .sort(["id"], SortMultipleOptions::default())
        .select([col("id"), col("cx"), col("cy")])
        .collect()
        .unwrap();
    let (s, server) = session_with_datasets(vec![
        ("splm_sim", dataset(df)),
        ("splm_units", dataset(units)),
    ]);
    neighbors(
        &server,
        &s,
        "splm_units",
        "w4",
        json!({"method": "knn", "k": 4}),
    )
    .await;
    (case, s, server)
}

fn panel(y: &str, extra: Value) -> Value {
    with(
        &json!({"dataset": "splm_sim", "y": y, "x": ["x1", "x2"], "entity_col": "id", "time_col": "time", "weights": "w4"}),
        extra,
    )
}

/// splm::spml's default spatial.error = "b": the default call is the within Baltagi error
/// model. Before: lag = false and spatial_error = None ran the plain within estimator.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_spatial_panel_ml_default_is_spatial() {
    let (case, s, server) = splm_session().await;
    let t = ok(&call(&server, &s, "spatial_panel_ml", panel("y_err", json!({}))).await);
    let lam = line_after(&t, "Spatial error coefficient (lambda)");
    printed(
        num_after(lam, "Estimate:"),
        case.scalar("individual_error_lambda"),
        6,
        "lambda",
    );
    let se = case.vector("individual_error_se");
    printed(num_after(lam, "SE:"), se[2], 6, "lambda SE");
    for (i, name) in ["x1", "x2"].iter().enumerate() {
        let r = row(&t, name);
        printed(r[0], case.vector("individual_error_coef")[i], 6, name);
        printed(r[1], se[i], 6, &format!("{name} SE"));
    }
    assert!(!t.contains("Spatial lag coefficient (rho)"), "{t}");
    let e = err(&call(
        &server,
        &s,
        "spatial_panel_ml",
        panel("y_err", json!({"lag": false, "spatial_error": "none"})),
    )
    .await);
    assert!(
        e.contains("no spatial parameter") && e.contains("panel_fixed_effects"),
        "{e}"
    );
}

/// The other spml surfaces: lag alone is SARAR with the default error (spml(lag = TRUE)), lag
/// with spatial_error "none" the lag model, model "random" the Baltagi random-effects error
/// model (phi, sigma_mu).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_spml() {
    let (case, s, server) = splm_session().await;
    let sarar = ok(&call(
        &server,
        &s,
        "spatial_panel_ml",
        panel("y_lag", json!({"lag": true})),
    )
    .await);
    let rho = line_after(&sarar, "Spatial lag coefficient (rho)");
    let lam = line_after(&sarar, "Spatial error coefficient (lambda)");
    printed(
        num_after(rho, "Estimate:"),
        case.scalar("individual_sarar_rho"),
        6,
        "sarar rho",
    );
    printed(
        num_after(lam, "Estimate:"),
        case.scalar("individual_sarar_lambda"),
        6,
        "sarar lambda",
    );
    printed(
        num_after(lam, "SE:"),
        case.vector("individual_sarar_se")[3],
        6,
        "sarar lambda SE",
    );
    let lag = ok(&call(
        &server,
        &s,
        "spatial_panel_ml",
        panel("y_lag", json!({"lag": true, "spatial_error": "none"})),
    )
    .await);
    let rho = line_after(&lag, "Spatial lag coefficient (rho)");
    printed(
        num_after(rho, "Estimate:"),
        case.scalar("individual_lag_rho"),
        6,
        "lag rho",
    );
    printed(
        num_after(rho, "SE:"),
        case.vector("individual_lag_se")[2],
        6,
        "lag rho SE",
    );
    printed(
        row(&lag, "x1")[0],
        case.vector("individual_lag_coef")[0],
        6,
        "lag x1",
    );
    assert!(!lag.contains("Spatial error coefficient (lambda)"), "{lag}");
    let re = ok(&call(
        &server,
        &s,
        "spatial_panel_ml",
        panel("y_re", json!({"model": "RE"})),
    )
    .await);
    printed(
        num_after(
            line_after(&re, "Spatial error coefficient (lambda)"),
            "Estimate:",
        ),
        case.scalar("random_b_lambda"),
        6,
        "re lambda",
    );
    printed(
        num_after(
            line(&re, "phi = sigma_mu^2/sigma^2:"),
            "phi = sigma_mu^2/sigma^2:",
        ),
        case.scalar("random_b_phi"),
        6,
        "phi",
    );
    printed(
        num_after(
            line(&re, "sigma_mu (individual):"),
            "sigma_mu (individual):",
        ),
        case.scalar("random_b_sigma2_mu").sqrt(),
        6,
        "sigma_mu",
    );
    for (i, name) in ["(Intercept)", "x1", "x2"].iter().enumerate() {
        printed(row(&re, name)[0], case.vector("random_b_coef")[i], 6, name);
    }
}

/// spgm (splm::spgm's defaults where p2a implements them): the default call is the KKP within
/// GM error model; lag = true with w2sls keeps the spatial error (spatial.error = TRUE) and
/// fits the within GS2SLS; spatial_error = false gives the within 2SLS lag model; the
/// random-effects methods default to the lag model and refuse the error model, naming w2sls.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_spgm() {
    let (case, s, server) = splm_session().await;
    let gm = ok(&call(&server, &s, "spatial_panel_gmm", panel("y_err", json!({}))).await);
    printed(
        num_after(line(&gm, "Spatial error coefficient (lambda"), "):"),
        case.scalar("spgm_within_error_rho"),
        6,
        "GM lambda",
    );
    for (i, name) in ["x1", "x2"].iter().enumerate() {
        let r = row(&gm, name);
        printed(r[0], case.vector("spgm_within_error_coef")[i], 6, name);
        printed(
            r[1],
            case.vector("spgm_within_error_se")[i],
            6,
            &format!("{name} SE"),
        );
    }
    let gs = ok(&call(
        &server,
        &s,
        "spatial_panel_gmm",
        panel("y_lag", json!({"lag": true})),
    )
    .await);
    let rho = line(&gs, "Spatial lag coefficient (rho):");
    printed(
        num_after(rho, "Spatial lag coefficient (rho):"),
        case.vector("gs2sls_coef")[2],
        6,
        "gs2sls rho",
    );
    printed(
        num_after(rho, "SE:"),
        case.vector("gs2sls_se")[2],
        6,
        "gs2sls rho SE",
    );
    printed(
        num_after(line(&gs, "Spatial error coefficient (lambda"), "):"),
        case.scalar("gs2sls_lambda"),
        6,
        "gs2sls lambda",
    );
    for (i, name) in ["x1", "x2"].iter().enumerate() {
        let r = row(&gs, name);
        printed(r[0], case.vector("gs2sls_coef")[i], 6, name);
        printed(r[1], case.vector("gs2sls_se")[i], 6, &format!("{name} SE"));
    }
    let lag = ok(&call(
        &server,
        &s,
        "spatial_panel_gmm",
        panel("y_lag", json!({"lag": true, "spatial_error": false})),
    )
    .await);
    let rho = line(&lag, "Spatial lag coefficient (rho):");
    printed(
        num_after(rho, "Spatial lag coefficient (rho):"),
        case.vector("w2sls_coef")[2],
        6,
        "w2sls rho",
    );
    printed(
        num_after(rho, "SE:"),
        case.vector("w2sls_se")[2],
        6,
        "w2sls rho SE",
    );
    printed(
        row(&lag, "x1")[0],
        case.vector("w2sls_coef")[0],
        6,
        "w2sls x1",
    );
    assert!(!lag.contains("Spatial error coefficient"), "{lag}");
    let g2 = ok(&call(
        &server,
        &s,
        "spatial_panel_gmm",
        panel("y_lag", json!({"lag": true, "method": "g2sls"})),
    )
    .await);
    assert!(
        g2.contains("Spatial lag coefficient (rho):") && !g2.contains("Spatial error coefficient"),
        "{g2}"
    );
    let e = err(&call(
        &server,
        &s,
        "spatial_panel_gmm",
        panel(
            "y_lag",
            json!({"lag": true, "spatial_error": true, "method": "g2sls"}),
        ),
    )
    .await);
    assert!(e.contains("w2sls"), "{e}");
}

// ----------------------------------------------------------------------------
// sphet
// ----------------------------------------------------------------------------

/// Kelejian-Prucha S2SLS with spatial HAC SEs on the coordinates (5-NN W, sphet's default
/// variable bandwidth and Epanechnikov kernel). Before: the description said the SEs do not
/// match sphet, and the text was built in the handler.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_sphet_hac() {
    let (s, server) = session_with("sphet_gmm_hac");
    let case = Case::load("sphet_gmm_hac");
    neighbors(
        &server,
        &s,
        "sphet_sim",
        "w5",
        json!({"method": "knn", "k": 5}),
    )
    .await;
    let base = json!({"dataset": "sphet_sim", "y": "y_sar", "x": ["x1", "x2"], "weights": "w5",
                      "model": "lag", "se_type": "hac", "coord_x": "cx", "coord_y": "cy"});
    for kernel in [None, Some("Epanechnikov")] {
        let mut a = base.clone();
        if let Some(k) = kernel {
            a["kernel"] = json!(k);
        }
        let t = ok(&call(&server, &s, "sphet_model", a).await);
        assert!(
            t.contains(
                "Spatial HAC (Kelejian-Prucha 2007; epanechnikov kernel, variable bandwidth"
            ),
            "{t}"
        );
        let se = case.vector("sar_se_hac_epa_var");
        let coef = case.vector("sar_coef");
        for (i, name) in ["(Intercept)", "x1", "x2"].iter().enumerate() {
            let r = row(&t, name);
            printed(r[0], coef[i], 6, name);
            printed(r[1], se[i], 6, &format!("{name} SE ({kernel:?})"));
        }
        let lam = line(&t, "Spatial lag (lambda):");
        printed(
            num_after(lam, "Spatial lag (lambda):"),
            coef[3],
            6,
            "lambda",
        );
        printed(num_after(lam, "SE:"), se[3], 6, "lambda SE");
    }
}

/// The MCP's het = true for the error model: KP (2010) GM rho, its SE and White SEs.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_sphet_sem_het() {
    let (s, server) = session_with("sphet_gmm_hac");
    let case = Case::load("sphet_gmm_hac");
    neighbors(
        &server,
        &s,
        "sphet_sim",
        "w5",
        json!({"method": "knn", "k": 5}),
    )
    .await;
    let t = ok(&call(
        &server,
        &s,
        "sphet_model",
        json!({"dataset": "sphet_sim", "y": "y_sem", "x": ["x1", "x2"], "weights": "w5", "model": "SEM"}),
    )
    .await);
    let rho = line(&t, "Spatial error (rho):");
    printed(
        num_after(rho, "Spatial error (rho):"),
        case.scalar("sem_het_rho"),
        6,
        "rho",
    );
    printed(
        num_after(rho, "SE:"),
        case.scalar("sem_het_rho_se"),
        6,
        "rho SE",
    );
    assert!(
        t.contains("GM moments for rho: Kelejian-Prucha (2010)"),
        "{t}"
    );
    for (i, name) in ["(Intercept)", "x1", "x2"].iter().enumerate() {
        let r = row(&t, name);
        printed(r[0], case.vector("sem_het_coef")[i], 6, name);
        printed(
            r[1],
            case.vector("sem_het_se_white")[i],
            6,
            &format!("{name} SE"),
        );
    }
}

/// The HAC fields need se_type = "hac" and each other; the removed lag bandwidth is refused;
/// an unknown kernel names the MCP field; sphet's seven kernels are reachable.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_sphet_hac_field_rules() {
    let (s, server) = session_with("sphet_gmm_hac");
    neighbors(
        &server,
        &s,
        "sphet_sim",
        "w5",
        json!({"method": "knn", "k": 5}),
    )
    .await;
    let base = json!({"dataset": "sphet_sim", "y": "y_sar", "x": ["x1", "x2"], "weights": "w5"});
    for (extra, fragment) in [
        (json!({"se_type": "hac"}), "coord_x and coord_y"),
        (json!({"se_type": "hac", "coord_x": "cx"}), "together"),
        (json!({"kernel": "parzen"}), "se_type = 'hac'"),
        (json!({"coord_x": "cx", "coord_y": "cy"}), "se_type = 'hac'"),
        (
            json!({"se_type": "hac", "coord_x": "cx", "coord_y": "cy", "kernel": "gaussian"}),
            "unknown kernel 'gaussian'",
        ),
        (
            json!({"se_type": "hac", "coord_x": "cx", "coord_y": "cy", "bandwidth": 4}),
            "bandwidth is not used",
        ),
    ] {
        let e = err(&call(&server, &s, "sphet_model", with(&base, extra.clone())).await);
        assert!(e.contains(fragment), "{extra}: {e}");
        assert!(
            !e.contains("hac_kernel"),
            "{extra}: names no MCP field: {e}"
        );
    }
    let t = ok(&call(
        &server,
        &s,
        "sphet_model",
        with(
            &base,
            json!({"se_type": "hac", "coord_x": "cx", "coord_y": "cy", "kernel": "parzen"}),
        ),
    )
    .await);
    assert!(t.contains("parzen kernel"), "{t}");
}

// ----------------------------------------------------------------------------
// Spatial probits
// ----------------------------------------------------------------------------

/// spatial_sim with a 0/1 outcome yb = 1(y > 1.9) and a 4-NN W.
async fn probit_session() -> (Session, AnalyticsServer) {
    let (_, mut df) = case_df("spatial_weights_moran");
    let yb: Vec<i64> = col_f64(&df, "y")
        .into_iter()
        .map(|v| i64::from(v > 1.9))
        .collect();
    df.with_column(Series::new("yb".into(), yb)).unwrap();
    let (s, server) = session_with_datasets(vec![("spatial_sim", dataset(df))]);
    neighbors(
        &server,
        &s,
        "spatial_sim",
        "w4",
        json!({"method": "knn", "k": 4}),
    )
    .await;
    (s, server)
}

/// Before: one "Std.Errors" row under the table that held only x1's SDs.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_sar_probit_impact_se_per_variable() {
    let (s, server) = probit_session().await;
    let t = ok(&call(
        &server,
        &s,
        "sar_probit_model",
        json!({"dataset": "spatial_sim", "y": "yb", "x": ["x1", "x2"], "weights": "w4",
               "n_draws": 200, "burn_in": 50, "seed": 3}),
    )
    .await);
    let block = t
        .split("Spatial impacts")
        .nth(1)
        .unwrap_or_else(|| panic!("no impacts in {t}"));
    assert!(!block.contains("Std.Errors"), "{block}");
    let (x1, x2) = (row(block, "x1"), row(block, "x2"));
    assert_eq!(
        (x1.len(), x2.len()),
        (6, 6),
        "Direct (SD) Indirect (SD) Total (SD) per row:\n{block}"
    );
    assert!(
        x1[1] > 0.0 && x2[1] > 0.0 && x1[1] != x2[1],
        "each variable has its own SDs:\n{block}"
    );
}

/// x = [] is an intercept-only model: with the default compute_impacts it is the core's error,
/// returned as a tool error; with compute_impacts = false it runs. Before the core refused it,
/// the handler indexed an empty vector.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_sar_probit_empty_x() {
    let (s, server) = probit_session().await;
    let a = json!({"dataset": "spatial_sim", "y": "yb", "x": [], "weights": "w4", "n_draws": 60, "burn_in": 20, "seed": 3});
    let e = err(&call(&server, &s, "sar_probit_model", a.clone()).await);
    assert!(
        e.contains("compute_impacts needs at least one regressor"),
        "{e}"
    );
    let t = ok(&call(
        &server,
        &s,
        "sar_probit_model",
        with(&a, json!({"compute_impacts": false})),
    )
    .await);
    assert!(
        row(&t, "(Intercept)").len() >= 2 && !t.contains("Spatial impacts"),
        "{t}"
    );
    ok(&call(&server, &s, "sem_probit_model", a).await);
}
