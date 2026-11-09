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

// ----------------------------------------------------------------------------
// Seeds
// ----------------------------------------------------------------------------

/// After set_seed(7), an unseeded probit equals the seed-7 run and differs from the seed-8 run.
/// Before: the handlers passed request.seed only.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_spatial_probit_global_seed() {
    let (s, server) = probit_session().await;
    let a = json!({"dataset": "spatial_sim", "y": "yb", "x": ["x1"], "weights": "w4", "n_draws": 60, "burn_in": 20});
    for tool in ["sar_probit_model", "sem_probit_model"] {
        let d = ok(&call(&server, &s, tool, a.clone()).await);
        assert_eq!(line(&d, "Seed:").trim(), "Seed: 42 (default)", "{tool}");
    }
    ok(&call(&server, &s, "set_seed", json!({"seed": 7})).await);
    for tool in ["sar_probit_model", "sem_probit_model"] {
        let u = ok(&call(&server, &s, tool, a.clone()).await);
        let s7 = ok(&call(&server, &s, tool, with(&a, json!({"seed": 7}))).await);
        let s8 = ok(&call(&server, &s, tool, with(&a, json!({"seed": 8}))).await);
        let strip = |t: &str| {
            t.lines()
                .filter(|l| !l.starts_with("Seed:"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        assert_eq!(
            strip(&u),
            strip(&s7),
            "{tool}: unseeded after set_seed(7) is seed 7"
        );
        assert_ne!(strip(&s7), strip(&s8), "{tool}: the seed matters");
        assert_eq!(line(&u, "Seed:").trim(), "Seed: 7 (set_seed)", "{tool}");
        assert_eq!(line(&s8, "Seed:").trim(), "Seed: 8 (request)", "{tool}");
    }
}

/// Request seed > set_seed > documented default, reported. ml_mboost's CV folds follow the
/// global seed; heterogeneity_test without any seed uses 42 (before: entropy in the core).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_ml_seeds_follow_request_then_set_seed_then_default() {
    let (s, server) = session_with("ml_mboost_bols");
    let mb = json!({"dataset": "mboost_sim", "y_col": "yg", "x_cols": ["x1", "x2", "x3"], "mstop": 30, "cv_folds": 5});
    let ht = json!({"dataset": "mboost_sim", "outcome": "yg", "treatment": "yb", "covariates": ["x1"], "n_permutations": 200});
    let seed_of = |t: &str| json_block(t)["seed"].clone();
    let unseeded_ht = ok(&call(&server, &s, "heterogeneity_test", ht.clone()).await);
    assert_eq!(seed_of(&unseeded_ht), json!(42));
    assert_eq!(
        unseeded_ht,
        ok(&call(
            &server,
            &s,
            "heterogeneity_test",
            with(&ht, json!({"seed": 42}))
        )
        .await)
    );
    assert_eq!(
        seed_of(&ok(&call(&server, &s, "ml_mboost", mb.clone()).await)),
        json!(42)
    );
    ok(&call(&server, &s, "set_seed", json!({"seed": 7})).await);
    let unseeded = ok(&call(&server, &s, "ml_mboost", mb.clone()).await);
    assert_eq!(seed_of(&unseeded), json!(7));
    assert_eq!(
        unseeded,
        ok(&call(&server, &s, "ml_mboost", with(&mb, json!({"seed": 7}))).await)
    );
    assert_eq!(
        seed_of(&ok(
            &call(&server, &s, "heterogeneity_test", ht.clone()).await
        )),
        json!(7)
    );
    assert_eq!(
        seed_of(&ok(&call(
            &server,
            &s,
            "heterogeneity_test",
            with(&ht, json!({"seed": 3}))
        )
        .await)),
        json!(3)
    );
}

// ----------------------------------------------------------------------------
// output_column
// ----------------------------------------------------------------------------

/// R kmeans(algorithm = "Lloyd") optimum on hclust_60: tot.withinss and the partition
/// km_unit_cluster (the global optimum). Before: output_column was an unknown field.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_kmeans_output_column() {
    let (s, server) = session_with("kmeans_lloyd_pca_mds");
    let case = Case::load("kmeans_lloyd_pca_mds");
    let t = ok(&call(
        &server,
        &s,
        "ml_kmeans",
        json!({"dataset": "hclust_60", "columns": ["a", "b", "c"], "k": 3, "seed": 1, "output_column": "km"}),
    )
    .await);
    let col: Vec<i64> = int_column(&s, "hclust_60", "km")
        .into_iter()
        .map(|v| v.expect("no nulls"))
        .collect();
    assert!(col.iter().all(|&c| (1..=3).contains(&c)), "{col:?}");
    assert!(
        same_partition(&col, &case.vector("km_unit_cluster")),
        "{col:?}"
    );
    let j = json_block(&t);
    close(
        j["inertia"].as_f64().unwrap(),
        case.scalar("km_unit_tot_withinss"),
        Tol::coef(60),
        "tot.withinss",
    );
    assert_eq!(
        f64s(&j["cluster_assignments"]),
        col.iter().map(|&c| c as f64).collect::<Vec<_>>()
    );
    assert_eq!(
        (
            j["output_column"].clone(),
            j["converged"].clone(),
            j["seed"].clone()
        ),
        (json!("km"), json!(true), json!(1))
    );
    assert!(
        t.contains("Cluster 1: 20 points")
            && !t.contains("Cluster 0:")
            && t.contains("Converged: yes"),
        "{t}"
    );
    assert!(t.contains("Wrote column 'km'"), "{t}");
    // no seed -> 42 (the core used entropy), then the set_seed value
    let unseeded = json!({"dataset": "hclust_60", "columns": ["a", "b", "c"], "k": 3});
    assert_eq!(
        json_block(&ok(&call(&server, &s, "ml_kmeans", unseeded.clone()).await))["seed"],
        json!(42)
    );
    ok(&call(&server, &s, "set_seed", json!({"seed": 7})).await);
    assert_eq!(
        json_block(&ok(&call(&server, &s, "ml_kmeans", unseeded).await))["seed"],
        json!(7)
    );
}

/// R cutree(hclust(dist(X), method), 3): the column is R's labels exactly.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_hierarchical_output_column() {
    let (s, server) = session_with("hclust_methods");
    let case = Case::load("hclust_methods");
    for (linkage, key, r_name) in [
        ("ward", "wardD2", "ward.D2"),
        ("mcquitty", "mcquitty", "mcquitty"),
        ("average", "average", "average"),
    ] {
        let t = ok(&call(
            &server,
            &s,
            "ml_hierarchical",
            json!({"dataset": "hclust_60", "columns": ["a", "b", "c"], "n_clusters": 3, "linkage": linkage, "output_column": "hc"}),
        )
        .await);
        let col: Vec<f64> = int_column(&s, "hclust_60", "hc")
            .into_iter()
            .map(|v| v.unwrap() as f64)
            .collect();
        assert_eq!(col, case.vector(&format!("{key}_cutree3")), "{linkage}");
        let j = json_block(&t);
        assert_eq!(f64s(&j["cluster_assignments"]), col);
        assert_eq!(j["linkage"], json!(r_name));
        let text_part = t.split("JSON Summary:").next().unwrap();
        let merge_rows = text_part
            .lines()
            .filter(|l| l.trim_start().starts_with(|c: char| c.is_ascii_digit()))
            .count();
        assert_eq!(merge_rows, 50, "the last 50 of 59 merges:\n{text_part}");
        assert!(
            text_part.contains("the last 50 of 59 merges"),
            "{text_part}"
        );
    }
}

/// prcomp(X)$x, one sign per component; explained variance = sdev^2.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_pca_output_column() {
    let (s, server) = session_with("mcp_pca_scores");
    let case = Case::load("mcp_pca_scores");
    let t = ok(&call(
        &server,
        &s,
        "ml_pca",
        json!({"dataset": "hclust_60", "columns": ["a", "b", "c"], "output_column": "PC"}),
    )
    .await);
    for j in 1..=3 {
        let got: Vec<f64> = float_column(&s, "hclust_60", &format!("PC{j}"))
            .into_iter()
            .map(|v| v.unwrap())
            .collect();
        let want = case.vector(&format!("pca_x_pc{j}"));
        let sign = (got[0] * want[0]).signum();
        for (g, w) in got.iter().zip(&want) {
            close(sign * g, *w, Tol::coef(60), &format!("PC{j}"));
        }
    }
    let ev = f64s(&json_block(&t)["explained_variance"]);
    assert_eq!(ev.len(), 3);
    for (e, sd) in ev.iter().zip(case.vector("pca_sdev")) {
        close(*e, sd * sd, Tol::coef(60), "variance");
    }
    let t2 = ok(&call(
        &server,
        &s,
        "ml_pca",
        json!({"dataset": "hclust_60", "columns": ["a", "b", "c"], "n_components": 2, "output_column": "Q"}),
    )
    .await);
    assert_eq!(json_block(&t2)["output_columns"], json!(["Q1", "Q2"]));
}

/// No R value for the regression forest (RNG): the column equals the core's in-sample
/// predictions for the same call.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_random_forest_output_column() {
    let (s, server) = session_with("shap_tree_rpart");
    let (_, df) = case_df("shap_tree_rpart");
    let t = ok(&call(
        &server,
        &s,
        "ml_random_forest",
        json!({"dataset": "shap_tree_sim", "features": ["x0", "x1", "x2", "x3"], "target": "y",
               "n_trees": 20, "max_depth": 5, "seed": 3, "output_column": "rf"}),
    )
    .await);
    let xs: Vec<Vec<f64>> = (0..4).map(|j| col_f64(&df, &format!("x{j}"))).collect();
    let x = ndarray::Array2::from_shape_fn((80, 4), |(i, j)| xs[j][i]);
    let y = ndarray::Array1::from(col_f64(&df, "y"));
    let names: Vec<String> = (0..4).map(|j| format!("x{j}")).collect();
    let core = p2a_core::random_forest(
        x.view(),
        y.view(),
        Some(20),
        Some(5),
        None,
        None,
        Some(3),
        Some(names),
    )
    .unwrap();
    let col: Vec<f64> = float_column(&s, "shap_tree_sim", "rf")
        .into_iter()
        .map(|v| v.unwrap())
        .collect();
    assert_eq!(col, core.predictions);
    assert_eq!(
        f64s(&json_block(&t)["predictions_sample"]),
        core.predictions[..20].to_vec()
    );
}

/// SVM drops incomplete rows: their predictions are null; the rest equal the core's
/// predictions on the complete rows.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_svm_output_column_omit() {
    let (_, mut df) = case_df("svm_libsvm");
    let x1: Vec<Option<f64>> = col_f64(&df, "x1")
        .into_iter()
        .enumerate()
        .map(|(i, v)| if i == 5 { None } else { Some(v) })
        .collect();
    let full = df.clone();
    df.with_column(Series::new("x1".into(), x1)).unwrap();
    let (s, server) = session_with_datasets(vec![("svm_sim", dataset(df))]);
    let t = ok(&call(
        &server,
        &s,
        "ml_svm",
        json!({"dataset": "svm_sim", "features": ["x1", "x2"], "target": "cls_lin", "output_column": "pred"}),
    )
    .await);
    let col = int_column(&s, "svm_sim", "pred");
    assert_eq!(col.len(), 160);
    assert_eq!(col[5], None);
    let keep: Vec<usize> = (0..160).filter(|&i| i != 5).collect();
    let (x1, x2, cls) = (
        col_f64(&full, "x1"),
        col_f64(&full, "x2"),
        col_f64(&full, "cls_lin"),
    );
    let x =
        ndarray::Array2::from_shape_fn(
            (159, 2),
            |(r, j)| {
                if j == 0 { x1[keep[r]] } else { x2[keep[r]] }
            },
        );
    let y = ndarray::Array1::from_iter(keep.iter().map(|&i| cls[i]));
    let names = vec!["x1".to_string(), "x2".to_string()];
    let core = p2a_core::linear_svm(x.view(), y.view(), None, None, None, Some(names)).unwrap();
    let got: Vec<i64> = keep.iter().map(|&i| col[i].unwrap()).collect();
    assert_eq!(
        got,
        core.predictions
            .iter()
            .map(|&p| i64::from(p))
            .collect::<Vec<_>>()
    );
    let j = json_block(&t);
    assert_eq!(j["missing"]["n_dropped"], json!(1), "{j}");
    assert!(j["predictions_sample"][5].is_null(), "{j}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_output_column_refuses_input_columns() {
    let (s, server) = session_with("hclust_methods");
    let e = err(&call(
        &server,
        &s,
        "ml_kmeans",
        json!({"dataset": "hclust_60", "columns": ["a", "b"], "k": 2, "output_column": "a"}),
    )
    .await);
    assert!(
        e.contains("output_column 'a' is a column this call reads (a, b)"),
        "{e}"
    );
    assert!(
        s.datasets.try_read().unwrap()["hclust_60"]
            .df()
            .column("a")
            .unwrap()
            .f64()
            .is_ok()
    );
    let e = err(&call(
        &server,
        &s,
        "ml_random_forest",
        json!({"dataset": "hclust_60", "features": ["a"], "target": "c", "output_column": "c"}),
    )
    .await);
    assert!(e.contains("output_column 'c'"), "{e}");
    let e = err(&call(
        &server,
        &s,
        "ml_pca",
        json!({"dataset": "hclust_60", "columns": ["a", "b"], "output_column": " "}),
    )
    .await);
    assert!(
        e.contains("output_column must be a non-empty column name"),
        "{e}"
    );
    ok(&call(
        &server,
        &s,
        "ml_kmeans",
        json!({"dataset": "hclust_60", "columns": ["a", "b"], "k": 2, "output_column": "cl", "seed": 1}),
    )
    .await);
    let t = ok(&call(
        &server,
        &s,
        "ml_kmeans",
        json!({"dataset": "hclust_60", "columns": ["a", "b"], "k": 3, "output_column": "cl", "seed": 1}),
    )
    .await);
    assert!(t.contains("Replaced column 'cl'"), "{t}");
    let t = ok(&call(
        &server,
        &s,
        "ml_kmeans",
        json!({"dataset": "hclust_60", "columns": ["a", "b"], "k": 2, "seed": 1}),
    )
    .await);
    assert!(
        json_block(&t)["output_column"].is_null() && !t.contains("Wrote column"),
        "{t}"
    );
}

/// dbscan::dbscan(X, eps = 1, minPts = 5) on hclust_60: cluster sizes and noise points (no MCP
/// tool reaches HDBSCAN; ml_dbscan is pinned instead).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_dbscan() {
    let (s, server) = session_with("kmeans_lloyd_pca_mds");
    let case = Case::load("kmeans_lloyd_pca_mds");
    let r = case.vector("dbscan_eps1_minpts5_cluster");
    let count = |c: f64| r.iter().filter(|&&v| v == c).count() as f64;
    let t = ok(&call(
        &server,
        &s,
        "ml_dbscan",
        json!({"dataset": "hclust_60", "columns": ["a", "b", "c"], "eps": 1.0, "min_samples": 5}),
    )
    .await);
    assert_eq!(
        num_after(line(&t, "Number of clusters:"), "Number of clusters:"),
        3.0
    );
    assert_eq!(
        num_after(
            line(&t, "Number of noise points:"),
            "Number of noise points:"
        ),
        count(0.0)
    );
    let mut sizes: Vec<f64> = t
        .lines()
        .filter(|l| l.trim_start().starts_with("Cluster ") && l.ends_with("points"))
        .map(|l| num_after(l, ":"))
        .collect();
    sizes.sort_by(f64::total_cmp);
    let mut want = vec![count(1.0), count(2.0), count(3.0)];
    want.sort_by(f64::total_cmp);
    assert_eq!(sizes, want);
}

/// The descriptions say what each tool returns, and the ones corrected earlier stay correct.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_ml_descriptions_say_what_is_returned() {
    let d = tool_description;
    for n in [
        "ml_kmeans",
        "ml_hierarchical",
        "ml_random_forest",
        "ml_svm",
        "ml_pca",
    ] {
        assert!(d(n).contains("output_column"), "{n}: {}", d(n));
    }
    assert!(
        d("ml_kmeans").contains("1..k") && d("ml_hierarchical").contains("cutree"),
        "cluster numbering"
    );
    assert!(
        d("ml_dbscan").contains("no per-row labels")
            && d("ml_tsne").contains("first 5 embedded points")
    );
    assert!(d("ml_pca").contains("loadings") && d("ml_pca").contains("prcomp"));
    for sub in [
        "regression",
        "no classification mode",
        "ntree 500",
        "max(floor(p/3), 1)",
        "nodesize 5",
        "ceil(sqrt(p))",
        "max_depth 10",
        "in-sample predictions",
    ] {
        assert!(
            d("ml_random_forest").contains(sub),
            "ml_random_forest lacks {sub:?}"
        );
    }
    assert!(d("ml_cutree").contains("1-based") || d("ml_cutree").contains("R's cutree numbering"));
}

// ----------------------------------------------------------------------------
// SHAP
// ----------------------------------------------------------------------------

/// Interventional Shapley values of a forest by enumeration over all 2^M coalitions;
/// background = the rows given.
fn interventional_shapley(
    trees: &[p2a_core::ml::trees::DecisionTree],
    x: ndarray::ArrayView1<f64>,
    bg: ndarray::ArrayView2<f64>,
) -> (f64, Vec<f64>) {
    let m = x.len();
    let f = |z: ndarray::ArrayView1<f64>| {
        trees.iter().map(|t| t.predict_one(&z)).sum::<f64>() / trees.len() as f64
    };
    let v = |mask: usize| -> f64 {
        bg.outer_iter()
            .map(|b| {
                let z = ndarray::Array1::from_shape_fn(m, |j| {
                    if mask & (1 << j) != 0 { x[j] } else { b[j] }
                });
                f(z.view())
            })
            .sum::<f64>()
            / bg.nrows() as f64
    };
    let fact = |k: usize| (1..=k).product::<usize>() as f64;
    let phi = (0..m)
        .map(|j| {
            (0..1usize << m)
                .filter(|s| s & (1 << j) == 0)
                .map(|s| {
                    let k = s.count_ones() as usize;
                    fact(k) * fact(m - k - 1) / fact(m) * (v(s | (1 << j)) - v(s))
                })
                .sum()
        })
        .collect();
    (v(0), phi)
}

fn shap_rows(j: &Value) -> Vec<Vec<f64>> {
    j["shap_values_sample"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|c| c["shap_value"].as_f64().unwrap())
                .collect()
        })
        .collect()
}

fn shap_inputs() -> (ndarray::Array2<f64>, ndarray::Array1<f64>) {
    let (_, df) = case_df("shap_tree_rpart");
    let xs: Vec<Vec<f64>> = ["x0", "x1", "x2", "x3"]
        .iter()
        .map(|c| col_f64(&df, c))
        .collect();
    (
        ndarray::Array2::from_shape_fn((80, 4), |(i, j)| xs[j][i]),
        ndarray::Array1::from(col_f64(&df, "y")),
    )
}

fn shap_forest(n_trees: usize, max_depth: usize) -> p2a_core::ml::trees::RandomForestModel {
    let (x, y) = shap_inputs();
    let names: Vec<String> = ["x0", "x1", "x2", "x3"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    p2a_core::random_forest_with_trees(
        x.view(),
        y.view(),
        Some(n_trees),
        Some(max_depth),
        Some(2),
        Some("sqrt"),
        Some(3),
        Some(names),
    )
    .unwrap()
}

/// method = "kernel": exact Kernel SHAP (M = 4, every coalition) of the handler's forest over
/// its background (all 80 training rows) equals the interventional Shapley values by
/// enumeration. Before: "method" was an unknown field, and n_samples changed nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_shap_kernel_path() {
    let (s, server) = session_with("shap_tree_rpart");
    let t = ok(&call(
        &server,
        &s,
        "ml_shap_values",
        json!({"dataset": "shap_tree_sim", "features": ["x0", "x1", "x2", "x3"], "target": "y",
               "n_trees": 10, "max_depth": 4, "max_obs": 5, "seed": 3, "method": "Kernel"}),
    )
    .await);
    let j = json_block(&t);
    assert_eq!(
        (
            j["method"].clone(),
            j["feature_perturbation"].clone(),
            j["exact"].clone(),
            j["n_coalitions"].clone(),
            j["kernel_background_rows"].clone()
        ),
        (
            json!("kernel"),
            json!("interventional"),
            json!(true),
            json!(14),
            json!(80)
        )
    );
    assert_eq!(j["additivity_check"], json!(true));
    let (x, _) = shap_inputs();
    let model = shap_forest(10, 4);
    for (i, got) in shap_rows(&j).iter().enumerate() {
        let (base, phi) = interventional_shapley(&model.trees, x.row(i), x.view());
        close(
            j["base_value"].as_f64().unwrap(),
            base,
            Tol::new(0.0, 1e-10),
            "base value E f",
        );
        for (g, w) in got.iter().zip(&phi) {
            close(*g, *w, Tol::new(0.0, 1e-10), &format!("row {i}"));
        }
    }
}

/// compute_interactions (TreeSHAP): each row of the interaction matrix sums to the SHAP value.
/// Before: unknown field; interaction values were never computed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_shap_interactions_sum_to_shap_values() {
    let (s, server) = session_with("shap_tree_rpart");
    let t = ok(&call(
        &server,
        &s,
        "ml_shap_values",
        json!({"dataset": "shap_tree_sim", "features": ["x0", "x1", "x2", "x3"], "target": "y",
               "n_trees": 10, "max_depth": 4, "max_obs": 5, "seed": 3, "compute_interactions": true}),
    )
    .await);
    let j = json_block(&t);
    assert_eq!(
        (
            j["method"].clone(),
            j["feature_perturbation"].clone(),
            j["exact"].clone()
        ),
        (json!("tree"), json!("tree_path_dependent"), json!(true))
    );
    let phi = shap_rows(&j);
    let inter = j["interaction_values_sample"]
        .as_array()
        .expect("interaction_values_sample");
    assert_eq!(inter.len(), 3);
    for (r, mat) in inter.iter().enumerate() {
        let mat: Vec<Vec<f64>> = mat.as_array().unwrap().iter().map(f64s).collect();
        for i in 0..4 {
            close(
                mat[i].iter().sum::<f64>(),
                phi[r][i],
                Tol::new(0.0, 1e-10),
                &format!("row {r} feature {i}"),
            );
            for k in 0..4 {
                close(mat[i][k], mat[k][i], Tol::new(0.0, 1e-12), "symmetric");
            }
        }
    }
}

/// Combinations the core does not compute are refused; TreeSHAP stays the default; n_samples
/// with TreeSHAP warns; a sampled Kernel SHAP uses exactly n_samples coalitions and stays
/// additive.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_shap_options_refused_and_defaults() {
    let (s, server) = session_with("shap_tree_rpart");
    let base = json!({"dataset": "shap_tree_sim", "features": ["x0", "x1", "x2", "x3"], "target": "y",
                      "n_trees": 5, "max_depth": 3, "max_obs": 3, "seed": 3});
    for (extra, fragment) in [
        (
            json!({"feature_perturbation": "observational"}),
            "not implemented",
        ),
        (
            json!({"method": "kernel", "compute_interactions": true}),
            "TreeSHAP only",
        ),
        (
            json!({"method": "kernel", "feature_perturbation": "tree_path_dependent"}),
            "tree_path_dependent needs TreeSHAP",
        ),
        (
            json!({"feature_perturbation": "interventional"}),
            "Kernel path",
        ),
        (json!({"method": "deep"}), "unknown method 'deep'"),
    ] {
        let e = err(&call(&server, &s, "ml_shap_values", with(&base, extra.clone())).await);
        assert!(e.contains(fragment), "{extra}: {e}");
    }
    let t = ok(&call(
        &server,
        &s,
        "ml_shap_values",
        with(&base, json!({"n_samples": 50})),
    )
    .await);
    assert_eq!(json_block(&t)["method"], json!("tree"));
    assert!(t.contains("Warning:") && t.contains("n_samples"), "{t}");
    assert_eq!(
        ok(&call(&server, &s, "ml_shap_values", base.clone()).await),
        ok(&call(
            &server,
            &s,
            "ml_shap_values",
            with(&base, json!({"method": "tree"}))
        )
        .await)
    );
    let t = ok(&call(
        &server,
        &s,
        "ml_shap_values",
        with(&base, json!({"method": "kernel", "n_samples": 10})),
    )
    .await);
    let j = json_block(&t);
    assert_eq!(
        (
            j["exact"].clone(),
            j["n_coalitions"].clone(),
            j["seed"].clone()
        ),
        (json!(false), json!(10), json!(3)),
        "{j}"
    );
    let (x, _) = shap_inputs();
    let model = shap_forest(5, 3);
    let base_value = j["base_value"].as_f64().unwrap();
    for (i, phi) in shap_rows(&j).iter().enumerate() {
        let fx = model
            .trees
            .iter()
            .map(|t| t.predict_one(&x.row(i)))
            .sum::<f64>()
            / model.trees.len() as f64;
        close(
            phi.iter().sum::<f64>(),
            fx - base_value,
            Tol::new(0.0, 1e-10),
            &format!("row {i} additivity"),
        );
    }
}

// ----------------------------------------------------------------------------
// SAR/SEM notes, causal forest
// ----------------------------------------------------------------------------

/// spatialreg lagsarlm / Durbin / errorsarlm and impacts, written out (4-NN W).
/// Before: no "Converged:" line.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_mcp_sar_sdm_sem() {
    let (s, server) = session_with("spatial_sar_sdm_sem");
    let case = Case::load("spatial_sar_sdm_sem");
    neighbors(
        &server,
        &s,
        "spatial_sim",
        "w4",
        json!({"method": "knn", "k": 4}),
    )
    .await;
    let base = json!({"dataset": "spatial_sim", "y": "y", "x": ["x1", "x2"], "weights": "w4"});
    // The rho/lambda block is "<header>\n-----\nEstimate: ..": the first "Estimate:" line is
    // the spatial parameter.
    let sar = ok(&call(&server, &s, "sar_model", base.clone()).await);
    printed(
        num_after(line(&sar, "Estimate:"), "Estimate:"),
        case.scalar("sar_rho"),
        6,
        "SAR rho",
    );
    let se_obs = case.vector("sar_se_obs");
    for (i, n) in ["(Intercept)", "x1", "x2"].iter().enumerate() {
        let r = row(&sar, n);
        printed(r[0], case.vector("sar_beta")[i], 6, n);
        printed(r[1], se_obs[i], 6, &format!("SAR {n} SE"));
    }
    let sar_imp = sar
        .split("Spatial Impacts")
        .nth(1)
        .unwrap_or_else(|| panic!("no SAR impacts: {sar}"));
    for n in ["x1", "x2"] {
        printed(
            row(sar_imp, n)[0],
            case.named("sar_impact_direct", n),
            6,
            &format!("SAR {n} direct"),
        );
    }
    let sdm = ok(&call(
        &server,
        &s,
        "sar_model",
        with(&base, json!({"durbin": true})),
    )
    .await);
    printed(
        num_after(line(&sdm, "Estimate:"), "Estimate:"),
        case.scalar("sdm_rho"),
        6,
        "SDM rho",
    );
    let imp = sdm
        .split("Spatial Impacts")
        .nth(1)
        .unwrap_or_else(|| panic!("no SDM impacts: {sdm}"));
    for n in ["x1", "x2"] {
        let r = row(imp, n);
        printed(
            r[0],
            case.named("sdm_impact_direct", n),
            6,
            &format!("{n} direct"),
        );
        printed(
            r[1],
            case.named("sdm_impact_indirect", n),
            6,
            &format!("{n} indirect"),
        );
        printed(
            r[2],
            case.named("sdm_impact_total", n),
            6,
            &format!("{n} total"),
        );
    }
    let sem = ok(&call(&server, &s, "sem_model", base).await);
    printed(
        num_after(line(&sem, "Estimate:"), "Estimate:"),
        case.scalar("sem_lambda"),
        6,
        "SEM lambda",
    );
    for (i, n) in ["(Intercept)", "x1", "x2"].iter().enumerate() {
        let r = row(&sem, n);
        printed(r[0], case.vector("sem_beta")[i], 6, n);
        printed(r[1], case.vector("sem_beta_se")[i], 6, &format!("{n} SE"));
    }
    for t in [&sar, &sdm, &sem] {
        assert!(t.contains("Converged: yes"), "{t}");
    }
}

/// The description and JSON tell what the causal forest computes. Before: "bootstrap variance
/// estimation" and no se_method / warnings in the JSON.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mcp_causal_forest_description_and_notes() {
    let d = tool_description("ml_causal_forest");
    assert!(
        d.contains("AIPW") && d.contains("local centering") && !d.contains("bootstrap variance"),
        "{d}"
    );
    let (s, server) = session_with("ml_mboost_bols");
    let t = ok(&call(
        &server,
        &s,
        "ml_causal_forest",
        json!({"dataset": "mboost_sim", "outcome": "yg", "treatment": "yb", "covariates": ["x1", "x2"], "n_trees": 50, "seed": 2}),
    )
    .await);
    let j = json_block(&t);
    assert_eq!(j["se_method"], json!("AIPW scores"));
    assert!(j["warnings"].is_array(), "{j}");
    assert_eq!(j["w_hat_range"].as_array().map(Vec::len), Some(2), "{j}");
}
