//! Interactive charts: Plotly figures written as HTML.
//!
//! The figure (traces and layout) is written as Plotly JSON from the f64 data, so values keep
//! double precision, and every [`InteractiveConfig`] option is applied. The HTML loads
//! plotly.js 3.0.1 and MathJax from cdn.plot.ly and cdn.jsdelivr.net when it is opened: viewing
//! a chart needs network access (docs/guides/DATA_SECURITY.md). Building one does not.
//!
//! Rows with a missing x or y value (null, or NaN in a float column) are left out and counted in
//! [`InteractivePlotResult::missing`]; infinite values cannot be drawn and are left out too, but
//! they are not missing values. A `Date` or `Datetime` x is written as ISO text. Groups follow
//! R's `factor()` order (numbers numerically, text in C-locale order), with a missing label
//! shown as the group "NA", last (ggplot2's convention).

use super::VisualizationError;
use super::colors::PLOTLY_PALETTE;
use crate::data::columns::{ColumnRequest, MissingSummary, factor_options};
use crate::resource;
use polars::prelude::*;
use serde::Serialize;
use serde_json::{Map, Value, json};

/// Configuration for interactive charts. Every field is applied when set.
#[derive(Debug, Clone, Default)]
pub struct InteractiveConfig {
    /// Figure width in pixels (`layout.width`); unset: the page width.
    pub width: Option<u32>,
    /// Figure height in pixels (`layout.height`); unset: Plotly's default.
    pub height: Option<u32>,
    /// Chart title.
    pub title: Option<String>,
    /// x-axis title.
    pub x_label: Option<String>,
    /// y-axis title.
    pub y_label: Option<String>,
    /// Legend title.
    pub legend_title: Option<String>,
    /// Opacity of every trace, 0 to 1.
    pub opacity: Option<f64>,
    /// Trace colours, cycled over the traces (default: the brand palette).
    pub colors: Option<Vec<(u8, u8, u8)>>,
}

/// Result of interactive plot generation.
#[derive(Debug, Clone)]
pub struct InteractivePlotResult {
    /// Full HTML page that draws the figure with plotly.js (loaded from its CDN when opened)
    pub html: String,
    /// Number of points plotted (rows with a missing or infinite plotted value are not)
    pub n_points: usize,
    /// Chart type
    pub chart_type: String,
    /// Rows left out because a plotted value is missing
    pub missing: MissingSummary,
    /// Notes on the drawing: rows left out because a plotted value is infinite (ggplot2's
    /// "Removed k rows containing non-finite values").
    pub warnings: Vec<String>,
}

impl std::fmt::Display for InteractivePlotResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Interactive {} Plot", self.chart_type)?;
        writeln!(f, "====================")?;
        writeln!(f, "Points: {}", self.n_points)?;
        for w in &self.warnings {
            writeln!(f, "Warning: {w}")?;
        }
        if let Some(note) = self.missing.note() {
            writeln!(f, "Missing values: {note}")?;
        }
        writeln!(f)?;
        writeln!(f, "HTML output: {} bytes", self.html.len())
    }
}

fn invalid(e: impl std::fmt::Display) -> VisualizationError {
    VisualizationError::InvalidData(e.to_string())
}

/// One plotted cell.
enum Cell {
    /// Null, or NaN in a float column.
    Missing,
    /// ±Inf: not missing, but JSON cannot hold it and Plotly cannot draw it.
    Infinite,
    /// A finite number, or a date as ISO text.
    Value(Datum),
}

/// A plotted value as it is written into the figure JSON.
#[derive(Clone, Serialize)]
#[serde(untagged)]
enum Datum {
    /// A finite double, written in the shortest digits that read back as the same double.
    Num(f64),
    /// A date or date-time as ISO text.
    Text(String),
}

/// One trace: its settings, then the plotted values (serialized directly, without building a
/// JSON value per point).
#[derive(Serialize)]
struct Trace {
    #[serde(flatten)]
    settings: Map<String, Value>,
    x: Vec<Datum>,
    #[serde(skip_serializing_if = "Option::is_none")]
    y: Option<Vec<Datum>>,
}

/// The Plotly figure.
#[derive(Serialize)]
struct Figure {
    data: Vec<Trace>,
    layout: Map<String, Value>,
}

/// `json!({...})` as a map (the trace settings).
fn settings(v: Value) -> Map<String, Value> {
    match v {
        Value::Object(m) => m,
        _ => Map::new(),
    }
}

/// The cells of a plotted column: a `Date` or `Datetime` column as ISO text, anything else
/// through the numeric rule of `data::columns` (integers, floats, Decimal, Boolean as 0/1).
fn column_cells(df: &DataFrame, name: &str) -> Result<Vec<Cell>, VisualizationError> {
    let n = df.height();
    let col = crate::data::columns::column(df, name).map_err(invalid)?;
    let mut cells: Vec<Cell> = resource::try_with_capacity(n, "chart values").map_err(invalid)?;
    if matches!(col.dtype(), DataType::Date | DataType::Datetime(..)) {
        let text = col.cast(&DataType::String).map_err(invalid)?;
        for v in text.str().map_err(invalid)?.iter() {
            cells.push(match v {
                Some(s) => Cell::Value(Datum::Text(s.to_string())),
                None => Cell::Missing,
            });
        }
        return Ok(cells);
    }
    let cols = ColumnRequest::new(df)
        .numeric(name)
        .allow_infinite(true)
        .min_rows(0, name)
        .extract()
        .map_err(invalid)?;
    cells.extend((0..n).map(|_| Cell::Missing));
    for (&row, &v) in cols.rows().iter().zip(cols.numeric(name)) {
        cells[row] = if v.is_finite() {
            Cell::Value(Datum::Num(v))
        } else {
            Cell::Infinite
        };
    }
    Ok(cells)
}

/// The missing-value summary of rows dropped because some plotted column is missing.
fn missing_summary(columns: &[(&str, &[Cell])], n: usize) -> MissingSummary {
    let mut by_column: Vec<(String, usize)> = Vec::new();
    for (name, cells) in columns {
        if by_column.iter().any(|(c, _)| c == name) {
            continue;
        }
        let count = cells.iter().filter(|c| matches!(c, Cell::Missing)).count();
        if count > 0 {
            by_column.push((name.to_string(), count));
        }
    }
    let n_dropped = (0..n)
        .filter(|&i| {
            columns
                .iter()
                .any(|(_, cells)| matches!(cells[i], Cell::Missing))
        })
        .count();
    MissingSummary {
        n_total: n,
        n_used: n - n_dropped,
        n_dropped,
        by_column,
    }
}

/// The trace index of every row, and the trace names: one trace without a group column;
/// otherwise one per level in R `factor()` order and "NA" last for missing labels.
fn trace_groups(
    df: &DataFrame,
    group_col: Option<&str>,
) -> Result<(Vec<usize>, Vec<String>), VisualizationError> {
    let n = df.height();
    let Some(gc) = group_col else {
        return Ok((vec![0; n], vec![String::new()]));
    };
    let (codes, mut labels) = factor_options(df, gc).map_err(invalid)?;
    let na = labels.len();
    let index: Vec<usize> = codes.iter().map(|c| c.map_or(na, |c| c as usize)).collect();
    if codes.iter().any(Option::is_none) {
        labels.push("NA".to_string());
    }
    Ok((index, labels))
}

/// `rgb(r, g, b)` of trace `i`, cycling over the configured colours.
fn trace_color(config: &InteractiveConfig, i: usize) -> String {
    let palette: &[(u8, u8, u8)] = match &config.colors {
        Some(c) if !c.is_empty() => c,
        _ => &PLOTLY_PALETTE,
    };
    let (r, g, b) = palette[i % palette.len()];
    format!("rgb({r}, {g}, {b})")
}

/// The figure's layout: each option that is set.
fn layout(config: &InteractiveConfig) -> Map<String, Value> {
    let mut layout = Map::new();
    let titled = |text: &str| json!({ "title": { "text": text } });
    if let Some(t) = &config.title {
        layout.insert("title".into(), json!({ "text": t }));
    }
    if let Some(x) = &config.x_label {
        layout.insert("xaxis".into(), titled(x));
    }
    if let Some(y) = &config.y_label {
        layout.insert("yaxis".into(), titled(y));
    }
    if let Some(l) = &config.legend_title {
        layout.insert("legend".into(), titled(l));
    }
    if let Some(w) = config.width {
        layout.insert("width".into(), json!(w));
    }
    if let Some(h) = config.height {
        layout.insert("height".into(), json!(h));
    }
    layout
}

/// The figure as JSON safe inside a `<script>` element: `<`, `>`, `&`, U+2028 and U+2029 are
/// written as `\u` escapes (they occur only inside JSON strings, so the JSON is unchanged).
fn script_json(figure: &Figure) -> Result<String, VisualizationError> {
    let text = serde_json::to_string(figure).map_err(invalid)?;
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len() + 16);
    let mut start = 0;
    let mut i = 0;
    while i < bytes.len() {
        let escape = match bytes[i] {
            b'<' => Some(("\\u003c", 1)),
            b'>' => Some(("\\u003e", 1)),
            b'&' => Some(("\\u0026", 1)),
            // U+2028 and U+2029 are E2 80 A8 and E2 80 A9 in UTF-8.
            0xE2 if bytes.get(i + 1) == Some(&0x80) && bytes.get(i + 2) == Some(&0xA8) => {
                Some(("\\u2028", 3))
            }
            0xE2 if bytes.get(i + 1) == Some(&0x80) && bytes.get(i + 2) == Some(&0xA9) => {
                Some(("\\u2029", 3))
            }
            _ => None,
        };
        match escape {
            Some((code, len)) => {
                out.push_str(&text[start..i]);
                out.push_str(code);
                i += len;
                start = i;
            }
            None => i += 1,
        }
    }
    out.push_str(&text[start..]);
    Ok(out)
}

/// The scripts plotly.js 3.0.1 and MathJax are loaded from when the page is opened.
const CDN_SCRIPTS: &str = r#"<script src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"></script>
    <script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>"#;

/// The page: the figure JSON in a data element, drawn by plotly.js from its CDN.
fn page(figure: &Figure) -> Result<String, VisualizationError> {
    Ok(format!(
        r#"<!doctype html>
<html lang="en">

<head>
    <meta charset="utf-8" />
    {CDN_SCRIPTS}
</head>

<body>
    <div>
        <div id="plotly-html-element" class="plotly-graph-div" style="height:100%; width:100%;"></div>
        <script type="application/json" id="p2a-figure">{}</script>
        <script type="module">
            const graph_div = document.getElementById("plotly-html-element");
            const figure = JSON.parse(document.getElementById("p2a-figure").textContent);
            await Plotly.newPlot(graph_div, figure);
        </script>
    </div>
</body>

</html>"#,
        script_json(figure)?
    ))
}

/// The plotted rows of `x` (and `y`), per trace: rows with a missing value are dropped, rows
/// with an infinite one left out.
struct Points {
    x: Vec<Vec<Datum>>,
    y: Vec<Vec<Datum>>,
    n_points: usize,
    missing: MissingSummary,
    /// Rows with no missing plotted value but an infinite one (not drawn).
    n_infinite: usize,
}

impl Points {
    /// The warning on the rows left out because a plotted value is infinite.
    fn warnings(&self) -> Vec<String> {
        match self.n_infinite {
            0 => Vec::new(),
            1 => vec!["1 row with an infinite value was not drawn".to_string()],
            k => vec![format!("{k} rows with an infinite value were not drawn")],
        }
    }
}

fn points(
    df: &DataFrame,
    x_col: &str,
    y_col: Option<&str>,
    group_col: Option<&str>,
) -> Result<(Points, Vec<String>), VisualizationError> {
    let n = df.height();
    let x = column_cells(df, x_col)?;
    let y = match y_col {
        Some(name) => Some(column_cells(df, name)?),
        None => None,
    };
    let (trace_of, names) = trace_groups(df, group_col)?;
    let mut plotted: Vec<(&str, &[Cell])> = vec![(x_col, x.as_slice())];
    if let (Some(name), Some(cells)) = (y_col, &y) {
        plotted.push((name, cells.as_slice()));
    }
    let missing = missing_summary(&plotted, n);
    let mut xs: Vec<Vec<Datum>> = vec![Vec::new(); names.len()];
    let mut ys: Vec<Vec<Datum>> = vec![Vec::new(); names.len()];
    let mut n_points = 0;
    let mut n_infinite = 0;
    for i in 0..n {
        if i % (1 << 16) == 0 {
            resource::checkpoint().map_err(invalid)?;
        }
        let y_cell = y.as_ref().map(|y| &y[i]);
        if matches!(x[i], Cell::Missing) || matches!(y_cell, Some(Cell::Missing)) {
            continue;
        }
        if matches!(x[i], Cell::Infinite) || matches!(y_cell, Some(Cell::Infinite)) {
            n_infinite += 1;
            continue;
        }
        let Cell::Value(xv) = &x[i] else { continue };
        let yv = match y_cell {
            Some(Cell::Value(v)) => Some(v),
            Some(_) => continue,
            None => None,
        };
        let t = trace_of[i];
        xs[t].push(xv.clone());
        if let Some(v) = yv {
            ys[t].push(v.clone());
        }
        n_points += 1;
    }
    Ok((
        Points {
            x: xs,
            y: ys,
            n_points,
            missing,
            n_infinite,
        },
        names,
    ))
}

/// Trace `i`'s name: the group label, or `fallback` without a group column.
fn trace_name(names: &[String], i: usize, grouped: bool, fallback: &str) -> String {
    if grouped {
        names[i].clone()
    } else {
        fallback.to_string()
    }
}

/// Create an interactive scatter plot from a DataFrame.
///
/// # Arguments
///
/// * `df` - Polars DataFrame containing the data
/// * `x_col` - Column name for x-axis values (numeric, `Date` or `Datetime`)
/// * `y_col` - Column name for y-axis values (numeric)
/// * `group_col` - Optional column name for grouping (one trace per level in R `factor()`
///   order; a missing label is the group "NA", last)
/// * `config` - Chart configuration
pub fn scatter_interactive(
    df: &DataFrame,
    x_col: &str,
    y_col: &str,
    group_col: Option<&str>,
    config: InteractiveConfig,
) -> Result<InteractivePlotResult, VisualizationError> {
    let (p, names) = points(df, x_col, Some(y_col), group_col)?;
    let warnings = p.warnings();
    let grouped = group_col.is_some();
    let mut data = Vec::with_capacity(names.len());
    for (i, (x, y)) in p.x.into_iter().zip(p.y).enumerate() {
        let mut s = settings(json!({
            "type": "scatter",
            "mode": "markers",
            "name": trace_name(&names, i, grouped, y_col),
            "marker": { "color": trace_color(&config, i) },
        }));
        if let Some(o) = config.opacity {
            s.insert("opacity".into(), json!(o));
        }
        data.push(Trace {
            settings: s,
            x,
            y: Some(y),
        });
    }
    let figure = Figure {
        data,
        layout: layout(&config),
    };
    Ok(InteractivePlotResult {
        html: page(&figure)?,
        n_points: p.n_points,
        chart_type: "Scatter".to_string(),
        missing: p.missing,
        warnings,
    })
}

/// Create an interactive histogram from a DataFrame column (plotly.js bins the values when the
/// page is drawn).
///
/// # Arguments
///
/// * `df` - Polars DataFrame containing the data
/// * `col` - Column name for histogram values (numeric, `Date` or `Datetime`)
/// * `group_col` - Optional column name for grouping (overlaid histograms in R `factor()`
///   order; a missing label is the group "NA", last)
/// * `config` - Chart configuration
pub fn histogram_interactive(
    df: &DataFrame,
    col: &str,
    group_col: Option<&str>,
    config: InteractiveConfig,
) -> Result<InteractivePlotResult, VisualizationError> {
    let (p, names) = points(df, col, None, group_col)?;
    let warnings = p.warnings();
    let grouped = group_col.is_some();
    let mut data = Vec::with_capacity(names.len());
    for (i, x) in p.x.into_iter().enumerate() {
        let mut s = settings(json!({
            "type": "histogram",
            "name": trace_name(&names, i, grouped, col),
            "marker": { "color": trace_color(&config, i) },
        }));
        if let Some(o) = config.opacity {
            s.insert("opacity".into(), json!(o));
        }
        data.push(Trace {
            settings: s,
            x,
            y: None,
        });
    }
    let mut layout = layout(&config);
    if grouped {
        layout.insert("barmode".into(), json!("overlay"));
    }
    let figure = Figure { data, layout };
    Ok(InteractivePlotResult {
        html: page(&figure)?,
        n_points: p.n_points,
        chart_type: "Histogram".to_string(),
        missing: p.missing,
        warnings,
    })
}

/// Create an interactive line chart from a DataFrame (points joined in row order).
///
/// # Arguments
///
/// * `df` - Polars DataFrame containing the data
/// * `x_col` - Column name for x-axis values (numeric, `Date` or `Datetime`)
/// * `y_col` - Column name for y-axis values (numeric)
/// * `config` - Chart configuration
pub fn line_interactive(
    df: &DataFrame,
    x_col: &str,
    y_col: &str,
    config: InteractiveConfig,
) -> Result<InteractivePlotResult, VisualizationError> {
    let (p, _) = points(df, x_col, Some(y_col), None)?;
    let warnings = p.warnings();
    let mut data = Vec::with_capacity(1);
    for (i, (x, y)) in p.x.into_iter().zip(p.y).enumerate() {
        let mut s = settings(json!({
            "type": "scatter",
            "mode": "lines",
            "name": y_col,
            "line": { "color": trace_color(&config, i) },
        }));
        if let Some(o) = config.opacity {
            s.insert("opacity".into(), json!(o));
        }
        data.push(Trace {
            settings: s,
            x,
            y: Some(y),
        });
    }
    let figure = Figure {
        data,
        layout: layout(&config),
    };
    Ok(InteractivePlotResult {
        html: page(&figure)?,
        n_points: p.n_points,
        chart_type: "Line".to_string(),
        missing: p.missing,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::df;

    fn make_test_df() -> DataFrame {
        df! {
            "x" => [1.0, 2.0, 3.0, 4.0, 5.0],
            "y" => [2.1, 3.9, 6.2, 7.8, 10.1],
            "group" => ["A", "A", "B", "B", "B"],
        }
        .unwrap()
    }

    #[test]
    fn test_scatter_interactive() {
        let df = make_test_df();
        let config = InteractiveConfig {
            title: Some("Test Scatter".to_string()),
            ..Default::default()
        };

        let result = scatter_interactive(&df, "x", "y", None, config).unwrap();
        assert_eq!(result.n_points, 5);
        assert_eq!(result.chart_type, "Scatter");
        assert!(result.html.contains("plotly"));
    }

    #[test]
    fn test_scatter_with_group() {
        let df = make_test_df();
        let config = InteractiveConfig::default();

        let result = scatter_interactive(&df, "x", "y", Some("group"), config).unwrap();
        assert_eq!(result.n_points, 5);
        assert!(result.html.contains("plotly"));
    }

    #[test]
    fn test_histogram_interactive() {
        let df = make_test_df();
        let config = InteractiveConfig {
            title: Some("Test Histogram".to_string()),
            ..Default::default()
        };

        let result = histogram_interactive(&df, "x", None, config).unwrap();
        assert_eq!(result.n_points, 5);
        assert_eq!(result.chart_type, "Histogram");
    }

    #[test]
    fn test_line_interactive() {
        let df = make_test_df();
        let config = InteractiveConfig::default();

        let result = line_interactive(&df, "x", "y", config).unwrap();
        assert_eq!(result.n_points, 5);
        assert_eq!(result.chart_type, "Line");
    }

    #[test]
    fn test_invalid_column() {
        let df = make_test_df();
        let config = InteractiveConfig::default();

        let result = scatter_interactive(&df, "nonexistent", "y", None, config);
        assert!(result.is_err());
    }

    /// The HTML without whitespace, so trace names compare as `"name":"<label>"`.
    fn squeezed(html: &str) -> String {
        html.chars().filter(|c| !c.is_whitespace()).collect()
    }

    #[test]
    fn test_interactive_charts_label_a_missing_group_na() {
        // DATA-27: plotlars 0.11.4 (common/polar.rs:24) unwraps every unique group label, so a
        // null group panicked. ggplot2 shows a missing discrete value as "NA"; p2a does the same.
        let df = df!(
            "x" => [1.0, 2.0, 3.0, 4.0],
            "y" => [2.1, 3.9, 6.2, 7.8],
            "g" => [Some("a"), None, Some("b"), Some("a")],
            "k" => [Some(1i64), Some(2), None, Some(1)]
        )
        .unwrap();
        for gc in ["g", "k"] {
            let s =
                scatter_interactive(&df, "x", "y", Some(gc), InteractiveConfig::default()).unwrap();
            assert!(squeezed(&s.html).contains("\"name\":\"NA\""), "{gc}");
            assert_eq!(s.n_points, 4);
            let h =
                histogram_interactive(&df, "x", Some(gc), InteractiveConfig::default()).unwrap();
            assert!(squeezed(&h.html).contains("\"name\":\"NA\""), "{gc}");
        }
        let s =
            scatter_interactive(&df, "x", "y", Some("g"), InteractiveConfig::default()).unwrap();
        assert!(squeezed(&s.html).contains("\"name\":\"a\""));
        let k =
            scatter_interactive(&df, "x", "y", Some("k"), InteractiveConfig::default()).unwrap();
        assert!(squeezed(&k.html).contains("\"name\":\"1\""));
    }

    /// The figure JSON of a page (the builder's `p2a-figure` element), unescaped.
    fn figure(html: &str) -> Value {
        let start = html
            .find(r#"<script type="application/json" id="p2a-figure">"#)
            .expect("figure element");
        let body = &html[start..];
        let body = &body[body.find('>').unwrap() + 1..];
        let end = body.find("</script>").unwrap();
        serde_json::from_str(&body[..end]).expect("figure JSON")
    }

    fn numbers(v: &Value) -> Vec<f64> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect()
    }

    /// DATA-64: plotlars cast every value to f32, so 20240103 showed as 20240104.
    #[test]
    fn test_interactive_keeps_f64_values() {
        let df = df!(
            "date" => [20240101i64, 20240102, 20240103],
            "revenue" => [16777217.25, 123456789.12, 1700000001.0]
        )
        .unwrap();
        let xs = [20240101.0, 20240102.0, 20240103.0];
        let ys = [16777217.25, 123456789.12, 1700000001.0];
        let c = InteractiveConfig::default();
        let s = scatter_interactive(&df, "date", "revenue", None, c.clone()).unwrap();
        let l = line_interactive(&df, "date", "revenue", c.clone()).unwrap();
        for html in [&s.html, &l.html] {
            let f = figure(html);
            assert_eq!(numbers(&f["data"][0]["x"]), xs);
            assert_eq!(numbers(&f["data"][0]["y"]), ys);
        }
        let h = histogram_interactive(&df, "revenue", None, c).unwrap();
        assert_eq!(numbers(&figure(&h.html)["data"][0]["x"]), ys);
    }

    /// DATA-87: axis titles were dropped without a title; size, legend title and opacity were
    /// never read; histogram and line ignored the colours.
    #[test]
    fn test_interactive_applies_every_option() {
        let df = make_test_df();
        let config = InteractiveConfig {
            width: Some(640),
            height: Some(480),
            title: None,
            x_label: Some("x axis".into()),
            y_label: Some("y axis".into()),
            legend_title: Some("Groups".into()),
            opacity: Some(0.4),
            colors: Some(vec![(1, 2, 3), (4, 5, 6)]),
        };
        let charts = [
            scatter_interactive(&df, "x", "y", Some("group"), config.clone()).unwrap(),
            histogram_interactive(&df, "x", Some("group"), config.clone()).unwrap(),
            line_interactive(&df, "x", "y", config.clone()).unwrap(),
        ];
        for chart in &charts {
            let f = figure(&chart.html);
            let layout = &f["layout"];
            assert_eq!(layout["xaxis"]["title"]["text"], "x axis");
            assert_eq!(layout["yaxis"]["title"]["text"], "y axis");
            assert_eq!(layout["width"], 640);
            assert_eq!(layout["height"], 480);
            assert_eq!(layout["legend"]["title"]["text"], "Groups");
            assert!(layout.get("title").is_none());
            let traces = f["data"].as_array().unwrap();
            for (i, t) in traces.iter().enumerate() {
                assert_eq!(t["opacity"], 0.4);
                let color = t["marker"]["color"]
                    .as_str()
                    .or(t["line"]["color"].as_str())
                    .unwrap();
                let want = if i % 2 == 0 {
                    "rgb(1, 2, 3)"
                } else {
                    "rgb(4, 5, 6)"
                };
                assert_eq!(color, want, "{}", chart.chart_type);
            }
        }
    }

    #[test]
    fn test_interactive_groups_in_factor_order() {
        let df = df!(
            "x" => [1.0, 2.0, 3.0, 4.0],
            "y" => [2.1, 3.9, 6.2, 7.8],
            "k" => [Some(2i64), Some(10), None, Some(2)],
            "s" => ["b", "B", "a", "b"]
        )
        .unwrap();
        let names = |gc: &str| -> Vec<String> {
            let s =
                scatter_interactive(&df, "x", "y", Some(gc), InteractiveConfig::default()).unwrap();
            figure(&s.html)["data"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| t["name"].as_str().unwrap().to_string())
                .collect()
        };
        assert_eq!(names("k"), ["2", "10", "NA"]);
        assert_eq!(names("s"), ["B", "a", "b"]);
    }

    #[test]
    fn test_interactive_escapes_text_in_script() {
        let df = df!(
            "x" => [1.0, 2.0, 3.0],
            "y" => [2.1, 3.9, 6.2],
            "g" => ["</script><b>&", "a\u{2028}b", "plain"]
        )
        .unwrap();
        let config = InteractiveConfig {
            title: Some("</script><script>alert(1)</script>".into()),
            ..Default::default()
        };
        let s = scatter_interactive(&df, "x", "y", Some("g"), config).unwrap();
        assert_eq!(s.html.matches("</script>").count(), 4, "{}", s.html);
        assert!(!s.html.contains('\u{2028}'));
        let f = figure(&s.html);
        assert_eq!(
            f["layout"]["title"]["text"],
            "</script><script>alert(1)</script>"
        );
        let names: Vec<&str> = f["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"</script><b>&") && names.contains(&"a\u{2028}b"));
    }

    #[test]
    fn test_interactive_missing_rows() {
        let df = df!(
            "x" => [Some(1i64), None, Some(3), Some(4), Some(5), Some(6)],
            "y" => [1.5, 2.5, f64::NAN, f64::INFINITY, 5.5, 6.5]
        )
        .unwrap();
        let s = scatter_interactive(&df, "x", "y", None, InteractiveConfig::default()).unwrap();
        // One null x and one NaN y are missing; the +Inf y cannot be drawn.
        assert_eq!(s.n_points, 3);
        assert_eq!(s.missing.n_dropped, 2);
        assert_eq!(
            s.missing.by_column,
            [("x".to_string(), 1), ("y".to_string(), 1)]
        );
        let f = figure(&s.html);
        assert_eq!(numbers(&f["data"][0]["y"]), [1.5, 5.5, 6.5]);
        assert!(!s.html.contains("null"));
        assert_eq!(s.warnings, ["1 row with an infinite value was not drawn"]);
        let h = histogram_interactive(&df, "y", None, InteractiveConfig::default()).unwrap();
        assert_eq!((h.n_points, h.missing.n_dropped), (4, 1));
        assert_eq!(h.warnings.len(), 1);

        let mut days = df!("d" => [0i32, 19723, 19724], "y" => [1.0, 2.0, 3.0]).unwrap();
        let d = days.column("d").unwrap().cast(&DataType::Date).unwrap();
        days.with_column(d).unwrap();
        let l = line_interactive(&days, "d", "y", InteractiveConfig::default()).unwrap();
        let x = &figure(&l.html)["data"][0]["x"];
        assert_eq!(x, &json!(["1970-01-01", "2024-01-01", "2024-01-02"]));
        assert!(l.missing.is_empty());
        assert!(l.warnings.is_empty());
    }
}
