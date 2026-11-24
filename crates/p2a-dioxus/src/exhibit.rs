//! Exhibits: the figures and tables an analysis answer contains, which can
//! be opened in a viewer and exported.
//!
//! A figure is a PNG a tool returned (base64); a table is a Markdown table in
//! the answer text, read here as plain cell text. Tables export as CSV,
//! LaTeX (booktabs) and Markdown and copy as tab-separated text, which pastes
//! into a spreadsheet cell by cell.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use dioxus::prelude::*;
use pulldown_cmark::{Event, Tag, TagEnd};

use crate::export::download_bytes;

/// A table as plain text: the header cells and the body rows.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TableData {
    pub header: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

/// One exhibit, as the viewer shows it.
#[derive(Debug, Clone, PartialEq)]
pub enum Exhibit {
    /// A PNG, base64-encoded, and its caption ("Figure 2").
    Figure { label: String, png: String },
    /// A table and its caption.
    Table { label: String, data: TableData },
}

/// The open exhibit and the last export notice, shared by the transcript
/// (which opens exhibits) and the viewer (which shows them).
#[derive(Clone, Copy, PartialEq)]
pub struct ExhibitViewer {
    pub open: Signal<Option<Exhibit>>,
    pub note: Signal<Option<String>>,
}

impl ExhibitViewer {
    pub fn new() -> Self {
        Self {
            open: Signal::new(None),
            note: Signal::new(None),
        }
    }

    /// Shows `text` for a few seconds (a later notice replaces it).
    pub fn flash(self, text: String) {
        let mut note = self.note;
        note.set(Some(text.clone()));
        spawn(async move {
            #[cfg(target_arch = "wasm32")]
            {
                gloo_timers::future::TimeoutFuture::new(3_000).await;
            }
            #[cfg(all(
                not(target_arch = "wasm32"),
                any(feature = "desktop", feature = "mobile")
            ))]
            {
                tokio::time::sleep(std::time::Duration::from_secs(3)).await;
            }
            if note.peek().as_deref() == Some(text.as_str()) {
                note.set(None);
            }
        });
    }
}

/// The ways a table leaves the app.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableFormat {
    Csv,
    Latex,
    Markdown,
}

impl TableFormat {
    /// The name on the viewer's tabs and buttons.
    pub fn name(self) -> &'static str {
        match self {
            TableFormat::Csv => "CSV",
            TableFormat::Latex => "LaTeX",
            TableFormat::Markdown => "Markdown",
        }
    }

    fn extension(self) -> &'static str {
        match self {
            TableFormat::Csv => "csv",
            TableFormat::Latex => "tex",
            TableFormat::Markdown => "md",
        }
    }

    fn mime(self) -> &'static str {
        match self {
            TableFormat::Csv => "text/csv",
            TableFormat::Latex => "application/x-tex",
            TableFormat::Markdown => "text/markdown",
        }
    }
}

impl TableData {
    /// Reads the events between `Start(Table)` and its end (either bound
    /// may be included): each cell's text, with inline code and math kept
    /// as their source and soft breaks as spaces.
    pub fn from_events(events: &[Event]) -> Self {
        let mut table = TableData::default();
        let mut row: Vec<String> = Vec::new();
        let mut cell: Option<String> = None;
        for event in events {
            match event {
                Event::Start(Tag::TableCell) => cell = Some(String::new()),
                Event::End(TagEnd::TableCell) => {
                    if let Some(c) = cell.take() {
                        row.push(c.trim().to_string());
                    }
                }
                Event::End(TagEnd::TableHead) => table.header = std::mem::take(&mut row),
                Event::End(TagEnd::TableRow) => table.rows.push(std::mem::take(&mut row)),
                Event::Text(t) | Event::Code(t) | Event::Html(t) | Event::InlineHtml(t) => {
                    if let Some(c) = cell.as_mut() {
                        c.push_str(t);
                    }
                }
                Event::InlineMath(m) => {
                    if let Some(c) = cell.as_mut() {
                        c.push('$');
                        c.push_str(m);
                        c.push('$');
                    }
                }
                Event::SoftBreak | Event::HardBreak => {
                    if let Some(c) = cell.as_mut() {
                        c.push(' ');
                    }
                }
                _ => {}
            }
        }
        table
    }

    fn all_rows(&self) -> impl Iterator<Item = &Vec<String>> {
        std::iter::once(&self.header)
            .filter(|h| !h.is_empty())
            .chain(&self.rows)
    }

    /// RFC 4180: fields with a comma, quote or line break are quoted, quotes doubled.
    pub fn to_csv(&self) -> String {
        fn field(s: &str) -> String {
            if s.contains([',', '"', '\n', '\r']) {
                format!("\"{}\"", s.replace('"', "\"\""))
            } else {
                s.to_string()
            }
        }
        self.all_rows()
            .map(|r| r.iter().map(|c| field(c)).collect::<Vec<_>>().join(","))
            .map(|line| line + "\r\n")
            .collect()
    }

    /// Tab-separated, for pasting into a spreadsheet.
    pub fn to_tsv(&self) -> String {
        self.all_rows()
            .map(|r| {
                r.iter()
                    .map(|c| c.replace(['\t', '\n'], " "))
                    .collect::<Vec<_>>()
                    .join("\t")
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// A GitHub-flavoured Markdown table; numeric columns right-aligned.
    pub fn to_markdown(&self) -> String {
        let n = self.width();
        let cell = |s: &str| s.replace('|', "\\|");
        let line = |r: &Vec<String>| {
            let cells: Vec<String> = (0..n)
                .map(|i| cell(r.get(i).map_or("", |s| s.as_str())))
                .collect();
            format!("| {} |", cells.join(" | "))
        };
        let numeric = self.numeric_columns();
        let rule: Vec<&str> = (0..n)
            .map(|i| if numeric[i] { "---:" } else { "---" })
            .collect();
        let mut out = vec![line(&self.header), format!("| {} |", rule.join(" | "))];
        out.extend(self.rows.iter().map(line));
        out.join("\n") + "\n"
    }

    /// A booktabs `tabular` in a `table` float; numeric columns right-aligned,
    /// text columns left; TeX specials escaped, except in cells that are
    /// inline math (`$...$`).
    pub fn to_latex(&self) -> String {
        let n = self.width();
        let numeric = self.numeric_columns();
        let spec: String = (0..n).map(|i| if numeric[i] { 'r' } else { 'l' }).collect();
        let line = |r: &Vec<String>| {
            let cells: Vec<String> = (0..n)
                .map(|i| tex_cell(r.get(i).map_or("", |s| s.as_str())))
                .collect();
            format!("{} \\\\", cells.join(" & "))
        };
        let mut out = vec![
            "% requires \\usepackage{booktabs}".to_string(),
            "\\begin{table}[htbp]".to_string(),
            "\\centering".to_string(),
            format!("\\begin{{tabular}}{{{spec}}}"),
            "\\toprule".to_string(),
        ];
        if !self.header.is_empty() {
            out.push(line(&self.header));
            out.push("\\midrule".to_string());
        }
        out.extend(self.rows.iter().map(line));
        out.extend([
            "\\bottomrule".to_string(),
            "\\end{tabular}".to_string(),
            "\\end{table}".to_string(),
        ]);
        out.join("\n") + "\n"
    }

    pub fn render(&self, format: TableFormat) -> String {
        match format {
            TableFormat::Csv => self.to_csv(),
            TableFormat::Latex => self.to_latex(),
            TableFormat::Markdown => self.to_markdown(),
        }
    }

    /// Number of columns: the widest row.
    pub fn width(&self) -> usize {
        self.all_rows().map(Vec::len).max().unwrap_or(0)
    }

    /// A column is numeric when every non-empty body cell reads as a number
    /// (after stripping significance stars, brackets, a percent sign and
    /// thousands separators) and at least one cell is non-empty.
    fn numeric_columns(&self) -> Vec<bool> {
        (0..self.width())
            .map(|i| {
                let cells: Vec<&str> = self
                    .rows
                    .iter()
                    .filter_map(|r| r.get(i).map(|s| s.trim()))
                    .filter(|s| !s.is_empty())
                    .collect();
                !cells.is_empty() && cells.iter().all(|c| looks_numeric(c))
            })
            .collect()
    }
}

fn looks_numeric(cell: &str) -> bool {
    let t = cell
        .trim()
        .trim_matches(|c: char| matches!(c, '*' | '\u{2020}' | ' '));
    let inner = t
        .trim_start_matches(['(', '['])
        .trim_end_matches([')', ']'])
        .trim();
    // An interval, "[0.1, 0.3]" or "(0.1; 0.3)"
    if inner.contains(", ") || inner.contains(';') {
        return inner.split([',', ';']).all(number);
    }
    number(inner)
}

/// A single number: a sign, thousands separators, a trailing percent sign
/// and a leading comparison are allowed.
fn number(s: &str) -> bool {
    let core = s
        .trim()
        .trim_end_matches('%')
        .replace([',', '\u{2009}'], "")
        .replace('\u{2212}', "-");
    let core = core.trim_start_matches(['<', '>', '\u{2248}']).trim();
    !core.is_empty()
        && (core.parse::<f64>().is_ok() || matches!(core, "NA" | "NaN" | "Inf" | "-Inf" | "-"))
}

fn tex_cell(s: &str) -> String {
    if s.len() >= 2 && s.starts_with('$') && s.ends_with('$') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' | '%' | '$' | '#' | '_' | '{' | '}' => {
                out.push('\\');
                out.push(c);
            }
            '~' => out.push_str("\\textasciitilde{}"),
            '^' => out.push_str("\\textasciicircum{}"),
            '\\' => out.push_str("\\textbackslash{}"),
            '\u{2212}' => out.push('-'),
            c => out.push(c),
        }
    }
    out
}

/// "Figure 2" -> "p2a_figure_2_20261124_101500.png".
pub fn exhibit_filename(label: &str, extension: &str) -> String {
    let slug: String = label
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>()
        .split('_')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("_");
    let stamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    format!("p2a_{slug}_{stamp}.{extension}")
}

/// What an export reports: where it went, or why it failed.
fn outcome(result: Result<Option<String>, String>, filename: &str) -> String {
    match result {
        Ok(Some(path)) => format!("Saved to {path}"),
        Ok(None) => format!("Downloaded {filename}"),
        Err(e) => format!("Export failed: {e}"),
    }
}

/// Downloads a table in `format` and returns the notice to show.
pub fn export_table(data: &TableData, label: &str, format: TableFormat) -> String {
    let filename = exhibit_filename(label, format.extension());
    let content = data.render(format);
    outcome(
        download_bytes(content.as_bytes(), &filename, format.mime()),
        &filename,
    )
}

/// Copies a table as tab-separated text and returns the notice to show.
pub fn copy_table(data: &TableData) -> String {
    match crate::export::copy_to_clipboard(&data.to_tsv()) {
        Ok(()) => "Copied; paste into a spreadsheet or a document".to_string(),
        Err(e) => format!("Copy failed: {e}"),
    }
}

/// Copies a table in `format` (its source text) and returns the notice to show.
pub fn copy_table_as(data: &TableData, format: TableFormat) -> String {
    match crate::export::copy_to_clipboard(&data.render(format)) {
        Ok(()) => format!("Copied as {}", format.name()),
        Err(e) => format!("Copy failed: {e}"),
    }
}

/// Downloads a figure as PNG and returns the notice to show.
pub fn export_figure(png: &str, label: &str) -> String {
    let filename = exhibit_filename(label, "png");
    match STANDARD.decode(png.trim()) {
        Ok(bytes) => outcome(download_bytes(&bytes, &filename, "image/png"), &filename),
        Err(e) => format!("Export failed: the image is not valid base64 ({e})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pulldown_cmark::{Options, Parser};

    fn table(md: &str) -> TableData {
        let events: Vec<Event> =
            Parser::new_ext(md, Options::ENABLE_TABLES | Options::ENABLE_MATH).collect();
        TableData::from_events(&events)
    }

    const COEFS: &str = "| Term | Estimate | SE |\n|---|---:|---:|\n| `value` | 0.1101*** | (0.0119) |\n| capital, net | -0.31 | 0.017 |\n";

    #[test]
    fn cells_are_read_as_text() {
        let t = table(COEFS);
        assert_eq!(t.header, ["Term", "Estimate", "SE"]);
        assert_eq!(
            t.rows,
            [
                ["value", "0.1101***", "(0.0119)"],
                ["capital, net", "-0.31", "0.017"]
            ]
        );
    }

    #[test]
    fn math_in_a_cell_keeps_its_delimiters() {
        let t = table("| a | b |\n|---|---|\n| $\\beta_1$ | 2 |\n");
        assert_eq!(t.rows[0][0], "$\\beta_1$");
        assert!(t.to_latex().contains("$\\beta_1$ & 2 \\\\"));
    }

    #[test]
    fn csv_quotes_fields_with_commas_and_quotes() {
        let mut t = table(COEFS);
        t.rows[0][0] = "say \"hi\"".into();
        assert_eq!(
            t.to_csv(),
            "Term,Estimate,SE\r\n\"say \"\"hi\"\"\",0.1101***,(0.0119)\r\n\"capital, net\",-0.31,0.017\r\n"
        );
    }

    #[test]
    fn tsv_has_one_line_per_row() {
        assert_eq!(
            table(COEFS).to_tsv(),
            "Term\tEstimate\tSE\nvalue\t0.1101***\t(0.0119)\ncapital, net\t-0.31\t0.017"
        );
    }

    #[test]
    fn latex_aligns_numbers_right_and_escapes_specials() {
        let mut t = table(COEFS);
        t.rows[1][0] = "R&D_share 50%".into();
        let tex = t.to_latex();
        assert!(tex.contains("\\begin{tabular}{lrr}"), "{tex}");
        assert!(
            tex.contains("Term & Estimate & SE \\\\\n\\midrule"),
            "{tex}"
        );
        assert!(
            tex.contains("R\\&D\\_share 50\\% & -0.31 & 0.017 \\\\"),
            "{tex}"
        );
        assert!(tex.trim_end().ends_with("\\end{table}"));
    }

    #[test]
    fn markdown_round_trips() {
        let t = table(COEFS);
        assert_eq!(table(&t.to_markdown()), t);
        assert!(t.to_markdown().contains("| --- | ---: | ---: |"));
    }

    #[test]
    fn ragged_rows_are_padded_in_latex_and_markdown() {
        let t = TableData {
            header: vec!["a".into(), "b".into()],
            rows: vec![vec!["x".into()]],
        };
        assert!(t.to_latex().contains("x &  \\\\"));
        assert!(t.to_markdown().contains("| x |  |"));
    }

    #[test]
    fn numbers_are_recognised_with_stars_brackets_and_separators() {
        for c in [
            "0.48***",
            "(0.012)",
            "[0.1, 0.3]",
            "1,234.5",
            "−0.2",
            "12%",
            "<0.001",
            "NA",
        ] {
            assert!(looks_numeric(c), "{c}");
        }
        for c in ["value", "FE", "yes", ""] {
            assert!(!looks_numeric(c), "{c}");
        }
    }

    #[test]
    fn filenames_are_slugs_with_a_stamp() {
        let f = exhibit_filename("Figure 2", "png");
        assert!(f.starts_with("p2a_figure_2_") && f.ends_with(".png"), "{f}");
        assert!(exhibit_filename("Table", "csv").starts_with("p2a_table_"));
    }
}
