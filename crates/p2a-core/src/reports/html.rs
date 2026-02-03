//! HTML report generation.
//!
//! Generates self-contained HTML reports with embedded CSS, tables, and charts.

use serde::{Deserialize, Serialize};

/// A complete HTML report structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HtmlReport {
    /// Report title
    pub title: String,
    /// Optional subtitle/description
    pub subtitle: Option<String>,
    /// Author name (optional)
    pub author: Option<String>,
    /// Generation timestamp
    pub generated_at: String,
    /// Report sections
    pub sections: Vec<ReportSection>,
}

/// A section within the report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSection {
    /// Section title
    pub title: String,
    /// Section content items
    pub content: Vec<ReportContent>,
}

/// Content types that can appear in a report section.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ReportContent {
    /// Plain text paragraph
    Text { text: String },
    /// Markdown text (will be converted to HTML)
    Markdown { markdown: String },
    /// Pre-formatted code block
    Code {
        code: String,
        language: Option<String>,
    },
    /// Data table
    Table(ReportTable),
    /// Chart image (base64 encoded PNG)
    Chart {
        title: Option<String>,
        image_base64: String,
        caption: Option<String>,
    },
    /// Key-value statistics
    Statistics { items: Vec<(String, String)> },
}

/// A table within the report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportTable {
    /// Table caption/title
    pub caption: Option<String>,
    /// Column headers
    pub headers: Vec<String>,
    /// Table rows (each row is a vector of cell values)
    pub rows: Vec<Vec<String>>,
    /// Optional alignment for columns ('l', 'c', 'r')
    pub alignments: Option<Vec<char>>,
}

impl HtmlReport {
    /// Create a new empty report.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            subtitle: None,
            author: None,
            generated_at: chrono_timestamp(),
            sections: Vec::new(),
        }
    }

    /// Set the report subtitle.
    pub fn with_subtitle(mut self, subtitle: impl Into<String>) -> Self {
        self.subtitle = Some(subtitle.into());
        self
    }

    /// Set the author name.
    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Add a section to the report.
    pub fn add_section(&mut self, section: ReportSection) {
        self.sections.push(section);
    }

    /// Generate the complete HTML document.
    pub fn to_html(&self) -> String {
        let mut html = String::new();

        // HTML header with embedded CSS
        html.push_str(&format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{}</title>
    {}
</head>
<body>
    <div class="container">
"#,
            escape_html(&self.title),
            CSS_STYLES
        ));

        // Report header
        html.push_str(&format!(
            r#"        <header class="report-header">
            <h1>{}</h1>
"#,
            escape_html(&self.title)
        ));

        if let Some(ref subtitle) = self.subtitle {
            html.push_str(&format!(
                "            <p class=\"subtitle\">{}</p>\n",
                escape_html(subtitle)
            ));
        }

        html.push_str(&format!(
            "            <p class=\"meta\">Generated: {}</p>\n",
            escape_html(&self.generated_at)
        ));

        if let Some(ref author) = self.author {
            html.push_str(&format!(
                "            <p class=\"meta\">Author: {}</p>\n",
                escape_html(author)
            ));
        }

        html.push_str("        </header>\n\n");

        // Table of contents
        if self.sections.len() > 1 {
            html.push_str("        <nav class=\"toc\">\n");
            html.push_str("            <h2>Table of Contents</h2>\n");
            html.push_str("            <ul>\n");
            for (i, section) in self.sections.iter().enumerate() {
                html.push_str(&format!(
                    "                <li><a href=\"#section-{}\">{}</a></li>\n",
                    i,
                    escape_html(&section.title)
                ));
            }
            html.push_str("            </ul>\n");
            html.push_str("        </nav>\n\n");
        }

        // Sections
        for (i, section) in self.sections.iter().enumerate() {
            html.push_str(&format!("        <section id=\"section-{}\">\n", i));
            html.push_str(&format!(
                "            <h2>{}</h2>\n",
                escape_html(&section.title)
            ));

            for content in &section.content {
                html.push_str(&render_content(content));
            }

            html.push_str("        </section>\n\n");
        }

        // Footer
        html.push_str(
            r#"        <footer>
            <p>Generated by prompt2analytics</p>
        </footer>
    </div>
</body>
</html>
"#,
        );

        html
    }
}

impl ReportSection {
    /// Create a new section.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            content: Vec::new(),
        }
    }

    /// Add text content.
    pub fn add_text(&mut self, text: impl Into<String>) {
        self.content.push(ReportContent::Text { text: text.into() });
    }

    /// Add a code block.
    pub fn add_code(&mut self, code: impl Into<String>, language: Option<&str>) {
        self.content.push(ReportContent::Code {
            code: code.into(),
            language: language.map(|s| s.to_string()),
        });
    }

    /// Add a table.
    pub fn add_table(&mut self, table: ReportTable) {
        self.content.push(ReportContent::Table(table));
    }

    /// Add a chart image.
    pub fn add_chart(
        &mut self,
        image_base64: impl Into<String>,
        title: Option<&str>,
        caption: Option<&str>,
    ) {
        self.content.push(ReportContent::Chart {
            title: title.map(|s| s.to_string()),
            image_base64: image_base64.into(),
            caption: caption.map(|s| s.to_string()),
        });
    }

    /// Add key-value statistics.
    pub fn add_statistics(&mut self, items: Vec<(String, String)>) {
        self.content.push(ReportContent::Statistics { items });
    }
}

impl ReportTable {
    /// Create a new table.
    pub fn new(headers: Vec<String>) -> Self {
        Self {
            caption: None,
            headers,
            rows: Vec::new(),
            alignments: None,
        }
    }

    /// Set the table caption.
    pub fn with_caption(mut self, caption: impl Into<String>) -> Self {
        self.caption = Some(caption.into());
        self
    }

    /// Add a row to the table.
    pub fn add_row(&mut self, row: Vec<String>) {
        self.rows.push(row);
    }

    /// Set column alignments.
    pub fn with_alignments(mut self, alignments: Vec<char>) -> Self {
        self.alignments = Some(alignments);
        self
    }
}

/// Render a content item to HTML.
fn render_content(content: &ReportContent) -> String {
    match content {
        ReportContent::Text { text } => {
            format!("            <p>{}</p>\n", escape_html(text))
        }
        ReportContent::Markdown { markdown } => {
            // Simple markdown conversion (basic support)
            format!(
                "            <div class=\"markdown\">{}</div>\n",
                simple_markdown_to_html(markdown)
            )
        }
        ReportContent::Code { code, language } => {
            let lang_class = language
                .as_ref()
                .map(|l| format!(" class=\"language-{}\"", l))
                .unwrap_or_default();
            format!(
                "            <pre><code{}>{}</code></pre>\n",
                lang_class,
                escape_html(code)
            )
        }
        ReportContent::Table(table) => render_table(table),
        ReportContent::Chart {
            title,
            image_base64,
            caption,
        } => {
            let mut html = String::from("            <figure class=\"chart\">\n");
            if let Some(t) = title {
                html.push_str(&format!(
                    "                <figcaption class=\"chart-title\">{}</figcaption>\n",
                    escape_html(t)
                ));
            }
            html.push_str(&format!(
                "                <img src=\"data:image/png;base64,{}\" alt=\"Chart\">\n",
                image_base64
            ));
            if let Some(c) = caption {
                html.push_str(&format!(
                    "                <figcaption>{}</figcaption>\n",
                    escape_html(c)
                ));
            }
            html.push_str("            </figure>\n");
            html
        }
        ReportContent::Statistics { items } => {
            let mut html = String::from("            <dl class=\"statistics\">\n");
            for (key, value) in items {
                html.push_str(&format!(
                    "                <dt>{}</dt><dd>{}</dd>\n",
                    escape_html(key),
                    escape_html(value)
                ));
            }
            html.push_str("            </dl>\n");
            html
        }
    }
}

/// Render a table to HTML.
fn render_table(table: &ReportTable) -> String {
    let mut html = String::from("            <div class=\"table-wrapper\">\n");
    html.push_str("            <table>\n");

    if let Some(ref caption) = table.caption {
        html.push_str(&format!(
            "                <caption>{}</caption>\n",
            escape_html(caption)
        ));
    }

    // Header row
    html.push_str("                <thead>\n                    <tr>\n");
    for (i, header) in table.headers.iter().enumerate() {
        let align = table
            .alignments
            .as_ref()
            .and_then(|a| a.get(i))
            .map(|&c| match c {
                'l' => " style=\"text-align: left;\"",
                'r' => " style=\"text-align: right;\"",
                'c' => " style=\"text-align: center;\"",
                _ => "",
            })
            .unwrap_or("");
        html.push_str(&format!(
            "                        <th{}>{}</th>\n",
            align,
            escape_html(header)
        ));
    }
    html.push_str("                    </tr>\n                </thead>\n");

    // Data rows
    html.push_str("                <tbody>\n");
    for row in &table.rows {
        html.push_str("                    <tr>\n");
        for (i, cell) in row.iter().enumerate() {
            let align = table
                .alignments
                .as_ref()
                .and_then(|a| a.get(i))
                .map(|&c| match c {
                    'l' => " style=\"text-align: left;\"",
                    'r' => " style=\"text-align: right;\"",
                    'c' => " style=\"text-align: center;\"",
                    _ => "",
                })
                .unwrap_or("");
            html.push_str(&format!(
                "                        <td{}>{}</td>\n",
                align,
                escape_html(cell)
            ));
        }
        html.push_str("                    </tr>\n");
    }
    html.push_str("                </tbody>\n");

    html.push_str("            </table>\n");
    html.push_str("            </div>\n");
    html
}

/// Escape HTML special characters.
fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// Simple markdown to HTML conversion (basic support).
fn simple_markdown_to_html(md: &str) -> String {
    let mut html = String::new();
    let mut in_code_block = false;

    for line in md.lines() {
        if line.starts_with("```") {
            if in_code_block {
                html.push_str("</code></pre>\n");
                in_code_block = false;
            } else {
                html.push_str("<pre><code>");
                in_code_block = true;
            }
            continue;
        }

        if in_code_block {
            html.push_str(&escape_html(line));
            html.push('\n');
            continue;
        }

        // Headers
        if line.starts_with("### ") {
            html.push_str(&format!("<h5>{}</h5>\n", escape_html(&line[4..])));
        } else if line.starts_with("## ") {
            html.push_str(&format!("<h4>{}</h4>\n", escape_html(&line[3..])));
        } else if line.starts_with("# ") {
            html.push_str(&format!("<h3>{}</h3>\n", escape_html(&line[2..])));
        }
        // Bold
        else if line.contains("**") {
            let processed = line
                .replace("**", "<strong>")
                .replace("<strong>", "</strong>");
            // This is a simple approach - proper parsing would be more complex
            html.push_str(&format!("<p>{}</p>\n", processed));
        }
        // List items
        else if line.starts_with("- ") || line.starts_with("* ") {
            html.push_str(&format!("<li>{}</li>\n", escape_html(&line[2..])));
        }
        // Empty line
        else if line.trim().is_empty() {
            html.push_str("<br>\n");
        }
        // Regular paragraph
        else {
            html.push_str(&format!("<p>{}</p>\n", escape_html(line)));
        }
    }

    html
}

/// Get current timestamp as ISO string.
fn chrono_timestamp() -> String {
    // Simple timestamp without external dependency
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();

    // Convert to date components (simple approach)
    let days = secs / 86400;
    let remaining = secs % 86400;
    let hours = remaining / 3600;
    let minutes = (remaining % 3600) / 60;
    let seconds = remaining % 60;

    // Days since Unix epoch to date (simplified)
    let mut year = 1970;
    let mut remaining_days = days as i64;

    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    let mut month = 1;
    let days_in_months = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    for &days_in_month in &days_in_months {
        if remaining_days < days_in_month as i64 {
            break;
        }
        remaining_days -= days_in_month as i64;
        month += 1;
    }

    let day = remaining_days + 1;

    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02} UTC",
        year, month, day, hours, minutes, seconds
    )
}

fn is_leap_year(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

/// Generate an HTML report from structured data.
pub fn generate_html_report(report: &HtmlReport) -> String {
    report.to_html()
}

/// CSS styles for the HTML report.
const CSS_STYLES: &str = r#"<style>
    :root {
        --primary-color: #2563eb;
        --text-color: #1f2937;
        --bg-color: #ffffff;
        --border-color: #e5e7eb;
        --code-bg: #f3f4f6;
    }

    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        line-height: 1.6;
        color: var(--text-color);
        background-color: var(--bg-color);
    }

    .container {
        max-width: 900px;
        margin: 0 auto;
        padding: 2rem;
    }

    .report-header {
        border-bottom: 2px solid var(--primary-color);
        padding-bottom: 1.5rem;
        margin-bottom: 2rem;
    }

    .report-header h1 {
        font-size: 2rem;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 0.5rem;
    }

    .meta {
        font-size: 0.9rem;
        color: #9ca3af;
    }

    .toc {
        background-color: var(--code-bg);
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 2rem;
    }

    .toc h2 {
        font-size: 1.1rem;
        margin-bottom: 0.75rem;
    }

    .toc ul {
        list-style: none;
        padding-left: 1rem;
    }

    .toc li {
        margin-bottom: 0.5rem;
    }

    .toc a {
        color: var(--primary-color);
        text-decoration: none;
    }

    .toc a:hover {
        text-decoration: underline;
    }

    section {
        margin-bottom: 2.5rem;
    }

    section h2 {
        font-size: 1.5rem;
        color: var(--primary-color);
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    p {
        margin-bottom: 1rem;
    }

    pre {
        background-color: var(--code-bg);
        padding: 1rem;
        border-radius: 6px;
        overflow-x: auto;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }

    code {
        font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Fira Mono', 'Droid Sans Mono', monospace;
    }

    .table-wrapper {
        overflow-x: auto;
        margin-bottom: 1.5rem;
    }

    table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
    }

    caption {
        font-weight: 600;
        padding: 0.75rem;
        text-align: left;
        color: var(--text-color);
    }

    th, td {
        padding: 0.75rem;
        border: 1px solid var(--border-color);
    }

    th {
        background-color: var(--code-bg);
        font-weight: 600;
        text-align: left;
    }

    tbody tr:nth-child(even) {
        background-color: #f9fafb;
    }

    tbody tr:hover {
        background-color: #f3f4f6;
    }

    .chart {
        margin: 1.5rem 0;
        text-align: center;
    }

    .chart img {
        max-width: 100%;
        height: auto;
        border: 1px solid var(--border-color);
        border-radius: 8px;
    }

    .chart-title {
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .chart figcaption {
        font-size: 0.9rem;
        color: #6b7280;
        margin-top: 0.5rem;
    }

    .statistics {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        background-color: var(--code-bg);
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }

    .statistics dt {
        font-weight: 600;
        color: #6b7280;
        font-size: 0.85rem;
    }

    .statistics dd {
        font-size: 1.25rem;
        color: var(--text-color);
        margin-top: 0.25rem;
    }

    footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--border-color);
        text-align: center;
        color: #9ca3af;
        font-size: 0.85rem;
    }

    @media print {
        .container {
            max-width: 100%;
            padding: 1rem;
        }

        .toc {
            page-break-after: always;
        }

        section {
            page-break-inside: avoid;
        }

        .chart img {
            max-height: 400px;
        }
    }
</style>"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_report() {
        let mut report = HtmlReport::new("Test Report")
            .with_subtitle("A test report")
            .with_author("Test Author");

        let mut section = ReportSection::new("Introduction");
        section.add_text("This is the introduction.");
        report.add_section(section);

        let html = report.to_html();
        assert!(html.contains("Test Report"));
        assert!(html.contains("A test report"));
        assert!(html.contains("Introduction"));
    }

    #[test]
    fn test_table_rendering() {
        let mut table = ReportTable::new(vec!["Name".to_string(), "Value".to_string()]);
        table.add_row(vec!["Alpha".to_string(), "1.0".to_string()]);
        table.add_row(vec!["Beta".to_string(), "2.0".to_string()]);

        let html = render_table(&table);
        assert!(html.contains("<table>"));
        assert!(html.contains("Alpha"));
        assert!(html.contains("Beta"));
    }

    #[test]
    fn test_html_escaping() {
        let text = "<script>alert('xss')</script>";
        let escaped = escape_html(text);
        assert!(!escaped.contains("<script>"));
        assert!(escaped.contains("&lt;script&gt;"));
    }
}
