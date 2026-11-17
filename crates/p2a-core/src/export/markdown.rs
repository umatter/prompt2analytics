//! Markdown table export for regression and analysis results.
//!
//! Produces GitHub-flavored markdown tables suitable for:
//! - README documentation
//! - GitHub issues and pull requests
//! - Jupyter notebooks
//! - General documentation
//!
//! Undefined values (NaN, e.g. aliased coefficients) print as `NA`, as R does. An estimate or
//! standard error that would round to zero at the style's decimals is written with that many
//! significant digits in scientific notation (`1.156e-07`).

use super::{LegendStyle, format_estimate, significance_legend};
use super::latex::format_cell;
use crate::regression::OlsResult;
use crate::traits::SignificanceLevel;

/// Style options for Markdown tables.
#[derive(Debug, Clone)]
pub struct MarkdownStyle {
    /// Show significance stars
    pub significance_stars: bool,
    /// Show standard errors in parentheses below coefficients
    pub se_in_parentheses: bool,
    /// Decimal places for coefficients
    pub coef_decimals: usize,
    /// Decimal places for standard errors
    pub se_decimals: usize,
    /// Include N in footer
    pub show_n: bool,
    /// Include R² in footer
    pub show_r_squared: bool,
    /// Include Adjusted R² in footer
    pub show_adj_r_squared: bool,
    /// Include F-statistic in footer
    pub show_f_stat: bool,
    /// Table caption (appears above table)
    pub caption: Option<String>,
}

impl Default for MarkdownStyle {
    fn default() -> Self {
        Self {
            significance_stars: true,
            se_in_parentheses: true,
            coef_decimals: 4,
            se_decimals: 4,
            show_n: true,
            show_r_squared: true,
            show_adj_r_squared: true,
            show_f_stat: true,
            caption: None,
        }
    }
}

/// Builder for Markdown regression tables.
#[derive(Debug)]
pub struct MarkdownTableBuilder {
    results: Vec<(String, OlsResult)>,
    style: MarkdownStyle,
}

impl MarkdownTableBuilder {
    /// Create a new Markdown table builder.
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            style: MarkdownStyle::default(),
        }
    }

    /// Add a model result to the table.
    ///
    /// # Arguments
    /// * `name` - Column header for this model (e.g., "(1)", "OLS", "FE"). Plain text:
    ///   Markdown characters in it are escaped.
    /// * `result` - The OLS regression result
    pub fn add_model(mut self, name: impl Into<String>, result: OlsResult) -> Self {
        self.results.push((name.into(), result));
        self
    }

    /// Set the table style.
    pub fn style(mut self, style: MarkdownStyle) -> Self {
        self.style = style;
        self
    }

    /// Set the table caption. Plain text: Markdown characters in it are escaped.
    pub fn caption(mut self, caption: impl Into<String>) -> Self {
        self.style.caption = Some(caption.into());
        self
    }

    /// Build the Markdown table string.
    pub fn build(&self) -> String {
        if self.results.is_empty() {
            return String::from("*No models to display*");
        }

        let n_models = self.results.len();

        // Collect all unique variable names across models
        let mut all_vars: Vec<String> = Vec::new();
        for (_, result) in &self.results {
            for var in &result.variable_names {
                if !all_vars.contains(var) {
                    all_vars.push(var.clone());
                }
            }
        }

        let mut md = String::new();

        // Caption
        if let Some(cap) = &self.style.caption {
            md.push_str(&format!("**{}**\n\n", escape_markdown(cap.trim())));
        }

        // Header row
        md.push_str("| Variable |");
        for (name, _) in &self.results {
            md.push_str(&format!(" {} |", escape_markdown(name)));
        }
        md.push('\n');

        // Separator row
        md.push_str("|----------|");
        for _ in 0..n_models {
            md.push_str("--------:|");
        }
        md.push('\n');

        // Coefficient rows
        for var in &all_vars {
            // Coefficient row
            md.push_str(&format!("| {} |", escape_markdown(var)));
            for (_, result) in &self.results {
                if let Some(idx) = result.variable_names.iter().position(|v| v == var) {
                    let coef = &result.coefficients[idx];
                    let stars = if self.style.significance_stars {
                        significance_stars(&coef.significance)
                    } else {
                        String::new()
                    };
                    md.push_str(&format!(
                        " {}{} |",
                        format_estimate(coef.estimate, self.style.coef_decimals),
                        stars
                    ));
                } else {
                    md.push_str(" |");
                }
            }
            md.push('\n');

            // Standard error row (if enabled)
            if self.style.se_in_parentheses {
                md.push_str("| |");
                for (_, result) in &self.results {
                    if let Some(idx) = result.variable_names.iter().position(|v| v == var) {
                        let coef = &result.coefficients[idx];
                        md.push_str(&format!(
                            " ({}) |",
                            format_estimate(coef.std_error, self.style.se_decimals)
                        ));
                    } else {
                        md.push_str(" |");
                    }
                }
                md.push('\n');
            }
        }

        // Footer statistics
        if self.style.show_n {
            md.push_str("| N |");
            for (_, result) in &self.results {
                md.push_str(&format!(" {} |", result.n_obs));
            }
            md.push('\n');
        }

        if self.style.show_r_squared {
            md.push_str("| R² |");
            for (_, result) in &self.results {
                md.push_str(&format!(" {} |", format_cell(result.r_squared, 4)));
            }
            md.push('\n');
        }

        if self.style.show_adj_r_squared {
            md.push_str("| Adj. R² |");
            for (_, result) in &self.results {
                md.push_str(&format!(" {} |", format_cell(result.adj_r_squared, 4)));
            }
            md.push('\n');
        }

        if self.style.show_f_stat {
            md.push_str("| F-stat |");
            for (_, result) in &self.results {
                md.push_str(&format!(" {} |", format_cell(result.f_statistic, 2)));
            }
            md.push('\n');
        }

        // Significance note
        if self.style.significance_stars {
            md.push_str("\n*");
            if self.style.se_in_parentheses {
                md.push_str("Standard errors in parentheses. ");
            }
            md.push_str(&significance_legend(LegendStyle::Markdown));
            md.push_str("*\n");
        }

        md
    }
}

impl Default for MarkdownTableBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert significance level to markdown-escaped stars.
fn significance_stars(level: &SignificanceLevel) -> String {
    match level {
        SignificanceLevel::TenthPercent => "\\*\\*\\*".to_string(), // p < 0.001
        SignificanceLevel::OnePercent => "\\*\\*".to_string(),      // p < 0.01
        SignificanceLevel::FivePercent => "\\*".to_string(),        // p < 0.05
        SignificanceLevel::TenPercent => "†".to_string(),           // p < 0.10
        SignificanceLevel::NotSignificant => String::new(),
    }
}

/// Escape text for a Markdown table cell or the caption, in one pass: a backslash first (so
/// an escape in the text stays literal), then `|`, `*`, `_` and `` ` ``; a line break (CR, LF
/// or CRLF) and any other control character become a space.
fn escape_markdown(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' | '|' | '*' | '_' | '`' => {
                out.push('\\');
                out.push(c);
            }
            '\r' => {
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
                out.push(' ');
            }
            c if c.is_control() => out.push(' '),
            c => out.push(c),
        }
    }
    out
}

impl OlsResult {
    /// Export result to Markdown table string.
    ///
    /// For multi-model tables, use `MarkdownTableBuilder`.
    pub fn to_markdown(&self) -> String {
        MarkdownTableBuilder::new()
            .add_model("(1)", self.clone())
            .build()
    }

    /// Export result to Markdown table with custom style.
    pub fn to_markdown_styled(&self, style: MarkdownStyle) -> String {
        MarkdownTableBuilder::new()
            .add_model("(1)", self.clone())
            .style(style)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_markdown() {
        assert_eq!(escape_markdown("a|b"), "a\\|b");
        assert_eq!(escape_markdown("*bold*"), "\\*bold\\*");
        assert_eq!(escape_markdown("var_1"), "var\\_1");
    }

    #[test]
    fn test_significance_stars() {
        assert_eq!(
            significance_stars(&SignificanceLevel::TenthPercent),
            "\\*\\*\\*"
        );
        assert_eq!(significance_stars(&SignificanceLevel::OnePercent), "\\*\\*");
        assert_eq!(significance_stars(&SignificanceLevel::FivePercent), "\\*");
        assert_eq!(significance_stars(&SignificanceLevel::TenPercent), "†");
        assert_eq!(significance_stars(&SignificanceLevel::NotSignificant), "");
    }

    /// The cells of a GFM table row: split at `|` not preceded by a backslash (a `\\` before a
    /// pipe is an escaped backslash, so the pipe after it splits).
    fn gfm_cells(line: &str) -> Vec<String> {
        let inner = line.trim().trim_start_matches('|');
        let mut cells = Vec::new();
        let mut cell = String::new();
        let mut chars = inner.chars();
        while let Some(c) = chars.next() {
            match c {
                '\\' => {
                    cell.push(c);
                    if let Some(next) = chars.next() {
                        cell.push(next);
                    }
                }
                '|' => cells.push(std::mem::take(&mut cell)),
                _ => cell.push(c),
            }
        }
        if !cell.trim().is_empty() {
            cells.push(cell);
        }
        cells
    }

    fn small_ols() -> OlsResult {
        use ndarray::{Array1, Array2};
        let x = Array2::from_shape_fn((8, 2), |(i, j)| if j == 0 { 1.0 } else { (i + 1) as f64 });
        let y = Array1::from(vec![1.1, 1.9, 3.2, 3.8, 5.1, 6.2, 6.8, 8.1]);
        let names = vec!["(Intercept)".to_string(), "x".to_string()];
        crate::regression::run_ols_raw(&x, &y, &names, "y", crate::regression::CovarianceType::HC1)
            .unwrap()
    }

    #[test]
    fn test_markdown_escapes_headers_and_caption() {
        let md = MarkdownTableBuilder::new()
            .add_model("OLS|robust", small_ols())
            .add_model("a\\|b", small_ols())
            .add_model("x\ny", small_ols())
            .add_model("x\r\ny", small_ols())
            .caption("**bold** caption")
            .build();
        let lines: Vec<&str> = md.lines().collect();
        let header = lines.iter().position(|l| l.starts_with("| Variable")).unwrap();
        let cells = gfm_cells(lines[header]);
        assert_eq!(cells.len(), 5, "{md}");
        assert_eq!(cells.len(), gfm_cells(lines[header + 1]).len(), "{md}");
        assert_eq!(cells[1].trim(), "OLS\\|robust", "{md}");
        assert_eq!(cells[2].trim(), "a\\\\\\|b", "{md}");
        assert_eq!(cells[3].trim(), "x y", "{md}");
        assert_eq!(cells[4].trim(), "x y", "{md}");
        assert!(md.starts_with("**\\*\\*bold\\*\\* caption**\n\n"), "{md}");
        assert_eq!(escape_markdown("a\\b_c`d"), "a\\\\b\\_c\\`d");
        assert_eq!(escape_markdown("t\tab"), "t ab");
    }

    #[test]
    fn test_empty_builder() {
        let builder = MarkdownTableBuilder::new();
        assert_eq!(builder.build(), "*No models to display*");
    }

    /// The table's rows, without the `---` separator; cells trimmed.
    fn markdown_rows(md: &str) -> Vec<Vec<String>> {
        md.lines()
            .filter(|l| l.starts_with('|') && !l.contains("---"))
            .map(|l| {
                l.trim_matches('|')
                    .split('|')
                    .map(|c| c.trim().to_string())
                    .collect()
            })
            .collect()
    }

    /// `-42.7144\*\*\*` as the number and the stars as written.
    fn number_and_stars(cell: &str) -> (f64, String) {
        let end = cell.find('\\').unwrap_or(cell.len());
        let number = cell[..end].trim_matches(|c| matches!(c, '(' | ')'));
        (number.parse().unwrap(), cell[end..].to_string())
    }

    #[test]
    fn test_validate_markdown_table_ols_grunfeld() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("export_ols_grunfeld");
        let fit = |x: &[&str]| {
            crate::regression::run_ols(
                &ds,
                "inv",
                x,
                true,
                crate::regression::CovarianceType::Standard,
            )
            .unwrap()
        };
        let md = MarkdownTableBuilder::new()
            .add_model("(1)", fit(&["value", "capital"]))
            .add_model("(2)", fit(&["value"]))
            .build();
        let rows = markdown_rows(&md);
        let row = |label: &str| rows.iter().position(|r| r[0] == label).unwrap();
        let mut check = Check::new(&case);
        for m in 0..2 {
            let tag = format!("m{}_", m + 1);
            let (mut names, mut coef, mut se) = (Vec::new(), Vec::new(), Vec::new());
            for name in ["(Intercept)", "value", "capital"] {
                let i = row(name);
                if rows[i].get(m + 1).is_none_or(|c| c.is_empty()) {
                    continue;
                }
                let (c, stars) = number_and_stars(&rows[i][m + 1]);
                let p = case.named(&format!("{tag}p"), name);
                let want = SignificanceLevel::from_p_value(p)
                    .stars()
                    .replace('*', "\\*");
                assert_eq!(stars, want, "{name}");
                names.push(name);
                coef.push(c);
                se.push(number_and_stars(&rows[i + 1][m + 1]).0);
            }
            let footer = |label: &str| number_and_stars(&rows[row(label)][m + 1]).0;
            check
                .named(&format!("{tag}coef"), &names, &coef, Tol::printed(4))
                .named(&format!("{tag}se"), &names, &se, Tol::printed(4))
                .scalar(&format!("{tag}r2"), footer("R²"), Tol::printed(4))
                .scalar(&format!("{tag}adj_r2"), footer("Adj. R²"), Tol::printed(4))
                .scalar(&format!("{tag}f_stat"), footer("F-stat"), Tol::printed(2))
                .scalar("n_obs", footer("N"), Tol::exact());
        }
        check.finish();
    }
}
