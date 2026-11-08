//! HTML export for regression and analysis results.
//!
//! Produces self-contained HTML with embedded CSS for:
//! - Web display
//! - Email reports
//! - Jupyter notebook integration
//! - Standalone reports
//!
//! Undefined values (NaN, e.g. aliased coefficients) print as `NA`, as R does.

use super::latex::format_cell;
use crate::econometrics::{DiscreteResult, PanelResult};
use crate::regression::OlsResult;
use crate::traits::SignificanceLevel;
use crate::traits::convergence::{ResultNotes, notes_text};

/// Style options for HTML tables.
#[derive(Debug, Clone)]
pub struct HtmlStyle {
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
    /// Include model statistics footer
    pub show_model_stats: bool,
    /// Table caption
    pub caption: Option<String>,
    /// Use Bootstrap CSS classes
    pub use_bootstrap: bool,
    /// Include inline CSS (for standalone HTML)
    pub include_css: bool,
}

impl Default for HtmlStyle {
    fn default() -> Self {
        Self {
            significance_stars: true,
            se_in_parentheses: true,
            coef_decimals: 4,
            se_decimals: 4,
            show_n: true,
            show_r_squared: true,
            show_model_stats: true,
            caption: None,
            use_bootstrap: false,
            include_css: true,
        }
    }
}

/// Builder for HTML regression tables.
#[derive(Debug)]
pub struct HtmlTableBuilder {
    results: Vec<(String, OlsResult)>,
    style: HtmlStyle,
}

impl HtmlTableBuilder {
    /// Create a new HTML table builder.
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            style: HtmlStyle::default(),
        }
    }

    /// Add a model result to the table.
    pub fn add_model(mut self, name: impl Into<String>, result: OlsResult) -> Self {
        self.results.push((name.into(), result));
        self
    }

    /// Set the table style.
    pub fn style(mut self, style: HtmlStyle) -> Self {
        self.style = style;
        self
    }

    /// Set the table caption.
    pub fn caption(mut self, caption: impl Into<String>) -> Self {
        self.style.caption = Some(caption.into());
        self
    }

    /// Build the HTML table string.
    pub fn build(&self) -> String {
        if self.results.is_empty() {
            return String::from("<p><em>No models to display</em></p>");
        }

        let _n_models = self.results.len();

        // Collect all unique variable names across models
        let mut all_vars: Vec<String> = Vec::new();
        for (_, result) in &self.results {
            for var in &result.variable_names {
                if !all_vars.contains(var) {
                    all_vars.push(var.clone());
                }
            }
        }

        let mut html = String::new();

        // Include CSS if requested
        if self.style.include_css {
            html.push_str(&embedded_css());
        }

        // Table wrapper
        let table_class = if self.style.use_bootstrap {
            "table table-striped table-hover"
        } else {
            "regression-table"
        };
        html.push_str(&format!("<table class=\"{}\">\n", table_class));

        // Caption
        if let Some(cap) = &self.style.caption {
            html.push_str(&format!("  <caption>{}</caption>\n", escape_html(cap)));
        }

        // Header
        html.push_str("  <thead>\n    <tr>\n");
        html.push_str("      <th>Variable</th>\n");
        for (name, _) in &self.results {
            html.push_str(&format!("      <th>{}</th>\n", escape_html(name)));
        }
        html.push_str("    </tr>\n  </thead>\n");

        // Body
        html.push_str("  <tbody>\n");

        for var in &all_vars {
            // Coefficient row
            html.push_str("    <tr>\n");
            html.push_str(&format!(
                "      <td class=\"var-name\">{}</td>\n",
                escape_html(var)
            ));
            for (_, result) in &self.results {
                if let Some(idx) = result.variable_names.iter().position(|v| v == var) {
                    let coef = &result.coefficients[idx];
                    let stars = if self.style.significance_stars {
                        significance_stars_html(&coef.significance)
                    } else {
                        String::new()
                    };
                    html.push_str(&format!(
                        "      <td class=\"coef\">{}{}</td>\n",
                        format_cell(coef.estimate, self.style.coef_decimals),
                        stars
                    ));
                } else {
                    html.push_str("      <td></td>\n");
                }
            }
            html.push_str("    </tr>\n");

            // Standard error row
            if self.style.se_in_parentheses {
                html.push_str("    <tr class=\"se-row\">\n");
                html.push_str("      <td></td>\n");
                for (_, result) in &self.results {
                    if let Some(idx) = result.variable_names.iter().position(|v| v == var) {
                        let coef = &result.coefficients[idx];
                        html.push_str(&format!(
                            "      <td class=\"se\">({})</td>\n",
                            format_cell(coef.std_error, self.style.se_decimals)
                        ));
                    } else {
                        html.push_str("      <td></td>\n");
                    }
                }
                html.push_str("    </tr>\n");
            }
        }

        html.push_str("  </tbody>\n");

        // Footer
        if self.style.show_model_stats {
            html.push_str("  <tfoot>\n");

            if self.style.show_n {
                html.push_str("    <tr>\n      <td>N</td>\n");
                for (_, result) in &self.results {
                    html.push_str(&format!("      <td>{}</td>\n", result.n_obs));
                }
                html.push_str("    </tr>\n");
            }

            if self.style.show_r_squared {
                html.push_str("    <tr>\n      <td>R<sup>2</sup></td>\n");
                for (_, result) in &self.results {
                    html.push_str(&format!(
                        "      <td>{}</td>\n",
                        format_cell(result.r_squared, 4)
                    ));
                }
                html.push_str("    </tr>\n");

                html.push_str("    <tr>\n      <td>Adj. R<sup>2</sup></td>\n");
                for (_, result) in &self.results {
                    html.push_str(&format!(
                        "      <td>{}</td>\n",
                        format_cell(result.adj_r_squared, 4)
                    ));
                }
                html.push_str("    </tr>\n");

                html.push_str("    <tr>\n      <td>F-statistic</td>\n");
                for (_, result) in &self.results {
                    html.push_str(&format!(
                        "      <td>{}</td>\n",
                        format_cell(result.f_statistic, 2)
                    ));
                }
                html.push_str("    </tr>\n");
            }

            html.push_str("  </tfoot>\n");
        }

        html.push_str("</table>\n");

        // Significance note
        if self.style.significance_stars {
            html.push_str("<p class=\"sig-note\"><small>");
            html.push_str("Standard errors in parentheses. ");
            html.push_str("* p &lt; 0.05, ** p &lt; 0.01, *** p &lt; 0.001");
            html.push_str("</small></p>\n");
        }

        html
    }

    /// Build a complete standalone HTML document.
    pub fn build_document(&self, title: &str) -> String {
        let table = self.build();
        format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{}</title>
    <style>
{}
    </style>
</head>
<body>
    <h1>{}</h1>
    {}
    <footer>
        <p><small>Generated by prompt2analytics</small></p>
    </footer>
</body>
</html>"#,
            escape_html(title),
            CSS_STYLES,
            escape_html(title),
            table
        )
    }
}

impl Default for HtmlTableBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate HTML for a DiscreteResult.
impl DiscreteResult {
    /// Export result to HTML table string.
    pub fn to_html(&self) -> String {
        let mut html = String::new();

        html.push_str(&embedded_css());

        html.push_str("<table class=\"regression-table\">\n");
        html.push_str(&format!(
            "  <caption>{} Regression Results</caption>\n",
            self.model_type
        ));

        // Header
        html.push_str("  <thead>\n    <tr>\n");
        html.push_str("      <th>Variable</th>\n");
        html.push_str("      <th>Coefficient</th>\n");
        html.push_str("      <th>Std. Error</th>\n");
        html.push_str("      <th>z</th>\n");
        html.push_str("      <th>P&gt;|z|</th>\n");
        html.push_str("      <th>Marginal Effect</th>\n");
        html.push_str("    </tr>\n  </thead>\n");

        // Body
        html.push_str("  <tbody>\n");
        for i in 0..self.variables.len() {
            let stars = significance_stars_html(&self.significance[i]);
            html.push_str("    <tr>\n");
            html.push_str(&format!(
                "      <td class=\"var-name\">{}</td>\n",
                escape_html(&self.variables[i])
            ));
            html.push_str(&format!(
                "      <td class=\"coef\">{}{}</td>\n",
                format_cell(self.coefficients[i], 4),
                stars
            ));
            html.push_str(&format!(
                "      <td class=\"se\">{}</td>\n",
                format_cell(self.std_errors[i], 4)
            ));
            html.push_str(&format!(
                "      <td>{}</td>\n",
                format_cell(self.z_stats[i], 2)
            ));
            html.push_str(&format!(
                "      <td>{}</td>\n",
                format_cell(self.p_values[i], 4)
            ));
            html.push_str(&format!(
                "      <td>{}</td>\n",
                format_cell(self.marginal_effects[i], 4)
            ));
            html.push_str("    </tr>\n");
        }
        html.push_str("  </tbody>\n");

        // Footer
        html.push_str("  <tfoot>\n");
        html.push_str(&format!(
            "    <tr><td colspan=\"6\">N = {}, Log-Likelihood = {}</td></tr>\n",
            self.n_obs,
            format_cell(self.log_likelihood, 4)
        ));
        html.push_str(&format!(
            "    <tr><td colspan=\"6\">Pseudo R<sup>2</sup> = {}, AIC = {}, BIC = {}</td></tr>\n",
            format_cell(self.pseudo_r_squared, 4),
            format_cell(self.aic, 2),
            format_cell(self.bic, 2)
        ));
        html.push_str(&notes_rows(self, 6));
        html.push_str("  </tfoot>\n");

        html.push_str("</table>\n");
        html.push_str("<p class=\"sig-note\"><small>* p &lt; 0.05, ** p &lt; 0.01, *** p &lt; 0.001</small></p>\n");

        html
    }
}

/// Generate HTML for a PanelResult.
impl PanelResult {
    /// Export result to HTML table string.
    pub fn to_html(&self) -> String {
        let mut html = String::new();

        html.push_str(&embedded_css());

        html.push_str("<table class=\"regression-table\">\n");
        html.push_str(&format!(
            "  <caption>{} Panel Regression</caption>\n",
            self.method
        ));

        // Header
        html.push_str("  <thead>\n    <tr>\n");
        html.push_str("      <th>Variable</th>\n");
        html.push_str("      <th>Coefficient</th>\n");
        html.push_str("      <th>Std. Error</th>\n");
        html.push_str("      <th>t</th>\n");
        html.push_str("      <th>P&gt;|t|</th>\n");
        html.push_str("    </tr>\n  </thead>\n");

        // Body
        html.push_str("  <tbody>\n");
        for i in 0..self.variables.len() {
            let stars = significance_stars_html(&self.significance[i]);
            html.push_str("    <tr>\n");
            html.push_str(&format!(
                "      <td class=\"var-name\">{}</td>\n",
                escape_html(&self.variables[i])
            ));
            html.push_str(&format!(
                "      <td class=\"coef\">{}{}</td>\n",
                format_cell(self.coefficients[i], 4),
                stars
            ));
            html.push_str(&format!(
                "      <td class=\"se\">{}</td>\n",
                format_cell(self.std_errors[i], 4)
            ));
            html.push_str(&format!(
                "      <td>{}</td>\n",
                format_cell(self.t_stats[i], 2)
            ));
            html.push_str(&format!(
                "      <td>{}</td>\n",
                format_cell(self.p_values[i], 4)
            ));
            html.push_str("    </tr>\n");
        }
        html.push_str("  </tbody>\n");

        // Footer
        html.push_str("  <tfoot>\n");
        html.push_str(&format!(
            "    <tr><td colspan=\"5\">N = {}, Groups = {}</td></tr>\n",
            self.n_obs, self.n_groups
        ));
        html.push_str(&format!(
            "    <tr><td colspan=\"5\">R<sup>2</sup> = {}, Adj. R<sup>2</sup> = {}</td></tr>\n",
            format_cell(self.r_squared, 4),
            format_cell(self.adj_r_squared, 4)
        ));
        html.push_str(&format!(
            "    <tr><td colspan=\"5\">F = {} (p = {})</td></tr>\n",
            format_cell(self.f_stat, 2),
            format_cell(self.f_p_value, 4)
        ));
        html.push_str(&notes_rows(self, 5));
        html.push_str("  </tfoot>\n");

        html.push_str("</table>\n");
        html.push_str("<p class=\"sig-note\"><small>* p &lt; 0.05, ** p &lt; 0.01, *** p &lt; 0.001</small></p>\n");

        html
    }
}

impl OlsResult {
    /// Export result to HTML table string.
    pub fn to_html(&self) -> String {
        HtmlTableBuilder::new()
            .add_model("(1)", self.clone())
            .build()
    }

    /// Export result to HTML with custom style.
    pub fn to_html_styled(&self, style: HtmlStyle) -> String {
        HtmlTableBuilder::new()
            .add_model("(1)", self.clone())
            .style(style)
            .build()
    }
}

/// Convert significance level to HTML superscript stars.
fn significance_stars_html(level: &SignificanceLevel) -> String {
    match level {
        SignificanceLevel::TenthPercent => "<sup>***</sup>".to_string(),
        SignificanceLevel::OnePercent => "<sup>**</sup>".to_string(),
        SignificanceLevel::FivePercent => "<sup>*</sup>".to_string(),
        SignificanceLevel::TenPercent => "<sup>&dagger;</sup>".to_string(),
        SignificanceLevel::NotSignificant => String::new(),
    }
}

/// The convergence line and warnings of a result (one line each), as footer rows.
fn notes_rows(result: &dyn ResultNotes, colspan: usize) -> String {
    notes_text(result)
        .lines()
        .map(|line| {
            format!(
                "    <tr><td colspan=\"{colspan}\">{}</td></tr>\n",
                escape_html(line)
            )
        })
        .collect()
}

/// Escape HTML special characters.
fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// Embedded CSS for standalone tables.
fn embedded_css() -> String {
    format!("<style>\n{}\n</style>\n", CSS_STYLES)
}

/// CSS styles for regression tables.
const CSS_STYLES: &str = r#"
.regression-table {
    border-collapse: collapse;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    font-size: 14px;
    margin: 1em 0;
    min-width: 400px;
}

.regression-table caption {
    font-weight: bold;
    font-size: 1.1em;
    padding: 0.5em;
    text-align: left;
}

.regression-table th,
.regression-table td {
    padding: 8px 12px;
    text-align: right;
    border-bottom: 1px solid #ddd;
}

.regression-table th {
    background-color: #f8f9fa;
    font-weight: 600;
    border-bottom: 2px solid #dee2e6;
}

.regression-table .var-name {
    text-align: left;
    font-weight: 500;
}

.regression-table .coef {
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
}

.regression-table .se {
    color: #6c757d;
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
}

.regression-table .se-row td {
    padding-top: 0;
    border-bottom: none;
}

.regression-table tfoot td {
    font-size: 0.9em;
    color: #6c757d;
    text-align: left;
}

.regression-table tfoot tr:first-child td {
    border-top: 2px solid #dee2e6;
}

.sig-note {
    color: #6c757d;
    margin-top: 0.5em;
}

.regression-table sup {
    color: #dc3545;
    font-weight: bold;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_html() {
        assert_eq!(escape_html("a < b"), "a &lt; b");
        assert_eq!(escape_html("a > b"), "a &gt; b");
        assert_eq!(escape_html("a & b"), "a &amp; b");
        assert_eq!(escape_html("\"quoted\""), "&quot;quoted&quot;");
    }

    #[test]
    fn test_significance_stars_html() {
        assert_eq!(
            significance_stars_html(&SignificanceLevel::TenthPercent),
            "<sup>***</sup>"
        );
        assert_eq!(
            significance_stars_html(&SignificanceLevel::OnePercent),
            "<sup>**</sup>"
        );
        assert_eq!(
            significance_stars_html(&SignificanceLevel::FivePercent),
            "<sup>*</sup>"
        );
        assert_eq!(
            significance_stars_html(&SignificanceLevel::NotSignificant),
            ""
        );
    }

    #[test]
    fn test_empty_builder() {
        let builder = HtmlTableBuilder::new();
        assert_eq!(builder.build(), "<p><em>No models to display</em></p>");
    }

    /// The rows of the table, each cell's inner text.
    fn html_rows(html: &str) -> Vec<Vec<String>> {
        html.split("<tr")
            .skip(1)
            .map(|row| {
                row.split("<td")
                    .skip(1)
                    .map(|cell| {
                        let inner = &cell[cell.find('>').unwrap() + 1..];
                        inner[..inner.find("</td>").unwrap()].to_string()
                    })
                    .collect()
            })
            .collect()
    }

    /// `-42.7144<sup>***</sup>` as the number and the stars.
    fn number_and_stars(cell: &str) -> (f64, String) {
        let (number, rest) = cell.split_once("<sup>").unwrap_or((cell, ""));
        let number = number.trim_matches(|c| matches!(c, '(' | ')'));
        (
            number.parse().unwrap(),
            rest.trim_end_matches("</sup>").to_string(),
        )
    }

    #[test]
    fn test_validate_html_table_ols_grunfeld() {
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
        let html = HtmlTableBuilder::new()
            .add_model("(1)", fit(&["value", "capital"]))
            .add_model("(2)", fit(&["value"]))
            .build();
        let rows = html_rows(&html);
        let row = |label: &str| {
            rows.iter()
                .position(|r| r.first().is_some_and(|c| c == label))
        };
        let mut check = Check::new(&case);
        for m in 0..2 {
            let tag = format!("m{}_", m + 1);
            let (mut names, mut coef, mut se) = (Vec::new(), Vec::new(), Vec::new());
            for name in ["(Intercept)", "value", "capital"] {
                let i = row(name).unwrap();
                if rows[i][m + 1].is_empty() {
                    continue;
                }
                let (c, stars) = number_and_stars(&rows[i][m + 1]);
                let p = case.named(&format!("{tag}p"), name);
                assert_eq!(stars, SignificanceLevel::from_p_value(p).stars(), "{name}");
                names.push(name);
                coef.push(c);
                se.push(number_and_stars(&rows[i + 1][m + 1]).0);
            }
            let footer = |label: &str| number_and_stars(&rows[row(label).unwrap()][m + 1]).0;
            check
                .named(&format!("{tag}coef"), &names, &coef, Tol::printed(4))
                .named(&format!("{tag}se"), &names, &se, Tol::printed(4))
                .scalar(
                    &format!("{tag}r2"),
                    footer("R<sup>2</sup>"),
                    Tol::printed(4),
                )
                .scalar(
                    &format!("{tag}adj_r2"),
                    footer("Adj. R<sup>2</sup>"),
                    Tol::printed(4),
                )
                .scalar(
                    &format!("{tag}f_stat"),
                    footer("F-statistic"),
                    Tol::printed(2),
                )
                .scalar("n_obs", footer("N"), Tol::exact());
        }
        check.finish();
    }
}
