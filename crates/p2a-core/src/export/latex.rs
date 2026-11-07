//! LaTeX table export for regression results.
//!
//! Produces publication-ready LaTeX tables with:
//! - Significance stars (*, **, ***)
//! - Standard errors in parentheses
//! - Model statistics footer (N, R², F-stat)
//!
//! Model names, captions and variable names are plain text: LaTeX special
//! characters in them (`\ { } _ % & # $ ^ ~ < > |`) are escaped, a `[` or
//! `*` that opens the text (after any leading spaces) gets an empty group
//! `{}` in front of it, and control characters become spaces, so pass no
//! LaTeX markup. Labels are restricted to `[A-Za-z0-9:._-]`.
//!
//! Undefined values (NaN, e.g. aliased coefficients) print as `NA`, as R does.

use crate::econometrics::{DiscreteResult, PanelResult};
use crate::regression::OlsResult;
use crate::traits::SignificanceLevel;
use crate::traits::convergence::{ResultNotes, notes_text};

/// A table's number: NaN prints `NA`, +inf `Inf` and -inf `-Inf` (R's `print`), anything else
/// with fixed decimals.
pub(super) fn format_cell(x: f64, decimals: usize) -> String {
    if x.is_nan() {
        "NA".to_string()
    } else if x.is_infinite() {
        if x > 0.0 { "Inf" } else { "-Inf" }.to_string()
    } else {
        format!("{x:.decimals$}")
    }
}

/// The convergence line and warnings of a result (one line each), as table notes.
fn notes_items(result: &dyn ResultNotes) -> String {
    notes_text(result)
        .lines()
        .map(|line| format!("\\item {}\n", escape_latex(line)))
        .collect()
}

/// Style options for LaTeX tables.
#[derive(Debug, Clone)]
pub struct LatexStyle {
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
    /// Table caption (optional). Plain text: LaTeX special characters are
    /// escaped, so pass no LaTeX markup.
    pub caption: Option<String>,
    /// Table label for cross-references (optional). Every character outside
    /// `[A-Za-z0-9:._-]` is replaced by `-`; an empty label is left out.
    pub label: Option<String>,
}

impl Default for LatexStyle {
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
            label: None,
        }
    }
}

/// Builder for LaTeX regression tables.
#[derive(Debug)]
pub struct LatexTableBuilder {
    results: Vec<(String, OlsResult)>,
    style: LatexStyle,
}

impl LatexTableBuilder {
    /// Create a new LaTeX table builder.
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            style: LatexStyle::default(),
        }
    }

    /// Add a model result to the table.
    ///
    /// # Arguments
    /// * `name` - Column header for this model (e.g., "(1)", "OLS", "FE").
    ///   Plain text: LaTeX special characters are escaped, so pass no markup.
    /// * `result` - The OLS regression result. Its `variable_names` are
    ///   plain text and escaped the same way.
    pub fn add_model(mut self, name: impl Into<String>, result: OlsResult) -> Self {
        self.results.push((name.into(), result));
        self
    }

    /// Set the table style.
    pub fn style(mut self, style: LatexStyle) -> Self {
        self.style = style;
        self
    }

    /// Set the table caption.
    ///
    /// The caption is plain text: LaTeX special characters are escaped, so
    /// pass no LaTeX markup.
    pub fn caption(mut self, caption: impl Into<String>) -> Self {
        self.style.caption = Some(caption.into());
        self
    }

    /// Set the table label for cross-references.
    ///
    /// Every character outside `[A-Za-z0-9:._-]` is replaced by `-`; an empty
    /// label is left out.
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.style.label = Some(label.into());
        self
    }

    /// Build the LaTeX table string.
    pub fn build(&self) -> String {
        if self.results.is_empty() {
            return String::from("% No models to display");
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

        let mut latex = String::new();

        // Begin table
        latex.push_str("\\begin{table}[htbp]\n");
        latex.push_str("\\centering\n");

        // Caption
        if let Some(cap) = &self.style.caption {
            latex.push_str(&format!("\\caption{{{}}}\n", escape_latex(cap)));
        }

        // Label
        if let Some(lab) = self.style.label.as_deref().and_then(sanitize_label) {
            latex.push_str(&format!("\\label{{{}}}\n", lab));
        }

        // Begin tabular with column alignment
        let col_spec = format!("l{}", "c".repeat(n_models));
        latex.push_str(&format!("\\begin{{tabular}}{{{}}}\n", col_spec));
        latex.push_str("\\hline\\hline\n");

        // Header row
        latex.push_str("Variable");
        for (name, _) in &self.results {
            latex.push_str(&format!(" & {}", escape_latex(name)));
        }
        latex.push_str(" \\\\\n");
        latex.push_str("\\hline\n");

        // Coefficient rows
        for var in &all_vars {
            // Coefficient row
            latex.push_str(&escape_latex(var));
            for (_, result) in &self.results {
                if let Some(idx) = result.variable_names.iter().position(|v| v == var) {
                    let coef = &result.coefficients[idx];
                    let stars = if self.style.significance_stars {
                        significance_stars(&coef.significance)
                    } else {
                        String::new()
                    };
                    latex.push_str(&format!(
                        " & {}{}",
                        format_cell(coef.estimate, self.style.coef_decimals),
                        stars
                    ));
                } else {
                    latex.push_str(" & ");
                }
            }
            latex.push_str(" \\\\\n");

            // Standard error row (if enabled)
            if self.style.se_in_parentheses {
                latex.push(' ');
                for (_, result) in &self.results {
                    if let Some(idx) = result.variable_names.iter().position(|v| v == var) {
                        let coef = &result.coefficients[idx];
                        latex.push_str(&format!(
                            " & ({})",
                            format_cell(coef.std_error, self.style.se_decimals)
                        ));
                    } else {
                        latex.push_str(" & ");
                    }
                }
                latex.push_str(" \\\\\n");
            }
        }

        // Footer statistics
        latex.push_str("\\hline\n");

        if self.style.show_n {
            latex.push('N');
            for (_, result) in &self.results {
                latex.push_str(&format!(" & {}", result.n_obs));
            }
            latex.push_str(" \\\\\n");
        }

        if self.style.show_r_squared {
            latex.push_str("$R^2$");
            for (_, result) in &self.results {
                latex.push_str(&format!(" & {}", format_cell(result.r_squared, 4)));
            }
            latex.push_str(" \\\\\n");
        }

        if self.style.show_adj_r_squared {
            latex.push_str("Adj. $R^2$");
            for (_, result) in &self.results {
                latex.push_str(&format!(" & {}", format_cell(result.adj_r_squared, 4)));
            }
            latex.push_str(" \\\\\n");
        }

        if self.style.show_f_stat {
            latex.push_str("F-stat");
            for (_, result) in &self.results {
                latex.push_str(&format!(" & {}", format_cell(result.f_statistic, 2)));
            }
            latex.push_str(" \\\\\n");
        }

        latex.push_str("\\hline\\hline\n");

        // End tabular
        latex.push_str("\\end{tabular}\n");

        // Significance note
        if self.style.significance_stars {
            latex.push_str("\\begin{tablenotes}\\footnotesize\n");
            latex.push_str("\\item Standard errors in parentheses.\n");
            latex.push_str("\\item * p < 0.1, ** p < 0.05, *** p < 0.01\n");
            latex.push_str("\\end{tablenotes}\n");
        }

        latex.push_str("\\end{table}\n");

        latex
    }
}

impl Default for LatexTableBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert significance level to stars.
fn significance_stars(level: &SignificanceLevel) -> String {
    match level {
        SignificanceLevel::TenthPercent => "***".to_string(), // p < 0.001
        SignificanceLevel::OnePercent => "**".to_string(),    // p < 0.01
        SignificanceLevel::FivePercent => "*".to_string(),    // p < 0.05
        SignificanceLevel::TenPercent => "†".to_string(),     // p < 0.10
        SignificanceLevel::NotSignificant => String::new(),
    }
}

/// Escape text for LaTeX in a single pass over its characters.
///
/// A chain of replacements would re-escape the braces that
/// `\textbackslash{}` introduces. Control characters become spaces.
///
/// Text whose first character after any spaces is `[` or `*` gets `{}`
/// directly before that character: the row break `\\` before a cell skips
/// spaces looking for an optional `*` or `[`, so such a cell would otherwise
/// break the table (`[`) or lose its first character (`*`). Leading spaces
/// stay ahead of the `{}`, where the cell's `\ignorespaces` still drops them.
fn escape_latex(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    let mut seen_text = false;
    for c in s.chars() {
        if !seen_text && c != ' ' && !c.is_control() {
            seen_text = true;
            if matches!(c, '[' | '*') {
                out.push_str("{}");
            }
        }
        match c {
            '\\' => out.push_str("\\textbackslash{}"),
            '{' => out.push_str("\\{"),
            '}' => out.push_str("\\}"),
            '_' => out.push_str("\\_"),
            '%' => out.push_str("\\%"),
            '&' => out.push_str("\\&"),
            '#' => out.push_str("\\#"),
            '$' => out.push_str("\\$"),
            '^' => out.push_str("\\textasciicircum{}"),
            '~' => out.push_str("\\textasciitilde{}"),
            '<' => out.push_str("\\textless{}"),
            '>' => out.push_str("\\textgreater{}"),
            '|' => out.push_str("\\textbar{}"),
            c if c.is_control() => out.push(' '),
            c => out.push(c),
        }
    }
    out
}

/// Restrict a `\label` key to `[A-Za-z0-9:._-]`, mapping any other character
/// to `-`. Returns `None` for an empty label.
fn sanitize_label(s: &str) -> Option<String> {
    if s.is_empty() {
        return None;
    }
    Some(
        s.chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || matches!(c, ':' | '.' | '_' | '-') {
                    c
                } else {
                    '-'
                }
            })
            .collect(),
    )
}

impl OlsResult {
    /// Export result to LaTeX table string.
    ///
    /// For multi-model tables, use `LatexTableBuilder`.
    pub fn to_latex(&self) -> String {
        LatexTableBuilder::new()
            .add_model("(1)", self.clone())
            .build()
    }

    /// Export result to LaTeX table with custom style.
    pub fn to_latex_styled(&self, style: LatexStyle) -> String {
        LatexTableBuilder::new()
            .add_model("(1)", self.clone())
            .style(style)
            .build()
    }
}

impl DiscreteResult {
    /// Export result to LaTeX table string.
    pub fn to_latex(&self) -> String {
        let mut latex = String::new();

        // Begin table
        latex.push_str("\\begin{table}[htbp]\n");
        latex.push_str("\\centering\n");
        latex.push_str(&format!(
            "\\caption{{{} Regression Results}}\n",
            escape_latex(&self.model_type.to_string())
        ));

        // Begin tabular
        latex.push_str("\\begin{tabular}{lcccc}\n");
        latex.push_str("\\hline\\hline\n");

        // Header row
        latex.push_str("Variable & Coefficient & Std. Error & z & P$>|$z$|$ \\\\\n");
        latex.push_str("\\hline\n");

        // Coefficient rows
        for i in 0..self.variables.len() {
            let stars = significance_stars(&self.significance[i]);
            latex.push_str(&format!(
                "{} & {}{} & {} & {} & {} \\\\\n",
                escape_latex(&self.variables[i]),
                format_cell(self.coefficients[i], 4),
                stars,
                format_cell(self.std_errors[i], 4),
                format_cell(self.z_stats[i], 2),
                format_cell(self.p_values[i], 4)
            ));
        }

        // Footer statistics
        latex.push_str("\\hline\n");
        latex.push_str(&format!(
            "N & \\multicolumn{{4}}{{c}}{{{}}} \\\\\n",
            self.n_obs
        ));
        latex.push_str(&format!(
            "Log-Likelihood & \\multicolumn{{4}}{{c}}{{{}}} \\\\\n",
            format_cell(self.log_likelihood, 4)
        ));
        latex.push_str(&format!(
            "Pseudo $R^2$ & \\multicolumn{{4}}{{c}}{{{}}} \\\\\n",
            format_cell(self.pseudo_r_squared, 4)
        ));
        latex.push_str(&format!(
            "AIC & \\multicolumn{{4}}{{c}}{{{}}} \\\\\n",
            format_cell(self.aic, 2)
        ));
        latex.push_str(&format!(
            "BIC & \\multicolumn{{4}}{{c}}{{{}}} \\\\\n",
            format_cell(self.bic, 2)
        ));

        latex.push_str("\\hline\\hline\n");
        latex.push_str("\\end{tabular}\n");

        // Significance note
        latex.push_str("\\begin{tablenotes}\\footnotesize\n");
        latex.push_str("\\item Standard errors in parentheses.\n");
        latex.push_str("\\item * p < 0.05, ** p < 0.01, *** p < 0.001\n");
        latex.push_str(&notes_items(self));
        latex.push_str("\\end{tablenotes}\n");

        latex.push_str("\\end{table}\n");

        latex
    }
}

impl PanelResult {
    /// Export result to LaTeX table string.
    pub fn to_latex(&self) -> String {
        let mut latex = String::new();

        // Begin table
        latex.push_str("\\begin{table}[htbp]\n");
        latex.push_str("\\centering\n");
        latex.push_str(&format!(
            "\\caption{{{} Panel Regression Results}}\n",
            escape_latex(&self.method.to_string())
        ));

        // Begin tabular
        latex.push_str("\\begin{tabular}{lcccc}\n");
        latex.push_str("\\hline\\hline\n");

        // Header row
        latex.push_str("Variable & Coefficient & Std. Error & t & P$>|$t$|$ \\\\\n");
        latex.push_str("\\hline\n");

        // Coefficient rows
        for i in 0..self.variables.len() {
            let stars = significance_stars(&self.significance[i]);
            latex.push_str(&format!(
                "{} & {}{} & {} & {} & {} \\\\\n",
                escape_latex(&self.variables[i]),
                format_cell(self.coefficients[i], 4),
                stars,
                format_cell(self.std_errors[i], 4),
                format_cell(self.t_stats[i], 2),
                format_cell(self.p_values[i], 4)
            ));
        }

        // Footer statistics
        latex.push_str("\\hline\n");
        latex.push_str(&format!(
            "N & \\multicolumn{{4}}{{c}}{{{}}} \\\\\n",
            self.n_obs
        ));
        latex.push_str(&format!(
            "Groups & \\multicolumn{{4}}{{c}}{{{}}} \\\\\n",
            self.n_groups
        ));
        latex.push_str(&format!(
            "$R^2$ & \\multicolumn{{4}}{{c}}{{{}}} \\\\\n",
            format_cell(self.r_squared, 4)
        ));
        latex.push_str(&format!(
            "Adj. $R^2$ & \\multicolumn{{4}}{{c}}{{{}}} \\\\\n",
            format_cell(self.adj_r_squared, 4)
        ));
        latex.push_str(&format!(
            "F-stat & \\multicolumn{{4}}{{c}}{{{}}} \\\\\n",
            format_cell(self.f_stat, 2)
        ));

        latex.push_str("\\hline\\hline\n");
        latex.push_str("\\end{tabular}\n");

        // Significance note
        latex.push_str("\\begin{tablenotes}\\footnotesize\n");
        latex.push_str("\\item Standard errors in parentheses.\n");
        latex.push_str("\\item * p < 0.05, ** p < 0.01, *** p < 0.001\n");
        latex.push_str(&notes_items(self));
        latex.push_str("\\end{tablenotes}\n");

        latex.push_str("\\end{table}\n");

        latex
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regression::{CovarianceType, run_ols_raw};
    use ndarray::{Array1, Array2};

    fn ols_with_names(names: [&str; 2]) -> OlsResult {
        let x = Array2::from_shape_fn((8, 2), |(i, j)| if j == 0 { 1.0 } else { (i + 1) as f64 });
        let y = Array1::from(vec![1.1, 1.9, 3.2, 3.8, 5.1, 6.2, 6.8, 8.1]);
        let names: Vec<String> = names.iter().map(|s| s.to_string()).collect();
        run_ols_raw(&x, &y, &names, "y", CovarianceType::HC1).expect("ols")
    }

    fn braces_balanced(tex: &str) -> bool {
        let s = tex.replace("\\{", "").replace("\\}", "");
        let mut depth = 0i64;
        for c in s.chars() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth < 0 {
                        return false;
                    }
                }
                _ => {}
            }
        }
        depth == 0
    }

    #[test]
    fn test_escape_latex() {
        assert_eq!(escape_latex("x_1"), "x\\_1");
        assert_eq!(escape_latex("100%"), "100\\%");
        assert_eq!(escape_latex("a & b"), "a \\& b");
    }

    #[test]
    fn test_escape_latex_complete() {
        assert_eq!(escape_latex("a\\b"), "a\\textbackslash{}b");
        assert_eq!(escape_latex("{x}"), "\\{x\\}");
        assert_eq!(
            escape_latex("x^2~y"),
            "x\\textasciicircum{}2\\textasciitilde{}y"
        );
        assert_eq!(escape_latex("a<b>c"), "a\\textless{}b\\textgreater{}c");
        assert_eq!(escape_latex("a|b"), "a\\textbar{}b");
        assert_eq!(escape_latex("\\input{f}"), "\\textbackslash{}input\\{f\\}");
        assert_eq!(escape_latex("a\nb\tc"), "a b c");
        assert_eq!(escape_latex("#1 $5"), "\\#1 \\$5");
    }

    #[test]
    fn test_escape_latex_guards_leading_bracket_and_star() {
        // `\\` looks past spaces for `*` or `[`, so a cell that opens with
        // either (after any leading spaces) gets an empty group directly in
        // front of it; leading spaces stay ahead of the group, where the
        // cell's `\ignorespaces` still drops them.
        assert_eq!(escape_latex("[x]"), "{}[x]");
        assert_eq!(escape_latex("*z"), "{}*z");
        assert_eq!(escape_latex("  [x]"), "  {}[x]");
        assert_eq!(escape_latex("\t*z"), " {}*z");
        assert_eq!(escape_latex("[_x]"), "{}[\\_x]");
        // a bracket or star anywhere else is left alone
        assert_eq!(escape_latex("a[1]"), "a[1]");
        assert_eq!(escape_latex("x*y"), "x*y");
        assert_eq!(escape_latex(""), "");
        assert_eq!(escape_latex("   "), "   ");
    }

    #[test]
    fn test_build_guards_rows_starting_with_bracket_or_star() {
        for (name, row) in [("[x]", "{}[x] & "), ("*z", "{}*z & ")] {
            let tex = LatexTableBuilder::new()
                .add_model("(1)", ols_with_names(["(Intercept)", name]))
                .build();
            assert!(tex.lines().any(|l| l.starts_with(row)), "{tex}");
            assert!(braces_balanced(&tex), "{tex}");
        }
    }

    #[test]
    fn test_build_escapes_names_caption_label() {
        let tex = LatexTableBuilder::new()
            .add_model("OLS & HC1", ols_with_names(["(Intercept)", "cap^2{x}"]))
            .add_model(
                "FE (50% sample)",
                ols_with_names(["(Intercept)", "cap^2{x}"]),
            )
            .caption("Wages & {hours}\\")
            .label("tab:main}\\input{x}")
            .build();
        assert!(
            tex.contains("\\caption{Wages \\& \\{hours\\}\\textbackslash{}}\n"),
            "{tex}"
        );
        assert!(tex.contains("\\label{tab:main--input-x-}\n"), "{tex}");
        assert!(
            tex.contains("Variable & OLS \\& HC1 & FE (50\\% sample) \\\\\n"),
            "{tex}"
        );
        assert!(tex.contains("cap\\textasciicircum{}2\\{x\\}"), "{tex}");
        assert!(!tex.contains("\\input{"), "{tex}");
        assert!(braces_balanced(&tex), "{tex}");
    }

    #[test]
    fn test_ordinary_names_unchanged() {
        // passes before and after
        let tex = LatexTableBuilder::new()
            .add_model("(1)", ols_with_names(["(Intercept)", "x_1"]))
            .add_model("OLS", ols_with_names(["(Intercept)", "x_1"]))
            .caption("Regression Results")
            .label("tab:results")
            .build();
        assert!(tex.contains("\\caption{Regression Results}\n\\label{tab:results}\n"));
        assert!(tex.contains("Variable & (1) & OLS \\\\\n"));
        assert!(tex.lines().any(|l| l.starts_with("x\\_1 & ")));
        assert!(tex.lines().any(|l| l.starts_with("(Intercept) & ")));
    }

    #[test]
    fn test_significance_stars() {
        assert_eq!(significance_stars(&SignificanceLevel::TenthPercent), "***");
        assert_eq!(significance_stars(&SignificanceLevel::OnePercent), "**");
        assert_eq!(significance_stars(&SignificanceLevel::FivePercent), "*");
        assert_eq!(significance_stars(&SignificanceLevel::TenPercent), "†");
        assert_eq!(significance_stars(&SignificanceLevel::NotSignificant), "");
    }

    /// Grunfeld, read directly: these tests pin no R case.
    fn grunfeld() -> crate::data::Dataset {
        let df = crate::data::DataLoader::load_csv(p2a_validation::repo_path(
            "validation/datasets/grunfeld.csv",
        ))
        .unwrap();
        crate::data::Dataset::new(df)
    }

    /// The two Grunfeld models of case `export_ols_grunfeld`.
    fn two_models(ds: &crate::data::Dataset) -> (OlsResult, OlsResult) {
        let fit = |x: &[&str]| {
            crate::regression::run_ols(ds, "inv", x, true, CovarianceType::Standard).unwrap()
        };
        (fit(&["value", "capital"]), fit(&["value"]))
    }

    #[test]
    fn test_export_tables_print_undefined_values_as_na() {
        use crate::export::{HtmlTableBuilder, MarkdownTableBuilder};
        let ds = grunfeld();
        let (mut m, _) = two_models(&ds);
        let c = &mut m.coefficients[2];
        (c.estimate, c.std_error, c.t_value, c.p_value) = (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
        c.significance = SignificanceLevel::NotSignificant;
        let tex = LatexTableBuilder::new().add_model("(1)", m.clone()).build();
        assert!(tex.contains("capital & NA \\\\\n  & (NA) \\\\\n"), "{tex}");
        let html = HtmlTableBuilder::new().add_model("(1)", m.clone()).build();
        assert!(
            html.contains("<td class=\"coef\">NA</td>")
                && html.contains("<td class=\"se\">(NA)</td>"),
            "{html}"
        );
        let md = MarkdownTableBuilder::new().add_model("(1)", m).build();
        assert!(md.contains("| capital | NA |\n| | (NA) |\n"), "{md}");

        // A fit that really has an aliased column: the row stays and prints NA.
        let frame = polars::prelude::df!(
            "y" => [1.1, 1.9, 3.2, 3.8, 5.1, 6.2, 6.8, 8.1],
            "x" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0],
            "dup" => [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 18.0]
        )
        .unwrap();
        let aliased = crate::regression::run_ols(
            &crate::data::Dataset::new(frame),
            "y",
            &["x", "dup"],
            true,
            CovarianceType::Standard,
        )
        .unwrap();
        assert_eq!(aliased.aliased, ["dup"]);
        let tex = LatexTableBuilder::new()
            .add_model("(1)", aliased.clone())
            .build();
        assert!(tex.contains("dup & NA \\\\\n  & (NA) \\\\\n"), "{tex}");
        let html = HtmlTableBuilder::new()
            .add_model("(1)", aliased.clone())
            .build();
        assert!(html.contains("<td class=\"coef\">NA</td>"), "{html}");
        let md = MarkdownTableBuilder::new()
            .add_model("(1)", aliased)
            .build();
        assert!(md.contains("| dup | NA |\n| | (NA) |\n"), "{md}");

        let mut fe =
            crate::econometrics::run_fixed_effects(&ds, "inv", &["value", "capital"], "firm")
                .unwrap();
        let j = fe.variables.iter().position(|v| v == "capital").unwrap();
        (
            fe.coefficients[j],
            fe.std_errors[j],
            fe.t_stats[j],
            fe.p_values[j],
        ) = (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
        fe.significance[j] = SignificanceLevel::NotSignificant;
        assert!(
            fe.to_latex().contains("capital & NA & NA & NA & NA \\\\"),
            "{}",
            fe.to_latex()
        );
        assert!(fe.to_html().contains("<td class=\"coef\">NA</td>"));
        // A panel fit's warnings reach both table footers.
        let frame = polars::prelude::df!(
            "firm" => [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "y" => [1.1, 2.3, 2.9, 4.2, 2.2, 2.8, 4.1, 5.3, 3.3, 3.9, 5.2, 5.8],
            "x" => [1.0, 2.0, 3.0, 4.5, 1.5, 2.5, 3.0, 4.0, 1.0, 2.5, 3.5, 4.0],
            "region" => [7.0, 7.0, 7.0, 7.0, 9.0, 9.0, 9.0, 9.0, 4.0, 4.0, 4.0, 4.0]
        )
        .unwrap();
        let warned = crate::econometrics::run_fixed_effects(
            &crate::data::Dataset::new(frame),
            "y",
            &["x", "region"],
            "firm",
        )
        .unwrap();
        assert!(!warned.warnings.is_empty());
        let tex = warned.to_latex();
        assert!(tex.contains("\\item Warning:"), "{tex}");
        let html = warned.to_html();
        assert!(html.contains("<td colspan=\"5\">Warning:"), "{html}");
        assert_eq!(
            (
                format_cell(f64::INFINITY, 2),
                format_cell(f64::NEG_INFINITY, 2)
            ),
            ("Inf".to_string(), "-Inf".to_string())
        );
        assert_eq!(
            (format_cell(-0.126, 2), format_cell(1.0, 4)),
            ("-0.13".to_string(), "1.0000".to_string())
        );

        // DiscreteResult: a non-separated logit fit, then a NaN row.
        let d = crate::data::Dataset::new(
            polars::prelude::df!(
                "y" => [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                "x" => [0.1, 0.4, 0.35, 0.8, 0.9, 1.2, 0.5, 1.5, 1.1, 2.0, 0.3, 1.7]
            )
            .unwrap(),
        );
        let mut lg = crate::econometrics::run_logit(&d, "y", &["x"]).unwrap();
        let tex = lg.to_latex();
        assert!(tex.contains("\\item Converged: yes"), "{tex}");
        assert!(lg.to_html().contains("Converged: yes"));
        let j = lg.variables.iter().position(|v| v == "x").unwrap();
        (
            lg.coefficients[j],
            lg.std_errors[j],
            lg.z_stats[j],
            lg.p_values[j],
            lg.marginal_effects[j],
        ) = (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
        lg.significance[j] = SignificanceLevel::NotSignificant;
        assert!(
            lg.to_latex().contains("x & NA & NA & NA & NA \\\\"),
            "{}",
            lg.to_latex()
        );
        assert!(
            lg.to_html().contains("<td class=\"coef\">NA</td>")
                && lg.to_html().contains("<td>NA</td>")
        );
    }

    /// The tabular's rows, split into trimmed cells.
    fn latex_rows(tex: &str) -> Vec<Vec<String>> {
        tex.lines()
            .skip_while(|l| !l.starts_with("\\begin{tabular}"))
            .take_while(|l| !l.starts_with("\\end{tabular}"))
            .filter_map(|l| l.strip_suffix(" \\\\"))
            .map(|l| l.split(" & ").map(|c| c.trim().to_string()).collect())
            .collect()
    }

    fn cell_number(c: &str) -> f64 {
        c.trim_matches(|ch| matches!(ch, '(' | ')' | '*' | '†'))
            .parse()
            .unwrap_or_else(|_| panic!("not a number: {c:?}"))
    }

    fn cell_stars(c: &str) -> String {
        c.chars().filter(|ch| matches!(ch, '*' | '†')).collect()
    }

    #[test]
    fn test_validate_latex_table_ols_grunfeld() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("export_ols_grunfeld");
        let (m1, m2) = two_models(&ds);
        let tex = LatexTableBuilder::new()
            .add_model("(1)", m1)
            .add_model("(2)", m2)
            .build();
        let rows = latex_rows(&tex);
        let names = ["(Intercept)", "value", "capital"];
        let row = |label: &str| rows.iter().position(|r| r[0] == label).unwrap();
        let mut coef = [Vec::new(), Vec::new()];
        let mut se = [Vec::new(), Vec::new()];
        let mut shown = [Vec::new(), Vec::new()];
        for name in names {
            let i = row(name);
            for m in 0..2 {
                let cell = rows[i].get(m + 1).map_or("", String::as_str);
                if cell.is_empty() {
                    continue;
                }
                coef[m].push(cell_number(cell));
                se[m].push(cell_number(&rows[i + 1][m + 1]));
                shown[m].push(name);
                let p = case.named(&format!("m{}_p", m + 1), name);
                assert_eq!(
                    cell_stars(cell),
                    SignificanceLevel::from_p_value(p).stars(),
                    "{name}"
                );
            }
        }
        assert_eq!(shown[1], ["(Intercept)", "value"]);
        let footer = |label: &str, m: usize| cell_number(&rows[row(label)][m]);
        assert_eq!(footer("N", 1), footer("N", 2));
        let mut check = Check::new(&case);
        check.scalar("n_obs", footer("N", 1), Tol::exact());
        for m in 0..2 {
            let tag = format!("m{}_", m + 1);
            check
                .named(&format!("{tag}coef"), &shown[m], &coef[m], Tol::printed(4))
                .named(&format!("{tag}se"), &shown[m], &se[m], Tol::printed(4))
                .scalar(&format!("{tag}r2"), footer("$R^2$", m + 1), Tol::printed(4))
                .scalar(
                    &format!("{tag}adj_r2"),
                    footer("Adj. $R^2$", m + 1),
                    Tol::printed(4),
                )
                .scalar(
                    &format!("{tag}f_stat"),
                    footer("F-stat", m + 1),
                    Tol::printed(2),
                );
        }
        check.finish();
    }

    #[test]
    fn test_latex_table_golden_grunfeld() {
        let (m1, m2) = two_models(&grunfeld());
        let tex = LatexTableBuilder::new()
            .add_model("(1)", m1)
            .add_model("(2)", m2)
            .caption("Investment & firm value (Größe_1)")
            .label("tab:grunfeld")
            .build();
        p2a_validation::golden::assert_golden(
            &p2a_validation::repo_path(
                "crates/p2a-core/tests/golden/export_latex_ols_grunfeld.tex",
            ),
            &tex,
        );
    }

    #[test]
    #[ignore = "needs pdflatex; run with: cargo test -p p2a-core --lib -- --ignored export::latex"]
    fn test_latex_table_compiles_with_pdflatex() {
        let table = std::fs::read_to_string(p2a_validation::repo_path(
            "crates/p2a-core/tests/golden/export_latex_ols_grunfeld.tex",
        ))
        .unwrap();
        // DATA-56: the exporter does not declare that its notes need threeparttable.
        let doc = format!(
            "\\documentclass{{article}}\\usepackage[utf8]{{inputenc}}\\usepackage{{threeparttable}}\
             \\begin{{document}}\n{table}\\end{{document}}\n"
        );
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("doc.tex"), doc).unwrap();
        let out = std::process::Command::new("pdflatex")
            .args(["-interaction=nonstopmode", "-halt-on-error", "doc.tex"])
            .current_dir(dir.path())
            .output()
            .expect("pdflatex runs");
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stdout)
        );
    }
}
