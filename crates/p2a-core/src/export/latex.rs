//! LaTeX table export for regression results.
//!
//! Produces publication-ready LaTeX tables with:
//! - Significance stars (*, **, ***)
//! - Standard errors in parentheses
//! - Model statistics footer (N, R², F-stat)
//!
//! Model names, captions and variable names are plain text: LaTeX special
//! characters in them (`\ { } _ % & # $ ^ ~ < > |`, and a `[` or `*` at the
//! start of the text) are escaped and control characters become spaces, so
//! pass no LaTeX markup. Labels are restricted to `[A-Za-z0-9:._-]`.

use crate::econometrics::{DiscreteResult, PanelResult};
use crate::regression::OlsResult;
use crate::traits::SignificanceLevel;

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
                        " & {:.prec$}{}",
                        coef.estimate,
                        stars,
                        prec = self.style.coef_decimals
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
                            " & ({:.prec$})",
                            coef.std_error,
                            prec = self.style.se_decimals
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
                latex.push_str(&format!(" & {:.4}", result.r_squared));
            }
            latex.push_str(" \\\\\n");
        }

        if self.style.show_adj_r_squared {
            latex.push_str("Adj. $R^2$");
            for (_, result) in &self.results {
                latex.push_str(&format!(" & {:.4}", result.adj_r_squared));
            }
            latex.push_str(" \\\\\n");
        }

        if self.style.show_f_stat {
            latex.push_str("F-stat");
            for (_, result) in &self.results {
                latex.push_str(&format!(" & {:.2}", result.f_statistic));
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
/// Text whose first character after any spaces is `[` or `*` gets a leading
/// `{}`: the row break `\\` before a cell skips spaces looking for an
/// optional `*` or `[`, so such a cell would otherwise break the table
/// (`[`) or lose its first character (`*`).
fn escape_latex(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    if s.chars()
        .find(|&c| c != ' ' && !c.is_control())
        .is_some_and(|c| matches!(c, '[' | '*'))
    {
        out.push_str("{}");
    }
    for c in s.chars() {
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
                "{} & {:.4}{} & {:.4} & {:.2} & {:.4} \\\\\n",
                escape_latex(&self.variables[i]),
                self.coefficients[i],
                stars,
                self.std_errors[i],
                self.z_stats[i],
                self.p_values[i]
            ));
        }

        // Footer statistics
        latex.push_str("\\hline\n");
        latex.push_str(&format!(
            "N & \\multicolumn{{4}}{{c}}{{{}}} \\\\\n",
            self.n_obs
        ));
        latex.push_str(&format!(
            "Log-Likelihood & \\multicolumn{{4}}{{c}}{{{:.4}}} \\\\\n",
            self.log_likelihood
        ));
        latex.push_str(&format!(
            "Pseudo $R^2$ & \\multicolumn{{4}}{{c}}{{{:.4}}} \\\\\n",
            self.pseudo_r_squared
        ));
        latex.push_str(&format!(
            "AIC & \\multicolumn{{4}}{{c}}{{{:.2}}} \\\\\n",
            self.aic
        ));
        latex.push_str(&format!(
            "BIC & \\multicolumn{{4}}{{c}}{{{:.2}}} \\\\\n",
            self.bic
        ));

        latex.push_str("\\hline\\hline\n");
        latex.push_str("\\end{tabular}\n");

        // Significance note
        latex.push_str("\\begin{tablenotes}\\footnotesize\n");
        latex.push_str("\\item Standard errors in parentheses.\n");
        latex.push_str("\\item * p < 0.05, ** p < 0.01, *** p < 0.001\n");
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
                "{} & {:.4}{} & {:.4} & {:.2} & {:.4} \\\\\n",
                escape_latex(&self.variables[i]),
                self.coefficients[i],
                stars,
                self.std_errors[i],
                self.t_stats[i],
                self.p_values[i]
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
            "$R^2$ & \\multicolumn{{4}}{{c}}{{{:.4}}} \\\\\n",
            self.r_squared
        ));
        latex.push_str(&format!(
            "Adj. $R^2$ & \\multicolumn{{4}}{{c}}{{{:.4}}} \\\\\n",
            self.adj_r_squared
        ));
        latex.push_str(&format!(
            "F-stat & \\multicolumn{{4}}{{c}}{{{:.2}}} \\\\\n",
            self.f_stat
        ));

        latex.push_str("\\hline\\hline\n");
        latex.push_str("\\end{tabular}\n");

        // Significance note
        latex.push_str("\\begin{tablenotes}\\footnotesize\n");
        latex.push_str("\\item Standard errors in parentheses.\n");
        latex.push_str("\\item * p < 0.05, ** p < 0.01, *** p < 0.001\n");
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
        // either (after any leading spaces) gets an empty group in front.
        assert_eq!(escape_latex("[x]"), "{}[x]");
        assert_eq!(escape_latex("*z"), "{}*z");
        assert_eq!(escape_latex("  [x]"), "{}  [x]");
        assert_eq!(escape_latex("\t*z"), "{} *z");
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
}
