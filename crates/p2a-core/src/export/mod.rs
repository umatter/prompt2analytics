//! Export functionality for regression and statistical results.
//!
//! This module provides four export formats for publication-ready output of
//! regression results, statistical tests, and econometric analyses.
//!
//! ## Export Formats
//!
//! | Format | Builder | Use Case |
//! |--------|---------|----------|
//! | **LaTeX** | [`LatexTableBuilder`] | Academic papers, journal submissions |
//! | **Markdown** | [`MarkdownTableBuilder`] | Documentation, GitHub READMEs |
//! | **HTML** | [`HtmlTableBuilder`] | Web display, reports |
//! | **CSV** | [`CsvExport`] trait | Data interchange, spreadsheets |
//!
//! ## LaTeX Tables
//!
//! Publication-ready LaTeX tables with customizable styling:
//! - Standard errors in parentheses
//! - Significance stars (`\textdagger{}` p < 0.10, * p < 0.05, ** p < 0.01, *** p < 0.001), with a
//!   legend built from the same mapping
//! - Multi-model comparison tables
//! - Custom notes and captions
//!
//! A table with notes puts them in a `threeparttable` environment and says so in its first
//! line (`% Requires \usepackage{threeparttable}`): the document must load that package.
//!
//! ```ignore
//! use p2a_core::export::{LatexTableBuilder, LatexStyle};
//! use p2a_core::regression::OlsResult;
//!
//! // Given an OlsResult from regression
//! let latex = LatexTableBuilder::new()
//!     .add_model("Model 1", ols_result)
//!     .caption("Regression Results")
//!     .label("tab:results")
//!     .build();
//! ```
//!
//! ## Markdown Tables
//!
//! GitHub-flavored markdown tables for documentation:
//!
//! ```ignore
//! use p2a_core::export::MarkdownTableBuilder;
//! use p2a_core::regression::OlsResult;
//!
//! // Given an OlsResult from regression
//! let md = MarkdownTableBuilder::new()
//!     .add_model("OLS", ols_result)
//!     .build();
//! ```
//!
//! ## HTML Tables
//!
//! Self-contained HTML with embedded CSS styling:
//!
//! ```ignore
//! use p2a_core::export::HtmlTableBuilder;
//! use p2a_core::regression::OlsResult;
//!
//! // Given an OlsResult from regression
//! let html = HtmlTableBuilder::new()
//!     .add_model("Results", ols_result)
//!     .build();
//! ```
//!
//! ## CSV Export
//!
//! The [`CsvExport`] trait is implemented for the result types listed below:
//!
//! ```ignore
//! use p2a_core::export::CsvExport;
//! use p2a_core::regression::OlsResult;
//!
//! // Given an OlsResult from regression
//! ols_result.to_csv("ols.csv")?;  // writes the CSV to `path`
//! let csv = ols_result.to_csv_string();  // or returns it as a string
//! ```
//!
//! ## Supported Result Types
//!
//! - The builders ([`LatexTableBuilder`], [`HtmlTableBuilder`], [`MarkdownTableBuilder`]) take
//!   [`OlsResult`](crate::regression::OlsResult).
//! - [`PanelResult`](crate::econometrics::PanelResult) and
//!   [`DiscreteResult`](crate::econometrics::DiscreteResult) have `to_latex` and `to_html`.
//! - [`CsvExport`] is implemented for `OlsResult`, `DiscreteResult`, `PanelResult`,
//!   `HausmanResult`, `PCAResult`, `KMeansResult` and `DBSCANResult`.
//!
//! `IVResult` has no export.

mod csv;
mod html;
mod latex;
mod markdown;

pub use csv::CsvExport;
pub use html::{HtmlStyle, HtmlTableBuilder};
pub use latex::{LatexStyle, LatexTableBuilder};
pub use markdown::{MarkdownStyle, MarkdownTableBuilder};

use crate::traits::SignificanceLevel;
use latex::format_cell;

/// An estimate, standard-error or marginal-effect cell of a regression table.
///
/// As [`format_cell`] (fixed `decimals`, `NA`, `Inf`, `-Inf`), except that a finite, non-zero
/// value whose fixed-decimal form would read as zero is written with `decimals` significant
/// digits in scientific notation with an exponent of at least two digits, as R writes it
/// (`1.156e-07`). Unlike R's `format`, trailing zeros are kept, so a column's cells line up.
pub(super) fn format_estimate(x: f64, decimals: usize) -> String {
    let fixed = format_cell(x, decimals);
    if !x.is_finite() || x == 0.0 || fixed.parse::<f64>().is_ok_and(|v| v != 0.0) {
        return fixed;
    }
    let s = format!("{x:.prec$e}", prec = decimals.max(1) - 1);
    let Some((mantissa, exponent)) = s.split_once('e') else {
        return s;
    };
    let (sign, digits) = match exponent.strip_prefix('-') {
        Some(d) => ('-', d),
        None => ('+', exponent),
    };
    format!("{mantissa}e{sign}{digits:0>2}")
}

/// How a table writes its significance legend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LegendStyle {
    /// `\textdagger{} $p<0.10$; * $p<0.05$; ...` (inside a `tablenotes` item).
    Latex,
    /// `&dagger; p &lt; 0.10, * p &lt; 0.05, ...`.
    Html,
    /// `† p < 0.10, \* p < 0.05, ...` (stars escaped).
    Markdown,
}

/// The levels a table marks, weakest first.
const LEGEND_LEVELS: [SignificanceLevel; 4] = [
    SignificanceLevel::TenPercent,
    SignificanceLevel::FivePercent,
    SignificanceLevel::OnePercent,
    SignificanceLevel::TenthPercent,
];

/// The significance legend, built from the mapping that assigns the stars
/// ([`SignificanceLevel::stars`] and the thresholds of [`SignificanceLevel::code`]), so the
/// legend and the stars cannot disagree.
pub(super) fn significance_legend(style: LegendStyle) -> String {
    let items: Vec<String> = LEGEND_LEVELS
        .iter()
        .map(|level| {
            let threshold = level.code().trim_start_matches("p<");
            let stars = level.stars();
            match style {
                LegendStyle::Latex => {
                    let symbol = stars.replace('†', "\\textdagger{}");
                    format!("{symbol} $p<{threshold}$")
                }
                LegendStyle::Html => {
                    let symbol = stars.replace('†', "&dagger;");
                    format!("{symbol} p &lt; {threshold}")
                }
                LegendStyle::Markdown => {
                    let symbol = stars.replace('*', "\\*");
                    format!("{symbol} p < {threshold}")
                }
            }
        })
        .collect();
    let separator = if style == LegendStyle::Latex {
        "; "
    } else {
        ", "
    };
    items.join(separator)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_estimate() {
        assert_eq!(format_estimate(0.0, 4), "0.0000");
        assert_eq!(format_estimate(0.00005, 4), "0.0001");
        assert_eq!(format_estimate(0.00004, 4), "4.000e-05");
        assert_eq!(format_estimate(-1.1556e-07, 4), "-1.156e-07");
        assert_eq!(format_estimate(-42.71436, 4), "-42.7144");
        assert_eq!(format_estimate(1.2e-120, 4), "1.200e-120");
        assert_eq!(format_estimate(3e-3, 2), "3.0e-03");
        assert_eq!(format_estimate(0.3, 0), "3e-01");
        assert_eq!(format_estimate(f64::NAN, 4), "NA");
        assert_eq!(format_estimate(f64::INFINITY, 4), "Inf");
        assert_eq!(format_estimate(f64::NEG_INFINITY, 4), "-Inf");
    }

    /// (symbol, threshold) pairs of a rendered legend.
    fn legend_entries(legend: &str, style: LegendStyle) -> Vec<(String, f64)> {
        let (separator, marker) = match style {
            LegendStyle::Latex => ("; ", "$p<"),
            LegendStyle::Html => (", ", "p &lt; "),
            LegendStyle::Markdown => (", ", "p < "),
        };
        legend
            .split(separator)
            .map(|item| {
                let (symbol, rest) = item.split_once(marker).expect(item);
                let threshold = rest.trim_end_matches('$').parse().expect(item);
                (symbol.trim().to_string(), threshold)
            })
            .collect()
    }

    #[test]
    fn test_legend_follows_the_star_mapping() {
        for style in [LegendStyle::Latex, LegendStyle::Html, LegendStyle::Markdown] {
            let legend = significance_legend(style);
            let entries = legend_entries(&legend, style);
            assert_eq!(entries.len(), LEGEND_LEVELS.len(), "{legend}");
            for (level, (symbol, t)) in LEGEND_LEVELS.iter().zip(&entries) {
                assert_eq!(
                    SignificanceLevel::from_p_value(t * (1.0 - 1e-12)),
                    *level,
                    "{legend}"
                );
                assert_ne!(SignificanceLevel::from_p_value(*t), *level, "{legend}");
                let want = match style {
                    LegendStyle::Latex => level.stars().replace('†', "\\textdagger{}"),
                    LegendStyle::Html => level.stars().replace('†', "&dagger;"),
                    LegendStyle::Markdown => level.stars().replace('*', "\\*"),
                };
                assert_eq!(symbol, &want, "{legend}");
            }
        }
        assert_eq!(
            significance_legend(LegendStyle::Latex),
            "\\textdagger{} $p<0.10$; * $p<0.05$; ** $p<0.01$; *** $p<0.001$"
        );
    }
}
