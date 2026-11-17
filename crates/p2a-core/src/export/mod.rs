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
//! - Significance stars (*, **, ***)
//! - Multi-model comparison tables
//! - Custom notes and captions
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
//! The [`CsvExport`] trait is implemented for all result types:
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
//! All export builders support these result types:
//! - [`OlsResult`](crate::regression::OlsResult) - OLS regression
//! - [`PanelResult`](crate::econometrics::PanelResult) - Fixed/random effects
//! - [`IVResult`](crate::econometrics::IVResult) - Instrumental variables
//! - [`DiscreteResult`](crate::econometrics::DiscreteResult) - Logit/probit
//! - And many more via the [`CsvExport`] trait

mod csv;
mod html;
mod latex;
mod markdown;

pub use csv::CsvExport;
pub use html::{HtmlStyle, HtmlTableBuilder};
pub use latex::{LatexStyle, LatexTableBuilder};
pub use markdown::{MarkdownStyle, MarkdownTableBuilder};

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
}
