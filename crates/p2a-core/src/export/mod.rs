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
//! let csv = ols_result.to_csv();  // Returns CSV string
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
