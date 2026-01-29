//! Report generation module.
//!
//! Provides functionality for generating HTML and PDF reports from analysis results.

mod html;

pub use html::{HtmlReport, ReportContent, ReportSection, ReportTable, generate_html_report};
