//! Export functionality for regression results.
//!
//! Provides multiple export formats for publication-ready output:
//! - LaTeX tables for academic papers
//! - Markdown tables for documentation and GitHub
//! - CSV export for data interchange
//! - HTML export for web display

mod csv;
mod html;
mod latex;
mod markdown;

pub use csv::CsvExport;
pub use html::{HtmlTableBuilder, HtmlStyle};
pub use latex::{LatexTableBuilder, LatexStyle};
pub use markdown::{MarkdownTableBuilder, MarkdownStyle};
