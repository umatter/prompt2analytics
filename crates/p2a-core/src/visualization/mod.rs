//! Visualization module for creating charts and plots.
//!
//! This module provides two types of visualizations:
//! - **Static charts** (via `plotters`): Generate PNG images encoded as base64 strings, each
//!   with a vector version (SVG) up to [`MAX_SVG_BYTES`]
//! - **Interactive charts**: a Plotly figure written as JSON into an HTML page that loads
//!   plotly.js 3.0.1 and MathJax from cdn.plot.ly and cdn.jsdelivr.net when opened; viewing needs
//!   network access (docs/guides/DATA_SECURITY.md). Values are kept in double precision.
//!
//! Use static charts when you need:
//! - Lightweight output for embedding
//! - No JavaScript dependencies
//! - Consistent rendering across all environments
//!
//! Use interactive charts when you need:
//! - Zoom, pan, and hover capabilities
//! - Interactive exploration of data
//! - HTML output for web display

mod axis;
pub mod charts;
pub mod colors;
pub mod heatmap;
pub mod interactive;
mod vector;

pub use charts::{
    Band, BoxPlotResult, ChartConfig, CoefficientPlotResult, DendrogramResult, EventStudyResult,
    HistogramResult, IrfPlotResult, LineChartResult, ResidualDiagnosticsResult, ScatterResult,
    box_plot, coefficient_plot, dendrogram, event_study_plot, histogram, irf_plot, line_chart,
    line_chart_with_band, residual_diagnostics, scatter_plot,
};
pub use colors::{
    BRAND_CYAN, BRAND_ORANGE, BRAND_ORANGE_DARK, BRAND_ORANGE_LIGHT, BRAND_SLATE, BRAND_TEAL,
    BRAND_TEAL_DARK, CHART_PALETTE, DEFAULT_SERIES_COLOR, OUTLIER_COLOR, PLOTLY_PALETTE,
    SECONDARY_COLOR, TREND_LINE_COLOR,
};
pub use heatmap::{HeatmapResult, correlation_heatmap};
pub use interactive::{
    InteractiveConfig, InteractivePlotResult, histogram_interactive, line_interactive,
    scatter_interactive,
};

pub use vector::MAX_SVG_BYTES;

use thiserror::Error;

/// Visualization-related errors.
#[derive(Debug, Error)]
pub enum VisualizationError {
    #[error("Plotting error: {0}")]
    PlottingError(String),

    #[error("Invalid data: {0}")]
    InvalidData(String),

    #[error("Encoding error: {0}")]
    EncodingError(String),
}

/// Default chart dimensions
pub const DEFAULT_WIDTH: u32 = 800;
pub const DEFAULT_HEIGHT: u32 = 600;
