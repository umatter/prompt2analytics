//! Visualization module for creating charts and plots.
//!
//! This module provides two types of visualizations:
//! - **Static charts** (via `plotters`): Generate PNG images encoded as base64 strings
//! - **Interactive charts** (via `plotlars`): Generate HTML with embedded Plotly.js
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

mod charts;
pub mod colors;
mod heatmap;
pub mod interactive;

pub use charts::{
    BoxPlotResult, ChartConfig, CoefficientPlotResult, DendrogramResult, EventStudyResult,
    HistogramResult, IrfPlotResult, LineChartResult, ResidualDiagnosticsResult, ScatterResult,
    box_plot, coefficient_plot, dendrogram, event_study_plot, histogram, irf_plot, line_chart,
    residual_diagnostics, scatter_plot,
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
