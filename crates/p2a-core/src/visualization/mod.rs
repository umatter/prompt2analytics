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
    histogram, scatter_plot, box_plot, line_chart,
    event_study_plot, coefficient_plot, irf_plot, residual_diagnostics, dendrogram,
    ChartConfig, HistogramResult, ScatterResult, BoxPlotResult, LineChartResult,
    EventStudyResult, CoefficientPlotResult, IrfPlotResult, ResidualDiagnosticsResult,
    DendrogramResult,
};
pub use heatmap::{correlation_heatmap, HeatmapResult};
pub use interactive::{
    scatter_interactive, histogram_interactive, line_interactive,
    InteractiveConfig, InteractivePlotResult,
};
pub use colors::{
    BRAND_ORANGE, BRAND_CYAN, BRAND_TEAL, BRAND_ORANGE_LIGHT, BRAND_TEAL_DARK, BRAND_ORANGE_DARK,
    BRAND_SLATE, CHART_PALETTE, PLOTLY_PALETTE, DEFAULT_SERIES_COLOR, SECONDARY_COLOR,
    OUTLIER_COLOR, TREND_LINE_COLOR,
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
