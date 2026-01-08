//! Visualization module for creating charts and plots.
//!
//! Generates PNG images encoded as base64 strings for MCP output.

mod charts;
mod heatmap;

pub use charts::{
    histogram, scatter_plot, box_plot, line_chart,
    event_study_plot, coefficient_plot, irf_plot, residual_diagnostics,
    ChartConfig, HistogramResult, ScatterResult, BoxPlotResult, LineChartResult,
    EventStudyResult, CoefficientPlotResult, IrfPlotResult, ResidualDiagnosticsResult,
};
pub use heatmap::{correlation_heatmap, HeatmapResult};

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
