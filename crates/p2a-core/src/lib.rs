//! # p2a-core
//!
//! Core analytics engine for prompt2analytics.
//!
//! This crate provides the data loading, statistical analysis, and machine learning
//! functionality that powers the MCP server.

// Foundation modules (pure Rust implementations)
pub mod linalg;
pub mod traits;
pub mod errors;

// Feature modules
pub mod data;
pub mod stats;
pub mod regression;
pub mod econometrics;
pub mod forecasting;
pub mod ml;
pub mod visualization;
pub mod reports;

// Re-export foundational types
pub use errors::{EconError, EconResult, EstimationWarning};
pub use traits::{LinearEstimator, SignificanceLevel};
pub use linalg::{DesignMatrix, DesignError};

pub use data::{Dataset, DataLoader, DatasetInfo};
pub use stats::{DescriptiveStats, CorrelationMatrix, correlation_matrix};
pub use regression::{OlsResult, run_ols, run_ols_clustered, DiagnosticsResult, run_diagnostics};
pub use econometrics::{
    PanelResult, HausmanResult, run_fixed_effects, run_random_effects, run_hausman_test,
    IVResult, run_iv2sls, FirstStageDiagnostics, run_first_stage_diagnostics,
    DiDResult, run_did,
    DiscreteResult, run_logit, run_probit,
    VarResult, VarmaResult, VecmResult, VarIrfResult, run_var, run_varma, run_vecm, run_var_irf,
};
pub use forecasting::{
    ArimaResult, ArimaForecastResult, run_arima, forecast_arima,
    MstlResult, run_mstl,
    ChangepointResult, SegmentStats, CostFunction, detect_changepoints, binary_segmentation,
    run_changepoint, run_binary_segmentation,
};
pub use ml::{
    KMeansResult, DBSCANResult, HierarchicalResult, Linkage, PCAResult, TsneResult,
    RandomForestResult, SvmResult,
    kmeans, dbscan, hierarchical, pca, pca_transform, pca_inverse_transform, tsne,
    random_forest, linear_svm, svm_predict,
};
pub use visualization::{
    ChartConfig, HistogramResult, ScatterResult, BoxPlotResult, LineChartResult, HeatmapResult,
    EventStudyResult, CoefficientPlotResult, IrfPlotResult, ResidualDiagnosticsResult, DendrogramResult,
    histogram, scatter_plot, box_plot, line_chart, correlation_heatmap,
    event_study_plot, coefficient_plot, irf_plot, residual_diagnostics, dendrogram,
    VisualizationError,
};
pub use reports::{
    HtmlReport, ReportSection, ReportTable, ReportContent, generate_html_report,
};

/// Re-export polars for downstream use
pub use polars;
