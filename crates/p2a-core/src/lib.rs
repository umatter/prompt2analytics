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
pub mod simulation;

// Re-export foundational types
pub use errors::{EconError, EconResult, EstimationWarning};
pub use traits::{LinearEstimator, SignificanceLevel};
pub use linalg::{DesignMatrix, DesignError};

pub use data::{Dataset, DataLoader, DatasetInfo};
pub use stats::{
    DescriptiveStats, CorrelationMatrix, correlation_matrix,
    AnovaResult, TwoWayAnovaResult, RegressionAnovaResult, GroupStats,
    run_one_way_anova, run_two_way_anova, anova_from_ols,
    TTestResult, Alternative, one_sample_t_test, two_sample_t_test, paired_t_test, t_test,
    // ACF/PACF/CCF
    AcfType, CcfType, AcfResult, PacfResult, CcfResult,
    acf, pacf, ccf, run_acf, run_pacf, run_ccf,
    // Chi-squared tests
    ChiSquaredResult, chisq_test_gof, chisq_test_independence,
    run_chisq_gof, run_chisq_independence,
    // Fisher's exact test
    FisherExactResult, FisherAlternative, fisher_exact_test, fisher_exact_test_int, run_fisher_test,
    // Wilcoxon tests
    WilcoxonResult, WilcoxonConfig, wilcoxon_rank_sum, wilcoxon_signed_rank, wilcoxon_test,
    // Shapiro-Wilk normality test
    ShapiroWilkResult, shapiro_wilk_test, run_shapiro_wilk,
    // Kolmogorov-Smirnov test
    KsTestResult, TheoreticalDistribution, ks_test_one_sample, ks_test_two_sample, ks_test, run_ks_test,
    // MANOVA
    ManovaResult, ManovaTestResult, ManovaTestStatistic, manova_one_way, run_manova,
    // Tukey HSD
    TukeyHsdResult, PairwiseComparison, tukey_hsd, run_tukey_hsd,
    // Bartlett's test
    BartlettResult, BartlettGroupStats, bartlett_test, run_bartlett_test,
    // Spectral density estimation
    SpectrumConfig, SpectrumResult, spectrum, spectrum_ar, run_spectrum, run_spectrum_ar,
    // Box-Pierce and Ljung-Box tests
    BoxTestResult, BoxTestType, box_test, run_box_test,
    // Phillips-Perron unit root test
    PPTestResult, pp_test, run_pp_test,
    // F test for comparing two variances
    VarTestResult, var_test, run_var_test,
    // Proportion tests
    PropTestResult, prop_test_one, prop_test_two, prop_test_k,
    // Exact binomial test
    BinomTestResult, binom_test,
    // Fligner-Killeen test for homogeneity of variances
    FlignerResult, fligner_test, run_fligner_test,
    // Ansari-Bradley test for scale parameters
    AnsariBradleyResult, ansari_test,
    // Mood test for scale parameters
    MoodTestResult, mood_test, run_mood_test,
    // Kruskal-Wallis rank sum test
    KruskalWallisResult, kruskal_test, run_kruskal_test,
    // Friedman rank sum test
    FriedmanResult, friedman_test, run_friedman_test,
    // Welch's one-way ANOVA
    OnewayTestResult, oneway_test, run_oneway_test,
    // McNemar's chi-squared test
    McnemarResult, mcnemar_test, mcnemar_test_matrix,
    // Pairwise t-tests with p-value adjustment
    PairwiseTTestResult, PValueAdjustMethod, pairwise_t_test, run_pairwise_t_test, p_adjust,
    // Pairwise Wilcoxon tests with p-value adjustment
    PairwiseWilcoxResult, pairwise_wilcox_test, run_pairwise_wilcox_test,
    // Quade test for unreplicated blocked data
    QuadeResult, quade_test, run_quade_test,
    // Cochran-Mantel-Haenszel test for stratified 2x2 tables
    MantelHaenszelResult, StratumStats, CmhAlternative, Table2x2,
    mantelhaen_test, run_mantelhaen_test,
    // Exact Poisson test
    PoissonTestResult, PoissonAlternative, poisson_test,
    // Mahalanobis distance
    MahalanobisResult, mahalanobis, mahalanobis_single, run_mahalanobis,
    // Correlation test
    CorTestResult, CorrelationMethod, cor_test, run_cor_test,
    // Power analysis
    PowerTTestResult, PowerPropTestResult, PowerAnovaTestResult,
    TTestType, PowerAlternative,
    power_t_test, power_prop_test, power_anova_test,
    run_power_t_test, run_power_prop_test, run_power_anova_test,
    // Prop trend test
    PropTrendTestResult, prop_trend_test, run_prop_trend_test,
    // Robust/descriptive stats
    FivenumResult, fivenum, run_fivenum,
    iqr, run_iqr, quantile,
    mad, run_mad,
    EcdfResult, ecdf, run_ecdf,
    DensityResult, DensityKernel, density, run_density,
    // Spline interpolation
    spline, splinefun, approx, approxfun,
    SplineResult, SplineMethod, ApproxResult, ApproxMethod, ApproxRule,
};
pub use regression::{
    OlsResult, run_ols, run_ols_raw, run_ols_clustered, DiagnosticsResult, run_diagnostics,
    // Nonlinear least squares
    NlsResult, NlsConfig, NlsAlgorithm, nls, nls_multi, run_nls, run_nls_with_config,
    model_exponential_decay, model_exponential_growth, model_michaelis_menten,
    model_logistic_growth, model_power, model_asymptotic, ModelFn,
    // LOESS (local polynomial regression)
    LoessResult, LoessConfig, LoessModel, loess, loess_predict, run_loess,
    // GLS (generalized least squares)
    GlsResult, CorrelationStructure, gls, gls_ar1_auto, run_gls,
    // Smoothing spline
    SmoothSplineResult, SmoothSplineConfig, smooth_spline, smooth_spline_predict, run_smooth_spline,
};
pub use econometrics::{
    PanelResult, HausmanResult, run_fixed_effects, run_random_effects, run_hausman_test,
    IVResult, run_iv2sls, FirstStageDiagnostics, run_first_stage_diagnostics,
    DiDResult, run_did,
    DiscreteResult, run_logit, run_probit,
    VarResult, VarmaResult, VecmResult, VarIrfResult, run_var, run_varma, run_vecm, run_var_irf,
    HdfeResult, HdfeConfig, FactorInfo, run_hdfe,
    // GLM with HDFE
    GlmFamily, FeglmConfig, FeglmResult, run_feglm,
    // Treatment effects
    Estimand, DRMethod, IpwConfig, DoublyRobustConfig,
    PropensityScoreSummary, IpwResult, DoublyRobustResult,
    run_ipw_treatment, run_doubly_robust,
    // Mediation analysis
    MediationConfig, MediationResult, run_mediation_analysis,
    // Synthetic control
    SynthConfig, SynthResult, PredictorSpec, PredictorBalance, TimeEffect,
    PlaceboResults, VOptimization, TimeAggregation, run_synthetic_control,
    // Regression discontinuity
    KernelType, BandwidthMethod, VceType, RdConfig, RdBandwidth,
    RdResult, FuzzyRdResult, run_rd, rd_bandwidth, run_fuzzy_rd,
    // Survival analysis
    TiesMethod, AftDistribution,
    KaplanMeierResult, LogRankResult, CoxConfig, CoxResult, AftConfig, AftResult,
    CumulativeIncidence, CompetingRisksResult,
    run_kaplan_meier, log_rank_test, run_cox_ph, run_aft, run_competing_risks,
};
pub use forecasting::{
    ArimaResult, ArimaForecastResult, run_arima, forecast_arima,
    MstlResult, run_mstl,
    ChangepointResult, SegmentStats, CostFunction, detect_changepoints, binary_segmentation,
    run_changepoint, run_binary_segmentation,
    // Holt-Winters exponential smoothing
    HoltWintersResult, HoltWintersConfig, HoltWintersCoefficients, SeasonalType,
    holt_winters, holt_winters_forecast, run_holt_winters,
    // Autoregressive model fitting
    ArResult, ArConfig, ArMethod, ar, run_ar, run_ar_with_order,
    // Classical decomposition
    DecomposeResult, DecomposeConfig, DecomposeType,
    decompose, run_decompose, run_decompose_with_filter,
    // Kalman filter
    StateSpaceModel, KalmanFilterResult, KalmanSmootherResult, KalmanForecastResult,
    kalman_filter, kalman_smoother, kalman_forecast, kalman_loglik,
    // Structural time series
    StructTsType, StructTsConfig, StructTsResult, StructTsCoefficients,
    struct_ts, run_struct_ts,
    // STL decomposition
    StlResult, StlConfig, stl, run_stl, run_stl_with_config,
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
pub use simulation::{
    generate_random_data, ColumnSpec, Distribution, GenerationError,
};

/// Re-export polars for downstream use
pub use polars;
