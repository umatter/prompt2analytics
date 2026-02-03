//! Analytics MCP Server implementation.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

use rmcp::{
    ErrorData as McpError, ServerHandler, handler::server::router::tool::ToolRouter,
    handler::server::wrapper::Parameters, model::*, tool, tool_handler, tool_router,
};
use schemars::JsonSchema;
use serde::Deserialize;

use p2a_core::{
    AftConfig,
    AftDistribution,
    ArConfig,
    ArMethod,
    AttEstimationMethod,
    BPBoundsConfig,
    BandwidthMethod,
    CTmleConfig,
    CTmleQModel,
    CausalImpactConfig,
    CbpsConfig,
    CbpsMethod,
    ChartConfig,
    ComparisonGroup,
    ControlGroup as EtwfeControlGroup,
    CostFunction,
    CoxConfig,
    DRMethod,
    DecomposeConfig,
    DecomposeType,
    DistanceMethod,
    DoublyRobustConfig,
    EffectEstimationMethod,
    EffectScale,
    EndRule,
    Estimand,
    EtwfeConfig,
    FeglmConfig,
    FilterMethod,
    FilterSides,
    GFormulaConfig,
    GFormulaData,
    GFormulaIntervention,
    GFormulaOutcomeType,
    GModel,
    GarchConfig,
    GeneralGmmConfig,
    GeneralGmmResult,
    GlmFamily,
    GmmConfig,
    GmmMethod,
    GmmResult,
    GmmStep,
    GmmTransform,
    GmmVcov,
    GsynthConfig,
    GsynthEstimator,
    GsynthForce,
    HdfeConfig,
    HetTestStat,
    HetTxConfig,
    // Reports
    HtmlReport,
    HurdleType,
    IVMTEConfig,
    IpwConfig,
    KernelType,
    // Traits
    LinearEstimator,
    Linkage,
    MatchMethod,
    MatchResult,
    MedflexConfig,
    MedflexResult,
    MediationConfig,
    MixedLogitConfig,
    MoranAlternative,
    // Spatial econometrics
    Neighbors,
    PanelGlsModel,
    // Panel GLS (FGLS)
    PanelGlsResult,
    PanelModel,
    PanelUnitRootConfig,
    PanelUnitRootTest,
    PoolingWeights,
    PprConfig,
    PredictorSpec,
    PropensityModel,
    PvcmType,
    QModel,
    RandomDistribution,
    RdConfig,
    RdMultiBandwidth,
    RdMultiConfig,
    RdMultiResult,
    ReportSection,
    ReportTable,
    SBWConfig,
    SBWEstimand,
    SCPIConfig,
    SCPIConstraint,
    SEMethod,
    SarConfig,
    SeasonalType,
    SelectionOrder,
    SemConfig,
    SmoothingMethod,
    SpatialErrorType,
    SpatialPanelEffect,
    SpatialPanelModel,
    SpatialProbitConfig,
    SpatialWeights,
    SpgmConfig,
    SpgmMethod,
    SphetConfig,
    SphetModel,
    SphetSE,
    SpmlConfig,
    StaggeredDidConfig,
    StdRegConfig,
    StdRegEstimand,
    StdRegModel,
    StopMethod,
    StoppingRule,
    StructTsConfig,
    StructTsType,
    SynthConfig,
    TiesMethod,
    TimeAggregation,
    TmleConfig,
    TwangConfig,
    TwangEstimand,
    VOptimization,
    VarianceMethod,
    VceType,
    WeightEstimand,
    WeightItConfig,
    WeightMethod,
    WeightStyle,
    acf_to_ar,
    ar,
    arima_sim,
    arma_acf,
    arma_to_ma,
    // Goodman-Bacon decomposition
    bacon_decomp,
    box_plot,
    // CausalImpact (Bayesian Structural Time Series)
    causal_impact,
    cmdscale,
    cmdscale_from_data,
    coefficient_plot,
    correlation_heatmap,
    // Cumulative periodogram
    cpgram,
    // C-TMLE (Collaborative TMLE with data-adaptive covariate selection)
    ctmle,
    cutree,
    data::{
        CleaningOperation,
        // Session management
        CleaningSession,
        DataLoader,
        Dataset,
        DatasetInfo,
        SuggestionPriority,
        duckdb_table_schema,
        // Data quality profiling
        generate_quality_profile,
        // Smart suggestions
        generate_suggestions,
        list_duckdb_tables,
        list_sqlite_tables,
        // Data munging
        munging::{
            AggFn,
            AggSpec,
            ArithOp,
            BinStrategy,
            FillStrategy,
            MutateExpr,
            bin,
            concat,
            deduplicate,
            diff,
            drop_columns,
            // Clean operations
            drop_na,
            fill_na,
            // Transform operations
            filter,
            full_join,
            // Aggregate operations
            group_by,
            inner_join,
            // Feature engineering
            lag,
            lead,
            // Join operations
            left_join,
            melt,
            mutate,
            normalize,
            one_hot_encode,
            pct_change,
            // Reshape operations
            pivot,
            regex_count,
            regex_extract,
            // Regex and string operations
            regex_replace,
            rename,
            replace,
            right_join,
            sample,
            select,
            sort,
            standardize,
            str_concat,
            str_length,
            str_split,
            str_substring,
            to_lowercase,
            to_uppercase,
            trim,
            value_counts,
        },
        // Verification and preview
        preview_cleaning,
        query_duckdb,
        query_file_with_duckdb,
        // Database connectivity
        query_sqlite,
        sqlite_table_schema,
        verify_cleaning,
    },
    dbscan,
    decompose,
    // Diagnostics
    diagnostics::{
        IdentificationReport, WarningSeverity, did_diagnostics, ipw_diagnostics, iv_diagnostics,
        matching_diagnostics, rd_diagnostics, staggered_did_diagnostics,
    },
    diffinv,
    embed,
    entropy_balance,
    event_study_plot,
    filter as ts_filter,
    forecast_arima,
    // GARCH (volatility modeling)
    garch,
    // Granger causality test
    granger_test,
    // Harvey-Collier test for linearity
    harvey_collier_test,
    hierarchical,
    // Visualization
    histogram,
    holt_winters_forecast,
    irf_plot,
    // Machine Learning
    kmeans,
    // Time series utilities (aliased to avoid collision with munging)
    lag as ts_lag,
    line_chart,
    linear_svm,
    log_rank_test,
    // MatchIt (Propensity Score Matching)
    match_it,
    moran_test,
    pca,
    // Projection Pursuit Regression
    ppr,
    random_forest,
    rd_bandwidth,
    regression::{
        BgTestType,
        CorrelationStructure,
        CovarianceType,
        ResetType,
        SmoothSplineConfig,
        // Breusch-Godfrey test for serial correlation
        bg_test,
        // GLS regression
        gls,
        quantreg_multi,
        // Ramsey's RESET test
        reset_test,
        run_diagnostics,
        // Tukey's resistant line
        run_line,
        run_ols,
        run_ols_clustered,
        // Quantile regression
        run_quantreg,
        // Bootstrap covariance estimation
        run_vcov_bootstrap,
        // Driscoll-Kraay panel-robust standard errors
        run_vcov_driscoll_kraay,
        // HAC (Newey-West) standard errors
        run_vcov_hac,
        // Smooth splines
        smooth_spline,
        smooth_spline_predict,
        // SuperSmoother
        supsmu,
        // Wald test for nested model comparison
        wald_test,
    },
    residual_diagnostics,
    run_aft,
    // Forecasting
    run_arima,
    // BART-based Causal Inference (bartCause style)
    run_bart_causal,
    run_binary_segmentation,
    // Balke-Pearl bounds for nonparametric IV
    run_bp_bounds,
    // Causal Forests (Wager & Athey 2018)
    run_causal_forest,
    // CBPS (Covariate Balancing Propensity Score)
    run_cbps,
    run_changepoint,
    run_competing_risks,
    run_cox_ph,
    run_did,
    run_doubly_robust,
    // Extended TWFE (Wooldridge)
    run_etwfe,
    // GLM with HDFE
    run_feglm,
    run_first_stage_diagnostics,
    // Econometrics
    run_fixed_effects,
    run_fuzzy_rd,
    // Parametric G-Formula for time-varying treatments (gfoRmula)
    run_gformula,
    // Arellano-Bond / System GMM
    run_gmm,
    // General GMM (Hansen 1982)
    run_gmm_iv,
    run_gmnl,
    run_granger_test,
    // Generalized synthetic control
    run_gsynth,
    run_hausman_test,
    run_hdfe,
    // Treatment Effect Heterogeneity Testing (hettx)
    run_hettx_dataset,
    run_holt_winters,
    // Hurdle models
    run_hurdle,
    // Treatment effects
    run_ipw_treatment,
    run_iv2sls,
    // Marginal Treatment Effects (MTE) for IV analysis
    run_ivmte,
    // Survival Analysis
    run_kaplan_meier,
    run_logit,
    // Natural Effect Models (medflex)
    run_medflex_dataset,
    // Mediation analysis
    run_mediation_analysis,
    // McFadden conditional logit (mlogit)
    run_mlogit,
    run_mstl,
    // Multinomial and ordered logit/probit
    run_multinom,
    // Negative binomial and zero-inflated models
    run_negbin,
    run_ordered_logit,
    run_ordered_probit,
    run_panel_gls,
    // Panel unit root tests
    run_panel_unit_root,
    run_pmg,
    run_probit,
    // Variable Coefficients Model (pvcm) and Mean Group (pmg)
    run_pvcm,
    run_random_effects,
    // Regression Discontinuity
    run_rd,
    // Multi-cutoff RD (rdmulti)
    run_rd_multi_dataset,
    run_sar,
    // Spatial probit models
    run_sar_probit,
    // Synthetic Control with Prediction Intervals (SCPI)
    run_scpi,
    run_sem,
    run_sem_probit,
    run_spgm,
    // Spatial GMM with heteroscedasticity robustness (sphet)
    run_sphet,
    // Spatial panel data models (splm)
    run_spml,
    run_staggered_did,
    // Regression Standardization / G-computation (stdReg)
    run_stdreg,
    run_step,
    // Synthetic control
    run_synthetic_control,
    // twang (GBM propensity score estimation)
    run_twang,
    // Time series
    run_var,
    run_var_irf,
    run_varma,
    run_vecm,
    run_zinb,
    run_zip,
    runmed,
    sargan_test,
    // SBW (Stable Balancing Weights)
    sbw,
    scatter_plot,
    // Simulation
    simulation::{ColumnSpec, Distribution, generate_random_data},
    spatial_lm_tests,
    stats::{
        AcfType,
        Alternative,
        ApproxMethod,
        ApproxRule,
        BoxTestType,
        CcfType,
        CmhAlternative,
        ContrastType,
        DescriptiveStats,
        PValueAdjustMethod,
        PoissonAlternative,
        SpectrumConfig,
        SplineMethod,
        TableType,
        approx,
        correlation_matrix,
        // Robust statistics
        fivenum,
        generate_contrasts,
        iqr,
        // Isotonic regression
        isoreg,
        // Log-linear models
        loglin,
        // McNemar's test
        mcnemar_test,
        model_tables,
        // Mood test
        mood_test,
        one_sample_t_test,
        paired_t_test,
        // Exact Poisson test
        poisson_test,
        // ACF/PACF/CCF
        run_acf,
        run_bartlett_test,
        // Box-Pierce and Ljung-Box tests
        run_box_test,
        run_ccf,
        // Chi-squared tests
        run_chisq_gof,
        run_chisq_independence,
        run_cov_wt,
        run_density,
        run_ecdf,
        // Friedman test
        run_friedman_test,
        // Kruskal-Wallis test
        run_kruskal_test,
        run_mad,
        run_mahalanobis,
        // MANOVA, Tukey HSD, and Bartlett test
        run_manova,
        // Cochran-Mantel-Haenszel test
        run_mantelhaen_test,
        // Mauchly's sphericity test
        run_mauchly_test,
        // Median polish
        run_medpolish,
        run_one_way_anova,
        // Welch's one-way ANOVA
        run_oneway_test,
        run_pacf,
        // Pairwise t-tests and Wilcoxon tests
        run_pairwise_t_test,
        run_pairwise_wilcox_test,
        // Phillips-Perron unit root test
        run_pp_test,
        // Quade test
        run_quade_test,
        // Spectral density estimation
        run_spectrum,
        run_spectrum_ar,
        run_tukey_hsd,
        run_two_way_anova,
        // Standard errors for contrasts
        se_contrast,
        // Spline interpolation
        spline,
        two_sample_t_test,
        // Weighted statistics
        weighted_mean,
    },
    // Kalman filter and StructTS
    struct_ts,
    tmle,
    // Toeplitz matrix construction
    toeplitz,
    toeplitz_asymmetric,
    toeplitz_to_vec,
    tsne,
    // WeightIt (Flexible inverse probability weighting)
    weightit,
    window as ts_window,
};

/// The main analytics server that handles MCP requests.
#[derive(Clone)]
pub struct AnalyticsServer {
    /// Currently loaded datasets, keyed by a unique ID
    pub(crate) datasets: Arc<RwLock<HashMap<String, Dataset>>>,
    /// Active cleaning sessions, keyed by session ID
    pub(crate) cleaning_sessions: Arc<RwLock<HashMap<String, CleaningSession>>>,
    /// Spatial weights matrices, keyed by a unique ID
    pub(crate) spatial_weights: Arc<RwLock<HashMap<String, SpatialWeights>>>,
    /// Global random seed for ML reproducibility
    pub(crate) global_seed: Arc<RwLock<Option<u64>>>,
    /// Memory profiler for tracking dataset memory usage
    pub(crate) memory_profiler: Arc<RwLock<p2a_core::MemoryProfiler>>,
    /// Tool router for handling tool calls
    tool_router: ToolRouter<Self>,
}

// ============================================================================
// Tool Input/Output Types
// ============================================================================

/// Request to load a dataset from a file.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct LoadDatasetRequest {
    /// Path to the data file
    #[schemars(
        description = "Absolute or relative path to the data file. Supports CSV, Parquet, Excel (xlsx, xls, xlsb, ods), Stata (dta), and SAS (sas7bdat) formats."
    )]
    pub path: String,

    /// Optional name/identifier for the dataset
    #[schemars(
        description = "Optional name to identify this dataset. If not provided, the filename will be used."
    )]
    pub name: Option<String>,
}

/// Request to upload and load a dataset from base64-encoded content.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct UploadDatasetRequest {
    /// Base64-encoded file content
    #[schemars(description = "The file content encoded as base64")]
    pub content: String,

    /// Original filename (used to determine format and default name)
    #[schemars(description = "Original filename including extension (e.g., 'data.csv')")]
    pub filename: String,

    /// Optional name/identifier for the dataset
    #[schemars(
        description = "Optional name to identify this dataset. If not provided, the filename will be used."
    )]
    pub name: Option<String>,
}

/// Request to create a dataset from inline CSV content.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CreateDatasetRequest {
    /// Name/identifier for the dataset
    #[schemars(description = "Name to identify this dataset (e.g., 'my_data')")]
    pub name: String,

    /// CSV content as plain text
    #[schemars(description = "CSV content with headers in first row (e.g., 'x,y\\n1,2\\n3,4')")]
    pub csv_content: String,
}

/// Request to export a dataset to a file.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExportDatasetRequest {
    /// Name/ID of the dataset to export
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Output file path
    #[schemars(
        description = "Path where the file will be saved. The format is determined by extension: .csv, .parquet, .json"
    )]
    pub path: String,

    /// Output format (optional, inferred from extension if not specified)
    #[schemars(
        description = "Output format: 'csv', 'parquet', or 'json'. If not specified, inferred from file extension."
    )]
    pub format: Option<String>,
}

/// Request to describe a loaded dataset.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DescribeDatasetRequest {
    /// Name/ID of the dataset to describe
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,
}

/// Request to preview rows from a dataset.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct HeadDatasetRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Number of rows to return (default: 5)
    #[schemars(description = "Number of rows to return. Default is 5.")]
    pub n: Option<usize>,
}

/// Request for autocorrelation function (ACF).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct AcfRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Time series column name
    #[schemars(description = "Name of the numeric time series column.")]
    pub column: String,

    /// Maximum lag
    #[schemars(description = "Maximum lag to compute. Default: min(10*log10(n), n-1).")]
    pub lag_max: Option<usize>,

    /// Type of ACF
    #[schemars(
        description = "Type of autocorrelation: 'correlation' (default), 'covariance', or 'partial' (PACF)."
    )]
    pub acf_type: Option<String>,
}

/// Request for cross-correlation function (CCF).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CcfRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// First time series column name
    #[schemars(description = "Name of the first (X) numeric time series column.")]
    pub x: String,

    /// Second time series column name
    #[schemars(description = "Name of the second (Y) numeric time series column.")]
    pub y: String,

    /// Maximum lag
    #[schemars(
        description = "Maximum lag to compute (in both directions). Default: min(10*log10(n), n-1)."
    )]
    pub lag_max: Option<usize>,
}

/// Request for spectral density estimation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SpectrumRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Time series column name
    #[schemars(description = "Name of the numeric time series column.")]
    pub column: String,

    /// Smoothing spans (Daniell kernel widths)
    #[schemars(
        description = "Vector of odd integers for modified Daniell smoothers, e.g., [3,3] or [5,5,5]. If omitted, returns raw periodogram. Multiple values are convolved for more smoothing."
    )]
    pub spans: Option<Vec<usize>>,

    /// Taper proportion
    #[schemars(
        description = "Proportion of data to taper at ends using split cosine bell (0 to 0.5). Default: 0.1. Reduces spectral leakage."
    )]
    pub taper: Option<f64>,

    /// Whether to detrend
    #[schemars(
        description = "Whether to remove linear trend before computing spectrum. Default: true."
    )]
    pub detrend: Option<bool>,

    /// Method for spectrum estimation
    #[schemars(
        description = "Estimation method: 'pgram' (periodogram, default) or 'ar' (autoregressive). AR method fits an AR model and computes its theoretical spectrum."
    )]
    pub method: Option<String>,

    /// AR order (for method='ar')
    #[schemars(
        description = "AR model order for method='ar'. If omitted, order is selected by AIC."
    )]
    pub ar_order: Option<usize>,
}

/// Request for Box-Pierce or Ljung-Box test.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct BoxTestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Time series column name
    #[schemars(
        description = "Name of the numeric time series column to test for autocorrelation."
    )]
    pub column: String,

    /// Number of lags
    #[schemars(
        description = "Number of autocorrelation lags to include in the test statistic. Default: 1. Common choices: 10 for short series, 20 for longer series."
    )]
    pub lag: Option<usize>,

    /// Test type
    #[schemars(
        description = "Test type: 'ljung-box' (default, better finite-sample properties) or 'box-pierce' (simpler, classic version)."
    )]
    pub test_type: Option<String>,

    /// Degrees of freedom adjustment
    #[schemars(
        description = "Number of parameters already estimated (subtract from df). When testing ARMA(p,q) residuals, set fitdf = p + q for proper df adjustment. Default: 0."
    )]
    pub fitdf: Option<usize>,
}

/// Request for Phillips-Perron unit root test.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PPTestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Time series column name
    #[schemars(description = "Name of the numeric time series column to test for unit root.")]
    pub column: String,

    /// Use short truncation lag
    #[schemars(
        description = "Whether to use short truncation lag formula: trunc(4*(n/100)^0.25). If false, uses long formula: trunc(12*(n/100)^0.25). Default: true."
    )]
    pub lshort: Option<bool>,
}

/// Request for OLS regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct OlsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,
}

/// Request for regression diagnostics.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DiagnosticsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,
}

/// Request for Breusch-Godfrey test for serial correlation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct BgTestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Order of serial correlation to test (default: 1)
    #[schemars(
        description = "Maximum lag order for serial correlation test. Default is 1 (first-order)."
    )]
    pub order: Option<usize>,

    /// Test statistic type: 'chisq' (default) or 'f'
    #[schemars(
        description = "Type of test statistic: 'chisq' for chi-squared (asymptotic) or 'f' for F-test (finite sample correction). Default: 'chisq'."
    )]
    pub test_type: Option<String>,

    /// Fill value for initial lagged residuals (default: 0.0)
    #[schemars(
        description = "Value to fill for initial lagged residuals before enough observations exist. Default: 0.0."
    )]
    pub fill: Option<f64>,
}

/// Request for Ramsey's RESET test for functional form.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ResetTestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Powers to test (default: [2, 3])
    #[schemars(
        description = "Powers to use for augmentation. Default: [2, 3] tests for quadratic and cubic terms."
    )]
    pub powers: Option<Vec<usize>>,

    /// Type of augmentation: 'fitted' (default), 'regressor', or 'princomp'
    #[schemars(
        description = "Type of augmentation: 'fitted' (powers of fitted values, default), 'regressor' (powers of regressors), or 'princomp' (powers of first principal component)."
    )]
    pub reset_type: Option<String>,
}

/// Request for Wald test comparing nested models.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct WaldTestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables for the unrestricted (full) model
    #[schemars(
        description = "Column names for the unrestricted (full) model. Must be a superset of restricted model variables."
    )]
    pub x_unrestricted: Vec<String>,

    /// Independent variables for the restricted (null) model
    #[schemars(
        description = "Column names for the restricted (null) model. Must be a subset of unrestricted model variables."
    )]
    pub x_restricted: Vec<String>,

    /// Use F-test (default: true) or Chi-squared test
    #[schemars(
        description = "If true (default), use F-test. If false, use Chi-squared test. F-test is more common for finite samples."
    )]
    pub use_f_test: Option<bool>,
}

/// Request for Harvey-Collier test for linearity.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct HarveyCollierRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,
}

/// Request for HAC (Newey-West) standard errors.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct HacRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Bandwidth (number of lags). Default: automatic Newey-West selection
    #[schemars(
        description = "Number of lags for HAC estimation. If not provided, uses automatic Newey-West bandwidth: floor(4 * (n/100)^(2/9))."
    )]
    pub bandwidth: Option<usize>,

    /// Kernel type: 'bartlett' (default), 'parzen', 'qs', 'truncated', 'tukey-hanning'
    #[schemars(
        description = "Kernel function for weighting lags: 'bartlett' (Newey-West, default), 'parzen', 'qs' (quadratic spectral), 'truncated', or 'tukey-hanning'."
    )]
    pub kernel: Option<String>,

    /// Whether to use VAR(1) prewhitening (default: false)
    #[schemars(description = "Apply VAR(1) prewhitening before HAC estimation. Default: false.")]
    pub prewhiten: Option<bool>,
}

/// Request for bootstrap covariance estimation (vcovBS).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct BootstrapCovRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Number of bootstrap replications (default: 999)
    #[schemars(
        description = "Number of bootstrap replications. Default: 999. More replications give more precise SE estimates."
    )]
    pub n_boot: Option<usize>,

    /// Bootstrap type: 'pairs' (default), 'residual', or 'wild'
    #[schemars(
        description = "Bootstrap method: 'pairs' (xy, most robust), 'residual' (assumes homoskedasticity), 'wild' (Rademacher weights, robust to heteroskedasticity)."
    )]
    pub bootstrap_type: Option<String>,

    /// Random seed for reproducibility
    #[schemars(description = "Random seed for reproducibility. If not provided, uses entropy.")]
    pub seed: Option<u64>,
}

/// Request for OLS with clustered standard errors.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct OlsClusteredRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// First cluster dimension column (e.g., "firm_id")
    #[schemars(description = "Column name for first clustering dimension (e.g., 'firm_id').")]
    pub cluster1: String,

    /// Second cluster dimension column (optional, for two-way clustering)
    #[schemars(
        description = "Optional column for second clustering dimension (e.g., 'year'). If provided, two-way clustering is used."
    )]
    pub cluster2: Option<String>,
}

/// Request for Driscoll-Kraay panel-robust standard errors.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DriscollKraayRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded panel dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Time period identifier column
    #[schemars(
        description = "Column containing time period identifiers (e.g., 'year', 'quarter'). Used to aggregate scores within periods."
    )]
    pub time_col: String,

    /// Bandwidth for HAC kernel (optional)
    #[schemars(
        description = "Bandwidth for HAC kernel. Default uses Newey-West rule: floor(T^0.25). Higher values allow for more serial correlation."
    )]
    pub bandwidth: Option<usize>,

    /// Kernel type for HAC
    #[schemars(
        description = "HAC kernel: 'bartlett' (default), 'parzen', 'qs', 'truncated', 'tukey-hanning'."
    )]
    pub kernel: Option<String>,
}

/// Request for quantile regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct QuantRegRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Quantile (tau) to estimate (0-1)
    #[schemars(
        description = "Quantile to estimate. Must be between 0 and 1. Common values: 0.25 (first quartile), 0.5 (median), 0.75 (third quartile). Default: 0.5 (median regression)."
    )]
    pub tau: Option<f64>,

    /// Multiple quantiles to estimate
    #[schemars(
        description = "Optional: estimate multiple quantiles at once (e.g., [0.25, 0.5, 0.75] for quartiles). If provided, 'tau' is ignored."
    )]
    pub taus: Option<Vec<f64>>,
}

/// Request for nonlinear least squares regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct NlsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variable (X) column name
    #[schemars(description = "Name of the independent variable (X) column.")]
    pub x: String,

    /// Model type to fit
    #[schemars(
        description = "Nonlinear model to fit: 'exponential_decay' (y = a*exp(-b*x) + c), 'exponential_growth' (y = a*exp(b*x)), 'michaelis_menten' (y = Vmax*x/(Km+x)), 'logistic' (y = K/(1+exp(-r*(x-x0)))), 'power' (y = a*x^b), 'asymptotic' (y = a - b*exp(-c*x))."
    )]
    pub model: String,

    /// Starting values for parameters
    #[schemars(
        description = "Initial parameter values. For exponential_decay: [a, b, c]. For exponential_growth: [a, b]. For michaelis_menten: [Vmax, Km]. For logistic: [K, r, x0]. For power: [a, b]. For asymptotic: [a, b, c]."
    )]
    pub start: Vec<f64>,

    /// Algorithm to use
    #[schemars(
        description = "Optimization algorithm: 'levenberg_marquardt' (default, more robust) or 'gauss_newton' (faster but may not converge)."
    )]
    pub algorithm: Option<String>,

    /// Maximum iterations
    #[schemars(description = "Maximum number of iterations (default: 200).")]
    pub max_iter: Option<usize>,
}

/// Request for LOESS (local polynomial regression) smoothing.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct LoessRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column to smooth.")]
    pub y: String,

    /// Independent variable (X) column name
    #[schemars(description = "Name of the independent variable (X) column.")]
    pub x: String,

    /// Smoothing parameter (span)
    #[schemars(
        description = "Smoothing parameter controlling neighborhood size. Range (0,1] uses proportion of data; >1 uses all points. Default: 0.75. Smaller = more wiggly, larger = smoother."
    )]
    pub span: Option<f64>,

    /// Polynomial degree (1=linear, 2=quadratic)
    #[schemars(
        description = "Degree of local polynomial: 1 (linear) or 2 (quadratic, default). Quadratic captures curvature better at boundaries."
    )]
    pub degree: Option<usize>,

    /// Use robust fitting
    #[schemars(
        description = "Use robust fitting with iterative reweighting to downweight outliers. Default: false (gaussian family). Set true for 'symmetric' family."
    )]
    pub robust: Option<bool>,
}

// ============================================================================
// Econometrics Tool Input Types
// ============================================================================

/// Request for Panel Fixed Effects regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PanelFERequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Entity/individual identifier column
    #[schemars(
        description = "Column name for entity/individual identifier (e.g., 'firm_id', 'person_id')."
    )]
    pub entity_var: String,
}

/// Request for Panel Random Effects regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PanelRERequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Entity/individual identifier column
    #[schemars(
        description = "Column name for entity/individual identifier (e.g., 'firm_id', 'person_id')."
    )]
    pub entity_var: String,
}

/// Request for Hausman specification test.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct HausmanRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Entity/individual identifier column
    #[schemars(
        description = "Column name for entity/individual identifier (e.g., 'firm_id', 'person_id')."
    )]
    pub entity_var: String,
}

/// Request for Arellano-Bond / System GMM dynamic panel estimation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GmmRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(
        description = "Names of the independent variable (X) columns. Include lagged dependent variable if desired."
    )]
    pub x: Vec<String>,

    /// Entity/individual identifier column
    #[schemars(
        description = "Column name for entity/individual identifier (e.g., 'firm_id', 'person_id')."
    )]
    pub entity_var: String,

    /// Time period identifier column
    #[schemars(description = "Column name for time period identifier (e.g., 'year', 'quarter').")]
    pub time_var: String,

    /// Number of lags of the dependent variable to include
    #[schemars(description = "Number of lags of Y to use as regressors. Default: 1.")]
    pub lags: Option<usize>,

    /// Transformation type: 'difference' (Arellano-Bond 1991) or 'system' (Blundell-Bond 1998)
    #[schemars(
        description = "Transformation: 'difference' for Arellano-Bond, 'system' for Blundell-Bond. Default: 'difference'."
    )]
    pub transform: Option<String>,

    /// Estimation step: 'onestep' or 'twostep'
    #[schemars(
        description = "Estimation step: 'onestep' or 'twostep'. Two-step is more efficient. Default: 'twostep'."
    )]
    pub step: Option<String>,

    /// Maximum lag for instruments
    #[schemars(
        description = "Maximum lag of Y to use as instruments. None means all available lags. Default: None."
    )]
    pub max_lag: Option<usize>,

    /// Minimum lag for instruments
    #[schemars(description = "Minimum lag of Y to use as instruments (must be >= 2). Default: 2.")]
    pub min_lag: Option<usize>,

    /// Collapse instruments to reduce count
    #[schemars(
        description = "If true, collapse instruments to reduce instrument count and avoid overfitting. Default: false."
    )]
    pub collapse: Option<bool>,

    /// Use robust standard errors
    #[schemars(
        description = "Use robust (Windmeijer-corrected for two-step) standard errors. Default: true."
    )]
    pub robust: Option<bool>,
}

/// Request for Panel GLS (Feasible Generalized Least Squares) estimation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PanelGlsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Entity/individual identifier column
    #[schemars(
        description = "Column name for entity/individual identifier (e.g., 'firm_id', 'person_id')."
    )]
    pub entity_var: String,

    /// Time period identifier column
    #[schemars(description = "Column name for time period identifier (e.g., 'year', 'quarter').")]
    pub time_var: String,

    /// Model type: 'fe' (fixed effects), 'pooling', or 'fd' (first difference)
    #[schemars(
        description = "Model type: 'fe' for fixed effects GLS (default), 'pooling' for pooled GLS, 'fd' for first-difference GLS."
    )]
    pub model: Option<String>,
}

/// Request for Variable Coefficients Model (PVCM).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PvcmRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded panel dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Entity/individual identifier column
    #[schemars(
        description = "Column name for entity/individual identifier (e.g., 'firm_id', 'person_id')."
    )]
    pub entity_var: String,

    /// Model type: 'within' or 'random'
    #[schemars(
        description = "Model type: 'within' for separate OLS per entity (default), 'random' for Swamy (1970) GLS estimator."
    )]
    pub model: Option<String>,
}

/// Request for panel unit root tests.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PanelUnitRootRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded panel dataset.")]
    pub dataset: String,

    /// Variable to test for unit root
    #[schemars(description = "Name of the variable column to test for unit root.")]
    pub variable: String,

    /// Unit/entity identifier column
    #[schemars(
        description = "Column name for panel unit identifier (e.g., 'country', 'firm_id')."
    )]
    pub unit_col: String,

    /// Time period column
    #[schemars(description = "Column name for time period identifier (e.g., 'year', 'quarter').")]
    pub time_col: String,

    /// Test type
    #[schemars(
        description = "Test type: 'llc' (Levin-Lin-Chu, default), 'ips' (Im-Pesaran-Shin), 'fisher' (Maddala-Wu), 'hadri' (stationarity null)."
    )]
    pub test: Option<String>,

    /// Model specification
    #[schemars(
        description = "Model: 'intercept' (default, individual intercepts), 'trend' (intercepts + trends), 'none' (no deterministics)."
    )]
    pub model: Option<String>,

    /// Number of lags
    #[schemars(
        description = "Number of lags for ADF-type regressions. None for automatic selection."
    )]
    pub lags: Option<usize>,
}

// ============================================================================
// Spatial Econometrics Request Types
// ============================================================================

/// Request for creating spatial neighbors.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SpatialNeighborsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column containing longitude or x coordinate
    #[schemars(description = "Column name for longitude or x coordinate.")]
    pub x_coord: String,

    /// Column containing latitude or y coordinate
    #[schemars(description = "Column name for latitude or y coordinate.")]
    pub y_coord: String,

    /// Neighbor method: 'knn' (default), 'distance', or 'distance_longlat'
    #[schemars(
        description = "Method for defining neighbors: 'knn' (k-nearest neighbors, default), 'distance' (within distance), 'distance_longlat' (great-circle distance for lon/lat)."
    )]
    pub method: Option<String>,

    /// Number of neighbors for knn method
    #[schemars(description = "Number of nearest neighbors (for 'knn' method). Default is 5.")]
    pub k: Option<usize>,

    /// Maximum distance (for distance-based methods)
    #[schemars(
        description = "Maximum distance threshold (for 'distance' or 'distance_longlat' methods). Units are in coordinate units or kilometers for longlat."
    )]
    pub d_max: Option<f64>,

    /// Minimum distance (for distance-based methods)
    #[schemars(description = "Minimum distance threshold (for 'distance' methods). Default is 0.")]
    pub d_min: Option<f64>,

    /// Name to store the spatial weights under
    #[schemars(
        description = "Name to store the spatial weights for later use. If not provided, uses dataset name + '_weights'."
    )]
    pub weights_name: Option<String>,

    /// Weight style
    #[schemars(
        description = "Weight style: 'W' or 'row' (row-standardized, default), 'B' (binary), 'C' (global standardized), 'U' (unstandardized)."
    )]
    pub style: Option<String>,
}

/// Request for Moran's I test for spatial autocorrelation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MoranTestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Variable to test for spatial autocorrelation
    #[schemars(description = "Name of the variable column to test for spatial autocorrelation.")]
    pub variable: String,

    /// Name of stored spatial weights
    #[schemars(
        description = "Name of previously created spatial weights (from spatial_neighbors tool)."
    )]
    pub weights: String,

    /// Alternative hypothesis
    #[schemars(
        description = "Alternative hypothesis: 'greater' (positive autocorrelation, default), 'less' (negative), 'two.sided'."
    )]
    pub alternative: Option<String>,
}

/// Request for spatial LM tests.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SpatialLmTestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Name of stored spatial weights
    #[schemars(
        description = "Name of previously created spatial weights (from spatial_neighbors tool)."
    )]
    pub weights: String,
}

/// Request for Spatial Autoregressive (SAR) model.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SarModelRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Name of stored spatial weights
    #[schemars(
        description = "Name of previously created spatial weights (from spatial_neighbors tool)."
    )]
    pub weights: String,

    /// Spatial Durbin model (include WX)
    #[schemars(
        description = "If true, estimates Spatial Durbin Model (SDM) which includes spatially lagged covariates (WX). Default is false."
    )]
    pub durbin: Option<bool>,

    /// Compute spatial impacts
    #[schemars(
        description = "If true, computes direct, indirect, and total spatial impacts. Default is true."
    )]
    pub compute_impacts: Option<bool>,
}

/// Request for Spatial Error Model (SEM).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SemModelRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Name of stored spatial weights
    #[schemars(
        description = "Name of previously created spatial weights (from spatial_neighbors tool)."
    )]
    pub weights: String,
}

/// Request for Spatial GMM with Heteroscedasticity Robustness (sphet).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SphetRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Name of stored spatial weights
    #[schemars(
        description = "Name of previously created spatial weights (from spatial_neighbors tool)."
    )]
    pub weights: String,

    /// Model type: 'lag' (SAR), 'error' (SEM), or 'sarar' (both)
    #[schemars(
        description = "Model type: 'lag' for SAR (y=lambda*Wy+Xb+e), 'error' for SEM (y=Xb+u, u=rho*Wu+e), or 'sarar' for combined. Default is 'lag'."
    )]
    pub model: Option<String>,

    /// Standard error type: 'robust', 'hac', or 'standard'
    #[schemars(
        description = "Standard error type: 'robust' for heteroscedasticity-robust (Kelejian-Prucha 2010), 'hac' for HAC (Kelejian-Prucha 2007), or 'standard' for homoscedastic. Default is 'robust'."
    )]
    pub se_type: Option<String>,

    /// HAC kernel type (for se_type='hac')
    #[schemars(
        description = "HAC kernel: 'bartlett', 'parzen', 'quadratic_spectral', 'tukey_hanning', or 'truncated'. Default is 'bartlett'."
    )]
    pub kernel: Option<String>,

    /// HAC bandwidth (for se_type='hac')
    #[schemars(
        description = "Bandwidth for HAC estimation. If not specified, uses automatic bandwidth selection."
    )]
    pub bandwidth: Option<usize>,

    /// Instrument order (default 2)
    #[schemars(description = "Order of spatial lag instruments [X, WX, W^2X, ...]. Default is 2.")]
    pub instrument_order: Option<usize>,
}

/// Request for SAR Probit model (spatial lag probit).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SarProbitRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Binary dependent variable (Y) column name
    #[schemars(description = "Name of the binary dependent variable (Y) column (0/1 values).")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Name of stored spatial weights
    #[schemars(
        description = "Name of previously created spatial weights (from spatial_neighbors tool)."
    )]
    pub weights: String,

    /// Number of MCMC draws
    #[schemars(description = "Number of MCMC draws after burn-in. Default is 1000.")]
    pub n_draws: Option<usize>,

    /// Burn-in draws
    #[schemars(description = "Number of burn-in draws to discard. Default is 200.")]
    pub burn_in: Option<usize>,

    /// Compute spatial impacts
    #[schemars(
        description = "If true, computes direct, indirect, and total spatial impacts. Default is true."
    )]
    pub compute_impacts: Option<bool>,

    /// Random seed
    #[schemars(description = "Random seed for reproducibility. Optional.")]
    pub seed: Option<u64>,
}

/// Request for SEM Probit model (spatial error probit).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SemProbitRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Binary dependent variable (Y) column name
    #[schemars(description = "Name of the binary dependent variable (Y) column (0/1 values).")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Name of stored spatial weights
    #[schemars(
        description = "Name of previously created spatial weights (from spatial_neighbors tool)."
    )]
    pub weights: String,

    /// Number of MCMC draws
    #[schemars(description = "Number of MCMC draws after burn-in. Default is 1000.")]
    pub n_draws: Option<usize>,

    /// Burn-in draws
    #[schemars(description = "Number of burn-in draws to discard. Default is 200.")]
    pub burn_in: Option<usize>,

    /// Random seed
    #[schemars(description = "Random seed for reproducibility. Optional.")]
    pub seed: Option<u64>,
}

/// Request for Spatial Panel ML estimation (spml).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SpmlRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset containing panel data.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Entity identifier column
    #[schemars(description = "Name of the entity/cross-sectional identifier column.")]
    pub entity_col: String,

    /// Time identifier column
    #[schemars(description = "Name of the time period identifier column.")]
    pub time_col: String,

    /// Name of stored spatial weights
    #[schemars(
        description = "Name of previously created spatial weights (from spatial_neighbors tool). Must match number of cross-sectional entities."
    )]
    pub weights: String,

    /// Panel model type
    #[schemars(
        description = "Panel model type: 'within' (fixed effects, default), 'random' (random effects), or 'pooling' (no effects)."
    )]
    pub model: Option<String>,

    /// Include spatial lag
    #[schemars(
        description = "If true, includes spatial lag of dependent variable (rho*W*y). Default is false."
    )]
    pub lag: Option<bool>,

    /// Spatial error type
    #[schemars(
        description = "Spatial error specification: 'none' (default), 'baltagi' (Baltagi-type), or 'kkp' (Kapoor-Kelejian-Prucha type)."
    )]
    pub spatial_error: Option<String>,

    /// Effect type
    #[schemars(description = "Effect type: 'individual' (default), 'time', or 'twoways'.")]
    pub effect: Option<String>,
}

/// Request for Spatial Panel GMM estimation (spgm).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SpgmRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset containing panel data.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Entity identifier column
    #[schemars(description = "Name of the entity/cross-sectional identifier column.")]
    pub entity_col: String,

    /// Time identifier column
    #[schemars(description = "Name of the time period identifier column.")]
    pub time_col: String,

    /// Name of stored spatial weights
    #[schemars(
        description = "Name of previously created spatial weights (from spatial_neighbors tool)."
    )]
    pub weights: String,

    /// Estimation method
    #[schemars(
        description = "GMM estimation method: 'w2sls' (within/fixed effects, default), 'g2sls' (GLS random effects), 'b2sls' (between), or 'ec2sls' (Baltagi EC2SLS)."
    )]
    pub method: Option<String>,

    /// Include spatial lag
    #[schemars(
        description = "If true, includes spatial lag of dependent variable (uses IV/GMM for identification). Default is false."
    )]
    pub lag: Option<bool>,

    /// Include spatial error
    #[schemars(
        description = "If true, includes spatially correlated error term. Default is true."
    )]
    pub spatial_error: Option<bool>,
}

/// Request for IV/2SLS regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct IV2SLSRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Exogenous independent variables
    #[schemars(
        description = "Names of exogenous independent variable columns (not instrumented)."
    )]
    pub x_exog: Vec<String>,

    /// Endogenous variable to be instrumented
    #[schemars(description = "Names of endogenous variables that need instruments.")]
    pub x_endog: Vec<String>,

    /// Instrumental variables
    #[schemars(description = "Names of instrument columns (excluded from structural equation).")]
    pub instruments: Vec<String>,

    /// Use robust standard errors
    #[schemars(
        description = "Whether to use heteroskedasticity-robust standard errors. Default is true."
    )]
    pub robust: Option<bool>,
}

/// Request for first-stage diagnostics.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FirstStageRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Endogenous variable name
    #[schemars(description = "Name of the endogenous variable to test instrument strength for.")]
    pub endogenous_var: String,

    /// Instrument variable names
    #[schemars(
        description = "Names of the instrumental variables (e.g., ['parents_edu', 'distance_to_college'])."
    )]
    pub instruments: Vec<String>,

    /// Control variable names (optional)
    #[schemars(description = "Optional control variables to include in first-stage regression.")]
    pub controls: Option<Vec<String>>,
}

/// Request for Sargan test of overidentifying restrictions.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SarganTestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Exogenous independent variables
    #[schemars(
        description = "Names of exogenous independent variable columns (not instrumented). May be empty."
    )]
    pub x_exog: Vec<String>,

    /// Endogenous variable to be instrumented
    #[schemars(description = "Names of endogenous variables that need instruments.")]
    pub x_endog: Vec<String>,

    /// Instrumental variables
    #[schemars(
        description = "Names of instrument columns. Must exceed number of endogenous variables for test to be valid."
    )]
    pub instruments: Vec<String>,
}

/// Request for Balke-Pearl bounds on the Average Causal Effect (ACE).
///
/// Balke-Pearl bounds provide sharp nonparametric bounds on the causal effect
/// using instrumental variables without assuming parametric models.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct BPBoundsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Instrument column (binary 0/1)
    #[schemars(
        description = "Name of the binary instrument column (Z). Example: randomized treatment assignment."
    )]
    pub instrument: String,

    /// Treatment column (binary 0/1)
    #[schemars(
        description = "Name of the binary treatment received column (D). Example: actual treatment uptake."
    )]
    pub treatment: String,

    /// Outcome column (binary 0/1)
    #[schemars(description = "Name of the binary outcome column (Y). Example: recovery status.")]
    pub outcome: String,

    /// Assume monotonicity (no defiers)
    #[schemars(
        description = "Whether to assume monotonicity (no defiers). If true, bounds tighten but assumption may be violated. Default is false."
    )]
    pub monotonicity: Option<bool>,

    /// Compute bootstrap confidence intervals
    #[schemars(
        description = "Whether to compute bootstrap confidence intervals for the bounds. Default is true."
    )]
    pub bootstrap_ci: Option<bool>,

    /// Number of bootstrap replications
    #[schemars(
        description = "Number of bootstrap replications for confidence intervals. Default is 1000."
    )]
    pub n_bootstrap: Option<usize>,

    /// Confidence level (1 - alpha)
    #[schemars(description = "Confidence level for intervals. Default is 0.95 (95% CI).")]
    pub confidence_level: Option<f64>,

    /// Random seed for reproducibility
    #[schemars(description = "Optional random seed for bootstrap reproducibility.")]
    pub seed: Option<u64>,
}

/// Request for Marginal Treatment Effects (MTE) estimation.
///
/// The MTE framework (Heckman & Vytlacil 2005) connects IV estimation to a
/// choice-theoretic model of treatment selection, revealing heterogeneity
/// in treatment effects.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct IVMTERequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable (Y) column name
    #[schemars(description = "Name of the outcome variable (Y) column.")]
    pub y: String,

    /// Treatment indicator column (binary 0/1)
    #[schemars(description = "Name of the binary treatment indicator column (D = 0 or 1).")]
    pub d: String,

    /// Instrument column name
    #[schemars(
        description = "Name of the instrumental variable (Z) column that affects treatment but not outcome directly."
    )]
    pub z: String,

    /// Covariate columns (optional)
    #[schemars(description = "Optional covariate column names to include in the second stage.")]
    pub x: Option<Vec<String>>,

    /// Polynomial degree for MTE curve
    #[schemars(
        description = "Polynomial degree for MTE approximation. Higher degrees allow more flexible MTE shapes but may overfit. Default is 2."
    )]
    pub mte_degree: Option<usize>,

    /// Propensity score model type
    #[schemars(
        description = "Propensity score model: 'probit' (default), 'logit', or 'linear'. Probit is standard in the MTE literature."
    )]
    pub propensity_model: Option<String>,

    /// Number of grid points for MTE curve
    #[schemars(description = "Number of grid points for evaluating MTE curve. Default is 100.")]
    pub n_grid: Option<usize>,
}

/// Request for Callaway-Sant'Anna staggered DiD estimation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct StaggeredDiDRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded panel dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column.")]
    pub outcome: String,

    /// Treatment timing column
    #[schemars(
        description = "Column indicating when each unit was first treated (period number). Use 0 or negative for never-treated units."
    )]
    pub treatment_time: String,

    /// Time period column
    #[schemars(
        description = "Column containing the time period identifier (e.g., year, quarter)."
    )]
    pub time_col: String,

    /// Unit identifier column
    #[schemars(
        description = "Column containing the unit/individual identifier (e.g., state_id, firm_id)."
    )]
    pub unit_col: String,

    /// Covariate columns (optional)
    #[schemars(description = "Optional covariates for conditional parallel trends assumption.")]
    pub covariates: Option<Vec<String>>,

    /// Comparison group strategy
    #[schemars(
        description = "Comparison group: 'never_treated' (default) uses only never-treated units, 'not_yet_treated' uses units not yet treated by that period."
    )]
    pub comparison_group: Option<String>,

    /// Estimation method
    #[schemars(
        description = "Estimation method: 'outcome_regression' (default), 'ipw', or 'doubly_robust'."
    )]
    pub estimation_method: Option<String>,

    /// Base period relative to treatment
    #[schemars(
        description = "Base period for pre-treatment comparison, relative to g. Default is -1 (one period before treatment)."
    )]
    pub base_period: Option<i32>,

    /// Number of bootstrap replications
    #[schemars(
        description = "Number of bootstrap replications for standard errors. Default is 999."
    )]
    pub bootstrap: Option<usize>,
}

/// Request for Goodman-Bacon decomposition of staggered DiD.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct BaconDecompRequest {
    /// Name/ID of the dataset
    #[schemars(
        description = "Name or ID of a previously loaded panel dataset with staggered treatment."
    )]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column.")]
    pub outcome: String,

    /// Unit identifier column
    #[schemars(
        description = "Column containing the unit/individual identifier (e.g., state_id, firm_id)."
    )]
    pub unit_col: String,

    /// Time period column
    #[schemars(
        description = "Column containing the time period identifier (e.g., year, quarter)."
    )]
    pub time_col: String,

    /// Treatment indicator column
    #[schemars(
        description = "Binary treatment indicator column (0 = untreated, 1 = treated). Should be 0 before treatment and 1 after for each unit."
    )]
    pub treatment_col: String,
}

/// Request for Extended Two-Way Fixed Effects (ETWFE) estimation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct EtwfeRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded panel dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column.")]
    pub outcome: String,

    /// Unit identifier column
    #[schemars(
        description = "Column containing the unit/individual identifier (e.g., state_id, firm_id)."
    )]
    pub unit_col: String,

    /// Time period column
    #[schemars(
        description = "Column containing the time period identifier (e.g., year, quarter)."
    )]
    pub time_col: String,

    /// Treatment indicator column
    #[schemars(
        description = "Column indicating treatment status (1 = currently treated, 0 = not treated). Binary indicator."
    )]
    pub treatment: String,

    /// First treatment period column
    #[schemars(
        description = "Column indicating when each unit was first treated (period number). Use 0 for never-treated units."
    )]
    pub first_treat: String,

    /// Control variables (optional)
    #[schemars(description = "Optional control variable columns.")]
    pub controls: Option<Vec<String>>,

    /// Control group strategy
    #[schemars(
        description = "Control group: 'notyet' (default) uses not-yet-treated units, 'never' uses only never-treated units."
    )]
    pub cgroup: Option<String>,
}

/// Request for Inverse Probability Weighting (IPW) treatment effect estimation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct IpwRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column.")]
    pub outcome: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    pub treatment: String,

    /// Covariate columns for propensity score model
    #[schemars(description = "Names of covariate columns to include in propensity score model.")]
    pub covariates: Vec<String>,

    /// Estimand: 'ate' (Average Treatment Effect) or 'att' (Average Treatment Effect on Treated)
    #[schemars(
        description = "Treatment effect estimand: 'ate' for Average Treatment Effect (default), 'att' for Average Treatment Effect on Treated."
    )]
    pub estimand: Option<String>,

    /// Trimming threshold for extreme propensity scores
    #[schemars(
        description = "Trim observations with propensity scores below trim or above 1-trim. Default is 0.05."
    )]
    pub trim: Option<f64>,

    /// Number of bootstrap replications for standard errors
    #[schemars(
        description = "Number of bootstrap replications for standard error estimation. Default is 999."
    )]
    pub bootstrap: Option<usize>,
}

/// Request for Doubly Robust (AIPW) treatment effect estimation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DoublyRobustRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column.")]
    pub outcome: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    pub treatment: String,

    /// Covariate columns for propensity score and outcome models
    #[schemars(
        description = "Names of covariate columns to include in both propensity score and outcome models."
    )]
    pub covariates: Vec<String>,

    /// Estimation method: 'aipw' (default), 'ipw', or 'regression'
    #[schemars(
        description = "Estimation method: 'aipw' for Augmented IPW (default, doubly robust), 'ipw' for IPW only, 'regression' for outcome regression only."
    )]
    pub method: Option<String>,

    /// Estimand: 'ate' (Average Treatment Effect) or 'att' (Average Treatment Effect on Treated)
    #[schemars(
        description = "Treatment effect estimand: 'ate' for Average Treatment Effect (default), 'att' for Average Treatment Effect on Treated."
    )]
    pub estimand: Option<String>,

    /// Trimming threshold for extreme propensity scores
    #[schemars(
        description = "Trim observations with propensity scores below trim or above 1-trim. Default is 0.05."
    )]
    pub trim: Option<f64>,

    /// Number of bootstrap replications for standard errors
    #[schemars(
        description = "Number of bootstrap replications for standard error estimation. Default is 999."
    )]
    pub bootstrap: Option<usize>,
}

/// Request for Double/Debiased Machine Learning (DoubleML) estimation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DoubleMLRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column (Y).")]
    pub outcome: String,

    /// Treatment variable column name
    #[schemars(
        description = "Name of the treatment variable column (D). Can be continuous or binary."
    )]
    pub treatment: String,

    /// Covariate columns for nuisance model estimation
    #[schemars(description = "Names of covariate columns (X) for nuisance model estimation.")]
    pub covariates: Vec<String>,

    /// Model type: 'plr' (Partially Linear Regression, default) or 'irm' (Interactive Regression Model)
    #[schemars(
        description = "DML model type: 'plr' for Partially Linear Regression (default, Y = theta*D + g(X) + eps), 'irm' for Interactive Regression Model (binary treatment, heterogeneous effects)."
    )]
    pub model_type: Option<String>,

    /// Number of cross-fitting folds (default: 5)
    #[schemars(
        description = "Number of folds for cross-fitting. Default is 5. Must be at least 2."
    )]
    pub n_folds: Option<usize>,

    /// Random seed for reproducible fold splits
    #[schemars(
        description = "Random seed for reproducible cross-fitting splits. If omitted, uses random seed."
    )]
    pub seed: Option<u64>,

    /// Trimming threshold for propensity scores (IRM only)
    #[schemars(
        description = "Trim propensity scores to [trim, 1-trim] for IRM model. Default is 0.01."
    )]
    pub trim: Option<f64>,
}

/// Request for Covariate Balancing Propensity Score (CBPS) estimation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CbpsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    pub treatment: String,

    /// Covariate columns for propensity score model
    #[schemars(description = "Names of covariate columns to include in propensity score model.")]
    pub covariates: Vec<String>,

    /// CBPS method: 'exact' (default), 'over', or 'just'
    #[schemars(
        description = "CBPS method: 'exact' for exact balance (default, overidentified), 'over' for over-balanced, 'just' for just-identified (standard logit)."
    )]
    pub method: Option<String>,

    /// Standardized difference threshold for balance
    #[schemars(
        description = "Threshold for standardized difference to consider a covariate balanced. Default is 0.1."
    )]
    pub balance_threshold: Option<f64>,
}

/// Request for flexible inverse probability weighting (WeightIt).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct WeightItRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    pub treatment: String,

    /// Covariate columns for balance
    #[schemars(description = "Names of covariate columns to balance between treatment groups.")]
    pub covariates: Vec<String>,

    /// Weighting method: 'logistic' (default), 'entropy', 'energy', or 'stable'
    #[schemars(
        description = "Weighting method: 'logistic' (standard PS, default), 'entropy' (entropy balancing), 'energy' (energy distance), 'stable' (stable weights)."
    )]
    pub method: Option<String>,

    /// Target estimand: 'ate' (default), 'att', or 'atc'
    #[schemars(
        description = "Target estimand: 'ate' (average treatment effect, default), 'att' (on treated), 'atc' (on control)."
    )]
    pub estimand: Option<String>,

    /// Whether to stabilize weights
    #[schemars(
        description = "Whether to stabilize weights by multiplying by marginal treatment probability. Default is false."
    )]
    pub stabilize: Option<bool>,

    /// Trimming quantile for extreme weights
    #[schemars(
        description = "Quantile for trimming extreme weights (e.g., 0.99 trims at 1st and 99th percentile). Default is 1.0 (no trimming)."
    )]
    pub trim_quantile: Option<f64>,
}

/// Request for entropy balancing.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct EntropyBalanceRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    pub treatment: String,

    /// Covariate columns for balance
    #[schemars(description = "Names of covariate columns to balance exactly on means.")]
    pub covariates: Vec<String>,

    /// Optional target means (defaults to treated group means for ATT)
    #[schemars(
        description = "Optional target means for covariates. If not provided, uses treated group means (ATT)."
    )]
    pub target_means: Option<Vec<f64>>,
}

/// Request for stable balancing weights (SBW).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SBWRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    pub treatment: String,

    /// Covariate columns for balance
    #[schemars(description = "Names of covariate columns to balance between treatment groups.")]
    pub covariates: Vec<String>,

    /// Target estimand: 'att' (default), 'ate', or 'atc'
    #[schemars(
        description = "Target estimand: 'att' (effect on treated, default), 'ate' (average treatment effect), 'atc' (effect on control)."
    )]
    pub estimand: Option<String>,

    /// Balance tolerance for approximate balance (0 = exact balance)
    #[schemars(
        description = "Tolerance for approximate balance. 0 means exact balance (default), positive values allow some deviation."
    )]
    pub balance_tol: Option<f64>,

    /// Minimum weight allowed (default 0 for non-negativity)
    #[schemars(description = "Minimum weight allowed. Default is 0 (non-negativity constraint).")]
    pub min_weight: Option<f64>,

    /// Penalty parameter for approximate balance (higher = stricter balance)
    #[schemars(
        description = "Penalty parameter for approximate balance. Higher values enforce stricter balance. Default is 1000."
    )]
    pub balance_penalty: Option<f64>,
}

/// Request for twang GBM propensity score estimation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct TwangRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    pub treatment: String,

    /// Covariate columns for propensity model
    #[schemars(
        description = "Names of covariate columns to include in the GBM propensity score model."
    )]
    pub covariates: Vec<String>,

    /// Stopping rule: 'es.mean' (default), 'es.max', 'ks.mean', 'ks.max'
    #[schemars(
        description = "Stopping rule for selecting optimal iterations: 'es.mean' (mean standardized effect size, default), 'es.max' (max effect size), 'ks.mean' (mean KS statistic), 'ks.max' (max KS statistic)."
    )]
    pub stop_method: Option<String>,

    /// Target estimand: 'att' (default), 'ate', or 'atc'
    #[schemars(
        description = "Target estimand: 'att' (effect on treated, default), 'ate' (average treatment effect), 'atc' (effect on control)."
    )]
    pub estimand: Option<String>,

    /// Maximum number of boosting iterations (default: 3000)
    #[schemars(description = "Maximum number of gradient boosting iterations. Default is 3000.")]
    pub n_trees: Option<usize>,

    /// Learning rate / shrinkage (default: 0.01)
    #[schemars(
        description = "Learning rate for gradient boosting. Smaller values need more iterations but often give better results. Default is 0.01."
    )]
    pub shrinkage: Option<f64>,

    /// Balance threshold for early stopping (default: 0.1)
    #[schemars(description = "Balance threshold below which to stop early. Default is 0.1.")]
    pub balance_threshold: Option<f64>,
}

/// Request for propensity score matching (MatchIt).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MatchItRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    pub treatment: String,

    /// Covariate columns for matching
    #[schemars(description = "Names of covariate columns to use for matching.")]
    pub covariates: Vec<String>,

    /// Matching method: 'nearest' (default), 'cem', 'full', or 'subclass'
    #[schemars(
        description = "Matching method: 'nearest' (nearest neighbor, default), 'cem' (coarsened exact matching), 'full' (full/optimal matching), 'subclass' (propensity score subclassification)."
    )]
    pub method: Option<String>,

    /// Distance metric: 'logit' (default), 'probit', 'mahalanobis', or 'euclidean'
    #[schemars(
        description = "Distance metric: 'logit' (propensity score via logit, default), 'probit', 'mahalanobis', 'euclidean'."
    )]
    pub distance: Option<String>,

    /// Matching ratio (1:k matching, default k=1)
    #[schemars(
        description = "For nearest neighbor: number of controls per treated unit (1:k matching). Default is 1."
    )]
    pub ratio: Option<usize>,

    /// Caliper width (in SD of propensity score)
    #[schemars(
        description = "For nearest neighbor: maximum distance for a valid match, in SD of propensity score. Default is no caliper."
    )]
    pub caliper: Option<f64>,

    /// Whether to sample with replacement
    #[schemars(
        description = "For nearest neighbor: whether to sample controls with replacement. Default is false."
    )]
    pub replace: Option<bool>,

    /// Number of subclasses for subclassification
    #[schemars(
        description = "For subclassification: number of subclasses to create. Default is 5."
    )]
    pub n_subclasses: Option<usize>,
}

/// Request for Targeted Maximum Likelihood Estimation (TMLE).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct TmleRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column. Can be binary or continuous.")]
    pub outcome: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    pub treatment: String,

    /// Covariate columns for propensity score and outcome models
    #[schemars(
        description = "Names of covariate columns to include in both propensity score and outcome models."
    )]
    pub covariates: Vec<String>,

    /// Outcome model type: 'logistic' (default) or 'linear'
    #[schemars(
        description = "Outcome model type: 'logistic' for binary outcomes (default), 'linear' for continuous outcomes."
    )]
    pub q_model: Option<String>,

    /// Lower bound for propensity score truncation
    #[schemars(description = "Lower bound for propensity score truncation. Default is 0.01.")]
    pub ps_lower: Option<f64>,

    /// Upper bound for propensity score truncation
    #[schemars(description = "Upper bound for propensity score truncation. Default is 0.99.")]
    pub ps_upper: Option<f64>,
}

/// Request for Collaborative Targeted Maximum Likelihood Estimation (C-TMLE).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CTmleRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column. Can be binary or continuous.")]
    pub outcome: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    pub treatment: String,

    /// Covariate columns (candidates for propensity score selection)
    #[schemars(
        description = "Names of candidate covariate columns. C-TMLE will select which ones to include in the propensity score model via cross-validation."
    )]
    pub covariates: Vec<String>,

    /// Outcome model type: 'logistic' (default) or 'linear'
    #[schemars(
        description = "Outcome model type: 'logistic' for binary outcomes (default), 'linear' for continuous outcomes."
    )]
    pub q_model: Option<String>,

    /// Number of cross-validation folds (default: 5)
    #[schemars(
        description = "Number of cross-validation folds for covariate selection. Default is 5."
    )]
    pub n_folds: Option<usize>,

    /// Maximum number of covariates to select (optional)
    #[schemars(
        description = "Maximum number of covariates to include in propensity score model. Default is no limit."
    )]
    pub max_covariates: Option<usize>,

    /// Stopping rule: 'cv_minimum' (default), 'one_se', or 'max_covariates'
    #[schemars(
        description = "Stopping rule for selection: 'cv_minimum' (stop at minimum CV risk, default), 'one_se' (one-standard-error rule for parsimony), 'max_covariates' (use max_covariates parameter)."
    )]
    pub stopping_rule: Option<String>,

    /// Lower bound for propensity score truncation (default: 0.025)
    #[schemars(description = "Lower bound for propensity score truncation. Default is 0.025.")]
    pub ps_lower: Option<f64>,

    /// Upper bound for propensity score truncation (default: 0.975)
    #[schemars(description = "Upper bound for propensity score truncation. Default is 0.975.")]
    pub ps_upper: Option<f64>,
}

/// Request for Parametric G-Formula with time-varying treatments.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GFormulaRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded panel dataset.")]
    pub dataset: String,

    /// Outcome variable column name (observed at final time point)
    #[schemars(description = "Name of the outcome variable column.")]
    pub outcome: String,

    /// Baseline (time-invariant) covariate column names
    #[schemars(description = "Names of baseline covariates that do not change over time.")]
    pub baseline_covariates: Vec<String>,

    /// Time-varying covariate column names (must have suffix _t0, _t1, etc.)
    #[schemars(
        description = "Base names of time-varying covariates. Columns must be named as 'varname_t0', 'varname_t1', etc. for each time point."
    )]
    pub time_varying_covariates: Vec<String>,

    /// Treatment column names for each time point (e.g., ['treat_t0', 'treat_t1'])
    #[schemars(
        description = "Column names for treatment at each time point. Order matters: first element is treatment at t=0, second at t=1, etc."
    )]
    pub treatment_cols: Vec<String>,

    /// Number of time points
    #[schemars(
        description = "Number of time points in the analysis (must match number of treatment columns)."
    )]
    pub time_points: usize,

    /// Intervention type: 'always_treat', 'never_treat', 'natural', or 'threshold'
    #[schemars(
        description = "Intervention type: 'always_treat' (default), 'never_treat', 'natural' (observed patterns), or 'threshold'."
    )]
    pub intervention: Option<String>,

    /// For threshold intervention: variable index (0-indexed into time-varying covariates)
    #[schemars(
        description = "For threshold intervention: index of the time-varying covariate to check (0-indexed)."
    )]
    pub threshold_variable: Option<usize>,

    /// For threshold intervention: cutoff value
    #[schemars(
        description = "For threshold intervention: threshold value for treatment decision."
    )]
    pub threshold_cutoff: Option<f64>,

    /// For threshold intervention: treat if above (true) or below (false)
    #[schemars(
        description = "For threshold intervention: if true, treat when variable > cutoff; if false, treat when variable <= cutoff."
    )]
    pub threshold_above: Option<bool>,

    /// Outcome type: 'continuous' (default), 'binary', or 'survival'
    #[schemars(
        description = "Outcome type: 'continuous' for linear model (default), 'binary' for logistic model, 'survival' for discrete hazard model."
    )]
    pub outcome_type: Option<String>,

    /// Number of Monte Carlo simulations (default: 1000)
    #[schemars(description = "Number of Monte Carlo simulations. Default is 1000.")]
    pub n_simulations: Option<usize>,

    /// Number of bootstrap samples for standard errors (default: 200)
    #[schemars(
        description = "Number of bootstrap samples for standard error estimation. Default is 200."
    )]
    pub n_bootstrap: Option<usize>,

    /// Confidence level (default: 0.95)
    #[schemars(description = "Confidence level for intervals. Default is 0.95.")]
    pub confidence_level: Option<f64>,

    /// Random seed for reproducibility
    #[schemars(description = "Optional random seed for reproducible results.")]
    pub seed: Option<u64>,
}

/// Request for Causal Mediation Analysis.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MediationRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column.")]
    pub outcome: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    pub treatment: String,

    /// Mediator variable column name
    #[schemars(
        description = "Name of the mediator variable column - the intermediate variable through which treatment may affect the outcome."
    )]
    pub mediator: String,

    /// Covariate columns for propensity score models
    #[schemars(
        description = "Names of covariate columns for adjustment in propensity score models."
    )]
    pub covariates: Vec<String>,

    /// Trimming threshold for extreme propensity scores
    #[schemars(
        description = "Trim observations with propensity scores below trim or above 1-trim. Default is 0.05."
    )]
    pub trim: Option<f64>,

    /// Number of bootstrap replications for standard errors
    #[schemars(
        description = "Number of bootstrap replications for standard error estimation. Default is 999."
    )]
    pub bootstrap: Option<usize>,
}

/// Request for Natural Effect Models (medflex) mediation analysis.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct NaturalEffectsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column (continuous).")]
    pub outcome: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Typically binary."
    )]
    pub treatment: String,

    /// Mediator variable column name
    #[schemars(
        description = "Name of the mediator variable column - the intermediate variable through which treatment may affect the outcome."
    )]
    pub mediator: String,

    /// Confounder columns for adjustment
    #[schemars(
        description = "Names of confounder columns for adjustment in mediator and outcome models. Can be empty."
    )]
    pub confounders: Option<Vec<String>>,

    /// Whether to include treatment-mediator interaction
    #[schemars(
        description = "Include treatment-mediator interaction term in outcome model. Default is true. Set to false for simple product-of-coefficients decomposition."
    )]
    pub allow_interaction: Option<bool>,

    /// Number of bootstrap replications for standard errors
    #[schemars(
        description = "Number of bootstrap replications for confidence intervals. Default is 1000. Set to 0 to use delta method instead."
    )]
    pub n_bootstrap: Option<usize>,

    /// Confidence level for intervals
    #[schemars(
        description = "Confidence level for intervals (e.g., 0.95 for 95% CI). Default is 0.95."
    )]
    pub confidence_level: Option<f64>,

    /// Effect scale (difference, ratio, odds_ratio)
    #[schemars(
        description = "Scale for reporting effects: 'difference' (default for continuous outcomes), 'ratio' (for log-link), 'odds_ratio' (for logit)."
    )]
    pub scale: Option<String>,
}

/// Predictor specification for synthetic control.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SynthPredictorSpec {
    /// Column name of the predictor variable
    #[schemars(description = "Column name of the predictor variable.")]
    pub column: String,

    /// How to aggregate the predictor over time
    #[schemars(description = "Aggregation method: 'mean' (default), 'first', 'last', or 'sum'.")]
    pub aggregation: Option<String>,

    /// Optional time window (start, end) for aggregation
    #[schemars(
        description = "Time window for predictor aggregation as [start, end]. If omitted, uses all pre-treatment periods."
    )]
    pub time_window: Option<(i64, i64)>,
}

/// Request for Synthetic Control Method.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SyntheticControlRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded panel dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column.")]
    pub outcome: String,

    /// Unit identifier column name
    #[schemars(description = "Name of the column identifying units (e.g., 'state', 'country').")]
    pub unit_col: String,

    /// Time period column name
    #[schemars(
        description = "Name of the column identifying time periods (must be integer, e.g., 'year')."
    )]
    pub time_col: String,

    /// Name/ID of the treated unit
    #[schemars(description = "Name or ID of the treated unit (must match values in unit_col).")]
    pub treated_unit: String,

    /// Treatment time (first post-treatment period)
    #[schemars(
        description = "First post-treatment period (treatment starts at or after this time)."
    )]
    pub treatment_time: i64,

    /// Predictor specifications
    #[schemars(
        description = "List of predictor specifications. Can be column names (strings) or detailed specs with aggregation and time windows."
    )]
    pub predictors: Vec<SynthPredictorSpec>,

    /// V matrix optimization method
    #[schemars(
        description = "Method for predictor importance weights: 'datadriven' (default), 'equal', or 'custom'."
    )]
    pub v_method: Option<String>,

    /// Custom V weights (if v_method is 'custom')
    #[schemars(
        description = "Custom predictor weights (only used if v_method is 'custom'). Must sum to 1."
    )]
    pub custom_v_weights: Option<Vec<f64>>,

    /// Whether to run placebo tests for inference
    #[schemars(
        description = "Whether to run placebo tests for inference. Default is false (can be slow with many units)."
    )]
    pub run_placebos: Option<bool>,

    /// Optimization window (start, end)
    #[schemars(
        description = "Time window for optimization [start, end]. If omitted, uses all pre-treatment periods."
    )]
    pub optimization_window: Option<(i64, i64)>,

    /// Convergence tolerance
    #[schemars(description = "Tolerance for optimization convergence. Default is 1e-6.")]
    pub tolerance: Option<f64>,

    /// Maximum iterations for V optimization
    #[schemars(description = "Maximum iterations for V optimization. Default is 1000.")]
    pub max_iter: Option<usize>,

    /// Minimum weight threshold for output
    #[schemars(
        description = "Minimum weight to display in output (for readability). Default is 0.001."
    )]
    pub weight_threshold: Option<f64>,
}

/// Request for Generalized Synthetic Control (gsynth) estimation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GsynthRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded panel dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column (Y).")]
    pub outcome: String,

    /// Treatment indicator column name
    #[schemars(
        description = "Name of the treatment indicator column (D, binary 0/1). Treatment can start at different times for different units."
    )]
    pub treatment: String,

    /// Unit identifier column name
    #[schemars(description = "Name of the column identifying units (e.g., 'state', 'country').")]
    pub unit_col: String,

    /// Time period column name
    #[schemars(description = "Name of the column identifying time periods (must be numeric).")]
    pub time_col: String,

    /// Covariate columns
    #[schemars(description = "Optional list of covariate column names to include in the model.")]
    pub covariates: Option<Vec<String>>,

    /// Number of factors (0 for auto-selection via CV)
    #[schemars(
        description = "Number of latent factors. Use 0 with cross_validate=true for automatic selection. Default is 2."
    )]
    pub n_factors: Option<usize>,

    /// Whether to cross-validate factor selection
    #[schemars(
        description = "Whether to select number of factors via cross-validation. Default is false."
    )]
    pub cross_validate: Option<bool>,

    /// Maximum factors to consider in CV
    #[schemars(
        description = "Maximum number of factors to consider during cross-validation. Default is 5."
    )]
    pub max_factors: Option<usize>,

    /// Estimator type
    #[schemars(
        description = "Estimator: 'ife' (interactive fixed effects, default) or 'mc' (matrix completion)."
    )]
    pub estimator: Option<String>,

    /// Fixed effects specification
    #[schemars(description = "Fixed effects: 'none', 'unit' (default), 'time', or 'twoWay'.")]
    pub force: Option<String>,

    /// Whether to compute bootstrap standard errors
    #[schemars(description = "Whether to compute bootstrap standard errors. Default is false.")]
    pub bootstrap_se: Option<bool>,

    /// Number of bootstrap iterations
    #[schemars(
        description = "Number of bootstrap iterations for standard errors. Default is 500."
    )]
    pub n_bootstrap: Option<usize>,
}

/// Request for Synthetic Control with Prediction Intervals (SCPI).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ScpiRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded panel dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column (Y).")]
    pub outcome: String,

    /// Unit identifier column name
    #[schemars(description = "Name of the column identifying units (e.g., 'state', 'country').")]
    pub unit_col: String,

    /// Time period column name
    #[schemars(description = "Name of the column identifying time periods (must be numeric).")]
    pub time_col: String,

    /// Treated unit identifier
    #[schemars(description = "Identifier of the treated unit (must match a value in unit_col).")]
    pub treated_unit: String,

    /// Treatment time period
    #[schemars(description = "First post-treatment time period (treatment starts at this time).")]
    pub treatment_time: i64,

    /// Constraint type
    #[schemars(
        description = "Weight constraint: 'simplex' (default, sum=1, non-negative), 'lasso', 'ridge', or 'lasso_simplex'."
    )]
    pub constraint: Option<String>,

    /// Lambda for Lasso/Ridge constraints
    #[schemars(
        description = "Regularization parameter for Lasso or Ridge constraints. Default is 0.1."
    )]
    pub lambda: Option<f64>,

    /// Significance level
    #[schemars(
        description = "Significance level for prediction intervals. Default is 0.05 (95% PI)."
    )]
    pub alpha: Option<f64>,

    /// Variance estimation method
    #[schemars(
        description = "Out-of-sample variance method: 'subgaussian' (default, more conservative), 'gaussian', 'loo_cv', or 'kfold_cv'."
    )]
    pub variance_method: Option<String>,

    /// Number of CV folds
    #[schemars(
        description = "Number of folds for K-fold cross-validation (if variance_method='kfold_cv'). Default is 5."
    )]
    pub cv_folds: Option<usize>,

    /// Minimum weight threshold
    #[schemars(
        description = "Minimum weight to report in output (for sparsity). Default is 0.001."
    )]
    pub weight_threshold: Option<f64>,
}

/// Request for Sharp Regression Discontinuity estimation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RdEstimateRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable (Y) column.")]
    pub outcome: String,

    /// Running variable column name
    #[schemars(description = "Name of the running (forcing) variable (X) column.")]
    pub running_var: String,

    /// Cutoff value
    #[schemars(description = "Cutoff value for the running variable. Default is 0.")]
    pub cutoff: Option<f64>,

    /// Polynomial order for estimation
    #[schemars(
        description = "Polynomial order for local polynomial estimation. Default is 1 (local linear)."
    )]
    pub p: Option<usize>,

    /// Kernel type
    #[schemars(
        description = "Kernel function: 'triangular' (default), 'epanechnikov', or 'uniform'."
    )]
    pub kernel: Option<String>,

    /// Bandwidth selection method
    #[schemars(
        description = "Bandwidth selection: 'mserd' (MSE-optimal, default), 'msetwo' (separate left/right), 'cerrd', or 'certwo'."
    )]
    pub bwselect: Option<String>,

    /// Main bandwidth (overrides automatic selection)
    #[schemars(
        description = "Main bandwidth h for estimation. If not specified, uses automatic MSE-optimal selection."
    )]
    pub h: Option<f64>,

    /// Bias bandwidth (overrides automatic selection)
    #[schemars(description = "Bias bandwidth b. Default is rho * h where rho = 1.")]
    pub b: Option<f64>,

    /// Confidence level
    #[schemars(description = "Confidence level for intervals. Default is 0.95.")]
    pub level: Option<f64>,
}

/// Request for RD bandwidth selection only.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RdBandwidthRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable (Y) column.")]
    pub outcome: String,

    /// Running variable column name
    #[schemars(description = "Name of the running (forcing) variable (X) column.")]
    pub running_var: String,

    /// Cutoff value
    #[schemars(description = "Cutoff value for the running variable. Default is 0.")]
    pub cutoff: Option<f64>,

    /// Polynomial order
    #[schemars(description = "Polynomial order for estimation. Default is 1.")]
    pub p: Option<usize>,

    /// Kernel type
    #[schemars(
        description = "Kernel function: 'triangular' (default), 'epanechnikov', or 'uniform'."
    )]
    pub kernel: Option<String>,

    /// Bandwidth selection method
    #[schemars(
        description = "Bandwidth selection: 'mserd' (MSE-optimal, default), 'msetwo', 'cerrd', or 'certwo'."
    )]
    pub bwselect: Option<String>,
}

/// Request for Fuzzy Regression Discontinuity estimation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FuzzyRdRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable (Y) column.")]
    pub outcome: String,

    /// Running variable column name
    #[schemars(description = "Name of the running (forcing) variable (X) column.")]
    pub running_var: String,

    /// Treatment indicator column name
    #[schemars(
        description = "Name of the treatment indicator column (actual treatment received, 0/1)."
    )]
    pub treatment: String,

    /// Cutoff value
    #[schemars(description = "Cutoff value for the running variable. Default is 0.")]
    pub cutoff: Option<f64>,

    /// Polynomial order
    #[schemars(description = "Polynomial order for estimation. Default is 1.")]
    pub p: Option<usize>,

    /// Kernel type
    #[schemars(
        description = "Kernel function: 'triangular' (default), 'epanechnikov', or 'uniform'."
    )]
    pub kernel: Option<String>,

    /// Bandwidth selection method
    #[schemars(
        description = "Bandwidth selection: 'mserd' (default), 'msetwo', 'cerrd', or 'certwo'."
    )]
    pub bwselect: Option<String>,

    /// Main bandwidth
    #[schemars(description = "Main bandwidth h. If not specified, uses automatic selection.")]
    pub h: Option<f64>,

    /// Confidence level
    #[schemars(description = "Confidence level for intervals. Default is 0.95.")]
    pub level: Option<f64>,
}

/// Request for Multi-Cutoff Regression Discontinuity estimation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RdMultiRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable (Y) column.")]
    pub outcome: String,

    /// Running variable column name
    #[schemars(description = "Name of the running (forcing) variable (X) column.")]
    pub running_var: String,

    /// Cutoff values
    #[schemars(description = "List of cutoff values c1, c2, ..., cJ for the running variable.")]
    pub cutoffs: Vec<f64>,

    /// Cutoff assignment column (optional)
    #[schemars(
        description = "Column indicating which cutoff each observation belongs to (0, 1, ...). If not specified, observations are assigned to the nearest cutoff."
    )]
    pub cutoff_col: Option<String>,

    /// Whether to compute pooled estimate
    #[schemars(
        description = "Whether to compute a pooled treatment effect across all cutoffs. Default is true."
    )]
    pub pooled: Option<bool>,

    /// Pooling weight scheme
    #[schemars(
        description = "Weighting scheme for pooling: 'sample_size' (default), 'inverse_variance', or 'equal'."
    )]
    pub pooling_weights: Option<String>,

    /// Bandwidth specification
    #[schemars(
        description = "Bandwidth specification: single value for global bandwidth, or omit for per-cutoff optimal."
    )]
    pub bandwidth: Option<f64>,

    /// Per-cutoff bandwidths
    #[schemars(
        description = "List of bandwidths for each cutoff. Must match length of cutoffs if specified."
    )]
    pub bandwidths: Option<Vec<f64>>,

    /// Polynomial order for estimation
    #[schemars(
        description = "Polynomial order for local polynomial estimation. Default is 1 (local linear)."
    )]
    pub p: Option<usize>,

    /// Kernel type
    #[schemars(
        description = "Kernel function: 'triangular' (default), 'epanechnikov', or 'uniform'."
    )]
    pub kernel: Option<String>,

    /// Whether to test for heterogeneity
    #[schemars(
        description = "Whether to perform a chi-squared test for heterogeneous effects across cutoffs. Default is true."
    )]
    pub test_heterogeneity: Option<bool>,

    /// Confidence level
    #[schemars(description = "Confidence level for intervals. Default is 0.95.")]
    pub level: Option<f64>,
}

/// Request for Logit regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct LogitRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name (must be binary 0/1)
    #[schemars(description = "Name of the dependent variable (Y) column. Must be binary (0/1).")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,
}

/// Request for Probit regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ProbitRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name (must be binary 0/1)
    #[schemars(description = "Name of the dependent variable (Y) column. Must be binary (0/1).")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,
}

/// Request for multinomial logit regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MultinomRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name (categorical outcome)
    #[schemars(description = "Name of the categorical dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Reference category (optional)
    #[schemars(
        description = "Reference category for the model. If not specified, the first category (alphabetically) is used."
    )]
    pub reference: Option<String>,
}

/// Request for McFadden's conditional logit (mlogit) model.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MlogitRequest {
    /// Name/ID of the dataset (in long format)
    #[schemars(
        description = "Name or ID of a dataset in long format (one row per individual-alternative combination)."
    )]
    pub dataset: String,

    /// Column identifying choice situations (individuals)
    #[schemars(
        description = "Column name identifying each choice situation (individual chooser)."
    )]
    pub choice_id: String,

    /// Column identifying alternatives
    #[schemars(
        description = "Column name identifying alternatives (e.g., 'car', 'bus', 'train')."
    )]
    pub alt_id: String,

    /// Column with binary choice indicator (1 = chosen)
    #[schemars(
        description = "Column with binary choice indicator (1 if alternative is chosen, 0 otherwise)."
    )]
    pub choice: String,

    /// Alternative-specific variables (generic coefficients)
    #[schemars(
        description = "Alternative-specific variables that vary across alternatives (e.g., 'price', 'time'). These get generic coefficients (same β across all alternatives)."
    )]
    pub alt_specific: Vec<String>,

    /// Individual-specific variables (alternative-specific coefficients)
    #[schemars(
        description = "Individual-specific variables that are constant across alternatives (e.g., 'income', 'age'). These get alternative-specific coefficients (different γⱼ for each alternative vs reference)."
    )]
    #[serde(default)]
    pub ind_specific: Vec<String>,

    /// Reference alternative (optional)
    #[schemars(
        description = "Reference alternative for identification. Default: first alternative (alphabetically)."
    )]
    pub reference: Option<String>,
}

/// Request for mixed logit (random parameters logit) estimation.
/// Covers both gmnl and mixl R packages.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MixedLogitRequest {
    /// Name/ID of the dataset (in long format)
    #[schemars(
        description = "Name or ID of a dataset in long format (one row per individual-alternative combination)."
    )]
    pub dataset: String,

    /// Column identifying choice situations (individuals)
    #[schemars(
        description = "Column name identifying each choice situation (individual chooser)."
    )]
    pub choice_id: String,

    /// Column identifying alternatives
    #[schemars(
        description = "Column name identifying alternatives (e.g., 'car', 'bus', 'train')."
    )]
    pub alt_id: String,

    /// Column with binary choice indicator (1 = chosen)
    #[schemars(
        description = "Column with binary choice indicator (1 if alternative is chosen, 0 otherwise)."
    )]
    pub choice: String,

    /// Variables to include in the model
    #[schemars(description = "Variable columns to include in the choice model.")]
    pub variables: Vec<String>,

    /// Variables with random coefficients
    #[schemars(
        description = "Variable names that should have random (mixed) coefficients. If not specified, all variables are random."
    )]
    pub random_vars: Option<Vec<String>>,

    /// Distribution for random parameters
    #[schemars(
        description = "Distribution for random parameters: 'normal' (default), 'lognormal', 'triangular', 'uniform'."
    )]
    pub distribution: Option<String>,

    /// Number of simulation draws
    #[schemars(
        description = "Number of simulation draws for MSL estimation. Default: 500. Higher values improve accuracy but increase computation time."
    )]
    pub n_draws: Option<usize>,

    /// Use Halton sequences (quasi-random)
    #[schemars(
        description = "Use Halton quasi-random sequences instead of pseudo-random draws. Default: true. Improves accuracy."
    )]
    pub halton: Option<bool>,
}

/// Request for ordered logit/probit regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct OrderedRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name (ordered categorical outcome)
    #[schemars(description = "Name of the ordered categorical dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Model type: "logit" (default) or "probit"
    #[schemars(description = "Model type: 'logit' (default) or 'probit'.")]
    pub model_type: Option<String>,
}

/// Request for negative binomial regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct NegBinRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name (count data)
    #[schemars(description = "Name of the count dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Initial theta (dispersion) parameter
    #[schemars(
        description = "Optional initial theta (dispersion) parameter. If not specified, estimated from data."
    )]
    pub init_theta: Option<f64>,
}

/// Request for zero-inflated models.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ZeroInflRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name (count data with excess zeros)
    #[schemars(description = "Name of the count dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names for count model
    #[schemars(description = "Names of the independent variable (X) columns for the count model.")]
    pub x: Vec<String>,

    /// Independent variables (Z) column names for zero-inflation model
    #[schemars(
        description = "Names of the variables for the zero-inflation model. If not specified, uses intercept only."
    )]
    pub z: Option<Vec<String>>,

    /// Model type: "poisson" (default) or "negbin"
    #[schemars(description = "Distribution for count model: 'poisson' (default) or 'negbin'.")]
    pub dist: Option<String>,
}

/// Request for hurdle model (two-part model for count data with zero inflation).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct HurdleModelRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name (count data with zeros)
    #[schemars(description = "Name of the count dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names for count model
    #[schemars(description = "Names of the independent variable (X) columns for the count model.")]
    pub x: Vec<String>,

    /// Independent variables (Z) column names for binary (hurdle) model
    #[schemars(
        description = "Names of the variables for the binary hurdle model. If not specified, uses same as x."
    )]
    pub z: Option<Vec<String>>,

    /// Model type: "poisson" (default) or "negbin"
    #[schemars(description = "Distribution for count model: 'poisson' (default) or 'negbin'.")]
    pub dist: Option<String>,
}

/// Request for High-Dimensional Fixed Effects (HDFE) regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PanelHdfeRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Fixed effect columns to absorb
    #[schemars(
        description = "Column names for fixed effects to absorb (e.g., ['firm_id', 'year']). Supports multiple dimensions."
    )]
    pub fe: Vec<String>,

    /// Convergence tolerance for MAP algorithm
    #[schemars(
        description = "Convergence tolerance for the Method of Alternating Projections. Default is 1e-8."
    )]
    pub tolerance: Option<f64>,

    /// Maximum iterations for MAP algorithm
    #[schemars(description = "Maximum iterations for the demeaning algorithm. Default is 10000.")]
    pub max_iterations: Option<usize>,

    /// Standard error type
    #[schemars(
        description = "Standard error type: 'standard', 'hc0', 'hc1' (default), 'hc2', or 'hc3'."
    )]
    pub se_type: Option<String>,
}

/// Request for Generalized Linear Model with High-Dimensional Fixed Effects (FEGLM).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FeglmRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(
        description = "Name of the dependent variable (Y) column. For logit/probit must be binary (0/1). For Poisson must be non-negative counts."
    )]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Fixed effect columns to absorb
    #[schemars(
        description = "Column names for fixed effects to absorb (e.g., ['firm_id', 'year']). Supports multiple dimensions."
    )]
    pub fe: Vec<String>,

    /// GLM family
    #[schemars(
        description = "GLM family: 'logit' (binomial logit, default), 'probit' (binomial probit), 'poisson' (count data), or 'gaussian' (continuous, equivalent to linear HDFE)."
    )]
    pub family: Option<String>,

    /// Maximum IRLS iterations
    #[schemars(description = "Maximum IRLS iterations for estimation. Default is 25.")]
    pub max_iter: Option<usize>,

    /// Convergence tolerance
    #[schemars(description = "Convergence tolerance for coefficient changes. Default is 1e-8.")]
    pub tolerance: Option<f64>,
}

// ============================================================================
// Survival Analysis Tool Input Types
// ============================================================================

// ============================================================================
// Time Series Tool Input Types
// ============================================================================

/// Request for VAR (Vector Autoregression) model.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct VarRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to include in the VAR model
    #[schemars(
        description = "Names of the columns to include in the VAR model (e.g., ['gdp', 'inflation', 'interest_rate'])."
    )]
    pub columns: Vec<String>,

    /// Number of lags
    #[schemars(description = "Number of lags to include in the VAR model.")]
    pub lags: usize,
}

/// Request for Granger causality test.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GrangerRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (the "caused" variable)
    #[schemars(
        description = "Name of the dependent variable column - the variable being predicted (the 'caused' variable)."
    )]
    pub dependent: String,

    /// Potential causing variable
    #[schemars(
        description = "Name of the potential causing variable column - tests whether this variable helps predict the dependent variable."
    )]
    pub cause: String,

    /// Number of lags (optional, default: automatic selection)
    #[schemars(
        description = "Number of lags to include in the test. Default: automatic selection using Schwert's rule."
    )]
    pub lags: Option<usize>,
}

/// Request for VARMA (Vector ARMA) model.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct VarmaRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to include in the VARMA model
    #[schemars(description = "Names of the columns to include in the VARMA model.")]
    pub columns: Vec<String>,

    /// AR lags (p)
    #[schemars(description = "Number of autoregressive (AR) lags.")]
    pub p: usize,

    /// MA lags (q)
    #[schemars(description = "Number of moving average (MA) lags.")]
    pub q: usize,
}

/// Request for VECM (Vector Error Correction Model).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct VecmRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to include in the VECM model
    #[schemars(
        description = "Names of the columns to include in the VECM model. Should be I(1) cointegrated series."
    )]
    pub columns: Vec<String>,

    /// Number of lags
    #[schemars(description = "Number of lags for the VECM (must be at least 2).")]
    pub lags: usize,

    /// Cointegration rank
    #[schemars(
        description = "Cointegration rank (number of cointegrating relationships). Must be between 1 and k-1 where k is the number of variables."
    )]
    pub rank: usize,
}

/// Request for VAR Impulse Response Functions.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct VarIrfRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to include in the VAR model
    #[schemars(description = "Names of the columns to include in the VAR model.")]
    pub columns: Vec<String>,

    /// Number of lags
    #[schemars(description = "Number of lags for the VAR model.")]
    pub lags: usize,

    /// Number of IRF steps/periods
    #[schemars(description = "Number of periods to compute impulse responses for.")]
    pub steps: usize,
}

// ============================================================================
// Forecasting Tool Input Types
// ============================================================================

/// Request for ARIMA model fitting.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ArimaRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name with time series values
    #[schemars(description = "Name of the column containing the time series values.")]
    pub column: String,

    /// AR order (p)
    #[schemars(description = "Number of autoregressive (AR) terms.")]
    pub p: usize,

    /// Differencing order (d)
    #[schemars(description = "Number of differences to make the series stationary.")]
    pub d: usize,

    /// MA order (q)
    #[schemars(description = "Number of moving average (MA) terms.")]
    pub q: usize,
}

/// Request for ARIMA forecasting.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ArimaForecastRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name with time series values
    #[schemars(description = "Name of the column containing the time series values.")]
    pub column: String,

    /// AR order (p)
    #[schemars(description = "Number of autoregressive (AR) terms.")]
    pub p: usize,

    /// Differencing order (d)
    #[schemars(description = "Number of differences to make the series stationary.")]
    pub d: usize,

    /// MA order (q)
    #[schemars(description = "Number of moving average (MA) terms.")]
    pub q: usize,

    /// Forecast horizon
    #[schemars(description = "Number of periods to forecast ahead.")]
    pub horizon: usize,
}

/// Request for GARCH model fitting (volatility modeling).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GarchRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name with return series
    #[schemars(
        description = "Name of the column containing returns (e.g., stock returns, percentage changes)."
    )]
    pub column: String,

    /// ARCH order (p) - default 1
    #[schemars(description = "Number of ARCH (lagged squared residual) terms. Default: 1.")]
    pub p: Option<usize>,

    /// GARCH order (q) - default 1
    #[schemars(description = "Number of GARCH (lagged variance) terms. Default: 1.")]
    pub q: Option<usize>,

    /// Include mean in model - default true
    #[schemars(description = "Whether to include a mean term in the model. Default: true.")]
    pub include_mean: Option<bool>,
}

/// Request for MSTL decomposition.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MstlRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name with time series values
    #[schemars(description = "Name of the column containing the time series values.")]
    pub column: String,

    /// Seasonal periods
    #[schemars(
        description = "Seasonal periods to extract (e.g., [7, 365] for daily data with weekly and yearly seasonality)."
    )]
    pub periods: Vec<usize>,
}

/// Request for changepoint detection.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ChangepointRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name with time series values
    #[schemars(description = "Name of the column containing the time series values.")]
    pub column: String,

    /// Penalty for adding a changepoint (optional, uses BIC if not specified)
    #[schemars(
        description = "Penalty for adding a changepoint. Higher values = fewer changepoints. Default uses BIC (log(n))."
    )]
    pub penalty: Option<f64>,

    /// Minimum segment length between changepoints
    #[schemars(description = "Minimum number of observations between changepoints. Default is 2.")]
    pub min_segment_length: Option<usize>,

    /// Detection method: 'pelt' or 'binary'
    #[schemars(
        description = "Algorithm to use: 'pelt' (Pruned Exact Linear Time, default) or 'binary' (Binary Segmentation)."
    )]
    pub method: Option<String>,

    /// Type of change to detect: 'mean', 'variance', or 'both'
    #[schemars(description = "Type of change to detect: 'mean' (default), 'variance', or 'both'.")]
    pub change_type: Option<String>,
}

/// Request for Holt-Winters exponential smoothing.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct HoltWintersRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name with time series values
    #[schemars(description = "Name of the column containing the time series values.")]
    pub column: String,

    /// Seasonal period
    #[schemars(
        description = "Number of observations per seasonal cycle (e.g., 12 for monthly data with yearly seasonality, 4 for quarterly)."
    )]
    pub period: usize,

    /// Seasonal type
    #[schemars(
        description = "Type of seasonality: 'additive' (default) for constant seasonal variation, 'multiplicative' for proportional variation."
    )]
    pub seasonal: Option<String>,

    /// Level smoothing parameter alpha (0-1)
    #[schemars(
        description = "Smoothing parameter for the level component (0-1). If not specified, will be optimized."
    )]
    pub alpha: Option<f64>,

    /// Trend smoothing parameter beta (0-1)
    #[schemars(
        description = "Smoothing parameter for the trend component (0-1). If not specified, will be optimized."
    )]
    pub beta: Option<f64>,

    /// Seasonal smoothing parameter gamma (0-1)
    #[schemars(
        description = "Smoothing parameter for the seasonal component (0-1). If not specified, will be optimized."
    )]
    pub gamma: Option<f64>,

    /// Forecast horizon (optional)
    #[schemars(
        description = "Number of periods to forecast ahead. If specified, returns forecasts in addition to fitted values."
    )]
    pub horizon: Option<usize>,
}

/// Request for AR (autoregressive) model fitting.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ArModelRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name with time series values
    #[schemars(description = "Name of the column containing the time series values.")]
    pub column: String,

    /// Use AIC for order selection
    #[schemars(description = "Whether to use AIC for automatic order selection. Default: true.")]
    pub aic: Option<bool>,

    /// Maximum order to consider
    #[schemars(description = "Maximum AR order to consider. Default: min(n-1, 10*log10(n)).")]
    pub order_max: Option<usize>,

    /// Specific order to use
    #[schemars(
        description = "Specific AR order to fit. If provided with aic=false, uses this exact order."
    )]
    pub order: Option<usize>,

    /// Fitting method
    #[schemars(description = "Method for fitting: 'yule-walker' (default), 'burg', or 'ols'.")]
    pub method: Option<String>,

    /// Demean the series
    #[schemars(description = "Whether to remove the mean before fitting. Default: true.")]
    pub demean: Option<bool>,
}

/// Request for classical seasonal decomposition.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DecomposeRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name with time series values
    #[schemars(description = "Name of the column containing the time series values.")]
    pub column: String,

    /// Seasonal period
    #[schemars(
        description = "Number of observations per seasonal cycle (e.g., 12 for monthly data with yearly seasonality, 4 for quarterly)."
    )]
    pub period: usize,

    /// Decomposition type
    #[schemars(
        description = "Type of decomposition: 'additive' (default, Y = Trend + Seasonal + Random) or 'multiplicative' (Y = Trend × Seasonal × Random)."
    )]
    pub decompose_type: Option<String>,
}

/// Request for structural time series model fitting.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct StructTsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name with time series values
    #[schemars(description = "Name of the column containing the time series values.")]
    pub column: String,

    /// Model type
    #[schemars(
        description = "Type of structural model: 'level' (local level/random walk + noise), 'trend' (local linear trend), or 'bsm' (basic structural model with seasonality)."
    )]
    pub model_type: Option<String>,

    /// Seasonal period (for BSM only)
    #[schemars(
        description = "Seasonal period for BSM model (e.g., 12 for monthly data with yearly seasonality). Required for 'bsm' type."
    )]
    pub period: Option<usize>,
}

/// Request for CausalImpact analysis (Bayesian Structural Time Series for causal inference).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CausalImpactRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name with response variable (outcome)
    #[schemars(description = "Name of the column containing the response variable to analyze.")]
    pub response_col: String,

    /// Column name with time index
    #[schemars(
        description = "Name of the column containing the time index (must be integer-like: integers, dates as days since epoch, etc.)."
    )]
    pub time_col: String,

    /// Start of pre-intervention period (inclusive)
    #[schemars(
        description = "Start time value of the pre-intervention period (inclusive). This is the period used to train the model."
    )]
    pub pre_period_start: i64,

    /// End of pre-intervention period (inclusive)
    #[schemars(description = "End time value of the pre-intervention period (inclusive).")]
    pub pre_period_end: i64,

    /// Start of post-intervention period (inclusive)
    #[schemars(
        description = "Start time value of the post-intervention period (inclusive). This is when the intervention occurs."
    )]
    pub post_period_start: i64,

    /// End of post-intervention period (inclusive)
    #[schemars(description = "End time value of the post-intervention period (inclusive).")]
    pub post_period_end: i64,

    /// Optional control series columns
    #[schemars(
        description = "Optional names of columns to use as control time series. These should be correlated with the response but unaffected by the intervention."
    )]
    pub control_cols: Option<Vec<String>>,

    /// Significance level (default 0.05)
    #[schemars(
        description = "Significance level for credible intervals. Default: 0.05 (95% credible intervals)."
    )]
    pub alpha: Option<f64>,

    /// Include trend component
    #[schemars(description = "Whether to include a trend component in the model. Default: false.")]
    pub include_trend: Option<bool>,

    /// Seasonal period
    #[schemars(
        description = "If the data has seasonality, specify the period (e.g., 12 for monthly data with yearly seasonality)."
    )]
    pub seasonal_period: Option<usize>,
}

/// Request for log-linear model fitting.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct LoglinRequest {
    /// Name/ID of the dataset
    #[schemars(
        description = "Name or ID of a previously loaded dataset containing contingency table data."
    )]
    pub dataset: String,

    /// Count column
    #[schemars(description = "Name of the column containing the cell counts.")]
    pub count_column: String,

    /// Factor columns that define the contingency table dimensions
    #[schemars(
        description = "Names of the factor columns that define the contingency table dimensions."
    )]
    pub factor_columns: Vec<String>,

    /// Model margins to fit
    #[schemars(
        description = "Margins to fit in the model. Each margin is a list of factor indices (0-based). For example, [[0,1], [1,2]] fits the (0,1) and (1,2) two-way interactions. If not specified, fits an independence model (all main effects only)."
    )]
    pub margins: Option<Vec<Vec<usize>>>,

    /// Convergence tolerance
    #[schemars(description = "Convergence tolerance for IPF algorithm (default: 0.1).")]
    pub eps: Option<f64>,

    /// Maximum iterations
    #[schemars(description = "Maximum iterations for IPF algorithm (default: 20).")]
    pub max_iter: Option<usize>,
}

// ============================================================================
// Report Generation Tool Input Types
// ============================================================================

/// A section in the HTML report.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReportSectionInput {
    /// Section title
    #[schemars(description = "Title for this section of the report.")]
    pub title: String,

    /// Content items for the section
    #[schemars(description = "Content items to include in this section.")]
    pub content: Vec<ReportContentInput>,
}

/// Content item for a report section.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReportContentInput {
    /// Type of content: 'text', 'code', 'table', 'chart', or 'stats'
    #[schemars(
        description = "Type of content: 'text' (paragraph), 'code' (code block), 'table' (data table), 'chart' (base64 image), or 'stats' (key-value pairs)."
    )]
    pub content_type: String,

    /// Text content (for text and code types)
    #[schemars(description = "Text content for 'text' or 'code' types.")]
    pub text: Option<String>,

    /// Programming language (for code blocks)
    #[schemars(description = "Programming language for code block syntax highlighting.")]
    pub language: Option<String>,

    /// Table headers (for table type)
    #[schemars(description = "Column headers for table content.")]
    pub headers: Option<Vec<String>>,

    /// Table rows (for table type) - each row is a list of cell values
    #[schemars(description = "Table rows, where each row is a list of string values.")]
    pub rows: Option<Vec<Vec<String>>>,

    /// Table caption
    #[schemars(description = "Caption for the table.")]
    pub caption: Option<String>,

    /// Base64-encoded chart image (for chart type)
    #[schemars(description = "Base64-encoded PNG image data for chart content.")]
    pub image_base64: Option<String>,

    /// Chart title
    #[schemars(description = "Title for the chart.")]
    pub chart_title: Option<String>,

    /// Chart caption
    #[schemars(description = "Caption for the chart.")]
    pub chart_caption: Option<String>,

    /// Key-value statistics (for stats type)
    #[schemars(
        description = "Key-value pairs for statistics display. Format: [[key, value], ...]"
    )]
    pub stats: Option<Vec<Vec<String>>>,
}

/// Request to generate an HTML report.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GenerateReportRequest {
    /// Report title
    #[schemars(description = "Title for the report.")]
    pub title: String,

    /// Report subtitle (optional)
    #[schemars(description = "Optional subtitle or description for the report.")]
    pub subtitle: Option<String>,

    /// Author name (optional)
    #[schemars(description = "Optional author name.")]
    pub author: Option<String>,

    /// Report sections
    #[schemars(description = "Sections to include in the report.")]
    pub sections: Vec<ReportSectionInput>,
}

// ============================================================================
// Machine Learning Tool Input Types
// ============================================================================

/// Request for K-means clustering.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct KMeansRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to use as features for clustering
    #[schemars(description = "Names of the numeric columns to use as features for clustering.")]
    pub columns: Vec<String>,

    /// Number of clusters (k)
    #[schemars(description = "Number of clusters to create.")]
    pub k: usize,

    /// Maximum iterations (optional, default: 300)
    #[schemars(description = "Maximum number of iterations. Default is 300.")]
    pub max_iterations: Option<usize>,

    /// Number of initializations (optional, default: 10)
    #[schemars(description = "Number of random initializations to try. Default is 10.")]
    pub n_init: Option<usize>,

    /// Random seed for reproducibility (optional)
    #[schemars(description = "Optional random seed for reproducible results.")]
    pub seed: Option<u64>,
}

/// Request for DBSCAN clustering.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DBSCANRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to use as features for clustering
    #[schemars(description = "Names of the numeric columns to use as features for clustering.")]
    pub columns: Vec<String>,

    /// Epsilon (neighborhood radius)
    #[schemars(
        description = "Maximum distance between two samples for them to be considered in the same neighborhood."
    )]
    pub eps: f64,

    /// Minimum samples for core point
    #[schemars(
        description = "Minimum number of samples in a neighborhood for a point to be considered a core point."
    )]
    pub min_samples: usize,
}

/// Request for PCA (Principal Component Analysis).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PCARequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to use as features
    #[schemars(description = "Names of the numeric columns to include in PCA.")]
    pub columns: Vec<String>,

    /// Number of principal components to keep (optional)
    #[schemars(
        description = "Number of principal components to keep. If not specified, keeps all components."
    )]
    pub n_components: Option<usize>,
}

/// Request for Hierarchical clustering.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct HierarchicalRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to use as features for clustering
    #[schemars(description = "Names of the numeric columns to use as features for clustering.")]
    pub columns: Vec<String>,

    /// Number of clusters to cut the dendrogram into (optional)
    #[schemars(
        description = "Number of clusters to create. If not specified, uses distance_threshold."
    )]
    pub n_clusters: Option<usize>,

    /// Distance threshold for cutting the dendrogram (optional)
    #[schemars(
        description = "Distance threshold for cutting. Used if n_clusters is not specified."
    )]
    pub distance_threshold: Option<f64>,

    /// Linkage method
    #[schemars(
        description = "Linkage method: 'single', 'complete', 'average', or 'ward'. Default is 'ward'."
    )]
    pub linkage: Option<String>,
}

/// Request for t-SNE dimensionality reduction.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct TsneRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to use as features
    #[schemars(description = "Names of the numeric columns to include in t-SNE.")]
    pub columns: Vec<String>,

    /// Number of output dimensions (default: 2)
    #[schemars(description = "Number of output dimensions. Default is 2.")]
    pub n_components: Option<usize>,

    /// Perplexity parameter (default: 30.0)
    #[schemars(
        description = "Perplexity parameter, related to number of nearest neighbors. Default is 30."
    )]
    pub perplexity: Option<f64>,

    /// Maximum iterations (default: 1000)
    #[schemars(description = "Maximum number of iterations. Default is 1000.")]
    pub max_iterations: Option<usize>,

    /// Learning rate (default: 200.0)
    #[schemars(description = "Learning rate for optimization. Default is 200.")]
    pub learning_rate: Option<f64>,

    /// Random seed for reproducibility
    #[schemars(description = "Optional random seed for reproducible results.")]
    pub seed: Option<u64>,
}

/// Request for Classical Multidimensional Scaling (cmdscale).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CmdscaleRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to use for computing distances
    #[schemars(
        description = "Names of the numeric columns to include. Euclidean distances will be computed from these columns."
    )]
    pub columns: Vec<String>,

    /// Number of output dimensions (default: 2)
    #[schemars(description = "Number of dimensions in the output configuration. Default is 2.")]
    pub k: Option<usize>,

    /// Whether input is already a distance matrix
    #[schemars(
        description = "Set to true if the input columns represent a distance matrix. Default is false (data is converted to distances)."
    )]
    pub is_distance_matrix: Option<bool>,
}

/// Request for cutting a hierarchical clustering tree (cutree).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CutreeRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to use for hierarchical clustering
    #[schemars(description = "Names of the numeric columns to cluster.")]
    pub columns: Vec<String>,

    /// Number of clusters to form
    #[schemars(
        description = "Number of clusters to cut the tree into. Takes priority over cut_height."
    )]
    pub k: Option<usize>,

    /// Height at which to cut the tree
    #[schemars(
        description = "Height (distance threshold) at which to cut the dendrogram. Ignored if k is specified."
    )]
    pub cut_height: Option<f64>,

    /// Linkage method for hierarchical clustering
    #[schemars(
        description = "Linkage method: 'single', 'complete', 'average', or 'ward'. Default is 'ward'."
    )]
    pub linkage: Option<String>,
}

/// Request for Random Forest regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RandomForestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Feature column names
    #[schemars(description = "Names of the feature columns (X variables).")]
    pub features: Vec<String>,

    /// Target column name
    #[schemars(description = "Name of the target column (Y variable).")]
    pub target: String,

    /// Number of trees (default: 100)
    #[schemars(description = "Number of trees in the forest. Default is 100.")]
    pub n_trees: Option<usize>,

    /// Maximum tree depth (default: 10)
    #[schemars(description = "Maximum depth of each tree. Default is 10.")]
    pub max_depth: Option<usize>,

    /// Minimum samples to split (default: 2)
    #[schemars(description = "Minimum samples required to split a node. Default is 2.")]
    pub min_samples_split: Option<usize>,

    /// Max features per split
    #[schemars(
        description = "Max features to consider per split: 'sqrt', 'log2', 'all', or a number. Default is 'sqrt'."
    )]
    pub max_features: Option<String>,

    /// Random seed for reproducibility
    #[schemars(description = "Optional random seed for reproducible results.")]
    pub seed: Option<u64>,
}

/// Request for Linear SVM classification.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SvmRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Feature column names
    #[schemars(description = "Names of the feature columns (X variables).")]
    pub features: Vec<String>,

    /// Target column name
    #[schemars(
        description = "Name of the binary target column (Y variable). Must have exactly 2 unique values."
    )]
    pub target: String,

    /// Regularization parameter C (default: 1.0)
    #[schemars(
        description = "Regularization parameter C. Larger values = less regularization. Default is 1.0."
    )]
    pub c: Option<f64>,

    /// Maximum iterations (default: 1000)
    #[schemars(description = "Maximum number of iterations. Default is 1000.")]
    pub max_iterations: Option<usize>,

    /// Convergence tolerance (default: 1e-3)
    #[schemars(description = "Convergence tolerance. Default is 0.001.")]
    pub tolerance: Option<f64>,
}

/// Request for Causal Forest estimation (Wager & Athey 2018).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CausalForestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome column name
    #[schemars(description = "Name of the outcome variable (Y).")]
    pub outcome: String,

    /// Treatment column name
    #[schemars(description = "Name of the binary treatment variable (W). Must be 0/1.")]
    pub treatment: String,

    /// Covariate column names
    #[schemars(description = "Names of the covariate columns (X variables).")]
    pub covariates: Vec<String>,

    /// Number of trees (default: 2000)
    #[schemars(description = "Number of trees in the forest. Default is 2000.")]
    pub n_trees: Option<usize>,

    /// Minimum node size (default: 5)
    #[schemars(description = "Minimum number of observations in each leaf. Default is 5.")]
    pub min_node_size: Option<usize>,

    /// Maximum tree depth (default: 10)
    #[schemars(description = "Maximum depth of each tree. Default is 10.")]
    pub max_depth: Option<usize>,

    /// Use honest splitting (default: true)
    #[schemars(
        description = "Whether to use honest splitting (separate data for tree structure and estimation). Default is true."
    )]
    pub honesty: Option<bool>,

    /// Random seed for reproducibility
    #[schemars(description = "Optional random seed for reproducible results.")]
    pub seed: Option<u64>,
}

/// Request for BART-based Causal Inference (bartCause style).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct BartCausalRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome column name
    #[schemars(description = "Name of the outcome variable (Y).")]
    pub outcome: String,

    /// Treatment column name
    #[schemars(description = "Name of the binary treatment variable (W). Must be 0/1.")]
    pub treatment: String,

    /// Covariate column names
    #[schemars(description = "Names of the covariate columns (X variables).")]
    pub covariates: Vec<String>,

    /// Number of trees per response surface (default: 200)
    #[schemars(description = "Number of trees in each response surface ensemble. Default is 200.")]
    pub n_trees: Option<usize>,

    /// Maximum tree depth (default: 4)
    #[schemars(
        description = "Maximum depth of each tree. Default is 4 (BART uses shallow trees)."
    )]
    pub max_depth: Option<usize>,

    /// Number of bootstrap samples for uncertainty (default: 100)
    #[schemars(
        description = "Number of bootstrap samples for confidence intervals. Default is 100."
    )]
    pub n_bootstrap: Option<usize>,

    /// Include propensity score as covariate (default: false)
    #[schemars(
        description = "Whether to include estimated propensity score as a covariate. Default is false."
    )]
    pub include_propensity: Option<bool>,

    /// Random seed for reproducibility
    #[schemars(description = "Optional random seed for reproducible results.")]
    pub seed: Option<u64>,
}

/// Request for Treatment Effect Heterogeneity Test (hettx).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct HetTxRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome column name
    #[schemars(description = "Name of the outcome variable (Y).")]
    pub outcome: String,

    /// Treatment column name
    #[schemars(description = "Name of the binary treatment indicator (0/1).")]
    pub treatment: String,

    /// Covariate column names
    #[schemars(description = "Names of covariate columns for matching and decomposition.")]
    pub covariates: Vec<String>,

    /// Number of permutations (default: 1000)
    #[schemars(description = "Number of permutations for Fisherian inference. Default is 1000.")]
    pub n_permutations: Option<usize>,

    /// Test statistic type (default: 'variance')
    #[schemars(description = "Test statistic: 'variance' (default), 'range', 'iqr', or 'mad'.")]
    pub test_statistic: Option<String>,

    /// Whether to decompose heterogeneity (default: true)
    #[schemars(
        description = "Whether to decompose heterogeneity into systematic and idiosyncratic components. Default is true."
    )]
    pub decompose: Option<bool>,

    /// Effect estimation method (default: 'matching')
    #[schemars(
        description = "Method for estimating individual effects: 'matching' (default), 'regression', or 'stratified'."
    )]
    pub effect_method: Option<String>,

    /// Number of nearest neighbors for matching (default: 3)
    #[schemars(
        description = "Number of nearest neighbors for matching-based imputation. Default is 3."
    )]
    pub n_neighbors: Option<usize>,

    /// Random seed for reproducibility
    #[schemars(description = "Optional random seed for reproducible results.")]
    pub seed: Option<u64>,
}

/// Request for Projection Pursuit Regression (PPR).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PprRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Feature column names
    #[schemars(description = "Names of the feature columns (X variables).")]
    pub features: Vec<String>,

    /// Target column name
    #[schemars(description = "Name of the target column (Y variable).")]
    pub target: String,

    /// Number of terms (default: 1)
    #[schemars(description = "Number of terms in the PPR model. Default is 1.")]
    pub nterms: Option<usize>,

    /// Maximum terms to consider (default: nterms)
    #[schemars(
        description = "Maximum terms to consider during forward selection. Default is nterms."
    )]
    pub max_terms: Option<usize>,

    /// Smoothing method
    #[schemars(
        description = "Smoothing method for ridge functions: 'supsmu' (default), 'spline', or 'gcvspline'."
    )]
    pub sm_method: Option<String>,

    /// Bass parameter for supsmu (0-10)
    #[schemars(
        description = "Bass parameter for supsmu smoothing (0-10). Higher = smoother. Default is 0."
    )]
    pub bass: Option<f64>,
}

/// Request for SuperSmoother (supsmu).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SupsmuRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// X column name
    #[schemars(description = "Name of the predictor column (X variable).")]
    pub x: String,

    /// Y column name
    #[schemars(description = "Name of the response column (Y variable).")]
    pub y: String,

    /// Optional weight column
    #[schemars(description = "Optional name of the weight column.")]
    pub weights: Option<String>,

    /// Span parameter (0-1)
    #[schemars(
        description = "Fixed span fraction (0-1). If not specified, cross-validation selects optimal span."
    )]
    pub span: Option<f64>,

    /// Bass parameter for smoothness (0-10)
    #[schemars(
        description = "Bass parameter controlling smoothness (0-10). Higher = smoother. Default is 0."
    )]
    pub bass: Option<f64>,

    /// Periodic boundary conditions
    #[schemars(description = "Whether to treat x as periodic on [0, 1]. Default is false.")]
    pub periodic: Option<bool>,
}

/// Request for Tukey's resistant line (line).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct LineRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// X column name
    #[schemars(description = "Name of the predictor column (X variable).")]
    pub x: String,

    /// Y column name
    #[schemars(description = "Name of the response column (Y variable).")]
    pub y: String,

    /// Number of polishing iterations
    #[schemars(description = "Number of polishing iterations. Default is 1.")]
    pub iter: Option<usize>,
}

/// Request for cumulative periodogram (cpgram).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CpgramRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Time series column name
    #[schemars(description = "Name of the time series column.")]
    pub column: String,

    /// Taper proportion
    #[schemars(description = "Proportion of data to taper at each end (0.0-0.5). Default is 0.1.")]
    pub taper: Option<f64>,
}

/// Request for Toeplitz matrix construction.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ToeplitzRequest {
    /// First column (and row for symmetric matrix)
    #[schemars(
        description = "Values for the first column. For a symmetric matrix, this is also the first row."
    )]
    pub column: Vec<f64>,

    /// First row (optional, for asymmetric matrix)
    #[schemars(
        description = "Values for the first row. If not specified, creates a symmetric matrix using column values."
    )]
    pub row: Option<Vec<f64>>,
}

/// Request for model tables from ANOVA.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ModelTablesRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Response column
    #[schemars(description = "Name of the response column (Y variable).")]
    pub response: String,

    /// Factor column(s) for ANOVA
    #[schemars(
        description = "Name of the factor column for one-way ANOVA, or list of two factor columns for two-way ANOVA."
    )]
    pub factors: Vec<String>,

    /// Type of table
    #[schemars(
        description = "Type of table: 'means' (default) or 'effects' (deviations from grand mean)."
    )]
    pub table_type: Option<String>,

    /// Whether to compute standard errors
    #[schemars(description = "Whether to compute standard errors. Default is true.")]
    pub se: Option<bool>,
}

/// Request for standard errors of contrasts (se.contrast).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SeContrastRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Response column
    #[schemars(description = "Name of the response column (Y variable).")]
    pub response: String,

    /// Factor column for ANOVA
    #[schemars(description = "Name of the factor column for one-way ANOVA.")]
    pub factor: String,

    /// Contrast coefficients
    #[schemars(
        description = "Contrast coefficients. Each inner array is one contrast (must sum to 0)."
    )]
    pub contrasts: Option<Vec<Vec<f64>>>,

    /// Contrast type (if contrasts not provided)
    #[schemars(
        description = "Type of contrasts to generate: 'treatment', 'helmert', 'sum', or 'poly'. Only used if contrasts not provided."
    )]
    pub contrast_type: Option<String>,
}

/// Request for weighted mean.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct WeightedMeanRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column to compute weighted mean of
    #[schemars(description = "Name of the numeric column to compute weighted mean.")]
    pub column: String,

    /// Weight column
    #[schemars(description = "Name of the weight column.")]
    pub weights: String,

    /// Whether to remove NA values
    #[schemars(description = "Whether to remove NA values before computation. Default is true.")]
    pub na_rm: Option<bool>,
}

/// Request for weighted covariance matrix.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CovWtRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to compute covariance for
    #[schemars(description = "Names of the numeric columns to include in the covariance matrix.")]
    pub columns: Vec<String>,

    /// Weight column
    #[schemars(description = "Name of the weight column.")]
    pub weights: String,

    /// Whether to center the data
    #[schemars(
        description = "Whether to center the data (subtract weighted mean). Default is true."
    )]
    pub center: Option<bool>,

    /// Covariance method
    #[schemars(description = "Method for computing covariance: 'unbiased' (default) or 'ml'.")]
    pub method: Option<String>,
}

/// Request for Mauchly's sphericity test.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MauchlyTestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns for repeated measures
    #[schemars(
        description = "Names of columns representing repeated measures (at least 3 required)."
    )]
    pub columns: Vec<String>,
}

/// Request for stepwise model selection.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct StepRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Response variable
    #[schemars(description = "Name of the response column (Y variable).")]
    pub response: String,

    /// All candidate predictors
    #[schemars(description = "Names of all candidate predictor columns.")]
    pub predictors: Vec<String>,

    /// Direction of stepwise selection
    #[schemars(description = "Direction: 'both' (default), 'forward', or 'backward'.")]
    pub direction: Option<String>,

    /// Selection criterion
    #[schemars(description = "Selection criterion: 'aic' (default) or 'bic'.")]
    pub criterion: Option<String>,
}

/// Request for time series lag operation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct LagRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column to lag
    #[schemars(description = "Name of the time series column to lag.")]
    pub column: String,

    /// Number of lags
    #[schemars(
        description = "Number of positions to lag. Positive = shift back, negative = shift forward. Default is 1."
    )]
    pub k: Option<i32>,
}

/// Request for time series embedding.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct EmbedRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column to embed
    #[schemars(description = "Name of the time series column to embed.")]
    pub column: String,

    /// Embedding dimension
    #[schemars(description = "Number of columns in the embedding matrix (lag dimension).")]
    pub dimension: usize,
}

/// Request for inverse differencing.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DiffinvRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column containing differences
    #[schemars(description = "Name of the column containing differenced values.")]
    pub column: String,

    /// Initial values for integration
    #[schemars(description = "Initial values to start the cumulative sum.")]
    pub xi: Option<Vec<f64>>,

    /// Difference lag
    #[schemars(description = "Lag for differencing. Default is 1.")]
    pub lag: Option<usize>,

    /// Number of differences to invert
    #[schemars(description = "Number of times differencing was applied. Default is 1.")]
    pub differences: Option<usize>,
}

/// Request for linear filtering.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FilterRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column to filter
    #[schemars(description = "Name of the time series column to filter.")]
    pub column: String,

    /// Filter coefficients
    #[schemars(description = "Filter coefficients for convolution or recursive filtering.")]
    pub filter: Vec<f64>,

    /// Filter method
    #[schemars(description = "Method: 'convolution' (default) or 'recursive'.")]
    pub method: Option<String>,

    /// Sides for convolution
    #[schemars(description = "For convolution: 1 (past only) or 2 (centered, default).")]
    pub sides: Option<usize>,
}

/// Request for time series window extraction.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct WindowRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column to extract window from
    #[schemars(description = "Name of the time series column.")]
    pub column: String,

    /// Start index (0-based)
    #[schemars(
        description = "Start index (0-based, inclusive). If not specified, starts from beginning."
    )]
    pub start: Option<usize>,

    /// End index (0-based)
    #[schemars(description = "End index (0-based, exclusive). If not specified, goes to end.")]
    pub end: Option<usize>,
}

/// Request for theoretical ARMA ACF.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ArmaAcfRequest {
    /// AR coefficients
    #[schemars(description = "Autoregressive coefficients (phi). Empty for MA-only model.")]
    pub ar: Option<Vec<f64>>,

    /// MA coefficients
    #[schemars(description = "Moving average coefficients (theta). Empty for AR-only model.")]
    pub ma: Option<Vec<f64>>,

    /// Maximum lag
    #[schemars(description = "Maximum lag for ACF computation. Default is 10.")]
    pub lag_max: Option<usize>,

    /// Whether to compute PACF
    #[schemars(description = "If true, compute partial ACF instead of ACF. Default is false.")]
    pub pacf: Option<bool>,
}

/// Request for ARMA to MA conversion.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ArmaToMaRequest {
    /// AR coefficients
    #[schemars(description = "Autoregressive coefficients (phi).")]
    pub ar: Option<Vec<f64>>,

    /// MA coefficients
    #[schemars(description = "Moving average coefficients (theta).")]
    pub ma: Option<Vec<f64>>,

    /// Number of MA coefficients to compute
    #[schemars(description = "Number of MA (psi) weights to compute. Default is 10.")]
    pub lag_max: Option<usize>,
}

/// Request for ACF to AR conversion.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct Acf2ArRequest {
    /// ACF values
    #[schemars(description = "Autocorrelation function values starting at lag 1.")]
    pub acf: Vec<f64>,
}

/// Request for ARIMA simulation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ArimaSimRequest {
    /// Number of observations to simulate
    #[schemars(description = "Number of observations to simulate.")]
    pub n: usize,

    /// AR coefficients
    #[schemars(description = "Autoregressive coefficients.")]
    pub ar: Option<Vec<f64>>,

    /// MA coefficients
    #[schemars(description = "Moving average coefficients.")]
    pub ma: Option<Vec<f64>>,

    /// Differencing order
    #[schemars(description = "Order of differencing. Default is 0.")]
    pub d: Option<usize>,

    /// Innovation standard deviation
    #[schemars(description = "Standard deviation of innovations. Default is 1.0.")]
    pub sd: Option<f64>,

    /// Random seed
    #[schemars(description = "Random seed for reproducibility.")]
    pub seed: Option<u64>,
}

/// Request for running median smoothing.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RunmedRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column to smooth
    #[schemars(description = "Name of the column to apply running median.")]
    pub column: String,

    /// Window width
    #[schemars(description = "Width of the median window (must be odd).")]
    pub k: usize,

    /// End rule
    #[schemars(description = "How to handle ends: 'keep' (default), 'constant', or 'median'.")]
    pub endrule: Option<String>,
}

// ============================================================================
// Database Tool Input Types
// ============================================================================

/// Request to query a SQLite database.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SqliteQueryRequest {
    /// Path to the SQLite database file
    #[schemars(description = "Path to the SQLite database file (.db, .sqlite, .sqlite3).")]
    pub db_path: String,

    /// SQL query to execute
    #[schemars(description = "SQL query to execute (SELECT statements only recommended).")]
    pub query: String,

    /// Optional name for the resulting dataset
    #[schemars(
        description = "Optional name for the resulting dataset. If not provided, a default name will be generated."
    )]
    pub name: Option<String>,
}

/// Request to list tables in a SQLite database.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SqliteListTablesRequest {
    /// Path to the SQLite database file
    #[schemars(description = "Path to the SQLite database file.")]
    pub db_path: String,
}

/// Request to get schema for a SQLite table.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SqliteSchemaRequest {
    /// Path to the SQLite database file
    #[schemars(description = "Path to the SQLite database file.")]
    pub db_path: String,

    /// Table name
    #[schemars(description = "Name of the table to get schema for.")]
    pub table_name: String,
}

/// Request to query a DuckDB database.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DuckDBQueryRequest {
    /// Path to the DuckDB database file
    #[schemars(
        description = "Path to the DuckDB database file (.duckdb, .db). Use ':memory:' for in-memory database."
    )]
    pub db_path: String,

    /// SQL query to execute
    #[schemars(description = "SQL query to execute. DuckDB supports advanced analytics SQL.")]
    pub query: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the resulting dataset.")]
    pub name: Option<String>,
}

/// Request to list tables in a DuckDB database.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DuckDBListTablesRequest {
    /// Path to the DuckDB database file
    #[schemars(description = "Path to the DuckDB database file.")]
    pub db_path: String,
}

/// Request to get schema for a DuckDB table.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DuckDBSchemaRequest {
    /// Path to the DuckDB database file
    #[schemars(description = "Path to the DuckDB database file.")]
    pub db_path: String,

    /// Table name
    #[schemars(description = "Name of the table to get schema for.")]
    pub table_name: String,
}

/// Request to query a file (Parquet, CSV) using DuckDB SQL.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DuckDBFileQueryRequest {
    /// Path to the data file (Parquet or CSV)
    #[schemars(
        description = "Path to the data file (.parquet, .csv). DuckDB can query these files directly with SQL."
    )]
    pub file_path: String,

    /// SQL query to execute
    #[schemars(
        description = "SQL query to execute. Use {file} as placeholder for the file path. Example: 'SELECT * FROM {file} WHERE amount > 100'"
    )]
    pub query: String,

    /// Optional name for the resulting dataset
    #[schemars(
        description = "Optional name for the resulting dataset. If not provided, one will be generated."
    )]
    pub name: Option<String>,
}

// ============================================================================
// Visualization Request Types
// ============================================================================

/// Request to generate a histogram.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct HistogramRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name to plot
    #[schemars(description = "Name of the numeric column to create histogram from.")]
    pub column: String,

    /// Number of bins (optional, auto-calculated if not specified)
    #[schemars(
        description = "Number of bins for the histogram. If not specified, uses Sturges' rule."
    )]
    pub bins: Option<usize>,

    /// Chart title (optional)
    #[schemars(description = "Optional title for the chart.")]
    pub title: Option<String>,
}

/// Request to generate a scatter plot.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ScatterPlotRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// X-axis column name
    #[schemars(description = "Name of the column for X-axis values.")]
    pub x_column: String,

    /// Y-axis column name
    #[schemars(description = "Name of the column for Y-axis values.")]
    pub y_column: String,

    /// Chart title (optional)
    #[schemars(description = "Optional title for the chart.")]
    pub title: Option<String>,
}

/// Request to generate a line chart.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct LineChartRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// X-axis column name
    #[schemars(description = "Name of the column for X-axis values (e.g., time index).")]
    pub x_column: String,

    /// Y-axis column names (one or more series)
    #[schemars(
        description = "Names of the columns to plot as lines (can be multiple for multi-series)."
    )]
    pub y_columns: Vec<String>,

    /// Chart title (optional)
    #[schemars(description = "Optional title for the chart.")]
    pub title: Option<String>,
}

/// Request to generate a box plot.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct BoxPlotRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column names to include in box plot
    #[schemars(description = "Names of numeric columns to create box plots for.")]
    pub columns: Vec<String>,

    /// Chart title (optional)
    #[schemars(description = "Optional title for the chart.")]
    pub title: Option<String>,
}

/// Request to generate a correlation heatmap.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct HeatmapRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column names to include (optional, uses all numeric if not specified)
    #[schemars(
        description = "Names of numeric columns to include. If not specified, uses all numeric columns."
    )]
    pub columns: Option<Vec<String>>,

    /// Chart title (optional)
    #[schemars(description = "Optional title for the heatmap.")]
    pub title: Option<String>,
}

/// Request to generate an interactive scatter plot (HTML/Plotly output).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ScatterInteractiveRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// X-axis column name
    #[schemars(description = "Name of the column for X-axis values.")]
    pub x_column: String,

    /// Y-axis column name
    #[schemars(description = "Name of the column for Y-axis values.")]
    pub y_column: String,

    /// Group column for separate traces (optional)
    #[schemars(
        description = "Optional column for grouping data points into separate traces with different colors."
    )]
    pub group_column: Option<String>,

    /// Chart title (optional)
    #[schemars(description = "Optional title for the chart.")]
    pub title: Option<String>,
}

/// Request to generate an interactive histogram (HTML/Plotly output).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct HistogramInteractiveRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name to plot
    #[schemars(description = "Name of the numeric column to create histogram from.")]
    pub column: String,

    /// Group column for separate traces (optional)
    #[schemars(
        description = "Optional column for grouping data into separate overlaid histograms."
    )]
    pub group_column: Option<String>,

    /// Chart title (optional)
    #[schemars(description = "Optional title for the chart.")]
    pub title: Option<String>,
}

/// Request to generate an interactive line chart (HTML/Plotly output).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct LineInteractiveRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// X-axis column name
    #[schemars(description = "Name of the column for X-axis values (e.g., time index).")]
    pub x_column: String,

    /// Y-axis column name
    #[schemars(description = "Name of the column for Y-axis values.")]
    pub y_column: String,

    /// Chart title (optional)
    #[schemars(description = "Optional title for the chart.")]
    pub title: Option<String>,
}

/// Request to generate an event study plot.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct EventStudyRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name for time/period relative to treatment
    #[schemars(
        description = "Column with time periods relative to treatment (e.g., -3, -2, -1, 0, 1, 2, 3)."
    )]
    pub time_column: String,

    /// Column name for point estimates
    #[schemars(description = "Column with coefficient estimates at each time period.")]
    pub estimate_column: String,

    /// Column name for lower confidence interval bound
    #[schemars(description = "Column with lower bound of confidence interval.")]
    pub ci_lower_column: String,

    /// Column name for upper confidence interval bound
    #[schemars(description = "Column with upper bound of confidence interval.")]
    pub ci_upper_column: String,

    /// Chart title (optional)
    #[schemars(description = "Optional title for the chart.")]
    pub title: Option<String>,
}

/// Request to generate a coefficient plot.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CoefficientPlotRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name for variable/coefficient names
    #[schemars(description = "Column with variable names or coefficient labels.")]
    pub name_column: String,

    /// Column name for coefficient estimates
    #[schemars(description = "Column with coefficient point estimates.")]
    pub estimate_column: String,

    /// Column name for lower confidence interval bound
    #[schemars(description = "Column with lower bound of confidence interval.")]
    pub ci_lower_column: String,

    /// Column name for upper confidence interval bound
    #[schemars(description = "Column with upper bound of confidence interval.")]
    pub ci_upper_column: String,

    /// Chart title (optional)
    #[schemars(description = "Optional title for the chart.")]
    pub title: Option<String>,

    /// Horizontal orientation (optional, default: true)
    #[schemars(
        description = "If true, draw horizontal error bars (default). If false, draw vertical."
    )]
    pub horizontal: Option<bool>,
}

/// Request to generate an IRF (Impulse Response Function) plot.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct IrfPlotRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name for time horizon
    #[schemars(description = "Column with time horizon (0, 1, 2, ...).")]
    pub horizon_column: String,

    /// Column name for response values
    #[schemars(description = "Column with impulse response values.")]
    pub response_column: String,

    /// Column name for lower confidence interval bound (optional)
    #[schemars(description = "Optional column with lower bound of confidence interval.")]
    pub ci_lower_column: Option<String>,

    /// Column name for upper confidence interval bound (optional)
    #[schemars(description = "Optional column with upper bound of confidence interval.")]
    pub ci_upper_column: Option<String>,

    /// Label for the shock (optional)
    #[schemars(description = "Optional label for the shock variable.")]
    pub shock_label: Option<String>,

    /// Label for the response (optional)
    #[schemars(description = "Optional label for the response variable.")]
    pub response_label: Option<String>,

    /// Chart title (optional)
    #[schemars(description = "Optional title for the chart.")]
    pub title: Option<String>,
}

/// Request to generate residual diagnostic plots.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ResidualDiagnosticsRequest {
    /// Name/ID of the dataset
    #[schemars(
        description = "Name or ID of a previously loaded dataset containing regression results."
    )]
    pub dataset: String,

    /// Column name for fitted/predicted values
    #[schemars(description = "Column with fitted (predicted) values from regression.")]
    pub fitted_column: String,

    /// Column name for residual values
    #[schemars(description = "Column with residual values (observed - fitted).")]
    pub residuals_column: String,

    /// Column name for leverage (hat) values (optional)
    #[schemars(
        description = "Optional column with leverage (hat) values. If not provided, will be estimated."
    )]
    pub leverage_column: Option<String>,
}

/// Request to batch process multiple datasets with the same operation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct BatchProcessRequest {
    /// Names/IDs of datasets to process
    #[schemars(description = "List of dataset names to process. Each must be previously loaded.")]
    pub datasets: Vec<String>,

    /// Operation to perform on each dataset
    #[schemars(
        description = "Operation to perform: 'describe' (summary stats), 'correlation' (correlation matrix), or 'ols' (regression)."
    )]
    pub operation: String,

    /// Columns to analyze (optional, defaults to all numeric for describe/correlation)
    #[schemars(
        description = "List of column names to analyze. For 'ols', first column is dependent variable."
    )]
    pub columns: Option<Vec<String>>,

    /// Whether to return combined summary across all datasets
    #[schemars(description = "If true, also returns an aggregated summary across all datasets.")]
    pub combine_results: Option<bool>,
}

/// Request to compare the same columns across multiple datasets.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CompareDatasetRequest {
    /// Names/IDs of datasets to compare
    #[schemars(description = "List of dataset names to compare. Each must be previously loaded.")]
    pub datasets: Vec<String>,

    /// Columns to compare
    #[schemars(description = "List of column names to compare across datasets.")]
    pub columns: Vec<String>,

    /// Type of comparison
    #[schemars(
        description = "Comparison type: 'summary' (side-by-side stats), 'correlation' (correlation differences), or 'distribution' (distribution comparison)."
    )]
    pub comparison_type: Option<String>,
}

/// Request to export the current analysis session.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExportSessionRequest {
    /// Path to save the session file
    #[schemars(
        description = "File path to save the session (JSON format). If not provided, returns session data as string."
    )]
    pub file_path: Option<String>,

    /// Whether to include dataset data (default: true)
    #[schemars(
        description = "Include full dataset data. If false, only metadata and file paths are saved."
    )]
    pub include_data: Option<bool>,
}

/// Request to import a previously exported session.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ImportSessionRequest {
    /// Path to the session file to import
    #[schemars(description = "File path to the session JSON file to import.")]
    pub file_path: String,

    /// Whether to merge with existing session (default: false, replaces)
    #[schemars(description = "If true, merges with existing datasets instead of replacing.")]
    pub merge: Option<bool>,
}

/// Request to set the global random seed for ML reproducibility.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SetSeedRequest {
    /// The random seed value
    #[schemars(description = "The random seed value. Set to null/omit to clear the global seed.")]
    pub seed: Option<u64>,
}

/// Request to get the current global seed.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetSeedRequest {}

/// Column specification for random data generation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ColumnSpecInput {
    /// Name of the column
    #[schemars(description = "Name of the column to generate.")]
    pub name: String,

    /// Distribution type and parameters
    #[schemars(
        description = "Distribution specification. Must include 'type' field and distribution-specific parameters."
    )]
    pub distribution: serde_json::Value,
}

/// Request to generate random data.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GenerateRandomDataRequest {
    /// Number of rows to generate
    #[schemars(description = "Number of rows to generate.")]
    pub n_rows: usize,

    /// Column specifications
    #[schemars(
        description = "Array of column specifications. Each must have 'name' and 'distribution' fields. Distribution types: 'uniform' (min, max), 'normal' (mean, std), 'binomial' (n, p), 'poisson' (lambda), 'exponential' (rate), 'bernoulli' (p), 'categorical' (categories, optional weights), 'uniform_int' (min, max), 'sequence' (start), 'constant' (value), 'constant_string' (value)."
    )]
    pub columns: Vec<ColumnSpecInput>,

    /// Random seed for reproducibility
    #[schemars(description = "Optional random seed for reproducible results.")]
    pub seed: Option<u64>,

    /// Name to assign to the generated dataset
    #[schemars(description = "Name to assign to the generated dataset. Defaults to 'generated'.")]
    pub name: Option<String>,
}

/// Request to visualize hierarchical clustering results as a dendrogram.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DendrogramRequest {
    /// Linkage matrix from hierarchical clustering (JSON array of arrays)
    #[schemars(
        description = "Linkage matrix from hierarchical clustering. Array of [cluster1, cluster2, distance, size] tuples."
    )]
    pub linkage_matrix: Vec<Vec<f64>>,

    /// Optional labels for leaf nodes
    #[schemars(
        description = "Optional labels for leaf nodes (original samples). If not provided, uses indices."
    )]
    pub labels: Option<Vec<String>>,

    /// Chart width
    #[schemars(description = "Width of the chart in pixels (default: 800).")]
    pub width: Option<u32>,

    /// Chart height
    #[schemars(description = "Height of the chart in pixels (default: 600).")]
    pub height: Option<u32>,

    /// Chart title
    #[schemars(description = "Title for the dendrogram (default: 'Dendrogram').")]
    pub title: Option<String>,
}

// ============================================================================
// Data Munging Tool Input Types
// ============================================================================

/// Request to filter rows in a dataset.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FilterDatasetRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset to filter.")]
    pub dataset: String,

    /// Column to filter on
    #[schemars(description = "Name of the column to filter on.")]
    pub column: String,

    /// Comparison operator
    #[schemars(
        description = "Comparison operator: 'eq', 'ne', 'gt', 'ge', 'lt', 'le', 'contains', 'starts_with', 'ends_with'."
    )]
    pub op: String,

    /// Value to compare against
    #[schemars(
        description = "Value to compare against (as string, will be parsed based on column type)."
    )]
    pub value: String,

    /// Optional name for the resulting dataset
    #[schemars(
        description = "Optional name for the filtered result. If not provided, overwrites the source dataset."
    )]
    pub result_name: Option<String>,
}

/// Request to select columns from a dataset.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SelectColumnsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Columns to select
    #[schemars(description = "List of column names to keep.")]
    pub columns: Vec<String>,

    /// Optional name for the resulting dataset
    #[schemars(
        description = "Optional name for the result. If not provided, overwrites the source dataset."
    )]
    pub result_name: Option<String>,
}

/// Request to drop columns from a dataset.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DropColumnsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Columns to drop
    #[schemars(description = "List of column names to drop.")]
    pub columns: Vec<String>,

    /// Optional name for the resulting dataset
    #[schemars(
        description = "Optional name for the result. If not provided, overwrites the source dataset."
    )]
    pub result_name: Option<String>,
}

/// Request to rename columns in a dataset.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RenameColumnsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Mapping of old names to new names
    #[schemars(
        description = "Mapping of old column names to new names as pairs: [[\"old1\", \"new1\"], [\"old2\", \"new2\"]]."
    )]
    pub renames: Vec<Vec<String>>,

    /// Optional name for the resulting dataset
    #[schemars(
        description = "Optional name for the result. If not provided, overwrites the source dataset."
    )]
    pub result_name: Option<String>,
}

/// Request to sort a dataset.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SortDatasetRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Columns to sort by
    #[schemars(description = "List of column names to sort by.")]
    pub by: Vec<String>,

    /// Sort in descending order
    #[schemars(description = "If true, sort in descending order. Default is ascending.")]
    pub descending: Option<bool>,

    /// Optional name for the resulting dataset
    #[schemars(
        description = "Optional name for the result. If not provided, overwrites the source dataset."
    )]
    pub result_name: Option<String>,
}

/// Request to join two datasets.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct JoinDatasetsRequest {
    /// Name/ID of the left dataset
    #[schemars(description = "Name or ID of the left dataset.")]
    pub left: String,

    /// Name/ID of the right dataset
    #[schemars(description = "Name or ID of the right dataset.")]
    pub right: String,

    /// Columns to join on (from left dataset)
    #[schemars(description = "Column names from the left dataset to join on.")]
    pub left_on: Vec<String>,

    /// Columns to join on (from right dataset)
    #[schemars(
        description = "Column names from the right dataset to join on. If not provided, uses left_on."
    )]
    pub right_on: Option<Vec<String>>,

    /// Type of join
    #[schemars(description = "Join type: 'left', 'right', 'inner', or 'full'. Default is 'left'.")]
    pub join_type: Option<String>,

    /// Suffix for duplicate column names
    #[schemars(
        description = "Suffix to add to duplicate column names from the right dataset. Default is '_right'."
    )]
    pub suffix: Option<String>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the joined result.")]
    pub result_name: Option<String>,
}

/// Request to concatenate datasets.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ConcatDatasetsRequest {
    /// Names/IDs of datasets to concatenate
    #[schemars(description = "List of dataset names to concatenate vertically (row-bind).")]
    pub datasets: Vec<String>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the concatenated result.")]
    pub result_name: Option<String>,
}

/// Request to group and aggregate a dataset.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GroupByRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Columns to group by
    #[schemars(description = "Column names to group by.")]
    pub by: Vec<String>,

    /// Aggregation specifications
    #[schemars(
        description = "Aggregation specs as [[\"column\", \"function\"], ...]. Functions: 'count', 'sum', 'mean', 'median', 'min', 'max', 'std', 'var', 'first', 'last'."
    )]
    pub aggs: Vec<Vec<String>>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the grouped result.")]
    pub result_name: Option<String>,
}

/// Request to compute value counts for a column.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ValueCountsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to count values in
    #[schemars(description = "Column name to compute value counts for.")]
    pub column: String,

    /// Whether to normalize to percentages
    #[schemars(description = "If true, return percentages instead of counts.")]
    pub normalize: Option<bool>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to pivot a dataset from long to wide format.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PivotDatasetRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Index columns (will remain as rows)
    #[schemars(description = "Column names to use as index (will remain as rows).")]
    pub index: Vec<String>,

    /// Column whose values become new column names
    #[schemars(description = "Column whose values become new column names.")]
    pub on: String,

    /// Column containing values to fill the new columns
    #[schemars(description = "Column containing values to fill the new columns.")]
    pub values: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the pivoted result.")]
    pub result_name: Option<String>,
}

/// Request to melt a dataset from wide to long format.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MeltDatasetRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// ID columns to keep as-is
    #[schemars(description = "Column names to keep as identifier variables.")]
    pub id_vars: Vec<String>,

    /// Value columns to unpivot
    #[schemars(description = "Column names to unpivot into rows.")]
    pub value_vars: Vec<String>,

    /// Name for the variable column
    #[schemars(description = "Name for the new variable column. Default is 'variable'.")]
    pub variable_name: Option<String>,

    /// Name for the value column
    #[schemars(description = "Name for the new value column. Default is 'value'.")]
    pub value_name: Option<String>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the melted result.")]
    pub result_name: Option<String>,
}

/// Request to drop rows with null values.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DropNaRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Columns to check for nulls
    #[schemars(
        description = "Column names to check for nulls. If not provided, checks all columns."
    )]
    pub columns: Option<Vec<String>>,

    /// How to drop rows
    #[schemars(
        description = "How to drop: 'any' (drop if any null) or 'all' (drop only if all null). Default is 'any'."
    )]
    pub how: Option<String>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to fill null values.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FillNaRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Columns to fill nulls in
    #[schemars(description = "Column names to fill nulls in. If not provided, fills all columns.")]
    pub columns: Option<Vec<String>>,

    /// Fill strategy
    #[schemars(
        description = "Fill strategy: 'mean', 'median', 'mode', 'forward', 'backward', or a constant value."
    )]
    pub strategy: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to remove duplicate rows.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DeduplicateRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Columns to check for duplicates
    #[schemars(
        description = "Column names to check for duplicates. If not provided, checks all columns."
    )]
    pub columns: Option<Vec<String>>,

    /// Which duplicate to keep
    #[schemars(
        description = "Which duplicate to keep: 'first', 'last', or 'none'. Default is 'first'."
    )]
    pub keep: Option<String>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

// =============================================================================
// STRING CLEANING REQUESTS
// =============================================================================

/// Request to trim whitespace from string columns.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct TrimRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Columns to trim
    #[schemars(description = "Column names to trim. If not provided, trims all string columns.")]
    pub columns: Option<Vec<String>>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to convert a string column to lowercase.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ToLowercaseRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to convert
    #[schemars(description = "Name of the string column to convert to lowercase.")]
    pub column: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to convert a string column to uppercase.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ToUppercaseRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to convert
    #[schemars(description = "Name of the string column to convert to uppercase.")]
    pub column: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to replace exact values in a column.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReplaceValueRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to modify
    #[schemars(description = "Name of the column to modify.")]
    pub column: String,

    /// Value to find
    #[schemars(description = "Exact value to search for and replace.")]
    pub old_value: String,

    /// Replacement value
    #[schemars(description = "Value to replace matches with.")]
    pub new_value: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to replace substrings matching a regex pattern.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RegexReplaceRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to modify
    #[schemars(description = "Name of the string column to modify.")]
    pub column: String,

    /// Regex pattern
    #[schemars(description = "Regular expression pattern to match.")]
    pub pattern: String,

    /// Replacement string
    #[schemars(description = "Replacement string. Use $1, $2, etc. for capture groups.")]
    pub replacement: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to extract substrings matching a regex pattern.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RegexExtractRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to extract from
    #[schemars(description = "Name of the string column to extract from.")]
    pub column: String,

    /// Regex pattern with capture groups
    #[schemars(
        description = "Regular expression pattern. Use capture groups () to specify what to extract."
    )]
    pub pattern: String,

    /// Name for the new column
    #[schemars(description = "Name for the new column containing extracted values.")]
    pub new_column: String,

    /// Which capture group to extract
    #[schemars(
        description = "Which capture group to extract: 0 = entire match, 1 = first group, etc. Default is 1."
    )]
    pub group: Option<usize>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to count regex matches in each row.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RegexCountRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to search in
    #[schemars(description = "Name of the string column to search in.")]
    pub column: String,

    /// Regex pattern
    #[schemars(description = "Regular expression pattern to count matches for.")]
    pub pattern: String,

    /// Name for the new count column
    #[schemars(description = "Name for the new column containing match counts.")]
    pub new_column: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to split a string column into multiple columns.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct StrSplitRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to split
    #[schemars(description = "Name of the string column to split.")]
    pub column: String,

    /// Pattern to split on
    #[schemars(description = "Pattern to split on (regex supported). E.g., ',' or '\\s+'.")]
    pub pattern: String,

    /// Maximum number of splits
    #[schemars(
        description = "Maximum number of splits. If not provided, splits on all occurrences."
    )]
    pub max_splits: Option<usize>,

    /// Prefix for new column names
    #[schemars(
        description = "Prefix for new column names. Creates columns named prefix_0, prefix_1, etc."
    )]
    pub prefix: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to concatenate multiple string columns.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct StrConcatRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Columns to concatenate
    #[schemars(description = "Names of the string columns to concatenate.")]
    pub columns: Vec<String>,

    /// Name for the new column
    #[schemars(description = "Name for the new concatenated column.")]
    pub new_column: String,

    /// Separator between values
    #[schemars(description = "Separator to insert between values. Default is empty string.")]
    pub separator: Option<String>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to get string lengths.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct StrLengthRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to measure
    #[schemars(description = "Name of the string column to measure lengths for.")]
    pub column: String,

    /// Name for the new length column
    #[schemars(description = "Name for the new column containing string lengths.")]
    pub new_column: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to extract a substring.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct StrSubstringRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to extract from
    #[schemars(description = "Name of the string column.")]
    pub column: String,

    /// Start index
    #[schemars(description = "Start index (0-based). Negative values count from end.")]
    pub start: i64,

    /// Length to extract
    #[schemars(description = "Number of characters to extract. If not provided, extracts to end.")]
    pub length: Option<usize>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to create lag or lead columns.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct LagLeadRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to shift
    #[schemars(description = "Column name to create lag/lead for.")]
    pub column: String,

    /// Number of periods to shift
    #[schemars(
        description = "Number of periods to shift. Positive for lag, negative for lead (or use 'direction')."
    )]
    pub periods: i64,

    /// Direction: 'lag' or 'lead'
    #[schemars(
        description = "Direction: 'lag' (shift forward) or 'lead' (shift backward). Default is 'lag'."
    )]
    pub direction: Option<String>,

    /// Columns to group by (for panel data)
    #[schemars(description = "Optional group-by columns for panel data (e.g., ['firm_id']).")]
    pub group_by: Option<Vec<String>>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to standardize or normalize columns.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct StandardizeRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Columns to transform
    #[schemars(description = "Column names to standardize/normalize.")]
    pub columns: Vec<String>,

    /// Method: 'standardize' or 'normalize'
    #[schemars(
        description = "Method: 'standardize' (z-score) or 'normalize' (0-1 range). Default is 'standardize'."
    )]
    pub method: Option<String>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to bin a continuous variable.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct BinColumnRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to bin
    #[schemars(description = "Column name to bin.")]
    pub column: String,

    /// Binning strategy
    #[schemars(
        description = "Binning strategy: 'uniform' (equal width), 'quantile' (equal frequency), or 'custom'."
    )]
    pub strategy: String,

    /// Number of bins or custom breaks
    #[schemars(
        description = "Number of bins (for uniform/quantile) or list of break points (for custom)."
    )]
    pub bins: Vec<f64>,

    /// Optional labels for bins
    #[schemars(description = "Optional labels for the bins.")]
    pub labels: Option<Vec<String>>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to one-hot encode a categorical column.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct OneHotEncodeRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to encode
    #[schemars(description = "Categorical column name to one-hot encode.")]
    pub column: String,

    /// Whether to drop the first category
    #[schemars(
        description = "If true, drop first category to avoid multicollinearity. Default is false."
    )]
    pub drop_first: Option<bool>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to compute differences or percent changes.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DiffRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to compute differences for
    #[schemars(description = "Column name to compute differences for.")]
    pub column: String,

    /// Number of periods
    #[schemars(description = "Number of periods for difference. Default is 1.")]
    pub periods: Option<i64>,

    /// Type of difference
    #[schemars(
        description = "Type: 'diff' (absolute difference) or 'pct_change' (percent change). Default is 'diff'."
    )]
    pub diff_type: Option<String>,

    /// Columns to group by (for panel data)
    #[schemars(description = "Optional group-by columns for panel data.")]
    pub group_by: Option<Vec<String>>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to sample rows from a dataset.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SampleDatasetRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Number of rows to sample
    #[schemars(description = "Number of rows to sample.")]
    pub n: usize,

    /// Whether to sample with replacement
    #[schemars(description = "If true, sample with replacement. Default is false.")]
    pub replace: Option<bool>,

    /// Random seed for reproducibility
    #[schemars(description = "Random seed for reproducible sampling.")]
    pub seed: Option<u64>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to create a new column by computation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MutateColumnRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Name for the new column
    #[schemars(description = "Name for the new column.")]
    pub new_column: String,

    /// Expression type
    #[schemars(
        description = "Expression type: 'arithmetic' (e.g., col1 + col2), 'function' (e.g., log(col)), or 'constant'."
    )]
    pub expr_type: String,

    /// Left operand (column name for arithmetic)
    #[schemars(
        description = "Left operand: column name for arithmetic, column for function, or constant value."
    )]
    pub left: String,

    /// Operator (for arithmetic: '+', '-', '*', '/')
    #[schemars(
        description = "Operator for arithmetic: '+', '-', '*', '/'. For function: function name ('log', 'exp', 'sqrt', 'abs', 'square')."
    )]
    pub operator: Option<String>,

    /// Right operand (column name for arithmetic)
    #[schemars(description = "Right operand: column name for arithmetic expressions.")]
    pub right: Option<String>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

// ============================================================================
// Robust Statistics Request Structs
// ============================================================================

/// Request for Tukey's five-number summary.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FivenumRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column to summarize
    #[schemars(description = "Name of the numeric column to compute five-number summary.")]
    pub column: String,
}

/// Request for interquartile range.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct IqrRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column to compute IQR for
    #[schemars(description = "Name of the numeric column.")]
    pub column: String,

    /// Quantile type (1-9, default 7)
    #[schemars(description = "Quantile type (1-9). Type 7 (default) matches R's default.")]
    pub qtype: Option<usize>,
}

/// Request for median absolute deviation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MadRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column to compute MAD for
    #[schemars(description = "Name of the numeric column.")]
    pub column: String,

    /// Center (default: median)
    #[schemars(description = "Center to use. Default is 'median'. Can also use 'mean'.")]
    pub center: Option<String>,

    /// Scaling constant (default: 1.4826 for normal consistency)
    #[schemars(
        description = "Scaling constant. Default 1.4826 makes MAD consistent for normal data."
    )]
    pub constant: Option<f64>,
}

/// Request for empirical CDF.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct EcdfRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column to compute ECDF for
    #[schemars(description = "Name of the numeric column.")]
    pub column: String,

    /// Optional values at which to evaluate the ECDF
    #[schemars(
        description = "Optional specific values at which to evaluate the ECDF. If omitted, returns ECDF at all sorted unique data values."
    )]
    pub at: Option<Vec<f64>>,
}

/// Request for kernel density estimation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DensityRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column to estimate density for
    #[schemars(description = "Name of the numeric column.")]
    pub column: String,

    /// Kernel type
    #[schemars(
        description = "Kernel function: 'gaussian' (default), 'epanechnikov', 'rectangular', 'triangular', 'biweight', or 'cosine'."
    )]
    pub kernel: Option<String>,

    /// Bandwidth
    #[schemars(
        description = "Bandwidth for smoothing. If omitted, uses Silverman's rule of thumb."
    )]
    pub bw: Option<f64>,

    /// Number of evaluation points
    #[schemars(description = "Number of points to evaluate density at. Default is 512.")]
    pub n: Option<usize>,
}

// ============================================================================
// Spline/Interpolation Request Structs
// ============================================================================

/// Request for spline interpolation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SplineRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// X column (knot locations)
    #[schemars(description = "Name of the x (independent) column.")]
    pub x: String,

    /// Y column (values at knots)
    #[schemars(description = "Name of the y (dependent) column.")]
    pub y: String,

    /// Points at which to interpolate
    #[schemars(
        description = "Points at which to interpolate. If omitted, returns the spline coefficients."
    )]
    pub xout: Option<Vec<f64>>,

    /// Spline method
    #[schemars(
        description = "Spline method: 'fmm' (default, Forsythe-Malcolm-Moler), 'natural', 'periodic', or 'hyman' (monotone)."
    )]
    pub method: Option<String>,
}

/// Request for linear approximation/interpolation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ApproxRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// X column
    #[schemars(description = "Name of the x (independent) column.")]
    pub x: String,

    /// Y column
    #[schemars(description = "Name of the y (dependent) column.")]
    pub y: String,

    /// Points at which to interpolate
    #[schemars(description = "Points at which to interpolate.")]
    pub xout: Vec<f64>,

    /// Interpolation method
    #[schemars(description = "Interpolation method: 'linear' (default) or 'constant'.")]
    pub method: Option<String>,

    /// Rule for extrapolation
    #[schemars(
        description = "Rule for handling points outside range: 'na' (default, return NA) or 'nearest' (use boundary value)."
    )]
    pub rule: Option<String>,
}

// ============================================================================
// GLS and Smooth Spline Request Structs
// ============================================================================

/// Request for Generalized Least Squares regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GlsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Response variable column
    #[schemars(description = "Name of the dependent (y) variable column.")]
    pub y: String,

    /// Predictor variable columns
    #[schemars(description = "Names of the independent (x) variable columns.")]
    pub x: Vec<String>,

    /// Include intercept
    #[schemars(description = "Whether to include an intercept. Default is true.")]
    pub intercept: Option<bool>,

    /// Correlation structure type
    #[schemars(
        description = "Correlation structure: 'ar1' (default), 'compound_symmetry', or 'identity' (OLS)."
    )]
    pub correlation: Option<String>,

    /// Correlation parameter (rho)
    #[schemars(
        description = "Correlation parameter rho for AR(1) or compound symmetry. If omitted with 'ar1', auto-estimated from OLS residuals."
    )]
    pub rho: Option<f64>,
}

/// Request for smooth spline fitting.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SmoothSplineRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// X column
    #[schemars(description = "Name of the x (independent) column.")]
    pub x: String,

    /// Y column
    #[schemars(description = "Name of the y (dependent) column.")]
    pub y: String,

    /// Smoothing parameter (spar)
    #[schemars(
        description = "Smoothing parameter (0 to 1). If omitted, uses generalized cross-validation (GCV) to select automatically."
    )]
    pub spar: Option<f64>,

    /// Degrees of freedom
    #[schemars(
        description = "Equivalent degrees of freedom. Alternative to spar. If both omitted, uses GCV."
    )]
    pub df: Option<f64>,

    /// Points at which to predict
    #[schemars(
        description = "Optional points at which to evaluate the fitted spline. If omitted, returns fit at data points."
    )]
    pub xout: Option<Vec<f64>>,
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Format diagnostic warnings from an identification report for output.
fn format_diagnostic_warnings(report: &IdentificationReport) -> String {
    let warnings: Vec<_> = report
        .warnings
        .iter()
        .filter(|w| w.severity >= WarningSeverity::Caution)
        .collect();

    if warnings.is_empty() {
        return String::new();
    }

    let mut output = String::from("\n\n--- Identification Diagnostics ---\n");
    for w in warnings {
        let severity_str = match w.severity {
            WarningSeverity::Critical => "CRITICAL",
            WarningSeverity::Warning => "WARNING",
            WarningSeverity::Caution => "CAUTION",
            WarningSeverity::Info => "INFO",
        };
        output.push_str(&format!(
            "\n[{}] {}\n{}\n",
            severity_str, w.title, w.message
        ));
        if !w.remediation.is_empty() {
            output.push_str("Suggested actions:\n");
            for r in &w.remediation {
                output.push_str(&format!("  - {}\n", r));
            }
        }
    }
    output
}

// ============================================================================
// Tool Router Implementation
// ============================================================================

#[tool_router]
impl AnalyticsServer {
    /// Create a new AnalyticsServer instance.
    pub fn new() -> Self {
        // Compose all routers from handler modules
        let tool_router = Self::tool_router()
            + Self::utils_router()
            + Self::database_router()
            + Self::data_router()
            + Self::viz_router()
            + Self::ml_router()
            + Self::stats_router()
            + Self::hypothesis_router()
            + Self::regression_router()
            + Self::panel_router()
            + Self::discrete_router()
            + Self::causal_router()
            + Self::timeseries_router()
            + Self::spatial_router()
            + Self::munging_router()
            + Self::survival_router()
            + Self::cleaning_router();

        Self {
            datasets: Arc::new(RwLock::new(HashMap::new())),
            cleaning_sessions: Arc::new(RwLock::new(HashMap::new())),
            spatial_weights: Arc::new(RwLock::new(HashMap::new())),
            global_seed: Arc::new(RwLock::new(None)),
            memory_profiler: Arc::new(RwLock::new(p2a_core::MemoryProfiler::new())),
            tool_router,
        }
    }

    /// Create a new AnalyticsServer with existing dataset storage.
    /// Used for HTTP transport where each session has its own dataset store.
    #[cfg(feature = "http")]
    pub fn with_session(session: &crate::session::Session) -> Self {
        // Compose all routers from handler modules
        let tool_router = Self::tool_router()
            + Self::utils_router()
            + Self::database_router()
            + Self::data_router()
            + Self::viz_router()
            + Self::ml_router()
            + Self::stats_router()
            + Self::hypothesis_router()
            + Self::regression_router()
            + Self::panel_router()
            + Self::discrete_router()
            + Self::causal_router()
            + Self::timeseries_router()
            + Self::spatial_router()
            + Self::munging_router()
            + Self::survival_router()
            + Self::cleaning_router();

        Self {
            datasets: session.datasets.clone(),
            cleaning_sessions: Arc::new(RwLock::new(HashMap::new())),
            spatial_weights: Arc::new(RwLock::new(HashMap::new())),
            global_seed: session.global_seed.clone(),
            memory_profiler: Arc::new(RwLock::new(p2a_core::MemoryProfiler::new())),
            tool_router,
        }
    }

    /// List available tools for HTTP API discovery.
    #[cfg(feature = "http")]
    pub fn list_tools(&self) -> Vec<crate::transport::http::ToolDefinition> {
        use crate::transport::http::ToolDefinition;

        // Tool definitions with their descriptions and input schemas
        // This is a static list matching the #[tool] definitions
        vec![
            // Data management tools
            ToolDefinition {
                name: "list_datasets".to_string(),
                description: "List all currently loaded datasets with their basic information (name, dimensions, column types).".to_string(),
                input_schema: serde_json::json!({"type": "object", "properties": {}}),
            },
            ToolDefinition {
                name: "load_dataset".to_string(),
                description: "Load a dataset from a file. Supports CSV, Parquet, Excel, Stata, and SAS formats.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the data file"},
                        "name": {"type": "string", "description": "Optional name for the dataset"}
                    },
                    "required": ["path"]
                }),
            },
            ToolDefinition {
                name: "upload_dataset".to_string(),
                description: "Upload and load a dataset from base64-encoded file content. Use this when the file is selected via browser file picker.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Base64-encoded file content"},
                        "filename": {"type": "string", "description": "Original filename with extension (e.g., 'data.csv')"},
                        "name": {"type": "string", "description": "Optional name for the dataset"}
                    },
                    "required": ["content", "filename"]
                }),
            },
            ToolDefinition {
                name: "create_dataset".to_string(),
                description: "Create a dataset from inline CSV content. Use this to create datasets on-the-fly from generated or inline data without needing a file.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name for the dataset (e.g., 'my_data')"},
                        "csv_content": {"type": "string", "description": "CSV content as plain text with headers in first row (e.g., 'x,y\\n1,2\\n3,4')"}
                    },
                    "required": ["name", "csv_content"]
                }),
            },
            ToolDefinition {
                name: "describe_dataset".to_string(),
                description: "Compute descriptive statistics for all columns in a dataset.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"}
                    },
                    "required": ["dataset"]
                }),
            },
            ToolDefinition {
                name: "head_dataset".to_string(),
                description: "Show the first N rows of a dataset.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "n": {"type": "integer", "description": "Number of rows (default: 5)"}
                    },
                    "required": ["dataset"]
                }),
            },
            ToolDefinition {
                name: "data_quality_profile".to_string(),
                description: "Generate a comprehensive data quality profile for LLM-assisted data cleaning. Returns column-level statistics (nulls, uniques, types), numeric outlier detection, string pattern analysis, and automated issue detection.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset to profile"}
                    },
                    "required": ["dataset"]
                }),
            },
            ToolDefinition {
                name: "compute_correlation".to_string(),
                description: "Compute correlation matrix for numeric columns.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"}
                    },
                    "required": ["dataset"]
                }),
            },
            // Regression tools
            ToolDefinition {
                name: "regression_ols".to_string(),
                description: "Run OLS regression with robust standard errors (HC0-HC3).".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "y": {"type": "string", "description": "Dependent variable"},
                        "x": {"type": "array", "items": {"type": "string"}, "description": "Independent variables"}
                    },
                    "required": ["dataset", "y", "x"]
                }),
            },
            ToolDefinition {
                name: "regression_diagnostics".to_string(),
                description: "Run regression diagnostics (Jarque-Bera, Breusch-Pagan, Durbin-Watson, VIF).".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "y": {"type": "string"},
                        "x": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["dataset", "y", "x"]
                }),
            },
            ToolDefinition {
                name: "regression_clustered".to_string(),
                description: "Run OLS with clustered standard errors.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "y": {"type": "string"},
                        "x": {"type": "array", "items": {"type": "string"}},
                        "cluster1": {"type": "string"},
                        "cluster2": {"type": "string"}
                    },
                    "required": ["dataset", "y", "x", "cluster1"]
                }),
            },
            // Panel econometrics
            ToolDefinition {
                name: "panel_fixed_effects".to_string(),
                description: "Run fixed effects panel regression.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "y": {"type": "string"},
                        "x": {"type": "array", "items": {"type": "string"}},
                        "entity_col": {"type": "string"},
                        "time_col": {"type": "string"}
                    },
                    "required": ["dataset", "y", "x", "entity_col"]
                }),
            },
            ToolDefinition {
                name: "panel_random_effects".to_string(),
                description: "Run random effects panel regression.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "y": {"type": "string"},
                        "x": {"type": "array", "items": {"type": "string"}},
                        "entity_col": {"type": "string"}
                    },
                    "required": ["dataset", "y", "x", "entity_col"]
                }),
            },
            ToolDefinition {
                name: "panel_hdfe".to_string(),
                description: "Run high-dimensional fixed effects regression.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "y": {"type": "string"},
                        "x": {"type": "array", "items": {"type": "string"}},
                        "fe": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["dataset", "y", "x", "fe"]
                }),
            },
            ToolDefinition {
                name: "hausman_test".to_string(),
                description: "Perform Hausman test for FE vs RE specification.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "y": {"type": "string"},
                        "x": {"type": "array", "items": {"type": "string"}},
                        "entity_col": {"type": "string"}
                    },
                    "required": ["dataset", "y", "x", "entity_col"]
                }),
            },
            ToolDefinition {
                name: "panel_gmm".to_string(),
                description: "Run Arellano-Bond or System GMM for dynamic panel data models.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "y": {"type": "string"},
                        "x": {"type": "array", "items": {"type": "string"}},
                        "entity_var": {"type": "string"},
                        "time_var": {"type": "string"},
                        "lags": {"type": "integer", "default": 1},
                        "transform": {"type": "string", "enum": ["difference", "system"], "default": "difference"},
                        "step": {"type": "string", "enum": ["onestep", "twostep"], "default": "twostep"},
                        "max_lag": {"type": "integer"},
                        "min_lag": {"type": "integer", "default": 2},
                        "collapse": {"type": "boolean", "default": false},
                        "robust": {"type": "boolean", "default": true}
                    },
                    "required": ["dataset", "y", "x", "entity_var", "time_var"]
                }),
            },
            // Causal inference
            ToolDefinition {
                name: "iv_2sls".to_string(),
                description: "Run two-stage least squares regression.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "y": {"type": "string"},
                        "endogenous": {"type": "array", "items": {"type": "string"}},
                        "instruments": {"type": "array", "items": {"type": "string"}},
                        "exogenous": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["dataset", "y", "endogenous", "instruments"]
                }),
            },
            ToolDefinition {
                name: "iv_first_stage".to_string(),
                description: "Run first-stage diagnostics for IV regression.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "endogenous": {"type": "string"},
                        "instruments": {"type": "array", "items": {"type": "string"}},
                        "exogenous": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["dataset", "endogenous", "instruments"]
                }),
            },
            ToolDefinition {
                name: "iv_sargan_test".to_string(),
                description: "Run Sargan test of overidentifying restrictions for IV/2SLS. Tests whether instruments are valid (uncorrelated with the error term). Requires more instruments than endogenous variables.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "y": {"type": "string", "description": "Dependent variable name"},
                        "x_exog": {"type": "array", "items": {"type": "string"}, "description": "Exogenous variables (may be empty)"},
                        "x_endog": {"type": "array", "items": {"type": "string"}, "description": "Endogenous variables to be instrumented"},
                        "instruments": {"type": "array", "items": {"type": "string"}, "description": "Instrumental variables (must exceed x_endog count)"}
                    },
                    "required": ["dataset", "y", "x_endog", "instruments"]
                }),
            },
            ToolDefinition {
                name: "diff_in_diff".to_string(),
                description: "Run difference-in-differences analysis.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "y": {"type": "string"},
                        "treatment_col": {"type": "string"},
                        "time_col": {"type": "string"},
                        "treatment_time": {"type": "number"}
                    },
                    "required": ["dataset", "y", "treatment_col", "time_col", "treatment_time"]
                }),
            },
            // Discrete choice
            ToolDefinition {
                name: "logit".to_string(),
                description: "Run logistic regression.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "y": {"type": "string"},
                        "x": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["dataset", "y", "x"]
                }),
            },
            ToolDefinition {
                name: "probit".to_string(),
                description: "Run probit regression.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "y": {"type": "string"},
                        "x": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["dataset", "y", "x"]
                }),
            },
            // Time series
            ToolDefinition {
                name: "ts_var".to_string(),
                description: "Estimate Vector Autoregression model.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "columns": {"type": "array", "items": {"type": "string"}},
                        "lags": {"type": "integer"}
                    },
                    "required": ["dataset", "columns", "lags"]
                }),
            },
            ToolDefinition {
                name: "ts_arima_fit".to_string(),
                description: "Fit ARIMA model to time series.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "column": {"type": "string"},
                        "p": {"type": "integer"},
                        "d": {"type": "integer"},
                        "q": {"type": "integer"}
                    },
                    "required": ["dataset", "column", "p", "d", "q"]
                }),
            },
            // Machine learning
            ToolDefinition {
                name: "ml_kmeans".to_string(),
                description: "Run K-means clustering.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "columns": {"type": "array", "items": {"type": "string"}},
                        "k": {"type": "integer"}
                    },
                    "required": ["dataset", "columns", "k"]
                }),
            },
            ToolDefinition {
                name: "ml_pca".to_string(),
                description: "Run Principal Component Analysis.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "columns": {"type": "array", "items": {"type": "string"}},
                        "n_components": {"type": "integer"}
                    },
                    "required": ["dataset", "columns"]
                }),
            },
            ToolDefinition {
                name: "ml_cmdscale".to_string(),
                description: "Classical Multidimensional Scaling for embedding distances into Euclidean space.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "columns": {"type": "array", "items": {"type": "string"}},
                        "k": {"type": "integer"},
                        "is_distance_matrix": {"type": "boolean"}
                    },
                    "required": ["dataset", "columns"]
                }),
            },
            ToolDefinition {
                name: "ml_cutree".to_string(),
                description: "Cut hierarchical clustering dendrogram into groups.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "columns": {"type": "array", "items": {"type": "string"}},
                        "k": {"type": "integer"},
                        "cut_height": {"type": "number"},
                        "linkage": {"type": "string"}
                    },
                    "required": ["dataset", "columns"]
                }),
            },
            // Visualization
            ToolDefinition {
                name: "viz_histogram".to_string(),
                description: "Create a histogram plot.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "column": {"type": "string"},
                        "bins": {"type": "integer"}
                    },
                    "required": ["dataset", "column"]
                }),
            },
            ToolDefinition {
                name: "viz_scatter".to_string(),
                description: "Create a scatter plot.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "x": {"type": "string"},
                        "y": {"type": "string"}
                    },
                    "required": ["dataset", "x", "y"]
                }),
            },
            ToolDefinition {
                name: "viz_line".to_string(),
                description: "Create a line chart.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "x": {"type": "string"},
                        "y": {"type": "string"}
                    },
                    "required": ["dataset", "x", "y"]
                }),
            },
            ToolDefinition {
                name: "viz_heatmap".to_string(),
                description: "Create a correlation heatmap.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"}
                    },
                    "required": ["dataset"]
                }),
            },
            // Interactive Visualization tools (HTML/Plotly output)
            ToolDefinition {
                name: "viz_scatter_interactive".to_string(),
                description: "Create an interactive scatter plot (HTML with Plotly.js).".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "x_column": {"type": "string"},
                        "y_column": {"type": "string"},
                        "group_column": {"type": "string"},
                        "title": {"type": "string"}
                    },
                    "required": ["dataset", "x_column", "y_column"]
                }),
            },
            ToolDefinition {
                name: "viz_histogram_interactive".to_string(),
                description: "Create an interactive histogram (HTML with Plotly.js).".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "column": {"type": "string"},
                        "group_column": {"type": "string"},
                        "title": {"type": "string"}
                    },
                    "required": ["dataset", "column"]
                }),
            },
            ToolDefinition {
                name: "viz_line_interactive".to_string(),
                description: "Create an interactive line chart (HTML with Plotly.js).".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "x_column": {"type": "string"},
                        "y_column": {"type": "string"},
                        "title": {"type": "string"}
                    },
                    "required": ["dataset", "x_column", "y_column"]
                }),
            },
            // Database tools
            ToolDefinition {
                name: "db_sqlite_query".to_string(),
                description: "Execute SQL query on SQLite database.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "database": {"type": "string"},
                        "query": {"type": "string"},
                        "name": {"type": "string"}
                    },
                    "required": ["database", "query"]
                }),
            },
            ToolDefinition {
                name: "db_duckdb_query".to_string(),
                description: "Execute SQL query on DuckDB database.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "database": {"type": "string"},
                        "query": {"type": "string"},
                        "name": {"type": "string"}
                    },
                    "required": ["database", "query"]
                }),
            },
            ToolDefinition {
                name: "db_query_file".to_string(),
                description: "Execute SQL query directly on a Parquet or CSV file using DuckDB. Use {file} as placeholder for the file path.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to the Parquet or CSV file"},
                        "query": {"type": "string", "description": "SQL query with {file} placeholder for the file path"},
                        "name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["file_path", "query"]
                }),
            },
            // Data munging tools
            ToolDefinition {
                name: "munge_filter".to_string(),
                description: "Filter rows in a dataset based on a condition. Supports operators: eq, ne, gt, ge, lt, le, contains, startswith, endswith.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset to filter"},
                        "column": {"type": "string", "description": "Column to filter on"},
                        "operator": {"type": "string", "description": "Comparison operator (eq, ne, gt, ge, lt, le, contains, startswith, endswith)"},
                        "value": {"type": "string", "description": "Value to compare against"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "column", "operator", "value"]
                }),
            },
            ToolDefinition {
                name: "munge_select".to_string(),
                description: "Select specific columns from a dataset.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "columns": {"type": "array", "items": {"type": "string"}, "description": "Columns to select"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "columns"]
                }),
            },
            ToolDefinition {
                name: "munge_drop_columns".to_string(),
                description: "Drop columns from a dataset.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "columns": {"type": "array", "items": {"type": "string"}, "description": "Columns to drop"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "columns"]
                }),
            },
            ToolDefinition {
                name: "munge_rename".to_string(),
                description: "Rename columns in a dataset.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "renames": {"type": "object", "additionalProperties": {"type": "string"}, "description": "Map of old names to new names"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "renames"]
                }),
            },
            ToolDefinition {
                name: "munge_sort".to_string(),
                description: "Sort a dataset by one or more columns.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "columns": {"type": "array", "items": {"type": "string"}, "description": "Columns to sort by"},
                        "descending": {"type": "boolean", "description": "Sort in descending order (default: false)"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "columns"]
                }),
            },
            ToolDefinition {
                name: "munge_mutate".to_string(),
                description: "Create a new column or modify an existing one using an expression. Supports arithmetic operations on columns.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "new_column": {"type": "string", "description": "Name of the new column"},
                        "expression": {"type": "string", "description": "Expression type: 'copy', 'constant', 'add', 'subtract', 'multiply', 'divide'"},
                        "left": {"type": "string", "description": "Left operand (column name or constant value)"},
                        "right": {"type": "string", "description": "Right operand (column name, for arithmetic operations)"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "new_column", "expression", "left"]
                }),
            },
            ToolDefinition {
                name: "munge_sample".to_string(),
                description: "Take a random sample of rows from a dataset.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "n": {"type": "integer", "description": "Number of rows to sample"},
                        "with_replacement": {"type": "boolean", "description": "Sample with replacement (default: false)"},
                        "seed": {"type": "integer", "description": "Optional random seed for reproducibility"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "n"]
                }),
            },
            ToolDefinition {
                name: "munge_join".to_string(),
                description: "Join two datasets on key columns. Supports left, right, inner, and full outer joins.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "left_dataset": {"type": "string", "description": "Name of the left dataset"},
                        "right_dataset": {"type": "string", "description": "Name of the right dataset"},
                        "on": {"type": "array", "items": {"type": "string"}, "description": "Columns to join on"},
                        "right_on": {"type": "array", "items": {"type": "string"}, "description": "Right key columns if different from left"},
                        "join_type": {"type": "string", "description": "Join type: left, right, inner, full (default: left)"},
                        "suffix": {"type": "string", "description": "Suffix for duplicate column names (default: _right)"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["left_dataset", "right_dataset", "on"]
                }),
            },
            ToolDefinition {
                name: "munge_concat".to_string(),
                description: "Concatenate multiple datasets vertically (row-bind).".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "datasets": {"type": "array", "items": {"type": "string"}, "description": "Names of datasets to concatenate"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["datasets"]
                }),
            },
            ToolDefinition {
                name: "munge_group_by".to_string(),
                description: "Group dataset by columns and compute aggregations (sum, mean, count, min, max, std, var, first, last, median).".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "by": {"type": "array", "items": {"type": "string"}, "description": "Columns to group by"},
                        "aggs": {"type": "array", "items": {"type": "object"}, "description": "Aggregation specs: [{column, function}]"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "by", "aggs"]
                }),
            },
            ToolDefinition {
                name: "munge_value_counts".to_string(),
                description: "Count occurrences of unique values in a column.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "column": {"type": "string", "description": "Column to count values in"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "column"]
                }),
            },
            ToolDefinition {
                name: "munge_pivot".to_string(),
                description: "Pivot a dataset from long to wide format.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "index": {"type": "array", "items": {"type": "string"}, "description": "Columns to use as index (row identifiers)"},
                        "on": {"type": "string", "description": "Column whose values become new column names"},
                        "values": {"type": "string", "description": "Column containing values to fill the new columns"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "index", "on", "values"]
                }),
            },
            ToolDefinition {
                name: "munge_melt".to_string(),
                description: "Melt a dataset from wide to long format (unpivot).".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "id_vars": {"type": "array", "items": {"type": "string"}, "description": "Columns to keep as identifiers"},
                        "value_vars": {"type": "array", "items": {"type": "string"}, "description": "Columns to unpivot into rows"},
                        "variable_name": {"type": "string", "description": "Name for the variable column (default: variable)"},
                        "value_name": {"type": "string", "description": "Name for the value column (default: value)"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "id_vars", "value_vars"]
                }),
            },
            ToolDefinition {
                name: "munge_drop_na".to_string(),
                description: "Drop rows with missing values.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "columns": {"type": "array", "items": {"type": "string"}, "description": "Columns to check for NA (all if not specified)"},
                        "how": {"type": "string", "description": "How to drop: 'any' or 'all' (default: any)"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset"]
                }),
            },
            ToolDefinition {
                name: "munge_fill_na".to_string(),
                description: "Fill missing values using a strategy (mean, median, constant, forward, backward, zero).".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "columns": {"type": "array", "items": {"type": "string"}, "description": "Columns to fill (all if not specified)"},
                        "strategy": {"type": "string", "description": "Fill strategy: mean, median, constant, forward, backward, zero"},
                        "constant_value": {"type": "number", "description": "Value to use for constant strategy"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "strategy"]
                }),
            },
            ToolDefinition {
                name: "munge_deduplicate".to_string(),
                description: "Remove duplicate rows from a dataset.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "subset": {"type": "array", "items": {"type": "string"}, "description": "Columns to consider for duplicates (all if not specified)"},
                        "keep": {"type": "string", "description": "Which duplicate to keep: 'first', 'last', or 'none' (default: first)"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset"]
                }),
            },
            ToolDefinition {
                name: "munge_lag_lead".to_string(),
                description: "Create lag or lead of a column (shift values forward or backward).".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "column": {"type": "string", "description": "Column to lag or lead"},
                        "periods": {"type": "integer", "description": "Number of periods to shift"},
                        "operation": {"type": "string", "description": "Operation: 'lag' or 'lead'"},
                        "group_by": {"type": "array", "items": {"type": "string"}, "description": "Optional group by columns for panel data"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "column", "periods", "operation"]
                }),
            },
            ToolDefinition {
                name: "munge_diff".to_string(),
                description: "Compute difference or percentage change of a column.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "column": {"type": "string", "description": "Column to difference"},
                        "periods": {"type": "integer", "description": "Number of periods for differencing (default: 1)"},
                        "pct_change": {"type": "boolean", "description": "Compute percentage change instead of difference (default: false)"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "column"]
                }),
            },
            ToolDefinition {
                name: "munge_standardize".to_string(),
                description: "Standardize (z-score) or normalize (0-1) columns.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "columns": {"type": "array", "items": {"type": "string"}, "description": "Columns to standardize"},
                        "method": {"type": "string", "description": "Method: 'standardize' (z-score) or 'normalize' (0-1)"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "columns"]
                }),
            },
            ToolDefinition {
                name: "munge_bin".to_string(),
                description: "Bin a continuous column into discrete categories.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "column": {"type": "string", "description": "Column to bin"},
                        "strategy": {"type": "string", "description": "Binning strategy: 'equal_width', 'quantile', or 'custom'"},
                        "n_bins": {"type": "integer", "description": "Number of bins for equal_width or quantile strategies"},
                        "breaks": {"type": "array", "items": {"type": "number"}, "description": "Custom bin edges for custom strategy"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "column", "strategy"]
                }),
            },
            ToolDefinition {
                name: "munge_one_hot_encode".to_string(),
                description: "One-hot encode a categorical column into dummy variables.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "column": {"type": "string", "description": "Column to one-hot encode"},
                        "drop_first": {"type": "boolean", "description": "Drop first category to avoid multicollinearity (default: false)"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "column"]
                }),
            },
        ]
    }

    /// Call a tool by name with session context (for HTTP transport).
    /// This creates a session-scoped server and dispatches the tool call.
    #[cfg(feature = "http")]
    pub async fn call_tool_with_session(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
        session: &crate::session::Session,
    ) -> Result<crate::transport::http::ToolResult, String> {
        use crate::transport::http::{ContentItem, ToolResult};

        // Create a session-scoped server instance that shares the session's datasets
        let session_server = Self::with_session(session);

        // Helper to convert CallToolResult to our ToolResult
        fn convert_result(call_result: CallToolResult) -> ToolResult {
            let is_error = call_result.is_error.unwrap_or(false);
            let content: Vec<ContentItem> = call_result
                .content
                .into_iter()
                .filter_map(|c| {
                    // Content in rmcp is an Annotated<RawContent>
                    // We need to access the inner raw content
                    match &c.raw {
                        RawContent::Text(text_content) => Some(ContentItem::Text {
                            text: text_content.text.clone(),
                        }),
                        RawContent::Image(img) => Some(ContentItem::Image {
                            data: img.data.clone(),
                            mime_type: img.mime_type.clone(),
                        }),
                        _ => None,
                    }
                })
                .collect();

            let error = if is_error {
                content.first().and_then(|c| {
                    if let ContentItem::Text { text } = c {
                        Some(text.clone())
                    } else {
                        None
                    }
                })
            } else {
                None
            };

            ToolResult {
                success: !is_error,
                content,
                error,
            }
        }

        // Parse arguments and dispatch to the appropriate tool method
        // For now, we support a subset of the most commonly used tools
        let result = match tool_name {
            "list_datasets" => session_server.list_datasets().await,
            "load_dataset" => {
                let req: LoadDatasetRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.load_dataset(Parameters(req)).await
            }
            "upload_dataset" => {
                let req: UploadDatasetRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.upload_dataset(Parameters(req)).await
            }
            "create_dataset" => {
                let req: CreateDatasetRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.create_dataset(Parameters(req)).await
            }
            "describe_dataset" => {
                let req: DescribeDatasetRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.describe_dataset(Parameters(req)).await
            }
            "head_dataset" => {
                let req: HeadDatasetRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.head_dataset(Parameters(req)).await
            }
            "data_quality_profile" => {
                let req: DataQualityProfileRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.data_quality_profile(Parameters(req)).await
            }
            "preview_cleaning" => {
                let req: PreviewCleaningRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.preview_cleaning(Parameters(req)).await
            }
            "verify_cleaning" => {
                let req: VerifyCleaningRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.verify_cleaning(Parameters(req)).await
            }
            "compute_correlation" => {
                let req: CorrelationRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.compute_correlation(Parameters(req)).await
            }
            "regression_ols" => {
                let req: OlsRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.regression_ols(Parameters(req)).await
            }
            "regression_diagnostics" => {
                let req: DiagnosticsRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.regression_diagnostics(Parameters(req)).await
            }
            "panel_fixed_effects" => {
                let req: PanelFERequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.panel_fixed_effects(Parameters(req)).await
            }
            "panel_random_effects" => {
                let req: PanelRERequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.panel_random_effects(Parameters(req)).await
            }
            "panel_gmm" => {
                let req: GmmRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.panel_gmm(Parameters(req)).await
            }
            "iv_2sls" => {
                let req: IV2SLSRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.iv_2sls(Parameters(req)).await
            }
            "diff_in_diff" => {
                let req: DiDRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.diff_in_diff(Parameters(req)).await
            }
            "treatment_ipw" => {
                let req: IpwRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.treatment_ipw(Parameters(req)).await
            }
            "treatment_doubly_robust" => {
                let req: DoublyRobustRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .treatment_doubly_robust(Parameters(req))
                    .await
            }
            "treatment_tmle" => {
                let req: TmleRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.treatment_tmle(Parameters(req)).await
            }
            "collaborative_tmle" => {
                let req: CTmleRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.collaborative_tmle(Parameters(req)).await
            }
            "ltmle" => {
                let req: LtmleRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.ltmle(Parameters(req)).await
            }
            "regression_standardization" => {
                let req: StdRegRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .regression_standardization(Parameters(req))
                    .await
            }
            "gformula" => {
                let req: GFormulaRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.gformula(Parameters(req)).await
            }
            "mediation_analysis" => {
                let req: MediationRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.mediation_analysis(Parameters(req)).await
            }
            "natural_effects_mediation" => {
                let req: NaturalEffectsRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .natural_effects_mediation(Parameters(req))
                    .await
            }
            "logit" => {
                let req: LogitRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.logit(Parameters(req)).await
            }
            "probit" => {
                let req: ProbitRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.probit(Parameters(req)).await
            }
            "multinom" => {
                let req: MultinomRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.multinom(Parameters(req)).await
            }
            "ordered_model" => {
                let req: OrderedRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.ordered_model(Parameters(req)).await
            }
            "negbin" => {
                let req: NegBinRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.negbin(Parameters(req)).await
            }
            "zeroinfl" => {
                let req: ZeroInflRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.zeroinfl(Parameters(req)).await
            }
            "ml_kmeans" => {
                let req: KMeansRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.ml_kmeans(Parameters(req)).await
            }
            "ml_pca" => {
                let req: PCARequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.ml_pca(Parameters(req)).await
            }
            "ml_cmdscale" => {
                let req: CmdscaleRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.ml_cmdscale(Parameters(req)).await
            }
            "ml_cutree" => {
                let req: CutreeRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.ml_cutree(Parameters(req)).await
            }
            "viz_histogram" => {
                let req: HistogramRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.viz_histogram(Parameters(req)).await
            }
            "viz_scatter" => {
                let req: ScatterPlotRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.viz_scatter(Parameters(req)).await
            }
            "viz_line" => {
                let req: LineChartRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.viz_line(Parameters(req)).await
            }
            "viz_heatmap" => {
                let req: HeatmapRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.viz_heatmap(Parameters(req)).await
            }
            "viz_scatter_interactive" => {
                let req: ScatterInteractiveRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .viz_scatter_interactive(Parameters(req))
                    .await
            }
            "viz_histogram_interactive" => {
                let req: HistogramInteractiveRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .viz_histogram_interactive(Parameters(req))
                    .await
            }
            "viz_line_interactive" => {
                let req: LineInteractiveRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.viz_line_interactive(Parameters(req)).await
            }
            "db_sqlite_query" => {
                let req: SqliteQueryRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.db_sqlite_query(Parameters(req)).await
            }
            "db_duckdb_query" => {
                let req: DuckDBQueryRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.db_duckdb_query(Parameters(req)).await
            }
            "db_query_file" => {
                let req: DuckDBFileQueryRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.db_query_file(Parameters(req)).await
            }
            // Data munging tools
            "munge_filter" => {
                let req: FilterDatasetRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.munge_filter(Parameters(req)).await
            }
            "munge_select" => {
                let req: SelectColumnsRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.munge_select(Parameters(req)).await
            }
            "munge_drop_columns" => {
                let req: DropColumnsRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.munge_drop_columns(Parameters(req)).await
            }
            "munge_rename" => {
                let req: RenameColumnsRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.munge_rename(Parameters(req)).await
            }
            "munge_sort" => {
                let req: SortDatasetRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.munge_sort(Parameters(req)).await
            }
            "munge_mutate" => {
                let req: MutateColumnRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.munge_mutate(Parameters(req)).await
            }
            "munge_sample" => {
                let req: SampleDatasetRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.munge_sample(Parameters(req)).await
            }
            "munge_join" => {
                let req: JoinDatasetsRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.munge_join(Parameters(req)).await
            }
            "munge_concat" => {
                let req: ConcatDatasetsRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.munge_concat(Parameters(req)).await
            }
            "munge_group_by" => {
                let req: GroupByRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.munge_group_by(Parameters(req)).await
            }
            "munge_value_counts" => {
                let req: ValueCountsRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.munge_value_counts(Parameters(req)).await
            }
            "munge_pivot" => {
                let req: PivotDatasetRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.munge_pivot(Parameters(req)).await
            }
            "munge_melt" => {
                let req: MeltDatasetRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.munge_melt(Parameters(req)).await
            }
            "munge_drop_na" => {
                let req: DropNaRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.munge_drop_na(Parameters(req)).await
            }
            "munge_fill_na" => {
                let req: FillNaRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.munge_fill_na(Parameters(req)).await
            }
            "munge_deduplicate" => {
                let req: DeduplicateRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.munge_deduplicate(Parameters(req)).await
            }
            "munge_lag_lead" => {
                let req: LagLeadRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.munge_lag_lead(Parameters(req)).await
            }
            "munge_diff" => {
                let req: DiffRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.munge_diff(Parameters(req)).await
            }
            "munge_standardize" => {
                let req: StandardizeRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.munge_standardize(Parameters(req)).await
            }
            "munge_bin" => {
                let req: BinColumnRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.munge_bin(Parameters(req)).await
            }
            "munge_one_hot_encode" => {
                let req: OneHotEncodeRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server.munge_one_hot_encode(Parameters(req)).await
            }
            _ => {
                return Err(format!("Unknown tool: {}", tool_name));
            }
        };

        match result {
            Ok(call_result) => Ok(convert_result(call_result)),
            Err(e) => Err(format!("Tool execution failed: {:?}", e)),
        }
    }

    // ========================================================================
    // Cleaning Session Management Tools
    // ========================================================================

    // ========================================================================
    // ANOVA Tools
    // ========================================================================

    // ========================================================================
    // Chi-Squared Tests
    // ========================================================================

    // ========================================================================
    // Correlation Test
    // ========================================================================

    // ========================================================================
    // Power Analysis Tools
    // ========================================================================

    // ========================================================================
    // Trend Test for Proportions
    // ========================================================================

    // ========================================================================
    // Econometrics Tools
    // ========================================================================

    // ========================================================================
    // Treatment Effect Estimation
    // ========================================================================

    // ========================================================================
    // Causal Mediation Analysis
    // ========================================================================

    // ========================================================================
    // Synthetic Control Method
    // ========================================================================

    // ========================================================================
    // Generalized Synthetic Control
    // ========================================================================

    // ========================================================================
    // Synthetic Control with Prediction Intervals (SCPI)
    // ========================================================================

    // ========================================================================
    // Regression Discontinuity
    // ========================================================================

    // ========================================================================
    // Survival Analysis
    // ========================================================================

    // ========================================================================
    // Discrete Choice Models
    // ========================================================================

    // ========================================================================
    // Spatial Econometrics
    // ========================================================================

    // ========================================================================
    // Time Series Models
    // ========================================================================

    // ========================================================================
    // Forecasting Models
    // ========================================================================

    // ========================================================================
    // Report Generation Tools
    // ========================================================================

    // ========================================================================
    // Machine Learning Tools
    // ========================================================================

    // ========================================================================
    // Database Tools
    // ========================================================================

    // ========================================================================
    // Visualization Tools
    // ========================================================================

    // ========================================================================
    // Data Munging Tools
    // ========================================================================

    // =========================================================================
    // STRING CLEANING TOOLS
    // =========================================================================

    // =========================================================================
    // REGEX TOOLS
    // =========================================================================

    // =========================================================================
    // STRING MANIPULATION TOOLS
    // =========================================================================

    // ========================================================================
    // Robust Statistics Tools
    // ========================================================================

    // ========================================================================
    // Spline/Interpolation Tools
    // ========================================================================

    // ========================================================================
    // GLS and Smooth Spline Tools
    // ========================================================================
}

/// Helper function to extract a single column as Vec<f64>.
fn extract_column_f64(dataset: &Dataset, column: &str) -> Result<Vec<f64>, String> {
    use p2a_core::polars::prelude::*;

    let df = dataset.df();
    let col = df
        .column(column)
        .map_err(|e| format!("Column '{}' not found: {}", column, e))?;

    let values: Vec<f64> = col
        .cast(&DataType::Float64)
        .map_err(|e| format!("Cannot convert column '{}' to numeric: {}", column, e))?
        .f64()
        .map_err(|e| format!("Column '{}' is not numeric: {}", column, e))?
        .into_iter()
        .map(|v: Option<f64>| v.unwrap_or(f64::NAN))
        .collect();

    Ok(values)
}

/// Helper function to extract numeric columns into an ndarray matrix.
fn extract_numeric_matrix(
    dataset: &Dataset,
    columns: &[String],
) -> Result<ndarray::Array2<f64>, String> {
    use p2a_core::polars::prelude::*;

    let df = dataset.df();
    let n_rows = df.height();
    let n_cols = columns.len();

    if columns.is_empty() {
        return Err("At least one column must be specified".to_string());
    }

    let mut data = ndarray::Array2::zeros((n_rows, n_cols));

    for (j, col_name) in columns.iter().enumerate() {
        let col = df
            .column(col_name)
            .map_err(|e| format!("Column '{}' not found: {}", col_name, e))?;

        let values: Vec<f64> = col
            .cast(&DataType::Float64)
            .map_err(|e| format!("Cannot convert column '{}' to numeric: {}", col_name, e))?
            .f64()
            .map_err(|e| format!("Column '{}' is not numeric: {}", col_name, e))?
            .into_iter()
            .map(|v: Option<f64>| v.unwrap_or(f64::NAN))
            .collect();

        for (i, &val) in values.iter().enumerate() {
            data[[i, j]] = val;
        }
    }

    Ok(data)
}

// ============================================================================
// ServerHandler Implementation
// ============================================================================

#[tool_handler(router = self.tool_router)]
impl ServerHandler for AnalyticsServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "prompt2analytics".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                ..Default::default()
            },
            instructions: Some(
                "prompt2analytics is a local data analytics engine. \
                 Use 'load_dataset' to load a CSV or Parquet file, or \
                 'db_sqlite_query'/'db_duckdb_query' to query databases. \
                 Then use 'describe_dataset' for summary statistics, \
                 'compute_correlation' for correlations, 'regression_ols' \
                 for linear regression, 'regression_diagnostics' for model validation, \
                 'panel_fixed_effects' or 'panel_random_effects' for panel data, \
                 'hausman_test' to choose between FE/RE, 'iv_2sls' for instrumental \
                 variables, 'diff_in_diff' for difference-in-differences, \
                 'logit' or 'probit' for binary outcomes, \
                 'ts_var' for VAR models, 'ts_varma' for VARMA models, \
                 'ts_vecm' for cointegration analysis, 'ts_var_irf' for impulse responses, \
                 'ml_kmeans' for K-means clustering, 'ml_dbscan' for DBSCAN clustering, \
                 'ml_pca' for principal component analysis, visualization tools: \
                 'viz_histogram', 'viz_scatter', 'viz_line', 'viz_boxplot', 'viz_heatmap' (static PNG), \
                 or interactive tools: 'viz_scatter_interactive', 'viz_histogram_interactive', \
                 'viz_line_interactive' (HTML/Plotly.js output)."
                    .to_string(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p2a_core::data::Dataset;
    use polars::prelude::df;
    use rmcp::model::RawContent;

    /// Helper to create a server with a pre-loaded dataset
    async fn server_with_dataset(name: &str, dataset: Dataset) -> AnalyticsServer {
        let server = AnalyticsServer::new();
        server
            .datasets
            .write()
            .await
            .insert(name.to_string(), dataset);
        server
    }

    /// Helper to extract text from Content (Annotated<RawContent>)
    fn get_text_content(content: &Content) -> Option<String> {
        match &content.raw {
            RawContent::Text(text_content) => Some(text_content.text.clone()),
            _ => None,
        }
    }

    // =========================================================================
    // Phase 6: LLM-Assisted Data Cleaning Tests
    // =========================================================================








}
