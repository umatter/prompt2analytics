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

use crate::tools::requests::*;

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
