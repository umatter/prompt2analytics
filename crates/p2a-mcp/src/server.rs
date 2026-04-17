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
        did_diagnostics, ipw_diagnostics, iv_diagnostics, matching_diagnostics, rd_diagnostics,
        staggered_did_diagnostics,
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
            + Self::cleaning_router()
            + Self::search_router();

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
            + Self::cleaning_router()
            + Self::search_router();

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
    /// Dynamically generates the tool list from all registered tool routers.
    #[cfg(feature = "http")]
    pub fn list_tools(&self) -> Vec<crate::transport::http::ToolDefinition> {
        use crate::transport::http::ToolDefinition;

        self.tool_router
            .list_all()
            .into_iter()
            .map(|tool| ToolDefinition {
                name: tool.name.to_string(),
                description: tool.description.unwrap_or_default().to_string(),
                input_schema: serde_json::to_value(&tool.input_schema).unwrap_or_default(),
            })
            .collect()
    }

    /// List tools filtered for LLM function calling (Tier 1 only).
    /// Auto-generates definitions from the router's JsonSchema derives.
    #[cfg(feature = "http")]
    pub fn list_tools_for_llm(&self) -> Vec<crate::llm::ToolDefinition> {
        /// Convert `prefixItems` (JSON Schema draft 2020-12, used by schemars for tuples)
        /// to `items` (draft 7, required by OpenAI). Recurses through the entire schema.
        fn fix_prefix_items(value: &mut serde_json::Value) {
            if let Some(obj) = value.as_object_mut() {
                // If this object has prefixItems but no items, convert it
                if obj.contains_key("prefixItems") && !obj.contains_key("items") {
                    if let Some(prefix) = obj.remove("prefixItems") {
                        // For tuples like (i64, i64), all items share a type — use first
                        if let Some(first) = prefix.as_array().and_then(|a| a.first()) {
                            obj.insert("items".to_string(), first.clone());
                        }
                        // Set type to array if not already
                        obj.entry("type").or_insert(serde_json::json!("array"));
                    }
                }
                // Recurse into all values
                for v in obj.values_mut() {
                    fix_prefix_items(v);
                }
            } else if let Some(arr) = value.as_array_mut() {
                for v in arr.iter_mut() {
                    fix_prefix_items(v);
                }
            }
        }

        use crate::llm::{INTERNAL_TOOLS, TIER1_TOOLS, ToolDefinition};

        self.tool_router
            .list_all()
            .into_iter()
            .filter(|t| {
                let name = t.name.as_ref();
                !INTERNAL_TOOLS.contains(&name) && TIER1_TOOLS.contains(&name)
            })
            .map(|t| {
                let mut params = serde_json::to_value(&t.input_schema).unwrap_or_default();
                // Fix JSON Schema compatibility: convert prefixItems (draft 2020-12)
                // to items (draft 7) for OpenAI API compatibility.
                fix_prefix_items(&mut params);
                ToolDefinition {
                    name: t.name.to_string(),
                    description: t.description.unwrap_or_default().to_string(),
                    parameters: params,
                }
            })
            .collect()
    }

    /// Call a tool by name with session context (for HTTP transport).
    /// Dispatches to all registered tool handler methods directly, bypassing the
    /// ToolRouter (which requires a Peer that cannot be constructed outside rmcp).
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
                .filter_map(|c| match &c.raw {
                    RawContent::Text(text_content) => Some(ContentItem::Text {
                        text: text_content.text.clone(),
                    }),
                    RawContent::Image(img) => Some(ContentItem::Image {
                        data: img.data.clone(),
                        mime_type: img.mime_type.clone(),
                    }),
                    _ => None,
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

        // Macro to reduce boilerplate for dispatching to tool handler methods.
        // Each arm deserializes the JSON arguments into the correct request type
        // and calls the corresponding async method on the session server.
        macro_rules! dispatch {
            ($name:literal => no_params => $method:ident) => {
                if tool_name == $name {
                    let result = session_server
                        .$method()
                        .await
                        .map_err(|e| format!("Tool execution failed: {:?}", e))?;
                    return Ok(convert_result(result));
                }
            };
            ($name:literal => $req:ty => $method:ident) => {
                if tool_name == $name {
                    let request: $req = serde_json::from_value(arguments.clone()).map_err(|e| {
                        format!("Failed to parse arguments for '{}': {}", tool_name, e)
                    })?;
                    let result = session_server
                        .$method(Parameters(request))
                        .await
                        .map_err(|e| format!("Tool execution failed: {:?}", e))?;
                    return Ok(convert_result(result));
                }
            };
        }

        // ===== Data management =====
        dispatch!("list_datasets" => no_params => list_datasets);
        dispatch!("load_dataset" => LoadDatasetRequest => load_dataset);
        dispatch!("export_dataset" => ExportDatasetRequest => export_dataset);
        dispatch!("upload_dataset" => UploadDatasetRequest => upload_dataset);
        dispatch!("create_dataset" => CreateDatasetRequest => create_dataset);
        dispatch!("describe_dataset" => DescribeDatasetRequest => describe_dataset);
        dispatch!("head_dataset" => HeadDatasetRequest => head_dataset);

        // ===== Utils =====
        dispatch!("set_seed" => SetSeedRequest => set_seed);
        dispatch!("get_seed" => GetSeedRequest => get_seed);
        dispatch!("server_stats" => ServerStatsRequest => server_stats);
        dispatch!("generate_random_data" => GenerateRandomDataRequest => generate_random_data);
        dispatch!("generate_report" => GenerateReportRequest => generate_report);
        dispatch!("export_session" => ExportSessionRequest => export_session);
        dispatch!("import_session" => ImportSessionRequest => import_session);

        // ===== Database =====
        dispatch!("db_sqlite_query" => SqliteQueryRequest => db_sqlite_query);
        dispatch!("db_sqlite_tables" => SqliteListTablesRequest => db_sqlite_tables);
        dispatch!("db_sqlite_schema" => SqliteSchemaRequest => db_sqlite_schema);
        dispatch!("db_duckdb_query" => DuckDBQueryRequest => db_duckdb_query);
        dispatch!("db_duckdb_tables" => DuckDBListTablesRequest => db_duckdb_tables);
        dispatch!("db_duckdb_schema" => DuckDBSchemaRequest => db_duckdb_schema);
        dispatch!("db_query_file" => DuckDBFileQueryRequest => db_query_file);

        // ===== Visualization =====
        dispatch!("viz_histogram" => HistogramRequest => viz_histogram);
        dispatch!("viz_scatter" => ScatterPlotRequest => viz_scatter);
        dispatch!("viz_line" => LineChartRequest => viz_line);
        dispatch!("viz_boxplot" => BoxPlotRequest => viz_boxplot);
        dispatch!("viz_heatmap" => HeatmapRequest => viz_heatmap);
        dispatch!("viz_scatter_interactive" => ScatterInteractiveRequest => viz_scatter_interactive);
        dispatch!("viz_histogram_interactive" => HistogramInteractiveRequest => viz_histogram_interactive);
        dispatch!("viz_line_interactive" => LineInteractiveRequest => viz_line_interactive);
        dispatch!("viz_event_study" => EventStudyRequest => viz_event_study);
        dispatch!("viz_coefficient" => CoefficientPlotRequest => viz_coefficient);
        dispatch!("viz_irf" => IrfPlotRequest => viz_irf);
        dispatch!("viz_residual_diagnostics" => ResidualDiagnosticsRequest => viz_residual_diagnostics);
        dispatch!("viz_dendrogram" => DendrogramRequest => viz_dendrogram);

        // ===== Cleaning =====
        dispatch!("data_quality_profile" => DataQualityProfileRequest => data_quality_profile);
        dispatch!("preview_cleaning" => PreviewCleaningRequest => preview_cleaning);
        dispatch!("verify_cleaning" => VerifyCleaningRequest => verify_cleaning);
        dispatch!("cleaning_session_start" => CleaningSessionStartRequest => cleaning_session_start);
        dispatch!("cleaning_session_status" => CleaningSessionStatusRequest => cleaning_session_status);
        dispatch!("list_cleaning_sessions" => ListCleaningSessionsRequest => list_cleaning_sessions);
        dispatch!("cleaning_session_apply" => CleaningSessionApplyRequest => cleaning_session_apply);
        dispatch!("cleaning_rollback" => CleaningRollbackRequest => cleaning_rollback);
        dispatch!("cleaning_session_checkpoints" => CleaningSessionCheckpointsRequest => cleaning_session_checkpoints);
        dispatch!("suggest_cleaning" => SuggestCleaningRequest => suggest_cleaning);

        // ===== Machine Learning =====
        dispatch!("ml_kmeans" => KMeansRequest => ml_kmeans);
        dispatch!("ml_dbscan" => DBSCANRequest => ml_dbscan);
        dispatch!("ml_hierarchical" => HierarchicalRequest => ml_hierarchical);
        dispatch!("ml_cutree" => CutreeRequest => ml_cutree);
        dispatch!("ml_pca" => PCARequest => ml_pca);
        dispatch!("ml_tsne" => TsneRequest => ml_tsne);
        dispatch!("ml_cmdscale" => CmdscaleRequest => ml_cmdscale);
        dispatch!("ml_random_forest" => RandomForestRequest => ml_random_forest);
        dispatch!("ml_svm" => SvmRequest => ml_svm);
        dispatch!("ml_ppr" => PprRequest => ml_ppr);
        dispatch!("ml_c50" => C50Request => ml_c50);
        dispatch!("ml_causal_forest" => CausalForestRequest => ml_causal_forest);
        dispatch!("ml_bart_causal" => BartCausalRequest => ml_bart_causal);
        dispatch!("heterogeneity_test" => HetTxRequest => heterogeneity_test);
        dispatch!("ml_cubist" => CubistRequest => ml_cubist);
        dispatch!("ml_shap_values" => ShapValuesRequest => ml_shap_values);
        dispatch!("ml_ctree" => CtreeRequest => ml_ctree);
        dispatch!("ml_mboost" => MboostRequest => ml_mboost);

        // ===== Statistics =====
        dispatch!("stats_loglin" => LoglinRequest => stats_loglin);
        dispatch!("stats_model_tables" => ModelTablesRequest => stats_model_tables);
        dispatch!("stats_se_contrast" => SeContrastRequest => stats_se_contrast);
        dispatch!("stats_weighted_mean" => WeightedMeanRequest => stats_weighted_mean);
        dispatch!("stats_cov_wt" => CovWtRequest => stats_cov_wt);
        dispatch!("stats_mauchly_test" => MauchlyTestRequest => stats_mauchly_test);
        dispatch!("stats_fivenum" => FivenumRequest => stats_fivenum);
        dispatch!("stats_iqr" => IqrRequest => stats_iqr);
        dispatch!("stats_mad" => MadRequest => stats_mad);
        dispatch!("stats_ecdf" => EcdfRequest => stats_ecdf);
        dispatch!("stats_density" => DensityRequest => stats_density);
        dispatch!("stats_spline" => SplineRequest => stats_spline);
        dispatch!("stats_approx" => ApproxRequest => stats_approx);
        dispatch!("anova_manova" => ManovaRequest => anova_manova);
        dispatch!("anova_one_way" => OneWayAnovaRequest => anova_one_way);
        dispatch!("anova_tukey_hsd" => TukeyHsdRequest => anova_tukey_hsd);
        dispatch!("anova_two_way" => TwoWayAnovaRequest => anova_two_way);
        dispatch!("compute_correlation" => CorrelationRequest => compute_correlation);
        dispatch!("descriptive_isoreg" => IsoregRequest => descriptive_isoreg);
        dispatch!("descriptive_medpolish" => MedpolishRequest => descriptive_medpolish);
        dispatch!("multivariate_cancor" => CancorRequest => multivariate_cancor);
        dispatch!("multivariate_factanal" => FactorAnalysisRequest => multivariate_factanal);
        dispatch!("multivariate_mahalanobis" => MahalanobisRequest => multivariate_mahalanobis);
        dispatch!("power_anova_test" => PowerAnovaTestRequest => power_anova_test);
        dispatch!("power_prop_test" => PowerPropTestRequest => power_prop_test);
        dispatch!("power_t_test" => PowerTTestRequest => power_t_test);

        // ===== Hypothesis tests =====
        dispatch!("hypothesis_bartlett_test" => BartlettTestRequest => hypothesis_bartlett_test);
        dispatch!("hypothesis_chisq_gof" => ChiSquaredGofRequest => hypothesis_chisq_gof);
        dispatch!("hypothesis_chisq_independence" => ChiSquaredIndependenceRequest => hypothesis_chisq_independence);
        dispatch!("hypothesis_cor_test" => CorTestRequest => hypothesis_cor_test);
        dispatch!("hypothesis_fisher_exact" => FisherExactRequest => hypothesis_fisher_exact);
        dispatch!("hypothesis_friedman" => FriedmanTestRequest => hypothesis_friedman);
        dispatch!("hypothesis_kruskal_wallis" => KruskalWallisRequest => hypothesis_kruskal_wallis);
        dispatch!("hypothesis_ks_test" => KsTestRequest => hypothesis_ks_test);
        dispatch!("hypothesis_mantelhaen" => MantelhaenTestRequest => hypothesis_mantelhaen);
        dispatch!("hypothesis_mcnemar" => McnemarTestRequest => hypothesis_mcnemar);
        dispatch!("hypothesis_mood_test" => MoodTestRequest => hypothesis_mood_test);
        dispatch!("hypothesis_oneway" => OnewayTestRequest => hypothesis_oneway);
        dispatch!("hypothesis_pairwise_t_test" => PairwiseTTestRequest => hypothesis_pairwise_t_test);
        dispatch!("hypothesis_pairwise_wilcox" => PairwiseWilcoxRequest => hypothesis_pairwise_wilcox);
        dispatch!("hypothesis_poisson" => PoissonTestRequest => hypothesis_poisson);
        dispatch!("hypothesis_prop_trend_test" => PropTrendTestRequest => hypothesis_prop_trend_test);
        dispatch!("hypothesis_quade" => QuadeTestRequest => hypothesis_quade);
        dispatch!("hypothesis_shapiro_wilk" => ShapiroWilkRequest => hypothesis_shapiro_wilk);
        dispatch!("hypothesis_t_test" => TTestRequest => hypothesis_t_test);
        dispatch!("hypothesis_wilcoxon" => WilcoxonTestRequest => hypothesis_wilcoxon);

        // ===== Regression =====
        dispatch!("regression_ols" => OlsRequest => regression_ols);
        dispatch!("regression_diagnostics" => DiagnosticsRequest => regression_diagnostics);
        dispatch!("regression_bgtest" => BgTestRequest => regression_bgtest);
        dispatch!("regression_resettest" => ResetTestRequest => regression_resettest);
        dispatch!("regression_waldtest" => WaldTestRequest => regression_waldtest);
        dispatch!("regression_harvtest" => HarveyCollierRequest => regression_harvtest);
        dispatch!("regression_hac" => HacRequest => regression_hac);
        dispatch!("regression_bootstrap_cov" => BootstrapCovRequest => regression_bootstrap_cov);
        dispatch!("regression_driscoll_kraay" => DriscollKraayRequest => regression_driscoll_kraay);
        dispatch!("regression_quantreg" => QuantRegRequest => regression_quantreg);
        dispatch!("regression_clustered" => OlsClusteredRequest => regression_clustered);
        dispatch!("regression_nls" => NlsRequest => regression_nls);
        dispatch!("regression_loess" => LoessRequest => regression_loess);
        dispatch!("regression_supsmu" => SupsmuRequest => regression_supsmu);
        dispatch!("regression_line" => LineRequest => regression_line);
        dispatch!("regression_step" => StepRequest => regression_step);
        dispatch!("regression_gls" => GlsRequest => regression_gls);
        dispatch!("regression_smooth_spline" => SmoothSplineRequest => regression_smooth_spline);
        dispatch!("regression_glmnet" => GlmnetRequest => regression_glmnet);
        dispatch!("regression_cv_glmnet" => CvGlmnetRequest => regression_cv_glmnet);
        dispatch!("regression_ridge" => RidgeRequest => regression_ridge);
        dispatch!("regression_lasso" => LassoRequest => regression_lasso);

        // ===== Panel data =====
        dispatch!("panel_fixed_effects" => PanelFERequest => panel_fixed_effects);
        dispatch!("panel_random_effects" => PanelRERequest => panel_random_effects);
        dispatch!("hausman_test" => HausmanRequest => hausman_test);
        dispatch!("panel_pvcm" => PvcmRequest => panel_pvcm);
        dispatch!("panel_pmg" => PvcmRequest => panel_pmg);
        dispatch!("panel_gmm" => GmmRequest => panel_gmm);
        dispatch!("panel_gls" => PanelGlsRequest => panel_gls);
        dispatch!("panel_unit_root" => PanelUnitRootRequest => panel_unit_root);
        dispatch!("panel_hdfe" => PanelHdfeRequest => panel_hdfe);

        // ===== Discrete choice =====
        dispatch!("logit" => LogitRequest => logit);
        dispatch!("probit" => ProbitRequest => probit);
        dispatch!("multinom" => MultinomRequest => multinom);
        dispatch!("mlogit" => MlogitRequest => mlogit);
        dispatch!("mixed_logit" => MixedLogitRequest => mixed_logit);
        dispatch!("ordered_model" => OrderedRequest => ordered_model);
        dispatch!("negbin" => NegBinRequest => negbin);
        dispatch!("zeroinfl" => ZeroInflRequest => zeroinfl);
        dispatch!("hurdle_model" => HurdleModelRequest => hurdle_model);
        dispatch!("feglm" => FeglmRequest => feglm);

        // ===== Causal inference =====
        dispatch!("iv_2sls" => IV2SLSRequest => iv_2sls);
        dispatch!("iv_first_stage" => FirstStageRequest => iv_first_stage);
        dispatch!("iv_sargan_test" => SarganTestRequest => iv_sargan_test);
        dispatch!("bp_bounds" => BPBoundsRequest => bp_bounds);
        dispatch!("iv_mte" => IVMTERequest => iv_mte);
        dispatch!("staggered_did" => StaggeredDiDRequest => staggered_did);
        dispatch!("bacon_decomp" => BaconDecompRequest => bacon_decomp);
        dispatch!("etwfe" => EtwfeRequest => etwfe);
        dispatch!("treatment_ipw" => IpwRequest => treatment_ipw);
        dispatch!("treatment_doubly_robust" => DoublyRobustRequest => treatment_doubly_robust);
        dispatch!("treatment_double_ml" => DoubleMLRequest => treatment_double_ml);
        dispatch!("treatment_cbps" => CbpsRequest => treatment_cbps);
        dispatch!("treatment_weightit" => WeightItRequest => treatment_weightit);
        dispatch!("treatment_entropy_balance" => EntropyBalanceRequest => treatment_entropy_balance);
        dispatch!("treatment_sbw" => SBWRequest => treatment_sbw);
        dispatch!("treatment_twang" => TwangRequest => treatment_twang);
        dispatch!("propensity_matching" => MatchItRequest => propensity_matching);
        dispatch!("treatment_tmle" => TmleRequest => treatment_tmle);
        dispatch!("collaborative_tmle" => CTmleRequest => collaborative_tmle);
        dispatch!("gformula" => GFormulaRequest => gformula);
        dispatch!("mediation_analysis" => MediationRequest => mediation_analysis);
        dispatch!("natural_effects_mediation" => NaturalEffectsRequest => natural_effects_mediation);
        dispatch!("synthetic_control" => SyntheticControlRequest => synthetic_control);
        dispatch!("gsynth" => GsynthRequest => gsynth);
        dispatch!("scpi" => ScpiRequest => scpi);
        dispatch!("rd_estimate" => RdEstimateRequest => rd_estimate);
        dispatch!("rd_bw" => RdBandwidthRequest => rd_bw);
        dispatch!("rd_fuzzy" => FuzzyRdRequest => rd_fuzzy);
        dispatch!("rd_multi" => RdMultiRequest => rd_multi);
        dispatch!("diff_in_diff" => DiDRequest => diff_in_diff);
        dispatch!("evalue" => EValueRequest => evalue);
        dispatch!("gmm_iv" => GeneralGmmIvRequest => gmm_iv);
        dispatch!("ltmle" => LtmleRequest => ltmle);
        dispatch!("marginal_effects" => MarginalEffectsRequest => marginal_effects);
        dispatch!("regression_standardization" => StdRegRequest => regression_standardization);
        dispatch!("sensemakr" => SensemakrRequest => sensemakr);

        // ===== Time series =====
        dispatch!("timeseries_acf" => AcfRequest => timeseries_acf);
        dispatch!("timeseries_ccf" => CcfRequest => timeseries_ccf);
        dispatch!("timeseries_spectrum" => SpectrumRequest => timeseries_spectrum);
        dispatch!("timeseries_box_test" => BoxTestRequest => timeseries_box_test);
        dispatch!("timeseries_pp_test" => PPTestRequest => timeseries_pp_test);
        dispatch!("ts_var" => VarRequest => ts_var);
        dispatch!("ts_granger" => GrangerRequest => ts_granger);
        dispatch!("ts_varma" => VarmaRequest => ts_varma);
        dispatch!("ts_vecm" => VecmRequest => ts_vecm);
        dispatch!("ts_var_irf" => VarIrfRequest => ts_var_irf);
        dispatch!("ts_arima_fit" => ArimaRequest => ts_arima_fit);
        dispatch!("ts_arima_forecast" => ArimaForecastRequest => ts_arima_forecast);
        dispatch!("ts_garch_fit" => GarchRequest => ts_garch_fit);
        dispatch!("ts_mstl" => MstlRequest => ts_mstl);
        dispatch!("ts_changepoint" => ChangepointRequest => ts_changepoint);
        dispatch!("ts_holt_winters" => HoltWintersRequest => ts_holt_winters);
        dispatch!("timeseries_ar" => ArModelRequest => timeseries_ar);
        dispatch!("timeseries_decompose" => DecomposeRequest => timeseries_decompose);
        dispatch!("timeseries_structts" => StructTsRequest => timeseries_structts);
        dispatch!("causal_impact_analysis" => CausalImpactRequest => causal_impact_analysis);
        dispatch!("timeseries_cpgram" => CpgramRequest => timeseries_cpgram);
        dispatch!("linalg_toeplitz" => ToeplitzRequest => linalg_toeplitz);
        dispatch!("timeseries_lag" => LagRequest => timeseries_lag);
        dispatch!("timeseries_embed" => EmbedRequest => timeseries_embed);
        dispatch!("timeseries_diffinv" => DiffinvRequest => timeseries_diffinv);
        dispatch!("timeseries_filter" => FilterRequest => timeseries_filter);
        dispatch!("timeseries_window" => WindowRequest => timeseries_window);
        dispatch!("timeseries_arma_acf" => ArmaAcfRequest => timeseries_arma_acf);
        dispatch!("timeseries_arma_to_ma" => ArmaToMaRequest => timeseries_arma_to_ma);
        dispatch!("timeseries_acf_to_ar" => Acf2ArRequest => timeseries_acf_to_ar);
        dispatch!("timeseries_arima_sim" => ArimaSimRequest => timeseries_arima_sim);
        dispatch!("timeseries_runmed" => RunmedRequest => timeseries_runmed);

        // ===== Spatial =====
        dispatch!("spatial_neighbors" => SpatialNeighborsRequest => spatial_neighbors);
        dispatch!("moran_test" => MoranTestRequest => moran_test);
        dispatch!("spatial_lm_tests_tool" => SpatialLmTestRequest => spatial_lm_tests_tool);
        dispatch!("sar_model" => SarModelRequest => sar_model);
        dispatch!("sem_model" => SemModelRequest => sem_model);
        dispatch!("sphet_model" => SphetRequest => sphet_model);
        dispatch!("sar_probit_model" => SarProbitRequest => sar_probit_model);
        dispatch!("sem_probit_model" => SemProbitRequest => sem_probit_model);
        dispatch!("spatial_panel_ml" => SpmlRequest => spatial_panel_ml);
        dispatch!("spatial_panel_gmm" => SpgmRequest => spatial_panel_gmm);

        // ===== Data munging =====
        dispatch!("batch_process" => BatchProcessRequest => batch_process);
        dispatch!("compare_datasets" => CompareDatasetRequest => compare_datasets);
        dispatch!("munge_filter" => FilterDatasetRequest => munge_filter);
        dispatch!("munge_select" => SelectColumnsRequest => munge_select);
        dispatch!("munge_drop_columns" => DropColumnsRequest => munge_drop_columns);
        dispatch!("munge_rename" => RenameColumnsRequest => munge_rename);
        dispatch!("munge_sort" => SortDatasetRequest => munge_sort);
        dispatch!("munge_join" => JoinDatasetsRequest => munge_join);
        dispatch!("munge_concat" => ConcatDatasetsRequest => munge_concat);
        dispatch!("munge_group_by" => GroupByRequest => munge_group_by);
        dispatch!("munge_value_counts" => ValueCountsRequest => munge_value_counts);
        dispatch!("munge_pivot" => PivotDatasetRequest => munge_pivot);
        dispatch!("munge_melt" => MeltDatasetRequest => munge_melt);
        dispatch!("munge_drop_na" => DropNaRequest => munge_drop_na);
        dispatch!("munge_fill_na" => FillNaRequest => munge_fill_na);
        dispatch!("munge_deduplicate" => DeduplicateRequest => munge_deduplicate);
        dispatch!("str_trim" => TrimRequest => str_trim);
        dispatch!("str_to_lowercase" => ToLowercaseRequest => str_to_lowercase);
        dispatch!("str_to_uppercase" => ToUppercaseRequest => str_to_uppercase);
        dispatch!("str_replace_value" => ReplaceValueRequest => str_replace_value);
        dispatch!("str_regex_replace" => RegexReplaceRequest => str_regex_replace);
        dispatch!("str_regex_extract" => RegexExtractRequest => str_regex_extract);
        dispatch!("str_regex_count" => RegexCountRequest => str_regex_count);
        dispatch!("str_split" => StrSplitRequest => str_split);
        dispatch!("str_concat" => StrConcatRequest => str_concat);
        dispatch!("str_length" => StrLengthRequest => str_length);
        dispatch!("str_substring" => StrSubstringRequest => str_substring);
        dispatch!("munge_lag_lead" => LagLeadRequest => munge_lag_lead);
        dispatch!("munge_standardize" => StandardizeRequest => munge_standardize);
        dispatch!("munge_bin" => BinColumnRequest => munge_bin);
        dispatch!("munge_one_hot_encode" => OneHotEncodeRequest => munge_one_hot_encode);
        dispatch!("munge_diff" => DiffRequest => munge_diff);
        dispatch!("munge_sample" => SampleDatasetRequest => munge_sample);
        dispatch!("munge_mutate" => MutateColumnRequest => munge_mutate);

        // ===== Survival =====
        dispatch!("kaplan_meier" => KaplanMeierRequest => kaplan_meier);
        dispatch!("log_rank" => LogRankRequest => log_rank);
        dispatch!("cox_ph" => CoxPhRequest => cox_ph);
        dispatch!("aft" => AftRequest => aft);
        dispatch!("competing_risks" => CompetingRisksRequest => competing_risks);

        // ===== Search/discovery =====
        dispatch!("search_tools" => SearchToolsRequest => search_tools);
        dispatch!("list_tool_categories" => ListToolCategoriesRequest => list_tool_categories);
        dispatch!("tool_info" => crate::tools::handlers::search::ToolInfoRequest => tool_info);

        // If no tool matched, return an error
        Err(format!("Unknown tool: {}", tool_name))
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
