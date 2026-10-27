//! Analytics MCP Server implementation.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

use rmcp::{
    ErrorData as McpError, RoleServer, ServerHandler, handler::server::router::tool::ToolRouter,
    handler::server::tool::ToolCallContext, handler::server::wrapper::Parameters, model::*,
    service::RequestContext, tool, tool_router,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::tools::limits::{LimitViolation, Limits, PIVOT_RULES, ROW_RULES};
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
    /// Audit log for the stdio transport's tool calls (`None`: not audited
    /// here; governed HTTP-side calls are audited by the governor).
    audit: Option<Arc<crate::audit::AuditLogger>>,
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
            audit: None,
        }
    }

    /// Audit this server's MCP tool calls (the stdio transport) to `logger`:
    /// one entry per call with origin and session id `stdio`.
    pub fn with_audit_logger(mut self, logger: Arc<crate::audit::AuditLogger>) -> Self {
        self.audit = Some(logger);
        self
    }

    /// The parameters `tool` declares, for the audit log
    /// (`audit::sanitize_arguments`): the top-level names, the `properties`
    /// of its input schema (empty for an unknown tool), which the log takes
    /// ahead of the keys a tool ignores, and the nested structs its request
    /// limits read fields of ([`crate::tools::limits::struct_paths`]), which
    /// the log replaces by their size when they come as JSON arrays.
    pub fn declared_parameters(&self, tool: &str) -> crate::audit::DeclaredParameters<'_> {
        let names = self
            .tool_router
            .map
            .get(tool)
            .and_then(|route| route.attr.input_schema.get("properties"))
            .and_then(serde_json::Value::as_object)
            .map(|properties| properties.keys().map(String::as_str).collect())
            .unwrap_or_default();
        crate::audit::DeclaredParameters {
            names,
            structs: crate::tools::limits::struct_paths(tool),
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
            audit: None,
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
    ///
    /// Within this crate only `exec::ToolGovernor::call` calls this, so every
    /// HTTP-side tool call runs under the governor. A direct call (integration
    /// tests, embedders) runs on the caller's task and skips everything the
    /// governor adds: memory admission, the session and global permits, the
    /// blocking pool, the deadline and cancellation budget, panic isolation,
    /// the session directory scope (with session directories on, file tools
    /// are refused without it), the in-flight registry (`/health` and the
    /// stuck watchdog do not count it), the session's active-call mark (which
    /// keeps a running session from eviction and expiry), the `touch` on
    /// success, the dataset quotas, the open public process-memory redaction
    /// of `server_stats` and the audit entry. The request limits
    /// below still apply. Unknown names, unparsable arguments and handler
    /// errors come back as `ToolCallError`; a tool's own error result is an
    /// `Ok` with `success == false`.
    ///
    /// Before the arguments are deserialised, the call is checked against
    /// the request limits in force on this thread (`tools::limits::current`,
    /// which the governor sets on its blocking thread): the tool must exist
    /// (an unknown name is echoed cut, as it may be a WebSocket message's
    /// whole payload), its arguments must be a JSON object (`null`, arguments
    /// left out, is the empty object, as over stdio; anything else is
    /// `InvalidArguments` in every profile, because a derived request struct
    /// would also take a top-level JSON array by field position, past the
    /// limit checks, the row-class lookup and the audit redaction, which read
    /// parameters by name), then its cost parameters must be within the
    /// catalogue's caps (outside the off profile this also refuses, as
    /// `LimitExceeded`, a nested struct the catalogue has rules inside given
    /// as a JSON array; the audit entry of such a call is bounded by the
    /// sanitiser's size and depth caps, not redacted by name), then the
    /// dataset a `ROW_RULES` tool names must be within its row class, and the
    /// wide table a `PIVOT_RULES` tool would make of it within the pivot
    /// budgets. After
    /// the handler returns, output above the `response_bytes` budget
    /// replaces the result. A refusal is
    /// `ToolCallError::LimitExceeded` (422 `limit_exceeded`). The response
    /// cap comes after the work, so a call refused by it has made its
    /// changes (a created or modified dataset stays, unless the governor's
    /// dataset quota undoes it and answers with the quota refusal instead),
    /// and its message says so.
    #[cfg(feature = "http")]
    pub async fn call_tool_with_session(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
        session: &crate::session::Session,
    ) -> Result<crate::transport::http::ToolResult, crate::exec::ToolCallError> {
        use crate::exec::ToolCallError;

        if !self.tool_router.has_route(tool_name) {
            return Err(ToolCallError::UnknownTool(
                crate::tools::limits::echo(tool_name).to_string(),
            ));
        }
        let arguments = match arguments {
            serde_json::Value::Null => serde_json::Value::Object(serde_json::Map::new()),
            other => other,
        };
        if let Some(kind) = crate::tools::limits::non_object_kind(&arguments) {
            return Err(ToolCallError::InvalidArguments(
                crate::tools::limits::non_object_message(tool_name, kind),
            ));
        }
        let limits = crate::tools::limits::current();
        enforce_request_limits(&limits, tool_name, &arguments, &session.datasets)
            .await
            .map_err(log_and_convert)?;

        // A session-scoped server instance that shares the session's datasets.
        let session_server = Self::with_session(session);
        let result = session_server.dispatch_tool(tool_name, arguments).await?;
        let output_bytes = crate::audit::ResultSummary::of_tool_result(&result).output_bytes();
        check_response_size(&limits, tool_name, output_bytes).map_err(log_and_convert)?;
        Ok(result)
    }

    /// Deserialise `arguments` into the tool's request type and run its
    /// handler on `self` (a session-scoped server). A name without an arm
    /// is `UnknownTool`.
    #[cfg(feature = "http")]
    async fn dispatch_tool(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<crate::transport::http::ToolResult, crate::exec::ToolCallError> {
        use crate::exec::ToolCallError;
        use crate::transport::http::{ContentItem, ToolResult};

        let session_server = self;

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
        // serde's message quotes a mistyped value in full, so the refusal
        // gives it cut (`quote_cut`).
        macro_rules! dispatch {
            ($name:literal => no_params => $method:ident) => {
                if tool_name == $name {
                    let result = session_server
                        .$method()
                        .await
                        .map_err(|e| ToolCallError::from_handler_error(tool_name, e))?;
                    return Ok(convert_result(result));
                }
            };
            ($name:literal => $req:ty => $method:ident) => {
                if tool_name == $name {
                    let request: $req = serde_json::from_value(arguments.clone()).map_err(|e| {
                        ToolCallError::InvalidArguments(format!(
                            "Failed to parse arguments for '{}': {}",
                            tool_name,
                            quote_cut(&e)
                        ))
                    })?;
                    let result = session_server
                        .$method(Parameters(request))
                        .await
                        .map_err(|e| ToolCallError::from_handler_error(tool_name, e))?;
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
        Err(ToolCallError::UnknownTool(
            crate::tools::limits::echo(tool_name).to_string(),
        ))
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
// Request limits (every entry path)
// ============================================================================

/// The request-limit checks that run before a tool call's arguments are
/// deserialised, shared by the governed HTTP-side path
/// (`call_tool_with_session`) and the stdio transport: the cost parameters
/// against the catalogue (`Limits::check_tool_arguments`), then, for the
/// tools in `ROW_RULES`, the height of the dataset the request names
/// (`Limits::check_rows`), and for the tools in `PIVOT_RULES`, the shape of
/// the wide table the request would make of it (`Limits::check_pivot`), both
/// skipped when limits are off. A missing dataset field or an unknown dataset
/// is left to the handler's usual error.
#[allow(
    clippy::result_large_err,
    reason = "LimitViolation is WP02's fixed public shape"
)]
async fn enforce_request_limits(
    limits: &Limits,
    tool: &str,
    args: &serde_json::Value,
    datasets: &RwLock<HashMap<String, Dataset>>,
) -> Result<(), LimitViolation> {
    limits.check_tool_arguments(tool, args)?;
    if limits.is_off() {
        return Ok(());
    }
    for rule in ROW_RULES.iter().filter(|r| r.tool == tool) {
        let Some(name) = args.get(rule.dataset_field).and_then(|v| v.as_str()) else {
            continue;
        };
        // The guard is dropped before any further await.
        let shape = datasets.read().await.get(name).map(|d| {
            let df = d.df();
            (df.height(), df.width())
        });
        if let Some((rows, cols)) = shape {
            limits.check_rows(tool, rows, cols)?;
        }
    }
    for rule in PIVOT_RULES.iter().filter(|r| r.tool == tool) {
        let Some(name) = args.get(rule.dataset_field).and_then(|v| v.as_str()) else {
            continue;
        };
        // An owned copy (the columns are shared), so the guard is dropped before the
        // distinct values are counted.
        let dataset = datasets.read().await.get(name).cloned();
        if let Some(dataset) = dataset {
            limits.check_pivot(rule, args, dataset.df())?;
        }
    }
    Ok(())
}

/// Refuses a tool result whose output (`bytes`, from
/// [`ResultSummary::output_bytes`](crate::audit::ResultSummary::output_bytes),
/// every content kind) is above the `response_bytes` budget (none when
/// limits are off).
#[allow(
    clippy::result_large_err,
    reason = "LimitViolation is WP02's fixed public shape"
)]
fn check_response_size(limits: &Limits, tool: &str, bytes: usize) -> Result<(), LimitViolation> {
    let Some(max) = limits.response_bytes else {
        return Ok(());
    };
    if u64::try_from(bytes).is_ok_and(|b| b <= max) {
        return Ok(());
    }
    Err(LimitViolation {
        tool: tool.to_string(),
        parameter: "response_bytes".to_string(),
        requested: bytes.to_string(),
        limit: max.to_string(),
        override_key: "response_bytes",
        // The handler has run: say so, so a caller does not retry a change
        // that was already made.
        message: format!(
            "`{tool}` ran, but its output is {bytes} bytes, above this server's limit of {max} \
             bytes per response; any change it made, for example to the session's datasets, was \
             kept. Ask for less output, for example fewer rows, columns or grid points."
        ),
    })
}

/// Logs a refusal with the knob that would raise the limit (`override_key`
/// is for operators; the caller sees only the message).
fn log_violation(v: &LimitViolation) {
    tracing::info!(
        tool = %v.tool,
        parameter = %v.parameter,
        requested = %v.requested,
        limit = %v.limit,
        override_key = v.override_key,
        "request refused by a resource limit"
    );
}

/// Logs a refusal and turns it into the stdio transport's JSON-RPC error:
/// `invalid_params` (-32602) whose data names the code, the parameter, the
/// requested value and the limit.
fn limit_error(v: LimitViolation) -> McpError {
    log_violation(&v);
    McpError::invalid_params(
        v.message,
        Some(serde_json::json!({
            "code": "limit_exceeded",
            "parameter": v.parameter,
            "requested": v.requested,
            "limit": v.limit,
        })),
    )
}

/// Logs a stdio tool call that panicked and turns it into its JSON-RPC
/// error: `internal_error` (-32603) with the governed path's text and data
/// code `internal`. The panic hook has already printed the panic's message
/// to stderr; the caller is not given it.
fn panic_error(tool: &str) -> McpError {
    let tool = crate::tools::limits::echo(tool);
    tracing::error!(tool, "tool call panicked");
    McpError::internal_error(
        format!("tool '{tool}' failed unexpectedly; the session is still usable"),
        Some(serde_json::json!({ "code": "internal" })),
    )
}

/// Runs `fut` to its end, or to a panic while it is polled, which comes
/// back as `Err` with the panic's payload, like
/// `futures::FutureExt::catch_unwind` (`futures` is an optional dependency
/// here, and the stdio transport is always built).
///
/// `AssertUnwindSafe` as on the governed path: a panicking call leaves this
/// server's state as it leaves a session's there. The tokio locks around the
/// datasets do not poison, and a held guard is released as the panic unwinds.
async fn catch_panic<F: std::future::Future>(fut: F) -> std::thread::Result<F::Output> {
    let mut fut = std::pin::pin!(fut);
    std::future::poll_fn(move |cx| {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| fut.as_mut().poll(cx))) {
            Ok(std::task::Poll::Ready(output)) => std::task::Poll::Ready(Ok(output)),
            Ok(std::task::Poll::Pending) => std::task::Poll::Pending,
            Err(panic) => std::task::Poll::Ready(Err(panic)),
        }
    })
    .await
}

/// Logs a refusal and turns it into the governed path's 422 error.
#[cfg(feature = "http")]
fn log_and_convert(v: LimitViolation) -> crate::exec::ToolCallError {
    log_violation(&v);
    crate::exec::ToolCallError::from(v)
}

/// The most of a parser's message that an error gives back, in bytes. serde
/// quotes a mistyped string or an unknown variant in full, and that string
/// is the caller's: as long as a request body or a websocket message.
#[cfg(feature = "http")]
pub(crate) const MAX_QUOTE_BYTES: usize = 256;

/// How many of [`MAX_QUOTE_BYTES`] a cut message keeps of its end, where
/// serde says what it expected and, for a parse error, where it stopped.
#[cfg(feature = "http")]
const QUOTE_TAIL_BYTES: usize = 96;

/// How many of [`MAX_QUOTE_BYTES`] a cut message keeps of its start.
#[cfg(feature = "http")]
const QUOTE_HEAD_BYTES: usize = MAX_QUOTE_BYTES - QUOTE_TAIL_BYTES;

/// The text of `message` when it is at most [`MAX_QUOTE_BYTES`] long.
/// A longer one gives its first [`QUOTE_HEAD_BYTES`] and its last
/// [`QUOTE_TAIL_BYTES`] (each cut at a char boundary, so possibly fewer)
/// around a note of how many bytes were left out. The text is written into
/// a buffer of that size, never built whole.
#[cfg(feature = "http")]
pub(crate) fn quote_cut(message: &dyn std::fmt::Display) -> String {
    let mut ends = QuoteEnds::default();
    // Writing to `QuoteEnds` never fails; a `Display` that fails anyway
    // leaves what it wrote before.
    let _ = std::fmt::write(&mut ends, format_args!("{message}"));
    ends.finish()
}

/// The longest end of `s` of at most `n` bytes that starts at a char
/// boundary.
#[cfg(feature = "http")]
fn last_bytes(s: &str, n: usize) -> &str {
    let mut start = s.len().saturating_sub(n);
    while !s.is_char_boundary(start) {
        start += 1;
    }
    &s[start..]
}

/// The `fmt::Write` behind [`quote_cut`]: keeps a message whole while it
/// fits in [`MAX_QUOTE_BYTES`], and then only its start and its end.
#[cfg(feature = "http")]
#[derive(Default)]
struct QuoteEnds {
    /// The message while it fits; its start once it did not.
    head: String,
    /// The last bytes written, once the message did not fit.
    tail: String,
    /// Whether the message did not fit.
    cut: bool,
    /// How many bytes were written in all.
    written: usize,
}

#[cfg(feature = "http")]
impl QuoteEnds {
    /// Append `s` to the tail, which keeps its last [`QUOTE_TAIL_BYTES`].
    fn push_tail(&mut self, s: &str) {
        if s.len() >= QUOTE_TAIL_BYTES {
            self.tail.clear();
            self.tail.push_str(last_bytes(s, QUOTE_TAIL_BYTES));
            return;
        }
        self.tail.push_str(s);
        let keep = last_bytes(&self.tail, QUOTE_TAIL_BYTES).len();
        let drop = self.tail.len() - keep;
        self.tail.drain(..drop);
    }

    fn finish(self) -> String {
        if !self.cut {
            return self.head;
        }
        let left_out = self.written - self.head.len() - self.tail.len();
        let unit = if left_out == 1 { "byte" } else { "bytes" };
        format!("{}[... {left_out} {unit} cut ...]{}", self.head, self.tail)
    }
}

#[cfg(feature = "http")]
impl std::fmt::Write for QuoteEnds {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.written = self.written.saturating_add(s.len());
        if self.cut {
            self.push_tail(s);
            return Ok(());
        }
        if self.head.len() + s.len() <= MAX_QUOTE_BYTES {
            self.head.push_str(s);
            return Ok(());
        }
        // It does not fit: keep the start in `head` and the end in `tail`.
        self.cut = true;
        if self.head.len() > QUOTE_HEAD_BYTES {
            let mut end = QUOTE_HEAD_BYTES;
            while !self.head.is_char_boundary(end) {
                end -= 1;
            }
            self.tail = self.head.split_off(end);
            self.push_tail(s);
        } else {
            let mut end = QUOTE_HEAD_BYTES - self.head.len();
            while !s.is_char_boundary(end) {
                end -= 1;
            }
            self.head.push_str(&s[..end]);
            self.push_tail(&s[end..]);
        }
        Ok(())
    }
}

// ============================================================================
// ServerHandler Implementation
// ============================================================================

/// The MCP handler of the stdio transport. Tool calls run the same request
/// limits as the governed HTTP-side path (arguments, dataset rows and pivot
/// shapes of this server's own datasets, response size) before and after
/// delegating to the tool router, and with `with_audit_logger` each call
/// leaves one audit entry
/// (origin and session id `stdio`). A call that panics is answered with the
/// JSON-RPC error -32603, data code `internal` (the governed path's 500
/// `internal`), and audited as `internal`; the server goes on serving.
/// Deadlines and dataset quotas are HTTP-only.
impl ServerHandler for AnalyticsServer {
    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParam>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        Ok(ListToolsResult::with_all_items(self.tool_router.list_all()))
    }

    async fn call_tool(
        &self,
        mut request: CallToolRequestParam,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let started = std::time::Instant::now();
        let tool = request.name.to_string();
        let known = self.tool_router.has_route(&tool);
        let limits = crate::tools::limits::current();

        // Look at the arguments as a JSON object without copying them (for
        // the limit check and the audit entry), then hand them back to the
        // request unchanged.
        let had_arguments = request.arguments.is_some();
        let args = serde_json::Value::Object(request.arguments.take().unwrap_or_default());
        let audit = self.audit.as_deref().filter(|a| a.is_enabled()).map(|a| {
            let declared = self.declared_parameters(&tool);
            (a, crate::audit::sanitize_arguments(&args, &declared))
        });
        // The error is paired with its audit outcome. A panic in the limit
        // checks, the handler or the size check becomes the `internal`
        // error, so the call is answered and audited like any other (rmcp
        // would drop the request's task without a reply).
        let outcome: Result<CallToolResult, (McpError, &'static str)> = catch_panic(async {
            let checked = if known {
                enforce_request_limits(&limits, &tool, &args, &self.datasets).await
            } else {
                Ok(())
            };
            if let (true, serde_json::Value::Object(map)) = (had_arguments, args) {
                request.arguments = Some(map);
            }
            checked.map_err(|v| (limit_error(v), "limit_exceeded"))?;
            // An unknown name gets the router's own error.
            let result = self
                .tool_router
                .call(ToolCallContext::new(self, request, context))
                .await
                .map_err(|e| {
                    let code = if !known {
                        "unknown_tool"
                    } else if e.code == ErrorCode::INVALID_PARAMS
                        || e.code == ErrorCode::INVALID_REQUEST
                    {
                        "invalid_arguments"
                    } else {
                        "internal"
                    };
                    (e, code)
                })?;
            let output_bytes = crate::audit::ResultSummary::of_call_result(&result).output_bytes();
            check_response_size(&limits, &tool, output_bytes)
                .map_err(|v| (limit_error(v), "limit_exceeded"))?;
            Ok(result)
        })
        .await
        .unwrap_or_else(|_panic| Err((panic_error(&tool), "internal")));

        if let Some((audit, arguments)) = audit {
            let (code, summary) = match &outcome {
                Ok(result) if result.is_error == Some(true) => (
                    "tool_error",
                    crate::audit::ResultSummary::of_call_result(result),
                ),
                Ok(result) => ("ok", crate::audit::ResultSummary::of_call_result(result)),
                Err((_, code)) => (*code, crate::audit::ResultSummary::default()),
            };
            let duration_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
            audit
                .log(crate::audit::AuditEntry::new(
                    "stdio",
                    "stdio",
                    &tool,
                    arguments,
                    code,
                    duration_ms,
                    summary,
                ))
                .await;
        }
        outcome.map_err(|(e, _)| e)
    }

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

    #[test]
    fn response_cap_counts_embedded_resources() {
        let limits = Limits::hosted()
            .with_overrides(&["response_bytes=200".to_string()])
            .unwrap();
        let result =
            CallToolResult::success(vec![Content::embedded_text("str:///t", "x".repeat(300))]);
        let bytes = crate::audit::ResultSummary::of_call_result(&result).output_bytes();
        let refused = check_response_size(&limits, "t", bytes);
        assert!(
            refused.is_err(),
            "300 bytes of resource text passed a 200-byte cap"
        );
        let v = refused.unwrap_err();
        assert_eq!(
            (v.parameter.as_str(), v.requested.as_str()),
            ("response_bytes", "300")
        );
    }

    #[test]
    fn response_cap_message_says_the_call_ran() {
        // The cap is checked after the handler ran: a state-changing tool
        // must not read as if nothing happened (a retry would then fail with
        // "already exists" or apply the change twice).
        let limits = Limits::hosted()
            .with_overrides(&["response_bytes=10".to_string()])
            .unwrap();
        let v = check_response_size(&limits, "create_dataset", 11).unwrap_err();
        assert!(
            v.message
                .starts_with("`create_dataset` ran, but its output is 11 bytes"),
            "{}",
            v.message
        );
        assert!(v.message.contains("was kept"), "{}", v.message);
        assert!(v.message.contains("Ask for less output"), "{}", v.message);
    }

    #[test]
    fn declared_parameters_are_the_schema_properties() {
        let server = AnalyticsServer::new();
        let mut sqlite = server.declared_parameters("db_sqlite_query");
        sqlite.names.sort_unstable();
        assert_eq!(sqlite.names, ["db_path", "name", "query"]);
        assert!(sqlite.structs.is_empty());
        assert_eq!(
            server.declared_parameters("no_such_tool"),
            crate::audit::DeclaredParameters::default()
        );
        // The structs are the ones the request limits read fields of.
        let report = server.declared_parameters("generate_report");
        assert_eq!(report.structs, ["sections[]", "sections[].content[]"]);
        let random = server.declared_parameters("generate_random_data");
        assert_eq!(random.structs, ["columns[]", "columns[].distribution"]);
        // No tool keeps a top-level parameter outside `properties`, where the
        // audit log would not see it as declared.
        for tool in server.tool_router.list_all() {
            let schema = &tool.input_schema;
            let hidden: Vec<&str> = ["$ref", "allOf", "anyOf", "oneOf"]
                .into_iter()
                .filter(|k| schema.contains_key(*k))
                .collect();
            assert!(hidden.is_empty(), "{}: {hidden:?}", tool.name);
        }
    }

    /// Top-level arguments that are not an object (a derived request struct
    /// would take an array by field position, past the checks that read
    /// parameters by name) are refused on the governed path in both
    /// profiles; `set_seed` must not run.
    #[cfg(feature = "http")]
    #[test]
    fn governed_path_refuses_arguments_that_are_not_an_object() {
        use crate::exec::ToolCallError;
        use crate::session::Session;

        for limits in [Limits::off(), Limits::hosted()] {
            // The limits are this thread's for the whole runtime.
            crate::tools::limits::with_limits(limits, || {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap()
                    .block_on(async {
                        let session = Session::new("shape".into(), None);
                        let server = AnalyticsServer::with_session(&session);
                        for (args, kind) in [
                            (serde_json::json!([42]), "an array"),
                            (serde_json::json!("42"), "a string"),
                            (serde_json::json!(42), "a number"),
                            (serde_json::json!(false), "a boolean"),
                        ] {
                            let refused = server
                                .call_tool_with_session("set_seed", args.clone(), &session)
                                .await;
                            match refused {
                                Err(ToolCallError::InvalidArguments(m)) => assert_eq!(
                                    m,
                                    format!(
                                        "The arguments of `set_seed` are {kind}, but tool \
                                         arguments must be a JSON object that names each \
                                         parameter. Send each value under its parameter name, \
                                         as the tool's input schema lists them."
                                    )
                                ),
                                other => panic!("{args}: {other:?}"),
                            }
                        }
                        assert_eq!(*session.global_seed.read().await, None, "set_seed ran");
                        // The object form runs.
                        let ran = server
                            .call_tool_with_session(
                                "set_seed",
                                serde_json::json!({"seed": 42}),
                                &session,
                            )
                            .await
                            .unwrap();
                        assert!(ran.success, "{ran:?}");
                        assert_eq!(*session.global_seed.read().await, Some(42));
                    });
            });
        }
    }

    /// Arguments left out (`null`) are the empty object, as over stdio: a
    /// tool whose parameters are all optional runs.
    #[cfg(feature = "http")]
    #[tokio::test]
    async fn governed_path_takes_absent_arguments_as_the_empty_object() {
        let session = crate::session::Session::new("absent".into(), None);
        let server = AnalyticsServer::with_session(&session);
        for tool in ["get_seed", "list_datasets"] {
            let r = server
                .call_tool_with_session(tool, serde_json::Value::Null, &session)
                .await
                .unwrap_or_else(|e| panic!("{tool}: {e:?}"));
            assert!(r.success, "{tool}: {r:?}");
        }
    }

    /// An unknown tool's name is the caller's string (up to the WebSocket
    /// message cap): the refusal echoes it cut.
    #[cfg(feature = "http")]
    #[tokio::test]
    async fn unknown_tool_refusal_echoes_a_bounded_name() {
        use crate::exec::ToolCallError;
        let session = crate::session::Session::new("unknown".into(), None);
        let server = AnalyticsServer::with_session(&session);
        let name = "x".repeat(1_000_000);
        let e = server
            .call_tool_with_session(&name, serde_json::json!({}), &session)
            .await
            .expect_err("an unknown tool ran");
        assert!(matches!(e, ToolCallError::UnknownTool(_)), "{e:?}");
        assert_eq!(e.to_string(), format!("Unknown tool: {}", "x".repeat(64)));
        let e = server
            .call_tool_with_session("no_such_tool", serde_json::json!({}), &session)
            .await
            .expect_err("an unknown tool ran");
        assert_eq!(e.to_string(), "Unknown tool: no_such_tool");
    }

    /// Splits what `quote_cut` made of a cut message into its start, the
    /// number of bytes it left out and its end.
    #[cfg(feature = "http")]
    fn quote_parts(quoted: &str) -> (&str, usize, &str) {
        let (head, rest) = quoted.split_once("[... ").expect("not cut");
        let (count, tail) = rest.split_once(" cut ...]").expect("not cut");
        let count = count.trim_end_matches(" bytes").trim_end_matches(" byte");
        (head, count.parse().unwrap(), tail)
    }

    /// A message of at most `MAX_QUOTE_BYTES` is given whole; a longer one
    /// gives its start and its end and how many bytes it left out.
    #[cfg(feature = "http")]
    #[test]
    fn quote_cut_keeps_the_two_ends_of_a_long_message() {
        let fits = "a".repeat(MAX_QUOTE_BYTES);
        assert_eq!(quote_cut(&fits), fits);
        let over = format!("{fits}b");
        assert_eq!(
            quote_cut(&over),
            format!(
                "{}[... 1 byte cut ...]{}b",
                "a".repeat(QUOTE_HEAD_BYTES),
                "a".repeat(QUOTE_TAIL_BYTES - 1)
            )
        );
        let long = format!("start{}end", "m".repeat(100_000));
        let quoted = quote_cut(&long);
        let (head, left_out, tail) = quote_parts(&quoted);
        assert_eq!(head.len(), QUOTE_HEAD_BYTES);
        assert_eq!(tail.len(), QUOTE_TAIL_BYTES);
        assert!(
            head.starts_with("start") && tail.ends_with("end"),
            "{quoted}"
        );
        assert_eq!(left_out, long.len() - MAX_QUOTE_BYTES);
    }

    /// Both ends are cut at char boundaries (here inside a three-byte
    /// character), so they may keep fewer bytes; what is left out is
    /// counted exactly.
    #[cfg(feature = "http")]
    #[test]
    fn quote_cut_cuts_at_char_boundaries() {
        let long = format!("xx{}", "€".repeat(200));
        let quoted = quote_cut(&long);
        let (head, left_out, tail) = quote_parts(&quoted);
        assert!(long.starts_with(head) && long.ends_with(tail), "{quoted}");
        assert!(head.len() < QUOTE_HEAD_BYTES, "{} bytes", head.len());
        assert!(tail.len() <= QUOTE_TAIL_BYTES, "{} bytes", tail.len());
        assert_eq!(head.len() + left_out + tail.len(), long.len());
    }

    /// A `Display` that writes its text in pieces of `self.1` characters.
    #[cfg(feature = "http")]
    struct InPieces<'a>(&'a str, usize);

    #[cfg(feature = "http")]
    impl std::fmt::Display for InPieces<'_> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let chars: Vec<char> = self.0.chars().collect();
            for piece in chars.chunks(self.1) {
                f.write_str(&piece.iter().collect::<String>())?;
            }
            Ok(())
        }
    }

    /// The cut does not depend on how the message is written: one piece,
    /// or many that fill the start and then the end.
    #[cfg(feature = "http")]
    #[test]
    fn quote_cut_is_the_same_however_the_message_is_written() {
        for text in [
            "a".repeat(MAX_QUOTE_BYTES),
            format!("é{}", "a".repeat(MAX_QUOTE_BYTES)),
            format!("head {} tail", "€é".repeat(1_000)),
        ] {
            let whole = quote_cut(&text);
            for piece in [1, 2, 7, 50, 97, 300, 10_000] {
                assert_eq!(
                    quote_cut(&InPieces(&text, piece)),
                    whole,
                    "pieces of {piece}"
                );
            }
        }
    }

    /// serde quotes a mistyped string argument in full: the refusal gives it
    /// cut, and keeps the end, where serde says what it expected.
    #[cfg(feature = "http")]
    #[tokio::test]
    async fn an_argument_parse_error_quotes_the_value_cut() {
        use crate::exec::ToolCallError;
        let session = crate::session::Session::new("parse".into(), None);
        let server = AnalyticsServer::with_session(&session);
        let seed = "7".repeat(100_000);
        let e = server
            .call_tool_with_session("set_seed", serde_json::json!({ "seed": seed }), &session)
            .await
            .expect_err("a string seed ran");
        let ToolCallError::InvalidArguments(m) = &e else {
            panic!("{e:?}")
        };
        assert!(
            m.starts_with("Failed to parse arguments for 'set_seed': invalid type: string \"777"),
            "{m}"
        );
        assert!(m.contains(" bytes cut ...]"), "{m}");
        assert!(m.ends_with("777\", expected u64"), "{m}");
        assert!(m.len() < 2 * MAX_QUOTE_BYTES, "{} bytes", m.len());
        assert_eq!(*session.global_seed.read().await, None, "set_seed ran");
        // A short one is given whole.
        let e = server
            .call_tool_with_session("set_seed", serde_json::json!({"seed": "seven"}), &session)
            .await
            .expect_err("a string seed ran");
        assert_eq!(
            e.to_string(),
            "Failed to parse arguments for 'set_seed': invalid type: string \"seven\", \
             expected u64"
        );
    }

    // =========================================================================
    // stdio transport: a tool call that panics
    // =========================================================================

    type StdioLines =
        tokio::io::Lines<tokio::io::BufReader<tokio::io::ReadHalf<tokio::io::DuplexStream>>>;
    type StdioWriter = tokio::io::WriteHalf<tokio::io::DuplexStream>;

    /// The name of the test route whose handler panics.
    const PANICKING_TOOL: &str = "test_handler_panics";

    /// The test route's handler, which panics as a tool with a bug does.
    async fn handler_that_panics() -> Result<CallToolResult, McpError> {
        panic!("a bug in a tool handler")
    }

    fn call_handler_that_panics(
        _context: ToolCallContext<'_, AnalyticsServer>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<CallToolResult, McpError>> + Send + '_>,
    > {
        Box::pin(handler_that_panics())
    }

    /// A route for [`PANICKING_TOOL`], standing in for any handler panic not
    /// fixed yet, so the test does not depend on one.
    fn panicking_route() -> rmcp::handler::server::router::tool::ToolRoute<AnalyticsServer> {
        let mut schema = serde_json::Map::new();
        schema.insert("type".into(), "object".into());
        schema.insert("properties".into(), serde_json::json!({}));
        rmcp::handler::server::router::tool::ToolRoute::new_dyn(
            Tool::new(PANICKING_TOOL, "Panics.", Arc::new(schema)),
            call_handler_that_panics,
        )
    }

    async fn send_line(w: &mut StdioWriter, msg: serde_json::Value) {
        use tokio::io::AsyncWriteExt;
        w.write_all(format!("{msg}\n").as_bytes()).await.unwrap();
        w.flush().await.unwrap();
    }

    async fn next_message(lines: &mut StdioLines) -> serde_json::Value {
        let line = lines
            .next_line()
            .await
            .unwrap()
            .expect("the server closed the stream");
        serde_json::from_str(&line).unwrap()
    }

    /// Serves `server` over an in-memory stream, as the stdio transport does
    /// over stdin/stdout, and initialises the MCP session.
    async fn stdio_client(
        server: AnalyticsServer,
    ) -> (StdioLines, StdioWriter, tokio::task::JoinHandle<()>) {
        use rmcp::ServiceExt;
        use tokio::io::AsyncBufReadExt;
        let (client, server_end) = tokio::io::duplex(1 << 20);
        let service = tokio::spawn(async move {
            let running = server
                .serve(tokio::io::split(server_end))
                .await
                .expect("serve");
            let _ = running.waiting().await;
        });
        let (r, mut w) = tokio::io::split(client);
        let mut lines = tokio::io::BufReader::new(r).lines();
        send_line(
            &mut w,
            serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "server-tests", "version": "0"}}}),
        )
        .await;
        while next_message(&mut lines).await["id"] != 1 {}
        send_line(
            &mut w,
            serde_json::json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        )
        .await;
        (lines, w, service)
    }

    /// How many ping round trips the reply to a request may come after.
    /// A reply that comes at all comes within the first: rmcp runs each
    /// request as a task on the test's current_thread runtime, and the calls
    /// here wait on nothing outside it. Counting round trips, not seconds,
    /// keeps the test independent of the machine's speed.
    const ROUND_TRIPS: u64 = 20;

    /// The reply to request `id`, or `None` when it has not come after
    /// [`ROUND_TRIPS`] pings sent after it were answered.
    async fn reply_to(
        lines: &mut StdioLines,
        w: &mut StdioWriter,
        id: u64,
    ) -> Option<serde_json::Value> {
        let first_ping = 1_000 * id;
        let mut ping = first_ping;
        send_line(
            w,
            serde_json::json!({"jsonrpc": "2.0", "id": ping, "method": "ping"}),
        )
        .await;
        loop {
            let msg = next_message(lines).await;
            if msg["id"] == id {
                return Some(msg);
            }
            if msg["id"] == ping {
                if ping - first_ping + 1 == ROUND_TRIPS {
                    return None;
                }
                ping += 1;
                send_line(
                    w,
                    serde_json::json!({"jsonrpc": "2.0", "id": ping, "method": "ping"}),
                )
                .await;
            }
        }
    }

    /// A tool call whose handler panics gets a JSON-RPC error over stdio, as
    /// the governed path answers 500 `internal`, and leaves one audit entry
    /// with outcome `internal`, in both profiles. The server goes on
    /// answering.
    #[test]
    fn stdio_answers_and_audits_a_tool_call_that_panics() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = crate::config::AuditConfig {
            enabled: true,
            path: dir.path().join("audit.log").to_string_lossy().into_owned(),
        };
        // This handle keeps the writer alive until the log is read.
        let logger = Arc::new(crate::audit::AuditLogger::new(&cfg).unwrap());
        let runtime = || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
        };
        for limits in [Limits::off(), Limits::hosted()] {
            crate::tools::limits::with_limits(limits, || {
                runtime().block_on(async {
                    let mut server = AnalyticsServer::new().with_audit_logger(logger.clone());
                    server.tool_router.add_route(panicking_route());
                    let (mut lines, mut w, service) = stdio_client(server).await;
                    send_line(
                        &mut w,
                        serde_json::json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                            "params": {"name": PANICKING_TOOL, "arguments": {}}}),
                    )
                    .await;
                    let reply = reply_to(&mut lines, &mut w, 2)
                        .await
                        .expect("the call that panicked got no reply");
                    assert_eq!(reply["error"]["code"], -32603, "{reply}");
                    assert_eq!(
                        reply["error"]["message"],
                        format!(
                            "tool '{PANICKING_TOOL}' failed unexpectedly; the session is still \
                             usable"
                        ),
                        "{reply}"
                    );
                    assert_eq!(reply["error"]["data"]["code"], "internal", "{reply}");
                    // The next call is answered.
                    send_line(
                        &mut w,
                        serde_json::json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                            "params": {"name": "list_datasets", "arguments": {}}}),
                    )
                    .await;
                    let next = reply_to(&mut lines, &mut w, 3)
                        .await
                        .expect("the call after the panic got no reply");
                    assert!(next.get("error").is_none(), "{next}");
                    assert_ne!(next["result"]["isError"], true, "{next}");
                    drop(w);
                    service.abort();
                });
            });
        }
        runtime().block_on(logger.flush());
        let text = std::fs::read_to_string(&cfg.path).unwrap();
        let entries: Vec<serde_json::Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .filter(|e: &serde_json::Value| e["tool_name"] == PANICKING_TOOL)
            .collect();
        assert_eq!(entries.len(), 2, "{text}");
        for entry in entries {
            assert_eq!(entry["origin"], "stdio", "{entry}");
            assert_eq!(entry["session_id"], "stdio", "{entry}");
            assert_eq!(entry["outcome"], "internal", "{entry}");
            assert_eq!(entry["success"], false, "{entry}");
        }
        drop(logger);
    }

    /// A handler can suspend before it panics or ends (`server_stats` waits
    /// on the datasets lock first): `catch_panic` catches a panic on a later
    /// poll and passes on an output that comes after a suspension.
    /// `yield_now` suspends once, so the second poll is the one that panics
    /// or ends.
    #[tokio::test]
    async fn catch_panic_covers_the_polls_after_a_suspension() {
        let panicked = catch_panic(async {
            tokio::task::yield_now().await;
            panic!("after a suspension")
        })
        .await;
        let payload = panicked.expect_err("a panic after a suspension came back as Ok");
        assert_eq!(
            payload.downcast_ref::<&str>(),
            Some(&"after a suspension"),
            "the payload is not the panic's"
        );
        let ended = catch_panic(async {
            tokio::task::yield_now().await;
            7
        })
        .await;
        assert_eq!(ended.ok(), Some(7));
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
