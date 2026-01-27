//! # p2a-core
//!
//! Core analytics engine for prompt2analytics.
//!
//! This crate provides the data loading, statistical analysis, and machine learning
//! functionality that powers the MCP server.
//!
//! See the [crate README](https://github.com/umatter/prompt2analytics/tree/main/crates/p2a-core)
//! for a complete overview of features and usage examples.

// Enforce documentation on public items (warn, not error, for gradual adoption)
#![warn(missing_docs)]

// Foundation modules (pure Rust implementations)
pub mod linalg;
pub mod traits;
pub mod errors;

// Core feature modules (always available)
pub mod data;
pub mod stats;
pub mod regression;
pub mod econometrics;
pub mod spatial;
pub mod ml;
pub mod reports;
pub mod simulation;
pub mod export;

// Optional feature modules
#[cfg(feature = "forecasting")]
pub mod forecasting;

#[cfg(feature = "visualization")]
pub mod visualization;

// Re-export foundational types
pub use errors::{EconError, EconResult, EstimationWarning};
pub use traits::{LinearEstimator, SignificanceLevel};
pub use linalg::{
    DesignMatrix, DesignError,
    // Toeplitz matrix construction
    toeplitz, toeplitz_asymmetric, toeplitz2, toeplitz_acf, toeplitz_to_vec,
};

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
    // Weighted statistics
    weighted_mean, run_weighted_mean,
    cov_wt, cov_wt_from_slice, run_cov_wt,
    CovWtResult, CovWtMethod, CovWtCenter,
    // Mauchly's sphericity test
    mauchly_test, mauchly_test_from_slice, run_mauchly_test,
    MauchlyResult,
    // Median polish
    medpolish, medpolish_array, run_medpolish,
    MedpolishResult,
    // Isotonic regression
    isoreg, isoreg_y, isoreg_predict, run_isoreg,
    IsoregResult,
    // Log-linear models
    loglin, loglin_independence, loglin_saturated, run_loglin,
    LoglinResult,
    // Constrained optimization
    constr_optim, run_constr_optim,
    ConstrOptimResult, ConstrOptimConfig, OptimMethod,
    // Standard errors for contrasts
    se_contrast, se_contrast_single, estimate_contrast,
    contrast_t_statistic, contrast_p_value, generate_contrasts, run_se_contrast,
    SeContrastResult, ContrastType,
    // Model tables
    model_tables, model_tables_means, model_tables_effects,
    model_tables_two_way, run_model_tables, format_model_tables,
    ModelTablesResult, TwoWayModelTablesResult, ModelTablesSE, TableType,
};

// Spectral density estimation (requires spectral-analysis feature)
#[cfg(feature = "spectral-analysis")]
pub use stats::{
    SpectrumConfig, SpectrumResult, spectrum, spectrum_ar, run_spectrum, run_spectrum_ar,
};
pub use regression::{
    OlsResult, run_ols, run_ols_raw, run_ols_clustered, DiagnosticsResult, run_diagnostics,
    // Breusch-Godfrey test for serial correlation
    BgTestResult, BgTestType, bg_test, run_bg_test, bg_test_from_ols,
    // Ramsey's RESET test for functional form
    ResetTestResult, ResetType, reset_test, run_reset_test, reset_test_from_ols,
    // Wald test for comparing nested models
    WaldTestResult, wald_test, run_wald_test, wald_test_from_ols,
    // Harvey-Collier test for linearity
    HarveyCollierResult, harvey_collier_test, run_harvey_collier, recursive_residuals,
    // HAC (Newey-West) standard errors
    HacResult, HacKernel, vcov_hac, run_vcov_hac,
    // Bootstrap covariance estimation
    BootstrapResult, BootstrapType, vcov_bootstrap, run_vcov_bootstrap,
    // Driscoll-Kraay panel-robust standard errors
    DriscollKraayResult, vcov_driscoll_kraay, run_vcov_driscoll_kraay,
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
    // Stepwise model selection
    StepResult, StepRecord, StepConfig, StepDirection,
    Add1Result, Drop1Result, TermEvaluation,
    step, run_step, add1, drop1,
    // Tukey's resistant line
    LineResult, line, run_line,
    // SuperSmoother
    SupsmuResult, SupsmuConfig, supsmu, run_supsmu,
    // Quantile regression
    QuantRegResult, QuantRegCoefficient, QuantRegConfig, QuantRegAlgorithm,
    quantreg, run_quantreg, quantreg_multi,
    // Sensitivity analysis for unmeasured confounding (sensemakr)
    SensemakrResult, SensitivityBound, ContourData,
    sensemakr, run_sensemakr, generate_contour_data,
    partial_r2, robustness_value, robustness_value_alpha,
    confounding_bias, adjusted_estimate, adjusted_se,
    // Average marginal effects (AME)
    MarginalEffectsResult, MarginalEffect, ModelType, ContrastsResult, ContrastEffect,
    marginal_effects, marginal_effects_ols, marginal_effects_discrete, contrasts,
    // E-value sensitivity analysis for unmeasured confounding
    EValueResult, EffectType,
    evalue_rr, evalue_rr_ci, evalue_or, evalue_hr, evalue_smd, evalue_rd,
    bias_factor, bounding_factor, could_explain_away,
};
pub use econometrics::{
    PanelResult, HausmanResult, run_fixed_effects, run_random_effects, run_hausman_test,
    // Panel GLS (FGLS)
    PanelGlsResult, PanelGlsModel, run_panel_gls, run_fegls, run_pooled_gls,
    // Arellano-Bond / System GMM (dynamic panel)
    GmmResult, GmmConfig, GmmTransform, GmmStep, run_gmm, run_arellano_bond,
    // Variable Coefficients Model (pvcm) and Mean Group (pmg)
    PvcmResult, PvcmType, run_pvcm, run_pmg,
    // General GMM (Hansen 1982)
    GeneralGmmConfig, GeneralGmmResult, GmmMethod, GmmVcov, MomentCondition,
    run_general_gmm, run_gmm_iv,
    IVResult, run_iv2sls, FirstStageDiagnostics, run_first_stage_diagnostics,
    // Sargan test for IV overidentification
    SarganTestResult, sargan_test, run_sargan_test,
    DiDResult, run_did,
    // Staggered DiD (Callaway-Sant'Anna)
    ComparisonGroup, AttEstimationMethod, StaggeredDidConfig, Aggregation,
    GroupTimeATT, AggregatedEffect, PreTrendTest, StaggeredDidResult,
    run_staggered_did,
    // Goodman-Bacon decomposition for staggered DiD
    ComparisonType, BaconComponent, BaconEstimatesByType, BaconDecompResult,
    bacon_decomp,
    // Extended TWFE (Wooldridge)
    EtwfeConfig, EtwfeResult, ControlGroup, CohortTimeEffect,
    EtwfeAggregatedEffect, run_etwfe,
    DiscreteResult, run_logit, run_probit,
    // Multinomial logit
    MultinomResult, run_multinom,
    // Ordered logit/probit
    OrderedModelType, OrderedResult, run_ordered_logit, run_ordered_probit,
    // Negative binomial regression
    NegBinResult, run_negbin,
    // Zero-inflated models
    ZeroInflatedType, ZeroInflResult, run_zip, run_zinb,
    // Hurdle models
    HurdleType, HurdleResult, run_hurdle,
    // McFadden conditional logit (mlogit)
    MlogitResult, run_mlogit, run_conditional_logit,
    // Mixed logit / Random parameters logit (gmnl, mixl)
    RandomDistribution, RandomParameterSpec, MixedLogitConfig, MixedLogitResult,
    run_mixed_logit, run_gmnl, run_mixl,
    VarResult, VarmaResult, VecmResult, VarIrfResult, run_var, run_varma, run_vecm, run_var_irf,
    // Granger causality test
    GrangerResult, granger_test, run_granger_test, granger_test_bidirectional,
    HdfeResult, HdfeConfig, FactorInfo, run_hdfe,
    // GLM with HDFE
    GlmFamily, FeglmConfig, FeglmResult, run_feglm,
    // Treatment effects
    Estimand, DRMethod, IpwConfig, DoublyRobustConfig,
    PropensityScoreSummary, IpwResult, DoublyRobustResult,
    run_ipw_treatment, run_doubly_robust,
    // Covariate Balancing Propensity Score (CBPS)
    CbpsMethod, CbpsConfig, CbpsResult, BalanceTable, CovariateBalance,
    run_cbps, cbps,
    // Targeted Maximum Likelihood Estimation (TMLE)
    QModel, GModel, TmleConfig, TmleResult, tmle, run_tmle,
    // Collaborative TMLE (C-TMLE) with data-adaptive covariate selection
    StoppingRule, SelectionOrder, CVCriterion, CTmleQModel,
    CTmleConfig, SelectionStep, CTmleResult, CTmleConfigSummary,
    ctmle, run_ctmle, ctmle_arrays,
    // Flexible inverse probability weighting (WeightIt)
    WeightMethod, WeightEstimand, WeightItConfig, WeightItResult,
    WeightItBalanceTable, WeightItCovariateBalance, EntropyBalanceResult,
    weightit, entropy_balance,
    // Propensity Score Matching (MatchIt)
    MatchMethod, DistanceMethod, MatchCovariateBalance, MatchBalanceTable,
    MatchInfo, SubclassInfo, MatchResult,
    match_it, nearest_neighbor_match, cem_match, full_match, subclass_match,
    // GBM Propensity Score Estimation (twang)
    StopMethod, TwangEstimand, TwangConfig, TwangCovariateBalance, TwangBalanceTable,
    DecisionStump, TwangResult, run_twang, twang,
    // Mediation analysis (IPW-based)
    MediationConfig, MediationResult, run_mediation_analysis,
    // Natural Effect Models for mediation (medflex)
    EffectScale, MedflexConfig, MedflexResult, run_medflex, run_medflex_dataset,
    // Synthetic control
    SynthConfig, SynthResult, PredictorSpec, PredictorBalance, TimeEffect,
    PlaceboResults, VOptimization, TimeAggregation, run_synthetic_control,
    // Generalized synthetic control (gsynth)
    GsynthConfig, GsynthResult, GsynthEstimator, GsynthForce, UnitEffect,
    run_gsynth,
    // Synthetic Control with Prediction Intervals (SCPI)
    SCPIConstraint, SCPIConfig, VarianceMethod, PredictionInterval, SCPIResult, run_scpi,
    // Regression discontinuity
    KernelType, BandwidthMethod, VceType, RdConfig, RdBandwidth,
    RdResult, FuzzyRdResult, run_rd, rd_bandwidth, run_fuzzy_rd,
    // Multi-cutoff RD (rdmulti)
    RdMultiBandwidth, PoolingWeights, RdMultiConfig, CutoffResult, HeterogeneityTest,
    RdMultiResult, run_rd_multi, run_rd_multi_dataset,
    // Survival analysis
    TiesMethod, AftDistribution,
    KaplanMeierResult, LogRankResult, CoxConfig, CoxResult, AftConfig, AftResult,
    CumulativeIncidence, CompetingRisksResult,
    run_kaplan_meier, log_rank_test, run_cox_ph, run_aft, run_competing_risks,
    // Panel unit root tests
    PanelUnitRootTest, PanelModel, PanelUnitRootConfig, PanelUnitRootResult,
    run_panel_unit_root,
    // Spatial regression models (SAR, SEM, SAC)
    SarConfig, SemConfig, SacConfig, SpatialImpacts, SarResult, SemResult, SacResult,
    run_sar, run_sem, run_sac, run_sar_dataset, run_sem_dataset, run_sac_dataset,
    // Spatial probit models (spatialprobit package)
    SpatialProbitModel, SpatialProbitConfig, SpatialProbitImpacts, SpatialProbitResult,
    run_sar_probit, run_sem_probit,
    // Spatial GMM with Heteroscedasticity-Robust Estimation (sphet package)
    SphetModel, SphetSE, SphetConfig, SphetResult, run_sphet, sphet,
    // Spatial Panel Data Models (splm package)
    SpatialPanelEffect, SpatialPanelModel, SpatialErrorType,
    SpmlConfig, SpmlResult, run_spml,
    SpgmMethod, SpgmMoments, SpgmConfig, SpgmResult, run_spgm,
    // Stable Balancing Weights (SBW)
    SBWEstimand, SBWConfig, BalanceStats, SBWResult, run_sbw, sbw,
    // Treatment Effect Heterogeneity Testing (hettx)
    HetTestStat, EffectEstimationMethod, HetTxConfig,
    HetDecomposition, EffectSummary, HetTxResult,
    run_hettx, run_hettx_dataset,
    // Regression Standardization / G-computation (stdReg)
    StdRegModel, StdRegEstimand, SEMethod, StdRegConfig,
    SubgroupEffect, StdRegResult, run_stdreg, stdreg,
    // Balke-Pearl bounds for nonparametric IV
    BPBoundsConfig, CellProbabilities, MarginalProbabilities, BPBoundsResult,
    run_bp_bounds, bp_bounds_from_probs,
    // Marginal Treatment Effects (MTE) for IV analysis (ivmte)
    MTEEstimand, PropensityModel, IVMTEConfig, MTEPoint,
    TreatmentEffectEstimate, PropensityStageResult, IVMTEResult,
    run_ivmte, ivmte, run_ivmte_multi_z,
};
pub use spatial::{
    // Neighbors and weights
    Neighbors, NeighborMethod, SpatialWeights, WeightStyle, SparseWeights,
    // Diagnostics
    MoranResult, GearyResult, SpatialLmTests, LmTestResult, MoranAlternative,
    moran_test, moran_test_residuals, geary_test, spatial_lm_tests,
    // Local Moran's I (LISA)
    LisaCluster, LocalMoranObs, LocalMoranResult, localmoran,
};
#[cfg(feature = "forecasting")]
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
    // Time series utilities
    LagResult, lag, lag_padded,
    EmbedResult, embed, embed_array,
    DiffinvResult, diffinv,
    FilterMethod, FilterSides, FilterResult, filter,
    WindowResult, window,
    ArmaAcfResult, arma_acf,
    ArmaToMaResult, arma_to_ma,
    Acf2ArResult, acf_to_ar,
    ArimaSimResult, arima_sim,
    EndRule, RunmedResult, runmed,
    // Cumulative periodogram
    CpgramResult, cpgram, run_cpgram,
    // GARCH (volatility modeling)
    GarchConfig, GarchResult, garch, garch_forecast, run_garch,
    // CausalImpact (Bayesian Structural Time Series for causal inference)
    CausalImpactConfig, CausalImpactSummary, CausalImpactSeries,
    CausalImpactModel, CausalInference, CausalImpactResult,
    causal_impact, run_causal_impact,
};
pub use ml::{
    KMeansResult, DBSCANResult, HierarchicalResult, Linkage, PCAResult, TsneResult,
    CmdscaleResult, CutreeResult, RandomForestResult, SvmResult,
    kmeans, dbscan, hierarchical, pca, pca_transform, pca_inverse_transform, tsne,
    cmdscale, cmdscale_from_data, run_cmdscale,
    cutree, cutree_multiple_k, run_cutree,
    random_forest, linear_svm, svm_predict,
    // Projection Pursuit Regression
    ppr, run_ppr, PprResult, PprConfig, SmoothingMethod,
    // Causal Forests (Wager & Athey 2018)
    causal_forest, causal_forest_arrays, causal_forest_predict, causal_forest_predict_arrays,
    average_treatment_effect, run_causal_forest,
    CausalForestConfig, CausalForestResult,
    // BART for Causal Inference (bcf, bartCause)
    bart_causal, bart_causal_arrays, bart_causal_predict, bart_causal_predict_arrays,
    run_bart_causal, BartCausalConfig, BartCausalResult,
};
#[cfg(feature = "visualization")]
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
pub use export::{
    CsvExport,
    HtmlTableBuilder, HtmlStyle,
    LatexTableBuilder, LatexStyle,
    MarkdownTableBuilder, MarkdownStyle,
};

/// Re-export polars for downstream use
pub use polars;
