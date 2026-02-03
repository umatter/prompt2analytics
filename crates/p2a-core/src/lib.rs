//! # p2a-core
//!
//! **Pure Rust econometrics and statistical analysis library** with 200+ methods
//! validated against R/Python reference implementations.
//!
//! This crate provides data loading, statistical analysis, econometrics, machine learning,
//! and visualization functionality. All algorithms are implemented in pure Rust without
//! external econometrics dependencies.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use p2a_core::{Dataset, run_ols};
//! use p2a_core::regression::CovarianceType;
//! use polars::prelude::*;
//!
//! // Load data
//! let df = CsvReadOptions::default()
//!     .try_into_reader_with_file_path(Some("data.csv".into()))
//!     .unwrap()
//!     .finish()
//!     .unwrap();
//! let dataset = Dataset::new(df);
//!
//! // Run OLS with robust standard errors
//! let result = run_ols(&dataset, "y", &["x1", "x2"], true, CovarianceType::HC1).unwrap();
//! println!("R² = {:.4}", result.r_squared);
//! ```
//!
//! ## Feature Flags
//!
//! - `forecasting` - Time series forecasting (ARIMA, GARCH, Holt-Winters, etc.)
//! - `visualization` - Chart generation (histograms, scatter plots, heatmaps)
//! - `spectral-analysis` - Spectral density estimation
//!
//! ## Method Catalog
//!
//! ### Statistical Tests (50+ methods)
//!
//! | Category | Methods | Module |
//! |----------|---------|--------|
//! | **T-tests** | [`one_sample_t_test`], [`two_sample_t_test`], [`paired_t_test`] | [`stats`] |
//! | **ANOVA** | [`run_one_way_anova`], [`run_two_way_anova`], [`oneway_test`] | [`stats`] |
//! | **Chi-squared** | [`chisq_test_gof`], [`chisq_test_independence`] | [`stats`] |
//! | **Nonparametric** | [`wilcoxon_rank_sum`], [`wilcoxon_signed_rank`], [`kruskal_test`], [`friedman_test`] | [`stats`] |
//! | **Normality** | [`shapiro_wilk_test`], [`ks_test_one_sample`] | [`stats`] |
//! | **Variance** | [`bartlett_test`], [`fligner_test`], [`var_test`] | [`stats`] |
//! | **Correlation** | [`cor_test`], [`correlation_matrix`] | [`stats`] |
//! | **Post-hoc** | [`tukey_hsd`], [`pairwise_t_test`], [`pairwise_wilcox_test`] | [`stats`] |
//! | **Time series** | [`acf`], [`pacf`], [`ccf`], [`box_test`], [`pp_test`] | [`stats`] |
//! | **Power analysis** | [`power_t_test`], [`power_prop_test`], [`power_anova_test`] | [`stats`] |
//! | **Proportions** | [`prop_test_one`], [`prop_test_two`], [`binom_test`] | [`stats`] |
//! | **Contingency** | [`fisher_exact_test`], [`mcnemar_test`], [`mantelhaen_test`] | [`stats`] |
//!
//! ### Regression Analysis (20+ methods)
//!
//! | Method | Description | Function |
//! |--------|-------------|----------|
//! | **OLS** | Ordinary least squares with HC0-HC3 robust SEs | [`run_ols`] |
//! | **Clustered SE** | Clustered standard errors | [`run_ols_clustered`] |
//! | **GLS** | Generalized least squares (AR1, MA1, ARMA) | [`run_gls`] |
//! | **Quantile Reg** | Quantile/median regression | [`run_quantreg`] |
//! | **LOESS** | Local polynomial regression | [`run_loess`] |
//! | **NLS** | Nonlinear least squares | [`run_nls`] |
//! | **Stepwise** | Forward/backward model selection | [`run_step`] |
//! | **Diagnostics** | JB, BP, DW, VIF tests | [`run_diagnostics`] |
//! | **HAC** | Newey-West standard errors | [`vcov_hac`] |
//! | **Bootstrap** | Bootstrap covariance estimation | [`vcov_bootstrap`] |
//!
//! ### Panel Data Econometrics (15+ methods)
//!
//! | Method | Description | Function |
//! |--------|-------------|----------|
//! | **Fixed Effects** | Within estimator | [`run_fixed_effects`] |
//! | **Random Effects** | GLS estimator | [`run_random_effects`] |
//! | **Hausman Test** | FE vs RE specification test | [`run_hausman_test`] |
//! | **HDFE** | High-dimensional fixed effects | [`run_hdfe`] |
//! | **FEGLM** | GLM with fixed effects | [`run_feglm`] |
//! | **Arellano-Bond** | Dynamic panel GMM | [`run_arellano_bond`] |
//! | **Panel GLS** | Feasible GLS | [`run_panel_gls`] |
//! | **Panel Unit Root** | LLC, IPS, Fisher tests | [`run_panel_unit_root`] |
//!
//! ### Discrete Choice Models (15+ methods)
//!
//! | Method | Description | Function |
//! |--------|-------------|----------|
//! | **Logit** | Binary logistic regression | [`run_logit`] |
//! | **Probit** | Binary probit | [`run_probit`] |
//! | **Ordered Logit** | Proportional odds model | [`run_ordered_logit`] |
//! | **Ordered Probit** | Ordered probit | [`run_ordered_probit`] |
//! | **Multinomial** | Multinomial logit | [`run_multinom`] |
//! | **Conditional Logit** | McFadden's choice model | [`run_conditional_logit`] |
//! | **Mixed Logit** | Random parameters logit | [`run_mixed_logit`] |
//! | **Negative Binomial** | Count data with overdispersion | [`run_negbin`] |
//! | **Zero-Inflated** | ZIP, ZINB models | [`run_zip`], [`run_zinb`] |
//! | **Hurdle** | Two-part count models | [`run_hurdle`] |
//!
//! ### Causal Inference (30+ methods)
//!
//! | Method | Description | Function |
//! |--------|-------------|----------|
//! | **IV/2SLS** | Instrumental variables | [`run_iv2sls`] |
//! | **DiD** | Difference-in-differences | [`run_did`] |
//! | **Staggered DiD** | Callaway-Sant'Anna | [`run_staggered_did`] |
//! | **Bacon Decomp** | Goodman-Bacon decomposition | [`bacon_decomp`] |
//! | **ETWFE** | Extended TWFE (Wooldridge) | [`run_etwfe`] |
//! | **IPW** | Inverse probability weighting | [`run_ipw_treatment`] |
//! | **AIPW** | Doubly robust estimation | [`run_doubly_robust`] |
//! | **CBPS** | Covariate balancing PS | [`run_cbps`] |
//! | **TMLE** | Targeted MLE | [`run_tmle`] |
//! | **C-TMLE** | Collaborative TMLE | [`run_ctmle`] |
//! | **Matching** | Nearest neighbor, CEM, full | [`match_it`] |
//! | **Synth Control** | Abadie synthetic control | [`run_synthetic_control`] |
//! | **GSynth** | Generalized synthetic control | [`run_gsynth`] |
//! | **RD** | Sharp/fuzzy regression discontinuity | [`run_rd`], [`run_fuzzy_rd`] |
//! | **Mediation** | Natural effect models | [`run_mediation_analysis`], [`run_medflex`] |
//! | **Sensemakr** | Sensitivity to confounding | [`run_sensemakr`] |
//! | **G-Formula** | Parametric g-computation | [`run_gformula`] |
//!
//! ### Time Series (20+ methods)
//!
//! | Method | Description | Function |
//! |--------|-------------|----------|
//! | **VAR** | Vector autoregression | [`run_var`] |
//! | **VARMA** | Vector ARMA | [`run_varma`] |
//! | **VECM** | Vector error correction | [`run_vecm`] |
//! | **IRF** | Impulse response functions | [`run_var_irf`] |
//! | **Granger** | Granger causality test | [`granger_test`] |
//! | **ARIMA** | ARIMA modeling (feature: forecasting) | `run_arima` |
//! | **GARCH** | Volatility modeling (feature: forecasting) | `run_garch` |
//! | **Holt-Winters** | Exponential smoothing (feature: forecasting) | `run_holt_winters` |
//! | **STL/MSTL** | Seasonal decomposition (feature: forecasting) | `run_stl`, `run_mstl` |
//! | **Changepoint** | Structural break detection (feature: forecasting) | `detect_changepoints` |
//! | **Kalman** | State space models (feature: forecasting) | `kalman_filter` |
//!
//! ### Spatial Econometrics (15+ methods)
//!
//! | Method | Description | Function |
//! |--------|-------------|----------|
//! | **SAR** | Spatial autoregressive | [`run_sar`] |
//! | **SEM** | Spatial error model | [`run_sem`] |
//! | **SAC** | Combined SAR + SEM | [`run_sac`] |
//! | **Spatial Probit** | SAR/SEM probit | [`run_sar_probit`] |
//! | **Spatial GMM** | sphet package methods | [`run_sphet`] |
//! | **Spatial Panel** | splm package methods | [`run_spml`] |
//! | **Moran's I** | Spatial autocorrelation | [`moran_test`] |
//! | **Geary's C** | Spatial association | [`geary_test`] |
//! | **LISA** | Local indicators | [`localmoran`] |
//!
//! ### Survival Analysis (5+ methods)
//!
//! | Method | Description | Function |
//! |--------|-------------|----------|
//! | **Kaplan-Meier** | Nonparametric survival | [`run_kaplan_meier`] |
//! | **Log-rank** | Survival curve comparison | [`log_rank_test`] |
//! | **Cox PH** | Proportional hazards | [`run_cox_ph`] |
//! | **AFT** | Accelerated failure time | [`run_aft`] |
//! | **Competing Risks** | Cumulative incidence | [`run_competing_risks`] |
//!
//! ### Machine Learning (20+ methods)
//!
//! | Method | Description | Function |
//! |--------|-------------|----------|
//! | **K-Means** | Clustering | [`kmeans`] |
//! | **DBSCAN** | Density-based clustering | [`dbscan`] |
//! | **Hierarchical** | Agglomerative clustering | [`hierarchical`] |
//! | **PCA** | Principal components | [`pca`] |
//! | **t-SNE** | Dimensionality reduction | [`tsne`] |
//! | **MDS** | Multidimensional scaling | [`cmdscale`] |
//! | **Random Forest** | Ensemble trees | [`random_forest`] |
//! | **SVM** | Support vector machines | [`linear_svm`] |
//! | **Causal Forest** | Heterogeneous treatment effects | [`causal_forest`] |
//! | **BART Causal** | Bayesian trees for causal | [`bart_causal`] |
//!
//! ### Export & Visualization
//!
//! | Category | Methods |
//! |----------|---------|
//! | **LaTeX** | [`LatexTableBuilder`] - Publication-ready tables |
//! | **HTML** | [`HtmlTableBuilder`] - Self-contained HTML tables |
//! | **Markdown** | [`MarkdownTableBuilder`] - GitHub-compatible tables |
//! | **CSV** | [`CsvExport`] trait for all result types |
//! | **Charts** | `histogram`, `scatter_plot`, `box_plot`, `correlation_heatmap` (feature: visualization) |
//!
//! ## Module Organization
//!
//! - [`data`] - Dataset loading and manipulation
//! - [`stats`] - Statistical tests and descriptive statistics
//! - [`regression`] - OLS, GLS, quantile regression, diagnostics
//! - [`econometrics`] - Panel data, discrete choice, causal inference, time series
//! - [`spatial`] - Spatial weights, autocorrelation tests, LISA
//! - [`ml`] - Clustering, dimensionality reduction, ensemble methods
//! - [`forecasting`] - Time series forecasting (feature-gated)
//! - [`visualization`] - Chart generation (feature-gated)
//! - [`export`] - LaTeX, HTML, Markdown, CSV export
//! - [`linalg`] - Matrix operations (X'X, Cholesky, etc.)
//! - [`traits`] - Common traits ([`LinearEstimator`])
//! - [`errors`] - Error types ([`EconError`])
//!
//! ## R Package Equivalents
//!
//! Many methods are validated against R packages:
//!
//! | R Package | p2a-core Equivalent |
//! |-----------|---------------------|
//! | `stats` | [`stats`], [`regression`] |
//! | `plm` | [`econometrics::panel`] |
//! | `fixest` | [`run_hdfe`], [`run_feglm`] |
//! | `lmtest` | [`run_diagnostics`], [`bg_test`], [`reset_test`] |
//! | `sandwich` | [`vcov_hac`], [`vcov_bootstrap`] |
//! | `MASS` | [`run_ordered_logit`], [`run_negbin`] |
//! | `nnet` | [`run_multinom`] |
//! | `survival` | [`run_cox_ph`], [`run_kaplan_meier`] |
//! | `did` | [`run_staggered_did`] |
//! | `rdrobust` | [`run_rd`] |
//! | `MatchIt` | [`match_it`] |
//! | `WeightIt` | [`weightit`] |
//! | `tmle` | [`run_tmle`] |
//! | `Synth` | [`run_synthetic_control`] |
//! | `spdep` | [`spatial`] module |
//! | `vars` | [`run_var`], [`run_vecm`] |

// Documentation lint disabled - handled at workspace level in Cargo.toml
// #![warn(missing_docs)]

// Foundation modules (pure Rust implementations)
pub mod errors;
pub mod linalg;
pub mod traits;

// Core feature modules (always available)
pub mod cache;
pub mod data;
pub mod diagnostics;
pub mod econometrics;
pub mod export;
pub mod memory;
pub mod ml;
pub mod regression;
pub mod reports;
pub mod simulation;
pub mod spatial;
pub mod stats;

// Optional feature modules
#[cfg(feature = "forecasting")]
pub mod forecasting;

#[cfg(feature = "visualization")]
pub mod visualization;

// Re-export foundational types
pub use cache::{CacheKey, CacheStats, ResultCache};
pub use errors::{EconError, EconResult, EstimationWarning};
pub use memory::{
    DatasetMemoryInfo, MemoryProfiler, MemorySnapshot, MemoryStats, MemoryTracker, ProcessMemory,
    estimate_dataset_memory, format_bytes, get_process_memory,
};
pub use linalg::{
    DesignError,
    DesignMatrix,
    // Toeplitz matrix construction
    toeplitz,
    toeplitz_acf,
    toeplitz_asymmetric,
    toeplitz_to_vec,
    toeplitz2,
};
pub use traits::{LinearEstimator, SignificanceLevel};

pub use data::{DataLoader, Dataset, DatasetInfo};
pub use stats::{
    AcfResult,
    // ACF/PACF/CCF
    AcfType,
    Alternative,
    AnovaResult,
    // Ansari-Bradley test for scale parameters
    AnsariBradleyResult,
    ApproxMethod,
    ApproxResult,
    ApproxRule,
    BartlettGroupStats,
    // Bartlett's test
    BartlettResult,
    // Exact binomial test
    BinomTestResult,
    // Box-Pierce and Ljung-Box tests
    BoxTestResult,
    BoxTestType,
    CcfResult,
    CcfType,
    // Chi-squared tests
    ChiSquaredResult,
    CmhAlternative,
    ConstrOptimConfig,
    ConstrOptimResult,
    ContrastType,
    // Correlation test
    CorTestResult,
    CorrelationMatrix,
    CorrelationMethod,
    CovWtCenter,
    CovWtMethod,
    CovWtResult,
    DensityKernel,
    DensityResult,
    DescriptiveStats,
    EcdfResult,
    FisherAlternative,
    // Fisher's exact test
    FisherExactResult,
    // Robust/descriptive stats
    FivenumResult,
    // Fligner-Killeen test for homogeneity of variances
    FlignerResult,
    // Friedman rank sum test
    FriedmanResult,
    GroupStats,
    IsoregResult,
    // Kruskal-Wallis rank sum test
    KruskalWallisResult,
    // Kolmogorov-Smirnov test
    KsTestResult,
    LoglinResult,
    // Mahalanobis distance
    MahalanobisResult,
    // MANOVA
    ManovaResult,
    ManovaTestResult,
    ManovaTestStatistic,
    // Cochran-Mantel-Haenszel test for stratified 2x2 tables
    MantelHaenszelResult,
    MauchlyResult,
    // McNemar's chi-squared test
    McnemarResult,
    MedpolishResult,
    ModelTablesResult,
    ModelTablesSE,
    // Mood test for scale parameters
    MoodTestResult,
    // Welch's one-way ANOVA
    OnewayTestResult,
    OptimMethod,
    // Phillips-Perron unit root test
    PPTestResult,
    PValueAdjustMethod,
    PacfResult,
    PairwiseComparison,
    // Pairwise t-tests with p-value adjustment
    PairwiseTTestResult,
    // Pairwise Wilcoxon tests with p-value adjustment
    PairwiseWilcoxResult,
    PoissonAlternative,
    // Exact Poisson test
    PoissonTestResult,
    PowerAlternative,
    PowerAnovaTestResult,
    PowerPropTestResult,
    // Power analysis
    PowerTTestResult,
    // Proportion tests
    PropTestResult,
    // Prop trend test
    PropTrendTestResult,
    // Quade test for unreplicated blocked data
    QuadeResult,
    RegressionAnovaResult,
    SeContrastResult,
    // Shapiro-Wilk normality test
    ShapiroWilkResult,
    SplineMethod,
    SplineResult,
    StratumStats,
    TTestResult,
    TTestType,
    Table2x2,
    TableType,
    TheoreticalDistribution,
    // Tukey HSD
    TukeyHsdResult,
    TwoWayAnovaResult,
    TwoWayModelTablesResult,
    // F test for comparing two variances
    VarTestResult,
    WilcoxonConfig,
    // Wilcoxon tests
    WilcoxonResult,
    acf,
    anova_from_ols,
    ansari_test,
    approx,
    approxfun,
    bartlett_test,
    binom_test,
    box_test,
    ccf,
    chisq_test_gof,
    chisq_test_independence,
    // Constrained optimization
    constr_optim,
    contrast_p_value,
    contrast_t_statistic,
    cor_test,
    correlation_matrix,
    cov_wt,
    cov_wt_from_slice,
    density,
    ecdf,
    estimate_contrast,
    fisher_exact_test,
    fisher_exact_test_int,
    fivenum,
    fligner_test,
    format_model_tables,
    friedman_test,
    generate_contrasts,
    iqr,
    // Isotonic regression
    isoreg,
    isoreg_predict,
    isoreg_y,
    kruskal_test,
    ks_test,
    ks_test_one_sample,
    ks_test_two_sample,
    // Log-linear models
    loglin,
    loglin_independence,
    loglin_saturated,
    mad,
    mahalanobis,
    mahalanobis_single,
    manova_one_way,
    mantelhaen_test,
    // Mauchly's sphericity test
    mauchly_test,
    mauchly_test_from_slice,
    mcnemar_test,
    mcnemar_test_matrix,
    // Median polish
    medpolish,
    medpolish_array,
    // Model tables
    model_tables,
    model_tables_effects,
    model_tables_means,
    model_tables_two_way,
    mood_test,
    one_sample_t_test,
    oneway_test,
    p_adjust,
    pacf,
    paired_t_test,
    pairwise_t_test,
    pairwise_wilcox_test,
    poisson_test,
    power_anova_test,
    power_prop_test,
    power_t_test,
    pp_test,
    prop_test_k,
    prop_test_one,
    prop_test_two,
    prop_trend_test,
    quade_test,
    quantile,
    run_acf,
    run_bartlett_test,
    run_box_test,
    run_ccf,
    run_chisq_gof,
    run_chisq_independence,
    run_constr_optim,
    run_cor_test,
    run_cov_wt,
    run_density,
    run_ecdf,
    run_fisher_test,
    run_fivenum,
    run_fligner_test,
    run_friedman_test,
    run_iqr,
    run_isoreg,
    run_kruskal_test,
    run_ks_test,
    run_loglin,
    run_mad,
    run_mahalanobis,
    run_manova,
    run_mantelhaen_test,
    run_mauchly_test,
    run_medpolish,
    run_model_tables,
    run_mood_test,
    run_one_way_anova,
    run_oneway_test,
    run_pacf,
    run_pairwise_t_test,
    run_pairwise_wilcox_test,
    run_power_anova_test,
    run_power_prop_test,
    run_power_t_test,
    run_pp_test,
    run_prop_trend_test,
    run_quade_test,
    run_se_contrast,
    run_shapiro_wilk,
    run_tukey_hsd,
    run_two_way_anova,
    run_var_test,
    run_weighted_mean,
    // Standard errors for contrasts
    se_contrast,
    se_contrast_single,
    shapiro_wilk_test,
    // Spline interpolation
    spline,
    splinefun,
    t_test,
    tukey_hsd,
    two_sample_t_test,
    var_test,
    // Weighted statistics
    weighted_mean,
    wilcoxon_rank_sum,
    wilcoxon_signed_rank,
    wilcoxon_test,
};

// Spectral density estimation (requires spectral-analysis feature)
pub use econometrics::{
    AftConfig,
    AftDistribution,
    AftResult,
    AggregatedEffect,
    Aggregation,
    AttEstimationMethod,
    // Balke-Pearl bounds for nonparametric IV
    BPBoundsConfig,
    BPBoundsResult,
    BaconComponent,
    BaconDecompResult,
    BaconEstimatesByType,
    BalanceStats,
    BalanceTable,
    BandwidthMethod,
    CTmleConfig,
    CTmleConfigSummary,
    CTmleQModel,
    CTmleResult,
    CVCriterion,
    CbpsConfig,
    // Covariate Balancing Propensity Score (CBPS)
    CbpsMethod,
    CbpsResult,
    CellProbabilities,
    CohortTimeEffect,
    // Staggered DiD (Callaway-Sant'Anna)
    ComparisonGroup,
    // Goodman-Bacon decomposition for staggered DiD
    ComparisonType,
    CompetingRisksResult,
    ControlGroup,
    CovariateBalance,
    CoxConfig,
    CoxResult,
    CumulativeIncidence,
    CutoffResult,
    DRMethod,
    DecisionStump,
    DiDResult,
    DiscreteResult,
    DistanceMethod,
    DoublyRobustConfig,
    DoublyRobustResult,
    EffectEstimationMethod,
    // Natural Effect Models for mediation (medflex)
    EffectScale,
    EffectSummary,
    EntropyBalanceResult,
    // Treatment effects
    Estimand,
    EtwfeAggregatedEffect,
    // Extended TWFE (Wooldridge)
    EtwfeConfig,
    EtwfeResult,
    FactorInfo,
    FeglmConfig,
    FeglmResult,
    FirstStageDiagnostics,
    FuzzyRdResult,
    GFormulaConfig,
    GFormulaData,
    // Parametric G-Formula for causal inference with time-varying treatments (gfoRmula)
    GFormulaIntervention,
    GFormulaOutcomeType,
    GFormulaResult,
    GModel,
    // General GMM (Hansen 1982)
    GeneralGmmConfig,
    GeneralGmmResult,
    // GLM with HDFE
    GlmFamily,
    GmmConfig,
    GmmMethod,
    // Arellano-Bond / System GMM (dynamic panel)
    GmmResult,
    GmmStep,
    GmmTransform,
    GmmVcov,
    // Granger causality test
    GrangerResult,
    GroupTimeATT,
    // Generalized synthetic control (gsynth)
    GsynthConfig,
    GsynthEstimator,
    GsynthForce,
    GsynthResult,
    HausmanResult,
    HdfeConfig,
    HdfeResult,
    HetDecomposition,
    // Treatment Effect Heterogeneity Testing (hettx)
    HetTestStat,
    HetTxConfig,
    HetTxResult,
    HeterogeneityTest,
    HurdleResult,
    // Hurdle models
    HurdleType,
    IVMTEConfig,
    IVMTEResult,
    IVResult,
    IpwConfig,
    IpwResult,
    KaplanMeierResult,
    // Regression discontinuity
    KernelType,
    LogRankResult,
    // Marginal Treatment Effects (MTE) for IV analysis (ivmte)
    MTEEstimand,
    MTEPoint,
    MarginalProbabilities,
    MatchBalanceTable,
    MatchCovariateBalance,
    MatchInfo,
    // Propensity Score Matching (MatchIt)
    MatchMethod,
    MatchResult,
    MedflexConfig,
    MedflexResult,
    // Mediation analysis (IPW-based)
    MediationConfig,
    MediationResult,
    MixedLogitConfig,
    MixedLogitResult,
    // McFadden conditional logit (mlogit)
    MlogitResult,
    MomentCondition,
    // Multinomial logit
    MultinomResult,
    // Negative binomial regression
    NegBinResult,
    // Ordered logit/probit
    OrderedModelType,
    OrderedResult,
    PanelGlsModel,
    // Panel GLS (FGLS)
    PanelGlsResult,
    PanelModel,
    PanelResult,
    PanelUnitRootConfig,
    PanelUnitRootResult,
    // Panel unit root tests
    PanelUnitRootTest,
    PlaceboResults,
    PoolingWeights,
    PreTrendTest,
    PredictionInterval,
    PredictorBalance,
    PredictorSpec,
    PropensityModel,
    PropensityScoreSummary,
    PropensityStageResult,
    // Variable Coefficients Model (pvcm) and Mean Group (pmg)
    PvcmResult,
    PvcmType,
    // Targeted Maximum Likelihood Estimation (TMLE)
    QModel,
    // Mixed logit / Random parameters logit (gmnl, mixl)
    RandomDistribution,
    RandomParameterSpec,
    RdBandwidth,
    RdConfig,
    // Multi-cutoff RD (rdmulti)
    RdMultiBandwidth,
    RdMultiConfig,
    RdMultiResult,
    RdResult,
    SBWConfig,
    // Stable Balancing Weights (SBW)
    SBWEstimand,
    SBWResult,
    SCPIConfig,
    // Synthetic Control with Prediction Intervals (SCPI)
    SCPIConstraint,
    SCPIResult,
    SEMethod,
    SacConfig,
    SacResult,
    // Spatial regression models (SAR, SEM, SAC)
    SarConfig,
    SarResult,
    // Sargan test for IV overidentification
    SarganTestResult,
    SelectionOrder,
    SelectionStep,
    SemConfig,
    SemResult,
    SpatialErrorType,
    SpatialImpacts,
    // Spatial Panel Data Models (splm package)
    SpatialPanelEffect,
    SpatialPanelModel,
    SpatialProbitConfig,
    SpatialProbitImpacts,
    // Spatial probit models (spatialprobit package)
    SpatialProbitModel,
    SpatialProbitResult,
    SpgmConfig,
    SpgmMethod,
    SpgmMoments,
    SpgmResult,
    SphetConfig,
    // Spatial GMM with Heteroscedasticity-Robust Estimation (sphet package)
    SphetModel,
    SphetResult,
    SphetSE,
    SpmlConfig,
    SpmlResult,
    StaggeredDidConfig,
    StaggeredDidResult,
    StdRegConfig,
    StdRegEstimand,
    // Regression Standardization / G-computation (stdReg)
    StdRegModel,
    StdRegResult,
    // GBM Propensity Score Estimation (twang)
    StopMethod,
    // Collaborative TMLE (C-TMLE) with data-adaptive covariate selection
    StoppingRule,
    SubclassInfo,
    SubgroupEffect,
    // Synthetic control
    SynthConfig,
    SynthResult,
    // Survival analysis
    TiesMethod,
    TimeAggregation,
    TimeEffect,
    TmleConfig,
    TmleResult,
    TreatmentEffectEstimate,
    TwangBalanceTable,
    TwangConfig,
    TwangCovariateBalance,
    TwangEstimand,
    TwangResult,
    UnitEffect,
    VOptimization,
    VarIrfResult,
    VarResult,
    VarianceMethod,
    VarmaResult,
    VceType,
    VecmResult,
    WeightEstimand,
    WeightItBalanceTable,
    WeightItConfig,
    WeightItCovariateBalance,
    WeightItResult,
    // Flexible inverse probability weighting (WeightIt)
    WeightMethod,
    ZeroInflResult,
    // Zero-inflated models
    ZeroInflatedType,
    bacon_decomp,
    bp_bounds_from_probs,
    cbps,
    cem_match,
    ctmle,
    ctmle_arrays,
    entropy_balance,
    full_match,
    gformula,
    granger_test,
    granger_test_bidirectional,
    ivmte,
    log_rank_test,
    match_it,
    nearest_neighbor_match,
    rd_bandwidth,
    run_aft,
    run_arellano_bond,
    run_bp_bounds,
    run_cbps,
    run_competing_risks,
    run_conditional_logit,
    run_cox_ph,
    run_ctmle,
    run_did,
    run_doubly_robust,
    run_etwfe,
    run_feglm,
    run_fegls,
    run_first_stage_diagnostics,
    run_fixed_effects,
    run_fuzzy_rd,
    run_general_gmm,
    run_gformula,
    run_gmm,
    run_gmm_iv,
    run_gmnl,
    run_granger_test,
    run_gsynth,
    run_hausman_test,
    run_hdfe,
    run_hettx,
    run_hettx_dataset,
    run_hurdle,
    run_ipw_treatment,
    run_iv2sls,
    run_ivmte,
    run_ivmte_multi_z,
    run_kaplan_meier,
    run_logit,
    run_medflex,
    run_medflex_dataset,
    run_mediation_analysis,
    run_mixed_logit,
    run_mixl,
    run_mlogit,
    run_multinom,
    run_negbin,
    run_ordered_logit,
    run_ordered_probit,
    run_panel_gls,
    run_panel_unit_root,
    run_pmg,
    run_pooled_gls,
    run_probit,
    run_pvcm,
    run_random_effects,
    run_rd,
    run_rd_multi,
    run_rd_multi_dataset,
    run_sac,
    run_sac_dataset,
    run_sar,
    run_sar_dataset,
    run_sar_probit,
    run_sargan_test,
    run_sbw,
    run_scpi,
    run_sem,
    run_sem_dataset,
    run_sem_probit,
    run_spgm,
    run_sphet,
    run_spml,
    run_staggered_did,
    run_stdreg,
    run_synthetic_control,
    run_tmle,
    run_twang,
    run_var,
    run_var_irf,
    run_varma,
    run_vecm,
    run_zinb,
    run_zip,
    sargan_test,
    sbw,
    sphet,
    stdreg,
    subclass_match,
    tmle,
    twang,
    weightit,
};
pub use export::{
    CsvExport, HtmlStyle, HtmlTableBuilder, LatexStyle, LatexTableBuilder, MarkdownStyle,
    MarkdownTableBuilder,
};
#[cfg(feature = "forecasting")]
pub use forecasting::{
    Acf2ArResult,
    ArConfig,
    ArMethod,
    // Autoregressive model fitting
    ArResult,
    ArimaForecastResult,
    ArimaResult,
    ArimaSimResult,
    ArmaAcfResult,
    ArmaToMaResult,
    // CausalImpact (Bayesian Structural Time Series for causal inference)
    CausalImpactConfig,
    CausalImpactModel,
    CausalImpactResult,
    CausalImpactSeries,
    CausalImpactSummary,
    CausalInference,
    ChangepointResult,
    CostFunction,
    // Cumulative periodogram
    CpgramResult,
    DecomposeConfig,
    // Classical decomposition
    DecomposeResult,
    DecomposeType,
    DiffinvResult,
    EmbedResult,
    EndRule,
    FilterMethod,
    FilterResult,
    FilterSides,
    // GARCH (volatility modeling)
    GarchConfig,
    GarchResult,
    HoltWintersCoefficients,
    HoltWintersConfig,
    // Holt-Winters exponential smoothing
    HoltWintersResult,
    KalmanFilterResult,
    KalmanForecastResult,
    KalmanSmootherResult,
    // Time series utilities
    LagResult,
    MstlResult,
    RunmedResult,
    SeasonalType,
    SegmentStats,
    // Kalman filter
    StateSpaceModel,
    StlConfig,
    // STL decomposition
    StlResult,
    StructTsCoefficients,
    StructTsConfig,
    StructTsResult,
    // Structural time series
    StructTsType,
    WindowResult,
    acf_to_ar,
    ar,
    arima_sim,
    arma_acf,
    arma_to_ma,
    binary_segmentation,
    causal_impact,
    cpgram,
    decompose,
    detect_changepoints,
    diffinv,
    embed,
    embed_array,
    filter,
    forecast_arima,
    garch,
    garch_forecast,
    holt_winters,
    holt_winters_forecast,
    kalman_filter,
    kalman_forecast,
    kalman_loglik,
    kalman_smoother,
    lag,
    lag_padded,
    run_ar,
    run_ar_with_order,
    run_arima,
    run_binary_segmentation,
    run_causal_impact,
    run_changepoint,
    run_cpgram,
    run_decompose,
    run_decompose_with_filter,
    run_garch,
    run_holt_winters,
    run_mstl,
    run_stl,
    run_stl_with_config,
    run_struct_ts,
    runmed,
    stl,
    struct_ts,
    window,
};
pub use ml::{
    BartCausalConfig,
    BartCausalResult,
    CausalForestConfig,
    CausalForestResult,
    CmdscaleResult,
    CutreeResult,
    DBSCANResult,
    HierarchicalResult,
    KMeansResult,
    Linkage,
    PCAResult,
    PprConfig,
    PprResult,
    RandomForestResult,
    SmoothingMethod,
    SvmResult,
    TsneResult,
    average_treatment_effect,
    // BART for Causal Inference (bcf, bartCause)
    bart_causal,
    bart_causal_arrays,
    bart_causal_predict,
    bart_causal_predict_arrays,
    // Causal Forests (Wager & Athey 2018)
    causal_forest,
    causal_forest_arrays,
    causal_forest_predict,
    causal_forest_predict_arrays,
    cmdscale,
    cmdscale_from_data,
    cutree,
    cutree_multiple_k,
    dbscan,
    hierarchical,
    kmeans,
    linear_svm,
    pca,
    pca_inverse_transform,
    pca_transform,
    // Projection Pursuit Regression
    ppr,
    random_forest,
    run_bart_causal,
    run_causal_forest,
    run_cmdscale,
    run_cutree,
    run_ppr,
    svm_predict,
    tsne,
};
pub use regression::{
    Add1Result,
    // Breusch-Godfrey test for serial correlation
    BgTestResult,
    BgTestType,
    // Bootstrap covariance estimation
    BootstrapResult,
    BootstrapType,
    ContourData,
    ContrastEffect,
    ContrastsResult,
    CorrelationStructure,
    DiagnosticsResult,
    // Driscoll-Kraay panel-robust standard errors
    DriscollKraayResult,
    Drop1Result,
    // E-value sensitivity analysis for unmeasured confounding
    EValueResult,
    EffectType,
    // GLS (generalized least squares)
    GlsResult,
    HacKernel,
    // HAC (Newey-West) standard errors
    HacResult,
    // Harvey-Collier test for linearity
    HarveyCollierResult,
    // Tukey's resistant line
    LineResult,
    LoessConfig,
    LoessModel,
    // LOESS (local polynomial regression)
    LoessResult,
    MarginalEffect,
    // Average marginal effects (AME)
    MarginalEffectsResult,
    ModelFn,
    ModelType,
    NlsAlgorithm,
    NlsConfig,
    // Nonlinear least squares
    NlsResult,
    OlsResult,
    QuantRegAlgorithm,
    QuantRegCoefficient,
    QuantRegConfig,
    // Quantile regression
    QuantRegResult,
    // Ramsey's RESET test for functional form
    ResetTestResult,
    ResetType,
    // Sensitivity analysis for unmeasured confounding (sensemakr)
    SensemakrResult,
    SensitivityBound,
    SmoothSplineConfig,
    // Smoothing spline
    SmoothSplineResult,
    StepConfig,
    StepDirection,
    StepRecord,
    // Stepwise model selection
    StepResult,
    SupsmuConfig,
    // SuperSmoother
    SupsmuResult,
    TermEvaluation,
    // Wald test for comparing nested models
    WaldTestResult,
    add1,
    adjusted_estimate,
    adjusted_se,
    bg_test,
    bg_test_from_ols,
    bias_factor,
    bounding_factor,
    confounding_bias,
    contrasts,
    could_explain_away,
    drop1,
    evalue_hr,
    evalue_or,
    evalue_rd,
    evalue_rr,
    evalue_rr_ci,
    evalue_smd,
    generate_contour_data,
    gls,
    gls_ar1_auto,
    harvey_collier_test,
    line,
    loess,
    loess_predict,
    marginal_effects,
    marginal_effects_discrete,
    marginal_effects_ols,
    model_asymptotic,
    model_exponential_decay,
    model_exponential_growth,
    model_logistic_growth,
    model_michaelis_menten,
    model_power,
    nls,
    nls_multi,
    partial_r2,
    quantreg,
    quantreg_multi,
    recursive_residuals,
    reset_test,
    reset_test_from_ols,
    robustness_value,
    robustness_value_alpha,
    run_bg_test,
    run_diagnostics,
    run_gls,
    run_harvey_collier,
    run_line,
    run_loess,
    run_nls,
    run_nls_with_config,
    run_ols,
    run_ols_clustered,
    run_ols_raw,
    run_quantreg,
    run_reset_test,
    run_sensemakr,
    run_smooth_spline,
    run_step,
    run_supsmu,
    run_vcov_bootstrap,
    run_vcov_driscoll_kraay,
    run_vcov_hac,
    run_wald_test,
    sensemakr,
    smooth_spline,
    smooth_spline_predict,
    step,
    supsmu,
    vcov_bootstrap,
    vcov_driscoll_kraay,
    vcov_hac,
    wald_test,
    wald_test_from_ols,
};
pub use reports::{HtmlReport, ReportContent, ReportSection, ReportTable, generate_html_report};
pub use simulation::{ColumnSpec, Distribution, GenerationError, generate_random_data};
pub use spatial::{
    GearyResult,
    // Local Moran's I (LISA)
    LisaCluster,
    LmTestResult,
    LocalMoranObs,
    LocalMoranResult,
    MoranAlternative,
    // Diagnostics
    MoranResult,
    NeighborMethod,
    // Neighbors and weights
    Neighbors,
    SparseWeights,
    SpatialLmTests,
    SpatialWeights,
    WeightStyle,
    geary_test,
    localmoran,
    moran_test,
    moran_test_residuals,
    spatial_lm_tests,
};
#[cfg(feature = "spectral-analysis")]
pub use stats::{
    SpectrumConfig, SpectrumResult, run_spectrum, run_spectrum_ar, spectrum, spectrum_ar,
};
#[cfg(feature = "visualization")]
pub use visualization::{
    BoxPlotResult, ChartConfig, CoefficientPlotResult, DendrogramResult, EventStudyResult,
    HeatmapResult, HistogramResult, IrfPlotResult, LineChartResult, ResidualDiagnosticsResult,
    ScatterResult, VisualizationError, box_plot, coefficient_plot, correlation_heatmap, dendrogram,
    event_study_plot, histogram, irf_plot, line_chart, residual_diagnostics, scatter_plot,
};

/// Re-export polars for downstream use
pub use polars;
