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
    // Parametric G-Formula for causal inference with time-varying treatments (gfoRmula)
    GFormulaIntervention, GFormulaOutcomeType, GFormulaConfig, GFormulaResult,
    run_gformula, gformula,
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
