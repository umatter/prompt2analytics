//! Econometrics module with pure Rust implementations.
//!
//! This module provides 100+ econometric methods organized into categories:
//!
//! ## Panel Data (R: plm, fixest)
//!
//! - [`run_fixed_effects`] - Within (FE) estimator
//! - [`run_random_effects`] - GLS (RE) estimator
//! - [`run_hausman_test`] - FE vs RE specification test
//! - [`run_hdfe`] - High-dimensional fixed effects (reghdfe-style)
//! - [`run_feglm`] - GLM with HDFE (logit, probit, poisson)
//! - [`run_arellano_bond`] - Dynamic panel GMM
//! - [`run_panel_gls`] - Feasible GLS
//! - [`run_panel_unit_root`] - LLC, IPS, Fisher tests
//!
//! ## Instrumental Variables
//!
//! - [`run_iv2sls`] - Two-stage least squares
//! - [`run_first_stage_diagnostics`] - Weak instrument tests
//! - [`sargan_test`] - Overidentification test
//! - [`run_general_gmm`] - General GMM (Hansen 1982)
//!
//! ## Discrete Choice (R: MASS, nnet, mlogit)
//!
//! - [`run_logit`], [`run_probit`] - Binary models
//! - [`run_ordered_logit`], [`run_ordered_probit`] - Ordinal outcomes
//! - [`run_multinom`] - Multinomial logit
//! - [`run_conditional_logit`] - McFadden's choice model
//! - [`run_mixed_logit`] - Random parameters logit
//! - [`run_negbin`] - Negative binomial (count data)
//! - [`run_zip`], [`run_zinb`] - Zero-inflated models
//! - [`run_hurdle`] - Two-part count models
//!
//! ## Causal Inference - Treatment Effects
//!
//! - [`run_ipw_treatment`] - Inverse probability weighting
//! - [`run_doubly_robust`] - AIPW estimator
//! - [`run_cbps`] - Covariate balancing propensity score
//! - [`run_tmle`] - Targeted maximum likelihood
//! - [`run_ctmle`] - Collaborative TMLE
//! - [`weightit`] - Flexible weighting (WeightIt)
//! - [`entropy_balance`] - Entropy balancing
//! - [`match_it`] - Propensity score matching
//! - [`run_twang`] - GBM propensity scores
//!
//! ## Causal Inference - Design-Based
//!
//! - [`run_did`] - Difference-in-differences
//! - [`run_staggered_did`] - Callaway-Sant'Anna
//! - [`bacon_decomp`] - Goodman-Bacon decomposition
//! - [`run_etwfe`] - Extended TWFE (Wooldridge)
//! - [`run_rd`], [`run_fuzzy_rd`] - Regression discontinuity
//! - [`run_rd_multi`] - Multi-cutoff RD
//! - [`run_synthetic_control`] - Abadie synthetic control
//! - [`run_gsynth`] - Generalized synthetic control
//! - [`run_scpi`] - SC with prediction intervals
//!
//! ## Mediation & Sensitivity
//!
//! - [`run_mediation_analysis`] - IPW-based mediation
//! - [`run_medflex`] - Natural effect models
//! - [`run_hettx`] - Treatment effect heterogeneity
//! - [`run_stdreg`] - G-computation / standardization
//! - [`run_bp_bounds`] - Balke-Pearl bounds
//! - [`run_ivmte`] - Marginal treatment effects
//! - [`run_gformula`] - Parametric g-formula
//!
//! ## Time Series (R: vars)
//!
//! - [`run_var`] - Vector autoregression
//! - [`run_varma`] - Vector ARMA
//! - [`run_vecm`] - Vector error correction
//! - [`run_var_irf`] - Impulse response functions
//! - [`granger_test`] - Granger causality
//!
//! ## Spatial Econometrics (R: spdep, splm, sphet)
//!
//! - [`run_sar`] - Spatial autoregressive
//! - [`run_sem`] - Spatial error model
//! - [`run_sac`] - Combined SAR + SEM
//! - [`run_sar_probit`] - Spatial probit
//! - [`run_spml`] - Spatial panel ML
//! - [`run_sphet`] - Spatial GMM
//!
//! ## Survival Analysis (R: survival)
//!
//! - [`run_kaplan_meier`] - Nonparametric survival
//! - [`log_rank_test`] - Compare survival curves
//! - [`run_cox_ph`] - Cox proportional hazards
//! - [`run_aft`] - Accelerated failure time
//! - [`run_competing_risks`] - Cumulative incidence

mod bacon;
mod bpbounds;
mod cbps;
mod ctmle;
mod did;
mod discrete;
mod doubleml;
mod etwfe;
mod feglm;
mod general_gmm;
mod gformula;
mod hdfe;
mod hettx;
mod iv;
mod ivmte;
mod ltmle;
mod matching;
mod medflex;
mod mediation;
mod panel;
mod panel_unit_root;
mod rd;
mod rdmulti;
mod sbw;
mod scpi;
mod spatial;
mod spatialprobit;
mod sphet;
mod splm;
mod staggered_did;
mod stdreg;
mod survival;
mod synth;
mod timeseries;
mod tmle;
mod treatment;
mod twang;
mod weightit;

pub use bacon::{
    BaconComponent,
    BaconDecompResult,
    BaconEstimatesByType,
    // Goodman-Bacon decomposition for staggered DiD
    ComparisonType,
    bacon_decomp,
};
pub use cbps::{
    BalanceTable, CbpsConfig, CbpsMethod, CbpsResult, CovariateBalance, cbps, run_cbps,
};
pub use ctmle::{
    CTmleConfig,
    CTmleConfigSummary,
    CTmleQModel,
    CTmleResult,
    CVCriterion,
    SelectionOrder,
    SelectionStep,
    // Collaborative Targeted Maximum Likelihood Estimation (C-TMLE)
    StoppingRule,
    ctmle,
    ctmle_arrays,
    run_ctmle,
};
pub use did::{DiDResult, run_did};
pub use discrete::{
    DiscreteModelType,
    DiscreteResult,
    HurdleResult,
    // Hurdle models
    HurdleType,
    MixedLogitConfig,
    MixedLogitResult,
    // McFadden conditional logit (mlogit)
    MlogitResult,
    // Multinomial logit
    MultinomResult,
    // Negative binomial regression
    NegBinResult,
    // Ordered logit/probit
    OrderedModelType,
    OrderedResult,
    // Mixed logit / Random parameters logit (gmnl, mixl)
    RandomDistribution,
    RandomParameterSpec,
    ZeroInflResult,
    // Zero-inflated models
    ZeroInflatedType,
    run_conditional_logit,
    run_gmnl,
    run_hurdle,
    run_logit,
    run_mixed_logit,
    run_mixl,
    run_mlogit,
    run_multinom,
    run_negbin,
    run_ordered_logit,
    run_ordered_probit,
    run_probit,
    run_zinb,
    run_zip,
};
pub use etwfe::{
    AggregatedEffect as EtwfeAggregatedEffect, CohortTimeEffect, ControlGroup, EtwfeConfig,
    EtwfeResult, run_etwfe,
};
pub use feglm::{FeglmConfig, FeglmResult, GlmFamily, run_feglm};
pub use general_gmm::{
    GeneralGmmConfig, GeneralGmmResult, GmmMethod, GmmVcov, MomentCondition, run_general_gmm,
    run_gmm_iv,
};
pub use hdfe::{FactorInfo, HdfeConfig, HdfeResult, run_hdfe};
pub use iv::{
    FirstStageDiagnostics,
    IVResult,
    // Sargan test for overidentifying restrictions
    SarganTestResult,
    run_first_stage_diagnostics,
    run_iv2sls,
    run_sargan_test,
    sargan_test,
};
pub use ltmle::{
    // Longitudinal Targeted Maximum Likelihood Estimation (LTMLE)
    InterventionType,
    LtmleConfig,
    LtmleData,
    LtmleQModel,
    LtmleResult,
    ltmle,
    run_ltmle,
};
pub use matching::{
    DistanceMethod,
    MatchBalanceTable,
    MatchCovariateBalance,
    MatchInfo,
    // Propensity Score Matching (MatchIt)
    MatchMethod,
    MatchResult,
    SubclassInfo,
    cem_match,
    full_match,
    match_it,
    nearest_neighbor_match,
    subclass_match,
};
pub use medflex::{
    // Natural Effect Models for mediation with exposure-mediator interactions (medflex)
    EffectScale,
    MedflexConfig,
    MedflexResult,
    run_medflex,
    run_medflex_dataset,
};
pub use mediation::{MediationConfig, MediationResult, run_mediation_analysis};
pub use panel::{
    GmmConfig,
    // Arellano-Bond / System GMM
    GmmResult,
    GmmStep,
    GmmTransform,
    HausmanResult,
    PanelGlsModel,
    // Panel GLS (FGLS)
    PanelGlsResult,
    PanelMethod,
    PanelResult,
    // Variable Coefficients Model (pvcm) and Mean Group (pmg)
    PvcmResult,
    PvcmType,
    run_arellano_bond,
    run_fegls,
    run_fixed_effects,
    run_gmm,
    run_hausman_test,
    run_panel_gls,
    run_pmg,
    run_pooled_gls,
    run_pvcm,
    run_random_effects,
};
pub use panel_unit_root::{
    PanelModel, PanelUnitRootConfig, PanelUnitRootResult, PanelUnitRootTest, run_panel_unit_root,
};
pub use rd::{
    BandwidthMethod, FuzzyRdResult, KernelType, RdBandwidth, RdConfig, RdResult, VceType,
    rd_bandwidth, run_fuzzy_rd, run_rd,
};
pub use rdmulti::{
    CutoffResult,
    HeterogeneityTest,
    PoolingWeights,
    // Multi-cutoff RD (rdmulti)
    RdMultiBandwidth,
    RdMultiConfig,
    RdMultiResult,
    run_rd_multi,
    run_rd_multi_dataset,
};
pub use scpi::{
    PredictionInterval,
    SCPIConfig,
    // Synthetic Control with Prediction Intervals (SCPI)
    SCPIConstraint,
    SCPIResult,
    VarianceMethod,
    run_scpi,
};
pub use spatial::{
    SacConfig, SacResult, SarConfig, SarResult, SemConfig, SemResult, SpatialImpacts, run_sac,
    run_sac_dataset, run_sar, run_sar_dataset, run_sem, run_sem_dataset,
};
pub use staggered_did::{
    AggregatedEffect, Aggregation, AttEstimationMethod, ComparisonGroup, GroupTimeATT,
    PreTrendTest, StaggeredDidConfig, StaggeredDidResult, run_staggered_did,
};
pub use survival::{
    AftConfig, AftDistribution, AftResult, CompetingRisksResult, CoxConfig, CoxResult,
    CumulativeIncidence, KaplanMeierResult, LogRankResult, TiesMethod, log_rank_test, run_aft,
    run_competing_risks, run_cox_ph, run_kaplan_meier,
};
pub use synth::{
    // Generalized synthetic control (gsynth)
    GsynthConfig,
    GsynthEstimator,
    GsynthForce,
    GsynthResult,
    PlaceboResults,
    PredictorBalance,
    PredictorSpec,
    SynthConfig,
    SynthResult,
    TimeAggregation,
    TimeEffect,
    UnitEffect,
    VOptimization,
    run_gsynth,
    run_synthetic_control,
};
pub use timeseries::{
    // Granger causality test
    GrangerResult,
    VarIrfResult,
    VarResult,
    VarmaResult,
    VecmResult,
    granger_test,
    granger_test_bidirectional,
    run_granger_test,
    run_var,
    run_var_irf,
    run_varma,
    run_vecm,
};
pub use tmle::{
    GModel,
    // Targeted Maximum Likelihood Estimation (TMLE)
    QModel,
    TmleConfig,
    TmleResult,
    run_tmle,
    tmle,
};
pub use treatment::{
    DRMethod, DoublyRobustConfig, DoublyRobustResult, Estimand, IpwConfig, IpwResult,
    PropensityScoreSummary, run_doubly_robust, run_ipw_treatment,
};
pub use twang::{
    DecisionStump,
    // GBM-based propensity score estimation (twang)
    StopMethod,
    TwangBalanceTable,
    TwangConfig,
    TwangCovariateBalance,
    TwangEstimand,
    TwangResult,
    run_twang,
    twang,
};
pub use weightit::{
    EntropyBalanceResult,
    WeightEstimand,
    WeightItBalanceTable,
    WeightItConfig,
    WeightItCovariateBalance,
    WeightItResult,
    // Flexible inverse probability weighting (WeightIt)
    WeightMethod,
    entropy_balance,
    weightit,
};

pub use bpbounds::{
    // Balke-Pearl bounds for nonparametric IV
    BPBoundsConfig,
    BPBoundsResult,
    CellProbabilities,
    MarginalProbabilities,
    bp_bounds_from_probs,
    run_bp_bounds,
};
pub use doubleml::{
    // Double/Debiased Machine Learning (DoubleML)
    DMLModelType,
    DoubleMLConfig,
    DoubleMLResult,
    MLMethod,
    NuisanceDiagnostics,
    TreatmentType,
    run_double_ml,
};
pub use gformula::{
    GFormulaConfig,
    GFormulaData,
    // Parametric G-Formula for causal inference with time-varying treatments (gfoRmula)
    GFormulaIntervention,
    GFormulaOutcomeType,
    GFormulaResult,
    gformula,
    run_gformula,
};
pub use hettx::{
    EffectEstimationMethod,
    EffectSummary,
    HetDecomposition,
    // Treatment Effect Heterogeneity Testing (hettx)
    HetTestStat,
    HetTxConfig,
    HetTxResult,
    run_hettx,
    run_hettx_dataset,
};
pub use ivmte::{
    IVMTEConfig,
    IVMTEResult,
    // Marginal Treatment Effects (MTE) for IV analysis (ivmte)
    MTEEstimand,
    MTEPoint,
    PropensityModel,
    PropensityStageResult,
    TreatmentEffectEstimate,
    ivmte,
    run_ivmte,
    run_ivmte_multi_z,
};
pub use sbw::{
    BalanceStats,
    SBWConfig,
    // Stable Balancing Weights (SBW)
    SBWEstimand,
    SBWResult,
    run_sbw,
    sbw,
};
pub use spatialprobit::{
    SpatialProbitConfig,
    SpatialProbitImpacts,
    // Spatial Probit Models (spatialprobit package)
    SpatialProbitModel,
    SpatialProbitResult,
    run_sar_probit,
    run_sem_probit,
};
pub use sphet::{
    SphetConfig,
    // Spatial GMM with Heteroscedasticity-Robust Estimation (sphet package)
    SphetModel,
    SphetResult,
    SphetSE,
    run_sphet,
    sphet,
};
pub use splm::{
    SpatialErrorType,
    // Spatial Panel Data Models (splm package)
    SpatialPanelEffect,
    SpatialPanelModel,
    SpatialPanelVariance,
    SpgmConfig,
    SpgmMethod,
    SpgmMoments,
    SpgmResult,
    SpmlConfig,
    SpmlResult,
    SpremlConfig,
    SpremlErrors,
    run_spgm,
    run_spml,
};
pub use stdreg::{
    SEMethod,
    StdRegConfig,
    StdRegEstimand,
    // Regression Standardization / G-computation (stdReg)
    StdRegModel,
    StdRegResult,
    SubgroupEffect,
    run_stdreg,
    stdreg,
};
