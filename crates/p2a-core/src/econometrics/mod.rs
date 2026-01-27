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

mod panel;
mod iv;
mod did;
mod staggered_did;
mod etwfe;
mod general_gmm;
mod discrete;
mod timeseries;
mod hdfe;
mod feglm;
mod treatment;
mod mediation;
mod medflex;
mod synth;
mod scpi;
mod rd;
mod rdmulti;
mod survival;
mod panel_unit_root;
mod spatial;
mod cbps;
mod bacon;
mod tmle;
mod ctmle;
mod ltmle;
mod weightit;
mod matching;
mod doubleml;
mod twang;
mod bpbounds;
mod sbw;
mod ivmte;
mod hettx;
mod stdreg;
mod gformula;
mod spatialprobit;
mod splm;
mod sphet;

pub use panel::{
    PanelResult, PanelMethod, HausmanResult,
    run_fixed_effects, run_random_effects, run_hausman_test,
    // Panel GLS (FGLS)
    PanelGlsResult, PanelGlsModel, run_panel_gls, run_fegls, run_pooled_gls,
    // Arellano-Bond / System GMM
    GmmResult, GmmConfig, GmmTransform, GmmStep,
    run_gmm, run_arellano_bond,
    // Variable Coefficients Model (pvcm) and Mean Group (pmg)
    PvcmResult, PvcmType, run_pvcm, run_pmg,
};
pub use iv::{
    IVResult, run_iv2sls, FirstStageDiagnostics, run_first_stage_diagnostics,
    // Sargan test for overidentifying restrictions
    SarganTestResult, sargan_test, run_sargan_test,
};
pub use did::{DiDResult, run_did};
pub use staggered_did::{
    ComparisonGroup, AttEstimationMethod, StaggeredDidConfig, Aggregation,
    GroupTimeATT, AggregatedEffect, PreTrendTest, StaggeredDidResult,
    run_staggered_did,
};
pub use etwfe::{
    EtwfeConfig, EtwfeResult, ControlGroup, CohortTimeEffect,
    AggregatedEffect as EtwfeAggregatedEffect, run_etwfe,
};
pub use general_gmm::{
    GmmMethod, GmmVcov, GeneralGmmConfig, MomentCondition, GeneralGmmResult,
    run_general_gmm, run_gmm_iv,
};
pub use discrete::{
    DiscreteModelType,
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
};
pub use timeseries::{
    VarResult, VarmaResult, VecmResult, VarIrfResult,
    run_var, run_varma, run_vecm, run_var_irf,
    // Granger causality test
    GrangerResult, granger_test, run_granger_test, granger_test_bidirectional,
};
pub use hdfe::{HdfeResult, HdfeConfig, FactorInfo, run_hdfe};
pub use feglm::{GlmFamily, FeglmConfig, FeglmResult, run_feglm};
pub use treatment::{
    Estimand, DRMethod, IpwConfig, DoublyRobustConfig,
    PropensityScoreSummary, IpwResult, DoublyRobustResult,
    run_ipw_treatment, run_doubly_robust,
};
pub use mediation::{MediationConfig, MediationResult, run_mediation_analysis};
pub use medflex::{
    // Natural Effect Models for mediation with exposure-mediator interactions (medflex)
    EffectScale, MedflexConfig, MedflexResult,
    run_medflex, run_medflex_dataset,
};
pub use synth::{
    SynthConfig, SynthResult, PredictorSpec, PredictorBalance, TimeEffect,
    PlaceboResults, VOptimization, TimeAggregation, run_synthetic_control,
    // Generalized synthetic control (gsynth)
    GsynthConfig, GsynthResult, GsynthEstimator, GsynthForce, UnitEffect,
    run_gsynth,
};
pub use scpi::{
    // Synthetic Control with Prediction Intervals (SCPI)
    SCPIConstraint, SCPIConfig, VarianceMethod, PredictionInterval, SCPIResult,
    run_scpi,
};
pub use rd::{
    KernelType, BandwidthMethod, VceType, RdConfig, RdBandwidth,
    RdResult, FuzzyRdResult, run_rd, rd_bandwidth, run_fuzzy_rd,
};
pub use rdmulti::{
    // Multi-cutoff RD (rdmulti)
    RdMultiBandwidth, PoolingWeights, RdMultiConfig, CutoffResult, HeterogeneityTest,
    RdMultiResult, run_rd_multi, run_rd_multi_dataset,
};
pub use survival::{
    TiesMethod, AftDistribution, KaplanMeierResult, LogRankResult,
    CoxConfig, CoxResult, AftConfig, AftResult,
    CumulativeIncidence, CompetingRisksResult,
    run_kaplan_meier, log_rank_test, run_cox_ph, run_aft, run_competing_risks,
};
pub use panel_unit_root::{
    PanelUnitRootTest, PanelModel, PanelUnitRootConfig, PanelUnitRootResult,
    run_panel_unit_root,
};
pub use spatial::{
    SarConfig, SemConfig, SacConfig, SpatialImpacts, SarResult, SemResult, SacResult,
    run_sar, run_sem, run_sac, run_sar_dataset, run_sem_dataset, run_sac_dataset,
};
pub use cbps::{
    CbpsMethod, CbpsConfig, CbpsResult, BalanceTable, CovariateBalance,
    run_cbps, cbps,
};
pub use bacon::{
    // Goodman-Bacon decomposition for staggered DiD
    ComparisonType, BaconComponent, BaconEstimatesByType, BaconDecompResult,
    bacon_decomp,
};
pub use tmle::{
    // Targeted Maximum Likelihood Estimation (TMLE)
    QModel, GModel, TmleConfig, TmleResult,
    tmle, run_tmle,
};
pub use ctmle::{
    // Collaborative Targeted Maximum Likelihood Estimation (C-TMLE)
    StoppingRule, SelectionOrder, CVCriterion, CTmleQModel,
    CTmleConfig, SelectionStep, CTmleResult, CTmleConfigSummary,
    ctmle, run_ctmle, ctmle_arrays,
};
pub use ltmle::{
    // Longitudinal Targeted Maximum Likelihood Estimation (LTMLE)
    InterventionType, LtmleQModel, LtmleConfig, LtmleData, LtmleResult,
    ltmle, run_ltmle,
};
pub use weightit::{
    // Flexible inverse probability weighting (WeightIt)
    WeightMethod, WeightEstimand, WeightItConfig, WeightItResult,
    WeightItBalanceTable, WeightItCovariateBalance, EntropyBalanceResult,
    weightit, entropy_balance,
};
pub use matching::{
    // Propensity Score Matching (MatchIt)
    MatchMethod, DistanceMethod, MatchCovariateBalance, MatchBalanceTable,
    MatchInfo, SubclassInfo, MatchResult,
    match_it, nearest_neighbor_match, cem_match, full_match, subclass_match,
};
pub use twang::{
    // GBM-based propensity score estimation (twang)
    StopMethod, TwangEstimand, TwangConfig, TwangCovariateBalance, TwangBalanceTable,
    DecisionStump, TwangResult, run_twang, twang,
};

pub use doubleml::{
    // Double/Debiased Machine Learning (DoubleML)
    DMLModelType, TreatmentType, MLMethod, DoubleMLConfig,
    NuisanceDiagnostics, DoubleMLResult,
    run_double_ml,
};
pub use bpbounds::{
    // Balke-Pearl bounds for nonparametric IV
    BPBoundsConfig, CellProbabilities, MarginalProbabilities, BPBoundsResult,
    run_bp_bounds, bp_bounds_from_probs,
};
pub use sbw::{
    // Stable Balancing Weights (SBW)
    SBWEstimand, SBWConfig, BalanceStats, SBWResult,
    run_sbw, sbw,
};
pub use ivmte::{
    // Marginal Treatment Effects (MTE) for IV analysis (ivmte)
    MTEEstimand, PropensityModel, IVMTEConfig, MTEPoint,
    TreatmentEffectEstimate, PropensityStageResult, IVMTEResult,
    run_ivmte, ivmte, run_ivmte_multi_z,
};
pub use hettx::{
    // Treatment Effect Heterogeneity Testing (hettx)
    HetTestStat, EffectEstimationMethod, HetTxConfig,
    HetDecomposition, EffectSummary, HetTxResult,
    run_hettx, run_hettx_dataset,
};
pub use stdreg::{
    // Regression Standardization / G-computation (stdReg)
    StdRegModel, StdRegEstimand, SEMethod, StdRegConfig,
    SubgroupEffect, StdRegResult,
    run_stdreg, stdreg,
};
pub use gformula::{
    // Parametric G-Formula for causal inference with time-varying treatments (gfoRmula)
    GFormulaIntervention, GFormulaOutcomeType, GFormulaConfig, GFormulaData, GFormulaResult,
    run_gformula, gformula,
};
pub use spatialprobit::{
    // Spatial Probit Models (spatialprobit package)
    SpatialProbitModel, SpatialProbitConfig, SpatialProbitImpacts, SpatialProbitResult,
    run_sar_probit, run_sem_probit,
};
pub use splm::{
    // Spatial Panel Data Models (splm package)
    SpatialPanelEffect, SpatialPanelModel, SpatialErrorType,
    SpmlConfig, SpmlResult, run_spml,
    SpgmMethod, SpgmMoments, SpgmConfig, SpgmResult, run_spgm,
    SpremlErrors, SpremlConfig, SpatialPanelVariance,
};
pub use sphet::{
    // Spatial GMM with Heteroscedasticity-Robust Estimation (sphet package)
    SphetModel, SphetSE, SphetConfig, SphetResult,
    run_sphet, sphet,
};
