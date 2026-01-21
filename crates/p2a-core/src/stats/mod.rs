//! Statistical analysis module.
//!
//! Provides descriptive statistics, correlation analysis, and hypothesis tests.

mod descriptive;
mod correlation;
mod anova;
mod ttest;
mod acf;
mod chisq;
pub mod fisher;
pub mod wilcoxon;
pub mod shapiro;
pub mod ks;
pub mod manova;
pub mod tukey;
pub mod bartlett;
pub mod spectrum;
pub mod boxtest;
pub mod pptest;
pub mod factanal;
pub mod cancor;
pub mod vartest;
pub mod proptest;
pub mod binomtest;
pub mod fligner;
pub mod ansari;
pub mod mood;
pub mod kruskal;
pub mod friedman;
pub mod oneway;
pub mod mcnemar;
pub mod pairwise;
pub mod quade;
pub mod mantelhaen;
pub mod poissontest;
pub mod mahalanobis;
pub mod cortest;
pub mod power;
pub mod proptrendtest;
pub mod robust;
pub mod spline;

pub use descriptive::{DescriptiveStats, ColumnStats};
pub use correlation::{correlation_matrix, CorrelationMatrix};
pub use anova::{
    run_one_way_anova, run_two_way_anova, anova_from_ols,
    AnovaResult, TwoWayAnovaResult, RegressionAnovaResult, GroupStats,
};
pub use ttest::{
    one_sample_t_test, two_sample_t_test, paired_t_test, t_test,
    TTestResult, Alternative,
};
pub use acf::{
    acf, pacf, ccf, run_acf, run_pacf, run_ccf,
    AcfType, CcfType, AcfResult, PacfResult, CcfResult,
};
pub use chisq::{
    chisq_test_gof, chisq_test_independence,
    run_chisq_gof, run_chisq_independence,
    ChiSquaredResult,
};
pub use fisher::{
    fisher_exact_test, fisher_exact_test_int, run_fisher_test,
    FisherExactResult, FisherAlternative,
};
pub use wilcoxon::{
    wilcoxon_rank_sum, wilcoxon_signed_rank, wilcoxon_test,
    WilcoxonResult, WilcoxonConfig,
};
pub use shapiro::{
    shapiro_wilk_test, run_shapiro_wilk,
    ShapiroWilkResult,
};
pub use ks::{
    ks_test_one_sample, ks_test_two_sample, ks_test, run_ks_test,
    KsTestResult, TheoreticalDistribution,
};
pub use manova::{
    manova_one_way, run_manova,
    ManovaResult, ManovaTestResult, ManovaTestStatistic,
};
pub use tukey::{
    tukey_hsd, run_tukey_hsd,
    TukeyHsdResult, PairwiseComparison,
};
pub use bartlett::{
    bartlett_test, run_bartlett_test,
    BartlettResult, BartlettGroupStats,
};
pub use spectrum::{
    spectrum, spectrum_ar, run_spectrum, run_spectrum_ar,
    SpectrumConfig, SpectrumResult,
};
pub use boxtest::{
    box_test, run_box_test,
    BoxTestResult, BoxTestType,
};
pub use pptest::{
    pp_test, run_pp_test,
    PPTestResult,
};
pub use factanal::{
    factanal, factanal_with_config, factanal_from_corr, run_factanal,
    FactorAnalysisResult, FactorAnalysisConfig,
    RotationMethod, ScoresMethod,
};
pub use cancor::{
    cancor, run_cancor,
    CancorResult,
};
pub use vartest::{
    var_test, run_var_test,
    VarTestResult,
};
pub use proptest::{
    prop_test_one, prop_test_two, prop_test_k,
    PropTestResult,
};
pub use binomtest::{
    binom_test,
    BinomTestResult,
};
pub use fligner::{
    fligner_test, run_fligner_test,
    FlignerResult,
};
pub use ansari::{
    ansari_test,
    AnsariBradleyResult,
};
pub use mood::{
    mood_test, run_mood_test,
    MoodTestResult,
};
pub use kruskal::{
    kruskal_test, run_kruskal_test,
    KruskalWallisResult,
};
pub use friedman::{
    friedman_test, run_friedman_test,
    FriedmanResult,
};
pub use oneway::{
    oneway_test, run_oneway_test,
    OnewayTestResult,
};
pub use mcnemar::{
    mcnemar_test, mcnemar_test_matrix,
    McnemarResult,
};
pub use pairwise::{
    pairwise_t_test, run_pairwise_t_test, p_adjust,
    pairwise_wilcox_test, run_pairwise_wilcox_test,
    PairwiseTTestResult, PairwiseWilcoxResult, PValueAdjustMethod,
};
pub use quade::{
    quade_test, run_quade_test,
    QuadeResult,
};
pub use mantelhaen::{
    mantelhaen_test, run_mantelhaen_test,
    MantelHaenszelResult, StratumStats, CmhAlternative, Table2x2,
};
pub use poissontest::{
    poisson_test,
    PoissonTestResult, PoissonAlternative,
};
pub use mahalanobis::{
    mahalanobis, mahalanobis_single, run_mahalanobis,
    MahalanobisResult,
};
pub use cortest::{
    cor_test, run_cor_test,
    CorTestResult, CorrelationMethod,
};
pub use power::{
    power_t_test, power_prop_test, power_anova_test,
    run_power_t_test, run_power_prop_test, run_power_anova_test,
    PowerTTestResult, PowerPropTestResult, PowerAnovaTestResult,
    TTestType, PowerAlternative,
};
pub use proptrendtest::{
    prop_trend_test, run_prop_trend_test,
    PropTrendTestResult,
};
pub use robust::{
    fivenum, run_fivenum, FivenumResult,
    iqr, run_iqr, quantile,
    mad, run_mad,
    ecdf, run_ecdf, EcdfResult,
    density, run_density, DensityResult, DensityKernel,
};
pub use spline::{
    spline, splinefun, approx, approxfun,
    SplineResult, SplineMethod, ApproxResult, ApproxMethod, ApproxRule,
};
