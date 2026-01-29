//! Statistical tests and descriptive statistics.
//!
//! This module provides 50+ statistical methods organized into categories:
//!
//! ## Descriptive Statistics
//!
//! - [`DescriptiveStats`] - Mean, median, std, min, max, quartiles
//! - [`fivenum`] - Five-number summary (Tukey)
//! - [`iqr`], [`mad`] - Robust spread measures
//! - [`ecdf`] - Empirical cumulative distribution
//! - [`density`] - Kernel density estimation
//!
//! ## Hypothesis Tests
//!
//! ### Location Tests
//! - [`one_sample_t_test`], [`two_sample_t_test`], [`paired_t_test`] - T-tests
//! - [`wilcoxon_rank_sum`], [`wilcoxon_signed_rank`] - Nonparametric alternatives
//! - [`kruskal_test`] - Kruskal-Wallis (nonparametric ANOVA)
//! - [`friedman_test`] - Repeated measures nonparametric
//!
//! ### Variance Tests
//! - [`bartlett_test`] - Homogeneity of variances (parametric)
//! - [`fligner_test`] - Fligner-Killeen (robust)
//! - [`var_test`] - F-test for two variances
//! - [`ansari_test`] - Ansari-Bradley scale test
//!
//! ### Distribution Tests
//! - [`shapiro_wilk_test`] - Normality test
//! - [`ks_test_one_sample`], [`ks_test_two_sample`] - Kolmogorov-Smirnov
//!
//! ### Contingency Tables
//! - [`chisq_test_gof`], [`chisq_test_independence`] - Chi-squared tests
//! - [`fisher_exact_test`] - Exact test for 2×2 tables
//! - [`mcnemar_test`] - Paired nominal data
//! - [`mantelhaen_test`] - Cochran-Mantel-Haenszel
//!
//! ### Correlation Tests
//! - [`cor_test`] - Test correlation significance
//! - [`correlation_matrix`] - Pairwise correlation matrix
//!
//! ## ANOVA & Post-hoc
//!
//! - [`run_one_way_anova`], [`run_two_way_anova`] - Analysis of variance
//! - [`oneway_test`] - Welch's ANOVA
//! - [`run_manova`] - Multivariate ANOVA
//! - [`tukey_hsd`] - Post-hoc comparisons
//! - [`pairwise_t_test`], [`pairwise_wilcox_test`] - Multiple comparisons
//!
//! ## Time Series
//!
//! - [`acf`], [`pacf`], [`ccf`] - Autocorrelation functions
//! - [`box_test`] - Ljung-Box / Box-Pierce portmanteau test
//! - [`pp_test`] - Phillips-Perron unit root test
//!
//! ## Power Analysis
//!
//! - [`power_t_test`] - Power for t-tests
//! - [`power_prop_test`] - Power for proportion tests
//! - [`power_anova_test`] - Power for ANOVA
//!
//! ## Other
//!
//! - [`mahalanobis`] - Mahalanobis distance
//! - [`isoreg`] - Isotonic regression
//! - [`medpolish`] - Median polish for two-way tables
//! - [`loglin`] - Log-linear models
//! - [`spline`], [`approx`] - Interpolation

mod acf;
mod anova;
pub mod ansari;
pub mod bartlett;
pub mod binomtest;
pub mod boxtest;
pub mod cancor;
mod chisq;
pub mod constroptim;
mod correlation;
pub mod cortest;
mod descriptive;
pub mod factanal;
pub mod fisher;
pub mod fligner;
pub mod friedman;
pub mod isoreg;
pub mod kruskal;
pub mod ks;
pub mod loglin;
pub mod mahalanobis;
pub mod manova;
pub mod mantelhaen;
pub mod mauchly;
pub mod mcnemar;
pub mod medpolish;
pub mod modeltables;
pub mod mood;
pub mod oneway;
pub mod pairwise;
pub mod poissontest;
pub mod power;
pub mod pptest;
pub mod proptest;
pub mod proptrendtest;
pub mod quade;
pub mod robust;
pub mod secontrast;
pub mod shapiro;
#[cfg(feature = "spectral-analysis")]
pub mod spectrum;
pub mod spline;
mod ttest;
pub mod tukey;
pub mod vartest;
pub mod weighted;
pub mod wilcoxon;

pub use acf::{
    AcfResult, AcfType, CcfResult, CcfType, PacfResult, acf, ccf, pacf, run_acf, run_ccf, run_pacf,
};
pub use anova::{
    AnovaResult, GroupStats, RegressionAnovaResult, TwoWayAnovaResult, anova_from_ols,
    run_one_way_anova, run_two_way_anova,
};
pub use ansari::{AnsariBradleyResult, ansari_test};
pub use bartlett::{BartlettGroupStats, BartlettResult, bartlett_test, run_bartlett_test};
pub use binomtest::{BinomTestResult, binom_test};
pub use boxtest::{BoxTestResult, BoxTestType, box_test, run_box_test};
pub use cancor::{CancorResult, cancor, run_cancor};
pub use chisq::{
    ChiSquaredResult, chisq_test_gof, chisq_test_independence, run_chisq_gof,
    run_chisq_independence,
};
pub use constroptim::{
    ConstrOptimConfig, ConstrOptimResult, OptimMethod, constr_optim, run_constr_optim,
};
pub use correlation::{CorrelationMatrix, correlation_matrix};
pub use cortest::{CorTestResult, CorrelationMethod, cor_test, run_cor_test};
pub use descriptive::{ColumnStats, DescriptiveStats};
pub use factanal::{
    FactorAnalysisConfig, FactorAnalysisResult, RotationMethod, ScoresMethod, factanal,
    factanal_from_corr, factanal_with_config, run_factanal,
};
pub use fisher::{
    FisherAlternative, FisherExactResult, fisher_exact_test, fisher_exact_test_int, run_fisher_test,
};
pub use fligner::{FlignerResult, fligner_test, run_fligner_test};
pub use friedman::{FriedmanResult, friedman_test, run_friedman_test};
pub use isoreg::{IsoregResult, isoreg, isoreg_predict, isoreg_y, run_isoreg};
pub use kruskal::{KruskalWallisResult, kruskal_test, run_kruskal_test};
pub use ks::{
    KsTestResult, TheoreticalDistribution, ks_test, ks_test_one_sample, ks_test_two_sample,
    run_ks_test,
};
pub use loglin::{LoglinResult, loglin, loglin_independence, loglin_saturated, run_loglin};
pub use mahalanobis::{MahalanobisResult, mahalanobis, mahalanobis_single, run_mahalanobis};
pub use manova::{ManovaResult, ManovaTestResult, ManovaTestStatistic, manova_one_way, run_manova};
pub use mantelhaen::{
    CmhAlternative, MantelHaenszelResult, StratumStats, Table2x2, mantelhaen_test,
    run_mantelhaen_test,
};
pub use mauchly::{MauchlyResult, mauchly_test, mauchly_test_from_slice, run_mauchly_test};
pub use mcnemar::{McnemarResult, mcnemar_test, mcnemar_test_matrix};
pub use medpolish::{MedpolishResult, medpolish, medpolish_array, run_medpolish};
pub use modeltables::{
    ModelTablesResult, ModelTablesSE, TableType, TwoWayModelTablesResult, format_model_tables,
    model_tables, model_tables_effects, model_tables_means, model_tables_two_way, run_model_tables,
};
pub use mood::{MoodTestResult, mood_test, run_mood_test};
pub use oneway::{OnewayTestResult, oneway_test, run_oneway_test};
pub use pairwise::{
    PValueAdjustMethod, PairwiseTTestResult, PairwiseWilcoxResult, p_adjust, pairwise_t_test,
    pairwise_wilcox_test, run_pairwise_t_test, run_pairwise_wilcox_test,
};
pub use poissontest::{PoissonAlternative, PoissonTestResult, poisson_test};
pub use power::{
    PowerAlternative, PowerAnovaTestResult, PowerPropTestResult, PowerTTestResult, TTestType,
    power_anova_test, power_prop_test, power_t_test, run_power_anova_test, run_power_prop_test,
    run_power_t_test,
};
pub use pptest::{PPTestResult, pp_test, run_pp_test};
pub use proptest::{PropTestResult, prop_test_k, prop_test_one, prop_test_two};
pub use proptrendtest::{PropTrendTestResult, prop_trend_test, run_prop_trend_test};
pub use quade::{QuadeResult, quade_test, run_quade_test};
pub use robust::{
    DensityKernel, DensityResult, EcdfResult, FivenumResult, density, ecdf, fivenum, iqr, mad,
    quantile, run_density, run_ecdf, run_fivenum, run_iqr, run_mad,
};
pub use secontrast::{
    ContrastType, SeContrastResult, contrast_p_value, contrast_t_statistic, estimate_contrast,
    generate_contrasts, run_se_contrast, se_contrast, se_contrast_single,
};
pub use shapiro::{ShapiroWilkResult, run_shapiro_wilk, shapiro_wilk_test};
#[cfg(feature = "spectral-analysis")]
pub use spectrum::{
    SpectrumConfig, SpectrumResult, run_spectrum, run_spectrum_ar, spectrum, spectrum_ar,
};
pub use spline::{
    ApproxMethod, ApproxResult, ApproxRule, SplineMethod, SplineResult, approx, approxfun, spline,
    splinefun,
};
pub use ttest::{
    Alternative, TTestResult, one_sample_t_test, paired_t_test, t_test, two_sample_t_test,
};
pub use tukey::{PairwiseComparison, TukeyHsdResult, run_tukey_hsd, tukey_hsd};
pub use vartest::{VarTestResult, run_var_test, var_test};
pub use weighted::{
    CovWtCenter, CovWtMethod, CovWtResult, cov_wt, cov_wt_from_slice, run_cov_wt,
    run_weighted_mean, weighted_mean,
};
pub use wilcoxon::{
    WilcoxonConfig, WilcoxonResult, wilcoxon_rank_sum, wilcoxon_signed_rank, wilcoxon_test,
};
