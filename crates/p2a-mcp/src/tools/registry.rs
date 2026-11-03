//! Tool Registry - Metadata for all MCP tools
//!
//! The registry is generated from the tool router: every routed tool's name, description and
//! input schema, plus one hand-kept [`CATEGORY_TABLE`] row per tool (category, related tools,
//! R counterpart). `search_tools`, `tool_info`, `list_tool_categories` and
//! `docs/mcp/tools-reference.md` read it, so they list exactly the tools this build routes.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::OnceLock;

/// Category of an analytics tool
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ToolCategory {
    /// Data loading, management, and export
    Data,
    /// Data quality profiling and cleaning
    Cleaning,
    /// Data transformation and munging
    Munging,
    /// Descriptive statistics and correlation
    Descriptive,
    /// Hypothesis tests and power analysis
    Statistics,
    /// OLS, GLS, NLS, quantile regression
    Regression,
    /// Panel data models (FE, RE, GMM)
    Panel,
    /// Instrumental variables and 2SLS
    IV,
    /// Difference-in-differences, synthetic control
    DiD,
    /// Regression discontinuity
    RD,
    /// Propensity score matching and weighting
    Matching,
    /// IPW, doubly robust, TMLE
    Treatment,
    /// Mediation analysis
    Mediation,
    /// Logit, probit, multinomial, count models
    Discrete,
    /// ARIMA, VAR, GARCH, time series utilities
    TimeSeries,
    /// Spatial regression and diagnostics
    Spatial,
    /// Survival analysis (Kaplan-Meier, Cox, AFT)
    Survival,
    /// Clustering, PCA, t-SNE, Random Forest
    MachineLearning,
    /// Plots and charts (static and interactive)
    Visualization,
    /// SQLite and DuckDB queries
    Database,
    /// Seed, reports, session management
    Utility,
}

impl ToolCategory {
    /// Get a human-readable description of the category
    pub fn description(&self) -> &'static str {
        match self {
            Self::Data => "Data loading, listing, describing, and export",
            Self::Cleaning => "Data quality profiling, cleaning sessions, rollback",
            Self::Munging => "Data transformation: filter, join, pivot, mutate",
            Self::Descriptive => "Descriptive statistics, correlation, ANOVA",
            Self::Statistics => "Hypothesis tests, power analysis, multivariate",
            Self::Regression => "OLS, GLS, NLS, quantile, robust standard errors",
            Self::Panel => "Fixed effects, random effects, Hausman, GMM, HDFE",
            Self::IV => "Instrumental variables, 2SLS, Sargan, bounds",
            Self::DiD => "Difference-in-differences, staggered DiD, synthetic control",
            Self::RD => "Regression discontinuity (sharp, fuzzy, multi-cutoff)",
            Self::Matching => "Propensity score matching, CEM, nearest neighbor",
            Self::Treatment => "IPW, doubly robust, TMLE, CBPS, entropy balancing",
            Self::Mediation => "Causal mediation, natural effects",
            Self::Discrete => "Logit, probit, multinomial, ordered, count models",
            Self::TimeSeries => "ARIMA, VAR, GARCH, Kalman, changepoint detection",
            Self::Spatial => "SAR, SEM, Moran's I, spatial weights, panel spatial",
            Self::Survival => "Kaplan-Meier, Cox PH, AFT, competing risks",
            Self::MachineLearning => "K-means, DBSCAN, PCA, t-SNE, Random Forest, SVM",
            Self::Visualization => "Histograms, scatter plots, heatmaps, interactive charts",
            Self::Database => "SQLite and DuckDB queries, schema inspection",
            Self::Utility => "Random seed, reports, session export/import",
        }
    }

    /// Get all categories
    pub fn all() -> &'static [ToolCategory] {
        use ToolCategory::*;
        &[
            Data,
            Cleaning,
            Munging,
            Descriptive,
            Statistics,
            Regression,
            Panel,
            IV,
            DiD,
            RD,
            Matching,
            Treatment,
            Mediation,
            Discrete,
            TimeSeries,
            Spatial,
            Survival,
            MachineLearning,
            Visualization,
            Database,
            Utility,
        ]
    }
}

/// Metadata for a single tool: the router's name, description and input schema, plus the
/// tool's [`CATEGORY_TABLE`] row (category, related tools, R counterpart).
#[derive(Debug, Clone, Serialize)]
pub struct ToolInfo {
    /// Tool name (MCP tool identifier), from the router
    pub name: String,
    /// Primary category, from [`CATEGORY_TABLE`]
    pub category: ToolCategory,
    /// The tool's description (`#[tool(description)]`), from the router
    pub description: String,
    /// Related tools (for discovery), from [`CATEGORY_TABLE`]
    #[serde(serialize_with = "serialize_static_strs")]
    pub related: &'static [&'static str],
    /// The R function the tool reproduces or names (its R counterpart), from
    /// [`CATEGORY_TABLE`]; a discovery hint, not a claim of numerical equality
    pub r_equivalent: Option<&'static str>,
    /// The tool's JSON input schema, from the router
    pub input_schema: serde_json::Value,
}

fn serialize_static_strs<S>(
    value: &&'static [&'static str],
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::ser::SerializeSeq;
    let mut seq = serializer.serialize_seq(Some(value.len()))?;
    for s in *value {
        seq.serialize_element(s)?;
    }
    seq.end()
}

use ToolCategory as C;

/// One row per routed tool, sorted by name: (name, category, related tools, R counterpart).
///
/// The registry's names, descriptions and schemas come from the tool router; this table adds
/// only what the router does not know. A new tool needs a row here (the tests fail
/// otherwise), and every `related` name must be a routed tool.
#[rustfmt::skip]
pub static CATEGORY_TABLE: &[(&str, ToolCategory, &[&str], Option<&str>)] = &[
    ("aft", C::Survival, &["cox_ph", "kaplan_meier"], Some("survival::survreg()")),
    ("anova_manova", C::Statistics, &["anova_one_way", "anova_two_way"], Some("manova()")),
    ("anova_one_way", C::Statistics, &["anova_two_way", "anova_tukey_hsd"], Some("aov()")),
    ("anova_tukey_hsd", C::Statistics, &["anova_one_way", "hypothesis_pairwise_t_test"], Some("TukeyHSD()")),
    ("anova_two_way", C::Statistics, &["anova_one_way", "anova_tukey_hsd"], Some("aov()")),
    ("bacon_decomp", C::DiD, &["staggered_did"], Some("bacondecomp::bacon()")),
    ("batch_process", C::Munging, &["describe_dataset", "regression_ols"], None),
    ("bp_bounds", C::IV, &["iv_2sls", "iv_mte"], Some("bpbounds package")),
    ("causal_impact_analysis", C::TimeSeries, &["ts_mstl", "synthetic_control"], Some("CausalImpact::CausalImpact()")),
    ("cleaning_rollback", C::Cleaning, &["cleaning_session_start"], None),
    ("cleaning_session_apply", C::Cleaning, &["cleaning_session_start", "cleaning_rollback"], None),
    ("cleaning_session_checkpoints", C::Cleaning, &["cleaning_session_status", "cleaning_rollback"], None),
    ("cleaning_session_start", C::Cleaning, &["cleaning_session_apply", "cleaning_rollback"], None),
    ("cleaning_session_status", C::Cleaning, &["cleaning_session_checkpoints", "list_cleaning_sessions"], None),
    ("collaborative_tmle", C::Treatment, &["treatment_tmle", "ltmle"], Some("ctmle package")),
    ("compare_datasets", C::Data, &["describe_dataset"], Some("all.equal()")),
    ("competing_risks", C::Survival, &["cox_ph", "kaplan_meier"], Some("cmprsk::cuminc()")),
    ("compute_correlation", C::Descriptive, &["describe_dataset", "viz_heatmap"], Some("cor()")),
    ("cox_ph", C::Survival, &["kaplan_meier", "aft"], Some("survival::coxph()")),
    ("create_dataset", C::Data, &["load_dataset", "generate_random_data"], None),
    ("data_quality_profile", C::Cleaning, &["suggest_cleaning", "preview_cleaning"], None),
    ("db_duckdb_query", C::Database, &["db_duckdb_tables"], Some("duckdb::dbGetQuery()")),
    ("db_duckdb_schema", C::Database, &["db_duckdb_tables", "db_duckdb_query"], Some("DBI::dbListFields()")),
    ("db_duckdb_tables", C::Database, &["db_duckdb_schema", "db_duckdb_query"], Some("DBI::dbListTables()")),
    ("db_query_file", C::Database, &["db_duckdb_query"], None),
    ("db_sqlite_query", C::Database, &["db_sqlite_tables", "db_duckdb_query"], Some("DBI::dbGetQuery()")),
    ("db_sqlite_schema", C::Database, &["db_sqlite_tables", "db_sqlite_query"], Some("DBI::dbListFields()")),
    ("db_sqlite_tables", C::Database, &["db_sqlite_schema", "db_sqlite_query"], Some("DBI::dbListTables()")),
    ("describe_dataset", C::Data, &["list_datasets", "head_dataset"], Some("summary()")),
    ("descriptive_isoreg", C::Descriptive, &["regression_line"], Some("isoreg()")),
    ("descriptive_medpolish", C::Descriptive, &["anova_two_way"], Some("medpolish()")),
    ("diff_in_diff", C::DiD, &["staggered_did", "etwfe"], Some("Basic DiD in lm()")),
    ("etwfe", C::DiD, &["diff_in_diff", "staggered_did"], Some("etwfe package")),
    ("evalue", C::Regression, &["sensemakr"], Some("EValue package")),
    ("export_dataset", C::Data, &["load_dataset"], Some("write.csv, write_parquet")),
    ("export_session", C::Utility, &["import_session"], Some("save(), saveRDS()")),
    ("feglm", C::Discrete, &["logit", "panel_hdfe"], Some("fixest::feglm()")),
    ("generate_random_data", C::Utility, &["set_seed", "create_dataset"], Some("Various rnorm, runif, etc.")),
    ("generate_report", C::Utility, &["export_session"], Some("rmarkdown::render()")),
    ("get_seed", C::Utility, &["set_seed"], None),
    ("gformula", C::Treatment, &["ltmle", "regression_standardization"], Some("gfoRmula::gformula()")),
    ("gmm_iv", C::IV, &["iv_2sls", "panel_gmm"], Some("gmm::gmm()")),
    ("gsynth", C::DiD, &["synthetic_control"], Some("gsynth package")),
    ("hausman_test", C::Panel, &["panel_fixed_effects", "panel_random_effects"], Some("plm::phtest()")),
    ("head_dataset", C::Data, &["describe_dataset"], Some("head()")),
    ("heterogeneity_test", C::Treatment, &["ml_causal_forest", "ml_bart_causal"], Some("hettx::detect_idiosyncratic()")),
    ("hurdle_model", C::Discrete, &["zeroinfl", "negbin"], Some("pscl::hurdle()")),
    ("hypothesis_bartlett_test", C::Statistics, &["anova_one_way", "hypothesis_oneway"], Some("bartlett.test()")),
    ("hypothesis_chisq_gof", C::Statistics, &["hypothesis_chisq_independence"], Some("chisq.test()")),
    ("hypothesis_chisq_independence", C::Statistics, &["hypothesis_chisq_gof", "hypothesis_fisher_exact"], Some("chisq.test()")),
    ("hypothesis_cor_test", C::Statistics, &["compute_correlation"], Some("cor.test()")),
    ("hypothesis_fisher_exact", C::Statistics, &["hypothesis_chisq_independence", "hypothesis_mcnemar"], Some("fisher.test()")),
    ("hypothesis_friedman", C::Statistics, &["hypothesis_quade", "hypothesis_kruskal_wallis"], Some("friedman.test()")),
    ("hypothesis_kruskal_wallis", C::Statistics, &["anova_one_way", "hypothesis_pairwise_wilcox"], Some("kruskal.test()")),
    ("hypothesis_ks_test", C::Statistics, &["hypothesis_shapiro_wilk"], Some("ks.test()")),
    ("hypothesis_mantelhaen", C::Statistics, &["hypothesis_chisq_independence"], Some("mantelhaen.test()")),
    ("hypothesis_mcnemar", C::Statistics, &["hypothesis_fisher_exact"], Some("mcnemar.test()")),
    ("hypothesis_mood_test", C::Statistics, &["hypothesis_bartlett_test"], Some("mood.test()")),
    ("hypothesis_oneway", C::Statistics, &["anova_one_way", "hypothesis_bartlett_test"], Some("oneway.test()")),
    ("hypothesis_pairwise_t_test", C::Statistics, &["anova_tukey_hsd", "hypothesis_pairwise_wilcox"], Some("pairwise.t.test()")),
    ("hypothesis_pairwise_wilcox", C::Statistics, &["hypothesis_pairwise_t_test", "hypothesis_kruskal_wallis"], Some("pairwise.wilcox.test()")),
    ("hypothesis_poisson", C::Statistics, &["hypothesis_prop_trend_test"], Some("poisson.test()")),
    ("hypothesis_prop_trend_test", C::Statistics, &["hypothesis_chisq_independence"], Some("prop.trend.test()")),
    ("hypothesis_quade", C::Statistics, &["hypothesis_friedman"], Some("quade.test()")),
    ("hypothesis_shapiro_wilk", C::Statistics, &["hypothesis_ks_test", "hypothesis_t_test"], Some("shapiro.test()")),
    ("hypothesis_t_test", C::Statistics, &["hypothesis_wilcoxon"], Some("t.test()")),
    ("hypothesis_wilcoxon", C::Statistics, &["hypothesis_t_test"], Some("wilcox.test()")),
    ("import_session", C::Utility, &["export_session"], Some("load()")),
    ("iv_2sls", C::IV, &["iv_first_stage", "iv_sargan_test"], Some("AER::ivreg()")),
    ("iv_first_stage", C::IV, &["iv_2sls"], Some("lmtest::waldtest()")),
    ("iv_mte", C::IV, &["iv_2sls", "bp_bounds"], Some("ivmte package")),
    ("iv_sargan_test", C::IV, &["iv_2sls"], Some("AER::summary.ivreg()")),
    ("kaplan_meier", C::Survival, &["cox_ph", "log_rank"], Some("survival::survfit()")),
    ("linalg_toeplitz", C::TimeSeries, &["timeseries_acf"], Some("toeplitz()")),
    ("list_cleaning_sessions", C::Cleaning, &["cleaning_session_start", "cleaning_session_status"], None),
    ("list_datasets", C::Data, &["load_dataset", "describe_dataset"], Some("ls()")),
    ("list_tool_categories", C::Utility, &["search_tools"], None),
    ("load_dataset", C::Data, &["create_dataset", "list_datasets"], Some("read.csv, read_parquet, read_dta")),
    ("log_rank", C::Survival, &["kaplan_meier", "cox_ph"], Some("survival::survdiff()")),
    ("logit", C::Discrete, &["probit", "multinom"], Some("glm(family=binomial)")),
    ("ltmle", C::Treatment, &["treatment_tmle"], Some("ltmle::ltmle()")),
    ("marginal_effects", C::Regression, &["logit", "probit"], Some("marginaleffects::avg_slopes()")),
    ("mediation_analysis", C::Mediation, &["natural_effects_mediation"], Some("mediation::mediate()")),
    ("mixed_logit", C::Discrete, &["mlogit", "multinom"], Some("gmnl::gmnl()")),
    ("ml_bart_causal", C::MachineLearning, &["ml_causal_forest", "heterogeneity_test"], Some("bartCause::bartc()")),
    ("ml_c50", C::MachineLearning, &["ml_ctree", "ml_cubist"], Some("C50::C5.0()")),
    ("ml_causal_forest", C::MachineLearning, &["ml_random_forest"], Some("grf::causal_forest()")),
    ("ml_cmdscale", C::MachineLearning, &["ml_pca", "ml_tsne"], Some("cmdscale()")),
    ("ml_ctree", C::MachineLearning, &["ml_c50", "ml_random_forest"], Some("partykit::ctree()")),
    ("ml_cubist", C::MachineLearning, &["ml_c50", "ml_random_forest"], Some("Cubist::cubist()")),
    ("ml_cutree", C::MachineLearning, &["ml_hierarchical", "viz_dendrogram"], Some("cutree()")),
    ("ml_dbscan", C::MachineLearning, &["ml_kmeans"], Some("dbscan::dbscan()")),
    ("ml_hierarchical", C::MachineLearning, &["ml_kmeans", "ml_cutree"], Some("hclust()")),
    ("ml_kmeans", C::MachineLearning, &["ml_dbscan", "ml_hierarchical"], Some("kmeans()")),
    ("ml_mboost", C::MachineLearning, &["ml_random_forest", "regression_glmnet"], Some("mboost::mboost()")),
    ("ml_pca", C::MachineLearning, &["ml_tsne"], Some("prcomp()")),
    ("ml_ppr", C::MachineLearning, &["regression_loess", "ml_random_forest"], Some("ppr()")),
    ("ml_random_forest", C::MachineLearning, &["ml_svm", "ml_causal_forest"], None),
    ("ml_shap_values", C::MachineLearning, &["ml_random_forest"], None),
    ("ml_svm", C::MachineLearning, &["logit", "ml_random_forest"], Some("e1071::svm(kernel = \"linear\")")),
    ("ml_tsne", C::MachineLearning, &["ml_pca"], Some("Rtsne::Rtsne()")),
    ("mlogit", C::Discrete, &["multinom", "mixed_logit"], Some("mlogit::mlogit()")),
    ("moran_test", C::Spatial, &["spatial_lm_tests_tool"], Some("spdep::moran.test()")),
    ("multinom", C::Discrete, &["mlogit", "ordered_model"], Some("nnet::multinom()")),
    ("multivariate_cancor", C::Statistics, &["compute_correlation", "multivariate_factanal"], Some("cancor()")),
    ("multivariate_factanal", C::Statistics, &["ml_pca", "multivariate_cancor"], Some("factanal()")),
    ("multivariate_mahalanobis", C::Statistics, &["stats_cov_wt"], Some("mahalanobis()")),
    ("munge_bin", C::Munging, &["munge_mutate", "munge_value_counts"], Some("cut()")),
    ("munge_cast", C::Munging, &["str_replace_value", "munge_mutate"], Some("as.numeric(), as.integer(), as.character(), as.logical()")),
    ("munge_concat", C::Munging, &["munge_join"], Some("rbind()")),
    ("munge_deduplicate", C::Munging, &["munge_filter"], Some("dplyr::distinct()")),
    ("munge_diff", C::Munging, &["munge_lag_lead", "timeseries_diffinv"], Some("diff()")),
    ("munge_drop_columns", C::Munging, &["munge_select", "munge_rename"], Some("dplyr::select()")),
    ("munge_drop_na", C::Munging, &["munge_fill_na"], Some("tidyr::drop_na()")),
    ("munge_fill_na", C::Munging, &["munge_drop_na"], Some("tidyr::fill()")),
    ("munge_filter", C::Munging, &["munge_select", "munge_sort"], Some("dplyr::filter()")),
    ("munge_group_by", C::Munging, &["munge_value_counts"], Some("dplyr::group_by() %>% summarize()")),
    ("munge_join", C::Munging, &["munge_concat"], Some("dplyr::left_join()")),
    ("munge_lag_lead", C::Munging, &["munge_diff", "timeseries_lag"], Some("dplyr::lag(), dplyr::lead()")),
    ("munge_melt", C::Munging, &["munge_pivot"], Some("tidyr::pivot_longer()")),
    ("munge_mutate", C::Munging, &["munge_standardize", "munge_bin"], Some("dplyr::mutate()")),
    ("munge_one_hot_encode", C::Munging, &["munge_bin"], Some("model.matrix()")),
    ("munge_pivot", C::Munging, &["munge_melt"], Some("tidyr::pivot_wider()")),
    ("munge_rename", C::Munging, &["munge_select", "munge_drop_columns"], Some("dplyr::rename()")),
    ("munge_sample", C::Munging, &["munge_filter"], Some("dplyr::slice_sample()")),
    ("munge_select", C::Munging, &["munge_drop_columns", "munge_rename"], Some("dplyr::select()")),
    ("munge_sort", C::Munging, &["munge_filter"], Some("dplyr::arrange()")),
    ("munge_standardize", C::Munging, &["munge_mutate"], Some("scale()")),
    ("munge_value_counts", C::Munging, &["munge_group_by", "munge_bin"], Some("table()")),
    ("natural_effects_mediation", C::Mediation, &["mediation_analysis"], Some("medflex::neModel()")),
    ("negbin", C::Discrete, &["zeroinfl", "hurdle_model"], Some("MASS::glm.nb()")),
    ("ordered_model", C::Discrete, &["multinom"], Some("MASS::polr()")),
    ("panel_fixed_effects", C::Panel, &["panel_random_effects", "hausman_test"], Some("plm::plm(model='within')")),
    ("panel_gls", C::Panel, &["panel_fixed_effects", "panel_random_effects"], Some("plm::pggls()")),
    ("panel_gmm", C::Panel, &["panel_fixed_effects", "gmm_iv"], Some("plm::pgmm()")),
    ("panel_hdfe", C::Panel, &["panel_fixed_effects", "feglm"], Some("fixest::feols()")),
    ("panel_pmg", C::Panel, &["panel_pvcm", "panel_fixed_effects"], Some("plm::pmg()")),
    ("panel_pvcm", C::Panel, &["panel_pmg", "panel_fixed_effects"], Some("plm::pvcm()")),
    ("panel_random_effects", C::Panel, &["panel_fixed_effects", "hausman_test"], Some("plm::plm(model='random')")),
    ("panel_unit_root", C::Panel, &["timeseries_pp_test", "panel_gmm"], Some("plm::purtest()")),
    ("power_anova_test", C::Statistics, &["power_t_test", "power_prop_test"], Some("power.anova.test()")),
    ("power_prop_test", C::Statistics, &["power_t_test", "power_anova_test"], Some("power.prop.test()")),
    ("power_t_test", C::Statistics, &["power_prop_test", "power_anova_test"], Some("power.t.test()")),
    ("preview_cleaning", C::Cleaning, &["verify_cleaning", "cleaning_session_apply"], None),
    ("probit", C::Discrete, &["logit"], Some("glm(family=binomial(probit))")),
    ("propensity_matching", C::Matching, &["treatment_ipw", "treatment_cbps"], Some("MatchIt::matchit()")),
    ("rd_bw", C::RD, &["rd_estimate"], Some("rdrobust::rdbwselect()")),
    ("rd_estimate", C::RD, &["rd_fuzzy", "rd_bw"], Some("rdrobust::rdrobust()")),
    ("rd_fuzzy", C::RD, &["rd_estimate"], Some("rdrobust::rdrobust(fuzzy=)")),
    ("rd_multi", C::RD, &["rd_estimate"], Some("rdmulti package")),
    ("regression_bgtest", C::Regression, &["regression_diagnostics", "regression_hac"], Some("lmtest::bgtest()")),
    ("regression_bootstrap_cov", C::Regression, &["regression_ols", "regression_hac"], Some("sandwich::vcovBS()")),
    ("regression_clustered", C::Regression, &["regression_ols", "regression_hac"], Some("sandwich::vcovCL()")),
    ("regression_cv_glmnet", C::Regression, &["regression_glmnet", "regression_lasso"], Some("glmnet::cv.glmnet()")),
    ("regression_diagnostics", C::Regression, &["regression_ols", "regression_bgtest"], Some("car::vif(), lmtest::bptest()")),
    ("regression_driscoll_kraay", C::Regression, &["regression_hac", "regression_clustered"], Some("plm::vcovSCC()")),
    ("regression_glmnet", C::Regression, &["regression_lasso", "regression_ridge", "regression_cv_glmnet"], Some("glmnet::glmnet()")),
    ("regression_gls", C::Regression, &["regression_ols"], Some("nlme::gls()")),
    ("regression_hac", C::Regression, &["regression_ols", "regression_driscoll_kraay"], Some("sandwich::vcovHAC()")),
    ("regression_harvtest", C::Regression, &["regression_resettest", "regression_diagnostics"], Some("lmtest::harvtest()")),
    ("regression_lasso", C::Regression, &["regression_glmnet", "regression_ridge"], Some("glmnet::glmnet(alpha = 1)")),
    ("regression_line", C::Regression, &["regression_ols", "regression_quantreg"], Some("line()")),
    ("regression_loess", C::Regression, &["regression_smooth_spline"], Some("loess()")),
    ("regression_nls", C::Regression, &["regression_ols"], Some("nls()")),
    ("regression_ols", C::Regression, &["regression_clustered", "regression_diagnostics"], Some("lm(), sandwich::vcovHC()")),
    ("regression_quantreg", C::Regression, &["regression_ols"], Some("quantreg::rq()")),
    ("regression_resettest", C::Regression, &["regression_harvtest", "regression_diagnostics"], Some("lmtest::resettest()")),
    ("regression_ridge", C::Regression, &["regression_glmnet", "regression_lasso"], Some("glmnet::glmnet(alpha = 0)")),
    ("regression_smooth_spline", C::Regression, &["regression_loess", "regression_supsmu"], Some("smooth.spline()")),
    ("regression_standardization", C::Treatment, &["gformula", "treatment_doubly_robust"], Some("stdReg::stdGlm()")),
    ("regression_step", C::Regression, &["regression_ols"], Some("step()")),
    ("regression_supsmu", C::Regression, &["regression_loess", "regression_smooth_spline"], Some("supsmu()")),
    ("regression_waldtest", C::Regression, &["regression_ols", "regression_resettest"], Some("lmtest::waldtest()")),
    ("sar_model", C::Spatial, &["sem_model", "moran_test"], Some("spatialreg::lagsarlm()")),
    ("sar_probit_model", C::Spatial, &["sem_probit_model", "sar_model"], Some("spatialprobit::sarprobit()")),
    ("scpi", C::DiD, &["synthetic_control", "gsynth"], Some("scpi::scpi()")),
    ("search_tools", C::Utility, &["tool_info", "list_tool_categories"], None),
    ("sem_model", C::Spatial, &["sar_model"], Some("spatialreg::errorsarlm()")),
    ("sem_probit_model", C::Spatial, &["sar_probit_model", "sem_model"], Some("spatialprobit::semprobit()")),
    ("sensemakr", C::Regression, &["evalue", "regression_ols"], Some("sensemakr::sensemakr()")),
    ("server_stats", C::Utility, &["list_datasets"], None),
    ("set_seed", C::Utility, &["generate_random_data"], Some("set.seed()")),
    ("spatial_lm_tests_tool", C::Spatial, &["moran_test", "sar_model", "sem_model"], Some("spdep::lm.LMtests()")),
    ("spatial_neighbors", C::Spatial, &["moran_test", "sar_model"], Some("spdep::knearneigh()")),
    ("spatial_panel_gmm", C::Spatial, &["spatial_panel_ml", "sphet_model"], Some("splm::spgm()")),
    ("spatial_panel_ml", C::Spatial, &["sar_model", "panel_fixed_effects"], Some("splm::spml()")),
    ("sphet_model", C::Spatial, &["sar_model", "spatial_panel_gmm"], Some("sphet::spreg()")),
    ("staggered_did", C::DiD, &["diff_in_diff", "bacon_decomp"], Some("did::att_gt()")),
    ("stats_approx", C::Descriptive, &["stats_spline"], Some("approx()")),
    ("stats_cov_wt", C::Descriptive, &["compute_correlation", "stats_weighted_mean"], Some("cov.wt()")),
    ("stats_density", C::Descriptive, &["stats_ecdf", "viz_histogram"], Some("density()")),
    ("stats_ecdf", C::Descriptive, &["stats_density"], Some("ecdf()")),
    ("stats_fivenum", C::Descriptive, &["stats_iqr", "viz_boxplot"], Some("fivenum()")),
    ("stats_iqr", C::Descriptive, &["stats_fivenum", "stats_mad"], Some("IQR()")),
    ("stats_loglin", C::Statistics, &["hypothesis_chisq_independence"], Some("loglin()")),
    ("stats_mad", C::Descriptive, &["stats_iqr"], Some("mad()")),
    ("stats_mauchly_test", C::Statistics, &["anova_manova"], Some("mauchly.test()")),
    ("stats_model_tables", C::Statistics, &["anova_one_way", "anova_two_way"], Some("model.tables()")),
    ("stats_se_contrast", C::Statistics, &["anova_one_way", "stats_model_tables"], Some("se.contrast()")),
    ("stats_spline", C::Descriptive, &["stats_approx", "regression_smooth_spline"], Some("spline()")),
    ("stats_weighted_mean", C::Descriptive, &["stats_cov_wt", "describe_dataset"], Some("weighted.mean()")),
    ("str_concat", C::Munging, &["str_split"], Some("paste()")),
    ("str_length", C::Munging, &["str_substring"], Some("nchar()")),
    ("str_regex_count", C::Munging, &["str_regex_extract", "str_regex_replace"], Some("stringr::str_count()")),
    ("str_regex_extract", C::Munging, &["str_regex_replace", "str_regex_count"], Some("stringr::str_extract()")),
    ("str_regex_replace", C::Munging, &["str_replace_value", "str_regex_extract"], Some("gsub()")),
    ("str_replace_value", C::Munging, &["str_regex_replace", "munge_cast"], None),
    ("str_split", C::Munging, &["str_concat"], Some("tidyr::separate()")),
    ("str_substring", C::Munging, &["str_length"], Some("substr()")),
    ("str_to_lowercase", C::Munging, &["str_to_uppercase", "str_trim"], Some("tolower()")),
    ("str_to_uppercase", C::Munging, &["str_to_lowercase", "str_trim"], Some("toupper()")),
    ("str_trim", C::Munging, &["str_to_lowercase"], Some("trimws()")),
    ("suggest_cleaning", C::Cleaning, &["data_quality_profile", "preview_cleaning"], None),
    ("synthetic_control", C::DiD, &["gsynth", "scpi"], Some("Synth package")),
    ("timeseries_acf", C::TimeSeries, &["timeseries_ccf", "timeseries_box_test"], Some("acf(), pacf()")),
    ("timeseries_acf_to_ar", C::TimeSeries, &["timeseries_ar", "timeseries_acf"], Some("acf2AR()")),
    ("timeseries_ar", C::TimeSeries, &["ts_arima_fit", "timeseries_acf_to_ar"], Some("ar()")),
    ("timeseries_arima_sim", C::TimeSeries, &["ts_arima_fit"], Some("arima.sim()")),
    ("timeseries_arma_acf", C::TimeSeries, &["timeseries_arma_to_ma", "timeseries_acf"], Some("ARMAacf()")),
    ("timeseries_arma_to_ma", C::TimeSeries, &["timeseries_arma_acf"], Some("ARMAtoMA()")),
    ("timeseries_box_test", C::TimeSeries, &["timeseries_acf", "regression_bgtest"], Some("Box.test()")),
    ("timeseries_ccf", C::TimeSeries, &["timeseries_acf", "ts_granger"], Some("ccf()")),
    ("timeseries_cpgram", C::TimeSeries, &["timeseries_spectrum", "timeseries_box_test"], Some("cpgram()")),
    ("timeseries_decompose", C::TimeSeries, &["ts_mstl", "timeseries_filter"], Some("decompose()")),
    ("timeseries_diffinv", C::TimeSeries, &["munge_diff", "timeseries_lag"], Some("diffinv()")),
    ("timeseries_embed", C::TimeSeries, &["timeseries_lag"], Some("embed()")),
    ("timeseries_filter", C::TimeSeries, &["timeseries_decompose", "timeseries_runmed"], Some("filter()")),
    ("timeseries_lag", C::TimeSeries, &["munge_lag_lead", "timeseries_embed"], None),
    ("timeseries_pp_test", C::TimeSeries, &["panel_unit_root"], Some("PP.test()")),
    ("timeseries_runmed", C::TimeSeries, &["timeseries_filter"], Some("runmed()")),
    ("timeseries_spectrum", C::TimeSeries, &["timeseries_cpgram"], Some("spectrum()")),
    ("timeseries_structts", C::TimeSeries, &["ts_arima_fit", "ts_holt_winters"], Some("StructTS()")),
    ("timeseries_window", C::TimeSeries, &["timeseries_lag"], Some("window()")),
    ("tool_info", C::Utility, &["search_tools"], None),
    ("treatment_cbps", C::Treatment, &["propensity_matching", "treatment_weightit"], Some("CBPS::CBPS()")),
    ("treatment_double_ml", C::Treatment, &["treatment_doubly_robust"], Some("DoubleML package")),
    ("treatment_doubly_robust", C::Treatment, &["treatment_ipw", "treatment_tmle"], Some("drtmle package")),
    ("treatment_entropy_balance", C::Treatment, &["treatment_cbps"], Some("ebal::ebalance()")),
    ("treatment_ipw", C::Treatment, &["treatment_doubly_robust", "treatment_tmle"], Some("ipw package")),
    ("treatment_sbw", C::Treatment, &["treatment_entropy_balance", "treatment_weightit"], Some("sbw::sbw()")),
    ("treatment_tmle", C::Treatment, &["treatment_doubly_robust", "collaborative_tmle"], Some("tmle::tmle()")),
    ("treatment_twang", C::Treatment, &["treatment_cbps"], Some("twang::ps()")),
    ("treatment_weightit", C::Treatment, &["treatment_cbps", "treatment_entropy_balance"], Some("WeightIt::weightit()")),
    ("ts_arima_fit", C::TimeSeries, &["ts_arima_forecast", "ts_garch_fit"], Some("arima()")),
    ("ts_arima_forecast", C::TimeSeries, &["ts_arima_fit"], Some("predict.Arima()")),
    ("ts_changepoint", C::TimeSeries, &["ts_mstl"], Some("changepoint::cpt.*()")),
    ("ts_garch_fit", C::TimeSeries, &["ts_arima_fit"], Some("rugarch::ugarchfit()")),
    ("ts_granger", C::TimeSeries, &["ts_var", "timeseries_ccf"], Some("lmtest::grangertest()")),
    ("ts_holt_winters", C::TimeSeries, &["ts_arima_fit", "timeseries_decompose"], Some("HoltWinters()")),
    ("ts_mstl", C::TimeSeries, &["timeseries_decompose"], Some("forecast::mstl()")),
    ("ts_var", C::TimeSeries, &["ts_vecm", "ts_var_irf"], Some("vars::VAR()")),
    ("ts_var_irf", C::TimeSeries, &["ts_var"], Some("vars::irf()")),
    ("ts_varma", C::TimeSeries, &["ts_var", "ts_vecm"], None),
    ("ts_vecm", C::TimeSeries, &["ts_var", "ts_varma"], Some("urca::ca.jo()")),
    ("upload_dataset", C::Data, &["load_dataset", "create_dataset"], None),
    ("verify_cleaning", C::Cleaning, &["preview_cleaning"], None),
    ("viz_boxplot", C::Visualization, &["stats_fivenum", "viz_histogram"], Some("boxplot()")),
    ("viz_coefficient", C::Visualization, &["regression_ols"], Some("coefplot::coefplot()")),
    ("viz_dendrogram", C::Visualization, &["ml_hierarchical", "ml_cutree"], Some("plot(hclust())")),
    ("viz_event_study", C::Visualization, &["staggered_did", "etwfe"], Some("fixest::iplot()")),
    ("viz_heatmap", C::Visualization, &["compute_correlation"], Some("heatmap()")),
    ("viz_histogram", C::Visualization, &["viz_histogram_interactive"], Some("hist()")),
    ("viz_histogram_interactive", C::Visualization, &["viz_histogram"], Some("plotly::plot_ly(type = \"histogram\")")),
    ("viz_irf", C::Visualization, &["ts_var_irf", "ts_var"], Some("plot(vars::irf())")),
    ("viz_line", C::Visualization, &["viz_line_interactive"], Some("plot(type='l')")),
    ("viz_line_interactive", C::Visualization, &["viz_line"], Some("plotly::plot_ly(mode = \"lines\")")),
    ("viz_residual_diagnostics", C::Visualization, &["regression_diagnostics", "regression_ols"], Some("plot(lm())")),
    ("viz_scatter", C::Visualization, &["viz_scatter_interactive"], Some("plot()")),
    ("viz_scatter_interactive", C::Visualization, &["viz_scatter"], Some("plotly::plot_ly()")),
    ("zeroinfl", C::Discrete, &["negbin", "hurdle_model"], Some("pscl::zeroinfl()")),
];

/// Rows whose tool exists only when an optional cargo feature compiles its router (D6, T3-WP53).
pub const OPTIONAL_TOOLS: &[&str] = &[
    "db_duckdb_query",
    "db_duckdb_schema",
    "db_duckdb_tables",
    "db_query_file",
    "db_sqlite_query",
    "db_sqlite_schema",
    "db_sqlite_tables",
];

/// The [`CATEGORY_TABLE`] row of `name`, if it has one.
#[allow(clippy::type_complexity)]
pub fn category_row(
    name: &str,
) -> Option<&'static (
    &'static str,
    ToolCategory,
    &'static [&'static str],
    Option<&'static str>,
)> {
    CATEGORY_TABLE.iter().find(|row| row.0 == name)
}

/// Canonical lowercase name of a category, accepted by [`parse_category`] ("timeseries",
/// "ml", "did", ...).
pub fn category_name(c: ToolCategory) -> &'static str {
    match c {
        C::Data => "data",
        C::Cleaning => "cleaning",
        C::Munging => "munging",
        C::Descriptive => "descriptive",
        C::Statistics => "statistics",
        C::Regression => "regression",
        C::Panel => "panel",
        C::IV => "iv",
        C::DiD => "did",
        C::RD => "rd",
        C::Matching => "matching",
        C::Treatment => "treatment",
        C::Mediation => "mediation",
        C::Discrete => "discrete",
        C::TimeSeries => "timeseries",
        C::Spatial => "spatial",
        C::Survival => "survival",
        C::MachineLearning => "ml",
        C::Visualization => "visualization",
        C::Database => "database",
        C::Utility => "utility",
    }
}

/// Registry entries for `tools`, sorted by name. A tool without a [`CATEGORY_TABLE`] row is
/// listed under Utility with no related tools and no R counterpart (the tests forbid that
/// state; production never panics).
pub fn build_registry(tools: &[rmcp::model::Tool]) -> Vec<ToolInfo> {
    let mut out: Vec<ToolInfo> = tools
        .iter()
        .map(|t| {
            let (category, related, r_equivalent) = match category_row(&t.name) {
                Some(&(_, c, related, r)) => (c, related, r),
                None => (C::Utility, &[][..], None),
            };
            ToolInfo {
                name: t.name.to_string(),
                category,
                description: t.description.as_deref().unwrap_or("").to_string(),
                related,
                r_equivalent,
                input_schema: serde_json::Value::Object((*t.input_schema).clone()),
            }
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// This build's registry: [`build_registry`] over the tool router's tools, built once.
pub fn registry() -> &'static [ToolInfo] {
    static REGISTRY: OnceLock<Vec<ToolInfo>> = OnceLock::new();
    REGISTRY.get_or_init(|| build_registry(&crate::server::AnalyticsServer::new().router_tools()))
}

/// Get the complete tool registry (every routed tool, sorted by name).
pub fn get_registry() -> Vec<ToolInfo> {
    registry().to_vec()
}

/// Get tools by category
pub fn tools_by_category(category: ToolCategory) -> Vec<ToolInfo> {
    registry()
        .iter()
        .filter(|t| t.category == category)
        .cloned()
        .collect()
}

/// Get category counts for summary
pub fn category_counts() -> HashMap<ToolCategory, usize> {
    let mut counts = HashMap::new();
    for tool in registry() {
        *counts.entry(tool.category).or_insert(0) += 1;
    }
    counts
}

/// Get total tool count
pub fn tool_count() -> usize {
    registry().len()
}

/// Search tools by keyword in name or description
pub fn search_tools(query: &str) -> Vec<ToolInfo> {
    let query_lower = query.to_lowercase();
    registry()
        .iter()
        .filter(|t| {
            t.name.to_lowercase().contains(&query_lower)
                || t.description.to_lowercase().contains(&query_lower)
        })
        .cloned()
        .collect()
}

/// Search result with relevance score
#[derive(Debug, Clone, Serialize)]
pub struct ToolSearchResult {
    /// The matched tool
    pub tool: ToolInfo,
    /// Relevance score (higher is better)
    pub score: f32,
    /// Keywords that matched
    pub matched_keywords: Vec<String>,
}

/// Advanced search with scoring and category filtering
pub fn search_tools_scored(
    query: &str,
    category: Option<ToolCategory>,
    limit: usize,
) -> Vec<ToolSearchResult> {
    let query_lower = query.to_lowercase();
    let query_keywords: Vec<&str> = query_lower.split_whitespace().collect();

    // Domain-specific synonyms for better matching
    let synonyms: std::collections::HashMap<&str, Vec<&str>> = [
        ("ols", vec!["regression", "linear", "least", "squares"]),
        ("fe", vec!["fixed", "effects", "within"]),
        ("re", vec!["random", "effects", "gls"]),
        (
            "iv",
            vec!["instrumental", "variables", "endogenous", "2sls"],
        ),
        ("did", vec!["difference", "differences", "treatment"]),
        ("rd", vec!["discontinuity", "regression", "sharp", "fuzzy"]),
        ("var", vec!["vector", "autoregression", "irf", "impulse"]),
        (
            "arima",
            vec!["autoregressive", "moving", "average", "forecast"],
        ),
        ("garch", vec!["volatility", "heteroskedasticity", "arch"]),
        ("hausman", vec!["specification", "test", "fe", "re"]),
        (
            "robust",
            vec!["heteroskedasticity", "hc0", "hc1", "hc2", "hc3"],
        ),
        ("cluster", vec!["clustered", "standard", "errors", "panel"]),
        ("serial", vec!["correlation", "autocorrelation", "bgtest"]),
        ("unit", vec!["root", "stationarity", "adf", "pp"]),
        ("propensity", vec!["score", "matching", "ipw", "treatment"]),
        ("synth", vec!["synthetic", "control", "counterfactual"]),
    ]
    .into_iter()
    .collect();

    // Expand query with synonyms
    let mut expanded_keywords: Vec<String> = query_keywords.iter().map(|s| s.to_string()).collect();
    for kw in &query_keywords {
        if let Some(syns) = synonyms.get(*kw) {
            expanded_keywords.extend(syns.iter().map(|s| s.to_string()));
        }
    }

    let mut results: Vec<ToolSearchResult> = Vec::new();

    for tool in registry() {
        // Filter by category if specified
        if let Some(ref cat) = category {
            if tool.category != *cat {
                continue;
            }
        }

        // Compute score
        let name_lower = tool.name.to_lowercase();
        let desc_lower = tool.description.to_lowercase();
        let name_parts: Vec<&str> = name_lower.split('_').collect();

        let mut score = 0.0f32;
        let mut matched_keywords = Vec::new();

        for kw in &expanded_keywords {
            // Exact name part match (highest weight)
            if name_parts.contains(&kw.as_str()) {
                score += 3.0;
                matched_keywords.push(kw.clone());
            }
            // Name contains keyword
            else if name_lower.contains(kw) {
                score += 2.0;
                matched_keywords.push(kw.clone());
            }
            // Description contains keyword
            else if desc_lower.contains(kw) {
                score += 1.0;
                matched_keywords.push(kw.clone());
            }
        }

        // Bonus for related tools matching query
        for related in tool.related {
            let related_lower = related.to_lowercase();
            for kw in &query_keywords {
                if related_lower.contains(*kw) {
                    score += 0.5;
                }
            }
        }

        if score > 0.0 {
            matched_keywords.sort();
            matched_keywords.dedup();
            results.push(ToolSearchResult {
                tool: tool.clone(),
                score,
                matched_keywords,
            });
        }
    }

    // Sort by score descending, ties by name (a total, deterministic order)
    results.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.tool.name.cmp(&b.tool.name))
    });

    // Return top N
    results.into_iter().take(limit).collect()
}

/// Parse category from string
pub fn parse_category(s: &str) -> Option<ToolCategory> {
    match s.to_lowercase().as_str() {
        "data" => Some(ToolCategory::Data),
        "cleaning" => Some(ToolCategory::Cleaning),
        "munging" => Some(ToolCategory::Munging),
        "descriptive" => Some(ToolCategory::Descriptive),
        "statistics" | "stats" => Some(ToolCategory::Statistics),
        "regression" => Some(ToolCategory::Regression),
        "panel" => Some(ToolCategory::Panel),
        "iv" | "instrumental" => Some(ToolCategory::IV),
        "did" | "difference" => Some(ToolCategory::DiD),
        "rd" | "discontinuity" => Some(ToolCategory::RD),
        "matching" => Some(ToolCategory::Matching),
        "treatment" => Some(ToolCategory::Treatment),
        "mediation" => Some(ToolCategory::Mediation),
        "discrete" => Some(ToolCategory::Discrete),
        "timeseries" | "time_series" | "ts" => Some(ToolCategory::TimeSeries),
        "spatial" => Some(ToolCategory::Spatial),
        "survival" => Some(ToolCategory::Survival),
        "ml" | "machine_learning" => Some(ToolCategory::MachineLearning),
        "visualization" | "viz" => Some(ToolCategory::Visualization),
        "database" | "db" => Some(ToolCategory::Database),
        "utility" | "utils" => Some(ToolCategory::Utility),
        _ => None,
    }
}

/// Generate the Markdown reference of every tool in this build (`docs/mcp/tools-reference.md`,
/// written by `cargo run -p p2a-mcp --example generate_tool_docs`).
pub fn generate_markdown_docs() -> String {
    /// A table cell: one line, with `|` escaped.
    fn cell(s: &str) -> String {
        s.replace("\r\n", " ")
            .replace(['\n', '\r'], " ")
            .replace('|', "\\|")
    }

    let mut docs = String::from(
        "# MCP Tools Reference\n\nGenerated from the MCP tool router by \
         `cargo run -p p2a-mcp --example generate_tool_docs`; do not edit by hand.\n\n",
    );
    docs.push_str(&format!("Total tools: {}\n\n", tool_count()));

    for category in ToolCategory::all() {
        let tools: Vec<&ToolInfo> = registry()
            .iter()
            .filter(|t| t.category == *category)
            .collect();
        docs.push_str(&format!("## {:?} ({} tools)\n\n", category, tools.len()));
        docs.push_str(&format!("*{}*\n\n", category.description()));
        if tools.is_empty() {
            docs.push_str("No tools in this build.\n\n");
            continue;
        }
        docs.push_str("| Tool | Description | R counterpart |\n");
        docs.push_str("|------|-------------|---------------|\n");
        for tool in tools {
            docs.push_str(&format!(
                "| `{}` | {} | {} |\n",
                tool.name,
                cell(&tool.description),
                cell(tool.r_equivalent.unwrap_or("-"))
            ));
        }
        docs.push('\n');
    }

    docs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::AnalyticsServer;
    use std::collections::{BTreeMap, BTreeSet};

    fn routed() -> BTreeSet<String> {
        AnalyticsServer::new()
            .router_tools()
            .into_iter()
            .map(|t| t.name.to_string())
            .collect()
    }

    #[test]
    fn test_search() {
        let results = search_tools("regression");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_every_routed_tool_is_in_the_registry_once() {
        let mut n: BTreeMap<String, usize> = BTreeMap::new();
        for t in get_registry() {
            *n.entry(t.name.to_string()).or_default() += 1;
        }
        let bad: Vec<_> = routed()
            .into_iter()
            .filter(|r| n.get(r) != Some(&1))
            .collect();
        assert!(
            bad.is_empty(),
            "{} routed tools missing or duplicated in the registry: {bad:?}",
            bad.len()
        );
    }

    #[test]
    fn test_registry_names_and_related_tools_are_routed() {
        let routed = routed();
        let bad: Vec<String> = get_registry()
            .iter()
            .flat_map(|t| {
                std::iter::once(t.name.to_string())
                    .chain(t.related.iter().map(|r| format!("{} -> {r}", t.name)))
            })
            .filter(|s| !routed.contains(s.rsplit(" -> ").next().unwrap()))
            .collect();
        assert!(bad.is_empty(), "registry names that are not tools: {bad:?}");
    }

    #[test]
    fn test_category_table_rows_name_routed_or_optional_tools() {
        let routed = routed();
        let mut seen = BTreeSet::new();
        for (name, _, related, _) in CATEGORY_TABLE {
            assert!(seen.insert(*name), "duplicate row {name}");
            assert!(
                routed.contains(*name) || OPTIONAL_TOOLS.contains(name),
                "row for a tool that is not routed: {name}"
            );
            for r in *related {
                assert!(
                    r != name && (routed.contains(*r) || OPTIONAL_TOOLS.contains(r)),
                    "{name}: related {r}"
                );
            }
        }
        for o in OPTIONAL_TOOLS {
            assert!(seen.contains(o), "optional tool {o} has no row");
        }
        for r in &routed {
            assert!(
                seen.contains(r.as_str()),
                "routed tool {r} has no category row"
            );
        }
        // sorted by name, as the table's doc says
        let names: Vec<&str> = CATEGORY_TABLE.iter().map(|r| r.0).collect();
        assert!(
            names.windows(2).all(|w| w[0] < w[1]),
            "CATEGORY_TABLE is not sorted"
        );
    }

    #[test]
    fn test_category_counts_sum_to_routed_tools() {
        assert_eq!(category_counts().values().sum::<usize>(), routed().len());
        assert_eq!(tool_count(), routed().len());
    }

    /// The finding's failure scenarios (ranks computed in planning with a mirror of
    /// search_tools_scored).
    #[test]
    fn test_search_finds_the_tools_the_old_registry_missed() {
        let routed = routed();
        for (q, first) in [
            ("kaplan meier survival", "kaplan_meier"),
            ("shap values", "ml_shap_values"),
            ("dendrogram", "viz_dendrogram"),
            ("serial correlation test panel", "regression_bgtest"),
        ] {
            let r = search_tools_scored(q, None, 5);
            assert_eq!(
                r[0].tool.name,
                first,
                "{q}: {:?}",
                r.iter().map(|x| &x.tool.name).collect::<Vec<_>>()
            );
            assert!(r.iter().all(|x| routed.contains(&x.tool.name)), "{q}");
        }
        assert!(
            search_tools_scored("kaplan meier", None, 25)
                .iter()
                .all(|x| !x.tool.name.starts_with("survival_"))
        );
    }

    #[test]
    fn test_search_ranking_is_deterministic() {
        let a: Vec<_> = search_tools_scored("test", None, 25)
            .into_iter()
            .map(|r| (r.tool.name, r.score))
            .collect();
        let b: Vec<_> = search_tools_scored("test", None, 25)
            .into_iter()
            .map(|r| (r.tool.name, r.score))
            .collect();
        assert_eq!(a, b);
        assert!(
            a.windows(2)
                .all(|w| w[0].1 > w[1].1 || (w[0].1 == w[1].1 && w[0].0 < w[1].0))
        );
    }

    #[test]
    fn test_markdown_docs_list_every_routed_tool() {
        let doc = generate_markdown_docs();
        assert!(doc.contains(&format!("Total tools: {}\n", routed().len())));
        for r in routed() {
            assert!(doc.contains(&format!("| `{r}` |")), "{r}");
        }
        assert!(!doc.contains("survival_cox"));
        for line in doc.lines().filter(|l| l.starts_with("| `")) {
            let b = line.as_bytes(); // '|' and '\\' are ASCII: compare bytes, never slice the str
            let pipes = (0..b.len())
                .filter(|&i| b[i] == b'|' && (i == 0 || b[i - 1] != b'\\'))
                .count();
            assert_eq!(pipes, 4, "{line}");
        }
    }

    #[test]
    fn test_every_category_has_a_row() {
        for c in ToolCategory::all() {
            assert!(CATEGORY_TABLE.iter().any(|r| r.1 == *c), "{c:?}");
        }
    }

    #[test]
    fn test_category_name_round_trips_through_parse_category() {
        for c in ToolCategory::all() {
            assert_eq!(parse_category(category_name(*c)), Some(*c), "{c:?}");
        }
    }
}
