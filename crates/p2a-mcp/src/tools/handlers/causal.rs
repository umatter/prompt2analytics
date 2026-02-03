//! Causal inference tools handlers.
//!
//! This module provides MCP tool handlers for causal inference:
//! - IV methods (GMM-IV, 2SLS, first-stage, Sargan, MTE, Balke-Pearl bounds)
//! - Difference-in-Differences (DiD, staggered DiD, Bacon decomposition, ETWFE)
//! - Treatment effects (IPW, AIPW, DoubleML, CBPS, WeightIt, entropy balance, SBW, TWANG)
//! - Propensity score matching (MatchIt)
//! - TMLE family (TMLE, C-TMLE, LTMLE)
//! - Standardization and G-formula
//! - Mediation analysis
//! - Synthetic control (classic, gsynth, SCPI)
//! - Regression discontinuity (sharp, fuzzy, multi-cutoff)

use rmcp::{
    ErrorData as McpError, handler::server::wrapper::Parameters, model::*, tool, tool_router,
};

use crate::server::AnalyticsServer;
use crate::tools::requests::causal::{
    BPBoundsRequest, BaconDecompRequest, CbpsRequest, CTmleRequest, DiDRequest, DoubleMLRequest,
    DoublyRobustRequest, EntropyBalanceRequest, EtwfeRequest, EValueRequest, FirstStageRequest,
    FuzzyRdRequest, GeneralGmmIvRequest, GFormulaRequest, GsynthRequest, IV2SLSRequest, IpwRequest,
    IVMTERequest, LtmleRequest, MarginalEffectsRequest, MatchItRequest, MediationRequest,
    NaturalEffectsRequest, RdBandwidthRequest, RdEstimateRequest, RdMultiRequest, SarganTestRequest,
    SBWRequest, ScpiRequest, SensemakrRequest, StaggeredDiDRequest, StdRegRequest,
    SyntheticControlRequest, TmleRequest, TwangRequest, WeightItRequest,
};

use p2a_core::{
    bacon_decomp, ctmle, entropy_balance, match_it, run_bp_bounds, run_cbps, run_did,
    run_doubly_robust, run_etwfe, run_first_stage_diagnostics, run_gmm_iv, run_gsynth,
    run_ipw_treatment, run_iv2sls, run_ivmte, run_mediation_analysis, run_medflex_dataset,
    run_rd, run_rd_multi_dataset, run_scpi, run_staggered_did, run_stdreg,
    run_synthetic_control, run_tmle, run_twang, run_gformula, rd_bandwidth, run_fuzzy_rd, sbw,
    sargan_test, tmle, weightit,
    AttEstimationMethod, BPBoundsConfig, CbpsConfig, CbpsMethod, ComparisonGroup,
    ControlGroup as EtwfeControlGroup, CTmleConfig, CTmleQModel, DistanceMethod, DoublyRobustConfig,
    DRMethod, EffectEstimationMethod, EffectScale, Estimand, EtwfeConfig, GFormulaConfig,
    GFormulaData, GFormulaIntervention, GFormulaOutcomeType, GeneralGmmConfig, GeneralGmmResult,
    GmmMethod, GmmVcov, GModel, GsynthConfig, GsynthEstimator, GsynthForce, IpwConfig, IVMTEConfig, KernelType,
    MatchMethod, MatchResult, MediationConfig, MedflexConfig, MedflexResult, PropensityModel,
    QModel, RdConfig, RdMultiBandwidth, RdMultiConfig, RdMultiResult, SBWConfig, SBWEstimand,
    SCPIConfig, SCPIConstraint, SEMethod, SelectionOrder, StaggeredDidConfig, StdRegConfig,
    StdRegEstimand, StdRegModel, StoppingRule, StopMethod, SynthConfig, TmleConfig, TwangConfig,
    TimeAggregation, TwangEstimand, VarianceMethod, VceType, VOptimization, WeightEstimand,
    WeightItConfig, WeightMethod, BandwidthMethod, PoolingWeights, PredictorSpec,
    diagnostics::{
        iv_diagnostics, did_diagnostics, ipw_diagnostics, matching_diagnostics, rd_diagnostics,
        staggered_did_diagnostics,
    },
};
use p2a_core::linalg::design::DesignMatrix;
use p2a_core::regression::HacKernel;

// Helper function for formatting diagnostic warnings
fn format_diagnostic_warnings(report: &p2a_core::diagnostics::IdentificationReport) -> String {
    let mut output = String::new();
    if !report.warnings.is_empty() {
        output.push_str("\n\n=== IDENTIFICATION DIAGNOSTICS ===\n");
        for warning in &report.warnings {
            let severity_str = match warning.severity {
                p2a_core::diagnostics::WarningSeverity::Critical => "[CRITICAL]",
                p2a_core::diagnostics::WarningSeverity::Warning => "[WARNING]",
                p2a_core::diagnostics::WarningSeverity::Caution => "[CAUTION]",
                p2a_core::diagnostics::WarningSeverity::Info => "[INFO]",
            };
            output.push_str(&format!("{} {}\n", severity_str, warning.message));
            if !warning.remediation.is_empty() {
                output.push_str(&format!("  Remediation: {}\n", warning.remediation.join("; ")));
            }
        }
    }
    output
}

#[tool_router(router = causal_router, vis = "pub")]
impl AnalyticsServer {

/// Run IV/2SLS regression.
    #[tool(
        description = "Run Instrumental Variables (2SLS) regression. Use when an explanatory variable is endogenous (correlated with the error term). Requires valid instruments."
    )]
    async fn iv_2sls(
        &self,
        Parameters(request): Parameters<IV2SLSRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let robust = request.robust.unwrap_or(true);
        let x_exog_refs: Vec<&str> = request.x_exog.iter().map(|s| s.as_str()).collect();
        let x_endog_refs: Vec<&str> = request.x_endog.iter().map(|s| s.as_str()).collect();
        let instruments_refs: Vec<&str> = request.instruments.iter().map(|s| s.as_str()).collect();

        let result = match run_iv2sls(
            dataset,
            &request.y,
            &x_exog_refs,
            &x_endog_refs,
            &instruments_refs,
            robust,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "IV/2SLS estimation failed: {}",
                    e
                ))]));
            }
        };

        // Run identification diagnostics
        let mut output = result.to_string();
        if let Ok(diag_report) = iv_diagnostics(dataset, &result) {
            output.push_str(&format_diagnostic_warnings(&diag_report));
        }

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

/// Run first-stage diagnostics for IV/2SLS.
    #[tool(
        description = "Run first-stage diagnostics to test instrument strength. Reports F-statistic (F > 10 suggests strong instruments), R-squared, and coefficient estimates. Essential before running 2SLS."
    )]
    async fn iv_first_stage(
        &self,
        Parameters(request): Parameters<FirstStageRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let instruments: Vec<&str> = request.instruments.iter().map(|s| s.as_str()).collect();
        let controls: Option<Vec<&str>> = request
            .controls
            .as_ref()
            .map(|c| c.iter().map(|s| s.as_str()).collect());

        let result = match run_first_stage_diagnostics(
            dataset,
            &request.endogenous_var,
            &instruments,
            controls.as_deref(),
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "First-stage diagnostics failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

/// Run Sargan test of overidentifying restrictions for IV/2SLS.
    #[tool(
        description = "Run Sargan test of overidentifying restrictions for IV/2SLS. Tests whether instruments are valid (uncorrelated with error term). H0: instruments are valid. Rejection suggests at least one invalid instrument. Requires more instruments than endogenous variables."
    )]
    async fn iv_sargan_test(
        &self,
        Parameters(request): Parameters<SarganTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let x_exog: Vec<&str> = request.x_exog.iter().map(|s| s.as_str()).collect();
        let x_endog: Vec<&str> = request.x_endog.iter().map(|s| s.as_str()).collect();
        let instruments: Vec<&str> = request.instruments.iter().map(|s| s.as_str()).collect();

        let result = match sargan_test(dataset, &request.y, &x_exog, &x_endog, &instruments) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Sargan test failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

/// Compute Balke-Pearl bounds on the Average Causal Effect (ACE).
    #[tool(
        description = "Compute Balke-Pearl bounds for nonparametric IV analysis. Provides sharp bounds on the Average Causal Effect (ACE) without parametric assumptions. All three variables (instrument Z, treatment D, outcome Y) must be binary (0/1). Returns bounds with optional bootstrap confidence intervals. Also reports the Wald (standard IV) estimate for comparison. Use monotonicity=true if you can assume no defiers (instrument only affects treatment in one direction)."
    )]
    async fn bp_bounds(
        &self,
        Parameters(request): Parameters<BPBoundsRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::linalg::design::DesignMatrix;

        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        // Extract instrument variable (Z)
        let z = match DesignMatrix::extract_column(dataset.df(), &request.instrument) {
            Ok(col) => col,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to extract instrument variable '{}': {:?}",
                    request.instrument, e
                ))]));
            }
        };

        // Extract treatment variable (D)
        let d = match DesignMatrix::extract_column(dataset.df(), &request.treatment) {
            Ok(col) => col,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to extract treatment variable '{}': {:?}",
                    request.treatment, e
                ))]));
            }
        };

        // Extract outcome variable (Y)
        let y = match DesignMatrix::extract_column(dataset.df(), &request.outcome) {
            Ok(col) => col,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to extract outcome variable '{}': {:?}",
                    request.outcome, e
                ))]));
            }
        };

        // Convert confidence level to alpha
        let alpha = 1.0 - request.confidence_level.unwrap_or(0.95);

        // Build config
        let config = BPBoundsConfig {
            monotonicity: request.monotonicity.unwrap_or(false),
            bootstrap_ci: request.bootstrap_ci.unwrap_or(true),
            n_bootstrap: request.n_bootstrap.unwrap_or(1000),
            alpha,
            seed: request.seed,
        };

        // Run Balke-Pearl bounds
        let result = match run_bp_bounds(&z.view(), &d.view(), &y.view(), config) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Balke-Pearl bounds failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

/// Estimate Marginal Treatment Effects (MTE) using instrumental variables.
    #[tool(
        description = "Estimate Marginal Treatment Effects (MTE) using the Heckman-Vytlacil framework. MTE reveals heterogeneity in treatment effects across the distribution of unobserved resistance to treatment. Returns MTE curve, ATE, ATT, ATU, and LATE estimates. The MTE framework shows how different IV estimands are weighted averages of the MTE curve, providing deeper insight into treatment effect heterogeneity than standard IV/2SLS."
    )]
    async fn iv_mte(
        &self,
        Parameters(request): Parameters<IVMTERequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::linalg::design::DesignMatrix;

        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        // Extract outcome variable
        let y = match DesignMatrix::extract_column(dataset.df(), &request.y) {
            Ok(col) => col,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to extract outcome variable '{}': {:?}",
                    request.y, e
                ))]));
            }
        };

        // Extract treatment variable
        let d = match DesignMatrix::extract_column(dataset.df(), &request.d) {
            Ok(col) => col,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to extract treatment variable '{}': {:?}",
                    request.d, e
                ))]));
            }
        };

        // Extract instrument variable
        let z = match DesignMatrix::extract_column(dataset.df(), &request.z) {
            Ok(col) => col,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to extract instrument variable '{}': {:?}",
                    request.z, e
                ))]));
            }
        };

        // Parse propensity model
        let propensity_model = match request.propensity_model.as_deref() {
            Some("logit") => PropensityModel::Logit,
            Some("linear") => PropensityModel::Linear,
            _ => PropensityModel::Probit, // default
        };

        // Build config
        let config = IVMTEConfig {
            mte_degree: request.mte_degree.unwrap_or(2),
            n_grid: request.n_grid.unwrap_or(100),
            propensity_model,
            ..Default::default()
        };

        // Note: Covariates are not currently supported in the array-based API
        // TODO: Add covariate support when needed

        let result = match run_ivmte(
            &y.view(),
            &d.view(),
            &z.view(),
            None, // covariates
            config,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "MTE estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

/// Run Callaway-Sant'Anna staggered difference-in-differences.
    #[tool(
        description = "Estimate causal effects with staggered treatment adoption using Callaway-Sant'Anna (2021) method. Handles multiple time periods, heterogeneous treatment timing, and dynamic treatment effects. Returns group-time ATTs, event study plots, and overall ATT with pre-trend tests."
    )]
    async fn staggered_did(
        &self,
        Parameters(request): Parameters<StaggeredDiDRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        // Parse comparison group
        let comparison_group = match request.comparison_group.as_deref() {
            Some("not_yet_treated") | Some("notyet") => ComparisonGroup::NotYetTreated,
            _ => ComparisonGroup::NeverTreated,
        };

        // Parse estimation method
        let estimation_method = match request.estimation_method.as_deref() {
            Some("ipw") => AttEstimationMethod::IPW,
            Some("doubly_robust") | Some("dr") | Some("aipw") => AttEstimationMethod::DoublyRobust,
            _ => AttEstimationMethod::OutcomeRegression,
        };

        // Build config
        let config = StaggeredDidConfig {
            comparison_group,
            estimation_method,
            base_period: request.base_period.unwrap_or(-1),
            bootstrap: request.bootstrap.unwrap_or(999),
            ..Default::default()
        };

        // Parse covariates
        let cov_refs: Option<Vec<&str>> = request
            .covariates
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());

        let result = match run_staggered_did(
            dataset,
            &request.outcome,
            &request.treatment_time,
            &request.time_col,
            &request.unit_col,
            cov_refs.as_deref(),
            config,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Staggered DiD estimation failed: {}",
                    e
                ))]));
            }
        };

        // Run identification diagnostics
        let mut output = result.to_string();
        let diag_report = staggered_did_diagnostics(&result);
        output.push_str(&format_diagnostic_warnings(&diag_report));

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

/// Perform Goodman-Bacon decomposition for staggered DiD.
    #[tool(
        description = "Decompose a two-way fixed effects (TWFE) DiD estimate into weighted 2x2 comparisons using Goodman-Bacon (2021) decomposition. Reveals which comparisons (treated vs. never-treated, treated vs. not-yet-treated, later vs. earlier treated) contribute to the overall estimate and with what weights. Essential for understanding potential biases from 'forbidden' comparisons when treatment effects are heterogeneous over time."
    )]
    async fn bacon_decomp(
        &self,
        Parameters(request): Parameters<BaconDecompRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let result = match bacon_decomp(
            dataset,
            &request.outcome,
            &request.unit_col,
            &request.time_col,
            &request.treatment_col,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Goodman-Bacon decomposition failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

/// Run Extended Two-Way Fixed Effects (ETWFE) estimation.
    #[tool(
        description = "Estimate treatment effects using Extended TWFE (Wooldridge 2021, 2023). Addresses heterogeneous treatment effects in staggered DiD by estimating saturated cohort-by-time interactions. Returns cohort-time ATT estimates, event study by relative time, cohort averages, and overall ATT. Robust to treatment effect heterogeneity across cohorts and over time."
    )]
    async fn etwfe(
        &self,
        Parameters(request): Parameters<EtwfeRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        // Parse control group
        let cgroup = match request.cgroup.as_deref() {
            Some("never") => EtwfeControlGroup::Never,
            _ => EtwfeControlGroup::NotYet,
        };

        // Build config
        let config = EtwfeConfig {
            tref: None,
            gref: None,
            cgroup,
            anticipation: 0,
        };

        // Parse controls
        let control_refs: Option<Vec<&str>> = request
            .controls
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());

        let result = match run_etwfe(
            dataset,
            &request.outcome,
            &request.unit_col,
            &request.time_col,
            &request.treatment,
            &request.first_treat,
            control_refs.as_deref(),
            Some(config),
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "ETWFE estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

/// Run IPW treatment effect estimation.
    #[tool(
        description = "Estimate Average Treatment Effect (ATE) or Average Treatment Effect on Treated (ATT) using Inverse Probability Weighting. Uses propensity scores to create pseudo-populations that balance covariates between treatment groups. Returns effect estimate with bootstrap standard errors and confidence intervals."
    )]
    async fn treatment_ipw(
        &self,
        Parameters(request): Parameters<IpwRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let cov_refs: Vec<&str> = request.covariates.iter().map(|s| s.as_str()).collect();

        // Parse estimand
        let estimand = match request.estimand.as_deref() {
            Some("att") | Some("ATT") => Estimand::ATT,
            _ => Estimand::ATE,
        };

        let config = IpwConfig {
            trim: request.trim.unwrap_or(0.05),
            estimand,
            bootstrap: request.bootstrap.unwrap_or(999),
            normalized: true,
            seed: None,
        };

        let result = match run_ipw_treatment(
            dataset,
            &request.outcome,
            &request.treatment,
            &cov_refs,
            config,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "IPW estimation failed: {}",
                    e
                ))]));
            }
        };

        // Run identification diagnostics
        let mut output = result.to_string();
        let diag_report = ipw_diagnostics(&result);
        output.push_str(&format_diagnostic_warnings(&diag_report));

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

/// Run Doubly Robust (AIPW) treatment effect estimation.
    #[tool(
        description = "Estimate treatment effects using Augmented IPW (doubly robust). Combines propensity score weighting with outcome regression. Consistent if either the propensity model OR the outcome model is correctly specified. Returns effect estimate with bootstrap standard errors and model fit diagnostics."
    )]
    async fn treatment_doubly_robust(
        &self,
        Parameters(request): Parameters<DoublyRobustRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let cov_refs: Vec<&str> = request.covariates.iter().map(|s| s.as_str()).collect();

        // Parse method
        let method = match request.method.as_deref() {
            Some("ipw") | Some("IPW") => DRMethod::IPW,
            Some("regression") | Some("reg") => DRMethod::Regression,
            _ => DRMethod::AIPW,
        };

        // Parse estimand
        let estimand = match request.estimand.as_deref() {
            Some("att") | Some("ATT") => Estimand::ATT,
            _ => Estimand::ATE,
        };

        let config = DoublyRobustConfig {
            method,
            trim: request.trim.unwrap_or(0.05),
            estimand,
            bootstrap: request.bootstrap.unwrap_or(999),
            seed: None,
        };

        let result = match run_doubly_robust(
            dataset,
            &request.outcome,
            &request.treatment,
            &cov_refs,
            config,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Doubly robust estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

/// Run Double/Debiased Machine Learning (DoubleML) estimation.
    #[tool(
        description = "Estimate causal treatment effects using Double/Debiased Machine Learning. Uses Neyman-orthogonal score functions and K-fold cross-fitting to achieve root-n consistent and asymptotically normal estimates. Supports Partially Linear Regression (PLR: Y = theta*D + g(X) + eps) and Interactive Regression Model (IRM: binary treatment with heterogeneous effects). Returns treatment effect estimate with influence function-based standard errors and diagnostic information."
    )]
    async fn treatment_double_ml(
        &self,
        Parameters(request): Parameters<DoubleMLRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::econometrics::{DMLModelType, DoubleMLConfig, TreatmentType, run_double_ml};

        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        // Extract data columns
        let df = dataset.df();

        // Extract outcome (Y)
        let y_col = df.column(&request.outcome).map_err(|_| {
            McpError::invalid_request(
                format!("Outcome column '{}' not found", request.outcome),
                None,
            )
        })?;
        let y_vec: Vec<f64> = y_col
            .f64()
            .map_err(|_| {
                McpError::invalid_request(
                    format!("Outcome column '{}' must be numeric", request.outcome),
                    None,
                )
            })?
            .iter()
            .flatten()
            .collect();

        // Extract treatment (D)
        let d_col = df.column(&request.treatment).map_err(|_| {
            McpError::invalid_request(
                format!("Treatment column '{}' not found", request.treatment),
                None,
            )
        })?;
        let d_vec: Vec<f64> = d_col
            .f64()
            .map_err(|_| {
                McpError::invalid_request(
                    format!("Treatment column '{}' must be numeric", request.treatment),
                    None,
                )
            })?
            .iter()
            .flatten()
            .collect();

        // Extract covariates (X)
        let n = y_vec.len();
        let p = request.covariates.len();
        let mut x_data = Vec::with_capacity(n * p);

        for col_name in &request.covariates {
            let col = df.column(col_name).map_err(|_| {
                McpError::invalid_request(
                    format!("Covariate column '{}' not found", col_name),
                    None,
                )
            })?;
            let col_f64 = col.f64().map_err(|_| {
                McpError::invalid_request(
                    format!("Covariate column '{}' must be numeric", col_name),
                    None,
                )
            })?;
            for v in col_f64.iter() {
                x_data.push(v.unwrap_or(0.0));
            }
        }

        // Convert to ndarray
        use ndarray::{Array1, Array2};
        let y = Array1::from(y_vec);
        let d = Array1::from(d_vec);

        // Reshape X: we collected column-wise, need to transpose
        let x_col_major = Array2::from_shape_vec((n, p), {
            // Reorganize from column-major to row-major
            let mut row_major = vec![0.0; n * p];
            for j in 0..p {
                for i in 0..n {
                    row_major[i * p + j] = x_data[j * n + i];
                }
            }
            row_major
        })
        .map_err(|e| {
            McpError::internal_error(format!("Failed to create covariate matrix: {}", e), None)
        })?;

        // Parse model type
        let model_type = match request.model_type.as_deref() {
            Some("irm") | Some("IRM") | Some("interactive") => DMLModelType::IRM,
            _ => DMLModelType::PLR,
        };

        // Determine treatment type for IRM
        let is_binary = d
            .iter()
            .all(|&di| (di - 0.0).abs() < 1e-10 || (di - 1.0).abs() < 1e-10);
        let treatment_type = if is_binary {
            TreatmentType::Binary
        } else {
            TreatmentType::Continuous
        };

        let config = DoubleMLConfig {
            n_folds: request.n_folds.unwrap_or(5),
            model_type,
            treatment_type,
            seed: request.seed,
            trim: request.trim.unwrap_or(0.01),
            ..Default::default()
        };

        let result = match run_double_ml(&y.view(), &d.view(), &x_col_major.view(), config) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Double ML estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

/// Run Covariate Balancing Propensity Score (CBPS) estimation.
    #[tool(
        description = "Estimate propensity scores using Covariate Balancing Propensity Score (CBPS). Unlike standard logistic regression, CBPS uses GMM to simultaneously estimate propensity scores AND achieve covariate balance. Returns propensity scores, IPW weights, balance diagnostics before and after weighting, and J-test for overidentification."
    )]
    async fn treatment_cbps(
        &self,
        Parameters(request): Parameters<CbpsRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let cov_refs: Vec<&str> = request.covariates.iter().map(|s| s.as_str()).collect();

        // Parse method
        let method = match request.method.as_deref() {
            Some("over") | Some("overbalance") => CbpsMethod::OverBalance,
            Some("just") | Some("justified") | Some("logit") => CbpsMethod::JustIdentified,
            _ => CbpsMethod::ExactBalance,
        };

        let config = CbpsConfig {
            method,
            balance_threshold: request.balance_threshold.unwrap_or(0.1),
            tolerance: 1e-8,
            max_iter: 100,
        };

        let result = match run_cbps(dataset, &request.treatment, &cov_refs, Some(config)) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "CBPS estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

/// Compute flexible inverse probability weights using multiple methods.
    #[tool(
        description = "Compute balancing weights for causal inference using WeightIt. Supports multiple methods: 'logistic' (standard propensity score), 'entropy' (entropy balancing for exact mean balance), 'energy' (energy distance minimization), 'stable' (stable balancing weights). Returns weights, balance diagnostics before/after, and effective sample size (ESS). Low ESS indicates high weight variability."
    )]
    async fn treatment_weightit(
        &self,
        Parameters(request): Parameters<WeightItRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let cov_refs: Vec<&str> = request.covariates.iter().map(|s| s.as_str()).collect();

        // Parse weighting method
        let method = match request.method.as_deref() {
            Some("entropy") | Some("ebal") => WeightMethod::Entropy,
            Some("energy") => WeightMethod::Energy,
            Some("stable") | Some("sbw") => WeightMethod::Stable,
            _ => WeightMethod::Logistic,
        };

        // Parse estimand
        let estimand = match request.estimand.as_deref() {
            Some("att") | Some("ATT") => WeightEstimand::ATT,
            Some("atc") | Some("ATC") => WeightEstimand::ATC,
            _ => WeightEstimand::ATE,
        };

        let config = WeightItConfig {
            method,
            estimand,
            intercept: true,
            stabilize: request.stabilize.unwrap_or(false),
            trim_quantile: request.trim_quantile.unwrap_or(1.0),
            max_iter: 200,
            tolerance: 1e-8,
        };

        let result = match weightit(dataset, &request.treatment, &cov_refs, config) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "WeightIt estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

/// Run entropy balancing for exact covariate mean balance.
    #[tool(
        description = "Compute entropy balancing weights (Hainmueller 2012). Reweights the control group to achieve exact mean balance on specified covariates with the treated group. Minimizes entropy (KL divergence) from uniform weights subject to balance constraints. Useful when exact balance is needed for bias reduction. Returns weights, balance table, and effective sample size."
    )]
    async fn treatment_entropy_balance(
        &self,
        Parameters(request): Parameters<EntropyBalanceRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let cov_refs: Vec<&str> = request.covariates.iter().map(|s| s.as_str()).collect();

        let result = match entropy_balance(
            dataset,
            &request.treatment,
            &cov_refs,
            request.target_means.as_deref(),
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Entropy balancing failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

/// Compute stable balancing weights for causal inference.
    #[tool(
        description = "Compute Stable Balancing Weights (SBW) using quadratic programming (Zubizarreta 2015). Directly optimizes for covariate balance rather than modeling the propensity score. Finds weights that minimize variance while achieving exact or approximate balance on covariate means. Advantages: directly targets balance (not propensity score fit), provides stable weights with lower variance than IPW, handles approximate balance when exact is infeasible. Returns weights, balance diagnostics before/after, effective sample size, and optimization details."
    )]
    async fn treatment_sbw(
        &self,
        Parameters(request): Parameters<SBWRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let cov_refs: Vec<&str> = request.covariates.iter().map(|s| s.as_str()).collect();

        // Parse estimand
        let estimand = match request.estimand.as_deref() {
            Some("ate") | Some("ATE") => SBWEstimand::ATE,
            Some("atc") | Some("ATC") => SBWEstimand::ATC,
            _ => SBWEstimand::ATT,
        };

        let config = SBWConfig {
            estimand,
            balance_tol: request.balance_tol.unwrap_or(0.0),
            min_weight: request.min_weight.unwrap_or(0.0),
            normalize_to_n: true,
            max_iter: 1000,
            tolerance: 1e-8,
            balance_penalty: request.balance_penalty.unwrap_or(1000.0),
        };

        let result = match sbw(dataset, &request.treatment, &cov_refs, config) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Stable Balancing Weights estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

/// Run twang GBM propensity score estimation.
    #[tool(
        description = "Estimate propensity scores using Gradient Boosted Machine (GBM) with automatic tuning for covariate balance (twang). Unlike logistic regression, twang uses machine learning and automatically selects the optimal number of iterations based on balance metrics (standardized effect sizes or KS statistics). Particularly useful when you need good balance across many covariates. Returns propensity scores, IPW weights, optimal iteration number, and balance diagnostics before/after weighting."
    )]
    async fn treatment_twang(
        &self,
        Parameters(request): Parameters<TwangRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let cov_refs: Vec<&str> = request.covariates.iter().map(|s| s.as_str()).collect();

        // Parse stopping rule
        let stop_method = match request.stop_method.as_deref() {
            Some("es.max") | Some("esmax") | Some("ESMax") => StopMethod::ESMax,
            Some("ks.mean") | Some("ksmean") | Some("KSMean") => StopMethod::KSMean,
            Some("ks.max") | Some("ksmax") | Some("KSMax") => StopMethod::KSMax,
            _ => StopMethod::ESMean,
        };

        // Parse estimand
        let estimand = match request.estimand.as_deref() {
            Some("ate") | Some("ATE") => TwangEstimand::ATE,
            Some("atc") | Some("ATC") => TwangEstimand::ATC,
            _ => TwangEstimand::ATT,
        };

        let config = TwangConfig {
            n_trees: request.n_trees.unwrap_or(3000),
            shrinkage: request.shrinkage.unwrap_or(0.01),
            stop_method,
            estimand,
            balance_threshold: request.balance_threshold.unwrap_or(0.1),
            min_iterations: 100,
            interaction_depth: 1,
            min_node_size: 10,
        };

        let result = match run_twang(dataset, &request.treatment, &cov_refs, Some(config)) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "twang estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

/// Run propensity score matching (MatchIt) for causal inference.
    #[tool(
        description = "Perform propensity score matching to create balanced comparison groups for causal inference. Methods: 'nearest' (nearest neighbor matching on propensity score), 'cem' (coarsened exact matching on covariate bins), 'full' (optimal full matching), 'subclass' (propensity score subclassification). Returns matched sample with weights, balance diagnostics (standardized mean differences, variance ratios, KS statistics) before and after matching. Low SMD (<0.1) indicates good balance."
    )]
    async fn propensity_matching(
        &self,
        Parameters(request): Parameters<MatchItRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let cov_refs: Vec<&str> = request.covariates.iter().map(|s| s.as_str()).collect();

        // Parse matching method
        let method = match request.method.as_deref() {
            Some("cem") | Some("CEM") | Some("coarsened") => MatchMethod::CoarsenedExact {
                cutpoints: None,
                n_bins: Some(4),
            },
            Some("full") | Some("optimal") => MatchMethod::Full {
                min_ratio: 0.5,
                max_ratio: 2.0,
            },
            Some("subclass") | Some("stratify") => MatchMethod::Subclass {
                n_subclasses: request.n_subclasses.unwrap_or(5),
            },
            _ => MatchMethod::NearestNeighbor {
                ratio: request.ratio.unwrap_or(1),
                caliper: request.caliper,
                replace: request.replace.unwrap_or(false),
            },
        };

        // Parse distance metric
        let distance = match request.distance.as_deref() {
            Some("probit") | Some("Probit") => Some(DistanceMethod::Probit),
            Some("mahalanobis") | Some("Mahalanobis") => Some(DistanceMethod::Mahalanobis),
            Some("euclidean") | Some("Euclidean") => Some(DistanceMethod::Euclidean),
            _ => Some(DistanceMethod::Logit),
        };

        let result: MatchResult =
            match match_it(dataset, &request.treatment, &cov_refs, method, distance) {
                Ok(r) => r,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Propensity score matching failed: {}",
                        e
                    ))]));
                }
            };

        // Run identification diagnostics
        let mut output = result.to_string();
        let diag_report = matching_diagnostics(&result);
        output.push_str(&format_diagnostic_warnings(&diag_report));

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

/// Run Targeted Maximum Likelihood Estimation (TMLE) for causal inference.
    #[tool(
        description = "Estimate Average Treatment Effect (ATE) using TMLE - a doubly robust, semiparametric efficient estimator. TMLE uses a targeting step to optimize the bias-variance tradeoff for the ATE. More efficient than standard AIPW due to the fluctuation model. Returns ATE estimate with influence curve-based standard errors and confidence intervals."
    )]
    async fn treatment_tmle(
        &self,
        Parameters(request): Parameters<TmleRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let cov_refs: Vec<&str> = request.covariates.iter().map(|s| s.as_str()).collect();

        // Parse outcome model type
        let q_model = match request.q_model.as_deref() {
            Some("linear") | Some("Linear") | Some("continuous") => QModel::Linear,
            _ => QModel::Logistic,
        };

        // Propensity score truncation bounds
        let ps_lower = request.ps_lower.unwrap_or(0.01);
        let ps_upper = request.ps_upper.unwrap_or(0.99);

        let config = TmleConfig {
            q_model,
            g_model: GModel::Logistic,
            truncate_ps: (ps_lower, ps_upper),
            max_iter: 100,
            tolerance: 1e-8,
        };

        let result = match tmle(
            dataset,
            &request.outcome,
            &request.treatment,
            &cov_refs,
            config,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "TMLE estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

/// Run Collaborative Targeted Maximum Likelihood Estimation (C-TMLE) for causal inference.
    #[tool(
        description = "Estimate Average Treatment Effect (ATE) using C-TMLE - extends TMLE with data-adaptive covariate selection for the propensity score model. Uses cross-validation to select which covariates to include, reducing finite-sample bias from including too many covariates while maintaining double robustness. Returns ATE estimate, selected covariates, selection path with CV criterion at each step, and influence curve-based standard errors."
    )]
    async fn collaborative_tmle(
        &self,
        Parameters(request): Parameters<CTmleRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let cov_refs: Vec<&str> = request.covariates.iter().map(|s| s.as_str()).collect();

        // Parse outcome model type
        let q_model = match request.q_model.as_deref() {
            Some("linear") | Some("Linear") | Some("continuous") => CTmleQModel::Linear,
            _ => CTmleQModel::Logistic,
        };

        // Parse stopping rule
        let stopping_rule = match request.stopping_rule.as_deref() {
            Some("one_se") | Some("OneSE") | Some("1se") => StoppingRule::OneSE,
            Some("max_covariates") | Some("MaxCovariates") => {
                StoppingRule::MaxCovariates(request.max_covariates.unwrap_or(10))
            }
            _ => StoppingRule::CVMinimum, // default
        };

        // Propensity score truncation bounds
        let ps_lower = request.ps_lower.unwrap_or(0.025);
        let ps_upper = request.ps_upper.unwrap_or(0.975);

        let config = CTmleConfig {
            n_folds: request.n_folds.unwrap_or(5),
            max_covariates: request.max_covariates,
            stopping_rule,
            order: SelectionOrder::Forward,
            q_model,
            gbound: (ps_lower, ps_upper),
            ..Default::default()
        };

        let result = match ctmle(
            dataset,
            &request.outcome,
            &request.treatment,
            &cov_refs,
            config,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "C-TMLE estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

/// Run Parametric G-Formula for time-varying treatments.
    #[tool(
        description = "Estimate causal effects of time-varying treatments using the parametric g-formula (generalized formula). Uses Monte Carlo simulation to estimate counterfactual outcomes under different treatment regimes. Suitable for longitudinal/panel data with time-varying confounders. Returns risk difference, risk ratio, potential outcomes, and bootstrap confidence intervals. Based on Robins (1986) and Hernan & Robins (2020). R equivalent: gfoRmula package."
    )]
    async fn gformula(
        &self,
        Parameters(request): Parameters<GFormulaRequest>,
    ) -> Result<CallToolResult, McpError> {
        use ndarray::{Array1, Array2};
        use p2a_core::linalg::design::DesignMatrix;

        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let n = dataset.df().height();
        let t_max = request.time_points;

        // Validate treatment columns match time points
        if request.treatment_cols.len() != t_max {
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "Number of treatment columns ({}) must match time_points ({})",
                request.treatment_cols.len(),
                t_max
            ))]));
        }

        // Extract baseline covariates
        let n_baseline = request.baseline_covariates.len();
        let mut baseline = Array2::zeros((n, n_baseline));
        for (j, col_name) in request.baseline_covariates.iter().enumerate() {
            let col = match DesignMatrix::extract_column(dataset.df(), col_name) {
                Ok(c) => c,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Failed to extract baseline covariate '{}': {:?}",
                        col_name, e
                    ))]));
                }
            };
            for i in 0..n {
                baseline[[i, j]] = col[i];
            }
        }

        // Extract time-varying covariates for each time point
        let n_tv = request.time_varying_covariates.len();
        let mut time_varying: Vec<Array2<f64>> = Vec::with_capacity(t_max);
        for t in 0..t_max {
            let mut tv_matrix = Array2::zeros((n, n_tv));
            for (j, base_name) in request.time_varying_covariates.iter().enumerate() {
                let col_name = format!("{}_t{}", base_name, t);
                let col = match DesignMatrix::extract_column(dataset.df(), &col_name) {
                    Ok(c) => c,
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Failed to extract time-varying covariate '{}': {:?}. \
                             Expected columns named '{}' for each time point t=0..{}.",
                            col_name,
                            e,
                            format!("{}_t0, {}_t1, ...", base_name, base_name),
                            t_max - 1
                        ))]));
                    }
                };
                for i in 0..n {
                    tv_matrix[[i, j]] = col[i];
                }
            }
            time_varying.push(tv_matrix);
        }

        // Extract treatments for each time point
        let mut treatments: Vec<Array1<f64>> = Vec::with_capacity(t_max);
        for col_name in &request.treatment_cols {
            let col = match DesignMatrix::extract_column(dataset.df(), col_name) {
                Ok(c) => c,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Failed to extract treatment column '{}': {:?}",
                        col_name, e
                    ))]));
                }
            };
            treatments.push(col);
        }

        // Extract outcome
        let outcome = match DesignMatrix::extract_column(dataset.df(), &request.outcome) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to extract outcome '{}': {:?}",
                    request.outcome, e
                ))]));
            }
        };

        // Create GFormulaData
        let gformula_data = match GFormulaData::new(baseline, time_varying, treatments, outcome) {
            Ok(d) => d,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to create g-formula data: {}",
                    e
                ))]));
            }
        };

        // Parse intervention type
        let intervention = match request.intervention.as_deref() {
            Some("never_treat") | Some("never") | Some("control") => {
                GFormulaIntervention::Static { treat_all: false }
            }
            Some("natural") | Some("natural_course") | Some("observed") => {
                GFormulaIntervention::NaturalCourse
            }
            Some("threshold") => {
                let variable_idx = request.threshold_variable.unwrap_or(0);
                let cutoff = request.threshold_cutoff.unwrap_or(0.5);
                let above = request.threshold_above.unwrap_or(true);
                GFormulaIntervention::Threshold {
                    variable_idx,
                    cutoff,
                    above,
                }
            }
            _ => GFormulaIntervention::Static { treat_all: true }, // default: always treat
        };

        // Parse outcome type
        let outcome_type = match request.outcome_type.as_deref() {
            Some("binary") | Some("Binary") | Some("logistic") => GFormulaOutcomeType::Binary,
            Some("survival") | Some("Survival") | Some("hazard") => GFormulaOutcomeType::Survival,
            _ => GFormulaOutcomeType::Continuous,
        };

        // Use per-tool seed if provided, otherwise fall back to global seed
        let global_seed = self.global_seed.read().await;
        let seed = request.seed.or(*global_seed);

        // Build config
        let config = GFormulaConfig {
            n_simulations: request.n_simulations.unwrap_or(1000),
            time_points: t_max,
            intervention,
            outcome_type,
            n_bootstrap: request.n_bootstrap.unwrap_or(200),
            seed,
            confidence_level: request.confidence_level.unwrap_or(0.95),
            max_iter: 100,
            tolerance: 1e-8,
        };

        // Run g-formula
        let result = match run_gformula(&gformula_data, config) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "G-formula estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

/// Run causal mediation analysis.
    #[tool(
        description = "Perform causal mediation analysis to decompose treatment effects into direct and indirect (mediated) effects. Uses IPW-based identification following Huber (2014). Returns Natural Direct Effect (NDE), Natural Indirect Effect (NIE), proportion mediated, and bootstrap inference."
    )]
    async fn mediation_analysis(
        &self,
        Parameters(request): Parameters<MediationRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let cov_refs: Vec<&str> = request.covariates.iter().map(|s| s.as_str()).collect();

        let config = MediationConfig {
            bootstrap: request.bootstrap.unwrap_or(999),
            trim: request.trim.unwrap_or(0.05),
            seed: None,
        };

        let result = match run_mediation_analysis(
            dataset,
            &request.outcome,
            &request.treatment,
            &request.mediator,
            &cov_refs,
            config,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Mediation analysis failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

/// Run Natural Effect Models for mediation analysis with treatment-mediator interactions.
    #[tool(
        description = "Perform Natural Effect Models (medflex) mediation analysis that allows for \
        treatment-mediator interactions. Decomposes total effect into Natural Direct Effect (NDE) and \
        Natural Indirect Effect (NIE) using the regression-based approach of Lange, Vansteelandt, & Bekaert (2012). \
        Unlike IPW-based mediation, this method uses regression models for both mediator and outcome, \
        with optional A*M interaction terms. Returns effect estimates, bootstrap CIs, and model diagnostics."
    )]
    async fn natural_effects_mediation(
        &self,
        Parameters(request): Parameters<NaturalEffectsRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let conf_refs: Vec<&str> = request
            .confounders
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default();

        // Parse effect scale
        let scale = match request.scale.as_deref() {
            Some("ratio") => EffectScale::Ratio,
            Some("odds_ratio") | Some("odds") => EffectScale::OddsRatio,
            _ => EffectScale::Difference,
        };

        let n_bootstrap = request.n_bootstrap.unwrap_or(1000);
        let config = MedflexConfig {
            allow_interaction: request.allow_interaction.unwrap_or(true),
            bootstrap_ci: n_bootstrap > 0,
            n_bootstrap,
            confidence_level: request.confidence_level.unwrap_or(0.95),
            scale,
            seed: None,
        };

        let result: MedflexResult = match run_medflex_dataset(
            dataset,
            &request.outcome,
            &request.treatment,
            &request.mediator,
            &conf_refs,
            config,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Natural effects mediation analysis failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

/// Run Synthetic Control Method for comparative case studies.
    #[tool(
        description = "Run Synthetic Control Method for comparative case studies with a single treated unit. \
        Creates a weighted combination of control (donor) units to construct a synthetic counterfactual. \
        Developed by Abadie, Diamond, and Hainmueller (2010). \
        Returns unit weights, predictor balance, treatment effects at each post-treatment period, \
        and optional placebo-based inference (p-values)."
    )]
    async fn synthetic_control(
        &self,
        Parameters(request): Parameters<SyntheticControlRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        // Convert predictor specs
        let predictors: Vec<PredictorSpec> = request
            .predictors
            .iter()
            .map(|spec| {
                let aggregation = match spec.aggregation.as_deref() {
                    Some("first") => TimeAggregation::First,
                    Some("last") => TimeAggregation::Last,
                    Some("sum") => TimeAggregation::Sum,
                    _ => TimeAggregation::Mean, // default
                };
                PredictorSpec {
                    column: spec.column.clone(),
                    aggregation,
                    time_window: spec.time_window,
                }
            })
            .collect();

        // Convert V optimization method
        let v_method = match request.v_method.as_deref() {
            Some("equal") => VOptimization::Equal,
            Some("custom") => {
                if let Some(weights) = &request.custom_v_weights {
                    VOptimization::Custom(weights.clone())
                } else {
                    return Ok(CallToolResult::error(vec![Content::text(
                        "custom_v_weights must be provided when v_method is 'custom'".to_string(),
                    )]));
                }
            }
            _ => VOptimization::DataDriven, // default
        };

        let config = SynthConfig {
            treatment_time: request.treatment_time,
            treated_unit: request.treated_unit.clone(),
            optimization_window: request.optimization_window,
            v_method,
            tolerance: request.tolerance.unwrap_or(1e-6),
            max_iter: request.max_iter.unwrap_or(1000),
            run_placebos: request.run_placebos.unwrap_or(false),
            weight_threshold: request.weight_threshold.unwrap_or(0.001),
        };

        let result = match run_synthetic_control(
            dataset,
            &request.outcome,
            &request.unit_col,
            &request.time_col,
            &predictors,
            config,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Synthetic control failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

/// Run Generalized Synthetic Control (gsynth) for panel data with multiple treated units.
    #[tool(
        description = "Run Generalized Synthetic Control Method for causal inference with interactive fixed effects. \
        Unlike traditional synthetic control (single treated unit), gsynth handles multiple treated units with staggered adoption. \
        Uses Interactive Fixed Effects (IFE) to model: Y_it = α_i + λ_i'f_t + X_it'β + τ_it·D_it + ε_it. \
        Factors f_t and loadings λ_i capture unobserved confounders. Developed by Xu (2017). \
        Returns ATT (average treatment effect on treated), unit-level effects, factor structure, and optional bootstrap inference."
    )]
    async fn gsynth(
        &self,
        Parameters(request): Parameters<GsynthRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        // Parse estimator
        let estimator = match request.estimator.as_deref() {
            Some("mc") => GsynthEstimator::MatrixCompletion,
            _ => GsynthEstimator::Ife, // default
        };

        // Parse force (fixed effects)
        let force = match request.force.as_deref() {
            Some("none") => GsynthForce::None,
            Some("time") => GsynthForce::Time,
            Some("twoWay") | Some("twoway") | Some("two_way") => GsynthForce::TwoWay,
            _ => GsynthForce::Unit, // default
        };

        let config = GsynthConfig {
            n_factors: request.n_factors.unwrap_or(2),
            cross_validate: request.cross_validate.unwrap_or(false),
            max_factors: request.max_factors.unwrap_or(5),
            estimator,
            force,
            bootstrap_se: request.bootstrap_se.unwrap_or(false),
            n_bootstrap: request.n_bootstrap.unwrap_or(500),
            ..Default::default()
        };

        let cov_refs: Vec<&str> = request
            .covariates
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default();

        let result = match run_gsynth(
            dataset,
            &request.outcome,
            &request.treatment,
            &request.unit_col,
            &request.time_col,
            &cov_refs,
            config,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Generalized synthetic control (gsynth) failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

/// Run Synthetic Control with Prediction Intervals (SCPI).
    #[tool(
        description = "Run Synthetic Control with Prediction Intervals (SCPI) for causal inference. \
        Extends the classic synthetic control method (Abadie et al. 2010) with proper uncertainty quantification \
        through prediction intervals that account for both in-sample and out-of-sample variance. \
        Supports multiple constraint types: simplex (classic SC), lasso (sparse), ridge (shrinkage), or lasso_simplex (sparse with sum=1). \
        Developed by Cattaneo, Feng & Titiunik (2021) in JASA. \
        Returns donor weights, treatment effects with prediction intervals, variance decomposition, and pre-treatment fit statistics."
    )]
    async fn scpi(
        &self,
        Parameters(request): Parameters<ScpiRequest>,
    ) -> Result<CallToolResult, McpError> {
        use ndarray::{Array1, Array2};
        use p2a_core::polars::prelude::{ChunkCompareEq, PolarsError};

        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let df = dataset.df();

        // Extract unique units and times
        let units: Vec<String> = match df.column(&request.unit_col) {
            Ok(col) => match col.str() {
                Ok(str_col) => str_col
                    .into_iter()
                    .filter_map(|s| s.map(|s| s.to_string()))
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect(),
                Err(_) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Unit column '{}' must be string type",
                        request.unit_col
                    ))]));
                }
            },
            Err(_) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unit column '{}' not found",
                    request.unit_col
                ))]));
            }
        };

        let mut times: Vec<i64> = match df.column(&request.time_col) {
            Ok(col) => match col.i64() {
                Ok(int_col) => int_col
                    .into_iter()
                    .flatten()
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect(),
                Err(_) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Time column '{}' must be integer type",
                        request.time_col
                    ))]));
                }
            },
            Err(_) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Time column '{}' not found",
                    request.time_col
                ))]));
            }
        };
        times.sort();

        // Validate treated unit exists
        if !units.contains(&request.treated_unit) {
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "Treated unit '{}' not found in data. Available units: {:?}",
                request.treated_unit,
                units.iter().take(10).collect::<Vec<_>>()
            ))]));
        }

        // Find treatment period index
        let treatment_idx = match times.iter().position(|&t| t == request.treatment_time) {
            Some(idx) => idx,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Treatment time {} not found in data. Available times: {:?}",
                    request.treatment_time,
                    times.iter().take(20).collect::<Vec<_>>()
                ))]));
            }
        };

        // Build treated series and donor matrix
        let n_times = times.len();
        let donor_units: Vec<String> = units
            .iter()
            .filter(|u| *u != &request.treated_unit)
            .cloned()
            .collect();
        let n_donors = donor_units.len();

        // Helper function to get value for unit at time
        let get_value = |df: &p2a_core::polars::frame::DataFrame,
                         outcome: &str,
                         unit_col: &str,
                         time_col: &str,
                         unit: &str,
                         time: i64|
         -> Result<f64, String> {
            let unit_col_data = df
                .column(unit_col)
                .map_err(|e: PolarsError| e.to_string())?;
            let unit_str = unit_col_data
                .str()
                .map_err(|_| "Unit column not string".to_string())?;
            let unit_mask = unit_str.equal(unit);

            let time_col_data = df
                .column(time_col)
                .map_err(|e: PolarsError| e.to_string())?;
            let time_i64 = time_col_data
                .i64()
                .map_err(|_| "Time column not int".to_string())?;
            let time_mask = time_i64.equal(time);

            let combined = &unit_mask & &time_mask;
            let filtered = df
                .filter(&combined)
                .map_err(|e: PolarsError| e.to_string())?;
            if filtered.height() == 0 {
                return Err(format!("No data for unit '{}' at time {}", unit, time));
            }
            let outcome_col = filtered
                .column(outcome)
                .map_err(|e: PolarsError| e.to_string())?;
            let val = outcome_col
                .f64()
                .map_err(|_| "Outcome not numeric".to_string())?
                .get(0)
                .ok_or_else(|| "Missing value".to_string())?;
            Ok(val)
        };

        // Extract outcome values for treated unit
        let mut treated_values = Array1::zeros(n_times);
        for (t_idx, &time) in times.iter().enumerate() {
            match get_value(
                df,
                &request.outcome,
                &request.unit_col,
                &request.time_col,
                &request.treated_unit,
                time,
            ) {
                Ok(v) => treated_values[t_idx] = v,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Error extracting treated unit data: {}",
                        e
                    ))]));
                }
            }
        }

        // Extract donor matrix
        let mut donor_matrix = Array2::zeros((n_times, n_donors));
        for (d_idx, donor) in donor_units.iter().enumerate() {
            for (t_idx, &time) in times.iter().enumerate() {
                match get_value(
                    df,
                    &request.outcome,
                    &request.unit_col,
                    &request.time_col,
                    donor,
                    time,
                ) {
                    Ok(v) => donor_matrix[[t_idx, d_idx]] = v,
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Error extracting donor unit '{}' data: {}",
                            donor, e
                        ))]));
                    }
                }
            }
        }

        // Parse constraint type
        let lambda = request.lambda.unwrap_or(0.1);
        let constraint = match request.constraint.as_deref() {
            Some("lasso") => SCPIConstraint::Lasso { lambda },
            Some("ridge") => SCPIConstraint::Ridge { lambda },
            Some("lasso_simplex") => SCPIConstraint::LassoSimplex { lambda },
            _ => SCPIConstraint::Simplex, // default
        };

        // Parse variance method
        let variance_method = match request.variance_method.as_deref() {
            Some("gaussian") => VarianceMethod::Gaussian,
            Some("loo_cv") => VarianceMethod::LooCv,
            Some("kfold_cv") => VarianceMethod::KFoldCv,
            _ => VarianceMethod::Subgaussian, // default
        };

        let config = SCPIConfig {
            constraint,
            alpha: request.alpha.unwrap_or(0.05),
            variance_method,
            cv_folds: request.cv_folds.unwrap_or(5),
            weight_threshold: request.weight_threshold.unwrap_or(0.001),
            ..Default::default()
        };

        let result = match run_scpi(
            &treated_values.view(),
            &donor_matrix.view(),
            treatment_idx,
            config,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "SCPI estimation failed: {}",
                    e
                ))]));
            }
        };

        // Format output with donor unit names
        let mut output = format!("{}\n", result);
        output.push_str("\nDonor Unit Names (for non-zero weights):\n");
        for (idx, weight) in &result.nonzero_weights {
            if let Some(name) = donor_units.get(*idx) {
                output.push_str(&format!("  {}: {:.4}\n", name, weight));
            }
        }

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

/// Run Sharp Regression Discontinuity estimation.
    #[tool(
        description = "Run Sharp Regression Discontinuity (RD) estimation. Implements local polynomial regression with robust bias-corrected inference following Calonico, Cattaneo & Titiunik (2014). Returns conventional, bias-corrected, and robust treatment effect estimates with confidence intervals."
    )]
    async fn rd_estimate(
        &self,
        Parameters(request): Parameters<RdEstimateRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let cutoff = request.cutoff.unwrap_or(0.0);

        // Parse kernel type
        let kernel = match request.kernel.as_deref() {
            Some("epanechnikov") => KernelType::Epanechnikov,
            Some("uniform") => KernelType::Uniform,
            _ => KernelType::Triangular, // default
        };

        // Parse bandwidth selection method
        let bwselect = match request.bwselect.as_deref() {
            Some("msetwo") => BandwidthMethod::MseTwo,
            Some("cerrd") => BandwidthMethod::CerRd,
            Some("certwo") => BandwidthMethod::CerTwo,
            _ => BandwidthMethod::MseRd, // default
        };

        let config = RdConfig {
            p: request.p.unwrap_or(1),
            q: None, // auto = p + 1
            h: request.h,
            b: request.b,
            kernel,
            bwselect,
            level: request.level.unwrap_or(0.95),
            ..Default::default()
        };

        let result = match run_rd(
            dataset,
            &request.outcome,
            &request.running_var,
            cutoff,
            config,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "RD estimation failed: {}",
                    e
                ))]));
            }
        };

        // Run identification diagnostics
        let mut output = result.to_string();
        let diag_report = rd_diagnostics(&result);
        output.push_str(&format_diagnostic_warnings(&diag_report));

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

/// Compute RD bandwidth only.
    #[tool(
        description = "Compute MSE-optimal bandwidth for Regression Discontinuity estimation. Returns bandwidth values without running the full estimation. Useful for inspecting bandwidth selection before estimation."
    )]
    async fn rd_bw(
        &self,
        Parameters(request): Parameters<RdBandwidthRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let cutoff = request.cutoff.unwrap_or(0.0);
        let p = request.p.unwrap_or(1);

        // Parse kernel type
        let kernel = match request.kernel.as_deref() {
            Some("epanechnikov") => KernelType::Epanechnikov,
            Some("uniform") => KernelType::Uniform,
            _ => KernelType::Triangular,
        };

        // Parse bandwidth selection method
        let bwselect = match request.bwselect.as_deref() {
            Some("msetwo") => BandwidthMethod::MseTwo,
            Some("cerrd") => BandwidthMethod::CerRd,
            Some("certwo") => BandwidthMethod::CerTwo,
            _ => BandwidthMethod::MseRd,
        };

        let result = match rd_bandwidth(
            dataset,
            &request.outcome,
            &request.running_var,
            cutoff,
            p,
            kernel,
            bwselect,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "RD bandwidth selection failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

/// Run Fuzzy Regression Discontinuity estimation.
    #[tool(
        description = "Run Fuzzy Regression Discontinuity estimation. For cases where treatment probability (not assignment) jumps at the cutoff. Uses a Wald estimator (ratio of reduced-form to first-stage). Returns Local Average Treatment Effect (LATE)."
    )]
    async fn rd_fuzzy(
        &self,
        Parameters(request): Parameters<FuzzyRdRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let cutoff = request.cutoff.unwrap_or(0.0);

        // Parse kernel type
        let kernel = match request.kernel.as_deref() {
            Some("epanechnikov") => KernelType::Epanechnikov,
            Some("uniform") => KernelType::Uniform,
            _ => KernelType::Triangular,
        };

        // Parse bandwidth selection method
        let bwselect = match request.bwselect.as_deref() {
            Some("msetwo") => BandwidthMethod::MseTwo,
            Some("cerrd") => BandwidthMethod::CerRd,
            Some("certwo") => BandwidthMethod::CerTwo,
            _ => BandwidthMethod::MseRd,
        };

        let config = RdConfig {
            p: request.p.unwrap_or(1),
            q: None,
            h: request.h,
            b: None,
            kernel,
            bwselect,
            level: request.level.unwrap_or(0.95),
            ..Default::default()
        };

        let result = match run_fuzzy_rd(
            dataset,
            &request.outcome,
            &request.running_var,
            &request.treatment,
            cutoff,
            config,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Fuzzy RD estimation failed: {}",
                    e
                ))]));
            }
        };

        // Run identification diagnostics on the outcome RD
        let mut output = result.to_string();
        let diag_report = rd_diagnostics(&result.outcome_rd);
        output.push_str(&format_diagnostic_warnings(&diag_report));

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

/// Run Multi-Cutoff Regression Discontinuity estimation.
    #[tool(
        description = "Run Multi-Cutoff Regression Discontinuity (rdmulti) estimation. Handles RD designs with multiple cutoffs (different thresholds) sharing the same running variable. Estimates cutoff-specific effects and optionally pools them into a single weighted estimate. Includes a heterogeneity test for whether effects differ across cutoffs. Reference: Cattaneo, Titiunik & Vazquez-Bare (2020)."
    )]
    async fn rd_multi(
        &self,
        Parameters(request): Parameters<RdMultiRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        if request.cutoffs.is_empty() {
            return Ok(CallToolResult::error(vec![Content::text(
                "At least one cutoff value must be specified in 'cutoffs'.",
            )]));
        }

        // Parse kernel type
        let kernel = match request.kernel.as_deref() {
            Some("epanechnikov") => KernelType::Epanechnikov,
            Some("uniform") => KernelType::Uniform,
            _ => KernelType::Triangular,
        };

        // Parse pooling weights
        let pooling_weights = match request.pooling_weights.as_deref() {
            Some("inverse_variance") | Some("iv") => PoolingWeights::InverseVariance,
            Some("equal") => PoolingWeights::Equal,
            _ => PoolingWeights::SampleSize,
        };

        // Determine bandwidth specification
        let bandwidth = if let Some(bws) = &request.bandwidths {
            if bws.len() != request.cutoffs.len() {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Number of bandwidths ({}) must match number of cutoffs ({})",
                    bws.len(),
                    request.cutoffs.len()
                ))]));
            }
            RdMultiBandwidth::PerCutoff(bws.clone())
        } else if let Some(h) = request.bandwidth {
            RdMultiBandwidth::Global(h)
        } else {
            RdMultiBandwidth::PerCutoffOptimal
        };

        let config = RdMultiConfig {
            cutoffs: request.cutoffs.clone(),
            bandwidth,
            kernel,
            p: request.p.unwrap_or(1),
            q: None,
            pooled: request.pooled.unwrap_or(true),
            pooling_weights,
            bwselect: BandwidthMethod::MseRd,
            vce: VceType::default(),
            level: request.level.unwrap_or(0.95),
            test_heterogeneity: request.test_heterogeneity.unwrap_or(true),
        };

        let result: RdMultiResult = match run_rd_multi_dataset(
            dataset,
            &request.outcome,
            &request.running_var,
            request.cutoff_col.as_deref(),
            config,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Multi-cutoff RD estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run Difference-in-Differences estimation.
    #[tool(
        description = "Run Difference-in-Differences (DiD) estimation. Estimates causal treatment effects by comparing treated vs control groups before and after treatment."
    )]
    async fn diff_in_diff(
        &self,
        Parameters(request): Parameters<DiDRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let result = match run_did(
            dataset,
            &request.dep_var,
            &request.treatment_var,
            &request.post_var,
            None,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "DiD estimation failed: {}",
                    e
                ))]));
            }
        };

        // Run identification diagnostics
        let mut output = result.to_string();
        let diag_report = did_diagnostics(&result);
        output.push_str(&format_diagnostic_warnings(&diag_report));

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    /// E-value sensitivity analysis for unmeasured confounding.
    #[tool(
        description = "Compute E-values for sensitivity analysis to unmeasured confounding (VanderWeele & Ding 2017). The E-value is the minimum strength of association that an unmeasured confounder would need with both treatment and outcome to fully explain away an observed effect. Supports risk ratios (RR), odds ratios (OR), hazard ratios (HR), standardized mean differences (SMD), and risk differences (RD). A large E-value means considerable confounding would be needed to explain away the effect. Returns E-value for point estimate and confidence interval limit closest to null."
    )]
    async fn evalue(
        &self,
        Parameters(request): Parameters<EValueRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::regression::{evalue_hr, evalue_or, evalue_rd, evalue_rr_ci, evalue_smd};

        let effect_type = request.effect_type.to_lowercase();

        let result = match effect_type.as_str() {
            "rr" | "risk_ratio" | "riskratio" => {
                evalue_rr_ci(request.point, request.ci_lower, request.ci_upper)
            }
            "or" | "odds_ratio" | "oddsratio" => {
                let rare = request.rare.unwrap_or(true);
                evalue_or(request.point, request.ci_lower, request.ci_upper, rare)
            }
            "hr" | "hazard_ratio" | "hazardratio" => {
                let rare = request.rare.unwrap_or(true);
                evalue_hr(request.point, request.ci_lower, request.ci_upper, rare)
            }
            "smd" | "standardized_mean_difference" => evalue_smd(request.point, request.se),
            "rd" | "risk_difference" | "riskdifference" => {
                let baseline = request.baseline_risk.ok_or_else(|| {
                    McpError::invalid_params("baseline_risk is required for risk difference", None)
                })?;
                evalue_rd(request.point, baseline, request.se)
            }
            _ => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown effect type '{}'. Valid types: rr, or, hr, smd, rd",
                    request.effect_type
                ))]));
            }
        };

        match result {
            Ok(r) => Ok(CallToolResult::success(vec![Content::text(r.to_string())])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "E-value calculation failed: {}",
                e
            ))])),
        }
    }

    /// Run general GMM IV estimation (Hansen 1982).
    #[tool(
        description = "Run general GMM (Generalized Method of Moments) IV estimation following Hansen (1982). Estimates parameters using moment conditions E[z(y - xβ)] = 0. Supports two-step, iterative, and CUE estimation with HAC weighting. Reports J-test for overidentifying restrictions. Use for IV estimation when you have more instruments than endogenous variables."
    )]
    async fn gmm_iv(
        &self,
        Parameters(request): Parameters<GeneralGmmIvRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();
        let z_refs: Vec<&str> = request.z.iter().map(|s| s.as_str()).collect();

        // Parse method
        let method = match request.method.as_deref() {
            Some("iterative") | Some("Iterative") => GmmMethod::Iterative,
            Some("cue") | Some("CUE") => GmmMethod::CUE,
            _ => GmmMethod::TwoStep, // Default
        };

        // Parse vcov type
        let vcov = match request.vcov.as_deref() {
            Some("iid") | Some("IID") => GmmVcov::IID,
            Some("fixed") | Some("Fixed") => GmmVcov::Fixed,
            _ => GmmVcov::HAC, // Default
        };

        // Parse kernel
        let kernel = match request.kernel.as_deref() {
            Some("parzen") | Some("Parzen") => HacKernel::Parzen,
            Some("qs") | Some("quadratic_spectral") => HacKernel::QuadraticSpectral,
            Some("truncated") | Some("Truncated") => HacKernel::Truncated,
            Some("tukey") | Some("tukey_hanning") => HacKernel::TukeyHanning,
            _ => HacKernel::Bartlett, // Default
        };

        let config = GeneralGmmConfig {
            method,
            vcov,
            kernel,
            bandwidth: request.bandwidth,
            ..Default::default()
        };

        let result: GeneralGmmResult =
            match run_gmm_iv(dataset, &request.y, &x_refs, &z_refs, Some(config)) {
                Ok(r) => r,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "GMM IV estimation failed: {}",
                        e
                    ))]));
                }
            };
        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run Longitudinal Targeted Maximum Likelihood Estimation (LTMLE) for time-varying treatments.
    #[tool(
        description = "Estimate causal effects of time-varying treatments using Longitudinal TMLE. LTMLE extends standard TMLE to longitudinal settings with multiple time points where treatments and confounders vary over time. Uses sequential regression (g-computation) combined with a targeting step at each time point to achieve double robustness. Estimates E[Y^{always treat}] - E[Y^{never treat}] under static intervention regimes. Returns ATE estimate, counterfactual means, influence curve-based standard errors, and diagnostics."
    )]
    async fn ltmle(
        &self,
        Parameters(request): Parameters<LtmleRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::econometrics::{LtmleConfig, LtmleData, LtmleQModel, run_ltmle};

        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        // Validate inputs
        let t_max = request.outcomes.len();
        if t_max < 2 {
            return Ok(CallToolResult::error(vec![Content::text(
                "LTMLE requires at least 2 time points. For single time point, use standard TMLE.",
            )]));
        }
        if request.treatments.len() != t_max || request.covariates.len() != t_max {
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "Number of time points must be consistent: outcomes={}, treatments={}, covariates={}",
                request.outcomes.len(),
                request.treatments.len(),
                request.covariates.len()
            ))]));
        }

        // Extract data from dataset
        let df = dataset.df();
        let n = df.height();

        // Extract outcomes at each time point
        let mut outcomes: Vec<ndarray::Array1<f64>> = Vec::with_capacity(t_max);
        for col_name in &request.outcomes {
            let col = match df.column(col_name) {
                Ok(c) => c,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Column '{}' not found: {}",
                        col_name, e
                    ))]));
                }
            };
            let values: Vec<f64> = col
                .f64()
                .map_err(|e| {
                    McpError::invalid_request(
                        format!("Column '{}' must be numeric: {}", col_name, e),
                        None,
                    )
                })?
                .into_no_null_iter()
                .collect();
            outcomes.push(ndarray::Array1::from_vec(values));
        }

        // Extract treatments at each time point
        let mut treatments: Vec<ndarray::Array1<f64>> = Vec::with_capacity(t_max);
        for col_name in &request.treatments {
            let col = match df.column(col_name) {
                Ok(c) => c,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Column '{}' not found: {}",
                        col_name, e
                    ))]));
                }
            };
            let values: Vec<f64> = col
                .f64()
                .map_err(|e| {
                    McpError::invalid_request(
                        format!("Column '{}' must be numeric: {}", col_name, e),
                        None,
                    )
                })?
                .into_no_null_iter()
                .collect();
            treatments.push(ndarray::Array1::from_vec(values));
        }

        // Extract covariates at each time point
        let mut covariates: Vec<ndarray::Array2<f64>> = Vec::with_capacity(t_max);
        for cov_list in &request.covariates {
            // Parse comma-separated covariate names
            let cov_names: Vec<&str> = cov_list.split(',').map(|s| s.trim()).collect();
            let k = cov_names.len();

            let mut cov_data = Vec::with_capacity(n * k);
            for row_idx in 0..n {
                for col_name in &cov_names {
                    let col = match df.column(col_name) {
                        Ok(c) => c,
                        Err(e) => {
                            return Ok(CallToolResult::error(vec![Content::text(format!(
                                "Column '{}' not found: {}",
                                col_name, e
                            ))]));
                        }
                    };
                    let value = col
                        .f64()
                        .map_err(|e| {
                            McpError::invalid_request(
                                format!("Column '{}' must be numeric: {}", col_name, e),
                                None,
                            )
                        })?
                        .get(row_idx)
                        .unwrap_or(f64::NAN);
                    cov_data.push(value);
                }
            }

            let cov_array = ndarray::Array2::from_shape_vec((n, k), cov_data).map_err(|e| {
                McpError::invalid_request(format!("Failed to create covariate matrix: {}", e), None)
            })?;
            covariates.push(cov_array);
        }

        // Create LTMLE data structure
        let data = match LtmleData::new(outcomes, treatments, covariates) {
            Ok(d) => d,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Invalid LTMLE data: {}",
                    e
                ))]));
            }
        };

        // Parse configuration
        let q_model = match request.q_model.as_deref() {
            Some("logistic") | Some("Logistic") | Some("binary") => LtmleQModel::Logistic,
            _ => LtmleQModel::Linear,
        };

        let ps_lower = request.ps_lower.unwrap_or(0.01);
        let ps_upper = request.ps_upper.unwrap_or(0.99);

        let config = LtmleConfig {
            q_model,
            gbounds: (ps_lower, ps_upper),
            ..Default::default()
        };

        let result = match run_ltmle(&data, config) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "LTMLE estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Compute average marginal effects from regression models.
    #[tool(
        description = "Compute average marginal effects (AME) from regression models. For OLS, marginal effects equal coefficients. For Logit/Probit, effects are averaged across observations accounting for nonlinearity. Returns effects, standard errors, z-values, p-values, and confidence intervals."
    )]
    async fn marginal_effects(
        &self,
        Parameters(request): Parameters<MarginalEffectsRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::regression::{marginal_effects, ModelType};

        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let model_type = match request.model.as_deref().unwrap_or("ols") {
            "ols" | "linear" => ModelType::Ols,
            "logit" | "logistic" => ModelType::Logit,
            "probit" => ModelType::Probit,
            other => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown model type '{}'. Supported: 'ols', 'logit', 'probit'.",
                    other
                ))]));
            }
        };

        let result = match marginal_effects(dataset, &request.y, &x_refs, model_type) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Marginal effects computation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run Regression Standardization (G-computation) for causal effect estimation.
    #[tool(
        description = "Estimate causal effects using regression standardization (G-computation/parametric g-formula). Fits an outcome model and averages predictions under different treatment values over the covariate distribution. Supports ATE (Average Treatment Effect), ATT (on Treated), ATC (on Controls). Returns effect estimate, potential outcomes E[Y(1)] and E[Y(0)], confidence intervals, and for binary outcomes: risk ratio, odds ratio, and NNT."
    )]
    async fn regression_standardization(
        &self,
        Parameters(request): Parameters<StdRegRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let cov_refs: Vec<&str> = request.covariates.iter().map(|s| s.as_str()).collect();

        // Parse model type
        let model_type = match request.model_type.as_deref() {
            Some("logistic") | Some("Logistic") | Some("binary") => StdRegModel::Logistic,
            Some("poisson") | Some("Poisson") | Some("count") => StdRegModel::Poisson,
            _ => StdRegModel::Linear,
        };

        // Parse estimand
        let estimand = match request.estimand.as_deref() {
            Some("att") | Some("ATT") => StdRegEstimand::ATT,
            Some("atc") | Some("ATC") => StdRegEstimand::ATC,
            Some("levels") | Some("Levels") => StdRegEstimand::Levels,
            _ => StdRegEstimand::ATE,
        };

        // Parse SE method
        let se_method = match request.se_method.as_deref() {
            Some("delta") | Some("Delta") => SEMethod::Delta,
            Some("sandwich") | Some("Sandwich") | Some("robust") => SEMethod::Sandwich,
            _ => SEMethod::Bootstrap,
        };

        let config = StdRegConfig {
            model_type,
            estimand,
            se_method,
            n_bootstrap: request.n_bootstrap.unwrap_or(999),
            confidence_level: request.confidence_level.unwrap_or(0.95),
            include_interactions: request.interactions.unwrap_or(false),
            seed: None,
            max_iter: 100,
            tolerance: 1e-8,
        };

        let result = match run_stdreg(
            dataset,
            &request.outcome,
            &request.treatment,
            &cov_refs,
            config,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Regression standardization failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Sensitivity analysis for unmeasured confounding (sensemakr).
    #[tool(
        description = "Run sensitivity analysis for unmeasured confounding (Cinelli & Hazlett 2020). Computes robustness value (RV) - the minimum confounding strength needed to nullify the treatment effect. Key outputs: (1) Partial R²: how much variance treatment explains in outcome, (2) RV_q: confounding needed to reduce effect by q%, (3) RV_alpha: confounding needed to make effect insignificant, (4) Benchmark bounds: adjusted estimates under various confounding scenarios. Essential for causal inference to assess how robust findings are to unmeasured confounding."
    )]
    async fn sensemakr(
        &self,
        Parameters(request): Parameters<SensemakrRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::regression::{generate_contour_data, run_ols, run_sensemakr, CovarianceType};

        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let covariate_refs: Vec<&str> = request.covariates.iter().map(|s| s.as_str()).collect();
        let benchmark_refs: Option<Vec<&str>> = request
            .benchmark_covariates
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());

        let q = request.q.unwrap_or(1.0);
        let alpha = request.alpha.unwrap_or(0.05);

        let mut result = match run_sensemakr(
            dataset,
            &request.y,
            &request.treatment,
            &covariate_refs,
            benchmark_refs.as_deref(),
            request.kd,
            request.ky,
            q,
            alpha,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Sensitivity analysis failed: {}",
                    e
                ))]));
            }
        };

        // Generate contour data if requested
        if request.contour_data.unwrap_or(false) {
            // Re-run OLS to get the result for contour generation
            let mut x_cols: Vec<&str> = vec![request.treatment.as_str()];
            x_cols.extend(covariate_refs.iter());

            if let Ok(ols_result) = run_ols(dataset, &request.y, &x_cols, true, CovarianceType::HC1)
            {
                if let Ok(contour) =
                    generate_contour_data(&ols_result, &request.treatment, Some(20), Some(0.5))
                {
                    result.contour_data = Some(contour);
                }
            }
        }

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }
}
