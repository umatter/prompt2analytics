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

/// Maximum number of elements allowed in a single matrix allocation.
/// Prevents denial-of-service via extremely large dimension inputs.
const MAX_MATRIX_ELEMENTS: usize = 10_000_000;

/// Maximum number of time points allowed for longitudinal methods.
const MAX_TIME_POINTS: usize = 1_000;

use crate::server::AnalyticsServer;
use crate::tools::requests::causal::{
    BPBoundsRequest, BaconDecompRequest, CTmleRequest, CbpsRequest, DiDRequest, DoubleMLRequest,
    DoublyRobustRequest, EValueRequest, EntropyBalanceRequest, EtwfeRequest, FirstStageRequest,
    FuzzyRdRequest, GFormulaRequest, GeneralGmmIvRequest, GsynthRequest, IV2SLSRequest,
    IVMTERequest, IpwRequest, LtmleRequest, MarginalEffectsRequest, MatchItRequest,
    MediationRequest, NaturalEffectsRequest, RdBandwidthRequest, RdEstimateRequest, RdMultiRequest,
    SBWRequest, SarganTestRequest, ScpiRequest, SensemakrRequest, StaggeredDiDRequest,
    StdRegRequest, SyntheticControlRequest, TmleRequest, TwangRequest, WeightItRequest,
};

use crate::tools::common::parse_option;
use p2a_core::econometrics::rd::rd_bandwidth_with;
use p2a_core::linalg::design::DesignMatrix;
use p2a_core::options::Choices;
use p2a_core::regression::HacKernel;
use p2a_core::regression::ModelType as MarginalModel;
use p2a_core::{
    AttEstimationMethod, BPBoundsConfig, BandwidthMethod, CTmleConfig, CTmleQModel, CbpsConfig,
    CbpsMethod, ComparisonGroup, ControlGroup as EtwfeControlGroup, DRMethod, DistanceMethod,
    DoublyRobustConfig, EffectEstimationMethod, EffectScale, Estimand, EtwfeConfig, GFormulaConfig,
    GFormulaData, GFormulaIntervention, GFormulaOutcomeType, GModel, GeneralGmmConfig,
    GeneralGmmResult, GmmMethod, GmmVcov, GsynthConfig, GsynthEstimator, GsynthForce, IVMTEConfig,
    IpwConfig, KernelType, MatchMethod, MatchResult, MedflexConfig, MedflexResult, MediationConfig,
    PoolingWeights, PredictorSpec, PropensityModel, QModel, RdConfig, RdMultiBandwidth,
    RdMultiConfig, RdMultiResult, SBWConfig, SBWEstimand, SCPIConfig, SCPIConstraint, SEMethod,
    SelectionOrder, StaggeredDidConfig, StdRegConfig, StdRegEstimand, StdRegModel, StopMethod,
    StoppingRule, SynthConfig, TimeAggregation, TmleConfig, TwangConfig, TwangEstimand,
    VOptimization, VarianceMethod, VceType, WeightEstimand, WeightItConfig, WeightMethod,
    bacon_decomp, ctmle,
    diagnostics::{
        did_diagnostics, ipw_diagnostics, iv_diagnostics, matching_diagnostics, rd_diagnostics,
        staggered_did_diagnostics,
    },
    entropy_balance, match_it, run_bp_bounds, run_cbps, run_did, run_doubly_robust, run_etwfe,
    run_first_stage_diagnostics, run_fuzzy_rd, run_gformula, run_gmm_iv, run_gsynth,
    run_ipw_treatment, run_iv2sls, run_ivmte, run_medflex_dataset, run_mediation_analysis, run_rd,
    run_rd_multi_dataset, run_scpi, run_staggered_did, run_stdreg, run_synthetic_control, run_tmle,
    run_twang, sargan_test, sbw, tmle, weightit,
};

/// An RD option (`vce`, `kernel`, `bwselect`) by name, `default` when it is absent, or the
/// message for a name the core refused: the core message alone, which names the parameter
/// and the valid values.
fn rd_option<T>(
    name: Option<&str>,
    default: T,
    parse: fn(&str) -> p2a_core::EconResult<T>,
) -> Result<T, String> {
    match name {
        None => Ok(default),
        Some(s) => parse(s).map_err(|e| match e {
            p2a_core::EconError::InvalidSpecification { message } => message,
            other => other.to_string(),
        }),
    }
}

/// The bare message of an option error (the option and the accepted values), else the error's
/// Display.
fn option_message(e: p2a_core::EconError) -> String {
    match e {
        p2a_core::EconError::InvalidSpecification { message } => message,
        other => other.to_string(),
    }
}

/// `None` gives `default`; `Some(s)` parses `s` with a local table, an unknown value giving the
/// bare message (the local twin of `common::parse_option`).
fn choice<T: Copy + PartialEq + 'static>(
    table: &'static Choices<T>,
    value: Option<&str>,
    default: T,
) -> Result<T, String> {
    match value {
        None => Ok(default),
        Some(s) => table.parse(s).map_err(option_message),
    }
}

/// Unwraps a `Result<T, String>`, or returns the message as a tool error from the handler.
macro_rules! tool_try {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
        }
    };
}

/// twang's stopping rules (`stop.method`).
static STOP_METHOD: Choices<StopMethod> = Choices {
    option: "stop_method",
    values: &[
        (&["es.mean", "esmean"], StopMethod::ESMean),
        (&["es.max", "esmax"], StopMethod::ESMax),
        (&["ks.mean", "ksmean"], StopMethod::KSMean),
        (&["ks.max", "ksmax"], StopMethod::KSMax),
    ],
};

/// twang's estimands.
static TWANG_ESTIMAND: Choices<TwangEstimand> = Choices {
    option: "estimand",
    values: &[
        (&["att"], TwangEstimand::ATT),
        (&["ate"], TwangEstimand::ATE),
        (&["atc"], TwangEstimand::ATC),
    ],
};

/// rd_multi's pooling weights across cutoffs.
static POOLING_WEIGHTS: Choices<PoolingWeights> = Choices {
    option: "pooling_weights",
    values: &[
        (
            &["sample_size", "samplesize", "n"],
            PoolingWeights::SampleSize,
        ),
        (
            &["inverse_variance", "iv", "precision"],
            PoolingWeights::InverseVariance,
        ),
        (&["equal", "uniform"], PoolingWeights::Equal),
    ],
};

/// The models marginal_effects fits.
static MARGINAL_MODEL: Choices<MarginalModel> = Choices {
    option: "model",
    values: &[
        (&["ols", "linear", "lm"], MarginalModel::Ols),
        (&["logit", "logistic"], MarginalModel::Logit),
        (&["probit"], MarginalModel::Probit),
    ],
};

/// The effect measures the E-value takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EffectMeasure {
    Rr,
    Or,
    Hr,
    Smd,
    Rd,
}

static EFFECT_MEASURE: Choices<EffectMeasure> = Choices {
    option: "effect_type",
    values: &[
        (
            &["rr", "risk_ratio", "riskratio", "relative_risk"],
            EffectMeasure::Rr,
        ),
        (&["or", "odds_ratio", "oddsratio"], EffectMeasure::Or),
        (&["hr", "hazard_ratio", "hazardratio"], EffectMeasure::Hr),
        (
            &[
                "smd",
                "standardized_mean_difference",
                "standardised_mean_difference",
            ],
            EffectMeasure::Smd,
        ),
        (
            &["rd", "risk_difference", "riskdifference"],
            EffectMeasure::Rd,
        ),
    ],
};

/// The regime an LTMLE estimate is contrasted with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContrastKind {
    Never,
    Always,
    Dynamic,
    None,
}

static LTMLE_CONTRAST: Choices<ContrastKind> = Choices {
    option: "contrast",
    values: &[
        (
            &["never", "never_treat", "control", "untreated"],
            ContrastKind::Never,
        ),
        (
            &["always", "always_treat", "treat", "treated"],
            ContrastKind::Always,
        ),
        (
            &["dynamic", "rule", "threshold_rule"],
            ContrastKind::Dynamic,
        ),
        (&["none", "no_contrast"], ContrastKind::None),
    ],
};

impl AnalyticsServer {
    /// The session's global seed, used when a request sets none.
    async fn session_seed(&self) -> Option<u64> {
        *self.global_seed.read().await
    }
}

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
                output.push_str(&format!(
                    "  Remediation: {}\n",
                    warning.remediation.join("; ")
                ));
            }
        }
    }
    output
}

#[tool_router(router = causal_router, vis = "pub")]
impl AnalyticsServer {
    /// Run IV/2SLS regression.
    #[tool(
        description = "Run Instrumental Variables (2SLS) regression. Use when an explanatory variable is endogenous (correlated with the error term). Requires valid instruments. Includes first-stage diagnostics: the partial F of the excluded instruments (the HC1 Wald F with robust = true, the default) and instrument strength. Prefer this over iv_first_stage for a complete IV analysis."
    )]
    pub async fn iv_2sls(
        &self,
        Parameters(request): Parameters<IV2SLSRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

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
        description = "Run ONLY first-stage diagnostics to test instrument strength. Reports the partial F of the excluded instruments (F > 10 suggests strong instruments), R-squared, and coefficient estimates. Note: iv_2sls already includes first-stage results — use this only when you need first-stage diagnostics without the full 2SLS estimation."
    )]
    pub async fn iv_first_stage(
        &self,
        Parameters(request): Parameters<FirstStageRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

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
    pub async fn iv_sargan_test(
        &self,
        Parameters(request): Parameters<SarganTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

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
    pub async fn bp_bounds(
        &self,
        Parameters(request): Parameters<BPBoundsRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::linalg::design::DesignMatrix;

        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

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
        let mut config = BPBoundsConfig::default();
        config.monotonicity = request.monotonicity.unwrap_or(false);
        config.bootstrap_ci = request.bootstrap_ci.unwrap_or(true);
        config.n_bootstrap = request.n_bootstrap.unwrap_or(1000);
        config.alpha = alpha;
        config.seed = request.seed;

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
    pub async fn iv_mte(
        &self,
        Parameters(request): Parameters<IVMTERequest>,
    ) -> Result<CallToolResult, McpError> {
        let propensity_model = tool_try!(parse_option(
            request.propensity_model.as_deref(),
            PropensityModel::Probit
        ));
        use p2a_core::linalg::design::DesignMatrix;

        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

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
        description = "Estimate group-time average treatment effects with staggered adoption as Callaway & Sant'Anna (2021) and R's did::att_gt with did::aggte aggregations (simple, dynamic/event, group). Without covariates every method is the difference in mean outcome changes; with covariates, 'outcome_regression' is DRDID reg_did_panel, 'ipw' std_ipw_did_panel and 'doubly_robust' drdid_panel, with covariates taken from the earlier of the two periods of each cell, as did. Comparison units: never treated (default) or not yet treated; anticipation shifts the base period. bootstrap = 0 gives analytic influence-function standard errors for every method. Returns ATT(g,t), the aggregations, a pre-trend test and identification diagnostics."
    )]
    pub async fn staggered_did(
        &self,
        Parameters(request): Parameters<StaggeredDiDRequest>,
    ) -> Result<CallToolResult, McpError> {
        let comparison_group = tool_try!(parse_option(
            request.comparison_group.as_deref(),
            ComparisonGroup::NeverTreated
        ));
        let estimation_method = tool_try!(parse_option(
            request.estimation_method.as_deref(),
            AttEstimationMethod::OutcomeRegression
        ));
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        // Build config
        let config = StaggeredDidConfig {
            comparison_group,
            estimation_method,
            base_period: request.base_period.unwrap_or(-1),
            bootstrap: request.bootstrap.unwrap_or(999),
            anticipation: request.anticipation.unwrap_or(0),
            seed: self.session_seed().await,
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
    pub async fn bacon_decomp(
        &self,
        Parameters(request): Parameters<BaconDecompRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

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
        description = "Estimate staggered-adoption treatment effects by extended two-way fixed effects as R's etwfe::etwfe(ivar = unit) and etwfe::emfx: the saturated cohort x period design with unit and period fixed effects (unbalanced panels exact), not-yet-treated (default) or never-treated controls, time-invariant controls demeaned within cohort and interacted with the cells and the periods, unit-clustered SEs (default 'cluster': CR1 with fixest's small-sample factor; etwfe and fixest default to iid SEs, which vcov = 'iid' gives), and the emfx aggregations simple, event, group and calendar with delta-method SEs. The treatment column must equal 1(period >= first_treat > 0)."
    )]
    pub async fn etwfe(
        &self,
        Parameters(request): Parameters<EtwfeRequest>,
    ) -> Result<CallToolResult, McpError> {
        let cgroup = tool_try!(parse_option(
            request.cgroup.as_deref(),
            EtwfeControlGroup::NotYet
        ));
        use p2a_core::econometrics::etwfe::{EtwfeAggregation, EtwfeVcov};
        let vcov = tool_try!(parse_option(request.vcov.as_deref(), EtwfeVcov::Cluster));
        let aggregation: Option<EtwfeAggregation> = match request.aggregation.as_deref() {
            None => None,
            Some(s) => Some(tool_try!(
                s.parse::<EtwfeAggregation>().map_err(option_message)
            )),
        };
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        // Build config
        let mut config = EtwfeConfig::default();
        config.tref = request.tref;
        config.gref = request.gref;
        config.cgroup = cgroup;
        config.vcov = vcov;
        config.anticipation = 0;

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

        let text = match aggregation {
            Some(which) => {
                let mut s = result.aggregation_text(which);
                crate::tools::common::append_notes(&mut s, None, &result.warnings);
                s
            }
            None => result.to_string(),
        };
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    /// Run IPW treatment effect estimation.
    #[tool(
        description = "Estimate Average Treatment Effect (ATE) or Average Treatment Effect on Treated (ATT) using Inverse Probability Weighting. Uses propensity scores to create pseudo-populations that balance covariates between treatment groups. Returns effect estimate with bootstrap standard errors and confidence intervals. Returns a point estimate (ATE/ATT). For matched samples with balance diagnostics, use propensity_matching instead."
    )]
    pub async fn treatment_ipw(
        &self,
        Parameters(request): Parameters<IpwRequest>,
    ) -> Result<CallToolResult, McpError> {
        let estimand = tool_try!(parse_option(request.estimand.as_deref(), Estimand::ATE));
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let cov_refs: Vec<&str> = request.covariates.iter().map(|s| s.as_str()).collect();

        let mut config = IpwConfig::default();
        config.trim = request.trim.unwrap_or(0.05);
        config.estimand = estimand;
        config.bootstrap = request.bootstrap.unwrap_or(999);
        config.normalized = true;
        config.seed = None;

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
    pub async fn treatment_doubly_robust(
        &self,
        Parameters(request): Parameters<DoublyRobustRequest>,
    ) -> Result<CallToolResult, McpError> {
        let method = tool_try!(parse_option(request.method.as_deref(), DRMethod::AIPW));
        let estimand = tool_try!(parse_option(request.estimand.as_deref(), Estimand::ATE));
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let cov_refs: Vec<&str> = request.covariates.iter().map(|s| s.as_str()).collect();

        let mut config = DoublyRobustConfig::default();
        config.method = method;
        config.trim = request.trim.unwrap_or(0.05);
        config.estimand = estimand;
        config.bootstrap = request.bootstrap.unwrap_or(999);
        config.seed = None;

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
        description = "Estimate causal treatment effects using Double/Debiased Machine Learning. Uses Neyman-orthogonal score functions and K-fold cross-fitting to achieve root-n consistent and asymptotically normal estimates. Supports Partially Linear Regression (PLR: Y = theta*D + g(X) + eps) and Interactive Regression Model (IRM: binary treatment with heterogeneous effects). Returns treatment effect estimate with influence function-based standard errors and diagnostic information. Columns must not contain missing values; integer and boolean columns are accepted."
    )]
    pub async fn treatment_double_ml(
        &self,
        Parameters(request): Parameters<DoubleMLRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::econometrics::{DMLModelType, DoubleMLConfig, TreatmentType, run_double_ml};

        let model_type = tool_try!(parse_option(
            request.model_type.as_deref(),
            DMLModelType::PLR
        ));

        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        // y, d and X are read jointly; DoubleML refuses missing data, so a missing value
        // in any of them is a tool error naming the column (NaPolicy::Fail).
        let n = dataset.df().height();
        let p = request.covariates.len();
        if n.checked_mul(p)
            .is_none_or(|total| total > MAX_MATRIX_ELEMENTS)
        {
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "Matrix dimensions too large: {} rows x {} covariates exceeds limit of {} elements.",
                n, p, MAX_MATRIX_ELEMENTS
            ))]));
        }
        let mut names: Vec<&str> = vec![request.outcome.as_str(), request.treatment.as_str()];
        names.extend(request.covariates.iter().map(String::as_str));
        let cols = match crate::tools::common::numeric_columns(
            dataset,
            &names,
            crate::tools::common::NaPolicy::Fail,
        ) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Double ML: {e}"
                ))]));
            }
        };
        let y = cols.numeric_array(&request.outcome);
        let d = cols.numeric_array(&request.treatment);
        let cov_names: Vec<&str> = request.covariates.iter().map(String::as_str).collect();
        let x = if cov_names.is_empty() {
            ndarray::Array2::zeros((cols.n(), 0))
        } else {
            match cols.matrix(&cov_names, false) {
                Ok(m) => m.data,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Double ML: {e}"
                    ))]));
                }
            }
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

        let result = match run_double_ml(&y.view(), &d.view(), &x.view(), config) {
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
    pub async fn treatment_cbps(
        &self,
        Parameters(request): Parameters<CbpsRequest>,
    ) -> Result<CallToolResult, McpError> {
        let method = tool_try!(parse_option(request.method.as_deref(), CbpsMethod::Over));
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let cov_refs: Vec<&str> = request.covariates.iter().map(|s| s.as_str()).collect();

        let mut config = CbpsConfig::default();
        config.method = method;
        config.balance_threshold = request.balance_threshold.unwrap_or(0.1);
        config.tolerance = 1e-8;
        config.max_iter = 100;

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
    pub async fn treatment_weightit(
        &self,
        Parameters(request): Parameters<WeightItRequest>,
    ) -> Result<CallToolResult, McpError> {
        let method = tool_try!(parse_option(
            request.method.as_deref(),
            WeightMethod::Logistic
        ));
        let estimand = tool_try!(parse_option(
            request.estimand.as_deref(),
            WeightEstimand::ATE
        ));
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let cov_refs: Vec<&str> = request.covariates.iter().map(|s| s.as_str()).collect();

        let mut config = WeightItConfig::default();
        config.method = method;
        config.estimand = estimand;
        config.intercept = true;
        config.stabilize = request.stabilize.unwrap_or(false);
        config.trim_quantile = request.trim_quantile.unwrap_or(1.0);
        config.max_iter = 200;
        config.tolerance = 1e-8;

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
    pub async fn treatment_entropy_balance(
        &self,
        Parameters(request): Parameters<EntropyBalanceRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

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
    pub async fn treatment_sbw(
        &self,
        Parameters(request): Parameters<SBWRequest>,
    ) -> Result<CallToolResult, McpError> {
        let estimand = tool_try!(parse_option(request.estimand.as_deref(), SBWEstimand::ATT));
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let cov_refs: Vec<&str> = request.covariates.iter().map(|s| s.as_str()).collect();

        let mut config = SBWConfig::default();
        config.estimand = estimand;
        config.balance_tol = request.balance_tol.unwrap_or(0.0);
        config.min_weight = request.min_weight.unwrap_or(0.0);
        config.normalize_to_n = true;
        config.max_iter = 1000;
        config.tolerance = 1e-8;
        config.balance_penalty = request.balance_penalty.unwrap_or(1000.0);

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
    pub async fn treatment_twang(
        &self,
        Parameters(request): Parameters<TwangRequest>,
    ) -> Result<CallToolResult, McpError> {
        let stop_method = tool_try!(choice(
            &STOP_METHOD,
            request.stop_method.as_deref(),
            StopMethod::ESMean
        ));
        let estimand = tool_try!(choice(
            &TWANG_ESTIMAND,
            request.estimand.as_deref(),
            TwangEstimand::ATT
        ));
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let cov_refs: Vec<&str> = request.covariates.iter().map(|s| s.as_str()).collect();

        let mut config = TwangConfig::default();
        config.n_trees = request.n_trees.unwrap_or(3000);
        config.shrinkage = request.shrinkage.unwrap_or(0.01);
        config.stop_method = stop_method;
        config.estimand = estimand;
        config.balance_threshold = request.balance_threshold.unwrap_or(0.1);
        config.min_iterations = 100;
        config.interaction_depth = 1;
        config.min_node_size = 10;

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
        description = "Perform propensity score matching to create balanced comparison groups for causal inference. Methods: 'nearest' (nearest neighbor matching on propensity score), 'cem' (coarsened exact matching on covariate bins), 'full' (optimal full matching), 'subclass' (propensity score subclassification). Returns matched sample with weights, balance diagnostics (standardized mean differences, variance ratios, KS statistics) before and after matching. Low SMD (<0.1) indicates good balance. Returns a matched dataset. For direct effect estimation via reweighting, use treatment_ipw instead."
    )]
    pub async fn propensity_matching(
        &self,
        Parameters(request): Parameters<MatchItRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::econometrics::matching::MatchMethodKind;
        let method_kind = tool_try!(parse_option(
            request.method.as_deref(),
            MatchMethodKind::Nearest
        ));
        let distance = Some(tool_try!(parse_option(
            request.distance.as_deref(),
            DistanceMethod::Logit
        )));
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let cov_refs: Vec<&str> = request.covariates.iter().map(|s| s.as_str()).collect();

        let method = match method_kind {
            MatchMethodKind::Cem => MatchMethod::CoarsenedExact {
                cutpoints: None,
                n_bins: Some(4),
            },
            MatchMethodKind::Full => MatchMethod::Full {
                min_ratio: 0.5,
                max_ratio: 2.0,
            },
            MatchMethodKind::Subclass => MatchMethod::Subclass {
                n_subclasses: request.n_subclasses.unwrap_or(5),
            },
            MatchMethodKind::Nearest => MatchMethod::NearestNeighbor {
                ratio: request.ratio.unwrap_or(1),
                caliper: request.caliper,
                replace: request.replace.unwrap_or(false),
            },
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
    pub async fn treatment_tmle(
        &self,
        Parameters(request): Parameters<TmleRequest>,
    ) -> Result<CallToolResult, McpError> {
        let q_model = tool_try!(parse_option(request.q_model.as_deref(), QModel::Logistic));
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let cov_refs: Vec<&str> = request.covariates.iter().map(|s| s.as_str()).collect();

        // Propensity score truncation bounds
        let ps_lower = request.ps_lower.unwrap_or(0.01);
        let ps_upper = request.ps_upper.unwrap_or(0.99);

        let mut config = TmleConfig::default();
        config.q_model = q_model;
        config.g_model = GModel::Logistic;
        config.truncate_ps = (ps_lower, ps_upper);
        config.max_iter = 100;
        config.tolerance = 1e-8;

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
    pub async fn collaborative_tmle(
        &self,
        Parameters(request): Parameters<CTmleRequest>,
    ) -> Result<CallToolResult, McpError> {
        let q_model = tool_try!(parse_option(
            request.q_model.as_deref(),
            CTmleQModel::Logistic
        ));
        let stopping_rule = match request.stopping_rule.as_deref() {
            None => StoppingRule::CVMinimum,
            Some(s) => tool_try!(
                StoppingRule::parse_with(s, request.max_covariates).map_err(option_message)
            ),
        };
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let cov_refs: Vec<&str> = request.covariates.iter().map(|s| s.as_str()).collect();

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
    pub async fn gformula(
        &self,
        Parameters(request): Parameters<GFormulaRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::econometrics::gformula::GFormulaRegime;
        let regime = tool_try!(parse_option(
            request.intervention.as_deref(),
            GFormulaRegime::AlwaysTreat
        ));
        let intervention = tool_try!(
            GFormulaIntervention::from_regime(
                regime,
                request.threshold_variable,
                request.threshold_cutoff,
                request.threshold_above,
            )
            .map_err(option_message)
        );
        let outcome_type = tool_try!(parse_option(
            request.outcome_type.as_deref(),
            GFormulaOutcomeType::Continuous
        ));
        use ndarray::{Array1, Array2};
        use p2a_core::linalg::design::DesignMatrix;

        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let n = dataset.df().height();
        let t_max = request.time_points;

        // Validate time points are within reasonable limits
        if t_max > MAX_TIME_POINTS {
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "time_points ({}) exceeds maximum allowed ({})",
                t_max, MAX_TIME_POINTS
            ))]));
        }

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

        // Use per-tool seed if provided, otherwise fall back to global seed
        let global_seed = self.global_seed.read().await;
        let seed = request.seed.or(*global_seed);

        // Build config
        let mut config = GFormulaConfig::default();
        config.n_simulations = request.n_simulations.unwrap_or(1000);
        config.time_points = t_max;
        config.intervention = intervention;
        config.outcome_type = outcome_type;
        config.n_bootstrap = request.n_bootstrap.unwrap_or(200);
        config.seed = seed;
        config.confidence_level = request.confidence_level.unwrap_or(0.95);
        config.max_iter = 100;
        config.tolerance = 1e-8;

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
    pub async fn mediation_analysis(
        &self,
        Parameters(request): Parameters<MediationRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let cov_refs: Vec<&str> = request.covariates.iter().map(|s| s.as_str()).collect();

        let mut config = MediationConfig::default();
        config.bootstrap = request.bootstrap.unwrap_or(999);
        config.trim = request.trim.unwrap_or(0.05);
        config.seed = None;

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
    pub async fn natural_effects_mediation(
        &self,
        Parameters(request): Parameters<NaturalEffectsRequest>,
    ) -> Result<CallToolResult, McpError> {
        let scale = tool_try!(parse_option(
            request.scale.as_deref(),
            EffectScale::Difference
        ));
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let conf_refs: Vec<&str> = request
            .confounders
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default();

        let n_bootstrap = request.n_bootstrap.unwrap_or(1000);
        let mut config = MedflexConfig::default();
        config.allow_interaction = request.allow_interaction.unwrap_or(true);
        config.bootstrap_ci = n_bootstrap > 0;
        config.n_bootstrap = n_bootstrap;
        config.confidence_level = request.confidence_level.unwrap_or(0.95);
        config.scale = scale;
        config.seed = None;

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
    pub async fn synthetic_control(
        &self,
        Parameters(request): Parameters<SyntheticControlRequest>,
    ) -> Result<CallToolResult, McpError> {
        let mut predictors: Vec<PredictorSpec> = Vec::with_capacity(request.predictors.len());
        for spec in &request.predictors {
            predictors.push(PredictorSpec {
                column: spec.column.clone(),
                aggregation: tool_try!(parse_option(
                    spec.aggregation.as_deref(),
                    TimeAggregation::Mean
                )),
                time_window: spec.time_window,
            });
        }
        let v_method = tool_try!(
            VOptimization::from_options(
                request.v_method.as_deref(),
                request.custom_v_weights.clone()
            )
            .map_err(option_message)
        );

        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let mut config = SynthConfig::default();
        config.treatment_time = request.treatment_time;
        config.treated_unit = request.treated_unit.clone();
        config.optimization_window = request.optimization_window;
        config.v_method = v_method;
        config.tolerance = request.tolerance.unwrap_or(1e-6);
        config.max_iter = request.max_iter.unwrap_or(1000);
        config.run_placebos = request.run_placebos.unwrap_or(false);
        config.weight_threshold = request.weight_threshold.unwrap_or(0.001);

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
    pub async fn gsynth(
        &self,
        Parameters(request): Parameters<GsynthRequest>,
    ) -> Result<CallToolResult, McpError> {
        let estimator = tool_try!(parse_option(
            request.estimator.as_deref(),
            GsynthEstimator::Ife
        ));
        let force = tool_try!(parse_option(request.force.as_deref(), GsynthForce::Unit));
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

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
        Returns donor weights, treatment effects with prediction intervals, variance decomposition, and pre-treatment fit statistics. \
        Units may be identified by text or integer ids; the panel must be balanced."
    )]
    pub async fn scpi(
        &self,
        Parameters(request): Parameters<ScpiRequest>,
    ) -> Result<CallToolResult, McpError> {
        let constraint = tool_try!(
            SCPIConstraint::from_options(request.constraint.as_deref(), request.lambda)
                .map_err(option_message)
        );
        let variance_method = tool_try!(parse_option(
            request.variance_method.as_deref(),
            VarianceMethod::Subgaussian
        ));
        use ndarray::{Array1, Array2};
        use p2a_core::data::columns::{ColumnRequest, NaPolicy};

        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        // Units (text or integer ids, as factor levels), whole-number times and the outcome
        // in one extractor request; a missing value is a tool error naming the column.
        let cols = match ColumnRequest::new(dataset.df())
            .factor(&request.unit_col)
            .time_index(&request.time_col)
            .numeric(&request.outcome)
            .na(NaPolicy::Fail)
            .extract()
        {
            Ok(c) => c,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        };
        let units = cols.factor(&request.unit_col);
        let time_of_row = cols.time_index(&request.time_col);
        let outcome = cols.numeric(&request.outcome);

        // Units in level order (integer ids numerically, text in byte order), times sorted
        let mut times: Vec<i64> = time_of_row.to_vec();
        times.sort_unstable();
        times.dedup();

        // Validate treated unit exists
        let Some(treated) = units.levels.iter().position(|u| u == &request.treated_unit) else {
            let shown: Vec<&str> = units.levels.iter().take(10).map(String::as_str).collect();
            let n_units = units.levels.len();
            let label = if n_units > 10 {
                format!("Available units (first 10 of {n_units})")
            } else {
                "Available units".to_string()
            };
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "Treated unit '{}' not found in '{}'. {}: {}",
                request.treated_unit,
                request.unit_col,
                label,
                shown.join(", ")
            ))]));
        };

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

        let n_units = units.n_levels();
        let n_times = times.len();
        let grid_len = match p2a_core::resource::checked_len(&[n_units, n_times], "scpi panel") {
            Ok(len) => len,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        };
        // A balanced panel has exactly one row per cell, so the grid never exceeds the rows.
        // With fewer rows than cells, name the first empty cell from the sorted cells of the
        // rows (it is among the first rows + 1 cells), without allocating the grid.
        let rows = units.codes.len();
        if grid_len > rows {
            let mut cells: Vec<usize> =
                match p2a_core::resource::try_with_capacity(rows, "scpi panel") {
                    Ok(v) => v,
                    Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
                };
            for (&code, &time) in units.codes.iter().zip(time_of_row) {
                if let Ok(t) = times.binary_search(&time) {
                    cells.push(code as usize * n_times + t);
                }
            }
            cells.sort_unstable();
            cells.dedup();
            let first = cells
                .iter()
                .enumerate()
                .position(|(i, &c)| i != c)
                .unwrap_or(cells.len());
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "scpi needs a balanced panel: {n_units} units x {n_times} times = {grid_len} \
                 cells, but the data has {rows} rows (unit {} has no row for time {}; cells \
                 without a row: {})",
                units.levels[first / n_times],
                times[first % n_times],
                grid_len - cells.len()
            ))]));
        }
        // One pass over the rows fills the units x times grid.
        let mut grid: Vec<Option<f64>> =
            match p2a_core::resource::try_vec(grid_len, None, "scpi panel") {
                Ok(g) => g,
                Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
            };
        for (i, (&code, &time)) in units.codes.iter().zip(time_of_row).enumerate() {
            let u = code as usize;
            let Ok(t) = times.binary_search(&time) else {
                continue; // every time is in `times` by construction
            };
            let cell = &mut grid[u * n_times + t];
            if cell.is_some() {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "scpi needs one row per (unit, time): unit {} appears twice at time {}",
                    units.levels[u], time
                ))]));
            }
            *cell = Some(outcome[i]);
        }
        // A guard: with no second row per cell and no more cells than rows, every cell is set.
        if let Some(first) = grid.iter().position(Option::is_none) {
            let k = grid.iter().filter(|c| c.is_none()).count();
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "scpi needs a balanced panel: unit {} has no row for time {} (cells without a \
                 row: {})",
                units.levels[first / n_times],
                times[first % n_times],
                k
            ))]));
        }
        let value = |u: usize, t: usize| grid[u * n_times + t].unwrap_or(f64::NAN);

        // Treated series and donor matrix; donors are the other units in level order
        let donor_units: Vec<String> = units
            .levels
            .iter()
            .enumerate()
            .filter(|(u, _)| *u != treated)
            .map(|(_, name)| name.clone())
            .collect();
        let donor_codes: Vec<usize> = (0..n_units).filter(|&u| u != treated).collect();
        let n_donors = donor_codes.len();
        let treated_values = Array1::from_iter((0..n_times).map(|t| value(treated, t)));
        let mut donor_matrix = Array2::zeros((n_times, n_donors));
        for (d_idx, &u) in donor_codes.iter().enumerate() {
            for t in 0..n_times {
                donor_matrix[[t, d_idx]] = value(u, t);
            }
        }

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
        description = "Run Sharp Regression Discontinuity (RD) estimation by local polynomial regression, with the bias-corrected robust inference of Calonico, Cattaneo & Titiunik (2014) as in R's rdrobust: conventional, bias-corrected and robust estimates, standard errors, p-values and confidence intervals. Without h, h and b are selected as rdrobust::rdbwselect (bwselect: mserd by default, msetwo, msesum, msecomb1, msecomb2, cerrd, certwo, cersum, cercomb1, cercomb2); the output names the source of each bandwidth; vce selects the variance estimator (nearest-neighbour by default, as rdrobust). Rows with missing values are dropped and reported."
    )]
    pub async fn rd_estimate(
        &self,
        Parameters(request): Parameters<RdEstimateRequest>,
    ) -> Result<CallToolResult, McpError> {
        // the option names are checked before the dataset lookup, so a bad name never
        // costs a data read
        let vce = match rd_option(request.vce.as_deref(), VceType::Nn, VceType::parse) {
            Ok(v) => v,
            Err(m) => return Ok(CallToolResult::error(vec![Content::text(m)])),
        };
        let kernel = match rd_option(
            request.kernel.as_deref(),
            KernelType::Triangular,
            KernelType::parse,
        ) {
            Ok(k) => k,
            Err(m) => return Ok(CallToolResult::error(vec![Content::text(m)])),
        };
        let bwselect = match rd_option(
            request.bwselect.as_deref(),
            BandwidthMethod::MseRd,
            BandwidthMethod::parse,
        ) {
            Ok(b) => b,
            Err(m) => return Ok(CallToolResult::error(vec![Content::text(m)])),
        };

        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let cutoff = request.cutoff.unwrap_or(0.0);

        let config = RdConfig {
            p: request.p.unwrap_or(1),
            q: None, // auto = p + 1
            h: request.h,
            b: request.b,
            kernel,
            bwselect,
            vce,
            level: request.level.unwrap_or(0.95),
            scaleregul: request.scaleregul.unwrap_or(1.0),
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
        description = "Select RD bandwidths h and b as rdrobust::rdbwselect (the ten bwselect methods, kernel, p, scaleregul) without estimating; returns h and b left and right, the observations inside h, the pilot bandwidth and mass-point information."
    )]
    pub async fn rd_bw(
        &self,
        Parameters(request): Parameters<RdBandwidthRequest>,
    ) -> Result<CallToolResult, McpError> {
        // the option names are checked before the dataset lookup
        let kernel = match rd_option(
            request.kernel.as_deref(),
            KernelType::Triangular,
            KernelType::parse,
        ) {
            Ok(k) => k,
            Err(m) => return Ok(CallToolResult::error(vec![Content::text(m)])),
        };
        let bwselect = match rd_option(
            request.bwselect.as_deref(),
            BandwidthMethod::MseRd,
            BandwidthMethod::parse,
        ) {
            Ok(b) => b,
            Err(m) => return Ok(CallToolResult::error(vec![Content::text(m)])),
        };

        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let cutoff = request.cutoff.unwrap_or(0.0);
        let config = RdConfig {
            p: request.p.unwrap_or(1),
            kernel,
            bwselect,
            scaleregul: request.scaleregul.unwrap_or(1.0),
            ..Default::default()
        };

        let result = match rd_bandwidth_with(
            dataset,
            &request.outcome,
            &request.running_var,
            None,
            cutoff,
            &config,
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
        description = "Run Fuzzy Regression Discontinuity estimation, for designs where the treatment probability (not assignment) jumps at the cutoff. Fits the outcome and the treatment on one bandwidth and reports the Wald ratio (LATE) with its joint conventional standard error and the bias-corrected robust estimate and interval, as rdrobust(fuzzy = ), plus the first-stage jump and F statistic (warning below 10). Without h, h and b are selected on the outcome as rdrobust::rdbwselect (bwselect: mserd by default, msetwo, msesum, msecomb1, msecomb2, cerrd, certwo, cersum, cercomb1, cercomb2); the output names the source of each bandwidth."
    )]
    pub async fn rd_fuzzy(
        &self,
        Parameters(request): Parameters<FuzzyRdRequest>,
    ) -> Result<CallToolResult, McpError> {
        // the option names are checked before the dataset lookup, so a bad name never
        // costs a data read
        let vce = match rd_option(request.vce.as_deref(), VceType::Nn, VceType::parse) {
            Ok(v) => v,
            Err(m) => return Ok(CallToolResult::error(vec![Content::text(m)])),
        };
        let kernel = match rd_option(
            request.kernel.as_deref(),
            KernelType::Triangular,
            KernelType::parse,
        ) {
            Ok(k) => k,
            Err(m) => return Ok(CallToolResult::error(vec![Content::text(m)])),
        };
        let bwselect = match rd_option(
            request.bwselect.as_deref(),
            BandwidthMethod::MseRd,
            BandwidthMethod::parse,
        ) {
            Ok(b) => b,
            Err(m) => return Ok(CallToolResult::error(vec![Content::text(m)])),
        };

        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let cutoff = request.cutoff.unwrap_or(0.0);

        let config = RdConfig {
            p: request.p.unwrap_or(1),
            q: None,
            h: request.h,
            b: request.b,
            kernel,
            bwselect,
            vce,
            level: request.level.unwrap_or(0.95),
            scaleregul: request.scaleregul.unwrap_or(1.0),
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
        description = "Run Multi-Cutoff Regression Discontinuity (rdmulti) estimation. Handles RD designs with multiple cutoffs (different thresholds) sharing the same running variable. Estimates cutoff-specific effects and optionally pools them into a single weighted estimate. Includes a heterogeneity test for whether effects differ across cutoffs. Reference: Cattaneo, Titiunik & Vazquez-Bare (2020). Each cutoff-specific estimate is rdrobust's bias-corrected estimate with its robust standard error; vce selects the variance estimator. Without a bandwidth, each cutoff's h and b are selected as rdrobust::rdbwselect (bwselect: mserd by default, msetwo, msesum, msecomb1, msecomb2, cerrd, certwo, cersum, cercomb1, cercomb2)."
    )]
    pub async fn rd_multi(
        &self,
        Parameters(request): Parameters<RdMultiRequest>,
    ) -> Result<CallToolResult, McpError> {
        let pooling_weights = tool_try!(choice(
            &POOLING_WEIGHTS,
            request.pooling_weights.as_deref(),
            PoolingWeights::SampleSize
        ));
        // the option names are checked before the dataset lookup, so a bad name never
        // costs a data read
        let vce = match rd_option(request.vce.as_deref(), VceType::Nn, VceType::parse) {
            Ok(v) => v,
            Err(m) => return Ok(CallToolResult::error(vec![Content::text(m)])),
        };
        let kernel = match rd_option(
            request.kernel.as_deref(),
            KernelType::Triangular,
            KernelType::parse,
        ) {
            Ok(k) => k,
            Err(m) => return Ok(CallToolResult::error(vec![Content::text(m)])),
        };
        let bwselect = tool_try!(rd_option(
            request.bwselect.as_deref(),
            BandwidthMethod::MseRd,
            BandwidthMethod::parse
        ));

        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        if request.cutoffs.is_empty() {
            return Ok(CallToolResult::error(vec![Content::text(
                "At least one cutoff value must be specified in 'cutoffs'.",
            )]));
        }

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

        let mut config = RdMultiConfig::default();
        config.cutoffs = request.cutoffs.clone();
        config.bandwidth = bandwidth;
        config.kernel = kernel;
        config.p = request.p.unwrap_or(1);
        config.q = None;
        config.pooled = request.pooled.unwrap_or(true);
        config.pooling_weights = pooling_weights;
        config.bwselect = bwselect;
        config.scaleregul = request.scaleregul.unwrap_or(1.0);
        config.vce = vce;
        config.level = request.level.unwrap_or(0.95);
        config.test_heterogeneity = request.test_heterogeneity.unwrap_or(true);

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
        description = "Run Difference-in-Differences (DiD) estimation. Estimates causal treatment effects by comparing treated vs control groups before and after treatment. For staggered treatment adoption across multiple periods, use staggered_did instead."
    )]
    pub async fn diff_in_diff(
        &self,
        Parameters(request): Parameters<DiDRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let result = match run_did(
            dataset,
            &request.dep_var,
            &request.treatment_var,
            &request.post_var,
            None,
            request.cluster_var.as_deref(),
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
    pub async fn evalue(
        &self,
        Parameters(request): Parameters<EValueRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::regression::{evalue_hr, evalue_or, evalue_rd, evalue_rr_ci, evalue_smd};

        let effect_type = tool_try!(choice(
            &EFFECT_MEASURE,
            Some(request.effect_type.as_str()),
            EffectMeasure::Rr
        ));

        let result = match effect_type {
            EffectMeasure::Rr => evalue_rr_ci(request.point, request.ci_lower, request.ci_upper),
            EffectMeasure::Or => {
                let rare = request.rare.unwrap_or(true);
                evalue_or(request.point, request.ci_lower, request.ci_upper, rare)
            }
            EffectMeasure::Hr => {
                let rare = request.rare.unwrap_or(true);
                evalue_hr(request.point, request.ci_lower, request.ci_upper, rare)
            }
            EffectMeasure::Smd => evalue_smd(request.point, request.se),
            EffectMeasure::Rd => {
                let baseline = request.baseline_risk.ok_or_else(|| {
                    McpError::invalid_params("baseline_risk is required for risk difference", None)
                })?;
                evalue_rd(request.point, baseline, request.se)
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
    pub async fn gmm_iv(
        &self,
        Parameters(request): Parameters<GeneralGmmIvRequest>,
    ) -> Result<CallToolResult, McpError> {
        let method = tool_try!(parse_option(request.method.as_deref(), GmmMethod::TwoStep));
        let vcov = tool_try!(parse_option(request.vcov.as_deref(), GmmVcov::HAC));
        let kernel = tool_try!(parse_option(request.kernel.as_deref(), HacKernel::Bartlett));
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();
        let z_refs: Vec<&str> = request.z.iter().map(|s| s.as_str()).collect();

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
        description = "Estimate causal effects of time-varying treatments using Longitudinal TMLE. LTMLE extends standard TMLE to longitudinal settings with multiple time points where treatments and confounders vary over time. Uses sequential regression (g-computation) combined with a targeting step at each time point to achieve double robustness. Estimates E[Y^{always treat}] - E[Y^{never treat}] under static intervention regimes. Returns ATE estimate, counterfactual means, influence curve-based standard errors, and diagnostics. Columns must not contain missing values (p2a has no censoring node); integer and boolean columns are accepted."
    )]
    pub async fn ltmle(
        &self,
        Parameters(request): Parameters<LtmleRequest>,
    ) -> Result<CallToolResult, McpError> {
        let q_model = tool_try!(parse_option(
            request.q_model.as_deref(),
            p2a_core::econometrics::LtmleQModel::Linear
        ));
        use p2a_core::econometrics::{LtmleConfig, LtmleData, LtmleQModel, run_ltmle};

        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

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

        // Every node is read in one extractor request. p2a's LTMLE has no censoring node,
        // so a missing value is a tool error naming the column (NaPolicy::Fail): dropping
        // the row would change the sample of every time point.
        let n = dataset.df().height();
        let cov_lists: Vec<Vec<&str>> = request
            .covariates
            .iter()
            .map(|list| list.split(',').map(str::trim).collect())
            .collect();
        for names_t in &cov_lists {
            let k = names_t.len();
            if n.checked_mul(k)
                .is_none_or(|total| total > MAX_MATRIX_ELEMENTS)
            {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Covariate matrix dimensions too large: {} rows x {} covariates exceeds limit of {} elements.",
                    n, k, MAX_MATRIX_ELEMENTS
                ))]));
            }
        }
        let mut names: Vec<&str> = request.outcomes.iter().map(String::as_str).collect();
        names.extend(request.treatments.iter().map(String::as_str));
        names.extend(cov_lists.iter().flatten().copied());
        let cols = match crate::tools::common::numeric_columns(
            dataset,
            &names,
            crate::tools::common::NaPolicy::Fail,
        ) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "LTMLE: {e}"
                ))]));
            }
        };
        let outcomes: Vec<ndarray::Array1<f64>> = request
            .outcomes
            .iter()
            .map(|c| cols.numeric_array(c))
            .collect();
        let treatments: Vec<ndarray::Array1<f64>> = request
            .treatments
            .iter()
            .map(|c| cols.numeric_array(c))
            .collect();
        let covariates: Vec<ndarray::Array2<f64>> = match cov_lists
            .iter()
            .map(|names_t| cols.matrix(names_t, false).map(|m| m.data))
            .collect()
        {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "LTMLE: {e}"
                ))]));
            }
        };

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
    pub async fn marginal_effects(
        &self,
        Parameters(request): Parameters<MarginalEffectsRequest>,
    ) -> Result<CallToolResult, McpError> {
        let model_type = tool_try!(choice(
            &MARGINAL_MODEL,
            request.model.as_deref(),
            MarginalModel::Ols
        ));
        use p2a_core::regression::marginal_effects;

        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

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
    pub async fn regression_standardization(
        &self,
        Parameters(request): Parameters<StdRegRequest>,
    ) -> Result<CallToolResult, McpError> {
        let model_type = tool_try!(parse_option(
            request.model_type.as_deref(),
            StdRegModel::Linear
        ));
        let estimand = tool_try!(parse_option(
            request.estimand.as_deref(),
            StdRegEstimand::ATE
        ));
        let se_method = tool_try!(parse_option(
            request.se_method.as_deref(),
            SEMethod::Bootstrap
        ));
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let cov_refs: Vec<&str> = request.covariates.iter().map(|s| s.as_str()).collect();

        let mut config = StdRegConfig::default();
        config.model_type = model_type;
        config.estimand = estimand;
        config.se_method = se_method;
        config.n_bootstrap = request.n_bootstrap.unwrap_or(999);
        config.confidence_level = request.confidence_level.unwrap_or(0.95);
        config.include_interactions = request.interactions.unwrap_or(false);
        config.seed = None;
        config.max_iter = 100;
        config.tolerance = 1e-8;

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
    pub async fn sensemakr(
        &self,
        Parameters(request): Parameters<SensemakrRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::regression::{CovarianceType, generate_contour_data, run_ols, run_sensemakr};

        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::common::test_support::{server_with, text};
    use p2a_core::polars::prelude::{DataFrame, DataType, NamedFrom, Series};
    use p2a_validation::Case;
    use serde_json::{Value, json};

    /// The case and a server holding its dataset as `rd`.
    fn case_server(id: &str) -> (Case, tokio::runtime::Runtime, AnalyticsServer) {
        let case = Case::load(id);
        let df = p2a_core::data::DataLoader::load_csv(case.dataset_path()).unwrap();
        let (rt, server) = server_with(vec![("rd", df)]);
        (case, rt, server)
    }

    /// `rd_sim.csv` as data only (no R value is compared).
    fn rd_frame() -> DataFrame {
        p2a_core::data::DataLoader::load_csv(p2a_validation::repo_path(
            "validation/datasets/generated/rd_sim.csv",
        ))
        .unwrap()
    }

    /// `col` cast to Float64 with the listed rows set to null.
    fn with_nulls(df: &DataFrame, col: &str, rows: &[usize]) -> DataFrame {
        let mut df = df.clone();
        let cast = df.column(col).unwrap().cast(&DataType::Float64).unwrap();
        let mut vals: Vec<Option<f64>> = cast.f64().unwrap().into_iter().collect();
        for &r in rows {
            vals[r] = None;
        }
        df.with_column(Series::new(col.into(), vals)).unwrap();
        df
    }

    /// One row of the RD estimate table (core's `Display` format).
    fn rd_row(label: &str, est: f64, se: f64, p: f64, ci: &[f64]) -> String {
        format!(
            "{:<20} {:>12.4} {:>12.4} {:>10.2} {:>10.4}  [{:>8.4}, {:>8.4}]",
            label,
            est,
            se,
            est / se,
            p,
            ci[0],
            ci[1]
        )
    }

    /// The Conventional and Robust rows of an rdrobust run `<prefix>_*`.
    fn rows(case: &Case, prefix: &str) -> [String; 2] {
        let v = |k: &str| case.scalar(&format!("{prefix}_{k}"));
        let ci = |k: &str| case.vector(&format!("{prefix}_{k}"));
        [
            rd_row(
                "Conventional",
                v("coef_conv"),
                v("se_conv"),
                v("p_conv"),
                &ci("ci_conv"),
            ),
            rd_row(
                "Robust",
                v("coef_bc"),
                v("se_robust"),
                v("p_robust"),
                &ci("ci_robust"),
            ),
        ]
    }

    /// `base` with the keys of `extra` added or replaced.
    fn merged(base: Value, extra: Value) -> Value {
        let mut v = base;
        for (k, x) in extra.as_object().unwrap() {
            v[k] = x.clone();
        }
        v
    }

    #[test]
    fn test_validate_mcp_rd_estimate_vce() {
        let (case, rt, server) = case_server("rd_sharp_fixed_bw");
        let base = json!({"dataset": "rd", "outcome": "y", "running_var": "x", "cutoff": 0.0, "h": 0.3, "b": 0.3});
        for (vce, prefix) in [
            (None, "tri_nn_hb"),
            (Some("nn"), "tri_nn_hb"),
            (Some("hc1"), "tri_hc1_hb"),
            (Some("HC3"), "tri_hc3_hb"),
        ] {
            let args = match vce {
                Some(v) => merged(base.clone(), json!({ "vce": v })),
                None => base.clone(),
            };
            let req: RdEstimateRequest = serde_json::from_value(args).unwrap();
            let r = rt.block_on(server.rd_estimate(Parameters(req))).unwrap();
            let out = text(&r);
            assert_ne!(r.is_error, Some(true), "{out}");
            for row in rows(&case, prefix) {
                assert!(out.contains(&row), "{vce:?}: {row}\n{out}");
            }
        }
        let req: RdEstimateRequest =
            serde_json::from_value(merged(base, json!({"level": 0.9}))).unwrap();
        let r = rt.block_on(server.rd_estimate(Parameters(req))).unwrap();
        let out = text(&r);
        let ci = case.vector("level90_ci_robust");
        assert!(out.contains("[90% CI]"), "{out}");
        assert!(
            out.contains(&format!("[{:>8.4}, {:>8.4}]", ci[0], ci[1])),
            "{out}"
        );
    }

    #[test]
    fn test_mcp_rd_estimate_rejects_unknown_vce() {
        // vce is checked before the dataset lookup, so no dataset is loaded
        let (rt, server) = server_with(vec![]);
        let base = json!({"dataset": "none", "outcome": "y", "running_var": "x", "vce": "robust"});
        let check = |r: CallToolResult| {
            let out = text(&r);
            assert_eq!(r.is_error, Some(true), "{out}");
            // the core message alone, without a second prefix
            assert!(
                out.starts_with("Unknown vce 'robust'; use \"nn\"") && out.contains("hc1"),
                "{out}"
            );
        };
        let req: RdEstimateRequest = serde_json::from_value(base.clone()).unwrap();
        check(rt.block_on(server.rd_estimate(Parameters(req))).unwrap());
        let req: FuzzyRdRequest =
            serde_json::from_value(merged(base.clone(), json!({"treatment": "d"}))).unwrap();
        check(rt.block_on(server.rd_fuzzy(Parameters(req))).unwrap());
        let req: RdMultiRequest =
            serde_json::from_value(merged(base, json!({"cutoffs": [0.0]}))).unwrap();
        check(rt.block_on(server.rd_multi(Parameters(req))).unwrap());
    }

    #[test]
    fn test_mcp_rd_rejects_unknown_kernel_and_bwselect() {
        // both names are checked before the dataset lookup, so no dataset is loaded
        let (rt, server) = server_with(vec![]);
        let base = json!({"dataset": "none", "outcome": "y", "running_var": "x"});
        let check = |r: CallToolResult, starts: &str, lists: &[&str]| {
            let out = text(&r);
            assert_eq!(r.is_error, Some(true), "{out}");
            // the core message alone, without a second prefix
            assert!(out.starts_with(starts), "{out}");
            for name in lists {
                assert!(out.contains(name), "{name}: {out}");
            }
        };
        let kernel_err = "Unknown kernel 'gaussian'; use \"triangular\"";
        let kernels = ["epanechnikov", "uniform", "\"epa\"", "\"uni\""];
        let k = merged(base.clone(), json!({"kernel": "gaussian"}));
        let req: RdEstimateRequest = serde_json::from_value(k.clone()).unwrap();
        check(
            rt.block_on(server.rd_estimate(Parameters(req))).unwrap(),
            kernel_err,
            &kernels,
        );
        let req: RdBandwidthRequest = serde_json::from_value(k.clone()).unwrap();
        check(
            rt.block_on(server.rd_bw(Parameters(req))).unwrap(),
            kernel_err,
            &kernels,
        );
        let req: FuzzyRdRequest =
            serde_json::from_value(merged(k.clone(), json!({"treatment": "d"}))).unwrap();
        check(
            rt.block_on(server.rd_fuzzy(Parameters(req))).unwrap(),
            kernel_err,
            &kernels,
        );
        let req: RdMultiRequest =
            serde_json::from_value(merged(k, json!({"cutoffs": [0.0]}))).unwrap();
        check(
            rt.block_on(server.rd_multi(Parameters(req))).unwrap(),
            kernel_err,
            &kernels,
        );

        let bw_err = "Unknown bwselect 'ik'; use \"mserd\"";
        let methods = ["msetwo", "cerrd", "certwo"];
        let b = merged(base, json!({"bwselect": "ik"}));
        let req: RdEstimateRequest = serde_json::from_value(b.clone()).unwrap();
        check(
            rt.block_on(server.rd_estimate(Parameters(req))).unwrap(),
            bw_err,
            &methods,
        );
        let req: RdBandwidthRequest = serde_json::from_value(b.clone()).unwrap();
        check(
            rt.block_on(server.rd_bw(Parameters(req))).unwrap(),
            bw_err,
            &methods,
        );
        let req: FuzzyRdRequest =
            serde_json::from_value(merged(b, json!({"treatment": "d"}))).unwrap();
        check(
            rt.block_on(server.rd_fuzzy(Parameters(req))).unwrap(),
            bw_err,
            &methods,
        );
    }

    #[test]
    fn test_mcp_rd_accepts_short_and_cased_names() {
        // the short names and any case select the named kernel and selector, never the
        // triangular / mserd fallback
        let (rt, server) = server_with(vec![("rd", rd_frame())]);
        let base = json!({"dataset": "rd", "outcome": "y", "running_var": "x", "cutoff": 0.0});
        for (kernel, bwselect, kernel_shown, method_shown) in [
            (
                "epa",
                "CERTWO",
                "Kernel: Epanechnikov",
                "certwo (CER-optimal, separate left and right)",
            ),
            (
                "uni",
                "mse_two",
                "Kernel: Uniform",
                "msetwo (MSE-optimal, separate left and right)",
            ),
            (
                "Uniform",
                "CerRd",
                "Kernel: Uniform",
                "cerrd (CER-optimal, one common bandwidth)",
            ),
            (
                "tri",
                "mserd",
                "Kernel: Triangular",
                "mserd (MSE-optimal, one common bandwidth)",
            ),
        ] {
            let args = merged(
                base.clone(),
                json!({"kernel": kernel, "bwselect": bwselect}),
            );
            let req: RdBandwidthRequest = serde_json::from_value(args.clone()).unwrap();
            let r = rt.block_on(server.rd_bw(Parameters(req))).unwrap();
            let out = text(&r);
            assert_ne!(r.is_error, Some(true), "{out}");
            assert!(out.contains(kernel_shown), "{kernel}: {out}");
            assert!(
                out.contains(&format!("Method: {method_shown}")),
                "{bwselect}: {out}"
            );
            let req: RdEstimateRequest = serde_json::from_value(args.clone()).unwrap();
            let r = rt.block_on(server.rd_estimate(Parameters(req))).unwrap();
            let out = text(&r);
            assert_ne!(r.is_error, Some(true), "{out}");
            assert!(out.contains(kernel_shown), "{kernel}: {out}");
            assert!(
                out.contains(&format!(
                    "Bandwidth source: h and b selected by {method_shown}"
                )),
                "{bwselect}: {out}"
            );
            let req: FuzzyRdRequest =
                serde_json::from_value(merged(args.clone(), json!({"treatment": "d"}))).unwrap();
            let r = rt.block_on(server.rd_fuzzy(Parameters(req))).unwrap();
            let out = text(&r);
            assert_ne!(r.is_error, Some(true), "{out}");
            assert!(out.contains(kernel_shown), "{kernel}: {out}");
            // rd_multi takes cutoffs (not cutoff); the selector name is accepted there too
            let mut multi_args = merged(args, json!({"cutoffs": [0.0], "bandwidth": 0.3}));
            let multi = multi_args.as_object_mut().unwrap();
            multi.remove("cutoff");
            let req: RdMultiRequest = serde_json::from_value(multi_args).unwrap();
            let r = rt.block_on(server.rd_multi(Parameters(req))).unwrap();
            let out = text(&r);
            assert_ne!(r.is_error, Some(true), "{out}");
            assert!(out.contains(kernel_shown), "{kernel}: {out}");
        }
    }

    #[test]
    fn test_validate_mcp_rd_fuzzy_b() {
        let (case, rt, server) = case_server("rd_fuzzy_fixed_bw");
        let base = json!({"dataset": "rd", "outcome": "yf", "running_var": "x", "treatment": "d"});
        for (extra, prefix) in [
            (json!({"h": 0.3, "b": 0.5}), "fuzzy_nn_h03_b05"),
            (json!({"h": 0.4, "vce": "hc1"}), "fuzzy_hc1"),
        ] {
            let req: FuzzyRdRequest = serde_json::from_value(merged(base.clone(), extra)).unwrap();
            let r = rt.block_on(server.rd_fuzzy(Parameters(req))).unwrap();
            let out = text(&r);
            assert_ne!(r.is_error, Some(true), "{out}");
            for row in rows(&case, prefix) {
                assert!(out.contains(&row), "{prefix}: {row}\n{out}");
            }
            let f = (case.scalar(&format!("{prefix}_first_stage_conv"))
                / case.scalar(&format!("{prefix}_first_stage_se_conv")))
            .powi(2);
            assert!(
                out.contains(&format!("F = {f:.2}")),
                "{prefix}: F = {f:.2}\n{out}"
            );
        }
    }

    #[test]
    fn test_validate_mcp_rd_multi_vce() {
        let (case, rt, server) = case_server("rd_sharp_fixed_bw");
        let req: RdMultiRequest = serde_json::from_value(json!({
            "dataset": "rd", "outcome": "y", "running_var": "x",
            "cutoffs": [0.0], "bandwidth": 0.3, "vce": "hc1"
        }))
        .unwrap();
        let r = rt.block_on(server.rd_multi(Parameters(req))).unwrap();
        let out = text(&r);
        assert_ne!(r.is_error, Some(true), "{out}");
        let cell = format!(
            "{:>10.4} {:>10.4}",
            case.scalar("tri_hc1_hb_coef_bc"),
            case.scalar("tri_hc1_hb_se_robust")
        );
        assert!(out.contains(&cell), "{cell}\n{out}");
    }

    #[test]
    fn test_mcp_rd_estimate_reports_dropped_rows() {
        let (rt, server) = server_with(vec![("rd", with_nulls(&rd_frame(), "y", &[0]))]);
        let req: RdEstimateRequest = serde_json::from_value(json!({
            "dataset": "rd", "outcome": "y", "running_var": "x", "h": 0.3
        }))
        .unwrap();
        let r = rt.block_on(server.rd_estimate(Parameters(req))).unwrap();
        let out = text(&r);
        assert_ne!(r.is_error, Some(true), "{out}");
        assert!(
            out.contains("Missing values: 1 of 1000 rows dropped because of missing values (y: 1)"),
            "{out}"
        );
    }

    // =========================================================================
    // treatment_double_ml extraction (T2-WP11: TOOL-5 DML site, TOOL-6)
    // =========================================================================

    /// `col` cast to Float64 (same values).
    fn as_f64(df: &DataFrame, col: &str) -> DataFrame {
        with_nulls(df, col, &[])
    }

    fn dml_request() -> DoubleMLRequest {
        serde_json::from_value(json!({
            "dataset": "rd", "outcome": "yf", "treatment": "d",
            "covariates": ["x", "ym"], "seed": 11
        }))
        .unwrap()
    }

    #[test]
    fn test_mcp_treatment_double_ml_accepts_int64_treatment() {
        let df = rd_frame();
        assert_eq!(df.column("d").unwrap().dtype(), &DataType::Int64);
        let run = |df: DataFrame| {
            let (rt, server) = server_with(vec![("rd", df)]);
            rt.block_on(server.treatment_double_ml(Parameters(dml_request())))
                .expect("data problems are tool errors")
        };
        let int = run(df.clone());
        assert_ne!(int.is_error, Some(true), "{}", text(&int));
        let float = run(as_f64(&df, "d"));
        assert_eq!(text(&int), text(&float));
    }

    #[test]
    fn test_mcp_treatment_double_ml_refuses_missing_values() {
        let df = rd_frame();
        let run = |df: DataFrame, req: DoubleMLRequest| {
            let (rt, server) = server_with(vec![("rd", df)]);
            let r = rt
                .block_on(server.treatment_double_ml(Parameters(req)))
                .expect("data problems are tool errors");
            assert_eq!(r.is_error, Some(true), "{}", text(&r));
            text(&r)
        };
        let out = run(with_nulls(&df, "x", &[5]), dml_request());
        assert!(
            out.contains("Column 'x' contains 1 missing value(s)"),
            "{out}"
        );
        let out = run(
            with_nulls(&with_nulls(&df, "yf", &[7]), "d", &[7]),
            dml_request(),
        );
        assert!(out.contains("Column 'yf'"), "{out}");
        let req: DoubleMLRequest = serde_json::from_value(json!({
            "dataset": "rd", "outcome": "yf", "treatment": "d", "covariates": ["x", "nope"]
        }))
        .unwrap();
        let out = run(df, req);
        assert!(out.contains("nope"), "{out}");
    }

    // =========================================================================
    // ltmle extraction (T2-WP11: TOOL-7, CAUS-24 handler sites)
    // =========================================================================

    fn ltmle_request() -> LtmleRequest {
        serde_json::from_value(json!({
            "dataset": "rd", "outcomes": ["y", "yf"], "treatments": ["d", "d"],
            "covariates": ["x", "ym"]
        }))
        .unwrap()
    }

    fn run_ltmle_on(df: DataFrame) -> CallToolResult {
        let (rt, server) = server_with(vec![("rd", df)]);
        rt.block_on(server.ltmle(Parameters(ltmle_request())))
            .expect("data problems are tool errors")
    }

    #[test]
    fn test_mcp_ltmle_accepts_int64_treatment() {
        let df = rd_frame();
        let int = run_ltmle_on(df.clone());
        assert_ne!(int.is_error, Some(true), "{}", text(&int));
        let float = run_ltmle_on(as_f64(&df, "d"));
        assert_eq!(text(&int), text(&float));
    }

    #[test]
    fn test_mcp_ltmle_refuses_missing_values() {
        let df = rd_frame();
        for (col, row, needle) in [
            ("yf", 3, "Column 'yf' contains 1 missing value(s)"),
            ("ym", 4, "Column 'ym'"),
            ("d", 2, "Column 'd'"),
        ] {
            let r = run_ltmle_on(with_nulls(&df, col, &[row]));
            let out = text(&r);
            assert_eq!(r.is_error, Some(true), "{col}: {out}");
            assert!(out.contains(needle), "{col}: {out}");
        }
    }

    // =========================================================================
    // scpi extraction (T2-WP11: TOOL-5 scpi site)
    // =========================================================================

    fn grunfeld() -> DataFrame {
        p2a_core::data::DataLoader::load_csv(p2a_validation::repo_path(
            "validation/datasets/grunfeld.csv",
        ))
        .unwrap()
    }

    fn run_scpi_on(df: DataFrame, extra: Value) -> CallToolResult {
        let req: ScpiRequest = serde_json::from_value(merged(
            json!({
                "dataset": "g", "outcome": "inv", "unit_col": "firm", "time_col": "year",
                "treated_unit": "1", "treatment_time": 1950
            }),
            extra,
        ))
        .unwrap();
        let (rt, server) = server_with(vec![("g", df)]);
        rt.block_on(server.scpi(Parameters(req)))
            .expect("data problems are tool errors")
    }

    #[test]
    fn test_mcp_scpi_accepts_integer_unit_ids() {
        let df = grunfeld();
        assert_eq!(df.column("firm").unwrap().dtype(), &DataType::Int64);
        let r = run_scpi_on(df, json!({}));
        let out = text(&r);
        assert_ne!(r.is_error, Some(true), "{out}");
        for needle in [
            "Pre-treatment periods: 15",
            "Post-treatment periods: 5",
            "Donor pool size: 9",
            "Donor Unit Names",
        ] {
            assert!(out.contains(needle), "{needle}\n{out}");
        }
    }

    #[test]
    fn test_mcp_scpi_reports_unknown_units_and_missing_cells() {
        let df = grunfeld();
        let err = |df: DataFrame, extra: Value| {
            let r = run_scpi_on(df, extra);
            assert_eq!(r.is_error, Some(true), "{}", text(&r));
            text(&r)
        };
        let out = err(df.clone(), json!({"treated_unit": "11"}));
        assert!(
            out.contains("Available units: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10"),
            "{out}"
        );
        let out = err(with_nulls(&df, "inv", &[0]), json!({}));
        assert!(
            out.contains("Column 'inv' contains 1 missing value(s)"),
            "{out}"
        );
        // row 25 (0-based) is firm 2, year 1940
        let keep: p2a_core::polars::prelude::BooleanChunked =
            (0..df.height()).map(|i| i != 25).collect();
        let out = err(df.filter(&keep).unwrap(), json!({}));
        assert!(
            out.contains("10 units x 20 times = 200 cells, but the data has 199 rows")
                && out.contains("unit 2 has no row for time 1940; cells without a row: 1)"),
            "{out}"
        );
        // row 0 (firm 1, 1935) a second time
        let out = err(df.vstack(&df.slice(0, 1)).unwrap(), json!({}));
        assert!(
            out.contains("one row per (unit, time): unit 1 appears twice at time 1935"),
            "{out}"
        );
    }

    #[test]
    fn test_mcp_scpi_says_the_unit_list_is_cut() {
        use polars::prelude::*;
        let df = grunfeld();
        let mut shifted = df.clone();
        let firm = shifted
            .column("firm")
            .unwrap()
            .as_materialized_series()
            .clone()
            + 10;
        shifted.with_column(firm.with_name("firm".into())).unwrap();
        let stacked = df.vstack(&shifted).unwrap();
        let r = run_scpi_on(stacked, json!({"treated_unit": "99"}));
        assert_eq!(r.is_error, Some(true), "{}", text(&r));
        let out = text(&r);
        assert!(
            out.contains("Available units (first 10 of 20): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10"),
            "{out}"
        );
    }

    #[test]
    fn test_local_option_tables_parse_aliases_and_reject_unknown() {
        STOP_METHOD.validate().unwrap();
        TWANG_ESTIMAND.validate().unwrap();
        POOLING_WEIGHTS.validate().unwrap();
        MARGINAL_MODEL.validate().unwrap();
        EFFECT_MEASURE.validate().unwrap();
        LTMLE_CONTRAST.validate().unwrap();
        assert_eq!(STOP_METHOD.parse("ES.Mean").unwrap(), StopMethod::ESMean);
        assert_eq!(STOP_METHOD.parse("ksmax").unwrap(), StopMethod::KSMax);
        assert_eq!(STOP_METHOD.parse("es-max").unwrap(), StopMethod::ESMax);
        assert_eq!(TWANG_ESTIMAND.parse("ATT").unwrap(), TwangEstimand::ATT);
        assert_eq!(
            POOLING_WEIGHTS.parse("iv").unwrap(),
            PoolingWeights::InverseVariance
        );
        assert_eq!(
            POOLING_WEIGHTS.parse("Sample Size").unwrap(),
            PoolingWeights::SampleSize
        );
        assert_eq!(
            MARGINAL_MODEL.parse("logistic").unwrap(),
            MarginalModel::Logit
        );
        assert_eq!(
            EFFECT_MEASURE.parse("Risk Ratio").unwrap(),
            EffectMeasure::Rr
        );
        assert_eq!(
            EFFECT_MEASURE.parse("hazard-ratio").unwrap(),
            EffectMeasure::Hr
        );
        assert_eq!(LTMLE_CONTRAST.parse("none").unwrap(), ContrastKind::None);
        match EFFECT_MEASURE.parse("hr2") {
            Err(p2a_core::EconError::InvalidSpecification { message }) => assert_eq!(
                message,
                "unknown effect_type 'hr2'; expected one of: rr, or, hr, smd, rd"
            ),
            other => panic!("{other:?}"),
        }
    }
}
