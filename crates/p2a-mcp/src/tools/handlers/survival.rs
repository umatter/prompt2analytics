//! Survival analysis tool handlers.
//!
//! This module contains MCP tool handlers for survival analysis methods including:
//! - Kaplan-Meier survival curves
//! - Log-rank test
//! - Cox proportional hazards model
//! - Accelerated failure time (AFT) models
//! - Competing risks analysis

use rmcp::{
    ErrorData as McpError, handler::server::wrapper::Parameters, model::*, tool, tool_router,
};

use p2a_core::{
    AftConfig, AftDistribution, CoxConfig, TiesMethod, log_rank_test, run_aft, run_competing_risks,
    run_cox_ph, run_kaplan_meier,
};

use crate::server::AnalyticsServer;
use crate::tools::requests::survival::{
    AftRequest, CompetingRisksRequest, CoxPhRequest, KaplanMeierRequest, LogRankRequest,
};

#[tool_router(router = survival_router, vis = "pub")]
impl AnalyticsServer {
    /// Run Kaplan-Meier survival curve estimation.
    #[tool(
        description = "Run Kaplan-Meier survival curve estimation. Computes the non-parametric product-limit estimator with Greenwood's variance formula for confidence intervals. Optionally computes stratified curves by group. Returns survival probabilities at each event time with standard errors and confidence intervals."
    )]
    pub async fn kaplan_meier(
        &self,
        Parameters(request): Parameters<KaplanMeierRequest>,
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

        let confidence_level = request.confidence_level.unwrap_or(0.95);

        let results = match run_kaplan_meier(
            dataset,
            &request.time,
            &request.event,
            request.group.as_deref(),
            confidence_level,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Kaplan-Meier estimation failed: {}",
                    e
                ))]));
            }
        };

        // Format all results
        let output: Vec<String> = results.iter().map(|r| r.to_string()).collect();
        Ok(CallToolResult::success(vec![Content::text(
            output.join("\n\n"),
        )]))
    }

    /// Run Log-Rank test comparing survival curves.
    #[tool(
        description = "Run Log-Rank test for comparing survival curves between groups. Tests the null hypothesis that the survival functions are equal across groups. Returns chi-squared statistic, p-value, and expected/observed event counts per group."
    )]
    pub async fn log_rank(
        &self,
        Parameters(request): Parameters<LogRankRequest>,
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

        let result = match log_rank_test(dataset, &request.time, &request.event, &request.group) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Log-rank test failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run Cox Proportional Hazards model.
    #[tool(
        description = "Run Cox Proportional Hazards (Cox PH) regression model. Semi-parametric regression estimating covariate effects on hazard rate without assuming a baseline hazard form. Uses Newton-Raphson optimization. Returns hazard ratios, coefficients, standard errors, confidence intervals, and concordance (C-index)."
    )]
    pub async fn cox_ph(
        &self,
        Parameters(request): Parameters<CoxPhRequest>,
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

        // Parse ties method
        let ties = match request.ties_method.as_deref() {
            Some("breslow") => TiesMethod::Breslow,
            _ => TiesMethod::Efron, // default
        };

        let cov_refs: Vec<&str> = request.covariates.iter().map(|s| s.as_str()).collect();

        let config = CoxConfig {
            ties,
            tolerance: request.tolerance.unwrap_or(1e-9),
            max_iter: request.max_iter.unwrap_or(100),
            robust_se: false,
        };

        let result = match run_cox_ph(
            dataset,
            &request.time,
            &request.event,
            &cov_refs,
            Some(config),
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cox PH estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run Accelerated Failure Time (AFT) model.
    #[tool(
        description = "Run Accelerated Failure Time (AFT) parametric survival model. Models log(T) as a linear function of covariates. Supports Weibull, Exponential, Log-Normal, and Log-Logistic distributions. Returns regression coefficients (as acceleration factors), standard errors, shape parameters, and AIC/BIC."
    )]
    pub async fn aft(
        &self,
        Parameters(request): Parameters<AftRequest>,
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

        // Parse distribution
        let distribution = match request.distribution.as_deref() {
            Some("exponential") => AftDistribution::Exponential,
            Some("lognormal") => AftDistribution::LogNormal,
            Some("loglogistic") => AftDistribution::LogLogistic,
            _ => AftDistribution::Weibull, // default
        };

        let cov_refs: Vec<&str> = request.covariates.iter().map(|s| s.as_str()).collect();

        let config = AftConfig {
            distribution,
            tolerance: request.tolerance.unwrap_or(1e-9),
            max_iter: request.max_iter.unwrap_or(100),
        };

        let result = match run_aft(
            dataset,
            &request.time,
            &request.event,
            &cov_refs,
            Some(config),
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "AFT estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run Competing Risks analysis (Aalen-Johansen estimator).
    #[tool(
        description = "Run Competing Risks analysis using the Aalen-Johansen estimator. Computes cumulative incidence functions (CIF) for multiple event types in the presence of competing risks. Properly accounts for subjects who experience a different event than the one of interest. Returns CIF curves with confidence intervals for each event type."
    )]
    pub async fn competing_risks(
        &self,
        Parameters(request): Parameters<CompetingRisksRequest>,
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

        let confidence_level = request.confidence_level.unwrap_or(0.95);

        let result =
            match run_competing_risks(dataset, &request.time, &request.event, confidence_level) {
                Ok(r) => r,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Competing risks analysis failed: {}",
                        e
                    ))]));
                }
            };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }
}
