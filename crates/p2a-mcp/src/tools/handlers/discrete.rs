//! Discrete choice model tools handlers.
//!
//! This module provides MCP tool handlers for discrete choice models:
//! - Binary choice (Logit, Probit)
//! - Multinomial logit (unordered categorical)
//! - McFadden's conditional logit (mlogit)
//! - Mixed logit (random parameters)
//! - Ordered logit/probit
//! - Count models (Negative binomial)
//! - Zero-inflated models (ZIP, ZINB)
//! - Hurdle models
//! - FEGLM (GLM with HDFE)

use rmcp::{
    ErrorData as McpError, handler::server::wrapper::Parameters, model::*, tool, tool_router,
};

use crate::server::AnalyticsServer;
use crate::tools::requests::discrete::{
    FeglmRequest, HurdleModelRequest, LogitRequest, MixedLogitRequest, MlogitRequest,
    MultinomRequest, NegBinRequest, OrderedRequest, ProbitRequest, ZeroInflRequest,
};

use p2a_core::{
    run_feglm, run_gmnl, run_hurdle, run_logit, run_mlogit, run_multinom, run_negbin,
    run_ordered_logit, run_ordered_probit, run_probit, run_zinb, run_zip, FeglmConfig, GlmFamily,
    HurdleType, MixedLogitConfig, RandomDistribution,
};

#[tool_router(router = discrete_router, vis = "pub")]
impl AnalyticsServer {
    /// Run Logit (logistic) regression for binary outcomes.
    #[tool(
        description = "Run Logit (logistic) regression for binary outcomes. Uses MLE with Newton-Raphson. Dependent variable must be 0/1."
    )]
    async fn logit(
        &self,
        Parameters(request): Parameters<LogitRequest>,
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

        let result = match run_logit(dataset, &request.y, &x_refs) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Logit estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run Probit regression.
    #[tool(
        description = "Run Probit regression for binary outcomes. Uses MLE with Newton-Raphson. Dependent variable must be 0/1."
    )]
    async fn probit(
        &self,
        Parameters(request): Parameters<ProbitRequest>,
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

        let result = match run_probit(dataset, &request.y, &x_refs) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Probit estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run multinomial logit regression for unordered categorical outcomes.
    #[tool(
        description = "Run multinomial logit regression for unordered categorical outcomes with 3+ categories. Uses MLE with Newton-Raphson. Equivalent to R's nnet::multinom()."
    )]
    async fn multinom(
        &self,
        Parameters(request): Parameters<MultinomRequest>,
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
        let reference = request.reference.as_deref();

        let result = match run_multinom(dataset, &request.y, &x_refs, reference) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Multinomial logit estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run McFadden's conditional logit (mlogit) for discrete choice analysis.
    #[tool(
        description = "Run McFadden's conditional logit (mlogit) for discrete choice analysis. Supports both alternative-specific variables (with generic coefficients) and individual-specific variables (with alternative-specific coefficients). Data must be in long format with one row per individual-alternative combination. Equivalent to R's mlogit::mlogit()."
    )]
    async fn mlogit(
        &self,
        Parameters(request): Parameters<MlogitRequest>,
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

        let alt_specific_refs: Vec<&str> =
            request.alt_specific.iter().map(|s| s.as_str()).collect();
        let ind_specific_refs: Vec<&str> =
            request.ind_specific.iter().map(|s| s.as_str()).collect();
        let reference = request.reference.as_deref();

        let result = match run_mlogit(
            dataset,
            &request.choice_id,
            &request.alt_id,
            &request.choice,
            &alt_specific_refs,
            &ind_specific_refs,
            reference,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Conditional logit (mlogit) estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run mixed logit (random parameters logit) for discrete choice with preference heterogeneity.
    #[tool(
        description = "Run mixed logit (random parameters logit) for discrete choice models with heterogeneous preferences. Allows coefficients to vary across individuals according to specified distributions (normal, lognormal, triangular, uniform). Uses Maximum Simulated Likelihood (MSL) with Halton sequences. Equivalent to R's gmnl::gmnl() or mixl::mixl()."
    )]
    async fn mixed_logit(
        &self,
        Parameters(request): Parameters<MixedLogitRequest>,
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

        let var_refs: Vec<&str> = request.variables.iter().map(|s| s.as_str()).collect();

        // Parse distribution
        let dist = match request.distribution.as_deref() {
            Some("lognormal") | Some("log-normal") => RandomDistribution::LogNormal,
            Some("triangular") => RandomDistribution::Triangular,
            Some("uniform") => RandomDistribution::Uniform,
            Some("fixed") => RandomDistribution::Fixed,
            _ => RandomDistribution::Normal,
        };

        // Convert random_vars to refs
        let random_refs: Option<Vec<&str>> = request
            .random_vars
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());

        // Build config
        let config = MixedLogitConfig {
            n_draws: request.n_draws.unwrap_or(500),
            halton: request.halton.unwrap_or(true),
            max_iter: 200,
            tolerance: 1e-6,
            seed: Some(42),
        };

        let result = match run_gmnl(
            dataset,
            &request.choice_id,
            &request.alt_id,
            &request.choice,
            &var_refs,
            random_refs.as_deref(),
            Some(dist),
            Some(config),
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Mixed logit estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run ordered logit/probit regression for ordinal outcomes.
    #[tool(
        description = "Run ordered logit or probit regression (proportional odds model) for ordered categorical outcomes. Estimates threshold (cut-point) parameters. Equivalent to R's MASS::polr()."
    )]
    async fn ordered_model(
        &self,
        Parameters(request): Parameters<OrderedRequest>,
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
        let model_type = request.model_type.as_deref().unwrap_or("logit");

        let result = match model_type.to_lowercase().as_str() {
            "logit" => run_ordered_logit(dataset, &request.y, &x_refs),
            "probit" => run_ordered_probit(dataset, &request.y, &x_refs),
            _ => {
                return Ok(CallToolResult::error(vec![Content::text(
                    "Invalid model_type. Use 'logit' or 'probit'.".to_string(),
                )]));
            }
        };

        match result {
            Ok(r) => Ok(CallToolResult::success(vec![Content::text(r.to_string())])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Ordered {} estimation failed: {}",
                model_type, e
            ))])),
        }
    }

    /// Run negative binomial regression for count data with overdispersion.
    #[tool(
        description = "Run negative binomial regression for count data with overdispersion (variance > mean). Estimates dispersion parameter (theta). Equivalent to R's MASS::glm.nb()."
    )]
    async fn negbin(
        &self,
        Parameters(request): Parameters<NegBinRequest>,
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

        let result = match run_negbin(dataset, &request.y, &x_refs, request.init_theta) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Negative binomial estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run zero-inflated regression for count data with excess zeros.
    #[tool(
        description = "Run zero-inflated Poisson or negative binomial regression for count data with excess zeros. Models both the zero-inflation probability and the count process. Equivalent to R's pscl::zeroinfl()."
    )]
    async fn zeroinfl(
        &self,
        Parameters(request): Parameters<ZeroInflRequest>,
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
        let z_refs: Option<Vec<&str>> = request
            .z
            .as_ref()
            .map(|zz| zz.iter().map(|s| s.as_str()).collect());
        let z_slice: Option<&[&str]> = z_refs.as_deref();
        let dist = request.dist.as_deref().unwrap_or("poisson");

        let result = match dist.to_lowercase().as_str() {
            "poisson" => run_zip(dataset, &request.y, &x_refs, z_slice),
            "negbin" | "negbinom" => run_zinb(dataset, &request.y, &x_refs, z_slice),
            _ => {
                return Ok(CallToolResult::error(vec![Content::text(
                    "Invalid dist. Use 'poisson' or 'negbin'.".to_string(),
                )]));
            }
        };

        match result {
            Ok(r) => Ok(CallToolResult::success(vec![Content::text(r.to_string())])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Zero-inflated {} estimation failed: {}",
                dist, e
            ))])),
        }
    }

    /// Run hurdle model for count data with excess zeros.
    #[tool(
        description = "Run a hurdle model (two-part model) for count data with excess zeros. The hurdle model separates the zero vs. positive decision (binary logit) from the count magnitude (truncated Poisson or negative binomial). Unlike zero-inflated models, hurdle assumes all zeros come from the binary part. Equivalent to R's pscl::hurdle()."
    )]
    async fn hurdle_model(
        &self,
        Parameters(request): Parameters<HurdleModelRequest>,
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
        let z_refs: Option<Vec<&str>> = request
            .z
            .as_ref()
            .map(|zz| zz.iter().map(|s| s.as_str()).collect());
        let z_slice: Option<&[&str]> = z_refs.as_deref();
        let dist = request.dist.as_deref().unwrap_or("poisson");

        let model_type = match dist.to_lowercase().as_str() {
            "poisson" => HurdleType::Poisson,
            "negbin" | "negbinom" => HurdleType::NegBin,
            _ => {
                return Ok(CallToolResult::error(vec![Content::text(
                    "Invalid dist. Use 'poisson' or 'negbin'.".to_string(),
                )]));
            }
        };

        match run_hurdle(dataset, &request.y, &x_refs, z_slice, model_type) {
            Ok(r) => Ok(CallToolResult::success(vec![Content::text(r.to_string())])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Hurdle {} estimation failed: {}",
                dist, e
            ))])),
        }
    }

    /// Run Generalized Linear Model with High-Dimensional Fixed Effects (FEGLM).
    #[tool(
        description = "Run Generalized Linear Model with high-dimensional fixed effects (FEGLM). Supports Logit, Probit, Poisson, and Gaussian families with multiple absorbed fixed effects. Uses IRLS + Method of Alternating Projections. Equivalent to R's alpaca::feglm()."
    )]
    async fn feglm(
        &self,
        Parameters(request): Parameters<FeglmRequest>,
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
        let fe_refs: Vec<&str> = request.fe.iter().map(|s| s.as_str()).collect();

        // Parse family
        let family = match request.family.as_deref() {
            Some("logit") | None => GlmFamily::Logit,
            Some("probit") => GlmFamily::Probit,
            Some("poisson") => GlmFamily::Poisson,
            Some("gaussian") => GlmFamily::Gaussian,
            Some(other) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown GLM family '{}'. Use 'logit', 'probit', 'poisson', or 'gaussian'.",
                    other
                ))]));
            }
        };

        // Build config from optional parameters
        let config = FeglmConfig {
            max_iter: request.max_iter.unwrap_or(25),
            tolerance: request.tolerance.unwrap_or(1e-8),
            ..Default::default()
        };

        let result = match run_feglm(dataset, &request.y, &x_refs, &fe_refs, family, Some(config)) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "FEGLM estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }
}
