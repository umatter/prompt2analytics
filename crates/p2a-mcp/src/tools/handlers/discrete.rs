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
use crate::tools::common::{MCP_DEFAULT_SEED, append_unshown_notes, parse_choice};
use crate::tools::requests::discrete::{
    FeglmRequest, HurdleModelRequest, LogitRequest, MixedLogitRequest, MlogitRequest,
    MultinomRequest, NegBinRequest, OrderedRequest, ProbitRequest, ZeroInflRequest,
};

use p2a_core::econometrics::discrete::conditional_logit::{MlogitConfig, run_mlogit_with};
use p2a_core::econometrics::{OrderedConfig, run_ordered_model_with};
use p2a_core::options::Choices;
use p2a_core::{
    FeglmConfig, GlmFamily, HurdleType, MixedLogitConfig, OrderedModelType, RandomDistribution,
    run_feglm, run_gmnl, run_hurdle, run_logit, run_multinom, run_negbin, run_probit, run_zinb,
    run_zip,
};

const MIXED_DISTRIBUTION: Choices<RandomDistribution> = Choices {
    option: "distribution",
    values: &[
        (&["normal", "n", "gaussian"], RandomDistribution::Normal),
        (
            &["lognormal", "log_normal", "ln"],
            RandomDistribution::LogNormal,
        ),
        (&["triangular", "t"], RandomDistribution::Triangular),
        (&["uniform", "u"], RandomDistribution::Uniform),
        (&["fixed", "none"], RandomDistribution::Fixed),
    ],
};

/// The link of ordered_model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OrderedLink {
    Logit,
    Probit,
}

const ORDERED_MODEL: Choices<OrderedLink> = Choices {
    option: "model_type",
    values: &[
        (&["logit", "logistic", "polr"], OrderedLink::Logit),
        (&["probit"], OrderedLink::Probit),
    ],
};

/// The count distribution of zeroinfl and hurdle_model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CountDist {
    Poisson,
    NegBin,
}

const COUNT_DIST: Choices<CountDist> = Choices {
    option: "dist",
    values: &[
        (&["poisson"], CountDist::Poisson),
        (
            &["negbin", "negbinom", "negative_binomial", "nb"],
            CountDist::NegBin,
        ),
    ],
};

const FEGLM_FAMILY: Choices<GlmFamily> = Choices {
    option: "family",
    values: &[
        (&["logit", "binomial", "logistic"], GlmFamily::Logit),
        (&["probit"], GlmFamily::Probit),
        (&["poisson"], GlmFamily::Poisson),
        (&["gaussian", "normal", "linear"], GlmFamily::Gaussian),
    ],
};

#[tool_router(router = discrete_router, vis = "pub")]
impl AnalyticsServer {
    /// Run Logit (logistic) regression for binary outcomes.
    #[tool(
        description = "Logistic regression for a 0/1 outcome, as R glm(family = binomial(link = 'logit')): IRLS as glm.fit (epsilon 1e-10, maxit 100: p2a's defaults; R's glm.control uses 1e-8 and 25). Aliased regressors get NaN coefficients with R's warning; fitted probabilities numerically 0 or 1 (separation) are warned. Returns coefficients, SEs, z, p, marginal effects, log-likelihood, AIC/BIC and convergence. The outcome must be coded 0/1 (integer and Boolean columns are accepted). Rows with a missing value in y or any x are dropped (R's na.omit) and the drop is reported."
    )]
    pub async fn logit(
        &self,
        Parameters(request): Parameters<LogitRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

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

        let mut text = result.to_string();
        append_unshown_notes(&mut text, &result);
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    /// Run Probit regression.
    #[tool(
        description = "Probit regression for a 0/1 outcome, as R glm(family = binomial(link = 'probit')): IRLS as glm.fit (epsilon 1e-10, maxit 100: p2a's defaults; R's glm.control uses 1e-8 and 25). Aliased regressors get NaN coefficients with R's warning; fitted probabilities numerically 0 or 1 (separation) are warned. Returns coefficients, SEs, z, p, marginal effects, log-likelihood, AIC/BIC and convergence. The outcome must be coded 0/1 (integer and Boolean columns are accepted). Rows with a missing value in y or any x are dropped (R's na.omit) and the drop is reported."
    )]
    pub async fn probit(
        &self,
        Parameters(request): Parameters<ProbitRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

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

        let mut text = result.to_string();
        append_unshown_notes(&mut text, &result);
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    /// Run multinomial logit regression for unordered categorical outcomes.
    #[tool(
        description = "Run multinomial logit regression for unordered categorical outcomes with 3+ categories (MLE, Newton-Raphson), as R's nnet::multinom(). Categories are ordered as R's factor() (numbers numerically, text by byte, as R under LC_COLLATE=C: uppercase before lowercase); the first is the reference unless 'reference' is given. Rows with a missing outcome or regressor are dropped and reported. Aliased (linearly dependent) regressors are an error."
    )]
    pub async fn multinom(
        &self,
        Parameters(request): Parameters<MultinomRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

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

        let mut text = result.to_string();
        append_unshown_notes(&mut text, &result);
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    /// Run McFadden's conditional logit (mlogit) for discrete choice analysis.
    #[tool(
        description = "McFadden's conditional logit as R mlogit::mlogit(choice ~ alt_specific | ind_specific): J - 1 alternative-specific constants by default (asc = false fits mlogit(... | 0 + ind_specific), the survival::clogit model); reflevel (or reference) names the base alternative, default the first in sorted order. Data in long format, one row per choice situation and alternative. Missing values in the id, alternative, choice or attribute columns are refused, as in R's mlogit."
    )]
    pub async fn mlogit(
        &self,
        Parameters(request): Parameters<MlogitRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let alt_specific_refs: Vec<&str> =
            request.alt_specific.iter().map(|s| s.as_str()).collect();
        let ind_specific_refs: Vec<&str> =
            request.ind_specific.iter().map(|s| s.as_str()).collect();
        let reflevel = match (&request.reflevel, &request.reference) {
            (Some(a), Some(b)) if a != b => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "reflevel ('{}') and reference ('{}') name different alternatives",
                    crate::tools::limits::echo(a),
                    crate::tools::limits::echo(b)
                ))]));
            }
            (a, b) => a.clone().or_else(|| b.clone()),
        };
        let mut config = MlogitConfig::default();
        config.asc = request.asc.unwrap_or(true);
        config.reflevel = reflevel;

        let result = match run_mlogit_with(
            dataset,
            &request.choice_id,
            &request.alt_id,
            &request.choice,
            &alt_specific_refs,
            &ind_specific_refs,
            &config,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Conditional logit (mlogit) estimation failed: {}",
                    e
                ))]));
            }
        };

        let mut text = result.to_string();
        append_unshown_notes(&mut text, &result);
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    /// Run mixed logit (random parameters logit) for discrete choice with preference heterogeneity.
    #[tool(
        description = "Run mixed logit (random parameters logit) for discrete choice with heterogeneous preferences: coefficients vary across decision makers as normal, lognormal, triangular or uniform random parameters. Maximum simulated likelihood with mlogit's draw scheme (Halton sequences, one segment per decision maker, the first 100 points dropped), BFGS with the analytic gradient from the conditional-logit estimates, standard errors from the Hessian. Set individual_id when the same person makes several choices (mlogit panel = TRUE). Missing values are refused. Equivalent to R's mlogit::mlogit(rpar = ..., halton = NA). distribution 'normal' (default), 'lognormal', 'triangular', 'uniform' or 'fixed'. Pseudo-random draws (halton = false) are seeded by the session seed, else 42."
    )]
    pub async fn mixed_logit(
        &self,
        Parameters(request): Parameters<MixedLogitRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let var_refs: Vec<&str> = request.variables.iter().map(|s| s.as_str()).collect();

        // Parse distribution
        let dist = match parse_choice(
            &MIXED_DISTRIBUTION,
            request.distribution.as_deref(),
            RandomDistribution::Normal,
        ) {
            Ok(d) => d,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        // Convert random_vars to refs
        let random_refs: Option<Vec<&str>> = request
            .random_vars
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());

        // Build config
        let mut config = MixedLogitConfig::default();
        config.n_draws = request.n_draws.unwrap_or(500);
        config.halton = request.halton.unwrap_or(true);
        config.individual_col = request.individual_id.clone();
        // pseudo-random draws (halton = false) follow the session seed, else the default
        config.seed = Some(self.global_seed.read().await.unwrap_or(MCP_DEFAULT_SEED));

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
        description = "Run ordered logit or probit regression (proportional odds model) for ordered categorical outcomes, as R's MASS::polr(). Numeric outcomes are ordered numerically; a text outcome needs 'levels' (its categories, lowest first). Rows with missing values are dropped and reported. Estimates threshold (cut-point) parameters. model_type 'logit' (default) or 'probit'."
    )]
    pub async fn ordered_model(
        &self,
        Parameters(request): Parameters<OrderedRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();
        let (link, model_type) = match parse_choice(
            &ORDERED_MODEL,
            request.model_type.as_deref(),
            OrderedLink::Logit,
        ) {
            Ok(OrderedLink::Logit) => (OrderedModelType::Logit, "logit"),
            Ok(OrderedLink::Probit) => (OrderedModelType::Probit, "probit"),
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        let config = OrderedConfig {
            link,
            levels: request.levels.clone(),
            ..OrderedConfig::default()
        };
        let result = run_ordered_model_with(dataset, &request.y, &x_refs, &config);

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
        description = "Run negative binomial regression for count data with overdispersion (variance > mean). Estimates dispersion parameter (theta). Equivalent to R's MASS::glm.nb(). The outcome must be non-negative counts."
    )]
    pub async fn negbin(
        &self,
        Parameters(request): Parameters<NegBinRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

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

        let mut text = result.to_string();
        append_unshown_notes(&mut text, &result);
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    /// Run zero-inflated regression for count data with excess zeros.
    #[tool(
        description = "Run zero-inflated Poisson or negative binomial regression for count data with excess zeros. Models both the zero-inflation probability and the count process. Equivalent to R's pscl::zeroinfl(). The outcome must be non-negative counts. dist 'poisson' (default) or 'negbin'."
    )]
    pub async fn zeroinfl(
        &self,
        Parameters(request): Parameters<ZeroInflRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();
        let z_refs: Option<Vec<&str>> = request
            .z
            .as_ref()
            .map(|zz| zz.iter().map(|s| s.as_str()).collect());
        let z_slice: Option<&[&str]> = z_refs.as_deref();
        let (result, dist) =
            match parse_choice(&COUNT_DIST, request.dist.as_deref(), CountDist::Poisson) {
                Ok(CountDist::Poisson) => {
                    (run_zip(dataset, &request.y, &x_refs, z_slice), "poisson")
                }
                Ok(CountDist::NegBin) => {
                    (run_zinb(dataset, &request.y, &x_refs, z_slice), "negbin")
                }
                Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
            };

        match result {
            Ok(r) => {
                let mut text = r.to_string();
                append_unshown_notes(&mut text, &r);
                Ok(CallToolResult::success(vec![Content::text(text)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Zero-inflated {} estimation failed: {}",
                dist, e
            ))])),
        }
    }

    /// Run hurdle model for count data with excess zeros.
    #[tool(
        description = "Run a hurdle model (two-part model) for count data with excess zeros. The hurdle model separates the zero vs. positive decision (binary logit) from the count magnitude (truncated Poisson or negative binomial). Unlike zero-inflated models, hurdle assumes all zeros come from the binary part. Equivalent to R's pscl::hurdle(). The outcome must be non-negative counts. dist 'poisson' (default) or 'negbin'."
    )]
    pub async fn hurdle_model(
        &self,
        Parameters(request): Parameters<HurdleModelRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();
        let z_refs: Option<Vec<&str>> = request
            .z
            .as_ref()
            .map(|zz| zz.iter().map(|s| s.as_str()).collect());
        let z_slice: Option<&[&str]> = z_refs.as_deref();
        let (model_type, dist) =
            match parse_choice(&COUNT_DIST, request.dist.as_deref(), CountDist::Poisson) {
                Ok(CountDist::Poisson) => (HurdleType::Poisson, "poisson"),
                Ok(CountDist::NegBin) => (HurdleType::NegBin, "negbin"),
                Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
            };

        match run_hurdle(dataset, &request.y, &x_refs, z_slice, model_type) {
            Ok(r) => {
                let mut text = r.to_string();
                append_unshown_notes(&mut text, &r);
                Ok(CallToolResult::success(vec![Content::text(text)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Hurdle {} estimation failed: {}",
                dist, e
            ))])),
        }
    }

    /// Run Generalized Linear Model with High-Dimensional Fixed Effects (FEGLM).
    #[tool(
        description = "Run Generalized Linear Model with high-dimensional fixed effects (FEGLM). Supports Logit, Probit, Poisson, and Gaussian families with multiple absorbed fixed effects. Uses IRLS + Method of Alternating Projections. Equivalent to R's alpaca::feglm(). family 'logit' (default), 'probit', 'poisson' or 'gaussian'; tolerance is the relative deviance change |dev - dev_old| / (|dev| + 0.1), as fixest."
    )]
    pub async fn feglm(
        &self,
        Parameters(request): Parameters<FeglmRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();
        let fe_refs: Vec<&str> = request.fe.iter().map(|s| s.as_str()).collect();

        // Parse family
        let family = match parse_choice(&FEGLM_FAMILY, request.family.as_deref(), GlmFamily::Logit)
        {
            Ok(f) => f,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
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

        let mut text = result.to_string();
        append_unshown_notes(&mut text, &result);
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_discrete_option_tables_validate() {
        assert_eq!(super::MIXED_DISTRIBUTION.validate(), Ok(()));
        assert_eq!(super::ORDERED_MODEL.validate(), Ok(()));
        assert_eq!(super::COUNT_DIST.validate(), Ok(()));
        assert_eq!(super::FEGLM_FAMILY.validate(), Ok(()));
    }

    use p2a_core::data::{DataLoader, Dataset};
    use p2a_core::econometrics::{OrderedConfig, OrderedModelType, run_ordered_model_with};
    use p2a_validation::{Case, Check, Tol};
    use rmcp::handler::server::wrapper::Parameters;
    use serde_json::{Value, json};

    use crate::server::AnalyticsServer;
    use crate::tools::common::test_support::{server_with, text};
    use crate::tools::requests::discrete::{LogitRequest, MixedLogitRequest, OrderedRequest};

    /// Case `id`, a runtime and a server holding the case's dataset under its file stem.
    fn server_with_case(id: &str) -> (Case, tokio::runtime::Runtime, AnalyticsServer) {
        let case = Case::load(id);
        let path = case.dataset_path();
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("dataset file stem")
            .to_string();
        let df = DataLoader::load_csv(&path).expect("case dataset");
        let (rt, server) = server_with(vec![(stem.as_str(), df)]);
        (case, rt, server)
    }

    fn case_dataset(case: &Case) -> Dataset {
        Dataset::new(DataLoader::load_csv(case.dataset_path()).expect("case dataset"))
    }

    /// The whitespace tokens of the first line whose first token is `name`.
    fn row<'a>(text: &'a str, name: &str) -> Vec<&'a str> {
        text.lines()
            .map(|l| l.split_whitespace().collect::<Vec<_>>())
            .find(|tokens| tokens.first() == Some(&name))
            .unwrap_or_else(|| panic!("no row {name:?} in:\n{text}"))
    }

    fn number(token: &str) -> f64 {
        token
            .parse()
            .unwrap_or_else(|_| panic!("{token:?} is not a number"))
    }

    /// A text outcome with `levels` is fitted in that order, as polr on factor(y): the
    /// tool's text is the core result's Display. Without `levels` the field was ignored and
    /// a text outcome was an error.
    #[test]
    fn test_validate_mcp_ordered_model_levels() {
        let (case, rt, server) = server_with_case("disc_ordered");
        let ds = case_dataset(&case);
        let levels: Vec<String> = ["low", "mid", "high"].map(String::from).to_vec();
        for (model_type, link, p) in [
            ("logit", OrderedModelType::Logit, "logit_"),
            ("probit", OrderedModelType::Probit, "probit_"),
        ] {
            let request: OrderedRequest = serde_json::from_value(json!({
                "dataset": "ordered", "y": "y_label", "x": ["x1", "x2"],
                "levels": ["low", "mid", "high"], "model_type": model_type
            }))
            .unwrap();
            let r = rt
                .block_on(server.ordered_model(Parameters(request)))
                .unwrap();
            let t = text(&r);
            assert_ne!(r.is_error, Some(true), "{t}");
            let config = OrderedConfig {
                link,
                levels: Some(levels.clone()),
                ..OrderedConfig::default()
            };
            let core = run_ordered_model_with(&ds, "y_label", &["x1", "x2"], &config).unwrap();
            assert_eq!(t, core.to_string());
            assert!(t.contains("low|mid:") && t.contains("mid|high:"), "{t}");
            assert!(t.contains("Converged: true ("), "{t}");
            let k = |s: &str| format!("{p}{s}");
            Check::new(&case)
                .named(
                    &k("coef"),
                    &core.variables,
                    &core.coefficients,
                    Tol::iterative(1e-6),
                )
                .vector(&k("zeta"), &core.thresholds, Tol::iterative(1e-6))
                .named(
                    &k("se_coef"),
                    &core.variables,
                    &core.std_errors,
                    Tol::iterative(1e-5),
                )
                .vector(
                    &k("se_zeta"),
                    &core.threshold_std_errors,
                    Tol::iterative(1e-5),
                )
                .scalar(&k("loglik"), core.log_likelihood, Tol::stat(400))
                .finish();

            // The numeric outcome needs no levels and gives the same fit.
            let request: OrderedRequest = serde_json::from_value(json!({
                "dataset": "ordered", "y": "y", "x": ["x1", "x2"], "model_type": model_type
            }))
            .unwrap();
            let r = rt
                .block_on(server.ordered_model(Parameters(request)))
                .unwrap();
            assert_ne!(r.is_error, Some(true), "{}", text(&r));
            let numeric = OrderedConfig {
                link,
                ..OrderedConfig::default()
            };
            let num = run_ordered_model_with(&ds, "y", &["x1", "x2"], &numeric).unwrap();
            assert_eq!(text(&r), num.to_string());
            assert_eq!(num.categories, ["1", "2", "3"]);
            let tight = Tol::new(1e-12, 1e-14);
            for (a, b) in num
                .coefficients
                .iter()
                .chain(&num.thresholds)
                .zip(core.coefficients.iter().chain(&core.thresholds))
            {
                assert!(tight.allows(*a, *b), "{a} vs {b}");
            }
        }
    }

    /// Without `levels`, a text outcome is a tool error that names the column, its labels
    /// and `levels`.
    #[test]
    fn test_mcp_ordered_model_text_outcome_needs_levels() {
        let (_, rt, server) = server_with_case("disc_ordered");
        let request: OrderedRequest = serde_json::from_value(json!({
            "dataset": "ordered", "y": "y_label", "x": ["x1", "x2"]
        }))
        .unwrap();
        let r = rt
            .block_on(server.ordered_model(Parameters(request)))
            .unwrap();
        let t = text(&r);
        assert_eq!(r.is_error, Some(true), "{t}");
        for part in ["'y_label'", "high", "low", "mid", "levels"] {
            assert!(t.contains(part), "{part:?} not in {t}");
        }
    }

    /// The printed means, SEs, SD and SD SE of the x (normal) and w (fixed) rows.
    fn check_printed_mixed_logit(case: &Case, t: &str) {
        let (x, w) = (row(t, "x"), row(t, "w"));
        // tokens: name, mean, SE, z, p, dist, SD, SD SE
        let names = ["mean_x", "mean_w", "sd_x"];
        let tol = Tol::printed(4);
        Check::new(case)
            .named(
                "coef",
                &names,
                &[number(x[1]), number(w[1]), number(x[6])],
                tol,
            )
            .named(
                "se",
                &names,
                &[number(x[2]), number(w[2]), number(x[7])],
                tol,
            )
            .finish();
        assert_eq!((w[6], w[7]), ("-", "-"), "{t}");
    }

    /// mlogit's estimate with Hessian SEs through the tool (the SEs were 0.1000).
    #[test]
    fn test_validate_mcp_mixed_logit() {
        let (case, rt, server) = server_with_case("disc_mixed_logit");
        let request: MixedLogitRequest = serde_json::from_value(json!({
            "dataset": "mixed_logit", "choice_id": "id", "alt_id": "alt", "choice": "choice",
            "variables": ["x", "w"], "random_vars": ["x"], "n_draws": 50
        }))
        .unwrap();
        let r = rt
            .block_on(server.mixed_logit(Parameters(request)))
            .unwrap();
        let t = text(&r);
        assert_ne!(r.is_error, Some(true), "{t}");
        check_printed_mixed_logit(&case, &t);
        assert!(t.contains("Converged:         Yes"), "{t}");
    }

    /// `individual_id` makes the person the decision maker (mlogit panel = TRUE).
    #[test]
    fn test_validate_mcp_mixed_logit_panel() {
        let (case, rt, server) = server_with_case("disc_mixed_logit_panel");
        let request: MixedLogitRequest = serde_json::from_value(json!({
            "dataset": "mixed_logit_panel", "choice_id": "id", "alt_id": "alt",
            "choice": "choice", "variables": ["x", "w"], "random_vars": ["x"],
            "n_draws": 50, "individual_id": "person"
        }))
        .unwrap();
        let r = rt
            .block_on(server.mixed_logit(Parameters(request)))
            .unwrap();
        let t = text(&r);
        assert_ne!(r.is_error, Some(true), "{t}");
        check_printed_mixed_logit(&case, &t);
        assert!(t.contains("Converged:         Yes"), "{t}");
        assert!(t.contains("Individuals:       100"), "{t}");
        assert!(t.contains("draws per individual"), "{t}");
    }

    /// The logit tool reports the rows it dropped.
    #[test]
    fn test_mcp_logit_reports_dropped_rows() {
        let (_, rt, server) = server_with_case("disc_choice_na");
        let request: LogitRequest = serde_json::from_value(json!({
            "dataset": "choice_na", "y": "choice_na", "x": ["x", "w"]
        }))
        .unwrap();
        let r = rt.block_on(server.logit(Parameters(request))).unwrap();
        let t = text(&r);
        assert_ne!(r.is_error, Some(true), "{t}");
        assert_eq!(
            t.lines().last(),
            Some("Missing values: 2 of 300 rows dropped because of missing values (choice_na: 2)"),
            "{t}"
        );
        assert!(t.contains("No. Observations: 298"), "{t}");
    }

    /// The schema documents `levels`, `individual_id` and the bounds of `init_theta`.
    #[test]
    fn test_mcp_discrete_schemas_document_new_fields() {
        let tools = AnalyticsServer::discrete_router().list_all();
        let tool = |name: &str| {
            tools
                .iter()
                .find(|t| t.name == name)
                .unwrap_or_else(|| panic!("{name} is advertised"))
        };
        let required = |name: &str| -> Vec<String> {
            tool(name)
                .input_schema
                .get("required")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(Value::as_str)
                        .map(String::from)
                        .collect()
                })
                .unwrap_or_default()
        };
        let ordered = tool("ordered_model");
        let levels = &ordered.input_schema["properties"]["levels"];
        let levels_text = levels.to_string();
        assert!(
            levels_text.contains("array") && levels_text.contains("string"),
            "{levels_text}"
        );
        assert!(!required("ordered_model").contains(&"levels".to_string()));
        assert!(
            ordered
                .description
                .as_deref()
                .unwrap_or("")
                .contains("levels"),
            "{:?}",
            ordered.description
        );
        let mixed = tool("mixed_logit");
        assert!(
            mixed.input_schema["properties"]
                .as_object()
                .unwrap()
                .contains_key("individual_id")
        );
        assert!(!required("mixed_logit").contains(&"individual_id".to_string()));
        let description = mixed.description.as_deref().unwrap_or("");
        assert!(
            description.contains("mlogit::mlogit") && description.contains("individual_id"),
            "{description}"
        );
        let init_theta = tool("negbin").input_schema["properties"]["init_theta"]["description"]
            .as_str()
            .unwrap_or("")
            .to_string();
        assert!(
            init_theta.contains("1e-8") && init_theta.contains("1e6"),
            "{init_theta}"
        );
    }
}
