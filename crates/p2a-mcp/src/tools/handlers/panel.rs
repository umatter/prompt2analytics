//! Panel data tools handlers.
//!
//! This module provides MCP tool handlers for panel data analysis:
//! - Fixed Effects (FE) regression
//! - Random Effects (RE) regression
//! - Hausman specification test
//! - Variable Coefficients Model (PVCM)
//! - Mean Group (PMG) estimator
//! - Arellano-Bond / System GMM
//! - Panel GLS
//! - Panel unit root tests (LLC, IPS, Hadri)
//! - High-dimensional fixed effects (HDFE)

use rmcp::{
    ErrorData as McpError, handler::server::wrapper::Parameters, model::*, tool, tool_router,
};

use crate::server::AnalyticsServer;
use crate::tools::requests::panel::{
    GmmRequest, HausmanRequest, PanelFERequest, PanelGlsRequest, PanelHdfeRequest, PanelRERequest,
    PanelUnitRootRequest, PvcmRequest,
};

use p2a_core::econometrics::{
    GmmConfig, GmmResult, GmmStep, GmmTransform, HdfeConfig, HdfeResult, PanelConfig,
    PanelGlsModel, PanelGlsResult, PanelModel, PanelUnitRootConfig, PanelUnitRootTest, PanelVcov,
    PvcmType, run_fixed_effects_with, run_gmm, run_hausman_test, run_hdfe, run_panel_gls,
    run_panel_unit_root, run_pmg, run_pvcm, run_random_effects_with,
};
use p2a_core::regression::CovarianceType;

/// The `vcov` option of `panel_fixed_effects` and `panel_random_effects`
/// (`PanelVcov::from_str`; absent means classical). A refusal names the caller's
/// value cut to the echo bound.
fn panel_config(vcov: Option<&str>) -> Result<PanelConfig, String> {
    let Some(s) = vcov else {
        return Ok(PanelConfig::default());
    };
    s.parse::<PanelVcov>()
        .map(|vcov| PanelConfig { vcov })
        .map_err(|e| {
            let shown = crate::tools::limits::echo(s);
            format!("Invalid vcov: {}", e.to_string().replace(s, shown))
        })
}

#[tool_router(router = panel_router, vis = "pub")]
impl AnalyticsServer {
    /// Run Fixed Effects panel regression.
    #[tool(
        description = "Run Fixed Effects (within) panel regression. Controls for time-invariant unobserved heterogeneity. Requires panel data with entity identifiers. Controls ONE fixed effect dimension. For multiple simultaneous FEs (e.g., firm + year), use panel_hdfe instead. Standard errors are classical by default (as plm); set vcov = 'cluster' for entity-clustered (Arellano) standard errors."
    )]
    pub async fn panel_fixed_effects(
        &self,
        Parameters(request): Parameters<PanelFERequest>,
    ) -> Result<CallToolResult, McpError> {
        let config = match panel_config(request.vcov.as_deref()) {
            Ok(c) => c,
            Err(msg) => return crate::tools::common::error_text(msg),
        };

        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let result = match run_fixed_effects_with(
            dataset,
            &request.y,
            &x_refs,
            &request.entity_var,
            &config,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Fixed Effects estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run Random Effects panel regression.
    #[tool(
        description = "Run Random Effects (Swamy-Arora GLS) panel regression, as plm::plm(model = 'random'). Assumes individual effects are uncorrelated with regressors. More efficient than FE if assumption holds. Standard errors are classical by default (as plm), with z statistics and a Wald chi-squared test of the slopes; set vcov = 'cluster' for entity-clustered (Arellano) standard errors."
    )]
    pub async fn panel_random_effects(
        &self,
        Parameters(request): Parameters<PanelRERequest>,
    ) -> Result<CallToolResult, McpError> {
        let config = match panel_config(request.vcov.as_deref()) {
            Ok(c) => c,
            Err(msg) => return crate::tools::common::error_text(msg),
        };

        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let result = match run_random_effects_with(
            dataset,
            &request.y,
            &x_refs,
            &request.entity_var,
            &config,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Random Effects estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run Hausman specification test.
    #[tool(
        description = "Run Hausman specification test to choose between Fixed Effects and Random Effects, using the classical FE and RE covariances as plm::phtest. Tests H0: RE is consistent. If p-value < 0.05, use Fixed Effects."
    )]
    pub async fn hausman_test(
        &self,
        Parameters(request): Parameters<HausmanRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let result = match run_hausman_test(dataset, &request.y, &x_refs, &request.entity_var) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Hausman test failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run Variable Coefficients Model (PVCM) for heterogeneous panels.
    #[tool(
        description = "Run Variable Coefficients Model (PVCM) for heterogeneous panel data. Allows slope coefficients to vary across entities. Two modes: 'within' runs separate OLS per entity, 'random' (Swamy 1970) computes a GLS weighted average. Also computes homogeneity test for coefficient equality."
    )]
    pub async fn panel_pvcm(
        &self,
        Parameters(request): Parameters<PvcmRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let model_type = match request.model.as_deref() {
            Some("random") | Some("Random") => PvcmType::Random,
            _ => PvcmType::Within,
        };

        let result = match run_pvcm(
            dataset,
            &request.y,
            &x_refs,
            &request.entity_var,
            model_type,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "PVCM estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run Mean Group (PMG) estimator for heterogeneous panels.
    #[tool(
        description = "Run Mean Group (MG) estimator (Pesaran & Smith 1995) for heterogeneous panel data. Computes simple average of individual-specific OLS estimates across entities. Equivalent to PVCM with 'within' model type."
    )]
    pub async fn panel_pmg(
        &self,
        Parameters(request): Parameters<PvcmRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let result = match run_pmg(dataset, &request.y, &x_refs, &request.entity_var) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "PMG estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run Arellano-Bond / System GMM dynamic panel estimation.
    #[tool(
        description = "Run Arellano-Bond (difference GMM) or Blundell-Bond (system GMM) estimation for dynamic panel data models. Handles endogeneity of lagged dependent variables using lagged levels/differences as instruments. Reports Sargan test for overidentifying restrictions and AR(1)/AR(2) tests for serial correlation."
    )]
    pub async fn panel_gmm(
        &self,
        Parameters(request): Parameters<GmmRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();
        let lags = request.lags.unwrap_or(1);

        // Parse transform type
        let transform = match request.transform.as_deref() {
            Some("system") | Some("System") => GmmTransform::System,
            _ => GmmTransform::Difference, // Default
        };

        // Parse step type
        let step = match request.step.as_deref() {
            Some("onestep") | Some("OneStep") | Some("one") | Some("1") => GmmStep::OneStep,
            _ => GmmStep::TwoStep, // Default
        };

        let config = GmmConfig {
            transform,
            step,
            max_lag: request.max_lag,
            min_lag: request.min_lag.unwrap_or(2),
            collapse: request.collapse.unwrap_or(false),
            robust: request.robust.unwrap_or(true),
        };

        let result: GmmResult = match run_gmm(
            dataset,
            &request.y,
            &x_refs,
            &request.entity_var,
            &request.time_var,
            lags,
            Some(config),
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "GMM estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run Panel GLS (Feasible Generalized Least Squares) estimation.
    #[tool(
        description = "Run Panel GLS (Feasible Generalized Least Squares) for panel data with heteroskedasticity and/or cross-sectional correlation. Supports fixed effects GLS, pooled GLS, and first-difference GLS. More efficient than standard FE/RE when error structure is known."
    )]
    pub async fn panel_gls(
        &self,
        Parameters(request): Parameters<PanelGlsRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        // Parse model type
        let model = match request.model.as_deref() {
            Some("pooling") | Some("Pooling") | Some("pool") => Some(PanelGlsModel::Pooling),
            Some("fd") | Some("FD") | Some("first_difference") | Some("firstdifference") => {
                Some(PanelGlsModel::FirstDifference)
            }
            Some("fe") | Some("FE") | Some("fixed_effects") | Some("fixedeffects") => {
                Some(PanelGlsModel::FixedEffects)
            }
            _ => None, // Default to FixedEffects
        };

        let result: PanelGlsResult = match run_panel_gls(
            dataset,
            &request.y,
            &x_refs,
            &request.entity_var,
            &request.time_var,
            model,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Panel GLS estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run panel unit root tests.
    #[tool(
        description = "Run panel unit root tests to test for stationarity in panel data. Supports LLC (Levin-Lin-Chu), IPS (Im-Pesaran-Shin), Fisher/Maddala-Wu, and Hadri tests. Panel tests have more power than univariate tests by exploiting cross-sectional variation. LLC assumes common unit root, IPS allows heterogeneous roots, Hadri tests null of stationarity."
    )]
    pub async fn panel_unit_root(
        &self,
        Parameters(request): Parameters<PanelUnitRootRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        // Parse test type
        let test_type = match request.test.as_deref() {
            Some("ips") | Some("IPS") | Some("im_pesaran_shin") => PanelUnitRootTest::IPS,
            Some("fisher") | Some("Fisher") | Some("maddala_wu") => PanelUnitRootTest::Fisher,
            Some("hadri") | Some("Hadri") => PanelUnitRootTest::Hadri,
            _ => PanelUnitRootTest::LLC, // Default
        };

        // Parse model type
        let model = match request.model.as_deref() {
            Some("none") | Some("None") => PanelModel::None,
            Some("trend") | Some("Trend") => PanelModel::Trend,
            _ => PanelModel::Intercept, // Default
        };

        let config = PanelUnitRootConfig {
            test_type,
            model,
            lags: request.lags,
            max_lags: None,
        };

        let result = match run_panel_unit_root(
            dataset,
            &request.variable,
            &request.unit_col,
            &request.time_col,
            config,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Panel unit root test failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run High-Dimensional Fixed Effects regression with multiple absorbed FE.
    #[tool(
        description = "Run High-Dimensional Fixed Effects (HDFE) regression with multiple absorbed fixed effects (e.g., firm + year + industry). Uses the Method of Alternating Projections (MAP) for efficient estimation. Equivalent to R's lfe::felm() or Stata's reghdfe."
    )]
    pub async fn panel_hdfe(
        &self,
        Parameters(request): Parameters<PanelHdfeRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();
        let fe_refs: Vec<&str> = request.fe.iter().map(|s| s.as_str()).collect();

        // Build config from optional parameters
        let config = HdfeConfig {
            tolerance: request.tolerance.unwrap_or(1e-8),
            max_iterations: request.max_iterations.unwrap_or(10000),
            accelerate: true,
        };

        // Parse SE type
        let cov_type = match request.se_type.as_deref() {
            Some("standard") => CovarianceType::Standard,
            Some("hc0") => CovarianceType::HC0,
            Some("hc1") | None => CovarianceType::HC1,
            Some("hc2") => CovarianceType::HC2,
            Some("hc3") => CovarianceType::HC3,
            Some(other) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown SE type '{}'. Use 'standard', 'hc0', 'hc1', 'hc2', or 'hc3'.",
                    other
                ))]));
            }
        };

        let result: HdfeResult = match run_hdfe(
            dataset,
            &request.y,
            &x_refs,
            &fe_refs,
            Some(config),
            cov_type,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "HDFE estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::common::test_support::{server_with, text};
    use p2a_validation::Case;
    use serde_json::{Value, json};

    /// The case and a server holding its dataset as `grunfeld`.
    fn case_server(id: &str) -> (Case, tokio::runtime::Runtime, AnalyticsServer) {
        let case = Case::load(id);
        let df = p2a_core::data::DataLoader::load_csv(case.dataset_path()).unwrap();
        let (rt, server) = server_with(vec![("grunfeld", df)]);
        (case, rt, server)
    }

    /// `inv ~ value + capital` by `firm` on `grunfeld`, plus `extra` keys.
    fn request<T: serde::de::DeserializeOwned>(extra: Value) -> T {
        let mut v = json!({"dataset": "grunfeld", "y": "inv", "x": ["value", "capital"], "entity_var": "firm"});
        for (k, x) in extra.as_object().unwrap() {
            v[k] = x.clone();
        }
        serde_json::from_value(v).unwrap()
    }

    /// A coefficient row of the `PanelResult` Display table (without the stars).
    fn row(name: &str, coef: f64, se: f64, stat: f64, p: f64) -> String {
        format!("{name:<20} {coef:>12.4} {se:>12.4} {stat:>10.2} {p:>10.3}")
    }

    #[test]
    fn test_validate_mcp_panel_random_effects() {
        let (case, rt, server) = case_server("panel_re_grunfeld");
        let r = rt
            .block_on(server.panel_random_effects(Parameters(request(json!({})))))
            .unwrap();
        assert_ne!(r.is_error, Some(true));
        let t = text(&r);
        for name in case.names("coef") {
            let expected = row(
                name,
                case.named("coef", name),
                case.named("se", name),
                case.named("z", name),
                case.named("p", name),
            );
            assert!(t.contains(&expected), "missing `{expected}` in\n{t}");
        }
        let wald = format!(
            "Wald chi-squared: {:.4} on {} DF",
            case.scalar("wald_chisq"),
            case.scalar("wald_df")
        );
        assert!(t.contains(&wald), "missing `{wald}` in\n{t}");
        assert!(t.contains("Standard errors: classical"), "{t}");
        assert!(!t.contains("168.5471"), "{t}");

        let r = rt
            .block_on(server.panel_random_effects(Parameters(request(json!({"vcov": "cluster"})))))
            .unwrap();
        assert_ne!(r.is_error, Some(true));
        let t = text(&r);
        for name in case.names("coef") {
            let se = format!("{:>12.4}", case.named("se_arellano_sss", name));
            assert!(t.contains(&se), "missing `{se}` in\n{t}");
        }
        assert!(
            t.contains("Standard errors: clustered by entity (Arellano, CR1/sss)"),
            "{t}"
        );
    }

    #[test]
    fn test_validate_mcp_panel_fixed_effects() {
        let (case, rt, server) = case_server("panel_fe_grunfeld");
        let r = rt
            .block_on(server.panel_fixed_effects(Parameters(request(json!({})))))
            .unwrap();
        assert_ne!(r.is_error, Some(true));
        let t = text(&r);
        for name in case.names("coef") {
            let expected = row(
                name,
                case.named("coef", name),
                case.named("se", name),
                case.named("t", name),
                case.named("p", name),
            );
            assert!(t.contains(&expected), "missing `{expected}` in\n{t}");
        }
        let adj = format!("Adj. R-squared (within): {:.4}", case.scalar("adj_r2"));
        assert!(t.contains(&adj), "missing `{adj}` in\n{t}");
        assert!(t.contains("Standard errors: classical"), "{t}");

        let r = rt
            .block_on(
                server.panel_fixed_effects(Parameters(request(json!({"vcov": "cluster-hc0"})))),
            )
            .unwrap();
        assert_ne!(r.is_error, Some(true));
        let t = text(&r);
        for name in case.names("coef") {
            let se = format!("{:>12.4}", case.named("se_arellano_hc0", name));
            assert!(t.contains(&se), "missing `{se}` in\n{t}");
        }
        assert!(
            t.contains("Standard errors: clustered by entity (Arellano, HC0)"),
            "{t}"
        );
    }

    #[test]
    fn test_validate_mcp_hausman_test() {
        let (case, rt, server) = case_server("panel_hausman_grunfeld");
        let r = rt
            .block_on(server.hausman_test(Parameters(request(json!({})))))
            .unwrap();
        assert_ne!(r.is_error, Some(true));
        let t = text(&r);
        for line in [
            format!("Chi2 Statistic: {:.4}", case.scalar("chisq")),
            format!("Degrees of Freedom: {}", case.scalar("df")),
            format!("P-Value: {:.4}", case.scalar("p")),
            "Covariance: classical FE and RE covariance (plm::phtest default)".to_string(),
        ] {
            assert!(t.contains(&line), "missing `{line}` in\n{t}");
        }
        assert!(!t.contains("nearly identical"), "{t}");
    }

    /// An unknown vcov is a tool error naming the allowed values, raised before
    /// the dataset is looked up; a long value is echoed cut to the echo bound.
    #[test]
    fn test_mcp_panel_vcov_rejects_unknown() {
        let (rt, server) = server_with(vec![]);
        let fe = rt
            .block_on(server.panel_fixed_effects(Parameters(request(json!({"vcov": "robust"})))))
            .unwrap();
        let re = rt
            .block_on(server.panel_random_effects(Parameters(request(json!({"vcov": "robust"})))))
            .unwrap();
        for r in [&fe, &re] {
            assert_eq!(r.is_error, Some(true));
            let t = text(r);
            assert!(t.starts_with("Invalid vcov: "), "{t}");
            assert!(t.contains("robust"), "{t}");
            assert!(
                t.contains("classical, cluster, cluster-hc0, cluster-hc1"),
                "{t}"
            );
        }
        let long = "x".repeat(300);
        let r = rt
            .block_on(server.panel_random_effects(Parameters(request(json!({"vcov": long})))))
            .unwrap();
        assert_eq!(r.is_error, Some(true));
        let t = text(&r);
        assert!(
            t.contains(&"x".repeat(64)) && !t.contains(&"x".repeat(65)),
            "{t}"
        );
    }
}
