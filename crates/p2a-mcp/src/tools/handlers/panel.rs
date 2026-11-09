//! Panel data tools handlers.
//!
//! This module provides MCP tool handlers for panel data analysis:
//! - Fixed Effects (FE) regression
//! - Random Effects (RE) regression
//! - Hausman specification test
//! - Variable Coefficients Model (PVCM)
//! - Mean Group (PMG) estimator
//! - Arellano-Bond difference GMM
//! - Panel GLS
//! - Panel unit root tests (LLC, IPS, Hadri)
//! - High-dimensional fixed effects (HDFE)

use rmcp::{
    ErrorData as McpError, handler::server::wrapper::Parameters, model::*, tool, tool_router,
};

use crate::server::AnalyticsServer;
use crate::tools::common::parse_option;
use crate::tools::handlers::regression::{append_unshown_notes, parse_choice};
use crate::tools::requests::panel::{
    GmmRequest, HausmanRequest, PanelFERequest, PanelGlsRequest, PanelHdfeRequest, PanelRERequest,
    PanelUnitRootRequest, PvcmRequest,
};

use p2a_core::econometrics::panel_unit_root::LagSelection;
use p2a_core::econometrics::{
    GmmConfig, GmmResult, GmmStep, GmmTransform, HdfeConfig, HdfeResult, PanelConfig,
    PanelGlsModel, PanelGlsResult, PanelModel, PanelUnitRootConfig, PanelUnitRootTest, PanelVcov,
    PvcmType, run_fixed_effects_with, run_gmm, run_hausman_test, run_hdfe, run_panel_gls,
    run_panel_unit_root, run_pmg, run_pvcm, run_random_effects_with,
};
use p2a_core::options::Choices;
use p2a_core::regression::CovarianceType;

const PVCM_MODEL: Choices<PvcmType> = Choices {
    option: "model",
    values: &[
        (&["within", "individual", "separate"], PvcmType::Within),
        (&["random", "swamy"], PvcmType::Random),
    ],
};

const PANEL_GLS_MODEL: Choices<PanelGlsModel> = Choices {
    option: "model",
    values: &[
        (
            &["fe", "within", "fixed_effects"],
            PanelGlsModel::FixedEffects,
        ),
        (&["pooling", "pool", "pooled"], PanelGlsModel::Pooling),
        (
            &["fd", "first_difference", "firstdifference"],
            PanelGlsModel::FirstDifference,
        ),
    ],
};

/// The text of a PVCM fit. For the within (mean-group) model the library's Display prints no
/// pooled coefficients, so the mean-group estimates (the average of the entity estimates, as
/// plm::pmg) are inserted before the homogeneity test.
fn pvcm_text(result: &p2a_core::econometrics::PvcmResult) -> String {
    let mut text = result.to_string();
    if result.model_type == PvcmType::Within {
        let mut table =
            String::from("Mean Group Coefficients (average of the entity estimates):\n");
        table.push_str(&format!(
            "{:<20} {:>12} {:>12} {:>10} {:>10}\n",
            "Variable", "Coef", "Std Err", "t", "P>|t|"
        ));
        table.push_str(&"-".repeat(70));
        table.push('\n');
        for (i, v) in result.variables.iter().enumerate() {
            let at = |x: &Vec<f64>| x.get(i).copied().unwrap_or(f64::NAN);
            table.push_str(&format!(
                "{:<20} {:>12.4} {:>12.4} {:>10.2} {:>10.3}{}\n",
                v,
                at(&result.coefficients),
                at(&result.std_errors),
                at(&result.t_stats),
                at(&result.p_values),
                result.significance.get(i).map(|s| s.stars()).unwrap_or("")
            ));
        }
        table.push('\n');
        match text.find("Homogeneity Test") {
            Some(at) => text.insert_str(at, &table),
            None => text.push_str(&table),
        }
    }
    append_unshown_notes(&mut text, result);
    text
}

/// The `vcov` option of `panel_fixed_effects` and `panel_random_effects`
/// (`PanelVcov::from_str`; absent means classical). A refusal names the caller's
/// value, trimmed as the parser trims it and cut to the echo bound; the message
/// is built here, so it does not depend on the core error's wording.
fn panel_config(vcov: Option<&str>) -> Result<PanelConfig, String> {
    let Some(s) = vcov else {
        return Ok(PanelConfig::default());
    };
    s.parse::<PanelVcov>()
        .map(|vcov| {
            let mut config = PanelConfig::default();
            config.vcov = vcov;
            config
        })
        .map_err(|_| {
            format!(
                "Invalid vcov '{}': expected one of: classical, cluster, cluster-hc0, \
                 cluster-hc1",
                crate::tools::limits::echo(s.trim())
            )
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

        let mut text = result.to_string();
        append_unshown_notes(&mut text, &result);
        Ok(CallToolResult::success(vec![Content::text(text)]))
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

        let mut text = result.to_string();
        append_unshown_notes(&mut text, &result);
        Ok(CallToolResult::success(vec![Content::text(text)]))
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
        description = "Run Variable Coefficients Model (PVCM) for heterogeneous panel data. Allows slope coefficients to vary across entities. Two modes: 'within' runs separate OLS per entity, 'random' (Swamy 1970) computes a GLS weighted average. Also computes homogeneity test for coefficient equality. model 'within' (default; also 'individual', 'separate') or 'random' (also 'swamy'); other values are an error. The within model also prints the mean-group coefficients (the average of the entity estimates)."
    )]
    pub async fn panel_pvcm(
        &self,
        Parameters(request): Parameters<PvcmRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let model_type = match parse_choice(&PVCM_MODEL, request.model.as_deref(), PvcmType::Within)
        {
            Ok(m) => m,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
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

        Ok(CallToolResult::success(vec![Content::text(pvcm_text(
            &result,
        ))]))
    }

    /// Run Mean Group (PMG) estimator for heterogeneous panels.
    #[tool(
        description = "Run Mean Group (MG) estimator (Pesaran & Smith 1995) for heterogeneous panel data. Computes simple average of individual-specific OLS estimates across entities. Equivalent to PVCM with 'within' model type. model: only 'within' (the default) is accepted; for Swamy's random coefficients use panel_pvcm with model = 'random'. Prints the mean-group coefficients (the average of the entity estimates) and the homogeneity test."
    )]
    pub async fn panel_pmg(
        &self,
        Parameters(request): Parameters<PvcmRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        // panel_pmg shares panel_pvcm's request; it runs only the within (mean-group) model
        match parse_choice(&PVCM_MODEL, request.model.as_deref(), PvcmType::Within) {
            Ok(PvcmType::Within) => {}
            Ok(PvcmType::Random) => {
                return Ok(CallToolResult::error(vec![Content::text(
                    "panel_pmg is the mean-group estimator, which is panel_pvcm with model 'within'; \
                     for Swamy's random-coefficients model use panel_pvcm with model = 'random'"
                        .to_string(),
                )]));
            }
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        }

        let result = match run_pmg(dataset, &request.y, &x_refs, &request.entity_var) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "PMG estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(pvcm_text(
            &result,
        ))]))
    }

    /// Run Arellano-Bond difference GMM dynamic panel estimation.
    #[tool(
        description = "Arellano-Bond difference GMM for dynamic panels, as R plm::pgmm(transformation = 'd'). Equations exist where y is observed at t, ..., t-lags-1 (time in steps of 1; gaps and late entry are handled; repeated (entity, time) rows are an error). GMM instruments are the observed lags y_{t-min_lag} ... y_{t-max_lag} (block-diagonal by period, or collapsed), and the exogenous x enter in differences. One-step (H-weighted) or two-step (default) estimates. Standard errors: one-step robust or Arellano-Bond (1991) classical; two-step Windmeijer (2005)-corrected (default) or uncorrected. Reports the Sargan/Hansen test (plm::sargan) and the robust Arellano-Bond AR(1)/AR(2) tests. Rows with missing values are dropped and reported. transform = 'system' runs Blundell-Bond system GMM as plm::pgmm(transformation = 'ld'): the difference equations stacked with level equations instrumented by y_{t-1} - y_{t-2} (and x in levels), with pgmm's 'full' one-step weight; one-step system fits need robust = true. The level equations have no intercept (plm::pgmm effect = 'individual'); with x in levels as its own instrument they are valid only when E[eta_i] = 0 or x has mean zero (Stata's xtabond2 adds _cons). transform 'difference' (default) or 'system'; step 'twostep' (default) or 'onestep'; unknown values are errors."
    )]
    pub async fn panel_gmm(
        &self,
        Parameters(request): Parameters<GmmRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();
        let lags = request.lags.unwrap_or(1);

        let transform = match parse_option::<GmmTransform>(
            request.transform.as_deref(),
            GmmTransform::Difference,
        ) {
            Ok(t) => t,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        let step = match parse_option::<GmmStep>(request.step.as_deref(), GmmStep::TwoStep) {
            Ok(t) => t,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        let mut config = GmmConfig::default();
        config.transform = transform;
        config.step = step;
        config.max_lag = request.max_lag;
        config.min_lag = request.min_lag.unwrap_or(2);
        config.collapse = request.collapse.unwrap_or(false);
        config.robust = request.robust.unwrap_or(true);

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

        let mut text = result.to_string();
        append_unshown_notes(&mut text, &result);
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    /// Run Panel GLS (Feasible Generalized Least Squares) estimation.
    #[tool(
        description = "Run Panel GLS (Feasible Generalized Least Squares) for panel data with heteroskedasticity and/or cross-sectional correlation. Supports fixed effects GLS, pooled GLS, and first-difference GLS. More efficient than standard FE/RE when error structure is known. model 'fe' (default; also 'within', 'fixed_effects'), 'pooling' (also 'pool', 'pooled') or 'fd' (also 'first_difference'); other values are an error."
    )]
    pub async fn panel_gls(
        &self,
        Parameters(request): Parameters<PanelGlsRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        // None keeps the library's default, fixed effects
        let model = match request.model.as_deref() {
            None => None,
            Some(m) => match parse_choice(&PANEL_GLS_MODEL, Some(m), PanelGlsModel::FixedEffects) {
                Ok(m) => Some(m),
                Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
            },
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

        let mut text = result.to_string();
        append_unshown_notes(&mut text, &result);
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    /// Run panel unit root tests.
    #[tool(
        description = "Run panel unit root tests to test for stationarity in panel data. Supports LLC (Levin-Lin-Chu), IPS (Im-Pesaran-Shin), Fisher/Maddala-Wu, and Hadri tests. Panel tests have more power than univariate tests by exploiting cross-sectional variation. LLC assumes common unit root, IPS allows heterogeneous roots, Hadri tests null of stationarity. Unit ids may be text or numbers; time must hold whole numbers or dates; repeated (unit, time) rows are an error; gaps are joined in time order as plm::purtest does (reported); rows with missing values are dropped and reported. test 'llc' (default; also 'levinlin'), 'ips', 'madwu' (also 'fisher') or 'hadri'; model 'none', 'intercept' (default) or 'trend'. If neither lags nor lag_selection is given, floor(T_max^(1/3)) lags for every unit; lag_selection = aic | sic | hall chooses per unit as plm::purtest (pmax default 10). Without lag_selection, p2a's default is a fixed floor(T_max^(1/3)) lags for every unit with dfcor = true; plm::purtest's default is SIC with pmax 10 and dfcor = FALSE."
    )]
    pub async fn panel_unit_root(
        &self,
        Parameters(request): Parameters<PanelUnitRootRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let test_type = match parse_option::<PanelUnitRootTest>(
            request.test.as_deref(),
            PanelUnitRootTest::LLC,
        ) {
            Ok(t) => t,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        let model =
            match parse_option::<PanelModel>(request.model.as_deref(), PanelModel::Intercept) {
                Ok(m) => m,
                Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
            };
        let lag_selection = match request
            .lag_selection
            .as_deref()
            .map(|s| LagSelection::from_option(s, request.lags))
            .transpose()
        {
            Ok(l) => l,
            Err(p2a_core::EconError::InvalidSpecification { message }) => {
                return Ok(CallToolResult::error(vec![Content::text(message)]));
            }
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        };

        let mut config = PanelUnitRootConfig::default();
        config.test_type = test_type;
        config.model = model;
        config.lags = request.lags;
        config.max_lags = None;
        config.lag_selection = lag_selection;
        config.pmax = request.pmax;

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

        let mut text = result.to_string();
        append_unshown_notes(&mut text, &result);
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    /// Run High-Dimensional Fixed Effects regression with multiple absorbed FE.
    #[tool(
        description = "Run High-Dimensional Fixed Effects (HDFE) regression with multiple absorbed fixed effects (e.g., firm + year + industry). Uses the Method of Alternating Projections (MAP) for efficient estimation. Equivalent to R's lfe::felm() or Stata's reghdfe. tolerance is relative: ||delta|| / ||x|| (fixest); se_type 'hc1' (default), 'standard', 'hc0', 'hc2' or 'hc3'."
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
        let mut config = HdfeConfig::default();
        config.tolerance = request.tolerance.unwrap_or(1e-8);
        config.max_iterations = request.max_iterations.unwrap_or(10000);
        config.accelerate = true;

        let cov_type =
            match parse_option::<CovarianceType>(request.se_type.as_deref(), CovarianceType::HC1) {
                Ok(c) => c,
                Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
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

    #[test]
    fn test_panel_option_tables_validate() {
        assert_eq!(super::PVCM_MODEL.validate(), Ok(()));
        assert_eq!(super::PANEL_GLS_MODEL.validate(), Ok(()));
    }

    /// The dropped-row note, which the library also lists among the warnings, prints once.
    #[test]
    fn test_mcp_panel_fixed_effects_missing_note_once() {
        use p2a_core::polars::prelude::*;
        let df = df! {
            "firm" => [1i64, 1, 1, 2, 2, 2, 3, 3, 3],
            "y" => [Some(1.1), Some(2.3), Some(2.9), Some(4.2), None, Some(6.1), Some(6.8), Some(8.3), Some(8.8)],
            "x" => [1.0, 2.1, 2.9, 4.0, 5.2, 6.1, 6.9, 8.0, 9.1],
        }
        .unwrap();
        let (rt, server) = server_with(vec![("grunfeld", df)]);
        let req: PanelFERequest = serde_json::from_value(
            json!({"dataset": "grunfeld", "y": "y", "x": ["x"], "entity_var": "firm"}),
        )
        .unwrap();
        let r = rt
            .block_on(server.panel_fixed_effects(Parameters(req)))
            .unwrap();
        let t = text(&r);
        assert_ne!(r.is_error, Some(true), "{t}");
        assert_eq!(t.matches("1 of 9 rows dropped").count(), 1, "{t}");
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
        let wald = format!(
            "Wald chi-squared (clustered): {:.4} on {} DF",
            case.scalar("wald_chisq_arellano_sss"),
            case.scalar("wald_df")
        );
        assert!(t.contains(&wald), "missing `{wald}` in\n{t}");
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
        // The F test follows the clustered covariance (plm's summary with a
        // vcov argument) and says so.
        let f = format!(
            "F-statistic (Wald, clustered): {:.4} on {} and {} DF (p-value: {:.4})",
            case.scalar("f_stat_arellano_hc0"),
            case.scalar("f_df1"),
            case.scalar("f_df2_arellano"),
            case.scalar("f_p_arellano_hc0")
        );
        assert!(t.contains(&f), "missing `{f}` in\n{t}");
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
            assert_eq!(
                t,
                "Invalid vcov 'robust': expected one of: classical, cluster, cluster-hc0, \
                 cluster-hc1"
            );
        }
        // A long value, also one padded with leading spaces, is echoed cut to
        // 64 characters (after the trim the parser applies).
        for long in [
            "x".repeat(300),
            format!("{}{}", " ".repeat(10), "x".repeat(300)),
        ] {
            let r = rt
                .block_on(server.panel_random_effects(Parameters(request(json!({"vcov": long})))))
                .unwrap();
            assert_eq!(r.is_error, Some(true));
            let t = text(&r);
            let shown = t
                .strip_prefix("Invalid vcov '")
                .and_then(|rest| rest.split_once("': expected one of:"))
                .map(|(v, _)| v)
                .unwrap_or_else(|| panic!("unexpected refusal: {t}"));
            assert_eq!(shown, "x".repeat(64), "{t}");
        }
    }

    /// `se_type`, the name `panel_hdfe` uses, reaches the strict parser: a
    /// valid value gives the clustered SEs, an invalid one is refused rather
    /// than ignored.
    #[test]
    fn test_mcp_panel_vcov_accepts_se_type_alias() {
        let (case, rt, server) = case_server("panel_fe_grunfeld");
        let r = rt
            .block_on(
                server.panel_fixed_effects(Parameters(request(json!({"se_type": "cluster"})))),
            )
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
        let (case, rt, server) = case_server("panel_re_grunfeld");
        let r = rt
            .block_on(
                server.panel_random_effects(Parameters(request(json!({"se_type": "cluster"})))),
            )
            .unwrap();
        assert_ne!(r.is_error, Some(true));
        let t = text(&r);
        for name in case.names("coef") {
            let se = format!("{:>12.4}", case.named("se_arellano_sss", name));
            assert!(t.contains(&se), "missing `{se}` in\n{t}");
        }
        let r = rt
            .block_on(server.panel_random_effects(Parameters(request(json!({"se_type": "hc1"})))))
            .unwrap();
        assert_eq!(r.is_error, Some(true));
        assert!(text(&r).starts_with("Invalid vcov 'hc1'"), "{}", text(&r));
    }
}
