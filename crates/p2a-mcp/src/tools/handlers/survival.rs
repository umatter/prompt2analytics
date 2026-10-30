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

use p2a_core::econometrics::{KmConfType, KmConfig, run_kaplan_meier_with};
use p2a_core::{
    AftConfig, AftDistribution, CoxConfig, TiesMethod, log_rank_test, run_aft, run_competing_risks,
    run_cox_ph,
};

use crate::server::AnalyticsServer;
use crate::tools::requests::survival::{
    AftRequest, CompetingRisksRequest, CoxPhRequest, KaplanMeierRequest, LogRankRequest,
};

#[tool_router(router = survival_router, vis = "pub")]
impl AnalyticsServer {
    /// Run Kaplan-Meier survival curve estimation.
    #[tool(
        description = "Kaplan-Meier survival curves, as R survival::survfit. Status coded as R's Surv(): 0/1 (1 = event), 1/2 (2 = event) or TRUE/FALSE. Greenwood standard errors; confidence intervals with conf_type 'log' (default, as R), 'log-log' or 'plain'. Median survival and its confidence limits as R's quantile.survfit (NA when a limit is not reached). Optional stratification by group (R factor order). Rows with a missing time, status or group are dropped and reported. The table lists every event time with both limits (up to 200 rows; the first 100 and last 50 beyond)."
    )]
    pub async fn kaplan_meier(
        &self,
        Parameters(request): Parameters<KaplanMeierRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let conf_type = match request
            .conf_type
            .as_deref()
            .map(str::parse::<KmConfType>)
            .transpose()
        {
            Ok(t) => t.unwrap_or_default(),
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Kaplan-Meier estimation failed: {e}"
                ))]));
            }
        };
        let config = KmConfig {
            conf_level: request.confidence_level.unwrap_or(0.95),
            conf_type,
        };

        let results = match run_kaplan_meier_with(
            dataset,
            &request.time,
            &request.event,
            request.group.as_deref(),
            &config,
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
        description = "Log-rank test comparing survival curves between groups, as R survival::survdiff (rho = 0). Status coded as R's Surv(): 0/1, 1/2 or TRUE/FALSE. Groups with no expected events are excluded from the test, as in R; df = groups with expected events - 1. Rows with a missing time, status or group are dropped and reported. Returns the chi-squared statistic, p-value, and observed and expected events per group."
    )]
    pub async fn log_rank(
        &self,
        Parameters(request): Parameters<LogRankRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

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
        description = "Run Cox Proportional Hazards (Cox PH) regression model. Semi-parametric regression estimating covariate effects on hazard rate without assuming a baseline hazard form. Uses Newton-Raphson optimization. Returns hazard ratios, coefficients, standard errors, confidence intervals, and concordance (C-index). Status coded as R's Surv(): 0/1, 1/2 or TRUE/FALSE. Rows with a missing value in any used column are dropped and reported."
    )]
    pub async fn cox_ph(
        &self,
        Parameters(request): Parameters<CoxPhRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

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
        description = "Run Accelerated Failure Time (AFT) parametric survival model. Models log(T) as a linear function of covariates. Supports Weibull, Exponential, Log-Normal, and Log-Logistic distributions. Returns regression coefficients (as acceleration factors), standard errors, shape parameters, and AIC/BIC. Status coded as R's Surv(): 0/1, 1/2 or TRUE/FALSE. Times must be positive, as R survreg requires. Rows with a missing value in any used column are dropped and reported."
    )]
    pub async fn aft(
        &self,
        Parameters(request): Parameters<AftRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

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
        description = "Run Competing Risks analysis using the Aalen-Johansen estimator. Computes cumulative incidence functions (CIF) for multiple event types in the presence of competing risks. Properly accounts for subjects who experience a different event than the one of interest. Returns CIF curves with confidence intervals for each event type. Event codes: 0 = censored, 1, 2, ... = event types. Rows with a missing time or code are dropped and reported."
    )]
    pub async fn competing_risks(
        &self,
        Parameters(request): Parameters<CompetingRisksRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::common::test_support::{server_with, text};
    use p2a_validation::Case;
    use serde_json::{Value, json};

    /// The case and a server holding its dataset as `name`.
    fn case_server(id: &str, name: &str) -> (Case, tokio::runtime::Runtime, AnalyticsServer) {
        let case = Case::load(id);
        let df = p2a_core::data::DataLoader::load_csv(case.dataset_path()).unwrap();
        let (rt, server) = server_with(vec![(name, df)]);
        (case, rt, server)
    }

    fn km_request(extra: Value) -> KaplanMeierRequest {
        let mut v = json!({"dataset": "lung", "time": "time", "event": "status"});
        for (k, x) in extra.as_object().unwrap() {
            v[k] = x.clone();
        }
        serde_json::from_value(v).unwrap()
    }

    /// "{:.4}" or "NA" (R's NA, NaN here).
    fn f4(v: f64) -> String {
        if v.is_nan() {
            "NA".to_string()
        } else {
            format!("{v:.4}")
        }
    }

    /// S(t), Std.Err and both limits of a Kaplan-Meier table row.
    fn row4(s: f64, se: f64, lo: f64, hi: f64) -> String {
        format!("{:>10.4} {:>12} {:>11} {:>11}", s, f4(se), f4(lo), f4(hi))
    }

    /// The case's summary row `i` of conf.type `tag`.
    fn case_row(case: &Case, tag: &str, i: usize) -> String {
        let at = |k: &str| case.vector(&format!("{tag}_{k}"))[i];
        row4(at("surv"), at("std_err"), at("lower"), at("upper"))
    }

    fn median_line(case: &Case, tag: &str) -> String {
        format!(
            "Median Survival: {:.3} (95% CI: {:.3} - {:.3})",
            case.scalar(&format!("{tag}_median")),
            case.scalar(&format!("{tag}_median_lower")),
            case.scalar(&format!("{tag}_median_upper"))
        )
    }

    #[test]
    fn test_validate_mcp_kaplan_meier_lung() {
        let (case, rt, server) = case_server("surv_km_lung", "lung");
        let r = rt
            .block_on(server.kaplan_meier(Parameters(km_request(json!({})))))
            .unwrap();
        assert_ne!(r.is_error, Some(true), "{}", text(&r));
        let t = text(&r);
        let n = case.scalar("n");
        let n_event = case.scalar("n_event");
        for expected in [
            "Confidence intervals: 95%, conf.type = log (R survfit default)".to_string(),
            median_line(&case, "log"),
            format!("N = {n}, Events = {n_event}, Censored = {}", n - n_event),
            case_row(&case, "log", 0), // t = 5
            case_row(&case, "log", 3), // t = 310
        ] {
            assert!(t.contains(&expected), "{expected:?} not in\n{t}");
        }
        // Every event time is listed (R summary(fit)); lung has 139.
        let rows = t
            .lines()
            .skip_while(|l| !l.starts_with("-----"))
            .skip(1)
            .filter(|l| !l.trim().is_empty())
            .count();
        assert_eq!(rows, 139, "{t}");

        for (conf_type, tag) in [("log-log", "loglog"), ("plain", "plain")] {
            let r = rt
                .block_on(
                    server.kaplan_meier(Parameters(km_request(json!({"conf_type": conf_type})))),
                )
                .unwrap();
            assert_ne!(r.is_error, Some(true), "{}", text(&r));
            let t = text(&r);
            for expected in [
                format!("Confidence intervals: 95%, conf.type = {conf_type}\n"),
                median_line(&case, tag),
                case_row(&case, tag, 0),
            ] {
                assert!(
                    t.contains(&expected),
                    "{conf_type}: {expected:?} not in\n{t}"
                );
            }
        }
    }

    #[test]
    fn test_validate_mcp_kaplan_meier_rejects_unknown_conf_type() {
        let (case, rt, server) = case_server("surv_km_lung", "lung");
        assert_eq!(case.scalar("n"), 228.0);
        let r = rt
            .block_on(server.kaplan_meier(Parameters(km_request(json!({"conf_type": "arcsin"})))))
            .unwrap();
        assert_eq!(r.is_error, Some(true));
        let t = text(&r);
        assert!(t.contains("arcsin") && t.contains("log-log"), "{t}");
    }

    #[test]
    fn test_validate_mcp_log_rank_lung() {
        let (case, rt, server) = case_server("surv_logrank_lung", "lung");
        let request = |group: &str| -> LogRankRequest {
            serde_json::from_value(json!({
                "dataset": "lung", "time": "time", "event": "status", "group": group
            }))
            .unwrap()
        };
        let r = rt
            .block_on(server.log_rank(Parameters(request("sex"))))
            .unwrap();
        assert_ne!(r.is_error, Some(true), "{}", text(&r));
        let t = text(&r);
        let (obs, exp, n) = (
            case.vector("sex_obs"),
            case.vector("sex_exp"),
            case.vector("sex_n"),
        );
        for expected in [
            format!(
                "Chi-squared = {:.4}, df = {}, p = {:.4}",
                case.scalar("sex_chisq"),
                case.scalar("sex_df"),
                case.scalar("sex_p")
            ),
            format!(
                "{:<15} {:>10} {:>10} {:>12.2} {:>12.2}",
                "1",
                n[0],
                obs[0],
                exp[0],
                obs[0] - exp[0]
            ),
        ] {
            assert!(t.contains(&expected), "{expected:?} not in\n{t}");
        }

        let r = rt
            .block_on(server.log_rank(Parameters(request("ph.ecog"))))
            .unwrap();
        let t = text(&r);
        let df = format!("df = {}", case.scalar("ecog_df"));
        assert!(t.contains(&df), "{df:?} not in\n{t}");
        assert!(
            t.contains(
                "Missing values: 1 of 228 rows dropped because of missing values (ph.ecog: 1)"
            ),
            "{t}"
        );
    }

    #[test]
    fn test_validate_mcp_survival_tools_accept_integer_columns() {
        let (case, rt, server) = case_server("surv_km_lung", "lung");
        let cox: CoxPhRequest = serde_json::from_value(json!({
            "dataset": "lung", "time": "time", "event": "status",
            "covariates": ["age", "sex", "ph.karno"]
        }))
        .unwrap();
        let r = rt.block_on(server.cox_ph(Parameters(cox))).unwrap();
        assert_ne!(r.is_error, Some(true), "{}", text(&r));
        let t = text(&r);
        // R coxph(Surv(time, status) ~ age + sex + ph.karno, lung): n = 227, nevent = 164.
        for expected in [
            "n = 227, events = 164",
            "Missing values: 1 of 228 rows dropped because of missing values (ph.karno: 1)",
        ] {
            assert!(t.contains(expected), "{expected:?} not in\n{t}");
        }

        let aft: AftRequest = serde_json::from_value(json!({
            "dataset": "lung", "time": "time", "event": "status", "covariates": ["age", "sex"]
        }))
        .unwrap();
        let r = rt.block_on(server.aft(Parameters(aft))).unwrap();
        assert_ne!(r.is_error, Some(true), "{}", text(&r));
        let expected = format!(
            "n = {}, events = {}",
            case.scalar("n"),
            case.scalar("n_event")
        );
        assert!(
            text(&r).contains(&expected),
            "{expected:?} not in\n{}",
            text(&r)
        );

        let (small, rt2, server2) = case_server("surv_km_small", "km_small");
        let cr: CompetingRisksRequest = serde_json::from_value(json!({
            "dataset": "km_small", "time": "time", "event": "event"
        }))
        .unwrap();
        let r = rt2
            .block_on(server2.competing_risks(Parameters(cr)))
            .unwrap();
        assert_ne!(r.is_error, Some(true), "{}", text(&r));
        let expected = format!("N = {}, Censored = ", small.vector("log_time").len());
        assert!(
            text(&r).contains(&expected),
            "{expected:?} not in\n{}",
            text(&r)
        );
    }

    #[test]
    fn test_mcp_kaplan_meier_schema_documents_conf_type() {
        let tools = AnalyticsServer::survival_router().list_all();
        let tool = |name: &str| {
            tools
                .iter()
                .find(|t| t.name == name)
                .unwrap_or_else(|| panic!("{name} is advertised"))
        };
        let km = tool("kaplan_meier");
        let description = km.description.as_deref().unwrap_or("");
        assert!(
            description.contains("log-log") && description.contains("Surv"),
            "{description}"
        );
        let properties = km.input_schema["properties"].as_object().unwrap();
        assert!(properties.contains_key("conf_type"), "{properties:?}");
        let required: Vec<&str> = km
            .input_schema
            .get("required")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        assert!(!required.contains(&"conf_type"), "{required:?}");
        for name in ["kaplan_meier", "log_rank", "cox_ph", "aft"] {
            let event = tool(name).input_schema["properties"]["event"]["description"]
                .as_str()
                .unwrap_or("")
                .to_string();
            assert!(event.contains("1/2"), "{name}: {event}");
        }
    }
}
