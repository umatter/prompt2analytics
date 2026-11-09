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

use p2a_core::data::columns::MissingSummary;
use p2a_core::econometrics::{KmConfType, KmConfig, run_kaplan_meier_with};
use p2a_core::{
    AftConfig, AftDistribution, CoxConfig, TiesMethod, log_rank_test, run_aft, run_competing_risks,
    run_cox_ph,
};

use crate::server::AnalyticsServer;
use crate::tools::common::parse_option;
use crate::tools::handlers::regression::append_unshown_notes;
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
        let mut config = KmConfig::default();
        config.conf_level = request.confidence_level.unwrap_or(0.95);
        config.conf_type = conf_type;

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

        // Format all results; every group carries the whole call's dropped-row summary, so
        // a grouped fit prints the note once, after the last group
        let output: Vec<String> = if results.len() > 1 {
            results
                .iter()
                .map(|r| {
                    let mut shown = r.clone();
                    shown.missing = MissingSummary::none(r.missing.n_total);
                    shown.to_string()
                })
                .collect()
        } else {
            results.iter().map(|r| r.to_string()).collect()
        };
        let mut text = output.join("\n\n");
        if results.len() > 1
            && let Some(note) = results.first().and_then(|r| r.missing.note())
        {
            if !text.ends_with('\n') {
                text.push('\n');
            }
            text.push_str(&format!("Missing values: {note}"));
        }
        Ok(CallToolResult::success(vec![Content::text(text)]))
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

        if let Err(msg) = check_log_rank_groups(dataset, &request.group) {
            return Ok(CallToolResult::error(vec![Content::text(msg)]));
        }

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
        description = "Cox proportional hazards as R survival::coxph: Efron ties (default, as coxph) or Breslow; model-based standard errors, robust (Lin-Wei) ones with robust_se = true, or clustered ones with cluster = <column> (implies robust). Status coded as R's Surv(): 0/1 (1 = event), 1/2 (2 = event) or TRUE/FALSE. Returns hazard ratios with intervals, coefficients, SEs, the likelihood-ratio, Wald and score tests (and the robust score test), concordance and convergence. Rows with a missing value in any used column are dropped and reported."
    )]
    pub async fn cox_ph(
        &self,
        Parameters(request): Parameters<CoxPhRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let ties =
            match parse_option::<TiesMethod>(request.ties_method.as_deref(), TiesMethod::Efron) {
                Ok(t) => t,
                Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
            };
        if request.robust_se == Some(false) && request.cluster.is_some() {
            return Ok(CallToolResult::error(vec![Content::text(
                "cluster gives clustered robust standard errors; it cannot be combined with \
                 robust_se = false"
                    .to_string(),
            )]));
        }

        let cov_refs: Vec<&str> = request.covariates.iter().map(|s| s.as_str()).collect();

        let mut config = CoxConfig::default();
        config.ties = ties;
        config.robust_se = request.robust_se.unwrap_or(false) || request.cluster.is_some();
        config.cluster = request.cluster.clone();
        if let Some(m) = request.max_iter {
            config.max_iter = m;
        }
        if let Some(t) = request.tolerance {
            config.tolerance = t;
        }

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

        let mut text = result.to_string();
        append_unshown_notes(&mut text, &result);
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    /// Run Accelerated Failure Time (AFT) model.
    #[tool(
        description = "Accelerated failure time model as R survival::survreg: distribution 'weibull' (default), 'exponential' (scale fixed at 1), 'lognormal' or 'loglogistic'. Returns coefficients on log(T) with SEs, the scale and Log(scale), the log-likelihood on the time scale (as survreg), AIC/BIC and convergence. Status coded as R's Surv(): 0/1, 1/2 or TRUE/FALSE. Times must be positive, as R survreg requires. Rows with a missing value in any used column are dropped and reported."
    )]
    pub async fn aft(
        &self,
        Parameters(request): Parameters<AftRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let distribution = match parse_option::<AftDistribution>(
            request.distribution.as_deref(),
            AftDistribution::Weibull,
        ) {
            Ok(d) => d,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        let cov_refs: Vec<&str> = request.covariates.iter().map(|s| s.as_str()).collect();

        let mut config = AftConfig::default();
        config.distribution = distribution;
        if let Some(t) = request.tolerance {
            config.tolerance = t;
        }
        if let Some(m) = request.max_iter {
            config.max_iter = m;
        }

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

        let mut text = result.to_string();
        append_unshown_notes(&mut text, &result);
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    /// Run Competing Risks analysis (Aalen-Johansen estimator).
    #[tool(
        description = "Run Competing Risks analysis using the Aalen-Johansen estimator. Computes cumulative incidence functions (CIF) for multiple event types in the presence of competing risks. Properly accounts for subjects who experience a different event than the one of interest. Returns CIF curves with confidence intervals for each event type. Event codes: 0 = censored, 1, 2, ... = event types. Rows with a missing time or code are dropped and reported. The cumulative incidence point estimates are the Aalen-Johansen estimator (R survfit multi-state); standard errors as survfit (delta method); log-scale intervals."
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

/// Refuses a log-rank test whose group column has more groups than the `max_groups` budget of
/// the installed limits, before the test runs. The count is the column's distinct values minus
/// one for null, an upper bound on the groups the test compares (it also drops rows whose time
/// or status is missing). Nothing is counted when `max_groups` is off; a missing column is left
/// to the core to report, and a count that fails is a refusal.
fn check_log_rank_groups(dataset: &p2a_core::data::Dataset, group: &str) -> Result<(), String> {
    let Some(max_groups) = crate::tools::limits::current().max_groups else {
        return Ok(());
    };
    let refused = |msg: String| format!("Log-rank test refused: {msg}");
    let Some(k) = crate::tools::common::budget_group_count(dataset.df(), group).map_err(refused)?
    else {
        return Ok(());
    };
    crate::tools::common::check_named_budget("log_rank", "max_groups", k, Some(max_groups as u64))
        .map_err(|msg| {
            refused(format!(
                "group column `{}` has {k} groups. {msg} Use a column with fewer groups or \
                 filter the data first.",
                crate::tools::limits::echo(group)
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::common::test_support::{server_with, text};
    use p2a_validation::Case;
    use serde_json::{Value, json};

    /// The dropped-row note of a grouped fit is printed once, after the last group.
    #[test]
    fn test_mcp_kaplan_meier_grouped_missing_note_once() {
        let df = p2a_core::data::DataLoader::load_csv(format!(
            "{}/../../validation/datasets/generated/lung.csv",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let (rt, server) = server_with(vec![("lung", df)]);
        let r = rt
            .block_on(server.kaplan_meier(Parameters(km_request(json!({"group": "ph.ecog"})))))
            .unwrap();
        let t = text(&r);
        assert_ne!(r.is_error, Some(true), "{t}");
        assert_eq!(t.matches("Missing values:").count(), 1, "{t}");
        assert!(
            t.trim_end()
                .lines()
                .last()
                .unwrap()
                .starts_with("Missing values: 1 of 228 rows dropped"),
            "{t}"
        );
    }

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

    /// A survival frame of `rows` rows whose group column `g` has `groups` levels,
    /// with noisy times and both statuses in every group.
    fn grouped(rows: usize, groups: usize) -> p2a_core::polars::prelude::DataFrame {
        let time: Vec<f64> = (0..rows)
            .map(|i| 1.0 + ((i * 7919) % 997) as f64 / 37.0)
            .collect();
        let status: Vec<i64> = (0..rows)
            .map(|i| (!(i / groups).is_multiple_of(3)) as i64)
            .collect();
        let g: Vec<String> = (0..rows).map(|i| format!("g{}", i % groups)).collect();
        p2a_core::polars::prelude::df! { "time" => time, "status" => status, "g" => g }.unwrap()
    }

    #[test]
    fn log_rank_group_count_is_held_to_max_groups() {
        use crate::tools::common::test_support::hosted_with;
        use crate::tools::limits::{self, Limits};
        let (rt, server) = server_with(vec![("many", grouped(606, 101)), ("few", grouped(606, 6))]);
        let run = |limits: Limits, dataset: &str| {
            let req: LogRankRequest = serde_json::from_value(json!({
                "dataset": dataset, "time": "time", "event": "status", "group": "g"
            }))
            .unwrap();
            limits::with_limits(limits, || {
                rt.block_on(server.log_rank(Parameters(req))).unwrap()
            })
        };
        // 101 groups > 100 (hosted max_groups): refused before the test
        let r = run(Limits::hosted(), "many");
        let t = text(&r);
        assert_eq!(r.is_error, Some(true), "{t}");
        assert!(
            t.contains("max_groups") && t.contains("101") && t.contains("`g`"),
            "{t}"
        );
        // at the budget it runs; above a lowered one it is refused; off counts nothing
        let ok = run(hosted_with(&["max_groups=6"]), "few");
        assert_ne!(ok.is_error, Some(true), "{}", text(&ok));
        let over = run(hosted_with(&["max_groups=5"]), "few");
        assert_eq!(over.is_error, Some(true), "{}", text(&over));
        assert!(text(&over).contains("max_groups"), "{}", text(&over));
        let off = run(Limits::off(), "many");
        assert_ne!(off.is_error, Some(true), "{}", text(&off));
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
