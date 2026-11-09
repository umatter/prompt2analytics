//! Regression tools handlers.
//!
//! This module provides MCP tool handlers for regression analysis:
//! - OLS regression
//! - Regression diagnostics
//! - Breusch-Godfrey test for serial correlation
//! - RESET test for functional form
//! - Wald test for nested models
//! - Harvey-Collier test for linearity
//! - HAC (Newey-West) standard errors
//! - Bootstrap covariance estimation
//! - Driscoll-Kraay panel-robust SEs
//! - Quantile regression
//! - Clustered standard errors
//! - Nonlinear least squares (NLS)
//! - LOESS local regression
//! - Super smoother (supsmu)
//! - Tukey's resistant line
//! - Stepwise regression
//! - GLS (Generalized Least Squares)
//! - Smoothing splines

use rmcp::{
    ErrorData as McpError, handler::server::wrapper::Parameters, model::*, tool, tool_router,
};

use crate::server::AnalyticsServer;
use crate::tools::common::{
    NaPolicy, missing_json, notes_json, numeric_column, numeric_columns, parse_option,
};
use crate::tools::requests::regression::{
    BgTestRequest, BootstrapCovRequest, CvGlmnetRequest, DiagnosticsRequest, DriscollKraayRequest,
    GlmnetRequest, GlsRequest, HacRequest, HarveyCollierRequest, LassoRequest, LineRequest,
    LoessRequest, NlsRequest, OlsClusteredRequest, OlsRequest, QuantRegRequest, ResetTestRequest,
    RidgeRequest, SmoothSplineRequest, StepRequest, SupsmuRequest, WaldTestRequest,
};

use p2a_core::data::Dataset;
use p2a_core::data::columns::ColumnRequest;
use p2a_core::regression::OlsResult;
use p2a_core::regression::{
    BgTestType, CorrelationStructure, CovarianceType, GlmnetConfig, GlmnetFamily, ResetType,
    SmoothSplineConfig, bg_test, gls, harvey_collier_test, quantreg_multi, reset_test,
    run_cv_glmnet, run_diagnostics, run_glmnet, run_lasso, run_line, run_loess, run_ols,
    run_ols_clustered, run_quantreg, run_ridge, run_step, run_vcov_bootstrap,
    run_vcov_driscoll_kraay, run_vcov_hac, smooth_spline, smooth_spline_predict, supsmu, wald_test,
};

use p2a_core::EconError;
use p2a_core::options::Choices;
use p2a_core::traits::convergence::ResultNotes;

/// Parses an option string through a `Choices` table (for core enums without `FromStr`):
/// `None` gives `default`; an unknown value is an error naming the option and the accepted
/// spellings.
pub(crate) fn parse_choice<T: Copy + PartialEq + 'static>(
    choices: &Choices<T>,
    value: Option<&str>,
    default: T,
) -> Result<T, String> {
    match value {
        None => Ok(default),
        Some(s) => choices.parse(s).map_err(|e| match e {
            EconError::InvalidSpecification { message } => message,
            other => other.to_string(),
        }),
    }
}

/// Appends the convergence line and the warnings of `result` that `text` does not show yet.
pub(crate) fn append_unshown_notes(text: &mut String, result: &dyn ResultNotes) {
    let converged = result
        .notes_convergence()
        .filter(|_| !text.contains("Converged:"))
        .map(|c| (c.converged, c.iterations));
    let warnings: Vec<String> = result
        .notes_warnings()
        .into_iter()
        .map(|w| {
            let w = w.trim();
            w.strip_prefix("Warning:").map_or(w, str::trim).to_string()
        })
        .filter(|w| !w.is_empty() && !text.contains(w.as_str()))
        .collect();
    crate::tools::common::append_notes(text, converged, &warnings);
}

/// Pretty JSON of a tool result; never panics.
pub(crate) fn json_text(value: &serde_json::Value) -> String {
    serde_json::to_string_pretty(value)
        .unwrap_or_else(|e| format!("{{\"error\": \"JSON serialisation failed: {e}\"}}"))
}

/// Compact JSON of a tool result (one line, for replies the chat loop may cut); never panics.
pub(crate) fn json_compact(value: &serde_json::Value) -> String {
    serde_json::to_string(value)
        .unwrap_or_else(|e| format!("{{\"error\": \"JSON serialisation failed: {e}\"}}"))
}

/// A JSON object written with its keys in the given order (serde_json's `Map` sorts keys), in
/// compact form. The glmnet tools put their summary before the long path this way, so the
/// part of a reply the chat loop keeps holds the warnings and the selection.
fn ordered_json(pairs: &[(&str, serde_json::Value)]) -> String {
    let body: Vec<String> = pairs
        .iter()
        .map(|(k, v)| {
            format!(
                "{}:{}",
                json_compact(&serde_json::Value::from(*k)),
                json_compact(v)
            )
        })
        .collect();
    format!("{{{}}}", body.join(","))
}

/// Predictors listed per glmnet fit before the listing switches to the largest |estimate|.
const GLMNET_LISTED_MAX: usize = 20;

/// One glmnet fit as JSON: lambda, df, deviance ratio, intercept and the coefficients (all of
/// them for up to 20 predictors, in column order; otherwise the 20 largest by |estimate| and
/// `more_nonzero`, the number of other non-zero ones).
fn glmnet_fit_json(
    names: &[String],
    fit: &p2a_core::regression::GlmnetCoefficients,
) -> serde_json::Value {
    let p = names.len().min(fit.coefficients.len());
    let mut entry = serde_json::json!({
        "lambda": fit.lambda, "df": fit.df, "dev_ratio": fit.dev_ratio, "intercept": fit.intercept,
    });
    let listed: Vec<usize> = if p <= GLMNET_LISTED_MAX {
        (0..p).collect()
    } else {
        let mut nz: Vec<usize> = (0..p).filter(|&j| fit.coefficients[j] != 0.0).collect();
        nz.sort_by(|&a, &b| {
            fit.coefficients[b]
                .abs()
                .total_cmp(&fit.coefficients[a].abs())
                .then(a.cmp(&b))
        });
        entry["more_nonzero"] = serde_json::json!(nz.len().saturating_sub(GLMNET_LISTED_MAX));
        nz.truncate(GLMNET_LISTED_MAX);
        nz
    };
    entry["coefficients"] = listed
        .iter()
        .map(|&j| serde_json::json!({"variable": names[j], "estimate": fit.coefficients[j]}))
        .collect();
    entry
}

/// The order in which variables enter a glmnet path: each variable that is non-zero at some
/// lambda (first non-zero fit, ties in column order), at most 20 listed, with its entry index
/// (1-based), the lambda there and its estimate at the last lambda; then the number of further
/// variables that enter and the number that never do.
fn glmnet_selection(r: &p2a_core::regression::GlmnetResult) -> (serde_json::Value, usize, usize) {
    let p = r.variable_names.len();
    let mut first: Vec<Option<usize>> = vec![None; p];
    for (i, fit) in r.fits.iter().enumerate() {
        for (j, c) in fit.coefficients.iter().take(p).enumerate() {
            if first[j].is_none() && *c != 0.0 {
                first[j] = Some(i);
            }
        }
    }
    let mut entered: Vec<(usize, usize)> = first
        .iter()
        .enumerate()
        .filter_map(|(j, f)| f.map(|i| (i, j)))
        .collect();
    entered.sort();
    let never = p - entered.len();
    let more = entered.len().saturating_sub(GLMNET_LISTED_MAX);
    let last = r.fits.last();
    let list: Vec<serde_json::Value> = entered
        .iter()
        .take(GLMNET_LISTED_MAX)
        .map(|&(i, j)| {
            serde_json::json!({
                "variable": r.variable_names[j],
                "enters_at_index": i + 1,
                "enters_at_lambda": r.fits[i].lambda,
                "estimate_at_last_lambda": last.and_then(|f| f.coefficients.get(j)).copied(),
            })
        })
        .collect();
    (serde_json::Value::from(list), more, never)
}

/// The JSON reply of regression_glmnet, regression_lasso and regression_ridge.
fn glmnet_json(r: &p2a_core::regression::GlmnetResult) -> String {
    use serde_json::json;
    let (selection, more, never) = glmnet_selection(r);
    let listed = if r.variable_names.len() <= GLMNET_LISTED_MAX {
        "all"
    } else {
        "the 20 largest |estimate| per lambda"
    };
    let path: Vec<serde_json::Value> = r
        .fits
        .iter()
        .map(|f| glmnet_fit_json(&r.variable_names, f))
        .collect();
    ordered_json(&[
        ("model", json!("glmnet")),
        ("family", json!(r.family.to_string())),
        ("alpha", json!(r.alpha)),
        ("standardize", json!(r.standardize)),
        ("intercept", json!(r.intercept)),
        ("dependent_var", json!(r.dependent_var)),
        ("variables", json!(r.variable_names)),
        ("n_obs", json!(r.n_obs)),
        ("n_lambda", json!(r.fits.len())),
        ("npasses", json!(r.npasses)),
        ("null_deviance", json!(r.null_deviance)),
        (
            "warnings",
            notes_json(None, &r.warnings)["warnings"].clone(),
        ),
        ("selection", selection),
        ("more_selected", json!(more)),
        ("never_selected", json!(never)),
        ("coefficients_listed", json!(listed)),
        ("path", json!(path)),
    ])
}

/// The JSON reply of regression_cv_glmnet; `seed` is the seed the folds were drawn from.
fn cv_glmnet_json(r: &p2a_core::regression::GlmnetCvResult, seed: u64) -> String {
    use serde_json::json;
    let names = &r.fit.variable_names;
    let at = |i: usize| {
        let mut e = r
            .fit
            .fits
            .get(i)
            .map(|f| glmnet_fit_json(names, f))
            .unwrap_or_else(|| json!({}));
        e["cvm"] = json!(r.cvm.get(i));
        e["cvsd"] = json!(r.cvsd.get(i));
        e
    };
    let curve: Vec<serde_json::Value> = (0..r.lambda.len())
        .map(|i| {
            json!({
                "lambda": r.lambda[i],
                "cvm": r.cvm.get(i),
                "cvsd": r.cvsd.get(i),
                "df": r.fit.fits.get(i).map(|f| f.df),
            })
        })
        .collect();
    let mut warnings = r.fit.warnings.clone();
    warnings.extend(
        r.warnings
            .iter()
            .filter(|w| !r.fit.warnings.contains(w))
            .cloned(),
    );
    ordered_json(&[
        ("model", json!("cv.glmnet")),
        ("family", json!(r.fit.family.to_string())),
        ("alpha", json!(r.fit.alpha)),
        ("standardize", json!(r.fit.standardize)),
        ("dependent_var", json!(r.fit.dependent_var)),
        ("variables", json!(names)),
        ("n_obs", json!(r.fit.n_obs)),
        ("nfolds", json!(r.nfolds)),
        ("seed", json!(seed)),
        ("lambda_min", json!(r.lambda_min)),
        ("index_min", json!(r.lambda_min_index + 1)),
        ("lambda_1se", json!(r.lambda_1se)),
        ("index_1se", json!(r.lambda_1se_index + 1)),
        ("warnings", notes_json(None, &warnings)["warnings"].clone()),
        ("at_lambda_min", at(r.lambda_min_index)),
        ("at_lambda_1se", at(r.lambda_1se_index)),
        ("cv_curve", json!(curve)),
        ("fold_path_lengths", json!(r.fold_path_lengths)),
    ])
}

/// The intercept flag and covariance type of an OLS request: `intercept`
/// defaults to true and `cov_type` to HC1 (parsed case-insensitively by
/// `CovarianceType::from_str`). An unknown `cov_type` is an error naming the
/// allowed values, with the caller's value cut to the echo bound.
fn ols_options(request: &OlsRequest) -> Result<(bool, CovarianceType), String> {
    let intercept = request.intercept.unwrap_or(true);
    let cov_type = match request.cov_type.as_deref() {
        None => CovarianceType::HC1,
        Some(raw) => raw.parse::<CovarianceType>().map_err(|_| {
            format!(
                "Invalid cov_type '{}': expected one of: standard (or classical), hc0, hc1, \
                 hc2, hc3",
                crate::tools::limits::echo(raw.trim())
            )
        })?,
    };
    Ok((intercept, cov_type))
}

/// Runs the OLS a request describes on `dataset`.
fn run_ols_request(dataset: &Dataset, request: &OlsRequest) -> Result<OlsResult, String> {
    let (intercept, cov_type) = ols_options(request)?;
    let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();
    run_ols(dataset, &request.y, &x_refs, intercept, cov_type)
        .map_err(|e| format!("Regression failed: {e}"))
}

#[tool_router(router = regression_router, vis = "pub")]
impl AnalyticsServer {
    /// Run OLS regression.
    #[tool(
        description = "Run Ordinary Least Squares (OLS) regression. Returns coefficients, standard errors, t-values, p-values, R-squared, and F-statistic. Standard errors default to HC1 (heteroskedasticity-robust); set cov_type = 'standard' for classical, or 'hc0', 'hc2', 'hc3'. Set intercept = false to omit the intercept. For comparing group means with a single categorical predictor, consider anova_one_way instead."
    )]
    pub async fn regression_ols(
        &self,
        Parameters(request): Parameters<OlsRequest>,
    ) -> Result<CallToolResult, McpError> {
        // An unknown cov_type is refused before the dataset is looked up.
        if let Err(msg) = ols_options(&request) {
            return Ok(CallToolResult::error(vec![Content::text(msg)]));
        }

        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        match run_ols_request(dataset, &request) {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(
                result.to_string(),
            )])),
            Err(msg) => Ok(CallToolResult::error(vec![Content::text(msg)])),
        }
    }

    /// Run regression diagnostics.
    #[tool(
        description = "Run comprehensive regression diagnostics. Tests include: Jarque-Bera (normality), Breusch-Pagan (heteroskedasticity), Durbin-Watson (autocorrelation), VIF (multicollinearity), and condition number."
    )]
    pub async fn regression_diagnostics(
        &self,
        Parameters(request): Parameters<DiagnosticsRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let result = match run_diagnostics(dataset, &request.y, &x_refs) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Diagnostics failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run Breusch-Godfrey test for serial correlation.
    #[tool(
        description = "Breusch-Godfrey test for higher-order serial correlation in regression residuals. More general than Durbin-Watson as it: (1) tests for AR(p) or MA(p) correlation, (2) allows lagged dependent variables as regressors, (3) is valid regardless of regressor properties. Returns LM statistic and p-value."
    )]
    pub async fn regression_bgtest(
        &self,
        Parameters(request): Parameters<BgTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();
        let order = request.order.unwrap_or(1);
        let fill = request.fill.unwrap_or(0.0);

        let test_type = match request.test_type.as_deref() {
            Some(t) => BgTestType::from_str(t).unwrap_or(BgTestType::Chisq),
            None => BgTestType::Chisq,
        };

        let result = match bg_test(dataset, &request.y, &x_refs, order, test_type, fill) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Breusch-Godfrey test failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run Ramsey's RESET test for functional form.
    #[tool(
        description = "Ramsey's RESET test for functional form misspecification. Tests whether nonlinear terms (powers of fitted values) should be added to the model. Significant result suggests the linear model is misspecified and nonlinear terms may be needed."
    )]
    pub async fn regression_resettest(
        &self,
        Parameters(request): Parameters<ResetTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();
        let powers = request.powers.unwrap_or_else(|| vec![2, 3]);

        let reset_type_enum = match request.reset_type.as_deref() {
            Some(t) => ResetType::from_str(t).unwrap_or(ResetType::Fitted),
            None => ResetType::Fitted,
        };

        let result = match reset_test(dataset, &request.y, &x_refs, &powers, reset_type_enum) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "RESET test failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Wald test for comparing nested linear models.
    #[tool(
        description = "Wald test (F-test) for comparing nested linear models. Tests whether variables excluded from the restricted model are jointly significant. Common uses: testing joint significance of multiple coefficients, testing nested model hypotheses, comparing model specifications."
    )]
    pub async fn regression_waldtest(
        &self,
        Parameters(request): Parameters<WaldTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_unrestricted_refs: Vec<&str> =
            request.x_unrestricted.iter().map(|s| s.as_str()).collect();
        let x_restricted_refs: Vec<&str> =
            request.x_restricted.iter().map(|s| s.as_str()).collect();
        let use_f_test = request.use_f_test.unwrap_or(true);

        let result = match wald_test(
            dataset,
            &request.y,
            &x_unrestricted_refs,
            &x_restricted_refs,
            use_f_test,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Wald test failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Harvey-Collier test for linearity using recursive residuals.
    #[tool(
        description = "Harvey-Collier test for detecting departure from linearity. Uses recursive residuals to test whether the mean of one-step-ahead forecast errors differs from zero. A significant result suggests convex or concave functional misspecification - the linear model may need polynomial terms. Equivalent to R's lmtest::harvtest()."
    )]
    pub async fn regression_harvtest(
        &self,
        Parameters(request): Parameters<HarveyCollierRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let result = match harvey_collier_test(dataset, &request.y, &x_refs) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Harvey-Collier test failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// HAC (Newey-West) standard errors for time series regression.
    #[tool(
        description = "Compute HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors using the Newey-West estimator. Essential for time series regression where errors may be both heteroskedastic and autocorrelated. Supports multiple kernel functions and automatic bandwidth selection."
    )]
    pub async fn regression_hac(
        &self,
        Parameters(request): Parameters<HacRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();
        let prewhiten = request.prewhiten.unwrap_or(false);

        let result = match run_vcov_hac(
            dataset,
            &request.y,
            &x_refs,
            request.bandwidth,
            request.kernel.as_deref(),
            prewhiten,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "HAC estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Bootstrap covariance estimation (vcovBS).
    #[tool(
        description = "Compute bootstrap covariance matrix and standard errors for OLS regression. Supports pairs bootstrap (most robust, resamples observations), residual bootstrap (assumes homoskedasticity), and wild bootstrap (robust to heteroskedasticity). Useful when asymptotic standard errors may be unreliable."
    )]
    pub async fn regression_bootstrap_cov(
        &self,
        Parameters(request): Parameters<BootstrapCovRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let result = match run_vcov_bootstrap(
            dataset,
            &request.y,
            &x_refs,
            request.n_boot,
            request.bootstrap_type.as_deref(),
            request.seed,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Bootstrap covariance estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Driscoll-Kraay panel-robust standard errors.
    #[tool(
        description = "Compute Driscoll-Kraay (1998) panel-robust standard errors. Robust to arbitrary cross-sectional correlation (spatial dependence) and serial correlation in panel data. Aggregates score vectors by time period and applies Newey-West HAC correction. Best for panels with large T (many time periods). Returns coefficient estimates with panel-robust SEs, t-stats, and p-values. time_col must hold whole numbers or dates; rows with a missing time id are dropped and the model is re-estimated (reported in the output)."
    )]
    pub async fn regression_driscoll_kraay(
        &self,
        Parameters(request): Parameters<DriscollKraayRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let result = match run_vcov_driscoll_kraay(
            dataset,
            &request.y,
            &x_refs,
            &request.time_col,
            request.bandwidth,
            request.kernel.as_deref(),
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Driscoll-Kraay estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Quantile regression.
    #[tool(
        description = "Run quantile regression to estimate conditional quantiles instead of conditional means. Useful when the relationship varies across the distribution, or when the error distribution is non-Gaussian. Can estimate single quantile (tau) or multiple quantiles simultaneously."
    )]
    pub async fn regression_quantreg(
        &self,
        Parameters(request): Parameters<QuantRegRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        // Handle multiple quantiles or single quantile
        if let Some(taus) = &request.taus {
            // Multiple quantiles
            let results = match quantreg_multi(dataset, &request.y, &x_refs, taus) {
                Ok(r) => r,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Quantile regression failed: {}",
                        e
                    ))]));
                }
            };

            // Format all results
            let output: String = results
                .iter()
                .map(|r| format!("{}", r))
                .collect::<Vec<_>>()
                .join("\n---\n");

            Ok(CallToolResult::success(vec![Content::text(output)]))
        } else {
            // Single quantile
            let tau = request.tau.unwrap_or(0.5);
            let result = match run_quantreg(dataset, &request.y, &x_refs, tau) {
                Ok(r) => r,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Quantile regression failed: {}",
                        e
                    ))]));
                }
            };

            Ok(CallToolResult::success(vec![Content::text(
                result.to_string(),
            )]))
        }
    }

    /// Run OLS with clustered standard errors.
    #[tool(
        description = "Run OLS regression with clustered standard errors. Supports one-way (firm, state) or two-way (firm + time) clustering. Essential for panel data with correlated errors. Rows with a missing cluster id are dropped and the model is re-estimated on the remaining rows (as fixest); the output reports how many."
    )]
    pub async fn regression_clustered(
        &self,
        Parameters(request): Parameters<OlsClusteredRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let result = match run_ols_clustered(
            dataset,
            &request.y,
            &x_refs,
            &request.cluster1,
            request.cluster2.as_deref(),
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Clustered regression failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run nonlinear least squares regression.
    #[tool(
        description = "Fit a nonlinear regression model using Levenberg-Marquardt algorithm. Supports common models: exponential decay/growth, Michaelis-Menten kinetics, logistic growth, power law, asymptotic. Returns parameter estimates, standard errors, t-values, and convergence info."
    )]
    pub async fn regression_nls(
        &self,
        Parameters(request): Parameters<NlsRequest>,
    ) -> Result<CallToolResult, McpError> {
        use ndarray::Array1;
        use p2a_core::linalg::design::DesignMatrix;
        use p2a_core::regression::{
            NlsAlgorithm, NlsConfig, model_asymptotic, model_exponential_decay,
            model_exponential_growth, model_logistic_growth, model_michaelis_menten, model_power,
            nls,
        };

        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        // Extract x and y columns
        let x = match DesignMatrix::extract_column(dataset.df(), &request.x) {
            Ok(col) => col,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to extract X column '{}': {:?}",
                    request.x, e
                ))]));
            }
        };

        let y = match DesignMatrix::extract_column(dataset.df(), &request.y) {
            Ok(col) => col,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to extract Y column '{}': {:?}",
                    request.y, e
                ))]));
            }
        };

        // Parse model type and get function + param names
        let (model_fn, param_names): (fn(&Array1<f64>, &Array1<f64>) -> f64, Vec<&str>) =
            match request.model.to_lowercase().as_str() {
                "exponential_decay" | "exp_decay" => {
                    if request.start.len() != 3 {
                        return Ok(CallToolResult::error(vec![Content::text(
                            "exponential_decay requires 3 starting values: [a, b, c] for y = a*exp(-b*x) + c",
                        )]));
                    }
                    (model_exponential_decay, vec!["a", "b", "c"])
                }
                "exponential_growth" | "exp_growth" => {
                    if request.start.len() != 2 {
                        return Ok(CallToolResult::error(vec![Content::text(
                            "exponential_growth requires 2 starting values: [a, b] for y = a*exp(b*x)",
                        )]));
                    }
                    (model_exponential_growth, vec!["a", "b"])
                }
                "michaelis_menten" | "mm" => {
                    if request.start.len() != 2 {
                        return Ok(CallToolResult::error(vec![Content::text(
                            "michaelis_menten requires 2 starting values: [Vmax, Km] for y = Vmax*x/(Km+x)",
                        )]));
                    }
                    (model_michaelis_menten, vec!["Vmax", "Km"])
                }
                "logistic" | "logistic_growth" => {
                    if request.start.len() != 3 {
                        return Ok(CallToolResult::error(vec![Content::text(
                            "logistic requires 3 starting values: [K, r, x0] for y = K/(1+exp(-r*(x-x0)))",
                        )]));
                    }
                    (model_logistic_growth, vec!["K", "r", "x0"])
                }
                "power" => {
                    if request.start.len() != 2 {
                        return Ok(CallToolResult::error(vec![Content::text(
                            "power requires 2 starting values: [a, b] for y = a*x^b",
                        )]));
                    }
                    (model_power, vec!["a", "b"])
                }
                "asymptotic" => {
                    if request.start.len() != 3 {
                        return Ok(CallToolResult::error(vec![Content::text(
                            "asymptotic requires 3 starting values: [a, b, c] for y = a - b*exp(-c*x)",
                        )]));
                    }
                    (model_asymptotic, vec!["a", "b", "c"])
                }
                _ => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Unknown model '{}'. Supported: exponential_decay, exponential_growth, michaelis_menten, logistic, power, asymptotic",
                        request.model
                    ))]));
                }
            };

        // Parse algorithm
        let algorithm = match request.algorithm.as_deref() {
            Some("gauss_newton") | Some("gn") => NlsAlgorithm::GaussNewton,
            Some("levenberg_marquardt") | Some("lm") | None => NlsAlgorithm::LevenbergMarquardt,
            Some(other) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown algorithm '{}'. Supported: levenberg_marquardt (default), gauss_newton",
                    other
                ))]));
            }
        };

        let config = NlsConfig {
            algorithm,
            ..Default::default()
        };

        let start = Array1::from_vec(request.start);

        let result = match nls(&x, &y, model_fn, &start, &param_names, config) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "NLS fitting failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run LOESS (local polynomial regression) smoothing.
    #[tool(
        description = "Fit a LOESS (LOcally Estimated Scatterplot Smoothing) model. LOESS fits local polynomial regressions at each point, weighted by distance from the target point using tricubic weights. Useful for non-parametric trend estimation, data smoothing, and exploring nonlinear relationships. Returns fitted values, residuals, R-squared, and equivalent number of parameters."
    )]
    pub async fn regression_loess(
        &self,
        Parameters(request): Parameters<LoessRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let span = request.span.unwrap_or(0.75);
        let degree = request.degree.unwrap_or(2);
        let robust = request.robust.unwrap_or(false);

        // Validate parameters
        if span <= 0.0 {
            return Ok(CallToolResult::error(vec![Content::text(
                "span must be positive".to_string(),
            )]));
        }
        if degree > 2 {
            return Ok(CallToolResult::error(vec![Content::text(
                "degree must be 0, 1, or 2".to_string(),
            )]));
        }

        let result = match run_loess(dataset, &request.y, &request.x, span, degree, robust) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "LOESS fitting failed: {}",
                    e
                ))]));
            }
        };

        // Build JSON output for LLM
        let mut output = serde_json::json!({
            "method": "LOESS",
            "span": result.span,
            "degree": result.degree,
            "family": if result.robust { "symmetric" } else { "gaussian" },
            "n_obs": result.n_obs,
            "equivalent_number_parameters": result.enp,
            "residual_sum_squares": result.rss,
            "residual_standard_error": result.residual_se,
            "r_squared": result.r_squared,
            "robust_iterations": result.robust_iterations,
            "fitted_values": result.fitted,
            "residuals": result.residuals,
        });
        if let Some(m) = missing_json(&result.missing) {
            output["missing"] = m;
        }

        Ok(CallToolResult::success(vec![Content::text(json_text(
            &output,
        ))]))
    }

    /// Fit Friedman's SuperSmoother.
    #[tool(
        description = "Friedman's SuperSmoother - a variable-bandwidth smoother that locally adapts to the data. Uses cross-validation to select optimal span at each point. Returns smoothed y values and can handle periodic data."
    )]
    pub async fn regression_supsmu(
        &self,
        Parameters(request): Parameters<SupsmuRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        // x, y and the weights with R's deletion of incomplete observations; too few
        // complete rows is reported against the dataset's row count
        let mut names = vec![request.x.as_str(), request.y.as_str()];
        if let Some(w) = request.weights.as_deref() {
            names.push(w);
        }
        let cols = match ColumnRequest::new(dataset.df())
            .numerics(&names)
            .na(NaPolicy::Omit)
            .min_rows(
                4,
                &format!("SuperSmoother of '{}' on '{}'", request.y, request.x),
            )
            .extract()
        {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "SuperSmoother failed: {e}"
                ))]));
            }
        };

        let result = match supsmu(
            cols.numeric(&request.x),
            cols.numeric(&request.y),
            request.weights.as_deref().map(|w| cols.numeric(w)),
            request.span,
            request.periodic.unwrap_or(false),
            request.bass.unwrap_or(0.0),
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "SuperSmoother failed: {}",
                    e
                ))]));
            }
        };

        // Format output
        let mut output = serde_json::json!({
            "n_unique_points": result.n,
            "bass": result.bass,
            "periodic": result.periodic,
            "x_sample": result.x.iter().take(20).collect::<Vec<_>>(),
            "y_smoothed_sample": result.y.iter().take(20).collect::<Vec<_>>(),
            "note": if result.n > 20 { Some(format!("Showing first 20 of {} points", result.n)) } else { None::<String> }
        });
        if let Some(m) = missing_json(cols.missing()) {
            output["missing"] = m;
        }

        Ok(CallToolResult::success(vec![Content::text(json_text(
            &output,
        ))]))
    }

    /// Fit Tukey's resistant line.
    #[tool(
        description = "Tukey's resistant line - robust regression using medians instead of means, making it resistant to outliers. Divides data into three groups and uses group medians to determine slope and intercept. Returns coefficients, fitted values, and residuals."
    )]
    pub async fn regression_line(
        &self,
        Parameters(request): Parameters<LineRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        // x and y with R's deletion of incomplete pairs; too few complete pairs is
        // reported against the dataset's row count
        let cols = match ColumnRequest::new(dataset.df())
            .numeric(&request.x)
            .numeric(&request.y)
            .na(NaPolicy::Omit)
            .min_rows(
                3,
                &format!(
                    "Tukey's resistant line of '{}' on '{}'",
                    request.y, request.x
                ),
            )
            .extract()
        {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Tukey's line failed: {e}"
                ))]));
            }
        };

        let result = match run_line(
            cols.numeric(&request.x),
            cols.numeric(&request.y),
            request.iter,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Tukey's line failed: {}",
                    e
                ))]));
            }
        };

        // Format output
        let mut output = serde_json::json!({
            "n_observations": result.n,
            "iterations": result.iter,
            "coefficients": {
                "intercept": result.intercept,
                "slope": result.slope
            },
            "fitted_sample": result.fitted.iter().take(20).collect::<Vec<_>>(),
            "residuals_sample": result.residuals.iter().take(20).collect::<Vec<_>>(),
            "note": if result.n > 20 { Some(format!("Showing first 20 of {} values", result.n)) } else { None::<String> }
        });
        if let Some(m) = missing_json(cols.missing()) {
            output["missing"] = m;
        }

        Ok(CallToolResult::success(vec![Content::text(json_text(
            &output,
        ))]))
    }

    /// Stepwise regression model selection.
    #[tool(
        description = "Stepwise regression model selection using AIC or BIC. Can perform forward selection, backward elimination, or both. Returns the best model and selection history."
    )]
    pub async fn regression_step(
        &self,
        Parameters(request): Parameters<StepRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let direction_str = request.direction.as_deref().unwrap_or("both");
        let use_bic = request.criterion.as_deref() == Some("bic");

        // For stepwise, scope_lower is empty (no forced variables), scope_upper is all predictors
        let scope_lower: Vec<&str> = vec![];
        let scope_upper: Vec<&str> = request.predictors.iter().map(|s| s.as_str()).collect();

        let result = match run_step(
            dataset,
            &request.response,
            &scope_lower,
            &scope_upper,
            direction_str,
            use_bic,
            true,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Stepwise selection failed: {}",
                    e
                ))]));
            }
        };

        let output = serde_json::json!({
            "final_model": {
                "variables": result.final_variables,
                "criterion": result.criterion_name,
                "k": result.k
            },
            "initial_variables": result.initial_variables,
            "direction": format!("{}", result.direction),
            "n_steps": result.n_steps,
            "steps": result.steps.iter().map(|s| serde_json::json!({
                "step": s.step,
                "action": s.action,
                "variables": s.variables,
                "df": s.df,
                "rss": s.rss,
                "criterion": s.criterion
            })).collect::<Vec<_>>()
        });

        Ok(CallToolResult::success(vec![Content::text(json_text(
            &output,
        ))]))
    }

    /// Fit a Generalized Least Squares model.
    #[tool(
        description = "Fit a Generalized Least Squares (GLS) regression model. GLS extends OLS to handle correlated or heteroscedastic errors. Correlation structures: 'ar1' (autoregressive), 'compound_symmetry' (equal correlation), 'identity' (OLS). For AR(1), rho can be specified or auto-estimated from OLS residuals. Returns coefficients, standard errors, t-values, p-values, and model fit statistics."
    )]
    pub async fn regression_gls(
        &self,
        Parameters(request): Parameters<GlsRequest>,
    ) -> Result<CallToolResult, McpError> {
        use ndarray::Array1;
        use p2a_core::linalg::design::DesignMatrix;

        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        // Extract y column (a missing value is an error, as nlme::gls's na.fail)
        let y = match numeric_column(dataset, &request.y, NaPolicy::Fail) {
            Ok((v, _)) => Array1::from_vec(v),
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "GLS regression failed: {e}"
                ))]));
            }
        };

        // Build design matrix
        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();
        let intercept = request.intercept.unwrap_or(true);

        let dm = match DesignMatrix::from_dataframe(dataset.df(), &x_refs, intercept) {
            Ok(dm) => dm,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to build design matrix: {}",
                    e
                ))]));
            }
        };

        let x = dm.view();

        // Determine correlation structure
        let corr_type = request.correlation.as_deref().unwrap_or("ar1");

        let correlation = match corr_type {
            "identity" | "ols" => CorrelationStructure::Identity,
            "compound_symmetry" | "cs" => {
                let rho = request.rho.unwrap_or(0.5);
                CorrelationStructure::CompoundSymmetry { rho }
            }
            "ar1" | "ar" => {
                // If rho not provided, auto-estimate from OLS residuals
                let rho = if let Some(r) = request.rho {
                    r
                } else {
                    // Run OLS to get residuals for AR(1) estimation
                    match gls(&y.view(), &x, CorrelationStructure::Identity) {
                        Ok(ols_result) => {
                            // Estimate rho from residuals
                            let residuals = &ols_result.residuals;
                            let n = residuals.len();
                            if n > 1 {
                                let mut sum_prod = 0.0;
                                let mut sum_sq = 0.0;
                                for i in 1..n {
                                    sum_prod += residuals[i] * residuals[i - 1];
                                    sum_sq += residuals[i - 1].powi(2);
                                }
                                (sum_prod / sum_sq).clamp(-0.99, 0.99)
                            } else {
                                0.0
                            }
                        }
                        Err(_) => 0.5, // fallback
                    }
                };
                CorrelationStructure::AR1 { rho }
            }
            other => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown correlation structure '{}'. Supported: ar1, compound_symmetry, identity.",
                    other
                ))]));
            }
        };

        match gls(&y.view(), &x, correlation) {
            Ok(result) => {
                // Build coefficient names
                let mut coef_names: Vec<String> = if intercept {
                    vec!["(Intercept)".to_string()]
                } else {
                    vec![]
                };
                coef_names.extend(request.x.iter().cloned());

                let mut coef_table: Vec<serde_json::Value> = Vec::new();
                for (i, name) in coef_names.iter().enumerate() {
                    coef_table.push(serde_json::json!({
                        "term": name,
                        "estimate": result.coefficients[i],
                        "std_error": result.std_errors[i],
                        "t_value": result.t_values[i],
                        "p_value": result.p_values[i]
                    }));
                }

                let json_output = serde_json::json!({
                    "model": "Generalized Least Squares",
                    "correlation": result.correlation,
                    "correlation_param": result.correlation_param,
                    "coefficients": coef_table,
                    "sigma": result.sigma,
                    "r_squared": result.r_squared,
                    "adj_r_squared": result.adj_r_squared,
                    "log_likelihood": result.log_likelihood,
                    "aic": result.aic,
                    "bic": result.bic,
                    "n_obs": result.n_obs,
                    "df_residual": result.df_residual
                });

                Ok(CallToolResult::success(vec![Content::text(json_text(
                    &json_output,
                ))]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "GLS regression failed: {}",
                e
            ))])),
        }
    }

    /// Fit a smoothing spline.
    #[tool(
        description = "Fit a smoothing spline to data. Smoothing splines balance goodness-of-fit against smoothness using a penalty on curvature. The smoothing parameter (spar) or degrees of freedom (df) controls the tradeoff. If neither is specified, uses generalized cross-validation (GCV) to automatically select optimal smoothing. Returns fitted values, effective degrees of freedom, and GCV score."
    )]
    pub async fn regression_smooth_spline(
        &self,
        Parameters(request): Parameters<SmoothSplineRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        // x and y; a missing value is an error, as R's smooth.spline refuses them
        let cols = match numeric_columns(
            dataset,
            &[request.x.as_str(), request.y.as_str()],
            NaPolicy::Fail,
        ) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Smooth spline failed: {e}"
                ))]));
            }
        };

        let config = SmoothSplineConfig {
            spar: request.spar,
            df: request.df,
            ..Default::default()
        };

        match smooth_spline(cols.numeric(&request.x), cols.numeric(&request.y), config) {
            Ok(result) => {
                // Get predictions at xout if specified, otherwise at data points
                let (pred_x, pred_y) = if let Some(ref xout) = request.xout {
                    match smooth_spline_predict(&result, xout) {
                        Ok(yout) => (xout.clone(), yout),
                        Err(e) => {
                            return Ok(CallToolResult::error(vec![Content::text(format!(
                                "Prediction failed: {}",
                                e
                            ))]));
                        }
                    }
                } else {
                    (result.x.clone(), result.y.clone())
                };

                let json_output = serde_json::json!({
                    "model": "Smoothing Spline",
                    "x": pred_x,
                    "fitted": pred_y,
                    "spar": result.spar,
                    "df": result.df,
                    "lambda": result.lambda,
                    "cv_crit": result.cv_crit,
                    "n_obs": result.n_obs,
                    "n_knots": result.n_knots
                });

                Ok(CallToolResult::success(vec![Content::text(json_text(
                    &json_output,
                ))]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Smooth spline failed: {}",
                e
            ))])),
        }
    }

    /// Run elastic net/lasso/ridge regression (glmnet).
    #[tool(
        description = "Elastic net, lasso or ridge regression as R glmnet::glmnet (coordinate descent, glmnet's standardisation and lambda path). alpha = 1 lasso (default), 0 ridge, in between elastic net; family 'gaussian' (default) or 'binomial'. Returns JSON with the coefficient path: for each lambda the intercept, df, deviance ratio and the coefficients (all of them for up to 20 predictors, otherwise the 20 largest in absolute value and the number of other non-zero ones). The selection summary (order of entry and the estimates at the smallest lambda) comes before the full path."
    )]
    pub async fn regression_glmnet(
        &self,
        Parameters(request): Parameters<GlmnetRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let family =
            match parse_option::<GlmnetFamily>(request.family.as_deref(), GlmnetFamily::Gaussian) {
                Ok(f) => f,
                Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
            };

        let mut config = GlmnetConfig::default();
        config.alpha = request.alpha.unwrap_or(1.0);
        config.lambda = request.lambda;
        config.nlambda = request.nlambda.unwrap_or(100);
        config.standardize = request.standardize.unwrap_or(true);
        config.family = family;

        match run_glmnet(dataset, &request.y, &x_refs, &config) {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(glmnet_json(
                &result,
            ))])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Glmnet failed: {}",
                e
            ))])),
        }
    }

    /// Run cross-validated glmnet for optimal lambda selection.
    #[tool(
        description = "Cross-validated elastic net as R glmnet::cv.glmnet: observations are assigned to nfolds folds (default 10) at random from seed (default: the session seed set by set_seed, else 42); each fold is fitted on its own lambda path. Returns JSON with lambda.min and lambda.1se and their 1-based indices, the coefficients at both, the CV curve (lambda, cvm, cvsd, df) and the seed used."
    )]
    pub async fn regression_cv_glmnet(
        &self,
        Parameters(request): Parameters<CvGlmnetRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let family =
            match parse_option::<GlmnetFamily>(request.family.as_deref(), GlmnetFamily::Gaussian) {
                Ok(f) => f,
                Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
            };

        let mut config = GlmnetConfig::default();
        config.alpha = request.alpha.unwrap_or(1.0);
        config.nlambda = request.nlambda.unwrap_or(100);
        config.standardize = request.standardize.unwrap_or(true);
        config.family = family;

        let nfolds = request.nfolds.unwrap_or(10);
        let seed = request.seed.or(*self.global_seed.read().await);

        match run_cv_glmnet(dataset, &request.y, &x_refs, &config, nfolds, seed) {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(
                cv_glmnet_json(
                    &result,
                    seed.unwrap_or(p2a_core::regression::glmnet::CV_GLMNET_DEFAULT_SEED),
                ),
            )])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "CV Glmnet failed: {}",
                e
            ))])),
        }
    }

    /// Run ridge regression (L2 penalty).
    #[tool(
        description = "Ridge regression: glmnet with alpha = 0, as R glmnet::glmnet(alpha = 0). Returns the coefficient path as JSON (see regression_glmnet)."
    )]
    pub async fn regression_ridge(
        &self,
        Parameters(request): Parameters<RidgeRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        match run_ridge(dataset, &request.y, &x_refs, request.lambda) {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(glmnet_json(
                &result,
            ))])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Ridge regression failed: {}",
                e
            ))])),
        }
    }

    /// Run lasso regression (L1 penalty).
    #[tool(
        description = "Lasso regression: glmnet with alpha = 1, as R glmnet::glmnet(alpha = 1); coefficients shrunk exactly to zero are not selected. Returns the coefficient path as JSON (see regression_glmnet)."
    )]
    pub async fn regression_lasso(
        &self,
        Parameters(request): Parameters<LassoRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        match run_lasso(dataset, &request.y, &x_refs, request.lambda) {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(glmnet_json(
                &result,
            ))])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Lasso regression failed: {}",
                e
            ))])),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::common::test_support::{server_with, text};
    use p2a_core::data::{DataLoader, Dataset};
    use p2a_validation::{Case, Check, Tol};
    use serde_json::{Value, json};

    #[test]
    fn test_parse_choice_parses_and_reports_the_option() {
        use p2a_core::options::Choices;
        const T: Choices<u8> = Choices {
            option: "criterion",
            values: &[(&["aic"], 1), (&["bic", "schwarz"], 2)],
        };
        assert_eq!(super::parse_choice(&T, None, 1), Ok(1));
        assert_eq!(super::parse_choice(&T, Some(" Schwarz "), 1), Ok(2));
        let e = super::parse_choice(&T, Some("cp"), 1).unwrap_err();
        assert!(
            e.contains("unknown criterion 'cp'") && e.contains("aic, bic"),
            "{e}"
        );
    }

    struct Notes(Option<(bool, usize)>, Vec<String>);
    impl p2a_core::traits::convergence::ResultNotes for Notes {
        fn notes_warnings(&self) -> Vec<String> {
            self.1.clone()
        }
        fn notes_convergence(&self) -> Option<p2a_core::traits::convergence::Convergence> {
            self.0.map(|(c, i)| {
                p2a_core::traits::convergence::Convergence::new(
                    c,
                    i,
                    100,
                    "relative change",
                    f64::NAN,
                    1e-8,
                )
            })
        }
    }

    #[test]
    fn test_append_unshown_notes_adds_only_what_is_missing() {
        let r = Notes(
            Some((true, 4)),
            vec!["Warning: x dropped".into(), "rows dropped (y: 2)".into()],
        );
        let mut t = "Model\nMissing values: 2 rows dropped (y: 2)\n".to_string();
        super::append_unshown_notes(&mut t, &r);
        assert!(
            t.ends_with("Converged: yes (4 iterations)\nWarning: x dropped"),
            "{t}"
        );
        assert_eq!(t.matches("rows dropped").count(), 1, "{t}");
        let before = t.clone();
        super::append_unshown_notes(&mut t, &r);
        assert_eq!(t, before);
        let mut quiet = "Model".to_string();
        super::append_unshown_notes(&mut quiet, &Notes(None, vec![]));
        assert_eq!(quiet, "Model");
    }

    #[test]
    fn test_json_text_never_panics() {
        assert_eq!(
            super::json_text(&serde_json::json!({"a": 1})),
            "{\n  \"a\": 1\n}"
        );
        assert!(super::json_text(&serde_json::json!({"x": f64::NAN})).contains("null"));
        assert_eq!(
            super::json_compact(&serde_json::json!({"a": [1, 2]})),
            "{\"a\":[1,2]}"
        );
    }

    /// `inv ~ value + capital` on `grunfeld`, plus `extra` keys.
    fn request(extra: Value) -> OlsRequest {
        let mut v = json!({"dataset": "grunfeld", "y": "inv", "x": ["value", "capital"]});
        for (k, x) in extra.as_object().unwrap() {
            v[k] = x.clone();
        }
        serde_json::from_value(v).unwrap()
    }

    /// The case and a server holding its dataset as `grunfeld`.
    fn case_server(id: &str) -> (Case, tokio::runtime::Runtime, AnalyticsServer) {
        let case = Case::load(id);
        let df = DataLoader::load_csv(case.dataset_path()).unwrap();
        let (rt, server) = server_with(vec![("grunfeld", df)]);
        (case, rt, server)
    }

    /// Rows `name est se t p` of the `OlsResult` Display table: the lines after
    /// the dashed rule under `Coefficients:`, up to the next empty line.
    fn rows(text: &str) -> Vec<(String, Vec<f64>)> {
        text.lines()
            .skip_while(|l| *l != "Coefficients:")
            .skip(3)
            .take_while(|l| !l.trim().is_empty())
            .map(|l| {
                let tokens: Vec<&str> = l.split_whitespace().collect();
                let numbers = tokens[1..5]
                    .iter()
                    .map(|t| {
                        t.parse::<f64>()
                            .unwrap_or_else(|_| panic!("`{t}` in `{l}`"))
                    })
                    .collect();
                (tokens[0].to_string(), numbers)
            })
            .collect()
    }

    fn call(
        rt: &tokio::runtime::Runtime,
        server: &AnalyticsServer,
        extra: Value,
    ) -> CallToolResult {
        rt.block_on(server.regression_ols(Parameters(request(extra))))
            .unwrap()
    }

    #[test]
    fn test_validate_mcp_regression_ols() {
        let (case, rt, server) = case_server("ols_grunfeld");
        for (extra, label, se_key) in [
            (json!({}), CovarianceType::HC1, "se_hc1"),
            (
                json!({"cov_type": "standard"}),
                CovarianceType::Standard,
                "se_classical",
            ),
        ] {
            let r = call(&rt, &server, extra);
            assert_ne!(r.is_error, Some(true));
            let t = text(&r);
            assert!(
                t.contains(&format!("Standard Errors: {label}")),
                "missing `Standard Errors: {label}` in\n{t}"
            );
            let (names, values): (Vec<String>, Vec<Vec<f64>>) = rows(&t).into_iter().unzip();
            let se: Vec<f64> = values.iter().map(|v| v[1]).collect();
            let coef: Vec<f64> = values.iter().map(|v| v[0]).collect();
            Check::new(&case)
                .named("coef", &names, &coef, Tol::printed(4))
                .named(se_key, &names, &se, Tol::printed(4))
                .finish();
        }
    }

    /// Every cov_type spelling, through the request-to-estimator path, against
    /// sandwich::vcovHC / lm at full tolerance.
    #[test]
    fn test_validate_mcp_regression_ols_cov_types() {
        let case = Case::load("ols_grunfeld");
        let ds = Dataset::new(DataLoader::load_csv(case.dataset_path()).unwrap());
        for (spelling, key) in [
            (None, "hc1"),
            (Some("standard"), "classical"),
            (Some("classical"), "classical"),
            (Some("hc0"), "hc0"),
            (Some("HC1"), "hc1"),
            (Some("hc2"), "hc2"),
            (Some("Hc3"), "hc3"),
        ] {
            let extra = match spelling {
                Some(s) => json!({"cov_type": s}),
                None => json!({}),
            };
            let r = run_ols_request(&ds, &request(extra)).unwrap();
            let n = r.n_obs;
            let names: Vec<&str> = r.coefficients.iter().map(|c| c.name.as_str()).collect();
            let pick = |f: fn(&p2a_core::regression::OlsCoefficient) -> f64| -> Vec<f64> {
                r.coefficients.iter().map(f).collect()
            };
            Check::new(&case)
                .named("coef", &names, &pick(|c| c.estimate), Tol::coef(n))
                .named(
                    &format!("se_{key}"),
                    &names,
                    &pick(|c| c.std_error),
                    Tol::se(n),
                )
                .named(
                    &format!("t_{key}"),
                    &names,
                    &pick(|c| c.t_value),
                    Tol::stat(n),
                )
                .named(
                    &format!("p_{key}"),
                    &names,
                    &pick(|c| c.p_value),
                    Tol::p_value(n),
                )
                .scalar("adj_r2", r.adj_r_squared, Tol::stat(n))
                .finish();
        }
    }

    /// An unknown cov_type is a tool error naming the allowed values; a long
    /// value is echoed cut to 64 characters.
    #[test]
    fn test_mcp_regression_ols_rejects_unknown_cov_type() {
        let (_case, rt, server) = case_server("ols_grunfeld");
        let r = call(&rt, &server, json!({"cov_type": "hc4"}));
        assert_eq!(r.is_error, Some(true));
        let t = text(&r);
        assert!(t.contains("cov_type") && t.contains("'hc4'"), "{t}");
        let lower = t.to_lowercase();
        assert!(lower.contains("standard") && lower.contains("hc1"), "{t}");

        let long = "x".repeat(200);
        let r = call(&rt, &server, json!({"cov_type": long}));
        assert_eq!(r.is_error, Some(true));
        let t = text(&r);
        assert!(t.contains(&format!("'{}'", "x".repeat(64))), "{t}");
        assert!(!t.contains(&"x".repeat(65)), "{t}");
    }

    /// No R case for the no-intercept model: this checks the option plumbing.
    #[test]
    fn test_mcp_regression_ols_intercept_false() {
        let (case, rt, server) = case_server("ols_grunfeld");
        let r = call(&rt, &server, json!({"intercept": false}));
        assert_ne!(r.is_error, Some(true));
        let t = text(&r);
        let rows = rows(&t);
        assert!(rows.iter().all(|(name, _)| name != "(Intercept)"), "{t}");
        assert_eq!(rows.len(), 2, "{t}");
        let ds = Dataset::new(DataLoader::load_csv(case.dataset_path()).unwrap());
        let expected = run_ols(
            &ds,
            "inv",
            &["value", "capital"],
            false,
            CovarianceType::HC1,
        )
        .unwrap();
        for ((name, v), c) in rows.iter().zip(&expected.coefficients) {
            assert_eq!(name, &c.name);
            assert_eq!(format!("{:.4}", v[1]), format!("{:.4}", c.std_error), "{t}");
        }
    }

    #[test]
    fn test_mcp_regression_ols_schema_documents_options() {
        let tool = AnalyticsServer::regression_router()
            .list_all()
            .into_iter()
            .find(|t| t.name == "regression_ols")
            .expect("regression_ols is advertised");
        let description = tool.description.as_deref().unwrap_or("");
        assert!(
            description.contains("HC1") && description.contains("cov_type = 'standard'"),
            "{description}"
        );
        let properties = tool.input_schema["properties"].as_object().unwrap();
        assert!(
            properties.contains_key("cov_type") && properties.contains_key("intercept"),
            "{properties:?}"
        );
        let required: Vec<&str> = tool
            .input_schema
            .get("required")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        assert!(
            !required.contains(&"cov_type") && !required.contains(&"intercept"),
            "{required:?}"
        );
    }

    /// One server holding the dataset of each case, named by its file stem
    /// (`grunfeld_cluster_na`, `smoothers_na`).
    fn cases_server(ids: &[&str]) -> (tokio::runtime::Runtime, AnalyticsServer) {
        let frames: Vec<(String, p2a_core::polars::prelude::DataFrame)> = ids
            .iter()
            .map(|id| {
                let path = Case::load(id).dataset_path();
                let stem = path.file_stem().unwrap().to_str().unwrap().to_string();
                (stem, DataLoader::load_csv(&path).unwrap())
            })
            .collect();
        server_with(
            frames
                .iter()
                .map(|(name, df)| (name.as_str(), df.clone()))
                .collect(),
        )
    }

    /// The CSV at `rel` under the repository root, loaded as `p2a data load` loads it
    /// but without an R case: for property tests and for datasets no R value is read
    /// from.
    fn repo_csv(rel: &str) -> p2a_core::polars::prelude::DataFrame {
        let path = p2a_validation::repo_path(rel);
        DataLoader::load_csv(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }

    /// `validation/datasets/generated/mtcars.csv` (`hp` is Int64, `model` is text).
    fn mtcars_csv() -> p2a_core::polars::prelude::DataFrame {
        repo_csv("validation/datasets/generated/mtcars.csv")
    }

    /// The tool's text output as JSON.
    fn json_of(r: &CallToolResult) -> Value {
        let t = text(r);
        serde_json::from_str(&t).unwrap_or_else(|e| panic!("not JSON ({e}): {t}"))
    }

    /// Every number of field `key` in the JSON text `json`: the value of each
    /// `"key": <number>` and the elements of each `"key": [<numbers>]`, read with
    /// `str::parse::<f64>`, which rounds correctly (as serde_json does too, since the
    /// workspace enables its `float_roundtrip` feature), so exact comparisons hold.
    fn json_numbers(json: &str, key: &str) -> Vec<f64> {
        let pattern = format!("\"{key}\":");
        let mut numbers = Vec::new();
        let mut rest = json;
        while let Some(at) = rest.find(&pattern) {
            rest = rest[at + pattern.len()..].trim_start();
            let (body, tail) = match rest.strip_prefix('[') {
                Some(array) => array.split_once(']').expect("an array closed by ]"),
                None => rest.split_at(rest.find([',', '}', '\n']).unwrap_or(rest.len())),
            };
            numbers.extend(
                body.split(',')
                    .map(str::trim)
                    .filter(|t| !t.is_empty())
                    .map(|t| {
                        t.parse::<f64>()
                            .unwrap_or_else(|e| panic!("`{t}` in field {key}: {e}"))
                    }),
            );
            rest = tail;
        }
        numbers
    }

    /// `missing` of reg_smoothers_na: 4 of 60 rows dropped, 2 missing x and 2 missing y.
    fn smoothers_missing() -> Value {
        json!({"n_total": 60, "n_used": 56, "n_dropped": 4, "by_column": {"x": 2, "y": 2}})
    }

    /// The complete (x, y) pairs of a case dataset, in row order.
    fn complete_pairs(case: &Case) -> (Vec<f64>, Vec<f64>) {
        let df = DataLoader::load_csv(case.dataset_path()).unwrap();
        let x = p2a_core::data::columns::numeric_options(&df, "x").unwrap();
        let y = p2a_core::data::columns::numeric_options(&df, "y").unwrap();
        x.iter().zip(&y).filter_map(|(a, b)| a.zip(*b)).unzip()
    }

    /// REG-21 through the tool: a missing cluster id drops its row (one- and two-way),
    /// the model is re-estimated, and the text reports the drop.
    #[test]
    fn test_validate_mcp_regression_clustered_na() {
        let case = Case::load("reg_cluster_na");
        let (rt, server) = cases_server(&["reg_cluster_na"]);
        let call = |extra: Value| -> String {
            let mut v = json!({"dataset": "grunfeld_cluster_na", "y": "inv",
                               "x": ["value", "capital"], "cluster1": "firm_label"});
            for (k, x) in extra.as_object().unwrap() {
                v[k] = x.clone();
            }
            let req: OlsClusteredRequest = serde_json::from_value(v).unwrap();
            let r = rt
                .block_on(server.regression_clustered(Parameters(req)))
                .unwrap();
            assert_ne!(r.is_error, Some(true), "{}", text(&r));
            text(&r)
        };
        let split = |t: &str| -> (Vec<String>, Vec<f64>, Vec<f64>) {
            let rows = rows(t);
            (
                rows.iter().map(|(n, _)| n.clone()).collect(),
                rows.iter().map(|(_, v)| v[0]).collect(),
                rows.iter().map(|(_, v)| v[1]).collect(),
            )
        };
        let t = call(json!({}));
        assert!(
            t.contains("Observations: 197\n") && t.contains("Clusters: 10\n"),
            "{t}"
        );
        assert!(
            t.contains("  - 3 of 200 rows dropped because of missing values (firm_label: 3)\n"),
            "{t}"
        );
        let (names, coef, se) = split(&t);
        Check::new(&case)
            .named("coef", &names, &coef, Tol::printed(4))
            .named("se_cluster_hc1", &names, &se, Tol::printed(4))
            .finish();
        let t = call(json!({"cluster2": "year_na"}));
        assert!(
            t.contains("Observations: 195\n") && t.contains("Clusters: 10 x 20\n"),
            "{t}"
        );
        assert!(
            t.contains(
                "  - 5 of 200 rows dropped because of missing values (firm_label: 3, year_na: 2)\n"
            ),
            "{t}"
        );
        let (names, coef, se) = split(&t);
        Check::new(&case)
            .named("coef_twoway", &names, &coef, Tol::printed(4))
            .named("se_cluster_twoway_hc1", &names, &se, Tol::printed(4))
            .finish();
    }

    /// REG-21 (Driscoll-Kraay): a missing time id drops its row; a fractional time id
    /// is a tool error instead of being truncated to a period. A property test (the DK
    /// SEs are not R-pinned, REG-24), so it loads the dataset without a case.
    #[test]
    fn test_mcp_regression_driscoll_kraay_missing_time() {
        use p2a_core::data::columns::{NaPolicy, numeric_column};
        use p2a_core::polars::prelude::{NamedFrom, Series};
        let df = repo_csv("validation/datasets/generated/grunfeld_cluster_na.csv");
        let (year, _) = numeric_column(&df, "year", NaPolicy::Fail).unwrap();
        let mut quarterly = df.clone();
        let t_q: Vec<f64> = year.iter().map(|y| y + 0.25).collect();
        quarterly
            .with_column(Series::new("t_q".into(), t_q))
            .unwrap();
        let (rt, server) = server_with(vec![("grunfeld_cluster_na", df), ("quarterly", quarterly)]);
        let call = |dataset: &str, time_col: &str| -> CallToolResult {
            let req: DriscollKraayRequest = serde_json::from_value(json!({
                "dataset": dataset, "y": "inv", "x": ["value", "capital"], "time_col": time_col
            }))
            .unwrap();
            rt.block_on(server.regression_driscoll_kraay(Parameters(req)))
                .unwrap()
        };
        let r = call("grunfeld_cluster_na", "year_na");
        let t = text(&r);
        assert_ne!(r.is_error, Some(true), "{t}");
        assert!(t.contains("N = 198 observations, T = 20 periods"), "{t}");
        assert!(
            t.contains("2 of 200 rows dropped because of missing values (year_na: 2)"),
            "{t}"
        );
        let r = call("quarterly", "t_q");
        let t = text(&r);
        assert_eq!(r.is_error, Some(true), "{t}");
        assert!(
            t.contains("Time column 't_q' must hold whole numbers") && t.contains("1935.25"),
            "{t}"
        );
    }

    /// REG-23 through the tool: an Int64 outcome (mtcars hp) is a number. A property
    /// test (no value is compared with R), so it loads the dataset without a case.
    #[test]
    fn test_mcp_regression_bootstrap_cov_integer_outcome() {
        let (rt, server) = server_with(vec![("mtcars", mtcars_csv())]);
        let req: BootstrapCovRequest = serde_json::from_value(json!({
            "dataset": "mtcars", "y": "hp", "x": ["wt"], "n_boot": 200, "seed": 42
        }))
        .unwrap();
        let r = rt
            .block_on(server.regression_bootstrap_cov(Parameters(req)))
            .unwrap();
        let t = text(&r);
        assert_ne!(r.is_error, Some(true), "{t}");
        assert!(t.contains("Bootstrap Covariance Estimation"), "{t}");
    }

    /// REG-25 through the tool: loess drops incomplete rows (R's na.omit; the number of
    /// rows used is R's `n_complete`) and reports them, and an Int64 x is a number.
    #[test]
    fn test_validate_mcp_regression_loess_missing_and_integer() {
        let case = Case::load("reg_smoothers_na");
        let (rt, server) = server_with(vec![
            (
                "smoothers_na",
                DataLoader::load_csv(case.dataset_path()).unwrap(),
            ),
            ("mtcars", mtcars_csv()),
        ]);
        let call = |v: Value| -> CallToolResult {
            let req: LoessRequest = serde_json::from_value(v).unwrap();
            rt.block_on(server.regression_loess(Parameters(req)))
                .unwrap()
        };
        let r = call(json!({"dataset": "smoothers_na", "y": "y", "x": "x"}));
        assert_ne!(r.is_error, Some(true), "{}", text(&r));
        let j = json_of(&r);
        Check::new(&case)
            .scalar("n_complete", j["n_obs"].as_f64().unwrap(), Tol::exact())
            .finish();
        assert_eq!(j["n_obs"], 56);
        assert_eq!(j["fitted_values"].as_array().unwrap().len(), 56);
        assert_eq!(j["missing"], smoothers_missing());
        let r = call(json!({"dataset": "mtcars", "y": "mpg", "x": "hp"}));
        assert_ne!(r.is_error, Some(true), "{}", text(&r));
        let j = json_of(&r);
        assert_eq!(j["n_obs"], 32);
        assert!(j.get("missing").is_none(), "{j}");
    }

    /// TOOL-3: the GLS y refuses missing values (nlme::gls stops) instead of a fit
    /// with null coefficients. A property test (no value is compared with R), so it
    /// loads the dataset without a case.
    #[test]
    fn test_mcp_regression_gls_missing_y_is_error() {
        let (rt, server) = server_with(vec![(
            "smoothers_na",
            repo_csv("validation/datasets/generated/smoothers_na.csv"),
        )]);
        let req: GlsRequest = serde_json::from_value(json!({
            "dataset": "smoothers_na", "y": "y", "x": ["id"], "correlation": "identity"
        }))
        .unwrap();
        let r = rt.block_on(server.regression_gls(Parameters(req))).unwrap();
        let t = text(&r);
        assert_eq!(r.is_error, Some(true), "{t}");
        assert!(t.contains("Column 'y' contains 2 missing value(s)"), "{t}");
    }

    /// REG-25 through the tool: the smoothing spline refuses missing values (as R's
    /// smooth.spline) and accepts an Int64 x. A property test (no value is compared with
    /// R), so it loads the datasets without a case.
    #[test]
    fn test_mcp_regression_smooth_spline_missing_is_error() {
        let (rt, server) = server_with(vec![
            (
                "smoothers_na",
                repo_csv("validation/datasets/generated/smoothers_na.csv"),
            ),
            ("mtcars", mtcars_csv()),
        ]);
        let call = |v: Value| -> CallToolResult {
            let req: SmoothSplineRequest = serde_json::from_value(v).unwrap();
            rt.block_on(server.regression_smooth_spline(Parameters(req)))
                .unwrap()
        };
        let r = call(json!({"dataset": "smoothers_na", "x": "x", "y": "y"}));
        let t = text(&r);
        assert_eq!(r.is_error, Some(true), "{t}");
        assert!(t.contains("Column 'x' contains 2 missing value(s)"), "{t}");
        let r = call(json!({"dataset": "mtcars", "x": "hp", "y": "mpg", "df": 4.0}));
        assert_ne!(r.is_error, Some(true), "{}", text(&r));
        assert_eq!(json_of(&r)["n_obs"], 32);
    }

    /// REG-13 through the tool: supsmu drops incomplete (x, y, weights) rows; its x grid
    /// is R's grid after the deletion and every smoothed value is a number.
    #[test]
    fn test_validate_mcp_regression_supsmu_na() {
        let case = Case::load("reg_smoothers_na");
        let (rt, server) = server_with(vec![
            (
                "smoothers_na",
                DataLoader::load_csv(case.dataset_path()).unwrap(),
            ),
            ("mtcars", mtcars_csv()),
        ]);
        let call = |v: Value| -> CallToolResult {
            let req: SupsmuRequest = serde_json::from_value(v).unwrap();
            rt.block_on(server.regression_supsmu(Parameters(req)))
                .unwrap()
        };
        let r = call(json!({"dataset": "smoothers_na", "x": "x", "y": "y"}));
        assert_ne!(r.is_error, Some(true), "{}", text(&r));
        let j = json_of(&r);
        assert_eq!(j["n_unique_points"], 56);
        let x: Vec<f64> = j["x_sample"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        assert_eq!(x, case.vector("supsmu_x")[..20].to_vec());
        let y = j["y_smoothed_sample"].as_array().unwrap();
        assert_eq!(y.len(), 20);
        assert!(
            y.iter().all(|v| v.as_f64().is_some_and(f64::is_finite)),
            "{j}"
        );
        assert_eq!(j["missing"], smoothers_missing());
        // Weights are read with x and y: a missing weight drops its row (it used to
        // become 1.0); here the weight column is x, so the same rows go.
        let r = call(json!({"dataset": "smoothers_na", "x": "x", "y": "y", "weights": "x"}));
        assert_ne!(r.is_error, Some(true), "{}", text(&r));
        assert_eq!(json_of(&r)["missing"], smoothers_missing());
        // A text weights column is an error (it used to be ignored silently).
        let r = call(json!({"dataset": "mtcars", "x": "wt", "y": "mpg", "weights": "model"}));
        let t = text(&r);
        assert_eq!(r.is_error, Some(true), "{t}");
        assert!(t.contains("'model'"), "{t}");
    }

    /// REG-13 through the tool: line uses the complete pairs (R's line), so the fit
    /// equals line() on them, and the drop is reported.
    #[test]
    fn test_validate_mcp_regression_line_na() {
        let case = Case::load("reg_smoothers_na");
        let (rt, server) = cases_server(&["reg_smoothers_na"]);
        let req: LineRequest =
            serde_json::from_value(json!({"dataset": "smoothers_na", "x": "x", "y": "y"})).unwrap();
        let r = rt
            .block_on(server.regression_line(Parameters(req)))
            .unwrap();
        assert_ne!(r.is_error, Some(true), "{}", text(&r));
        let j = json_of(&r);
        Check::new(&case)
            .scalar(
                "n_complete",
                j["n_observations"].as_f64().unwrap(),
                Tol::exact(),
            )
            .finish();
        let (xc, yc) = complete_pairs(&case);
        let fit = p2a_core::regression::line::line(&xc, &yc, None).unwrap();
        let raw = text(&r);
        assert_eq!(json_numbers(&raw, "intercept"), [fit.intercept]);
        assert_eq!(json_numbers(&raw, "slope"), [fit.slope]);
        assert_eq!(j["missing"], smoothers_missing());
    }

    /// REG-13 through the tools: with too few complete rows, line and supsmu fail with
    /// the extractor's error, which names the method and the columns and counts the
    /// rows of the dataset (not only the complete ones the core function would see).
    #[test]
    fn test_mcp_regression_line_supsmu_too_few_complete_rows() {
        // 2 of 6 rows have both x and y.
        let df = p2a_core::polars::prelude::df!(
            "x" => [Some(1.2), Some(2.1), None, Some(4.3), None, Some(6.1)],
            "y" => [Some(2.3), None, Some(3.9), Some(5.2), None, None]
        )
        .unwrap();
        let (rt, server) = server_with(vec![("two_complete", df)]);
        let args = json!({"dataset": "two_complete", "x": "x", "y": "y"});
        let req: LineRequest = serde_json::from_value(args.clone()).unwrap();
        let r = rt
            .block_on(server.regression_line(Parameters(req)))
            .unwrap();
        let t = text(&r);
        assert_eq!(r.is_error, Some(true), "{t}");
        assert!(
            t.contains("need at least 3 observations, got 2")
                && t.contains(
                    "Tukey's resistant line of 'y' on 'x': 2 of 6 rows have no missing value"
                ),
            "{t}"
        );
        let req: SupsmuRequest = serde_json::from_value(args).unwrap();
        let r = rt
            .block_on(server.regression_supsmu(Parameters(req)))
            .unwrap();
        let t = text(&r);
        assert_eq!(r.is_error, Some(true), "{t}");
        assert!(
            t.contains("need at least 4 observations, got 2")
                && t.contains("SuperSmoother of 'y' on 'x': 2 of 6 rows have no missing value"),
            "{t}"
        );
    }
}
