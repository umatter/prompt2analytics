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
    MCP_DEFAULT_SEED, append_unshown_notes, json_compact, json_text, parse_choice,
};
use crate::tools::common::{
    NaPolicy, append_missing_note, missing_json, notes_json, numeric_column, numeric_columns,
    parse_option,
};
use crate::tools::requests::regression::{
    BgTestRequest, BootstrapCovRequest, CvGlmnetRequest, DiagnosticsRequest, DriscollKraayRequest,
    GlmnetRequest, GlsRequest, HacRequest, HarveyCollierRequest, LassoRequest, LineRequest,
    LoessRequest, NlsRequest, OlsClusteredRequest, OlsRequest, QuantRegRequest, ResetTestRequest,
    RidgeRequest, SmoothSplineRequest, StepRequest, SupsmuRequest, WaldTestRequest,
};

use p2a_core::data::Dataset;
use p2a_core::data::columns::ColumnRequest;
use p2a_core::polars::prelude::{DataFrame, NamedFrom, Series};
use p2a_core::regression::{
    BgTestType, CovarianceType, GlmnetConfig, GlmnetFamily, OlsResult, ResetType,
    SmoothSplineConfig, bg_test, harvey_collier_test, reset_test, run_cv_glmnet, run_diagnostics,
    run_glmnet, run_lasso, run_line, run_loess, run_ols, run_ols_clustered, run_ols_complete_cases,
    run_ridge, run_step, run_vcov_bootstrap, run_vcov_driscoll_kraay, run_vcov_hac, run_wls,
    smooth_spline, smooth_spline_predict, supsmu, wald_test,
};

use p2a_core::EconError;
use p2a_core::options::Choices;
use p2a_core::traits::convergence::ResultNotes;

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

/// The JSON reply of regression_glmnet, regression_lasso and regression_ridge. With
/// `path_coefficients` every path entry carries its intercept and coefficients; without it the
/// entries list lambda, df and the deviance ratio, and `at_last_lambda` gives the coefficients
/// at the last lambda, so a default 100-lambda path stays well inside the chat loop's cut.
fn glmnet_json(r: &p2a_core::regression::GlmnetResult, path_coefficients: bool) -> String {
    use serde_json::json;
    let (selection, more, never) = glmnet_selection(r);
    let listed = if r.variable_names.len() <= GLMNET_LISTED_MAX {
        "all".to_string()
    } else {
        format!("the {GLMNET_LISTED_MAX} largest |estimate| per lambda")
    };
    let path: Vec<serde_json::Value> = r
        .fits
        .iter()
        .map(|f| {
            if path_coefficients {
                glmnet_fit_json(&r.variable_names, f)
            } else {
                json!({"lambda": f.lambda, "df": f.df, "dev_ratio": f.dev_ratio})
            }
        })
        .collect();
    let at_last = if path_coefficients {
        serde_json::Value::Null
    } else {
        r.fits.last().map_or(serde_json::Value::Null, |f| {
            glmnet_fit_json(&r.variable_names, f)
        })
    };
    let mut pairs = vec![
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
        ("path_coefficients", json!(path_coefficients)),
    ];
    if !at_last.is_null() {
        pairs.push(("at_last_lambda", at_last));
    }
    pairs.push(("path", json!(path)));
    ordered_json(&pairs)
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

/// The selection criterion of regression_step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StepCriterion {
    Aic,
    Bic,
}

const STEP_CRITERION: Choices<StepCriterion> = Choices {
    option: "criterion",
    values: &[
        (&["aic"], StepCriterion::Aic),
        (&["bic", "sbc", "schwarz"], StepCriterion::Bic),
    ],
};

/// The model families of regression_nls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NlsModel {
    ExpDecay,
    ExpGrowth,
    MichaelisMenten,
    Logistic,
    Power,
    Asymptotic,
}

const NLS_MODEL: Choices<NlsModel> = Choices {
    option: "model",
    values: &[
        (&["exponential_decay", "exp_decay"], NlsModel::ExpDecay),
        (&["exponential_growth", "exp_growth"], NlsModel::ExpGrowth),
        (&["michaelis_menten", "mm"], NlsModel::MichaelisMenten),
        (&["logistic", "logistic_growth"], NlsModel::Logistic),
        (&["power", "power_law"], NlsModel::Power),
        (&["asymptotic"], NlsModel::Asymptotic),
    ],
};

const NLS_ALGORITHM: Choices<p2a_core::regression::NlsAlgorithm> = Choices {
    option: "algorithm",
    values: &[
        (
            &["levenberg_marquardt", "lm", "marquardt"],
            p2a_core::regression::NlsAlgorithm::LevenbergMarquardt,
        ),
        (
            &["gauss_newton", "gn"],
            p2a_core::regression::NlsAlgorithm::GaussNewton,
        ),
    ],
};

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

/// How many fitted values a smoother returns without `full`.
const SMOOTHER_SAMPLE: usize = 20;

/// Put a smoother's vectors into `output` by the one rule the four smoothers
/// share: with `full` every value under its key; otherwise the first
/// [`SMOOTHER_SAMPLE`] under `<key>_sample`, and, when there are more, a
/// `note` giving the total.
fn insert_smoother_values(
    output: &mut serde_json::Value,
    full: bool,
    n: usize,
    vectors: &[(&str, &[f64])],
) {
    for (key, values) in vectors {
        if full {
            output[*key] = serde_json::json!(values);
        } else {
            let shown = &values[..values.len().min(SMOOTHER_SAMPLE)];
            output[format!("{key}_sample")] = serde_json::json!(shown);
        }
    }
    if !full && n > SMOOTHER_SAMPLE {
        output["note"] = serde_json::json!(format!(
            "first {SMOOTHER_SAMPLE} of {n} values; set full = true for all"
        ));
    }
}

/// Runs the OLS a request describes on `dataset`, with its options as
/// [`ols_options`] parsed them.
fn run_ols_request(
    dataset: &Dataset,
    request: &OlsRequest,
    (intercept, cov_type): (bool, CovarianceType),
) -> Result<OlsResult, String> {
    let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();
    // rows missing in y, a regressor or the weights are dropped jointly, as lm's na.omit
    run_ols_complete_cases(
        dataset,
        &request.y,
        &x_refs,
        request.weights.as_deref(),
        intercept,
        cov_type,
    )
    .map_err(|e| format!("Regression failed: {e}"))
}

/// The columns of a saved coefficient table, in order; `viz_coefficient(model = ...)`
/// reads `term`, `estimate`, `ci_lower` and `ci_upper` from it.
pub(crate) const COEFFICIENT_TABLE_COLUMNS: [&str; 7] = [
    "term",
    "estimate",
    "std_error",
    "t_value",
    "p_value",
    "ci_lower",
    "ci_upper",
];

/// An OLS fit's coefficients as a dataset, one row per term (the 95% interval as OLS
/// reports it; an aliased term has NaN in every numeric column).
fn coefficient_table(result: &OlsResult) -> Result<Dataset, String> {
    let c = &result.coefficients;
    let col = |name: &str, f: fn(&p2a_core::regression::OlsCoefficient) -> f64| {
        Series::new(name.into(), c.iter().map(f).collect::<Vec<f64>>()).into()
    };
    DataFrame::new(vec![
        Series::new(
            "term".into(),
            c.iter().map(|k| k.name.clone()).collect::<Vec<String>>(),
        )
        .into(),
        col("estimate", |k| k.estimate),
        col("std_error", |k| k.std_error),
        col("t_value", |k| k.t_value),
        col("p_value", |k| k.p_value),
        col("ci_lower", |k| k.ci_lower_95),
        col("ci_upper", |k| k.ci_upper_95),
    ])
    .map(Dataset::new)
    .map_err(|e| e.to_string())
}

/// A coefficient table from parallel columns (any estimator): one row per term.
pub(crate) fn coefficient_table_from(
    terms: &[String],
    estimate: &[f64],
    std_error: &[f64],
    statistic: &[f64],
    p_value: &[f64],
    ci: &[(f64, f64)],
) -> Result<Dataset, String> {
    let n = terms.len();
    if [
        estimate.len(),
        std_error.len(),
        statistic.len(),
        p_value.len(),
        ci.len(),
    ]
    .iter()
    .any(|&len| len != n)
    {
        return Err("the coefficient columns differ in length".to_string());
    }
    let col = |name: &str, v: Vec<f64>| Series::new(name.into(), v).into();
    DataFrame::new(vec![
        Series::new("term".into(), terms.to_vec()).into(),
        col("estimate", estimate.to_vec()),
        col("std_error", std_error.to_vec()),
        col("t_value", statistic.to_vec()),
        col("p_value", p_value.to_vec()),
        col("ci_lower", ci.iter().map(|c| c.0).collect()),
        col("ci_upper", ci.iter().map(|c| c.1).collect()),
    ])
    .map(Dataset::new)
    .map_err(|e| e.to_string())
}

/// The reply with the saved-table line first (the chat loop keeps only the start of an
/// earlier turn's tool result, and a later "plot the coefficients" needs it), or a note
/// at the end when the table could not be saved.
pub(crate) fn with_saved_table_line(saved: Result<String, String>, text: String) -> String {
    match saved {
        Ok(name) => format!(
            "Coefficients saved as dataset '{name}' (columns {}); plot them with \
             viz_coefficient(model = '{name}').\n\n{text}",
            COEFFICIENT_TABLE_COLUMNS.join(", ")
        ),
        Err(e) => format!("{text}\n\nNote: the coefficient table was not saved: {e}"),
    }
}

/// Whether `ds` has exactly the columns of a saved coefficient table.
pub(crate) fn is_coefficient_table(ds: &Dataset) -> bool {
    let names = ds.df().get_column_names();
    names.len() == COEFFICIENT_TABLE_COLUMNS.len()
        && names
            .iter()
            .zip(COEFFICIENT_TABLE_COLUMNS)
            .all(|(n, want)| n.as_str() == want)
}

/// Saves `table` under `base`, replacing an earlier coefficient table of that name but
/// never another dataset: if `base` holds one, the table goes to `base_2`, `base_3`, ...
/// Returns the name it was saved under.
pub(crate) fn save_coefficient_table(
    datasets: &mut std::collections::HashMap<String, Dataset>,
    base: &str,
    table: Result<Dataset, String>,
) -> Result<String, String> {
    Ok(save_table(datasets, base, table?, is_coefficient_table))
}

/// Saves `table` under `base`, or `base_2`, `base_3`, ... while that names a dataset that is
/// not `replaceable`. Returns the name.
pub(crate) fn save_table(
    datasets: &mut std::collections::HashMap<String, Dataset>,
    base: &str,
    table: Dataset,
    replaceable: fn(&Dataset) -> bool,
) -> String {
    let free = |name: &str| datasets.get(name).is_none_or(replaceable);
    let name = std::iter::once(base.to_string())
        .chain((2..).map(|i| format!("{base}_{i}")))
        .find(|n| free(n))
        .expect("an unbounded sequence of names has a free one");
    datasets.insert(name.clone(), table);
    name
}

#[tool_router(router = regression_router, vis = "pub")]
impl AnalyticsServer {
    /// Run OLS regression.
    #[tool(
        description = "Run Ordinary Least Squares (OLS) regression. Returns coefficients, standard errors, t-values, p-values, R-squared, and F-statistic. Standard errors default to HC1 (heteroskedasticity-robust); set cov_type = 'standard' for classical, or 'hc0', 'hc2', 'hc3'. Set intercept = false to omit the intercept. Set weights to a column for weighted least squares (lm(weights = w)), e.g. the 'weights' column of a matched sample. Rows with a missing value in y, a regressor or the weights are dropped, as lm does, and the reply says how many. The coefficients are also saved as a dataset named in the reply's first line, which viz_coefficient plots directly. For comparing group means with a single categorical predictor, consider anova_one_way instead."
    )]
    pub async fn regression_ols(
        &self,
        Parameters(request): Parameters<OlsRequest>,
    ) -> Result<CallToolResult, McpError> {
        // An unknown cov_type is refused before the dataset is looked up.
        let options = match ols_options(&request) {
            Ok(options) => options,
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
        };

        let fitted = {
            let datasets = self.datasets.read().await;
            let dataset = crate::get_dataset!(datasets, &request.dataset);
            run_ols_request(dataset, &request, options)
        };

        match fitted {
            Ok(result) => {
                let mut text = result.to_string();
                if !request.intercept.unwrap_or(true) {
                    // without an intercept R-squared is the uncentred one (summary.lm)
                    text = text
                        .replace("\nR-squared: ", "\nR-squared (uncentered): ")
                        .replace("\nAdj. R-squared: ", "\nAdj. R-squared (uncentered): ");
                }
                append_unshown_notes(&mut text, &result);
                append_missing_note(&mut text, &result.missing);
                let saved = {
                    let mut datasets = self.datasets.write().await;
                    save_coefficient_table(
                        &mut datasets,
                        &format!("{}_ols_{}", request.dataset, request.y),
                        coefficient_table(&result),
                    )
                };
                let text = with_saved_table_line(saved, text);
                Ok(CallToolResult::success(vec![Content::text(text)]))
            }
            Err(msg) => Ok(CallToolResult::error(vec![Content::text(msg)])),
        }
    }

    /// Run regression diagnostics.
    #[tool(
        description = "Run comprehensive regression diagnostics. Tests include: Jarque-Bera (normality), Breusch-Pagan (heteroskedasticity), Durbin-Watson (autocorrelation), VIF (multicollinearity), and condition number. Rows with a missing response or regressor are dropped, as R's lm (na.omit); the output reports how many."
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
        description = "Breusch-Godfrey test for higher-order serial correlation in regression residuals. More general than Durbin-Watson as it: (1) tests for AR(p) or MA(p) correlation, (2) allows lagged dependent variables as regressors, (3) is valid regardless of regressor properties. Returns LM statistic and p-value. test_type 'chisq' (LM, default) or 'f', as lmtest::bgtest(type =); df from the rank of the augmented design (lmtest counts columns; they differ only when a term aliases). Rows with a missing response or regressor are dropped, as R's lm (na.omit); the output reports how many."
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

        let test_type =
            match parse_option::<BgTestType>(request.test_type.as_deref(), BgTestType::Chisq) {
                Ok(t) => t,
                Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
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
        description = "Ramsey's RESET test for functional form misspecification. Tests whether nonlinear terms (powers of fitted values) should be added to the model. Significant result suggests the linear model is misspecified and nonlinear terms may be needed. reset_type 'fitted' (default), 'regressor' or 'princomp' (powers of the first principal component of the regressors), as lmtest::resettest(type =); df from the rank of the augmented design (lmtest counts columns; they differ only when a term aliases). Rows with a missing response or regressor are dropped, as R's lm (na.omit); the output reports how many."
    )]
    pub async fn regression_resettest(
        &self,
        Parameters(request): Parameters<ResetTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();
        let powers = request.powers.unwrap_or_else(|| vec![2, 3]);

        let reset_type_enum =
            match parse_option::<ResetType>(request.reset_type.as_deref(), ResetType::Fitted) {
                Ok(t) => t,
                Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
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

        let mut text = result.to_string();
        append_unshown_notes(&mut text, &result);
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    /// Wald test for comparing nested linear models.
    #[tool(
        description = "Wald test (F-test) for comparing nested linear models. Tests whether variables excluded from the restricted model are jointly significant. Common uses: testing joint significance of multiple coefficients, testing nested model hypotheses, comparing model specifications. Rows with a missing response or regressor are dropped, as R's lm (na.omit); the output reports how many."
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

        let mut text = result.to_string();
        append_unshown_notes(&mut text, &result);
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    /// Harvey-Collier test for linearity using recursive residuals.
    #[tool(
        description = "Harvey-Collier test for detecting departure from linearity. Uses recursive residuals to test whether the mean of one-step-ahead forecast errors differs from zero. A significant result suggests convex or concave functional misspecification - the linear model may need polynomial terms. Equivalent to R's lmtest::harvtest(). Rows with a missing response or regressor are dropped, as R's lm (na.omit); the output reports how many."
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
        description = "Compute HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors using the Newey-West estimator. Essential for time series regression where errors may be both heteroskedastic and autocorrelated. Supports multiple kernel functions and automatic bandwidth selection. kernel 'bartlett' (default), 'parzen', 'qs' (Quadratic Spectral), 'truncated' or 'tukey-hanning' (sandwich's spellings are accepted). Rows with a missing response or regressor are dropped, as R's lm (na.omit); the output reports how many."
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

        let mut text = result.to_string();
        append_unshown_notes(&mut text, &result);
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    /// Bootstrap covariance estimation (vcovBS).
    #[tool(
        description = "Compute bootstrap covariance matrix and standard errors for OLS regression. Supports pairs bootstrap (most robust, resamples observations), residual bootstrap (assumes homoskedasticity), and wild bootstrap (robust to heteroskedasticity). Useful when asymptotic standard errors may be unreliable. As sandwich::vcovBS with type 'pairs' (default), 'residual', 'wild' (Rademacher), 'mammen' or 'webb'. Seeded: seed, else the session seed (set_seed), else 42. Rows with a missing response or regressor are dropped, as R's lm (na.omit); the output reports how many."
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
            Some(
                request
                    .seed
                    .or(*self.global_seed.read().await)
                    .unwrap_or(MCP_DEFAULT_SEED),
            ),
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Bootstrap covariance estimation failed: {}",
                    e
                ))]));
            }
        };

        let mut text = result.to_string();
        append_unshown_notes(&mut text, &result);
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    /// Driscoll-Kraay panel-robust standard errors.
    #[tool(
        description = "Compute Driscoll-Kraay (1998) panel-robust standard errors. Robust to arbitrary cross-sectional correlation (spatial dependence) and serial correlation in panel data. Aggregates score vectors by time period and applies Newey-West HAC correction. Best for panels with large T (many time periods). Returns coefficient estimates with panel-robust SEs, t-stats, and p-values. time_col must hold whole numbers or dates; rows with a missing time id are dropped and the model is re-estimated (reported in the output). kernel 'bartlett' (default), 'parzen', 'qs' (Quadratic Spectral), 'truncated' or 'tukey-hanning' (sandwich's spellings are accepted). Rows with a missing response or regressor are dropped, as R's lm (na.omit); the output reports how many."
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

        let mut text = result.to_string();
        append_unshown_notes(&mut text, &result);
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    /// Quantile regression.
    #[tool(
        description = "Quantile regression as R quantreg::rq: method 'auto' (default: 'br' up to 5,000 rows, 'fn' above; rq's own default is 'br' at every n), 'br' (Barrodale-Roberts simplex), 'fn' (Frisch-Newton interior point) or 'irls' (an approximation, not an R method); standard errors as summary.rq with se 'nid' (default), 'iid', 'ker' or 'boot'. Give one quantile tau (default 0.5) or a list taus, not both. Returns coefficients, standard errors, t, p and 95% intervals from t(n - k) for each quantile."
    )]
    pub async fn regression_quantreg(
        &self,
        Parameters(request): Parameters<QuantRegRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::regression::quantreg::{
            QuantRegConfig, QuantRegMethod, QuantRegSe, quantreg, quantreg_multi_with,
        };

        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let mut config = QuantRegConfig::default();
        config.algorithm =
            match parse_option::<QuantRegMethod>(request.method.as_deref(), QuantRegMethod::Auto) {
                Ok(m) => m,
                Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
            };
        config.se = match parse_option::<QuantRegSe>(request.se.as_deref(), QuantRegSe::Nid) {
            Ok(s) => s,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        // the bootstrap's seed (se = boot only); None = QUANTREG_BOOT_DEFAULT_SEED
        config.seed = *self.global_seed.read().await;

        let fail = |e: p2a_core::EconError| {
            Ok(CallToolResult::error(vec![Content::text(format!(
                "Quantile regression failed: {e}"
            ))]))
        };
        let results = match (request.tau, &request.taus) {
            (Some(_), Some(_)) => {
                return Ok(CallToolResult::error(vec![Content::text(
                    "give either tau or taus, not both".to_string(),
                )]));
            }
            (_, Some(taus)) => {
                match quantreg_multi_with(dataset, &request.y, &x_refs, taus, &config) {
                    Ok(r) => r,
                    Err(e) => return fail(e),
                }
            }
            (tau, None) => {
                config.tau = tau.unwrap_or(0.5);
                match quantreg(dataset, &request.y, &x_refs, &config) {
                    Ok(r) => vec![r],
                    Err(e) => return fail(e),
                }
            }
        };

        let output: String = results
            .iter()
            .map(|r| {
                let mut block = r.to_string();
                append_unshown_notes(&mut block, r);
                block
            })
            .collect::<Vec<_>>()
            .join("\n---\n");

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    /// Run OLS with clustered standard errors.
    #[tool(
        description = "Run OLS regression with clustered standard errors. Supports one-way (firm, state) or two-way (firm + time) clustering. Essential for panel data with correlated errors. Rows with a missing cluster id, response or regressor are dropped and the model is re-estimated on the remaining rows (as fixest, and lm's na.omit); the output reports how many."
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

        let mut text = result.to_string();
        append_unshown_notes(&mut text, &result);
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    /// Run nonlinear least squares regression.
    #[tool(
        description = "Fit a nonlinear regression model using Levenberg-Marquardt algorithm. Supports common models: exponential decay/growth, Michaelis-Menten kinetics, logistic growth, power law, asymptotic. Returns parameter estimates, standard errors, t-values, and convergence info. algorithm 'levenberg_marquardt' (default; also 'lm') or 'gauss_newton' (also 'gn'); model names are matched ignoring case and separators."
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
        let model = match parse_choice(&NLS_MODEL, Some(&request.model), NlsModel::ExpDecay) {
            Ok(m) => m,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        type ModelFn = fn(&Array1<f64>, &Array1<f64>) -> f64;
        let (model_fn, param_names, usage): (ModelFn, Vec<&str>, &str) = match model {
            NlsModel::ExpDecay => (
                model_exponential_decay,
                vec!["a", "b", "c"],
                "exponential_decay requires 3 starting values: [a, b, c] for y = a*exp(-b*x) + c",
            ),
            NlsModel::ExpGrowth => (
                model_exponential_growth,
                vec!["a", "b"],
                "exponential_growth requires 2 starting values: [a, b] for y = a*exp(b*x)",
            ),
            NlsModel::MichaelisMenten => (
                model_michaelis_menten,
                vec!["Vmax", "Km"],
                "michaelis_menten requires 2 starting values: [Vmax, Km] for y = Vmax*x/(Km+x)",
            ),
            NlsModel::Logistic => (
                model_logistic_growth,
                vec!["K", "r", "x0"],
                "logistic requires 3 starting values: [K, r, x0] for y = K/(1+exp(-r*(x-x0)))",
            ),
            NlsModel::Power => (
                model_power,
                vec!["a", "b"],
                "power requires 2 starting values: [a, b] for y = a*x^b",
            ),
            NlsModel::Asymptotic => (
                model_asymptotic,
                vec!["a", "b", "c"],
                "asymptotic requires 3 starting values: [a, b, c] for y = a - b*exp(-c*x)",
            ),
        };
        if request.start.len() != param_names.len() {
            return Ok(CallToolResult::error(vec![Content::text(
                usage.to_string(),
            )]));
        }

        let algorithm = match parse_choice(
            &NLS_ALGORITHM,
            request.algorithm.as_deref(),
            NlsAlgorithm::LevenbergMarquardt,
        ) {
            Ok(a) => a,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
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
        description = "Fit a LOESS (LOcally Estimated Scatterplot Smoothing) model. LOESS fits local polynomial regressions at each point, weighted by distance from the target point using tricubic weights. Useful for non-parametric trend estimation, data smoothing, and exploring nonlinear relationships. Returns R-squared, the equivalent number of parameters and the fitted values and residuals (the first 20 with a note unless full = true). Neighbourhoods as R's loess with surface = 'direct'; a tied predictor value shared by more points than the span gives the tie mean with a warning (lowess's rule; R's loess returns 0 there). The JSON lists the warnings."
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
        });
        insert_smoother_values(
            &mut output,
            request.full.unwrap_or(false),
            result.fitted.len(),
            &[
                ("fitted_values", &result.fitted),
                ("residuals", &result.residuals),
            ],
        );
        if let Some(m) = missing_json(&result.missing) {
            output["missing"] = m;
        }
        output["warnings"] = notes_json(None, &result.warnings)["warnings"].clone();

        Ok(CallToolResult::success(vec![Content::text(json_text(
            &output,
        ))]))
    }

    /// Fit Friedman's SuperSmoother.
    #[tool(
        description = "Friedman's SuperSmoother - a variable-bandwidth smoother that locally adapts to the data. Uses cross-validation to select optimal span at each point. Returns the smoothed y values (the first 20 with a note unless full = true) and can handle periodic data. periodic = true needs x in [0, 1], as R."
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
                p2a_core::regression::supsmu::MIN_OBSERVATIONS,
                &match request.weights.as_deref() {
                    Some(w) => format!(
                        "SuperSmoother of '{}' on '{}' with weights '{}'",
                        request.y, request.x, w
                    ),
                    None => format!("SuperSmoother of '{}' on '{}'", request.y, request.x),
                },
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
        });
        insert_smoother_values(
            &mut output,
            request.full.unwrap_or(false),
            result.x.len(),
            &[("x", &result.x), ("y_smoothed", &result.y)],
        );
        if let Some(m) = missing_json(cols.missing()) {
            output["missing"] = m;
        }

        Ok(CallToolResult::success(vec![Content::text(json_text(
            &output,
        ))]))
    }

    /// Fit Tukey's resistant line.
    #[tool(
        description = "Tukey's resistant line - robust regression using medians instead of means, making it resistant to outliers. Divides data into three groups and uses group medians to determine slope and intercept. Returns coefficients, fitted values and residuals (the first 20 with a note unless full = true)."
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
                p2a_core::regression::line::MIN_PAIRS,
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
        });
        insert_smoother_values(
            &mut output,
            request.full.unwrap_or(false),
            result.fitted.len(),
            &[
                ("fitted_values", &result.fitted),
                ("residuals", &result.residuals),
            ],
        );
        if let Some(m) = missing_json(cols.missing()) {
            output["missing"] = m;
        }

        Ok(CallToolResult::success(vec![Content::text(json_text(
            &output,
        ))]))
    }

    /// Stepwise regression model selection.
    #[tool(
        description = "Stepwise regression model selection using AIC or BIC. Can perform forward selection, backward elimination, or both. Returns the best model and selection history. criterion 'aic' (default) or 'bic'; direction 'both' (default), 'forward' or 'backward'. The criterion values are those R's step() prints (extractAIC)."
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
        let use_bic = match parse_choice(
            &STEP_CRITERION,
            request.criterion.as_deref(),
            StepCriterion::Aic,
        ) {
            Ok(c) => c == StepCriterion::Bic,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

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
        description = "Generalized least squares as R nlme::gls. correlation 'ar1' (default, a p2a convention: nlme's default is no correlation; within each group, in row order), 'compound_symmetry' (exchangeable within each group, nlme corCompSymm) or 'identity'. Unless rho is given it is estimated by REML (default) or ML (method), as nlme; group names the grouping column (nlme form = ~ 1 | group) and is required for compound symmetry. Returns JSON with the coefficients, the correlation parameter and whether it was estimated or user-supplied, sigma, log-likelihood, AIC and BIC as nlme, and convergence."
    )]
    pub async fn regression_gls(
        &self,
        Parameters(request): Parameters<GlsRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::regression::{GlsCorrelation, GlsMethod, GlsOptions, run_gls_with};

        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let correlation = match parse_option::<GlsCorrelation>(
            request.correlation.as_deref(),
            GlsCorrelation::Ar1,
        ) {
            Ok(c) => c,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        let method = match parse_option::<GlsMethod>(request.method.as_deref(), GlsMethod::Reml) {
            Ok(m) => m,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();
        let intercept = request.intercept.unwrap_or(true);

        let mut options = GlsOptions::default();
        options.correlation = correlation;
        options.rho = request.rho;
        options.group_col = request.group.clone();
        options.method = method;

        let result = match run_gls_with(dataset, &request.y, &x_refs, intercept, &options) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "GLS regression failed: {e}"
                ))]));
            }
        };

        let mut coef_names: Vec<String> = if intercept {
            vec!["(Intercept)".to_string()]
        } else {
            vec![]
        };
        coef_names.extend(request.x.iter().cloned());

        let coef_table: Vec<serde_json::Value> = coef_names
            .iter()
            .enumerate()
            .take(result.coefficients.len())
            .map(|(i, name)| {
                serde_json::json!({
                    "term": name,
                    "estimate": result.coefficients[i],
                    "std_error": result.std_errors.get(i),
                    "t_value": result.t_values.get(i),
                    "p_value": result.p_values.get(i)
                })
            })
            .collect();

        let source = if result.correlation_param.is_none() {
            "not applicable".to_string()
        } else if !result.rho_estimated {
            "user-supplied".to_string()
        } else {
            format!("estimated ({})", result.method.label())
        };

        let mut json_output = serde_json::json!({
            "model": "Generalized Least Squares",
            "method": result.method.label(),
            "correlation": result.correlation,
            "correlation_param": result.correlation_param,
            "correlation_param_source": source,
            "group": request.group,
            "n_groups": result.n_groups,
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
        let notes = notes_json(
            result
                .convergence
                .as_ref()
                .map(|c| (c.converged, c.iterations)),
            &result.warnings,
        );
        if let (Some(out), Some(n)) = (json_output.as_object_mut(), notes.as_object()) {
            out.extend(n.clone());
        }

        Ok(CallToolResult::success(vec![Content::text(json_text(
            &json_output,
        ))]))
    }

    /// Fit a smoothing spline.
    #[tool(
        description = "Cubic smoothing spline. spar or df fix the smoothing; otherwise lambda is chosen by criterion: 'gcv' (default, R smooth.spline(cv = FALSE)) or 'cv' (leave-one-out, cv = TRUE). Returns JSON with the fitted values (the first 20 with a note unless full = true) or every prediction at xout, df, lambda (on R's scale, x scaled to [0, 1]), the criterion used (null when spar or df was given), its value and warnings."
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

        let mut config = SmoothSplineConfig::default();
        config.spar = request.spar;
        config.df = request.df;
        config.criterion = match parse_option::<p2a_core::regression::smooth_spline::SplineCriterion>(
            request.criterion.as_deref(),
            p2a_core::regression::smooth_spline::SplineCriterion::Gcv,
        ) {
            Ok(c) => c,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
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

                // Predictions at xout are always returned in full (the xout
                // array rule bounds them); fitted values follow `full`.
                let full = request.full.unwrap_or(false) || request.xout.is_some();
                let mut json_output = serde_json::json!({
                    "model": "Smoothing Spline",
                    "spar": result.spar,
                    "df": result.df,
                    "lambda": result.lambda,
                    "cv_crit": result.cv_crit,
                    "criterion": result.criterion.map(|c| c.to_string()),
                    "n_obs": result.n_obs,
                    "n_knots": result.n_knots,
                    "warnings": notes_json(None, &result.warnings)["warnings"].clone()
                });
                insert_smoother_values(
                    &mut json_output,
                    full,
                    pred_x.len(),
                    &[("x", &pred_x), ("fitted", &pred_y)],
                );

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
        description = "Elastic net, lasso or ridge regression as R glmnet::glmnet (coordinate descent, glmnet's standardisation and lambda path). alpha = 1 lasso (default), 0 ridge, in between elastic net; family 'gaussian' (default) or 'binomial'. Returns JSON: the selection summary (order of entry and the estimates at the smallest lambda), the coefficients at the last lambda (at_last_lambda; all of them for up to 20 predictors, otherwise the 20 largest in absolute value and the number of other non-zero ones), and the path with lambda, df and deviance ratio per lambda. path_coefficients = true adds the intercept and coefficients at every lambda of the path."
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

        let path_coefficients = request.path_coefficients.unwrap_or(false);
        match run_glmnet(dataset, &request.y, &x_refs, &config) {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(glmnet_json(
                &result,
                path_coefficients,
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

        let path_coefficients = request.path_coefficients.unwrap_or(false);
        match run_ridge(dataset, &request.y, &x_refs, request.lambda) {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(glmnet_json(
                &result,
                path_coefficients,
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

        let path_coefficients = request.path_coefficients.unwrap_or(false);
        match run_lasso(dataset, &request.y, &x_refs, request.lambda) {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(glmnet_json(
                &result,
                path_coefficients,
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

    /// regression_ols saves its coefficients as a dataset that viz_coefficient plots
    /// with its default columns, replaces only its own earlier table, and never
    /// overwrites a dataset of the user's.
    #[test]
    fn ols_saves_a_coefficient_table_viz_coefficient_plots() {
        let df = p2a_core::polars::prelude::df!(
            "x" => [1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
            "y" => [2.3, 3.8, 6.4, 7.7, 10.2, 11.9]
        )
        .unwrap();
        let user = p2a_core::polars::prelude::df!("a" => [1.0, 2.0]).unwrap();
        let (rt, server) = server_with(vec![("d", df), ("d_ols_x", user)]);
        let ols = |y: &str, x: &str| {
            let args = json!({"dataset": "d", "y": y, "x": [x], "cov_type": "standard"});
            let r = rt
                .block_on(server.regression_ols(Parameters(serde_json::from_value(args).unwrap())))
                .unwrap();
            assert_ne!(r.is_error, Some(true), "{}", text(&r));
            text(&r)
        };

        let reply = ols("y", "x");
        assert!(
            reply.starts_with("Coefficients saved as dataset 'd_ols_y'"),
            "{reply}"
        );
        let table = rt.block_on(async { server.datasets.read().await.get("d_ols_y").cloned() });
        let table = table.expect("saved table");
        assert!(is_coefficient_table(&table));
        let df = table.df();
        let terms: Vec<&str> = df
            .column("term")
            .unwrap()
            .str()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(terms, ["(Intercept)", "x"]);
        let fit = run_ols(
            rt.block_on(async { server.datasets.read().await.get("d").cloned() })
                .as_ref()
                .unwrap(),
            "y",
            &["x"],
            true,
            CovarianceType::Standard,
        )
        .unwrap();
        for (col, want) in [
            ("estimate", fit.coefficients[1].estimate),
            ("ci_lower", fit.coefficients[1].ci_lower_95),
            ("ci_upper", fit.coefficients[1].ci_upper_95),
        ] {
            let got = df.column(col).unwrap().f64().unwrap().get(1).unwrap();
            assert_eq!(got, want, "{col}");
        }

        // viz_coefficient needs only the dataset
        let args = json!({"dataset": "d_ols_y"});
        let r = rt
            .block_on(server.viz_coefficient(Parameters(serde_json::from_value(args).unwrap())))
            .unwrap();
        assert!(text(&r).contains("Coefficients: 2"), "{}", text(&r));

        // a rerun replaces its own table; a user's dataset of the same name is kept
        assert!(ols("y", "x").starts_with("Coefficients saved as dataset 'd_ols_y'"));
        assert!(ols("x", "y").starts_with("Coefficients saved as dataset 'd_ols_x_2'"));
        let kept = rt.block_on(async { server.datasets.read().await.get("d_ols_x").cloned() });
        assert_eq!(kept.unwrap().df().get_column_names()[0].as_str(), "a");

        // the raw data are not a coefficient table, and the error says what is
        let args = json!({"dataset": "d"});
        let r = rt
            .block_on(server.viz_coefficient(Parameters(serde_json::from_value(args).unwrap())))
            .unwrap();
        assert_eq!(r.is_error, Some(true));
        assert!(text(&r).contains("regression_ols"), "{}", text(&r));
    }

    /// The four smoothers return their vectors by one rule: the first 20 under
    /// `<key>_sample` with a note by default, every value with `full`.
    #[test]
    fn test_smoothers_share_one_output_policy() {
        let n = 56usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.25).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| v.sin() + 0.1 * ((i * 7 % 11) as f64 - 5.0) / 5.0)
            .collect();
        let df = p2a_core::polars::prelude::df!("x" => x, "y" => y).unwrap();
        let (rt, server) = server_with(vec![("d", df)]);
        let run = |tool: &str, extra: Value| -> Value {
            let mut args = json!({"dataset": "d", "x": "x", "y": "y"});
            for (k, v) in extra.as_object().unwrap() {
                args[k] = v.clone();
            }
            let r = rt
                .block_on(async {
                    match tool {
                        "loess" => {
                            server
                                .regression_loess(Parameters(serde_json::from_value(args).unwrap()))
                                .await
                        }
                        "supsmu" => {
                            server
                                .regression_supsmu(Parameters(
                                    serde_json::from_value(args).unwrap(),
                                ))
                                .await
                        }
                        "line" => {
                            server
                                .regression_line(Parameters(serde_json::from_value(args).unwrap()))
                                .await
                        }
                        _ => {
                            server
                                .regression_smooth_spline(Parameters(
                                    serde_json::from_value(args).unwrap(),
                                ))
                                .await
                        }
                    }
                })
                .unwrap();
            assert_ne!(r.is_error, Some(true), "{tool}: {}", text(&r));
            json_of(&r)
        };
        let note = "first 20 of 56 values; set full = true for all";
        for (tool, keys) in [
            ("loess", &["fitted_values", "residuals"][..]),
            ("supsmu", &["x", "y_smoothed"][..]),
            ("line", &["fitted_values", "residuals"][..]),
            ("spline", &["x", "fitted"][..]),
        ] {
            let sampled = run(tool, json!({}));
            assert_eq!(sampled["note"], note, "{tool}: {sampled}");
            let full = run(tool, json!({"full": true}));
            assert!(full.get("note").is_none(), "{tool}: {full}");
            for key in keys {
                let sample = sampled[format!("{key}_sample")].as_array().unwrap();
                assert_eq!(sample.len(), 20, "{tool} {key}");
                assert!(sampled.get(*key).is_none(), "{tool} {key}");
                let all = full[*key].as_array().unwrap();
                assert_eq!(all.len(), 56, "{tool} {key}");
                assert_eq!(&all[..20], &sample[..], "{tool} {key}");
            }
        }
        // Predictions at xout are always returned in full.
        let xout: Vec<f64> = (0..30).map(|i| i as f64 * 0.4).collect();
        let j = run("spline", json!({"xout": xout}));
        assert_eq!(j["fitted"].as_array().unwrap().len(), 30, "{j}");
        assert!(j.get("note").is_none(), "{j}");
    }

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

    /// When the weights column removes the rows, the row-count error names it.
    #[test]
    fn test_mcp_regression_supsmu_error_names_weights() {
        use p2a_core::polars::prelude::*;
        let df = df! {
            "x" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y" => [1.1, 1.9, 3.2, 3.8, 5.1, 6.2],
            "w" => [Some(1.0), None, None, None, Some(1.0), Some(1.0)],
        }
        .unwrap();
        let (rt, server) = server_with(vec![("sw", df)]);
        let req: SupsmuRequest =
            serde_json::from_value(json!({"dataset": "sw", "x": "x", "y": "y", "weights": "w"}))
                .unwrap();
        let r = rt
            .block_on(server.regression_supsmu(Parameters(req)))
            .unwrap();
        let t = text(&r);
        assert_eq!(r.is_error, Some(true), "{t}");
        assert!(
            t.contains(
                "SuperSmoother of 'y' on 'x' with weights 'w': 3 of 6 rows have no missing value"
            ),
            "{t}"
        );
    }

    #[test]
    fn test_regression_option_tables_validate() {
        assert_eq!(super::STEP_CRITERION.validate(), Ok(()));
        assert_eq!(super::NLS_MODEL.validate(), Ok(()));
        assert_eq!(super::NLS_ALGORITHM.validate(), Ok(()));
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
            let req = request(extra);
            let r = run_ols_request(&ds, &req, ols_options(&req).unwrap()).unwrap();
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

    /// `regression_ols` with `weights` is weighted least squares (case `wls_sim`,
    /// `lm(y ~ x1 + x2 + d, weights = w)`): the printed coefficients and HC1 SEs, the
    /// weighted title and the note on the rows with weight 0.
    #[test]
    fn test_validate_mcp_regression_ols_weights() {
        let (case, rt, server) = case_server("wls_sim");
        let r = call(
            &rt,
            &server,
            json!({"y": "y", "x": ["x1", "x2", "d"], "weights": "w"}),
        );
        assert_ne!(r.is_error, Some(true));
        let t = text(&r);
        assert!(
            t.starts_with("Weighted Least Squares Results")
                || t.contains("\nWeighted Least Squares Results"),
            "{t}"
        );
        assert!(
            t.contains("Weights: w") && t.contains("51 row(s) with weight 0"),
            "{t}"
        );
        let (names, values): (Vec<String>, Vec<Vec<f64>>) = rows(&t).into_iter().unzip();
        let coef: Vec<f64> = values.iter().map(|v| v[0]).collect();
        let se: Vec<f64> = values.iter().map(|v| v[1]).collect();
        Check::new(&case)
            .named("coef", &names, &coef, Tol::printed(4))
            .named("se_hc1", &names, &se, Tol::printed(4))
            .finish();
        let r = call(&rt, &server, json!({"weights": "nope"}));
        assert_eq!(r.is_error, Some(true));
    }

    /// Missing cells drop their rows, as lm's na.omit, and the reply ends with the drop.
    #[test]
    fn test_validate_mcp_regression_ols_na() {
        let (case, rt, server) = case_server("ols_na");
        let r = call(&rt, &server, json!({"y": "y", "x": ["x1", "x2"]}));
        assert_ne!(r.is_error, Some(true), "{}", text(&r));
        let t = text(&r);
        assert!(
            t.trim_end().ends_with(
                "Missing values: 8 of 120 rows dropped because of missing values (y: 5, x1: 4)"
            ),
            "{t}"
        );
        assert!(
            t.starts_with("Coefficients saved as dataset 'grunfeld_ols_y'"),
            "{t}"
        );
        let (names, values): (Vec<String>, Vec<Vec<f64>>) = rows(&t).into_iter().unzip();
        let coef: Vec<f64> = values.iter().map(|v| v[0]).collect();
        let se: Vec<f64> = values.iter().map(|v| v[1]).collect();
        Check::new(&case)
            .named("coef", &names, &coef, Tol::printed(4))
            .named("se_hc1", &names, &se, Tol::printed(4))
            .finish();
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
        assert!(
            t.contains("\nR-squared (uncentered): ")
                && t.contains("\nAdj. R-squared (uncentered): "),
            "{t}"
        );
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

    /// A missing response or regressor no longer stops the tools built on an lm fit: HAC and
    /// the Breusch-Godfrey test run on the complete rows, match R there, and say which rows
    /// were dropped.
    #[test]
    fn test_validate_mcp_regression_lm_tests_na() {
        let case = Case::load("reg_lm_tests_na");
        let (rt, server) = cases_server(&["reg_lm_tests_na"]);
        let note = "Missing values: 5 of 200 rows dropped because of missing values \
                    (inv: 3, value: 2); the other rows are used in their order";
        let req: HacRequest = serde_json::from_value(json!({"dataset": "grunfeld_lm_na",
            "y": "inv", "x": ["value", "capital"], "bandwidth": 3}))
        .unwrap();
        let r = rt.block_on(server.regression_hac(Parameters(req))).unwrap();
        let t = text(&r);
        assert_ne!(r.is_error, Some(true), "{t}");
        assert!(t.contains(note), "{t}");
        let (mut names, mut coef, mut se) = (Vec::new(), Vec::new(), Vec::new());
        for l in t.lines().skip_while(|l| !l.starts_with("---")).skip(1) {
            let tok: Vec<&str> = l.split_whitespace().collect();
            if tok.len() != 3 {
                break;
            }
            names.push(tok[0].to_string());
            coef.push(tok[1].parse::<f64>().unwrap());
            se.push(tok[2].parse::<f64>().unwrap());
        }
        Check::new(&case)
            .named("coef", &names, &coef, Tol::printed(6))
            .named("se_nw_lag3", &names, &se, Tol::printed(6))
            .finish();
        let req: BgTestRequest = serde_json::from_value(json!({"dataset": "grunfeld_lm_na",
            "y": "inv", "x": ["value", "capital"], "order": 2}))
        .unwrap();
        let r = rt
            .block_on(server.regression_bgtest(Parameters(req)))
            .unwrap();
        let t = text(&r);
        assert_ne!(r.is_error, Some(true), "{t}");
        assert!(t.contains(note), "{t}");
        let stat: f64 = t
            .lines()
            .find_map(|l| l.strip_prefix("LM Statistic: "))
            .unwrap()
            .parse()
            .unwrap();
        Check::new(&case)
            .scalar("bg_stat", stat, Tol::printed(4))
            .finish();
    }

    /// REG-21 (Driscoll-Kraay): a missing time id drops its row; a fractional time id
    /// is a tool error instead of being truncated to a period. A property test of the
    /// estimation sample (no R case holds the missing-id data), so it loads the dataset
    /// without a case.
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
        let r = call(json!({"dataset": "smoothers_na", "y": "y", "x": "x", "full": true}));
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
