//! Time series analysis commands

use clap::Subcommand;
use p2a_core::data::columns::{NaPolicy, numeric_column};
use p2a_core::econometrics::granger_test;
use p2a_core::forecasting::{ArimaSpec, forecast_arima_with, run_arima_with};
use p2a_core::{
    ArConfig,
    ArMethod,
    CostFunction,
    DecomposeType,
    SeasonalType,
    StructTsType,
    // AR model
    ar,
    detect_changepoints,
    // Causal impact
    run_causal_impact,
    // Classical decomposition
    run_decompose,
    run_garch,
    run_holt_winters,
    run_mstl,
    // STL decomposition
    run_stl,
    // Structural time series
    run_struct_ts,
    run_var,
    run_var_irf,
    run_varma,
    run_vecm,
};
#[cfg(feature = "spectral-analysis")]
use p2a_core::{SpectrumConfig, run_spectrum};

use p2a_core::options::Choices;

use super::causal::{print_result_notes, with_result_notes};
use crate::output::{OutputFormat, print_error};
use crate::session::SessionManager;

// The R names of the time-series options. The core enums have no FromStr, so the tables live
// here; the first spelling of each entry is canonical.

/// `--change-type`, as the changepoint package's cpt.mean / cpt.var / cpt.meanvar.
pub(crate) static CHANGE_TYPES: Choices<CostFunction> = Choices {
    option: "change type",
    values: &[
        (&["mean"], CostFunction::MeanChange),
        (&["variance", "var"], CostFunction::VarianceChange),
        (
            &["meanvar", "both", "mean_and_variance", "mean_var"],
            CostFunction::MeanAndVariance,
        ),
    ],
};

/// `--seasonal`, as HoltWinters(seasonal = ).
pub(crate) static SEASONAL_TYPES: Choices<SeasonalType> = Choices {
    option: "seasonal type",
    values: &[
        (&["additive", "add"], SeasonalType::Additive),
        (
            &["multiplicative", "mult", "multi"],
            SeasonalType::Multiplicative,
        ),
    ],
};

/// `--method` of `ts ar`, as ar(method = ).
pub(crate) static AR_METHODS: Choices<ArMethod> = Choices {
    option: "AR method",
    values: &[
        (&["yule-walker", "yw"], ArMethod::YuleWalker),
        (&["burg"], ArMethod::Burg),
        (&["ols"], ArMethod::Ols),
    ],
};

/// `--decompose-type`, as decompose(type = ).
pub(crate) static DECOMPOSE_TYPES: Choices<DecomposeType> = Choices {
    option: "decomposition type",
    values: &[
        (&["additive", "add"], DecomposeType::Additive),
        (
            &["multiplicative", "mult", "multi"],
            DecomposeType::Multiplicative,
        ),
    ],
};

/// `--model-type`, as StructTS(type = ).
pub(crate) static STRUCT_TS_TYPES: Choices<StructTsType> = Choices {
    option: "StructTS type",
    values: &[
        (&["level", "local_level"], StructTsType::Level),
        (&["trend", "local_trend", "llt"], StructTsType::Trend),
        (&["BSM", "basic_structural"], StructTsType::BSM),
    ],
};

fn table_arg<T: Copy + PartialEq>(table: &Choices<T>, s: &str) -> Result<T, String> {
    table.parse(s).map_err(|e| match e {
        p2a_core::EconError::InvalidSpecification { message } => message,
        other => other.to_string(),
    })
}

fn parse_change_type(s: &str) -> Result<CostFunction, String> {
    table_arg(&CHANGE_TYPES, s)
}

fn parse_seasonal(s: &str) -> Result<SeasonalType, String> {
    table_arg(&SEASONAL_TYPES, s)
}

fn parse_ar_method(s: &str) -> Result<ArMethod, String> {
    table_arg(&AR_METHODS, s)
}

fn parse_decompose_type(s: &str) -> Result<DecomposeType, String> {
    table_arg(&DECOMPOSE_TYPES, s)
}

fn parse_struct_ts_type(s: &str) -> Result<StructTsType, String> {
    table_arg(&STRUCT_TS_TYPES, s)
}

#[derive(Subcommand)]
pub enum TimeseriesCommands {
    /// ARIMA model estimation and forecasting
    #[command(after_help = "\
EXAMPLES:
    # Fit ARIMA(1,1,1)
    p2a ts arima mydata --col sales -p 1 -d 1 -q 1

    # With 12-step forecast (standard errors and 80/95% intervals)
    p2a ts arima mydata --col sales -p 1 -d 1 -q 1 --horizon 12

    # Random walk with drift and an MA(1) term
    p2a ts arima mydata --col sales -p 0 -d 1 -q 1 --drift --horizon 12
")]
    Arima {
        /// Dataset name
        dataset: String,

        /// Time series column
        #[arg(long)]
        col: String,

        /// Autoregressive order (p)
        #[arg(short = 'p', long, default_value = "1")]
        ar: usize,

        /// Differencing order (d)
        #[arg(short = 'd', long, default_value = "0")]
        diff: usize,

        /// Moving average order (q)
        #[arg(short = 'q', long, default_value = "1")]
        ma: usize,

        /// Forecast horizon (optional)
        #[arg(long)]
        horizon: Option<usize>,

        /// Estimate the mean (default: true when d = 0, as R); --include-mean false drops it
        #[arg(long, value_name = "BOOL")]
        include_mean: Option<bool>,

        /// Fit a drift (linear trend in the levels), as forecast::Arima(include.drift = TRUE); d = 0 or 1
        #[arg(long)]
        drift: bool,
    },

    /// MSTL seasonal decomposition
    Mstl {
        /// Dataset name
        dataset: String,

        /// Time series column
        #[arg(long)]
        col: String,

        /// Seasonal periods (e.g., 7 for weekly, 365 for yearly)
        #[arg(long, num_args = 1..)]
        periods: Vec<usize>,
    },

    /// Vector Autoregression
    #[command(after_help = "\
EXAMPLES:
    # VAR(2) with two variables
    p2a ts var mydata --cols gdp inflation --lags 2
")]
    Var {
        /// Dataset name
        dataset: String,

        /// Variable columns
        #[arg(long, num_args = 2..)]
        cols: Vec<String>,

        /// Number of lags
        #[arg(long, default_value = "1")]
        lags: usize,
    },

    /// Vector ARMA model
    Varma {
        /// Dataset name
        dataset: String,

        /// Variable columns
        #[arg(long, num_args = 2..)]
        cols: Vec<String>,

        /// AR order (p)
        #[arg(short = 'p', long, default_value = "1")]
        ar: usize,

        /// MA order (q)
        #[arg(short = 'q', long, default_value = "1")]
        ma: usize,
    },

    /// Vector Error Correction Model (cointegration)
    Vecm {
        /// Dataset name
        dataset: String,

        /// Variable columns
        #[arg(long, num_args = 2..)]
        cols: Vec<String>,

        /// Lag order of the VAR in levels (urca::ca.jo's K, at least 2)
        #[arg(long, default_value = "2")]
        lags: usize,

        /// Cointegration rank
        #[arg(long, default_value = "1")]
        rank: usize,
    },

    /// Impulse Response Function from VAR model
    Irf {
        /// Dataset name
        dataset: String,

        /// Variable columns
        #[arg(long, num_args = 2..)]
        cols: Vec<String>,

        /// Number of lags for VAR
        #[arg(long, default_value = "1")]
        lags: usize,

        /// Number of horizons including the impact h = 0 (vars::irf n.ahead = steps - 1)
        #[arg(long, default_value = "10")]
        steps: usize,
    },

    /// Changepoint detection (PELT algorithm)
    Changepoint {
        /// Dataset name
        dataset: String,

        /// Time series column
        #[arg(long)]
        col: String,

        /// Penalty; default changepoint's MBIC: (k + 2) ln n plus ln(segment length) per segment
        /// (k = 1 for mean or variance, 2 for meanvar)
        #[arg(long)]
        penalty: Option<f64>,

        /// Minimum segment length (default: 2)
        #[arg(long)]
        min_segment: Option<usize>,

        /// Type of change, as the changepoint package: mean (cpt.mean, unit variance; default),
        /// variance (cpt.var) or meanvar (cpt.meanvar; alias both)
        #[arg(long, default_value = "mean", value_parser = parse_change_type)]
        change_type: CostFunction,
    },

    /// GARCH volatility model
    #[command(after_help = "\
EXAMPLES:
    # GARCH(1,1) for returns
    p2a ts garch mydata --col returns -p 1 -q 1

    # With volatility forecast
    p2a ts garch mydata --col returns -p 1 -q 1 --horizon 5
")]
    Garch {
        /// Dataset name
        dataset: String,

        /// Time series column
        #[arg(long)]
        col: String,

        /// ARCH order (p) - number of lagged squared residuals
        #[arg(short = 'p', long, default_value = "1")]
        arch: usize,

        /// GARCH order (q) - number of lagged variances
        #[arg(short = 'q', long, default_value = "1")]
        garch: usize,

        /// Forecast horizon (optional)
        #[arg(long)]
        horizon: Option<usize>,
    },

    /// Holt-Winters exponential smoothing
    #[command(after_help = "\
EXAMPLES:
    # Additive seasonal (monthly data)
    p2a ts holt-winters mydata --col sales --period 12

    # Multiplicative seasonal
    p2a ts holt-winters mydata --col sales --period 12 --seasonal multiplicative

    # Fixed smoothing parameters and 12 forecasts with 95% prediction intervals
    p2a ts holt-winters mydata --col sales --period 12 --alpha 0.3 --beta 0 --gamma 0.2 --horizon 12
")]
    HoltWinters {
        /// Dataset name
        dataset: String,

        /// Time series column
        #[arg(long)]
        col: String,

        /// Seasonal period (e.g., 12 for monthly, 4 for quarterly)
        #[arg(long)]
        period: usize,

        /// Seasonal component, as HoltWinters: additive (default) or multiplicative (alias mult)
        #[arg(long, default_value = "additive", value_parser = parse_seasonal)]
        seasonal: SeasonalType,

        /// Level smoothing alpha in (0, 1]; default: optimised
        #[arg(long)]
        alpha: Option<f64>,

        /// Trend smoothing beta in [0, 1]; 0 fits no trend (R beta = FALSE); default: optimised
        #[arg(long)]
        beta: Option<f64>,

        /// Seasonal smoothing gamma in [0, 1]; 0 fits no seasonal component (R gamma = FALSE);
        /// default: optimised
        #[arg(long)]
        gamma: Option<f64>,

        /// Forecast h steps ahead with prediction intervals (predict.HoltWinters); a
        /// multiplicative seasonal fit gets point forecasts only
        #[arg(long)]
        horizon: Option<usize>,

        /// Level of the prediction intervals (predict.HoltWinters level)
        #[arg(long, default_value_t = 0.95)]
        level: f64,
    },

    /// Granger causality test
    #[command(after_help = "\
EXAMPLES:
    # Test if x Granger-causes y
    p2a ts granger mydata --x money_supply --y inflation --lags 4
")]
    Granger {
        /// Dataset name
        dataset: String,

        /// First variable (potential cause)
        #[arg(long)]
        x: String,

        /// Second variable (potential effect)
        #[arg(long)]
        y: String,

        /// Number of lags
        #[arg(long, default_value = "1")]
        lags: usize,
    },

    /// Autoregressive (AR) model fitting
    Ar {
        /// Dataset name
        dataset: String,

        /// Time series column
        #[arg(long)]
        col: String,

        /// Fit exactly this order (AIC selection is then not used)
        #[arg(long)]
        order: Option<usize>,

        /// Method, as ar(): yule-walker (default; alias yw), burg or ols
        #[arg(long, default_value = "yule-walker", value_parser = parse_ar_method)]
        method: ArMethod,

        /// Select the order by AIC up to min(n - 1, floor(10 log10 n)) (true/false; default true);
        /// false fits that maximum order, as R's ar(aic = FALSE). --order fixes the order and
        /// overrides both
        #[arg(
            long,
            value_name = "BOOL",
            num_args = 0..=1,
            default_value_t = true,
            default_missing_value = "true",
            action = clap::ArgAction::Set,
            value_parser = clap::builder::BoolishValueParser::new()
        )]
        aic: bool,
    },

    /// STL decomposition (Seasonal-Trend using LOESS)
    #[command(after_help = "\
EXAMPLES:
    # STL for monthly data
    p2a ts stl mydata --col sales --period 12

    # With robust fitting
    p2a ts stl mydata --col sales --period 12 --robust true
")]
    Stl {
        /// Dataset name
        dataset: String,

        /// Time series column
        #[arg(long)]
        col: String,

        /// Seasonal period (e.g., 12 for monthly, 4 for quarterly)
        #[arg(long)]
        period: usize,

        /// Robust (outlier-resistant) fitting (true/false; default false)
        #[arg(
            long,
            value_name = "BOOL",
            num_args = 0..=1,
            default_value_t = false,
            default_missing_value = "true",
            action = clap::ArgAction::Set,
            value_parser = clap::builder::BoolishValueParser::new()
        )]
        robust: bool,
    },

    /// Classical seasonal decomposition (moving averages)
    Decompose {
        /// Dataset name
        dataset: String,

        /// Time series column
        #[arg(long)]
        col: String,

        /// Seasonal period
        #[arg(long)]
        period: usize,

        /// Decomposition, as decompose(): additive (default) or multiplicative
        #[arg(long, default_value = "additive", value_parser = parse_decompose_type)]
        decompose_type: DecomposeType,
    },

    /// Structural time series (state space) model
    StructTs {
        /// Dataset name
        dataset: String,

        /// Time series column
        #[arg(long)]
        col: String,

        /// Model, as StructTS: level (default), trend or BSM (needs --period)
        #[arg(long, default_value = "level", value_parser = parse_struct_ts_type)]
        model_type: StructTsType,

        /// Seasonal period (required for "bsm" model type)
        #[arg(long)]
        period: Option<usize>,
    },

    /// Bayesian structural time series causal inference (CausalImpact)
    #[command(after_help = "\
EXAMPLES:
    # Estimate causal impact of intervention
    p2a ts causal-impact mydata --response sales --time date \\
        --pre-start 1 --pre-end 100 --post-start 101 --post-end 150

    # With control series
    p2a ts causal-impact mydata --response sales --time date \\
        --pre-start 1 --pre-end 100 --post-start 101 --post-end 150 \\
        --controls competitor_sales,market_index
")]
    CausalImpact {
        /// Dataset name
        dataset: String,

        /// Response variable column
        #[arg(long)]
        response: String,

        /// Time column
        #[arg(long)]
        time: String,

        /// Pre-intervention period start (inclusive)
        #[arg(long)]
        pre_start: i64,

        /// Pre-intervention period end (inclusive)
        #[arg(long)]
        pre_end: i64,

        /// Post-intervention period start (inclusive)
        #[arg(long)]
        post_start: i64,

        /// Post-intervention period end (inclusive)
        #[arg(long)]
        post_end: i64,

        /// Control series columns (comma-separated)
        #[arg(long)]
        controls: Option<String>,
    },

    /// Spectral density estimation (periodogram)
    #[cfg(feature = "spectral-analysis")]
    Spectrum {
        /// Dataset name
        dataset: String,

        /// Time series column
        #[arg(long)]
        col: String,

        /// Modified Daniell smoothing spans, comma-separated odd integers, e.g. 3,3
        #[arg(long, value_delimiter = ',', num_args = 1..)]
        spans: Vec<usize>,

        /// Taper proportion (0.0 to 0.5, default: 0.1)
        #[arg(long, default_value = "0.1")]
        taper: f64,

        /// Remove a least-squares linear trend first (true/false; default true, as spec.pgram)
        #[arg(
            long,
            value_name = "BOOL",
            num_args = 0..=1,
            default_value_t = true,
            default_missing_value = "true",
            action = clap::ArgAction::Set,
            value_parser = clap::builder::BoolishValueParser::new()
        )]
        detrend: bool,
    },
}

pub fn execute(
    cmd: &TimeseriesCommands,
    format: &OutputFormat,
    _quiet: bool,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    match cmd {
        TimeseriesCommands::Arima {
            dataset,
            col,
            ar,
            diff,
            ma,
            horizon,
            include_mean,
            drift,
        } => {
            let spec = ArimaSpec {
                p: *ar,
                d: *diff,
                q: *ma,
                include_mean: *include_mean,
                include_drift: *drift,
            };
            execute_arima(dataset, col, &spec, *horizon, format, session)
        }
        TimeseriesCommands::Mstl {
            dataset,
            col,
            periods,
        } => execute_mstl(dataset, col, periods, format, session),
        TimeseriesCommands::Var {
            dataset,
            cols,
            lags,
        } => execute_var(dataset, cols, *lags, format, session),
        TimeseriesCommands::Varma {
            dataset,
            cols,
            ar,
            ma,
        } => execute_varma(dataset, cols, *ar, *ma, format, session),
        TimeseriesCommands::Vecm {
            dataset,
            cols,
            lags,
            rank,
        } => execute_vecm(dataset, cols, *lags, *rank, format, session),
        TimeseriesCommands::Irf {
            dataset,
            cols,
            lags,
            steps,
        } => execute_irf(dataset, cols, *lags, *steps, format, session),
        TimeseriesCommands::Changepoint {
            dataset,
            col,
            penalty,
            min_segment,
            change_type,
        } => execute_changepoint(
            dataset,
            col,
            *penalty,
            *min_segment,
            *change_type,
            format,
            session,
        ),
        TimeseriesCommands::Garch {
            dataset,
            col,
            arch,
            garch,
            horizon,
        } => execute_garch(dataset, col, *arch, *garch, *horizon, format, session),
        TimeseriesCommands::HoltWinters {
            dataset,
            col,
            period,
            seasonal,
            alpha,
            beta,
            gamma,
            horizon,
            level,
        } => execute_holt_winters(
            dataset,
            col,
            *period,
            *seasonal,
            (*alpha, *beta, *gamma),
            horizon.map(|h| (h, *level)),
            format,
            session,
        ),
        TimeseriesCommands::Granger {
            dataset,
            x,
            y,
            lags,
        } => execute_granger(dataset, x, y, *lags, format, session),
        TimeseriesCommands::Ar {
            dataset,
            col,
            order,
            method,
            aic,
        } => execute_ar(dataset, col, *order, *method, *aic, format, session),
        TimeseriesCommands::Stl {
            dataset,
            col,
            period,
            robust,
        } => execute_stl(dataset, col, *period, *robust, format, session),
        TimeseriesCommands::Decompose {
            dataset,
            col,
            period,
            decompose_type,
        } => execute_decompose(dataset, col, *period, *decompose_type, format, session),
        TimeseriesCommands::StructTs {
            dataset,
            col,
            model_type,
            period,
        } => execute_struct_ts(dataset, col, *model_type, *period, format, session),
        TimeseriesCommands::CausalImpact {
            dataset,
            response,
            time,
            pre_start,
            pre_end,
            post_start,
            post_end,
            controls,
        } => execute_causal_impact(
            dataset,
            response,
            time,
            *pre_start,
            *pre_end,
            *post_start,
            *post_end,
            controls.as_deref(),
            format,
            session,
        ),
        #[cfg(feature = "spectral-analysis")]
        TimeseriesCommands::Spectrum {
            dataset,
            col,
            spans,
            taper,
            detrend,
        } => execute_spectrum(dataset, col, spans, *taper, *detrend, format, session),
    }
}

fn execute_arima(
    dataset_name: &str,
    col: &str,
    spec: &ArimaSpec,
    horizon: Option<usize>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error(
                "No session active. Use --session <file> to enable dataset storage.",
                format,
            );
            return Ok(());
        }
    };
    let Some(ds) = dataset else {
        print_error(&format!("Dataset '{}' not found", dataset_name), format);
        return Ok(());
    };

    // One fit: with a horizon the forecast carries the fit it comes from.
    let (fit, forecast) = match horizon {
        Some(h) => match forecast_arima_with(ds, col, spec, h) {
            Ok(mut fc) => match fc.fit.take() {
                Some(fit) => (fit, Some(fc)),
                None => {
                    print_error("ARIMA failed: the forecast carries no fit", format);
                    return Ok(());
                }
            },
            Err(e) => {
                print_error(&ts_error_message("ARIMA failed", &e), format);
                return Ok(());
            }
        },
        None => match run_arima_with(ds, col, spec) {
            Ok(fit) => (fit, None),
            Err(e) => {
                print_error(&ts_error_message("ARIMA failed", &e), format);
                return Ok(());
            }
        },
    };

    match format {
        OutputFormat::Json => {
            let coefficients: serde_json::Map<String, serde_json::Value> = fit
                .coef_names
                .iter()
                .zip(&fit.coefficients)
                .map(|(name, value)| (name.clone(), serde_json::json!(value)))
                .collect();
            let mut json = serde_json::json!({
                // "model" keeps its earlier value; forecast::Arima's label is "label".
                "model": format!("ARIMA({},{},{})", spec.p, spec.d, spec.q),
                "label": spec.label(),
                "method": fit.method,
                "column": fit.column,
                "coefficients": coefficients,
                "ar_coeffs": fit.ar_coeffs,
                "ma_coeffs": fit.ma_coeffs,
                "intercept": fit.intercept,
                "constant": fit.constant,
                "drift": fit.drift,
                "include_mean": fit.include_mean,
                "sigma2": fit.sigma2,
                "loglik": fit.loglik,
                "aic": fit.aic,
                "ssr": fit.ssr,
                "n_obs": fit.n_obs,
                "n_cond": fit.n_cond,
                "converged": fit.converged,
                "iterations": fit.iterations,
                "warnings": fit.warnings,
            });
            if let Some(fc) = &forecast {
                let limit =
                    |rows: &Vec<Vec<f64>>, i: usize| rows.get(i).cloned().unwrap_or_default();
                json["forecast"] = serde_json::json!({
                    "horizon": fc.horizon,
                    "mean": fc.forecast,
                    "std_errors": fc.std_errors,
                    "levels": fc.levels,
                    "lower_80": limit(&fc.lower, 0),
                    "upper_80": limit(&fc.upper, 0),
                    "lower_95": limit(&fc.lower, 1),
                    "upper_95": limit(&fc.upper, 1),
                    "warnings": fc.warnings,
                });
            }
            println!("{}", serde_json::to_string_pretty(&json)?);
        }
        _ => {
            println!("\n{fit}");
            if let Some(mut fc) = forecast {
                fc.fit = Some(fit);
                println!("\n{fc}");
            }
        }
    }
    Ok(())
}

/// `prefix: err`, plus how to fill the gap when the error is about missing values. The hint
/// narrows the generic "Drop or impute them first." of the missing-value error to imputing.
fn ts_error_message(prefix: &str, err: &dyn std::fmt::Display) -> String {
    let message = err.to_string();
    let mut text = format!("{prefix}: {message}");
    if message.contains("missing value") {
        text.push_str(
            " For a time series, impute rather than drop: fill the gaps first (for example: p2a \
             munge fill-na <dataset> --method forward --columns <column>); dropping a value \
             would splice the series.",
        );
    }
    text
}

fn execute_mstl(
    dataset_name: &str,
    col: &str,
    periods: &[usize],
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error(
                "No session active. Use --session <file> to enable dataset storage.",
                format,
            );
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => match run_mstl(ds, col, periods) {
            Ok(result) => match format {
                OutputFormat::Json => {
                    let json = serde_json::json!({
                        "trend": result.trend,
                        "seasonal": result.seasonal,
                        "residuals": result.residuals,
                        "n_obs": result.n_obs,
                    });
                    println!("{}", serde_json::to_string_pretty(&json)?);
                }
                _ => {
                    println!("\nMSTL Decomposition Results");
                    println!("{}", "=".repeat(50));
                    println!("Periods: {:?}", periods);
                    println!("Observations: {}", result.n_obs);
                    println!(
                        "\nTrend (first 10 values): {:?}",
                        result.trend.iter().take(10).collect::<Vec<_>>()
                    );
                    println!("Number of seasonal components: {}", result.seasonal.len());
                    println!(
                        "Residuals (first 10 values): {:?}",
                        result.residuals.iter().take(10).collect::<Vec<_>>()
                    );
                }
            },
            Err(e) => {
                print_error(&ts_error_message("MSTL failed", &e), format);
            }
        },
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

fn execute_var(
    dataset_name: &str,
    cols: &[String],
    lags: usize,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error(
                "No session active. Use --session <file> to enable dataset storage.",
                format,
            );
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();

            match run_var(ds, &col_refs, lags) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": "VAR",
                            "lags": lags,
                            "variables": result.var_names,
                            "n_vars": result.n_vars,
                            "n_obs": result.n_obs,
                            "coefficients": result.coefficients,
                            "sigma_u": result.sigma_u,
                            "aic": result.aic,
                            "bic": result.bic,
                            "log_likelihood": result.log_likelihood,
                            "coef_names": result.coef_names,
                            "std_errors": result.std_errors,
                            "t_values": result.t_values,
                            "p_values": result.p_values,
                            "df_residual": result.df_residual,
                            "sigma_u_ml": result.sigma_u_ml,
                            "log_det_sigma_ml": result.log_det_sigma_ml,
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => println!("\n{result}"),
                },
                Err(e) => {
                    print_error(&ts_error_message("VAR failed", &e), format);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

fn execute_varma(
    dataset_name: &str,
    cols: &[String],
    ar: usize,
    ma: usize,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();

            match run_varma(ds, &col_refs, ar, ma) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": format!("VARMA({},{})", ar, ma),
                            "p_lags": result.p_lags,
                            "q_lags": result.q_lags,
                            "n_vars": result.n_vars,
                            "n_obs": result.n_obs,
                            "ar_params": result.ar_params,
                            "ma_params": result.ma_params,
                            "sigma_u": result.sigma_u,
                            "aic": result.aic,
                            "bic": result.bic,
                            "log_likelihood": result.log_likelihood,
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nVARMA({},{}) Results", ar, ma);
                        println!("{}", "=".repeat(50));
                        println!("Number of variables: {}", result.n_vars);
                        println!("Observations: {}", result.n_obs);
                        println!("\nAIC: {:.4}", result.aic);
                        println!("BIC: {:.4}", result.bic);
                        println!("Log-likelihood: {:.4}", result.log_likelihood);
                    }
                },
                Err(e) => print_error(&ts_error_message("VARMA failed", &e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_vecm(
    dataset_name: &str,
    cols: &[String],
    lags: usize,
    rank: usize,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();

            match run_vecm(ds, &col_refs, lags, rank) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": "VECM",
                            "rank": result.rank,
                            "lags": result.lags,
                            "n_vars": result.n_vars,
                            "n_obs": result.n_obs,
                            "eigenvalues": result.eigenvalues,
                            "trace_stats": result.trace_stats,
                            "trace_crit_values": result.trace_crit_values,
                            "beta": result.beta,
                            "alpha": result.alpha,
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nVECM Results (rank = {})", rank);
                        println!("{}", "=".repeat(50));
                        println!("Cointegration rank: {}", result.rank);
                        println!("Lags: {}", result.lags);
                        println!("Number of variables: {}", result.n_vars);
                        println!("Observations: {}", result.n_obs);
                        println!("\nJohansen Trace Test:");
                        for (i, (stat, crit)) in result
                            .trace_stats
                            .iter()
                            .zip(&result.trace_crit_values)
                            .enumerate()
                        {
                            let sig = if *stat > *crit { "*" } else { "" };
                            println!(
                                "  r <= {}: stat={:.4}, crit(5%)={:.4} {}",
                                i, stat, crit, sig
                            );
                        }
                        println!("\nEigenvalues: {:?}", result.eigenvalues);
                    }
                },
                Err(e) => print_error(&ts_error_message("VECM failed", &e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

/// `{impulse: {response: [irf[h][response][impulse] for each h]}}`; NaN (null) where the
/// tensor is short.
fn impulse_responses(result: &p2a_core::VarIrfResult) -> serde_json::Value {
    let names = &result.var_names;
    let mut by_impulse = serde_json::Map::new();
    for (i, impulse) in names.iter().enumerate() {
        let mut by_response = serde_json::Map::new();
        for (r, response) in names.iter().enumerate() {
            let path: Vec<f64> = result
                .irf
                .iter()
                .map(|phi| {
                    phi.get(r)
                        .and_then(|row| row.get(i))
                        .copied()
                        .unwrap_or(f64::NAN)
                })
                .collect();
            by_response.insert(response.clone(), serde_json::json!(path));
        }
        by_impulse.insert(impulse.clone(), serde_json::Value::Object(by_response));
    }
    serde_json::Value::Object(by_impulse)
}

fn execute_irf(
    dataset_name: &str,
    cols: &[String],
    lags: usize,
    steps: usize,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();

            match run_var_irf(ds, &col_refs, lags, steps) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": "VAR IRF",
                            "var_names": result.var_names,
                            "steps": result.steps,
                            "axes": ["horizon", "response", "impulse"],
                            "irf": result.irf,
                            "impulse_responses": impulse_responses(&result),
                            "orthogonalized": result.orthogonalized,
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => println!("\n{result}"),
                },
                Err(e) => print_error(&ts_error_message("IRF failed", &e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_changepoint(
    dataset_name: &str,
    col: &str,
    penalty: Option<f64>,
    min_segment: Option<usize>,
    cost_fn: CostFunction,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            // Extract column data: Int64 accepted, missing values and text refused by name
            let data = match numeric_column(ds.df(), col, NaPolicy::Fail) {
                Ok((values, _)) => values,
                Err(e) => {
                    print_error(
                        &ts_error_message("Changepoint detection failed", &e),
                        format,
                    );
                    return Ok(());
                }
            };

            match detect_changepoints(&data, penalty, min_segment, cost_fn) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": result.method,
                            "change_type": CHANGE_TYPES.canonical(cost_fn),
                            "n_changepoints": result.n_changepoints,
                            "changepoints": result.changepoints,
                            "penalty": result.penalty,
                            "total_cost": result.total_cost,
                            "segments": result.segments.iter().map(|s| {
                                serde_json::json!({
                                    "start": s.start,
                                    "end": s.end,
                                    "n_points": s.n_points,
                                    "mean": s.mean,
                                    "variance": s.variance,
                                })
                            }).collect::<Vec<_>>(),
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nChangepoint Detection Results");
                        println!("{}", "=".repeat(50));
                        println!("Method: {}", result.method);
                        println!("Penalty: {:.4}", result.penalty);
                        println!("\nChangepoints detected: {}", result.n_changepoints);
                        if !result.changepoints.is_empty() {
                            println!("Changepoint indices: {:?}", result.changepoints);
                        }
                        println!("\nSegments:");
                        for (i, seg) in result.segments.iter().enumerate() {
                            println!(
                                "  Segment {}: [{}, {}) n={}, mean={:.4}, var={:.4}",
                                i + 1,
                                seg.start,
                                seg.end,
                                seg.n_points,
                                seg.mean,
                                seg.variance
                            );
                        }
                    }
                },
                Err(e) => print_error(
                    &ts_error_message("Changepoint detection failed", &e),
                    format,
                ),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_garch(
    dataset_name: &str,
    col: &str,
    arch: usize,
    garch: usize,
    horizon: Option<usize>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            // Extract column data: Int64 accepted, missing values and text refused by name
            let data = match numeric_column(ds.df(), col, NaPolicy::Fail) {
                Ok((values, _)) => values,
                Err(e) => {
                    print_error(&ts_error_message("GARCH failed", &e), format);
                    return Ok(());
                }
            };

            match run_garch(&data, Some(arch), Some(garch), Some(true)) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "model": format!("GARCH({},{})", arch, garch),
                            "omega": result.omega,
                            "alpha": result.alpha,
                            "beta": result.beta,
                            "persistence": result.persistence,
                            "unconditional_variance": result.unconditional_variance,
                            "log_likelihood": result.log_likelihood,
                            "aic": result.aic,
                            "bic": result.bic,
                            "n_obs": result.n_obs,
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nGARCH({},{}) Results", arch, garch);
                        println!("{}", "=".repeat(50));
                        println!("\nParameters:");
                        println!("  Omega (constant): {:.8}", result.omega);
                        for (i, a) in result.alpha.iter().enumerate() {
                            println!("  Alpha{}: {:.6}", i + 1, a);
                        }
                        for (i, b) in result.beta.iter().enumerate() {
                            println!("  Beta{}: {:.6}", i + 1, b);
                        }
                        println!("\nPersistence: {:.6}", result.persistence);
                        println!(
                            "Unconditional Variance: {:.6}",
                            result.unconditional_variance
                        );
                        println!("\nLog-likelihood: {:.4}", result.log_likelihood);
                        println!("AIC: {:.4}", result.aic);
                        println!("BIC: {:.4}", result.bic);
                        println!("Observations: {}", result.n_obs);

                        if let Some(h) = horizon {
                            println!("\nVariance forecasts (h={}):", h);
                            for (i, sigma2) in
                                result.conditional_variance.iter().rev().take(h).enumerate()
                            {
                                println!("  t+{}: {:.6}", i + 1, sigma2);
                            }
                        }
                    }
                },
                Err(e) => print_error(&ts_error_message("GARCH failed", &e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

/// Holt-Winters forecasts for the CLI: predict.HoltWinters's limits for additive and
/// non-seasonal fits; point forecasts only for a multiplicative seasonal fit (its interval
/// formula in R is wrong, and the core refuses it).
struct HwForecast {
    horizon: usize,
    level: Option<f64>,
    mean: Vec<f64>,
    std_errors: Option<Vec<f64>>,
    lower: Option<Vec<f64>>,
    upper: Option<Vec<f64>>,
    residual_variance: f64,
    warning: Option<String>,
}

const HW_MULT_INTERVALS: &str =
    "prediction intervals are implemented for additive and non-seasonal fits";

fn hw_forecast(
    result: &p2a_core::forecasting::HoltWintersResult,
    h: usize,
    level: f64,
) -> p2a_core::EconResult<HwForecast> {
    let multiplicative = result.coefficients.seasonal.is_some()
        && result.seasonal_type == SeasonalType::Multiplicative;
    if multiplicative {
        let mean = p2a_core::holt_winters_forecast(result, h)?;
        // var(residuals(fit)): sample variance of the one-step errors
        let errors: Vec<f64> = result
            .residuals
            .iter()
            .copied()
            .filter(|e| e.is_finite())
            .collect();
        let m = errors.len() as f64;
        let residual_variance = if errors.len() >= 2 {
            let mu = errors.iter().sum::<f64>() / m;
            errors.iter().map(|e| (e - mu).powi(2)).sum::<f64>() / (m - 1.0)
        } else {
            f64::NAN
        };
        return Ok(HwForecast {
            horizon: h,
            level: None,
            mean,
            std_errors: None,
            lower: None,
            upper: None,
            residual_variance,
            warning: Some(format!(
                "{HW_MULT_INTERVALS}; a multiplicative seasonal fit gets point forecasts only"
            )),
        });
    }
    let p = result.predict(h, level)?;
    Ok(HwForecast {
        horizon: p.horizon,
        level: Some(p.level),
        mean: p.forecast,
        std_errors: Some(p.std_errors),
        lower: Some(p.lower),
        upper: Some(p.upper),
        residual_variance: p.residual_variance,
        warning: None,
    })
}

/// A level as a percentage the way it was typed: 0.95 -> "95", 0.975 -> "97.5".
fn percent(level: f64) -> String {
    let pct = (level * 100.0 * 1e9).round() / 1e9;
    format!("{pct}")
}

fn execute_holt_winters(
    dataset_name: &str,
    col: &str,
    period: usize,
    seasonal_type: SeasonalType,
    (alpha, beta, gamma): (Option<f64>, Option<f64>, Option<f64>),
    horizon: Option<(usize, f64)>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };
    let Some(ds) = dataset else {
        print_error(&format!("Dataset '{}' not found", dataset_name), format);
        return Ok(());
    };

    // run_holt_winters(dataset, column, period, seasonal, alpha, beta, gamma)
    let result = match run_holt_winters(ds, col, period, seasonal_type, alpha, beta, gamma) {
        Ok(r) => r,
        Err(e) => {
            print_error(&ts_error_message("Holt-Winters failed", &e), format);
            return Ok(());
        }
    };
    let forecast = match horizon {
        Some((h, level)) => match hw_forecast(&result, h, level) {
            Ok(f) => Some(f),
            Err(e) => {
                print_error(
                    &ts_error_message("Holt-Winters forecast failed", &e),
                    format,
                );
                return Ok(());
            }
        },
        None => None,
    };
    let warnings: Vec<String> = forecast
        .as_ref()
        .and_then(|f| f.warning.clone())
        .into_iter()
        .collect();

    match format {
        OutputFormat::Json => {
            let mut json = serde_json::json!({
                "model": "Holt-Winters",
                "seasonal_type": SEASONAL_TYPES.canonical(result.seasonal_type),
                "period": result.period,
                "alpha": result.alpha,
                "beta": result.beta,
                "gamma": result.gamma,
                "fitted_sample": result.fitted.iter().take(10).collect::<Vec<_>>(),
                "sse": result.sse,
                "warnings": warnings,
            });
            if let Some(f) = &forecast {
                json["forecast"] = serde_json::json!({
                    "horizon": f.horizon,
                    "level": f.level,
                    "mean": f.mean,
                    "std_errors": f.std_errors,
                    "lower": f.lower,
                    "upper": f.upper,
                    "residual_variance": f.residual_variance,
                });
            }
            println!("{}", serde_json::to_string_pretty(&json)?);
        }
        _ => {
            println!("\nHolt-Winters Exponential Smoothing");
            println!("{}", "=".repeat(50));
            println!(
                "\nSeasonal type: {}",
                SEASONAL_TYPES.canonical(result.seasonal_type)
            );
            println!("Period: {}", result.period);
            println!("\nSmoothing parameters:");
            println!("  Alpha (level): {:.4}", result.alpha);
            if let Some(beta) = result.beta {
                println!("  Beta (trend): {:.4}", beta);
            }
            if let Some(gamma) = result.gamma {
                println!("  Gamma (seasonal): {:.4}", gamma);
            }
            println!("\nSSE: {:.4}", result.sse);
            if let Some(f) = &forecast {
                match (f.level, &f.lower, &f.upper) {
                    (Some(level), Some(lower), Some(upper)) => {
                        println!(
                            "\nForecasts (h = {}, {}% prediction interval):",
                            f.horizon,
                            percent(level)
                        );
                        for (i, ((fit, lo), hi)) in f.mean.iter().zip(lower).zip(upper).enumerate()
                        {
                            println!("  t+{}: {fit:.4}  [{lo:.4}, {hi:.4}]", i + 1);
                        }
                    }
                    _ => {
                        println!("\nForecasts (h = {}; {HW_MULT_INTERVALS}):", f.horizon);
                        for (i, fit) in f.mean.iter().enumerate() {
                            println!("  t+{}: {fit:.4}", i + 1);
                        }
                    }
                }
            }
            crate::output::print_notes(None, &warnings, format);
        }
    }
    Ok(())
}

fn execute_granger(
    dataset_name: &str,
    x: &str,
    y: &str,
    lags: usize,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            // granger_test(dataset, dependent, cause, lags)
            // Tests if x (cause) Granger-causes y (dependent)
            match granger_test(ds, y, x, lags) {
                Ok(result) => {
                    let granger_causes = result.p_value < 0.05;
                    match format {
                        OutputFormat::Json => {
                            let mut json = serde_json::json!({
                                "test": "Granger Causality",
                                "cause": result.cause,
                                "dependent": result.dependent,
                                "lags": result.lags,
                                "f_statistic": result.f_statistic,
                                "p_value": result.p_value,
                                "n_obs": result.n_obs,
                                "df1": result.df1,
                                "df2": result.df2,
                                "granger_causes": granger_causes,
                            });
                            with_result_notes(&mut json, &result);
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nGranger Causality Test");
                            println!("{}", "=".repeat(50));
                            println!(
                                "\nH0: {} does not Granger-cause {}",
                                result.cause, result.dependent
                            );
                            println!("H1: {} Granger-causes {}", result.cause, result.dependent);
                            println!("\nLags: {}", result.lags);
                            println!("Observations: {}", result.n_obs);
                            println!(
                                "\nF-statistic: {:.4} (df1={}, df2={})",
                                result.f_statistic, result.df1, result.df2
                            );
                            println!("P-value: {:.4}", result.p_value);
                            println!(
                                "\nConclusion: {}",
                                if granger_causes {
                                    format!(
                                        "{} Granger-causes {} at 5% significance",
                                        result.cause, result.dependent
                                    )
                                } else {
                                    format!(
                                        "No evidence that {} Granger-causes {}",
                                        result.cause, result.dependent
                                    )
                                }
                            );
                            print_result_notes(&result, format);
                        }
                    }
                }
                Err(e) => print_error(&ts_error_message("Granger test failed", &e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_ar(
    dataset_name: &str,
    col: &str,
    order: Option<usize>,
    ar_method: ArMethod,
    aic: bool,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            // Extract column data: Int64 accepted, missing values and text refused by name
            let data = match numeric_column(ds.df(), col, NaPolicy::Fail) {
                Ok((values, _)) => values,
                Err(e) => {
                    print_error(&ts_error_message("AR model failed", &e), format);
                    return Ok(());
                }
            };

            let config = ArConfig {
                aic,
                order,
                method: ar_method,
                ..Default::default()
            };

            match ar(&data, config) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let mut json = serde_json::json!({
                            "model": format!("AR({})", result.order),
                            "order": result.order,
                            "ar_coefficients": result.ar,
                            "var_pred": result.var_pred,
                            "x_mean": result.x_mean,
                            "method": AR_METHODS.canonical(result.method),
                            "n_obs": result.n_obs,
                            "aic": result.aic,
                            "partial_acf": result.partial_acf,
                        });
                        with_result_notes(&mut json, &result);
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nAR({}) Model Results", result.order);
                        println!("{}", "=".repeat(50));
                        println!("Method: {}", AR_METHODS.canonical(result.method));
                        println!("Observations: {}", result.n_obs);
                        println!("\nCoefficients:");
                        for (i, coef) in result.ar.iter().enumerate() {
                            println!("  AR{}: {:.6}", i + 1, coef);
                        }
                        println!("\nMean: {:.6}", result.x_mean);
                        println!("Prediction variance: {:.6}", result.var_pred);
                        if let Some(ref aic_vals) = result.aic {
                            println!("\nAIC (relative to minimum):");
                            for (i, aic_val) in aic_vals.iter().enumerate().take(10) {
                                if *aic_val < 100.0 {
                                    println!("  AR({}): {:.2}", i, aic_val);
                                }
                            }
                        }
                        print_result_notes(&result, format);
                    }
                },
                Err(e) => print_error(&ts_error_message("AR model failed", &e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_stl(
    dataset_name: &str,
    col: &str,
    period: usize,
    robust: bool,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => match run_stl(ds, col, period, robust) {
            Ok(result) => match format {
                OutputFormat::Json => {
                    let json = serde_json::json!({
                        "method": "STL",
                        "period": result.period,
                        "n_obs": result.n_obs,
                        "robust": result.robust,
                        "seasonal_strength": result.seasonal_strength,
                        "trend_strength": result.trend_strength,
                        "trend_sample": result.trend.iter().take(10).collect::<Vec<_>>(),
                        "seasonal_sample": result.seasonal.iter().take(10).collect::<Vec<_>>(),
                        "remainder_sample": result.remainder.iter().take(10).collect::<Vec<_>>(),
                    });
                    println!("{}", serde_json::to_string_pretty(&json)?);
                }
                _ => {
                    println!("\nSTL Decomposition Results");
                    println!("{}", "=".repeat(50));
                    println!("Period: {}", result.period);
                    println!("Observations: {}", result.n_obs);
                    println!("Robust fitting: {}", result.robust);
                    println!("\nStrength of components:");
                    println!("  Seasonal strength: {:.4}", result.seasonal_strength);
                    println!("  Trend strength: {:.4}", result.trend_strength);
                    println!(
                        "\nTrend (first 10 values): {:?}",
                        result.trend.iter().take(10).collect::<Vec<_>>()
                    );
                    println!(
                        "Seasonal (first 10 values): {:?}",
                        result.seasonal.iter().take(10).collect::<Vec<_>>()
                    );
                    println!(
                        "Remainder (first 10 values): {:?}",
                        result.remainder.iter().take(10).collect::<Vec<_>>()
                    );
                }
            },
            Err(e) => print_error(&ts_error_message("STL decomposition failed", &e), format),
        },
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_decompose(
    dataset_name: &str,
    col: &str,
    period: usize,
    dtype: DecomposeType,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => match run_decompose(ds, col, period, dtype) {
            Ok(result) => match format {
                OutputFormat::Json => {
                    let json = serde_json::json!({
                        "method": "Classical Decomposition",
                        "decompose_type": DECOMPOSE_TYPES.canonical(result.decompose_type),
                        "period": result.period,
                        "n_obs": result.n_obs,
                        "figure": result.figure,
                        "trend_sample": result.trend.iter().take(10).collect::<Vec<_>>(),
                        "seasonal_sample": result.seasonal.iter().take(10).collect::<Vec<_>>(),
                        "random_sample": result.random.iter().take(10).collect::<Vec<_>>(),
                    });
                    println!("{}", serde_json::to_string_pretty(&json)?);
                }
                _ => {
                    println!("\nClassical Decomposition Results");
                    println!("{}", "=".repeat(50));
                    println!("Type: {}", DECOMPOSE_TYPES.canonical(result.decompose_type));
                    println!("Period: {}", result.period);
                    println!("Observations: {}", result.n_obs);
                    println!("\nSeasonal figure (one cycle):");
                    for (i, f) in result.figure.iter().enumerate() {
                        println!("  Position {}: {:.4}", i, f);
                    }
                    println!("\nTrend (first 10 values, may contain NaN at boundaries):");
                    for (i, t) in result.trend.iter().take(10).enumerate() {
                        println!("  {}: {:.4}", i, t);
                    }
                }
            },
            Err(e) => print_error(&ts_error_message("Decomposition failed", &e), format),
        },
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_struct_ts(
    dataset_name: &str,
    col: &str,
    ts_type: StructTsType,
    period: Option<usize>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => match run_struct_ts(ds, col, ts_type, period) {
            Ok(result) => match format {
                OutputFormat::Json => {
                    let model_type = STRUCT_TS_TYPES.canonical(result.model_type);
                    let mut json = serde_json::json!({
                        "model": format!("StructTS {model_type}"),
                        "model_type": model_type,
                        "n_obs": result.n_obs,
                        "n_params": result.n_params,
                        "converged": result.converged,
                        "log_likelihood": result.log_likelihood,
                        "aic": result.aic,
                        "bic": result.bic,
                        "coefficients": {
                            "level": result.coef.level,
                            "slope": result.coef.slope,
                            "seasonal": result.coef.seasonal,
                            "epsilon": result.coef.epsilon,
                        },
                        "fitted_sample": result.fitted.iter().take(10).collect::<Vec<_>>(),
                    });
                    with_result_notes(&mut json, &result);
                    println!("{}", serde_json::to_string_pretty(&json)?);
                }
                _ => {
                    println!("\nStructural Time Series Model Results");
                    println!("{}", "=".repeat(50));
                    println!(
                        "Model type: {}",
                        STRUCT_TS_TYPES.canonical(result.model_type)
                    );
                    println!("Observations: {}", result.n_obs);
                    println!("Parameters: {}", result.n_params);
                    println!("\nVariance estimates:");
                    println!("  Level: {:.6}", result.coef.level);
                    if let Some(slope) = result.coef.slope {
                        println!("  Slope: {:.6}", slope);
                    }
                    if let Some(seasonal) = result.coef.seasonal {
                        println!("  Seasonal: {:.6}", seasonal);
                    }
                    println!("  Observation: {:.6}", result.coef.epsilon);
                    println!("\nLog-likelihood: {:.4}", result.log_likelihood);
                    println!("AIC: {:.4}", result.aic);
                    println!("BIC: {:.4}", result.bic);
                    print_result_notes(&result, format);
                }
            },
            Err(e) => print_error(&ts_error_message("StructTS failed", &e), format),
        },
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_causal_impact(
    dataset_name: &str,
    response: &str,
    time: &str,
    pre_start: i64,
    pre_end: i64,
    post_start: i64,
    post_end: i64,
    controls: Option<&str>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            let control_cols: Option<Vec<&str>> =
                controls.map(|c| c.split(',').map(|s| s.trim()).collect());

            match run_causal_impact(
                ds,
                response,
                time,
                (pre_start, pre_end),
                (post_start, post_end),
                control_cols.as_deref(),
            ) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let mut json = serde_json::json!({
                            "method": "CausalImpact",
                            "summary": {
                                "average_effect": result.summary.average_effect,
                                "average_effect_lower": result.summary.average_effect_lower,
                                "average_effect_upper": result.summary.average_effect_upper,
                                "cumulative_effect": result.summary.cumulative_effect,
                                "cumulative_effect_lower": result.summary.cumulative_effect_lower,
                                "cumulative_effect_upper": result.summary.cumulative_effect_upper,
                                "relative_effect": result.summary.relative_effect,
                                "p_value": result.summary.p_value,
                                "significant": result.summary.significant,
                            },
                            "model": {
                                "n_pre": result.model.n_pre,
                                "n_post": result.model.n_post,
                                "level_variance": result.model.level_variance,
                                "observation_variance": result.model.observation_variance,
                                "log_likelihood": result.model.log_likelihood,
                                "aic": result.model.aic,
                            },
                            "inference": {
                                "prob_positive": result.inference.prob_positive,
                                "prob_negative": result.inference.prob_negative,
                                "expected_effect": result.inference.expected_effect,
                                "null_rejected": result.inference.null_rejected,
                            },
                        });
                        with_result_notes(&mut json, &result);
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nCausal Impact Analysis Results");
                        println!("{}", "=".repeat(50));
                        println!("\nPre-period: {} to {}", pre_start, pre_end);
                        println!("Post-period: {} to {}", post_start, post_end);
                        println!("Pre-period observations: {}", result.model.n_pre);
                        println!("Post-period observations: {}", result.model.n_post);

                        println!("\n--- Summary ---");
                        println!(
                            "Average causal effect: {:.4} [{:.4}, {:.4}]",
                            result.summary.average_effect,
                            result.summary.average_effect_lower,
                            result.summary.average_effect_upper
                        );
                        println!(
                            "Cumulative effect: {:.4} [{:.4}, {:.4}]",
                            result.summary.cumulative_effect,
                            result.summary.cumulative_effect_lower,
                            result.summary.cumulative_effect_upper
                        );
                        println!(
                            "Relative effect: {:.2}%",
                            result.summary.relative_effect * 100.0
                        );
                        println!("P-value: {:.4}", result.summary.p_value);
                        println!(
                            "Significant at {}%: {}",
                            (result.summary.alpha * 100.0) as i32,
                            if result.summary.significant {
                                "Yes"
                            } else {
                                "No"
                            }
                        );

                        println!("\n--- Inference ---");
                        println!(
                            "Posterior prob (effect > 0): {:.2}%",
                            result.inference.prob_positive * 100.0
                        );
                        println!(
                            "Posterior prob (effect < 0): {:.2}%",
                            result.inference.prob_negative * 100.0
                        );
                        println!(
                            "Null hypothesis rejected: {}",
                            if result.inference.null_rejected {
                                "Yes"
                            } else {
                                "No"
                            }
                        );
                        print_result_notes(&result, format);
                    }
                },
                Err(e) => print_error(&ts_error_message("CausalImpact failed", &e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

#[cfg(feature = "spectral-analysis")]
fn execute_spectrum(
    dataset_name: &str,
    col: &str,
    spans: &[usize],
    taper: f64,
    detrend: bool,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            let mut config = SpectrumConfig::default();
            config.spans = (!spans.is_empty()).then(|| spans.to_vec());
            config.taper = taper;
            config.detrend = detrend;
            config.demean = false;
            config.pad_ratio = 0.0;

            match run_spectrum(ds, col, config) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": result.method,
                            "n_obs": result.n_obs,
                            "n_used": result.n_used,
                            "bandwidth": result.bandwidth,
                            "df": result.df,
                            "taper": result.taper,
                            "detrend": result.detrend,
                            "kernel_spans": result.kernel_spans,
                            "peak_frequency": result.peak_frequency().map(|(f, s)| {
                                serde_json::json!({"frequency": f, "spectrum": s, "period": 1.0 / f})
                            }),
                            "freq_sample": result.freq.iter().take(10).collect::<Vec<_>>(),
                            "spec_sample": result.spec.iter().take(10).collect::<Vec<_>>(),
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nSpectral Density Estimation Results");
                        println!("{}", "=".repeat(50));
                        println!("Method: {}", result.method);
                        println!("Observations: {}", result.n_obs);
                        println!("Series length used: {}", result.n_used);
                        println!("Bandwidth: {:.4}", result.bandwidth);
                        println!("Degrees of freedom: {:.2}", result.df);
                        println!("Taper: {:.1}%", result.taper * 100.0);
                        println!("Detrend: {}", if result.detrend { "Yes" } else { "No" });
                        if let Some(ref spans) = result.kernel_spans {
                            println!("Kernel spans: {:?}", spans);
                        }
                        if let Some((peak_freq, peak_spec)) = result.peak_frequency() {
                            println!(
                                "\nPeak frequency: {:.4} (period = {:.2})",
                                peak_freq,
                                1.0 / peak_freq
                            );
                            println!("Peak spectral density: {:.4e}", peak_spec);
                        }
                        println!("\nFrequency    Spectrum");
                        println!("---------    --------");
                        for (f, s) in result.freq.iter().zip(result.spec.iter()).take(10) {
                            println!("{:9.4}    {:.4e}", f, s);
                        }
                        if result.freq.len() > 10 {
                            println!("   ...         ...");
                        }
                    }
                },
                Err(e) => print_error(&ts_error_message("Spectrum estimation failed", &e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_option_tables_parse_aliases_and_reject_unknown() {
        for v in [
            CHANGE_TYPES.validate(),
            SEASONAL_TYPES.validate(),
            AR_METHODS.validate(),
            DECOMPOSE_TYPES.validate(),
            STRUCT_TS_TYPES.validate(),
        ] {
            v.unwrap()
        }
        assert_eq!(
            parse_change_type("both").unwrap(),
            CostFunction::MeanAndVariance
        );
        assert_eq!(
            parse_change_type("VAR").unwrap(),
            CostFunction::VarianceChange
        );
        assert_eq!(
            parse_seasonal("Mult").unwrap(),
            SeasonalType::Multiplicative
        );
        assert_eq!(
            parse_ar_method("Yule Walker").unwrap(),
            ArMethod::YuleWalker
        );
        assert_eq!(parse_ar_method("yw").unwrap(), ArMethod::YuleWalker);
        assert_eq!(
            parse_decompose_type("multi").unwrap(),
            DecomposeType::Multiplicative
        );
        assert_eq!(parse_struct_ts_type("bsm").unwrap(), StructTsType::BSM);
        for (e, want) in [
            (
                parse_change_type("trend").unwrap_err(),
                "unknown change type 'trend'; expected one of: mean, variance, meanvar",
            ),
            (
                parse_ar_method("mle").unwrap_err(),
                "unknown AR method 'mle'; expected one of: yule-walker, burg, ols",
            ),
            (
                parse_struct_ts_type("arima").unwrap_err(),
                "unknown StructTS type 'arima'; expected one of: level, trend, BSM",
            ),
        ] {
            assert!(e.contains(want), "{e}")
        }
        assert_eq!(
            CHANGE_TYPES.canonical(CostFunction::MeanAndVariance),
            "meanvar"
        );
    }
}
