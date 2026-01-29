//! Time series analysis commands

use clap::Subcommand;
use p2a_core::econometrics::granger_test;
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
    forecast_arima,
    run_arima,
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

use crate::output::{OutputFormat, print_error};
use crate::session::SessionManager;

#[derive(Subcommand)]
pub enum TimeseriesCommands {
    /// ARIMA model estimation and forecasting
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

        /// Number of lags
        #[arg(long, default_value = "1")]
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

        /// Number of steps for IRF
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

        /// Penalty value (default: automatic BIC)
        #[arg(long)]
        penalty: Option<f64>,

        /// Minimum segment length (default: 2)
        #[arg(long)]
        min_segment: Option<usize>,

        /// Type of change: "mean" (default), "variance", "both"
        #[arg(long, default_value = "mean")]
        change_type: String,
    },

    /// GARCH volatility model
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
    HoltWinters {
        /// Dataset name
        dataset: String,

        /// Time series column
        #[arg(long)]
        col: String,

        /// Seasonal period (e.g., 12 for monthly, 4 for quarterly)
        #[arg(long)]
        period: usize,

        /// Seasonal type: "additive" (default) or "multiplicative"
        #[arg(long, default_value = "additive")]
        seasonal: String,

        /// Forecast horizon (optional)
        #[arg(long)]
        horizon: Option<usize>,
    },

    /// Granger causality test
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

        /// AR order (if not using AIC selection)
        #[arg(long)]
        order: Option<usize>,

        /// Method: "yule-walker" (default), "burg", or "ols"
        #[arg(long, default_value = "yule-walker")]
        method: String,

        /// Use AIC for order selection (default: true)
        #[arg(long, default_value = "true")]
        aic: bool,
    },

    /// STL decomposition (Seasonal-Trend using LOESS)
    Stl {
        /// Dataset name
        dataset: String,

        /// Time series column
        #[arg(long)]
        col: String,

        /// Seasonal period (e.g., 12 for monthly, 4 for quarterly)
        #[arg(long)]
        period: usize,

        /// Use robust fitting (default: false)
        #[arg(long, default_value = "false")]
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

        /// Decomposition type: "additive" (default) or "multiplicative"
        #[arg(long, default_value = "additive")]
        decompose_type: String,
    },

    /// Structural time series (state space) model
    StructTs {
        /// Dataset name
        dataset: String,

        /// Time series column
        #[arg(long)]
        col: String,

        /// Model type: "level" (default), "trend", or "bsm"
        #[arg(long, default_value = "level")]
        model_type: String,

        /// Seasonal period (required for "bsm" model type)
        #[arg(long)]
        period: Option<usize>,
    },

    /// Bayesian structural time series causal inference (CausalImpact)
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

        /// Smoothing spans (comma-separated odd integers, e.g., "3,3")
        #[arg(long)]
        spans: Option<String>,

        /// Taper proportion (0.0 to 0.5, default: 0.1)
        #[arg(long, default_value = "0.1")]
        taper: f64,

        /// Whether to detrend (default: true)
        #[arg(long, default_value = "true")]
        detrend: bool,
    },
}

pub fn execute(
    cmd: &TimeseriesCommands,
    format: &OutputFormat,
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
        } => execute_arima(dataset, col, *ar, *diff, *ma, *horizon, format, session),
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
            change_type,
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
            horizon,
        } => execute_holt_winters(dataset, col, *period, seasonal, *horizon, format, session),
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
        } => execute_ar(dataset, col, *order, method, *aic, format, session),
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
        } => execute_decompose(dataset, col, *period, decompose_type, format, session),
        TimeseriesCommands::StructTs {
            dataset,
            col,
            model_type,
            period,
        } => execute_struct_ts(dataset, col, model_type, *period, format, session),
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
        } => execute_spectrum(
            dataset,
            col,
            spans.as_deref(),
            *taper,
            *detrend,
            format,
            session,
        ),
    }
}

fn execute_arima(
    dataset_name: &str,
    col: &str,
    ar: usize,
    diff: usize,
    ma: usize,
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

    match dataset {
        Some(ds) => {
            // run_arima takes 5 args (no horizon)
            match run_arima(ds, col, ar, diff, ma) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "model": format!("ARIMA({},{},{})", ar, diff, ma),
                            "ar_coeffs": result.ar_coeffs,
                            "ma_coeffs": result.ma_coeffs,
                            "intercept": result.intercept,
                            "ssr": result.ssr,
                            "aic": result.aic,
                            "n_obs": result.n_obs,
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nARIMA({},{},{}) Results", ar, diff, ma);
                        println!("{}", "=".repeat(50));

                        if !result.ar_coeffs.is_empty() {
                            println!("\nAR Coefficients:");
                            for (i, coef) in result.ar_coeffs.iter().enumerate() {
                                println!("  AR{}: {:.6}", i + 1, coef);
                            }
                        }

                        if !result.ma_coeffs.is_empty() {
                            println!("\nMA Coefficients:");
                            for (i, coef) in result.ma_coeffs.iter().enumerate() {
                                println!("  MA{}: {:.6}", i + 1, coef);
                            }
                        }

                        println!("\nIntercept: {:.6}", result.intercept);
                        println!("SSR: {:.6}", result.ssr);
                        println!("AIC: {:.4}", result.aic);
                        println!("Observations: {}", result.n_obs);
                    }
                },
                Err(e) => {
                    print_error(&format!("ARIMA failed: {}", e), format);
                }
            }

            // If horizon is specified, also produce forecasts
            if let Some(h) = horizon {
                match forecast_arima(ds, col, ar, diff, ma, h) {
                    Ok(forecast_result) => match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "forecasts": forecast_result.forecast,
                                "horizon": forecast_result.horizon,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nForecasts (horizon = {}):", h);
                            for (i, forecast) in forecast_result.forecast.iter().enumerate() {
                                println!("  t+{}: {:.4}", i + 1, forecast);
                            }
                        }
                    },
                    Err(e) => {
                        print_error(&format!("Forecasting failed: {}", e), format);
                    }
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
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
                print_error(&format!("MSTL failed: {}", e), format);
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
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nVAR({}) Results", lags);
                        println!("{}", "=".repeat(50));
                        println!("Variables: {:?}", result.var_names);
                        println!("Number of variables: {}", result.n_vars);
                        println!("Observations: {}", result.n_obs);
                        println!("\nAIC: {:.4}", result.aic);
                        println!("BIC: {:.4}", result.bic);
                        println!("Log-likelihood: {:.4}", result.log_likelihood);
                    }
                },
                Err(e) => {
                    print_error(&format!("VAR failed: {}", e), format);
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
                Err(e) => print_error(&format!("VARMA failed: {}", e), format),
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
                Err(e) => print_error(&format!("VECM failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
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
                Ok(result) => {
                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "method": "VAR IRF",
                                "var_names": result.var_names,
                                "steps": result.steps,
                                "irf": result.irf,
                                "orthogonalized": result.orthogonalized,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nImpulse Response Functions");
                            println!("{}", "=".repeat(50));
                            println!("Variables: {:?}", result.var_names);
                            println!("Steps: {}", result.steps);
                            println!("Orthogonalized: {}", result.orthogonalized);
                            println!(
                                "\nIRF matrix dimensions: {} x {} x {}",
                                result.irf.len(),
                                if !result.irf.is_empty() {
                                    result.irf[0].len()
                                } else {
                                    0
                                },
                                if !result.irf.is_empty() && !result.irf[0].is_empty() {
                                    result.irf[0][0].len()
                                } else {
                                    0
                                }
                            );

                            // Print first few steps for each impulse-response pair
                            let n_vars = result.var_names.len();
                            for impulse in 0..n_vars.min(2) {
                                for response in 0..n_vars.min(2) {
                                    println!(
                                        "\n{} -> {} (first 5 steps):",
                                        result.var_names[impulse], result.var_names[response]
                                    );
                                    for step in 0..5.min(result.steps) {
                                        println!(
                                            "  t+{}: {:.6}",
                                            step, result.irf[step][impulse][response]
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => print_error(&format!("IRF failed: {}", e), format),
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
    change_type: &str,
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
            // Extract column data
            let column = ds.df().column(col);
            let data: Vec<f64> = match column {
                Ok(col) => col
                    .f64()
                    .map_or_else(|_| vec![], |ca| ca.into_no_null_iter().collect()),
                Err(_) => {
                    print_error(&format!("Column '{}' not found", col), format);
                    return Ok(());
                }
            };

            if data.is_empty() {
                print_error(&format!("Column '{}' is empty or not numeric", col), format);
                return Ok(());
            }

            let cost_fn = match change_type.to_lowercase().as_str() {
                "variance" => CostFunction::VarianceChange,
                "both" => CostFunction::MeanAndVariance,
                _ => CostFunction::MeanChange,
            };

            match detect_changepoints(&data, penalty, min_segment, cost_fn) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": result.method,
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
                Err(e) => print_error(&format!("Changepoint detection failed: {}", e), format),
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
            // Extract column data
            let column = ds.df().column(col);
            let data: Vec<f64> = match column {
                Ok(c) => c
                    .f64()
                    .map_or_else(|_| vec![], |ca| ca.into_no_null_iter().collect()),
                Err(_) => {
                    print_error(&format!("Column '{}' not found", col), format);
                    return Ok(());
                }
            };

            if data.is_empty() {
                print_error(&format!("Column '{}' is empty or not numeric", col), format);
                return Ok(());
            }

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
                Err(e) => print_error(&format!("GARCH failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_holt_winters(
    dataset_name: &str,
    col: &str,
    period: usize,
    seasonal: &str,
    _horizon: Option<usize>,
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
            let seasonal_type = match seasonal.to_lowercase().as_str() {
                "multiplicative" | "mult" => SeasonalType::Multiplicative,
                _ => SeasonalType::Additive,
            };

            // run_holt_winters(dataset, column, period, seasonal, alpha, beta, gamma)
            match run_holt_winters(ds, col, period, seasonal_type, None, None, None) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "model": "Holt-Winters",
                            "seasonal_type": format!("{:?}", result.seasonal_type),
                            "period": result.period,
                            "alpha": result.alpha,
                            "beta": result.beta,
                            "gamma": result.gamma,
                            "fitted_sample": result.fitted.iter().take(10).collect::<Vec<_>>(),
                            "sse": result.sse,
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nHolt-Winters Exponential Smoothing");
                        println!("{}", "=".repeat(50));
                        println!("\nSeasonal type: {:?}", result.seasonal_type);
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
                    }
                },
                Err(e) => print_error(&format!("Holt-Winters failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
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
                            let json = serde_json::json!({
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
                        }
                    }
                }
                Err(e) => print_error(&format!("Granger test failed: {}", e), format),
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
    method: &str,
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
            // Extract column data
            let column = ds.df().column(col);
            let data: Vec<f64> = match column {
                Ok(c) => c
                    .f64()
                    .map_or_else(|_| vec![], |ca| ca.into_no_null_iter().collect()),
                Err(_) => {
                    print_error(&format!("Column '{}' not found", col), format);
                    return Ok(());
                }
            };

            if data.is_empty() {
                print_error(&format!("Column '{}' is empty or not numeric", col), format);
                return Ok(());
            }

            let ar_method = match method.to_lowercase().as_str() {
                "burg" => ArMethod::Burg,
                "ols" => ArMethod::Ols,
                _ => ArMethod::YuleWalker,
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
                        let json = serde_json::json!({
                            "model": format!("AR({})", result.order),
                            "order": result.order,
                            "ar_coefficients": result.ar,
                            "var_pred": result.var_pred,
                            "x_mean": result.x_mean,
                            "method": format!("{:?}", result.method),
                            "n_obs": result.n_obs,
                            "aic": result.aic,
                            "partial_acf": result.partial_acf,
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nAR({}) Model Results", result.order);
                        println!("{}", "=".repeat(50));
                        println!("Method: {:?}", result.method);
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
                    }
                },
                Err(e) => print_error(&format!("AR model failed: {}", e), format),
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
            Err(e) => print_error(&format!("STL decomposition failed: {}", e), format),
        },
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_decompose(
    dataset_name: &str,
    col: &str,
    period: usize,
    decompose_type: &str,
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
            let dtype = match decompose_type.to_lowercase().as_str() {
                "multiplicative" | "mult" => DecomposeType::Multiplicative,
                _ => DecomposeType::Additive,
            };

            match run_decompose(ds, col, period, dtype) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": "Classical Decomposition",
                            "decompose_type": format!("{:?}", result.decompose_type),
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
                        println!("Type: {:?}", result.decompose_type);
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
                Err(e) => print_error(&format!("Decomposition failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_struct_ts(
    dataset_name: &str,
    col: &str,
    model_type: &str,
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
        Some(ds) => {
            let ts_type = match model_type.to_lowercase().as_str() {
                "trend" => StructTsType::Trend,
                "bsm" => StructTsType::BSM,
                _ => StructTsType::Level,
            };

            match run_struct_ts(ds, col, ts_type, period) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "model": format!("StructTS {:?}", result.model_type),
                            "model_type": format!("{:?}", result.model_type),
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
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nStructural Time Series Model Results");
                        println!("{}", "=".repeat(50));
                        println!("Model type: {:?}", result.model_type);
                        println!("Observations: {}", result.n_obs);
                        println!("Parameters: {}", result.n_params);
                        println!("Converged: {}", result.converged);
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
                    }
                },
                Err(e) => print_error(&format!("StructTS failed: {}", e), format),
            }
        }
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
                        let json = serde_json::json!({
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
                    }
                },
                Err(e) => print_error(&format!("CausalImpact failed: {}", e), format),
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
    spans: Option<&str>,
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
            let span_vec: Option<Vec<usize>> = spans.map(|s| {
                s.split(',')
                    .filter_map(|v| v.trim().parse::<usize>().ok())
                    .collect()
            });

            let config = SpectrumConfig {
                spans: span_vec,
                taper,
                detrend,
                demean: false,
                pad_ratio: 0.0,
            };

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
                Err(e) => print_error(&format!("Spectrum estimation failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}
