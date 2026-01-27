//! Time series analysis commands

use clap::Subcommand;
use p2a_core::{
    run_arima, forecast_arima, run_mstl, run_var,
    run_varma, run_vecm, run_var_irf,
    detect_changepoints, CostFunction,
    run_garch, run_holt_winters,
    SeasonalType,
};
use p2a_core::econometrics::granger_test;

use crate::output::{print_error, OutputFormat};
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
        } => execute_changepoint(dataset, col, *penalty, *min_segment, change_type, format, session),
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
                Ok(result) => {
                    match format {
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
                    }
                }
                Err(e) => {
                    print_error(&format!("ARIMA failed: {}", e), format);
                }
            }

            // If horizon is specified, also produce forecasts
            if let Some(h) = horizon {
                match forecast_arima(ds, col, ar, diff, ma, h) {
                    Ok(forecast_result) => {
                        match format {
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
                        }
                    }
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
        Some(ds) => {
            match run_mstl(ds, col, periods) {
                Ok(result) => {
                    match format {
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
                            println!("\nTrend (first 10 values): {:?}",
                                result.trend.iter().take(10).collect::<Vec<_>>());
                            println!("Number of seasonal components: {}",
                                result.seasonal.len());
                            println!("Residuals (first 10 values): {:?}",
                                result.residuals.iter().take(10).collect::<Vec<_>>());
                        }
                    }
                }
                Err(e) => {
                    print_error(&format!("MSTL failed: {}", e), format);
                }
            }
        }
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
                Ok(result) => {
                    match format {
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
                    }
                }
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
                Ok(result) => {
                    match format {
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
                    }
                }
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
                Ok(result) => {
                    match format {
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
                            for (i, (stat, crit)) in result.trace_stats.iter().zip(&result.trace_crit_values).enumerate() {
                                let sig = if *stat > *crit { "*" } else { "" };
                                println!("  r <= {}: stat={:.4}, crit(5%)={:.4} {}", i, stat, crit, sig);
                            }
                            println!("\nEigenvalues: {:?}", result.eigenvalues);
                        }
                    }
                }
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
                            println!("\nIRF matrix dimensions: {} x {} x {}",
                                result.irf.len(),
                                if !result.irf.is_empty() { result.irf[0].len() } else { 0 },
                                if !result.irf.is_empty() && !result.irf[0].is_empty() { result.irf[0][0].len() } else { 0 });

                            // Print first few steps for each impulse-response pair
                            let n_vars = result.var_names.len();
                            for impulse in 0..n_vars.min(2) {
                                for response in 0..n_vars.min(2) {
                                    println!("\n{} -> {} (first 5 steps):",
                                        result.var_names[impulse], result.var_names[response]);
                                    for step in 0..5.min(result.steps) {
                                        println!("  t+{}: {:.6}", step, result.irf[step][impulse][response]);
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
                Ok(col) => col.f64().map_or_else(
                    |_| vec![],
                    |ca| ca.into_no_null_iter().collect()
                ),
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
                Ok(result) => {
                    match format {
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
                                println!("  Segment {}: [{}, {}) n={}, mean={:.4}, var={:.4}",
                                    i + 1, seg.start, seg.end, seg.n_points, seg.mean, seg.variance);
                            }
                        }
                    }
                }
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
                Ok(c) => c.f64().map_or_else(
                    |_| vec![],
                    |ca| ca.into_no_null_iter().collect()
                ),
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
                Ok(result) => {
                    match format {
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
                            println!("Unconditional Variance: {:.6}", result.unconditional_variance);
                            println!("\nLog-likelihood: {:.4}", result.log_likelihood);
                            println!("AIC: {:.4}", result.aic);
                            println!("BIC: {:.4}", result.bic);
                            println!("Observations: {}", result.n_obs);

                            if let Some(h) = horizon {
                                println!("\nVariance forecasts (h={}):", h);
                                for (i, sigma2) in result.conditional_variance.iter().rev().take(h).enumerate() {
                                    println!("  t+{}: {:.6}", i + 1, sigma2);
                                }
                            }
                        }
                    }
                }
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
                Ok(result) => {
                    match format {
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
                    }
                }
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
                            println!("\nH0: {} does not Granger-cause {}", result.cause, result.dependent);
                            println!("H1: {} Granger-causes {}", result.cause, result.dependent);
                            println!("\nLags: {}", result.lags);
                            println!("Observations: {}", result.n_obs);
                            println!("\nF-statistic: {:.4} (df1={}, df2={})",
                                result.f_statistic, result.df1, result.df2);
                            println!("P-value: {:.4}", result.p_value);
                            println!("\nConclusion: {}", if granger_causes {
                                format!("{} Granger-causes {} at 5% significance",
                                    result.cause, result.dependent)
                            } else {
                                format!("No evidence that {} Granger-causes {}",
                                    result.cause, result.dependent)
                            });
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
