//! Time series analysis commands

use clap::Subcommand;
use p2a_core::{run_arima, forecast_arima, run_mstl, run_var};

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
