//! Time Series Example
//!
//! Demonstrates time series analysis with:
//! - ACF/PACF analysis
//! - ARIMA modeling and forecasting (requires `forecasting` feature)
//! - Holt-Winters exponential smoothing (requires `forecasting` feature)
//!
//! Run with: cargo run -p p2a-core --example time_series --features forecasting

use p2a_core::{
    data::Dataset,
    stats::{AcfType, acf, pacf},
};
use polars::prelude::*;

#[cfg(feature = "forecasting")]
use p2a_core::forecasting::{
    HoltWintersConfig, SeasonalType, forecast_arima, holt_winters, holt_winters_forecast, run_arima,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Time Series Analysis Example ===\n");

    // Create sample quarterly GDP data (seasonally adjusted, in billions)
    let gdp_values: Vec<f64> = vec![
        18200.0, 18350.0, 18500.0, 18700.0, // 2019
        18900.0, 17500.0, 18100.0, 18600.0, // 2020 (COVID dip)
        19000.0, 19300.0, 19600.0, 19900.0, // 2021
        20200.0, 20400.0, 20600.0, 20800.0, // 2022
        21000.0, 21200.0, 21400.0, 21600.0, // 2023
    ];

    let quarters: Vec<String> = (0..20)
        .map(|i| format!("{}Q{}", 2019 + i / 4, 1 + i % 4))
        .collect();

    println!(
        "Time series: {} quarterly observations (2019Q1-2023Q4)\n",
        gdp_values.len()
    );

    // ACF Analysis
    println!("--- Autocorrelation Function (ACF) ---");
    let acf_result = acf(&gdp_values, Some(8), AcfType::Correlation, true, false)?;
    println!("Lag    ACF");
    println!("---  ------");
    let crit_val = acf_result.confidence_bound.unwrap_or(0.4);
    for (lag, &acf_val) in acf_result.lags.iter().zip(acf_result.values.iter()) {
        let stars = if acf_val.abs() > crit_val { "*" } else { "" };
        println!("{:>3}  {:>6.3} {}", lag, acf_val, stars);
    }
    println!("(* exceeds 95% CI bound: ±{:.3})\n", crit_val);

    // PACF Analysis
    println!("--- Partial Autocorrelation Function (PACF) ---");
    let pacf_result = pacf(&gdp_values, Some(8))?;
    println!("Lag   PACF");
    println!("---  ------");
    for (lag, &pacf_val) in pacf_result.lags.iter().zip(pacf_result.values.iter()) {
        let stars = if pacf_val.abs() > crit_val { "*" } else { "" };
        println!("{:>3}  {:>6.3} {}", lag, pacf_val, stars);
    }
    println!();

    #[cfg(feature = "forecasting")]
    {
        // Create dataset for ARIMA
        let df = df! {
            "quarter" => quarters.clone(),
            "gdp" => gdp_values.clone(),
        }?;
        let dataset = Dataset::new(df);

        // ARIMA Modeling
        println!("--- ARIMA(1,1,0) Model ---");
        let arima_result = run_arima(&dataset, "gdp", 1, 1, 0)?;
        println!(
            "ARIMA({},{},{}) fitted on '{}' (n={})",
            arima_result.p, arima_result.d, arima_result.q, arima_result.column, arima_result.n_obs
        );
        println!("AIC: {:.2}", arima_result.aic);
        println!("SSR: {:.2}", arima_result.ssr);
        if !arima_result.ar_coeffs.is_empty() {
            println!("AR coefficients: {:?}", arima_result.ar_coeffs);
        }
        if !arima_result.ma_coeffs.is_empty() {
            println!("MA coefficients: {:?}", arima_result.ma_coeffs);
        }
        println!();

        // Forecast with ARIMA
        println!("--- ARIMA Forecast (4 quarters ahead) ---");
        let forecast = forecast_arima(&dataset, "gdp", 1, 1, 0, 4)?;
        println!("Horizon: {} quarters", forecast.horizon);
        println!("Quarter    Forecast");
        println!("-------  ----------");
        for h in 0..4 {
            let quarter = format!("{}Q{}", 2024 + h / 4, 1 + h % 4);
            println!("{:<7}  {:>10.1}", quarter, forecast.forecast[h]);
        }
        println!();

        // Holt-Winters Exponential Smoothing
        println!("--- Holt-Winters Exponential Smoothing ---");
        let hw_config = HoltWintersConfig {
            seasonal: SeasonalType::Additive,
            period: 4, // quarterly data
            ..Default::default()
        };

        let hw_result = holt_winters(&gdp_values, hw_config)?;
        println!("Smoothing parameters:");
        println!("  Alpha (level):    {:.4}", hw_result.alpha);
        if let Some(beta) = hw_result.beta {
            println!("  Beta (trend):     {:.4}", beta);
        }
        if let Some(gamma) = hw_result.gamma {
            println!("  Gamma (seasonal): {:.4}", gamma);
        }
        println!();

        // Holt-Winters Forecast
        println!("--- Holt-Winters Forecast (4 quarters ahead) ---");
        let hw_forecast = holt_winters_forecast(&hw_result, 4)?;
        println!("Quarter    Forecast");
        println!("-------  ----------");
        for h in 0..4 {
            let quarter = format!("{}Q{}", 2024 + h / 4, 1 + h % 4);
            println!("{:<7}  {:>10.1}", quarter, hw_forecast[h]);
        }
        println!();

        // Compare forecasts
        println!("--- Forecast Comparison ---");
        println!("{:<10} {:>12} {:>15}", "Quarter", "ARIMA", "Holt-Winters");
        println!("{:-<10} {:-<12} {:-<15}", "", "", "");
        for h in 0..4 {
            let quarter = format!("{}Q{}", 2024 + h / 4, 1 + h % 4);
            println!(
                "{:<10} {:>12.1} {:>15.1}",
                quarter, forecast.forecast[h], hw_forecast[h]
            );
        }
    }

    #[cfg(not(feature = "forecasting"))]
    {
        println!("Note: ARIMA and Holt-Winters require the 'forecasting' feature.");
        println!("Run with: cargo run -p p2a-core --example time_series --features forecasting");
    }

    Ok(())
}
