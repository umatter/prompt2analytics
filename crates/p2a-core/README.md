# p2a-core

Pure Rust econometrics, statistics, and machine learning library for data analysis.

## Overview

`p2a-core` provides a comprehensive set of statistical and econometric methods implemented in pure Rust. It is the analytics engine that powers the prompt2analytics ecosystem (CLI, MCP server, and GUI).

## Features

The crate is organized into optional features to reduce compile times and binary size:

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `visualization` | Charts, plots, and heatmaps | plotters, plotlars, image, base64 |
| `forecasting` | ARIMA, MSTL, ETS, and time series forecasting | arima, augurs-*, stlrs |
| `file-formats` | Excel file support (xlsx, xls, xlsb, ods) | calamine |
| `database` | SQLite and DuckDB connectivity | rusqlite, duckdb |
| `spectral-analysis` | FFT-based spectral density estimation | rustfft |
| `full` | All features (default) | All of the above |

### Feature Usage

```toml
# Full functionality (default)
[dependencies]
p2a-core = "0.1"

# Minimal - core statistics and regression only
[dependencies]
p2a-core = { version = "0.1", default-features = false }

# Selected features
[dependencies]
p2a-core = { version = "0.1", default-features = false, features = ["visualization"] }
```

## Quick Start

```rust
use p2a_core::{Dataset, run_ols, CovarianceType};
use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a dataset
    let df = df! {
        "y" => [2.1, 4.2, 5.8, 8.1, 9.9],
        "x1" => [1.0, 2.0, 3.0, 4.0, 5.0],
        "x2" => [0.5, 1.0, 1.5, 2.0, 2.5],
    }?;
    let dataset = Dataset::new(df);

    // Run OLS regression with robust standard errors
    let result = run_ols(
        &dataset,
        "y",
        &["x1", "x2"],
        true,  // intercept
        CovarianceType::HC1,
    )?;

    println!("{}", result);
    Ok(())
}
```

## Module Overview

### Core Statistics (`stats`)
- Descriptive statistics and correlation matrices
- Hypothesis tests: t-test, ANOVA, chi-squared, Fisher's exact, Wilcoxon, etc.
- Normality tests: Shapiro-Wilk, Kolmogorov-Smirnov
- ACF/PACF for time series analysis

### Regression (`regression`)
- OLS with HC0-HC3 heteroskedasticity-robust standard errors
- Clustered standard errors
- Diagnostics: Jarque-Bera, Breusch-Pagan, Durbin-Watson, VIF
- Nonlinear least squares, LOESS, quantile regression
- GLS and HAC (Newey-West) standard errors

### Econometrics (`econometrics`)
- Panel data: Fixed effects, random effects, Hausman test, dynamic GMM
- Instrumental variables: 2SLS with first-stage diagnostics
- Discrete choice: Logit, probit, multinomial, ordered, mixed logit
- Difference-in-differences: Standard and staggered (Callaway-Sant'Anna)
- Regression discontinuity: Sharp and fuzzy RD
- Treatment effects: IPW, doubly robust, CBPS, TMLE
- Survival analysis: Kaplan-Meier, Cox PH, AFT

### Machine Learning (`ml`)
- Clustering: K-means, DBSCAN, hierarchical, GMM, spectral
- Dimensionality reduction: PCA, t-SNE, MDS
- Trees: Random forest, causal forest, BART

### Forecasting (`forecasting`) [requires `forecasting` feature]
- ARIMA and MSTL decomposition
- Holt-Winters exponential smoothing
- Kalman filter and structural time series
- Changepoint detection
- GARCH volatility modeling

### Visualization (`visualization`) [requires `visualization` feature]
- Static charts: histogram, scatter, box plot, line chart
- Heatmaps for correlation matrices
- Econometric plots: event study, coefficient plot, IRF

### Data Loading (`data`)
- CSV, Parquet, Stata DTA, SAS7BDAT
- Excel (xlsx, xls, xlsb, ods) [requires `file-formats` feature]
- SQLite and DuckDB queries [requires `database` feature]

## API Design

All regression functions use a **column-based API** (not formula-based):

```rust
// Column-based API (this crate)
run_ols(&dataset, "y", &["x1", "x2"], true, CovarianceType::HC1)

// NOT formula-based like R
// run_ols("y ~ x1 + x2")  // This is NOT supported
```

### LinearEstimator Trait

All estimators implement the `LinearEstimator` trait for consistent output:

```rust
pub trait LinearEstimator {
    fn coefficients(&self) -> &Array1<f64>;
    fn std_errors(&self) -> &Array1<f64>;
    fn t_values(&self) -> Array1<f64>;
    fn p_values(&self) -> Array1<f64>;
    fn residuals(&self) -> Array1<f64>;
    fn n_obs(&self) -> usize;
    fn df(&self) -> usize;
}
```

## Export Formats

Results can be exported to multiple formats:

```rust
use p2a_core::export::{LatexTableBuilder, MarkdownTableBuilder, HtmlTableBuilder, CsvExport};

// LaTeX for publication
let latex = LatexTableBuilder::new(&result).build();

// Markdown for documentation
let md = MarkdownTableBuilder::new(&result).build();

// HTML for web display
let html = HtmlTableBuilder::new(&result).build();

// CSV for data processing
let csv = result.to_csv();
```

## Requirements

- Rust 1.85+ (Edition 2024)
- Linux/macOS/Windows

### System Dependencies

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install libopenblas-dev
```

**macOS:**
```bash
brew install openblas
```

## License

MIT OR Apache-2.0
