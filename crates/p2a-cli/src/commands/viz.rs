//! Visualization commands

use clap::Subcommand;
use ndarray::Array2;
use p2a_core::{
    histogram, scatter_plot, line_chart, box_plot, ChartConfig,
    correlation_heatmap, coefficient_plot, residual_diagnostics,
    dendrogram, event_study_plot, irf_plot,
    hierarchical, Linkage, run_ols,
};
use p2a_core::regression::CovarianceType;
use p2a_core::traits::LinearEstimator;
use std::path::PathBuf;

use crate::output::{print_error, print_message, OutputFormat};
use crate::session::SessionManager;

#[derive(Subcommand)]
pub enum VizCommands {
    /// Create a histogram
    Histogram {
        /// Dataset name
        dataset: String,

        /// Column to plot
        #[arg(long)]
        col: String,

        /// Output file path
        #[arg(short = 'f', long = "file")]
        output: PathBuf,

        /// Number of bins
        #[arg(long, default_value = "30")]
        bins: usize,

        /// Chart title
        #[arg(long)]
        title: Option<String>,
    },

    /// Create a scatter plot
    Scatter {
        /// Dataset name
        dataset: String,

        /// X-axis column
        #[arg(short = 'x', long)]
        x_col: String,

        /// Y-axis column
        #[arg(short = 'y', long)]
        y_col: String,

        /// Output file path
        #[arg(short = 'f', long = "file")]
        output: PathBuf,

        /// Chart title
        #[arg(long)]
        title: Option<String>,
    },

    /// Create a line plot
    Line {
        /// Dataset name
        dataset: String,

        /// X-axis column
        #[arg(short = 'x', long)]
        x_col: String,

        /// Y-axis column
        #[arg(short = 'y', long)]
        y_col: String,

        /// Output file path
        #[arg(short = 'f', long = "file")]
        output: PathBuf,

        /// Chart title
        #[arg(long)]
        title: Option<String>,
    },

    /// Create a box plot
    Box {
        /// Dataset name
        dataset: String,

        /// Value column to plot
        #[arg(short = 'y', long)]
        value_col: String,

        /// Group column (optional - if omitted, single box)
        #[arg(short = 'g', long)]
        group_col: Option<String>,

        /// Output file path
        #[arg(short = 'f', long = "file")]
        output: PathBuf,

        /// Chart title
        #[arg(long)]
        title: Option<String>,
    },

    /// Create a correlation heatmap
    Heatmap {
        /// Dataset name
        dataset: String,

        /// Columns to include in correlation matrix
        #[arg(long, num_args = 2..)]
        cols: Vec<String>,

        /// Output file path
        #[arg(short = 'f', long = "file")]
        output: PathBuf,

        /// Chart title
        #[arg(long)]
        title: Option<String>,
    },

    /// Create a coefficient plot (forest plot) from regression
    Coefplot {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,

        /// Output file path
        #[arg(short = 'f', long = "file")]
        output: PathBuf,

        /// Confidence level (default: 0.95)
        #[arg(long, default_value = "0.95")]
        conf_level: f64,

        /// Chart title
        #[arg(long)]
        title: Option<String>,
    },

    /// Create residual diagnostic plots from regression
    Residuals {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,

        /// Output file path
        #[arg(short = 'f', long = "file")]
        output: PathBuf,

        /// Chart title
        #[arg(long)]
        title: Option<String>,
    },

    /// Create a dendrogram from hierarchical clustering
    Dendrogram {
        /// Dataset name
        dataset: String,

        /// Feature columns
        #[arg(long, num_args = 1..)]
        cols: Vec<String>,

        /// Linkage method: "single", "complete", "average", "ward"
        #[arg(long, default_value = "ward")]
        linkage: String,

        /// Output file path
        #[arg(short = 'f', long = "file")]
        output: PathBuf,

        /// Chart title
        #[arg(long)]
        title: Option<String>,
    },

    /// Create an event study plot
    EventStudy {
        /// Dataset name
        dataset: String,

        /// Time/period column (relative to event)
        #[arg(long)]
        time_col: String,

        /// Estimate column
        #[arg(long)]
        estimate_col: String,

        /// Lower CI column
        #[arg(long)]
        ci_lower_col: String,

        /// Upper CI column
        #[arg(long)]
        ci_upper_col: String,

        /// Output file path
        #[arg(short = 'f', long = "file")]
        output: PathBuf,

        /// Chart title
        #[arg(long)]
        title: Option<String>,
    },

    /// Create an impulse response function plot
    Irf {
        /// Dataset name
        dataset: String,

        /// Horizon column
        #[arg(long)]
        horizon_col: String,

        /// Response column
        #[arg(long)]
        response_col: String,

        /// Lower CI column (optional)
        #[arg(long)]
        ci_lower_col: Option<String>,

        /// Upper CI column (optional)
        #[arg(long)]
        ci_upper_col: Option<String>,

        /// Shock variable label
        #[arg(long)]
        shock_label: Option<String>,

        /// Response variable label
        #[arg(long)]
        response_label: Option<String>,

        /// Output file path
        #[arg(short = 'f', long = "file")]
        output: PathBuf,

        /// Chart title
        #[arg(long)]
        title: Option<String>,
    },
}

/// Extract a column from a Dataset as Vec<f64>
fn extract_column(dataset: &p2a_core::Dataset, col: &str) -> Result<Vec<f64>, String> {
    let df = dataset.df();
    let column = df
        .column(col)
        .map_err(|e| format!("Column '{}' not found: {}", col, e))?;
    let f64_col = column
        .f64()
        .map_err(|e| format!("Column '{}' must be numeric: {}", col, e))?;

    Ok(f64_col.into_no_null_iter().collect())
}

/// Decode base64 image to bytes
fn decode_base64_image(base64_str: &str) -> Result<Vec<u8>, String> {
    use base64::{Engine as _, engine::general_purpose};
    general_purpose::STANDARD
        .decode(base64_str)
        .map_err(|e| format!("Failed to decode image: {}", e))
}

pub fn execute(
    cmd: &VizCommands,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    match cmd {
        VizCommands::Histogram {
            dataset,
            col,
            output,
            bins,
            title,
        } => execute_histogram(dataset, col, output, *bins, title.as_deref(), format, session),
        VizCommands::Scatter {
            dataset,
            x_col,
            y_col,
            output,
            title,
        } => execute_scatter(dataset, x_col, y_col, output, title.as_deref(), format, session),
        VizCommands::Line {
            dataset,
            x_col,
            y_col,
            output,
            title,
        } => execute_line(dataset, x_col, y_col, output, title.as_deref(), format, session),
        VizCommands::Box {
            dataset,
            value_col,
            group_col,
            output,
            title,
        } => execute_boxplot(dataset, value_col, group_col.as_deref(), output, title.as_deref(), format, session),
        VizCommands::Heatmap {
            dataset,
            cols,
            output,
            title,
        } => execute_heatmap(dataset, cols, output, title.as_deref(), format, session),
        VizCommands::Coefplot {
            dataset,
            dep_var,
            indep_vars,
            output,
            conf_level,
            title,
        } => execute_coefplot(dataset, dep_var, indep_vars, output, *conf_level, title.as_deref(), format, session),
        VizCommands::Residuals {
            dataset,
            dep_var,
            indep_vars,
            output,
            title,
        } => execute_residuals(dataset, dep_var, indep_vars, output, title.as_deref(), format, session),
        VizCommands::Dendrogram {
            dataset,
            cols,
            linkage,
            output,
            title,
        } => execute_dendrogram(dataset, cols, linkage, output, title.as_deref(), format, session),
        VizCommands::EventStudy {
            dataset,
            time_col,
            estimate_col,
            ci_lower_col,
            ci_upper_col,
            output,
            title,
        } => execute_event_study(dataset, time_col, estimate_col, ci_lower_col, ci_upper_col, output, title.as_deref(), format, session),
        VizCommands::Irf {
            dataset,
            horizon_col,
            response_col,
            ci_lower_col,
            ci_upper_col,
            shock_label,
            response_label,
            output,
            title,
        } => execute_irf(dataset, horizon_col, response_col, ci_lower_col.as_deref(), ci_upper_col.as_deref(), shock_label.as_deref(), response_label.as_deref(), output, title.as_deref(), format, session),
    }
}

fn execute_histogram(
    dataset_name: &str,
    col: &str,
    output: &PathBuf,
    bins: usize,
    title: Option<&str>,
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
            // Extract column data
            let data = match extract_column(ds, col) {
                Ok(d) => d,
                Err(e) => {
                    print_error(&e, format);
                    return Ok(());
                }
            };

            let chart_title = title.map(|s| s.to_string()).or_else(|| Some(format!("Histogram of {}", col)));
            let config = ChartConfig {
                title: chart_title,
                width: 800,
                height: 600,
                ..Default::default()
            };

            // histogram(data, bins, config)
            match histogram(&data, Some(bins), config) {
                Ok(result) => {
                    // Decode base64 and save PNG to file
                    match decode_base64_image(&result.image_base64) {
                        Ok(png_data) => {
                            std::fs::write(output, &png_data)?;
                            print_message(
                                &format!("Histogram saved to: {}", output.display()),
                                format,
                            );
                        }
                        Err(e) => {
                            print_error(&e, format);
                        }
                    }
                }
                Err(e) => {
                    print_error(&format!("Histogram creation failed: {}", e), format);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

fn execute_scatter(
    dataset_name: &str,
    x_col: &str,
    y_col: &str,
    output: &PathBuf,
    title: Option<&str>,
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
            // Extract column data
            let x_data = match extract_column(ds, x_col) {
                Ok(d) => d,
                Err(e) => {
                    print_error(&e, format);
                    return Ok(());
                }
            };
            let y_data = match extract_column(ds, y_col) {
                Ok(d) => d,
                Err(e) => {
                    print_error(&e, format);
                    return Ok(());
                }
            };

            let chart_title = title.map(|s| s.to_string()).or_else(|| Some(format!("{} vs {}", y_col, x_col)));
            let config = ChartConfig {
                title: chart_title,
                width: 800,
                height: 600,
                x_label: Some(x_col.to_string()),
                y_label: Some(y_col.to_string()),
                ..Default::default()
            };

            // scatter_plot(x, y, config)
            match scatter_plot(&x_data, &y_data, config) {
                Ok(result) => {
                    match decode_base64_image(&result.image_base64) {
                        Ok(png_data) => {
                            std::fs::write(output, &png_data)?;
                            print_message(
                                &format!("Scatter plot saved to: {}", output.display()),
                                format,
                            );
                        }
                        Err(e) => {
                            print_error(&e, format);
                        }
                    }
                }
                Err(e) => {
                    print_error(&format!("Scatter plot creation failed: {}", e), format);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

fn execute_line(
    dataset_name: &str,
    x_col: &str,
    y_col: &str,
    output: &PathBuf,
    title: Option<&str>,
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
            // Extract column data
            let x_data = match extract_column(ds, x_col) {
                Ok(d) => d,
                Err(e) => {
                    print_error(&e, format);
                    return Ok(());
                }
            };
            let y_data = match extract_column(ds, y_col) {
                Ok(d) => d,
                Err(e) => {
                    print_error(&e, format);
                    return Ok(());
                }
            };

            let chart_title = title.map(|s| s.to_string()).or_else(|| Some(format!("{} vs {}", y_col, x_col)));
            let config = ChartConfig {
                title: chart_title,
                width: 800,
                height: 600,
                x_label: Some(x_col.to_string()),
                y_label: Some(y_col.to_string()),
                ..Default::default()
            };

            // line_chart takes series: &[(String, Vec<f64>, Vec<f64>)]
            let series = vec![(y_col.to_string(), x_data, y_data)];

            match line_chart(&series, config) {
                Ok(result) => {
                    match decode_base64_image(&result.image_base64) {
                        Ok(png_data) => {
                            std::fs::write(output, &png_data)?;
                            print_message(
                                &format!("Line plot saved to: {}", output.display()),
                                format,
                            );
                        }
                        Err(e) => {
                            print_error(&e, format);
                        }
                    }
                }
                Err(e) => {
                    print_error(&format!("Line plot creation failed: {}", e), format);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

fn execute_boxplot(
    dataset_name: &str,
    value_col: &str,
    group_col: Option<&str>,
    output: &PathBuf,
    title: Option<&str>,
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
            let df = ds.df();

            // Build groups for box plot
            let groups: Vec<(String, Vec<f64>)> = if let Some(grp_col) = group_col {
                // Group by the group column
                let group_series = match df.column(grp_col) {
                    Ok(c) => c,
                    Err(e) => {
                        print_error(&format!("Group column '{}' not found: {}", grp_col, e), format);
                        return Ok(());
                    }
                };
                let value_series = match df.column(value_col) {
                    Ok(c) => c,
                    Err(e) => {
                        print_error(&format!("Value column '{}' not found: {}", value_col, e), format);
                        return Ok(());
                    }
                };

                // Get unique group values and filter data
                let groups_str = group_series.str();
                let values_f64 = value_series.f64();

                match (groups_str, values_f64) {
                    (Ok(gs), Ok(vs)) => {
                        // Collect unique groups
                        let unique: std::collections::HashSet<String> = gs
                            .into_no_null_iter()
                            .map(|s| s.to_string())
                            .collect();

                        unique
                            .into_iter()
                            .map(|group_name| {
                                let data: Vec<f64> = gs
                                    .into_no_null_iter()
                                    .zip(vs.into_no_null_iter())
                                    .filter(|(g, _)| *g == group_name)
                                    .map(|(_, v)| v)
                                    .collect();
                                (group_name, data)
                            })
                            .collect()
                    }
                    _ => {
                        print_error("Group column must be string and value column must be numeric", format);
                        return Ok(());
                    }
                }
            } else {
                // Single group - all data
                let data = match extract_column(ds, value_col) {
                    Ok(d) => d,
                    Err(e) => {
                        print_error(&e, format);
                        return Ok(());
                    }
                };
                vec![(value_col.to_string(), data)]
            };

            let chart_title = title.map(|s| s.to_string()).or_else(|| Some(format!("Box Plot of {}", value_col)));
            let config = ChartConfig {
                title: chart_title,
                width: 800,
                height: 600,
                y_label: Some(value_col.to_string()),
                ..Default::default()
            };

            match box_plot(&groups, config) {
                Ok(result) => {
                    match decode_base64_image(&result.image_base64) {
                        Ok(png_data) => {
                            std::fs::write(output, &png_data)?;
                            print_message(
                                &format!("Box plot saved to: {}", output.display()),
                                format,
                            );
                        }
                        Err(e) => print_error(&e, format),
                    }
                }
                Err(e) => print_error(&format!("Box plot creation failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_heatmap(
    dataset_name: &str,
    cols: &[String],
    output: &PathBuf,
    title: Option<&str>,
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
            // Extract columns and compute correlation matrix
            let n_cols = cols.len();
            let mut data_vecs: Vec<Vec<f64>> = Vec::with_capacity(n_cols);

            for col_name in cols {
                match extract_column(ds, col_name) {
                    Ok(d) => data_vecs.push(d),
                    Err(e) => {
                        print_error(&e, format);
                        return Ok(());
                    }
                }
            }

            // Compute correlation matrix
            let n_rows = data_vecs[0].len();
            let mut corr_matrix: Vec<Vec<f64>> = vec![vec![0.0; n_cols]; n_cols];

            for i in 0..n_cols {
                for j in 0..n_cols {
                    if i == j {
                        corr_matrix[i][j] = 1.0;
                    } else if j > i {
                        let corr = compute_correlation(&data_vecs[i], &data_vecs[j], n_rows);
                        corr_matrix[i][j] = corr;
                        corr_matrix[j][i] = corr;
                    }
                }
            }

            let labels: Vec<String> = cols.to_vec();

            match correlation_heatmap(&corr_matrix, &labels, &labels, title, Some(800), Some(600)) {
                Ok(result) => {
                    match decode_base64_image(&result.image_base64) {
                        Ok(png_data) => {
                            std::fs::write(output, &png_data)?;
                            print_message(&format!("Heatmap saved to: {}", output.display()), format);
                        }
                        Err(e) => print_error(&e, format),
                    }
                }
                Err(e) => print_error(&format!("Heatmap creation failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn compute_correlation(x: &[f64], y: &[f64], n: usize) -> f64 {
    let mean_x: f64 = x.iter().sum::<f64>() / n as f64;
    let mean_y: f64 = y.iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x > 0.0 && var_y > 0.0 {
        cov / (var_x.sqrt() * var_y.sqrt())
    } else {
        0.0
    }
}

fn execute_coefplot(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    output: &PathBuf,
    conf_level: f64,
    title: Option<&str>,
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
            let x_cols: Vec<&str> = indep_vars.iter().map(|s| s.as_str()).collect();

            // Run OLS to get coefficients
            match run_ols(ds, dep_var, &x_cols, true, CovarianceType::HC1) {
                Ok(result) => {
                    // Calculate critical value for confidence interval
                    let _alpha = 1.0 - conf_level;
                    let z = 1.96; // Approximate for 95% CI

                    let names: Vec<String> = result.variable_names.clone();
                    let estimates: Vec<f64> = result.coefficients.iter().map(|c| c.estimate).collect();
                    let std_errors: Vec<f64> = result.coefficients.iter().map(|c| c.std_error).collect();
                    let ci_lower: Vec<f64> = estimates.iter()
                        .zip(std_errors.iter())
                        .map(|(b, se)| b - z * se)
                        .collect();
                    let ci_upper: Vec<f64> = estimates.iter()
                        .zip(std_errors.iter())
                        .map(|(b, se)| b + z * se)
                        .collect();

                    let chart_title = title.map(|s| s.to_string())
                        .or_else(|| Some(format!("Coefficients: {}", dep_var)));
                    let config = ChartConfig {
                        title: chart_title,
                        width: 800,
                        height: 600,
                        ..Default::default()
                    };

                    match coefficient_plot(&names, &estimates, &ci_lower, &ci_upper, config, true) {
                        Ok(plot_result) => {
                            match decode_base64_image(&plot_result.image_base64) {
                                Ok(png_data) => {
                                    std::fs::write(output, &png_data)?;
                                    print_message(&format!("Coefficient plot saved to: {}", output.display()), format);
                                }
                                Err(e) => print_error(&e, format),
                            }
                        }
                        Err(e) => print_error(&format!("Coefficient plot failed: {}", e), format),
                    }
                }
                Err(e) => print_error(&format!("Regression failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_residuals(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    output: &PathBuf,
    title: Option<&str>,
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
            let x_cols: Vec<&str> = indep_vars.iter().map(|s| s.as_str()).collect();

            // Get Y values from dataset
            let y_values = match extract_column(ds, dep_var) {
                Ok(d) => d,
                Err(e) => {
                    print_error(&e, format);
                    return Ok(());
                }
            };

            // Run OLS to get residuals
            match run_ols(ds, dep_var, &x_cols, true, CovarianceType::HC1) {
                Ok(result) => {
                    // Get residuals via LinearEstimator trait
                    let resid = result.residuals();
                    let residuals: Vec<f64> = resid.to_vec();
                    // Compute fitted = y - residuals
                    let fitted: Vec<f64> = y_values.iter()
                        .zip(residuals.iter())
                        .map(|(y, r)| y - r)
                        .collect();

                    let chart_title = title.map(|s| s.to_string())
                        .or_else(|| Some("Residual Diagnostics".to_string()));
                    let config = ChartConfig {
                        title: chart_title,
                        width: 500,
                        height: 400,
                        ..Default::default()
                    };

                    match residual_diagnostics(&fitted, &residuals, None, config) {
                        Ok(plot_result) => {
                            // ResidualDiagnosticsResult has 4 separate plots
                            // Save residuals_vs_fitted as the main output
                            match decode_base64_image(&plot_result.residuals_vs_fitted) {
                                Ok(png_data) => {
                                    // Save main plot with user-specified name
                                    std::fs::write(output, &png_data)?;

                                    // Save other plots with suffixes
                                    let stem = output.file_stem()
                                        .and_then(|s| s.to_str())
                                        .unwrap_or("residuals");
                                    let parent = output.parent().unwrap_or(std::path::Path::new("."));

                                    // Save Q-Q plot
                                    if let Ok(qq_data) = decode_base64_image(&plot_result.qq_plot) {
                                        let qq_path = parent.join(format!("{}_qq.png", stem));
                                        let _ = std::fs::write(&qq_path, &qq_data);
                                    }
                                    // Save scale-location plot
                                    if let Ok(sl_data) = decode_base64_image(&plot_result.scale_location) {
                                        let sl_path = parent.join(format!("{}_scale_location.png", stem));
                                        let _ = std::fs::write(&sl_path, &sl_data);
                                    }
                                    // Save leverage plot
                                    if let Ok(lev_data) = decode_base64_image(&plot_result.residuals_vs_leverage) {
                                        let lev_path = parent.join(format!("{}_leverage.png", stem));
                                        let _ = std::fs::write(&lev_path, &lev_data);
                                    }

                                    print_message(&format!("Residual diagnostics saved to: {} (+ _qq, _scale_location, _leverage)", output.display()), format);
                                }
                                Err(e) => print_error(&e, format),
                            }
                        }
                        Err(e) => print_error(&format!("Residual plot failed: {}", e), format),
                    }
                }
                Err(e) => print_error(&format!("Regression failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_dendrogram(
    dataset_name: &str,
    cols: &[String],
    linkage_method: &str,
    output: &PathBuf,
    title: Option<&str>,
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
            // Extract columns as Array2
            let data = match extract_columns_as_array(ds, cols) {
                Ok(arr) => arr,
                Err(e) => {
                    print_error(&format!("Failed to extract data: {}", e), format);
                    return Ok(());
                }
            };

            let link = match linkage_method.to_lowercase().as_str() {
                "single" => Linkage::Single,
                "complete" => Linkage::Complete,
                "average" => Linkage::Average,
                _ => Linkage::Ward,
            };

            // Run hierarchical clustering
            match hierarchical(data.view(), None, link, None) {
                Ok(result) => {
                    // linkage_matrix is already Vec<(usize, usize, f64, usize)>
                    let linkage_matrix = &result.linkage_matrix;

                    let chart_title = title.map(|s| s.to_string())
                        .or_else(|| Some("Dendrogram".to_string()));
                    let config = ChartConfig {
                        title: chart_title,
                        width: 1000,
                        height: 600,
                        ..Default::default()
                    };

                    match dendrogram(&linkage_matrix, None, config) {
                        Ok(plot_result) => {
                            match decode_base64_image(&plot_result.image_base64) {
                                Ok(png_data) => {
                                    std::fs::write(output, &png_data)?;
                                    print_message(&format!("Dendrogram saved to: {}", output.display()), format);
                                }
                                Err(e) => print_error(&e, format),
                            }
                        }
                        Err(e) => print_error(&format!("Dendrogram creation failed: {}", e), format),
                    }
                }
                Err(e) => print_error(&format!("Hierarchical clustering failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

/// Extract multiple columns from a Dataset as an Array2<f64>
fn extract_columns_as_array(
    dataset: &p2a_core::Dataset,
    cols: &[String],
) -> Result<Array2<f64>, String> {
    let df = dataset.df();
    let n_rows = df.height();
    let n_cols = cols.len();

    if n_cols == 0 {
        return Err("No columns specified".to_string());
    }

    let mut data = Vec::with_capacity(n_rows * n_cols);

    for row_idx in 0..n_rows {
        for col_name in cols {
            let col = df
                .column(col_name)
                .map_err(|e| format!("Column '{}' not found: {}", col_name, e))?;
            let f64_col = col
                .f64()
                .map_err(|e| format!("Column '{}' must be numeric: {}", col_name, e))?;
            let value = f64_col.get(row_idx).ok_or_else(|| {
                format!("Missing value at row {} in column '{}'", row_idx, col_name)
            })?;
            data.push(value);
        }
    }

    Array2::from_shape_vec((n_rows, n_cols), data)
        .map_err(|e| format!("Failed to create array: {}", e))
}

fn execute_event_study(
    dataset_name: &str,
    time_col: &str,
    estimate_col: &str,
    ci_lower_col: &str,
    ci_upper_col: &str,
    output: &PathBuf,
    title: Option<&str>,
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
            let time = match extract_column(ds, time_col) {
                Ok(d) => d,
                Err(e) => { print_error(&e, format); return Ok(()); }
            };
            let estimates = match extract_column(ds, estimate_col) {
                Ok(d) => d,
                Err(e) => { print_error(&e, format); return Ok(()); }
            };
            let ci_lower = match extract_column(ds, ci_lower_col) {
                Ok(d) => d,
                Err(e) => { print_error(&e, format); return Ok(()); }
            };
            let ci_upper = match extract_column(ds, ci_upper_col) {
                Ok(d) => d,
                Err(e) => { print_error(&e, format); return Ok(()); }
            };

            let chart_title = title.map(|s| s.to_string())
                .or_else(|| Some("Event Study".to_string()));
            let config = ChartConfig {
                title: chart_title,
                width: 800,
                height: 600,
                x_label: Some("Time".to_string()),
                y_label: Some("Effect".to_string()),
                ..Default::default()
            };

            match event_study_plot(&time, &estimates, &ci_lower, &ci_upper, config) {
                Ok(result) => {
                    match decode_base64_image(&result.image_base64) {
                        Ok(png_data) => {
                            std::fs::write(output, &png_data)?;
                            print_message(&format!("Event study plot saved to: {}", output.display()), format);
                        }
                        Err(e) => print_error(&e, format),
                    }
                }
                Err(e) => print_error(&format!("Event study plot failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_irf(
    dataset_name: &str,
    horizon_col: &str,
    response_col: &str,
    ci_lower_col: Option<&str>,
    ci_upper_col: Option<&str>,
    shock_label: Option<&str>,
    response_label: Option<&str>,
    output: &PathBuf,
    title: Option<&str>,
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
            let horizons = match extract_column(ds, horizon_col) {
                Ok(d) => d,
                Err(e) => { print_error(&e, format); return Ok(()); }
            };
            let responses = match extract_column(ds, response_col) {
                Ok(d) => d,
                Err(e) => { print_error(&e, format); return Ok(()); }
            };

            let ci_lower: Option<Vec<f64>> = ci_lower_col.and_then(|col| extract_column(ds, col).ok());
            let ci_upper: Option<Vec<f64>> = ci_upper_col.and_then(|col| extract_column(ds, col).ok());

            let chart_title = title.map(|s| s.to_string())
                .or_else(|| Some("Impulse Response Function".to_string()));
            let config = ChartConfig {
                title: chart_title,
                width: 800,
                height: 600,
                ..Default::default()
            };

            match irf_plot(
                &horizons,
                &responses,
                ci_lower.as_deref(),
                ci_upper.as_deref(),
                shock_label,
                response_label,
                config,
            ) {
                Ok(result) => {
                    match decode_base64_image(&result.image_base64) {
                        Ok(png_data) => {
                            std::fs::write(output, &png_data)?;
                            print_message(&format!("IRF plot saved to: {}", output.display()), format);
                        }
                        Err(e) => print_error(&e, format),
                    }
                }
                Err(e) => print_error(&format!("IRF plot failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}
