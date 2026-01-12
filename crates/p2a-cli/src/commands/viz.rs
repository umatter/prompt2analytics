//! Visualization commands

use clap::Subcommand;
use p2a_core::{histogram, scatter_plot, line_chart, ChartConfig};
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
