//! Visualization commands

use clap::Subcommand;
use p2a_core::data::columns::{ColumnRequest, Columns, MissingSummary, NaPolicy, numeric_column};
use p2a_core::linalg::solve::least_squares;
use p2a_core::regression::CovarianceType;
use p2a_core::traits::LinearEstimator;
use p2a_core::visualization::charts::residual_diagnostics_with;
use p2a_core::visualization::{
    InteractiveConfig, histogram_interactive, line_interactive, scatter_interactive,
};
use p2a_core::{
    ChartConfig, Linkage, box_plot, coefficient_plot, correlation_heatmap, dendrogram,
    event_study_plot, hierarchical, histogram, irf_plot, line_chart, run_ols, scatter_plot,
};
use std::path::PathBuf;

use super::causal::{print_result_notes, with_result_notes};
use crate::output::{OutputFormat, missing_json, print_error, print_message, print_missing_note};
use crate::session::SessionManager;

#[derive(Subcommand)]
pub enum VizCommands {
    /// Create a histogram
    #[command(after_help = "\
EXAMPLES:
    p2a --session s.json viz histogram mydata --col income -f hist.png --bins 50
")]
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
    #[command(after_help = "\
EXAMPLES:
    p2a --session s.json viz scatter mydata -x age -y income -f scatter.png --title 'Age vs Income'
")]
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
    #[command(after_help = "\
EXAMPLES:
    p2a --session s.json viz line mydata -x date -y price -f timeseries.png
")]
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
    #[command(after_help = "\
EXAMPLES:
    # Single box
    p2a --session s.json viz box mydata -y score -f boxplot.png

    # Grouped by category
    p2a --session s.json viz box mydata -y score -g treatment -f boxplot.png
")]
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
    #[command(after_help = "\
EXAMPLES:
    p2a --session s.json viz heatmap mydata --cols x1 x2 x3 x4 -f corr.png
")]
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
    #[command(after_help = "\
EXAMPLES:
    p2a --session s.json viz coefplot mydata -y outcome -x x1 x2 x3 -f coef.png
")]
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
    #[command(after_help = "\
EXAMPLES:
    # Creates 4 diagnostic plots: residuals, Q-Q, scale-location, leverage
    p2a --session s.json viz residuals mydata -y outcome -x x1 x2 -f resid.png
")]
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
    #[command(after_help = "\
EXAMPLES:
    p2a --session s.json viz dendrogram mydata --cols x1 x2 x3 --linkage ward.D2 -f tree.png
")]
    Dendrogram {
        /// Dataset name
        dataset: String,

        /// Feature columns
        #[arg(long, num_args = 1..)]
        cols: Vec<String>,

        /// Linkage, as stats::hclust: single, complete, average (also upgma), ward.D2 (default;
        /// also ward), mcquitty (also weighted), ward.D, centroid or median. R's deprecated
        /// hclust(method = "ward") is ward.D
        #[arg(long, default_value = "ward.D2", value_parser = super::causal::option_arg::<Linkage>)]
        linkage: Linkage,

        /// Output file path
        #[arg(short = 'f', long = "file")]
        output: PathBuf,

        /// Chart title
        #[arg(long)]
        title: Option<String>,
    },

    /// Create an event study plot
    #[command(after_help = "\
EXAMPLES:
    p2a --session s.json viz event-study results --time-col period \\
        --estimate-col estimate --ci-lower-col ci_low --ci-upper-col ci_high -f event.png
")]
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
    #[command(after_help = "\
EXAMPLES:
    p2a --session s.json viz irf irf_data --horizon-col h --response-col response \\
        --ci-lower-col lower --ci-upper-col upper -f irf.png
")]
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

    /// Create an interactive scatter plot (HTML output with Plotly)
    #[command(after_help = "\
EXAMPLES:
    # Interactive scatter with hover
    p2a --session s.json viz scatter-interactive mydata -x age -y income -f scatter.html

    # Color by group
    p2a --session s.json viz scatter-interactive mydata -x x -y y -g category -f plot.html
")]
    ScatterInteractive {
        /// Dataset name
        dataset: String,

        /// X-axis column
        #[arg(short = 'x', long)]
        x_col: String,

        /// Y-axis column
        #[arg(short = 'y', long)]
        y_col: String,

        /// Group column for separate traces (optional)
        #[arg(short = 'g', long)]
        group_col: Option<String>,

        /// Output HTML file path
        #[arg(short = 'f', long = "file")]
        output: PathBuf,

        /// Chart title
        #[arg(long)]
        title: Option<String>,
    },

    /// Create an interactive histogram (HTML output with Plotly)
    #[command(after_help = "\
EXAMPLES:
    p2a --session s.json viz histogram-interactive mydata --col income -f hist.html
")]
    HistogramInteractive {
        /// Dataset name
        dataset: String,

        /// Column to plot
        #[arg(long)]
        col: String,

        /// Group column for separate traces (optional)
        #[arg(short = 'g', long)]
        group_col: Option<String>,

        /// Output HTML file path
        #[arg(short = 'f', long = "file")]
        output: PathBuf,

        /// Chart title
        #[arg(long)]
        title: Option<String>,
    },

    /// Create an interactive line plot (HTML output with Plotly)
    #[command(after_help = "\
EXAMPLES:
    p2a --session s.json viz line-interactive mydata -x date -y price -f line.html
")]
    LineInteractive {
        /// Dataset name
        dataset: String,

        /// X-axis column
        #[arg(short = 'x', long)]
        x_col: String,

        /// Y-axis column
        #[arg(short = 'y', long)]
        y_col: String,

        /// Output HTML file path
        #[arg(short = 'f', long = "file")]
        output: PathBuf,

        /// Chart title
        #[arg(long)]
        title: Option<String>,
    },
}

/// "viz <command>, columns 'a', 'b'": the context of a too-few-rows error.
fn extraction_context(command: &str, names: &[&str]) -> String {
    let quoted: Vec<String> = names.iter().map(|n| format!("'{n}'")).collect();
    let noun = if names.len() == 1 {
        "column"
    } else {
        "columns"
    };
    format!("viz {command}, {noun} {}", quoted.join(", "))
}

/// `names` on the rows where none of them is missing (R `complete.cases`: a plot of several
/// columns drops a row missing in any of them). Int64 and the other numeric dtypes are
/// accepted; fewer than `min_rows` complete rows is an error naming `viz <command>` and the
/// columns.
fn complete_rows(
    dataset: &p2a_core::Dataset,
    names: &[&str],
    min_rows: usize,
    command: &str,
) -> Result<Columns, String> {
    ColumnRequest::new(dataset.df())
        .numerics(names)
        .na(NaPolicy::Omit)
        .min_rows(min_rows, &extraction_context(command, names))
        .extract()
        .map_err(|e| e.to_string())
}

/// The value column split by the group column's levels in R factor order (numbers in numeric
/// order, text in C order; rows missing either column dropped, as `boxplot(y ~ g)` does), or
/// one group named after `value_col` when `group_col` is None. The group column may be text,
/// integer, float, boolean or date.
fn box_groups(
    dataset: &p2a_core::Dataset,
    value_col: &str,
    group_col: Option<&str>,
) -> Result<(Vec<(String, Vec<f64>)>, MissingSummary), String> {
    let Some(group_col) = group_col else {
        let cols = complete_rows(dataset, &[value_col], 1, "box")?;
        let values = cols.numeric(value_col).to_vec();
        return Ok((
            vec![(value_col.to_string(), values)],
            cols.missing().clone(),
        ));
    };
    let cols = ColumnRequest::new(dataset.df())
        .numeric(value_col)
        .factor(group_col)
        .na(NaPolicy::Omit)
        .min_rows(1, &extraction_context("box", &[value_col, group_col]))
        .extract()
        .map_err(|e| e.to_string())?;
    let factor = cols.factor(group_col);
    let mut groups: Vec<(String, Vec<f64>)> = factor
        .levels
        .iter()
        .map(|level| (level.clone(), Vec::new()))
        .collect();
    for (&code, &value) in factor.codes.iter().zip(cols.numeric(value_col)) {
        groups[code as usize].1.push(value);
    }
    Ok((groups, cols.missing().clone()))
}

/// Pearson correlations of `cols` over the rows complete in all of them (R
/// `cor(use = "complete.obs")`), as a k x k matrix, what was dropped, and the indices of the
/// columns that are constant on those rows. Such a column's correlations with the other
/// columns are NaN (R's NA); its diagonal cell is 1, as in R.
fn complete_case_correlations(
    dataset: &p2a_core::Dataset,
    cols: &[String],
) -> Result<(Vec<Vec<f64>>, MissingSummary, Vec<usize>), String> {
    let names: Vec<&str> = cols.iter().map(String::as_str).collect();
    let columns = complete_rows(dataset, &names, 2, "heatmap")?;
    let k = names.len();
    let mut corr = vec![vec![0.0; k]; k];
    for i in 0..k {
        corr[i][i] = 1.0;
        for j in (i + 1)..k {
            let r = compute_correlation(columns.numeric(names[i]), columns.numeric(names[j]));
            corr[i][j] = r;
            corr[j][i] = r;
        }
    }
    // With one column there is no correlation to be NA.
    let constant = (0..k)
        .filter(|&i| k > 1 && is_constant(columns.numeric(names[i])))
        .collect();
    Ok((corr, columns.missing().clone(), constant))
}

/// "correlations with c are NA: it is constant on the 5 complete rows" for the columns at
/// `constant` (constant on the `n_complete` complete rows); `None` when there is none.
fn na_note(cols: &[String], constant: &[usize], n_complete: usize) -> Option<String> {
    let constant: Vec<&str> = constant
        .iter()
        .filter_map(|&i| cols.get(i).map(String::as_str))
        .collect();
    let pronoun = match constant.len() {
        0 => return None,
        1 => "it is",
        _ => "they are",
    };
    Some(format!(
        "correlations with {} are NA: {pronoun} constant on the {n_complete} complete rows",
        constant.join(", ")
    ))
}

/// Text: `msg`, then the dropped-rows note; JSON: one line `{"message": msg}`, with a
/// `"missing"` object when rows were dropped.
fn print_saved(msg: &str, missing: &MissingSummary, format: &OutputFormat) {
    match format {
        OutputFormat::Json => {
            let mut json = serde_json::json!({ "message": msg });
            if let (Some(summary), Some(object)) = (missing_json(missing), json.as_object_mut()) {
                object.insert("missing".to_string(), summary);
            }
            println!("{json}");
        }
        _ => {
            println!("{msg}");
            print_missing_note(missing, format);
        }
    }
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
    _quiet: bool,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    match cmd {
        VizCommands::Histogram {
            dataset,
            col,
            output,
            bins,
            title,
        } => execute_histogram(
            dataset,
            col,
            output,
            *bins,
            title.as_deref(),
            format,
            session,
        ),
        VizCommands::Scatter {
            dataset,
            x_col,
            y_col,
            output,
            title,
        } => execute_scatter(
            dataset,
            x_col,
            y_col,
            output,
            title.as_deref(),
            format,
            session,
        ),
        VizCommands::Line {
            dataset,
            x_col,
            y_col,
            output,
            title,
        } => execute_line(
            dataset,
            x_col,
            y_col,
            output,
            title.as_deref(),
            format,
            session,
        ),
        VizCommands::Box {
            dataset,
            value_col,
            group_col,
            output,
            title,
        } => execute_boxplot(
            dataset,
            value_col,
            group_col.as_deref(),
            output,
            title.as_deref(),
            format,
            session,
        ),
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
        } => execute_coefplot(
            dataset,
            dep_var,
            indep_vars,
            output,
            *conf_level,
            title.as_deref(),
            format,
            session,
        ),
        VizCommands::Residuals {
            dataset,
            dep_var,
            indep_vars,
            output,
            title,
        } => execute_residuals(
            dataset,
            dep_var,
            indep_vars,
            output,
            title.as_deref(),
            format,
            session,
        ),
        VizCommands::Dendrogram {
            dataset,
            cols,
            linkage,
            output,
            title,
        } => execute_dendrogram(
            dataset,
            cols,
            *linkage,
            output,
            title.as_deref(),
            format,
            session,
        ),
        VizCommands::EventStudy {
            dataset,
            time_col,
            estimate_col,
            ci_lower_col,
            ci_upper_col,
            output,
            title,
        } => execute_event_study(
            dataset,
            time_col,
            estimate_col,
            ci_lower_col,
            ci_upper_col,
            output,
            title.as_deref(),
            format,
            session,
        ),
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
        } => execute_irf(
            dataset,
            horizon_col,
            response_col,
            ci_lower_col.as_deref(),
            ci_upper_col.as_deref(),
            shock_label.as_deref(),
            response_label.as_deref(),
            output,
            title.as_deref(),
            format,
            session,
        ),
        VizCommands::ScatterInteractive {
            dataset,
            x_col,
            y_col,
            group_col,
            output,
            title,
        } => execute_scatter_interactive(
            dataset,
            x_col,
            y_col,
            group_col.as_deref(),
            output,
            title.as_deref(),
            format,
            session,
        ),
        VizCommands::HistogramInteractive {
            dataset,
            col,
            group_col,
            output,
            title,
        } => execute_histogram_interactive(
            dataset,
            col,
            group_col.as_deref(),
            output,
            title.as_deref(),
            format,
            session,
        ),
        VizCommands::LineInteractive {
            dataset,
            x_col,
            y_col,
            output,
            title,
        } => execute_line_interactive(
            dataset,
            x_col,
            y_col,
            output,
            title.as_deref(),
            format,
            session,
        ),
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
            // Rows with a missing value are left out, as hist() does.
            let cols = match complete_rows(ds, &[col], 1, "histogram") {
                Ok(c) => c,
                Err(e) => {
                    print_error(&e, format);
                    return Ok(());
                }
            };
            let data = cols.numeric(col);

            let chart_title = title
                .map(|s| s.to_string())
                .or_else(|| Some(format!("Histogram of {}", col)));
            let config = ChartConfig {
                title: chart_title,
                width: 800,
                height: 600,
                ..Default::default()
            };

            // histogram(data, bins, config)
            match histogram(data, Some(bins), config) {
                Ok(result) => {
                    // Decode base64 and save PNG to file
                    match decode_base64_image(&result.image_base64) {
                        Ok(png_data) => {
                            std::fs::write(output, &png_data)?;
                            print_saved(
                                &format!("Histogram saved to: {}", output.display()),
                                cols.missing(),
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
            // Complete (x, y) pairs only.
            let cols = match complete_rows(ds, &[x_col, y_col], 1, "scatter") {
                Ok(c) => c,
                Err(e) => {
                    print_error(&e, format);
                    return Ok(());
                }
            };
            let (x_data, y_data) = (cols.numeric(x_col), cols.numeric(y_col));

            let chart_title = title
                .map(|s| s.to_string())
                .or_else(|| Some(format!("{} vs {}", y_col, x_col)));
            let config = ChartConfig {
                title: chart_title,
                width: 800,
                height: 600,
                x_label: Some(x_col.to_string()),
                y_label: Some(y_col.to_string()),
                ..Default::default()
            };

            // scatter_plot(x, y, config)
            match scatter_plot(x_data, y_data, config) {
                Ok(result) => match decode_base64_image(&result.image_base64) {
                    Ok(png_data) => {
                        std::fs::write(output, &png_data)?;
                        print_saved(
                            &format!("Scatter plot saved to: {}", output.display()),
                            cols.missing(),
                            format,
                        );
                    }
                    Err(e) => {
                        print_error(&e, format);
                    }
                },
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
            // Complete (x, y) pairs only.
            let cols = match complete_rows(ds, &[x_col, y_col], 1, "line") {
                Ok(c) => c,
                Err(e) => {
                    print_error(&e, format);
                    return Ok(());
                }
            };
            let x_data = cols.numeric(x_col).to_vec();
            let y_data = cols.numeric(y_col).to_vec();

            let chart_title = title
                .map(|s| s.to_string())
                .or_else(|| Some(format!("{} vs {}", y_col, x_col)));
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
                Ok(result) => match decode_base64_image(&result.image_base64) {
                    Ok(png_data) => {
                        std::fs::write(output, &png_data)?;
                        print_saved(
                            &format!("Line plot saved to: {}", output.display()),
                            cols.missing(),
                            format,
                        );
                    }
                    Err(e) => {
                        print_error(&e, format);
                    }
                },
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
            // Groups in R factor order; rows missing the value or the group are left out.
            let (groups, missing) = match box_groups(ds, value_col, group_col) {
                Ok(g) => g,
                Err(e) => {
                    print_error(&e, format);
                    return Ok(());
                }
            };

            let chart_title = title
                .map(|s| s.to_string())
                .or_else(|| Some(format!("Box Plot of {}", value_col)));
            let config = ChartConfig {
                title: chart_title,
                width: 800,
                height: 600,
                y_label: Some(value_col.to_string()),
                ..Default::default()
            };

            match box_plot(&groups, config) {
                Ok(result) => match decode_base64_image(&result.image_base64) {
                    Ok(png_data) => {
                        std::fs::write(output, &png_data)?;
                        print_saved(
                            &format!("Box plot saved to: {}", output.display()),
                            &missing,
                            format,
                        );
                    }
                    Err(e) => print_error(&e, format),
                },
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
            // Correlations over the complete rows (cor(use = "complete.obs")).
            let (corr_matrix, missing, constant) = match complete_case_correlations(ds, cols) {
                Ok(c) => c,
                Err(e) => {
                    print_error(&e, format);
                    return Ok(());
                }
            };

            let labels: Vec<String> = cols.to_vec();

            match correlation_heatmap(&corr_matrix, &labels, &labels, title, Some(800), Some(600)) {
                Ok(result) => match decode_base64_image(&result.image_base64) {
                    Ok(png_data) => {
                        std::fs::write(output, &png_data)?;
                        let mut message = format!("Heatmap saved to: {}", output.display());
                        if let Some(note) = na_note(&labels, &constant, missing.n_used) {
                            message.push_str(&format!(" ({note})"));
                        }
                        print_saved(&message, &missing, format);
                    }
                    Err(e) => print_error(&e, format),
                },
                Err(e) => print_error(&format!("Heatmap creation failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

/// Whether every value of `x` equals the first (R's sd is then exactly 0).
fn is_constant(x: &[f64]) -> bool {
    x.windows(2).all(|w| w[0] == w[1])
}

/// Pearson correlation of two aligned columns; NaN (R's NA) when either has no variance.
fn compute_correlation(x: &[f64], y: &[f64]) -> f64 {
    if is_constant(x) || is_constant(y) {
        return f64::NAN;
    }
    let n = x.len().min(y.len());
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
        f64::NAN
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
                    let estimates: Vec<f64> =
                        result.coefficients.iter().map(|c| c.estimate).collect();
                    let std_errors: Vec<f64> =
                        result.coefficients.iter().map(|c| c.std_error).collect();
                    let ci_lower: Vec<f64> = estimates
                        .iter()
                        .zip(std_errors.iter())
                        .map(|(b, se)| b - z * se)
                        .collect();
                    let ci_upper: Vec<f64> = estimates
                        .iter()
                        .zip(std_errors.iter())
                        .map(|(b, se)| b + z * se)
                        .collect();

                    let chart_title = title
                        .map(|s| s.to_string())
                        .or_else(|| Some(format!("Coefficients: {}", dep_var)));
                    let config = ChartConfig {
                        title: chart_title,
                        width: 800,
                        height: 600,
                        ..Default::default()
                    };

                    match coefficient_plot(&names, &estimates, &ci_lower, &ci_upper, config, true) {
                        Ok(plot_result) => match decode_base64_image(&plot_result.image_base64) {
                            Ok(png_data) => {
                                std::fs::write(output, &png_data)?;
                                print_saved(
                                    &format!("Coefficient plot saved to: {}", output.display()),
                                    &MissingSummary::none(result.n_obs()),
                                    format,
                                );
                            }
                            Err(e) => print_error(&e, format),
                        },
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

/// Hat values h_i = x_i' (X'X)^{-1} x_i of a least-squares fit, over its identified columns.
fn hat_values(x: &ndarray::Array2<f64>, ls: &p2a_core::linalg::solve::LeastSquares) -> Vec<f64> {
    let keep: Vec<usize> = (0..x.ncols()).filter(|j| !ls.aliased.contains(j)).collect();
    x.rows()
        .into_iter()
        .map(|row| {
            keep.iter()
                .map(|&j| {
                    let cx: f64 = keep.iter().map(|&l| ls.unscaled_cov[[j, l]] * row[l]).sum();
                    row[j] * cx
                })
                .sum()
        })
        .collect()
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
    let Some(ds) = dataset else {
        print_error(&format!("Dataset '{}' not found", dataset_name), format);
        return Ok(());
    };

    // The model's columns, refused on a missing value as run_ols does.
    let x_cols: Vec<&str> = indep_vars.iter().map(|s| s.as_str()).collect();
    let columns = match ColumnRequest::new(ds.df())
        .numeric(dep_var)
        .numerics(&x_cols)
        .na(NaPolicy::Fail)
        .extract()
    {
        Ok(c) => c,
        Err(e) => {
            print_error(&format!("Regression failed: {e}"), format);
            return Ok(());
        }
    };
    let x = match columns.matrix(&x_cols, true) {
        Ok(m) => m.data,
        Err(e) => {
            print_error(&format!("Regression failed: {e}"), format);
            return Ok(());
        }
    };
    let y = columns.numeric_array(dep_var);
    let ls = match least_squares(x.view(), y.view()) {
        Ok(ls) => ls,
        Err(e) => {
            print_error(&format!("Regression failed: {e}"), format);
            return Ok(());
        }
    };
    let hat = hat_values(&x, &ls);
    let fitted = ls.fitted.to_vec();
    let residuals = ls.residuals.to_vec();

    let chart_title = title
        .map(|s| s.to_string())
        .or_else(|| Some("Residual Diagnostics".to_string()));
    let config = ChartConfig {
        title: chart_title,
        width: 500,
        height: 400,
        ..Default::default()
    };

    let plot_result =
        match residual_diagnostics_with(&fitted, &residuals, Some(&hat), Some(ls.rank), config) {
            Ok(r) => r,
            Err(e) => {
                print_error(&format!("Residual plot failed: {}", e), format);
                return Ok(());
            }
        };
    // The four plots: residuals vs fitted under the given name, the others with suffixes.
    let png_data = match decode_base64_image(&plot_result.residuals_vs_fitted) {
        Ok(d) => d,
        Err(e) => {
            print_error(&e, format);
            return Ok(());
        }
    };
    std::fs::write(output, &png_data)?;
    let mut files = vec![output.display().to_string()];
    let stem = output
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("residuals");
    let parent = output.parent().unwrap_or(std::path::Path::new("."));
    for (suffix, image) in [
        ("qq", &plot_result.qq_plot),
        ("scale_location", &plot_result.scale_location),
        ("leverage", &plot_result.residuals_vs_leverage),
    ] {
        // A plot that cannot be decoded or written is an error, as for the main plot, so the
        // message never names a file that was not written.
        let data = match decode_base64_image(image) {
            Ok(d) => d,
            Err(e) => {
                print_error(&format!("the {suffix} plot: {e}"), format);
                return Ok(());
            }
        };
        let path = parent.join(format!("{stem}_{suffix}.png"));
        std::fs::write(&path, &data)?;
        files.push(path.display().to_string());
    }

    let msg = format!(
        "Residual diagnostics saved to: {} (+ _qq, _scale_location, _leverage)",
        output.display()
    );
    match format {
        OutputFormat::Json => {
            let mut json = serde_json::json!({
                "message": msg,
                "files": files,
                "n_obs": plot_result.n_observations,
                "n_params": plot_result.n_params,
                "sigma": plot_result.sigma,
                "leverage": plot_result.leverage,
                "standardized_residuals": plot_result.standardized_residuals,
                "cooks_distance": plot_result.cooks_distance,
            });
            with_result_notes(&mut json, &plot_result);
            println!("{}", serde_json::to_string_pretty(&json)?);
        }
        _ => {
            print_saved(&msg, &MissingSummary::none(y.len()), format);
            print_result_notes(&plot_result, format);
        }
    }
    Ok(())
}

fn execute_dendrogram(
    dataset_name: &str,
    cols: &[String],
    link: Linkage,
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
            // hclust of rows with a missing value is undefined: refuse them (Int64 accepted).
            let names: Vec<&str> = cols.iter().map(String::as_str).collect();
            // hclust needs two objects
            let columns = match ColumnRequest::new(ds.df())
                .numerics(&names)
                .na(NaPolicy::Fail)
                .min_rows(2, &extraction_context("dendrogram", &names))
                .extract()
            {
                Ok(c) => c,
                Err(e) => {
                    print_error(&format!("Failed to extract data: {}", e), format);
                    return Ok(());
                }
            };
            let data = match columns.matrix(&names, false) {
                Ok(m) => m.data,
                Err(e) => {
                    print_error(&format!("Failed to extract data: {}", e), format);
                    return Ok(());
                }
            };

            // Run hierarchical clustering
            match hierarchical(data.view(), None, link, None) {
                Ok(result) => {
                    // linkage_matrix is already Vec<(usize, usize, f64, usize)>
                    let linkage_matrix = &result.linkage_matrix;

                    let chart_title = title
                        .map(|s| s.to_string())
                        .or_else(|| Some("Dendrogram".to_string()));
                    let config = ChartConfig {
                        title: chart_title,
                        width: 1000,
                        height: 600,
                        ..Default::default()
                    };

                    match dendrogram(linkage_matrix, None, config) {
                        Ok(plot_result) => match decode_base64_image(&plot_result.image_base64) {
                            Ok(png_data) => {
                                std::fs::write(output, &png_data)?;
                                let msg = format!("Dendrogram saved to: {}", output.display());
                                match format {
                                    OutputFormat::Json => {
                                        let mut json = serde_json::json!({
                                            "message": msg,
                                            "file": output.display().to_string(),
                                            "linkage": link.to_string(),
                                        });
                                        if let Some(m) = missing_json(columns.missing()) {
                                            json["missing"] = m;
                                        }
                                        println!("{json}");
                                    }
                                    _ => print_saved(&msg, columns.missing(), format),
                                }
                            }
                            Err(e) => print_error(&e, format),
                        },
                        Err(e) => {
                            print_error(&format!("Dendrogram creation failed: {}", e), format)
                        }
                    }
                }
                Err(e) => print_error(&format!("Hierarchical clustering failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
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
            // Rows complete in time, estimate and both limits.
            let names = [time_col, estimate_col, ci_lower_col, ci_upper_col];
            let cols = match complete_rows(ds, &names, 1, "event-study") {
                Ok(c) => c,
                Err(e) => {
                    print_error(&e, format);
                    return Ok(());
                }
            };
            let (time, estimates) = (cols.numeric(time_col), cols.numeric(estimate_col));
            let (ci_lower, ci_upper) = (cols.numeric(ci_lower_col), cols.numeric(ci_upper_col));

            let chart_title = title
                .map(|s| s.to_string())
                .or_else(|| Some("Event Study".to_string()));
            let config = ChartConfig {
                title: chart_title,
                width: 800,
                height: 600,
                x_label: Some("Time".to_string()),
                y_label: Some("Effect".to_string()),
                ..Default::default()
            };

            match event_study_plot(time, estimates, ci_lower, ci_upper, config) {
                Ok(result) => match decode_base64_image(&result.image_base64) {
                    Ok(png_data) => {
                        std::fs::write(output, &png_data)?;
                        print_saved(
                            &format!("Event study plot saved to: {}", output.display()),
                            cols.missing(),
                            format,
                        );
                    }
                    Err(e) => print_error(&e, format),
                },
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
            // Rows complete in horizon, response and every band column given; a band column
            // that cannot be read is an error, not a plot without the band.
            let mut names = vec![horizon_col, response_col];
            names.extend(ci_lower_col);
            names.extend(ci_upper_col);
            let cols = match complete_rows(ds, &names, 1, "irf") {
                Ok(c) => c,
                Err(e) => {
                    print_error(&e, format);
                    return Ok(());
                }
            };
            let (horizons, responses) = (cols.numeric(horizon_col), cols.numeric(response_col));
            let ci_lower = ci_lower_col.map(|col| cols.numeric(col));
            let ci_upper = ci_upper_col.map(|col| cols.numeric(col));

            let chart_title = title
                .map(|s| s.to_string())
                .or_else(|| Some("Impulse Response Function".to_string()));
            let config = ChartConfig {
                title: chart_title,
                width: 800,
                height: 600,
                ..Default::default()
            };

            match irf_plot(
                horizons,
                responses,
                ci_lower,
                ci_upper,
                shock_label,
                response_label,
                config,
            ) {
                Ok(result) => match decode_base64_image(&result.image_base64) {
                    Ok(png_data) => {
                        std::fs::write(output, &png_data)?;
                        print_saved(
                            &format!("IRF plot saved to: {}", output.display()),
                            cols.missing(),
                            format,
                        );
                    }
                    Err(e) => print_error(&e, format),
                },
                Err(e) => print_error(&format!("IRF plot failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_scatter_interactive(
    dataset_name: &str,
    x_col: &str,
    y_col: &str,
    group_col: Option<&str>,
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
            let config = InteractiveConfig {
                title: title.map(|s| s.to_string()),
                x_label: Some(x_col.to_string()),
                y_label: Some(y_col.to_string()),
                ..Default::default()
            };

            match scatter_interactive(ds.df(), x_col, y_col, group_col, config) {
                Ok(result) => {
                    std::fs::write(output, &result.html)?;
                    print_message(
                        &format!("Interactive scatter plot saved to: {}", output.display()),
                        format,
                    );
                }
                Err(e) => print_error(&format!("Scatter plot failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_histogram_interactive(
    dataset_name: &str,
    col: &str,
    group_col: Option<&str>,
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
            let config = InteractiveConfig {
                title: title.map(|s| s.to_string()),
                x_label: Some(col.to_string()),
                y_label: Some("Frequency".to_string()),
                ..Default::default()
            };

            match histogram_interactive(ds.df(), col, group_col, config) {
                Ok(result) => {
                    std::fs::write(output, &result.html)?;
                    print_message(
                        &format!("Interactive histogram saved to: {}", output.display()),
                        format,
                    );
                }
                Err(e) => print_error(&format!("Histogram failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_line_interactive(
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
            let config = InteractiveConfig {
                title: title.map(|s| s.to_string()),
                x_label: Some(x_col.to_string()),
                y_label: Some(y_col.to_string()),
                ..Default::default()
            };

            match line_interactive(ds.df(), x_col, y_col, config) {
                Ok(result) => {
                    std::fs::write(output, &result.html)?;
                    print_message(
                        &format!("Interactive line plot saved to: {}", output.display()),
                        format,
                    );
                }
                Err(e) => print_error(&format!("Line plot failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use p2a_validation::{Case, Check, Tol};

    /// viz heatmap correlates complete rows (R cor(use = "complete.obs")); the missing cells were
    /// read as zeros (r(x, y) 0.7955).
    #[test]
    fn test_validate_cli_viz_heatmap_complete_obs() {
        let case = Case::load("cli_stat_descriptive_na");
        let ds = p2a_core::DataLoader::load(case.dataset_path()).unwrap();
        let cols = ["x", "y", "xi"].map(String::from);
        let (m, missing, _) = complete_case_correlations(&ds, &cols).unwrap();
        let flat: Vec<f64> = m.iter().flatten().copied().collect();
        Check::new(&case)
            .vector("cor_complete", &flat, Tol::stat(6))
            .scalar("n_complete", missing.n_used as f64, Tol::exact())
            .scalar("n_rows", missing.n_total as f64, Tol::exact())
            .finish();
        assert_eq!(
            missing.by_column,
            vec![
                ("x".to_string(), 15),
                ("y".to_string(), 15),
                ("xi".to_string(), 2)
            ]
        );
    }

    /// R's cor(use = "complete.obs") is NA for the correlations of a column that is constant on
    /// the complete rows with the other columns (it warns "the standard deviation is zero"), and
    /// keeps 1 on the diagonal, the constant column's included; the heatmap plotted 0. A
    /// property of the rule, not R-pinned (no case).
    #[test]
    fn test_cli_viz_heatmap_constant_column_is_na() {
        let df = polars::df! {
            "x" => [1.2, 2.9, 3.1, 4.8, 5.3, 6.1],
            "c" => [Some(2.0), Some(2.0), None, Some(2.0), Some(2.0), Some(2.0)],
            "y" => [2.1, 3.9, 6.2, 7.8, 10.1, 12.4],
        }
        .unwrap();
        let ds = p2a_core::Dataset::new(df);
        let cols = ["x", "c", "y"].map(String::from);
        let (m, missing, constant) = complete_case_correlations(&ds, &cols).unwrap();
        for (i, j) in [(0, 1), (1, 0), (1, 2), (2, 1)] {
            assert!(m[i][j].is_nan(), "[{i}][{j}] = {}", m[i][j]);
        }
        assert_eq!((m[0][0], m[1][1], m[2][2]), (1.0, 1.0, 1.0));
        assert!(m[0][2].is_finite() && m[0][2] > 0.9, "{}", m[0][2]);
        assert_eq!(missing.n_dropped, 1);
        assert_eq!(constant, [1]);
        assert_eq!(
            na_note(&cols, &constant, missing.n_used).as_deref(),
            Some("correlations with c are NA: it is constant on the 5 complete rows")
        );
        assert_eq!(na_note(&cols, &[], 6), None);
        // A constant 0.1, whose mean is not exactly 0.1 in f64: still NA, as in R.
        let tenth =
            polars::df! { "x" => [1.2, 2.9, 3.1, 4.8], "t" => [0.1, 0.1, 0.1, 0.1] }.unwrap();
        let cols = ["x", "t"].map(String::from);
        let (m, _, constant) =
            complete_case_correlations(&p2a_core::Dataset::new(tenth), &cols).unwrap();
        assert!(m[0][1].is_nan() && m[1][0].is_nan(), "{m:?}");
        assert_eq!((m[0][0], m[1][1]), (1.0, 1.0), "{m:?}");
        assert_eq!(constant, [1]);
        // Two constant columns: both named.
        let both = polars::df! { "a" => [1.0, 1.0, 1.0], "b" => [3.0, 3.0, 3.0] }.unwrap();
        let cols = ["a", "b"].map(String::from);
        let (m, missing, constant) =
            complete_case_correlations(&p2a_core::Dataset::new(both), &cols).unwrap();
        assert!(m[0][1].is_nan() && m[0][0] == 1.0, "{m:?}");
        assert_eq!(
            na_note(&cols, &constant, missing.n_used).as_deref(),
            Some("correlations with a, b are NA: they are constant on the 3 complete rows")
        );
    }

    /// viz box -g: an Int64 group in R factor order (2 < 10; it was refused, and the order was
    /// random, CLI-40), the row with a missing code dropped (case stat_contingency_na_codes:
    /// table(code, out)).
    #[test]
    fn test_validate_cli_viz_box_groups() {
        let case = Case::load("stat_contingency_na_codes");
        let ds = p2a_core::DataLoader::load(case.dataset_path()).unwrap();
        let (groups, missing) = box_groups(&ds, "out", Some("code")).unwrap();
        assert_eq!(
            groups.iter().map(|(l, _)| l.as_str()).collect::<Vec<_>>(),
            ["2", "10"]
        );
        // out is 0/1: R's table(code, out), row-major
        let tab: Vec<f64> = groups
            .iter()
            .flat_map(|(_, v)| [0.0, 1.0].map(|o| v.iter().filter(|&&x| x == o).count() as f64))
            .collect();
        Check::new(&case)
            .vector("code_out_table", &tab, Tol::exact())
            .finish();
        assert_eq!(
            (missing.n_total, missing.n_used, missing.by_column.clone()),
            (12, 11, vec![("code".to_string(), 1)])
        );
        let (single, none) = box_groups(&ds, "out", None).unwrap();
        assert_eq!(
            (
                single.len(),
                single[0].0.as_str(),
                single[0].1.len(),
                none.n_dropped
            ),
            (1, "out", 12, 0)
        );
    }
}
