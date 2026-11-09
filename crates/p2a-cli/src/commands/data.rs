//! Data loading and inspection commands

use clap::Subcommand;
use p2a_core::data::LoadError;
use p2a_core::simulation::{ColumnSpec, Distribution, generate_random_data};
use p2a_core::{DataLoader, Dataset};
use polars::prelude::*;
use std::path::{Path, PathBuf};

use crate::output::{OutputFormat, format_dataset_summary, print_error, print_message};
use crate::progress::Progress;
use crate::session::SessionManager;

#[derive(Subcommand)]
pub enum DataCommands {
    /// Load a dataset from a file
    #[command(after_help = "\
EXAMPLES:
    # Load CSV and auto-name from filename
    p2a --session s.json data load sales.csv

    # Load with custom name
    p2a --session s.json data load data/monthly.csv --name monthly_sales

    # A CSV export that writes missing values as '.' (Stata, SAS)
    p2a --session s.json data load stata_export.csv --name s --na-strings .
")]
    Load {
        /// Path to the data file (CSV, Parquet, Excel, Stata, SAS)
        path: PathBuf,

        /// Name to assign to the dataset
        #[arg(short, long)]
        name: Option<String>,

        /// Strings read as missing values in a CSV file, comma-separated (e.g. "NA,,." = NA,
        /// the empty field and "."). Replaces the default list, NA and the empty field (R
        /// read.csv); an empty unquoted field is missing in every case. Use --na-strings=-99
        /// for a marker that starts with '-'.
        #[arg(long = "na-strings", value_delimiter = ',', value_name = "LIST")]
        na_strings: Option<Vec<String>>,
    },

    /// List all loaded datasets
    List,

    /// Show dataset summary and statistics
    Describe {
        /// Dataset name
        dataset: String,
    },

    /// Show first N rows of a dataset
    Head {
        /// Dataset name
        dataset: String,

        /// Number of rows to display
        #[arg(short, long, default_value = "10")]
        n: usize,
    },

    /// Generate random data with specified distributions
    #[command(after_help = "\
EXAMPLES:
    # Generate normal and uniform columns
    p2a --session s.json data generate -n 1000 -d simdata \\
        --columns '[{\"name\":\"x\",\"distribution\":{\"type\":\"normal\",\"mean\":0,\"std\":1}},
                    {\"name\":\"e\",\"distribution\":{\"type\":\"uniform\",\"min\":-1,\"max\":1}}]'

With --session the result is stored next to the session file (<session>.datasets/) and is
available to later commands and exported scripts.
")]
    Generate {
        /// Number of rows to generate
        #[arg(short = 'n', long, default_value = "100")]
        rows: usize,

        /// Name to assign to the generated dataset
        #[arg(short = 'd', long, default_value = "generated")]
        name: String,

        /// Column specifications in JSON format
        /// Example: '[{"name": "x", "distribution": {"type": "normal", "mean": 0, "std": 1}}]'
        #[arg(short, long)]
        columns: String,

        /// Random seed for reproducibility
        #[arg(short, long)]
        seed: Option<u64>,
    },

    /// Save/export a dataset to a file
    #[command(after_help = "\
EXAMPLES:
    # Export to CSV
    p2a --session s.json data save mydata --output results.csv

    # Export to Parquet
    p2a --session s.json data save mydata --output results.parquet

    # File format given explicitly
    p2a --session s.json data save mydata --output results.dat --file-format csv
")]
    Save {
        /// Dataset name
        dataset: String,

        /// Output file path (format inferred from extension: .csv, .parquet, .json)
        #[arg(short, long)]
        output: PathBuf,

        /// File format (csv, parquet, json; any case); overrides the file extension
        #[arg(short = 'f', long, value_parser = ["csv", "parquet", "json"], ignore_case = true)]
        file_format: Option<String>,
    },
}

pub fn execute(
    cmd: &DataCommands,
    format: &OutputFormat,
    quiet: bool,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    match cmd {
        DataCommands::Load {
            path,
            name,
            na_strings,
        } => execute_load(
            path,
            name.as_deref(),
            na_strings.as_deref(),
            format,
            quiet,
            session,
        ),
        DataCommands::List => execute_list(format, session),
        DataCommands::Describe { dataset } => execute_describe(dataset, format, session),
        DataCommands::Head { dataset, n } => execute_head(dataset, *n, format, session),
        DataCommands::Generate {
            rows,
            name,
            columns,
            seed,
        } => execute_generate(*rows, name, columns, *seed, format, quiet, session),
        DataCommands::Save {
            dataset,
            output,
            file_format,
        } => execute_save(
            dataset,
            output,
            file_format.as_deref(),
            format,
            quiet,
            session,
        ),
    }
}

/// Read a data file of any format [`DataLoader::load`] reads, with the `--na-strings`
/// markers of a CSV load (`None`: the loader's default, `NA` and the empty field).
/// `data load` and the session's reload share it, so a reload reads the file as the load
/// did.
pub(crate) fn load_with_markers(path: &Path, na: Option<&[String]>) -> Result<Dataset, LoadError> {
    match na {
        Some(na) => {
            let na: Vec<&str> = na.iter().map(String::as_str).collect();
            DataLoader::load_with_na(path, &na)
        }
        None => DataLoader::load(path),
    }
}

fn execute_load(
    path: &PathBuf,
    name: Option<&str>,
    na_strings: Option<&[String]>,
    format: &OutputFormat,
    quiet: bool,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    log::info!("Loading dataset from: {}", path.display());

    // Determine dataset name from filename if not provided
    let dataset_name = name.map(|s| s.to_string()).unwrap_or_else(|| {
        path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("dataset")
            .to_string()
    });

    // Determine file format from extension
    let extension = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();

    log::debug!(
        "Dataset name: '{}', file extension: '{}'",
        dataset_name,
        extension
    );

    // Show progress spinner for file loading
    let progress = Progress::spinner(
        &format!(
            "Loading {}...",
            path.file_name().unwrap_or_default().to_string_lossy()
        ),
        quiet,
    );

    // --na-strings is a CSV option: refused for every other extension
    if na_strings.is_some() && extension != "csv" {
        progress.finish_and_clear();
        let file = if extension.is_empty() {
            "a file without an extension".to_string()
        } else {
            format!("a '{extension}' file")
        };
        print_error(
            &format!(
                "--na-strings applies to CSV files only, not to {file}; \
                 parquet, Excel, Stata and SAS files store missing values natively"
            ),
            format,
        );
        return Ok(());
    }

    // Load the dataset (an unknown extension is an error listing the readable ones)
    let loaded = load_with_markers(path, na_strings);
    progress.finish_and_clear();
    let dataset = loaded?;

    let df = dataset.df();
    let nrows = df.height();
    let ncols = df.width();
    let columns: Vec<String> = df
        .get_column_names()
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    log::info!(
        "Successfully loaded '{}': {} rows × {} columns",
        dataset_name,
        nrows,
        ncols
    );
    log::debug!("Columns: {:?}", columns);

    // Register with session if available
    if let Some(mgr) = session {
        mgr.register_dataset(
            &dataset_name,
            path.clone(),
            &extension,
            &dataset,
            na_strings.map(<[String]>::to_vec),
        );
        mgr.store_dataset(dataset_name.clone(), dataset);
    }

    // Output success message
    let summary = format_dataset_summary(&dataset_name, nrows, ncols, &columns, format);
    println!("{}", summary);

    Ok(())
}

fn execute_list(format: &OutputFormat, session: Option<&mut SessionManager>) -> anyhow::Result<()> {
    match session {
        Some(mgr) => {
            let datasets = mgr.list_datasets();
            if datasets.is_empty() {
                print_message("No datasets loaded", format);
            } else {
                match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "datasets": datasets
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("Loaded datasets:");
                        for name in datasets {
                            println!("  - {}", name);
                        }
                    }
                }
            }
        }
        None => {
            print_message(
                "No session active. Use --session <file> to enable dataset storage.",
                format,
            );
        }
    }
    Ok(())
}

fn execute_describe(
    dataset_name: &str,
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
            let df = ds.df();
            let nrows = df.height();
            let ncols = df.width();
            let columns: Vec<String> = df
                .get_column_names()
                .into_iter()
                .map(|s| s.to_string())
                .collect();

            let summary = format_dataset_summary(dataset_name, nrows, ncols, &columns, format);
            println!("{}", summary);

            // Show column types
            match format {
                OutputFormat::Json => {
                    let col_info: Vec<serde_json::Value> = df
                        .get_columns()
                        .iter()
                        .map(|c: &Column| {
                            serde_json::json!({
                                "name": c.name().to_string(),
                                "dtype": format!("{:?}", c.dtype()),
                            })
                        })
                        .collect();
                    println!("{}", serde_json::to_string_pretty(&col_info)?);
                }
                _ => {
                    println!("\nColumn types:");
                    for col in df.get_columns() {
                        println!("  {}: {:?}", col.name(), col.dtype());
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

fn execute_head(
    dataset_name: &str,
    n: usize,
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
            let df = ds.df();
            let head = df.head(Some(n));

            match format {
                OutputFormat::Json => {
                    // Convert to JSON
                    let mut rows = Vec::new();
                    for i in 0..head.height() {
                        let mut row = serde_json::Map::new();
                        for col in head.get_columns() {
                            let val = col.get(i).map_or("null".to_string(), |v| format!("{}", v));
                            row.insert(col.name().to_string(), serde_json::Value::String(val));
                        }
                        rows.push(serde_json::Value::Object(row));
                    }
                    println!("{}", serde_json::to_string_pretty(&rows)?);
                }
                _ => {
                    println!("{}", head);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

/// Column spec input for JSON parsing
#[derive(serde::Deserialize)]
struct ColumnSpecInput {
    name: String,
    distribution: serde_json::Value,
}

fn execute_generate(
    n_rows: usize,
    name: &str,
    columns_json: &str,
    seed: Option<u64>,
    format: &OutputFormat,
    quiet: bool,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    log::info!("Generating random dataset '{}' with {} rows", name, n_rows);
    log::debug!("Seed: {:?}", seed);

    // Parse the columns JSON
    let col_inputs: Vec<ColumnSpecInput> = match serde_json::from_str(columns_json) {
        Ok(cols) => cols,
        Err(e) => {
            print_error(
                &format!(
                    "Invalid columns JSON: {}\n\n\
                     Expected format: '[{{\"name\": \"col1\", \"distribution\": {{\"type\": \"normal\", \"mean\": 0, \"std\": 1}}}}]'\n\n\
                     Available distribution types:\n\
                     - uniform: {{\"type\": \"uniform\", \"min\": 0.0, \"max\": 1.0}}\n\
                     - normal: {{\"type\": \"normal\", \"mean\": 0.0, \"std\": 1.0}}\n\
                     - binomial: {{\"type\": \"binomial\", \"n\": 10, \"p\": 0.5}}\n\
                     - poisson: {{\"type\": \"poisson\", \"lambda\": 5.0}}\n\
                     - exponential: {{\"type\": \"exponential\", \"rate\": 1.0}}\n\
                     - bernoulli: {{\"type\": \"bernoulli\", \"p\": 0.5}}\n\
                     - categorical: {{\"type\": \"categorical\", \"categories\": [\"A\", \"B\", \"C\"]}}\n\
                     - uniform_int: {{\"type\": \"uniform_int\", \"min\": 1, \"max\": 10}}\n\
                     - sequence: {{\"type\": \"sequence\", \"start\": 1}}\n\
                     - constant: {{\"type\": \"constant\", \"value\": 42.0}}\n\
                     - constant_string: {{\"type\": \"constant_string\", \"value\": \"text\"}}",
                    e
                ),
                format,
            );
            return Ok(());
        }
    };

    // Convert to ColumnSpec
    let mut columns: Vec<ColumnSpec> = Vec::with_capacity(col_inputs.len());
    for col_input in col_inputs {
        let dist: Distribution = match serde_json::from_value(col_input.distribution) {
            Ok(d) => d,
            Err(e) => {
                print_error(
                    &format!(
                        "Invalid distribution for column '{}': {}",
                        col_input.name, e
                    ),
                    format,
                );
                return Ok(());
            }
        };
        columns.push(ColumnSpec::new(&col_input.name, dist));
    }

    if columns.is_empty() {
        print_error("At least one column specification is required", format);
        return Ok(());
    }

    // Generate the data with progress indicator
    log::debug!("Generating {} column(s)", columns.len());
    let progress = Progress::spinner(&format!("Generating {} rows...", n_rows), quiet);

    let dataset = match generate_random_data(n_rows, columns, seed) {
        Ok(ds) => {
            progress.finish_and_clear();
            ds
        }
        Err(e) => {
            progress.finish_and_clear();
            log::error!("Data generation failed: {}", e);
            print_error(&format!("Failed to generate data: {}", e), format);
            return Ok(());
        }
    };

    let df = dataset.df();
    let nrows = df.height();
    let ncols = df.width();
    let col_names: Vec<String> = df
        .get_column_names()
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    log::info!(
        "Generated dataset '{}': {} rows × {} columns",
        name,
        nrows,
        ncols
    );

    // With a session the dataset is written next to the session file, so later
    // invocations find it
    if let Some(mgr) = session {
        if let Err(e) = mgr.persist_dataset(name, dataset) {
            print_error(&e.to_string(), format);
            return Ok(());
        }
    }

    // Output success message
    match format {
        OutputFormat::Json => {
            let json = serde_json::json!({
                "name": name,
                "rows": nrows,
                "columns": ncols,
                "column_names": col_names,
                "seed": seed,
            });
            println!("{}", serde_json::to_string_pretty(&json)?);
        }
        _ => {
            println!("Generated random dataset '{}'", name);
            println!("  Rows: {}", nrows);
            println!("  Columns: {}", ncols);
            println!("  Column names: {}", col_names.join(", "));
            if let Some(s) = seed {
                println!("  Seed: {}", s);
            }
        }
    }

    Ok(())
}

fn execute_save(
    dataset_name: &str,
    output: &PathBuf,
    fmt: Option<&str>,
    format: &OutputFormat,
    quiet: bool,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    log::info!("Saving dataset '{}' to: {}", dataset_name, output.display());

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

    let ds = match dataset {
        Some(ds) => ds,
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
            return Ok(());
        }
    };

    // Determine format from option or extension, in any case (as `execute_load` reads extensions)
    let output_format = fmt
        .or_else(|| output.extension().and_then(|ext| ext.to_str()))
        .map(str::to_ascii_lowercase);

    // Show progress spinner for file saving
    let progress = Progress::spinner(
        &format!(
            "Saving to {}...",
            output.file_name().unwrap_or_default().to_string_lossy()
        ),
        quiet,
    );

    let result: Result<&str, String> = match output_format.as_deref() {
        Some("csv") => ds.to_csv(output).map(|_| "CSV").map_err(|e| e.to_string()),
        Some("parquet") => ds
            .to_parquet(output)
            .map(|_| "Parquet")
            .map_err(|e| e.to_string()),
        Some("json") => ds
            .to_json_string()
            .map_err(|e| e.to_string())
            .and_then(|json| std::fs::write(output, json).map_err(|e| e.to_string()))
            .map(|_| "JSON"),
        Some(other) => {
            progress.finish_and_clear();
            print_error(
                &format!(
                    "Unsupported file format '{other}'. Use --file-format csv|parquet|json or a .csv, .parquet or .json extension."
                ),
                format,
            );
            return Ok(());
        }
        None => {
            progress.finish_and_clear();
            print_error(
                "Could not determine the file format. Use --file-format csv|parquet|json or a .csv, .parquet or .json extension.",
                format,
            );
            return Ok(());
        }
    };

    progress.finish_and_clear();

    match result {
        Ok(fmt_name) => {
            log::info!(
                "Successfully saved '{}' to {} ({} rows × {} columns)",
                dataset_name,
                output.display(),
                ds.nrows(),
                ds.ncols()
            );
            match format {
                OutputFormat::Json => {
                    let json = serde_json::json!({
                        "dataset": dataset_name,
                        "path": output.display().to_string(),
                        "format": fmt_name,
                        "rows": ds.nrows(),
                        "columns": ds.ncols(),
                    });
                    println!("{}", serde_json::to_string_pretty(&json)?);
                }
                _ => {
                    println!(
                        "Successfully exported dataset '{}' to {} format",
                        dataset_name, fmt_name
                    );
                    println!("  Path: {}", output.display());
                    println!("  Rows: {}", ds.nrows());
                    println!("  Columns: {}", ds.ncols());
                }
            }
        }
        Err(e) => {
            log::error!("Failed to save dataset: {}", e);
            print_error(&format!("Failed to save dataset: {}", e), format);
        }
    }

    Ok(())
}
