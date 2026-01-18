//! Data loading and inspection commands

use clap::Subcommand;
use p2a_core::{Dataset, DataLoader};
use p2a_core::simulation::{generate_random_data, ColumnSpec, Distribution};
use polars::prelude::*;
use std::path::PathBuf;

use crate::output::{format_dataset_summary, print_error, print_message, OutputFormat};
use crate::session::SessionManager;

#[derive(Subcommand)]
pub enum DataCommands {
    /// Load a dataset from a file
    Load {
        /// Path to the data file (CSV, Parquet, Excel, Stata, SAS)
        path: PathBuf,

        /// Name to assign to the dataset
        #[arg(short, long)]
        name: Option<String>,
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
}

pub fn execute(
    cmd: &DataCommands,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    match cmd {
        DataCommands::Load { path, name } => {
            execute_load(path, name.as_deref(), format, session)
        }
        DataCommands::List => execute_list(format, session),
        DataCommands::Describe { dataset } => execute_describe(dataset, format, session),
        DataCommands::Head { dataset, n } => execute_head(dataset, *n, format, session),
        DataCommands::Generate { rows, name, columns, seed } => {
            execute_generate(*rows, name, columns, *seed, format, session)
        }
    }
}

fn execute_load(
    path: &PathBuf,
    name: Option<&str>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    // Determine dataset name from filename if not provided
    let dataset_name = name
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
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

    // Load the dataset
    let df = match extension.as_str() {
        "csv" => DataLoader::load_csv(path)?,
        "parquet" => DataLoader::load_parquet(path)?,
        "xlsx" | "xls" | "xlsb" => DataLoader::load_excel(path, None)?,
        "dta" => DataLoader::load_stata(path)?,
        "sas7bdat" => DataLoader::load_sas(path)?,
        _ => {
            print_error(
                &format!("Unsupported file format: {}", extension),
                format,
            );
            return Ok(());
        }
    };
    let dataset = Dataset::new(df);

    let df = dataset.df();
    let nrows = df.height();
    let ncols = df.width();
    let columns: Vec<String> = df
        .get_column_names()
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    // Register with session if available
    if let Some(mgr) = session {
        mgr.register_dataset(&dataset_name, path.clone(), &extension, &dataset);
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
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
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
                    &format!("Invalid distribution for column '{}': {}", col_input.name, e),
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

    // Generate the data
    let dataset = match generate_random_data(n_rows, columns, seed) {
        Ok(ds) => ds,
        Err(e) => {
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

    // Register with session if available
    if let Some(mgr) = session {
        mgr.store_dataset(name.to_string(), dataset);
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
