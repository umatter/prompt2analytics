//! Machine learning commands

use clap::Subcommand;
use ndarray::Array2;
use p2a_core::{kmeans, pca};

use crate::output::{print_error, OutputFormat};
use crate::session::SessionManager;

#[derive(Subcommand)]
pub enum MlCommands {
    /// K-means clustering
    Kmeans {
        /// Dataset name
        dataset: String,

        /// Feature columns
        #[arg(long, num_args = 1..)]
        cols: Vec<String>,

        /// Number of clusters
        #[arg(short, long)]
        k: usize,

        /// Random seed
        #[arg(long)]
        seed: Option<u64>,

        /// Maximum iterations
        #[arg(long, default_value = "100")]
        max_iter: usize,
    },

    /// Principal Component Analysis
    Pca {
        /// Dataset name
        dataset: String,

        /// Feature columns
        #[arg(long, num_args = 1..)]
        cols: Vec<String>,

        /// Number of components to keep
        #[arg(short, long)]
        n_components: Option<usize>,

        /// Whether to compute transformed data
        #[arg(long, default_value = "false")]
        transform: bool,
    },
}

pub fn execute(
    cmd: &MlCommands,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    match cmd {
        MlCommands::Kmeans {
            dataset,
            cols,
            k,
            seed,
            max_iter,
        } => execute_kmeans(dataset, cols, *k, *seed, *max_iter, format, session),
        MlCommands::Pca {
            dataset,
            cols,
            n_components,
            transform,
        } => execute_pca(dataset, cols, *n_components, *transform, format, session),
    }
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

    // Build column-major then convert, or build row-major directly
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

fn execute_kmeans(
    dataset_name: &str,
    cols: &[String],
    k: usize,
    seed: Option<u64>,
    max_iter: usize,
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
            // Extract columns as Array2
            let data = match extract_columns_as_array(ds, cols) {
                Ok(arr) => arr,
                Err(e) => {
                    print_error(&format!("Failed to extract data: {}", e), format);
                    return Ok(());
                }
            };

            // kmeans(data, k, max_iterations, tolerance, n_init, seed)
            match kmeans(data.view(), k, Some(max_iter), None, None, seed) {
                Ok(result) => {
                    // Convert centroids to Vec<Vec<f64>> for JSON
                    let centroids_vec: Vec<Vec<f64>> = result
                        .centroids
                        .rows()
                        .into_iter()
                        .map(|row| row.to_vec())
                        .collect();

                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "method": "K-means",
                                "k": k,
                                "features": cols,
                                "centroids": centroids_vec,
                                "inertia": result.inertia,
                                "n_iterations": result.n_iterations,
                                "cluster_sizes": result.cluster_sizes,
                                "labels": result.labels,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nK-means Clustering Results (k={})", k);
                            println!("{}", "=".repeat(50));
                            println!("Features: {:?}", cols);
                            println!("Iterations: {}", result.n_iterations);
                            println!("Inertia (within-cluster sum of squares): {:.4}", result.inertia);
                            println!("\nCluster sizes:");
                            for (i, size) in result.cluster_sizes.iter().enumerate() {
                                println!("  Cluster {}: {} observations", i, size);
                            }
                            println!("\nCentroids:");
                            for (i, centroid) in centroids_vec.iter().enumerate() {
                                let centroid_str: Vec<String> = centroid
                                    .iter()
                                    .map(|v| format!("{:.4}", v))
                                    .collect();
                                println!("  Cluster {}: [{}]", i, centroid_str.join(", "));
                            }
                        }
                    }
                }
                Err(e) => {
                    print_error(&format!("K-means failed: {}", e), format);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

fn execute_pca(
    dataset_name: &str,
    cols: &[String],
    n_components: Option<usize>,
    transform: bool,
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
            // Extract columns as Array2
            let data = match extract_columns_as_array(ds, cols) {
                Ok(arr) => arr,
                Err(e) => {
                    print_error(&format!("Failed to extract data: {}", e), format);
                    return Ok(());
                }
            };

            // pca(data, n_components, transform)
            match pca(data.view(), n_components, transform) {
                Ok(result) => {
                    // Convert arrays to Vec for JSON serialization
                    let explained_variance: Vec<f64> = result.explained_variance.to_vec();
                    let explained_variance_ratio: Vec<f64> = result.explained_variance_ratio.to_vec();

                    // Compute cumulative variance ratio
                    let mut cumulative = 0.0;
                    let cumulative_variance_ratio: Vec<f64> = explained_variance_ratio
                        .iter()
                        .map(|r| {
                            cumulative += r;
                            cumulative
                        })
                        .collect();

                    // Convert components (loadings) to Vec<Vec<f64>>
                    let components_vec: Vec<Vec<f64>> = result
                        .components
                        .rows()
                        .into_iter()
                        .map(|row| row.to_vec())
                        .collect();

                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "method": "PCA",
                                "features": cols,
                                "n_components": result.n_components,
                                "explained_variance": explained_variance,
                                "explained_variance_ratio": explained_variance_ratio,
                                "cumulative_variance_ratio": cumulative_variance_ratio,
                                "total_variance": result.total_variance,
                                "components": components_vec,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nPCA Results");
                            println!("{}", "=".repeat(50));
                            println!("Features: {:?}", cols);
                            println!("Number of components: {}", result.n_components);
                            println!("Total variance: {:.4}", result.total_variance);

                            println!("\nExplained Variance:");
                            for (i, (var, ratio)) in explained_variance
                                .iter()
                                .zip(explained_variance_ratio.iter())
                                .enumerate()
                            {
                                let cum = cumulative_variance_ratio[i];
                                println!(
                                    "  PC{}: variance={:.4}, ratio={:.2}%, cumulative={:.2}%",
                                    i + 1,
                                    var,
                                    ratio * 100.0,
                                    cum * 100.0
                                );
                            }

                            println!("\nPrincipal Components (rows are components, cols are features):");
                            for (i, component) in components_vec.iter().enumerate() {
                                let comp_str: Vec<String> = component
                                    .iter()
                                    .map(|v| format!("{:.4}", v))
                                    .collect();
                                println!("  PC{}: [{}]", i + 1, comp_str.join(", "));
                            }
                        }
                    }
                }
                Err(e) => {
                    print_error(&format!("PCA failed: {}", e), format);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}
