//! Machine learning commands

use clap::Subcommand;
use ndarray::{Array1, Array2};
use p2a_core::{Linkage, dbscan, hierarchical, kmeans, linear_svm, pca, random_forest, tsne};

use crate::output::{OutputFormat, print_error};
use crate::session::SessionManager;

#[derive(Subcommand)]
pub enum MlCommands {
    /// K-means clustering
    #[command(after_help = "\
EXAMPLES:
    # Cluster into 3 groups
    p2a --session s.json ml kmeans mydata --cols x1 x2 x3 -k 3

    # With reproducible seed
    p2a --session s.json ml kmeans mydata --cols income age --k 5 --seed 42
")]
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
    #[command(after_help = "\
EXAMPLES:
    # Keep top 3 components
    p2a --session s.json ml pca mydata --cols x1 x2 x3 x4 x5 -n 3

    # All components with transformed data
    p2a --session s.json ml pca mydata --cols x1 x2 x3 --transform
")]
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

    /// t-SNE dimensionality reduction
    #[command(after_help = "\
EXAMPLES:
    # 2D embedding (default)
    p2a --session s.json ml tsne mydata --cols x1 x2 x3 x4 x5

    # Custom perplexity and learning rate
    p2a --session s.json ml tsne mydata --cols feature1 feature2 feature3 \\
        --perplexity 50 --learning-rate 100 --seed 42
")]
    Tsne {
        /// Dataset name
        dataset: String,

        /// Feature columns
        #[arg(long, num_args = 1..)]
        cols: Vec<String>,

        /// Number of output dimensions (default: 2)
        #[arg(short, long, default_value = "2")]
        n_components: usize,

        /// Perplexity parameter (default: 30.0)
        #[arg(long, default_value = "30.0")]
        perplexity: f64,

        /// Maximum iterations (default: 1000)
        #[arg(long, default_value = "1000")]
        max_iter: usize,

        /// Learning rate (default: 200.0)
        #[arg(long, default_value = "200.0")]
        learning_rate: f64,

        /// Random seed
        #[arg(long)]
        seed: Option<u64>,
    },

    /// Random Forest regression/classification
    #[command(after_help = "\
EXAMPLES:
    # Random forest with 100 trees
    p2a --session s.json ml random-forest mydata --cols x1 x2 x3 -y target --n-trees 100

    # Custom depth and features
    p2a --session s.json ml random-forest mydata --cols age income education \\
        -y outcome --max-depth 5 --max-features sqrt --seed 42
")]
    RandomForest {
        /// Dataset name
        dataset: String,

        /// Feature columns
        #[arg(long, num_args = 1..)]
        cols: Vec<String>,

        /// Target column
        #[arg(short = 'y', long)]
        target: String,

        /// Number of trees (default: 100)
        #[arg(long, default_value = "100")]
        n_trees: usize,

        /// Maximum tree depth (default: 10)
        #[arg(long, default_value = "10")]
        max_depth: usize,

        /// Minimum samples to split (default: 2)
        #[arg(long, default_value = "2")]
        min_samples_split: usize,

        /// Max features per split: "sqrt", "log2", "all", or number
        #[arg(long, default_value = "sqrt")]
        max_features: String,

        /// Random seed
        #[arg(long)]
        seed: Option<u64>,
    },

    /// DBSCAN density-based clustering
    #[command(after_help = "\
EXAMPLES:
    p2a --session s.json ml dbscan mydata --cols x y --eps 0.5 --min-samples 5
")]
    Dbscan {
        /// Dataset name
        dataset: String,

        /// Feature columns
        #[arg(long, num_args = 1..)]
        cols: Vec<String>,

        /// Maximum distance between samples for neighborhood (epsilon)
        #[arg(long)]
        eps: f64,

        /// Minimum samples in neighborhood for a core point
        #[arg(long, default_value = "5")]
        min_samples: usize,
    },

    /// Hierarchical (agglomerative) clustering
    #[command(after_help = "\
EXAMPLES:
    # Ward linkage with 4 clusters
    p2a --session s.json ml hierarchical mydata --cols x1 x2 x3 -n 4 --linkage ward

    # Cut by distance threshold
    p2a --session s.json ml hierarchical mydata --cols x1 x2 --distance-threshold 2.5
")]
    Hierarchical {
        /// Dataset name
        dataset: String,

        /// Feature columns
        #[arg(long, num_args = 1..)]
        cols: Vec<String>,

        /// Number of clusters to form
        #[arg(short, long)]
        n_clusters: Option<usize>,

        /// Linkage method: single, complete, average, ward
        #[arg(long, default_value = "ward")]
        linkage: String,

        /// Distance threshold for cutting the dendrogram
        #[arg(long)]
        distance_threshold: Option<f64>,
    },

    /// Linear Support Vector Machine (SVM)
    #[command(after_help = "\
EXAMPLES:
    # Binary classification
    p2a --session s.json ml svm mydata --cols x1 x2 x3 -y label -c 1.0
")]
    Svm {
        /// Dataset name
        dataset: String,

        /// Feature columns
        #[arg(long, num_args = 1..)]
        cols: Vec<String>,

        /// Target column (binary classification)
        #[arg(short = 'y', long)]
        target: String,

        /// Regularization parameter C (default: 1.0)
        #[arg(short, long, default_value = "1.0")]
        c: f64,

        /// Maximum iterations (default: 1000)
        #[arg(long, default_value = "1000")]
        max_iter: usize,

        /// Convergence tolerance (default: 1e-3)
        #[arg(long, default_value = "0.001")]
        tolerance: f64,
    },
}

pub fn execute(
    cmd: &MlCommands,
    format: &OutputFormat,
    _quiet: bool,
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
        MlCommands::Tsne {
            dataset,
            cols,
            n_components,
            perplexity,
            max_iter,
            learning_rate,
            seed,
        } => execute_tsne(
            dataset,
            cols,
            *n_components,
            *perplexity,
            *max_iter,
            *learning_rate,
            *seed,
            format,
            session,
        ),
        MlCommands::RandomForest {
            dataset,
            cols,
            target,
            n_trees,
            max_depth,
            min_samples_split,
            max_features,
            seed,
        } => execute_random_forest(
            dataset,
            cols,
            target,
            *n_trees,
            *max_depth,
            *min_samples_split,
            max_features,
            *seed,
            format,
            session,
        ),
        MlCommands::Dbscan {
            dataset,
            cols,
            eps,
            min_samples,
        } => execute_dbscan(dataset, cols, *eps, *min_samples, format, session),
        MlCommands::Hierarchical {
            dataset,
            cols,
            n_clusters,
            linkage,
            distance_threshold,
        } => execute_hierarchical(
            dataset,
            cols,
            *n_clusters,
            linkage,
            *distance_threshold,
            format,
            session,
        ),
        MlCommands::Svm {
            dataset,
            cols,
            target,
            c,
            max_iter,
            tolerance,
        } => execute_svm(
            dataset, cols, target, *c, *max_iter, *tolerance, format, session,
        ),
    }
}

/// Extract multiple columns from a Dataset as an `Array2<f64>`
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
                            println!(
                                "Inertia (within-cluster sum of squares): {:.4}",
                                result.inertia
                            );
                            println!("\nCluster sizes:");
                            for (i, size) in result.cluster_sizes.iter().enumerate() {
                                println!("  Cluster {}: {} observations", i, size);
                            }
                            println!("\nCentroids:");
                            for (i, centroid) in centroids_vec.iter().enumerate() {
                                let centroid_str: Vec<String> =
                                    centroid.iter().map(|v| format!("{:.4}", v)).collect();
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
                    let explained_variance_ratio: Vec<f64> =
                        result.explained_variance_ratio.to_vec();

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

                            println!(
                                "\nPrincipal Components (rows are components, cols are features):"
                            );
                            for (i, component) in components_vec.iter().enumerate() {
                                let comp_str: Vec<String> =
                                    component.iter().map(|v| format!("{:.4}", v)).collect();
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

fn execute_tsne(
    dataset_name: &str,
    cols: &[String],
    n_components: usize,
    perplexity: f64,
    max_iter: usize,
    learning_rate: f64,
    seed: Option<u64>,
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
            let data = match extract_columns_as_array(ds, cols) {
                Ok(arr) => arr,
                Err(e) => {
                    print_error(&format!("Failed to extract data: {}", e), format);
                    return Ok(());
                }
            };

            match tsne(
                data.view(),
                Some(n_components),
                Some(perplexity),
                Some(max_iter),
                Some(learning_rate),
                seed,
            ) {
                Ok(result) => {
                    // Convert embedding to Vec<Vec<f64>> for JSON
                    let embedding_vec: Vec<Vec<f64>> = result
                        .embedding
                        .rows()
                        .into_iter()
                        .map(|row| row.to_vec())
                        .collect();

                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "method": "t-SNE",
                                "n_components": result.n_components,
                                "perplexity": result.perplexity,
                                "n_iterations": result.n_iterations,
                                "kl_divergence": result.kl_divergence,
                                "embedding": embedding_vec,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nt-SNE Results");
                            println!("{}", "=".repeat(50));
                            println!("Features: {:?}", cols);
                            println!("Output dimensions: {}", result.n_components);
                            println!("Perplexity: {}", result.perplexity);
                            println!("Iterations: {}", result.n_iterations);
                            println!("KL Divergence: {:.6}", result.kl_divergence);
                            println!(
                                "\nEmbedding shape: {} x {}",
                                embedding_vec.len(),
                                result.n_components
                            );
                            println!("(Use JSON output for full embedding data)");
                        }
                    }
                }
                Err(e) => print_error(&format!("t-SNE failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_random_forest(
    dataset_name: &str,
    cols: &[String],
    target_col: &str,
    n_trees: usize,
    max_depth: usize,
    min_samples_split: usize,
    max_features: &str,
    seed: Option<u64>,
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
            let data = match extract_columns_as_array(ds, cols) {
                Ok(arr) => arr,
                Err(e) => {
                    print_error(&format!("Failed to extract feature data: {}", e), format);
                    return Ok(());
                }
            };

            // Extract target column
            let target: Array1<f64> = {
                let col = ds.df().column(target_col);
                match col {
                    Ok(c) => match c.f64() {
                        Ok(ca) => ca.into_no_null_iter().collect(),
                        Err(e) => {
                            print_error(&format!("Target column must be numeric: {}", e), format);
                            return Ok(());
                        }
                    },
                    Err(e) => {
                        print_error(&format!("Target column not found: {}", e), format);
                        return Ok(());
                    }
                }
            };

            let feature_names: Vec<String> = cols.to_vec();

            match random_forest(
                data.view(),
                target.view(),
                Some(n_trees),
                Some(max_depth),
                Some(min_samples_split),
                Some(max_features),
                seed,
                Some(feature_names.clone()),
            ) {
                Ok(result) => {
                    let importances: Vec<f64> = result.feature_importances.to_vec();

                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "method": "Random Forest",
                                "n_trees": result.n_trees,
                                "features": feature_names,
                                "feature_importances": importances,
                                "oob_score": result.oob_score,
                                "predictions": result.predictions,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nRandom Forest Results");
                            println!("{}", "=".repeat(50));
                            println!("Number of trees: {}", result.n_trees);
                            println!("Max depth: {}", max_depth);
                            println!("Max features per split: {}", max_features);

                            if let Some(oob) = result.oob_score {
                                println!("\nOOB Score: {:.4}", oob);
                            }

                            println!("\nFeature Importances:");
                            let mut indexed: Vec<(usize, &f64)> =
                                importances.iter().enumerate().collect();
                            indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
                            for (idx, imp) in indexed.iter().take(10) {
                                println!("  {}: {:.4}", feature_names[*idx], imp);
                            }
                        }
                    }
                }
                Err(e) => print_error(&format!("Random Forest failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_dbscan(
    dataset_name: &str,
    cols: &[String],
    eps: f64,
    min_samples: usize,
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
            let data = match extract_columns_as_array(ds, cols) {
                Ok(arr) => arr,
                Err(e) => {
                    print_error(&format!("Failed to extract data: {}", e), format);
                    return Ok(());
                }
            };

            match dbscan(data.view(), eps, min_samples) {
                Ok(result) => {
                    // Count cluster sizes
                    let mut cluster_sizes: std::collections::HashMap<i32, usize> =
                        std::collections::HashMap::new();
                    for &label in &result.labels {
                        *cluster_sizes.entry(label).or_insert(0) += 1;
                    }

                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "method": "DBSCAN",
                                "eps": eps,
                                "min_samples": min_samples,
                                "features": cols,
                                "n_clusters": result.n_clusters,
                                "n_noise": result.n_noise,
                                "n_core_samples": result.core_sample_indices.len(),
                                "cluster_sizes": cluster_sizes,
                                "labels": result.labels,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nDBSCAN Clustering Results");
                            println!("{}", "=".repeat(50));
                            println!("Features: {:?}", cols);
                            println!("Epsilon (eps): {}", eps);
                            println!("Min samples: {}", min_samples);
                            println!("Number of clusters: {}", result.n_clusters);
                            println!("Number of noise points: {}", result.n_noise);
                            println!(
                                "Number of core samples: {}",
                                result.core_sample_indices.len()
                            );
                            println!("\nCluster sizes:");
                            let mut labels_sorted: Vec<_> = cluster_sizes.keys().cloned().collect();
                            labels_sorted.sort();
                            for label in &labels_sorted {
                                if *label == -1 {
                                    println!("  Noise: {} points", cluster_sizes[label]);
                                } else {
                                    println!(
                                        "  Cluster {}: {} points",
                                        label, cluster_sizes[label]
                                    );
                                }
                            }
                        }
                    }
                }
                Err(e) => print_error(&format!("DBSCAN failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_hierarchical(
    dataset_name: &str,
    cols: &[String],
    n_clusters: Option<usize>,
    linkage_str: &str,
    distance_threshold: Option<f64>,
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
            let data = match extract_columns_as_array(ds, cols) {
                Ok(arr) => arr,
                Err(e) => {
                    print_error(&format!("Failed to extract data: {}", e), format);
                    return Ok(());
                }
            };

            // Parse linkage method
            let linkage: Linkage = match linkage_str.parse() {
                Ok(l) => l,
                Err(e) => {
                    print_error(&format!("Invalid linkage method: {}", e), format);
                    return Ok(());
                }
            };

            match hierarchical(data.view(), n_clusters, linkage, distance_threshold) {
                Ok(result) => {
                    // Count cluster sizes
                    let mut cluster_sizes: std::collections::HashMap<usize, usize> =
                        std::collections::HashMap::new();
                    for &label in &result.labels {
                        *cluster_sizes.entry(label).or_insert(0) += 1;
                    }

                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "method": "Hierarchical Clustering",
                                "linkage": result.linkage,
                                "features": cols,
                                "n_clusters": result.n_clusters,
                                "cluster_sizes": cluster_sizes,
                                "labels": result.labels,
                                "merge_distances": result.merge_distances,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nHierarchical Clustering Results");
                            println!("{}", "=".repeat(50));
                            println!("Features: {:?}", cols);
                            println!("Linkage method: {}", result.linkage);
                            println!("Number of clusters: {}", result.n_clusters);
                            println!("\nCluster sizes:");
                            let mut labels_sorted: Vec<_> = cluster_sizes.keys().collect();
                            labels_sorted.sort();
                            for &label in &labels_sorted {
                                println!("  Cluster {}: {} points", label, cluster_sizes[label]);
                            }
                            if !result.merge_distances.is_empty() {
                                println!("\nDendrogram (last 5 merges):");
                                for (i, dist) in
                                    result.merge_distances.iter().rev().take(5).enumerate()
                                {
                                    println!(
                                        "  Merge {}: distance = {:.4}",
                                        result.merge_distances.len() - i,
                                        dist
                                    );
                                }
                            }
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

fn execute_svm(
    dataset_name: &str,
    cols: &[String],
    target_col: &str,
    c: f64,
    max_iter: usize,
    tolerance: f64,
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
            let data = match extract_columns_as_array(ds, cols) {
                Ok(arr) => arr,
                Err(e) => {
                    print_error(&format!("Failed to extract feature data: {}", e), format);
                    return Ok(());
                }
            };

            // Extract target column
            let target: Array1<f64> = {
                let col = ds.df().column(target_col);
                match col {
                    Ok(c) => match c.f64() {
                        Ok(ca) => ca.into_no_null_iter().collect(),
                        Err(e) => {
                            print_error(&format!("Target column must be numeric: {}", e), format);
                            return Ok(());
                        }
                    },
                    Err(e) => {
                        print_error(&format!("Target column not found: {}", e), format);
                        return Ok(());
                    }
                }
            };

            let feature_names: Vec<String> = cols.to_vec();

            match linear_svm(
                data.view(),
                target.view(),
                Some(c),
                Some(max_iter),
                Some(tolerance),
                Some(feature_names.clone()),
            ) {
                Ok(result) => {
                    // Count predictions per class
                    let neg_count = result.predictions.iter().filter(|&&p| p < 0).count();
                    let pos_count = result.predictions.len() - neg_count;

                    match format {
                        OutputFormat::Json => {
                            let json = serde_json::json!({
                                "method": "Linear SVM",
                                "features": feature_names,
                                "c": c,
                                "converged": result.converged,
                                "n_iterations": result.n_iterations,
                                "n_support_vectors": result.n_support_vectors,
                                "weights": result.weights,
                                "bias": result.bias,
                                "class_labels": result.class_labels,
                                "predictions": result.predictions,
                            });
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nLinear SVM Results");
                            println!("{}", "=".repeat(50));
                            println!("Features: {:?}", feature_names);
                            println!("Regularization C: {}", c);
                            println!("Converged: {}", result.converged);
                            println!("Iterations: {}", result.n_iterations);
                            println!("Support vectors: {}", result.n_support_vectors);
                            println!("Bias: {:.6}", result.bias);

                            if let Some((neg, pos)) = result.class_labels {
                                println!("\nClass labels: {} (negative), {} (positive)", neg, pos);
                            }

                            println!("\nFeature Weights (top 10 by magnitude):");
                            let mut indexed: Vec<(usize, f64)> = result
                                .weights
                                .iter()
                                .enumerate()
                                .map(|(i, &v)| (i, v))
                                .collect();
                            indexed.sort_by(|a, b| {
                                b.1.abs()
                                    .partial_cmp(&a.1.abs())
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            });
                            for (idx, weight) in indexed.iter().take(10) {
                                println!("  {}: {:.6}", feature_names[*idx], weight);
                            }

                            println!(
                                "\nPrediction distribution: {} negative, {} positive",
                                neg_count, pos_count
                            );
                        }
                    }
                }
                Err(e) => print_error(&format!("SVM failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}
