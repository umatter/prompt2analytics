//! Spatial econometrics commands

use clap::Subcommand;
use p2a_core::{
    MoranAlternative,
    // Spatial weights infrastructure
    Neighbors,
    SacConfig,
    SarConfig,
    SemConfig,
    SpatialWeights,
    WeightStyle,
    // Spatial diagnostics
    moran_test,
    run_sac,
    // Spatial regression models
    run_sar,
    run_sem,
};

use crate::output::{OutputFormat, print_error};
use crate::session::SessionManager;

#[derive(Subcommand)]
pub enum SpatialCommands {
    /// Spatial Autoregressive Model (SAR / lagsarlm)
    ///
    /// Estimates: y = rho * W * y + X * beta + epsilon
    #[command(after_help = "\
EXAMPLES:
    # SAR with 5 nearest neighbors
    p2a --session s.json spatial sar mydata -y price -x sqft bedrooms \\
        --coord-x longitude --coord-y latitude -k 5

    # Spatial Durbin model with impacts
    p2a --session s.json spatial sar mydata -y price -x sqft bedrooms \\
        --coord-x lon --coord-y lat --durbin --impacts
")]
    Sar {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,

        /// Coordinate X column (for building spatial weights)
        #[arg(long)]
        coord_x: String,

        /// Coordinate Y column (for building spatial weights)
        #[arg(long)]
        coord_y: String,

        /// Number of nearest neighbors for spatial weights (default: 5)
        #[arg(short = 'k', long, default_value = "5")]
        k_neighbors: usize,

        /// Weight style: "rowstd" (default), "binary", "minmax"
        #[arg(long, default_value = "rowstd")]
        weight_style: String,

        /// Estimate Spatial Durbin Model (include spatially lagged X)
        #[arg(long)]
        durbin: bool,

        /// Compute spatial impacts (direct, indirect, total effects)
        #[arg(long)]
        impacts: bool,
    },

    /// Spatial Error Model (SEM / errorsarlm)
    ///
    /// Estimates: y = X * beta + u, where u = lambda * W * u + epsilon
    #[command(after_help = "\
EXAMPLES:
    p2a --session s.json spatial sem mydata -y price -x sqft bedrooms \\
        --coord-x longitude --coord-y latitude -k 5
")]
    Sem {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,

        /// Coordinate X column (for building spatial weights)
        #[arg(long)]
        coord_x: String,

        /// Coordinate Y column (for building spatial weights)
        #[arg(long)]
        coord_y: String,

        /// Number of nearest neighbors for spatial weights (default: 5)
        #[arg(short = 'k', long, default_value = "5")]
        k_neighbors: usize,

        /// Weight style: "rowstd" (default), "binary", "minmax"
        #[arg(long, default_value = "rowstd")]
        weight_style: String,
    },

    /// Spatial Autoregressive Combined Model (SAC / SARAR)
    ///
    /// Estimates: y = rho * W * y + X * beta + u, where u = lambda * W * u + epsilon
    #[command(after_help = "\
EXAMPLES:
    p2a --session s.json spatial sac mydata -y price -x sqft bedrooms \\
        --coord-x longitude --coord-y latitude -k 5
")]
    Sac {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,

        /// Coordinate X column (for building spatial weights)
        #[arg(long)]
        coord_x: String,

        /// Coordinate Y column (for building spatial weights)
        #[arg(long)]
        coord_y: String,

        /// Number of nearest neighbors for spatial weights (default: 5)
        #[arg(short = 'k', long, default_value = "5")]
        k_neighbors: usize,

        /// Weight style: "rowstd" (default), "binary", "minmax"
        #[arg(long, default_value = "rowstd")]
        weight_style: String,
    },

    /// Moran's I test for spatial autocorrelation
    #[command(after_help = "\
EXAMPLES:
    # Test for spatial clustering in residuals
    p2a --session s.json spatial moran mydata -y residuals \\
        --coord-x longitude --coord-y latitude -k 5
")]
    Moran {
        /// Dataset name
        dataset: String,

        /// Variable column to test for spatial autocorrelation
        #[arg(short = 'y', long)]
        variable: String,

        /// Coordinate X column (for building spatial weights)
        #[arg(long)]
        coord_x: String,

        /// Coordinate Y column (for building spatial weights)
        #[arg(long)]
        coord_y: String,

        /// Number of nearest neighbors for spatial weights (default: 5)
        #[arg(short = 'k', long, default_value = "5")]
        k_neighbors: usize,

        /// Weight style: "rowstd" (default), "binary", "minmax"
        #[arg(long, default_value = "rowstd")]
        weight_style: String,

        /// Alternative hypothesis: "two-sided" (default), "greater", "less"
        #[arg(long, default_value = "two-sided")]
        alternative: String,
    },
}

pub fn execute(
    cmd: &SpatialCommands,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    match cmd {
        SpatialCommands::Sar {
            dataset,
            dep_var,
            indep_vars,
            coord_x,
            coord_y,
            k_neighbors,
            weight_style,
            durbin,
            impacts,
        } => execute_sar(
            dataset,
            dep_var,
            indep_vars,
            coord_x,
            coord_y,
            *k_neighbors,
            weight_style,
            *durbin,
            *impacts,
            format,
            session,
        ),
        SpatialCommands::Sem {
            dataset,
            dep_var,
            indep_vars,
            coord_x,
            coord_y,
            k_neighbors,
            weight_style,
        } => execute_sem(
            dataset,
            dep_var,
            indep_vars,
            coord_x,
            coord_y,
            *k_neighbors,
            weight_style,
            format,
            session,
        ),
        SpatialCommands::Sac {
            dataset,
            dep_var,
            indep_vars,
            coord_x,
            coord_y,
            k_neighbors,
            weight_style,
        } => execute_sac(
            dataset,
            dep_var,
            indep_vars,
            coord_x,
            coord_y,
            *k_neighbors,
            weight_style,
            format,
            session,
        ),
        SpatialCommands::Moran {
            dataset,
            variable,
            coord_x,
            coord_y,
            k_neighbors,
            weight_style,
            alternative,
        } => execute_moran(
            dataset,
            variable,
            coord_x,
            coord_y,
            *k_neighbors,
            weight_style,
            alternative,
            format,
            session,
        ),
    }
}

/// Helper to parse weight style from CLI string
fn parse_weight_style(style: &str) -> WeightStyle {
    match style.to_lowercase().as_str() {
        "binary" | "b" => WeightStyle::Binary,
        "minmax" | "mm" => WeightStyle::MinMax,
        _ => WeightStyle::RowStd,
    }
}

/// Helper to build spatial weights from dataset coordinates
fn build_spatial_weights(
    dataset: &p2a_core::Dataset,
    coord_x: &str,
    coord_y: &str,
    k: usize,
    style: WeightStyle,
) -> anyhow::Result<SpatialWeights> {
    let df = dataset.df();

    // Extract coordinates
    let x_col = df
        .column(coord_x)
        .map_err(|_| anyhow::anyhow!("Column '{}' not found", coord_x))?;
    let y_col = df
        .column(coord_y)
        .map_err(|_| anyhow::anyhow!("Column '{}' not found", coord_y))?;

    let x_vals: Vec<f64> = x_col
        .f64()
        .map_err(|_| anyhow::anyhow!("Column '{}' must be numeric", coord_x))?
        .into_no_null_iter()
        .collect();
    let y_vals: Vec<f64> = y_col
        .f64()
        .map_err(|_| anyhow::anyhow!("Column '{}' must be numeric", coord_y))?
        .into_no_null_iter()
        .collect();

    if x_vals.len() != y_vals.len() {
        return Err(anyhow::anyhow!("Coordinate columns have different lengths"));
    }

    let coords: Vec<(f64, f64)> = x_vals.into_iter().zip(y_vals).collect();

    // Build neighbors and weights
    let nb = Neighbors::from_knn(&coords, k);
    let listw = SpatialWeights::from_neighbors(&nb, style);

    Ok(listw)
}

#[allow(clippy::too_many_arguments)]
fn execute_sar(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    coord_x: &str,
    coord_y: &str,
    k_neighbors: usize,
    weight_style: &str,
    durbin: bool,
    compute_impacts: bool,
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
            let style = parse_weight_style(weight_style);
            let mut listw = match build_spatial_weights(ds, coord_x, coord_y, k_neighbors, style) {
                Ok(w) => w,
                Err(e) => {
                    print_error(&format!("Failed to build spatial weights: {}", e), format);
                    return Ok(());
                }
            };

            let x_refs: Vec<&str> = indep_vars.iter().map(|s| s.as_str()).collect();

            let config = SarConfig {
                durbin,
                compute_impacts,
                ..Default::default()
            };

            match run_sar(ds, dep_var, &x_refs, &mut listw, config) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": if result.is_durbin { "Spatial Durbin Model (SDM)" } else { "Spatial Autoregressive Model (SAR)" },
                            "coefficients": result.coef_names.iter().zip(result.coefficients.iter()).map(|(n, c)| {
                                serde_json::json!({ "name": n, "estimate": c })
                            }).collect::<Vec<_>>(),
                            "std_errors": result.std_errors,
                            "z_values": result.z_values,
                            "p_values": result.p_values,
                            "rho": result.rho,
                            "rho_se": result.rho_se,
                            "rho_z": result.rho_z,
                            "rho_p": result.rho_p,
                            "sigma2": result.sigma2,
                            "log_likelihood": result.log_likelihood,
                            "aic": result.aic,
                            "bic": result.bic,
                            "n_obs": result.n_obs,
                            "df": result.df,
                            "impacts": result.impacts.as_ref().map(|imp| {
                                serde_json::json!({
                                    "direct": imp.direct,
                                    "indirect": imp.indirect,
                                    "total": imp.total,
                                    "var_names": imp.var_names,
                                })
                            }),
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        let model_name = if result.is_durbin {
                            "Spatial Durbin Model (SDM)"
                        } else {
                            "Spatial Autoregressive Model (SAR)"
                        };
                        println!("\n{}", model_name);
                        println!("{}", "=".repeat(60));

                        println!("\nCoefficients:");
                        println!(
                            "{:<15} {:>12} {:>12} {:>10} {:>10}",
                            "Variable", "Estimate", "Std.Err", "z-value", "Pr(>|z|)"
                        );
                        println!("{}", "-".repeat(60));
                        for i in 0..result.coefficients.len() {
                            let sig = significance_stars(result.p_values[i]);
                            println!(
                                "{:<15} {:>12.6} {:>12.6} {:>10.4} {:>10.4}{}",
                                result.coef_names[i],
                                result.coefficients[i],
                                result.std_errors[i],
                                result.z_values[i],
                                result.p_values[i],
                                sig
                            );
                        }

                        println!("\nSpatial autoregressive coefficient (rho):");
                        let rho_sig = significance_stars(result.rho_p);
                        println!(
                            "  rho = {:.6} (SE: {:.6}, z: {:.4}, p: {:.4}){}",
                            result.rho, result.rho_se, result.rho_z, result.rho_p, rho_sig
                        );

                        println!("\nModel fit:");
                        println!("  Log-Likelihood: {:.4}", result.log_likelihood);
                        println!("  AIC: {:.4}", result.aic);
                        println!("  BIC: {:.4}", result.bic);
                        println!("  Sigma^2: {:.6}", result.sigma2);
                        println!("  N: {}, df: {}", result.n_obs, result.df);

                        if let Some(impacts) = &result.impacts {
                            println!("\nSpatial Impacts:");
                            println!(
                                "{:<15} {:>12} {:>12} {:>12}",
                                "Variable", "Direct", "Indirect", "Total"
                            );
                            println!("{}", "-".repeat(55));
                            for i in 0..impacts.var_names.len() {
                                println!(
                                    "{:<15} {:>12.6} {:>12.6} {:>12.6}",
                                    impacts.var_names[i],
                                    impacts.direct[i],
                                    impacts.indirect[i],
                                    impacts.total[i]
                                );
                            }
                        }
                    }
                },
                Err(e) => {
                    print_error(&format!("SAR estimation failed: {}", e), format);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_sem(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    coord_x: &str,
    coord_y: &str,
    k_neighbors: usize,
    weight_style: &str,
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
            let style = parse_weight_style(weight_style);
            let mut listw = match build_spatial_weights(ds, coord_x, coord_y, k_neighbors, style) {
                Ok(w) => w,
                Err(e) => {
                    print_error(&format!("Failed to build spatial weights: {}", e), format);
                    return Ok(());
                }
            };

            let x_refs: Vec<&str> = indep_vars.iter().map(|s| s.as_str()).collect();

            let config = SemConfig::default();

            match run_sem(ds, dep_var, &x_refs, &mut listw, config) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": "Spatial Error Model (SEM)",
                            "coefficients": result.coef_names.iter().zip(result.coefficients.iter()).map(|(n, c)| {
                                serde_json::json!({ "name": n, "estimate": c })
                            }).collect::<Vec<_>>(),
                            "std_errors": result.std_errors,
                            "z_values": result.z_values,
                            "p_values": result.p_values,
                            "lambda": result.lambda,
                            "lambda_se": result.lambda_se,
                            "lambda_z": result.lambda_z,
                            "lambda_p": result.lambda_p,
                            "sigma2": result.sigma2,
                            "log_likelihood": result.log_likelihood,
                            "aic": result.aic,
                            "bic": result.bic,
                            "n_obs": result.n_obs,
                            "df": result.df,
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nSpatial Error Model (SEM)");
                        println!("{}", "=".repeat(60));

                        println!("\nCoefficients:");
                        println!(
                            "{:<15} {:>12} {:>12} {:>10} {:>10}",
                            "Variable", "Estimate", "Std.Err", "z-value", "Pr(>|z|)"
                        );
                        println!("{}", "-".repeat(60));
                        for i in 0..result.coefficients.len() {
                            let sig = significance_stars(result.p_values[i]);
                            println!(
                                "{:<15} {:>12.6} {:>12.6} {:>10.4} {:>10.4}{}",
                                result.coef_names[i],
                                result.coefficients[i],
                                result.std_errors[i],
                                result.z_values[i],
                                result.p_values[i],
                                sig
                            );
                        }

                        println!("\nSpatial error coefficient (lambda):");
                        let lambda_sig = significance_stars(result.lambda_p);
                        println!(
                            "  lambda = {:.6} (SE: {:.6}, z: {:.4}, p: {:.4}){}",
                            result.lambda,
                            result.lambda_se,
                            result.lambda_z,
                            result.lambda_p,
                            lambda_sig
                        );

                        println!("\nModel fit:");
                        println!("  Log-Likelihood: {:.4}", result.log_likelihood);
                        println!("  AIC: {:.4}", result.aic);
                        println!("  BIC: {:.4}", result.bic);
                        println!("  Sigma^2: {:.6}", result.sigma2);
                        println!("  N: {}, df: {}", result.n_obs, result.df);
                    }
                },
                Err(e) => {
                    print_error(&format!("SEM estimation failed: {}", e), format);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_sac(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    coord_x: &str,
    coord_y: &str,
    k_neighbors: usize,
    weight_style: &str,
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
            let style = parse_weight_style(weight_style);
            let mut listw = match build_spatial_weights(ds, coord_x, coord_y, k_neighbors, style) {
                Ok(w) => w,
                Err(e) => {
                    print_error(&format!("Failed to build spatial weights: {}", e), format);
                    return Ok(());
                }
            };

            let x_refs: Vec<&str> = indep_vars.iter().map(|s| s.as_str()).collect();

            let config = SacConfig::default();

            match run_sac(ds, dep_var, &x_refs, &mut listw, config) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": "Spatial Autoregressive Combined Model (SAC/SARAR)",
                            "coefficients": result.coef_names.iter().zip(result.coefficients.iter()).map(|(n, c)| {
                                serde_json::json!({ "name": n, "estimate": c })
                            }).collect::<Vec<_>>(),
                            "std_errors": result.std_errors,
                            "z_values": result.z_values,
                            "p_values": result.p_values,
                            "rho": result.rho,
                            "rho_se": result.rho_se,
                            "rho_z": result.rho_z,
                            "rho_p": result.rho_p,
                            "lambda": result.lambda,
                            "lambda_se": result.lambda_se,
                            "lambda_z": result.lambda_z,
                            "lambda_p": result.lambda_p,
                            "sigma2": result.sigma2,
                            "log_likelihood": result.log_likelihood,
                            "aic": result.aic,
                            "bic": result.bic,
                            "n_obs": result.n_obs,
                            "df": result.df,
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nSpatial Autoregressive Combined Model (SAC/SARAR)");
                        println!("{}", "=".repeat(60));

                        println!("\nCoefficients:");
                        println!(
                            "{:<15} {:>12} {:>12} {:>10} {:>10}",
                            "Variable", "Estimate", "Std.Err", "z-value", "Pr(>|z|)"
                        );
                        println!("{}", "-".repeat(60));
                        for i in 0..result.coefficients.len() {
                            let sig = significance_stars(result.p_values[i]);
                            println!(
                                "{:<15} {:>12.6} {:>12.6} {:>10.4} {:>10.4}{}",
                                result.coef_names[i],
                                result.coefficients[i],
                                result.std_errors[i],
                                result.z_values[i],
                                result.p_values[i],
                                sig
                            );
                        }

                        println!("\nSpatial autoregressive coefficient (rho):");
                        let rho_sig = significance_stars(result.rho_p);
                        println!(
                            "  rho = {:.6} (SE: {:.6}, z: {:.4}, p: {:.4}){}",
                            result.rho, result.rho_se, result.rho_z, result.rho_p, rho_sig
                        );

                        println!("\nSpatial error coefficient (lambda):");
                        let lambda_sig = significance_stars(result.lambda_p);
                        println!(
                            "  lambda = {:.6} (SE: {:.6}, z: {:.4}, p: {:.4}){}",
                            result.lambda,
                            result.lambda_se,
                            result.lambda_z,
                            result.lambda_p,
                            lambda_sig
                        );

                        println!("\nModel fit:");
                        println!("  Log-Likelihood: {:.4}", result.log_likelihood);
                        println!("  AIC: {:.4}", result.aic);
                        println!("  BIC: {:.4}", result.bic);
                        println!("  Sigma^2: {:.6}", result.sigma2);
                        println!("  N: {}, df: {}", result.n_obs, result.df);
                    }
                },
                Err(e) => {
                    print_error(&format!("SAC estimation failed: {}", e), format);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_moran(
    dataset_name: &str,
    variable: &str,
    coord_x: &str,
    coord_y: &str,
    k_neighbors: usize,
    weight_style: &str,
    alternative: &str,
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
            let style = parse_weight_style(weight_style);
            let listw = match build_spatial_weights(ds, coord_x, coord_y, k_neighbors, style) {
                Ok(w) => w,
                Err(e) => {
                    print_error(&format!("Failed to build spatial weights: {}", e), format);
                    return Ok(());
                }
            };

            // Extract the variable as an array
            let df = ds.df();
            let y_col = df
                .column(variable)
                .map_err(|_| anyhow::anyhow!("Column '{}' not found", variable))?;
            let y_vals: Vec<f64> = y_col
                .f64()
                .map_err(|_| anyhow::anyhow!("Column '{}' must be numeric", variable))?
                .into_no_null_iter()
                .collect();
            let y = ndarray::Array1::from_vec(y_vals);

            let alt = match alternative.to_lowercase().as_str() {
                "greater" => MoranAlternative::Greater,
                "less" => MoranAlternative::Less,
                _ => MoranAlternative::TwoSided,
            };

            match moran_test(&y, &listw, alt) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": "Moran's I Test for Spatial Autocorrelation",
                            "variable": variable,
                            "moran_i": result.statistic,
                            "expected_i": result.expectation,
                            "variance": result.variance,
                            "z_score": result.z_score,
                            "p_value": result.p_value,
                            "alternative": format!("{:?}", result.alternative),
                            "n": result.n,
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nMoran's I Test for Spatial Autocorrelation");
                        println!("{}", "=".repeat(50));
                        println!("\nVariable: {}", variable);
                        println!("Alternative: {:?}", result.alternative);
                        println!();
                        println!("Moran's I: {:.6}", result.statistic);
                        println!("Expected I: {:.6}", result.expectation);
                        println!("Variance: {:.6}", result.variance);
                        println!("z-score: {:.4}", result.z_score);
                        let sig = significance_stars(result.p_value);
                        println!("p-value: {:.4}{}", result.p_value, sig);
                        println!("N: {}", result.n);

                        println!();
                        if result.p_value < 0.05 {
                            if result.statistic > result.expectation {
                                println!(
                                    "Conclusion: Evidence of positive spatial autocorrelation"
                                );
                                println!("            (similar values cluster together)");
                            } else {
                                println!(
                                    "Conclusion: Evidence of negative spatial autocorrelation"
                                );
                                println!("            (dissimilar values cluster together)");
                            }
                        } else {
                            println!("Conclusion: No significant spatial autocorrelation detected");
                        }
                    }
                },
                Err(e) => {
                    print_error(&format!("Moran's I test failed: {}", e), format);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

/// Helper function to add significance stars
fn significance_stars(p: f64) -> &'static str {
    if p < 0.001 {
        " ***"
    } else if p < 0.01 {
        " **"
    } else if p < 0.05 {
        " *"
    } else if p < 0.1 {
        " ."
    } else {
        ""
    }
}
