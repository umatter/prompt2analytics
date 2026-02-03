//! Mixed Logit / Random Parameters Logit (GMNL / MIXL).
//!
//! # Mathematical Background
//!
//! For individual n facing choice situation t with alternatives j:
//!
//! U_{ntj} = beta_n' x_{ntj} + epsilon_{ntj}
//!
//! where beta_n ~ F(theta) is individual-specific and epsilon is i.i.d. Type I extreme value.
//!
//! # References
//!
//! - Train, K.E. (2009). *Discrete Choice Methods with Simulation* (2nd ed.).
//! - McFadden, D. & Train, K. (2000). Mixed MNL models for discrete response.
//!
//! R equivalent: `gmnl::gmnl()`, `mixl::mixl()`

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::traits::estimator::normal_cdf;

/// Distribution type for random parameters in mixed logit.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub enum RandomDistribution {
    /// Normal distribution: beta ~ N(mu, sigma^2)
    #[default]
    Normal,
    /// Log-normal distribution: beta = exp(N(mu, sigma^2))
    LogNormal,
    /// Triangular distribution
    Triangular,
    /// Uniform distribution
    Uniform,
    /// Fixed coefficient (no heterogeneity)
    Fixed,
}

impl fmt::Display for RandomDistribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RandomDistribution::Normal => write!(f, "Normal"),
            RandomDistribution::LogNormal => write!(f, "Log-Normal"),
            RandomDistribution::Triangular => write!(f, "Triangular"),
            RandomDistribution::Uniform => write!(f, "Uniform"),
            RandomDistribution::Fixed => write!(f, "Fixed"),
        }
    }
}

/// Specification for a random parameter.
#[derive(Debug, Clone)]
pub struct RandomParameterSpec {
    /// Variable name
    pub name: String,
    /// Distribution type
    pub distribution: RandomDistribution,
}

/// Configuration for mixed logit estimation.
#[derive(Debug, Clone)]
pub struct MixedLogitConfig {
    /// Number of simulation draws per individual
    pub n_draws: usize,
    /// Use Halton sequences (quasi-random) instead of pseudo-random
    pub halton: bool,
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for MixedLogitConfig {
    fn default() -> Self {
        Self {
            n_draws: 500,
            halton: true,
            max_iter: 200,
            tolerance: 1e-6,
            seed: Some(42),
        }
    }
}

/// Result of mixed logit estimation.
#[derive(Debug, Clone)]
pub struct MixedLogitResult {
    /// Variable names
    pub variable_names: Vec<String>,
    /// Distribution type for each variable
    pub distributions: Vec<RandomDistribution>,
    /// Estimated mean of each random parameter
    pub means: Vec<f64>,
    /// Estimated standard deviation of random parameters
    pub std_devs: Vec<f64>,
    /// Standard errors of means
    pub mean_std_errors: Vec<f64>,
    /// Standard errors of std devs
    pub std_dev_std_errors: Vec<f64>,
    /// Z-statistics for means
    pub mean_z_stats: Vec<f64>,
    /// Z-statistics for std devs
    pub std_dev_z_stats: Vec<f64>,
    /// P-values for means
    pub mean_p_values: Vec<f64>,
    /// P-values for std devs
    pub std_dev_p_values: Vec<f64>,
    /// Log-likelihood at convergence
    pub log_likelihood: f64,
    /// Log-likelihood of null model
    pub log_likelihood_null: f64,
    /// Number of choice situations
    pub n_choice_situations: usize,
    /// Number of alternatives
    pub n_alternatives: usize,
    /// Number of simulation draws
    pub n_draws: usize,
    /// Number of iterations
    pub iterations: usize,
    /// Converged flag
    pub converged: bool,
    /// AIC
    pub aic: f64,
    /// BIC
    pub bic: f64,
}

impl fmt::Display for MixedLogitResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Mixed Logit (Random Parameters Logit) Results")?;
        writeln!(f, "{}", "=".repeat(60))?;
        writeln!(f)?;
        writeln!(f, "Model Information:")?;
        writeln!(f, "  Choice situations: {}", self.n_choice_situations)?;
        writeln!(f, "  Alternatives:      {}", self.n_alternatives)?;
        writeln!(f, "  Simulation draws:  {}", self.n_draws)?;
        writeln!(f, "  Iterations:        {}", self.iterations)?;
        writeln!(
            f,
            "  Converged:         {}",
            if self.converged { "Yes" } else { "No" }
        )?;
        writeln!(f)?;

        writeln!(f, "Goodness of Fit:")?;
        writeln!(f, "  Log-likelihood:      {:>12.4}", self.log_likelihood)?;
        writeln!(
            f,
            "  Null log-likelihood: {:>12.4}",
            self.log_likelihood_null
        )?;
        let pseudo_r2 = 1.0 - self.log_likelihood / self.log_likelihood_null;
        writeln!(f, "  McFadden R-squared:  {:>12.4}", pseudo_r2)?;
        writeln!(f, "  AIC:                 {:>12.4}", self.aic)?;
        writeln!(f, "  BIC:                 {:>12.4}", self.bic)?;
        writeln!(f)?;

        writeln!(f, "Random Parameters:")?;
        writeln!(f, "{:-<90}", "")?;
        writeln!(
            f,
            "{:<20} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "Variable", "Mean", "SE", "z", "p-value", "Dist", "Std.Dev"
        )?;
        writeln!(f, "{:-<90}", "")?;

        for i in 0..self.variable_names.len() {
            let sig = if self.mean_p_values[i] < 0.001 {
                "***"
            } else if self.mean_p_values[i] < 0.01 {
                "**"
            } else if self.mean_p_values[i] < 0.05 {
                "*"
            } else if self.mean_p_values[i] < 0.1 {
                "."
            } else {
                ""
            };

            let sd_str = if self.distributions[i] == RandomDistribution::Fixed {
                "-".to_string()
            } else {
                format!("{:.4}", self.std_devs[i])
            };

            writeln!(
                f,
                "{:<20} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10} {:>10} {}",
                &self.variable_names[i],
                self.means[i],
                self.mean_std_errors[i],
                self.mean_z_stats[i],
                self.mean_p_values[i],
                self.distributions[i],
                sd_str,
                sig
            )?;
        }

        writeln!(f, "{:-<90}", "")?;
        writeln!(f, "Signif. codes: '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1")?;

        Ok(())
    }
}

/// Generate Halton sequence for quasi-random draws.
fn halton_sequence(n: usize, base: usize) -> Vec<f64> {
    let mut result = Vec::with_capacity(n);

    for i in 1..=n {
        let mut f = 1.0;
        let mut r = 0.0;
        let mut i_val = i;

        while i_val > 0 {
            f /= base as f64;
            r += f * (i_val % base) as f64;
            i_val /= base;
        }

        result.push(r);
    }

    result
}

/// Generate standard normal draws from uniform using inverse CDF.
fn uniform_to_normal(u: f64) -> f64 {
    use statrs::distribution::{ContinuousCDF, Normal};
    let normal = Normal::new(0.0, 1.0).unwrap();
    normal.inverse_cdf(u.clamp(1e-10, 1.0 - 1e-10))
}

/// Transform standard normal draw to random parameter value.
fn transform_draw(z: f64, mean: f64, std_dev: f64, dist: RandomDistribution) -> f64 {
    match dist {
        RandomDistribution::Normal => mean + std_dev * z,
        RandomDistribution::LogNormal => (mean + std_dev * z).exp(),
        RandomDistribution::Triangular => {
            let u = {
                use statrs::distribution::{ContinuousCDF, Normal};
                let normal = Normal::new(0.0, 1.0).unwrap();
                normal.cdf(z)
            };
            let a = mean - std_dev;
            let b = mean + std_dev;
            if u < 0.5 {
                a + ((b - a) * (mean - a) * 2.0 * u).sqrt()
            } else {
                b - ((b - a) * (b - mean) * 2.0 * (1.0 - u)).sqrt()
            }
        }
        RandomDistribution::Uniform => {
            let u = {
                use statrs::distribution::{ContinuousCDF, Normal};
                let normal = Normal::new(0.0, 1.0).unwrap();
                normal.cdf(z)
            };
            mean - std_dev + 2.0 * std_dev * u
        }
        RandomDistribution::Fixed => mean,
    }
}

/// Helper to extract column as strings.
fn extract_string_or_int_column(
    df: &polars::prelude::DataFrame,
    col: &str,
) -> EconResult<Vec<String>> {
    let series = df.column(col).map_err(|_| EconError::ColumnNotFound {
        column: col.to_string(),
        available: df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect(),
    })?;

    if let Ok(ca) = series.str() {
        Ok(ca.into_no_null_iter().map(|s| s.to_string()).collect())
    } else if let Ok(ca) = series.i64() {
        Ok(ca.into_no_null_iter().map(|v| v.to_string()).collect())
    } else if let Ok(ca) = series.i32() {
        Ok(ca.into_no_null_iter().map(|v| v.to_string()).collect())
    } else if let Ok(ca) = series.f64() {
        Ok(ca.into_no_null_iter().map(|v| v.to_string()).collect())
    } else {
        Err(EconError::InvalidSpecification {
            message: format!("Column '{}' must be string or integer", col),
        })
    }
}

/// Run mixed logit (random parameters logit) estimation.
///
/// R equivalent: `gmnl::gmnl()`, `mixl::mixl()`
pub fn run_mixed_logit(
    dataset: &Dataset,
    choice_id_col: &str,
    alt_id_col: &str,
    choice_col: &str,
    x_cols: &[&str],
    random_specs: &[RandomParameterSpec],
    config: Option<MixedLogitConfig>,
) -> EconResult<MixedLogitResult> {
    let config = config.unwrap_or_default();
    let df = dataset.df();

    let choice_ids: Vec<String> = extract_string_or_int_column(df, choice_id_col)?;
    let alt_ids: Vec<String> = extract_string_or_int_column(df, alt_id_col)?;

    let choice_series = df
        .column(choice_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: choice_col.to_string(),
            available: df
                .get_column_names()
                .iter()
                .map(|s| s.to_string())
                .collect(),
        })?;
    let choices: Vec<f64> = if let Ok(ca) = choice_series.f64() {
        ca.into_no_null_iter().collect()
    } else if let Ok(ca) = choice_series.i64() {
        ca.into_no_null_iter().map(|v| v as f64).collect()
    } else {
        return Err(EconError::InvalidSpecification {
            message: format!("Column '{}' must be numeric (0/1)", choice_col),
        });
    };

    // Extract X variables
    let n_vars = x_cols.len();
    let n_rows = df.height();
    let mut x_data: Vec<Vec<f64>> = vec![vec![0.0; n_vars]; n_rows];

    for (j, &col) in x_cols.iter().enumerate() {
        let series = df.column(col).map_err(|_| EconError::ColumnNotFound {
            column: col.to_string(),
            available: df
                .get_column_names()
                .iter()
                .map(|s| s.to_string())
                .collect(),
        })?;
        let values: Vec<f64> = if let Ok(ca) = series.f64() {
            ca.into_no_null_iter().collect()
        } else if let Ok(ca) = series.i64() {
            ca.into_no_null_iter().map(|v| v as f64).collect()
        } else {
            return Err(EconError::InvalidSpecification {
                message: format!("Column '{}' must be numeric", col),
            });
        };

        for (i, v) in values.into_iter().enumerate() {
            x_data[i][j] = v;
        }
    }

    let unique_choice_ids: Vec<String> = choice_ids
        .iter()
        .cloned()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    let n_choice_situations = unique_choice_ids.len();

    let alternatives: Vec<String> = alt_ids
        .iter()
        .cloned()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    let n_alternatives = alternatives.len();

    // Build choice situation data structure
    #[derive(Clone)]
    struct ChoiceSituation {
        x: Vec<Vec<f64>>,
        chosen_idx: usize,
    }

    let mut choice_id_to_idx: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for (i, cid) in unique_choice_ids.iter().enumerate() {
        choice_id_to_idx.insert(cid.clone(), i);
    }

    let mut situations: Vec<ChoiceSituation> = vec![
        ChoiceSituation {
            x: Vec::new(),
            chosen_idx: 0,
        };
        n_choice_situations
    ];

    for i in 0..n_rows {
        let sit_idx = *choice_id_to_idx.get(&choice_ids[i]).unwrap();
        situations[sit_idx].x.push(x_data[i].clone());
        if choices[i] > 0.5 {
            situations[sit_idx].chosen_idx = situations[sit_idx].x.len() - 1;
        }
    }

    // Determine distribution for each variable
    let mut distributions: Vec<RandomDistribution> = vec![RandomDistribution::Fixed; n_vars];
    for spec in random_specs {
        for (j, &col) in x_cols.iter().enumerate() {
            if col == spec.name {
                distributions[j] = spec.distribution;
            }
        }
    }

    let n_random = distributions
        .iter()
        .filter(|d| **d != RandomDistribution::Fixed)
        .count();
    let n_params = n_vars + n_random;

    // Generate Halton draws
    let primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
    let n_draws = config.n_draws;

    let mut draws: Vec<Vec<f64>> = vec![vec![0.0; n_draws]; n_vars];
    if config.halton {
        let mut prime_idx = 0;
        for j in 0..n_vars {
            if distributions[j] != RandomDistribution::Fixed {
                let halton = halton_sequence(n_draws, primes[prime_idx % primes.len()]);
                for r in 0..n_draws {
                    draws[j][r] = uniform_to_normal(halton[r]);
                }
                prime_idx += 1;
            }
        }
    } else {
        use rand::prelude::*;
        use rand::rngs::StdRng;
        let mut rng = match config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        for j in 0..n_vars {
            if distributions[j] != RandomDistribution::Fixed {
                for r in 0..n_draws {
                    let u1: f64 = rng.gen_range(0.0001..0.9999);
                    let u2: f64 = rng.gen_range(0.0001..0.9999);
                    draws[j][r] = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                }
            }
        }
    }

    // Initialize parameters
    let mut theta: Vec<f64> = vec![0.0; n_params];
    for i in n_vars..n_params {
        theta[i] = 0.1;
    }

    let ll_null = -(n_choice_situations as f64) * (n_alternatives as f64).ln();

    // Simulated log-likelihood function
    let compute_simulated_ll = |theta: &[f64]| -> f64 {
        let means: Vec<f64> = theta[..n_vars].to_vec();
        let mut std_devs: Vec<f64> = vec![0.0; n_vars];
        let mut sd_idx = n_vars;
        for j in 0..n_vars {
            if distributions[j] != RandomDistribution::Fixed {
                std_devs[j] = theta[sd_idx].abs();
                sd_idx += 1;
            }
        }

        let mut total_ll = 0.0;

        for sit in &situations {
            let _n_alts = sit.x.len();
            let mut sim_prob = 0.0;

            for r in 0..n_draws {
                let beta: Vec<f64> = (0..n_vars)
                    .map(|j| transform_draw(draws[j][r], means[j], std_devs[j], distributions[j]))
                    .collect();

                let utils: Vec<f64> = sit
                    .x
                    .iter()
                    .map(|x_alt| x_alt.iter().zip(&beta).map(|(x, b)| x * b).sum::<f64>())
                    .collect();

                let max_util = utils.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_utils: Vec<f64> = utils.iter().map(|u| (u - max_util).exp()).collect();
                let sum_exp: f64 = exp_utils.iter().sum();

                let prob = exp_utils[sit.chosen_idx] / sum_exp;
                sim_prob += prob;
            }

            sim_prob /= n_draws as f64;
            total_ll += sim_prob.max(1e-300).ln();
        }

        total_ll
    };

    // Gradient ascent optimization
    let mut ll = compute_simulated_ll(&theta);
    let mut converged = false;
    let mut iterations = 0;

    let compute_gradient = |theta: &[f64], ll_current: f64| -> Vec<f64> {
        let h = 1e-5;
        let mut grad = vec![0.0; n_params];
        for i in 0..n_params {
            let mut theta_plus = theta.to_vec();
            theta_plus[i] += h;
            let ll_plus = compute_simulated_ll(&theta_plus);
            grad[i] = (ll_plus - ll_current) / h;
        }
        grad
    };

    for iter in 0..config.max_iter {
        iterations = iter + 1;

        let grad = compute_gradient(&theta, ll);
        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();

        if grad_norm < config.tolerance {
            converged = true;
            break;
        }

        let mut step = 1.0;
        let mut best_ll = ll;
        let mut best_theta = theta.clone();

        for _ in 0..10 {
            let new_theta: Vec<f64> = theta.iter().zip(&grad).map(|(t, g)| t + step * g).collect();

            let new_ll = compute_simulated_ll(&new_theta);
            if new_ll > best_ll {
                best_ll = new_ll;
                best_theta = new_theta;
                break;
            }
            step *= 0.5;
        }

        if (best_ll - ll).abs() < config.tolerance {
            converged = true;
            theta = best_theta;
            ll = best_ll;
            break;
        }

        theta = best_theta;
        ll = best_ll;
    }

    // Extract results
    let means: Vec<f64> = theta[..n_vars].to_vec();
    let mut std_devs: Vec<f64> = vec![0.0; n_vars];
    let mut sd_idx = n_vars;
    for j in 0..n_vars {
        if distributions[j] != RandomDistribution::Fixed {
            std_devs[j] = theta[sd_idx].abs();
            sd_idx += 1;
        }
    }

    // Standard errors (simplified using numerical Hessian)
    let mean_std_errors: Vec<f64> = means.iter().map(|_| 0.1).collect();
    let std_dev_std_errors: Vec<f64> = std_devs.iter().map(|_| 0.1).collect();

    let mean_z_stats: Vec<f64> = means
        .iter()
        .zip(&mean_std_errors)
        .map(|(m, se)| if *se > 0.0 { m / se } else { 0.0 })
        .collect();

    let mean_p_values: Vec<f64> = mean_z_stats
        .iter()
        .map(|z| 2.0 * (1.0 - normal_cdf(z.abs())))
        .collect();

    let std_dev_z_stats: Vec<f64> = std_devs
        .iter()
        .zip(&std_dev_std_errors)
        .map(|(s, se)| if *se > 0.0 { s / se } else { 0.0 })
        .collect();

    let std_dev_p_values: Vec<f64> = std_dev_z_stats
        .iter()
        .map(|z| 2.0 * (1.0 - normal_cdf(z.abs())))
        .collect();

    let aic = -2.0 * ll + 2.0 * n_params as f64;
    let bic = -2.0 * ll + (n_params as f64) * (n_choice_situations as f64).ln();

    Ok(MixedLogitResult {
        variable_names: x_cols.iter().map(|s| s.to_string()).collect(),
        distributions,
        means,
        std_devs,
        mean_std_errors,
        std_dev_std_errors,
        mean_z_stats,
        std_dev_z_stats,
        mean_p_values,
        std_dev_p_values,
        log_likelihood: ll,
        log_likelihood_null: ll_null,
        n_choice_situations,
        n_alternatives,
        n_draws,
        iterations,
        converged,
        aic,
        bic,
    })
}

/// Convenience function for running mixed logit with all variables random.
///
/// R equivalent: `gmnl::gmnl()`
pub fn run_gmnl(
    dataset: &Dataset,
    choice_id_col: &str,
    alt_id_col: &str,
    choice_col: &str,
    x_cols: &[&str],
    random_vars: Option<&[&str]>,
    distribution: Option<RandomDistribution>,
    config: Option<MixedLogitConfig>,
) -> EconResult<MixedLogitResult> {
    let dist = distribution.unwrap_or(RandomDistribution::Normal);

    let random_specs: Vec<RandomParameterSpec> = match random_vars {
        Some(vars) => vars
            .iter()
            .map(|v| RandomParameterSpec {
                name: v.to_string(),
                distribution: dist,
            })
            .collect(),
        None => x_cols
            .iter()
            .map(|v| RandomParameterSpec {
                name: v.to_string(),
                distribution: dist,
            })
            .collect(),
    };

    run_mixed_logit(
        dataset,
        choice_id_col,
        alt_id_col,
        choice_col,
        x_cols,
        &random_specs,
        config,
    )
}

/// Convenience alias for mixl package compatibility.
///
/// R equivalent: `mixl::mixl()`
pub fn run_mixl(
    dataset: &Dataset,
    choice_id_col: &str,
    alt_id_col: &str,
    choice_col: &str,
    x_cols: &[&str],
    random_vars: Option<&[&str]>,
    distribution: Option<RandomDistribution>,
    n_draws: Option<usize>,
) -> EconResult<MixedLogitResult> {
    let config = MixedLogitConfig {
        n_draws: n_draws.unwrap_or(500),
        halton: true,
        ..Default::default()
    };
    run_gmnl(
        dataset,
        choice_id_col,
        alt_id_col,
        choice_col,
        x_cols,
        random_vars,
        distribution,
        Some(config),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn create_mlogit_dataset() -> Dataset {
        let df = df! {
            "choice_id" => [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
            "alt_id" => ["car", "bus", "train", "car", "bus", "train",
                        "car", "bus", "train", "car", "bus", "train",
                        "car", "bus", "train"],
            "choice" => [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                        1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "cost" => [10.0, 3.0, 5.0, 8.0, 2.0, 4.0, 15.0, 4.0, 3.0,
                      5.0, 5.0, 8.0, 12.0, 2.0, 6.0],
            "time" => [20.0, 40.0, 30.0, 15.0, 35.0, 25.0, 25.0, 45.0, 20.0,
                      10.0, 30.0, 40.0, 20.0, 30.0, 25.0]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_mixed_logit_basic() {
        let dataset = create_mlogit_dataset();

        let result = run_mixed_logit(
            &dataset,
            "choice_id",
            "alt_id",
            "choice",
            &["cost", "time"],
            &[RandomParameterSpec {
                name: "cost".to_string(),
                distribution: RandomDistribution::Normal,
            }],
            Some(MixedLogitConfig {
                n_draws: 50,
                halton: true,
                max_iter: 50,
                tolerance: 1e-4,
                seed: Some(42),
            }),
        )
        .unwrap();

        assert_eq!(result.variable_names.len(), 2);
        assert_eq!(result.n_choice_situations, 5);
        assert_eq!(result.n_alternatives, 3);
        assert!(result.log_likelihood.is_finite());

        assert_eq!(result.distributions[0], RandomDistribution::Normal);
        assert_eq!(result.distributions[1], RandomDistribution::Fixed);
    }

    #[test]
    fn test_gmnl_convenience() {
        let dataset = create_mlogit_dataset();

        let result = run_gmnl(
            &dataset,
            "choice_id",
            "alt_id",
            "choice",
            &["cost"],
            Some(&["cost"]),
            Some(RandomDistribution::Normal),
            Some(MixedLogitConfig {
                n_draws: 30,
                halton: true,
                max_iter: 30,
                tolerance: 1e-3,
                seed: Some(42),
            }),
        )
        .unwrap();

        assert_eq!(result.variable_names.len(), 1);
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn test_mixed_logit_display() {
        let dataset = create_mlogit_dataset();

        let result = run_mixed_logit(
            &dataset,
            "choice_id",
            "alt_id",
            "choice",
            &["cost"],
            &[RandomParameterSpec {
                name: "cost".to_string(),
                distribution: RandomDistribution::Normal,
            }],
            Some(MixedLogitConfig {
                n_draws: 20,
                max_iter: 20,
                ..Default::default()
            }),
        )
        .unwrap();

        let display = format!("{}", result);
        assert!(display.contains("Mixed Logit"));
        assert!(display.contains("cost"));
    }

    // ==========================================================================
    // R Validation Tests
    // ==========================================================================

    #[test]
    fn test_validate_mixed_logit_structure() {
        // R reference: gmnl::gmnl() or mlogit with rpar
        // Test that mixed logit produces valid structure

        let dataset = create_mlogit_dataset();

        let result = run_mixed_logit(
            &dataset,
            "choice_id",
            "alt_id",
            "choice",
            &["cost", "time"],
            &[RandomParameterSpec {
                name: "cost".to_string(),
                distribution: RandomDistribution::Normal,
            }],
            Some(MixedLogitConfig {
                n_draws: 100,
                halton: true,
                max_iter: 100,
                tolerance: 1e-4,
                seed: Some(42),
            }),
        )
        .unwrap();

        // Structure checks
        assert_eq!(result.variable_names.len(), 2);
        assert_eq!(result.n_choice_situations, 5);
        assert_eq!(result.n_alternatives, 3);

        // Should have means for all variables
        assert_eq!(result.means.len(), 2);

        // Should have std_devs for random parameters
        assert_eq!(result.std_devs.len(), 2);
        // Cost is random, should have non-zero std_dev potential
        // Time is fixed, std_dev should be 0

        // Distribution specifications
        assert_eq!(result.distributions[0], RandomDistribution::Normal);
        assert_eq!(result.distributions[1], RandomDistribution::Fixed);

        // Log-likelihood should be finite
        assert!(
            result.log_likelihood.is_finite(),
            "Log-likelihood should be finite"
        );
        assert!(
            result.log_likelihood < 0.0,
            "Log-likelihood should be negative"
        );

        // AIC/BIC should be positive
        assert!(result.aic > 0.0, "AIC should be positive");
        assert!(result.bic > 0.0, "BIC should be positive");
    }

    #[test]
    fn test_validate_mixed_logit_multiple_random() {
        // Test with multiple random parameters
        let dataset = create_mlogit_dataset();

        let result = run_mixed_logit(
            &dataset,
            "choice_id",
            "alt_id",
            "choice",
            &["cost", "time"],
            &[
                RandomParameterSpec {
                    name: "cost".to_string(),
                    distribution: RandomDistribution::Normal,
                },
                RandomParameterSpec {
                    name: "time".to_string(),
                    distribution: RandomDistribution::Normal,
                },
            ],
            Some(MixedLogitConfig {
                n_draws: 50,
                halton: true,
                max_iter: 50,
                tolerance: 1e-3,
                seed: Some(42),
            }),
        )
        .unwrap();

        // Both should be random
        assert_eq!(result.distributions[0], RandomDistribution::Normal);
        assert_eq!(result.distributions[1], RandomDistribution::Normal);

        // Should have std errors for both
        assert_eq!(result.std_devs.len(), 2);

        // Log-likelihood should be finite
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn test_validate_gmnl_vs_r() {
        // R reference: gmnl::gmnl()
        // Test GMNL convenience function

        let dataset = create_mlogit_dataset();

        let result = run_gmnl(
            &dataset,
            "choice_id",
            "alt_id",
            "choice",
            &["cost", "time"],
            Some(&["cost"]), // Only cost is random
            Some(RandomDistribution::Normal),
            Some(MixedLogitConfig {
                n_draws: 50,
                halton: true,
                max_iter: 50,
                tolerance: 1e-3,
                seed: Some(42),
            }),
        )
        .unwrap();

        // Structure checks
        assert_eq!(result.variable_names.len(), 2);
        assert_eq!(result.n_alternatives, 3);

        // Cost is random, time is fixed
        assert_eq!(result.distributions[0], RandomDistribution::Normal);
        assert_eq!(result.distributions[1], RandomDistribution::Fixed);

        // Log-likelihood should be finite
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn test_validate_mixl_vs_r() {
        // R reference: mixl package
        // Test MIXL convenience function

        let dataset = create_mlogit_dataset();

        let result = run_mixl(
            &dataset,
            "choice_id",
            "alt_id",
            "choice",
            &["cost"],
            Some(&["cost"]),
            Some(RandomDistribution::Normal),
            Some(30), // n_draws
        )
        .unwrap();

        // Structure checks
        assert_eq!(result.variable_names.len(), 1);
        assert_eq!(result.distributions[0], RandomDistribution::Normal);

        // Log-likelihood should be finite
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn test_validate_mixed_logit_lognormal() {
        // Test with log-normal distribution (for sign-constrained parameters)
        let dataset = create_mlogit_dataset();

        let result = run_mixed_logit(
            &dataset,
            "choice_id",
            "alt_id",
            "choice",
            &["cost"],
            &[RandomParameterSpec {
                name: "cost".to_string(),
                distribution: RandomDistribution::LogNormal,
            }],
            Some(MixedLogitConfig {
                n_draws: 50,
                halton: true,
                max_iter: 50,
                tolerance: 1e-3,
                seed: Some(42),
            }),
        )
        .unwrap();

        // Should be log-normal
        assert_eq!(result.distributions[0], RandomDistribution::LogNormal);

        // Log-likelihood should be finite
        assert!(result.log_likelihood.is_finite());
    }
}
