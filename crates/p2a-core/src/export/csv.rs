//! CSV export functionality for analysis results.
//!
//! Provides a generic trait for exporting results to CSV format,
//! suitable for:
//! - Data interchange with other tools (Excel, R, Python)
//! - Archiving results
//! - Downstream processing

use std::io::Write;
use std::path::Path;

use crate::econometrics::{DiscreteResult, HausmanResult, PanelResult};
use crate::ml::{DBSCANResult, KMeansResult, PCAResult};
use crate::regression::OlsResult;

/// Trait for exporting results to CSV format.
pub trait CsvExport {
    /// Export to a CSV string.
    fn to_csv_string(&self) -> String;

    /// Export to a CSV file.
    fn to_csv(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let content = self.to_csv_string();
        std::fs::write(path, content)
    }

    /// Export to a writer.
    fn write_csv<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(self.to_csv_string().as_bytes())
    }
}

impl CsvExport for OlsResult {
    fn to_csv_string(&self) -> String {
        let mut csv = String::new();
        csv.push_str("variable,estimate,std_error,t_value,p_value,significance\n");

        for coef in &self.coefficients {
            csv.push_str(&format!(
                "{},{:.6},{:.6},{:.6},{:.6},{}\n",
                escape_csv(&coef.name),
                coef.estimate,
                coef.std_error,
                coef.t_value,
                coef.p_value,
                coef.significance.code()
            ));
        }

        // Model statistics
        csv.push_str("\n# Model Statistics\n");
        csv.push_str("statistic,value\n");
        csv.push_str(&format!("n_obs,{}\n", self.n_obs));
        csv.push_str(&format!("r_squared,{:.6}\n", self.r_squared));
        csv.push_str(&format!("adj_r_squared,{:.6}\n", self.adj_r_squared));
        csv.push_str(&format!("f_statistic,{:.6}\n", self.f_statistic));
        csv.push_str(&format!("f_p_value,{:.6}\n", self.f_p_value));
        csv.push_str(&format!("df_model,{}\n", self.df_model));
        csv.push_str(&format!("df_resid,{}\n", self.df_resid));
        csv.push_str(&format!(
            "residual_std_error,{:.6}\n",
            self.residual_std_error
        ));

        csv
    }
}

impl CsvExport for DiscreteResult {
    fn to_csv_string(&self) -> String {
        let mut csv = String::new();

        // Header
        csv.push_str(
            "variable,coefficient,std_error,z_stat,p_value,significance,marginal_effect\n",
        );

        // Coefficients
        for i in 0..self.variables.len() {
            csv.push_str(&format!(
                "{},{:.6},{:.6},{:.6},{:.6},{},{:.6}\n",
                escape_csv(&self.variables[i]),
                self.coefficients[i],
                self.std_errors[i],
                self.z_stats[i],
                self.p_values[i],
                self.significance[i].code(),
                self.marginal_effects[i]
            ));
        }

        // Model statistics
        csv.push_str("\n# Model Statistics\n");
        csv.push_str("statistic,value\n");
        csv.push_str(&format!("model_type,{}\n", self.model_type));
        csv.push_str(&format!("dep_var,{}\n", escape_csv(&self.dep_var)));
        csv.push_str(&format!("n_obs,{}\n", self.n_obs));
        csv.push_str(&format!("n_positive,{}\n", self.n_positive));
        csv.push_str(&format!("log_likelihood,{:.6}\n", self.log_likelihood));
        csv.push_str(&format!(
            "log_likelihood_null,{:.6}\n",
            self.log_likelihood_null
        ));
        csv.push_str(&format!("pseudo_r_squared,{:.6}\n", self.pseudo_r_squared));
        csv.push_str(&format!("aic,{:.6}\n", self.aic));
        csv.push_str(&format!("bic,{:.6}\n", self.bic));
        csv.push_str(&format!("iterations,{}\n", self.iterations));
        csv.push_str(&format!("converged,{}\n", self.converged));

        csv
    }
}

impl CsvExport for PanelResult {
    fn to_csv_string(&self) -> String {
        let mut csv = String::new();

        // Header
        csv.push_str("variable,coefficient,std_error,t_stat,p_value,significance\n");

        // Coefficients
        for i in 0..self.variables.len() {
            csv.push_str(&format!(
                "{},{:.6},{:.6},{:.6},{:.6},{}\n",
                escape_csv(&self.variables[i]),
                self.coefficients[i],
                self.std_errors[i],
                self.t_stats[i],
                self.p_values[i],
                self.significance[i].code()
            ));
        }

        // Model statistics
        csv.push_str("\n# Model Statistics\n");
        csv.push_str("statistic,value\n");
        csv.push_str(&format!("method,{}\n", self.method));
        csv.push_str(&format!("dep_var,{}\n", escape_csv(&self.dep_var)));
        csv.push_str(&format!("entity_var,{}\n", escape_csv(&self.entity_var)));
        csv.push_str(&format!("n_obs,{}\n", self.n_obs));
        csv.push_str(&format!("n_groups,{}\n", self.n_groups));
        csv.push_str(&format!("r_squared,{:.6}\n", self.r_squared));
        csv.push_str(&format!("adj_r_squared,{:.6}\n", self.adj_r_squared));
        csv.push_str(&format!("f_stat,{:.6}\n", self.f_stat));
        csv.push_str(&format!("f_p_value,{:.6}\n", self.f_p_value));
        csv.push_str(&format!("df,{}\n", self.df));

        if let Some(sigma_u) = self.sigma_u {
            csv.push_str(&format!("sigma_u,{:.6}\n", sigma_u));
        }
        if let Some(sigma_e) = self.sigma_e {
            csv.push_str(&format!("sigma_e,{:.6}\n", sigma_e));
        }
        if let Some(theta) = self.theta {
            csv.push_str(&format!("theta,{:.6}\n", theta));
        }

        csv
    }
}

impl CsvExport for HausmanResult {
    fn to_csv_string(&self) -> String {
        let mut csv = String::new();

        // Test results
        csv.push_str("# Hausman Test Results\n");
        csv.push_str("statistic,value\n");
        csv.push_str(&format!("chi2_statistic,{:.6}\n", self.chi2_statistic));
        csv.push_str(&format!("p_value,{:.6}\n", self.p_value));
        csv.push_str(&format!("df,{}\n", self.df));
        csv.push_str(&format!(
            "recommendation,{}\n",
            escape_csv(&self.recommendation)
        ));

        // Coefficient comparison
        csv.push_str("\n# Coefficient Comparison\n");
        csv.push_str("variable,fe_coef,fe_se,re_coef,re_se,difference\n");

        let n_fe = self.fe_result.variables.len();
        let n_re = self.re_result.variables.len();

        // Skip intercept in RE (first variable) for comparison
        let re_offset = if n_re > n_fe { 1 } else { 0 };

        for i in 0..n_fe {
            let fe_coef = self.fe_result.coefficients[i];
            let fe_se = self.fe_result.std_errors[i];

            let (re_coef, re_se) = if i + re_offset < n_re {
                (
                    self.re_result.coefficients[i + re_offset],
                    self.re_result.std_errors[i + re_offset],
                )
            } else {
                (f64::NAN, f64::NAN)
            };

            csv.push_str(&format!(
                "{},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                escape_csv(&self.fe_result.variables[i]),
                fe_coef,
                fe_se,
                re_coef,
                re_se,
                fe_coef - re_coef
            ));
        }

        csv
    }
}

impl CsvExport for PCAResult {
    fn to_csv_string(&self) -> String {
        let mut csv = String::new();

        // Summary
        csv.push_str("# PCA Summary\n");
        csv.push_str("component,explained_variance,variance_ratio,cumulative_ratio\n");

        let mut cumulative = 0.0;
        for i in 0..self.n_components {
            cumulative += self.explained_variance_ratio[i];
            csv.push_str(&format!(
                "PC{},{:.6},{:.6},{:.6}\n",
                i + 1,
                self.explained_variance[i],
                self.explained_variance_ratio[i],
                cumulative
            ));
        }

        // Loadings (components)
        csv.push_str("\n# Loadings (eigenvectors)\n");
        csv.push_str("feature");
        for i in 0..self.n_components {
            csv.push_str(&format!(",PC{}", i + 1));
        }
        csv.push('\n');

        let n_features = self.components.ncols();
        for j in 0..n_features {
            csv.push_str(&format!("feature_{}", j + 1));
            for i in 0..self.n_components {
                csv.push_str(&format!(",{:.6}", self.components[[i, j]]));
            }
            csv.push('\n');
        }

        // Statistics
        csv.push_str("\n# Statistics\n");
        csv.push_str("statistic,value\n");
        csv.push_str(&format!("n_components,{}\n", self.n_components));
        csv.push_str(&format!("total_variance,{:.6}\n", self.total_variance));

        csv
    }
}

impl CsvExport for KMeansResult {
    fn to_csv_string(&self) -> String {
        let mut csv = String::new();

        // Summary
        csv.push_str("# K-Means Summary\n");
        csv.push_str("statistic,value\n");
        csv.push_str(&format!("n_clusters,{}\n", self.centroids.nrows()));
        csv.push_str(&format!("n_iterations,{}\n", self.n_iterations));
        csv.push_str(&format!("inertia,{:.6}\n", self.inertia));

        // Cluster sizes
        csv.push_str("\n# Cluster Sizes\n");
        csv.push_str("cluster,size\n");
        for (i, &size) in self.cluster_sizes.iter().enumerate() {
            csv.push_str(&format!("{},{}\n", i, size));
        }

        // Centroids
        csv.push_str("\n# Centroids\n");
        let n_features = self.centroids.ncols();
        csv.push_str("cluster");
        for j in 0..n_features {
            csv.push_str(&format!(",feature_{}", j + 1));
        }
        csv.push('\n');

        for i in 0..self.centroids.nrows() {
            csv.push_str(&format!("{}", i));
            for j in 0..n_features {
                csv.push_str(&format!(",{:.6}", self.centroids[[i, j]]));
            }
            csv.push('\n');
        }

        // Labels (cluster assignments)
        csv.push_str("\n# Labels (cluster assignments)\n");
        csv.push_str("observation,cluster\n");
        for (i, &label) in self.labels.iter().enumerate() {
            csv.push_str(&format!("{},{}\n", i, label));
        }

        csv
    }
}

impl CsvExport for DBSCANResult {
    fn to_csv_string(&self) -> String {
        let mut csv = String::new();

        // Summary
        csv.push_str("# DBSCAN Summary\n");
        csv.push_str("statistic,value\n");
        csv.push_str(&format!("n_clusters,{}\n", self.n_clusters));
        csv.push_str(&format!("n_noise,{}\n", self.n_noise));
        csv.push_str(&format!(
            "n_core_points,{}\n",
            self.core_sample_indices.len()
        ));

        // Compute cluster sizes from labels
        let mut cluster_sizes: std::collections::HashMap<i32, usize> =
            std::collections::HashMap::new();
        for &label in &self.labels {
            if label >= 0 {
                *cluster_sizes.entry(label).or_insert(0) += 1;
            }
        }

        // Cluster sizes
        csv.push_str("\n# Cluster Sizes\n");
        csv.push_str("cluster,size\n");
        let mut cluster_ids: Vec<_> = cluster_sizes.keys().cloned().collect();
        cluster_ids.sort();
        for id in cluster_ids {
            csv.push_str(&format!("{},{}\n", id, cluster_sizes[&id]));
        }

        // Labels (cluster assignments, -1 = noise)
        csv.push_str("\n# Labels (cluster assignments, -1 = noise)\n");
        csv.push_str("observation,cluster\n");
        for (i, &label) in self.labels.iter().enumerate() {
            csv.push_str(&format!("{},{}\n", i, label));
        }

        // Core sample indices
        csv.push_str("\n# Core Sample Indices\n");
        csv.push_str("index\n");
        for &idx in &self.core_sample_indices {
            csv.push_str(&format!("{}\n", idx));
        }

        csv
    }
}

/// Escape a string for CSV (handle commas and quotes).
fn escape_csv(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_csv() {
        assert_eq!(escape_csv("simple"), "simple");
        assert_eq!(escape_csv("with,comma"), "\"with,comma\"");
        assert_eq!(escape_csv("with\"quote"), "\"with\"\"quote\"");
        assert_eq!(escape_csv("with\nnewline"), "\"with\nnewline\"");
    }
}
