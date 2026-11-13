//! Correlation analysis for datasets.

use crate::data::Dataset;
use crate::resource;
use crate::stats::cortest::pearson_correlation;
use polars::prelude::*;
use serde::{Deserialize, Serialize};

/// Check if a DataType is numeric.
fn is_numeric_dtype(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
    )
}

/// Result of a correlation matrix computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationMatrix {
    pub columns: Vec<String>,
    /// Pearson correlations, each over the rows where both of its columns are present (R's
    /// `cor(use = "pairwise.complete.obs")`). NaN for a pair with fewer than two complete rows,
    /// an infinite value or a constant column, as R.
    pub matrix: Vec<Vec<f64>>,
    /// Rows where both columns of each pair are present (`pair_n[i][i]`: the present values of
    /// column i).
    #[serde(default)]
    pub pair_n: Vec<Vec<usize>>,
    /// Rows of the dataset.
    #[serde(default)]
    pub n_rows: usize,
}

impl CorrelationMatrix {
    /// Format the correlation matrix as a readable string. When a missing value dropped rows from
    /// some pair, a last line gives how many rows the entries use.
    pub fn to_string_table(&self) -> String {
        let mut result = String::new();

        // Header row
        result.push_str(&format!("{:>12}", ""));
        for col in &self.columns {
            result.push_str(&format!("{:>12}", p2a_text::ellipsize(col, 10)));
        }
        result.push('\n');

        // Data rows
        for (i, row) in self.matrix.iter().enumerate() {
            result.push_str(&format!(
                "{:>12}",
                p2a_text::ellipsize(&self.columns[i], 10)
            ));
            for val in row {
                result.push_str(&format!("{:>12.4}", val));
            }
            result.push('\n');
        }

        if let Some(note) = self.pairwise_note() {
            result.push_str(&note);
            result.push('\n');
        }

        result
    }

    /// "Pairwise complete observations: 35 to 38 of 40 rows" when some pair dropped rows.
    fn pairwise_note(&self) -> Option<String> {
        let min = self.pair_n.iter().flatten().min().copied()?;
        let max = self.pair_n.iter().flatten().max().copied()?;
        if min >= self.n_rows {
            return None;
        }
        let range = if min == max {
            min.to_string()
        } else {
            format!("{min} to {max}")
        };
        Some(format!(
            "Pairwise complete observations: {range} of {} rows",
            self.n_rows
        ))
    }
}

/// Compute the Pearson correlation matrix of the numeric columns of a dataset.
///
/// Each entry uses the rows where both of its columns are present (null and NaN cells are
/// missing), as R's `cor(use = "pairwise.complete.obs")`; `pair_n` reports how many. A pair with
/// fewer than two complete rows, an infinite value among them or a constant column is NaN, as in
/// R; the diagonal is 1 for a column with at least two present values, all finite and not all
/// equal, and NaN otherwise.
pub fn correlation_matrix(dataset: &Dataset) -> PolarsResult<CorrelationMatrix> {
    let df = dataset.df();
    let n_rows = df.height();

    // Select only numeric columns
    let numeric: Vec<&Column> = df
        .get_columns()
        .iter()
        .filter(|col| is_numeric_dtype(col.dtype()))
        .collect();
    let numeric_cols: Vec<String> = numeric.iter().map(|c| c.name().to_string()).collect();
    let k = numeric.len();

    // Each column as f64 with missing cells (null or NaN) as NaN
    let mut values: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut has_missing: Vec<bool> = Vec::with_capacity(k);
    for col in &numeric {
        let cast = col.as_materialized_series().cast(&DataType::Float64)?;
        let ca = cast.f64()?;
        let mut v: Vec<f64> =
            resource::try_with_capacity(n_rows, "correlation column").map_err(to_polars)?;
        v.extend(ca.iter().map(|x| x.unwrap_or(f64::NAN)));
        has_missing.push(v.iter().any(|x| x.is_nan()));
        values.push(v);
    }

    let mut matrix = vec![vec![f64::NAN; k]; k];
    let mut pair_n = vec![vec![0usize; k]; k];
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    if has_missing.iter().any(|&m| m) {
        xs = resource::try_with_capacity(n_rows, "correlation pair").map_err(to_polars)?;
        ys = resource::try_with_capacity(n_rows, "correlation pair").map_err(to_polars)?;
    }
    for i in 0..k {
        resource::checkpoint().map_err(to_polars)?;
        for j in i..k {
            let (r, count) = if !has_missing[i] && !has_missing[j] {
                (pair_correlation(&values[i], &values[j]), n_rows)
            } else {
                xs.clear();
                ys.clear();
                for (&a, &b) in values[i].iter().zip(&values[j]) {
                    if !a.is_nan() && !b.is_nan() {
                        xs.push(a);
                        ys.push(b);
                    }
                }
                (pair_correlation(&xs, &ys), xs.len())
            };
            let r = if i == j && !r.is_nan() { 1.0 } else { r };
            matrix[i][j] = r;
            matrix[j][i] = r;
            pair_n[i][j] = count;
            pair_n[j][i] = count;
        }
    }

    Ok(CorrelationMatrix {
        columns: numeric_cols,
        matrix,
        pair_n,
        n_rows,
    })
}

/// Pearson r of two complete, equally long samples; NaN below two rows, for an infinite value or
/// a constant sample (R's `cor`).
fn pair_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() < 2 {
        return f64::NAN;
    }
    pearson_correlation(x, y)
}

fn to_polars(e: crate::errors::EconError) -> PolarsError {
    PolarsError::ComputeError(e.to_string().into())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REG-17 (correlation.rs): names over 10 bytes were cut at byte 7; '合' spans bytes 6..9 of "収入合計額"
    /// and 'е' bytes 6..8 of "Население", so the compute_correlation tool panicked.
    #[test]
    fn test_to_string_table_non_ascii_names() {
        let cm = CorrelationMatrix {
            columns: vec![
                "収入合計額".into(),
                "Население".into(),
                "Größe_kg_netto".into(),
            ],
            matrix: vec![
                vec![1.0, 0.42137, -0.16752],
                vec![0.42137, 1.0, 0.30561],
                vec![-0.16752, 0.30561, 1.0],
            ],
            pair_n: vec![],
            n_rows: 0,
        };
        assert_eq!(
            cm.to_string_table(),
            concat!(
                "                   収入合計額   Население  Größe_k...\n",
                "       収入合計額      1.0000      0.4214     -0.1675\n",
                "   Население      0.4214      1.0000      0.3056\n",
                "  Größe_k...     -0.1675      0.3056      1.0000\n"
            )
        );
        let ascii = CorrelationMatrix {
            columns: vec!["income_total_usd".into()],
            matrix: vec![vec![1.0]],
            pair_n: vec![],
            n_rows: 0,
        };
        assert_eq!(
            ascii.to_string_table(),
            "              income_...\n  income_...      1.0000\n"
        );
    }

    /// R's cor(use = "pairwise.complete.obs") and the pair counts (case stat_cor_matrix_t4).
    /// Fails before: means over all present values of a column, sums over complete pairs.
    #[test]
    fn test_validate_correlation_matrix_pairwise_t4() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_cor_matrix_t4");
        let ds = Dataset::new(ds.df().drop("id").unwrap());
        let cm = correlation_matrix(&ds).unwrap();
        assert_eq!(cm.columns, ["a", "b", "c", "i"]);
        let flat: Vec<f64> = cm.matrix.concat();
        let counts: Vec<f64> = cm.pair_n.concat().iter().map(|&c| c as f64).collect();
        let small = crate::data::DataLoader::load_csv(p2a_validation::repo_path(
            "validation/datasets/generated/cor_matrix_small_t4.csv",
        ))
        .unwrap();
        let small = Dataset::new(small.drop("id").unwrap());
        let sm = correlation_matrix(&small).unwrap();
        Check::new(&case)
            .vector("pairwise", &flat, Tol::coef(35))
            .vector("pair_n", &counts, Tol::exact())
            .scalar("n_rows", cm.n_rows as f64, Tol::exact())
            .scalar("small_ab", sm.matrix[0][1], Tol::coef(4))
            .finish();
        assert_eq!(sm.pair_n, vec![vec![5, 4], vec![4, 4]]);
    }

    /// An infinite value makes every pair it enters NaN, diagonal included (R's cor); a row
    /// another column misses does not enter that pair. A NaN cell is missing.
    #[test]
    fn test_correlation_matrix_infinite_and_nan_cells() {
        let df = df!(
            "x" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.1],
            "inf" => [2.1, f64::INFINITY, 2.9, 4.2, 4.8, 6.3],
            "y" => [1.3, 1.8, 3.4, 3.9, 5.2, 5.7],
            "gap" => [Some(0.5), None, Some(1.7), Some(1.1), Some(2.6), Some(2.2)],
            "nan" => [0.3, 0.9, f64::NAN, 1.6, 1.2, 2.4],
        )
        .unwrap();
        let cm = correlation_matrix(&Dataset::new(df)).unwrap();
        let at = |a: &str, b: &str| {
            let i = cm.columns.iter().position(|c| c == a).unwrap();
            let j = cm.columns.iter().position(|c| c == b).unwrap();
            (cm.matrix[i][j], cm.pair_n[i][j])
        };
        for other in ["x", "y", "nan", "inf"] {
            assert!(at("inf", other).0.is_nan(), "inf-{other}");
        }
        // the infinite cell sits in the row `gap` misses
        let (r, n) = at("inf", "gap");
        assert!(r.is_finite() && n == 5, "{r} {n}");
        // the NaN cell is missing: 5 rows, and the same r as without that row
        let (r, n) = at("x", "nan");
        assert_eq!(n, 5);
        let x = [1.0, 2.0, 4.0, 5.0, 6.1];
        let v = [0.3, 0.9, 1.6, 1.2, 2.4];
        assert!((r - pearson_correlation(&x, &v)).abs() < 1e-15);
        assert_eq!(at("x", "x"), (1.0, 6));
        assert_eq!(at("nan", "nan"), (1.0, 5));
    }

    #[test]
    fn test_correlation_matrix_pair_with_one_row_is_nan() {
        let df = df!(
            "a" => [Some(1.0), None, Some(3.0)],
            "b" => [None, Some(2.0), Some(1.5)],
        )
        .unwrap();
        let cm = correlation_matrix(&Dataset::new(df)).unwrap();
        assert!(cm.matrix[0][1].is_nan());
        assert_eq!(cm.pair_n[0][1], 1);
        assert_eq!(cm.matrix[0][0], 1.0);
    }

    /// The pair-count line appears only when some pair dropped rows.
    #[test]
    fn test_to_string_table_pairwise_note() {
        let full = df!("a" => [1.0, 2.0, 3.5, 4.0], "b" => [2.0, 1.0, 3.0, 4.5]).unwrap();
        let t = correlation_matrix(&Dataset::new(full))
            .unwrap()
            .to_string_table();
        assert!(!t.contains("Pairwise"), "{t}");
        let gaps = df!(
            "a" => [Some(1.0), Some(2.0), None, Some(4.0), Some(5.0)],
            "b" => [Some(2.0), Some(1.0), Some(3.0), Some(4.5), Some(4.0)],
        )
        .unwrap();
        let t = correlation_matrix(&Dataset::new(gaps))
            .unwrap()
            .to_string_table();
        assert!(
            t.ends_with("Pairwise complete observations: 4 to 5 of 5 rows\n"),
            "{t}"
        );
    }
}
