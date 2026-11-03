//! Column extraction shared by the VAR-family estimators (crate-internal), and the test data of the sibling modules' tests.

use ndarray::Array2;
use polars::prelude::*;

use crate::data::columns::{ColumnRequest, NaPolicy};
use crate::errors::{EconError, EconResult};

/// Convert selected columns from a Polars DataFrame to an ndarray Array2.
///
/// One request with `NaPolicy::Fail` (vars::VAR refuses NA too): integer and Boolean
/// columns are accepted, text is `NonNumericColumn`, and a missing value (null or NaN) is
/// `NullValues` with the column's count. The matrix is a copy of the k extracted columns.
pub(crate) fn df_to_array2(
    df: &DataFrame,
    columns: &[&str],
) -> EconResult<(Array2<f64>, Vec<String>)> {
    if columns.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "No columns specified for time series analysis".to_string(),
        });
    }
    let cols = ColumnRequest::new(df)
        .numerics(columns)
        .na(NaPolicy::Fail)
        .extract()?;
    let data = cols.matrix(columns, false)?.data;
    let var_names = columns.iter().map(|c| c.to_string()).collect();
    Ok((data, var_names))
}

/// Convert ndarray Array2 to Vec<Vec<f64>>.
pub(crate) fn array2_to_vec(arr: &Array2<f64>) -> Vec<Vec<f64>> {
    arr.rows().into_iter().map(|row| row.to_vec()).collect()
}

/// Test data shared by the test modules of the sibling files.
#[cfg(test)]
pub(crate) mod test_fixtures {
    use super::*;
    use crate::data::Dataset;

    pub(crate) fn create_granger_test_dataset() -> Dataset {
        // Create time series where x Granger-causes y
        // y_t = 0.5 * y_{t-1} + 0.3 * x_{t-1} + noise
        // x_t is random walk
        let x: Vec<f64> = vec![
            1.0, 1.2, 1.5, 1.3, 1.6, 1.8, 2.1, 1.9, 2.2, 2.4, 2.6, 2.3, 2.8, 3.0, 2.7, 3.2, 3.4,
            3.1, 3.6, 3.8, 4.0, 3.7, 4.2, 4.5, 4.3, 4.7, 4.9, 4.6, 5.0, 5.2,
        ];
        // y is influenced by lagged x
        let y: Vec<f64> = vec![
            0.5, 0.8, 1.0, 1.2, 1.3, 1.6, 1.9, 2.0, 2.3, 2.5, 2.7, 2.9, 3.0, 3.3, 3.4, 3.7, 3.9,
            4.0, 4.3, 4.5, 4.7, 4.8, 5.1, 5.3, 5.4, 5.7, 5.9, 6.0, 6.3, 6.5,
        ];

        let df = df! {
            "y" => y,
            "x" => x
        }
        .unwrap();
        Dataset::new(df)
    }

    /// Linear Congruential Generator for deterministic test data
    pub(crate) fn lcg_rand_ts(seed: &mut u64) -> f64 {
        let a: u64 = 1103515245;
        let c: u64 = 12345;
        let m: u64 = 2_u64.pow(31);
        *seed = (a.wrapping_mul(*seed).wrapping_add(c)) % m;
        (*seed as f64) / (m as f64)
    }

    /// Box-Muller transform for normal distribution
    pub(crate) fn box_muller_ts(seed: &mut u64) -> f64 {
        let u1 = lcg_rand_ts(seed).max(1e-10);
        let u2 = lcg_rand_ts(seed);
        ((-2.0_f64 * u1.ln()).sqrt()) * (2.0 * std::f64::consts::PI * u2).cos()
    }

    pub(crate) fn create_var_validation_dataset() -> Dataset {
        // Create bivariate VAR(1) process:
        // y1_t = 0.5*y1_{t-1} + 0.3*y2_{t-1} + e1_t
        // y2_t = 0.2*y1_{t-1} + 0.4*y2_{t-1} + e2_t
        let n = 100;
        let mut seed: u64 = 42;

        let mut y1 = vec![0.0; n];
        let mut y2 = vec![0.0; n];

        // Initialize with noise
        y1[0] = box_muller_ts(&mut seed);
        y2[0] = box_muller_ts(&mut seed);

        for t in 1..n {
            let e1 = box_muller_ts(&mut seed);
            let e2 = box_muller_ts(&mut seed);
            y1[t] = 0.5 * y1[t - 1] + 0.3 * y2[t - 1] + e1;
            y2[t] = 0.2 * y1[t - 1] + 0.4 * y2[t - 1] + e2;
        }

        let df = df! {
            "y1" => y1,
            "y2" => y2
        }
        .unwrap();
        Dataset::new(df)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array2_to_vec() {
        let arr = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let vec = array2_to_vec(&arr);
        assert_eq!(vec.len(), 2);
        assert_eq!(vec[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(vec[1], vec![4.0, 5.0, 6.0]);
    }
}
