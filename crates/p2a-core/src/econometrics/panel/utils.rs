//! Utility functions for panel data estimation.
//!
//! Contains helper functions for entity ID extraction, demeaning,
//! and other operations common to panel data estimators.

use ndarray::{Array1, Array2};
use std::collections::HashMap;
use statrs::distribution::{Normal, ContinuousCDF};

use crate::data::Dataset;
use crate::errors::{EconResult, EconError};

/// Extract entity IDs from a DataFrame column and return as Vec<usize>.
pub fn extract_entity_ids(dataset: &Dataset, entity_var: &str) -> EconResult<(Vec<usize>, usize)> {
    let df = dataset.df();
    let col = df.column(entity_var)
        .map_err(|_| EconError::ColumnNotFound {
            column: entity_var.to_string(),
            available: df.get_column_names().iter().map(|s| s.to_string()).collect(),
        })?;

    // Create a mapping from unique values to integer IDs
    let mut id_map: HashMap<String, usize> = HashMap::new();
    let mut next_id = 0usize;

    let ids: Vec<usize> = if let Ok(int_col) = col.i64() {
        int_col.into_iter()
            .map(|v| {
                let key = v.unwrap_or(0).to_string();
                *id_map.entry(key).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                })
            })
            .collect()
    } else if let Ok(str_col) = col.str() {
        str_col.into_iter()
            .map(|v| {
                let key = v.unwrap_or("").to_string();
                *id_map.entry(key).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                })
            })
            .collect()
    } else {
        // Try to cast to string
        let casted = col.cast(&polars::prelude::DataType::String)
            .map_err(|e| EconError::Internal(format!("Cannot convert entity column to IDs: {}", e)))?;
        let str_col = casted.str()
            .map_err(|e| EconError::Internal(format!("Cannot read entity column as string: {}", e)))?;
        str_col.into_iter()
            .map(|v| {
                let key = v.unwrap_or("").to_string();
                *id_map.entry(key).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                })
            })
            .collect()
    };

    let n_groups = id_map.len();
    Ok((ids, n_groups))
}

/// Compute entity-level means for demeaning.
pub fn compute_entity_means(data: &Array1<f64>, entity_ids: &[usize], n_groups: usize) -> Array1<f64> {
    let n = data.len();
    let mut group_sums = vec![0.0; n_groups];
    let mut group_counts = vec![0usize; n_groups];

    for (i, &val) in data.iter().enumerate() {
        let g = entity_ids[i];
        group_sums[g] += val;
        group_counts[g] += 1;
    }

    let group_means: Vec<f64> = group_sums.iter()
        .zip(group_counts.iter())
        .map(|(&sum, &count)| if count > 0 { sum / count as f64 } else { 0.0 })
        .collect();

    // Create array with entity means for each observation
    let mut means = Array1::zeros(n);
    for i in 0..n {
        means[i] = group_means[entity_ids[i]];
    }
    means
}

/// Demean a vector by entity (for Fixed Effects).
pub fn demean_by_entity(data: &Array1<f64>, entity_ids: &[usize], n_groups: usize) -> Array1<f64> {
    let means = compute_entity_means(data, entity_ids, n_groups);
    data - &means
}

/// Demean a matrix by entity (for Fixed Effects).
pub fn demean_matrix_by_entity(x: &Array2<f64>, entity_ids: &[usize], n_groups: usize) -> Array2<f64> {
    let (n, k) = x.dim();
    let mut x_demeaned = Array2::zeros((n, k));

    for j in 0..k {
        let col = x.column(j).to_owned();
        let col_demeaned = demean_by_entity(&col, entity_ids, n_groups);
        x_demeaned.column_mut(j).assign(&col_demeaned);
    }

    x_demeaned
}

/// Extract time period IDs from a DataFrame column.
pub fn extract_time_ids(dataset: &Dataset, time_var: &str) -> EconResult<(Vec<usize>, Vec<i64>)> {
    let df = dataset.df();
    let col = df.column(time_var)
        .map_err(|_| EconError::ColumnNotFound {
            column: time_var.to_string(),
            available: df.get_column_names().iter().map(|s| s.to_string()).collect(),
        })?;

    // Extract unique time values and map to indices
    let mut time_values: Vec<i64> = Vec::new();
    let mut time_map: HashMap<i64, usize> = HashMap::new();

    let times: Vec<i64> = if let Ok(int_col) = col.i64() {
        int_col.into_iter()
            .map(|v| v.unwrap_or(0))
            .collect()
    } else if let Ok(f_col) = col.f64() {
        f_col.into_iter()
            .map(|v| v.unwrap_or(0.0) as i64)
            .collect()
    } else {
        return Err(EconError::InvalidSpecification {
            message: "Time variable must be numeric".to_string()
        });
    };

    // Get unique sorted times
    let mut unique_times: Vec<i64> = times.iter().copied().collect();
    unique_times.sort_unstable();
    unique_times.dedup();

    for (idx, &t) in unique_times.iter().enumerate() {
        time_map.insert(t, idx);
        time_values.push(t);
    }

    let time_ids: Vec<usize> = times.iter()
        .map(|t| *time_map.get(t).unwrap())
        .collect();

    Ok((time_ids, time_values))
}

/// Build the instrument matrix for difference GMM.
///
/// For period t, valid instruments are y_{i,t-2}, y_{i,t-3}, ..., y_{i,1}
/// The instrument matrix is block-diagonal across time periods.
pub fn build_gmm_instrument_matrix(
    y_lagged: &[Vec<f64>],  // y values by entity, each vec is time series
    n_groups: usize,
    n_periods: usize,
    min_lag: usize,
    max_lag: Option<usize>,
    collapse: bool,
) -> (Array2<f64>, usize) {
    // For difference GMM, we use lags 2, 3, ... as instruments for differenced equation
    // The instrument matrix grows with T

    let max_lag = max_lag.unwrap_or(n_periods - 1);
    let max_lag = max_lag.min(n_periods - 1);

    // Calculate number of instrument columns
    let n_inst_cols = if collapse {
        // Collapsed: one column per lag depth
        max_lag.saturating_sub(min_lag) + 1
    } else {
        // Full: sum of available lags for each period
        // For t=min_lag+1, we have 1 instrument; for t=min_lag+2, we have 2, etc.
        let mut total = 0;
        for t in (min_lag + 1)..n_periods {
            let n_lags = (t - min_lag).min(max_lag - min_lag + 1);
            total += n_lags;
        }
        total
    };

    // Number of rows = number of groups × (number of valid time periods)
    let n_rows = n_groups * (n_periods.saturating_sub(min_lag + 1));

    if n_inst_cols == 0 || n_rows == 0 {
        return (Array2::zeros((1, 1)), 0);
    }

    let mut z = Array2::zeros((n_rows, n_inst_cols));

    let mut row = 0;
    for i in 0..n_groups {
        if y_lagged[i].len() < n_periods {
            continue;
        }

        for t in (min_lag + 1)..n_periods {
            if collapse {
                // Collapsed instruments: one column per lag
                for (col, lag) in (min_lag..=max_lag.min(t - 1)).enumerate() {
                    if lag < y_lagged[i].len() {
                        z[[row, col]] = y_lagged[i][t - 1 - lag + min_lag];
                    }
                }
            } else {
                // Full instruments: separate columns for each (t, lag) combination
                let mut col = 0;
                for s in (min_lag + 1)..t {
                    let n_lags = (s - min_lag).min(max_lag - min_lag + 1);
                    col += n_lags;
                }
                for lag in min_lag..=max_lag.min(t - 1) {
                    if lag < y_lagged[i].len() && col < n_inst_cols {
                        z[[row, col]] = y_lagged[i][t - 1 - lag + min_lag];
                        col += 1;
                    }
                }
            }
            row += 1;
        }
    }

    (z, n_inst_cols)
}

/// Compute Arellano-Bond test for serial correlation.
pub fn compute_ab_ar_test(
    resid: &Array1<f64>,
    valid_obs: &[(usize, usize)],
    n_groups: usize,
    order: usize,
) -> (f64, f64) {
    // AR(order) test: test correlation between e_it and e_{i,t-order}
    let mut numerator = 0.0;
    let mut var_e = 0.0;
    let mut var_e_lag = 0.0;

    for i in 0..n_groups {
        let obs_i: Vec<(usize, &(usize, usize))> = valid_obs.iter()
            .enumerate()
            .filter(|(_, (e, _))| *e == i)
            .collect();

        if obs_i.len() <= order {
            continue;
        }

        for idx in order..obs_i.len() {
            let (row_idx, _) = obs_i[idx];
            let (row_idx_lag, _) = obs_i[idx - order];

            let e_it = resid[row_idx];
            let e_it_lag = resid[row_idx_lag];

            numerator += e_it * e_it_lag;
            var_e += e_it * e_it;
            var_e_lag += e_it_lag * e_it_lag;
        }
    }

    let denominator = (var_e * var_e_lag).sqrt();
    let z_stat = if denominator > 0.0 {
        numerator / denominator * (valid_obs.len() as f64).sqrt()
    } else {
        0.0
    };

    let p_value = 2.0 * (1.0 - Normal::new(0.0, 1.0)
        .map(|n| n.cdf(z_stat.abs()))
        .unwrap_or(0.5));

    (z_stat, p_value)
}
