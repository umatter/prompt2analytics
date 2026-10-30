//! Utility functions for panel data estimation.
//!
//! Contains helper functions for entity ID extraction, demeaning,
//! and other operations common to panel data estimators.

use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, Factor, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::resource;

/// Entity codes of every row of the dataset (one per row) and the number of entities.
///
/// Codes follow R `factor()` order of the ids (numeric ids numerically, text ids in C
/// order). A missing id is an error here (`EconError::NullValues`), because this
/// function returns one value per dataset row; estimators that drop incomplete rows
/// use [`entity_codes`] on their own [`ColumnRequest`] instead.
pub fn extract_entity_ids(dataset: &Dataset, entity_var: &str) -> EconResult<(Vec<usize>, usize)> {
    let cols = ColumnRequest::new(dataset.df())
        .factor(entity_var)
        .na(NaPolicy::Fail)
        .extract()?;
    Ok(entity_codes(cols.factor(entity_var)))
}

/// Compute entity-level means for demeaning.
pub fn compute_entity_means(
    data: &Array1<f64>,
    entity_ids: &[usize],
    n_groups: usize,
) -> Array1<f64> {
    let n = data.len();
    let mut group_sums = vec![0.0; n_groups];
    let mut group_counts = vec![0usize; n_groups];

    for (i, &val) in data.iter().enumerate() {
        let g = entity_ids[i];
        group_sums[g] += val;
        group_counts[g] += 1;
    }

    let group_means: Vec<f64> = group_sums
        .iter()
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
pub fn demean_matrix_by_entity(
    x: &Array2<f64>,
    entity_ids: &[usize],
    n_groups: usize,
) -> Array2<f64> {
    let (n, k) = x.dim();
    let mut x_demeaned = Array2::zeros((n, k));

    for j in 0..k {
        let col = x.column(j).to_owned();
        let col_demeaned = demean_by_entity(&col, entity_ids, n_groups);
        x_demeaned.column_mut(j).assign(&col_demeaned);
    }

    x_demeaned
}

/// Period index of every row of the dataset (0 = earliest) and the sorted distinct
/// times.
///
/// Times are integers of any width, whole-number floats or dates (days since
/// 1970-01-01); a fractional time is `InvalidSpecification` ("must hold whole
/// numbers"). A missing time is an error here (`EconError::NullValues`), because this
/// function returns one value per dataset row; estimators that drop incomplete rows
/// use [`time_codes`] on their own [`ColumnRequest`] instead.
pub fn extract_time_ids(dataset: &Dataset, time_var: &str) -> EconResult<(Vec<usize>, Vec<i64>)> {
    let cols = ColumnRequest::new(dataset.df())
        .time_index(time_var)
        .na(NaPolicy::Fail)
        .extract()?;
    Ok(time_codes(cols.time_index(time_var)))
}

/// Entity codes of the kept rows (0-based, in `factor.levels` order: R factor() order,
/// numeric ids numerically, text ids in C order) and the number of entities present.
pub(crate) fn entity_codes(factor: &Factor) -> (Vec<usize>, usize) {
    let codes = factor.codes.iter().map(|&c| c as usize).collect();
    (codes, factor.n_levels())
}

/// Dense period index (0 = earliest) of each time value, and the sorted distinct values.
/// O(n log n): a sorted, de-duplicated copy and a binary search per value.
pub(crate) fn time_codes(times: &[i64]) -> (Vec<usize>, Vec<i64>) {
    let mut distinct = times.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    let codes = times
        .iter()
        .map(|t| distinct.binary_search(t).unwrap_or_default())
        .collect();
    (codes, distinct)
}

/// Build the instrument matrix for difference GMM.
///
/// For period t, valid instruments are y_{i,t-2}, y_{i,t-3}, ..., y_{i,1}
/// The instrument matrix is block-diagonal across time periods.
///
/// The caller guarantees `n_periods >= 1` and `min_lag <= max_lag` after
/// `max_lag` is limited to `n_periods - 1`. The matrix is sized with checked
/// arithmetic and allocated through [`resource`].
pub fn build_gmm_instrument_matrix(
    y_lagged: &[Vec<f64>], // y values by entity, each vec is time series
    n_groups: usize,
    n_periods: usize,
    min_lag: usize,
    max_lag: Option<usize>,
    collapse: bool,
) -> EconResult<(Array2<f64>, usize)> {
    // For difference GMM, we use lags 2, 3, ... as instruments for differenced equation
    // The instrument matrix grows with T
    let what = "GMM instrument matrix";

    let max_lag = max_lag.unwrap_or(n_periods - 1);
    let max_lag = max_lag.min(n_periods - 1);
    debug_assert!(min_lag <= max_lag, "callers validate min_lag <= max_lag");

    // Calculate number of instrument columns
    let n_inst_cols = if collapse {
        // Collapsed: one column per lag depth
        max_lag.saturating_sub(min_lag) + 1
    } else {
        // Full: sum of available lags for each period
        // For t=min_lag+1, we have 1 instrument; for t=min_lag+2, we have 2, etc.
        let mut total: usize = 0;
        for t in (min_lag + 1)..n_periods {
            let n_lags = (t - min_lag).min(max_lag - min_lag + 1);
            total = total.checked_add(n_lags).ok_or_else(|| {
                EconError::ResourceLimit(format!(
                    "{what} is too large: the requested size overflows"
                ))
            })?;
        }
        total
    };

    // Number of rows = number of groups × (number of valid time periods)
    let n_rows = resource::checked_len(
        &[
            n_groups,
            n_periods.saturating_sub(min_lag.saturating_add(1)),
        ],
        what,
    )?;

    if n_inst_cols == 0 || n_rows == 0 {
        return Ok((Array2::zeros((1, 1)), 0));
    }

    let mut z = resource::try_zeros2(n_rows, n_inst_cols, what)?;

    let mut row = 0;
    for i in 0..n_groups {
        resource::checkpoint()?;
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

    Ok((z, n_inst_cols))
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

    // Rows of each entity, in row order, grouped in one pass
    let mut rows_by_entity: Vec<Vec<usize>> = vec![Vec::new(); n_groups];
    for (row, &(e, _)) in valid_obs.iter().enumerate() {
        if e < n_groups {
            rows_by_entity[e].push(row);
        }
    }

    for obs_i in &rows_by_entity {
        if obs_i.len() <= order {
            continue;
        }

        for idx in order..obs_i.len() {
            let row_idx = obs_i[idx];
            let row_idx_lag = obs_i[idx - order];

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

    let p_value = 2.0
        * (1.0
            - Normal::new(0.0, 1.0)
                .map(|n| n.cdf(z_stat.abs()))
                .unwrap_or(0.5));

    (z_stat, p_value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DataLoader;
    use crate::data::columns::{ColumnRequest, NaPolicy, numeric_column};
    use polars::prelude::{Column, DataType};

    /// `panel_grunfeld_na` and the complete Grunfeld panel it was built from.
    fn fixtures() -> (Dataset, Dataset) {
        let (case, na) = crate::test_support::load_case("panel_grunfeld_na");
        let path = p2a_validation::repo_path(&case.datasets[0]);
        assert!(
            path.ends_with("validation/datasets/grunfeld.csv"),
            "{path:?}"
        );
        (na, Dataset::new(DataLoader::load_csv(&path).unwrap()))
    }

    /// `ds` with column `name` replaced by `column`.
    fn with(ds: &Dataset, column: Column) -> Dataset {
        let mut df = ds.df().clone();
        df.with_column(column).unwrap();
        Dataset::new(df)
    }

    /// Integer times of any width and whole floats give the same periods; a fractional
    /// time is an error; a missing time is `NullValues` (the legacy shape has one value
    /// per row) instead of period 0.
    #[test]
    fn test_validate_na_extract_time_ids_integer_widths_and_fractions() {
        let (na, g) = fixtures();
        let expected = extract_time_ids(&g, "year").unwrap();
        assert_eq!(expected.1, (1935..=1954).collect::<Vec<i64>>());
        let year = g.df().column("year").unwrap().clone();
        for dtype in [
            DataType::Int32,
            DataType::Int16,
            DataType::UInt16,
            DataType::Float64,
        ] {
            let cast = year.cast(&dtype).unwrap();
            let got = extract_time_ids(&with(&g, cast), "year")
                .unwrap_or_else(|e| panic!("{dtype:?}: {e}"));
            assert_eq!(got, expected, "{dtype:?}");
        }
        let (mut years, _) = numeric_column(g.df(), "year", NaPolicy::Fail).unwrap();
        years[0] = 1935.25;
        match extract_time_ids(&with(&g, Column::new("year".into(), years)), "year") {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(message.contains("must hold whole numbers"), "{message}")
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
        match extract_time_ids(&na, "year") {
            Err(EconError::NullValues { column, count }) => {
                assert_eq!((column.as_str(), count), ("year", 2))
            }
            other => panic!("expected NullValues for year, got {other:?}"),
        }
    }

    /// Entity codes follow R factor order (numeric ids numerically); a missing id is
    /// `NullValues` instead of id 0 or the pseudo-entity "".
    #[test]
    fn test_validate_na_extract_entity_ids_factor_order_and_nulls() {
        let (na, g) = fixtures();
        let firm: Vec<usize> = numeric_column(g.df(), "firm", NaPolicy::Fail)
            .unwrap()
            .0
            .iter()
            .map(|&f| f as usize - 1)
            .collect();
        let (ids, n_groups) = extract_entity_ids(&g, "firm").unwrap();
        assert_eq!((ids.as_slice(), n_groups), (firm.as_slice(), 10));
        let cast = g
            .df()
            .column("firm")
            .unwrap()
            .cast(&DataType::UInt32)
            .unwrap();
        assert_eq!(
            extract_entity_ids(&with(&g, cast), "firm").unwrap(),
            (firm.clone(), 10)
        );
        for column in ["firm_id", "firm_name"] {
            match extract_entity_ids(&na, column) {
                Err(EconError::NullValues { column: c, count }) => {
                    assert_eq!((c.as_str(), count), (column, 2))
                }
                other => panic!("{column}: expected NullValues, got {other:?}"),
            }
        }
        assert!(
            matches!(
                extract_entity_ids(&na, "nope"),
                Err(EconError::ColumnNotFound { .. })
            ),
            "unknown column"
        );
    }

    /// The crate-internal codes: dense period indexes and factor codes as `usize`.
    #[test]
    fn test_validate_na_entity_and_time_codes() {
        assert_eq!(
            time_codes(&[2001, 1999, 2001, 2000]),
            (vec![2, 0, 2, 1], vec![1999, 2000, 2001])
        );
        let (na, _) = fixtures();
        let cols = ColumnRequest::new(na.df())
            .factor("firm_id")
            .extract()
            .unwrap();
        let (codes, n_groups) = entity_codes(cols.factor("firm_id"));
        assert_eq!((codes.len(), n_groups), (198, 10));
        let ids = numeric_column(na.df(), "firm_id", NaPolicy::Omit)
            .unwrap()
            .0;
        assert_eq!(ids.len(), 198);
        for (code, id) in codes.iter().zip(&ids) {
            assert_eq!(*code, *id as usize, "firm_id {id} has code {code}");
        }
    }
}
