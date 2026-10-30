//! Contingency tables built from factor columns.
//!
//! The rows and columns are the levels of two factor columns in R `factor()` order
//! (`crate::data::columns`); rows with a missing label in either column are dropped and
//! reported, as R's `table()` excludes `NA`.
//!
//! Implemented by T2-WP05; until then the functions return `EconError::NotImplemented`.
//! The [`ContingencyTable`] accessors are implemented.

use polars::prelude::DataFrame;
use serde::{Deserialize, Serialize};

use crate::data::columns::MissingSummary;
use crate::errors::{EconError, EconResult};

/// An r x c table of counts with its level labels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContingencyTable {
    /// Name of the row variable.
    pub row_var: String,
    /// Name of the column variable.
    pub col_var: String,
    /// Row level labels, in factor order.
    pub row_levels: Vec<String>,
    /// Column level labels, in factor order.
    pub col_levels: Vec<String>,
    /// row-major r x c counts
    pub counts: Vec<f64>,
    /// Rows dropped because of a missing label.
    pub missing: MissingSummary,
}

impl ContingencyTable {
    /// Number of rows (`row_levels.len()`).
    pub fn n_rows(&self) -> usize {
        self.row_levels.len()
    }

    /// Number of columns (`col_levels.len()`).
    pub fn n_cols(&self) -> usize {
        self.col_levels.len()
    }

    /// The count in row `i`, column `j`.
    ///
    /// # Panics
    /// If `i >= n_rows()` or `j >= n_cols()`.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        assert!(
            i < self.n_rows() && j < self.n_cols(),
            "cell ({i}, {j}) is outside the {} x {} table",
            self.n_rows(),
            self.n_cols()
        );
        self.counts[i * self.n_cols() + j]
    }

    /// The counts as `[[a, b], [c, d]]` for a 2 x 2 table; `None` for any other shape.
    pub fn as_2x2(&self) -> Option<[[f64; 2]; 2]> {
        if self.n_rows() != 2 || self.n_cols() != 2 || self.counts.len() != 4 {
            return None;
        }
        Some([
            [self.counts[0], self.counts[1]],
            [self.counts[2], self.counts[3]],
        ])
    }

    /// Sum of all counts.
    pub fn total(&self) -> f64 {
        self.counts.iter().sum()
    }
}

/// r x c table from two factor columns, rows with a missing label in either dropped.
///
/// Implemented by T2-WP05; until then returns `EconError::NotImplemented`.
pub fn contingency_table(
    df: &DataFrame,
    row_col: &str,
    col_col: &str,
) -> EconResult<ContingencyTable> {
    Err(EconError::NotImplemented(
        "T2-WP05: contingency_table".to_string(),
    ))
}

/// One factor's level labels and counts (goodness of fit); `extra_levels` adds unobserved
/// levels with count 0.
///
/// Implemented by T2-WP05; until then returns `EconError::NotImplemented`.
pub fn level_counts(
    df: &DataFrame,
    col: &str,
    extra_levels: &[String],
) -> EconResult<(Vec<String>, Vec<f64>, MissingSummary)> {
    Err(EconError::NotImplemented(
        "T2-WP05: level_counts".to_string(),
    ))
}

/// One 2 x 2 table per stratum level (Mantel-Haenszel), rows with a missing value in any
/// of the three columns dropped.
///
/// Implemented by T2-WP05; until then returns `EconError::NotImplemented`.
pub fn stratified_tables(
    df: &DataFrame,
    row_col: &str,
    col_col: &str,
    stratum_col: &str,
) -> EconResult<(Vec<String>, Vec<ContingencyTable>, MissingSummary)> {
    Err(EconError::NotImplemented(
        "T2-WP05: stratified_tables".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(rows: &[&str], cols: &[&str], counts: &[f64]) -> ContingencyTable {
        ContingencyTable {
            row_var: "r".into(),
            col_var: "c".into(),
            row_levels: rows.iter().map(|s| s.to_string()).collect(),
            col_levels: cols.iter().map(|s| s.to_string()).collect(),
            counts: counts.to_vec(),
            missing: MissingSummary::none(counts.iter().sum::<f64>() as usize),
        }
    }

    #[test]
    fn test_contingency_table_accessors() {
        let t = table(
            &["a", "b"],
            &["x", "y", "z"],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        assert_eq!(t.n_rows(), 2);
        assert_eq!(t.n_cols(), 3);
        assert_eq!(t.get(1, 2), 6.0);
        assert_eq!(t.get(0, 1), 2.0);
        assert_eq!(t.total(), 21.0);
        assert_eq!(t.as_2x2(), None);
        let t2 = table(&["a", "b"], &["x", "y"], &[3.0, 1.0, 1.0, 3.0]);
        assert_eq!(t2.as_2x2(), Some([[3.0, 1.0], [1.0, 3.0]]));
    }

    #[test]
    #[should_panic(expected = "outside the 2 x 3 table")]
    fn test_contingency_table_get_out_of_range_panics() {
        let t = table(
            &["a", "b"],
            &["x", "y", "z"],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        t.get(2, 0);
    }
}
