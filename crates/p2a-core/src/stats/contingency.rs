//! Contingency tables built from factor columns.
//!
//! The rows and columns are the levels of factor columns in R `factor()` order
//! (`crate::data::columns`): numbers by value (`2 < 10`), text in byte order (R's order
//! under `LC_COLLATE=C`), booleans `FALSE < TRUE`, dates chronologically. Rows with a
//! missing value (null, or NaN in a float column) in a used column are dropped and
//! reported in the `missing` summary, as R's `table()` excludes `NA`. The levels are those
//! observed in the kept rows, so no row or column of a table is empty (R's `table(x, y)`
//! keeps a level of `x` that occurs only where `y` is missing as an all-zero row).
//!
//! Each builder counts in one O(n) pass over the factor codes into a buffer whose size is
//! checked and allocated through [`crate::resource`], calling
//! [`resource::checkpoint`](crate::resource::checkpoint) every 65,536 rows. No polars lazy
//! query runs, so the tables can be built inside an async runtime.
//!
//! ```
//! use p2a_core::stats::contingency_table;
//! use polars::prelude::*;
//! let df = df!("code" => [Some(10i64), Some(2), None, Some(2)], "y" => ["a", "b", "a", "a"]).unwrap();
//! let t = contingency_table(&df, "code", "y").unwrap();
//! assert_eq!(t.row_levels, ["2", "10"]);              // numeric order, as R factor()
//! assert_eq!(t.counts, vec![1.0, 1.0, 1.0, 0.0]);       // the NA row is dropped
//! assert_eq!(t.missing.n_dropped, 1);
//! ```

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::hash::Hash;

use polars::prelude::DataFrame;
use serde::{Deserialize, Serialize};

use crate::data::columns::{
    ColumnRequest, Columns, MissingSummary, NaPolicy, column, is_numeric_dtype,
};
use crate::errors::{EconError, EconResult};
use crate::resource;

/// Passes over rows or keys call [`resource::checkpoint`] this often.
const CHECK_EVERY: usize = 65_536;

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

/// r x c table of two factor columns; rows with a missing value (null, or NaN in a float
/// column) in either column are dropped and reported (R `table()`); levels in R `factor()`
/// order over the kept rows. `row_col == col_col` gives the diagonal table, as R's
/// `table(x, x)`.
///
/// # Errors
///
/// - [`EconError::ColumnNotFound`] for a column that does not exist.
/// - [`EconError::InvalidSpecification`] for a column whose type cannot be categorical
///   (datetime, struct, list, ...), naming it.
/// - [`EconError::InsufficientData`] when no row is left.
/// - [`EconError::ResourceLimit`] for a table too large to allocate.
/// - [`EconError::Cancelled`] when the thread's budget is cancelled.
pub fn contingency_table(
    df: &DataFrame,
    row_col: &str,
    col_col: &str,
) -> EconResult<ContingencyTable> {
    let cols = extract_factors(df, &[row_col, col_col], "contingency table")?;
    count_table(&cols, row_col, col_col)
}

/// [`contingency_table`] for Fisher's exact test: the level counts are checked on the
/// extracted columns, before any table is allocated, so that two columns with many levels
/// are refused for their shape and not for the size of a table that could only be refused.
///
/// # Errors
///
/// As [`contingency_table`], and [`EconError::InvalidSpecification`] naming both columns'
/// levels when either does not have exactly 2.
pub(crate) fn contingency_table_2x2(
    df: &DataFrame,
    row_col: &str,
    col_col: &str,
) -> EconResult<ContingencyTable> {
    let cols = extract_factors(df, &[row_col, col_col], "contingency table")?;
    require_2x2(
        &cols,
        row_col,
        col_col,
        "Fisher's exact test needs a 2 x 2 table",
        "; p2a implements the 2 x 2 case of R's fisher.test (use the chi-squared test for \
         larger tables)",
    )?;
    count_table(&cols, row_col, col_col)
}

/// The factor columns `names`, rows with a missing value dropped, at least one row left.
fn extract_factors(df: &DataFrame, names: &[&str], context: &str) -> EconResult<Columns> {
    ColumnRequest::new(df)
        .factors(names)
        .na(NaPolicy::Omit)
        .min_rows(1, context)
        .extract()
}

/// [`EconError::InvalidSpecification`] unless the extracted factors `row_col` and `col_col`
/// both have 2 levels: "`needs`, but after removing rows with missing values ... `remedy`",
/// naming both columns' levels.
fn require_2x2(
    cols: &Columns,
    row_col: &str,
    col_col: &str,
    needs: &str,
    remedy: &str,
) -> EconResult<()> {
    let (rows, columns) = (cols.factor(row_col), cols.factor(col_col));
    if rows.n_levels() == 2 && columns.n_levels() == 2 {
        return Ok(());
    }
    Err(EconError::InvalidSpecification {
        message: format!(
            "{needs}, but after removing rows with missing values '{row_col}' has {} level(s) \
             [{}] and '{col_col}' has {} level(s) [{}]{remedy}",
            rows.n_levels(),
            level_list(&rows.levels),
            columns.n_levels(),
            level_list(&columns.levels)
        ),
    })
}

/// The r x c table of the extracted factors `row_col` and `col_col`, counted in one pass.
fn count_table(cols: &Columns, row_col: &str, col_col: &str) -> EconResult<ContingencyTable> {
    let rows = cols.factor(row_col);
    let columns = cols.factor(col_col);
    let n_cols = columns.n_levels();
    let cells = resource::checked_len(&[rows.n_levels(), n_cols], "contingency table")?;
    let mut counts = resource::try_vec(cells, 0.0, "contingency table")?;
    for (i, (&r, &c)) in rows.codes.iter().zip(&columns.codes).enumerate() {
        if i % CHECK_EVERY == 0 {
            resource::checkpoint()?;
        }
        counts[r as usize * n_cols + c as usize] += 1.0;
    }
    Ok(ContingencyTable {
        row_var: row_col.to_string(),
        col_var: col_col.to_string(),
        row_levels: rows.levels.clone(),
        col_levels: columns.levels.clone(),
        counts,
        missing: cols.missing().clone(),
    })
}

/// Level labels and counts of one factor column (missing cells dropped). `extra_levels`
/// that match no observed level are appended in the order given, with count 0 (numeric
/// columns match by value: "2.0" is the level "2"; see [`same_category`]).
///
/// Every observed label and every extra is matched through a hash of its canonical form,
/// so `e` extras cost O(k + e) for `k` levels.
///
/// # Errors
///
/// The errors of [`contingency_table`] for the one column, and
/// [`EconError::InvalidSpecification`] when two extras that match no observed level name
/// the same value ("3" and "3.0" of a numeric column, or one text twice).
pub fn level_counts(
    df: &DataFrame,
    col: &str,
    extra_levels: &[String],
) -> EconResult<(Vec<String>, Vec<f64>, MissingSummary)> {
    let cols = ColumnRequest::new(df)
        .factor(col)
        .na(NaPolicy::Omit)
        .min_rows(1, "category counts")
        .extract()?;
    resource::checkpoint()?;
    let factor = cols.factor(col);
    let len = factor
        .n_levels()
        .checked_add(extra_levels.len())
        .ok_or_else(|| EconError::ResourceLimit("too many categories".to_string()))?;
    let mut labels: Vec<String> = resource::try_with_capacity(len, "category labels")?;
    labels.extend(factor.levels.iter().cloned());
    let mut counts: Vec<f64> = resource::try_with_capacity(len, "category counts")?;
    counts.extend(factor.counts().into_iter().map(|c| c as f64));
    if !extra_levels.is_empty() {
        let numeric = is_numeric_dtype(column(df, col)?.dtype(), false);
        let observed = CategoryIndex::new(&factor.levels, numeric)?;
        let mut appended: HashMap<CategoryKey<'_>, usize> =
            try_map(extra_levels.len(), "category labels")?;
        for (i, key) in extra_levels.iter().enumerate() {
            if i % CHECK_EVERY == 0 {
                resource::checkpoint()?;
            }
            if observed.find(key).is_some() {
                continue;
            }
            match appended.entry(CategoryKey::of(key, numeric)) {
                Entry::Occupied(first) => {
                    return Err(EconError::InvalidSpecification {
                        message: format!(
                            "categories '{}' and '{key}' name the same value of column '{col}'",
                            extra_levels[*first.get()]
                        ),
                    });
                }
                Entry::Vacant(slot) => {
                    slot.insert(i);
                    labels.push(key.clone());
                    counts.push(0.0);
                }
            }
        }
    }
    Ok((labels, counts, cols.missing().clone()))
}

/// One table per stratum level (R `factor()` order); row and column levels are common to
/// all strata (levels of the kept rows), so a stratum without some level has zeros there.
/// Rows missing any of the three columns are dropped; the overall summary is the third
/// element, and each stratum's table has `MissingSummary::none(n_k)`.
///
/// # Errors
///
/// As [`contingency_table`], for the three columns, the K x r x c counts and the copy of the
/// level labels each stratum's table holds.
pub fn stratified_tables(
    df: &DataFrame,
    row_col: &str,
    col_col: &str,
    stratum_col: &str,
) -> EconResult<(Vec<String>, Vec<ContingencyTable>, MissingSummary)> {
    let cols = extract_factors(df, &[row_col, col_col, stratum_col], "stratified tables")?;
    count_strata(&cols, row_col, col_col, stratum_col)
}

/// [`stratified_tables`] for the Mantel-Haenszel test: the row and column level counts are
/// checked on the extracted columns, before any table is allocated (see
/// [`contingency_table_2x2`]).
///
/// # Errors
///
/// As [`stratified_tables`], and [`EconError::InvalidSpecification`] naming both columns'
/// levels when either does not have exactly 2.
pub(crate) fn stratified_tables_2x2(
    df: &DataFrame,
    row_col: &str,
    col_col: &str,
    stratum_col: &str,
) -> EconResult<(Vec<String>, Vec<ContingencyTable>, MissingSummary)> {
    let cols = extract_factors(df, &[row_col, col_col, stratum_col], "stratified tables")?;
    require_2x2(
        &cols,
        row_col,
        col_col,
        "Mantel-Haenszel test needs 2 x 2 x K tables",
        "",
    )?;
    count_strata(&cols, row_col, col_col, stratum_col)
}

/// One r x c table per level of the extracted factor `stratum_col`, each counted in place.
fn count_strata(
    cols: &Columns,
    row_col: &str,
    col_col: &str,
    stratum_col: &str,
) -> EconResult<(Vec<String>, Vec<ContingencyTable>, MissingSummary)> {
    const WHAT: &str = "stratified tables";
    let rows = cols.factor(row_col);
    let columns = cols.factor(col_col);
    let strata = cols.factor(stratum_col);
    let (r, c, k) = (rows.n_levels(), columns.n_levels(), strata.n_levels());
    let per = resource::checked_len(&[r, c], WHAT)?;
    // the K x r x c counts, checked as one buffer and held per stratum
    resource::check_alloc::<f64>(resource::checked_len(&[k, per], WHAT)?, WHAT)?;
    // and every stratum's table holds its own names and level labels
    let label_bytes = rows
        .levels
        .iter()
        .chain(&columns.levels)
        .map(String::len)
        .chain([row_col.len(), col_col.len()])
        .try_fold(0usize, |acc, len| {
            acc.checked_add(len)?
                .checked_add(std::mem::size_of::<String>())
        })
        .ok_or_else(|| resource::overflow("stratified tables' level labels"))?;
    resource::check_alloc::<u8>(
        resource::checked_len(&[k, label_bytes], "stratified tables' level labels")?,
        "stratified tables' level labels",
    )?;
    let mut counts: Vec<Vec<f64>> = resource::try_with_capacity(k, WHAT)?;
    for s in 0..k {
        if s % CHECK_EVERY == 0 {
            resource::checkpoint()?;
        }
        counts.push(resource::try_vec(per, 0.0, WHAT)?);
    }
    for (i, ((&ri, &ci), &si)) in rows
        .codes
        .iter()
        .zip(&columns.codes)
        .zip(&strata.codes)
        .enumerate()
    {
        if i % CHECK_EVERY == 0 {
            resource::checkpoint()?;
        }
        counts[si as usize][ri as usize * c + ci as usize] += 1.0;
    }
    let n_by_stratum = strata.counts();
    let mut tables: Vec<ContingencyTable> = resource::try_with_capacity(k, WHAT)?;
    for (s, (stratum_counts, n_k)) in counts.into_iter().zip(n_by_stratum).enumerate() {
        if s % CHECK_EVERY == 0 {
            resource::checkpoint()?;
        }
        tables.push(ContingencyTable {
            row_var: row_col.to_string(),
            col_var: col_col.to_string(),
            row_levels: rows.levels.clone(),
            col_levels: columns.levels.clone(),
            counts: stratum_counts,
            missing: MissingSummary::none(n_k),
        });
    }
    Ok((strata.levels.clone(), tables, cols.missing().clone()))
}

/// Same category for a level label and a user-given key: the exact label, or, when
/// `numeric` (the column holds numbers), an equal numeric value (`"2" == "2.0"`,
/// `"-0" == "0"`; NaN never matches by value). [`CategoryIndex`] implements this rule by
/// hashing.
pub(crate) fn same_category(label: &str, key: &str, numeric: bool) -> bool {
    label == key || (numeric && numeric_key(label).is_some_and(|v| Some(v) == numeric_key(key)))
}

/// The labels joined with ", " for a message, at most 10 and then ", ...".
pub(crate) fn level_list(labels: &[String]) -> String {
    let mut out = labels
        .iter()
        .take(10)
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(", ");
    if labels.len() > 10 {
        out.push_str(", ...");
    }
    out
}

/// A key's numeric value as bits, `-0` made `0`; `None` for text or NaN.
fn numeric_key(s: &str) -> Option<u64> {
    let v = s.parse::<f64>().ok().filter(|v| !v.is_nan())?;
    Some(if v == 0.0 { 0.0f64 } else { v }.to_bits())
}

/// The canonical form [`same_category`] compares: the value of a key that parses as a
/// number on a numeric column, else the key itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CategoryKey<'a> {
    Label(&'a str),
    Value(u64),
}

impl<'a> CategoryKey<'a> {
    fn of(key: &'a str, numeric: bool) -> Self {
        let value = if numeric { numeric_key(key) } else { None };
        match value {
            Some(bits) => CategoryKey::Value(bits),
            None => CategoryKey::Label(key),
        }
    }
}

/// A hash index of level labels for matching user keys by [`same_category`]: an exact
/// label first, then (numeric columns) an equal value. Two labels with one value (integers
/// above 2^53) keep the first for the value match; their exact labels still match.
pub(crate) struct CategoryIndex<'a> {
    exact: HashMap<&'a str, usize>,
    by_value: HashMap<u64, usize>,
    numeric: bool,
}

impl<'a> CategoryIndex<'a> {
    /// Index `labels` (a column's levels, then any appended categories).
    pub(crate) fn new(labels: &'a [String], numeric: bool) -> EconResult<Self> {
        let mut exact = try_map(labels.len(), "category index")?;
        let mut by_value = try_map(if numeric { labels.len() } else { 0 }, "category index")?;
        for (i, label) in labels.iter().enumerate() {
            if i % CHECK_EVERY == 0 {
                resource::checkpoint()?;
            }
            exact.entry(label.as_str()).or_insert(i);
            if numeric && let Some(bits) = numeric_key(label) {
                by_value.entry(bits).or_insert(i);
            }
        }
        Ok(CategoryIndex {
            exact,
            by_value,
            numeric,
        })
    }

    /// The index of the label `key` names, if any.
    pub(crate) fn find(&self, key: &str) -> Option<usize> {
        if let Some(&i) = self.exact.get(key) {
            return Some(i);
        }
        if self.numeric {
            return numeric_key(key).and_then(|bits| self.by_value.get(&bits).copied());
        }
        None
    }
}

/// An empty map with room for `cap` entries, its size checked against the allocation
/// ceiling (with room for the table's load factor) and reserved fallibly.
fn try_map<K: Eq + Hash, V>(cap: usize, what: &str) -> EconResult<HashMap<K, V>> {
    resource::check_alloc::<(K, V)>(resource::checked_len(&[cap, 2], what)?, what)?;
    let mut map = HashMap::new();
    map.try_reserve(cap)
        .map_err(|_| resource::reserve_failed(what, cap, std::mem::size_of::<(K, V)>()))?;
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::EconError;
    use p2a_validation::{Check, Tol};
    use polars::prelude::*;

    #[test]
    fn test_validate_contingency_table_na_codes() {
        // R: table(g, h) drops the 3 NA h cells -> [[3, 1], [1, 4]]; table(code, out) has rows
        // "2", "10" (numeric order, STAT-50) -> [[4, 2], [1, 4]] after dropping the NA code
        // (case stat_contingency_na_codes).
        let (case, ds) = crate::test_support::load_case("stat_contingency_na_codes");
        let gh = contingency_table(ds.df(), "g", "h").unwrap();
        let co = contingency_table(ds.df(), "code", "out").unwrap();
        Check::new(&case)
            .vector("gh_observed", &gh.counts, Tol::exact())
            .scalar("gh_n", gh.total(), Tol::exact())
            .vector("code_out_table", &co.counts, Tol::exact())
            .finish();
        assert_eq!(gh.row_levels, ["a", "b"]);
        assert_eq!(gh.col_levels, ["u", "v"]);
        assert_eq!(co.row_levels, ["2", "10"]);
        assert_eq!(co.col_levels, ["0", "1"]);
        assert_eq!(
            (gh.missing.n_total, gh.missing.n_used, gh.missing.n_dropped),
            (12, 9, 3)
        );
        assert_eq!(gh.missing.by_column, vec![("h".to_string(), 3)]);
        assert_eq!(co.missing.by_column, vec![("code".to_string(), 1)]);
        assert_eq!((gh.row_var.as_str(), gh.col_var.as_str()), ("g", "h"));
    }

    #[test]
    fn test_validate_na_contingency_int64_equals_float64() {
        // design 2.4 (a): an Int64 code column and the same values as Float64 give one table
        // and the same labels.
        let (_case, ds) = crate::test_support::load_case("stat_contingency_na_codes");
        assert_eq!(ds.df().column("code").unwrap().dtype(), &DataType::Int64);
        let mut df = ds.df().clone();
        let f = df.column("code").unwrap().cast(&DataType::Float64).unwrap();
        df.with_column(f).unwrap();
        assert_eq!(
            contingency_table(&df, "code", "out").unwrap(),
            contingency_table(ds.df(), "code", "out").unwrap()
        );
    }

    #[test]
    fn test_na_contingency_nan_float_is_missing() {
        // design 2.2: the one inline fixture, a NaN cell (the harness cannot write one). NaN is
        // missing, as R's is.na.
        let df =
            df!("x" => [1.0, 2.0, f64::NAN, 2.0, 10.0], "y" => ["p", "q", "p", "p", "q"]).unwrap();
        let t = contingency_table(&df, "x", "y").unwrap();
        assert_eq!(t.row_levels, ["1", "2", "10"]);
        assert_eq!(t.counts, vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        assert_eq!(
            (t.missing.n_dropped, t.missing.by_column.clone()),
            (1, vec![("x".to_string(), 1)])
        );
    }

    #[test]
    fn test_validate_na_contingency_rejects_datetime_and_absent_columns() {
        // design 2.4 (c)/(d): a dtype that cannot be a factor is named; an absent column is
        // ColumnNotFound.
        let (_case, ds) = crate::test_support::load_case("stat_contingency_na_codes");
        let mut df = ds.df().clone();
        let dt = df
            .column("out")
            .unwrap()
            .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
            .unwrap()
            .with_name("stamp".into());
        df.with_column(dt).unwrap();
        let e = contingency_table(&df, "stamp", "g")
            .unwrap_err()
            .to_string();
        assert!(e.contains("stamp"), "{e}");
        assert!(matches!(
            contingency_table(&df, "g", "nope"),
            Err(EconError::ColumnNotFound { .. })
        ));
        assert!(matches!(
            level_counts(&df, "nope", &[]),
            Err(EconError::ColumnNotFound { .. })
        ));
        assert!(matches!(
            stratified_tables(&df, "g", "h", "nope"),
            Err(EconError::ColumnNotFound { .. })
        ));
    }

    #[test]
    fn test_validate_level_counts_extra_levels() {
        // R: table(factor(cat, levels = c("a","b","c","d"))) -> 5, 30, 15, 0 (case
        // stat_chisq_gof, key named4_observed); the two NA cells are dropped.
        let (case, ds) = crate::test_support::load_case("stat_chisq_gof");
        let (labels, counts, missing) =
            level_counts(ds.df(), "cat", &["d".to_string(), "a".to_string()]).unwrap();
        assert_eq!(labels, ["a", "b", "c", "d"]); // "a" is observed, so not appended again
        Check::new(&case)
            .vector("named4_observed", &counts, Tol::exact())
            .scalar("n_used", missing.n_used as f64, Tol::exact())
            .finish();
        assert_eq!(missing.by_column, vec![("cat".to_string(), 2)]);
        // numeric column: "10.0" is the observed level "10", "33" is new
        let (_c2, ds2) = crate::test_support::load_case("stat_contingency_na_codes");
        let (l2, c2, _) =
            level_counts(ds2.df(), "code", &["10.0".to_string(), "33".to_string()]).unwrap();
        assert_eq!(
            (l2, c2),
            (
                vec!["2".to_string(), "10".into(), "33".into()],
                vec![6.0, 5.0, 0.0]
            )
        );
        // two extras naming one value are an error
        assert!(level_counts(ds2.df(), "code", &["33".to_string(), "33.0".to_string()]).is_err());
    }

    #[test]
    fn test_validate_stratified_tables_mantelhaen_na() {
        // R: table(treat, outcome, stratum) drops the 3 NA strata: 237 rows, strata s1 s2 s3
        // (case stat_mantelhaen_na).
        let (case, ds) = crate::test_support::load_case("stat_mantelhaen_na");
        let (strata, tables, missing) =
            stratified_tables(ds.df(), "treat", "outcome", "stratum").unwrap();
        assert_eq!(strata, ["s1", "s2", "s3"]);
        assert!(
            tables
                .iter()
                .all(|t| t.row_levels == ["ctl", "trt"] && t.col_levels == ["no", "yes"])
        );
        let total: f64 = tables.iter().map(ContingencyTable::total).sum();
        Check::new(&case)
            .scalar("n_used", total, Tol::exact())
            .scalar("n_used", missing.n_used as f64, Tol::exact())
            .finish();
        assert_eq!(missing.by_column, vec![("stratum".to_string(), 3)]);
        assert!(tables.iter().all(|t| t.missing.is_empty()));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_validate_contingency_table_inside_tokio_runtime() {
        // STAT-10: the builder never calls polars' lazy engine, whose streaming executor calls
        // block_on.
        let (_case, ds) = crate::test_support::load_case("stat_fisher_rows_2000");
        let t = contingency_table(ds.df(), "treat", "outcome").unwrap();
        assert_eq!(t.counts, vec![520.0, 480.0, 480.0, 520.0]);
    }

    #[test]
    fn test_same_category_matches_labels_and_numeric_values() {
        assert!(same_category("a", "a", false));
        assert!(!same_category("a", "A", false));
        assert!(!same_category("2", "2.0", false)); // text columns compare labels only
        assert!(same_category("2", "2.0", true));
        assert!(same_category("0", "-0", true));
        assert!(same_category("10", "1e1", true));
        assert!(!same_category("2", "3", true));
        assert!(same_category("NaN", "NaN", true)); // equal labels match
        assert!(!same_category("NaN", "nan", true)); // a NaN value never matches by value
        assert!(!same_category("1", "NaN", true));
        assert!(!same_category("1", "one", true));
    }

    #[test]
    fn test_level_counts_matches_many_extras_by_hash() {
        // every observed level and every extra is matched by its canonical key, so
        // a long list of extras is linear; the extras keep the order given, after the levels.
        let df = df!("k" => [3i64, 1, 2, 3]).unwrap();
        let extras: Vec<String> = (0..20_000).rev().map(|i| format!("{}.0", i + 1)).collect();
        let (labels, counts, _) = level_counts(&df, "k", &extras).unwrap();
        assert_eq!(labels.len(), 20_000);
        assert_eq!(&labels[..4], ["1", "2", "3", "20000.0"]);
        assert_eq!(&counts[..4], [1.0, 1.0, 2.0, 0.0]);
        assert_eq!(counts.iter().sum::<f64>(), 4.0);
        let e = level_counts(&df, "k", &["7".to_string(), "x".into(), "7.00".into()])
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("'7'") && e.contains("'7.00'") && e.contains("'k'"),
            "{e}"
        );
        // a text key on a numeric column matches nothing and is appended
        let (labels, counts, _) = level_counts(&df, "k", &["x".to_string()]).unwrap();
        assert_eq!((labels.len(), counts[3]), (4, 0.0));
    }

    #[test]
    fn test_contingency_table_of_one_column_with_itself_is_diagonal() {
        // R: table(x, x)
        let df = df!("x" => ["b", "a", "b", "c"]).unwrap();
        let t = contingency_table(&df, "x", "x").unwrap();
        assert_eq!(t.counts, vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_contingency_table_honours_cancellation_and_ceiling() {
        use crate::resource::{self, Budget, CancelToken};
        let df = df!("x" => ["b", "a", "b"], "y" => ["u", "v", "v"]).unwrap();
        let token = CancelToken::new();
        token.cancel();
        for r in [
            resource::with_budget(Budget::new(None, token.clone()), || {
                contingency_table(&df, "x", "y").map(|_| ())
            }),
            resource::with_budget(Budget::new(None, token.clone()), || {
                level_counts(&df, "x", &["c".to_string()]).map(|_| ())
            }),
            resource::with_budget(Budget::new(None, token), || {
                stratified_tables(&df, "x", "y", "x").map(|_| ())
            }),
        ] {
            assert!(matches!(r, Err(EconError::Cancelled(_))), "{r:?}");
        }
        let a: Vec<i64> = (0..40).collect();
        let wide = df!("a" => a.clone(), "b" => a).unwrap();
        let r = resource::with_alloc_ceiling(Some(1024), || contingency_table(&wide, "a", "b"));
        assert!(
            matches!(&r, Err(EconError::ResourceLimit(m)) if m.contains("contingency table")),
            "{r:?}"
        );
    }

    #[test]
    fn test_stratified_tables_check_the_label_copies_against_the_ceiling() {
        // 1000 strata of 2 x 2 with 100-byte labels: the counts (32 kB) and the tables
        // themselves (about 170 kB) fit a 300 kB ceiling, but every stratum's table holds its
        // own copy of the 4 labels (about 500 kB), which used to be cloned without a check.
        use crate::resource;
        let n = 1000;
        let label = |c: &str| c.repeat(100);
        let (a, b, u, v) = (label("a"), label("b"), label("u"), label("v"));
        let df = df!(
            "t" => (0..n).map(|i| if i % 2 == 0 { a.as_str() } else { b.as_str() }).collect::<Vec<_>>(),
            "y" => (0..n).map(|i| if i % 3 == 0 { u.as_str() } else { v.as_str() }).collect::<Vec<_>>(),
            "s" => (0..n).map(|i| format!("s{i:04}")).collect::<Vec<_>>()
        )
        .unwrap();
        let r =
            resource::with_alloc_ceiling(Some(300_000), || stratified_tables(&df, "t", "y", "s"));
        assert!(
            matches!(&r, Err(EconError::ResourceLimit(m)) if m.contains("level labels")),
            "{:?}",
            r.map(|(s, _, _)| s.len())
        );
        let (strata, tables, _) = stratified_tables(&df, "t", "y", "s").unwrap();
        assert_eq!((strata.len(), tables.len()), (n, n));
        assert_eq!(tables[0].counts, vec![1.0, 0.0, 0.0, 0.0]); // row 0: t = a, y = u
        assert_eq!(tables[1].counts, vec![0.0, 0.0, 0.0, 1.0]); // row 1: t = b, y = v
        assert_eq!(tables[999].row_levels, [a, b]);
    }

    #[test]
    fn test_two_by_two_builders_name_both_columns_levels() {
        // The 2 x 2 builders refuse another shape with both columns' levels (the ceiling tests
        // of run_fisher_test and run_mantelhaen_test show that this comes before counting),
        // and otherwise build what the general builders build.
        let df = df!("x" => ["b", "a", "c", "a"], "y" => ["u", "v", "v", "u"]).unwrap();
        let r = contingency_table_2x2(&df, "x", "y");
        assert!(
            matches!(&r, Err(EconError::InvalidSpecification { message })
                if message.starts_with("Fisher's exact test needs a 2 x 2 table")
                    && message.contains("'x' has 3 level(s) [a, b, c]")
                    && message.contains("'y' has 2 level(s) [u, v]")),
            "{r:?}"
        );
        let r = stratified_tables_2x2(&df, "y", "x", "y");
        assert!(
            matches!(&r, Err(EconError::InvalidSpecification { message })
                if message.contains("2 x 2 x K") && message.contains("'x' has 3 level(s)")),
            "{:?}",
            r.map(|(s, _, _)| s.len())
        );
        let df2 = df!("x" => ["b", "a", "b", "a", "a"], "y" => ["u", "v", "v", "u", "u"]).unwrap();
        let t = contingency_table_2x2(&df2, "x", "y").unwrap();
        assert_eq!(t.as_2x2(), Some([[2.0, 1.0], [1.0, 1.0]]));
        assert_eq!(t, contingency_table(&df2, "x", "y").unwrap());
        assert_eq!(
            stratified_tables_2x2(&df2, "x", "y", "y").unwrap(),
            stratified_tables(&df2, "x", "y", "y").unwrap()
        );
    }

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
